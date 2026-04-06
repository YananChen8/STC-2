import json, os, torch, functools, tqdm, random, sys, argparse
from dataclasses import replace
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS
from demo.model_path import resolve_model_path

logger = logging.get_logger(__name__)


class ReducedLogitsMCQTrainer(Trainer):
    """Reduce per-step logits to option ids before cross-process padding/gather."""

    def __init__(self, *args, strict_option_ids: list[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.strict_option_ids = strict_option_ids

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
        )

        if logits is None:
            return loss, logits, labels
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits.get("logits")

        if logits is None or logits.ndim < 3:
            return loss, logits, labels

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            last_pos = torch.full(
                (logits.shape[0],), logits.shape[1] - 1, device=logits.device, dtype=torch.long
            )
        else:
            last_pos = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)

        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        reduced_logits = logits[batch_idx, last_pos][:, self.strict_option_ids].argmax(dim=-1).to(torch.int64)
        return loss, reduced_logits, labels

def _read_may1fps_video_decord(ele: dict):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]
    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')
    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)
    video_fps = vr.get_avg_fps()
    clip_idxs, clip_pts = None, None
    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        video_start = video_pts[0] if not video_start else video_start
        video_end = video_pts[-1] if not video_end else video_end
        video_start = min(max(video_pts[0], video_start), video_pts[-1])
        video_end = min(max(video_pts[0], video_end), video_pts[-1])
        video_end = max(video_start + 1, video_end)
        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]
        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)
    total_frames_for_smart_nframes = total_frames
    video_fps_for_smart_nframes = video_fps
    if total_frames < 2:
        total_frames_for_smart_nframes = 2
    if video_fps < FPS:
        total_frames_for_smart_nframes = int(total_frames * FPS / video_fps)
        video_fps_for_smart_nframes = FPS
    nframes = smart_nframes(ele, total_frames=total_frames_for_smart_nframes, video_fps=video_fps_for_smart_nframes) 
    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]
    clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps
    return clip, sample_fps

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 

class OvoBenchMCQDataset(Dataset):
    def __init__(self, path, question_prefix, question_postfix, answer_prefix, sample: int = None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]
        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]
        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix
        self.data_dir = os.path.dirname(path)
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        conversation = [{"role": "user", "content": []}]
        video_inputs = None
        if datum['task'] in ['REC', 'SSR', 'CRR']: # 'REC', 'SSR', 'CRR' have already been chunked
            query = datum['question']
        else:
            query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        video, _ = _read_may1fps_video_decord({
            'video': os.path.join(self.data_dir, datum['video']), 
            'video_start': datum['video_start'], 
            'video_end': datum['video_end']
        })
        video = _spatial_resize_video(video)
        conversation[0]['content'].append({"type": "video", "video": video})
        video_inputs = [video]
        conversation[0]['content'].append({"type": "text", "text": query})
        if video_inputs is None:
            for _ in range(10):
                try:
                    _, video_inputs = process_vision_info(conversation)
                    break
                except:
                    print(f"{_}-th process_vision_info failed. retry...")
        return conversation, video_inputs[0]

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)
        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
        texts = [text + self.answer_prefix for text in texts]
        inputs = processor(
            text=texts,
            images=None,
            videos=list(video_inputs),
            padding=True,
            return_tensors="pt",
        )
        return inputs

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    options: list[str], 
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
):
    strict_option_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in options] 
    dataset = OvoBenchMCQDataset(benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix)
    trainer = ReducedLogitsMCQTrainer(
        model=model, 
        strict_option_ids=strict_option_ids,
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', use_liger_kernel=use_liger_kernel
        ), 
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
    )
    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions
    return letter_idxs_predictions, dataset.datums, trainer.args.process_index

def evaluate_ovobench_results(results: list):
    task_to_counts = {}
    for result in results:
        task = result['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        if result['response'][:len(result['answer'])] == result['answer']:
            task_to_counts[task]['correct'] += 1
    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in task_to_counts.items():
        print(f'{task}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
            rt_accs.append(counts['correct']/counts['total'])
        elif task in ['EPM', 'ASI', 'HLD']:
            bt_accs.append(counts['correct']/counts['total'])
        else:
            fr_accs.append(counts['correct']/counts['total'])
    if rt_accs:
        print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    if bt_accs:
        print(f'Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs)/len(bt_accs)}')
    if fr_accs:
        print(f'Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs)/len(fr_accs)}')

def enable_qwen2_vl_selective_recompute_cache(
    model,
    cache_interval: int | None = None,
    update_token_ratio: float | None = None,
    similarity_threshold: float | None = None,
):
    from qwen_vl_with_cacher import register_cache_for_qwen2_vl
    from controller import get_config

    config = get_config()
    cache_config = config.cache
    if cache_interval is not None:
        cache_config = replace(cache_config, cache_interval=cache_interval)
    if update_token_ratio is not None:
        cache_config = replace(cache_config, update_token_ratio=update_token_ratio)
    if similarity_threshold is not None:
        cache_config = replace(cache_config, similarity_threshold=similarity_threshold)
    config.cache = cache_config

    register_cache_for_qwen2_vl(model)
    logger.warning(
        "Enabled Qwen2-VL selective recomputation cache: "
        f"cache_interval={config.cache.cache_interval}, "
        f"update_token_ratio={config.cache.update_token_ratio}, "
        f"similarity_threshold={config.cache.similarity_threshold}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format OVO-Bench dataset JSONL file.")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Path to ovobench dir.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model id, snapshot dir, or HF cache repo dir.")
    parser.add_argument("--output_dir", type=str, default="evaluation/ovobench/results", help="Directory to save results.")
    parser.add_argument(
        "--enable_selective_recompute_cache",
        action="store_true",
        help="Enable selective recomputation cache in qwen_vl_with_cacher.py for OVO-Bench inference.",
    )
    parser.add_argument("--cache_interval", type=int, default=None, help="Cache chunk interval for selective recomputation.")
    parser.add_argument("--update_token_ratio", type=float, default=None, help="Per-frame token recompute ratio.")
    parser.add_argument("--similarity_threshold", type=float, default=None, help="Similarity threshold used by STC cacher.")
    args = parser.parse_args()
    benchmark_path = os.path.join(args.benchmark_dir, 'ovo-bench-formatted.jsonl')

    model_path = resolve_model_path(args.model_name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    if args.enable_selective_recompute_cache:
        enable_qwen2_vl_selective_recompute_cache(
            model,
            cache_interval=args.cache_interval,
            update_token_ratio=args.update_token_ratio,
            similarity_threshold=args.similarity_threshold,
        )
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']
    
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=benchmark_path, 
        options=options, use_liger_kernel='LiveCC' in model_path,
        answer_prefix = 'The answer is:\n', 
        abcd_previous_str = '\n',
    )
    if process_index == 0:
        results = []
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            results.append({
                'id': datum['id'],
                "task": datum['task'],
                "question": datum['question'],
                "answer": datum['answer'],
                "response": options[letter_idx_prediction],
            })
        os.makedirs(args.output_dir, exist_ok=True)
        save_json_path = os.path.join(args.output_dir, f'{os.path.basename(args.model_name_or_path.rstrip("/"))}.json')
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_ovobench_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/ovobench/distributed_evaluate_ovobench.py --benchmark_dir /path/to/ovobench --model_name_or_path chenjoya/LiveCC-7B-Instruct
