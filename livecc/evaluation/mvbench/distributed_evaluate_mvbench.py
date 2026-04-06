import argparse
import torch, json, functools, tqdm, random, sys, os
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, logging, Qwen2VLForConditionalGeneration, AutoProcessor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torchvision.io import read_image
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS
from data.lmm_dataset import bytes_to_pil, pil_to_tensor
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

def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 

class MVBenchMCQDataset(Dataset):
    def __init__(self, remote_loader, path, question_prefix, question_postfix, answer_prefix, sample: int = None):
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
        self.remote_loader = remote_loader
        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix
        conversation = [{"role": "user", "content": []}]
        video_inputs = None
        if 'video' in datum:
            if 'tvqa' in datum['video']:
                nframes = smart_nframes({'fps': FPS}, total_frames=len(datum['frames']), video_fps=FPS) # suggest this has been fpsed
                sampler = torch.linspace(0, len(datum['frames']) - 1, nframes).round().long()
                images = [read_image(os.path.join(datum['video'], datum['frames'][i])) for i in sampler]
                video = torch.stack(images)
                video = _spatial_resize_video(video)
                conversation[0]['content'].append({"type": "video", "video": video})
                video_inputs = [video]
            else:
                conversation[0]['content'].append(
                    {"type": "video", "video": datum['video'], 'remote_loader': self.remote_loader},
                )
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
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

def mcq_predict(
    model, 
    processor, 
    benchmark_path: str, 
    letters: list[str], 
    remote_loader: callable,
    question_prefix: str = '', 
    question_postfix: str = '\nPlease select the correct answer.', 
    answer_prefix: str = 'Answer:', 
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 2,
    dataloader_num_workers: int = 4,
):
    strict_letter_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in letters] 
    dataset = MVBenchMCQDataset(remote_loader, benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix)
    trainer = ReducedLogitsMCQTrainer(
        model=model, 
        strict_option_ids=strict_letter_ids,
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

def evaluate_mvbench_results(results: list):
    task_type_to_counts = {}
    for video_item in results:
        for question_item in video_item['questions']:
            task_type = question_item['task_type']
            if task_type not in task_type_to_counts:
                task_type_to_counts[task_type] = {'correct': 0, 'total': 0}
            task_type_to_counts[task_type]['total'] += 1
            if question_item['response'][0] == question_item['answer']:
                task_type_to_counts[task_type]['correct'] += 1
    accs = []
    for task_type, counts in task_type_to_counts.items():
        print(f'{task_type}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        accs.append(counts["correct"]/counts["total"])
    print(f'Average: {sum(accs)/len(accs)}')

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for MVBench.")
    parser.add_argument("--model_name_or_path", required=True, help="HF model id, snapshot dir, or HF cache repo dir.")
    parser.add_argument("--benchmark_path", default="mvbench_video_existed.jsonl", help="Path to benchmark jsonl.")
    parser.add_argument("--output_dir", default="evaluation/mvbench/results", help="Directory to save results.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = resolve_model_path(args.model_name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=args.benchmark_path,
        letters=['A', 'B', 'C', 'D', 'E'], use_liger_kernel='LiveCC' in model_path,
    )
    if process_index == 0:
        video_to_results = {}
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            video = datum['video']
            if video not in video_to_results:
                video_to_results[video] = {
                    'video': video,
                    'questions': [],
                }
            video_to_results[video]['questions'].append(
                {
                    "task_type": datum['task_type'],
                    "question": datum['question'],
                    "options": datum['options'],
                    "answer": datum['answer'],
                    "response": datum['options'][letter_idx_prediction],
                },
            )
        results = list(video_to_results.values())
        os.makedirs(args.output_dir, exist_ok=True)
        save_json_path = os.path.join(args.output_dir, f'{os.path.basename(args.model_name_or_path.rstrip("/"))}.json')
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_mvbench_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/mvbench/distributed_evaluate_mvbench.py --model_name_or_path chenjoya/LiveCC-7B-Instruct
