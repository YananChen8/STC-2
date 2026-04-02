import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from evaluation.distributed_mcq_predictor import mcq_predict
from evaluation.utils import save_function_print
from demo.model_path import resolve_model_path

def evaluate_livesports3kqa_results(results: list):
    q_type_to_counts = {}
    ocr_to_counts = {'correct': 0, 'total': 0}
    for video_item in results:
        for question_item in video_item['questions']:
            q_type = question_item['q_type']
            if q_type not in q_type_to_counts:
                q_type_to_counts[q_type] = {'correct': 0, 'total': 0}
            if question_item['OCR'] == 1:
                ocr_to_counts['total'] += 1
            q_type_to_counts[q_type]['total'] += 1
            if question_item['response'][0] == question_item['answer']:
                q_type_to_counts[q_type]['correct'] += 1
                if question_item['OCR'] == 1:
                    ocr_to_counts['correct'] += 1
    correct, total = 0, 0
    for q_type, counts in q_type_to_counts.items():
        correct += counts['correct']
        total += counts['total']
        print(f'{q_type}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
    print(f'OCR: {ocr_to_counts["correct"]}/{ocr_to_counts["total"]}={ocr_to_counts["correct"]/ocr_to_counts["total"]}')
    print(f'Overall: {correct}/{total}={correct/total}')

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed evaluation for LiveSports3KQA.")
    parser.add_argument("--model_name_or_path", required=True, help="HF model id, snapshot dir, or HF cache repo dir.")
    parser.add_argument("--benchmark_path", default="sports3k-qa.jsonl", help="Path to benchmark jsonl.")
    parser.add_argument("--output_dir", default="evaluation/livesports3kqa/results", help="Directory to save results.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = resolve_model_path(args.model_name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", attn_implementation='flash_attention_2')
    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')
    letter_idxs_predictions, benchmark_datums, process_index = mcq_predict(
        model=model, processor=processor, benchmark_path=args.benchmark_path,
        letters=['A', 'B', 'C', 'D'], use_liger_kernel='LiveCC' in model_path,
    )
    if process_index == 0:
        video_id_to_results = {}
        for datum, letter_idx_prediction in zip(benchmark_datums, letter_idxs_predictions):
            video_id = datum['video_id']
            if video_id not in video_id_to_results:
                video_id_to_results[video_id] = {
                    'video_id': video_id,
                    'questions': [],
                }
            video_id_to_results[video_id]['questions'].append(
                {
                    "question_id": datum['question_id'],
                    "q_type": datum['q_type'],
                    'OCR': datum['OCR'],
                    "question": datum['question'],
                    "options": datum['options'],
                    "answer": datum['answer'],
                    "response": datum['options'][letter_idx_prediction],
                },
            )
        results = list(video_id_to_results.values())
        os.makedirs(args.output_dir, exist_ok=True)
        save_json_path = os.path.join(args.output_dir, f'{os.path.basename(args.model_name_or_path.rstrip("/"))}.json')
        json.dump(results, open(save_json_path, 'w'))
        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_livesports3kqa_results,
            save_txt_path,
            results
        )

# torchrun --standalone --nproc_per_node=8 evaluation/livesports3kqa/distributed_evaluate_livesports3kqa.py --model_name_or_path chenjoya/LiveCC-7B-Instruct
