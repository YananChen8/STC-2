"""StreamingBench solver integrated into the video_qa pipeline.

Handles all StreamingBench variants (real, SQA, proactive) within
the standard distributed inference framework.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

from logzero import logger
from tqdm import tqdm

from .base import BaseVQA


# ------------------------------------------------------------------
# Video splitting utility (uses ffmpeg)
# ------------------------------------------------------------------

def _split_video(video_file: str, start_time: int, end_time: int) -> str:
    """Split a video clip using ffmpeg. Returns path to the output clip."""
    import ffmpeg

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if os.path.exists(output_file):
        return output_file

    try:
        (
            ffmpeg
            .input(video_file, ss=int(start_time))
            .output(output_file, t=(int(end_time) - int(start_time)),
                    vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")

    return output_file


def _ts_to_seconds(ts: str) -> int:
    """Convert timestamp like '00:03:10' to seconds."""
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(ts.split(":"))))


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_MC_PROMPT = '''You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {}

Options:
{}
{}
{}
{}'''

_OPEN_PROMPT = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a question related to the video. Your task is to carefully analyze the video and provide the answer to the question. 

Question: {}
'''

_PROACTIVE_PROMPT = '''You are an advanced image question-answering AI assistant. You have been provided with images and a question related to the images. Your task is to carefully analyze the images and provide the answer to the question. You need to carefully confirm whether the images content meet the conditions of the question, and then output the correct content.

Question: {}

The answer is:
'''

_SQA_PROMPT = '''You are an advanced video question-answering AI assistant. You have been provided with a video and a multiple-choice question related to the video. Your task is to carefully analyze the video and the provided context to answer the question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

{}

Here is the question. Answer it and don't confuse it with the previous conversation.
Question: {}

Options:
{}
{}
{}
{}

The best option is:'''


class StreamingBenchVQA(BaseVQA):
    """StreamingBench solver — handles real, SQA, and proactive benchmarks.

    Processes benchmark data in-process (no subprocess delegation).
    Uses ffmpeg to split video clips on-the-fly.
    """

    def __init__(self, model, processor, args) -> None:
        super().__init__(model, processor, args)
        self.benchmark_name = getattr(args, "benchmark_name", "Streaming")
        self.context_time = getattr(args, "context_time", -1)

    # ------------------------------------------------------------------
    # Main entry — processes a list of annotation entries (not single sample)
    # ------------------------------------------------------------------

    def run_benchmark(self, data: list, output_file: str) -> None:
        """Run the appropriate benchmark variant on the full data."""
        if self.benchmark_name == "StreamingProactive":
            self._eval_proactive(data, output_file)
        elif self.benchmark_name == "StreamingSQA":
            self._eval_sqa(data, output_file)
        else:
            self._eval_standard(data, output_file)

    # ------------------------------------------------------------------
    # Standard StreamingBench (real / omni)
    # ------------------------------------------------------------------

    def _eval_standard(self, data: list, output_file: str) -> None:
        model_key = "rekv"
        for subset in tqdm(data, desc="StreamingBench"):
            for question in subset["questions"]:
                if model_key in question and question[model_key]:
                    continue

                video_path = subset["video_path"]
                timestamp = _ts_to_seconds(question["time_stamp"])
                time_start = max(0, timestamp - self.context_time) if self.context_time > 0 else 0
                clip = _split_video(video_path, time_start, timestamp)

                inp = self._format_question(question)
                response = self.run_clip_inference(clip, inp)
                question[model_key] = response

                self._checkpoint(data, output_file)

    # ------------------------------------------------------------------
    # SQA variant (sequential context)
    # ------------------------------------------------------------------

    def _eval_sqa(self, data: list, output_file: str) -> None:
        model_key = "rekv"
        for video_data in tqdm(data, desc="StreamingBench-SQA"):
            context = ""
            for subset in video_data:
                for question in subset["questions"]:
                    if model_key in question and question[model_key]:
                        continue

                    video_path = subset["video_path"]
                    timestamp = _ts_to_seconds(question["time_stamp"])
                    time_start = max(0, timestamp - self.context_time) if self.context_time > 0 else 0
                    clip = _split_video(video_path, time_start, timestamp)

                    ques = question["question"]
                    options = self._format_options(question["options"])
                    inp = _SQA_PROMPT.format(context, ques, *options)

                    response = self.run_clip_inference(clip, inp)
                    question[model_key] = response

                    if not context:
                        context += ("Here are the contextual information related to the video. "
                                    "Please answer the questions based on the contextual information: ")
                    context += (f"At timestamp {question['time_stamp']}, the following question and answer occurred: "
                                f"Question: {ques}; Options: {', '.join(options)}; Answer: {question['answer']}; ")

                    self._checkpoint(data, output_file)

    # ------------------------------------------------------------------
    # Proactive variant (polling with time window)
    # ------------------------------------------------------------------

    def _eval_proactive(self, data: list, output_file: str) -> None:
        model_key = "rekv"
        for subset in tqdm(data, desc="StreamingBench-Proactive"):
            for question in subset["questions"]:
                if model_key in question and question[model_key]:
                    if question[model_key].get('dialog_history', [{}])[-1].get('content'):
                        continue

                video_path = subset["video_path"]
                start_time = _ts_to_seconds(question["time_stamp"])
                gt_time = _ts_to_seconds(question["ground_truth_time_stamp"])
                max_time = gt_time + 4

                dialog_history = []
                answered = False
                query = (f"{question['question']} Is it the right time to output "
                         f"\"{question['ground_truth_output']}\"? You can only answer yes or no.")
                inp = _PROACTIVE_PROMPT.format(query)

                current_time = start_time + 1
                while current_time <= max_time:
                    clip = _split_video(video_path, start_time, current_time)

                    t0 = time.time()
                    response = self.run_clip_inference(clip, inp)
                    cost = time.time() - t0

                    dialog_history.append({'role': 'user', 'content': query, 'time': current_time, 'cost': cost})
                    dialog_history.append({'role': 'assistant', 'content': response, 'time': current_time, 'cost': cost})

                    if 'yes' in (response or "").strip().lower():
                        inp2 = _PROACTIVE_PROMPT.format(question['question'])
                        t0 = time.time()
                        response2 = self.run_clip_inference(clip, inp2)
                        cost2 = time.time() - t0

                        dialog_history.append({'role': 'user', 'content': question['question'], 'time': current_time, 'cost': cost2})
                        dialog_history.append({'role': 'assistant', 'content': response2, 'time': current_time, 'cost': cost2})
                        answered = current_time
                        break

                    current_time += 1

                question[model_key] = {"answered": answered, "dialog_history": dialog_history}
                self._checkpoint(data, output_file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_question(question: dict) -> str:
        ques = question["question"]
        if "options" in question:
            options = question["options"]
            if not options[0].startswith("A."):
                options = [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
            return _MC_PROMPT.format(ques, *options) + "\n\nThe best option is:"
        return _OPEN_PROMPT.format(ques) + "\n\nAnswer:"

    @staticmethod
    def _format_options(options: list) -> list[str]:
        if not options[0].startswith("A."):
            return [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
        return options

    @staticmethod
    def _checkpoint(data: list, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_results(self, save_path: str) -> None:
        """No-op: StreamingBench results are saved inline via checkpointing."""
        pass


# ------------------------------------------------------------------
# Scoring utilities
# ------------------------------------------------------------------

def score_streamingbench(
    results_path: str,
    model_name: str = "rekv",
    task: str = "real",
) -> dict[str, dict]:
    """Calculate StreamingBench scores from result JSON."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    stats = defaultdict(lambda: defaultdict(int))

    if task == "sqa":
        for ques in data:
            for entry in ques:
                for question in entry["questions"]:
                    task_type = question["task_type"]
                    if model_name not in question or not question.get(model_name):
                        continue
                    model_answer = _get_answer(question, model_name)
                    correct_answer = question["answer"]
                    if model_answer:
                        stats[task_type]["total"] += 1
                        if correct_answer == model_answer:
                            stats[task_type]["correct"] += 1

    elif task == "proactive":
        for entry in data:
            for question in entry["questions"]:
                if model_name not in question:
                    continue
                gt_ts = question["ground_truth_time_stamp"]
                gt_time = _ts_to_seconds(gt_ts)
                task_type = question["task_type"]
                model_data = question.get(model_name)
                if not model_data:
                    continue
                history = model_data["dialog_history"]
                last_time = history[-1]["time"]
                last_answer = history[-1]["content"]

                stats[task_type]["total"] += 1
                if -2 <= last_time - gt_time <= 2:
                    stats[task_type]["time_correct"] += 1
                    if question["ground_truth_output"] in (last_answer or ""):
                        stats[task_type]["answer_correct"] += 1
    else:
        # real / omni
        for entry in data:
            for question in entry["questions"]:
                task_type = question["task_type"]
                if model_name not in question:
                    continue
                model_answer = _get_answer(question, model_name)
                correct_answer = question["answer"]
                if model_answer:
                    stats[task_type]["total"] += 1
                    stats["total"]["total"] += 1
                    if model_answer == correct_answer:
                        stats[task_type]["correct"] += 1
                        stats["total"]["correct"] += 1

    result = {}
    if task == "proactive":
        for tt, counts in stats.items():
            total = counts["total"]
            result[tt] = {
                "total": total,
                "time_correct": counts.get("time_correct", 0),
                "time_accuracy": counts.get("time_correct", 0) / total if total > 0 else 0,
                "answer_correct": counts.get("answer_correct", 0),
                "answer_accuracy": counts.get("answer_correct", 0) / total if total > 0 else 0,
            }
    else:
        for tt, counts in stats.items():
            total = counts["total"]
            correct = counts.get("correct", 0)
            result[tt] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
            }

    return result


def _get_answer(question: dict, model_key: str) -> str | None:
    raw = question.get(model_key)
    if raw is None:
        return None
    ans = raw[0] if isinstance(raw, list) else raw
    return ans.strip() if isinstance(ans, str) else None
