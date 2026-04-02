"""OVOBench solver integrated into the video_qa pipeline.

Handles the three OVOBench task categories (backward, realtime, forward)
within the standard distributed inference framework.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from logzero import logger

from .base import BaseVQA


# ------------------------------------------------------------------
# Task categories
# ------------------------------------------------------------------

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]
ALL_TASKS = BACKWARD_TASKS + REALTIME_TASKS + FORWARD_TASKS

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""


class OVOBenchVQA(BaseVQA):
    """OVOBench solver — processes annotation items with chunked video clips.

    Unlike standard solvers, OVOBench annotations are per-item (not per-video),
    and each item has a pre-chunked video clip.
    """

    def __init__(self, model, processor, args) -> None:
        super().__init__(model, processor, args)
        self.chunked_dir = getattr(args, "chunked_dir", "data/ovobench/chunked_videos")

    # ------------------------------------------------------------------
    # Main entry — processes a single OVOBench annotation item
    # ------------------------------------------------------------------

    def __call__(self, anno_item: dict) -> list[dict]:
        task = anno_item["task"]

        if task in BACKWARD_TASKS + REALTIME_TASKS:
            result = self._process_br(anno_item)
            self.results.append(result)
            return [result]
        elif task in FORWARD_TASKS:
            results = self._process_forward(anno_item)
            self.results.extend(results)
            return results
        else:
            logger.warning(f"Unknown OVOBench task: {task}")
            return []

    # ------------------------------------------------------------------
    # Backward / Realtime tasks
    # ------------------------------------------------------------------

    def _process_br(self, item: dict) -> dict:
        prompt = self._build_br_prompt(item["question"], item["options"])
        chunk_path = os.path.join(self.chunked_dir, f"{item['id']}.mp4")

        response = self.run_clip_inference(chunk_path, prompt, strip_last_line=True)

        return {
            "id": item["id"],
            "video": item.get("video", ""),
            "task": item["task"],
            "question": item["question"],
            "response": response,
            "ground_truth": chr(65 + item["gt"]),
        }

    # ------------------------------------------------------------------
    # Forward tasks (REC, SSR, CRR)
    # ------------------------------------------------------------------

    def _process_forward(self, item: dict) -> list[dict]:
        item_result = {
            "id": item["id"],
            "video": item.get("video", ""),
            "task": item["task"],
            "test_info": [],
        }

        for i, ti in enumerate(item.get("test_info", [])):
            prompt = self._build_forward_prompt(item["task"], item, i)
            chunk_path = os.path.join(self.chunked_dir, f"{item['id']}_{i}.mp4")

            response = self.run_clip_inference(chunk_path, prompt, strip_last_line=True)

            ti_result = {**ti, "response": response}
            item_result["test_info"].append(ti_result)

        return [item_result]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_br_prompt(question: str, options: list) -> str:
        formatted = '; '.join(f'{chr(65 + i)}. {opt}' for i, opt in enumerate(options)) + ';'
        return BR_PROMPT_TEMPLATE.format(question, formatted)

    @staticmethod
    def _build_forward_prompt(task: str, anno: dict, index: int) -> str:
        if task == "REC":
            q = "How many times did they " + anno["activity"] + "?"
            return REC_PROMPT_TEMPLATE.format(q)
        elif task == "SSR":
            return SSR_PROMPT_TEMPLATE.format(anno["test_info"][index]["step"])
        elif task == "CRR":
            return CRR_PROMPT_TEMPLATE.format(anno["question"])
        raise ValueError(f"Unknown forward task: {task}")

    # ------------------------------------------------------------------
    # Persistence — override to save as JSON (not CSV)
    # ------------------------------------------------------------------

    def save_results(self, save_path: str) -> None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self._to_categorized_results(), f, indent=2)
        logger.info(f"Saved {len(self.results)} OVOBench results to {save_path}")

    def _to_categorized_results(self) -> dict:
        categorized = {"backward": [], "realtime": [], "forward": []}
        for r in self.results:
            task = r.get("task", "")
            if task in BACKWARD_TASKS:
                categorized["backward"].append(r)
            elif task in REALTIME_TASKS:
                categorized["realtime"].append(r)
            elif task in FORWARD_TASKS:
                categorized["forward"].append(r)
        return categorized


# ------------------------------------------------------------------
# Scoring utilities
# ------------------------------------------------------------------

def score_ovobench(results: dict) -> dict[str, float]:
    """Calculate OVOBench scores from categorized results."""
    scores = {}
    category_avgs = {}

    for category in ("backward", "realtime"):
        items = results.get(category, [])
        if not items:
            continue
        task_scores: dict[str, list[int]] = {}
        for item in items:
            task = item["task"]
            task_scores.setdefault(task, []).append(
                _score_br(item.get("response"), item.get("ground_truth", ""))
            )

        cat_accs = []
        for task, vals in task_scores.items():
            acc = 100 * sum(vals) / len(vals)
            scores[f"{category}/{task}"] = acc
            cat_accs.append(sum(vals) / len(vals))

        cat_avg = 100 * sum(cat_accs) / len(cat_accs) if cat_accs else 0
        scores[f"{category}/avg"] = cat_avg
        category_avgs[category] = cat_avg

    forward_items = results.get("forward", [])
    if forward_items:
        task_scores: dict[str, list[int]] = {}
        for item in forward_items:
            task = item["task"]
            if task == "REC":
                for ti in item.get("test_info", []):
                    task_scores.setdefault("REC", []).append(
                        _score_rec(ti.get("response"), ti.get("count"))
                    )
            elif task in ("SSR", "CRR"):
                for ti in item.get("test_info", []):
                    gt = "No" if ti.get("type") == 0 else "Yes"
                    task_scores.setdefault(task, []).append(
                        _score_yn(ti.get("response"), gt)
                    )

        fwd_accs = []
        for task, vals in task_scores.items():
            acc = 100 * sum(vals) / len(vals)
            scores[f"forward/{task}"] = acc
            fwd_accs.append(sum(vals) / len(vals))

        fwd_avg = 100 * sum(fwd_accs) / len(fwd_accs) if fwd_accs else 0
        scores["forward/avg"] = fwd_avg
        category_avgs["forward"] = fwd_avg

    if category_avgs:
        scores["total_avg"] = sum(category_avgs.values()) / len(category_avgs)

    return scores


def score_ovobench_from_dir(result_dir: str) -> dict[str, float]:
    """Load all JSON files from a directory and score."""
    merged = {"backward": [], "realtime": [], "forward": []}
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    for fname in os.listdir(result_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(result_dir, fname), "r") as f:
            data = json.load(f)
        for key in merged:
            merged[key].extend(data.get(key, []))
    return score_ovobench(merged)


def _score_br(response: str | None, gt: str) -> int:
    if response is None:
        return 0
    return int(gt in response)


def _score_rec(response: str | None, gt_count) -> int:
    if response is None:
        return 0
    digits = re.findall(r'\d+', response)
    return int("".join(digits) == str(gt_count))


def _score_yn(response: str | None, gt: str) -> int:
    if response is None:
        return 0
    r = response.strip()
    if (r == "N" and gt == "No") or (r == "Y" and gt == "Yes"):
        return 1
    return int(gt in r)
