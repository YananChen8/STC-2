"""Open-ended video QA evaluation using GPT semantic matching.

Provides both CLI and programmatic API for evaluating open-ended QA.

CLI Usage::

    python -m model.video_qa.eval.eval_open_ended \\
        --pred_path results.csv \\
        --output_dir annotations/ \\
        --output_json combined_results.json \\
        --num_tasks 16

Programmatic Usage::

    from model.video_qa.eval.eval_open_ended import run_open_ended_eval
    run_open_ended_eval("results.csv", "annotations/", "combined.json")
"""

from __future__ import annotations

import ast
import json
import os
import time
import argparse
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm
import openai


# ---------------------------------------------------------------
# GPT Service
# ---------------------------------------------------------------

class GPTService:
    """Wrapper for GPT-based semantic matching evaluation."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0613",
        max_tokens: int = 300,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", None),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    def evaluate(self, prompt: list[dict]) -> str | None:
        """Call GPT with retries (up to 10 attempts)."""
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0,
                )
                content = json.loads(completion.model_dump_json())
                return content["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"GPT error: {e}")
                time.sleep(1)
        return None


_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the correctness "
    "of generative outputs for question-answer pairs. "
    "Your task is to compare the predicted answer with the correct answer "
    "and determine if they match meaningfully. "
    "Focus on the meaningful match between the predicted answer and the "
    "correct answer. Consider synonyms or paraphrases as valid matches. "
    "Evaluate the correctness of the prediction compared to the answer."
)

_USER_TEMPLATE = (
    "Please evaluate the following video-based question-answer pair:\n\n"
    "Question: {question}\n"
    "Correct Answer: {answer}\n"
    "Predicted Answer: {pred}\n\n"
    "Provide your evaluation only as a yes/no and score where the score is "
    "an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
    "Please generate the response in the form of a Python dictionary string "
    "with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' "
    "and value of 'score' is in INTEGER, not STRING. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. "
    "Only provide the Python dictionary string. "
    "For example, your response should look like this: {{'pred': 'yes', 'score': 4}}."
)


# ---------------------------------------------------------------
# Worker
# ---------------------------------------------------------------

def _annotate_batch(prediction_set: dict, caption_files: list[str], output_dir: str) -> None:
    """Evaluate QA pairs using GPT and save per-sample annotations."""
    gpt = GPTService()
    for fname in tqdm(caption_files, leave=False):
        key = fname[:-5]  # Strip .json
        qa = prediction_set[key]
        try:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    question=qa["question"], answer=qa["answer"], pred=qa["pred_answer"],
                )},
            ]
            response = gpt.evaluate(messages)
            if response is None:
                continue
            result = ast.literal_eval(response)
            with open(os.path.join(output_dir, f"{key}.json"), "w") as f:
                json.dump([result, qa], f)
        except Exception as e:
            print(f"Error processing '{key}': {e}")


# ---------------------------------------------------------------
# Core evaluation function (programmatic API)
# ---------------------------------------------------------------

def run_open_ended_eval(
    pred_path: str,
    output_dir: str,
    output_json: str,
    num_tasks: int = 16,
) -> dict:
    """Run GPT-based open-ended QA evaluation.

    Args:
        pred_path: Path to prediction CSV (must contain video_id, question, answer, pred_answer).
        output_dir: Directory for per-sample annotation JSONs.
        output_json: Path for combined results JSON.
        num_tasks: Number of parallel annotation workers.

    Returns:
        Combined results dict with accuracy and average_score.
    """
    pred_contents = pd.read_csv(pred_path).to_dict(orient="records")

    # Filter to specific config if present
    if pred_contents and "retrieve_size" in pred_contents[0]:
        pred_contents = [
            x for x in pred_contents
            if x["retrieve_size"] == 64 and x["chunk_size"] == 1
        ]

    # Deduplicate video_ids
    video_id_counts: dict[str, int] = {}
    for sample in pred_contents:
        vid = sample["video_id"]
        video_id_counts[vid] = video_id_counts.get(vid, -1) + 1
        sample["video_id"] = f"{vid}_{video_id_counts[vid]}"

    caption_files = [f"{s['video_id']}.json" for s in pred_contents]
    os.makedirs(output_dir, exist_ok=True)

    prediction_set = {
        s["video_id"]: {
            "question": s["question"],
            "answer": s["answer"],
            "pred_answer": s["pred_answer"],
        }
        for s in pred_contents
    }

    # Retry loop until all samples processed
    workers = num_tasks
    while True:
        completed = set(os.listdir(output_dir))
        incomplete = [f for f in caption_files if f not in completed]
        print(f"Completed: {len(completed)}, Remaining: {len(incomplete)}")
        if not incomplete:
            break

        if len(incomplete) <= workers:
            workers = 1

        part_len = max(1, len(incomplete) // workers)
        parts = [incomplete[i: i + part_len] for i in range(0, len(incomplete), part_len)]
        task_args = [(prediction_set, part, output_dir) for part in parts]

        try:
            with Pool() as pool:
                pool.starmap(_annotate_batch, task_args)
        except Exception as e:
            print(f"Pool error: {e}")

    # Combine annotations
    combined: dict = {}
    for fname in os.listdir(output_dir):
        if fname.endswith(".json"):
            with open(os.path.join(output_dir, fname)) as f:
                combined[fname[:-5]] = json.load(f)

    # Compute summary
    score_sum = yes_count = no_count = count = 0
    for result in combined.values():
        if not isinstance(result, list) or len(result) < 1:
            continue
        count += 1
        score_sum += int(result[0].get("score", 0))
        pred = result[0].get("pred", result[0].get("prev", ""))
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    if count > 0:
        avg_score = score_sum / count
        accuracy = yes_count / max(yes_count + no_count, 1)
        combined["average_score"] = avg_score
        combined["accuracy"] = accuracy
        print(f"\nYes: {yes_count}, No: {no_count}")
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(f"Average score: {avg_score:.2f}")

    with open(output_json, "w") as f:
        json.dump(combined, f)
    print(f"Results saved: {output_json}")

    return combined


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Open-ended QA evaluation via GPT")
    parser.add_argument("--pred_path", required=True, help="Path to prediction CSV")
    parser.add_argument("--output_dir", required=True, help="Directory for annotations")
    parser.add_argument("--output_json", required=True, help="Path for combined JSON")
    parser.add_argument("--num_tasks", default=16, type=int, help="Parallel workers")
    args = parser.parse_args()

    run_open_ended_eval(args.pred_path, args.output_dir, args.output_json, args.num_tasks)


if __name__ == "__main__":
    main()
