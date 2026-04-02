"""Unified evaluation entry point for all benchmarks.

Dispatches to the appropriate evaluation strategy based on ``--eval_type``.

Usage::

    python -m model.video_qa.eval.evaluate --save_dir results/ --eval_type multiple_choice
    python -m model.video_qa.eval.evaluate --save_dir results/ --eval_type videomme
    python -m model.video_qa.eval.evaluate --save_dir results/ --eval_type ovobench
    python -m model.video_qa.eval.evaluate --save_dir results/ --eval_type streamingbench \
        --results_path results/streaming_results.json --streaming_task real
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------

def load_results(save_dir: str, results_path: str | None = None) -> pd.DataFrame:
    """Load results CSV from either explicit path or save_dir/results.csv."""
    if results_path is not None:
        return pd.read_csv(results_path)
    return pd.read_csv(os.path.join(save_dir, "results.csv"))


def group_by_config(df: pd.DataFrame) -> dict | list:
    """Group results by (retrieve_size, chunk_size) if columns exist, else return list."""
    if "retrieve_size" not in df.columns:
        return df.to_dict(orient="records")

    groups: dict[tuple, list] = {}
    for _, row in df.iterrows():
        key = (row["retrieve_size"], row["chunk_size"])
        record = {c: row[c] for c in df.columns if c not in ("retrieve_size", "chunk_size")}
        groups.setdefault(key, []).append(record)
    return groups


def calc_average_metric(
    results: dict | list,
    save_dir: str,
    metric: str,
    *,
    task: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> float | None:
    """Compute and display average metric. Generates heatmap for multi-config results."""
    if isinstance(results, list):
        items = [r for r in results if (task is None or r.get("task") == task)]
        if not items:
            print(f"  No samples for task={task}")
            return None
        avg = sum(r[metric] for r in items) / len(items)
        label = f"[{task}] " if task else ""
        print(f"  {label}#Samples: {len(items)}, Average {metric}: {avg:.2f}")
        return avg

    if isinstance(results, dict):
        averages = {}
        for key, records in results.items():
            items = [r for r in records if (task is None or r.get("task") == task)]
            averages[key] = (sum(r[metric] for r in items) / len(items)) if items else None

        _plot_heatmap(averages, save_dir, metric, task=task, vmin=vmin, vmax=vmax)
        sample_count = len(next(iter(results.values())))
        print(f"  #Samples per config: {sample_count}")
        return None

    raise ValueError(f"Invalid results type: {type(results)}")


def _plot_heatmap(
    averages: dict, save_dir: str, metric: str, *, task: str | None = None,
    vmin: float | None = None, vmax: float | None = None,
) -> None:
    """Generate and save a seaborn heatmap for config-wise metrics."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  [warn] matplotlib/seaborn not available, skipping heatmap.")
        return

    df = pd.DataFrame.from_dict(averages, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index, names=["retrieve_size", "chunk_size"])
    df = df.reset_index()
    df.columns = ["retrieve_size", "chunk_size", "value"]
    heatmap_data = df.pivot(index="chunk_size", columns="retrieve_size", values="value")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_data, annot=True, fmt=".1f", cmap="RdPu",
        cbar_kws={"label": "Value"}, xticklabels=True, yticklabels=True,
        vmin=vmin, vmax=vmax,
    )
    ax.invert_yaxis()
    suffix = f"-{task}" if task else ""
    plt.title(f"Heatmap of Average {metric.capitalize()}{suffix}")
    plt.xlabel("Retrieve Size")
    plt.ylabel("Chunk Size")
    plt.tight_layout()

    filename = f"{task + '-' if task else ''}{metric}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"  Heatmap saved: {filepath}")


def count_format_errors(df: pd.DataFrame, *, debug: bool = False) -> None:
    """Count and report prediction format errors."""
    if "pred_choice" not in df.columns:
        return
    valid = set("ABCDEFGH")
    errors = 0
    for _, row in df.iterrows():
        pred = str(row.get("pred_answer", ""))
        if not pred or pred[0] not in valid:
            errors += 1
            if debug:
                print(f"  Error: video={row['video_id']}, gt={row.get('correct_choice')}, pred={pred}")
    print(f"  Format errors: {errors}/{len(df)} ({errors / max(len(df), 1) * 100:.2f}%)")


# ---------------------------------------------------------------
# Evaluation strategies
# ---------------------------------------------------------------

def eval_multiple_choice(df: pd.DataFrame, save_dir: str, debug: bool = False) -> None:
    """Evaluate multiple-choice QA accuracy."""
    print("\n=== Multiple Choice Evaluation ===")
    results = group_by_config(df)
    metrics = ["recall", "precision", "f1", "qa_acc", "acc_at_gqa"] if "recall" in df.columns else ["qa_acc"]
    for metric in metrics:
        calc_average_metric(results, save_dir, metric)
    count_format_errors(df, debug=debug)


def eval_videomme(df: pd.DataFrame, save_dir: str, debug: bool = False) -> None:
    """Evaluate VideoMME with per-duration breakdown."""
    print("\n=== VideoMME Evaluation ===")

    total = len(df)
    correct = (df["qa_acc"] == 100.0).sum()
    overall = correct / total * 100 if total > 0 else 0
    print(f"  Overall: {overall:.2f}% ({correct}/{total})")

    for duration in ("short", "medium", "long"):
        sub = df[df["duration"] == duration]
        if len(sub) > 0:
            dur_correct = (sub["qa_acc"] == 100.0).sum()
            dur_acc = dur_correct / len(sub) * 100
            print(f"  {duration.capitalize()}: {dur_acc:.2f}% ({dur_correct}/{len(sub)})")
        else:
            print(f"  {duration.capitalize()}: N/A")

    print(f"  Mean qa_acc: {df['qa_acc'].mean():.2f}, Std: {df['qa_acc'].std():.2f}")

    results = group_by_config(df)
    metrics = ["recall", "precision", "f1", "qa_acc", "acc_at_gqa"] if "recall" in df.columns else ["qa_acc"]
    for metric in metrics:
        calc_average_metric(results, save_dir, metric)
    count_format_errors(df, debug=debug)


def eval_mlvu_by_task(df: pd.DataFrame, save_dir: str, debug: bool = False) -> None:
    """Evaluate MLVU with per-task breakdown."""
    print("\n=== MLVU Per-Task Evaluation ===")
    results = group_by_config(df)
    tasks = ["plotQA", "findNeedle", "ego", "count", "order", "anomaly_reco", "topic_reasoning"]
    for task in tasks:
        print(f"\n  Task: {task}")
        calc_average_metric(results, save_dir, "qa_acc", task=task)
    count_format_errors(df, debug=debug)


def eval_egoschema(df: pd.DataFrame, save_dir: str, **_) -> None:
    """Convert EgoSchema predictions to submission format."""
    print("\n=== EgoSchema Submission Generation ===")
    if "retrieve_size" in df.columns:
        mask = (df["retrieve_size"] == df["retrieve_size"].max()) & (df["chunk_size"] == 1)
        records = df[mask].to_dict(orient="records")
    else:
        records = df.to_dict(orient="records")

    submission = []
    for r in records:
        choice = str(r.get("pred_choice", "A"))
        if choice not in "ABCDE":
            choice = "A"
        submission.append({"q_uid": r["video_id"], "answer": ord(choice) - ord("A")})

    out_path = os.path.join(save_dir, "submission.csv")
    pd.DataFrame(submission).to_csv(out_path, index=False)
    print(f"  Submission saved: {out_path} ({len(submission)} samples)")


def eval_egoschema_subset(df: pd.DataFrame, save_dir: str, **_) -> None:
    """Evaluate EgoSchema subset with basic accuracy metrics."""
    print("\n=== EgoSchema Subset Evaluation ===")
    total = len(df)
    correct = (df["qa_acc"] > 0).sum()
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

    if "pred_choice" in df.columns:
        pred_dist = df["pred_choice"].value_counts().sort_index()
        print(f"  Prediction distribution: {pred_dist.to_dict()}")

    count_format_errors(df)


def eval_open_ended(save_dir: str, args: argparse.Namespace) -> None:
    """Run GPT-based open-ended QA evaluation."""
    print("\n=== Open-Ended QA Evaluation (GPT) ===")
    pred_path = getattr(args, "pred_path", None) or os.path.join(save_dir, "results.csv")
    output_dir = getattr(args, "output_dir", None) or os.path.join(save_dir, "annotations")
    output_json = getattr(args, "output_json", None) or os.path.join(save_dir, "combined_results.json")
    num_tasks = getattr(args, "num_tasks", 16)

    from .eval_open_ended import run_open_ended_eval
    run_open_ended_eval(pred_path, output_dir, output_json, num_tasks)


def eval_ovobench(save_dir: str, args: argparse.Namespace) -> None:
    """Evaluate OVOBench results from JSON files in result directory."""
    print("\n=== OVOBench Evaluation ===")
    import json
    from model.video_qa.ovobench import score_ovobench, score_ovobench_from_dir

    result_path = getattr(args, "results_path", None)
    if result_path and result_path.endswith(".json"):
        # Single JSON file
        with open(result_path, "r") as f:
            data = json.load(f)
        scores = score_ovobench(data)
    else:
        # Try results.json in save_dir, or scan directory for JSONs
        json_file = os.path.join(save_dir, "results.json")
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)
            scores = score_ovobench(data)
        else:
            scores = score_ovobench_from_dir(save_dir)

    for key, val in sorted(scores.items()):
        print(f"  {key}: {val:.2f}")


def eval_streamingbench(save_dir: str, args: argparse.Namespace) -> None:
    """Evaluate StreamingBench results from output JSON."""
    print("\n=== StreamingBench Evaluation ===")
    from model.video_qa.streamingbench import score_streamingbench

    results_path = getattr(args, "results_path", None) or os.path.join(save_dir, "results.json")
    model_name = getattr(args, "streaming_model", "rekv")
    task = getattr(args, "streaming_task", "real")

    scores = score_streamingbench(results_path, model_name=model_name, task=task)
    for task_type, stats in sorted(scores.items()):
        if "accuracy" in stats:
            print(f"  {task_type}: {stats['accuracy']*100:.2f}% ({stats['correct']}/{stats['total']})")
        else:
            print(f"  {task_type}: time_acc={stats.get('time_accuracy',0)*100:.2f}%, "
                  f"ans_acc={stats.get('answer_accuracy',0)*100:.2f}% (n={stats['total']})")


# ---------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------

_EVAL_STRATEGIES = {
    "multiple_choice": eval_multiple_choice,
    "videomme": eval_videomme,
    "mlvu_by_task": eval_mlvu_by_task,
    "egoschema": eval_egoschema,
    "egoschema_subset": eval_egoschema_subset,
}

# Strategies that take (save_dir, args) instead of (df, save_dir, debug)
_SPECIAL_STRATEGIES = {
    "open_ended": eval_open_ended,
    "ovobench": eval_ovobench,
    "streamingbench": eval_streamingbench,
}


def run_eval(eval_type: str, save_dir: str, args: argparse.Namespace) -> None:
    """Public API: run evaluation by type name."""
    if eval_type in _SPECIAL_STRATEGIES:
        _SPECIAL_STRATEGIES[eval_type](save_dir, args)
        return

    strategy = _EVAL_STRATEGIES.get(eval_type)
    if strategy is None:
        all_types = list(_EVAL_STRATEGIES) + list(_SPECIAL_STRATEGIES)
        raise ValueError(f"Unknown eval_type: {eval_type}. Available: {all_types}")

    df = load_results(save_dir, getattr(args, "results_path", None))
    if getattr(args, "results_path", None):
        save_dir = os.path.dirname(args.results_path)
    strategy(df, save_dir, debug=getattr(args, "debug", False))


def resolve_eval_type(eval_script: str) -> str:
    """Convert a legacy eval_script path to an eval_type key."""
    eval_script = eval_script.replace("\\", "/")
    stem = Path(eval_script).stem.replace("eval_", "")
    all_types = set(_EVAL_STRATEGIES) | set(_SPECIAL_STRATEGIES)
    if stem in all_types:
        return stem
    raise ValueError(f"Cannot resolve eval_type from script: {eval_script}")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    all_types = list(_EVAL_STRATEGIES) + list(_SPECIAL_STRATEGIES)
    parser = argparse.ArgumentParser(description="Unified evaluation for video QA benchmarks")
    parser.add_argument("--save_dir", required=True, help="Directory containing results")
    parser.add_argument("--results_path", default=None, help="Explicit path to results file")
    parser.add_argument("--eval_type", required=True, choices=all_types, help="Evaluation strategy")
    parser.add_argument("--debug", action="store_true")
    # Open-ended specific
    parser.add_argument("--pred_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--num_tasks", type=int, default=16)
    # StreamingBench specific
    parser.add_argument("--streaming_model", default="rekv")
    parser.add_argument("--streaming_task", default="real",
                        choices=["real", "omni", "sqa", "proactive"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_eval(args.eval_type, args.save_dir, args)


if __name__ == "__main__":
    main()
