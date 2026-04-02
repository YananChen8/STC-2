"""Evaluation dispatch utilities."""

from __future__ import annotations

import argparse

from logzero import logger


def run_evaluation(eval_type: str, save_dir: str) -> None:
    """Run evaluation using the unified evaluator.

    Args:
        eval_type: Evaluation strategy key (e.g. 'multiple_choice', 'videomme').
        save_dir: Directory containing results.csv.
    """
    from model.video_qa.eval.evaluate import run_eval

    logger.info(f"Running evaluation: type={eval_type}, save_dir={save_dir}")
    args = argparse.Namespace(
        save_dir=save_dir,
        results_path=None,
        debug=False,
        pred_path=None,
        output_dir=None,
        output_json=None,
        num_tasks=16,
    )
    run_eval(eval_type, save_dir, args)
