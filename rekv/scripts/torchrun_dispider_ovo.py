#!/usr/bin/env python3
"""Torchrun wrapper for running single-GPU Dispider OVOBench inference in parallel.

This wrapper does NOT require patching Dispider internals. It:
1) shards OVOBench annotations by rank,
2) launches the original dispider command once per rank,
3) merges per-rank outputs on rank0.

Example:
  torchrun --standalone --nproc_per_node=4 rekv/scripts/torchrun_dispider_ovo.py \
    --anno-path data/ovobench/ovo_bench_new.json \
    --work-dir Dispider \
    --base-cmd "python dispider.py --model_path ... --video_dir ..." \
    --final-output outputs/ovobench/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torchrun wrapper for Dispider on OVOBench")
    parser.add_argument("--anno-path", required=True, help="Full OVOBench annotation JSON")
    parser.add_argument("--base-cmd", required=True, help="Dispider command without anno/output args")
    parser.add_argument("--final-output", required=True, help="Merged output JSON path")
    parser.add_argument("--work-dir", default=".", help="Working directory to launch Dispider command")

    parser.add_argument(
        "--anno-arg-name",
        default="--anno_path",
        help="Arg name used by dispider command for annotation path",
    )
    parser.add_argument(
        "--output-arg-name",
        default="--output_path",
        help="Arg name used by dispider command for output path",
    )
    parser.add_argument(
        "--video-prefix",
        default="",
        help="Optional prefix to join with item['video'] before running",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional OVOBench task filter, e.g. EPM ASI HLD",
    )
    return parser.parse_args()


def _rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _load_and_shard(anno_path: str, rank: int, world_size: int, tasks: list[str] | None, video_prefix: str) -> list[dict]:
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if tasks:
        task_set = set(tasks)
        annotations = [a for a in annotations if a.get("task") in task_set]

    if video_prefix:
        for item in annotations:
            if "video" in item:
                item["video"] = str(Path(video_prefix) / item["video"])

    return annotations[rank::world_size]


def _build_rank_paths(final_output: Path, rank: int) -> tuple[Path, Path]:
    tmp_root = final_output.parent / ".dispider_torchrun_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    rank_anno = tmp_root / f"anno_rank{rank}.json"
    rank_out = tmp_root / f"pred_rank{rank}.json"
    return rank_anno, rank_out


def _run_rank_cmd(args: argparse.Namespace, rank_anno: Path, rank_out: Path, local_rank: int) -> None:
    cmd = shlex.split(args.base_cmd)
    cmd.extend([args.anno_arg_name, str(rank_anno), args.output_arg_name, str(rank_out)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    proc = subprocess.run(cmd, cwd=args.work_dir, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Rank command failed with code {proc.returncode}: {' '.join(cmd)}")


def _merge_results(final_output: Path, world_size: int) -> None:
    tmp_root = final_output.parent / ".dispider_torchrun_tmp"
    merged: list = []

    for rank in range(world_size):
        rank_out = tmp_root / f"pred_rank{rank}.json"
        if not rank_out.exists():
            raise FileNotFoundError(f"Missing rank output: {rank_out}")
        with open(rank_out, "r", encoding="utf-8") as f:
            rank_data = json.load(f)
        if isinstance(rank_data, list):
            merged.extend(rank_data)
        else:
            merged.append(rank_data)

    final_output.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = _rank_info()

    final_output = Path(args.final_output)
    rank_anno, rank_out = _build_rank_paths(final_output, rank)

    rank_data = _load_and_shard(
        anno_path=args.anno_path,
        rank=rank,
        world_size=world_size,
        tasks=args.tasks,
        video_prefix=args.video_prefix,
    )
    with open(rank_anno, "w", encoding="utf-8") as f:
        json.dump(rank_data, f, ensure_ascii=False, indent=2)

    _run_rank_cmd(args, rank_anno, rank_out, local_rank)

    if world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(backend="gloo", init_method="env://")
        dist.barrier()
        if rank == 0:
            _merge_results(final_output, world_size)
        dist.barrier()
        dist.destroy_process_group()
    else:
        _merge_results(final_output, 1)

    if rank == 0:
        print(f"Merged results saved to: {final_output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[torchrun_dispider_ovo] ERROR: {exc}", file=sys.stderr)
        raise
