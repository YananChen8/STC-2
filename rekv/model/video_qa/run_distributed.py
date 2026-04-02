"""Distributed video QA inference with automatic evaluation.

One-command pipeline: inference → gather → save → evaluate.
All benchmarks use a unified torchrun-based distributed flow.

Usage::

    torchrun --nproc_per_node=N -m model.video_qa.run_distributed \
        --dataset mlvu --model llava_ov_7b --save_dir results/mlvu
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from logzero import logger
from tqdm import tqdm

from .configs import DATASETS


# ------------------------------------------------------------------
# Main pipeline (unified for all benchmarks)
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset_config = DATASETS[args.dataset]

    # Inject extra config into args so solvers can access it
    for key, val in dataset_config.extra.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)

    _init_distributed(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load data and split across ranks
    data_chunk = _load_and_split_data(dataset_config, args, rank, world_size)

    # Load model
    from .utils import load_model
    model, processor = load_model(
        args.model, n_local=args.n_local, device=rank % torch.cuda.device_count(),
        topk=args.retrieve_size, chunk_size=args.retrieve_chunk_size,
    )
    dist.barrier()

    # Create solver and run inference
    from .solver_factory import create_solver
    solver = create_solver(dataset_config.solver, model, processor, args)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    solver_name = dataset_config.solver

    if solver_name == "streamingbench_vqa":
        # StreamingBench: solver manages its own iteration and checkpointing
        temp_file = save_dir / f"results_rank{rank}.json"
        solver.run_benchmark(data_chunk, str(temp_file))
    else:
        # Standard / OVOBench: iterate over samples
        for sample in tqdm(data_chunk, desc=f"[Rank {rank}/{world_size}]", disable=rank != 0):
            solver(sample)

    # Gather and save results
    _gather_and_save(solver, dataset_config, args, rank, world_size, save_dir)

    dist.destroy_process_group()


# ------------------------------------------------------------------
# Data loading (polymorphic by solver type)
# ------------------------------------------------------------------

def _load_and_split_data(config, args, rank: int, world_size: int) -> list:
    """Load annotations and split across ranks.

    Different benchmarks have different data structures, but the
    split logic is uniform (contiguous chunking for online benchmarks,
    strided indexing for offline).
    """
    solver_name = config.solver

    if solver_name == "ovobench_vqa":
        return _load_ovobench_data(config, args, rank, world_size)

    if solver_name == "streamingbench_vqa":
        return _load_streamingbench_data(config, rank, world_size)

    # Standard offline / streaming benchmarks: strided split
    from .utils import load_and_split_anno
    return load_and_split_anno(config.anno_path, world_size, rank)


def _load_ovobench_data(config, args, rank: int, world_size: int) -> list:
    """Load OVOBench annotations, filter by tasks, and split by rank."""
    with open(config.anno_path, "r") as f:
        annotations = json.load(f)

    video_dir = getattr(args, "video_dir", "data/ovobench/src_videos")
    for item in annotations:
        item["video"] = os.path.join(video_dir, item["video"])

    tasks = getattr(args, "tasks", None) or []
    if tasks:
        annotations = [a for a in annotations if a["task"] in tasks]

    # Contiguous split
    chunk_size = (len(annotations) + world_size - 1) // world_size
    chunk = annotations[rank * chunk_size: min((rank + 1) * chunk_size, len(annotations))]

    if rank == 0:
        logger.info(f"OVOBench: {len(annotations)} total, {len(chunk)} for rank {rank}")
    return chunk


def _load_streamingbench_data(config, rank: int, world_size: int) -> list:
    """Load StreamingBench data and split by rank."""
    with open(config.anno_path, "r") as f:
        data = json.load(f)

    chunk_size = (len(data) + world_size - 1) // world_size
    chunk = data[rank * chunk_size: min((rank + 1) * chunk_size, len(data))]

    if rank == 0:
        logger.info(f"StreamingBench: {len(data)} total, {len(chunk)} for rank {rank}")
    return chunk


# ------------------------------------------------------------------
# Gather, save, and evaluate
# ------------------------------------------------------------------

def _gather_and_save(solver, config, args, rank, world_size, save_dir: Path) -> None:
    """Gather results from all ranks, save, and optionally evaluate."""
    solver_name = config.solver

    if solver_name == "streamingbench_vqa":
        _gather_streamingbench(config, args, rank, world_size, save_dir)
    elif solver_name == "ovobench_vqa":
        _gather_and_save_json(solver, config, args, rank, world_size, save_dir)
    else:
        _gather_and_save_csv(solver, config, args, rank, world_size, save_dir)


def _gather_and_save_csv(solver, config, args, rank, world_size, save_dir: Path) -> None:
    """Standard CSV-based gather for offline/streaming benchmarks."""
    if rank == 0:
        logger.info(f"Solver: {config.solver}, samples: {len(solver.results)}")

    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(obj=solver.results, object_gather_list=gathered, dst=0)

    if rank == 0:
        all_results = [r for rank_results in gathered for r in rank_results]
        logger.info(f"Gathered {len(all_results)} results from {world_size} ranks")

        result_file = save_dir / "results.csv"
        pd.DataFrame(all_results).to_csv(result_file, index=False)
        logger.info(f"Results saved: {result_file}")

        if not args.skip_eval:
            _run_evaluation(config.eval_type, str(save_dir), args)


def _gather_and_save_json(solver, config, args, rank, world_size, save_dir: Path) -> None:
    """JSON-based gather for OVOBench (categorized results)."""
    if rank == 0:
        logger.info(f"OVOBench rank 0: {len(solver.results)} results")

    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(obj=solver.results, object_gather_list=gathered, dst=0)

    if rank == 0:
        all_results = [r for rank_results in gathered for r in rank_results]
        logger.info(f"OVOBench gathered {len(all_results)} results from {world_size} ranks")

        # Re-use solver's save logic for categorization
        from .ovobench import OVOBenchVQA
        tmp_solver = OVOBenchVQA.__new__(OVOBenchVQA)
        tmp_solver.results = all_results

        result_file = save_dir / "results.json"
        tmp_solver.save_results(str(result_file))

        if not args.skip_eval:
            _run_evaluation("ovobench", str(save_dir), args)


def _gather_streamingbench(config, args, rank, world_size, save_dir: Path) -> None:
    """File-based gather for StreamingBench (each rank writes temp JSON)."""
    dist.barrier()

    if rank == 0:
        merged = []
        for r in range(world_size):
            fpath = save_dir / f"results_rank{r}.json"
            if fpath.exists():
                with open(fpath, "r") as f:
                    rank_data = json.load(f)
                    if isinstance(rank_data, list):
                        merged.extend(rank_data)
                fpath.unlink()

        streaming_task = getattr(args, "streaming_task", "real")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = save_dir / f"results_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(f"StreamingBench results saved: {result_file} ({len(merged)} entries)")

        if not args.skip_eval:
            args.results_path = str(result_file)
            args.streaming_model = "rekv"
            args.streaming_task = streaming_task
            _run_evaluation("streamingbench", str(save_dir), args)


# ------------------------------------------------------------------
# Distributed setup
# ------------------------------------------------------------------

def _init_distributed(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "Distributed inference requires CUDA"
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()

    torch.manual_seed(args.global_seed * world_size + rank)
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(
            f"Distributed init: world_size={world_size}, "
            f"dataset={args.dataset}, model={args.model}"
        )
    dist.barrier()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run_evaluation(eval_type: str, save_dir: str, args: argparse.Namespace) -> None:
    from .eval.evaluate import run_eval
    logger.info(f"Running evaluation: type={eval_type}")
    try:
        run_eval(eval_type, save_dir, args)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified video QA: inference → evaluation pipeline"
    )

    # Required
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Dataset to evaluate")
    parser.add_argument("--save_dir", required=True, help="Output directory for results")

    # Model
    parser.add_argument("--model", default="llava_ov_7b", help="Model name from registry")
    parser.add_argument("--n_local", type=int, default=15000, help="Local KV cache size")
    parser.add_argument("--retrieve_size", type=int, default=64, help="Retrieved KV chunks")
    parser.add_argument("--retrieve_chunk_size", type=int, default=1, help="Chunk size for retrieval")

    # STC compression
    parser.add_argument("--cache_strategy", default="none", help="STC cache strategy")
    parser.add_argument("--prune_strategy", default="full_tokens", help="STC prune strategy")
    parser.add_argument("--update_token_ratio", type=float, default=0.3)
    parser.add_argument("--token_per_frame", type=int, default=196)
    parser.add_argument("--encode_chunk_size", type=int, default=1)
    parser.add_argument("--cache_interval", type=int, default=2)

    # Data
    parser.add_argument("--sample_fps", type=float, default=0.5, help="Video sampling FPS")
    parser.add_argument("--sample_dir", default="./samples")

    # Profiling
    parser.add_argument("--profile", action="store_true", help="Enable GPU timing/memory profiling")

    # Distributed
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 for matmul")

    # Evaluation control
    parser.add_argument("--skip_eval", action="store_true", help="Skip auto-evaluation after inference")

    # Open-ended eval options
    parser.add_argument("--pred_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--num_tasks", type=int, default=16)

    # StreamingBench options
    parser.add_argument("--streaming_model", default="rekv")
    parser.add_argument("--streaming_task", default=None)

    parser.add_argument("--results_path", default=None)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
