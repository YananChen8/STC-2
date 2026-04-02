"""Dataset configurations for all video QA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable dataset evaluation configuration.

    Attributes:
        name: Dataset identifier.
        anno_path: Path to annotation JSON file.
        solver: Solver name registered in :mod:`solver_factory`.
        eval_type: Evaluation type key (see :mod:`eval.evaluate`).
        extra: Extra solver-specific configuration (e.g., benchmark_name).
    """

    name: str
    anno_path: str
    solver: str
    eval_type: str
    extra: dict = field(default_factory=dict)


DATASETS: dict[str, DatasetConfig] = {
    # ---- VideoMME ----
    "videomme": DatasetConfig(
        name="videomme",
        anno_path="data/videomme/random_videomme.json",
        solver="rekv_offline_vqa",
        eval_type="videomme",
    ),
    "videomme_subset": DatasetConfig(
        name="videomme_subset",
        anno_path="data/videomme/videomme_subset.json",
        solver="rekv_offline_vqa",
        eval_type="videomme",
    ),
    # ---- Multiple Choice ----
    "mlvu": DatasetConfig(
        name="mlvu",
        anno_path="data/mlvu/dev_debug_mc.json",
        solver="rekv_offline_vqa",
        eval_type="multiple_choice",
    ),
    "qaego4d": DatasetConfig(
        name="qaego4d",
        anno_path="data/qaego4d/test_mc.json",
        solver="rekv_offline_vqa",
        eval_type="multiple_choice",
    ),
    "cgbench": DatasetConfig(
        name="cgbench",
        anno_path="data/cgbench/full_mc.json",
        solver="rekv_offline_vqa",
        eval_type="multiple_choice",
    ),
    # ---- EgoSchema ----
    "egoschema": DatasetConfig(
        name="egoschema",
        anno_path="data/egoschema/full.json",
        solver="rekv_offline_vqa",
        eval_type="egoschema",
    ),
    "egoschema_subset": DatasetConfig(
        name="egoschema_subset",
        anno_path="data/egoschema_subset/egoschema_subset.json",
        solver="rekv_offline_vqa",
        eval_type="egoschema_subset",
    ),
    # ---- Open-Ended ----
    "activitynet_qa": DatasetConfig(
        name="activitynet_qa",
        anno_path="data/activitynet_qa/test.json",
        solver="rekv_offline_vqa",
        eval_type="open_ended",
    ),
    # ---- Streaming ----
    "rvs_ego": DatasetConfig(
        name="rvs_ego",
        anno_path="data/rvs/ego/ego4d_oe.json",
        solver="rekv_stream_vqa",
        eval_type="open_ended",
    ),
    "rvs_movie": DatasetConfig(
        name="rvs_movie",
        anno_path="data/rvs/movie/movienet_oe.json",
        solver="rekv_stream_vqa",
        eval_type="open_ended",
    ),

    # ==== Online Benchmarks ====

    # ---- OVOBench ----
    "ovobench": DatasetConfig(
        name="ovobench",
        anno_path="data/ovobench/ovo_bench_new.json",
        solver="ovobench_vqa",
        eval_type="ovobench",
        extra={
            "video_dir": "data/ovobench/src_videos",
            "chunked_dir": "data/ovobench/chunked_videos",
            "tasks": ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
        },
    ),

    # ---- StreamingBench ----
    "streamingbench": DatasetConfig(
        name="streamingbench",
        anno_path="data/streamingbench/questions_real.json",
        solver="streamingbench_vqa",
        eval_type="streamingbench",
        extra={
            "benchmark_name": "Streaming",
            "context_time": -1,
            "streaming_task": "real",
        },
    ),
    "streamingbench_sqa": DatasetConfig(
        name="streamingbench_sqa",
        anno_path="data/streamingbench/questions_sqa.json",
        solver="streamingbench_vqa",
        eval_type="streamingbench",
        extra={
            "benchmark_name": "StreamingSQA",
            "context_time": -1,
            "streaming_task": "sqa",
        },
    ),
    "streamingbench_proactive": DatasetConfig(
        name="streamingbench_proactive",
        anno_path="data/streamingbench/questions_proactive.json",
        solver="streamingbench_vqa",
        eval_type="streamingbench",
        extra={
            "benchmark_name": "StreamingProactive",
            "context_time": -1,
            "streaming_task": "proactive",
        },
    ),
}
