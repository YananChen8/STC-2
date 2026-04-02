"""Factory for creating VQA solver instances."""

from __future__ import annotations

from logzero import logger


def create_solver(solver_name: str, model, processor, args):
    """Instantiate a VQA solver by name.

    Uses lazy imports to avoid circular dependencies.
    """
    from .rekv_offline import ReKVOfflineVQA
    from .rekv_stream import ReKVStreamVQA
    from .ovobench import OVOBenchVQA
    from .streamingbench import StreamingBenchVQA

    registry = {
        "rekv_offline_vqa": ReKVOfflineVQA,
        "rekv_stream_vqa": ReKVStreamVQA,
        "ovobench_vqa": OVOBenchVQA,
        "streamingbench_vqa": StreamingBenchVQA,
    }

    if solver_name not in registry:
        logger.warning(f"Unknown solver '{solver_name}', falling back to rekv_offline_vqa")
        solver_name = "rekv_offline_vqa"

    return registry[solver_name](model, processor, args)
