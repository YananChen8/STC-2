"""Model loading and device management utilities."""

from __future__ import annotations

import torch
from logzero import logger

from model import llava_onevision_rekv, longva_rekv, video_llava_rekv


MODEL_REGISTRY: dict[str, dict] = {
    "llava_ov_0.5b": {
        "load_func": llava_onevision_rekv.load_model,
        "model_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    },
    "llava_ov_7b": {
        "load_func": llava_onevision_rekv.load_model,
        "model_path": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    },
    "video_llava_7b": {
        "load_func": video_llava_rekv.load_model,
        "model_path": "LanguageBind/Video-LLaVA-7B-hf",
    },
    "longva_7b": {
        "load_func": longva_rekv.load_model,
        "model_path": "model_zoo/LongVA-7B",
    },
}


def load_model(
    model_name: str,
    device=None,
    n_local: int = 15000,
    topk: int = 64,
    chunk_size: int = 1,
):
    """Load a video QA model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}"
        )

    entry = MODEL_REGISTRY[model_name]
    return entry["load_func"](
        model_path=entry["model_path"],
        device=device,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )


def get_device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
