"""Data loading and splitting utilities."""

from __future__ import annotations

import json
from typing import Any


def load_and_split_anno(
    anno_path: str,
    world_size: int,
    rank: int,
) -> list[dict[str, Any]]:
    """Load annotations and split by rank using strided indexing.

    Each rank receives ``anno[rank::world_size]``, following the
    ``DistributedSampler`` convention for inference.
    """
    with open(anno_path, "r") as f:
        anno = json.load(f)
    return anno[rank::world_size]


def chunk_video(video, chunk_size: int):
    """Yield video chunks of ``chunk_size`` frames."""
    for i in range(0, video.shape[0], chunk_size):
        yield video[i : i + chunk_size]
