"""Base class for multi-stage dot-product attention."""

from __future__ import annotations

from typing import List, Optional

import torch


class MultiStageDotProductionAttention:
    """Abstract base for multi-stage attention computation.

    Subclasses must implement :meth:`append` to accumulate Q/K/V
    from multiple stages before computing the final attention output.

    Args:
        q_shape: Shape of the query tensor.
        dtype: Data type for the output buffer.
        device: Device for the output buffer.
    """

    def __init__(
        self,
        q_shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        self.end = False
        self.ret = torch.zeros(q_shape, dtype=dtype, device=device)
        self.score_list: List[Optional[torch.Tensor]] = []

    def append(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sliding_window=None,
        complement_sliding_window: bool = False,
        end: bool = False,
        get_score: bool = False,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def get_result(self):
        return self.ret, self.score_list
