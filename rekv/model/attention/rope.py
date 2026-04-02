"""Rotary Position Embedding (RoPE) for transformer attention.

Based on the RoFormer approach (https://arxiv.org/abs/2104.09864).
Supports 2-D, 3-D, and 4-D input tensors with cached cos/sin tables.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


class RotaryEmbeddingESM(torch.nn.Module):
    """Rotary position embedding with cached cos/sin tables.

    Args:
        dim: Embedding dimension (must be even).
        base: Base for frequency computation.
        distance_scale: Scaling factor for position indices.
    """

    def __init__(
        self,
        dim: int,
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        length: int,
        right: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to *x* using a slice ``[right-length : right]``."""
        dtype = x.dtype
        cos = _slice_along_seq(cos, right - length, right)
        sin = _slice_along_seq(sin, right - length, right)
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def apply_rotary_pos_emb_one_angle(
        self,
        x: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        """Apply RoPE at a single position *index*."""
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(index, x.device)
        cos = _slice_along_seq(cos, index - 1, index)
        sin = _slice_along_seq(sin, index - 1, index)
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _update_cos_sin_tables(
        self, x: torch.Tensor, seq_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            emb = self._compute_embedding(seq_len, x.device)
            self._cos_cached, self._sin_cached = _reshape_for_dim(
                emb.cos(), emb.sin(), x.dim()
            )
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(
        self,
        seq_len: int,
        device: torch.device,
        dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._seq_len_cached:
            if dim is None:
                assert self._cos_cached is not None
                dim = self._cos_cached.dim()
            self._seq_len_cached = seq_len
            emb = self._compute_embedding(seq_len, device)
            self._cos_cached, self._sin_cached = _reshape_for_dim(
                emb.cos(), emb.sin(), dim
            )
        return self._cos_cached, self._sin_cached

    def _compute_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t * self.distance_scale, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_dim: int = -2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
        )


# ======================================================================
# Module-level helpers
# ======================================================================

def _slice_along_seq(
    tensor: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """Slice the sequence dimension of a 2/3/4-D cos/sin table."""
    if tensor.dim() == 2:
        return tensor[start:end, :]
    elif tensor.dim() == 3:
        return tensor[:, start:end, :]
    else:  # dim == 4
        return tensor[:, :, start:end, :]


def _reshape_for_dim(
    cos: torch.Tensor,
    sin: torch.Tensor,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reshape 2-D embeddings ``[seq, d]`` to match a target tensor dimensionality."""
    if dim == 2:
        return cos, sin
    elif dim == 3:
        return cos[None, :, :], sin[None, :, :]
    else:  # dim == 4
        return cos[None, None, :, :], sin[None, None, :, :]
