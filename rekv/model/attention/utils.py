"""Utility functions for attention operations."""

from __future__ import annotations

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query head count.

    Equivalent to ``torch.repeat_interleave(x, dim=1, repeats=n_rep)``.

    Args:
        hidden_states: ``(batch, n_kv_heads, seqlen, head_dim)``
        n_rep: Number of times to repeat.

    Returns:
        ``(batch, n_attention_heads, seqlen, head_dim)``
    """
    if n_rep == 1:
        return hidden_states
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return expanded.reshape(batch, n_kv_heads * n_rep, slen, head_dim)
