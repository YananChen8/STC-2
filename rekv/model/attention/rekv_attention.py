"""ReKV attention forward: manages KV-cache retrieval and sliding-window attention.

This is the core attention mechanism that integrates:
    - Block-wise KV-cache managed by ContextManager
    - Optional retrieved-KV pruning via token filtering strategies
    - Sliding-window + initial-token attention for inference
"""

from __future__ import annotations

import copy
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .dot_production_attention import get_multi_stage_dot_production_attention
from .kv_cache_manager import ContextManager


# ======================================================================
# Token filtering strategies for retrieved KV pruning
# ======================================================================

def _expand_grouped_kv(key: torch.Tensor, query_head_number: int = 28) -> torch.Tensor:
    """Expand grouped KV heads to match query head count.

    Args:
        key: ``(batch, kv_heads, length, dim)``
    Returns:
        ``(batch, length, dim * query_head_number)``
    """
    batch, kv_heads, length, dim = key.shape
    num_group = query_head_number // kv_heads
    return (
        key.view(batch, kv_heads, 1, length, dim)
        .expand(batch, kv_heads, num_group, length, dim)
        .reshape(batch, length, dim * query_head_number)
    )


def _filter_by_cosine_similarity(
    video_tensor: torch.Tensor,
    memory_mean: torch.Tensor,
    token_per_frame: int,
) -> torch.Tensor:
    """Keep tokens with lowest cosine similarity to memory mean (most novel)."""
    _, token_number, channel = video_tensor.shape
    frame_number = token_number // token_per_frame

    flat = video_tensor.view(-1, channel)
    expanded_mean = memory_mean.expand(token_number, -1)
    cosine_sim = F.cosine_similarity(flat, expanded_mean, dim=1)

    kept_indices = []
    for i in range(frame_number):
        start = i * token_per_frame
        frame_sim = cosine_sim[start : start + token_per_frame]
        num_keep = token_per_frame // 2
        _, top_idx = torch.topk(frame_sim, num_keep, largest=False)
        kept_indices.append(top_idx + start)

    return torch.cat(kept_indices, dim=0)


_FILTER_STRATEGIES = {
    "filter_tokens_simple": _filter_by_cosine_similarity,
}


def _apply_retrieved_kv_pruning(
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    past_key_value: ContextManager,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally prune retrieved KV based on environment config."""
    strategy = os.getenv("retrieved_KV_COMPRESSION_STRATEGY", "full_kv")
    if strategy not in _FILTER_STRATEGIES:
        return past_k, past_v

    token_per_frame = int(os.getenv("TOKEN_PER_FRAME", "196"))
    query_head_number = 28

    memory_tokens = past_key_value.origin_block_k[0].data[: past_key_value.length].float()
    memory_mean = memory_tokens.mean(dim=0, keepdim=True)

    grouped_k = _expand_grouped_kv(past_k, query_head_number)
    image_k = grouped_k[:, 13:, :]  # skip text tokens

    indices = _FILTER_STRATEGIES[strategy](image_k, memory_mean, token_per_frame)
    global_indices = indices + 13

    return past_k[:, :, global_indices, :], past_v[:, :, global_indices, :]


# ======================================================================
# Main ReKV attention forward factory
# ======================================================================

def rekv_attention_forward(
    n_local: int,
    n_init: int,
    topk: int,
    chunk_size: int,
    block_size: int,
    max_cached_block: int,
    exc_block_size: int,
    fattn: bool,
    async_global_stream: bool = True,
    pin_memory: bool = False,
    *args,
    **kwargs,
):
    """Factory that creates the ReKV attention forward function.

    Returns a callable that replaces the standard attention forward in each
    transformer layer.
    """
    Attn, _ = get_multi_stage_dot_production_attention(fattn)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        use_cache: bool,
        past_key_value,
        project_q,
        project_k,
        project_v,
        attention_out,
        dim_head: int,
        num_heads: int,
        num_heads_kv: int,
    ):
        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)
        assert use_cache

        # --- Project QKV ---
        h_q = project_q(query).view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()
        h_k = project_k(key_value).view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()
        h_v = project_v(key_value).view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()

        # Ensure position bias is on the correct device
        if position_bias._cos_cached is not None and position_bias._cos_cached.device != h_q.device:
            position_bias = copy.deepcopy(position_bias)
            if position_bias.inv_freq.device != h_q.device:
                position_bias.inv_freq = position_bias.inv_freq.to(h_q.device)
            if position_bias._cos_cached is not None:
                position_bias._cos_cached = position_bias._cos_cached.to(h_q.device)
            if position_bias._sin_cached is not None:
                position_bias._sin_cached = position_bias._sin_cached.to(h_q.device)

        # --- Initialize ContextManager on first call ---
        if past_key_value is None:
            past_key_value = ContextManager(
                position_bias,
                n_init, n_local,
                block_size, max_cached_block, topk, chunk_size, exc_block_size,
                fattn, async_global_stream, pin_memory,
            )

        local_q, local_k, local_v = h_q, h_k, h_v
        global_q, global_k, global_v = h_q, h_k, h_v

        # =============================================================
        # Question-answering mode (retrieval / sliding-window)
        # =============================================================
        if type(past_key_value) is not ContextManager or past_key_value.to_retrieve:
            if type(past_key_value) is ContextManager:
                # Retrieval mode
                if past_key_value.retrieved_block_indices is None:
                    past_k, past_v = past_key_value.get_retrieved_kv(global_q)
                else:
                    past_k, past_v = past_key_value.get_retrieved_kv()

                # Optional pruning of retrieved KV
                if os.getenv("PRUNE_RETIREVED_KV", "no") == "yes":
                    past_k, past_v = _apply_retrieved_kv_pruning(
                        past_k, past_v, past_key_value
                    )

                update_kv_cache = False
            else:
                # Standard sliding-window
                past_k = past_key_value[0]
                past_v = past_key_value[1]
                update_kv_cache = True

            # Concatenate with current KV
            h_k = torch.cat([past_k, h_k], dim=-2)
            h_v = torch.cat([past_v, h_v], dim=-2)
            len_k += past_k.shape[2]

            # Update KV cache
            if update_kv_cache:
                if len_k <= n_local + n_init:
                    h_k_cache, h_v_cache = h_k, h_v
                else:
                    h_k_cache = torch.cat([
                        h_k[:, :, :n_init, :],
                        h_k[:, :, max(0, h_k.size(-2) - n_local):, :],
                    ], dim=2)
                    h_v_cache = torch.cat([
                        h_v[:, :, :n_init, :],
                        h_v[:, :, max(0, h_k.size(-2) - n_local):, :],
                    ], dim=2)
                current_key_value = (h_k_cache, h_v_cache)
            else:
                current_key_value = (past_k, past_v)

            # Apply RoPE to local window
            h_q_, h_k_, h_v_ = h_q, h_k, h_v
            if len_q + n_local < h_k_.size(-2):
                h_k_ = h_k_[:, :, h_k_.size(-2) - len_q - n_local:, :]
                h_v_ = h_v_[:, :, h_v_.size(-2) - len_q - n_local:, :]

            local_h_q, local_h_k = position_bias(h_q_, h_k_)
            local_h_v = h_v_

            # Apply RoPE to initial tokens
            if len_k > n_local:
                init_h_q = position_bias.apply_rotary_pos_emb_one_angle(h_q, n_local)
                init_h_k = h_k[:, :, :n_init, :].contiguous()
                init_h_v = h_v[:, :, :n_init, :].contiguous()
            else:
                init_h_q = h_q
                init_h_k = torch.empty(batch_size, num_heads_kv, 0, dim_head, device=h_k.device, dtype=h_k.dtype)
                init_h_v = torch.empty(batch_size, num_heads_kv, 0, dim_head, device=h_v.device, dtype=h_v.dtype)

            # Multi-stage sliding-window attention
            attn = Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
            attn.append(local_h_q, local_h_k, local_h_v, sliding_window=n_local)
            attn.append(
                init_h_q, init_h_k, init_h_v,
                end=True,
                sliding_window=(len_k - len_q, n_local),
                complement_sliding_window=True,
            )
            score, _ = attn.get_result()

            score = score.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            score = score.reshape(batch_size, len_q, num_heads * dim_head)
            return attention_out(score), current_key_value

        # =============================================================
        # Video encoding mode (managed by ContextManager)
        # =============================================================
        else:
            o = past_key_value.append(
                local_q, local_k, local_v,
                global_q, global_k, global_v,
            )
            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
            o = o.reshape(batch_size, len_q, dim_head * num_heads)
            return attention_out(o), past_key_value

    return forward
