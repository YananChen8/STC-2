"""
STC-Cacher adaptation for Qwen2.5-VL Vision Encoder.

This module provides selective recomputation caching for Qwen2.5VL's ViT encoder.
Since Qwen2.5VL processes all frames in parallel, we simulate streaming by:
1. Splitting frames into chunks based on cache_interval
2. Processing each chunk sequentially through the ViT
3. Using reference frame caching + selective token recomputation

Key Architecture:
- Qwen2.5VL uses Qwen2VLVisionBlock which contains:
  - norm1, attn (Qwen2VLVisionSdpaAttention), norm2, mlp
- We replace the entire ViT forward to enable chunk-based processing
"""

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import types
import os

from stc.vit_with_cacher.utils import STC_CACHER
from stc.controller import get_config


def register_cache_for_qwen2_5_vl(model: nn.Module) -> None:
    """
    Register STC caching mechanism for Qwen2.5VL vision encoder.
    
    This patches the visual model's forward to enable streaming-like processing
    with selective recomputation caching.
    
    Args:
        model: The Qwen2_5_VLForConditionalGeneration model
    """
    visual_model = model.model.visual
    
    # Store original forward
    visual_model._original_forward = visual_model.forward
    
    # Replace with cached version
    visual_model.forward = types.MethodType(
        qwen2_5_vl_visual_forward_with_cache, 
        visual_model
    )
    
    # Register block-level caching on each vision block
    for layer_idx, block in enumerate(visual_model.blocks):
        block._layer_idx = layer_idx
        block._original_forward = block.forward


def qwen2_5_vl_visual_forward_with_cache(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """
    Cached forward for Qwen2.5VL vision encoder with streaming simulation.
    
    This processes video frames in chunks, applying reference frame caching:
    - Every cache_interval chunks: full computation, save reference
    - Other chunks: selective recomputation based on K similarity
    
    Args:
        hidden_states: Raw pixel values (NOT patch embeddings yet!)
        grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
        
    Returns:
        Merged visual features after processing
    """
    # IMPORTANT: First apply patch embedding to convert pixel values to patches
    # This is the step that was missing before!
    hidden_states = self.patch_embed(hidden_states)
    
    config = get_config()
    cache_interval = config.cache.cache_interval
    update_token_ratio = config.cache.update_token_ratio
    
    # Get spatial dimensions
    spatial_merge_size = self.spatial_merge_size
    
    # Calculate tokens per frame for each video (after patch embedding)
    tokens_per_frame_list = (grid_thw[:, 1] * grid_thw[:, 2]).tolist()  # h * w per video
    num_frames_per_video = grid_thw[:, 0].tolist()
    
    # Compute rotary position embeddings
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    
    # Initialize STC cacher
    cacher = STC_CACHER.new_instance(
        chunk_idx=0,
        update_token_ratio=update_token_ratio,
        similarity_threshold=config.cache.similarity_threshold,
    )
    
    # Process frames in chunks for each video
    all_outputs = []
    global_offset = 0
    
    for vid_idx, (num_frames, tokens_per_frame) in enumerate(zip(num_frames_per_video, tokens_per_frame_list)):
        video_tokens = num_frames * tokens_per_frame
        video_hidden = hidden_states[global_offset:global_offset + video_tokens]
        video_pos = rotary_pos_emb[global_offset:global_offset + video_tokens]
        
        # Build cu_seqlens for this video (for flash attention)
        video_cu_seqlens = torch.arange(
            0, (num_frames + 1) * tokens_per_frame, tokens_per_frame,
            device=hidden_states.device, dtype=torch.int32
        )
        
        # Process this video's frames in chunks
        video_output = _process_video_with_streaming_cache(
            self,
            video_hidden,
            video_pos,
            num_frames,
            tokens_per_frame,
            cache_interval,
            update_token_ratio,
            cacher,
        )
        
        all_outputs.append(video_output)
        global_offset += video_tokens
    
    # Concatenate all video outputs
    output = torch.cat(all_outputs, dim=0)
    
    # Apply the merger (spatial merging)
    output = self.merger(output)
    
    return output


def _process_video_with_streaming_cache(
    visual_model: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    cache_interval: int,
    update_token_ratio: float,
    cacher: STC_CACHER,
) -> torch.Tensor:
    """
    Process a single video's frames with streaming cache simulation.
    
    Splits frames into chunks and processes each chunk with caching logic.
    """
    device = hidden_states.device
    hidden_dim = hidden_states.shape[-1]
    
    # Reshape to [num_frames, tokens_per_frame, hidden_dim]
    hidden_states = hidden_states.view(num_frames, tokens_per_frame, hidden_dim)
    rotary_pos_emb = rotary_pos_emb.view(num_frames, tokens_per_frame, -1)
    
    # Split into chunks based on cache_interval
    num_chunks = (num_frames + cache_interval - 1) // cache_interval
    
    all_chunk_outputs = []
    
    log_stats = os.getenv("STC_LOG_STATS", "0") == "1"
    is_rank0 = True
    if dist.is_available() and dist.is_initialized():
        is_rank0 = dist.get_rank() == 0

    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * cache_interval
        end_frame = min((chunk_idx + 1) * cache_interval, num_frames)
        
        chunk_hidden = hidden_states[start_frame:end_frame]  # [chunk_size, tokens, dim]
        chunk_pos = rotary_pos_emb[start_frame:end_frame]
        chunk_size = end_frame - start_frame
        
        # Update cacher state
        cacher.chunk_idx = chunk_idx
        
        # Flatten for block processing: [chunk_size * tokens_per_frame, hidden_dim]
        chunk_hidden_flat = chunk_hidden.reshape(-1, hidden_dim)
        chunk_pos_flat = chunk_pos.reshape(-1, chunk_pos.shape[-1])
        
        # Build cu_seqlens for this chunk (for flash attention within chunk)
        chunk_cu_seqlens = torch.arange(
            0, (chunk_size + 1) * tokens_per_frame, tokens_per_frame,
            device=device, dtype=torch.int32
        )
        
        # Determine if this is a reference chunk
        is_reference_chunk = (chunk_idx % cache_interval == 0)

        if log_stats and is_rank0:
            if is_reference_chunk:
                update_per_frame = tokens_per_frame
            else:
                update_per_frame = max(1, int(tokens_per_frame * update_token_ratio))
            update_total = update_per_frame * chunk_size
            denom = max(tokens_per_frame, 1)
            reuse_ratio = 1.0 - (update_per_frame / denom)
            print(
                f"[STC-Cacher][Qwen2.5-VL] chunk={chunk_idx} "
                f"ref={is_reference_chunk} cache_interval={cache_interval} "
                f"frames={chunk_size} tokens/frame={tokens_per_frame} "
                f"update/frame={update_per_frame} update/total={update_total} "
                f"reuse_attn={reuse_ratio:.2f} reuse_mlp={reuse_ratio:.2f}"
            )
        
        # Process through vision blocks with caching
        chunk_output = _process_chunk_through_blocks(
            visual_model,
            chunk_hidden_flat,
            chunk_pos_flat,
            chunk_cu_seqlens,
            tokens_per_frame,
            chunk_size,
            is_reference_chunk,
            update_token_ratio,
        )
        
        all_chunk_outputs.append(chunk_output)
    
    # Concatenate all chunks
    output = torch.cat(all_chunk_outputs, dim=0)
    
    return output


def _process_chunk_through_blocks(
    visual_model: nn.Module,
    hidden_states: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tokens_per_frame: int,
    num_frames: int,
    is_reference_chunk: bool,
    update_token_ratio: float,
) -> torch.Tensor:
    """
    Process a chunk through all vision blocks with optional caching.
    
    For reference chunks: Full computation, save K/V/AttnOut/MLPOut
    For other chunks: Selective recomputation based on K similarity
    """
    # Compute position embeddings (cos, sin) from rotary embeddings
    # Qwen2.5VL uses rotary position embeddings for vision
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    
    # Process through each block
    for block_idx, block in enumerate(visual_model.blocks):
        if is_reference_chunk:
            # Full computation with caching
            hidden_states = _block_forward_full_with_cache(
                block,
                hidden_states,
                cu_seqlens,
                position_embeddings,
                tokens_per_frame,
                num_frames,
                block_idx,
            )
        else:
            # Selective recomputation using cached reference
            hidden_states = _block_forward_selective_recompute(
                block,
                hidden_states,
                cu_seqlens,
                position_embeddings,
                tokens_per_frame,
                num_frames,
                block_idx,
                update_token_ratio,
            )
    
    return hidden_states


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to vision Q/K tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()

    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k_rotated = torch.cat((-k2, k1), dim=-1)

    q_embed = (q * cos) + (q_rotated * sin)
    k_embed = (k * cos) + (k_rotated * sin)

    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _apply_attn_out_proj(attn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply attention output projection with robust attribute fallback."""
    if hasattr(attn, "o_proj"):
        return attn.o_proj(x)
    if hasattr(attn, "out_proj"):
        return attn.out_proj(x)
    if hasattr(attn, "proj"):
        return attn.proj(x)
    return x


def _block_forward_full_with_cache(
    block: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    tokens_per_frame: int,
    num_frames: int,
    block_idx: int,
) -> torch.Tensor:
    """
    Full forward through a vision block, caching reference outputs.
    """
    # Qwen2VLVisionBlock structure: norm1 -> attn -> residual -> norm2 -> mlp -> residual
    hidden_dim = hidden_states.shape[-1]
    
    residual = hidden_states
    hidden_states_norm = block.norm1(hidden_states)
    
    # Attention forward
    attn_output = block.attn(
        hidden_states_norm,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    
    hidden_states = residual + attn_output
    
    # MLP forward
    residual2 = hidden_states
    hidden_states_norm2 = block.norm2(hidden_states)
    mlp_output = block.mlp(hidden_states_norm2)
    hidden_states = residual2 + mlp_output
    
    # Cache reference frame outputs (last frame of this chunk)
    total_tokens = num_frames * tokens_per_frame
    if hidden_states.shape[0] == total_tokens:
        # Reshape to get per-frame view
        reshaped_norm = hidden_states_norm.view(num_frames, tokens_per_frame, hidden_dim)
        reshaped_attn = attn_output.view(num_frames, tokens_per_frame, hidden_dim)
        reshaped_mlp = mlp_output.view(num_frames, tokens_per_frame, hidden_dim)
        
        # Store reference (last frame's outputs for caching)
        block._ref_hidden_norm = reshaped_norm[-1].detach().clone()  # [tokens_per_frame, hidden_dim]
        block._ref_attn_out = reshaped_attn[-1].detach().clone()
        block._ref_mlp_out = reshaped_mlp[-1].detach().clone()
        
        # Cache reference K for similarity comparison
        # Qwen2VL attention uses qkv projection that outputs [seq, 3*hidden_dim]
        with torch.no_grad():
            qkv = block.attn.qkv(block._ref_hidden_norm)
            # qkv shape: [tokens_per_frame, 3 * hidden_dim]
            # Split: Q, K, V each have hidden_dim
            block._ref_key = qkv[:, hidden_dim:2*hidden_dim].detach().clone()
            block._ref_value = qkv[:, 2*hidden_dim:3*hidden_dim].detach().clone()
    
    return hidden_states


def _block_forward_selective_recompute(
    block: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    tokens_per_frame: int,
    num_frames: int,
    block_idx: int,
    update_token_ratio: float,
) -> torch.Tensor:
    """
    Selective recomputation forward using cached reference.
    
    Only recomputes tokens with low K similarity to reference.
    """
    device = hidden_states.device
    hidden_dim = hidden_states.shape[-1]
    total_tokens = num_frames * tokens_per_frame
    
    # Check if we have cached reference
    if (
        not hasattr(block, "_ref_key")
        or block._ref_key is None
        or not hasattr(block, "_ref_value")
        or block._ref_value is None
    ):
        # No cache, do full computation
        return _block_forward_full_with_cache(
            block, hidden_states, cu_seqlens, position_embeddings,
            tokens_per_frame, num_frames, block_idx
        )
    
    residual = hidden_states
    hidden_states_norm = block.norm1(hidden_states)
    
    attn = block.attn
    num_heads = attn.num_heads
    head_dim = hidden_dim // num_heads

    # Compute Q/K/V once for similarity + selective attention
    qkv_full = attn.qkv(hidden_states_norm)
    # qkv shape: [total_tokens, 3 * hidden_dim]
    qkv_full = qkv_full.view(total_tokens, 3, num_heads, head_dim)
    q, k, v = qkv_full.unbind(dim=1)  # [total_tokens, num_heads, head_dim]

    # Compute current K for all tokens to determine which need updating
    current_k = k.reshape(total_tokens, hidden_dim)  # [total_tokens, hidden_dim]
    
    # Reshape for per-frame similarity computation
    current_k_reshaped = current_k.view(num_frames, tokens_per_frame, hidden_dim)
    
    # Compute similarity with reference K
    ref_key = block._ref_key  # [tokens_per_frame, hidden_dim]
    similarity = F.cosine_similarity(
        current_k_reshaped,
        ref_key.unsqueeze(0),
        dim=-1
    )  # [num_frames, tokens_per_frame]
    
    # Select tokens to update (lowest similarity = most different from reference)
    num_update = max(1, int(tokens_per_frame * update_token_ratio))
    update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices
    
    # Apply rotary position embeddings to Q/K for attention
    cos, sin = position_embeddings
    q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

    q = q.view(num_frames, tokens_per_frame, num_heads, head_dim).transpose(1, 2)
    k = k.view(num_frames, tokens_per_frame, num_heads, head_dim).transpose(1, 2)

    # Build value states from reference, update only selected tokens
    ref_value = block._ref_value  # [tokens_per_frame, hidden_dim]
    value_states_full = ref_value.unsqueeze(0).expand(num_frames, -1, -1).clone()
    update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    current_v = v.reshape(total_tokens, hidden_dim).view(num_frames, tokens_per_frame, hidden_dim)
    v_selected = current_v.gather(1, update_idx_expanded)
    value_states_full.scatter_(1, update_idx_expanded, v_selected)
    value_states_full = value_states_full.view(num_frames, tokens_per_frame, num_heads, head_dim).transpose(1, 2)

    # Gather selected queries for attention
    update_idx_for_attn = update_indices.unsqueeze(1).unsqueeze(-1).expand(
        num_frames, num_heads, num_update, head_dim
    )
    q_selected = q.gather(2, update_idx_for_attn)

    # Compute attention for selected tokens only
    attn_output_selected = F.scaled_dot_product_attention(
        q_selected,
        k,
        value_states_full,
        dropout_p=0.0,
        is_causal=False,
    )
    attn_output_selected = (
        attn_output_selected.transpose(1, 2).reshape(num_frames, num_update, hidden_dim)
    )
    attn_output_selected = _apply_attn_out_proj(attn, attn_output_selected)

    # Scatter selected attention outputs into cached reference
    attn_output_full = block._ref_attn_out.unsqueeze(0).expand(num_frames, -1, -1).clone()
    attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)

    hidden_states = residual + attn_output_full.reshape(-1, hidden_dim)
    
    # MLP with selective recomputation
    residual2 = hidden_states
    hidden_states_norm2 = block.norm2(hidden_states)
    
    reshaped_norm2 = hidden_states_norm2.view(num_frames, tokens_per_frame, hidden_dim)

    # Initialize MLP output from reference
    ref_mlp_out = block._ref_mlp_out  # [tokens_per_frame, hidden_dim]
    mlp_output = ref_mlp_out.unsqueeze(0).expand(num_frames, -1, -1).clone()

    # Compute MLP only for selected tokens
    tokens_to_update = reshaped_norm2.gather(1, update_idx_expanded)  # [num_frames, num_update, hidden_dim]

    # Flatten, compute MLP, reshape back
    mlp_selected = block.mlp(tokens_to_update.reshape(-1, hidden_dim))
    mlp_selected = mlp_selected.view(num_frames, num_update, hidden_dim)

    # Scatter update
    mlp_output.scatter_(1, update_idx_expanded, mlp_selected)

    # Flatten and apply residual
    mlp_output_flat = mlp_output.reshape(-1, hidden_dim)
    hidden_states = residual2 + mlp_output_flat
    
    return hidden_states


def unregister_cache_for_qwen2_5_vl(model: nn.Module) -> None:
    """
    Remove STC caching from Qwen2.5VL vision encoder.
    Restores original forward methods.
    """
    visual_model = model.model.visual
    
    if hasattr(visual_model, '_original_forward'):
        visual_model.forward = visual_model._original_forward
        delattr(visual_model, '_original_forward')
    
    for block in visual_model.blocks:
        if hasattr(block, '_original_forward'):
            block.forward = block._original_forward
            delattr(block, '_original_forward')
        
        # Clean up cached references
        for attr in ['_ref_hidden_norm', '_ref_attn_out', '_ref_mlp_out', '_ref_key', '_ref_value', '_layer_idx']:
            if hasattr(block, attr):
                delattr(block, attr)
