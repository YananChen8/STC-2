"""
STC-Pruner adaptation for Qwen2.5-VL model.

This module patches the Qwen2_5_VLModel forward method to apply STC token compression
after visual encoding, before feeding to the LLM.

STC uses Gaussian similarity scores (frame, video, memory) to select important tokens.
"""

from typing import Optional, Union, List, Tuple
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    is_torchdynamo_compiling,
)

from stc.controller import get_config


class ScoreCalculator:
    """Calculates Gaussian similarity scores for token selection."""
    
    @staticmethod
    def gaussian_similarity(
        features: torch.Tensor, 
        target: torch.Tensor, 
        alphas: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Compute multi-scale Gaussian kernel similarity."""
        if alphas is None:
            alphas = [2**k for k in range(-3, 2)]
        diff = features - target  # Broadcasting occurs here
        l2_dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, N]
        scores = sum(torch.exp(-l2_dist_sq / (2 * alpha)) for alpha in alphas)
        return scores

    @staticmethod
    def compute_scores(
        reshaped_features: torch.Tensor, 
        memory_mean: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Frame, Video, and Memory similarity scores."""
        # Normalize features
        features_norm = F.normalize(reshaped_features, dim=-1)
        
        # 1. Frame Mean: mean of current frame
        frame_means = features_norm.mean(dim=1, keepdim=True)  # [Frames, 1, D]
        frame_scores = ScoreCalculator.gaussian_similarity(features_norm, frame_means)
        
        # 2. Video Mean: mean of entire video segment
        video_mean = features_norm.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
        video_scores = ScoreCalculator.gaussian_similarity(features_norm, video_mean)
        
        # 3. Memory Mean: historical memory mean
        memory_mean_norm = F.normalize(memory_mean, dim=-1).view(1, 1, -1)
        memory_scores = ScoreCalculator.gaussian_similarity(features_norm, memory_mean_norm)
        
        return frame_scores, video_scores, memory_scores


class STC_Pruner_Qwen2_5_VL:
    """
    STC Pruner adapted for Qwen2.5-VL with dynamic tokens_per_frame support.
    
    Key differences from base STC_Pruner:
    - Supports dynamic tokens_per_frame computed from grid_thw
    - Uses linear index mapping (flat) for Qwen2.5VL
    - Maintains memory across video segments
    """
    
    def __init__(self):
        self.past_memory_mean_token: List[torch.Tensor] = []
    
    def reset(self):
        """Reset memory for a new video."""
        self.past_memory_mean_token = []
    
    def _update_memory(self, current_features: torch.Tensor) -> torch.Tensor:
        """Update memory with current video segment and return aggregated memory mean."""
        current_chunk_mean = current_features.mean(dim=(0, 1), keepdim=True)  # [1, 1, Dim]
        self.past_memory_mean_token.append(current_chunk_mean)
        history = self.past_memory_mean_token 
        return torch.mean(torch.cat(history, dim=0), dim=0)

    def select_feature_channel(self, tensor: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
        """Select channels with lowest variance (least informative for scoring)."""
        channel_var = tensor.var(dim=0, unbiased=False)
        k = int(channel_var.shape[0] * keep_ratio)
        _, indices = torch.topk(channel_var, k=k, largest=False)
        return tensor[:, indices]

    def _map_flat_indices(
        self, 
        indices_list: List[torch.Tensor], 
        tokens_per_frame: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Map local per-frame indices to global flattened indices."""
        num_frames = len(indices_list)
        offsets = torch.arange(num_frames, device=device) * tokens_per_frame
        global_indices = torch.cat([idx + off for idx, off in zip(indices_list, offsets)])
        return global_indices

    def compress(
        self,
        flattened_features: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        keep_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress video tokens using STC scoring mechanism.
        
        Args:
            flattened_features: Video token embeddings [total_tokens, hidden_dim]
            grid_thw: Grid dimensions [num_videos, 3] (time, height, width)
            spatial_merge_size: Spatial merge size from visual encoder
            keep_ratio: Ratio of tokens to keep per frame (default from config)
            
        Returns:
            compressed_features: Compressed video tokens
            keep_indices: Global indices of kept tokens
        """
        if keep_ratio is None:
            # Get ratio from config (token_per_frame / original_tokens)
            cfg = get_config()
            # For Qwen2.5VL, we use a ratio-based approach
            keep_ratio = float(os.getenv("R_RATIO", "0.25"))
        
        device = flattened_features.device
        
        # Split features by video segments based on grid_thw
        split_sizes = (grid_thw.prod(-1) // spatial_merge_size**2).tolist()
        feature_splits = torch.split(flattened_features, split_sizes)
        
        compressed_chunks: List[torch.Tensor] = []
        kept_indices_list: List[torch.Tensor] = []
        offset = 0
        
        for grid, feat in zip(grid_thw, feature_splits):
            t, h, w = grid.tolist()
            tokens_per_frame = (h * w) // (spatial_merge_size ** 2)
            num_frames = t
            
            if tokens_per_frame <= 0 or feat.numel() == 0:
                # No compression needed
                compressed_chunks.append(feat)
                kept_indices_list.append(
                    torch.arange(feat.shape[0], device=device) + offset
                )
                offset += feat.shape[0]
                continue
            
            # Calculate tokens to keep per frame
            tokens_to_keep = max(1, int(tokens_per_frame * keep_ratio))
            
            # Select low variance channels for scoring
            selected_features = self.select_feature_channel(feat)
            
            # Reshape to [num_frames, tokens_per_frame, channels]
            reshaped_features = selected_features.view(num_frames, tokens_per_frame, -1)
            
            # Update memory and get memory mean
            memory_mean_token = self._update_memory(reshaped_features)
            
            # Compute scores
            frame_score, video_score, memory_score = ScoreCalculator.compute_scores(
                reshaped_features, memory_mean_token
            )
            
            # Combine scores: lower score = more different = more important to keep
            combined_score = memory_score + frame_score
            
            # Select tokens per frame
            frame_kept_indices: List[torch.Tensor] = []
            for i in range(num_frames):
                # Select tokens with lowest similarity (most different/important)
                _, idx = torch.topk(combined_score[i], k=tokens_to_keep, largest=False)
                frame_kept_indices.append(idx.sort().values)
            
            # Map to global indices within this segment
            local_global_indices = self._map_flat_indices(
                frame_kept_indices, tokens_per_frame, device
            )
            
            # Gather compressed features
            compressed_chunks.append(feat[local_global_indices])
            kept_indices_list.append(local_global_indices + offset)
            
            offset += feat.shape[0]
        
        # Concatenate all compressed chunks
        compressed_features = torch.cat(compressed_chunks, dim=0)
        keep_indices = torch.cat(kept_indices_list, dim=0)
        
        # Sort indices for consistent ordering
        keep_indices = torch.sort(keep_indices).values
        
        return compressed_features, keep_indices


def Qwen2_5_VLModel_forward_pruner(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    """Patched forward that enables STC token compression for Qwen2.5-VL."""

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None

    # Process image embeddings (no compression)
    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Process video embeddings with STC compression
    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Pre-compute position ids before pruning so we can safely slice them later.
    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids = position_ids + delta.to(position_ids.device)

    # Check if STC compression should be applied
    compression_on = (
        os.getenv("COMPRESSOR") == "stc_pruner"
        and pixel_values_videos is not None
        and video_grid_thw is not None
        and (past_key_values is None or past_key_values.get_seq_length() == 0)
    )

    if compression_on:
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            compression_on = False

    if compression_on:
        # Initialize or get STC pruner (stored on model for persistence)
        if not hasattr(self, '_stc_pruner'):
            self._stc_pruner = STC_Pruner_Qwen2_5_VL()
        pruner = self._stc_pruner
        pruner.reset()  # Reset memory for new video
        
        merge_size = self.visual.spatial_merge_size
        
        # Apply STC compression
        compressed_video, keep_indices = pruner.compress(
            flattened_features=video_embeds,
            grid_thw=video_grid_thw,
            spatial_merge_size=merge_size,
        )
        
        video_embeds = compressed_video

        # Prune the full sequence based on kept video token indices
        video_token_positions = video_mask[..., 0][0].nonzero(as_tuple=False).squeeze(-1)
        kept_video_positions = video_token_positions[keep_indices]
        all_positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        non_video_positions = all_positions[~video_mask[..., 0][0]]
        keep_token_indices = torch.cat((non_video_positions, kept_video_positions)).sort().values

        def _prune_attention(attn: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if attn is None:
                return None
            if attn.dim() == 2:
                return attn[:, keep_token_indices]
            if attn.dim() == 4:
                return attn[:, :, keep_token_indices, :][:, :, :, keep_token_indices]
            return attn

        inputs_embeds = inputs_embeds[:, keep_token_indices, :]
        if input_ids is not None:
            input_ids = input_ids[:, keep_token_indices]
        attention_mask = _prune_attention(attention_mask)
        position_ids = position_ids[:, :, keep_token_indices]

        if image_mask is not None:
            image_mask = image_mask[:, keep_token_indices, :]
        if video_mask is not None:
            video_mask = video_mask[:, keep_token_indices, :]

        # Re-scatter compressed video embeddings
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds.dtype))

    # Forward through language model
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )

