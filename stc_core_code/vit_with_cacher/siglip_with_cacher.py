import torch
from torch import nn
from typing import Optional, Tuple, Union,List

import torch.nn.functional as F
import os

from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipEncoderLayer, SiglipConfig
from transformers.modeling_outputs import BaseModelOutput
from stc.vit_with_cacher.utils import *

import time
import math
import torch
from typing import Optional, Tuple
from collections import defaultdict
from logzero import logger  # 添加这行
from stc.controller import get_config
import types
import torch.distributed as dist





def register_cache_by_key_Siglip(vision_tower: nn.Module) -> None:
    for layer in vision_tower.vision_model.encoder.layers:
        setattr(layer, "_old_forward", layer.forward)
        layer.forward = types.MethodType(forward_with_selective_key_recompute, layer)
        layer.new_attn= types.MethodType(new_siglip_sdpa_attn_forward, layer)
        
       
def register_cache_by_key_CLIP(vision_tower: nn.Module) -> None:
    for layer in vision_tower.vision_model.encoder.layers:
        setattr(layer, "_old_forward", layer.forward)
        layer.forward = types.MethodType(forward_with_selective_key_recompute_clip, layer)
        layer.new_attn= types.MethodType(new_siglip_sdpa_attn_forward, layer)


def register_cache_by_threshold_Siglip(vision_tower: nn.Module) -> None:
    """
    使用基于cosine similarity阈值的选择性重计算策略注册Siglip模型。
    与register_cache_by_key_Siglip的区别是使用阈值动态选择token数量，而非固定百分比。
    """
    for layer in vision_tower.vision_model.encoder.layers:
        setattr(layer, "_old_forward", layer.forward)
        layer.forward = types.MethodType(forward_with_selective_key_recompute_threshold, layer)
        layer.new_attn = types.MethodType(new_siglip_sdpa_attn_forward, layer)


def register_frame_by_frame_cache_Siglip(vision_tower: nn.Module) -> None:
    """
    使用逐帧滑动缓存策略注册Siglip模型。
    与register_cache_by_key_Siglip的区别：
    - 原策略：每cache_interval帧完整计算一帧，其余帧使用固定reference
    - 本策略：每帧都与前一帧比较相似度，每帧完成后更新reference（滑动窗口）
    """
    for layer in vision_tower.vision_model.encoder.layers:
        setattr(layer, "_old_forward", layer.forward)
        layer.forward = types.MethodType(forward_with_frame_by_frame_cache, layer)
        layer.new_attn = types.MethodType(new_siglip_sdpa_attn_forward, layer)


def register_layer0_dynamic_detection_Siglip(vision_tower: nn.Module) -> None:
    """
    使用第0层动态token检测策略注册Siglip模型。
    核心特点：
    - 仅在第0层计算动态token索引，后续层复用这些索引
    - 注意力计算时K和V完全使用参考帧，Q使用当前帧的动态tokens
    - 中间层只处理动态tokens，仅在最后一层恢复完整tensor
    """
    layers = vision_tower.vision_model.encoder.layers
    num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        setattr(layer, "_old_forward", layer.forward)
        setattr(layer, "layer_idx", layer_idx)  # 设置层索引，用于区分第0层和后续层
        setattr(layer, "num_layers", num_layers)  # 设置总层数，用于识别最后一层
        layer.forward = types.MethodType(forward_with_layer0_dynamic_detection, layer)
        layer.new_attn = types.MethodType(new_siglip_sdpa_attn_forward, layer)

        
def forward_with_selective_key_recompute(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):

    
    cache2 = STC_CACHER()
    chunk_idx = cache2.chunk_idx
    cache_interval=get_config().cache.cache_interval
    update_cache = (chunk_idx % cache_interval == 0)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if update_cache :
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        # 注意：保存的是projection后的张量，shape为[T, C]
        
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]  # 修复这里！

        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 奇数chunk：基于Key相似度的选择性重计算 ==========
    else:
        cache2 = STC_CACHER()
        update_token_ratio = cache2.update_token_ratio  
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：基于Key识别需要更新的token ==========
        # 计算当前帧的Key向量（用于相似度计算）
        key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
        
        # Reference frame的Key
        ref_key_for_sim = self.reference_frame_key  # [T, C]
        # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
        similarity = torch.nn.functional.cosine_similarity(
            key_states_full,
            ref_key_for_sim.unsqueeze(0),
            dim=-1
        )

        num_update = int(seq_len * update_token_ratio)
        num_update = max(1, min(num_update, seq_len))
        
        # 对每一帧，选择相似度最低的token索引
        update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
    
        # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        rank = dist.get_rank()
        
        if rank == 0:
    
            logger.info(f"SigLIP | Vocab size: 729, Tokens to update: {tokens_to_update.shape[1]}")
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs

def forward_with_selective_key_recompute_threshold(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    基于cosine similarity阈值的选择性重计算策略：
    - 与forward_with_selective_key_recompute的区别是使用阈值而非百分比来动态选择token数量
    - 偶数chunk：完整计算，保存最后一帧的K, V, AttnOut, MLPOut
    - 奇数chunk：选择cosine similarity低于阈值的token进行重计算
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHER()
    chunk_idx = cache2.chunk_idx
    cache_interval = get_config().cache.cache_interval
    update_cache = (chunk_idx % cache_interval == 0)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if update_cache:
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        # 注意：保存的是projection后的张量，shape为[T, C]
        
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]

        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 奇数chunk：基于cosine similarity阈值的选择性重计算 ==========
    else:
        cache2 = STC_CACHER()
        similarity_threshold = cache2.similarity_threshold  # 使用阈值而非百分比
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：基于Key识别需要更新的token（使用阈值） ==========
        # 计算当前帧的Key向量（用于相似度计算）
        key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
        
        # Reference frame的Key
        ref_key_for_sim = self.reference_frame_key  # [T, C]
        # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
        similarity = torch.nn.functional.cosine_similarity(
            key_states_full,
            ref_key_for_sim.unsqueeze(0),
            dim=-1
        )

        # 使用阈值选择：similarity < threshold 的token需要更新
        # 为了保持batch操作，计算每帧低于阈值的token数量，取最大值
        below_threshold_mask = similarity < similarity_threshold  # [F, T]
        num_below_threshold_per_frame = below_threshold_mask.sum(dim=1)  # [F]
        num_update = int(num_below_threshold_per_frame.max().item())
        
        # 确保至少更新1个token，最多更新所有token
        num_update = max(1, min(num_update, seq_len))
        
        # 对每一帧，选择相似度最低的token索引（保证低于阈值的一定被选中）
        update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
    
        # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        rank = dist.get_rank()
        
        if rank == 0:
            avg_tokens_below_threshold = num_below_threshold_per_frame.float().mean().item()
            logger.info(f"SigLIP Threshold | Vocab size: {seq_len}, Threshold: {similarity_threshold:.3f}, "
                       f"Avg tokens below threshold: {avg_tokens_below_threshold:.1f}, Tokens to update: {num_update}")
        
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs


def forward_with_layer0_dynamic_detection(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    第0层动态token检测策略（优化版）：
    - 仅在第0层计算动态token索引（基于Key相似度），后续层复用这些索引
    - 注意力计算时K和V完全使用参考帧，Q使用当前帧的动态tokens
    - 中间层只处理动态tokens [F, num_update, C]，不做恢复
    - 仅在最后一层输出前恢复到完整tensor [F, T, C]
    
    数据流：
    - Layer 0: [F, T, C] -> 提取动态tokens -> [F, num_update, C]
    - Layer 1~N-2: [F, num_update, C] -> 处理 -> [F, num_update, C]
    - Layer N-1: [F, num_update, C] -> 处理 + 恢复 -> [F, T, C]
    
    Args:
        hidden_states: 
            - 更新chunk时: [F, T, C]
            - 非更新chunk第0层: [F, T, C]
            - 非更新chunk中间层: [F, num_update, C]
    """
    
    cache2 = STC_CACHER()
    chunk_idx = cache2.chunk_idx
    cache_interval = get_config().cache.cache_interval
    update_cache = (chunk_idx % cache_interval == 0)
    layer_idx = getattr(self, 'layer_idx', 0)  # 获取当前层索引
    num_layers = getattr(self, 'num_layers', 27)  # 获取总层数
    is_last_layer = (layer_idx == num_layers - 1)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if update_cache:
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V作为reference
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]
        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的完整输出（用于最后一层恢复静态tokens）
        with torch.no_grad():
            self.reference_frame_output = hidden_states[-1].detach()  # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 非更新chunk：基于第0层动态token检测的选择性重计算 ==========
    else:
        update_token_ratio = cache2.update_token_ratio
        batch_size = hidden_states.shape[0]
        embed_dim = hidden_states.shape[-1]
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 第0层：计算动态token索引并提取动态tokens ==========
        if layer_idx == 0:
            # 输入是完整tensor [F, T, C]
            seq_len = hidden_states.shape[1]
            
            residual1 = hidden_states
            hidden_states_ln1 = self.layer_norm1(hidden_states)
            
            # 计算当前帧的Key向量（用于相似度计算）
            key_states_for_sim = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
            
            # Reference frame的Key
            ref_key_for_sim = self.reference_frame_key  # [T, C]
            
            # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
            similarity = torch.nn.functional.cosine_similarity(
                key_states_for_sim,
                ref_key_for_sim.unsqueeze(0),
                dim=-1
            )
            
            num_update = int(seq_len * update_token_ratio)
            num_update = max(1, min(num_update, seq_len))
            
            # 对每一帧，选择相似度最低的token索引（变化最大的tokens）
            update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
            
            # 缓存信息供后续层使用
            cache2.update_indices = update_indices
            cache2.original_seq_len = seq_len  # 保存原始序列长度用于最后一层恢复
            
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                logger.info(f"SigLIP Layer0 Dynamic Detection | Vocab size: {seq_len}, "
                           f"Dynamic tokens: {num_update}, Layer: {layer_idx}")
            
            # 提取动态tokens的特征
            update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
            dynamic_hidden_ln1 = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
            dynamic_residual1 = residual1.gather(1, update_idx_expanded)  # [F, num_update, C]
            
        else:
            # ========== 后续层：输入已经是动态tokens [F, num_update, C] ==========
            update_indices = cache2.update_indices
            num_update = update_indices.shape[1]
            
            dynamic_residual1 = hidden_states  # [F, num_update, C]
            dynamic_hidden_ln1 = self.layer_norm1(hidden_states)  # [F, num_update, C]
            
            update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        
        num_update = update_indices.shape[1]
        seq_len_ref = self.reference_frame_key.shape[0]  # 参考帧的序列长度
        
        # ========== 阶段1：只为动态tokens计算Q ==========
        q_proj = self.self_attn.q_proj
        
        # 只为动态tokens计算Q
        query_selected = q_proj(dynamic_hidden_ln1)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段2：K和V完全使用参考帧 ==========
        # 从reference初始化完整的K矩阵
        key_states_ref = self.reference_frame_key.unsqueeze(0).expand(batch_size, -1, -1)  # [F, T, C]
        key_states_ref = key_states_ref.view(batch_size, seq_len_ref, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # 从reference初始化完整的V矩阵
        value_states_ref = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1)  # [F, T, C]
        value_states_ref = value_states_ref.view(batch_size, seq_len_ref, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段3：部分计算注意力（Q_selected @ K_ref^T @ V_ref） ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_ref,
            value_states=value_states_ref,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )  # attn_output_selected: [F, num_update, C]
        
        # Residual connection（只处理动态tokens）
        hidden_states = dynamic_residual1 + attn_output_selected  # [F, num_update, C]
        
        # ========== 阶段4：选择性MLP计算（只处理动态tokens） ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 只计算动态tokens的MLP
        mlp_output = self.mlp(hidden_states_ln2)  # [F, num_update, C]
        
        # Residual connection
        hidden_states = residual2 + mlp_output  # [F, num_update, C]
        
        # ========== 阶段5：仅在最后一层恢复完整tensor ==========
        if is_last_layer:
            # 从reference初始化完整输出
            full_output = self.reference_frame_output.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
            
            # Scatter更新动态tokens的输出
            full_output.scatter_(1, update_idx_expanded, hidden_states)
            
            hidden_states = full_output  # [F, T, C]
        
        # 否则输出仍是动态tokens [F, num_update, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs


def register_early_dynamic_detection_Siglip(vision_tower: nn.Module) -> None:
    """
    使用动态帧早期检测策略注册Siglip模型。
    在第0层判断每帧动态性，动态帧使用原始ViT，静态帧使用stc-cacher。
    """
    for layer_idx, layer in enumerate(vision_tower.vision_model.encoder.layers):
        setattr(layer, "_old_forward", layer.forward)
        setattr(layer, "layer_idx", layer_idx)  # 设置层索引
        layer.forward = types.MethodType(forward_with_early_dynamic_detection, layer)
        layer.new_attn = types.MethodType(new_siglip_sdpa_attn_forward, layer)
        layer._forward_stc_cacher = types.MethodType(_forward_stc_cacher, layer)


def forward_with_frame_by_frame_cache(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    逐帧滑动缓存策略：每帧都与前一帧计算相似度，通过cache减少计算。
    与forward_with_selective_key_recompute的区别：
    - 原函数：每cache_interval帧完整计算一帧，其余帧使用固定reference
    - 本函数：每帧都与前一帧比较，每帧完成后更新reference（滑动窗口）
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHER()
    update_token_ratio = cache2.update_token_ratio
    
    # 判断是否存在前帧reference（第一帧时不存在）
    has_reference = hasattr(self, 'reference_frame_key') and self.reference_frame_key is not None
    
    # ========== 无前帧reference：完整计算（第一帧） ==========
    if not has_reference:
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V作为下一帧的reference
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]
        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为下一帧的reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 有前帧reference：基于相似度的选择性重计算 ==========
    else:
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：基于Key识别需要更新的token ==========
        # 计算当前帧的Key向量（用于相似度计算）
        key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
        
        # 前一帧的Key作为reference
        ref_key_for_sim = self.reference_frame_key  # [T, C]
        
        # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
        similarity = torch.nn.functional.cosine_similarity(
            key_states_full,
            ref_key_for_sim.unsqueeze(0),
            dim=-1
        )
        
        num_update = int(seq_len * update_token_ratio)
        num_update = max(1, min(num_update, seq_len))
        
        # 对每一帧，选择相似度最低的token索引（变化最大的token）
        update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
        
        # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        rank = dist.get_rank()
        if rank == 0:
            logger.info(f"SigLIP Frame-by-Frame | Vocab size: {seq_len}, Tokens to update: {tokens_to_update.shape[1]}")
        
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从前一帧的reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full_reshaped = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full_reshaped,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从前一帧的reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从前一帧的reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full
        
        # ========== 关键差异：每帧都更新reference（滑动窗口） ==========
        # 保存当前帧（最后一帧）的状态作为下一帧的reference
        with torch.no_grad():
            # 更新K和V的reference（使用完整计算的key_states_full的最后一帧）
            self.reference_frame_key = key_states_full[-1].clone().detach()  # [T, C]
            
            # V需要从scatter后的完整V矩阵中取最后一帧
            # value_states_full: [F, num_heads, T, head_dim] -> 取最后一帧并reshape回 [T, C]
            self.reference_frame_value = value_states_full[-1].transpose(0, 1).contiguous().view(seq_len, embed_dim).detach()  # [T, C]
            
            # 更新AttnOut和MLPOut的reference
            self.reference_frame_attn_out = attn_output_full[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output_full[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs


def new_siglip_sdpa_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    q_len=query_states.shape[-2]
    batch_size=query_states.shape[0]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = False 

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

    attn_output = self.self_attn.out_proj(attn_output)
    return attn_output, None

def siglip_sdpa_attn_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    q_len=query_states.shape[-2]
    batch_size=query_states.shape[0]
    
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = False 

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, q_len, self.embed_dim)

    attn_output = self.self_attn.out_proj(attn_output)
    
    return attn_output, None
def forward_with_selective_key_recompute_clip(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
):
    """
    选择性重计算cache策略（K/V操作互换后）：
    - 偶数chunk：完整计算，保存最后一帧的K, V, AttnOut, MLPOut
    - 奇数chunk：基于Key相似度选择变化最剧烈的token，只为这些token计算Q和V
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHER()
    chunk_idx = cache2.chunk_idx
    is_even_chunk = (chunk_idx % 2 == 0)
    
    # ========== 偶数chunk：完整计算并保存reference frame ==========
    if is_even_chunk :
        # 标准的Transformer层计算
        residual1 = hidden_states
        
        # Layer Norm 1
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        # 获取attention模块的投影层
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        o_proj = self.self_attn.out_proj
        
        # 计算Q, K, V
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # [F, T, C] -> [F, T, C]
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V, AttnOut, MLPOut作为reference
        # 注意：保存的是projection后的张量，shape为[T, C]
        
        self.reference_frame_key = key_states[-1].clone().detach()  # [T, C]
        self.reference_frame_value = value_states[-1].clone().detach()  # [T, C]  # 修复这里！

        
        # Reshape for multi-head attention: [F, T, C] -> [F, num_heads, T, head_dim]
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # Residual connection
        hidden_states = residual1 + attn_output
        
        # Layer Norm 2 + MLP
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        # 保存最后一帧的AttnOut, MLPOut作为reference
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()  # [T, C]
            self.reference_frame_mlp_out = mlp_output[-1].detach()    # [T, C]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 奇数chunk：基于Key相似度的选择性重计算 ==========
    else:
        cache2 = STC_CACHER()
        update_token_ratio = cache2.update_token_ratio  
             
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        # ========== 阶段1：基于Key识别需要更新的token ==========
        # 计算当前帧的Key向量（用于相似度计算）
        key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
        
        # Reference frame的Key
        ref_key_for_sim = self.reference_frame_key  # [T, C]
        # 计算cosine相似度：[F, T, C] vs [T, C] -> [F, T]
        similarity = torch.nn.functional.cosine_similarity(
            key_states_full,
            ref_key_for_sim.unsqueeze(0),
            dim=-1
        )

        num_update = int(seq_len * update_token_ratio)
        num_update = max(1, min(num_update, seq_len))
        
        # 对每一帧，选择相似度最低的token索引
        update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
    
        # ========== 阶段2：只为选定token计算Q和V ==========
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        # 提取需要更新的token的特征
        update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
        tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只为这些token计算Q和V
        query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
        value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
        
        # Reshape: [F, num_update, C] -> [F, num_heads, num_update, head_dim]
        query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
        
        # ========== 阶段3：更新V矩阵（Scatter Update） ==========
        # 从reference初始化完整的V矩阵
        value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # Scatter: 将V_selected更新到对应位置
        update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
            batch_size, num_heads, num_update, head_dim
        )  # [F, num_heads, num_update, head_dim]
        value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
        
        # ========== 阶段4：完整计算K矩阵（全部使用新的） ==========
        key_states_full = k_proj(hidden_states_ln1)  # [F, T, C]
        key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # [F, num_heads, T, head_dim]
        
        # ========== 阶段5：部分计算注意力 ==========
        attn_output_selected, attn_weights = self.new_attn(
            query_states=query_selected,
            key_states=key_states_full,
            value_states=value_states_full,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )      
        
        # ========== 阶段6：Scatter Update到缓存的Attention输出 ==========
        # 从reference初始化
        attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # Scatter更新选定token的输出
        attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
        
        # Residual connection
        hidden_states = residual1 + attn_output_full
        
        # ========== 阶段7：选择性MLP计算 ==========
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        
        # 从reference初始化MLP输出
        mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()  # [F, T, C]
        
        # 提取需要更新的token（从ln2之后的特征）
        ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)  # [F, num_update, C]
        
        # 只计算选定token的MLP
        mlp_selected = self.mlp(ln2_tokens_to_update)  # [F, num_update, C]
        
        # Scatter更新
        mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
        
        # Residual connection
        hidden_states = residual2 + mlp_output_full

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)  # 只计算了部分token的attention
        
        return outputs


# ==================== 动态帧早期检测策略 ====================
# 在第0层根据帧与reference的平均相似度判断动态性，
# 动态帧在所有层使用原始ViT，静态帧使用stc-cacher

DYNAMIC_THRESHOLD = 0.7  # 平均相似度低于此值视为动态帧


def forward_with_early_dynamic_detection(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    动态帧早期检测策略：
    - Layer 0：计算每帧与reference的平均cosine相似度，判断是否为动态帧
    - Layer 1-N：根据Layer 0的判断结果，动态帧使用原始ViT，静态帧使用stc-cacher
    
    Args:
        hidden_states: [F, T, C] - F帧，T个token，C个通道
    """
    
    cache2 = STC_CACHER()
    chunk_idx = cache2.chunk_idx
    cache_interval = get_config().cache.cache_interval
    update_cache = (chunk_idx % cache_interval == 0)
    layer_idx = getattr(self, 'layer_idx', 0)  # 获取当前层索引
    
    # ========== 偶数chunk：完整计算并保存reference frame（所有层统一） ==========
    if update_cache:
        residual1 = hidden_states
        hidden_states_ln1 = self.layer_norm1(hidden_states)
        
        q_proj = self.self_attn.q_proj
        k_proj = self.self_attn.k_proj
        v_proj = self.self_attn.v_proj
        
        batch_size, seq_len, embed_dim = hidden_states_ln1.shape
        num_heads = self.self_attn.num_heads
        head_dim = embed_dim // num_heads
        
        query_states = q_proj(hidden_states_ln1)
        key_states = k_proj(hidden_states_ln1)
        value_states = v_proj(hidden_states_ln1)
        
        # 保存最后一帧的K, V作为reference
        self.reference_frame_key = key_states[-1].clone().detach()
        self.reference_frame_value = value_states[-1].clone().detach()
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.new_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = residual1 + attn_output
        
        residual2 = hidden_states
        hidden_states_ln2 = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states_ln2)
        hidden_states = residual2 + mlp_output
        
        with torch.no_grad():
            self.reference_frame_attn_out = attn_output[-1].detach()
            self.reference_frame_mlp_out = mlp_output[-1].detach()
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # ========== 非更新chunk ==========
    else:
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # ========== Layer 0：计算动态性并标记 ==========
        if layer_idx == 0:
            hidden_states_ln1 = self.layer_norm1(hidden_states)
            
            # 计算当前帧的Key用于动态性判断
            key_states_for_sim = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
            ref_key = self.reference_frame_key  # [T, C]
            
            # 计算每帧每个token的cosine相似度
            similarity = torch.nn.functional.cosine_similarity(
                key_states_for_sim,
                ref_key.unsqueeze(0),
                dim=-1
            )  # [F, T]
            
            # 计算每帧的平均相似度
            avg_similarity = similarity.mean(dim=-1)  # [F]
            
            # 判断哪些帧是动态帧（平均相似度低于阈值）
            dynamic_frame_mask = avg_similarity < DYNAMIC_THRESHOLD  # [F]
            
            # 存储到STC_CACHER供后续层使用
            cache2.dynamic_frame_mask = dynamic_frame_mask
            
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                num_dynamic = dynamic_frame_mask.sum().item()
                logger.info(f"SigLIP Early Detection Layer 0 | Total frames: {batch_size}, "
                           f"Dynamic frames: {num_dynamic}, Avg sim range: [{avg_similarity.min():.3f}, {avg_similarity.max():.3f}]")
        else:
            # Layer 1-N：从cache读取动态帧标记
            dynamic_frame_mask = cache2.dynamic_frame_mask
        
        # ========== 根据动态性分别处理每帧 ==========
        # 如果没有动态帧标记（异常情况），全部使用原始forward
        if dynamic_frame_mask is None:
            return self._old_forward(hidden_states, attention_mask, output_attentions)
        
        # 分离动态帧和静态帧
        num_dynamic = dynamic_frame_mask.sum().item()
        num_static = batch_size - num_dynamic
        
        # 如果全是动态帧，使用原始forward
        if num_dynamic == batch_size:
            return self._old_forward(hidden_states, attention_mask, output_attentions)
        
        # 如果全是静态帧，使用stc-cacher
        if num_static == batch_size:
            return self._forward_stc_cacher(hidden_states, attention_mask, output_attentions)
        
        # ========== 混合情况：分别处理动态帧和静态帧 ==========
        device = hidden_states.device
        
        # 分离输入
        dynamic_hidden = hidden_states[dynamic_frame_mask]  # [D, T, C]
        static_hidden = hidden_states[~dynamic_frame_mask]  # [S, T, C]
        
        # 处理动态帧（使用原始forward）
        dynamic_output = self._old_forward(dynamic_hidden, attention_mask, output_attentions)[0]
        
        # 处理静态帧（使用stc-cacher）
        static_output = self._forward_stc_cacher(static_hidden, attention_mask, output_attentions)[0]
        
        # 合并结果
        output = torch.zeros_like(hidden_states)
        output[dynamic_frame_mask] = dynamic_output
        output[~dynamic_frame_mask] = static_output
        
        outputs = (output,)
        if output_attentions:
            outputs += (None,)
        
        return outputs


def _forward_stc_cacher(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_attentions: bool = False,
):
    """
    STC-Cacher 选择性重计算逻辑（从 forward_with_selective_key_recompute 提取）
    用于处理静态帧
    
    Args:
        hidden_states: [F, T, C] - 仅静态帧
    """
    cache2 = STC_CACHER()
    update_token_ratio = cache2.update_token_ratio
    
    residual1 = hidden_states
    hidden_states_ln1 = self.layer_norm1(hidden_states)
    
    batch_size, seq_len, embed_dim = hidden_states_ln1.shape
    num_heads = self.self_attn.num_heads
    head_dim = embed_dim // num_heads
    
    # 计算当前帧的Key向量
    key_states_full = self.self_attn.k_proj(hidden_states_ln1)  # [F, T, C]
    
    # Reference frame的Key
    ref_key_for_sim = self.reference_frame_key  # [T, C]
    
    # 计算cosine相似度
    similarity = torch.nn.functional.cosine_similarity(
        key_states_full,
        ref_key_for_sim.unsqueeze(0),
        dim=-1
    )  # [F, T]
    
    num_update = int(seq_len * update_token_ratio)
    num_update = max(1, min(num_update, seq_len))
    
    # 选择相似度最低的token索引
    update_indices = torch.topk(similarity, k=num_update, dim=1, largest=False).indices  # [F, num_update]
    
    q_proj = self.self_attn.q_proj
    k_proj = self.self_attn.k_proj
    v_proj = self.self_attn.v_proj
    
    # 提取需要更新的token
    update_idx_expanded = update_indices.unsqueeze(-1).expand(-1, -1, embed_dim)  # [F, num_update, C]
    tokens_to_update = hidden_states_ln1.gather(1, update_idx_expanded)  # [F, num_update, C]
    
    # 只为这些token计算Q和V
    query_selected = q_proj(tokens_to_update)  # [F, num_update, C]
    value_selected = v_proj(tokens_to_update)  # [F, num_update, C]
    
    # Reshape
    query_selected = query_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
    value_selected = value_selected.view(batch_size, num_update, num_heads, head_dim).transpose(1, 2)
    
    # 从reference初始化V矩阵
    value_states_full = self.reference_frame_value.unsqueeze(0).expand(batch_size, -1, -1).clone()
    value_states_full = value_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scatter更新V
    update_idx_for_scatter = update_indices.unsqueeze(1).unsqueeze(-1).expand(
        batch_size, num_heads, num_update, head_dim
    )
    value_states_full.scatter_(2, update_idx_for_scatter, value_selected)
    
    # 完整计算K矩阵
    key_states_full = k_proj(hidden_states_ln1)
    key_states_full = key_states_full.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # 部分计算注意力
    attn_output_selected, attn_weights = self.new_attn(
        query_states=query_selected,
        key_states=key_states_full,
        value_states=value_states_full,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )
    
    # 从reference初始化attention输出
    attn_output_full = self.reference_frame_attn_out.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    # Scatter更新
    attn_output_full.scatter_(1, update_idx_expanded, attn_output_selected)
    
    # Residual connection
    hidden_states = residual1 + attn_output_full
    
    # 选择性MLP计算
    residual2 = hidden_states
    hidden_states_ln2 = self.layer_norm2(hidden_states)
    
    # 从reference初始化MLP输出
    mlp_output_full = self.reference_frame_mlp_out.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    # 提取需要更新的token
    ln2_tokens_to_update = hidden_states_ln2.gather(1, update_idx_expanded)
    
    # 只计算选定token的MLP
    mlp_selected = self.mlp(ln2_tokens_to_update)
    
    # Scatter更新
    mlp_output_full.scatter_(1, update_idx_expanded, mlp_selected)
    
    # Residual connection
    hidden_states = residual2 + mlp_output_full
    
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (None,)
    
    return outputs


