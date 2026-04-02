"""Monkey-patch HuggingFace LLM models for ReKV attention.

Replaces the standard self-attention forward and model forward to support
the ReKV KV-cache management (block-wise caching, retrieval, sliding window).
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from model.attention import RotaryEmbeddingESM, rekv_attention_forward


# ======================================================================
# Attention forward wrapper
# ======================================================================

def _wrap_attention_forward(forward):
    """Adapt ReKV attention forward to HuggingFace attention interface."""

    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions, "output_attentions not supported with ReKV"
        ret = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_value,
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.head_dim, self.num_heads, self.num_key_value_heads,
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None
        return o, None, pkv

    return hf_forward


# ======================================================================
# Model forward replacement
# ======================================================================

def _make_model_forward():
    """Create a replacement model forward that routes through ReKV attention."""
    from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

    def model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *args,
        **kwargs,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
                inputs_embeds = inputs_embeds * self.config.scale_emb

        pkv = tuple() if use_cache else None
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_out[0]

            if use_cache:
                cache = layer_out[2 if output_attentions else 1]
                pkv = pkv + (cache,)
            if output_attentions:
                all_self_attns += (layer_out[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, pkv, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=pkv,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return model_forward


# ======================================================================
# Main patching entry point
# ======================================================================

_SUPPORTED_MODELS = {
    "LlamaForCausalLM", "MistralForCausalLM",
    "Qwen2ForCausalLM", "Qwen2Model", "MiniCPMForCausalLM",
}


def patch_hf(
    model,
    attn_kwargs: dict | None = None,
    base=None,
    distance_scale=None,
    **kwargs,
):
    """Patch a HuggingFace causal LM to use ReKV attention.

    Args:
        model: A HuggingFace causal LM (Llama, Qwen2, Mistral, etc.)
        **kwargs: ReKV config (n_init, n_local, topk, etc.)

    Returns:
        The patched model.
    """
    if attn_kwargs is None:
        attn_kwargs = {}
    attn_kwargs.update(kwargs)

    # Identify model architecture
    from transformers import LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model

    model_cls_name = model.__class__.__name__
    if model_cls_name not in _SUPPORTED_MODELS and not isinstance(
        model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen2Model)
    ):
        raise ValueError(
            f"Unsupported model: {model_cls_name}. "
            f"Supported: {_SUPPORTED_MODELS}"
        )

    Attention = model.model.layers[0].self_attn.__class__
    Model = model.model.__class__

    # Build RoPE
    hf_rope = model.model.layers[0].self_attn.rotary_emb
    if isinstance(hf_rope, Qwen2RotaryEmbedding):
        rope_base = hf_rope.base
        rope_distance_scale = 1.0
        rope_dim = hf_rope.dim
    else:
        rope_base = hf_rope.config.rope_theta
        rope_distance_scale = distance_scale if distance_scale is not None else 1.0
        partial = getattr(hf_rope.config, "partial_rotary_factor", 1.0)
        rope_dim = int(
            (hf_rope.config.hidden_size // hf_rope.config.num_attention_heads) * partial
        )

    rope = RotaryEmbeddingESM(rope_dim, rope_base, rope_distance_scale)
    model.model.position_bias = rope

    # Patch attention layers
    forward = _wrap_attention_forward(rekv_attention_forward(**attn_kwargs))

    def _set_forward(m):
        if isinstance(m, Attention):
            m._old_forward = m.forward
            m.forward = forward.__get__(m, Attention)

    model.apply(_set_forward)

    # Patch model forward
    model_forward = _make_model_forward()
    model.model._old_forward = model.model.forward
    model.model.forward = model_forward.__get__(model.model, Model)

    return model
