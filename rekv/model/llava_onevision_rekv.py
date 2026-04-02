"""LLaVA-OneVision + ReKV + STC integration.

This module provides the primary model adapter that combines:
    - LLaVA-OneVision vision-language model
    - ReKV KV-cache retrieval for streaming
    - STC-Cacher (selective key recompute in SigLIP)
    - STC-Pruner (token pruning after ViT)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from logzero import logger
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from model.abstract_rekv import Abstract_ReKV
from model.patch import patch_hf

from stc_core_code.vit_with_cacher.utils import STC_CACHE
from stc_core_code.untils import get_config
from stc_core_code.vit_with_cacher.siglip_with_cacher import register_cache_by_key_Siglip

from stc_core_code.pruner import STC_Pruner


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    """LLaVA-OneVision with ReKV retrieval and STC acceleration."""

    def __init__(
        self, config, processor, n_frame_tokens,
        init_prompt_ids, n_local, topk, chunk_size,
    ):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(
            self, processor, n_frame_tokens,
            init_prompt_ids, n_local, topk, chunk_size,
        )
        # --- STC components ---
        register_cache_by_key_Siglip(self.vision_tower)
        STC_CACHE.new_instance(chunk_idx=0, update_token_ratio=0.25)
        self.stc_pruner = STC_Pruner()

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def get_prompt(self, query: str, mc: bool = False) -> str:
        prompt = f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    # ------------------------------------------------------------------
    # Vision feature extraction with STC pruning
    # ------------------------------------------------------------------

    def _get_video_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        flat_pixels = pixel_values_videos.view(batch_size * frames, channels, height, width)

        hidden_states = self.vision_tower(flat_pixels, output_hidden_states=True)
        selected = hidden_states.hidden_states[self.config.vision_feature_layer]

        if self.config.vision_feature_select_strategy == "default":
            selected = selected[:, 1:]  # remove CLS token

        projected = self.multi_modal_projector(selected)
        pooled = self.apply_pooling(projected)
        flat_features = pooled.reshape(-1, pooled.size(-1))

        # STC-Pruner: compress tokens
        token_per_frame = get_config().model.token_per_frame
        compressed = self.stc_pruner.compress(flat_features)

        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"LLM | Vocab size: 196, Tokens retained: {token_per_frame}")

        num_output_frames = compressed.shape[0] // token_per_frame
        return compressed.reshape(batch_size, num_output_frames * token_per_frame, -1)

    # ------------------------------------------------------------------
    # Autoregressive QA generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def question_answering(
        self,
        input_text,
        max_new_tokens: int = 128,
        retrieved_indices=None,
    ) -> str:
        device = self.device
        tokenizer = self.processor.tokenizer
        stop_ids = {tokenizer.eos_token_id}

        # Parse input
        if isinstance(input_text, str):
            question_text = prompt_text = input_text
        else:
            question_text = input_text["question"]
            prompt_text = input_text["prompt"]

        # Retrieval phase
        input_ids = torch.as_tensor(
            [tokenizer(question_text).input_ids], device=device
        )

        for layer_kv in self.kv_cache:
            layer_kv.set_retrieval()

        if retrieved_indices is None:
            out = self.language_model(
                input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache
            )
        else:
            for layer_kv in self.kv_cache:
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(
                input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache
            )
        past_key_values = out.past_key_values

        for layer_kv in self.kv_cache:
            layer_kv.reset_retrieval()

        # Autoregressive generation
        output_ids: list[int] = []
        for step in range(max_new_tokens):
            if step == 0:
                ids = torch.as_tensor(
                    [tokenizer(prompt_text).input_ids], device=device
                )
                embeds = self.get_input_embeddings()(ids)
                out = self.language_model(
                    inputs_embeds=embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )

            past_key_values = out.past_key_values
            logits = out.logits[0, -1, :]
            _, top_indices = torch.topk(logits, 2)
            token = top_indices[0].item()

            # Avoid emitting EOS on the very first token
            if step == 0 and token in stop_ids:
                token = top_indices[1].item() if top_indices.numel() > 1 else 1

            output_ids.append(token)
            if token in stop_ids:
                break

        return tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )


# ======================================================================
# Model loader
# ======================================================================

def load_model(
    model_path: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    device=None,
    n_init: int | None = None,
    n_local: int = 15000,
    topk: int = 64,
    chunk_size: int = 1,
):
    """Load LlavaOneVision_ReKV with STC patches applied."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    token_per_frame = get_config().model.token_per_frame
    n_frame_tokens = int(token_per_frame)

    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    init_prompt = (
        "<|im_start|>system \nYou are a helpful assistant."
        "<|im_end|><|im_start|>user "
    )
    init_prompt_ids = processor.tokenizer(
        init_prompt, return_tensors="pt"
    ).input_ids.to(device)

    inf_llm_config = {
        "n_init": init_prompt_ids.shape[1] if n_init is None else n_init,
        "n_local": n_local,
        "fattn": True,
        "block_size": n_frame_tokens,
        "topk": topk,
        "chunk_size": chunk_size,
        "max_cached_block": 128,
        "exc_block_size": n_frame_tokens,
        "pin_memory": True,
    }

    model = LlavaOneVision_ReKV.from_pretrained(
        model_path,
        device_map={"": device},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    model.language_model = patch_hf(model.language_model, **inf_llm_config)

    if dist.is_initialized() and dist.get_rank() == 0:
        for k, v in inf_llm_config.items():
            logger.info(f"{k}: {v}")
        logger.info(f"n_frame_tokens: {n_frame_tokens}")

    model.eval()
    return model, processor
