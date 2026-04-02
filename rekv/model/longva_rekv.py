"""LongVA + ReKV integration.

LongVA uses a CLIP vision tower with 2D pooling, producing 144 tokens per frame.
"""

from __future__ import annotations

import os

import torch
from logzero import logger
from transformers import AutoTokenizer

from model.abstract_rekv import Abstract_ReKV
from model.longva.model import LlavaQwenForCausalLM
from model.patch import patch_hf


class LongVA_ReKV(LlavaQwenForCausalLM, Abstract_ReKV):
    """LongVA with ReKV KV-cache retrieval."""

    def __init__(
        self, config, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
    ):
        LlavaQwenForCausalLM.__init__(self, config)
        processor = self.get_model().get_vision_tower().image_processor
        Abstract_ReKV.__init__(
            self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
        )

    def get_prompt(self, query: str, mc: bool = False) -> str:
        prompt = f"\n{query}<|im_end|>\n<|im_start|>assistant\n"
        if mc:
            prompt += "Best option: ("
        return prompt

    def _get_video_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        features = self.get_model().get_vision_tower()(pixel_values_videos)
        features = self.get_model().mm_projector(features)
        features = self.get_2dPool(features)
        return features.unsqueeze(0)

    def _encode_video_chunk(self, video_chunk: torch.Tensor) -> None:
        pixels = self.processor.preprocess(
            video_chunk, return_tensors="pt"
        ).pixel_values.to(self.device, self.dtype)
        features = self._get_video_features(pixels)
        assert self.n_local >= features.shape[1]
        output = self.language_model(
            inputs_embeds=features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def question_answering(
        self, input_text, max_new_tokens: int = 128, retrieved_indices=None,
    ) -> str:
        device = self.device
        tokenizer = self.processor.tokenizer
        stop_ids = {tokenizer.eos_token_id}

        # Retrieval phase
        input_ids = torch.as_tensor(
            [tokenizer(input_text["question"]).input_ids], device=device
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
                    [tokenizer(input_text["prompt"]).input_ids], device=device
                )
                embeds = self.get_input_embeddings()(ids)
                out = self.language_model(
                    inputs_embeds=embeds, use_cache=True, past_key_values=past_key_values
                )
                logits = self.lm_head(out["last_hidden_state"])
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.lm_head(out["last_hidden_state"])

            past_key_values = out.past_key_values
            _, top_indices = torch.topk(logits[0, -1, :], 2)
            token = top_indices[0].item()
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
    model_path: str = "model_zoo/LongVA-7B",
    device=None,
    n_init: int | None = None,
    n_local: int = 8000,
    topk: int = 32,
    chunk_size: int = 1,
):
    token_per_frame = int(os.getenv("TOKEN_PER_FRAME", "144"))
    n_frame_tokens = token_per_frame

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    init_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    init_prompt_ids = tokenizer(init_prompt).input_ids

    inf_llm_config = {
        "n_init": len(init_prompt_ids) if n_init is None else n_init,
        "n_local": n_local,
        "fattn": True,
        "block_size": n_frame_tokens,
        "topk": topk,
        "chunk_size": chunk_size,
        "max_cached_block": 128,
        "exc_block_size": n_frame_tokens,
        "pin_memory": True,
    }

    model = LongVA_ReKV.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map="auto")
    processor = vision_tower.image_processor
    processor.tokenizer = tokenizer

    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model

    for k, v in inf_llm_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"n_frame_tokens: {n_frame_tokens}")

    model.eval()
    return model, processor
