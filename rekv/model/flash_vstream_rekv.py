"""Flash-VStream + ReKV integration.

Flash-VStream uses a spatial compression step producing 64 tokens per frame.
"""

from __future__ import annotations

import torch
from logzero import logger
from transformers import AutoTokenizer

from flash_vstream import VStreamLlamaForCausalLM
from model.abstract_rekv import Abstract_ReKV
from model.patch import patch_hf


class FlashVStream_ReKV(VStreamLlamaForCausalLM, Abstract_ReKV):
    """Flash-VStream with ReKV KV-cache retrieval."""

    def __init__(
        self, config, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
    ):
        VStreamLlamaForCausalLM.__init__(self, config)
        Abstract_ReKV.__init__(
            self, None, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
        )

    def get_prompt(self, query: str, mc: bool = False) -> str:
        prompt = f"\n{query}ASSISTANT:"
        if mc:
            prompt += "Best option: ("
        return prompt

    def _get_video_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        features = self.encode_images(pixel_values_videos)
        features = self.compress_spatial_features(features, 8)
        features = self.get_model().mm_projector(features)
        return features.flatten(0, 1).unsqueeze(0)

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

        # Retrieval phase (remove leading <s> for Llama tokenizer)
        input_ids = torch.as_tensor(
            [tokenizer(input_text["question"]).input_ids[1:]], device=device
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
                    [tokenizer(input_text["prompt"]).input_ids[1:]], device=device
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
    model_path: str = "checkpoints/flash-vstream-7b",
    device=None,
    n_init: int | None = None,
    n_local: int = 4000,
    topk: int = 16,
    chunk_size: int = 1,
):
    n_frame_tokens = 64
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    init_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions. USER: "
    )
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

    model = FlashVStream_ReKV.from_pretrained(
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
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.float16)

    processor = vision_tower.image_processor
    processor.tokenizer = tokenizer
    model.processor = processor

    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model

    for k, v in inf_llm_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"n_frame_tokens: {n_frame_tokens}")

    model.eval()
    return model, processor
