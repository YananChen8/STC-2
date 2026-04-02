"""Video-LLaVA + ReKV integration.

Video-LLaVA produces 257 tokens per frame (256 patch + 1 CLS).
"""

from __future__ import annotations

import torch
from logzero import logger
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

from model.abstract_rekv import Abstract_ReKV
from model.patch import patch_hf


class VideoLlava_ReKV(VideoLlavaForConditionalGeneration, Abstract_ReKV):
    """Video-LLaVA with ReKV KV-cache retrieval."""

    def __init__(
        self, config, processor, n_frame_tokens,
        init_prompt_ids, n_local, topk, chunk_size,
    ):
        VideoLlavaForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(
            self, processor, n_frame_tokens,
            init_prompt_ids, n_local, topk, chunk_size,
        )
        self.processor.video_processor = self.processor.image_processor

    def get_prompt(self, query: str, mc: bool = False) -> str:
        prompt = f"\n{query} ASSISTANT:"
        if mc:
            prompt += " Best option: ("
        return prompt

    def _get_video_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        _, video_features, _ = self._get_vision_features(
            pixel_values_videos=pixel_values_videos,
            vision_feature_layer=self.config.vision_feature_layer,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        return video_features.reshape(batch_size, frames * video_features.shape[1], -1)

    def _encode_video_chunk(self, video_chunk: torch.Tensor) -> None:
        pixels = self.processor.video_processor(
            images=None, videos=video_chunk, return_tensors="pt"
        ).pixel_values_videos.to(self.device, self.dtype)
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
                logits = out.logits
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits

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
    model_path: str = "model_zoo/Video-LLaVA-7B-hf",
    device=None,
    n_init: int | None = None,
    n_local: int | None = None,
    topk: int = 8,
    chunk_size: int = 1,
):
    if device is None:
        device = "cuda"

    n_frame_tokens = 257
    processor = VideoLlavaProcessor.from_pretrained(model_path)

    init_prompt = "USER: "
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

    model = VideoLlava_ReKV.from_pretrained(
        model_path,
        device_map="auto",
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

    for k, v in inf_llm_config.items():
        logger.info(f"{k}: {v}")
    logger.info(f"n_frame_tokens: {n_frame_tokens}")

    model.eval()
    return model, processor
