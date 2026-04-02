"""Abstract base class for ReKV-style streaming video models.

Provides shared logic for:
    - Initial prompt encoding
    - Chunked video encoding with STC caching
    - KV-cache lifecycle management
"""

from __future__ import annotations

import torch
from logzero import logger

from stc_core_code.vit_with_cacher.utils import STC_CACHE
from stc_core_code.controller import get_config



class Abstract_ReKV:
    """Mixin base that any HuggingFace model can inherit alongside its
    native ``*ForCausalLM`` / ``*ForConditionalGeneration`` class.

    Subclasses **must** implement:
        - ``_get_video_features(pixel_values_videos) -> Tensor``
        - ``question_answering(input_text, ...) -> str``
    """

    processor = None
    kv_cache = None

    def __init__(
        self,
        processor,
        n_frame_tokens: int,
        init_prompt_ids,
        n_local: int,
        topk: int,
        chunk_size: int,
    ) -> None:
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Cache lifecycle
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self.kv_cache = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_init_prompt(self) -> None:
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(
                [self.init_prompt_ids], device=self.device
            )
        output = self.language_model(
            input_ids=self.init_prompt_ids,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values

    # ------------------------------------------------------------------
    # Video feature extraction (template method)
    # ------------------------------------------------------------------

    def _get_video_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Extract visual features. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Chunk-wise video encoding
    # ------------------------------------------------------------------

    def _encode_video_chunk(self, video_chunk: torch.Tensor) -> None:
        pixel_values_videos = self.processor.video_processor(
            video_chunk, return_tensors="pt"
        ).pixel_values_videos.to(self.device, self.dtype)
        video_features = self._get_video_features(pixel_values_videos)
        assert self.n_local >= video_features.shape[1], (
            f"n_local ({self.n_local}) < video features ({video_features.shape[1]})"
        )
        output = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> None:
        """Encode video frame-by-frame with STC caching.

        Args:
            video: ``(N_frames, H, W, 3)`` uint8 tensor.
        """
        cfg = get_config()
        chunk_size = cfg.model.encode_chunk_size
        num_frames = video.shape[0]

        for chunk_idx in range(num_frames // chunk_size):
            ratio = cfg.cache.update_token_ratio
            if cfg.cache.strategy == "none":
                STC_CACHE.new_instance(0, ratio)
            else:
                STC_CACHE.new_instance(chunk_idx, ratio)

            start = chunk_idx * chunk_size
            self._encode_video_chunk(video[start : start + chunk_size])

        # Handle remaining frames
        remaining = num_frames % chunk_size
        if remaining > 0:
            start = (num_frames // chunk_size) * chunk_size
            self._encode_video_chunk(video[start : start + remaining])

    # ------------------------------------------------------------------
    # Question answering (template method)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def question_answering(
        self, input_text, max_new_tokens: int = 128, **kwargs
    ) -> str:
        """Generate answer given the encoded video context. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def calc_memory_usage(self) -> int:
        """Estimate KV-cache memory in bytes."""
        n_layers = len(self.kv_cache)
        return n_layers * self.kv_cache[0].calculate_cpu_memory()
