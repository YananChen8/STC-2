"""Base class for video question answering solvers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from decord import VideoReader, cpu
from logzero import logger

from .utils.data_utils import chunk_video


class BaseVQA:
    """Video QA base class providing shared encoding, inference, and formatting logic.

    Subclasses must implement :meth:`answer_single` or override :meth:`__call__`.
    """

    CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

    def __init__(self, model, processor, args) -> None:
        self.model = model
        self.processor = processor
        self.args = args
        self.results: list[dict] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, video_sample: dict) -> list[dict]:
        video = self.load_video(video_sample["video_path"], self.args.sample_fps)
        video_tensor = self._to_tensor(video)
        self.encode_video(video_tensor)
        return self.answer_questions(video_sample)

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_video(video_path: str, sample_fps: float = 1) -> "np.ndarray":
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        frame_idx = list(range(0, len(vr), max(1, int(fps / sample_fps))))
        video = vr.get_batch(frame_idx).asnumpy()
        logger.debug(f"Loaded video: {video.shape}")
        return video

    @staticmethod
    def _to_tensor(video) -> torch.Tensor:
        if isinstance(video, torch.Tensor):
            return video
        return torch.from_numpy(video)

    # ------------------------------------------------------------------
    # Video encoding
    # ------------------------------------------------------------------

    def encode_video(self, video: torch.Tensor) -> None:
        self.model.clear_cache()
        self.model.encode_init_prompt()
        self.model.encode_video(video)

        if dist.is_initialized() and dist.get_rank() == 0:
            size_gb = self.model.calc_memory_usage() / (1024 ** 3)
            logger.debug(f"Video encoded, cache size: {size_gb:.1f} GB")

    # ------------------------------------------------------------------
    # Single-clip inference (load → encode → QA in one shot)
    # ------------------------------------------------------------------

    def run_clip_inference(self, video_path: str, prompt: str, *, strip_last_line: bool = False) -> str | None:
        """Load a video clip, encode it, and answer a single prompt.

        This is used by online benchmarks (OVOBench, StreamingBench) where
        each question operates on a different video clip.

        Args:
            video_path: Path to video file.
            prompt: The prompt string to feed the model.
            strip_last_line: If True, return only the last non-empty line.
        """
        try:
            self.model.past_memory_mean_token = []
            video = self.load_video(video_path)
            video_tensor = self._to_tensor(video)

            self.model.clear_cache()
            self.model.encode_init_prompt()
            self.model.encode_video(video_tensor)

            response = self.model.question_answering(prompt)
            if strip_last_line and response:
                return response.strip().splitlines()[-1]
            return response
        except Exception as e:
            logger.error(f"Clip inference error ({video_path}): {e}")
            return None

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def answer_questions(self, video_sample: dict) -> list[dict]:
        results = []
        for qa in video_sample["conversations"]:
            result = self.answer_single(qa, video_sample["video_id"])
            results.append(result)
            self.results.append(result)
        return results

    def answer_single(self, qa_pair: dict, video_id: str) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_mcqa_prompt(self, question: str, choices: list[str]) -> str:
        formatted = "\n".join(
            f"({self.CHOICE_LETTERS[i]}) {c}" for i, c in enumerate(choices)
        )
        text = f"Question: {question}\nOptions:\n{formatted}\nOnly give the best option."
        return self.model.get_prompt(text, mc=True)

    def format_openqa_prompt(self, question: str) -> str:
        return self.model.get_prompt(question)

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_choice(pred_text: str) -> str:
        pred_text = pred_text.strip()
        if ")" in pred_text:
            idx = pred_text.index(")")
            return pred_text[idx - 1 : idx]
        return pred_text[0] if pred_text else "A"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, save_path: str) -> None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved {len(self.results)} results to {save_path}")
