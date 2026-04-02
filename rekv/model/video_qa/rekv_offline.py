"""Offline (non-streaming) video QA solver using ReKV.

Also handles VideoMME-specific features (GPU profiling, duration metadata)
via configuration flags — no separate subclass needed.
"""

from __future__ import annotations

import torch
from logzero import logger

from .base import BaseVQA


class ReKVOfflineVQA(BaseVQA):
    """Offline video QA — encodes the full video then answers questions.

    Supports:
    - Multiple-choice QA (answer as text or letter)
    - Open-ended QA
    - Optional GPU timing/memory profiling (set ``args.profile = True``)
    - Optional duration metadata injection (for VideoMME)
    """

    def __init__(self, model, processor, args) -> None:
        super().__init__(model, processor, args)
        self._profile = getattr(args, "profile", False)
        self._acc_time: float = 0.0
        self._max_mem: float = 0.0
        self._current_duration: str | None = None

    # ------------------------------------------------------------------
    # Video encoding (with optional profiling)
    # ------------------------------------------------------------------

    def encode_video(self, video: torch.Tensor) -> None:
        if not self._profile:
            return super().encode_video(video)

        self.model.clear_cache()
        self.model.encode_init_prompt()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self.model.encode_video(video)

        end_event.record()
        torch.cuda.synchronize()

        gpu_time = start_event.elapsed_time(end_event) / 1000.0
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        self._acc_time += gpu_time
        self._max_mem = max(peak_mem, self._max_mem)

        logger.debug(f"Encode: {gpu_time:.2f}s, peak mem: {peak_mem:.1f}MB")
        logger.debug(f"Accumulated: {self._acc_time:.2f}s, max mem: {self._max_mem:.1f}MB")

    # ------------------------------------------------------------------
    # QA dispatch
    # ------------------------------------------------------------------

    def answer_questions(self, video_sample: dict) -> list[dict]:
        self._acc_time = 0.0
        self._max_mem = 0.0
        self._current_duration = video_sample.get("duration")
        return super().answer_questions(video_sample)

    def answer_single(self, qa_pair: dict, video_id: str) -> dict:
        if "choices" in qa_pair:
            return self._multiple_choice(qa_pair, video_id)
        return self._open_ended(qa_pair, video_id)

    # ------------------------------------------------------------------
    # Open-ended QA
    # ------------------------------------------------------------------

    def _open_ended(self, qa_pair: dict, video_id: str) -> dict:
        question = qa_pair["question"]
        prompt = self.format_openqa_prompt(question)
        pred = self.model.question_answering(
            {"question": question, "prompt": prompt}, max_new_tokens=1024
        )
        return {
            "video_id": video_id,
            "question": question,
            "answer": qa_pair.get("answer"),
            "pred_answer": pred.replace("\n", ""),
        }

    # ------------------------------------------------------------------
    # Multiple-choice QA
    # ------------------------------------------------------------------

    def _multiple_choice(self, qa_pair: dict, video_id: str) -> dict:
        question = qa_pair["question"]
        choices = qa_pair["choices"]
        prompt = self.format_mcqa_prompt(question, choices)

        pred = self.model.question_answering(
            {"question": question, "prompt": prompt}, max_new_tokens=16
        )
        pred_choice = self.extract_choice(pred)
        correct = self._get_correct_choice(qa_pair)

        result = {
            "video_id": video_id,
            "question": question,
            "choices": choices,
            "answer": qa_pair.get("answer"),
            "correct_choice": correct,
            "pred_answer": pred.replace("\n", ""),
            "pred_choice": pred_choice,
            "qa_acc": float(pred_choice == correct) * 100,
        }

        # Inject duration metadata if available (VideoMME)
        if self._current_duration:
            result["duration"] = self._current_duration

        return result

    def _get_correct_choice(self, qa_pair: dict) -> str:
        """Determine the correct choice letter.

        Handles two formats:
        - Answer is a letter (e.g., "A") — used by VideoMME, EgoSchema
        - Answer is the choice text — used by MLVU, etc.
        """
        answer = qa_pair.get("answer")
        if answer is None:
            return self.CHOICE_LETTERS[0]

        # Direct letter answer (VideoMME format)
        if answer in self.CHOICE_LETTERS:
            return answer

        # Text answer — find in choices
        try:
            idx = qa_pair["choices"].index(answer)
            return self.CHOICE_LETTERS[idx]
        except ValueError:
            logger.warning(f"Answer '{answer}' not in choices")
            return self.CHOICE_LETTERS[0]
