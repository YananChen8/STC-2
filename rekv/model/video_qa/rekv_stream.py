"""Streaming (incremental) video QA solver using ReKV.

Inherits QA logic (open-ended + multiple-choice) from ReKVOfflineVQA.
Only overrides video loading and encoding to support incremental frame feeding.
"""

from __future__ import annotations

import numpy as np
from decord import VideoReader, cpu

from .rekv_offline import ReKVOfflineVQA


class ReKVStreamVQA(ReKVOfflineVQA):
    """Streaming video QA — incrementally encodes frames per question.

    Reuses :meth:`answer_single` from :class:`ReKVOfflineVQA` for both
    multiple-choice and open-ended QA.
    """

    def __call__(self, video_sample: dict) -> list[dict]:
        video = self._load_stream_video(video_sample["video_path"])
        video_tensor = self._to_tensor(video)

        self.model.clear_cache()
        self.model.encode_init_prompt()

        video_start_idx = 0
        video_end_idx = 0

        for qa in video_sample["conversations"]:
            _, temporal_end = self._get_temporal_window(qa)
            if temporal_end > video_end_idx:
                video_end_idx = temporal_end
                new_frames = video_tensor[int(video_start_idx) : int(video_end_idx)]
                self.model.encode_video(new_frames)
                video_start_idx = video_end_idx

            result = self.answer_single(qa, video_sample["video_id"])
            self.results.append(result)

        return self.results

    def _load_stream_video(self, video_path: str):
        if video_path.endswith(".npy"):
            video = np.load(video_path)
            fps_ratio = self.args.sample_fps
            assert fps_ratio <= 1, "sample_fps should be <= 1 for .npy files"
            idx = np.linspace(0, len(video) - 1, int(len(video) * fps_ratio), dtype=int)
            return video[idx]

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = round(vr.get_avg_fps())
        frame_idx = list(range(0, len(vr), max(1, int(fps / self.args.sample_fps))))
        return vr.get_batch(frame_idx).asnumpy()

    def _get_temporal_window(self, qa_pair: dict) -> tuple[float, float]:
        start = qa_pair.get("start_time", 0) * self.args.sample_fps
        end = qa_pair.get("end_time", float("inf")) * self.args.sample_fps
        return start, end
