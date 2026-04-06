"""
Compatibility wrapper for Qwen2-VL selective recomputation cache.

LiveCC inference uses `Qwen2VLForConditionalGeneration`. The existing cache
implementation lives in `qwen2_5_vl_with_cacher.py`, and this module exposes
Qwen2-VL-oriented names for easier integration.
"""

from qwen2_5_vl_with_cacher import (
    qwen2_5_vl_visual_forward_with_cache,
    register_cache_for_qwen2_5_vl,
    unregister_cache_for_qwen2_5_vl,
)


def register_cache_for_qwen2_vl(model):
    return register_cache_for_qwen2_5_vl(model)


def unregister_cache_for_qwen2_vl(model):
    return unregister_cache_for_qwen2_5_vl(model)


def qwen2_vl_visual_forward_with_cache(self, hidden_states, grid_thw):
    return qwen2_5_vl_visual_forward_with_cache(self, hidden_states, grid_thw)
