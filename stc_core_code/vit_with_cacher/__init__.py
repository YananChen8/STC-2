"""STC ViT caching implementations for various vision encoders."""
from .utils import STC_CACHER, Singleton
from .siglip_with_cacher import (
    register_cache_by_key_Siglip,
    register_cache_by_key_CLIP,
    register_cache_by_threshold_Siglip,
    register_frame_by_frame_cache_Siglip,
)
from .qwen2_5_vl_with_cacher import (
    register_cache_for_qwen2_5_vl,
    unregister_cache_for_qwen2_5_vl,
)

__all__ = [
    "STC_CACHER",
    "Singleton",
    "register_cache_by_key_Siglip",
    "register_cache_by_key_CLIP",
    "register_cache_by_threshold_Siglip",
    "register_frame_by_frame_cache_Siglip",
    "register_cache_for_qwen2_5_vl",
    "unregister_cache_for_qwen2_5_vl",
]

