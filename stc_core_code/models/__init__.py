"""STC model-specific integrations for various VideoLLMs."""
from .qwen2_5_vl_pruner import (
    STC_Pruner_Qwen2_5_VL,
    Qwen2_5_VLModel_forward_pruner,
)

__all__ = [
    "STC_Pruner_Qwen2_5_VL",
    "Qwen2_5_VLModel_forward_pruner",
]

