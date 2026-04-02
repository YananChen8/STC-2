"""
STC (Streaming Token Compression) Package

This package provides token compression mechanisms for VideoLLMs:
- STC-Pruner: Embedding-level token compression based on Gaussian similarity scores
- STC-Cacher: ViT-level selective recomputation caching
- OracleSTC-Pruner: Query-aware token compression using LLM attention scores

Supported Models:
- LLaVA-OV, LLaVA-Video, CLIP (original implementations)
- Qwen2.5-VL (new adaptations)
"""

from .controller import get_config, GlobalConfig, CacheConfig, ModelConfig
from .pruner import STC_Pruner, ScoreCalculator, IndexMapper
from .oracle_pruner import OracleSTC_Pruner, AttentionHook

__all__ = [
    "get_config",
    "GlobalConfig",
    "CacheConfig",
    "ModelConfig",
    "STC_Pruner",
    "ScoreCalculator",
    "IndexMapper",
    "OracleSTC_Pruner",
    "AttentionHook",
]

