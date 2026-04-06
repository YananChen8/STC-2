from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Any, Dict
from pathlib import Path
import json
import os
import yaml

@dataclass(frozen=True)  # 建议 frozen=True 保证配置不可变，防抖屏
class CacheConfig:
    strategy: Literal['none', 'threshold','ratio','frame_by_frame','early_dynamic_detection'] = 'ratio'
    update_token_ratio: float = 0.25
    cache_interval: int = 4
    similarity_threshold: float = 0.9

@dataclass(frozen=True)
class ModelConfig:
    token_per_frame: int = 196
    prune_strategy: str = 'full_tokens'
    encode_chunk_size: int = 1
    ovo_fps: float = 1.0
    oracle_pruner: bool = False  # Enable query-aware oracle pruner (requires query to be set before encoding)

@dataclass
class GlobalConfig:
    """系统级配置容器"""
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __repr__(self) -> str:
        """结构化输出配置内容"""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _load_config_from_yaml() -> GlobalConfig:
    """从YAML文件加载配置，通过环境变量 EXP_CONFIG 指定配置文件名"""
    config_dir = Path(__file__).parent / "configs"
    config_name = os.environ.get("EXP_CONFIG", "default.yaml")
    
    # 如果用户只提供了文件名（不带.yaml后缀），自动补全
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    
    config_path = config_dir / config_name
    
    if not config_path.exists():
        print(f"[Warning] Config file not found: {config_path}, using default values")
        return GlobalConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    cache_data = data.get('cache', {})
    model_data = data.get('model', {})
    
    return GlobalConfig(
        cache=CacheConfig(**cache_data),
        model=ModelConfig(**model_data)
    )


config = _load_config_from_yaml()

def get_config() -> GlobalConfig:
    return config
