"""Multi-stage dot-product attention implementations.

Provides both a pure PyTorch fallback and an optional Triton-based
flash attention implementation. Use :func:`get_multi_stage_dot_production_attention`
to obtain the appropriate class.
"""

from typing import Tuple


def get_multi_stage_dot_production_attention(flash_attn: bool = False) -> Tuple[type, bool]:
    """Return the appropriate multi-stage attention class.

    Args:
        flash_attn: If ``True``, attempt to load the Triton implementation.

    Returns:
        ``(attention_class, is_flash)`` tuple.
    """
    if flash_attn:
        try:
            from .triton_impl import TritonMultiStageDotProductionAttention
            return TritonMultiStageDotProductionAttention, True
        except Exception as exc:
            if _should_warn():
                from warnings import warn
                warn(f"Failed to load Triton flash attention: {exc}. Using PyTorch impl.")

    from .torch_impl import TorchMultiStageDotProductionAttention
    return TorchMultiStageDotProductionAttention, False


# One-shot warning flag
_warned = False

def _should_warn() -> bool:
    global _warned
    if not _warned:
        _warned = True
        return True
    return False
