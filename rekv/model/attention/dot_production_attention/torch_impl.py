"""Pure PyTorch implementation of multi-stage dot-product attention."""

from __future__ import annotations

import math

import torch

from .base import MultiStageDotProductionAttention


class TorchMultiStageDotProductionAttention(MultiStageDotProductionAttention):
    """Multi-stage attention using standard PyTorch operations.

    Accumulates logits from multiple ``append`` calls, then computes
    a single global softmax and weighted sum in :meth:`finalize`.
    """

    def __init__(self, q_shape, dtype, device):
        super().__init__(q_shape, dtype, device)
        self.logits_list: list[torch.Tensor] = []
        self.v_list: list[torch.Tensor] = []
        self.mask_list: list[torch.Tensor] = []
        self.get_score_list: list[bool] = []
        self.kv_len_list: list[int] = []

    def finalize(self) -> None:
        """Compute final attention output from accumulated stages."""
        logits = torch.cat(self.logits_list, dim=-1)
        p = torch.softmax(logits, dim=-1)

        st = 0
        for kv_len, mask, get_score, v in zip(
            self.kv_len_list, self.mask_list, self.get_score_list, self.v_list
        ):
            ed = st + kv_len
            stage_p = torch.masked_fill(p[:, :, :, st:ed], ~mask, 0)

            self.score_list.append(stage_p.sum(dim=-2) if get_score else None)
            self.ret.add_(torch.matmul(stage_p, v))
            st = ed

    def append(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sliding_window=None,
        complement_sliding_window: bool = False,
        end: bool = False,
        get_score: bool = False,
        *args,
        **kwargs,
    ) -> None:
        len_q = q.size(-2)
        len_k = k.size(-2)

        # Handle grouped-query attention
        num_heads = q.size(1)
        num_heads_kv = k.size(1)
        if num_heads != num_heads_kv:
            num_group = num_heads // num_heads_kv
            shape = list(k.shape)
            k = (
                k[:, :, None, :, :]
                .expand(shape[0], shape[1], num_group, shape[2], shape[3])
                .reshape(shape[0], num_heads, shape[2], shape[3])
            )
            v = (
                v[:, :, None, :, :]
                .expand(shape[0], shape[1], num_group, shape[2], shape[3])
                .reshape(shape[0], num_heads, shape[2], shape[3])
            )

        # Build attention mask
        if sliding_window is None:
            mask = torch.ones((len_q, len_k), dtype=torch.bool, device=q.device)
        else:
            if isinstance(sliding_window, int):
                sliding_window = (len_k - len_q, sliding_window)

            dist = (
                torch.arange(len_q, dtype=torch.int64, device=q.device)[:, None]
                - torch.arange(len_k, dtype=torch.int64, device=k.device)[None, :]
                + sliding_window[0]
            )
            if complement_sliding_window:
                mask = dist >= sliding_window[1]
            else:
                mask = (dist < sliding_window[1]) & (dist >= 0)

        # Broadcast mask to [1, 1, len_q, len_k]
        m_shape = [1] * (4 - mask.dim()) + list(mask.shape)
        mask = mask.view(m_shape)

        self.v_list.append(v)
        self.mask_list.append(mask)
        self.get_score_list.append(get_score)
        self.kv_len_list.append(len_k)

        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = torch.masked_fill(logits, ~mask, float("-inf"))
        logits.mul_(1 / math.sqrt(q.size(-1)))
        self.logits_list.append(logits)

        if end:
            self.finalize()
