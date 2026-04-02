"""KV-Cache management for ReKV-style streaming video attention.

This module provides GPU/CPU memory management for KV-Cache in the
streaming inference pipeline.  Key components:

* :class:`CudaCache` — fixed-size GPU block allocator.
* :class:`MemoryUnit` — CPU↔GPU data-transfer unit for a single block.
* :class:`VectorTensor` — dynamically-growing GPU vector store for
  per-block representative keys and various scoring utilities.
* :class:`ContextManager` — the main manager that orchestrates block
  retrieval, token compression, and multi-stage attention.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .dot_production_attention import get_multi_stage_dot_production_attention
from .aks import adaptive_keyframe_sampling
from .dpc_knn import dpc_knn_select_tokens_batched
from .sparse_loading import compute_image_attention_scores, get_kept_token_indices


# ======================================================================
# GPU Block Allocator
# ======================================================================

class CudaCache:
    """Fixed-size GPU memory pool for KV-Cache blocks.

    Pre-allocates a contiguous buffer of ``num_units`` blocks on the GPU.
    Each block has size ``unit_size`` elements of the given *dtype*.

    Args:
        num_units: Number of blocks to pre-allocate.
        unit_size: Flat element count per block
                   (typically ``block_size * hidden_dim * 2``).
        dtype: Element data type.
    """

    def __init__(self, num_units: int, unit_size: int, dtype: torch.dtype):
        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty((num_units, unit_size), device="cuda", dtype=dtype)
        self.idle_set: set[int] = set(range(num_units))

    def alloc(self) -> Tuple[torch.Tensor, int]:
        """Allocate one block and return ``(view, block_id)``."""
        assert len(self.idle_set) > 0, "CudaCache exhausted"
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx: int) -> None:
        """Return a block to the idle pool."""
        assert idx not in self.idle_set
        self.idle_set.add(idx)


# ======================================================================
# CPU↔GPU Data Transfer Unit
# ======================================================================

class MemoryUnit:
    """Manages a single KV-Cache block with CPU storage and optional GPU cache.

    Args:
        kv: Tuple of ``(key, value)`` tensors for one block.
        cache: The :class:`CudaCache` allocator.
        load_to_cache: If ``True``, immediately copy to GPU cache.
        pin_memory: If ``True``, pin the CPU copy for faster transfers.
    """

    def __init__(
        self,
        kv: Tuple[torch.Tensor, torch.Tensor],
        cache: CudaCache,
        load_to_cache: bool = False,
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(t.detach().to("cpu", non_blocking=True).contiguous() for t in kv)
        else:
            cpu_data = tuple(t.contiguous() for t in kv)

        if pin_memory:
            cpu_data = tuple(t.pin_memory() for t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    def load(
        self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[bool, Optional[torch.cuda.Event]]:
        """Load from CPU to GPU; optionally copy into *target* tensors.

        Returns:
            ``(was_loaded_from_cpu, target_event)``
        """
        if self.gpu_data is not None:
            # Already on GPU
            target_event = None
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            return False, target_event

        # Load from CPU
        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)

        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)
        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = None

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        return True, target_event

    def get(self) -> torch.Tensor:
        """Return the GPU-resident KV data (waits for pending copies)."""
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    def offload(self) -> None:
        """Release the GPU copy back to the :class:`CudaCache` pool."""
        assert self.gpu_data is not None
        self.event.wait()
        self.cache.delete(self.gpu_data_id)
        self.gpu_data = None
        self.gpu_data_id = None

    def calculate_cpu_memory(self) -> int:
        """Return approximate CPU memory usage in bytes."""
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# ======================================================================
# Dynamic GPU Vector Store
# ======================================================================

class VectorTensor:
    """Dynamically-growing GPU vector buffer for block representative keys.

    Stores per-block summary vectors and provides various similarity /
    scoring utilities used for block retrieval.

    Args:
        hidden_size: Dimensionality of each vector.
        element_dtype: Data type of the stored vectors.
        device: Target device (should be a CUDA device).
    """

    _INIT_CAPACITY = 16

    def __init__(self, hidden_size: int, element_dtype: torch.dtype, device: torch.device):
        self.hidden_size = hidden_size
        self.length = 0
        self._capacity = self._INIT_CAPACITY
        self.data = torch.empty(
            (self._capacity, hidden_size), dtype=element_dtype, device=device
        )

    # ------------------------------------------------------------------
    # Core storage operations
    # ------------------------------------------------------------------

    def append(self, tensor: torch.Tensor) -> None:
        """Append *tensor* (shape ``[n, hidden_size]``) to the buffer."""
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size
        assert tensor.is_contiguous()

        n = tensor.size(0)
        while self.length + n > self._capacity:
            self._grow()
        self.data[self.length : self.length + n].copy_(tensor)
        self.length += n

    def get_data(self) -> torch.Tensor:
        """Return the valid portion of the buffer."""
        return self.data[: self.length]

    def __len__(self) -> int:
        return self.length

    # ------------------------------------------------------------------
    # Similarity / scoring utilities
    # ------------------------------------------------------------------

    def get_cosine_similarity(self, query: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between *query* ``[D]`` and all stored vectors.

        Returns:
            1-D tensor of shape ``[length]``.
        """
        assert query.dim() == 1 and query.size(0) == self.hidden_size
        keys = self.data[: self.length].float()
        q = query[None, :].float()
        return torch.matmul(q, keys.T)[0]

    def get_block_query_relevance(self, query: torch.Tensor) -> torch.Tensor:
        """Mean-pooled cosine similarity between *query* tokens and stored keys."""
        keys = F.normalize(self.data[: self.length].float(), p=2, dim=1)
        q_norm = F.normalize(query.float(), p=2, dim=1)
        return torch.mm(q_norm, keys.T).mean(dim=0)

    def get_global_block_uniqueness(self) -> torch.Tensor:
        """Per-block cosine similarity to the global mean vector."""
        features = self.data[: self.length].float()
        mean_vec = features.mean(dim=0, keepdim=True).expand_as(features)
        return F.cosine_similarity(features, mean_vec, dim=1)

    def get_global_block_l2norm(
        self, global_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """L2 norm of each stored (or provided) block vector."""
        features = global_k.float() if global_k is not None else self.data[: self.length].float()
        return torch.norm(features, p=2, dim=1)

    def calculate_inter_frame_similarity(
        self, global_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Consecutive-frame cosine dissimilarity ``1 - cos(f_i, f_{i-1})``.

        Returns:
            Tensor of shape ``[frame_number]`` where index 0 is 0.
        """
        tensor = global_k.float() if global_k is not None else self.data[: self.length].float()
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2-D tensor, got {tensor.dim()}-D")
        if tensor.size(0) < 2:
            return torch.ones(tensor.size(0), device=tensor.device)

        sims = F.cosine_similarity(tensor[:-1], tensor[1:], dim=1)
        first = torch.tensor([0.0], device=tensor.device)
        return 1.0 - torch.cat([first, sims])

    def dpc_knn_select_tokens(self, k: int = 20, n_to_keep: int = 64) -> torch.Tensor:
        """Run DPC-KNN directly on stored block vectors."""
        from .dpc_knn import dpc_knn_select_tokens as _select
        return _select(self.data[: self.length].float(), k, n_to_keep)

    # ------------------------------------------------------------------
    # Bias utilities for temporal weighting
    # ------------------------------------------------------------------

    def get_bias_video_length(self) -> float:
        return math.log(float(self.length) + 1)

    def get_bias_memory_decay_exp(self) -> torch.Tensor:
        """Exponential recency bias, normalized to [0, 1]."""
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        decay = torch.exp(idx / (self.length - 1 + 1e-8))
        return (decay - decay.min()) / (decay.max() - decay.min() + 1e-8)

    def get_bias_memory_decay_linear(self) -> torch.Tensor:
        """Linear decay bias (most recent = highest)."""
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        return 1.0 - idx / (self.length - 1 + 1e-8)

    def get_bias_memory_linear_increase(self) -> torch.Tensor:
        """Linear increase bias (most recent = highest)."""
        if self.length == 0:
            return torch.tensor([], dtype=self.data.dtype, device=self.data.device)
        idx = torch.arange(self.length, dtype=self.data.dtype, device=self.data.device)
        return idx / (self.length - 1 + 1e-8)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        new_data = torch.empty(
            (new_capacity,) + self.data.shape[1:],
            device=self.data.device,
            dtype=self.data.dtype,
        )
        new_data[: self._capacity].copy_(self.data)
        self.data = new_data
        self._capacity = new_capacity


# ======================================================================
# Global CUDA stream for asynchronous context loading
# ======================================================================

_GLOBAL_STREAM: Optional[torch.cuda.Stream] = None


def _get_global_stream() -> torch.cuda.Stream:
    global _GLOBAL_STREAM
    if _GLOBAL_STREAM is None:
        _GLOBAL_STREAM = torch.cuda.Stream()
    return _GLOBAL_STREAM


# ======================================================================
# Context Manager (main orchestrator)
# ======================================================================

class ContextManager:
    """Orchestrates KV-Cache block storage, retrieval, and attention.

    Manages the lifecycle of context memory blocks:
    1. Incoming KV pairs are split into blocks and offloaded to CPU.
    2. Per-block representative keys are stored in :class:`VectorTensor`.
    3. At query time, top-k blocks are retrieved via cosine similarity
       and loaded back to GPU for attention computation.

    Args:
        position_embedding: :class:`RotaryEmbeddingESM` instance.
        n_init: Number of initial (always-resident) KV tokens.
        n_local: Local sliding window size.
        block_size: Tokens per context block.
        max_cached_block: Max blocks kept on GPU simultaneously.
        topk: Number of blocks to retrieve.
        chunk_size: Chunk granularity for block retrieval.
        exc_block_size: Execution block size for chunked attention.
        fattn: Whether to use Triton flash attention.
        async_global_stream: Enable async context loading.
        pin_memory: Pin CPU memory for faster transfers.
    """

    def __init__(
        self,
        position_embedding,
        n_init: int,
        n_local: int,
        block_size: int,
        max_cached_block: int,
        topk: int,
        chunk_size: int,
        exc_block_size: int,
        fattn: bool = False,
        async_global_stream: bool = False,
        pin_memory: bool = False,
    ):
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local
        self.topk = topk
        self.chunk_size = chunk_size
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory

        if self.async_global_stream:
            _get_global_stream()

        self.reset_retrieval()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, local_q, local_k, local_v, global_q, global_k, global_v):
        """Initialize internal buffers using tensor metadata (shape/dtype/device)."""
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert t.size(0) == batch_size
            assert t.size(1) in (num_heads, num_heads_kv)
            assert t.size(2) == len_q
            assert t.size(3) == dim_head
            assert t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        # Context memory blocks: list[batch] of list[MemoryUnit]
        self.global_blocks: List[list] = [[] for _ in range(self.num_units)]
        # LRU tracking: dict mapping block_id → load_count
        self.cached_blocks: List[dict] = [{} for _ in range(self.num_units)]
        self.num_global_block = 0

        hidden_dim = dim_head * self.unit_size

        # Per-block representative keys
        self.block_k = [
            VectorTensor(hidden_dim, global_k.dtype, global_k.device)
            for _ in range(self.num_units)
        ]

        # Local KV (sliding window)
        empty_kv = lambda dt: torch.empty(
            (self.num_units, self.unit_size_kv, 0, dim_head),
            dtype=dt, device=local_k.device,
        )
        self.local_k = empty_kv(local_k.dtype)
        self.local_v = empty_kv(local_v.dtype)

        # Global remainder (not yet processed into blocks)
        self.global_remainder = (
            empty_kv(global_k.dtype),
            empty_kv(global_v.dtype),
        )

        # Init KV (always-resident prefix)
        self.init_k = empty_kv(global_k.dtype)
        self.init_v = empty_kv(global_k.dtype)
        self.init_exc = False
        self.dtype = local_q.dtype

        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # Global buffer for retrieved KV
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, buffer_len, dim_head),
            dtype=global_k.dtype, device=global_k.device,
        )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0

        # CUDA cache pool
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype,
        )
        self.initialized = True

    # ------------------------------------------------------------------
    # Retrieval control
    # ------------------------------------------------------------------

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def set_retrieved_block_indices_score(self, block_score):
        if isinstance(block_score, torch.Tensor):
            block_score = block_score.cpu().tolist()
        self.block_score = block_score

    # ------------------------------------------------------------------
    # GQA (Grouped Query Attention) helpers
    # ------------------------------------------------------------------

    def _from_group_kv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Expand KV heads for GQA: ``[B, H_kv, L, D] → [B, H, L, D]``."""
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        return (
            tensor.view(self.num_units, self.unit_size_kv, 1, length, dim_head)
            .expand(self.num_units, self.unit_size_kv, num_group, length, dim_head)
            .reshape(self.num_units, self.num_heads, length, dim_head)
        )

    def _get_block_representative(self, block_k_tensor: torch.Tensor) -> torch.Tensor:
        """Compute per-block representative key by averaging over tokens.

        Args:
            block_k_tensor: ``[B, H, block_size, D]``

        Returns:
            ``[B, 1, H*D]``
        """
        block_k_mean = block_k_tensor.mean(dim=-2)  # [B, H, D]
        return block_k_mean.reshape(self.num_units, -1)[:, None, :]

    # ------------------------------------------------------------------
    # Block top-k retrieval
    # ------------------------------------------------------------------

    def _calc_block_topk(
        self, global_h_q: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Retrieve top-k blocks by cosine similarity with the query.

        This is the primary retrieval method used during inference.

        Args:
            global_h_q: Query tensor ``[B, H, L, D]``.

        Returns:
            ``(block_indices, scores)`` where *block_indices* is
            ``batch_size × topk`` and *scores* is the similarity logits.
        """
        # Flatten query: [B, H*D]
        global_h_q = global_h_q.mean(dim=2)
        assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
        global_h_q = global_h_q.reshape(self.num_units, -1)

        logits = None
        indices_score = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:
                # Local window not yet filled — retrieve from remainder
                assert self.global_remainder[0].size(-2) > self.n_init
                global_k = self._from_group_kv(
                    self.global_remainder[0][:, :, self.n_init:, :]
                )
                assert global_k.size(-2) % self.block_size == 0
                block_num = global_k.size(-2) // self.block_size

                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]
                    indices_score = [
                        [1] * len(self.global_blocks[0])
                        for _ in range(self.num_units)
                    ]
                else:
                    global_k = (
                        global_k.transpose(1, 2)
                        .reshape(self.num_units, block_num, self.block_size, -1)
                        .mean(dim=-2)
                    )
                    logits = torch.matmul(
                        global_k, global_h_q[:, :, None]
                    ).squeeze(-1)
            else:
                ret = [
                    list(range(len(self.global_blocks[0])))
                    for _ in range(self.num_units)
                ]
                indices_score = [
                    [1] * len(self.global_blocks[0])
                    for _ in range(self.num_units)
                ]
        else:
            logits = torch.stack([
                self.block_k[u].get_cosine_similarity(global_h_q[u])
                for u in range(self.num_units)
            ])

        if logits is not None:
            self.similarity = logits
            ret, indices_score = self._chunked_topk_from_logits(logits)

        return ret, indices_score

    def _chunked_topk_from_logits(
        self, logits: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Convert block-level logits to chunked top-k indices.

        Groups blocks into chunks of ``chunk_size``, averages logits within
        each chunk, selects top-k chunks, then expands back to block indices.
        """
        assert self.topk % self.chunk_size == 0
        remainder_size = logits.shape[1] % self.chunk_size

        chunked = (
            logits[:, : logits.shape[1] - remainder_size]
            .reshape(self.num_units, -1, self.chunk_size)
            .mean(dim=-1)
        )
        if remainder_size > 0:
            tail = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)
            chunked = torch.cat([chunked, tail], dim=1)

        scores, indices = chunked.topk(
            self.topk // self.chunk_size, dim=1, largest=True
        )

        # Expand chunk indices to block indices
        sorted_indices = indices.sort(dim=1)[0][:, :, None]
        expanded = (
            sorted_indices * self.chunk_size
            + torch.arange(self.chunk_size, device=indices.device)[None, None, :]
        )
        ret = expanded.reshape(self.num_units, -1).cpu().tolist()

        # Filter overflow from the last partial chunk
        num_blocks = logits.shape[1]
        for u in range(self.num_units):
            ret[u] = [idx for idx in ret[u] if idx < num_blocks]

        return ret, scores

    # ------------------------------------------------------------------
    # KV retrieval and loading
    # ------------------------------------------------------------------

    def _remove_lru_blocks(
        self, u: int, num_remove: Optional[int] = None, ignore_blocks=None
    ):
        """Evict least-recently-used blocks from GPU for unit *u*."""
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block
        if num_remove <= 0:
            return

        lst = sorted(self.cached_blocks[u].items(), key=lambda x: x[1])
        removed = 0
        for idx, _ in lst:
            if ignore_blocks is not None and idx in ignore_blocks:
                continue
            self.global_blocks[u][idx].offload()
            self.cached_blocks[u].pop(idx)
            removed += 1
            if removed >= num_remove:
                return

    def get_retrieved_kv(
        self, query: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve context KV blocks based on query similarity.

        If *query* is provided, computes block top-k retrieval.
        Otherwise, uses previously set ``retrieved_block_indices``.

        Returns:
            ``(global_h_k, global_h_v)`` — concatenation of init KV and
            retrieved context KV.
        """
        stream = _get_global_stream() if self.async_global_stream else torch.cuda.current_stream()

        if query is not None:
            block_topk, indices_score = self._calc_block_topk(query)
            self.set_retrieved_block_indices(block_topk)
            self.set_retrieved_block_indices_score(indices_score)

        assert len(self.retrieved_block_indices) == self.num_units

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(stream):
            if self.init_exc:
                self._load_retrieved_from_offloaded(global_h_k, global_h_v)
            else:
                ed = self._load_retrieved_from_remainder(global_h_k, global_h_v)

            # Trim to actual length
            init_ed = self.init_k.size(-2)
            last_block = self.retrieved_block_indices[0]
            if last_block:
                ed = init_ed + len(last_block) * self.block_size
            else:
                ed = init_ed
            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(stream)

        assert global_h_k.size(-2) <= self.n_init + self.n_local
        return global_h_k, global_h_v

    def _load_retrieved_from_offloaded(self, global_h_k, global_h_v):
        """Load retrieved blocks from CPU-offloaded global_blocks."""
        for u in range(self.num_units):
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block
            for b_idx in self.retrieved_block_indices[u]:
                if b_idx not in self.cached_blocks[u]:
                    num_remove += 1
            self._remove_lru_blocks(u, num_remove, self.retrieved_block_indices[u])

        self.load_count += 1
        for u in range(self.num_units):
            for b_idx in self.retrieved_block_indices[u]:
                self.cached_blocks[u][b_idx] = self.load_count

        init_ed = self.init_k.size(-2)
        for u in range(self.num_units):
            assert self.retrieved_block_indices[u][-1] < self.num_global_block
            for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                st = init_ed + cnt * self.block_size
                ed = st + self.block_size
                self.global_blocks[u][b_idx].load(
                    (global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :])
                )

    def _load_retrieved_from_remainder(self, global_h_k, global_h_v) -> int:
        """Load retrieved blocks from in-memory remainder (before offloading)."""
        init_st = 0
        init_ed = self.n_init
        global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][:, :, init_st:init_ed]
        global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][:, :, init_st:init_ed]
        ed = init_ed

        for u in range(self.num_units):
            for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                remainder_st = init_ed + b_idx * self.block_size
                remainder_ed = remainder_st + self.block_size
                if remainder_st >= self.global_remainder[0].size(2):
                    break
                st = init_ed + cnt * self.block_size
                ed = st + self.block_size
                global_h_k[u, :, st:ed] = self.global_remainder[0][u, :, remainder_st:remainder_ed]
                global_h_v[u, :, st:ed] = self.global_remainder[1][u, :, remainder_st:remainder_ed]
        return ed

    # ------------------------------------------------------------------
    # Init KV loading
    # ------------------------------------------------------------------

    def get_global_hidden_and_mask(self, exc_length: int):
        """Prepare init KV in the global buffer; grow init_k/init_v if needed."""
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # Fill init KV until capacity
        if not self.init_exc and global_remainder_len > self.n_local:
            gk, gv = self.global_remainder
            append_len = self.n_init - self.init_k.size(-2)

            self.init_k = torch.cat(
                (self.init_k, gk[:, :, global_remainder_st : global_remainder_st + append_len, :]),
                dim=-2,
            )
            self.init_v = torch.cat(
                (self.init_v, gv[:, :, global_remainder_st : global_remainder_st + append_len, :]),
                dim=-2,
            )
            global_remainder_st += append_len
            global_remainder_len -= append_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # Copy init KV into buffer if needed
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:
            global_h_k[:, :, init_st:init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st:init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        return global_h_k[:, :, :init_ed, :], global_h_v[:, :, :init_ed, :]

    # ------------------------------------------------------------------
    # KV compression strategies
    # ------------------------------------------------------------------

    def _compress_tokens_by_similarity(
        self,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
        pooling_k: torch.Tensor,
        token_per_frame: int,
        keep_ratio: float = 0.5,
        random: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress tokens based on cosine similarity to historical pooling key.

        Retains the *keep_ratio* fraction of tokens with the **lowest**
        similarity (i.e., the most novel tokens).
        """
        batch_size, head, token_number, channel = current_k.shape
        num_group = self.num_heads // self.num_heads_kv

        # Expand to full head count for similarity computation
        group_k = (
            current_k.view(self.num_units, head, 1, token_number, channel)
            .expand(self.num_units, head, num_group, token_number, channel)
            .reshape(self.num_units, self.num_heads, token_number, channel)
        )
        reshaped_k = group_k.permute(0, 2, 1, 3).reshape(
            batch_size, token_number, self.num_heads * channel
        )

        keep_n = int(token_per_frame * keep_ratio)

        if not random:
            rk_norm = F.normalize(reshaped_k, p=2, dim=-1)
            pk_norm = F.normalize(pooling_k, p=2, dim=-1)
            similarity = torch.matmul(rk_norm, pk_norm.transpose(-2, -1)).squeeze(-1)
            _, topk_indices = torch.topk(similarity, k=keep_n, dim=-1, largest=False)
        else:
            noise = torch.rand(batch_size, token_number, device=current_k.device)
            _, topk_indices = torch.topk(noise, k=keep_n, dim=-1, largest=False)

        topk_indices, _ = torch.sort(topk_indices, dim=-1)
        expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(-1, head, -1, channel)
        return torch.gather(current_k, 2, expanded), torch.gather(current_v, 2, expanded)

    def _pack_kv_for_compression(self):
        """Reshape local KV and compute historical pooling key."""
        current_k, current_v = self.local_k, self.local_v
        batch_size, head, token_number, channel = current_k.shape
        num_group = self.num_heads // self.num_heads_kv

        group_k = (
            current_k.view(self.num_units, head, 1, token_number, channel)
            .expand(self.num_units, head, num_group, token_number, channel)
            .reshape(self.num_units, self.num_heads, token_number, channel)
        )
        reshaped_k = group_k.permute(0, 2, 1, 3).reshape(
            batch_size, token_number, self.num_heads * channel
        )

        # Historical mean as pooling key
        pooling_k = None
        for u in range(self.num_units):
            if len(self.block_k[u]) > 0:
                pooling_k = self.block_k[u].get_data().mean(dim=0, keepdim=True)

        return reshaped_k, current_k, current_v, pooling_k

    def dynamic_processor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the KV compression strategy specified by environment variable.

        Strategies (set ``KV_COMPRESSION_STRATEGY``):
            - ``full_kv``: No compression (default).
            - ``compress_video_tokens``: Similarity-based token pruning.
            - ``random_compress_video_tokens``: Random token pruning.
        """
        strategy = os.getenv("KV_COMPRESSION_STRATEGY", "full_kv")
        token_per_frame = int(os.getenv("TOKEN_PER_FRAME", "196"))
        keep_ratio = float(os.getenv("KV_COMPRESSION_RATIO", "0.5"))

        if strategy == "full_kv":
            return self.local_k, self.local_v

        reshaped_k, current_k, current_v, pooling_k = self._pack_kv_for_compression()
        random = strategy == "random_compress_video_tokens"

        if strategy in ("compress_video_tokens", "random_compress_video_tokens"):
            return self._compress_tokens_by_similarity(
                current_k, current_v, pooling_k,
                token_per_frame, keep_ratio, random=random,
            )

        raise ValueError(f"Invalid KV_COMPRESSION_STRATEGY: {strategy}")

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def _append(self, local_q, local_k, local_v, global_q):
        """Compute attention for one execution block."""
        stream = _get_global_stream() if self.async_global_stream else torch.cuda.current_stream()

        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v,
            get_score=False, sliding_window=self.n_local,
        )

        with torch.cuda.stream(stream):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(
                exc_length=global_q.size(-2)
            )

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(stream)

        attn.append(
            global_h_q, global_h_k, global_h_v,
            end=True, get_score=False,
            sliding_window=None, complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            stream.wait_stream(torch.cuda.current_stream())

        return o.view(self.batch_size, self.num_heads, -1, self.dim_head)

    def _append_global(self):
        """Offload processed context blocks to CPU."""
        ed = self._global_remainder_ed
        st = self._global_remainder_st
        remaining = ed - st

        if not self.init_exc:
            return

        assert remaining % self.block_size == 0

        while remaining > 0:
            remaining -= self.block_size

            # Create MemoryUnit and append to global_blocks
            for u in range(self.num_units):
                self.global_blocks[u].append(
                    MemoryUnit(
                        (
                            self.global_remainder[0][u, :, st : st + self.block_size, :],
                            self.global_remainder[1][u, :, st : st + self.block_size, :],
                        ),
                        self.cuda_cache,
                        False,
                        self.pin_memory,
                    )
                )

            # Compute and store block representative key
            block_k_raw = self.global_remainder[0][:, :, st : st + self.block_size, :]
            block_k = self._from_group_kv(block_k_raw)
            block_k_repr = self._get_block_representative(block_k)

            for u in range(self.num_units):
                self.block_k[u].append(block_k_repr[u])

            self.num_global_block += 1
            st += self.block_size

        self._global_remainder_ed = ed
        self._global_remainder_st = st

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ) -> torch.Tensor:
        """Process new KV pairs: compute attention, manage context memory.

        Returns:
            Attention output of shape ``[B, H, L, D]``.
        """
        if not self.initialized:
            self.init(local_q, local_k, local_v, global_q, global_k, global_v)

        input_length = local_q.size(-2)
        stream = _get_global_stream() if self.async_global_stream else torch.cuda.current_stream()

        if self.async_global_stream:
            stream.wait_stream(torch.cuda.current_stream())

        # Extend local KV
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # Extend global remainder
        with torch.cuda.stream(stream):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)
            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        # Apply RoPE to global query
        with torch.cuda.stream(stream):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        # Process in execution blocks
        o_list = []
        for st in range(0, input_length, self.exc_block_size):
            ed = min(st + self.exc_block_size, input_length)
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length

            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st:kv_ed, :],
                self.local_v[:, :, kv_st:kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            with torch.cuda.stream(stream):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(stream)

        # Optional KV compression
        use_compression = os.getenv("USE_KV_COMPRESSION", "no")
        if use_compression == "yes" and self.local_k.shape[-2] > 13:
            self.local_k, self.local_v = self.select_top_half_kv(
                self.local_k, self.local_v, o_list
            )

        self.length += input_length

        # Shrink local KV to window size
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :].contiguous()
            self.local_v = self.local_v[:, :, -self.n_local:, :].contiguous()

        # Shrink global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        with torch.cuda.stream(stream):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :].contiguous(),
                self.global_remainder[1][:, :, self._global_remainder_st:, :].contiguous(),
            )

        return torch.cat(o_list, dim=-2)

    # ------------------------------------------------------------------
    # Attention-based local KV compression
    # ------------------------------------------------------------------

    def select_top_half_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        attention_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-50% tokens per frame based on attention output magnitude.

        Args:
            local_k: ``[B, H_kv, T, D]``
            local_v: ``[B, H_kv, T, D]``
            attention_outputs: List of attention outputs per frame.

        Returns:
            ``(pruned_k, pruned_v)``
        """
        B, H, T, D = local_k.shape
        frame_num = len(attention_outputs)
        token_per_frame = attention_outputs[0].shape[-2]

        last_k = local_k[:, :, -frame_num * token_per_frame :, :]
        last_v = local_v[:, :, -frame_num * token_per_frame :, :]
        before_k = local_k[:, :, : -frame_num * token_per_frame, :]
        before_v = local_v[:, :, : -frame_num * token_per_frame, :]

        sel_k, sel_v = [], []
        half = math.ceil(token_per_frame / 2)

        for f_idx in range(frame_num):
            s = f_idx * token_per_frame
            e = s + token_per_frame
            k_frame = last_k[:, :, s:e, :]
            v_frame = last_v[:, :, s:e, :]

            attn = attention_outputs[f_idx]
            score = attn.mean(dim=(1, 3))  # [B, tokens]
            _, top_idx = torch.topk(score, k=half, dim=-1)
            top_idx = top_idx.unsqueeze(1).expand(-1, H, -1)

            sel_k.append(torch.gather(k_frame, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D)))
            sel_v.append(torch.gather(v_frame, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, D)))

        return (
            torch.cat([before_k] + sel_k, dim=2),
            torch.cat([before_v] + sel_v, dim=2),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def size(self, *args, **kwargs) -> int:
        return self.length

    def calculate_cpu_memory(self) -> int:
        """Total CPU memory used by offloaded blocks (bytes)."""
        return sum(
            block.calculate_cpu_memory()
            for u in range(self.num_units)
            for block in self.global_blocks[u]
        )
