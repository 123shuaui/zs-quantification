"""Chunk-level mixed-precision KV cache compression.

This module follows the idea of Cocktail and the thesis method:
1. split historical KV cache into chunks;
2. score each chunk using semantic relevance, recency, and structural importance;
3. assign FP16 / INT4 / INT2 under a KV budget;
4. reorder chunks by bitwidth;
5. compute attention with group-wise logits but global softmax.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .quantization import QuantizedTensor, dequantize_tensor, estimate_storage_bytes, quantize_tensor


@dataclass
class ChunkInfo:
    index: int
    start: int
    end: int
    score: float
    bit: int


@dataclass
class CompressedKV:
    groups_k: Dict[int, List[QuantizedTensor]]
    groups_v: Dict[int, List[QuantizedTensor]]
    chunks: List[ChunkInfo]
    order: List[int]


class CocktailKVCompressor:
    def __init__(
        self,
        chunk_size: int = 32,
        bitwidths: Sequence[int] = (16, 4, 2),
        tau_high: float = 0.70,
        tau_low: float = 0.35,
        alpha: float = 0.55,
        beta: float = 0.25,
        gamma: float = 0.20,
    ):
        self.chunk_size = chunk_size
        self.bitwidths = tuple(bitwidths)
        self.high_bit, self.mid_bit, self.low_bit = self.bitwidths
        self.tau_high = tau_high
        self.tau_low = tau_low
        s = alpha + beta + gamma
        self.alpha, self.beta, self.gamma = alpha / s, beta / s, gamma / s

    def split_chunks(self, x: torch.Tensor) -> List[Tuple[int, int, torch.Tensor]]:
        """Split [T, H, D] tensor into chunk list."""
        chunks = []
        total = x.shape[0]
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            chunks.append((start, end, x[start:end]))
        return chunks

    def compute_scores(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        structural_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunk importance score I_i = alpha*S_i + beta*R_i + gamma*P_i.

        Args:
            query: [H, D] or [D] current query representation.
            key_cache: [T, H, D] historical K cache.
            structural_scores: optional [num_chunks], values in [0, 1].
        """
        chunks = self.split_chunks(key_cache)
        q = query.float().reshape(-1)
        semantic = []
        for _, _, kc in chunks:
            # mean key as chunk representation
            c = kc.float().mean(dim=(0, 1)).reshape(-1)
            semantic.append(F.cosine_similarity(q[: c.numel()].unsqueeze(0), c.unsqueeze(0)).item())
        semantic = torch.tensor(semantic, device=key_cache.device)
        semantic = (semantic - semantic.min()) / (semantic.max() - semantic.min() + 1e-6)

        # Recency: newer chunks get larger score.
        n = len(chunks)
        pos = torch.arange(n, device=key_cache.device, dtype=torch.float32)
        distance = (n - 1 - pos).clamp_min(0)
        recency = torch.exp(-distance / max(n / 3, 1.0))
        recency = (recency - recency.min()) / (recency.max() - recency.min() + 1e-6)

        if structural_scores is None:
            structural = torch.zeros(n, device=key_cache.device)
            # protect first chunk as prompt/task instruction and last chunk as local context
            if n > 0:
                structural[0] = 1.0
                structural[-1] = 1.0
        else:
            structural = structural_scores.to(key_cache.device).float().clamp(0, 1)

        score = self.alpha * semantic + self.beta * recency + self.gamma * structural
        return score

    def assign_bits(self, scores: torch.Tensor) -> List[int]:
        bits = []
        for s in scores.tolist():
            if s >= self.tau_high:
                bits.append(self.high_bit)
            elif s < self.tau_low:
                bits.append(self.low_bit)
            else:
                bits.append(self.mid_bit)
        return bits

    def enforce_budget(
        self,
        bits: List[int],
        scores: torch.Tensor,
        chunk_numel: Sequence[int],
        budget_bytes: float,
    ) -> List[int]:
        """Downgrade low-score chunks until the estimated KV storage meets budget."""
        def total_bytes(cur_bits):
            return sum(estimate_storage_bytes(n, b) * 2 for n, b in zip(chunk_numel, cur_bits))

        bits = list(bits)
        if total_bytes(bits) <= budget_bytes:
            return bits

        # Low score chunks are downgraded first: 16 -> 4 -> 2 or 4 -> 2.
        order = torch.argsort(scores).tolist()
        changed = True
        while total_bytes(bits) > budget_bytes and changed:
            changed = False
            for idx in order:
                if bits[idx] == self.high_bit:
                    bits[idx] = self.mid_bit
                    changed = True
                elif bits[idx] == self.mid_bit:
                    bits[idx] = self.low_bit
                    changed = True
                if total_bytes(bits) <= budget_bytes:
                    break
        return bits

    def compress(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        query: torch.Tensor,
        budget_bytes: float | None = None,
        structural_scores: torch.Tensor | None = None,
    ) -> CompressedKV:
        k_chunks = self.split_chunks(key_cache)
        v_chunks = self.split_chunks(value_cache)
        scores = self.compute_scores(query, key_cache, structural_scores)
        bits = self.assign_bits(scores)

        if budget_bytes is not None:
            chunk_numel = [kc.numel() for _, _, kc in k_chunks]
            bits = self.enforce_budget(bits, scores, chunk_numel, budget_bytes)

        groups_k: Dict[int, List[QuantizedTensor]] = {b: [] for b in self.bitwidths}
        groups_v: Dict[int, List[QuantizedTensor]] = {b: [] for b in self.bitwidths}
        chunks: List[ChunkInfo] = []
        for i, ((s, e, kc), (_, _, vc), bit) in enumerate(zip(k_chunks, v_chunks, bits)):
            groups_k[bit].append(quantize_tensor(kc, bit))
            groups_v[bit].append(quantize_tensor(vc, bit))
            chunks.append(ChunkInfo(index=i, start=s, end=e, score=float(scores[i]), bit=bit))

        # Physical reorder: same-bit chunks are contiguous. Logical index is recorded.
        order = []
        for bit in self.bitwidths:
            order.extend([c.index for c in chunks if c.bit == bit])
        return CompressedKV(groups_k=groups_k, groups_v=groups_v, chunks=chunks, order=order)

    def attention(self, query: torch.Tensor, compressed: CompressedKV) -> torch.Tensor:
        """Equivalent attention: group-wise logits construction + global softmax.

        Args:
            query: [H, D]
        Returns:
            output: [H, D]
        """
        logits_parts = []
        value_parts = []
        scale = query.shape[-1] ** -0.5

        for bit in self.bitwidths:
            if len(compressed.groups_k[bit]) == 0:
                continue
            k = torch.cat([dequantize_tensor(x).float() for x in compressed.groups_k[bit]], dim=0)  # [Tg,H,D]
            v = torch.cat([dequantize_tensor(x).float() for x in compressed.groups_v[bit]], dim=0)
            # logits for each head: [H, Tg]
            logits = torch.einsum("hd,thd->ht", query.float(), k) * scale
            logits_parts.append(logits)
            value_parts.append(v)

        logits_all = torch.cat(logits_parts, dim=-1)
        probs = torch.softmax(logits_all, dim=-1)

        outputs = []
        offset = 0
        for v in value_parts:
            length = v.shape[0]
            p = probs[:, offset : offset + length]  # [H,Tg]
            outputs.append(torch.einsum("ht,thd->hd", p, v))
            offset += length
        return sum(outputs)
