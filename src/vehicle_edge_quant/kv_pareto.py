"""KV Pareto style configuration search."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .memory_model import ModelSpec, kv_memory_gb, prefill_peak_gb, total_memory_gb, weight_memory_gb


@dataclass
class ConfigResult:
    name: str
    weight_bits: int
    k_bits: int
    v_bits: int
    chunk_size: int
    memory_gb: float
    accuracy_proxy: float
    ttft_proxy: float
    tpot_proxy: float


class ParetoSearcher:
    def __init__(
        self,
        spec: ModelSpec,
        seq_len: int = 4096,
        weight_bits_candidates=(16, 8, 4),
        kv_candidates=((16, 16), (8, 8), (8, 4), (4, 4), (4, 2), (2, 2)),
        chunk_candidates=(64, 128, 256, 512, 1024),
    ):
        self.spec = spec
        self.seq_len = seq_len
        self.weight_bits_candidates = weight_bits_candidates
        self.kv_candidates = kv_candidates
        self.chunk_candidates = chunk_candidates

    def evaluate_proxy(self, weight_bits: int, k_bits: int, v_bits: int, chunk_size: int) -> ConfigResult:
        mem = total_memory_gb(self.spec, self.seq_len, weight_bits, k_bits, v_bits, chunk_size)

        # Quality proxy: lower bitwidth introduces degradation. This is only for code-flow demo.
        weight_penalty = {16: 0.0, 8: 0.005, 4: 0.025}[weight_bits]
        kv_penalty = (16 - k_bits) * 0.0015 + (16 - v_bits) * 0.0018
        chunk_penalty = 0.0 if chunk_size <= 256 else (chunk_size - 256) / 1024 * 0.01
        accuracy = max(0.0, 0.70 - weight_penalty - kv_penalty - chunk_penalty)

        # Larger chunks usually reduce repeated launch overhead but may increase peak memory.
        ttft = 1.0 + self.seq_len / max(chunk_size, 1) * 0.015 + prefill_peak_gb(self.seq_len, chunk_size) * 0.02
        tpot = 1.0 + kv_memory_gb(self.spec, self.seq_len, k_bits, v_bits) * 0.08

        name = f"w{weight_bits}a16_k{k_bits}v{v_bits}_chunk{chunk_size}"
        return ConfigResult(name, weight_bits, k_bits, v_bits, chunk_size, mem, accuracy, ttft, tpot)

    def search(self, memory_budget_gb: float | None = None) -> List[ConfigResult]:
        results = []
        for wb in self.weight_bits_candidates:
            for kb, vb in self.kv_candidates:
                for c in self.chunk_candidates:
                    r = self.evaluate_proxy(wb, kb, vb, c)
                    if memory_budget_gb is None or r.memory_gb <= memory_budget_gb:
                        results.append(r)
        return results

    @staticmethod
    def pareto_frontier(results: List[ConfigResult]) -> List[ConfigResult]:
        """Maximize accuracy, minimize memory. Keep non-dominated points."""
        frontier = []
        for r in results:
            dominated = False
            for q in results:
                if (
                    q.memory_gb <= r.memory_gb
                    and q.accuracy_proxy >= r.accuracy_proxy
                    and (q.memory_gb < r.memory_gb or q.accuracy_proxy > r.accuracy_proxy)
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(r)
        return sorted(frontier, key=lambda x: x.memory_gb)
