"""Memory model used by the Pareto search."""
from dataclasses import dataclass

from .quantization import estimate_storage_bytes


@dataclass
class ModelSpec:
    params_billion: float = 7.0
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    batch_size: int = 1


def weight_memory_gb(spec: ModelSpec, weight_bits: int) -> float:
    params = spec.params_billion * 1e9
    return estimate_storage_bytes(int(params), weight_bits, metadata_ratio=0.02) / (1024 ** 3)


def kv_memory_gb(spec: ModelSpec, seq_len: int, k_bits: int, v_bits: int) -> float:
    elems = spec.batch_size * spec.num_layers * spec.num_heads * spec.head_dim * seq_len
    k_bytes = estimate_storage_bytes(elems, k_bits, metadata_ratio=0.03)
    v_bytes = estimate_storage_bytes(elems, v_bits, metadata_ratio=0.03)
    return (k_bytes + v_bytes) / (1024 ** 3)


def prefill_peak_gb(seq_len: int, chunk_size: int, hidden_size: int = 4096, layers: int = 32) -> float:
    """A simple peak-memory proxy.

    It is not a hardware profiler. It captures the monotonic relation:
    larger chunk size -> larger activation/attention peak.
    """
    active_tokens = min(seq_len, chunk_size)
    # activation-like term and attention-like term
    act = active_tokens * hidden_size * layers * 2
    att = active_tokens * active_tokens * layers * 2
    runtime_buffer = 0.8 * (1024 ** 3)
    return (act + att + runtime_buffer) / (1024 ** 3)


def total_memory_gb(spec: ModelSpec, seq_len: int, weight_bits: int, k_bits: int, v_bits: int, chunk_size: int) -> float:
    return (
        weight_memory_gb(spec, weight_bits)
        + kv_memory_gb(spec, seq_len, k_bits, v_bits)
        + prefill_peak_gb(seq_len, chunk_size, hidden_size=spec.num_heads * spec.head_dim, layers=spec.num_layers)
    )
