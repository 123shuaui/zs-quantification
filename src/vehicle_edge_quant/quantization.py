"""Basic asymmetric RTN quantization utilities.

The tensors are stored in int8 containers for readability. For real deployment,
INT2/INT4 should be packed and computed by custom kernels.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class QuantizedTensor:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    original_shape: Tuple[int, ...]


def _reduce_dims(x: torch.Tensor, group_dim: Optional[int]):
    if group_dim is None:
        return tuple(range(x.ndim))
    return tuple(d for d in range(x.ndim) if d != group_dim)


def quantize_tensor(x: torch.Tensor, bits: int = 4, group_dim: Optional[int] = None) -> QuantizedTensor:
    """Signed asymmetric round-to-nearest quantization.

    Args:
        x: FP tensor.
        bits: 2 / 4 / 8 / 16. bits=16 returns a fake quantized wrapper.
        group_dim: if given, keep this dimension and compute scale per group.
    """
    if bits == 16:
        return QuantizedTensor(
            q=x.to(torch.float16),
            scale=torch.ones((), device=x.device, dtype=x.dtype),
            zero_point=torch.zeros((), device=x.device, dtype=x.dtype),
            bits=16,
            original_shape=tuple(x.shape),
        )

    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4, 8, or 16, got {bits}")

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    reduce_dims = _reduce_dims(x, group_dim)
    xmin = x.amin(dim=reduce_dims, keepdim=True)
    xmax = x.amax(dim=reduce_dims, keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-6) / float(qmax - qmin)
    zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax).to(torch.int8)

    return QuantizedTensor(q=q, scale=scale, zero_point=zero_point, bits=bits, original_shape=tuple(x.shape))


def dequantize_tensor(qt: QuantizedTensor) -> torch.Tensor:
    if qt.bits == 16:
        return qt.q.to(torch.float16)
    return (qt.q.float() - qt.zero_point.float()) * qt.scale.float()


def estimate_storage_bytes(num_elements: int, bits: int, metadata_ratio: float = 0.03) -> float:
    """Estimate compressed tensor storage in bytes."""
    if bits == 16:
        return num_elements * 2
    return num_elements * bits / 8.0 * (1.0 + metadata_ratio)
