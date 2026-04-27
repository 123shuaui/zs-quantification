import argparse
import json
from pathlib import Path

import torch

from .cocktail_kv import CocktailKVCompressor
from .kv_pareto import ParetoSearcher
from .memory_model import ModelSpec


def run_cocktail_demo():
    torch.manual_seed(7)
    seq_len, heads, dim = 512, 8, 64
    key_cache = torch.randn(seq_len, heads, dim, dtype=torch.float16)
    value_cache = torch.randn(seq_len, heads, dim, dtype=torch.float16)
    query = torch.randn(heads, dim, dtype=torch.float16)

    compressor = CocktailKVCompressor(chunk_size=32)
    compressed = compressor.compress(
        key_cache=key_cache,
        value_cache=value_cache,
        query=query,
        budget_bytes=seq_len * heads * dim * 2 * 2 * 0.45,  # about 45% of FP16 KV
    )
    out = compressor.attention(query, compressed)

    counts = {}
    for c in compressed.chunks:
        counts[c.bit] = counts.get(c.bit, 0) + 1

    print("=== Cocktail KV demo ===")
    print("chunk bitwidth counts:", counts)
    print("physical reorder order(first 20):", compressed.order[:20])
    print("attention output shape:", tuple(out.shape))
    print("attention output checksum:", float(out.float().sum()))


def run_pareto_demo():
    spec = ModelSpec(params_billion=7.0, num_layers=32, num_heads=32, head_dim=128, batch_size=1)
    searcher = ParetoSearcher(spec=spec, seq_len=4096)
    results = searcher.search(memory_budget_gb=24.0)
    frontier = searcher.pareto_frontier(results)

    print("=== KV Pareto demo ===")
    print(f"valid configs: {len(results)}")
    print("Pareto frontier:")
    for r in frontier[:12]:
        print(
            f"{r.name:24s} | mem={r.memory_gb:6.2f} GB | "
            f"acc_proxy={r.accuracy_proxy:.4f} | ttft={r.ttft_proxy:.3f} | tpot={r.tpot_proxy:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cocktail", "pareto"], default="cocktail")
    args = parser.parse_args()

    if args.mode == "cocktail":
        run_cocktail_demo()
    else:
        run_pareto_demo()


if __name__ == "__main__":
    main()
