"""Example notes for integrating with HuggingFace models.

In real LLaMA/Qwen/Mistral inference, KV cache is usually produced inside the
attention module. The minimal integration idea is:

1. During prefill, collect past_key_value for each layer.
2. For historical context KV, call CocktailKVCompressor.compress().
3. Store compressed KV groups and chunk order.
4. During decode, replace standard `Q @ K.T -> softmax -> V` with:
   - group-wise Q @ K_group.T;
   - concatenate all logits;
   - global softmax;
   - group-wise probability @ V_group;
   - sum outputs.
5. Keep query tokens and newly generated tokens in FP16 if accuracy protection is needed.

This file intentionally avoids monkey-patching a specific transformers version,
because attention signatures change frequently across models.
"""
