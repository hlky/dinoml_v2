# LFM2 source notes

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/lfm2`
- Generated implementation: `modeling_lfm2.py` says it is generated from `modular_lfm2.py`; future source edits should inspect `modular_lfm2.py` first and use `modeling_lfm2.py` as the concrete generated runtime basis.

## Files inspected

- `configuration_lfm2.py`
  - `Lfm2Config` model type is `lfm2`.
  - Defaults: `hidden_size=2560`, `intermediate_size=12288`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=8`, `max_position_embeddings=128000`, `norm_eps=1e-5`, `use_cache=True`, `tie_word_embeddings=True`, `conv_bias=False`, `conv_L_cache=3`.
  - If `layer_types` is absent, the config builds it from `full_attn_idxs`; if both are absent, every layer becomes `full_attention`.
  - Back-compat aliases: `tie_embedding` feeds `tie_word_embeddings`; `block_ff_dim` feeds `intermediate_size`.
- `modeling_lfm2.py`
  - RMSNorm computes variance in fp32 over the last dim and multiplies by learned weight.
  - RoPE computes `inv_freq = 1 / rope_theta ** (arange(0, head_dim, 2) / head_dim)`, then builds `cos/sin` from position ids in fp32.
  - Attention is causal self-attention with GQA: q has `num_attention_heads`, k/v have `num_key_value_heads`; q and k each get per-head RMSNorm before RoPE.
  - Short-conv layers use `in_proj: hidden -> 3*hidden`, split into `B, C, x`, multiply `B*x`, run causal depthwise Conv1d with `groups=hidden_size`, then multiply by `C` and apply `out_proj`.
  - Forward builds one `DynamicCache(config=...)` when `use_cache` is true. Full-attention layers update KV cache; conv layers update `conv_states` through cache utilities.
- `modular_lfm2.py`
  - Authoritative modular file; imports pieces from Llama/Gemma2/Bamba before generation.
- `X:/H/transformers/src/transformers/cache_utils.py`
  - `layer_types` map `"full_attention"` to `DynamicLayer` and `"conv"` to `LinearAttentionLayer`.
  - Linear/conv cache owns `conv_states`; shape is initialized from the first conv-state tensor and updated in place to preserve static address for cudagraphs.
  - `reorder_cache` handles `conv_states` with `index_select(0, beam_idx)`; LFM2 does not use recurrent states.
- `X:/H/transformers/src/transformers/configuration_utils.py` and `modeling_rope_utils.py`
  - Legacy `rope_theta` is standardized into `rope_parameters`.
  - Default RoPE validation/standardization should be treated as source-owned config normalization, not an LFM2 modeling op.

## Representative config sweep

HF configs were fetched from `https://huggingface.co/{model_id}/raw/main/config.json` on 2026-05-13. These configs are useful for operator-significant variation but are not commit-pinned snapshots.

| Model id | model_type | Hidden | Layers | Attn heads | KV heads | Head dim | Attention layers | Conv layers | FF config | RoPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `LiquidAI/LFM2-350M` | `lfm2` | 1024 | 16 | 16 | 8 | 64 | 6 via `full_attn_idxs=[2,5,8,10,12,14]` | 10 | `block_ff_dim=6656`, auto-adjust true -> effective 4608 | legacy `rope_theta=1000000.0` |
| `LiquidAI/LFM2-700M` | `lfm2` | 1536 | 16 | 24 | 8 | 64 | same 6-index pattern | 10 | `block_ff_dim=10240`, auto-adjust true -> effective 6912 | legacy `rope_theta=1000000.0` |
| `LiquidAI/LFM2-1.2B` | `lfm2` | 2048 | 16 | 32 | 8 | 64 | same 6-index pattern | 10 | `block_ff_dim=12288`, auto-adjust true -> effective 8192 | legacy `rope_theta=1000000.0` |
| `LiquidAI/LFM2-2.6B` | `lfm2` | 2048 | 30 | 32 | 8 | 64 | 8 explicit `layer_types` attention layers | 22 | `intermediate_size=10752`, auto-adjust false | legacy `rope_theta=1000000.0`, `theta=1000000.0` ignored by modeling |
| `LiquidAI/LFM2.5-350M` | `lfm2` | 1024 | 16 | 16 | 8 | 64 | 6 explicit `layer_types` attention layers | 10 | `intermediate_size=6656`, auto-adjust true -> effective 4608 | `rope_parameters.default`, theta `1000000.0` |
| `LiquidAI/LFM2.5-1.2B-Thinking` | `lfm2` | 2048 | 16 | 32 | 8 | 64 | 6 explicit `layer_types` attention layers | 10 | `intermediate_size=12288`, auto-adjust true -> effective 8192 | legacy `rope_theta=1000000.0` |

## Important source caveats

- `LiquidAI/LFM2-24B-A2B` and `LiquidAI/LFM2-8B-A1B` use `model_type=lfm2_moe` and should be routed to the separate `lfm2_moe` audit, not this report.
- `lfm2_vl`, audio, ONNX, GGUF, MLX, and quantized repos may wrap or convert the base LFM2 body. This report covers the in-library PyTorch `lfm2` causal LM source only.
- No tokenizer, processor, generation config, safetensors metadata, GGUF metadata, or remote-code files were inspected for this audit.
- No DinoML tests, Transformers imports, or model instantiation were run.
