# ERNIE 4.5 Source Notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit verified with `git rev-parse HEAD`: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Inspected files:
  - `src/transformers/models/ernie4_5/configuration_ernie4_5.py`
  - `src/transformers/models/ernie4_5/modeling_ernie4_5.py`
  - `src/transformers/models/ernie4_5/modular_ernie4_5.py`
  - `src/transformers/models/ernie4_5/convert_ernie4_5_tokenizer.py`
  - Shared helpers by grep only: `modeling_rope_utils.py`, `masking_utils.py`, `cache_utils.py`

## Source observations

- `modeling_ernie4_5.py` is generated from `modular_ernie4_5.py`; the generated file contains the complete concrete classes used for this audit.
- The in-tree `ernie4_5` source is a dense decoder-only causal LM, not the MoE or VL variants. Public configs for `ernie4_5_moe` and `ernie4_5_moe_vl` were sampled only as family-variation traps.
- Dense attention uses separate `q_proj`, `k_proj`, `v_proj`, `o_proj` linear layers. Projection widths are explicit:
  - Q: `hidden_size -> num_attention_heads * head_dim`
  - K/V: `hidden_size -> num_key_value_heads * head_dim`
  - O: `num_attention_heads * head_dim -> hidden_size`
- The default dense config has `hidden_size=1024`, `num_attention_heads=16`, `num_key_value_heads=2`, and `head_dim=128`, so `num_attention_heads * head_dim = 2048`, not `hidden_size`.
- RoPE is computed in float32 from `rope_parameters["rope_theta"]` and `head_dim`. The apply function uses GLM-style interleaving: cos/sin are sliced to half dim then `repeat_interleave(2)`, and `rotate_half` uses even/odd pairs.
- Cache update happens after RoPE: cached K tensors are already position-encoded; V tensors are unrotated projected values.
- Eager attention repeats KV heads before matmul, adds the causal/padding mask before softmax, computes softmax in float32, casts to query dtype, then multiplies by V.
- `ALL_ATTENTION_FUNCTIONS` can dispatch to eager, SDPA, FlashAttention, or FlexAttention depending on config/backend, but the dense source fallback is ordinary causal GQA attention.
- No model-specific quantization/packed-weight code was present in `ernie4_5`.

## Representative HF configs checked

Raw config URLs fetched without importing Transformers:

- `https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT/raw/main/config.json`

All sampled raw configs were accessible. The MoE and VL configs require other `model_type`s and source files outside the requested local `ernie4_5` directory.
