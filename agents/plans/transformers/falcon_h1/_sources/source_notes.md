# Falcon H1 Source Notes

## Scope

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family path: `src/transformers/models/falcon_h1`
- Report target: DinoML causal language-model inference for native `FalconH1ForCausalLM`.

## Local source files inspected

- `configuration_falcon_h1.py`
- `modeling_falcon_h1.py`
- `modular_falcon_h1.py`
- `convert_mamba_ssm_checkpoint.py`
- `cache_utils.py` for shared `DynamicCache` / `LinearAttentionLayer` state behavior.

`modeling_falcon_h1.py` is generated from `modular_falcon_h1.py`; the generated file is the runtime import target, but future Transformers source edits should be made in the modular source.

## HF configs sampled

Fetched via direct `https://huggingface.co/{model_id}/raw/main/config.json` reads on 2026-05-13:

- `tiiuae/Falcon-H1-Tiny-R-90M`
- `tiiuae/Falcon-H1-0.5B-Base`
- `tiiuae/Falcon-H1-0.5B-Instruct`
- `tiiuae/Falcon-H1-1.5B-Base`
- `tiiuae/Falcon-H1-1.5B-Instruct`
- `tiiuae/Falcon-H1-1.5B-Deep-Base`
- `tiiuae/Falcon-H1-1.5B-Deep-Instruct`
- `tiiuae/Falcon-H1-3B-Base`
- `tiiuae/Falcon-H1-7B-Base`
- `tiiuae/Falcon-H1-7B-Instruct`
- `tiiuae/Falcon-H1-34B-Base`
- `tiiuae/Falcon-H1-0.5B-Instruct-GPTQ-Int4`

No gated source was required for these config reads. We did not fetch weights or run model imports/tests.

## Key source-derived facts

- Every native layer is `layers_block_type == "hybrid"` and contains both `FalconH1Mixer` and `FalconH1Attention`.
- Attention uses separate dense projections, not a fused runtime `qkv` module:
  - `q_proj: hidden_size -> num_attention_heads * head_dim`
  - `k_proj/v_proj: hidden_size -> num_key_value_heads * head_dim`
  - `o_proj: num_attention_heads * head_dim -> hidden_size`
- The code uses explicit config `head_dim` when present. Some configs have `hidden_size != num_attention_heads * head_dim`; `o_proj` bridges the attention output width back to `hidden_size`.
- Mamba2-style mixer uses:
  - `intermediate_size = mamba_d_ssm` when set, otherwise `mamba_expand * hidden_size`
  - `conv_dim = intermediate_size + 2 * mamba_n_groups * mamba_d_state`
  - `in_proj: hidden_size -> intermediate_size + conv_dim + mamba_n_heads`
  - split order in normal inference path: `gate, hidden_states_B_C, dt`
  - split order inside `hidden_states_B_C`: `hidden_states, B, C`
- Cache is the shared Transformers `DynamicCache` with mixed layer type support. For Falcon H1 configs, layer type maps to `hybrid`, which creates one layer capable of attention KV plus linear-attention state.
- Linear-attention state layer owns:
  - `conv_states`: `[batch, conv_dim, mamba_d_conv]`
  - `recurrent_states`: `[batch, mamba_n_heads, mamba_d_head, mamba_d_state]`
  - `has_previous_state` flag used to choose decode-step path when `seq_len == 1`.
- Attention KV cache stores RoPE-applied keys and raw values before GQA repeat expansion:
  - key/value per layer: `[batch, num_key_value_heads, cached_seq, head_dim]`.
- Mamba fast path depends on optional hub kernels: `mamba-ssm` and `causal-conv1d`. Eager fallback is source-present but likely too slow for production.
- There is no MoE in native Falcon H1 source.

## Source gaps / cautions

- `configuration_falcon_h1.py` defines `rope_parameters`, while representative configs use `rope_theta`. Transformers config normalization likely maps legacy/top-level RoPE fields, but this was not verified by import because the task forbids tests/imports.
- `convert_mamba_ssm_checkpoint.py` passes fields such as `head_dim` and `rope_theta` that are not class annotations in the inspected config file. Treat this as source compatibility surface to verify during config-loader work.
- GPTQ config exists in HF repos, but native modeling source does not implement GPTQ packed matmul itself; this is a loader/provider admission issue.
