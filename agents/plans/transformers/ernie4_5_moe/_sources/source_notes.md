# ernie4_5_moe source notes

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- `git log -1 --oneline`: `b75feb2af6 fix(minicpmv4_6): skip invalid failing tests (#45836)`
- DinoML report target: `H:/dinoml_v2/agents/plans/transformers/ernie4_5_moe/report.md`
- No imports, model execution, or DinoML tests were run.

## Files inspected

- `src/transformers/models/ernie4_5_moe/configuration_ernie4_5_moe.py`
- `src/transformers/models/ernie4_5_moe/modular_ernie4_5_moe.py`
- `src/transformers/models/ernie4_5_moe/modeling_ernie4_5_moe.py`
- `src/transformers/integrations/moe.py`
- Related shared attention/cache helpers were identified by import, but this audit did not expand into full cache/masking implementation internals.

`modeling_ernie4_5_moe.py` says it is generated from `modular_ernie4_5_moe.py`; future source edits should target the modular file. The generated file is still the concrete in-library runtime file inspected for exact forward behavior.

## Official config snapshots inspected

Fetched from Hugging Face raw `config.json` on 2026-05-13:

- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-PT/raw/main/config.json`
- `https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-Base-PT/raw/main/config.json`

The configs were accessible without gated auth. Weight files and tokenizer files were not downloaded; this report is based on model source plus config dimensions.

## High-signal source facts

- Generated modeling file header: `modeling_ernie4_5_moe.py` is generated from `modular_ernie4_5_moe.py`.
- Config defaults: `hidden_size=2560`, `num_hidden_layers=28`, `num_attention_heads=20`, `num_key_value_heads=4`, `moe_num_experts=64`, `moe_k=6`, `moe_num_shared_experts=2`, `max_position_embeddings=131072`, `default_theta=500000.0`.
- Attention: separate `q_proj`, `k_proj`, `v_proj`, `o_proj`; GQA with `num_key_value_groups = num_attention_heads // num_key_value_heads`; RoPE applied to Q/K before cache update.
- RoPE: ERNIE uses "glm rope style" full-dim interleaving: cos/sin are truncated to half, repeated with `repeat_interleave(2)`, then combined with `rotate_half`.
- Cache: if `use_cache=True` and no cache is supplied, source constructs `DynamicCache(config=self.config)`; `past_key_values.update(key_states, value_states, layer_idx)` stores post-RoPE K and raw V.
- MoE router: router logits use fp32 `F.linear(hidden_states.float(), weight.float())`; softmax over experts; `moe_statics` adds `e_score_correction_bias`; top-k on corrected scores; gather original routing weights; renormalize by clamped top-k sum.
- MoE experts eager path: expert weights are `gate_up_proj[num_experts, 2*moe_intermediate_size, hidden]` and `down_proj[num_experts, hidden, moe_intermediate_size]`. Eager path one-hots top-k routing, loops active experts, gathers token rows, does gate/up linear then activation multiply, down linear, scales by route weight, and `index_add_` accumulates.
- MoE optimized integrations: `@use_experts_implementation` can dispatch `batched_mm`, `grouped_mm`, or `sonicmoe`. The grouped implementation sorts token-expert rows, uses `histc`/cumsum offsets, grouped matrix multiplication, unsorts, then reshapes and sums per token. It differs from eager `index_add_` in accumulation order and often fp32 stability.
- Heads: `Ernie4_5_MoeForCausalLM` uses `logits_to_keep` slicing before `lm_head`; tied embeddings are config-dependent, with `_tied_weights_keys` for `lm_head.weight`.
- Training-only or optional: load-balancing auxiliary loss only matters when `output_router_logits` and labels/loss are requested. `_keys_to_ignore_on_load_unexpected = ["mtp"]`; config `num_nextn_predict_layers` is not implemented in this source.

## Config variation notes

- 21B A3B PT/Base/Thinking: hidden 2560, 28 layers, 20 query heads, 4 KV heads, head_dim inferred 128, MoE layers 1-27 by default, 64 experts, top-k 6, shared experts 2, tied embeddings true, bf16.
- 300B A47B PT/Base: hidden 8192, 54 layers, 64 query heads, 8 KV heads, head_dim inferred 128, MoE layers 3-53, 64 experts, top-k 8, no shared experts, tied embeddings false, bf16.
- Thinking config includes historical or implementation-specific fields `moe_capacity`, `moe_gate`, `moe_use_aux_free`; the inspected in-library source does not read these fields.
- 21B/300B PT configs include `num_nextn_predict_layers=1`; inspected source ignores MTP keys.

