# qwen3_5_moe source notes

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit verified with `git -C X:/H/transformers rev-parse HEAD`: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Source directory: `X:/H/transformers/src/transformers/models/qwen3_5_moe`
- Files inspected:
  - `configuration_qwen3_5_moe.py`
  - `modeling_qwen3_5_moe.py`
  - `modular_qwen3_5_moe.py`
  - `__init__.py`
- Related cache ABI source inspected:
  - `X:/H/transformers/src/transformers/cache_utils.py`
- The generated modeling file states it is generated from `src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py`; use `modular_qwen3_5_moe.py` for future Transformers source edits, but use the generated file for exact in-library runtime behavior.

## Representative config fetches

Fetched with `Invoke-RestMethod https://huggingface.co/{model_id}/raw/main/config.json` on 2026-05-13.

| Model id | model_type | architecture | hidden | layers | full/linear layers | heads / KV heads / head_dim | linear K/V heads | experts / top-k | MoE/shared intermediate | max positions | vision out | quantization |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.5-35B-A3B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 2048 | 40 | 10 / 30 | 16 / 2 / 256 | 16 / 32 | 256 / 8 | 512 / 512 | 262144 | 2048 | none |
| `Qwen/Qwen3.5-122B-A10B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 3072 | 48 | 12 / 36 | 32 / 2 / 256 | 16 / 64 | 256 / 8 | 1024 / 1024 | 262144 | 3072 | none |
| `Qwen/Qwen3.5-397B-A17B` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 4096 | 60 | 15 / 45 | 32 / 2 / 256 | 16 / 64 | 512 / 10 | 1024 / 1024 | 262144 | 4096 | none |
| `Qwen/Qwen3.5-122B-A10B-FP8` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 3072 | 48 | 12 / 36 | 32 / 2 / 256 | 16 / 64 | 256 / 8 | 1024 / 1024 | 262144 | 3072 | FP8 config, dynamic activations, `[128,128]` weight blocks, many modules excluded |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | 3072 | 48 | 12 / 36 | 32 / 2 / 256 | 16 / 64 | 256 / 8 | 1024 / 1024 | 262144 | 3072 | GPTQ int4, group size 128, attention/shared/visual/MTP excluded by dynamic rules |

Shared config facts from those snapshots:

- `dtype`: `bfloat16`
- `vocab_size`: `248320`
- text RoPE parameters: `rope_type="default"`, `rope_theta=10000000`, `partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `mrope_interleaved=true`
- `attention_bias=false`, `attention_dropout=0.0`, `hidden_act="silu"`
- `full_attention_interval=4` in public configs, yielding 3 linear-attention layers followed by 1 full-attention layer.
- `image_token_id=248056`, `video_token_id=248057`, `vision_start_token_id=248053`, `vision_end_token_id=248054`.

## Source-derived implementation notes

- Text config defaults are `hidden_size=2048`, `num_hidden_layers=40`, `num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=256`, `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_num_key_heads=16`, `linear_num_value_heads=32`, `num_experts=256`, `num_experts_per_tok=8`.
- `__post_init__` creates `layer_types` when absent by repeating `linear_attention` except each `full_attention_interval`th layer is `full_attention`.
- Full attention:
  - `q_proj`: `hidden_size -> num_attention_heads * head_dim * 2`; second half is an output gate, not key/value.
  - `k_proj`: `hidden_size -> num_key_value_heads * head_dim`.
  - `v_proj`: `hidden_size -> num_key_value_heads * head_dim`.
  - `o_proj`: `num_attention_heads * head_dim -> hidden_size`.
  - Q and K get per-head RMSNorm before RoPE.
  - Attention output is multiplied by `sigmoid(gate)` before `o_proj`.
- Linear attention (`Qwen3_5MoeGatedDeltaNet`):
  - `in_proj_qkv`: `hidden_size -> 2 * key_dim + value_dim`
  - `in_proj_z`: `hidden_size -> value_dim`
  - `in_proj_b`: `hidden_size -> linear_num_value_heads`
  - `in_proj_a`: `hidden_size -> linear_num_value_heads`
  - `key_dim = linear_num_key_heads * linear_key_head_dim`
  - `value_dim = linear_num_value_heads * linear_value_head_dim`
  - depthwise causal `Conv1d` over concatenated Q/K/V channels, kernel `linear_conv_kernel_dim`, `groups=conv_dim`, `padding=kernel-1`, followed by SiLU in fallback path.
  - recurrent state shape is `[batch, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`.
  - conv state shape is `[batch, conv_dim, linear_conv_kernel_dim]`.
- MoE:
  - Router weight: `[num_experts, hidden_size]`.
  - Router softmax computes in float, then top-k, renormalizes selected weights, casts to router dtype.
  - Expert packed `gate_up_proj`: `[num_experts, 2 * moe_intermediate_size, hidden_size]`; split order is `[gate, up]`.
  - Expert `down_proj`: `[num_experts, hidden_size, moe_intermediate_size]`.
  - Shared expert is a dense SwiGLU MLP with `shared_expert_intermediate_size` and a scalar `shared_expert_gate: hidden_size -> 1` sigmoid.
- Multimodal body:
  - Vision patch embed is `Conv3d(in_channels, hidden_size, kernel=stride=[temporal_patch_size, patch_size, patch_size])`, reshaped from processor-packed patches.
  - Vision transformer uses packed varlen attention over `cu_seqlens` derived from `grid_thw`.
  - Vision merger maps `vision_hidden * spatial_merge_size^2 -> text_hidden`.
  - Image/video features are inserted into token embeddings with `masked_scatter` after placeholder count validation.
- Cache:
  - `DynamicCache(config=...)` builds per-layer cache objects from `layer_types`.
  - `full_attention` uses dynamic KV layers.
  - `linear_attention` uses `LinearAttentionLayer` with fixed-size conv/recurrent state and no growing sequence length.
  - Cache reorder indexes both full-attention KV layers and linear-attention state tensors on batch dimension.

## Gated or not fully inspected sources

- No gated errors occurred for the five config JSONs above.
- Processor/tokenizer files were not fetched in this pass. Placeholder IDs and grid ABI were taken from config and modeling source, not processor source.
- Quantized checkpoint configs were inspected at config level only. Safetensors metadata and actual quantized tensor names/scales were not downloaded.
