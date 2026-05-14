# OLMoE Source Notes

Source basis:
- Transformers checkout: `X:/H/transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Main source: `src/transformers/models/olmoe/modeling_olmoe.py`.
- Modular source: `src/transformers/models/olmoe/modular_olmoe.py`.
- Config source: `src/transformers/models/olmoe/configuration_olmoe.py`.
- Conversion helper: `src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py`.

Authoritative source relationship:
- `modeling_olmoe.py` states that it is generated from `modular_olmoe.py`; future upstream edits should be made to `modular_olmoe.py`.
- The generated file is still the runtime implementation imported by the package and was used for exact line-level behavior.

Important source anchors:
- Config defaults: `configuration_olmoe.py:58-81`.
- TP plan and packed expert hints: `configuration_olmoe.py:47-55`.
- RMSNorm float32 accumulation then cast back: `modeling_olmoe.py:48-63`.
- RoPE inverse-frequency and cos/sin generation: `modeling_olmoe.py:69-131`.
- RoPE apply and rotate-half layout: `modeling_olmoe.py:150-180`.
- Eager GQA/MHA attention fallback: `modeling_olmoe.py:183-217`.
- Attention projection dimensions, q/k RMSNorm, optional qkv clamp, RoPE-before-cache update: `modeling_olmoe.py:220-298`.
- Expert packed tensors and source eager routing/scatter-add loop: `modeling_olmoe.py:301-338`.
- Router softmax/top-k and optional top-k renormalization: `modeling_olmoe.py:341-359`.
- Decoder block residual order: `modeling_olmoe.py:378-416`.
- Model embedding, `DynamicCache`, position IDs, causal mask, shared position embeddings: `modeling_olmoe.py:447-519`.
- Causal LM logits slicing with `logits_to_keep`: `modeling_olmoe.py:604-705`.
- Legacy conversion splits original fused QKV rows as `[Q, K, V]`: `convert_olmoe_weights_to_hf.py:126-136`.
- Legacy conversion transposes original down-projection expert weights before HF save: `convert_olmoe_weights_to_hf.py:146-157`.

Representative config snapshots:
- Stored in `config_sweep.json`.
- Fetched from official Hugging Face model repos on 2026-05-13.
- All inspected official configs share the same runtime-significant topology: 16 layers, hidden size 2048, 16 Q heads, 16 KV heads, head dim inferred as 128, 64 experts, top-8 routing, expert intermediate size 1024, bf16 weights, 4096 context, no attention bias, no qkv clamp, no top-k renorm.

Source gaps / cautions:
- Current source exposes `_supports_flash_attn` and `_supports_sdpa`, but DinoML should define its own attention admission around causal self-attention with RoPE-applied K cache and MHA/GQA dimensions.
- `sliding_window` is passed to the attention interface if present, but the model-level causal mask comment says no sliding; inspected official configs do not set `sliding_window`.
- Config class defaults include `intermediate_size=2048`, while official checkpoints use `intermediate_size=1024`.
- Current config class uses `rope_parameters`; inspected checkpoint JSONs use historical `rope_theta` / `rope_scaling` fields. Treat the effective in-library normalization as a config-loading contract, not a separate graph op.
