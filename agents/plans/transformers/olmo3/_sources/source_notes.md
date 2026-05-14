# OLMo3 Source Notes

Scope: Transformers `olmo3` source audit for DinoML planning. No DinoML tests or imports were run.

## Local source

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Runtime files:
  - `src/transformers/models/olmo3/modeling_olmo3.py`
  - `src/transformers/models/olmo3/configuration_olmo3.py`
  - `src/transformers/models/olmo3/modular_olmo3.py`
  - `src/transformers/models/olmo3/convert_olmo3_weights_to_hf.py`
  - Shared helpers inspected where behavior is delegated:
    - `src/transformers/modeling_rope_utils.py`
    - `src/transformers/masking_utils.py`
    - `src/transformers/cache_utils.py`
    - `src/transformers/configuration_utils.py`

`modeling_olmo3.py` and `configuration_olmo3.py` are generated from `modular_olmo3.py`; the generated files are the runtime source basis, while modular is the upstream edit authority.

## Hugging Face configs fetched

Fetched via raw/config API on 2026-05-13:

- `allenai/Olmo-3-7B-Instruct`, model API sha `096bb5469fe34348bc88d851a69edb3bf6f40df4`
- `allenai/Olmo-3-7B-Think`, model API sha `7c991fde2671813ab41745054310f60a610b0fac`
- `allenai/Olmo-3-32B-Think`, model API sha `ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`

Gated or unavailable through anonymous raw access:

- `allenai/Olmo-3-32B-Instruct`: 401 Unauthorized
- `allenai/Olmo-3-7B`: 401 Unauthorized
- `allenai/Olmo-3-32B`: 401 Unauthorized

Access to those gated repos would resolve whether their configs differ from the accessible instruct/think variants.

## Config highlights

Accessible configs all use:

- `architectures=["Olmo3ForCausalLM"]`
- `model_type="olmo3"`
- `hidden_act="silu"`
- `attention_bias=false`
- `attention_dropout=0.0`
- `rms_norm_eps=1e-6`
- `max_position_embeddings=65536`
- `sliding_window=4096`
- `rope_scaling`/`rope_theta` legacy fields, normalized by `PreTrainedConfig` into `rope_parameters`
- YaRN RoPE with `factor=8.0`, `original_max_position_embeddings=8192`, `rope_theta=500000`, `attention_factor=1.2079441541679836`
- `tie_word_embeddings=false`
- `vocab_size=100278`
- `use_cache=false` in config, despite source support for `use_cache=True`

Accessible shape differences:

- 7B variants: `hidden_size=4096`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=32`, `intermediate_size=11008`.
- 32B Think: `hidden_size=5120`, `num_hidden_layers=64`, `num_attention_heads=40`, `num_key_value_heads=8`, `intermediate_size=27648`.

## Source behavior notes

- `Olmo3Config.__post_init__` defaults `num_key_value_heads` to `num_attention_heads`.
- If `layer_types` is omitted, layers are `sliding_attention` except every 4th layer is `full_attention`.
- `Olmo3Attention` derives `head_dim` from explicit `config.head_dim` if present, else `hidden_size // num_attention_heads`.
- Q/K/V projections are separate linear modules. Q and K are RMSNormed immediately after projection and before reshape/RoPE/cache update. V is not normalized.
- Q/K RoPE is applied before cache update, so cached keys are post-RoPE.
- Eager attention materializes repeated K/V for GQA through `repeat_kv`; optimized attention backends receive the unrepeated KV tensor plus module metadata.
- `Olmo3DecoderLayer` applies RMSNorm to the attention branch output before residual add, and RMSNorm to the MLP branch output before residual add.
- Final `Olmo3Model.norm` applies RMSNorm to decoder output before LM head.
- `Olmo3ForCausalLM.forward` supports `logits_to_keep` as either int or tensor index and only applies `lm_head` to that slice.
- `DynamicCache(config=...)` builds sliding cache layers from `layer_types`; sliding cache stores bounded previous K/V, full layers store growing K/V.

