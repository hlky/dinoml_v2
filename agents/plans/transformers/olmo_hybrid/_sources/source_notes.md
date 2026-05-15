# OLMo Hybrid Source Notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `transformers/src/transformers/models/olmo_hybrid`
- Generated files note: `configuration_olmo_hybrid.py` and `modeling_olmo_hybrid.py` are generated from `modular_olmo_hybrid.py`; report cites the generated runtime files because DinoML will need to match the actual imported implementation.

## Files inspected

- `configuration_olmo_hybrid.py`
  - `OlmoHybridConfig`, defaults, strict validation, default layer cadence, linear-attention dimension defaults.
- `modeling_olmo_hybrid.py`
  - `OlmoHybridDynamicCache`, RMSNorm variants, short convolution fallback, causal attention, RoPE/NoPE branch, GatedDeltaNet fallback math, decoder layers, model, causal LM head.
- `modular_olmo_hybrid.py`
  - Source for the generated implementation.
- `convert_olmo_hybrid_weights_to_hf.py`
  - OLMo-core-to-HF mapping, layer-type extraction, FLA weight names, dtype/config conversion behavior.
- `transformers/tests/models/olmo_hybrid/test_modeling_olmo_hybrid.py`
  - Cache shape expectations, multi-token cached forward parity test, integration checkpoint id.
- `transformers/src/transformers/generation/utils.py`
  - Generation cache admission comments for OLMoHybrid.
- `transformers/src/transformers/conversion_mapping.py`
  - Checkpoint key rename aliases for OLMo Hybrid layer norms.

## Hub/config sources

- [allenai/Olmo-Hybrid-7B config](https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/main/config.json)
  - Public base 7B config: `hidden_size=3840`, `num_hidden_layers=32`, `num_attention_heads=30`, `num_key_value_heads=30`, `linear_key_head_dim=96`, `linear_value_head_dim=192`, `max_position_embeddings=65536`, `rope_parameters=null`.
- [allenai/Olmo-Hybrid-Instruct-SFT-7B config](https://huggingface.co/allenai/Olmo-Hybrid-Instruct-SFT-7B/blob/main/config.json)
  - Same architecture dims; `max_position_embeddings=32768`, `rope_parameters=null`.
- [allenai/Olmo-Hybrid-Instruct-DPO-7B config](https://huggingface.co/allenai/Olmo-Hybrid-Instruct-DPO-7B/blob/main/config.json)
  - Same architecture dims; `dtype="bfloat16"`, `max_position_embeddings=32768`, `rope_parameters={"rope_theta": null, "rope_type": "default"}`.
- [allenai/Olmo-Hybrid-Think-SFT-7B config](https://huggingface.co/allenai/Olmo-Hybrid-Think-SFT-7B/blob/main/config.json)
  - Same architecture dims; `max_position_embeddings=32768`, `rope_parameters=null`.
- [OLMo Hybrid model docs](https://huggingface.co/docs/transformers/model_doc/olmo_hybrid)
  - Public docs identify the model family and checkpoint examples.

## Source gaps / caveats

- No imports or tests were run per request.
- FLA kernels were not inspected from `flash-linear-attention`; report treats them as an optional provider with the local torch fallback as semantic reference.
- No safetensors metadata was downloaded; parameter counts and dtype statements are from Hub metadata/configs or source inference.
- Public Hub configs are accessible; no gated 401/403 source was encountered for the configs used.
