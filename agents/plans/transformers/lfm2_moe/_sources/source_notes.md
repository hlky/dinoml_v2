# LFM2-MoE Source Notes

Audit target: `lfm2_moe` at local Transformers checkout
`transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Local source files

- `src/transformers/models/lfm2_moe/configuration_lfm2_moe.py`
  - Config defaults: vocab 65536, hidden 2048, dense FFN 7168, MoE FFN 1792,
    32 layers, 32 query heads, 8 KV heads, cache enabled, short-conv kernel
    cache length 3, 2 dense layers before MoE layers, 32 experts, top-4 routing.
  - `default_theta = 1000000.0`.
  - `tie_embedding` is accepted as a backwards-compatible alias for
    `tie_word_embeddings`.
- `src/transformers/models/lfm2_moe/modular_lfm2_moe.py`
  - Authoritative edit source. It composes LFM2 attention/short-conv with
    Qwen2-MoE-style packed expert tensors and Llama CausalLM wrappers.
- `src/transformers/models/lfm2_moe/modeling_lfm2_moe.py`
  - Generated expanded source used for exact behavior in this report.
  - Key anchors:
    - `Lfm2MoeRMSNorm`: line 57.
    - `Lfm2MoeRotaryEmbedding`: line 77.
    - `Lfm2MoeMLP`: line 142.
    - `Lfm2MoeExperts`: line 156.
    - `Lfm2MoeSparseMoeBlock`: line 195.
    - `apply_rotary_pos_emb`: line 239.
    - `eager_attention_forward`: line 276.
    - `Lfm2MoeAttention`: line 302.
    - `Lfm2MoeShortConv`: line 376.
    - `Lfm2MoeDecoderLayer`: line 476.
    - `Lfm2MoeModel`: line 553.
    - `Lfm2MoeForCausalLM`: line 631.
- Cross-checks:
  - `src/transformers/cache_utils.py`
    - `LinearAttentionLayer` keeps `conv_states` and updates them with static
      address semantics.
    - `LAYER_TYPE_CACHE_MAPPING` maps `"full_attention"` to dynamic KV cache
      layers and `"conv"` to linear-attention cache layers.
  - `docs/source/en/model_doc/lfm2_moe.md`
    - Describes LFM2-MoE as short-range input-aware gated convolutions plus GQA
      and sparse MoE FFNs. Used only as secondary context, not as the source of
      operator requirements.
  - `tests/models/lfm2_moe/test_modeling_lfm2_moe.py`
    - Confirms mixed `layer_types` behavior and official integration target
      `LiquidAI/LFM2-8B-A1B`.

## HF config snapshots saved here

Downloaded to `_sources/` with `Invoke-WebRequest`:

- `LiquidAI_LFM2-8B-A1B_config.json`
- `LiquidAI_LFM2-8B-A1B_generation_config.json`
- `LiquidAI_LFM2-8B-A1B_tokenizer_config.json`
- `LiquidAI_LFM2-8B-A1B-ONNX_config.json`
- `LiquidAI_LFM2-24B-A2B_config.json`
- `LiquidAI_LFM2-24B-A2B_generation_config.json`
- `LiquidAI_LFM2-24B-A2B_tokenizer_config.json`
- `LiquidAI_LFM2-24B-A2B-ONNX_config.json`
- `tiny-random_lfm2-moe_config.json`

Representative topology summary from `config.json`:

| Snapshot | Layers | Conv | Attention | H | Q/KV heads | Head dim | Dense I | MoE I | Experts/top-k | Max pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tiny-random/lfm2-moe` | 3 | 2 | 1 | 64 | 2/1 | 32 | 128 | 128 | 32/4 | 128000 |
| `LiquidAI/LFM2-8B-A1B` | 24 | 18 | 6 | 2048 | 32/8 | 64 | 7168 | 1792 | 32/4 | 128000 |
| `LiquidAI/LFM2-8B-A1B-ONNX` | 24 | 18 | 6 | 2048 | 32/8 | 64 | 7168 | 1792 | 32/4 | 128000 |
| `LiquidAI/LFM2-24B-A2B` | 40 | 30 | 10 | 2048 | 32/8 | 64 | 11776 | 1536 | 64/4 | 128000 |
| `LiquidAI/LFM2-24B-A2B-ONNX` | 40 | 30 | 10 | 2048 | 32/8 | 64 | 11776 | 1536 | 64/4 | 128000 |

Notes:

- The 8B raw config uses legacy top-level `rope_theta`; the current config
  utilities standardize legacy RoPE fields into `rope_parameters`.
- ONNX snapshots add `transformers.js_config` and quantized ONNX metadata. The
  native Transformers source inspected here does not read those fields for the
  PyTorch graph.
- Tokenizer configs contain text, tool, FIM, and image-like special tokens, but
  the audited model source is text-only and consumes only token IDs / optional
  input embeddings plus masks and position IDs.
