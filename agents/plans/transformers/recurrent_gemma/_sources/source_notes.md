# RecurrentGemma Source Notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model source directory: `transformers/src/transformers/models/recurrent_gemma`

## Files inspected

| File | SHA256 | Notes |
|---|---:|---|
| `configuration_recurrent_gemma.py` | `A7C4EE122796FED4CFCD98DA5FE141F0F3FE14A2257C960FCF6D78BEB449F391` | Config fields, defaults, strict validation, block type expansion, RoPE defaults. |
| `modeling_recurrent_gemma.py` | `06B136FF1597292E3561F08EC4DB7F44A28CC4C31E200F90122D8D20D6A6F114` | Model, recurrent block, RG-LRU scan, sliding-window attention, cache handling, logits soft cap. |
| `convert_recurrent_gemma_to_hf.py` | `71DC8FF2894BB19ABC754D2CF9AB12B389FFEC4E19ADF85EB76449B72B9D7D85` | Converter presets and weight-name/shape transforms. |
| `tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py` | `220B716C8873206DFD88BC228E1B500A85F087ED1A4E1C1BBBAC5C37600395C2` | Integration behavior, skipped generation/cache tests, long-window generation coverage. |
| `docs/source/en/model_doc/recurrent_gemma.md` | not hashed | Architecture summary and release/source pointers. |

## Key line anchors

- `configuration_recurrent_gemma.py:57-100`: `sliding_window` alias, default dimensions, `lru_width` defaulting to `hidden_size`, `head_dim`, `num_key_value_heads`, `partial_rotary_factor=0.5`, repeating `layers_block_type`.
- `modeling_recurrent_gemma.py:44-59`: RMSNorm computes in fp32 and applies `(1 + weight.float())` before casting to input dtype.
- `modeling_recurrent_gemma.py:65-129`: default-only RoPE, `rope_theta`, `partial_rotary_factor`, fp32 trig, cos/sin cast back to input dtype.
- `modeling_recurrent_gemma.py:181-247`: attention projections, partial RoPE split/concat, `DynamicCache.update`, KV repeat, SDPA with sliding-window mask.
- `modeling_recurrent_gemma.py:267-375`: RG-LRU gates, `softplus`, `exp`, `sqrt(1 - a^2)`, reset on `position_ids == 0`, fp32 recurrent state scan.
- `modeling_recurrent_gemma.py:378-452`: recurrent block linear branches, depthwise causal Conv1d, mutable `conv1d_state`, mutable `rg_lru.recurrent_states`, prefill/decode split.
- `modeling_recurrent_gemma.py:458-470`: gated MLP with activation on gate branch and biased projections.
- `modeling_recurrent_gemma.py:594-690`: token embedding, DynamicCache creation, first-attention-layer cache hack, sliding-window causal mask creation, embedding scale, layer loop, final RMSNorm.
- `modeling_recurrent_gemma.py:698-785`: LM head, `logits_to_keep`, tanh logits soft cap, no `past_key_values` returned in `CausalLMOutput`.
- `convert_recurrent_gemma_to_hf.py:56-67`: converter config presets: explicit `2B` preset and default `7B` preset.
- `convert_recurrent_gemma_to_hf.py:89-122`: source weight mapping, Conv1d weight transform, MLP split, embedding/LM-head cloning.
- `tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py:35-124`: tests skip many generation features because past key values are not returned and left/right padding differs.
- `tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py:135-248`: integration tests for `google/recurrentgemma-2b`, 8-bit alternate repo, long context, and window shorter than prompt.

## Hugging Face config access

Attempted direct `config.json` fetches with `Invoke-WebRequest`:

- `https://huggingface.co/google/recurrentgemma-2b/raw/main/config.json` -> `401 Unauthorized`
- `https://huggingface.co/google/recurrentgemma-2b-it/raw/main/config.json` -> `401 Unauthorized`
- `https://huggingface.co/google/recurrentgemma-9b/raw/main/config.json` -> `401 Unauthorized`
- `https://huggingface.co/google/recurrentgemma-9b-it/raw/main/config.json` -> `401 Unauthorized`

Hugging Face search and model pages identify these as gated Google repos. Access approval to the Google RecurrentGemma license would resolve this gap. The report therefore treats local source defaults, the converter presets, docs, and tests as the inspectable config basis.

## Source-derived traps

- The converter's `2B` preset differs from the config class defaults: it sets `num_key_value_heads=1` and `intermediate_size=15360`; the class default uses `num_key_value_heads=None -> num_attention_heads` and `intermediate_size=3 * 2560`.
- The converter names `gemma_7b_config = RecurrentGemmaConfig()`, while public gated repos include 9B variants. This report does not infer 9B dimensions from that preset.
- Recurrent block cache state is stored mutably on modules, not returned through the model output. DinoML should model this as explicit session state, not as hidden compiler/runtime state.
- The recurrent block allocates `conv1d_state` as `(batch, hidden_size, conv1d_width - 1)` even though the Conv1d channels are `lru_width`. This is harmless for inspectable defaults where `lru_width == hidden_size`, but it is a guard/failure case if configs diverge.
