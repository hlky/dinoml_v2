# EXAONE4 source notes

## Scope

- DinoML audit target: `exaone4`
- Transformers checkout: `transformers`
- Verified commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Report output: `H:/dinoml_v2/agents/plans/transformers/exaone4/report.md`
- No DinoML tests or Python imports were run.

## Local source files inspected

- `transformers/src/transformers/models/exaone4/configuration_exaone4.py`
- `transformers/src/transformers/models/exaone4/modeling_exaone4.py`
- `transformers/src/transformers/models/exaone4/modular_exaone4.py`
- Supporting inherited/shared behavior:
  - `transformers/src/transformers/models/gemma2/modeling_gemma2.py`
  - `transformers/src/transformers/models/llama/modeling_llama.py`
  - `transformers/src/transformers/models/olmo2/modeling_olmo2.py`
  - `transformers/src/transformers/modeling_rope_utils.py`
  - `transformers/src/transformers/cache_utils.py`
  - `transformers/src/transformers/masking_utils.py`
  - `transformers/src/transformers/generation/utils.py`

## Hugging Face config URLs checked

- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/raw/main/config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/raw/main/config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-AWQ/raw/main/config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-FP8/raw/main/config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/raw/main/generation_config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/raw/main/generation_config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-AWQ/raw/main/generation_config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-FP8/raw/main/generation_config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/raw/main/tokenizer_config.json`
- `https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/raw/main/tokenizer_config.json`

## Representative config facts captured

- `EXAONE-4.0-1.2B`: 30 layers, hidden size 2048, 32 query heads, 8 KV heads, head dim 64, all `full_attention`, no sliding window, tied token embeddings/LM head, BF16 config dtype.
- `EXAONE-4.0-32B`: 64 layers, hidden size 5120, 40 query heads, 8 KV heads, head dim 128, repeating `LLLG` layer pattern, sliding window 4096, untied LM head, BF16 config dtype.
- `EXAONE-4.0-32B-AWQ`: same graph dimensions as 32B with `quantization_config` `{quant_method: awq, bits: 4, group_size: 128, zero_point: true, version: gemm, modules_to_not_convert: ["lm_head"]}`.
- `EXAONE-4.0-32B-FP8`: same graph dimensions as 32B with `quantization_config` `{quant_method: fp8, activation_scheme: dynamic, weight_block_size: [128, 128], modules_to_not_convert: null}`.
- 32B generation configs set `cache_implementation: "hybrid"`; current Transformers generation code unsets default `"hybrid"` before preparation so dynamic hybrid cache is used by default unless explicitly requested.
- Tokenizer configs identify `GPT2Tokenizer`, `bos_token=[BOS]`, `eos_token=[|endofturn|]`, `pad_token=[PAD]`, `unk_token=[UNK]`, and an effectively unbounded `model_max_length`; chat template details were not fully expanded for this operator audit.

## Source gaps and cautions

- No official small debug checkpoint with miniature dimensions was found in the checked public family. The report uses source defaults as the debug/small synthetic baseline and public 1.2B/32B configs as representative checkpoints.
- `modeling_exaone4.py` and `configuration_exaone4.py` are generated from `modular_exaone4.py`; the generated files are the import-time source for this commit, while modular is authoritative for future upstream edits.
- Quantization behavior in the inspected Transformers source is not EXAONE4-specific model code. AWQ/FP8 must be treated as loader/provider contracts driven by config and backend integration, not as Python graph operators in `modeling_exaone4.py`.
