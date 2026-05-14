# longcat_flash source notes

Audit date: 2026-05-13

Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `src/transformers/models/longcat_flash`

Inspected local source:

- `src/transformers/models/longcat_flash/configuration_longcat_flash.py`
- `src/transformers/models/longcat_flash/modeling_longcat_flash.py`
- `src/transformers/models/longcat_flash/modular_longcat_flash.py`
- `src/transformers/modeling_rope_utils.py`
- `src/transformers/masking_utils.py`
- `src/transformers/cache_utils.py`
- `src/transformers/integrations/tensor_parallel.py`
- `tests/models/longcat_flash/test_modeling_longcat_flash.py`
- `docs/source/en/model_doc/longcat_flash.md`

HF config URLs checked:

- `https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/raw/main/config.json`
- `https://huggingface.co/meituan-longcat/LongCat-Flash-Chat-FP8/raw/main/config.json`
- `https://huggingface.co/meituan-longcat/LongCat-Flash-Thinking/raw/main/config.json`
- `https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/raw/main/config.json`
- `https://huggingface.co/meituan-longcat/LongCat-Flash-Lite-FP8/raw/main/config.json`
- `https://huggingface.co/meituan-longcat/LongCat-Flash-Omni/raw/main/config.json`
- `https://huggingface.co/tiny-random/longcat-flash/raw/main/config.json`

Representative config observations:

- Chat/Chat-FP8/Thinking/Omni configs use `architectures="LongcatFlashForCausalLM"` and `auto_map` entries for `configuration_longcat_flash` / `modeling_longcat_flash`.
- Lite/Lite-FP8 configs use `architectures="LongcatFlashNgramForCausalLM"` and `auto_map` entries for `configuration_longcat_ngram` / `modeling_longcat_ngram`; these are not present in the inspected in-library family directory and should be treated as remote-code or separate-audit targets.
- Chat/Thinking/Omni raw configs omit `model_type` but the in-library config class sets `model_type="longcat_flash"`.
- Chat/Thinking/Omni raw configs omit `rope_scaling`; the in-library standardization turns `rope_theta=10000000.0` into default RoPE parameters.
- Lite configs include YaRN `rope_scaling` with `original_max_position_embeddings=32768`, `factor=10`, `beta_fast=32`, `beta_slow=1`, `mscale=1`, `mscale_all_dim=1`, but native `longcat_flash` does not implement the Lite N-gram wrapper.
- FP8 configs include `quantization_config` with `quant_method="fp8"`, `fmt="e4m3"`, dynamic activation scheme, block size `[128,128]`, and long `ignored_layers`; the inspected modeling code does not implement FP8 kernels directly.

No DinoML tests or model imports were run.
