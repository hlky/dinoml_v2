# Source Notes

Audit date: 2026-05-13.

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/sam3_lite_text`
- Files inspected:
  - `configuration_sam3_lite_text.py`
  - `modeling_sam3_lite_text.py`
  - `modular_sam3_lite_text.py`
  - `convert_sam3_lite_text_to_hf.py`
  - `__init__.py`
- Shared SAM3 files inspected for inherited vision/detector contracts:
  - `X:/H/transformers/src/transformers/models/sam3/configuration_sam3.py`
  - `X:/H/transformers/src/transformers/models/sam3/modeling_sam3.py`

## Hub sources

- HF documentation: `https://huggingface.co/docs/transformers/model_doc/sam3_lite_text`
- Configs fetched without importing model code:
  - `https://huggingface.co/yonigozlan/sam3-litetext-s0/raw/main/config.json`
  - `https://huggingface.co/yonigozlan/sam3-litetext-s1/raw/main/config.json`
  - `https://huggingface.co/yonigozlan/sam3-litetext-l/raw/main/config.json`
  - `https://huggingface.co/yonigozlan/sam3-litetext-s0/raw/main/tokenizer_config.json`
  - `https://huggingface.co/yonigozlan/sam3-litetext-s0/raw/main/processor_config.json`

## Source observations

- No code imports, model construction, or tests were run.
- The local generated `modeling_sam3_lite_text.py` states it is generated from `modular_sam3_lite_text.py`; implementation facts were checked against the generated file because that is the runtime source.
- The full `Sam3LiteTextModel` uses the same SAM3 detector-style geometry/DETR/mask pipeline, with the heavy text encoder replaced by the MobileCLIP-style lightweight text tower.
- The model has `_keys_to_ignore_on_load_unexpected` for `tracker_model` and `tracker_neck`, but the inspected `sam3_lite_text` implementation is image/detector segmentation only. No tracker forward/session state implementation is present in this family directory.
- Public model repos were accessible. No gated checkpoint access was needed for config/tokenizer/processor metadata.
