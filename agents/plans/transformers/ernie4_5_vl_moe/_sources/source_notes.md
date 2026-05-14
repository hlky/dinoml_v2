# ernie4_5_vl_moe source notes

Scope: docs-only audit for DinoML. No imports, no model execution, no DinoML tests.

## Pinned local source

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/ernie4_5_vl_moe`
- Generated source warning: `modeling_ernie4_5_vl_moe.py`, `configuration_ernie4_5_vl_moe.py`, and `image_processing_ernie4_5_vl_moe.py` are generated from `modular_ernie4_5_vl_moe.py`; future Transformers edits should target the modular file.

## Local files inspected

- `configuration_ernie4_5_vl_moe.py`
- `modeling_ernie4_5_vl_moe.py`
- `modular_ernie4_5_vl_moe.py`
- `processing_ernie4_5_vl_moe.py`
- `image_processing_ernie4_5_vl_moe.py`
- `image_processing_pil_ernie4_5_vl_moe.py`
- `video_processing_ernie4_5_vl_moe.py`
- `convert_ernie4_5_vl_moe_to_hf.py`

## Key source anchors

- Text M-RoPE: generated modeling lines 50-138; modular lines 226-348.
- Text attention and cache update: generated modeling lines 212-271.
- Dense SwiGLU MLP: generated modeling lines 293-309.
- MoE router/experts/modality split: generated modeling lines 310-488; modular lines 378-515.
- Decoder layer: generated modeling lines 491-543.
- Vision RoPE and attention: generated modeling lines 543-645.
- Vision block/MLP: generated modeling lines 647-686 and 819-829.
- Patch embedding/vision tower: generated modeling lines 830-958.
- Variable-resolution resampler: generated modeling lines 977-1076; modular lines 670-846.
- Multimodal 3D position construction and feature stitching: generated modeling lines 1102-1477.
- Conditional generation/cache/generation helpers: generated modeling lines 1566-1912.
- Processor placeholder expansion and token type IDs: `processing_ernie4_5_vl_moe.py`.
- Image patch packing: `image_processing_ernie4_5_vl_moe.py`.
- Video sampling, timestamp drawing, packing: `video_processing_ernie4_5_vl_moe.py`.
- Legacy config conversion: `convert_ernie4_5_vl_moe_to_hf.py` lines around 167-221.

## HF snapshots saved

Saved under this `_sources` directory:

- `baidu__ERNIE-4.5-VL-28B-A3B-Base-PT__config.json`
- `baidu__ERNIE-4.5-VL-28B-A3B-Base-PT__preprocessor_config.json`
- `baidu__ERNIE-4.5-VL-28B-A3B-Paddle__config.json`
- `baidu__ERNIE-4.5-VL-28B-A3B-Paddle__preprocessor_config.json`
- `baidu__ERNIE-4.5-VL-424B-A47B-Paddle__config.json`
- `baidu__ERNIE-4.5-VL-424B-A47B-Paddle__preprocessor_config.json`
- `baidu__ERNIE-4.5-VL-28B-A3B-Thinking__config.json`
- `baidu__ERNIE-4.5-VL-28B-A3B-Thinking__preprocessor_config.json`
- `tiny-random__ernie-4.5-vl-moe__config.json`
- `tiny-random__ernie-4.5-vl-moe__preprocessor_config.json`
- `tiny-random__ernie-4.5-vl-moe__processor_config.json`

Official Baidu repos probed did not expose `processor_config.json` at the checked URLs; they exposed legacy-style `preprocessor_config.json` instead. The tiny-random mirror exposed both.

## Source-basis caveats

- Several official configs use legacy `model_type: "ernie4_5_moe_vl"` and remote-code `auto_map` names such as `configuration_ernie4_5_vl.Ernie4_5_VLMoEConfig`. The in-library native config class uses `model_type: "ernie4_5_vl_moe"` and nested `text_config`/`vision_config`.
- `convert_ernie4_5_vl_moe_to_hf.py` is therefore part of the source basis for interpreting official legacy configs against native Transformers code.
- Legacy configs include many training/distributed/fusion fields that the inspected native modeling file does not read. The report treats them as ignored for this source basis unless conversion maps them into native config fields.
- Video processor config files were not found for the official Baidu snapshots. Native source defaults were used for video preprocessing and explicitly labeled as source defaults.
