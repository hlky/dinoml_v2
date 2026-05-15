# PP-DocLayoutV3 Source Notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary family directory: `transformers/src/transformers/models/pp_doclayout_v3`
- Adjacent delta reference: `transformers/src/transformers/models/pp_doclayout_v2`
- Delegated backbone directory: `transformers/src/transformers/models/hgnet_v2`

Important source files:

- `configuration_pp_doclayout_v3.py`
- `image_processing_pp_doclayout_v3.py`
- `modeling_pp_doclayout_v3.py`
- `modular_pp_doclayout_v3.py`
- `../pp_doclayout_v2/{configuration,image_processing,modeling}_pp_doclayout_v2.py`
- `../hgnet_v2/{configuration,modeling}_hgnet_v2.py`

`modeling_pp_doclayout_v3.py` is generated from `modular_pp_doclayout_v3.py`; the generated file states future source edits should be made in the modular file.

## Hugging Face configs inspected

- `https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/raw/main/config.json`
- `https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/raw/main/preprocessor_config.json`
- `https://huggingface.co/PaddlePaddle/PP-DocLayoutV2_safetensors/raw/main/config.json`
- `https://huggingface.co/PaddlePaddle/PP-DocLayoutV2_safetensors/raw/main/preprocessor_config.json`
- `https://huggingface.co/api/models/PaddlePaddle/PP-DocLayoutV3_safetensors`
- `https://huggingface.co/api/models?search=PP-DocLayoutV3`
- `https://huggingface.co/api/models?search=PP-DocLayoutV2`

The only official Transformers-native PP-DocLayoutV3 checkpoint found in the HF search/API sweep was `PaddlePaddle/PP-DocLayoutV3_safetensors`. Other v3 repos found were PaddleOCR-native or ONNX mirrors/exports and are not treated as additional in-library config variants.

## Notable source anchors

- V3 config defaults and HF checkpoint overrides: `configuration_pp_doclayout_v3.py`, lines 31-190.
- V3 preprocessing and postprocess ABI: `image_processing_pp_doclayout_v3.py`, lines 39-303.
- Generated-from-modular notice: `modeling_pp_doclayout_v3.py`, lines 1-6.
- GlobalPointer reading-order head: `modeling_pp_doclayout_v3.py`, lines 52-69.
- Eager multiscale deformable attention fallback with `grid_sample`: `modeling_pp_doclayout_v3.py`, lines 72-124.
- Deformable attention wrapper and offset/reference-point math: `modeling_pp_doclayout_v3.py`, lines 127-231.
- Self-attention, MLP, AIFI, FPN/PAN, mask FPN: `modeling_pp_doclayout_v3.py`, lines 401-1057.
- Decoder layer and per-layer mask/order prediction: `modeling_pp_doclayout_v3.py`, lines 1060-1308.
- Backbone load and frozen BN replacement: `modeling_pp_doclayout_v3.py`, lines 1311-1403.
- Mask-to-box reference-point refinement: `modeling_pp_doclayout_v3.py`, lines 1529-1573.
- Main model forward and detector outputs: `modeling_pp_doclayout_v3.py`, lines 1581-2137.
- V2 detector/reading-order contrast: `modeling_pp_doclayout_v2.py`, lines 2348-2505.
- V2 postprocess contrast: `image_processing_pp_doclayout_v2.py`, lines 91-190.

## Validation notes

No DinoML tests, Python imports, model imports, or code execution were run. The audit used source reading plus HTTP reads of JSON/model metadata.
