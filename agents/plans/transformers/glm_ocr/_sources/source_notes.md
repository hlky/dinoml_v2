# GLM-OCR audit source notes

## Local Transformers checkout

- Checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/glm_ocr`
- Main generated files:
  - `configuration_glm_ocr.py`
  - `modeling_glm_ocr.py`
  - `modular_glm_ocr.py`
- `configuration_glm_ocr.py` and `modeling_glm_ocr.py` are generated from `modular_glm_ocr.py`; future source edits should target the modular file, but runtime behavior was audited from the generated files.

## Processor boundary

`glm_ocr` has no local processor or image processor files. The official `zai-org/GLM-OCR` preprocessor snapshot names:

- `processor_class`: `Glm46VProcessor`
- `image_processor_type`: `Glm46VImageProcessor`

The audited processor/preprocessor behavior therefore comes from:

- `X:/H/transformers/src/transformers/models/glm46v/processing_glm46v.py`
- `X:/H/transformers/src/transformers/models/glm46v/image_processing_glm46v.py`
- `X:/H/transformers/src/transformers/models/glm46v/video_processing_glm46v.py`

`glm4v` processor files were also inspected because `glm46v` is generated from the same GLM-V processor pattern and the `glm_ocr` generated modeling code shares that ABI.

## Hugging Face snapshots saved locally

Downloaded under this directory:

- `zai-org_GLM-OCR_config.json`
- `zai-org_GLM-OCR_preprocessor_config.json`
- `zai-org_GLM-OCR_generation_config.json`
- `tiny-random_glm-ocr_config.json`
- `yujiepan_glm-ocr-tiny-random_config.json`
- `mlx-community_GLM-OCR-bf16_config.json`
- `onnx-community_GLM-OCR-ONNX_config.json`
- `unsloth_GLM-OCR_config.json`

Representative sweep summary:

| Snapshot | Scope note |
| --- | --- |
| `zai-org/GLM-OCR` | Official production config, model_type `glm_ocr`, bf16, 16 text layers, 24 vision layers. |
| `mlx-community/GLM-OCR-bf16` | Mirror/export config; same operator-significant dimensions as official. |
| `onnx-community/GLM-OCR-ONNX` | ONNX mirror config; same operator-significant dimensions as official. |
| `unsloth/GLM-OCR` | Mirror config; same operator-significant dimensions as official. |
| `tiny-random/glm-ocr` and `yujiepan/glm-ocr-tiny-random` | Debug configs; intentionally tiny dimensions but retain GQA, M-RoPE, multimodal stitch, and vision patch/merge structure. |

## Web URLs

- Official config: https://huggingface.co/zai-org/GLM-OCR/blob/main/config.json
- Official preprocessor config: https://huggingface.co/zai-org/GLM-OCR/blob/main/preprocessor_config.json
- Official generation config: https://huggingface.co/zai-org/GLM-OCR/blob/main/generation_config.json
- Model page: https://huggingface.co/zai-org/GLM-OCR
