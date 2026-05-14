# glm4v_moe source notes

Audit date: 2026-05-13

## Local Transformers source

- Repository: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `src/transformers/models/glm4v_moe`
- Generated source notice: `modeling_glm4v_moe.py` and `configuration_glm4v_moe.py` are generated from `modular_glm4v_moe.py`; future source edits should target the modular file.

Snapshots copied for local provenance:

- `configuration_glm4v_moe.py`
- `modular_glm4v_moe.py`
- `modeling_glm4v_moe.py`
- `processing_glm4v.py`
- `image_processing_glm4v.py`
- `video_processing_glm4v.py`

Important line anchors in the snapshot:

- Text attention: `modeling_glm4v_moe.py:129`
- MoE block and routing: `modeling_glm4v_moe.py:259`, `modeling_glm4v_moe.py:279`
- Vision patch embed: `modeling_glm4v_moe.py:513`
- Vision model forward/packing: `modeling_glm4v_moe.py:757`
- Text M-RoPE: `modeling_glm4v_moe.py:886`
- Multimodal rope index: `modeling_glm4v_moe.py:1176`
- Placeholder masks and `masked_scatter`: `modeling_glm4v_moe.py:1322`, `modeling_glm4v_moe.py:1448`, `modeling_glm4v_moe.py:1454`
- Generation hooks: `modeling_glm4v_moe.py:1723`, `modeling_glm4v_moe.py:1761`
- Processor placeholder expansion: `processing_glm4v.py:113`, `processing_glm4v.py:123`
- Processor modality type IDs: `processing_glm4v.py:245`
- Image processor patch ABI: `image_processing_glm4v.py:194`, `image_processing_glm4v.py:210`, `image_processing_glm4v.py:224`
- Video processor sampling and patch ABI: `video_processing_glm4v.py:96`, `video_processing_glm4v.py:204`, `video_processing_glm4v.py:236`

## Hugging Face config snapshots

Fetched from public Hugging Face `resolve/main` URLs:

- `https://huggingface.co/zai-org/GLM-4.5V/resolve/main/config.json`
- `https://huggingface.co/zai-org/GLM-4.5V/resolve/main/generation_config.json`
- `https://huggingface.co/zai-org/GLM-4.5V/resolve/main/preprocessor_config.json`
- `https://huggingface.co/zai-org/GLM-4.5V-FP8/resolve/main/config.json`
- `https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/resolve/main/config.json`
- `https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/resolve/main/preprocessor_config.json`

`https://huggingface.co/zai-org/GLM-4.5V/resolve/main/processor_config.json` returned 404. The available preprocessor config names `processor_class: Glm4vProcessor`.

Config scope notes:

- `zai-org/GLM-4.5V` and `zai-org/GLM-4.5V-FP8` use `model_type: glm4v_moe` and `Glm4vMoeForConditionalGeneration`.
- `zai-org/GLM-4.1V-9B-Thinking` uses `model_type: glm4v` and `Glm4vForConditionalGeneration`; it is useful for shared GLM-4V vision/processor behavior, but is out of scope for MoE text routing.
- `GLM-4.5V-FP8` adds `compressed-tensors` quantization metadata. This report treats that as a separate weight-loading/provider contract, not as native dense bf16 behavior.
