# VitDet Source Notes

Audit date: 2026-05-13

Pinned Transformers checkout:

- `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Verified with `git -C X:/H/transformers rev-parse HEAD`.

Local source files inspected:

- `X:/H/transformers/src/transformers/models/vitdet/configuration_vitdet.py`
- `X:/H/transformers/src/transformers/models/vitdet/modeling_vitdet.py`
- `X:/H/transformers/src/transformers/models/vitdet/__init__.py`
- `X:/H/transformers/src/transformers/models/auto/auto_mappings.py`
- `X:/H/transformers/src/transformers/models/auto/modeling_auto.py`
- `X:/H/transformers/src/transformers/models/auto/image_processing_auto.py`
- `X:/H/transformers/docs/source/en/model_doc/vitdet.md`
- `X:/H/transformers/tests/models/vitdet/test_modeling_vitdet.py`

Remote/source documentation:

- Hugging Face docs: https://huggingface.co/docs/transformers/model_doc/vitdet
- Transformers source basis equivalent URL: https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitdet
- Original paper: https://huggingface.co/papers/2203.16527
- Original project mentioned by Transformers docs: https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet

Representative configs fetched:

- `google/vitdet-base-patch16-224`: https://huggingface.co/google/vitdet-base-patch16-224/raw/main/config.json returned HTTP 401. The config class uses this as the autodoc checkpoint, but the repo/config was not accessible without credentials.
- `google/vitdet-large-patch16-224`: https://huggingface.co/google/vitdet-large-patch16-224/raw/main/config.json returned HTTP 401.
- `hustvl/vitmatte-base-composition-1k`: https://huggingface.co/hustvl/vitmatte-base-composition-1k/raw/main/config.json fetched successfully. The nested `backbone_config` is `model_type: vitdet`.
- `hustvl/vitmatte-small-composition-1k`: https://huggingface.co/hustvl/vitmatte-small-composition-1k/raw/main/config.json fetched successfully. The nested `backbone_config` is `model_type: vitdet`.
- `hustvl/vitmatte-large-composition-1k`: https://huggingface.co/hustvl/vitmatte-large-composition-1k/raw/main/config.json returned HTTP 401.
- `dgcnz/dinov2_vitdet_DINO_12ep`, `hngan/ViTDet_COCO`, and `FlyingDutchman123/VitDet_FastMRI_knee_robust_IN` did not expose `config.json` at the default raw path during this audit.
- `wanglab/medsam-vit-base` fetched successfully but uses `model_type: sam`, not `model_type: vitdet`; it was used only as a contrast to avoid confusing SAM ViT configs with the VitDet family.

Image processor/postprocess note:

- No `VitDetImageProcessor` or VitDet entry in `IMAGE_PROCESSOR_MAPPING_NAMES` was found in the pinned Transformers tree.
- Transformers docs state: "At the moment, only the backbone is available."
- Therefore this audit treats detection/segmentation heads and postprocessing as downstream framework/model responsibility rather than VitDet source behavior.

