# EoMT Source Notes

Audit date: 2026-05-13

Transformers checkout: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Primary files:

- `src/transformers/models/eomt/modular_eomt.py`: authoritative modular source for future edits.
- `src/transformers/models/eomt/modeling_eomt.py`: generated runtime source inspected for exact classes and helper functions.
- `src/transformers/models/eomt/configuration_eomt.py`: generated config source.
- `src/transformers/models/eomt/image_processing_eomt.py`: Torchvision-backed processor/postprocessor source.
- `src/transformers/models/eomt/image_processing_pil_eomt.py`: PIL-backed processor with matching public ABI.
- `src/transformers/models/eomt/convert_eomt_to_hf.py`: conversion source showing DINOv2-backed checkpoint mappings and special giant/SwiGLU handling.
- `tests/models/eomt/test_modeling_eomt.py` and `tests/models/eomt/test_image_processing_eomt.py`: representative integration shapes and postprocess expectations.
- `docs/source/en/model_doc/eomt.md`: task/preprocessing descriptions; used only to cross-check source-derived processor behavior.

Representative HF config/preprocessor URLs inspected:

- `https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/raw/main/config.json`
- `https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/raw/main/preprocessor_config.json`
- `https://huggingface.co/tue-mps/coco_instance_eomt_large_640/raw/main/config.json`
- `https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/raw/main/config.json`
- `https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/raw/main/preprocessor_config.json`
- `https://huggingface.co/tue-mps/coco_panoptic_eomt_base_640_2x/raw/main/config.json`
- `https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_640/raw/main/config.json`
- `https://huggingface.co/tue-mps/coco_panoptic_eomt_7b_640/raw/main/config.json`
- `https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/raw/main/config.json`
- `https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/raw/main/config.json`

No gated/401 EoMT DINOv2 config links were encountered in this pass. DINOv3 EoMT checkpoints use a different `eomt_dinov3` model family and are out of scope for this report.
