# Source Notes: pp_ocrv5_server_det

## Scope

- DinoML audit target: `pp_ocrv5_server_det`
- Transformers source root: `X:/H/transformers`
- Transformers commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Report written under: `H:/dinoml_v2/agents/plans/transformers/pp_ocrv5_server_det/`
- No DinoML tests, Transformers imports, or model execution were run.

## Local source files inspected

- `src/transformers/models/pp_ocrv5_server_det/configuration_pp_ocrv5_server_det.py`
- `src/transformers/models/pp_ocrv5_server_det/image_processing_pp_ocrv5_server_det.py`
- `src/transformers/models/pp_ocrv5_server_det/modeling_pp_ocrv5_server_det.py`
- `src/transformers/models/pp_ocrv5_server_det/modular_pp_ocrv5_server_det.py`
- `src/transformers/models/pp_ocrv5_server_det/__init__.py`
- `src/transformers/models/hgnet_v2/configuration_hgnet_v2.py`
- `src/transformers/models/hgnet_v2/modeling_hgnet_v2.py`
- `src/transformers/models/hgnet_v2/modular_hgnet_v2.py`
- `tests/models/pp_ocrv5_server_det/test_modeling_pp_ocrv5_server_det.py`
- `docs/source/en/model_doc/pp_ocrv5_server_det.md`

## Hugging Face sources checked

- `PaddlePaddle/PP-OCRv5_server_det_safetensors`
  - API metadata SHA observed: `cbea9f3c3254c6ff7b0016cfbf90549e1ad4c5bb`
  - `config.json` inspected from `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors/raw/main/config.json`
  - `preprocessor_config.json` inspected from `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det_safetensors/raw/main/preprocessor_config.json`
- `PaddlePaddle/PP-OCRv5_server_det`
  - API metadata SHA observed: `ca867c897ecbca8873081573a802ad70d499cb94`
  - PaddleOCR-style `config.json` inspected from `https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det/raw/main/config.json`
- `PaddlePaddle/PP-OCRv5_mobile_det_safetensors`
  - Config briefly checked only to confirm it is a separate `model_type` (`pp_ocrv5_mobile_det`) and should not be folded into this family audit.

## Key source anchors

- Generated files state that future source edits should go to `modular_pp_ocrv5_server_det.py`.
- `PPOCRV5ServerDetConfig` delegates `backbone_config` through `AutoConfig`, defaulting to `hgnet_v2` with `out_features=["stage1","stage2","stage3","stage4"]`.
- `PPOCRV5ServerDetModel.forward` calls `self.backbone(pixel_values)` and consumes `backbone_outputs.feature_maps`.
- `PPOCRV5ServerDetNeck.forward` expects four NCHW feature maps, performs nearest upsample, additions, 9x9 convolutions, bottom-up stride-2 convolution, intraclass blocks, per-level upsample by `scale_factor_list`, and `torch.cat(..., dim=1)`.
- `PPOCRV5ServerDetHead.forward` applies segmentation and local-refinement heads, two sigmoid calls, and averages the initial and refined maps.
- `PPOCRV5ServerDetImageProcessor.get_image_size` rounds resized height/width to multiples of 32 and clamps each side to at least 32.
- `post_process_object_detection` requires `target_sizes`, thresholds the probability map on CPU, calls OpenCV contours/min-area boxes, unclipping, score filtering, and returns variable-length rotated boxes with label 0.

## Gaps and caveats

- Only one native Transformers server-det checkpoint config was found. The synthetic test config and original PaddleOCR config were used as secondary variation evidence; they are not independent production Transformers checkpoints.
- Safetensors weight metadata was not inspected.
- Source defaults fill several HGNetV2 fields omitted by the safetensors config. Paddle-era fields such as `mode`, `upsample_mode`, `use_lab`, `use_last_conv`, `class_expand`, `class_num`, and `head_in_channels` were present in the HF config but no source reads were found in the inspected in-library model.
- The report treats `hgnet_v2` as a composed backbone dependency. Full HGNetV2 operator parity should be tracked by a separate backbone audit if DinoML wants reusable support beyond this detector.
