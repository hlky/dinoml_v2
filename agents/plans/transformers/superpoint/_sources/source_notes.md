# SuperPoint Source Notes

Scope: Transformers `superpoint` family only, inspected from `X:/H/transformers`
at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Local source files

- `src/transformers/models/superpoint/configuration_superpoint.py`
- `src/transformers/models/superpoint/modeling_superpoint.py`
- `src/transformers/models/superpoint/image_processing_superpoint.py`
- `src/transformers/models/superpoint/image_processing_pil_superpoint.py`
- `src/transformers/models/superpoint/convert_superpoint_to_pytorch.py`
- `tests/models/superpoint/test_modeling_superpoint.py`
- `tests/models/superpoint/test_image_processing_superpoint.py`

## Hub configs sampled

- `magic-leap-community/superpoint`
  - `config.json`: native `model_type="superpoint"`, architecture field is
    historical/odd (`SuperPointModel`) even though current source exports
    `SuperPointForKeypointDetection`.
  - `preprocessor_config.json`: resize to `480x640`, rescale by `1/255`,
    `image_processor_type="SuperPointImageProcessor"`.
- `stevenbucaille/superpoint`
  - Same operator-significant config as Magic Leap, with architecture
    `SuperPointForKeypointDetection`.
  - Same processor config.
- `ETH-CVG/lightglue_superpoint`
  - `model_type="lightglue"` wrapper with nested `keypoint_detector_config`
    equal to the full SuperPoint defaults.
  - Processor is `LightGlueImageProcessor`, `do_grayscale=true`, same
    `480x640` resize/rescale.
- `stevenbucaille/lightglue_superpoint`
  - `model_type="lightglue"` wrapper with nested
    `keypoint_detector_config={"model_type":"superpoint"}`; effective
    SuperPoint dimensions come from `SuperPointConfig` defaults.
  - Processor is `LightGlueImageProcessor`, `do_grayscale=true`, same
    `480x640` resize/rescale.
- `AXERA-TECH/superpoint`
  - Not a Transformers SuperPoint config: `model_type="ONNX"` for an Axera NPU
    export. Treat as out of scope for this report.

## Line anchors

- Config defaults: `configuration_superpoint.py:24`.
- Border filtering, top-k, simple NMS: `modeling_superpoint.py:37`,
  `modeling_superpoint.py:47`, `modeling_superpoint.py:55`.
- Encoder block stack: `modeling_superpoint.py:110`, `modeling_superpoint.py:140`.
- Detector head and score-to-pixel shuffle: `modeling_superpoint.py:190`,
  `modeling_superpoint.py:225`, `modeling_superpoint.py:236`.
- Descriptor head and descriptor sampling: `modeling_superpoint.py:262`,
  `modeling_superpoint.py:304`.
- RGB-to-single-channel extraction in model: `modeling_superpoint.py:330`,
  `modeling_superpoint.py:411`.
- Variable keypoint padding/mask and relative coordinate output:
  `modeling_superpoint.py:435`, `modeling_superpoint.py:452`.
- Torchvision processor defaults/preprocess/postprocess:
  `image_processing_superpoint.py:70`, `image_processing_superpoint.py:84`,
  `image_processing_superpoint.py:114`.
- PIL processor defaults/preprocess/postprocess:
  `image_processing_pil_superpoint.py:72`, `image_processing_pil_superpoint.py:86`,
  `image_processing_pil_superpoint.py:114`.
- Test notes: smaller synthetic config at `test_modeling_superpoint.py:39`,
  top-k batching flake at `test_modeling_superpoint.py:135`, integration
  expected keypoint counts at `test_modeling_superpoint.py:264`.

## Source-derived traps

- Current source processes each batch item through keypoint extraction and
  descriptor sampling using Python loops after the shared encoder.
- Detector output is value-dependent: score threshold, NMS, `nonzero`, optional
  `topk`, then padding to the maximum keypoint count in the batch.
- `top_k_keypoints` uses `torch.topk`; the upstream test marks batching
  equivalence flaky because equal-score tie indices are not stable.
- `_extract_keypoints` gets full-resolution score map shape but passes
  `height * 8, width * 8` to `remove_keypoints_from_borders`; source parity
  requires matching that current behavior unless intentionally bug-fixed.
- The model always takes the first channel from `pixel_values` at runtime,
  regardless of whether the processor converted RGB to grayscale.
- Torchvision and PIL processors differ in grayscale ordering relative to resize:
  torchvision grayscale before resize, PIL resize/rescale before grayscale.
