# OpenPose and DWPose

## Coverage

- Diffusers: not covered.
- Transformers: not covered for DWPose/OpenPose detector execution.
- Third-party/UI: DWPose is ONNXRuntime based in InvokeAI; OpenPose-compatible render output is used by ControlNet UIs.

## Runtime Contract

InvokeAI's `DWOpenposeDetector` loads `yzd-v/DWPose/yolox_l.onnx` and `dw-ll_ucoco_384.onnx`. It runs object/person detection, pose estimation, keypoint remapping into OpenPose body order, optional face/hand extraction, then draws pose lines onto a black RGB canvas.

`controlnet_aux` includes two related but distinct paths:

- `OpenposeDetector` loads classic OpenPose body, hand, and face PyTorch weights (`body_pose_model.pth`, `hand_pose_model.pth`, `facenet.pth`) and supports `include_body`, `include_hand`, and `include_face`.
- `DWposeDetector` uses MMDetection/MMPose configs by default, with YOLOX person detection and `wanghaofan/dw-ll_ucoco_384` top-down pose weights. It remaps whole-body keypoints to OpenPose-compatible body order, filters by score thresholds, and draws body, hand, and face maps.

The processor registry aliases OpenPose body/face/hand combinations (`openpose`, `openpose_face`, `openpose_faceonly`, `openpose_full`, `openpose_hand`) and `dwpose`.

## Operators

- ONNX object detector and pose model.
- PyTorch/MMDetection/MMPose detector and pose model in the `controlnet_aux` DWPose path.
- Keypoint tensor indexing, score thresholding, coordinate normalization.
- CPU/GPU drawing of body, hand, and face maps.

## DinoML Notes

First support should treat ONNXRuntime as an external detector provider or CPU preprocessor. End-to-end behavior is not just model logits; the OpenPose-compatible canvas drawing and thresholding are part of the conditioning contract.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/open_pose/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/open_pose/model.py`
- `H:/controlnet_aux/src/controlnet_aux/dwpose/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/dwpose/wholebody.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `invoke-ai/InvokeAI/invokeai/backend/image_util/dw_openpose/__init__.py`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:308`
