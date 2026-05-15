# DensePose, MediaPipe Face Mesh, and MeshGraphormer

## Coverage

- Diffusers: not covered.
- Transformers: not covered as these detector/rendering stacks.
- Third-party/UI: DensePose and mesh/face helpers are UI or package integrations.

## Runtime Contract

These produce pose, UV, mesh, or face landmark condition renderings. Their output is usually a drawn RGB conditioning map with thresholding, keypoint/mesh topology, and optional face/body region selection.

`controlnet_aux.MediapipeFaceDetector` covers the MediaPipe face branch. It converts input to HWC3, resizes to `detect_resolution` (default 512), calls `generate_annotation` with `max_faces=1` and `min_confidence=0.5` by default, then resizes the rendered annotation map to `image_resolution`.

## Operators

- Detector/landmark model execution, often ONNX or external package.
- Indexed landmark/mesh topology transforms.
- Rasterization/drawing into condition maps.

## DinoML Notes

Treat as external preprocessing providers until a specific model implementation is selected. The drawing topology and coordinate normalization are as important as network inference.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/mediapipe_face/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/mediapipe_face/mediapipe_face_common.py`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:700`
- `invoke-ai/InvokeAI/invokeai/app/invocations/mediapipe_face.py:9`
- `agents/plans/auxiliary/control_preprocessors.md`
