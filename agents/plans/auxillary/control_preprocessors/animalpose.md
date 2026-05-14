# AnimalPose

## Coverage

- Diffusers: not covered.
- Transformers: not covered as a named model.
- Third-party/UI: exposed through ControlNet auxiliary preprocessor stacks.

## Runtime Contract

AnimalPose produces pose keypoints and a rendered pose condition image for animals. It is similar to OpenPose/DWPose at the runtime boundary but uses different keypoint topology, detector assumptions, and drawing rules.

## Operators

- Detector/keypoint model, often external package or ONNX.
- Keypoint score thresholding and coordinate normalization.
- Topology-specific drawing to an RGB condition map.

## DinoML Notes

Do not reuse the human OpenPose topology. The keypoint schema and renderer need separate metadata.

## Sources

- `agents/plans/auxiliary/control_preprocessors.md`
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`

