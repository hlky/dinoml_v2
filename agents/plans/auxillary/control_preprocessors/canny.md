# Canny

## Coverage

- Diffusers: not a model component. Diffusers consumes the resulting condition image through ControlNet/T2I-style pipelines.
- Transformers: not covered.
- Third-party/UI: OpenCV-style preprocessing in UIs; no learned weights.

## Runtime Contract

Input is an RGB or grayscale image resized according to the control unit policy. Output is usually a 1-channel or RGB edge map with caller-selected low/high thresholds and optional resize back to generation resolution.

`controlnet_aux.CannyDetector` makes this contract explicit: it converts input to 3-channel HWC, resizes to `detect_resolution` (default 512), runs `cv2.Canny` with `low_threshold=100` and `high_threshold=200` by default, converts the edge map back to HWC3, and resizes to `image_resolution` (default 512).

## DinoML Notes

Treat Canny as CPU/data-pipeline preprocessing first. If moved into DinoML runtime, model it as image preprocessing with explicit threshold, blur, resize, output channel, dtype, and range metadata rather than as a neural model.

## Sources

- `agents/plans/auxiliary/control_preprocessors.md`
- `H:/controlnet_aux/src/controlnet_aux/canny/__init__.py`
- `H:/uis/Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py:310`
