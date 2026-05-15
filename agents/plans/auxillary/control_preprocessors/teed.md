# TEED

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: typically routed through `controlnet_aux` or UI-vendored detector code.

## Runtime Contract

TEED is an edge detector used as a ControlNet condition preprocessor. `controlnet_aux.TEEDdetector` loads a `TED` network from a Hugging Face file, with README usage pointing at `fal-ai/teed` and `5_model.pth`. The call path converts input to HWC3, resizes to `detect_resolution` (default 512), runs NCHW float inference, resizes each side edge output, averages them, applies sigmoid, optionally applies `safe_step(edge, safe_steps)` with `safe_steps=2` by default, scales to `uint8`, and resizes to the original output size.

## Operators

- Conv2d, ConvTranspose2d, MaxPool2d, GroupNorm, ReLU-style activations, PixelShuffle.
- Dense blocks, up-conv blocks, and multi-output edge fusion.
- Resize, sigmoid, safe-step thresholding, HWC/NCHW conversion.

## DinoML Notes

Do not fold TEED into ControlNet support. It needs a separate image-to-condition contract with detector weight identity, resize policy, side-output fusion, `safe_steps`, and output range. The neural graph is small enough to be a candidate for native conv coverage after the preprocessing contract is stable.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/teed/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/teed/ted.py`
- `agents/plans/auxiliary/control_preprocessors.md`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py:715`
- `vladmandic/sdnext/modules/control/processor.py:34`
