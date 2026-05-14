# PiDiNet

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: vendored in InvokeAI from ControlNet auxiliary preprocessing code.

## Runtime Contract

PiDiNet is an edge detector. InvokeAI's vendored implementation defines partial/difference convolution variants (`cv`, `cd`, `ad`, `rd`) and a `PiDiNet` backbone with four stages, depthwise/pointwise conv blocks, optional spatial attention, dilation modules, and side-output reductions.

`controlnet_aux.PidiNetDetector` defaults to `table5_pidinet.pth`, converts PIL/NumPy input to HWC3, resizes to `detect_resolution`, flips RGB to BGR, scales to 0..1, takes the final network edge output, and optionally applies `safe`, `apply_filter`, and `scribble` postprocessing. The processor registry exposes the same underlying detector as `softedge_pidinet`, `softedge_pidsafe`, `scribble_pidinet`, and `scribble_pidsafe`.

## Operators

- Conv2d, depthwise Conv2d, pointwise Conv2d.
- MaxPool2d, ReLU, Sigmoid.
- Custom PDC forms that can be converted to vanilla 3x3 or 5x5 convolution for inference.
- Multi-scale side-output fusion and resize.

## DinoML Notes

This is a good graph-rewrite candidate: convert PDC blocks to ordinary convolution during weight load when the checkpoint uses converted weights or when a deterministic transform is available. Preserve detector resize, threshold, and output polarity as artifact-visible preprocessor metadata.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/pidi/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `H:/uis/invoke-ai/InvokeAI/invokeai/backend/image_util/pidi/model.py`
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`
