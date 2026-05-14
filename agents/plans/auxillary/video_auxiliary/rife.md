# RIFE Frame Interpolation

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: Comfy and Wan2GP include RIFE/IFNet frame interpolation implementations.

## Runtime Contract

RIFE takes two frames plus a timestep and predicts an interpolated frame. Wan2GP's `IFNet` uses multi-scale flow refinement, feature encoders, warping through `grid_sample`, mask blending, pixel shuffle, and cached coordinate grids. Output is an RGB frame at source resolution.

## Operators

- Conv2d, ConvTranspose2d, LeakyReLU, residual conv blocks.
- Interpolate, PixelShuffle.
- Grid generation, grid_sample/warp, sigmoid mask blend.
- Temporal insertion policy for arbitrary interpolation ratios.

## DinoML Notes

The hard runtime primitive is `grid_sample` with flow-normalized coordinates. Preserve timestep, padding-to-multiple, frame order, and cached grid policy.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/rife/RIFE_V4.py`
- `H:/uis/Comfy-Org/ComfyUI/comfy_extras/frame_interpolation_models/ifnet.py`

