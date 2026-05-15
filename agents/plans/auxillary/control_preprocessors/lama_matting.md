# LaMa, AnimeFace Segment, and Matting Helpers

## Coverage

- Diffusers: not covered.
- Transformers: not covered for LaMa/matting as named UI preprocessors.
- Third-party/UI: exposed through control/detailer/inpaint helper stacks.

## Runtime Contract

These helpers produce masks, segmentation maps, or inpaint-ready image/mask pairs. LaMa-style paths are image-inpainting/mask-refinement models rather than diffusion denoisers.

## Operators

- Model dependent: usually conv/residual image-to-image networks.
- Mask thresholding, dilation/erosion/blur, alpha compositing.
- Optional segmentation class filtering.

## DinoML Notes

This is design-sensitive because outputs feed inpaint/control workflows. First support can be external-provider based, but mask shape, polarity, feathering, and resize must be explicit.

## Sources

- `agents/plans/auxiliary/control_preprocessors.md`
- `lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`

