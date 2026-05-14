# Depth Anything V3 Video

## Coverage

- Diffusers: not covered.
- Transformers: Depth Anything is covered for image depth; V3 video wrapper is UI-vendored/third-party.
- Third-party/UI: Wan2GP includes `DepthV3VideoAnnotator`.

## Runtime Contract

The video annotator produces temporally coherent depth maps from frame sequences. The important auxiliary contract includes frame chunking, temporal context, resize policy, and output sequence assembly, not just the depth backbone.

## Operators

Needs source-specific follow-up. Expected surface includes image/video transformer depth inference plus temporal chunking and depth normalization.

## DinoML Notes

Keep separate from single-image Depth Anything. If native support is added, represent frame count, chunk overlap, and output depth normalization explicitly.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/preprocessing/depth_anything_v3/depth.py:266`
- `X:/H/transformers/src/transformers/models/depth_anything`

