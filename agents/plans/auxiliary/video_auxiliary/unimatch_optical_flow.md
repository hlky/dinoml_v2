# UniMatch Optical Flow

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: listed as a video auxiliary model family; exact local implementation was not deeply audited in this pass.

## Runtime Contract

UniMatch-style optical flow consumes pairs or sequences of frames and produces dense flow fields. Compared with RAFT, it may use transformer/global matching components and different refinement stages.

## Operators

- Image feature encoder.
- Correlation or global matching.
- Attention/transformer matching blocks depending on implementation.
- Flow upsample/refinement and warping.

## DinoML Notes

Keep separate from RAFT. A future audit should install or inspect the exact package/UI code before admitting operators.

## Sources

- `agents/plans/auxiliary/video_auxiliary.md`

