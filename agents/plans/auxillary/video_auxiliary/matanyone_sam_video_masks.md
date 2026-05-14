# MatAnyone and SAM Video Masks

## Coverage

- Diffusers: not covered.
- Transformers: SAM, SAM2, and SAM3 are covered as model families; MatAnyone is not covered.
- Third-party/UI: video mask propagation appears through UI helper stacks.

## Runtime Contract

These tools generate or propagate masks across frames. SAM-style models provide image embeddings and prompt-conditioned masks; video propagation adds temporal memory/state and frame-by-frame update policy. Matting tools additionally output alpha mattes rather than hard masks.

## Operators

- SAM vision encoder, prompt encoder, mask decoder.
- Video memory/state update for SAM2/SAM3-style trackers.
- Mask resize/crop/threshold and alpha compositing.

## DinoML Notes

Use Transformers SAM reports for model graph coverage where possible, but model the auxiliary mask lifecycle separately: cached image embeddings, prompt points/boxes/masks, per-frame outputs, and mask postprocess.

## Sources

- `X:/H/transformers/src/transformers/models/sam`
- `X:/H/transformers/src/transformers/models/sam2`
- `X:/H/transformers/src/transformers/models/sam3`
- `agents/plans/auxiliary/video_auxiliary.md`

