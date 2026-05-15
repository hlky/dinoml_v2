# FILM and GIMM Frame Interpolation

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: Comfy frame interpolation nodes expose FILM/GIMM-style models.

## Runtime Contract

These are alternate frame interpolation models. They consume neighboring frames and produce in-between frames, with model-specific scale/padding and temporal recursion policies.

## Operators

Not deeply audited in this pass. Expected surface includes image conv/residual blocks, resize/pyramid operations, optical-flow or synthesis branches, and blend/postprocess logic.

## DinoML Notes

Keep as separate auxiliary targets from RIFE. Do not assume RIFE's flow/mask contract applies to FILM or GIMM without source inspection.

## Sources

- `Comfy-Org/ComfyUI/comfy_extras/nodes_frame_interpolation.py:9`
- `agents/plans/auxiliary/video_auxiliary.md`

