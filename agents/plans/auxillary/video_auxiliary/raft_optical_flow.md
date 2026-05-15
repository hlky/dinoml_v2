# RAFT Optical Flow

## Coverage

- Diffusers: not covered.
- Transformers: not covered in the inspected tree.
- Third-party/UI: Wan2GP vendors RAFT; Comfy can route through torchvision RAFT helpers.

## Runtime Contract

RAFT estimates optical flow between two frames. The local Wan2GP model normalizes images to `[-1, 1]`, runs feature and context encoders, builds a correlation volume, iteratively updates coordinate grids for a configured iteration count, and upsamples flow by learned convex combination or bilinear fallback.

## Operators

- Conv encoders, normalization, recurrent/update block.
- Correlation volume lookup.
- Coordinate grids, iterative loop, detach/update semantics.
- Unfold, softmax, weighted upsample.

## DinoML Notes

RAFT is loop/state heavy. First integration should keep it as an external preprocessing provider unless correlation and iterative update operators are admitted explicitly.

## Sources

- `deepbeepmeep/Wan2GP/preprocessing/raft/raft.py`
- `deepbeepmeep/Wan2GP/preprocessing/flow.py:19`
- `Comfy-Org/ComfyUI/comfy_extras/nodes_void.py:15`

