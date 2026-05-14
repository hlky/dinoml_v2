# NLF, Vista4D, and Camera Helpers

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: Wan2GP includes NLF pose and Vista4D helpers.

## Runtime Contract

These are 3D/camera conditioning helpers used around video generation. NLF handles multiperson pose-like processing; Vista4D paths build depth/camera point renderings and add runtime modules for camera/geometry-aware conditioning.

## Operators

- Pose/depth/camera preprocessing.
- Geometry transforms and raster/point rendering.
- Model-specific conditioning tensor assembly.

## DinoML Notes

Treat outputs as structured conditioning artifacts, not just images. Camera matrices, frame coordinates, and trajectory metadata should remain artifact-visible.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/models/wan/scail/scail_pose_nlf.py:69`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/scail/nlf/multiperson_model.py:17`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/vista4d/preprocess.py:147`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/vista4d/runtime.py:6`

