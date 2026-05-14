# Wan VACE, MultiTalk, Camera, and Trajectory Conditioning

## Coverage

- Diffusers: Wan base pipeline coverage exists in `agents/plans/diffusers/wan/report.md`, but these UI conditioning paths are not one generic Diffusers feature.
- Transformers: Wav2Vec2/Whisper components may be Transformers-covered when used for audio features.
- Third-party/UI: Wan2GP and Comfy custom Wan stacks.

## Runtime Contract

Wan UI workflows add mask/control video, trajectory, camera embedding, VACE context encoding, reference images, and MultiTalk audio conditioning. These change denoiser inputs, conditioning token/layout contracts, and scheduling/runtime state.

## Operators

- Wan transformer/audio projection modules.
- Video mask/control preprocessing.
- Camera/trajectory embedding math.
- Audio encoder/projector if MultiTalk is enabled.

## DinoML Notes

Treat these as separate conditioning reports rather than base Wan support. First parity can reject them unless a workflow explicitly selects a conditioning schema.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:912`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:1048`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/any2video.py:1085`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/multitalk/multitalk_model.py:353`

