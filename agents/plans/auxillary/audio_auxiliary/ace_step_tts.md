# ACE-Step and ACE-Step 1.5 TTS

## Coverage

- Diffusers: ACE-Step has an active Diffusers planning report for the diffusion/audio model family.
- Transformers: text/audio encoders may overlap, but the ACE-Step pipelines are not pure Transformers models.
- Third-party/UI: Wan2GP includes TTS-oriented ACE-Step and ACE-Step1.5 pipeline wrappers.

## Runtime Contract

ACE-Step pipelines are audio diffusion/generation workflows with their own transformer, scheduler, autoencoder/vocoder, and text/audio conditioning paths. Wan2GP wraps them as TTS/audio tools rather than base image diffusion pipelines.

## Operators

- Diffusion transformer blocks.
- Audio latent autoencoder or Oobleck codec.
- Scheduler loop and guidance.
- Text/token conditioning and waveform decode.

## DinoML Notes

Reuse `agents/plans/diffusers/ace_step/report.md` for base model coverage, then add TTS pipeline-specific prompt/audio frontend and codec reports when targeting UI parity.

## Sources

- `agents/plans/diffusers/ace_step/report.md`
- `deepbeepmeep/Wan2GP/models/TTS/ace_step/pipeline_ace_step.py`
- `deepbeepmeep/Wan2GP/models/TTS/ace_step15/pipeline_ace_step15.py`

