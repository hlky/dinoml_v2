# SeedVC, OpenVoice, SpeechBrain, and Pyannote Utilities

## Coverage

- Diffusers: not covered.
- Transformers: partial overlap only where submodels use standard encoders.
- Third-party/UI: Wan2GP integrates SeedVC, OpenVoice modules, SpeechBrain-style utilities, and Pyannote diarization.

## Runtime Contract

These are voice conversion, speaker, diarization, and speech utility surfaces. They involve audio file decoding, speaker/reference embeddings, segmentation/diarization records, conversion models, and waveform output.

## Operators

- Model dependent: speech encoders, speaker embeddings, sequence models, vocoders.
- Audio chunking and resampling.
- Diarization segment metadata and per-speaker routing.

## DinoML Notes

Keep them as external auxiliary providers until a specific target is selected. The runtime needs sample-rate metadata, time spans, speaker IDs, and chunking records, not just tensors.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/seedvc/api.py:210`
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/seedvc/seed_vc_wrapper.py`
- `H:/uis/deepbeepmeep/Wan2GP/preprocessing/speakers_separator.py:36`
- `agents/plans/auxiliary/audio_auxiliary.md`

