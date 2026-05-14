# Chatterbox TTS and Voice Conversion

## Coverage

- Diffusers: not covered.
- Transformers: overlaps only through internal language/audio components if mapped separately.
- Third-party/UI: Wan2GP includes Chatterbox multilingual TTS and VC wrappers.

## Runtime Contract

`ChatterboxPipeline` loads `ChatterboxMultilingualTTS`, prepares optional reference audio conditionals, supports language IDs, and exposes generation controls such as exaggeration, pace/CFG weight, temperature, repetition penalty, min-p, and top-p. Output is waveform plus sample rate, usually 44.1 kHz.

## Operators

- Text/audio conditioning.
- TTS generation stack with internal voice encoder, token generator, and waveform decoder.
- Sampling controls and reference-audio preprocessing.

## DinoML Notes

Treat as a composite TTS product target. Native support should begin by identifying and separately auditing `ve`, `s3gen`, `t3`, and condition modules.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/chatterbox/pipeline.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/chatterbox/mtl_tts.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/chatterbox/vc.py`

