# Audio Separation

## Coverage

- Diffusers: not covered.
- Transformers: not covered as a general audio-separator surface.
- Third-party/UI: Wan2GP references `audio-separator` and local preprocessing utilities.

## Runtime Contract

Audio separation splits mixed audio into stems or speaker/source components before downstream TTS, voice conversion, or audio-conditioned video. Output is one or more waveform tracks plus sample-rate and timing metadata.

## Operators

- Model dependent: source-separation networks, STFT/mel frontend, overlap-add chunking.
- Audio resampling, normalization, and file IO.

## DinoML Notes

Start as an external provider. If native support is selected, require an exact model family audit because separator architectures vary widely.

## Sources

- `agents/plans/auxiliary/audio_auxiliary.md`
- `deepbeepmeep/Wan2GP/preprocessing/speakers_separator.py:36`

