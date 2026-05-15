# Whisper Audio Encoder

## Coverage

- Diffusers: not covered as an auxiliary encoder.
- Transformers: covered under `src/transformers/models/whisper`.
- Third-party/UI: Comfy vendors a Whisper Large V3 style encoder for audio conditioning.

## Runtime Contract

Comfy's encoder converts audio to mono, pads/truncates to 30 seconds at 16 kHz, computes 128-bin log-mel features with `n_fft=400` and `hop_length=160`, then runs two Conv1d layers and 32 Transformer encoder layers with hidden size 1280 and 20 heads. It returns the final sequence and intermediate hidden states.

## Operators

- Mel spectrogram/STFT frontend.
- Conv1d, GELU, LayerNorm.
- MHA, FFN, residual add.
- Positional embedding.

## DinoML Notes

Reuse Transformers Whisper report for canonical configs, but the Comfy feature extractor and "return all hidden states" contract are auxiliary-specific and should be explicit for audio-conditioned video.

## Sources

- `transformers/src/transformers/models/whisper`
- `Comfy-Org/ComfyUI/comfy/audio_encoders/whisper.py`

