# BigVGAN Vocoder

## Coverage

- Diffusers: not covered.
- Transformers: not covered as a native Transformers model.
- Third-party/UI: Comfy and Wan2GP vendor BigVGAN variants for audio generation/TTS/vocoder paths.

## Runtime Contract

BigVGAN converts mel or acoustic features to waveform. Comfy's implementation uses a pre-conv, transposed Conv1d upsamplers, AMP residual blocks with Snake/SnakeBeta anti-aliased periodic activations, post-conv, and final tanh.

## Operators

- Conv1d, ConvTranspose1d.
- Dilated residual Conv1d blocks.
- Snake/SnakeBeta activations and anti-aliased activation wrappers.
- Tanh output.

## DinoML Notes

Audio providers need strong Conv1d and transposed Conv1d coverage. SnakeBeta should be added as a named activation if BigVGAN enters native scope.

## Sources

- `Comfy-Org/ComfyUI/comfy/ldm/mmaudio/vae/bigvgan.py`
- `deepbeepmeep/Wan2GP/models/TTS/index_tts2/BigVGAN`
- `deepbeepmeep/Wan2GP/postprocessing/mmaudio/ext/bigvgan`

