# Oobleck Audio VAE

## Coverage

- Diffusers: covered as `AutoencoderOobleck`.
- Transformers: not covered.
- Third-party/UI: Comfy and Wan2GP include Oobleck-style audio autoencoders.

## Runtime Contract

Comfy's `AudioOobleckVAE` encodes stereo or multi-channel waveform tensors through weight-normalized Conv1d residual blocks and strided downsampling, splits mean/scale for a diagonal Gaussian bottleneck, samples latents, and decodes through ConvTranspose1d or nearest-upsample Conv1d blocks.

## Operators

- Conv1d, ConvTranspose1d, weight norm.
- Residual units with dilations 1/3/9.
- ELU or SnakeBeta activation.
- Softplus/log variance sampling path for stochastic encode.

## DinoML Notes

For inference generation, decode-only may be enough in some paths. If encode is supported, define deterministic mode versus stochastic sampling and expose sample-rate/time-downscale metadata.

## Sources

- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_oobleck.py`
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/audio/autoencoder.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/TTS/ace_step15/models/autoencoder_oobleck.py`

