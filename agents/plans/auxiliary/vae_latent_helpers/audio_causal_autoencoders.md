# Audio and Causal Autoencoders

## Coverage

- Diffusers: Oobleck is covered as `AutoencoderOobleck`; other UI causal audio autoencoders are third-party.
- Transformers: not covered.
- Third-party/UI: Comfy Lightricks audio VAE and Wan2GP audio codec stacks.

## Runtime Contract

Audio autoencoders convert waveform or mel-like audio into latent sequences and back. Causal variants expose chunking, stride/downscale, overlap, and sample-rate constraints.

## Operators

- Conv1d and ConvTranspose1d.
- Residual/dilated audio blocks.
- Causal state/chunk handling.
- Latent bottleneck sampling or deterministic encode.

## DinoML Notes

Audio codec contracts need sample rate, channel count, temporal scale, chunk size, and output waveform range. Treat stochastic encode mode as separate from deterministic decode-only use.

## Sources

- `diffusers/src/diffusers/models/autoencoders/autoencoder_oobleck.py`
- `Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/causal_audio_autoencoder.py`
- `Comfy-Org/ComfyUI/comfy/ldm/audio/autoencoder.py`

