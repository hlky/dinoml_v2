# TAESD

## Coverage

- Diffusers: covered as `AutoencoderTiny`.
- Transformers: not covered.
- Third-party/UI: Comfy and A1111/SD.Next include TAESD preview/fast VAE paths.

## Runtime Contract

Comfy's TAESD has a small Conv2d encoder/decoder with residual blocks, stride-2 downsampling, nearest upsampling, and optional group norm midblocks. It exposes latent scaling helpers: raw latents to `[0, 1]` and back using magnitude 3 and shift 0.5. Flux2 variants unpack/repack 128-channel latent tensors into 32 channels with 2x spatial expansion.

## Operators

- Conv2d, ReLU, GroupNorm.
- Upsample, stride conv downsample.
- Tanh clamp and affine latent scaling.
- Pixel/latent pack-unpack for Flux2 variant.

## DinoML Notes

This is a fast preview/codec path, not identical to full AutoencoderKL. Track latent channels, scale/shift parameters, and model variant explicitly.

## Sources

- `diffusers/src/diffusers/models/autoencoders/autoencoder_tiny.py`
- `Comfy-Org/ComfyUI/comfy/taesd/taesd.py`
- `AUTOMATIC1111/stable-diffusion-webui/modules/sd_vae_taesd.py`

