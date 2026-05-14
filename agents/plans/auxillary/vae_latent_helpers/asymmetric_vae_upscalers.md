# Asymmetric VAE Upscalers

## Coverage

- Diffusers: covered as `AsymmetricAutoencoderKL`.
- Transformers: not covered.
- Third-party/UI: SD.Next exposes asymmetric VAE and Wan VAE upscale helpers.

## Runtime Contract

Asymmetric VAE upscalers encode/decode with asymmetric treatment for masked/inpaint or upscaling workflows. UI usage couples the model to tiling, scale selection, and latent/image resize policy.

## Operators

- AutoencoderKL-like conv/resnet/downsample/upsample blocks.
- Optional mask/image conditioning depending on selected path.
- Tiling and output stitching.

## DinoML Notes

Reuse Diffusers model coverage, but record UI scale/tiling contracts separately. Do not treat it as the same artifact as normal AutoencoderKL.

## Sources

- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_asym_kl.py`
- `H:/uis/vladmandic/sdnext/modules/upscaler_vae.py`

