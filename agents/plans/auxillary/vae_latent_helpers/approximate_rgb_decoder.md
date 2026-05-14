# Approximate RGB Decoders

## Coverage

- Diffusers: not covered as a named model family.
- Transformers: not covered.
- Third-party/UI: A1111 uses approximate VAE preview paths.

## Runtime Contract

Approximate decoders turn latent tensors into preview RGB images quickly, often through tiny learned or fixed mappings rather than full VAE decode. The important contract is latent format, scaling, output range, and preview-only quality.

## Operators

- Usually small Conv2d or linear/1x1-style mapping.
- Latent scaling/shift.
- Clamp/range conversion to display image.

## DinoML Notes

Mark output quality as preview-only. Do not substitute approximate decoders for parity-critical VAE decode unless the workflow explicitly selects preview mode.

## Sources

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/sd_vae_approx.py`
- `agents/plans/auxiliary/vae_and_latent_helpers.md`

