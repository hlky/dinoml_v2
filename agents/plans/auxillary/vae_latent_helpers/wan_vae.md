# Wan VAE

## Coverage

- Diffusers: covered as `AutoencoderKLWan`.
- Transformers: not covered.
- Third-party/UI: Comfy and Wan2GP include original Wan VAE implementations.

## Runtime Contract

Wan VAE is a causal 3D video autoencoder. The inspected Comfy implementation uses causal Conv3d, RMS norm, residual blocks, 2D/3D down/up sampling, VAE attention folded over frames, temporal cache maps, and chunked encode/decode over frame groups.

## Operators

- Causal Conv3d with cache.
- Conv2d over folded `(B*T)` frames.
- RMS norm, SiLU, residual add.
- Attention in VAE middle/resolution blocks.
- Temporal upsample/downsample and chunk stitching.

## DinoML Notes

This is a core video codec target. Preserve causal temporal cache and frame grouping exactly; naive full-video Conv3d may not match memory/runtime behavior.

## Sources

- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py`
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/wan/vae.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/modules/vae.py`

