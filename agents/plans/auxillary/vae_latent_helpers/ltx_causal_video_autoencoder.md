# LTX Causal Video Autoencoder

## Coverage

- Diffusers: covered as `AutoencoderKLLTXVideo`.
- Transformers: not covered.
- Third-party/UI: Comfy and Wan2GP include Lightricks/LTX causal video autoencoder code.

## Runtime Contract

LTX video codecs use causal video autoencoder blocks, temporal/spatial compression, and model-specific latent scale/shift handling. They are consumed by LTX and LTX-2 video pipelines and can have audio-video conditioning adjacent to the codec.

## Operators

- 3D/causal conv video autoencoder blocks.
- Temporal chunking and overlap policy.
- Latent scaling and packing.

## DinoML Notes

Reuse the Diffusers LTX autoencoder report where available, but keep UI codec chunking and latent boundary transforms separate from the denoiser.

## Sources

- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx.py`
- `H:/uis/Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/causal_video_autoencoder.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/ltx_video/models/autoencoders/causal_video_autoencoder.py`

