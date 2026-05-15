# VAE and Latent Helpers

## Why This Matters

UIs use fast/approximate VAEs for preview, latent upscale workflows, asymmetric
upscale VAEs, Wan upscaling VAEs, and temporal/video/audio autoencoders. Some
of this overlaps `diffusers`, but the useful UI behavior often depends on
special load keys, preview formats, tiling, or temporal chunking.

## Model Families

- TAESD image preview VAEs.
- TAEHV temporal/spatial video preview decoder.
- Approximate RGB decoders.
- AutoencoderSmall and asymmetric VAE upscalers.
- Wan VAE upscale path.
- Audio/video causal VAEs in Comfy and Wan2GP.

## Code Anchors

- `AUTOMATIC1111/stable-diffusion-webui/modules/sd_vae_approx.py:10`
  approximate VAE path.
- `AUTOMATIC1111/stable-diffusion-webui/modules/sd_vae_taesd.py:108`
  TAESD path.
- `AUTOMATIC1111/stable-diffusion-webui/modules/models/sd3/other_impls.py:314`
  custom SD3 T5 tokenizer/layers, adjacent to nonstandard model helper paths.
- `Comfy-Org/ComfyUI/nodes.py:729`
  `VAELoader`.
- `Comfy-Org/ComfyUI/nodes.py:755`
  `load_taesd`.
- `Comfy-Org/ComfyUI/comfy/taesd/taesd.py`.
- `Comfy-Org/ComfyUI/comfy/taesd/taehv.py`.
- `Comfy-Org/ComfyUI/comfy/ldm/modules/temporal_ae.py`.
- `Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/causal_video_autoencoder.py`.
- `Comfy-Org/ComfyUI/comfy/ldm/lightricks/vae/causal_audio_autoencoder.py`.
- `vladmandic/sdnext/modules/vae/sd_vae_taesd.py:139`
  TAESD/AutoencoderSmall handling.
- `vladmandic/sdnext/modules/upscaler_vae.py:29`
  asymmetric VAE upscaler.
- `vladmandic/sdnext/modules/upscaler_vae.py:68`
  Wan VAE upscale.
- `vladmandic/sdnext/modules/video_models/video_load.py:213`
  Wan upscale VAE load.
- `deepbeepmeep/Wan2GP/models/wan/modules/vae.py:928`
  Wan VAE.
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/tcdecoder.py:170`
  TAEHV temporal/spatial decoder.

## DinoML Gap

Moderate. These should be represented as explicit encode/decode/preview
contracts with latent channel layout, scaling factor, spatial/temporal downscale,
tiling, dtype, and output range. For video/audio VAEs, temporal chunking and
overlap should be artifact-visible.

