# UI Model Support Audits

This directory contains one review per cloned UI under `H:/uis`.

These notes track all model support visible in the UIs: main image/video/audio
models, adapters, checkpoints, quantized/runtime formats, and auxiliary models.
They complement `agents/plans/auxiliary/`, which focuses only on non-core
auxiliary categories.

## Reviews

- [feature_reference.md](feature_reference.md): cross-UI reference list of
  model families, adapters, auxiliary features, runtimes, and product surfaces.
- [automatic1111/review.md](automatic1111/review.md)
- [comfyui/review.md](comfyui/review.md)
- [sdnext/review.md](sdnext/review.md)
- [invokeai/review.md](invokeai/review.md)
- [forge/review.md](forge/review.md)
- [reforge/review.md](reforge/review.md)
- [forge_classic/review.md](forge_classic/review.md)
- [wan2gp/review.md](wan2gp/review.md)
- [swarmui/review.md](swarmui/review.md)
- [stabilitymatrix/review.md](stabilitymatrix/review.md)

## Common Model Categories

- Main image models: SD 1.x, SD 2.x, SDXL, SD3/3.5, Flux, Flux 2, Qwen Image,
  Z-Image, AuraFlow, HunyuanDiT/Image, Kandinsky, HiDream, Segmind SSD.
- Main video models: Stable Video Diffusion, Wan, Hunyuan Video, LTX/LTX-2,
  Mochi, CogVideo/CogVideoX, Cosmos/world models.
- Audio models: Stable Audio, ACE-Step, MMAudio, TTS families, voice cloning,
  vocoders, audio encoders.
- Adapters and conditioning: LoRA, LyCORIS/OFT where present, textual
  inversion, embeddings, ControlNet, T2I-Adapter, IP-Adapter, Control LoRA,
  LLLite, reference/style/identity adapters.
- Model components: VAE, approximate VAE, CLIP/T5/text encoders, CLIP Vision,
  SigLIP/Qwen vision encoders, GLIGEN, style models.
- Image upscaling and restoration.
- Face restoration, detection, detailer, and paste-back workflows.
- ControlNet/T2I/IP/Flux conditioning preprocessors.
- Segmentation, detection, and mask generation.
- Captioning, tagging, interrogators, and prompt helpers.
- VAE, approximate VAE, latent preview, and latent upscaling helpers.
- Video interpolation, optical flow, video super-resolution, and video masks.
- Audio encoders, video-to-audio, TTS, vocoders, and voice tools.
- Quantized formats, sparse attention, custom kernels, and memory residency.
