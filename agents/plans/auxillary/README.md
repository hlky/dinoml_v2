# Auxiliary UI Model Gap Audit

This directory tracks UI-facing model and feature gaps found in the cloned
repositories under `H:/uis`.

## Scope

The goal is to identify UI features and auxiliary model families that are not
covered merely by treating `diffusers` and `transformers` as model references.
Some entries do use those packages for part of the stack, but still require
extra loader formats, postprocessing, preprocessors, model categories, runtime
contracts, or non-core packages.

## Category Files

- [upscalers_and_restoration.md](upscalers_and_restoration.md): Spandrel,
  ESRGAN/RealESRGAN/SwinIR-style upscalers, GFPGAN, CodeFormer, face detailers.
- [control_preprocessors.md](control_preprocessors.md): ControlNet/T2I/IP/Flux
  conditioning preprocessors such as HED, MLSD, pose, depth, normals, lineart,
  semantic segmentation.
- [segmentation_detection_masks.md](segmentation_detection_masks.md): SAM,
  SAM2/SAM3, GroundingDINO, RT-DETR, YOLO, CLIPSeg, rembg, mask workflows.
- [video_auxiliary.md](video_auxiliary.md): RIFE/FILM/GIMM, RAFT/flow,
  FlashVSR, Wan/LTX/video VAEs, schedulers, video conditioning.
- [audio_auxiliary.md](audio_auxiliary.md): MMAudio, TTS, voice conversion,
  speech, diarization, audio encoders/VAEs/vocoders.
- [captioning_and_prompt_tools.md](captioning_and_prompt_tools.md): BLIP,
  OpenCLIP/clip-interrogator, DeepDanbooru, WD taggers, prompt helpers.
- [vae_and_latent_helpers.md](vae_and_latent_helpers.md): TAESD/TAEHV,
  approximate VAEs, latent upscalers, asymmetric/Wan VAE upscalers.
- [quantization_and_runtime_kernels.md](quantization_and_runtime_kernels.md):
  GGUF/NVFP4/Nunchaku, sparse attention, Triton kernels, low-VRAM runtime
  support.
- [ui_coverage_map.md](ui_coverage_map.md): where each cloned UI contributes
  useful evidence.

## Highest-Value Follow-Ups

1. Build an auxiliary-model taxonomy separate from diffusion model taxonomy:
   upscaler, restoration, detection, segmentation, pose, depth, normal,
   line/edge, optical flow, interpolation, video VAE, audio encoder, vocoder,
   TTS, speaker/audio utility, scheduler, qtype/provider.
2. Pick Spandrel-style upscaling as the first bounded non-diffusion auxiliary
   runtime target: one RRDB/RealESRGAN-like model, tiled image input,
   explicit tile/overlap artifact metadata, PyTorch reference validation, and a
   CUDA/provider checklist.
3. Pick one preprocessor family as a second target, likely HED/PiDiNet/MLSD or
   Depth Anything. Keep it independent from ControlNet model execution and model
   the image-to-condition output contract explicitly.
4. Record SAM/YOLO/GroundingDINO/FaceDetailer as workflow-level gaps. Even if
   transformers covers base modules, DinoML still needs mask/detection
   postprocess and stateful selection contracts.
5. Treat Wan2GP as a future large-model runtime requirements source: video shape
   contracts, flow schedulers, tiled temporal VAEs, audio-video conditioning,
   sparse attention providers, and qtype/residency policy.

