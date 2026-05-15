# Diffusers initial candidate taxonomy

Source basis:

```text
Diffusers checkout: diffusers
Commit inspected: b3a515080752a3ba7ca92161e25530c7f280f629
Recent upstream context: post release 0.38.0 plus local fixes
Primary scan paths:
  src/diffusers/pipelines
  src/diffusers/models
  src/diffusers/schedulers
  src/diffusers/image_processor.py
  src/diffusers/video_processor.py
```

This is an initial planning inventory, not a completed operator report. The scan
found roughly 283 pipeline classes, 128 model-ish classes, 53 scheduler classes,
and a broad family split across UNets, 2D/3D diffusion transformers, VAEs,
ControlNets, adapters, attention processors, schedulers, image/video/audio
processors, and loader mixins.

## Why diffusers needs a different assessment shape

Transformers reports can often start from `modeling_*.py`. Diffusers reports
should start from the pipeline and follow the component graph. For example,
Stable Diffusion text-to-image uses `pipeline_stable_diffusion.py` for prompt
encoding, latent initialization, CFG, scheduler stepping, and VAE decode; the
denoiser lives in `models/unets/unet_2d_condition.py`; UNet blocks are assembled
from `unet_2d_blocks.py`, `resnet.py`, `attention.py`, and
`attention_processor.py`; scheduler math lives in `scheduling_*.py`; VAE decode
lives under `models/autoencoders/`.

Newer families such as Flux and SD3 move the main denoiser from UNet blocks to
patch/token transformer models, but the full runtime still spans pipeline
helpers, scheduler state, text encoders, VAE code, and shared attention,
embedding, and normalization modules.

Layout note: Diffusers/PyTorch source usually expresses image and latent maps as
NCHW and video maps as NCDHW, while Dinoml should generally prefer
NHWC/NDHWC/channel-last for vision optimization when a region is local and fully
controlled. Candidate reports should preserve source axes for semantic parity,
then identify guarded layout-elision or layout-translation regions with the
needed axis rewrites, weight transforms, and no-layout-translation guards.

## Candidate classes

### A. Baseline latent image pipelines

These are the best first reports because they exercise the classic latent
diffusion loop without video temporal state or many side adapters.

| Candidate | Source anchors | Denoiser | Scheduler shape | Why it matters | Priority |
| --- | --- | --- | --- | --- | --- |
| Stable Diffusion 1.x text-to-image | `pipelines/stable_diffusion/pipeline_stable_diffusion.py`, `models/unets/unet_2d_condition.py` | 2D conditional UNet | broad compatible set such as DDIM/Euler/LMS/PNDM/DPM variants | Smallest useful end-to-end latent diffusion stack: CLIP text, source NCHW latents with NHWC optimization candidates, UNet ResNet+cross-attn, CFG, VAE encode/decode accounting | P0 |
| Stable Diffusion 2.x text-to-image | `pipelines/stable_diffusion/pipeline_stable_diffusion.py`, `models/unets/unet_2d_condition.py` plus SD2 component configs | 2D conditional UNet | broad compatible set, checkpoint-specific defaults | Similar flow to SD 1.x but different text encoder/projection and checkpoint variants; should be a separate report | P1 |
| Stable Diffusion XL | `pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`, `UNet2DConditionModel` | 2D conditional UNet | broad compatible set, checkpoint-specific defaults | Adds dual text encoders, pooled prompt embeddings, size/crop conditioning, larger UNet widths | P1 |
| PixArt Alpha/Sigma | `pipelines/pixart_alpha`, `models/transformers/pixart_transformer_2d.py` | 2D transformer | diffusion scheduler loop | Good bridge from image UNet to patch-token DiT without the full SD3/Flux component surface | P1 |

Initial report recommendation: start with Stable Diffusion 1.5 or a tiny SD
checkpoint for source tracing, then SDXL for conditioning variation.

### B. Modern DiT / joint-attention image pipelines

These matter because current image generation is moving away from UNet
convolution-heavy denoisers toward token/patch transformers with adaptive norms,
joint text-image attention, QK norm, and flow-matching schedulers.

| Candidate | Source anchors | Denoiser | Distinct ops | Priority |
| --- | --- | --- | --- | --- |
| Stable Diffusion 3 / 3.5 | `pipelines/stable_diffusion_3`, `models/transformers/transformer_sd3.py`, `models/attention.py` | `SD3Transformer2DModel` with `JointTransformerBlock` | patch embed/unpatchify, joint attention, AdaLayerNormZero/Continuous, skip-layer guidance, FlowMatch scheduler | P0 |
| Flux / Flux.1 | `pipelines/flux`, `models/transformers/transformer_flux.py`, `attention_dispatch.py` | `FluxTransformer2DModel` | packed latent image IDs, text/image token concatenation, RoPE on image/text sequence, QK norm, guidance embedding, true CFG variants | P0 |
| Sana | `pipelines/sana`, `models/transformers/sana_transformer.py`, `attention_processor.py` | efficient 2D transformer | multiscale/linear attention variants and mobile-ish design pressure | P2 |
| AuraFlow / Lumina / Hunyuan image / QwenImage / ZImage | corresponding pipeline dirs and transformer files | 2D transformer variants | family-specific adaptive norm, RoPE, text conditioning, possible cache/context-parallel hooks | P2 |

Initial report recommendation: SD3 and Flux should be separate early reports.
They look superficially similar as transformer denoisers but differ in pipeline
conditioning and attention/data packing.

### C. VAE and latent codecs

VAE decode is independently useful and provides a bounded stage before full
denoising-loop integration.

| Candidate | Source anchors | Model | Distinct ops | Priority |
| --- | --- | --- | --- | --- |
| AutoencoderKL | `models/autoencoders/autoencoder_kl.py`, `vae.py`, `unet_2d_blocks.py` | 2D KL VAE | encode and decode Conv2d/ResNet/GroupNorm/SiLU/up/downsample, optional attention, quant/post-quant convs, scaling/shift factors | P0 |
| AutoencoderTiny | `models/autoencoders/autoencoder_tiny.py` | tiny VAE | compact conv-only decode path | P1 |
| Video VAEs | `autoencoder_kl_cogvideox.py`, `autoencoder_kl_wan.py`, `autoencoder_kl_ltx*.py`, `autoencoder_kl_mochi.py` | temporal/3D VAEs | 3D convs, temporal compression, tiled decode, frame chunking | P2 |
| Audio VAEs/codecs | `autoencoder_oobleck.py`, `autoencoder_longcat_audio_dit.py`, `autoencoder_kl_ltx2_audio.py` | 1D/audio codecs | Conv1d, residual audio blocks, up/downsample | P3 |

Initial report recommendation: write a standalone AutoencoderKL decode report
because many pipelines share it and it is a clean operator island.

### D. Scheduler families and denoising-loop arithmetic

Schedulers are not just helpers; their state and arithmetic define the compiled
runtime loop contract.

| Candidate | Source anchors | Key behavior | Priority |
| --- | --- | --- | --- |
| DDIM/DDPM/PNDM/Euler | `scheduling_ddim.py`, `scheduling_ddpm.py`, `scheduling_pndm.py`, `scheduling_euler_discrete.py` | epsilon/sample/v-prediction conversion, alpha/sigma tables, simple first-order or multistep loops | P0 for SD baseline |
| FlowMatchEulerDiscrete | `scheduling_flow_match_euler_discrete.py` | sigma schedules, dynamic shifting, flow update, stochastic sampling branch, per-token timesteps branch | P0 for SD3/Flux |
| DPMSolver/UniPC/DEIS/SASolver | `scheduling_dpmsolver_multistep.py`, `scheduling_unipc_multistep.py`, etc. | multistep solver state, order warmup, model output conversion | P1 |
| Video/audio-specific schedulers | `scheduling_ltx_euler_ancestral_rf.py`, CogVideoX schedulers, block refinement | model-specific tables and state | P2/P3 |

Initial report recommendation: scheduler reports should be attached to pipeline
reports, but a short standalone scheduler matrix will probably pay off before
compiling loops.

### E. Control, adapter, and variant review candidates

These should follow after the base denoiser path because they add side branches,
conditioning tensors, weight/embedding mutation, or variant-specific
preprocessing contracts. They are separate review candidates, not items to hide
in a skip/defer list.

| Candidate | Source anchors | Distinct runtime surface | Priority |
| --- | --- | --- | --- |
| SD1 LoRA/textual inversion/runtime adapters | `loaders/lora_pipeline.py`, `loaders/textual_inversion.py`, `loaders/peft.py`, SD pipeline mixins | tokenizer/text-encoder embedding mutation, UNet/text-encoder adapter weights, fuse/unfuse/load/unload artifact state | P1 |
| SD1 img2img / inpaint / depth2img / upscale variants | `pipeline_stable_diffusion_img2img.py`, `pipeline_stable_diffusion_inpaint.py`, `pipeline_stable_diffusion_depth2img.py`, `pipeline_stable_diffusion_upscale.py`, `pipeline_stable_diffusion_latent_upscale.py` | VAE encode, strength/timestep slicing, mask/masked-image latents, depth conditioning, low-resolution image or latent conditioning | P1 |
| ControlNet for Stable Diffusion | `pipelines/controlnet`, `models/controlnets/controlnet.py`, `pipelines/controlnet/multicontrolnet.py` | conditioning image encoder, extra down/mid residuals into UNet, multi-control aggregation | P1 after SD baseline |
| SD3/Flux/Sana/QwenImage ControlNets | `controlnet_sd3.py`, `controlnet_flux.py`, `controlnet_sana.py`, `controlnet_qwenimage.py` | transformer-side residual/control streams | P2 |
| IP-Adapter | `loaders/ip_adapter.py`, `models/attention_processor.py`, `models/embeddings.py`, family processors | added K/V image branches, image projection layers, masks, scale lists | P2 |
| T2I-Adapter / SparseCtrl / ControlNetXS | `pipelines/t2i_adapter`, `models/adapter.py`, `controlnet_sparsectrl.py`, `controlnet_xs.py` | feature pyramids and lightweight residual/control injection | P2/P3 |
| GLIGEN | `pipelines/deprecated/stable_diffusion_gligen`, `models/embeddings.py` | grounded generation inputs, boxes/phrases/image grounding projections, gated attention branches | P3/deprecated |

### F. Video generation pipelines

Video models are high-value but should come after image DiT/UNet basics because
they add frame packing, temporal attention, 3D tensors, and temporal VAEs.

| Candidate | Source anchors | Denoiser/codecs | Distinct ops | Priority |
| --- | --- | --- | --- | --- |
| Wan | `pipelines/wan`, `models/transformers/transformer_wan*.py`, `autoencoder_kl_wan.py` | 3D transformer + Wan VAE | video latent packing, two-transformer variants, temporal RoPE/attention, frame decode | P1/P2 |
| CogVideoX | `pipelines/cogvideo`, `cogvideox_transformer_3d.py`, `autoencoder_kl_cogvideox.py` | 3D transformer + video VAE | temporal patching, CogVideoX schedulers | P2 |
| Hunyuan Video / LTX / Mochi / Allegro / SkyReels | family pipeline dirs and transformer/autoencoder files | 3D transformer variants | frame/latent compression, temporal attention, custom schedulers | P2/P3 |
| AnimateDiff / Stable Video Diffusion | `pipelines/animatediff`, `stable_video_diffusion`, UNet motion models | UNet + temporal modules | motion adapters, spatio-temporal conditioning | P2 |

### G. Audio pipelines

Audio appears as a distinct branch with 1D convolutions, spectrogram/audio
feature processors, and audio-specific transformers/codecs.

| Candidate | Source anchors | Distinct ops | Priority |
| --- | --- | --- | --- |
| Stable Audio | `pipelines/stable_audio`, `models/transformers/stable_audio_transformer.py` | 1D/audio transformer, timestep embeddings, audio latents | P2/P3 |
| AudioLDM / MusicLDM | `pipelines/audioldm2`, `pipelines/musicldm` | text/audio conditioning, spectrogram/VAE/vocoder-like pieces | P3 |
| ACE-Step / LongCat Audio DiT | `pipelines/ace_step`, `models/transformers/ace_step_transformer.py`, `transformer_longcat_audio_dit.py` | audio DiT blocks and audio codecs | P3 |

### H. Quantization, offload, and loaders

These are integration surfaces rather than first operator targets.

| Candidate | Source anchors | Notes | Priority |
| --- | --- | --- | --- |
| LoRA/PEFT loaders | `loaders/`, per-family loader mixins | Separate review candidate for weight transforms, runtime adapter state, and fused projections | P2 |
| TorchAO/quanto/nvidia modelopt/gguf dummy utilities | `quantizers/`, `utils/dummy_*` | Relevant to future encoded constants and runtime-load policies, but not first parity | P3 |
| `model_cpu_offload_seq` pipeline metadata | pipeline classes | Useful model for Dinoml explicit residency planning; do not copy Accelerate behavior blindly | P1 design input |
| Attention backend registry | `models/attention_dispatch.py` | Good source for backend constraints and varlen/flash/xformers/flex gaps | P1 provider input |

## Cross-cutting required op families

Likely first-order operator coverage from source scan:

- Dense tensor/layout ops: reshape, view/unflatten, transpose, concat/split,
  chunk, repeat/expand, gather, crop, pad, interpolate, NCHW and patch-token
  layout transforms, plus guarded NCHW/NCDHW -> NHWC/NDHWC layout translation.
- Convolutional ops: Conv2d, ConvTranspose/up/downsample, some Conv1d/Conv3d
  for audio/video, GroupNorm + SiLU-heavy ResNet blocks.
- GEMM/linear ops: bias and bias-free Linear, packed/fused QKV projections,
  projection heads for timestep/text/guidance embeddings.
- Attention: self-attention, cross-attention, joint text-image attention,
  added-KV/IP-Adapter branches, optional QK norm, RoPE, SDPA/flash/xformers
  backend variants.
- Norm/adaptive conditioning: GroupNorm, LayerNorm, RMSNorm, AdaGroupNorm,
  AdaLayerNormZero/Single/Continuous, scale-shift/gated residual paths.
- Embeddings/custom math: sinusoidal timestep embeddings, Gaussian Fourier
  projection, size/crop conditioning, 2D/3D sin-cos and rotary embeddings,
  guidance-scale embeddings.
- Scheduler/guidance arithmetic: CFG concatenation/chunking, true CFG separate
  denoiser calls, guidance rescale, flow/epsilon/v/sample prediction
  conversions, sigma/alpha table lookup, multistep state.
- VAE/postprocess: latent scaling/shift, decode convs, tiling/slicing branches,
  image normalization. Safety/NSFW filtering is out of scope for the current
  audit unless explicitly selected.

## Initial report queue

Status legend: `done` means a calibration report exists, `in progress` means a
distributed full-audit worker owns the report path, and `queued` means not yet
assigned.

1. `stable_diffusion_1_x` (`done`): baseline latent image family with UNet2DCondition,
   CLIP tokenizer/text encoder classes, CFG, broad compatible scheduler set,
   AutoencoderKL encode/decode accounting, and extension inventory.
2. `autoencoder_kl` (`done`): standalone VAE encode/decode operator island shared by SD,
   SDXL, SD3, Flux, and many image pipelines.
3. `stable_diffusion_2_x` (`done; originals removed, use sd2-community configs`): separate SD 2.x report for text encoder and
   checkpoint variation inside the same pipeline folder.
4. `stable_diffusion_xl` (`done`): dual text encoder and richer conditioning while still
   using the familiar UNet denoiser.
5. `stable_diffusion_3` (`done`): first high-priority joint-attention DiT plus
   FlowMatch scheduler and skip-layer guidance.
6. `flux` (`done`): DiT pipeline folder with multiple variants such as base, fill,
   control, controlnet, Kontext, prior redux, and local family helpers; choose
   the main model/pipeline first and inventory variants separately.
7. `sd1_lora_textual_inversion_adapters` (`done`): loader/runtime mutation surfaces.
8. `sd1_img2img_inpaint_depth_upscale` (`done; SD2 depth uses sd2-community special-case config`): variant pipeline contracts around VAE
   encode, masks, depth, and low-resolution conditioning.
9. `controlnet_sd` (`done`): add side-conditioning residuals after base SD is understood.
10. `sd1_ip_adapter` (`done`): added K/V image-conditioning attention branch.
11. `sd1_t2i_adapter` (`done`): adapter feature pyramid residuals.
12. `sd1_gligen` (`done; text-image branch source-audited, not config-swept`): grounded-generation branch, deprecated but distinct.
13. `wan` (`done; Wan2.2 I2V has null image encoder/processor slots`): first video target once image denoiser and VAE reports
   are in decent shape.
14. `cogvideox` (`done; local cache had model_index only, raw configs accessible`): second video target for temporal patching,
   CogVideoX transformer, scheduler, and CogVideoX VAE contracts.
15. `video_autoencoders` (`done; local cache had model_index only, raw VAE configs accessible`): standalone temporal codec comparison
   across Wan, CogVideoX, LTX, Mochi, and related video VAEs.
16. `autoencoder_tiny` (`done`): compact TAESD-style VAE family and
   differences from AutoencoderKL.
17. `sdxl_controlnet_adapters` (`done; local placeholders, raw config/header data accessible`): SDXL ControlNet, T2I-Adapter,
   and IP-Adapter variant surfaces.
18. `pixart` (`done; Sigma 512 scheduler config 404`): PixArt Alpha/Sigma DiT bridge with T5-style
   conditioning, patch tokens, and scheduler/VAE contracts.
19. `sana` (`done; official configs saved under H:/configs`): efficient image transformer family and attention
   variants.
20. `scheduler_matrix` (`done`): standalone scheduler/runtime-loop
   staging matrix across SD, SDXL, SD3, Flux, Wan, and CogVideoX.
21. `flux_variants_control` (`done; BFL configs require authenticated reads, succeeded`): Flux img2img/fill/control/
   controlnet/IP-Adapter/Kontext/prior-redux variant contracts.
22. `qwen_image` (`done; base processor 404 is expected absent component`): QwenImage pipeline/transformer family,
   language-model conditioning, scheduler, VAE, and control variant inventory.
23. `hunyuan_image` (`done; official configs saved under H:/configs`): Hunyuan image DiT pipeline family,
   conditioning, patching, scheduler, and video separation notes.
24. `ltx_video` (`done`): LTX Video pipeline, transformer, temporal VAE,
   scheduler, and layout contracts.
25. `stable_audio` (`done; official repo gated, mirror configs used; authenticated retry recommended`): first audio diffusion target with audio
   latent/codec and transformer contracts.
26. `auraflow_lumina` (`done; Lumina Next needed HF-cache configs after raw LFS pointers/404`): remaining image DiT families covering
   AuraFlow and Lumina pipeline/model contracts.
27. `hunyuan_video` (`done; local cache model_index only, raw configs accessible`): Hunyuan video pipeline, transformer, codec,
   scheduler, and image/text conditioning contracts.
28. `mochi_video` (`done`): Mochi video pipeline, transformer, Mochi VAE,
   scheduler, and temporal layout contracts.
29. `audioldm_musicldm` (`done; expected absent-component 404s only`): AudioLDM/AudioLDM2/MusicLDM audio
   latent, codec/vocoder, UNet, and scheduler contracts.
30. `z_image` (`done; ControlNet single-file config paths 404, base configs accessible`): Z-Image pipeline/transformer family from
   non-deprecated `pipelines/z_image`.
31. `stable_cascade` (`done`): Stable Cascade prior/decoder stack and
   multi-stage latent contracts.
32. `deepfloyd_if` (`done; missing paths are expected absent stage components`): DeepFloyd IF pixel/super-resolution
   cascaded pipeline contracts.
33. `kandinsky_family` (`done`): Kandinsky 1/2.2/3/5 prior-decoder and
   image-conditioning families.
34. `ace_step` (`done; current 1.5 has no Diffusers model_index/scheduler config`):
   ACE-Step audio DiT and Oobleck/FlowMatch contracts.
35. `allegro` (`done; TI2V references missing transformer class in checkout`):
   Allegro video pipeline, transformer, VAE, and scheduler contracts.
36. `animatediff_stable_video_diffusion` (`done; Lightning root config 404`):
   AnimateDiff motion adapters and SVD spatio-temporal UNet contracts.
37. `bria_bria_fibo` (`done; Bria mirror used, Fibo-Edit official configs 403`):
   Bria/Bria FIBO image transformer family.
38. `chroma` (`done; official component configs saved under H:/configs`):
   Chroma Flux-like transformer family.
39. `chronoedit` (`done; official configs no-access and source/config scheduler mismatch`):
   ChronoEdit video edit pipeline and Wan-adjacent contracts.
40. `cogview3_4` (`done; CogView4-Control configs gated/unavailable`):
   CogView3Plus and CogView4 image transformer families.
41. `consisid` (`done; local cache model_index only, raw configs accessible`):
   ConsisID identity-conditioned video pipeline.
42. `consistency_ddim_ddpm_latent_diffusion` (`done`): classic/generic
   diffusion and latent-diffusion pipeline contracts.
43. `controlnet_sd3` (`done; sd3.5-large ControlNet config 404`): SD3
   ControlNet and inpaint-control variants.
44. `controlnet_hunyuandit` (`done; raw ControlNet configs accessible`):
   HunyuanDiT ControlNet variants.
45. `cosmos` (`done; Transfer2.5 gated even with authenticated hf`): Cosmos
   video/world-model pipelines and codecs.
46. `dit` (`done`): class-conditional DiT pipeline baseline.
47. `easyanimate` (`done`): EasyAnimate video transformer and MagVIT VAE family.
48. `ernie_glm_image` (`done`): ERNIE Image and GLM Image transformer families.
49. `flux2` (`done; official BFL 9B/KV configs gated, mirrors used for shape evidence`):
   Flux2/Klein image transformer family.
50. `helios` (`done; internal tiny raw configs 401 but production configs public`):
   Helios image/video pipeline and pyramid model family.
51. `hunyuandit` (`done`): base HunyuanDiT image pipeline family,
   separate from Hunyuan Image 2.1 and HunyuanDiT ControlNet variants.
52. `kolors` (`done; inpaint/ControlNet/IP repos are separate candidate surfaces`):
   Kolors SDXL-adjacent image pipeline family.
53. `latent_consistency_models` (`done; some repos are LoRA-only or UNet-only`):
   LCM pipeline and scheduler distillation variants.
54. `hidream_image` (`done; external gated Meta Llama 3.1 required, E1 edit class absent`):
   HiDream image transformer family.
55. `latte` (`done; older Latte cache not a current full pipeline contract`):
   Latte video DiT family.
56. `ledits_pp` (`done; SDXL IP-Adapter branch source-marked incomplete`):
   LEDITS++ edit pipeline family.
57. `llada2` (`done; external Transformers wiring, no model_index contract`):
   LLaDA2 image generation pipeline family.
58. `longcat_audio_dit` (`done; scheduler config absent, constructor FlowMatch fallback required`):
   LongCat audio DiT pipeline and codec.
59. `longcat_image` (`done; edit latent shortcut has source validation note`):
   LongCat image generation family.
60. `ltx2` (`done; official 2.3 Diffusers configs absent, mirror facts labeled`):
   LTX2 video/audio pipeline family.
61. `lucy` (`done`): Lucy Wan-adjacent video edit pipeline family.
62. `marigold` (`done; HR pipeline advertised by config but class absent`):
   Marigold depth/normal/intrinsics estimation
   diffusion family.
63. `nucleusmoe_image` (`done; single official full config set found`):
   NucleusMoe image transformer family.
64. `omnigen` (`done; OmniGen2 custom classes absent from checkout`):
   OmniGen image generation/editing family.
65. `ovis_image` (`done; tiny internal configs 401, official repo public`):
   Ovis image generation family.
66. `pag` (`done; source/runtime wrapper over existing family configs`):
   Perturbed Attention Guidance pipeline variants.
67. `prx` (`done; component configs public, safetensors indexes partly absent`):
   PRX pipeline family.
68. `sana_video` (`done; LongLive config references absent causal transformer class`):
   Sana video transformer family.
69. `shap_e` (`done; renderer alias/bin-vs-fp16-safetensors caveat`):
   Shap-E 3D generation pipeline family.
70. `skyreels_v2` (`done`): SkyReels v2 video pipeline family.
71. `visualcloze` (`done; official 384/512 configs saved under H:/configs`):
   VisualCloze pipeline family.

## Exhaustive non-deprecated backlog

The queue above started from high-value integration targets. The audit scope is
now expanded to all non-deprecated Diffusers pipeline folders. Remaining
non-deprecated targets not yet reported: none.

## Prompt review notes to remember

- Always start from the pipeline `__call__`, not the model class.
- Always record scheduler config and loop arithmetic; it affects parity as much
  as denoiser ops.
- Always distinguish one checkpoint's default scheduler from the pipeline
  family's supported scheduler set.
- Treat text encoders as external components at first, but record their exact
  output tensors, tokenizer classes, and duplication rules.
- For folder-level families with multiple pipelines or local variants, such as
  `flux`, inspect directory contents before choosing the "main" report target.
- For DiT families, inspect `embeddings.py`, `normalization.py`, and
  `attention_dispatch.py`; many important operations are not local to the
  transformer file.
- For UNet families, inspect block factory files and shared ResNet/up/downsample
  implementations; the top-level UNet constructor only names block types.
- For all vision/video families, report source layout and candidate optimized
  layout around every conv, norm, pooling, patchify/unpatchify, VAE, and
  processor boundary. NHWC/NDHWC should be treated as a guarded optimization,
  not an unqualified semantic translation.
- For video/audio, report tensor layout and temporal/audio compression before
  listing ops.
