# Diffusers Stable Diffusion XL Operator and Integration Report

Target slug: `stable_diffusion_xl`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  stabilityai/stable-diffusion-xl-base-1.0
  stabilityai/stable-diffusion-xl-refiner-1.0
  stabilityai/sdxl-turbo
  echarlaix/tiny-random-stable-diffusion-xl

Config sources:
  H:/configs/stabilityai/stable-diffusion-xl-base-1.0/
  H:/configs/stabilityai/stable-diffusion-xl-refiner-1.0/
  H:/configs/stabilityai/sdxl-turbo/
  H:/configs/echarlaix/tiny-random-stable-diffusion-xl/
  Component JSON files were fetched with `hf download` after local cache only
  contained `model_index.json` for the primary StabilityAI repos.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_instruct_pix2pix.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/resnet.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  X:/H/diffusers/src/diffusers/loaders/textual_inversion.py
  X:/H/diffusers/src/diffusers/loaders/ip_adapter.py
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet.py
  X:/H/diffusers/src/diffusers/models/adapter.py

External component configs inspected:
  CLIPTextModel config for text_encoder.
  CLIPTextModelWithProjection config for text_encoder_2.

Any missing files or assumptions:
  This audit covers SDXL base/refiner text-to-image latent denoising, with
  img2img, inpaint, instruct-pix2pix, ControlNet, T2I-Adapter, IP-Adapter,
  LoRA/textual inversion/runtime adapter mutation, and upscalers inventoried as
  separate candidates. Safety/watermarking, callbacks/interrupt, training,
  multi-GPU/context parallel, XLA/NPU/MPS/Flax/ONNX, and dropout/loss paths are
  out of scope. No official config path remained blocked.
```

## 2. Pipeline and component graph

SDXL keeps the classic latent-diffusion UNet loop but changes conditioning:
two CLIP text encoders feed concatenated token embeddings, the second CLIP
encoder supplies pooled embeddings, and the UNet receives added size/crop time
IDs through `addition_embed_type="text_time"`.

```text
prompt / prompt_2 or cached embeddings
  -> CLIPTokenizer + CLIPTextModel hidden states [B,77,768]
  -> CLIPTokenizer + CLIPTextModelWithProjection hidden states [B,77,1280]
     and pooled projection [B,1280]
  -> concat token embeddings on feature axis -> [B,77,2048]
  -> create added time IDs: original size, crop top-left, target size
  -> initialize latent noise [B,4,H/8,W/8]
  -> denoising loop:
       CFG batch concat, scheduler.scale_model_input
       -> UNet2DConditionModel(latents, t, prompt_embeds,
                               added_cond_kwargs={text_embeds,time_ids})
       -> CFG and optional guidance_rescale
       -> scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

Base required components: `CLIPTokenizer`, `CLIPTextModel`,
`CLIPTextModelWithProjection`, `UNet2DConditionModel`, `EulerDiscreteScheduler`
or compatible scheduler, and `AutoencoderKL`. First Dinoml slice can accept
`prompt_embeds`, `negative_prompt_embeds`, `pooled_prompt_embeds`, and
`negative_pooled_prompt_embeds` externally.

Refiner differences: the official refiner model index uses
`StableDiffusionXLImg2ImgPipeline`, only `tokenizer_2`/`text_encoder_2`, a wider
UNet, `requires_aesthetics_score=true`, and `force_zeros_for_empty_prompt=false`.
It is normally chained after base with `denoising_start`/`denoising_end`.

Separate candidate reports:

| Candidate | Classes/files | Pipeline delta |
| --- | --- | --- |
| `sdxl_lora_textual_inversion_adapters` | `StableDiffusionXLLoraLoaderMixin`, `TextualInversionLoaderMixin`, PEFT loader paths | Mutates tokenizer/text-encoder embeddings and UNet/text-encoder attention or linear weights. |
| `sdxl_ip_adapter` | `IPAdapterMixin`, `IPAdapterAttnProcessor*`, image encoder/projection classes | Adds CLIP image encoder inputs, image projection, optional masks, and added K/V attention branches. |
| `sdxl_controlnet` | `StableDiffusionXLControlNetPipeline` in `pipelines/controlnet`, `ControlNetModel`, `MultiControlNetModel` | Adds conditioning image preprocessing and down/mid residuals into the SDXL UNet. |
| `sdxl_t2i_adapter` | `StableDiffusionXLAdapterPipeline`, `T2IAdapter`, `MultiAdapter`, `FullAdapterXL` | Adds adapter feature tensors through UNet intrablock residuals. |
| `sdxl_img2img_refiner` | `StableDiffusionXLImg2ImgPipeline` | Adds VAE encode, strength/timestep slicing, and refiner aesthetic-score conditioning. |
| `sdxl_inpaint` | `StableDiffusionXLInpaintPipeline` | Adds mask preprocessing, masked image latents, and 4-channel or 9-channel UNet contracts. |
| `sdxl_instruct_pix2pix` | `StableDiffusionXLInstructPix2PixPipeline` | Adds image conditioning and altered guidance/control arithmetic. |
| `sdxl_upscale` | `stabilityai/sd-x2-latent-upscaler`, SD upscale pipeline family | Upscaling is not in the SDXL folder's base class; review separately for low-res latent/image conditioning. |
| `sdxl_gligen` | No direct SDXL GLIGEN pipeline found in this folder | Existing GLIGEN support is SD1 deprecated; treat any XL grounded variant as separate if required. |

## 3. Important config dimensions

| Config | UNet sample | UNet channels | blocks | layers | cross dim | head dim | transformer layers/block | added text/time input | scheduler |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| SDXL base 1.0 | 128 | 4 -> 4 | 320/640/1280 | 2 | 2048 | 5/10/20 | 1/2/10 | 2816 | EulerDiscrete |
| SDXL refiner 1.0 | 128 | 4 -> 4 | 384/768/1536/1536 | 2 | 1280 | 6/12/24/24 | 4 | 2560 | EulerDiscrete |
| SDXL Turbo | 64 | 4 -> 4 | 320/640/1280 | 2 | 2048 | 5/10/20 | 1/2/10 | 2816 | EulerAncestral |
| tiny random SDXL | 32 | 4 -> 4 | 32/64 | 2 | 64 | 2/4 | 1/2 | 80 | EulerDiscrete |

| Text encoder | Class | hidden | projection | layers | heads | max tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base/turbo encoder 1 | `CLIPTextModel` | 768 | 768 | 12 | 12 | 77 |
| base/turbo encoder 2 | `CLIPTextModelWithProjection` | 1280 | 1280 | 32 | 20 | 77 |
| refiner encoder 2 | `CLIPTextModelWithProjection` | 1280 | 1280 | 32 | 20 | 77 |
| tiny encoders | CLIP variants | 32 | 32 | 5 | 4 | 77 |

| VAE config | sample | latent | blocks | scale | force upcast | notes |
| --- | ---: | ---: | --- | ---: | --- | --- |
| SDXL base/refiner/turbo | 1024 | 4 | 128/256/512/512 | 0.13025 | true | omitted quant/post-quant flags use `AutoencoderKL` defaults: enabled |
| tiny random SDXL | 128 | 4 | 32/64 | 0.18215 | omitted, effective true | debug shape, not production-equivalent |

Scheduler support: the type annotation is `KarrasDiffusionSchedulers`, and the
pipeline should be treated as scheduler-swappable across Euler/DDIM/LMS/PNDM/DPM
style schedulers with compatible APIs. Official base/refiner defaults are
EulerDiscrete with `prediction_type=epsilon`, `timestep_spacing=leading`,
`steps_offset=1`, `use_karras_sigmas=false`; Turbo uses Euler Ancestral with
`timestep_spacing=trailing`. Recommended first Dinoml scheduler slice: Euler
Discrete for base/refiner parity, then Euler Ancestral for Turbo.

## 3a. Family variation traps

- Base prompt embeddings are `[B,77,2048]`; refiner uses only the bigG CLIP path
  and has `[B,77,1280]` cross-attention.
- `text_time` addition is required. Six scalar IDs are Fourier-projected with
  `addition_time_embed_dim=256` and concatenated with pooled CLIP embeddings.
- Base/turbo `projection_class_embeddings_input_dim=2816` is `1280 + 6*256`;
  refiner `2560` is `1280 + 5*256` when aesthetic score replaces target size.
- Refiner and base are commonly chained with denoising range controls; do not
  treat refiner as the same text-to-image entry contract.
- Source latent maps are NCHW. NHWC is only a guarded conv-island optimization;
  GroupNorm, channel concat, VAE stats, mask/inpaint concat, and scheduler
  broadcasting need axis rewrites or guards.
- Optional branches in `UNet2DConditionModel` include class embeddings,
  image/image_hint additions, ControlNet residuals, T2I-Adapter residuals,
  GLIGEN, IP-Adapter, and guidance-scale embeddings. Vanilla base/refiner only
  require `addition_embed_type="text_time"`.
- `time_cond_proj_dim` is absent for inspected SDXL base/refiner/turbo configs,
  so true CFG is batch concat/chunk, not embedded guidance.
- `force_zeros_for_empty_prompt=true` on base/turbo makes missing negative
  prompts zero tensors; refiner config sets it false.

## 4. Runtime tensor contract

For 1024x1024 base, one image per prompt:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| CLIP hidden 1 | token hidden | `[B,S,C]` | `[B,77,768]` |
| CLIP hidden 2 | token hidden | `[B,S,C]` | `[B,77,1280]` |
| prompt embeds | concatenated hidden | `[B,S,C]` | `[B,77,2048]`; `[2B,77,2048]` with CFG |
| pooled embeds | bigG pooled projection | `[B,C]` | `[B,1280]`; `[2B,1280]` with CFG |
| add time IDs | original/crop/target | `[B,6]` base | repeated to CFG/image batch |
| latent state | noisy latents | NCHW | `[B,4,128,128]` |
| UNet input | latent_model_input | NCHW | `[2B,4,128,128]` with CFG |
| UNet output | noise prediction | NCHW | `[2B,4,128,128]` before chunk |
| scheduler output | updated latents | NCHW | `[B,4,128,128]` |
| VAE decode input | `latents / 0.13025` | NCHW | `[B,4,128,128]` |
| decoded image | VAE sample | NCHW | `[B,3,1024,1024]` |

Refiner uses the same 4-channel latent map but commonly starts from base latents
or an encoded image, slices timesteps by strength/denoising range, and feeds
`[B,77,1280]` prompt embeddings plus aesthetic-score time IDs. VAE encode is
required for img2img/refiner/inpaint candidates, while base text-to-image only
requires decode.

Precomputable: prompt embeddings, pooled embeddings, negative embeddings, added
time IDs for fixed sizes/crops, scheduler timesteps/sigmas, and initial latents
when provided by caller. GPU/runtime work: latent noising, UNet forward, CFG,
scheduler step, VAE decode/encode.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCHW latent/image tensors with guarded NHWC islands.
- Prompt hidden concat on feature axis and CFG concat/chunk on batch axis.
- Added time IDs flatten, sinusoidal/Fourier projection, reshape, concat with
  pooled text embeddings.
- Spatial skip connection tuple management and UNet skip concat.
- Per-channel VAE scale and optional `latents_mean/std` handling.
- Inpaint candidate: channel concat `[latents, mask, masked_image_latents]`.

### Convolution/downsample/upsample ops

- Base UNet `Conv2d(4 -> 320, 3x3, padding=1)`, refiner
  `Conv2d(4 -> 384, 3x3, padding=1)`.
- ResnetBlock2D pairs of 3x3 convs, optional 1x1 shortcuts.
- Downsample2D stride/down convs and Upsample2D interpolate/nearest + conv.
- VAE encoder/decoder Conv2d/ResNet/down/up blocks and 1x1 quant/post-quant
  convs from effective `AutoencoderKL` defaults.

### GEMM/linear ops

- Timestep MLP and added text/time embedding MLP.
- ResNet time-bias projection.
- Cross-attention Q from latent tokens, K/V from CLIP token embeddings.
- Feed-forward GEGLU/GELU blocks in `BasicTransformerBlock`.
- Optional CLIP text encoders as external first, later Transformers coverage.

### Attention primitives

- Spatial latent self-attention and cross-attention in SDXL UNet transformer
  blocks.
- Base cross-attention dim 2048; refiner cross-attention dim 1280.
- Eager and PyTorch SDPA attention processors define parity; xFormers/fused
  QKV are optional source-supported mutations.

### Normalization and adaptive conditioning

- GroupNorm over channel axis in UNet/VAE.
- LayerNorm over token dimension in transformer blocks.
- SiLU, GELU/GEGLU.
- Time embedding add into ResNet path; no SD3/Flux-style AdaLayerNorm in the
  SDXL UNet first slice.

### Scheduler and guidance arithmetic

- EulerDiscrete `scale_model_input`, sigma/timestep tables, epsilon prediction
  update.
- Euler Ancestral for Turbo.
- CFG: `uncond + guidance_scale * (text - uncond)`.
- Optional guidance rescale reduction over all non-batch dims.

## 6. Denoiser/model breakdown

Base SDXL UNet active path:

```text
sample [B,4,H,W]
-> Timesteps + TimestepEmbedding
-> add_embedding(concat(pooled_text, time_id_projection))
-> Conv2d(4 -> 320)
-> DownBlock2D(320)
-> CrossAttnDownBlock2D(640), transformer_layers_per_block=2
-> CrossAttnDownBlock2D(1280), transformer_layers_per_block=10
-> UNetMidBlock2DCrossAttn(1280)
-> CrossAttnUpBlock2D(1280)
-> CrossAttnUpBlock2D(640)
-> UpBlock2D(320)
-> GroupNorm -> SiLU -> Conv2d(320 -> 4)
```

Refiner widens to 384/768/1536/1536, uses cross-attention only at the middle
resolutions, and has four transformer layers per attention block. Both use
`resnet_time_scale_shift="default"`, so ResNet time conditioning is additive
rather than scale-shift.

`BasicTransformerBlock` pattern is the SD UNet pattern:

```text
flatten spatial map -> [B,H*W,C]
LayerNorm -> self attention -> residual
LayerNorm -> cross attention to prompt embeds -> residual
LayerNorm -> GEGLU/GELU feed-forward -> residual
reshape back to NCHW map
```

## 7. Attention requirements

- Noncausal, no KV cache.
- Query sequence is latent spatial tokens at each UNet resolution.
- Base text K/V length is 77 and width 2048; refiner width is 1280.
- Per-block channel widths are 320/640/1280 for base and 384/768/1536/1536 for
  refiner; head dims from configs are 5/10/20 and 6/12/24/24 respectively.
- `attention_processor.py` is the primary implementation path for this target.
- Default modern path is `AttnProcessor2_0` using
  `torch.nn.functional.scaled_dot_product_attention`; eager `AttnProcessor`
  defines fallback parity.
- `UNet2DConditionModel.fuse_qkv_projections()` and fused processors are
  source-supported, but first parity should keep unfused projections.
- Flash-style Dinoml lowering is plausible for mask-free self/cross attention
  with dropout 0 and supported head dims/dtypes. It must be guarded off for
  added K/V IP-Adapter branches, unsupported masks, custom processors, or LoRA
  mutation not folded into weights.

## 8. Scheduler and denoising-loop contract

Base/refiner default scheduler:

```text
EulerDiscreteScheduler
num_train_timesteps=1000
beta_start=0.00085, beta_end=0.012, beta_schedule=scaled_linear
prediction_type=epsilon
timestep_spacing=leading
steps_offset=1
use_karras_sigmas=false
```

Loop-side graph:

```text
latent_model_input = cat([latents, latents]) if CFG else latents
latent_model_input = scheduler.scale_model_input(latent_model_input, t)
noise_pred = unet(latent_model_input, t, prompt_embeds, added_cond_kwargs)
noise_pred = uncond + scale * (text - uncond)
if guidance_rescale > 0: noise_pred = rescale_noise_cfg(...)
latents = scheduler.step(noise_pred, t, latents)
```

Keep iteration, timestep slicing, denoising_start/end, and scheduler state in
host-visible control first. Compile one UNet step plus CFG and Euler update as
separate kernels. Turbo changes the sampled default to
`EulerAncestralDiscreteScheduler` with trailing timestep spacing; that is a
separate first-family scheduler slice after EulerDiscrete parity.

## 9. Position, timestep, and custom math

SDXL requires standard sinusoidal timestep embeddings and added size/crop time
conditioning. Base time IDs:

```text
[original_height, original_width,
 crop_top, crop_left,
 target_height, target_width]
```

Each scalar is projected by `add_time_proj`; the result is reshaped per batch
and concatenated with pooled CLIP embeddings before `add_embedding`. For base,
`1280 + 6*256 = 2816`. Refiner aesthetic mode replaces part of the size tuple
with aesthetic scores so the input is `1280 + 5*256 = 2560`.

Guidance rescale computes per-sample standard deviations across all non-batch
axes, so any layout pass must express this as layout-polymorphic reduction, not
hard-code NCHW axis positions.

## 10. Preprocessing and input packing

CPU/data-pipeline:

- Tokenization for two CLIP tokenizers; `prompt_2` defaults to `prompt`.
- Textual inversion may expand multi-vector tokens before tokenization.
- CLIP hidden-state selection uses penultimate hidden state by default or
  `-(clip_skip + 2)` when `clip_skip` is set.
- Negative prompts either tokenize normally or become zeros when
  `force_zeros_for_empty_prompt` applies.
- Image postprocess and optional watermarking are outside Dinoml first scope.

GPU/runtime:

- Accept cached prompt embeddings and pooled embeddings.
- Duplicate embeddings for `num_images_per_prompt`.
- Concatenate negative/positive embeddings for CFG.
- Prepare NCHW latent noise scaled by `scheduler.init_noise_sigma`.
- VAE decode final latents; VAE encode belongs to img2img/refiner/inpaint.

## 11. Graph rewrite / lowering opportunities

### Rewrite: SDXL NCHW ResNet/conv island to guarded NHWC

Preconditions: region contains Conv2d, GroupNorm, SiLU, residual add, down/up
sample, and pointwise ops with all channel axes rewritten. Attention
flatten/reshape and external scheduler boundaries must be either outside the
island or layout-aware.

Replacement: `NCHW boundary -> NHWC conv island -> NCHW boundary`.
Weight transform: OIHW Conv2d weights to HWIO. Failure cases: inpaint channel
concat, VAE stats, GroupNorm dim, or attention reshape not rewritten.
Parity sketch: base/refiner ResnetBlock2D and Down/Up blocks at 128, 64, 32,
and 16 latent resolutions.

### Rewrite: SDXL text_time addition

Preconditions: pooled text width and count of time IDs match
`projection_class_embeddings_input_dim`. Replacement:
`time_ids -> sinusoidal projection -> concat pooled -> MLP`. Failure cases:
refiner `requires_aesthetics_score` mismatch or missing pooled embeddings.

### Rewrite: attention canonicalization

Preconditions: no IP-Adapter/additional K/V, no custom masks, LoRA either
folded or represented. Replacement: spatial flatten -> Q/K/V GEMMs -> attention
-> output GEMM -> reshape. Failure cases: added K/V processors, xFormers-only
processor state, unsupported mask backend.

### Rewrite: Euler scheduler step as fused elementwise

Preconditions: scheduler sigma index and prediction type are explicit and fixed
to epsilon. Replacement: scale input and step arithmetic fused over latent map.
Failure cases: ancestral noise branch, v-prediction/sample prediction, custom
sigmas, or multistep solver state.

## 12. Kernel fusion candidates

Highest priority:

- Conv2d + GroupNorm + SiLU in UNet and VAE ResNet blocks.
- Cross-attention Q/K/V projection + attention + output projection.
- `text_time` addition MLP and timestep embedding projection.
- CFG arithmetic, guidance rescale, and Euler scheduler update.
- VAE decode conv island for 1024 output.

Medium priority:

- Guarded NHWC islands across SDXL UNet resolution stages.
- GEGLU/GELU feed-forward fusion in transformer blocks.
- Fused QKV/KV projections after unfused parity.
- Refiner/base chaining with latent handoff and denoising range slicing.

Lower priority:

- Euler Ancestral Turbo branch after EulerDiscrete.
- IP-Adapter added K/V attention.
- ControlNet/T2I residual injection and inpaint channel concat.
- VAE tiling/slicing memory policies.

## 13. Runtime staging plan

Stage 1: Parse SDXL base/refiner/turbo component configs and reconcile omitted
AutoencoderKL defaults.

Stage 2: Load UNet and VAE weights; accept external prompt and pooled
embeddings.

Stage 3: Implement `text_time` conditioning and one ResnetBlock2D plus one
BasicTransformerBlock parity.

Stage 4: Compile one SDXL base UNet forward at fixed latent shape with supplied
embeddings.

Stage 5: Add EulerDiscrete `scale_model_input`, CFG, and one-step scheduler
parity.

Stage 6: Add full Python-controlled denoising loop and VAE decode.

Stage 7: Add refiner UNet and img2img latent handoff/denoising range support.

Stage 8: Add Turbo Euler Ancestral/trailing timestep parity.

Stage 9: Add guarded NHWC conv islands and attention provider lowering.

Stage 10: Separate reports for inpaint, ControlNet, T2I-Adapter, IP-Adapter,
LoRA/textual inversion/adapters, instruct-pix2pix, and upscaling.

## 14. Parity and validation plan

- Config parse parity for base/refiner/turbo/tiny JSON fields.
- `text_time` embedding parity for base six-ID and refiner aesthetic-ID forms.
- Prompt embedding concat/duplication/CFG parity with cached CLIP outputs.
- ResnetBlock2D parity at 320/640/1280 and 384/768/1536 channels.
- BasicTransformerBlock parity for cross dims 2048 and 1280.
- Full UNet forward parity on tiny SDXL, then base/refiner shapes.
- EulerDiscrete `set_timesteps`, `scale_model_input`, and `step` parity.
- Euler Ancestral parity for Turbo.
- CFG and guidance rescale parity.
- VAE decode parity for `[B,4,128,128] -> [B,3,1024,1024]`.
- Refiner handoff parity: base latent output to refiner img2img start.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 initially
  `rtol=2e-2, atol=2e-2`, then tighten per kernel.

## 15. Performance probes

- Text encoder throughput separately from denoiser when CLIP is in scope.
- One SDXL base UNet step by batch, resolution, dtype, and CFG mode.
- Refiner UNet step separately because channels/depth differ.
- Conv/resnet time vs attention time at 1024 output resolution.
- Attention backend comparison: eager, SDPA, and guarded Dinoml flash-style.
- Scheduler/guidance overhead per step.
- VAE decode throughput and fp32-upcast cost.
- Base+refiner chained latency by denoising split.
- NCHW faithful path vs NHWC conv-island path.
- VRAM/workspace usage for base, refiner, and Turbo.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `sdxl_lora_textual_inversion_adapters`: SDXL LoRA loader, textual inversion,
  PEFT adapter state and fuse/unfuse paths.
- `sdxl_ip_adapter`: image encoder/projection and added K/V attention.
- `sdxl_controlnet`: ControlNet down/mid residuals and multi-control
  aggregation.
- `sdxl_t2i_adapter`: `FullAdapterXL` feature pyramids and intrablock residuals.
- `sdxl_img2img_refiner`: VAE encode, strength slicing, aesthetic conditioning,
  and base/refiner latent handoff.
- `sdxl_inpaint`: masks, masked-image latents, and 9-channel UNet variants.
- `sdxl_instruct_pix2pix`: image-conditioning and altered guidance inputs.
- `sdxl_upscale`: SDXL-related or SD x2/x4 upscalers with low-resolution
  conditioning.
- `sdxl_turbo`: can share much of base UNet coverage, but scheduler/timestep
  and low-step parity deserve a narrow follow-up.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker, invisible watermarking, and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse SDXL base/refiner/turbo model indexes and component configs.
- [ ] Load base UNet, refiner UNet, and SDXL VAE weights.
- [ ] Accept external prompt, negative prompt, pooled, and negative pooled embeddings.
- [ ] Implement prompt concat/duplication and CFG batch construction.
- [ ] Implement added `text_time` IDs and embedding path.
- [ ] Implement Conv2d/GroupNorm/SiLU/ResnetBlock2D parity for base/refiner widths.
- [ ] Implement SDXL BasicTransformerBlock self/cross-attention parity.
- [ ] Implement EulerDiscrete scheduler first slice.
- [ ] Implement CFG and guidance rescale kernels.
- [ ] Add one-step base denoising parity.
- [ ] Add full base denoising loop with scheduler in host control.
- [ ] Add AutoencoderKL decode with SDXL scaling factor 0.13025.
- [ ] Add refiner img2img handoff and aesthetic-score conditioning.
- [ ] Add Euler Ancestral/Turbo parity.
- [ ] Add guarded NHWC conv-island rewrite and tests.
- [ ] Add attention provider/flash-style guarded lowering.
- [ ] Create separate candidate reports for IP-Adapter, ControlNet, T2I-Adapter,
      LoRA/textual inversion/adapters, inpaint, instruct-pix2pix, and upscaling.
