# Diffusers Stable Diffusion 1.x Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.
  Remote upstream: https://github.com/huggingface/diffusers.git

Model id(s):
  Family target: Stable Diffusion 1.x text-to-image.
  Primary worked example: stable-diffusion-v1-5/stable-diffusion-v1-5.
  Additional sizing references: CompVis/stable-diffusion-v1-4,
  hf-internal-testing/tiny-stable-diffusion-pipe.

Config sources:
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/model_index.json
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/unet/config.json
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/vae/config.json
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/scheduler/scheduler_config.json
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/text_encoder/config.json
  Plus the v1.4 and tiny component configs listed above.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  X:/H/diffusers/src/diffusers/image_processor.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/resnet.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_pndm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  Class inventory scan for related candidates:
    X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
    X:/H/diffusers/src/diffusers/loaders/textual_inversion.py
    X:/H/diffusers/src/diffusers/loaders/ip_adapter.py
    X:/H/diffusers/src/diffusers/models/adapter.py
    X:/H/diffusers/src/diffusers/models/controlnets/controlnet.py
    X:/H/diffusers/src/diffusers/pipelines/controlnet/
    X:/H/diffusers/src/diffusers/pipelines/t2i_adapter/
    X:/H/diffusers/src/diffusers/pipelines/deprecated/stable_diffusion_gligen/

External component configs inspected:
  CLIPTextModel config from the SD 1.5 repo.

Any missing files or assumptions:
  Text encoder and tokenizer internals live in Transformers and are treated as
  an external prompt-embedding stage for the first Dinoml slice. IP-Adapter,
  LoRA, textual inversion, runtime adapters, ControlNet, T2I-Adapter, GLIGEN,
  and img2img/inpaint/depth/upscale variants are inventoried as separate review
  candidates. Safety checker/NSFW filtering, callback mutation, distributed
  paths, training-only paths, and XLA, NPU, MPS, Flax, and ONNX-specific
  variants are ignored unless they affect shared CPU/CUDA inference source
  structure. The report targets inference-only CUDA with scheduler control
  initially in host code.
```

## 2. Pipeline and component graph

Stable Diffusion 1.x is a latent text-to-image family. The SD 1.5 model index wires:
`CLIPTokenizer`, `CLIPTextModel`, `UNet2DConditionModel`, `AutoencoderKL`,
`PNDMScheduler`, and optional image/safety components in the artifact. For this
audit, `StableDiffusionSafetyChecker` and NSFW filtering are ignored scope. The
pipeline declares offload order `text_encoder->image_encoder->unet->vae`, with
`image_encoder` relevant to IP-Adapter candidate reports.

```text
prompt strings / prompt_embeds
  -> CLIP tokenizer + CLIPTextModel, or externally supplied prompt embeddings
  -> duplicate/concat negative and positive embeddings for CFG
  -> initialize latent noise [B,4,H/8,W/8]
  -> denoising loop:
       cat CFG latents -> optional scheduler scale_model_input
       -> UNet2DCondition(latents, timestep, encoder_hidden_states)
       -> CFG arithmetic and optional guidance_rescale
       -> scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

First required components:

- Required: prompt embeddings as an input boundary, UNet denoiser, scheduler
  tables/step state, VAE decode, image postprocess.
- Required for family accounting: AutoencoderKL decode for text-to-image and
  AutoencoderKL encode for img2img/inpaint-style family variants.
- Optional after parity: CLIP text encoder/tokenizer and a broader scheduler
  set. Related extensions/variants below are separate report candidates, not
  base first-slice ops.

Independently cacheable stages:

- Prompt embeddings and negative prompt embeddings.
- Scheduler timesteps/tables for a fixed scheduler configuration and step count.
- Initial latents when caller supplies `latents`.
- VAE decode can be validated independently from the denoising loop.

Separate candidate reports for this family:

| Candidate | Primary classes/files | Pipeline delta from base text-to-image |
| --- | --- | --- |
| `sd1_lora_textual_inversion_adapters` | `StableDiffusionPipeline` inherits `TextualInversionLoaderMixin`, `StableDiffusionLoraLoaderMixin`, `IPAdapterMixin`; loader anchors `loaders/textual_inversion.py`, `loaders/lora_pipeline.py`, `loaders/peft.py` | Mutates tokenizer/text-encoder embeddings, UNet/text-encoder adapter weights, and possibly active attention/linear modules at load or runtime; should be reviewed as weight/embedding mutation and artifact-state handling. |
| `sd1_ip_adapter` | `IPAdapterMixin`, `UNet2DConditionLoadersMixin._load_ip_adapter_weights`, `IPAdapterAttnProcessor`, `IPAdapterAttnProcessor2_0`, `IPAdapterXFormersAttnProcessor`, `IPAdapterMaskProcessor`, image projection classes such as `MultiIPAdapterImageProjection` | Adds image encoder/image embeds, image projection layers, added K/V attention branches, per-adapter scales, and optional masks. |
| `sd1_controlnet` | `StableDiffusionControlNetPipeline`, `StableDiffusionControlNetImg2ImgPipeline`, `StableDiffusionControlNetInpaintPipeline`, `ControlNetModel`, `ControlNetConditioningEmbedding`, `MultiControlNetModel` | Adds conditioning image preprocessing, ControlNet down/mid residuals, conditioning scales, guess mode, and single/multi-control aggregation before UNet residual injection. |
| `sd1_t2i_adapter` | `StableDiffusionAdapterPipeline`, `T2IAdapter`, `MultiAdapter`, `FullAdapter`, `LightAdapter` | Adds adapter feature pyramid/residual tensors into UNet down blocks, with image conditioning preprocessing distinct from ControlNet. |
| `sd1_gligen` | deprecated `StableDiffusionGLIGENPipeline`, `StableDiffusionGLIGENTextImagePipeline`, `GLIGENTextBoundingboxProjection` | Adds grounded generation inputs such as phrases/boxes/images and gated grounding attention/position embeddings. Deprecated, but still a distinct candidate if compatibility matters. |
| `sd1_img2img` | `StableDiffusionImg2ImgPipeline` | Uses `image` plus `strength`; VAE-encodes image to latents, selects a shortened timestep range, adds noise, then runs the standard denoising loop. |
| `sd1_inpaint` | `StableDiffusionInpaintPipeline` | Adds mask/image preprocessing, mask latents, masked image latents, and usually a 9-channel UNet input contract for inpaint checkpoints. |
| `sd1_depth2img` | `StableDiffusionDepth2ImgPipeline` | Adds depth estimator/depth map conditioning and a different UNet input/conditioning contract; source is in the SD folder but depends on additional Transformers depth components. |
| `sd_upscale` | `StableDiffusionUpscalePipeline`, `StableDiffusionLatentUpscalePipeline` | Super-resolution/latent-upscale flows add low-resolution image conditioning/noise-level conditioning and differ enough from SD1.x base to review separately; `StableDiffusionUpscalePipeline` is SD2-oriented. |

## 3. Important config dimensions

Primary SD 1.5 component dimensions:

| Component | Field | Value | Source |
| --- | --- | ---: | --- |
| pipeline | class | `StableDiffusionPipeline` | `model_index.json` |
| scheduler | sampled default class | `PNDMScheduler` | scheduler config |
| scheduler | beta schedule | `scaled_linear`, 1000 train steps | scheduler config |
| scheduler | prediction type | omitted/null, effective `epsilon` | config + PNDM source default |
| scheduler | compatible family | `KarrasDiffusionSchedulers` in pipeline typing; DDIM/LMS/PNDM noted in docs, conversion code also builds Euler/Euler ancestral/DPMSolver variants | source |
| text encoder | hidden size | 768 | CLIP config |
| text encoder | max positions | 77 | CLIP config |
| text encoder | vocab size | 49408 | CLIP config |
| UNet | latent sample size | 64 | UNet config |
| UNet | in/out channels | 4 / 4 | UNet config |
| UNet | block channels | 320, 640, 1280, 1280 | UNet config |
| UNet | layers per block | 2 | UNet config |
| UNet | cross attention dim | 768 | UNet config |
| UNet | attention head dim | 8 | UNet config |
| UNet | norm groups | 32 | UNet config |
| VAE | sample size | 512 | VAE config |
| VAE | latent channels | 4 | VAE config |
| VAE | block channels | 128, 256, 512, 512 | VAE config |
| VAE | scaling factor | omitted/null, effective `0.18215` | config + AutoencoderKL source default |

Representative checkpoint sweep:

| Checkpoint | UNet sample | UNet channels | UNet blocks | cross dim | VAE sample | VAE blocks | scheduler | Notes |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| stable-diffusion-v1-5/stable-diffusion-v1-5 | 64 | 4 -> 4 | 320/640/1280/1280 | 768 | 512 | 128/256/512/512 | PNDM | Primary target |
| CompVis/stable-diffusion-v1-4 | 64 | 4 -> 4 | 320/640/1280/1280 | 768 | 512 | 128/256/512/512 | not fetched here | Same core operator shape |
| hf-internal-testing/tiny-stable-diffusion-pipe | 32 | 4 -> 4 | 32/64 | 32 | 32 | 32/64 | FlaxDDIM config in repo | Useful debug shape, not architecture-equivalent |

Scheduler support note: SD 1.x should not be modeled as "PNDM only." PNDM is
the SD 1.5 sampled default, but the pipeline accepts compatible scheduler
objects and the conversion utilities construct DDIM, PNDM, LMS, Euler, Euler
ancestral, and DPMSolver-style schedulers from related configs. First Dinoml
parity can choose one or two scheduler slices, but the family report should
preserve the broader supported-set expectation.

## 3a. Family variation traps

- SD 1.x source tensors are NCHW for latent maps, but Dinoml should prefer NHWC
  only as a guarded layout/fusion optimization. GroupNorm axes, concat/chunk,
  skip connections, scheduler broadcasting, and VAE scale/divide are
  axis-sensitive.
- Many optional branches exist in `UNet2DConditionModel` but are inactive for
  vanilla SD 1.5: class embeddings, addition embeddings, GLIGEN, IP-Adapter,
  ControlNet residuals, T2I-Adapter residuals, and time guidance embeddings.
- The SD 1.5 VAE config omits `scaling_factor`; current AutoencoderKL source
  supplies `0.18215`. Do not treat a missing field as no scaling.
- The SD 1.5 scheduler config has `_class_name=PNDMScheduler`, but the pipeline
  accepts broad compatible scheduler swaps. DDIM, Euler, LMS, PNDM, and DPM
  solver variants need distinct step math and state contracts.
- The pipeline constructor mutates stale scheduler configs for `steps_offset`
  and `clip_sample`; those compatibility repairs should be represented if
  loading old artifacts.
- The text encoder is external Transformers code. First Dinoml parity can use
  supplied `prompt_embeds` to avoid compiling CLIP immediately.

## 4. Runtime tensor contract

First slice boundary:

| Boundary | Tensor | Source layout | Candidate optimized layout | Shape for 512x512, CFG on |
| --- | --- | --- | --- | --- |
| prompt embeddings | `prompt_embeds` | `[B,77,768]` | same | `[2B,77,768]` after CFG concat |
| latent state | `latents` | NCHW | NHWC guarded candidate | `[B,4,64,64]` |
| UNet input | `latent_model_input` | NCHW | NHWC guarded candidate | `[2B,4,64,64]` |
| UNet output | `noise_pred` | NCHW | NHWC guarded candidate | `[2B,4,64,64]` before CFG chunk |
| scheduler output | `latents` | NCHW | must match loop layout | `[B,4,64,64]` |
| VAE input | `latents / scaling_factor` | NCHW | NHWC guarded candidate | `[B,4,64,64]` |
| VAE output | decoded image | NCHW | NHWC guarded candidate | `[B,3,512,512]` |
| VAE encode input | image for img2img/inpaint variants | NCHW after processor | NHWC guarded candidate | `[B,3,512,512]` |
| VAE encode output | latent distribution/sample | NCHW | NHWC guarded candidate | `[B,4,64,64]` before scaling |
| postprocess | image | NCHW torch -> PIL/np | CPU output layout | denormalized image |

Scheduler state:

- PNDM owns `timesteps`, `prk_timesteps`, `plms_timesteps`, `counter`, and
  `ets` model-output history.
- DDIM owns alpha cumulative product tables and optional eta variance.
- Euler owns sigma tables on CPU by default, a step index, and upcasts step math
  for precision.
- DPM/LMS/Euler ancestral variants are compatible scheduler-family candidates
  but need separate reports or scheduler matrix entries.

Precompute/reuse:

- Prompt embeddings can be reused across denoising runs with the same prompt.
- Timesteps and scheduler tables can be cached per scheduler config and step
  count.
- Time embeddings depend on the current timestep but not spatial size.
- UNet/VAE weights are static constants.

## 5. Operator coverage checklist

### Tensor/layout ops

- Shape validation for image size divisible by VAE scale factor 8.
- NCHW source tensor handling, with guarded NHWC layout translation candidates.
- `torch.cat([negative, positive])` for prompt embeddings and CFG latent batch.
- `chunk(2)` on batch dimension for CFG noise prediction.
- Skip connection tuple push/pop in UNet down/up path.
- Spatial concat for up-block skip joins inside shared blocks.
- Broadcast scalar/timestep scheduler coefficients over `[B,C,H,W]`.

### Convolution/downsample/upsample ops

- UNet `conv_in`: `Conv2d(4 -> 320, 3x3, padding=1)`.
- UNet ResnetBlock2D: two `3x3` convolutions, optional `1x1` shortcut.
- Downsample2D conv stride-2 path in down blocks.
- Upsample2D nearest/interpolate + optional conv path in up blocks.
- VAE encoder/decoder convs, ResNet blocks, up/downsample, `1x1` quant and
  post-quant convs.

### GEMM/linear ops

- Time embedding MLP from sinusoidal timestep embedding to time embedding.
- ResNet time projection into channel bias or scale/shift path.
- Cross-attention Q from latent tokens and K/V from text tokens.
- Attention output projection.
- Feed-forward GEGLU/GELU MLP inside BasicTransformerBlock.

### Attention primitives

- Noncausal latent self-attention in transformer blocks when configured.
- Cross-attention from latent spatial tokens to CLIP text tokens.
- Source `Attention` supports `AttnProcessor2_0` SDPA when available and eager
  `AttnProcessor` otherwise.
- Optional xFormers/fused projections exist but are not required for first
  parity.

### Normalization and adaptive conditioning

- GroupNorm with 32 groups over channel axis in UNet and VAE.
- LayerNorm over token hidden dimension inside BasicTransformerBlock.
- Time embedding add or scale-shift in ResnetBlock2D, depending config.
- SiLU activations.

### Position/timestep/guidance embeddings

- `Timesteps` sinusoidal embedding with `flip_sin_to_cos=True`,
  `freq_shift=0` in SD 1.5.
- Optional guidance-scale embedding only if `unet.config.time_cond_proj_dim` is
  set; inactive for vanilla SD 1.5.

### Scheduler and guidance arithmetic

- PNDM PRK/PLMS step functions and model-output history for sampled SD 1.5
  default.
- DDIM, Euler, LMS, Euler ancestral, and DPM solver compatible scheduler
  variants as family support, each with separate state/math contracts.
- CFG: `uncond + guidance_scale * (text - uncond)`.
- Optional guidance rescale: per-sample standard deviations over all non-batch
  dims.
- Latent initialization with `randn_tensor(...)*scheduler.init_noise_sigma`.

### VAE/postprocessing ops

- Divide latents by `vae.config.scaling_factor` before decode.
- Encode image tensors through encoder and optional quant conv for img2img and
  inpaint-style variants.
- Decode through post-quant conv and decoder.
- Denormalize decoded image to `[0,1]` in `VaeImageProcessor`.
- Safety checker and NSFW filtering are ignored for this audit.

## 6. Denoiser/model breakdown

Vanilla SD 1.5 UNet forward:

```text
sample [B,4,H,W]
if center_input_sample: sample = 2*sample - 1        # inactive in SD 1.5
t_emb = Timesteps(timestep)
emb = TimestepEmbedding(t_emb)
sample = Conv2d(4 -> 320, 3x3, pad=1)

down blocks:
  CrossAttnDownBlock2D x3:
    repeated ResnetBlock2D
    BasicTransformerBlock with latent tokens attending to text tokens
    optional downsample between resolutions
  DownBlock2D x1:
    repeated ResnetBlock2D

mid block:
  UNetMidBlock2DCrossAttn:
    ResnetBlock2D -> cross-attention transformer -> ResnetBlock2D

up blocks:
  UpBlock2D / CrossAttnUpBlock2D:
    concatenate matching skip features
    repeated ResnetBlock2D
    optional cross-attention transformer
    optional upsample

post:
  GroupNorm -> SiLU -> Conv2d(320 -> 4, 3x3, pad=1)
```

ResnetBlock2D active pattern:

```text
h = GroupNorm(x)
h = SiLU(h)
h = optional up/downsample(h)
h = Conv2d(h)
temb = SiLU(emb) -> Linear(...)
if default: h = h + temb[:, :, None, None]
if scale_shift: h = GroupNorm(h) * (1 + scale) + shift
else: h = GroupNorm(h)
h = SiLU(h)
h = dropout(h)              # inactive at inference if p=0
h = Conv2d(h)
skip = optional shortcut(x)
out = (skip + h) / output_scale_factor
```

BasicTransformerBlock active pattern:

```text
tokens = flatten spatial latent map to [B, H*W, C]
tokens = tokens + self_attention(LayerNorm(tokens))       # when enabled
tokens = tokens + cross_attention(LayerNorm(tokens), text)
tokens = tokens + FeedForward(LayerNorm(tokens))          # GEGLU by default
reshape tokens back to spatial map
```

## 7. Attention requirements

- Noncausal attention; no KV cache.
- Cross-attention dominates the text-conditioning path.
- Query sequence length is spatial: `H*W` per UNet resolution.
- Text K/V sequence length is 77 for CLIP SD 1.5.
- Heads are derived from `attention_head_dim=8` and per-block channel width.
  For channels 320/640/1280, likely head counts are 40/80/160 if head dim stays
  8; confirm from block construction in a full parity run.
- Eager processor: Q/K/V Linear, reshape heads, attention scores/softmax/value,
  output Linear.
- SDPA processor: uses `torch.nn.functional.scaled_dot_product_attention`.
- Fused QKV/KV projections are source-supported through `fuse_qkv_projections`,
  but should be an optimization after unfused parity.
- xFormers path is optional and has warning surface around masks/custom
  processors.
- Flash-style lowering should be checked rather than assumed. The SD 1.x
  source path gives parity through eager/SDPA attention processors; a Dinoml
  flash-style provider may still be valid for mask-free or supported-mask
  self/cross attention if shapes, dtype, dropout=0, and processor semantics are
  proven. Conversely, added K/V, custom processors, or unsupported masks can
  rule it out for a specific branch.

## 8. Scheduler and denoising-loop contract

For SD 1.5 sampled default PNDM:

- `set_timesteps(num_inference_steps, device)` builds PRK and PLMS timesteps,
  resets `ets` and `counter`.
- `step()` dispatches to `step_prk` while `counter` is inside PRK warmup unless
  `skip_prk_steps`; otherwise `step_plms`.
- `step_plms` stores recent model outputs and applies Adams-Bashforth-style
  combinations for 1, 2, 3, or 4 history entries.
- `_get_prev_sample` converts model output according to `prediction_type`;
  SD 1.5 effectively uses epsilon prediction.

Loop-side graph work:

```text
latent_model_input = cat([latents, latents]) if CFG else latents
latent_model_input = scheduler.scale_model_input(latent_model_input, t) if present
noise_pred = unet(latent_model_input, t, prompt_embeds)
noise_pred = uncond + scale * (text - uncond)
if guidance_rescale: noise_pred = rescale_noise_cfg(...)
latents = scheduler.step(noise_pred, t, latents)
```

Initial Dinoml staging should keep timestep iteration and scheduler state in
host code, compiling one denoiser step and small scheduler arithmetic kernels
separately. A later loop compiler can make DDIM/Euler/PNDM/DPM state explicit in
a runtime plan. For a first scheduler implementation, DDIM or Euler may be
simpler than exact PNDM, but that is a staging choice rather than an SD 1.x
family limitation.

## 9. Position, timestep, and custom math

Sinusoidal timestep embedding is required. It can be generated per timestep or
precomputed for the scheduler timestep table.

Guidance rescale:

```python
std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
```

Layout caution: this std reduction is "all dims except batch". A layout pass must
not hard-code channel/spatial axes differently for NCHW vs NHWC.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenization, CLIP text encoding, negative prompt construction.
- Textual inversion and LoRA scaling are separate candidate-report surfaces.
- Safety checker and NSFW filtering are ignored for this audit.

GPU/runtime work:

- Accept `prompt_embeds[B,77,768]` and optional negative prompt embeddings.
- Duplicate embeddings for `num_images_per_prompt`.
- CFG concatenates negative and positive prompt embeddings on batch dimension.
- Generate or accept latent noise `[B,4,H/8,W/8]`.
- For img2img/inpaint/depth-style variants, preprocess and VAE-encode images
  into latents, then combine noised latents and scheduler strength/timestep
  selection before denoising.
- Decode final latents through VAE and postprocess to output images.

Image processor:

- `VaeImageProcessor` normalizes inputs to `[-1,1]` for encode-style paths and
  denormalizes decoded outputs to `[0,1]`.
- It resizes image inputs to multiples of `vae_scale_factor=8` when needed.

## 11. Graph rewrite / lowering opportunities

### Rewrite: source NCHW conv island -> guarded NHWC conv island

Preconditions:

- Region includes only Conv2d, GroupNorm, SiLU, residual add, nearest upsample,
  downsample, and pointwise ops whose axes can be rewritten.
- All consumers inside region accept translated layout.
- GroupNorm channel axis is rewritten from dim 1 to last channel.
- Scheduler and CFG arithmetic either run layout-polymorphic pointwise code or
  are outside the translated island with explicit boundary transposes.

Replacement:

```text
NCHW boundary -> NHWC internal conv/norm/resnet region -> NCHW boundary
```

Weight transform:

```python
# OIHW source Conv2d weight to HWIO for NHWC conv implementation
w_hwio = w_oihw.permute(2, 3, 1, 0)
```

Failure cases:

- Attention flatten/reshape assumes source spatial/channel ordering and is not
  included in the region.
- Upstream PyTorch comments note `upsample_nearest_nhwc` problems for some large
  shapes; Dinoml must validate its own kernel behavior instead of assuming
  PyTorch channel-last success.

Parity sketch:

- Compare one ResnetBlock2D and one down/up block in source NCHW vs NHWC-lowered
  implementation at 64, 32, 16, and 8 latent resolutions.

### Rewrite: cross-attention projections -> GEMM + attention

Preconditions:

- Hidden states and encoder states are dense row-major token matrices after
  flattening spatial tokens.
- Q projection input is latent tokens, K/V projection input is text tokens.
- Bias settings match source `Attention` config.

Replacement:

```text
FlattenSpatial -> GEMM_RCR(Q)
TextEmbeds -> GEMM_RCR(K), GEMM_RCR(V)
Attention(Q,K,V) -> GEMM_RCR(out) -> ReshapeSpatial
```

Failure cases:

- Custom attention processors, added K/V branches, IP-Adapter, LoRA-in-flight,
  or masks not represented in the lowered contract.

Parity sketch:

- Compare one BasicTransformerBlock with fixed random text embeddings and latent
  tokens before/after attention.

### Rewrite: CFG batched UNet -> explicit two-call or batched mode

Preconditions:

- Positive and negative prompt embeddings are known and CFG is active.
- UNet is pure with respect to batch entries.

Replacement options:

```text
batched: cat latents/prompts -> one UNet -> chunk -> CFG
explicit: UNet(latents, neg) and UNet(latents, pos) -> CFG
```

Failure cases:

- Memory planning may prefer two calls for large batches.
- Callback mutation and IP-Adapter masks can alter tensor sets.

Parity sketch:

- Compare batched CFG against two separate UNet calls for one timestep.

### Rewrite: scheduler scalar table lookup -> broadcast fused elementwise

Preconditions:

- Scheduler step selected and state (`counter`, history, timestep index) is
  explicit.
- Prediction type is fixed to epsilon for SD 1.5 first slice.

Replacement:

```text
load scalar alpha/sigma values -> fused elementwise over latent tensor
```

Failure cases:

- Selected scheduler state/history not represented.
- v-prediction/sample prediction variants need different equations.

## 12. Kernel fusion candidates

Highest priority:

- GroupNorm + SiLU + Conv2d in UNet/VAE ResNet blocks. This dominates the
  convolutional denoiser and VAE decode path.
- Cross-attention Q/K/V projection + SDPA/Flash-style attention + output
  projection for latent/text attention.
- CFG arithmetic and guidance rescale as small fused elementwise/reduction
  kernels.
- Scheduler step arithmetic for DDIM/Euler first, then PNDM stateful PLMS and
  DPM/LMS variants.
- AutoencoderKL encode/decode conv/resnet/down/up paths, with a separate
  autoencoder report recommended for codec-specific optimization.

Medium priority:

- NHWC conv islands with boundary elision across consecutive ResNet/down/up
  blocks.
- Fused QKV/KV projection using source-supported projection fusion as a guide.
- Time embedding MLP and ResNet time-bias application fusion.
- Spatial flatten/attention/reshape layout fusion.

Lower priority:

- PNDM PRK warmup if first product target uses DDIM/Euler instead.

Separate candidate reports, not ranked as base-kernel work:

- IP-Adapter added K/V attention and mask/scaling paths.
- ControlNet and T2I-Adapter residual/control injection.
- LoRA/textual inversion/runtime adapter mutation.
- Img2img/inpaint/depth/upscale variant preprocessing and denoising contracts.

## 13. Runtime staging plan

Stage 1: Parse SD 1.5 component configs and load UNet/VAE weights. Treat CLIP
prompt embeddings as external inputs.

Stage 2: Add standalone operator parity for GroupNorm, SiLU, Conv2d, nearest
upsample, Downsample2D, GEGLU, timestep embedding, and Attention processor.

Stage 3: Compile and validate one ResnetBlock2D and one BasicTransformerBlock.

Stage 4: Compile one UNet down/mid/up block slice at fixed latent shape.

Stage 5: Run one denoising step parity: supplied latents, timestep,
prompt embeddings, UNet output, CFG, scheduler step. Choose the first scheduler
slice explicitly, without treating that choice as the whole family contract.

Stage 6: Full denoising loop with scheduler in Python and compiled UNet step.

Stage 7: Add AutoencoderKL decode parity and end-to-end latent-to-image output;
add AutoencoderKL encode parity for img2img/inpaint-family readiness or track it
in the separate autoencoder report.

Stage 8: Add NHWC guarded conv islands and attention/provider fusions.

Stage 9: Add CLIP text encoder or prompt-embedding cache integration.

## 14. Parity and validation plan

- Random tensor parity for GroupNorm axis behavior in NCHW and candidate NHWC.
- ResnetBlock2D parity for channel sizes 320, 640, 1280.
- BasicTransformerBlock parity with `cross_attention_dim=768`.
- UNet single block parity at 64x64, 32x32, 16x16, and 8x8 latent resolutions.
- One full UNet forward parity for tiny SD first, then SD 1.5 shapes.
- CFG arithmetic parity, including `guidance_rescale=0` and nonzero.
- PNDM `set_timesteps` and `step` parity for the SD 1.5 sampled default.
- DDIM/Euler scheduler parity if they are selected for first loop simplicity,
  plus a follow-up scheduler matrix for LMS/DPM variants.
- AutoencoderKL decode parity from latent tensors.
- AutoencoderKL encode parity from preprocessed image tensors.
- End-to-end deterministic smoke using fixed prompt embeddings and fixed noise.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 initial
  `rtol=2e-2, atol=2e-2`, tightened per kernel.

## 15. Performance probes

- One UNet forward by batch, resolution, dtype, and scheduler step.
- UNet conv/resnet time vs attention time.
- Attention backend comparison: eager, SDPA, future Dinoml provider.
- NCHW faithful path vs guarded NHWC conv islands.
- CFG batched one-call vs explicit two-call memory and latency.
- Scheduler/guidance overhead as a percentage of one step.
- VAE decode throughput by batch and output resolution.
- Full denoising loop throughput by step count.
- VRAM and temporary/workspace usage across UNet + VAE.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `sd1_lora_textual_inversion_adapters`: `StableDiffusionLoraLoaderMixin`,
  `TextualInversionLoaderMixin`, `PeftAdapterMixin`, UNet/text-encoder adapter
  mutation.
- `sd1_ip_adapter`: `IPAdapterMixin`, `IPAdapterAttnProcessor*`, image
  projection classes, IP-Adapter masks/scales.
- `sd1_controlnet`: `StableDiffusionControlNetPipeline`,
  `StableDiffusionControlNetImg2ImgPipeline`,
  `StableDiffusionControlNetInpaintPipeline`, `ControlNetModel`,
  `MultiControlNetModel`.
- `sd1_t2i_adapter`: `StableDiffusionAdapterPipeline`, `T2IAdapter`,
  `MultiAdapter`, `FullAdapter`, `LightAdapter`.
- `sd1_gligen`: deprecated `StableDiffusionGLIGENPipeline`,
  `StableDiffusionGLIGENTextImagePipeline`, `GLIGENTextBoundingboxProjection`.
- `sd1_img2img`: `StableDiffusionImg2ImgPipeline`, VAE encode and strength-based
  timestep slicing.
- `sd1_inpaint`: `StableDiffusionInpaintPipeline`, mask/masked-image latents
  and inpaint UNet input contract.
- `sd1_depth2img`: `StableDiffusionDepth2ImgPipeline`, depth preprocessing and
  depth-conditioned UNet contract.
- `sd_upscale`: `StableDiffusionUpscalePipeline`,
  `StableDiffusionLatentUpscalePipeline`, low-resolution image or latent
  conditioning.

Ignored/out of scope for this audit unless explicitly selected:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX pipeline variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse SD 1.5 `model_index.json` and component configs.
- [ ] Reconcile omitted config fields against Diffusers source defaults.
- [ ] Load UNet and VAE weights.
- [ ] Accept external `prompt_embeds` and `negative_prompt_embeds`.
- [ ] Implement Conv2d/GroupNorm/SiLU/ResnetBlock2D parity.
- [ ] Implement Downsample2D and Upsample2D parity.
- [ ] Implement timestep embedding and ResNet time conditioning.
- [ ] Implement BasicTransformerBlock cross-attention parity.
- [ ] Implement CFG concat/chunk/arithmetic.
- [ ] Inventory supported scheduler families and choose first scheduler slice.
- [ ] Implement first scheduler step contract, likely DDIM/Euler for simplicity or PNDM for exact SD 1.5 default.
- [ ] Compile one UNet block slice.
- [ ] Add one-step denoising parity.
- [ ] Implement AutoencoderKL decode.
- [ ] Implement or separately track AutoencoderKL encode.
- [ ] Create separate candidate reports for LoRA/textual inversion/adapters,
      IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img,
      and upscaling.
- [ ] Add guarded NHWC conv-island rewrite and parity tests.
- [ ] Benchmark UNet step, scheduler/guidance, and VAE decode separately.
