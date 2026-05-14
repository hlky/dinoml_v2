# Diffusers ControlNet for Stable Diffusion Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.
  Remote upstream available as https://github.com/huggingface/diffusers.git.

Model id(s):
  Primary family target: SD 1.x ControlNet.
  Representative ControlNet configs:
    lllyasviel/control_v11p_sd15_canny
    lllyasviel/sd-controlnet-canny
    lllyasviel/control_v11p_sd15_openpose
    lllyasviel/control_v11p_sd15_inpaint
    hf-internal-testing/tiny-controlnet
  Base SD configs referenced:
    stable-diffusion-v1-5/stable-diffusion-v1-5
    stable-diffusion-v1-5/stable-diffusion-inpainting

Config sources:
  Local cache:
    H:/configs/lllyasviel/control_v11p_sd15_canny/model_index.json
    H:/configs/lllyasviel/sd-controlnet-canny/model_index.json
    H:/configs/lllyasviel/control_v11p_sd15_inpaint/model_index.json
    H:/configs/lllyasviel/control_v11p_sd15_openpose/model_index.json
    H:/configs/krea/aesthetic-controlnet/model_index.json
    H:/configs/MVRL/VectorSynth/model_index.json
    H:/configs/ras-diff/ras-diff-corridor/model_index.json
    H:/configs/vllab/controlnet-hands/model_index.json
    H:/configs/stable-diffusion-v1-5/stable-diffusion-v1-5/model_index.json
  Network-inspected raw JSON, not saved because this audit owns only this report path:
    https://huggingface.co/lllyasviel/control_v11p_sd15_canny/raw/main/config.json
    https://huggingface.co/lllyasviel/sd-controlnet-canny/raw/main/config.json
    https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/raw/main/config.json
    https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/raw/main/config.json
    https://huggingface.co/hf-internal-testing/tiny-controlnet/raw/main/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/unet/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/vae/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/scheduler/scheduler_config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/raw/main/unet/config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet/multicontrolnet.py
  SDXL, Union, Flax, and BLIP ControlNet files were inventoried but are not first-slice SD 1.x scope.

Model files inspected:
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet.py
  X:/H/diffusers/src/diffusers/models/controlnets/multicontrolnet.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/resnet.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_pndm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py

External component configs inspected:
  Base SD 1.5 CLIP/tokenizer/VAE/UNet behavior is inherited from the
  stable_diffusion_1_5 audit. Text encoder internals remain an external
  prompt-embedding stage for the first ControlNet slice.

Any missing files or assumptions:
  The lllyasviel local cache entries contain empty model_index.json files and
  no local component configs; public raw config fetches succeeded, so no gated
  config blocker was encountered. Safety/NSFW, training/loss/dropout/gradient
  checkpointing, callbacks/interrupt, multi-GPU/context parallel, XLA/NPU/MPS,
  Flax, and ONNX paths are out of scope. Runtime target is inference-only CUDA,
  faithful NCHW translation first, with NHWC/channel-last only as guarded
  optimization.
```

## 2. Pipeline and component graph

`StableDiffusionControlNetPipeline` is the SD 1.x text-to-image pipeline plus a
ControlNet side model. The constructor registers `vae`, `text_encoder`,
`tokenizer`, `unet`, `controlnet`, `scheduler`, optional safety components, and
optional `image_encoder` for IP-Adapter. A list or tuple of `ControlNetModel`
instances is wrapped in `MultiControlNetModel`. The offload sequence is
`text_encoder->image_encoder->unet->vae`; the ControlNet is called beside the
UNet in the denoising loop and is not an output component.

```text
prompt strings or prompt_embeds
  -> CLIP tokenizer/text encoder, or externally supplied prompt embeddings
  -> CFG negative/positive prompt batching
  -> control image preprocessing to NCHW [B,3,H,W] in [0,1]
  -> latent initialization [B,4,H/8,W/8]
  -> denoising loop:
       scheduler.scale_model_input
       -> ControlNetModel(latents, t, prompt_embeds, control_image)
       -> down/mid residual tensors, scaled and optionally guess-mode padded
       -> UNet2DConditionModel(..., down_block_additional_residuals,
          mid_block_additional_residual)
       -> CFG arithmetic
       -> scheduler.step
  -> VAE decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

Required first-slice components are external prompt embeddings, preprocessed
control image tensor, ControlNet, base SD 1.5 UNet, scheduler step state, VAE
decode, and image postprocess. Cacheable stages include prompt embeddings,
control images at the selected height/width, scheduler timestep tables, and
caller-supplied initial latents.

Separate candidate reports:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `controlnet_sd_img2img` | `StableDiffusionControlNetImg2ImgPipeline`, `pipeline_controlnet_img2img.py` | VAE-encodes init image, selects timesteps by `strength`, adds noise to image latents, then uses the same ControlNet residual path. |
| `controlnet_sd_inpaint` | `StableDiffusionControlNetInpaintPipeline`, `pipeline_controlnet_inpaint.py` | Adds mask and masked-image latent preprocessing. With 9-channel inpaint UNet, concatenates `latents, mask, masked_image_latents`; with 4-channel base UNet, blends latents after each scheduler step. |
| `controlnet_sd_multi` | `MultiControlNetModel`, `models/controlnets/multicontrolnet.py` | Runs multiple ControlNets and sums matching down/mid residual tensors. |
| `controlnet_sdxl` | SDXL ControlNet pipeline/model variants | Adds SDXL dual text encoders, pooled embeddings, time IDs, and wider configs; separate from SD 1.x. |
| `controlnet_union` | `pipeline_controlnet_union_*`, `controlnet_union.py` | Adds control modes/type embeddings and union-specific conditioning. |
| `sd1_lora_textual_inversion_ip_adapter` | inherited loader mixins and IP-Adapter attention processors | Mutates prompt/adapter weights or adds image K/V attention branches; not required for ControlNet first parity. |
| `sd1_t2i_adapter`, `sd1_gligen`, `sd1_depth2img`, `sd_upscale` | SD-family sibling pipelines | Related SD conditioning variants but not part of this ControlNet first slice. |

## 3. Important config dimensions

Representative ControlNet component configs:

| Repo/config | Class | in/control channels | block channels | blocks | layers | cross dim | attention head dim | cond embed channels | global pool |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `lllyasviel/control_v11p_sd15_canny` | `ControlNetModel` | 4 / 3 | 320,640,1280,1280 | CA,CA,CA,Down | 2 | 768 | 8 | 16,32,96,256 | omitted -> `False` |
| `lllyasviel/sd-controlnet-canny` | `ControlNetModel` | 4 / 3 | 320,640,1280,1280 | CA,CA,CA,Down | 2 | 768 | 8 | 16,32,96,256 | omitted -> `False` |
| `lllyasviel/control_v11p_sd15_openpose` | `ControlNetModel` | 4 / 3 | 320,640,1280,1280 | CA,CA,CA,Down | 2 | 768 | 8 | 16,32,96,256 | omitted -> `False` |
| `lllyasviel/control_v11p_sd15_inpaint` | `ControlNetModel` | 4 / 3 | 320,640,1280,1280 | CA,CA,CA,Down | 2 | 768 | 8 | 16,32,96,256 | omitted -> `False` |
| `hf-internal-testing/tiny-controlnet` | `ControlNetModel` | 4 / 3 | 32,64 | Down,CA | 2 | 32 | 8 | 16,32 | `False` |

Base SD config dimensions that ControlNet must match:

| Component | SD 1.5 value | Inpaint checkpoint value | Source |
| --- | --- | --- | --- |
| UNet `in_channels` | 4 | 9 | component config |
| UNet `out_channels` | 4 | 4 | component config |
| UNet `sample_size` | 64 | 64 | component config |
| UNet block channels | 320,640,1280,1280 | same | component config |
| UNet cross-attention dim | 768 | 768 | component config |
| VAE latent channels | 4 | assumed same SD VAE family | component/source |
| VAE scale factor | 8 from 4 VAE block levels | source/config |
| VAE scaling factor | omitted in old config, effective `0.18215` | AutoencoderKL source default |
| SD 1.5 sampled scheduler | PNDM, `scaled_linear`, 1000 train steps, epsilon prediction default | same family | scheduler config/source |

`StableDiffusionControlNetPipeline` accepts `KarrasDiffusionSchedulers`, so the
first Dinoml scheduler slice should be explicit. PNDM gives checkpoint-default
parity for SD 1.5; DDIM or Euler is simpler for staging but is not the whole
family contract.

## 3a. Family variation traps

- ControlNet configs mirror the SD 1.5 UNet down/mid shape, but the ControlNet
  output is residual tensors, not final noise.
- The SD 1.5 ControlNet model `in_channels` is 4 even for
  `control_v11p_sd15_inpaint`; inpaint-specific mask/image concatenation belongs
  to the inpaint pipeline and base UNet contract.
- Source tensors are NCHW. NHWC may be valuable in conv islands, but GroupNorm
  axes, BGR channel flip `dims=[1]`, concat on channel dim, and residual tuple
  ordering require layout guards.
- `attention_head_dim` is a historical naming trap. `ControlNetModel.__init__`
  maps missing `num_attention_heads` to `attention_head_dim`; for SD 1.5 configs
  this means 8 heads per attention block, not 8-wide heads.
- `guess_mode` changes the ControlNet batch path under CFG: ControlNet runs only
  the conditional batch, then zero residuals are concatenated for the
  unconditional batch.
- `control_guidance_start/end` multiply per-step `controlnet_keep` gates into
  conditioning scales. Multi-ControlNet turns these into per-ControlNet lists.
- `global_pool_conditions=True` forces `guess_mode` and replaces each residual
  with spatial mean `[B,C,1,1]`; not active in inspected SD 1.5 configs.
- IP-Adapter hooks are present in the same pipeline, but inactive unless image
  embeds are supplied.
- Inpaint has two source-relevant variants: 9-channel inpaint UNet concat and
  4-channel base UNet latent blending after scheduler step.

## 4. Runtime tensor contract

For a 512x512 SD 1.5 text-to-image ControlNet run:

| Boundary | Tensor | Source layout | Shape, CFG off | Shape, CFG on non-guess | Notes |
| --- | --- | --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,T,C]` | `[B,77,768]` | `[2B,77,768]` | negative/positive concatenated for CFG |
| control image | `image` / `controlnet_cond` | NCHW | `[B,3,512,512]` | `[2B,3,512,512]` | in `[0,1]`, not normalized to `[-1,1]` |
| latents | `latents` | NCHW | `[B,4,64,64]` | `[B,4,64,64]` | scaled by scheduler init sigma |
| denoiser input | `latent_model_input` | NCHW | `[B,4,64,64]` | `[2B,4,64,64]` | after scheduler scale |
| guess-mode ControlNet input | `control_model_input` | NCHW | `[B,4,64,64]` | `[B,4,64,64]` | conditional-only under CFG |
| ControlNet down residuals | tuple | NCHW | 12 tensors | 12 tensors | shapes below |
| ControlNet mid residual | tensor | NCHW | `[B,1280,8,8]` | `[2B,1280,8,8]` | guess mode pads zeros for uncond |
| UNet output | `noise_pred` | NCHW | `[B,4,64,64]` | `[2B,4,64,64]` | chunked for CFG |
| VAE decode input | latents / scale | NCHW | `[B,4,64,64]` | same | final output path |
| decoded image | image | NCHW | `[B,3,512,512]` | same | postprocess to PIL/np/pt |

SD 1.5 ControlNet residual shape sequence for 512x512 is:

```text
[B,  320,64,64]  conv_in/control add
[B,  320,64,64]  block0 resnet0
[B,  320,64,64]  block0 resnet1
[B,  320,32,32]  block0 downsample
[B,  640,32,32]  block1 resnet0
[B,  640,32,32]  block1 resnet1
[B,  640,16,16]  block1 downsample
[B, 1280,16,16]  block2 resnet0
[B, 1280,16,16]  block2 resnet1
[B, 1280, 8, 8]  block2 downsample
[B, 1280, 8, 8]  block3 resnet0
[B, 1280, 8, 8]  block3 resnet1
mid: [B,1280,8,8]
```

CPU/data-pipeline work: PIL/NumPy/torch input normalization, resizing to a
multiple of 8, RGB conversion, optional BGR channel flip, mask binarization for
inpaint, tokenization, and CLIP encoding. GPU/runtime work: ControlNet and UNet
for each timestep, residual additions/scales, CFG, scheduler math, and VAE
encode/decode where variants need them.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor handling for latents, control images, masks, and decoded images.
- Batch `cat`, `chunk`, `repeat_interleave`, `repeat`, and list/tuple residual
  push/pop.
- Channel `cat` for inpaint 9-channel UNet path.
- `torch.flip(controlnet_cond, dims=[1])` for BGR configs.
- Per-step scalar/list conditioning scale multiplication and residual sums.
- Broadcast zeros-like padding in guess mode.
- Optional spatial mean over `(2,3)` for global pooled conditions.

Convolution/downsample/upsample ops:

- ControlNet `conv_in`: `Conv2d(4 -> 320, 3x3, padding=1)`.
- Conditioning embedding: `Conv2d(3 -> 16, 3x3)`, then alternating
  `3x3` conv and stride-2 `3x3` conv through 16 -> 32 -> 96 -> 256, then
  zero-initialized `Conv2d(256 -> 320, 3x3)`.
- ControlNet down blocks mirror SD 1.5 UNet down blocks: ResnetBlock2D,
  cross-attention transformer blocks, and Downsample2D.
- Zero-initialized `1x1` convs on each down residual and mid residual.
- Base UNet up blocks and VAE encode/decode from the SD 1.5 report.

GEMM/linear ops:

- Timestep embedding MLP.
- Attention Q/K/V/out projections in ControlNet and base UNet.
- GEGLU/GELU feed-forward projections inside BasicTransformerBlock.
- Optional encoder hidden projection/addition embedding branches in source, but
  inactive for inspected SD 1.5 configs.

Attention primitives:

- Noncausal latent self-attention and CLIP cross-attention in ControlNet down
  and mid blocks, plus same in base UNet.
- No KV cache. No ControlNet-specific added K/V attention unless IP-Adapter is
  separately enabled.

Normalization and embeddings:

- GroupNorm over channel axis in ResNet blocks.
- LayerNorm over token hidden dim in transformer blocks.
- SiLU activations.
- Sinusoidal timestep embedding with `flip_sin_to_cos=True`, `freq_shift=0`.

Scheduler/guidance arithmetic:

- `scheduler.scale_model_input`, `scheduler.step`, `init_noise_sigma`.
- CFG: `uncond + guidance_scale * (text - uncond)`.
- Control guidance gate: scale times per-step keep mask.

VAE/postprocessing:

- Text-to-image needs decode. Img2img/inpaint need encode as separate first-class
  variant work.
- `VaeImageProcessor` postprocess denormalizes decoded images.

## 6. Denoiser/model breakdown

ControlNet forward path:

```text
controlnet_cond:
  optional channel flip for BGR

timestep:
  scalar/tensor -> batch-expanded timesteps
  -> Timesteps -> TimestepEmbedding
  -> optional class/addition embeddings

latent sample:
  Conv2d(4 -> 320, 3x3)

control image:
  ControlNetConditioningEmbedding([B,3,H,W])
  -> [B,320,H/8,W/8]

sample = conv_in(latents) + cond_embedding(control_image)

down path:
  same block sequence as SD 1.5 UNet down path
  each returned residual is passed through a zero 1x1 conv

mid path:
  UNetMidBlock2DCrossAttn
  -> zero 1x1 conv

scale:
  normal mode: every residual *= conditioning_scale
  guess mode: residual_i *= logspace(0.1..1.0)_i * conditioning_scale
  global pool: spatial mean to [B,C,1,1]
```

The base UNet receives the residual tuple after its own down path:

```text
for each saved down_block_res_sample:
  down_block_res_sample += controlnet_down_block_residual
mid sample += controlnet_mid_block_residual
continue through UNet up blocks and conv_out
```

`MultiControlNetModel` loops over control images, scales, and ControlNet modules,
then adds corresponding residual tensors elementwise before the base UNet sees
them.

## 7. Attention requirements

The required SD 1.5 ControlNet attention is the same noncausal `Attention` /
`BasicTransformerBlock` surface as SD 1.5 UNet, duplicated in the ControlNet
down/mid path. With inspected configs, `attention_head_dim=8` is used as
`num_attention_heads` through the legacy source fallback, so per-block head
counts are 8 and head dims are 40, 80, and 160 for channels 320, 640, and 1280.
Text K/V sequence length is normally 77 and cross-attention dim is 768.

Primary parity path is `attention_processor.py`: eager `AttnProcessor` or
PyTorch 2 `AttnProcessor2_0` with `torch.nn.functional.scaled_dot_product_attention`.
`attention_dispatch.py` contains broader backend dispatch for flash, native
flash, xFormers, flex, sage, and varlen backends, but these are not required for
SD 1.5 ControlNet parity. A Dinoml flash-style provider is valid only under
guards: no unsupported mask form, dropout 0, noncausal dense Q/K/V, supported
dtype/head dim, and no active IP-Adapter/added-KV processor. Eager/SDPA defines
the semantic fallback.

Fused QKV projections are source-supported by `Attention` but should be a later
weight/layout optimization. The first implementation should keep Q, K, V, and
out projections explicit.

## 8. Scheduler and denoising-loop contract

ControlNet does not replace scheduler math. It adds a per-step side model before
the base UNet call:

```text
latent_model_input = cat([latents, latents]) if CFG else latents
latent_model_input = scheduler.scale_model_input(latent_model_input, t)

if guess_mode and CFG:
  control_model_input = scheduler.scale_model_input(latents, t)
  controlnet_prompt_embeds = positive prompt embeds only
else:
  control_model_input = latent_model_input
  controlnet_prompt_embeds = prompt_embeds

cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
down_res, mid_res = controlnet(..., conditioning_scale=cond_scale)
if guess_mode and CFG:
  down_res = cat([zeros_like(res), res]) for each residual
  mid_res = cat([zeros_like(mid), mid])

noise_pred = unet(..., down_block_additional_residuals=down_res,
                  mid_block_additional_residual=mid_res)
noise_pred = CFG(noise_pred) if guidance_scale > 1
latents = scheduler.step(noise_pred, t, latents)
```

`controlnet_keep` is host-computed from `control_guidance_start/end` as a
per-step scalar or list. Keep scheduler iteration and ControlNet scale/gate
selection in host control flow for the first slice; compile ControlNet, UNet,
CFG, and simple scheduler arithmetic once individual parity is proven.

## 9. Position, timestep, and custom math

ControlNet uses the same sinusoidal timestep path as SD 1.5:
`Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)` followed by
`TimestepEmbedding`. Timestep embeddings can be precomputed per scheduler
timestep table, but the cast to sample dtype and optional `timestep_cond` must
match source.

ControlNet-specific custom math:

```python
scales = torch.logspace(-1, 0, len(down_residuals) + 1, device=sample.device)
down_residuals = [sample * scale * conditioning_scale for sample, scale in zip(...)]
mid_residual = mid_residual * scales[-1] * conditioning_scale
```

Global pooling, when configured, reduces residuals over spatial dims `(2,3)`.
Layout translation must rewrite those axes for NHWC or guard this branch out.

## 10. Preprocessing and input packing

Control image preprocessing uses `VaeImageProcessor` with
`do_convert_rgb=True`, `do_normalize=False`, and `vae_scale_factor=8`.
Therefore PIL/NumPy inputs become NCHW torch tensors in `[0,1]`, resized to
height/width multiples of 8, not VAE-normalized to `[-1,1]`. Tensor inputs that
already have latent-channel count can bypass generic image processing, so shape
validation must distinguish latent tensors from image tensors.

`prepare_image` repeats a single control image across prompt batch or repeats
per prompt image count when image batch matches prompt batch. In non-guess CFG,
it duplicates the control image batch with `torch.cat([image] * 2)`. In guess
mode, it does not duplicate; the pipeline later pads ControlNet residuals with
zeros for the unconditional batch.

Img2img adds image preprocessing and VAE encode to initial latents plus
strength-based timestep slicing. Inpaint adds mask preprocessing
(`do_binarize=True`, grayscale), masked image construction, masked image latents,
and either 9-channel UNet concatenation or 4-channel latent blending.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ControlNet and UNet shared down-block lowering

Source pattern: ControlNet down path uses the same block constructors and
weights shape family as SD 1.5 UNet down blocks, followed by zero 1x1 residual
projections.

Replacement: reuse SD 1.5 UNet block lowerings for ControlNet down/mid blocks,
then append explicit `1x1` conv residual heads.

Preconditions: block types, channels, attention heads, cross-attention dim,
norm eps/groups, and time conditioning match config. Do not assume up blocks:
ControlNet has no up path.

Failure cases: SDXL, union, global-pool, added embeddings, non-SD channel dims,
or inpaint 9-channel base UNet paths must select separate configs.

Parity test: compare one ControlNet down block plus its residual 1x1 head
against PyTorch for 64x64, 32x32, 16x16, and 8x8 latent maps.

### Rewrite: NCHW conv island -> guarded NHWC conv island

Source pattern: conditioning embedding, ControlNet ResNet/downsample blocks,
zero 1x1 residual heads, and VAE/UNet conv regions are NCHW conv/norm/SiLU
islands.

Replacement: translate local conv islands to NHWC/HWIO kernels with NCHW
boundaries or fuse boundary transposes across adjacent safe regions.

Preconditions: all consumers in the island are layout-rewritten; GroupNorm
channel axis changes from dim 1 to last dim; BGR flip changes from dim 1 to
last dim; spatial mean changes from `(2,3)` to `(1,2)`; channel concat for
inpaint changes from dim 1 to last dim.

Weight transform: OIHW conv weights become HWIO.

Failure cases: attention flatten/reshape, residual tuple ABI, scheduler
broadcast assumptions, and VAE/component boundaries not included in the island.

### Rewrite: Multi-ControlNet residual reduction

Source pattern: run N ControlNets, then elementwise add each corresponding
down/mid residual.

Replacement: explicit residual accumulation buffer per residual slot, optionally
fused with scale multiply.

Preconditions: all ControlNets produce identical residual slot shapes and dtypes.

Failure cases: heterogeneous SD/SDXL ControlNets, control guidance lists with
dynamic length, or global pooled residuals mixed with full-spatial residuals.

### Rewrite: guess-mode zero padding

Source pattern: conditional-only ControlNet residuals are padded with
`cat([zeros_like(d), d])`.

Replacement: pass residuals with a batch offset/mask to the UNet addition site,
or materialize a fused zero-pad+add kernel.

Preconditions: CFG batch order is always `[uncond, cond]`.

Failure cases: custom CFG batching, callback mutation, or different UNet batch
layout.

## 12. Kernel fusion candidates

Highest priority:

- ControlNet conditioning embedding conv + SiLU stack, especially stride-2 convs
  and final zero conv.
- Shared UNet/ControlNet ResnetBlock2D: GroupNorm + SiLU + Conv2d and time-bias
  add.
- Cross-attention Q/K/V + attention + out projection for ControlNet and UNet.
- Residual scale/add injection: ControlNet output scale, Multi-ControlNet sum,
  guess-mode pad, and UNet down/mid residual add.
- Scheduler + CFG elementwise/reduction kernels from the base SD report.

Medium priority:

- Guarded NHWC conv islands across ControlNet conditioning/down blocks.
- Fused `1x1` residual projection plus conditioning scale.
- Inpaint mask/latent concat or 4-channel latent blend kernels.
- VAE encode/decode fusions for img2img/inpaint variants.

Lower priority:

- Fused QKV projection weight transforms.
- Global pooled ControlNet residual branch, because it is inactive in inspected
  SD 1.5 configs.
- Multi-ControlNet dynamic-list specialization after single-ControlNet parity.

## 13. Runtime staging plan

Stage 1: Parse SD 1.5 base UNet/VAE/scheduler configs and one
`ControlNetModel` config. Treat prompt embeddings and preprocessed control image
as external inputs.

Stage 2: Reuse SD 1.5 operator parity for Conv2d, GroupNorm, SiLU, timestep
embedding, ResnetBlock2D, Downsample2D, BasicTransformerBlock, and Attention.

Stage 3: Add ControlNetConditioningEmbedding parity and zero `1x1` residual
head parity.

Stage 4: Compile one ControlNet down-block slice and validate all 12 down
residual slots plus mid residual at fixed timestep.

Stage 5: One denoising step parity: ControlNet residuals, UNet residual
injection, CFG, scheduler step. Use supplied prompt embeddings and control image.

Stage 6: Full text-to-image loop with scheduler in Python and compiled
ControlNet+UNet step. Add VAE decode from the existing SD slice.

Stage 7: Add `MultiControlNetModel` residual summation.

Stage 8: Add `controlnet_sd_img2img`, then `controlnet_sd_inpaint` as separate
candidate reports/slices. For inpaint, handle both 4-channel and 9-channel UNet
contracts explicitly.

Stage 9: Add NHWC conv-island and attention/provider fusions under guards.

## 14. Parity and validation plan

- Random tensor parity for `ControlNetConditioningEmbedding` at 512 and 768
  image sizes.
- Single zero `1x1` residual head parity, including zero-initialized weights.
- One ControlNet down block parity per resolution and channel width.
- Full `ControlNetModel.forward` parity for tiny config, then SD 1.5 canny
  config, fixed timestep, fixed prompt embeddings, fixed control image.
- Verify residual slot count, order, shape, dtype, and scale values.
- Guess-mode parity: conditional-only ControlNet call plus zero residual padding.
- Multi-ControlNet parity with two identical tiny ControlNets and independent
  scales.
- One UNet injection parity: PyTorch UNet with supplied residuals vs Dinoml
  lowered residual add sites.
- One denoising step parity including scheduler scale, ControlNet, UNet, CFG,
  and scheduler step.
- Img2img candidate: VAE encode, timestep slicing, noise add.
- Inpaint candidate: mask preprocessing, masked-image latents, 9-channel concat
  and 4-channel blend branches.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, tighten per kernel.

## 15. Performance probes

- ControlNet forward time by resolution, batch, dtype, and control type.
- UNet forward time with and without residual injection.
- Combined ControlNet+UNet denoising-step time and VRAM.
- Conditioning embedding cost versus ControlNet down/mid cost.
- Attention backend comparison for ControlNet and UNet blocks.
- Multi-ControlNet scaling: one, two, and three ControlNets.
- Guess-mode speed/memory versus non-guess CFG path.
- NCHW faithful path versus guarded NHWC conv islands.
- Img2img/inpaint VAE encode and mask/latent preprocessing overhead.
- Scheduler/guidance/control-scale overhead as separate small-kernel probes.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `controlnet_sd_img2img`: image encode and strength/timestep contract.
- `controlnet_sd_inpaint`: mask, masked image latents, 4-channel blend and
  9-channel UNet concat contracts.
- `controlnet_sd_multi`: list inputs, per-ControlNet scales/windows, residual
  reduction.
- `controlnet_sdxl`: SDXL prompt/time conditioning and wider configs.
- `controlnet_union`: control mode/type embeddings and union model logic.
- `sd1_ip_adapter`: added image embeds and added K/V attention processors.
- `sd1_lora_textual_inversion_adapters`: loader/runtime weight and embedding
  mutation.
- `sd1_t2i_adapter`, `sd1_gligen`, `sd1_depth2img`, `sd_upscale`: related SD
  conditioning variants with different pipeline contracts.
- Rare scheduler variants beyond the first scheduler slice.

Ignored/out of scope for this audit:

- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Callback mutation and interactive interrupt.
- Multi-GPU/context parallel paths.
- XLA, NPU, MPS, Flax, and ONNX-specific paths.

## 17. Final implementation checklist

- [ ] Parse one SD 1.5 base pipeline config and one `ControlNetModel` config.
- [ ] Load ControlNet weights separately from the base UNet weights.
- [ ] Accept external `prompt_embeds`, `negative_prompt_embeds`, control image,
      latents, timestep, and scheduler state.
- [ ] Implement `ControlNetConditioningEmbedding`.
- [ ] Reuse/lower SD 1.5 down-block and mid-block operators for ControlNet.
- [ ] Implement zero `1x1` residual heads and preserve residual slot order.
- [ ] Implement conditioning scale, control guidance windows, and guess-mode
      scale schedule.
- [ ] Implement UNet down/mid residual injection.
- [ ] Add one-step ControlNet+UNet+CFG+scheduler parity.
- [ ] Add full text-to-image loop smoke with scheduler in Python.
- [ ] Add VAE decode reuse from SD 1.5 slice.
- [ ] Add Multi-ControlNet residual accumulation.
- [ ] Create separate img2img and inpaint ControlNet candidate reports before
      implementing those variants.
- [ ] Add guarded NHWC conv-island rewrite for conditioning/down/mid conv regions.
- [ ] Benchmark ControlNet, UNet, VAE, scheduler/guidance, and residual
      injection separately.
