# Diffusers Stable Cascade Operator and Integration Report

Target slug: `stable_cascade`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  stabilityai/stable-cascade-prior
  stabilityai/stable-cascade
  Disty0/sotediffusion-wuerstchen3
  Disty0/sotediffusion-wuerstchen3-decoder
  Disty0/sotediffusion-v2

Config sources:
  H:/configs/stabilityai/stable-cascade-prior/model_index.json
  H:/configs/stabilityai/stable-cascade/model_index.json
  H:/configs/Disty0/sotediffusion-wuerstchen3/model_index.json
  H:/configs/Disty0/sotediffusion-wuerstchen3-decoder/model_index.json
  H:/configs/Disty0/sotediffusion-v2/model_index.json
  H:/configs/k33pCum/stable-cascade/model_index.json
  Official component JSON was read from Hugging Face raw URLs without saving
  because this task's owned write path is only this report.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_prior.py
  X:/H/diffusers/src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade.py
  X:/H/diffusers/src/diffusers/pipelines/stable_cascade/pipeline_stable_cascade_combined.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_stable_cascade.py
  X:/H/diffusers/src/diffusers/pipelines/deprecated/wuerstchen/modeling_paella_vq_model.py
  X:/H/diffusers/src/diffusers/pipelines/deprecated/wuerstchen/modeling_wuerstchen_common.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vq_model.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddpm_wuerstchen.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_lcm.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/single_file_utils.py

External component configs inspected:
  CLIPTextModelWithProjection config from both official repos.
  CLIPVisionModelWithProjection and CLIPImageProcessor configs from the prior repo.
  CLIPTokenizer tokenizer_config from both official repos.

Any missing files or assumptions:
  Stable Cascade pipelines are marked deprecated in current Diffusers with
  `_last_supported_version = "0.35.2"`, but the source remains non-deprecated
  under `pipelines/stable_cascade`. Official component configs were accessible;
  no gated official config blocked this audit. Ignore safety/NSFW, callbacks,
  training/dropout behavior, gradient checkpointing, multi-GPU/offload mechanics,
  XLA/NPU/MPS/Flax/ONNX, as requested.
```

## 2. Pipeline and component graph

Stable Cascade is a two-stage latent cascade plus a Paella VQ decoder. Stage C
is the prior: CLIP text and optional CLIP image embeddings denoise a compact
16-channel image embedding grid. Stage B is the decoder: pooled CLIP text plus
the Stage C grid condition a 4-channel Paella latent grid, which Paella decodes
to RGB.

```text
prompt and optional image conditioning
  -> prior tokenizer + CLIPTextModelWithProjection
  -> optional CLIPImageProcessor + CLIPVisionModelWithProjection
  -> Stage C StableCascadeUNet prior denoising loop
       latent image embedding [B,16,ceil(H/42.67),ceil(W/42.67)]
  -> decoder tokenizer + CLIPTextModelWithProjection, pooled embedding only
  -> Stage B StableCascadeUNet decoder denoising loop
       Paella latent [B,4,int(Hc*10.67),int(Wc*10.67)]
  -> multiply by Paella scale_factor
  -> PaellaVQModel.decode(..., force_not_quantize=True)
  -> clamp [0,1] and PIL/np/pt postprocess
```

Required first-slice components: `StableCascadePriorPipeline`,
`StableCascadeDecoderPipeline`, two `StableCascadeUNet` configs, two
`DDPMWuerstchenScheduler` instances, `CLIPTokenizerFast`,
`CLIPTextModelWithProjection`, and `PaellaVQModel`. The combined pipeline is a
thin wrapper that instantiates the prior and decoder pipelines and forwards
outputs between them. Optional prior image conditioning uses
`CLIPImageProcessor` and `CLIPVisionModelWithProjection`; when absent, the
prior supplies zero CLIP image embeddings.

Independently cacheable stages: prior prompt hidden states, prior pooled text
embeddings, optional CLIP image embeddings, Stage C image embeddings, decoder
pooled text embeddings, scheduler timestep tables, and caller-supplied initial
latents for either denoising loop.

Separate candidate reports:

| Surface | Classes/files | Pipeline delta |
| --- | --- | --- |
| `stable_cascade_lite` | `decoder_lite/config.json`, `prior_lite/config.json`, `single_file_utils.py` | Same pipeline contracts but narrower/deeper settings change attention head counts and block repetition. |
| `stable_cascade_lcm_decoder` | `Disty0/sotediffusion-v2`, `LCMScheduler` | Decoder swaps DDPMWuerstchen for LCM epsilon scheduler; first parity should not claim this unless LCM loop is admitted. |
| `stable_cascade_single_file` | `loaders/single_file_utils.py`, `single_file_model.py` | Converts original Stage B/C checkpoints, including QKV split and `clip_mapper` rename. |
| `wuerstchen_legacy` | `pipelines/deprecated/wuerstchen/*` | Related prior/decoder family with different model classes; useful only after Stable Cascade or for migration parity. |
| LoRA/textual inversion/runtime adapters | Generic loaders only; Stable Cascade pipelines do not inherit LoRA/textual inversion mixins | Not active in the base family. Treat any external adapter mutation as a separate loader admission task. |
| IP-Adapter | No Stable Cascade pipeline IP-Adapter path found | Prior already has optional CLIP image conditioning, but not added-K/V IP-Adapter attention. |
| ControlNet/T2I-Adapter/GLIGEN/img2img/inpaint/depth/upscale | No Stable Cascade-specific pipeline classes found | Separate only if a downstream repo adds custom classes; not in first Stable Cascade source scope. |

## 3. Important config dimensions

| Component | Repo/config | Channels | Blocks/layers | Attention | Conditioning | Patch/scale |
| --- | --- | --- | --- | --- | --- | --- |
| Stage C prior | `stabilityai/stable-cascade-prior/prior` | `16 -> 16`, hidden `2048/2048` | down `8/24`, up `24/8` | heads `32/32`, dim head 64 | pooled CLIP 1280, token CLIP 1280, image CLIP 768, `clip_seq=4`, cond dim 2048 | `patch_size=1`, `switch_level=[false]` |
| Stage C prior lite | official `prior_lite` | `16 -> 16`, hidden `1536/1536` | down `4/12`, up `12/4` | heads `24/24`, dim head 64 | same CLIP widths, cond dim 1536 | `patch_size=1`, `switch_level=[false]` |
| Stage B decoder | `stabilityai/stable-cascade/decoder` | `4 -> 4`, hidden `320/640/1280/1280` | down `2/6/28/6`, up `6/28/6/2` | heads `0/0/20/20`, dim head 64 | pooled CLIP 1280 only, effnet 16, pixels 3, cond dim 1280 | `patch_size=2`, `latent_dim_scale=10.67` |
| Stage B decoder lite | official `decoder_lite` | `4 -> 4`, hidden `320/576/1152/1152` | down `2/4/14/4`, up `4/14/4/2` | heads `0/9/18/18`, dim head 64 | pooled CLIP 1280 only, effnet 16, pixels 3, cond dim 1280 | `patch_size=2`, `latent_dim_scale=10.67` |
| Paella VQ | `vqgan/config.json` | RGB `3`, latent `4`, embed `384` | levels 2, bottleneck 12 | none | vector quantizer 8192 codes but decode defaults to no quantization | `up_down_scale_factor=2`, `scale_factor=0.3764` |

| External encoder | Class | hidden/projection | layers/heads | input |
| --- | --- | ---: | ---: | --- |
| Text encoder | `CLIPTextModelWithProjection` | hidden 1280, projection 1280 | 32 layers, 20 heads | CLIP tokenizer max length 77 |
| Optional image encoder | `CLIPVisionModelWithProjection` | hidden 1024, projection 768 | 24 layers, 16 heads | 224x224 image, patch 14 |
| Feature extractor | `CLIPImageProcessor` | normalize/rescale/crop | n/a | center crop 224, CLIP mean/std |

Scheduler defaults: official prior and decoder use `DDPMWuerstchenScheduler`
with `s=0.008`, `scaler=1.0`, `init_noise_sigma=1.0`, and timesteps as float
ratios from `1.0` to `0.0`. The pipelines drop the final zero timestep for this
scheduler. `Disty0/sotediffusion-v2` is a decoder-only variant using
`LCMScheduler` with epsilon prediction, `original_inference_steps=50`, and
`timestep_scaling=10.0`; treat it as a separate scheduler candidate. Recommended
first Dinoml scheduler slice: `DDPMWuerstchenScheduler` only.

## 3a. Family variation traps

- There are two denoisers with the same `StableCascadeUNet` class but different
  channel counts, conditioning inputs, patch size, and scheduler loop defaults.
- Stage C latent size uses `ceil(height / 42.67)` and `ceil(width / 42.67)`.
  For the combined pipeline default 512x512, this is `12x12`; for the prior
  standalone default 1024x1024, this is `24x24`.
- Stage B latent size is `int(stage_c_h * 10.67)` by `int(stage_c_w * 10.67)`.
  A 24x24 Stage C grid becomes a 256x256 Paella latent grid, while 12x12 becomes
  128x128.
- Stage C CFG concatenates positive then negative prompt branches and uses
  `torch.lerp(uncond, text, guidance_scale)`. Stage B defaults
  `guidance_scale=0.0`, so CFG is inactive unless explicitly set above 1.
- Prior image conditioning is not IP-Adapter. It maps CLIP image embeddings into
  extra conditioning tokens through `clip_img_mapper`.
- Decoder uses only pooled CLIP text; it does not pass token hidden states to
  Stage B under the inspected configs.
- `StableCascadeUNet` source is NCHW but frequently permutes to NHWC for
  `LayerNorm`, channel MLPs, and `GlobalResponseNorm`. A layout pass must guard
  all permute/view assumptions.
- `SDCascadeAttnBlock` passes a 4D NCHW tensor to `Attention`; the attention
  processor flattens spatial tokens internally. If `self_attn=true`, K/V tokens
  are the normalized image tokens concatenated with conditioning tokens.
- Lite configs are not just smaller weights; Stage B lite enables attention at
  the second level with 9 heads, while full Stage B has no attention in the
  first two levels.
- `single_file_utils.py` detects full versus lite by checkpoint tensor shapes
  and splits original `in_proj_weight` into Diffusers Q/K/V weights.

## 4. Runtime tensor contract

For a 1024x1024 prior-only call with one image per prompt:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| CLIP tokens | `prompt_embeds` | `[B,S,C]` | `[B,77,1280]`; CFG `[2B,77,1280]` |
| CLIP pooled | `prompt_embeds_pooled` | `[B,1,C]` | `[B,1,1280]`; CFG `[2B,1,1280]` |
| optional CLIP image | `image_embeds` | `[B,N,C]` | `[B,N,768]`; zero-filled when absent |
| Stage C latent | image embedding | NCHW | `[B,16,24,24]` for 1024; `[B,16,12,12]` for 512 |
| Stage C UNet output | predicted embedding/noise | NCHW | same as Stage C latent |
| Stage B latent | Paella latent | NCHW | `[B,4,256,256]` for 24x24 Stage C; `[B,4,128,128]` for 12x12 |
| Stage B conditioning | `effnet` | NCHW | Stage C embedding `[B,16,Hc,Wc]`, interpolated to hidden map sizes |
| Stage B pixels | default zeros | NCHW | `[B,3,8,8]`, interpolated if pixel mapper present |
| Paella decode input | scaled Stage B latent | NCHW | `0.3764 * latents` |
| image output | decoded RGB | NCHW then NHWC for np/PIL | `[B,3,H,W]`, clamped `[0,1]` |

`StableCascadeUNet.forward` inputs:

```text
sample: NCHW latent map
timestep_ratio: [B] float ratio
clip_text_pooled: [B,1,1280] or [B,1280]
clip_text: optional [B,77,1280] for Stage C
clip_img: optional [B,N,768] for Stage C
effnet: optional [B,16,Hc,Wc] for Stage B
pixels: optional [B,3,Hp,Wp], defaults to zeros
sca/crp: optional [B] extra timestep-ratio conditioning
```

The model builds a timestep embedding of width 64 for `timestep_ratio`, then
concatenates another 64-wide embedding per configured conditioning type. Stage C
uses `sca` and `crp`, so the adaptive timestep block receives 192 channels.
Stage B uses only `sca`, so it receives 128 channels. When `sca`/`crp` are not
passed, zeros are embedded and still occupy the configured width.

CPU/data-pipeline work: tokenization, CLIP preprocessing, PIL/np conversion, and
postprocess conversion. GPU/runtime work: CLIP encoders if admitted, both
denoising loops, CFG arithmetic, scheduler step, Paella decode.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCHW latent maps; NHWC internal permutations for layer norm and channel MLP.
- `ceil`/`int` latent shape calculations in host admission.
- CFG batch concat and output chunk.
- Token concat for Stage C conditioning: mapped text tokens, mapped pooled text
  tokens, mapped CLIP image tokens.
- Spatial flatten/unflatten in `Attention` and self-attention K/V concat.
- `PixelUnshuffle(patch_size)` and `PixelShuffle(patch_size)` in UNet stem/head.
- Bilinear interpolate with `align_corners=True` for effnet/pixels mappers and
  skip-size repairs.
- Paella RGB NHWC/NCHW conversion for output only.

### Convolution/downsample/upsample ops

- Stage C stem `Conv2d(16 -> 2048, 1x1)` after pixel unshuffle.
- Stage B stem `PixelUnshuffle(2)` then `Conv2d(16 -> 320, 1x1)`.
- Depthwise Conv2d in every `SDCascadeResBlock`, with groups equal channels.
- 1x1 mapper Conv2d repeat mappers and channel transition mappers.
- Downsample Conv2d kernel 2 stride 2 in normal configs; `UpDownBlock2d`
  bilinear+1x1 path when `switch_level` requests it.
- ConvTranspose2d kernel 2 stride 2 in upscalers for normal configs.
- Paella PixelUnshuffle, 1x1 convs, depthwise 3x3 with replication pad,
  Conv2d kernel 4 stride 2 down, ConvTranspose2d kernel 4 stride 2 up.

### GEMM/linear ops

- CLIP pooled/text/image mappers to conditioning width.
- Timestep adaptive scale/bias Linear to `2*C` per block and per condition.
- Channelwise MLP: `Linear(C + Cskip -> 4C) -> GELU -> GlobalResponseNorm ->
  Linear(4C -> C)`.
- Attention Q/K/V/out projections with bias.
- Paella channelwise MLPs in `MixingResidualBlock`.

### Attention primitives

- Noncausal self-plus-conditioning attention in Stage C both levels.
- Noncausal self-plus-pooled-text attention in Stage B attention levels.
- Default processor is `AttnProcessor2_0` when PyTorch SDPA is available and
  `scale_qk=true`; eager `AttnProcessor` defines fallback parity.
- No QK norm, RoPE, masks, added-KV, GQA, or varlen path required by base configs.

### Normalization and adaptive conditioning

- `SDCascadeLayerNorm`: LayerNorm over channel after NCHW to NHWC permutation.
- `GlobalResponseNorm`: L2 norm over spatial NHW axes in NHWC channel MLP.
- BatchNorm2d in Paella encoder tail; decode-only first slice does not use it.
- Adaptive scale/shift in `SDCascadeTimestepBlock`: `x * (1 + a) + b`.

### Scheduler and guidance arithmetic

- Wuerstchen cosine alpha-cumprod from float timestep ratios.
- DDPM-style stochastic step with per-sample timestep vector broadcasting.
- `torch.lerp(uncond, text, guidance_scale)` for CFG.
- LCM decoder variant as separate follow-up.

## 6. Denoiser/model breakdown

`StableCascadeUNet` shared forward:

```text
condition tokens:
  pooled CLIP -> Linear -> [B, clip_seq, conditioning_dim]
  optional text CLIP -> Linear -> [B,77,conditioning_dim]
  optional image CLIP -> Linear -> [B,N*clip_seq,conditioning_dim]
  concat -> LayerNorm

sample:
  PixelUnshuffle(patch_size) -> Conv2d(1x1) -> channel LayerNorm
  + optional effnet_mapper(interpolate(effnet))
  + optional pixels_mapper(interpolate(pixels))
  -> down blocks
  -> up blocks with first-resblock skip concat at each higher level
  -> channel LayerNorm -> Conv2d(1x1) -> PixelShuffle(patch_size)
```

`SDCascadeResBlock`:

```text
residual = x
depthwise Conv2d(groups=C) -> channel LayerNorm
optional channel concat with skip
NHWC Linear -> GELU -> GlobalResponseNorm -> Dropout(source; ignored for inference)
-> Linear -> NCHW -> residual add
```

`SDCascadeTimestepBlock`: split the concatenated timestep embedding into base
and configured condition chunks; each chunk maps through a Linear to scale/bias,
sums scale/bias, and applies channel-wise adaptive affine.

`SDCascadeAttnBlock`: channel LayerNorm; map conditioning tokens through
`SiLU -> Linear(c_cond -> C)`; when `self_attn=true`, prepend normalized spatial
tokens to K/V tokens; call `Attention(query_dim=C, heads=nhead, dim_head=C/nhead,
bias=True)` over the NCHW query map.

Stage C full has 8 and 24 repetitions down, then 24 and 8 up, all levels
attention-enabled at hidden 2048 with 32 heads. Stage B full has 4 levels;
attention is active only at hidden 1280 levels, with 20 heads and head dim 64.

## 7. Attention requirements

- Attention file: `models/attention_processor.py` is the primary implementation
  path via `Attention`.
- Query input may be 4D NCHW; `AttnProcessor2_0` flattens to `[B,HW,C]`.
- Stage C full attention: hidden 2048, 32 heads, head dim 64. Conditioning K/V
  sequence is `HW + 77 + 4 + image_count*4` when self-attention and optional
  image conditioning are active.
- Stage C lite: hidden 1536, 24 heads, head dim 64.
- Stage B full attention: hidden 1280, 20 heads, head dim 64. K/V sequence is
  `HW + 4` pooled-text tokens for active levels.
- Stage B lite additionally has hidden 576, 9 heads at level 1 and hidden 1152,
  18 heads at deeper levels.
- No masks are passed by base pipelines. SDPA receives `dropout_p=0.0` and
  `is_causal=False`.
- Source supports processor swaps and xFormers through generic pipeline helpers,
  but first parity should keep unfused Q/K/V projections and SDPA/eager shape.
- Dinoml flash-style attention is valid under strict guards: no masks, no custom
  processor, no added K/V, no training dropout, head dim 64, and conditioning
  tokens already materialized. The fallback must remain the explicit Q/K/V,
  softmax, V matmul, output projection path.

## 8. Scheduler and denoising-loop contract

Official Wuerstchen loop:

```text
scheduler.set_timesteps(num_inference_steps, device)
timesteps = scheduler.timesteps[:-1]
latents = randn(shape) * init_noise_sigma
for t in timesteps:
  timestep_ratio = t.expand(batch)
  model_output = unet(sample=CFG_cat(latents), timestep_ratio=CFG_cat(timestep_ratio), ...)
  if CFG: model_output = lerp(uncond, text, guidance_scale)
  latents = scheduler.step(model_output, timestep_ratio, latents).prev_sample
```

`DDPMWuerstchenScheduler.set_timesteps` creates `num_inference_steps + 1`
linearly spaced float ratios from 1 to 0 if custom timesteps are absent. The
pipeline discards the final zero ratio before iteration. `scale_model_input` is
identity. `step` computes a cosine alpha-cumprod at current and previous ratio,
forms a DDPM mean, samples Gaussian noise, and suppresses noise for the final
previous ratio equal to zero.

The source includes compatibility code for non-Wuerstchen schedulers: if a
scheduler has `clip_sample=true`, the pipeline mutates it to false and computes
a timestep-ratio conditioning from `alphas_cumprod` or a normalized timestep.
That path is active for the LCM decoder variant and should be a separate
admission item because it changes both scheduler math and model timestep input.

Keep timestep table generation, custom timesteps validation, stochastic noise
source, and loop iteration as host-visible state first. Compile one scheduler
step only after the Wuerstchen timestep ratio and generator/noise policy are
explicit in the artifact.

## 9. Position, timestep, and custom math

`StableCascadeUNet.get_timestep_ratio_embedding` is sinusoidal:

```text
r = timestep_ratio * 10000
freq = exp(arange(half_dim) * -log(10000)/(half_dim-1))
embedding = concat(sin(r*freq), cos(r*freq))
```

The scheduler's alpha-cumprod is also custom ratio math:

```text
alpha_bar(t) = cos((t + s) / (1 + s) * pi/2)^2 / init_alpha
```

with optional `scaler` remapping of `t` before the cosine. For official configs
`scaler=1.0`. Stage C/Stage B also embed zero default `sca`/`crp` conditions
when they are configured but not passed; Dinoml must preserve the extra chunks
because the timestep block weights expect them.

`GlobalResponseNorm` is custom but simple: for NHWC `x`, compute L2 norm over
spatial axes `(1,2)`, divide by the mean over channels, then apply learned
gamma/beta and residual.

Precomputable: timestep ratio embeddings for fixed schedules and batch size,
CLIP mapped conditioning tokens, zero image embeddings, zero pixel mapper input,
and scheduler alpha tables. Dynamic: image size determines latent grid sizes;
guidance scale affects CFG arithmetic; stochastic scheduler noise depends on
generator state each step.

## 10. Preprocessing and input packing

- Prior tokenization uses CLIP max length 77, pads/truncates, and uses the final
  hidden state plus `text_embeds.unsqueeze(1)` as pooled text.
- Decoder only needs pooled text embeddings; if prompt embeddings are supplied
  from prior outputs, the combined pipeline reuses them.
- Negative prompts are generated only when guidance is active; no
  `force_zeros_for_empty_prompt` option exists in these pipelines.
- Optional prior image conditioning runs CLIP preprocessing: RGB conversion,
  resize shortest edge 224, center crop 224, rescale, normalize by CLIP mean/std,
  then CLIP vision projection to 768 and `unsqueeze(1)`.
- Stage C image embeddings may also be supplied directly through `image_embeds`.
- Stage B adjusts `num_images_per_prompt` by `image_embeddings.shape[0] //
  batch_size`, because the prior may already have expanded images per prompt.
- Paella decode multiplies latents by `scale_factor=0.3764`, decodes without
  vector quantization by default, clamps to `[0,1]`, and permutes to NHWC for
  NumPy/PIL outputs.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Cascade NCHW/NHWC channel-MLP block

Source pattern: `DepthwiseConv2d NCHW -> LayerNorm via permute -> optional
channel concat -> NHWC Linear/GELU/GRN/Linear -> permute -> residual`.

Replacement: a channel-last block with NCHW boundaries only at component
interfaces, or a faithful NCHW block with explicit permutes.

Preconditions: no consumer observes intermediate layout; channel concat axis is
rewritten from `dim=1` to channel-last; GRN reduction axes remain spatial; Conv2d
weights are transformed OIHW to HWIO if using NHWC kernels.

Failure cases: attention flatten expects NCHW view semantics, skip tensors at
different layout, or Paella/UNet boundary tensors consumed by source-layout code.
Parity sketch: test `SDCascadeResBlock` at 320, 1280, 1536, and 2048 channels
with and without skip.

### Rewrite: PixelUnshuffle plus 1x1 conv

Source pattern: `PixelUnshuffle(p) -> Conv2d(C*p*p -> hidden, 1x1)`.

Replacement: patch-extract/linear projection over p-by-p source pixels.

Preconditions: patch size fixed by config, NCHW flatten order matches PyTorch
PixelUnshuffle, bias preserved, no padding. Weight transform must map
`[hidden, C*p*p, 1, 1]` to a patch projection with the exact unshuffle channel
order. Failure cases: dynamic patch size, channel-last pass without order guard.

### Rewrite: Wuerstchen scheduler step

Source pattern: per-step cosine alpha calculation, DDPM mean, random noise, and
final-step noise mask.

Replacement: host-precomputed scalar coefficients plus one fused pointwise
latent update and optional noise input.

Preconditions: scheduler is `DDPMWuerstchenScheduler`, `scaler/s` fixed, timestep
table explicit, stochastic noise tensor supplied or generator policy defined.
Failure cases: LCM scheduler variant, custom non-monotonic timesteps, or desire
for exact PyTorch RNG parity inside a compiled kernel.

### Rewrite: conditioning token materialization

Source pattern: mapped CLIP text, mapped pooled tokens, mapped image tokens,
concat, layer norm; reused every denoising step.

Replacement: precompute conditioning token tensor once per request and pass it
directly to each attention block.

Preconditions: CLIP outputs, `clip_seq`, and optional image count fixed for the
request; no adapter mutates mapper weights mid-loop. Failure cases: dynamic
prompt/image embedding mutation or processor hooks.

## 12. Kernel fusion candidates

Highest priority:

- Depthwise Conv2d + channel LayerNorm + channel MLP + GRN + residual in both
  Stage C and Stage B.
- Attention Q/K/V/out projection plus SDPA/softmax attention for head dim 64.
- Wuerstchen scheduler step and CFG lerp over NCHW latents.
- Paella decode blocks: LayerNorm/Conv depthwise/channel MLP and ConvTranspose.

Medium priority:

- PixelUnshuffle/PixelShuffle and 1x1 conv projection fusions.
- Effnet/pixel mapper interpolation + 1x1 conv + add in Stage B.
- Timestep embedding + adaptive scale/shift block.
- Guarded NHWC conv/MLP islands for Cascade UNet and Paella decode.

Lower priority:

- LCM decoder variant scheduler math.
- Single-file checkpoint QKV split and `clip_mapper` rename as loader rewrites.
- Paella encode and vector quantizer, because base text-to-image only decodes.

## 13. Runtime staging plan

Stage 1: Parse official prior, decoder, prior_lite, decoder_lite, scheduler,
text/image encoder, and Paella configs. Admit full Wuerstchen first; record lite
as shape variants.

Stage 2: Load Stage B decoder and Paella weights; accept external Stage C image
embeddings and pooled CLIP embeddings. Validate one Stage B block and Paella
decode.

Stage 3: Implement one Stage B denoising step with `DDPMWuerstchenScheduler`,
CFG disabled by default, scheduler in host control.

Stage 4: Add Stage C prior with external prompt/image embeddings, then CLIP text
embedding cache integration.

Stage 5: Chain Stage C output into Stage B and Paella decode for a deterministic
short loop.

Stage 6: Add optional prior image conditioning through external CLIP image
embeddings; treat CLIP vision execution itself as a later Transformers-backed
stage.

Stage 7: Add lite configs and then Disty LCM decoder as separate scheduler
admission.

Stage 8: Add guarded NHWC/channel-last islands, attention provider lowering, and
PixelUnshuffle projection rewrites.

Stub initially: CLIP encoders, optional prior image encoder, Paella encode,
single-file conversion, and all absent ControlNet/IP-Adapter/inpaint/upscale
surfaces.

## 14. Parity and validation plan

- Config parse parity for official full and lite Stage C/Stage B JSON.
- `get_timestep_ratio_embedding` parity for scalar and batched ratios.
- `GlobalResponseNorm` random tensor parity in fp32/fp16/bf16.
- `SDCascadeLayerNorm` parity with NCHW input and NHWC optimized candidate.
- `SDCascadeResBlock`, `SDCascadeTimestepBlock`, and `SDCascadeAttnBlock` parity
  at representative full/lite channel widths.
- Full `StableCascadeUNet` forward parity on tiny synthetic shapes, then official
  Stage B 12x12 and Stage C 24x24 latent grids.
- `DDPMWuerstchenScheduler.set_timesteps`, `add_noise`, `previous_timestep`, and
  one `step` parity with fixed noise tensor/generator.
- CFG `torch.lerp` parity for prior and decoder.
- Paella decode parity from `[B,4,128,128]` and `[B,4,256,256]` latents.
- End-to-end smoke with cached CLIP embeddings and fixed initial latents.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; model fp32
  `rtol=1e-4, atol=1e-5`; fp16/bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Stage C one-step latency by latent grid 12x12, 24x24, and batch/CFG mode.
- Stage B one-step latency by Paella latent 128x128 and 256x256.
- Attention time versus ConvNeXt-style channel MLP time in Stage C and deep
  Stage B blocks.
- Paella decode throughput for 512 and 1024 outputs.
- Wuerstchen scheduler and CFG overhead per step.
- Cached CLIP text embedding throughput versus denoiser-only loop.
- Optional CLIP vision image-conditioning overhead.
- NCHW faithful path versus guarded NHWC channel-MLP/conv islands.
- Full cascade latency split: prior loop, decoder loop, Paella decode.
- VRAM/workspace usage for full and lite variants.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `stable_cascade_lite`: official lite Stage C/B configs with different widths,
  depth, and Stage B attention placement.
- `stable_cascade_lcm_decoder`: Disty decoder variant with `LCMScheduler`.
- `stable_cascade_single_file`: checkpoint loader conversion and original
  attention QKV split/rename rules.
- `wuerstchen_legacy`: deprecated Wuerstchen prior/decoder classes and older
  model contracts.
- `stable_cascade_clip_encoders`: CLIP text and optional CLIP vision execution
  if Dinoml wants to compile the encoders rather than accept cached embeddings.
- `stable_cascade_paella_encode_quantize`: Paella encode and vector quantizer
  paths, which are not required for base text-to-image decode.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel and offload behavior.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX paths.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- ControlNet, T2I-Adapter, GLIGEN, IP-Adapter, img2img, inpaint, depth2img, and
  upscaling because no Stable Cascade-specific source classes were found for
  those surfaces in the inspected Diffusers checkout.

## 17. Final implementation checklist

- [ ] Parse Stable Cascade prior/decoder full and lite component configs.
- [ ] Load `StableCascadeUNet` Stage B weights and Paella VQ decoder weights.
- [ ] Accept external Stage C image embeddings and decoder pooled CLIP embeddings.
- [ ] Implement `SDCascadeLayerNorm`, `GlobalResponseNorm`, and depthwise-channel MLP block parity.
- [ ] Implement PixelUnshuffle/PixelShuffle contracts for Stage B.
- [ ] Implement Stage B attention at hidden 1280 and head dim 64.
- [ ] Implement `DDPMWuerstchenScheduler` timesteps and one-step update.
- [ ] Add Stage B one-step parity with scheduler in host control.
- [ ] Add Paella decode parity with `scale_factor=0.3764`.
- [ ] Load Stage C prior weights and accept external CLIP text/image embeddings.
- [ ] Implement Stage C attention and timestep-conditioning parity.
- [ ] Add full prior-to-decoder handoff smoke with cached CLIP embeddings.
- [ ] Add optional prior image embedding path as external tensor input.
- [ ] Add lite config shape/admission tests.
- [ ] Add guarded NHWC channel-MLP/conv rewrite tests.
- [ ] Add attention provider lowering guarded for no masks/custom processors.
- [ ] Add separate LCM decoder variant only after LCM scheduler admission.
