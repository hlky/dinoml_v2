# Diffusers LEDITS++ Pipeline Audit

Target slug: `ledits_pp`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  No LEDITS++-specific checkpoint is required by the pipeline classes.
  SD1-style slow test loads stable-diffusion-v1-5/stable-diffusion-v1-5.
  SDXL-style slow test loads stabilityai/stable-diffusion-xl-base-1.0.
  Representative config sweep also used stabilityai/stable-diffusion-xl-refiner-1.0,
  stabilityai/sdxl-turbo, and echarlaix/tiny-random-stable-diffusion-xl.

Config sources:
  H:/configs/stable-diffusion-v1-5/stable-diffusion-v1-5/
  H:/configs/stabilityai/stable-diffusion-xl-base-1.0/
  H:/configs/stabilityai/stable-diffusion-xl-refiner-1.0/
  H:/configs/stabilityai/sdxl-turbo/
  H:/configs/echarlaix/tiny-random-stable-diffusion-xl/
  The SD1 component JSON files were missing from the local cache and were
  fetched with `hf download` from the official repo during this audit.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion.py
  diffusers/src/diffusers/pipelines/ledits_pp/pipeline_leditspp_stable_diffusion_xl.py
  diffusers/src/diffusers/pipelines/ledits_pp/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  diffusers/src/diffusers/image_processor.py
  diffusers/tests/pipelines/ledits_pp/test_ledits_pp_stable_diffusion.py
  diffusers/tests/pipelines/ledits_pp/test_ledits_pp_stable_diffusion_xl.py
  Related reports: stable_diffusion_1_5, stable_diffusion_xl,
  sd1_img2img_inpaint_depth_upscale, scheduler_matrix.

External component configs inspected:
  CLIPTextModel for SD1.
  CLIPTextModel and CLIPTextModelWithProjection for SDXL.
  CLIPTokenizer tokenizer_config files for SDXL.

Any missing files or assumptions:
  No gated config blocked the audit. LEDITS++ is an inversion/edit overlay over
  SD1-style and SDXL-style latent diffusion components, not a distinct weight
  family. Safety checking, watermarking, callbacks, XLA/NPU/MPS/Flax/ONNX,
  training, losses, gradient checkpointing, and multi-GPU/context parallel paths
  are out of scope. The deprecated `semantic_stable_diffusion` pipeline was
  searched only as related history and is not part of this non-deprecated target.
```

## 2. Pipeline and component graph

LEDITS++ has two non-deprecated pipeline classes:

| Class | File | Base component shape |
| --- | --- | --- |
| `LEditsPPPipelineStableDiffusion` | `pipeline_leditspp_stable_diffusion.py` | SD1-style single CLIP encoder, `UNet2DConditionModel`, `AutoencoderKL`, `DDIMScheduler` or `DPMSolverMultistepScheduler` |
| `LEditsPPPipelineStableDiffusionXL` | `pipeline_leditspp_stable_diffusion_xl.py` | SDXL dual CLIP encoders, SDXL `text_time` conditioning, `UNet2DConditionModel`, `AutoencoderKL`, `DDIMScheduler` or `DPMSolverMultistepScheduler` |

The public edit `__call__` requires a previous `invert(...)` call. Inversion
preprocesses and VAE-encodes the input image, computes inversion latents and
per-step variance-noise tensors, and stores them on the pipeline instance.

```text
input image
  -> VaeImageProcessor preprocess
  -> AutoencoderKL encode mode(), scaled by VAE scaling_factor
  -> inversion scheduler timesteps and add_noise table
  -> inversion loop:
       UNet on xt, optional source prompt CFG
       -> compute_noise(...) recovers variance noise z and corrected xt-1
  -> stored init_latents, zs, inversion_steps

negative/edit prompts or cached embeddings
  -> CLIP text encoder(s)
  -> concat unconditional + one or more edit concepts
  -> edit loop:
       duplicate stored latents for edit concepts
       -> scheduler.scale_model_input
       -> UNet denoiser
       -> semantic edit guidance, optional masks, optional guidance_rescale
       -> scheduler.step(..., variance_noise=zs[i])
  -> VAE decode and image postprocess
```

Required components for the first runtime slice:

- VAE encode and decode, not decode-only.
- UNet denoiser matching the selected SD1 or SDXL checkpoint.
- Prompt embeddings as an accepted external boundary; text encoders can be
  staged later.
- Scheduler state for DDIM or DPM-Solver SDE, including inversion-side
  `compute_noise` and edit-side `variance_noise`.
- LEDITS++ semantic guidance tensors, threshold masks, and optional
  cross-attention map capture.

Independently cacheable stages:

- Encoded/scaled input image latent `x0`.
- Negative, source, and edit prompt embeddings.
- SDXL pooled prompt embeddings and added time IDs.
- Scheduler timesteps, sigmas/alphas, and inversion `zs`.
- Attention-store maps only when mask visualization or cross-attention masks
  are enabled.

Separate candidate reports:

| Candidate | Class/file anchors | LEDITS++ delta |
| --- | --- | --- |
| LoRA / textual inversion / runtime adapters | SD class inherits `StableDiffusionLoraLoaderMixin`, `TextualInversionLoaderMixin`, `IPAdapterMixin`, `FromSingleFileMixin`; SDXL class inherits `StableDiffusionXLLoraLoaderMixin`, `TextualInversionLoaderMixin`, `IPAdapterMixin`, `FromSingleFileMixin` | Adapter/token/weight mutation can affect text encoders and UNet projections before inversion/edit. Model this as explicit artifact state. |
| IP-Adapter | SD/SDXL inherit `IPAdapterMixin`; SDXL `__call__` has `ip_adapter_image` branch and `added_cond_kwargs["image_embeds"]` | SDXL branch is present but source has a TODO around image encoding. Treat as separate from base LEDITS++ first slice. |
| ControlNet | No LEDITS++ ControlNet pipeline in `pipelines/ledits_pp`; use SD/SDXL ControlNet reports | Would add residual/control branches to the base UNet, not present in selected classes. |
| T2I-Adapter | No LEDITS++ T2I pipeline in this folder | Would add intrablock residuals, not present here. |
| GLIGEN | No non-deprecated LEDITS++ GLIGEN class | Deprecated SD GLIGEN remains separate history. |
| img2img / inpaint / depth / upscale | LEDITS++ itself is image-editing and always VAE-encodes an input image; no separate LEDITS++ inpaint/depth/upscale files | SD1/SDXL variant reports remain the anchor for mask/depth/low-res conditioning. LEDITS++ adds inversion and semantic masks instead of changing UNet channels. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Checkpoint/config | Pipeline class used by LEDITS++ tests/source | UNet in/out | Block channels | Cross dim | Text encoders | VAE scale | Source scheduler config | LEDITS++ effective scheduler |
| --- | --- | --- | --- | ---: | --- | ---: | --- | --- |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | `LEditsPPPipelineStableDiffusion` | 4 -> 4 | 320/640/1280/1280 | 768 | `CLIPTextModel` hidden 768, 77 tokens | omitted, effective 0.18215 | PNDM epsilon | Constructor converts unsupported PNDM to `DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=2)` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `LEditsPPPipelineStableDiffusionXL` | 4 -> 4 | 320/640/1280 | 2048 | CLIP 768 + CLIP bigG 1280; pooled 1280 | 0.13025, `force_upcast=true` | EulerDiscrete epsilon | Constructor converts unsupported Euler to `DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=2)` |
| `stabilityai/stable-diffusion-xl-refiner-1.0` | same SDXL class can load if components match | 4 -> 4 | 384/768/1536/1536 | 1280 | bigG-only model index | 0.13025 | EulerDiscrete epsilon | Same DPM conversion unless caller supplies DDIM/DPM |
| `stabilityai/sdxl-turbo` | source-compatible SDXL, but LEDITS++ not tested here | 4 -> 4 | 320/640/1280 | 2048 | dual CLIP | 0.13025 | EulerAncestral trailing | Converted to DPM if passed directly; Turbo low-step parity is not asserted |
| `echarlaix/tiny-random-stable-diffusion-xl` | debug-sized SDXL config | 4 -> 4 | 32/64 | 64 | tiny dual CLIP | 0.18215 | EulerDiscrete | Useful shape smoke only |

Scheduler support:

- The constructors state and enforce only `DDIMScheduler` and
  `DPMSolverMultistepScheduler`.
- Any other scheduler object is replaced from its config by
  `DPMSolverMultistepScheduler` with `algorithm_type="sde-dpmsolver++"` and
  `solver_order=2`.
- Inversion sets `scheduler.config.timestep_spacing = "leading"`, calls
  `set_timesteps(int(num_inversion_steps * (1 + skip)))`, and keeps the last
  `num_inversion_steps` timesteps.
- The edit loop then calls `scheduler.set_timesteps(len(self.scheduler.timesteps))`
  rather than accepting a fresh `num_inference_steps` argument.
- Recommended first Dinoml scheduler slice: DPM-Solver multistep
  `sde-dpmsolver++`, solver order 2, leading spacing, explicit
  `variance_noise`. DDIM with `eta=1.0` is a smaller second parity slice.

Guidance modes:

| Guidance | SD pipeline | SDXL pipeline | Notes |
| --- | --- | --- | --- |
| Source inversion guidance | Optional `source_prompt`; SD runs uncond and cond separately, SDXL batches them when CFG active | Same concept plus pooled/time conditioning | Used only during `invert`. |
| Edit guidance | One or more `editing_prompt` entries; batch is `[uncond, edit1, edit2, ...]` | Same, plus pooled edit embeddings and SDXL time IDs | Computes `edit - uncond`, optional reverse sign, scale, and masks. |
| `sem_guidance` override | List of precomputed tensors per step | Same | Bypasses prompt-derived edit guidance for supplied steps. |
| `guidance_rescale` | Optional std-ratio rescale | Optional std-ratio rescale | Reduction over all non-batch dims. |
| Embedded guidance | Not active for inspected SD1/SDXL configs (`time_cond_proj_dim=null`) | Source has SDXL branch if UNet declares `time_cond_proj_dim` | Separate from true edit guidance. |

## 3a. Family variation traps

- This target is not text-to-image. `invert` is mandatory and stores mutable
  pipeline state (`init_latents`, `zs`, `inversion_steps`, `batch_size`, and
  SDXL `size`).
- Passing an SD/SDXL default scheduler does not preserve that scheduler. PNDM,
  Euler, and Euler Ancestral are converted to DPM-Solver SDE by the constructors.
- SD1 `invert` runs unconditional and conditional UNet calls separately when a
  source prompt is present; SDXL batches source CFG during inversion.
- `invert` resets the UNet attention processor to eager `AttnProcessor`.
  Therefore edit calls after inversion use eager attention unless
  `use_cross_attn_mask` installs `LEDITSCrossAttnProcessor` for selected
  cross-attention layers.
- Cross-attention masks require attention probabilities from down/up `attn2`
  layers, aggregate only selected resolutions, smooth with a fixed 3x3 Gaussian,
  quantile-threshold, and upsample to latent resolution.
- Source tensors are NCHW. LEDITS++ masking math hard-codes channel dim 1 and
  repeats masks to 4 channels or `unet.config.in_channels`; NHWC lowering needs
  explicit axis rewrites.
- SDXL uses the base SDXL `text_time` conditioning path: pooled text embeddings
  plus six size/crop IDs. Refiner-style five-ID aesthetic conditioning is not
  represented in the LEDITS++ SDXL source call.
- The SDXL `ip_adapter_image` branch exists but includes a TODO comment around
  image encoding and should not inflate the first-slice contract.
- SDXL VAE `force_upcast` changes encode/decode dtype handling; SD1 VAE config
  omits `scaling_factor`, so source default 0.18215 applies.
- `num_zero_noise_steps` in SDXL inversion zeros final `zs` entries to avoid
  artifacts with DPM-Solver; SD1 has no matching argument.

## 4. Runtime tensor contract

For a 512x512 SD1 image:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| Preprocessed image | normalized image | NCHW | `[B,3,512,512]` |
| VAE encoded latent | `x0 = scaling_factor * mode(encode(image))` | NCHW | `[B,4,64,64]` |
| Inversion noisy path | `xts` | step, NCHW | `[num_steps+1,B,4,64,64]` after concatenating `x0` |
| Inversion variance noise | `zs` | step, NCHW | `[num_steps,B,4,64,64]`, flipped before storage |
| Stored edit start | `init_latents` | NCHW | `[B,4,64,64]` |
| SD1 prompt embeds | negative/edit CLIP hidden | `[B,S,C]` | negative `[B,77,768]`; edit `[num_edits*B,77,768]` |
| SD1 edit UNet input | duplicated latents | NCHW | `[(1+E)B,4,64,64]` |
| SD1 UNet output | noise predictions | NCHW | chunked into uncond plus E edit tensors |
| Edit mask | activation mask | NCHW | `[steps,E,B,4,64,64]` stored on CPU |
| Decoded output | VAE sample | NCHW -> postprocess | `[B,3,512,512]` before PIL/np |

For a 1024x1024 SDXL base image:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| VAE latent | `x0` | NCHW | `[B,4,128,128]` |
| Prompt embeds | dual CLIP concat | `[B,77,2048]` | negative plus edit concept batch |
| Pooled embeds | bigG pooled output | `[B,1280]` | negative plus edit concept batch |
| Time IDs | original/crop/target size | `[B,6]` | repeated and concatenated to edit batch |
| Added conditioning | `added_cond_kwargs` | dict | `{"text_embeds": pooled, "time_ids": ids}` |
| UNet latent input | duplicated latents | NCHW | `[(1+E)B,4,128,128]` |
| Output | VAE decode | NCHW -> postprocess | `[B,3,1024,1024]` before PIL/np |

CPU/data-pipeline work:

- PIL/NumPy/torch image normalization and resize/crop/fill.
- Tokenization, textual inversion token conversion, CLIP encoding.
- Optional output PIL/NumPy conversion.

GPU/runtime work:

- VAE encode/decode.
- UNet inversion and edit denoising.
- Scheduler `add_noise`, `compute_noise`, `scale_model_input`, and `step`.
- Semantic edit guidance, cross-attention aggregation, quantile masks, Gaussian
  smoothing, interpolate/repeat, and guidance rescale.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image/latent tensors; guarded NHWC candidates only inside local conv
  islands.
- `cat`, `chunk`, `repeat`, `view`, `reshape`, `permute`, `flatten`,
  `unsqueeze`, `squeeze`.
- Per-step tensors with leading step dimension for `xts`, `zs`,
  `sem_guidance`, and `activation_mask`.
- Quantile over flattened spatial maps, `where`, `abs`, `sum(dim=1)`,
  threshold compare, dtype upcast/downcast around quantile.
- `F.pad(..., mode="reflect")`, `F.interpolate`, fixed 3x3 `conv2d` smoothing.

Convolution/downsample/upsample ops:

- Same SD1/SDXL UNet and AutoencoderKL Conv2d/ResNet/down/up coverage as the
  base reports.
- Additional tiny depthwise-like smoothing conv: input `[B,1,h,w]`, kernel
  `[1,1,3,3]`, no padding after explicit reflect pad.

GEMM/linear ops:

- CLIP text encoders are external initially.
- UNet time embedding MLP, ResNet time projections, cross-attention Q/K/V/out,
  GEGLU/GELU feed-forward paths.
- SDXL `text_time` addition MLP from pooled CLIP plus projected size/crop IDs.

Attention primitives:

- Eager attention processor is parity-critical because `invert` resets to
  `AttnProcessor`.
- Optional `LEDITSCrossAttnProcessor` computes attention probabilities and
  stores them before `bmm(attention_probs, value)`.
- SDPA/flash paths are optimization candidates only when attention masks are not
  requested and the eager processor semantics are preserved.

Normalization and adaptive conditioning:

- UNet/VAE GroupNorm over channel axis; token LayerNorm in transformer blocks.
- SDXL `text_time` conditioning and optional `time_cond_proj_dim` guidance
  embedding branch if a future checkpoint enables it.

Scheduler and guidance arithmetic:

- DDIM `add_noise`, `step(..., variance_noise=...)`, and inversion
  `compute_noise_ddim`.
- DPM-Solver multistep `sde-dpmsolver++` conversion, model-output history,
  first/second-order update, explicit `variance_noise`.
- LEDITS++ edit guidance:
  `noise_guidance += scale * sign * (noise_pred_edit - noise_pred_uncond)`.
- Optional user mask multiplication, attention mask multiplication, intersection
  mask multiplication, and `guidance_rescale`.

VAE/postprocessing ops:

- VAE encode uses latent distribution mode, not sample.
- VAE decode divides by `scaling_factor`.
- SDXL may upcast VAE encode/decode when `force_upcast=true`.

## 6. Denoiser/model breakdown

The denoiser is inherited from SD1 or SDXL:

```text
sample [B,4,H,W]
-> timestep embedding
-> optional class/additional embedding
-> conv_in
-> down blocks with ResnetBlock2D and BasicTransformerBlock
-> mid block
-> up blocks with skip concat
-> GroupNorm -> SiLU -> conv_out
```

SD1 active path:

- Single CLIP text encoder hidden state width 768.
- `conv_in` is `Conv2d(4 -> 320, 3x3, padding=1)`.
- Cross-attention dim 768 and block channels 320/640/1280/1280.
- No `added_cond_kwargs` for vanilla SD1 configs.

SDXL active path:

- Prompt hidden states are feature-concatenated to width 2048 for base/turbo or
  width 1280 for refiner-style bigG-only configs.
- `conv_in` is `Conv2d(4 -> 320, 3x3, padding=1)` for base/turbo and
  `Conv2d(4 -> 384, 3x3, padding=1)` for refiner.
- `addition_embed_type="text_time"` requires pooled CLIP embeddings and time
  IDs.
- Base has transformer layers per block `[1,2,10]`; refiner has 4.

LEDITS++ changes the loop around the denoiser, not the UNet architecture. The
main new denoiser-side behavior is attention processor replacement for
cross-attention mask capture.

## 7. Attention requirements

Required for parity:

- Noncausal latent self/cross attention from the base SD1/SDXL UNet reports.
- Eager attention probability materialization when `use_cross_attn_mask` is
  active.
- Cross-attention store only for `attn2` processors in down/up blocks; mid block
  is deliberately not replaced by `LEDITSCrossAttnProcessor`.
- Attention maps are split by batch/head layout, reshaped to selected spatial
  resolutions, summed over heads, summed over edit tokens, then thresholded.

Backend dispatch:

- `attention_processor.py` is the primary implementation for this target.
- `invert` calls `self.unet.set_attn_processor(AttnProcessor())`, which rules
  out treating default SDPA as the source parity path for inversion/edit after
  inversion.
- `LEDITSCrossAttnProcessor` is an eager Q/K/V + attention-scores + BMM
  processor. It needs explicit attention probabilities, so fused flash kernels
  that do not expose probabilities cannot implement the cross-attention mask
  branch directly.
- Dinoml flash feasibility:
  - Valid candidate for no-mask reconstruction/editing when processors are
    standard, dropout is zero, and no attention maps are requested.
  - Invalid for `use_cross_attn_mask=True` unless a provider can return or
    separately compute the required probabilities at selected layers.
  - IP-Adapter added K/V branches and LoRA runtime mutation are separate
    admission surfaces.

## 8. Scheduler and denoising-loop contract

Inversion setup:

```text
self.eta = 1.0
scheduler.config.timestep_spacing = "leading"
scheduler.set_timesteps(int(num_inversion_steps * (1 + skip)))
inversion_steps = scheduler.timesteps[-num_inversion_steps:]
```

Inversion latent/noise construction:

```text
x0 = scaling_factor * VAE.encode(image).latent_dist.mode()
for reversed t in inversion_steps:
  xts[idx] = scheduler.add_noise(x0, random_noise, t)
xts = cat([x0[None], xts])
```

Inversion denoising:

```text
noise_pred = UNet(xt, t, uncond_embeds)
if source_prompt != "":
  noise_pred_cond = UNet(xt, t, source_embeds)
  noise_pred = noise_pred + source_guidance_scale * (noise_pred_cond - noise_pred)
z, corrected = compute_noise(scheduler, xtm1, xt, t, noise_pred, eta=1.0)
zs[idx] = z
xts[idx] = corrected
```

Edit loop:

```text
latent_model_input = cat([latents] * (1 + num_edit_prompts))
latent_model_input = scheduler.scale_model_input(latent_model_input, t)
noise_pred = UNet(latent_model_input, t, edit/negative conditioning)
noise_pred_uncond, noise_pred_edit... = chunk(noise_pred)
noise_guidance = sum(masked scale_c * sign_c * (edit_c - uncond))
noise_pred = uncond + noise_guidance
latents = scheduler.step(noise_pred, t, latents, variance_noise=zs[idx]).prev_sample
```

Initial Dinoml staging should keep the inversion/edit step loop in host-visible
control. The scheduler tables, `zs`, model-output history, and step indices
must be artifact/runtime-visible because LEDITS++ depends on replaying the
inversion noise, not fresh random variance noise.

## 9. Position, timestep, and custom math

Custom LEDITS++ math:

```text
edit_delta = noise_pred_edit - noise_pred_uncond
if reverse: edit_delta = -edit_delta
edit_delta = edit_guidance_scale * edit_delta
noise_pred = noise_pred_uncond + sum(masked edit_delta)
```

Mask from edit delta:

```text
q = repeat(sum(abs(edit_delta), dim=channel, keepdim=True), channels)
thr = quantile(flatten_spatial(q), edit_threshold)
mask = where(q >= thr[..., None, None], 1, 0)
```

Cross-attention mask:

```text
attn = aggregate down/up cross-attention maps at latent/4 resolution
attn = sum over selected edit-token columns
attn = reflect_pad_1 + conv2d fixed Gaussian 3x3
thr = quantile(flatten(attn), edit_threshold)
attn_mask = where(attn >= thr, 1, 0)
attn_mask = interpolate(attn_mask, latent H/W).repeat(channel)
```

Inversion scheduler custom functions:

- `compute_noise_ddim` mirrors DDIM equations and solves for the variance noise
  that maps `mu_xt` to known `prev_latents`.
- `compute_noise_sde_dpm_pp_2nd` mirrors DPM-Solver SDE first/second-order
  update state and solves for `noise`, then increments scheduler step state.

Precomputable:

- Fixed Gaussian smoothing kernel.
- Text embeddings, pooled text embeddings, and added time IDs.
- Scheduler alpha/sigma/lambda tables for the chosen step count.

Dynamic:

- Quantile thresholds, masks, and semantic guidance tensors depend on each
  denoising step and edit prompt.
- `skip`, `num_inversion_steps`, and SDXL `num_zero_noise_steps` change the
  stored inversion trajectory.

## 10. Preprocessing and input packing

Image preprocessing:

- `VaeImageProcessor.preprocess` handles image input, optional `height`,
  `width`, resize mode, and crop coordinates.
- Both pipelines require height and width divisible by 32 after preprocessing.
- Encoded latents use `latent_dist.mode()` and are multiplied by VAE
  `scaling_factor`.
- The inversion output returns both resized input images and VAE
  reconstruction images.

Prompt preprocessing:

- SD1 negative prompt uses empty string if omitted; edit prompts can be a list
  of multiple simultaneous edits.
- SD1 edit prompt tokenization repeats each edit prompt for each inverted image
  and records `num_edit_tokens = tokenized_length - 2`.
- SDXL negative prompts go through one or two tokenizers/text encoders, concat
  hidden states on the feature axis, and produce pooled embeddings from the
  final text encoder.
- SDXL can zero negative embeddings when `force_zeros_for_empty_prompt` applies.
- Textual inversion conversion can mutate token sequences before tokenization.

Packing:

- CFG/edit batching is on batch dimension, not sequence dimension.
- SDXL conditioning packs edit concepts by concatenating prompt embeddings,
  pooled embeddings, and repeated time IDs along batch.
- User masks are expected to be broadcast-compatible with latent maps; first
  Dinoml admission should require explicit NCHW `[B,1,H,W]` or `[B,4,H,W]`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: inversion trajectory as explicit runtime state

Source pattern:

```text
pipeline mutates self.init_latents, self.zs, self.inversion_steps
```

Replacement:

```text
InversionArtifact { init_latents, zs, timesteps, batch_size, latent_shape, sdxl_size }
```

Preconditions: scheduler family and config match between inversion and edit;
latent shape and batch size match edit call; `num_zero_noise_steps` represented
for SDXL.

Failure cases: changing scheduler object, changing image size, changing batch
size, or replaying SDXL edit without stored `size`.

Parity test: invert once, serialize explicit state, edit from state, compare to
Diffusers pipeline-state edit.

### Rewrite: edit-guidance mask as fused reduction plus pointwise

Source pattern:

```text
abs -> sum(channel) -> repeat(channel) -> quantile(spatial) -> where -> multiply
```

Replacement: one reduction/selection kernel to compute threshold plus one
pointwise mask/multiply kernel.

Preconditions: fixed NCHW source axis or explicit NHWC axis rewrite; `edit_threshold`
scalar per edit; quantile parity defined for fp16 via fp32 upcast.

Failure cases: arbitrary user mask dtype/shape, non-contiguous layout,
backend quantile differences.

Parity test: random edit deltas across SD1 and SDXL latent sizes, thresholds
0.5/0.8/0.9, fp32 and fp16.

### Rewrite: cross-attention mask capture as optional debug/output path

Source pattern:

```text
custom attention processor materializes attention_probs and stores selected maps
```

Replacement: attention provider mode with optional probability capture for named
layers, or eager fallback for mask-enabled edits.

Preconditions: selected `attn2` layer names, edit prompt count, batch splitting,
and resolution filters are explicit.

Failure cases: flash provider cannot return probabilities; IP-Adapter/custom
processor changes K/V shape; prompt token count mismatch.

Parity test: compare aggregate attention maps and final masks for one down and
one up cross-attention block.

### Rewrite: SDXL added time IDs precompute

Source pattern:

```text
original_size + crop_top_left + target_size -> tensor -> add_embedding with pooled text
```

Replacement: precompute IDs and projected embeddings for fixed edit image size.

Preconditions: image size and crop coordinates fixed from inversion; base SDXL
six-ID conditioning, not refiner aesthetic-score variant.

Failure cases: mismatched `projection_class_embeddings_input_dim`, missing
`text_encoder_2.projection_dim`, refiner-style five-ID conditioning.

## 12. Kernel fusion candidates

Highest priority:

- DPM-Solver SDE scheduler state plus explicit `variance_noise` replay. This is
  the LEDITS++ correctness hinge.
- VAE encode/decode parity with scaling and SDXL force-upcast behavior.
- Edit guidance reduction/quantile/mask/multiply kernels.
- Eager attention probability capture for optional cross-attention masks.
- Base SD1/SDXL UNet Conv2d/GroupNorm/SiLU and attention providers reused from
  the base reports.

Medium priority:

- `compute_noise_ddim` as a DDIM inversion helper.
- Fused Gaussian smoothing and thresholding for attention masks.
- Guidance rescale fused std reduction over all non-batch dimensions.
- SDXL `text_time` conditioning precompute and validation.

Lower priority:

- Flash/SDPA attention for no-mask LEDITS++ edit/reconstruction.
- IP-Adapter image embeddings in SDXL LEDITS++.
- LoRA/textual-inversion runtime mutation after explicit adapter state exists.

## 13. Runtime staging plan

Stage 1: Parse SD1 and SDXL component configs and admit LEDITS++ only with
explicit `DDIMScheduler` or DPM-Solver SDE order-2 scheduler.

Stage 2: Implement VAE encode/decode and inversion-state schema. Accept external
prompt embeddings to avoid compiling CLIP initially.

Stage 3: Run inversion parity with fixed UNet outputs or a tiny Diffusers UNet:
`add_noise`, `compute_noise`, `zs`, `init_latents`, and VAE reconstruction.

Stage 4: Add one edit step parity without cross-attention mask:
UNet output chunks, edit guidance, optional user mask, `variance_noise` scheduler
step.

Stage 5: Full Python-controlled edit loop using compiled UNet/VAE blocks and
host-visible scheduler state.

Stage 6: Add SDXL prompt packing: dual embeddings, pooled embeddings, time IDs,
force-upcast VAE, and `num_zero_noise_steps`.

Stage 7: Add cross-attention mask capture through eager fallback first, then a
provider mode if attention probabilities can be exposed.

Stage 8: Optimize quantile masks, guidance rescale, scheduler arithmetic, and
guarded NHWC conv islands.

## 14. Parity and validation plan

- Config parse tests for SD1, SDXL base, refiner, turbo, and tiny SDXL; verify
  constructor scheduler conversion behavior.
- VAE encode mode and reconstruction parity for SD1 and SDXL force-upcast.
- DDIM and DPM-Solver `set_timesteps` parity with `skip` and leading spacing.
- `compute_noise_ddim` and `compute_noise_sde_dpm_pp_2nd` random tensor parity.
- Inversion parity for tiny SD and tiny SDXL fixtures, matching stored
  `init_latents` and `zs`.
- Prompt packing parity for SD1 multi-edit prompts and SDXL dual encoder
  negative/edit embeddings.
- One edit step parity without masks, with `sem_guidance`, and with user mask.
- Quantile mask parity for fp32/fp16 and thresholds `[0.5, 0.75, 0.9]`.
- Cross-attention mask parity on a tiny UNet with one down/up cross-attention
  layer.
- Full short edit loop parity with fixed generator and output latents.
- Suggested tolerances: scheduler arithmetic fp32 `rtol=1e-5, atol=1e-6`;
  UNet/VAE fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- VAE encode and decode throughput by image size and dtype.
- Inversion loop time split: UNet calls, scheduler `compute_noise`, and
  `add_noise` trajectory setup.
- Edit loop time by number of edit prompts E; UNet batch is `(1+E)B`.
- Mask overhead: no mask vs user mask vs quantile mask vs cross-attention mask.
- Attention backend comparison for no-mask edits: eager, SDPA, and future
  Dinoml flash-style provider.
- Attention probability capture overhead for mask-enabled edits.
- Scheduler/guidance arithmetic overhead compared with one UNet step.
- VRAM for storing `xts`, `zs`, `sem_guidance`, and `activation_mask` at
  512/768/1024 resolutions.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items, not ignored:

- `ledits_pp_scheduler_inversion`: DDIM and DPM-Solver SDE inversion helpers,
  variance-noise replay, and explicit inversion state.
- `ledits_pp_attention_masks`: attention probability capture, aggregation,
  smoothing, quantile masks, and visualization storage.
- `sd1_lora_textual_inversion_adapters` and SDXL equivalent: adapter/token
  mutation before inversion and edit.
- `sd1_ip_adapter` / `sdxl_ip_adapter`: especially the SDXL
  `ip_adapter_image` branch in this source.
- SD/SDXL ControlNet and T2I-Adapter: no LEDITS++ class here, but possible
  future composition with base reports.
- SD/SDXL inpaint/depth/upscale: separate variant conditioning; LEDITS++ does
  not add their widened-channel contracts in this folder.
- `scheduler_inverse_editing`: broader inverse/edit scheduler matrix beyond the
  two source-supported schedulers.

Ignored/out of scope for this audit:

- Deprecated `semantic_stable_diffusion`.
- Safety checker, NSFW filtering, and SDXL watermarking.
- Callback mutation and interactive interrupt behavior.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Multi-GPU/context parallel and offload implementation details.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse LEDITS++ SD1 and SDXL component configs.
- [ ] Admit only DDIM or DPM-Solver SDE order-2 schedulers for first parity.
- [ ] Represent inversion state explicitly: `init_latents`, `zs`, timesteps,
      batch size, latent shape, and SDXL size/crop metadata.
- [ ] Implement VAE encode mode and decode with scaling factor.
- [ ] Implement SDXL force-upcast VAE behavior.
- [ ] Accept external negative/source/edit prompt embeddings.
- [ ] Implement SD1 and SDXL prompt packing for multiple edit prompts.
- [ ] Implement SDXL pooled embeddings and six-ID `text_time` conditioning.
- [ ] Implement DDIM `compute_noise` inversion helper.
- [ ] Implement DPM-Solver `sde-dpmsolver++` `compute_noise` helper.
- [ ] Implement scheduler `step(..., variance_noise=zs[idx])` replay.
- [ ] Implement edit guidance arithmetic and reverse direction.
- [ ] Implement `sem_guidance` override.
- [ ] Implement user-mask multiplication with explicit NCHW shape admission.
- [ ] Implement quantile activation masks.
- [ ] Add eager attention-probability capture for cross-attention masks.
- [ ] Implement Gaussian smoothing, attention-mask interpolation, and
      intersection masks.
- [ ] Add inversion parity tests for tiny SD and tiny SDXL.
- [ ] Add one edit-step and short-loop parity tests.
- [ ] Add no-mask attention-provider optimization only after eager parity.
- [ ] Benchmark inversion, edit loop, mask overhead, VAE encode/decode, and
      stored-state memory.
