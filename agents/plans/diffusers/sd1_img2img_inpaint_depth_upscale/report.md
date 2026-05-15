# SD1 Variant Pipeline Candidate Family Audit: img2img, inpaint, depth2img, upscale, latent upscale

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.
  Remote upstream present: https://github.com/huggingface/diffusers.git

Model id(s):
  Base reference: stable-diffusion-v1-5/stable-diffusion-v1-5
  img2img cache example: radames/stable-diffusion-v1-5-img2img
  inpaint representative: stable-diffusion-v1-5/stable-diffusion-inpainting
  depth2img representative in source docs: stabilityai/stable-diffusion-2-depth
  depth2img config source for this audit: sd2-community/stable-diffusion-2-depth
  image upscale representative: stabilityai/stable-diffusion-x4-upscaler
  latent upscale representative: stabilityai/sd-x2-latent-upscaler

Config sources:
  Local: H:/configs/radames/stable-diffusion-v1-5-img2img/model_index.json
  Local: H:/configs/stable-diffusion-v1-5/stable-diffusion-inpainting/model_index.json
  Raw official fetch: stable-diffusion-v1-5/stable-diffusion-inpainting/{model_index.json,unet/config.json,vae/config.json,scheduler/scheduler_config.json,text_encoder/config.json}
  Local/raw official fetch: stabilityai/stable-diffusion-x4-upscaler/{model_index.json,unet/config.json,vae/config.json,scheduler/scheduler_config.json,text_encoder/config.json,low_res_scheduler/scheduler_config.json}
  Raw official fetch: stabilityai/sd-x2-latent-upscaler/{model_index.json,unet/config.json,vae/config.json,scheduler/scheduler_config.json,text_encoder/config.json}
  Local/raw sd2-community fetch: sd2-community/stable-diffusion-2-depth/{model_index.json,unet/config.json,vae/config.json,scheduler/scheduler_config.json,text_encoder/config.json,feature_extractor/preprocessor_config.json,depth_estimator/config.json}

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_latent_upscale.py

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
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/schedulers/scheduling_pndm.py
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_lms_discrete.py

External component configs inspected:
  CLIPTextModel configs for SD1.5 inpaint, SD2 depth, x4 upscaler, and latent upscaler.
  DPTForDepthEstimation and DPTImageProcessor configs from sd2-community/stable-diffusion-2-depth.

Any missing files or assumptions:
  SD2.x is a special case because the original StabilityAI SD2 repos were removed.
  Per project steering, use sd2-community/stable-diffusion-2-depth as the SD2
  depth config source rather than treating it as an access fallback.
  Safety/NSFW, callbacks/interrupt, training/loss/dropout/gradient checkpointing, multi-GPU/context parallel,
  XLA/NPU/MPS/Flax/ONNX are out of this audit except where source structure is shared.
```

## 2. Pipeline and component graph

These variants are not new denoiser families in the simple case; they add input-conditioning and timestep-entry contracts around the SD latent denoising loop. Inpaint, depth, x4 upscale, and latent upscale change the UNet channel or conditioning contract enough that each deserves its own deeper report before implementation.

```text
prompt + optional negative prompt
  -> CLIP tokenizer/text encoder, or external prompt_embeds
  -> variant image/mask/depth/low-res preprocessing
  -> VAE encode when the input is RGB image-conditioned
  -> strength/noise/timestep preparation
  -> denoising loop: UNet2DConditionModel + CFG + scheduler
  -> VAE decode and image postprocess
```

Components by variant:

| Variant | Class | Required components beyond base SD1 text2img | Cacheable or precomputable boundaries |
| --- | --- | --- | --- |
| img2img | `StableDiffusionImg2ImgPipeline` | input image, VAE encode, `strength` timestep slicing, `scheduler.add_noise` | prompt embeds, encoded/scaled image latents, timesteps for strength |
| inpaint | `StableDiffusionInpaintPipeline` | input image, mask image, mask processor, masked-image latents, optional 9-channel inpaint UNet | prompt embeds, image latents, mask latents, masked image latents |
| depth2img | `StableDiffusionDepth2ImgPipeline` | depth map or DPT depth estimator + image processor, VAE encode, 5-channel UNet input | prompt embeds, supplied depth map, normalized latent-resolution depth mask, image latents |
| x4 upscale | `StableDiffusionUpscalePipeline` | low-res RGB image conditioning, `low_res_scheduler`, noise-level class labels, 7-channel UNet input | prompt embeds, noised low-res image condition for fixed noise level |
| latent upscale | `StableDiffusionLatentUpscalePipeline` | low-res latent or RGB image to encode, pooled prompt embeds, K-style UNet, Euler scheduler, 8-channel input, 5-channel output with dropped variance channel | prompt embeds, pooled prompt embeds, low-res latent, upsampled latent condition |

Separate candidate reports:

| Candidate slug | Class/file anchors | Why separate |
| --- | --- | --- |
| `sd1_img2img` | `pipeline_stable_diffusion_img2img.py`, `StableDiffusionImg2ImgPipeline` | Smallest variant delta from base SD1.5; mainly VAE encode plus strength/timestep entry. |
| `sd1_inpaint` | `pipeline_stable_diffusion_inpaint.py`, `StableDiffusionInpaintPipeline` | 9-channel inpaint checkpoints and legacy 4-channel inpaint blend path need distinct tensor tests. |
| `sd_depth2img` | `pipeline_stable_diffusion_depth2img.py`, `StableDiffusionDepth2ImgPipeline` | Actually SD2-depth in representative configs: CLIP width 1024, UNet `in_channels=5`, depth estimator preprocessing. |
| `sd_x4_upscale` | `pipeline_stable_diffusion_upscale.py`, `StableDiffusionUpscalePipeline` | SD2-style text encoder width 1024, `in_channels=7`, class-label noise levels, VAE scaling factor 0.08333. |
| `sd_latent_upscale` | `pipeline_stable_diffusion_latent_upscale.py`, `StableDiffusionLatentUpscalePipeline` | K-blocks, Fourier timesteps, `time_cond_proj_dim=896`, original-sample Euler scheduler math, `out_channels=5`. |

## 3. Important config dimensions

| Representative | Source | Pipeline class | UNet input/output | UNet channels | cross dim | text hidden | scheduler | Prediction |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| SD1.5 base | prior baseline | `StableDiffusionPipeline` | 4 -> 4 | 320/640/1280/1280 | 768 | 768 | PNDM | epsilon effective |
| SD1.5 img2img cache | local model_index only | `StableDiffusionPipeline` in cached artifact; source target class is `StableDiffusionImg2ImgPipeline` | inferred base 4 -> 4 | inferred base | inferred 768 | inferred 768 | PNDM | inferred epsilon |
| SD1.5 inpaint | official configs | `StableDiffusionInpaintPipeline` | 9 -> 4 | 320/640/1280/1280 | 768 | 768 | DDIM | epsilon default |
| SD2 depth | sd2-community configs | `StableDiffusionDepth2ImgPipeline` | 5 -> 4 | 320/640/1280/1280 | 1024 | 1024 | PNDM | epsilon |
| SD x4 upscale | official configs | `StableDiffusionUpscalePipeline` | 7 -> 4 | 256/512/512/1024 | 1024 | 1024 | DDIM + low-res DDPM | v_prediction for denoiser, epsilon for low-res noise |
| SD x2 latent upscale | official configs | `StableDiffusionLatentUpscalePipeline` | 8 -> 5 | 384/384/768/768 | 768 | 768 | EulerDiscrete | original_sample |

Other operator-significant config facts:

| Variant | VAE | Attention / block differences | Conditioning dimensions |
| --- | --- | --- | --- |
| img2img | SD1 VAE, latent channels 4, scaling default 0.18215 if omitted | Same UNet as base | no extra UNet channels; initial latents come from encoded image plus noise |
| inpaint | official VAE omits scaling factor, effective source default 0.18215; `sample_size=256` in config but pipeline uses requested image size | Same SD1 blocks; `conv_in` is `Conv2d(9 -> 320, 3x3)` | concat `[latents(4), mask(1), masked_image_latents(4)]` over channel dim |
| depth2img | sd2-community VAE omits scaling factor, effective source default 0.18215 | SD2 uses `use_linear_projection=true`, attention head dims `[5,10,20,20]`, CLIP width 1024 | concat `[latents(4), depth_mask(1)]` |
| x4 upscale | VAE has 3 down/up blocks and `scaling_factor=0.08333`; pipeline warns older configs may need this repair | `conv_in` `7 -> 256`; `num_class_embeds=1000`; `only_cross_attention=[true,true,true,false]`; `use_linear_projection=true` | concat `[latents(4), noised_low_res_rgb(3)]`; class labels are integer `noise_level` |
| latent upscale | VAE SD1-like with `scaling_factor=0.18215`; can encode RGB input or accept 4-channel latent input | KDown/KCrossAttn blocks, `conv_in_kernel=1`, `conv_out_kernel=1`, `resnet_time_scale_shift=scale_shift`, `time_embedding_type=fourier`, `norm_num_groups=null` | concat `[latents(4), upsampled_low_res_latent_cond(4)]`; timestep condition is `[128 noise embedding + pooled prompt 768] = 896` |

Recommended first scheduler slice:

| Slice | Reason |
| --- | --- |
| img2img/inpaint first | Reuse base DDIM/PNDM/Euler machinery plus `add_noise` and `strength`. |
| x4 upscale second | Adds low-res DDPM `add_noise`, class labels, and v-prediction scheduler parity. |
| latent upscale separate | Euler original-sample plus Karras preconditioning in the pipeline is a different denoising contract. |

## 3a. Family variation traps

- "Stable diffusion variant" does not mean SD1-compatible dimensions. Depth and x4 upscaler representatives are SD2-style with CLIP width/cross-attention dim 1024. Latent upscaler returns a 5-channel UNet output and discards the last variance-like channel.
- Inpaint has two source paths: 9-channel inpaint checkpoints concatenate mask and masked latents into the UNet input, while 4-channel checkpoints use a base UNet and blend masked/unmasked latents after each scheduler step.
- `strength` changes both the initial noising and the number of denoising steps. The scheduler's `order` participates in slicing through `t_start * scheduler.order`.
- Source latent/image tensors are NCHW. Candidate NHWC/channel-last lowering must rewrite concat dim 1 to last channel and protect scheduler broadcasting, mask interpolation, depth reductions, and VAE scale/divide with layout guards.
- Inpaint mask preprocessing uses grayscale conversion and binarization; mask semantic is white/repaint and black/preserve.
- Depth2img normalizes each depth map by per-sample min/max over `[1,2,3]` after interpolation. A zero depth range would need parity with PyTorch behavior.
- x4 upscale uses an RGB low-res condition at latent spatial size, not VAE-encoded low-res latents. Latent upscale uses latent-space conditioning and doubles latent H/W.
- Latent upscale uses pooled CLIP text output in addition to sequence prompt embeds. Base SD1.5 first-slice prompt embedding APIs that only expose `[B,77,768]` are insufficient for this variant.
- The shared UNet class has many inactive branches; only count `class_labels` for x4 upscale and `timestep_cond` for latent upscale here.

## 4. Runtime tensor contract

For 512x512 SD1 img2img/inpaint examples, latent size is 64x64. For SD2 depth `sample_size=32` means the canonical checkpoint was 512-ish SD2 latent sizing but image size remains runtime-controlled. For x4 upscale, a 128x128 low-res image leads to a 128x128 latent/noised RGB condition and a 512x512 decoded output because the VAE scale factor is 4 from its 3 block levels.

| Variant | Pipeline inputs after CPU preprocessing | Denoiser input | Denoiser output | Decode input |
| --- | --- | --- | --- | --- |
| img2img | image tensor `[B,3,H,W]` normalized to `[-1,1]`; VAE encode -> `[B,4,H/8,W/8] * scaling_factor` | `[2B,4,H/8,W/8]` for CFG | `[2B,4,H/8,W/8]`, CFG -> `[B,4,H/8,W/8]` | final latents / scaling factor |
| inpaint 9ch | image `[B,3,H,W]`, mask `[B,1,H,W]`, masked image encode `[B,4,H/8,W/8]`, mask interpolate `[B,1,H/8,W/8]` | concat over C: `[2B,9,H/8,W/8]` | `[2B,4,H/8,W/8]` | final latents / scaling factor |
| inpaint 4ch | same as inpaint plus original image latents retained | `[2B,4,H/8,W/8]` | `[2B,4,H/8,W/8]`; after scheduler step, blend `(1-mask)*init_latents_proper + mask*latents` | blended latents / scaling factor |
| depth2img | image encode latents; depth estimator or supplied depth -> normalized depth `[B,1,H/8,W/8]` | concat over C: `[2B,5,H/8,W/8]` | `[2B,4,H/8,W/8]` | final latents / scaling factor |
| x4 upscale | low-res image `[B,3,h,w]` normalized, DDPM-noised at `noise_level`; no VAE encode for condition | concat `[2B,7,h,w]` | `[2B,4,h,w]` | latents `[B,4,h,w] / 0.08333`, VAE decode gives roughly `4h x 4w` |
| latent upscale | image RGB encode to latent `[B,4,h/8,w/8]` or accept latent `[B,4,h,w]`; condition nearest-upsampled by 2 | concat `[2B,8,2h,2w]` | `[2B,5,2h,2w]`, drop last channel before CFG/scheduler | latents `[B,4,2h,2w] / 0.18215` |

Scheduler/control tensors:

- `timesteps`: scheduler-owned int/float tensor on target device after `set_timesteps`.
- `latent_timestep`: first selected timestep repeated to batch for img2img/inpaint/depth initial `add_noise`.
- `noise`: random tensor same shape as encoded image latents or low-res image condition.
- `noise_level`: x4 upscale integer class labels repeated to CFG batch; latent upscale float zero-level embedding plus pooled prompt forms `timestep_cond`.
- CFG prompt embeddings are concatenated over batch; all mask/depth/condition tensors are also duplicated over batch before UNet when CFG is active.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW source handling, channel concat for inpaint/depth/upscale, batch concat/chunk for CFG.
- `repeat`, `cat`, `chunk`, `interpolate`, `amin`, `amax`, scalar broadcasting over latent maps.
- Guarded NHWC rewrite candidates must rewrite `dim=1` channel concat/reductions to last channel and preserve scheduler scalar broadcasts.

Convolution/downsample/upsample:

- Base/img2img: SD1 UNet `Conv2d(4 -> 320, 3x3, padding=1)`.
- Inpaint: `Conv2d(9 -> 320, 3x3, padding=1)`.
- SD2 depth: `Conv2d(5 -> 320, 3x3, padding=1)`.
- x4 upscale: `Conv2d(7 -> 256, 3x3, padding=1)`, VAE with scale factor 4.
- Latent upscale: `Conv2d(8 -> 384, 1x1)` and `Conv2d(384 -> 5, 1x1)`, KDown/KUp resize blocks.
- AutoencoderKL encode and decode are required for img2img/inpaint/depth/latent-upscale RGB input; x4 only decodes generated latents.

GEMM/linear:

- CLIP text encoders are external in the first slice; latent upscale additionally needs pooled prompt embeddings.
- UNet time embedding MLP; x4 `num_class_embeds=1000`; latent upscale Fourier timestep embedding plus `timestep_cond` projection.
- Cross-attention Q/K/V/out projections and GEGLU/GELU feed-forward paths.

Attention:

- Base SD1-style self/cross attention for img2img/inpaint.
- SD2 depth/x4 use cross-attention dim 1024 and `use_linear_projection=true`.
- Latent upscale KAttentionBlock uses K-block conditioned attention with `cross_attention_norm="layer_norm"` and `attention_head_dim=64`.

Scheduler/guidance arithmetic:

- `scheduler.add_noise` for img2img/inpaint/depth initial latents and x4 low-res image noise.
- DDIM/PNDM/Euler `scale_model_input` and `step`.
- CFG `uncond + guidance_scale * (text - uncond)`.
- Latent upscale Karras-style preconditioning: `timestep = log(sigma) * 0.25`; `inv_sigma = 1/(sigma^2+1)`; recombine model output before CFG.

VAE/postprocessing:

- `vae.encode(...).latent_dist.sample()` or mode/latents retrieval, multiply by scaling factor.
- `vae.decode(latents / scaling_factor)`.
- `VaeImageProcessor` normalize to `[-1,1]`, denormalize to `[0,1]`, mask binarize/grayscale.

## 6. Denoiser/model breakdown

img2img uses the base SD1.5 denoiser unchanged. The forward path is still:

```text
latents -> conv_in -> CrossAttnDownBlock2D/DownBlock2D -> mid cross-attn block
        -> UpBlock2D/CrossAttnUpBlock2D -> GroupNorm/SiLU/conv_out -> noise_pred
```

Inpaint 9-channel changes only the first UNet channel contract:

```text
latent_model_input = scheduler.scale_model_input(cat_batch(latents), t)
latent_model_input = cat_channel(latent_model_input, mask, masked_image_latents)
UNet2DConditionModel(in_channels=9, out_channels=4)
```

Depth2img changes the first channel contract and text width:

```text
depth_mask = bicubic(depth_estimator(image) or supplied_depth) to latent H/W
depth_mask = per-sample normalize to [-1,1]
latent_model_input = cat_channel(scaled_latents, depth_mask)
UNet2DConditionModel(in_channels=5, cross_attention_dim=1024)
```

x4 upscale adds class embedding and low-res RGB condition:

```text
image = VaeImageProcessor.preprocess(low_res_rgb)
image = low_res_scheduler.add_noise(image, noise, noise_level)
latent_model_input = cat_channel(scheduler.scale_model_input(latents, t), image)
UNet2DConditionModel(..., class_labels=noise_level)
```

Latent upscale uses K-blocks:

```text
image = latent input or VAE-encoded RGB
image_cond = nearest_upsample_2x(image) * inv_noise_level
timestep_condition = cat(noise_level_embed[128], pooled_prompt_embeds[768])
scaled_model_input = scheduler.scale_model_input(latents, t)
UNet2DConditionModel(KDown/KCrossAttn/KUp blocks, timestep_cond=timestep_condition)
drop output channel 4, Karras precondition, then CFG and Euler step
```

K-block source evidence: `KDownBlock2D` and `KCrossAttnDownBlock2D` build `ResnetBlockCondNorm2D` with `time_embedding_norm="ada_group"` and group counts computed as `channels // resnet_group_size`; K cross-attention uses `KAttentionBlock`.

## 7. Attention requirements

- Base/img2img/inpaint first parity can reuse SD1 attention from the base report: noncausal latent self/cross attention, CLIP text K/V length 77, cross dim 768.
- Depth/x4 need cross dim 1024 and CLIP hidden size 1024. Attention head dimensions are list-valued for SD2 depth `[5,10,20,20]` and scalar 8 for x4.
- x4 has `only_cross_attention=[true,true,true,false]`; do not assume all blocks include self-attention.
- Latent upscale uses `attention_head_dim=64`, K blocks, layer-normalized cross attention, and `cross_attention_dim=768`.
- `attention_processor.py` remains the parity anchor for eager/SDPA-style processors. Fused projections are optimization candidates only.
- Dinoml flash-style providers are plausible for dense, mask-free self/cross attention after proving processor semantics. Added K/V IP-Adapter paths are not part of this candidate.

## 8. Scheduler and denoising-loop contract

Common img2img/inpaint/depth strength handling:

```text
scheduler.set_timesteps(num_inference_steps, device)
init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
t_start = max(num_inference_steps - init_timestep, 0)
timesteps = scheduler.timesteps[t_start * scheduler.order:]
scheduler.set_begin_index(t_start * scheduler.order) when supported
latent_timestep = timesteps[:1].repeat(batch)
latents = scheduler.add_noise(encoded_image_latents, noise, latent_timestep)
```

Inpaint special cases:

- `strength == 1.0` can initialize pure noise scaled by `scheduler.init_noise_sigma`.
- 4-channel inpaint path re-noises original image latents for the next timestep and blends with the mask after every scheduler step.
- Constructor compatibility repairs set stale `steps_offset` to 1 and `skip_prk_steps` to true for old scheduler configs.

x4 upscale:

- Main scheduler default is DDIM with `prediction_type=v_prediction`.
- Low-res condition scheduler is DDPMScheduler with epsilon prediction and `clip_sample=true`; it only adds noise to the RGB condition at integer `noise_level`.
- UNet receives `class_labels=noise_level`, so class embedding is required.

Latent upscale:

- Scheduler is EulerDiscreteScheduler with `prediction_type=original_sample`.
- Pipeline does Karras-style log-sigma timestep conversion before the UNet and a custom recombination before CFG.
- This should remain a separate scheduler/runtime slice; it is not just the base Euler loop.

## 9. Position, timestep, and custom math

Required math beyond base SD1:

```python
# depth2img per-sample depth normalization after latent-resolution interpolation
depth = 2.0 * (depth - amin(depth, [1, 2, 3])) / (amax(depth, [1, 2, 3]) - amin(depth, [1, 2, 3])) - 1.0

# latent upscale Karras-style loop-side math
timestep = torch.log(sigma) * 0.25
inv_sigma = 1 / (sigma**2 + 1)
noise_pred = inv_sigma * latent_model_input + scheduler.scale_model_input(sigma, t) * noise_pred
```

Precompute:

- Prompt embeddings and negative prompt embeddings for all variants.
- Latent upscale pooled prompt embeddings and the constant zero-noise embedding pattern.
- Depth map when supplied or estimated once per input image.
- Mask and masked image latents for repeated inpaint attempts with the same image/mask.

Dynamic:

- `strength` timestep slicing, initial noise, scheduler state, and x4 `noise_level`.
- Image/mask/depth resize/interpolation depends on input H/W.

## 10. Preprocessing and input packing

img2img:

- `VaeImageProcessor` preprocesses image to normalized NCHW tensor.
- If image has 4 channels, it is treated as latents and not VAE-encoded; otherwise VAE encode and multiply by scaling factor.
- Add noise at the selected first timestep.

inpaint:

- Image processor normalizes RGB image.
- Mask processor uses `do_normalize=False`, `do_binarize=True`, `do_convert_grayscale=True`.
- Optional `padding_mask_crop` computes a crop region and uses fill resize; otherwise default resize.
- `masked_image = init_image * (mask_condition < 0.5)` when masked latents are not provided.
- For 9-channel UNet, mask and masked-image latents are concatenated with scaled latents over channels.

depth2img:

- If `depth_map` is absent, DPTImageProcessor prepares 384x384 normalized inputs for DPTForDepthEstimation.
- Predicted or supplied depth is interpolated bicubic to latent size and normalized to `[-1,1]`.
- Depth mask duplicates to prompt batch and CFG batch.

x4 upscale:

- Low-res image is resized/normalized by `VaeImageProcessor(..., resample="bicubic")`.
- DDPM noise is added in pixel/RGB conditioning space, not latent space.
- Condition is concatenated with latent denoiser input over channels.

latent upscale:

- RGB input is VAE-encoded to latent; 4-channel input is accepted as latent.
- Low-res latent condition is nearest-neighbor upsampled by 2.
- Sequence prompt embeds and pooled prompt embeds are both required.

## 11. Graph rewrite / lowering opportunities

Rewrite: variant channel-concat into widened `conv_in`.

- Source pattern: `cat_channel(latents, condition...) -> UNet.conv_in`.
- Replacement: preserve explicit concat initially; later fuse concat into `conv_in` by slicing input-channel weights per source tensor.
- Preconditions: all condition tensors have same B/H/W and source NCHW channel order; no aliasing mutation; CFG duplication already resolved.
- Shape equations:
  - inpaint: `C = 4 + 1 + 4 = 9`.
  - depth: `C = 4 + 1 = 5`.
  - x4: `C = 4 + 3 = 7`.
  - latent upscale: `C = 4 + 4 = 8`.
- Weight transform: split `conv_in.weight[:, channel_ranges, kh, kw]` by input source; compute separate convs and sum, or pack conditions into a fused input tile.
- Layout constraints: NHWC lowering rewrites channel concat from dim 1 to dim -1 and OIHW weights to HWIO.
- Failure cases: mismatched condition H/W, 4-channel legacy inpaint path, latent upscale `1x1` conv differs from SD conv island assumptions.
- Parity test: compare fused first-conv result against explicit concat for each C count.

Rewrite: mask blend fusion for 4-channel inpaint.

- Source pattern: `latents = (1 - mask) * init_latents_proper + mask * latents`.
- Replacement: fused pointwise `lerp(init, latents, mask)`.
- Preconditions: mask in `[0,1]`, broadcast or exact `[B,1,H,W]`, source layout known.
- Failure cases: non-binary soft masks should still be supported as linear interpolation; do not clamp unless source does.
- Parity test: random mask and binary mask tests in fp32/fp16.

Rewrite: depth normalization kernel.

- Source pattern: per-sample `amin/amax` over C/H/W followed by affine normalize.
- Replacement: fused reduction plus pointwise normalize.
- Preconditions: one-channel depth map or reduction over all non-batch dims; handle dtype promotion rules deliberately.
- Failure cases: `depth_max == depth_min` parity with PyTorch must be defined before optimizing.
- Parity test: random depth maps and constant maps.

Rewrite: latent-upscale Karras preconditioning.

- Source pattern: scalar sigma math plus two scheduler scaling calls and pointwise recombination.
- Replacement: fused scalar-table lookup plus pointwise kernel.
- Preconditions: Euler scheduler state exposes `sigma[i]`, prediction type fixed to original-sample contract, UNet output variance channel dropped.
- Failure cases: non-Euler scheduler, different prediction type, keeping variance channel.
- Parity test: one latent-upscale denoising step at fixed sigma.

## 12. Kernel fusion candidates

Highest priority:

- VAE encode path for img2img/inpaint/depth and latent-upscale RGB input. These variants need encode parity, not only decode.
- Widened `conv_in` concat fusion for inpaint/depth/x4/latent-upscale.
- Scheduler `add_noise` and strength slicing parity for img2img/inpaint/depth.
- Mask interpolation/binarization and masked-latent concat for inpaint.
- CFG concat/chunk/arithmetic across all variants.

Medium priority:

- x4 low-res DDPM noise addition plus class-label embedding.
- Depth min/max normalization and bicubic interpolation.
- SD2-style cross-attention dim 1024 and `use_linear_projection=true` for depth/x4.
- Latent-upscale K-block AdaGroupNorm and 1x1 conv-heavy path.

Lower priority:

- Latent-upscale custom Karras preconditioning after base variants are stable.
- Crop/fill overlay behavior for `padding_mask_crop`; useful for UI parity but not denoiser first-slice.

## 13. Runtime staging plan

Stage 1: Implement `sd1_img2img` using supplied prompt embeddings, SD1 VAE encode/decode, base UNet, and one scheduler with `add_noise` plus `strength`.

Stage 2: Implement `sd1_inpaint` 9-channel path with mask processor outputs supplied as tensors first, then add RGB/mask preprocessing. Keep 4-channel legacy inpaint as a separate compatibility sub-slice.

Stage 3: Add depth2img as a deeper report before implementation. First implementation should accept precomputed depth maps and skip DPT; later add DPT as external Transformers component or keep it outside Dinoml runtime.

Stage 4: Add x4 upscaler as a separate deeper report. It is SD2-like and needs CLIP width 1024, v-prediction DDIM, low-res DDPM noise, and class-label conditioning.

Stage 5: Add latent upscaler as a separate deeper report. It needs pooled CLIP output, K-blocks, Fourier timestep embedding, Euler original-sample/preconditioning math, and 5-channel UNet output handling.

Stage 6: Optimize common concat-first-conv, VAE encode/decode, CFG, and scheduler arithmetic kernels.

## 14. Parity and validation plan

- Img2img VAE encode parity: image tensor -> scaled latents, single and batched generator paths.
- `get_timesteps` parity for `strength` values 0.0, 0.25, 0.8, 1.0 across DDIM/PNDM/Euler order behavior.
- `scheduler.add_noise` parity for initial image latents.
- Inpaint mask processor parity: PIL/np/torch masks, grayscale, binarize, resize to latent H/W.
- Inpaint 9-channel one-step parity at fixed timestep and fixed prompt embeddings.
- Inpaint 4-channel blend parity after scheduler step.
- Depth prepare-depth parity with supplied depth maps first; DPT estimator smoke only if Transformers runtime is in scope.
- x4 low-res scheduler noise parity and class-label embedding one-step parity.
- Latent-upscale one-step parity including sigma preconditioning and output channel drop.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=2e-2, atol=2e-2` and tighten per kernel.

## 15. Performance probes

- VAE encode throughput by batch/resolution for img2img/inpaint/depth.
- One denoiser step by variant channel count: C=4,5,7,8,9.
- Explicit concat + conv versus fused condition-first-conv.
- Mask/depth preprocessing GPU cost versus CPU preprocessing cost.
- Scheduler/add-noise overhead by step count and strength.
- x4 upscaler low-res image condition memory footprint at 128, 256, 512 input.
- Latent-upscale K-block step time versus base SD1 UNet step time.
- Full loop memory with CFG batched call versus two-call CFG.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `sd1_img2img`: bounded first variant; source `pipeline_stable_diffusion_img2img.py`.
- `sd1_inpaint`: includes both 9-channel and 4-channel inpaint contracts; source `pipeline_stable_diffusion_inpaint.py`.
- `sd_depth2img`: source `pipeline_stable_diffusion_depth2img.py`; representative configs are SD2-style and depth-estimator-coupled.
- `sd_x4_upscale`: source `pipeline_stable_diffusion_upscale.py`; low-res scheduler and class-label conditioning.
- `sd_latent_upscale`: source `pipeline_stable_diffusion_latent_upscale.py`; K-blocks and custom Euler/preconditioning loop.
- `sd1_lora_textual_inversion_adapters`, `sd1_ip_adapter`, `sd1_controlnet`, `sd1_t2i_adapter`, and `sd1_gligen` remain related but outside this selected variant-family report.

Ignored/out of scope:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker, NSFW filtering, and watermarker.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Add img2img config parsing and source class mapping.
- [ ] Implement VAE encode path and latent scaling parity.
- [ ] Implement `strength` timestep slicing and `scheduler.add_noise`.
- [ ] Add img2img one-step parity with external prompt embeddings.
- [ ] Add inpaint mask preprocessing parity.
- [ ] Implement 9-channel inpaint UNet input packing.
- [ ] Implement 4-channel inpaint mask blend compatibility path.
- [ ] Add inpaint one-step parity for 9-channel checkpoint.
- [ ] Create deeper `sd_depth2img` report before implementation.
- [ ] Create deeper `sd_x4_upscale` report before implementation.
- [ ] Create deeper `sd_latent_upscale` report before implementation.
- [ ] Add concat-first-conv lowering tests for C=5,7,8,9.
- [ ] Benchmark VAE encode, denoiser step, and condition packing separately.
