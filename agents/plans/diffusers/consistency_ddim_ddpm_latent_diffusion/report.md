# Diffusers Classic / Generic Diffusion Pipeline Audit

Candidate slug: `consistency_ddim_ddpm_latent_diffusion`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Pixel DDPM/DDIM examples:
    google/ddpm-cifar10-32
    google/ddpm-cat-256
    google/ddpm-ema-bedroom-256
    fusing/ddim-lsun-bedroom
    fusing/ddim-celeba-hq
  Consistency model examples:
    openai/diffusers-cd_imagenet64_l2
    openai/diffusers-ct_imagenet64
    openai/diffusers-cd_cat256_lpips
  Latent diffusion examples:
    CompVis/ldm-text2im-large-256
    CompVis/ldm-celebahq-256
    CompVis/ldm-super-resolution-4x-openimages

Config sources:
  Local cache model indexes:
    H:/configs/google/ddpm-cifar10-32/model_index.json
    H:/configs/google/ddpm-cat-256/model_index.json
    H:/configs/google/ddpm-ema-bedroom-256/model_index.json
    H:/configs/openai/diffusers-cd_imagenet64_l2/model_index.json
    H:/configs/openai/diffusers-ct_imagenet64/model_index.json
    H:/configs/openai/diffusers-cd_cat256_lpips/model_index.json
    H:/configs/CompVis/ldm-text2im-large-256/model_index.json
    H:/configs/CompVis/ldm-celebahq-256/model_index.json
    H:/configs/CompVis/ldm-super-resolution-4x-openimages/model_index.json
  Official raw Hugging Face configs inspected transiently, not saved because
  this task's owned write path is limited to this report:
    google/ddpm-cifar10-32: config.json, scheduler_config.json
    google/ddpm-cat-256: config.json, scheduler_config.json
    google/ddpm-ema-bedroom-256: config.json, scheduler_config.json
    fusing/ddim-lsun-bedroom: model_index.json, config.json, scheduler_config.json
    fusing/ddim-celeba-hq: model_index.json, config.json, scheduler_config.json
    openai/diffusers-cd_imagenet64_l2: unet/config.json, scheduler/scheduler_config.json
    openai/diffusers-ct_imagenet64: unet/config.json, scheduler/scheduler_config.json
    openai/diffusers-cd_cat256_lpips: unet/config.json, scheduler/scheduler_config.json
    CompVis/ldm-text2im-large-256: bert/config.json, tokenizer/tokenizer_config.json,
      tokenizer/vocab.txt, unet/config.json, vqvae/config.json, scheduler/scheduler_config.json
    CompVis/ldm-celebahq-256: unet/config.json, vqvae/config.json, scheduler/scheduler_config.json
    CompVis/ldm-super-resolution-4x-openimages: unet/config.json, vqvae/config.json,
      scheduler/scheduler_config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
  diffusers/src/diffusers/pipelines/ddim/pipeline_ddim.py
  diffusers/src/diffusers/pipelines/consistency_models/pipeline_consistency_models.py
  diffusers/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py
  diffusers/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion_superresolution.py

Model files inspected:
  diffusers/src/diffusers/models/unets/unet_2d.py
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/downsampling.py
  diffusers/src/diffusers/models/upsampling.py
  diffusers/src/diffusers/models/autoencoders/vq_model.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_consistency_models.py
  diffusers/src/diffusers/schedulers/scheduling_pndm.py
  diffusers/src/diffusers/schedulers/scheduling_lms_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  LDMBert local source/config in pipeline_latent_diffusion.py and
  official raw CompVis/ldm-text2im-large-256 bert/config.json.

Any missing files or assumptions:
  Official raw configs were available without gated blockers. They were not
  retained in H:/configs because this task restricts writes to this report. The
  fusing DDIM repos are old-format artifacts with top-level `UNetModel` /
  `GaussianDDPMScheduler` config names, so they are used only as historical
  DDIM examples. `CompVis/ldm-celebahq-256` advertises deprecated `LDMPipeline`;
  its component configs are still useful for latent-shape variation, but the
  selected non-deprecated source path is `pipelines/latent_diffusion`.
```

## 2. Pipeline and component graph

This target covers classic diffusion pipelines before the Stable Diffusion
loader/adapter ecosystem dominates. There are three first-class contracts:

```text
DDPM/DDIM pixel generation:
  random pixel noise [B,3,H,W]
  -> UNet2DModel(noisy image, timestep)
  -> DDPM or DDIM scheduler step
  -> pixel postprocess to [0,1] NHWC/PIL

Consistency pixel generation:
  random pixel noise [B,3,H,W] * sigma_max
  -> CM scheduler scale_model_input
  -> UNet2DModel(noisy image, continuous timestep, optional class_labels)
  -> CM boundary-condition denoise + optional stochastic re-noise
  -> pixel postprocess to [0,1]

Latent diffusion:
  prompt or low-resolution image preprocessing
  -> LDMBert text encoder or image preprocessing
  -> latent noise [B,C,h,w]
  -> denoising loop: UNet + scheduler + optional CFG
  -> VQModel / AutoencoderKL decode
  -> image postprocess
```

Required components by variant:

| Variant | Required components | Optional/cacheable boundaries |
| --- | --- | --- |
| `DDPMPipeline` | `UNet2DModel`, `DDPMScheduler` | initial noise, scheduler timestep tables |
| `DDIMPipeline` | `UNet2DModel`, coerced `DDIMScheduler.from_config(...)` | eta/noise branch, timestep tables |
| `ConsistencyModelPipeline` | `UNet2DModel`, `CMStochasticIterativeScheduler` | custom timesteps, class labels, initial latents |
| `LDMTextToImagePipeline` | `VQModel` or `AutoencoderKL`, `LDMBertModel`, tokenizer, `UNet2DConditionModel`, DDIM/PNDM/LMS scheduler | prompt embeddings, negative embeddings, scheduler tables, final VAE decode |
| `LDMSuperResolutionPipeline` | `VQModel`, `UNet2DModel`, DDIM/PNDM/LMS/Euler/EulerA/DPM scheduler | preprocessed low-res image, scheduler tables |

Separate candidate reports:

| Surface | Support in this family | Suggested candidate |
| --- | --- | --- |
| LoRA/textual inversion/runtime adapters | Not present in these simple pipelines. LDM text uses fixed LDMBert/tokenizer, not SD mixins. | Keep under SD loader reports. |
| IP-Adapter | Not present. | None for this target. |
| ControlNet / T2I-Adapter / GLIGEN | Not present in selected folders. | SD-family reports only. |
| img2img / inpaint / depth2img | Not present as separate classic pipeline classes here. | SD variant reports. |
| upscaling | Present as `LDMSuperResolutionPipeline`, source `pipeline_latent_diffusion_superresolution.py`; concatenates noisy latent/image channels with low-res image. | `ldm_superresolution` if deeper codec/upscale parity is needed. |
| deprecated unconditional LDM | `CompVis/ldm-celebahq-256` points to deprecated `LDMPipeline`, not the current `pipeline_latent_diffusion.py`. | `latent_diffusion_uncond_deprecated` only if old artifact compatibility matters. |

## 3. Important config dimensions

Representative pixel DDPM/DDIM checkpoints:

| Model | Pipeline/config class | Image/sample | UNet channels | Blocks | Attention | Scheduler | Notes |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `google/ddpm-cifar10-32` | `DDPMPipeline` / `UNet2DModel` | 32 | 3 -> 3 | 128/256/256/256, 2 layers | one `AttnDownBlock2D` and one `AttnUpBlock2D`; `attention_head_dim` omitted, effective source default 8 | `DDPMScheduler`, linear, fixed_large, clip sample | Smallest first pixel slice. |
| `google/ddpm-cat-256` | `DDPMPipeline` / `UNet2DModel` | 256 | 3 -> 3 | 128/128/256/256/512/512, 2 layers | attention at one low-resolution stage | `DDPMScheduler`, linear, fixed_small, clip sample | Production pixel resolution. |
| `google/ddpm-ema-bedroom-256` | `DDPMPipeline` / `UNet2DModel` | 256 | 3 -> 3 | same shape as cat | same | `DDPMScheduler`, linear, fixed_small, clip sample | Same operator shape as cat. |
| `fusing/ddim-lsun-bedroom` | old `DDIM` / `UNetModel` config | old flat config, not current `UNet2DModel` | 3 input | not represented in current component fields | historical | old `GaussianDDPMScheduler`, linear, fixed_small | Use as old-artifact caution, not first slice. |

Representative consistency checkpoints:

| Model | Image/sample | Class labels | UNet channels | Blocks | Attention | Scheduler |
| --- | ---: | --- | --- | --- | --- | --- |
| `openai/diffusers-cd_imagenet64_l2` | 64 | 1000 ImageNet classes | 3 -> 3; 192/384/576/768; 3 layers | `ResnetDownsampleBlock2D`, then 3 attention down blocks; mirrored attention up blocks | `attention_head_dim=64` | `CMStochasticIterativeScheduler`, 40 train steps, sigma 80 -> 0.002 |
| `openai/diffusers-ct_imagenet64` | 64 | 1000 ImageNet classes | same as above | same | same | CM scheduler, 201 train steps |
| `openai/diffusers-cd_cat256_lpips` | 256 | none | 3 -> 3; 256/256/512/512/1024/1024; 2 layers | 3 ResNet-down then 3 attention-down; mirrored | `attention_head_dim=64` | CM scheduler, 40 train steps |

Representative latent checkpoints:

| Model | Pipeline | Denoiser | Codec | Scheduler | Pixel/latent contract |
| --- | --- | --- | --- | --- | --- |
| `CompVis/ldm-text2im-large-256` | `LDMTextToImagePipeline` | `UNet2DConditionModel`, 4 -> 4, sample 32, blocks 320/640/1280/1280, cross attention | `AutoencoderKL`, latent_channels 4, block 128/256/512/512, `scaling_factor` omitted but source default 0.18215 | `DDIMScheduler`, linear, clip_sample false | text prompt -> LDMBert `[B,77,1280]`; latent `[B,4,H/8,W/8]`; decode to RGB |
| `CompVis/ldm-celebahq-256` | deprecated `LDMPipeline` artifact | `UNet2DModel`, 3 -> 3, sample 64, blocks 224/448/672/896 | `VQModel`, latent_channels 3, blocks 128/256/512 | `DDIMScheduler`, scaled_linear, clip_sample false | unconditional latent image; source is deprecated |
| `CompVis/ldm-super-resolution-4x-openimages` | `LDMSuperResolutionPipeline` | `UNet2DModel`, 6 -> 3, sample 64, blocks 160/320/320/640 | `VQModel`, latent_channels 3, blocks 128/256/512 | `DDIMScheduler`, scaled_linear, steps_offset 1, clip_sample false | low-res RGB `[B,3,H,W]` concat noisy latent `[B,3,H,W]` into 6 channels, then VQ decode |

Scheduler set:

| Pipeline | Source accepted scheduler set | Recommended first Dinoml scheduler slice |
| --- | --- | --- |
| DDPM | Annotated `DDPMScheduler`; docs mention DDPM or DDIM | `DDPMScheduler` epsilon, fixed_small/fixed_large, no learned variance first |
| DDIM | Constructor coerces any compatible config through `DDIMScheduler.from_config` | `DDIMScheduler` epsilon, `eta=0`, clip/threshold disabled unless config requires |
| Consistency | Only `CMStochasticIterativeScheduler` | one-step CM (`num_inference_steps=1`) then custom `[22,0]` multistep |
| LDM text | DDIM, PNDM, LMS | DDIM epsilon with CFG and AutoencoderKL decode |
| LDM super-resolution | DDIM, PNDM, LMS, Euler, EulerA, DPM-Solver | DDIM epsilon first; Euler/DPM swaps later |

## 3a. Family variation traps

- Pixel pipelines denoise RGB-shaped samples directly. Their final scheduler
  sample is already an image tensor in `[-1,1]`; there is no VAE decode.
- LDM pipelines denoise latent maps and must decode through `VQModel` or
  `AutoencoderKL`. The codec scale factor matters: LDM text divides latents by
  `vqvae.config.scaling_factor` before decode.
- `LDMSuperResolutionPipeline` is named latent diffusion but its runtime
  denoiser input is a 6-channel NCHW concat of noisy latent/sample and low-res
  RGB image; the scheduler step updates only the first 3-channel latent tensor.
- Consistency models use continuous log-sigma timesteps from Karras sigmas, not
  DDPM integer alpha-product timesteps.
- Consistency `step` rejects integer loop indices; it expects values from
  `scheduler.timesteps`.
- `UNet2DModel` supports positional, Fourier, and learned timestep embeddings.
  The sampled configs here use the source default unless config says otherwise.
- `num_class_embeds` activates class embedding addition to the time embedding.
  This is required for ImageNet-64 consistency configs but inactive for cat-256.
- Source layout is NCHW throughout pipelines, UNets, and codecs. NHWC is only a
  guarded optimization. GroupNorm dim, channel concat, attention flattening,
  VQ vector quantization, scheduler broadcasting, and postprocess permutes must
  be axis-aware.
- Older configs can name classes that no longer correspond to the current
  non-deprecated pipeline source (`UNetModel`, `GaussianDDPMScheduler`,
  `LDMPipeline`). Treat them as compatibility/migration inputs.

## 4. Runtime tensor contract

Pixel DDPM/DDIM:

| Boundary | Tensor | Source layout | Shape examples | Notes |
| --- | --- | --- | --- | --- |
| initial noise | `image` | NCHW | `[B,3,32,32]` or `[B,3,256,256]` | sampled in `unet.dtype` |
| denoiser input | noisy image | NCHW | same | no `scale_model_input` in DDPM/DDIM source because it is identity |
| denoiser output | `model_output` | NCHW | same, or double channels for learned variance in general DDPM source | sampled configs use fixed variance |
| scheduler tables | alphas/betas/timesteps | scalar tables | 1000 train steps typical | cache per scheduler config and inference step count |
| final image | postprocessed | NCHW -> NHWC CPU | `[B,H,W,3]` | `(sample / 2 + 0.5).clamp(0,1)` |

Consistency:

| Boundary | Tensor | Source layout | Shape examples | Notes |
| --- | --- | --- | --- | --- |
| initial sample | `sample` | NCHW | `[B,3,64,64]`, `[B,3,256,256]` | random normal multiplied by `init_noise_sigma=sigma_max` |
| class labels | `class_labels` | vector | `[B]` int | required when `num_class_embeds` is configured |
| scaled input | `sample / sqrt(sigma^2 + sigma_data^2)` | NCHW | same | scalar sigma from scheduler step index |
| model output | consistency output | NCHW | same | combined with input sample by boundary scalings |
| scheduler output | `prev_sample` | NCHW | same | may add stochastic noise for multistep |

LDM text:

| Boundary | Tensor | Source layout | Shape for 256 target | Notes |
| --- | --- | --- | --- | --- |
| token ids | `input_ids` | `[B,77]` | fixed length 77 | LDMBert/tokenizer CPU/data stage first |
| text embeds | `prompt_embeds` | `[B,77,1280]` | `[B,77,1280]` | CFG concatenates negative and positive on batch |
| latents | `latents` | NCHW | `[B,4,32,32]` | source hard-codes `height // 8`, `width // 8` |
| UNet input | latent + context | NCHW + token context | `[2B,4,32,32]` under CFG | `UNet2DConditionModel` cross-attends to LDMBert |
| decoded image | VAE output | NCHW | `[B,3,256,256]` | `latents / scaling_factor` before decode |

LDM super-resolution:

| Boundary | Tensor | Source layout | Shape | Notes |
| --- | --- | --- | --- | --- |
| input image | low-res image | NCHW | `[B,3,H,W]` | PIL path resizes to multiple of 32, normalizes to `[-1,1]` |
| latents | noisy sample | NCHW | `[B,3,H,W]` because `unet.in_channels // 2` | multiplied by scheduler `init_noise_sigma` |
| UNet input | concat | NCHW | `[B,6,H,W]` | `torch.cat([latents, image], dim=1)` |
| UNet output | noise prediction | NCHW | `[B,3,H,W]` | scheduler updates 3-channel latents |
| decoded image | VQ decode | NCHW -> NHWC | usually 4x codec output depending VQ config | clamped then postprocessed |

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor allocation, shape validation, batch list generator validation.
- `torch.cat` on batch for CFG and channel for super-resolution.
- `chunk(2)` on batch for CFG.
- NCHW -> NHWC CPU postprocess permute.
- Broadcast scalar scheduler coefficients over `[B,C,H,W]`.
- Skip-connection tuple slicing and spatial/channel shape matching inside UNet.

Convolution/downsample/upsample ops:

- `UNet2DModel.conv_in`: examples `Conv2d(3 -> 128, 3x3)`,
  `Conv2d(3 -> 192, 3x3)`, `Conv2d(6 -> 160, 3x3)`.
- UNet ResnetBlock2D: GroupNorm, SiLU, two `3x3` convs, time embedding add or
  scale-shift depending config.
- `DownBlock2D`, `AttnDownBlock2D`, `ResnetDownsampleBlock2D`.
- `UpBlock2D`, `AttnUpBlock2D`, `ResnetUpsampleBlock2D`.
- VQ/Autoencoder encoders and decoders: Conv2d, ResNet blocks, up/downsample,
  mid attention, `1x1` quant/post-quant convs.

GEMM/linear ops:

- Timestep embedding MLP.
- Optional class embedding lookup and addition.
- Attention Q/K/V/out projections in UNet attention blocks.
- LDMBert embeddings, self-attention projections, FFN `1280 -> 5120 -> 1280`.
- `UNet2DConditionModel` cross-attention projections for LDM text.

Attention primitives:

- UNet spatial self-attention in `AttnDownBlock2D`, `AttnUpBlock2D`, and mid
  block; query length is spatial `H*W`.
- LDM text cross-attention from latent tokens to LDMBert context length 77.
- LDMBert bidirectional self-attention with additive expanded mask.
- Eager/SDPA attention processors define parity; fused projections are
  optional optimization.

Normalization and adaptive conditioning:

- GroupNorm over source channel axis.
- LayerNorm in transformer blocks and LDMBert.
- Optional class-conditioned time embedding addition.
- CM scheduler boundary scalings are model-output conditioning, not a norm.

Scheduler and guidance arithmetic:

- DDPM alpha-product prediction conversion, clipping/thresholding, variance
  branch, and noise add.
- DDIM alpha-product deterministic and eta stochastic update.
- CM Karras sigma table, input scaling, boundary condition scalings, stochastic
  re-noise.
- CFG arithmetic for LDM text only.

VAE/postprocessing:

- AutoencoderKL `quant_conv`, diagonal Gaussian sample/mode for encode paths,
  `post_quant_conv`, decoder.
- VQModel vector quantizer on decode unless `force_not_quantize=True`.
- Clamp and denormalize to `[0,1]`.

## 6. Denoiser/model breakdown

`UNet2DModel` forward:

```text
optional center_input_sample
timesteps -> Timesteps/GaussianFourier/Embedding -> TimestepEmbedding
optional class embedding -> add to time embedding
Conv2d(in_channels -> block_out_channels[0])
down blocks:
  ResnetBlock2D repeated
  optional spatial self-attention
  optional downsample or ResNet downsample
mid block:
  ResnetBlock2D -> optional Attention -> ResnetBlock2D
up blocks:
  pop skip tensors
  ResnetBlock2D repeated
  optional spatial self-attention
  optional upsample or ResNet upsample
GroupNorm -> SiLU -> Conv2d(block_out_channels[0] -> out_channels)
optional skip_sample add for certain block types
optional Fourier output divide by timestep
```

`UNet2DConditionModel` in LDM text follows the Stable Diffusion 1.x shape:
ResNet down/mid/up blocks plus `BasicTransformerBlock` cross-attention using
LDMBert hidden states. The LDM text config uses block widths
`320/640/1280/1280`, latent channels 4, cross-attention context inferred from
LDMBert `d_model=1280`, and `attention_head_dim=8`.

`LDMBertModel` local source:

```text
token ids -> token embedding + learned position embedding
32 encoder layers:
  LayerNorm -> multi-head self-attention (8 heads, head_dim 64)
  residual
  LayerNorm -> GELU FFN 1280 -> 5120 -> 1280
  residual
final LayerNorm
```

`VQModel` decode:

```text
optional vector quantization over latent codebook
post_quant_conv 1x1
Decoder: conv/resnet/down-up mirrored blocks, optional mid attention
```

`AutoencoderKL` decode:

```text
latents / scaling_factor from pipeline
post_quant_conv 1x1 if configured
Decoder conv/resnet/up blocks
```

## 7. Attention requirements

- Pixel DDPM/DDIM/CM attention is spatial self-attention only. No masks, no
  causal attention, no KV cache.
- Attention head shape comes from config or source fallback. When
  `attention_head_dim` is omitted in `UNet2DModel`, source uses the constructor
  default 8; when explicitly set to 64, head count per block is width / 64
  where divisible.
- LDM text adds cross-attention in `UNet2DConditionModel`: latent spatial tokens
  query LDMBert sequence length 77, hidden dim 1280.
- LDMBert self-attention uses explicit `torch.bmm`, softmax, dropout in source,
  with additive expanded attention mask if supplied.
- Primary Diffusers parity path for UNet attention remains
  `attention.py`/`attention_processor.py`; `attention_dispatch.py` is not the
  classic family owner.
- Flash-style Dinoml providers are plausible for mask-free spatial self-attn
  and LDM cross-attn when dtype/head_dim constraints hold. LDMBert masked
  attention needs additive-mask support or an eager/SDPA fallback.
- IP-Adapter added-KV, joint attention, RoPE, QK norm, GQA, and varlen paths are
  not required by this target.

## 8. Scheduler and denoising-loop contract

DDPM:

```text
set_timesteps(num_inference_steps)
for t in scheduler.timesteps:
  model_output = unet(sample, t).sample
  pred_x0 = convert epsilon/sample/v_prediction
  pred_x0 = optional threshold/clip
  mean = coeff_x0 * pred_x0 + coeff_xt * sample
  if t > 0: add variance noise
```

First slice: epsilon prediction, fixed_small/fixed_large variance, `clip_sample`
as configured. Learned variance is source-supported but not in sampled configs.

DDIM:

```text
constructor coerces scheduler = DDIMScheduler.from_config(scheduler.config)
set_timesteps(num_inference_steps)
for t:
  model_output = unet(sample, t).sample
  pred_x0 / pred_epsilon = prediction_type conversion
  optional clip/threshold
  prev = sqrt(alpha_prev) * pred_x0 + sqrt(1-alpha_prev-std^2) * pred_epsilon
  if eta > 0: add std * noise
```

First slice: `eta=0`, epsilon prediction, no thresholding unless config enables.

Consistency:

```text
set_timesteps(num_inference_steps or custom descending timesteps)
sigmas = Karras schedule; timesteps = 250 * log(sigmas)
sample = noise * sigma_max
for t in timesteps:
  scaled = sample / sqrt(sigma^2 + sigma_data^2)
  model_output = unet(scaled, t, optional class_labels)
  denoised = c_out * model_output + c_skip * sample
  optional clamp [-1,1]
  prev = denoised + stochastic_noise * sqrt(sigma_next^2 - sigma_min^2)
```

First slice: one-step deterministic (`num_inference_steps=1`) and then two-step
custom schedule. Scheduler `sigmas` are stored on CPU in source after
`set_timesteps`; Dinoml should make this table explicit.

LDM text and super-resolution loops match the scheduler matrix alpha-product
contracts. LDM text adds CFG batching; super-resolution adds channel concat and
`scheduler.scale_model_input` before UNet for compatible schedulers.

## 9. Position, timestep, and custom math

UNet timestep embedding:

- Positional `Timesteps` default with `flip_sin_to_cos=True`, `freq_shift=0`
  unless config overrides.
- Fourier path divides final sample by timestep; not seen in sampled configs but
  source-supported.
- Learned timestep embedding requires `num_train_timesteps`.

Consistency sigma math:

```text
t = 250 * log(sigma + 1e-44)
c_skip = sigma_data^2 / ((sigma - sigma_min)^2 + sigma_data^2)
c_out = (sigma - sigma_min) * sigma_data / sqrt(sigma^2 + sigma_data^2)
```

LDMBert positions are learned absolute positions of length 77. Prompt embeds
can be precomputed; timestep embeddings depend on selected scheduler timesteps;
CM sigma tables depend on `num_inference_steps` or explicit custom timesteps.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- Pixel DDPM/DDIM/CM have no tokenizer or image input. They start from random
  noise unless caller supplies CM latents.
- LDM text tokenizes prompts and empty strings to fixed length 77; LDMBert
  produces hidden states.
- LDM super-resolution PIL preprocessing resizes to the nearest lower multiple
  of 32, converts RGB to NCHW float, and normalizes to `[-1,1]`.

GPU/runtime work:

- Random noise allocation in target dtype.
- LDM text CFG: concatenate negative and positive prompt embeddings and latents
  on batch dimension, then chunk model output.
- LDM super-resolution: concatenate `[latents, low_res_image]` on channel
  dimension every step.
- VQ/Autoencoder decode and final postprocess after the loop.

## 11. Graph rewrite / lowering opportunities

### Rewrite: NCHW UNet conv island to guarded NHWC

Source pattern:

```text
Conv2d -> GroupNorm(channel axis=1) -> SiLU -> Conv2d -> residual
```

Replacement: NHWC conv/norm/residual island with NCHW boundaries preserved at
pipeline edges.

Preconditions: all consumers inside island use rewritten channel axis; attention
flatten/unflatten is either outside the island or explicitly rewritten; channel
concat for super-resolution remains correct.

Weight transform: OIHW -> HWIO for NHWC conv kernels.

Failure cases: VQ quantizer, LDMBert token attention, scheduler table indexing,
or arbitrary `view` assumptions inside a translated region.

Parity test: one ResnetBlock2D and one down/up block at CIFAR 32 and 256-scale
intermediate resolutions.

### Rewrite: scheduler scalar step to fused pointwise

Source pattern: multiple scalar table lookups followed by broadcast arithmetic
over `[B,C,H,W]`.

Replacement: per-scheduler fused kernel with explicit config fields and
timestep index.

Preconditions: scheduler family, prediction type, variance mode, clip/threshold
mode, and stochastic-noise input are fixed in the execution plan.

Failure cases: learned variance output split, DDIM eta noise, dynamic
thresholding, CM custom timetable not admitted.

Parity test: fixed random tensors against Diffusers `step` for DDPM, DDIM, and
CM.

### Rewrite: LDM CFG batched mode to explicit two-call option

Source pattern: `cat([latents]*2)` and `cat([neg,prompt])`, one UNet call,
`chunk(2)`, CFG arithmetic.

Replacement: artifact-level choice between batched UNet and two denoiser calls.

Preconditions: UNet is pure per batch entry; conditioning tensors align exactly.

Failure cases: memory planning, future adapters, or side inputs with per-branch
state.

Parity test: compare batched and two-call LDM text step.

### Rewrite: super-resolution concat + conv_in folding

Source pattern: `torch.cat([latents, image], dim=1)` followed by
`Conv2d(6 -> 160, 3x3)`.

Replacement: split `conv_in` weights into latent/image halves and compute
`conv(latents, W_latent) + conv(image, W_image) + bias`, optionally caching the
low-res image branch across timesteps.

Preconditions: first op after concat is linear conv; low-res image tensor is
constant across denoising steps; padding/layout identical.

Failure cases: nonlinear preprocessing between concat and conv, dynamic image
changes, non-conv first consumer.

Parity test: exact fp32 equality within conv tolerance for the first UNet stem.

## 12. Kernel fusion candidates

Highest priority:

- DDPM/DDIM/CM scheduler step pointwise kernels. These pipelines are otherwise
  simple enough that scheduler overhead and bandwidth are visible.
- GroupNorm + SiLU + Conv2d in UNet and VQ/Autoencoder decoders.
- UNet spatial attention QKV + attention + output projection for pixel and LDM
  blocks.
- LDM CFG arithmetic and optional prompt/context batch handling.

Medium priority:

- NHWC conv islands for UNet and codec decode.
- Super-resolution first-conv split/cache for the static low-res branch.
- VQ/Autoencoder decode upsample/resnet fusions.
- LDMBert self-attention and FFN fusions if text encoder moves into Dinoml.

Lower priority:

- DDPM learned-variance and dynamic thresholding.
- DDIM `eta > 0` stochastic branch.
- CM multistep stochastic noise branch beyond one/two-step parity.
- PNDM/LMS/Euler/DPM scheduler swaps for LDM super-resolution before DDIM is
  stable.

## 13. Runtime staging plan

Stage 1: Pixel DDPM CIFAR first. Parse flat `model_index.json`, `config.json`,
and `scheduler_config.json`; load `UNet2DModel`; run one DDPM scheduler step
with fixed tensors.

Stage 2: Pixel UNet block parity. Validate Conv2d, GroupNorm, SiLU,
Down/UpBlock2D, AttnDown/UpBlock2D, and timestep embedding on CIFAR and cat
shapes.

Stage 3: Full DDPM loop with scheduler in host code and compiled/fused UNet
step. Postprocess to image tensor.

Stage 4: DDIM slice. Reuse pixel UNet and implement DDIM epsilon `eta=0`; add
eta noise later.

Stage 5: Consistency one-step. Implement CM scheduler table generation,
`scale_model_input`, boundary-condition `step`, and optional class embedding.

Stage 6: LDM text denoiser-only. Accept externally supplied LDMBert embeddings,
run one CFG denoising step on `[B,4,32,32]`, and decode with AutoencoderKL.

Stage 7: LDM super-resolution. Add low-res preprocessing boundary, 6-channel
UNet stem, and VQ decode.

Stage 8: Optimize: NHWC conv islands, attention kernels, super-resolution
static-branch conv caching, and codec decode fusions.

## 14. Parity and validation plan

- Config parse tests for old flat configs and current component-subfolder
  configs.
- Random tensor tests for DDPM `step`, `add_noise`, and `get_velocity`.
- Random tensor tests for DDIM `step` with `eta=0`, then `eta>0` with supplied
  variance noise.
- CM sigma table, timestep conversion, one-step and custom-timestep step parity.
- `UNet2DModel` ResnetBlock2D and attention block parity at 32, 64, 256-family
  shapes.
- Class-conditional consistency parity with fixed ImageNet labels.
- LDM text CFG arithmetic and one denoising step parity with fixed prompt
  embeddings.
- VQModel decode parity and AutoencoderKL decode parity.
- Super-resolution concat/stem parity and full one-step scheduler update.
- Short deterministic loops with fixed generator/noise and scheduler in host
  code.
- Suggested tolerances: fp32 scheduler `rtol=1e-5, atol=1e-6`; fp32 UNet/codec
  `rtol=1e-4, atol=1e-5`; fp16/bf16 initial `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Pixel UNet forward latency by sample size: 32, 64, 256.
- DDPM/DDIM scheduler overhead as a percentage of step time.
- CM one-step vs two-step latency and stochastic-noise overhead.
- UNet conv/resnet vs attention split.
- NCHW faithful path vs guarded NHWC conv islands.
- LDM text denoiser step with and without CFG batching.
- AutoencoderKL and VQModel decode throughput.
- Super-resolution first-conv static-branch cache benefit.
- VRAM and temporary tensor usage for 256 pixel UNet and LDM text.

## 16. Scope boundary and separate candidates

Separate candidate reports related but not first implementation:

- `ldm_superresolution`: `LDMSuperResolutionPipeline`,
  `pipeline_latent_diffusion_superresolution.py`; deeper audit of low-res image
  conditioning, scheduler swaps, VQ decode, and first-conv branch caching.
- `latent_diffusion_uncond_deprecated`: deprecated `LDMPipeline` for
  `CompVis/ldm-celebahq-256`; only needed for old artifact compatibility.
- `scheduler_ddpm_ddim_extended`: learned variance, dynamic thresholding,
  v-pred/sample prediction, DDIM eta/noise variants, old Gaussian scheduler
  migration.
- `scheduler_consistency_models`: CM multistep schedules, stochastic sampling,
  class-conditional ImageNet variants, and consistency decoder separation.
- `ldm_bert_text_encoder`: compile LDMBert rather than accepting prompt
  embeddings externally.
- `vqmodel_autoencoderkl_codecs`: codec-specific optimization beyond the
  already completed AutoencoderKL report.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt behavior.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- SD-family LoRA, textual inversion, runtime adapters, IP-Adapter, ControlNet,
  T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and SD upscaling.

## 17. Final implementation checklist

- [ ] Parse old flat and current component Diffusers configs.
- [ ] Load `UNet2DModel` weights for `google/ddpm-cifar10-32`.
- [ ] Implement `DDPMScheduler` epsilon fixed-variance step.
- [ ] Add DDPM scheduler table and `add_noise` parity tests.
- [ ] Implement `UNet2DModel` Conv/GroupNorm/SiLU/ResNet/down/up parity.
- [ ] Implement spatial self-attention parity for `AttnDownBlock2D` and `AttnUpBlock2D`.
- [ ] Run one full pixel DDPM loop with host scheduler.
- [ ] Implement `DDIMScheduler` epsilon `eta=0` step.
- [ ] Implement CM scheduler Karras sigma tables and boundary-condition step.
- [ ] Add class embedding path for ImageNet-64 consistency configs.
- [ ] Accept external LDMBert prompt embeddings for LDM text first slice.
- [ ] Implement LDM CFG concat/chunk/arithmetic.
- [ ] Implement AutoencoderKL and VQModel decode boundaries or reuse codec reports.
- [ ] Add LDM super-resolution channel-concat contract.
- [ ] Add guarded NHWC conv-island rewrite tests.
- [ ] Benchmark pixel UNet, scheduler kernels, LDM denoiser, and codec decode separately.
