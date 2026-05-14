# DeepFloyd IF Diffusers Audit

Candidate slug: `deepfloyd_if`

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  DeepFloyd/IF-I-M-v1.0
  DeepFloyd/IF-I-L-v1.0
  DeepFloyd/IF-I-XL-v1.0
  DeepFloyd/IF-II-M-v1.0
  DeepFloyd/IF-II-L-v1.0
  stabilityai/stable-diffusion-x4-upscaler

Config sources:
  H:/configs/DeepFloyd/IF-I-M-v1.0/
  H:/configs/DeepFloyd/IF-I-L-v1.0/
  H:/configs/DeepFloyd/IF-I-XL-v1.0/
  H:/configs/DeepFloyd/IF-II-M-v1.0/
  H:/configs/DeepFloyd/IF-II-L-v1.0/
  H:/configs/stabilityai/stable-diffusion-x4-upscaler/

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/resnet.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/pipelines/deepfloyd_if/timesteps.py
  X:/H/diffusers/docs/source/en/api/pipelines/deepfloyd_if.md

External component configs inspected:
  T5EncoderModel/T5Tokenizer configs from DeepFloyd stage I and II repos.
  CLIPTextModel/CLIPTokenizer configs from stabilityai/stable-diffusion-x4-upscaler.

Missing files or assumptions:
  Official JSON configs were reachable through huggingface_hub and saved locally.
  Expected absent components returned 404: Stage I has no image_noising_scheduler,
  low_res_scheduler, or VAE; Stage II has no low_res_scheduler or VAE; Stage III
  has low_res_scheduler rather than image_noising_scheduler. Safety checker and
  watermarker source were out of runtime scope except as optional pipeline slots.
```

## 2. Pipeline and component graph

DeepFloyd IF is a three-stage cascade. Stage I and II are pixel-space diffusion
pipelines; they denoise RGB image tensors directly, not VAE latents. Stage III is
the Stable Diffusion x4 upscaler and switches to latent-space denoising plus VAE
decode.

```text
prompt text
  -> Stage I T5 tokenizer/text encoder, or cached prompt embeddings
  -> Stage I pixel denoising: UNet2DConditionModel + CFG + DDPMScheduler
  -> 64x64 RGB tensor in [-1, 1]
  -> Stage II preprocess/upscale/noise low-res RGB image
  -> Stage II pixel super-resolution denoising with T5 embeddings and class noise level
  -> 256x256 RGB tensor in [-1, 1]
  -> Stage III CLIP tokenizer/text encoder, or cached prompt embeddings
  -> low-res RGB preprocessing/noising + latent initialization
  -> Stable Diffusion x4 latent denoising: UNet2DConditionModel + CFG + DDIMScheduler
  -> AutoencoderKL decode/postprocess
  -> 1024x1024 image
```

Stage I `IFPipeline` required runtime components are `tokenizer`,
`text_encoder`, `unet`, and `scheduler`, unless prompt embeddings are supplied.
Optional components are `safety_checker`, `feature_extractor`, and `watermarker`.
`model_cpu_offload_seq` is `text_encoder->unet`.

Stage II `IFSuperResolutionPipeline` adds `image_noising_scheduler` and accepts a
low-resolution image. The constructor warns when the UNet does not have
`in_channels == 6`, because Stage II concatenates current RGB sample and noised
upscaled RGB conditioning. `model_cpu_offload_seq` is again `text_encoder->unet`.

Stage III `StableDiffusionUpscalePipeline` required runtime components are
`vae`, `text_encoder`, `tokenizer`, `unet`, `scheduler`, and
`low_res_scheduler`. Its offload sequence is `text_encoder->unet->vae`.

Independently cacheable stages:

- T5 prompt and negative prompt embeddings for Stage I/II: `[B, 77, 4096]`,
  duplicated per image and concatenated for CFG when `guidance_scale > 1`.
- Stage I RGB output tensor for Stage II and Stage III.
- Stage II upscaled/noised conditioning image for all denoising steps at fixed
  `noise_level`.
- Stage III CLIP prompt embeddings and noised low-res image.
- Scheduler timestep tables and scalar alpha/beta tables per step count.

Separate candidate reports inventory:

| Surface | Support in family | Class/file anchors | Why separate |
| --- | --- | --- | --- |
| LoRA | Loader mixin present on IF and x4 upscaler pipelines | `StableDiffusionLoraLoaderMixin`; IF pipeline files; `pipeline_stable_diffusion_upscale.py` | Runtime weight mutation and text-encoder/UNet adapter policy should follow the loader report. |
| Textual inversion | Not in IF Stage I/II class inheritance; supported by Stage III Stable Diffusion pipeline family | Stable Diffusion loaders and `StableDiffusionUpscalePipeline` | Tokenizer/embedding mutation is not needed for first IF pixel cascade parity. |
| Runtime adapter mutation | Generic attention processor replacement is source-supported | `attention_processor.py`, pipeline `cross_attention_kwargs` | Processor/backend selection changes attention lowering. |
| IP-Adapter | Not active in IF configs | `models/attention_processor.py`, loader surfaces | Added image K/V branches are absent from selected configs. |
| ControlNet | No DeepFloyd IF ControlNet pipeline in this folder | N/A for selected folder | Separate family if a downstream repo adds IF control residuals. |
| T2I-Adapter | Not present in IF folder | N/A | Separate SD adapter surface. |
| GLIGEN | Not present in IF folder | N/A | Deprecated SD branch, unrelated to IF first slice. |
| img2img | Present | `pipeline_if_img2img.py`, `pipeline_if_img2img_superresolution.py` | Adds image noising/strength timestep slicing before denoising. |
| inpaint | Present | `pipeline_if_inpainting.py`, `pipeline_if_inpainting_superresolution.py` | Adds mask preprocessing and masked-image blending contracts. |
| depth2img | Not present for IF | N/A | No depth encoder in selected family. |
| upscaling | Core Stage II/III behavior | `pipeline_if_superresolution.py`, `pipeline_stable_diffusion_upscale.py` | Stage II belongs in this report; Stage III is best admitted after latent SD x4/VAE slices exist. |

## 3. Important config dimensions

### Representative checkpoint sweep

| Repo | Pipeline class | Space | Sample target | UNet in/out | Blocks | Cross-attn dim | Text encoder | Scheduler default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DeepFloyd/IF-I-M-v1.0` | `IFPipeline` | pixel RGB | 64x64 | 3 -> 6 | 192,384,576,768; 3 layers/block | 768 | T5 d_model 4096 | DDPM epsilon, learned_range variance |
| `DeepFloyd/IF-I-L-v1.0` | `IFPipeline` | pixel RGB | 64x64 | 3 -> 6 | 320,640,960,1280; 3 layers/block | 1280 | T5 d_model 4096 | DDPM epsilon, learned_range variance |
| `DeepFloyd/IF-I-XL-v1.0` | `IFPipeline` | pixel RGB | 64x64 | 3 -> 6 | 704,1408,2112,2816; 3 layers/block | 2816 | T5 d_model 4096 | DDPM epsilon, learned_range variance |
| `DeepFloyd/IF-II-M-v1.0` | `IFSuperResolutionPipeline` | pixel RGB SR | 256x256 | 6 -> 6 | 128,256,512,768,768; layers 2,2,3,4,4 | 768 | T5 d_model 4096 | DDPM epsilon, learned_range; image noising DDPM fixed_small |
| `DeepFloyd/IF-II-L-v1.0` | `IFSuperResolutionPipeline` | pixel RGB SR | 256x256 | 6 -> 6 | 160,320,640,960,1280; layers 2,2,3,5,5 | 1280 | T5 d_model 4096 | DDPM epsilon, learned_range; image noising DDPM fixed_small |
| `stabilityai/stable-diffusion-x4-upscaler` | `StableDiffusionUpscalePipeline` | latent SR | 128 latent grid for 1024 output | 7 -> 4 | 256,512,512,1024; 2 layers/block | 1024 | CLIP hidden 1024 | DDIM v_prediction; low-res DDPM fixed_small |

### IF Stage I/II UNet config facts

| Field | Stage I XL | Stage II L |
| --- | --- | --- |
| `sample_size` | 64 | 256 |
| `in_channels` / `out_channels` | 3 / 6 | 6 / 6 |
| `down_block_types` | `ResnetDownsampleBlock2D`, then 3x `SimpleCrossAttnDownBlock2D` | 3x `ResnetDownsampleBlock2D`, then 2x `SimpleCrossAttnDownBlock2D` |
| `up_block_types` | 3x `SimpleCrossAttnUpBlock2D`, then `ResnetUpsampleBlock2D` | 2x `SimpleCrossAttnUpBlock2D`, then 3x `ResnetUpsampleBlock2D` |
| `mid_block_type` | `UNetMidBlock2DSimpleCrossAttn` | `UNetMidBlock2DSimpleCrossAttn` |
| `addition_embed_type` | `text` | `text` |
| `encoder_hid_dim_type` | `text_proj`, 4096 -> cross-attn dim | `text_proj`, 4096 -> cross-attn dim |
| `resnet_time_scale_shift` | `scale_shift` | `scale_shift` |
| `class_embed_type` | null | `timestep` for `noise_level` |
| `act_fn` | `gelu` | `gelu` |
| `attention_head_dim` | 64 | 64 |
| `cross_attention_norm` | `group_norm` | `group_norm` |

### Stage III x4 config facts

| Component | Important dimensions |
| --- | --- |
| CLIP text encoder | hidden size 1024, 23 layers, 16 heads, max positions 77, vocab 49408 |
| UNet | latent channels 4 plus low-res RGB 3 -> `in_channels=7`, `out_channels=4`, `sample_size=128`, `use_linear_projection=true`, `only_cross_attention=[true,true,true,false]`, class labels via `num_class_embeds=1000` |
| VAE | `AutoencoderKL`, `latent_channels=4`, block channels 128,256,512, `scaling_factor=0.08333`, scale factor 4 from three block widths |
| Main scheduler | `DDIMScheduler`, `prediction_type=v_prediction`, `beta_schedule=scaled_linear`, `clip_sample=false`, `steps_offset=1` |
| Low-res scheduler | `DDPMScheduler`, fixed-small epsilon noising, `beta_schedule=scaled_linear`, max noise level 350 from pipeline config |

Recommended first Dinoml scheduler slice: DDPM epsilon with learned-range
variance for IF Stage I/II, plus DDPM `add_noise` for Stage II low-res image
conditioning. Stage III requires the already-separate SD x4 latent upscaler
surface: DDIM v-pred, DDPM low-res noising, AutoencoderKL decode.

## 3a. Family variation traps

- DeepFloyd IF Stage I/II are pixel-space RGB diffusion. Do not apply Stable
  Diffusion latent scaling, VAE encode/decode, or 4-channel latent assumptions.
- IF UNet `out_channels=6` means the model predicts noise plus variance. The
  pipeline splits variance handling differently depending on scheduler
  `variance_type`.
- Stage II `in_channels=6` is `[current RGB sample, noised upscaled RGB]`, not
  mask concatenation or VAE latent concatenation.
- Stage II class labels are the low-res `noise_level` passed through the UNet
  class embedding path (`class_embed_type="timestep"`).
- Stage I M/L/XL change width and cross-attention dim; Stage II M/L change width
  and per-block depth. A single hard-coded IF width is wrong.
- Stage III is not an IF pipeline class. It uses CLIP text, latent denoising,
  DDIM v-pred, `low_res_scheduler`, and VAE decode.
- IF SimpleCrossAttn blocks use `Attention` with `AttnAddedKVProcessor2_0` when
  PyTorch SDPA exists. This is self-attention over image tokens with added text
  K/V, not classic `BasicTransformerBlock` cross-attention.
- Source tensors are NCHW. NHWC/channel-last can be a guarded optimization only
  inside local conv/resnet/attention islands; scheduler broadcasting, `dim=1`
  channel splits/concats, GroupNorm axes, and output postprocess permutations
  must preserve source semantics.
- `timesteps.py` defines hand-picked IF timestep schedules such as `smart27`,
  `smart50`, `smart100`, `super27`, `super40`, and `super100`; pipelines accept
  explicit `timesteps`.
- Img2img/inpaint variants add strength-based timestep slicing and image/mask
  contracts; do not count them as base first-slice inputs.

## 4. Runtime tensor contract

### Stage I

CPU/data-pipeline work:

- Optional caption cleaning (`bs4`, `ftfy`) and lowercasing.
- T5 tokenization with max length 77, padding/truncation, attention mask.

GPU/runtime work:

- T5 encoder output: `prompt_embeds` `[B, 77, 4096]`; projected inside UNet to
  cross-attention dim 768/1280/2816 by `encoder_hid_proj`.
- CFG: concatenate negative and positive embeddings along batch to
  `[2B * images_per_prompt, 77, 4096]`.
- Initial sample: `intermediate_images` `[B, 3, H, W]`, default `H=W=64`,
  Gaussian noise scaled by `scheduler.init_noise_sigma`.
- Per step model input: `[2B, 3, 64, 64]` under CFG after
  `scheduler.scale_model_input`.
- UNet output: `[2B, 6, 64, 64]`; pipeline splits into 3-channel predicted noise
  and 3-channel predicted variance.
- Scheduler step updates the 3-channel pixel sample.
- Postprocess for PIL/np: `(image / 2 + 0.5).clamp(0,1)`, NCHW -> NHWC CPU
  float numpy, then PIL conversion.

### Stage II

- Input image accepts PIL/np/torch/list. PIL/np are normalized to `[-1,1]` and
  transposed NHWC -> NCHW.
- Low-res conditioning image: `[B, 3, h, w]`, repeated per prompt, bilinear
  interpolated to `[B, 3, 256, 256]` with `align_corners=True`.
- `image_noising_scheduler.add_noise` applies DDPM noising at scalar
  `noise_level`, default 250. The noised upscaled image is constant across
  denoising steps.
- Initial current sample: `[B, 3, 256, 256]`.
- Per step model input before CFG batch duplication:
  `cat([intermediate_images, upscaled], dim=1)` -> `[B, 6, 256, 256]`.
- With CFG: `[2B, 6, 256, 256]`; `noise_level` is also duplicated to `[2B]`.
- UNet output: `[2B, 6, 256, 256]`; guidance uses the first 3 channels as noise
  and carries predicted variance from the conditional half.

### Stage III

- Low-res image is preprocessed by `VaeImageProcessor` to NCHW in `[-1,1]`,
  then DDPM noised at default `noise_level=20`.
- CLIP prompt embeds are `[B, 77, 1024]`, CFG-concatenated when needed.
- Latents: `[B, 4, H/4, W/4]` for the upscaled output; for a 1024 image from
  256 input, latent grid is 256 by source image dimensions in the pipeline path,
  with UNet config `sample_size=128` as a training/default config value.
- Per step model input: scale latent input, then channel concat with low-res RGB:
  `[2B, 4, h, w] + [2B, 3, h, w] -> [2B, 7, h, w]`.
- UNet output: `[2B, 4, h, w]`, CFG arithmetic, DDIM step on latents.
- Decode: `vae.decode(latents / scaling_factor)`, then image processor
  postprocess.

Precomputable tensors: prompt embeddings, negative prompt embeddings, low-res
noised conditioning images for fixed noise level, scheduler tables, and
per-step scalar coefficients.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor concat/split/chunk on `dim=1` for RGB/noise/variance and
  low-res conditioning.
- Repeat, repeat_interleave, view/reshape for prompt duplication and attention
  head packing.
- NCHW <-> NHWC transpose for PIL/np boundaries.
- Bilinear resize with `align_corners=True` for IF Stage II conditioning.
- Stage III bicubic image processor resize/normalize/postprocess.

Convolution/downsample/upsample ops:

- IF Stage I XL starts with `Conv2d(3 -> 704, 3x3, padding=1)` and ends with
  `Conv2d(704 -> 6, 3x3, padding=1)`.
- IF Stage II L starts with `Conv2d(6 -> 160, 3x3, padding=1)` and ends with
  `Conv2d(160 -> 6, 3x3, padding=1)`.
- ResNet downsample/upsample blocks in IF use `ResnetBlock2D(..., down=True)`
  or `up=True`, not only standalone strided conv/interpolate.
- Stage III x4 uses SD-style DownBlock/CrossAttn blocks and AutoencoderKL
  conv/resnet/upsample decode.

GEMM/linear ops:

- Timestep projection and `TimestepEmbedding` MLPs.
- IF text projection 4096 -> 768/1280/2816.
- Added-K/V attention projections for text keys/values, query/key/value
  projections for image states, output projections.
- Stage III CLIP text encoder and UNet cross-attention linear projections.

Attention primitives:

- IF SimpleCrossAttn `Attention` over flattened spatial tokens with added text
  K/V and optional SDPA backend.
- GroupNorm before IF attention (`norm_num_groups=32`) and optional
  group-normalized cross-attention context (`cross_attention_norm="group_norm"`).
- Stage III standard cross-attention in `CrossAttnDownBlock2D`/`CrossAttnUpBlock2D`.

Normalization/adaptive conditioning:

- GroupNorm, scale-shift ResNet timestep conditioning, GELU/SiLU, residual
  scaling by `resnet_out_scale_factor`.
- Stage II class/noise-level embedding through UNet class embedding.
- Stage III VAE GroupNorm and CLIP LayerNorm.

Scheduler/guidance arithmetic:

- DDPM `set_timesteps`, `add_noise`, learned-range variance step math,
  thresholding/clip-sample for IF main scheduler.
- CFG formula `uncond + scale * (text - uncond)` with IF variance channel
  carry-through.
- Stage III DDIM v-pred step and DDPM low-res noising.

## 6. Denoiser/model breakdown

IF `UNet2DConditionModel` forward path:

```text
NCHW sample
  -> optional center input (false in sampled configs)
  -> sinusoidal timestep projection + TimestepEmbedding
  -> optional class embedding (Stage II noise_level only)
  -> optional addition text embedding
  -> encoder_hidden_states text projection
  -> conv_in
  -> down blocks with skip states
  -> UNetMidBlock2DSimpleCrossAttn
  -> up blocks consuming skip states
  -> norm/activation/conv_out
```

IF ResNet block:

```text
GroupNorm -> GELU -> Conv2d
  -> timestep scale-shift modulation
  -> GroupNorm -> GELU -> dropout-disabled inference -> Conv2d
  -> optional shortcut/down/up path
  -> residual / output_scale_factor
```

IF SimpleCrossAttn block:

```text
ResnetBlock2D
  -> Attention(query_dim=channels, cross_attention_dim=channels,
               added_kv_proj_dim=config cross_attention_dim,
               heads=channels/64, dim_head=64,
               processor=AttnAddedKVProcessor2_0 when SDPA exists)
  -> repeat per layer
  -> optional ResnetBlock2D down/up sampler
```

`AttnAddedKVProcessor2_0` flattens NCHW spatial states to `[B, HW, C]`, applies
GroupNorm over channels, projects image Q/K/V, projects text added K/V, concats
text K/V before image K/V when `only_cross_attention=false`, runs
`scaled_dot_product_attention`, projects out, reshapes to NCHW, and adds the
residual.

Stage III x4 denoiser is the classic SD UNet path with latent+low-res RGB
channel concat. It uses `DownBlock2D`, `CrossAttnDownBlock2D`, `UNetMidBlock2D`
default from source, `CrossAttnUpBlock2D`, and `UpBlock2D`, with `num_class_embeds=1000`
for low-res noise level labels.

## 7. Attention requirements

IF Stage I/II required attention:

- Spatial self-attention query sequence length is `H*W` at the block resolution.
- Added-K/V text context length is 77.
- Head dim is 64; head count is `channels / 64`, so Stage I XL reaches 44 heads
  at 2816 channels.
- Primary implementation path in current PyTorch is `Attention` with
  `AttnAddedKVProcessor2_0` in `attention_processor.py`; fallback is
  `AttnAddedKVProcessor` when SDPA is unavailable.
- No RoPE, QK norm, varlen packing, causal mask, or joint text-image query
  attention is required for IF configs.
- `attention_mask` is normally absent in the pipeline call; T5 attention mask is
  used by the text encoder, not passed into UNet.

Stage III required attention:

- CLIP text self-attention outside Diffusers model source.
- UNet cross-attention with latent spatial tokens and CLIP context dim 1024.
- `use_linear_projection=true` changes the attention projection shape in
  `Transformer2DModel`-based cross-attention blocks.

A Dinoml flash-style provider is valid for IF only under guards that support
added text K/V, residual output, GroupNorm pre-attention, and NCHW flatten/reshape
parity. Eager/SDPA parity is the first reference.

## 8. Scheduler and denoising-loop contract

IF Stage I/II main scheduler:

- `DDPMScheduler`, `num_train_timesteps=1000`, `beta_schedule=squaredcos_cap_v2`,
  `prediction_type=epsilon`, `variance_type=learned_range`,
  `clip_sample=true`, `thresholding=true`.
- Pipeline calls `set_timesteps(num_inference_steps)` or
  `set_timesteps(timesteps=custom_list)`, then `set_begin_index(0)` when
  available.
- `scale_model_input` is identity for DDPM source.
- Step receives model output with predicted variance channels when
  `variance_type=learned_range`.

Stage II image noising scheduler:

- `DDPMScheduler`, same squared cosine beta schedule, `variance_type=fixed_small`,
  `thresholding=false`.
- Only `add_noise(upscaled, noise, noise_level)` is required for the base Stage
  II path.

Stage III scheduler contract:

- Main `DDIMScheduler` with scaled-linear beta schedule, v-prediction, no
  clipping, `steps_offset=1`.
- Low-res scheduler is DDPM fixed-small add-noise only.
- Stage III scheduler surface should reuse the SD scheduler matrix and x4
  upscaler admission rather than being folded into IF pixel-stage support.

Loop-side graph work: host should own schedule construction, timestep list
validation, progress/callbacks ignored by current scope, and generator plumbing.
Compiled/runtime candidates are CFG arithmetic, DDPM step pointwise maps,
DDPM add-noise, and IF variance channel split/concat.

## 9. Position, timestep, and custom math

- IF uses positional sinusoidal timestep embeddings from `UNet2DConditionModel`
  defaults (`flip_sin_to_cos=true`, `freq_shift=0`) followed by
  `TimestepEmbedding`.
- Stage I default time embedding dim is `block_out_channels[0] * 4` because
  `time_embedding_dim` is null; Stage II L sets `time_embedding_dim=1280`.
- Stage II `noise_level` is converted through the class embedding path with
  `class_embed_type="timestep"` and added to the timestep embedding.
- IF `addition_embed_type="text"` adds text-derived conditioning through
  `TextTimeEmbedding` using projected text context.
- Hand-authored timestep lists in `timesteps.py` are valid custom schedules.
  Dinoml should store custom timesteps as artifact-visible scheduler inputs.
- Dynamic image size affects spatial sequence lengths; prompt length is fixed at
  77 by pipeline tokenization.

## 10. Preprocessing and input packing

Stage I/II prompt processing:

- Clean caption defaults to true in pipeline calls. It requires optional `bs4`
  and `ftfy`; otherwise source logs and falls back.
- T5Tokenizer uses max length 77 despite T5 supporting longer sequences.
- Negative prompt defaults to empty string per batch when CFG is active.
- Prompt/negative embeddings are duplicated with `repeat` and `view`, then
  concatenated along batch outside the UNet.

Stage II image processing:

- PIL image -> float32 numpy `/127.5 - 1.0` -> NCHW torch.
- Numpy image -> stack -> transpose NHWC to NCHW.
- Torch image list supports 3D stack or 4D concat.
- Bilinear upsample to target size, DDPM noising at scalar `noise_level`, then
  channel concat with the current sample.

Stage III:

- `VaeImageProcessor` handles low-res image preprocessing and postprocess.
- Low-res RGB is DDPM-noised and concatenated to latent channels.
- VAE scaling uses `latents / vae.config.scaling_factor` before decode.

## 11. Graph rewrite / lowering opportunities

1. IF added-K/V attention canonicalization

- Source pattern: NCHW -> flatten/transpose -> GroupNorm -> Q projection,
  added text K/V projections, optional image K/V projections, K/V concat, SDPA,
  output projection, transpose/reshape, residual add.
- Replacement: `AddedKVSpatialAttention2D(query_channels=C, text_dim=D,
  heads=C/64, head_dim=64)`.
- Preconditions: NCHW source layout, no attention mask or broadcast mask parity
  implemented, `only_cross_attention=false` for sampled configs,
  `added_kv_proj_dim=D`, no IP-Adapter kwargs.
- Shape equation: image tokens `S=H*W`; K/V tokens `77+S`; output `[B,C,H,W]`.
- Layout constraints: NHWC optimization may keep channels-last internally only
  if GroupNorm axis and flatten order are rewritten and residual consumer stays
  in the same optimized region.
- Failure cases: custom attention processor, non-null masks, `only_cross_attention`
  config changes, LoRA/adapters not materialized into weights.
- Parity test: fixed random NCHW hidden/text tensors through one attention
  module in fp32 and fp16.

2. IF pixel DDPM learned-range step

- Source pattern: UNet output split into noise and variance, CFG on noise,
  concat predicted variance, `DDPMScheduler.step`.
- Replacement: explicit fused pointwise/reduction scheduler op over RGB sample.
- Preconditions: scheduler config exactly `epsilon + learned_range`, matching
  beta/alpha tables and thresholding/clip settings.
- Shape equation: model output `[B,6,H,W]`, sample `[B,3,H,W]`.
- Failure cases: scheduler swap, variance type not learned_range, custom
  thresholding/clipping behavior omitted.
- Parity test: one DDPM step at fixed timestep against Diffusers.

3. Stage II low-res conditioning pack

- Source pattern: preprocess -> bilinear resize -> DDPM add_noise -> concat
  with current sample each step.
- Replacement: precompute noised low-res conditioning once, keep it resident,
  fuse per-step channel concat with UNet input staging.
- Preconditions: fixed image/noise/noise_level for request and source NCHW concat
  order `[sample, upscaled]`.
- Failure cases: img2img/inpaint variant modifies image or mask per step.
- Parity test: compare noised conditioning and first UNet input tensor.

4. Conv/GroupNorm/GELU/scale-shift ResNet fusion

- Source pattern: ResnetBlock2D with GroupNorm, GELU, Conv2d, time scale-shift,
  GroupNorm, GELU, Conv2d, residual scaling.
- Replacement: fused block kernels or scheduled conv+norm+activation lowering.
- Preconditions: inference dropout disabled, static group count 32, source
  scale-shift path admitted.
- Layout constraints: channel-last candidate must rewrite GroupNorm channel axis
  and keep conv weights transformed OIHW -> HWIO only inside a guarded region.
- Parity test: block-level fp32/fp16 comparisons for Stage I XL and Stage II L.

## 12. Kernel fusion candidates

Highest priority:

- IF ResnetBlock2D Conv2d + GroupNorm + GELU + scale-shift conditioning. These
  blocks dominate the pixel-space UNets and exercise very wide channels.
- Added-K/V spatial attention: Q/image K/V/text K/V projections plus SDPA and
  output projection, especially Stage I XL high-channel blocks.
- DDPM learned-range scheduler step and CFG arithmetic over RGB tensors.
- Stage II `add_noise` and low-res conditioning concat staging.

Medium priority:

- Timestep/class/text embedding MLPs for IF conditioning.
- NCHW channel concat/split elision around UNet outputs and scheduler steps.
- Stage III DDIM v-pred and low-res DDPM add-noise fusion, reusing SD scheduler
  work.
- AutoencoderKL decode conv/resnet/upsample fusions for Stage III.

Lower priority:

- Caption-cleaning/tokenization acceleration; this is CPU/data-pipeline work.
- Watermark/safety checker paths, out of runtime scope.
- Rare custom timestep schedule table generation; host metadata is sufficient.

## 13. Runtime staging plan

Stage 1: Parse and admit DeepFloyd Stage I M configs. Load UNet weights and
accept externally supplied T5 prompt embeddings. Keep scheduler in host Python.

Stage 2: Implement one IF ResnetDownsampleBlock2D and one
SimpleCrossAttnDownBlock2D parity test with random NCHW tensors.

Stage 3: Compile one Stage I denoising step for `IF-I-M-v1.0`: RGB input,
UNet output split, CFG, DDPM learned-range step. No safety checker or watermark.

Stage 4: Full Stage I 64x64 loop with custom timesteps supported as explicit
tables. T5 encoder may remain external/cached.

Stage 5: Add Stage II M super-resolution: low-res image preprocessing/noising,
6-channel UNet input, `noise_level` class labels, 256x256 loop.

Stage 6: Add larger L/XL width variants after memory/perf probes. Stage I XL is
the real stress test but not the first admission target.

Stage 7: Reuse separate Stable Diffusion x4/AutoencoderKL support for Stage III.
Do not block IF Stage I/II admission on Stage III latent VAE decode.

Stage 8: Add img2img and inpainting IF variants as separate admission items.

## 14. Parity and validation plan

- Config parsing tests for all five DeepFloyd repos plus x4 upscaler component
  configs under `H:/configs`.
- T5 prompt embed duplication and CFG batch concat parity with fixed dummy
  embeddings.
- Random tensor tests for `AttnAddedKVProcessor2_0` attention blocks at one
  Stage I and one Stage II resolution/channel size.
- ResnetBlock2D scale-shift parity, including Stage II `skip_time_act=true`.
- One Stage I UNet forward parity at small M width if weights are locally
  available; otherwise block parity first.
- DDPM `add_noise` parity for Stage II low-res conditioning.
- DDPM learned-range one-step parity with thresholding enabled.
- Short deterministic Stage I loop parity with fixed prompt embeddings and
  random seed.
- Stage II first-step parity: verify upscaled/noised image, 6-channel model
  input, class labels, and scheduler output.
- Stage III smoke should be covered by SD x4 upscaler tests: CLIP prompt
  embeddings, DDIM v-pred, VAE decode.

Suggested tolerances: fp32 scheduler/block tests `rtol=1e-5, atol=1e-6`; fp16
UNet block tests initially `rtol=2e-2, atol=2e-2`, tightening after fused kernels
are stable.

## 15. Performance probes

- T5 encoder throughput and memory separately from cached prompt-embed pipeline
  throughput.
- One Stage I denoising step by variant M/L/XL and batch size.
- Stage I full loop by timestep schedule: `smart27`, `smart50`, `smart100`, and
  default 100.
- Stage II noising/preprocess cost versus UNet denoising cost at 256x256.
- Added-K/V attention backend comparison: eager fallback, SDPA, and Dinoml
  fused provider.
- Conv/resnet versus attention time split inside Stage I XL and Stage II L.
- VRAM and temporary workspace for wide Stage I XL blocks.
- Stage III x4 latent denoising and VAE decode separately, reusing SD x4 probes.

No benchmark measurements were run for this audit.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items:

- IF img2img: `pipeline_if_img2img.py` and
  `pipeline_if_img2img_superresolution.py`; strength/noising/timestep slicing.
- IF inpainting: `pipeline_if_inpainting.py` and
  `pipeline_if_inpainting_superresolution.py`; mask image preprocessing and
  masked-image contracts.
- Stage III SD x4 upscaler: `pipeline_stable_diffusion_upscale.py`;
  latent-space UNet, DDIM v-pred scheduler, low-res DDPM, AutoencoderKL decode.
- LoRA/runtime adapter mutation for IF and x4: loader mixins plus attention
  processor replacement.
- Textual inversion for Stage III Stable Diffusion family.
- Rare scheduler swaps or custom timestep schedule admission beyond DDPM for
  IF and DDIM for x4.

Ignored/out of scope for this audit:

- Safety checker, NSFW filtering, and watermarking runtime behavior.
- Training, losses, dropout, and gradient checkpointing paths.
- Multi-GPU/context parallel, callbacks, and interactive interrupt behavior.
- XLA, NPU, MPS, Flax, and ONNX branches.
- DreamBooth training scripts and conversion utilities.

## 17. Final implementation checklist

- [ ] Parse DeepFloyd Stage I/II model_index, UNet, scheduler, text encoder, and tokenizer configs.
- [ ] Admit pixel-space RGB diffusion contract separately from latent Stable Diffusion.
- [ ] Load IF UNet weights for `IF-I-M-v1.0` first.
- [ ] Accept externally supplied T5 prompt and negative prompt embeddings.
- [ ] Implement ResnetBlock2D scale-shift block parity for IF configs.
- [ ] Implement AddedKV spatial attention parity for SimpleCrossAttn blocks.
- [ ] Implement DDPM learned-range scheduler step with thresholding/clip settings.
- [ ] Implement DDPM `add_noise` for Stage II low-res image conditioning.
- [ ] Add one Stage I denoising step parity test.
- [ ] Add Stage I short-loop parity with custom timestep tables.
- [ ] Add Stage II 6-channel super-resolution first-step parity.
- [ ] Reuse SD x4/AutoencoderKL work for Stage III rather than folding it into IF Stage I/II admission.
- [ ] Benchmark Stage I M/L/XL one-step UNet, attention, and resnet time splits.
