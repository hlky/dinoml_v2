# Diffusers Stable Diffusion 2.x Operator and Integration Report

Target slug: `stable_diffusion_2_x`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Family target: Stable Diffusion 2.x text-to-image.
  Representative sd2-community config sources inspected:
    sd2-community/stable-diffusion-2
    sd2-community/stable-diffusion-2-base
    sd2-community/stable-diffusion-2-1
    Manojb/stable-diffusion-2-1-base
    sd2-community/stable-diffusion-2-depth
    sd2-community/stable-diffusion-2-inpainting
    stabilityai/stable-diffusion-x4-upscaler

Config sources:
  H:/configs/sd2-community/stable-diffusion-2/
  H:/configs/sd2-community/stable-diffusion-2-base/
  H:/configs/sd2-community/stable-diffusion-2-1/
  H:/configs/Manojb/stable-diffusion-2-1-base/
  H:/configs/sd2-community/stable-diffusion-2-depth/
  H:/configs/sd2-community/stable-diffusion-2-inpainting/
  H:/configs/stabilityai/stable-diffusion-x4-upscaler/

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_depth2img.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_upscale.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_latent_upscale.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip_img2img.py

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
  diffusers/src/diffusers/schedulers/scheduling_pndm.py
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  Loader/extension anchors:
    diffusers/src/diffusers/loaders/lora_pipeline.py
    diffusers/src/diffusers/loaders/textual_inversion.py
    diffusers/src/diffusers/loaders/ip_adapter.py
    diffusers/src/diffusers/models/controlnets/controlnet.py
    diffusers/src/diffusers/models/adapter.py
    diffusers/src/diffusers/pipelines/controlnet/
    diffusers/src/diffusers/pipelines/t2i_adapter/
    diffusers/src/diffusers/pipelines/deprecated/stable_diffusion_gligen/

External component configs inspected:
  OpenCLIP CLIPTextModel/CLIPTokenizer configs from the sd2-community sources above.
  DPTForDepthEstimation/DPTImageProcessor configs from sd2-community/stable-diffusion-2-depth.

Any missing files or assumptions:
  Per project steering, SD2.x is a special case because the original
  StabilityAI repos were removed; use https://huggingface.co/sd2-community for
  SD2.x component configs in this audit. `Manojb/stable-diffusion-2-1-base`
  remains a supplemental 2.1-base config until an sd2-community 2.1-base config
  is added to the local cache. Text encoder internals live in Transformers and
  are treated as an external prompt-embedding stage for the first Dinoml slice.
  Safety/NSFW, callbacks/interrupt, training/loss/dropout/gradient checkpointing,
  distributed paths, XLA/NPU/MPS, Flax, and ONNX are out of scope.
```

## 2. Pipeline and component graph

The base SD 2.x text-to-image path uses `StableDiffusionPipeline`, the same
Diffusers pipeline class as SD 1.x, but SD2 checkpoints swap the text encoder to
an OpenCLIP-sized `CLIPTextModel` with hidden size 1024 and use a UNet configured
for `cross_attention_dim=1024`, `use_linear_projection=true`, and v-prediction
for the common 768 checkpoints. The pipeline offload order is
`text_encoder->image_encoder->unet->vae`; `image_encoder` is relevant to
IP-Adapter, not the base first slice.

```text
prompt strings / prompt_embeds
  -> CLIPTokenizer + CLIPTextModel, or externally supplied prompt embeddings
  -> duplicate/concat negative and positive embeddings for CFG
  -> initialize latent noise [B,4,H/8,W/8]
  -> denoising loop:
       cat CFG latents
       -> scheduler.scale_model_input
       -> UNet2DConditionModel(latents, timestep, encoder_hidden_states)
       -> CFG arithmetic and optional guidance_rescale
       -> scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> VaeImageProcessor postprocess
```

Required first-slice components:

- Required: external `prompt_embeds`, UNet denoiser, scheduler tables/step state,
  VAE decode, image postprocess.
- Required for text encoder integration: `CLIPTokenizer`, `CLIPTextModel`
  hidden states shaped `[B,77,1024]`.
- Required for family accounting, but separable: VAE encode for img2img,
  inpaint, depth2img, and upscaling variants.

Independently cacheable stages:

- Prompt and negative prompt embeddings after tokenization/text encoding.
- Scheduler timesteps/tables per scheduler config and step count.
- Initial latent noise when caller supplies `latents`.
- VAE decode can be tested as a standalone stage.

Separate candidate reports:

| Candidate | Primary classes/files | Pipeline delta from base text-to-image |
| --- | --- | --- |
| `sd2_lora_textual_inversion_adapters` | `StableDiffusionPipeline` inherits `TextualInversionLoaderMixin`, `StableDiffusionLoraLoaderMixin`, `IPAdapterMixin`; anchors `loaders/textual_inversion.py`, `loaders/lora_pipeline.py`, `loaders/peft.py` | Tokenizer/text-encoder embedding mutation plus UNet/text-encoder adapter weights. Same runtime surface as SD1 but with 1024-wide text states. |
| `sd2_ip_adapter` | `IPAdapterMixin`, `IPAdapterAttnProcessor*`, `MultiIPAdapterImageProjection` | Adds image encoder/image embeds, projection layers, added K/V attention branches, scales, and masks. |
| `sd2_controlnet` | `StableDiffusionControlNetPipeline`, `StableDiffusionControlNetImg2ImgPipeline`, `StableDiffusionControlNetInpaintPipeline`, `ControlNetModel`, `MultiControlNetModel` | Adds conditioning image preprocessing and ControlNet down/mid residuals into the SD2 UNet shape. |
| `sd2_t2i_adapter` | `StableDiffusionAdapterPipeline`, `T2IAdapter`, `MultiAdapter`, `FullAdapter`, `LightAdapter` | Adds adapter feature pyramids/residual tensors. |
| `sd2_gligen` | Deprecated `StableDiffusionGLIGENPipeline`, `StableDiffusionGLIGENTextImagePipeline`, `GLIGENTextBoundingboxProjection` | Adds grounded phrases/boxes/images and gated attention branches. |
| `sd2_img2img` | `StableDiffusionImg2ImgPipeline` | VAE-encodes input image, slices timesteps by `strength`, adds noise, then denoises. |
| `sd2_inpaint` | `StableDiffusionInpaintPipeline`; sampled config `sd2-community/stable-diffusion-2-inpainting` | Adds mask and masked-image latents; sampled SD2 inpaint UNet uses `in_channels=9`. |
| `sd2_depth2img` | `StableDiffusionDepth2ImgPipeline`, `DPTForDepthEstimation`, `DPTImageProcessor`; sampled config `sd2-community/stable-diffusion-2-depth` | Adds DPT depth prediction or provided depth map; UNet input is `[latents, depth_mask]` with `in_channels=5`. |
| `sd2_upscale` | `StableDiffusionUpscalePipeline`, `StableDiffusionLatentUpscalePipeline`; sampled `stabilityai/stable-diffusion-x4-upscaler` | Adds low-resolution image/noise-level conditioning, `low_res_scheduler`, class labels, and `in_channels=7` UNet. |
| `sd2_unclip` | `StableUnCLIPPipeline`, `StableUnCLIPImg2ImgPipeline` | Adds prior/image-embedding conditioning and image normalizer; separate from text-to-image SD2 base. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config source | Pipeline | UNet sample | UNet in/out | block channels | cross dim | attention head dim | text hidden | VAE sample | scheduler | prediction |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `sd2-community/stable-diffusion-2` | `StableDiffusionPipeline` | 96 | 4 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 768 | DDIM | v-prediction |
| `sd2-community/stable-diffusion-2-base` | `StableDiffusionPipeline` | 64 | 4 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 512 | DDIM | omitted, DDIM default epsilon in source |
| `sd2-community/stable-diffusion-2-1` | `StableDiffusionPipeline` | 96 | 4 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 768 | DDIM | v-prediction |
| `Manojb/stable-diffusion-2-1-base` | `StableDiffusionPipeline` | 64 | 4 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 768 | PNDM | epsilon |
| `sd2-community/stable-diffusion-2-depth` | `StableDiffusionDepth2ImgPipeline` | 32 | 5 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 256 | PNDM | epsilon |
| `sd2-community/stable-diffusion-2-inpainting` | `StableDiffusionInpaintPipeline` | 64 | 9 / 4 | 320/640/1280/1280 | 1024 | 5/10/20/20 | 1024 | 512 | PNDM | omitted, PNDM default epsilon |
| `stabilityai/stable-diffusion-x4-upscaler` | `StableDiffusionUpscalePipeline` | 128 | 7 / 4 | 256/512/512/1024 | 1024 | 8 | 1024 | 256 | DDIM + low-res DDPM | v-prediction main, epsilon low-res |

Base text encoder dimensions from component configs:

| Field | Value | Source |
| --- | ---: | --- |
| class | `CLIPTextModel` / `CLIPTokenizer` | model index |
| hidden size | 1024 | text encoder config |
| projection dim | 512 | text encoder config |
| layers | 23 | text encoder config |
| heads | 16 | text encoder config |
| max positions | 77 | text encoder config |
| vocab size | 49408 | text encoder config |

UNet/VAE common dimensions:

| Component | Field | Value |
| --- | --- | --- |
| UNet base/full | `down_block_types` | `CrossAttnDownBlock2D` x3, `DownBlock2D` |
| UNet base/full | `up_block_types` | `UpBlock2D`, `CrossAttnUpBlock2D` x3 |
| UNet base/full | `layers_per_block` | 2 |
| UNet base/full | `use_linear_projection` | true |
| UNet base/full | `norm_num_groups` | 32 |
| VAE base/full | latent channels | 4 |
| VAE base/full | scale factor | omitted in configs, effective AutoencoderKL default `0.18215` |
| VAE base/full | blocks | 128/256/512/512, 2 layers per block |
| VAE x4 upscaler | scale factor | `0.08333`, effective pipeline constructor repairs stale `0.08333` cases |

Scheduler support:

- The base pipeline types `scheduler` as `KarrasDiffusionSchedulers`, so SD2 is
  a broad scheduler-swap family like SD1, not a DDIM-only or PNDM-only runtime.
- The sampled SD2/SD2.1 768 sd2-community configs use DDIM with
  `prediction_type=v_prediction`.
- The sampled supplemental SD2.1-base config uses PNDM with
  `prediction_type=epsilon`; this may reflect config-source/default drift rather
  than the whole family contract.
- Recommended first Dinoml scheduler slice: DDIM v-prediction for SD2/2.1 768
  parity, plus PNDM epsilon/v-prediction as a follow-up because several SD2
  variant configs default to PNDM.

## 3a. Family variation traps

- SD2 is not just SD1 with bigger images: prompt embeddings are 1024-wide and
  the UNet uses `cross_attention_dim=1024`.
- `attention_head_dim=[5,10,20,20]` means the block head counts are still
  64 heads at channels 320/640/1280/1280 when constructed as
  `out_channels // attention_head_dim`. Do not copy SD1's head-dim-8 assumption.
- `use_linear_projection=true` changes the `Transformer2DModel` projection path
  inside UNet attention blocks. Include it in parity tests.
- The 768 full models use latent sample size 96 and VAE sample size 768; base
  models use latent sample size 64 for 512 output. The convolutional operator
  shape is otherwise similar.
- Scheduler prediction type varies across configs: common SD2/2.1 full configs
  are v-prediction, while some base/depth/inpaint configs use epsilon.
- Depth2img, inpaint, and upscaling change UNet input channels to 5, 9, and 7
  respectively. These are separate candidates, not base first-slice ops.
- Source tensors are NCHW. NHWC/channel-last is only a guarded optimization for
  local conv/ResNet/VAE islands; GroupNorm, channel concat, posterior chunk,
  scheduler broadcasting, and VAE scaling are axis-sensitive.
- Pipeline constructors repair stale scheduler `steps_offset` and `clip_sample`
  fields for base/img2img, and inpaint repairs `skip_prk_steps` for stale PNDM
  configs. Artifact loading must preserve those compatibility mutations.
- Many `UNet2DConditionModel` branches are inactive for vanilla SD2 text-to-image:
  class embeddings, addition embeddings, GLIGEN, IP-Adapter, ControlNet residuals,
  T2I residuals, and `time_cond_proj_dim`.

## 4. Runtime tensor contract

For 768 output with CFG enabled:

| Boundary | Tensor | Source layout | Candidate optimized layout | Shape |
| --- | --- | --- | --- | --- |
| prompt embeddings | `prompt_embeds` | `[B,77,1024]` | same | `[2B,77,1024]` after CFG concat |
| latent state | `latents` | NCHW | NHWC guarded candidate | `[B,4,96,96]` |
| UNet input | `latent_model_input` | NCHW | NHWC guarded candidate | `[2B,4,96,96]` |
| UNet output | `noise_pred` | NCHW | NHWC guarded candidate | `[2B,4,96,96]` before CFG chunk |
| scheduler output | `latents` | NCHW | must match loop layout | `[B,4,96,96]` |
| VAE decode input | `latents / 0.18215` | NCHW | NHWC guarded candidate | `[B,4,96,96]` |
| VAE output | decoded image | NCHW | NHWC guarded candidate | `[B,3,768,768]` |

For 512 base, replace spatial latent shape with `[B,4,64,64]` and decoded image
with `[B,3,512,512]`.

Variant tensor contracts:

- Depth2img: preprocessed image is VAE-encoded/noised to `[B,4,h,w]`; depth map
  is predicted or supplied, interpolated to `[B,1,h,w]`, normalized to `[-1,1]`,
  duplicated for CFG, and concatenated with latents on channel dim before UNet.
- Inpaint: UNet input is usually `[latent_model_input, mask, masked_image_latents]`
  as 4 + 1 + 4 channels.
- x4 upscaler: low-res image is resized/preprocessed, noised by a
  `DDPMScheduler`, repeated to batch, and concatenated with latent input; the
  UNet receives `class_labels=noise_level`.

CPU/data-pipeline work:

- Tokenization, CLIP text encoding, DPT depth estimation, image resize/crop/PIL
  conversion, and safety checker are outside the first compiled slice.

GPU/runtime work:

- Denoising UNet forward, CFG/guidance rescale arithmetic, scheduler step,
  VAE encode/decode, and variant channel concat/noising paths.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCHW latent/image tensors with guarded NHWC conv islands.
- `cat` prompt embeddings and CFG latents on batch dim.
- `chunk(2)` on batch dim for CFG.
- Channel concat for depth/inpaint/upscale variants.
- Skip tuple push/pop and spatial concat in UNet up path.
- Broadcast scheduler coefficients over `[B,C,H,W]`.
- Reductions over all non-batch dims for `guidance_rescale`.

### Convolution/downsample/upsample ops

- Base/full UNet `conv_in`: `Conv2d(4 -> 320, 3x3, padding=1)`.
- Depth UNet `conv_in`: `Conv2d(5 -> 320, 3x3, padding=1)`.
- Inpaint UNet `conv_in`: `Conv2d(9 -> 320, 3x3, padding=1)`.
- x4 upscaler `conv_in`: `Conv2d(7 -> 256, 3x3, padding=1)`.
- ResnetBlock2D two-conv blocks, optional 1x1 shortcut.
- Downsample2D stride/conv path and Upsample2D nearest/interpolate + conv path.
- AutoencoderKL encoder/decoder convs, ResNet blocks, up/downsample, quant and
  post-quant 1x1 convs for common SD2 VAEs.

### GEMM/linear ops

- Sinusoidal timestep embedding followed by TimestepEmbedding MLP.
- ResNet time projection to channel bias or scale/shift.
- Spatial transformer projections; with `use_linear_projection=true`, the
  projected token path must match `Transformer2DModel` source.
- Cross-attention Q from latent tokens and K/V from 1024-wide text tokens.
- GEGLU/GELU feed-forward MLPs inside `BasicTransformerBlock`.
- x4 upscaler class/noise-level embedding path.

### Attention primitives

- Noncausal latent self-attention and cross-attention in UNet transformer blocks.
- Text sequence length 77, text feature width 1024.
- Heads per common SD2 block are 64, with head dims 5, 10, 20, 20 by resolution.
- Eager/SDPA `Attention` processors define parity; xFormers/fused projections
  are optimization surfaces.

### Normalization and adaptive conditioning

- GroupNorm over source channel axis, usually 32 groups.
- LayerNorm over token hidden dimensions.
- SiLU activations.
- Optional ResNet time scale-shift if config enables it; inactive in sampled
  base/full configs.

### Scheduler and guidance arithmetic

- Latent initialization with `randn_tensor * scheduler.init_noise_sigma`.
- DDIM v-prediction conversion for common SD2/2.1 full configs.
- PNDM epsilon/v-prediction state for base/depth/inpaint sampled defaults.
- CFG arithmetic: `uncond + guidance_scale * (text - uncond)`.
- Optional guidance rescale std reduction and blend.

### VAE/postprocessing ops

- Divide by VAE `scaling_factor` before decode.
- VAE encode and posterior sample/mode for img2img/depth/inpaint/upscale.
- Image postprocess denormalization to `[0,1]`.

## 6. Denoiser/model breakdown

Base SD2 UNet forward:

```text
sample [B,4,H,W]
-> optional center_input_sample                          # inactive in sampled configs
-> Timesteps + TimestepEmbedding
-> Conv2d(4 -> 320, 3x3)
-> CrossAttnDownBlock2D x3:
     ResnetBlock2D x2
     Transformer2DModel/BasicTransformerBlock with self + cross attention
     optional downsample
-> DownBlock2D x1:
     ResnetBlock2D x2
-> UNetMidBlock2DCrossAttn:
     ResnetBlock2D -> BasicTransformerBlock -> ResnetBlock2D
-> UpBlock2D / CrossAttnUpBlock2D:
     concat skip features, ResnetBlock2D, cross-attention where configured,
     optional upsample
-> GroupNorm -> SiLU -> Conv2d(320 -> 4, 3x3)
```

ResnetBlock2D active path:

```text
GroupNorm -> SiLU -> Conv2d
time embedding projection -> add to channels
GroupNorm -> SiLU -> Conv2d
residual add / output_scale_factor
```

BasicTransformerBlock active path:

```text
spatial map -> token sequence
LayerNorm -> self-attention -> residual
LayerNorm -> cross-attention(text K/V) -> residual
LayerNorm -> FeedForward(GEGLU/GELU) -> residual
tokens -> spatial map
```

Variant deltas:

- Depth2img concatenates a single-channel normalized depth map to the latent
  input before the same UNet block structure.
- Inpaint concatenates mask and masked image latents before the same block
  structure.
- x4 upscaler changes base channels to 256/512/512/1024, starts with a
  `DownBlock2D`, adds low-res image channels, and passes `class_labels=noise_level`.

## 7. Attention requirements

Required for base SD2:

- Noncausal self-attention over spatial latent tokens.
- Noncausal cross-attention from latent tokens to 77 OpenCLIP text tokens.
- No KV cache and no causal mask.
- No QK norm or RoPE in the base UNet path.
- `Attention` and `BasicTransformerBlock` in `attention.py` plus
  `attention_processor.py` define the parity path.
- Source supports native SDPA through `AttnProcessor2_0` when available and
  eager processors otherwise.
- `UNet2DConditionModel.fuse_qkv_projections()` exists but rejects added-KV
  processors; unfused projections should be first parity.

Shape notes:

- For common SD2 channels 320/640/1280/1280 and
  `attention_head_dim=[5,10,20,20]`, heads are 64 at each resolution.
- Cross-attention K/V input width is 1024, not 768.
- Query sequence length depends on latent resolution: for 768 output, 96x96 at
  the first resolution and lower resolutions after downsampling.

Flash-style constraints:

- Base mask-free self/cross attention is a plausible Dinoml flash-style provider
  candidate if dtype, head dimension, sequence length, dropout=0, and exact
  scaling semantics pass provider guards.
- Head dims 5/10/20 are unusual for flash kernels that prefer standard multiples;
  the provider must guard or fall back to SDPA/eager-equivalent math.
- Added K/V branches from IP-Adapter or other custom processors are separate
  candidates and should disable the base flash rewrite unless supported.
- Fused QKV/KV projection is an optimization, not a source default.

## 8. Scheduler and denoising-loop contract

Base loop:

```text
scheduler.set_timesteps(num_inference_steps, device)
latents = randn([B,4,H/8,W/8]) * scheduler.init_noise_sigma
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = unet(latent_model_input, t, prompt_embeds)
  if CFG:
    uncond, text = chunk(noise_pred, 2)
    noise_pred = uncond + guidance_scale * (text - uncond)
    optional guidance_rescale
  latents = scheduler.step(noise_pred, t, latents)
```

DDIM v-prediction details:

- `set_timesteps` constructs a descending inference timetable.
- `step` converts model output to predicted original sample differently for
  `epsilon`, `sample`, and `v_prediction`.
- For v-prediction, the model output is velocity-like and must be converted with
  alpha/beta products before the previous sample update.

PNDM details:

- Maintains PRK/PLMS timestep tables, `counter`, and `ets` history.
- Supports `epsilon` and `v_prediction`.
- Compatibility repairs in constructors matter for old configs.

Host/runtime split:

- Keep loop iteration, scheduler object state, and custom timestep/sigma handling
  host-visible first.
- Compile one UNet step, CFG/guidance rescale, and one scheduler step as bounded
  runtime kernels.
- Recommended first parity is DDIM v-prediction for 768 SD2/2.1 configs; add
  PNDM epsilon/v-prediction to cover base/depth/inpaint configs.

## 9. Position, timestep, and custom math

Required custom math:

- Sinusoidal timestep embedding with `flip_sin_to_cos` and `freq_shift` from
  UNet config/source defaults.
- TimestepEmbedding MLP feeding ResNet blocks.
- DDIM v-prediction conversion and PNDM epsilon/v-prediction conversion.
- CFG and optional guidance rescale:

```python
std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
```

No RoPE, learned 2D position embedding, patchify, or joint attention is required
for base SD2 UNet. DPT depth estimation and x4 upscaler low-res noise-level
conditioning are variant candidate math.

Precompute candidates:

- Prompt embeddings per prompt and negative prompt.
- Timestep embeddings per scheduler table.
- Scheduler alpha/sigma tables per config/step count.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenization and OpenCLIP text encoding.
- Prompt truncation handling at CLIP max length 77.
- Image loading, resize/crop, and PIL/NumPy conversion.
- DPT depth prediction for depth2img if the caller does not supply `depth_map`.

GPU/runtime work:

- Accept external `prompt_embeds [B,77,1024]` and negative embeddings.
- Duplicate embeddings for `num_images_per_prompt`.
- Concatenate negative/positive embeddings on batch dim for CFG.
- Generate or accept latent noise `[B,4,H/8,W/8]`.
- Decode latents through VAE.

Variant preprocessing:

- Img2img/depth2img encode source image with VAE, select timesteps by `strength`,
  and add scheduler noise.
- Depth2img predicts or receives depth, interpolates to latent spatial size,
  min/max normalizes to `[-1,1]`, and channel-concats with latents.
- Inpaint prepares mask latents and masked image latents, then channel-concats.
- x4 upscaler noises the low-res image with `low_res_scheduler.add_noise` and
  passes the integer noise level as UNet class labels.

## 11. Graph rewrite / lowering opportunities

### Rewrite: NCHW UNet/VAE conv island -> guarded NHWC island

Source pattern:

```text
Conv2d / GroupNorm / SiLU / residual add / upsample / downsample
```

Replacement:

```text
NCHW boundary -> NHWC local region -> NCHW boundary
```

Preconditions:

- All producers/consumers in the region are local and layout-aware.
- GroupNorm channel axis is rewritten from dim 1 to last channel.
- Channel concat/split for depth/inpaint/upscale is either outside the region or
  rewritten to last-channel concat.
- VAE posterior `chunk(..., dim=1)` and scale/shift broadcasting have explicit
  axis rewrites if included.

Weight transform:

```python
w_hwio = w_oihw.permute(2, 3, 1, 0)
```

Failure cases:

- Attention flatten/reshape regions assuming NCHW are accidentally captured.
- Scheduler/guidance code assumes channel axis for reductions or broadcasting.
- Variant inputs concatenate along source dim 1 without a layout rewrite.

Parity sketch:

- Compare one ResnetBlock2D and one full down/up block at 96, 64, 48, 32, 16,
  and 8 latent resolutions.

### Rewrite: SD2 attention canonicalization

Source pattern:

```text
spatial flatten -> Q/K/V linear projections -> attention -> output projection -> reshape
```

Replacement:

```text
FlattenSpatial -> GEMM Q and GEMM K/V(text) -> Attention -> GEMM out -> ReshapeSpatial
```

Preconditions:

- Processor is eager or SDPA-compatible without added K/V.
- Text hidden width is 1024 and sequence length is plan-visible.
- Head dim provider supports 5/10/20 or falls back.
- `use_linear_projection=true` projection path is represented exactly.

Failure cases:

- IP-Adapter/custom attention processors, active LoRA mutation not folded into
  weights, masks not represented, or unsupported flash head dimensions.

Parity sketch:

- Compare one `BasicTransformerBlock` with random `[B,H*W,C]` latent tokens and
  `[B,77,1024]` text embeddings.

### Rewrite: CFG batched UNet -> explicit two-call option

Preconditions:

- UNet is pure across batch entries.
- Positive/negative prompt embeddings and latents are explicit.

Replacement options:

```text
batched: cat latents/prompts -> one UNet -> chunk -> CFG
explicit: two UNet calls -> CFG
```

Failure cases:

- Memory planning, callback mutation, IP-Adapter masks, or variant side inputs
  differ across positive/negative branches.

### Rewrite: scheduler scalar update fusion

Preconditions:

- Scheduler family, prediction type, timestep index, and history state are
  explicit.
- First slice fixes DDIM v-prediction or a specific PNDM mode.

Replacement:

```text
table lookups + prediction conversion + previous sample update as fused elementwise
```

Failure cases:

- Custom timesteps/sigmas, eta/noise stochastic branch, or PNDM history not in
  artifact-visible state.

## 12. Kernel fusion candidates

Highest priority:

- GroupNorm + SiLU + Conv2d in UNet/VAE ResNet blocks.
- Cross-attention Q/K/V projection + attention + output projection with 1024
  text K/V and guarded head-dim support.
- DDIM v-prediction scheduler step and CFG arithmetic.
- VAE decode conv/resnet/up blocks for 512 and 768 outputs.

Medium priority:

- NHWC conv islands across consecutive UNet and VAE blocks.
- Guidance rescale reduction/elementwise fusion.
- Time embedding MLP and ResNet time-bias fusion.
- PNDM PRK/PLMS stateful step kernels.
- x4 upscaler low-res noising and class/noise embedding path.

Lower priority:

- Fused QKV/KV projection mutation parity.
- DPT depth estimator compilation.
- IP-Adapter added-KV attention, ControlNet residual paths, T2I adapters, and
  GLIGEN grounded attention.

## 13. Runtime staging plan

Stage 1: Parse SD2 component configs and reconcile omitted defaults, especially
VAE `scaling_factor`, scheduler `prediction_type`, and UNet attention dimensions.

Stage 2: Load UNet and VAE weights. Accept external `[B,77,1024]` prompt
embeddings; stub tokenizer/text encoder.

Stage 3: Add operator parity for Conv2d, GroupNorm, SiLU, Downsample2D,
Upsample2D, GEGLU/GELU, timestep embedding, and SD2 attention.

Stage 4: Compile one ResnetBlock2D and one BasicTransformerBlock with
`cross_attention_dim=1024`, `use_linear_projection=true`, and head dims 5/10/20.

Stage 5: Run one UNet block slice, then full UNet forward at 64x64 and 96x96
latent sizes.

Stage 6: Add DDIM v-prediction one-step parity and CFG/guidance rescale.

Stage 7: Full denoising loop with scheduler in Python and compiled UNet step.

Stage 8: Add AutoencoderKL decode, then end-to-end latent-to-image smoke.

Stage 9: Add PNDM epsilon/v-prediction parity and variant reports for img2img,
inpaint, depth2img, upscaling, ControlNet, IP-Adapter, LoRA/textual inversion.

Stage 10: Add guarded NHWC conv islands and attention/provider fusions.

## 14. Parity and validation plan

- Config parser tests for SD2 768, SD2 base, SD2.1 768, SD2.1 base, depth,
  inpaint, and x4 upscaler configs.
- Operator parity for GroupNorm axis behavior in NCHW and candidate NHWC.
- ResnetBlock2D parity at channels 320, 640, 1280.
- BasicTransformerBlock parity with `cross_attention_dim=1024`,
  `use_linear_projection=true`, and head dims 5/10/20.
- UNet forward parity at latent shapes `[B,4,64,64]` and `[B,4,96,96]`.
- DDIM `set_timesteps` and v-prediction `step` parity.
- PNDM epsilon/v-prediction parity for variant/default coverage.
- CFG arithmetic and guidance rescale parity.
- VAE decode parity for `[B,4,64,64] -> [B,3,512,512]` and
  `[B,4,96,96] -> [B,3,768,768]`.
- VAE encode parity for img2img/depth/inpaint readiness.
- Depth map concat parity for `[B,5,h,w]` depth2img UNet input.
- Inpaint concat parity for `[B,9,h,w]` UNet input.
- x4 upscaler low-res scheduler noising and `[B,7,h,w]` UNet input parity.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per kernel.

## 15. Performance probes

- One UNet forward by output size 512 vs 768, batch, dtype, and CFG mode.
- UNet conv/resnet time vs attention time with 1024-wide text K/V.
- Attention backend comparison for head dims 5/10/20/8.
- DDIM v-prediction scheduler + CFG overhead as a percentage of one step.
- PNDM overhead and history-state cost.
- VAE decode throughput at 512 and 768.
- NCHW faithful path vs guarded NHWC conv islands.
- Batched CFG one-call vs explicit two-call memory/latency.
- Variant probes: depth/inpaint channel-concat overhead and x4 upscaler low-res
  image/noise-level overhead.
- VRAM and temporary/workspace usage across UNet + VAE.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `sd2_lora_textual_inversion_adapters`: `StableDiffusionLoraLoaderMixin`,
  `TextualInversionLoaderMixin`, PEFT adapter paths, 1024-wide text encoder.
- `sd2_ip_adapter`: `IPAdapterMixin`, `IPAdapterAttnProcessor*`, image
  projection classes, added K/V and masks.
- `sd2_controlnet`: ControlNet pipelines and `ControlNetModel` with SD2
  cross-attention dimensions.
- `sd2_t2i_adapter`: T2I adapter feature pyramid residuals.
- `sd2_gligen`: deprecated grounded generation branch.
- `sd2_img2img`: VAE encode, strength timestep slicing, image latent noising.
- `sd2_inpaint`: mask/masked image latents and 9-channel UNet contract.
- `sd2_depth2img`: DPT depth preprocessing and 5-channel UNet contract.
- `sd2_upscale`: x4 upscaler low-res image/noise-level conditioning and
  7-channel UNet contract.
- `sd2_unclip`: Stable unCLIP prior/image-conditioning path.
- `sd2_scheduler_matrix`: DDIM/PNDM/Euler/LMS/DPM-compatible scheduler swaps,
  especially v-prediction coverage.

Ignored/out of scope for this audit unless explicitly selected:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX pipeline variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse SD2 component configs and reconcile source defaults.
- [ ] Load UNet and VAE weights.
- [ ] Accept external `[B,77,1024]` prompt and negative prompt embeddings.
- [ ] Implement Conv2d/GroupNorm/SiLU/ResnetBlock2D parity.
- [ ] Implement Downsample2D and Upsample2D parity.
- [ ] Implement timestep embedding and ResNet time conditioning.
- [ ] Implement `use_linear_projection=true` BasicTransformerBlock parity.
- [ ] Implement cross-attention with 1024-wide text K/V and head dims 5/10/20.
- [ ] Implement CFG concat/chunk/arithmetic and guidance rescale.
- [ ] Implement DDIM v-prediction scheduler step.
- [ ] Add PNDM epsilon/v-prediction follow-up scheduler parity.
- [ ] Compile one UNet block slice at 64 and 96 latent sizes.
- [ ] Add one-step denoising parity.
- [ ] Implement AutoencoderKL decode for 512 and 768 outputs.
- [ ] Implement or separately track AutoencoderKL encode for variants.
- [ ] Create separate candidate reports for LoRA/textual inversion/adapters,
      IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img,
      upscaling, unCLIP, and scheduler matrix.
- [ ] Add guarded NHWC conv-island rewrite and parity tests.
- [ ] Benchmark UNet, scheduler/guidance, and VAE decode separately.
