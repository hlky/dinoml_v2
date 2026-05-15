# Diffusers Latent Consistency Models Operator and Integration Report

Target slug: `latent_consistency_models`

Runtime scope: non-deprecated Diffusers `latent_consistency_models` text-to-image pipeline as the base slice, with the sibling img2img pipeline, LCM LoRA/distilled UNet-only repos, SDXL/SSD LCM variants, IP-Adapter, LoRA, textual inversion, and broader Stable Diffusion variants inventoried as separate candidates.

Ignored per task scope: XLA/NPU/MPS, Flax/ONNX, safety/NSFW filtering, training/loss/dropout/gradient checkpointing, callbacks/interactive mutation, and multi-GPU/context-parallel paths.

## 1. Source basis

```text
Diffusers commit/version:
  diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Base LCM pipeline examples:
    SimianLuo/LCM_Dreamshaper_v7
    echarlaix/tiny-random-latent-consistency
    optimum-intel-internal-testing/tiny-random-latent-consistency
  Related LCMScheduler / distilled-weight references:
    rupeshs/LCM-runwayml-stable-diffusion-v1-5
    Disty0/LCM_SoteMix
    Lykon/dreamshaper-8-lcm
    latent-consistency/lcm-sdxl
    latent-consistency/lcm-ssd-1b
    latent-consistency/lcm-lora-sdv1-5
    latent-consistency/lcm-lora-sdxl
    PixArt-alpha/PixArt-LCM-XL-2-1024-MS

Config sources:
  Local cache model indexes:
    H:/configs/SimianLuo/LCM_Dreamshaper_v7/model_index.json
    H:/configs/echarlaix/tiny-random-latent-consistency/model_index.json
    H:/configs/optimum-intel-internal-testing/tiny-random-latent-consistency/model_index.json
    H:/configs/rupeshs/LCM-runwayml-stable-diffusion-v1-5/model_index.json
    H:/configs/Disty0/LCM_SoteMix/model_index.json
    H:/configs/Lykon/dreamshaper-8-lcm/model_index.json
    H:/configs/PixArt-alpha/PixArt-LCM-XL-2-1024-MS/model_index.json
  Official raw Hugging Face configs inspected transiently:
    SimianLuo/LCM_Dreamshaper_v7: model_index.json, unet/config.json,
      scheduler/scheduler_config.json, vae/config.json,
      text_encoder/config.json, tokenizer/tokenizer_config.json.
    echarlaix/tiny-random-latent-consistency and
      optimum-intel-internal-testing/tiny-random-latent-consistency:
      model_index.json plus unet/scheduler/vae/text/tokenizer configs.
    rupeshs/LCM-runwayml-stable-diffusion-v1-5,
      Disty0/LCM_SoteMix, Lykon/dreamshaper-8-lcm:
      model_index.json plus unet/scheduler/vae/text/tokenizer configs.
    latent-consistency/lcm-sdxl and latent-consistency/lcm-ssd-1b:
      root config.json UNet-only distilled configs.
  Authenticated Hugging Face API check:
    hf auth user `hlky`; latent-consistency/lcm-sdxl,
    lcm-ssd-1b, lcm-lora-sdv1-5, lcm-lora-sdxl, and lcm-lora-ssd-1b
    are public/non-gated. The LoRA repos contain LoRA weights only, not
    component JSON configs. lcm-sdxl and lcm-ssd-1b expose root UNet configs,
    not pipeline model_index/component subfolders.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/latent_consistency_models/__init__.py
  diffusers/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_text2img.py
  diffusers/src/diffusers/pipelines/latent_consistency_models/pipeline_latent_consistency_img2img.py
  diffusers/src/diffusers/pipelines/auto_pipeline.py

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
  diffusers/src/diffusers/schedulers/scheduling_lcm.py
  diffusers/src/diffusers/schedulers/scheduling_pndm.py
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/textual_inversion.py
  diffusers/src/diffusers/loaders/ip_adapter.py

External component configs inspected:
  CLIPTextModel and CLIPTokenizer configs from the official LCM repos.

Any missing files or assumptions:
  No gated blocker remained. Some `H:/configs/latent-consistency/*` entries are
  `{}` placeholders because those repos are LoRA-only or UNet-only, not full
  Diffusers pipelines. Component configs fetched from official raw URLs were
  not saved back to H:/configs because this worker owns only this report path.
```

## 2. Pipeline and component graph

The selected base pipeline is a Stable-Diffusion-shaped latent image generator with one important LCM change: the pipeline does not run true classifier-free guidance with negative/positive batch concatenation. Instead, `guidance_scale` is converted to a sinusoidal `w_embedding`, passed to the UNet as `timestep_cond`, and the UNet must have `time_cond_proj_dim` configured.

```text
prompt strings or cached prompt embeddings
  -> CLIPTokenizer + CLIPTextModel, or external prompt_embeds
  -> latent noise [B,4,H/8,W/8]
  -> guidance-scale embedding w = guidance_scale - 1
  -> denoising loop:
       UNet2DConditionModel(latents, timestep,
         timestep_cond=w_embedding,
         encoder_hidden_states=prompt_embeds,
         optional IP-Adapter image embeds)
       -> LCMScheduler.step returns (prev_sample, denoised)
  -> AutoencoderKL decode(denoised / scaling_factor)
  -> VaeImageProcessor postprocess
```

Required base components:

| Component | Class | File/source | Base role |
| --- | --- | --- | --- |
| Pipeline | `LatentConsistencyModelPipeline` | `pipeline_latent_consistency_text2img.py` | Text-to-image LCM loop. |
| Text encoder | `CLIPTextModel` | Transformers | First Dinoml slice can accept cached `[B,77,C]` embeddings. |
| Tokenizer | `CLIPTokenizer` | Transformers | CPU/data-pipeline stage. |
| Denoiser | `UNet2DConditionModel` | `unet_2d_condition.py` | SD-style conditional UNet with `time_cond_proj_dim`. |
| Scheduler | `LCMScheduler` | `scheduling_lcm.py` | LCM timestep schedule, boundary-condition denoise, stochastic reinjection except final step. |
| VAE | `AutoencoderKL` | `autoencoder_kl.py` | Decode required for text-to-image; encode required for img2img. |
| Image processor | `VaeImageProcessor` | `image_processor.py` | Preprocess for img2img, postprocess for decoded images. |

Independently cacheable stages:

- Prompt embeddings for fixed prompt/clip-skip/LoRA/textual-inversion state.
- Guidance embedding `w_embedding` for fixed batch and guidance scale.
- LCM timestep tables for `num_inference_steps`, `original_inference_steps`, `strength`, and any custom timesteps.
- Initial latent noise if caller supplies `latents`.
- VAE decode can be tested as an independent stage using `denoised`.

Separate candidate reports:

| Surface | Support in this family | Candidate slug/order |
| --- | --- | --- |
| LoRA | Pipeline inherits `StableDiffusionLoraLoaderMixin`; LCM-LoRA repos provide LoRA weights for SD/SDXL/SSD rather than full pipeline configs. | `lcm_lora_runtime_adapters`, after SD LoRA report. |
| Textual inversion | Pipeline inherits `TextualInversionLoaderMixin`; prompt conversion path is active before tokenization. | Fold into `lcm_lora_runtime_adapters` or SD textual inversion path. |
| Runtime PEFT adapters | Loader machinery can mutate UNet/text-encoder modules; not an LCM base op. | `lcm_lora_runtime_adapters`. |
| IP-Adapter | Pipeline inherits `IPAdapterMixin` and prepares `image_embeds` for UNet `added_cond_kwargs`. | `lcm_ip_adapter`, after SD1 IP-Adapter. |
| ControlNet | No LCM ControlNet pipeline in this folder. SD ControlNet can use LCM schedulers/LoRAs externally. | `sd_controlnet_lcm_scheduler_variant` if selected. |
| T2I-Adapter | No LCM T2I-Adapter pipeline in this folder. | `sd_t2i_adapter_lcm_scheduler_variant` if selected. |
| GLIGEN | Shared UNet/BasicTransformerBlock has GLIGEN kwargs, but LCM pipelines do not pass them. | `lcm_gligen_like_branch` only for a concrete fork. |
| img2img | Present as `LatentConsistencyModelImg2ImgPipeline`; adds VAE encode, strength slicing, and `LCMScheduler.add_noise`. | `latent_consistency_img2img`, high-priority follow-up. |
| inpaint/depth/upscale | No folder-local inpaint, depth2img, or upscale classes. | Reuse SD variant reports with LCMScheduler as a scheduler variant. |
| SDXL/SSD LCM | `latent-consistency/lcm-sdxl` and `lcm-ssd-1b` are UNet-only distilled configs with SDXL-style `text_time` conditioning and `time_cond_proj_dim`. | `sdxl_lcm_distilled_unet`, separate from this SD1-shaped pipeline. |
| PixArt LCM | `PixArt-LCM-XL-2-1024-MS` uses PixArt pipeline + LCMScheduler. | Already inventoried under `pixart_lcm`. |

## 3. Important config dimensions

Representative pipeline/component sweep:

| Repo | Pipeline class | UNet sample | Image default | UNet channels | Blocks | Cross dim | Attention head config | `time_cond_proj_dim` | VAE scale | Scheduler |
| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- |
| `SimianLuo/LCM_Dreamshaper_v7` | `LatentConsistencyModelPipeline` | 96 | 768 | 4 -> 4 | 320/640/1280/1280, 2 layers | 768 | `attention_head_dim=8` | 256 | 0.18215 | `LCMScheduler` epsilon |
| `echarlaix/tiny-random-latent-consistency` | `LatentConsistencyModelPipeline` | 32 | 256 if scale 8, VAE sample 32 | 4 -> 4 | 4/8, 1 layer | 32 | `attention_head_dim=8` | 32 | 0.18215 | `LCMScheduler` epsilon |
| `optimum-intel-internal-testing/tiny-random-latent-consistency` | `LatentConsistencyModelPipeline` | 32 | tiny/debug | 4 -> 4 | 4/8, 1 layer | 32 | `attention_head_dim=8` | 32 | 0.18215 | `LCMScheduler` epsilon |
| `rupeshs/LCM-runwayml-stable-diffusion-v1-5` | `StableDiffusionPipeline` | 64 | 512 | 4 -> 4 | 320/640/1280/1280 | 768 | 8 | omitted | 0.18215 | `LCMScheduler` epsilon |
| `Disty0/LCM_SoteMix` | `StableDiffusionPipeline` | 64 | 512 | 4 -> 4 | 320/640/1280/1280 | 768 | 8 | omitted | 0.18215 | `LCMScheduler` epsilon |
| `Lykon/dreamshaper-8-lcm` | `StableDiffusionPipeline` | 64 | 512 | 4 -> 4 | 320/640/1280/1280 | 768 | 8 | omitted | 0.18215 | `PNDMScheduler` in model index |
| `latent-consistency/lcm-sdxl` | UNet-only root config | 128 | SDXL 1024 context | 4 -> 4 | 320/640/1280 | 2048 | 5/10/20 | 256 | external SDXL VAE | no pipeline scheduler config |
| `latent-consistency/lcm-ssd-1b` | UNet-only root config | 128 | SDXL/SSD context | 4 -> 4 | 320/640/1280 | 2048 | 5/10/20 | 256 | external SDXL/SSD VAE | no pipeline scheduler config |

LCMScheduler config sweep:

| Repo | `num_train_timesteps` | `original_inference_steps` | `prediction_type` | `beta_schedule` | `clip_sample` | `set_alpha_to_one` | `steps_offset` | `timestep_scaling` |
| --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| `SimianLuo/LCM_Dreamshaper_v7` | 1000 | 50 | epsilon | scaled_linear | false | true | 1 | omitted, source default 10.0 |
| tiny random LCM | 1000 | 50 | epsilon | scaled_linear | false | true | 1 | omitted, source default 10.0 |
| `rupeshs/LCM-runwayml-stable-diffusion-v1-5` | 1000 | 50 | epsilon | scaled_linear | false | false | 1 | 10.0 |
| `Disty0/LCM_SoteMix` | 1000 | 50 | epsilon | scaled_linear | false | false | 1 | 10.0 |

Text encoder dimensions:

| Repo | Text class | Hidden | Layers | Heads | Max tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| `SimianLuo/LCM_Dreamshaper_v7` | `CLIPTextModel` | 768 | 12 | 12 | 77 |
| tiny random LCM | `CLIPTextModel` | 32 | 3 | 8 | 77 |
| SD1-compatible LCM references | `CLIPTextModel` | 768 | 12 | 12 | 77 |

Supported scheduler set:

- The folder-local LCM pipeline constructor is typed as `LCMScheduler` and docs say it currently only supports `LCMScheduler`.
- The copied `retrieve_timesteps` helper has generic custom-timesteps/custom-sigmas admission, but `LCMScheduler.set_timesteps` supports custom `timesteps`, not `sigmas`.
- Other Stable Diffusion pipelines can use `LCMScheduler` as a compatible scheduler swap, but that is not equivalent to this LCM pipeline because vanilla `StableDiffusionPipeline` performs true CFG batch concat/chunk and usually does not pass `timestep_cond=w_embedding`.

Recommended first Dinoml scheduler slice: `LCMScheduler` epsilon prediction, standard evenly spaced LCM timesteps, no thresholding, no clipping, no custom timesteps, and stochastic noise injection for non-final steps. Add custom timesteps and img2img `strength` after text-to-image one-step/multistep parity.

## 3a. Family variation traps

- `guidance_scale` in the LCM pipeline is embedded guidance, not true CFG. The pipeline property `do_classifier_free_guidance` returns `False`, negative prompts are intentionally ignored, and no negative/positive UNet batch is constructed.
- The embedded guidance tensor uses `w = guidance_scale - 1`, multiplied by 1000 inside sinusoidal embedding generation, then projected through `UNet2DConditionModel.time_embedding.cond_proj`.
- Base LCM UNets must have `time_cond_proj_dim` configured. Passing `timestep_cond` to a UNet without `cond_proj` is not the same graph.
- `LCMScheduler.step` returns two tensors: `prev_sample` for the next loop iteration and `denoised` for final VAE decode/preview. The pipeline decodes `denoised`, not necessarily the last `latents` variable by name.
- LCM standard timesteps are a subset of the original distillation schedule. The default `num_inference_steps=4` is not a normal 50-step DDIM/PNDM timetable.
- Img2img passes `strength` into `LCMScheduler.set_timesteps`; text-to-image does not. The image path uses VAE encode + `add_noise` at the first selected timestep.
- `SimianLuo/LCM_Dreamshaper_v7` uses `sample_size=96`, so default output is 768x768 with the standard VAE scale factor 8. SD1 LCM references use 64/512.
- SDXL/SSD LCM distilled UNets are SDXL-style `addition_embed_type="text_time"` and cross-attend to width 2048. They are not covered by the SD1-shaped LCM pipeline's CLIP single-encoder contract.
- Source layout is NCHW for latents and VAE images. NHWC/channel-last should be a guarded optimization only.
- Source supports optional IP-Adapter image embeds through `added_cond_kwargs`; inactive in base configs.

## 4. Runtime tensor contract

Base text-to-image, `SimianLuo/LCM_Dreamshaper_v7`, one image per prompt:

| Boundary | Tensor | Source layout | Shape | Notes |
| --- | --- | --- | --- | --- |
| token ids | `input_ids` | `[B,S]` | `[B,77]` | CPU tokenizer path. |
| prompt embeds | `prompt_embeds` | `[B,S,C]` | `[B,77,768]` | Repeated for `num_images_per_prompt`; no negative concat. |
| latent noise | `latents` | NCHW | `[B,4,96,96]` for 768 output | `randn_tensor * init_noise_sigma`; LCM sigma is 1.0. |
| guidance scalar | `w` | `[B]` | `[B]` | `guidance_scale - 1`. |
| guidance embed | `w_embedding` | `[B,D]` | `[B,256]` | Sin/cos embedding, dtype cast to latents. |
| denoiser input | latents, timestep, prompt, `w_embedding` | NCHW + BSC + BD | `[B,4,h,w]`, scalar `t`, `[B,77,C]`, `[B,D]` | No scheduler `scale_model_input`; LCM returns identity. |
| denoiser output | `model_pred` | NCHW | `[B,4,h,w]` | Prediction type interpreted by scheduler. |
| scheduler output | `latents`, `denoised` | NCHW | two `[B,4,h,w]` tensors | Non-final `latents` includes fresh noise; `denoised` is boundary-condition sample. |
| VAE decode input | `denoised / scale` | NCHW | `[B,4,h,w]` | Scale usually 0.18215 for SD1 LCMs. |
| decoded image | `image` | NCHW before postprocess | `[B,3,H,W]` | Postprocess to PIL/NumPy/torch. |

Img2img additional tensors:

| Boundary | Tensor | Source layout | Shape | Notes |
| --- | --- | --- | --- | --- |
| input image | `image_processor.preprocess(image)` | NCHW | `[B,3,H,W]` or latent `[B,4,h,w]` | Normalized CPU/GPU image path from `VaeImageProcessor`. |
| encoded latents | `vae.encode(image).latent_dist.sample()` | NCHW | `[B,4,h,w]` | Multiplied by VAE scaling factor. |
| image noise | `noise` | NCHW | same as latents | Generated once before loop. |
| noised init | `scheduler.add_noise(init_latents, noise, timesteps[:1])` | NCHW | `[B,4,h,w]` | Strength controls selected initial timestep schedule. |

Scheduler state:

- `alphas_cumprod`, `final_alpha_cumprod`, `timesteps`, `_step_index`, `_begin_index`.
- `num_inference_steps`, `custom_timesteps`, and `original_inference_steps`.
- Step scalar coefficients: `alpha_prod_t`, `alpha_prod_t_prev`, `beta_prod_t`, `beta_prod_t_prev`, `c_skip`, `c_out`.

CPU/data-pipeline work: tokenization, prompt truncation warnings, textual-inversion token expansion, PIL/image preprocessing, optional CLIP image encoder for IP-Adapter, and output postprocess. GPU/runtime work: UNet, guidance embedding if not precomputed, scheduler step/add_noise, VAE encode/decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent/image tensors, shape checks for text-to-image H/W divisible by 8.
- Prompt embedding repeat/view for `num_images_per_prompt`.
- Img2img image/latent batch duplication rules.
- Scalar/timestep table indexing and broadcasting over `[B,C,H,W]`.
- Optional IP-Adapter image embed list concat/repeat.
- NCHW postprocess and optional `output_type="latent"` return.

Convolution/downsample/upsample ops:

- Base LCM UNet stem: `Conv2d(4 -> 320, 3x3, padding=1)` for SimianLuo; tiny `Conv2d(4 -> 4, 3x3)`.
- SD-style ResnetBlock2D: GroupNorm, SiLU, Conv2d, time embedding add, GroupNorm, SiLU, Conv2d, residual.
- Downsample2D and Upsample2D in UNet.
- AutoencoderKL encode/decode Conv2d, ResNet, GroupNorm, SiLU, down/up blocks, quant/post-quant 1x1 convs.

GEMM/linear ops:

- CLIP text encoder if admitted: token/position embeddings, QKV/out projections, MLP.
- UNet timestep MLP plus `cond_proj` from `w_embedding`.
- ResNet time projection.
- Attention Q/K/V/out projections in spatial self-attention and prompt cross-attention.
- GEGLU/GELU feed-forward layers inside `BasicTransformerBlock`.

Attention primitives:

- Noncausal latent spatial self-attention.
- Cross-attention from latent spatial tokens to CLIP prompt tokens.
- Optional IP-Adapter added K/V attention through attention processors.
- No base mask, RoPE, QK norm, varlen, or KV cache requirements.

Normalization and adaptive conditioning:

- GroupNorm over source channel axis.
- LayerNorm over token hidden dimension.
- Timestep embedding plus guidance `timestep_cond` addition in `TimestepEmbedding`.
- SiLU and GELU/GEGLU activations.

Scheduler and guidance arithmetic:

- LCM timestep schedule generation from `original_inference_steps`, `num_train_timesteps`, `strength`, and custom timesteps.
- Prediction conversion for epsilon/sample/v-prediction, though sampled configs use epsilon.
- Boundary coefficients `c_skip`, `c_out`.
- Optional threshold/clip branches are source-supported but inactive in sampled configs.
- Stochastic noise reinjection on non-final LCM steps.
- Img2img `add_noise`.

VAE/postprocessing ops:

- VAE scaling factor multiply on encode and divide on decode.
- AutoencoderKL diagonal Gaussian sample/mode for img2img.
- Decode final `denoised` tensor rather than true-CFG-updated latent.
- Denormalize/clamp and layout conversion in `VaeImageProcessor`.

## 6. Denoiser/model breakdown

For `SimianLuo/LCM_Dreamshaper_v7`, the active UNet path is the SD1 UNet plus guidance conditioning:

```text
sample [B,4,H,W]
-> Timesteps(timestep), positional, flip_sin_to_cos=true
-> guidance embedding w_embedding [B,256]
-> TimestepEmbedding:
     t_emb = t_emb + cond_proj(w_embedding)
     Linear -> SiLU -> Linear
-> Conv2d(4 -> 320)
-> CrossAttnDownBlock2D x3:
     ResnetBlock2D repeated, BasicTransformerBlock cross-attn, downsample
-> DownBlock2D x1
-> UNetMidBlock2DCrossAttn
-> UpBlock2D / CrossAttnUpBlock2D with skip concats
-> GroupNorm -> SiLU -> Conv2d(320 -> 4)
```

ResnetBlock2D active pattern:

```text
GroupNorm -> SiLU -> Conv2d
time embedding projection -> add as channel bias
GroupNorm -> SiLU -> Conv2d
optional shortcut -> residual add
```

BasicTransformerBlock active pattern:

```text
flatten spatial map to [B,H*W,C]
LayerNorm -> self-attention -> residual
LayerNorm -> cross-attention to CLIP tokens -> residual
LayerNorm -> feed-forward -> residual
reshape to NCHW
```

Config-controlled branches:

- `time_cond_proj_dim` is required by this pipeline's guidance embedding path.
- `addition_embed_type`, class embeddings, ControlNet residuals, T2I-Adapter residuals, and GLIGEN branches are inactive for sampled SD1-shaped LCM configs.
- IP-Adapter is active only when image embeds are supplied and the UNet has the corresponding encoder projection/attention processors loaded.

## 7. Attention requirements

| Attention | Query | Key/value | Mask | Processor path |
| --- | --- | --- | --- | --- |
| UNet self-attn | latent spatial tokens `[B,H*W,C]` | same | none in base path | `attention.py` + `attention_processor.py` |
| UNet cross-attn | latent spatial tokens | CLIP prompt tokens `[B,77,768]` | none in base LCM prompt path | same |
| IP-Adapter branch | latent spatial tokens | prompt tokens plus image added K/V | optional adapter scale/mask path | `IPAdapterAttnProcessor*` |

Head geometry follows SD1-style config. `attention_head_dim=8` is the legacy field; in current `UNet2DConditionModel`, `num_attention_heads = num_attention_heads or attention_head_dim`, so sampled SD1-shaped LCM configs use 8 attention heads per relevant block under current source semantics.

Eager and PyTorch SDPA processors define parity. Source supports fused QKV/KV projections through attention processor mutation, but first Dinoml parity should keep unfused linears. A Dinoml flash-style provider is feasible for base mask-free self/cross attention if dtype, head dimension, dropout=0, and processor state are fixed. It must be guarded off for IP-Adapter added K/V, GLIGEN/custom attention kwargs, unsupported masks, or runtime LoRA/adapters not folded into weights.

`attention_dispatch.py` is not the primary path for this target.

## 8. Scheduler and denoising-loop contract

`LCMScheduler.set_timesteps`:

```text
original_steps = original_inference_steps or config.original_inference_steps
k = num_train_timesteps // original_steps
lcm_origin_timesteps = [1..floor(original_steps * strength)] * k - 1
if custom timesteps:
  validate descending, range, and membership warnings
  apply img2img strength slicing
else:
  reverse lcm_origin_timesteps
  inference_indices = floor(linspace(0, len(origin), num_inference_steps, endpoint=False))
  timesteps = lcm_origin_timesteps[inference_indices]
```

`LCMScheduler.step`:

```text
prev_timestep = timesteps[step_index + 1] or current timestep at final step
alpha_t, alpha_prev = alphas_cumprod[t], alphas_cumprod[prev_timestep]
beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev
c_skip = sigma_data^2 / ((t * timestep_scaling)^2 + sigma_data^2)
c_out = (t * timestep_scaling) / sqrt((t * timestep_scaling)^2 + sigma_data^2)

pred_x0 =
  epsilon: (sample - sqrt(beta_t) * model_output) / sqrt(alpha_t)
  sample: model_output
  v_prediction: sqrt(alpha_t) * sample - sqrt(beta_t) * model_output

pred_x0 = optional threshold/clip
denoised = c_out * pred_x0 + c_skip * sample
if not final step:
  prev_sample = sqrt(alpha_prev) * denoised + sqrt(beta_prev) * randn_like(model_output)
else:
  prev_sample = denoised
```

Loop-side graph work:

```text
for t in timesteps:
  model_pred = unet(latents, t, timestep_cond=w_embedding, prompt_embeds)
  latents, denoised = scheduler.step(model_pred, t, latents, generator)
decode denoised / vae_scale
```

Host/runtime split: keep `set_timesteps`, custom timetable validation, step index, and random generator ownership host-visible first. Compile/fuse `step` arithmetic after the LCM schedule and prediction type are explicit in the artifact. The first Dinoml slice should implement the sampled epsilon/no-threshold/no-clip branch, then add source-supported sample/v-pred and custom timesteps.

LCM vs other schedulers:

- `LCMScheduler` is required for the folder-local LCM pipeline parity.
- PNDM/DDIM/Euler/DPM schedulers remain Stable Diffusion family scheduler swaps, not direct replacements for this pipeline because they do not consume the LCM boundary-condition denoise contract.
- A normal `StableDiffusionPipeline` with `LCMScheduler` is a separate compatibility surface: it uses true CFG arithmetic and lacks the LCM pipeline guidance embedding unless a LoRA/distilled workflow adds it through another path.

## 9. Position, timestep, and custom math

Required custom math:

- Standard UNet sinusoidal timestep embedding from `Timesteps`.
- LCM guidance embedding:

```text
w = guidance_scale - 1
w_scaled = 1000 * w
emb = concat(sin(w_scaled * freqs), cos(w_scaled * freqs))
optional zero-pad if odd embedding_dim
```

- `TimestepEmbedding.forward(sample, condition)` adds `cond_proj(condition)` to the timestep sample before the first linear.
- Boundary-condition scalings use `sigma_data = 0.5` and `timestep_scaling`, defaulting to 10.0 when omitted by config.

Precomputable:

- Prompt embeddings and token IDs for fixed prompt/tokenizer/LoRA/TI state.
- `w_embedding` for fixed guidance scale and batch.
- LCM timesteps and scalar coefficient tables for fixed scheduler config.

Dynamic:

- Img2img VAE encode sample and `add_noise` depend on input image and random noise.
- Scheduler stochastic noise depends on generator at each non-final step.
- Custom timesteps and `strength` change the scheduler table.

## 10. Preprocessing and input packing

Text path:

- Textual inversion may expand multi-vector tokens before tokenization.
- Tokenizer pads/truncates to CLIP max length 77.
- `clip_skip` can choose an intermediate hidden state and applies final CLIP LayerNorm afterward.
- Prompt embeddings are duplicated for `num_images_per_prompt`.
- Negative prompts are intentionally not supported by this LCM pipeline; the empty unconditional prompt was baked into guided distillation.

Image path:

- Text-to-image starts from Gaussian latent noise.
- Img2img preprocesses PIL/NumPy/torch image inputs through `VaeImageProcessor`, VAE-encodes RGB images unless the input already has 4 channels, scales latents by `vae.config.scaling_factor`, and adds scheduler noise at the first selected timestep.
- IP-Adapter preprocessing uses CLIP image processor and `CLIPVisionModelWithProjection` when image inputs rather than precomputed embeds are supplied.

Postprocess:

- Decode `denoised / scaling_factor`.
- Run `VaeImageProcessor.postprocess` to requested output type. Safety checker is ignored for Dinoml audit scope.

## 11. Graph rewrite / lowering opportunities

### Rewrite: LCM guidance embedding precompute

Source pattern: scalar `guidance_scale` -> `w = guidance_scale - 1` -> sin/cos embedding -> UNet `timestep_cond`.

Replacement: precompute `w_embedding` per batch/guidance-scale or represent it as a tiny compiled embedding kernel.

Preconditions: `guidance_scale`, `time_cond_proj_dim`, dtype, and batch size are known; UNet config has `time_cond_proj_dim`.

Failure cases: runtime guidance changes per sample, UNet has no `cond_proj`, or a non-LCM pipeline uses true CFG instead.

Parity test: compare `get_guidance_scale_embedding` and `TimestepEmbedding` input sums for dimensions 32 and 256.

### Rewrite: LCM scheduler step fused pointwise

Source pattern: alpha table lookup, prediction conversion, boundary-condition blend, optional noise reinjection.

Replacement: one fused scheduler kernel for epsilon/no-clip/no-threshold plus optional random-noise input.

Preconditions: epsilon prediction, static scheduler config, explicit step index, provided noise for non-final steps.

Failure cases: thresholding, clipping, sample/v-pred branches, custom timesteps not represented, or random generator ordering not matched.

Parity test: one-step and four-step scheduler parity with fixed `model_output`, `sample`, and supplied noise tensors.

### Rewrite: NCHW UNet/VAE conv islands to guarded NHWC

Source pattern: Conv2d -> GroupNorm -> SiLU -> Conv2d -> residual in UNet and VAE.

Replacement: NHWC island with NCHW boundaries preserved at pipeline/attention/scheduler contracts.

Preconditions: all channel-axis ops in island are rewritten; attention flatten/reshape boundaries are explicit; scheduler pointwise kernels are layout-polymorphic or outside the island.

Weight transform: OIHW -> HWIO for Conv2d kernels.

Failure cases: wrong GroupNorm axis, channel concat in up blocks not rewritten, VAE scale applied to wrong axis, or attention consumers expecting source NCHW.

Parity test: one ResnetBlock2D, one CrossAttnDownBlock2D, one VAE decode block at LCM 96/48/24 latent resolutions.

### Rewrite: UNet attention canonicalization

Source pattern: spatial flatten -> Q/K/V linears -> SDPA/eager attention -> output linear -> reshape.

Replacement: GEMM + attention provider + GEMM, with optional fused QKV/KV only after unfused parity.

Preconditions: default attention processor, no IP-Adapter/GLIGEN/custom kwargs, no runtime LoRA mutation unless folded into weights, dropout 0.

Failure cases: added K/V adapter processors, unsupported masks, xFormers-only processor state.

Parity test: one BasicTransformerBlock at SD1 LCM hidden widths with fixed prompt embeddings.

## 12. Kernel fusion candidates

Highest priority:

- `LCMScheduler.step` fused arithmetic, including boundary coefficients and stochastic reinjection. LCM's small step count makes scheduler overhead visible, and parity depends on decoding `denoised`.
- UNet GroupNorm + SiLU + Conv2d in ResNet blocks.
- Cross-attention Q/K/V + attention + output projection for SD1 latent/text attention.
- Guidance embedding + timestep embedding `cond_proj` path.
- VAE decode conv/resnet/up blocks.

Medium priority:

- Img2img VAE encode + `LCMScheduler.add_noise`.
- NHWC conv islands across UNet/VAE stages.
- GEGLU/GELU feed-forward fusion inside transformer blocks.
- Optional IP-Adapter added K/V attention once the base path is stable.

Lower priority:

- Custom timestep warnings/admission beyond standard schedule.
- Source-supported sample/v-prediction and threshold/clip branches absent from sampled LCM configs.
- SDXL/SSD LCM distilled UNets, because they require SDXL dual-text/time-id conditioning.
- LoRA loader/adaptor mutation.

## 13. Runtime staging plan

Stage 1: Parse `SimianLuo/LCM_Dreamshaper_v7` and tiny LCM component configs. Enforce `UNet2DConditionModel.time_cond_proj_dim` for the folder-local LCM pipeline.

Stage 2: Accept external prompt embeddings and implement guidance-scale embedding plus UNet timestep conditioning.

Stage 3: Validate one UNet ResnetBlock2D and one BasicTransformerBlock on tiny LCM, then SimianLuo SD1-shaped widths.

Stage 4: Implement `LCMScheduler.set_timesteps` standard schedule and epsilon `step` with supplied random noise; keep loop control in host code.

Stage 5: One denoising-step parity with fixed latents, prompt embeddings, guidance scale, timestep, and random noise.

Stage 6: Four-step text-to-image loop parity with compiled/fused UNet step and host scheduler control.

Stage 7: Add AutoencoderKL decode from final `denoised`; validate output-type latent and decoded-image paths.

Stage 8: Add img2img: VAE encode, `strength`, `add_noise`, and initial timestep schedule.

Stage 9: Add optimized attention, scheduler, and NHWC conv-island fusions.

Stage 10: Separate SDXL/SSD LCM, LCM-LoRA, IP-Adapter, and SD variant scheduler reports.

## 14. Parity and validation plan

- Config parse tests for `LatentConsistencyModelPipeline` model indexes and component defaults.
- Reject or separately route `StableDiffusionPipeline` + `LCMScheduler` repos when `time_cond_proj_dim` is missing for the folder-local pipeline.
- Guidance embedding parity for `embedding_dim=32` and 256.
- `TimestepEmbedding` parity with `timestep_cond`.
- `LCMScheduler.set_timesteps` parity for `num_inference_steps=1,2,4,8`, `original_inference_steps=50`.
- `LCMScheduler.step` parity for epsilon prediction with supplied fixed noise and final-step no-noise branch.
- `LCMScheduler.add_noise` parity for img2img.
- UNet block parity on tiny LCM and SD1 LCM widths.
- One full UNet forward parity with cached prompt embeddings.
- One denoising step parity covering `latents`, `model_pred`, `prev_sample`, and `denoised`.
- VAE decode parity from final `denoised / scaling_factor`.
- Img2img VAE encode/sample/scale/noise parity.
- Suggested tolerances: fp32 scheduler/guidance `rtol=1e-5, atol=1e-6`; fp32 UNet/VAE `rtol=1e-4, atol=1e-5`; fp16/bf16 initial `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One LCM UNet step by latent size 64 and 96, batch, dtype, and guidance scale.
- Four-step loop latency split: UNet, scheduler, VAE decode, prompt encoding.
- Scheduler step overhead with and without fused random-noise reinjection.
- Guidance embedding overhead when precomputed vs generated per call.
- VAE decode throughput for 512 and 768 outputs.
- Img2img encode + add_noise overhead by resolution and strength.
- Attention backend comparison: eager/SDPA vs Dinoml flash-style provider.
- NCHW faithful path vs guarded NHWC conv islands.
- VRAM/temp usage for LCM one-step, four-step, and img2img.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `latent_consistency_img2img`: `LatentConsistencyModelImg2ImgPipeline`, VAE encode, strength slicing, and `LCMScheduler.add_noise`.
- `lcm_lora_runtime_adapters`: `StableDiffusionLoraLoaderMixin`, PEFT LoRA state, `latent-consistency/lcm-lora-*` repos, and fuse/unfuse/runtime adapter mutation.
- `lcm_textual_inversion`: tokenizer/text-encoder embedding mutation if not folded into the broader SD textual inversion support.
- `lcm_ip_adapter`: `IPAdapterMixin`, CLIP image encoder/projection, and added K/V attention processors.
- `sdxl_lcm_distilled_unet`: root UNet configs `latent-consistency/lcm-sdxl` and `lcm-ssd-1b`; SDXL text/time conditioning and SDXL scheduler integration.
- `stable_diffusion_lcm_scheduler_swap`: normal SD pipeline using `LCMScheduler` and true CFG arithmetic.
- `pixart_lcm`: PixArt Alpha pipeline with LCMScheduler, already noted in the PixArt report.
- `lcm_controlnet_t2i_adapter_variants`: only when a concrete SD/SDXL control pipeline is selected with LCM scheduler or LCM LoRA.
- `lcm_inpaint_depth_upscale_variants`: no folder-local classes; route through SD variant reports plus LCMScheduler.

Genuinely out of scope for this audit:

- XLA/NPU/MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Callback mutation and interactive interrupt.
- Training, losses, dropout, and gradient checkpointing.
- Multi-GPU/context parallel paths.

## 17. Final implementation checklist

- [ ] Parse LCM pipeline model indexes and component configs.
- [ ] Enforce `time_cond_proj_dim` for folder-local LCM pipeline admission.
- [ ] Load SD1-shaped LCM UNet/VAE/text configs and weights.
- [ ] Accept external CLIP prompt embeddings for first slice.
- [ ] Implement LCM guidance-scale embedding.
- [ ] Implement `TimestepEmbedding` with `timestep_cond`.
- [ ] Implement UNet ResnetBlock2D and BasicTransformerBlock parity.
- [ ] Implement `LCMScheduler.set_timesteps` standard schedule.
- [ ] Implement epsilon/no-clip/no-threshold `LCMScheduler.step`.
- [ ] Add stochastic-noise reinjection with explicit supplied noise tensors.
- [ ] Decode final `denoised`, not just loop `latents`.
- [ ] Add one-step and four-step text-to-image parity.
- [ ] Add AutoencoderKL decode boundary.
- [ ] Add img2img VAE encode, strength, and `add_noise` follow-up.
- [ ] Add guarded NHWC conv-island tests.
- [ ] Add attention provider/flash feasibility guards.
- [ ] Keep LoRA/TI/IP-Adapter, SDXL/SSD LCM, ControlNet/T2I, inpaint/depth/upscale, and PixArt LCM as separate candidate surfaces.
