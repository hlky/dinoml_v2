# Diffusers AnimateDiff / Stable Video Diffusion Audit

Candidate slug: `animatediff_stable_video_diffusion`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  AnimateDiff motion adapter examples:
    guoyww/animatediff-motion-adapter-v1-5-2
    guoyww/animatediff-motion-adapter-v1-5-3
    guoyww/animatediff-motion-adapter-sdxl-beta
    wangfuyun/AnimateLCM
    ByteDance/AnimateDiff-Lightning (repo index only locally; root config 404)
  AnimateDiff base UNet examples:
    frankjoshua/toonyou_beta6
    runwayml/stable-diffusion-v1-5
    stabilityai/stable-diffusion-xl-base-1.0
  Stable Video Diffusion examples:
    stabilityai/stable-video-diffusion-img2vid
    stabilityai/stable-video-diffusion-img2vid-xt

Config sources:
  Local cache checked first:
    H:/configs/guoyww/animatediff-motion-adapter-v1-5-2/model_index.json
    H:/configs/guoyww/animatediff-motion-adapter-v1-5-3/model_index.json
    H:/configs/guoyww/animatediff-motion-adapter-sdxl-beta/model_index.json
    H:/configs/wangfuyun/AnimateLCM/model_index.json
    H:/configs/ByteDance/AnimateDiff-Lightning/model_index.json
    H:/configs/stabilityai/stable-video-diffusion-img2vid/model_index.json
    H:/configs/stabilityai/stable-video-diffusion-img2vid-xt/model_index.json
  Official raw component configs fetched in-memory:
    guoyww/animatediff-motion-adapter-v1-5-2/config.json
    guoyww/animatediff-motion-adapter-v1-5-3/config.json
    guoyww/animatediff-motion-adapter-sdxl-beta/config.json
    wangfuyun/AnimateLCM/config.json
    frankjoshua/toonyou_beta6/unet/config.json
    runwayml/stable-diffusion-v1-5/unet/config.json
    stabilityai/stable-diffusion-xl-base-1.0/unet/config.json
    stabilityai/stable-diffusion-xl-base-1.0/vae/config.json
    stabilityai/stable-video-diffusion-img2vid/{unet,vae,scheduler,image_encoder,feature_extractor}/config.json
    stabilityai/stable-video-diffusion-img2vid-xt/{unet,vae,scheduler,image_encoder}/config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff.py
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_sdxl.py
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_controlnet.py
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video_controlnet.py
  diffusers/src/diffusers/pipelines/animatediff/pipeline_animatediff_sparsectrl.py
  diffusers/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py

Model files inspected:
  diffusers/src/diffusers/models/unets/unet_motion_model.py
  diffusers/src/diffusers/models/unets/unet_spatio_temporal_condition.py
  diffusers/src/diffusers/models/unets/unet_3d_blocks.py
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/transformers/transformer_temporal.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_lcm.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/pipelines/free_noise_utils.py

External component configs inspected:
  SVD CLIPVisionModelWithProjection config from official image_encoder.
  SVD CLIPImageProcessor preprocessor config from official feature_extractor.
  AnimateDiff text encoder/tokenizer behavior is inherited from SD1/SDXL and
  treated as an external prompt-embedding stage for first Dinoml admission.

Missing files or assumptions:
  ByteDance/AnimateDiff-Lightning root config.json returned 404. The local cache
  has only an empty model_index.json for that repo, so it is listed as a
  scheduler/distillation variant but not used for operator-significant fields.
  No gated official config blocked this audit. Official component configs were
  read in-memory and not saved because the task-owned write path is this report.
  XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient
  checkpointing, multi-GPU/context parallel, callbacks, and interrupt mutation
  were ignored.
```

## 2. Pipeline and component graph

AnimateDiff and Stable Video Diffusion are both UNet-centered video families,
but they enter video space differently.

AnimateDiff text-to-video:

```text
prompt / prompt_embeds
  -> CLIP text encoder or externally supplied prompt embeddings
  -> optional IP-Adapter image embeddings
  -> initialize video latents [B,4,F,H/8,W/8]
  -> denoising loop:
       CFG batch concat -> scheduler scale_model_input
       -> UNetMotionModel: SD UNet2D blocks + motion modules
       -> CFG arithmetic -> scheduler.step
  -> AutoencoderKL decode per frame
  -> VideoProcessor postprocess
```

Stable Video Diffusion image-to-video:

```text
input image
  -> CLIP image preprocessing + CLIPVisionModelWithProjection image_embeds [B,1,1024]
  -> VAE encode noisy conditioning image -> image_latents [B,4,H/8,W/8]
  -> repeat image_latents across frames
  -> initialize noise latents [B,F,4,H/8,W/8]
  -> denoising loop:
       CFG batch concat -> scheduler scale_model_input
       -> concat noisy video latents and conditioning image_latents on channel dim
       -> UNetSpatioTemporalConditionModel(latents8ch, image_embeds, fps/motion/noise ids)
       -> per-frame CFG ramp -> scheduler.step
  -> AutoencoderKLTemporalDecoder decode chunks with temporal convs
  -> VideoProcessor postprocess
```

Required first-slice components:

| Family | Required components | Optional components |
| --- | --- | --- |
| AnimateDiff | `AutoencoderKL`, `CLIPTextModel` or supplied prompt embeds, `UNetMotionModel`, `MotionAdapter`, compatible scheduler, `VideoProcessor` | IP-Adapter image encoder/projections, LoRA/textual inversion/PEFT, FreeInit/FreeNoise, SDXL conditioning, ControlNet/SparseCtrl, video2video VAE encode |
| SVD | `AutoencoderKLTemporalDecoder`, `CLIPVisionModelWithProjection` or supplied image embeddings, `UNetSpatioTemporalConditionModel`, `EulerDiscreteScheduler`, `VideoProcessor`, `CLIPImageProcessor` | Custom sigmas, reduced decode chunks, future temporal ControlNet-style variants outside this pipeline |

Separate candidate reports:

| Surface | Classes/files | Why separate |
| --- | --- | --- |
| AnimateDiff SDXL | `AnimateDiffSDXLPipeline`, `UNetMotionModel`, SDXL text encoders and added text/time conditioning | Same motion-module pattern but SDXL dual text encoders, pooled text embeddings, size/crop time ids, and 3-block motion adapter. |
| AnimateDiff video2video | `AnimateDiffVideoToVideoPipeline` | Adds input video preprocessing, VAE encode, strength/timestep slicing, and scheduler `add_noise`. |
| AnimateDiff ControlNet | `AnimateDiffControlNetPipeline`, `AnimateDiffVideoToVideoControlNetPipeline`, `ControlNetModel`, `MultiControlNetModel` | Adds conditioning image/video residual branches into the motion UNet. |
| AnimateDiff SparseCtrl | `AnimateDiffSparseControlNetPipeline`, `SparseControlNetModel` | Sparse frame conditions and masks are a separate control tensor contract. |
| AnimateDiff IP-Adapter | `prepare_ip_adapter_image_embeds`, `IPAdapterAttnProcessor*`, `ImageProjection` | Adds image K/V branches and image projection layers to SD attention. |
| LoRA/textual inversion/runtime adapters | AnimateDiff inherits SD loader mixins; `loaders/lora_pipeline.py`, `loaders/textual_inversion.py`, `loaders/peft.py` | Mutates text encoder/UNet weights or token embeddings at load/runtime. |
| SVD temporal ControlNet community variants | e.g. cached `CiaraRowles/temporal-controlnet-*-svd-v1` | Not part of official `StableVideoDiffusionPipeline`; side-conditioning deserves separate source/config audit. |
| img2img/inpaint/depth/upscale | SD/SDXL base variants rather than this first slice | AnimateDiff video2video covers video conditioning; classic image variants are already separate SD candidates. |

## 3. Important config dimensions

AnimateDiff representative configs:

| Repo | Component | Class | Operator-significant fields |
| --- | --- | --- | --- |
| `guoyww/animatediff-motion-adapter-v1-5-2` | motion adapter | `MotionAdapter` | blocks `320/640/1280/1280`, `motion_layers_per_block=2`, mid block enabled, `motion_num_attention_heads=8`, `motion_max_seq_length=32`, group norm 32, activation `geglu`, cross-attn dim null |
| `guoyww/animatediff-motion-adapter-v1-5-3` | motion adapter | `MotionAdapter` | same block widths/layers, mid block disabled |
| `guoyww/animatediff-motion-adapter-sdxl-beta` | motion adapter | `MotionAdapter` | blocks `320/640/1280`, 2 motion layers, mid block disabled, heads 8 |
| `wangfuyun/AnimateLCM` | motion adapter | `MotionAdapter` | SD1-style 4 blocks, mid block enabled, optional `conv_in_channels=null`; used with LCM-style schedulers in examples |
| `frankjoshua/toonyou_beta6` | base UNet | `UNet2DConditionModel` | SD1 shape: in/out 4, sample 64, blocks `320/640/1280/1280`, cross dim 768, 2 layers/block, attention head dim 8 |
| `stabilityai/stable-diffusion-xl-base-1.0` | base UNet | `UNet2DConditionModel` | sample 128, blocks `320/640/1280`, cross dim 2048, transformer layers `[1,2,10]`, projection class input 2816 |

SVD representative configs:

| Repo | Component | Class | Operator-significant fields |
| --- | --- | --- | --- |
| `stabilityai/stable-video-diffusion-img2vid` | UNet | `UNetSpatioTemporalConditionModel` | sample 96, `num_frames=14`, in 8, out 4, blocks `320/640/1280/1280`, heads `5/10/20/20`, cross dim 1024, `addition_time_embed_dim=256`, class projection input 768 |
| `stabilityai/stable-video-diffusion-img2vid-xt` | UNet | `UNetSpatioTemporalConditionModel` | same as img2vid but `num_frames=25` |
| both SVD repos | VAE | `AutoencoderKLTemporalDecoder` | in/out 3, latent 4, blocks `128/256/512/512`, sample 768, scaling `0.18215`, `force_upcast=true` |
| both SVD repos | image encoder | `CLIPVisionModelWithProjection` | hidden 1280, layers 32, heads 16, patch 14, image size 224, projection 1024, fp16 metadata |
| both SVD repos | scheduler | `EulerDiscreteScheduler` | `prediction_type=v_prediction`, scaled-linear betas, 1000 train steps, Karras sigmas enabled, `timestep_type=continuous`, `sigma_min=0.002`, `sigma_max=700`, leading spacing |

Recommended first Dinoml scheduler slices:

| Family | Source/default scheduler | First slice recommendation |
| --- | --- | --- |
| AnimateDiff SD1 | Compatible DDIM/PNDM/LMS/Euler/Euler ancestral/DPM; examples commonly DDIM, AnimateLCM uses LCM | Reuse SD1 `DDIMScheduler` or `LCMScheduler` only when the selected motion adapter/checkpoint requires it; start with DDIM epsilon for conventional AnimateDiff. |
| AnimateDiff SDXL | SDXL-compatible DDIM/Euler/DPM set | Defer until SDXL base and SD1 AnimateDiff motion modules are stable. |
| SVD | `EulerDiscreteScheduler` v-pred continuous Karras | First SVD parity should implement exact Euler v-pred/Karras continuous contract, not substitute DDIM. |

## 3a. Family variation traps

- AnimateDiff source latents are `[B,C,F,H,W]` in the pipeline, but
  `UNetMotionModel.forward` documents and uses `sample.shape[2]` as frames after
  receiving `[B,C,F,H,W]`; it permutes to `[B,F,C,H,W]` internally before
  flattening to `[B*F,C,H,W]`. SVD source latents are `[B,F,C,H,W]` at the
  pipeline/model boundary.
- AnimateDiff motion modules are adapter weights inserted around a 2D UNet. The
  base UNet controls SD1/SDXL text conditioning; the motion adapter controls
  temporal self-attention only.
- `MotionAdapter` config must match base UNet block count and layers per block.
  SD1 adapters have four blocks; SDXL beta has three.
- AnimateDiff v1-5-3 and SDXL beta disable the motion mid block. Do not assume a
  mid motion module exists.
- SVD UNet `in_channels=8` because the model input concatenates 4 noisy video
  latent channels and 4 repeated conditioning image latent channels.
- SVD `prepare_latents` uses `num_channels_latents // 2`; denoised state is 4
  channels even though UNet input is 8 channels.
- SVD guidance scale is a tensor ramp over frames, shaped `[B,F,1,1,1]`, not a
  scalar. AnimateDiff SD1 uses scalar CFG in the base pipeline.
- SVD `fps` is decremented by 1 before added-time conditioning.
- `AutoencoderKLTemporalDecoder` encode is still 2D image encode; its decode adds
  temporal behavior via `SpatioTemporalResBlock` and final `Conv3d(3->3,kT=3)`.
- NCDHW/NDHWC layout passes need guards around `permute`, frame flattening,
  GroupNorm axes, dim-1 channel concats in AnimateDiff internals, dim-2 channel
  concat in SVD pipeline, and alpha blending over frame indicator tensors.

## 4. Runtime tensor contract

AnimateDiff text-to-video, SD1-style 512x512, 16 frames:

| Boundary | Source tensor | Shape | Layout notes |
| --- | --- | --- | --- |
| prompt embeds | CLIP hidden states | `[B,77,768]`, CFG `[2B,77,768]`, then repeated to `[2B*F,77,768]` | Token-major; external first slice can supply this. |
| latent state | video latents | `[B,4,F,64,64]` | Source pipeline layout is `BCFHW`. |
| UNet pre-flatten | motion UNet sample | `[B,4,F,Hl,Wl]` | `UNetMotionModel` reads `num_frames=sample.shape[2]`. |
| UNet conv/attention | flattened frames | `[B*F,C,Hl,Wl]` | 2D ResNet/cross-attn operate per frame; motion modules reassemble temporal sequences. |
| motion module tokens | temporal attention input | `[B*Hl*Wl,F,C]` after GroupNorm/proj | Temporal sequence length is frames; `motion_max_seq_length=32` for sampled adapters. |
| UNet output | predicted noise | `[B,4,F,Hl,Wl]` | Reshaped back after `conv_out`. |
| VAE decode input | frame latents | `[B*F,4,Hl,Wl]` | Pipeline permutes `[B,4,F,H,W] -> [B*F,4,H,W]`. |
| VAE decode output | video tensor | `[B,3,F,H,W]` | Postprocess video returns list/np/pt. |

SVD img2vid, 576x1024:

| Boundary | Source tensor | Shape | Layout notes |
| --- | --- | --- | --- |
| CLIP image input | image tensor | `[B,3,224,224]` after antialias resize and CLIP normalization | CPU/data + image encoder stage. |
| image embeddings | CLIP projection | `[B,1,1024]`, CFG `[2B,1,1024]` | Repeated per video count only, then model repeats per frame. |
| VAE image input | preprocessed/noised image | `[B,3,576,1024]` | Noise augmentation before encode. |
| conditioning image latents | VAE mode | `[B,4,72,128]`, CFG `[2B,4,72,128]`, repeated to `[2B,F,4,72,128]` | No explicit scaling multiply in `_encode_vae_image`; Diffusers SVD source uses posterior mode directly. |
| denoised latent state | video latents | `[B,F,4,72,128]` | Source pipeline boundary is `BFCHW`. |
| UNet input | concat noisy + image latents | `[2B,F,8,72,128]` when CFG | Channel concat uses dim 2. |
| added time ids | fps/motion/noise | `[2B,3]` | Encoded by sinusoidal projection with dim 256 each; expected input to add MLP is 768. |
| UNet output | noise prediction | `[2B,F,4,72,128]` | CFG chunk on batch dim; per-frame guidance ramp broadcasts. |
| VAE decode | chunked latents | flatten `[B*F,4,72,128]`, decode with `num_frames=chunk` | Temporal decoder quality depends on chunk size; default decodes all frames. |
| output video | frames | `[B,3,F,576,1024]` before postprocess | `VideoProcessor.postprocess_video`. |

Precomputable/reusable:

- AnimateDiff prompt embeddings and negative prompt embeddings.
- SVD CLIP image embeddings, image latents, added-time ids, and scheduler
  timesteps/sigmas for fixed image, fps, motion bucket, noise strength, and step
  count.
- Motion adapter compatibility can be resolved at load/admission time.

## 5. Operator coverage checklist

Tensor/layout ops:

- AnimateDiff: `permute(0,2,1,3,4)`, reshape `[B,F,C,H,W] <-> [B*F,C,H,W]`,
  temporal token reshape `[B*H*W,F,C]`, `repeat_interleave` prompt embeds by
  frame count, CFG cat/chunk on batch, VAE frame decode chunking.
- SVD: image preprocessing resize/normalize/denormalize, repeat image latents
  over frames, concat channels on dim 2 for `[B,F,C,H,W]`, flatten decode chunks,
  per-frame guidance ramp broadcast.

Convolution/downsample/upsample:

- SD-style Conv2d/ResnetBlock2D/downsample/upsample for AnimateDiff.
- SVD `conv_in`: `Conv2d(8 -> 320, 3x3,pad=1)`.
- SVD `SpatioTemporalResBlock`: spatial `ResnetBlock2D` plus temporal
  `TemporalResnetBlock` with Conv3d over `[B,C,F,H,W]`.
- SVD temporal VAE decode: 2D decoder conv/resnet/up blocks plus temporal
  decoder blocks and final `Conv3d(3 -> 3, kernel=(3,1,1))`.

GEMM/linear:

- Timestep MLPs, added-time MLPs, projection class embeddings.
- AnimateDiff temporal `proj_in/proj_out` linears around temporal attention.
- SVD `TransformerSpatioTemporalModel` spatial and temporal Q/K/V/out
  projections and GEGLU feed-forward.

Attention:

- AnimateDiff base spatial self/cross attention inherited from SD1/SDXL.
- AnimateDiff temporal self-attention over frame tokens; sampled motion adapters
  use no temporal cross attention.
- SVD spatial cross-attention to CLIP image embeddings and temporal attention
  through `TemporalBasicTransformerBlock`.
- Attention processors are `Attention` + `attention_processor.py`; SDPA
  `AttnProcessor2_0` is the parity path when available, eager `AttnProcessor`
  is fallback.

Norm/adaptive conditioning:

- GroupNorm over channel axis for NCHW/NCDHW-like tensors.
- LayerNorm in BasicTransformerBlock and TemporalBasicTransformerBlock.
- AlphaBlender learned/fixed mix between spatial and temporal streams.
- SVD added-time conditioning: sinusoidal projection of `[fps-1,
  motion_bucket_id, noise_aug_strength]`, then TimestepEmbedding add to `emb`.

Schedulers/guidance:

- AnimateDiff DDIM/PNDM/Euler/DPM/LCM-compatible step math depending selected
  checkpoint.
- SVD EulerDiscrete v-pred continuous Karras, `scale_model_input`, scheduler
  step, and frame-ramp CFG.

VAE/postprocessing:

- AnimateDiff AutoencoderKL decode per frame with scalar unscale
  `latents / scaling_factor`.
- SVD AutoencoderKLTemporalDecoder encode image mode, decode chunks with
  `num_frames`, temporal conv, force-upcast handling.

## 6. Denoiser/model breakdown

AnimateDiff `UNetMotionModel`:

```text
input [B,4,F,H,W]
  -> permute/reshape to [B*F,4,H,W]
  -> SD conv_in
  -> down blocks:
       ResnetBlock2D + spatial self/cross BasicTransformerBlock
       motion module after/before resnet depending block:
         GroupNorm over [B,C,F,H,W]
         reshape to temporal tokens [B*H*W,F,C]
         Linear(C -> heads*head_dim)
         BasicTransformerBlock with double self-attention, GEGLU
         Linear -> reshape back -> residual
       optional Downsample2D
  -> optional motion mid block
  -> up blocks with skip concat, spatial attention, motion modules, Upsample2D
  -> GroupNorm + SiLU + Conv2d(320 -> 4)
  -> reshape to [B,4,F,H,W]
```

Motion adapter contract:

- `MotionAdapter` stores only temporal modules, not the whole UNet.
- `UNetMotionModel.from_unet2d` rewrites block type names from
  `CrossAttnDownBlock2D` to `CrossAttnDownBlockMotion` and
  `CrossAttnUpBlock2D` to `CrossAttnUpBlockMotion`.
- It copies base UNet Conv2d/ResNet/spatial attention weights, then loads
  motion module weights from the adapter.
- Compatibility checks enforce same block count and expanded `layers_per_block`
  as adapter `motion_layers_per_block`.
- If adapter `conv_in_channels` is set, PIA-style UNets can become 9-channel;
  sampled adapters here have null `conv_in_channels`, so this is not first-slice.

SVD `UNetSpatioTemporalConditionModel`:

```text
input [B,F,8,H,W]
  -> time embedding(timestep) + added time embedding(fps,motion,noise)
  -> flatten frames to [B*F,8,H,W]
  -> Conv2d(8 -> 320)
  -> image_only_indicator zeros [B,F]
  -> down blocks:
       SpatioTemporalResBlock:
         spatial ResnetBlock2D over [B*F,C,H,W]
         reshape to [B,C,F,H,W]
         TemporalResnetBlock with Conv3d
         AlphaBlender spatial/temporal mix
       TransformerSpatioTemporalModel in cross-attn blocks:
         spatial BasicTransformerBlock over H*W tokens
         temporal BasicTransformerBlock over F tokens per spatial position
         AlphaBlender mix
       Downsample2D where configured
  -> UNetMidBlockSpatioTemporal
  -> up blocks with skip concat, spatio-temporal res/attention, Upsample2D
  -> GroupNorm + SiLU + Conv2d(320 -> 4)
  -> reshape [B,F,4,H,W]
```

SVD `AutoencoderKLTemporalDecoder`:

```text
encode image: Encoder2D -> quant_conv -> DiagonalGaussianDistribution
decode video latents:
  [B*F,4,Hl,Wl] -> Conv2d(4 -> 512)
  -> MidBlockTemporalDecoder / UpBlockTemporalDecoder
       SpatioTemporalResBlock and temporal mixing
  -> GroupNorm -> SiLU -> Conv2d(128 -> 3)
  -> reshape [B,3,F,H,W] -> Conv3d(3 -> 3,k=(3,1,1)) -> flatten
```

## 7. Attention requirements

AnimateDiff:

- Spatial attention is unchanged from SD1/SDXL: latent tokens attend to CLIP
  text tokens. SD1 uses cross dim 768; SDXL uses 2048 and extra conditioning.
- Motion modules use temporal self-attention over sequence length `F` for each
  spatial location. For sampled SD1 adapters, heads=8 and head dim is
  `C/8`: 40, 80, 160, 160 at channels 320, 640, 1280, 1280.
- Motion modules use `BasicTransformerBlock` with sinusoidal positional
  embeddings and `num_positional_embeddings=motion_max_seq_length=32`.
- No temporal mask, no causal KV cache, no RoPE, no QK norm in the inspected
  motion modules.
- IP-Adapter branches can add external image K/V processors but are separate
  candidates.

SVD:

- `TransformerSpatioTemporalModel` uses spatial `BasicTransformerBlock` with
  cross-attention dim 1024, heads 5/10/20/20 and head dim 64 at all block
  widths.
- Temporal branch uses `TemporalBasicTransformerBlock`; it reshapes spatial
  token outputs to `[B*HW,F,C]`, adds frame-position timestep embeddings, runs
  temporal self-attention, optional temporal cross-attention with first-frame
  image context, and GEGLU feed-forward.
- `AlphaBlender(learned_with_images)` mixes spatial and temporal streams.
  Current pipeline passes zeros for `image_only_indicator`, so learned sigmoid
  alpha is active rather than image-only bypass.
- Source attention processor path is `Attention` in `attention.py` dispatched to
  processors in `attention_processor.py`; `AttnProcessor2_0` SDPA defines the
  native optimized parity path, with eager processor fallback.
- Fused QKV projections are source-supported for compatible attention modules,
  but added-KV/IP-Adapter processors disable simple fusion and are not required
  for first slice.

Flash-style Dinoml provider note: A flash provider can be valid for mask-free
spatial/temporal attention under dtype/shape guards, but parity must start from
eager or SDPA semantics. Temporal attention uses many short sequences (`F<=32`)
where flash overhead may not win; spatial attention at large latent grids is
more likely to benefit.

## 8. Scheduler and denoising-loop contract

AnimateDiff loop:

```text
scheduler.set_timesteps(num_inference_steps)
latents = randn([B,4,F,H/8,W/8]) * init_noise_sigma
for t in timesteps:
  model_input = cat([latents]*2) if CFG else latents
  model_input = scheduler.scale_model_input(model_input, t)
  noise_pred = UNetMotionModel(model_input, t, repeated_prompt_embeds)
  noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents, eta/generator if accepted).prev_sample
```

AnimateDiff scheduler support is broad because it inherits SD scheduler shapes:
DDIM, PNDM, LMS, Euler, Euler ancestral, and DPM solver are typed in the base
pipeline; AnimateLCM/Lightning-style repos are distilled variants and should be
admitted through an LCM-specific scheduler report rather than silently changing
the base loop.

SVD loop:

```text
retrieve_timesteps(EulerDiscreteScheduler, num_steps, sigmas?)
latents = randn([B,F,4,H/8,W/8]) * init_noise_sigma
guidance_scale = linspace(min,max,F)[B,F,1,1,1]
for t in timesteps:
  model_input = cat([latents]*2) if CFG else latents
  model_input = scheduler.scale_model_input(model_input, t)
  model_input = cat([model_input, image_latents], dim=2)
  noise_pred = UNetSpatioTemporalConditionModel(model_input, t, image_embeds, added_time_ids)
  noise_pred = uncond + guidance_scale * (cond - uncond)
  latents = scheduler.step(noise_pred, t, latents).prev_sample
```

SVD exact first parity needs EulerDiscrete `v_prediction`,
`use_karras_sigmas=true`, `timestep_type=continuous`, leading spacing, and
custom `sigmas` support through `retrieve_timesteps`. Keep schedule generation,
custom-sigma validation, and step index in host-visible scheduler state first;
compile/fuse `scale_model_input`, CFG, and step arithmetic after table parity.

## 9. Position, timestep, and custom math

AnimateDiff:

- Base UNet uses SD sinusoidal timestep embedding (`Timesteps` +
  `TimestepEmbedding`).
- Motion modules use BasicTransformerBlock with sinusoidal positional
  embeddings over temporal length, bounded by `motion_max_seq_length`.
- Prompt embeds are repeated per frame after CFG concat in the base pipeline.

SVD:

- Main timestep embedding: `Timesteps(320, flip_sin_to_cos=True,
  downscale_freq_shift=0)` then `TimestepEmbedding(320 -> 1280)`.
- Added-time ids `[fps-1, motion_bucket_id, noise_aug_strength]` are flattened,
  embedded with `Timesteps(256, True, 0)`, reshaped to `[B,768]`, passed through
  `TimestepEmbedding(768 -> 1280)`, and added to the main time embedding.
- Temporal block frame positions use `torch.arange(num_frames)` through
  `Timesteps(in_channels, True, 0)` and `TimestepEmbedding(in_channels ->
  in_channels)`.
- `AlphaBlender` applies `alpha * spatial + (1-alpha) * temporal`, where alpha
  is either learned sigmoid or image-only override.

Dynamic dependencies:

- Frame count changes temporal attention sequence length, temporal positional
  embeddings, SVD guidance ramp, and VAE decode chunk behavior.
- SVD `fps`, `motion_bucket_id`, and `noise_aug_strength` affect added-time ids
  and can be cached per request.
- AnimateDiff prompt length and SDXL dual-encoder composition follow their base
  SD reports.

## 10. Preprocessing and input packing

AnimateDiff CPU/data pipeline:

- Tokenize text with CLIP tokenizer or accept prompt embeds.
- Apply textual inversion/LoRA scaling if loaders are active; separate
  candidate.
- Optional IP-Adapter image preprocessing and image embedding preparation;
  separate candidate.

AnimateDiff GPU/runtime:

- Generate `[B,4,F,H/8,W/8]` latents.
- Repeat prompt embeds over frames to match flattened frame batch.
- Decode final latents frame-by-frame through `AutoencoderKL`.

SVD CPU/data pipeline:

- Validate image input and dimensions divisible by 8.
- CLIP image path: preprocess without resize, normalize before antialias resize
  to 224, denormalize, then CLIP feature extractor normalization.
- VAE image path: resize/preprocess to output height/width and add Gaussian
  image noise with `noise_aug_strength`.

SVD GPU/runtime:

- VAE encode conditioning image with posterior `mode()`.
- Repeat conditioning latents across frames and concatenate with noisy video
  latents at each denoising step.
- Build added-time id tensor and per-frame guidance scale tensor.
- Decode `[B,F,4,H,W]` latents by flattening frames and passing `num_frames` to
  the temporal decoder.

## 11. Graph rewrite / lowering opportunities

### Rewrite: AnimateDiff temporal module canonicalization

Source pattern:

```text
[B*F,C,H,W] -> reshape/permute -> GroupNorm over [B,C,F,H,W]
  -> reshape [B*H*W,F,C] -> Linear -> BasicTransformerBlock -> Linear
  -> reshape/permute -> [B*F,C,H,W] -> residual add
```

Replacement:

```text
TemporalAttentionIsland(C, F, H, W, heads, head_dim)
```

Preconditions:

- `num_frames <= motion_max_seq_length`.
- No temporal cross-attention, no IP-Adapter processor inside the motion module.
- Source layout is preserved at island boundaries or all axis rewrites are
  explicit.

Failure cases:

- PIA adapters with `conv_in_channels`, FreeNoise altered latent preparation,
  or SDXL adapters with different block counts need separate guards.

Parity sketch: compare one `AnimateDiffTransformer3D` for C=320/640/1280,
F=16/32, H/W representative latent resolutions.

### Rewrite: SVD BFCHW channel concat + conv_in

Source pattern:

```text
model_input [B,F,4,H,W], image_latents [B,F,4,H,W]
cat dim=2 -> [B,F,8,H,W] -> flatten frames -> Conv2d(8 -> 320)
```

Replacement:

```text
Flatten frames first, then fused/packed Conv2d input construction
```

Preconditions:

- Image latents are frame-repeated and static over denoising steps.
- Channel order is exactly `[noisy_latents, image_latents]`.
- Layout pass rewrites channel dim 2 for BFCHW; no silent NCHW assumption.

Failure cases:

- Future SVD control variants or altered conditioning channel counts.

### Rewrite: guarded NCHW/NCDHW -> NHWC/NDHWC conv islands

Source pattern:

```text
Conv2d/Conv3d + GroupNorm + SiLU + residual + up/downsample
```

Replacement:

```text
NHWC/NDHWC local island with boundary transposes elided when all consumers agree
```

Preconditions:

- GroupNorm channel axis rewrite `dim=1 -> dim=-1`.
- Conv weights transformed from OIHW/OITHW to HWIO/THWIO.
- AlphaBlender, temporal cache/reshape, VAE scaling, and attention flatten
  regions are either rewritten with proven axis mapping or protected by
  `no_layout_translation()`.

Failure cases:

- Temporal token reshape order changes; SVD dim-2 channel concat not rewritten;
  latent mean/scale broadcasting assumes channel dim 1.

### Rewrite: per-frame CFG ramp fusion

Source pattern:

```text
noise = uncond + guidance_scale[B,F,1,1,1] * (cond - uncond)
```

Replacement: fused elementwise kernel over BFCHW video latents.

Preconditions: CFG batch split is exactly first/second half and guidance ramp is
already materialized in latent dtype.

Parity sketch: random BFCHW tensors with scalar AnimateDiff and per-frame SVD
guidance.

## 12. Kernel fusion candidates

Highest priority:

- AnimateDiff SD UNet ResNet Conv2d + GroupNorm + SiLU blocks, reused from SD1.
- AnimateDiff temporal attention island: GroupNorm + temporal QKV/proj + GEGLU.
- SVD SpatioTemporalResBlock: spatial ResNet + temporal Conv3d + AlphaBlender.
- SVD Euler v-pred scheduler step and per-frame CFG ramp.
- SVD `conv_in` after channel concat with static image latents.

Medium priority:

- SVD `TransformerSpatioTemporalModel` spatial and temporal attention kernels.
- AutoencoderKLTemporalDecoder temporal decode blocks and final Conv3d.
- VAE scale/unscale and image noise augmentation pointwise kernels.
- Prompt/image embedding repetition and flatten/reshape elimination.

Lower priority:

- FreeNoise/FreeInit latent preparation.
- AnimateLCM/Lightning distilled loop specialization before base AnimateDiff
  parity.
- Decode chunk scheduling as a memory policy.

## 13. Runtime staging plan

Stage 1: Reuse SD1 config/weight loading and admit `MotionAdapter` config
parsing. Validate adapter/base UNet compatibility without compiling the full
pipeline.

Stage 2: Compile one AnimateDiff motion module with externally supplied hidden
states and verify temporal attention reshape parity.

Stage 3: Compile one `UNetMotionModel` down/mid/up slice at small SD1 shape with
external prompt embeds and scheduler stub.

Stage 4: Full AnimateDiff one denoising step: `[B,4,F,H,W]` latents, prompt
embeds repeated by frame, scalar CFG, DDIM epsilon scheduler in host code.

Stage 5: Add AutoencoderKL frame decode and short text-to-video smoke.

Stage 6: Admit SVD separately: parse SVD UNet/VAE/image encoder/scheduler
configs, accept external image embeddings and image latents first.

Stage 7: Compile SVD one denoising step with BFCHW latents, 8-channel concat,
added-time ids, per-frame CFG, and Euler v-pred/Karras scheduler in host code.

Stage 8: Add `AutoencoderKLTemporalDecoder` decode parity and image VAE encode
for SVD conditioning.

Stage 9: Optimize NHWC/NDHWC guarded islands, attention providers, and scheduler
fusions.

First admission recommendation: start with `animatediff_sd1_motion_module_step`.
It reuses SD1 UNet/VAE/scheduler coverage and adds the smallest new video
surface: temporal attention modules over 2D latent features. Admit SVD second as
`svd_unet_spatiotemporal_step` because it requires a distinct BFCHW contract,
8-channel conditioning concat, CLIP image conditioning, temporal ResNet blocks,
temporal decoder VAE, and exact Euler v-pred/Karras scheduler.

## 14. Parity and validation plan

- `MotionAdapter` config compatibility tests for SD1 four-block, SDXL
  three-block, mid-block enabled/disabled, and invalid layer count.
- `AnimateDiffTransformer3D` random tensor parity for C=320/640/1280,
  F=16/25/32.
- One AnimateDiff down block and up block parity with fixed prompt embeddings.
- Full `UNetMotionModel` tiny-shape parity at one timestep.
- AnimateDiff CFG and scheduler step parity with DDIM epsilon.
- SVD preprocessing parity for CLIP image path and VAE image path.
- SVD `UNetSpatioTemporalConditionModel` one-step parity for img2vid 14 and XT
  25 frame configs.
- SVD added-time embedding parity for fps decrement and noise strength float.
- SVD Euler v-pred/Karras `set_timesteps`, `scale_model_input`, and `step`
  parity, including custom `sigmas`.
- `AutoencoderKLTemporalDecoder` decode parity with decode chunks 1, 8, 14/25.
- End-to-end smoke: fixed latents/image embeddings/image latents; compare latent
  loop outputs before decoding.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5` for blocks and
`rtol=1e-5, atol=1e-6` for pure scheduler arithmetic; fp16/bf16 start at
`rtol=2e-2, atol=2e-2` and tighten by provider.

## 15. Performance probes

- AnimateDiff one denoiser step by frames F=8/16/25/32 and latent resolution.
- Temporal attention cost split from spatial SD UNet cost.
- Motion module sequence-length sweep and attention backend comparison.
- AnimateDiff VAE frame decode throughput with decode chunk sizes.
- SVD one denoiser step by frames 14 vs 25 and 576x1024 vs smaller synthetic.
- SVD spatial vs temporal ResNet/attention time split.
- SVD Euler scheduler + per-frame CFG overhead as percentage of denoiser step.
- SVD temporal VAE decode throughput by chunk size; quality/parity check for
  chunk-size-sensitive temporal decoder behavior.
- Faithful NCHW/BFCHW path vs guarded NHWC/NDHWC conv islands.
- VRAM/workspace for CFG batched vs separate positive/negative denoiser calls.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `animatediff_sdxl`: SDXL dual text conditioning and 3-block motion adapter.
- `animatediff_video2video`: input video encode, strength slicing, scheduler
  `add_noise`.
- `animatediff_controlnet`: ControlNet residuals into motion UNet.
- `animatediff_sparsectrl`: sparse control frames and masks.
- `animatediff_ip_adapter`: image K/V attention processors and image projection
  layers.
- `animatediff_lcm_lightning`: LCM/Lightning distilled scheduler and step-count
  contracts; ByteDance root config was unavailable in this audit.
- `svd_temporal_controlnet`: community SVD temporal control branches.
- `autoencoder_kl_temporal_decoder`: focused temporal decoder VAE optimization
  if SVD decode becomes a bottleneck.

Ignored/out of scope:

- XLA/NPU/MPS/Flax/ONNX branches.
- Safety checker/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupts.
- FreeNoise/FreeInit as first-slice runtime behavior.

## 17. Final implementation checklist

- [ ] Parse AnimateDiff `MotionAdapter` configs and validate base UNet compatibility.
- [ ] Load SD1 base UNet weights plus motion adapter weights into explicit artifact state.
- [ ] Implement `AnimateDiffTransformer3D` reshape/norm/temporal-attention parity.
- [ ] Compile one `UNetMotionModel` step with external prompt embeddings.
- [ ] Implement AnimateDiff scalar CFG and selected first scheduler slice, likely DDIM epsilon.
- [ ] Reuse AutoencoderKL frame decode from SD reports for AnimateDiff output.
- [ ] Parse SVD UNet/VAE/scheduler/image encoder configs.
- [ ] Represent SVD BFCHW latent contract and 8-channel conditioning concat.
- [ ] Implement SVD added-time embeddings and per-frame guidance ramp.
- [ ] Implement EulerDiscrete v-pred continuous Karras scheduler parity.
- [ ] Compile one `UNetSpatioTemporalConditionModel` step with external image embeddings/latents.
- [ ] Implement or call `AutoencoderKLTemporalDecoder` decode with explicit chunk policy.
- [ ] Add VAE image encode mode path for SVD conditioning.
- [ ] Add guarded NHWC/NDHWC conv-island rewrites only after faithful layout parity.
- [ ] Benchmark temporal attention, spatio-temporal blocks, scheduler/CFG, and temporal VAE decode separately.
