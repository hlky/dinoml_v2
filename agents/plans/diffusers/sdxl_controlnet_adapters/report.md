# SDXL Control and Adapter Variant Audit

Target slug: `sdxl_controlnet_adapters`

This report treats three related SDXL extension surfaces as separate admission
candidates: ControlNet SDXL, T2I-Adapter SDXL, and IP-Adapter SDXL.

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Base SDXL reference:
    stabilityai/stable-diffusion-xl-base-1.0
  ControlNet SDXL:
    diffusers/controlnet-canny-sdxl-1.0
    diffusers/controlnet-canny-sdxl-1.0-small
    diffusers/controlnet-depth-sdxl-1.0
    diffusers/controlnet-zoe-depth-sdxl-1.0
  T2I-Adapter SDXL:
    TencentARC/t2i-adapter-canny-sdxl-1.0
    TencentARC/t2i-adapter-depth-midas-sdxl-1.0
    TencentARC/t2i-adapter-depth-zoe-sdxl-1.0
    TencentARC/t2i-adapter-sketch-sdxl-1.0
    Adapter/t2iadapter, sketch_sdxl_1.0 from the SD1 adapter report
  IP-Adapter SDXL:
    h94/IP-Adapter, sdxl_models/*

Config sources:
  Local cache checked first:
    H:/configs/stabilityai/stable-diffusion-xl-base-1.0/
    H:/configs/diffusers/controlnet-canny-sdxl-1.0/model_index.json
    H:/configs/diffusers/controlnet-canny-sdxl-1.0-small/model_index.json
    H:/configs/diffusers/controlnet-depth-sdxl-1.0/model_index.json
    H:/configs/diffusers/controlnet-zoe-depth-sdxl-1.0/model_index.json
    H:/configs/TencentARC/t2i-adapter-canny-sdxl-1.0/model_index.json
    H:/configs/TencentARC/t2i-adapter-depth-midas-sdxl-1.0/model_index.json
    H:/configs/TencentARC/t2i-adapter-depth-zoe-sdxl-1.0/model_index.json
    H:/configs/TencentARC/t2i-adapter-sketch-sdxl-1.0/model_index.json
    H:/configs/h94/IP-Adapter/model_index.json
  The SDXL base component configs were already present/fetched for the SDXL
  base report. The ControlNet, T2I-Adapter, and IP-Adapter local cache entries
  above were placeholder model indexes only, so component configs and
  safetensors headers were inspected from official Hugging Face repo URLs
  without saving them, because this task owns only this report path.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet/pipeline_controlnet_inpaint_sd_xl.py
  X:/H/diffusers/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet.py
  X:/H/diffusers/src/diffusers/models/controlnets/multicontrolnet.py
  X:/H/diffusers/src/diffusers/models/adapter.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_condition.py
  X:/H/diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/loaders/ip_adapter.py
  X:/H/diffusers/src/diffusers/loaders/unet.py
  X:/H/diffusers/src/diffusers/image_processor.py
  Existing reports:
    agents/plans/diffusers/stable_diffusion_xl/report.md
    agents/plans/diffusers/controlnet_sd/report.md
    agents/plans/diffusers/sd1_t2i_adapter/report.md
    agents/plans/diffusers/sd1_ip_adapter/report.md

External component configs inspected:
  h94/IP-Adapter/sdxl_models/image_encoder/config.json
  h94/IP-Adapter SDXL safetensors headers for standard and Plus variants.

Any missing files or assumptions:
  No official config path remained blocked. Safety/NSFW, callbacks/interrupt,
  training/loss/dropout/gradient checkpointing, multi-GPU/context parallel,
  XLA/NPU/MPS, Flax, and ONNX paths are out of scope. The runtime target is
  inference-only CUDA. Faithful NCHW translation is the semantic first path;
  NHWC/channel-last is only a guarded optimization.
```

## 2. Pipeline and component graph

All three candidates extend the SDXL base denoising loop:

```text
prompt / prompt_2 or cached embeddings
  -> dual CLIP text encoders, or external prompt embeddings
  -> prompt_embeds [B,77,2048], pooled_prompt_embeds [B,1280]
  -> added time IDs: original size, crop top-left, target size
  -> latent initialization [B,4,H/8,W/8]
  -> denoising loop:
       scheduler.scale_model_input
       -> optional control/adapter/IP side path
       -> UNet2DConditionModel with text_time and optional extension tensors
       -> CFG and optional guidance_rescale
       -> scheduler.step
  -> AutoencoderKL decode and image postprocess
```

Candidate split:

| Candidate | Classes/files | Runtime delta from SDXL base |
| --- | --- | --- |
| ControlNet SDXL | `StableDiffusionXLControlNetPipeline`, `StableDiffusionXLControlNetImg2ImgPipeline`, `StableDiffusionXLControlNetInpaintPipeline`, `ControlNetModel`, `MultiControlNetModel` | Runs ControlNet every timestep. Its down/mid residual tensors are scaled and injected into the SDXL UNet through `down_block_additional_residuals` and `mid_block_additional_residual`. |
| T2I-Adapter SDXL | `StableDiffusionXLAdapterPipeline`, `T2IAdapter`, `FullAdapterXL`, `MultiAdapter` | Runs a small adapter once before denoising. Feature tensors are duplicated for CFG and injected through `down_intrablock_additional_residuals`; SDXL can also consume a shape-matched remaining adapter tensor at the mid block. |
| IP-Adapter SDXL | `IPAdapterMixin`, `UNet2DConditionLoadersMixin`, `MultiIPAdapterImageProjection`, `ImageProjection`, `IPAdapterPlusImageProjection`, `IPAdapterAttnProcessor2_0` | Mutates SDXL UNet cross-attention processors so text attention is augmented by image-token K/V attention branches. No separate denoiser is added. |

Cacheable stages: text/pool embeddings, added time IDs for fixed size/crop,
ControlNet condition image tensors, T2I-Adapter feature pyramids, IP-Adapter
image embeds/projected image tokens, scheduler timesteps/sigmas, and supplied
latents. ControlNet is not loop-invariant; T2I-Adapter and IP image projection
are loop-invariant.

## 3. Important config dimensions

Base SDXL dimensions inherited from `stable_diffusion_xl`:

| Field | SDXL base value |
| --- | --- |
| Latent shape at 1024 | `[B,4,128,128]` NCHW |
| Prompt hidden | `[B,77,2048]` from CLIP-L 768 + CLIP-bigG 1280 |
| Pooled text | `[B,1280]` |
| Added time IDs | six scalars, projected by `addition_time_embed_dim=256` |
| Added embed input | `1280 + 6*256 = 2816` |
| UNet channels | 320, 640, 1280 |
| Cross-attention dim | 2048 |
| Transformer layers/block | 1, 2, 10 |
| Scheduler default | EulerDiscrete for base, epsilon prediction |

ControlNet SDXL configs:

| Repo | blocks | channels | layers | cross dim | attention head dim | transformer layers | conditioning embed | text_time |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `diffusers/controlnet-canny-sdxl-1.0` | Down, CA, CA | 320,640,1280 | 2 | 2048 | 5,10,20 | 1,2,10 | 16,32,96,256 | yes, 2816 |
| `diffusers/controlnet-depth-sdxl-1.0` | Down, CA, CA | 320,640,1280 | 2 | 2048 | 5,10,20 | 1,2,10 | 16,32,96,256 | yes, 2816 |
| `diffusers/controlnet-zoe-depth-sdxl-1.0` | Down, CA, CA | 320,640,1280 | 2 | 2048 | 5,10,20 | 1,2,10 | 16,32,96,256 | yes, 2816 |
| `diffusers/controlnet-canny-sdxl-1.0-small` | Down, Down, Down | 320,640,1280 | 2 | 2048 | 5,10,20 | 0,0,0 | 16,32,96,256 | yes, 2816 |

`num_attention_heads` is omitted in these ControlNet configs. Source maps it to
`attention_head_dim` for backward compatibility, so this is the SDXL convention
where values 5/10/20 are treated as attention heads and imply 64-wide heads at
320/640/1280 channels.

T2I-Adapter SDXL configs:

| Repo | class | adapter type | input channels | channels | residual blocks | downscale | feature slots at 1024 |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `TencentARC/t2i-adapter-canny-sdxl-1.0` | `T2IAdapter` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 2 | 16 | 320@64, 640@64, 1280@32, 1280@32 |
| `TencentARC/t2i-adapter-depth-midas-sdxl-1.0` | `T2IAdapter` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 2 | 16 | same |
| `TencentARC/t2i-adapter-depth-zoe-sdxl-1.0` | `T2IAdapter` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 2 | 16 | same |
| `TencentARC/t2i-adapter-sketch-sdxl-1.0` | `T2IAdapter` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 2 | 16 | same |

IP-Adapter SDXL inspected configs/headers:

| Weight/config | Image input | Projection signal | Image tokens | Cross-attn dim | IP K/V examples |
| --- | --- | --- | ---: | ---: | --- |
| `sdxl_models/image_encoder/config.json` | CLIP vision hidden 1664, projection 1280, 48 layers, 16 heads, image 224, patch 14 | external image encoder | n/a | n/a | n/a |
| `ip-adapter_sdxl.safetensors` | pooled 1280 | `image_proj.proj.weight [8192,1280]` | 4 | 2048 | `to_k_ip [640,2048]`, later `[1280,2048]` |
| `ip-adapter_sdxl_vit-h.safetensors` | pooled 1024 | `image_proj.proj.weight [8192,1024]` | 4 | 2048 | same UNet-side widths |
| `ip-adapter-plus_sdxl_vit-h.safetensors` | hidden sequence 1280 | latents `[1,16,1280]`, proj out `[2048,1280]` | 16 | 2048 | same UNet-side widths |
| `ip-adapter-plus-face_sdxl_vit-h.safetensors` | hidden sequence 1280 | Plus projection with 16 latents | 16 | 2048 | same UNet-side widths |

Recommended first Dinoml scheduler slice for all three candidates is the SDXL
base EulerDiscrete loop with scheduler in host control. The extension surfaces
are mostly scheduler-independent except for latent scaling, timestep selection,
and per-step gating.

## 3a. Family variation traps

- All three candidates require SDXL `addition_embed_type="text_time"`:
  `added_cond_kwargs` must include pooled text embeddings and time IDs.
- ControlNet SDXL differs from SD1 ControlNet by dual text embeddings,
  2048-wide cross attention, 2816 added time/text embedding input, and only
  three down blocks for base SDXL configs instead of four SD1 blocks.
- ControlNet small configs preserve residual tensor channels/shapes but remove
  cross-attention blocks. Do not use the small config as proof that XL
  ControlNet attention is absent.
- T2I-Adapter XL uses `PixelUnshuffle(16)` and the `FullAdapterXL` shape pattern
  `[320,64,64]`, `[640,64,64]`, `[1280,32,32]`, `[1280,32,32]` for 1024 input.
  This is not the SD1 adapter's 64/32/16/8 feature pyramid.
- SDXL adapter features are added after the first DownBlock2D has already
  downsampled to 64x64, then into the following cross-attention down blocks.
- IP-Adapter SDXL has 2048-dim image tokens and K/V projection inputs, unlike
  SD1's 768-dim cross-attention. It can be combined with ControlNet or
  T2I-Adapter in the same SDXL pipelines.
- Source tensors are NCHW. NHWC/channel-last needs explicit guards around
  GroupNorm axes, PixelUnshuffle channel expansion, channel concat, spatial
  means, attention flatten/restore, mask resize/downsample, and residual ABI.
- Loader mutations matter for IP-Adapter: UNet `encoder_hid_dim_type` becomes
  `ip_image_proj`, attention processors are replaced, and scale is stored on
  processor objects unless Dinoml makes it artifact-visible.

## 4. Runtime tensor contract

For 1024x1024 SDXL with CFG:

| Boundary | Base tensor | Shape/layout |
| --- | --- | --- |
| prompt embeddings | negative/positive concatenated | `[2B,77,2048]` |
| pooled embeddings | negative/positive concatenated | `[2B,1280]` |
| time IDs | negative/positive concatenated and repeated | `[2B,6]` |
| latents | denoising state | `[B,4,128,128]` NCHW |
| latent model input | CFG duplicate after scheduler scale | `[2B,4,128,128]` NCHW |
| UNet output | noise prediction before CFG chunk | `[2B,4,128,128]` NCHW |
| VAE decode input | `latents / 0.13025` | `[B,4,128,128]` NCHW |

ControlNet SDXL adds:

| Tensor | Shape/layout at 1024 | Notes |
| --- | --- | --- |
| control image | `[B or 2B,3,1024,1024]` NCHW | `VaeImageProcessor`, RGB, `[0,1]`, no VAE normalization |
| ControlNet input | `[B or 2B,4,128,128]` NCHW | conditional-only batch in guess mode under CFG |
| down residual slots | 9 tensors | `[320,128,128]`, `320@128`, `320@128`, `320@64`, `640@64`, `640@64`, `640@32`, `1280@32`, `1280@32` |
| mid residual | `[B or 2B,1280,32,32]` | added to SDXL UNet mid sample |

T2I-Adapter SDXL adds:

| Tensor | Shape/layout at 1024 | Notes |
| --- | --- | --- |
| adapter input image | `[B,3,1024,1024]` NCHW | PIL path resizes with Lanczos and scales to `[0,1]`; tensor path is trusted |
| unshuffle output | `[B,3*256,64,64]` | `PixelUnshuffle(16)` |
| adapter features | `[B,320,64,64]`, `[B,640,64,64]`, `[B,1280,32,32]`, `[B,1280,32,32]` | loop-invariant |
| CFG adapter features | `[2B,...]` | duplicated on batch axis |
| UNet argument | `down_intrablock_additional_residuals` | cloned each step because UNet pops entries |

IP-Adapter SDXL adds:

| Tensor | Shape/layout | Notes |
| --- | --- | --- |
| image encoder pooled output | `[B,1280]` or `[B,1024]` | standard weights differ by image encoder/projection variant |
| image encoder hidden output | `[B,257,1280]` for Plus-style paths | hidden-state path uses penultimate CLIP vision states |
| projected image embeds | standard `[B,num_images,4,2048]`; Plus `[B,num_images,16,2048]` | wrapped in list, one element per loaded adapter |
| UNet encoder hidden states | `(text_tokens, image_embeds_list)` | produced by `process_encoder_hidden_states` |
| IP attention branch | same query length as latent spatial attention | separate image K/V attention, added before output projection |

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent, image, control, and adapter tensors.
- Batch `cat`, `chunk`, `repeat`, `repeat_interleave`, list/tuple residual
  slot management.
- Channel concat for inpaint variants and PixelUnshuffle channel expansion.
- Spatial resize/downsample for control images and optional IP masks.
- `text_time` flatten/project/reshape/concat path.

Convolution/downsample/upsample ops:

- ControlNet conditioning embedding: Conv2d stack from 3 channels through
  16/32/96/256 to 320.
- ControlNet down blocks: SDXL-shaped Conv2d, ResnetBlock2D, Downsample2D, and
  zero 1x1 residual heads.
- T2I-Adapter XL: `PixelUnshuffle(16)`, Conv2d `768 -> 320`, AdapterBlock
  conv/ReLU/1x1 residual blocks, one AvgPool2d ceil stride-2 block.
- Base SDXL UNet/VAE conv, downsample, upsample, and VAE decode inherited.

GEMM/linear ops:

- SDXL timestep and added text/time embedding MLPs.
- ControlNet/UNet attention projections and feed-forward projections.
- IP-Adapter standard projection: Linear `1280 or 1024 -> 8192`, reshape to
  four 2048-dim tokens, LayerNorm.
- IP-Adapter Plus projection: `proj_in`, learned latents, Perceiver-style
  attention/feed-forward blocks, `proj_out -> 2048`, LayerNorm.
- IP K/V projections per cross-attention processor:
  Linear `2048 -> hidden_size` for hidden sizes 320/640/1280.

Attention primitives:

- SDXL UNet and ControlNet noncausal self/cross attention, cross dim 2048.
- IP-Adapter extra image K/V attention branches with shared latent query.
- Optional IP spatial masks multiply branch outputs after attention.

Normalization and adaptive conditioning:

- GroupNorm over channel axis, LayerNorm over token dim, SiLU/ReLU/GELU/GEGLU.
- SDXL `text_time` conditioning added to time embedding.
- No new adapter timestep embedding for T2I-Adapter.

Scheduler and guidance arithmetic:

- EulerDiscrete first slice: `scale_model_input`, CFG, optional
  guidance_rescale, `step`.
- ControlNet `control_guidance_start/end` per-step keep gates.
- T2I-Adapter `adapter_conditioning_factor` step gate.
- IP-Adapter per-layer/per-adapter scale.

## 6. Model breakdown

ControlNet SDXL:

```text
control image -> ControlNetConditioningEmbedding -> [B,320,128,128]
latents -> Conv2d(4 -> 320)
sample += condition embedding
down path:
  DownBlock2D 320, then CrossAttnDownBlock2D 640, then CrossAttnDownBlock2D 1280
  transformer layers per block 1,2,10 for full configs
mid:
  UNetMidBlock2DCrossAttn at 1280,32,32
residual heads:
  zero 1x1 conv for every saved down sample and mid sample
scale:
  conditioning_scale, optional guess-mode logspace scale, optional global pool
```

The SDXL UNet receives the resulting residual tuple after its own down path and
adds the mid residual before up blocks. MultiControlNet runs multiple
ControlNets and sums matching residual slots.

T2I-Adapter SDXL:

```text
condition image [B,3,1024,1024]
  -> PixelUnshuffle(16): [B,768,64,64]
  -> Conv2d(768 -> 320, 3x3)
  -> AdapterBlock 320 -> 320, no down: feature 0 [B,320,64,64]
  -> AdapterBlock 320 -> 640, no down: feature 1 [B,640,64,64]
  -> AdapterBlock 640 -> 1280, AvgPool2d stride 2: feature 2 [B,1280,32,32]
  -> AdapterBlock 1280 -> 1280, no down: feature 3 [B,1280,32,32]
```

UNet injection uses `down_intrablock_additional_residuals`. For the non-cross
first SDXL down block, the feature is added to the block output after
downsampling. For cross-attention down blocks, the feature is passed into the
block as `additional_residuals`. If a feature remains after the mid block and
matches the mid sample shape, SDXL source adds it to the mid sample.

IP-Adapter SDXL:

```text
image embeds list
  -> MultiIPAdapterImageProjection
  -> projected image tokens, one tensor per adapter
UNet process_encoder_hidden_states:
  text tokens [B,77,2048]
  -> (text tokens, projected image tokens)
IPAdapterAttnProcessor2_0 cross-attention:
  base = SDPA(Q_latent, K_text, V_text, text_mask)
  ip_j = SDPA(Q_latent, K_image_j, V_image_j, None)
  hidden = base + sum(scale_j * ip_j)
  -> output projection -> residual/reshape
```

## 7. Attention requirements

ControlNet SDXL requires the same SDXL `BasicTransformerBlock` attention as the
base UNet, duplicated in ControlNet full/depth/zoe models. Query tokens are
latent spatial tokens at 64x64 and 32x32 resolutions for the inspected
ControlNet configs; text K/V length is normally 77 with width 2048.

T2I-Adapter SDXL adds no attention in the adapter itself. It changes UNet hidden
states before/downstream of attention blocks, so block parity must include exact
intrablock addition sites.

IP-Adapter SDXL changes the attention contract. `attention_processor.py` is the
semantic source: `IPAdapterAttnProcessor2_0` uses PyTorch SDPA for text attention
and one independent image attention per adapter. Eager `IPAdapterAttnProcessor`
is the fallback parity path. A Dinoml flash-style provider is valid for each
branch under dense, noncausal, dropout-0, supported dtype/head-dim guards. A
single concatenated text+image K/V attention is not generally equivalent because
IP scales and optional masks are applied branch-wise.

## 8. Scheduler and denoising-loop contract

ControlNet SDXL per step:

```text
latent_model_input = cat([latents] * 2) if CFG else latents
latent_model_input = scheduler.scale_model_input(latent_model_input, t)
added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
down_res, mid_res = controlnet(control_model_input, t, prompt_embeds,
                               controlnet_cond=image,
                               conditioning_scale=cond_scale,
                               added_cond_kwargs=controlnet_added_cond_kwargs)
if guess_mode and CFG:
  residuals = cat([zeros_like(res), res])
noise_pred = unet(..., down_block_additional_residuals=down_res,
                  mid_block_additional_residual=mid_res,
                  added_cond_kwargs=added_cond_kwargs)
latents = scheduler.step(CFG(noise_pred), t, latents)
```

T2I-Adapter SDXL computes `adapter_state` once before the loop, scales and
duplicates it, then supplies cloned features only while
`i < int(num_inference_steps * adapter_conditioning_factor)`.

IP-Adapter SDXL prepares image embeds once, inserts them into
`added_cond_kwargs["image_embeds"]`, and relies on the mutated UNet attention
processors every step. Scheduler math is otherwise unchanged.

Keep timestep iteration, scheduler state, denoising range, ControlNet keep
gates, adapter step gates, and dynamic callback paths in host-visible control
initially. Compile one denoiser step once the extension tensors are explicit.

## 9. Position, timestep, and custom math

All candidates inherit SDXL sinusoidal timestep embeddings and added size/crop
conditioning:

```text
add_time_ids = [original_h, original_w, crop_top, crop_left, target_h, target_w]
time_embeds = add_time_proj(flatten(add_time_ids)).reshape(batch, -1)
add_embeds = concat(pooled_text_embeds, time_embeds)
```

ControlNet guess mode scales residuals with a logspace ramp from 0.1 to 1.0
before multiplying by conditioning scale. Global pooling, when configured,
reduces residuals over NCHW spatial axes `(2,3)`.

T2I-Adapter custom math is PixelUnshuffle plus small residual conv blocks. The
XL variant has only one adapter downsampling block after the initial unshuffle.

IP-Adapter custom math is branch-wise image attention:

```text
hidden = text_attention(Q, K_text, V_text)
for each adapter:
  hidden += scale * image_attention(Q, K_ip, V_ip)
```

With IP masks, mask downsampling maps spatial masks to latent query tokens and
multiplies the image-attention output after attention.

## 10. Preprocessing and input packing

ControlNet SDXL uses `VaeImageProcessor` with RGB conversion and no
normalization for control images. Image tensors are NCHW in `[0,1]`, repeated or
CFG-duplicated unless guess mode uses conditional-only ControlNet.

T2I-Adapter SDXL uses `_preprocess_adapter_image`: PIL images are resized with
Lanczos, converted to float32 `[0,1]`, transposed NHWC to NCHW, and tensors are
passed through directly. `_default_height_width` rounds down to a multiple of
`adapter.downscale_factor` (16 for XL).

IP-Adapter SDXL accepts either `ip_adapter_image` or precomputed
`ip_adapter_image_embeds`. Image inputs go through a CLIP image processor and
vision encoder; precomputed embeds bypass that. Under CFG, negative/positive
image embeds are concatenated in the same batch order as prompt embeddings.

Img2img and inpaint variants add VAE encode, strength/timestep slicing, mask
latents, and for inpaint the possible channel concat:
`[latents, mask, masked_image_latents]`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ControlNet SDXL as shared SDXL down/mid lowering

Source pattern: ControlNet down/mid blocks mirror SDXL UNet down/mid block
classes, then pass every saved state through zero 1x1 residual heads.

Replacement: reuse SDXL down/mid block lowerings and add explicit residual-head
nodes and residual ABI metadata.

Preconditions: block channels, block types, transformer layer counts,
`text_time` dimensions, attention head mapping, and cross-attention dim match
the ControlNet config. Failure cases: ControlNet union, small no-attention
configs, global pooling, or mixed SD1/SDXL residual slot shapes.

### Rewrite: T2I-Adapter XL feature extractor

Source pattern: `PixelUnshuffle(16) -> Conv2d -> AdapterBlock*`.

Replacement: explicit PixelUnshuffle plus conv island first; later fuse
PixelUnshuffle+Conv when input layout, channel count, and divisibility are
static.

Preconditions: NCHW input, H/W divisible by 16, adapter type
`full_adapter_xl`, static feature-slot count. Failure cases: tensor inputs with
unexpected layout, non-XL adapter types, or dimensions where adapter features
do not match UNet injection states.

### Rewrite: IP-Adapter shared-Q attention branch

Source pattern: text attention plus one or more image attention branches using
the same Q.

Replacement: compound attention node or explicit text attention, image
attention, scale/add, and output projection.

Preconditions: `IPAdapterAttnProcessor2_0` or eager equivalent, no spatial IP
masks for first slice, fixed adapter count, fixed token counts, dropout 0,
noncausal dense attention. Failure cases: masked multi-image IP path, FaceID
LoRA coupling, SD3/Flux processors, or dynamic processor mutation after compile.

### Rewrite: guarded NHWC conv islands

Source pattern: ControlNet conditioning/down blocks, T2I-Adapter conv blocks,
UNet/VAE conv regions.

Replacement: NCHW boundary -> NHWC internal conv island -> NCHW boundary, with
OIHW to HWIO weights.

Preconditions: all GroupNorm/channel concat/PixelUnshuffle/pooling/spatial mean
axes rewritten and all consumers in the island are layout-aware. Failure cases:
attention flatten boundaries, residual tuple ABI, inpaint channel concat, IP
mask downsample, or scheduler broadcasting that assumes NCHW.

## 12. Kernel fusion candidates

Highest priority:

- SDXL `text_time` embedding path shared by all three candidates.
- ControlNet conditioning embedding conv stack and zero 1x1 residual heads.
- ControlNet/UNet residual scale/add injection.
- T2I-Adapter XL PixelUnshuffle + first Conv2d, then Conv/ReLU/1x1 residual
  blocks.
- IP-Adapter standard image projection and added K/V projections.
- SDXL cross-attention and branch-wise IP attention lowering.

Medium priority:

- MultiControlNet residual accumulation and guess-mode zero padding.
- T2I-Adapter feature cache and `adapter_conditioning_factor` gating.
- IP-Adapter Plus projection/resampler.
- IP mask downsample and masked branch output multiply/add.
- Guarded NHWC conv islands for ControlNet/T2I/UNet/VAE regions.

Lower priority:

- ControlNet small no-attention specialization.
- Full inpaint concat/blend variants.
- FaceID/LoRA-coupled IP paths.
- xFormers-specific processor parity.

## 13. Runtime staging plan

Recommended first admission order:

1. ControlNet SDXL single full config with externally supplied prompt/pool
   embeddings, control image tensor, latents, and Euler scheduler in host
   control. This reuses the largest amount of SDXL base work and proves
   `text_time` side-model residual contracts.
2. T2I-Adapter SDXL single `full_adapter_xl` with preprocessed adapter tensor.
   It is smaller than ControlNet at runtime and loop-invariant, but its
   injection sites differ from ControlNet and need separate tests.
3. IP-Adapter SDXL standard precomputed image-embeds path with one adapter, four
   image tokens, no masks, no Plus resampler. This admits the added-K/V
   attention contract without owning CLIP vision first.

ControlNet Stage 1: parse configs and load one ControlNet plus SDXL base UNet.
Stage 2: ControlNet conditioning embedding and one down block parity. Stage 3:
full ControlNet residual tuple parity. Stage 4: one ControlNet+UNet denoising
step parity.

T2I Stage 1: adapter-only feature pyramid parity. Stage 2: UNet intrablock
addition parity. Stage 3: one denoising step with cached adapter features.

IP Stage 1: accept projected image tokens or lower standard `ImageProjection`.
Stage 2: one cross-attention processor parity. Stage 3: full UNet step with
mutated processors represented in manifest/runtime state.

## 14. Parity and validation plan

- Config parse parity for SDXL base, ControlNet SDXL full/small, T2I XL, and
  IP-Adapter SDXL projection headers.
- `text_time` embedding parity for all candidates.
- ControlNet conditioning embedding parity at 1024 and 768 sizes.
- ControlNet residual slot count/order/shape/dtype parity, including guess-mode
  zero padding and MultiControlNet residual sums.
- T2I-Adapter XL PixelUnshuffle and feature pyramid parity.
- UNet adapter injection parity at every down/mid candidate site.
- IP-Adapter standard projection parity for pooled 1280 and 1024 inputs.
- IP attention processor parity for hidden sizes 320, 640, and 1280 at query
  lengths 4096, 1024, and 64/256 as exercised by SDXL blocks.
- One-step parity for each candidate with scheduler in Python and fixed
  latents/embeddings.
- Short deterministic loop smoke and VAE decode smoke after one-step parity.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2` and tighten per kernel.

## 15. Performance probes

- ControlNet forward time by resolution, batch, dtype, and full vs small config.
- Combined ControlNet+UNet step time and residual injection overhead.
- T2I-Adapter feature extraction time, amortized over denoising step count.
- UNet step with and without T2I features.
- IP-Adapter projection time and per-cross-attention added branch cost.
- Attention backend comparison for base SDXL vs IP branch-wise attention.
- Scheduler/CFG/control-scale/adapter-gate overhead as separate small kernels.
- VAE decode throughput inherited from SDXL base.
- NCHW faithful path versus guarded NHWC conv islands.
- VRAM and temporary tensors for ControlNet residual tuples, cached T2I
  features, and IP image tokens.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `controlnet_sdxl`: the recommended first candidate from this bundle.
- `controlnet_sdxl_img2img`: VAE encode, strength slicing, and ControlNet
  residuals.
- `controlnet_sdxl_inpaint`: masks, masked image latents, 4-channel/9-channel
  contracts plus ControlNet residuals.
- `controlnet_union_sdxl`: control mode/type embeddings and union-specific
  ControlNet logic.
- `t2i_adapter_sdxl`: the second recommended candidate from this bundle.
- `t2i_adapter_sdxl_multi`: MultiAdapter list inputs and feature-slot sums.
- `ip_adapter_sdxl`: the third recommended candidate from this bundle.
- `ip_adapter_sdxl_plus_masks_faceid`: Plus resampler, masks, FaceID, and
  possible LoRA coupling.
- `sdxl_lora_textual_inversion_adapters`: weight/token mutation, separate from
  control tensors.
- Rare scheduler swaps beyond EulerDiscrete first parity.

Ignored/out of scope:

- Safety checker, watermarking, and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Callback mutation and interactive interrupt.
- Multi-GPU/context parallel paths.
- XLA, NPU, MPS, Flax, and ONNX-specific paths.

## 17. Final implementation checklist

- [ ] Parse SDXL base configs and ControlNet/T2I/IP component configs.
- [ ] Represent SDXL `text_time` conditioning as explicit runtime tensors.
- [ ] Load one SDXL ControlNet and preserve residual slot ABI.
- [ ] Implement ControlNet SDXL conditioning embedding and down/mid residual heads.
- [ ] Add ControlNet residual injection into the SDXL UNet.
- [ ] Add one-step ControlNet SDXL parity with EulerDiscrete in host control.
- [ ] Implement T2I-Adapter XL PixelUnshuffle and feature pyramid.
- [ ] Add SDXL UNet `down_intrablock_additional_residuals` lowering.
- [ ] Keep T2I-Adapter features cached outside the denoising loop.
- [ ] Add one-step T2I-Adapter SDXL parity.
- [ ] Represent IP-Adapter processor mutation in manifest/runtime state.
- [ ] Implement standard SDXL IP image projection or accept projected tokens.
- [ ] Add branch-wise image K/V attention for SDXL cross-attention.
- [ ] Expose IP scale as explicit runtime data.
- [ ] Add one-step IP-Adapter SDXL parity without masks.
- [ ] Add targeted performance probes for ControlNet, T2I, IP attention, VAE,
      scheduler/CFG, and memory.
