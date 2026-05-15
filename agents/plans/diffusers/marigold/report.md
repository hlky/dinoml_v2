# Diffusers Marigold Operator and Integration Report

Candidate slug: `marigold`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Primary official configs:
    prs-eth/marigold-depth-v1-1
    prs-eth/marigold-depth-v1-0
    prs-eth/marigold-depth-lcm-v1-0
    prs-eth/marigold-normals-v1-1
    prs-eth/marigold-normals-v0-1
    prs-eth/marigold-normals-lcm-v0-1
    prs-eth/marigold-iid-appearance-v1-1
    prs-eth/marigold-iid-lighting-v1-1
  Variant/config caveat:
    prs-eth/marigold-depth-hr-v1-1 advertises MarigoldDepthHRPipeline, but this
    checkout has no MarigoldDepthHRPipeline class/file.

Config sources:
  H:/configs/prs-eth/<repo>/model_index.json
  H:/configs/prs-eth/<repo>/unet/config.json
  H:/configs/prs-eth/<repo>/vae/config.json
  H:/configs/prs-eth/<repo>/scheduler/scheduler_config.json
  H:/configs/prs-eth/<repo>/text_encoder/config.json
  H:/configs/prs-eth/<repo>/tokenizer/tokenizer_config.json
  Component configs were fetched from official Hub repos with `hf download`
  because the local cache initially contained only model_index.json files.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/marigold/pipeline_marigold_depth.py
  diffusers/src/diffusers/pipelines/marigold/pipeline_marigold_normals.py
  diffusers/src/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py
  diffusers/src/diffusers/pipelines/marigold/marigold_image_processing.py
  diffusers/src/diffusers/pipelines/marigold/__init__.py

Model files inspected:
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_lcm.py
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  CLIPTextModel and CLIPTokenizer configs in the official Marigold repos.

Any missing files or assumptions:
  The report is inference-only CUDA-oriented. It ignores XLA/NPU/MPS/Flax/ONNX,
  safety, training/loss/dropout/gradient checkpointing, callbacks, and
  multi-GPU/context parallel. The empty CLIP text embedding is active source
  behavior, but first Dinoml staging can accept it as a cached tensor boundary.
```

Important class/file anchors:

| Surface | Anchors |
| --- | --- |
| Depth pipeline | `MarigoldDepthPipeline` at `pipeline_marigold_depth.py:104`, `__call__` at `:349`, latent prep at `:621`, decode at `:658`, ensemble at `:673` |
| Normals pipeline | `MarigoldNormalsPipeline` at `pipeline_marigold_normals.py:99`, `__call__` at `:334`, latent prep at `:595`, decode at `:632`, ensemble at `:661` |
| Intrinsics pipeline | `MarigoldIntrinsicsPipeline` at `pipeline_marigold_intrinsics.py:120`, `__call__` at `:361`, `n_targets` at `:207`, latent prep at `:627`, decode at `:665`, ensemble at `:679` |
| Image processor | `MarigoldImageProcessor` at `marigold_image_processing.py:37`, preprocessing at `:216`, resize/pad/unpad at `:92`, `:108`, `:129`, `:145` |
| Denoiser | `UNet2DConditionModel` at `unet_2d_condition.py:75`, construction at `:177`, forward at `:981` |
| VAE | `AutoencoderKL` at `autoencoder_kl.py:36`, encode at `:171`, decode at `:214` |
| Attention | `Attention` at `attention_processor.py:52`, default `AttnProcessor2_0` at `:2696`, fused processor at `:3668` |
| Schedulers | `DDIMScheduler` at `scheduling_ddim.py:139`; `LCMScheduler` at `scheduling_lcm.py:142` |

## 2. Pipeline and component graph

Marigold is an image-conditioned latent diffusion family for dense prediction,
not a text-to-image generator. The source uses Stable-Diffusion-style
`UNet2DConditionModel`, `AutoencoderKL`, CLIP empty-text conditioning, and either
DDIM or LCM scheduling. The user image is VAE-encoded once, replicated for
ensembles, concatenated with the predicted target latent, and held fixed while
the target latent is denoised.

```text
input image(s) [0,1]
  -> MarigoldImageProcessor: canonicalize to NCHW, normalize to [-1,1],
     optional max-edge resize, replicate-pad to VAE multiple
  -> CLIP tokenizer + CLIPTextModel on empty prompt, cached as [1,2,1024]
  -> AutoencoderKL encode image to scaled image_latent [N*E,4,h,w]
  -> initialize pred_latent noise or accept supplied latents
  -> denoising loop by batch:
       cat image_latent and pred_latent over channel
       -> UNet2DConditionModel(..., empty_text_embedding)
       -> scheduler.step(noise, t, pred_latent)
  -> AutoencoderKL decode predicted latent(s)
  -> target-specific postprocess, unpad, optional ensemble, optional resize
  -> `pt` NCHW or `np` NHWC output
```

Required components for the first Marigold slice:

| Component | Required? | Notes |
| --- | --- | --- |
| `MarigoldImageProcessor` | Yes | NCHW canonicalization, range check, `[-1,1]` normalization, resize, replicate padding, unpad, output transpose. |
| `CLIPTokenizer` + `CLIPTextModel` | Active, but cacheable | Only the empty prompt is encoded. The cached tensor is repeated per denoiser batch. |
| `AutoencoderKL` | Yes | Encode image with posterior `mode()`, scale latent by VAE scaling factor; decode target latents. |
| `UNet2DConditionModel` | Yes | SD2-shaped conditional UNet with widened input/output channels. |
| `DDIMScheduler` | Yes for v1/v1.1 default parity | v-prediction, scaled-linear beta schedule, leading or trailing spacing by checkpoint. |
| `LCMScheduler` | Yes for LCM variants | v-prediction, 1-step defaults, boundary-condition scalings. |

Independently cacheable stages:

- Empty text embedding from tokenizer/text encoder.
- Preprocessed/padded image and VAE image latent for repeated target runs.
- Scheduler timesteps/tables for a fixed scheduler config and step count.
- Supplied output latents from `output_latent=True` can seed later calls.
- VAE decode can be tested separately per target postprocess.

Separate candidate reports and variant inventory:

| Candidate | Classes/files | Status and pipeline delta |
| --- | --- | --- |
| `marigold_depth` | `MarigoldDepthPipeline`, `pipeline_marigold_depth.py` | Base depth/disparity surface. UNet input `[image_latent(4), depth_latent(4)]`, output 4-channel latent, decoded RGB averaged to one depth channel and normalized to `[0,1]`. |
| `marigold_normals` | `MarigoldNormalsPipeline`, `pipeline_marigold_normals.py` | Same denoising contract as depth, but decoded RGB is clipped/renormalized to unit 3D normals; optional positive-z remap through `use_full_z_range=False`. |
| `marigold_intrinsics` | `MarigoldIntrinsicsPipeline`, `pipeline_marigold_intrinsics.py` | Multi-target IID. UNet input is `(1+T)*4` channels and output is `T*4` channels; decoded target latents become `T` RGB maps. |
| `marigold_depth_lcm` | Same depth pipeline + `LCMScheduler` | Same model topology with LCM scheduler and `default_denoising_steps=1`. Needs separate scheduler parity from DDIM. |
| `marigold_normals_lcm` | Same normals pipeline + `LCMScheduler` | Same as above for normals. |
| `marigold_depth_hr` | Hub model index says `MarigoldDepthHRPipeline` | Blocked in this checkout: no source class or exported file exists. Do not infer HR-specific resizing/tiling behavior from config alone. |
| LoRA/textual inversion/runtime adapters | Generic loaders exist in Diffusers, but Marigold pipeline classes do not inherit loader mixins | Not a source-supported Marigold pipeline surface here. Treat any LoRA repo as external UNet/text-encoder mutation, separate from this target. |
| IP-Adapter | No Marigold IP-Adapter pipeline or image encoder component | Not supported by the Marigold target in this checkout. |
| ControlNet/T2I-Adapter/GLIGEN | Generic SD components exist elsewhere | Not wired into Marigold source; separate generic SD candidates only. |
| img2img/inpaint/depth2img/upscale | Marigold itself is image-to-dense-prediction | No Marigold variants with these SD pipeline contracts in this checkout. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Target | Scheduler | Steps | Proc. res | UNet C in/out | Targets | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `prs-eth/marigold-depth-v1-1` | `MarigoldDepthPipeline` | depth | DDIM v-pred trailing | 4 | 768 | 8 / 4 | 1 | Current depth default. |
| `prs-eth/marigold-depth-v1-0` | `MarigoldDepthPipeline` | depth | DDIM v-pred leading default | 10 | 768 | 8 / 4 | 1 | Older schedule length/spacing. |
| `prs-eth/marigold-depth-lcm-v1-0` | `MarigoldDepthPipeline` | depth | LCM v-pred leading | 1 | 768 | 8 / 4 | 1 | Short schedule, `original_inference_steps=50`. |
| `prs-eth/marigold-normals-v1-1` | `MarigoldNormalsPipeline` | normals | DDIM v-pred trailing | 4 | 768 | 8 / 4 | 1 | `use_full_z_range=True`. |
| `prs-eth/marigold-normals-v0-1` | `MarigoldNormalsPipeline` | normals | DDIM v-pred leading | 10 | 768 | 8 / 4 | 1 | Older normals schedule. |
| `prs-eth/marigold-normals-lcm-v0-1` | `MarigoldNormalsPipeline` | normals | LCM v-pred leading | 1 | 768 | 8 / 4 | 1 | Short schedule. |
| `prs-eth/marigold-iid-appearance-v1-1` | `MarigoldIntrinsicsPipeline` | intrinsics | DDIM v-pred trailing | 4 | 768 | 12 / 8 | 2 | Targets: albedo and material stack. |
| `prs-eth/marigold-iid-lighting-v1-1` | `MarigoldIntrinsicsPipeline` | intrinsics | DDIM v-pred trailing | 4 | 768 | 16 / 12 | 3 | Targets: albedo, shading, residual. |
| `prs-eth/marigold-depth-hr-v1-1` | `MarigoldDepthHRPipeline` | depth | DDIM v-pred trailing | 10 | 768 | 8 / 4 | 1 | Blocked: pipeline class absent from checkout. |

Shared UNet dimensions:

| Field | Value |
| --- | --- |
| `sample_size` | 96 latent units for 768 processing resolution and VAE scale 8 |
| `down_block_types` | `CrossAttnDownBlock2D` x3, then `DownBlock2D` |
| `up_block_types` | `UpBlock2D`, then `CrossAttnUpBlock2D` x3 |
| `block_out_channels` | `[320, 640, 1280, 1280]` |
| `layers_per_block` | 2 |
| `cross_attention_dim` | 1024 |
| `attention_head_dim` | `[5, 10, 20, 20]`, implying 64 heads at each attention width (`320/5`, `640/10`, `1280/20`) |
| `norm_num_groups` | 32 |
| `use_linear_projection` | true |
| inactive config branches | no class embedding, no addition embedding, `only_cross_attention=false`, positional timestep embedding, default ResNet time shift |

VAE dimensions and effective defaults:

| Field | Value / effective default |
| --- | --- |
| `sample_size` | 768 |
| `in_channels` / `out_channels` | 3 / 3 |
| `latent_channels` | 4 |
| `block_out_channels` | `[128, 256, 512, 512]`, so `vae_scale_factor=8` |
| `layers_per_block` | 2 |
| `scaling_factor` | Some depth configs omit it; `AutoencoderKL` default is `0.18215`. Normals/IID configs include `0.18215`. |
| `force_upcast` | Some configs omit it; `AutoencoderKL` default is `True`. |
| `use_quant_conv` / `use_post_quant_conv` | Omitted in sampled configs; source defaults are `True`. |
| `mid_block_add_attention` | Omitted; source default is `True`. |

External CLIP text encoder:

| Field | Value |
| --- | --- |
| Architecture | `CLIPTextModel` |
| Hidden size | 1024 |
| Layers / heads | 23 / 16 |
| Max positions | 77 |
| Vocab size | 49408 |
| Runtime prompt | Empty string only; tokenizer uses no padding, so observed hidden states are `[1,2,1024]`. |

Scheduler support and first Dinoml slice:

| Scheduler | Source/config use | First-slice recommendation |
| --- | --- | --- |
| DDIM | Main v1/v1.1 depth, normals, IID configs; `prediction_type=v_prediction`, `clip_sample=false`, `set_alpha_to_one=false`, `steps_offset=1`, leading or trailing spacing | First parity slice for Marigold because it covers current default v1.1 and IID. Must include v-pred conversion and timestep-spacing differences. |
| LCM | LCM depth/normals configs; `prediction_type=v_prediction`, `original_inference_steps=50`, one-step defaults | Second slice. It changes step math through LCM boundary-condition scaling, even though the UNet/VAE topology is the same. |

## 3a. Family variation traps

- Marigold is image-conditioned latent denoising, not CFG text-to-image. There
  is no positive/negative prompt, no CFG batch concat, and no guidance scale.
- The text encoder is still active: the empty prompt embedding is a real UNet
  cross-attention context. It is tiny and cacheable, but not absent.
- Configs share SD2-style CLIP width 1024 and `cross_attention_dim=1024`, not
  SD1 width 768.
- Input channels depend on target count: depth/normals use 8 channels, IID
  appearance uses 12, IID lighting uses 16. Output latent channels similarly
  grow from 4 to 8 or 12.
- Source layout is NCHW everywhere in preprocessing, VAE, UNet, scheduler, and
  postprocess. NHWC/channel-last can be a guarded optimization inside conv
  islands only.
- Axis-sensitive ops include channel concat `dim=1`, depth mean over RGB
  `dim=1`, normals vector norm over `dim=1`, channel chunk/reshape for IID
  target latents, pad/unpad over last two spatial dims, and image processor
  `permute(0,2,3,1)` for NumPy output.
- Ensembling can be simple tensor reduction for normals/IID but depth affine
  alignment uses SciPy BFGS when scale/shift invariant. That should stay host
  or be a separate postprocess candidate.
- `processing_resolution=0` means native max edge, while positive resolution
  rescales the long edge and then pads to a VAE multiple. Latent shape is based
  on the padded processed size.
- `ensemble_size=2` is only warned against, not rejected.
- HR depth is a config-level variant with no source class in this checkout.

## 4. Runtime tensor contract

For default 768 processing on a square image, `PH=PW=768`, `PPH=PPW=768`, and
latent shape is `[B,4,96,96]`. For arbitrary input, preprocessing computes
`PH = H * processing_resolution // max(H,W)` and `PW = W * processing_resolution // max(H,W)`, then pads to `PPH/PPW`
divisible by 8.

| Boundary | Tensor | Source layout | Candidate optimized layout | Shape |
| --- | --- | --- | --- | --- |
| input canonical image | `image` | NCHW | CPU/data pipeline | `[N,3,H,W]`, grayscale repeated to 3 channels |
| normalized processed image | `image` | NCHW | NHWC only after guarded boundary | `[N,3,PPH,PPW]`, values `[-1,1]` |
| empty text embedding | `empty_text_embedding` | `[B,S,C]` | same | cached `[1,2,1024]`, repeated to effective batch |
| VAE image latent | `image_latent` | NCHW | NHWC guarded candidate inside VAE/UNet | `[N*E,4,h,w]`, scaled by VAE factor |
| target latent depth/normals | `pred_latent` | NCHW | same as denoiser | `[N*E,4,h,w]` |
| target latent IID | `pred_latent` | NCHW packed channels | channel-last requires target/channel rewrite | `[N*E,T*4,h,w]` |
| UNet input depth/normals | `batch_latent` | NCHW | guarded NHWC candidate | `[B,8,h,w] = cat([image_latent,pred_latent], dim=1)` |
| UNet input IID | `batch_latent` | NCHW | guarded NHWC candidate | `[B,(1+T)*4,h,w]` |
| UNet output | `noise` | NCHW | same as pred latent | `[B,4,h,w]` or `[B,T*4,h,w]` |
| scheduler output | `prev_sample` | NCHW | same as pred latent | same shape as pred latent |
| decoded depth | `prediction` | NCHW | NHWC only after output conversion | `[N*E,1,PPH,PPW]` before unpad |
| decoded normals | `prediction` | NCHW | same | `[N*E,3,PPH,PPW]` before unpad |
| decoded IID | `prediction` | NCHW | same | `[N*E*T,3,PPH,PPW]` before unpad |
| NumPy outputs | `prediction` / `uncertainty` | NHWC | output format | depth `[N,H,W,1]`, normals `[N,H,W,3]`, IID `[N*T,H,W,3]` |

Latent preparation:

- `vae.encode(image).latent_dist.mode()` is used when the encoder output has a
  posterior distribution. No VAE posterior sampling is active for image latents.
- `image_latent *= vae.config.scaling_factor`.
- Initial predicted target latents are `randn_tensor` unless supplied by the
  caller. Supplied latents must match the exact target latent shape.
- `generator` and `latents` are mutually exclusive.

Decode/postprocess contracts:

| Target | Decode transform |
| --- | --- |
| Depth | `vae.decode(pred_latent / scaling_factor)` -> mean RGB over channel -> clip `[-1,1]` -> `(x+1)/2`; ensemble may affine-align and renormalize. |
| Normals | VAE decode -> clip `[-1,1]`; if `use_full_z_range=False`, remap z by `z=0.5*z+0.5`; normalize vector over channel with clamp eps. |
| IID | Reshape latent to `[N*E*T,4,h,w]`; VAE decode each target latent -> clip `[-1,1]` -> `(x+1)/2`; target semantics come from `target_properties`. |

CPU/data-pipeline versus GPU/runtime:

- CPU/data-pipeline: PIL/NumPy loading, dtype checks, unsigned integer scaling,
  optional colormap/export visualization, SciPy depth alignment solver.
- GPU/runtime candidates: image tensor normalization/resize/pad, VAE encode,
  latent concat, UNet, scheduler step, VAE decode, target postprocess, unpad,
  resize output.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCHW input handling and NHWC output conversion for NumPy.
- `repeat`, `repeat_interleave`, `cat(dim=0)`, `cat(dim=1)`, batch slicing.
- `reshape` for IID `[N*E,T*4,h,w] -> [N*E*T,4,h,w]` and ensemble grouping.
- `permute(0,2,3,1)` for output NumPy.
- `F.interpolate` resize with bilinear/bicubic/area/nearest modes and optional antialias on input resize.
- Replicate `F.pad(image, (0,pw,0,ph), mode="replicate")`.
- Unpad spatial slice `image[:, :, :uh, :uw]`.
- Scalar/channel broadcasting for VAE scaling and scheduler coefficients.
- Reductions: mean over RGB for depth, norm/sum/argmax/gather for normals,
  median/mean/std/absolute deviation for ensembling.

### Convolution/downsample/upsample ops

- Marigold UNet `conv_in`:
  - depth/normals `Conv2d(8 -> 320, 3x3, padding=1)`.
  - IID appearance `Conv2d(12 -> 320, 3x3, padding=1)`.
  - IID lighting `Conv2d(16 -> 320, 3x3, padding=1)`.
- Marigold UNet `conv_out`:
  - depth/normals `Conv2d(320 -> 4, 3x3, padding=1)`.
  - IID appearance `Conv2d(320 -> 8, 3x3, padding=1)`.
  - IID lighting `Conv2d(320 -> 12, 3x3, padding=1)`.
- SD-style `ResnetBlock2D`, `Downsample2D`, `Upsample2D`.
- AutoencoderKL encoder/decoder convs, ResNet blocks, down/up sample, and
  `quant_conv` / `post_quant_conv` 1x1 convs by source default.

### GEMM/linear ops

- CLIP empty prompt embedding path: token embeddings, positional embeddings,
  Transformer layers, MLPs. First Dinoml can consume cached `[1,2,1024]`.
- UNet timestep embedding MLP.
- Cross-attention Q/K/V/out projections.
- Transformer feed-forward GEGLU/GELU MLPs.
- VAE mid-block attention projections when `mid_block_add_attention=True`.

### Attention primitives

- UNet latent self-attention and cross-attention in `CrossAttnDownBlock2D`,
  `UNetMidBlock2DCrossAttn`, and `CrossAttnUpBlock2D`.
- Cross-attention context is empty CLIP hidden states of sequence length 2 and
  hidden size 1024.
- VAE bottleneck self-attention from `AutoencoderKL` source default.
- No causal mask, no KV cache, no CFG, no added K/V for base Marigold.

### Normalization and adaptive conditioning

- GroupNorm with 32 groups over source channel axis in UNet/VAE.
- LayerNorm inside `BasicTransformerBlock`.
- SiLU activations.
- Timestep embedding applied to ResNet blocks through default time-bias path.
- Normals postprocess vector normalization over channel axis.

### Scheduler and guidance arithmetic

- DDIM `set_timesteps` with leading/trailing spacing and v-prediction step.
- LCM `set_timesteps` with `original_inference_steps=50`, v-prediction, and
  boundary-condition scalings.
- Random target latent initialization.
- No classifier-free guidance arithmetic.

### VAE/postprocessing ops

- VAE encode with posterior `mode()`.
- VAE decode after dividing target latents by scaling factor.
- Depth RGB mean, clip, rescale, optional affine-invariant ensemble.
- Normals clip, optional z remap, normalize, cosine similarity, argmax/gather.
- IID target reshape, clip, rescale, target property visualization transforms.

## 6. Denoiser/model breakdown

The Marigold denoiser is `UNet2DConditionModel` with SD2-like cross-attention
dimensions and widened channel boundaries:

```text
sample [B,Cin,h,w]
  -> conv_in Cin -> 320
  -> timestep projection + TimestepEmbedding
  -> down:
       CrossAttnDownBlock2D(320) x2 resnets + attentions + downsample
       CrossAttnDownBlock2D(640) x2 resnets + attentions + downsample
       CrossAttnDownBlock2D(1280) x2 resnets + attentions + downsample
       DownBlock2D(1280) x2 resnets
  -> mid:
       UNetMidBlock2DCrossAttn(1280)
  -> up:
       UpBlock2D(1280)
       CrossAttnUpBlock2D(1280)
       CrossAttnUpBlock2D(640)
       CrossAttnUpBlock2D(320)
  -> GroupNorm/SiLU/conv_out to Cout
```

Active branch notes:

- `only_cross_attention=false`, so configured attention blocks can include
  self-attention plus cross-attention as in the SD2 UNet family.
- No class labels, addition embeddings, ControlNet residuals, T2I adapter
  residuals, GLIGEN, IP-Adapter, or guidance embeddings are active from the
  sampled configs.
- `use_linear_projection=true` affects transformer projections inside attention
  blocks and should be preserved during config parsing.

Target channel formulas:

```text
depth/normals:  T=1, Cin=(1+T)*4=8,  Cout=T*4=4
IID appearance: T=2, Cin=12,        Cout=8
IID lighting:   T=3, Cin=16,        Cout=12
```

Autoencoder path:

```text
encode image:
  image [N,3,PPH,PPW]
  -> Encoder conv/resnet/down/mid/norm/conv_out
  -> quant_conv if present
  -> DiagonalGaussianDistribution
  -> mode() [N,4,h,w]
  -> scale by 0.18215

decode target:
  pred_latent / 0.18215
  -> post_quant_conv if present
  -> Decoder conv/mid/up/resnet/norm/conv_out
  -> RGB-like sample [B,3,PPH,PPW]
```

## 7. Attention requirements

Parity implementation path:

- `attention_processor.py` is the primary attention backend for both UNet and
  VAE. `Attention` defaults to `AttnProcessor2_0` when PyTorch SDPA is available
  and falls back to eager `AttnProcessor` otherwise.
- UNet attention is dense, noncausal, and mask-free for this target. Query
  sequence length is `h*w` at each latent resolution; context sequence length is
  2 for the empty CLIP prompt.
- Per-resolution channel/head shape from configs:
  - 320 channels, head dim 5, 64 heads.
  - 640 channels, head dim 10, 64 heads.
  - 1280 channels, head dim 20, 64 heads.
- VAE mid attention is mask-free spatial self-attention, usually at the codec
  bottleneck.
- Fused QKV/KV projections are source-supported through
  `fuse_qkv_projections()` on UNet/VAE, installing `FusedAttnProcessor2_0`, but
  not required for first parity.

Flash feasibility:

- A Dinoml flash-style provider is plausible under strict preconditions:
  dense no-mask attention, dropout 0, supported dtype, exact scaling, and
  source-equivalent Q/K/V head layout.
- The tiny context length 2 for cross-attention means projection and layout
  overhead may dominate; flash is more valuable for latent self-attention and
  VAE bottleneck attention than for empty-prompt K/V attention.
- Added-KV/IP-Adapter, joint attention, RoPE, varlen packing, and causal masks
  are not active requirements for Marigold.

## 8. Scheduler and denoising-loop contract

Marigold loops are host-visible and batch over `num_images * ensemble_size`:

```text
for batch:
  scheduler.set_timesteps(num_inference_steps, device)
  for t in scheduler.timesteps:
    batch_latent = cat([batch_image_latent, batch_pred_latent], dim=1)
    noise = unet(batch_latent, t, encoder_hidden_states=empty_text)
    batch_pred_latent = scheduler.step(noise, t, batch_pred_latent, generator).prev_sample
```

DDIM default contract:

- `prediction_type=v_prediction`.
- `beta_schedule=scaled_linear`, train steps 1000.
- `clip_sample=false`, `set_alpha_to_one=false`, `steps_offset=1`.
- v1.1 and IID use `timestep_spacing=trailing`; older depth/normals v1.0 omit
  the field and therefore use DDIM source default `leading`.
- `scale_model_input` is identity for DDIM.

LCM variant contract:

- Same UNet/VAE tensor contract, but scheduler is `LCMScheduler`.
- `default_denoising_steps=1`, `original_inference_steps=50`,
  `prediction_type=v_prediction`, `timestep_spacing=leading`.
- `step()` computes `predicted_original_sample`, applies LCM boundary scalings
  `c_skip` and `c_out`, and returns the denoised/previous sample depending on
  step index. This needs a separate parity slice from DDIM.

Initial staging should keep `set_timesteps`, step iteration, and ensemble loops
in host code while compiling one denoiser step plus scheduler pointwise update
as explicit graph/runtime work. There is no CFG batching to model.

## 9. Position, timestep, and custom math

Required custom math:

- UNet sinusoidal/positional timestep embedding from `UNet2DConditionModel`.
- DDIM v-prediction conversion:
  `pred_original_sample = sqrt(alpha) * sample - sqrt(beta) * model_output`,
  then previous sample update using alpha products.
- LCM v-prediction conversion plus boundary scalings:
  `c_skip = sigma_data^2 / (scaled_timestep^2 + sigma_data^2)` and
  `c_out = scaled_timestep / sqrt(scaled_timestep^2 + sigma_data^2)`.
- Depth postprocess:
  `mean(dim=1)`, clip to `[-1,1]`, map to `[0,1]`.
- Normals postprocess:
  channel vector norm, clamp epsilon, divide; optional z remap.
- Depth ensemble affine/scale alignment:
  initializes per-member scale/shift from min/max, optimizes pairwise RMS
  disagreement with SciPy, then normalizes final map to `[0,1]`.
- IID ensemble:
  median or mean across ensemble dimension; optional median absolute deviation
  or std uncertainty.

Precompute:

- Empty prompt hidden state.
- Scheduler timesteps and scalar tables per config/step count.
- VAE image latent for a fixed preprocessed image.

Dynamic:

- Input H/W and processing resolution.
- Ensemble size and per-ensemble random target latent.
- Batch slicing and output resize.

## 10. Preprocessing and input packing

Image processor input rules:

- Accepts PIL, NumPy, torch tensor, or list. 2D images become one-channel batch
  images; one-channel inputs are repeated to 3 channels.
- Unsigned integer arrays/tensors are scaled by dtype max; float inputs are
  expected in `[0,1]`.
- Signed integer, complex, bool, and non-1/3-channel inputs are rejected.
- Source canonical layout after loading is NCHW.

Processing:

```text
load/canonicalize -> optional range check -> image * 2 - 1
  -> if processing_resolution > 0: resize long edge to processing_resolution
  -> replicate-pad bottom/right to multiple of vae_scale_factor
```

Packing into the denoiser:

- Image latent and target latent are concatenated over channel dimension.
- IID packs multiple target latent tensors by channel, not by batch:
  `[B,T*4,h,w]`.
- There is no pipeline-level patchify/unpatchify. The UNet and VAE operate on
  spatial latent maps.

Postprocessing:

- Remove padding first.
- Optional ensemble.
- Optional resize output back to original input resolution.
- `output_type="np"` transposes to NHWC; `output_type="pt"` preserves NCHW.
- Visualization/export helpers are not part of the first runtime contract, but
  they document target ranges and axis expectations.

## 11. Graph rewrite / lowering opportunities

### Rewrite: target channel concat into widened `conv_in`

Source pattern:

```text
batch_latent = cat([image_latent, pred_latent], dim=1)
conv_in(batch_latent)
```

Replacement:

```text
conv(image_latent, W[:, image_channels]) + conv(pred_latent, W[:, target_channels]) + bias
```

Preconditions:

- Source NCHW channel order is preserved or explicitly rewritten to NHWC.
- Image and target latents have equal B/H/W.
- Target count is known from config: 1, 2, or 3.
- No consumer observes the concatenated tensor.

Shape equations:

- `Cin=(1+T)*4`, `Cout=320`, kernel 3x3.

Failure cases:

- Supplied latents with wrong packed target layout.
- Future HR/tiling pipeline that chunks spatially around the concat.
- Layout pass fails to rewrite concat dim 1 to last channel.

Parity test sketch:

- Compare explicit concat+conv against split-conv-sum for C=8, 12, 16 at
  several latent sizes.

### Rewrite: guarded NCHW -> NHWC conv/resnet islands

Source pattern:

```text
Conv2d / GroupNorm / SiLU / residual / downsample / upsample blocks
```

Replacement:

```text
NCHW boundary -> NHWC internal island -> NCHW boundary
```

Preconditions:

- All ops in the island have rewritten axes: GroupNorm channel axis, concat
  channel axis, resize spatial axes, pad/unpad spatial axes.
- Attention flatten/reshape is either outside the island or has an explicit
  layout-aware lowering.
- Scheduler scalar broadcasts are layout-polymorphic or bounded by layout
  transitions.

Weight transform:

```python
w_hwio = w_oihw.permute(2, 3, 1, 0)
```

Failure cases:

- IID target-channel packing and normals `norm(dim=1)` are not rewritten.
- VAE posterior `chunk(dim=1)` from `AutoencoderKL` encode is accidentally
  included without channel-axis rewrite.

Parity test sketch:

- VAE encode/decode block parity, UNet down/up block parity, and full one-step
  denoiser parity for depth/normals at 768 and a non-square padded size.

### Rewrite: cached empty text embedding boundary

Source pattern:

```text
tokenizer("") -> CLIPTextModel -> repeat(batch_size,1,1)
```

Replacement:

```text
load/capture constant empty_text_embedding [1,2,1024] -> repeat/slice per batch
```

Preconditions:

- Tokenizer/text encoder weights and tokenizer config match the checkpoint.
- No runtime adapter mutates CLIP weights or token embeddings after caching.
- Prompt is exactly the empty string with `padding="do_not_pad"`.

Failure cases:

- LoRA/textual-inversion-like mutation of text encoder/tokenizer.
- Tokenizer configs differ in pad/bos/eos behavior and change sequence length.

Parity test sketch:

- Compare cached tensor against Diffusers tokenizer/text encoder output for each
  representative repo.

### Rewrite: postprocess fusion kernels

Depth:

- Source: VAE RGB output -> channel mean -> clip -> affine map.
- Replacement: fused pointwise/reduction over channel.
- Guard: source layout and dtype; channel axis rewrite for NHWC.

Normals:

- Source: clip -> optional z remap -> L2 normalize over channel.
- Replacement: fused vector-normalize kernel.
- Guard: axis rewrite and eps clamp.

IID:

- Source: reshape targets -> decode each target -> clip -> affine map.
- Replacement: keep reshape explicit first; later batch target decodes.
- Guard: target count and latent channel stride.

## 12. Kernel fusion candidates

Highest priority:

- Widened Marigold `conv_in` concat fusion for C=8/12/16.
- UNet and VAE Conv2d + GroupNorm + SiLU ResNet block fusion.
- DDIM v-pred scheduler step as fused pointwise over target latent.
- VAE encode/decode scale/unscale plus target postprocess kernels.
- Empty text embedding as an artifact-visible constant/cache boundary.

Medium priority:

- SDPA/flash-style attention provider for mask-free latent self-attention and
  VAE mid attention; cross-attention to sequence length 2 may not pay back until
  projection/layout overhead is optimized.
- LCM step fusion for one-step variants.
- Output resize/unpad and normals normalization fusion.
- IID multi-target decode batching and target-channel packing helpers.

Lower priority:

- Depth affine-invariant ensemble solver acceleration. It uses SciPy BFGS and
  is better kept as host postprocess until base parity is stable.
- Visualization/export colormap kernels.
- HR depth variant until a source class is present.

## 13. Runtime staging plan

Stage 1: Parse Marigold configs and admit the shared SD2-shaped UNet, VAE, CLIP
empty embedding, and DDIM v-pred scheduler. Accept cached empty text embedding
as an input/constant first.

Stage 2: Implement depth v1.1 one-step/loop parity with supplied image tensors,
VAE encode mode, random/supplied target latents, UNet C=8, DDIM trailing
v-pred, VAE decode, and depth postprocess.

Stage 3: Add normals using the same denoiser contract plus normals normalize
postprocess and ensemble reductions.

Stage 4: Add IID appearance and lighting by generalizing target count `T`,
UNet C=12/16 input, C=8/12 output, latent reshape, and multi-target decode.

Stage 5: Add LCM depth/normals scheduler parity.

Stage 6: Add guarded NHWC conv islands, concat-first-conv fusion, and attention
provider optimization.

Stage 7: Decide whether CLIP empty text encoder is compiled, precomputed at
artifact build/load time, or a required runtime constant.

## 14. Parity and validation plan

- Config parsing tests for all official repos listed above, including omitted
  VAE defaults and timestep spacing differences.
- `MarigoldImageProcessor.preprocess` parity for PIL, NumPy, torch, grayscale,
  uint8, non-square resize, range errors, and replicate padding.
- VAE image encode parity using posterior `mode()` and scale factor.
- Latent initialization and supplied-latents shape checks for depth/normals and
  IID target counts.
- One UNet forward parity for C=8, C=12, and C=16 inputs at `[96,96]` and one
  non-square padded latent size.
- DDIM v-pred `set_timesteps` and `step` parity for leading and trailing.
- LCM one-step parity for `original_inference_steps=50`.
- Decode/postprocess parity:
  - depth RGB mean/clip/rescale;
  - normals clip/z-remap/normalize;
  - IID reshape/decode/clip/rescale.
- Ensemble parity for simple mean/median modes; depth affine alignment can be a
  host-side parity test against SciPy output.
- End-to-end deterministic smoke with fixed image tensor and fixed target
  latents, scheduler in host control first.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
`rtol=2e-2, atol=2e-2` and tighten per kernel.

## 15. Performance probes

- VAE encode/decode throughput at 768 square and representative non-square
  padded sizes.
- One denoiser step for C=8, C=12, and C=16 inputs, with and without split
  concat-first-conv fusion.
- DDIM 4-step vs 10-step and LCM 1-step scheduler overhead.
- NCHW faithful path versus guarded NHWC conv islands for UNet and VAE.
- Attention backend comparison for latent self-attention and cross-attention
  context length 2.
- Batch size and ensemble size scaling: one large batch vs source-style batched
  loop over `num_images * ensemble_size`.
- Postprocess cost for depth/normals/IID and resize-to-input-resolution.

## 16. Scope boundary and separate candidates

Separate candidates, not ignored:

- `marigold_depth`: first base target.
- `marigold_normals`: same denoiser, normals-specific decode/postprocess.
- `marigold_intrinsics`: multi-target channel-packing and decode.
- `marigold_lcm`: LCM scheduler variants for depth/normals.
- `marigold_depth_hr`: blocked until source class is available in the checkout.
- `marigold_postprocess_ensemble`: depth affine alignment, normals closest
  selection, IID uncertainty reductions as optional optimized postprocess.

Not source-supported in this Marigold target:

- Marigold-specific LoRA/textual inversion/runtime adapter mixins.
- Marigold IP-Adapter, ControlNet, T2I-Adapter, GLIGEN.
- Marigold img2img/inpaint/depth2img/upscale variants.

Ignored/out of scope:

- XLA/NPU/MPS/Flax/ONNX.
- Safety/NSFW filtering.
- Training/loss/dropout/gradient checkpointing.
- Callback mutation and interactive interrupt.
- Multi-GPU/context parallel.

## 17. Final implementation checklist

- [ ] Parse official Marigold depth/normals/IID configs from `H:/configs/prs-eth`.
- [ ] Reconcile omitted VAE defaults: scaling factor, force upcast, quant convs,
      post-quant conv, and mid-block attention.
- [ ] Materialize empty CLIP text embedding as a cached runtime boundary or
      compile the tiny empty-prompt CLIP path.
- [ ] Implement `MarigoldImageProcessor` preprocessing parity.
- [ ] Implement AutoencoderKL encode with posterior `mode()` and scaling.
- [ ] Implement UNet C=8 depth/normals denoiser step.
- [ ] Implement DDIM v-pred leading/trailing scheduler parity.
- [ ] Implement depth decode/postprocess.
- [ ] Add normals postprocess and ensemble reductions.
- [ ] Generalize target count for IID C=12/16 input and C=8/12 output.
- [ ] Add LCM scheduler variant parity.
- [ ] Add concat-first-conv fusion tests for C=8, 12, 16.
- [ ] Add guarded NHWC conv-island tests for VAE and UNet.
- [ ] Keep `marigold-depth-hr-v1-1` blocked until `MarigoldDepthHRPipeline`
      source exists or a human selects a different checkout/revision.
