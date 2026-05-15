# Diffusers AutoencoderKL Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Family target: AutoencoderKL 2D image latent codec.
  Representative configs:
    stable-diffusion-v1-5/stable-diffusion-v1-5 vae
    stabilityai/sd-vae-ft-mse
    stabilityai/sdxl-vae
    madebyollin/sdxl-vae-fp16-fix

Config sources:
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/vae/config.json
  https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json
  https://huggingface.co/stabilityai/sdxl-vae/raw/main/config.json
  https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/raw/main/config.json
  stabilityai/stable-diffusion-3-medium-diffusers vae/config.json via authenticated hf CLI
  black-forest-labs/FLUX.1-schnell vae/config.json via authenticated hf CLI
  Saved authenticated component configs:
    H:/configs/stabilityai/stable-diffusion-3-medium-diffusers/vae/config.json
    H:/configs/black-forest-labs/FLUX.1-schnell/vae/config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
  diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
  diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
  diffusers/src/diffusers/pipelines/flux/pipeline_flux.py

Model files inspected:
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/image_processor.py indirectly through pipeline VAE usage.

External component configs inspected:
  None. AutoencoderKL is a Diffusers-native component.

Any missing files or assumptions:
  This report is inference-only and covers 2D AutoencoderKL encode/decode. It
  ignores training losses, dropout behavior, gradient checkpointing,
  multi-GPU/context parallel paths, callbacks, safety/NSFW, and XLA/NPU/MPS/Flax/
  ONNX variants. AutoencoderTiny, AsymmetricAutoencoderKL, ConsistencyDecoderVAE,
  and video/audio autoencoders are separate candidate reports.
```

## 2. Pipeline and component graph

`AutoencoderKL` is the latent image codec used by many image pipelines. It is
not the denoiser. Pipelines call it at component boundaries:

```text
text-to-image:
  denoised latents [B,C,H/scale,W/scale]
  -> unscale/unshift latents
  -> AutoencoderKL.decode
  -> VaeImageProcessor postprocess

img2img/inpaint/control:
  preprocessed image [B,3,H,W]
  -> AutoencoderKL.encode
  -> posterior sample or mode [B,C,H/scale,W/scale]
  -> scale/shift latents
  -> denoising loop
```

Internal codec graph:

```text
encode:
  image NCHW
  -> Encoder: conv_in -> down blocks -> mid block -> norm/SiLU/conv_out
  -> optional quant_conv 1x1
  -> DiagonalGaussianDistribution(mean, logvar)
  -> mode() or sample()

decode:
  latent NCHW
  -> optional post_quant_conv 1x1
  -> Decoder: conv_in -> mid block -> up blocks -> norm/SiLU/conv_out
  -> decoded image NCHW
```

Required components for this target:

- `AutoencoderKL`, `Encoder`, `Decoder`, `DiagonalGaussianDistribution`.
- `DownEncoderBlock2D`, `UpDecoderBlock2D`, `UNetMidBlock2D`.
- `ResnetBlock2D`, `Downsample2D`, `Upsample2D`, `Attention`.
- `quant_conv` and `post_quant_conv` when config enables them.

Separate candidate reports:

| Candidate | Classes/files | Why separate |
| --- | --- | --- |
| `autoencoder_tiny` | `AutoencoderTiny`, `EncoderTiny`, `DecoderTiny` | TAESD-style compact conv codec with different scaling conventions and no KL posterior. |
| `asymmetric_autoencoder_kl` | `AsymmetricAutoencoderKL`, `MaskConditionEncoder`, `MaskConditionDecoder` | Mask-conditioned decoder used by inpaint-style codecs; extra mask/image tensor contract. |
| `video_autoencoders` | `autoencoder_kl_cogvideox.py`, `autoencoder_kl_wan.py`, `autoencoder_kl_ltx*.py`, `autoencoder_kl_mochi.py` | Temporal/3D paths, chunked/tiled decode, temporal compression. |
| `audio_autoencoders` | `autoencoder_oobleck.py`, `autoencoder_kl_ltx2_audio.py`, audio DiT codecs | Conv1d/audio latent contracts. |

## 3. Important config dimensions

Source defaults from `AutoencoderKL.__init__`:

| Field | Default | Runtime effect |
| --- | ---: | --- |
| `in_channels` / `out_channels` | 3 / 3 | RGB image boundary. |
| `latent_channels` | 4 | Latent channel count before diffusion denoiser. |
| `block_out_channels` | `(64,)` | Number of down/up stages and scale factor. |
| `layers_per_block` | 1 | ResNet count per block, decoder uses `layers_per_block + 1`. |
| `norm_num_groups` | 32 | GroupNorm axis-sensitive channel grouping. |
| `sample_size` | 32 | Also used as tile sample min size. |
| `scaling_factor` | 0.18215 | Pipeline scales latents before denoiser and unscales before decode. |
| `shift_factor` | `None` | Newer pipelines may unshift/shift latents around VAE. |
| `latents_mean` / `latents_std` | `None` / `None` | SDXL pipeline optionally denormalizes with per-channel stats. |
| `force_upcast` | `True` | Some pipelines run VAE in fp32 to avoid fp16 overflow. |
| `use_quant_conv` / `use_post_quant_conv` | `True` / `True` | 1x1 convs around latent distribution/code space. |
| `mid_block_add_attention` | `True` | Enables mid-block spatial self-attention. |

Representative config sweep:

| Config | sample size | latent channels | blocks | layers | scale | shift | force upcast | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| SD 1.5 VAE | 512 | 4 | 128/256/512/512 | 2 | omitted, effective 0.18215 | omitted | omitted, effective true | Classic SD codec. |
| `stabilityai/sd-vae-ft-mse` | 256 | 4 | 128/256/512/512 | 2 | omitted, effective 0.18215 | omitted | omitted, effective true | Same operator shape; sample size differs. |
| `stabilityai/sdxl-vae` | 1024 | 4 | 128/256/512/512 | 2 | 0.13025 | omitted | omitted, effective true | SDXL pipeline may upcast to fp32. |
| `madebyollin/sdxl-vae-fp16-fix` | 512 | 4 | 128/256/512/512 | 2 | 0.13025 | omitted | false | Same shape, fp16-friendly config. |
| `stabilityai/stable-diffusion-3-medium-diffusers` VAE | 1024 | 16 | 128/256/512/512 | 2 | 1.5305 | 0.0609 | true | No quant/post-quant conv; 16-channel latent contract. |
| `black-forest-labs/FLUX.1-schnell` VAE | 1024 | 16 | 128/256/512/512 | 2 | 0.3611 | 0.1159 | true | No quant/post-quant conv; Flux packs 2x2 latents outside the VAE. |

Spatial scale factor from blocks is `2 ** (len(block_out_channels) - 1)`. With
four blocks, image 512x512 maps to latent 64x64 and image 1024x1024 maps to
latent 128x128. Flux pipelines additionally pack 2x2 latent patches for the
transformer and use `vae_scale_factor * 2` at image processor boundaries; that
packing is pipeline-level, not AutoencoderKL itself.

## 3a. Family variation traps

- Configs may omit `scaling_factor`, `force_upcast`, `use_quant_conv`,
  `use_post_quant_conv`, `shift_factor`, `latents_mean`, and `latents_std`.
  Reconcile omissions against source defaults.
- SD1/SDXL commonly use 4-channel latents with plain scale/unscale. SD3 and
  Flux use 16-channel latents and shift-aware formulas such as
  `(latents / scaling_factor) + shift_factor`.
- SD3 and Flux configs set `use_quant_conv=false` and `use_post_quant_conv=false`.
  Do not require the 1x1 latent convs for those checkpoints.
- SDXL has optional `latents_mean` and `latents_std` denormalization before
  decode. This is channel-axis sensitive.
- `force_upcast=True` is a runtime dtype policy, not an operator. Dinoml should
  validate fp16/bf16/fp32 numerics rather than blindly copying PyTorch upcast
  behavior.
- Source layout is NCHW. NHWC is a strong candidate inside conv islands, but
  GroupNorm, chunk of posterior parameters on dim 1, latent mean/std views,
  tile blending axes, and pipeline scale/shift broadcasting are axis-sensitive.
- `mid_block_add_attention=True` adds spatial self-attention at the bottleneck.
  Configs with `False` remove the attention requirement.
- `use_slicing` splits batch dimension; `use_tiling` splits spatial tiles and
  blends overlaps. These are memory strategies and separate runtime candidates,
  not first parity requirements.
- `forward(sample)` performs encode then decode and can sample from posterior;
  pipelines usually call `encode` or `decode` directly.

## 4. Runtime tensor contract

| Boundary | Tensor | Source layout | Candidate optimized layout | Typical SD shape |
| --- | --- | --- | --- | --- |
| encode input | preprocessed image | NCHW | NHWC guarded candidate | `[B,3,512,512]` |
| encoder output before quant | moments | NCHW | NHWC guarded candidate | `[B,8,64,64]` for latent 4 |
| posterior mean/logvar | chunked moments | NCHW | channel-last requires axis rewrite | two `[B,4,64,64]` tensors |
| posterior sample/mode | latent | NCHW | NHWC guarded candidate | `[B,4,64,64]` |
| pipeline latent scale | `z * scaling_factor` or `(z - shift) * scale` | NCHW | layout-polymorphic if channel stats handled | `[B,4,64,64]` |
| decode input | unscaled/unshifted latent | NCHW | NHWC guarded candidate | `[B,4,64,64]` |
| decoder output | image sample | NCHW | NHWC guarded candidate | `[B,3,512,512]` |

For SD3/Flux-style 1024x1024 configs, the same boundaries are typically
`[B,3,1024,1024] -> moments [B,32,128,128] -> latent [B,16,128,128]`, with
pipeline-level shift/scale and, for Flux, 2x2 latent packing outside the VAE.

Posterior contract:

```text
moments = quant_conv(encoder(image))      # [B, 2*latent_channels, h, w]
mean, logvar = chunk(moments, 2, dim=1)
logvar = clamp(logvar, -30, 20)
std = exp(0.5 * logvar)
sample = mean + std * randn_like(mean)
mode = mean
```

For deterministic inference parity, prefer `mode()` unless the pipeline path
requires posterior sampling. Img2img paths often call helper logic that samples
or uses mode depending on generator and output object.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCHW image and latent tensors; guarded NHWC conv-island candidate.
- `chunk(moments, 2, dim=1)` for mean/logvar.
- `clamp`, `exp`, random normal, multiply/add for posterior sampling.
- Per-channel scale/shift/mean/std broadcasting over `[B,C,H,W]`.
- Batch split/cat for VAE slicing.
- Spatial slicing, concat on height/width, overlap blending for VAE tiling.

### Convolution/downsample/upsample ops

- Encoder `conv_in`: `Conv2d(3 -> 128, 3x3, padding=1)` for sampled configs.
- Encoder ResNet blocks over 128/256/512/512 channels.
- Downsample2D with conv path between resolutions, padding 0 in encoder blocks.
- `conv_out`: `Conv2d(512 -> 2*latent_channels, 3x3, padding=1)`, e.g. 8 for
  SD/SDXL latent 4 or 32 for SD3/Flux latent 16.
- Optional `quant_conv`: `Conv2d(2C -> 2C, 1x1)`, active in SD/SDXL sampled
  configs and absent in SD3/Flux sampled configs.
- Optional `post_quant_conv`: `Conv2d(C -> C, 1x1)`, active in SD/SDXL sampled
  configs and absent in SD3/Flux sampled configs.
- Decoder `conv_in`: `Conv2d(latent_channels -> 512, 3x3, padding=1)`.
- Decoder ResNet/up blocks and nearest/interpolate + conv upsample.
- Decoder `conv_out`: `Conv2d(128 -> 3, 3x3, padding=1)`.

### GEMM/linear ops

- None in the common conv-only SD-style encoder/decoder outside attention
  projections.

### Attention primitives

- Optional noncausal spatial self-attention in `UNetMidBlock2D`.
- Query/key/value projections inside `Attention`; no cross-attention, masks, or
  KV cache for standard AutoencoderKL mid-block attention.

### Normalization and adaptive conditioning

- GroupNorm with `eps=1e-6`, usually 32 groups.
- SiLU activation.
- SpatialNorm only for decoder `norm_type="spatial"` paths, not active in
  common AutoencoderKL configs.

### VAE/postprocessing ops

- Latent scale/unscale and optional shift.
- Optional per-channel `latents_mean`/`latents_std`.
- VAE output remains model sample; image processor denormalization to `[0,1]`
  belongs to pipeline postprocess.

## 6. Encoder/decoder breakdown

Encoder active SD-style path:

```text
image [B,3,H,W]
-> Conv2d(3 -> 128, 3x3)
-> DownEncoderBlock2D 128: Resnet x2 -> downsample
-> DownEncoderBlock2D 256: Resnet x2 -> downsample
-> DownEncoderBlock2D 512: Resnet x2 -> downsample
-> DownEncoderBlock2D 512: Resnet x2
-> UNetMidBlock2D 512: Resnet -> optional Attention -> Resnet
-> GroupNorm(512, groups=32) -> SiLU -> Conv2d(512 -> 8, 3x3)
-> optional quant_conv 1x1
-> DiagonalGaussianDistribution
```

Decoder active SD-style path:

```text
latent [B,4,H/8,W/8]
-> optional post_quant_conv 1x1
-> Conv2d(4 -> 512, 3x3)
-> UNetMidBlock2D 512: Resnet -> optional Attention -> Resnet
-> UpDecoderBlock2D 512: Resnet x3 -> upsample
-> UpDecoderBlock2D 512: Resnet x3 -> upsample
-> UpDecoderBlock2D 256: Resnet x3 -> upsample
-> UpDecoderBlock2D 128: Resnet x3
-> GroupNorm(128, groups=32) -> SiLU -> Conv2d(128 -> 3, 3x3)
```

`ResnetBlock2D` has the same core pattern as UNet ResNet blocks but no timestep
embedding in this codec path:

```text
GroupNorm -> SiLU -> Conv2d -> GroupNorm -> SiLU -> Conv2d -> residual add
```

## 7. Attention requirements

For common configs with `mid_block_add_attention=True`, the VAE needs only
bottleneck spatial self-attention:

- Noncausal self-attention over flattened latent spatial tokens.
- No text/context K/V, no added K/V, no mask, no KV cache.
- Channel width usually 512 at 64x64 input latent bottleneck for decode.
- `Attention` comes from `attention_processor.py`; eager/native SDPA paths
  define parity.
- `AutoencoderKL.fuse_qkv_projections()` can fuse QKV for self-attention and
  installs `FusedAttnProcessor2_0`, but first parity should use unfused
  projections.

Flash-style lowering is plausible under stricter Dinoml preconditions because
this is mask-free self-attention with no added K/V, but it must still verify
dtype, head layout, dropout=0, exact scaling, and processor selection. If
`mid_block_add_attention=False`, no attention provider is needed for that config.

## 8. Scheduler and denoising-loop contract

AutoencoderKL has no scheduler. Its integration contract is the latent boundary
with pipelines:

```text
SD-style encode:     latents = scaling_factor * vae.encode(image).latent_dist.sample_or_mode()
SD-style decode:     image = vae.decode(latents / scaling_factor)
SD3/Flux-style:      image = vae.decode((latents / scaling_factor) + shift_factor)
SD3/Flux encode:     latents = (vae.encode(image) - shift_factor) * scaling_factor
SDXL optional stats: latents = latents * latents_std / scaling_factor + latents_mean
```

The exact choice of sample vs mode belongs to the calling pipeline. Dinoml should
make that explicit in the runtime plan instead of hiding it inside a Python
helper.

## 9. Position, timestep, and custom math

There are no timestep or positional embeddings in the common AutoencoderKL path.
Custom math to reproduce:

- Gaussian posterior split, logvar clamp, std/var exponentials, sampling.
- Latent scaling, optional shifting, optional mean/std denormalization.
- Tile overlap blending for tiled encode/decode if that mode is selected.

Axis-sensitive reductions from posterior `kl` and `nll` are training/evaluation
methods and ignored for current inference integration.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- Image loading, resize/crop, PIL/NumPy conversion.
- `VaeImageProcessor` normalization to `[-1,1]` for encode paths and
  denormalization after decode.

GPU/runtime work:

- Encode input tensor `[B,3,H,W]`.
- Decode latent tensor `[B,C,H/scale,W/scale]`.
- Scale/shift arithmetic around the VAE boundary.
- Optional slicing over batch for memory.
- Optional tiling over height/width for large images.

Flux-specific latent packing/unpacking is pipeline-level: the VAE still consumes
and produces spatial latent maps, while Flux packs 2x2 latent patches for its
transformer.

## 11. Graph rewrite / lowering opportunities

### Rewrite: NCHW codec conv island -> guarded NHWC codec island

Preconditions:

- Region includes convs, GroupNorm, SiLU, residual add, downsample/upsample, and
  pointwise scale/shift with all axes rewritten.
- Posterior channel split, latent mean/std, and VAE scale/shift either run in
  translated layout with explicit axis rewrite or are outside the island.
- Attention flatten/reshape is handled separately or bounded by layout
  transposes.

Replacement:

```text
NCHW boundary -> NHWC encoder/decoder conv island -> NCHW boundary
```

Weight transform:

```python
w_hwio = w_oihw.permute(2, 3, 1, 0)
```

Failure cases:

- Tiled encode/decode blending uses source `height=dim2`, `width=dim3`.
- Posterior `chunk(..., dim=1)` not rewritten to last channel.
- Consumer pipeline expects NCHW latents without a boundary conversion.

### Rewrite: posterior distribution construction

Preconditions:

- `double_z=True` and `moments` shape is `[B,2*C,H,W]`.
- Inference path chooses mode or sample explicitly.

Replacement:

```text
moments -> split mean/logvar -> clamp/exp -> mode or sample
```

Failure cases:

- Deterministic distribution variant or nonstandard latent layout.
- Random generator parity needed but RNG state is not represented.

### Rewrite: tile blend loops -> compiled overlap kernels

Preconditions:

- Tiling enabled and tile sizes/overlap factors are static or plan-visible.
- Source layout and crop limits are explicit.

Replacement:

```text
per-tile encode/decode -> vertical/horizontal linear blend -> concat/crop
```

Failure cases:

- Dynamic image shapes without planned tile grid.
- Parity target expects non-tiled output; tiled output is documented as not
  exactly identical to non-tiled output.

## 12. Kernel fusion candidates

Highest priority:

- Conv2d + GroupNorm + SiLU in encoder/decoder ResNet blocks.
- NHWC conv islands across consecutive VAE blocks with boundary elision.
- Decoder upsample + conv + GroupNorm/SiLU around the high-resolution half of
  decode.
- Latent scale/shift/mean/std as fused channel-broadcast pointwise kernels.

Medium priority:

- Mid-block self-attention QKV projection + attention + output projection.
- `quant_conv`/`post_quant_conv` 1x1 conv lowering to pointwise GEMM/conv.
- Posterior split/clamp/exp/sample or mode fusion for encode paths.
- Tiled encode/decode blend kernels for high-resolution memory-sensitive use.

Lower priority:

- Fused QKV projection API parity for VAE mid attention.
- SpatialNorm/asymmetric mask-conditioned decoder variants.
- AutoencoderTiny separate compact conv path.

## 13. Runtime staging plan

Stage 1: Parse AutoencoderKL configs and reconcile omitted defaults, especially
`scaling_factor`, `force_upcast`, quant conv flags, and `mid_block_add_attention`.

Stage 2: Load weights and run standalone decode parity for a tiny/random config:
`post_quant_conv -> decoder`.

Stage 3: Add encoder parity through posterior moments and `mode()`.

Stage 4: Add posterior sampling parity with explicit RNG handling.

Stage 5: Validate SD1.5/SDXL decode shapes and scale/unscale formulas.

Stage 6: Add encode integration for img2img/inpaint candidate reports.

Stage 7: Add guarded NHWC conv islands and layout parity tests.

Stage 8: Add optional fp32 VAE policy, slicing, and tiling as plan-visible
runtime modes.

## 14. Parity and validation plan

- Operator parity: Conv2d, GroupNorm, SiLU, Downsample2D, Upsample2D.
- Block parity: one `ResnetBlock2D`, one `DownEncoderBlock2D`, one
  `UpDecoderBlock2D`, one `UNetMidBlock2D` with and without attention.
- Full SD/SDXL decode parity for `[B,4,64,64] -> [B,3,512,512]`.
- Full SD/SDXL encode parity for `[B,3,512,512] -> moments [B,8,64,64]`.
- Full SD3/Flux-shape decode parity for `[B,16,128,128] -> [B,3,1024,1024]`
  when weights/configs are available.
- Full SD3/Flux-shape encode parity for `[B,3,1024,1024] -> moments
  [B,32,128,128]`.
- Posterior `mode()` parity and posterior `sample()` parity with fixed RNG.
- Scale/shift parity:
  - SD style `latents / scaling_factor`.
  - SD3/Flux style `(latents / scaling_factor) + shift_factor`.
  - SDXL optional mean/std path.
- Tiling/slicing parity only after non-tiled parity; tiled output should be
  compared to Diffusers tiled output, not non-tiled output.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per kernel.

## 15. Performance probes

- Decode throughput by batch and image resolution.
- Encode throughput by batch and image resolution.
- Conv/resnet vs mid-attention time split.
- NCHW faithful path vs guarded NHWC conv-island path.
- fp16/bf16/fp32 VAE numerics and latency, especially SDXL decode.
- Tiled vs non-tiled memory and latency for large images.
- `quant_conv`/`post_quant_conv` overhead.
- VAE boundary scale/shift overhead as part of pipeline post/preprocessing.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `autoencoder_tiny`: compact TAESD-style codec.
- `asymmetric_autoencoder_kl`: mask-conditioned decoder and inpaint codec path.
- `consistency_decoder_vae`: separate decoder architecture.
- `video_autoencoders`: CogVideoX/Wan/LTX/Mochi temporal codecs.
- `audio_autoencoders`: Oobleck/LTX2 audio codecs.
- `vae_tiling_slicing_runtime`: memory-policy report if Dinoml wants explicit
  tile/slice plans.

Ignored/out of scope for this audit unless explicitly selected:

- Training, KL loss use, NLL, dropout, and gradient checkpointing.
- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.

## 17. Final implementation checklist

- [ ] Parse AutoencoderKL config and reconcile source defaults.
- [ ] Load encoder, decoder, quant conv, and post-quant conv weights.
- [ ] Implement Conv2d/GroupNorm/SiLU/ResnetBlock2D parity.
- [ ] Implement DownEncoderBlock2D and UpDecoderBlock2D parity.
- [ ] Implement optional mid-block self-attention parity.
- [ ] Implement optional `quant_conv` and `post_quant_conv`, with absent-conv
      handling for SD3/Flux configs.
- [ ] Implement DiagonalGaussianDistribution mode/sample path.
- [ ] Implement latent scale, shift, mean, and std boundary transforms.
- [ ] Validate decode for SD1.5 and SDXL VAE configs.
- [ ] Validate encode for img2img/inpaint readiness.
- [ ] Add guarded NHWC conv-island rewrite and parity tests.
- [ ] Add fp32/upcast policy decision for SDXL-style configs.
- [ ] Add tiling/slicing as explicit later-stage runtime modes if needed.
