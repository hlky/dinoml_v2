# Diffusers AutoencoderTiny Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  madebyollin/taesd
  madebyollin/taesdxl
  madebyollin/taesd3
  madebyollin/taef1

Config sources:
  H:/configs had madebyollin repo directories/model_index placeholders but no
  component config.json files for these four checkpoints at audit start.
  Fetched official component configs with huggingface_hub for inspection:
    madebyollin/taesd config.json
    madebyollin/taesdxl config.json
    madebyollin/taesd3 config.json
    madebyollin/taef1 config.json
  To honor the owned-write-path constraint, fetched config files were not
  retained in H:/configs after inspection.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
  diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py
  diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
  diffusers/src/diffusers/pipelines/flux/pipeline_flux.py

Model files inspected:
  diffusers/src/diffusers/models/autoencoders/autoencoder_tiny.py
  diffusers/src/diffusers/models/autoencoders/vae.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  None. AutoencoderTiny is a Diffusers-native model component.

Any missing files or assumptions:
  No official config was gated or blocked. This report is inference-only and
  ignores XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient
  checkpointing, multi-GPU/context parallel, callbacks, and interrupt paths.
```

## 2. Component graph

`AutoencoderTiny` is a TAESD-style compact latent image codec. It can be used as
a fast approximate replacement for heavier VAE components in SD/SDXL/SD3/Flux
families, but it is not a denoiser and has no scheduler.

```text
encode:
  image tensor [B,3,H,W] in Diffusers [-1,1] convention
  -> EncoderTiny rescales image to [0,1]
  -> conv / stride-2 conv / AutoencoderTinyBlock stack
  -> raw latents [B,C,H/8,W/8]

decode:
  raw latents [B,C,H/8,W/8]
  -> tanh clamp: tanh(x / 3) * 3
  -> conv / AutoencoderTinyBlock / nearest upsample stack
  -> decoder output in [0,1]
  -> rescale to Diffusers [-1,1]
```

The `forward()` convenience path does encode, byte-style latent quantization,
unquantization, and decode. Pipelines normally call `encode()` or `decode()`
directly; first Dinoml staging should treat byte quantization as a separate
round-trip feature, not part of decode parity.

Separate candidate reports:

| Candidate | Classes/files | Reason |
| --- | --- | --- |
| `autoencoder_kl` | `AutoencoderKL`, `Encoder`, `Decoder`, `DiagonalGaussianDistribution` | Heavier KL posterior codec with GroupNorm, optional attention, quant/post-quant convs, and broader scale/shift variants. |
| `asymmetric_autoencoder_kl` | `AutoencoderAsymKL`, mask-conditioned helpers | Inpaint-specific mask/image conditioning changes the decode contract. |
| `vae_tiling_slicing_runtime` | `AutoencoderMixin`, codec tiling methods | Memory policy with tile overlap and blending loops; useful after untiled parity. |
| `flux_latent_packing` | Flux pipeline `_pack_latents` / `_unpack_latents` | Pipeline-level 2x2 latent token packing is outside AutoencoderTiny but important for Flux-family decode boundaries. |

## 3. Important config dimensions

Source defaults from `AutoencoderTiny.__init__`:

| Field | Default | Runtime effect |
| --- | ---: | --- |
| `in_channels` / `out_channels` | 3 / 3 | RGB image input/output. |
| `latent_channels` | 4 | Latent map channels; SD3/Flux tiny configs use 16. |
| `encoder_block_out_channels` | `(64,64,64,64)` | Width of each encoder stage. |
| `decoder_block_out_channels` | `(64,64,64,64)` | Width of each decoder stage. |
| `num_encoder_blocks` | `(1,3,3,3)` | Residual tiny blocks after each encoder stage conv. |
| `num_decoder_blocks` | `(3,3,3,1)` | Residual tiny blocks before each decoder stage conv. |
| `act_fn` | `relu` | Activation in tiny residual blocks. |
| `upsample_fn` | `nearest` | Decoder upsample mode. |
| `upsampling_scaling_factor` | 2 | Per decoder upsample scale. |
| `latent_magnitude` | 3 | Used by `scale_latents`/`unscale_latents` and decoder tanh clamp. |
| `latent_shift` | 0.5 | Used by byte-storage scaling helpers. |
| `scaling_factor` | 1.0 | Pipeline latent scale/unscale factor; TAESD configs keep identity. |
| `shift_factor` | 0.0 | Pipeline latent shift factor; source default covers older configs that omit it. |
| `force_upcast` | false | Registered false regardless of constructor argument. |

Representative config sweep:

| Config | latent channels | blocks | activation | scaling | shift | upsample | notes |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| `madebyollin/taesd` | 4 | enc `(1,3,3,3)`, dec `(3,3,3,1)` | ReLU | 1.0 | 0.0 | nearest x2 | SD1/SD2-style tiny codec. |
| `madebyollin/taesdxl` | 4 | same | ReLU | 1.0 | 0.0 | nearest x2 | SDXL tiny codec; same operator shape as TAESD. |
| `madebyollin/taesd3` | 16 | same | ReLU | 1.0 | omitted, effective 0.0 | nearest x2 | SD3 latent channel contract; config includes legacy `block_out_channels`. |
| `madebyollin/taef1` | 16 | same | ReLU | 1.0 | 0.0 | nearest x2 | Flux-family latent channel contract; config includes legacy `block_out_channels`. |

All inspected configs use four stages. Encoder stages after stage 0 use
`Conv2d(64 -> 64, 3x3, stride=2, padding=1, bias=false)`, so `[B,C,H,W]` maps
to `[B,latent_channels,H/8,W/8]` when `H` and `W` are multiples of 8. Decoder
has three nearest-neighbor upsample steps, so it maps back by x8.

## 3a. Family variation traps

- `taesd` and `taesdxl` use 4-channel latents; `taesd3` and `taef1` use
  16-channel latents. Do not bake in channel 4.
- `taesd3` omits `shift_factor`; current source default is effective `0.0`.
- `AutoencoderTiny` has no posterior, no `DiagonalGaussianDistribution`, no
  `quant_conv`, no `post_quant_conv`, no GroupNorm, and no attention.
- `EncoderTiny` internally converts Diffusers image tensors from `[-1,1]` to
  `[0,1]`; `DecoderTiny` returns Diffusers convention by multiplying by 2 and
  subtracting 1.
- `DecoderTiny` clamps raw latents with `tanh(x / 3) * 3` before conv layers.
  This is part of decode parity and should not be confused with pipeline
  `scaling_factor`/`shift_factor`.
- `scale_latents()` maps raw latents to `[0,1]` as
  `x / (2 * latent_magnitude) + latent_shift`, clamped to `[0,1]`.
  `unscale_latents()` reverses that helper. These helpers are used by
  `forward()` byte round-trip, not by plain `encode()` or `decode()`.
- Tiling uses `spatial_scale_factor = 2 ** out_channels`; with RGB configs this
  is 8. Treat this as source behavior, but guard if non-RGB configs appear.
- Flux pipelines pack/unpack latents outside the VAE. `taef1` still consumes a
  spatial NCHW latent map, not packed transformer tokens.

## 4. Runtime tensor contract

| Boundary | Tensor | Source layout | Candidate optimized layout | Example shapes |
| --- | --- | --- | --- | --- |
| preprocess output | image | NCHW, `[-1,1]` | NHWC only with boundary rewrite | `[B,3,512,512]`, `[B,3,1024,1024]` |
| encoder internal input | image in `[0,1]` | NCHW | NHWC guarded conv island | `[B,3,H,W]` |
| encode output | raw latents | NCHW | NHWC guarded candidate | TAESD `[B,4,H/8,W/8]`; TAEF1 `[B,16,H/8,W/8]` |
| denoiser boundary | scaled/shifted latents | pipeline-owned NCHW | usually preserve NCHW initially | family-specific |
| decode input | raw latents after pipeline unscale/unshift | NCHW | NHWC guarded conv island | `[B,4,h,w]` or `[B,16,h,w]` |
| decoder output | image sample | NCHW, `[-1,1]` | NHWC with postprocess boundary | `[B,3,H,W]` |
| postprocess output | image | NCHW tensor, NHWC NumPy/PIL | CPU/data path | output type dependent |

Pipeline scale/shift formulas remain component-boundary work:

```text
SD/SDXL-style decode:  vae.decode(latents / vae.config.scaling_factor)
SD3/Flux-style decode: vae.decode((latents / scaling_factor) + shift_factor)
AutoencoderTiny configs: scaling_factor = 1.0, shift_factor = 0.0
```

For `taef1` in Flux, the pipeline first unpacks `[B, tokens, packed_dim]`
latents back to `[B,16,H/8,W/8]`, then applies the VAE scale/shift formula and
calls `decode()`.

CPU/data-pipeline work includes PIL/NumPy conversion, resize to multiples of
`vae_scale_factor`, normalization/denormalization, and output conversion.
GPU/runtime work includes the codec conv graph, tanh clamp, image convention
rescale, optional pipeline scale/shift, and optional tile/slice loops.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image and latent maps.
- Elementwise image rescale: `(x + 1) / 2` in encode and `x * 2 - 1` after decode.
- Elementwise latent clamp: `tanh(x / 3) * 3`.
- Optional helper path: divide/add/clamp/mul/round/byte/cast for byte latent
  round-trip in `forward()`.
- Optional batch split/cat for slicing.
- Optional spatial tile slicing, `copy_`, and overlap blending.

Convolution/downsample/upsample ops:

- Encoder first conv: `Conv2d(3 -> 64, 3x3, padding=1)`.
- Encoder downsample stages: three `Conv2d(64 -> 64, 3x3, stride=2, padding=1,
  bias=false)`.
- Encoder final conv: `Conv2d(64 -> latent_channels, 3x3, padding=1)`.
- Decoder first conv: `Conv2d(latent_channels -> 64, 3x3, padding=1)`.
- Decoder stage convs: `Conv2d(64 -> 64, 3x3, padding=1, bias=false)` for
  non-final stages; final `Conv2d(64 -> 3, 3x3, padding=1, bias=true)`.
- Decoder nearest upsample x2 after the first three decoder stages.
- `AutoencoderTinyBlock`: three 3x3 convs, two configured activations inside
  the branch, residual add, final ReLU; skip is identity for inspected configs.

Normalization, attention, and GEMM:

- No normalization required for inspected AutoencoderTiny configs.
- No attention required.
- No linear/GEMM required except optional lowering of 1x1/3x3 convs.

## 6. Model breakdown

EncoderTiny with inspected configs:

```text
input [B,3,H,W] in [-1,1]
-> add(1).div(2)
-> Conv2d(3 -> 64, 3x3)
-> AutoencoderTinyBlock x1
-> Conv2d(64 -> 64, 3x3, stride=2, bias=false)
-> AutoencoderTinyBlock x3
-> Conv2d(64 -> 64, 3x3, stride=2, bias=false)
-> AutoencoderTinyBlock x3
-> Conv2d(64 -> 64, 3x3, stride=2, bias=false)
-> AutoencoderTinyBlock x3
-> Conv2d(64 -> latent_channels, 3x3)
```

DecoderTiny with inspected configs:

```text
input [B,latent_channels,h,w]
-> tanh(x / 3) * 3
-> Conv2d(latent_channels -> 64, 3x3)
-> ReLU
-> AutoencoderTinyBlock x3 -> Upsample(nearest, x2) -> Conv2d(64 -> 64, 3x3, bias=false)
-> AutoencoderTinyBlock x3 -> Upsample(nearest, x2) -> Conv2d(64 -> 64, 3x3, bias=false)
-> AutoencoderTinyBlock x3 -> Upsample(nearest, x2) -> Conv2d(64 -> 64, 3x3, bias=false)
-> AutoencoderTinyBlock x1 -> Conv2d(64 -> 3, 3x3, bias=true)
-> mul(2).sub(1)
```

`AutoencoderTinyBlock(64,64)` active shape:

```text
branch: Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d
skip: identity
out: ReLU(branch + skip)
```

Source supports a 1x1 skip conv if `in_channels != out_channels`, but the
representative configs keep block widths fixed at 64 so this branch is inactive.

## 7. Attention requirements

None for `AutoencoderTiny`. `attention_processor.py` and `attention_dispatch.py`
are not part of the active first-slice parity path. Flash-style kernels are not
needed for this codec.

## 8. Scheduler and loop contract

AutoencoderTiny has no scheduler and no denoising-loop state. The loop contract
is purely the latent boundary:

- Text-to-image decode-only paths pass final denoised latents through pipeline
  unscale/unshift, then `vae.decode()`, then image postprocess.
- Img2img/inpaint/control paths can call `vae.encode()` to produce latents for
  noise injection, denoiser input composition, or mask/image latent contracts.
- Since inspected tiny configs use `scaling_factor=1.0` and effective
  `shift_factor=0.0`, pipeline scale/shift is identity for these checkpoints,
  but Dinoml should still represent the fields because the pipeline code reads
  them generically.

## 9. Position, timestep, and custom math

No position embeddings, timestep embeddings, Fourier features, RoPE, or adaptive
conditioning occur in AutoencoderTiny.

Custom math to preserve:

```text
encode image convention:  x_01 = (x_minus1_to_1 + 1) / 2
decode latent clamp:      x = tanh(x / 3) * 3
decode image convention:  image = image_01 * 2 - 1
byte helper scale:        scaled = clamp(raw / (2 * latent_magnitude) + latent_shift, 0, 1)
byte helper unscale:      raw = (scaled - latent_shift) * (2 * latent_magnitude)
```

The byte helper path additionally does `mul(255).round().byte()` and
`byte / 255.0` inside `forward()`.

## 10. Preprocessing and input packing

`VaeImageProcessor` provides image boundary work:

- resize/crop/fill to requested dimensions and multiples of `vae_scale_factor`;
- PIL/NumPy/PyTorch conversion;
- optional RGB/grayscale conversion;
- normalization to `[-1,1]`;
- postprocess denormalization and conversion to `pt`, `np`, or `pil`;
- early return for tensor inputs whose channel count equals `vae_latent_channels`.

Flux latent packing is not AutoencoderTiny work. The Flux pipeline unpacks
transformer latents back into NCHW latent maps before calling the VAE.

## 11. Graph rewrite / lowering opportunities

### Rewrite: AutoencoderTiny conv island to guarded NHWC

Source pattern:

```text
NCHW Conv2d/ReLU/residual/Upsample stack
```

Replacement:

```text
NCHW boundary -> NHWC conv island -> NCHW boundary
```

Preconditions:

- Region is limited to image convention pointwise ops, convs, ReLU, residual
  adds, tanh clamp, and nearest upsample.
- All consumers inside the island are layout-rewritten together.
- Tile/slice loops either remain outside the island or rewrite height/width
  indexing from source `[-2]`/`[-1]` consistently.
- Pipeline latent packing/unpacking remains outside this rewrite.

Weight transform:

```text
OIHW -> HWIO for Conv2d weights
```

Failure cases:

- Layout pass changes the public latent boundary expected by denoisers.
- Tiled encode/decode blending keeps NCHW indexing while the tile tensor is
  translated.
- Non-nearest `upsample_fn` variants appear and require backend-specific parity.

### Rewrite: tiny block fusion

Source pattern:

```text
Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> Add(skip) -> ReLU
```

Replacement:

```text
fused tiny residual block kernel or scheduled conv block
```

Preconditions:

- Static channel width, padding=1, stride=1 inside block.
- Skip is identity or explicit 1x1 conv with known weights.
- Activation is ReLU for inspected configs.

Parity test sketch:

- Compare one block for `[1,64,64,64]` and `[2,64,128,128]` in fp32/fp16.
- Include a synthetic `in_channels != out_channels` config only after the
  common identity-skip path is stable.

## 12. Kernel fusion candidates

Highest priority:

- `Conv2d + ReLU` and `Conv2d + Add + ReLU` in `AutoencoderTinyBlock`.
- Decoder nearest upsample + following 3x3 conv scheduling, especially at high
  spatial resolutions.
- Latent tanh clamp fused with decoder input conversion.
- Image convention rescale fused with first encoder or final decoder boundary.

Medium priority:

- Whole-stage NHWC conv islands for encoder/decode.
- Byte helper path fusion for applications that store TAESD latents as RGBA-like
  uint8 tensors.
- Tile overlap blend kernels if tiled tiny decode is admitted.

Lower priority:

- Generic activation support beyond ReLU.
- Non-identity skip conv branch in tiny blocks.
- Slicing/tiling as memory-policy kernels.

## 13. Runtime staging recommendation

First Dinoml staging/admission recommendation: admit `AutoencoderTiny` before
`AutoencoderKL` as a standalone codec micro-target, starting with decode-only
TAESD/TAESDXL parity. It has a small, regular conv graph, no posterior RNG, no
GroupNorm, no attention, and no scheduler coupling, so it is a good first
Diffusers VAE admission path and a useful CUDA/NHWC conv-island proving ground.

Suggested stages:

1. Parse `AutoencoderTiny` configs and reconcile omitted defaults, especially
   `shift_factor=0.0` for `taesd3`.
2. Load weights for `madebyollin/taesd` or `madebyollin/taesdxl`.
3. Implement decode parity for `[B,4,64,64] -> [B,3,512,512]`.
4. Add encode parity for `[B,3,512,512] -> [B,4,64,64]`.
5. Add 16-channel decode/encode parity for `taesd3` and `taef1`.
6. Add guarded NHWC conv-island lowering and compare against faithful NCHW.
7. Add optional byte round-trip `forward()` parity.
8. Add slicing/tiling only after untiled encode/decode parity is stable.

## 14. Parity and validation plan

- Unit parity for `AutoencoderTinyBlock(64,64)` with ReLU.
- Encoder stage parity for each stride-2 downsample boundary.
- Decoder stage parity for nearest upsample + conv boundaries.
- Full decode parity:
  - TAESD/TAESDXL: `[B,4,64,64] -> [B,3,512,512]`.
  - TAESD3/TAEF1: `[B,16,128,128] -> [B,3,1024,1024]` for common pipeline shapes.
- Full encode parity:
  - `[B,3,512,512] -> [B,4,64,64]`.
  - `[B,3,1024,1024] -> [B,16,128,128]`.
- Boundary math parity for `(x + 1) / 2`, `tanh(x / 3) * 3`, `x * 2 - 1`,
  `scale_latents()`, and `unscale_latents()`.
- Pipeline boundary parity for identity scale/shift and generic
  `(latents / scaling_factor) + shift_factor`.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per kernel.

## 15. Performance probes

- Decode throughput by batch and output resolution.
- Encode throughput by batch and input resolution.
- NCHW faithful path versus guarded NHWC conv island.
- Time split by conv stages and high-resolution decoder stages.
- Nearest upsample + conv memory bandwidth and temporary usage.
- 4-channel versus 16-channel latent decode overhead.
- Byte round-trip overhead if admitted.
- Tiled versus untiled memory and latency after untiled parity.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `autoencoder_kl`: heavier latent codec with posterior, GroupNorm, optional
  attention, scale/shift variants, and tiling/slicing parity.
- `asymmetric_autoencoder_kl`: inpaint/mask-conditioned decode path.
- `vae_tiling_slicing_runtime`: explicit memory-policy admission for tile and
  batch slicing.
- `flux_latent_packing`: packed transformer latent token contract outside VAE.
- `image_processor_runtime`: host preprocessing/postprocessing policy if Dinoml
  later wants more of PIL/NumPy/resize/normalize in graph form.

Ignored/out of scope for this audit:

- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.

## 17. Final implementation checklist

- [ ] Parse `AutoencoderTiny` config and apply source defaults.
- [ ] Load TAESD/TAESDXL 4-channel weights.
- [ ] Implement `AutoencoderTinyBlock` conv/ReLU/residual parity.
- [ ] Implement `DecoderTiny` decode parity for 4-channel latents.
- [ ] Implement `EncoderTiny` encode parity for 4-channel latents.
- [ ] Add 16-channel config/weight handling for TAESD3 and TAEF1.
- [ ] Preserve image convention rescale and decoder tanh clamp.
- [ ] Preserve generic pipeline `scaling_factor` and `shift_factor` boundary fields.
- [ ] Add guarded NHWC conv-island rewrite with axis/layout tests.
- [ ] Add optional byte latent round-trip parity.
- [ ] Defer slicing/tiling until untiled codec parity is stable.
