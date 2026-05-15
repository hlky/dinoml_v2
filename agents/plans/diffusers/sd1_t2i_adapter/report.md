# Diffusers SD1 T2I-Adapter Runtime Surface Report

## 1. Source Basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.
  Remote upstream: https://github.com/huggingface/diffusers.git

Model id(s):
  Primary SD 1.x adapter configs:
    TencentARC/t2iadapter_canny_sd15v2
    TencentARC/t2iadapter_color_sd14v1
    TencentARC/t2iadapter_depth_sd15v2
    TencentARC/t2iadapter_sketch_sd15v2
    TencentARC/t2iadapter_zoedepth_sd15v1
  Base SD reference:
    stable-diffusion-v1-5/stable-diffusion-v1-5
  Related variant inventory only:
    Adapter/t2iadapter, sketch_sdxl_1.0
    TencentARC/t2i-adapter-canny-sdxl-1.0
    TencentARC/t2i-adapter-depth-midas-sdxl-1.0
    TencentARC/t2i-adapter-sketch-sdxl-1.0

Config sources:
  Local cache checked first:
    H:/configs/TencentARC/t2iadapter_canny_sd15v2/model_index.json
    H:/configs/TencentARC/t2iadapter_color_sd14v1/model_index.json
    H:/configs/TencentARC/t2iadapter_depth_sd15v2/model_index.json
    H:/configs/TencentARC/t2iadapter_sketch_sd15v2/model_index.json
    H:/configs/TencentARC/t2iadapter_zoedepth_sd15v1/model_index.json
    H:/configs/Adapter/t2iadapter/model_index.json
  These local files exist but contain only `{}` and no component config fields.
  Network-inspected raw JSON, not saved because this task owns only this report path:
    https://huggingface.co/TencentARC/t2iadapter_canny_sd15v2/raw/main/config.json
    https://huggingface.co/TencentARC/t2iadapter_color_sd14v1/raw/main/config.json
    https://huggingface.co/TencentARC/t2iadapter_depth_sd15v2/raw/main/config.json
    https://huggingface.co/TencentARC/t2iadapter_sketch_sd15v2/raw/main/config.json
    https://huggingface.co/TencentARC/t2iadapter_zoedepth_sd15v1/raw/main/config.json
    https://huggingface.co/Adapter/t2iadapter/raw/main/sketch_sdxl_1.0/config.json
    https://huggingface.co/TencentARC/t2i-adapter-canny-sdxl-1.0/raw/main/config.json
    https://huggingface.co/TencentARC/t2i-adapter-depth-midas-sdxl-1.0/raw/main/config.json
    https://huggingface.co/TencentARC/t2i-adapter-sketch-sdxl-1.0/raw/main/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/unet/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/vae/config.json
    https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/raw/main/scheduler/scheduler_config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_adapter.py
  diffusers/src/diffusers/pipelines/t2i_adapter/pipeline_stable_diffusion_xl_adapter.py
  diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

Model files inspected:
  diffusers/src/diffusers/models/adapter.py
  diffusers/src/diffusers/models/unets/unet_2d_condition.py
  diffusers/src/diffusers/models/unets/unet_2d_blocks.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/schedulers/scheduling_pndm.py
  diffusers/src/diffusers/schedulers/scheduling_ddim.py
  diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py

External component configs inspected:
  Base SD 1.5 UNet, VAE, and scheduler configs listed above. CLIP text encoder
  details are inherited from the stable_diffusion_1_5 report and are treated as
  external prompt embeddings for the first adapter slice.

Any missing files or assumptions:
  No gated official config blocker was encountered. The local adapter cache was
  present but empty, so source dimensions below come from official raw config
  JSON and source defaults. Safety/NSFW, training/loss/dropout/gradient
  checkpointing, callbacks/interrupt, multi-GPU/context parallel, XLA/NPU/MPS,
  Flax, and ONNX paths are out of scope. Runtime target is inference-only CUDA,
  faithful NCHW first, with NHWC/channel-last only as a guarded optimization.
```

## 2. Pipeline And Component Graph

`StableDiffusionAdapterPipeline` is the SD 1.x text-to-image pipeline plus an
adapter side model. It registers `vae`, `text_encoder`, `tokenizer`, `unet`,
`adapter`, `scheduler`, and optional safety components. If the constructor gets a
list or tuple of `T2IAdapter` instances, it wraps them in `MultiAdapter`.
Offload sequence is `text_encoder->adapter->unet->vae`.

```text
adapter image / condition preprocessing
  -> T2IAdapter or MultiAdapter, once before denoising
  -> scaled feature pyramid tensors
prompt strings or prompt_embeds
  -> CLIP tokenizer/text encoder, or external prompt embeddings
  -> CFG negative/positive prompt batching
latent initialization [B,4,H/8,W/8]
  -> denoising loop:
       scheduler.scale_model_input
       -> UNet2DConditionModel(..., down_intrablock_additional_residuals=adapter_state)
       -> CFG arithmetic
       -> scheduler.step
  -> VAE decode(latents / scaling_factor)
  -> image postprocess
```

Contrast with ControlNet: ControlNet runs a large side UNet every denoising
timestep and injects `down_block_additional_residuals` plus a mid residual after
base UNet down/mid blocks. T2I-Adapter runs a small conv/residual feature
extractor once before the loop and injects `down_intrablock_additional_residuals`
inside the UNet down path. There is no adapter mid residual in SD 1.x.

Required first-slice components are external prompt embeddings, preprocessed
adapter condition image tensor, `T2IAdapter`, base SD 1.5 `UNet2DConditionModel`,
scheduler state/step, VAE decode, and postprocess. Cacheable stages include
prompt embeddings, adapter features for a fixed condition image and output size,
scheduler timestep tables, and caller-supplied initial latents.

Separate candidate reports:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `sd1_t2i_adapter_multi` | `MultiAdapter`, `models/adapter.py` | Runs multiple adapters over a list of condition images, then weighted-sums matching feature slots. This is small enough to stage after single-adapter parity but deserves explicit list/scale tests. |
| `sdxl_t2i_adapter` | `StableDiffusionXLAdapterPipeline`, `FullAdapterXL` | SDXL dual encoders, pooled/time IDs, optional IP-Adapter image embeds, `adapter_conditioning_factor`, and XL feature shapes. Separate from SD1. |
| `sd1_controlnet` | `StableDiffusionControlNetPipeline`, `ControlNetModel` | Full side UNet every step, down/mid residual injection. Already audited separately. |
| `sd1_ip_adapter` | `IPAdapterMixin`, `IPAdapterAttnProcessor*` | Adds image embeddings and attention K/V branches. Not active for SD1 T2I-Adapter first slice. |
| `sd1_lora_textual_inversion_adapters` | SD adapter pipeline inherits LoRA and textual inversion mixins | Text/token embedding and weight mutation surface. |
| `sd1_img2img`, `sd1_inpaint`, `sd1_depth2img`, `sd_upscale`, `sd1_gligen` | SD-family sibling pipelines | Different image/latent preprocessing or conditioning contracts; not part of this adapter report. |

## 3. Important Config Dimensions

Representative SD 1.x adapter configs:

| Repo/config | Class | adapter type | input channels | channels | residual blocks | downscale field | effective downscale | feature slots |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `TencentARC/t2iadapter_canny_sd15v2` | `T2IAdapter` | `full_adapter` | 1 | 320,640,1280,1280 | 2 | omitted | 8 source default | 4 |
| `TencentARC/t2iadapter_sketch_sd15v2` | `T2IAdapter` | `full_adapter` | 1 | 320,640,1280,1280 | 2 | omitted | 8 source default | 4 |
| `TencentARC/t2iadapter_depth_sd15v2` | `T2IAdapter` | `full_adapter` | 3 | 320,640,1280,1280 | 2 | omitted | 8 source default | 4 |
| `TencentARC/t2iadapter_zoedepth_sd15v1` | `T2IAdapter` | `full_adapter` | 3 | 320,640,1280,1280 | 2 | omitted | 8 source default | 4 |
| `TencentARC/t2iadapter_color_sd14v1` | `T2IAdapter` | `light_adapter` | 3 | 320,640,1280 | 4 | omitted | 8 source default | 4 |

Base SD 1.5 dimensions inherited by the adapter pipeline:

| Component | Field | Value | Source |
| --- | --- | ---: | --- |
| UNet | latent sample size | 64 | component config |
| UNet | in/out channels | 4 / 4 | component config |
| UNet | block channels | 320,640,1280,1280 | component config |
| UNet | down blocks | CA,CA,CA,Down | component config |
| UNet | layers per block | 2 | component config |
| UNet | cross-attention dim | 768 | component config |
| UNet | norm groups | 32 | component config |
| VAE | sample size / latent channels | 512 / 4 | component config |
| VAE | scale factor | 8 from 4 VAE block levels | source/config |
| VAE | scaling factor | omitted in old config, effective `0.18215` | source default |
| scheduler | sampled default | `PNDMScheduler`, `scaled_linear`, 1000 train steps | scheduler config |
| scheduler | prediction type | omitted, effective epsilon | source default |

Adapter feature shapes for 512x512 SD 1.x:

| Adapter | Input | Feature slots before CFG duplication |
| --- | --- | --- |
| full adapter, downscale 8 | `[B,C,512,512]` | `[B,320,64,64]`, `[B,640,32,32]`, `[B,1280,16,16]`, `[B,1280,8,8]` |
| light adapter, downscale 8 | `[B,3,512,512]` | `[B,320,64,64]`, `[B,640,32,32]`, `[B,1280,16,16]`, `[B,1280,8,8]` |

SDXL/T2I variants inventoried, not deeply audited:

| Repo/config | adapter type | input channels | channels | downscale | notable delta |
| --- | --- | ---: | --- | ---: | --- |
| `Adapter/t2iadapter`, `sketch_sdxl_1.0` | `full_adapter_xl` | 1 | 320,640,1280,1280 | 16 | XL feature pattern has two 64x64 slots and two 32x32 slots for 1024 inputs. |
| `TencentARC/t2i-adapter-canny-sdxl-1.0` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 16 | SDXL pipeline uses added text/time IDs and optional IP-Adapter. |
| `TencentARC/t2i-adapter-depth-midas-sdxl-1.0` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 16 | Same runtime adapter skeleton, different condition image semantics. |
| `TencentARC/t2i-adapter-sketch-sdxl-1.0` | `full_adapter_xl` | 3 | 320,640,1280,1280 | 16 | Same XL adapter skeleton. |

Recommended first Dinoml scheduler slice: inherit the SD 1.5 report's explicit
scheduler choice. PNDM is checkpoint-default parity; DDIM or Euler may be easier
for staging, but adapter-specific logic is scheduler-independent except for the
normal latent scale/step loop.

## 3a. Family Variation Traps

- Adapter config files often omit `downscale_factor`; source default is 8.
- Adapter image preprocessing does not use `VaeImageProcessor`; it resizes with
  Lanczos, converts to float `[0,1]`, and transposes NHWC to NCHW. It does not
  normalize to `[-1,1]`.
- PIL grayscale images become `[B,1,H,W]`; RGB images become `[B,3,H,W]`.
  Tensor inputs are passed through directly unless supplied as a list, so caller
  shape/channel validation must match the adapter config.
- `_default_height_width` rounds height/width down to a multiple of
  `adapter.downscale_factor`, not the full adapter `total_downscale_factor`.
  For standard SD1 full/light adapters this is multiple-of-8.
- `PixelUnshuffle(downscale_factor)` requires height and width divisible by the
  downscale factor and changes channels from `C` to `C * factor^2`.
- MultiAdapter requires identical `downscale_factor` and
  `total_downscale_factor` across all adapters; it does not check channel or
  feature width compatibility beyond the eventual tensor adds.
- Source tensors are NCHW. NHWC is attractive for conv/pointwise islands, but
  PixelUnshuffle axes, channel concat/repeat, AvgPool2d spatial axes, and UNet
  injection shapes need guards and explicit axis rewrites.
- SD1 injects adapter features only in the down path. SDXL can pass remaining
  adapter features into the mid block when the shape matches.
- SDXL adds `adapter_conditioning_factor`, allowing adapter residuals only for
  the first fraction of timesteps. SD1 applies them for every denoising step.

## 4. Runtime Tensor Contract

For 512x512 SD1 with CFG:

| Boundary | Tensor | Source layout | Shape | Notes |
| --- | --- | --- | --- | --- |
| adapter condition | `image` after preprocessing | NCHW | full canny/sketch `[B,1,512,512]`; depth/color `[B,3,512,512]` | Values in `[0,1]`; no VAE normalization. |
| adapter unshuffle | internal | NCHW | downscale 8: `[B,C*64,64,64]` | PixelUnshuffle source layout. |
| adapter features | `adapter_state` | NCHW | full/light slots listed above | Computed once before loop, then scaled. |
| adapter features after `num_images_per_prompt` | `adapter_state[k]` | NCHW | `[B*N,C,H,W]` | `repeat(num_images_per_prompt,1,1,1)`. |
| adapter features after CFG | `adapter_state[k]` | NCHW | `[2*B*N,C,H,W]` | `torch.cat([v] * 2, dim=0)`. |
| prompt embeddings | `prompt_embeds` | `[B,T,C]` | `[2*B*N,77,768]` | External boundary for first slice. |
| latents | `latents` | NCHW | `[B*N,4,64,64]` | Scheduler init sigma applied. |
| UNet input | `latent_model_input` | NCHW | `[2*B*N,4,64,64]` | CFG batch duplicated. |
| UNet adapter arg | `down_intrablock_additional_residuals` | list of NCHW tensors | cloned list per timestep | UNet pops entries destructively. |
| UNet output | `noise_pred` | NCHW | `[2*B*N,4,64,64]` | Chunked for CFG. |
| scheduler output | `latents` | NCHW | `[B*N,4,64,64]` | Fed to next step. |
| VAE decode input | `latents / scaling_factor` | NCHW | `[B*N,4,64,64]` | Base SD path. |

Adapter injection contract in `UNet2DConditionModel`:

- For each cross-attention down block, UNet pops one adapter tensor and passes it
  as `additional_residuals` to the block.
- `CrossAttnDownBlock2D` adds that residual after the last resnet+attention pair
  of the block, before output states and before any downsampler output is
  appended.
- For a non-cross-attention down block, UNet runs the block and then adds one
  adapter tensor directly to the block output `sample`.
- Adapter tensors are not added to the saved `down_block_res_samples` as a
  ControlNet-style residual tuple; they alter the hidden state flowing forward.
- In SDXL only, if an adapter tensor remains after the mid block and its shape
  matches the mid sample, source adds it to the mid sample.

CPU/data-pipeline work: image resize/color conversion, tokenization, optional
textual inversion/LoRA prompt handling, and PIL/NumPy postprocess. GPU/runtime
work: adapter feature extraction, UNet step, CFG, scheduler arithmetic, and VAE
decode.

## 5. Operator Coverage Checklist

Tensor/layout ops:

- PIL/NumPy image to NCHW float tensor in `[0,1]`; grayscale singleton channel.
- `PixelUnshuffle(factor=8)` for SD1 and `factor=16` for SDXL variants.
- Batch `repeat`, `cat`, and `chunk` for image count and CFG.
- List/tuple feature slots cloned per denoising step because UNet pops them.
- Shape checks that adapter feature slots match UNet injection hidden states.

Adapter convolution/pooling ops:

- Full adapter: `Conv2d(C*64 -> 320, 3x3, padding=1)`.
- Full adapter blocks: optional `AvgPool2d(2,stride=2,ceil_mode=True)`,
  optional `Conv2d(in -> out, 1x1)`, then `num_res_blocks` of
  `Conv2d(C -> C, 3x3,pad=1) -> ReLU -> Conv2d(C -> C, 1x1) -> add`.
- Light adapter blocks: optional `AvgPool2d(2,stride=2,ceil_mode=True)`,
  `Conv2d(in -> out/4, 1x1)`, repeated
  `Conv2d(mid -> mid, 3x3,pad=1) -> ReLU -> Conv2d(mid -> mid,3x3,pad=1) -> add`,
  then `Conv2d(mid -> out,1x1)`.
- SDXL `FullAdapterXL`: `PixelUnshuffle(16)`, input conv, same AdapterBlock
  kind, but only one downsampling block.

Base SD operators inherited from the SD1 report:

- UNet ResnetBlock2D, Downsample2D/Upsample2D, GroupNorm, SiLU.
- Cross-attention and feed-forward GEMMs/attention.
- Timestep embedding MLP.
- CFG arithmetic and scheduler step.
- VAE decode.

Control/adapter ops:

- Single-adapter feature scaling by scalar `adapter_conditioning_scale`.
- MultiAdapter per-adapter feature scaling and slot-wise summation.
- UNet intrablock residual add at exact down-block sites.

## 6. Adapter And Denoiser Breakdown

Full SD1 adapter:

```text
condition image [B,C,H,W]
  -> PixelUnshuffle(8): [B,C*64,H/8,W/8]
  -> Conv2d(C*64 -> 320, 3x3,pad=1)
  -> AdapterBlock 320 -> 320, no down
       AdapterResnetBlock x2
       output feature 0: [B,320,H/8,W/8]
  -> AdapterBlock 320 -> 640, AvgPool2d ceil stride 2 + 1x1
       AdapterResnetBlock x2
       output feature 1: [B,640,ceil(H/16),ceil(W/16)]
  -> AdapterBlock 640 -> 1280, AvgPool2d ceil stride 2 + 1x1
       output feature 2
  -> AdapterBlock 1280 -> 1280, AvgPool2d ceil stride 2
       output feature 3
```

For normal 512x512 inputs divisible by 8, the AvgPool2d ceil behavior produces
the expected 64, 32, 16, and 8 feature resolutions. If a caller supplies unusual
multiple-of-8 but not multiple-of-64 dimensions, ceil pooling can produce
adapter feature sizes that must still match the UNet hidden states; first
Dinoml admission should require the normal SD resolution divisibility by the
UNet downsampling ladder, not merely the PixelUnshuffle factor.

Light SD1 adapter:

```text
condition image [B,3,H,W]
  -> PixelUnshuffle(8): [B,192,H/8,W/8]
  -> LightAdapterBlock 192 -> 320, no down
  -> LightAdapterBlock 320 -> 640, AvgPool2d ceil stride 2
  -> LightAdapterBlock 640 -> 1280, AvgPool2d ceil stride 2
  -> LightAdapterBlock 1280 -> 1280, AvgPool2d ceil stride 2
```

Base UNet path with adapter:

```text
sample -> UNet conv_in
down block 0 CrossAttnDownBlock2D:
  resnet/attention pairs
  add adapter feature 0 after final pair
  downsample
down block 1 CrossAttnDownBlock2D:
  add adapter feature 1 after final pair
down block 2 CrossAttnDownBlock2D:
  add adapter feature 2 after final pair
down block 3 DownBlock2D:
  run block, then sample += adapter feature 3
mid/up/output:
  normal SD1 UNet path
```

## 7. Attention Requirements

T2I-Adapter itself has no attention. The required attention surface is exactly
the base SD1 UNet attention inherited from the stable_diffusion_1_5 report:
noncausal self/cross attention in `BasicTransformerBlock`, CLIP text K/V,
`attention_processor.py` eager or SDPA path as parity reference, optional fused
QKV as a later optimization. Adapter injection does not add masks, K/V branches,
RoPE, QK norm, or varlen requirements.

The adapter does affect attention indirectly because it changes hidden states
before down-block output states and downstream attention blocks. One-block
parity should test the addition site and not only compare final UNet output.

## 8. Scheduler And Denoising-Loop Contract

Adapter-specific denoising loop work:

```text
adapter_state = adapter(adapter_input)
adapter_state[k] *= adapter_conditioning_scale       # single adapter
adapter_state = MultiAdapter(adapter_inputs, scales) # multi adapter
repeat for num_images_per_prompt
duplicate batch for CFG

for each timestep:
  latent_model_input = cat([latents] * 2) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = unet(..., down_intrablock_additional_residuals=[state.clone() for state in adapter_state])
  noise_pred = uncond + guidance_scale * (text - uncond) if CFG
  latents = scheduler.step(noise_pred, t, latents).prev_sample
```

The adapter feature extraction is loop-invariant for SD1 and should not be
recomputed per timestep. Scheduler state remains the same SD1 contract:
`set_timesteps`, optional `scale_model_input`, prediction conversion, stateful
step, and CFG. Keep scheduler loop control in host code initially.

## 9. Position, Timestep, And Custom Math

Adapter custom math is small:

- PixelUnshuffle is a deterministic layout rearrangement:
  `[B,C,H,W] -> [B,C*r*r,H/r,W/r]`.
- Full adapter residual block:

```python
h = relu(conv3x3(x))
h = conv1x1(h)
out = h + x
```

- Light adapter residual block:

```python
h = relu(conv3x3(x))
h = conv3x3(h)
out = h + x
```

No adapter timestep embedding exists. Time embeddings, optional
`time_cond_proj_dim` guidance embeddings, and scheduler math are inherited from
the base SD1 pipeline. `adapter_conditioning_scale` may be a scalar for a single
adapter or a list for MultiAdapter.

## 10. Preprocessing And Input Packing

`_preprocess_adapter_image` behavior:

- `torch.Tensor` input is returned unchanged.
- `PIL.Image.Image` is wrapped as a list.
- PIL images are resized to `(width,height)` using Lanczos.
- Grayscale arrays become `[1,H,W,1]` per image; RGB arrays remain
  `[1,H,W,C]`.
- Concatenated NumPy image batch is cast to float32, divided by 255, transposed
  to NCHW, then converted to torch.
- A list of 3D tensors is stacked on dim 0; a list of 4D tensors is concatenated
  on dim 0.

MultiAdapter image input is a Python list with one condition image or image
batch per adapter. The pipeline preprocesses each image independently and passes
a list of tensors into `MultiAdapter.forward`. The implementation iterates
`zip(xs, adapter_weights, self.adapters)`, so first-slice Dinoml staging should
model MultiAdapter as separate adapter calls plus feature-slot reduction rather
than a single channel-concatenated tensor despite the docstring wording.

## 11. Graph Rewrite / Lowering Opportunities

### Rewrite: PixelUnshuffle plus conv as strided patch conv

Source pattern:

```text
PixelUnshuffle(r) -> Conv2d(C*r*r -> O, 3x3,pad=1)
```

Replacement: either implement PixelUnshuffle as a layout op followed by normal
conv, or fuse into a specialized conv over `r x r` input patches.

Preconditions: fixed `r`, NCHW input, height/width divisible by `r`, static
adapter input channels, no intervening consumer of the unshuffled tensor.

Failure cases: tensor input with unexpected layout, dynamic image size not
divisible by `r`, or NHWC path without explicit pixel-unshuffle axis rewrite.

Parity test: compare unshuffle+conv and fused lowering for grayscale and RGB
inputs at 512 and 768 sizes.

### Rewrite: Adapter conv island to guarded NHWC

Source pattern: PixelUnshuffle, Conv2d, AvgPool2d, ReLU, residual add, 1x1/3x3
convs in adapter blocks.

Replacement: NCHW boundary -> NHWC internal adapter feature extractor -> NCHW
feature outputs for UNet injection, or keep UNet down path in the same NHWC
island under stronger guards.

Preconditions: all adapter ops in the island are layout-rewritten; PixelUnshuffle
maps channel expansion to the NHWC channel axis; AvgPool spatial axes become
H/W axes; conv weights transform from OIHW to HWIO; residual feature outputs are
transposed back unless the consuming UNet region is also NHWC.

Failure cases: direct tensor input already in source NCHW, feature outputs
consumed by a faithful NCHW UNet, or odd size paths where ceil pooling and UNet
downsample shape rules diverge.

### Rewrite: MultiAdapter slot reduction

Source pattern:

```text
features_i = adapter_i(image_i)
accum[k] = sum_i scale_i * features_i[k]
```

Replacement: explicit feature-slot accumulation buffers, optionally fusing
scale multiply with the first consumer add into UNet.

Preconditions: same number of feature slots, same shapes/dtypes per slot,
compatible downscale factors, static adapter count for compile.

Failure cases: heterogeneous SD1/SDXL adapters, dynamic adapter list length, or
shape mismatch hidden until feature add.

### Rewrite: Adapter residual add into UNet block epilogue

Source pattern: hidden state after a down block's final resnet+attention pair,
or non-cross block output, plus adapter feature.

Replacement: fuse adapter addition into the block output write or into the next
pointwise/norm input load when layout and dtype match.

Preconditions: adapter feature tensor already materialized and same shape/layout
as hidden state; addition occurs exactly at the source site, not ControlNet's
saved residual tuple site.

Failure cases: wrong residual ordering, ControlNet path, SDXL mid residual
condition, or NHWC/NCHW mismatch.

## 12. Kernel Fusion Candidates

Highest priority:

- Adapter feature extractor: PixelUnshuffle + first Conv2d, then small
  Conv/ReLU/Conv residual blocks. This is the only new model compute beyond the
  base SD1 pipeline and is loop-invariant.
- Adapter residual injection into UNet down blocks. Fuse add where it avoids a
  separate tensor pass and verify exact source sites.
- MultiAdapter scale/sum, after single-adapter parity.
- Base SD1 UNet/VAE fusions from the stable_diffusion_1_5 report: Conv2d +
  GroupNorm + SiLU, attention projections/attention, CFG/scheduler arithmetic.

Medium priority:

- Guarded NHWC adapter conv island, especially if the base UNet down path also
  moves to NHWC.
- PixelUnshuffle+Conv fusion once the plain PixelUnshuffle op is admitted.
- Adapter feature cache keyed by image, adapter weights, dtype, and size for
  repeated prompts over the same condition image.

Lower priority:

- SDXL `adapter_conditioning_factor` step gating and mid-block adapter addition.
- Specialized ceil-pooling dynamic-size variants outside normal SD sizes.

## 13. Runtime Staging Plan

Stage 1: Parse one SD1 base pipeline config and one `T2IAdapter` config. Admit
external prompt embeddings and preprocessed adapter image tensor.

Stage 2: Implement adapter-only parity for `PixelUnshuffle`, Conv2d, ReLU,
AvgPool2d ceil mode, residual add, FullAdapter and LightAdapter blocks.

Stage 3: Run adapter feature pyramid parity for canny/sketch grayscale,
depth/zoedepth RGB, and color light-adapter configs at 512x512.

Stage 4: Add UNet intrablock residual injection to the existing SD1 UNet
lowering. Validate feature slot order and exact add sites.

Stage 5: One denoising step parity with supplied prompt embeddings, latents,
timestep, adapter image, adapter features, CFG, and scheduler step.

Stage 6: Cache adapter features outside the denoising loop and run a short
deterministic loop with scheduler in Python.

Stage 7: Add VAE decode reuse from SD1.

Stage 8: Add MultiAdapter slot-wise scale/sum.

Stage 9: Consider NHWC adapter/UNet conv islands and PixelUnshuffle+Conv fusion.

Stage 10: Create a separate `sdxl_t2i_adapter` report before implementing SDXL
time IDs, dual text encoders, IP-Adapter interaction, adapter gating, or XL
mid-block residual behavior.

## 14. Parity And Validation Plan

- PixelUnshuffle parity for grayscale and RGB tensors, including invalid
  divisibility rejection.
- FullAdapter block parity for canny/sketch and depth/zoedepth configs.
- LightAdapter block parity for `t2iadapter_color_sd14v1`.
- Feature pyramid slot count, order, shape, dtype, and scaling parity.
- MultiAdapter parity with two tiny adapters and independent scales, including
  default equal-weight behavior.
- UNet down-block injection parity for each SD1 injection site:
  cross-attention block 0/1/2 and final non-cross down block.
- One full UNet forward parity with precomputed adapter features.
- One denoising step parity including CFG and selected scheduler.
- Short deterministic text-to-image smoke with scheduler in Python and VAE
  decode reused from SD1.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, tighten per kernel and source dtype.

## 15. Performance Probes

- Adapter feature extraction time by adapter type, input channels, resolution,
  batch, and dtype.
- Adapter cost amortized over denoising step count, proving it is computed once.
- UNet step time with and without adapter injection.
- Residual injection overhead per down-block site.
- MultiAdapter scaling with 2 and 3 adapters.
- NCHW faithful adapter path versus guarded NHWC path.
- PixelUnshuffle+Conv fused versus unfused.
- Full denoising loop by step count, separating adapter, UNet, scheduler/CFG,
  and VAE decode.
- VRAM/temporary usage for cached adapter features.

## 16. Scope Boundary And Separate Candidates

Separate candidate reports, not ignored:

- `sd1_t2i_adapter_multi`: MultiAdapter list inputs and feature-slot reduction.
- `sdxl_t2i_adapter`: `StableDiffusionXLAdapterPipeline`, `FullAdapterXL`,
  SDXL text/time conditioning, IP-Adapter hooks, `adapter_conditioning_factor`,
  and XL mid-block shape behavior.
- `sd1_controlnet`: already separate; different residual surface and side-model
  runtime cost.
- `sd1_ip_adapter`: image embedding and added K/V attention processors.
- `sd1_lora_textual_inversion_adapters`: text/weight mutation.
- `sd1_img2img`, `sd1_inpaint`, `sd1_depth2img`, `sd_upscale`, `sd1_gligen`:
  related SD pipelines with different image/latent conditioning contracts.
- Rare scheduler variants beyond the selected first scheduler slice.

Ignored/out of scope for this audit:

- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Callback mutation and interactive interrupt.
- Multi-GPU/context parallel paths.
- XLA, NPU, MPS, Flax, and ONNX-specific paths.

## 17. Final Implementation Checklist

- [ ] Parse SD1 base component configs and one `T2IAdapter` config.
- [ ] Admit adapter config defaults, especially omitted `downscale_factor=8`.
- [ ] Load adapter weights independently from base SD weights.
- [ ] Accept external prompt embeddings, latents, timestep, and adapter image tensor.
- [ ] Implement `PixelUnshuffle` or a guarded equivalent.
- [ ] Implement adapter Conv2d/ReLU/AvgPool2d ceil/residual blocks.
- [ ] Validate FullAdapter and LightAdapter feature pyramids.
- [ ] Add UNet `down_intrablock_additional_residuals` lowering at source add sites.
- [ ] Keep adapter feature extraction outside the denoising loop.
- [ ] Implement single-adapter feature scaling.
- [ ] Add one-step SD1 T2I-Adapter parity with CFG and scheduler.
- [ ] Add short loop smoke with VAE decode.
- [ ] Add MultiAdapter scale/sum after single-adapter parity.
- [ ] Add guarded NHWC adapter conv-island rewrite only after faithful NCHW parity.
- [ ] Write a separate SDXL T2I-Adapter report before implementing XL variants.
