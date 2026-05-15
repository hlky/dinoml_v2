# Diffusers CogView3/4 Operator and Integration Report

Candidate slug: `cogview3_4`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  THUDM/CogView3-Plus-3B and zai-org/CogView3-Plus-3B
  THUDM/CogView4-6B and zai-org/CogView4-6B
  THUDM/CogView4-6B-Control / zai-org/CogView4-6B-Control as source-only variant; official config blocked.

Config sources:
  H:/configs/zai-org/CogView3-Plus-3B/model_index.json
  H:/configs/zai-org/CogView4-6B/model_index.json
  H:/configs/zai-org/GLM-Image/model_index.json was checked only to avoid conflating GLM-Image with CogView4.
  Official raw HF component configs for CogView3-Plus-3B and CogView4-6B:
    model_index.json, transformer/config.json, vae/config.json,
    scheduler/scheduler_config.json, text_encoder/config.json,
    tokenizer/tokenizer_config.json.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/cogview3/pipeline_cogview3plus.py
  diffusers/src/diffusers/pipelines/cogview3/pipeline_output.py
  diffusers/src/diffusers/pipelines/cogview4/pipeline_cogview4.py
  diffusers/src/diffusers/pipelines/cogview4/pipeline_cogview4_control.py
  diffusers/src/diffusers/pipelines/cogview4/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_cogview3plus.py
  diffusers/src/diffusers/models/transformers/transformer_cogview4.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  diffusers/src/diffusers/models/autoencoders/vae.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddim_cogvideox.py
  diffusers/src/diffusers/schedulers/scheduling_dpm_cogvideox.py
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  T5EncoderModel/T5Tokenizer configs for CogView3Plus.
  GlmModel/PreTrainedTokenizerFast configs for CogView4.

Any missing files or assumptions:
  Local cache had only model_index.json for CogView3Plus/CogView4, so component facts are from official raw HF reads
  and were not saved because this task's write path is limited to this report. CogView4-Control returned 401 over raw
  HTTP and 404/RepositoryNotFound through huggingface_hub without an accepted token; source was inspected, but official
  control component configs remain blocked. This report targets text-to-image base pipelines first and inventories
  control/LoRA/GLM-Image separately. Backend/training/safety/callback/multi-device paths are out of scope.
```

## 2. Pipeline and component graph

CogView3Plus and CogView4 are latent text-to-image DiT-style pipelines with NCHW 2D latents, an AutoencoderKL image
codec, text encoders outside Diffusers, patch-token transformers, explicit CFG arithmetic, and VAE image postprocess.

```text
prompt / negative prompt
  -> tokenizer + text encoder
  -> prompt embeddings, optionally negative prompt embeddings
  -> latent noise [B,C,H/8,W/8]
  -> denoising loop:
       transformer patchify + text/image joint attention + unpatchify
       CFG arithmetic
       scheduler step
  -> AutoencoderKL decode
  -> VaeImageProcessor postprocess
```

CogView3Plus required components:

| Component | Class | Notes |
| --- | --- | --- |
| pipeline | `CogView3PlusPipeline` | No optional components; offload sequence `text_encoder->transformer->vae`. |
| tokenizer | `T5Tokenizer` | Pipeline default max sequence length is 224 in `encode_prompt`; lower helper default is 226. |
| text encoder | `T5EncoderModel` | Produces `[B,S,4096]` last hidden state. |
| denoiser | `CogView3PlusTransformer2DModel` | 2D patch transformer with 2D sin-cos image position add and CogVideoX attention processor. |
| scheduler | `CogVideoXDDIMScheduler` or `CogVideoXDPMScheduler` | Official config uses DDIM v-pred with SNR shift 4. |
| VAE | `AutoencoderKL` | 16-channel latent codec, scaling factor 1.0. |

CogView4 required components:

| Component | Class | Notes |
| --- | --- | --- |
| pipeline | `CogView4Pipeline` | Mixes in `CogView4LoraLoaderMixin`; offload sequence `text_encoder->transformer->vae`. |
| tokenizer | `AutoTokenizer` / `PreTrainedTokenizerFast` | Pads GLM input IDs on the left to a multiple of 16. |
| text encoder | `GlmModel` | Uses `hidden_states[-2]`, not last hidden state. |
| denoiser | `CogView4Transformer2DModel` | 2D patch transformer with image-token RoPE and cache contexts for cond/uncond calls. |
| scheduler | `FlowMatchEulerDiscreteScheduler` | Dynamic shifting with CogView4-specific base/max shift. |
| VAE | `AutoencoderKL` | Same broad 16-channel VAE shape as CogView3Plus; `shift_factor=0.0`. |

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `cogview4_control` | `CogView4ControlPipeline`, `CogView4Transformer2DModel` | VAE-encodes a control image, concatenates denoising latents and control latents on channel axis, likely 32-channel transformer input. Config is gated/unavailable here. |
| `cogview4_lora` | `CogView4LoraLoaderMixin`, `PeftAdapterMixin` on transformer | Transformer-only LoRA/PEFT adapter load, hotswap, fuse/unfuse state. |
| `glm_image` | `pipelines/glm_image`, `GlmImageTransformer2DModel` | Related branding but distinct pipeline/model: ByT5 tokenizer, GLM image VLM, T5 text encoder, and GLM-Image transformer. |
| `cogview3_dpm_scheduler` | `CogVideoXDPMScheduler` | Optional DPM path with caller-visible previous-original-sample state. |

No CogView3/4 source pipeline in this checkout provides img2img, inpaint, depth2img, upscaling, IP-Adapter, T2I-Adapter,
GLIGEN, or ControlNet-style residual injection for the base family. CogView4 control is a folder-level variant rather
than a generic ControlNet module.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Text encoder | Denoiser | Hidden | Patch | Latents | VAE | Scheduler |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| CogView3-Plus-3B | `CogView3PlusPipeline` | T5 XXL, `d_model=4096`, 24 layers | 30 layers, 64 heads, head dim 40 | 2560 | 2 | 16 NCHW channels, default 1024x1024 -> 128x128 | scale 8, scaling factor 1.0, no quant/post-quant conv | CogVideoX DDIM, v-pred, trailing, SNR shift 4 |
| CogView4-6B | `CogView4Pipeline` | GLM, hidden 4096, 40 layers, GQA 32 q / 2 kv heads | 28 layers, 32 heads, head dim 128 | 4096 | 2 | 16 NCHW channels, default 1024x1024 -> 128x128 | scale 8, scaling factor 1.0, shift 0.0, no quant/post-quant conv | FlowMatch Euler, dynamic linear shift |
| CogView4-Control | `CogView4ControlPipeline` | GLM by source annotation | source uses CogView4 transformer | unknown | 2 by source default | source expects `in_channels // 2` denoise + VAE control concat | source VAE encode/decode | FlowMatch Euler by source annotation; config blocked |

Transformer dimensions:

| Field | CogView3Plus | CogView4 |
| --- | ---: | ---: |
| `in_channels` / `out_channels` | 16 / 16 | 16 / 16 |
| `patch_size` | 2 | 2 |
| `num_layers` | 30 | 28 |
| `num_attention_heads` | 64 | 32 |
| `attention_head_dim` | 40 | 128 |
| inner dim | 2560 | 4096 |
| `text_embed_dim` | 4096 | 4096 |
| `time_embed_dim` | 512 | 512 |
| size/crop condition dim | 256 | 256 |
| image position | 2D sin-cos table, max 128x128 latent patches | RoPE with `rope_axes_dim=(256,256)` |

VAE config shared shape:

| Field | Value |
| --- | --- |
| class | `AutoencoderKL` |
| input/output channels | 3 / 3 |
| latent channels | 16 |
| block channels | `[128,512,1024,1024]` |
| layers per block | 3 |
| down/up block types | `DownEncoderBlock2D` / `UpDecoderBlock2D` |
| mid attention | false |
| norm groups | 32 |
| quant/post-quant conv | false |
| scaling factor | 1.0 |
| source layout | NCHW image and latent maps |

Scheduler details:

| Scheduler | Source/default config | First Dinoml slice |
| --- | --- | --- |
| CogVideoX DDIM | `prediction_type="v_prediction"`, `beta_schedule="scaled_linear"`, `rescale_betas_zero_snr=true`, `timestep_spacing="trailing"`, `clip_sample=false`, `set_alpha_to_one=true`, `snr_shift_scale=4.0` | First slice for CogView3Plus parity. |
| CogVideoX DPM | Accepted by CogView3Plus constructor and loop, but not official sampled default | Separate after DDIM. |
| FlowMatch Euler | `use_dynamic_shifting=true`, `time_shift_type="linear"`, base image seq 256, max image seq 4096, base shift 0.25, max shift 0.75, shift 1.0 | First slice for CogView4 parity. |

## 3a. Family variation traps

- CogView3Plus and CogView4 are not minor config variants: text encoder class, scheduler family, position encoding,
  attention processor, CFG strategy, and transformer width all differ.
- Both use NCHW latent maps in the pipeline and model. NHWC is only a guarded optimization inside local Conv2d/VAE or
  patchify regions.
- CogView3Plus batches CFG as `[uncond, cond]` in one transformer call. CogView4 performs separate cached `cond` and
  `uncond` transformer calls, so first parity must not assume CFG batch concatenation.
- CogView3Plus negative prompt defaults to zero prompt embeddings when CFG is on and `negative_prompt is None`.
  CogView4 defaults negative prompt text to `""` and encodes it through GLM.
- CogView4 tokenization pads the sequence length to a multiple of 16 before `GlmModel`; prompt length is not fixed to
  224/226.
- CogView4 uses `hidden_states[-2]` from GLM, not the final hidden state.
- CogView4 FlowMatch timesteps are generated in the pipeline as float timesteps/sigmas and shifted with `mu` based on
  image patch sequence length.
- CogView4 Control source sets denoise latent channels to `transformer.config.in_channels // 2` and concatenates VAE
  control latents with noisy latents. The control checkpoint likely changes `in_channels` to 32, but config is blocked.
- CogView3Plus uses 2D sin-cos position add from `CogView3PlusPatchEmbed`; CogView4 uses RoPE only on image token spans.
- CogView3Plus source supports fused attention processor mutation through inherited attention mixin patterns, but the
  official path uses `CogVideoXAttnProcessor2_0`.
- VAE configs disable quant/post-quant 1x1 convs and mid-block attention; do not import those AutoencoderKL branches into
  the first required op set for these checkpoints.

## 4. Runtime tensor contract

For 1024x1024 base generation:

| Boundary | CogView3Plus | CogView4 |
| --- | --- | --- |
| prompt IDs | `[B,224]` by pipeline default max sequence length | variable `S`, padded on left to multiple of 16, max 1024 |
| prompt embeddings | `[B,224,4096]` T5 last hidden state | `[B,S_pad,4096]` GLM hidden state `[-2]` |
| CFG prompt embeddings | concat to `[2B,224,4096]` | separate cond/uncond tensors, each `[B,S_pad,4096]` |
| latent noise | `[B,16,128,128]` NCHW | `[B,16,128,128]` NCHW float32 initially |
| patch tokens | `[B,4096,2560]` image tokens plus text tokens | `[B,4096,4096]` image tokens plus text tokens |
| denoiser output | `[B,16,128,128]` v-pred model output | `[B,16,128,128]` flow derivative |
| scheduler state | alpha-product tables and timesteps | sigma/timestep tables, step index, dynamic shift `mu` |
| VAE decode input | `latents / 1.0` NCHW | `latents.to(vae.dtype) / 1.0` NCHW |
| decoded image | `[B,3,1024,1024]` | `[B,3,1024,1024]` |

Patchify contracts:

- CogView3Plus/CogView4 source patchify is NCHW map -> view/reshape to `[B,H/p,W/p,C,p,p]` patch payload -> Linear.
- CogView3Plus `CogView3PlusPatchEmbed` concatenates projected text tokens before image tokens and adds zero text
  position plus 2D sin-cos image position.
- CogView4 `CogView4PatchEmbed` returns image tokens and text tokens separately; RoPE tables are generated from the
  original latent map shape and applied in attention to the image suffix after text/image concat.
- Unpatchify projects image tokens to `p*p*out_channels`, reshapes to `[B,H/p,W/p,C,p,p]`, permutes, and flattens back to
  NCHW.

CPU/data-pipeline work: tokenization, text encoder if not supplied, PIL/NumPy image postprocess. GPU/runtime work:
latent RNG, patch transformer, CFG arithmetic, scheduler step, VAE decode, and for CogView4-Control VAE encode of the
control image. Prompt embeddings, size/crop tensors, position tables, and scheduler tables are cacheable.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation, randn, scalar multiply by `init_noise_sigma`.
- View/reshape/permute/flatten for 2D patchify and unpatchify.
- Text/image token concat/split for joint attention.
- CogView3Plus CFG batch concat/chunk; CogView4 separate denoiser calls and pointwise CFG.
- Size/crop tensor construction and repetition: original size, target size, crop coords.
- CogView4 left padding of token IDs to sequence multiple of 16 is tokenizer-side but affects text sequence length.
- Control variant: NCHW control image preprocess, VAE encode, channel concat `[latents, control_latents]`.

Convolution/downsample/upsample ops:

- AutoencoderKL faithful source path: Conv2d encode/decode, ResNet blocks, GroupNorm, SiLU, spatial downsample/upsample.
- No quant/post-quant conv and no VAE mid attention for sampled configs.

GEMM/linear ops:

- Patch projection: CogView3Plus `Linear(16*2*2=64 -> 2560)`, CogView4 `Linear(64 -> 4096)`.
- Text projection: CogView3Plus/CogView4 `Linear(4096 -> inner_dim)`.
- Timestep and size/crop embeddings: sinusoidal projections plus MLP/text projection to 512.
- Attention Q/K/V and output projections, bias true.
- FeedForward GELU-approximate MLP in each transformer block.
- Final projection: CogView3Plus `Linear(2560 -> 64)`, CogView4 `Linear(4096 -> 64)`.

Attention primitives:

- Joint text-image noncausal SDPA over concatenated tokens.
- LayerNorm QK norm over head dim.
- CogView4 image-token RoPE; CogView3Plus no RoPE in the inspected model.
- Optional CogView4 text attention mask support exists in source but base pipeline does not pass an attention mask.

Normalization and adaptive conditioning:

- CogView3Plus `CogView3PlusAdaLayerNormZeroTextImage`: SiLU + Linear to 12 modulation/gate vectors.
- CogView4 `CogView4AdaLayerNormZero`: Linear to 12 modulation/gate vectors without the extra SiLU at this layer; the
  transformer applies SiLU to the shared time/size embedding before blocks.
- LayerNorm, final AdaLayerNormContinuous, CogView4 final custom AdaLN with no activation before the linear.
- VAE GroupNorm.

Scheduler and guidance arithmetic:

- CogView3Plus DDIM v-pred alpha-product conversion and CFG.
- CogView3Plus optional DPM state handoff.
- CogView4 FlowMatch Euler `prev = sample + (sigma_next - sigma) * model_output`.
- FlowMatch dynamic shift table generation from image sequence length.

## 6. Denoiser/model breakdown

CogView3Plus forward:

```text
hidden_states [B,16,H,W], encoder_hidden_states [B,S,4096]
-> CogView3PlusPatchEmbed:
     NCHW patch pack -> Linear(64,2560)
     text Linear(4096,2560)
     concat text+image and add 2D sin-cos image positions
-> CogView3CombinedTimestepSizeEmbeddings(timestep, original_size, target_size, crop_coords)
-> split text/image tokens
-> 30 x CogView3PlusTransformerBlock
-> final AdaLayerNormContinuous + Linear(2560,64)
-> unpatchify to [B,16,H,W]
```

CogView3Plus block:

```text
CogView3PlusAdaLayerNormZeroTextImage(image, text, emb)
-> joint attention with CogVideoXAttnProcessor2_0
-> gated residual to image and text streams
-> LayerNorm + scale/shift on each stream
-> concat text+image -> FeedForward(GELU approximate)
-> split and gated residual
```

CogView4 forward:

```text
hidden_states [B,16,H,W], encoder_hidden_states [B,S,4096]
-> CogView4RotaryPosEmbed from latent H/W
-> CogView4PatchEmbed:
     NCHW patch pack -> Linear(64,4096)
     text Linear(4096,4096)
-> CogView3CombinedTimestepSizeEmbeddings -> SiLU
-> 28 x CogView4TransformerBlock
-> CogView4AdaLayerNormContinuous + Linear(4096,64)
-> unpatchify to [B,16,H,W]
```

CogView4 block:

```text
CogView4AdaLayerNormZero(image, text, temb)
-> joint attention with CogView4AttnProcessor and image-token RoPE
-> gated residual to image and text streams
-> separate LayerNorm + scale/shift + shared FeedForward on image and text streams
-> gated residuals
```

## 7. Attention requirements

CogView3Plus uses `Attention` with `CogVideoXAttnProcessor2_0` from `attention_processor.py`.

- Joint text-image self-attention over `cat(text, image)` tokens.
- Heads/head dim: 64 x 40.
- Q/K/V projections are bias true; QK norm is LayerNorm with no elementwise affine.
- No RoPE in the base CogView3Plus path; positional information is added before blocks.
- Base pipeline passes no attention mask.
- Eager/native parity path is PyTorch `F.scaled_dot_product_attention`.

CogView4 uses local `CogView4AttnProcessor` in `transformer_cogview4.py`.

- Joint text-image self-attention over `cat(text, image)` tokens.
- Heads/head dim: 32 x 128.
- QK LayerNorm is applied after head reshape.
- 2D RoPE is applied to query/key image-token suffix only.
- Source supports a text attention mask by building a dense mixed token mask, but the base pipeline does not pass it.
- There is a `CogView4TrainingAttnProcessor` with varlen/sample-list behavior; it is training/out-of-scope for first
  inference parity.

Flash-style provider candidates are valid only under guards for no active mask, supported head dim, dtype, and sequence
length. CogView4 at 1024x1024 has 4096 image tokens plus a GLM text sequence padded to a multiple of 16; CogView3Plus has
4096 image tokens plus 224 text tokens. QK norm and RoPE remain explicit pre-attention work unless fused with a
provider that proves parity.

## 8. Scheduler and denoising-loop contract

CogView3Plus loop:

```text
timesteps = retrieve_timesteps(CogVideoXDDIMScheduler, num_steps)
latents = randn([B,16,H/8,W/8]) * scheduler.init_noise_sigma
prompt_embeds = cat([negative, positive]) if CFG
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = transformer(latent_model_input, prompt_embeds, t, size/crop cond)
  noise_pred = uncond + guidance * (text - uncond) if CFG
  latents = scheduler.step(noise_pred, t, latents)
decode(latents / vae.scaling_factor)
```

DDIM step for the official v-pred config:

```text
pred_original = sqrt(alpha_t) * sample - sqrt(1 - alpha_t) * model_output
prev = a_t * sample + b_t * pred_original
```

CogView4 loop:

```text
image_seq_len = (H/8 * W/8) / patch_size^2
timesteps = linspace(1000, 1, num_steps).astype(float32)
sigmas = timesteps / 1000
mu = calculate_shift(image_seq_len, base_image_seq_len=256, base_shift=0.25, max_shift=0.75)
retrieve_timesteps(FlowMatchEulerDiscreteScheduler, timesteps, sigmas, mu)
for t in timesteps:
  noise_pred_cond = transformer(latents, prompt_embeds, t, size/crop cond)
  noise_pred_uncond = transformer(latents, negative_prompt_embeds, t, size/crop cond) if CFG
  noise_pred = uncond + guidance * (cond - uncond)
  latents = scheduler.step(noise_pred, t, latents)
decode(latents / vae.scaling_factor)
```

FlowMatch Euler step is first-order and stateful through `step_index`; the non-stochastic branch is `sample + dt *
model_output`. Keep timestep/sigma table construction and step index host-visible initially; compile CFG and pointwise
scheduler update after table parity is fixed.

## 9. Position, timestep, and custom math

- `CogView3CombinedTimestepSizeEmbeddings` is shared by both models: sinusoidal timestep projection, sinusoidal
  original/crop/target size projection, TimestepEmbedding MLP, PixArt-style text projection, and sum.
- CogView3Plus uses a precomputed 2D sin-cos table sliced to `[H/p,W/p]`; text positions are zero.
- CogView4 RoPE uses independent H/W inverse frequencies, samples a `rope_axes_dim=(256,256)` grid down to the current
  patch grid, concatenates H and W frequencies, then returns cos/sin.
- CogView4 applies `F.silu` to the combined time/size embedding before transformer blocks.
- CogView3Plus block modulation applies `SiLU(emb)` inside `CogView3PlusAdaLayerNormZeroTextImage`; CogView4 block
  modulation does not apply another activation inside `CogView4AdaLayerNormZero`.
- Precompute candidates: prompt embeddings, negative prompt embeddings, size/crop condition embeddings for fixed shape,
  CogView3Plus 2D pos slice, CogView4 RoPE tables, and scheduler timesteps/sigmas.

## 10. Preprocessing and input packing

CogView3Plus:

- Tokenizes prompt with T5 tokenizer, max length 224 by `encode_prompt` default, padding to max length.
- Negative prompt embeddings default to zeros when CFG is active and no negative prompt is provided.
- Duplicates prompt embeddings by `num_images_per_prompt`.
- CFG concatenates negative then positive embeddings and duplicates latents in the batch axis.

CogView4:

- Tokenizes with GLM tokenizer using longest padding/truncation, max sequence length 1024.
- Pads token IDs on the left to a multiple of 16 before `GlmModel`.
- Uses `output_hidden_states=True` and selects `hidden_states[-2]`.
- Duplicates embeddings by `num_images_per_prompt`.
- CFG uses separate transformer calls under `cache_context("cond")` and `cache_context("uncond")`.

VAE and postprocess:

- Base text-to-image path only needs VAE decode; control variant additionally needs VAE encode of `control_image`.
- Latent decode boundary is NCHW `latents / scaling_factor`.
- `VaeImageProcessor.postprocess` handles denormalization and PIL/NumPy/latent output conversion.

## 11. Graph rewrite / lowering opportunities

### Rewrite: 2D latent patchify/unpatchify

Source pattern:

```text
NCHW -> reshape(B,C,H/p,p,W/p,p) -> permute/flatten to tokens -> Linear(C*p*p, inner)
Linear(inner, C*p*p) -> reshape(B,H/p,W/p,C,p,p) -> permute/flatten -> NCHW
```

Replacement: canonical `patchify2d` / `unpatchify2d` ops or im2col-plus-GEMM.

Preconditions: NCHW source layout, `patch_size=2`, H/W divisible by 2, in/out channels 16 for sampled configs, no
intervening view alias assumptions. NHWC lowering requires OIHW/HWIO-equivalent weight interpretation for VAE convs but
not for the Linear patch weights; it must preserve the source patch flatten order `C,p_h,p_w`. Failure cases: control
variant channel count mismatch and dynamic shapes not divisible by patch size.

### Rewrite: joint attention canonicalization

Source pattern:

```text
cat(text,image) -> QKV -> QK LayerNorm -> optional RoPE on image suffix -> SDPA -> output proj -> split
```

Replacement: joint-attention primitive with explicit text/image spans.

Preconditions: no active dense attention mask for first slice, supported head dim, known text span, RoPE table length
matches image token length. Failure cases: CogView4 attention mask path, training varlen processor, provider sequence
limit, LoRA-mutated projections not materialized.

### Rewrite: FlowMatch Euler step

Source pattern:

```text
dt = sigma_next - sigma
prev = sample + dt * model_output
```

Replacement: fused pointwise update.

Preconditions: CogView4 official non-stochastic config, scalar timestep branch, no per-token timesteps. Failure cases:
stochastic sampling, custom scheduler config with inverted sigmas, or per-token timestep branch.

### Rewrite: CogVideoX DDIM v-pred step

Source pattern:

```text
pred_original = sqrt(alpha) * sample - sqrt(beta) * model_output
prev = a * sample + b * pred_original
```

Replacement: fused pointwise update with precomputed scalar coefficients.

Preconditions: CogView3Plus official DDIM config, `eta=0`, no DPM scheduler. Failure cases: optional DPM scheduler or
non-v-pred config.

### Guarded layout rewrite: VAE decode channel-last island

Source pattern: NCHW AutoencoderKL decode with Conv2d/GroupNorm/SiLU/up blocks.

Replacement: NHWC conv island.

Preconditions: all consumers inside the island accept channel-last, GroupNorm axes are rewritten from channel axis 1 to
last channel, Conv2d weights transformed, and the island returns NCHW at the pipeline boundary. Failure cases: VAE
slicing/tiling, quant/post-quant conv changes, mid-attention variants, or mixed downstream layout.

## 12. Kernel fusion candidates

Highest priority:

- Large Linear/GEMM coverage for CogView4 4096-wide and CogView3Plus 2560-wide transformer projections.
- Joint QKV + QK LayerNorm + RoPE/SinCos-position-aware attention provider, with CogView4 head dim 128 and CogView3Plus
  head dim 40 both covered.
- AdaLN-zero modulation, gated residual, and FeedForward GELU-approximate epilogues.
- Patchify/unpatchify kernels with source patch flatten order parity.
- Scheduler pointwise updates and CFG arithmetic for DDIM v-pred and FlowMatch Euler.

Medium priority:

- AutoencoderKL decode Conv2d + GroupNorm + SiLU + residual fusion.
- VAE upsample/resnet decode throughput at 1024 output.
- CogView4 RoPE table generation/cache and application.
- CogView4 separate cond/uncond cache-context staging.
- Control-image VAE encode and channel concat for the control variant.

Lower priority:

- CogVideoX DPM scheduler state.
- LoRA hotswap/fuse/unfuse materialization.
- CogView4 dense attention mask path and training varlen processor.
- VAE tiling/slicing policies.

## 13. Runtime staging plan

1. Parse CogView3Plus and CogView4 component configs from local `model_index.json` plus official component configs.
2. Accept externally supplied prompt and negative prompt embeddings; defer T5/GLM text encoders.
3. Implement NCHW latent contract, size/crop condition tensors, and 2D patchify/unpatchify parity.
4. Bring up CogView3Plus one-block and full-transformer parity on reduced shapes, then 1024 shape if memory allows.
5. Implement CogVideoX DDIM v-pred scheduler and one-step CogView3Plus denoise parity.
6. Add AutoencoderKL decode for the CogView VAE config without tiling/slicing.
7. Bring up CogView4 RoPE, local attention processor, separate cond/uncond CFG calls, and FlowMatch Euler dynamic shift.
8. Add short deterministic end-to-end smoke with Python host loop and compiled denoiser blocks.
9. Add GLM/T5 prompt-embedding cache integration after denoiser/VAE parity.
10. Split CogView4-Control, CogView4 LoRA, GLM-Image, and optional DPM scheduler into separate candidates.

## 14. Parity and validation plan

- Config reconciliation tests for local `model_index.json` and official raw component configs.
- Prompt embedding shape tests: CogView3Plus T5 max length/zero negative embeds; CogView4 GLM left padding to multiple of
  16 and `hidden_states[-2]`.
- Patchify/unpatchify parity for `[B,16,128,128]` and smaller synthetic grids.
- CogView3Plus 2D sin-cos position slice parity.
- CogView4 RoPE table and image-suffix application parity.
- Single `CogView3PlusTransformerBlock` and `CogView4TransformerBlock` random tensor parity.
- Full transformer forward parity at reduced latent sizes.
- CFG parity: CogView3Plus batch concat/chunk and CogView4 separate calls.
- DDIM v-pred `set_timesteps` and one `step` parity for CogView3Plus.
- FlowMatch Euler dynamic-shift `set_timesteps` and one `step` parity for CogView4.
- AutoencoderKL decode parity for sampled VAE config.
- Suggested tolerances: fp32 scheduler/custom ops `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`;
  bf16/fp16 start at `rtol=2e-2, atol=2e-2` and tighten by provider.

## 15. Performance probes

- One denoiser step by resolution: 512, 1024, and any supported high-resolution shape.
- CogView3Plus vs CogView4 attention time split by text length and image token count.
- Patchify/unpatchify overhead versus attention/FFN.
- CogView4 cond/uncond separate call cost versus hypothetical CFG batch concat, without changing parity by default.
- FlowMatch scheduler and CFG overhead compared with denoiser time.
- VAE decode throughput and memory at 1024.
- Attention backend comparison for CogView3Plus head dim 40 and CogView4 head dim 128.
- VRAM/workspace for 4096 image tokens plus text sequence.
- Control variant VAE encode and doubled-channel transformer input once config is accessible.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `cogview4_control`: source-inspected control-image VAE encode and channel-concat transformer path; official config gated
  or unavailable in this run.
- `cogview4_lora`: `CogView4LoraLoaderMixin` and transformer `PeftAdapterMixin` adapter state.
- `glm_image`: separate GLM-Image pipeline/model family, despite nearby branding and cached model index.
- `cogview3_dpm_scheduler`: optional CogVideoX DPM scheduler path.
- `autoencoder_kl_cogview`: codec-specific optimization for the 16-channel no-quant/no-mid-attention VAE config.
- Rare scheduler swaps/custom timesteps and sigmas beyond the official configs.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker/NSFW filtering.
- Training processors, losses, dropout behavior, and gradient checkpointing.
- Backend-specific attention kernels except as provider candidates.

## 17. Final implementation checklist

- [ ] Parse CogView3Plus and CogView4 model/component configs.
- [ ] Load transformer and VAE weights for one CogView3Plus checkpoint.
- [ ] Accept external T5 prompt/negative prompt embeddings.
- [ ] Implement NCHW latent contract and 2D patchify/unpatchify.
- [ ] Implement size/crop/timestep conditioning embeddings.
- [ ] Implement CogView3Plus AdaLN-zero, joint attention, FFN, and final projection.
- [ ] Implement CogVideoX DDIM v-pred scheduler slice.
- [ ] Add one-step CogView3Plus denoise parity.
- [ ] Implement AutoencoderKL decode for CogView VAE config.
- [ ] Add CogView4 GLM prompt embedding cache contract.
- [ ] Implement CogView4 RoPE and local attention processor.
- [ ] Implement CogView4 FlowMatch Euler dynamic-shift scheduler slice.
- [ ] Add CogView4 separate cond/uncond CFG parity.
- [ ] Benchmark denoiser step, attention backend, patchify, scheduler, and VAE decode.
- [ ] Split CogView4-Control, CogView4 LoRA, GLM-Image, and optional DPM work into separate tickets.
