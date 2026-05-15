# Diffusers CogVideoX Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  zai-org/CogVideoX-2b
  zai-org/CogVideoX-5b
  zai-org/CogVideoX-5b-I2V
  zai-org/CogVideoX1.5-5B
  zai-org/CogVideoX1.5-5B-I2V

Config sources:
  H:/configs/zai-org/*/model_index.json for the five repos above.
  Official raw Hugging Face component configs for transformer, VAE, scheduler,
  text_encoder, and tokenizer were inspected over HTTP because the local cache
  only contained model_index.json files. They were not saved locally because this
  task's owned write path is limited to this report.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_video2video.py
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_fun_control.py
  diffusers/src/diffusers/pipelines/cogvideo/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/cogvideox_transformer_3d.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/downsampling.py
  diffusers/src/diffusers/models/upsampling.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_ddim_cogvideox.py
  diffusers/src/diffusers/schedulers/scheduling_dpm_cogvideox.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  T5EncoderModel and T5Tokenizer configs in official CogVideoX repos.

Any missing files or assumptions:
  No gated blocker was hit. Local component configs were missing but official
  raw configs were available. This report targets text-to-video first, with I2V,
  video2video, Fun-Control, LoRA, and scheduler swaps inventoried separately.
  Multi-GPU/context parallel, callbacks/interrupt, XLA/NPU/MPS/Flax/ONNX,
  safety/NSFW, and training/loss/dropout/gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

CogVideoX is a latent video diffusion family with a joint text/video transformer,
T5 text conditioning, CogVideoX-specific DDIM or DPM schedulers, and a causal
3D VAE.

```text
prompt / negative prompt
  -> T5Tokenizer + T5EncoderModel token embeddings [B,226,4096]
  -> latent video noise [B,T_lat,C,H/8,W/8]
  -> denoising loop:
       optional CFG batch concat
       CogVideoXTransformer3DModel(video latents, T5 embeds, timestep, RoPE)
       CogVideoXDDIMScheduler or CogVideoXDPMScheduler step
  -> drop temporal padding for 1.5 when added
  -> permute latents to [B,C,T,H,W], divide by VAE scaling factor
  -> AutoencoderKLCogVideoX decode
  -> VideoProcessor postprocess
```

Image-to-video adds:

```text
input image
  -> VideoProcessor.preprocess to NCHW
  -> unsqueeze temporal frame
  -> AutoencoderKLCogVideoX encode
  -> scale or inverse-scale image latent, pad future frames with zeros
  -> concatenate noisy latents and image latents along latent channel axis
  -> same denoising loop and decode path
```

Required first-slice components:

- `CogVideoXPipeline`, `CogVideoXTransformer3DModel`,
  `AutoencoderKLCogVideoX`, `CogVideoXDDIMScheduler`.
- External or cached T5 prompt and negative prompt embeddings first.
- VAE decode for T2V output; VAE encode only for I2V/video2video variants.

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `cogvideox_i2v` | `CogVideoXImageToVideoPipeline` | Adds image preprocessing, VAE encode of first frame, latent padding, channel concat, and for 1.5 I2V an `ofs` conditioning embedding. |
| `cogvideox_video2video` | `CogVideoXVideoToVideoPipeline` | Adds source video preprocess, VAE encode, scheduler `add_noise`, and strength-based timestep slicing. |
| `cogvideox_fun_control` | `CogVideoXFunControlPipeline` | Adds control video/image latent branch and requires latent frame count divisible by `patch_size_t` for 1.5. |
| `cogvideox_lora_adapters` | `CogVideoXLoraLoaderMixin`, PEFT mixin on transformer | Runtime/load-time adapter mutation for transformer and possibly text encoder weights. |
| `cogvideox_dpm_scheduler` | `CogVideoXDPMScheduler` | Stochastic DPM-Solver++-style step with `old_pred_original_sample` state and generated noise. |
| `cogvideox_vae_tiling_slicing` | `AutoencoderKLCogVideoX` tiling/slicing | Memory-policy report for causal-conv cache, temporal chunk decode, and tile blending. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Transformer | Patch | Frames | Latents | VAE scale | Scheduler | Special |
| --- | --- | --- | --- | ---: | --- | ---: | --- | --- |
| CogVideoX-2b | `CogVideoXPipeline` | 30 layers, 30 heads, head 64, inner 1920 | spatial 2, no temporal patch | 49 sample -> 13 latent | 16 -> 16 | 1.15258426 | DDIM, v-pred, SNR shift 3 | no RoPE; 3D sin-cos positional embedding |
| CogVideoX-5b | `CogVideoXPipeline` | 42 layers, 48 heads, head 64, inner 3072 | spatial 2, no temporal patch | 49 -> 13 | 16 -> 16 | 0.7 | DDIM, v-pred, SNR shift 1 | 3D RoPE |
| CogVideoX-5b-I2V | `CogVideoXImageToVideoPipeline` | same 5B width/depth | spatial 2, no temporal patch | 49 -> 13 | 32 in, 16 out | 0.7 | DDIM, v-pred | learned positional embeddings; noisy+image latent concat |
| CogVideoX1.5-5B | `CogVideoXPipeline` | same 5B width/depth | spatial 2, temporal 2 | 81 -> 21 latent, padded to even if needed | 16 -> 16 | 0.7 inverse decode/encode quirk | DDIM, v-pred | RoPE slice grid; no patch bias |
| CogVideoX1.5-5B-I2V | `CogVideoXImageToVideoPipeline` | same 5B width/depth | spatial 2, temporal 2 | 81 -> 21 latent, padded to even if needed | 32 in, 16 out | 0.7 inverse decode/encode quirk | DDIM, v-pred | `ofs_embed_dim=512`, `ofs=2.0` |

Common text encoder:

| Component | Config facts |
| --- | --- |
| `T5Tokenizer` | `model_max_length=226`, pad/eos/unk plus 100 extra IDs. |
| `T5EncoderModel` | `d_model=4096`, 24 layers, 64 heads, `d_ff=10240`, gated GELU, vocab 32128. |

Common VAE:

| Field | Value |
| --- | --- |
| `latent_channels` | 16 |
| `block_out_channels` | `[128,256,256,512]` |
| `layers_per_block` | 3 |
| `temporal_compression_ratio` | 4 |
| spatial scale | 8, inferred from four block channel levels |
| quant/post-quant conv | false in sampled official configs |
| tiling defaults | sample half-resolution tiles, overlap height 1/6 and width 1/5 |

Recommended first Dinoml scheduler slice:

- Start with official `CogVideoXDDIMScheduler` configs:
  `prediction_type="v_prediction"`, `beta_schedule="scaled_linear"`,
  `rescale_betas_zero_snr=true`, `timestep_spacing="trailing"`,
  `clip_sample=false`, `set_alpha_to_one=true`.
- Use `CogVideoX-2b` first if smaller width matters, but `CogVideoX-5b` is the
  better first RoPE path because it exercises the modern attention geometry.
- Defer `CogVideoXDPMScheduler` until DDIM one-step and loop parity are stable.

## 3a. Family variation traps

- Source denoiser layout is `[B,T,C,H,W]`, but VAE source layout is `[B,C,T,H,W]`.
  Do not silently treat the family as a single NCDHW contract.
- `sample_frames` in transformer configs is pre-VAE frame count. Latent frames
  are `(num_frames - 1) // 4 + 1`.
- CogVideoX 1.5 uses `patch_size_t=2`, so the transformer tokenization patches
  time as well as H/W. The pipeline may pad latent frames and then discard added
  frames before decode.
- CogVideoX 1.0 uses per-frame Conv2d patch embedding; CogVideoX 1.5 uses
  explicit temporal-spatial patch reshape followed by Linear.
- 2B has no rotary positional embeddings and relies on 3D sin-cos position add;
  5B and 1.5 use RoPE.
- I2V doubles transformer input channels to 32 by concatenating noisy latents
  and image latents along channel axis 2 in `[B,T,C,H,W]`.
- CogVideoX1.5 VAE configs set `invert_scale_latents=true`; I2V encode uses
  inverse scale, while decode in the shared pipeline still divides by
  `vae_scaling_factor_image`.
- Learned positional embeddings in `CogVideoX-5b-I2V` reject non-default latent
  height/width in `CogVideoXPatchEmbed`.
- `use_dynamic_cfg` changes the guidance scale per step with a cosine formula;
  it is loop-side CFG arithmetic, not a model input embedding except for the
  separate 1.5 I2V `ofs` embedding.
- `FunControl` and video2video reject 1.5 latent frame counts not divisible by
  `patch_size_t`; base T2V/I2V instead pad and crop.

## 4. Runtime tensor contract

For 720x480, 49-frame CogVideoX-5b T2V:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,226,4096]` | T5 last hidden state, duplicated per video. |
| CFG embeds | concat | `[2B,226,4096]` | Negative then positive when guidance scale > 1. |
| latents | noise/sample | `[B,13,16,60,90]` | Source denoiser layout BTCHW. |
| model input | after CFG + scale | `[2B,13,16,60,90]` | `scale_model_input` is identity for CogVideoX schedulers. |
| patch tokens | transformer image tokens | `[B,13*30*45,inner]` | 17,550 image tokens at 720x480 with spatial patch 2. |
| joint tokens | text + image | `[B,226+17550,inner]` | Attention processor attends over both. |
| noise pred | transformer output | `[B,13,16,60,90]` | BTCHW. |
| scheduler state | alphas/timesteps | scalar tables + optional old pred | DDIM is first-order; DPM keeps previous predicted original sample. |
| VAE decode input | permuted/scaled | `[B,16,13,60,90]` | `latents.permute(0,2,1,3,4) / scaling_factor`. |
| decoded video | sample | `[B,3,49,480,720]` | Postprocessed to PIL/NumPy/list output. |

For 1360x768, 81-frame CogVideoX1.5-5B:

- Latent grid is `[B,21,16,96,170]` before padding.
- Because `patch_size_t=2`, base pipeline pads to 22 latent frames for the
  transformer, token length is `(22/2)*(96/2)*(170/2)=44,880`.
- Output latents are cropped back by `additional_frames` before VAE decode.

I2V conditioning:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| input image | preprocessed | `[B,3,H,W]` | From `VideoProcessor.preprocess`. |
| VAE encode input | image frame | `[B,3,1,H,W]` | Unsqueezed temporal dim. |
| image latents | after encode | `[B,1,16,H/8,W/8]` | Permuted to BTCHW and scaled. |
| padded image latents | condition | `[B,T_lat,16,H/8,W/8]` | First frame plus zeros. |
| denoiser input | concat | `[B,T_lat,32,H/8,W/8]` | Noisy latents plus image latents along channel axis. |

CPU/data-pipeline work includes tokenization, T5 execution when prompt embeds
are not supplied, PIL/video preprocessing, and output conversion. GPU/runtime
work includes latent noise, VAE encode/decode, transformer denoise, CFG, and
scheduler pointwise updates.

## 5. Operator coverage checklist

### Tensor/layout ops

- BTCHW to BCTHW permutes between transformer and VAE.
- View/reshape/permute/flatten for spatial and temporal patchify/unpatchify.
- Text/video token concat and split.
- CFG batch concat/chunk.
- I2V channel concat in BTCHW.
- Temporal padding/cropping for 1.5.
- Per-channel and scalar latent scaling.

### Convolution/downsample/upsample ops

- CogVideoX 1.0 transformer patch embed: Conv2d over flattened `B*T`,
  `in_channels -> inner_dim`, kernel/stride 2.
- CogVideoX VAE: causal Conv3d, safe chunked Conv3d, Conv2d spatial
  downsample/upsample after flattening frames.
- Temporal downsample uses AvgPool1d over frames with special first-frame
  handling for odd frame counts.
- Temporal upsample uses interpolate with special first-frame handling.

### GEMM/linear ops

- T5 external encoder if admitted later.
- Text projection: Linear(4096 -> inner_dim).
- Time embedding: sinusoidal Timesteps -> TimestepEmbedding MLP.
- 1.5 I2V `ofs` embedding MLP.
- Joint attention Q/K/V, optional fused QKV, output projection.
- FeedForward GELU-approximate MLP.
- Final projection: Linear(inner_dim -> patch volume * out_channels).

### Attention primitives

- Joint text+video self-attention over `[text_seq + video_seq]`.
- QK LayerNorm in attention heads when `qk_norm=true`.
- Optional 3D RoPE on image/video token span only.
- No base attention mask in the main pipeline.

### Normalization and adaptive conditioning

- `CogVideoXLayerNormZero`: LayerNorm plus timestep-conditioned shift, scale,
  and gates for video and text streams.
- `AdaLayerNorm` final modulation from timestep embedding.
- VAE GroupNorm and SpatialNorm3D conditioned by latent sample in decoder.

### Scheduler and guidance arithmetic

- True CFG batch concat and `uncond + scale * (text - uncond)`.
- Dynamic CFG cosine step schedule when enabled.
- CogVideoX DDIM v-prediction conversion and alpha table lookup.
- DPM variant: stochastic noise plus previous predicted-original state.

### Video-specific ops

- Temporal compression ratio 4.
- 1.5 temporal patch size 2.
- Causal Conv3d cache/chunking in VAE encode/decode.
- Video postprocess from BCTHW tensor.

## 6. Denoiser/model breakdown

`CogVideoXTransformer3DModel.forward`:

```text
hidden_states [B,T,C,H,W]
-> timestep embedding, plus optional ofs embedding
-> CogVideoXPatchEmbed(text_embeds, video latents)
   - text Linear 4096->inner
   - 1.0 video Conv2d per frame or 1.5 temporal-spatial patch Linear
   - concat text tokens then video tokens
   - add 3D sin-cos/learned position embedding when RoPE disabled or learned enabled
-> split back to encoder_hidden_states text and hidden_states video
-> N x CogVideoXBlock
-> final LayerNorm
-> AdaLayerNorm from timestep embedding
-> Linear to patch volume
-> unpatchify to [B,T,C,H,W]
```

`CogVideoXBlock`:

```text
CogVideoXLayerNormZero(video, text, temb)
-> joint attention over cat(text, video)
-> gated residual add to both text and video streams
CogVideoXLayerNormZero(video, text, temb)
-> concat text+video -> FeedForward
-> split FF output and gated residual add to both streams
```

Shape examples:

- 2B inner dim is `30 * 64 = 1920`, depth 30.
- 5B and 1.5 inner dim is `48 * 64 = 3072`, depth 42.
- 1.0 output projection is `2 * 2 * out_channels = 64`.
- 1.5 output projection is `2 * 2 * 2 * out_channels = 128`.

## 7. Attention requirements

Primary implementation is `CogVideoXAttnProcessor2_0` in
`attention_processor.py`, used by `Attention` modules created in
`cogvideox_transformer_3d.py`. The fused mutation path switches to
`FusedCogVideoXAttnProcessor2_0`.

- Attention is noncausal SDPA over concatenated text and video tokens.
- Query/key/value shapes become `[B,heads,seq,head_dim]`.
- Heads/head dim: 30x64 for 2B, 48x64 for 5B/1.5.
- QK norm is LayerNorm over head dim when `qk_norm=true`.
- RoPE applies only to the video token suffix: `query[:, :, text_seq_length:]`
  and same key suffix when not cross-attention.
- Base path has no mask; mask handling exists in processor but is inactive for
  the main pipeline.
- Fused QKV is source-supported via `pipeline.fuse_qkv_projections()` and
  `transformer.fuse_qkv_projections()`, but not required by default.

Flash-style constraints:

- Base CogVideoX attention is a plausible Dinoml flash-style provider candidate
  when sequence length, dtype, head dim 64, and memory fit.
- QK LayerNorm and RoPE must remain explicit pre-attention operations.
- Text/video sequence concat and split must be represented so the provider
  output can update both streams.
- 720x480 5B already has about 17,776 joint tokens; 1.5 at 1360x768 has about
  45,106 joint tokens. Provider admission needs strong sequence-length and
  workspace guards.
- Eager PyTorch SDPA path defines parity.

## 8. Scheduler and denoising-loop contract

Official sampled configs use `CogVideoXDDIMScheduler` by default:

```text
num_train_timesteps = 1000
beta_start = 0.00085
beta_end = 0.012
beta_schedule = scaled_linear
prediction_type = v_prediction
rescale_betas_zero_snr = true
timestep_spacing = trailing
clip_sample = false
set_alpha_to_one = true
snr_shift_scale = 3.0 for 2B, 1.0 for 5B/1.5
```

DDIM loop:

```text
timesteps = scheduler.set_timesteps(num_inference_steps)
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = transformer(latent_model_input, prompt_embeds, t)
  if CFG:
    noise_pred = uncond + guidance * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

DDIM step converts v-prediction to predicted original sample:

```text
pred_original = sqrt(alpha_t) * sample - sqrt(1 - alpha_t) * model_output
prev = a_t * sample + b_t * pred_original
```

`CogVideoXDPMScheduler` is supported by pipeline source and changes the loop:
it carries `old_pred_original_sample`, receives `timestep_back`, computes log-SNR
variables, and adds random noise inside the step. Treat it as a second scheduler
candidate after DDIM.

First Dinoml slice should keep timestep iteration and scheduler state on the
host, compile one transformer call, CFG arithmetic, and DDIM pointwise update as
explicit graph/runtime pieces.

## 9. Position, timestep, and custom math

- 2B uses 3D sin-cos positional embeddings from `CogVideoXPatchEmbed`.
- 5B/1.5 use `get_3d_rotary_pos_embed` from the pipeline.
- CogVideoX 1.0 RoPE uses resize/crop coordinates to map arbitrary generation
  grid to base `sample_width/patch_size` and `sample_height/patch_size`.
- CogVideoX 1.5 RoPE uses `grid_type="slice"` and temporal size
  `(latent_frames + patch_size_t - 1) // patch_size_t`.
- Timesteps use Diffusers `Timesteps(inner_dim, flip_sin_to_cos=True,
  freq_shift=0)` and `TimestepEmbedding`.
- `CogVideoXLayerNormZero` produces shift, scale, gate for video and text from
  `SiLU(temb) -> Linear(6*inner)`.
- Dynamic CFG scale is:
  `1 + guidance_scale * (1 - cos(pi * ((num_steps - t) / num_steps) ** 5)) / 2`.

Precompute candidates:

- Prompt embeddings and negative prompt embeddings.
- 3D sin-cos position embeddings for fixed 2B shape.
- RoPE cos/sin tables for fixed height/width/frame count.
- Scheduler alpha/timestep tables.

## 10. Preprocessing and input packing

Text preprocessing:

- Tokenize with `T5Tokenizer`, max length 226, padding to max length,
  truncation, special tokens.
- T5 encoder output `[B,226,4096]`.
- Duplicate embeddings for videos per prompt, though pipeline forces
  `num_videos_per_prompt=1`.
- Negative prompt defaults to empty string when CFG is active.

Video/image preprocessing:

- T2V starts from random latent noise in BTCHW.
- I2V preprocesses image to NCHW, unsqueezes temporal dimension, encodes through
  VAE, permutes to BTCHW, scales, pads zeros for remaining latent frames, and
  concatenates with noisy latents.
- Video2video preprocesses source video to BCTHW, VAE-encodes it, scales, adds
  noise at sliced timesteps based on `strength`.
- Decode path always permutes BTCHW latents to BCTHW before VAE decode.

NHWC/NDHWC guarded notes:

- Source semantics should stay BTCHW for transformer and BCTHW for VAE.
- NDHWC optimization is plausible inside VAE Conv3d islands only with explicit
  rewrites for channel axis, GroupNorm/SpatialNorm axes, causal temporal pads,
  Conv3d/Conv2d weights, and cache tensors.
- Transformer token core is layout-neutral after patchify; patchify/unpatchify
  and I2V channel concat need no-layout-translation guards until rewritten.

## 11. Graph rewrite / lowering opportunities

### Rewrite: CogVideoX 1.0 patchify/unpatchify

Source pattern:

```text
BTCHW -> reshape(B*T,C,H,W) -> Conv2d(k=s=2)
-> view(B,T,inner,H/2,W/2) -> flatten spatial/time to tokens
Linear(inner -> C*2*2) -> reshape/permute/flatten back to BTCHW
```

Replacement: explicit video-token patchify/unpatchify ops.

Preconditions: `patch_size=2`, `patch_size_t=None`, BTCHW source layout, H/W
divisible by 2. Weight transform is Conv2d weight as-is for source layout, or
OIHW->HWIO only under guarded NHWC provider lowering. Failure cases: 1.5
temporal patching and learned-position resolution restrictions.

### Rewrite: CogVideoX 1.5 temporal-spatial patchify

Source pattern:

```text
BTCHW -> BTHWC -> reshape(B,T/pt,pt,H/p,p,W/p,p,C)
-> permute(B,T/pt,H/p,W/p,C,pt,p,p) -> flatten patch -> Linear
```

Replacement: temporal-spatial patch pack + GEMM.

Preconditions: latent frames divisible by `patch_size_t=2` after pipeline pad,
H/W divisible by 2, channel count fixed. Failure cases: Fun-Control/video2video
paths that reject instead of pad.

### Rewrite: Joint attention canonicalization

Source pattern:

```text
concat(text, video) -> QKV -> QK LayerNorm -> RoPE on video suffix -> SDPA
-> output projection -> split(text, video)
```

Replacement: canonical joint attention provider with explicit text/video spans.

Preconditions: no active mask, supported dtype/head_dim/sequence, RoPE table
matches video token count. Failure cases: provider sequence limit, fused QKV
mutation mismatch, future added-KV branches.

### Rewrite: DDIM v-pred step

Source pattern:

```text
pred_original = sqrt(alpha) * sample - sqrt(beta) * model_output
prev = a * sample + b * pred_original
```

Replacement: pointwise scheduler kernel reading precomputed scalar tables.

Preconditions: official DDIM config, no DPM scheduler, eta unused. Failure
cases: alternate scheduler class or custom scheduler config.

## 12. Kernel fusion candidates

Highest priority:

- Large GEMM/Linear coverage for 5B attention QKV, FFN, text projection, and
  final projection.
- Joint QKV + QK LayerNorm + RoPE + attention provider with sequence guards.
- `CogVideoXLayerNormZero` modulation + gated residual epilogues.
- Patchify/unpatchify kernels, especially 1.5 temporal-spatial packing.
- DDIM v-pred scheduler update and CFG arithmetic.

Medium priority:

- VAE causal Conv3d + GroupNorm/SpatialNorm + SiLU + residual blocks.
- VAE temporal down/up sample special first-frame paths.
- I2V latent condition pack and channel concat.
- RoPE table generation/caching.

Lower priority:

- DPM stochastic scheduler state and noise generation.
- VAE tiling blend kernels.
- LoRA fuse/unfuse/runtime adapter state.
- Fun-Control control latent branch.

## 13. Runtime staging plan

Stage 1: Parse configs for `zai-org/CogVideoX-5b` and accept external T5
prompt/negative prompt embeddings.

Stage 2: Implement BTCHW latent shape contract, RoPE generation, spatial
Conv2d patchify/unpatchify, and one `CogVideoXBlock` parity.

Stage 3: Full `CogVideoXTransformer3DModel` forward parity for 5B random tensors
at a reduced latent grid, then official 720x480x49 shape if memory allows.

Stage 4: Add true CFG two-batch arithmetic and one fixed-timestep denoising
parity.

Stage 5: Implement `CogVideoXDDIMScheduler` official v-pred trailing timestep
slice.

Stage 6: Add `AutoencoderKLCogVideoX` decode without tiling/slicing, preserving
BCTHW layout and causal conv behavior.

Stage 7: Short deterministic T2V loop with scheduler in host control and VAE
decode.

Stage 8: Add CogVideoX1.5 temporal patching, temporal padding/cropping, and
inverse-scale VAE quirk.

Stage 9: Add I2V VAE encode/condition concat and 1.5 I2V `ofs` embedding.

Stage 10: Separate DPM scheduler, video2video, Fun-Control, LoRA, and VAE
tiling/chunking work.

## 14. Parity and validation plan

- Config/default reconciliation tests for local model_index plus fetched
  component defaults.
- T5 prompt embedding duplication and CFG concat parity with supplied embeddings.
- 1.0 patchify/unpatchify parity for `[B,13,16,60,90]`.
- 1.5 temporal patchify/unpatchify parity for padded `[B,22,16,96,170]`.
- RoPE table parity for CogVideoX-5b and CogVideoX1.5-5B shapes.
- One `CogVideoXBlock` parity for 2B and 5B widths.
- Full transformer forward parity with fixed random tensors.
- DDIM `set_timesteps` and one `step` parity for 2B SNR shift 3 and 5B SNR
  shift 1.
- CFG and dynamic CFG arithmetic parity.
- VAE decode parity for `[B,16,13,60,90] -> [B,3,49,480,720]`.
- I2V condition encode/scale/pad/concat parity.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per provider.

## 15. Performance probes

- One transformer step by latent grid: small synthetic, 720x480x49, and
  1360x768x81.
- Attention backend comparison by joint sequence length and dtype.
- 2B vs 5B block time split: QKV/attention/FFN/adaptive norm.
- Patchify/unpatchify overhead for 1.0 and 1.5.
- CFG one vs two batch cost and memory.
- DDIM scheduler overhead compared with denoiser time.
- VAE decode throughput and memory with no tiling, temporal chunking, and tiling.
- I2V VAE encode and condition-pack overhead.
- Guarded NDHWC VAE Conv3d layout candidate versus faithful BCTHW.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `cogvideox_i2v`: image VAE encode, condition padding, 32-channel transformer,
  and 1.5 `ofs` conditioning.
- `cogvideox_video2video`: source video VAE encode, `strength`, timestep slice,
  scheduler `add_noise`.
- `cogvideox_fun_control`: control video/image branch and 1.5 frame divisibility
  contract.
- `cogvideox_lora_adapters`: `CogVideoXLoraLoaderMixin` and PEFT adapter state.
- `cogvideox_dpm_scheduler`: stochastic DPM step with previous prediction state.
- `cogvideox_vae_tiling_slicing`: tiled encode/decode, blending, slicing, and
  temporal cache residency.
- `cogvideox_2b_no_rope`: smaller model but distinct position-embedding path.

Ignored/out of scope for this audit:

- Multi-GPU/context-parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse CogVideoX model index and component configs from `H:/configs` plus official component configs.
- [ ] Load `CogVideoXTransformer3DModel` weights for `zai-org/CogVideoX-5b`.
- [ ] Accept external T5 prompt and negative prompt embeddings.
- [ ] Implement BTCHW denoiser latent contract and BCTHW VAE boundary.
- [ ] Implement CogVideoX 1.0 spatial patchify/unpatchify.
- [ ] Implement 3D RoPE generation and application to video token suffix.
- [ ] Implement `CogVideoXLayerNormZero`, joint attention, FFN, and gated residuals.
- [ ] Implement full transformer forward parity for 5B.
- [ ] Implement true CFG and dynamic CFG arithmetic.
- [ ] Implement official `CogVideoXDDIMScheduler` v-pred trailing timestep slice.
- [ ] Implement `AutoencoderKLCogVideoX` decode without tiling/slicing.
- [ ] Add one-step denoising parity and short-loop smoke.
- [ ] Add CogVideoX1.5 temporal patching and frame padding/cropping.
- [ ] Add I2V VAE encode/condition concat and `ofs` embedding.
- [ ] Add DPM scheduler, video2video, Fun-Control, LoRA, and VAE tiling as separate candidates.
- [ ] Add guarded NDHWC/VAE layout optimization tests after faithful source-layout parity.
