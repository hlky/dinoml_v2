# Diffusers LTX Video Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Lightricks/LTX-Video
  Lightricks/LTX-Video-0.9.1
  Lightricks/LTX-Video-0.9.5
  Lightricks/LTX-Video-0.9.7-dev
  Lightricks/LTX-Video-0.9.7-distilled
  Lightricks/LTX-Video-0.9.8-13B-distilled
  Lightricks/ltxv-spatial-upscaler-0.9.7
  Lightricks/LTX-2, inspected only to separate LTX2 scope.

Config sources:
  H:/configs/Lightricks/*/model_index.json for the repos above.
  Official Hugging Face raw component configs for transformer, VAE, scheduler,
  text_encoder, tokenizer, and latent_upsampler were inspected in-memory where
  available. They were not saved because this task's owned write path is only
  this report.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx_i2v_long_multi_prompt.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/modeling_latent_upsampler.py
  X:/H/diffusers/src/diffusers/pipelines/ltx/pipeline_output.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_ltx.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_ltx2.py
    (listed only to split LTX2 scope)
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py
    (listed only to split LTX2 scope)
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ltx_euler_ancestral_rf.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  T5EncoderModel and T5TokenizerFast configs from official LTX-Video repos.

Missing files or assumptions:
  Local config cache mostly contained only model_index.json files. Official raw
  component configs were accessible; no gated/authenticated blocker was hit.
  The selected target is the LTX 0.9.x video family. LTX2, audio, latent
  upscaling, ICLoRA/control-style condition adapters, and long multi-prompt
  variants are inventoried as separate candidates. Multi-GPU/context parallel,
  callbacks/interrupt, XLA/NPU/MPS/Flax/ONNX, safety/NSFW, and
  training/loss/dropout/gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

LTX 0.9.x is a latent video diffusion family with packed latent tokens, a
T5-XXL text encoder, `LTXVideoTransformer3DModel`, FlowMatch Euler scheduling,
and `AutoencoderKLLTXVideo`. The base T2V path keeps the transformer denoiser in
token space `[B,S,D]`, then unpacks to `[B,C,T,H,W]` only for VAE decode.

```text
prompt / negative prompt
  -> T5TokenizerFast + T5EncoderModel -> prompt embeds [B,L,4096] + mask
  -> latent noise [B,128,T_lat,H/32,W/32] NCTHW
  -> pack latents -> [B,T_lat*H_lat*W_lat,128]
  -> denoising loop:
       CFG batch concat
       LTXVideoTransformer3DModel(latent tokens, T5 embeds, timestep, mask, RoPE)
       optional guidance rescale
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latents -> [B,128,T_lat,H/32,W/32]
  -> denormalize with VAE mean/std/scaling
  -> optional decode-time noise and VAE timestep conditioning
  -> AutoencoderKLLTXVideo decode
  -> VideoProcessor postprocess
```

Image-to-video adds VAE encode of the first frame, repeats the encoded latent
condition across latent frames, creates a packed conditioning mask, and denoises
only frames after the first latent frame through unpack/step/repack in each
scheduler step.

`LTXConditionPipeline` generalizes this to image/video conditions at arbitrary
frame indices. It builds packed `video_coords`, optional extra condition tokens,
conditioning masks, per-token timesteps, and calls
`scheduler.step(-noise_pred, ..., per_token_timesteps=timestep)`.

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Base T2V pipeline | `LTXPipeline`, `pipeline_ltx.py` | Use as the first runtime contract. |
| Denoiser | `LTXVideoTransformer3DModel`, `transformer_ltx.py` | Required; operates on packed latent tokens. |
| VAE | `AutoencoderKLLTXVideo`, `autoencoder_kl_ltx.py` | Decode required for output; encode required for I2V/conditions. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Required first scheduler; official configs use static or dynamic shift. |
| Text encoder | `T5EncoderModel`, `T5TokenizerFast` | Accept external prompt embeddings first. |

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `ltx_image2video` | `LTXImageToVideoPipeline` | Adds image preprocessing, VAE encode of first frame, packed conditioning mask, masked timestep input, and per-step unpack/repack to preserve first latent frame. |
| `ltx_condition` | `LTXConditionPipeline`, `LTXVideoCondition` | Adds arbitrary image/video conditions, extra condition tokens, `video_coords`, per-token timesteps, hard-condition masks, and optional image-condition noise. |
| `ltx_i2v_long_multi_prompt` | `pipeline_ltx_i2v_long_multi_prompt.py` | Long-video/multi-prompt orchestration on top of condition pipeline. |
| `ltx_latent_upsampler` | `LTXLatentUpsamplePipeline`, `LTXLatentUpsamplerModel` | Separate latent-space super-resolution model plus VAE decode. |
| `ltx_lora_adapters` | `LTXVideoLoraLoaderMixin`, transformer `PeftAdapterMixin` | Runtime/load-time adapter mutation. ICLoRA repos should be reviewed here or as condition adapters. |
| `ltx2_video` | `pipeline_ltx2*.py`, `transformer_ltx2.py`, `autoencoder_kl_ltx2.py` | Different text encoder, connectors, audio/vocoder components, and LTX2 VAE/transformer classes. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer | Cross dim | VAE shape | Scheduler | Variant trap |
| --- | --- | --- | ---: | --- | --- | --- |
| `LTX-Video` | `LTXPipeline` | 28 layers, 32 heads, head 64, inner 2048 | 2048 | latent 128, blocks `[128,256,512,512]`, default spatial 32, temporal 8 | FlowMatch Euler, dynamic shift, terminal 0.1 | Original base T2V; smaller denoiser. |
| `LTX-Video-0.9.1` | `LTXPipeline` | Same as base | 2048 | wider decoder config, timestep-conditioned VAE decode | FlowMatch Euler, dynamic shift, terminal 0.1 | VAE config differs from base. |
| `LTX-Video-0.9.5` | `LTXPipeline` | Same 28-layer denoiser | 2048 | latent 128, blocks `[128,256,512,1024,2048]`, spatial 32, temporal 8 | FlowMatch Euler, static shift | Codec structure matches newer 0.9.x but denoiser is smaller. |
| `LTX-Video-0.9.7-dev` | `LTXConditionPipeline` | 48 layers, 32 heads, head 128, inner 4096 | 4096 | same 128-channel, spatial 32, temporal 8, timestep-conditioned VAE | FlowMatch Euler, static shift | Condition pipeline and larger transformer. |
| `LTX-Video-0.9.7-distilled` | `LTXConditionPipeline` | same as 0.9.7-dev | 4096 | same as 0.9.7-dev | FlowMatch Euler, static shift | Distilled weights, same operator shape. |
| `LTX-Video-0.9.8-13B-distilled` | `LTXConditionPipeline` | same 48 layers/head 128 | 4096 | same as 0.9.7-dev | FlowMatch Euler, static shift | 13B-scale checkpoint, same core config fields. |
| `ltxv-spatial-upscaler-0.9.7` | `LTXLatentUpsamplePipeline` | no transformer | n/a | same LTX VAE | no scheduler | Separate latent upsampler model. |

Transformer fields:

| Field | Base / 0.9.1 / 0.9.5 | 0.9.7+ |
| --- | ---: | ---: |
| `in_channels`, `out_channels` | 128 / 128 | 128 / 128 |
| `patch_size`, `patch_size_t` | 1 / 1 | 1 / 1 |
| `num_layers` | 28 | 48 |
| `num_attention_heads` | 32 | 32 |
| `attention_head_dim` | 64 | 128 |
| `inner_dim` | 2048 | 4096 |
| `caption_channels` | 4096 | 4096 |
| `cross_attention_dim` | 2048 | 4096 |
| `qk_norm` | `rms_norm_across_heads` | `rms_norm_across_heads` |

VAE fields:

| Field | LTX 0.9.x value |
| --- | --- |
| `in_channels`, `out_channels` | 3 / 3 |
| `latent_channels` | 128 |
| `patch_size`, `patch_size_t` | 4 / 1 at codec boundary |
| `spatial_compression_ratio` | 32 in sampled explicit configs; source default computes `4 * 2^3 = 32` |
| `temporal_compression_ratio` | 8 in sampled explicit configs; source default computes `1 * 2^3 = 8` |
| `latents_mean/std` | source buffers: zeros / ones, shape `[128]` |
| `scaling_factor` | 1.0 |
| `encoder_causal`, `decoder_causal` | true / false |
| `timestep_conditioning` | false in original base config, true in 0.9.1+ sampled configs |

Text encoder config facts:

| Component | Fields |
| --- | --- |
| `T5TokenizerFast` | T5 v1.1 tokenizer, vocab 32128 plus 100 extra IDs, pad/eos/unk. Pipeline default `max_sequence_length=128`. |
| `T5EncoderModel` | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_ff=10240`, gated GELU, `layer_norm_epsilon=1e-6`. |

Recommended first Dinoml scheduler slice:

- `FlowMatchEulerDiscreteScheduler` with LTX 0.9.7 static-shift config:
  `num_train_timesteps=1000`, `shift=1.0`, `use_dynamic_shifting=false`,
  `base_shift=0.5`, `max_shift=1.15`, no terminal shift, no stochastic
  sampling.
- Add dynamic shifting and `shift_terminal=0.1` next for the original
  `LTX-Video`/0.9.1 configs.

## 3a. Family variation traps

- Denoiser source contract is packed tokens `[B,S,D]`, not NCTHW maps. VAE
  source contract remains `[B,C,T,H,W]`.
- VAE codec patching (`patch_size=4`) is separate from transformer token
  packing (`patch_size=1` in sampled configs). Do not conflate them.
- 0.9.7+ doubles attention head dim and inner width from 2048 to 4096 and
  changes cross-attention width from 2048 to 4096.
- Base configs use `LTXPipeline`; newer configs use `LTXConditionPipeline`,
  which changes scheduler sign, per-token timesteps, RoPE coordinate input, and
  condition token/mask behavior.
- I2V base pipeline preserves the first latent frame by unpacking latents,
  stepping only `[:, :, 1:]`, and repacking each scheduler step.
- Condition pipeline passes `-noise_pred` into the FlowMatch scheduler and
  uses `per_token_timesteps`; base T2V passes `noise_pred` directly.
- VAE timestep-conditioned decode adds random noise to latents before decode
  and passes a decode timestep tensor to the decoder.
- Source `LTXEulerAncestralRFScheduler` exists, but official sampled model
  indexes/configs use `FlowMatchEulerDiscreteScheduler`; treat the ancestral RF
  scheduler as a separate scheduler candidate.
- NDHWC/NHWC is only an optimization candidate. Pipeline packing, VAE
  patchify, channel mean/std, RMSNorm, and Conv3d channel axes all need guards.

## 4. Runtime tensor contract

For a typical 768x512, 161-frame LTX T2V run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,128,4096]` | T5 hidden states, duplicated per video. |
| prompt mask | `prompt_attention_mask` | `[B,128]` bool | Converted to additive attention bias inside transformer. |
| latent noise map | before pack | `[B,128,21,16,24]` NCTHW | `T_lat=(161-1)/8+1`, H/32, W/32. |
| packed latents | denoiser input | `[B,8064,128]` | With transformer patch 1, `S=21*16*24`. |
| CFG model input | packed tokens | `[2B,8064,128]` | Negative then positive. |
| timestep | base T2V | `[2B]` | Expanded scalar timestep. |
| transformer output | `noise_pred` | `[2B,8064,128]` | Same token shape as input. |
| scheduler sample | `latents` | `[B,8064,128]` | FlowMatch step in token space. |
| unpacked latents | VAE input | `[B,128,21,16,24]` | Denormalized with mean/std/scaling. |
| decoded video | output | `[B,3,161,512,768]` | Postprocessed by `VideoProcessor`. |

Condition pipeline tensors:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| conditions | image/video tensors | `[B,3,T,H,W]` NCTHW | Preprocessed and VAE-encoded. |
| condition latents | normalized | `[B,128,T_cond_lat,H/32,W/32]` | Posterior sampled/mode by `retrieve_latents`. |
| packed condition tokens | extra prefix tokens | `[B,S_cond,128]` | Prepended before base latent tokens when condition is not at frame 0. |
| `video_coords` | coordinates | `[B,3,S_total]` then float | Scaled by temporal/spatial compression and frame rate; feeds RoPE. |
| `conditioning_mask` | mask | `[B,S_total]` | Used for per-token timesteps and final token update mask. |
| per-token timestep | timestep | `[B,S_total,1]` | `min(t, (1-mask)*1000)` for hard conditions. |

CPU/data-pipeline work includes tokenization, T5 execution when prompt embeds
are not supplied, image/video resize and normalization, and output conversion.
GPU/runtime work includes latent generation, VAE encode/decode, token
pack/unpack, transformer denoise, CFG/guidance rescale, scheduler updates, and
condition masking.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW VAE maps and `[B,S,D]` packed denoiser tokens.
- Pack/unpack reshape and permute:
  `[B,C,F,H,W] -> [B,F/pt,H/p,W/p,C,pt,p,p] -> [B,S,C*pt*p*p]`.
- CFG batch concat/chunk, prompt/mask concat, condition token concat.
- `torch.where` token update masks for condition pipeline.
- Per-channel VAE mean/std/scaling broadcast over `[B,C,T,H,W]`.
- `video_coords` meshgrid, scale, concat, gather, and frame-rate adjustment.

### Convolution/downsample/upsample ops

- LTX VAE `LTXVideoCausalConv3d`, including causal and noncausal temporal
  edge padding by repeating first/last frames.
- VAE codec patchify via encoder input channels `3 * 4 * 4`.
- 3D ResNet blocks with RMSNorm over channel-last view, SiLU/swish, Conv3d,
  optional shortcut Conv3d.
- `LTXVideoDownsampler3d` reshape/permute/mean residual path plus causal Conv3d.
- `LTXVideoUpsampler3d` reshape/permute upsample plus causal Conv3d, optional
  residual branch.

### GEMM/linear ops

- T5 text encoder if admitted later.
- `proj_in: Linear(128 -> inner_dim)`.
- PixArt text projection from 4096 to inner dim.
- Timestep/AdaLayerNormSingle MLPs.
- Attention Q/K/V and output projections with bias.
- FeedForward GELU-approximate MLP.
- `proj_out: Linear(inner_dim -> 128)`.

### Attention primitives

- Latent-token self-attention with QK RMSNorm and 3D RoPE.
- Text cross-attention with additive T5 mask bias.
- No added-KV/IP-Adapter branch in base LTX source.

### Normalization and adaptive conditioning

- Transformer RMSNorm and final LayerNorm.
- Attention Q/K RMSNorm over flattened head width.
- Adaptive scale/shift/gate from timestep embedding in each block.
- VAE RMSNorm over channel axis implemented by movedim to channel-last.
- Optional decoder timestep conditioning in VAE ResNet blocks.

### Scheduler and guidance arithmetic

- FlowMatch Euler table generation and step update.
- CFG arithmetic: `uncond + scale * (text - uncond)`.
- Guidance rescale: std reductions over non-batch axes and blend.
- Condition pipeline `per_token_timesteps` sigma lookup/update.

### Video-specific ops

- Temporal compression formula `(frames - 1) // 8 + 1`.
- Codec temporal tiling and blending, deferred from first slice.
- Video postprocess from decoded NCTHW.

## 6. Denoiser/model breakdown

`LTXVideoTransformer3DModel.forward`:

```text
hidden_states [B,S,128]
-> 3D RoPE from num_frames/height/width or video_coords
-> encoder_attention_mask [B,L] to additive bias [B,1,L]
-> Linear proj_in to inner_dim
-> AdaLayerNormSingle(timestep) -> temb and embedded_timestep
-> PixArtAlphaTextProjection(T5 embeds) -> [B,L,inner_dim]
-> N x LTXVideoTransformerBlock
-> LayerNorm + final timestep scale/shift
-> Linear proj_out -> [B,S,128]
```

`LTXVideoTransformerBlock`:

```text
RMSNorm
-> timestep scale/shift for MSA
-> self-attention(QKV, QK RMSNorm, RoPE, dispatch_attention_fn)
-> gated residual add
-> text cross-attention(Q from latents, K/V from projected T5, mask bias)
-> residual add
-> RMSNorm + timestep scale/shift
-> GELU-approximate FeedForward
-> gated residual add
```

Width examples:

- Base/0.9.5: inner dim 2048, heads 32, head dim 64, 28 blocks.
- 0.9.7/0.9.8: inner dim 4096, heads 32, head dim 128, 48 blocks.

## 7. Attention requirements

Primary implementation is `LTXVideoAttnProcessor` in `transformer_ltx.py`,
calling `dispatch_attention_fn` from `attention_dispatch.py`.

- Self-attention is noncausal over latent video tokens.
- Cross-attention queries latent tokens and keys/values projected T5 tokens.
- Q/K normalization uses `torch.nn.RMSNorm(dim_head * heads)` before unflatten
  to `[B,S,heads,head_dim]`.
- RoPE applies to self-attention Q/K before head unflatten. It is generated in
  fp32 from either regular grids or condition-pipeline `video_coords`.
- Cross-attention uses additive mask bias derived from T5 attention mask.
- Processor supports Diffusers attention backend dispatch; eager/native
  dispatch path defines parity. Source-supported fused QKV mutation is not a
  default requirement for LTX.

Flash-style constraints:

- Base self-attention is a candidate for Dinoml flash-style kernels when
  head dim is supported: 64 for base/0.9.5, 128 for 0.9.7+.
- Cross-attention with mask bias is a separate provider shape; a flash path must
  support additive masks or fall back.
- QK RMSNorm and RoPE are explicit pre-attention ops, not hidden provider
  behavior unless fused under exact guards.
- Condition-pipeline extra tokens and `video_coords` change RoPE inputs but not
  the attention primitive itself.

## 8. Scheduler and denoising-loop contract

Official configs use `FlowMatchEulerDiscreteScheduler`.

Base T2V loop:

```text
sigmas = linspace(1.0, 1 / num_steps, num_steps)
mu = calculate_shift(video_sequence_length, base_image_seq_len, max_image_seq_len, base_shift, max_shift)
set_timesteps(num_steps, sigmas=sigmas, mu=mu)
for t in timesteps:
  latent_model_input = cat([latents]*2) if CFG else latents
  noise_pred = transformer(latent_model_input, prompt_embeds, t)
  noise_pred = CFG/guidance_rescale(noise_pred)
  latents = scheduler.step(noise_pred, t, latents)
```

Image-to-video loop differs by unpacking token outputs to NCTHW, stepping only
frames after the first latent frame, concatenating the preserved first frame,
and repacking.

Condition loop differs by:

```text
timestep = min(t, (1 - conditioning_mask) * 1000)
noise_pred = transformer(..., timestep=[B,S,1], video_coords=...)
denoised = scheduler.step(-noise_pred, t, latents, per_token_timesteps=timestep)
latents = where(tokens_to_denoise_mask, denoised, latents)
```

First Dinoml slice should keep scheduler iteration and table generation as
host-visible state. Compile one denoiser call, CFG arithmetic, and a static
FlowMatch step first; add dynamic shifting, terminal shift, and per-token
timesteps after base parity.

## 9. Position, timestep, and custom math

- `LTXVideoRotaryPosEmbed` builds `[B,S,inner_dim]` cos/sin tables in fp32.
  Frequencies use `inner_dim // 6` per axis, with padding when the dim is not
  divisible by 6.
- Base RoPE grid uses latent frame/height/width token coordinates and optional
  interpolation scale:
  `(vae_temporal_compression_ratio / frame_rate, vae_spatial_compression_ratio,
  vae_spatial_compression_ratio)`.
- Condition pipeline bypasses regular grid generation and supplies scaled
  `video_coords`.
- Timestep embeddings use `AdaLayerNormSingle`; block modulation table produces
  shift/scale/gate for attention and MLP.
- Final output LayerNorm receives a separate shift/scale from embedded timestep.
- VAE decode timestep conditioning, when enabled, uses decode timestep as
  `temb` and may add random noise to latents before decode.

Precompute candidates: prompt embeddings, prompt masks, scheduler sigmas and
timesteps, RoPE tables for fixed video shape and frame rate, and regular
packed-coordinate maps.

## 10. Preprocessing and input packing

Text:

- Tokenize with T5 tokenizer, max length 128 by pipeline default.
- T5 encoder output `[B,128,4096]`.
- Negative prompt defaults to empty string when CFG is active.
- CFG concatenates negative/positive embeddings and masks on batch dim.

Video/image:

- T2V starts from random NCTHW latents and immediately packs to `[B,S,128]`.
- I2V preprocesses image to NCHW, encodes `[B,3,1,H,W]`, normalizes, repeats
  across latent frames, mixes with noise by a first-frame mask, then packs.
- Condition pipeline preprocesses images/videos to NCTHW conditions, VAE-encodes
  each condition, packs extra condition tokens, and constructs scaled
  `video_coords`.
- Decode path unpacks tokens, denormalizes latents, optionally applies
  decode-time noise/timestep conditioning, decodes VAE, and postprocesses video.

NHWC/NDHWC guarded notes:

- Preserve source NCTHW at VAE boundaries and packed `[B,S,D]` at transformer
  boundaries initially.
- Candidate NDHWC islands are VAE Conv3d/ResNet blocks only after rewriting
  channel-axis RMSNorm, Conv3d weights, causal temporal padding, patchify order,
  posterior channel split, and latent mean/std broadcast.
- Mark pack/unpack, VAE codec patchify, and scheduler token updates as
  no-layout-translation regions until shape tests prove equivalence.

## 11. Graph rewrite / lowering opportunities

### Rewrite: pipeline latent pack/unpack

Source pattern:

```text
NCTHW reshape by temporal/spatial patch -> permute -> flatten to [B,S,D]
inverse reshape -> permute -> flatten back to NCTHW
```

Replacement: explicit video-token pack/unpack op.

Preconditions: source NCTHW layout, `F % patch_size_t == 0`, `H/W % patch_size
== 0`, patch sizes from transformer config. For sampled LTX configs this is
patch 1, but keep the general path. Failure cases: VAE codec patching is a
different operation.

### Rewrite: LTX attention prelude

Source pattern:

```text
Linear Q/K/V -> RMSNorm(Q,K) -> RoPE(Q,K) -> dispatch_attention_fn
```

Replacement: canonical attention provider with explicit QK norm and RoPE
pre-ops.

Preconditions: supported head dim, dtype, sequence length, noncausal mode, and
mask support for cross-attention. Failure cases: additive cross-attention mask
unsupported, condition sequence too long, backend lacks head_dim=128.

### Rewrite: FlowMatch Euler static step

Source pattern:

```text
prev = sample + (sigma_next - sigma) * model_output
```

Replacement: pointwise scheduler step over packed tokens.

Preconditions: official static-shift config, no stochastic sampling, scalar
timestep path. Failure cases: per-token timesteps, dynamic shift table mismatch,
condition pipeline sign convention not represented.

### Rewrite: VAE codec patchify/downsample island

Source pattern:

```text
RGB NCTHW -> codec patch/channel pack -> causal Conv3d/ResNet/downsample stack
```

Replacement: layout-aware codec pack plus Conv3d island.

Preconditions: `patch_size=4`, `patch_size_t=1`, faithful channel order,
NCTHW boundary. Failure cases: NDHWC provider without exact channel/patch
permutation and RMSNorm axis rewrite.

## 12. Kernel fusion candidates

Highest priority:

- Large Linear/GEMM coverage for 4096-wide 0.9.7+ transformer QKV, cross-attn,
  FFN, text projection, and proj_out.
- QK RMSNorm + RoPE + attention provider prelude for self-attention.
- Ada scale/shift/gate plus residual epilogues around attention and FFN.
- Packed-token FlowMatch step plus CFG arithmetic and guidance rescale.
- VAE causal/noncausal Conv3d + RMSNorm + SiLU + residual blocks.

Medium priority:

- Pipeline pack/unpack kernels and condition token/video-coordinate pack.
- VAE codec patchify/unpatchify and down/up sampler reshape-permute patterns.
- VAE timestep-conditioned decode epilogues and decode-noise blend.
- I2V first-frame preservation unpack/step/repack.

Lower priority:

- Dynamic shift and terminal shift scheduler table variants.
- Spatial/temporal VAE tiling blend kernels.
- Latent upsampler model kernels.
- LoRA/ICLoRA runtime adapter application.
- LTX Euler ancestral RF stochastic scheduler.

## 13. Runtime staging plan

Stage 1: Parse configs for `Lightricks/LTX-Video-0.9.7-dev` or
`0.9.7-distilled`; accept external T5 prompt/negative embeddings.

Stage 2: Implement packed latent token contract, pack/unpack parity, regular
RoPE generation, and one `LTXVideoTransformerBlock` parity at reduced sequence.

Stage 3: Full `LTXVideoTransformer3DModel` random-tensor forward parity for the
48-layer/4096-width config where memory allows; otherwise validate block stack
incrementally.

Stage 4: Add CFG and guidance rescale over packed tokens.

Stage 5: Implement static `FlowMatchEulerDiscreteScheduler` step for packed
tokens with scheduler state in host control.

Stage 6: Add `AutoencoderKLLTXVideo` decode for 128-channel latents, no tiling
or framewise decode first, including timestep-conditioned decode.

Stage 7: Run a short deterministic T2V loop with scheduler on host and VAE
decode.

Stage 8: Add original `LTX-Video` dynamic shifting and terminal shift.

Stage 9: Add `ltx_image2video`, then `ltx_condition` per-token timesteps and
condition masks.

Stage 10: Treat latent upsampler, ICLoRA/control adapters, LTX2, and rare
schedulers as separate admissions.

## 14. Parity and validation plan

- Config/default reconciliation for LTX base, 0.9.5, 0.9.7, and 0.9.8.
- Pack/unpack parity for `[B,128,21,16,24]`.
- RoPE parity for regular grid and condition `video_coords`.
- One attention processor parity with and without cross-attention mask.
- One `LTXVideoTransformerBlock` parity for 2048 and 4096 widths.
- Full transformer forward parity on small synthetic grids.
- CFG and guidance-rescale parity over packed tokens.
- FlowMatch `set_timesteps` and one `step` parity for static and dynamic shift
  configs.
- VAE decode parity for `[B,128,T_lat,H_lat,W_lat]`, including
  timestep-conditioned decode.
- VAE encode posterior/mode parity for I2V/condition readiness.
- I2V first-frame preservation step parity.
- Condition pipeline per-token timestep and `torch.where` update parity.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per provider.

## 15. Performance probes

- One denoiser step by sequence length: small synthetic, 512x768x161, and
  larger LTX grids.
- 2048-width vs 4096-width transformer block time split.
- Attention backend comparison for head_dim 64 and 128.
- CFG batch concat memory and time compared with separate denoiser calls.
- Pack/unpack and RoPE generation overhead by frame count.
- VAE decode throughput and memory for 128-channel latents.
- VAE timestep-conditioned decode cost with and without decode noise blend.
- Condition pipeline overhead: VAE encode, extra tokens, per-token timesteps,
  and masked update.
- Faithful NCTHW VAE path versus guarded NDHWC Conv3d island.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `ltx_image2video`: image VAE encode, first-frame mask, masked timesteps, and
  first-frame-preserving scheduler update.
- `ltx_condition`: arbitrary image/video conditions, extra packed condition
  tokens, `video_coords`, per-token timesteps, hard-condition masks.
- `ltx_i2v_long_multi_prompt`: long-video and multi-prompt orchestration.
- `ltx_latent_upsampler`: `LTXLatentUpsamplePipeline` and
  `LTXLatentUpsamplerModel`, no scheduler.
- `ltx_lora_ic_lora`: LoRA/ICLoRA adapter loading and runtime mutation.
- `ltx_scheduler_ancestral_rf`: `LTXEulerAncestralRFScheduler`, stochastic RF
  update separate from official FlowMatch Euler configs.
- `ltx_vae_tiling_framewise`: spatial tiling, temporal tiling, batch slicing,
  and framewise encode/decode memory policy.
- `ltx2_video`: LTX2 text connectors, Gemma text encoder, video/audio
  components, LTX2 transformer, LTX2 VAE, vocoder.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse LTX 0.9.x model index and component configs from `H:/configs` plus official component configs.
- [ ] Load `LTXVideoTransformer3DModel` weights for a 0.9.7-style checkpoint.
- [ ] Accept external T5 prompt and negative prompt embeddings plus masks.
- [ ] Implement packed `[B,S,128]` latent token contract.
- [ ] Implement NCTHW latent pack/unpack parity.
- [ ] Implement LTX 3D RoPE from regular grids.
- [ ] Implement QK RMSNorm + RoPE + attention fallback/provider.
- [ ] Implement `LTXVideoTransformerBlock` adaptive norm, self-attn, cross-attn, FFN, and gates.
- [ ] Implement full transformer forward parity.
- [ ] Implement CFG and guidance rescale over packed tokens.
- [ ] Implement static FlowMatch Euler scheduler slice for packed tokens.
- [ ] Implement `AutoencoderKLLTXVideo` decode with timestep conditioning, tiling disabled.
- [ ] Add VAE encode posterior/mode parity for I2V/condition variants.
- [ ] Add short T2V loop parity with scheduler in host control.
- [ ] Add dynamic shift/terminal shift configs for older LTX checkpoints.
- [ ] Add `ltx_image2video` first-frame conditioning as a separate stage.
- [ ] Add `ltx_condition` per-token timestep and `video_coords` support.
- [ ] Add guarded NDHWC/VAE layout optimization only after faithful NCTHW parity.
