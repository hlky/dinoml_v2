# Diffusers Mochi Video Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  genmo/mochi-1-preview

Config sources:
  H:/configs/genmo/mochi-1-preview/model_index.json
  Official Hugging Face raw component configs inspected in memory:
    genmo/mochi-1-preview/transformer/config.json
    genmo/mochi-1-preview/vae/config.json
    genmo/mochi-1-preview/scheduler/scheduler_config.json
    genmo/mochi-1-preview/text_encoder/config.json
    genmo/mochi-1-preview/tokenizer/tokenizer_config.json
    genmo/mochi-1-preview/tokenizer/special_tokens_map.json
  Official repo API metadata inspected in memory:
    repo sha 14be5fcea23095ed330cb214647916a451e38b6e
    transformer fp32 index total_size 40110710976, 5 shards
    transformer bf16 index total_size 20055355488, 3 shards
    text_encoder index total_size 19049242624, 4 shards

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/mochi/pipeline_mochi.py
  diffusers/src/diffusers/pipelines/mochi/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_mochi.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_mochi.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/single_file_utils.py
  diffusers/src/diffusers/loaders/single_file_model.py

External component configs inspected:
  T5EncoderModel and T5 tokenizer configs from the official Mochi repo.

Any missing files or assumptions:
  The local config cache only contained model_index.json. Official component
  configs and metadata were accessible without a gated/authenticated retry and
  were not saved because this task's owned write path is only this report.
  The official Diffusers repo exposes one base text-to-video pipeline and a bf16
  transformer weight variant, not separate official img2img/inpaint/control
  pipeline configs. XLA/NPU/MPS/Flax/ONNX, multi-GPU/context parallel,
  callbacks/interrupt mutation, safety/NSFW, training/loss/dropout, and
  gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

Mochi in Diffusers is a text-to-video latent diffusion pipeline with a single
T5-XXL text encoder, `MochiTransformer3DModel`, `FlowMatchEulerDiscreteScheduler`,
and `AutoencoderKLMochi`. The denoiser is an asymmetric joint video/text DiT:
video tokens have width 3072, text context tokens are projected to width 1536,
and attention returns separate video and context streams until the final block.

```text
prompt / negative prompt
  -> T5Tokenizer/T5TokenizerFast + T5EncoderModel
  -> prompt embeds [B,256,4096] + bool attention mask
  -> latent video noise [B,12,T_lat,H/8,W/8] NCTHW
  -> denoising loop:
       CFG batch concat
       MochiTransformer3DModel(latents, T5 embeds, timestep, mask)
       FP32 CFG arithmetic
       FlowMatchEulerDiscreteScheduler.step
  -> denormalize latents with 12-channel mean/std and scaling_factor
  -> AutoencoderKLMochi decode
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Pipeline | `MochiPipeline`, `pipeline_mochi.py` | Source of runtime tensor contract and custom sigma schedule. |
| Denoiser | `MochiTransformer3DModel`, `transformer_mochi.py` | Required; operates on NCTHW latent maps then internal 2D patches per latent frame. |
| VAE | `AutoencoderKLMochi`, `autoencoder_kl_mochi.py` | Decode required for T2V output; encode is a separate codec/variant candidate. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Required with Mochi-specific custom sigmas and `invert_sigmas=true`. |
| Text encoder | `T5EncoderModel` + `T5Tokenizer`/`T5TokenizerFast` | Accept external prompt embeddings and masks first. |
| Loader mutation | `Mochi1LoraLoaderMixin` | Separate candidate; mutates transformer adapters only. |

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `mochi_lora_adapters` | `Mochi1LoraLoaderMixin`, `lora_pipeline.py` | Runtime/load-time LoRA adapter mutation for the transformer through PEFT. |
| `mochi_single_file_conversion` | `single_file_utils.py`, `single_file_model.py` | Original checkpoint key conversion and component mapping. |
| `mochi_bf16_transformer` | same pipeline/model classes, bf16 transformer index | Weight precision/loading variant; same operator graph, lower memory. |
| `mochi_vae_encode_15ch` | `AutoencoderKLMochi` | VAE encode has 15-channel input, Fourier features, and attention-bearing blocks; not needed for base T2V decode. |
| `mochi_vae_tiling_framewise` | `AutoencoderKLMochi` tiling/slicing/framewise flags | Spatial tiling, batch slicing, and framewise decode memory policies. |

No official Diffusers Mochi IP-Adapter, ControlNet, T2I-Adapter, GLIGEN,
img2img, inpaint, depth2img, or upscaling pipeline classes were present in the
Mochi folder at the inspected commit.

## 3. Important config dimensions

Representative config sweep:

| Source | Pipeline | Transformer | VAE | Scheduler | Weight metadata |
| --- | --- | --- | --- | --- | --- |
| `genmo/mochi-1-preview` local model_index | `MochiPipeline` | `MochiTransformer3DModel` | `AutoencoderKLMochi` | `FlowMatchEulerDiscreteScheduler` | Component classes only. |
| Official component configs | T2V | 48 layers, 24 heads, head 128, inner 3072, patch 2 | 12 latent channels, 15 input channels, 3 output channels | FlowMatch Euler, `invert_sigmas=true` | fp32 transformer index 40.1 GB, text encoder 19.0 GB. |
| Official bf16 transformer index | Same class graph | Same 1071 transformer tensors | Same VAE config | Same scheduler | bf16 transformer index 20.1 GB, 3 shards. |
| Source defaults | Same class graph | Same as official transformer defaults | Same as official VAE defaults | FlowMatch default plus Mochi overrides in config | Useful for omitted-field reconciliation. |

Transformer dimensions:

| Field | Value | Runtime effect |
| --- | ---: | --- |
| `in_channels`, `out_channels` | 12 / effective 12 | Latent map channels and denoiser output channels. |
| `patch_size` | 2 | Per-frame Conv2d patchify over latent H/W. |
| `num_layers` | 48 | Final block uses `context_pre_only=true`. |
| `num_attention_heads`, `attention_head_dim` | 24 / 128 | Video hidden width 3072. |
| `pooled_projection_dim` | 1536 | Context stream width after T5 projection. |
| `text_embed_dim` | 4096 | T5 hidden width. |
| `time_embed_dim` | 256 | Sinusoidal timestep projection width. |
| `qk_norm` | `rms_norm` | Implemented as Mochi RMSNorm over head dim. |
| `max_sequence_length` | 256 | Pipeline default prompt length. |

VAE dimensions:

| Field | Value | Runtime effect |
| --- | ---: | --- |
| `in_channels`, `out_channels` | 15 / 3 | Encode is not plain RGB; decode returns RGB/video channels. |
| `latent_channels` | 12 | Denoiser latent channel count. |
| encoder channels | `[64,128,256,384]` | Narrower encoder than decoder. |
| decoder channels | `[128,256,512,768]` | Decode starts with Conv3d 12 -> 768. |
| `layers_per_block` | `[3,3,4,6,3]` | Mid/down/up block depth. |
| temporal expansions | `[1,2,3]` | Temporal compression/expansion product 6. |
| spatial expansions | `[2,2,2]` | Spatial compression/expansion product 8. |
| attention blocks | `[false,true,true,true,true]` | Encoder has attention in down/mid blocks; decoder disables attention. |
| `latents_mean/std` | 12 values each | Per-channel decode denormalization. |
| `scaling_factor` | 1.0 | Included in formula but no scalar scale change. |

Text encoder config facts:

| Component | Fields |
| --- | --- |
| `T5EncoderModel` | `d_model=4096`, 24 layers, 64 heads, `d_ff=10240`, gated GELU, vocab 32128, `layer_norm_epsilon=1e-6`. |
| tokenizer | T5 tokenizer files with pad/eos/unk and 100 extra IDs. `model_index.json` names `T5Tokenizer`; source imports `T5TokenizerFast`. |

Recommended first Dinoml scheduler slice:

- Use `FlowMatchEulerDiscreteScheduler` with official Mochi config:
  `num_train_timesteps=1000`, `shift=1.0`, `use_dynamic_shifting=false`,
  `invert_sigmas=true`, `base_shift=0.5`, `max_shift=1.15`.
- Preserve the pipeline's custom `linear_quadratic_schedule(num_steps, 0.025)`
  and pass those values as custom `sigmas` to `set_timesteps`.
- This differs from the generic SD3/Flux FlowMatch slice because Mochi requires
  inverted sigmas and a terminal sigma of 1.0.

## 3a. Family variation traps

- Source latent/video layout is NCTHW. The transformer temporarily flattens
  `[B,T,C,H,W]` into per-frame BCHW for `PatchEmbed`; the VAE remains NCTHW.
- Mochi's pipeline checks only height/width divisible by 8, but transformer
  patchify also requires latent H/W divisible by `patch_size=2`; effective image
  H/W should be divisible by 16 for faithful transformer shapes.
- Default output dimensions are 480x848 and 19 frames. This yields latent shape
  `[B,12,4,60,106]` and token length `4*30*53=6360`.
- The VAE temporal expansion/compression product is 6, and decode drops the
  first 5 upscaled frames by default, yielding `(T_lat - 1) * 6 + 1` frames.
- CFG is implemented by batch concatenation and FP32 guidance arithmetic, not by
  an embedded guidance tensor.
- Attention is joint video/text attention, but text tokens are mask-filtered per
  batch item before SDPA. This is a varlen-like attention shape in Python loops,
  not a simple dense additive-mask call.
- `MochiAttention` uses non-square projections: video width 3072, context width
  1536, shared attention inner width 3072, and separate output projections.
- `MochiVaeAttnProcessor2_0` has a single-frame shortcut that skips Q/K
  attention and applies V plus output projection.
- VAE encode is not part of base T2V inference and has 15 input channels plus
  Fourier features. Do not admit it as a plain RGB video encoder.
- NDHWC/NHWC is a guarded optimization only. Axis-sensitive source ops include
  Conv3d/Conv2d channel axes, GroupNorm over dim 1 after per-frame flatten,
  latent mean/std broadcast, posterior split along dim 1, temporal cache dims,
  and VAE tiling H/W dimensions.

## 4. Runtime tensor contract

For the default 480x848, 19-frame T2V path:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| token ids | `input_ids` | `[B,256]` | CPU/tokenizer output, padded/truncated. |
| prompt embeds | `prompt_embeds` | `[B,256,4096]` | T5 encoder hidden states. |
| prompt mask | `prompt_attention_mask` | `[B,256]` bool | Required when prompt embeds are supplied directly. |
| CFG embeds | concat | `[2B,256,4096]` and `[2B,256]` | Negative then positive if `guidance_scale > 1`. |
| noisy latents | `latents` | `[B,12,4,60,106]` NCTHW | `T_lat=(19-1)//6+1`, H/8, W/8. |
| model input | CFG latents | `[2B,12,4,60,106]` | Duplicated on batch dim. |
| timestep | expanded | `[2B]`, dtype follows latents | Pipeline passes scheduler timestep `t`, and records `1000 - t` for compatibility. |
| patch tokens | hidden video tokens | `[2B,6360,3072]` | Per-frame Conv2d patchify with patch 2. |
| context tokens | caption projection | `[2B,256,1536]` | From `caption_proj(T5 hidden)`. |
| denoiser output | `noise_pred` | `[2B,12,4,60,106]` | Unpatchified by model. |
| CFG output | guided noise | `[B,12,4,60,106]` fp32 then cast | `uncond + scale * (text - uncond)`. |
| scheduler sample | latents | `[B,12,4,60,106]` | FlowMatch Euler update. |
| VAE decode input | denormalized latents | `[B,12,4,60,106]` | `latents * latents_std / scaling_factor + latents_mean`. |
| decoded video | sample | `[B,3,19,480,848]` NCTHW | Decode creates 24 frames then drops first 5. |
| pipeline output | frames | PIL list, NumPy, or tensor | `VideoProcessor.postprocess_video`; pt/np are `[B,T,C,H,W]`. |

For the example 480x848, 163-frame path used in docs, `T_lat=28` and token
length is `28*30*53=44520`.

CPU/data-pipeline work: tokenization, T5 execution when embeddings are not
provided, prompt truncation warnings, and output conversion. GPU/runtime work:
latent noise generation, transformer denoise, CFG arithmetic, scheduler update,
VAE denormalization/decode, and postprocess tensor transforms.

Cacheable across denoising steps or requests: prompt embeddings and masks,
negative prompt embeddings and masks, RoPE tables for fixed latent grid,
scheduler sigma/timestep tables for fixed step count, VAE latent mean/std
broadcast constants, and static transformer weights.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW latent maps; permute/flatten/unflatten/reshape around transformer
  patchify and VAE blocks.
- BCHW Conv2d patchify per latent frame:
  `[B*T,12,H,W] -> Conv2d(12 -> 3072,k=2,s=2) -> [B,T*H/2*W/2,3072]`.
- Unpatchify:
  `Linear(3072 -> 12*2*2) -> reshape [B,T,H2,W2,2,2,12] -> permute -> NCTHW`.
- CFG concat/chunk over batch dim.
- Prompt-mask filtering using `nonzero` and per-sample variable text lengths.
- VAE posterior split/chunk along channel dim for encode.
- Per-channel latent mean/std broadcast over `[B,12,T,H,W]`.

### Convolution/downsample/upsample ops

- Transformer patch embedding Conv2d with bias.
- VAE decoder `Conv3d(12 -> 768, 1x1x1)`.
- VAE causal Conv3d in residual blocks via `CogVideoXCausalConv3d`.
- VAE upsample blocks implemented as Linear projection followed by
  temporal/spatial unpatchify, not interpolate.
- VAE encoder downsample Conv3d with kernel/stride equal to temporal/spatial
  expansion factors, plus Fourier features and Linear input projection.

### GEMM/linear ops

- T5 external text encoder if admitted later.
- Time embedding MLP, caption projection `Linear(4096 -> 1536)`, and
  attention-pool Q/K/V/output projections.
- Transformer video Q/K/V: `Linear(3072 -> 3072, bias=false)`.
- Context Q/K/V: `Linear(1536 -> 3072, bias=false)`.
- Attention outputs: video `Linear(3072 -> 3072)`, context
  `Linear(3072 -> 1536)` for non-final blocks.
- SwiGLU feed-forward with inner dims `(4*dim*2)//3`: video 8192, context 4096.
- VAE per-position Linear projections in encoder/decoder and upsample blocks.

### Attention primitives

- Transformer joint video/text SDPA with QK RMSNorm and 3D RoPE on video Q/K.
- Per-sample valid text token packing before attention; padded output restores
  full text sequence length.
- Attention-pool SDPA over one pooled query plus text tokens.
- VAE attention over per-spatial-location temporal sequences, causal, with L2
  QK norm, in encoder/mid blocks only.

### Normalization and adaptive conditioning

- Mochi RMSNorm in fp32 with optional weight.
- Adaptive RMSNormZero producing `scale_msa`, `gate_msa`, `scale_mlp`,
  `gate_mlp`; gates pass through `tanh`.
- `MochiLayerNormContinuous` for final context-pre-only block.
- `AdaLayerNormContinuous` final output norm.
- VAE `MochiChunkedGroupNorm3D`: per-frame GroupNorm over flattened `B*T`.

### Position/timestep/guidance embeddings

- Sinusoidal `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`.
- `MochiAttentionPool` masked text pooling and caption projection.
- 3D RoPE using learned `pos_frequencies` parameter `[3,24,64]`.
- CFG arithmetic only; no embedded guidance conditioning.

### Scheduler and guidance arithmetic

- `linear_quadratic_schedule` host table generation.
- FlowMatch Euler custom-sigma `set_timesteps`, inverted sigmas, `step_index`.
- FP32 scheduler step: `prev = sample + (sigma_next - sigma) * model_output`.
- CFG FP32: `uncond + guidance_scale * (text - uncond)`.

### VAE/postprocessing ops

- Fourier feature expansion for encode: input plus sin/cos at powers 2^6 and
  2^7.
- KL posterior mean/logvar split and `DiagonalGaussianDistribution`.
- Decode temporal frame drop `dec[:, :, temporal_compression_ratio - 1:]`.
- Spatial tiling blend and batch slicing deferred.
- Video postprocess NCTHW -> per-batch TCHW -> PIL/NumPy/tensor.

## 6. Denoiser/model breakdown

`MochiTransformer3DModel.forward`:

```text
hidden_states [B,12,T,H,W]
-> time_embed(timestep, T5 embeds, mask)
   - Timesteps -> TimestepEmbedding gives temb [B,3072]
   - MochiAttentionPool(T5 embeds, mask) gives pooled [B,3072]
   - caption_proj gives context [B,256,1536]
-> permute to [B*T,12,H,W]
-> PatchEmbed Conv2d(k=s=2) -> [B*T,(H/2)*(W/2),3072]
-> reshape to video tokens [B,T*(H/2)*(W/2),3072]
-> learned-frequency 3D RoPE table for T,H/2,W/2
-> 48 x MochiTransformerBlock
-> AdaLayerNormContinuous + Linear(3072 -> 48)
-> unpatchify to [B,12,T,H,W]
```

`MochiTransformerBlock`:

```text
video RMSNormZero(temb) -> normed video, gates/scales
context RMSNormZero(temb) for blocks 0..46, or LayerNormContinuous for block 47
MochiAttention:
  video QKV + context QKV into shared 24x128 head space
  Q/K RMSNorm for both streams
  RoPE on video Q/K only
  per-sample valid text-token concat
  noncausal SDPA over [video tokens + valid context tokens]
  split back to video/context streams and project outputs
video gated residual attention
video RMSNorm(scale_mlp) -> SwiGLU FeedForward -> gated residual
context attention/FF residuals only for blocks 0..46
```

`AutoencoderKLMochi` decode:

```text
latents [B,12,T,H,W]
-> Conv3d(12 -> 768, 1)
-> MochiMidBlock3D without attention
-> up blocks:
     repeated MochiResnetBlock3D with causal Conv3d
     Linear(C -> C_out * temporal_expansion * spatial_expansion^2)
     reshape/permute unpatchify to larger T/H/W
-> MochiMidBlock3D without attention
-> SiLU
-> Linear(128 -> 3) per position
-> drop first 5 frames by default
```

`AutoencoderKLMochi` encode, for separate admission:

```text
sample [B,15,T,H,W]
-> FourierFeatures: input + sin/cos features
-> Linear over channel-last positions
-> mid/down/mid blocks with causal Conv3d and causal temporal attention
-> GroupNorm + SiLU
-> Linear -> [B,24,T/6,H/8,W/8] posterior moments
```

## 7. Attention requirements

Primary denoiser implementation is `MochiAttention` plus
`MochiAttnProcessor2_0` in `attention_processor.py`; it calls PyTorch
`F.scaled_dot_product_attention` directly rather than `attention_dispatch.py`.

Transformer attention:

- Video stream: query/key/value from 3072-wide tokens, 24 heads, head dim 128,
  bias false.
- Context stream: query/key/value from 1536-wide projected T5 tokens into the
  same 3072 inner attention width.
- Q/K normalization: `MochiRMSNorm(dim_head=128)` for video and context Q/K.
- RoPE: applies only to video query/key before transpose to `[B,H,S,D]`.
- Masking: uses bool text mask to select valid context token indices per batch
  item, concatenates video tokens with only valid context tokens, then pads SDPA
  output back to the full context length.
- Causality: noncausal in transformer attention.
- Output: split to video and context lengths; context output projection exists
  for all but the final block.

VAE attention:

- Source uses generic `Attention` with `MochiVaeAttnProcessor2_0`,
  `heads=channels//32`, `dim_head=32`, `qk_norm="l2"`, and `is_causal=True`.
- Attention is applied after reshaping `[B,C,T,H,W]` to
  `[B*H*W,T,C]`, so it is temporal attention per spatial location.
- Single-frame branch skips Q/K attention and uses V plus output projection.

Flash-style constraints:

- A Dinoml flash-style provider for base transformer attention needs varlen or
  segmented text suffix support, because valid text token count differs per
  batch item and source loops over batch.
- RoPE and QK RMSNorm are explicit pre-attention ops. They can be fused only
  under exact head-dim/dtype/layout guards.
- Context output is not the same width as video output. Provider epilogues must
  preserve separate video/context projections or fall back.
- Head dim 128 and token counts up to roughly 44.5k video tokens for 163-frame
  480x848 generation require strong workspace and sequence-length admission.
- Eager PyTorch SDPA plus the source per-sample packing loop defines parity.

## 8. Scheduler and denoising-loop contract

Mochi uses `FlowMatchEulerDiscreteScheduler` but not the generic default sigma
schedule. The pipeline generates a Genmo-style linear/quadratic schedule:

```text
threshold_noise = 0.025
sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
retrieve_timesteps(scheduler, sigmas=sigmas)
```

The official scheduler config sets `invert_sigmas=true`. In `set_timesteps`,
custom sigmas are shifted, then inverted:

```text
sigmas = 1.0 - sigmas
timesteps = sigmas * num_train_timesteps
sigmas = cat([sigmas, ones(1)])
```

Denoising loop:

```text
for t in scheduler.timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  timestep = t.expand(batch).to(latents.dtype)
  noise_pred = transformer(latent_model_input, prompt_embeds, timestep, mask)
  noise_pred = noise_pred.float()
  if CFG:
    noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents.float())
  latents = latents.to(original_latents_dtype)
```

First Dinoml slice should keep `linear_quadratic_schedule`,
`set_timesteps`, and scheduler `step_index` as host-visible state. Compile the
transformer, CFG arithmetic, and one FlowMatch step only after custom-sigma and
inversion parity are locked.

## 9. Position, timestep, and custom math

- `linear_quadratic_schedule` is model-coupled pipeline math and should be
  represented in scheduler metadata for Mochi parity.
- `MochiRoPE` computes centered H/W positions scaled by
  `sqrt((192*192)/(height*width))` and temporal positions `0..T-1`, then uses
  learned `pos_frequencies` with einsum `nd,dhf->nhf`.
- RoPE cos/sin are computed in fp32; application rotates even/odd head
  features and returns original dtype.
- `MochiAttentionPool` builds a pooled query from the masked mean of T5 tokens,
  prepends it to the text sequence, and performs one-query SDPA.
- Adaptive block math uses `tanh(gate)` for residual gating and `(1 + scale)`
  for modulation.
- VAE encode Fourier features add sin/cos of each input channel at frequencies
  `2^6 * 2*pi` and `2^7 * 2*pi`.

Precompute candidates: prompt embeddings and masks, pooled/caption-projected
text if the transformer boundary accepts those directly, RoPE tables for fixed
T/H/W, scheduler timesteps/sigmas for fixed step count, and VAE mean/std
broadcast tensors.

## 10. Preprocessing and input packing

Text:

- Tokenize with T5 tokenizer, `max_sequence_length=256`, padding to max length,
  truncation, and special tokens.
- Source imports `T5TokenizerFast`; model_index names `T5Tokenizer`. Runtime
  should accept either tokenizer artifact but expose the resulting IDs/mask
  contract.
- T5 encoder output is `[B,256,4096]`; embeddings and masks are repeated for
  `num_videos_per_prompt`.
- Negative prompt defaults to empty string when CFG is active.
- Optional `force_zeros_for_empty_prompt` zeros IDs and mask for empty prompt
  compatibility.

Video/latents:

- T2V starts from random NCTHW latents with `randn_tensor` in fp32, cast to
  prompt dtype.
- `num_frames` is converted to latent frames by `(num_frames - 1)//6 + 1`.
  The source does not round requested frame count upward; decode returns the
  frame count implied by latent frames and drop policy.
- Transformer patchify happens inside the model, not at the pipeline boundary.
- Decode path denormalizes latents, calls VAE, then `VideoProcessor` converts
  NCTHW to user output format.

NHWC/NDHWC guarded notes:

- Preserve source NCTHW at pipeline and VAE boundaries initially.
- Candidate NHWC island: transformer per-frame Conv2d patchify/unpatchify, with
  explicit OIHW->HWIO weight transform and inverse token order tests.
- Candidate NDHWC island: VAE decode Conv3d/resnet blocks only after rewriting
  GroupNorm channel axis, causal cache axis, Conv3d weights, and temporal drop.
- No-layout-translation guards should cover text-mask packing, attention
  sequence split/pad, VAE posterior split, latent stats broadcast, and tiling.

## 11. Graph rewrite / lowering opportunities

### Rewrite: per-frame patchify/unpatchify

Source pattern:

```text
NCTHW -> B*T,C,H,W -> Conv2d(k=s=2) -> flatten/transposed tokens
Linear(3072 -> 12*2*2) -> reshape/permute -> NCTHW
```

Replacement: explicit video-frame patchify/unpatchify ops.

Preconditions: source NCTHW layout, latent H/W divisible by 2, `patch_size=2`,
no positional embedding add in `PatchEmbed`, Conv2d bias preserved. Failure
cases: NHWC provider without exact weight transform or inverse flatten order.

### Rewrite: Mochi joint varlen attention

Source pattern:

```text
video QKV + context QKV -> QK RMSNorm -> RoPE(video Q/K)
for each batch item:
  valid_context = nonzero(mask)
  SDPA(cat(video, valid_context))
  pad output back to full context length
```

Replacement: segmented joint-attention provider or faithful loop fallback.

Preconditions: supported head dim 128, noncausal, valid text suffix lengths
known, no dropout, dtype admitted, output split/projection represented.
Failure cases: dense-only flash provider, unsupported sequence length, or
attempt to include padded context tokens without matching source behavior.

### Rewrite: FlowMatch Mochi scheduler step

Source pattern:

```text
custom sigmas -> shift -> invert -> terminal one
prev = sample + (sigma_next - sigma) * model_output
```

Replacement: host-generated sigma table plus pointwise step kernel.

Preconditions: official scheduler config, no stochastic sampling, scalar
timestep path, no per-token timesteps. Failure cases: generic FlowMatch table
without inversion or custom Genmo schedule.

### Rewrite: VAE decode upsample block

Source pattern:

```text
resnet stack -> channel-last Linear(C -> C_out*st*sh*sw)
-> reshape [B,C_out,st,sh,sw,T,H,W] -> permute -> expanded NCTHW
```

Replacement: pixel-shuffle-like 3D unpatchify op plus GEMM.

Preconditions: exact temporal/spatial expansion factors from config, source
NCTHW boundary, no tiling/framewise decode. Failure cases: NDHWC translation
without matching expansion order or framewise cache mode.

## 12. Kernel fusion candidates

Highest priority:

- Large Linear/GEMM coverage for transformer Q/K/V, context projections, FFN,
  caption projection, and output projection.
- QK RMSNorm + RoPE + segmented/joint attention provider with batch-varlen text
  suffix support.
- Adaptive RMSNorm scale/gate + residual epilogues around attention and FFN.
- Transformer Conv2d patchify/unpatchify and frame/token layout transforms.
- Mochi-specific FlowMatch step plus FP32 CFG arithmetic.

Medium priority:

- `MochiAttentionPool` masked one-query SDPA and caption projection.
- VAE decode Conv3d + ChunkedGroupNorm + SiLU + residual blocks.
- VAE Linear upsample/unpatchify kernels for temporal/spatial expansions.
- Latent denormalization pointwise kernel over `[B,12,T,H,W]`.

Lower priority:

- VAE encoder attention and Fourier-feature path.
- VAE spatial tiling blend kernels and framewise decode cache policy.
- LoRA adapter application/fuse/unfuse.
- Single-file original checkpoint conversion.

## 13. Runtime staging plan

Stage 1: Parse `genmo/mochi-1-preview` model index and component configs; load
or stub weights for `MochiTransformer3DModel`; accept external T5 prompt and
negative prompt embeddings plus masks.

Stage 2: Implement NCTHW latent contract, per-frame patchify/unpatchify parity,
Mochi RoPE generation/application, timestep/text conditioning, and one
`MochiTransformerBlock` parity at reduced sequence length.

Stage 3: Implement full transformer forward parity on small synthetic grids,
then default latent grid `[B,12,4,60,106]`.

Stage 4: Add CFG batch concat/chunk and FP32 guidance arithmetic.

Stage 5: Implement Mochi's FlowMatch Euler custom-sigma/inverted-sigma scheduler
slice with host-visible timestep/sigma tables.

Stage 6: Add `AutoencoderKLMochi` decode only, tiling/slicing/framewise flags
disabled, preserving NCTHW and temporal drop.

Stage 7: Run a short deterministic T2V loop with scheduler in host control and
VAE decode smoke.

Stage 8: Add bf16 transformer loading/admission and attention provider guards
for head_dim 128 and varlen text.

Stage 9: Treat LoRA, single-file conversion, VAE encode, and VAE tiling/framewise
memory policy as separate admissions.

First Dinoml admission recommendation: start with
`mochi_transformer_step_external_t5`, not full end-to-end Mochi. The first
bounded slice should compile one denoiser call at reduced latent grid with
external prompt embeddings/masks and the eager SDPA-compatible attention
fallback. Add scheduler and VAE decode after transformer shape parity. Mochi
VAE encode should not be the first video-codec admission because the 15-channel
input and encoder attention make it less bounded than Wan/CogVideoX decode.

## 14. Parity and validation plan

- Config/default reconciliation for local model_index plus official component
  configs.
- `linear_quadratic_schedule` values and `FlowMatchEulerDiscreteScheduler`
  `set_timesteps(sigmas=...)` parity, including `invert_sigmas=true`.
- Patchify/unpatchify parity for `[B,12,4,60,106]`.
- RoPE table parity for default 19-frame and long 163-frame latent grids.
- `MochiAttentionPool` parity with masks containing different valid lengths.
- `MochiAttnProcessor2_0` parity with per-batch valid text-token packing.
- One `MochiTransformerBlock` parity at 3072/1536 widths.
- Full `MochiTransformer3DModel` forward parity on small synthetic grids.
- CFG FP32 arithmetic parity.
- FlowMatch one-step parity with fixed denoiser output.
- VAE decode parity for `[B,12,4,60,106] -> [B,3,19,480,848]`.
- Temporal-drop parity: `T_lat=1`, `T_lat=4`, and `T_lat=28`.
- VAE encode posterior/mode parity only in the separate encode candidate.
- Suggested tolerances: fp32 scheduler arithmetic `rtol=1e-5, atol=1e-6`;
  transformer/VAE fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per provider.

## 15. Performance probes

- One transformer step by latent grid: small synthetic, default 480x848x19, and
  480x848x163.
- Attention backend comparison for head_dim 128 and varlen text suffix lengths.
- 48-block time split: QKV/context projections, attention, FFN, adaptive norm,
  patchify/unpatchify.
- CFG batch concat memory/time compared with separate cond/uncond calls.
- FlowMatch scheduler overhead versus denoiser time.
- VAE decode throughput and memory for `[B,12,4,60,106]` and long-frame grids.
- Faithful NCTHW VAE path versus guarded NDHWC Conv3d island.
- bf16 transformer weight variant memory, load time, and numeric drift.
- Prompt embedding/T5 throughput if text encoder is admitted later.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `mochi_lora_adapters`: `Mochi1LoraLoaderMixin` transformer PEFT adapter
  loading, hotswap, fuse/unfuse, and adapter state.
- `mochi_single_file_conversion`: original Genmo/Comfy-style checkpoint key
  conversion and non-Diffusers weight layout.
- `mochi_bf16_transformer`: bf16 transformer weight shards and dtype-specific
  admission; same graph but different precision/memory contract.
- `mochi_vae_encode_15ch`: 15-channel input, Fourier features, posterior
  moments, and attention-bearing encoder.
- `mochi_vae_tiling_framewise`: VAE batch slicing, spatial tiling/blending, and
  framewise decode cache policy.
- `mochi_text_encoder_t5_xxl`: T5-XXL prompt encoder admission and cache
  contract, if Dinoml chooses to compile text encoders.

Unsupported/not present in official Diffusers Mochi at this commit:

- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN.
- Official Mochi img2img, inpaint, depth2img, or upscaling pipelines.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse `genmo/mochi-1-preview` model index and component configs.
- [ ] Load or stub `MochiTransformer3DModel` weights and accept external T5 embeddings/masks.
- [ ] Implement NCTHW latent shape contract and default latent-grid derivation.
- [ ] Implement per-frame Conv2d patchify/unpatchify parity.
- [ ] Implement Mochi timestep embedding, caption projection, and masked attention pool.
- [ ] Implement Mochi 3D RoPE table generation and application.
- [ ] Implement `MochiRMSNormZero`, `MochiModulatedRMSNorm`, gates, and residual epilogues.
- [ ] Implement Mochi joint video/text attention fallback with source text-mask packing.
- [ ] Add one-block and full-transformer forward parity.
- [ ] Implement FP32 CFG arithmetic.
- [ ] Implement Mochi FlowMatch custom linear/quadratic, inverted-sigma scheduler slice.
- [ ] Add one-step denoising parity with fixed prompt embeddings.
- [ ] Implement `AutoencoderKLMochi` decode with tiling/slicing/framewise disabled.
- [ ] Add VAE latent denormalization and temporal-drop parity.
- [ ] Add short T2V loop smoke with scheduler in host control.
- [ ] Add bf16 transformer loading/admission as a precision variant.
- [ ] Keep LoRA, single-file conversion, VAE encode, VAE tiling/framewise, and T5 compilation as separate candidates.
- [ ] Add guarded NHWC/NDHWC optimization tests only after faithful source-layout parity.
