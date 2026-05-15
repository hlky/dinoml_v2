# Diffusers EasyAnimate Operator and Integration Report

Candidate slug: `easyanimate`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  alibaba-pai/EasyAnimateV5.1-7b-zh-diffusers
  alibaba-pai/EasyAnimateV5.1-12b-zh-InP-diffusers
  alibaba-pai/EasyAnimateV5.1-12b-zh-Control-diffusers
  Local stale/older cache entries also checked:
    H:/configs/alibaba-pai/EasyAnimateV2-XL-2-768x768/model_index.json
    H:/configs/alibaba-pai/EasyAnimateV4-XL-2-InP/model_index.json
    H:/configs/alibaba-pai/EasyAnimateV5-12b-zh*/model_index.json

Config sources:
  Existing H:/configs entries above were inspected first. They mostly point at
  older PixArt/Hunyuan/Transformer2D-style classes or only model_index.json.
  Official V5.1 component configs were fetched with `hf download` and inspected
  in-memory from the three model ids above. The temporary fetched config copies
  were removed afterward because this task's owned write path is only this
  report.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/easyanimate/pipeline_easyanimate.py
  diffusers/src/diffusers/pipelines/easyanimate/pipeline_easyanimate_inpaint.py
  diffusers/src/diffusers/pipelines/easyanimate/pipeline_easyanimate_control.py
  diffusers/src/diffusers/pipelines/easyanimate/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_easyanimate.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_magvit.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  Qwen2VLForConditionalGeneration and Qwen2Tokenizer configs from the official
  V5.1 Diffusers repos. Text encoder execution is treated as external for the
  first Dinoml slice.

Any missing files or assumptions:
  The selected target is current Diffusers EasyAnimate V5.1. Older local cache
  model indexes are recorded as family-history traps, not first-slice targets.
  Multi-GPU/offload mechanics, callbacks/interrupt mutation, XLA/NPU/MPS/Flax/
  ONNX, safety/NSFW, and training/loss/dropout/gradient-checkpointing paths are
  out of scope.
```

## 2. Pipeline and component graph

EasyAnimate V5.1 is a latent video transformer family. The base pipeline uses
Qwen2-VL text embeddings, `EasyAnimateTransformer3DModel`, FlowMatch Euler, and
`AutoencoderKLMagvit`.

```text
prompt / negative prompt
  -> Qwen2Tokenizer chat template + Qwen2VLForConditionalGeneration hidden state
  -> latent video noise [B,16,T_lat,H/8,W/8]
  -> denoising loop:
       CFG batch concat
       EasyAnimateTransformer3DModel(latents, timestep, text embeds)
       optional guidance rescale
       FlowMatchEulerDiscreteScheduler.step
  -> latents / vae.scaling_factor
  -> AutoencoderKLMagvit decode
  -> VideoProcessor postprocess
```

Inpaint adds optional video encode, masks, masked-video latents, strength-based
timestep slicing, and either channel concatenation into the transformer input or
post-step latent replacement. Control adds control-video VAE encode or camera
mask resizing, plus optional reference-image latent channels.

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Base T2V pipeline | `EasyAnimatePipeline`, `pipeline_easyanimate.py` | Use as the first runtime contract. |
| Denoiser | `EasyAnimateTransformer3DModel`, `transformer_easyanimate.py` | Required; patchifies NCTHW video latents by running Conv2d per frame. |
| VAE | `AutoencoderKLMagvit`, `autoencoder_kl_magvit.py` | Decode required for output; encode required for inpaint/control. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | Required; configs use dynamic shifting with `mu=1` from the pipeline. |
| Text encoder | `Qwen2VLForConditionalGeneration` + `Qwen2Tokenizer` | Accept external prompt embeddings and attention masks first. |

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `easyanimate_inpaint` | `EasyAnimateInpaintPipeline` | VAE encode of input/masked video, mask encode or trilinear resize, strength slicing, optional latent replacement after scheduler step. |
| `easyanimate_control` | `EasyAnimateControlPipeline` | Control video/camera/ref-image latents concatenated to model input; 48-channel transformer config. |
| `easyanimate_magvit_vae` | `AutoencoderKLMagvit` | Causal Conv3d codec with framewise encode/decode caches, tiling, and spatial GroupNorm. |
| `easyanimate_qwen2vl_text` | Qwen2-VL text encoder/tokenizer | Chat-template prompt encoding and 3584-wide LLM hidden states. |
| `easyanimate_older_v2_v4_v5` | stale local cache model indexes | Older repos point at PixArt/Hunyuan/Transformer2D classes and should not be mixed with V5.1. |
| `easyanimate_lora_adapters` | generic Diffusers loader mixins if supported by checkpoints | Runtime/load-time adapter mutation separate from base denoiser parity. |

## 3. Important config dimensions

Representative config sweep:

| Repo | Pipeline | Transformer in/out | Layers / MMDiT | Heads x dim | VAE | Scheduler | Variant trap |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `EasyAnimateV5.1-7b-zh-diffusers` | `EasyAnimatePipeline` | 16 -> 16 | 36 / 18 | 48 x 64 | latent 16, scale 8/4, scaling 0.7125 | FlowMatch Euler dynamic shift | Best first target; base T2V. |
| `EasyAnimateV5.1-12b-zh-InP-diffusers` | `EasyAnimateInpaintPipeline` | 33 -> 16 | 48 / 48 | 48 x 64 | same VAE | same scheduler | Adds 16 latent + 1 mask + 16 masked-video channels. |
| `EasyAnimateV5.1-12b-zh-Control-diffusers` | `EasyAnimateControlPipeline` | 48 -> 16 | 48 / 48 | 48 x 64 | same VAE | same scheduler | Adds 16 latent + 16 control + 16 ref-image channels. |

Transformer fields from official component configs:

| Field | V5.1 value | Runtime effect |
| --- | --- | --- |
| `patch_size` | 2 | Per-frame Conv2d patch embedding and Linear unpatchify over H/W only. |
| `inner_dim` | 3072 | `48 * 64`; Q/K/V, FFN, text projection width. |
| `text_embed_dim` | 3584 | Qwen2-VL hidden state width. |
| `text_embed_dim_t5` | null in sampled V5.1 | Secondary T5 concat path inactive. |
| `add_norm_text_encoder` | true | RMSNorm before text projection. |
| `enable_text_attention_mask` | true | Pipeline requires prompt attention masks with external embeds. |
| `time_embed_dim` | 512 | Timestep MLP width for adaptive norm/gates. |
| `time_position_encoding_type` | `3d_rope` | 3D RoPE over latent frame, height, width token grid. |

VAE fields:

| Field | V5.1 Magvit value |
| --- | --- |
| `in_channels`, `out_channels` | 3 / 3 |
| `latent_channels` | 16 |
| `block_out_channels` | `[128,256,512,512]` |
| `down/up block types` | one spatial block, then three spatial-temporal blocks |
| `layers_per_block` | 2 |
| `spatial_compression_ratio` | source-derived `2 ** (4 - 1) = 8` |
| `temporal_compression_ratio` | source-derived `2 ** (4 - 2) = 4` |
| `scaling_factor` | 0.7125 |
| `spatial_group_norm` | true |
| `mini_batch_encoder/decoder` | 4 sample frames / 1 latent frame in config; source names are 4 and 1 |

Qwen2-VL text encoder config facts:

| Field | Value |
| --- | --- |
| hidden size | 3584 |
| layers / attention heads / KV heads | 28 / 28 / 4 |
| intermediate size | 18944 |
| max position embeddings | 32768 |
| dtype metadata | bfloat16 |
| vision config | present but base EasyAnimate prompt path uses text content only |

Recommended first Dinoml scheduler slice:

- `FlowMatchEulerDiscreteScheduler` with official EasyAnimate fields:
  `num_train_timesteps=1000`, `shift=3.0`, `use_dynamic_shifting=true`,
  `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`,
  `max_image_seq_len=4096`.
- The pipeline calls `retrieve_timesteps(..., mu=1)` for FlowMatch. Start with
  this fixed-`mu` dynamic-shift table, not the SD3/Flux image-sequence dynamic
  `mu` heuristic.

## 3a. Family variation traps

- Source video/latent layout is NCTHW. Treat NDHWC as a guarded optimization
  only.
- The transformer patch embed is `Conv2d` applied to `[B*T,C,H,W]`, not Conv3d
  tubelet patching.
- Base T2V uses transformer `in_channels=16`; inpaint uses 33; control uses
  48. Do not admit all three through one hidden channel contract.
- If transformer output channels are not VAE latent channels, the pipeline
  chunks the output on dim 1 and keeps the first half.
- `EasyAnimateAttnProcessor2_0` performs joint MMDiT attention by concatenating
  text and video tokens inside one SDPA call. Text tokens also receive attention
  outputs and FFN updates for `mmdit_layers`; later blocks can be image-only if
  `num_layers > mmdit_layers`.
- `encoder_hidden_states_t5` and `text_proj_t5` exist in source but are inactive
  for sampled V5.1 configs.
- Local `H:/configs/alibaba-pai/EasyAnimateV2/V4/V5` model indexes are older
  family artifacts, not current Diffusers EasyAnimate V5.1 component configs.
- VAE causal Conv3d stores mutable per-module caches. Dinoml needs explicit
  per-session cache state and reset boundaries.
- VAE `spatial_group_norm=true` means many GroupNorms flatten `[B,T]` and run
  2D GroupNorm per frame; this differs from GroupNorm over full NCTHW.
- Inpaint `resize_inpaint_mask_directly=true` uses a 1-channel mask resized to
  latent shape rather than VAE-encoding mask channels.
- Control camera path resizes mask/control values directly to latent shape and
  multiplies by 6; normal control video path VAE-encodes frames.

## 4. Runtime tensor contract

For a typical 49-frame, 512x512 base run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,256,3584]` | Qwen2-VL hidden state `[-2]`; max length default 256. |
| prompt mask | `prompt_attention_mask` | `[B,256]` | Required when prompt embeds are supplied directly. |
| latent noise | `latents` | `[B,16,13,64,64]` NCTHW | `T_lat=(49-1)//4+1`; H/W divided by 8. |
| CFG model input | `latent_model_input` | `[2B,16,13,64,64]` | Batch concat for negative/positive prompts. |
| timestep | `t_expand` | `[2B]` | Scalar per sample, dtype matched to latents. |
| patch tokens | hidden | `[B,13*32*32,3072]` | Conv2d patch size 2 per frame. |
| text tokens | encoder hidden | `[B,256,3072]` after RMSNorm+Linear | Concatenated into joint attention. |
| noise prediction | `noise_pred` | `[B,16,13,64,64]` | May be chunked if transformer returns wider output. |
| scheduler state | sigmas/timesteps/step index | host-visible tables plus scalar per step | FlowMatch Euler update over NCTHW latents. |
| VAE decode input | unscaled latents | `[B,16,13,64,64]` | Pipeline divides by 0.7125 before decode. |
| decoded video | output | `[B,3,49,512,512]` NCTHW | Postprocessed by `VideoProcessor`. |

Inpaint tensors:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| input video | `video` | `[B,3,T,H,W]` NCTHW after preprocessing | Encoded for img2vid/inpaint strength paths. |
| mask video | `mask_video` | `[B,1 or 3,T,H,W]` source; tiled to latent channels for replacement | Trilinear resized to latent grid. |
| mask latents | `mask_latents` | `[B,1,T_lat,H/8,W/8]` when direct resize | 1 channel for 33-channel transformer. |
| masked video latents | `masked_video_latents` | `[B,16,T_lat,H/8,W/8]` | VAE mode plus scaling factor. |
| inpaint concat | `inpaint_latents` | `[2B,17,T_lat,H/8,W/8]` with CFG | Concatenated with 16 noisy channels in transformer. |

Control tensors:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| control video latents | `control_latents` part 1 | `[B,16,T_lat,H/8,W/8]` | From VAE encode, or resized camera control * 6. |
| reference image latents | `control_latents` part 2 | `[B,16,T_lat,H/8,W/8]` | First latent frame filled from reference image encode. |
| transformer input after concat | internal | `[B,48,T_lat,H/8,W/8]` | 16 noisy + 32 control/ref channels. |

CPU/data-pipeline work: tokenizer chat template, Qwen2-VL text encoder when
embeds are not supplied, image/video resize and output conversion. GPU/runtime
work: latent generation, VAE encode/decode, transformer denoising, CFG,
guidance rescale, scheduler step, and inpaint/control packing.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW latent/video tensors; reshape, view, flatten, unflatten, permute,
  transpose, concat, chunk, tile/repeat.
- Per-frame transformer patchify:
  `[B,C,T,H,W] -> [B*T,C,H,W] -> Conv2d -> [B,T,C,H/2,W/2] -> [B,S,C]`.
- Unpatchify:
  `[B,S,out_channels*2*2] -> [B,C,T,H,W]`.
- CFG batch concat/chunk and output-channel chunking.
- Trilinear interpolation over masks/control tensors.
- VAE latent scale/unscale by scalar `0.7125`.

### Convolution/downsample/upsample ops

- Transformer patch `Conv2d(in_channels -> 3072, kernel=2, stride=2)`.
- Magvit VAE causal `Conv3d`, 1x1 `quant_conv`/`post_quant_conv`.
- VAE residual blocks: GroupNorm, SiLU, causal Conv3d, residual add.
- VAE downsample: causal Conv3d stride `(2,2,2)` or `(1,2,2)` with explicit
  H/W padding.
- VAE upsample: nearest spatial interpolate, causal Conv3d, optional temporal
  interpolate.

### GEMM/linear ops

- Qwen2-VL external text encoder if admitted later.
- RMSNorm/Linear text projection `3584 -> 3072`.
- Timestep embedding: sinusoidal Timesteps width 3072, MLP to 512.
- Attention Q/K/V/add-Q/add-K/add-V/output/add-output projections.
- FeedForward GELU-approximate MLP for image and text streams.
- Final Linear `3072 -> patch_size^2 * out_channels`.

### Attention primitives

- Joint text/video self-attention over `[text_tokens + video_tokens]`.
- QK LayerNorm on both image and added text Q/K.
- 3D RoPE applied only to video-token Q/K slices.
- Optional image-only blocks after `mmdit_layers` if a config uses
  `num_layers > mmdit_layers`.

### Normalization and adaptive conditioning

- FP32LayerNorm / LayerNorm on transformer tokens.
- RMSNorm for text projection when `add_norm_text_encoder=true`.
- `EasyAnimateLayerNormZero`: SiLU + Linear to shift/scale/gate for image and
  text streams.
- `AdaLayerNorm` final output modulation.
- VAE per-frame GroupNorm when `spatial_group_norm=true`.

### Scheduler and guidance arithmetic

- FlowMatch Euler `set_timesteps(..., mu=1)` and `step`.
- `scale_model_input` if present.
- CFG: `uncond + guidance * (text - uncond)`.
- Guidance rescale: std reduction over all non-batch axes and blend.
- Inpaint strength slicing and FlowMatch `scale_noise` for noising input video
  latents.

### Video-specific ops

- Temporal compression `(frames - 1) // 4 + 1`.
- VAE framewise encode in chunks of 4 sample frames and framewise decode in
  chunks of 1 latent frame with causal cache.
- Control/reference latent packing.
- Video postprocess from NCTHW.

## 6. Denoiser/model breakdown

`EasyAnimateTransformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> timestep Timesteps + TimestepEmbedding -> temb [B,512]
-> 3D RoPE table from latent grid and patch size
-> optional concat inpaint/control channels on dim 1
-> per-frame Conv2d patch projection -> [B,T*H/2*W/2,3072]
-> text RMSNorm/Linear projection -> [B,L,3072]
-> N x EasyAnimateTransformerBlock
-> LayerNorm
-> AdaLayerNorm(temb)
-> Linear to patch pixels
-> unpatchify to [B,out_channels,T,H,W]
```

`EasyAnimateTransformerBlock`:

```text
EasyAnimateLayerNormZero(image,text,temb)
-> joint attention:
     image QKV and text added-QKV
     QK LayerNorm
     concat text/image tokens
     apply 3D RoPE to image token Q/K slice
     SDPA
     split text/image outputs and project
-> gated residual add to both streams
EasyAnimateLayerNormZero(image,text,temb)
-> image FeedForward and text FeedForward when MMDiT block
-> gated residual add to both streams
```

Base 7B uses 36 blocks, with the first 18 configured as MMDiT-style joint
text/image blocks. The 12B inpaint/control sampled configs use 48/48, so all
blocks update the text stream.

## 7. Attention requirements

Primary implementation is `EasyAnimateAttnProcessor2_0` local to
`transformer_easyanimate.py`, not the generic `attention_dispatch.py` path.

- Attention backend is PyTorch `F.scaled_dot_product_attention` in eager source.
- Query/key/value shapes after projection are `[B,heads,seq,head_dim]`, with
  heads 48 and head dim 64 in sampled configs.
- For MMDiT blocks, text and video tokens are concatenated into one attention
  problem. Text added Q/K use their own projection and optional QK norms.
- RoPE applies only to video token positions. Text tokens remain unrotated.
- Source passes no explicit text attention mask into transformer forward today,
  despite pipeline carrying `prompt_attention_mask`; parity is the mask-free
  denoiser path unless source changes.
- Fused projections are not required by source, but Dinoml can fuse QKV/add-QKV
  under exact weight/layout guards.

Flash-style constraints:

- Base joint attention is a possible provider candidate for mask-free, noncausal
  sequence attention with head dim 64.
- A provider must either support the concatenated text+video sequence with RoPE
  already applied to only the video suffix, or Dinoml must materialize Q/K
  pre-ops before the provider call.
- Text-output return is semantically required for MMDiT blocks; do not treat
  text as read-only cross-attention context.

## 8. Scheduler and denoising-loop contract

Official V5.1 configs use `FlowMatchEulerDiscreteScheduler`:

```text
num_train_timesteps = 1000
shift = 3.0
use_dynamic_shifting = true
base_shift = 0.5
max_shift = 1.15
base_image_seq_len = 256
max_image_seq_len = 4096
```

Base loop:

```text
retrieve_timesteps(scheduler, steps, device, timesteps, mu=1)
for t in timesteps:
  model_input = cat([latents]*2) if CFG else latents
  model_input = scheduler.scale_model_input(model_input, t) if available
  t_expand = tensor([t] * batch)
  noise_pred = transformer(model_input, t_expand, text_embeds)
  if channels != 16: noise_pred = chunk(noise_pred, dim=1)[0]
  if CFG: noise_pred = uncond + scale * (text - uncond)
  if guidance_rescale: rescale by std over non-batch axes
  latents = scheduler.step(noise_pred, t, latents)
```

Keep `set_timesteps`, dynamic-shift table generation, step index, and scheduler
iteration host-visible first. Compile one transformer call, CFG arithmetic, and
one FlowMatch step only after table parity is proven.

## 9. Position, timestep, and custom math

- Timestep embedding uses Diffusers `Timesteps(inner_dim=3072,
  flip_sin_to_cos=true, freq_shift=0)` followed by `TimestepEmbedding` to
  512 channels with SiLU activation.
- `EasyAnimateRotaryPosEmbed` builds 3D RoPE from latent shape, patch size 2,
  and a fixed base latent grid `90x60` before patching. Crop coordinates are
  computed by a resize/crop helper.
- RoPE dimensions use the attention head dim value from config. Sampled configs
  have head dim 64.
- Adaptive block math is
  `norm(x) * (1 + scale) + shift`, then gated residuals from the same
  `SiLU(temb) -> Linear(6 * dim)` projection.
- VAE causal Conv3d uses replicate temporal left padding on first chunk and
  cached previous features on later chunks.

Precompute candidates: prompt embeddings/masks, scheduler timesteps/sigmas for
step count and `mu=1`, and RoPE tables for fixed latent frame/height/width.

## 10. Preprocessing and input packing

Text:

- Pipeline wraps prompt strings in Qwen chat-template messages and pads/truncates
  to max sequence length 256.
- Text encoder output uses hidden state `[-2]`.
- Negative prompts follow the same path when CFG is active.
- External prompt embeddings must be accompanied by attention masks, although
  the current transformer does not consume the mask.

Video/image:

- Base T2V starts from random NCTHW latents.
- Decode divides latents by VAE scaling factor, decodes Magvit VAE, then uses
  `VideoProcessor.postprocess_video`.
- Inpaint preprocesses input video by flattening `[B,T]` into images for
  `VaeImageProcessor`, then restores NCTHW.
- Inpaint masks are resized to latent shape, or encoded through the VAE
  depending on config.
- Control video is either VAE-encoded to 16 latent channels or directly resized
  for camera control. Reference image latents are copied into the first latent
  frame before channel concat.

Layout guard notes:

- Preserve NCTHW at pipeline and VAE boundaries initially.
- Transformer patchify/unpatchify is a no-layout-translation region until its
  exact frame-major token order is tested.
- NDHWC VAE islands require Conv3d weight transforms, GroupNorm axis rewrites,
  causal cache axis rewrites, and latent scaling broadcast rewrites.

## 11. Graph rewrite / lowering opportunities

### Rewrite: per-frame Conv2d patchify/unpatchify

Source pattern:

```text
[B,C,T,H,W] -> permute/flatten [B*T,C,H,W]
Conv2d kernel=stride=2
-> restore [B,T,C,H/2,W/2] -> [B,S,C]
Linear -> reshape/permute/flatten -> [B,C,T,H,W]
```

Replacement: explicit video-patch pack/unpack op plus Conv/GEMM primitives.

Preconditions: source NCTHW layout, `H` and `W` divisible by patch size, no
temporal patching, exact frame-major token order. Failure cases: attempting to
fold with VAE temporal compression or changing token order under NDHWC.

### Rewrite: EasyAnimate joint attention prelude

Source pattern:

```text
image QKV + text added-QKV -> QK LayerNorm -> concat text/image seq
-> apply RoPE to image suffix -> SDPA -> split outputs
```

Replacement: canonical joint-attention provider with explicit text/image token
segments and RoPE suffix metadata.

Preconditions: mask-free noncausal attention, head dim 64, text suffix/prefix
split known, dtype supported. Failure cases: provider treats text as read-only
cross-attention or applies RoPE to text tokens.

### Rewrite: Magvit framewise codec loop

Source pattern:

```text
first frame through causal Conv3d stack
then sample/latent frame chunks with cached previous features
clear cache after encode/decode
```

Replacement: VAE encode/decode region with explicit per-session causal-cache
buffers and reset markers.

Preconditions: tiling/slicing disabled, source chunk sizes 4 encode / 1 decode,
NCTHW layout. Failure cases: hidden module-global caches, spatial tiling,
control/inpaint encode sampling differences.

### Rewrite: FlowMatch fixed-mu step

Source pattern:

```text
set_timesteps(..., mu=1)
latents = scheduler.step(model_output, t, latents)
```

Replacement: host-generated FlowMatch tables plus pointwise latent update.

Preconditions: official EasyAnimate scheduler config, no stochastic branch,
scalar timestep per batch. Failure cases: alternate `mu`, custom sigmas, or
inpaint strength slicing without `set_begin_index` parity.

## 12. Kernel fusion candidates

Highest priority:

- 3072-wide Linear/GEMM coverage for QKV, added text QKV, FFN, text projection,
  and output projection.
- QK LayerNorm + video-suffix RoPE + joint attention.
- AdaLayerNormZero scale/shift/gate plus residual epilogues.
- Per-frame Conv2d patchify/unpatchify with layout-preserving token order.
- Magvit VAE causal Conv3d + per-frame GroupNorm + SiLU + residual blocks.

Medium priority:

- FlowMatch Euler step plus CFG and guidance-rescale pointwise/reduction work.
- VAE framewise decode cache management and scalar latent unscale.
- Inpaint mask resize/pack and latent replacement kernels.
- Control/ref-image latent concat and camera-control direct-resize path.

Lower priority:

- VAE spatial tiling blend kernels.
- Runtime adapter/LoRA mutation.
- Qwen2-VL text encoder integration.
- Older V2/V4/V5 non-current pipeline families.

## 13. Runtime staging plan

Stage 1: Parse current V5.1 base configs and load weights for
`EasyAnimateV5.1-7b-zh-diffusers`; accept external Qwen2-VL prompt and negative
prompt embeddings plus masks.

Stage 2: Implement faithful NCTHW latent contract, per-frame Conv2d patchify,
3D RoPE, timestep embedding, and one `EasyAnimateTransformerBlock` parity on
small synthetic grids.

Stage 3: Full `EasyAnimateTransformer3DModel` random-tensor forward parity for
the 7B 36-layer config, with MMDiT text-stream updates included.

Stage 4: Add CFG and guidance rescale.

Stage 5: Implement FlowMatch Euler dynamic-shift scheduler with fixed `mu=1`
and validate one step.

Stage 6: Add `AutoencoderKLMagvit` decode with tiling/slicing disabled and
explicit causal cache reset.

Stage 7: Run a short deterministic base T2V loop with scheduler in host
control and VAE decode.

Stage 8: Add inpaint encode/mask/strength path for the 33-channel 12B config.

Stage 9: Add control video/camera/ref-image latent packing for the 48-channel
control config.

Stage 10: Admit guarded NDHWC/Conv3d layout islands and attention fusions after
faithful NCTHW parity.

## 14. Parity and validation plan

- Config/default reconciliation for 7B base, 12B inpaint, and 12B control.
- Patchify/unpatchify parity for `[B,16,13,64,64]`.
- 3D RoPE parity for 49-frame 512x512 and non-square grids.
- One joint attention processor parity with text and video tokens.
- One transformer block parity for MMDiT and non-MMDiT settings.
- Full transformer forward parity on reduced frame/height/width grids.
- CFG and guidance-rescale parity over NCTHW latents.
- FlowMatch `set_timesteps(mu=1)` and one `step` parity.
- Magvit VAE decode parity with cache reset for one and multiple latent frames.
- Magvit VAE encode posterior mode/sample parity for inpaint/control.
- Inpaint mask pack and latent replacement parity.
- Control video/ref-image pack parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; model fp32
  `rtol=1e-4, atol=1e-5`; fp16/bf16 start with `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step by latent grid: 512x512x49, 720x1280-style grids, and small
  synthetic grids.
- 7B versus 12B transformer block time split: GEMM, attention, FFN.
- Joint attention backend comparison for text length 256 and video sequence
  length sweeps.
- CFG batch concat memory versus separate positive/negative calls.
- Patchify/unpatchify overhead by frame count.
- Magvit VAE decode throughput and cache memory for 13 latent frames and longer
  videos.
- Inpaint/control VAE encode and channel-pack overhead.
- Faithful NCTHW VAE path versus guarded NDHWC Conv3d island.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `easyanimate_inpaint`: 33-channel transformer, VAE encode, mask resize/encode,
  strength slicing, FlowMatch `scale_noise`, and latent replacement.
- `easyanimate_control`: 48-channel transformer, control video/camera/ref-image
  latent packing.
- `easyanimate_magvit_vae`: focused codec admission for causal Conv3d caches,
  framewise encode/decode, tiling, and spatial GroupNorm.
- `easyanimate_qwen2vl_text`: Qwen2-VL text/tokenizer path and possible vision
  prompt behavior if future pipelines use it.
- `easyanimate_legacy_v2_v4_v5`: older Alibaba cache entries that use PixArt,
  HunyuanDiT, Transformer2D, Bert/T5, and DDPM rather than current V5.1 classes.
- `easyanimate_lora_adapters`: adapter mutation/load/unload state if official
  EasyAnimate LoRA workflows are selected later.
- `easyanimate_scheduler_variants`: custom timesteps/sigmas and alternate
  schedulers beyond the official FlowMatch config.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel and CPU/GPU offload behavior.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse V5.1 EasyAnimate model index and component configs.
- [ ] Load `EasyAnimateTransformer3DModel` weights for the 7B base config.
- [ ] Accept external Qwen2-VL prompt and negative prompt embeddings plus masks.
- [ ] Implement faithful NCTHW latent contract.
- [ ] Implement per-frame Conv2d patchify/unpatchify parity.
- [ ] Implement EasyAnimate 3D RoPE.
- [ ] Implement timestep embedding and AdaLayerNormZero gates.
- [ ] Implement joint text/video attention with QK norm and video-only RoPE.
- [ ] Implement `EasyAnimateTransformerBlock` image/text FFN and residuals.
- [ ] Implement full transformer forward parity.
- [ ] Implement CFG and guidance rescale over video latents.
- [ ] Implement FlowMatch Euler fixed-`mu=1` dynamic-shift scheduler slice.
- [ ] Implement `AutoencoderKLMagvit` decode with explicit causal cache reset.
- [ ] Add Magvit encode posterior parity for inpaint/control readiness.
- [ ] Add short T2V loop parity with scheduler in host control.
- [ ] Add inpaint mask/video latent pack as a separate stage.
- [ ] Add control video/camera/ref-image latent pack as a separate stage.
- [ ] Add guarded NDHWC/VAE layout optimization only after faithful NCTHW parity.
