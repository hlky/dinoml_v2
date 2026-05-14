# Diffusers Lucy Pipeline Audit

Candidate slug: `lucy`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  decart-ai/Lucy-Edit-Dev
  decart-ai/Lucy-Edit-1.1-Dev

Config sources:
  H:/configs/decart-ai/Lucy-Edit-Dev/model_index.json
  H:/configs/decart-ai/Lucy-Edit-1.1-Dev/model_index.json
  Official Hugging Face repo metadata and component configs read with
  huggingface_hub for both repos:
    model_index.json
    transformer/config.json
    vae/config.json
    scheduler/scheduler_config.json
    text_encoder/config.json
    text_encoder/model.safetensors.index.json
    tokenizer/tokenizer_config.json
    tokenizer/special_tokens_map.json
  The official repos are public and not gated. No authenticated retry blocker
  was encountered. Component configs were not copied into H:/configs because
  this worker owns only this report path.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/lucy/pipeline_lucy_edit.py
  X:/H/diffusers/src/diffusers/pipelines/lucy/pipeline_output.py
  X:/H/diffusers/src/diffusers/pipelines/lucy/__init__.py
  Nearby Wan variant files for separate-candidate inventory:
    X:/H/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py
    X:/H/diffusers/src/diffusers/pipelines/wan/pipeline_wan_i2v.py
    X:/H/diffusers/src/diffusers/pipelines/wan/pipeline_wan_video2video.py
    X:/H/diffusers/src/diffusers/pipelines/wan/pipeline_wan_vace.py
    X:/H/diffusers/src/diffusers/pipelines/wan/pipeline_wan_animate.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_wan.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  Existing nearby reports:
    agents/plans/diffusers/wan/report.md
    agents/plans/diffusers/video_autoencoders/report.md
    agents/plans/diffusers/scheduler_matrix/report.md

External component configs inspected:
  UMT5EncoderModel / T5TokenizerFast configs for google/umt5-xxl-derived text
  encoder and tokenizer slots.

Any missing files or assumptions:
  Lucy currently has one non-deprecated pipeline class, `LucyEditPipeline`.
  This report targets video-to-video edit inference with prompt embeddings,
  condition-video VAE encode, Wan transformer denoising, UniPC flow scheduler,
  and Wan VAE decode. It ignores XLA/NPU/MPS/Flax/ONNX, safety, training,
  callbacks, and multi-GPU/context-parallel paths except where source structure
  exposes shared CPU/CUDA behavior.
```

## 2. Pipeline and component graph

Lucy is a Decart-modified Wan video-edit pipeline. The important family delta is
that it always receives a condition video and appends its encoded latents to the
noisy denoising latents along the channel dimension. There is no separate image
encoder or mask branch in the active Lucy configs.

```text
condition video + prompt / negative prompt
  -> VideoProcessor preprocess: list/tensor/array video to [B,3,F,H,W] NCTHW, normalized
  -> T5TokenizerFast + UMT5EncoderModel prompt embeds [B,L,4096]
  -> AutoencoderKLWan encode condition video, posterior mode
  -> latent mean/std normalization of condition latents
  -> random or caller-provided noisy latents [B,48,T,H/16,W/16]
  -> denoising loop:
       concat noisy + condition latents -> [B,96,T,H/16,W/16]
       WanTransformer3DModel(hidden_states, per-token timestep, prompt embeds)
       optional second negative prompt call for true CFG
       UniPCMultistepScheduler.step on only the noisy latent tensor
  -> latent mean/std denormalization
  -> AutoencoderKLWan decode
  -> VideoProcessor postprocess to np/pt/pil or return latent
```

Required components for the first slice:

| Component | Class | Required? | Notes |
| --- | --- | --- | --- |
| `tokenizer` | `T5TokenizerFast` | Can be external at first | CPU prompt preprocessing, padding/truncation. |
| `text_encoder` | `UMT5EncoderModel` | Can be external at first | Produces `[B,L,4096]`; text encoder is large and separable. |
| `transformer` | `WanTransformer3DModel` | Yes | Active denoiser; `in_channels=96`, `out_channels=48`. |
| `transformer_2` | `WanTransformer3DModel` | No for sampled configs | Optional source surface for two-stage Wan2.2; both Lucy configs set null. |
| `scheduler` | `UniPCMultistepScheduler` | Yes | Official configs use UniPC flow, despite pipeline type annotation naming FlowMatch Euler. |
| `vae` | `AutoencoderKLWan` | Yes | Both encode and decode are required for Lucy edit. |
| `video_processor` | `VideoProcessor` | Constructed in `__init__` | NCTHW preprocessing/postprocessing around VAE. |

Cacheable stages:

- Prompt and negative prompt embeddings can be cached per prompt, sequence
  length, dtype, and tokenizer/text-encoder revision.
- Condition-video latents can be cached per input video, resize size, VAE dtype,
  and VAE revision.
- Scheduler timesteps/sigmas can be cached per scheduler config and inference
  step count.
- RoPE tables in `WanRotaryPosEmbed` are buffer-backed by maximum axis length
  and sliced/expanded per latent grid.

Separate candidate reports:

| Surface | Present for Lucy/Wan? | Class/file anchors | Variant delta |
| --- | --- | --- | --- |
| LoRA/runtime adapters | Yes | `WanLoraLoaderMixin` in `loaders/lora_pipeline.py` | Loads/fuses/unfuses PEFT LoRA on `transformer` or `transformer_2`; also has T2V-to-I2V zero-padding helper for added image K/V layers. |
| Textual inversion | Not wired in Lucy | Generic `TextualInversionLoaderMixin` | Lucy does not inherit textual inversion; tokenizer mutation is not a base Lucy surface. |
| IP-Adapter | Not wired in Lucy | Generic `loaders/ip_adapter.py`; Wan I2V image K/V is source-local, not IP-Adapter | Base Lucy configs have `added_kv_proj_dim=null` and no image encoder. |
| ControlNet | No Lucy-specific class | Wan VACE is the nearest control-style branch | Treat Wan VACE as separate control candidate rather than a Lucy base requirement. |
| T2I-Adapter | No | `pipelines/t2i_adapter` is SD/SDXL-oriented | Not part of Wan/Lucy component graph. |
| GLIGEN | No non-deprecated Lucy support | Deprecated SD GLIGEN only | Out of base Lucy scope. |
| img2img | Lucy itself is video edit, not SD img2img | `LucyEditPipeline` | Uses condition video latents, not strength slicing over an init latent. |
| inpaint | No Lucy-specific inpaint pipeline | Wan VACE docs mention inpaint-like use | Separate Wan VACE/control review if needed. |
| depth2img | No | none in Lucy/Wan | Not present. |
| upscaling | No Lucy upscaler class | none in Lucy/Wan | Not present. |
| Wan video2video | Related separate candidate | `WanVideoToVideoPipeline` | Adds source-video noising and strength/timestep slicing; unlike Lucy, does not append condition latents every denoiser call. |
| Wan I2V | Related separate candidate | `WanImageToVideoPipeline` | Adds first/last image latent condition, mask channels, and optionally CLIPVision image embeds. |
| Wan VACE | Related separate candidate | `WanVACEPipeline`, `WanVACETransformer3DModel` | Adds control hidden states and conditioning scales into special transformer layers. |
| Wan Animate | Related separate candidate | `WanAnimatePipeline`, `WanAnimateTransformer3DModel` | Adds character image, pose/face video encoders, and face-block attention. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer | VAE | Scheduler | Weight metadata |
| --- | --- | --- | --- | --- | --- |
| `decart-ai/Lucy-Edit-Dev` | `LucyEditPipeline`, `boundary_ratio=null`, `expand_timesteps=true` | `in=96`, `out=48`, 30 layers, 24 heads, head dim 128, ffn 14336, patch `[1,2,2]` | `z_dim=48`, `in/out=12`, base 160, decoder base 256, patch size 2, temporal scale 4, spatial scale 16 | UniPC flow, shift 5, order 2, `bh2`, `predict_x0=true`, linspace, final zero | Transformer safetensors size 20,001,594,880 bytes; VAE 2,818,777,808 bytes; text encoder index total 11,361,820,672 bytes. |
| `decart-ai/Lucy-Edit-1.1-Dev` | Same shape config | Same | Same | Same | Same file sizes and text index tensor count in official metadata. |

Active model dimensions:

| Field | Value | Source |
| --- | --- | --- |
| Latent channels | 48 noisy + 48 condition = 96 transformer input channels | transformer and VAE configs plus pipeline concat |
| VAE scale | temporal 4, spatial 16 | VAE config |
| Default call size | `num_frames=81`, `height=480`, `width=832` | pipeline `__call__` defaults |
| Latent grid at defaults | `[B,48,21,30,52]` | inferred from `(F-1)//4+1`, `H/16`, `W/16` |
| Transformer token count at defaults | `21*15*26 = 8190` | patch `[1,2,2]` over latent grid |
| Transformer inner dim | `24*128 = 3072` | transformer config |
| Text context | up to 512 tokens by `__call__`, width 4096 | pipeline default and text config |
| QK norm | RMSNorm across flattened head width | transformer config/source |
| Cross-attn norm | enabled FP32LayerNorm | transformer config/source |
| Image K/V branch | disabled | `image_dim=null`, `added_kv_proj_dim=null` |

Text encoder/tokenizer:

| Component | Key fields |
| --- | --- |
| `UMT5EncoderModel` | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_kv=64`, `d_ff=10240`, gated GELU, vocab 256384, `torch_dtype=bfloat16`, relative attention buckets 32/max distance 128. |
| `T5TokenizerFast` | tokenizer class `T5Tokenizer`, pad/eos/bos/unk present, 300 extra IDs, model max length is effectively unbounded in tokenizer config; pipeline clamps by `max_sequence_length`. |

Scheduler dimensions:

| Field | Value |
| --- | --- |
| Source default scheduler config | `UniPCMultistepScheduler` |
| Pipeline constructor annotation | `FlowMatchEulerDiscreteScheduler` |
| Prediction type | `flow_prediction` |
| Flow sigmas | `use_flow_sigmas=true`, `flow_shift=5.0`, `use_dynamic_shifting=false` |
| Solver | order 2, `solver_type="bh2"`, `predict_x0=true`, `lower_order_final=true` |
| Timesteps | `timestep_spacing="linspace"`, `num_train_timesteps=1000`, final sigma zero |
| Recommended first Dinoml scheduler slice | UniPC flow-prediction with flow sigmas and shift 5, aligned with `scheduler_matrix`; do not start with FlowMatch Euler for Lucy parity. |

Component default reconciliation:

- Pipeline `encode_prompt` source default is `max_sequence_length=226`, but
  `__call__` overrides its public default to 512.
- `AutoencoderKLWan` source defaults are the 16-channel Wan VAE shape
  (`z_dim=16`, spatial scale 8, `in/out=3`, no patch size). Lucy config
  overrides to the TI2V-style 48-channel, spatial-scale-16, patch-size-2 VAE.
- `WanTransformer3DModel` source defaults are 40 layers and 16 channels; Lucy
  config overrides to 30 layers, 96 input channels, 48 output channels.
- `UniPCMultistepScheduler` source default `flow_shift=1.0` and
  `prediction_type="epsilon"` are overridden by Lucy scheduler config.

## 3a. Family variation traps

- Lucy is Wan-derived but not the same as `WanPipeline`: transformer input is
  `[noisy, condition]` channel concat, so first-slice admission must require
  `in_channels == 2 * out_channels`.
- The active VAE uses `patch_size=2`; VAE encode first patchifies `[B,3,F,H,W]`
  into `[B,12,F,H/2,W/2]`, and decode unpatchifies back to RGB. Treat that as a
  VAE-internal pixel packing contract, distinct from transformer patch tokens.
- Pipeline input validation only checks `height % 16 == 0` and `width % 16 == 0`.
  Because the transformer then patches latent H/W by 2, robust Dinoml admission
  should guard the active config to height/width divisible by 32 unless parity
  testing proves Diffusers intentionally tolerates border-dropping shapes.
- `num_frames` is rounded to `k*4+1`; default 81 becomes 21 latent frames.
- `expand_timesteps=true` is active, so timestep is not just `[B]`; it becomes
  `[B, T_lat * (H_lat/2) * (W_lat/2)]`.
- There is true CFG, implemented as a second denoiser call with negative prompt
  embeddings and explicit `uncond + scale * (cond - uncond)`. There is no
  embedded guidance tensor in the active model.
- `transformer_2` and `boundary_ratio` are optional source surfaces but absent
  from official Lucy configs. Do not inflate base scope with dual-transformer
  dispatch.
- Source layout is NCTHW/NCHW. NDHWC/channel-last is a guarded optimization
  only; channel concat, VAE patchify, per-channel mean/std, RMSNorm channel
  axes, and Conv3d/Conv2d weights all need axis-aware rewrites.
- `WanAttnProcessor` supports added image K/V branches in source, but active
  Lucy config disables them.
- `cache_context("cond")` and `cache_context("uncond")` surround model calls;
  the base Wan transformer forward does not require hidden cache tensors for the
  active Lucy path, but Dinoml should keep cache state explicit if future
  attention caching is admitted.

## 4. Runtime tensor contract

For the public default call (`B=1`, `F=81`, `H=480`, `W=832`, `L=512`):

| Boundary | Tensor | Shape/layout/dtype | Notes |
| --- | --- | --- | --- |
| Input video after preprocess | `video` | `[B,3,81,480,832]`, NCTHW, float32 | PIL/np/torch accepted; normalized by `VaeImageProcessor` to `[-1,1]` unless already negative. |
| Prompt embeds | `prompt_embeds` | `[B,512,4096]`, transformer dtype | Tokenizer pads to max length, UMT5 output is trimmed by attention mask then zero padded back. |
| Negative embeds | `negative_prompt_embeds` | `[B,512,4096]` when `guidance_scale>1` | Empty string default if negative prompt omitted. |
| Condition posterior | `vae.encode(video).latent_dist.mode()` | `[B,48,21,30,52]` | `retrieve_latents(..., sample_mode="argmax")`; no sampling noise. |
| Condition latents | `condition_latents` | `[B,48,21,30,52]`, float32 then transformer dtype in concat | Normalized as `(latents - mean) * (1/std)`. |
| Noisy latents | `latents` | `[B,48,21,30,52]`, float32 | Random normal unless caller supplies latents. |
| Transformer input | `latent_model_input` | `[B,96,21,30,52]`, NCTHW | `torch.cat([latents, condition_latents], dim=1)`. |
| Expanded timestep | `timestep` | `[B,8190]` | `mask[0][0][:, ::2, ::2] * t` flattened and batch-expanded. |
| Transformer tokens | internal hidden | `[B,8190,3072]` | Conv3d patch `[1,2,2]`, flatten(2), transpose(1,2). |
| Denoiser output | `noise_pred` | `[B,48,21,30,52]` | Unpatchified model output. |
| Scheduler state | `sigmas`, `model_outputs`, `timestep_list`, `last_sample`, `step_index` | Scalar tables plus history of latent-shaped tensors | UniPC stores sigmas on CPU and moves scalar values to sample device. |
| VAE decode input | denormalized latents | `[B,48,21,30,52]`, VAE dtype | `latents / (1/std) + mean`, equivalent to `latents * std + mean`. |
| Decoded video | `video` | `[B,3,81,480,832]`, NCTHW | VAE clamps to `[-1,1]`; postprocess returns np/pt/pil. |
| Output object | `LucyPipelineOutput.frames` | np stack, torch stack, or list of PIL frames | `output_type="latent"` returns latent tensor directly. |

CPU/data-pipeline work:

- Prompt cleaning, tokenization, UMT5 text encoding if prompt embeddings are not
  supplied.
- Video loading/resizing and host-side output formatting.
- Scheduler table generation can begin as host/runtime metadata.

GPU/runtime work:

- VAE encode of condition video.
- Latent normalization/denormalization.
- Random noise generation or input latent validation.
- Per-step latent concat, per-token timestep expansion, transformer call(s),
  CFG arithmetic, UniPC step, and VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCTHW/NCHW reshape, view, flatten, transpose, permute, contiguous, cat,
  split/chunk, stack, repeat/expand, broadcast.
- VAE pixel patchify:
  `[B,C,F,H,W] -> view(B,C,F,H/p,p,W/p,p) -> permute(0,1,6,4,2,3,5) -> [B,C*p*p,F,H/p,W/p]`.
- VAE unpatchify inverse.
- Transformer patchify/unpatchify:
  `Conv3d(96 -> 3072, kernel=stride [1,2,2])`, flatten tokens, final linear
  to `48*4` patch volume, reshape/permute/flatten to `[B,48,T,H,W]`.
- Per-channel mean/std broadcast over `[B,48,T,H,W]`.
- Per-token timestep expansion from `[B,48,T,H,W]` mask to `[B,T*(H/2)*(W/2)]`.

Convolution/downsample/upsample ops:

- Wan VAE causal Conv3d with asymmetric temporal padding/cache support.
- VAE Conv2d-style spatial attention and resampling after frame flattening.
- VAE nearest-exact upsample, zero pad, avg/downsample and duplicate/up patterns.
- Transformer Conv3d patch embedding.

GEMM/linear ops:

- UMT5 external text encoder if admitted later.
- Wan condition embedding: sinusoidal timestep projection, `TimestepEmbedding`,
  `time_proj`, `PixArtAlphaTextProjection`.
- Attention Q/K/V and output linears; cross-attention K/V from text.
- FeedForward `gelu-approximate` MLP in every transformer block.
- Final `proj_out: Linear(3072 -> 192)`.

Attention primitives:

- Video-token self-attention over 8190 tokens at defaults, 24 heads, head dim
  128, with QK RMSNorm and 3D RoPE.
- Text cross-attention with 512 text tokens, no active text attention mask.
- VAE spatial attention per frame with Conv2d QKV and PyTorch SDPA.
- Added image K/V source branch is available in `WanAttnProcessor` but inactive
  for Lucy configs.

Normalization and adaptive conditioning:

- `FP32LayerNorm` affine and non-affine.
- `torch.nn.RMSNorm` over Q/K flattened head width.
- Wan VAE `WanRMS_norm` over channel axis.
- AdaLayerNorm-style scale/shift/gate from timestep projection:
  six block modulation tensors plus final output scale/shift.

Scheduler and guidance arithmetic:

- True CFG two-call arithmetic over `[B,48,T,H,W]`.
- UniPC flow conversion: with `predict_x0=true`, `x0 = sample - sigma * model_output`.
- UniP/UniC history updates, `expm1`, log alpha/sigma, order warmup, optional
  corrector state.

VAE/postprocessing ops:

- Posterior mean/mode path from `DiagonalGaussianDistribution`.
- Clamp decode output to `[-1,1]`.
- Denormalize to `[0,1]`, NCTHW to per-frame NCHW, then np/pt/pil conversion.

## 6. Denoiser/model breakdown

`LucyEditPipeline.__call__`:

```text
validate video/prompt/size/guidance
round num_frames to k*4+1
encode prompt and negative prompt
set UniPC timesteps
preprocess video -> VAE encode condition -> normalize condition latents
sample noisy latents
for each timestep:
  concat noisy and condition latents on channel axis
  build scalar or token-expanded timestep; Lucy configs use token-expanded
  run transformer with prompt embeds
  optionally run transformer with negative prompt embeds
  apply CFG
  scheduler.step(noise_pred, t, latents)
denormalize latents -> VAE decode -> postprocess
```

`WanTransformer3DModel.forward` for Lucy:

```text
hidden_states [B,96,T,H,W]
-> 3D RoPE table from source latent grid
-> Conv3d patch_embedding, kernel/stride [1,2,2]
-> flatten/transpose to [B,T*(H/2)*(W/2),3072]
-> timestep embedding; token-shaped when expand_timesteps=true
-> text projection from [B,L,4096] to [B,L,3072]
-> 30 x WanTransformerBlock
-> adaptive output FP32LayerNorm
-> Linear to 48*1*2*2 patch volume
-> reshape/permute/flatten back to [B,48,T,H,W]
```

`WanTransformerBlock`:

```text
scale_shift_table + timestep projection -> shift/scale/gate for attention and FFN
FP32LayerNorm -> adaptive scale/shift -> self-attention(QK RMSNorm + 3D RoPE)
gated residual add
optional cross-attn FP32LayerNorm -> text cross-attention
FP32LayerNorm -> adaptive scale/shift -> approximate-GELU FeedForward
gated residual add
```

`AutoencoderKLWan` active Lucy path:

```text
encode:
  [B,3,F,H,W] -> patchify to [B,12,F,H/2,W/2]
  causal WanEncoder3d chunks first frame then 4-frame chunks
  quant_conv -> DiagonalGaussianDistribution
  pipeline takes latent_dist.mode()

decode:
  post_quant_conv
  WanDecoder3d one latent frame at a time with feature cache
  unpatchify from [B,12,F,H/2,W/2] to [B,3,F,H,W]
  clamp [-1,1]
```

## 7. Attention requirements

Primary path:

- `WanAttnProcessor` in `transformer_wan.py` calls `dispatch_attention_fn`
  from `attention_dispatch.py`.
- Self-attention Q/K/V come from video tokens. Query/key are RMS-normalized,
  reshaped to `[B,seq,heads,head_dim]`, then 3D RoPE is applied to both.
- Cross-attention uses video-token queries and projected text K/V; active Lucy
  source passes no attention mask.
- Added K/V image attention exists in source when `add_k_proj` is not null, but
  Lucy configs set `added_kv_proj_dim=null`, so this is separate Wan I2V/VACE
  style scope.

Backend/flash feasibility:

- Parity baseline is `dispatch_attention_fn` with native/eager PyTorch behavior.
- Diffusers attention dispatch supports native SDPA, native flash/efficient/math
  selectors, flash-attn 2/3/varlen/hub, sage, flex, aiter, and xFormers, but
  `WanAttnProcessor` leaves `_attention_backend=None` by default.
- A Dinoml flash-style provider is plausible for Lucy self-attention and text
  cross-attention under strict guards: no attention mask, noncausal, head dim
  128, contiguous `[B,seq,heads,head_dim]`, fp16/bf16/fp32 support as selected,
  RoPE and QK RMSNorm performed before the provider call.
- The active token count at default size is 8190, so attention memory is a real
  bottleneck. However, Dinoml must keep self-attention and cross-attention as
  separate provider shapes, and must not fold inactive added-image K/V logic
  into the base Lucy graph.
- Varlen packing is not required by active source; all prompts are padded back
  to fixed `max_sequence_length`.

## 8. Scheduler and denoising-loop contract

Official Lucy scheduler config is UniPC flow:

```text
num_train_timesteps = 1000
prediction_type = flow_prediction
use_flow_sigmas = true
flow_shift = 5.0
solver_order = 2
solver_type = bh2
predict_x0 = true
timestep_spacing = linspace
final_sigmas_type = zero
lower_order_final = true
thresholding = false
```

`set_timesteps(num_inference_steps, device)` for `use_flow_sigmas` builds
linspace sigmas from 1 down to `1/num_train_timesteps`, applies the static flow
shift:

```text
sigma = flow_shift * sigma / (1 + (flow_shift - 1) * sigma)
timestep = sigma * num_train_timesteps
append final sigma 0
```

Loop-side work:

```text
for t in scheduler.timesteps:
  latent_model_input = cat([latents, condition_latents], dim=1)
  timestep = expanded per token because expand_timesteps=true
  cond = transformer(latent_model_input, timestep, prompt_embeds)
  if guidance_scale > 1:
    uncond = transformer(latent_model_input, timestep, negative_prompt_embeds)
    model_output = uncond + guidance_scale * (cond - uncond)
  latents = scheduler.step(model_output, t, latents)
```

Initial staging should keep `set_timesteps`, `step_index`, `model_outputs`,
`timestep_list`, `last_sample`, `lower_order_nums`, and `this_order` as
host-visible runtime state. Compile/fuse CFG and one scheduler step only after
scalar-table parity and one-step UniPC flow parity are proven.

## 9. Position, timestep, and custom math

- `WanRotaryPosEmbed` splits head dim 128 into temporal, height, and width
  parts: `h_dim=w_dim=2*(head_dim//6)=42`, `t_dim=44`.
- RoPE tables are generated per axis up to `rope_max_seq_len=1024`, expanded
  over `(T, H/2, W/2)`, concatenated, and reshaped to `[1,seq,1,128]`.
- Timestep embedding uses Diffusers `Timesteps(num_channels=256,
  flip_sin_to_cos=True, downscale_freq_shift=0)`, then `TimestepEmbedding`,
  SiLU, and a linear projection to six modulation vectors.
- With `expand_timesteps=true`, the scheduler timestep is repeated per
  transformer token before embedding; this makes modulation token-dependent.
- Text projection uses `PixArtAlphaTextProjection(4096 -> 3072, gelu_tanh)`.
- VAE latent normalization is per-channel mean/std, not SD scalar
  `scaling_factor`: encode normalizes with `(z - mean) / std`; decode reverses
  with `z * std + mean`.

Precomputable:

- RoPE base buffers and axis slices for fixed latent grid.
- Text prompt embeddings and negative embeddings.
- Condition-video latents if the input video is reused.
- Scheduler sigma/timestep tables for fixed step count.

Dynamic:

- Token-expanded timesteps depend on latent token count and current timestep.
- VAE chunk/cache execution depends on frame count and tiling/slicing flags.
- CFG scale is runtime scalar and may vary per request.

## 10. Preprocessing and input packing

Prompt path:

- `prompt_clean` uses ftfy/html/regex whitespace cleanup when ftfy is available.
- Tokenizer uses max-length padding, truncation, special tokens, and attention
  mask.
- Text encoder returns `last_hidden_state`; source trims each sequence to the
  actual mask length and pads with zeros back to the requested max sequence.
- Embeddings are duplicated for `num_videos_per_prompt` by repeat/view.

Video path:

- `VideoProcessor.preprocess_video` accepts list of PIL frames, batched video
  tensors, numpy arrays, or lists. It preprocesses each frame through
  `VaeImageProcessor.preprocess`, stacks videos, then permutes to NCTHW.
- `VaeImageProcessor.preprocess` resizes, converts to RGB where configured,
  converts to tensor, and normalizes `[0,1] -> [-1,1]`.
- Lucy does not build masks or image embeddings. The only conditioning payload
  into the transformer is encoded video latents concatenated with noisy latents.

Packing distinctions:

- VAE patchify is pixel-space packing around the codec:
  RGB `[B,3,F,H,W]` becomes `[B,12,F,H/2,W/2]` before VAE encoder blocks.
- Transformer patchify is latent-token packing:
  `[B,96,T,H/16,W/16]` becomes `[B,T*(H/32)*(W/32),3072]`.
- A layout pass must not merge these two patterns without proving the exact
  reshape/permute order and inverse mapping.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Lucy latent-condition concat admission

Source pattern:

```text
condition = normalize(vae.encode(video).mode())
latent_model_input = cat([latents, condition], dim=1)
transformer(latent_model_input, ...)
```

Replacement pattern:

```text
plan-visible two-input denoiser wrapper:
  noisy_latents [B,48,T,H,W]
  condition_latents [B,48,T,H,W]
  materialize or fused-pack to [B,96,T,H,W] at transformer boundary
```

Preconditions:

- `transformer.in_channels == 2 * transformer.out_channels`.
- No branch mutates condition latents inside loop.
- Source channel axis is NCTHW dim 1; NDHWC pass must rewrite concat axis.

Failure cases:

- Wan I2V/VACE masks/control streams, transformer `in_channels` not equal to
  double output channels, or `condition_latents` dynamic shape mismatch.

Parity test sketch:

- Random latents and condition latents, run source concat + transformer block
  stub versus lowered wrapper with identical input ordering.

### Rewrite: VAE patchify/unpatchify as layout transform

Source pattern:

```text
view -> permute(0,1,6,4,2,3,5) -> contiguous -> view
```

Replacement:

- Metadata-visible pixel-unshuffle/shuffle with exact channel order
  `[C, w_patch, h_patch]` as source expresses it.

Preconditions:

- 5D NCTHW tensor.
- H and W divisible by `patch_size`.
- Downstream VAE Conv3d weights remain in source channel order or are permuted
  by a guarded weight transform.

Failure cases:

- Non-contiguous views, NDHWC without explicit axis rewrite, or patch size not
  matching config.

### Rewrite: Transformer patchify/unpatchify

Source pattern:

```text
Conv3d(kernel=stride=[1,2,2]) -> flatten(2).transpose(1,2)
Linear -> reshape(B,T,H/2,W/2,1,2,2,C) -> permute -> flatten
```

Replacement:

- Lower Conv3d as patch-projection GEMM or keep Conv3d provider; lower final
  linear plus unpatchify as a fused token-to-latent scatter when shape is static.

Preconditions:

- Latent H/W divisible by 2.
- Patch size exactly `[1,2,2]`.
- Output channels and final projection patch volume match config.

Failure cases:

- Odd latent H/W, alternate patch sizes, or consumers expecting token layout.

### Rewrite: CFG arithmetic fusion

Source:

```text
noise_uncond + scale * (noise_pred - noise_uncond)
```

Replacement:

- Single pointwise kernel over `[B,48,T,H,W]`, optionally fused with scheduler
  model-output conversion.

Preconditions:

- Same shape/dtype for cond and uncond outputs.
- Scalar guidance scale.

Failure cases:

- Future per-layer/skip guidance, guidance rescale reductions, or separate
  guidance scales in dual-transformer mode.

### Rewrite: UniPC flow first-order scalar fusion

Source:

```text
x0 = sample - sigma * model_output
UniPC predictor/corrector scalar history update
```

Replacement:

- Host-generated scalar coefficients plus tensor pointwise/einsum kernels.

Preconditions:

- Official Lucy config: `use_flow_sigmas=true`, `solver_order=2`,
  `predict_x0=true`, `thresholding=false`.

Failure cases:

- Dynamic shifting, Karras/exponential/beta sigma modes, thresholding, or
  solver-p composition.

## 12. Kernel fusion candidates

Highest priority:

- Flash/SDPA attention provider for Wan self-attention and cross-attention.
  Lucy default token count is 8190, head dim 128, and 30 layers, making this
  the dominant denoiser cost.
- FP32LayerNorm/RMSNorm + QKV projection + RoPE staging. QK norm and RoPE are
  mandatory before attention and should avoid unnecessary materialization.
- AdaLayerNorm modulation + residual gates around attention and FFN.
- CFG arithmetic over video latents, ideally fused with scheduler input staging.
- VAE decode/encode causal Conv3d and patchify/unpatchify kernels because Lucy
  requires both codec directions.

Medium priority:

- Transformer Conv3d patch embedding as patch-GEMM when shape is static.
- GEGLU/GELU/GELU-tanh feed-forward MLP fusion for Wan blocks and text
  projection.
- UniPC flow scheduler pointwise/history kernels after one-step parity.
- Latent mean/std normalization fused with VAE encode/decode boundary copies.
- NCTHW to NDHWC guarded Conv3d islands in the VAE after source-layout parity.

Lower priority:

- Dual-transformer `boundary_ratio` dispatch for future Wan/Lucy variants.
- Added image K/V attention branch; inactive for Lucy configs.
- VAE tiling/slicing fusion; memory policy first, optimization later.
- LoRA hotswap/fuse path; separate artifact mutation candidate.

## 13. Runtime staging plan

1. Parse Lucy component configs and admit one official config pair as the same
   shape family: `Lucy-Edit-Dev` and `Lucy-Edit-1.1-Dev`.
2. Support externally supplied prompt and negative prompt embeddings
   `[B,512,4096]` and precomputed condition latents `[B,48,T,H/16,W/16]`;
   compile one Wan transformer block for fixed default grid.
3. Add full `WanTransformer3DModel` denoiser with source NCTHW layout,
   token-expanded timesteps, text projection, RoPE, QK RMSNorm, and eager/native
   attention parity.
4. Add Lucy VAE encode path with posterior mode and mean/std normalization so
   condition videos can be supplied as frames.
5. Add one denoising step parity: concat condition/noisy latents, cond/uncond
   transformer calls, CFG, and UniPC flow step in host-visible scheduler state.
6. Add VAE decode and postprocess smoke; keep scheduler loop in Python/host
   until one-step parity is stable.
7. Add full short-loop parity at default shape or a reduced shape satisfying
   the same divisibility constraints.
8. Optimize attention/norm/MLP/patchify/codec kernels and then revisit LoRA and
   Wan variant candidates.

Initial stubs allowed:

- External UMT5 prompt embeddings.
- Precomputed condition latents for denoiser-only tests.
- Scheduler model output fixed/stubbed for scalar-table and step tests.
- VAE tiling/slicing disabled.

## 14. Parity and validation plan

- Config parsing tests for both Decart repos: assert same active dimensions,
  scheduler fields, null `transformer_2`, and `expand_timesteps=true`.
- VAE patchify/unpatchify random tensor roundtrip for `patch_size=2`, including
  exact channel order.
- Wan VAE encode parity on small `[B,3,5,H,W]` tensors: Diffusers posterior
  mode and normalized latent output.
- Wan VAE decode parity on `[B,48,T,H/16,W/16]`, including denormalization and
  clamp behavior.
- RoPE table parity for multiple `(T,H,W)` latent grids.
- Single `WanTransformerBlock` parity with random token tensors, timestep
  projection, text context, and fixed dtype.
- Full denoiser parity at reduced valid size, with `expand_timesteps=true`.
- CFG parity with cond/uncond fixed tensors and scalar scales 1.0 and 5.0.
- UniPC scheduler tests: `set_timesteps` table parity for 1, 4, 10, and 50
  steps; one-step and two-step flow-prediction parity.
- One complete denoising step with Diffusers transformer output captured or
  stubbed, comparing updated latents.
- End-to-end smoke with very short valid frame count and small valid H/W if the
  checkpoint can run locally.

Suggested tolerances:

- Pure scheduler/packing fp32: `rtol=1e-5`, `atol=1e-6`.
- Transformer/VAE bf16/fp16 first pass: `rtol=2e-2`, `atol=2e-2`, then tighten
  per fused kernel and accumulation policy.

## 15. Performance probes

- UMT5 prompt embedding throughput and cache hit rate.
- VAE encode throughput for condition videos by frames and resolution.
- VAE decode throughput by latent frames and resolution.
- One Wan transformer step at 480x832x81: cond-only and cond+uncond CFG.
- Attention backend comparison: native SDPA/math/flash-equivalent over token
  lengths from 1024 to 8190+.
- Norm/QKV/RoPE materialization bandwidth.
- Full denoising loop by step count: 4/10/20/50.
- UniPC scheduler/guidance overhead relative to denoiser time.
- VRAM and temporary workspace for NCTHW source layout versus guarded NDHWC
  codec islands.
- Condition-latent cache timing versus raw video VAE encode timing.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `lucy_lora_adapters`: `WanLoraLoaderMixin` runtime/load-time LoRA state,
  `transformer`/`transformer_2` selection, hotswap, fuse/unfuse, and T2V/I2V
  zero-padding logic.
- `wan_video2video_strength`: `WanVideoToVideoPipeline` strength/timestep
  slicing and source-video noising, distinct from Lucy condition concat.
- `wan_i2v_added_kv`: `WanImageToVideoPipeline` image/mask latents and optional
  CLIPVision added K/V context.
- `wan_vace_control`: VACE control latents, control-layer scales, and
  `WanVACETransformer3DModel`.
- `wan_animate`: character image, pose/face video encoders, face-block cross
  attention, and animate/replace modes.
- `wan_dual_transformer`: `transformer_2`, `boundary_ratio`, and separate
  guidance scales if future Lucy configs use them.
- `wan_autoencoder_policy`: VAE tiling/slicing/chunk-cache memory policy and
  NDHWC codec optimization.
- `scheduler_unipc_flow`: broader UniPC flow modes beyond Lucy's fixed
  `flow_shift=5`, order-2, non-thresholding slice.

Ignored/out-of-scope for this audit:

- Multi-GPU/context parallel and model offload scheduling internals.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX-specific branches.
- Safety checker/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Deprecated GLIGEN and SD-specific adapter/control pipelines except as
  contrast for extension inventory.

## 17. Final implementation checklist

- [ ] Parse Lucy `model_index.json` and component configs for both official repos.
- [ ] Admit active config guards: `z_dim=48`, `in_channels=96`, `out_channels=48`, `patch=[1,2,2]`, `expand_timesteps=true`.
- [ ] Load Wan transformer and VAE weights with explicit dtype policy.
- [ ] Implement VAE `patchify`/`unpatchify` and causal Conv3d encode/decode parity.
- [ ] Implement condition-video VAE encode with posterior mode and per-channel latent normalization.
- [ ] Implement Wan transformer patch embedding, RoPE, token-expanded timestep embedding, text projection, 30 blocks, and unpatchify.
- [ ] Implement or call attention provider for self-attention and text cross-attention; keep eager/native parity fallback.
- [ ] Implement true CFG as two model calls plus pointwise blend.
- [ ] Implement UniPC flow scheduler table generation and order-2 step state.
- [ ] Add one-step denoising parity with externally supplied prompt embeds and condition latents.
- [ ] Add VAE decode/postprocess smoke for `[B,48,T,H/16,W/16]` latents.
- [ ] Add end-to-end short-loop parity at a reduced but valid video shape.
- [ ] Benchmark denoiser attention, VAE encode/decode, CFG+scheduler overhead, and full loop.
