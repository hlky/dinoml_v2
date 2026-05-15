# Diffusers Wan Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  Wan-AI/Wan2.1-T2V-14B-Diffusers
  Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
  Wan-AI/Wan2.2-T2V-A14B-Diffusers
  Wan-AI/Wan2.2-I2V-A14B-Diffusers
  Wan-AI/Wan2.2-TI2V-5B-Diffusers

Config sources:
  H:/configs/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.1-T2V-14B-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.2-T2V-A14B-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.2-I2V-A14B-Diffusers/model_index.json
  H:/configs/Wan-AI/Wan2.2-TI2V-5B-Diffusers/model_index.json
  Component configs were fetched and inspected from official repos with
  `hf download` because the existing local cache only had model_index.json
  files. The fetched component files were not retained because this task's
  owned write path is limited to this report.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/wan/pipeline_wan.py
  diffusers/src/diffusers/pipelines/wan/pipeline_wan_i2v.py
  diffusers/src/diffusers/pipelines/wan/pipeline_wan_video2video.py
  diffusers/src/diffusers/pipelines/wan/pipeline_wan_vace.py
  diffusers/src/diffusers/pipelines/wan/pipeline_wan_animate.py
  diffusers/src/diffusers/pipelines/wan/image_processor.py
  diffusers/src/diffusers/pipelines/wan/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_wan.py
  diffusers/src/diffusers/models/transformers/transformer_wan_vace.py
  diffusers/src/diffusers/models/transformers/transformer_wan_animate.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  UMT5EncoderModel / T5TokenizerFast configs for Wan text encoders.
  CLIPVisionModelWithProjection / CLIPImageProcessor configs for Wan2.1 I2V.

Any missing files or assumptions:
  The main report target is the text-to-video `WanPipeline` plus the image-to-video
  conditioning delta because Wan's first video path is not useful without the I2V
  variant inventory. Wan2.2 I2V model_index declares null image_encoder and
  image_processor, so there were no official component configs to fetch for those
  slots. Multi-GPU/context parallel, callbacks/interrupts, XLA/NPU/MPS/Flax/ONNX,
  safety/NSFW, and training/loss/dropout/gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

Wan is a latent video diffusion family with a 3D transformer denoiser, UMT5 text
conditioning, UniPC flow-prediction scheduler configs, and a Wan-specific
temporal VAE. Diffusers source type hints name `FlowMatchEulerDiscreteScheduler`
in pipeline constructors, but the official sampled configs use
`UniPCMultistepScheduler` with `prediction_type="flow_prediction"` and
`use_flow_sigmas=true`.

```text
prompt / negative prompt
  -> T5TokenizerFast + UMT5EncoderModel token embeddings [B,L,4096]
  -> latent video noise [B,C,T_lat,H/scale,W/scale]
  -> denoising loop:
       WanTransformer3DModel(video latents, timestep, text embeds)
       optional second WanTransformer3DModel below boundary timestep
       true CFG separate negative transformer call
       UniPCMultistepScheduler.step
  -> denormalize latents with Wan VAE mean/std
  -> AutoencoderKLWan decode
  -> VideoProcessor postprocess
```

Image-to-video adds:

```text
input image
  -> VideoProcessor preprocess to NCHW image
  -> first-frame video condition
  -> AutoencoderKLWan encode -> normalized latent condition
  -> concat mask/condition channels with noisy latents
  -> optional CLIPVision hidden-state image embeddings for Wan2.1 I2V only
```

Required first-slice components:

- `WanPipeline` for T2V, with `WanTransformer3DModel`, UMT5 prompt embeddings,
  `UniPCMultistepScheduler`, and `AutoencoderKLWan` decode.
- `WanImageToVideoPipeline` as the first variant candidate because it changes
  transformer `in_channels`, VAE encode requirements, and optional image
  embeddings.
- Prompt embeddings can be external initially; the text encoder is UMT5-XXL.
- VAE decode is required for T2V output; VAE encode is required for I2V.

Separate candidate reports:

| Candidate | Primary classes/files | Runtime delta |
| --- | --- | --- |
| `wan_i2v` | `WanImageToVideoPipeline`, `WanTransformer3DModel` | Adds VAE encode of first/last-frame condition, mask channels, and for Wan2.1 I2V a CLIPVision hidden-state branch. |
| `wan_video2video` | `WanVideoToVideoPipeline` | Adds source video preprocessing, VAE encode, strength/timestep slicing, and video-to-video noising. |
| `wan_vace` | `WanVACEPipeline`, `WanVACETransformer3DModel` | Adds VACE hint/control tensors and special transformer blocks. |
| `wan_animate` | `WanAnimatePipeline`, `WanAnimateTransformer3DModel` | Adds character image, pose/face video encoders, and face-block cross attention. |
| `wan_lora_adapters` | `WanLoraLoaderMixin` | Runtime/load-time adapter mutation, including T2V-to-I2V LoRA key expansion helpers. |
| `wan_ti2v_5b` | `WanPipeline` with `expand_timesteps=true` | Uses per-token timestep expansion and 48-channel latents/transformer outputs. |
| `wan_2_2_dual_transformer` | `transformer` + `transformer_2`, `boundary_ratio` | Switches high-noise and low-noise transformer modules and may use separate guidance scales. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Transformer channels | Layers/heads/head dim | FFN | VAE z/base | Scheduler | Special |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| Wan2.1 T2V 1.3B | `WanPipeline` | 16 -> 16 | 30 / 12 / 128 | 8960 | z=16, base=96 | UniPC flow, shift 3 | Smallest first target. |
| Wan2.1 T2V 14B | `WanPipeline` | 16 -> 16 | 40 / 40 / 128 | 13824 | z=16, base=96 | UniPC flow, shift 3 | Same operator shape, wider. |
| Wan2.1 I2V 14B 480P | `WanImageToVideoPipeline` | 36 -> 16 | 40 / 40 / 128 | 13824 | z=16, base=96 | UniPC flow, shift 3 | `image_dim=1280`, `added_kv_proj_dim=5120`, CLIPVision. |
| Wan2.2 T2V A14B | `WanPipeline` | 16 -> 16 | 40 / 40 / 128 | 13824 | z=16, base=96 | UniPC flow, shift 3 | `transformer_2`, `boundary_ratio=0.875`. |
| Wan2.2 I2V A14B | `WanImageToVideoPipeline` | 36 -> 16 | 40 / 40 / 128 | 13824 | z=16, base=96 | UniPC flow, shift 3 | `transformer_2`, `boundary_ratio=0.9`, no CLIP image encoder. |
| Wan2.2 TI2V 5B | `WanPipeline` | 48 -> 48 | 30 / 24 / 128 | 14336 | z=48, base=160, decoder=256 | UniPC flow, shift 5 | `expand_timesteps=true`, VAE spatial scale 16, `patch_size=2`. |

Common transformer fields:

| Field | Wan2.1/2.2 common value | Runtime effect |
| --- | --- | --- |
| `patch_size` | `[1,2,2]` | Conv3d patchify over H/W only, then token sequence. |
| `text_dim` | 4096 | UMT5 hidden width. |
| `freq_dim` | 256 | Sinusoidal timestep embedding width. |
| `qk_norm` | `rms_norm_across_heads` | Query/key RMSNorm before attention. |
| `cross_attn_norm` | true | FP32LayerNorm before cross-attention. |
| `rope_max_seq_len` | 1024 | Per-axis 3D RoPE table limit. |

VAE dimensions:

| VAE config | Channels | Compression | Notes |
| --- | --- | --- | --- |
| Wan2.1/2.2 16-channel VAE | `in=3`, `out=3`, `z_dim=16`, `base_dim=96` | temporal 4, spatial 8 by source defaults | Uses 16-element `latents_mean/std`, causal Conv3d, quant/post-quant conv. |
| Wan2.2 TI2V 5B VAE | `in=12`, `out=12`, `z_dim=48`, `base_dim=160`, `decoder_base_dim=256` | temporal 4, spatial 16 | Uses `patch_size=2`, residual blocks, 48-element `latents_mean/std`. |

Text encoder:

| Component | Key fields |
| --- | --- |
| `UMT5EncoderModel` | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_ff=10240`, gated GELU, vocab 256384. |
| `T5TokenizerFast` | T5 tokenizer class, pad/eos/bos/unk, 300 extra IDs. Pipeline max sequence defaults to 512 in `__call__`, but `encode_prompt` source default is 226. |

Recommended first Dinoml scheduler slice:

- Start with `UniPCMultistepScheduler` configured as the official Wan configs:
  `prediction_type="flow_prediction"`, `solver_order=2`, `solver_type="bh2"`,
  `predict_x0=true`, `use_flow_sigmas=true`, `flow_shift=3.0`,
  `timestep_spacing="linspace"`, `final_sigmas_type="zero"`.
- Do not start from `FlowMatchEulerDiscreteScheduler` even though constructor
  annotations mention it; that would miss checkpoint parity for the sampled
  official repos.

## 3a. Family variation traps

- Source video/latent layout is NCTHW. Treat NDHWC as a guarded optimization
  only, with explicit axis rewrites for Conv3d, RMSNorm, masks, mean/std, and
  VAE temporal chunking.
- `num_frames` must satisfy `(num_frames - 1) % vae_scale_factor_temporal == 0`;
  the pipeline rounds to `k*4 + 1`.
- Height/width must be multiples of `vae_scale_factor_spatial * patch_size`,
  usually 16 for Wan2.1/2.2 16-channel models and 32 for TI2V 5B.
- Transformer `in_channels` differs by variant: 16 for T2V, 36 for normal I2V
  because latents are concatenated with mask/condition channels, and 48 for
  TI2V 5B.
- Wan2.1 I2V uses CLIPVision hidden states as added image K/V context; Wan2.2
  I2V official config declares null image encoder and instead relies on latent
  conditioning.
- Wan2.2 A14B uses two same-shaped transformer modules with a scheduler-timestep
  boundary; `guidance_scale_2` only makes sense when `boundary_ratio` is set.
- TI2V 5B uses `expand_timesteps=true`: timestep becomes `[B,seq_len]` and
  adaptive norm modulation becomes token-dependent.
- The Wan VAE is not the shared 2D `AutoencoderKL`: it has causal Conv3d,
  temporal caches, temporal chunk decode/encode, and per-channel latent mean/std.
- VAE tiling/slicing and causal feature caches are memory/runtime strategies;
  first parity should leave them explicit and disabled unless validating that
  mode.

## 4. Runtime tensor contract

For a typical 480x832, 81-frame Wan2.1 T2V run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | `[B,512,4096]` | UMT5 last hidden state, padded after trimming by attention mask. |
| negative embeds | `negative_prompt_embeds` | `[B,512,4096]` | Used for true CFG when `guidance_scale > 1`. |
| noisy latents | `latents` | `[B,16,21,60,104]` NCTHW | `T_lat=(81-1)/4+1`, H/8, W/8. |
| timestep | `timestep` | `[B]` or `[B,seq]` | `[B,seq]` only for `expand_timesteps`. |
| transformer tokens | after Conv3d patch | `[B,21*30*52,inner]` | Patch `[1,2,2]`; inner is heads * head_dim. |
| transformer output | `noise_pred` | `[B,16,21,60,104]` | Unpatchified in model forward. |
| scheduler state | `sigmas`, history | CPU/GPU scalar tables plus model output history | UniPC has step index, model_outputs, timestep_list, last_sample. |
| VAE decode input | denormalized latents | `[B,16,21,60,104]` | `latents / (1/std) + mean`, equivalent to `latents * std + mean`. |
| decoded video | `video` | `[B,3,81,480,832]` NCTHW | Then postprocessed to requested output. |

I2V conditioning:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| input image | preprocessed | `[B,3,H,W]` | From `VideoProcessor.preprocess`. |
| video condition | before VAE encode | `[B,3,T,H,W]` | First frame plus zeros or first/last frame. |
| latent condition | normalized VAE latents | `[B,16,T_lat,H/8,W/8]` | Repeated to batch and normalized with mean/std. |
| mask channels | `mask_lat_size` | `[B,4,T_lat,H/8,W/8]` | Built from frame mask and temporal factor. |
| transformer input | concat | `[B,36,T_lat,H/8,W/8]` | `16 noisy + 4 mask + 16 condition`. |
| Wan2.1 image embeds | CLIPVision hidden states | `[B,257,1280]` source hidden state | Projected to model dim and used as added image K/V. |

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW latent/video tensors; view, flatten, transpose, permute, reshape.
- Conv3d patchify and unpatchify in `WanTransformer3DModel`.
- VAE `patchify`/`unpatchify` for TI2V 5B VAE with 2D patch channels.
- Cat/split for CFG calls, I2V mask/condition channels, image/text context.
- Per-channel latent mean/std broadcast over `[B,C,T,H,W]`.
- Frame masks, `repeat_interleave`, temporal chunking, zero padding.

### Convolution/downsample/upsample ops

- Transformer patch embedding: `Conv3d(in_channels -> inner_dim, kernel=[1,2,2], stride=[1,2,2])`.
- Wan VAE causal `Conv3d`, `Conv2d` spatial resample after flattening `B*T`,
  zero padding, nearest-exact upsample, temporal conv in up/downsample3d.
- `AvgDown3D` and `DupUp3D` reshape/permute/mean/repeat patterns.

### GEMM/linear ops

- UMT5 external text encoder if admitted later.
- Time/text/image embedding MLPs: Timesteps -> `TimestepEmbedding`, `time_proj`,
  `PixArtAlphaTextProjection`, optional `WanImageEmbedding`.
- Attention Q/K/V, cross-attention K/V, added image K/V, output projections.
- FeedForward GELU-approximate MLP.
- Final `proj_out: Linear(inner_dim -> out_channels * patch_volume)`.

### Attention primitives

- Video latent self-attention over transformer tokens with 3D RoPE.
- Text cross-attention with no attention mask in first path.
- Optional I2V added-K/V image attention branch.
- VAE spatial self-attention inside `WanAttentionBlock`, per frame.

### Normalization and adaptive conditioning

- `FP32LayerNorm` with affine and non-affine variants.
- `torch.nn.RMSNorm` across Q/K projection width.
- Wan VAE custom RMS norm over channel axis.
- AdaLayerNorm-style scale/shift/gate from timestep projection.

### Scheduler and guidance arithmetic

- True CFG: separate positive and negative denoiser calls and
  `uncond + scale * (cond - uncond)`.
- UniPC flow-prediction conversion and multistep predictor/corrector state.
- Dual-transformer stage switch by `t >= boundary_ratio * num_train_timesteps`.

### Video-specific ops

- Temporal latent compression `(frames - 1) // 4 + 1`.
- Causal Conv3d padding/cache contract.
- Decode one latent frame/chunk at a time while carrying VAE feature cache.
- Video postprocess from NCTHW tensor to list/array output.

## 6. Denoiser/model breakdown

`WanTransformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> WanRotaryPosEmbed from source latent shape
-> Conv3d patch_embedding, kernel/stride [1,2,2]
-> flatten to tokens [B,T*(H/2)*(W/2),inner]
-> timestep embedding and text projection
-> optional image embedding projection and concat before text context
-> N x WanTransformerBlock
-> adaptive output LayerNorm
-> Linear to patch volume
-> reshape/permute/flatten back to [B,out_channels,T,H,W]
```

`WanTransformerBlock`:

```text
scale_shift_table + timestep projection -> six modulation tensors
FP32LayerNorm -> scale/shift -> self-attention with QK RMSNorm + 3D RoPE
gated residual add
optional FP32LayerNorm -> text cross-attention, plus optional image added-K/V branch
FP32LayerNorm -> scale/shift -> GELU-approximate FeedForward
gated residual add
```

For Wan2.1 T2V 1.3B, `inner_dim=12*128=1536`. For 14B/A14B,
`inner_dim=40*128=5120`. TI2V 5B uses `inner_dim=24*128=3072` and token-wise
timestep modulation.

## 7. Attention requirements

Primary implementation is `WanAttnProcessor` in `transformer_wan.py`, calling
`dispatch_attention_fn` from `attention_dispatch.py`.

- Self-attention: mask-free, noncausal, over video patch tokens, Q/K/V shape
  `[B,seq,heads,head_dim]`, QK RMSNorm before RoPE.
- Cross-attention: query from video tokens, key/value from UMT5 projected text
  embeddings. The pipeline does not pass a text mask after padding.
- I2V added image branch: when `added_kv_proj_dim` is not null, the processor
  assumes combined context contains image tokens before 512 text tokens, runs a
  separate attention from video queries to image K/V, then adds it to text
  attention output.
- VAE attention: per-frame spatial self-attention with Conv2d QKV and PyTorch
  scaled-dot-product attention.

Flash-style constraints:

- Base Wan self-attention and cross-attention are plausible Dinoml
  flash-style provider candidates when dtype/head_dim/sequence limits pass.
- RoPE and QK RMSNorm are pre-attention operations and must remain explicit.
- Cross-attention and added image K/V are separate provider shapes; the I2V
  added branch cannot be silently folded into base text cross-attention unless
  the context split and output add are represented.
- The eager/native `dispatch_attention_fn` path is the parity definition.

## 8. Scheduler and denoising-loop contract

Official Wan configs inspected use `UniPCMultistepScheduler`:

```text
prediction_type = flow_prediction
use_flow_sigmas = true
flow_shift = 3.0 for 1.3B/14B/A14B, 5.0 for TI2V 5B
solver_order = 2
solver_type = bh2
predict_x0 = true
timestep_spacing = linspace
final_sigmas_type = zero
```

Pipeline loop:

```text
scheduler.set_timesteps(num_inference_steps, device)
for t in timesteps:
  choose transformer or transformer_2 by boundary_timestep
  timestep = t.expand(B) or per-token expanded timestep
  noise_pred = transformer(latents, timestep, prompt_embeds)
  if CFG:
    noise_uncond = transformer(latents, timestep, negative_prompt_embeds)
    noise_pred = noise_uncond + guidance * (noise_pred - noise_uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

First Dinoml slice should keep scheduler iteration and UniPC history as
host-visible state, compile one denoiser call plus explicit CFG/scheduler
pointwise/update kernels only after one-step parity is proven. The second
transformer switch is a separate plan-visible dispatch guard, not a hidden model
mutation.

## 9. Position, timestep, and custom math

- `WanRotaryPosEmbed` splits head dimension into temporal, height, and width
  portions: `h_dim=w_dim=2*(head_dim//6)`, `t_dim=head_dim-h_dim-w_dim`.
- RoPE tables are generated per axis and expanded over the patch grid to
  `[1,seq,1,head_dim]`.
- Timestep embeddings use Diffusers `Timesteps` with `flip_sin_to_cos=True` and
  `downscale_freq_shift=0`, then `TimestepEmbedding` and SiLU + linear.
- Timestep projection produces six block modulation vectors and final output
  scale/shift uses the base time embedding.
- TI2V 5B per-token timesteps flow through the same embedding path after
  flatten/unflatten.
- Wan VAE latent normalization is mean/std, not SD-style scalar scale/shift:
  decode uses `latents = latents / (1/std) + mean`.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- Prompt cleaning and T5 tokenization.
- UMT5 text encoder if not supplied as cached prompt embeddings.
- Image loading/resizing/normalization for I2V and optional CLIPVision path.
- Video output conversion.

GPU/runtime work:

- Generate or accept NCTHW latent noise.
- Build I2V first/last-frame condition and masks.
- VAE encode/decode and per-channel normalization.
- Transformer patchify/unpatchify.
- Denoising loop tensor arithmetic.

Source axis-sensitive patterns:

- `VideoProcessor.preprocess` returns NCHW images; the VAE consumes NCTHW video.
- I2V masks use temporal dim 2 and channel dim 1.
- VAE temporal chunking slices dim 2 and caches causal conv features.
- NDHWC translation must rewrite Conv3d weights, RMSNorm channel axis, mask
  concatenation channel axis, and mean/std broadcast shape.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Conv3d patchify/unpatchify

Source pattern:

```text
Conv3d kernel=stride [1,2,2] -> flatten(2).transpose(1,2)
Linear output -> reshape [B,T,H2,W2,pt,ph,pw,C] -> permute -> flatten
```

Replacement:

```text
video latent map <-> video token matrix with explicit patch layout
```

Preconditions: NCTHW source layout, H/W divisible by 2, patch size exactly
`[1,2,2]`, no layout translation inside the sequence unless the inverse is
rewritten. Failure cases: TI2V VAE `patch_size=2` is a different codec-level
patchification and must not be confused with transformer patch embedding.

### Rewrite: Wan attention prelude

Source pattern:

```text
Q/K/V projections -> RMSNorm(Q,K) -> unflatten heads -> optional RoPE -> SDPA
```

Replacement: canonical attention provider input with explicit QK norm and RoPE
pre-ops.

Preconditions: mask-free, noncausal, supported head dim 128, dtype supported,
no added-K/V branch unless lowered as a separate attention/add. Failure cases:
Wan2.1 I2V added image branch, backend sequence/head limitations.

### Rewrite: I2V condition concat

Source pattern:

```text
latent_condition = (vae.encode(video_condition).mode - mean) * (1/std)
latent_model_input = cat([latents, mask_lat_size, latent_condition], dim=1)
```

Replacement: plan-visible condition pack kernel.

Preconditions: normal I2V 16-channel VAE, 4 mask channels from temporal factor,
NCTHW layout. Failure cases: `expand_timesteps` path and 48-channel TI2V model
use different condition/mask math.

### Rewrite: UniPC flow conversion

Source pattern:

```text
x0_pred = sample - sigma * model_output
multistep UniP/UniC history update
```

Replacement: explicit scheduler state tensors and pointwise kernels around a
host-controlled multistep loop.

Preconditions: official Wan scheduler fields above. Failure cases: alternate
sigma schedules, dynamic shifting, custom `solver_p`, or non-flow prediction.

## 12. Kernel fusion candidates

Highest priority:

- Large GEMMs for Q/K/V, cross-attention K/V, FFN, text projection, and
  `proj_out`; 14B uses 5120-wide blocks.
- QK RMSNorm + RoPE + attention provider prelude.
- Ada scale/shift/gate + residual epilogues around attention and FFN.
- Conv3d patchify/unpatchify and video layout transforms.
- Wan VAE causal Conv3d + RMSNorm + SiLU + residual blocks.

Medium priority:

- I2V condition pack: VAE latent normalization, mask construction, channel cat.
- UniPC scheduler conversion/update kernels with flow prediction.
- VAE temporal chunk decode with feature-cache residency represented explicitly.
- Per-channel latent mean/std pointwise kernels.

Lower priority:

- Wan2.1 I2V added image K/V fusion.
- VAE tiled encode/decode blend kernels.
- LoRA fuse/unfuse/runtime adapter state.
- VACE and Animate side encoders/attention.

## 13. Runtime staging plan

Stage 1: Parse official Wan configs and load weights for
`Wan2.1-T2V-1.3B-Diffusers`; accept external prompt embeddings.

Stage 2: Implement transformer patchify/unpatchify, 3D RoPE, timestep/text
conditioning, one `WanTransformerBlock`, and full transformer forward parity
for random tensors.

Stage 3: Add true CFG as two explicit denoiser calls and one fixed-timestep
noise prediction parity.

Stage 4: Implement the official UniPC flow-prediction scheduler slice with
host-visible multistep state; validate one denoising step.

Stage 5: Add `AutoencoderKLWan` decode for z=16, base=96, no tiling/slicing.

Stage 6: Run a short deterministic T2V loop with scheduler in host control and
VAE decode.

Stage 7: Add Wan2.1 I2V condition encode/concat, then separately the CLIPVision
added-K/V branch.

Stage 8: Add Wan2.2 dual-transformer boundary dispatch and `guidance_scale_2`.

Stage 9: Add TI2V 5B `expand_timesteps`, z=48 VAE, and VAE `patch_size=2`.

## 14. Parity and validation plan

- Config/default reconciliation tests for omitted VAE scale factors and
  transformer defaults.
- Patch embedding/unpatchify parity for `[B,16,21,60,104]`.
- 3D RoPE table/index parity for several frame/height/width grids.
- One `WanTransformerBlock` parity at 1.3B and 14B widths.
- Full `WanTransformer3DModel` forward parity for T2V 1.3B random tensors.
- CFG arithmetic parity with fixed positive/negative embeddings.
- UniPC `set_timesteps` and one `step` parity for flow-prediction config.
- Wan VAE decode parity for `[B,16,21,60,104] -> [B,3,81,480,832]`.
- I2V condition pack parity: VAE encode mode, mean/std normalization, mask
  channels, channel concat.
- Dual-transformer boundary selection parity for Wan2.2.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 start at
  `rtol=2e-2, atol=2e-2`, then tighten per provider.

## 15. Performance probes

- One denoiser step by latent grid and frame count: 480x832x81, 720P variants,
  and small synthetic grids.
- Attention backend comparison by token length and head count.
- 1.3B vs 14B transformer block time split: GEMM, attention, FFN.
- CFG one-call vs two-call total loop cost.
- UniPC scheduler overhead compared with denoiser time.
- Wan VAE decode throughput and memory, with and without temporal chunking.
- I2V VAE encode and condition-pack overhead.
- NDHWC guarded VAE/Conv3d candidate vs faithful NCTHW.
- Dual-transformer Wan2.2 load/offload and stage switch overhead.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `wan_i2v`: first-frame/last-frame conditioning, CLIPVision for Wan2.1 I2V,
  and 36-channel transformer inputs.
- `wan_video2video`: source video VAE encode, strength/timestep slicing.
- `wan_vace`: VACE hint/control streams and `WanVACETransformer3DModel`.
- `wan_animate`: character image, pose/face video encoders, animate/replacement
  contracts.
- `wan_lora_adapters`: `WanLoraLoaderMixin`, including T2V-to-I2V LoRA
  expansion.
- `wan_2_2_dual_transformer`: high/low-noise transformer scheduling and
  separate guidance scale.
- `wan_ti2v_5b`: per-token timesteps, z=48/in=48/out=48 transformer, z=48 VAE
  with spatial scale 16 and codec patching.
- `wan_vae_tiling_slicing`: explicit memory-policy report for tiled/chunked
  encode/decode.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Wan model index and component configs from `H:/configs/Wan-AI`.
- [ ] Load `WanTransformer3DModel` and `AutoencoderKLWan` weights for Wan2.1 T2V 1.3B.
- [ ] Accept external UMT5 prompt and negative prompt embeddings.
- [ ] Implement NCTHW Conv3d patchify/unpatchify parity.
- [ ] Implement Wan 3D RoPE and timestep/text embedding path.
- [ ] Implement `WanTransformerBlock` attention, cross-attention, FFN, and gated residuals.
- [ ] Implement QK RMSNorm + RoPE + attention provider fallback.
- [ ] Implement true CFG two-call arithmetic.
- [ ] Implement official UniPC flow-prediction scheduler slice.
- [ ] Implement Wan VAE z=16 decode with latent mean/std.
- [ ] Add one-step denoising parity and short-loop smoke.
- [ ] Add I2V VAE encode/condition-mask concat as a separate stage.
- [ ] Add Wan2.1 I2V CLIPVision added-K/V branch.
- [ ] Add Wan2.2 dual-transformer boundary dispatch.
- [ ] Add TI2V 5B per-token timestep and z=48 VAE support.
- [ ] Add guarded NDHWC/Conv3d/VAE layout optimization tests after faithful NCTHW parity.
