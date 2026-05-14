# Diffusers Sana Video Operator and Integration Report

Target slug: `sana_video`

Status: focused full-audit report for the non-deprecated Sana Video target.

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Efficient-Large-Model/SANA-Video_2B_480p_diffusers
  Efficient-Large-Model/SANA-Video_2B_480p_LongLive_diffusers
  Efficient-Large-Model/SANA-Video_2B_720p_diffusers

Config sources:
  H:/configs/Efficient-Large-Model/SANA-Video_2B_720p_diffusers/model_index.json
  Official Hugging Face raw component configs inspected in-memory for the
  three repos above:
    model_index.json
    transformer/config.json
    vae/config.json
    scheduler/scheduler_config.json
    text_encoder/config.json
    tokenizer/tokenizer_config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/sana_video/pipeline_sana_video.py
  X:/H/diffusers/src/diffusers/pipelines/sana_video/pipeline_sana_video_i2v.py
  X:/H/diffusers/src/diffusers/pipelines/sana_video/pipeline_output.py
  X:/H/diffusers/src/diffusers/pipelines/sana_video/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_sana_video.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_ltx2.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/video_processor.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  Gemma2Model text encoder configs and GemmaTokenizer/GemmaTokenizerFast
  tokenizer configs from the repos above.

Missing files or assumptions:
  No gated config blocker was hit; official raw config reads succeeded without
  authenticated retry. The local cache only contained the 720p model_index.json.
  `SANA-Video_2B_480p_LongLive_diffusers` references
  `SanaVideoCausalTransformer3DModel`, but this checkout exports only
  `SanaVideoTransformer3DModel`; LongLive is therefore a blocked/source-mismatch
  variant in this report. Multi-GPU/context parallel, callbacks/interrupt
  mutation, XLA/NPU/MPS/Flax/ONNX, safety/NSFW, and training/loss/dropout/
  gradient checkpointing paths are out of scope.
```

## 2. Pipeline and component graph

Sana Video is a Gemma-conditioned latent video diffusion family. The base T2V
pipeline wires `GemmaTokenizerFast`/`Gemma2Model`, `SanaVideoTransformer3DModel`,
`DPMSolverMultistepScheduler`, and either a Wan VAE for 480p or an LTX2 VAE for
720p.

```text
prompt / negative prompt preprocessing
  -> GemmaTokenizerFast + Gemma2Model
  -> prompt embeddings [B,L,2304] + prompt attention mask [B,L]
  -> latent noise [B,C,T_lat,H_lat,W_lat] in source NCTHW
  -> denoising loop:
       CFG batch concat
       SanaVideoTransformer3DModel(latents, text embeds, text mask, timestep)
       CFG arithmetic
       DPM-Solver++ flow-prediction scheduler step
  -> latent denormalization with VAE mean/std
  -> video VAE decode
  -> optional resize/crop from aspect-ratio bin
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class/file | First-slice status |
| --- | --- | --- |
| Base T2V pipeline | `SanaVideoPipeline`, `pipeline_sana_video.py` | Use as the primary runtime contract. |
| Denoiser | `SanaVideoTransformer3DModel`, `transformer_sana_video.py` | Required; consumes source NCTHW latent maps and patchifies internally. |
| Scheduler | `DPMSolverMultistepScheduler` | Required first scheduler; official configs use DPM-Solver++ with `flow_prediction` and flow sigmas. |
| VAE | 480p `AutoencoderKLWan`; 720p `AutoencoderKLLTX2Video` | Decode required for output; encode required for I2V. |
| Text encoder | `Gemma2Model` / `GemmaTokenizerFast` | Accept external prompt embeddings first. |

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `sana_video_i2v` | `SanaVideoImageToVideoPipeline`, `pipeline_sana_video_i2v.py` | Adds image preprocessing, VAE encode of first frame, first-latent-frame preservation, and token-shaped conditioning timesteps. |
| `sana_video_longlive` | Config references `SanaVideoCausalTransformer3DModel` | Blocked in this checkout because the referenced class is absent; likely causal/long-video transformer behavior needs a future source-backed report. |
| `sana_video_720p_ltx2_vae` | `AutoencoderKLLTX2Video`, `autoencoder_kl_ltx2.py` | Uses 128-channel latents, temporal compression 8, spatial compression 32, and LTX2 codec blocks. |
| `sana_video_wan_vae` | `AutoencoderKLWan`, `autoencoder_kl_wan.py` | 480p path uses Wan 16-channel video VAE, temporal compression 4, spatial compression 8. |
| `sana_video_lora_adapters` | `SanaLoraLoaderMixin`, transformer/text encoder `PeftAdapterMixin` | Runtime/load-time adapter mutation for transformer and Gemma2 layers. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer class | Latent C | Patch | Tokens for default 81-frame bin | VAE | Scheduler | Status |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `SANA-Video_2B_480p_diffusers` | `SanaVideoPipeline` | `SanaVideoTransformer3DModel` | 16 | `[1,2,2]` | `21*30*52 = 32760` for 480x832 | `AutoencoderKLWan` | DPM++ flow, order 2 | Best first target. |
| `SANA-Video_2B_480p_LongLive_diffusers` | `SanaVideoPipeline` | `SanaVideoCausalTransformer3DModel` | 16 | `[1,2,2]` | Same nominal grid as 480p | `AutoencoderKLWan` | DPM++ flow, order 2 | Blocked: class absent in checkout. |
| `SANA-Video_2B_720p_diffusers` | `SanaVideoPipeline` | `SanaVideoTransformer3DModel` | 128 | `[1,1,1]` | `11*22*40 = 9680` for 704x1280 bin | `AutoencoderKLLTX2Video` | DPM++ flow, order 2 | Variant with different VAE/channel contract. |

Transformer fields:

| Field | 480p | 720p |
| --- | ---: | ---: |
| `num_layers` | 20 | 20 |
| `num_attention_heads * attention_head_dim` | `20 * 112 = 2240` | `20 * 112 = 2240` |
| `num_cross_attention_heads * cross_attention_head_dim` | `20 * 112 = 2240` | `20 * 112 = 2240` |
| `caption_channels` | 2304 | 2304 |
| `mlp_ratio` | 3.0 | 3.0 |
| `qk_norm` | `rms_norm_across_heads` | `rms_norm_across_heads` |
| `rope_max_seq_len` | 1024 per axis | 1024 per axis |
| `sample_size` | 30, selects 480 bins | 22, selects 720 bins |
| `guidance_embeds` | false | false |

VAE fields:

| Field | 480p Wan VAE | 720p LTX2 VAE |
| --- | --- | --- |
| Class | `AutoencoderKLWan` | `AutoencoderKLLTX2Video` |
| Channels | RGB in/out, `z_dim=16` | RGB in/out, `latent_channels=128` |
| Compression | temporal 4, spatial 8 | temporal 8, spatial 32 |
| Boundary stats | 16-element `latents_mean/std` from config | persistent zero mean / one std buffers unless weights override |
| Codec structure | Wan causal Conv3d/residual/down/up stack | LTX2 causal Conv3d, patch size 4, runtime causal encode/decode flags |
| First-slice implication | smaller denoiser channels, more tokens | wider latent channel, fewer tokens, heavier codec |

Text encoder and tokenizer:

| Component | Config facts |
| --- | --- |
| `Gemma2Model` | `hidden_size=2304`, `num_hidden_layers=26`, `num_attention_heads=8`, `num_key_value_heads=4`, `head_dim=256`, `max_position_embeddings=8192`, `model_type=gemma2`. |
| Tokenizer | Model index declares `GemmaTokenizerFast`; tokenizer config says `GemmaTokenizer`. Pipeline accepts both and forces `padding_side="right"`. |
| Prompt sequence | Pipeline default `max_sequence_length=300`, then selects token 0 plus the last `max_sequence_length - 1` tokens. |

Scheduler config for all sampled official repos:

| Field | Value |
| --- | --- |
| Class | `DPMSolverMultistepScheduler` |
| `prediction_type` | `flow_prediction` |
| `algorithm_type` | `dpmsolver++` |
| `solver_order` / `solver_type` | `2` / `midpoint` |
| `use_flow_sigmas` / `flow_shift` | `true` / `8.0` |
| `timestep_spacing` | `linspace` |
| `final_sigmas_type` | `zero` |
| `thresholding` | false |

Recommended first Dinoml scheduler slice: DPM-Solver++ order-2 midpoint with
`flow_prediction`, `use_flow_sigmas=true`, `flow_shift=8.0`,
`final_sigmas_type="zero"`, and host-visible multistep state.

Scheduler support note: the pipeline uses the shared `retrieve_timesteps`
helper with both `timesteps` and `sigmas` arguments, but
`DPMSolverMultistepScheduler.set_timesteps` in this checkout accepts custom
`timesteps` and does not accept a `sigmas` argument. A Dinoml admission path
should therefore reject user-supplied `sigmas` for the official Sana Video
DPM-Solver slice instead of treating the generic pipeline signature as support.

## 3a. Family variation traps

- 480p and 720p are not simple resolution variants: 480p uses 16-channel Wan
  latents and transformer patch `[1,2,2]`; 720p uses 128-channel LTX2 latents
  and patch `[1,1,1]`.
- Source latents and decoded videos are NCTHW. NDHWC is only a guarded
  optimization candidate for local Conv3d/Conv2d islands.
- Transformer patchification happens inside `SanaVideoTransformer3DModel`.
  It is separate from LTX2 VAE codec patching (`patch_size=4`) and Wan VAE
  temporal/spatial compression.
- Self-attention is Sana linear attention with ReLU feature maps and RoPE, not
  standard softmax attention. Cross-attention is standard SDPA/dispatch
  attention with additive prompt masks.
- 480p default token count is larger than 720p because the Wan VAE has weaker
  spatial/temporal compression even though the pixel resolution is lower.
- I2V does not just prepend an image: it VAE-encodes the first frame into
  latent frame 0, sets timestep 0 for conditioned patch tokens, skips scheduler
  updates for frame 0, then concatenates the preserved first latent frame back.
- `SANA-Video_2B_480p_LongLive_diffusers` is not source-backed in this checkout
  because `SanaVideoCausalTransformer3DModel` is absent.
- Pipeline constructor has no scheduler compatibility mutation, but it derives
  VAE temporal/spatial compression differently for Wan/DC versus LTX2 classes.
- T2V constructor typing and official configs agree on
  `DPMSolverMultistepScheduler`; the I2V source annotation names
  `FlowMatchEulerDiscreteScheduler`, but no official I2V config was found, so
  I2V scheduler parity should start from the loaded scheduler object/config
  rather than the annotation alone.

## 4. Runtime tensor contract

For 480p T2V at default 480x832, 81 frames, one video per prompt:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt embeds | `prompt_embeds` | `[B,300,2304]` | Gemma2 hidden states after selection/duplication. |
| Prompt mask | `prompt_attention_mask` | `[B,300]` | Converted in transformer to additive bias `[B,1,300]`. |
| CFG prompt embeds | concat negative/positive | `[2B,300,2304]` | When `guidance_scale > 1`. |
| Latent noise | `latents` | `[B,16,21,60,104]` NCTHW | `T_lat=(81-1)//4+1`, H/8, W/8. |
| Transformer input | `latent_model_input` | `[2B,16,21,60,104]` for CFG | Cast to transformer dtype. |
| Patch tokens | internal | `[B,32760,2240]` | Conv3d patch `[1,2,2]` then flatten/transpose. |
| Cross text | internal | `[B,300,2240]` | `PixArtAlphaTextProjection(2304 -> 2240)` plus RMSNorm. |
| Denoiser output | `noise_pred` | `[2B,16,21,60,104]` | Learned-sigma split branch inactive because out=latent C. |
| Scheduler state | sigmas/history | CPU/GPU scalar tables plus model-output history | DPM-Solver++ order 2. |
| VAE decode input | denormalized latents | `[B,16,21,60,104]` | `latents / (1/std) + mean`, equivalent to `latents*std + mean`. |
| Decoded video | `video` | `[B,3,81,480,832]` NCTHW | Postprocessed to list/NumPy/Torch output. |

For 720p at the 16:9-ish default bin, height/width classify to 704x1280:
latents are `[B,128,11,22,40]`, tokens are `[B,9680,2240]`, and decode uses
the LTX2 VAE with temporal compression 8 and spatial compression 32.

I2V first-frame additions:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Input image | preprocessed | `[B,3,H,W]` | `VideoProcessor.preprocess` / `VaeImageProcessor.preprocess`. |
| VAE encode input | image as video | `[B,3,1,H,W]` | `image.unsqueeze(2)`. |
| Encoded image latent | `image_latents` | `[B,C,1,H_lat,W_lat]` | `retrieve_latents(..., sample_mode="argmax")`. |
| Noisy latents | after insert | `[B,C,T_lat,H_lat,W_lat]` | Frame 0 overwritten with normalized image latent. |
| Conditioning mask | transformer patch grid | `[B,1,T_patch,H_patch,W_patch]` | First patch-frame set to 1; duplicated for CFG. |
| I2V timestep | per patch token | shape matching mask | `t * (1 - conditioning_mask)`, so first frame gets timestep 0. |
| Scheduler sample | non-first frames | `[B,C,T_lat-1,H_lat,W_lat]` | Scheduler updates only `latents[:, :, 1:]`. |

CPU/data-pipeline work: tokenization, Gemma2 when prompt embeds are not
supplied, caption cleaning, complex prompt instruction prefix, resolution-bin
choice, image/video conversion, final PIL/NumPy formatting. GPU/runtime work:
latent generation, VAE encode/decode, transformer denoising, CFG arithmetic,
DPM-Solver++ step, and optional resize/crop.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCTHW latent allocation and validation.
- Conv3d patch embedding and unpatchify:
  `[B,C,T,H,W] -> Conv3d -> [B,2240,T/pt,H/ph,W/pw] -> [B,S,2240]`;
  inverse reshape/permute/flatten after `proj_out`.
- CFG concat/chunk on batch dim and elementwise guidance arithmetic.
- Prompt mask conversion from keep mask to additive attention bias.
- I2V first-frame latent overwrite, patch-grid conditioning mask, timestep
  broadcast, slice/cat preserving frame 0.
- Per-channel latent mean/std broadcast over `[B,C,T,H,W]`.
- Video resize/crop uses `(B*T,C,H,W)` bilinear interpolation and center crop.

Convolution/downsample/upsample ops:

- Transformer patch embedding `Conv3d(C -> 2240, kernel=stride patch)`.
- `GLUMBTempConv`: per-frame Conv2d 1x1, depthwise Conv2d 3x3, gated SiLU,
  Conv2d 1x1, plus temporal Conv2d with kernel `(3,1)` over `[B,C,T,H*W]`.
- Wan VAE causal Conv3d/residual/down/up blocks for 480p.
- LTX2 VAE causal Conv3d, PerChannelRMSNorm, spatial/temporal down/up blocks,
  and codec patch/unpatchify for 720p.

GEMM/linear ops:

- Gemma2 text encoder later; external embeddings first.
- Timestep embedding via `AdaLayerNormSingle`.
- Caption projection `PixArtAlphaTextProjection(2304 -> 2240)` plus RMSNorm.
- Self-attention Q/K/V/out projections.
- Cross-attention Q/K/V/out projections.
- Final `proj_out(2240 -> patch_volume * out_channels)`.

Attention primitives:

- Sana self linear attention over video patch tokens with QK RMSNorm and 3D
  RoPE.
- Text cross-attention with additive prompt mask through `dispatch_attention_fn`.
- No IP-Adapter/added-KV branch in the base Sana Video pipeline.

Normalization and adaptive conditioning:

- LayerNorm without affine before attention and feed-forward.
- RMSNorm for Q/K projections and caption projection.
- `AdaLayerNormSingle` produces per-token timestep modulation chunks.
- `SanaModulatedNorm` before final projection.
- Wan/LTX2 codec channel norms in separate VAE stages.

Scheduler and guidance arithmetic:

- DPM-Solver++ flow-prediction conversion:
  `x0_pred = sample - sigma_t * model_output`.
- Order-2 midpoint multistep history and lower-order warmup/final behavior.
- CFG arithmetic: `uncond + guidance_scale * (text - uncond)`.

Video-specific ops:

- Temporal compression formulas:
  480p Wan `(frames - 1)//4 + 1`; 720p LTX2 `(frames - 1)//8 + 1`.
- 3D RoPE table generation over post-patch time/height/width coordinates.
- I2V first-frame preservation and per-patch timestep masking.

## 6. Denoiser/model breakdown

`SanaVideoTransformer3DModel.forward`:

```text
hidden_states [B,C,T,H,W]
-> 3D RoPE from source latent shape and patch size
-> Conv3d patch_embedding
-> flatten to tokens [B,S,2240]
-> AdaLayerNormSingle(timestep) -> modulation chunks + embedded timestep
-> caption_projection + RMSNorm(text) -> [B,L,2240]
-> N x SanaVideoTransformerBlock
-> SanaModulatedNorm + Linear proj_out
-> reshape/permute unpatchify -> [B,out_channels,T,H,W]
```

`SanaVideoTransformerBlock`:

```text
scale_shift_table + timestep -> shift/scale/gate for MSA and MLP
LayerNorm -> adaptive scale/shift
  -> SanaLinearAttnProcessor3_0 self-attention with 3D RoPE
  -> gated residual
cross-attention to text with additive prompt mask -> residual
LayerNorm -> adaptive scale/shift
  -> unflatten tokens to [B,T,H,W,C]
  -> GLUMBTempConv
  -> flatten tokens
  -> gated residual
```

`GLUMBTempConv` is not a plain MLP. It converts `[B,T,H,W,C]` to per-frame
NCHW Conv2d, applies 1x1 expansion, depthwise 3x3, gated SiLU, 1x1 projection,
then runs a temporal Conv2d over `[B,C,T,H*W]`. This creates a fusion target
distinct from both image Sana `GLUMBConv` and standard transformer FFNs.

## 7. Attention requirements

Primary implementation:

- Self-attention uses local `SanaLinearAttnProcessor3_0` in
  `transformer_sana_video.py`.
- Cross-attention uses local `SanaAttnProcessor2_0`, which routes through
  `dispatch_attention_fn` in `attention_dispatch.py`.

Self linear-attention contract:

- Q/K/V projected from `[B,S,2240]`; 20 heads with head dim 112.
- Q/K RMSNorm is active in official configs.
- Q/K/V reshape to `[B,S,H,D]`.
- ReLU feature map is applied to Q/K before linear attention.
- 3D RoPE applies to rotated Q/K paths.
- Internals upcast rotated Q/K and V to fp32.
- Denominator is `1 / (key.sum(...).T @ query + 1e-15)`.
- Output is cast back to original dtype and passed through output projection.

Cross-attention contract:

- Query is video token sequence; key/value are projected Gemma caption tokens.
- Prompt masks become additive masks. Mask-free flash assumptions are unsafe.
- Eager/native dispatch path defines parity; optimized backends are candidates
  only when additive masks, dtype, head dim 112, and sequence lengths pass.

Flash feasibility:

- Standard flash attention is not a drop-in for Sana self-attention because the
  source operation is linear attention, not softmax attention.
- Cross-attention can use a flash-style provider only under explicit guards.
  Diffusers flash-attn 2/3/4 paths reject normal `attn_mask`; native SDPA/math
  is the safer parity fallback for masked text cross-attention.
- QK RMSNorm and RoPE should remain explicit pre-attention ops unless a fused
  provider exactly models them.

## 8. Scheduler and denoising-loop contract

T2V loop:

```text
timesteps = scheduler.set_timesteps(num_inference_steps, device)
latents = randn([B,C,T_lat,H_lat,W_lat], fp32)
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  timestep = t.expand(latent_model_input.shape[0])
  noise_pred = transformer(latent_model_input, prompt_embeds, mask, timestep)
  if CFG:
    noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

I2V loop differences:

```text
conditioning_mask[:, :, 0] = 1
timestep = t.expand(conditioning_mask.shape) * (1 - conditioning_mask)
noise_pred = transformer(..., timestep=timestep)
noise_pred = noise_pred[:, :, 1:]
pred_latents = scheduler.step(noise_pred, t, latents[:, :, 1:])
latents = cat([latents[:, :, :1], pred_latents], dim=2)
```

Scheduler state should stay host-visible first:

- `set_timesteps` computes flow sigmas using `flow_shift=8.0`.
- `final_sigmas_type="zero"` appends terminal zero sigma.
- `convert_model_output` maps flow prediction to data prediction via
  `sample - sigma_t * model_output`.
- `model_outputs` ring/history, `step_index`, `lower_order_nums`, and final
  lower-order fallback are part of parity.
- Compile the denoiser and pointwise CFG first; compile scheduler pointwise and
  multistep updates only after exact table/state parity is established.

## 9. Position, timestep, and custom math

- `WanRotaryPosEmbed` splits head dim 112 into temporal/height/width pieces:
  `h_dim=w_dim=36`, `t_dim=40`. It builds 1D sin/cos tables per axis and
  expands them over the post-patch video grid.
- Timestep embedding uses `AdaLayerNormSingle(2240)` for official configs.
  `guidance_embeds=false`, so `SanaCombinedTimestepGuidanceEmbeddings` is
  source-available but inactive for sampled official repos.
- `SanaVideoTransformerBlock` expects timestep modulation shaped by token
  groups. T2V uses one group per batch item; I2V supplies a patch-grid timestep
  so first-frame tokens use timestep zero.
- Prompt preprocessing optionally cleans captions, prepends a complex human
  instruction, tokenizes to a longer max length when the instruction is used,
  then selects token 0 plus the last 299 tokens.
- Resolution binning maps requested height/width to fixed 480 or 720 bins based
  on `transformer.config.sample_size`, then resizes/crops decoded video back.

Precompute candidates: prompt embeddings/masks, negative prompt embeddings,
regular RoPE tables for fixed latent grids, scheduler timesteps/sigmas, and
VAE mean/std broadcast tensors.

## 10. Preprocessing and input packing

Text:

- Pipeline forces tokenizer right padding.
- Default `max_sequence_length=300`.
- Negative prompt defaults to empty string.
- Prompt and negative prompt embeddings/masks are repeated for
  `num_videos_per_prompt`.
- CFG concatenates negative then positive embeddings and masks on batch dim.

Video/image:

- Base T2V starts from random NCTHW latents.
- I2V preprocesses one image to NCHW, unsqueezes to `[B,3,1,H,W]`, VAE-encodes
  it, normalizes with VAE stats, and writes it into latent frame 0.
- No pipeline-level packed token layout is exposed. Patch tokens are strictly
  model-internal.
- Decode denormalizes latents, calls VAE decode, optionally resizes/crops
  NCTHW decoded tensors, then postprocesses each video by permuting frames to
  `[T,C,H,W]`.

Layout notes:

- Preserve NCTHW at pipeline and VAE boundaries for first parity.
- Mark transformer patch/unpatchify, I2V frame-0 preservation, VAE latent
  stats, and scheduler slice/cat as no-layout-translation regions until tests
  prove an optimized layout.
- Candidate NDHWC islands are VAE Conv3d blocks only after rewriting Conv3d
  weights, channel norms, posterior split, mean/std broadcast, temporal cache
  axes, and tiling/blending axes.

## 11. Graph rewrite / lowering opportunities

### Rewrite: video patchify/unpatchify

Source pattern: Conv3d patch embedding with kernel/stride equal to
`transformer.config.patch_size`, token flatten, and inverse reshape/permute
after final linear.

Replacement: explicit video patchify/unpatchify op plus Conv3d/GEMM lowering.

Preconditions: source NCTHW layout; T/H/W divisible by patch sizes; patch order
matches source `reshape(..., p_t,p_h,p_w,C) -> permute(0,7,1,4,2,5,3,6)`.
Failure cases: confusing transformer patching with LTX2 VAE codec patching, or
NDHWC translation changing token order.

Parity test: random NCTHW tensors for 480p `[1,16,21,60,104]` and 720p
`[1,128,11,22,40]` through source patch/unpatchify versus lowered form.

### Rewrite: Sana Video linear attention provider

Source pattern: QKV projections -> QK RMSNorm -> ReLU Q/K -> 3D RoPE on Q/K
rotate paths -> fp32 linear attention -> denominator normalize -> output proj.

Replacement: provider-backed `sana_video_linear_attention` or decomposed
fallback with explicit RoPE and norm.

Preconditions: noncausal self-attention, known head dim 112, no attention mask,
ReLU feature map, denominator epsilon `1e-15`, fp32 accumulation preserved.
Failure cases: LongLive causal class once source exists, changed feature map,
cross-attention path, or unsupported RoPE layout.

### Rewrite: GLUMBTempConv fusion

Source pattern: token grid unflatten -> per-frame 1x1 Conv2d -> depthwise 3x3
-> gated SiLU -> 1x1 Conv2d -> temporal Conv2d over `[B,C,T,H*W]`.

Replacement: local NCHW/NCTHW conv island with fused activation/gate and
temporal aggregation.

Preconditions: dense source layout, channel axis known, frame/height/width
unflatten metadata preserved. Failure cases: layout pass does not rewrite the
chunk/channel axis, or temporal conv is treated as a spatial 3D conv without
matching source shape.

### Rewrite: I2V first-frame preservation

Source pattern: VAE encode image -> normalize -> write latent frame 0; timestep
mask makes first patch-frame zero; scheduler updates only frames 1..end.

Replacement: explicit condition-pack/update kernel.

Preconditions: one-frame image condition, source NCTHW layout, patch temporal
size known, no additional condition frames. Failure cases: future multi-frame
conditioning or LongLive causal transformer changes timestep semantics.

### Rewrite: DPM-Solver++ flow step

Source pattern: `flow_prediction` conversion plus order-2 midpoint multistep
update.

Replacement: scheduler state tensors and pointwise kernels around host loop.

Preconditions: official scheduler fields listed above, no stochastic SDE mode,
no dynamic shifting. Failure cases: custom timesteps/sigmas not validated,
lower-order-final behavior ignored, or I2V sample shape excludes frame 0.

## 12. Kernel fusion candidates

Highest priority:

- `SanaLinearAttnProcessor3_0` provider or fused fallback. This is the
  family-defining operation and cannot be replaced by standard flash attention.
- Large GEMMs/Conv3d projection path for QKV, cross-attention, caption
  projection, and final projection at inner dim 2240.
- Adaptive LayerNorm scale/shift/gate plus residual epilogues around attention
  and GLUMBTempConv.
- GLUMBTempConv spatial/depthwise/gate/temporal aggregation fusion.
- DPM-Solver++ flow scheduler pointwise plus CFG arithmetic.

Medium priority:

- Cross-attention backend selection with additive prompt-mask support.
- Video patchify/unpatchify kernels for `[1,2,2]` and `[1,1,1]`.
- I2V first-frame latent encode/normalize/mask/update kernels.
- Wan VAE 16-channel decode and LTX2 VAE 128-channel decode as separate codec
  stages.
- Resolution-bin resize/crop when tensor output parity is required.

Lower priority:

- Gemma2 text encoder compilation; prompt embeddings can be cached externally.
- VAE tiling/slicing and temporal cache policies.
- LoRA/runtime adapter state.
- LongLive causal transformer once source is available.

## 13. Runtime staging plan

Stage 1: Parse official 480p configs and load weights. Accept external Gemma
prompt/negative embeddings and masks.

Stage 2: Implement faithful NCTHW transformer patch/unpatchify, RoPE generation,
Ada timestep modulation, and one `SanaVideoTransformerBlock` parity on reduced
grids.

Stage 3: Implement/decompose `SanaLinearAttnProcessor3_0`; validate self-attn,
cross-attn with mask, GLUMBTempConv, and one full block.

Stage 4: Compile full 480p `SanaVideoTransformer3DModel` denoiser for
`[B,16,21,60,104]`, keeping VAE and scheduler outside the artifact.

Stage 5: Add CFG arithmetic and one DPM-Solver++ flow-prediction scheduler step
with scheduler state in Python/host runtime.

Stage 6: Add Wan VAE decode for 480p or call out to an existing Wan VAE codec
artifact.

Stage 7: Add I2V as a separate stage: image VAE encode, first-frame preserve,
conditioning mask, timestep grid, and non-first-frame scheduler update.

Stage 8: Add 720p after 480p parity: `C=128`, LTX2 VAE, patch `[1,1,1]`, and
larger channel bandwidth.

Stage 9: Revisit LongLive only when a source-backed
`SanaVideoCausalTransformer3DModel` is available.

## 14. Parity and validation plan

- Config/default reconciliation for 480p, 720p, and LongLive blocked variant.
- Prompt embedding/mask duplication and select-index parity.
- 3D RoPE table parity for 480p and 720p latent grids.
- Patch embedding/unpatchify parity for `[1,16,21,60,104]` and
  `[1,128,11,22,40]`.
- `SanaLinearAttnProcessor3_0` parity for fp32/bf16/fp16 with QK RMSNorm.
- `SanaAttnProcessor2_0` cross-attention parity with additive text masks.
- `GLUMBTempConv` parity at small T/H/W and official channel width.
- One `SanaVideoTransformerBlock` parity and full transformer random-tensor
  parity on manageable grids.
- CFG arithmetic parity.
- DPM-Solver++ `set_timesteps`, `convert_model_output`, and one-step parity for
  flow-prediction order-2 midpoint config.
- Wan VAE decode parity for 480p latents, tiling disabled.
- LTX2 VAE decode parity for 720p latents, tiling disabled.
- I2V image encode/first-frame preservation/timestep mask/scheduler slice
  parity.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 start at `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step for 480p token length 32760 and 720p token length 9680.
- Attention time split: self linear attention versus masked text
  cross-attention.
- GLUMBTempConv split: 1x1, depthwise 3x3, gate, temporal Conv2d.
- CFG batch concat memory compared with separate positive/negative calls.
- DPM-Solver++ host/runtime overhead relative to denoiser time.
- Wan VAE 16-channel decode throughput and memory.
- LTX2 VAE 128-channel decode throughput and memory.
- I2V overhead: image VAE encode, first-frame condition pack, timestep mask,
  and frame-slice scheduler update.
- Faithful NCTHW versus guarded NDHWC VAE Conv3d islands.
- Adapter load/fuse/unfuse overhead when LoRA support is admitted.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `sana_video_i2v`: image preprocessing, VAE encode, first-frame latent
  preservation, patch timestep mask, non-first-frame scheduler update.
- `sana_video_longlive`: blocked source/config mismatch for
  `SanaVideoCausalTransformer3DModel`; audit when source exists.
- `sana_video_720p_ltx2_vae`: 128-channel LTX2 VAE decode/encode and tiling.
- `sana_video_wan_vae`: 16-channel Wan VAE decode/encode for 480p and I2V.
- `sana_video_lora_adapters`: `SanaLoraLoaderMixin` and PEFT adapter state for
  transformer/text encoder.
- Scheduler variants beyond official DPM-Solver++ flow order-2 midpoint.

Related surfaces not supported by the base Sana Video family in this checkout:

- Textual inversion: no Sana Video textual-inversion loader mixin; prompt
  embedding override is available but tokenizer/embedding mutation is not a
  pipeline feature here.
- IP-Adapter: no image added-K/V branch in Sana Video base transformer.
- ControlNet/T2I-Adapter/GLIGEN: no Sana Video pipeline/model class in this
  folder; image Sana ControlNet is a separate non-video target.
- Img2img/inpaint/depth2img/upscaling: no non-deprecated Sana Video classes
  beyond T2V and I2V in this checkout.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Sana Video 480p/720p model indexes and component configs.
- [ ] Fence `SanaVideoCausalTransformer3DModel` LongLive configs until source exists.
- [ ] Accept external Gemma prompt and negative prompt embeddings plus masks.
- [ ] Implement source NCTHW latent tensor contracts.
- [ ] Implement Conv3d patch embed and exact unpatchify order.
- [ ] Implement 3D RoPE table generation for post-patch grids.
- [ ] Implement QK RMSNorm plus `SanaLinearAttnProcessor3_0`.
- [ ] Implement masked text cross-attention fallback/provider.
- [ ] Implement Ada timestep modulation, gated residuals, and `SanaModulatedNorm`.
- [ ] Implement `GLUMBTempConv` decomposition/fusion.
- [ ] Add full 480p transformer denoiser parity.
- [ ] Implement CFG arithmetic.
- [ ] Implement DPM-Solver++ flow-prediction order-2 scheduler slice.
- [ ] Add Wan VAE decode boundary for 480p.
- [ ] Add I2V image encode and first-frame preservation as a separate stage.
- [ ] Add 720p LTX2 VAE and 128-channel denoiser variant.
- [ ] Benchmark 480p and 720p denoiser step, attention, GLUMBTempConv, and VAE decode.
- [ ] Add guarded NDHWC/VAE layout optimization only after faithful NCTHW parity.
