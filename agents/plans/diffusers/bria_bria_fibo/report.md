# Diffusers Bria / Bria FIBO Operator and Integration Report

Target slug: `bria_bria_fibo`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  briaai/BRIA-3.2 (official repo referenced by docs/tests; unavailable during config read)
  SahilCarterr/BRIA-3.2 (open mirror used for Bria 3.2 component configs)
  briaai/FIBO
  briaai/Fibo-Edit
  briaai/FIBO-VLM-prompt-to-JSON (prompt-to-JSON helper, not part of first runtime)

Config sources:
  H:/configs had no local BRIA-3.2/FIBO base configs.
  briaai/BRIA-3.2 returned 404 for model_info and component raw reads even with an HF token.
  SahilCarterr/BRIA-3.2 raw JSON configs were inspected as an open mirror.
  briaai/FIBO raw JSON configs were accessible with the authenticated token.
  briaai/Fibo-Edit model_info succeeded and showed component files, but raw config reads returned 403 with the token.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/bria/pipeline_bria.py
  X:/H/diffusers/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo.py
  X:/H/diffusers/src/diffusers/pipelines/bria_fibo/pipeline_bria_fibo_edit.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_bria.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_bria_fibo.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/image_processor.py

External component configs inspected:
  Bria 3.2 mirror T5EncoderModel/T5TokenizerFast configs.
  FIBO SmolLM3ForCausalLM/AutoTokenizer configs.

Any missing files or assumptions:
  Bria 3.2 official configs are blocked/unavailable as 404 despite authenticated retry; mirror facts are labeled.
  Fibo-Edit official configs remain blocked as 403 despite authenticated retry; source and tests define its runtime shape.
  XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training/loss/dropout/gradient checkpointing,
  callbacks/interrupt, and multi-GPU/context parallel paths are out of scope.
```

## 2. Pipeline and component graph

Bria 3.2 and FIBO are Flux-like image denoising pipelines with transformer token
latents, RoPE image/text IDs, FlowMatch Euler scheduling, CFG, and VAE decode.
They are not interchangeable first slices:

```text
Bria 3.2 text prompt
  -> T5TokenizerFast + T5EncoderModel
  -> zero-padded prompt embeddings [B,128..512,4096] + text IDs
  -> random NCHW AutoencoderKL latents [B,4,H/8*2,W/8*2]
  -> pack 2x2 latent cells to tokens [B,(H/16)*(W/16),16]
  -> BriaTransformer2DModel + CFG + FlowMatch/Euler-style scheduler
  -> unpack tokens to NCHW [B,4,H/8,W/8]
  -> AutoencoderKL decode((latents / 0.13025) + shift_factor)
  -> VaeImageProcessor postprocess

FIBO structured JSON prompt
  -> AutoTokenizer + SmolLM3ForCausalLM hidden states
  -> prompt embeddings concat(last,last-1) [B,S,4096]
     plus per-layer hidden states [B,S,2048]
  -> random Wan-VAE latents [B,48,H/16,W/16]
  -> optional token packing:
       default no-patch [B,(H/16)*(W/16),48]
       do_patching [B,(H/32)*(W/32),192]
  -> BriaFiboTransformer2DModel + full joint attention mask + CFG + FlowMatch Euler
  -> unpack tokens to NCHW, unsqueeze temporal frame [B,48,1,H/16,W/16]
  -> Wan VAE latent mean/std unscale + AutoencoderKLWan decode
  -> squeeze frame + VaeImageProcessor postprocess

FIBO edit
  -> same text path
  -> optional image/mask preprocessing, VAE encode source image to tokens
  -> append image latent tokens to denoiser sequence and image IDs with id axis 0 = 1
  -> transformer predicts over combined sequence
  -> scheduler updates only generated latent-token prefix
  -> Wan VAE decode generated latents
```

Required first-slice components:

| Variant | Required components | Cacheable first |
| --- | --- | --- |
| Bria 3.2 | `BriaTransformer2DModel`, `FlowMatchEulerDiscreteScheduler`, `AutoencoderKL`, T5 tokenizer/text encoder | Prompt embeddings, text IDs, image IDs, scheduler sigmas/timesteps, initial latents |
| FIBO | `BriaFiboTransformer2DModel`, `FlowMatchEulerDiscreteScheduler`, `AutoencoderKLWan`, SmolLM3 tokenizer/text encoder | Prompt embeds, per-layer text hidden states, attention mask, text/image IDs, scheduler tables |
| FIBO edit | FIBO components plus image processor and Wan VAE encode | Source image latents, mask-composited image, image IDs, attention mask |

Separate candidate reports:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `bria_lora_ip_adapter_controlnet` | `FluxLoraLoaderMixin`, `PeftAdapterMixin`, `controlnet_block_samples`, IP-Adapter test helpers | Runtime adapter mutation, optional image projection/added K/V processors, and transformer residual injection. |
| `bria_controlnet_3_2` | Bria ControlNet repos and remote-code files referenced by HF search | Official control variants are gated/manual and use separate files; review apart from base Bria. |
| `bria_fibo_edit` | `BriaFiboEditPipeline` | Image/mask preprocessing, Wan VAE encode, appended source-image latent tokens, scheduler update over generated prefix only. |
| `bria_fibo_prompt_to_json` | `FIBO-VLM-prompt-to-JSON` modular pipeline | Prompt authoring helper, not a denoiser/runtime tensor requirement. |
| `bria_fibo_wan_vae_codec` | `AutoencoderKLWan` | 3D causal conv image-as-one-frame codec with patchify/unpatchify, mean/std latent normalization, tiling. |

No in-folder ControlNet, T2I-Adapter, GLIGEN, depth2img, or upscaling pipelines
were found for this family in the Diffusers checkout.

## 3. Important config dimensions

Representative checkpoint/config sweep:

| Config | Source | Pipeline | Text encoder | Transformer dims | Latents | Scheduler |
| --- | --- | --- | --- | --- | --- | --- |
| Bria 3.2 mirror | `SahilCarterr/BRIA-3.2` | `BriaPipeline` | T5, `d_model=4096`, 24 layers, 64 heads | 8 joint + 28 single layers, 24 x 96 heads, inner 2304, joint dim 4096, in/out 16, RoPE axes `[0,48,48]` | AutoencoderKL 4 channels, pack to 16-token dim, scale 0.13025, `shift_factor=None` repaired to 0 | FlowMatch Euler dynamic shift, base seq 256, max seq 4096, shift 0.5..1.15 |
| FIBO | `briaai/FIBO` | `BriaFiboPipeline` | SmolLM3, hidden 2048, 36 layers, 16 Q heads, 4 KV heads | 8 joint + 38 single layers, 24 x 128 heads, inner 3072, joint dim 4096, text layer dim 2048, in/out 48, RoPE axes `[16,56,56]` | AutoencoderKLWan z=48, spatial scale 16, temporal scale 4, patch_size 2, mean/std vectors length 48 | FlowMatch Euler dynamic shift, base seq 256, max seq 4096, shift 0.5..1.15 |
| FIBO edit | source/tests; raw official configs 403 | `BriaFiboEditPipeline` | SmolLM3 path as FIBO | same class; tests use tiny 1+1 layer, 2 x 8 heads | Wan VAE encode/decode; tests use z=16; source default sample size 32 | FlowMatch Euler dynamic shift in source |

Scheduler support: constructors type schedulers as `FlowMatchEulerDiscreteScheduler | KarrasDiffusionSchedulers`.
The active official sampled configs use FlowMatch Euler. Bria 3.2 source also
has branches for `DDIMScheduler` and `EulerAncestralDiscreteScheduler`, plus a
custom `get_original_sigmas` branch for non-dynamic schedulers. Recommended
first Dinoml scheduler slice: FlowMatch Euler dynamic shift with custom sigmas,
matching FIBO and Bria mirror defaults.

## 3a. Family variation traps

- Bria 3.2 packs 4-channel AutoencoderKL latents into 16-wide image tokens.
  FIBO default does not 2x2-pack: it uses 48-channel Wan latents as 48-wide
  tokens. FIBO `do_patching=True` changes the token width to 192.
- Bria 3.2 default 1024 output creates 4096 image tokens:
  `height=1024`, VAE scale 8, pipeline doubles latent H/W before packing, then
  packs to `(128/2)*(128/2)`.
- FIBO default 1024 output creates 4096 image tokens without packing:
  `(1024/16)*(1024/16)`.
- Bria 3.2 source accepts `guidance_scale=0.0` in the slow test; `do_classifier_free_guidance`
  is only `guidance_scale > 1`, so first parity must distinguish no-CFG,
  CFG, and the pipeline's optional normalize/clip postprocessing.
- FIBO uses a full `[B,1,total_seq,total_seq]` additive attention mask from
  text mask plus latent mask; Bria 3.2 has no prompt attention mask in its base path.
- FIBO text conditioning is layer-dependent: each transformer block receives
  the current encoder state's first half concatenated with a projected SmolLM3
  hidden layer. Bria 3.2 only projects the final T5 prompt embeddings once.
- Bria 3.2 official repo referenced by source/docs was not readable; mirror
  configs may drift from official gating.
- FIBO edit raw configs are blocked; source shows image-token concatenation and
  prefix-only scheduler update, but production dimensions should be confirmed
  after gate access.
- Source latent maps and VAE tensors are NCHW/NCTHW. Transformer tokens are
  `[B,S,D]`. NHWC/NDHWC is only a guarded local optimization around conv codec
  islands; token packing order must remain source-faithful.

## 4. Runtime tensor contract

Bria 3.2 mirror, 1024x1024:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt embeds | T5 hidden | `[B,128,4096]` default; max 512 | Source zero-pads each prompt to max length. |
| CFG prompt embeds | concat | `[2B,S,4096]` | Negative prompt becomes zeros if absent/empty. |
| Text IDs | `txt_ids` | `[S,3]` zeros | Repeated per image first, then squeezed to 2D. |
| Initial latent map | before pack | `[B,4,256,256]` for 1024 | Pipeline computes `2*(H/8)`, then packs 2x2. |
| Latent tokens | transformer input | `[B,4096,16]` | Pack order: view B,C,H/2,2,W/2,2; permute B,H,W,C,2,2; reshape. |
| Image IDs | `img_ids` | `[4096,3]` | Axis 1/2 contain latent token row/col. |
| Denoiser output | token velocity/noise | `[B,4096,16]` | Same token contract as input. |
| Decode latent map | unpacked | `[B,4,128,128]` | `latents / scaling_factor + shift_factor`; source repairs missing shift to 0. |
| Decoded image | VAE sample | `[B,3,1024,1024]` NCHW | Postprocess to PIL/NumPy. |

FIBO, 1024x1024 default no-patch:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| Prompt embeds | concat last two SmolLM3 states | `[B,S,4096]`, default max 3000 | Structured JSON prompt expected by docs. |
| Text layers | all hidden states | list of `[B,S,2048]` | Truncated/padded to transformer block count 46. |
| Attention mask | text + latent | `[B,1,S+4096,S+4096]` | Built by outer product, 0 keep and `-inf` ignore. |
| Latent tokens | no patch | `[B,4096,48]` | From NCHW `[B,48,64,64]` permute to BHWC then flatten. |
| Latent tokens | `do_patching=True` | `[B,1024,192]` | 2x2 pack/unpack path. |
| Denoiser output | token update | same as input token width | CFG chunks batch if guidance > 1. |
| VAE decode input | NCTHW | `[B,48,1,64,64]` | Per-channel `latent / (1/std) + mean`; decoded one sample at a time. |
| Decoded image | Wan VAE sample | `[B,3,1,1024,1024]` then squeeze frame | `AutoencoderKLWan` clamps to `[-1,1]`. |

FIBO edit appends `image_latents_bsd` to `latent_model_input` along sequence
dimension and appends `image_ids` with the first ID axis set to 1. Scheduler
step receives `noise_pred[:, : latents.shape[1], ...]`, so only generated
latent tokens are updated.

CPU/data-pipeline work: tokenization, JSON prompt validation/conversion, VLM
prompt helper, PIL/mask compositing, resize/preprocess. GPU/runtime work:
transformer denoiser, attention mask application, CFG, scheduler step, VAE
decode, and FIBO edit VAE encode if admitted.

## 5. Operator coverage checklist

Tensor/layout ops:

- Token pack/unpack for 2x2 Bria/FIBO optional patching with exact
  view/permute/reshape order.
- No-patch FIBO NCHW `[B,C,H,W]` to token `[B,H*W,C]` and inverse.
- Text/latent/image ID generation, concat, squeeze legacy 3D ID inputs.
- CFG batch concat/chunk, prompt/layer padding, per-layer list indexing.
- Full attention mask creation: `einsum("bi,bj->bij")`, `where`, unsqueeze.
- VAE latent mean/std channel broadcast for Wan: vectors over `z_dim`.

GEMM/linear ops:

- `x_embedder`: Bria `16 -> 2304`; FIBO `48 -> 3072` or patch `192 -> 3072`.
- Context embedder `4096 -> inner_dim`.
- FIBO `caption_projection`: 46 separate `2048 -> 1536` bias-free projections.
- Q/K/V and added Q/K/V projections for joint attention.
- FeedForward GELU approximate MLPs and single-block `proj_mlp`/`proj_out`.
- Final projection: Bria `2304 -> 16`; FIBO `3072 -> 48` or `192` if patching is configured.

Attention primitives:

- Joint text-image attention with QK RMSNorm and RoPE over concatenated text/image IDs.
- Bria/FIBO dual-stream blocks with added K/V projections for context tokens.
- FIBO single blocks use `Attention` with `BriaAttnProcessor`, QK RMSNorm, and full mask.
- `dispatch_attention_fn` is the backend path; eager/native attention defines parity.

Normalization and adaptive conditioning:

- RMSNorm on Q/K and added Q/K.
- `AdaLayerNormZero`, `AdaLayerNormZeroSingle`, `AdaLayerNormContinuous`.
- LayerNorm without affine, GELU approximate/tanh, gated residuals.
- Timestep sinusoidal embedding via 256-channel projection + MLP.

VAE/postprocessing ops:

- Bria `AutoencoderKL`: Conv2d, ResNet, GroupNorm, SiLU, attention midblock,
  quant/post-quant 1x1 conv, scaling/shift.
- FIBO `AutoencoderKLWan`: Causal Conv3d, RMSNorm, residual blocks,
  spatial/temporal up/downsample, patchify/unpatchify, clamp, latent mean/std.

## 6. Denoiser/model breakdown

Bria 3.2 transformer:

```text
latent tokens [B,S,16] -> Linear(16,2304)
timestep -> BriaTimesteps(256) -> TimestepEmbedding(2304)
T5 prompt [B,T,4096] -> Linear(4096,2304)
RoPE IDs concat text + image -> cos/sin axes [0,48,48]
8 BriaTransformerBlock:
  AdaLNZero image/context -> joint attention -> gated residual
  LayerNorm -> GELU FF -> gated residual
  context LayerNorm -> GELU FF -> gated residual
28 BriaSingleTransformerBlock:
  concat text+image -> AdaLNZeroSingle -> self attention + MLP -> gated residual
  split text/image
AdaLayerNormContinuous -> Linear(2304,16)
```

FIBO transformer differs in three important places: inner dim is 3072, the
RoPE axes are `[16,56,56]`, and before every block it replaces the second half
of context features with a block-specific projected SmolLM3 layer:

```text
encoder_hidden_states = context_embedder(prompt_embeds)  # [B,T,3072]
for each block i:
  layer_i = caption_projection[i](smollm_hidden_i)       # [B,T,1536]
  encoder_hidden_states = concat(encoder_hidden_states[:,:,:1536], layer_i)
  run joint or single transformer block
```

FIBO edit uses the same denoiser but feeds image latent tokens after the
generated latent tokens. This changes attention sequence length and RoPE IDs
but not the transformer class.

## 7. Attention requirements

- Noncausal joint attention over text and image tokens.
- Q/K/V shape before dispatch: `[B,seq,heads,head_dim]`.
- Bria 3.2 mirror: 24 heads x 96 dim; FIBO: 24 heads x 128 dim.
- QK RMSNorm is mandatory for image and added text projections.
- RoPE applies to both query and key with `sequence_dim=1`; axes are 3D ID
  fields where text IDs are zeros and image IDs carry row/col coordinates.
- FIBO passes a dense additive attention mask `[B,1,total,total]`. A flash-style
  Dinoml provider must support this mask or fall back.
- `attention_dispatch.py` is the primary backend path for the custom Bria
  processors. Source-supported fused projection helpers exist but are not
  required for first parity.
- IP-Adapter kwargs are quietly accepted/ignored by these processors unless a
  different processor is installed; adapter branches are separate candidates.

Flash-style lowering is valid only with guards for dtype, head dim, RoPE layout,
QK RMSNorm placement, no adapter mutation, and FIBO mask support. Bria 3.2
mask-free base attention is the simpler first attention provider target.

## 8. Scheduler and denoising-loop contract

FIBO and the Bria mirror use `FlowMatchEulerDiscreteScheduler` with dynamic
shift:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
seq_len = image token count
mu = calculate_shift(seq_len, base_image_seq_len=256, max_image_seq_len=4096,
                     base_shift=0.5, max_shift=1.15)
retrieve_timesteps(scheduler, sigmas=sigmas, mu=mu)
for t:
  latent_model_input = cat([latents]*2) if CFG else latents
  noise_pred = transformer(...)
  noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

Bria 3.2 source conditionally calls `scheduler.scale_model_input` for
non-FlowMatch schedulers and has DDIM/EulerAncestral branches. Keep those as
separate scheduler-admission work. First Dinoml parity should use FlowMatch
Euler dynamic shift with host-visible timesteps/sigmas and compiled CFG plus
step arithmetic.

## 9. Position, timestep, and custom math

- Timestep embedding: `get_timestep_embedding(..., num_channels=256,
  flip_sin_to_cos=True, downscale_freq_shift=0, max_period=time_theta)` followed
  by `TimestepEmbedding`.
- RoPE: per-axis 1D rotary cos/sin are concatenated across three ID axes.
  `repeat_interleave_real=True` makes cos/sin interleaved to the full axis dim.
- Bria 3.2 optional `normalize` rescales CFG output by global standard
  deviation: `noise_pred * (0.7 * cfg_text_std / noise_pred.std()) + 0.3 * noise_pred`.
- Optional `clip_value` clamps denoiser output before scheduler step.
- FIBO latent unscale uses source formula `latent / (1/std) + mean`, equivalent
  to `latent * std + mean` per channel.

Text embeddings, text IDs, image IDs, RoPE cos/sin for fixed shape, scheduler
tables, and FIBO layer projections can be precomputed or cached if prompt/shape
are fixed. Timestep embeddings and dynamic shift depend on step/sequence length.

## 10. Preprocessing and input packing

Bria 3.2:

- T5 tokenization is per prompt, truncated to `max_sequence_length`, then
  zero-padded manually to fixed length.
- Negative prompt absent/empty creates `zeros_like(prompt_embeds)`.
- Pipeline warns and effectively rounds output dimensions down to divisibility
  by `vae_scale_factor * 2` through latent shape math.
- Pack/unpack is pipeline-level and must not be confused with model-internal patching.

FIBO:

- Prompt is expected to be structured JSON; docs explicitly warn against raw
  freeform prompts.
- SmolLM3 returns hidden states; final two states are concatenated for
  `encoder_hidden_states`, and all hidden states feed per-block projections.
- Positive and negative prompts are padded to the same token length; masks are
  padded in parallel.
- Default max sequence length is 3000.
- FIBO edit validates JSON with `edit_instruction`, can auto-resize image to a
  preferred 1024-ish aspect bucket, can paste mask over image as gray, then VAE
  encodes the source image.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Bria 2x2 latent packing

Source pattern: NCHW latent map -> view `[B,C,H/2,2,W/2,2]` -> permute
`[B,H/2,W/2,C,2,2]` -> reshape `[B,H/2*W/2,4C]`.

Replacement: layout-aware pack kernel or reshape/transpose canonical op.
Preconditions: even latent H/W, contiguous source or represented strides,
consumer expects row-major token order. Failure cases: FIBO default no-patch,
NHWC translation without matching axis rewrite. Test: roundtrip pack/unpack for
Bria C=4 and FIBO C=48.

### Rewrite: joint attention canonicalization

Source pattern: QKV + added QKV projection, QK RMSNorm, concat text/image
sequence, RoPE, dispatch attention, split, output projections.

Replacement: provider-backed joint attention op with explicit context
subsequence and output splits. Preconditions: no adapter processor mutation,
mask support declared, fixed head dim, RoPE axes available. Failure cases:
FIBO dense masks unsupported, IP-Adapter branches, fused projection weights not
materialized.

### Rewrite: FIBO per-layer text projection hoist

Source pattern: each denoiser block projects one SmolLM3 hidden state, then
concats it into context features.

Replacement: precompute all `caption_projection[i](hidden_i)` once per prompt.
Preconditions: prompt layers fixed across denoising steps, LoRA/adapters frozen,
projection weights not timestep-dependent. Failure cases: runtime adapter scale
changes between steps.

### Rewrite: FlowMatch step fusion

Source pattern: CFG arithmetic followed by FlowMatch Euler step over token
latents.

Replacement: fused elementwise token update. Preconditions: scheduler sigma
tables and step index explicit, no stochastic branch, prediction is flow
derivative. Failure cases: non-FlowMatch scheduler branch or custom callback
mutation.

## 12. Kernel fusion candidates

Highest priority:

- Joint QKV/add-QKV projection + QK RMSNorm + RoPE + attention for Bria/FIBO.
- AdaLayerNormZero/Single/Continuous with gated residual epilogues.
- Bria/FIBO pack/unpack/no-patch token layout kernels.
- CFG + FlowMatch Euler step over `[B,S,D]` token latents.
- FIBO dense attention-mask support or a guarded fallback path.

Medium priority:

- FIBO caption projection hoisting and batched GEMM for 46 projections.
- GELU approximate feed-forward fusion.
- Bria AutoencoderKL decode Conv2d/GroupNorm/SiLU conv islands.
- FIBO Wan VAE single-frame decode CausalConv3d/RMSNorm/upscale islands.

Lower priority:

- Non-FlowMatch scheduler swaps in Bria source.
- Bria ControlNet/IP-Adapter/LoRA runtime mutation.
- FIBO edit image VAE encode and mask preprocessing.
- Wan VAE tiling/slicing and temporal multi-frame paths.

## 13. Runtime staging plan

Stage 1: Admit a transformer-denoiser-only Bria 3.2 mirror slice with external
T5 prompt embeddings, fixed 1024 shape, FlowMatch Euler dynamic shift, and
token output parity.

Stage 2: Add token pack/unpack roundtrip and one denoising step with CFG and
FlowMatch scheduler in host-visible control.

Stage 3: Add Bria AutoencoderKL decode boundary with `scaling_factor=0.13025`
and source `shift_factor=0` repair.

Stage 4: Add FIBO base denoiser slice with external SmolLM3 prompt embeddings,
per-layer hidden states, dense attention mask, no-patch tokens, and FlowMatch
Euler.

Stage 5: Add FIBO Wan VAE decode as a separate codec island; keep SmolLM3
outside compiled runtime first.

Stage 6: Add FIBO `do_patching=True` as a shape/layout variant after no-patch
parity.

Stage 7: Add FIBO edit image encode/appended-token path as a separate admission.

First Dinoml admission recommendation: start with `bria_3_2_denoiser_step`
using supplied prompt embeddings and packed latent tokens. It exercises the
shared Bria attention/norm stack without FIBO's dense mask, per-layer text
projection, and Wan VAE complexity. Follow with `fibo_base_denoiser_step` once
masked joint attention is stable.

## 14. Parity and validation plan

- Config parse tests for Bria mirror and FIBO official configs; blocked tests
  for official Bria/Fibo-Edit should assert the exact 404/403 status until
  access is granted.
- Pack/unpack random tensor parity for Bria `[B,4,256,256] -> [B,4096,16]`
  and FIBO `[B,48,64,64]` no-patch plus patch mode.
- RoPE cos/sin parity for axes `[0,48,48]` and `[16,56,56]`.
- Bria/FIBO attention processor parity with QK RMSNorm, RoPE, and FIBO masks.
- One `BriaTransformerBlock` and one `BriaSingleTransformerBlock` parity.
- One `BriaFiboTransformerBlock`, single block, and caption projection parity.
- Full tiny pipeline parity from Diffusers tests.
- FlowMatch Euler dynamic-shift table and one-step parity.
- CFG/no-CFG parity, plus Bria optional normalize/clip tests.
- Bria AutoencoderKL decode parity; FIBO AutoencoderKLWan single-frame decode
  parity and edit encode parity when admitted.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 initially
  `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Bria one denoiser step at token lengths 1024, 4096, and rectangular shapes.
- FIBO one denoiser step at prompt lengths 512, 1000, 2048, 3000 plus 4096 image tokens.
- Attention backend comparison: mask-free Bria vs masked FIBO.
- FIBO caption projection time and memory for 46 layer projections.
- CFG batch factor overhead versus separate positive/negative calls.
- FlowMatch scheduler/CFG pointwise overhead on token latents.
- Bria AutoencoderKL decode throughput at 1024.
- FIBO Wan VAE decode/encode throughput for one-frame NCTHW and patch_size=2.
- NCHW faithful codec path versus guarded NHWC/NDHWC conv island.
- VRAM/workspace with FIBO long prompt masks, especially dense `[S_total,S_total]`.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `bria_controlnet_3_2`: gated/manual Bria 3.2 ControlNet variants and remote-code files.
- `bria_lora_ip_adapter_adapters`: LoRA/PEFT runtime mutation and IP-Adapter
  added-K/V processors.
- `bria_fibo_edit`: source-image VAE encode, masks, appended image tokens, and
  prefix-only scheduler update.
- `bria_fibo_prompt_to_json`: VLM/Gemini prompt-to-JSON helper pipelines.
- `bria_fibo_wan_vae_codec`: Wan VAE single-frame image codec optimization.
- Non-FlowMatch Bria scheduler swaps: DDIM, Euler Ancestral, and broad Karras set.

Genuinely ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Bria mirror and FIBO component configs; track official Bria/Fibo-Edit access blockers.
- [ ] Load `BriaTransformer2DModel` weights and accept external T5 prompt embeddings.
- [ ] Implement Bria 2x2 latent pack/unpack parity.
- [ ] Implement Bria RoPE ID generation for text/image tokens.
- [ ] Implement Bria joint attention with QK RMSNorm and RoPE.
- [ ] Implement AdaLayerNormZero/Single/Continuous gated residual paths.
- [ ] Implement FlowMatch Euler dynamic-shift scheduler table and one-step update.
- [ ] Add Bria CFG/no-CFG one-step parity.
- [ ] Add Bria AutoencoderKL decode boundary with scaling and shift repair.
- [ ] Load `BriaFiboTransformer2DModel` and accept external SmolLM3 embeddings/layers.
- [ ] Implement FIBO no-patch token layout and optional patch layout.
- [ ] Implement FIBO dense additive attention mask support.
- [ ] Hoist or validate FIBO per-layer caption projections.
- [ ] Add FIBO base one-step denoiser parity.
- [ ] Add AutoencoderKLWan single-frame decode parity.
- [ ] Open separate admission for FIBO edit image encode/appended-token path.
