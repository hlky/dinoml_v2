# Diffusers AuraFlow and Lumina Operator and Integration Report

Target slug: `auraflow_lumina`

Runtime scope: focused audit for the remaining image DiT families AuraFlow,
Lumina Next, and Lumina 2. First Dinoml slice should admit one denoiser-step
artifact with externally supplied text embeddings, source NCHW latents, explicit
FlowMatch Euler scheduler state, and VAE decode as an adjacent boundary.

Ignored per user scope: XLA/NPU/MPS, Flax/ONNX, safety/NSFW, training/loss/
dropout/gradient checkpointing, multi-GPU/context parallel, callbacks/interrupt.

## 1. Source basis

```text
Diffusers commit/version:
  diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  fal/AuraFlow
  fal/AuraFlow-v0.2
  fal/AuraFlow-v0.3
  Alpha-VLLM/Lumina-Next-SFT-diffusers
  Alpha-VLLM/Lumina-Image-2.0
  duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40
  neta-art/Neta-Lumina-diffusers
  Alpha-VLLM/Lumina-DiMOO, model_index only; not a Diffusers latent DiT pipeline.

Config sources:
  Local model indexes:
    H:/configs/fal/AuraFlow*/model_index.json
    H:/configs/Alpha-VLLM/Lumina-Next-SFT-diffusers/model_index.json
    H:/configs/Alpha-VLLM/Lumina-Image-2.0/model_index.json
    H:/configs/Alpha-VLLM/Lumina-DiMOO/model_index.json
    H:/configs/duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40/model_index.json
    H:/configs/neta-art/Neta-Lumina-diffusers/model_index.json
  Official raw HF component configs inspected without writing to H:/configs:
    fal/AuraFlow*, Alpha-VLLM/Lumina-Image-2.0,
    duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40,
    neta-art/Neta-Lumina-diffusers.
  Lumina Next official raw URLs returned Git-LFS pointer text for component
  configs; real configs were read from the authenticated/local HF cache under:
    C:/Users/user/.cache/huggingface/hub/models--Alpha-VLLM--Lumina-Next-SFT-diffusers/snapshots/0ee5ec90043acf5cb41fe96274af36eb7fad8d95/

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py
  diffusers/src/diffusers/pipelines/lumina/pipeline_lumina.py
  diffusers/src/diffusers/pipelines/lumina2/pipeline_lumina2.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/auraflow_transformer_2d.py
  diffusers/src/diffusers/models/transformers/lumina_nextdit2d.py
  diffusers/src/diffusers/models/transformers/transformer_lumina2.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/single_file_model.py
  diffusers/src/diffusers/loaders/single_file_utils.py

External component configs inspected:
  UMT5EncoderModel/LlamaTokenizerFast for AuraFlow.
  GemmaModel/GemmaTokenizerFast for Lumina Next.
  Gemma2Model/GemmaTokenizerFast for Lumina 2.

Any missing files or assumptions:
  H:/configs did not contain component configs for the selected repos. Official
  AuraFlow and Lumina 2 configs were accessible. Lumina Next scheduler/VAE and
  text/transformer configs were available through the Hugging Face local cache
  after raw URLs returned LFS pointer text; no repo path remains blocked for the
  selected latent DiT audit. Lumina-DiMOO is inventoried as a separate non-DiT
  candidate because its model_index declares tokenizer + VQModel only.
```

## 2. Pipeline and component graph

AuraFlow:

```text
prompt -> LlamaTokenizerFast + UMT5EncoderModel
  -> masked UMT5 prompt embeddings [B,L,2048]
  -> NCHW latent noise [B,4,H/8,W/8]
  -> denoising loop: AuraFlowTransformer2DModel + CFG + FlowMatchEuler step
  -> AutoencoderKL decode(latents / 0.13025)
  -> VaeImageProcessor postprocess
```

Lumina Next:

```text
prompt -> GemmaTokenizerFast + GemmaModel hidden_states[-2]
  -> prompt embeddings/masks [B,L,2048]
  -> NCHW latent noise [B,4,H/8,W/8]
  -> per-step 2D RoPE from timestep-dependent scaling
  -> LuminaNextDiT2DModel + learned-sigma trim + partial-channel CFG + FlowMatch Euler
  -> AutoencoderKL decode(latents / 0.13025)
  -> postprocess
```

Lumina 2:

```text
system prompt + prompt -> GemmaTokenizerFast + Gemma2Model hidden_states[-2]
  -> prompt embeddings/masks [B,L,2304]
  -> NCHW latent noise [B,16,H/8,W/8], even latent H/W
  -> FlowMatch Euler schedule with custom sigmas and computed mu
  -> conditional transformer call; optional separate unconditional call
  -> optional normalization-based CFG; scheduler.step
  -> AutoencoderKL decode((latents / 0.3611) + 0.1159)
  -> postprocess
```

Required first-slice components:

| Family | Pipeline | Denoiser | Text encoder/tokenizer | Scheduler | VAE |
| --- | --- | --- | --- | --- | --- |
| AuraFlow | `AuraFlowPipeline` | `AuraFlowTransformer2DModel` | `UMT5EncoderModel`, `LlamaTokenizerFast` | `FlowMatchEulerDiscreteScheduler` | `AutoencoderKL` |
| Lumina Next | `LuminaPipeline` / deprecated `LuminaText2ImgPipeline` | `LuminaNextDiT2DModel` | `GemmaModel`, `GemmaTokenizerFast` | `FlowMatchEulerDiscreteScheduler` | `AutoencoderKL` |
| Lumina 2 | `Lumina2Pipeline` / deprecated `Lumina2Text2ImgPipeline` | `Lumina2Transformer2DModel` | `Gemma2Model`, `GemmaTokenizerFast` | `FlowMatchEulerDiscreteScheduler` | `AutoencoderKL` |

Cacheable stages: prompt embeddings and masks, AuraFlow learned-position subset
indices per latent grid, Lumina/Lumina 2 RoPE tables for fixed prompt mask and
latent grid, scheduler timesteps/sigmas per step count, and VAE decoded latents
when output type is `latent`.

Separate candidate reports:

| Surface | Status |
| --- | --- |
| LoRA/runtime adapters | AuraFlow has `AuraFlowLoraLoaderMixin`; Lumina 2 has `Lumina2LoraLoaderMixin`. Both mutate transformer and text encoder adapter state and should be separate reports. Lumina Next has no family-local LoRA mixin. |
| Textual inversion | Not wired in inspected AuraFlow/Lumina pipelines. |
| IP-Adapter | Not wired in these base pipelines or processors. |
| ControlNet/T2I-Adapter/GLIGEN | No family-local base variants inspected. Shared attention blocks do not make these first-slice requirements. |
| img2img/inpaint/depth/upscale | No inspected family-local variant files for AuraFlow/Lumina/Lumina 2. |
| Single-file conversion | `single_file_model.py` and `single_file_utils.py` include AuraFlow and Lumina 2 conversion paths; separate loader candidate. |
| Lumina-DiMOO | `LuminaDiMOOPipeline` model index has tokenizer + VQModel, not the image DiT latent pipeline. Separate candidate if requested. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Family | latent C | VAE scale/shift | sample | patch | layers | heads x dim | hidden | text width | scheduler |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| `fal/AuraFlow` | AuraFlow | 4 | 0.13025 / null | 64 | 2 | 4 MMDiT + 32 single | 12 x 256 | 3072 | 2048 | FlowMatch Euler shift 1.73 |
| `fal/AuraFlow-v0.2` | AuraFlow | 4 | 0.13025 / null | 64 | 2 | 4 + 32 | 12 x 256 | 3072 | 2048 | FlowMatch Euler shift 1.73 |
| `fal/AuraFlow-v0.3` | AuraFlow | 4 | 0.13025 / null | 64 | 2 | 4 + 32 | 12 x 256 | 3072 | 2048 | FlowMatch Euler shift 1.73; `pos_embed_max_size=9216` |
| `Alpha-VLLM/Lumina-Next-SFT-diffusers` | Lumina Next | 4 | 0.13025 / null | 128 | 2 | 24 | 32 x 72 | 2304 | 2048 | FlowMatch Euler shift 1.0 |
| `Alpha-VLLM/Lumina-Image-2.0` | Lumina 2 | 16 | 0.3611 / 0.1159 | 128 | 2 | 2 ctx + 2 noise + 26 joint | 24 x 96, 8 KV | 2304 | 2304 | FlowMatch Euler shift 6.0, custom sigmas + `mu` |
| `duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40` | Lumina 2 community | 16 | 0.3611 / 0.1159 | 128 | 2 | same as Lumina 2 | 24 x 96, 8 KV | 2304 | 2304 | same config as official Lumina 2 |
| `neta-art/Neta-Lumina-diffusers` | Lumina 2 community | 16 | 0.3611 / 0.1159 | 128 | 2 | same as Lumina 2 | 24 x 96, 8 KV | 2304 | 2304 | same config as official Lumina 2 |

Important source/config defaults:

| Field | AuraFlow | Lumina Next | Lumina 2 |
| --- | --- | --- | --- |
| Pipeline default image | 1024x1024 | `sample_size * 8` = 1024 | `sample_size * 8` = 1024 |
| Required H/W multiple | `vae_scale_factor * 2` = 16 | 16 | 16 |
| Pipeline prompt max | 256 | 256 | 256, hard error above 512 |
| Denoiser output | 4 channels | 8 channels when `learn_sigma=true`; pipeline keeps first 4 | 16 channels |
| Guidance | true CFG by batch concat | true CFG by batch concat, only first 3 output channels guided | separate conditional/unconditional calls with optional normalization |
| Text class | `UMT5EncoderModel` | `GemmaModel` | `Gemma2Model` |
| Text hidden state | encoder output `[0]` | `hidden_states[-2]` | `hidden_states[-2]` |
| Scheduler first slice | FlowMatch Euler static shift | FlowMatch Euler static shift | FlowMatch Euler custom sigmas + `mu`; static shift effective because config says dynamic false |

Recommended first Dinoml scheduler slice: FlowMatch Euler deterministic update
`prev = sample + (sigma_next - sigma) * model_output` with static shift,
custom sigmas, and explicit `step_index`. AuraFlow is the smallest parity target
because it has one denoiser call per CFG batch and no RoPE/mask-length packing.

## 3a. Family variation traps

- AuraFlow uses learned absolute position embeddings selected from a centered
  square grid; Lumina uses RoPE. Do not share positional code paths blindly.
- AuraFlow patchify is linear over manually flattened NCHW patches, not Conv2d.
- Lumina Next and Lumina 2 use GQA by repeating KV heads to query-head count.
- Lumina Next has `learn_sigma=true` and pipeline trims `chunk(2, dim=1)[0]`;
  Lumina 2 has 16 output channels and no learned-sigma trim.
- Lumina Next CFG guides only `noise_pred[:, :3]`, then copies the guided
  3-channel epsilon back into both batch halves before selecting one half.
- Lumina 2 does separate conditional and unconditional transformer calls rather
  than concatenating CFG batch. `cfg_trunc_ratio` can disable CFG late in the
  loop, and `cfg_normalization` adds vector-norm reductions.
- Lumina 2 `prepare_latents` produces latent H/W directly as even `height/8`
  and `width/8`; there is no pipeline-level 2x2 packing despite the comment.
- VAE scale differs materially: AuraFlow/Lumina Next use 4-channel SDXL-style
  `0.13025`; Lumina 2 uses 16-channel FLUX-style `0.3611` plus `shift_factor`.
- Source semantic layout is NCHW at VAE and denoiser boundaries, BNC inside
  transformer blocks. NHWC is only a guarded local optimization for VAE and
  patch/unpatch islands.
- Lumina 2 uses Python loops over batch examples to build variable-length joint
  text+image sequences from masks. First parity should fix prompt length or
  preserve this as host/data-pipeline work.

## 4. Runtime tensor contract

For 1024x1024, one image per prompt:

| Boundary | AuraFlow | Lumina Next | Lumina 2 |
| --- | --- | --- | --- |
| Text ids/mask | `[B,256]` LlamaTokenizerFast | `[B,256]` GemmaTokenizerFast | `[B,256]` GemmaTokenizerFast, optional system prompt prefix |
| Prompt embeds | `[B,256,2048]`, masked by expand multiply | `[B,L,2048]` | `[B,L,2304]` |
| CFG composition | cat negative, positive on batch | cat positive, negative on batch | separate denoiser calls |
| Latents | `[B,4,128,128]` NCHW | `[B,4,128,128]` NCHW | `[B,16,128,128]` NCHW |
| Timestep passed to model | `[B]`, `t / 1000` | `[B]`, `1 - t / 1000` | `[B]`, `1 - t / 1000` |
| Patch tokens | `[B,4096,3072]` | `[B,4096,2304]` | `[B,4096,2304]` |
| Text tokens inside model | register 8 + projected `[B,L,3072]` | Gemma `[B,L,2048]` used as cross context | projected Gemma2 `[B,L,2304]` |
| Denoiser output | `[B,4,128,128]` | `[B,8,128,128]` then first 4 | `[B,16,128,128]` |
| Scheduler tensors | `timesteps [S]`, `sigmas [S+1]`, step index | same | same plus custom sigmas and computed `mu` |
| Decode input | `latents / 0.13025` | `latents / 0.13025` | `(latents / 0.3611) + 0.1159` |

Patchify/unpatchify:

- AuraFlow patchify:
  `[B,C,H,W] -> view [B,C,H/P,P,W/P,P] -> permute [B,H/P,W/P,C,P,P] ->
  flatten patch -> flatten grid -> Linear(P*P*C -> hidden) + learned pos`.
- Lumina Next patchify:
  same manual NCHW patch flatten through `LuminaPatchEmbed`, `Linear(P*P*C ->
  hidden)`, full valid mask, and RoPE slice `[1,H/P*W/P,D]`.
- Lumina 2 patchify:
  `view [B,C,H/P,P,W/P,P] -> permute [B,H/P,W/P,P,P,C] -> flatten patch ->
  flatten grid -> Linear(P*P*C -> hidden)`, with separate RoPE for context,
  image, and joint sequences based on prompt mask lengths.
- Unpatchify for all three is token-to-NCHW with `view`/`permute`/`einsum`-like
  axis order. NHWC lowering must rewrite channel chunk/slice axes explicitly.

CPU/data-pipeline work: tokenization, prompt cleaning/system prompt, mask length
processing, Lumina 2 joint sequence packing, scheduler table generation, PIL
postprocess. GPU/runtime work: denoiser, CFG arithmetic, optional CFG norm,
scheduler step pointwise update, VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW random latent allocation, scaling, concat/chunk/split, per-step scalar
  broadcast, reshape/view/permute/flatten, stack, gather for Lumina 2 RoPE ids,
  bool masks and mask expansion.
- Patchify/unpatchify with source NCHW semantics and exact row-major image-token
  order.
- Lumina 2 variable-length joint sequence copy/scatter into padded `[B,S,D]`.

GEMM/linear ops:

- AuraFlow patch Linear(16 -> 3072), context Linear(2048 -> 3072), timestep
  MLP 256 -> 3072, attention projections, SwiGLU-like FFN, final Linear(3072 -> 16).
- Lumina Next patch Linear(16 -> 2304), caption/time embed, attention
  projections with GQA, `LuminaFeedForward`, final Linear(2304 -> 32).
- Lumina 2 patch Linear(64 -> 2304), caption Linear(2304 -> 2304), refiner and
  joint attention/FFN blocks, final Linear(2304 -> 64).

Attention primitives:

- AuraFlow joint text-image SDPA with QK FP32 LayerNorm and added Q/K/V
  projections, plus single-block self-attention on text+image sequence.
- Lumina Next self-attention and cross-attention through `LuminaAttnProcessor2_0`,
  RoPE, GQA, bool additive/keep mask, and proportional attention scale.
- Lumina 2 self/joint attention through `Lumina2AttnProcessor2_0`, RoPE over
  text/image/joint axes, GQA, optional mask only when prompt lengths differ.

Normalization/adaptive conditioning:

- FP32 LayerNorm, RMSNorm, `AdaLayerNormZero`, `LuminaRMSNormZero`,
  `LuminaLayerNormContinuous`, tanh gates, SiLU-conditioned scale/gate paths.

Scheduler/VAE/postprocessing:

- FlowMatch Euler set_timesteps, sigma shifting, custom sigmas, deterministic
  step update, optional Lumina 2 CFG vector norm reduction.
- AutoencoderKL decode Conv2d/GroupNorm/SiLU/up blocks; encode is a future
  variant need, not base text-to-image.

## 6. Denoiser/model breakdown

AuraFlow:

```text
NCHW latents
  -> AuraFlowPatchEmbed + learned position subset
  -> Timesteps(scale=1000) + TimestepEmbedding
  -> context Linear(2048 -> 3072) + 8 register tokens
  -> 4 AuraFlowJointTransformerBlock layers
  -> concat text+image -> 32 AuraFlowSingleTransformerBlock layers
  -> AuraFlowPreFinalBlock + Linear -> unpatchify NCHW
```

`AuraFlowJointTransformerBlock` uses AdaLayerNormZero on image and context,
joint SDPA over concatenated context+image Q/K/V, FP32 LayerNorm, gated
attention residuals, and SiLU-gated two-linear FFNs. Single blocks use the same
adaptive norm/attention/FFN pattern on a combined sequence and return only image
tokens after the block stack.

Lumina Next:

```text
NCHW latents + image RoPE
  -> LuminaPatchEmbed
  -> LuminaCombinedTimestepCaptionEmbedding
  -> 24 LuminaNextDiTBlock layers
  -> LuminaLayerNormContinuous + Linear -> unpatchify NCHW
```

Each block performs adaptive RMSNormZero self-attention with RoPE, cross-attends
to Gemma context with query RoPE only, gates cross-attention per head using a
learned tanh gate, applies one shared output projection, then a gated SiLU FFN.

Lumina 2:

```text
NCHW latents + Gemma2 context/mask
  -> timestep embedding + caption RMSNorm/Linear
  -> Lumina2RotaryPosEmbed builds context/image/joint RoPE and patch tokens
  -> x_embedder Linear
  -> 2 context refiner blocks without modulation
  -> 2 noise refiner blocks with modulation
  -> padded joint text+image sequence
  -> 26 modulated joint transformer blocks
  -> LuminaLayerNormContinuous + Linear -> per-example unpatchify -> stack
```

## 7. Attention requirements

AuraFlow uses `AuraFlowAttnProcessor2_0` from `attention_processor.py`; fused
QKV is source-supported through `FusedAuraFlowAttnProcessor2_0` but is a runtime
mutation, not checkpoint format. Joint blocks concatenate context and image
Q/K/V before noncausal SDPA, then split outputs back into image and context.
There is no attention mask in the base AuraFlow denoiser path.

Lumina Next uses `LuminaAttnProcessor2_0`. It applies optional QK norm,
reshapes to heads/KV heads, applies RoPE to query and optionally key, repeats KV
heads for GQA, expands a bool attention mask to `[B,heads,query,key]`, then uses
SDPA. Self-attention passes image mask and RoPE for Q/K; cross-attention passes
text mask, image RoPE for Q, and no key RoPE.

Lumina 2 uses local `Lumina2AttnProcessor2_0` in `transformer_lumina2.py`. It
has the same SDPA/GQA/RoPE shape but accepts an optional mask; joint layers only
use a mask when per-example text+image sequence lengths differ. Proportional
attention scaling from `base_sequence_length` exists in the processor but
Lumina 2 pipeline does not set it.

Flash-style constraints:

- AuraFlow joint attention is a valid dense noncausal flash candidate only if
  the provider supports concatenated context+image output splitting and QK norm
  pre-processing. Eager/SDPA defines parity.
- Lumina Next/Lumina 2 require RoPE, GQA, and bool masks. A provider must accept
  head dim 72 for Lumina Next and 96 for Lumina 2, plus repeated KV or native
  GQA. Mask-free assumptions are unsafe for cross-attention and variable-length
  Lumina 2 joint sequences.
- `attention_dispatch.py` is not the primary implementation path for these
  models; `attention_processor.py` and the Lumina 2 local processor are.

## 8. Scheduler and denoising-loop contract

All selected latent DiT pipelines use `FlowMatchEulerDiscreteScheduler`.

Common step:

```text
set_timesteps(num_steps, sigmas?, mu?)
model_output = denoiser(...)
model_output = loop-side CFG / sign transform
prev = sample + (sigma_next - sigma) * model_output
```

AuraFlow passes scheduler timesteps `t` to `step` but passes `t / 1000` to the
model. It uses CFG batch concat and does not call `scale_model_input`.

Lumina Next reverses model timestep with `1 - t / num_train_timesteps`, builds
per-step RoPE scaling, trims learned sigma channels, guides only the first
three channels, negates `noise_pred`, then calls scheduler step.

Lumina 2 synthesizes default sigmas as `linspace(1.0, 1 / steps, steps)`, calls
`calculate_shift(...)` for `mu`, and passes both `sigmas` and `mu` to
`set_timesteps`. Its sampled scheduler config has `use_dynamic_shifting=false`,
so `mu` is accepted but not used by the scheduler unless configs change. It
negates model output before `step`.

Keep `set_timesteps`, step index, schedule validation, and Lumina 2 CFG
truncation host-visible first. Compile CFG arithmetic, optional CFG norm, sign
negation, and deterministic FlowMatch pointwise update after denoiser parity.

## 9. Position, timestep, and custom math

- AuraFlow learned positions are a parameter `[1,pos_embed_max_size,hidden]`
  indexed from a centered square grid. v0.3 changes `pos_embed_max_size` from
  4096 to 9216.
- AuraFlow timestep uses `Timesteps(num_channels=256, scale=1000,
  flip_sin_to_cos=True)` after pipeline already divides scheduler `t` by 1000.
- Lumina Next computes `get_2d_rotary_pos_embed_lumina(head_dim, 384, 384,
  linear_factor, ntk_factor)` each step. The factors switch around
  `scaling_watershed` based on reversed timestep.
- Lumina 2 precomputes 1D RoPE tables for axes `(caption,row,col)` and gathers
  per-example frequencies based on effective caption mask length plus image
  grid IDs.
- `LuminaFeedForward` is SiLU-gated: `linear_2(silu(linear_1(x)) *
  linear_3(x))`, with hidden dimension rounded to `multiple_of`.

Precompute fixed prompt embeddings, masks, learned position indices, RoPE
tables for fixed masks/grids, and scheduler sigmas. Timestep embeddings and
Lumina Next RoPE factor choice are per-step dynamic.

## 10. Preprocessing and input packing

AuraFlow tokenizes with max-length padding/truncation, masks embeddings by
expanding `attention_mask` to the embedding shape, repeats embeddings/masks for
`num_images_per_prompt`, then concatenates negative/positive embeds for CFG.

Lumina Next lowercases/cleans captions, uses Gemma hidden state `[-2]`, pads to
multiple of 8, concatenates positive then negative embeddings for CFG, and
builds `base_sequence_length=(default_image_size // 16) ** 2` when proportional
attention is enabled.

Lumina 2 prepends a system prompt plus `<Prompt Start>` by default, forces
right-padding, uses Gemma2 hidden state `[-2]`, repeats embeddings/masks, and
does not concatenate CFG batches. The transformer itself packs text and image
tokens into per-example padded joint sequences.

There is no pipeline-level packed latent-token representation in these
families. Patchify/unpatchify is model-internal.

## 11. Graph rewrite / lowering opportunities

### Rewrite: linear patchify/unpatchify island

Source pattern: NCHW `view/permute/flatten -> Linear(P*P*C -> hidden)` and the
inverse `Linear -> view/permute/flatten`.

Replacement: layout-aware patchify and unpatchify primitives.

Preconditions: H/W divisible by patch size, source NCHW flatten order preserved,
known patch size 2, and explicit output channel axis for learned-sigma or CFG
channel slicing. NHWC lowering requires channel axis rewrite and different
patch flatten strides.

Failure cases: dynamic H/W without recomputed position/RoPE metadata, Lumina 2
variable-length packing fused before masks are represented, or NHWC pass leaving
`dim=1` chunk/slice unchanged.

### Rewrite: FlowMatch Euler deterministic step

Source pattern: scheduler table lookup plus `sample + (sigma_next - sigma) *
model_output`.

Replacement: fused pointwise update over latent map.

Preconditions: `stochastic_sampling=false`, no per-token timesteps, explicit
step index, and model_output already sign-adjusted for Lumina.

Failure cases: stochastic branch, dynamic shifting not represented, or img2img
begin-index behavior.

### Rewrite: CFG arithmetic variants

Source patterns:

- AuraFlow: batch chunk, `uncond + scale * (text - uncond)`.
- Lumina Next: learned-sigma trim, guide only first three channels, duplicate
  guided half, then select batch half.
- Lumina 2: separate positive/negative calls, optional norm ratio
  `||cond|| / ||guided||` along last dim.

Replacement: family-specific fused CFG kernels, not one generic CFG lowering.

Preconditions: known batch order and axis semantics. Lumina 2 norm uses source
`dim=-1` on NCHW output, which is width axis; preserve this exact behavior for
parity unless source changes.

### Rewrite: attention projection/GQA provider

Source pattern: Q/K/V linear projections, optional QK norm, RoPE, KV repeat,
SDPA, output projection.

Replacement: provider-backed SDPA/GQA attention with pre/post ops.

Preconditions: provider supports RoPE layout, bool/additive masks, head dims
72/96/256, noncausal mode, and optional fused projections only when source
processor mutation is active.

## 12. Kernel fusion candidates

Highest priority:

- Linear patchify/unpatchify kernels for NCHW DiT latents.
- Q/K/V + QK norm + RoPE/GQA + SDPA attention for Lumina; joint SDPA for AuraFlow.
- Adaptive norm/gate/residual epilogues: `AdaLayerNormZero`,
  `LuminaRMSNormZero`, `LuminaLayerNormContinuous`.
- SiLU-gated FFNs in AuraFlow and Lumina.
- FlowMatch Euler step and family-specific CFG arithmetic.

Medium priority:

- Lumina 2 joint sequence packing/scatter for fixed prompt masks.
- Lumina Next per-step RoPE generation/scaling.
- VAE decode Conv2d/GroupNorm/SiLU/up-block NHWC island.
- CFG normalization reductions for Lumina 2.

Lower priority:

- Text encoder compilation; external embeddings are enough for first slice.
- LoRA loader mutation and single-file conversion.
- VAE encode, tiling, and slicing.

## 13. Runtime staging plan

Stage 1: Admit `auraflow_denoiser_step` with external UMT5 prompt embeddings,
fixed `[B,4,128,128]` NCHW latents, static FlowMatch timestep tensor, and eager
SDPA fallback. This is the recommended first Dinoml staging/admission slice.

Stage 2: Add FlowMatch Euler deterministic scheduler step and AuraFlow CFG
arithmetic in host-visible loop state.

Stage 3: Add AutoencoderKL decode boundary for 4-channel, `scaling_factor=0.13025`.

Stage 4: Add Lumina Next transformer parity, including RoPE, GQA, learned-sigma
trim, sign negation, and partial-channel CFG.

Stage 5: Add Lumina 2 transformer parity with 16-channel VAE boundary, Gemma2
2304-width embeddings, separate CFG denoiser calls, and CFG normalization.

Stage 6: Add optimized attention/norm/patch kernels and guarded NHWC VAE islands.

Stage 7: Open separate reports for AuraFlow LoRA, Lumina 2 LoRA, single-file
conversion, Lumina-DiMOO, and any community img2img/control variants if a real
artifact requires them.

## 14. Parity and validation plan

- Config parse tests for all sampled model indexes and component configs.
- Random patchify/unpatchify parity for AuraFlow, Lumina Next, and Lumina 2.
- Position/RoPE parity: AuraFlow learned index selection, Lumina Next 2D RoPE,
  Lumina 2 caption/image/joint RoPE gather with varied prompt masks.
- One block parity per family with fp32 first, then bf16/fp16.
- Attention processor parity for AuraFlow joint, Lumina Next self/cross, and
  Lumina 2 GQA masked/unmasked paths.
- Full denoiser forward parity with fixed external prompt embeddings.
- CFG parity for all three family-specific guidance paths.
- FlowMatch Euler `set_timesteps` and one-step update parity.
- VAE decode scaling/shift parity for 4-channel and 16-channel configs.
- Short deterministic loop smoke with scheduler in Python.
- Suggested tolerances: fp32 scheduler `rtol=1e-5, atol=1e-6`; fp32 denoiser
  `rtol=1e-4, atol=1e-5`; fp16/bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Denoiser step latency by family at 1024px and batch 1/2.
- Attention backend comparison by token length: 4096 image tokens plus prompt
  length 256, and AuraFlow extra 8 register tokens.
- GQA KV-repeat overhead versus native grouped-query provider.
- Patchify/unpatchify time and memory traffic.
- Lumina 2 joint sequence packing overhead with varied prompt mask lengths.
- CFG overhead: AuraFlow/Lumina Next batch concat versus Lumina 2 two-call CFG.
- FlowMatch scheduler and CFG pointwise overhead.
- VAE decode throughput for 4-channel SDXL-style and 16-channel FLUX-style VAEs.
- VRAM/workspace by family, dtype, CFG mode, and prompt length.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `auraflow_lora_adapters`: `AuraFlowLoraLoaderMixin`, transformer/text-encoder
  PEFT scaling, fuse/unfuse/load/unload state.
- `lumina2_lora_adapters`: `Lumina2LoraLoaderMixin` plus non-Diffusers Lumina 2
  LoRA conversion utilities.
- `auraflow_lumina_single_file`: AuraFlow and Lumina 2 single-file checkpoint
  conversion paths.
- `lumina_dimoo`: VQModel/tokenizer pipeline surface, not the latent DiT image
  transformer audited here.
- `lumina_community_variants`: Neta/NetaYume and other community checkpoints
  once they differ beyond weights/config metadata.
- `autoencoder_kl_lumina2_flux_style`: 16-channel AutoencoderKL scale/shift
  path if not covered by the standalone AutoencoderKL report.
- Any concrete img2img/inpaint/control/upscale variant if a future repo adds
  family-local pipeline files.

Genuinely ignored/out of scope:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse AuraFlow and Lumina component configs, including Lumina Next cache fallback.
- [ ] Load `AuraFlowTransformer2DModel` weights first.
- [ ] Accept external UMT5/Gemma/Gemma2 prompt embeddings and masks.
- [ ] Implement AuraFlow linear patchify and learned-position selection.
- [ ] Implement AuraFlow joint and single transformer blocks with QK FP32 norm.
- [ ] Implement AuraFlow CFG and FlowMatch Euler deterministic step.
- [ ] Add AutoencoderKL decode boundary for 4-channel `0.13025` configs.
- [ ] Implement Lumina Next patchify, RoPE, GQA attention, learned-sigma trim, and partial CFG.
- [ ] Implement Lumina 2 16-channel patchify, RoPE gather, refiner/joint blocks, separate-call CFG, and decode scale/shift.
- [ ] Add family-specific CFG parity tests.
- [ ] Add FlowMatch scheduler table and one-step parity tests.
- [ ] Add attention processor parity for masks, RoPE, QK norm, and GQA.
- [ ] Benchmark denoiser step, attention backend, patchify/unpatchify, and VAE decode.
- [ ] Keep LoRA, single-file conversion, Lumina-DiMOO, and future variants separate.
