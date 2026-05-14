# Diffusers DiT Operator and Integration Report

Target slug: `dit`

Runtime scope: class-conditional ImageNet DiT textless image generation through
`DiTPipeline`, with label lookup treated as CPU/data-pipeline work. First Dinoml
slice should accept class ids, run the `DiTTransformer2DModel` denoiser on NCHW
latents, keep DDIM loop state host-visible, and decode with AutoencoderKL.

Ignored per user scope: backend/training/safety/callback paths, XLA/NPU/MPS,
Flax/ONNX, multi-GPU/context parallel, dropout/gradient-checkpointing, and
generic loader/offload machinery except where it changes component contracts.

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  facebook/DiT-XL-2-256
  facebook/DiT-XL-2-512
  kashif/DiT-XL-2-256, mirror/reference config
  kashif/DiT-XL-2-512, mirror/reference config

Config sources:
  Local cache:
    H:/configs/facebook/DiT-XL-2-256/model_index.json
    H:/configs/facebook/DiT-XL-2-512/model_index.json
    H:/configs/kashif/DiT-XL-2-256/model_index.json
    H:/configs/kashif/DiT-XL-2-512/model_index.json
  Official/raw HF URLs inspected, not written back because this task's owned
  write path is limited to this report:
    facebook/DiT-XL-2-256 transformer/config.json, scheduler/scheduler_config.json, vae/config.json
    facebook/DiT-XL-2-512 transformer/config.json, scheduler/scheduler_config.json, vae/config.json
    kashif/DiT-XL-2-256 transformer/config.json, scheduler/scheduler_config.json, vae/config.json
    kashif/DiT-XL-2-512 transformer/config.json, scheduler/scheduler_config.json, vae/config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/dit/pipeline_dit.py
  X:/H/diffusers/src/diffusers/pipelines/dit/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/dit_transformer_2d.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/image_processor.py indirectly through pipeline postprocess helpers.

External component configs inspected:
  None. DiT has no tokenizer, text encoder, image encoder, or external prompt
  encoder in the base pipeline.

Any missing files or assumptions:
  Local cache had only model_index.json for the selected DiT repos. Component
  configs were available from raw HF URLs without authentication. Configs name
  the transformer class as `Transformer2DModel`; `DiTPipeline` documents this
  historical mismatch and wires it to `DiTTransformer2DModel`.
```

## 2. Pipeline and component graph

DiT is a class-conditional latent image pipeline. It does not tokenize text and
does not use cross-attention. The conditioning input is an ImageNet class id;
classifier-free guidance uses a learned null class id `1000`.

```text
class label lookup / class id input
  -> latent noise initialization [B,4,S,S]
  -> denoising loop:
       optional CFG duplication with class ids [labels, null_class]
       DDIM scale_model_input, no-op for DDIM
       DiTTransformer2DModel NCHW latents + timestep + class labels
       CFG over epsilon channels only, preserving remaining learned-sigma channels
       learned-sigma channel trim to 4 channels
       DDIM scheduler.step
  -> AutoencoderKL decode(latents / scaling_factor)
  -> clamp, NCHW to NHWC CPU float32, PIL/NumPy output
```

Required first-slice components:

| Component | Class | File | Notes |
| --- | --- | --- | --- |
| Pipeline | `DiTPipeline` | `pipeline_dit.py` | Class-conditioned ImageNet generation; offload sequence `transformer->vae`. |
| Denoiser | `DiTTransformer2DModel` | `dit_transformer_2d.py` | Patch-token transformer, `ada_norm_zero`, self-attention only. |
| Scheduler | `DDIMScheduler` | `scheduling_ddim.py` | Official configs use epsilon prediction and `clip_sample=false`. |
| VAE | `AutoencoderKL` | `autoencoder_kl.py`, `vae.py` | Decode required; encode not used in base pipeline. |
| Labels | `id2label` in model index | `model_index.json` | Optional string-to-id helper; runtime can accept ids directly. |

Separate candidate reports:

| Surface | DiT status | Candidate |
| --- | --- | --- |
| LoRA/runtime adapters | Pipeline does not mix in LoRA-specific loaders; generic model mutation may still work through shared Diffusers utilities. | `dit_lora_adapters` only for a concrete artifact. |
| Textual inversion | Not applicable to base DiT; no tokenizer/text encoder. | None for base. |
| IP-Adapter | Not wired; no cross-attention or image-encoder branch. | None for base. |
| ControlNet/T2I-Adapter/GLIGEN | Not wired by `DiTPipeline`. Shared `BasicTransformerBlock` has GLIGEN code, but DiT passes no `cross_attention_kwargs`. | Separate only for forks. |
| img2img/inpaint/depth/upscale | No files in `pipelines/dit` for these variants. | External/community variants only. |
| Scheduler swaps | Constructor accepts `KarrasDiffusionSchedulers`, but official configs use DDIM. Examples show DPMSolver replacement. | `dit_dpmsolver_swap` if scheduler parity beyond DDIM is needed. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | image | latent sample | patch | latent channels | out channels | layers | heads x dim | inner | classes | scheduler |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `facebook/DiT-XL-2-256` | 256 | 32 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 1000 + null | DDIM epsilon |
| `facebook/DiT-XL-2-512` | 512 | 64 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 1000 + null | DDIM epsilon |
| `kashif/DiT-XL-2-256` | 256 | 32 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 1000 + null | DDIM epsilon |
| `kashif/DiT-XL-2-512` | config mirror uses 512 transformer but VAE config reports sample_size 256 | 64 | 2 | 4 | 8 | 28 | 16 x 72 | 1152 | 1000 + null | DDIM epsilon |

Transformer config fields:

| Field | Value |
| --- | --- |
| `_class_name` | `Transformer2DModel` in config, loaded as `DiTTransformer2DModel` |
| `norm_type` | `ada_norm_zero` |
| `activation_fn` | `gelu-approximate` |
| `attention_bias` | true |
| `cross_attention_dim` | null |
| `only_cross_attention` | false |
| `num_embeds_ada_norm` | 1000 |
| `norm_elementwise_affine` | false |
| `norm_eps` source default | `1e-5` for block norms; final DiT `norm_out` uses `1e-6` |
| `upcast_attention` | false |

Scheduler and VAE:

| Component | Config-derived fields |
| --- | --- |
| DDIM | `beta_start=0.0001`, `beta_end=0.02`, `beta_schedule=linear`, `num_train_timesteps=1000`, `prediction_type=epsilon`, `clip_sample=false`, `set_alpha_to_one=true`, `steps_offset=0`; omitted source defaults include `timestep_spacing=leading`, `thresholding=false`, `eta=0` at pipeline call. |
| VAE | SD VAE shape: `latent_channels=4`, `block_out_channels=[128,256,512,512]`, `layers_per_block=2`, GroupNorm groups 32, SiLU, up/down blocks all 2D. `scaling_factor` is omitted in configs and comes from `AutoencoderKL` source default `0.18215`; `force_upcast` omitted and defaults to true. |

Recommended first Dinoml scheduler slice: DDIM epsilon with `clip_sample=false`,
`thresholding=false`, `eta=0`, `timestep_spacing=leading`, and no stochastic
variance branch. DPMSolver is useful for user examples but should be a scheduler
swap candidate, not the first parity requirement.

## 3a. Family variation traps

- This is not PixArt or SD3: no text encoder, no prompt embeddings, no
  cross-attention, no joint text-image attention, no pooled text conditioning,
  no size/crop conditioning, no RoPE, and no QK norm.
- CFG ordering is positive labels first and null labels second. PixArt/SD
  text pipelines often concatenate negative first; do not reuse that assumption.
- CFG is applied only to the first `latent_channels` epsilon channels. The
  remaining channels are carried from the raw model output before the learned
  sigma trim.
- Denoising loop stores duplicated latents during CFG and chunks back to one
  batch after the loop.
- `out_channels=8` and `in_channels=4` means learned-sigma-style doubled output
  is active; scheduler receives only 4 channels.
- Source latents and VAE tensors are NCHW. NHWC is a guarded Conv2d/VAE/patch
  optimization only; `dim=1` channel splits and final NCHW-to-NHWC postprocess
  must be rewritten if a layout pass is admitted.
- `DiTPipeline.get_label_ids` contains a likely source bug: non-list strings are
  converted with `list(label)`, splitting into characters. First Dinoml runtime
  should accept integer class ids and treat label text lookup as noncritical UI
  parity.
- `PatchEmbed` can support position interpolation/cropping in shared source,
  but active DiT configs use fixed square latent grids and no `pos_embed_max_size`.

## 4. Runtime tensor contract

For `DiT-XL-2-256`, batch `B`, CFG enabled:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| class ids | `class_labels` | Python/list or tensor `[B]` int64 | Values 0-999. |
| null ids | `class_null` | `[B]` int64 | Constant value 1000 for CFG. |
| class labels input | `class_labels_input` | `[2B]` int64 | Positive labels then null labels. |
| latent noise | `latents` | `[B,4,32,32]` NCHW | dtype `transformer.dtype`; random normal. |
| loop latent state | `latent_model_input` | `[2B,4,32,32]` NCHW under CFG | Duplicated every step from first half. |
| timesteps | `timesteps` | `[2B]` int64 or scheduler dtype/device | Scalar scheduler timestep expanded to batch. |
| patch tokens | internal | `[2B,256,1152]` | Conv2d patch size 2 then flatten/transpose. |
| block conditioning | internal | `[2B,1152]` | sinusoidal timestep MLP plus class embedding. |
| raw model output | `noise_pred` | `[2B,8,32,32]` NCHW | epsilon + learned sigma-like channels. |
| guided output | `noise_pred` | `[2B,8,32,32]` | only epsilon half is CFG-combined. |
| scheduler model output | `model_output` | `[2B,4,32,32]` | First 4 channels after learned-sigma split. |
| final latents | `latents` | `[B,4,32,32]` | First half after loop. |
| VAE decode input | `latents / 0.18215` | `[B,4,32,32]` or `[B,4,64,64]` | 512 model uses latent sample 64. |
| decoded sample | `samples` | `[B,3,256,256]` or `[B,3,512,512]` NCHW | Clamp `(x/2+0.5)` then CPU NHWC. |

Patchify/unpatchify:

```text
Patchify:
  hidden_states [B,4,S,S]
  Conv2d(4 -> 1152, kernel=2, stride=2, bias=true)
  flatten(2).transpose(1,2): [B,1152,S/2,S/2] -> [B,(S/2)^2,1152]
  add fixed 2D sin-cos position embedding

Unpatchify:
  Linear(1152 -> 2*2*out_channels)
  reshape [B,Ht,Wt,2,2,Cout]
  einsum "nhwpqc->nchpwq"
  reshape [B,Cout,Ht*2,Wt*2]
```

CPU/data-pipeline work: optional ImageNet label string mapping, random generator
ownership, PIL/NumPy postprocess. GPU/runtime work: denoiser, CFG, scheduler
step, VAE decode, and optional latent RNG when Dinoml admits random init.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation, scaling, cat, chunk, split, reshape, flatten,
  transpose, einsum-style unpatchify.
- Scalar timestep tensor creation/expand to batch.
- Channel-axis split for learned sigma and batch-axis split for CFG.
- NCHW to NHWC postprocess on CPU for image output.

Convolution/downsample/upsample ops:

- Patch embedding `Conv2d(4 -> 1152, 2x2, stride=2, bias=true)`.
- AutoencoderKL decode Conv2d, ResNet, GroupNorm, SiLU, upsample blocks, and
  mid-block attention from the shared SD VAE.

GEMM/linear ops:

- Timestep embedding MLP: sinusoidal 256 -> Linear(256 -> 1152) -> SiLU ->
  Linear(1152 -> 1152).
- Label embedding table: 1001 x 1152.
- Per block self-attention Q/K/V/output Linear(1152 -> 1152), bias true.
- Per block FFN with approximate GELU; source `FeedForward` default multiplier
  should be inspected in parity tests but active hidden size is 1152.
- AdaLN-Zero modulation Linear(1152 -> 6912) per block.
- Final conditioning Linear(1152 -> 2304), final projection Linear(1152 -> 32).

Attention primitives:

- Dense noncausal self-attention over patch tokens: 256 tokens for 256px,
  1024 tokens for 512px; 16 heads x 72 head dim.
- No attention mask, no cross-attention, no RoPE, no QK norm.
- Default `AttnProcessor2_0` uses PyTorch SDPA when available; eager processor
  is fallback parity.

Normalization and adaptive conditioning:

- LayerNorm without affine, eps `1e-6` in `AdaLayerNormZero` and final norm.
- AdaLN-Zero scale/shift/gate for attention and MLP residuals.
- VAE GroupNorm.

Scheduler and guidance arithmetic:

- DDIM epsilon prediction conversion, deterministic `eta=0` first.
- CFG over epsilon channels with positive/null order.
- Learned-sigma channel trim.
- Latent scaling for VAE decode.

## 6. Denoiser/model breakdown

`DiTTransformer2DModel.forward`:

```text
NCHW latents
  -> PatchEmbed Conv2d + 2D sin-cos position
  -> 28 x BasicTransformerBlock(norm_type=ada_norm_zero)
  -> conditioning = first_block.norm1.emb(timestep, class_labels)
  -> SiLU + Linear(1152 -> 2304), chunk shift/scale
  -> LayerNorm + final scale/shift
  -> Linear(1152 -> patch*patch*out_channels)
  -> unpatchify to NCHW
```

Active `BasicTransformerBlock` path:

```text
AdaLayerNormZero:
  CombinedTimestepLabelEmbeddings(timestep, class)
  SiLU + Linear(D -> 6D)
  LayerNorm(hidden) * (1 + scale_msa) + shift_msa

self-attn:
  Q/K/V projection -> noncausal SDPA/eager attention -> output projection
  gate_msa * attention_output + residual

MLP:
  LayerNorm -> scale_mlp/shift_mlp -> FeedForward GELU-approximate
  gate_mlp * ff_output + residual
```

Inactive source branches for this target: cross-attention, double self-attn,
GLIGEN fuser, positional embeddings inside `BasicTransformerBlock`,
chunked feed-forward, training gradient checkpointing.

## 7. Attention requirements

| Attention | Query/key/value | Tokens | Mask | Backend path |
| --- | --- | ---: | --- | --- |
| DiT self-attn 256 | `[B,256,1152]` | 256 | none | `Attention` + `AttnProcessor2_0` or eager |
| DiT self-attn 512 | `[B,1024,1152]` | 1024 | none | same |

Required properties:

- 16 heads, head dim 72, inner dim 1152.
- Dense bidirectional attention, `is_causal=false`.
- No padding mask or varlen packing required.
- No QK norm or RoPE.
- Attention bias exists in Q/K/V projections.
- Fused projections are source-supported by the shared attention mixin
  (`fuse_qkv_projections`) and `FusedAttnProcessor2_0`, but not required by
  checkpoint format or pipeline load.

Dinoml flash-style provider validity: this is a clean candidate because there
is no mask, no cross branch, and no rotary/QK-norm complication. Preconditions
should still guard dtype, head_dim=72 support, sequence length, contiguous BNC
layout, and exact softmax scaling.

## 8. Scheduler and denoising-loop contract

Base loop:

```text
scheduler.set_timesteps(num_inference_steps)
latent_model_input = cat([latents, latents]) if CFG else latents
class_labels_input = cat([labels, null_1000]) if CFG else labels
for t in scheduler.timesteps:
  if CFG:
    half = latent_model_input[:B]
    latent_model_input = cat([half, half], dim=0)
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)  # DDIM no-op
  noise_pred = transformer(latent_model_input, timestep=t_batch, class_labels=class_labels_input)
  if CFG:
    eps, rest = noise_pred[:, :4], noise_pred[:, 4:]
    cond_eps, uncond_eps = split(eps, 2, dim=0)
    guided = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = cat([guided, guided], dim=0)
    noise_pred = cat([eps, rest], dim=1)
  model_output = split(noise_pred, 4, dim=1)[0]
  latent_model_input = scheduler.step(model_output, t, latent_model_input).prev_sample
latents = latent_model_input[:B] if CFG else latent_model_input
```

DDIM first-slice state:

- `timesteps` from `set_timesteps`, source default `timestep_spacing=leading`
  because old configs omit the field.
- `alphas_cumprod` and `final_alpha_cumprod` from linear beta schedule.
- `scale_model_input` is identity.
- epsilon conversion:
  `x0 = (sample - sqrt(beta_prod_t) * eps) / sqrt(alpha_prod_t)`.
- `clip_sample=false` in official configs; dynamic threshold omitted and source
  default false.
- `eta=0` in pipeline call, so no stochastic variance noise is required first.

Host/runtime split: keep `set_timesteps`, loop index, and scalar coefficient
table ownership host-visible initially. Compile denoiser, CFG/learned-sigma
arithmetic, and one DDIM pointwise update once scheduler state is explicit.

## 9. Position, timestep, and custom math

Position:

- `PatchEmbed` registers fixed 2D sin-cos position embeddings for the configured
  latent token grid.
- Active configs are square and fixed: token grid 16x16 for 256px and 32x32 for
  512px after patch size 2.
- Shared `PatchEmbed` interpolation/cropping paths are not active for DiT base.

Timestep/class conditioning:

- `CombinedTimestepLabelEmbeddings` uses sinusoidal timestep projection with
  256 channels, `flip_sin_to_cos=true`, `downscale_freq_shift=1`.
- Timestep MLP maps 256 -> 1152 -> 1152 with SiLU.
- `LabelEmbedding` has 1001 rows because dropout probability `0.1` creates a
  learned CFG/null class slot at index 1000. In inference, the pipeline passes
  null labels explicitly instead of relying on training dropout.
- Each block computes 6 modulation vectors: `shift_msa`, `scale_msa`,
  `gate_msa`, `shift_mlp`, `scale_mlp`, `gate_mlp`.
- Final layer reuses the first block's embedding module, then applies SiLU and
  Linear(1152 -> 2304) for final shift/scale.

Precomputable: fixed 2D position embeddings, class embedding table lookup for
static labels/null labels, scheduler alpha tables for a given step count.
Dynamic per step: timestep embeddings, AdaLN modulation, DDIM coefficients,
and random initial latents.

## 10. Preprocessing and input packing

- No tokenization, text cleanup, prompt truncation, or prompt embedding cache.
- `id2label` is stored in `model_index.json` for human-friendly class lookup;
  runtime can bypass it with integer class labels.
- Latents are initialized at `transformer.config.sample_size`; `height`/`width`
  are not public pipeline inputs.
- There is no pipeline-level latent packing. Patchify lives inside the model.
- VAE decode applies `latents / vae.config.scaling_factor`; in sampled configs
  the value is source default `0.18215`.
- Postprocess clamps decoded NCHW tensors into `[0,1]`, moves to CPU, permutes
  to NHWC, casts to float32 NumPy, then optionally PIL.

## 11. Graph rewrite / lowering opportunities

### Rewrite: DiT PatchEmbed canonical op

Source pattern:

```text
Conv2d(4 -> 1152, kernel=2, stride=2) -> flatten(2) -> transpose(1,2) -> add pos
```

Replacement: patch-embedding primitive or Conv2d plus planned NCHW-to-BNC
layout transform.

Preconditions: source NCHW, square latent grid equal to configured `sample_size`,
H/W divisible by 2, fixed sin-cos position table for the exact token grid.
NHWC lowering requires OIHW -> HWIO weight transform and exact flatten order.

Failure cases: dynamic H/W, active `pos_embed_max_size` cropping, or downstream
consumer expecting source BNC order but layout pass silently changes token order.

Parity test: random `[B,4,32,32]` and `[B,4,64,64]` patchify comparison.

### Rewrite: Unpatchify primitive

Source pattern:

```text
Linear(D -> 2*2*C) -> reshape [B,H,W,2,2,C] -> einsum("nhwpqc->nchpwq") -> reshape
```

Replacement: token-to-NCHW unpatchify primitive.

Preconditions: token count is a perfect square, patch size 2, output channels 8
before learned-sigma trim, final consumer uses NCHW or explicit axis rewrite.

Failure cases: NHWC layout pass leaving channel split on `dim=1`, non-square
token count, or alternate patch size.

### Rewrite: CFG epsilon-only fusion

Source pattern:

```text
eps = noise_pred[:, :4]
rest = noise_pred[:, 4:]
cond_eps, uncond_eps = split(eps, 2, batch)
guided = uncond_eps + scale * (cond_eps - uncond_eps)
noise_pred = cat([guided, guided], batch) + rest passthrough
model_output = noise_pred[:, :4]
```

Replacement: fused pointwise kernel that emits the scheduler's `[2B,4,S,S]`
model output directly, or `[B,4,S,S]` if the scheduler update is also fused and
the duplicated latent state is eliminated.

Preconditions: CFG enabled, class-label batch order `[cond, uncond]`, doubled
output channels, DDIM step can consume the chosen batch shape.

Failure cases: guidance disabled, non-doubled output channels, scheduler
implementation expecting duplicated sample shape, or reused negative-first CFG
logic from text pipelines.

### Rewrite: Self-attention fused QKV

Source pattern: separate Q/K/V linears from the same hidden states, SDPA/eager
attention, output projection.

Replacement: fused QKV projection plus flash-style self-attention provider.

Preconditions: self-attention only, no mask, no added K/V, no QK norm, no
processor mutation, provider supports head dim 72 and selected dtype.

Failure cases: `fuse_qkv_projections` runtime mutation not reflected in loaded
weights, provider head-dim constraints, or non-contiguous BNC token layout.

## 12. Kernel fusion candidates

Highest priority:

- PatchEmbed Conv2d + flatten/transpose + position add.
- AdaLN-Zero LayerNorm + modulation + gated residual epilogues.
- Self-attention QKV + SDPA/flash + output projection for 256/1024 token grids.
- GELU-approximate feed-forward MLP.
- CFG epsilon-only arithmetic plus learned-sigma trim.

Medium priority:

- DDIM epsilon scheduler pointwise update with precomputed alpha coefficients.
- Timestep/class embedding MLP and AdaLN modulation cache within one step.
- AutoencoderKL decode Conv2d/GroupNorm/SiLU/up-block NHWC island, shared with
  SD/PixArt/SD3 reports.

Lower priority:

- Label string lookup, PIL/NumPy postprocess.
- DDIM stochastic `eta>0`, clip/threshold variants not active in configs.
- DPMSolver scheduler swap from examples.
- Generic LoRA/adapter mutation for community DiT forks.

## 13. Runtime staging plan

Stage 1: Parse `facebook/DiT-XL-2-256` model index and component configs,
including the `Transformer2DModel` -> `DiTTransformer2DModel` class alias.

Stage 2: Load transformer weights and validate patchify, timestep/class
embedding, one AdaLN-Zero block, and final unpatchify on random tensors.

Stage 3: Compile full DiT-XL-2-256 denoiser for fixed NCHW latents
`[B,4,32,32]`, class ids, and timesteps. Keep VAE and scheduler outside.

Stage 4: Add CFG epsilon-only arithmetic and learned-sigma trim around one
denoiser step.

Stage 5: Add DDIM epsilon scheduler setup/step with host-visible loop state and
short deterministic denoising-loop parity.

Stage 6: Add AutoencoderKL decode boundary with scaling factor `0.18215`.

Stage 7: Add DiT-XL-2-512 by widening latent grid to `[B,4,64,64]` and token
count to 1024.

Stage 8: Add optimized attention, AdaLN, patchify/unpatchify, and VAE decode
fusions. Keep DPMSolver swaps and adapter mutation separate.

## 14. Parity and validation plan

- Random `PatchEmbed` parity for 32 and 64 latent grids.
- `CombinedTimestepLabelEmbeddings` parity for class ids and null id 1000.
- `AdaLayerNormZero` parity including gate/shift/scale chunking.
- One `BasicTransformerBlock` parity with self-attention and MLP.
- Full `DiTTransformer2DModel` parity for `[B,4,32,32]`, then `[B,4,64,64]`.
- CFG epsilon-only arithmetic and learned-sigma trim parity.
- DDIM `set_timesteps`, epsilon step, and `clip_sample=false` parity.
- VAE decode scaling parity and standalone AutoencoderKL decode parity.
- Short deterministic denoising-loop smoke with fixed generator/class ids.
- Suggested tolerances: fp32 scheduler arithmetic `rtol=1e-5, atol=1e-6`;
  transformer fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 initially
  `rtol=2e-2, atol=2e-2` until attention/provider choices settle.

## 15. Performance probes

- One denoiser step for DiT-XL-2-256 and DiT-XL-2-512 by batch and dtype.
- Attention backend comparison for sequence lengths 256 and 1024, head dim 72.
- Patchify/unpatchify overhead relative to transformer block time.
- AdaLN-Zero and feed-forward time split per block.
- DDIM scheduler plus CFG overhead across 25/50 steps.
- VAE decode throughput at 256 and 512.
- VRAM/workspace usage for CFG enabled vs disabled.
- Weight-load and first-run latency for transformer plus VAE.

## 16. Scope boundary and separate candidates

Separate candidate reports:

- `dit_dpmsolver_swap`: examples replace DDIM with
  `DPMSolverMultistepScheduler`; this needs multistep scheduler state and
  conversion parity beyond the official default.
- `dit_lora_adapters`: only if a concrete DiT LoRA/PEFT artifact is selected;
  base pipeline has no LoRA-specific mixin.
- `dit_external_variants`: community img2img/inpaint/control/upscale forks, if
  encountered, because `pipelines/dit` contains only base class-conditioned
  generation.
- `autoencoder_kl_dit`: best folded into the existing AutoencoderKL report
  unless DiT-specific VAE config differences become meaningful.

Genuinely out of scope for this audit:

- Text encoders, tokenizers, textual inversion, prompt caches.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN side inputs.
- Safety checker and NSFW filtering.
- Training, losses, dropout, gradient checkpointing.
- XLA/NPU/MPS, Flax/ONNX, multi-GPU/context parallel.
- Callback mutation and interactive interrupt paths.

## 17. Final implementation checklist

- [ ] Parse DiT model indexes and component configs.
- [ ] Map `Transformer2DModel` config entries to `DiTTransformer2DModel`.
- [ ] Load DiT-XL-2-256 transformer weights.
- [ ] Accept integer ImageNet class ids and null class id 1000.
- [ ] Implement fixed NCHW latent contract `[B,4,32,32]`.
- [ ] Implement PatchEmbed Conv2d + fixed 2D sin-cos position path.
- [ ] Implement timestep sinusoidal projection and class embedding.
- [ ] Implement AdaLN-Zero modulation and gated residuals.
- [ ] Implement DiT self-attention, MLP, final norm/projection, and unpatchify.
- [ ] Implement CFG epsilon-only arithmetic with DiT batch ordering.
- [ ] Implement learned-sigma channel trim.
- [ ] Implement DDIM epsilon `clip_sample=false`, `eta=0` first slice.
- [ ] Add AutoencoderKL decode boundary with scaling factor default `0.18215`.
- [ ] Add DiT-XL-2-512 latent/token-grid parity.
- [ ] Add guarded QKV/attention and patch layout fusions.
- [ ] Keep DPMSolver swaps, adapters, and community variants separate.
