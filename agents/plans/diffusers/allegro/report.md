# Diffusers Allegro Operator and Integration Report

Candidate slug: `allegro`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  rhymes-ai/Allegro
  rhymes-ai/Allegro-TI2V

Config sources:
  H:/configs/rhymes-ai/Allegro/model_index.json
  H:/configs/rhymes-ai/Allegro-TI2V/model_index.json
  Official raw Hugging Face component configs inspected in memory:
    rhymes-ai/Allegro/transformer/config.json
    rhymes-ai/Allegro/vae/config.json
    rhymes-ai/Allegro/scheduler/scheduler_config.json
    rhymes-ai/Allegro/text_encoder/config.json
    rhymes-ai/Allegro/tokenizer/tokenizer_config.json
    rhymes-ai/Allegro-TI2V/transformer/config.json
    rhymes-ai/Allegro-TI2V/vae/config.json
    rhymes-ai/Allegro-TI2V/scheduler/scheduler_config.json
    rhymes-ai/Allegro-TI2V/text_encoder/config.json
    rhymes-ai/Allegro-TI2V/tokenizer/tokenizer_config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/allegro/pipeline_allegro.py
  diffusers/src/diffusers/pipelines/allegro/pipeline_output.py
  diffusers/src/diffusers/pipelines/allegro/__init__.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_allegro.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_allegro.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/resnet.py
  diffusers/src/diffusers/models/downsampling.py
  diffusers/src/diffusers/models/upsampling.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_euler_ancestral_discrete.py
  diffusers/src/diffusers/schedulers/scheduling_ddpm.py
  diffusers/src/diffusers/video_processor.py
  diffusers/src/diffusers/image_processor.py

External component configs inspected:
  T5EncoderModel and T5Tokenizer configs from the two official repos.

Any missing files or assumptions:
  The main runtime target is base text-to-video `AllegroPipeline` with
  `AllegroTransformer3DModel`. The official `rhymes-ai/Allegro-TI2V`
  model_index and transformer config name `AllegroTransformerTI2V3DModel`, but
  this checkout does not expose that class or a TI2V pipeline source file under
  `pipelines/allegro` or `models/transformers`. TI2V is therefore inventoried
  as a blocked/separate variant until the matching source lands. Multi-GPU,
  callbacks/interrupt mutation, XLA/NPU/MPS/Flax/ONNX, safety/NSFW, training,
  losses, dropout, and gradient checkpointing are out of scope.
```

## 2. Pipeline and component graph

Allegro is a latent text-to-video pipeline using T5 text conditioning, a
spatial-patch video transformer, Euler ancestral sampling, and an Allegro
3D/KL VAE. The pipeline constructor has no optional components and declares
`model_cpu_offload_seq = "text_encoder->transformer->vae"`.

```text
prompt / negative prompt
  -> T5Tokenizer + T5EncoderModel token embeddings and attention masks
  -> latent noise [B,4,T_lat,H/8,W/8] in NCTHW source layout
  -> 3D RoPE table/grid for latent patch tokens
  -> denoising loop:
       CFG batch concat
       EulerAncestralDiscreteScheduler.scale_model_input
       AllegroTransformer3DModel(latents, T5 embeds, masks, timestep, RoPE)
       CFG arithmetic
       EulerAncestralDiscreteScheduler.step
  -> latents / vae.scaling_factor
  -> AutoencoderKLAllegro tiled decode
  -> crop to requested frames/height/width
  -> VideoProcessor postprocess
```

Required first-slice components:

| Component | Class | Source / config fact |
| --- | --- | --- |
| Pipeline | `AllegroPipeline` | `pipeline_allegro.py`; text-to-video only in current folder. |
| Tokenizer | `T5Tokenizer` | Official tokenizer config, max length 512, 100 extra IDs. |
| Text encoder | `T5EncoderModel` | T5-XXL width: `d_model=4096`, 24 layers, 64 heads. |
| Denoiser | `AllegroTransformer3DModel` | 32 blocks, 24 heads, head dim 96, 4 latent channels. |
| VAE | `AutoencoderKLAllegro` | 4 latent channels, temporal compression 4, spatial compression 8. |
| Scheduler | `EulerAncestralDiscreteScheduler` | Official scheduler config, epsilon prediction, linear betas. |

Independently cacheable stages: prompt/negative prompt embeddings and masks,
RoPE tables for a fixed latent grid, scheduler timesteps/sigmas for a fixed
step count, initial latents when supplied, and VAE output postprocessing.

Separate candidate reports:

| Candidate | Class/file anchors | Runtime delta |
| --- | --- | --- |
| `allegro_ti2v` | Official config names `AllegroTransformerTI2V3DModel`; source absent in inspected checkout | Text/image-to-video variant cannot be audited from this checkout. Needs matching model and pipeline files before admission. |
| `allegro_vae_tiling` | `AutoencoderKLAllegro.tiled_encode/decode` | Tiled decode is required by current source; non-tiled encode/decode raises `NotImplementedError`. Tile blending and coverage constraints deserve a focused codec report. |
| `allegro_lora_adapters` | Generic Diffusers LoRA/PEFT loader mixins, if enabled externally | No Allegro-specific loader mixin in the pipeline class; runtime adapter mutation should be a separate generic transformer/attention report if used. |
| `allegro_scheduler_swaps` | Pipeline type is `KarrasDiffusionSchedulers` | The official config uses Euler Ancestral, but the constructor type admits compatible scheduler swaps. |

ControlNet, T2I-Adapter, IP-Adapter, GLIGEN, img2img, inpaint, depth2img, and
upscaling are not present as Allegro pipeline classes in this checkout.

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | Transformer class in config | Transformer dimensions | VAE | Scheduler | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `rhymes-ai/Allegro` | `AllegroPipeline` | `AllegroTransformer3DModel` | 32 layers, 24 heads, head dim 96, inner 2304, 4 in/out channels, patch 2x2 spatial, `patch_size_t=1` | 4 latent channels, `block_out_channels=[128,256,512,512]`, temporal compression 4, spatial 8, scale 0.13 | Euler ancestral, epsilon, linear betas, linspace | First Dinoml audit target. |
| `rhymes-ai/Allegro-TI2V` | `AllegroPipeline` in model_index | `AllegroTransformerTI2V3DModel` | Same visible config shape as base plus `norm_num_groups=32`; no matching class in checkout | Same VAE config as base | Same scheduler config as base | Blocked variant: config accessible, source absent. |

Base transformer config facts:

| Field | Value | Runtime effect |
| --- | --- | --- |
| `sample_size` | `[90,160]` latent grid | Pipeline default output is 720x1280 after VAE spatial scale 8. |
| `sample_frames` / `sample_size_t` | 22 latent frames | Pipeline default `num_frames = 22 * 4 = 88`, but latent prep maps even 88 frames to 22 latents. |
| `patch_size`, `patch_size_t` | 2, 1 | Patchify each frame spatially only; no temporal patching in base. |
| `num_attention_heads`, `attention_head_dim` | 24, 96 | Inner width 2304; head dim divisible into three 32-dim RoPE axis chunks. |
| `cross_attention_dim`, `caption_channels` | 2304, 4096 | T5 output is projected from 4096 to 2304 before cross-attention. |
| `activation_fn` | `gelu-approximate` | Feed-forward uses GELU approximate. |
| `norm_type` | `ada_norm_single` in config | Implemented by `AdaLayerNormSingle` plus per-block scale/shift tables. |
| `interpolation_scale_t/h/w` | 2.2 / 2.0 / 2.0 | Used by Allegro 3D RoPE generation. |

VAE config facts:

| Field | Value |
| --- | --- |
| `in_channels`, `out_channels` | 3, 3 |
| `latent_channels` | 4 |
| `block_out_channels` | `[128,256,512,512]` |
| `layers_per_block` | 2 |
| `temporal_downsample_blocks` | `[true,true,false,false]` |
| `temporal_upsample_blocks` | `[false,true,true,false]` |
| `temporal_compression_ratio` | 4 |
| spatial compression | 8, inferred from `2 ** (len(block_out_channels)-1)` in source |
| `scaling_factor` | 0.13 |
| tiling kernel/stride source defaults | kernel `(24,320,320)`, overlap `(8,120,80)`, stride `(16,200,240)` |

Text encoder config facts:

| Component | Fields |
| --- | --- |
| `T5Tokenizer` | `model_max_length=512`, pad/eos/unk, 100 extra IDs, 103 added token decoder entries. |
| `T5EncoderModel` | `d_model=4096`, `d_ff=10240`, 24 layers, 64 heads, `d_kv=64`, gated GELU, vocab 32128. |

Recommended first Dinoml scheduler slice:

- Start with `EulerAncestralDiscreteScheduler` using official Allegro config:
  `prediction_type="epsilon"`, linear beta schedule, `timestep_spacing="linspace"`,
  `steps_offset=0`, `rescale_betas_zero_snr=false`.
- This differs from Wan/CogVideoX first video scheduler slices; it is stochastic
  ancestral Euler and requires explicit noise input in `step`.

## 3a. Family variation traps

- Source denoiser, VAE, and pipeline latent layout is NCTHW. NDHWC is only a
  guarded optimization candidate.
- The VAE decode path returns `[B,T,C,H,W]`; the pipeline variable name/comment
  says channels first but `VideoProcessor.postprocess_video` accepts the BTCHW
  layout for output conversion.
- `AutoencoderKLAllegro._decode` and `_encode` raise `NotImplementedError`
  unless `use_tiling` is enabled. The example calls `pipe.enable_vae_tiling()`.
  First end-to-end parity must include tiled VAE decode or stop at latent output.
- VAE tiling has hard coverage assumptions: output tile counts use
  `floor((dim - kernel) / stride) + 1`; default 88x720x1280 output maps exactly
  through latent 22x90x160 and one temporal tile by multiple H/W tiles. Smaller
  random shapes may fail unless chosen to satisfy tile formulas.
- Pipeline `check_inputs` only requires height/width divisible by 8, but
  transformer patchify additionally requires latent H/W divisible by patch size
  2. Practical H/W should be divisible by 16 for transformer parity.
- Prompt embeddings are unsqueezed from `[B,L,D]` to `[B,1,L,D]`, then flattened
  inside the transformer after caption projection.
- CFG is batched: negative and positive embeddings/masks are concatenated before
  one transformer call.
- Base Allegro uses no temporal patching (`patch_size_t=1`); do not generalize
  from CogVideoX 1.5 or Wan TI2V temporal patch paths.
- TI2V configs are accessible but the required source class is absent in this
  checkout; this is a hard variant blocker, not an operator inference.

## 4. Runtime tensor contract

For the official default 88-frame, 720x1280 base run:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| token ids / masks | tokenizer output | `[B,512]` | CPU/data pipeline; prompt cleaning optional. |
| prompt embeds | T5 output | `[B,512,4096]` | Duplicated per video; `num_videos_per_prompt` is forced to 1. |
| CFG embeds | after concat/unsqueeze | `[2B,1,512,4096]` | Negative then positive when `guidance_scale > 1`. |
| CFG mask | after concat | `[2B,512]` | Transformer converts to additive bias `[B,1,K]`. |
| latents | noise/sample | `[B,4,22,90,160]` NCTHW | Scaled by `scheduler.init_noise_sigma`. |
| model input | CFG concat + scheduler scale | `[2B,4,22,90,160]` | Euler ancestral divides by `sqrt(sigma^2+1)`. |
| patch tokens | transformer hidden | `[2B,22*45*80,2304]` | 79,200 video tokens at default shape. |
| text context | caption projection | `[2B,512,2304]` | Cross-attention K/V context. |
| RoPE positions | `freqs`, `grid` | three axis frequency tables plus `[3,1,79200]` index grid | Applies only to self-attention Q/K. |
| denoiser output | noise prediction | `[2B,4,22,90,160]` | Chunked into uncond/text for CFG. |
| scheduler state | timesteps/sigmas | timesteps on device, sigmas CPU in source | Step index and stochastic noise per step. |
| VAE decode input | scaled latents | `[B,4,22,90,160]` NCTHW | `latents / 0.13`. |
| VAE decoder tile output | sample | `[B,88,3,720,1280]` BTCHW | Source `tiled_decode` returns BTCHW. |
| postprocess input | cropped video | `[B,88,3,720,1280]` | Converted to PIL/NumPy/list by `VideoProcessor`. |

CPU/data-pipeline work: caption cleaning, tokenization, T5 execution if prompt
embeddings are not supplied, output conversion, scheduler table setup. GPU work:
latent initialization, transformer forward, CFG arithmetic, scheduler pointwise
and stochastic update, tiled VAE decode.

## 5. Operator coverage checklist

### Tensor/layout ops

- NCTHW latent tensors; BTCHW VAE output tensors.
- `permute`, `flatten`, `unflatten`, `view`, `reshape`, `transpose`, `contiguous`.
- CFG `cat`/`chunk` over batch.
- Attention mask conversion from keep mask to additive bias.
- `max_pool3d` for optional video attention masks, inactive in base pipeline.
- Tiled VAE crop/scatter/add and overlap blend.

### Convolution/downsample/upsample ops

- Transformer patch embed via Diffusers `PatchEmbed`: per-frame Conv2d
  `4 -> 2304`, kernel/stride 2.
- VAE Conv2d in/out and quant/post-quant: latent moments split on channel dim.
- VAE Conv3d temporal layers with explicit temporal edge padding by frame copy.
- VAE 2D ResnetBlock paths over flattened `B*T` frames.
- `Downsample2D` and `Upsample2D` spatial operations inside VAE blocks.
- Temporal down/up inside `AllegroTemporalConvLayer`: stride-2 temporal Conv3d
  for downsample and channel-unflatten/reorder for upsample.

### GEMM/linear ops

- T5 external encoder if admitted later.
- `PixArtAlphaTextProjection`: 4096 -> 2304 caption projection.
- AdaLayerNormSingle timestep embedding and block modulation.
- Q/K/V/out projections for self-attention and cross-attention.
- GEGLU/GELU feed-forward projections.
- Final `proj_out`: `2304 -> 4 * 2 * 2 = 16`.

### Attention primitives

- Self-attention over video patch tokens with Allegro 3D RoPE on Q/K.
- Cross-attention from video tokens to projected T5 context with additive text
  attention mask.
- VAE mid-block spatial self-attention over flattened frame maps.

### Normalization and adaptive conditioning

- LayerNorm with affine disabled in transformer config.
- AdaLayerNormSingle timestep conditioning.
- Per-block six-way scale/shift/gate table addition.
- VAE GroupNorm and optional SpatialNorm in attention helper paths.

### Scheduler and guidance arithmetic

- Euler ancestral `scale_model_input`, epsilon-to-x0 conversion, derivative
  update, stochastic `sigma_up` noise addition, step-index state.
- CFG `uncond + guidance * (text - uncond)`.

### Video-specific ops

- Temporal compression by 4 in latent prep.
- Tiled video VAE decode over latent cubes and overlap blending in T/H/W.
- Video postprocess from BTCHW.

## 6. Denoiser/model breakdown

`AllegroTransformer3DModel.forward`:

```text
hidden_states [B,4,T,H,W]
-> optional attention mask pooling to patch-token bias
-> encoder mask to additive text bias
-> AdaLayerNormSingle(timestep) -> block modulation tensors
-> BTCHW flatten to per-frame NCHW
-> PatchEmbed Conv2d(k=s=2) -> [B*T,H/2*W/2,2304]
-> unflatten/flatten time to [B,T*H/2*W/2,2304]
-> caption_projection(T5 embeds) -> [B,512,2304]
-> 32 AllegroTransformerBlock layers
-> final LayerNorm + adaptive scale/shift
-> Linear to patch pixels
-> unpatchify back to [B,4,T,H,W]
```

`AllegroTransformerBlock`:

```text
scale_shift_table + timestep -> shift/scale/gate for MSA and MLP
LayerNorm -> scale/shift -> self-attention with RoPE -> gated residual
cross-attention to projected T5 context -> residual
LayerNorm -> scale/shift -> GEGLU/GELU FeedForward -> gated residual
```

At default resolution, sequence length is `22 * 45 * 80 = 79200` tokens per
sample, so attention provider admission is primarily a sequence/workspace
problem even though head dim is only 96.

## 7. Attention requirements

Primary implementation is `AllegroAttnProcessor2_0` in
`attention_processor.py`, called through the shared `Attention` module. It uses
PyTorch `scaled_dot_product_attention` as the parity path.

- Self-attention: noncausal, mask optional, Q/K/V projected from video tokens,
  24 heads, head dim 96, RoPE applied to Q/K only when not cross-attention.
- Cross-attention: video queries attend to projected T5 tokens; encoder mask is
  additive bias after shape conversion.
- RoPE: `apply_rotary_emb_allegro` chunks the 96-dim head into three 32-dim
  temporal/height/width chunks and gathers cos/sin tables by grid indices.
- VAE spatial attention: shared `Attention` on NCHW frame maps in the mid block,
  with GroupNorm-like spatial attention behavior.
- Fused QKV is not required by the source path. Dinoml may fuse projections
  under exact weight/bias and mask/RoPE preconditions.

Flash-style constraints:

- Base self-attention is valid for a flash-style provider only if sequence
  length, dtype, mask, and workspace guards pass; the default 79,200-token
  sequence is a stress case.
- RoPE gather/apply and adaptive norm/gating remain explicit pre/post ops.
- Cross-attention has a different key length and additive mask; do not fold it
  into self-attention.
- Eager/native SDPA defines parity.

## 8. Scheduler and denoising-loop contract

Official Allegro configs use `EulerAncestralDiscreteScheduler`:

```text
num_train_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
beta_schedule = linear
prediction_type = epsilon
timestep_spacing = linspace
steps_offset = 0
rescale_betas_zero_snr = false
```

Loop contract:

```text
timesteps = retrieve_timesteps(scheduler, num_inference_steps, device, timesteps)
scheduler.set_timesteps(num_inference_steps, device=device)  # called again by source
latents = randn(shape) * scheduler.init_noise_sigma
for t in timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  noise_pred = transformer(latent_model_input, prompt_embeds, masks, t, rope)
  if CFG:
    noise_pred = uncond + guidance_scale * (text - uncond)
  latents = scheduler.step(noise_pred, t, latents, generator=generator)
```

Euler ancestral step keeps `_step_index`, uses CPU `sigmas` in source, converts
epsilon to `pred_original_sample = sample - sigma * model_output`, computes the
ODE derivative, and adds random noise scaled by `sigma_up`. First Dinoml parity
should keep timestep iteration, sigmas, and random noise source host-visible.
The stochastic branch means deterministic validation must pass a fixed generator
or an explicit variance noise tensor equivalent.

## 9. Position, timestep, and custom math

- Allegro 3D RoPE uses per-axis frequency tables for T/H/W. For head dim 96,
  each axis gets 32 dimensions.
- Pipeline builds positions with `torch.cartesian_prod(grid_t, grid_h, grid_w)`,
  reshaped to `[3,1,seq]`.
- Timestep conditioning is Diffusers `AdaLayerNormSingle`; per-block
  `scale_shift_table` combines learned constants with timestep projections to
  produce six modulation tensors.
- Final output scale/shift uses a separate two-way table plus embedded timestep.
- RoPE tables and position grids can be precomputed per latent T/H/W and
  interpolation scale.
- Scheduler timesteps/sigmas can be precomputed per scheduler config and
  inference step count, but stochastic step noise remains dynamic.

## 10. Preprocessing and input packing

Text preprocessing:

- Optional caption cleaning uses BeautifulSoup/ftfy helpers when available.
- Tokenization pads/truncates to `max_sequence_length`, default 512.
- T5 output and attention mask are duplicated for videos per prompt, though the
  call path forces `num_videos_per_prompt = 1`.
- Negative prompt defaults to empty string and is encoded separately for CFG.

Latent and video preprocessing:

- T2V starts from random NCTHW latent noise.
- Even `num_frames` maps to `ceil(num_frames / 4)` latent frames; odd maps to
  `ceil((num_frames - 1)/4) + 1`.
- Pipeline defaults output size from transformer latent size times VAE scale:
  88 frames, 720 height, 1280 width.
- Decode scales latents by `1 / vae.config.scaling_factor`.
- VAE tiled decode crops latent cubes, decodes them, blends overlaps, returns
  BTCHW video, and the pipeline crops to requested output dimensions.

NDHWC guarded notes:

- Preserve NCTHW/BTCHW source semantics initially.
- NDHWC islands are plausible only inside VAE Conv3d/Conv2d regions after
  rewriting channel axes for GroupNorm, posterior split, temporal pad/cat,
  overlap blending axes, Conv weights, and VideoProcessor boundary layout.
- Transformer token core is layout-neutral after patchify, but patchify and
  unpatchify should be protected by `no_layout_translation()` until exact token
  order and Conv2d weight transforms are tested.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Allegro spatial video patchify

Source pattern:

```text
NCTHW -> BTCHW flatten to [B*T,C,H,W]
-> PatchEmbed Conv2d(k=s=2)
-> [B,T,H/2*W/2,inner] -> [B,T*H/2*W/2,inner]
```

Replacement: explicit video spatial patchify op plus GEMM/Conv provider.

Preconditions: `patch_size_t=1`, `patch_size=2`, latent H/W divisible by 2,
source token order preserved. Weight transform is source OIHW for faithful NCHW
or OIHW->HWIO only inside a guarded NHWC provider. Failure cases: TI2V unknown
source, alternate patch sizes, layout-translated consumers.

### Rewrite: Allegro RoPE attention prelude

Source pattern:

```text
Q/K/V projections -> reshape heads -> Allegro per-axis RoPE gather/apply -> SDPA
```

Replacement: canonical attention call with explicit RoPE pre-op.

Preconditions: head dim divisible by 3 and each chunk even, noncausal attention,
supported mask shape, provider supports sequence/workspace. Failure cases:
cross-attention, active unusual masks, sequence too large for provider.

### Rewrite: CFG + scheduler staging

Source pattern:

```text
cat latents -> denoiser -> chunk -> uncond + scale * (text - uncond)
-> Euler ancestral step with stochastic noise
```

Replacement: plan-visible CFG batch strategy and scheduler pointwise kernels.

Preconditions: same latent shape for positive/negative paths, epsilon
prediction, fixed scheduler config. Failure cases: guidance variants, scheduler
swaps, stochastic noise not made explicit.

### Rewrite: VAE tiled blend

Source pattern:

```text
crop latent cube -> post_quant_conv -> decoder -> overlap ramp multiply -> add into output
```

Replacement: explicit tile schedule plus overlap blend kernels.

Preconditions: tile kernel/stride covers requested latent grid, batch size 1 as
source local tile batch, NCTHW input and BTCHW output represented. Failure cases:
non-tiled path unavailable, small shapes that produce zero tiles, NDHWC without
axis rewrites.

## 12. Kernel fusion candidates

Highest priority:

- QKV/out GEMMs and GEGLU feed-forward GEMMs for 2304-wide transformer blocks.
- RoPE gather/apply + attention provider prelude with strict sequence guards.
- AdaLayerNormSingle scale/shift/gate and residual epilogues.
- CFG arithmetic plus Euler ancestral pointwise update.
- VAE Conv2d/Conv3d + GroupNorm + SiLU residual blocks used by tiled decode.

Medium priority:

- Patchify/unpatchify layout kernels for NCTHW video latents.
- Tiled VAE overlap blend kernels across T/H/W.
- Scheduler table generation/cache and stochastic noise staging.
- Caption projection 4096 -> 2304 when admitting text encoder boundary.

Lower priority:

- Generic LoRA/adapter mutation.
- Scheduler swaps beyond official Euler ancestral.
- TI2V variant support until source exists.
- NDHWC VAE Conv3d islands after faithful NCTHW/BTCHW parity.

## 13. Runtime staging plan

Stage 1: Parse `rhymes-ai/Allegro` configs and load base transformer/VAE
weights; accept external prompt/negative prompt embeddings and masks.

Stage 2: Implement NCTHW latent contract, spatial patchify/unpatchify, Allegro
3D RoPE, one `AllegroTransformerBlock`, and full transformer random-tensor
parity at reduced shape.

Stage 3: Add CFG batched denoiser parity and one fixed-timestep noise prediction
parity.

Stage 4: Implement official Euler ancestral epsilon scheduler slice with
explicit stochastic noise and host-visible step index/sigmas.

Stage 5: Add latent-only short denoising loop parity with supplied prompt
embeddings.

Stage 6: Add `AutoencoderKLAllegro` tiled decode for official tile-compatible
default geometry; keep non-tiled decode unsupported because source does.

Stage 7: End-to-end smoke for base Allegro at default or tile-compatible reduced
geometry, then performance probes.

Stage 8: Revisit `allegro_ti2v` only after matching
`AllegroTransformerTI2V3DModel` source is available.

## 14. Parity and validation plan

- Config/default reconciliation for base Allegro and TI2V blocked-source
  detection.
- Prompt embedding/mask duplication and CFG concat parity with supplied T5
  embeddings.
- Patchify/unpatchify parity for `[B,4,T,90,160]` and smaller H/W multiples of
  2.
- Allegro RoPE table and position-grid parity for default `22x45x80` token grid.
- One `AllegroTransformerBlock` parity at inner dim 2304.
- Full `AllegroTransformer3DModel` forward parity at reduced and default token
  counts as memory allows.
- Euler ancestral `set_timesteps`, `scale_model_input`, and one `step` parity
  with fixed random noise/generator.
- CFG arithmetic parity.
- VAE tiled decode parity for default `[B,4,22,90,160] -> [B,88,3,720,1280]`.
- Output crop/postprocess smoke for PIL/NumPy output.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer/VAE
  fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One transformer step by latent grid: reduced synthetic, default 88x720x1280,
  and frame-count sweeps.
- Attention backend comparison at head dim 96 and long sequences.
- Block time split: patchify, QKV/attention, cross-attention, FFN, adaptive
  modulation.
- CFG memory/time for batch concat versus two separate calls.
- Euler ancestral scheduler overhead and RNG/noise cost.
- Tiled VAE decode throughput by H/W tile count and overlap size.
- VRAM/workspace peaks for transformer attention and VAE tiled decode.
- Faithful NCTHW/BTCHW versus guarded NDHWC VAE Conv3d island after parity.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `allegro_ti2v`: official configs reference `AllegroTransformerTI2V3DModel`,
  but matching source is absent in the inspected checkout.
- `allegro_vae_tiling`: current VAE source requires tiled encode/decode; tile
  scheduling and overlap blending are runtime-significant.
- `allegro_scheduler_swaps`: constructor accepts `KarrasDiffusionSchedulers`,
  while official parity requires Euler ancestral first.
- `allegro_lora_adapters`: only generic loader surfaces; defer adapter state to
  a separate mutation/loading report.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- ControlNet, T2I-Adapter, IP-Adapter, GLIGEN, img2img, inpaint, depth2img, and
  upscaling, because no Allegro-specific source surface was found in this
  checkout.

## 17. Final implementation checklist

- [ ] Parse `rhymes-ai/Allegro` model index and component configs.
- [ ] Reject or fence `rhymes-ai/Allegro-TI2V` until matching source is present.
- [ ] Load `AllegroTransformer3DModel` weights and accept external T5 embeds.
- [ ] Implement NCTHW latent contract and spatial patchify/unpatchify.
- [ ] Implement Allegro 3D RoPE table/index generation and application.
- [ ] Implement `AllegroTransformerBlock` attention, cross-attention, FFN, and adaptive gates.
- [ ] Implement full transformer forward parity.
- [ ] Implement CFG batch concat/chunk arithmetic.
- [ ] Implement official Euler ancestral epsilon scheduler with explicit RNG/noise.
- [ ] Add one-step and short-loop latent parity.
- [ ] Implement `AutoencoderKLAllegro` tiled decode for tile-compatible shapes.
- [ ] Add VAE scale and postprocess/crop parity.
- [ ] Benchmark transformer attention sequence lengths and tiled VAE decode.
- [ ] Add guarded NDHWC/VAE layout experiments only after faithful source-layout parity.
