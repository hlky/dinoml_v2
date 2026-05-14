# Diffusers SD3 ControlNet Operator and Integration Report

Candidate slug: `controlnet_sd3`

Status: focused audit report. This report covers `StableDiffusion3ControlNetPipeline`, `SD3ControlNetModel`, `SD3MultiControlNetModel`, and the SD3 ControlNet inpainting variant. It builds on the existing `stable_diffusion_3` report for the base SD3 text encoders, SD3 transformer, VAE, and FlowMatch scheduler.

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Base model context:
    stabilityai/stable-diffusion-3-medium-diffusers
    stabilityai/stable-diffusion-3.5-large
  SD3 ControlNet configs:
    InstantX/SD3-Controlnet-Canny
    InstantX/SD3-Controlnet-Depth
    InstantX/SD3-Controlnet-Pose
    InstantX/SD3-Controlnet-Tile
    alimama-creative/SD3-Controlnet-Inpainting
  Small/debug configs:
    DavyMorgan/tiny-controlnet-sd3
    DavyMorgan/tiny-controlnet-sd35
  Probed but blocked/incomplete:
    calcuis/sd3.5-large-controlnet

Config sources:
  Local cache already present:
    H:/configs/InstantX/SD3-Controlnet-{Canny,Depth,Pose,Tile}/model_index.json
    H:/configs/alimama-creative/SD3-Controlnet-Inpainting/model_index.json
    H:/configs/DavyMorgan/tiny-controlnet-{sd3,sd35}/model_index.json
    H:/configs/stabilityai/stable-diffusion-3-medium-diffusers/{transformer,vae,scheduler}/
    H:/configs/stabilityai/stable-diffusion-3.5-large/{transformer,vae,scheduler}/
  Network-inspected with `hf download` during this audit, then removed again
  to preserve the task's owned-write-path constraint:
    InstantX/SD3-Controlnet-Canny config.json
    InstantX/SD3-Controlnet-Depth config.json
    InstantX/SD3-Controlnet-Pose config.json
    InstantX/SD3-Controlnet-Tile config.json
    alimama-creative/SD3-Controlnet-Inpainting config.json
    DavyMorgan/tiny-controlnet-sd3 config.json
    DavyMorgan/tiny-controlnet-sd35 config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet_inpainting.py
  X:/H/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/controlnets/controlnet_sd3.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_sd3.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py

External component configs inspected:
  SD3 medium transformer, VAE, and scheduler configs from H:/configs/stabilityai.
  SD3.5 large transformer, VAE, and scheduler configs from H:/configs/stabilityai.

Any missing files or assumptions:
  Cached model_index.json files for the inspected ControlNet repos are empty
  `{}`, so transiently fetched component config.json files are the
  authoritative config evidence here.
  `calcuis/sd3.5-large-controlnet/config.json` returned HF 404 through the
  authenticated CLI path; no SD3.5-large ControlNet config fields are inferred.
  Safety, training, backend-specific XLA/NPU/MPS/ONNX/Flax, callback mutation,
  and multi-GPU/context-parallel paths are out of scope.
```

## 2. Pipeline and component graph

`StableDiffusion3ControlNetPipeline` registers the base SD3 components plus a single `SD3ControlNetModel`, a list/tuple wrapped into `SD3MultiControlNetModel`, or an already-wrapped multi-controlnet. Required components are `vae`, three text encoders/tokenizers, `transformer`, `scheduler`, and `controlnet`. Optional components are `image_encoder` and `feature_extractor` for IP-Adapter. The offload sequence is `text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae`.

```text
prompt strings or precomputed SD3 prompt embeddings
  -> CLIP1/CLIP2/T5 encode and SD3 prompt composition
  -> CFG negative/positive prompt batching
  -> control image preprocessing to NCHW RGB image tensor
  -> VAE encode control image to SD3 latent tensor
  -> latent initialization [B,16,H/8,W/8]
  -> denoising loop:
       CFG latent batch
       -> SD3ControlNetModel(latents, timestep, prompt/context, control latent)
       -> per-block token residuals
       -> SD3Transformer2DModel(..., block_controlnet_hidden_states=residuals)
       -> CFG arithmetic
       -> FlowMatchEulerDiscreteScheduler.step
  -> VAE decode/postprocess, or return latents
```

Independently cacheable stages: prompt embeddings and pooled embeddings, optional `controlnet_pooled_projections`, preprocessed/encoded control latents at fixed size, scheduler timesteps/sigmas, and caller-supplied initial latents.

Separate candidate reports and surfaces:

| Candidate | Classes/files | Delta |
| --- | --- | --- |
| `controlnet_sd3_multi` | `SD3MultiControlNetModel`, `controlnet_sd3.py` | Runs multiple SD3 ControlNets and sums corresponding token residual slots. |
| `controlnet_sd3_inpaint` | `StableDiffusion3ControlNetInpaintingPipeline`, same `SD3ControlNetModel` | Builds control condition as `[masked_image_latents, inverted_mask]` with `extra_conditioning_channels=1`. |
| `sd3_ip_adapter` | `SD3IPAdapterMixin`, `SD3IPAdapterJointAttnProcessor2_0` | Adds image embeddings into SD3 joint attention; inactive unless image embeds are supplied. |
| `sd3_lora_textual_inversion_runtime_adapters` | `SD3LoraLoaderMixin`, PEFT adapter paths | Mutates transformer and text encoder linear layers or prompt embeddings. |
| `sd3_img2img` | `StableDiffusion3Img2ImgPipeline` | Adds VAE encode of init image, noise add, and strength-based timestep slicing. |
| `sd3_inpaint_base` | `StableDiffusion3InpaintPipeline` | Adds mask and masked-image latent contracts independent of ControlNet. |
| `sd3_5_controlnet_large_or_8b` | SD3.5 large/8B ControlNet configs not fully available in cache | Source supports `use_pos_embed=False` and `joint_attention_dim=None`, but this needs official config evidence. |

## 3. Important config dimensions

Representative SD3 ControlNet configs:

| Repo | Class | in ch | extra cond ch | patch | layers | heads x dim | inner dim | joint dim | caption dim | pooled dim | pos max | qk norm | force zero pooled |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `InstantX/SD3-Controlnet-Canny` | `SD3ControlNetModel` | 16 | omitted -> 0 | 2 | 6 | 24 x 64 | 1536 | 4096 | 1536 | 2048 | 192 | omitted -> none | omitted -> `True` |
| `InstantX/SD3-Controlnet-Depth` | `SD3ControlNetModel` | 16 | omitted -> 0 | 2 | 12 | 24 x 64 | 1536 | 4096 | 1536 | 2048 | 192 | omitted -> none | omitted -> `True` |
| `InstantX/SD3-Controlnet-Pose` | `ControlNetSD3Model` in config, source class is `SD3ControlNetModel` | 16 | omitted -> 0 | 2 | 6 | 24 x 64 | 1536 | 4096 | 1536 | 2048 | 192 | omitted -> none | omitted -> `True` |
| `InstantX/SD3-Controlnet-Tile` | `SD3ControlNetModel` | 16 | omitted -> 0 | 2 | 6 | 24 x 64 | 1536 | 4096 | 1536 | 2048 | 192 | omitted -> none | omitted -> `True` |
| `alimama-creative/SD3-Controlnet-Inpainting` | `SD3ControlNetModel` | 16 | 1 | 2 | 23 | 24 x 64 | 1536 | 4096 | 1536 | 2048 | 192 | omitted -> none | omitted -> `True` |
| `DavyMorgan/tiny-controlnet-sd3` | `SD3ControlNetModel` | 8 | 0 | 1 | 1 | 4 x 8 | 32 | 32 | 32 | 64 | 96 | omitted -> none | omitted -> `True` |
| `DavyMorgan/tiny-controlnet-sd35` | `SD3ControlNetModel` | 8 | 0 | 1 | 1 | 4 x 8 | 32 | 32 | 32 | 64 | 96 | `rms_norm` | omitted -> `True` |

Base model context from cached configs:

| Base | Transformer | Latent ch | patch | layers | heads x dim | caption dim | pos max | VAE scale/shift | Scheduler |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| SD3 medium | `SD3Transformer2DModel` | 16 | 2 | 24 | 24 x 64 | 1536 | 192 | `1.5305 / 0.0609` | FlowMatch Euler, `shift=3.0` |
| SD3.5 large | `SD3Transformer2DModel` | 16 | 2 | 38 | 38 x 64 | 2432 | 192 | `1.5305 / 0.0609` | FlowMatch Euler, `shift=3.0` |

Recommended first Dinoml scheduler slice remains the SD3 FlowMatch Euler static-shift slice. ControlNet adds no new scheduler family.

## 3a. Family variation traps

- SD3 ControlNet residuals are transformer token residuals, not UNet NCHW down/mid residuals.
- Main ControlNet pipelines VAE-encode RGB control images first. The ControlNet model receives latent maps, not 3-channel images.
- `force_zeros_for_pooled_projection=True` is a source default and active for inspected configs that omit it. The main pipeline also sets VAE shift factor to zero for control-image encoding under this flag.
- The base transformer uses SD3 pooled prompt embeddings, but inspected ControlNets usually receive zero pooled projections.
- `num_layers` can be much smaller than the base transformer depth. Injection maps ControlNet residual slots to base transformer blocks by `int(index_block / (base_layers / control_layers))`.
- Inpaint ControlNet uses `extra_conditioning_channels=1`: `pos_embed_input` patchifies 17 latent+mask channels for SD3 medium inpaint ControlNet.
- Source supports `joint_attention_dim=None` and `use_pos_embed=False` for SD3.5 8B-style ControlNets. That branch expects tokenized hidden states and no text/context input, but official configs were not available in the inspected cache.
- Source tensors at image/VAE boundaries are NCHW. A channel-last pass must rewrite VAE image/latent axes, mask concat `dim=1`, interpolate spatial axes, and patch embedding channel assumptions.
- `guess_mode` is a parameter to `prepare_image`, but the SD3 ControlNet pipeline passes `guess_mode=False`; do not import SD1 ControlNet guess-mode behavior into first SD3 scope.

## 4. Runtime tensor contract

For SD3 medium at 1024 x 1024, CFG enabled, one image per prompt:

| Boundary | Tensor | Source layout | Shape |
| --- | --- | --- | --- |
| prompt embeds | `prompt_embeds` | sequence | `[2B, 77 + max_sequence_length, 4096]` after CLIP pad + T5 concat |
| pooled prompt embeds | `pooled_prompt_embeds` | batch | `[2B, 2048]` |
| preprocessed control image | `control_image` before VAE | NCHW RGB | `[2B,3,1024,1024]` in image processor range |
| encoded control latent | `controlnet_cond` | NCHW latent | `[2B,16,128,128]` for normal SD3 ControlNet |
| initial latents | `latents` | NCHW latent | `[B,16,128,128]` |
| denoiser/control input | `latent_model_input` | NCHW latent | `[2B,16,128,128]` |
| SD3 token stream | `hidden_states` after patch | token | `[2B,4096,1536]` |
| ControlNet block residuals | `controlnet_block_samples` | token | `num_control_layers` tensors, each `[2B,4096,1536]` for SD3 medium ControlNets |
| transformer output | `noise_pred` | NCHW latent | `[2B,16,128,128]`, then CFG chunked to `[B,16,128,128]` |
| scheduler output | `latents` | NCHW latent | `[B,16,128,128]` |
| VAE decode input | scaled/shifted latents | NCHW latent | `[B,16,128,128]` |
| final image | decoded/postprocessed | NCHW then PIL/np/pt | `[B,3,1024,1024]` before postprocess conversion |

For inpaint ControlNet, `prepare_image_with_mask` returns control condition directly:

```text
image -> normalize to [-1,1] -> masked image -> VAE encode -> [B,16,H/8,W/8]
mask  -> grayscale/binarize -> interpolate to [B,1,H/8,W/8] -> invert as 1 - mask
control_image = cat([image_latents, mask], dim=1) -> [B,17,H/8,W/8]
CFG duplicates to [2B,17,H/8,W/8]
```

CPU/data-pipeline work: PIL/NumPy conversion, resize, RGB or grayscale conversion, mask binarization, prompt tokenization. GPU/runtime work: VAE encode/decode, ControlNet forward, SD3 transformer forward with residual injection, CFG, scheduler step, and optional image postprocess to tensor.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image and latent tensors.
- Batch `cat`, `chunk`, `repeat_interleave`, `view`, `reshape`, `transpose`, `einsum`.
- Channel concat for inpaint control condition: `cat([image_latents, mask], dim=1)`.
- `interpolate(mask, size=(H/8,W/8))`.
- Residual list indexing by block interval and elementwise add into transformer hidden states.
- Multi-ControlNet elementwise residual summation across matching token slots.

Convolution/patch ops:

- VAE encode/decode convolutional stacks from SD3 AutoencoderKL.
- `PatchEmbed` for latent input: Conv2d-like patch projection from NCHW `[B,16,H,W]` to tokens.
- `pos_embed_input` zero-initialized PatchEmbed from `[B,16+extra,H,W]` to token residual conditioning.
- Final SD3 transformer unpatchify reshape/einsum inherited from base SD3.

GEMM/linear ops:

- `context_embedder: Linear(4096 -> caption_projection_dim)`.
- `time_text_embed` timestep + pooled projection MLP.
- Per-ControlNet residual heads: zero-initialized `Linear(inner_dim -> inner_dim)` for each ControlNet layer.
- Joint attention Q/K/V/add-Q/K/V/output projections.
- Feed-forward approximate GELU MLPs in joint transformer blocks.

Attention primitives:

- SD3 joint image/text attention via `JointAttnProcessor2_0`.
- Optional dual attention when `dual_attention_layers` is present, as in tiny SD35 config.
- Optional IP-Adapter joint attention branch is source-supported but out of first ControlNet slice.

Normalization and adaptive conditioning:

- `AdaLayerNormZero`, `AdaLayerNormContinuous`, optional `SD35AdaLayerNormZeroX`.
- `LayerNorm` over token hidden dim.
- Optional QK RMSNorm for SD3.5-style configs.

Scheduler and guidance arithmetic:

- FlowMatch Euler `set_timesteps`, custom `sigmas`, and `step`.
- True CFG: `uncond + guidance_scale * (text - uncond)`.
- Control guidance keep windows and `conditioning_scale` multiplication.

## 6. Denoiser/model breakdown

`SD3ControlNetModel` forward:

```text
hidden_states:
  if use_pos_embed: NCHW latents -> PatchEmbed + positional embedding
  else: already-tokenized hidden states

timestep + pooled_projections:
  CombinedTimestepTextProjEmbeddings -> temb

encoder_hidden_states:
  if joint_attention_dim is not None:
    Linear(joint_attention_dim -> caption_projection_dim)
  else:
    no text/context input, SD3.5 8B-style source branch

controlnet_cond:
  pos_embed_input PatchEmbed(in_channels + extra_conditioning_channels)
  hidden_states += pos_embed_input(controlnet_cond)

for each ControlNet transformer block:
  JointTransformerBlock or SD3SingleTransformerBlock
  save hidden_states token residual

for each saved residual:
  zero Linear(inner_dim -> inner_dim)
  multiply by conditioning_scale
return tuple/list of token residuals
```

`SD3Transformer2DModel` injection:

```text
for index_block, block in base transformer blocks:
  run block unless skip_layers says to skip
  if block_controlnet_hidden_states is not None and not block.context_pre_only:
    interval = len(base_blocks) / len(control_residuals)
    hidden_states += control_residuals[int(index_block / interval)]
```

For SD3 medium with 24 base blocks and InstantX 6-layer ControlNet, each residual is reused for four consecutive base blocks. For the 12-layer depth ControlNet, each residual is reused for two consecutive blocks. For alimama 23-layer inpaint ControlNet, the interval is non-integer (`24/23`), so the indexing pattern is uneven and must be reproduced exactly.

## 7. Attention requirements

The primary attention implementation is `attention_processor.py`, specifically `JointAttnProcessor2_0`, selected by `JointTransformerBlock` when PyTorch exposes `scaled_dot_product_attention`. It projects image tokens through `to_q/to_k/to_v`, text/context tokens through `add_q_proj/add_k_proj/add_v_proj`, reshapes to `[B, heads, seq, head_dim]`, applies optional QK normalization, concatenates image and text along sequence, runs noncausal dense SDPA, then splits image/text outputs and applies output projections.

Required first-slice attention:

- Joint noncausal image+text attention.
- No attention mask in the normal SD3 ControlNet path.
- Head dim 64 for production SD3 ControlNets, 8 for tiny configs.
- Context sequence is the base SD3 prompt sequence unless `joint_attention_dim=None`.
- Optional QK RMSNorm must be supported before claiming SD3.5-style coverage.

Flash-style Dinoml provider candidate:

- Valid under guards for dense noncausal Q/K/V, dropout 0, no unsupported IP-Adapter branch, supported dtype/head dim, and exact image/text concat/split semantics.
- Fused projections are source-supported through `fuse_qkv_projections()` and `FusedJointAttnProcessor2_0`, but default configs do not require fused weights.
- Eager/native SDPA semantics define parity.

## 8. Scheduler and denoising-loop contract

ControlNet does not change SD3 scheduler math. The pipeline uses `FlowMatchEulerDiscreteScheduler` and the copied `retrieve_timesteps` helper with `num_inference_steps` or custom `sigmas`.

Per-step loop:

```text
latent_model_input = cat([latents, latents]) if guidance_scale > 1 else latents
timestep = t.expand(latent_model_input.batch)
cond_scale = controlnet_conditioning_scale * controlnet_keep[i]
control_block_samples = controlnet(latent_model_input, timestep, prompt/context, control_image, cond_scale)
noise_pred = transformer(latent_model_input, timestep, prompt/context, block_controlnet_hidden_states)
noise_pred = uncond + guidance_scale * (text - uncond) if CFG
latents = scheduler.step(noise_pred, t, latents)
```

`controlnet_keep` is host-computed from `control_guidance_start/end` as:

```text
1.0 - float(i / len(timesteps) < start or (i + 1) / len(timesteps) > end)
```

Keep scheduler table generation, control keep-window selection, and Python loop state host-visible first. Compile ControlNet, transformer residual injection, CFG arithmetic, and FlowMatch step kernels after one-step parity is established.

## 9. Position, timestep, and custom math

Position and timestep math is SD3-style:

- `PatchEmbed` supplies 2D positional embeddings for latent tokens when `use_pos_embed=True`.
- `pos_embed_input` intentionally has `pos_embed_type=None`; it patchifies the control condition without adding a second positional table.
- `CombinedTimestepTextProjEmbeddings` consumes timestep and pooled projections.
- `force_zeros_for_pooled_projection=True` makes the pipeline use `zeros_like(pooled_prompt_embeds)` for the ControlNet pooled projection and zero shift factor for normal control-image VAE encode.
- `JointTransformerBlock` gates attention and MLP residuals from adaptive norm outputs.
- `SD3Transformer2DModel` unpatchifies with `reshape [B,Ht,Wt,p,p,C]`, `einsum("nhwpqc->nchpwq")`, then final NCHW reshape.

Dynamic dependencies: image size controls token length; prompt length controls joint attention sequence; `num_layers` controls residual tuple length; `control_guidance_start/end` controls per-step zeroing; `guidance_scale` controls CFG batch width.

## 10. Preprocessing and input packing

Main SD3 ControlNet:

- `prepare_image` calls `VaeImageProcessor.preprocess(image, height, width)` for non-tensor input, repeats to prompt batch, casts to transformer dtype/device, and duplicates for CFG.
- It then VAE-encodes the image and applies `(latent - vae_shift_factor) * vae.config.scaling_factor`.
- Under inspected configs, `force_zeros_for_pooled_projection=True` sets `vae_shift_factor=0` for the control image path.
- The control condition passed to the model is latent NCHW `[2B,16,H/8,W/8]`, not RGB.

Inpaint SD3 ControlNet:

- `image_processor` has resize, RGB conversion, and normalization enabled.
- `mask_processor` has resize, grayscale conversion, no normalization, and binarization enabled.
- The masked image sets masked RGB pixels to `-1`, VAE-encodes, applies normal SD3 `(latent - shift) * scale`, interpolates the mask to latent size, inverts it, and channel-concats to `[B,17,H/8,W/8]`.
- `extra_conditioning_channels=1` in the alimama config matches this 17-channel patch embedding.

## 11. Graph rewrite / lowering opportunities

### Rewrite: SD3 ControlNet block residual injection

Source pattern: `controlnet` emits `N` token residuals; base transformer with `M` blocks adds `residual[int(index / (M/N))]` after each non-`context_pre_only` block.

Replacement pattern: represent residual tuple as explicit graph inputs to the transformer, with a generated block-index-to-residual-slot map.

Preconditions: `M`, `N`, block `context_pre_only` flags, and residual shapes are known from configs/source. Preserve Python `int()` floor behavior for non-integer intervals.

Failure cases: SD3.5/8B branches with different block classes or unavailable configs; skip-layer guidance changing block execution; residual count mismatch.

Parity sketch: compare base SD3 transformer with random residual tensors for 6, 12, and 23 residual slots against Diffusers at fixed timestep and prompt embeddings.

### Rewrite: VAE control-image encode cache

Source pattern: every request preprocesses and VAE-encodes the control image before the denoising loop.

Replacement pattern: admit an externally supplied `controlnet_cond` latent tensor or cache encoded control latents keyed by image, size, VAE config, dtype, `force_zeros_for_pooled_projection`, and inpaint mask.

Preconditions: same VAE weights/config, same resize/normalize/mask preprocessing, same scale/shift formula.

Failure cases: tensor input already preprocessed, dynamic image size, stochastic VAE sampling if generator behavior must be reproduced exactly.

### Rewrite: PatchEmbed as Conv2d + flatten

Source pattern: NCHW latent or control condition goes through `PatchEmbed`.

Replacement pattern: lower to strided Conv2d/linear patch projection plus flatten/transpose and optional positional add.

Preconditions: static patch size and channel count; preserve NCHW flatten order and positional embedding crop/shape behavior.

Weight transform: keep OIHW conv layout for faithful NCHW first; optional HWIO only inside guarded NHWC islands.

Failure cases: `use_pos_embed=False` token-input branch; dynamic sizes outside positional max; inpaint `extra_conditioning_channels`.

### Rewrite: Multi-ControlNet residual reduction

Source pattern: loop over controlnets and elementwise add corresponding residual slots.

Replacement pattern: explicit residual accumulation per slot, optionally fused with per-ControlNet scale.

Preconditions: identical residual count, token length, hidden width, dtype.

Failure cases: mixed SD3/SD3.5 ControlNets, heterogeneous layer counts, dynamic list length.

## 12. Kernel fusion candidates

Highest priority:

- PatchEmbed/pos_embed_input lowering and flatten/transpose cleanup for 16-channel and 17-channel latent maps.
- Joint attention Q/K/V/add-Q/K/V projections, optional QK norm, SDPA, output projection.
- AdaLayerNorm gating plus residual epilogues in `JointTransformerBlock`.
- ControlNet residual head `Linear + conditioning_scale` and transformer residual add.
- CFG arithmetic and FlowMatch Euler step fusion inherited from SD3.

Medium priority:

- VAE encode path for control image and inpaint masked image; this is on the critical path before every denoising loop unless cached.
- Multi-ControlNet residual accumulation.
- Guarded NHWC VAE/patch-prelude conv islands, with NCHW ABI boundaries.
- Inpaint mask interpolate, invert, channel concat, and masked pixel fill kernels.

Lower priority:

- Fused QKV weight transforms through Diffusers' experimental fused processor path.
- SD3.5 QK RMSNorm and dual-attention specialization until representative production ControlNet configs are available.
- IP-Adapter attention branch in the same pipeline.

## 13. Runtime staging plan

Stage 1: Parse base SD3 medium configs plus one InstantX 6-layer ControlNet config. Treat prompt embeddings, pooled embeddings, latents, and encoded control latents as external inputs.

Stage 2: Lower `SD3ControlNetModel` with `use_pos_embed=True`, `joint_attention_dim=4096`, `extra_conditioning_channels=0`, and zero pooled projections. Validate tiny config first, then InstantX canny.

Stage 3: Implement residual tuple ABI and SD3 transformer block residual injection. Validate one denoising step with fixed prompt embeddings, control latent, timestep, and latents.

Stage 4: Add VAE control-image encode preprocessing for normal ControlNet, including the `force_zeros_for_pooled_projection` shift-factor branch.

Stage 5: Run full denoising loop with scheduler in Python and compiled/control-lowered ControlNet + transformer step.

Stage 6: Add 12-layer depth and 23-layer inpaint residual-slot mapping. Include exact non-integer interval mapping for 23/24.

Stage 7: Add inpaint control condition `[masked_image_latents, inverted_mask]` with `extra_conditioning_channels=1`.

Stage 8: Add Multi-ControlNet residual summation.

Stage 9: Add attention, norm, patch, and VAE encode/decode fusions under guards.

## 14. Parity and validation plan

- Config parser tests for InstantX canny/depth/pose/tile, alimama inpaint, and DavyMorgan tiny SD3/SD35.
- Random tensor parity for `PatchEmbed` and `pos_embed_input` with `[B,16,128,128]` and `[B,17,128,128]`.
- `SD3ControlNetModel.forward` parity for tiny SD3 and InstantX canny at fixed timestep, prompt embeddings, pooled projections, and control latent.
- Verify residual count, order, shape, dtype, and scale for 6, 12, and 23-layer configs.
- Transformer injection parity with random residual tuples and exact block-slot mapping.
- One denoising-step parity: ControlNet -> SD3 transformer -> CFG -> FlowMatch step.
- VAE encode parity for normal control image: with and without zero shift factor.
- Inpaint preprocessing parity: mask binarization, masked image fill, VAE encode, mask resize/invert, 17-channel concat.
- Multi-ControlNet parity with two identical tiny ControlNets and independent scales/windows.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 begin at `rtol=2e-2, atol=2e-2`, then tighten by kernel.

## 15. Performance probes

- ControlNet forward time by layer count: 1, 6, 12, 23 residual blocks.
- Base SD3 transformer time with and without residual injection.
- One denoising step split: ControlNet, transformer, CFG, scheduler.
- VAE control-image encode cost versus denoising loop cost; cache hit/miss timing.
- Inpaint preprocessing and VAE encode cost.
- Attention backend comparison for joint attention at 4096 image tokens plus SD3 prompt tokens.
- Residual injection bandwidth and Multi-ControlNet accumulation cost.
- NCHW faithful path versus guarded NHWC VAE/patch-prelude paths.
- VRAM and temporary tensor usage for CFG duplicated control latents.

## 16. Scope boundary and separate candidates

Separate candidates, not ignored:

- `controlnet_sd3_inpaint`: included here as required variant inventory, but deserves a separate implementation slice for mask/image latent contracts.
- `controlnet_sd3_multi`: list inputs and residual reduction across multiple ControlNets.
- `sd3_5_controlnet_large_or_8b`: source branches for `use_pos_embed=False` and `joint_attention_dim=None`; blocked on representative official configs.
- `sd3_ip_adapter`: image embeddings and `SD3IPAdapterJointAttnProcessor2_0`.
- `sd3_lora_textual_inversion_runtime_adapters`: adapter and embedding mutation.
- `sd3_img2img` and base `sd3_inpaint`: VAE encode/noise/timestep contracts without ControlNet.
- Advanced FlowMatch scheduler variants: dynamic shift, custom timesteps, stochastic/per-token paths.

Ignored/out of scope:

- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Callback mutation and interactive interrupt.
- Multi-GPU/context parallel paths.
- XLA, NPU, MPS, Flax, ONNX, and Core ML-specific branches.

## 17. Final implementation checklist

- [ ] Parse SD3 medium base configs and one `SD3ControlNetModel` config.
- [ ] Load SD3 ControlNet weights separately from base transformer weights.
- [ ] Accept external prompt embeddings, pooled embeddings, latents, timestep, and encoded control latent.
- [ ] Implement/lower `PatchEmbed` for latent and control condition inputs.
- [ ] Implement `CombinedTimestepTextProjEmbeddings` and `context_embedder`.
- [ ] Lower `JointTransformerBlock` for ControlNet, including adaptive norm, joint attention, FFN, and optional QK norm guard.
- [ ] Implement zero linear residual heads and conditioning scale.
- [ ] Implement residual tuple ABI and exact SD3 transformer block injection mapping.
- [ ] Add FlowMatch one-step parity with CFG.
- [ ] Add VAE control-image encode preprocessing and cache admission.
- [ ] Add inpaint 17-channel control condition support.
- [ ] Add Multi-ControlNet residual accumulation.
- [ ] Add guarded patch/attention/norm/VAE fusions.
- [ ] Benchmark ControlNet, transformer, VAE encode/decode, scheduler/guidance, and residual injection separately.
