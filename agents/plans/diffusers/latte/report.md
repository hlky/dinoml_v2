# Diffusers Latte Operator and Integration Report

Target slug: `latte`

Runtime scope: non-deprecated `LattePipeline` text-to-video generation with external/cached T5 prompt embeddings allowed for the first Dinoml slice, DDIM scheduler in host-visible loop state, `LatteTransformer3DModel` denoiser on NCTHW latents, and framewise `AutoencoderKL` decode. Ignore XLA/NPU/MPS/Flax/ONNX, safety, training, callbacks, and multi-GPU/context parallel.

## 1. Source basis

```text
Diffusers commit/version:
  X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  maxin-cn/Latte-1
  maxin-cn/Latte, checked only as an old/partial repo reference.

Config sources:
  Local cache:
    H:/configs/maxin-cn/Latte-1/model_index.json
    H:/configs/maxin-cn/Latte/model_index.json, empty {}
    H:/configs/maxin-cn/Cinemo/model_index.json, unrelated StableDiffusionPipeline
  Official raw/API HF reads, not saved because this worker owns only this report:
    maxin-cn/Latte-1/model_index.json
    maxin-cn/Latte-1/transformer/config.json
    maxin-cn/Latte-1/scheduler/scheduler_config.json
    maxin-cn/Latte-1/vae/config.json
    maxin-cn/Latte-1/text_encoder/config.json
    maxin-cn/Latte-1/tokenizer/tokenizer_config.json
    maxin-cn/Latte-1/vae_temporal_decoder/config.json
    maxin-cn/Latte-1/text_encoder/model.safetensors.index.json
    maxin-cn/Latte-1 README and API metadata

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/latte/pipeline_latte.py
  X:/H/diffusers/src/diffusers/pipelines/latte/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/latte_transformer_3d.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py, by config inventory only
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_ddim.py
  X:/H/diffusers/src/diffusers/video_processor.py

External component configs inspected:
  T5EncoderModel and T5Tokenizer configs from maxin-cn/Latte-1.

Any missing files or assumptions:
  maxin-cn/Latte-1 is public and ungated; no authenticated retry was needed.
  maxin-cn/Latte raw full pipeline paths returned 404 except an old VAE config,
  so no full current operator contract is inferred from it. The repo lists a
  `vae_temporal_decoder` component, but `model_index.json` wires `vae` to
  `AutoencoderKL`; temporal decoder is inventoried as a separate candidate.
```

## 2. Pipeline and component graph

`LattePipeline` components are `tokenizer`, `text_encoder`, `vae`, `transformer`, and `scheduler` (`pipeline_latte.py:179`). `tokenizer` and `text_encoder` are optional components (`pipeline_latte.py:170`), which makes externally supplied prompt embeddings a clean first runtime boundary.

```text
prompt / negative prompt
  -> caption cleanup + T5Tokenizer(max_length=120 in pipeline)
  -> T5EncoderModel hidden states [B,L,4096]
  -> optional attention-mask text trimming/masking
  -> latent noise [B,4,F,H/8,W/8] NCTHW
  -> denoising loop:
       optional CFG batch concat [negative, positive]
       DDIM scale_model_input
       LatteTransformer3DModel(latents, T5 embeds, timestep)
       true CFG arithmetic
       learned-sigma channel trim
       DDIM scheduler.step
  -> permute latents to frame batch [B*F,4,H/8,W/8]
  -> AutoencoderKL decode in chunks
  -> reshape to [B,3,F,H,W] and VideoProcessor postprocess
```

Required first-slice components:

| Component | Class | Anchor | Notes |
| --- | --- | --- | --- |
| Pipeline | `LattePipeline` | `pipeline_latte.py:145` | Text-to-video, offload sequence `text_encoder->transformer->vae`. |
| Denoiser | `LatteTransformer3DModel` | `latte_transformer_3d.py:27` | Alternating spatial text-conditioned and temporal self-attention blocks. |
| Scheduler | `DDIMScheduler` | `scheduling_ddim.py:139` | Official config is epsilon, linear beta, `clip_sample=false`. |
| Text encoder | `T5EncoderModel` | external transformers config | T5 v1.1 XXL-like encoder, may be external first. |
| Tokenizer | `T5Tokenizer` | external tokenizer config | Pipeline truncates/pads to 120 tokens despite tokenizer `model_max_length=512`. |
| VAE | `AutoencoderKL` | `autoencoder_kl.py` | 2D framewise decode from video latents. |

Separate candidate reports:

| Surface | Latte status | Candidate |
| --- | --- | --- |
| LoRA/runtime adapters | `LattePipeline` does not mix in LoRA loader classes; `LatteTransformer3DModel` inherits shared model mixins but no Latte-specific loader path exists. | `latte_lora_adapters` only for a concrete artifact. |
| Textual inversion | Not wired; tokenizer/text encoder are T5, no textual inversion mixin on the pipeline. | None for base. |
| IP-Adapter | Not wired; no image encoder, image projection, or IP attention processor in Latte pipeline. | None for base. |
| ControlNet / T2I-Adapter / GLIGEN | No Latte pipeline variants or side-input models. Shared `BasicTransformerBlock` has GLIGEN code, but Latte passes no `cross_attention_kwargs`. | Fork-specific only. |
| img2img / inpaint / depth2img / upscaling | No files in `pipelines/latte`; base has no VAE encode or mask/image/depth condition path. | Community variants only. |
| Temporal decoder VAE | `maxin-cn/Latte-1` includes `vae_temporal_decoder/*`, but model index does not register it. | `latte_temporal_decoder_vae`. |
| Scheduler swaps | Constructor accepts `KarrasDiffusionSchedulers`; source retrieve helper can pass custom timesteps/sigmas if supported. | `latte_scheduler_swaps` after DDIM parity. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline config | Transformer | Scheduler | VAE | Config status |
| --- | --- | --- | --- | --- | --- |
| `maxin-cn/Latte-1` | `LattePipeline` | `LatteTransformer3DModel` 28 layers, 16 heads, head dim 72 | DDIM epsilon, linear beta | `AutoencoderKL`, scalar scale 0.18215 | Official full component configs accessible. |
| `maxin-cn/Latte` | local `{}`; raw full paths mostly 404 | Not inferred | Not inferred | old SD VAE config only | Not a full Diffusers pipeline source for this audit. |

`maxin-cn/Latte-1` transformer config:

| Field | Value | Runtime effect |
| --- | --- | --- |
| `sample_size` | 64 | Default latent H/W; with VAE scale 8 gives 512x512 output. |
| `in_channels` / `out_channels` | 4 / 8 | Model predicts doubled channels; pipeline trims to 4 for non-learned scheduler variance. |
| `patch_size` | 2 | Spatial token grid is 32x32 at 512 output. |
| `num_layers` | 28 spatial blocks plus 28 temporal blocks | Each layer pair does text-conditioned spatial attention and optional temporal attention. |
| `num_attention_heads` x `attention_head_dim` | 16 x 72 | Inner dim 1152. |
| `cross_attention_dim` | 1152 | After caption projection from T5 4096. |
| `caption_channels` | 4096 | T5 hidden width. |
| `activation_fn` | `gelu-approximate` | Feed-forward activation in shared `FeedForward`. |
| `attention_bias` | true | Q/K/V projections use bias. |
| `norm_type` | `ada_norm_single` | PixArt-style adaptive norm/gated residual path. |
| `norm_elementwise_affine` | false | LayerNorms are non-affine in active blocks. |
| `norm_eps` | `1e-6` | Differs from source default `1e-5`. |
| `num_embeds_ada_norm` | 1000 | Timestep embedding training range. |

Text/VAE/scheduler:

| Component | Config-derived fields |
| --- | --- |
| T5 | `d_model=4096`, `num_layers=24`, `num_heads=64`, `d_ff=10240`, gated GELU, `torch_dtype=float16`, total indexed text weights about 19.0 GB. |
| Tokenizer | T5 tokenizer, vocab 32128 through text config, 100 extra IDs, `model_max_length=512`; pipeline uses `max_length=120`. |
| VAE | SD-style 2D `AutoencoderKL`, `latent_channels=4`, block channels `[128,256,512,512]`, `layers_per_block=2`, GroupNorm 32, `scaling_factor=0.18215`, `force_upcast=true`, spatial scale factor 8 from pipeline source. |
| DDIM | `beta_start=0.0001`, `beta_end=0.02`, `beta_schedule=linear`, `num_train_timesteps=1000`, `prediction_type=epsilon`, `variance_type=fixed_small`, `thresholding=false`, `clip_sample=false`, `timestep_spacing=leading`. |

Recommended first Dinoml scheduler slice: DDIM epsilon with `clip_sample=false`, `thresholding=false`, `eta=0`, `timestep_spacing=leading`, and no learned variance. This exactly matches `Latte-1` and is the same stateless alpha-product family staged in `scheduler_matrix`.

## 3a. Family variation traps

- Latte is text-conditioned but not joint text-video attention. Spatial blocks do self-attention over each frame's patches, then cross-attention to T5 embeddings; temporal blocks do self-attention over frames for each spatial patch.
- Source latents are NCTHW. The transformer immediately permutes to frame-major 2D NCHW before patch embedding, then returns NCTHW. NDHWC is only a guarded optimization candidate inside VAE/patch islands.
- `video_length` is a runtime pipeline argument, but `LatteTransformer3DModel` registers `temp_pos_embed` for constructor `video_length` default 16. A different frame count needs validation or config/model construction that matches the buffer length.
- `enable_temporal_attentions=False` is a real source branch. First parity should keep the default `True`, but tests should guard the branch rather than assume temporal blocks always execute.
- Prompt masking is config-sensitive: for batch size 1 the source trims to `attention_mask.sum()` tokens; for batch size >1 it masks/preserves the full 120-token length.
- CFG concat order is negative/unconditional first, positive second (`pipeline_latte.py:764`). Do not reuse DiT's class-conditioned positive/null order.
- Output channels are doubled to 8, but official DDIM `variance_type=fixed_small` means the pipeline chunks on channel dim and keeps the first 4 channels (`pipeline_latte.py:836`).
- The pipeline decodes frames independently through a 2D VAE, not a native temporal video VAE. The separate `vae_temporal_decoder` repo files are not active in `model_index.json`.
- The text encoder is much larger than the denoiser support work may initially want. Treat prompt embeddings as an independently cacheable/external stage first.

## 4. Runtime tensor contract

Default 512x512, 16-frame, CFG-enabled run with batch `B`:

| Boundary | Tensor | Shape/layout | Notes |
| --- | --- | --- | --- |
| token ids | `input_ids` | `[B,120]` int | Pipeline max length 120. |
| attention mask | `attention_mask` | `[B,120]` bool/int | Used only to trim/mask text embeddings, not passed into transformer in `__call__`. |
| prompt embeds | T5 output | `[B,L,4096]` | `L<=120` for batch-1 masking; usually 120 for multi-batch. |
| CFG prompt embeds | concat | `[2B,L,4096]` | Negative then positive when `guidance_scale>1`. |
| latent noise | `latents` | `[B,4,16,64,64]` NCTHW | Generated or supplied, multiplied by `scheduler.init_noise_sigma` 1. |
| model input | CFG + scale | `[2B,4,16,64,64]` | DDIM `scale_model_input` is identity. |
| timestep | `current_timestep` | `[2B]` | Expanded to model batch. |
| frame patch input | internal | `[2B*16,4,64,64]` NCHW | `permute(0,2,1,3,4).reshape`. |
| spatial tokens | internal | `[2B*16,1024,1152]` | `PatchEmbed` Conv2d 2x2 + flatten/transpose + 2D sin-cos. |
| spatial text | internal | `[2B*16,L,1152]` | T5 embeds projected 4096 -> 1152 and repeated per frame. |
| temporal tokens | internal | `[2B*1024,16,1152]` | Reshape/permute after each spatial block. |
| raw noise pred | transformer output | `[2B,8,16,64,64]` | NCTHW, doubled output channels. |
| guided pred | CFG result | `[B,8,16,64,64]` | `uncond + scale*(text-uncond)`. |
| scheduler model output | channel trim | `[B,4,16,64,64]` | First half only for `fixed_small` variance. |
| final latents | denoised | `[B,4,16,64,64]` | Passed to decode unless `output_type="latent"`. |
| VAE decode input | frame batch | `[B*16,4,64,64]` | Permute to `[B,F,C,H,W]`, flatten, divide by 0.18215. |
| decoded video | frames | `[B,3,16,512,512]` | VAE chunks by `decode_chunk_size`, then postprocess. |

Patchify/unpatchify:

```text
Patchify:
  [B,C,F,H,W] -> permute [B,F,C,H,W] -> [B*F,C,H,W]
  Conv2d(C -> 1152, kernel=2, stride=2, bias=true)
  flatten(2).transpose(1,2): [B*F,1152,H/2,W/2] -> [B*F,(H/2)*(W/2),1152]
  add fixed/interpolated 2D sin-cos position embedding

Temporal rotation:
  [B*F,P,D] -> [B,F,P,D] -> permute [B,P,F,D] -> [B*P,F,D]
  add 1D temporal sin-cos once before temporal block 0

Unpatchify:
  Linear(1152 -> 2*2*out_channels)
  reshape [B*F,H/2,W/2,2,2,Cout]
  einsum "nhwpqc->nchpwq"
  reshape [B*F,Cout,H,W] -> [B,Cout,F,H,W]
```

CPU/data-pipeline work: caption cleanup, tokenization, T5 if not cached, output conversion. GPU/runtime work: latent RNG or supplied latents, denoiser, CFG, scheduler pointwise step, VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCTHW latent allocation, multiply by scalar, batch concat/chunk, channel chunk, and NCTHW-to-frame-batch flatten/restore.
- Patch `reshape`, `permute`, `flatten`, `transpose`, `repeat_interleave`, `view`, `einsum`.
- Prompt embedding duplication and optional token masking/trimming.
- VAE decode frame chunk loop and `[B*F,C,H,W] <-> [B,C,F,H,W]`.

Convolution/downsample/upsample ops:

- `PatchEmbed` Conv2d(4 -> 1152, kernel 2, stride 2, bias true).
- AutoencoderKL framewise decode Conv2d/ResNet/GroupNorm/SiLU/upsample blocks.

GEMM/linear ops:

- Caption projection: Linear(4096 -> 1152) -> approximate GELU -> Linear(1152 -> 1152).
- Per block spatial self-attention Q/K/V/out and spatial cross-attention Q plus text K/V.
- Per block temporal self-attention Q/K/V/out.
- FeedForward MLP with approximate GELU in both spatial and temporal blocks.
- AdaLN-single timestep embedding and modulation Linear(1152 -> 6912) plus final Linear(1152 -> 2304-equivalent via two chunks after table add).
- Output projection Linear(1152 -> 32).

Attention primitives:

- Spatial self-attention over 1024 patch tokens per frame.
- Spatial cross-attention from 1024 patch tokens to `L<=120` text tokens.
- Temporal self-attention over 16 frames for each spatial patch.
- No RoPE, no QK norm, no attention mask in the active pipeline call, no added K/V.

Normalization/adaptive conditioning:

- LayerNorm non-affine eps `1e-6`.
- AdaLN-single shift/scale/gate around attention and MLP.
- Final LayerNorm plus shift/scale table.
- VAE GroupNorm.

Scheduler/guidance/VAE:

- DDIM epsilon step with alpha tables, deterministic `eta=0` first.
- True CFG arithmetic negative-first.
- Learned-sigma channel trim.
- VAE latent scale `latents / 0.18215`, framewise decode, postprocess to PIL/NumPy/PT.

## 6. Denoiser/model breakdown

`LatteTransformer3DModel.forward` (`latte_transformer_3d.py:166`):

```text
hidden_states [B,4,F,H,W] NCTHW
-> frame flatten to [B*F,4,H,W]
-> PatchEmbed to [B*F,P,1152]
-> AdaLayerNormSingle(timestep) -> modulation tensors and embedded timestep
-> caption projection [B,L,4096] -> [B,L,1152], repeat per frame
-> for each of 28 layer pairs:
     spatial BasicTransformerBlock on [B*F,P,D]
       AdaLN-single self-attn, text cross-attn, FFN
     if enabled:
       rotate to [B*P,F,D]
       add temporal position before first temporal block
       temporal BasicTransformerBlock with self-attn + FFN only
       rotate back to [B*F,P,D]
-> final AdaLN modulation + Linear to patch pixels
-> unpatchify to [B,8,F,H,W]
```

`BasicTransformerBlock` active Latte path (`attention.py:827`, `attention.py:989`):

- `norm_type="ada_norm_single"` builds a six-vector `scale_shift_table`.
- Self-attention gets `LayerNorm -> scale/shift -> Attention -> gate -> residual`.
- Spatial block also has cross-attention because `cross_attention_dim=1152`; for `ada_norm_single`, source skips norm2 before cross-attention (`attention.py:1031`).
- FFN gets `LayerNorm -> scale/shift -> FeedForward -> gate -> residual`.
- Temporal block has `cross_attention_dim=None`, so it has no `attn2`.

Inactive source branches for base Latte: class labels, AdaLN-zero, GLIGEN fuser, added image K/V, QK norm, RoPE, chunked feed-forward unless explicitly configured, and temporal attention disabled branch.

## 7. Attention requirements

Primary implementation is shared `Attention` plus `AttnProcessor2_0` (`attention_processor.py:2696`), which calls PyTorch `scaled_dot_product_attention`. Fused projection mutation is source-supported by shared `AttentionMixin.fuse_qkv_projections`, switching self-attn to fused QKV and cross-attn to fused KV, but the checkpoint format and first parity do not require it.

| Attention | Query | Key/value | Heads | Mask | Sequence shape at 512/16 |
| --- | --- | --- | --- | --- | --- |
| Spatial self | video patch tokens | same | 16 x 72 | none | `[B*F,1024,1152]` |
| Spatial cross | video patch tokens | T5 projected text | 16 x 72 | none in `__call__` | query 1024, key `L<=120` |
| Temporal self | per-spatial-position frame tokens | same | 16 x 72 | none | `[B*1024,16,1152]` |

Flash feasibility:

- Spatial self-attention and temporal self-attention are clean flash-style candidates: dense, noncausal, no masks, no RoPE/QK norm, head dim 72. Guard provider support for head dim 72, dtype, sequence length, and contiguous BNC layout.
- Spatial cross-attention is also feasible as a dense Q/K/V attention provider, but separate from self-attention because query length 1024 and text length up to 120 differ and K/V come from caption projection.
- Fused QKV is valid for temporal and spatial self-attention only; spatial cross-attention can fuse K/V but not QKV.
- Eager/native SDPA path defines parity.

## 8. Scheduler and denoising-loop contract

Latte uses the standard `retrieve_timesteps` helper copied from Stable Diffusion, with support for custom timesteps/sigmas only if the scheduler signature accepts them. Official `Latte-1` uses `DDIMScheduler`.

Loop contract:

```text
prompt_embeds = cat([negative, positive]) if guidance
timesteps = scheduler.set_timesteps(num_inference_steps)
latents = randn([B,4,F,H/8,W/8]) * init_noise_sigma
for t in timesteps:
  latent_model_input = cat([latents, latents]) if guidance else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  timestep_batch = expand(t, latent_model_input.batch)
  noise_pred = transformer(latent_model_input, prompt_embeds, timestep_batch)
  if guidance:
    uncond, text = noise_pred.chunk(2)
    noise_pred = uncond + guidance_scale * (text - uncond)
  if scheduler.variance_type not in learned variants:
    noise_pred = noise_pred.chunk(2, dim=1)[0]
  latents = scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
```

DDIM first slice:

- `scale_model_input` is identity.
- `prediction_type="epsilon"` gives `x0 = (sample - sqrt(beta_t) * eps) / sqrt(alpha_t)`.
- `clip_sample=false` and `thresholding=false` avoid clamp/reduction branches.
- `eta=0` avoids variance noise; pipeline accepts `eta` but first parity should fix it at 0.
- Scheduler state should expose timestep and alpha tables plus current loop index; the denoiser, CFG/channel trim, and pointwise DDIM update are compile candidates after one-step parity.

## 9. Position, timestep, and custom math

- Spatial position: `PatchEmbed` registers 2D sin-cos positional embeddings and interpolates if runtime latent H/W differ from configured `sample_size` (`embeddings.py:554`). First parity should use 512x512 output so latent H/W is 64 and token grid 32x32.
- Temporal position: `LatteTransformer3DModel` registers a non-persistent 1D sin-cos `temp_pos_embed` for configured `video_length` (`latte_transformer_3d.py:158`) and adds it once before the first temporal block.
- Timestep: `AdaLayerNormSingle` uses PixArt-style combined timestep embeddings, SiLU, and Linear to `6*inner_dim` (`normalization.py:235`). `use_additional_conditions=False`; resolution/aspect are passed as `None`.
- Caption projection: `PixArtAlphaTextProjection` is Linear -> tanh-approx GELU -> Linear (`embeddings.py:2191`).
- Final modulation: `scale_shift_table[2,D] + embedded_timestep`, chunked to shift/scale before final projection.

Precomputable: fixed spatial and temporal sin-cos tables for shape 64x64x16, prompt/negative embeddings, caption projection for static prompt if the runtime boundary is after projection, and scheduler alpha tables. Dynamic per step: timestep embeddings, AdaLN gates, CFG, and DDIM coefficients.

## 10. Preprocessing and input packing

Text preprocessing:

- Optional caption cleanup lowercases, strips, removes URLs/HTML and punctuation-heavy artifacts. Treat as CPU/data-pipeline.
- Tokenizer uses `padding="max_length"`, `max_length=120`, truncation, attention mask, and special tokens (`pipeline_latte.py:257`).
- If `mask_feature=True`, batch-1 prompt embeds are trimmed to the true token count and negative embeds are sliced to the same `keep_indices`; multi-batch embeds are masked but keep full length.
- `prompt_embeds` and `negative_prompt_embeds` can bypass text encoder work, but direct embeds must match shape when both are supplied.

Latent/video packing:

- No VAE encode in base T2V. Initial latent shape is `[B,4,F,H/8,W/8]`.
- No CogVideoX/Wan-style temporal VAE compression. The frame count is represented directly in latent dim 2.
- Model-internal patchify is per frame; temporal attention happens after patchify by rotating `[B*F,P,D]` to `[B*P,F,D]`.
- Decode uses 2D VAE frame chunks: `[B,4,F,H,W] -> [B*F,4,H,W] -> AutoencoderKL.decode -> [B,3,F,8H,8W]`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Latte frame patch embedding

Source pattern:

```text
NCTHW -> permute/reshape [B*F,C,H,W]
Conv2d(C -> D, k=s=2) -> flatten(2).transpose(1,2) -> add 2D pos
```

Replacement: explicit frame-patchify primitive or Conv2d plus token layout transform.

Preconditions: NCTHW source layout, H/W divisible by 2, configured positional grid or validated interpolation, patch size 2, Conv2d weight OIHW. NHWC lowering requires OIHW->HWIO transform and exact token-order parity. Failure cases: dynamic H/W with interpolation not implemented, layout pass changing frame/patch order.

Parity test: random `[B,4,16,64,64]` patchify against Diffusers.

### Rewrite: Spatial-temporal token rotation

Source pattern:

```text
[B*F,P,D] -> reshape [B,F,P,D] -> permute [B,P,F,D] -> [B*P,F,D]
temporal block
[B*P,F,D] -> [B,P,F,D] -> permute [B,F,P,D] -> [B*F,P,D]
```

Replacement: view/stride-aware token-rotation op or layout-planned transpose.

Preconditions: known `B`, `F`, `P`; contiguous after permute when required by provider. Failure cases: temporal attention disabled, dynamic frame count mismatched to `temp_pos_embed`.

Parity test: one layer pair with temporal attention on/off.

### Rewrite: Spatial cross-attention K/V reuse

Source pattern: caption projection once, repeat per frame, cross-attention in every spatial block.

Replacement: precompute projected text and optionally K/V per spatial block per prompt, with frame broadcast.

Preconditions: prompt embeddings fixed across denoising steps, no LoRA/processor mutation mid-run, no text mask passed to attention. Failure cases: adapter mutation, changing prompt embeds through callback-like path, future attention masks.

### Rewrite: CFG + channel trim + DDIM step

Source pattern:

```text
uncond, text = noise_pred.chunk(2)
guided = uncond + s * (text - uncond)
eps = guided[:, :4]
ddim_step(eps, latents, alpha_t)
```

Replacement: fused pointwise kernel directly consuming `[2B,8,F,H,W]` and emitting next `[B,4,F,H,W]`.

Preconditions: CFG enabled, negative-first ordering, scheduler variance not learned, DDIM epsilon config. Failure cases: `guidance_scale<=1`, learned variance scheduler, non-DDIM scheduler swap.

### Rewrite: Framewise VAE decode batching

Source pattern: `permute -> flatten frames -> chunked AutoencoderKL.decode -> reshape`.

Replacement: frame-batch decode stage with explicit chunk policy or compiled VAE island.

Preconditions: active VAE is 2D `AutoencoderKL`, no temporal decoder, scaling factor scalar, output frames independent. Failure cases: manually swapped `AutoencoderKLTemporalDecoder` or future temporal consistency decoder.

## 12. Kernel fusion candidates

Highest priority:

- GEMM/Linear coverage for 1152-wide attention/FFN/projection layers and T5 caption projection.
- Spatial self-attention, spatial cross-attention, and temporal self-attention providers with head dim 72.
- AdaLN-single scale/shift/gate plus residual epilogues around attention and FFN.
- Patchify/unpatchify and spatial-temporal token rotation kernels.
- CFG + channel trim + DDIM epsilon step fusion.

Medium priority:

- Framewise AutoencoderKL decode Conv2d/GroupNorm/SiLU/up-block NHWC island, shared with AutoencoderKL report.
- Prompt embedding mask/trim and repeat kernels when prompt embeddings are runtime GPU tensors.
- Caption K/V precompute/cache per prompt and block.
- DDIM table generation/cache and custom timestep admission.

Lower priority:

- Temporal decoder VAE support.
- Scheduler swaps beyond DDIM.
- Text encoder compilation; the 19 GB T5 encoder is best treated as external until the denoiser loop is stable.
- Generic LoRA/adapter state.

## 13. Runtime staging plan

Stage 1: Parse `maxin-cn/Latte-1` model index and component configs; accept externally supplied positive/negative T5 embeddings.

Stage 2: Implement NCTHW latent contract `[B,4,16,64,64]`, frame patchify, 2D/1D sin-cos position, and one spatial+temporal layer-pair parity.

Stage 3: Compile full `LatteTransformer3DModel` forward for default shape with temporal attention enabled; keep scheduler and VAE outside.

Stage 4: Add true CFG negative-first arithmetic and doubled-channel trim.

Stage 5: Add DDIM epsilon scheduler setup/step with host-visible loop state and one-step parity.

Stage 6: Add framewise `AutoencoderKL` decode boundary with `scaling_factor=0.18215`, no temporal decoder.

Stage 7: Short deterministic 16-frame denoising loop with external prompt embeddings and VAE decode.

Stage 8: Add optimized attention/AdaLN/patch/token-rotation fusions, then scheduler swaps or temporal decoder VAE as separate candidates.

## 14. Parity and validation plan

- Config/default reconciliation for `Latte-1` and source defaults.
- T5 prompt preprocessing/tokenization/masking parity for batch 1 and batch >1.
- Caption projection parity: `[B,L,4096] -> [B,L,1152]`.
- PatchEmbed parity for `[B,4,16,64,64]`.
- Temporal token rotation and `temp_pos_embed` add parity.
- One spatial `BasicTransformerBlock` parity with self + cross attention.
- One temporal `BasicTransformerBlock` parity with self attention only.
- Full transformer forward parity for default 512x512x16 and smaller synthetic grids where supported.
- CFG negative-first arithmetic and channel trim parity.
- DDIM `set_timesteps` and one epsilon step parity with `eta=0`.
- VAE decode parity for `[B,4,16,64,64] -> [B,3,16,512,512]` using decode chunks.
- Short deterministic loop smoke with fixed latents/prompt embeds.
- Suggested tolerances: fp32 scheduler `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step by batch, frame count, and resolution: default 16x512x512 plus small synthetic.
- Spatial self-attention length 1024 vs temporal self-attention length 16 vs cross-attention 1024xL timing.
- Attention backend comparison for head dim 72 and dtype fp16/bf16/fp32.
- Spatial-temporal token rotation overhead per layer pair.
- AdaLN-single and FFN time split.
- CFG doubled-batch memory and latency.
- DDIM scheduler overhead versus denoiser time.
- Framewise VAE decode throughput with `decode_chunk_size` 1, 4, 14, 16.
- VRAM and weight-load timing with external prompt embeddings vs full T5 loaded.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `latte_temporal_decoder_vae`: `maxin-cn/Latte-1/vae_temporal_decoder` uses `AutoencoderKLTemporalDecoder`; it changes decode from independent framewise 2D VAE to a temporal decoder contract.
- `latte_scheduler_swaps`: Karras-compatible scheduler swaps and custom timesteps/sigmas beyond official DDIM.
- `latte_lora_adapters`: only if a concrete Latte adapter artifact is selected; base pipeline has no Latte-specific loader mixin.
- `latte_text_encoder`: T5 v1.1 XXL encoder compile/load path if Dinoml chooses to own prompt encoding instead of caching embeddings.
- `latte_community_variants`: any img2img/inpaint/control/upscale forks, because no non-deprecated Latte source files implement them here.

Genuinely out of scope for this audit:

- XLA/NPU/MPS/Flax/ONNX paths.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt.
- Safety checker/NSFW filtering.
- Training, dropout behavior, losses, gradient checkpointing.
- Base `maxin-cn/Latte` old training/checkpoint zoo details that are not represented as a current Diffusers pipeline.

## 17. Final implementation checklist

- [ ] Parse `maxin-cn/Latte-1` configs and reject empty/partial `maxin-cn/Latte` as a full pipeline source.
- [ ] Load `LatteTransformer3DModel` weights and accept external T5 positive/negative embeddings.
- [ ] Implement source NCTHW latent contract `[B,4,16,64,64]`.
- [ ] Implement frame PatchEmbed Conv2d + 2D sin-cos position.
- [ ] Implement caption projection and per-frame prompt repeat.
- [ ] Implement AdaLN-single modulation and gated residuals.
- [ ] Implement spatial self-attention, spatial cross-attention, temporal self-attention.
- [ ] Implement temporal position embedding and token rotation.
- [ ] Implement final modulation, projection, and unpatchify to `[B,8,F,H,W]`.
- [ ] Implement negative-first CFG and doubled-channel trim.
- [ ] Implement official DDIM epsilon `clip_sample=false`, `eta=0` first slice.
- [ ] Implement framewise AutoencoderKL decode with scalar latent scale.
- [ ] Add one-step and short-loop parity tests.
- [ ] Add attention/AdaLN/patch/token-rotation fusion probes.
- [ ] Keep temporal decoder VAE, text encoder compilation, scheduler swaps, and adapters as separate candidates.
