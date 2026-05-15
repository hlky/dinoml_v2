# Diffusers Chroma Operator and Integration Report

Candidate slug: `chroma`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  lodestones/Chroma
  lodestones/Chroma1-Base
  lodestones/Chroma1-Flash
  lodestones/Chroma1-HD

Config sources:
  H:/configs/lodestones/Chroma/model_index.json
  H:/configs/lodestones/Chroma/transformer/config.json
  H:/configs/lodestones/Chroma/scheduler/scheduler_config.json
  H:/configs/lodestones/Chroma/vae/config.json
  H:/configs/lodestones/Chroma/text_encoder/config.json
  H:/configs/lodestones/Chroma/tokenizer/tokenizer_config.json
  H:/configs/lodestones/Chroma1-Base/model_index.json
  H:/configs/lodestones/Chroma1-Base/transformer/config.json
  H:/configs/lodestones/Chroma1-Base/scheduler/scheduler_config.json
  H:/configs/lodestones/Chroma1-Base/vae/config.json
  H:/configs/lodestones/Chroma1-Base/text_encoder/config.json
  H:/configs/lodestones/Chroma1-Base/tokenizer/tokenizer_config.json
  H:/configs/lodestones/Chroma1-Flash/model_index.json
  H:/configs/lodestones/Chroma1-Flash/transformer/config.json
  H:/configs/lodestones/Chroma1-Flash/scheduler/scheduler_config.json
  H:/configs/lodestones/Chroma1-Flash/vae/config.json
  H:/configs/lodestones/Chroma1-Flash/text_encoder/config.json
  H:/configs/lodestones/Chroma1-Flash/tokenizer/tokenizer_config.json
  H:/configs/lodestones/Chroma1-HD/model_index.json
  H:/configs/lodestones/Chroma1-HD/transformer/config.json
  H:/configs/lodestones/Chroma1-HD/scheduler/scheduler_config.json
  H:/configs/lodestones/Chroma1-HD/vae/config.json
  H:/configs/lodestones/Chroma1-HD/text_encoder/config.json
  H:/configs/lodestones/Chroma1-HD/tokenizer/tokenizer_config.json

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/chroma/pipeline_chroma.py
  diffusers/src/diffusers/pipelines/chroma/pipeline_chroma_img2img.py
  diffusers/src/diffusers/pipelines/chroma/pipeline_chroma_inpainting.py
  diffusers/src/diffusers/pipelines/chroma/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_chroma.py
  diffusers/src/diffusers/models/transformers/transformer_flux.py
  diffusers/src/diffusers/models/attention.py
  diffusers/src/diffusers/models/attention_processor.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/ip_adapter.py
  diffusers/src/diffusers/loaders/transformer_flux.py

External component configs inspected:
  T5EncoderModel / T5Tokenizer configs from Chroma repos, matching
  google/t5-v1_1-xxl dimensions.

Any missing files or assumptions:
  Local config cache initially had only model_index.json files; component
  configs above were fetched with `hf download` into H:/configs. The report
  focuses on base text-to-image `ChromaPipeline` and inventories img2img,
  inpaint, IP-Adapter, LoRA, textual inversion, and ControlNet hooks as separate
  candidates. Backend/training/safety/callback paths are out of scope except
  where they change inference tensor contracts.
```

## 2. Pipeline and component graph

Chroma is a Flux-like flow-matching latent image pipeline with a Chroma-specific
transformer and a single T5 text encoder. The model index wires
`ChromaPipeline`, `FlowMatchEulerDiscreteScheduler`, `T5EncoderModel`,
`T5Tokenizer`, `ChromaTransformer2DModel`, and `AutoencoderKL`. Optional
`image_encoder` and `feature_extractor` slots are present for IP-Adapter.
Pipeline offload order is `text_encoder->image_encoder->transformer->vae`.

```text
prompt / negative prompt
  -> T5Tokenizer + T5EncoderModel with attention_mask
  -> Chroma prompt mask keeping one padding token unmasked
  -> initialize VAE latent noise [B,16,H/8,W/8]
  -> pack 2x2 latent tiles to tokens [B,(H/16)*(W/16),64]
  -> latent image ids + zero text ids for RoPE
  -> denoising loop:
       ChromaTransformer2DModel(latents, T5 embeds, timestep, ids, mask)
       optional true CFG second transformer call
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens to [B,16,H/8,W/8]
  -> AutoencoderKL decode((latents / scaling_factor) + shift_factor)
  -> VaeImageProcessor postprocess
```

Required first-slice components:

- `ChromaTransformer2DModel` denoiser.
- Single T5 prompt embedding input: `[B,L,4096]` plus prompt attention mask.
- Packed latent preparation/unpacking and latent image/text id generation.
- `FlowMatchEulerDiscreteScheduler`, with Chroma Base beta-sigma variation and
  Chroma/Flash/HD shift variations admitted separately.
- AutoencoderKL decode boundary using 16 latent channels and Flux-style
  scale/shift.

Separate candidate reports:

| Candidate | Classes/files | Delta from base |
| --- | --- | --- |
| `chroma_img2img` | `ChromaImg2ImgPipeline` in `pipeline_chroma_img2img.py` | Adds image preprocessing, VAE encode, strength/timestep slicing, `scheduler.scale_noise`, and packed image latents. |
| `chroma_inpaint` | `ChromaInpaintPipeline` in `pipeline_chroma_inpainting.py` | Adds mask processor, masked-image latents, mask packing, and mask blend after each scheduler step for 64-channel transformer inputs. |
| `chroma_lora_textual_inversion` | `FluxLoraLoaderMixin`, `TextualInversionLoaderMixin`, `FluxTransformer2DLoadersMixin`, `PeftAdapterMixin` | Runtime/load-time text encoder and transformer adapter mutation plus tokenizer prompt conversion. |
| `chroma_ip_adapter` | `FluxIPAdapterMixin`, `FluxIPAdapterAttnProcessor`, optional `CLIPVisionModelWithProjection` | Adds image encoder/projection and extra attention branch through `joint_attention_kwargs["ip_adapter_image_embeds"]`. |
| `chroma_controlnet` | `controlnet_block_samples` and `controlnet_single_block_samples` in `ChromaTransformer2DModel.forward` | Transformer residual injection surface exists, but no Chroma ControlNet pipeline was in the folder. |
| `chroma_single_file` | `FromSingleFileMixin`, `FromOriginalModelMixin` | Single-file loading/conversion surface, not a first runtime op target. |

No Chroma-specific T2I-Adapter, GLIGEN, depth2img, or upscaling pipeline was
present in `pipelines/chroma`; treat those as absent for this family unless a
checkpoint uses an external wrapper.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Transformer | Scheduler | Shift / dynamic | Text encoder | VAE latent | Notes |
| --- | --- | --- | --- | --- | ---: | --- |
| `lodestones/Chroma` | 19 dual + 38 single blocks | FlowMatch Euler | `shift=3.0`, dynamic true | T5 XXL 4096 | 16 | Older dev config, no `guidance_embeds` field consumed by current Chroma transformer. |
| `lodestones/Chroma1-Base` | same | FlowMatch Euler | `use_beta_sigmas=true`; source defaults apply for other fields | T5 XXL 4096 | 16 | Operator-significant beta sigma schedule. |
| `lodestones/Chroma1-Flash` | same | FlowMatch Euler | `shift=1.0`, dynamic false | T5 XXL 4096 | 16 | Best first scheduler parity slice. |
| `lodestones/Chroma1-HD` | same | FlowMatch Euler | `shift=3.0`, dynamic false | T5 XXL 4096 | 16 | Main source example repo. |

Transformer dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| `in_channels` | 64 | transformer config; packed 2x2 latent tokens |
| VAE latent channels | 16 | VAE config; `in_channels // 4` in pipeline |
| `patch_size` | 1 | transformer config |
| `num_layers` | 19 | transformer config |
| `num_single_layers` | 38 | transformer config |
| `num_attention_heads` | 24 | transformer config |
| `attention_head_dim` | 128 | transformer config |
| inner dim | 3072 | inferred as heads * head dim |
| `joint_attention_dim` | 4096 | transformer config / T5 hidden size |
| `axes_dims_rope` | `(16,56,56)` | config/source default |
| approximator | `in=64`, hidden 5120, 5 layers | transformer config |

Text and VAE dimensions:

| Component | Key fields |
| --- | --- |
| T5 text encoder | `d_model=4096`, `d_ff=10240`, 24 layers, 64 heads, gated GELU, vocab 32128, bf16 metadata. |
| T5 tokenizer | `T5Tokenizer`, `model_max_length=512`, pad/eos/unk plus 100 extra ids. |
| AutoencoderKL | 3 input/output channels, 16 latent channels, block channels 128/256/512/512, 2 layers/block, mid attention true, `scaling_factor=0.3611`, `shift_factor=0.1159`, no quant/post-quant convs. |

Scheduler set and first Dinoml slice:

- Source pipeline type is specifically `FlowMatchEulerDiscreteScheduler`.
- Recommended first slice: `Chroma1-Flash` FlowMatch Euler with explicit custom
  sigmas and static shift 1.0.
- Follow with `Chroma1-HD` static shift 3.0, then `Chroma1-Base`
  beta-sigma conversion, then older `lodestones/Chroma` dynamic shifting.

## 3a. Family variation traps

- Chroma has no CLIP text encoder or pooled CLIP embedding in the base pipeline;
  it uses only T5 token embeddings and Chroma-specific attention masks.
- `in_channels=64` is packed token width, not VAE latent channels.
- The transformer ignores `guidance_embeds`; true CFG is a second positive vs
  negative transformer call in the pipeline.
- Chroma differs from Flux by using a prompt attention mask in T5 and transformer
  attention. Flash-style attention must support this dense mask or fall back.
- The Chroma transformer builds per-block modulation tensors from a distilled
  guidance approximator, not the Flux combined timestep/text pooled projection.
- Source latents are NCHW at VAE encode/decode boundaries and token-major inside
  the transformer. Treat NHWC only as a guarded VAE/local packing optimization.
- Img2img and inpaint share the base transformer but add VAE encode,
  `scale_noise`, masks, masked latents, and strength slicing.
- ControlNet residual hooks exist in the transformer forward even when inactive
  in base Chroma; do not include them in first-slice base ops.
- IP-Adapter kwargs mutate the attention path by adding an extra image attention
  branch; keep it separate.

## 4. Runtime tensor contract

For 1024x1024, batch `B`, one image per prompt, max T5 length `L=512`:

| Boundary | Tensor | Shape / layout | Notes |
| --- | --- | --- | --- |
| token ids | `text_input_ids` | `[B,L]` CPU/GPU int | T5Tokenizer output. |
| T5 mask | `tokenizer_mask` | `[B,L]` | Passed into T5 encoder. |
| prompt embeds | `prompt_embeds` | `[B,L,4096]` | External first-slice input can bypass T5. |
| Chroma prompt mask | `prompt_attention_mask` | `[B,L]` | `mask_indices <= seq_lengths`, so one padding token remains unmasked. |
| text ids | `text_ids` | `[L,3]` | All zeros; concatenated with image ids for RoPE. |
| latent noise map | source VAE latent | `[B,16,128,128]`, NCHW | Generated before packing. |
| packed latents | transformer input | `[B,4096,64]` | 2x2 NCHW tile packing. |
| latent image ids | `latent_image_ids` | `[4096,3]` | columns encode zero, row, column. |
| full attention mask | `attention_mask` | `[B,L+4096]`, then `[B,1,L+S,L+S]` | Dense pair mask inside Chroma blocks. |
| timestep | `timestep` | `[B]` | Pipeline passes `t / 1000`; transformer multiplies by 1000. |
| transformer output | `noise_pred` | `[B,4096,64]` | Same packed token shape. |
| scheduler state | `sigmas`, `timesteps`, step index | host/runtime-visible | FlowMatch Euler appends terminal sigma 0 unless inverted. |
| unpacked latents | decode input map | `[B,16,128,128]`, NCHW | 2x2 unpack. |
| VAE decode input | shifted/unscaled | `[B,16,128,128]` | `(latents / 0.3611) + 0.1159`. |
| output image tensor | before postprocess | `[B,3,1024,1024]` | AutoencoderKL decode, then image processor. |

CPU/data-pipeline work includes tokenization, prompt truncation, PIL/array
pre/postprocess, and optional image/mask crop preprocessing. GPU/runtime work
includes T5 if not cached, latent packing, transformer denoising, scheduler
pointwise update, VAE decode, and optional VAE encode for variants.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent random tensor generation.
- 2x2 packing: `view(B,C,H/2,2,W/2,2) -> permute(0,2,4,1,3,5) -> reshape(B,H/2*W/2,4C)`.
- 2x2 unpacking inverse to NCHW.
- Text/image id concat for 3-axis RoPE.
- Text and image token concat/split in transformer blocks.
- Dense attention mask expansion from `[B,L+S]` to pairwise `[B,1,L+S,L+S]`.
- True CFG arithmetic: `neg + scale * (pos - neg)`.

GEMM/linear ops:

- `x_embedder`: Linear(64 -> 3072).
- `context_embedder`: Linear(4096 -> 3072).
- Distilled modulation approximator: Linear(64 -> 5120), 5 residual
  `PixArtAlphaTextProjection` layers with RMSNorm, Linear(5120 -> 3072).
- Flux attention Q/K/V and added Q/K/V projections, all bias-enabled.
- MLP projections in single blocks: Linear(3072 -> 12288), GELU tanh,
  Linear(15360 -> 3072).
- Dual-block feed-forward GEGLU/GELU approximate `FeedForward`.
- Final Linear(3072 -> 64).

Attention primitives:

- Dual-stream joint text/image attention in 19 `ChromaTransformerBlock`s.
- Single-stream self-attention over concatenated text+image tokens in 38
  `ChromaSingleTransformerBlock`s.
- QK RMSNorm, RoPE, noncausal attention, dense mask support.
- Optional IP-Adapter added attention branch as separate candidate.

Normalization and adaptive conditioning:

- LayerNorm eps 1e-6 without affine for adaptive norm paths.
- RMSNorm in attention Q/K and approximator.
- Chroma pruned AdaLayerNorm zero variants producing gates, shifts, scales.
- Final Chroma adaptive continuous norm with two modulation tensors.

Scheduler/VAE ops:

- FlowMatch Euler set_timesteps with custom sigmas, static/dynamic shift, beta
  sigma conversion for Base.
- Step update `sample + (sigma_next - sigma) * model_output` for deterministic
  branches.
- AutoencoderKL decode Conv2d/ResNet/GroupNorm/SiLU/up/downsample/mid attention.

## 6. Denoiser/model breakdown

Top-level `ChromaTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> x_embedder -> [B,S_img,3072]
encoder_hidden_states [B,L,4096] -> context_embedder -> [B,L,3072]
timestep * 1000 -> ChromaCombinedTimestepTextProjEmbeddings -> [B,N_mod,64]
modulation sequence -> ChromaApproximator -> [B,N_mod,3072]
ids = cat(txt_ids, img_ids) -> FluxPosEmbed RoPE
19 dual-stream ChromaTransformerBlock
concat text+image
38 ChromaSingleTransformerBlock
drop text tokens
final adaptive norm using last 2 modulation entries
proj_out -> [B,S_img,64]
```

Dual-stream block:

```text
image modulation slice: LayerNorm -> scale/shift -> QKV
text modulation slice:  LayerNorm -> scale/shift -> added QKV
QK RMSNorm on both streams
concat text+image Q/K/V -> RoPE -> masked joint attention
split outputs; gated residual attention
LayerNorm -> adaptive MLP scale/shift -> FeedForward -> gated residual
repeat for text stream with separate FFN and gates
```

Single-stream block:

```text
hidden_states = cat(text, image)
LayerNorm -> adaptive scale/shift -> gate
parallel attention branch and GELU MLP branch
concat(attn_output, mlp_hidden) -> Linear -> gate -> residual
```

Inactive branches to guard:

- `controlnet_block_samples` and `controlnet_single_block_samples` add residuals
  at block intervals.
- `joint_attention_kwargs["ip_adapter_image_embeds"]` introduces image adapter
  projections and extra attention output.
- Gradient checkpointing and training/dropout paths are not inference first
  slice.

## 7. Attention requirements

Chroma reuses `FluxAttention` and `FluxAttnProcessor` from
`transformer_flux.py`. The processor obtains Q/K/V projections, unflattens to
`[B,seq,heads,head_dim]`, applies RMSNorm to Q and K, optionally concatenates
added text Q/K/V, applies RoPE, then calls `dispatch_attention_fn`.

Required base attention traits:

- Noncausal self/joint attention.
- Heads 24, head dim 128.
- Query/key/value dtype and device must match.
- QK RMSNorm before attention.
- 3-axis Flux RoPE over concatenated text/image ids.
- Dense prompt-derived attention mask. Chroma blocks convert a sequence mask to
  pairwise mask by outer product.
- Eager/native dispatch is the parity path.

Flash/provider constraints:

- Unlike base Flux, Chroma normally has an attention mask, so a Dinoml
  flash-style provider needs dense mask support or an admission guard proving
  the mask is all true.
- QK RMSNorm and RoPE should remain explicit pre-attention ops unless fused with
  matching parity tests.
- IP-Adapter adds a separate image K/V branch through `FluxIPAdapterAttnProcessor`;
  base first slice should reject or route to fallback when `ip_hidden_states` is
  present.
- Diffusers `attention_dispatch.py` can route to native SDPA, flash, varlen
  flash, flex, sage, xFormers, or hub kernels depending on package and
  constraints. Dinoml should model backend admission explicitly rather than
  assuming flash availability.

## 8. Scheduler and denoising-loop contract

Base Chroma call:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = packed_latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
for t in timesteps:
  noise_pred = transformer(latents, t / 1000, prompt_embeds, ids, mask)
  if guidance_scale > 1:
    neg = transformer(latents, t / 1000, negative_embeds, ids, negative_mask)
    noise_pred = neg + guidance_scale * (noise_pred - neg)
  latents = scheduler.step(noise_pred, t, latents)
```

Scheduler source math:

- `set_timesteps` accepts `num_inference_steps`, `sigmas`, `timesteps`, and
  optional `mu`.
- Dynamic shifting requires `mu`; static shifting applies
  `shift * sigma / (1 + (shift - 1) * sigma)`.
- `use_beta_sigmas` converts the sigma table after shifting.
- Deterministic step upcasts sample to fp32 and computes
  `prev_sample = sample + (sigma_next - sigma) * model_output`, then casts back
  to model output dtype.

Host/runtime split:

- Keep schedule generation, beta conversion, dynamic `mu`, step index, and
  img2img/inpaint timestep slicing host-visible first.
- Compile transformer forward, CFG arithmetic, and deterministic scheduler step
  after schedule tables are explicit.

## 9. Position, timestep, and custom math

Custom math to reproduce:

- Chroma prompt mask: `attention_mask = (arange(L) <= seq_lengths[:,None])`,
  intentionally preserving one padding token.
- 2x2 latent packing/unpacking exactly as in Flux.
- Timestep scaling: pipeline sends `t / 1000`, transformer multiplies by 1000
  before sinusoidal embedding.
- `ChromaCombinedTimestepTextProjEmbeddings` creates timestep projection,
  zero-guidance projection, and a persistent sinusoidal `mod_proj` indexed over
  modulation slots.
- `ChromaApproximator` maps those 64-wide modulation tokens to 3072-wide block
  modulation tensors through residual RMSNorm + PixArt projection layers.
- `FluxPosEmbed` computes 3-axis RoPE from zero text ids and row/column image
  ids.

Precompute candidates:

- `text_ids`, latent image ids for a fixed resolution, and many RoPE tables.
- Prompt embeddings and Chroma prompt masks for repeated prompts.
- Scheduler sigma/timestep tables per checkpoint, resolution, step count, and
  custom sigma list.

## 10. Preprocessing and input packing

Text:

- `T5Tokenizer` pads/truncates to `max_sequence_length <= 512`.
- T5 receives `attention_mask`; this differs from Flux.
- Embeddings and Chroma mask are repeated for `num_images_per_prompt`.
- Negative prompt embeddings and masks are generated only when
  `guidance_scale > 1`.

Latents:

- Default image size is `default_sample_size=128` times VAE scale factor 8,
  yielding 1024.
- Height/width should be divisible by `vae_scale_factor * 2 = 16`; the pipeline
  warns and image processor resizing handles preprocessing boundaries.
- Base text-to-image generates NCHW latent noise and packs to tokens.

Variants:

- Img2img encodes input image through VAE, scales with
  `(latents - shift_factor) * scaling_factor`, then `scheduler.scale_noise`.
- Inpaint preprocesses mask and image, encodes masked image if needed, repeats
  masks to batch, packs mask repeated across latent channels to 64-wide tokens,
  and blends latent updates with init latents after each step.
- IP-Adapter preprocesses images through optional CLIP image processor and
  vision model, then threads image embeds through attention kwargs.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Chroma latent pack/unpack

Source pattern:

```text
NCHW [B,16,H,W] -> view(B,16,H/2,2,W/2,2)
  -> permute(0,2,4,1,3,5) -> [B,(H/2)*(W/2),64]
```

Replacement: a layout-aware pack/unpack primitive.

Preconditions:

- Source map is NCHW.
- H and W are divisible by 2 after VAE scaling.
- Channel count is 16 and token width is 64 for sampled configs.

Failure cases:

- NHWC layout translation without matching flatten-order rewrite.
- Inpaint mask tokens, where mask is repeated across latent channels before the
  same pack.

### Rewrite: Chroma dense-mask attention canonicalization

Source pattern:

```text
Linear QKV/add-QKV -> unflatten heads -> QK RMSNorm
  -> concat text/image -> RoPE -> dense masked attention -> split
```

Replacement: canonical joint-attention op with explicit mask, RoPE, and stream
split metadata.

Preconditions:

- No IP-Adapter branch.
- Mask shape and dtype are admitted by backend.
- Text/image sequence lengths are known to split outputs.

Failure cases:

- Flash backend lacks dense mask support.
- IP-Adapter or ControlNet residual side inputs are active.

### Rewrite: Chroma modulation approximator

Source pattern:

```text
timestep/mod sinusoidal projections -> Linear -> 5 residual PixArt projections
  with RMSNorm -> Linear -> per-block modulation slices
```

Replacement: keep as explicit MLP first; later fuse RMSNorm + SiLU projection
inside approximator layers.

Preconditions: fixed modulation slot count from block counts.

Parity test: compare full `pooled_temb` slices for random timesteps across fp32
and bf16.

### Rewrite: FlowMatch Euler deterministic step

Source pattern: `latents = latents + (sigma_next - sigma) * noise_pred`.

Preconditions:

- `stochastic_sampling=false`.
- No per-token timestep branch.
- Explicit scheduler step index and sigma table.

Failure cases: Chroma Base beta-sigma table not generated exactly; dynamic
shift omitted for older `lodestones/Chroma`.

## 12. Kernel fusion candidates

Highest priority:

- Dense masked joint attention with QK RMSNorm + RoPE prelude.
- Chroma adaptive LayerNorm gates plus residual add in dual and single blocks.
- GEGLU/GELU approximate feed-forward projections.
- FlowMatch Euler deterministic scheduler update.
- 2x2 latent pack/unpack.

Medium priority:

- ChromaApproximator RMSNorm + PixArt projection residual layers.
- CFG arithmetic for two-call guidance.
- Final adaptive norm + projection.
- VAE decode conv/resnet/groupnorm/silu island, shared with Flux.

Lower priority:

- IP-Adapter added image attention.
- ControlNet residual injection.
- Img2img/inpaint mask blend kernels.
- LoRA fusion/unfusion and single-file conversion.

## 13. Runtime staging plan

Stage 1: Parse Chroma model index and component configs from
`H:/configs/lodestones/Chroma1-Flash`; load transformer and VAE weights. Accept
external T5 embeddings and masks.

Stage 2: Implement Chroma prompt mask contract, latent pack/unpack, text ids,
and latent image ids.

Stage 3: Compile one `ChromaTransformerBlock` and one
`ChromaSingleTransformerBlock` with random tensors and dense masks.

Stage 4: Compile full `ChromaTransformer2DModel` for `Chroma1-Flash` at fixed
1024 shape, no CFG.

Stage 5: Add FlowMatch Euler static shift 1.0 scheduler and one denoising-step
parity.

Stage 6: Add true CFG two-call path and VAE decode.

Stage 7: Add HD shift 3.0 and Base beta-sigma schedule parity.

Stage 8: Add optional T5 execution/cache integration.

Stage 9: Separate reports/implementations for img2img, inpaint, IP-Adapter,
ControlNet residual hooks, LoRA/textual inversion.

## 14. Parity and validation plan

- Prompt mask parity for varied prompt lengths, confirming one padding token is
  preserved.
- T5 embedding cache contract: supplied `prompt_embeds` must require supplied
  `prompt_attention_mask`.
- Pack/unpack parity for `[B,16,128,128] <-> [B,4096,64]`.
- Latent image ids and RoPE table parity by resolution.
- Chroma timestep/modulation approximator parity.
- One dual-stream block parity with dense masks.
- One single-stream block parity over concatenated text+image tokens.
- Full transformer forward parity for `Chroma1-Flash`.
- FlowMatch scheduler table/step parity for Flash, HD, Base beta sigmas, and
  dynamic shifting.
- One denoising-step parity with fixed embeddings, mask, latents, and timestep.
- VAE decode scale/shift parity.
- Img2img/inpaint variant parity only after base.
- Suggested tolerances: fp32 scheduler `rtol=1e-5, atol=1e-6`; fp32 model
  blocks `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- T5 encoder throughput and prompt cache hit/miss overhead.
- Full transformer step at sequence lengths 1024, 4096, and larger if admitted.
- Attention backend comparison with and without dense masks.
- Dual-stream vs single-stream block timing.
- ChromaApproximator timing and memory.
- CFG one-call vs two-call total latency.
- FlowMatch scheduler overhead and beta-sigma table generation.
- VAE decode throughput at 1024 and larger.
- Img2img VAE encode and inpaint mask blend overhead in separate variant probes.
- VRAM/workspace by dtype and sequence length.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `chroma_img2img`: VAE encode, strength slicing, `scale_noise`, image latent
  packing.
- `chroma_inpaint`: mask preprocessing, mask/latent packing, per-step mask
  blend.
- `chroma_lora_textual_inversion`: tokenizer prompt conversion and PEFT/LoRA
  mutation for T5 and transformer.
- `chroma_ip_adapter`: CLIP image encoder/projection and added attention branch.
- `chroma_controlnet`: transformer residual side inputs if a Chroma ControlNet
  pipeline/checkpoint appears.
- `flowmatch_beta_sigmas`: Base config uses beta sigma conversion and deserves
  scheduler validation separate from Flash static shift.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.
- Backend/training/callback paths outside the user-requested Chroma inference
  source surface.

## 17. Final implementation checklist

- [ ] Parse Chroma model index and component configs.
- [ ] Load `ChromaTransformer2DModel` and AutoencoderKL weights.
- [ ] Accept external T5 prompt embeddings and Chroma prompt masks.
- [ ] Implement Chroma prompt mask generation.
- [ ] Implement latent pack/unpack and image/text id generation.
- [ ] Implement Chroma timestep/modulation approximator.
- [ ] Implement `ChromaTransformerBlock` dual-stream parity.
- [ ] Implement `ChromaSingleTransformerBlock` parity.
- [ ] Implement QK RMSNorm + RoPE + dense masked joint attention.
- [ ] Implement FlowMatch Euler static shift 1.0 scheduler slice.
- [ ] Add HD shift 3.0 and Base beta-sigma schedule validation.
- [ ] Add true CFG two-call arithmetic.
- [ ] Add one-step denoising parity.
- [ ] Add VAE scale/shift decode boundary.
- [ ] Add guarded flash/provider lowering for mask-compatible attention.
- [ ] Create separate candidate reports for img2img, inpaint, IP-Adapter, LoRA,
  and ControlNet residual hooks.
