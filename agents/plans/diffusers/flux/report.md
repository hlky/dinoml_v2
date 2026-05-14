# Diffusers Flux Operator and Integration Report

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  black-forest-labs/FLUX.1-schnell
  black-forest-labs/FLUX.1-dev

Config sources:
  H:/configs/black-forest-labs/FLUX.1-schnell/model_index.json
  H:/configs/black-forest-labs/FLUX.1-schnell/transformer/config.json
  H:/configs/black-forest-labs/FLUX.1-schnell/scheduler/scheduler_config.json
  H:/configs/black-forest-labs/FLUX.1-schnell/vae/config.json
  H:/configs/black-forest-labs/FLUX.1-schnell/text_encoder/config.json
  H:/configs/black-forest-labs/FLUX.1-schnell/text_encoder_2/config.json
  H:/configs/black-forest-labs/FLUX.1-dev/model_index.json
  H:/configs/black-forest-labs/FLUX.1-dev/transformer/config.json
  H:/configs/black-forest-labs/FLUX.1-dev/scheduler/scheduler_config.json
  Component configs were saved with authenticated `hf download`.

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_img2img.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_inpaint.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_fill.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_control.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_controlnet.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
  X:/H/diffusers/src/diffusers/pipelines/flux/pipeline_flux_prior_redux.py
  X:/H/diffusers/src/diffusers/pipelines/flux/modeling_flux.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_flux.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py
  X:/H/diffusers/src/diffusers/loaders/ip_adapter.py
  X:/H/diffusers/src/diffusers/loaders/transformer_flux.py

External component configs inspected:
  CLIPTextModel / CLIPTokenizer for `openai/clip-vit-large-patch14`.
  T5EncoderModel / T5TokenizerFast for `google/t5-v1_1-xxl`.

Any missing files or assumptions:
  This report focuses on the main text-to-image `FluxPipeline` and
  `FluxTransformer2DModel`. Flux ControlNet, control/fill/inpaint/img2img,
  Kontext, prior redux, IP-Adapter, LoRA, and Flux2/Klein are documented as
  separate candidates. Multi-GPU/context parallel, callback mutation/interrupt,
  XLA/NPU/MPS/Flax/ONNX variants, safety/NSFW, and training/dropout/gradient
  checkpointing are ignored unless explicitly selected.
```

## 2. Pipeline and component graph

Flux is a flow-matching latent image pipeline with a transformer denoiser. The
main model index wires `FluxPipeline`, `FlowMatchEulerDiscreteScheduler`,
`CLIPTextModel`, `T5EncoderModel`, `CLIPTokenizer`, `T5TokenizerFast`,
`FluxTransformer2DModel`, and `AutoencoderKL`. The pipeline offload sequence is
`text_encoder->text_encoder_2->image_encoder->transformer->vae`, with
`image_encoder` relevant to IP-Adapter paths.

```text
prompt / prompt_2
  -> CLIPTokenizer + CLIPTextModel pooled output [B,768]
  -> T5TokenizerFast + T5EncoderModel token embeddings [B,L,4096]
  -> initialize VAE latent noise [B,16,H/8,W/8]
  -> pack 2x2 latent tiles to tokens [B,(H/16)*(W/16),64]
  -> create latent image ids and text ids for RoPE
  -> denoising loop:
       FluxTransformer2DModel(hidden_states, T5 embeds, CLIP pooled,
                              timestep, optional guidance, ids)
       optional true CFG second transformer call
       FlowMatchEulerDiscreteScheduler.step
  -> unpack latent tokens back to [B,16,H/8,W/8]
  -> AutoencoderKL decode((latents / scaling_factor) + shift_factor)
  -> VaeImageProcessor postprocess
```

Required first-slice components:

- `FluxTransformer2DModel` denoiser.
- Packed latent preparation/unpacking and latent image/text id generation.
- `FlowMatchEulerDiscreteScheduler`.
- Prompt embeddings as external inputs first: T5 token embeddings and CLIP
  pooled embeddings.
- `AutoencoderKL` decode boundary with 16-channel shift/scale contract.

Separate candidate reports for the Flux family:

| Candidate | Primary classes/files | Pipeline delta |
| --- | --- | --- |
| `flux_img2img` | `FluxImg2ImgPipeline` | Adds VAE encode, image latent packing, strength/timestep slicing, and initial latent noising. |
| `flux_inpaint_fill` | `FluxInpaintPipeline`, `FluxFillPipeline` | Adds masks, masked/image latents, fill conditioning, and different latent packing inputs. |
| `flux_control` | `FluxControlPipeline`, `pipeline_flux_control_img2img.py`, `pipeline_flux_control_inpaint.py` | Adds explicit control image latents without the full ControlNet model path. |
| `flux_controlnet` | `FluxControlNetPipeline`, `FluxControlNetImg2ImgPipeline`, `FluxControlNetInpaintPipeline`, `FluxControlNetModel` | Adds transformer-side control residuals and control guidance start/end schedules. |
| `flux_ip_adapter` | `FluxIPAdapterMixin`, `FluxIPAdapterAttnProcessor`, `MultiIPAdapterImageProjection` | Adds image encoder/projection and extra IP attention output branches. |
| `flux_lora_adapters` | `FluxLoraLoaderMixin`, `PeftAdapterMixin` on `FluxTransformer2DModel` | Runtime/load-time adapter mutation for transformer and text encoders. |
| `flux_kontext` | `FluxKontextPipeline`, `FluxKontextInpaintPipeline` | Kontext image editing/conditioning variants in the same folder. |
| `flux_prior_redux` | `FluxPriorReduxPipeline`, `ReduxImageEncoder`, `SiglipImageProcessor` | Prior/image encoder path that produces conditioning for Flux. |
| `flux2_klein` | `pipelines/flux2/*Klein*`, `Flux2LoraLoaderMixin` | Distinct Flux2/Klein family in a separate folder; inspect separately before assuming Flux1 parity. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | transformer | guidance embeds | scheduler shift | dynamic shifting | VAE latent | VAE scale/shift | Text encoders |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| FLUX.1-schnell | `FluxTransformer2DModel` | false | 1.0 | false | 16 | 0.3611 / 0.1159 | CLIP-L pooled 768 + T5-XXL 4096 |
| FLUX.1-dev | `FluxTransformer2DModel` | true | 3.0 | true | 16 | same expected VAE family | CLIP-L pooled 768 + T5-XXL 4096 |

Transformer config:

| Field | Value | Source |
| --- | ---: | --- |
| `in_channels` | 64 | transformer config |
| `patch_size` | 1 | transformer config |
| `num_layers` | 19 | transformer config |
| `num_single_layers` | 38 | transformer config |
| `num_attention_heads` | 24 | transformer config |
| `attention_head_dim` | 128 | transformer config |
| inner dim | 3072 | inferred as heads * head dim |
| `joint_attention_dim` | 4096 | transformer config / T5 hidden |
| `pooled_projection_dim` | 768 | transformer config / CLIP pooled |
| `axes_dims_rope` | source default `(16,56,56)` | source default, omitted in sampled configs |

Text/image codec config:

| Component | Key dimensions |
| --- | --- |
| CLIP text encoder | hidden 768, max positions 77, 12 layers, 12 heads, vocab 49408 |
| T5 text encoder | `d_model=4096`, 24 layers, 64 heads, vocab 32128, gated GELU FFN |
| AutoencoderKL | 16 latent channels, blocks 128/256/512/512, `scaling_factor=0.3611`, `shift_factor=0.1159`, no quant/post-quant conv |

Scheduler support:

- Flux main pipeline is FlowMatch-shaped. Do not assume SD-style DDIM/Euler/DPM
  scheduler swaps for Flux.
- Source default scheduler class is `FlowMatchEulerDiscreteScheduler`.
- Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigma
  support, static `schnell` shift first, then `dev` dynamic shifting.

## 3a. Family variation traps

- Flux latent tokens are packed 2x2 VAE latent tiles. With 1024x1024 images,
  VAE latents are `[B,16,128,128]`, packed tokens are `[B,4096,64]`.
- Transformer `in_channels=64` is not the VAE latent channel count; it is
  `16 * 2 * 2` after packing.
- `schnell` and `dev` differ materially: `schnell` has no embedded guidance and
  static scheduler shifting; `dev` has `guidance_embeds=true` and dynamic
  shifting.
- Flux supports embedded guidance and true CFG as separate concepts. Embedded
  guidance is a conditioning vector to the model; true CFG runs an additional
  negative transformer call.
- Source includes IP-Adapter and ControlNet branches in the main transformer and
  pipeline folder. These are separate candidates, not first-slice base ops.
- Attention uses joint text/image attention with QK RMSNorm and RoPE. This is
  not UNet cross-attention.
- Pipeline callbacks and `_interrupt` can mutate tensors but are ignored for
  this audit.
- Context-parallel plan exists in `FluxTransformer2DModel`; ignored for current
  single-device audit.

## 4. Runtime tensor contract

For 1024x1024, one image per prompt:

| Boundary | Tensor | Shape | Notes |
| --- | --- | --- | --- |
| T5 prompt embeds | `prompt_embeds` | `[B,L,4096]`, often `L=512` max | External first-slice input. |
| CLIP pooled embeds | `pooled_prompt_embeds` | `[B,768]` | Time/text conditioning input. |
| text ids | `text_ids` | `[L,3]` | Concatenated with image ids for RoPE. |
| VAE latent noise | before pack | `[B,16,128,128]` | Source layout NCHW. |
| packed latent tokens | `latents` / transformer input | `[B,4096,64]` | 2x2 packing of 16-channel latent map. |
| latent image ids | `latent_image_ids` | `[4096,3]` | Prepared from packed latent grid. |
| timestep | per step | `[B]` | Pipeline passes `timestep / 1000`; model multiplies by 1000. |
| embedded guidance | `guidance` | `[B]` or `None` | Only when `transformer.config.guidance_embeds`. |
| transformer output | `noise_pred` | `[B,4096,64]` | Same packed token shape. |
| scheduler output | `latents` | `[B,4096,64]` | Flow Euler update. |
| unpacked latents | decode input map | `[B,16,128,128]` | After `_unpack_latents`. |
| VAE decode input | shifted/unscaled | `[B,16,128,128]` | `(latents / scale) + shift`. |
| pipeline output | image | `[B,3,1024,1024]` before postprocess | From AutoencoderKL decode. |

Source layout is NCHW only at the VAE boundary. Main Flux transformer runtime is
token-major `[B,sequence,channels]`; NHWC optimization applies to VAE and maybe
packing/unpacking boundaries, not to the transformer token matmul core.

## 5. Operator coverage checklist

### Tensor/layout ops

- VAE NCHW latent noise generation.
- 2x2 latent packing: view, permute, reshape.
- Latent unpacking: view, permute, reshape.
- Text/image id concat for RoPE.
- Token concat/split between text and image streams in single-stream blocks.
- Optional true CFG second call and elementwise `neg + scale * (pos - neg)`.

### GEMM/linear ops

- `x_embedder`: Linear(64 -> 3072).
- `context_embedder`: Linear(4096 -> 3072).
- Timestep/guidance/pooled-text embedding MLPs.
- Flux attention Q/K/V/add-Q/add-K/add-V projections.
- Feed-forward GEGLU/GELU approximate MLPs.
- Final `proj_out`: Linear(3072 -> 64).

### Attention primitives

- Joint text/image attention in 19 dual-stream `FluxTransformerBlock`s.
- Joint single-stream attention over concatenated text+image sequence in 38
  `FluxSingleTransformerBlock`s.
- QK RMSNorm on image and text query/key streams.
- RoPE applied to concatenated text/image query/key.
- Optional IP-Adapter added attention branch in separate candidate.

### Normalization and adaptive conditioning

- `AdaLayerNormZero` for dual-stream image and text paths.
- `AdaLayerNormZeroSingle` for single-stream blocks.
- `AdaLayerNormContinuous` before final projection.
- LayerNorm and RMSNorm.
- Gated residual attention/MLP paths.

### Position/timestep/guidance embeddings

- `FluxPosEmbed` 3-axis RoPE over text/image ids.
- `CombinedTimestepTextProjEmbeddings` for schnell.
- `CombinedTimestepGuidanceTextProjEmbeddings` for dev guidance-distilled
  model.

### Scheduler and guidance arithmetic

- FlowMatch Euler sigma schedule, optional custom sigmas.
- Dynamic resolution-dependent shifting for dev.
- `prev_sample = sample + (sigma_next - sigma) * model_output`.
- Embedded guidance vector when enabled.
- True CFG optional two-call branch.

### VAE/postprocessing ops

- AutoencoderKL decode with 16 latent channels.
- Flux scale/shift: `(latents / 0.3611) + 0.1159` before decode.
- Image postprocess to requested output type.

## 6. Denoiser/model breakdown

Top-level `FluxTransformer2DModel.forward`:

```text
hidden_states [B,S_img,64] -> Linear 64->3072
encoder_hidden_states [B,S_txt,4096] -> Linear 4096->3072
timestep (+ optional guidance) + pooled CLIP -> conditioning emb [B,3072]
ids = cat(txt_ids, img_ids) -> FluxPosEmbed RoPE
19 FluxTransformerBlock dual-stream blocks
38 FluxSingleTransformerBlock single-stream blocks
AdaLayerNormContinuous -> Linear 3072->64
```

Dual-stream block:

```text
image: AdaLayerNormZero -> QKV
text:  AdaLayerNormZero -> added QKV
QK RMSNorm on both streams
concat text+image Q/K/V -> RoPE -> attention
split attention output back to text/image
gated residual attention
LayerNorm -> adaptive scale/shift -> FF -> gated residual
```

Single-stream block:

```text
cat(text, image)
AdaLayerNormZeroSingle -> gate
parallel attention input and MLP input
FluxAttention(pre_only=True) + GELU MLP
concat(attn, mlp) -> Linear -> gate -> residual
split text/image
```

## 7. Attention requirements

Flux attention path is `FluxAttnProcessor` in `transformer_flux.py`, which calls
`dispatch_attention_fn` from `attention_dispatch.py`. Important traits:

- Noncausal joint attention.
- Query/key/value shape after unflatten is `[B,seq,heads,head_dim]`.
- Heads 24, head dim 128, inner dim 3072.
- QK RMSNorm before attention.
- RoPE applied after optional text/image concatenation.
- Base path has no attention mask in the main pipeline.
- Text and image streams are concatenated for attention and split after output.
- `FluxIPAdapterAttnProcessor` adds an extra IP image-attention branch and is a
  separate candidate.

Flash/native/xFormers/flex/sage notes:

- `attention_dispatch.py` exposes native SDPA, flash-attn 2/3/hub, flash
  varlen, flex, sage, and xFormers backends, subject to package/version and mask
  constraints.
- Several flash/sage paths reject arbitrary masks; base Flux main path is
  mask-free, so Dinoml flash-style attention is plausible.
- QK RMSNorm and RoPE must be represented before the attention provider call.
- Joint text+image sequence lengths are large: for 1024x1024 with T5 length 512,
  attention length is about 4608 tokens.
- Eager/native dispatch is the parity path; flash-style lowering should be a
  guarded provider choice, not assumed unavailable or automatically valid.

## 8. Scheduler and denoising-loop contract

Flux uses `FlowMatchEulerDiscreteScheduler`, not SD-style DDIM/PNDM/Euler.

Pipeline setup:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = packed_latents.shape[1]
mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len,
                     base_shift, max_shift)
scheduler.set_timesteps(..., sigmas=sigmas, mu=mu)
```

Scheduler step:

```text
dt = sigma_next - sigma
prev_sample = sample + dt * model_output
```

`FLUX.1-schnell` sampled config:

- `shift=1.0`, `use_dynamic_shifting=false`.
- `guidance_embeds=false`; `guidance_scale` does not create a model guidance
  embedding.

`FLUX.1-dev` sampled config:

- `shift=3.0`, `use_dynamic_shifting=true`.
- `guidance_embeds=true`; pipeline passes a guidance vector to the transformer.

Host/runtime split:

- Keep loop iteration, custom sigma list, and dynamic shift host-visible first.
- Compile one transformer step plus scheduler pointwise update first.
- Add true CFG as a runtime option that may invoke a second transformer call.

## 9. Position, timestep, and custom math

Custom math that matters:

- 2x2 latent pack/unpack.
- `calculate_shift` for dynamic scheduler shift, based on image sequence length.
- Timestep is divided by 1000 in the pipeline and multiplied by 1000 in the
  transformer before embedding.
- Guidance embeddings multiply guidance by 1000 before embedding.
- `FluxPosEmbed` computes 3-axis RoPE over concatenated text/image ids.
- Final VAE decode applies scale/shift, not only scale.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- CLIP/T5 tokenization and truncation handling.
- Text encoder execution if not supplied as cached embeddings.
- Image/PIL output conversion.

GPU/runtime work:

- Accept external `prompt_embeds [B,L,4096]` and
  `pooled_prompt_embeds [B,768]`.
- Generate or accept packed latent tokens.
- Generate latent image ids and text ids.
- Denoise packed latent tokens.
- Unpack tokens for VAE decode.

Text encoder details:

- `tokenizer` / `text_encoder`: `CLIPTokenizer`, `CLIPTextModel`; pooled output
  feeds `pooled_prompt_embeds`.
- `tokenizer_2` / `text_encoder_2`: `T5TokenizerFast`, `T5EncoderModel`; token
  embeddings feed `encoder_hidden_states`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: latent pack/unpack as layout transform

Preconditions:

- VAE latent map is NCHW `[B,16,H,W]` with even H/W.
- Transformer expects `[B,(H/2)*(W/2),64]`.

Replacement:

```text
NCHW 2x2 tile pack -> token matrix
token matrix -> NCHW 2x2 tile unpack
```

Failure cases:

- H/W not divisible by 2 after VAE scale.
- NHWC VAE layout pass changes flatten order without matching pack rewrite.

### Rewrite: joint attention canonicalization

Preconditions:

- No IP-Adapter branch, no mask, fixed heads/head_dim.
- QK RMSNorm and RoPE are explicit pre-attention ops.
- Text/image concat and split sizes are plan-visible.

Replacement:

```text
linear QKV/add-QKV -> QK RMSNorm -> concat -> RoPE -> attention -> split -> output projections
```

Failure cases:

- IP-Adapter adds separate image-attention output.
- Backend cannot handle sequence length/head dim/dtype.

### Rewrite: FlowMatch Euler step

Preconditions:

- Scheduler sigma index/state is explicit.
- `stochastic_sampling=false` and no per-token timesteps.

Replacement:

```text
latents + (sigma_next - sigma) * noise_pred
```

Failure cases:

- Per-token timesteps, stochastic sampling, or dynamic scheduler state not
  represented.

## 12. Kernel fusion candidates

Highest priority:

- QKV/add-QKV projection + QK RMSNorm + RoPE + attention provider.
- AdaLayerNormZero/Single gating + residual add.
- GELU approximate feed-forward MLP.
- FlowMatch Euler pointwise scheduler update.
- Latent pack/unpack kernels with VAE layout compatibility.

Medium priority:

- Embedded guidance/timestep/pooled projection MLP fusion.
- Final AdaLayerNormContinuous + projection.
- True CFG arithmetic and possible two-call scheduling.
- VAE 16-channel decode conv island, already covered by AutoencoderKL.

Lower priority:

- IP-Adapter attention branch.
- ControlNet residual injection.
- LoRA runtime mutation/fusion.
- Cache/context-parallel paths.

## 13. Runtime staging plan

Stage 1: Parse Flux configs and load transformer/VAE weights. Accept external T5
and CLIP embeddings.

Stage 2: Implement latent pack/unpack and id generation parity.

Stage 3: Compile one `FluxTransformerBlock` and one `FluxSingleTransformerBlock`
with fixed random tensors.

Stage 4: Compile full transformer forward for `FLUX.1-schnell` shape with
embedded guidance disabled.

Stage 5: Add FlowMatch Euler scheduler step and one denoising-step parity.

Stage 6: Add full loop with scheduler host control, then VAE decode.

Stage 7: Add `FLUX.1-dev` guidance embeddings and dynamic shift.

Stage 8: Add attention provider optimization and guarded flash-style lowering.

Stage 9: Separate reports for img2img/inpaint/control/IP-Adapter/LoRA/Kontext.

## 14. Parity and validation plan

- Pack/unpack parity for `[B,16,128,128] <-> [B,4096,64]`.
- RoPE id generation parity for text/image ids.
- `FluxTransformerBlock` parity with random text/image tokens.
- `FluxSingleTransformerBlock` parity.
- Full `FluxTransformer2DModel` forward parity for schnell config.
- Guidance embedding parity for dev config.
- FlowMatch scheduler `set_timesteps` parity for static and dynamic shifting.
- One denoising-step parity with fixed embeddings, latents, and timestep.
- VAE scale/shift decode parity.
- Optional true CFG parity: separate positive/negative transformer calls.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 initially
  `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- Transformer forward by image sequence length: 1024, 4096, larger if supported.
- Joint attention backend comparison: native, flash-style Dinoml provider,
  xFormers/flex/sage if relevant.
- Dual-stream block vs single-stream block time.
- Pack/unpack overhead.
- Flow scheduler overhead.
- True CFG one-call equivalent not available; compare one vs two transformer
  calls.
- VAE decode overhead for 1024x1024.
- VRAM/workspace by sequence length and dtype.

## 16. Scope boundary and separate candidates

Separate review candidates, not ignored:

- `flux_img2img`
- `flux_inpaint_fill`
- `flux_control`
- `flux_controlnet`
- `flux_ip_adapter`
- `flux_lora_adapters`
- `flux_kontext`
- `flux_prior_redux`
- `flux2_klein`
- `flow_match_scheduler_matrix` for advanced FlowMatch options such as Karras,
  exponential, beta, stochastic, and per-token timesteps.

Ignored/out of scope for this audit unless explicitly selected:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker and NSFW filtering.
- Training, losses, dropout, and gradient checkpointing.

## 17. Final implementation checklist

- [ ] Parse Flux model index and component configs from `H:/configs`.
- [ ] Load `FluxTransformer2DModel` weights.
- [ ] Accept external T5 token embeddings and CLIP pooled embeddings.
- [ ] Implement latent pack/unpack and id generation.
- [ ] Implement timestep/text/guidance embedding path.
- [ ] Implement `FluxTransformerBlock` dual-stream parity.
- [ ] Implement `FluxSingleTransformerBlock` parity.
- [ ] Implement QK RMSNorm + RoPE + joint attention.
- [ ] Implement FlowMatch Euler static scheduler slice.
- [ ] Add dynamic shift and guidance embeddings for dev.
- [ ] Add one-step denoising parity.
- [ ] Add VAE 16-channel scale/shift decode boundary.
- [ ] Add guarded attention provider lowering.
- [ ] Create separate candidate reports for Flux variants and adapters.
