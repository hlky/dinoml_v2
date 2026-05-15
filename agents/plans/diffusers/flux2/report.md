# Diffusers Flux2 Operator and Integration Report

Candidate slug: `flux2`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  black-forest-labs/FLUX.2-dev
  black-forest-labs/FLUX.2-klein-4B
  black-forest-labs/FLUX.2-klein-base-4B
  black-forest-labs/FLUX.2-klein-9B
  black-forest-labs/FLUX.2-klein-9b-kv
  tiny-random/flux2

Config sources:
  H:/configs/black-forest-labs/FLUX.2-dev-NVFP4/model_index.json
    local placeholder only: {}
  H:/configs/black-forest-labs/FLUX.2-klein-4B/model_index.json
  H:/configs/black-forest-labs/FLUX.2-klein-base-4B/model_index.json
  H:/configs/black-forest-labs/FLUX.2-klein-9b-kv-fp8/model_index.json
    local placeholder only: {}
  H:/configs/tiny-random/flux2/model_index.json
  Hugging Face raw/API reads for component configs listed below.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/flux2/pipeline_flux2.py
  diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
  diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein_kv.py
  diffusers/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py
  diffusers/src/diffusers/pipelines/flux2/image_processor.py
  diffusers/src/diffusers/pipelines/flux2/pipeline_output.py
  diffusers/src/diffusers/pipelines/flux2/system_messages.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_flux2.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/normalization.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/loaders/lora_pipeline.py
  diffusers/src/diffusers/loaders/lora_conversion_utils.py
  diffusers/src/diffusers/loaders/single_file_model.py
  diffusers/src/diffusers/loaders/single_file_utils.py

External component configs inspected:
  black-forest-labs/FLUX.2-dev:
    model_index.json, transformer/config.json, vae/config.json,
    scheduler/scheduler_config.json, text_encoder/config.json,
    tokenizer/processor_config.json, tokenizer/preprocessor_config.json.
  black-forest-labs/FLUX.2-klein-4B and FLUX.2-klein-base-4B:
    model_index.json, transformer/config.json, vae/config.json,
    scheduler/scheduler_config.json, text_encoder/config.json,
    tokenizer/tokenizer_config.json.
  tiny-random/flux2:
    model_index.json, transformer/config.json, vae/config.json,
    scheduler/scheduler_config.json, text_encoder/config.json.
  9B mirror configs:
    Runware/BFL-FLUX.2-klein-9B, Runware/BFL-FLUX.2-klein-base-9B,
    mlx-community/FLUX.2-klein-9B, tonera/FLUX.2-klein-9B-Nunchaku.

Any missing files or assumptions:
  Official black-forest-labs/FLUX.2-dev and 9B/KV repos are gated. Authenticated
  HF user `hlky` could list gated repo metadata and file sizes, but direct config
  reads for black-forest-labs/FLUX.2-klein-9B, FLUX.2-klein-base-9B,
  FLUX.2-klein-base-9b-fp8, and FLUX.2-klein-9b-kv returned 403. Open mirror
  configs are used only to characterize 9B dimensions and are labeled as mirror
  evidence. No fetched configs were persisted because this task owns only this
  report path.
```

## 2. Pipeline and component graph

Flux2 is a flow-matching latent image transformer family. The base `Flux2Pipeline`
uses a Mistral3/Pixtral multimodal text encoder stack; the Klein pipelines use
Qwen3 text-only prompt embeddings. Both feed `Flux2Transformer2DModel`,
`FlowMatchEulerDiscreteScheduler`, and `AutoencoderKLFlux2`.

```text
prompt and optional reference images
  -> text / multimodal encoder hidden-state stack
  -> prompt_embeds [B, L, joint_attention_dim] and text ids [B, L, 4]
  -> latent noise map in patched VAE space [B, 128, H/16, W/16]
  -> packed image tokens [B, (H/16)*(W/16), 128] and image ids [B, S, 4]
  -> denoising loop:
       Flux2Transformer2DModel(hidden_states, prompt_embeds, timestep,
                               optional embedded guidance, ids,
                               optional reference image tokens/KV cache)
       optional true CFG second call for non-distilled Klein
       FlowMatchEulerDiscreteScheduler.step
  -> unpack tokens to [B, 128, H/16, W/16]
  -> VAE BN unnormalization -> unpatchify to [B, 32, H/8, W/8]
  -> AutoencoderKLFlux2 decode/postprocess
```

Required first-slice components:

- `Flux2Transformer2DModel` denoiser with external prompt embeddings.
- `AutoencoderKLFlux2` decode boundary and its BN plus 2x2 patch/unpatch contract.
- `FlowMatchEulerDiscreteScheduler` with Flux2 empirical `mu`.
- Prompt embeddings as external inputs first; compiling Mistral3/Pixtral or Qwen3
  is useful later but not required for denoiser admission.

Separate candidate reports:

| Candidate | Primary classes/files | Why separate |
| --- | --- | --- |
| `flux2_dev_mistral3_pixtral_text` | `Flux2Pipeline`, `Mistral3ForConditionalGeneration`, `PixtralProcessor` | Multimodal prompt formatting, caption upsampling, image tokens, and stacked hidden-state embeddings differ from Klein. |
| `flux2_klein_qwen3_text` | `Flux2KleinPipeline`, `Qwen3ForCausalLM` | Qwen3 hidden-state stack width and true CFG behavior differ from dev. |
| `flux2_klein_kv` | `Flux2KleinKVPipeline`, KV processors in `transformer_flux2.py` | Reference-image K/V extraction/cached attention changes the denoising loop and attention masks. |
| `flux2_klein_inpaint` | `Flux2KleinInpaintPipeline` | VAE encode, mask pack, strength timestep slicing, inpaint blend, and optional reference images. |
| `flux2_lora_runtime` | `Flux2LoraLoaderMixin`, Flux2 LoRA conversion utilities | Runtime adapter mutation and fused single-block projection key mapping. |
| `flux2_single_file_quantized` | `single_file_utils.py`, single-file model map | Direct checkpoint conversion and future GGUF/FP8/NVFP4 ingestion are integration surfaces, not first denoiser ops. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Denoiser width/depth | Text encoder | Prompt embed width | Guidance | Scheduler |
| --- | --- | --- | --- | ---: | --- | --- |
| `black-forest-labs/FLUX.2-dev` | `Flux2Pipeline` | 48 heads x 128, 8 dual + 48 single, inner 6144 | Mistral3/Pixtral | 15360 | source default `guidance_embeds=true` | FlowMatch dynamic shift |
| `black-forest-labs/FLUX.2-klein-4B` | `Flux2KleinPipeline`, distilled | 24 heads x 128, 5 dual + 20 single, inner 3072 | Qwen3 2.5B-ish config | 7680 | `guidance_embeds=false`; true CFG ignored because distilled | FlowMatch dynamic shift |
| `black-forest-labs/FLUX.2-klein-base-4B` | `Flux2KleinPipeline`, base | same as 4B distilled | Qwen3 hidden 2560 | 7680 | `guidance_embeds=false`; true CFG enabled by pipeline when `scale > 1` | FlowMatch dynamic shift |
| 9B Klein mirrors | `Flux2KleinPipeline` | 32 heads x 128, 8 dual + 24 single, inner 4096 | Qwen3 hidden 4096 | 12288 | `guidance_embeds=false`; distilled flag depends repo | FlowMatch dynamic shift |
| `tiny-random/flux2` | `Flux2Pipeline` | 2 heads x 32, 2 dual + 2 single, inner 64 | tiny Mistral3 stub | 8 | source default if omitted | FlowMatch dynamic shift |

Transformer config fields:

| Field | Dev | Klein 4B | Klein 9B mirror | Source default |
| --- | ---: | ---: | ---: | ---: |
| `in_channels` | 128 | 128 | 128 | 128 |
| `out_channels` | null -> 128 | null -> 128 | null -> 128 | null -> `in_channels` |
| `patch_size` | 1 | 1 | 1 | 1 |
| `num_layers` | 8 | 5 | 8 | 8 |
| `num_single_layers` | 48 | 20 | 24 | 48 |
| `num_attention_heads` | 48 | 24 | 32 | 48 |
| `attention_head_dim` | 128 | 128 | 128 | 128 |
| inner dim | 6144 | 3072 | 4096 | heads * head_dim |
| `joint_attention_dim` | 15360 | 7680 | 12288 | 15360 |
| `axes_dims_rope` | `[32,32,32,32]` | same | same | `(32,32,32,32)` |
| `rope_theta` | 2000 | 2000 | 2000 | 2000 |
| `mlp_ratio` | 3.0 | 3.0 | 3.0 | 3.0 |
| `guidance_embeds` | omitted, effective true | false | false | true |

VAE config:

| Field | Value | Source |
| --- | --- | --- |
| class | `AutoencoderKLFlux2` | model index/config |
| latent channels | 32 | VAE config |
| patch size | `[2,2]` | VAE config and source |
| packed transformer channels | 128 = 32 * 2 * 2 | source/config inference |
| scale factor | 8 from 4 VAE blocks; image processor uses 16 because of 2x2 patching | source |
| blocks | Down/UpEncoderBlock2D x4, channels 128/256/512/512, 2 layers per block | config |
| quant/post-quant conv | true/true | config |
| extra normalization | non-affine `BatchNorm2d(128)` over patchified latents | source/config |

Scheduler:

- Source and sampled configs use `FlowMatchEulerDiscreteScheduler`.
- Configs use `shift=3.0`, `base_shift=0.5`, `max_shift=1.15`,
  `use_dynamic_shifting=true`, `time_shift_type="exponential"`,
  `stochastic_sampling=false`.
- Pipeline supplies custom `sigmas = linspace(1.0, 1 / steps, steps)` unless
  `scheduler.config.use_flow_sigmas` is true.
- Pipeline computes `mu` with `compute_empirical_mu(image_seq_len, num_steps)`,
  not the older Flux1 `calculate_shift` helper.
- Recommended first Dinoml scheduler slice: FlowMatch Euler with custom sigmas,
  `mu`, dynamic shift table generation, and deterministic step update.

## 3a. Family variation traps

- Flux2 transformer channel width 128 is patchified latent width. The VAE latent
  map is 32 channels at H/8 x W/8; `_patchify_latents` converts it to 128
  channels at H/16 x W/16 before `_pack_latents`.
- Flux1 reports use 3-axis ids and 16-channel VAE latents; Flux2 uses 4-axis
  ids `(T,H,W,L)` and 32-channel VAE latents with BN normalization.
- Dev and Klein use different text encoders and different selected hidden-state
  layers: dev stacks Mistral3 layers `(10,20,30)` to 15360, while Klein stacks
  Qwen3 layers `(9,18,27)` to 7680 or 12288 depending hidden size.
- Dev passes embedded guidance into the model; Klein configs set
  `guidance_embeds=false` and use true CFG only when `is_distilled=false`.
- `Flux2Pipeline` can process optional input/reference images and append their
  packed tokens to the image token sequence. Scheduler updates only the
  generated latent span.
- Klein KV changes attention semantics: reference tokens self-attend during
  extract mode, then cached reference K/V are inserted into later attention
  calls.
- Single-stream blocks fuse QKV and MLP-in projections into one Linear and fuse
  attention-out plus MLP-out into one Linear. DinoML should preserve this as a
  fusion candidate rather than expanding accidentally into Flux1-like pieces.
- VAE decode requires BN unnormalization before unpatchify; VAE encode requires
  patchify then BN normalization. This replaces Flux1 scale/shift arithmetic.
- Official 9B/KV configs were gated for this account; mirror data should not be
  treated as official licensing or release metadata.

## 4. Runtime tensor contract

For a 1024x1024 generated image:

| Boundary | Tensor | Shape | Layout / notes |
| --- | --- | --- | --- |
| prompt embeds, dev | `prompt_embeds` | `[B,512,15360]` | Mistral3 hidden layers 10/20/30 stacked then flattened. |
| prompt embeds, Klein 4B | `prompt_embeds` | `[B,512,7680]` | Qwen3 hidden 2560 x 3 layers. |
| prompt embeds, Klein 9B | `prompt_embeds` | `[B,512,12288]` | mirror config: Qwen3 hidden 4096 x 3 layers. |
| text ids | `txt_ids` | `[B,512,4]` | Cartesian `(T,H,W,L)` with text length in L axis. |
| latent noise map | internal | `[B,128,64,64]` | Patchified VAE latent map, source NCHW. |
| packed latents | `latents` | `[B,4096,128]` | Token-major transformer input. |
| latent ids | `latent_ids` | `[B,4096,4]` | Generated from patchified spatial grid. |
| reference image latents | optional | `[B,S_ref,128]` | VAE encode -> patchify -> BN normalize -> pack. |
| timestep | per step | `[B]` | Pipeline passes `t / 1000`; transformer multiplies by 1000. |
| guidance | dev only | `[B]` | Pipeline expands `guidance_scale`; transformer multiplies by 1000. |
| transformer output | `noise_pred` | `[B,S_total,128]` then sliced to `[B,S_gen,128]` | Output tokens for generated span only feed scheduler. |
| scheduler state | `latents` | `[B,4096,128]` | Flow update over generated tokens. |
| unpacked patched latents | decode prep | `[B,128,64,64]` | Scatter by ids or fixed H/W in Klein. |
| VAE latent map | decode input | `[B,32,128,128]` | BN unnormalize then unpatchify. |
| decoded image | pre-postprocess | `[B,3,1024,1024]` | NCHW from VAE decoder. |

CPU/data pipeline:

- Prompt chat template formatting, tokenization, optional caption upsampling,
  image validation/resizing/cropping, PIL/NumPy postprocess.

GPU/runtime candidates:

- Prompt embeddings can be precomputed and reused.
- Latent ids, text ids, and FlowMatch timestep/sigma tables can be cached per
  shape/step count.
- VAE reference image latents can be precomputed for repeated editing requests.
- KV caches in `Flux2KleinKVPipeline` persist across denoising steps but not
  across unrelated requests unless explicitly owned by a runtime session.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW image preprocessing boundary; token-major transformer core.
- 2x2 VAE latent patchify:
  `[B,32,H/8,W/8] -> [B,128,H/16,W/16]`.
- Token pack/unpack:
  `[B,128,H/16,W/16] <-> [B,(H/16)*(W/16),128]`.
- Scatter-based unpack by ids for variable reference layouts.
- Text/image id creation and concatenation for 4-axis RoPE.
- Reference-token concat along sequence dim; generated-token output slicing.
- Inpaint mask resize to packed spatial size, pack to `[B,S,1]`, broadcast blend.

GEMM/linear ops:

- `x_embedder`: Linear(128 -> inner), no bias.
- `context_embedder`: Linear(joint_attention_dim -> inner), no bias.
- Timestep/guidance embedding MLPs: sinusoidal 256 -> inner.
- Modulation Linear(inner -> inner * 6) for dual streams and Linear(inner -> inner * 3) for single stream.
- Dual-stream attention image Q/K/V and text add-Q/add-K/add-V projections.
- Single-stream fused projection:
  Linear(inner -> 3*inner + 2*mlp_hidden), where `mlp_hidden = inner * 3`.
- Single-stream fused output:
  Linear(inner + mlp_hidden -> inner).
- Dual-stream SwiGLU feed-forward:
  Linear(inner -> 2*mlp_hidden) -> SiLU(x1) * x2 -> Linear(mlp_hidden -> inner).
- Final `proj_out`: Linear(inner -> 128), no bias.

Attention primitives:

- Noncausal joint text+image attention in dual blocks.
- Noncausal single-stream self-attention over concatenated text+image tokens.
- QK RMSNorm per head before RoPE and attention.
- 4-axis RoPE with `axes_dims_rope`.
- KV-cache attention for Klein KV:
  extract mode has two attention calls (`txt+img` to all tokens, ref self-only);
  cached mode injects cached ref K/V.

Normalization/adaptive conditioning:

- LayerNorm without affine in transformer blocks.
- RMSNorm for Q/K and added Q/K.
- `AdaLayerNormContinuous` before output projection.
- Non-affine BatchNorm2d over VAE patchified latent width 128.
- GroupNorm inside VAE encoder/decoder inherited from shared VAE blocks.

Scheduler/guidance arithmetic:

- FlowMatch Euler table generation and step.
- Optional true CFG for non-distilled Klein:
  `neg + guidance_scale * (pos - neg)`.
- Inpaint `scale_noise` for init image latents and per-step masked blend.

VAE/postprocessing:

- Conv2d, ResNetBlock2D, GroupNorm, SiLU, up/down blocks, mid attention.
- Quant and post-quant 1x1 convs.
- BN normalize/unnormalize plus patchify/unpatchify around VAE latent map.

## 6. Denoiser/model breakdown

Top-level `Flux2Transformer2DModel.forward`:

```text
hidden_states [B,S_img,128] -> x_embedder -> [B,S_img,inner]
encoder_hidden_states [B,S_txt,joint_dim] -> context_embedder -> [B,S_txt,inner]
timestep (+ optional guidance) -> time_guidance_embed -> [B,inner]
time embedding -> modulation tensors for dual image, dual text, and single blocks
ids -> Flux2PosEmbed 4-axis RoPE for text and image
dual-stream blocks
concat text + image streams
single-stream parallel blocks
drop text/ref tokens
AdaLayerNormContinuous -> Linear(inner -> 128)
```

Dual block:

```text
image: LayerNorm -> adaptive shift/scale -> QKV -> QK RMSNorm
text:  LayerNorm -> adaptive shift/scale -> add-QKV -> added QK RMSNorm
concat text+image Q/K/V -> RoPE -> dispatch_attention_fn
split text/image outputs -> output projections
gated residual attention
LayerNorm -> adaptive shift/scale -> SwiGLU FF -> gated residual
same FF path for text stream
```

Single block:

```text
optional concat text + image if not already concatenated
LayerNorm -> adaptive shift/scale
single fused Linear -> QKV plus SwiGLU input
QK RMSNorm -> RoPE -> attention
SwiGLU in parallel
concat(attn_output, mlp_output) -> fused output Linear
gated residual
```

KV-cache mode:

- First step with reference image tokens prepended to image tokens calls
  `kv_cache_mode="extract"`, stores post-RoPE reference K/V for each dual and
  single block, and uses reference-specific modulation at fixed timestep 0.
- Later steps call `kv_cache_mode="cached"` with generated latents only; cached
  reference K/V are inserted between text and generated image K/V.

## 7. Attention requirements

Primary implementation is `transformer_flux2.py` processors calling
`dispatch_attention_fn` from `attention_dispatch.py`.

Required variants:

| Variant | Source processor | Sequence | Notes |
| --- | --- | --- | --- |
| dual joint attention | `Flux2AttnProcessor` | `[text, image]` after QKV concat | No mask in base source path; added text projections produce text output too. |
| single parallel attention | `Flux2ParallelSelfAttnProcessor` | `[text, image]` | QKV and MLP-in are one projection. |
| dual KV attention | `Flux2KVAttnProcessor` | `[text, ref, image]` or `[text, image] + cache` | Extract stores ref K/V; ref tokens self-attend only in extract. |
| single KV attention | `Flux2KVParallelSelfAttnProcessor` | same | Same causal/cache rule, plus fused MLP projection. |

Flash constraints and provider notes:

- Base non-KV Flux2 is dense, noncausal, mask-free, bf16-oriented, with head dim
  128. A flash-style provider is plausible if QK RMSNorm and RoPE are explicit
  pre-attention ops.
- KV extract mode is not a single plain full-attention call: it splits ref
  queries into a self-attention island and a `txt+img` to all-tokens island.
  Flash lowering needs either two provider calls or a supported block mask.
- KV cached mode changes K/V length by injecting cached reference tokens. The
  runtime ABI must carry cache tensors per layer with shape
  `[B,S_ref,heads,head_dim]`.
- `attention.py` explicitly skips QKV fusion for modules such as Flux2 whose
  single-block QKV projections are already fused with MLP projections.
- Eager/native dispatch through `dispatch_attention_fn` is the parity path;
  flash/xFormers/flex/sage choices should be guarded provider decisions.

## 8. Scheduler and denoising-loop contract

Flux2 uses `FlowMatchEulerDiscreteScheduler`.

Pipeline setup:

```text
sigmas = linspace(1.0, 1 / num_inference_steps, num_inference_steps)
if scheduler.config.use_flow_sigmas: sigmas = None
image_seq_len = generated_latents.shape[1]
mu = compute_empirical_mu(image_seq_len, num_inference_steps)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
```

`compute_empirical_mu` is piecewise:

```text
if image_seq_len > 4300:
  mu = 0.00016927 * image_seq_len + 0.45666666
else:
  interpolate between a 10-step and 200-step line using num_steps
```

Step math remains FlowMatch Euler:

```text
prev_sample = sample + (sigma_next - sigma) * model_output
```

Guidance modes:

- Dev embedded guidance: pipeline passes `guidance_scale` tensor to the model;
  transformer embeds it if `guidance_embedder` exists.
- Klein true CFG: `do_classifier_free_guidance` is
  `guidance_scale > 1 and not is_distilled`. It performs positive and negative
  transformer calls using `cache_context("cond")` / `cache_context("uncond")`.
- Klein distilled: source warns that guidance scale is ignored.

Host/runtime split:

- Keep schedule setup, image/reference preprocessing, caption upsampling,
  callback/interrupt, and KV cache lifetime host-visible first.
- Compile one transformer step plus scheduler update first.
- Treat true CFG and KV cache as explicit loop variants, not hidden Python
  control flow inside a single denoiser op.

## 9. Position, timestep, and custom math

Position:

- `Flux2PosEmbed` loops over `len(axes_dims_rope)` rather than `ids.shape[-1]`.
- Dev/Klein configs use four axes `[32,32,32,32]`; ids are `(T,H,W,L)`.
- Text ids use `T=0,H=0,W=0,L=0..L-1`.
- Generated latent ids use `T=0,H,W,L=0`.
- Reference image ids use T offsets `10,20,...` by default.

Timestep/guidance:

- Pipeline divides timestep by 1000 before the model call.
- Transformer multiplies timestep and guidance by 1000 before sinusoidal
  embedding.
- KV extract mode computes reference modulation with `ref_fixed_timestep=0.0`.

VAE custom math:

- Encode: VAE encode -> sample/mode -> patchify -> BN normalize.
- Decode: unpack -> BN unnormalize -> unpatchify -> VAE decode.
- Direct latent image inputs to inpaint/reference paths are still BN-normalized
  before entering transformer token space.

Precomputable:

- Text ids and prompt embeds by prompt/max length/layer set.
- Latent ids by output resolution.
- Reference image packed latents and ids.
- FlowMatch timesteps/sigmas/mu by generated sequence length and step count.

## 10. Preprocessing and input packing

Base dev:

- Formats prompts as chat messages with a system prompt and uses
  `PixtralProcessor.apply_chat_template`.
- Optional caption upsampling calls `text_encoder.generate` and can include
  images. This is CPU/LLM host work, not first denoiser runtime.
- `_get_mistral_3_small_prompt_embeds` stacks hidden states from layers
  `(10,20,30)` and flattens `[B,3,L,5120] -> [B,L,15360]`.

Klein:

- Qwen3 chat template with `enable_thinking=False`.
- `_get_qwen3_prompt_embeds` stacks layers `(9,18,27)`.
- 4B config gives `[B,L,3*2560]`; 9B mirror gives `[B,L,3*4096]`.

Images/reference:

- `Flux2ImageProcessor` validates PIL images, min side 64, max aspect ratio 8,
  and max area 1024x1024 for reference/conditioning paths.
- Images are resized to multiples of `vae_scale_factor * 2` (normally 16).
- Multiple reference images may be horizontally concatenated for prompt
  upsampling; denoiser reference images are encoded separately and concatenated
  as token sequences.

Inpaint:

- Init image is VAE encoded to patchified latents.
- Mask is resized directly to packed spatial size `[H/16,W/16]` and packed to
  `[B,S,1]`.
- Condition tokens include the encoded original image and optional reference
  image tokens.
- Per step, scheduler output is blended:
  `(1 - mask) * init_latents_proper + mask * latents`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: Flux2 VAE latent codec boundary

Source pattern:

```text
encode: image -> VAE latent [B,32,H/8,W/8] -> patchify -> BN normalize -> pack
decode: unpack -> BN unnormalize -> unpatchify -> VAE decode
```

Replacement: expose a `flux2_latent_codec` boundary with explicit BN parameters
and 2x2 patch order.

Preconditions:

- VAE `latent_channels=32`, `patch_size=(2,2)`, BN running stats shape 128.
- H/W divisible by 16 at image boundary.

Failure cases:

- Reusing Flux1 scale/shift instead of BN stats.
- NHWC layout translation that changes patch flatten order without a matching
  weight/layout rewrite.

Parity test:

- Random `[B,32,128,128]` latent map through patchify/BN/pack/unpack/unBN/
  unpatchify must round-trip exactly except floating-point BN arithmetic.

### Rewrite: single-stream parallel transformer block

Source pattern:

```text
LayerNorm/adaptive -> Linear(QKV + 2*MLP) -> attention and SwiGLU in parallel
concat(attn, mlp) -> Linear -> gated residual
```

Replacement: a fused block descriptor with one input projection and one output
projection, not separate QKV and MLP-in GEMMs.

Preconditions:

- `Flux2ParallelSelfAttention`, no custom processor that changes projection
  layout.
- MLP activation is `SwiGLU` splitting the projected MLP part in half.

Failure cases:

- Treating the module as ordinary Flux1 single block and losing the fused
  projection weight layout.

Parity test:

- One random `Flux2SingleTransformerBlock` against Diffusers for 4B and dev
  shapes, with and without KV processor.

### Rewrite: KV extract/cache attention

Source pattern:

```text
step 0: [ref, img] tokens -> store ref K/V after RoPE, special attention
steps 1+: [img] tokens + cached ref K/V -> attention
```

Replacement: explicit per-layer cache ABI and two attention modes.

Preconditions:

- Fixed reference token count and layer count.
- Cache dtype/head_dim match current transformer config.
- Reference modulation uses fixed timestep 0.

Failure cases:

- Folding reference tokens into normal full attention for extract mode; ref
  tokens are supposed to self-attend only.
- Treating cache as request-global mutable state instead of session-owned state.

Parity test:

- Compare first two denoising steps of `Flux2KleinKVPipeline` against source:
  step 0 cache extraction, step 1 cached forward.

### Rewrite: FlowMatch empirical-mu schedule

Source pattern:

```text
sigmas linspace -> compute_empirical_mu(seq_len, steps) -> set_timesteps -> step
```

Replacement: host-generated schedule table with scalar step kernel.

Preconditions:

- `stochastic_sampling=false`, scalar timesteps, no per-token timesteps.

Failure cases:

- Reusing Flux1 shift calculation or scheduler config `base/max` shift alone.

Parity test:

- Compare timesteps/sigmas and one-step output for several sequence lengths:
  1024, 4096, and >4300.

## 12. Kernel fusion candidates

Highest priority:

- Flux2 single-stream fused projection block:
  QKV + SwiGLU input projection, QK RMSNorm, RoPE, attention, fused output.
- Dual-stream QKV/add-QKV + QK RMSNorm + RoPE + joint attention.
- Adaptive modulation + LayerNorm + gated residual.
- VAE patchify/BN/pack and unpack/BN/unpatchify kernels.
- FlowMatch Euler pointwise step and true CFG arithmetic.

Medium priority:

- Timestep/guidance embedding MLP plus modulation Linear generation.
- KV cache extraction/injection attention provider.
- Inpaint mask blend and `scale_noise`.
- VAE decode conv island with GroupNorm/SiLU/upsample fusion.

Lower priority:

- Mistral3/Pixtral and Qwen3 text encoder compilation.
- Caption upsampling generate path.
- LoRA conversion/runtime mutation.
- Single-file and quantized checkpoint ingestion.

## 13. Runtime staging plan

Stage 1: Parse Flux2 configs and load weights for `tiny-random/flux2` and
`black-forest-labs/FLUX.2-klein-4B` metadata. Accept external prompt embeddings.

Stage 2: Implement Flux2 latent codec helpers: patchify, pack, ids, unpack, BN
unnormalize, unpatchify. Validate VAE decode boundary separately.

Stage 3: Compile one dual block and one single parallel block for the 4B shape.

Stage 4: Full `Flux2Transformer2DModel` forward for Klein 4B with no true CFG
and no reference images.

Stage 5: Add FlowMatch Euler empirical-mu scheduler and one denoising-step
parity.

Stage 6: Add `FLUX.2-dev` dimensions and embedded guidance, still with external
prompt embeddings.

Stage 7: Add non-distilled Klein true CFG as a two-call loop contract.

Stage 8: Add `flux2_klein_inpaint` with VAE encode external first, then mask
pack/blend.

Stage 9: Add `flux2_klein_kv` with explicit per-layer cache ABI.

Stage 10: Add attention/norm/parallel-block fusion and guarded flash providers.

## 14. Parity and validation plan

- Patchify/unpatchify and pack/unpack round-trip for `[B,32,128,128]`.
- BN normalize/unnormalize parity using VAE running stats.
- Text/id generation parity for prompt length 512 and latent sizes 64x64 plus
  non-square dimensions.
- `Flux2TransformerBlock` parity with random text/image tokens.
- `Flux2SingleTransformerBlock` parity with fused parallel projection.
- Full transformer forward parity for Klein 4B random tensors.
- Dev embedded-guidance forward parity.
- FlowMatch `compute_empirical_mu`, `set_timesteps`, and one `step` parity.
- True CFG parity for base Klein: positive/negative calls and arithmetic.
- Inpaint strength slicing, `scale_noise`, mask pack, and blend parity.
- KV cache parity for extract and cached modes.
- VAE decode parity with prepared `[B,32,H/8,W/8]` latents.
- Suggested tolerances: scheduler fp32 `rtol=1e-5, atol=1e-6`; transformer fp32
  `rtol=1e-4, atol=1e-5`; bf16/fp16 initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- One denoiser step by config: Klein 4B, Klein 9B mirror shape, dev shape.
- Sequence-length sweep: 1024, 4096, >4300 tokens to cross empirical-mu branch.
- Dual blocks versus single parallel blocks.
- Attention backend comparison for base dense attention, then KV extract/cached.
- VAE latent codec overhead: patchify/BN/pack/unpack/BN/unpatchify.
- VAE decode throughput for 1024x1024.
- True CFG one-call versus two-call loop overhead.
- KV cache memory: per layer `S_ref * heads * head_dim * 2`.
- Inpaint blend/mask overhead and VAE encode cost.
- Weight loading/offload timing for large dev/9B BF16, FP8, NVFP4, or GGUF
  variants as separate runtime-storage probes.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `flux2_dev_mistral3_pixtral_text`: Mistral3 multimodal prompt encoder,
  Pixtral image preprocessing, and caption upsampling.
- `flux2_klein_qwen3_text`: Qwen3 prompt encoder, hidden-state layer selection,
  base versus distilled guidance.
- `flux2_klein_inpaint`: mask/image/reference preprocessing, strength slicing,
  and masked latent blend.
- `flux2_klein_kv`: reference-image K/V cache extraction and cached attention.
- `flux2_lora_runtime`: `Flux2LoraLoaderMixin`, Kohya/ai-toolkit conversion,
  PEFT hotswap, adapter scale.
- `flux2_quantized_loading`: official NVFP4/FP8/GGUF/community quantized
  checkpoints and explicit encoded-constant/offload policies.
- `flow_match_scheduler_advanced`: stochastic/per-token/Karras/exponential/beta
  FlowMatch options not active in sampled configs.

Out of scope for this audit:

- Implementing Dinoml ops or runtime changes.
- Multi-GPU/context parallel execution.
- Callback mutation and interactive interrupt behavior.
- XLA, NPU, MPS, Flax, and ONNX branches.
- Safety checker/NSFW filtering.
- Training, losses, dropout, gradient checkpointing.
- Persisting fetched configs into `H:/configs` under this write-limited task.

## 17. Final implementation checklist

- [ ] Parse Flux2 model indexes and component configs.
- [ ] Admit external prompt embeddings for dev `[B,L,15360]`, Klein 4B
      `[B,L,7680]`, and Klein 9B `[B,L,12288]`.
- [ ] Implement 4-axis text/image id generation.
- [ ] Implement VAE patchify/BN/pack and unpack/BN/unpatchify helpers.
- [ ] Implement `AutoencoderKLFlux2` decode boundary or call it as a separate
      codec stage first.
- [ ] Implement Flux2 dual-stream block parity.
- [ ] Implement Flux2 single-stream parallel block parity.
- [ ] Implement QK RMSNorm + RoPE + dense joint attention.
- [ ] Implement FlowMatch Euler empirical-mu scheduler table and step.
- [ ] Add one-step denoising parity for Klein 4B.
- [ ] Add dev embedded-guidance parity.
- [ ] Add base Klein true-CFG two-call parity.
- [ ] Add inpaint mask/strength/blend parity.
- [ ] Add KV-cache extract/cached attention ABI and parity.
- [ ] Add guarded flash/attention provider lowering after eager parity.
- [ ] Add performance probes for block, full denoiser, VAE decode, scheduler,
      CFG, and KV cache memory.
