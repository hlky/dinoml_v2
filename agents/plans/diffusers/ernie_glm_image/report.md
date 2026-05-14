# Diffusers ERNIE/GLM Image Operator and Integration Report

Candidate slug: `ernie_glm_image`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  baidu/ERNIE-Image
  baidu/ERNIE-Image-Turbo
  akshan-main/tiny-ernie-image-modular-pipe
  zai-org/GLM-Image
  models123/GLM-Image
  Disty0/GLM-Image-SDNQ-4bit-dynamic
  Intel/GLM-Image-int4-AutoRound

Config sources:
  H:/configs/zai-org/GLM-Image/model_index.json
  H:/configs/models123/GLM-Image/model_index.json
  H:/configs/Disty0/GLM-Image-SDNQ-4bit-dynamic/model_index.json
  H:/configs/Intel/GLM-Image-int4-AutoRound/model_index.json
  Hugging Face cache reads for official component configs:
    zai-org/GLM-Image at snapshot 2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4
    baidu/ERNIE-Image at snapshot 5346b31d68c9c23758ba56ef8be5e9dc174c7f99
    baidu/ERNIE-Image-Turbo at snapshot bc68c81e2a1730a394d5fc9fae70713dee940140
    akshan-main/tiny-ernie-image-modular-pipe at snapshot b55394c5a8a7f54b034db6e5ca253c3b62733e51

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/ernie_image/pipeline_ernie_image.py
  X:/H/diffusers/src/diffusers/pipelines/glm_image/pipeline_glm_image.py
  X:/H/diffusers/src/diffusers/modular_pipelines/ernie_image/*.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_ernie_image.py
  X:/H/diffusers/src/diffusers/models/transformers/transformer_glm_image.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_flux2.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py
  X:/H/diffusers/src/diffusers/models/autoencoders/vae.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py
  X:/H/diffusers/src/diffusers/models/normalization.py
  X:/H/diffusers/src/diffusers/image_processor.py
  X:/H/diffusers/src/diffusers/loaders/lora_pipeline.py

External component configs inspected:
  ERNIE: Mistral3Model / TokenizersBackend text encoder and Ministral3ForCausalLM prompt enhancer configs.
  GLM: ByT5Tokenizer, T5EncoderModel glyph encoder, GlmImageProcessor, and GlmImageForConditionalGeneration configs.

Any missing files or assumptions:
  No official config read was blocked. ERNIE has no processor or vision_language_encoder component by design. GLM has no
  PE component. This report focuses on base text-to-image denoiser/runtime staging and inventories ERNIE prompt
  enhancer, ERNIE LoRA, and GLM image-to-image/KV-cache conditioning as separate candidates. Backend/training/safety/
  callbacks, XLA/NPU/MPS/Flax/ONNX, and multi-device paths are out of scope.
```

## 2. Pipeline and component graph

ERNIE-Image is a latent image DiT pipeline with optional prompt enhancement, variable-length Mistral3 text embeddings,
FlowMatch Euler, a 2x2 latent patchifier around a Flux2-style VAE, and conventional CFG batching.

```text
prompt
  -> optional Ministral3 prompt enhancer
  -> TokenizersBackend + Mistral3Model hidden_states[-2]
  -> variable-length text hidden list -> pad to [B,Tmax,3072] + lengths
  -> noisy latent map [B,128,H/16,W/16]
  -> denoising loop:
       ErnieImageTransformer2DModel over image tokens + text tokens
       CFG batch concat/chunk
       FlowMatchEulerDiscreteScheduler.step
  -> BN unstandardize -> 2x2 latent unpack [B,32,H/8,W/8]
  -> AutoencoderKLFlux2 decode -> clamp/postprocess
```

GLM-Image is a hybrid autoregressive plus diffusion decoder pipeline. The AR model produces prior visual-token IDs, the
ByT5/T5 glyph encoder supplies text embeddings, and the diffusion transformer adds prior-token embeddings to patchified
latents before joint text-image attention.

```text
prompt / optional condition images
  -> GlmImageProcessor + GlmImageForConditionalGeneration.generate
  -> prior_token_ids [B,S_img] and optional source image prior IDs/grids
  -> ByT5Tokenizer + T5EncoderModel prompt embeddings [B,L,1472]
  -> latent noise [B,16,H/8,W/8]
  -> optional image VAE encode -> per-layer KV cache write/read for i2i
  -> denoising loop:
       GlmImageTransformer2DModel(latents, text, prior tokens, size/crop, timestep)
       separate cond/uncond CFG calls
       FlowMatchEulerDiscreteScheduler.step
  -> latent unstandardize -> AutoencoderKL decode -> VaeImageProcessor
```

Required first-slice components:

| Family | Required components | Optional/cacheable stages |
| --- | --- | --- |
| ERNIE | `ErnieImageTransformer2DModel`, external text hidden tensors, `FlowMatchEulerDiscreteScheduler`, latent pack/unpack, `AutoencoderKLFlux2` decode boundary | Prompt enhancer output, prompt embeddings, text padding/lengths, scheduler sigmas, VAE BN stats |
| GLM | `GlmImageTransformer2DModel`, external prior token IDs, external T5 prompt embeddings, `FlowMatchEulerDiscreteScheduler`, AutoencoderKL decode boundary | AR prior-token generation, condition-image KV cache, prompt embeddings, RoPE tables, scheduler timesteps/sigmas |

Separate candidate reports:

| Candidate | Classes/files | Runtime delta |
| --- | --- | --- |
| `ernie_image_prompt_enhancer` | `ErnieImagePipeline._enhance_prompt_with_pe`, modular `ErnieImagePromptEnhancerStep` | Ministral3 causal LM generation, chat template JSON prompt rewrite, CPU/GPU text-generation stage before diffusion. |
| `ernie_image_lora` | `ErnieImageLoraLoaderMixin`, `PeftAdapterMixin` on transformer | Transformer LoRA load/fuse/unfuse/runtime adapter mutation. |
| `ernie_image_modular_pipeline` | `modular_pipelines/ernie_image/*` | Same base graph split into pipeline blocks and a `ClassifierFreeGuidance` component; useful staging surface. |
| `glm_image_i2i` | `GlmImagePipeline.generate_prior_tokens`, `GlmImageKVCache`, VAE encode path | Condition-image preprocessing, AR image-token extraction, VAE encode, per-layer KV cache write/read/skip. |
| `glm_image_quantized_repos` | Disty0/Intel GLM-Image model indexes | Same Diffusers pipeline class with quantized/offloaded component loading concerns; separate encoded-weight candidate. |
| `glm_image_ar_prior` | Transformers `GlmImageForConditionalGeneration` and `GlmImageProcessor` | 9B AR visual-token generation and processor grid contracts before DiT decode. |

No family-local ControlNet, T2I-Adapter, IP-Adapter, GLIGEN, inpaint, depth2img, or upscaling pipeline was found in the
inspected folders. GLM image-to-image is integrated into the same pipeline, not a separate Diffusers class.

## 3. Important config dimensions

Representative checkpoint sweep:

| Config | Pipeline | Denoiser | Hidden | Heads x dim | Layers | Latents | Text/prior | Scheduler |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `baidu/ERNIE-Image` | `ErnieImagePipeline` | `ErnieImageTransformer2DModel` | 4096 | 32 x 128 | 36 | packed 128 channels at H/16,W/16; unpacked VAE z=32 | Mistral3 hidden 3072; PE Ministral3 3B | FlowMatch static shift 4.0 |
| `baidu/ERNIE-Image-Turbo` | same | same shape | 4096 | 32 x 128 | 36 | same | same | FlowMatch static shift 4.0; docs use 8 NFEs and guidance 1.0 |
| `akshan-main/tiny-ernie-image-modular-pipe` | same | tiny debug | 32 | 4 x 8 | 1 | packed 16; VAE z=4 | GPT2 hidden 32 | FlowMatch static shift 1.0 |
| `zai-org/GLM-Image` | `GlmImagePipeline` | `GlmImageTransformer2DModel` | 4096 | 32 x 128 | 30 | VAE z=16 NCHW at H/8,W/8; patch tokens at H/16,W/16 | ByT5/T5 d_model 1472 + prior VQ IDs | FlowMatch dynamic linear shift |
| Disty0/Intel GLM variants | same model index | same class | config not fully swept | same expected | same expected | same expected | quantized component repos | same pipeline contract |

ERNIE transformer:

| Field | Official value | Source default note |
| --- | ---: | --- |
| `hidden_size` | 4096 | Source default 3072. |
| `num_layers` | 36 | Source default 24. |
| `num_attention_heads` | 32 | Head dim 128. |
| `ffn_hidden_size` | 12288 | SwiGLU-style separate gate/up projection. |
| `in_channels/out_channels` | 128 / 128 | This is packed latent channel count, not VAE latent channels. |
| `patch_size` | 1 | Model Conv2d patch embed is identity-sized on already packed latents. |
| `text_in_dim` | 3072 | Mistral3 hidden width. |
| `rope_axes_dim` / `rope_theta` | `[32,48,48]` / 256 | 3-axis frame-or-text/height/width sections sum to head dim. |

GLM transformer:

| Field | Official value |
| --- | ---: |
| `patch_size` | 2 |
| `in_channels/out_channels` | 16 / 16 |
| `num_layers` | 30 |
| `num_attention_heads` / `attention_head_dim` | 32 / 128 |
| inner dim | 4096 |
| `text_embed_dim` | 1472 |
| `time_embed_dim` / `condition_dim` | 512 / 256 |
| prior VQ codebook | 16384 |

VAE and scheduler:

| Family | VAE | Decode scale/stat contract | Scheduler config |
| --- | --- | --- | --- |
| ERNIE | `AutoencoderKLFlux2`, z=32, block channels `[128,256,512,512]`, layers/block 2, mid attention true, quant/post-quant true | Pipeline denoises packed 128-channel latents, then `latents * sqrt(bn.running_var + eps) + bn.running_mean`, then 2x2 unpack to z=32 before decode | `FlowMatchEulerDiscreteScheduler`, `shift=4.0`, `use_dynamic_shifting=false`, `time_shift_type=exponential`, `stochastic_sampling=false` |
| GLM | `AutoencoderKL`, z=16, block channels `[128,512,1024,1024]`, layers/block 3, no mid attention, no quant/post-quant conv | `latents * latents_std + latents_mean`, then decode; config also has `scaling_factor=0.18215` but pipeline uses explicit mean/std for this family | `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=true`, `base_shift=0.25`, `max_shift=0.75`, `base_image_seq_len=256`, `max_image_seq_len=4096`, `time_shift_type=linear` |

Recommended first Dinoml scheduler slice: FlowMatch Euler non-stochastic step. Stage ERNIE static-shift/custom-sigmas
first because the source schedule is just `linspace(1,0,N+1)[:-1]`; add GLM dynamic linear `mu` and paired
custom `timesteps/sigmas` next.

## 3a. Family variation traps

- ERNIE and GLM both use FlowMatch Euler and DiT denoisers, but their latent contracts differ: ERNIE denoises packed
  128-channel maps at H/16,W/16; GLM denoises normal 16-channel maps at H/8,W/8 and patchifies internally to tokens.
- ERNIE `in_channels=128` comes from 2x2 packing of VAE z=32 latents. GLM `in_channels=16` is the VAE latent channel
  count.
- ERNIE text is a list of variable-length hidden tensors from `hidden_states[-2]`, padded only after prompt and negative
  prompt expansion. Text masks are built from explicit `text_lens`.
- GLM requires prior token IDs for every image token. Prompt-only Dinoml denoiser parity can stub these IDs, but full
  pipeline parity depends on the AR model and processor grid semantics.
- GLM image-to-image writes condition-image K/V caches through the same transformer before denoising. Do not silently
  fold that state into base text-to-image.
- ERNIE CFG batches unconditional and conditional samples in one transformer call. GLM performs separate cond/uncond
  calls because prior-token drop and optional KV-cache modes differ.
- ERNIE Turbo has the same config shape as base but different intended loop settings: docs use 8 steps and guidance 1.0.
- ERNIE VAE uses BatchNorm running stats as latent standardization metadata; GLM uses explicit config vectors.
- GLM height/width must be divisible by `vae_scale_factor * patch_size * 2 = 32` in `check_inputs`, not merely by 16.
- NHWC/channel-last is only a guarded optimization inside VAE conv islands. Transformer token regions and scheduler
  broadcasting should be protected by a no-layout-translation guard.

## 4. Runtime tensor contract

For 1024x1024 text-to-image:

| Boundary | ERNIE | GLM |
| --- | --- | --- |
| Prompt text | Optional PE-rewritten string | Processor chat template with target grid markers; can include condition images |
| Text encoder output | Per-prompt hidden `hidden_states[-2]`, variable `[T,3072]` | T5 glyph encoder output `[B,L,1472]`, max sequence default 2048 |
| Text conditioning to denoiser | `text_bth [B,Tmax,3072]`, `text_lens [B]`; CFG doubles these lists before padding in classic pipeline | `prompt_embeds [B,L,1472]`; separate negative embeds for CFG |
| Prior tokens | None | `prior_token_ids [B,S_img]`, usually S_img=(H/16)*(W/16)=4096 for 1024px |
| Initial latent | `[B,128,64,64]` NCHW packed latent map | `[B,16,128,128]` NCHW latent map |
| Model token shape | ERNIE internally Conv2d p=1 -> image tokens `[B,4096,4096]`; concat image then text in `[S,B,H]` order inside blocks | Patchify p=2 -> image tokens `[B,4096,4096]`; text tokens projected to 4096 |
| Timestep | `[B]` dtype of transformer; source passes scheduler timestep directly | `t.expand(B) - 1` before transformer |
| Denoiser output | `[B,128,64,64]` | `[B,16,128,128]` |
| Scheduler state | sigma/timestep table from explicit sigmas | custom timesteps/sigmas plus dynamic `mu` from image sequence length |
| Decode input | BN unstandardize packed latent, unpack to `[B,32,128,128]` | mean/std unstandardize `[B,16,128,128]` |
| Decoded image | `[B,3,1024,1024]`, clamped and manually converted to PIL/NumPy | `[B,3,1024,1024]`, `VaeImageProcessor.postprocess` |

CPU/data-pipeline work: prompt enhancer, tokenization, AR prior-token generation, image validation/resizing, PIL/NumPy
conversion. GPU/runtime work: denoiser, latent pack/unpack, VAE encode/decode, scheduler/CFG arithmetic, GLM i2i KV-cache
prepass when enabled.

Precompute/cache candidates: prompt embeddings, negative embeddings, GLM prior token IDs, ERNIE text padding for fixed
batch, RoPE tables, scheduler tables, GLM condition-image KV cache, VAE BN or mean/std vectors.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and random normal.
- ERNIE 2x2 latent unpack/pack: `[B,128,H/16,W/16] <-> [B,32,H/8,W/8]` with flatten order `C,dh,dw,h,w`.
- GLM patchify/unpatchify: `[B,16,H/8,W/8] -> [B,(H/16)*(W/16),64] -> Linear`, reverse final projection.
- Text padding, lengths, mask construction, concat/chunk or separate CFG calls.
- GLM prior-token embedding lookup and boolean `prior_token_drop` zeroing.
- GLM optional KV-cache concat along sequence dimension for i2i.

GEMM/linear ops:

- ERNIE Conv2d patch embed `Conv2d(128 -> 4096, 1x1)`, text projection `Linear(3072 -> 4096, no bias)`.
- ERNIE per-block Q/K/V/O projections, no bias in sampled attention path; MLP gate/up/down `4096 -> 12288 -> 4096`.
- ERNIE AdaLN modulation `SiLU + Linear(4096 -> 6*4096)`, final AdaLN `Linear(4096 -> 8192)`, final `Linear(4096 -> 128)`.
- GLM image projector `Linear(16*2*2=64 -> 4096)`.
- GLM glyph projector `FeedForward(1472 -> 4096)`, prior embedding/projector, QKV/O, GELU FFN, final `Linear(4096 -> 64)`.
- GLM timestep/size/crop MLPs and PixArt-style projection.

Attention primitives:

- ERNIE single-stream joint image+text self-attention with RMSNorm Q/K, 3-axis RoPE, bool padding mask, head dim 128.
- GLM joint text+image attention with LayerNorm Q/K, image-token RoPE only, optional dense mixed mask, head dim 128.
- `attention_dispatch.py` native SDPA/eager path defines parity; flash-style provider is a guarded optimization.

Normalization and adaptive conditioning:

- ERNIE RMSNorm before attention/MLP, AdaLN scale/shift/gates, final LayerNorm-based AdaLN.
- GLM AdaLayerNormZero with 12 modulation/gate vectors for image and text streams, LayerNorm FFN norms, final custom AdaLN.
- VAE GroupNorm, ERNIE VAE BatchNorm stats, GLM VAE GroupNorm.

Scheduler/guidance:

- FlowMatch sigma/timestep table creation.
- Pointwise step `sample + (sigma_next - sigma) * model_output`.
- ERNIE CFG batch concat/chunk; GLM cond/uncond separate calls and prior-token drop.

VAE/postprocessing:

- ERNIE `AutoencoderKLFlux2`: Conv2d encoder/decoder, quant/post-quant conv, GroupNorm, SiLU, mid attention, tiling/slicing deferred.
- GLM `AutoencoderKL`: Conv2d ResNet/down/up blocks, GroupNorm, SiLU, no mid attention and no quant/post-quant for official config.

## 6. Denoiser/model breakdown

ERNIE forward:

```text
hidden_states [B,128,H/16,W/16]
-> Conv2d p=1 image embed -> image tokens [S_img,B,4096]
text_bth [B,T,3072] -> optional Linear -> text tokens [T,B,4096]
concat image then text
position ids:
  image ids = [text_len, y, x]
  text ids = [text_position,0,0]
3-axis RoPE and valid-text attention mask
timestep -> Timesteps(4096) -> TimestepEmbedding(4096)
global SiLU+Linear -> six AdaLN vectors broadcast over sequence
36 x ErnieImageSharedAdaLNBlock
final AdaLN + Linear -> image patch output [B,128,H/16,W/16]
```

ERNIE block:

```text
RMSNorm -> adaptive scale/shift -> QKV -> QK RMSNorm -> RoPE -> SDPA -> O projection
gated residual
RMSNorm -> adaptive scale/shift -> gate_proj/up_proj/GELU/product/down_proj
gated residual
```

GLM forward:

```text
hidden_states [B,16,H/8,W/8]
-> RoPE table for [H/16,W/16]
-> 2x2 patchify + Linear(64,4096)
encoder_hidden_states [B,L,1472] -> glyph FeedForward -> [B,L,4096]
prior_token_id [B,S_img] -> Embedding(16384,4096) -> prior FeedForward -> add to image tokens
timestep + target_size + crop_coords -> combined condition [B,512]
30 x GlmImageTransformerBlock
final AdaLN + Linear(4096,64) -> unpatchify -> [B,16,H/8,W/8]
```

GLM block:

```text
AdaLayerNormZero(image,text,temb) -> 12 scale/shift/gate vectors
cat text+image -> QKV -> QK LayerNorm -> image-suffix RoPE -> optional KV cache/mask -> SDPA -> split
gated residuals for image and text
LayerNorm + adaptive scale/shift -> shared GELU FeedForward on both streams
gated residuals for image and text
```

## 7. Attention requirements

ERNIE attention:

- Processor: `ErnieImageSingleStreamAttnProcessor` in `transformer_ernie_image.py`.
- Sequence order is image tokens first, text tokens second.
- Q/K/V shape before dispatch is `[B,S,heads,head_dim]`; official head shape 32 x 128.
- QK norm is RMSNorm when `qk_layernorm=true`.
- RoPE uses a non-interleaved rotate-half implementation and 3-axis frequency tensor `[B,S,1,head_dim]`.
- Bool mask is `[B,1,1,S]`, valid image tokens plus valid text positions.

GLM attention:

- Processor: `GlmImageAttnProcessor` in `transformer_glm_image.py`.
- Sequence order inside attention is text tokens first, image tokens second.
- QK norm is LayerNorm without affine over head dim.
- RoPE applies only to the image-token suffix, using `(cos,sin)` from `GlmImageRotaryPosEmbed`.
- I2I can prepend cached condition-image K/V per layer before attention.
- Source supports a dense mixed mask derived from text mask; base pipeline usually does not pass a text attention mask.

Flash/provider notes:

- Both official ERNIE and GLM use head dim 128 and noncausal attention, making flash-style kernels plausible for no-mask
  base cases.
- ERNIE padding masks and GLM mixed masks require mask-capable providers or fallback.
- QK norm, RoPE, prior-token addition, and AdaLN gates remain explicit first. Fuse only under strict span/order guards.

## 8. Scheduler and denoising-loop contract

ERNIE source loop:

```text
sigmas = linspace(1.0, 0.0, num_steps + 1)[:-1]
scheduler.set_timesteps(sigmas=sigmas, device=device)
for t in scheduler.timesteps:
  latent_model_input = cat([latents, latents]) if CFG else latents
  text_bth/text_lens already padded for uncond+cond when CFG
  pred = transformer(latent_model_input, t_batch, text_bth, text_lens)
  pred = uncond + scale * (cond - uncond) if CFG
  latents = scheduler.step(pred, t, latents).prev_sample
```

GLM source loop:

```text
timesteps = linspace(1000, 1, steps+1)[:-1] or custom
sigmas = timesteps / 1000
mu = (image_seq_len / base_seq_len)**0.5 * max_shift + base_shift
retrieve_timesteps(scheduler, timesteps, sigmas, mu=mu)
for t in timesteps:
  timestep = t.expand(B) - 1
  cond = transformer(latents, prompt_embeds, prior_ids, drop=False, timestep, size/crop, kv_cache)
  uncond = transformer(latents, negative_embeds, prior_ids, drop=True, timestep, size/crop, kv_cache) if CFG
  noise_pred = uncond + scale * (cond - uncond)
  latents = scheduler.step(noise_pred, t, latents)
```

Keep schedule construction, GLM AR prior generation, GLM i2i KV-cache write/read orchestration, and loop iteration as
host-visible state initially. Compile one denoiser step and pointwise scheduler/CFG arithmetic after table parity.

## 9. Position, timestep, and custom math

- ERNIE RoPE position IDs combine text length with image y/x and place text positions on axis 0. This is not the same
  as GLM/CogView image-only RoPE.
- ERNIE MLP is `down(up(x) * gelu(gate(x)))`; it is not Diffusers `FeedForward` GEGLU code.
- ERNIE decode unstandardizes packed latents with VAE `BatchNorm2d` running stats and `batch_norm_eps`, then unpatchifies.
- GLM `calculate_shift` is `sqrt(image_seq_len / base_seq_len) * max_shift + base_shift`, not the Flux/Qwen linear
  interpolation.
- GLM prior-token IDs are upsampled from AR tokens with nearest neighbor before DiT decode.
- GLM condition embedding sums sinusoidal timestep, crop coords, and target size projections, then applies SiLU.
- GLM transformer subtracts one from scheduler timesteps before embedding.

Precomputable: ERNIE/GLM RoPE tables for fixed shape and text length, ERNIE text padding for fixed batch, GLM prior
tokens, GLM size/crop embeddings, scheduler tables. Dynamic: prompt length, image resolution, CFG on/off, GLM condition
image count and cache, ERNIE PE output text.

## 10. Preprocessing and input packing

ERNIE:

- Optional PE builds a JSON user payload containing prompt/width/height and runs `pe.generate` through a chat template.
- Text tokenizer uses special tokens, truncation, no padding; empty prompts fall back to BOS or token 0.
- Text encoder runs with `output_hidden_states=True` and uses `hidden_states[-2][0]`.
- Positive and negative prompt embeddings are Python lists until padded to `[B,Tmax,text_in_dim]`.
- Source text-to-image initializes packed latents directly at `[B,128,H/16,W/16]`; VAE decode later unpacks to `[B,32,H/8,W/8]`.

GLM:

- Processor chat template emits AR model inputs, image grid metadata, and optional condition-image pixel values.
- AR model generates visual prior tokens; for text-to-image the target grid tokens are selected and upsampled.
- ByT5 token IDs are padded to an even length using a front pad expression, then padded to batch max length.
- Image-to-image preprocesses each condition image to a VAE-compatible multiple, encodes it, standardizes with mean/std,
  and runs transformer cache-write calls with empty text and condition prior IDs.
- Base output decodes mean/std-unstandardized NCHW latents through AutoencoderKL and `VaeImageProcessor`.

## 11. Graph rewrite / lowering opportunities

### Rewrite: ERNIE packed-latent unpack

Source pattern: packed `[B,4C,H,W] -> reshape(B,C,2,2,H,W) -> permute(0,1,4,2,5,3) -> [B,C,2H,2W]`.

Replacement: `ernie_latent_unpack2x2` or canonical reshape/permute/reshape.

Preconditions: packed channels divisible by 4, source NCHW semantic layout, VAE expects unpacked z channels, no tiled
decode path active. NHWC lowering must rewrite flatten order explicitly.

Parity test: random packed tensor round-trip through modular `ErnieImagePachifier`.

### Rewrite: ERNIE single-stream joint attention

Source pattern: concat image/text tokens, RMSNorm Q/K, 3-axis RoPE, SDPA, gated residual.

Replacement: joint-attention primitive with explicit image/text spans and mask.

Preconditions: sequence order image then text, qk RMSNorm enabled, RoPE axes match config, provider supports bool padding
mask or mask is absent.

Failure cases: LoRA/adapters not materialized, nonstandard `rope_axes_dim`, future context parallel split behavior.

### Rewrite: GLM prior-token patch transformer

Source pattern: patchify image latents, project text, embed prior IDs, add prior hidden states, run joint attention.

Replacement: explicit `glm_prior_conditioned_patch_dit_step` with prior-token embedding as a first-class input.

Preconditions: prior token count equals image patch token count, `prior_token_drop` is all false/true per cond/uncond
call, patch size 2, VQ codebook 16384.

Failure cases: i2i cache write/read mode, multiple condition images, mismatched AR grid and latent grid.

### Rewrite: FlowMatch Euler ERNIE/GLM slices

Source pattern: scalar `dt = sigma_next - sigma`, update `sample + dt * model_output`.

Replacement: fused pointwise scheduler step with artifact-visible sigma tables.

Preconditions: non-stochastic scheduler, scalar timesteps, no per-token sigma branch. GLM additionally needs dynamic
linear `mu` table parity.

## 12. Kernel fusion candidates

Highest priority:

- ERNIE/GLM head-dim-128 joint attention fallback and flash-style provider guards.
- QK norm + RoPE application kernels for ERNIE 3-axis and GLM image-suffix variants.
- AdaLN modulation, gated residual, and MLP epilogues in both transformer blocks.
- ERNIE packed latent unpack and GLM patchify/unpatchify.
- FlowMatch Euler step and CFG arithmetic.

Medium priority:

- ERNIE `AutoencoderKLFlux2` decode island: Conv2d/GroupNorm/SiLU/residual/mid-attention plus BN-stat unstandardize.
- GLM AutoencoderKL decode: no quant/post-quant and no mid attention for official config.
- GLM prior-token embedding/projection plus zero/drop behavior.
- GLM condition-image KV-cache staging for i2i.
- Text mask construction and attention-mask lowering.

Lower priority:

- ERNIE prompt enhancer and GLM AR model compilation.
- LoRA/adapter mutation.
- VAE tiling/slicing overlap blend.
- Quantized GLM repo ingestion/offload policy.

## 13. Runtime staging plan

Stage 1: Parse component configs for ERNIE base/Turbo/tiny and GLM official. Admit external prompt embeddings and, for
GLM, external prior token IDs.

Stage 2: Implement latent contracts: ERNIE packed latent map and unpack; GLM NCHW latent map plus 2D patchify/unpatchify.

Stage 3: Bring up one ERNIE block and one GLM block with random tensors, including QK norm, RoPE, AdaLN gates, and masks.

Stage 4: Full tiny ERNIE transformer parity, then official ERNIE one denoiser step at 1024 shape.

Stage 5: GLM transformer one-step parity with stubbed prior tokens and prompt embeddings. Add prior-token drop behavior
for cond/uncond CFG.

Stage 6: FlowMatch Euler scheduler parity: ERNIE static/custom-sigma slice, then GLM dynamic linear `mu`.

Stage 7: VAE decode boundaries: ERNIE Flux2 BN+unpack+decode, GLM AutoencoderKL mean/std+decode.

Stage 8: Full Python-host denoising loop with compiled denoiser step. Add prompt enhancer, GLM AR prior, GLM i2i KV-cache,
LoRA, and quantized variants as separate slices.

First Dinoml staging recommendation: start with `ernie_image_denoiser_step_external_text`, because it avoids AR prior
tokens and image-condition KV state while still exercising the same head-dim-128 attention, AdaLN gates, large GEMMs,
FlowMatch step, and packed latent decode boundary needed by GLM.

## 14. Parity and validation plan

- Config parse tests for ERNIE base, ERNIE Turbo, tiny ERNIE, and GLM official component configs.
- ERNIE text encoding contract: hidden `[-2]`, variable-length list, padding, lengths, positive/negative CFG ordering.
- ERNIE latent unpack parity and BN-stat unstandardize parity.
- GLM prompt/glyph embedding shape tests and prior-token ID shape validation.
- GLM AR prior-token extraction/upsample parity, initially as CPU/data-pipeline tests.
- ERNIE 3-axis RoPE parity and GLM image-suffix RoPE parity.
- Single attention processor parity for ERNIE and GLM with no mask and with text mask.
- Single block parity for ERNIE and GLM.
- Full tiny ERNIE transformer parity; reduced-shape GLM transformer parity with stubbed prior IDs.
- FlowMatch scheduler table and one-step parity for ERNIE static shift and GLM dynamic linear shift.
- CFG parity: ERNIE batch concat/chunk and GLM separate cond/uncond with prior-token drop.
- VAE decode parity for ERNIE Flux2 and GLM AutoencoderKL.
- Suggested tolerances: scheduler/custom fp32 `rtol=1e-5, atol=1e-6`; transformer fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16
  initially `rtol=2e-2, atol=2e-2`.

## 15. Performance probes

- ERNIE denoiser step by resolution and text length: 512, 1024, 2048 where memory allows.
- GLM denoiser step with prior token lengths 1024, 4096, and high-res grids.
- Attention backend comparison for head dim 128 with and without masks.
- Per-block time split: QKV/QK norm/RoPE/attention/projection/MLP.
- ERNIE CFG one-call batch doubling versus guidance disabled/Turbo guidance 1.0.
- GLM separate cond/uncond call overhead and prior-token drop overhead.
- Scheduler and CFG arithmetic overhead versus denoiser time.
- ERNIE Flux2 decode and GLM AutoencoderKL decode throughput.
- GLM AR prior-token generation latency, separately from diffusion decode.
- VRAM/workspace by dtype, prompt length, and sequence length.

## 16. Scope boundary and separate candidates

Separate candidate reports, not ignored:

- `ernie_image_prompt_enhancer`: Ministral3 generation and chat-template prompt rewrite.
- `ernie_image_lora`: transformer LoRA load/fuse/unfuse/runtime adapter state.
- `ernie_image_flux2_vae`: Flux2 VAE decode/encode with BN-stat latent contract, tiling/slicing, and mid attention.
- `glm_image_ar_prior`: `GlmImageForConditionalGeneration` visual-token generation and processor grid contract.
- `glm_image_i2i`: condition-image preprocessing, VAE encode, prior image IDs, and transformer KV cache write/read/skip.
- `glm_image_quantized`: Disty0/Intel quantized component repositories and encoded/offloaded weight policy.
- Rare FlowMatch options: stochastic sampling, advanced sigma conversions, and per-token timesteps.

Ignored/out of scope for this audit:

- Multi-GPU/context parallel paths.
- Callback mutation and interactive interrupt.
- XLA, NPU, MPS, Flax, and ONNX variants.
- Safety checker/NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- ControlNet, T2I-Adapter, IP-Adapter, GLIGEN, inpaint, depth2img, and upscaling because no active implementation was
  found in the inspected ERNIE/GLM folders.

## 17. Final implementation checklist

- [ ] Parse ERNIE and GLM model/component configs.
- [ ] Accept external ERNIE text hidden lists and GLM prompt embeddings/prior token IDs.
- [ ] Implement ERNIE text padding and length-mask contract.
- [ ] Implement ERNIE packed latent unpack and BN-stat unstandardize.
- [ ] Implement GLM 2D patchify/unpatchify and prior-token embedding/drop behavior.
- [ ] Implement ERNIE 3-axis RoPE and GLM image-suffix RoPE.
- [ ] Implement ERNIE and GLM QK norm + attention fallback parity.
- [ ] Implement ERNIE and GLM AdaLN/gated residual/MLP block parity.
- [ ] Add full tiny ERNIE transformer parity.
- [ ] Add reduced-shape GLM transformer parity with stub prior IDs.
- [ ] Implement FlowMatch Euler ERNIE static/custom-sigma scheduler slice.
- [ ] Implement FlowMatch Euler GLM dynamic linear-shift scheduler slice.
- [ ] Add ERNIE CFG batch concat/chunk parity.
- [ ] Add GLM separate cond/uncond CFG and prior-token-drop parity.
- [ ] Add ERNIE Flux2 VAE decode boundary.
- [ ] Add GLM AutoencoderKL decode boundary.
- [ ] Benchmark attention, denoiser step, scheduler/CFG, and VAE decode.
- [ ] Split prompt enhancer, GLM AR prior, GLM i2i KV-cache, LoRA, and quantized repos into separate tickets.
