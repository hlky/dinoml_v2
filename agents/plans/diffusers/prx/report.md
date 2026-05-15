# Diffusers PRX Operator and Integration Report

Target slug: `prx`

Runtime scope: PRX text-to-image base pipeline. First Dinoml slice should accept
externally supplied T5-Gemma prompt embeddings and attention masks, run the PRX
transformer denoiser on NCHW latents, keep FlowMatch Euler scheduler state
host-visible, and decode through the Flux-style `AutoencoderKL`.

Ignored per user scope: XLA/NPU/MPS, Flax/ONNX, safety/NSFW, training/loss/
dropout/gradient checkpointing, callbacks/interrupt mutation, and multi-GPU or
context parallel paths.

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  Photoroom/prx-512-t2i
  Photoroom/prx-1024-t2i-beta

Config sources:
  H:/configs/Photoroom/prx-512-t2i/model_index.json
  H:/configs/Photoroom/prx-1024-t2i-beta/model_index.json
  Official HF configs fetched through huggingface_hub for inspection:
    model_index.json
    transformer/config.json
    scheduler/scheduler_config.json
    vae/config.json
    text_encoder/config.json
    text_encoder/model.safetensors.index.json
    tokenizer/tokenizer_config.json
    tokenizer/special_tokens_map.json
  Official HF repo file listings inspected for safetensors shard/file sizes.

Pipeline files inspected:
  diffusers/src/diffusers/pipelines/prx/pipeline_prx.py
  diffusers/src/diffusers/pipelines/prx/pipeline_output.py

Model files inspected:
  diffusers/src/diffusers/models/transformers/transformer_prx.py
  diffusers/src/diffusers/models/attention_dispatch.py
  diffusers/src/diffusers/models/normalization.py
  diffusers/src/diffusers/models/embeddings.py
  diffusers/src/diffusers/models/autoencoders/autoencoder_kl.py

Scheduler/processors/helpers inspected:
  diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  diffusers/src/diffusers/image_processor.py
  C:/Users/user/AppData/Local/Programs/Python/Python312/Lib/site-packages/transformers/models/t5gemma/modeling_t5gemma.py

Closest prior reports inspected:
  agents/plans/diffusers/scheduler_matrix/report.md
  agents/plans/diffusers/flux/report.md
  agents/plans/diffusers/pixart/report.md
  agents/plans/diffusers/stable_diffusion_3/report.md

External component configs inspected:
  T5GemmaEncoder config and tokenizer config from the official PRX repos.

Any missing files or assumptions:
  Local H:/configs only had top-level model_index.json files. Official component
  JSON configs were accessible without an auth blocker through huggingface_hub,
  but they were not written back to H:/configs because this worker owns only this
  report path. Top-level and transformer safetensors index files are absent
  (404); transformer and VAE weights are single safetensors files. Text encoder
  safetensors index metadata is available. No gated-config blocker remains.
```

## 2. Pipeline and component graph

PRX is a latent text-to-image pipeline with a T5-Gemma text encoder, a
PRX-specific image transformer denoiser, FlowMatch Euler scheduler, and an
optional VAE. The base pipeline declares `model_cpu_offload_seq =
"text_encoder->transformer->vae"` and optional component `vae`.

```text
prompt / negative_prompt
  -> TextPreprocessor clean_text
  -> GemmaTokenizer/GemmaTokenizerFast, max_length=256, right padding
  -> T5GemmaEncoder last_hidden_state [B, Ltxt, 2304] and bool mask [B, Ltxt]
  -> CFG batch concat of negative and positive text embeddings/masks
  -> latent noise initialization [B,16,H/8,W/8]
  -> denoising loop:
       duplicate latents for CFG
       normalized timestep t / scheduler.num_train_timesteps
       PRXTransformer2DModel(NCHW latents, T5-Gemma embeddings, text mask)
       true CFG arithmetic
       FlowMatchEulerDiscreteScheduler.step
  -> AutoencoderKL decode((latents / 0.3611) + 0.1159)
  -> optional resolution-bin resize/crop
  -> PixArtImageProcessor postprocess
```

Required first-slice components:

| Component | Class | File | First-slice treatment |
| --- | --- | --- | --- |
| Pipeline | `PRXPipeline` | `pipeline_prx.py` | Source of runtime contract, CFG batching, binning, and decode scaling. |
| Denoiser | `PRXTransformer2DModel` | `transformer_prx.py` | Main compiled target. |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | `scheduling_flow_match_euler_discrete.py` | First slice: static shift 3.0, deterministic Euler update. |
| Text encoder | `T5GemmaEncoder` | Transformers `modeling_t5gemma.py` | Adjacent stage; first denoiser slice can accept cached embeddings/masks. |
| Tokenizer | `GemmaTokenizer` / `GemmaTokenizerFast` | external Transformers | CPU/data-pipeline stage. |
| VAE | `AutoencoderKL` | `autoencoder_kl.py` | Decode required for image output; encode only for future variants. |
| Processor | `PixArtImageProcessor` | `image_processor.py` | Resolution binning and postprocess. |

Separate candidate reports:

| Surface | PRX status | Candidate slug/order |
| --- | --- | --- |
| LoRA/runtime adapters | `PRXPipeline` inherits `LoraLoaderMixin`; transformer inherits `AttentionMixin`, but no PRX-specific LoRA path is active in `__call__`. | `prx_lora_adapters` after base denoiser. |
| Textual inversion | `PRXPipeline` inherits `TextualInversionLoaderMixin`; tokenizer/embedding mutation is outside the base first-slice graph. | `prx_textual_inversion` after tokenizer/encoder integration. |
| Single-file loading | `FromSingleFileMixin` is inherited. This is a loading/weight-format path, not a denoiser op change. | `prx_single_file_loading` if needed. |
| IP-Adapter | No PRX pipeline or `PRXAttention` image-encoder side branch is present. | No PRX candidate unless a fork adds it. |
| ControlNet | No PRX ControlNet class or side-residual call path is present. | No PRX candidate. |
| T2I-Adapter | Not wired in PRX source. | No PRX candidate. |
| GLIGEN | No GLIGEN branch in PRX pipeline or block. | No PRX candidate. |
| img2img | No PRX img2img pipeline in the folder; VAE encode is not used by base `__call__`. | `prx_img2img` only if a new pipeline appears. |
| inpaint | No mask/image latent path in PRX source. | No PRX candidate. |
| depth2img | Not present. | No PRX candidate. |
| upscaling | Not present. | No PRX candidate. |
| AutoencoderDC codec variant | `PRXPipeline` accepts `AutoencoderDC` and uses `spatial_compression_ratio` when present, but sampled official configs use `AutoencoderKL`. | `prx_autoencoderdc_codec` only if an official DC checkpoint is selected. |

## 3. Important config dimensions

Representative checkpoint sweep:

| Repo | Pipeline | default image | latent | patch | tokens at native square | hidden | heads x dim | depth | text width | scheduler | VAE |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `Photoroom/prx-512-t2i` | `PRXPipeline` | 512 | `[B,16,64,64]` | 2 | 1024 | 1792 | 28 x 64 | 16 | 2304 | FlowMatch Euler shift 3.0 | Flux `AutoencoderKL` scale 0.3611 / shift 0.1159 |
| `Photoroom/prx-1024-t2i-beta` | `PRXPipeline` | 1024 | `[B,16,128,128]` | 2 | 4096 | 1792 | 28 x 64 | 16 | 2304 | FlowMatch Euler shift 3.0 | same VAE config |

Transformer config from official `transformer/config.json`:

| Field | Value | Source |
| --- | ---: | --- |
| `in_channels` | 16 | component config |
| `patch_size` | 2 | component config |
| patch feature width | 64 | inferred as `16 * 2 * 2` |
| `context_in_dim` | 2304 | component config / T5-Gemma hidden size |
| `hidden_size` | 1792 | component config |
| `mlp_ratio` | 3.5 | component config |
| MLP hidden | 6272 | inferred as `int(1792 * 3.5)` |
| `num_heads` | 28 | component config |
| head dim | 64 | inferred as `1792 / 28` |
| `depth` | 16 | component config |
| `axes_dim` | `[32, 32]` | component config |
| `theta` | 10000 | component config |
| `time_factor` | 1000.0 | component config |
| `time_max_period` | 10000 | component config |

Text encoder and tokenizer config:

| Field | Value | Source |
| --- | ---: | --- |
| Architecture | `T5GemmaEncoder` | text encoder config |
| hidden size | 2304 | text encoder config |
| layers | 26 | text encoder config |
| attention heads | 8 | text encoder config |
| KV heads | 4 | text encoder config |
| head dim | 256 | text encoder config |
| intermediate size | 9216 | text encoder config |
| layer types | alternating `sliding_attention`, `full_attention` | text encoder config |
| sliding window | 4096 | text encoder config |
| vocab size | 256000 | text encoder config |
| max positions | 8192 | text encoder config |
| `rms_norm_eps` | `1e-6` | text encoder config |
| activation | `gelu_pytorch_tanh` | text encoder config |
| tokenizer class | `GemmaTokenizer` in tokenizer config, `GemmaTokenizerFast` in model index | configs |
| tokenizer max length | 256 | tokenizer config |
| padding side | right | tokenizer config |

VAE and scheduler config:

| Component | Field | Value | Source |
| --- | --- | ---: | --- |
| VAE | class | `AutoencoderKL` | component config |
| VAE | source name | `black-forest-labs/FLUX.1-dev` | component config |
| VAE | latent channels | 16 | component config |
| VAE | sample size | 1024 | component config |
| VAE | block channels | `[128,256,512,512]` | component config |
| VAE | layers per block | 2 | component config |
| VAE | mid-block attention | true | component config |
| VAE | force upcast | true | component config |
| VAE | scale factor | 8 | inferred as `2 ** (len(block_out_channels)-1)` |
| VAE | scaling / shift | 0.3611 / 0.1159 | component config |
| Scheduler | class | `FlowMatchEulerDiscreteScheduler` | scheduler config |
| Scheduler | train timesteps | 1000 | scheduler config |
| Scheduler | shift | 3.0 | scheduler config |
| Scheduler | dynamic shifting | false | source default, omitted config |
| Scheduler | stochastic sampling | false | source default, omitted config |

Weight metadata from official HF file listing:

| Repo | Component | Files | Size metadata |
| --- | --- | --- | --- |
| both | text encoder | 3 sharded safetensors plus `model.safetensors.index.json` | index metadata says 2,614,341,888 params, 10,457,367,552 bytes |
| `prx-512-t2i` | transformer | `transformer/diffusion_pytorch_model.safetensors` | 4,682,786,544 bytes; no index |
| `prx-1024-t2i-beta` | transformer | `transformer/diffusion_pytorch_model.safetensors` | 4,682,786,544 bytes; no index |
| both | VAE | `vae/diffusion_pytorch_model.safetensors` | 335,306,212 bytes; no index |

## 3a. Family variation traps

- PRX is not Flux despite sharing a Flux VAE and Flux-inspired RoPE helpers. It
  does not pack latents in the pipeline; patchify/unpatchify happens inside
  `PRXTransformer2DModel`.
- The pipeline checks image height/width divisibility by `vae_scale_factor`
  only. The transformer additionally requires latent height and width divisible
  by `patch_size=2`, so unbinned custom image dimensions should be guarded as
  multiples of `vae_scale_factor * patch_size = 16` for the sampled configs.
- CFG is a single doubled-batch transformer call, not two separate model calls.
  Text embeddings and text masks are concatenated over batch.
- The scheduler input to the transformer is normalized in the pipeline:
  `t_cont = t / scheduler.config.num_train_timesteps`; the scheduler step still
  receives the original scheduler timestep `t`.
- The pipeline does not call `scheduler.scale_model_input`.
- The source supports optional `AutoencoderDC` and `vae=None`, but official
  sampled configs use `AutoencoderKL`. `vae=None` can only return latent/pt
  outputs.
- Attention is image-query attention over concatenated text plus image keys and
  values. Text tokens are not updated by the PRX blocks.
- Text padding masks matter. The tokenizer pads to max length 256 and the PRX
  attention processor expands the text mask into a dense joint mask over
  text+image keys.
- Source NCHW layout is semantic for VAE, latent tensor, patchify, and
  unpatchify. Channel-last is an optimization candidate only inside local
  Conv/VAE regions or linearized token regions with explicit entry/exit guards.
- Resolution binning changes model runtime dimensions before decode, then
  resizes/crops decoded tensors back to the requested size.

## 4. Runtime tensor contract

Pipeline inputs after CPU preprocessing:

| Input | Shape/type | Notes |
| --- | --- | --- |
| `prompt` / `negative_prompt` | string or list of strings | Cleaned by `TextPreprocessor`, tokenized to max length 256. |
| `prompt_embeds` | `[B,Ltxt,2304]` | Can bypass tokenizer/text encoder. |
| `negative_prompt_embeds` | `[B,Ltxt,2304]` | Required when precomputed embeds are supplied and `guidance_scale > 1`. |
| `prompt_attention_mask` | `[B,Ltxt]`, bool | Optional with precomputed embeds; generated by tokenizer otherwise. |
| `height`, `width` | ints | Default to `default_sample_size`; optional aspect-ratio binning first. |
| `latents` | `[B*num_images,16,H/8,W/8]` | Optional pre-noised latent input. |

Denoiser step contract:

| Tensor | Shape for 512 native | Shape for 1024 native | Layout |
| --- | ---: | ---: | --- |
| latents | `[B,16,64,64]` | `[B,16,128,128]` | NCHW |
| CFG latents input | `[2B,16,64,64]` | `[2B,16,128,128]` | NCHW |
| prompt/context embeds after CFG | `[2B,256,2304]` | same | batch, sequence, feature |
| prompt mask after CFG | `[2B,256]` | same | bool |
| transformer timestep | `[2B]` | `[2B]` | `t / 1000` for sampled scheduler |
| patch tokens | `[2B,1024,64]` | `[2B,4096,64]` | internal |
| hidden tokens | `[2B,1024,1792]` | `[2B,4096,1792]` | internal |
| transformer output | `[2B,16,64,64]` | `[2B,16,128,128]` | NCHW |
| CFG output | `[B,16,H/8,W/8]` | `[B,16,H/8,W/8]` | NCHW |

Patchify/unpatchify in `transformer_prx.py`:

```text
img2seq:
  [B,C,H,W]
  -> reshape [B,C,H/p,p,W/p,p]
  -> einsum "nchpwq->nhwcpq"
  -> reshape [B,(H/p)*(W/p),C*p*p]

seq2img:
  [B,(H/p)*(W/p),C*p*p]
  -> reshape [B,H/p,W/p,C,p,p]
  -> einsum "nhwcpq->nchpwq"
  -> reshape [B,C,H,W]
```

VAE decode:

```text
vae_input = (latents / vae.config.scaling_factor) + vae.config.shift_factor
image = vae.decode(vae_input, return_dict=False)[0]
if binning: image = resize_and_crop_tensor(image, orig_width, orig_height)
image = image_processor.postprocess(image, output_type)
```

Precomputable/cacheable tensors:

- Token IDs and attention masks for prompts.
- T5-Gemma prompt embeddings and negative prompt embeddings.
- CFG-concatenated prompt embedding batches when prompt/guidance batch is
  stable.
- Image patch ids and RoPE tensors for a fixed latent shape.
- Scheduler timesteps and sigmas for a fixed step count/custom-timestep list.
- VAE decode can be staged separately from denoising.

## 5. Operator coverage checklist

Tensor/layout ops:

- NCHW latent allocation and random normal initialization.
- `cat` over batch for CFG latents and embeddings.
- `chunk` over batch for CFG output split.
- `reshape`, `view`, `permute`, `transpose`, `einsum`-equivalent patchify and
  unpatchify.
- `cat` over attention key/value sequence dimension for text plus image K/V.
- `arange`, zero init, coordinate grid fill, repeat for image ids.
- Boolean mask concat, cast, unsqueeze, expand.
- Bilinear interpolate and center crop in `PixArtImageProcessor` for binned
  output resize.

GEMM/linear ops:

- Patch input projection: `Linear(64 -> 1792, bias=True)`.
- Text projection: `Linear(2304 -> 1792, bias=True)`.
- Timestep MLP: `Linear(256 -> 1792) -> SiLU -> Linear(1792 -> 1792)`.
- Per-block modulation: `SiLU -> Linear(1792 -> 6*1792)`.
- Per-block image QKV: `Linear(1792 -> 3*1792, bias=False)`.
- Per-block text KV: `Linear(1792 -> 2*1792, bias=False)`.
- Per-block attention output: `Linear(1792 -> 1792, bias=False)`.
- Per-block MLP: gate/up `Linear(1792 -> 6272, bias=False)`, down
  `Linear(6272 -> 1792, bias=False)`.
- Final adaptive norm MLP: `SiLU -> Linear(1792 -> 2*1792)`.
- Final projection: `Linear(1792 -> 64, bias=True)`.

Attention primitives:

- Non-causal cross/source attention: Q from image tokens, K/V from concatenated
  text and image tokens.
- 28 heads, head dim 64, query length `(H/16)*(W/16)`, key length
  `256 + query_length` for sampled max text length.
- Q/K RMSNorm for image Q and K, K RMSNorm for text K.
- 2D RoPE applied only to image Q and image K.
- Dense bool attention mask `[B, heads, L_img, L_txt + L_img]` when text mask is
  present.

Normalization and adaptive conditioning:

- LayerNorm without affine, eps `1e-6`, on image tokens.
- RMSNorm with affine, eps `1e-6`, on per-head Q/K tensors.
- Adaptive LayerNorm scale/shift/gate from timestep embedding.
- Final adaptive LayerNorm scale/shift.

Position/timestep/custom math:

- `get_timestep_embedding` with embedding dim 256, max period 10000,
  `flip_sin_to_cos=True`, `downscale_freq_shift=0.0`, and `scale=1000.0`.
- 2D RoPE frequency construction with `theta=10000`, axes `[32,32]`.
- `apply_rope` uses a 2x2 rotation matrix representation and casts
  intermediate math to fp32.

Scheduler and guidance arithmetic:

- FlowMatch Euler `set_timesteps`.
- Deterministic update `prev_sample = sample + (sigma_next - sigma) *
  model_output`.
- CFG `noise_uncond + guidance_scale * (noise_text - noise_uncond)`.
- Optional `eta` kwargs probe is inactive for FlowMatch Euler.

VAE/postprocessing ops:

- `AutoencoderKL` decode with Conv2d, Resnet/attention mid block, upsampling,
  GroupNorm, SiLU, and output Conv2d.
- Decode scale/shift arithmetic.
- Postprocess from model tensor to PIL/NumPy/latent as requested.

## 6. Denoiser/model breakdown

`PRXTransformer2DModel.forward`:

```text
encoder_hidden_states [B,Ltxt,2304]
  -> txt_in [B,Ltxt,1792]

hidden_states [B,16,H,W]
  -> img2seq patch_size=2 [B,(H/2)*(W/2),64] in latent pixel units
  -> img_in [B,Limg,1792]

image ids [B,Limg,2]
  -> PRXEmbedND axes [32,32] -> image RoPE [B,1,Limg,32,2,2]

timestep [B]
  -> sinusoidal embedding [B,256]
  -> MLPEmbedder [B,1792]

16 x PRXBlock
  -> FinalLayer
  -> seq2img patch_size=2 [B,16,H,W]
```

`PRXBlock`:

```text
temb -> Modulation -> (attn shift/scale/gate, mlp shift/scale/gate)
hidden_mod = (1 + attn_scale) * LayerNorm(hidden) + attn_shift
attn_out = PRXAttention(hidden_mod, text_tokens, text_mask, image_rope)
hidden = hidden + attn_gate * attn_out
x = (1 + mlp_scale) * LayerNorm(hidden) + mlp_shift
mlp = down_proj(GELU_tanh(gate_proj(x)) * up_proj(x))
hidden = hidden + mlp_gate * mlp
```

`PRXAttention`:

```text
img_qkv = Linear(hidden) -> reshape [3,B,H,Limg,64]
img_q, img_k = RMSNorm(img_q), RMSNorm(img_k)
txt_kv = Linear(text) -> reshape [2,B,H,Ltxt,64]
txt_k = RMSNorm(txt_k)
img_q/img_k = RoPE(img_q/img_k)
k = cat(txt_k, img_k, dim=sequence)
v = cat(txt_v, img_v, dim=sequence)
attention(query=img_q, key=k, value=v, optional joint mask)
to_out Linear -> image-token residual
```

Branch controls active in sampled configs:

- No guidance embedding inside the model; CFG is loop-side arithmetic only.
- No pooled CLIP path.
- No text-token update path.
- No learned sigma / doubled output channel branch.
- Dropout modules exist but are `0.0` in inference.

## 7. Attention requirements

Required PRX denoiser attention:

| Field | Value |
| --- | --- |
| Attention type | image-query attention over text+image K/V |
| Query | image hidden tokens `[B,Limg,28,64]` |
| Key/value | text tokens first, then image tokens `[B,256+Limg,28,64]` |
| Mask | optional bool mask expanded to `[B,28,Limg,256+Limg]` |
| Causality | non-causal |
| RoPE | image Q/K only, 2D axes `[32,32]` |
| QK norm | RMSNorm on image Q/K and text K |
| GQA | no, Q/K/V all use 28 heads after PRX projections |
| Primary source path | `PRXAttnProcessor2_0` -> `dispatch_attention_fn` |
| Eager/native parity | PyTorch native/eager attention through Diffusers dispatch |

Flash feasibility:

- Diffusers dispatch supports flash, flash varlen, native SDPA, flex, sage, and
  xFormers backends, but PRX itself passes a dense bool joint mask whenever text
  masks are present.
- Diffusers flash-attn 2/3/4 non-varlen paths reject `attn_mask`; sage paths
  also reject `attn_mask`. Native SDPA/eager is therefore the reliable parity
  path for padded prompts.
- A Dinoml flash-style provider is feasible under stricter preconditions:
  all text tokens unmasked, or a varlen implementation that lowers the text
  padding mask to packed Q/KV lengths while preserving the concatenated
  text+image key order. It must support different Q and KV lengths, non-causal
  attention, fp16/bf16, head dim 64, and pre-applied RoPE/QK norm.
- The most useful first fused attention boundary is
  `img_qkv + txt_kv + QK norms + RoPE + attention + output projection`, but a
  staged implementation can start with separate GEMMs and a native SDPA-like
  attention kernel.

T5-Gemma encoder attention, if compiled later:

- Encoder-only stack, 26 layers, hidden 2304.
- Alternating sliding-window and full self-attention.
- 8 query heads, 4 KV heads, head dim 256, RoPE, RMSNorm, tanh-GELU MLP.
- First PRX denoiser integration can avoid this by accepting cached prompt
  embeddings.

## 8. Scheduler and denoising-loop contract

PRX uses `FlowMatchEulerDiscreteScheduler` directly.

Pipeline setup:

```text
if timesteps is provided:
  scheduler.set_timesteps(timesteps=timesteps, device=device)
else:
  scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps
```

Loop-side work:

```text
if CFG:
  latents_in = cat([latents, latents], batch)
  t_cont = (t.float() / scheduler.config.num_train_timesteps).repeat(2)
else:
  latents_in = latents
  t_cont = t.float() / scheduler.config.num_train_timesteps

noise_pred = transformer(latents_in, t_cont, ca_embed, ca_mask)
if CFG:
  noise_pred = uncond + guidance_scale * (text - uncond)
latents = scheduler.step(noise_pred, t, latents).prev_sample
```

Sampled scheduler config:

- `num_train_timesteps=1000`.
- `shift=3.0`.
- Omitted source defaults: `use_dynamic_shifting=False`,
  `stochastic_sampling=False`, `invert_sigmas=False`, no Karras/exponential/
  beta conversion, no terminal shift.

Recommended first Dinoml scheduler slice:

- FlowMatch Euler with static shift 3.0.
- `num_inference_steps` and custom descending `timesteps`.
- Deterministic `prev_sample = sample + dt * model_output`.
- Host-visible `timesteps`, `sigmas`, and `step_index`.
- Defer `sigmas` input because PRX pipeline does not expose it even though the
  scheduler class supports it.
- Defer dynamic shifting, per-token timesteps, stochastic sampling, Karras/
  exponential/beta conversions, and `scale_noise` until an img2img/inpaint or
  alternate PRX variant needs them.

## 9. Position, timestep, and custom math

Timestep embedding:

```text
t_cont = scheduler_timestep / 1000
emb = get_timestep_embedding(t_cont, 256, max_period=10000,
                             scale=1000, flip_sin_to_cos=True)
vec = Linear(256,1792) -> SiLU -> Linear(1792,1792)
```

The pipeline normalization and embedding scale cancel for sampled configs, so
the sinusoidal embedding effectively sees the scheduler timestep while the
model API receives normalized `t_cont`.

Image RoPE:

```text
ids = [row, col] for each latent patch
for each axis:
  omega = 1 / theta ** (arange(0, axis_dim, 2) / axis_dim)
  angles = pos * omega
  build [[cos,-sin],[sin,cos]]
concat axes -> [B,Limg,head_dim/2,2,2]
```

Custom math to reproduce:

- `apply_rope` reshapes head dim into pairs and applies the 2x2 rotation matrix;
  it computes in fp32 and casts back to query dtype.
- Adaptive residual gates multiply attention and MLP outputs before residual
  addition.
- MLP activation is `GELU(approximate="tanh")` on `gate_proj(x)`, multiplied by
  `up_proj(x)`.
- CFG arithmetic and FlowMatch Euler update are small elementwise kernels and
  can be fused later.

Precompute candidates:

- `image_ids` and `image_rotary_emb` for each latent resolution bucket.
- Scheduler sigma/timestep table for fixed step schedules.
- Prompt embeddings and masks for repeated prompts.

## 10. Preprocessing and input packing

CPU/data-pipeline work:

- `TextPreprocessor.clean_text` lowercases, URL-strips, removes CJK ranges,
  normalizes punctuation/dashes, applies `ftfy.fix_text` when available, and
  removes spam-like patterns.
- Tokenizer uses `padding="max_length"`, `max_length=tokenizer.model_max_length`
  (256), truncation, and returns attention masks.
- Negative and positive prompts are encoded together when CFG is active, then
  split back into uncond/cond embeddings and masks.
- `num_images_per_prompt` repeats embeddings and masks.

GPU/runtime work:

- Latents are sampled as NCHW Gaussian tensors with dtype matching text
  embeddings.
- No pipeline-level latent packing; transformer handles patchify internally.
- CFG concatenates latents and text conditions over batch.
- Output image resize/crop after VAE decode is tensor bilinear interpolation and
  slicing before final postprocess.

Layout candidates:

- Preserve NCHW at pipeline, scheduler, transformer boundary, and VAE boundary.
- A local token-layout region starts after `img2seq` and ends before `seq2img`;
  this region is GEMM/attention dominated and does not need NHWC.
- Channel-last may be useful inside VAE Conv2d/GroupNorm/upsample regions, but
  it needs an explicit no-layout-translation guard at the transformer boundary
  unless patchify/unpatchify and VAE contracts are rewritten together.

## 11. Graph rewrite / lowering opportunities

Patchify as strided view plus GEMM:

```text
source pattern:
  reshape [B,C,H/p,p,W/p,p] -> einsum nchpwq->nhwcpq -> reshape [B,L,C*p*p] -> Linear(C*p*p,Hid)
replacement:
  im2col/patch-view lowering feeding GEMM(64,1792)
preconditions:
  NCHW contiguous input; H and W divisible by p; p=2 for sampled configs.
shape:
  L=(H/p)*(W/p), K=C*p*p=64.
weight transform:
  none if patch feature order matches [C,p,q] from source einsum.
failure cases:
  non-contiguous latents, odd latent H/W, alternate patch order.
test:
  random NCHW tensor, compare img2seq + img_in against lowered patch GEMM.
```

Unpatchify as inverse patch scatter/view:

```text
source pattern:
  Linear(Hid,64) -> reshape [B,H/p,W/p,C,p,p] -> einsum nhwcpq->nchpwq -> reshape [B,C,H,W]
replacement:
  GEMM output followed by deterministic inverse patch layout kernel.
preconditions:
  same p=2 and feature order as patchify.
failure cases:
  changed final projection width, non-divisible H/W.
test:
  seq2img(img2seq(x)) == x for representative bins.
```

PRX attention canonicalization:

```text
source pattern:
  img_qkv linear, text_kv linear, RMSNorm Q/K, image RoPE, cat K/V, dense mask,
  attention, output linear
replacement:
  canonical cross_attention(query=image, kv=[text,image]) with explicit
  source ranges and mask.
preconditions:
  non-causal; same head count for text/image KV; head dim 64; text mask either
  dense bool or converted to varlen metadata.
failure cases:
  backend cannot handle dense mask, text length changes without mask update,
  RoPE accidentally applied to text K.
test:
  block-level parity with padding and no-padding masks.
```

Adaptive LayerNorm/gated residual fusion:

```text
source pattern:
  mod = Linear(SiLU(temb)); chunk scale/shift/gate; LayerNorm; affine; residual
replacement:
  fused AdaLN + gate kernel around attention/MLP outputs.
preconditions:
  hidden size 1792, affine=False LayerNorm eps 1e-6, broadcast [B,1,H].
failure cases:
  affine LayerNorm variant or changed modulation chunk order.
test:
  PRXBlock parity at fp32 and fp16 tolerances.
```

FlowMatch Euler plus CFG fusion:

```text
source pattern:
  chunk noise_pred; uncond + w*(text-uncond); sample + dt*pred
replacement:
  one elementwise kernel for CFG and scheduler update.
preconditions:
  deterministic FlowMatch Euler; no stochastic sampling; no per-token timesteps.
shape:
  [B,16,H/8,W/8].
failure cases:
  no-CFG path, stochastic scheduler, custom per-token timesteps.
test:
  one scheduler step parity across CFG scales 1 and >1.
```

VAE decode scale/shift sink:

```text
source pattern:
  latents / scaling_factor + shift_factor -> VAE decode first conv
replacement:
  fold scale/shift into first Conv2d bias/weights when decode-only weights are
  fixed.
preconditions:
  VAE first op is affine Conv2d, constants fixed, no caller needs intermediate
  unscaled latents.
failure cases:
  dynamic VAE weights, AutoencoderDC variant, decode slicing/tiling not audited.
test:
  VAE decode parity before and after folded first conv.
```

## 12. Kernel fusion candidates

Highest priority:

- PRX attention QKV/KV projection, QK RMSNorm, RoPE, attention, output
  projection. This dominates at 1024 where image query length is 4096 and KV
  length is 4352.
- AdaLayerNorm/gated residual around attention and MLP. Every block uses the
  same modulation pattern.
- Gated GELU MLP: `GELU_tanh(gate_proj(x)) * up_proj(x) -> down_proj`.
- Patchify/unpatchify plus adjacent linears, especially for avoiding materialized
  `[B,L,64]` copies.

Medium priority:

- CFG plus FlowMatch Euler update elementwise fusion.
- Timestep embedding MLP and modulation precompute per step.
- Text projection `Linear(2304 -> 1792)` cache for fixed prompt embeddings
  across denoising steps.
- VAE decode Conv2d/GroupNorm/SiLU/upsample fusions, likely shared with Flux and
  SD3 16-channel VAE reports.

Lower priority:

- TextPreprocessor/tokenizer runtime acceleration; keep CPU-side initially.
- Full T5-Gemma encoder compilation; useful later, but denoiser parity can
  start with cached embeddings.
- Output resize/crop fusion after VAE decode.

## 13. Runtime staging plan

Stage 1: config and tensor-contract loader.

- Parse top-level, transformer, scheduler, tokenizer, text encoder, and VAE
  configs for `Photoroom/prx-512-t2i`.
- Accept external `prompt_embeds`, `negative_prompt_embeds`, and masks.
- Reject image sizes not divisible by 16 for sampled PRX transformer configs.

Stage 2: single PRX denoiser block parity.

- Implement patchify/unpatchify, timestep embedding, RoPE, PRX attention,
  adaptive norms, and gated MLP for one block.
- Validate fp32 with random tensors and padded masks.

Stage 3: full transformer denoiser.

- Load transformer weights.
- Run one denoiser forward at 512 native latent shape with external embeddings.
- Add 1024 token-length stress test after 512 is stable.

Stage 4: one denoising step.

- Add FlowMatch Euler static-shift scheduler tables and deterministic step.
- Validate CFG doubled-batch arithmetic and one-step parity.

Stage 5: Python-controlled denoising loop.

- Keep scheduler loop host-visible.
- Compile denoiser call; keep tokenizer/text encoder and VAE decode as adjacent
  stages or PyTorch fallbacks.

Stage 6: VAE decode.

- Reuse Flux/SD3 VAE decode work for 16-channel `AutoencoderKL` scale/shift.
- Add binned output resize/crop parity.

Stage 7: optimization.

- Add attention backend selection, patchify/linear fusion, AdaLN fusion, MLP
  fusion, CFG+scheduler elementwise fusion, and optional T5-Gemma encoder
  compilation.

## 14. Parity and validation plan

- `img2seq`/`seq2img` inverse tests for 512, 1024, and non-square bins.
- RoPE construction and `apply_rope` random tensor parity in fp32/fp16.
- PRX attention parity with all-ones mask and padded text mask.
- PRXBlock parity at fixed random `temb`, text tokens, image tokens, and mask.
- Full transformer parity for one step at `[1,16,64,64]` and `[1,16,128,128]`.
- Scheduler table parity for 28 steps and a custom `timesteps` list.
- CFG parity for `guidance_scale=1.0`, `4.0`, and a higher value.
- VAE decode parity for `(latents / 0.3611) + 0.1159`.
- Resolution binning parity for a non-square request, including final
  resize/crop.
- End-to-end smoke: fixed generator, 2-4 denoising steps, compare latents and
  decoded image tensor against Diffusers.

Suggested tolerances:

- fp32 unit tests: `rtol=1e-4`, `atol=1e-5` for block-level math; looser for
  full loop due to attention/scheduler accumulation.
- fp16/bf16: use component-level tolerances around `rtol=5e-2`, `atol=5e-2`
  first, tighten per kernel after backend choice is fixed.

## 15. Performance probes

- T5-Gemma encoder throughput for batch 1/2 and max length 256.
- PRX transformer one-step latency at 512 and 1024 native square shapes.
- Attention time split by query length 1024 vs 4096 and padded vs all-ones mask.
- GEMM time for projection/MLP families: 1792x5376, 1792x3584, 1792x6272, and
  6272x1792.
- Patchify/unpatchify materialization bandwidth.
- Full denoising loop by step count, especially default 28.
- VAE decode throughput for 512 and 1024.
- CFG+scheduler overhead as a separate elementwise probe.
- VRAM and temporary workspace usage for 1024 CFG doubled batch.
- Attention backend comparison: native SDPA/eager parity path, no-mask flash
  candidate, and padded-mask varlen candidate if implemented.

## 16. Scope boundary and separate candidates

Separate candidate reports related to PRX:

- `prx_lora_adapters`: `PRXPipeline` inherits `LoraLoaderMixin`; audit runtime
  adapter mutation and weight merging separately from base denoiser ops.
- `prx_textual_inversion`: `TextualInversionLoaderMixin` can mutate tokenizer
  and embeddings; treat as a tokenizer/text-encoder integration report.
- `prx_single_file_loading`: `FromSingleFileMixin` loading path if single-file
  PRX checkpoints matter.
- `prx_autoencoderdc_codec`: source accepts `AutoencoderDC`, but sampled
  official configs use `AutoencoderKL`.
- `prx_t5gemma_encoder`: compile the 26-layer T5-Gemma encoder after denoiser
  parity; it has alternating sliding/full attention and GQA.
- Future `prx_img2img` / `prx_inpaint` only if corresponding non-deprecated
  pipeline files or official configs appear.

Genuinely ignored/out of scope for this audit:

- XLA, NPU, MPS, Flax, ONNX.
- Safety checker and NSFW filtering.
- Training, losses, dropout behavior, and gradient checkpointing.
- Callback mutation and interactive interrupt behavior.
- Multi-GPU/context parallel attention paths.
- Unofficial third-party PRX forks not represented in the inspected Diffusers
  checkout or official Photoroom configs.

## 17. Final implementation checklist

- [ ] Parse PRX top-level and component configs.
- [ ] Add PRX latent-size guard: image height/width divisible by `vae_scale * patch_size`.
- [ ] Load PRX transformer weights.
- [ ] Implement `img2seq` / `seq2img` patch layout parity.
- [ ] Implement timestep embedding with PRX scale/period defaults.
- [ ] Implement 2D RoPE and `apply_rope`.
- [ ] Implement PRX attention with image Q and text+image K/V.
- [ ] Support dense bool PRX attention masks.
- [ ] Implement PRX adaptive LayerNorm modulation and gated residuals.
- [ ] Implement PRX gated GELU MLP.
- [ ] Implement full 16-block `PRXTransformer2DModel`.
- [ ] Implement FlowMatch Euler static-shift scheduler slice.
- [ ] Implement CFG doubled-batch arithmetic.
- [ ] Add one-step denoiser+scheduler parity.
- [ ] Add Python-controlled short denoising-loop parity.
- [ ] Integrate Flux-style `AutoencoderKL` decode scale/shift.
- [ ] Add VAE decode and postprocess parity.
- [ ] Benchmark 512 and 1024 denoiser steps.
- [ ] Add attention backend feasibility probes for padded and unpadded prompts.
