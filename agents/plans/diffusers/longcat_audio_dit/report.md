# LongCat Audio DiT Diffusers Audit

Candidate slug: `longcat_audio_dit`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  ruixiangma/LongCat-AudioDiT-1B-Diffusers
  meituan-longcat/LongCat-AudioDiT-1B, original non-Diffusers/reference config
  meituan-longcat/LongCat-AudioDiT-3.5B, original non-Diffusers wider variant
  drbaph/LongCat-AudioDiT-3.5B-bf16, mirror/format variant matching the 3.5B config shape

Config sources:
  H:/configs/ruixiangma/LongCat-AudioDiT-1B-Diffusers/model_index.json
  H:/configs/ruixiangma/LongCat-AudioDiT-1B-Diffusers/transformer/config.json
  H:/configs/ruixiangma/LongCat-AudioDiT-1B-Diffusers/vae/config.json
  H:/configs/ruixiangma/LongCat-AudioDiT-1B-Diffusers/text_encoder/config.json
  H:/configs/ruixiangma/LongCat-AudioDiT-1B-Diffusers/tokenizer/tokenizer_config.json
  H:/configs/meituan-longcat/LongCat-AudioDiT-1B/config.json
  H:/configs/meituan-longcat/LongCat-AudioDiT-3.5B/config.json
  H:/configs/drbaph/LongCat-AudioDiT-3.5B-bf16/config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/longcat_audio_dit/pipeline_longcat_audio_dit.py
  X:/H/diffusers/src/diffusers/pipelines/longcat_audio_dit/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/transformer_longcat_audio_dit.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_longcat_audio_dit.py
  X:/H/diffusers/src/diffusers/models/normalization.py for RMSNorm

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/utils/constants.py for default attention backend

External component configs inspected:
  UMT5EncoderModel/T5 tokenizer configs packaged in the Diffusers repo above.
  The original Meituan config names google/umt5-base.

Any missing files or assumptions:
  The public Diffusers conversion has no scheduler/scheduler_config.json file in
  the repo listing. The model_index names FlowMatchEulerDiscreteScheduler, and
  the pipeline constructor falls back to FlowMatchEulerDiscreteScheduler
  shift=1.0, invert_sigmas=True when the loaded scheduler is absent or not the
  expected class. No gated official config blocker remained: authenticated
  huggingface_hub/HF access as user `hlky` could read the public repos. The
  only unavailable path was the non-existent
  meituan-longcat/LongCat-AudioDiT-1B-Diffusers repo, which returned 404.
```

Primary source anchors:

- `LongCatAudioDiTPipeline.__init__`: `pipeline_longcat_audio_dit.py:99`.
- Pipeline `encode_prompt`: `pipeline_longcat_audio_dit.py:136`.
- Pipeline `prepare_latents`: `pipeline_longcat_audio_dit.py:164`.
- Pipeline `__call__`: `pipeline_longcat_audio_dit.py:221`.
- Transformer attention processors: `transformer_longcat_audio_dit.py:184` and `:284`.
- `AudioDiTBlock`: `transformer_longcat_audio_dit.py:349`.
- `LongCatAudioDiTTransformer`: `transformer_longcat_audio_dit.py:455`.
- VAE residual/codec blocks: `autoencoder_longcat_audio_dit.py:107`, `:124`, `:156`.
- `LongCatAudioDiTVae.encode` / `decode`: `autoencoder_longcat_audio_dit.py:348` and `:374`.

## 2. Pipeline and component graph

`LongCatAudioDiTPipeline` registers `vae: LongCatAudioDiTVae`,
`text_encoder: UMT5EncoderModel`, `tokenizer: PreTrainedTokenizerBase`,
`transformer: LongCatAudioDiTTransformer`, and
`scheduler: FlowMatchEulerDiscreteScheduler`. CPU offload order is
`text_encoder->transformer->vae`.

```text
prompt text
  -> normalize text, T5/UMT5 tokenizer, UMT5 encoder
  -> layer_norm(last_hidden_state) + layer_norm(first_hidden_state)
  -> latent duration choice and latent noise [B, T_latent, 64]
  -> denoising loop:
       LongCatAudioDiTTransformer(latents, prompt embeds, masks, timestep, zero latent_cond)
       optional true CFG second transformer call with negative/zero prompt embeds
       FlowMatch Euler scheduler step
  -> VAE decode from latents.permute(0, 2, 1)
  -> waveform [B, 1, T_latent * 2048]
```

Required first-slice components are prompt embeddings or an external prompt
embedding ABI, rank-3 audio latents, masks, the LongCat audio DiT transformer,
FlowMatch Euler custom-sigma loop, and VAE decode. Cacheable stages are
normalized prompt embeddings and lengths, text masks, duration masks, scheduler
timesteps/sigmas, RoPE tables for latent and prompt sequence lengths, and VAE
decode shape buckets.

Separate candidate reports:

- `longcat_audio_dit_codec`: `LongCatAudioDiTVae` encode/decode is a reusable
  Conv1d/ConvTranspose1d/Snake/weight_norm waveform codec.
- `longcat_audio_dit_text_encoder`: UMT5 encoder plus tokenizer cache ABI. The
  DiT can be staged with external prompt embeddings first.
- `longcat_audio_dit_3_5b`: original Meituan 3.5B config changes DiT width,
  depth, MLP ratio, and max duration but is not a Diffusers pipeline repo in
  the inspected cache.
- `longcat_image`: a separate pipeline/model folder exists under
  `pipelines/longcat_image` and `models/transformers/transformer_longcat_image.py`.
- LoRA/textual inversion/runtime adapter mutation: no LongCat Audio pipeline
  loader mixins were found. Generic attention backend selection exists, but no
  base runtime adapter contract is exposed by this pipeline.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img,
  upscaling: not present in the `longcat_audio_dit` folder and not part of the
  base audio pipeline.

## 3. Important config dimensions

Representative config sweep:

| Repo/config | Pipeline shape | DiT | Text encoder | Codec | Scheduler evidence |
| --- | --- | --- | --- | --- | --- |
| `ruixiangma/LongCat-AudioDiT-1B-Diffusers` | Diffusers pipeline | 1536 dim, 24 layers, 24 heads | UMT5 base, 768 dim | LongCat VAE, 24 kHz, hop 2048 | model_index names FlowMatch; scheduler config absent |
| `meituan-longcat/LongCat-AudioDiT-1B` | original config, not Diffusers model_index | same 1B shape | google/umt5-base | same | source/reference config only |
| `meituan-longcat/LongCat-AudioDiT-3.5B` | original config, not Diffusers model_index | 2560 dim, 32 layers, 32 heads | google/umt5-base | same | source/reference config only |
| `drbaph/LongCat-AudioDiT-3.5B-bf16` | mirror/format variant | matches 3.5B shape | google/umt5-base | same | source/reference config only |

Transformer dimensions:

| Field | 1B Diffusers | 3.5B original | Source |
| --- | ---: | ---: | --- |
| `latent_dim` | 64 | 64 | component/reference config |
| `dit_dim` | 1536 | 2560 | component/reference config |
| `dit_depth` | 24 | 32 | component/reference config |
| `dit_heads` | 24 | 32 | component/reference config |
| head dim | 64 | 80 | inferred `dit_dim / dit_heads` |
| `dit_text_dim` | 768 | 768 | component/reference config |
| `ff_mult` | source default 4.0, omitted in Diffusers config | 3.6 | source default/reference config |
| `qk_norm` | true | true | config |
| `text_conv` | true | true | config |
| `cross_attn` | true | true | config |
| `use_latent_condition` | true | true | config |
| `adaln_type` | global | global | config |

Audio codec dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| waveform channels | 1 | VAE config `in_channels` |
| sample rate | 24000 Hz | VAE config |
| latent hop / downsampling ratio | 2048 samples | VAE config |
| latent frame rate | 11.71875 Hz | inferred `24000 / 2048` |
| max duration in Diffusers pipeline | 30 s | source default |
| 1B max duration in original config | 30 s | reference config |
| 3.5B max duration in original config | 60 s | reference config |
| max latent frames at 30 s | 351 | inferred floor `30 * 24000 / 2048` |
| VAE base channels | 128 | VAE config |
| VAE channel multipliers | `[1,2,4,8,16]` | VAE config |
| VAE strides | `[2,4,4,8,8]` | VAE config |
| encoder latent parameter channels | 128 | VAE config, split into mean/std for 64 latents |
| latent scale | 0.71 | VAE config |
| activation | Snake1d | `act_fn=null`, `use_snake=true` |

Text and tokenizer dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| text encoder class | `UMT5EncoderModel` | model_index/text config |
| `d_model` | 768 | text config |
| `d_ff` | 2048 | text config |
| layers / heads | 12 / 12 | text config |
| `d_kv` | 64 | text config |
| vocab size | 256384 | text config |
| relative attention buckets / max distance | 32 / 128 | text config |
| tokenizer class | T5Tokenizer/T5TokenizerFast compatible | model_index/tokenizer config |
| tokenizer model max length | huge sentinel in config, clamped to 512 by pipeline | tokenizer config and source |

Scheduler:

| Field | Value |
| --- | --- |
| model_index class | `FlowMatchEulerDiscreteScheduler` |
| component scheduler config | absent in the public Diffusers repo |
| pipeline fallback | `FlowMatchEulerDiscreteScheduler(shift=1.0, invert_sigmas=True)` |
| pipeline custom sigmas | `linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)` |
| source default steps / guidance | 16 / 4.0 |
| first Dinoml scheduler slice | FlowMatch Euler custom sigmas, shift 1, invert_sigmas true, non-stochastic |

Weight metadata from HF file metadata:

| Repo | File | Size |
| --- | --- | ---: |
| `ruixiangma/LongCat-AudioDiT-1B-Diffusers` | `text_encoder/model.safetensors` | 1.13 GB |
| same | `transformer/diffusion_pytorch_model.safetensors` | 3.93 GB |
| same | `vae/diffusion_pytorch_model.safetensors` | 0.62 GB |
| `meituan-longcat/LongCat-AudioDiT-1B` | `model.safetensors` | 5.68 GB |
| `meituan-longcat/LongCat-AudioDiT-3.5B` | `model.safetensors` | 15.34 GB |

## 3a. Family variation traps

- The selected Diffusers target has no scheduler config file. Artifact loading
  must mirror the constructor fallback or explicitly materialize a scheduler
  config from source defaults.
- The public Diffusers config omits `ff_mult`; the current class default is
  4.0. Original 3.5B config uses 3.6, so DiT MLP width is config-sensitive.
- Tokenizer `model_max_length` is an enormous sentinel; the pipeline clamps it
  to 512. Do not build prompt-cache shapes from the raw tokenizer field alone.
- Latents are `[B, T_latent, 64]` token/audio layout inside the DiT. The VAE
  boundary is `[B, 64, T_latent]` NCL. This is neither image NCHW nor Stable
  Audio's NCL denoiser layout.
- `audio_duration_s` is converted by floor division to latent frames. Decode
  returns exactly `T_latent * 2048` samples, so requested waveform duration can
  be rounded down by up to 2047 samples.
- The pipeline always builds `latent_cond` as zeros, but the transformer config
  enables latent conditioning. This creates extra latent embedding work even in
  the base text-to-audio path.
- True CFG is two separate transformer calls, not batch concatenation.
- Attention masks are boolean length masks. They are passed into the dispatch
  attention path and also applied as post-attention/output masks.
- RoPE is applied to both latent-token self-attention and prompt-key
  cross-attention. Prompt length therefore affects a separate RoPE table.
- The text-conditioning path includes four ConvNeXtV2-style depthwise Conv1d
  blocks over prompt tokens before cross-attention. Prompt masks are reapplied
  after this text conv stack.
- VAE weight-normalized Conv1d/ConvTranspose1d should be treated as a load-time
  weight materialization candidate for frozen inference weights.
- The 3.5B original config doubles the practical sequence budget through
  `max_wav_duration=60` and changes attention head dim from 64 to 80, which is
  important for flash-provider admission.

## 4. Runtime tensor contract

Pipeline inputs:

- `prompt: str | list[str]`, normalized to lowercase with quote replacement and
  whitespace squashing.
- Optional `negative_prompt` for CFG. When omitted, negative prompt embeddings
  are zero tensors matching the positive prompt shape.
- Optional `audio_duration_s`. Ignored if caller supplies `latents`.
- Optional `latents [B, T_latent, 64]`.
- `num_inference_steps`, `guidance_scale`, and generator.
- Outputs are `AudioPipelineOutput(audios=...)` containing either latents
  `[B,T,64]` for `output_type="latent"` or waveform `[B,1,T*2048]` as torch or
  NumPy.

Prompt boundary:

- Tokenizer output: `input_ids [B,L]`, `attention_mask [B,L]`, where source
  clamps `L <= 512`.
- UMT5 output: `last_hidden_state [B,L,768]` and `hidden_states[0] [B,L,768]`.
- Pipeline prompt embeds:
  `layer_norm(last_hidden_state) + layer_norm(first_hidden_state)`, shape
  `[B,L,768]`.
- Prompt lengths: `attention_mask.sum(dim=1)`.
- Text mask: `_lens_to_mask(prompt_lengths, length=L)`, shape `[B,L]`.

Latent and mask boundary:

- `T_latent = floor(audio_duration_s * 24000 / 2048)` when duration is given,
  clamped to `[1, 351]` if latents are not supplied.
- If duration is omitted, source estimates seconds from prompt characters with
  `0.082` seconds per English/other char or `0.21` per Chinese/other char,
  capped at 30 seconds before conversion to latent frames.
- Latents: `[B,T_latent,64]`, dtype follows prompt embeddings.
- Attention mask: `[B,T_latent]`, all true for base generation because every
  request in the batch uses the shared chosen duration.
- Latent condition: zero tensor `[B,T_latent,64]`.

One denoising step:

- Current scheduler timestep `t` is divided by
  `scheduler.config.num_train_timesteps` and expanded to `[B]`.
- Positive transformer call:
  `hidden_states [B,T,64]`, `encoder_hidden_states [B,L,768]`,
  `encoder_attention_mask [B,L]`, `timestep [B]`, `attention_mask [B,T]`,
  `latent_cond [B,T,64]`.
- Optional negative transformer call uses zero or separately encoded negative
  prompt embeddings with the same latent/mask inputs.
- CFG: `null_pred + (pred - null_pred) * guidance_scale`.
- Scheduler update: FlowMatch Euler `prev_sample = sample + dt * model_output`
  in the non-stochastic branch.

Codec boundary:

- Decode input from pipeline: `latents.permute(0,2,1)`, shape `[B,64,T]`.
- VAE decode multiplies by scale `0.71`, casts to decoder dtype, runs decoder,
  and casts decoded output back to fp32 when decoder weights are not fp32.
- Decode output: `[B,1,T*2048]` for the configured even strides.
- Encode, not used by the primary text-to-audio `__call__`, consumes waveform
  `[B,1,S]`, produces encoder params `[B,128,floor(S/2048)]`, splits to mean
  and softplus std `[B,64,T]`, optionally samples, then divides by `0.71`.

CPU/data-pipeline work includes string normalization, tokenization, duration
estimation, NumPy conversion, and callback mutation. GPU/runtime work includes
UMT5 if admitted, DiT, scheduler/CFG arithmetic, and VAE decode.

## 5. Operator coverage checklist

Tensor/layout ops:

- Rank-3 layouts `[B,T,C]` and `[B,C,T]`, `permute(0,2,1)`, transpose around
  Conv1d text blocks, `cat`, `chunk`, `view`, `reshape`, `repeat_interleave`,
  `masked_fill`, boolean masks, `sum`, `mean`, `clamp`, `norm`, `where`-like
  mask application, crop/slice if encode is admitted.

Convolution/audio ops:

- DiT text ConvNeXtV2 stack: four depthwise
  `Conv1d(1536 -> 1536, groups=1536, kernel=7, padding=3)` blocks over prompt
  tokens, each with LayerNorm, Linear
  `1536 -> 3072`, SiLU, GRN, Linear `3072 -> 1536`, residual.
- VAE encoder initial `Conv1d(1 -> 128, kernel=7, padding=3)`.
- VAE encoder blocks with residual Conv1d kernel 7 dilations 1/3/9, 1x1 Conv1d,
  strided weight-norm Conv1d kernels `[4,8,8,16,16]`, strides `[2,4,4,8,8]`.
- VAE decoder mirrors with weight-norm ConvTranspose1d kernels
  `[16,16,8,8,4]`, strides `[8,8,4,4,2]`, residual Conv1d units, final
  `Conv1d(128 -> 1, kernel=7, bias=false)`.
- DownsampleShortcut average and UpsampleShortcut duplicate/pixel-shuffle 1D.
- Weight normalization on all VAE Conv1d/ConvTranspose1d modules.

GEMM/linear ops:

- UMT5 encoder if included: relative-bias encoder transformer with gated-GELU
  feed-forward and 12 heads.
- DiT timestep MLP: sinusoidal 256 -> Linear `256 -> dim` -> SiLU -> Linear
  `dim -> dim`.
- Latent embedder: Linear `64 -> dim` -> SiLU -> Linear `dim -> dim`.
- Text embedder: Linear `768 -> dim` -> SiLU -> Linear `dim -> dim`.
- Latent condition embedder: Linear `2*dim -> dim` -> SiLU -> Linear
  `dim -> dim`.
- Per DiT block: Q/K/V/out Linear `dim -> dim` for self-attn and cross-attn,
  FFN Linear `dim -> int(dim*ff_mult)` -> GELU tanh -> Linear back to dim.
- Global AdaLN MLP: SiLU -> Linear `dim -> 6*dim`; final AdaLN Linear
  `dim -> 2*dim`; output Linear `dim -> 64`.

Attention primitives:

- Self-attention over latent tokens `T <= 351` for 30 s 1B Diffusers default,
  or `T <= 703` for 60 s 3.5B original config.
- Cross-attention from latent tokens to prompt tokens `L <= 512`.
- 24 heads, head dim 64 for 1B; 32 heads, head dim 80 for 3.5B.
- Q/K RMSNorm, RoPE on latent Q/K and prompt K, dense non-causal masks.

Normalization and adaptive conditioning:

- LayerNorm on UMT5/text conv/DiT hidden states, RMSNorm for Q/K, AdaLN scale
  shift gate for self-attention and FFN, final AdaLN, GRN in text ConvNeXtV2,
  Snake1d in VAE.

Position/timestep/custom math:

- Sinusoidal timestep embedding with scale 1000.
- 1D RoPE with base 100000 and cached max position at least 2048.
- FlowMatch custom sigma list and inverted sigma schedule.
- CFG pointwise arithmetic.

## 6. Denoiser/model breakdown

`LongCatAudioDiTTransformer.forward`:

```text
hidden_states [B,T,64]
encoder_hidden_states [B,L,768]
timestep [B]
attention_mask [B,T]
encoder_attention_mask [B,L]
latent_cond [B,T,64]

timestep -> sinusoidal 256 -> MLP -> timestep_embed [B,D]
text_embed(encoder_hidden_states, text_mask) -> [B,L,D]
optional 4x text ConvNeXtV2 blocks -> [B,L,D], then mask zeros
input_embed(hidden_states, mask) -> [B,T,D]
latent_embed(latent_cond, mask), concat with input embedding -> latent_cond_embedder -> [B,T,D]
residual clone if long_skip
RoPE tables for T latent tokens and L prompt tokens
global AdaLN condition = timestep_embed + masked text mean
24 or 32 AudioDiTBlock layers
optional long skip add
final AdaLN with global condition
Linear D -> 64
mask output
```

`AudioDiTBlock` with `adaln_type="global"`:

```text
adaln_global_out + learned per-block scale_shift -> gate/scale/shift chunks
LayerNorm(hidden) -> scale/shift -> self-attn(QK RMSNorm + latent RoPE + mask)
gated residual
optional cross-attn(LayerNorm identity for sampled config, Q latent RoPE, K prompt RoPE, cond mask)
LayerNorm(hidden) -> scale/shift -> GELU-tanh FFN
gated residual
```

The sampled 1B Diffusers config enables cross-attention, text conv, qk_norm,
long skip, global AdaLN, text-conditioned global AdaLN, and latent conditioning.
`cross_attn_norm` is false, so the cross-attn normalizers are identities.

## 7. Attention requirements

The primary implementation is local to
`transformer_longcat_audio_dit.py`: `AudioDiTSelfAttnProcessor` and
`AudioDiTCrossAttnProcessor` call `dispatch_attention_fn` from
`attention_dispatch.py`. The default dispatch backend is the environment-backed
`DIFFUSERS_ATTN_BACKEND`, whose source default is `"native"`.

Self-attention:

- Q/K/V from `[B,T,D]`.
- Shape before dispatch: `[B,T,heads,head_dim]`.
- Q and K are RMSNormed when `qk_norm=true`.
- 1D RoPE is applied to Q and K.
- `attention_mask [B,T]` is passed as `attn_mask` and also used to zero masked
  output positions.
- Non-causal, no dropout in sampled config.

Cross-attention:

- Q from latent tokens `[B,T,D]`.
- K/V from text tokens `[B,L,D]` after text embedder/conv.
- Q gets latent RoPE; K gets prompt RoPE.
- `cond_mask [B,L]` is passed as `attn_mask`; latent mask is applied as
  `post_attention_mask`.
- Query/key lengths differ; there is no GQA in the LongCat Audio DiT class
  because K/V project to the same number of heads as Q.

Flash feasibility:

- Native/eager parity is `dispatch_attention_fn` with dense non-causal masks.
- Flash-style Dinoml support is feasible for first-slice no-padding or
  all-true masks, with Q/K RMSNorm and RoPE pre-applied before attention.
- Masked prompt batches need either an attention provider that accepts key masks
  or a varlen/packed path. Standard no-mask flash kernels are insufficient for
  arbitrary prompt padding unless first-slice admission requires prompt batches
  with the same valid length or falls back to native attention.
- Head dim 64 for 1B is friendly to common flash kernels; head dim 80 for 3.5B
  must be admitted separately.
- Source does not provide fused QKV projections for this class. Q, K, and V are
  separate Linear modules, so fusion is a Dinoml lowering opportunity rather
  than a required source behavior.

## 8. Scheduler and denoising-loop contract

The pipeline uses FlowMatch Euler with custom sigmas:

```text
sigmas = linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
scheduler.set_timesteps(sigmas=sigmas, device=device)
scheduler.set_begin_index(0)
for t in scheduler.timesteps:
  curr_t = (t / num_train_timesteps).expand(B).to(prompt_dtype)
  pred = transformer(latents, prompt_embeds, masks, curr_t, latent_cond=zeros)
  if guidance_scale > 1:
    null_pred = transformer(latents, negative_prompt_embeds, masks, curr_t, latent_cond=zeros)
    pred = null_pred + (pred - null_pred) * guidance_scale
  latents = scheduler.step(pred, t, latents).prev_sample
```

With the constructor fallback `invert_sigmas=True`, `set_timesteps` first
creates shifted sigmas, then inverts them and appends terminal sigma 1.0. For
the source custom sigma list and shift 1.0, the effective scheduler sigmas are
`[0, ..., 1 - 1/N, 1]`, and timesteps are `sigmas * 1000`. The non-stochastic
FlowMatch step computes `sample + (sigma_next - sigma) * model_output`.

Initial Dinoml staging should keep `set_timesteps`, scheduler step index, and
the exact inverted custom sigma table as explicit host/runtime state. The
compiled kernel candidate is the pointwise FlowMatch update plus optional CFG,
but only after the artifact records the scheduler family, `invert_sigmas`,
custom-sigma policy, and no stochastic sampling.

## 9. Position, timestep, and custom math

Custom math to preserve:

```text
duration frames = floor(audio_duration_s * sample_rate / downsampling_ratio)
prompt duration estimate = max(chars_zh * 0.21, chars_en * 0.082) capped at 30 s
timestep embedding = sin/cos(scale * timestep * exp(-log(10000) * i / (half_dim - 1)))
RoPE base = 100000, rotate_half convention
Snake1d(x) = x + 1 / (exp(beta) + 1e-9) * sin(exp(alpha) * x)^2
VAE encode std = softplus(scale_param) + 1e-4
VAE encode latents = (mean or mean + std * noise) / 0.71
VAE decode latents = latents * 0.71
GRN = gamma * (x * (||x||_2_time / mean_channel(||x||_2_time))) + beta + x
```

RoPE tables can be precomputed per latent length and prompt length. The text
mean used for global AdaLN depends on masked prompt embeddings after the text
embedding and text ConvNeXtV2 stack. Scheduler tables depend on step count and
the fallback scheduler config. Duration masks and latent shape depend on
request duration or supplied latents.

## 10. Preprocessing and input packing

Prompt preprocessing:

- Text is lowercased.
- ASCII and curly quotes are replaced by spaces.
- Repeated whitespace is collapsed.
- Tokenization uses padding `"longest"`, truncation, and effective
  `max_length=512` because the tokenizer sentinel max length is clamped.
- Positive prompt embeddings are normalized and augmented with normalized first
  hidden-state embeddings.
- If no negative prompt is supplied, unconditional embeddings are zeros with the
  same shape and mask as the positive prompt.

Latent/audio packing:

- There is no spectrogram, vocoder, or external audio processor. The VAE codec
  operates directly on mono waveform tensors.
- The denoiser layout is token-like `[B,T,64]`.
- The decode boundary transposes to VAE NCL `[B,64,T]`.
- There is no pipeline-level audio encode path in `__call__`; VAE encode is
  only exposed on the autoencoder model for separate variants or roundtrip use.

CPU/data-pipeline work includes prompt normalization/tokenization and duration
estimation. GPU/runtime work includes UMT5, DiT, scheduler/CFG, and VAE decode.

## 11. Graph rewrite / lowering opportunities

1. Weight-norm Conv1d materialization
   - Source pattern: VAE uses `torch.nn.utils.weight_norm` wrapped Conv1d and
     ConvTranspose1d.
   - Replacement: precompute normalized dense weights at load time.
   - Preconditions: inference-only, no runtime mutation of VAE weight-g/v
     parameters, PyTorch weight_norm epsilon/axis semantics matched.
   - Shape equations: same Conv1d/ConvTranspose1d shapes; only weight storage
     changes.
   - Failure cases: training, LoRA/adapters targeting VAE conv weights, or
     checkpoint parameters stored in an unexpected parametrization.
   - Parity test: one residual unit, one encoder block, one decoder block, and
     full decode before/after materialization.

2. VAE NCL Conv1d channel-last local region
   - Source pattern: codec tensors are `[B,C,T]` with Conv1d, Snake, shortcut
     reshapes, and ConvTranspose1d.
   - Replacement: guarded NTC/channel-last Conv1d provider region or explicit
     NCL provider.
   - Preconditions: region owns all channel-axis ops; Snake parameters
     `[C]` broadcast correctly; shortcut view/permute equations are rewritten;
     public boundary remains `[B,C,T]`.
   - Failure cases: exposing intermediate tensors or admitting arbitrary
     strided inputs without layout guards.
   - Parity test: decode for latent lengths 1, 58, 351, and 703.

3. Text ConvNeXtV2 depthwise-conv fusion
   - Source pattern: transpose `[B,L,D] -> [B,D,L]`, depthwise Conv1d, transpose
     back, LayerNorm, Linear, SiLU, GRN, Linear, residual.
   - Replacement: fused text-token ConvNeXt block or layout-elided NTC depthwise
     conv.
   - Preconditions: D fixed, kernel 7 padding 3 dilation 1, dense prompt token
     layout, mask reapplied after the block stack.
   - Failure cases: prompt lengths above provider limit or masked-token
     behavior folded incorrectly before convolution.
   - Parity test: text_conv stack with variable prompt lengths and masks.

4. QKV projection + RMSNorm + RoPE staging
   - Source pattern: separate Q/K/V Linear, RMSNorm on Q/K, RoPE on Q/K, dense
     attention dispatch.
   - Replacement: fused projection/norm/rope staging feeding a flash/native
     attention provider.
   - Preconditions: static heads/head_dim, no dropout, non-causal, mask support
     matched or fallback selected.
   - Failure cases: padded prompts with no varlen/key-mask support, 3.5B head
     dim 80 not supported by selected kernel.
   - Parity test: self-attn and cross-attn random tensor tests with masks.

5. CFG plus FlowMatch update fusion
   - Source pattern: optional second transformer call, CFG arithmetic, then
     FlowMatch Euler pointwise step.
   - Replacement: one pointwise kernel over `[B,T,64]` after transformer output.
   - Preconditions: non-stochastic FlowMatch, custom inverted sigma table,
     fixed guidance mode.
   - Failure cases: `guidance_scale <= 1`, future stochastic scheduler branch,
     per-token timesteps.
   - Parity test: one-step scheduler parity with fixed pred/null_pred.

## 12. Kernel fusion candidates

Highest priority:

- LongCat DiT attention with Q/K RMSNorm and RoPE. It is the denoiser core and
  has two variants: latent self-attention and prompt cross-attention.
- AdaLN + residual gating around self-attention and FFN. Every DiT layer uses
  the same scale/shift/gate structure.
- VAE Conv1d/ConvTranspose1d + Snake1d decode. This is the audio-specific
  waveform generation island.
- FlowMatch Euler + CFG pointwise fusion over `[B,T,64]`.

Medium priority:

- Text ConvNeXtV2 stack over prompt tokens, especially for long prompts.
- GELU-tanh FFN fusion in DiT blocks.
- Weight-norm materialization with constant provenance.
- UMT5 prompt embedding cache ABI, then selected UMT5 encoder blocks if needed.

Lower priority:

- VAE encode posterior sampling, because base pipeline only decodes.
- Duration text heuristic on device; keep host-side initially.
- NumPy output conversion.
- 3.5B-specific head-dim-80 flash tuning after 1B parity.

## 13. Runtime staging plan

Stage 1: parse the Diffusers 1B model_index, transformer, VAE, tokenizer, and
text encoder configs. Materialize an explicit fallback scheduler config because
the repo has no scheduler_config.json.

Stage 2: transformer one-step parity with externally supplied
`prompt_embeds [B,L,768]`, `text_mask [B,L]`, `latents [B,T,64]`,
`attention_mask [B,T]`, zero `latent_cond`, and timestep `[B]`.

Stage 3: FlowMatch custom-sigma/invert-sigma scheduler table and one-step
update parity with fixed model outputs, then short latent loop with the DiT.

Stage 4: VAE decode-only island `[B,64,T] -> [B,1,T*2048]`, no encode first.

Stage 5: integrate UMT5 prompt encoding or a prompt-embedding cache interface.

Stage 6: full text-to-audio pipeline parity with `output_type="latent"` first,
then waveform decode.

Stage 7: VAE encode roundtrip and original 3.5B config admission as separate
milestones.

First Dinoml admission recommendation: `longcat_audio_dit_1b_one_step` with
external prompt embeddings and explicit FlowMatch scheduler metadata. Add VAE
decode as the next bounded audio-codec milestone.

## 14. Parity and validation plan

- Config parsing tests for the public Diffusers 1B repo and original 1B/3.5B
  configs, including omitted `ff_mult` and absent scheduler config.
- Unit tests for `_lens_to_mask`, duration-to-latent conversion, prompt max
  length clamp, timestep sinusoidal embedding, RoPE, Snake1d, GRN, and VAE
  scale.
- Attention parity tests for self-attn `[B,T,D]` with `T=58,351` and
  cross-attn with prompt lengths 1, 128, 512.
- Single `AudioDiTBlock` parity for 1B `D=1536, heads=24, head_dim=64`, masks
  on and off.
- Full transformer one-step parity at `[1,58,64]` and `[1,351,64]`, fixed
  prompt embeddings and timestep.
- Scheduler table tests for 16 and 20 steps: effective sigmas, timesteps,
  terminal sigma, begin index, and step index.
- CFG parity with no negative prompt and with separately encoded negative
  prompt embeddings.
- VAE decode parity for latent lengths 1, 58, 351; VAE encode mean/std/sample
  parity as a separate test.
- Short deterministic denoising-loop parity using fixed generator and
  `output_type="latent"`.
- End-to-end smoke for 5 seconds with waveform decode after transformer,
  scheduler, and VAE are individually validated.

Suggested tolerances: fp32 custom math and scheduler `rtol=1e-5, atol=1e-6`;
fp16/bf16 DiT/VAE first pass `rtol=2e-2, atol=2e-2`, then tighten per provider.

## 15. Performance probes

- One DiT step by latent length: 58 frames for 5 s, 176 for 15 s, 351 for 30 s,
  and 703 for the 3.5B 60 s config.
- Attention backend comparison: native dense masks vs no-mask flash vs
  varlen/key-mask-capable flash for prompt padding.
- Text ConvNeXtV2 stack time by prompt length 32, 128, 512.
- VAE decode throughput by audio duration 5, 15, 30, 60 seconds.
- Scheduler/CFG overhead for 16, 20, and 50 steps.
- UMT5 encoder throughput and prompt-embedding cache hit rate.
- Memory probes for true CFG two-call mode, prompt length 512, and 3.5B width.
- NCL versus guarded NTC Conv1d provider performance in VAE decode.

No benchmark measurements were run in this audit.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items:

- `longcat_audio_dit_codec`: LongCat VAE encode/decode with Conv1d,
  ConvTranspose1d, Snake1d, shortcut reshapes, and weight_norm.
- `longcat_audio_dit_text_encoder`: UMT5 encoder/tokenizer cache ABI and
  optional text encoder compilation.
- `longcat_audio_dit_scheduler`: FlowMatch Euler custom sigmas plus
  `invert_sigmas=True` fallback config and no scheduler_config.json loading
  parity.
- `longcat_audio_dit_3_5b`: original 3.5B wider DiT and 60 s max duration.
- `longcat_image`: separate image generation/editing family.
- `generic_attention_backend_selection`: Diffusers `attention_backend` context
  and dispatch-provider constraints.

Ignored/out of scope for this audit:

- XLA, NPU, MPS, Flax, and ONNX paths.
- Training, losses, dropout behavior, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt behavior.
- Safety checker and NSFW filtering; not present.
- LoRA/textual inversion/runtime adapter mutation until a LongCat Audio loader
  surface or checkpoint is selected.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, and
  upscaling, which are not supported by this audio pipeline.

## 17. Final implementation checklist

- [ ] Parse LongCat Audio DiT 1B Diffusers component configs and original 1B/3.5B reference configs.
- [ ] Materialize explicit FlowMatch Euler fallback scheduler metadata for the missing scheduler_config.json case.
- [ ] Admit audio latents `[B,T,64]` and VAE codec boundary `[B,64,T]`.
- [ ] Add duration-to-latent and prompt max-length clamp parity tests.
- [ ] Implement/validate timestep sinusoidal embedding and 1D RoPE.
- [ ] Implement LongCat attention with Q/K RMSNorm, RoPE, and mask fallback.
- [ ] Add `AudioDiTBlock` and full transformer one-step parity tests.
- [ ] Implement CFG plus FlowMatch pointwise scheduler parity.
- [ ] Add VAE decode-only tests for Conv1d, ConvTranspose1d, Snake1d, shortcuts, and weight_norm materialization.
- [ ] Add prompt-embedding cache ABI before compiling UMT5.
- [ ] Add short latent-loop and waveform decode smoke tests.
- [ ] Benchmark DiT attention, text ConvNeXt stack, VAE decode, scheduler/CFG, and UMT5 cache behavior.
