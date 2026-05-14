# ACE-Step Diffusers Audit

Candidate slug: `ace_step`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  ACE-Step/Ace-Step1.5
  ACE-Step/ACE-Step-v1-3.5B as an older, separate variant shape
  ACE-Step/ACE-Step-v1-chinese-rap-LoRA as a LoRA-only variant

Config sources:
  H:/configs/ACE-Step/* contained placeholder model_index.json files with `{}` only.
  Official Hugging Face component configs were fetched through huggingface_hub
  into the user HF cache and inspected there:
    ACE-Step/Ace-Step1.5 snapshot 19671f406d603126926c1b7e2adc169acbcade22
    ACE-Step/ACE-Step-v1-3.5B snapshot 82cd0d7b6322bd28cd4e830fe675ddb6180ce36c

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/ace_step/pipeline_ace_step.py
  X:/H/diffusers/src/diffusers/pipelines/ace_step/modeling_ace_step.py
  X:/H/diffusers/src/diffusers/pipelines/ace_step/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/ace_step_transformer.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_oobleck.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
  X:/H/diffusers/src/diffusers/guiders/adaptive_projected_guidance.py
  X:/H/diffusers/src/diffusers/models/attention_dispatch.py
  X:/H/diffusers/src/diffusers/models/embeddings.py for 1D RoPE and Timesteps
  X:/H/diffusers/src/diffusers/models/normalization.py for RMSNorm

External component configs inspected:
  ACE-Step/Ace-Step1.5 Qwen3-Embedding-0.6B config/tokenizer config.
  ACE-Step/Ace-Step1.5 acestep-5Hz-lm-1.7B config as optional audio-code LM/tokenizer source.
  ACE-Step/ACE-Step-v1-3.5B UMT5 encoder/tokenizer, AutoencoderDC, vocoder, and transformer configs.

Any missing files or assumptions:
  No official model_index.json or scheduler_config.json is present in the inspected
  ACE-Step repos. Current Diffusers source constructs an AceStepPipeline from
  component classes and expects FlowMatchEulerDiscreteScheduler configured with
  num_train_timesteps=1 and shift=1.0; the pipeline then passes custom sigmas.
  No gated config blocker was encountered.
```

## 2. Pipeline and component graph

`AceStepPipeline` components are `vae: AutoencoderOobleck`, `text_encoder: PreTrainedModel`, `tokenizer: PreTrainedTokenizerFast`, `transformer: AceStepTransformer1DModel`, `condition_encoder: AceStepConditionEncoder`, `scheduler: FlowMatchEulerDiscreteScheduler`, and optional `audio_tokenizer: AceStepAudioTokenizer` plus `audio_token_detokenizer: AceStepAudioTokenDetokenizer`. CPU offload order is `text_encoder->condition_encoder->audio_tokenizer->audio_token_detokenizer->transformer->vae`.

```text
prompt + lyrics + music metadata + optional source/reference audio/audio codes
  -> Qwen3 tokenizer/text encoder and token embedding lookup
  -> ACE condition encoder: text projection + lyric encoder + timbre encoder + packed sequence
  -> source/silence/audio-code latents + chunk mask
  -> latent noise [B,T,64]
  -> denoising loop: AceStepTransformer1DModel + APG/cover blending + FlowMatch Euler
  -> AutoencoderOobleck decode [B,64,T] -> stereo waveform
  -> peak normalization and output conversion
```

Required first-slice components are the condition sequence ABI, rank-3 acoustic latents, `AceStepTransformer1DModel`, FlowMatch Euler with custom sigma input, APG disabled or explicit, and Oobleck decode. Cacheable stages are Qwen prompt hidden states, lyric token embeddings, packed condition states, silence/reference timbre latents, source latents, chunk masks, RoPE tables, and scheduler sigmas/timesteps.

Separate candidate reports:

- `ace_step_oobleck_codec`: `AutoencoderOobleck` encode/decode Conv1d/ConvTranspose1d/Snake/weight_norm codec.
- `ace_step_audio_codes`: optional `AceStepAudioTokenizer`, `_AceStepResidualFSQ`, and `AceStepAudioTokenDetokenizer` for cover conditioning from `<|audio_code_N|>` strings.
- `ace_step_audio_to_audio_tasks`: repaint, cover, extract, lego, and complete source-audio contracts; these add VAE encode, chunk-mask windows, source-latent substitution, and optional cover blending.
- `ace_step_lora`: `ACE-Step/ACE-Step-v1-chinese-rap-LoRA` ships LoRA weights only; adapter loading/mutation is separate from the base pipeline.
- `ace_step_v1_3_5b_legacy`: older `ACEStepTransformer2DModel` + `AutoencoderDC` + ADaMoS HiFi-GAN-style vocoder shape; do not merge with ACE-Step 1.5 first slice.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, inpaint/depth/upscale image variants: not present in the ACE-Step pipeline folder.

## 3. Important config dimensions

Representative config sweep:

| Repo | Source shape | Text encoder | Codec | Scheduler |
| --- | --- | --- | --- | --- |
| `ACE-Step/Ace-Step1.5` | current ACE-Step 1.5 config, turbo | Qwen3Model hidden 1024 | `AutoencoderOobleck` stereo 48 kHz | FlowMatch Euler by source, no scheduler config file |
| `ACE-Step/ACE-Step-v1-3.5B` | legacy Diffusers 0.32 config | UMT5EncoderModel hidden 768 | `AutoencoderDC` + ADaMoS vocoder | not represented in current pipeline source |
| `ACE-Step/ACE-Step-v1-chinese-rap-LoRA` | LoRA weights only | base-dependent | base-dependent | base-dependent |

ACE-Step 1.5 transformer and condition config:

| Field | Value | Source |
| --- | ---: | --- |
| `hidden_size` | 2048 | `Ace-Step1.5/config.json` |
| `intermediate_size` | 6144 | config |
| DiT layers | 24 | config |
| lyric/timbre encoder layers | 8 / 4 | config |
| attention heads / KV heads | 16 / 8 | config, GQA |
| head dim | 128 | config |
| patch size | 2 latent frames | config/source |
| `in_channels` | 192 | context 128 + noisy 64 |
| acoustic latent dim | 64 | config |
| sliding window | 128 patch tokens on alternating layers | config/source |
| layer pattern | alternating sliding/full attention | config |
| RoPE theta | 1,000,000 | config |
| dtype metadata | bfloat16 | config |
| turbo metadata | `is_turbo=true`, `model_version="turbo"` | config |

Codec and audio dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| waveform sample rate | 48000 Hz | `vae/config.json` |
| waveform channels | 2 | config |
| downsampling ratios | `[2,4,4,6,10]` | config |
| hop length / downsample | 1920 samples | product, source |
| latent frame rate | 25 fps | inferred `48000/1920` |
| 60 s latent length | 1500 frames | pipeline formula |
| 30 s timbre reference length | 750 frames | pipeline formula |
| Oobleck latent channels | 64 decode, 128 encode parameters | config/source |

Text and optional code-token dimensions:

| Component | Shape facts | Source |
| --- | --- | --- |
| Qwen3 text encoder | hidden 1024, 28 layers, 16 Q heads, 8 KV heads, bf16 | config |
| Text prompt max length | 256 | pipeline default |
| Lyrics max length | 2048 | pipeline default |
| Lyrics path | embedding lookup only, then ACE lyric encoder | source |
| Optional 5 Hz LM/tokenizer | Qwen3 hidden 2048, vocab 217204 | config |
| FSQ audio code levels | `[8,8,8,5,5,5]`, codebook 64000 | config/source |
| audio code expansion | 5 Hz tokens -> 25 Hz acoustic via pool window 5 | source |

Scheduler and guidance:

| Field | Value |
| --- | --- |
| source scheduler class | `FlowMatchEulerDiscreteScheduler` |
| required config by source comment | `num_train_timesteps=1`, `shift=1.0` |
| pipeline schedule | custom sigmas from `linspace(1,0,N+1)[:-1]`, shifted by `shift * t / (1 + (shift - 1) * t)` |
| default steps / shift | 8 / 3.0 |
| turbo guidance | CFG/APG coerced to off (`guidance_scale=1`) |
| base/SFT guidance | APG with learned null condition, momentum -0.75, norm threshold 2.5, `norm_dim=(1,)` |
| first Dinoml scheduler slice | FlowMatch Euler custom-sigmas, non-stochastic, scalar timesteps in `[0,1]` |

## 3a. Family variation traps

- Current ACE-Step source is not the older `ACE-Step-v1-3.5B` 2D/vocoder stack. First Dinoml work should target `Ace-Step1.5` and document 3.5B separately.
- Source latents are rank-3 `[B,T,64]`, while Oobleck codec boundaries are NCL `[B,64,T]`; this is not image NCHW.
- DiT patching concatenates `context_latents [B,T,128]` with noisy latents `[B,T,64]` to `[B,T,192]`, pads time to a multiple of 2, transposes to NCT, and applies `Conv1d(192 -> 2048, kernel=2, stride=2)`.
- Output uses `ConvTranspose1d(2048 -> 64, kernel=2, stride=2)` and crops back to the original latent length.
- Alternating self-attention masks differ by backend: native/eager path builds dense 4D band masks for sliding layers; flash backends use `window_size` where possible and require varlen flash for padded masks.
- Cross-attention ignores `encoder_attention_mask` in the DiT call today (`None` is passed), but lyric encoder uses masks and packing.
- Turbo checkpoints disable APG/CFG. Base/SFT checkpoints need `condition_encoder.null_condition_emb`; do not use empty negative prompts.
- Oobleck tiling changes execution boundaries for long audio; first-slice decode should either admit tiling explicitly or keep no-tiling for parity tests.
- Audio-code cover conditioning adds FSQ quantization/dequantization and token expansion, but it is optional in the pipeline constructor.

## 4. Runtime tensor contract

Pipeline inputs:

- `prompt: str | list[str]`, `lyrics: str | list[str]`, `audio_duration`, optional `vocal_language`, BPM/key/time-signature metadata.
- Optional source/reference audio tensors are `[channels, samples]` at 48 kHz; source may also be `[B, channels, samples]` after normalization.
- Optional audio code strings parse tokens like `<|audio_code_123|>`.
- Optional `latents` are `[B, latent_length, 64]`.

Condition tensors:

- Text tokenizer output: `[B,L_text<=256]`; Qwen encoder hidden states `[B,L_text,1024]`.
- Lyrics tokenizer output: `[B,L_lyric<=2048]`; lyrics hidden states are token embeddings `[B,L_lyric,1024]`.
- Text projector maps `[B,L_text,1024] -> [B,L_text,2048]`.
- Lyric encoder maps `[B,L_lyric,1024] -> [B,L_lyric,2048]` with masks.
- Timbre encoder consumes reference/silence acoustic latents `[B,750,64]` for 30 seconds and returns packed timbre tokens `[B,N_timbre,2048]`.
- `_pack_sequences` concatenates lyric+timbre then packed+text, sorting valid mask tokens first with `argsort`/`gather`.
- DiT cross-attention consumes packed `encoder_hidden_states [B,L_cond,2048]`, then `condition_embedder` projects to hidden size. For the sampled 1.5 config, projection is 2048 -> 2048.

Latents and denoiser step:

- Source/silence latents: `[B,T,64]`.
- Chunk mask: `[B,T,64]`, ones for generated spans and zeros for kept source spans.
- Context latents: `[B,T,128] = cat(src_latents, chunk_mask, dim=-1)`.
- Noisy latent input/output: `[B,T,64]`.
- CFG/APG batch when active: `[2B,T,64]` hidden states, `[2B,T,128]` context, `[2B,L_cond,2048]` condition.
- Denoiser output `vt`: velocity field `[B,T,64]`.
- Scheduler update: `prev = sample + (sigma_next - sigma) * vt`.

Codec boundary:

- Oobleck encode input: waveform `[B,2,S]`; output distribution parameters `[B,128,ceil-ish S/1920]`, split into mean/std for sampled latents `[B,64,T]`.
- Oobleck decode input: `[B,64,T]`; output waveform `[B,2,T*1920]` subject to ConvTranspose1d boundary effects and tiling.
- Pipeline transposes DiT latents `[B,T,64] -> [B,64,T]` before decode and normalizes waveform peaks to -1 dBFS.

CPU/data-pipeline work includes prompt formatting, tokenization, string parsing, metadata construction, audio crop/repeat, output NumPy conversion, and callback/interrupt paths. GPU/runtime work includes text encoder if admitted, ACE condition encoder, Oobleck encode/decode, DiT, APG reductions, scheduler arithmetic, and waveform normalization.

## 5. Operator coverage checklist

Tensor/layout ops:

- `cat`, `chunk`, `split`, `transpose NTC<->NCT`, reshape, unflatten, pad on time, crop/slice, expand, repeat, `where`, `argsort`, `gather`, `one_hot`, matmul for unpacking timbre embeddings, bool masks, clamp, arange.

Convolution/audio ops:

- DiT patchify `Conv1d(192 -> 2048, kernel=2, stride=2)` on NCT.
- DiT unpatchify `ConvTranspose1d(2048 -> 64, kernel=2, stride=2)`.
- Oobleck encoder/decoder `Conv1d`, dilated `Conv1d(kernel=7,dilation=1/3/9)`, `Conv1d(kernel=1)`, strided `Conv1d(kernel=2*stride,stride=stride)`, `ConvTranspose1d(kernel=2*stride,stride=stride)`.
- Weight-normalized Conv1d/ConvTranspose1d weights.

GEMM/linear ops:

- Qwen3 text encoder if included: GQA transformer with RMSNorm/SwiGLU/RoPE.
- ACE condition projection `Linear(1024 -> 2048, bias=False)`, lyric/timbre input projections, optional condition projection.
- Timestep MLPs: two `Timesteps(256) -> Linear(256,2048) -> SiLU -> Linear(2048,2048) -> SiLU -> Linear(2048,12288)`.
- Per block Q/K/V/out projections: Q 2048->2048, K/V 2048->1024, out 2048->2048, bias false.
- MLP: gate/up 2048->6144, down 6144->2048.
- Optional FSQ tokenizer/detokenizer linear projections.

Attention primitives:

- GQA self-attention with 16 Q heads, 8 KV heads, head dim 128.
- GQA cross-attention to condition sequence, same head layout.
- Alternating full and sliding-window self-attention.
- Q/K RMSNorm and 1D RoPE on self-attention only.

Normalization and custom math:

- RMSNorm over hidden/head dimensions, AdaLN-like shift/scale/gate from timestep projections, SiLU, SwiGLU, Snake1d, softplus posterior scale, APG L2 norms/projections, waveform peak reductions.

Scheduler/guidance:

- FlowMatch Euler custom sigma tables, step-index state, APG momentum buffer, CFG/APG batched denoiser call, cover-strength blending.

## 6. Denoiser/model breakdown

`AceStepTransformer1DModel` forward:

```text
timestep and (timestep - timestep_r)
  -> two sinusoidal timestep MLPs
  -> temb [B,2048] and AdaLN projection [B,6,2048]
cat(context_latents [B,T,128], hidden_states [B,T,64])
  -> pad T to multiple of 2
  -> transpose to [B,192,T]
  -> Conv1d stride-2 patchify -> [B,2048,T/2]
  -> transpose to tokens [B,T/2,2048]
condition_embedder(encoder_hidden_states)
RoPE tables for T/2 tokens
24 transformer blocks
RMSNorm + output AdaLN shift/scale
ConvTranspose1d stride-2 -> [B,T_padded,64]
crop to original T
```

Each transformer block:

```text
scale_shift_table + timestep_proj -> shift/scale/gate for self-attn and MLP
RMSNorm -> affine -> GQA self-attn with RoPE and optional sliding mask -> gated residual
RMSNorm -> GQA cross-attn to packed condition sequence -> residual
RMSNorm -> affine -> SwiGLU MLP -> gated residual
```

`AceStepConditionEncoder`:

- Text path: `Linear(1024 -> 2048, bias=False)`.
- Lyric path: `Linear(1024 -> 2048)` then 8 pre-LN `AceStepEncoderLayer`s with alternating sliding/full self-attention and mask.
- Timbre path: `Linear(64 -> 2048)` then 4 pre-LN layers, first-token pooling, and batch unpack using `argsort`, `one_hot`, and `matmul`.
- Packing path sorts valid lyric/timbre/text tokens first.

`AceStepAudioTokenizer/Detokenizer` optional path:

- Tokenizer groups 25 Hz acoustic latents into windows of 5, uses attention pooling, then residual finite scalar quantization.
- Detokenizer expands each 5 Hz token back to 5 acoustic frames using special tokens plus 2 attention layers and `Linear(2048 -> 64)`.

## 7. Attention requirements

The primary implementation is `AceStepAttnProcessor2_0` plus `dispatch_attention_fn` in `attention_dispatch.py`, not the generic `Attention` module. Native/eager/SDPA defines parity; flash, flash_hub, flash_varlen, flash_varlen_hub, sage, and other registry backends are optional provider choices.

Self-attention:

- Input tokens `[B,S,2048]`; Q `[B,S,16,128]`, K/V `[B,S,8,128]`.
- RMSNorm on per-head Q/K.
- 1D RoPE on Q/K with Qwen-style rotate-half convention.
- Alternating layers use either full dense attention or non-causal sliding window 128.
- Native path uses additive 4D masks `[1,1,S,S]` for sliding layers.

Cross-attention:

- Query `[B,S,2048]`; K/V from condition tokens `[B,L_cond,2048]`.
- Same GQA layout and Q/K RMSNorm, no RoPE.
- Source passes no cross-attention mask in DiT today.

Flash-style Dinoml constraints:

- Must support dense non-causal self/cross attention, GQA or explicit K/V expansion, head dim 128, bf16, Q/K RMSNorm, pre-applied RoPE, and different query/key lengths for cross-attention.
- Sliding-window attention needs either a windowed flash kernel or a fallback to masked native attention.
- Varlen/padded-mask flash is relevant to lyric/condition encoders; first denoiser slice can accept packed condition states and avoid dynamic padded masks.
- QKV fusion is source-disabled because Q and K/V widths differ under GQA; separate Q/K/V projection fusion is still possible under guarded metadata.

## 8. Scheduler and denoising-loop contract

ACE-Step computes its own sigma schedule before calling the scheduler:

```text
t = linspace(1, 0, N + 1)[:-1]
if shift != 1:
  t = shift * t / (1 + (shift - 1) * t)
scheduler.set_timesteps(sigmas=t.tolist(), device=device)
for t_sched in scheduler.timesteps:
  vt = transformer(...)
  vt = APG(vt_cond, vt_uncond) if enabled
  vt = cover_strength * vt + (1 - cover_strength) * vt_non_cover if enabled
  xt = scheduler.step(vt, t_sched, xt).prev_sample
```

`FlowMatchEulerDiscreteScheduler` appends terminal sigma 0 and for the non-stochastic first slice computes `prev_sample = sample + (sigma_next - sigma) * model_output`. Since the pipeline passes sigmas in `[0,1]` and expects `num_train_timesteps=1`, scheduler timesteps equal sigmas. Keep scheduler `step_index`, custom sigma list, terminal zero, and validation as host-visible state initially.

APG is not vanilla CFG. For base/SFT, the model runs conditional and learned-null batches together, then `normalized_guidance` computes `diff = pred_cond - pred_uncond`, updates momentum with -0.75, clamps norm to 2.5 over `dim=(1,)`, projects the update parallel/orthogonal to normalized `pred_cond`, and returns `pred_cond + (guidance_scale - 1) * update`. Turbo checkpoints skip this.

## 9. Position, timestep, and custom math

- Text prompt formatting is part of the model contract: instruction/caption/metas template and lyric language header include `<|endoftext|>`.
- Timestep embeddings use Diffusers `Timesteps(num_channels=256, flip_sin_to_cos=True)` after multiplying by scale 1000.
- The DiT uses dual timestep paths for `t` and `t-r`; inference passes `r=t`, so the second path sees zero but still contributes learned MLP/projection output.
- RoPE is `_ace_step_rotary_freqs(seq_len, head_dim=128, theta=1e6)` with `repeat_interleave_real=False` and `apply_rotary_emb(..., use_real_unbind_dim=-2)`.
- FSQ custom math uses clamp/tanh/floor/round and mixed fp32 linear projections.
- Oobleck custom math is `Snake1d(x) = x + reciprocal(exp(beta)+1e-9) * sin(exp(alpha) * x)^2`.
- APG uses double-precision projection math on CUDA/CPU before casting back.

Precompute per request: text/Qwen hidden states, lyric embeddings, packed condition, silence/reference timbre latents, RoPE by token length, scheduler tables, chunk masks. Dynamic: audio duration, source/reference audio length, task type, repaint span, guidance interval, and custom timesteps.

## 10. Preprocessing and input packing

Text prompt path uses tokenizer padding `"longest"`, truncation, max 256. Lyrics use the same tokenizer, padding `"longest"`, max 2048, but only the embedding layer is run before the ACE lyric encoder. Negative prompts are not encoded; learned `null_condition_emb` supplies unconditional conditioning.

Audio input path:

- Source audio is encoded by Oobleck to `[B,64,T]`, transposed to `[B,T,64]`.
- Reference audio is repeated/cropped to 30 seconds by taking front/middle/back 10-second chunks, encoded, then transposed for timbre.
- No source audio uses `condition_encoder.silence_latent`, sliced/repeated to target latent length.
- Repaint replaces source latents inside the generation window with silence latents and uses chunk mask 1 inside the window.
- Audio codes parse integer IDs, use FSQ `get_output_from_indices`, and detokenize 5 Hz codes to 25 Hz acoustic latents.

## 11. Graph rewrite / lowering opportunities

1. DiT patchify Conv1d as strided linear over two-frame windows
   - Source pattern: `[B,T,192] -> pad -> transpose -> Conv1d(kernel=2,stride=2,out=2048) -> transpose`.
   - Replacement: guarded token GEMM over flattened `[B,ceil(T/2),384]`.
   - Preconditions: fixed patch size 2, no dilation/padding except trailing zero time pad, contiguous NTC input, Conv1d bias preserved.
   - Failure cases: future patch sizes or nonzero padding.
   - Test: odd/even T patchify parity.

2. ConvTranspose1d unpatchify as token linear plus scatter
   - Source pattern: `[B,S,2048] -> transpose -> ConvTranspose1d(kernel=2,stride=2,out=64) -> transpose -> crop`.
   - Replacement: linear to 128 then reshape to two frames.
   - Preconditions: kernel=stride=2, padding=0, no overlap, static/captured output crop.
   - Failure cases: overlapping transposed conv variants.
   - Test: output parity for T=1499/1500.

3. Weight-norm materialization for Oobleck
   - Source pattern: frozen `weight_norm(Conv1d/ConvTranspose1d)`.
   - Replacement: load-time dense normalized weights.
   - Preconditions: inference-only and no runtime adapter mutation of codec weights.
   - Failure cases: training or adapter-tuned codec.
   - Test: one encoder/decoder block before/after transform.

4. NTC/NCT layout region around DiT
   - Source pattern: public DiT and condition tensors are NTC, Conv1d wants NCT.
   - Replacement: keep transformer tokens NTC; lower patch/unpatch as GEMM or internal NCT conv island.
   - Preconditions: no external observer between transpose and conv; time axis crop handled explicitly.
   - Failure cases: direct compilation of arbitrary Conv1d module with public NCT ABI.
   - Test: full DiT pre/post stem parity.

5. APG plus FlowMatch pointwise fusion
   - Source pattern: chunk, reductions/norms, projection update, cover blend, Euler step.
   - Replacement: staged kernels: APG reduction/projection then fused pointwise scheduler update.
   - Preconditions: APG mode fixed, norm dim `(1,)`, non-stochastic FlowMatch.
   - Failure cases: turbo no-APG path or custom APG parameters.
   - Test: one denoising step with fixed model outputs and momentum state.

## 12. Kernel fusion candidates

Highest priority:

- RMSNorm + Q/K projection/norm + attention for ACE GQA. This is the denoiser core and exercises head dim 128 plus sliding-window/full attention.
- SwiGLU MLP projections in 24 DiT blocks and 12 condition-encoder layers.
- DiT patchify/unpatchify Conv1d lowering to GEMM-like token transforms.
- FlowMatch Euler + turbo no-CFG pointwise step over `[B,T,64]`.

Medium priority:

- APG reductions/projection/momentum state for base/SFT models.
- Oobleck Conv1d/ConvTranspose1d + Snake decode island.
- Lyric/timbre condition encoders, including packing/gather.
- FSQ audio-code tokenizer/detokenizer for cover variants.

Lower priority:

- Full Qwen3 text encoder compilation; start with cached embeddings.
- Oobleck tiling orchestration for long decode/encode.
- Waveform peak normalization and NumPy conversion.
- Legacy 3.5B AutoencoderDC/vocoder stack.

## 13. Runtime staging plan

Stage 1: parse ACE-Step 1.5 component configs from official repo cache and admit externally supplied `encoder_hidden_states`, `context_latents`, latents, timestep, and RoPE for one DiT step.

Stage 2: DiT block parity for fixed `[B=1,T=1500,64]` latents, source/cached condition sequence, turbo guidance disabled.

Stage 3: FlowMatch Euler custom-sigmas scheduler table and one-step parity, then short 8-step turbo latent loop with Oobleck external.

Stage 4: Oobleck decode-only codec island for `[B,64,T] -> [B,2,T*1920]`, no tiling first.

Stage 5: condition encoder integration with external Qwen hidden states and lyric embeddings; add packing/gather parity.

Stage 6: optional Qwen3 text encoder/tokenizer cache ABI.

Stage 7: base/SFT APG admission if non-turbo configs become first-class.

Stage 8: audio-to-audio/audio-code variants and Oobleck encode.

First Dinoml admission recommendation: start with `ace_step_1_5_turbo_dit_one_step` and FlowMatch custom-sigma loop using externally supplied condition/context tensors. It avoids APG and text-encoder/Oobleck complexity while exercising the unique audio DiT ops.

## 14. Parity and validation plan

- Config parsing tests for `Ace-Step1.5/config.json`, `vae/config.json`, and Qwen3 config; prove placeholder local model_index files are ignored.
- Random tensor tests for `_ace_step_rotary_freqs`, `apply_rotary_emb` convention, RMSNorm, timestep embedding, patchify/unpatchify, and FlowMatch custom sigma tables.
- Single attention parity for full and sliding-window self-attention plus cross-attention with GQA.
- Single `AceStepTransformerBlock` parity at `[1,750,2048]` tokens.
- Full DiT one-step parity at `[1,1500,64]` with cached condition states and context latents.
- FlowMatch Euler one-step and 8-step fixed-output parity.
- APG parity with fixed `pred_cond`, `pred_uncond`, and momentum state.
- Condition encoder parity for text/lyric/timbre packing.
- Oobleck decode parity for latent lengths 512, 750, 1500; encode posterior mean/std parity for source/reference audio.
- End-to-end smoke with `output_type="latent"` before waveform decode.

Suggested tolerances: fp32 custom math `rtol=1e-5, atol=1e-6`; bf16/fp16 DiT/codec initially `rtol=2e-2, atol=2e-2`, tightened by kernel.

## 15. Performance probes

- One DiT step by latent length 250, 750, 1500, and 3000.
- Attention backend comparison: native dense mask vs flash windowed vs fallback for sliding layers.
- DiT patchify/unpatchify time as Conv1d vs GEMM rewrite.
- Condition encoder time split: lyric encoder, timbre encoder, packing.
- Oobleck decode throughput for 10, 30, 60, and 120 seconds with and without tiling.
- Scheduler/APG overhead for turbo 8 steps and base guidance.
- Memory probes for APG batch doubling and long lyric length.
- Qwen3 text encoder throughput and prompt cache hit rate.

No benchmark measurements were run in this audit.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items:

- `ace_step_oobleck_codec`: Conv1d/ConvTranspose1d/Snake/weight-norm waveform codec.
- `ace_step_condition_encoder`: lyric/timbre/text packing, masks, and GQA encoder layers.
- `ace_step_apg_guidance`: learned-null APG for base/SFT non-turbo checkpoints.
- `ace_step_audio_codes`: FSQ tokenizer/detokenizer and audio-code string path.
- `ace_step_audio_to_audio`: repaint, cover, extract, lego, complete source-audio contracts.
- `ace_step_lora`: LoRA-only repo and adapter loading/fusion.
- `ace_step_v1_3_5b_legacy`: old 2D transformer, AutoencoderDC, and vocoder stack.
- `qwen3_embedding_text_encoder`: external text encoder/tokenizer cache integration.

Ignored/out of scope for this audit:

- XLA/NPU/MPS/Flax/ONNX paths.
- Training, loss, dropout behavior, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt behavior.
- Safety/NSFW filtering; not present.
- Image-only ControlNet/IP-Adapter/T2I-Adapter/GLIGEN/depth/upscale variants.

## 17. Final implementation checklist

- [ ] Parse ACE-Step 1.5 transformer, condition, Qwen, Oobleck, and scheduler metadata.
- [ ] Admit rank-3 audio latents `[B,T,64]` and explicit codec boundary `[B,64,T]`.
- [ ] Implement DiT Conv1d patchify/unpatchify or guarded GEMM rewrites.
- [ ] Implement ACE GQA attention with Q/K RMSNorm, RoPE, and sliding-window fallback.
- [ ] Add `AceStepTransformerBlock` and full-DiT one-step parity tests.
- [ ] Implement FlowMatch Euler custom-sigma table and non-stochastic step parity.
- [ ] Add turbo 8-step latent-loop parity with external condition/context tensors.
- [ ] Add Oobleck decode-only codec island tests.
- [ ] Add condition encoder packing/gather parity.
- [ ] Add APG guidance parity as a separate non-turbo milestone.
- [ ] Benchmark DiT attention, patch/unpatch, scheduler/APG, condition encoder, and Oobleck decode.
