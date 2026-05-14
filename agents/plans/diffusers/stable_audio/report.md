# Stable Audio Diffusers Audit

Candidate slug: `stable_audio`

## 1. Source basis

```text
Diffusers commit/version:
  Local checkout X:/H/diffusers at b3a515080752a3ba7ca92161e25530c7f280f629.

Model id(s):
  stabilityai/stable-audio-open-1.0, official repo gated for file contents.
  Open mirrors inspected for component configs:
    AEmotionStudio/stable-audio-open-models
    ford442/stable-audio-open-1.0
    LanguaMan/stable-audio-open-1-0-model
    ModelsLab/stable-audio-open-1.0

Config sources:
  Local H:/configs had model_index.json only for the mirrors above.
  Official stabilityai tree view exposed file names but blocked raw contents behind
  an access gate. No HF token was present in the environment, so authenticated
  retry could not proceed.
  Mirror raw configs were read without writing them to H:/configs:
    model_index.json, model_config.json
    transformer/config.json
    vae/config.json
    projection_model/config.json
    scheduler/scheduler_config.json
    text_encoder/config.json
    tokenizer/tokenizer_config.json

Pipeline files inspected:
  X:/H/diffusers/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py
  X:/H/diffusers/src/diffusers/pipelines/stable_audio/modeling_stable_audio.py
  X:/H/diffusers/src/diffusers/pipelines/stable_audio/__init__.py

Model files inspected:
  X:/H/diffusers/src/diffusers/models/transformers/stable_audio_transformer.py
  X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_oobleck.py

Scheduler/processors/helpers inspected:
  X:/H/diffusers/src/diffusers/schedulers/scheduling_cosine_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/schedulers/scheduling_edm_dpmsolver_multistep.py
  X:/H/diffusers/src/diffusers/models/attention_processor.py
  X:/H/diffusers/src/diffusers/models/embeddings.py for get_1d_rotary_pos_embed/apply_rotary_emb
  X:/H/diffusers/src/diffusers/models/attention.py for Attention and FeedForward

External component configs inspected:
  T5EncoderModel/T5 tokenizer configs from the open mirror.

Missing files or assumptions:
  Official stabilityai component config contents are gated. Mirror configs match
  each other for inspected fields and label themselves as mirrors of
  stabilityai/stable-audio-open-1.0, but they are not the official repo.
```

## 2. Pipeline and component graph

`StableAudioPipeline` components are `vae: AutoencoderOobleck`, `text_encoder: T5EncoderModel`, `projection_model: StableAudioProjectionModel`, `tokenizer: T5Tokenizer | T5TokenizerFast`, `transformer: StableAudioDiTModel`, and `scheduler`. The source type annotation names `EDMDPMSolverMultistepScheduler`; the sampled configs name `CosineDPMSolverMultistepScheduler`. CPU offload order is `text_encoder->projection_model->transformer->vae`.

```text
prompt + duration seconds + optional initial audio
  -> T5 tokenizer/text encoder
  -> StableAudioProjectionModel text and seconds projection
  -> latent noise init, optionally plus Oobleck encode(initial audio)
  -> denoising loop: StableAudioDiTModel + CFG + DPM scheduler
  -> AutoencoderOobleck decode
  -> crop to requested waveform start/end
```

Required first-slice components are the projected conditioning tensors, audio latents, transformer denoiser, selected scheduler, and Oobleck decode. Text encoding and Oobleck encode can be staged as cacheable/external boundaries: Dinoml can initially accept `text_audio_duration_embeds`, `audio_duration_embeds`, and latents directly.

Separate candidate reports:

- Stable Audio Oobleck codec: `AutoencoderOobleck` encode/decode deserves its own audio-codec report because it is Conv1d/ConvTranspose1d/Snake-heavy and independently useful.
- Stable Audio scheduler compatibility: `CosineDPMSolverMultistepScheduler` is the config default, while current pipeline source imports EDM DPM-Solver. Treat this as a focused scheduler admission item.
- LoRA/textual inversion/runtime adapters: no Stable Audio pipeline loader mixins or textual inversion path were found in the selected pipeline. Generic attention processor mutation exists through `AttentionMixin`, but there is no base first-slice runtime adapter contract.
- IP-Adapter, ControlNet, T2I-Adapter, GLIGEN, img2img, inpaint, depth2img, upscaling: not present in the `stable_audio` pipeline folder. The only variant-like surface is optional initial audio waveform conditioning, which uses Oobleck encode plus latent noising/addition rather than an img2img scheduler-strength contract.

## 3. Important config dimensions

Representative Stable Audio Open config sweep:

| Repo/config source | Pipeline | Transformer | VAE | Scheduler |
| --- | --- | --- | --- | --- |
| `H:/configs/ford442/stable-audio-open-1.0/model_index.json` plus mirror raw component configs | `StableAudioPipeline` | `StableAudioDiTModel` | `AutoencoderOobleck` | `CosineDPMSolverMultistepScheduler` |
| `H:/configs/LanguaMan/stable-audio-open-1-0-model/model_index.json` plus mirror raw component configs | same | same | same | same |
| `H:/configs/ModelsLab/stable-audio-open-1.0/model_index.json` plus mirror raw component configs | same | same | same | same |
| `H:/configs/AEmotionStudio/stable-audio-open-models/model_index.json` plus mirror raw component configs | same | same | same | same |

Transformer dimensions from mirror `transformer/config.json`:

| Field | Value | Source |
| --- | ---: | --- |
| `sample_size` | `1024.0` latent frames | component config |
| `in_channels` / `out_channels` | 64 / 64 | component config |
| layers | 24 | component config |
| attention heads | 24 query heads | component config |
| key/value heads | 12 | component config, GQA |
| head dim | 64 | component config |
| inner dim | 1536 | inferred from source `heads * head_dim` |
| cross attention dim | 768 | component config |
| cross attention input dim | 768 | component config |
| global states input dim | 1536 | component config |
| timestep projection dim | 256 | component config |

Audio codec and text dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| raw sample rate | 44100 Hz | VAE config |
| audio channels | 2 | VAE config |
| downsampling ratios | `[2, 4, 4, 8, 8]` | VAE config |
| hop length | 2048 | source/config product |
| max raw samples | 2097152 | mirror `model_config.json` |
| max duration | about 47.55 s | inferred `1024 * 2048 / 44100` |
| latent channels | 64 | transformer/VAE decoder input config |
| tokenizer max length | 128 | tokenizer config |
| T5 hidden size | 768 | text encoder config |
| T5 layers/heads | 12 / 12 | text encoder config |
| projection conditioning dim | 768 | projection config |
| duration min/max | 0 / 512 seconds | projection config |

Scheduler dimensions:

| Field | Config default |
| --- | --- |
| source default in current pipeline annotation | `EDMDPMSolverMultistepScheduler` |
| checkpoint/mirror default | `CosineDPMSolverMultistepScheduler` |
| `prediction_type` | `v_prediction` |
| `sigma_schedule` | `exponential` |
| `sigma_min` / `sigma_max` / `sigma_data` | `0.3` / `500` / `1.0` |
| `solver_order` / `solver_type` | `2` / `midpoint` |
| final sigma | `zero` |
| first Dinoml scheduler slice | `CosineDPMSolverMultistepScheduler` v-pred, exponential sigmas, solver order 2 |

## 3a. Family variation traps

- The source imports EDM DPM-Solver, but Stable Audio Open configs use Cosine DPM-Solver. Config parsing must admit the checkpoint scheduler class rather than trusting the pipeline type annotation.
- Audio latents are rank-3 NCL tensors `[batch, channels=64, latent_time=1024]`, not image NCHW or video NCDHW.
- The transformer internally transposes latents to token layout `[batch, sequence, channels]`, prepends one global duration/time token, then transposes back and drops that prepended token.
- Rotary length is `latent_time + audio_duration_seq_len`; for the base config that is `1024 + 1`.
- CFG duplication is asymmetric: text/audio duration condition tensors are duplicated per waveform, while if guidance has no explicit negative prompt the negative conditioning is all zeros for cross-attention tokens but duration global states are duplicated.
- Oobleck encode emits distribution parameters with `2 * latent_channels`; sampling uses softplus scale and random noise. Decode consumes 64-channel latents.
- Oobleck tiling/slicing changes execution shape and can alter encode parity at tile boundaries. Keep first-slice no-tiling/no-slicing unless explicitly admitted.
- Weight-normalized Conv1d/ConvTranspose1d parameters must be materialized or represented explicitly. Treat `weight_norm` as a load-time weight transform candidate, not a new runtime op if weights are frozen.

## 4. Runtime tensor contract

Pipeline inputs:

- `prompt: str | list[str]`, or externally supplied `prompt_embeds [B, 128, 768]` plus optional mask.
- `audio_start_in_s`, `audio_end_in_s`: scalar or list, clamped/admitted to `[0, 512]`.
- Optional `negative_prompt` or `negative_prompt_embeds` for CFG.
- Optional `initial_audio_waveforms [B, C_audio, raw_time]` or `[B, raw_time]`, with sampling rate equal to 44100.
- Optional `latents [B * waveforms, 64, 1024]`.

Conditioning tensors:

- T5 encoder output: `[B, 128, 768]`.
- Projected prompt embeddings: `[B or 2B, 128, 768]`.
- Duration start/end hidden states: each `[B or 2B, 1, 768]`.
- Cross-attention states: `text_audio_duration_embeds [B_eff, 130, 768]`.
- Global duration states: `audio_duration_embeds [B_eff, 1, 1536]`, formed by concatenating start/end seconds on the feature axis.

Latents and denoiser step:

- Latent source layout is NCL `[B_eff, 64, 1024]`.
- Noise is scaled by `scheduler.init_noise_sigma`. For the cosine config this is `sqrt(500^2 + 1)`.
- Denoiser input under CFG is `[2 * B_eff, 64, 1024]`, scaled by scheduler `c_in = 1 / sqrt(sigma^2 + sigma_data^2)`.
- Transformer returns noise/model output `[2 * B_eff, 64, 1024]` or `[B_eff, 64, 1024]`.
- CFG computes `uncond + guidance_scale * (text - uncond)` before scheduler step.

Codec boundary:

- Oobleck encode input: raw audio `[B, 2, raw_samples]`, padded/cropped to `1024 * 2048 = 2097152` samples in the pipeline initial-audio path.
- Oobleck encode output distribution parameters: `[B, 128, latent_time]`, split to mean/scale `[B, 64, latent_time]`.
- Oobleck decode input: `[B_eff, 64, 1024]`.
- Oobleck decode output: waveform `[B_eff, 2, about 2097152]`, then cropped to `[waveform_start:waveform_end]`.

CPU/data-pipeline work includes tokenization, prompt length validation, input audio channel conversion, crop/pad, and output NumPy conversion. GPU/runtime work includes T5 if admitted, projection MLPs, latent noise, transformer, scheduler arithmetic, Oobleck encode/decode, and final crop.

## 5. Operator coverage checklist

Tensor/layout ops:

- rank-3 transpose NCL `<->` NLC, reshape/view, cat on sequence and batch axes, chunk on batch axis, repeat/repeat_interleave, unsqueeze, crop/slice, pad/zero fill, clamp, where for masked negative prompt embeddings.

Convolution/audio ops:

- Transformer `Conv1d(64 -> 64, kernel=1, bias=False)` pre/post residual convs.
- Oobleck `Conv1d`, dilated `Conv1d(kernel=7, dilation=1/3/9)`, `Conv1d(kernel=1)`, strided `Conv1d(kernel=2*stride, stride=stride)`, `ConvTranspose1d(kernel=2*stride, stride=stride)`.
- Weight normalization on Conv1d/ConvTranspose1d weights.

GEMM/linear ops:

- T5 encoder linear stack if included.
- Projection model `Linear(768 -> 768)` is identity for base config text projection, plus seconds `Linear(257 -> 768)`.
- Transformer timestep MLP `Linear(256 -> 1536) -> SiLU -> Linear(1536 -> 1536)`.
- Global MLP `Linear(1536 -> 1536, bias=False) -> SiLU -> Linear(1536 -> 1536, bias=False)`.
- Cross projection `Linear(768 -> 768, bias=False) -> SiLU -> Linear(768 -> 768, bias=False)`.
- Per block QKV/out projections: self-attn Q/K/V/out 1536-wide MHA; cross-attn Q 1536-wide and K/V 768-to-768 GQA; FFN SwiGLU inner projection from 1536 to hidden MLP width from source default.

Attention primitives:

- SDPA self-attention over 1025 tokens with RoPE on query/key.
- SDPA cross-attention from 1025 query tokens to 130 condition tokens.
- GQA repeat for cross-attention K/V from 12 KV heads to 24 query heads.

Normalization and custom math:

- LayerNorm on token hidden states.
- Snake1d activation `x + exp(beta)^-1 * sin(exp(alpha) * x)^2`.
- SiLU, SwiGLU, softplus, sin/cos Fourier embeddings, atan/log/sqrt/exponential scheduler math.

Scheduler/guidance:

- Exponential sigma table, cosine timestep preconditioning `atan(sigma) / pi * 2`, input/output EDM preconditioning, BrownianTree noise sampling for cosine scheduler, solver-order history buffers, CFG pointwise arithmetic.

## 6. Denoiser/model breakdown

`StableAudioDiTModel`:

1. Project cross-attention states `[B, 130, 768] -> [B, 130, 768]`.
2. Project global states `[B, 1, 1536] -> [B, 1, 1536]`.
3. Fourier timestep projection: timestep `[B?] -> [B, 256] -> [B, 1536]`; add to global state.
4. Preprocess latents: `Conv1d(64, 64, 1)` plus residual, then transpose `[B, 64, 1024] -> [B, 1024, 64]`.
5. `proj_in: Linear(64 -> 1536, bias=False)`.
6. Prepend global token, giving `[B, 1025, 1536]`.
7. Run 24 `StableAudioDiTBlock`s.
8. `proj_out: Linear(1536 -> 64, bias=False)`, transpose to `[B, 64, 1025]`, drop the global token to `[B, 64, 1024]`.
9. `postprocess_conv(64, 64, 1)` plus residual.

Each block is pre-norm:

```text
LayerNorm -> self-attention with RoPE -> residual
LayerNorm -> cross-attention with GQA K/V -> residual
LayerNorm -> SwiGLU feed-forward -> residual
```

## 7. Attention requirements

The active processor is `StableAudioAttnProcessor2_0` in `attention_processor.py`. It uses `torch.nn.functional.scaled_dot_product_attention`, not `attention_dispatch.py`.

Self-attention:

- Query/key/value from hidden tokens `[B, 1025, 1536]`.
- 24 heads, head dim 64.
- Rotary embedding applied to query and key partial head dimensions.
- No causal mask. Optional attention mask is prepared as `[B, heads, source, target]`, but the base pipeline does not pass one into the transformer loop.

Cross-attention:

- Query from `[B, 1025, 1536]`.
- Key/value from projected condition tokens `[B, 130, 768]`.
- Query heads 24, KV heads 12, head dim 64. Source repeats K/V heads to query-head count before SDPA.
- No RoPE on cross-attention keys in the active path; RoPE argument is only passed to self-attention.

Flash-style Dinoml constraints:

- Eager/native parity is PyTorch SDPA with explicit K/V repeat for GQA.
- A flash provider is valid for no-dropout, non-causal, dense attention when it supports 1D RoPE-preprocessed Q/K, head dim 64, and either native GQA or pre-expanded K/V heads.
- Cross-attention has different query/key sequence lengths and GQA; do not assume self-attention-only kernels cover it.
- No packed/varlen path is active in source. Mask support can be deferred if first slice accepts already-masked prompt embeddings and no transformer attention masks.

## 8. Scheduler and denoising-loop contract

The Stable Audio Open configs require `CosineDPMSolverMultistepScheduler`, not the current pipeline annotation. It is an SDE DPM-Solver++-style multistep scheduler with BrownianTree noise.

Loop contract:

```text
scheduler.set_timesteps(num_inference_steps, device)
for t in timesteps:
  latent_model_input = cat([latents] * 2) if CFG else latents
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)
  model_output = transformer(latent_model_input, t.unsqueeze(0), cond, global, rope)
  model_output = CFG(model_output) if enabled
  latents = scheduler.step(model_output, t, latents, generator).prev_sample
```

Cosine scheduler details:

- `set_timesteps` creates exponential or Karras sigmas; base config uses exponential.
- Timesteps are `atan(sigma) / pi * 2`, unlike EDM DPM-Solver's `0.25 * log(sigma)`.
- `scale_model_input` multiplies by `1 / sqrt(sigma^2 + sigma_data^2)`.
- `convert_model_output` computes denoised `x0 = c_skip * sample + c_out * model_output`; base config uses `v_prediction`, so `c_out` is negative.
- Solver order 2 uses one-step warmup, then second-order multistep with model-output history.
- BrownianTree noise is initialized lazily from the generator seed and used in both first- and second-order updates.
- `sigmas` are kept on CPU between calls and moved/scalar-indexed as needed.

Initial Dinoml staging should keep `set_timesteps`, BrownianTree noise ownership, model-output history, and step index in host-visible state. Compile the pointwise `scale_model_input`, CFG, preconditioning, and one scheduler step only after the cosine scheduler state schema is explicit.

## 9. Position, timestep, and custom math

Stable Audio uses three positional/time mechanisms:

- Duration number conditioning: clamp seconds, normalize to `[0, 1]`, apply learned Fourier frequencies with sin/cos plus the normalized scalar, then `Linear(257 -> 768)`.
- Denoising timestep: fixed Gaussian Fourier projection with `time_proj_dim=256`, `flip_sin_to_cos=True`, no log, then MLP to 1536.
- 1D RoPE: `get_1d_rotary_pos_embed(rotary_embed_dim=32, sequence_length=1025, use_real=True, repeat_interleave_real=False)`, then partial RoPE on Q/K head channels.

Custom math that needs parity:

```text
Snake1d(x) = x + reciprocal(exp(beta) + 1e-9) * sin(exp(alpha) * x)^2
Cosine timestep = atan(sigma) / pi * 2
Duration normalized = clamp(seconds, min, max) / (max - min)
```

RoPE is precomputable for fixed latent length and duration-token count. Duration embeddings depend on request start/end seconds. Scheduler tables depend on step count and config.

## 10. Preprocessing and input packing

Tokenization pads/truncates to tokenizer max length 128. Negative prompts are encoded separately when present; masked negative prompt tokens are zeroed. The projection model then maps text and seconds into the common 768-dim condition space.

The pipeline concatenates prompt, start seconds, and end seconds along sequence to make 130 cross-attention tokens. It concatenates the two duration embeddings along feature dim to make a single 1536-dim global token. Both are duplicated for `num_waveforms_per_prompt`.

Initial audio path:

- Accept `[B, T]`, `[B, 1, T]`, or `[B, 2, T]`.
- Convert mono/stereo to model channel count.
- Pad/crop to `sample_size * hop_length`.
- Encode with Oobleck, sample posterior, repeat per waveform, and add to the scaled noise latents.

There is no spectrogram or vocoder path in this pipeline; the codec operates directly on waveform tensors.

## 11. Graph rewrite / lowering opportunities

1. `weight_norm` Conv1d materialization
   - Source pattern: frozen `weight_norm(nn.Conv1d/ConvTranspose1d)` in Oobleck.
   - Replacement: precompute normalized dense conv weights at load time.
   - Preconditions: inference-only, no training weight updates, epsilon/parity matches PyTorch weight norm parametrization.
   - Failure cases: runtime adapter mutation or unfrozen codec weights.
   - Test: compare one Oobleck block before/after materialization.

2. NCL Conv1d channel-last local region
   - Source pattern: Oobleck and transformer Conv1d operate on `[B, C, T]`.
   - Replacement: guarded NTC/channel-last internal conv provider region.
   - Preconditions: all consumers in the region accept translated layout; axis rewrites for `dim=1`, concat/crop on time, and Snake parameters `[1, C, 1]`.
   - Failure cases: crossing transformer token boundary or public codec boundary without explicit transpose.
   - Test: block-level Conv1d/Snake/ConvTranspose1d parity for odd lengths and all stride ratios.

3. Transformer NCL-to-token transpose sinking
   - Source pattern: Conv1d residual in NCL, transpose, Linear over channels.
   - Replacement: fuse `Conv1d(1x1)` plus transpose/proj_in staging when layout and strides are static.
   - Preconditions: latent time fixed or bucketed; dense contiguous tensors; no external observer between conv and transpose.
   - Failure cases: dynamic latent lengths with tiling or direct latent output debug.
   - Test: transformer preamble parity.

4. CFG and scheduler pointwise fusion
   - Source pattern: chunk, `uncond + scale * (text - uncond)`, EDM/cosine preconditioning.
   - Replacement: single pointwise kernel over `[B, 64, T]`.
   - Preconditions: fixed scheduler class/prediction type and no guidance rescale.
   - Failure cases: no-CFG path or future guidance variants.
   - Test: one-step scheduler parity with fixed model output.

## 12. Kernel fusion candidates

Highest priority:

- Conv1d/ConvTranspose1d + Snake1d in Oobleck. This is the unique audio codec bottleneck and exercises unported Conv1d audio kernels.
- StableAudio SDPA with RoPE and GQA. Self-attention length 1025 and cross-attention length 130 are central to denoiser cost.
- LayerNorm + QKV/MLP projections in the 24 transformer blocks.
- Cosine DPM-Solver scheduler pointwise maps and CFG arithmetic over audio latents.

Medium priority:

- Timestep/global/cross projection MLPs with SiLU.
- SwiGLU feed-forward fusion.
- Weight-norm materialization and constant provenance.
- NCL/NTC transpose elision around transformer pre/post convs.

Lower priority:

- Oobleck tiling/slicing chunk orchestration.
- T5 encoder compilation; accept cached prompt embeddings first.
- NumPy output conversion and waveform crop.

## 13. Runtime staging plan

Stage 1: parse Stable Audio component configs and admit rank-3 audio latent tensors, using mirror configs until official gated configs are available.

Stage 2: transformer-only one-step parity with externally supplied `text_audio_duration_embeds`, `audio_duration_embeds`, latents, timestep, and precomputed RoPE.

Stage 3: implement/validate `CosineDPMSolverMultistepScheduler` host state and one-step arithmetic with fixed transformer output.

Stage 4: short denoising loop with scheduler in host control, Oobleck decode still external or PyTorch.

Stage 5: Oobleck decode operator island: Conv1d/ConvTranspose1d, Snake1d, weight-norm materialization, no tiling.

Stage 6: projection model and duration conditioning integration.

Stage 7: optional Oobleck encode for initial audio conditioning.

Stage 8: T5 tokenizer/text encoder integration or prompt-embedding cache ABI.

First admission recommendation: start with `stable_audio_dit_one_step` plus cosine scheduler metadata, not full end-to-end audio. The first public slice should require externally supplied conditioning and latents, then add Oobleck decode as a separate codec milestone.

## 14. Parity and validation plan

- Random tensor tests for `Snake1d`, duration Fourier embedding, timestep Fourier embedding, partial RoPE, and GQA K/V repeat.
- Single `StableAudioDiTBlock` parity at `[B=1, tokens=1025, hidden=1536]` plus cross states `[1, 130, 768]`.
- Full transformer one-step parity for `[1, 64, 1024]`, fixed timestep, fixed conditioning, fp32 first, then fp16/bf16.
- Cosine scheduler table parity for 100 and 200 steps: sigmas, timesteps, final sigma, `init_noise_sigma`.
- Cosine scheduler one-step and two-step parity with controlled Brownian noise or fixed noise sampler output.
- CFG parity with and without negative prompt embeddings.
- Oobleck decode parity for latent lengths 256, 512, 1024.
- Oobleck encode posterior mean/scale/sample parity for initial-audio path.
- End-to-end smoke with small step count and output type `latent`, then waveform decode.

Suggested tolerances: fp32 custom math `rtol=1e-5, atol=1e-6`; fp16/bf16 transformer/codec initially `rtol=2e-2, atol=2e-2`, tightened per kernel.

## 15. Performance probes

- One transformer denoiser step by batch, waveform count, and latent length.
- Attention split: self-attention length 1025 vs cross-attention length 130.
- Oobleck decode throughput for 10s, 30s, and 47.55s clips.
- Oobleck encode throughput for optional initial audio.
- Scheduler/guidance overhead for 100 and 200 steps.
- T5 text encoder throughput and prompt embedding cache hit rate.
- Memory probes for CFG batch doubling and Oobleck decode temporaries.
- Conv1d layout comparison: source NCL vs guarded channel-last provider.

No benchmark observations were run in this audit.

## 16. Scope boundary and separate candidates

Separate candidate reports or work items:

- `stable_audio_oobleck_codec`: Conv1d/ConvTranspose1d/Snake codec decode and encode.
- `scheduler_cosine_dpm_audio`: Cosine DPM-Solver with BrownianTree noise and v-prediction.
- `stable_audio_initial_audio_conditioning`: optional initial waveform encode/addition path.
- `stable_audio_text_encoder`: T5 encoder and tokenizer cache integration.
- `audio_ldm_music_ldm`: spectrogram/vocoder-style audio diffusion family, separate from Stable Audio waveform codec.
- `longcat_audio_dit`: newer audio DiT/codecs in `transformer_longcat_audio_dit.py` and `autoencoder_longcat_audio_dit.py`.

Ignored/out of scope for this audit:

- XLA/NPU/MPS/Flax/ONNX paths.
- Training, losses, dropout behavior, and gradient checkpointing.
- Multi-GPU/context parallel.
- Callback mutation and interactive interrupt behavior.
- Safety/NSFW filtering; not present in this pipeline.
- Runtime adapter/LoRA/textual inversion support unless a future Stable Audio adapter surface is identified.

## 17. Final implementation checklist

- [ ] Parse `StableAudioPipeline`, `StableAudioDiTModel`, `AutoencoderOobleck`, projection, T5, and scheduler configs.
- [ ] Add artifact-visible scheduler admission for `CosineDPMSolverMultistepScheduler` with v-prediction and exponential sigmas.
- [ ] Represent audio latents as rank-3 NCL `[B, C, T]` tensors with explicit optional NTC optimization regions.
- [ ] Implement or lower duration Fourier embeddings and timestep Fourier embeddings.
- [ ] Implement partial 1D RoPE and StableAudio GQA attention parity.
- [ ] Add transformer block parity tests with self- and cross-attention.
- [ ] Add CFG plus one-step cosine scheduler parity tests.
- [ ] Add Oobleck decode operator island tests for Conv1d, ConvTranspose1d, Snake1d, and weight-norm materialization.
- [ ] Add optional Oobleck encode/posterior sampling tests for initial audio.
- [ ] Benchmark denoiser attention, Oobleck decode, scheduler/CFG overhead, and text encoder cache behavior.
