# VITS Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: primary examples `facebook/mms-tts-eng` and `kakao-enterprise/vits-vctk`; additional config sweep from `facebook/mms-tts-spa`, `facebook/mms-tts-deu`, `facebook/mms-tts-fra`, `kakao-enterprise/vits-ljs`; attempted `facebook/mms-tts-cmn` returned 401.

Config source: `src/transformers/models/vits/configuration_vits.py`; representative raw HF `config.json` and tokenizer files summarized in `config_sweep.md`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/vits/modeling_vits.py`
- `X:/H/transformers/src/transformers/models/vits/configuration_vits.py`
- `X:/H/transformers/src/transformers/models/vits/tokenization_vits.py`
- `X:/H/transformers/src/transformers/models/vits/convert_original_checkpoint.py`
- `X:/H/transformers/src/transformers/pipelines/text_to_audio.py`
- `X:/H/transformers/tests/models/vits/test_modeling_vits.py`
- `X:/H/transformers/tests/models/vits/test_tokenization_vits.py`

Source URLs at the inspected commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vits/modeling_vits.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vits/configuration_vits.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vits/tokenization_vits.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vits/convert_original_checkpoint.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/pipelines/text_to_audio.py

Representative config URLs:

- https://huggingface.co/facebook/mms-tts-eng/raw/main/config.json
- https://huggingface.co/facebook/mms-tts-spa/raw/main/config.json
- https://huggingface.co/facebook/mms-tts-deu/raw/main/config.json
- https://huggingface.co/facebook/mms-tts-fra/raw/main/config.json
- https://huggingface.co/kakao-enterprise/vits-ljs/raw/main/config.json
- https://huggingface.co/kakao-enterprise/vits-vctk/raw/main/config.json
- https://huggingface.co/facebook/mms-tts-cmn/raw/main/config.json returned 401/unauthorized during this audit.

Any missing files or assumptions: no remote code is required for the in-library VITS path. `facebook/mms-tts-cmn` raw config access returned 401; treat that as a checkpoint coverage gap, not a source gap. This report targets inference-only text-to-waveform synthesis.

## 2. High-level architecture

VITS is a forward-only, stochastic TTS model, not an autoregressive text decoder. The runtime path is:

```text
text/tokenizer frontend -> text encoder -> stochastic duration predictor -> duration expansion/alignment -> prior latent sampling -> reverse flow -> HiFi-GAN vocoder -> waveform
```

Stage decomposition:

- CPU/data-pipeline: text normalization, optional uroman romanization, optional phonemizer/espeak conversion, blank insertion, padding and attention-mask construction.
- GPU/runtime text encoder: token embedding, noncausal Transformer encoder with relative position attention and convolutional feed-forward layers.
- GPU/runtime acoustic path: duration predictor, duration-to-alignment construction, latent sampling, reverse residual coupling flows.
- GPU/runtime vocoder: 1D convolution/transposed-convolution HiFi-GAN stack producing `[B, audio_samples]`.
- Training-only/deferred: posterior encoder and loss path. `labels` in `VitsModel.forward` raises `NotImplementedError`, so posterior encoder is not required for first inference parity.

The text encoder, duration predictor, flow, and vocoder can be validated independently with source-module parity before wiring end-to-end waveform tests.

## 3. Important config dimensions

| Field | Source default | Representative observed values | Runtime impact |
| --- | ---: | --- | --- |
| `vocab_size` | 38 | 38, 44, 45, 178 | Embedding table and tokenizer coupling vary by checkpoint/language. |
| `hidden_size` | 192 | 192 | Encoder channels and WaveNet/filter channels. |
| `num_hidden_layers` | 6 | 6 | Text encoder repeat count. |
| `num_attention_heads` | 2 | 2 | MHA with `head_dim=96`. |
| `window_size` | 4 | 4 | Relative attention embedding window. `None`/0 would remove rel-pos paths. |
| `ffn_dim` | 768 | 768 | Conv1d feed-forward inner width. |
| `flow_size` | 192 | 192 | Prior/flow/vocoder spectrogram channel count. |
| `spectrogram_bins` | 513 | 513 | Training posterior spectrogram input only for first inference target. |
| `use_stochastic_duration_prediction` | true | true | Selects flow-based stochastic predictor instead of simpler deterministic conv predictor. |
| `duration_predictor_num_flows` | 4 | 4 | Flow count in stochastic duration predictor. |
| `duration_predictor_flow_bins` | 10 | 10 | Rational-quadratic spline bin count. |
| `prior_encoder_num_flows` | 4 | 4 | Reverse residual coupling flow count after duration expansion. |
| `posterior_encoder_num_wavenet_layers` | 16 | 16 | Training-only posterior path. |
| `num_speakers` | 1 | 1, 109 | Multi-speaker path adds speaker embedding and conditioning convs. |
| `speaker_embedding_size` | 0 | 0, 256 | Enables conditioning in duration predictor, flow WaveNet, posterior encoder, vocoder. |
| `upsample_rates` | `[8,8,2,2]` | same | Waveform length multiplier is product 256. |
| `sampling_rate` | 16000 | 16000, 22050 | Output audio metadata and checkpoint parity. |
| `noise_scale`, `noise_scale_duration` | 0.667, 0.8 | same | Runtime RNG amplitude for latents and durations. |

Checkpoint sweep:

| Model id | Vocab | Speakers | Speaker emb | Rate | Topology notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `facebook/mms-tts-eng` | 38 | 1 | 0 | 16000 | MMS character-ish tokenizer, `phonemize=false`, pad token `"k"`. |
| `facebook/mms-tts-spa` | 45 | 1 | 0 | 16000 | Same operator structure; language/vocab variation only in fetched config. |
| `facebook/mms-tts-deu` | 45 | 1 | 0 | 16000 | Same operator structure; language/vocab variation only in fetched config. |
| `facebook/mms-tts-fra` | 44 | 1 | 0 | 16000 | Same operator structure; language/vocab variation only in fetched config. |
| `kakao-enterprise/vits-ljs` | 178 | 1 | 0 | 22050 | Same graph, larger phoneme vocab. |
| `kakao-enterprise/vits-vctk` | 178 | 109 | 256 | 22050 | Multi-speaker conditioning paths are required. |

## 3a. Family variation traps

- VITS is not KV-cache generation. The pipeline calls `model(**inputs)` because `VitsModel` is forward-only.
- The tokenizer is model-coupled. MMS English has `phonemize=false`; VCTK has `phonemize=true` and depends on the optional Python `phonemizer` package with espeak backend.
- `pad_token_id` can be absent from config while tokenizer supplies a pad token; runtime graph receives explicit `attention_mask`, and embedding padding behavior may differ if config lacks `pad_token_id`.
- Multi-speaker VCTK enables extra conv conditioning paths. A single-speaker integration must reject or separately admit `speaker_id` and `speaker_embedding_size != 0`.
- Output length is dynamic and data-dependent: predicted durations are `ceil(exp(log_duration) * mask / speaking_rate)`, summed, clamped to at least 1, then multiplied by `prod(upsample_rates)`.
- Stochastic inputs are required for source parity: duration predictor reverse samples `[B,2,T_text]`; acoustic prior samples like `prior_means`.
- `use_stochastic_duration_prediction=false` is implemented but not observed in the representative configs; it swaps the duration predictor to a simpler deterministic Conv1d stack.
- Training-only posterior encoder exists and samples with `randn_like`, but `labels` inference/training path is intentionally not implemented in HF source.
- Layout is axis-sensitive. Source alternates `[B,T,C]` in encoder/attention with `[B,C,T]` for Conv1d/WaveNet/vocoder. Do not apply blanket channel-last translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, `transpose`, `permute`, `reshape/view`, `contiguous`, `split`, `cat`, `flip(dim=1)`, `squeeze`, `unsqueeze`.
- Dynamic `arange`, `cumsum`, `ceil`, `sum`, `clamp_min`, comparisons, boolean masks, dtype casts.
- `pad` for 1D/flattened relative-position shifts and asymmetric feed-forward padding.
- `matmul`/`bmm` for attention and duration expansion alignment.
- Advanced boolean indexing/scatter-like writes inside rational-quadratic spline source (`outputs[mask] = ...`) unless rewritten to masked elementwise/gather form.

Neural network primitives:

- `Linear(192 -> 192)` Q/K/V/O with optional bias.
- `Conv1d` pointwise, standard, depthwise grouped, dilated, and `ConvTranspose1d`.
- `LayerNorm` over channel-last temporary layouts for Conv1d outputs.
- Activations: `relu`, `gelu`, `leaky_relu`, `tanh`, `sigmoid`, `softplus`, `logsigmoid`, `exp`, `log`, `sqrt`, `pow`, `softmax`.
- Dropout should be disabled in `eval`; LayerDrop is training-only.

Attention primitives:

- Noncausal self-attention only, MHA with `num_heads=2`, `head_dim=96`.
- Relative position key/value additions with local embedding table but dense `[T,T]` attention scores.
- No causal mask, no cross-attention use in current source, no KV cache.

Preprocessing-coupled ops:

- Text normalization/lowercasing, language-specific Romanian character mapping, optional uroman romanization, optional phonemizer/espeak phonemization, vocab filtering, blank insertion with token id 0/pad symbol.
- These are CPU/data-pipeline responsibilities for first DinoML integration.

Optional/deferred codec/diffusion/vocoder generation ops:

- HiFi-GAN vocoder is not optional for end-to-end VITS waveform parity; it is the decoder inside `VitsModel`.
- Posterior encoder/vocoder training losses and discriminator-side behavior are deferred.

## 5. Layer/block breakdown

Text encoder:

```text
input_ids [B,T_text]
padding_mask [B,T_text,1]
x = Embedding(vocab,192)(input_ids) * sqrt(192)
repeat 6:
  attn_mask = noncausal 4D mask [B,1,T_text,T_text]
  q,k,v = Linear(192 -> 192) with bias
  scores = bmm(q, k^T) + relative_key_bias + mask
  probs = softmax(scores)
  attn = bmm(probs, v) + relative_value_bias
  x = LayerNorm(x + Linear(attn))
  ff = Conv1d(192 -> 768, kernel=3, same-ish pad) -> relu -> Conv1d(768 -> 192, kernel=3)
  x = LayerNorm(x + ff)
stats = Conv1d(192 -> 384, kernel=1)(x^T)^T * mask
prior_means, prior_log_variances = split(stats, 192, dim=2)
```

Stochastic duration predictor, inference reverse path:

```text
h [B,192,T_text] = detach(text_hidden^T)
h = Conv1d(192 -> 192, 1)
if speaker: h += Conv1d(256 -> 192, 1)(speaker_embedding [B,256,1])
h = depthwise-dilated DDS conv stack + pointwise conv + LayerNorm/GELU
h = Conv1d(192 -> 192, 1) * padding_mask
z = randn([B,2,T_text]) * noise_scale_duration
for selected reverse flows:
  z = flip(z, dim=1)
  z = ElementwiseAffine or ConvFlow reverse with spline
log_duration = split(z, [1,1], dim=1)[0]
```

Duration expansion and prior sampling:

```text
duration = ceil(exp(log_duration) * input_padding_mask / speaking_rate)
predicted_lengths = clamp_min(sum(duration over channel/text), 1).long()
output_padding_mask [B,1,T_spec] from arange(max(predicted_lengths))
attn [B,1,T_spec,T_text] from cumsum(duration), arange, pad/diff, and masks
expanded_mean/logvar [B,192,T_spec] = matmul(attn.squeeze(1), prior_*)^T
prior_latents = expanded_mean + randn_like(expanded_mean) * exp(expanded_logvar) * noise_scale
```

Reverse flow:

```text
repeat 4 reversed residual coupling layers:
  z = flip(z, dim=1)
  first, second = split(z, [96,96], dim=1)
  h = Conv1d(96 -> 192, 1)(first) * mask
  h = WaveNet 4 layers with gated tanh/sigmoid residual-skip convs
  mean = Conv1d(192 -> 96, 1)(h) * mask
  second = second - mean
  z = cat(first, second)
spectrogram = z * output_padding_mask
```

HiFi-GAN vocoder:

```text
x = Conv1d(192 -> 512, kernel=7, pad=3)(spectrogram)
if speaker: x += Conv1d(256 -> 512, 1)(speaker_embedding)
for stages i=0..3 with rates [8,8,2,2]:
  x = leaky_relu(x, slope=0.1)
  x = ConvTranspose1d(C_i -> C_i/2, kernel=[16,16,4,4][i], stride=rate, pad=(kernel-rate)//2)
  x = average of 3 residual blocks with kernels [3,7,11] and dilations [1,3,5]
waveform = tanh(Conv1d(final_channels -> 1, kernel=7, pad=3, bias=false)(leaky_relu(x)))
waveform = squeeze channel -> [B,T_audio]
```

## 6. Attention requirements

Required attention is encoder-style, noncausal self-attention:

- MHA, not MQA/GQA: `num_attention_heads=2`, key/value heads equal query heads, `head_dim=hidden_size/heads=96`.
- Query, key, value, output widths are all 192. Projection bias follows `use_bias`.
- Attention mask is additive 4D noncausal mask `[B,1,T,T]` derived from `attention_mask`.
- Relative key path computes `matmul(query_states, rel_k^T)` then converts relative positions to absolute `[B*H,T,T]` by pad/view/slice.
- Relative value path converts absolute attention probs to relative weights, then `matmul(relative_weights, rel_v)` and adds to attention output.
- No packed/varlen support, no sliding-window sparse attention despite the small relative embedding window, no decode cache.
- FlashAttention/SDPA can cover only the base dense QK softmax V path if relative K/V terms are fused or disabled; source uses explicit `bmm`, so first parity should implement dense attention plus relative-bias transforms directly.

## 7. Position encoding and custom math

Relative attention embedding:

```python
rel = emb_rel[:, slice_start:slice_end]  # length 2*T-1 after optional pad
relative_logits = matmul(q, rel.transpose(-2, -1))
scores += relative_to_absolute(relative_logits)
relative_weights = absolute_to_relative(attn_probs)
attn_output += matmul(relative_weights, rel_v)
```

Spline flow used by stochastic duration ConvFlow:

```python
inside = (x >= -tail_bound) & (x <= tail_bound)
widths = softmax(width_logits, dim=-1)
widths = min_width + (1 - min_width * bins) * widths
cumwidths = pad(cumsum(widths), left=1) * (hi - lo) + lo
heights = softmax(height_logits, dim=-1)
derivatives = min_derivative + softplus(derivative_logits_padded)
bin_idx = sum(x[..., None] >= selected_cum_bins, dim=-1) - 1
gather per-bin width/height/derivative values
apply rational-quadratic forward or inverse formula
outside interval: identity
```

Precomputable: relative embedding slice shapes for fixed `T`, vocoder/conv weights, and static spline constants. Dynamic: masks, durations, output length, alignment matrix, stochastic samples, and spline bin indices.

## 8. Preprocessing and input packing

Tokenizer contract:

- `VitsTokenizer` consumes text and emits `input_ids` and `attention_mask`.
- `normalize=true` lowercases while preserving exact vocab/special-token matches; if `phonemize=false`, it filters characters outside the vocab.
- Optional `is_uroman=true` romanizes non-ASCII text if the optional `uroman` package is installed.
- Optional `phonemize=true` requires `phonemizer` and espeak; the source hardcodes `language="en-us"` for phonemizer even when tokenizer `language` differs.
- `_tokenize` operates at character/token string level and, when `add_blank=true`, inserts token id 0 between all tokens and at both ends.
- Pipeline output is CPU NumPy waveform plus `sampling_rate` resolved from model config.

No waveform/audio input preprocessing is used for inference. `spectrogram_bins`, posterior encoder, and spectrogram labels are training-only for this source basis.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d kernel=1 to GEMM

Source pattern: pointwise `Conv1d(C_in -> C_out, kernel=1)` in projections, conditioning, and flow/vocoder pre/post paths.

Replacement:

```text
[B,C,T] -> transpose to [B*T,C] -> GEMM(weight.T) + bias -> [B,C_out,T]
```

Preconditions: stride 1, padding 0, dilation 1, groups 1, contiguous or known-stride `[B,C,T]`. Weight transform from PyTorch Conv1d `[C_out,C_in,1]` to GEMM RHS `[C_in,C_out]`. Failure cases: grouped/depthwise convs and `ConvTranspose1d`.

Parity test: compare random fp32/fp16 tensors for each pointwise conv module shape with exact source module weights.

### Rewrite: Encoder FFN Conv1d kernel=3 to local-im2col GEMM

Source pattern: `pad -> Conv1d(192->768,k=3) -> relu -> Conv1d(768->192,k=3)`, with `[B,T,C] <-> [B,C,T]` transposes.

Replacement: guarded static/local 1D convolution lowering via direct Conv1d provider or im2col+GEMM.

Preconditions: stride 1, dilation 1, groups 1, source padding `[1,1]` for kernel 3, channel-first logical conv region preserved. Layout constraints: either keep `[B,C,T]` inside this region or rewrite all adjacent transpose/LayerNorm axes together.

Failure cases: dynamic padding variants, non-kernel-3 configs, or fusing across attention residual/norm boundaries without axis rewrite.

### Rewrite: Duration alignment construction to segment-repeat/prefix-sum kernel

Source pattern: `ceil(exp(log_duration)) -> cumsum -> arange comparisons -> pad/diff -> matmul(attn, prior)`.

Replacement: generate expanded prior means/logvars by repeating each text timestep for its integer duration, avoiding materializing `[B,T_spec,T_text]` attention.

Preconditions: durations are nonnegative integer after `ceil`, per-batch output length bounded by caller/runtime max, no need to return `attn`. Shape equation: `T_spec_b = max(1, sum_i duration[b,i])`; output audio length `T_audio_b = T_spec_b * prod(upsample_rates)`.

Failure cases: if attentions/debug output for alignment is required, or if batch examples need ragged output without a max buffer/report contract.

Parity test: compare expanded means/logvars to source `matmul(attn, prior)` for random durations including zeros and masked text positions.

### Rewrite: Relative position transforms to fused skew kernels

Source pattern: relative-to-absolute and absolute-to-relative use pad/view/slice sequences.

Replacement: dedicated skew/unskew kernel or attention-relative-bias epilogue.

Preconditions: dense attention, fixed `T` per batch bucket, rel embedding length `2*T-1` after source padding/slicing. Failure cases: output attentions requiring exact intermediate shape reuse should still match values.

### Rewrite: Weight norm removal/folding

Source modules may contain parametrized weight norm in converted checkpoints for WaveNet/vocoder pieces. In eval, fold `weight_g`/`weight_v` to dense Conv weights or require `remove_weight_norm()` before export.

Preconditions: frozen inference weights. Failure cases: unfused parametrization state not materialized by loader.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d/ConvTranspose1d provider coverage for `[B,C,T]`, including dilation, depthwise groups, and transposed upsampling. This dominates flow/vocoder work.
- Duration expansion prefix-sum/repeat kernel. It removes a large temporary alignment matrix and handles dynamic output shape honestly.
- HiFi-GAN residual block fusion: leaky_relu + conv + leaky_relu + conv + residual for repeated small kernels.
- WaveNet gated block fusion: conv + optional conditioning slice + `tanh * sigmoid` + residual/skip split.

Medium priority:

- Dense encoder MHA with relative key/value bias. Encoder is small, but correctness is custom.
- LayerNorm around Conv1d with transpose elimination under a local layout guard.
- Pointwise Conv1d to GEMM for projector/conditioning paths.
- Stochastic duration ConvFlow spline kernel to avoid boolean indexing and many small gathers.

Lower priority:

- Output attentions and hidden states. Useful for debugging, not first waveform parity.
- Deterministic duration predictor path, unless a checkpoint with `use_stochastic_duration_prediction=false` is targeted.
- Posterior encoder training path.

## 11. Runtime staging plan

Stage 1: parse config/tokenizer metadata and load weights for `facebook/mms-tts-eng`; reject training labels and multi-speaker configs initially.

Stage 2: implement text encoder parity on token IDs and masks, including relative attention transforms and Conv1d FFN.

Stage 3: implement stochastic duration predictor reverse path with explicit RNG tensors supplied by runtime for deterministic parity.

Stage 4: implement duration expansion with dynamic shape reporting for spectrogram/audio lengths.

Stage 5: implement reverse flow and validate spectrogram parity before vocoder.

Stage 6: implement HiFi-GAN vocoder and end-to-end waveform parity for `facebook/mms-tts-eng`.

Stage 7: add multi-speaker conditioning for `kakao-enterprise/vits-vctk`.

Stage 8: optimize with Conv1d/GEMM rewrites, fused WaveNet/HiFi-GAN blocks, and compact duration-repeat lowering.

Stub initially: tokenizer phonemizer/uroman in DinoML runtime, output attentions, hidden-state returns, posterior encoder, training losses, and pipeline postprocessing beyond returning waveform plus sampling rate.

## 12. Parity and validation plan

- Tokenizer snapshots: fixed English strings from HF tests, with and without normalization; VCTK phonemizer path should be a CPU-pipeline test gated on optional dependency.
- Text encoder: random `input_ids` and masks, compare `last_hidden_state`, `prior_means`, `prior_log_variances`.
- Relative attention helpers: direct tests for `relative_to_absolute` and `absolute_to_relative` over small `T`.
- Duration predictor reverse: provide fixed RNG samples or fixed seed and compare `log_duration`.
- Duration expansion: compare predicted lengths, attention-derived expanded priors, and sequence length reports.
- Flow reverse: compare spectrogram latents from fixed prior sample.
- Vocoder: compare waveform from a fixed spectrogram and speaker embedding.
- End-to-end: reproduce HF integration shape `(1,87040)` for the quoted MMS English sentence at seed 555, then compare a waveform slice. Source tests use `rtol=1e-4, atol=1e-4` on CPU fp32/fp16 slices; DinoML CUDA fp16 may need a looser staged tolerance until ConvTranspose and stochastic sampling are stabilized.

## 13. Performance probes

- Tokenizer/frontend throughput separately from neural runtime, especially phonemizer/espeak.
- Text encoder latency by `T_text` and batch size.
- Duration predictor reverse latency and spline/gather cost.
- Duration expansion temporary memory: source attention matrix versus compact repeat kernel.
- Reverse flow latency by `T_spec`.
- HiFi-GAN vocoder throughput by `T_spec`, sample rate, and batch size.
- End-to-end waveform samples/sec and requests/sec across short/medium/long texts.
- Dynamic output allocation/report overhead for ragged per-batch waveform lengths.
- Multi-speaker conditioning overhead for VCTK.

## 14. Skip/defer list

- Training, posterior encoder losses, discriminators, mel/spectrogram label ingestion.
- Autoregressive generation machinery, beam search, KV cache, speculative decode.
- Output attentions/hidden states for production path.
- `use_stochastic_duration_prediction=false` until a real checkpoint requires it.
- Phonemizer/uroman inside GPU graph; keep as CPU/data-pipeline work.
- Quantization and packed weights; representative checkpoints are F32 safetensors.
- Multi-GPU/tensor parallel.
- Full ragged waveform return without max-buffer and output-shape-report support; first integration can require bounded max output length.

## 15. Final implementation checklist

- [ ] Parse `VitsConfig` and tokenizer metadata.
- [ ] Load VITS weights, including optional folded weight norm.
- [ ] Implement embedding and noncausal text attention with relative K/V position paths.
- [ ] Implement Conv1d, depthwise/dilated Conv1d, and ConvTranspose1d coverage for `[B,C,T]`.
- [ ] Implement LayerNorm with local transpose/layout guards.
- [ ] Implement stochastic duration predictor reverse path with explicit RNG input contract.
- [ ] Implement rational-quadratic spline helper or guarded lowering rewrite.
- [ ] Implement duration prefix-sum/repeat expansion and dynamic output shape reporting.
- [ ] Implement reverse residual coupling flow.
- [ ] Implement HiFi-GAN vocoder.
- [ ] Add single-speaker `facebook/mms-tts-eng` module parity tests.
- [ ] Add multi-speaker `kakao-enterprise/vits-vctk` admission and parity tests.
- [ ] Benchmark text encoder, duration expansion, flow, vocoder, and end-to-end waveform throughput separately.
