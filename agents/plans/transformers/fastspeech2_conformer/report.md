# FastSpeech2 Conformer Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: espnet/fastspeech2_conformer, espnet/fastspeech2_conformer_hifigan, espnet/fastspeech2_conformer_with_hifigan
Config source: HF raw config.json for the three repos above, fetched 2026-05-13
Source files inspected:
  X:/H/transformers/src/transformers/models/fastspeech2_conformer/configuration_fastspeech2_conformer.py
  X:/H/transformers/src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py
  X:/H/transformers/src/transformers/models/fastspeech2_conformer/tokenization_fastspeech2_conformer.py
  X:/H/transformers/tests/models/fastspeech2_conformer/test_modeling_fastspeech2_conformer.py
  X:/H/transformers/src/transformers/models/fastspeech2_conformer/convert_*.py
Any missing files or assumptions:
  No processor or feature extractor file exists for this family. Text normalization and phonemization live in the tokenizer and depend on g2p_en. Acoustic training labels require an external audio/alignment pipeline; the tests explicitly use dummy pitch/energy/duration/spectrogram labels for training-path coverage.
```

Primary runtime target for DinoML: inference for `FastSpeech2ConformerModel`, producing log-mel spectrograms from phoneme token IDs. The optional `FastSpeech2ConformerHifiGan` vocoder boundary is documented separately and should be a later or separately admitted stage. `FastSpeech2ConformerWithHifiGan` is a wrapper that calls both.

Relevant source links:

- [modeling_fastspeech2_conformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/fastspeech2_conformer/modeling_fastspeech2_conformer.py)
- [configuration_fastspeech2_conformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/fastspeech2_conformer/configuration_fastspeech2_conformer.py)
- [tokenization_fastspeech2_conformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/fastspeech2_conformer/tokenization_fastspeech2_conformer.py)
- [espnet/fastspeech2_conformer config](https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json)
- [espnet/fastspeech2_conformer_hifigan config](https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json)
- [espnet/fastspeech2_conformer_with_hifigan config](https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json)

## 2. High-level architecture

This is non-autoregressive text-to-speech, not language generation. There is no decode loop, logits head, sampling, or KV cache. The acoustic model maps phoneme IDs to mel frames through:

```text
text cleanup + g2p_en phonemization -> phoneme IDs/attention_mask
  -> token embedding + relative positional encoding
  -> Conformer encoder
  -> optional speaker/language conditioning
  -> duration, pitch, energy predictors
  -> pitch/energy embedding add
  -> duration-driven length regulator
  -> Conformer decoder over expanded frame sequence
  -> linear mel projection + convolutional postnet
  -> log-mel spectrogram
  -> optional HiFi-GAN vocoder -> waveform
```

Stage decomposition:

- CPU/data pipeline: tokenizer regex cleanup, uppercase conversion, `g2p_en.G2p()`, vocabulary lookup, padding mask construction. No waveform feature extraction is owned by the inference path.
- Acoustic GPU/runtime candidate: token embedding, Conformer encoder, variance predictors, length regulator, Conformer decoder, mel projection/postnet.
- Vocoder GPU/runtime candidate: HiFi-GAN 1D convolution/transpose-convolution stack that consumes `[B, T_mel, 80]` and emits waveform `[B, T_audio]`.
- Independently validatable boundaries: encoder hidden states; duration/pitch/energy outputs; length-regulated hidden states and output frame length; mel spectrogram; vocoder waveform.

## 3. Important config dimensions

Representative Transformers-native checkpoint sweep:

| Config source | Architecture | Scope | Key dimensions |
| --- | --- | --- | --- |
| `espnet/fastspeech2_conformer` | `FastSpeech2ConformerModel` | acoustic model | `hidden_size=384`, `vocab_size=78`, `num_mel_bins=80`, encoder/decoder layers `4/4`, heads `2/2`, FF conv units `1536`, reduction factor `1` |
| `espnet/fastspeech2_conformer_hifigan` | `FastSpeech2ConformerHifiGan` | vocoder only | `model_in_dim=80`, upsample initial channels `512`, rates `[8,8,2,2]`, kernels `[16,16,4,4]`, resblock kernels `[3,7,11]` |
| `espnet/fastspeech2_conformer_with_hifigan` | `FastSpeech2ConformerWithHifiGan` | wrapper | embeds the same acoustic config plus vocoder config; `torch_dtype=float32` |

Acoustic config fields:

| Field | Value in official acoustic config | Runtime impact |
| --- | ---: | --- |
| `hidden_size` | 384 | embedding width, attention width, Conformer conv channels |
| `vocab_size` | 78 | phoneme vocabulary size |
| `num_mel_bins` | 80 | final spectrogram channel count and vocoder input width |
| `encoder_layers` / `decoder_layers` | 4 / 4 | repeated Conformer blocks |
| `encoder_num_attention_heads` / `decoder_num_attention_heads` | 2 / 2 | MHA heads; `head_dim=192` |
| `encoder_linear_units` / `decoder_linear_units` | 1536 / 1536 | feed-forward Conv1d intermediate channels |
| `positionwise_conv_kernel_size` | 3 | FFN replacement Conv1d kernels |
| `encoder_kernel_size` / `decoder_kernel_size` | 7 / 31 | Conformer depthwise conv kernels |
| `use_macaron_style_in_conformer` | true | adds pre-attention half-scale FF conv block |
| `use_cnn_in_conformer` | true | adds GLU + depthwise conv module |
| `duration_predictor_layers/channels/kernel` | 2 / 256 / 3 | duration predictor Conv1d stack |
| `pitch_predictor_layers/channels/kernel` | 5 / 256 / 5 | pitch predictor Conv1d stack |
| `energy_predictor_layers/channels/kernel` | 2 / 256 / 3 | energy predictor Conv1d stack |
| `pitch_embed_kernel_size` / `energy_embed_kernel_size` | 1 / 1 | Conv1d projection from scalar variance to hidden width |
| `speaking_speed` | 1.0 | scales predicted durations before repeat expansion if not 1.0 |
| `num_speakers`, `num_languages`, `speaker_embed_dim` | null | optional conditioning branches absent in official configs |

## 3a. Family variation traps

- Duration inference is value-dependent and discrete: `exp -> subtract 1 -> round -> clamp(min=0) -> long`, then `repeat_interleave`. Small numeric differences can change output length; upstream tests skip batching equivalence for this reason.
- `length_regulator` allocates output length from the max per-batch duration sum. This is a dynamic, data-dependent shape and the biggest first-integration gap.
- Padding masks are axis-sensitive. Source attention expects `attention_mask` shaped `[B, 1, T]`, then broadcasts to scores `[B, H, T, T]`. Conformer conv masking derives all-masked rows by reducing the mask on `dim=2`.
- All sequence tensors are semantic `[B, T, C]` around attention and layernorm. Conv1d regions transpose to `[B, C, T]`, run Conv1d/BatchNorm1d, then transpose back. A layout pass can optimize these local conv islands but must not silently reinterpret model-wide axes.
- `normalize_before`, `concat_after`, `use_macaron_style_in_conformer`, `use_cnn_in_conformer`, and `convolution_bias` materially change block topology. Official configs use post-norm, no concat-after, macaron on, CNN on, conv bias on.
- Config validation requires odd kernels for positionwise conv, encoder/decoder conv, duration/pitch/energy predictors, and pitch/energy embeddings. DinoML should reject even kernels up front.
- Optional conditioning adds speaker ID embeddings, language ID embeddings, or normalized external speaker embeddings plus a projection from `hidden_size + speaker_embed_dim` to `hidden_size`.
- Historical/config gap: the current `FastSpeech2ConformerHifiGanConfig` source declares `model_type="fastspeech2_conformer_hifigan"`, but the public standalone and nested vocoder configs contain `"model_type": "hifigan"`. DinoML should admit by architecture/class context, not by raw `model_type` alone, or explicitly route this legacy value.
- Historical fields in HF configs such as `input_dim`, generation fields, `sampling_rate`, and `tie_word_embeddings` are not read by the inspected modeling forward path. Treat them as metadata unless a loader uses them.
- Several older ESPnet Hub repos named `*_conformer_fastspeech2` are ESPnet-library models and do not expose Transformers `config.json`; they are out of scope for this Transformers-native audit.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,T_text] -> [B,T_text,384]`, padding index 0.
- Transpose `[B,T,C] <-> [B,C,T]` around Conv1d and BatchNorm1d.
- View/reshape for attention heads: `[B,T,384] -> [B,T,2,192]`, output contiguous view back to `[B,T,384]`.
- Cat on feature axis for optional `concat_after` and speaker embedding projection.
- Mask unsqueeze, boolean invert, equality compare, `masked_fill`, `masked_select` only for training loss.
- Dynamic zeros allocation and per-example copy for length regulation.
- `repeat_interleave` equivalent over time with per-token durations.

Neural network primitives:

- Linear `384 -> 384` for Q/K/V/out and relative-position projection.
- Linear optional concat-after `768 -> 384`.
- Linear optional speaker projection `384 + speaker_embed_dim -> 384`.
- Conv1d FFN replacement: `384 -> 1536`, ReLU, dropout, `1536 -> 384`, kernel 3, same padding.
- LayerNorm over channel axis for Conformer residual points and predictor layers.
- Conv1d predictor layers: first `384 -> 256`, then `256 -> 256`, kernel 3/5, ReLU, LayerNorm, dropout.
- Predictor linear heads: duration `256 -> 1` squeezed to `[B,T]`; pitch/energy `256 -> 1` kept `[B,T,1]`.
- Variance embedding Conv1d: `1 -> 384`, kernel 1 in official config.
- Mel projection: Linear `384 -> 80 * reduction_factor`, then view to `[B, T_frame * reduction_factor, 80]`.
- Postnet: five Conv1d + BatchNorm1d layers over mel channels, first `80 -> 256`, middle `256 -> 256`, final `256 -> 80`, kernel 5, bias false, Tanh except final.

Attention primitives:

- Dense noncausal self-attention only, no cross-attention, no KV cache.
- Relative positional attention with learned `pos_bias_u/v` and linear positional projection.
- Two matmuls for content and position scores, relative shift, mask fill to dtype min, softmax over key/time axis, optional post-softmax mask zeroing, attention-value matmul.

Position/custom math:

- Sin/cos relative table of length `2*T-1`, with positive positions reversed and negative positions concatenated.
- Input hidden states are scaled by `sqrt(hidden_size)` before positional dropout.

Preprocessing-coupled ops:

- Tokenizer regex substitutions, uppercase conversion, `g2p_en` phonemization, optional stripping of space tokens, append `<sos/eos>`.
- No audio feature extractor for acoustic inference; training labels come from an external aligner/feature pipeline.

Vocoder ops, optional/deferred for first acoustic target:

- Conv1d pre-net `80 -> 512`, kernel 7.
- Four ConvTranspose1d upsamplers with rates `[8,8,2,2]` and kernel sizes `[16,16,4,4]`.
- For each upsample stage, three residual blocks with kernel sizes `[3,7,11]`; each block has dilated Conv1d layers with dilations `[1,3,5]` followed by same-kernel dilation-1 convs.
- LeakyReLU, residual adds, average across resblocks, Conv1d post `channels -> 1`, Tanh, squeeze/flatten.

## 5. Layer/block breakdown

Acoustic inference:

```text
input_ids [B,T_text]
attention_mask [B,T_text] default ones
text_masks = attention_mask[:, None, :]  # [B,1,T_text]

x = Embedding(vocab_size, 384)(input_ids)
x, pos = RelPositionalEncoding(x)        # x [B,T_text,384], pos [1,2*T_text-1,384]
repeat encoder layer 4 times

optional:
  x += speaker_id_embedding(speaker_ids)[:, None, :]
  x += language_id_embedding(lang_ids)[:, None, :]
  x = Linear(cat(x, normalize(speaker_embedding).expand(T_text)), 384)

duration = DurationPredictor(x)          # [B,T_text], integer durations in eval
pitch = VariancePredictor(x, mask)       # [B,T_text,1]
energy = VariancePredictor(x, mask)      # [B,T_text,1]
x = x + Conv1d(pitch) + Conv1d(energy)
x = LengthRegulator(x, duration)         # [B,T_frame,384], dynamic T_frame

x, pos = RelPositionalEncoding(x)
repeat decoder layer 4 times
mel_before = Linear(384, 80 * r)(x).view(B, -1, 80)
mel_after = mel_before + Postnet(mel_before)
```

Conformer layer, official topology:

```text
if macaron:
  x = x + 0.5 * Dropout(ConvFFN(LayerNorm? or post-LN branch))

attn = RelPosSelfAttention(LayerNorm? x, pos, mask)
x = x + Dropout(attn)                    # or concat_after Linear([x, attn])
x = LayerNorm(x)                         # official post-norm

conv = PointwiseConv1d(384,768) -> GLU(C axis)
conv = masked_fill(padded rows, 0)
conv = DepthwiseConv1d(384,384,k=7 encoder or 31 decoder, groups=384)
conv = BatchNorm1d -> SiLU -> PointwiseConv1d(384,384)
x = LayerNorm(x + Dropout(conv))

x = LayerNorm(x + 0.5 * Dropout(ConvFFN(x)))
x = final LayerNorm(x)
```

The source uses the same class for encoder and decoder; only input embedding presence and module config differ.

## 6. Attention requirements

Attention is required, but only as dense noncausal encoder-style self-attention.

| Property | Requirement |
| --- | --- |
| Type | self-attention in encoder and expanded-frame decoder |
| Causality | noncausal |
| Heads | MHA, `H=2`, `head_dim=hidden_size/H=192` in official configs |
| Q/K/V width | all `384 -> 384`, value width equals key width |
| Query/key lengths | square within each layer: text length for encoder, duration-expanded frame length for decoder |
| Mask | source passes `[B,1,T]`; attention converts zeros to a broadcast key mask |
| Relative bias | Transformer-XL-style relative positional term with learned per-head `u/v` biases |
| Cache | no KV cache, no generation cache |
| Flash/SDPA | not a drop-in match because of the `matrix_bd` relative-position term and shift; a custom relative-attention fused kernel would be needed |

Attention math:

```text
q,k,v = Linear(x).view(B,T,H,D)
p = Linear(pos_emb).view(1,2*T-1,H,D)
AC = matmul((q + u).transpose(1,2), k.permute(0,2,3,1))
BD = matmul((q + v).transpose(1,2), p.permute(0,2,3,1))
BD = relative_shift(BD)[:, :, :, :T]
scores = (AC + BD) / sqrt(D)
scores = masked_fill(mask == 0, finfo_min)
weights = softmax(scores, dim=-1)
weights = masked_fill(mask == 0, 0)
out = Linear(matmul(weights, v.transpose(1,2)).transpose(1,2).reshape(B,T,384))
```

## 7. Position encoding and custom math

Relative positional encoding is deterministic for a given length/dtype/device and can be precomputed or generated at compile/runtime bucket boundaries. It is not RoPE.

```python
def rel_pos_table(T, C):
    pos = arange(T).float()[:, None]
    div = exp(arange(0, C, 2).float() * -(log(10000.0) / C))
    pos_pos[:, 0::2] = sin(pos * div)
    pos_pos[:, 1::2] = cos(pos * div)
    pos_neg[:, 0::2] = sin(-pos * div)
    pos_neg[:, 1::2] = cos(-pos * div)
    return cat([flip(pos_pos, [0])[None], pos_neg[1:][None]], dim=1)
```

The relative shift is a reshape/slice trick over `[B,H,T,2*T-1]`; DinoML can lower it as a specialized view/copy/gather with the source shape guard `last_dim == 2*T-1`.

Duration and length regulation custom math:

```python
duration = clamp(round(exp(log_duration) - 1.0), min=0).long()
if speaking_speed != 1.0:
    duration = round(duration.float() * speaking_speed).long()
if all durations in an example are zero:
    duration[row] = 1
expanded = repeat_each_token_by_duration(x, duration)
```

For parity, duration rounding and the all-zero fallback must match exactly.

## 8. Preprocessing and input packing

Tokenizer contract:

- Requires `g2p_en`.
- Text cleanup replaces semicolon/colon with comma, hyphen with space, ampersand with `and`, removes `()[]<>"`, collapses whitespace, and uppercases.
- `g2p_en.G2p()` emits phoneme/string tokens; tokenizer optionally strips space tokens. Official tokenizer config has `should_strip_spaces=true`.
- Appends `<sos/eos>` after tokenization.
- Official vocab has 78 entries, `<blank>=0`, `<unk>=1`, `<sos/eos>=77`.
- Model inputs are `input_ids [B,T_text]` and `attention_mask [B,T_text]`.

No HF processor/preprocessor exists. For the acoustic inference target, waveform/audio preprocessing is not part of the graph. For training parity, duration/pitch/energy labels and mel spectrogram labels come from external alignment and feature extraction, not from this Transformers family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d kernel-1 variance embedding -> Linear

Source pattern: pitch/energy embedding `Conv1d(1,384,k=1)` around transpose `[B,T,1] -> [B,1,T] -> [B,384,T] -> [B,T,384]`.

Replacement: per-token Linear `1 -> 384`.

Preconditions:

- `kernel_size == 1`, `padding == 0`, `stride == 1`, `dilation == 1`, `groups == 1`.
- Source tensor is contiguous or lowered with explicit strides preserving `[B,T,1]`.

Weight transform: Conv1d weight `[384,1,1]` becomes Linear weight `[384,1]`; bias unchanged.

Failure cases: kernel size > 1, nonzero padding, or dynamic layout pass that changes time/channel axes without rewriting transposes.

### Rewrite: Positionwise Conv1d FFN -> local temporal convolution provider

Source pattern: transpose, Conv1d `384 -> 1536`, ReLU, Conv1d `1536 -> 384`, transpose back.

Replacement: keep as Conv1d provider initially; optional fused two-conv temporal block later.

Preconditions:

- Odd kernel with same padding.
- Input/output semantic layout `[B,T,C]`.
- Dropout disabled in eval.

Failure cases: treating this as GEMM is only valid for `kernel_size=1`; official kernel is 3, so a general Linear rewrite would be wrong.

### Rewrite: Conformer pointwise Conv1d k=1 -> Linear plus GLU

Source pattern: `[B,T,384] -> transpose -> Conv1d(384,768,k=1) -> GLU(dim=C)`.

Replacement: Linear `384 -> 768`, split channel into `[a,b]`, compute `a * sigmoid(b)`.

Preconditions:

- `kernel_size=1`, no padding, stride 1, groups 1.
- GLU split order matches PyTorch channel split: first half multiplied by sigmoid(second half).

### Rewrite: Relative position table precompute per bucket

Source pattern: runtime table extension from zeros/arange/sin/cos.

Replacement: compile-time or artifact-side constants for admitted max/bucketed text and frame lengths.

Preconditions:

- Known max/bucket lengths and dtype policy.
- Guard if runtime length exceeds precomputed table.

Failure cases: fully unbounded duration-expanded decoder length; must fall back to runtime table generation or reject.

### Rewrite: Length regulator as segmented repeat/indexed gather

Source pattern: Python loop over batch plus `torch.repeat_interleave`.

Replacement: compute prefix sums of durations, allocate `[B,max(sum_d),C]`, gather/scatter token rows to frame rows, zero-pad beyond each example length.

Preconditions:

- Nonnegative integer durations.
- Bounded `max(sum_d)` from runtime guard or compile bucket.
- Exact all-zero fallback.

Failure cases: no admission bound for frame length; output length would be unknown to the runtime allocator.

### Rewrite: HiFi-GAN ConvTranspose1d island

Source pattern: mel spectrogram transpose to `[B,80,T]`, Conv1d, repeated ConvTranspose1d/resblocks, Tanh.

Replacement: separate vocoder backend/kernel family or fallback to delegated PyTorch initially.

Preconditions:

- Vocoder config admitted separately.
- Weight-norm parametrizations removed or materialized before export.
- `model_in_dim == acoustic num_mel_bins` for wrapper.

Failure cases: attempting to fuse vocoder into acoustic graph before dynamic mel length and ConvTranspose1d coverage are stable.

## 10. Kernel fusion candidates

Highest priority:

- Duration predictor and length regulator: this is the unique TTS bottleneck and a correctness gate because it controls decoder length.
- Relative-position self-attention: standard SDPA/FlashAttention cannot directly reproduce the positional `BD` term, so a specialized dense relative attention path is valuable.
- Conformer convolution module: pointwise k=1 + GLU + depthwise Conv1d + BatchNorm + SiLU + pointwise k=1 is repeated in every encoder/decoder layer.
- Conv1d + LayerNorm predictor stacks for duration/pitch/energy.

Medium priority:

- Conv1d FFN replacement block: two same-padding temporal Conv1d layers with ReLU.
- Postnet Conv1d + BatchNorm + Tanh stack over mel channels.
- Relative position table generation and shift as bucketed constants/specialized kernels.

Lower priority:

- Optional speaker/language conditioning branches; absent in official configs.
- HiFi-GAN vocoder fusion; useful for end-to-end latency but should follow acoustic parity and ConvTranspose1d admission.
- Training-only masked losses and weighted masking.

## 11. Runtime staging plan

1. Parse acoustic config and tokenizer metadata; reject non-official topology mutations initially except bounded hidden/head/layer dimensions.
2. Load weights for `espnet/fastspeech2_conformer`; run one encoder Conformer layer parity with fixed `[B,T_text,384]`.
3. Add full encoder parity including relative positional attention and masks.
4. Add duration/pitch/energy predictors. For first inference, allow caller-supplied duration labels or a fixed-duration debug override to decouple decoder shape work.
5. Implement bounded length regulator with explicit max frame length admission and exact rounding/all-zero behavior.
6. Add decoder Conformer and mel projection/postnet parity for mel spectrogram output.
7. Add optional `FastSpeech2ConformerWithHifiGan` by composing the separately admitted HiFi-GAN stage.
8. Optimize fusions and bucketed runtime dispatch for common text/frame length ranges.

Stub initially:

- Training loss path.
- Speaker/language/external speaker embedding branches.
- Standalone vocoder, unless the integration target explicitly requires waveform output.
- Older ESPnet-library checkpoints that are not Transformers-native configs.

## 12. Parity and validation plan

- Unit parity for `length_regulator` covering zero durations, all-zero rows, `speaking_speed != 1.0`, mixed batch sums, and padding tail zeros.
- Unit parity for duration inference math: log-domain output in train mode vs integer linear-domain output in eval mode.
- Relative position table and relative shift parity for `T=1,2,7,205` and dtype fp32/fp16.
- Attention parity for one layer with and without masks; include mask rows where padded keys are all masked.
- Predictor stack parity for duration/pitch/energy with official dimensions and small test dimensions from HF tests.
- Full acoustic model parity on the integration text `"Test that this generates speech"`: expected output shape `[1,205,80]` for `espnet/fastspeech2_conformer` per upstream test.
- Wrapper parity after vocoder admission: expected waveform shape `[1,52480]` for `espnet/fastspeech2_conformer_with_hifigan` per upstream test.

Recommended tolerances: start with fp32 `rtol=2e-4, atol=2e-4` for mel slices to match upstream integration tolerance; introduce fp16/bf16 only after duration rounding parity is stable, because small hidden-state drift can alter output lengths.

## 13. Performance probes

- Tokenizer/g2p throughput separately from model runtime.
- Encoder-only throughput over `B` and `T_text`.
- Duration/pitch/energy predictor throughput and duration-sum distribution.
- Length regulator throughput versus `sum(duration)` and batch padding waste.
- Decoder-only throughput over expanded frame lengths `T_frame`.
- Mel postnet throughput over `T_frame`.
- End-to-end acoustic latency: text IDs to mel spectrogram.
- Optional vocoder latency: mel frames to waveform, separated by ConvTranspose1d stage.
- Bucket sensitivity: common `T_text` and `T_frame` buckets, with output allocation overhead measured explicitly.

## 14. Skip/defer list

- Training losses, masking losses, and weighted masking.
- Gradient checkpointing and dropout behavior beyond eval-mode no-op.
- Beam search/generation helpers; not applicable.
- KV cache and autoregressive decode; not applicable.
- Speaker/language conditioning until a representative config requires it.
- ESPnet-library checkpoints without Transformers `config.json`.
- HiFi-GAN weight norm application/removal APIs for first acoustic target.
- Direct waveform output until acoustic spectrogram parity is proven.

## 15. Final implementation checklist

- [ ] Parse `FastSpeech2ConformerConfig` and reject unsupported topology/config gaps.
- [ ] Load phoneme embedding and acoustic weights with stable parameter names.
- [ ] Implement Conv1d/BatchNorm1d/LayerNorm/Linear/Embedding coverage for `[B,T,C]` acoustic graphs.
- [ ] Implement relative positional table, relative shift, and dense relative self-attention.
- [ ] Implement Conformer convolution module with GLU and depthwise Conv1d.
- [ ] Implement duration predictor eval math exactly.
- [ ] Implement pitch and energy predictors plus variance embedding add.
- [ ] Implement bounded length regulator with explicit max-frame runtime guard.
- [ ] Implement decoder Conformer and mel projection/postnet.
- [ ] Add acoustic end-to-end mel parity for `espnet/fastspeech2_conformer`.
- [ ] Add config admission test for legacy vocoder `model_type="hifigan"` in wrapper context.
- [ ] Separately audit/admit HiFi-GAN ConvTranspose1d vocoder before wrapper waveform parity.
