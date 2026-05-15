# wav2vec2_bert Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/w2v-bert-2.0 as base; CTC finetunes sampled below
Config source: HF config.json/preprocessor_config.json snapshots saved in this folder
Source files inspected:
  transformers/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py
  transformers/src/transformers/models/wav2vec2_bert/modular_wav2vec2_bert.py
  transformers/src/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py
  transformers/src/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py
  transformers/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py
Any missing files or assumptions:
  modeling_wav2vec2_bert.py is generated from modular_wav2vec2_bert.py; future upstream source edits should use the modular file.
  No native raw-waveform conv frontend exists in Wav2Vec2BertModel. Raw audio is converted by SeamlessM4TFeatureExtractor.
  No gated/401 checkpoint was encountered in the representative sweep. facebook/w2v-bert-2.0 has no tokenizer/vocab because it is a base encoder.
```

Primary URLs:

- [facebook/w2v-bert-2.0 config](https://huggingface.co/facebook/w2v-bert-2.0/raw/main/config.json)
- [facebook/w2v-bert-2.0 preprocessor](https://huggingface.co/facebook/w2v-bert-2.0/raw/main/preprocessor_config.json)
- [hf-audio/wav2vec2-bert-CV16-en config](https://huggingface.co/hf-audio/wav2vec2-bert-CV16-en/raw/main/config.json)
- [alvanlii/wav2vec2-BERT-cantonese config](https://huggingface.co/alvanlii/wav2vec2-BERT-cantonese/raw/main/config.json)
- [Xenova/wav2vec2-bert-CV16-en config](https://huggingface.co/Xenova/wav2vec2-bert-CV16-en/raw/main/config.json)
- [spygaurad/wav2vec2-bert config](https://huggingface.co/spygaurad/wav2vec2-bert/raw/main/config.json)

Report target: inference for feature-input audio encoder and CTC ASR head. Classification and x-vector heads are documented as optional head coverage.

## 2. High-level architecture

Wav2Vec2-BERT is an audio feature encoder built from Conformer-style encoder layers. Despite the name, this family is not a text BERT and has no autoregressive decoder or KV cache in the native source.

```text
raw waveform -> CPU/data fbank extractor -> packed input_features [B,T,160]
input_features -> LayerNorm + Linear(160 -> H) -> N conformer encoder layers
encoder output -> optional adapter/subsampler -> task head -> logits/embeddings
```

Stage decomposition:

- CPU/data pipeline: mono/stereo waveform handling, Kaldi-style log-mel fbank, per-mel normalization, padding/truncation, stride-2 frame packing, attention mask packing.
- GPU/runtime base: `input_features` projection, bidirectional encoder attention, conformer convolution modules.
- Optional adapter: strided Conv1d + GLU residual and attention branches, then adapter self-attention and FFN.
- Heads: CTC frame logits are first target. Sequence classification pools time. Audio-frame classification is per-frame linear. XVector uses TDNN/stat-pooling.

## 3. Important config dimensions

Source defaults from `Wav2Vec2BertConfig`:

| Field | Default / observed |
|---|---:|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 16 |
| `head_dim` | 64, inferred as `hidden_size // num_attention_heads` |
| `intermediate_size` | 4096 |
| `feature_projection_input_dim` | 160 |
| `hidden_act` | `swish` |
| `layer_norm_eps` | `1e-5` |
| `position_embeddings_type` | `relative_key` by default; source also supports `relative`, `rotary`, or `None` |
| `relative_key` clip | left 64, right 8 |
| `conv_depthwise_kernel_size` | 31 |
| `add_adapter` | false by default; common CTC finetunes set true |
| `adapter_kernel_size/stride/layers` | 3 / 2 / 1 |
| `output_hidden_size` | defaults to `hidden_size` |
| `vocab_size` | `None` for base; required for CTC |
| cache support | none; encoder-only bidirectional attention |

Representative checkpoint sweep:

| Model | Arch | H/L/A | FFN | Pos | Adapter | Vocab / pad | Preprocessor |
|---|---|---:|---:|---|---|---:|---|
| `facebook/w2v-bert-2.0` | `Wav2Vec2BertModel` | 1024/24/16 | 4096 | `relative_key` | no | none / 0 | 16 kHz, 80 mel, stride 2, pad 1 |
| `hf-audio/wav2vec2-bert-CV16-en` | `Wav2Vec2BertForCTC` | 1024/24/16 | 4096 | `relative_key` | yes | 33 / 0 | 16 kHz, 80 mel, stride 2, pad 1, normalize |
| `alvanlii/wav2vec2-BERT-cantonese` | `Wav2Vec2BertForCTC` | 1024/24/16 | 4096 | `relative_key` | yes | 2699 / 2696 | 16 kHz, 80 mel, stride 2, pad 0 |
| `Xenova/wav2vec2-bert-CV16-en` | `Wav2Vec2BertForCTC` | 1024/24/16 | 4096 | `relative_key` | yes | 33 / 0 | open mirror of CV16-en, dtype omitted |
| `spygaurad/wav2vec2-bert` | `Wav2Vec2BertForCTC` | 1024/24/16 | 4096 | `relative_key` | yes | 66 / 63 | 16 kHz, 80 mel, stride 2, pad 0 |

No true tiny native `wav2vec2-bert` checkpoint was found in the sampled public results. The visible `hf-internal-testing/tiny-random-SpeechEncoderDecoderModel-wav2vec2-bert` is `speech-encoder-decoder` with a `wav2vec2` encoder and BERT decoder, so it is out of scope for this family.

## 3a. Family variation traps

- Base checkpoints may have `vocab_size=null`; CTC construction rejects configs without a vocab size.
- Common ASR finetunes set `add_adapter=true`, adding stride-2 adapter layers after the main encoder. This changes time length and output width contract.
- `adapter_act` may be absent/null in some configs; current strict config default is `relu`, but a loader should verify effective config normalization.
- `position_embeddings_type` changes attention math. `relative_key` is common, but source also implements Transformer-XL-like `relative`, RoPE-before-QK-projection, and no positional attention.
- `hidden_size` must be divisible by `num_attention_heads`; source infers `head_dim` by integer division and has no explicit GQA/MQA.
- Processor padding values vary between 0 and 1. This affects only padded fbank feature values; model masking still requires `attention_mask` for batched inference.
- The model body consumes rank-3 feature sequences, not NCHW/NHWC images. Layout optimization is about `[B,T,C]` versus Conv1d `[B,C,T]` transposes, not NHWC.
- Training-only SpecAugment and LayerDrop are in source; inference should disable them through eval mode/no `mask_time_indices`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank-3 `[B,T,C]` tensors, `transpose(1,2)`, `view/reshape`, `expand`, `masked_fill`, `cat`, `stack`, `arange`, `clamp`, `einsum`, reductions over time and feature axes.
- Attention mask conversion: `[B,T] -> [B,1,T,T]` additive mask using dtype minimum.
- Feature-length math for CTC/adapter masks.

Neural primitives:

- `LayerNorm(160)` for feature projection, `LayerNorm(H)` throughout.
- `Linear(160 -> H)` feature projection.
- Encoder FFN twice per layer: `Linear(H -> I)`, `swish`, `Linear(I -> H)`, residual scale `0.5`.
- Conv module: pointwise `Conv1d(H -> 2H, k=1, bias=False)`, `GLU(dim=channel)`, left pad `K-1`, depthwise `Conv1d(H -> H, k=31, groups=H, bias=False)`, LayerNorm over channel, `swish`, pointwise `Conv1d(H -> H, k=1, bias=False)`.
- Adapter: optional `Linear(H -> output_hidden_size)` + LayerNorm; per adapter layer two strided `Conv1d(C -> 2C, k=3, stride=2, padding=1)` + GLU branches; adapter FFN `Linear(C -> I)`, activation, `Linear(I -> C)`.

Attention primitives:

- Bidirectional self-attention only; no causal mask, no cross-attention, no KV cache.
- MHA with Q/K/V/out independent biased Linear layers: `Linear(H -> H)` each, shape `[B,heads,T,head_dim]`.
- Common relative-key path requires distance embedding table `[left+right+1, head_dim]` and `einsum("bhld,lrd->bhlr")`.

Position/custom math:

- `relative_key`, `relative`, `rotary`, and none. See sections 6-7.

Preprocessing-coupled ops:

- Optional CPU fbank pipeline: STFT/spectrogram, Kaldi mel filters, log clamp/floor, per-bin normalization, padding, stride-2 reshape from 80 to 160 channels.

Heads:

- CTC: dropout then `Linear(output_hidden_size or H -> vocab_size)`.
- Sequence classification: optional weighted layer sum, `Linear(H -> classifier_proj_size)`, masked mean pooling, `Linear(classifier_proj_size -> num_labels)`.
- Audio frame classification: optional weighted layer sum, `Linear(H -> num_labels)`.
- XVector: `Linear(H -> tdnn_dim[0])`, TDNN conv stack, mean/std pooling, `Linear(2*tdnn_last -> xvector_output_dim)`, `Linear(xvector_output_dim -> xvector_output_dim)`.

## 5. Layer/block breakdown

Base feature projection:

```text
x: [B,T,160]
extract_features = LayerNorm(160)(x)
h = Linear(160 -> H)(extract_features)
h = Dropout(h)
```

Encoder layer, repeated `num_hidden_layers`:

```text
res = h
h = LayerNorm(H)(h)
h = Linear(H -> I) -> swish -> Dropout -> Linear(I -> H) -> Dropout
h = 0.5 * h + res

res = h
h = LayerNorm(H)(h)
q,k,v = Linear(H -> H) each, bias=True
scores = bidirectional_attention(q,k,v, additive_mask, position_math)
h = Linear(H -> H)(attn(scores,v))
h = Dropout(h) + res

res = h
h = LayerNorm(H)(h)
h = masked_fill padded positions to 0 when mask exists
h = transpose [B,T,H] -> [B,H,T]
h = Conv1d(H -> 2H,k=1,bias=False) -> GLU(channel)
h = left_pad(K-1) -> depthwise Conv1d(H -> H,k=K,groups=H,bias=False)
h = transpose -> LayerNorm(H) -> transpose
h = swish -> Conv1d(H -> H,k=1,bias=False) -> Dropout
h = transpose [B,H,T] -> [B,T,H]
h = h + res

res = h
h = LayerNorm(H)(h)
h = Linear(H -> I) -> swish -> Dropout -> Linear(I -> H) -> Dropout
h = 0.5 * h + res
h = final LayerNorm(H)
```

Optional adapter layer:

```text
res = LayerNorm(C)(h) -> transpose -> Conv1d(C -> 2C,k=3,stride=2,pad=1) -> GLU -> transpose
h = LayerNorm(C)(h) -> transpose -> Conv1d(C -> 2C,k=3,stride=2,pad=1) -> GLU -> transpose
h = adapter self-attention(no position embedding) -> Dropout + res
h = h + FFN(LayerNorm(C)(h), C -> I -> C, adapter_act)
```

## 6. Attention requirements

- Type: encoder-style noncausal bidirectional self-attention.
- Heads: MHA, `num_attention_heads=16`, `head_dim=64` for observed 1024-wide checkpoints.
- Q/K/V width: Q, K, V all `H`; no separate KV head count and no rectangular cross-attention.
- Masking: processor/model attention mask is `[B,T]`; encoder converts it to additive `[B,1,T,T]` with zero for valid keys and dtype min for masked keys. Conv modules also zero padded positions before depthwise convolution.
- Packed/varlen: no native cu-seqlens or packed attention backend. Everything is dense padded tensors.
- Sliding/local: none. Convolution is causal-left-padded depthwise Conv1d but attention is full bidirectional.
- Cache: not applicable. There is no autoregressive decode path.
- FlashAttention/SDPA: source uses explicit `matmul -> add mask/position -> softmax -> dropout -> matmul`; no optimized backend dispatch in this file. A fused attention kernel must preserve relative-key addition order before softmax.

## 7. Position encoding and custom math

Default `relative_key`:

```python
distance = arange(Tk)[None, :] - arange(Tq)[:, None]
distance = clamp(distance, -left, right)
pos = distance_embedding(distance + left)  # [Tq,Tk,Dh]
scores = q @ k.transpose(-2, -1) / sqrt(Dh)
scores += einsum("bhld,lrd->bhlr", q, pos) / sqrt(Dh)
```

Transformer-XL-like `relative` path:

```python
rel = linear_pos(relative_position_embeddings).view(1, 2*T-1, heads, Dh)
scores_ac = matmul(q + pos_bias_u, k.T)
scores_bd = matmul(q + pos_bias_v, rel_shifted.T)
scores = (scores_ac + shifted(scores_bd)) / sqrt(Dh)
```

RoPE path is unusual for many LLM lowering assumptions: source applies RoPE to the hidden states before Q/K projections, not to projected Q/K tensors.

```python
x = x.view(B,T,heads,Dh)
rot = cat([-x[..., Dh//2:], x[..., :Dh//2]], dim=-1)
x = x * cos[:T] + rot * sin[:T]
q = linear_q(x.reshape(B,T,H))
k = linear_k(x.reshape(B,T,H))
```

Precomputable: distance index tables for fixed T, relative-key embedding lookup plans, RoPE cos/sin tables by sequence length. Dynamic: sequence length, dtype/device casting, attention masks, adapter-resampled masks.

## 8. Preprocessing and input packing

`Wav2Vec2BertProcessor` delegates audio to `SeamlessM4TFeatureExtractor` and text labels to a tokenizer for CTC training/inference postprocessing. The model forward consumes:

```text
input_features: float tensor [B,T,160]
attention_mask: optional int/bool tensor [B,T]
labels: optional CTC labels [B,U], head/loss only
```

Raw audio feature extraction contract:

- Input sampling rate: 16 kHz in sampled configs.
- Mono/stereo: if waveform is 2D, source takes `waveform[0]`, effectively the left/first channel.
- Scale: waveform multiplied by `2**15` for Kaldi compliance.
- Spectrogram: frame length 400 samples, hop 160, FFT 512, power 2.0, `center=False`, preemphasis 0.97, remove DC offset.
- Window: 400-sample Povey window, non-periodic.
- Mel: 80 Kaldi-scale triangular filters, frequency bins 257, min 20 Hz, max sampling_rate/2, `log_mel="log"`, floor `1.192092955078125e-07`.
- Normalization: default `do_normalize_per_mel_bins=True`, per mel bin `(x - mean) / sqrt(var(ddof=1)+1e-7)`.
- Padding/truncation: feature extractor pads fbank frames, default `pad_to_multiple_of=2`.
- Packing: `[B,F,80]` is truncated to even `F`, reshaped to `[B,F/2,160]`; attention mask keeps frame indices where `index % 2 == 1`.

CPU/data pipeline should own this first. DinoML runtime admission can accept precomputed `[B,T,160]` features for stage 1.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature projection LayerNorm + Linear

Source pattern: `LayerNorm(160) -> Linear(160,H)`.

Replacement: canonical normalized dense projection.

Preconditions: input ABI is packed fbank `[B,T,160]`; no SpecAugment in eval; preserve `extract_features` if returning it.

Parity test: compare feature projection output and `extract_features` for random `[B,T,160]`.

### Rewrite: Conv1d pointwise to Linear

Source pattern: transpose to `[B,C,T]`, `Conv1d(Cin,Cout,k=1)`, transpose back.

Replacement:

```text
[B,T,Cin] -> Linear(Cin -> Cout) -> [B,T,Cout]
```

Preconditions: `kernel_size=1`, `stride=1`, `padding=0`, `dilation=1`, `groups=1`. Weight transform is `linear.weight = conv.weight[:, :, 0]`; preserve `bias=False` for conformer pointwise convs.

Failure cases: adapter convolutions have `k=3,stride=2` and are not pointwise.

### Rewrite: GLU pointwise conv block

Source pattern: `Conv1d(H -> 2H,k=1,bias=False)` then `GLU(dim=1)`.

Replacement: `Linear(H -> 2H,bias=False)` split last dim into `(a,b)` and compute `a * sigmoid(b)`.

Preconditions: after layout rewrite tensor is `[B,T,2H]`; split order follows PyTorch GLU along channel dimension, first half multiplied by sigmoid(second half).

### Rewrite: depthwise Conv1d layout island

Source pattern: `[B,T,H] -> transpose -> left_pad(K-1) -> depthwise Conv1d(groups=H) -> transpose`.

Replacement: keep as a Conv1d layout island or lower to a specialized time-depthwise kernel over `[B,T,H]`.

Preconditions: padding is entirely left-side, not symmetric SAME despite config error text; `groups==H`, `bias=False`, `stride=1`.

Layout constraints: protect with `no_layout_translation()` unless the whole transpose/pad/depthwise/LN/activation/pointwise region is rewritten together. Axis-sensitive LN is over channels after transposing back to `[B,T,H]`.

### Rewrite: adapter strided Conv1d + GLU

Source pattern: two branches of `Conv1d(C -> 2C,k=3,stride=2,padding=1) -> GLU`.

Replacement: strided temporal convolution or im2col+GEMM, followed by split/sigmoid multiply.

Preconditions: static `k=3,stride=2,pad=1,dilation=1,groups=1`; output length follows PyTorch Conv1d floor formula.

Failure cases: dynamic shape or mismatched mask resampling must fall back until `_compute_new_attention_mask` parity exists.

### Rewrite: relative-key attention bias

Source pattern: dense QK scores plus query-dependent Shaw bias.

Replacement: fused attention pre-score hook that adds `einsum(q, distance_embedding[clamped_j_minus_i]) / sqrt(Dh)` before softmax.

Preconditions: `position_embeddings_type == "relative_key"`; fixed left/right clip; full dense attention. Not valid for `relative`, `rotary`, or `None`.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + Linear for feature projection and block pre-norms.
- Encoder FFN `Linear -> swish -> Linear` with residual scale `0.5`.
- Dense MHA with relative-key pre-softmax bias.
- Conformer conv module: pointwise GLU, depthwise time convolution, activation, pointwise projection.

Medium priority:

- Adapter strided Conv1d + GLU + mask-resampling.
- CTC head `Dropout(eval no-op) -> Linear` and optional log-softmax for decoder integration.
- Weighted-layer-sum for classification/x-vector heads.

Lower priority:

- Transformer-XL-like `relative` and RoPE variants, unless checkpoints using them are admitted.
- XVector TDNN/stat-pooling and AMSoftmax training objective; useful for speaker verification but not first ASR target.

## 11. Runtime staging plan

1. Parse config and reject unsupported combinations explicitly: unknown `position_embeddings_type`, `hidden_size % heads != 0`, adapter null activation before normalization, missing CTC vocab for CTC target.
2. Load base weights and run feature-input `Wav2Vec2BertModel` with `[B,T,160]` tensors, no adapter.
3. Implement encoder block parity with default `relative_key` attention and conformer conv module.
4. Add CTC head and verify common adapter-enabled ASR checkpoints.
5. Add CPU/data-pipeline fbank parity or define a stable precomputed-feature ABI if DinoML will not own preprocessing initially.
6. Add sequence/audio-frame classification heads.
7. Add XVector head if speaker verification is in scope.
8. Optimize fusions/layout islands after correctness.

## 12. Parity and validation plan

- Feature extractor snapshot test: short mono waveform and stereo waveform; verify `[B,F/2,160]`, left-channel behavior, even-frame truncation, and packed mask.
- Feature projection parity: random `[B,T,160]`, compare `hidden_states` and `extract_features`.
- Attention parity: single layer with `position_embeddings_type` in `relative_key`, then source-default full block.
- Encoder parity: after 1, 2, and 24 layers with masks containing padding.
- Adapter parity: CTC finetune config with `add_adapter=true`; verify output length and new attention mask.
- CTC logits parity: `hf-audio/wav2vec2-bert-CV16-en` and one non-English vocab checkpoint with precomputed features.
- Head parity: sequence classifier masked pooling, audio-frame logits, XVector mean/std pooling if admitted.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 after fusions `rtol=1e-2, atol=1e-2`, with attention softmax checked separately.

No DinoML tests were run for this audit.

## 13. Performance probes

- Processor throughput: waveform seconds/sec for fbank extraction, normalization, padding, and stride packing.
- Encoder-only throughput by `(B,T)` with and without padding masks.
- Attention backend comparison: explicit dense attention versus fused relative-key attention.
- Conformer conv module microbench: pointwise GLU, depthwise K=31, and full conv block.
- Adapter-enabled CTC throughput and output time-length reduction.
- CTC head/log-softmax throughput by vocab size: 33, 66, 2699.
- Memory probe: attention score tensor `[B,heads,T,T]` dominates long audio.
- Layout probe: keep Conv1d NCT island versus channel-last `[B,T,C]` specialized depthwise kernel.

## 14. Skip/defer list

- Training losses, SpecAugment, LayerDrop, gradient checkpointing.
- Autoregressive generation, beam search controller, KV cache: not native to this family.
- `position_embeddings_type="relative"` and `"rotary"` until a checkpoint requiring them is admitted.
- XVector AMSoftmax loss and PEFT/LoRA warnings in TDNN.
- Multi-GPU/FSDP/DeepSpeed sync behavior.
- Full raw-audio feature extraction on GPU; accept precomputed `[B,T,160]` first.

## 15. Final implementation checklist

- [ ] Parse `Wav2Vec2BertConfig` and normalize effective defaults.
- [ ] Load base encoder and CTC adapter checkpoints.
- [ ] Admit precomputed `input_features [B,T,160]` plus `attention_mask [B,T]`.
- [ ] Implement feature projection `LayerNorm(160) + Linear(160 -> H)`.
- [ ] Implement default `relative_key` MHA bias and dense bidirectional attention.
- [ ] Implement conformer encoder layer with two half-scaled FFNs and Conv1d module.
- [ ] Implement optional adapter strided Conv1d/GLU/self-attention/FFN path.
- [ ] Implement CTC logits head and output-length helper.
- [ ] Add guarded rewrites for pointwise Conv1d to Linear and GLU split order.
- [ ] Add layout guard for depthwise Conv1d temporal island.
- [ ] Add fbank processor parity tests or document precomputed-feature boundary.
- [ ] Add single-layer, full-encoder, adapter, and CTC-head parity tests.
- [ ] Benchmark encoder attention, conformer conv, adapter, and CTC vocab-size sweeps.
