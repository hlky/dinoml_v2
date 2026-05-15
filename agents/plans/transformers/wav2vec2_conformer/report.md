# Wav2Vec2-Conformer Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/wav2vec2-conformer-rel-pos-large, facebook/wav2vec2-conformer-rel-pos-large-960h-ft, facebook/wav2vec2-conformer-rope-large, facebook/wav2vec2-conformer-rope-large-960h-ft, juliensimon/wav2vec2-conformer-rel-pos-large-finetuned-speech-commands
Config source: Hugging Face config/preprocessor/tokenizer JSON snapshots saved beside this report.
Source files inspected:
- transformers/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py
- transformers/src/transformers/models/wav2vec2_conformer/configuration_wav2vec2_conformer.py
- transformers/src/transformers/models/wav2vec2_conformer/modular_wav2vec2_conformer.py
- transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
- transformers/src/transformers/feature_extraction_sequence_utils.py
Any missing files or assumptions: no gated/401 model config links were encountered. tokenizer_config.json is absent/404 for the pretraining checkpoints and the Speech Commands classifier mirror; that is expected because the first two are not CTC tokenizers and the classifier does not tokenize text.
```

`modeling_wav2vec2_conformer.py` is the generated importable source used by AutoModel. `modular_wav2vec2_conformer.py` is the smaller modular source and is useful for future source edits; this report treats the generated modeling file as authoritative for runtime behavior.

## 2. High-level architecture

This is an audio encoder family over raw waveform input, not a text-generation model. The first useful DinoML runtime target should be `Wav2Vec2ConformerForCTC` for ASR logits, with the base encoder independently stageable.

```text
mono waveform + optional sample mask
-> CPU feature extractor padding/normalization
-> Conv1d feature encoder
-> feature LayerNorm + Linear projection
-> Conformer encoder blocks
-> optional adapter
-> CTC dropout + Linear vocab head
-> logits [batch, feature_time, vocab]
```

Other heads in source:

| Head | Status for first target | Notes |
|---|---:|---|
| `Wav2Vec2ConformerModel` | Required | Base encoder ABI and hidden states. |
| `Wav2Vec2ConformerForCTC` | Required | Primary ASR target. |
| `ForPreTraining` | Deferred | Gumbel vector quantizer, negative sampling, contrastive losses are training-oriented. |
| `ForSequenceClassification` | Optional | Keyword/speech-command pooling head. |
| `ForAudioFrameClassification` | Optional | Per-frame classifier over encoder output. |
| `ForXVector` | Deferred | Speaker embedding head with TDNN and AMSoftmax objective. |

## 3. Important config dimensions

Source defaults from `Wav2Vec2ConformerConfig`:

| Field | Default |
|---|---:|
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | `hidden_size // num_attention_heads` |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `feat_extract_norm` | `group` |
| `conv_dim` | `(512,512,512,512,512,512,512)` |
| `conv_stride` | `(5,2,2,2,2,2,2)` |
| `conv_kernel` | `(10,3,3,3,3,2,2)` |
| `inputs_to_logits_ratio` | 320 |
| `num_conv_pos_embeddings` | 128 |
| `num_conv_pos_embedding_groups` | 16 |
| `position_embeddings_type` | `relative` |
| `rotary_embedding_base` | 10000 |
| `max_source_positions` | 5000 |
| `conv_depthwise_kernel_size` | 31 |
| `add_adapter` | false |

Representative checkpoint sweep from downloaded HF configs:

| Checkpoint | Arch | Vocab/labels | H | Layers | Heads | FFN | Pos | Feature norm | Conv bias | Hidden act |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `facebook/wav2vec2-conformer-rel-pos-large` | PreTraining | n/a | 1024 | 24 | 16 | 4096 | relative | layer | true | swish |
| `facebook/wav2vec2-conformer-rel-pos-large-960h-ft` | CTC | 32 | 1024 | 24 | 16 | 4096 | relative | layer | true | swish |
| `facebook/wav2vec2-conformer-rope-large` | PreTraining | n/a | 1024 | 24 | 16 | 4096 | rotary | layer | true | swish |
| `facebook/wav2vec2-conformer-rope-large-960h-ft` | CTC | 32 | 1024 | 24 | 16 | 4096 | rotary | layer | true | swish |
| `juliensimon/...speech-commands` | SequenceClassification | 36 labels | 1024 | 24 | 16 | 4096 | relative | layer | true | swish |

Effective large-checkpoint feature time for raw waveform length `L` is repeated floor Conv1d length:

```python
for kernel, stride in zip([10,3,3,3,3,2,2], [5,2,2,2,2,2,2]):
    L = floor((L - kernel) / stride) + 1
```

For 16,000 samples, this yields 49 encoder frames.

## 3a. Family variation traps

- `position_embeddings_type` changes the attention math. Relative position uses a Transformer-XL style learned `linear_pos` plus `pos_bias_u/v` and a relative-shift reshape. Rotary applies RoPE to hidden states before Q/K projection, not after Q/K projection.
- Source defaults say `feat_extract_norm="group"` and `conv_bias=False`, but representative Conformer large checkpoints use `feat_extract_norm="layer"` and `conv_bias=true`; do not hard-code base Wav2Vec2 assumptions.
- Checkpoint configs contain historical fields such as `do_stable_layer_norm`, `feat_extract_dropout`, `hidden_dropout_prob`, and `gradient_checkpointing`; the inspected Wav2Vec2-Conformer source does not use these in the forward graph.
- `head_dim` is inferred as integer division. DinoML should require `hidden_size % num_attention_heads == 0` even though the config validator only checks Conv1d list lengths.
- `add_adapter=true` inserts post-encoder stride-2 GLU Conv1d adapter layers and can change output hidden width; CTC supports it, sequence/audio-frame classification explicitly reject adapters.
- `apply_spec_augment` and mask indices mutate hidden states during training. Inference should disable or reject `mask_time_indices` unless pretraining parity is in scope.
- The source uses time-major comments inconsistently; runtime tensors are `[B,T,C]` except Conv1d regions, which transpose to `[B,C,T]`.
- Layout translation should be guarded. The model is 1D audio, so NHWC/NCHW vision terminology maps to `BTC` versus `BCT`; Conv1d and BatchNorm1d require `BCT`, while LayerNorm, Linear, attention, pooling, and heads use `BTC`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Unsqueeze raw waveform `[B,L] -> [B,1,L]`.
- Transpose `BCT <-> BTC`, reshape/view for heads, flatten for CTC/training losses.
- Boolean mask creation from sample-level `attention_mask`, cumsum, flip, scatter assignment at output length minus one.
- In-place or functional masked fill/zero for padded encoder states; DinoML can lower to guarded elementwise select.

Neural primitives:

- Conv1d feature stack: `1->512 k10 s5`, then six `512->512` layers with kernels `3,3,3,3,2,2`, strides `2`.
- LayerNorm over channel dim for layer-normalized feature extractor and feature projection.
- GroupNorm first Conv1d only for source-default group-norm configs; groups equal output channels.
- Linear `512 -> hidden_size` feature projection.
- Per Conformer layer: two FFNs `hidden -> intermediate -> hidden`, activation from config, dropout no-op in eval.
- Pointwise Conv1d `hidden -> 2*hidden`, GLU over channel dim, depthwise Conv1d `hidden -> hidden k=31 groups=hidden`, BatchNorm1d, activation, pointwise Conv1d `hidden -> hidden`.
- Final encoder LayerNorm.
- CTC lm head `output_hidden_size or hidden_size -> vocab_size`.

Attention primitives:

- Dense noncausal self-attention only. No causal mask and no KV cache.
- MHA with `num_heads`, `head_dim=hidden/heads`, Q/K/V/O all `hidden -> hidden` with bias.
- Additive expanded padding mask `[B,1,T,T]` with `torch.finfo(dtype).min`.
- Softmax on key dimension, dropout in training.

Position/custom math:

- Grouped positional Conv1d `hidden -> hidden`, kernel `num_conv_pos_embeddings`, groups `num_conv_pos_embedding_groups`, weight norm, same-pad crop, activation. This is defined but not called in `Wav2Vec2ConformerEncoder.forward` at this commit; DinoML should not include it in the active graph unless source behavior changes.
- Relative sinusoidal table `[1, 2*T-1, hidden]`, `linear_pos`, head reshape, `pos_bias_u/v`, relative-shift.
- Rotary table `[2,T,1,1,head_dim]`, applied to `query_key_states` before Q/K Linear.

Preprocessing-coupled ops:

- CPU/data-pipeline mono waveform validation, padding/truncation, optional `pad_to_multiple_of`, float32 conversion, and zero-mean/unit-variance normalization with `var + 1e-7`.
- No STFT, FFT, mel bins, or spectrogram extraction. The neural graph consumes raw normalized waveform.

Optional/deferred:

- Gumbel softmax quantizer, argmax one-hot scatter, codebook gather/sum, cosine similarity, negative sampling for pretraining.
- TDNN Conv1d-by-linear-weight view, statistic pooling, L2 normalize, AMSoftmax for xvector.

## 5. Layer/block breakdown

Base encoder forward:

```text
input_values [B,L]
features = Conv1d stack(input_values[:,None,:])                 # [B,512,T]
features = features.transpose(1,2)                              # [B,T,512]
hidden, extract_features = LayerNorm(512) -> Linear(512,H)      # [B,T,H], [B,T,512]
attention_mask = reduce_sample_mask_to_feature_mask(optional)   # [B,T] bool
hidden = SpecAugment mask only in training/when mask_time_indices provided
hidden = ConformerEncoder(hidden, attention_mask)
hidden = optional adapter(hidden)
```

Conformer encoder layer, repeated `N` times:

```text
res = x
x = LayerNorm(x)
x = Linear(H,I) -> activation -> Linear(I,H)
x = 0.5 * x + res

res = x
x = LayerNorm(x)
x = MHA_with_relative_or_rotary_position(x, mask)
x = Dropout(x) + res

res = x
x = LayerNorm(x)
x = transpose BTC->BCT
x = Conv1d_1x1(H,2H,bias=False) -> GLU(channel)
x = DepthwiseConv1d(H,H,k=31,pad=15,groups=H,bias=False)
x = BatchNorm1d(H) -> activation
x = Conv1d_1x1(H,H,bias=False) -> Dropout
x = transpose BCT->BTC
x = x + res

res = x
x = LayerNorm(x)
x = Linear(H,I) -> activation -> Linear(I,H)
x = 0.5 * x + res
x = final LayerNorm(x)
```

CTC head:

```text
hidden = base_encoder(input_values, attention_mask).last_hidden_state
logits = Linear(H_or_output_H, vocab_size)(Dropout(hidden))
```

Representative large CTC shape: `H=1024`, `I=4096`, `heads=16`, `head_dim=64`, `vocab=32`.

## 6. Attention requirements

The required attention is encoder-only, noncausal, dense self-attention:

| Field | Requirement |
|---|---|
| causal | No |
| self/cross | Self-attention only |
| MHA/MQA/GQA | MHA only |
| Q/K/V widths | all `hidden_size` |
| head count | `num_attention_heads` |
| KV cache | Not applicable |
| mask | optional padding mask expanded to `[B,1,T,T]` and added before softmax |
| local/sliding/sparse | Not implemented |
| packed/varlen | Not implemented |
| FlashAttention/SDPA | Source uses explicit matmul/softmax/matmul; FlashAttention is an optimization only if relative/rotary math and mask semantics are preserved. |

Relative attention is not a standard QK-only attention score: it adds `scores_ac + scores_bd`, where `scores_bd` is derived from projected relative positions and shifted by reshape/slice. This likely needs a custom fused relative-attention path or a decomposition fallback before FlashAttention can be used.

Rotary attention is also nonstandard relative to common LLM RoPE placement: source rotates the full hidden states before Q/K projection. A Q/K RoPE fusion is invalid unless weights are transformed or the source graph is explicitly changed with parity proof.

## 7. Position encoding and custom math

Relative positional table:

```python
def rel_pos_table(T, H):
    pos = arange(T)[:, None]
    div = exp(arange(0, H, 2) * -(log(10000.0) / H))
    pe_pos[:, 0::2] = sin(pos * div)
    pe_pos[:, 1::2] = cos(pos * div)
    pe_neg[:, 0::2] = sin(-pos * div)
    pe_neg[:, 1::2] = cos(-pos * div)
    return concat([flip(pe_pos, time), pe_neg[1:]], dim=time)[None]
```

Relative attention score:

```python
pos = linear_pos(relative_position_embeddings).view(1, 2*T-1, heads, head_dim)
pos = pos.transpose(1, 2).transpose(2, 3)       # [1, heads, head_dim, 2*T-1]
q_u = q + pos_bias_u[None, :, None, :]
q_v = q + pos_bias_v[None, :, None, :]
scores_ac = matmul(q_u, k.transpose(-2, -1))
scores_bd = matmul(q_v, pos)
scores_bd = relative_shift(scores_bd)[:, :, :, :T]
scores = (scores_ac + scores_bd) / sqrt(head_dim)
```

Rotary source placement:

```python
def source_rotary_before_projection(x, cos, sin):
    x = x.view(B, T, heads, head_dim).transpose(0, 1)
    a, b = x[..., :head_dim//2], x[..., head_dim//2:]
    rot = concat([-b, a], dim=-1)
    return ((x * cos[:T]) + (rot * sin[:T])).transpose(0, 1).reshape(B, T, H)
```

The relative sinusoidal tables and RoPE cos/sin are sequence-length dependent and can be precomputed per bucket/dtype/device. Relative tables may extend beyond `max_source_positions` dynamically through `extend_pe`.

## 8. Preprocessing and input packing

Feature extractor ABI from `Wav2Vec2FeatureExtractor` and saved preprocessor configs:

| Field | Value |
|---|---:|
| Input | mono raw speech, one float per sample |
| Sampling rate | 16000 Hz |
| `feature_size` | 1 |
| `padding_value` | 0 |
| `do_normalize` | true |
| `return_attention_mask` | true in inspected Conformer configs |
| Output tensors | `input_values [B,L] float32`, optional `attention_mask [B,L] int32/long` |

There is no STFT/mel frontend. Audio decode, resampling to 16 kHz, channel mixing to mono, padding/truncation, and normalization are CPU/data-pipeline work for first integration. Normalization uses only unpadded samples when an attention mask is present:

```text
normed[:length] = (x[:length] - mean(x[:length])) / sqrt(var(x[:length]) + 1e-7)
normed[length:] = padding_value
```

The model consumes `attention_mask` only after reducing it through the convolution output-length formula. The reduced mask is used to zero padded hidden states and to add large negative attention scores. For group-norm configs, Wav2Vec2 docs warn that attention masks may be inappropriate; the inspected Conformer checkpoints use layer norm and return masks.

CTC tokenizer coupling is outside the neural graph. Fine-tuned CTC checkpoints use `Wav2Vec2CTCTokenizer` with `word_delimiter_token="|"`, `pad_token="<pad>"`, `unk_token="<unk>"`, and `vocab_size=32`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d feature stack to provider Conv1d

Source pattern: seven sequential Conv1d layers over `BCT`, each followed by optional normalization and activation.

Replacement: keep as Conv1d initially; later lower fixed small-channel 1D convs to im2col+GEMM or direct CUDA Conv1d kernels.

Preconditions:

- Input is contiguous `[B,C,T]`.
- Kernel/stride/bias match config exactly.
- For layer norm conv layers, transpose to `[B,T,C]` for LayerNorm and back.
- Output-length guards use source floor formula.

Failure cases: grouped first-layer norm versus all-layer LayerNorm must dispatch different blocks; `conv_bias` differs between source defaults and checkpoints.

Parity test sketch: random waveform lengths around kernel/stride boundaries, compare all feature-stack outputs against PyTorch fp32.

### Rewrite: Conformer pointwise Conv1d 1x1 to Linear

Source pattern:

```text
BTC -> LayerNorm -> transpose BCT -> Conv1d(k=1) -> GLU -> Conv1d(k=1) -> transpose BTC
```

Replacement:

```text
LayerNorm(BTC) -> Linear(H,2H,bias=False) -> GLU(last_dim) -> transpose for depthwise Conv1d -> ... -> Linear(H,H,bias=False)
```

Weight transform:

```python
linear.weight = conv1d.weight[:, :, 0]
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Bias handling matches source (`bias=False` in Conformer conv module).
- GLU axis rewritten from channel dim in `BCT` to last dim in `BTC`.

Failure cases: cannot remove transpose around depthwise Conv1d/BatchNorm unless a full `BTC` depthwise conv+BN lowering exists.

### Rewrite: Depthwise Conv1d + BatchNorm inference folding

Source pattern: depthwise Conv1d without bias followed by BatchNorm1d in eval mode.

Replacement: folded depthwise Conv1d with bias.

Preconditions:

- Module in eval mode with frozen BN running mean/variance.
- `groups == hidden_size`, `padding=(k-1)//2`, odd kernel.
- Use BN `eps`, `weight`, `bias`, `running_mean`, `running_var`.

Weight transform:

```python
scale = bn.weight / sqrt(bn.running_var + bn.eps)
w_fold = conv.weight * scale[:, None, None]
b_fold = bn.bias - bn.running_mean * scale
```

Failure cases: training mode, missing BN running stats, or dynamic batch-stat behavior.

### Rewrite: Relative attention decomposition

Source pattern: explicit `scores_ac`, `scores_bd`, relative shift, mask add, softmax, matmul V.

Replacement: custom `relative_mha_encoder` op or decomposed matmul path.

Preconditions:

- Noncausal self-attention.
- `T_q == T_k`.
- Relative table shape `[1,2*T-1,H]`.
- No KV cache.

Failure cases: standard FlashAttention cannot consume `scores_bd` directly; do not replace with plain SDPA.

### Rewrite: Rotary Conformer attention guard

Source pattern: RoPE before Q/K projections.

Replacement: preserve exact source order initially. Only consider Q/K RoPE fusion with a formal weight transform and parity tests.

Failure cases: common LLM RoPE kernels rotate projected Q/K and will not match this source graph.

### Layout guidance: `BTC`/`BCT` instead of NHWC/NCHW

Initial semantic lowering should preserve source axes. Candidate optimized regions:

- Feature Conv1d stack can remain `BCT` from waveform unsqueeze through final Conv1d, then transpose once to `BTC`.
- Attention/FFN/heads should stay `BTC`.
- Conformer conv module has a local, fully controlled `BTC -> BCT -> BTC` region; a layout pass may fuse pointwise Linear forms while guarding depthwise Conv1d and BatchNorm1d.

Axis rewrites required if keeping conv module in `BTC`: GLU `dim=1` becomes last-dim GLU before depthwise transpose, BatchNorm1d must become channel-last batch norm or stay in `BCT`, reductions/pooling over `dim=1` in heads are time-axis reductions and must not be confused with channel reductions.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d feature stack with LayerNorm/activation support. It dominates raw-audio front-end cost and fixes input ABI parity.
- LayerNorm + Linear for feature projection and FFNs.
- Conformer convolution module: pointwise projection + GLU, depthwise Conv1d + folded BatchNorm + activation.
- Relative-position attention score kernel for relative checkpoints, because plain SDPA/FlashAttention is not semantically enough.
- CTC head Linear and optional log-softmax for training/eval postprocessing.

Medium priority:

- Rotary pre-projection transform + Q/K/V projection fusion for rope checkpoints, after parity proof.
- Mask reduction from raw sample mask to feature mask as a small generated shape/mask kernel.
- Encoder block residual/scale fusions around Macaron FFNs.
- Weighted layer-sum for classifier/xvector variants.

Lower priority:

- Gumbel quantizer/pretraining losses.
- TDNN/xvector statistic pooling and AMSoftmax.
- Adapter Conv1d GLU stack for rare `add_adapter=true` configs.

## 11. Runtime staging plan

1. Parse config and processor JSON; admit `feat_extract_norm="layer"`, `position_embeddings_type in {"relative","rotary"}`, `add_adapter=false`, CTC head first.
2. Load weights and run feature extractor Conv1d stack parity on fp32 waveforms.
3. Implement one Conformer encoder layer parity with both relative and rotary configs.
4. Run full base encoder parity for a fixed raw length and attention mask.
5. Add CTC head and compare logits for `facebook/wav2vec2-conformer-*-960h-ft`.
6. Add optimized Conv1d/Linear/LayerNorm/attention lowerings behind graph-rewrite guards.
7. Extend to sequence classification pooling, then optional frame classification.
8. Revisit adapters, xvector, and pretraining only after ASR encoder/CTC is stable.

Initially stub/dropout/layerdrop/specaugment as eval-mode no-ops, reject `training=True`, reject `mask_time_indices`, and reject output attentions unless the decomposed attention tensors are explicitly materialized.

## 12. Parity and validation plan

- Feature extractor CPU reference: random mono waveforms with lengths near 399, 400, 401, 16000, plus batched padding; compare Conv1d output and reduced attention mask.
- Preprocessor ABI: compare normalization and padding output with `Wav2Vec2FeatureExtractor` for masked and unmasked batches.
- Position math: compare relative table slices, relative-shift output, and rotary pre-projection transform for multiple `T`.
- Single Conformer layer: fp32 relative and rotary configs with dropout disabled; tolerance `rtol=1e-4, atol=1e-4`.
- Full encoder: large config, short waveform buckets, compare final hidden state; fp16 tolerance should be looser after validating fp32.
- CTC logits: compare `facebook/wav2vec2-conformer-rel-pos-large-960h-ft` and rope 960h logits for fixed audio.
- Classification optional: compare speech-commands pooled logits, especially attention-mask mean pooling.
- Negative tests: reject stereo/rank-3 audio in preprocessing path, invalid sample rate, unsupported `position_embeddings_type`, unsupported adapters for classifier heads, and configs where `hidden_size` is not divisible by heads.

## 13. Performance probes

- CPU preprocessing throughput: decode/resample outside DinoML, then normalization/padding samples/sec.
- Feature Conv1d stack latency versus waveform length and batch size.
- Encoder-only throughput over feature lengths 49, 99, 199, 399.
- Relative attention score kernel versus decomposed matmul/shift implementation.
- Rope checkpoint attention throughput with preserved pre-projection RoPE.
- Conformer conv module standalone throughput, with and without BatchNorm folding.
- End-to-end CTC logits throughput for 1s, 5s, 10s, and 30s audio.
- Memory and temporary allocation probe for dense attention `[B,heads,T,T]`.
- Layout probe for `BTC/BCT` transpose elimination inside controlled conv regions.

## 14. Skip/defer list

- Training losses, SpecAugment random masks, Gumbel softmax, negative sampling, diversity loss.
- Gradient checkpointing, layerdrop stochastic execution, dropout stochasticity.
- CTC loss; first inference target needs logits and external decode only.
- Beam search/LM decoding and tokenizer postprocessing beyond greedy CTC smoke tests.
- XVector TDNN/AMSoftmax speaker verification.
- Adapter language loading and `target_lang` adapter-file conventions.
- Quantization/packed weights; no source-coupled quantized format is present in inspected configs.
- Multi-GPU, distributed, FSDP/DeepSpeed-specific branches.

## 15. Final implementation checklist

- [ ] Parse `Wav2Vec2ConformerConfig` and processor config.
- [ ] Admit/evaluate raw mono waveform ABI `[B,L]` plus optional `[B,L]` attention mask.
- [ ] Implement CPU/data-pipeline normalization parity.
- [ ] Implement Conv1d feature extractor with layer/group norm variants.
- [ ] Implement feature output-length and reduced feature attention-mask generation.
- [ ] Implement feature projection LayerNorm + Linear.
- [ ] Implement Conformer FFN with 0.5 residual scaling.
- [ ] Implement dense encoder MHA with padding mask.
- [ ] Implement relative positional embedding and shifted relative attention scores.
- [ ] Implement rotary pre-projection positional transform.
- [ ] Implement Conformer conv module with GLU, depthwise Conv1d, BatchNorm1d, activation.
- [ ] Implement final encoder LayerNorm.
- [ ] Implement CTC dropout no-op in eval and Linear vocab head.
- [ ] Add graph rewrite for Conv1d k=1 to Linear where guarded.
- [ ] Add graph rewrite for eval BatchNorm folding into depthwise Conv1d.
- [ ] Add one-layer, full-encoder, and CTC-logit parity tests for relative and rope checkpoints.
- [ ] Benchmark preprocessing, Conv1d stack, attention, conformer conv module, and end-to-end CTC logits.
