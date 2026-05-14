# UniSpeech full-audit report

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/unispeech-large-1500h-cv and related open microsoft/unispeech-* checkpoints
Config source: local configuration_unispeech.py plus HF config.json snapshots downloaded 2026-05-13
Source files inspected:
- X:/H/transformers/src/transformers/models/unispeech/modeling_unispeech.py
- X:/H/transformers/src/transformers/models/unispeech/configuration_unispeech.py
- X:/H/transformers/src/transformers/models/unispeech/modular_unispeech.py
- X:/H/transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
- X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
- UniSpeech uses the Wav2Vec2 feature extractor class in processor configs; there is no separate UniSpeech feature extractor file.
- `modular_unispeech.py` subclasses Wav2Vec2 modules, but the generated `modeling_unispeech.py` contains the concrete runtime implementation inspected here.
- `microsoft/unispeech-large-fr-ft` is public but has only `.gitattributes`; `config.json` and `preprocessor_config.json` returned 404, so it is a repo-content gap rather than a gated/401 gap.
- No gated/401 UniSpeech checkpoint was encountered in the sampled official Microsoft repos.
```

Local source URL shape for future review:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/unispeech/modeling_unispeech.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/unispeech/configuration_unispeech.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py`

Config snapshots saved beside this report:

- `config_snapshot_summary.json`
- `microsoft__unispeech-large-1500h-cv__config.json`
- `microsoft__unispeech-large-multi-lingual-1500h-cv__config.json`
- `microsoft__unispeech-1350-en-168-es-ft-1h__config.json`
- `microsoft__unispeech-1350-en-17h-ky-ft-1h__config.json`
- `microsoft__unispeech-1350-en-353-fr-ft-1h__config.json`
- `microsoft__unispeech-1350-en-90-it-ft-1h__config.json`
- matching `preprocessor_config.json` files where present.

Primary runtime target for DinoML: `UniSpeechForCTC` encoder inference for automatic speech recognition. The base `UniSpeechModel` encoder is required. `UniSpeechForPreTraining` and `UniSpeechForSequenceClassification` are documented but can be deferred for first ASR integration.

## 2. High-level architecture

UniSpeech is an audio encoder family derived from Wav2Vec2-style raw-waveform modeling. It is not an autoregressive decoder and has no KV cache. For ASR, the runtime contract is:

```text
CPU audio decode/resample/pad/normalize -> raw waveform [B, T_audio]
-> Conv1d feature encoder [B, 512, T_feat]
-> transpose to [B, T_feat, 512]
-> LayerNorm + Linear(512 -> hidden_size)
-> bidirectional Transformer encoder
-> optional task head: CTC Linear(hidden_size -> vocab_size)
-> logit postprocessing / CTC decode outside the model graph
```

Stage decomposition:

- CPU/data pipeline: mono waveform loading, sampling-rate guard, right padding, optional truncation, attention mask construction, zero-mean/unit-variance normalization.
- GPU/runtime stage 1: strided Conv1d feature extraction over `[B, 1, T_audio]`.
- GPU/runtime stage 2: feature projection and bidirectional encoder.
- GPU/runtime stage 3: CTC or classification head.
- Postprocessing: CTC greedy/beam decode, tokenizer vocabulary, blank handling, and label cleanup are outside the neural graph.

The convolutional feature encoder and Transformer encoder can be validated independently by comparing `extract_features` and `last_hidden_state`. The CTC head can be validated independently once encoder parity is established.

## 3. Important config dimensions

Source defaults from `UniSpeechConfig`:

| Field | Default |
| --- | --- |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | 64, inferred as `hidden_size // num_attention_heads` |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `vocab_size` | 32 |
| `conv_dim` | `(512, 512, 512, 512, 512, 512, 512)` |
| `conv_stride` | `(5, 2, 2, 2, 2, 2, 2)` |
| `conv_kernel` | `(10, 3, 3, 3, 3, 2, 2)` |
| `inputs_to_logits_ratio` | 320 |
| conv receptive field | 400 samples, inferred from kernels/strides |
| `feat_extract_norm` | `group` |
| `conv_bias` | `False` |
| `num_conv_pos_embeddings` | 128 |
| `num_conv_pos_embedding_groups` | 16 |
| `do_stable_layer_norm` | `False` |
| `layer_norm_eps` | `1e-5` |
| `num_codevectors_per_group` | 320 |
| `num_codevector_groups` | 2 |
| `codevector_dim` | 256 |
| `proj_codevector_dim` | 256 |
| cache support | no autoregressive cache |

Representative checkpoint sweep from official Microsoft HF configs:

| Model id | Architecture | Hidden/layers/heads | FFN | Vocab or classes | Norm | Stable LN | Conv | Preprocessor |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `microsoft/unispeech-large-1500h-cv` | `UniSpeechForPreTraining` | 1024 / 24 / 16 | 4096 | `num_ctc_classes=80` | layer | true | 7 layers, stride product 320 | 16 kHz, normalize, return mask |
| `microsoft/unispeech-large-multi-lingual-1500h-cv` | `UniSpeechForPreTraining` | 1024 / 24 / 16 | 4096 | `num_ctc_classes=531` | layer | true | same | 16 kHz, normalize, return mask |
| `microsoft/unispeech-1350-en-168-es-ft-1h` | `UniSpeechForCTC` | 1024 / 24 / 16 | 4096 | `vocab_size=45`, `pad_token_id=41` | layer | true | same | 16 kHz, normalize, return mask |
| `microsoft/unispeech-1350-en-17h-ky-ft-1h` | `UniSpeechForCTC` | 1024 / 24 / 16 | 4096 | `vocab_size=43`, `pad_token_id=39` | layer | true | same | 16 kHz, normalize, return mask |
| `microsoft/unispeech-1350-en-353-fr-ft-1h` | `UniSpeechForCTC` | 1024 / 24 / 16 | 4096 | `vocab_size=46`, `pad_token_id=42` | layer | true | same | 16 kHz, normalize, return mask |
| `microsoft/unispeech-1350-en-90-it-ft-1h` | `UniSpeechForCTC` | 1024 / 24 / 16 | 4096 | `vocab_size=65`, `pad_token_id=61` | layer | true | same | 16 kHz, normalize, return mask |

For a 16 kHz one-second input of length 16000, the default conv stack yields 49 feature frames. 32000 yields 99, 48000 yields 149, and 160000 yields 499 by the source floor formula.

## 3a. Family variation traps

- Source defaults describe a base 12-layer, `feat_extract_norm="group"` model, but official sampled checkpoints are 24-layer large models with `feat_extract_norm="layer"` and `do_stable_layer_norm=True`.
- `hidden_size` must be divisible by `num_attention_heads`; the source raises if not. Sampled large configs use `1024 / 16 = 64`.
- No GQA/MQA: Q, K, and V all project `hidden_size -> hidden_size`; KV head count is the same as query head count.
- Attention is bidirectional self-attention for encoder inference, not causal generation.
- `feat_extract_norm` changes Conv1d block structure: `"layer"` applies LayerNorm after every conv on `[B, T, C]`; `"group"` applies GroupNorm only to the first conv and no norm to later convs.
- `do_stable_layer_norm` changes residual/LN ordering. Sampled large checkpoints use pre-LN/stable encoder with a final encoder LayerNorm.
- Positional embedding is a grouped weight-normalized Conv1d over hidden channels, not RoPE, ALiBi, or learned absolute token embeddings.
- `num_conv_pos_embeddings` even values remove one trailing position after same-padding. Default and sampled configs use 128, so the positional conv output is cropped by one on the time axis.
- CTC `pad_token_id` varies with vocabulary and is the blank id for training loss and decoding conventions.
- `UniSpeechForCTC` can load language adapters if `adapter_attn_dim` exists and `target_lang` is passed. None of the sampled configs include adapters; DinoML should reject or separately audit adapter configs first.
- `UniSpeechForSequenceClassification` supports optional `use_weighted_layer_sum`, which requires all hidden states, stack over layer axis, softmax layer weights, weighted sum, projection, pooling, and classifier.
- Pretraining uses Gumbel vector quantization, argmax/scatter in eval, random Bernoulli replacement, and CTC projection. It is not needed for first ASR inference.
- Sampled configs contain extra metadata fields that current source either inherits generically from `PreTrainedConfig` or does not use in `modeling_unispeech.py`; do not convert such fields into required graph behavior without a source read.
- Layout trap: source Conv1d regions are NCL (`[B, C, T]`) and encoder regions are BTC (`[B, T, C]`). Any channel-last/NHWC-style optimization must be local and must rewrite Conv1d, LayerNorm, GroupNorm, transpose, attention, and pooling axes together.

## 4. Operator coverage checklist

Tensor/layout ops:

- Unsqueeze raw waveform `[B, T_audio] -> [B, 1, T_audio]`.
- Transpose `[B, C, T] <-> [B, T, C]`.
- Reshape/view Q/K/V projections to `[B, T, num_heads, head_dim]`, then transpose to `[B, heads, T, head_dim]`.
- Contiguous after attention transpose.
- Boolean mask expansion/repeat for padded feature frames.
- In-place-equivalent masked fill/indexed zeroing can be lowered to guarded elementwise select.
- Stack/sum for optional weighted layer sum in classification.

Neural network primitives:

- Conv1d no padding for feature extractor: layer 0 `1 -> 512`, kernel 10, stride 5; layers 1-4 `512 -> 512`, kernel 3, stride 2; layers 5-6 `512 -> 512`, kernel 2, stride 2; bias from `conv_bias` default false.
- Feature Conv1d norm variants: GroupNorm with `num_groups=out_channels` for first layer in `"group"` mode, or LayerNorm over channels after each conv in `"layer"` mode.
- GELU or configured `feat_extract_activation` after each Conv1d.
- Feature projection LayerNorm over 512, Linear `512 -> hidden_size`, dropout disabled at inference.
- Positional grouped Conv1d: `hidden_size -> hidden_size`, kernel `num_conv_pos_embeddings`, padding floor half, groups `num_conv_pos_embedding_groups`, weight norm, same-pad crop for even kernels, activation.
- Encoder LayerNorm over `hidden_size`.
- Linear projections: Q/K/V/O all `hidden_size -> hidden_size`, bias true.
- FFN: Linear `hidden_size -> intermediate_size`, activation `hidden_act`, Linear `intermediate_size -> hidden_size`.
- CTC head: Linear `hidden_size -> vocab_size`.
- Sequence classification optional: Linear `hidden_size -> classifier_proj_size`, pooled mean or mask-aware mean, Linear `classifier_proj_size -> num_labels`.

Attention primitives:

- Dense bidirectional self-attention over feature frames.
- Eager path: `Q @ K^T * head_dim^-0.5`, add mask, softmax over key length, `P @ V`, output projection.
- Source advertises FlashAttention, SDPA, and FlexAttention support through Transformers attention interfaces, but the semantic graph remains dense bidirectional attention with padding mask.

Position/custom math:

- Weight-normalized grouped Conv1d positional embedding.
- Conv output-length and feature-vector attention-mask conversion.

Preprocessing-coupled ops:

- CPU/data-pipeline mono waveform validation.
- Sampling-rate equality guard, usually 16000 Hz.
- Right padding with value 0.
- Optional attention mask returned by processor; sampled checkpoints set `return_attention_mask=True`.
- Zero-mean/unit-variance normalization over valid samples, with padded tail reset to `padding_value`.

Training/pretraining-only or optional:

- SpecAugment random span mask generation.
- Gumbel softmax/product quantizer.
- Argmax + scatter one-hot in quantizer eval.
- CTC loss and label masking.
- Dropout and LayerDrop are inactive for inference.

## 5. Layer/block breakdown

Feature encoder:

```text
input_values: [B, T_audio]
x = input_values[:, None]                         # [B, 1, T_audio]
for i in 0..6:
  x = Conv1d(C_in -> 512, kernel=conv_kernel[i], stride=conv_stride[i], padding=0, bias=conv_bias)(x)
  if feat_extract_norm == "layer":
    x = transpose(x, -2, -1)                      # [B, T_i, 512]
    x = LayerNorm(512)(x)
    x = transpose(x, -2, -1)                      # [B, 512, T_i]
  elif i == 0 and feat_extract_norm == "group":
    x = GroupNorm(num_groups=512, num_channels=512)(x)
  x = activation(x)
extract_features_ncl = x                          # [B, 512, T_feat]
extract_features = transpose(x, 1, 2)             # [B, T_feat, 512]
```

Feature projection:

```text
norm_extract_features = LayerNorm(512)(extract_features)
hidden_states = Linear(512 -> hidden_size)(norm_extract_features)
```

Stable encoder block, used by sampled large configs:

```text
before layers:
  hidden_states = hidden_states + PositionalConv(hidden_states)
  hidden_states = Dropout(hidden_states)           # no-op in eval

repeated num_hidden_layers times:
  residual = hidden_states
  hidden_states = LayerNorm(hidden_size)(hidden_states)
  attn = MHA(hidden_states, bidirectional_mask)
  hidden_states = residual + attn
  hidden_states = hidden_states + FFN(LayerNorm(hidden_size)(hidden_states))
  if adapter_attn_dim is not None:
    hidden_states = hidden_states + Adapter(hidden_states)

after layers:
  hidden_states = LayerNorm(hidden_size)(hidden_states)
```

Non-stable encoder block, source default:

```text
before layers:
  zero padded frames if attention_mask is present
  hidden_states = hidden_states + PositionalConv(hidden_states)
  hidden_states = LayerNorm(hidden_size)(hidden_states)
  hidden_states = Dropout(hidden_states)

repeated num_hidden_layers times:
  residual = hidden_states
  hidden_states = MHA(hidden_states, bidirectional_mask)
  hidden_states = residual + Dropout(hidden_states)
  hidden_states = LayerNorm(hidden_size)(hidden_states)
  hidden_states = hidden_states + FFN(hidden_states)
  hidden_states = LayerNorm(hidden_size)(hidden_states)
```

CTC head:

```text
hidden_states = Dropout(encoder_last_hidden_state) # no-op in eval
logits = Linear(hidden_size -> vocab_size)(hidden_states) # [B, T_feat, vocab_size]
```

Pretraining head, deferred:

```text
quantized = GumbelVectorQuantizer(norm_extract_features)
quantized = Linear(codevector_dim -> proj_codevector_dim)(quantized)
quantized = Linear(proj_codevector_dim -> hidden_size)(quantized)
logits_source = bernoulli replace between transformer features and quantized features
ctc_logits = Linear(hidden_size -> num_ctc_classes)(logits_source)
```

## 6. Attention requirements

UniSpeech requires encoder-style noncausal self-attention only for the primary ASR target.

| Requirement | Source-derived behavior |
| --- | --- |
| Causal? | No, bidirectional mask |
| Cross-attention? | Class supports `key_value_states`, but UniSpeech encoder path does not use cross-attention |
| MHA/MQA/GQA | MHA only |
| Heads/head dim | source default 12 x 64; sampled large checkpoints 16 x 64 |
| Q/K/V width | all `hidden_size`; source validates divisibility by `num_attention_heads` |
| Query and KV lengths | same `T_feat` in encoder self-attention |
| Masking | padding mask converted from raw-sample mask to feature-frame mask, then passed through `create_bidirectional_mask` |
| Packed/varlen | no model-specific packed sequence metadata |
| Sliding/local | none |
| RoPE/relative bias | none |
| KV cache | not applicable |
| Flash/SDPA compatibility | source declares support and dispatches through Transformers attention interface |

Attention math order in eager source:

```python
scores = matmul(query, key.transpose(2, 3)) * (head_dim ** -0.5)
scores = scores + attention_mask   # when present
probs = softmax(scores, dim=-1)
attn = matmul(probs, value)
attn = attn.transpose(1, 2).contiguous().reshape(B, T, hidden_size)
out = out_proj(attn)
```

For optimized attention, DinoML can lower this to dense bidirectional SDPA/FlashAttention when padding-mask semantics match. There is no decode path, no cache reorder, and no growing sequence state.

## 7. Position encoding and custom math

UniSpeech has convolutional positional embeddings:

```python
def unispeech_pos_conv(x_btc, conv, groups, kernel, activation):
    y = x_btc.transpose(1, 2)       # [B, H, T]
    y = conv1d_weight_norm(y, padding=kernel // 2, groups=groups)
    if kernel % 2 == 0:
        y = y[:, :, :-1]
    y = activation(y)
    return y.transpose(1, 2)        # [B, T, H]
```

Weight norm is applied to the positional convolution weight at module construction. For inference import, DinoML should either materialize the normalized effective Conv1d weight or preserve `weight_g`/`weight_v` metadata and compute:

```text
weight = weight_g * weight_v / norm(weight_v, dim=2, keepdim=True)
```

The feature output length helper is source-significant because it maps waveform masks to feature masks:

```python
def feat_len(input_length):
    for kernel, stride in zip(conv_kernel, conv_stride):
        input_length = floor((input_length - kernel) / stride) + 1
    return input_length
```

Feature-vector attention mask construction:

```python
non_padded = attention_mask.cumsum(dim=-1)[:, -1]
out_len = feat_len(non_padded)
mask = zeros([B, T_feat])
mask[arange(B), out_len - 1] = 1
mask = flip(cumsum(flip(mask, [-1]), -1), [-1]).bool()
```

This is a prefix mask for right-padded audio. First integration can replace it with a direct length-to-prefix-mask helper under a right-padding guard.

## 8. Preprocessing and input packing

Processor source is `Wav2Vec2FeatureExtractor`.

Input waveform contract:

- Raw mono speech only; batched numpy rank greater than 2 is rejected.
- Input values are converted to float32.
- Sampling rate should be passed; if supplied and unequal to processor sampling rate, source raises. Sampled configs use 16000 Hz.
- Padding is right-sided in sampled configs, value 0.
- `attention_mask` dtype is int32 in the processor output when returned.
- `do_normalize=True` in sampled configs. With an attention mask, normalization uses only valid samples, then resets padded tail to padding value. Without padding, each sequence normalizes over the whole array.

No STFT, FFT, mel filterbank, clamp, spectrogram, chunk packing, overlap metadata, or reassembly metadata is used. The neural model consumes raw waveform samples directly.

Model input ABI for first DinoML integration:

```text
input_values: float32 [B, T_audio]
attention_mask: optional int/bool [B, T_audio], right-padded prefix semantics
output logits for CTC: float [B, T_feat, vocab_size]
```

CPU/data-pipeline recommended ownership:

- Audio decode and resampling.
- Padding/truncation policy.
- Zero-mean/unit-variance normalization.
- Tokenizer and CTC postprocessing.

GPU/runtime recommended ownership:

- Conv feature extraction.
- Raw attention-mask length reduction or a compiled length-to-feature-prefix-mask helper if `attention_mask` is supplied.
- Encoder and head.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature Conv1d stack as provider Conv1d or im2col GEMM

Source pattern:

```text
[B, C_in, T] -> Conv1d(kernel=k, stride=s, padding=0, groups=1, bias=conv_bias) -> norm/activation
```

Replacement pattern:

```text
Conv1d provider path, or guarded Unfold1d -> GEMM(weight_flat.T) -> optional BiasAdd -> activation
```

Preconditions:

- Static kernel, stride, dilation 1, padding 0.
- `groups == 1` for feature extractor convs.
- Input layout is source NCL.
- Output length must use floor formula exactly.

Shape equations:

```text
T_out = floor((T_in - kernel) / stride) + 1
weight: [C_out, C_in, K]
im2col: [B * T_out, C_in * K]
output: [B, C_out, T_out]
```

Weight transform:

```python
w_flat = conv.weight.reshape(C_out, C_in * K)
```

Failure cases:

- Dynamic input too short for any conv layer.
- Future configs with grouped feature convs.
- Nonzero padding/dilation.

Parity test sketch:

- Compare each conv layer output against PyTorch for random `[B, T]` lengths and sampled config kernels/strides.

### Rewrite: positional grouped Conv1d weight norm materialization

Source pattern:

```text
weight_norm(Conv1d(H -> H, kernel=K, padding=K//2, groups=G), dim=2)
-> crop last frame when K is even
-> activation
```

Replacement pattern:

```text
precompute normalized grouped-conv weight at load time
-> grouped Conv1d
-> static crop
-> activation
```

Preconditions:

- Inference weights are frozen.
- `num_conv_pos_embeddings` and `num_conv_pos_embedding_groups` are static.
- Materialized weight matches PyTorch weight-norm dim 2.

Shape equations:

```text
input [B, T, H] -> transpose [B, H, T]
conv output length before crop = T + (K % 2 == 0 ? 1 : 0)
after crop = T
```

Failure cases:

- Runtime-mutated weight norm parameters.
- Layout pass that forgets crop for even kernels.

Parity test sketch:

- Compare positional embedding output before encoder add for K=128 and an odd K synthetic config.

### Rewrite: prefix feature mask from lengths

Source pattern:

```text
cumsum raw attention mask -> output_lengths -> scatter one at length-1 -> reverse cumsum -> bool prefix mask
```

Replacement pattern:

```text
valid_raw_lengths = sum(attention_mask, axis=-1)
valid_feat_lengths = conv_length_chain(valid_raw_lengths)
feature_mask[b, t] = t < valid_feat_lengths[b]
```

Preconditions:

- Raw `attention_mask` is right-padded prefix mask.
- Conv stack is the source stack with no padding.
- `output_lengths > 0`; reject audio too short otherwise.

Failure cases:

- Non-prefix masks.
- Left padding or arbitrary interior gaps.

Parity test sketch:

- Generate variable right-padded batch masks, compare source mask and replacement.

### Rewrite: MHA QKV projection packing

Source pattern:

```text
q = Linear(H -> H)
k = Linear(H -> H)
v = Linear(H -> H)
reshape each to [B, heads, T, head_dim]
```

Replacement pattern:

```text
single packed Linear(H -> 3H) -> split [q, k, v] in all-Q/all-K/all-V order
```

Preconditions:

- Same input tensor for Q/K/V, which is true for encoder self-attention.
- All projections have bias and identical input/output width.
- Packed weight order must be `[q_proj, k_proj, v_proj]`, not interleaved by head.

Weight transform:

```python
w_qkv = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
b_qkv = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
```

Failure cases:

- Cross-attention path with separate `key_value_states`.
- Any future checkpoint with missing projection bias.

Parity test sketch:

- Compare Q/K/V tensors before attention for random hidden states.

### Rewrite: CTC inference skips training-only log-softmax/loss

Source pattern:

```text
logits = lm_head(hidden_states)
if labels: log_softmax + ctc_loss
```

Replacement pattern:

```text
inference returns logits only; CTC decode/postprocess outside compiled graph
```

Preconditions:

- `labels is None`.
- End-to-end parity test decodes logits with same tokenizer/blank id.

Failure cases:

- Training/eval loss parity requested.

### Layout guidance: NCL Conv1d and BTC encoder boundary

Source pattern:

```text
feature convs use [B, C, T]
feature projection/attention use [B, T, C]
```

Optimization opportunity:

- Keep semantic lowering faithful first.
- A local layout pass may keep conv outputs in a provider-preferred layout only if it rewrites the immediate transpose, LayerNorm axis, attention reshape axes, and pooling axes.

No-layout-translation guard:

- The feature-mask length math, sequence pooling `dim=1`, softmax `dim=-1`, and CTC logits `[B, T_feat, vocab]` ABI must retain source axis meaning.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d + activation stack, with LayerNorm/GroupNorm variants. This is the front-end bottleneck for raw waveform and has seven small-channel strided convs.
- Feature projection LayerNorm + Linear. Standard audio encoder transition and easy parity boundary.
- Packed QKV projection + bidirectional attention + output projection. Sampled large configs have 24 layers and sequence lengths proportional to audio duration.
- LayerNorm + FFN Linear/GELU/Linear residual block. Large hidden size 1024 and FFN 4096 dominate encoder math.

Medium priority:

- Positional grouped Conv1d with materialized weight norm, crop, activation, and add.
- Prefix feature-mask generation from raw attention lengths.
- CTC head Linear plus optional log-softmax for decode pipelines that want normalized log-probs.
- Masked zeroing of padded frames as fused select before encoder.

Lower priority:

- Sequence classification weighted-layer sum and mask-aware mean pooling.
- Pretraining quantizer, perplexity, and Bernoulli replacement.
- Adapter layer path behind `adapter_attn_dim`.
- Training-only CTC loss, SpecAugment, dropout, and LayerDrop.

## 11. Runtime staging plan

Stage 1: config and preprocessing ABI.

- Parse `UniSpeechConfig`.
- Load Wav2Vec2 feature extractor config.
- Support raw waveform `[B, T]`, 16 kHz guard, optional attention mask, and output-length math.
- Stub training-only SpecAugment/dropout/loss.

Stage 2: feature encoder parity.

- Implement Conv1d stack, norm variant, activation, transpose, and output-length checks.
- Validate `extract_features` against PyTorch for source default and sampled large config.

Stage 3: encoder block parity.

- Implement feature projection, positional Conv1d, stable and non-stable encoder layer order.
- Run one-layer and full-encoder parity with eager attention.

Stage 4: CTC inference.

- Add `lm_head`, logits ABI `[B, T_feat, vocab_size]`, tokenizer/blank metadata handoff.
- Validate sampled CTC checkpoints.

Stage 5: attention/provider optimization.

- Lower MHA to DinoML attention primitive or SDPA/Flash-compatible provider path.
- Add QKV packing and FFN fusions under strict weight/order guards.

Stage 6: optional heads.

- Add sequence classification pooling path.
- Separately decide whether pretraining quantizer has runtime value.

Stage 7: production audio batching.

- Bucket audio lengths by feature-frame count.
- Benchmark raw conv, encoder, and decode separately.

## 12. Parity and validation plan

- Processor parity: compare normalized padded `input_values` and `attention_mask` with `Wav2Vec2FeatureExtractor` for mono arrays, batched arrays, truncation, and padding.
- Output-length parity: random valid raw lengths through `_get_feat_extract_output_lengths`; include short-input rejection cases.
- Feature-mask parity: right-padded masks with varied lengths; compare source scatter/reverse-cumsum mask with DinoML prefix-mask helper.
- Conv layer parity: compare each feature conv block output for `feat_extract_norm="layer"` and `"group"` configs.
- Positional conv parity: compare materialized weight-norm Conv1d output for kernel 128 and synthetic odd kernel.
- Attention parity: one encoder layer, eager attention, masks on/off, fp32 tolerance about `1e-5` absolute/relative.
- Full encoder parity: `UniSpeechModel` last hidden state for a short random waveform batch; fp32 target `1e-4` to start because Conv1d/attention accumulation order may differ.
- CTC head parity: sampled fine-tuned checkpoint logits shape and values for short audio; decode a tiny known sample if fixture audio is available.
- Classification optional: mask-aware mean pooling and weighted-layer sum parity with synthetic config.
- Deferred pretraining: quantizer eval path parity for argmax/scatter/product-codebook only if pretraining is admitted.

Recommended reduced-precision approach: establish fp32 first. For fp16/bf16, keep LayerNorm/softmax accumulation in fp32 and accept looser tolerances around `1e-2` until provider behavior is characterized.

## 13. Performance probes

- Preprocessing throughput: audio decode/resample/normalize samples per second on CPU.
- Conv feature encoder throughput by audio seconds and batch size.
- Encoder-only throughput by `[B, T_feat, H]`, especially one-second, ten-second, and long-form audio.
- Attention backend comparison: eager GEMM+softmax, SDPA, FlashAttention-style bidirectional with padding mask.
- Batch-size sweep: `B=1,2,4,8,16` at fixed audio length.
- Sequence-length sweep: 1 s, 5 s, 10 s, 30 s, and padded mixed-length batches.
- Mask overhead: no mask vs right-padded mask vs feature prefix-mask helper.
- Conv layout variants: source NCL, provider-preferred layout, and any local transposition elimination.
- CTC head/log-softmax throughput by vocab size 43/45/46/65 and pretraining classes 80/531.
- Memory probes: hidden-state footprint for optional `output_hidden_states` and weighted layer sum.

## 14. Skip/defer list

- Training, gradients, gradient checkpointing.
- SpecAugment random masking.
- LayerDrop/dropout stochastic behavior.
- CTC loss.
- Pretraining quantizer, Gumbel softmax, perplexity, Bernoulli replacement, contrastive loss.
- Sequence classification unless a target requires it.
- Adapter loading and `target_lang` path until a checkpoint with `adapter_attn_dim` is selected.
- Cross-attention branch in `UniSpeechAttention`; not used by UniSpeech encoder.
- Beam search/language-model CTC decoding; keep as postprocessing/controller work.
- Multi-GPU/FSDP/DeepSpeed-specific synchronization.
- Gated/remote-code behavior; none observed for sampled official UniSpeech configs.

## 15. Final implementation checklist

- [ ] Parse `UniSpeechConfig`, including conv lists, stable-LN flag, norm mode, vocab size, and pad/blank id.
- [ ] Load Wav2Vec2 feature extractor config for sampling rate, normalization, padding, and attention mask policy.
- [ ] Implement Conv1d feature encoder with `"layer"` and `"group"` norm variants.
- [ ] Implement conv output-length helper and right-padding feature-mask helper.
- [ ] Implement feature projection LayerNorm + Linear.
- [ ] Implement positional grouped Conv1d with weight-norm materialization and even-kernel crop.
- [ ] Implement stable and non-stable encoder block ordering.
- [ ] Implement dense bidirectional MHA with padding mask.
- [ ] Add packed-QKV rewrite under self-attention guards.
- [ ] Implement FFN Linear + GELU + Linear and fusion candidate.
- [ ] Implement CTC head logits ABI.
- [ ] Add processor parity tests for waveform normalization/padding.
- [ ] Add feature encoder, one-block, full-encoder, and CTC-logits parity tests.
- [ ] Benchmark preprocessing, Conv1d stack, encoder attention, FFN, and CTC head separately.
