# Wav2Vec2 Transformers Family Audit

Primary target: audio encoder plus CTC inference/ASR on CUDA. This report treats tokenizer decoding, CTC loss, pretraining quantization, and classification heads as secondary unless they affect the encoder/logit contract.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: wav2vec2
Primary model id: facebook/wav2vec2-base-960h
Primary task: raw waveform -> encoder frames -> CTC logits
Config source: Hugging Face config.json and preprocessor_config.json files fetched from model repos listed below.
Source files inspected:
- X:/H/transformers/src/transformers/models/wav2vec2/configuration_wav2vec2.py
- X:/H/transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
- X:/H/transformers/src/transformers/models/wav2vec2/processing_wav2vec2.py
- X:/H/transformers/src/transformers/models/wav2vec2/tokenization_wav2vec2.py
- X:/H/transformers/src/transformers/models/wav2vec2/modeling_wav2vec2.py
- X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions: no remote-code files are needed for the inspected checkpoints. Processor configs are usually absent; Wav2Vec2Processor is assembled from feature_extractor + tokenizer files.
```

Representative checkpoint configs inspected:

| Model id | Source | Why included |
|---|---|---|
| `hf-internal-testing/tiny-random-wav2vec2` | HF `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | tiny/debug shape variant |
| `facebook/wav2vec2-base-960h` | HF `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | common CTC production baseline |
| `facebook/wav2vec2-large-960h-lv60-self` | HF `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | large stable LayerNorm + attention-mask baseline |
| `facebook/wav2vec2-xls-r-300m` | HF `config.json`, `preprocessor_config.json` | pretraining/base encoder variant with `vocab_size=null` |
| `facebook/mms-1b-all` | HF `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | large multilingual CTC family with adapter-attention metadata |

Relevant source anchors at this commit:

- `Wav2Vec2Config`: `configuration_wav2vec2.py:27`, default dims at `:165`.
- `Wav2Vec2FeatureExtractor`: `feature_extraction_wav2vec2.py:28`.
- Conv feature encoder/projection/positional conv: `modeling_wav2vec2.py:254`, `:382`, `:422`, `:326`.
- Attention/encoder blocks: `modeling_wav2vec2.py:466`, `:575`, `:611`, `:657`, `:729`.
- Length/mask reduction: `modeling_wav2vec2.py:1004`, `:1025`.
- Base model/CTC head: `modeling_wav2vec2.py:1251`, `:1604`.
- Bidirectional attention mask factory: `masking_utils.py:1019`.

## 2. High-level architecture

Wav2Vec2 is an audio-only encoder. For ASR it is:

```text
CPU/data pipeline raw mono waveform padding/normalization
-> Conv1d feature extractor over waveform samples
-> time-major feature projection and LayerNorm
-> convolutional positional embedding
-> noncausal Transformer encoder
-> optional adapter stack
-> dropout
-> Linear CTC vocabulary head
-> CTC decode outside core model
```

Stage decomposition:

| Stage | Runtime boundary | Notes |
|---|---|---|
| Audio loading/resampling | CPU/data pipeline | Source expects mono float waveform at configured sampling rate, normally 16 kHz. It does not perform resampling, STFT, FFT, mel, or windowing. |
| Feature extractor processor | CPU/data pipeline by default | Pads/truncates, casts to `float32`, optional zero-mean/unit-variance normalization, optional attention mask. |
| Conv feature encoder | GPU graph target | Seven Conv1d layers for common checkpoints, input `[B, T]` becomes `[B, C=512, T']`. |
| Feature projection | GPU graph target | Transpose to `[B, T', C]`, LayerNorm over C, Linear `C -> hidden_size`. |
| Encoder | GPU graph target | Positional depth/group Conv1d, MHA, FFN, LayerNorm. No autoregressive cache. |
| CTC logits | GPU graph target | Linear `hidden/output_hidden_size -> vocab_size`, logits `[B, T'', vocab]`. |
| CTC decode | Controller/postprocess | Greedy/beam CTC collapse and tokenizer decoding are outside the neural graph. |

Validation can be split cleanly: processor tensor parity, convolutional length parity, encoder-frame parity, CTC logit parity, then decode parity.

## 3. Important config dimensions

Source defaults from `Wav2Vec2Config`:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 32 | CTC head output classes; can be `null` for pretraining-only encoders. |
| `hidden_size` | 768 | Encoder width and attention projection width. |
| `num_hidden_layers` | 12 | Transformer block count. |
| `num_attention_heads` | 12 | MHA heads; `head_dim = hidden_size / heads`. |
| `intermediate_size` | 3072 | FFN expansion. |
| `hidden_act` | `gelu` | FFN activation and default conv activation. |
| `conv_dim` | `[512]*7` | Conv1d channel ladder; first layer input channel is 1. |
| `conv_stride` | `[5,2,2,2,2,2,2]` | Length reduction. Product is 320, but exact length uses floor per layer. |
| `conv_kernel` | `[10,3,3,3,3,2,2]` | Conv1d kernels, no padding in feature extractor. |
| `conv_bias` | `False` | Bias may become true in large/stable checkpoints. |
| `feat_extract_norm` | `group` | First conv only gets GroupNorm for `group`; every conv gets transpose + LayerNorm for `layer`. |
| `num_conv_pos_embeddings` | 128 | Positional Conv1d kernel with same-pad crop if even. |
| `num_conv_pos_embedding_groups` | 16 | Grouped positional Conv1d. |
| `do_stable_layer_norm` | `False` | Switches encoder block norm order and final encoder norm. |
| `add_adapter` | `False` | Optional post-encoder Conv1d+GLU stack; can reduce time length further. |
| `adapter_stride` / `num_adapter_layers` | 2 / 3 | Adapter CTC frame-rate change if enabled. |
| `output_hidden_size` | defaults to `hidden_size` | CTC head input when adapters are enabled. |
| `adapter_attn_dim` | `None` | Optional per-layer attention adapter bottleneck. |

Representative sweep:

| Model id | Arch | Hidden | Layers | Heads | Head dim | FFN | Conv norm/bias | Stable LN | CTC vocab | Preprocessor mask |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| tiny-random | unspecified | 16 | 4 | 2 | 8 | 20 | layer / false, 3 convs | true | 32 | false |
| wav2vec2-base-960h | CTC | 768 | 12 | 12 | 64 | 3072 | group / false | false | 32 | false |
| wav2vec2-large-960h-lv60-self | CTC | 1024 | 24 | 16 | 64 | 4096 | layer / true | true | 32 | true |
| wav2vec2-xls-r-300m | pretraining | 1024 | 24 | 16 | 64 | 4096 | layer / true | true | null | true |
| mms-1b-all | CTC | 1280 | 48 | 16 | 80 | 5120 | layer / true | true | 154 | true |

Omitted fields in common configs inherit source defaults. Notable omissions: base/large configs omit adapter and classification fields, so `add_adapter=False`, `adapter_stride=2`, `num_adapter_layers=3`, `output_hidden_size=hidden_size`, `classifier_proj_size=256`, and TDNN defaults apply if those heads are instantiated.

## 3a. Family variation traps

- `feat_extract_norm` is structurally significant. `group` uses GroupNorm only on conv layer 0; `layer` uses LayerNorm after every conv with `NCL -> NLC -> NCL` transposes.
- Feature extractor attention-mask policy is model-coupled. HF docs in source say `group` models such as base should normally be padded with zeros and called without `attention_mask`; `layer` models should receive an attention mask for batched inference.
- `conv_bias` changes between base (`False`) and large/XLS-R/MMS (`True`).
- Exact frame count is not simply `ceil(T/320)` or `T/320`; it is repeated floor Conv1d output length.
- `do_stable_layer_norm` changes block order. Base uses attention -> residual -> LN -> FFN -> final LN; stable models pre-norm attention and FFN then apply final encoder LN after all layers.
- No KV cache, causal mask, RoPE, ALiBi, GQA, or MQA for the encoder target. Attention is full bidirectional MHA.
- `vocab_size` may be `null` for pretraining-only checkpoints. `Wav2Vec2ForCTC` rejects such configs unless a vocab is supplied.
- Adapters come in two flavors: post-encoder Conv1d+GLU adapters controlled by `add_adapter`, and per-encoder-layer attention adapters controlled by `adapter_attn_dim`. MMS config exposes `adapter_attn_dim=16` but `add_adapter=false`.
- Adapter Conv1d stack changes output length: `_get_feat_extract_output_lengths(..., add_adapter=True)` applies `num_adapter_layers` additional kernel-1, stride-`adapter_stride` reductions.
- Positional Conv1d uses weight norm and grouped Conv1d with `padding=kernel//2`, then drops one trailing frame for even kernels.
- Layout-axis traps are heavy: waveform `[B,T]` -> Conv1d `[B,C,T]` -> encoder `[B,T,C]` -> attention `[B,H,T,D]`. A channel-last optimization must be local and must rewrite every `transpose`, LayerNorm axis, GroupNorm/channel axis, GLU dim, mean/sum dim, and softmax dim.

## 4. Operator coverage checklist

Required for audio encoder + CTC inference:

Tensor/layout ops:

- `unsqueeze(input_values, dim=1)` from `[B,T]` to `[B,1,T]`.
- `transpose(1,2)` between `[B,C,T]` and `[B,T,C]`.
- `view/reshape` for Q/K/V `[B,T,Hid] -> [B,T,H,D]`.
- `transpose(1,2)` for attention `[B,T,H,D] -> [B,H,T,D]`.
- `contiguous/reshape` attention output `[B,H,T,D] -> [B,T,Hid]`.
- Mask expansion/repeat, boolean indexing or equivalent multiply/select, cumsum/flip for feature-vector masks.

Neural network primitives:

- Conv1d feature extractor, e.g. base: `1->512 k10 s5`, then six `512->512` convs with kernels `[3,3,3,3,2,2]`, strides `[2,2,2,2,2,2]`, no padding.
- GroupNorm with `num_groups=channels` for base conv layer 0.
- LayerNorm over channel dimension for `feat_extract_norm="layer"` convs and for feature projection/encoder.
- Linear projection `conv_dim[-1] -> hidden_size`: base `512 -> 768`, large `512 -> 1024`, MMS `512 -> 1280`.
- GELU activations in conv and FFN paths.
- FFN linears: base `768 -> 3072 -> 768`; large `1024 -> 4096 -> 1024`; MMS `1280 -> 5120 -> 1280`.
- CTC head: base/large `hidden -> 32`; MMS `1280 -> 154`.
- Dropout is inactive in inference but should be represented or erased under eval.

Attention primitives:

- Noncausal self-attention MHA.
- Q/K/V/out linears all `hidden_size -> hidden_size` with bias.
- Attention score scale `head_dim ** -0.5`.
- Additive 4D bidirectional padding mask for eager/SDPA paths when mask is present.
- Softmax over key length, attention-value BMM, output projection.

Position/custom math:

- Grouped Conv1d positional embedding with weight normalization.
- Same-pad crop for even `num_conv_pos_embeddings`.

Preprocessing-coupled ops:

- Per-example waveform normalization `(x - mean(valid)) / sqrt(var(valid) + 1e-7)`.
- Right padding with `padding_value=0.0`.
- Optional attention-mask length reduction through Conv1d length formula.

Optional/deferred heads:

- Sequence classification: weighted layer sum, projection, masked mean pooling, classifier.
- Audio frame classification: weighted layer sum, per-frame classifier.
- XVector: TDNN Conv1d-equivalent layers, statistic pooling, AMSoftmax loss.
- Pretraining: Gumbel vector quantizer, contrastive logits, negative sampling.

## 5. Layer/block breakdown

Feature extractor:

```text
input_values: [B, T_raw] float32
x = input_values[:, None]                         # [B, 1, T_raw]
for i in conv layers:
  x = Conv1d(Cin -> Cout, kernel=conv_kernel[i], stride=conv_stride[i], padding=0, bias=conv_bias)(x)
  if feat_extract_norm == "group" and i == 0:
    x = GroupNorm(num_groups=Cout, num_channels=Cout)(x)
  if feat_extract_norm == "layer":
    x = transpose(x, [B,C,T] -> [B,T,C])
    x = LayerNorm(Cout)(x)
    x = transpose(x, [B,T,C] -> [B,C,T])
  x = GELU(x)
extract_features = transpose(x, [B,C,T'] -> [B,T',C])
```

Feature projection:

```text
norm_features = LayerNorm(conv_dim[-1])(extract_features)
hidden = Linear(conv_dim[-1] -> hidden_size)(norm_features)
hidden = Dropout(hidden)      # erased in eval
```

Encoder, base/post-norm form, repeated `num_hidden_layers`:

```text
pos = PositionalConv(hidden)                         # [B,T',H]
hidden = LayerNorm(hidden + pos)
hidden = Dropout(hidden)

for block:
  residual = hidden
  attn = MHA(hidden, bidirectional_padding_mask)
  hidden = residual + Dropout(attn)
  hidden = LayerNorm(hidden)
  hidden = hidden + FFN(hidden)
  hidden = LayerNorm(hidden)
```

Encoder, stable/pre-norm form:

```text
hidden = hidden + PositionalConv(hidden)
hidden = Dropout(hidden)
for block:
  residual = hidden
  attn = MHA(LayerNorm(hidden), mask)
  hidden = residual + Dropout(attn)
  hidden = hidden + FFN(LayerNorm(hidden))
  if adapter_attn_dim is not None:
    hidden = hidden + Linear(adapter_dim -> hidden)(ReLU(Linear(hidden -> adapter_dim)(LayerNorm(hidden))))
hidden = final LayerNorm(hidden)
```

CTC head:

```text
hidden = Wav2Vec2Model(input_values, attention_mask).last_hidden_state
hidden = Dropout(hidden)        # erased in eval
logits = Linear(output_hidden_size or hidden_size -> vocab_size)(hidden)
```

For a 16 kHz one-second input (`T_raw=16000`) with the common 7-conv stack, exact feature lengths are `[3199,1599,799,399,199,99,49]`, so CTC logits are `[B,49,V]` before any adapter downsampling.

## 6. Attention requirements

| Property | Requirement |
|---|---|
| Type | Encoder-only bidirectional self-attention. |
| Causality/cache | Noncausal; no KV cache for primary target. |
| Heads | MHA only; `num_key_value_heads` is not present. |
| Head dim | `hidden_size // num_attention_heads`; source validates divisibility. |
| Projections | Separate Q, K, V, O linears, all bias-enabled. |
| Masking | Reduced waveform attention mask becomes feature-frame boolean mask, then `create_bidirectional_mask` creates backend-specific mask. Eager mask uses 0 for valid and dtype min for invalid positions. |
| Backend compatibility | Source advertises flash-attn, SDPA, and flex-attn support through `ALL_ATTENTION_FUNCTIONS` and `_supports_*` flags. |
| Dropout | Attention dropout is `0.0` in eval, `config.attention_dropout` in training. |
| Output attentions | For some optimized attention implementations, requesting attentions can force/effect fallback depending on global Transformers behavior. First DinoML path can omit attention weights. |

Eager math order inferred from source:

```text
q = Linear(x).view(B,T,H,D).transpose(1,2)
k = Linear(x).view(B,T,H,D).transpose(1,2)
v = Linear(x).view(B,T,H,D).transpose(1,2)
attn = attention_interface(q, k, v, mask, scaling=head_dim**-0.5)
out = attn.reshape(B,T,H*D).contiguous()
out = Linear(out)
```

## 7. Position encoding and custom math

Wav2Vec2 uses convolutional positional embeddings, not RoPE/ALiBi/absolute learned tables.

Short parity snippet:

```python
def wav2vec2_pos_conv(x, weight_norm_conv, activation, num_conv_pos_embeddings):
    # x: [B, T, H]
    y = x.transpose(1, 2)                     # [B, H, T]
    y = weight_norm_conv(y)                   # Conv1d H->H, groups=config.num_conv_pos_embedding_groups
    if num_conv_pos_embeddings % 2 == 0:
        y = y[:, :, :-1]                      # same-pad crop for even kernels
    y = activation(y)
    return y.transpose(1, 2)                  # [B, T, H]
```

Weight norm is part of the stored module parameterization. For import, DinoML should either materialize the effective Conv1d weight from HF weights or represent weight norm explicitly before lowering.

Feature length reduction:

```python
def wav2vec2_conv_length(length, kernels, strides):
    for kernel, stride in zip(kernels, strides):
        length = (length - kernel) // stride + 1
    return length
```

Feature-vector attention mask reduction:

```python
valid = attention_mask.cumsum(dim=-1)[:, -1]
out_len = wav2vec2_conv_length(valid, conv_kernel, conv_stride)
mask = zeros([B, feature_vector_length])
mask[arange(B), out_len - 1] = 1
mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
```

## 8. Preprocessing and input packing

Processor contract from `Wav2Vec2FeatureExtractor`:

| Field | Observed/default value | Runtime meaning |
|---|---|---|
| `feature_size` | 1 | Mono waveform, one float per timestep. |
| `sampling_rate` | 16000 in inspected configs | Caller must supply/resample externally; extractor validates if `sampling_rate` argument is passed. |
| `padding_side` | right | Padding extends trailing samples. |
| `padding_value` | 0.0 | Also used to reset padded normalized samples. |
| `do_normalize` | true | Per-example zero mean/unit variance with epsilon `1e-7`. |
| `return_attention_mask` | base false, large/XLS-R/MMS true | Whether processor returns waveform `attention_mask`. |

Input tensors:

```text
input_values: float32 [B, T_raw_padded]
attention_mask: optional int32/long/bool-like [B, T_raw_padded], 1 for valid waveform samples
```

No STFT/FFT/hop/window/mel/log-mel/clamp exists in Wav2Vec2 preprocessing. Audio decode, resampling, chunking/windowing policy for long audio, and CTC segment stitching are application/controller work, not model source behavior.

Tokenizer/decoder coupling for ASR:

- `Wav2Vec2Processor.__call__` can process `audio`, `text`, or both; with both, it returns `input_values` plus `labels`.
- The model does not consume text token IDs for inference.
- CTC blank id is `config.pad_token_id` in loss code; greedy decode must collapse repeats and remove blank/pad according to tokenizer behavior.
- Base/large tokenizer configs include `<pad>` and `<unk>`; MMS tokenizer config includes `word_delimiter_token="|"` and `target_lang="eng"`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1d Feature Extractor Lowering

Source pattern:

```text
[B, Cin, T] -> Conv1d(Cin,Cout,kernel,stride,padding=0,dilation=1,groups=1) -> GELU/norm
```

Replacement:

```text
im2col/window extraction over time -> GEMM/Conv provider -> optional bias -> norm -> GELU
```

Preconditions:

- `groups == 1`, `dilation == 1`, `padding == 0` for feature extractor convs.
- Static or bounded dynamic `T` with exact floor output formula.
- Preserve NCL source layout until a local conv stack layout plan is proven.

Shape equations:

```text
T_out = floor((T_in - kernel) / stride) + 1
GEMM M = B * T_out, K = Cin * kernel, N = Cout
```

Weight transform:

```python
w_gemm = conv.weight.reshape(Cout, Cin * kernel).T
```

Failure cases:

- Positional conv has `groups>1` and padding/crop, so do not apply this generic feature-extractor rewrite to it.
- LayerNorm conv variants interleave axis transposes; a fused Conv+LN requires exact channel-axis handling.

Parity test sketch: compare every conv layer output and final `extract_features` for base and large configs over padded and unpadded waveform lengths.

### Rewrite: QKV Projection Packing

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
single Linear(H, 3H) -> split q/k/v -> reshape heads
```

Preconditions:

- Self-attention only; no `key_value_states`.
- Same input tensor and dtype.
- Preserve separate bias values via concatenated bias.

Weight transform:

```python
w_qkv = concat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
b_qkv = concat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
```

Failure cases: output-attention/debug hooks that require intermediate named tensors can still be supported if split outputs are retained.

### Rewrite: Eval Dropout Erasure

Source pattern: Dropout after projection, attention, FFN, encoder input, and CTC head.

Replacement: identity in eval/inference artifacts.

Preconditions: model is compiled for inference/eval only.

### Rewrite: Feature Mask Reduction as Shape/Mask Helper

Source pattern: `attention_mask.sum/cumsum -> conv length formula -> scatter at last valid frame -> flip+cumsum+flip`.

Replacement: dedicated mask-length helper producing `[B,T_feature]` boolean mask.

Preconditions:

- Attention mask is right-padded 1/0 waveform mask.
- Conv stack parameters are known.
- `output_lengths >= 1`; reject too-short raw audio before indexing `out_len-1`.

Failure cases: non-right-padded masks or arbitrary holes in waveform mask should preserve the source cumsum behavior or be rejected.

### Rewrite: Positional Conv WeightNorm Materialization

Source pattern: parametrized weight-norm Conv1d.

Replacement: materialize effective Conv1d weight at import, lower as grouped Conv1d.

Preconditions:

- Inference-only weights are frozen.
- Materialization matches PyTorch weight norm dim `2`.

Failure cases: fine-tuning or adapter loading that mutates parametrized weight tensors after import.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d feature extractor kernels or GEMM lowering. It is the only frontend audio transform in the model and controls frame count.
- LayerNorm + Linear feature projection for `[B,T',512] -> [B,T',H]`.
- QKV packed projection + attention backend for noncausal MHA.
- FFN `Linear -> GELU -> Linear` with optional bias/dropout erased.
- CTC head Linear for large batch/frame throughput.

Medium priority:

- Positional grouped Conv1d + GELU + residual add.
- Mask reduction helper from waveform mask to feature mask.
- LayerNorm/residual fusion in base and stable encoder blocks.
- Adapter Conv1d+GLU stack for MMS/language-adapter variants.

Lower priority:

- Weighted-layer-sum heads for classification tasks.
- TDNN/XVector kernels.
- Pretraining Gumbel quantizer and contrastive negative sampling.
- CTC beam search/decoder integration; useful end-to-end, but outside the neural graph.

## 11. Runtime staging plan

1. Parse Wav2Vec2 config and feature extractor config. Enforce mono waveform, sampling-rate metadata, conv arrays, norm flavor, stable-LN flag, and CTC vocab availability.
2. Load HF weights for `Wav2Vec2ForCTC`, including positional Conv1d effective weight materialization.
3. Implement/evaluate processor parity in Python CPU pipeline: padding, attention mask, normalization.
4. Implement conv feature extractor with exact length checks and compare `extract_features`.
5. Implement feature projection and one encoder block parity for base and stable-LN variants.
6. Implement full encoder parity with bidirectional padding mask.
7. Add CTC head logits parity.
8. Add optional adapter path only after base CTC is stable.
9. Add optimized Conv1d/GEMM, QKV packing, attention, and FFN fusions behind guarded rewrites.

Initially stub/defer: tokenizer decode, CTC loss, pretraining quantizer, classification/XVector heads, training-time SpecAugment, dropout, LayerDrop.

## 12. Parity and validation plan

- Processor parity: feed ragged mono arrays through HF feature extractor and DinoML preprocessing; compare `input_values` and `attention_mask`.
- Length parity: for a sweep of raw lengths including 16000, 16001, 32000, very short invalid lengths, and padded batches, compare `_get_feat_extract_output_lengths`.
- Conv stack parity: compare each conv layer output for base (`group`, no bias) and large (`layer`, bias).
- Projection parity: compare feature LayerNorm and projection outputs.
- Positional conv parity: compare after weight-norm materialization and same-pad crop.
- Single encoder block parity: base post-norm and stable pre-norm.
- Full encoder parity: compare `last_hidden_state` for `facebook/wav2vec2-base-960h` and `facebook/wav2vec2-large-960h-lv60-self`.
- CTC logits parity: compare logits `[B,T,V]` before any decode.
- Masked batch parity: right-padded two-sample batch with `attention_mask` for layer-norm models.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for full graph; fp16/bf16 should use looser staged tolerances and keep reductions/softmax accumulations in fp32 where possible.

## 13. Performance probes

- CPU preprocessing throughput: waveform padding + normalization samples/sec.
- Conv feature extractor throughput by raw length and batch size.
- Encoder-only frames/sec for `[B,T',H]` after projection.
- Attention backend comparison for bidirectional MHA with padding masks: eager/SDPA/Flash-compatible path.
- End-to-end CTC logits throughput for 1s, 5s, 10s, and 30s clips.
- Batch-size sweep with right padding to expose wasted padded frames.
- Dynamic-length bucketing probe: group by raw length/frame count before encoder.
- Adapter path overhead for MMS-style variants if enabled.
- Memory probe for large/MMS configs: activation footprint at 24/48 layers and long audio.

No benchmark observations are included here; these are proposed probes from source-derived structure.

## 14. Skip/defer list

Safe to defer for first ASR logits integration:

- Training, gradients, dropout, LayerDrop, gradient checkpointing.
- SpecAugment masking.
- CTC loss.
- Beam search and language-model-assisted CTC decoding.
- Pretraining vector quantizer, negative sampling, contrastive loss.
- Sequence classification, audio frame classification, XVector heads.
- Adapter loading from remote `adapter.<lang>.safetensors` files, unless targeting MMS language adapters explicitly.
- Flash/flex attention-specific packed metadata beyond ordinary bidirectional padding masks.
- Quantization and multi-GPU/tensor parallel.

Do not defer:

- Raw waveform processor contract.
- Exact Conv1d stack and length reduction.
- `feat_extract_norm` group-vs-layer behavior.
- Stable LayerNorm order.
- Attention mask downsampling for layer-norm checkpoints.
- CTC Linear logits shape and vocab size.

## 15. Final implementation checklist

- [ ] Parse `Wav2Vec2Config` and `Wav2Vec2FeatureExtractor` config.
- [ ] Validate mono `float32` waveform input `[B,T]` and sampling-rate metadata.
- [ ] Implement CPU/data-pipeline padding, truncation, attention mask, and normalization parity.
- [ ] Implement Conv1d feature extractor with `group` and `layer` norm variants.
- [ ] Implement exact feature length and feature-vector mask reduction helper.
- [ ] Implement feature projection LayerNorm + Linear.
- [ ] Materialize/lower positional weight-norm grouped Conv1d with same-pad crop.
- [ ] Implement base and stable encoder block variants.
- [ ] Implement bidirectional MHA with padding mask and no cache.
- [ ] Implement FFN GELU block.
- [ ] Implement CTC dropout-erased Linear head and logits output.
- [ ] Add base and large checkpoint parity tests at processor, conv, encoder, and logits levels.
- [ ] Add layout-axis guards for NCL conv regions and BTC encoder regions.
- [ ] Add guarded Conv1d-to-GEMM and QKV-packing rewrites.
- [ ] Benchmark preprocessing, conv extractor, encoder, attention, and CTC logits separately.
