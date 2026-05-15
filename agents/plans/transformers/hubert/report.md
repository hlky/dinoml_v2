# HuBERT Transformers Family Audit

Primary target: raw waveform -> HuBERT audio encoder -> CTC logits for ASR on CUDA. The base encoder and sequence-classification head are documented because they share the same runtime body. Training-only SpecAugment, CTC loss, and tokenizer decoding are deferred for first DinoML integration unless explicitly needed.

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: hubert
Primary model id: facebook/hubert-large-ls960-ft
Primary task: raw waveform -> encoder frames -> CTC logits
Config source: Hugging Face config.json and preprocessor_config.json files fetched from model repos listed below.
Source files inspected:
- transformers/src/transformers/models/hubert/configuration_hubert.py
- transformers/src/transformers/models/hubert/modular_hubert.py
- transformers/src/transformers/models/hubert/modeling_hubert.py
- transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
- transformers/src/transformers/models/wav2vec2/processing_wav2vec2.py
- transformers/src/transformers/models/wav2vec2/tokenization_wav2vec2.py
- transformers/src/transformers/masking_utils.py
Any missing files or assumptions: no remote-code files are needed for the inspected checkpoints. `modeling_hubert.py` is generated from `modular_hubert.py`; future Transformers source edits should target the modular file, while DinoML should inspect/import the generated file behavior. HuBERT uses `Wav2Vec2FeatureExtractor`, `Wav2Vec2Processor`, and `Wav2Vec2CTCTokenizer` for processor/tokenizer coupling.
```

Representative checkpoint configs inspected:

| Model id | Source | Why included |
|---|---|---|
| `hf-tiny-model-private/tiny-random-HubertForCTC` | HF `config.json`, `preprocessor_config.json` | tiny/debug CTC shape variant |
| `facebook/hubert-base-ls960` | HF `config.json`, `preprocessor_config.json` | common base encoder, group-norm conv frontend |
| `facebook/hubert-large-ls960-ft` | HF `config.json`, `preprocessor_config.json`, `vocab.json` | common production CTC checkpoint |
| `facebook/hubert-large-ll60k` | HF `config.json`, `preprocessor_config.json` | large base encoder / pretraining-style feature extraction |
| `facebook/hubert-xlarge-ls960-ft` | HF `config.json`, `preprocessor_config.json` | xlarge CTC scaling point |
| `superb/hubert-base-superb-ks` and `superb/hubert-large-superb-er` | HF `config.json`, `preprocessor_config.json` | sequence-classification variants with weighted layer sum |

Relevant source anchors at this commit:

- `HubertConfig`: `configuration_hubert.py:27`; default dims at `:124`; conv defaults at `:141`; `inputs_to_logits_ratio` at `:183`.
- HuBERT feature/position/attention/encoder classes: `modeling_hubert.py:45`, `:178`, `:216`, `:262`, `:371`, `:407`, `:504`, `:550`.
- Length and mask reduction helpers: `modeling_hubert.py:674`, `:689`; SpecAugment helper at `:702`.
- Base model and heads: `modeling_hubert.py:822`, `:968`, `:1116`.
- Wav2Vec2 feature extractor: `feature_extraction_wav2vec2.py:28`, normalization at `:78`, input call at `:99`.
- Wav2Vec2 processor: `processing_wav2vec2.py:28`.
- Wav2Vec2 CTC tokenizer: `tokenization_wav2vec2.py:101`, CTC blank filtering around `:316`, decode around `:410`.
- Bidirectional mask factory: `masking_utils.py:1019`.

## 2. High-level architecture

HuBERT is an audio-only, bidirectional encoder. For ASR CTC inference:

```text
CPU/data pipeline raw mono waveform padding/normalization
-> Conv1d feature extractor over waveform samples
-> NCL to NLC transpose
-> feature projection LayerNorm + Linear
-> convolutional positional embedding
-> noncausal Transformer encoder
-> dropout
-> Linear CTC vocabulary head
-> CTC decode outside core model
```

Stage decomposition:

| Stage | Runtime boundary | Notes |
|---|---|---|
| Audio loading/resampling | CPU/data pipeline | Source expects mono waveform sampled at the feature extractor rate, usually 16 kHz. It does not perform resampling, STFT, FFT, mel conversion, or windowing. |
| Processor feature extraction | CPU/data pipeline by default | Pads/truncates, casts to `float32`, optional zero-mean/unit-variance normalization, optional attention mask. |
| Conv feature encoder | GPU graph target | Typical checkpoints use seven Conv1d layers from `[B,T]` to `[B,512,T']`; tiny random uses three layers. |
| Feature projection | GPU graph target | Transpose to `[B,T',C]`, optional LayerNorm over C, Linear `C -> hidden_size`. |
| HuBERT encoder | GPU graph target | Positional grouped Conv1d, full bidirectional MHA, FFN, LayerNorm. No autoregressive KV cache. |
| CTC head | GPU graph target | Linear `hidden_size -> vocab_size`, logits `[B,T',vocab]`. |
| CTC decode | Controller/postprocess | Collapse repeats, remove pad/blank token, map delimiter to spaces. Not part of the neural graph. |
| Sequence classification | Optional GPU graph target | Uses encoder hidden states, optional learned layer weighting, projection, masked mean pooling, classifier. |

Independent validation slices: processor tensor parity, Conv1d output-length parity, feature projection parity, one encoder block parity, full encoder parity, CTC logits, then decode parity.

## 3. Important config dimensions

Source defaults from `HubertConfig`:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 32 | CTC head output classes; must be set for `HubertForCTC`. |
| `hidden_size` | 768 | Encoder width and Q/K/V/out projection width. |
| `num_hidden_layers` | 12 | Transformer block count. |
| `num_attention_heads` | 12 | MHA heads; `head_dim = hidden_size / num_attention_heads`. |
| `intermediate_size` | 3072 | FFN expansion width. |
| `hidden_act` | `gelu` | FFN activation. |
| `hidden_dropout` / `activation_dropout` / `attention_dropout` | 0.1 / 0.1 / 0.1 | Inactive in eval; training only for first integration. |
| `feat_proj_layer_norm` | `True` | Adds LayerNorm before feature projection. |
| `feat_proj_dropout` / `final_dropout` | 0.0 / 0.1 | Projection/head dropout; erased in eval. |
| `feat_extract_norm` | `group` | `group`: GroupNorm only on first conv. `layer`: LayerNorm after every conv with NCL/NLC transposes. |
| `feat_extract_activation` | `gelu` | Conv frontend and positional conv activation. |
| `conv_dim` | `[512]*7` | Conv1d channel ladder; first input channel is 1. |
| `conv_stride` | `[5,2,2,2,2,2,2]` | Length reduction. Product is 320, but exact length is repeated floor Conv1d length. |
| `conv_kernel` | `[10,3,3,3,3,2,2]` | Conv1d kernels, no padding in feature extractor. |
| `conv_bias` | `False` | Some configs can override; inspected public HuBERT configs keep `False`. |
| `num_conv_pos_embeddings` | 128 | Positional Conv1d kernel size. |
| `num_conv_pos_embedding_groups` | 16 | Positional grouped Conv1d groups. |
| `conv_pos_batch_norm` | `False` | If false, positional conv uses weight norm; if true, BatchNorm1d before conv. |
| `do_stable_layer_norm` | `False` | Switches encoder block from post-norm to stable/pre-norm. |
| `apply_spec_augment` and mask fields | `True`, time prob 0.05 | Training/masked-pretraining path; not required for eval unless caller explicitly passes `mask_time_indices`. |
| `use_weighted_layer_sum` | `False` | Sequence classification can stack all hidden states and learn a weighted sum. |
| `classifier_proj_size` | 256 | Sequence classification projector width. |

Representative sweep:

| Model id | Arch | Hidden | Layers | Heads | Head dim | FFN | Conv stack | Conv norm | Stable LN | Head |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `hf-tiny-model-private/tiny-random-HubertForCTC` | CTC | 16 | 4 | 2 | 8 | 20 | `32,32,32`, k/s `8/4` x3 | group | false | vocab 32 |
| `facebook/hubert-base-ls960` | base encoder | 768 | 12 | 12 | 64 | 3072 | seven 512-channel convs | group | false | no task head in arch |
| `facebook/hubert-large-ls960-ft` | CTC | 1024 | 24 | 16 | 64 | 4096 | seven 512-channel convs | layer | true | vocab 32 |
| `facebook/hubert-large-ll60k` | base encoder | 1024 | 24 | 16 | 64 | 4096 | seven 512-channel convs | layer | true | no task head in arch |
| `facebook/hubert-xlarge-ls960-ft` | CTC | 1280 | 48 | 16 | 80 | 5120 | seven 512-channel convs | layer | true | vocab 32 |
| `superb/hubert-base-superb-ks` | sequence classification | 768 | 12 | 12 | 64 | 3072 | seven 512-channel convs | group | false | weighted sum + classifier |
| `superb/hubert-large-superb-er` | sequence classification | 1024 | 24 | 16 | 64 | 4096 | seven 512-channel convs | layer | true | weighted sum + classifier |

Preprocessor sweep:

| Model id | Feature extractor | Sampling rate | Normalize | Return attention mask |
|---|---|---:|---|---|
| tiny random CTC | `Wav2Vec2FeatureExtractor` | 16000 | true | false |
| `facebook/hubert-base-ls960` | `Wav2Vec2FeatureExtractor` | 16000 | true | false |
| `facebook/hubert-large-ls960-ft` | `Wav2Vec2FeatureExtractor` | 16000 | true | true |
| `facebook/hubert-xlarge-ls960-ft` | `Wav2Vec2FeatureExtractor` | 16000 | true | true |
| `superb/hubert-base-superb-ks` | `Wav2Vec2FeatureExtractor` | 16000 | false | true |

Omitted fields in many checkpoint configs inherit source defaults. Notable defaults: `feat_proj_layer_norm=True`, `conv_pos_batch_norm=False`, `apply_spec_augment=True`, mask lengths/probs as above, `classifier_proj_size=256`, `ctc_loss_reduction="sum"`, `ctc_zero_infinity=False`, and `pad/bos/eos_token_id = 0/1/2`.

## 3a. Family variation traps

- `feat_extract_norm` is structurally significant. `group` applies GroupNorm only to conv layer 0; `layer` applies LayerNorm after every feature Conv1d and requires `NCL -> NLC -> NCL` layout flips.
- Feature-extractor attention-mask policy is checkpoint-coupled. Base group-norm checkpoints often have `return_attention_mask=false`; large layer-norm checkpoints return masks for batched inference.
- Frame count is exact repeated floor Conv1d math, not simply `T / 320`. For default convs: `L = floor((L-k)/s)+1` per layer.
- `do_stable_layer_norm` changes encoder semantics: base post-norm vs large/xlarge stable pre-norm with a final encoder LayerNorm.
- Positional Conv1d is source-NLC externally but internally transposes to NCL, applies optional BatchNorm1d or weight-normalized grouped Conv1d, crops one frame for even kernels, applies activation, then transposes back.
- Attention is full bidirectional MHA. There is no causal mask, RoPE, ALiBi, relative bias, GQA/MQA, sliding window, or autoregressive KV cache for the primary target.
- `mask_time_indices` can be passed even in eval and performs boolean indexed replacement with `masked_spec_embed`. First inference integration should reject or route this path unless masked feature inference is needed.
- `apply_spec_augment` and generated random masks are training-only when `self.training`; they should not enter normal eval graphs.
- Sequence classification with `use_weighted_layer_sum=True` requires `output_hidden_states=True`, stacks `num_hidden_layers + 1` tensors, softmaxes learned weights, and reduces across layer axis.
- `conv_pos_batch_norm=True` is implemented by source but not seen in inspected representative configs. It changes positional conv normalization from weight norm to BatchNorm1d.
- Adapters: `HubertForCTC` has adapter-loading logic keyed by `adapter_attn_dim`, and stable encoder layers can include attention bottleneck adapters. The base `HubertModel` does not keep Wav2Vec2's post-encoder adapter stack; `modular_hubert.py` deletes `self.adapter`.
- Layout-axis traps are central: raw `[B,T]` -> feature Conv1d `[B,C,T]` -> encoder `[B,T,C]` -> attention `[B,H,T,D]`. Any NCL/NLC optimization needs guards around LayerNorm axes, GroupNorm/channel axes, `dim=1` pooling, softmax `dim=-1`, and all transposes.

## 4. Operator coverage checklist

Required runtime operators for encoder + CTC inference:

Tensor/layout ops:

- `unsqueeze(input_values, dim=1)` from `[B,T_raw]` to `[B,1,T_raw]`.
- `transpose(1,2)` between `[B,C,T]` and `[B,T,C]`.
- `view/reshape` Q/K/V `[B,T,Hid] -> [B,T,num_heads,head_dim]`.
- `transpose(1,2)` attention `[B,T,H,D] -> [B,H,T,D]`.
- `contiguous/reshape` attention output `[B,H,T,D] -> [B,T,Hid]`.
- Mask conversion: output-length computation, zeros allocation, advanced index set, `flip`, `cumsum`, `bool`, `unsqueeze`, `repeat`.
- Optional `stack`, broadcast multiply, softmax over layer weights, and sum over layer axis for sequence classification.

Neural network primitives:

- Conv1d feature extractor:
  - default base/large: `1 -> 512, k=10, s=5`, then `512 -> 512` with `(k,s)=(3,2),(3,2),(3,2),(3,2),(2,2),(2,2)`.
  - tiny random: `1 -> 32`, `32 -> 32`, `32 -> 32`, each `k=8, s=4`.
- GroupNorm with `num_groups=channels`, `num_channels=channels`, affine, only first conv for `feat_extract_norm="group"`.
- LayerNorm over last/channel dimension for conv-layer norm, feature projection, encoder blocks, and classification adapter/projection paths.
- Linear feature projection: base `512 -> 768`; large `512 -> 1024`; xlarge `512 -> 1280`.
- GELU for conv frontend, positional conv, and FFN unless config changes `hidden_act`/`feat_extract_activation`.
- Encoder FFN linears: base `768 -> 3072 -> 768`; large `1024 -> 4096 -> 1024`; xlarge `1280 -> 5120 -> 1280`.
- Positional Conv1d: grouped `hidden_size -> hidden_size`, `kernel=128`, `padding=64`, `groups=16`, weight norm by default.
- CTC head: `hidden_size -> vocab_size`, usually `768/1024/1280 -> 32`.
- Dropout is erased in eval.

Attention primitives:

- Noncausal self-attention MHA.
- Q/K/V/out linears, all `hidden_size -> hidden_size`, with bias.
- Score scale `head_dim ** -0.5`.
- Additive bidirectional padding mask for eager attention or compatible mask object for SDPA/Flash/Flex paths.
- Softmax over key length, dropout in training, `matmul(attn, value)`.

Preprocessing-coupled ops:

- Mono waveform validation and `float32` conversion.
- Padding/truncation with `padding_value=0.0`.
- Optional per-example normalization over valid samples:

```python
normed = (x[:length] - mean(x[:length])) / sqrt(var(x[:length]) + 1e-7)
normed[length:] = padding_value
```

- Attention mask reduction from raw samples to feature frames.

Optional/deferred:

- CTC training loss: `log_softmax(logits, dim=-1, dtype=float32).transpose(0,1)`, `ctc_loss`, label `masked_select`.
- Sequence classification loss.
- Adapter loading and attention adapter bottleneck.
- Training SpecAugment random mask generation.

## 5. Layer/block breakdown

Feature encoder:

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
if feat_proj_layer_norm:
  extract_features = LayerNorm(conv_dim[-1])(extract_features)
hidden = Linear(conv_dim[-1] -> hidden_size)(extract_features)
hidden = Dropout(hidden)       # eval no-op
```

Positional convolution:

```text
pos = transpose(hidden, [B,T,H] -> [B,H,T])
if conv_pos_batch_norm:
  pos = BatchNorm1d(H)(pos)
pos = GroupedConv1d(H -> H, kernel=num_conv_pos_embeddings, padding=kernel//2, groups=num_conv_pos_embedding_groups)(pos)
if kernel is even:
  pos = pos[:, :, :-1]
pos = GELU(pos)
pos = transpose(pos, [B,H,T] -> [B,T,H])
```

Post-norm encoder layer, used by base/group-norm defaults:

```text
residual = hidden
attn = MHA(hidden, bidirectional_padding_mask)
hidden = residual + Dropout(attn)
hidden = LayerNorm(hidden)
hidden = hidden + FFN(hidden)
hidden = LayerNorm(hidden)
```

Stable/pre-norm encoder layer, used by large/xlarge defaults:

```text
residual = hidden
hidden = LayerNorm(hidden)
attn = MHA(hidden, bidirectional_padding_mask)
hidden = residual + Dropout(attn)
hidden = hidden + FFN(LayerNorm(hidden))
if adapter_attn_dim is not None:
  hidden = hidden + AdapterLayer(hidden)
```

Stable encoder applies final `LayerNorm(hidden)` after all layers. Non-stable encoder applies `LayerNorm(hidden + pos)` before the layer loop.

CTC head:

```text
hidden = HubertModel(input_values, attention_mask).last_hidden_state  # [B,T',H]
hidden = Dropout(hidden)                                              # eval no-op
logits = Linear(H -> vocab_size)(hidden)                              # [B,T',V]
```

Sequence classification head:

```text
outputs = HubertModel(..., output_hidden_states=use_weighted_layer_sum)
if use_weighted_layer_sum:
  hs = stack(all_hidden_states, dim=1)                 # [B,L+1,T,H]
  w = softmax(layer_weights, dim=-1)                   # [L+1]
  hidden = (hs * w.view(1,-1,1,1)).sum(dim=1)          # [B,T,H]
else:
  hidden = last_hidden_state
hidden = Linear(H -> classifier_proj_size)(hidden)
pooled = masked_mean(hidden, dim=1) or mean(dim=1)
logits = Linear(classifier_proj_size -> num_labels)(pooled)
```

## 6. Attention requirements

HuBERT uses encoder-only self-attention:

| Property | Requirement |
|---|---|
| Causality | Noncausal / bidirectional |
| Attention type | Self-attention; source supports cross-attention args in `HubertAttention`, but HuBERT encoder calls self-attention only |
| Head structure | Standard MHA, no GQA/MQA |
| Head counts | base `12 x 64`; large `16 x 64`; xlarge `16 x 80`; tiny `2 x 8` |
| QKV layout | Separate `q_proj`, `k_proj`, `v_proj` linears with bias, each `[H,H]` in PyTorch linear storage `[out,in]` |
| Masking | Bidirectional padding mask after raw attention-mask reduction to feature-frame length |
| Packed/varlen | No source-specific packed sequence metadata |
| Sliding/local | None |
| Position interaction | Positional grouped Conv1d is added before Transformer layers; no per-head position math inside attention |
| KV cache | None for primary target |
| Backend compatibility | Source declares FlashAttention, SDPA, and Flex support; eager fallback is matmul + additive mask + softmax + matmul |

Eager attention math order:

```text
q = Linear(hidden).view(B,T,H,D).transpose(1,2)
k = Linear(hidden).view(B,T,H,D).transpose(1,2)
v = Linear(hidden).view(B,T,H,D).transpose(1,2)
scores = matmul(q, k.transpose(-2,-1)) * (D ** -0.5)
scores = scores + attention_mask       # if present
probs = softmax(scores, dim=-1)
out = matmul(probs, v).transpose(1,2).contiguous().reshape(B,T,Hid)
out = out_proj(out)
```

The cache section is intentionally not applicable: HuBERT has no autoregressive generation loop and no KV cache. Encoder outputs can be cached by an application only as ordinary audio features; that is not a Transformers cache contract.

## 7. Position encoding and custom math

HuBERT uses convolutional positional embeddings, not RoPE/ALiBi/relative bias.

```python
def hubert_positional_conv(hidden, conv, batch_norm=None, kernel=128, activation=gelu):
    # hidden: [batch, frames, hidden_size]
    x = hidden.transpose(1, 2)          # [B, H, T]
    if batch_norm is not None:
        x = batch_norm(x)
    x = conv(x)                         # grouped Conv1d, padding=kernel//2
    if kernel % 2 == 0:
        x = x[:, :, :-1]                # same-pad crop
    x = activation(x)
    return x.transpose(1, 2)            # [B, T, H]
```

Default positional conv uses weight normalization on the Conv1d weight. DinoML can either materialize the effective inference weight before lowering or represent weight norm as a constant transform:

```python
effective_weight = weight_v * (weight_g / norm(weight_v, dim=2, keepdim=True))
```

The effective weight is constant for inference unless runtime weight mutation is supported. The positional output length depends only on `num_conv_pos_embeddings` parity and should match input frame length after the even-kernel crop.

Feature-frame length math:

```python
def hubert_conv_out_len(length, kernels, strides):
    for k, s in zip(kernels, strides):
        length = floor((length - k) / s) + 1
    return length
```

Mask reduction uses this length per batch item, sets the last valid frame to 1, then `flip -> cumsum -> flip -> bool` to mark all frames before it as valid.

## 8. Preprocessing and input packing

HuBERT consumes raw mono waveform tensors, not log-mel features.

CPU/data-pipeline contract from `Wav2Vec2FeatureExtractor`:

- Input type: mono audio sequence as list/NumPy/torch-compatible values; batched arrays must be rank <= 2.
- Default sampling rate: 16,000 Hz. Passing a different `sampling_rate` raises.
- Feature size: 1.
- Padding value: `0.0`.
- Optional padding/truncation/max length/pad-to-multiple handled in processor.
- Values are cast to `float32`.
- If `do_normalize=True`, normalize each sample sequence to zero mean and unit variance. With padding and an attention mask, statistics use only valid samples and padded positions are reset to `padding_value`.
- Output tensors: `input_values` `[B,T_raw]` float32 and optionally `attention_mask` `[B,T_raw]` int/bool-like.

GPU/runtime graph contract:

- Start from `input_values` and optional `attention_mask`.
- If attention mask is present, reduce it to feature-frame mask after Conv1d length calculation.
- If attention mask is absent, encoder attention can be unmasked and source may skip bidirectional mask creation for optimized attention backends.

Tokenizer/decode contract for CTC:

- `Wav2Vec2CTCTokenizer` is used for HuBERT ASR.
- `pad_token_id` is the CTC blank, usually 0.
- Decode filters blank/pad, collapses repeated tokens by default, and maps word delimiter token `|` to a space.
- This decode is controller/postprocess, not a neural graph op.

No prompt packing, modality token stitch, cu-seqlens metadata, STFT, mel bins, or decoder input IDs are part of the core HuBERT inference graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: eval dropout erase

Source pattern:

```text
Dropout(x, p=..., training=False)
```

Replacement:

```text
identity(x)
```

Preconditions:

- Module is in eval/inference mode.
- No training parity target in the compiled artifact.

Failure cases:

- Training, SpecAugment, LayerDrop, or dropout-probability parity tests.

Parity test sketch:

- Compare full encoder and CTC logits in `model.eval()` with dropout nodes erased.

### Rewrite: weight-normalized positional Conv1d constant fold

Source pattern:

```text
weight_norm(Conv1d(..., groups=G, padding=K//2, bias=True/False), name="weight", dim=2)
```

Replacement:

```text
Conv1d(effective_weight, bias, groups=G, padding=K//2) -> same-pad crop -> activation
```

Preconditions:

- Inference-only weights are immutable.
- Both `weight_g` and `weight_v` are available.
- Weight norm dimension is exactly source `dim=2`.

Shape equations:

- Input `[B,H,T]`, output before crop `[B,H,T+1]` for even default `K=128`, after crop `[B,H,T]`.

Failure cases:

- Runtime weight mutation, training, missing parametrization metadata, `conv_pos_batch_norm=True` needing a separate BN fold decision.

Parity test sketch:

- Run positional conv alone on random `[B,T,H]` for base and large configs, compare folded Conv1d result to Transformers.

### Rewrite: feature Conv1d frontend as Conv1d provider path

Source pattern:

```text
[B,T] -> unsqueeze -> series of Conv1d/GELU/norm layers
```

Replacement:

```text
Use explicit Conv1d kernels or a 1D-im2col + GEMM fallback per layer
```

Preconditions:

- Preserve source NCL layout through Conv1d region.
- Support no-padding feature convs with strides from config.
- Support first-layer GroupNorm for `feat_extract_norm="group"` and per-layer NLC LayerNorm for `feat_extract_norm="layer"`.

Failure cases:

- Assuming channel-last globally; LayerNorm and GroupNorm axes differ.
- Approximating output length as `ceil(T / product(strides))`.

Parity test sketch:

- For raw lengths near kernel boundaries and padded batch items, compare each conv layer output and final reduced attention mask.

### Rewrite: local NLC encoder island

Source pattern:

```text
feature extractor NCL -> transpose -> all encoder operations in NLC except attention head transposes
```

Replacement:

```text
Keep encoder activations in NLC; fuse Linear/GELU/LayerNorm/residual paths around that layout
```

Preconditions:

- Feature Conv1d NCL region ends before feature projection.
- Positional conv internally handles NLC->NCL->NLC.
- Attention implementation expects `[B,T,H]` at block boundaries.

Failure cases:

- Moving GroupNorm or Conv1d into NLC without rewriting axes.
- Rewriting classification pooling `mean(dim=1)` incorrectly.

Parity test sketch:

- One-layer and full-encoder parity for base and large configs with and without attention masks.

### Rewrite: CTC last projection as GEMM

Source pattern:

```text
Linear(hidden_size -> vocab_size) over [B,T,H]
```

Replacement:

```text
flatten [B*T,H] -> GEMM_RCR(weight [V,H]) + bias -> reshape [B,T,V]
```

Preconditions:

- Dense contiguous hidden states or explicit stride support.
- Vocab head weight uses PyTorch Linear layout `[out_features, in_features]`.

Failure cases:

- Accidentally tying/non-tying weights: HuBERT has no input embedding/LM-head tie.

Parity test sketch:

- Compare logits for fixed encoder hidden states across base/large/xlarge dimensions.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d feature extractor kernels: the raw waveform frontend is mandatory and dominates early runtime for long audio; it also drives exact frame length.
- LayerNorm + Linear feature projection: appears for every checkpoint and is the bridge from 512 conv channels to hidden size.
- Encoder MHA with additive padding mask: full bidirectional attention is the main quadratic bottleneck for long audio.
- FFN Linear + GELU + Linear: large/xlarge models have 24/48 repetitions with wide FFNs.
- Positional grouped Conv1d with same-pad crop: small but required for parity; weight-norm folding avoids runtime parametrization overhead.

Medium priority:

- Residual + LayerNorm fusion for both post-norm and stable/pre-norm block variants.
- Conv frontend GroupNorm/LayerNorm + GELU fusion where layout is controlled.
- CTC head GEMM for `[B*T,H] -> [B*T,V]`.
- Attention-mask reduction kernel for batched variable-length audio.

Lower priority:

- Sequence-classification weighted layer sum and masked mean pooling.
- Adapter bottleneck layers.
- CTC decode acceleration; controller-level postprocess can start on CPU.
- Training-only SpecAugment/masking and CTC loss.

## 11. Runtime staging plan

Stage 1: config and processor contract.

- Parse `HubertConfig`.
- Load `Wav2Vec2FeatureExtractor` metadata.
- Produce `input_values` and optional `attention_mask` test fixtures from HF processor.

Stage 2: Conv1d feature encoder.

- Implement/route Conv1d, GroupNorm, LayerNorm-with-transposes, GELU.
- Validate feature-frame lengths and reduced attention masks.

Stage 3: base encoder block parity.

- Implement feature projection, positional Conv1d, MHA, FFN, residual/LN ordering.
- Test one layer and all layers for `facebook/hubert-base-ls960`.

Stage 4: stable-LN large encoder parity.

- Add stable/pre-norm encoder ordering.
- Validate `facebook/hubert-large-ls960-ft` and `facebook/hubert-xlarge-ls960-ft` shapes.

Stage 5: CTC head.

- Add final projection and logit parity.
- Keep tokenizer decode as a CPU/controller postprocess.

Stage 6: optional sequence classification.

- Add weighted hidden-state sum, projector, masked mean pooling, classifier.
- Validate SUPERB checkpoints.

Stage 7: optimized lowering/fusions.

- Fold positional weight norm.
- Add Conv1d provider path or im2col+GEMM fallback.
- Enable SDPA/Flash-style bidirectional attention when mask/layout constraints match.

Initially stub or reject: training losses, random SpecAugment, adapter loading by `target_lang`, `conv_pos_batch_norm=True` if no checkpoint requires it.

## 12. Parity and validation plan

Processor parity:

- Compare HF processor outputs for single and batched raw audio with padding/truncation.
- Include `do_normalize=True/False`, `return_attention_mask=True/False`, and invalid sampling-rate error.

Shape and mask parity:

- For raw lengths around conv boundaries, compare `_get_feat_extract_output_lengths`.
- For padded batches, compare reduced feature attention masks exactly.

Operator parity:

- Conv1d frontend after each layer for tiny, base, and large configs.
- Positional Conv1d with weight norm folded vs unfurled source.
- Eager MHA for random `[B,T,H]` with and without padding mask.
- FFN and LayerNorm/residual ordering for both post-norm and stable/pre-norm blocks.

Model parity:

- Tiny CTC end-to-end logits on random waveform.
- Base encoder `last_hidden_state` on random waveform and short real fixture.
- Large CTC logits for `facebook/hubert-large-ls960-ft`.
- Xlarge shape-smoke and a reduced numeric parity slice if memory permits.
- SUPERB sequence-classification logits for weighted-layer-sum path.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` for most blocks; attention/long Conv1d may need `atol=3e-4`.
- fp16/bf16: start with `rtol=5e-2`, `atol=5e-2` for full model; tighten per fused op after accumulation policy is fixed.
- Mask and length outputs must match exactly.

End-to-end ASR parity:

- Compare logits first, then greedy CTC decoded text for a short 16 kHz waveform.
- Decode parity should verify blank removal, repeat collapse, and `|` word delimiter handling.

## 13. Performance probes

- Processor throughput: raw audio padding/normalization samples/sec on CPU.
- Conv frontend throughput: batch-size sweep and raw duration sweep.
- Feature-frame length sweep: 1s, 5s, 10s, 30s audio at 16 kHz.
- Encoder-only throughput for base, large, xlarge at fixed `T'`.
- Attention backend comparison: eager matmul vs SDPA/Flash-compatible bidirectional path with no mask and with padding mask.
- FFN GEMM throughput by hidden/intermediate size.
- End-to-end CTC logits/sec and audio-hours/hour.
- Batch padding efficiency: grouped similar lengths vs mixed lengths.
- Memory probes: activation footprint for full hidden-state output enabled vs disabled; xlarge 48-layer peak memory.
- Optional classification throughput: weighted layer sum overhead when `output_hidden_states=True`.

Any measured numbers should be recorded separately with hardware, dtype, and exact commit; this report contains source-derived probes only.

## 14. Skip/defer list

Safe to defer for first ASR inference:

- Training losses: CTC loss, sequence-classification loss, gradient checkpointing.
- Training LayerDrop/dropout behavior.
- Random SpecAugment mask generation and NumPy `_compute_mask_indices`.
- `mask_time_indices` eval-time masked feature replacement, unless a masked-feature inference use case appears.
- Adapter loading by `target_lang` and attention adapter bottlenecks.
- `conv_pos_batch_norm=True` until a target checkpoint requires it.
- Beam-search CTC decode and language-model-assisted decode.
- Multi-GPU/FSDP/DeepSpeed-specific branches.
- Output attentions/hidden states except when sequence classification uses weighted layer sum.
- Quantization and GGUF ingestion until dense parity is stable.

Not safe to defer:

- Raw waveform processor contract.
- Exact Conv1d length math and reduced attention masks.
- NCL/NLC transposes and axis guards.
- Positional Conv1d same-pad crop.
- Stable vs non-stable LayerNorm block ordering.

## 15. Final implementation checklist

- [ ] Parse `HubertConfig` including conv, norm, stable-LN, and head fields.
- [ ] Load `Wav2Vec2FeatureExtractor` metadata for HuBERT processors.
- [ ] Load dense weights, including weight-normalized positional conv parameters.
- [ ] Implement raw waveform input contract: `[B,T]` float32 plus optional raw attention mask.
- [ ] Implement Conv1d feature extractor with exact length math.
- [ ] Implement `feat_extract_norm="group"` first-layer GroupNorm path.
- [ ] Implement `feat_extract_norm="layer"` per-conv LayerNorm transpose path.
- [ ] Implement feature projection LayerNorm + Linear.
- [ ] Implement reduced feature attention mask helper.
- [ ] Implement positional grouped Conv1d with weight-norm folding and even-kernel crop.
- [ ] Implement bidirectional MHA with additive padding mask.
- [ ] Implement post-norm HuBERT encoder layer.
- [ ] Implement stable/pre-norm HuBERT encoder layer.
- [ ] Implement CTC head Linear and logits output.
- [ ] Add processor parity tests for normalization, padding, masks, and sampling-rate rejection.
- [ ] Add conv length and reduced-mask parity tests.
- [ ] Add one-block and full-encoder parity tests for base and large configs.
- [ ] Add CTC logit parity for tiny and `facebook/hubert-large-ls960-ft`.
- [ ] Add optional sequence-classification weighted-layer-sum parity.
- [ ] Benchmark processor, conv frontend, encoder, attention backend, and end-to-end CTC logits.
