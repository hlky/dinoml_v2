# WavLM Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  wavlm family. Primary DinoML target recommended here: WavLMModel encoder inference from raw waveform tensors.
  Heads documented: CTC, sequence classification, audio frame classification, XVector speaker embeddings.

Config source:
  Local WavLMConfig defaults plus Hugging Face config/preprocessor/tokenizer JSON snapshots under
  H:/dinoml_v2/agents/plans/transformers/wavlm/_sources.

Source files inspected:
  transformers/src/transformers/models/wavlm/configuration_wavlm.py
  transformers/src/transformers/models/wavlm/modular_wavlm.py
  transformers/src/transformers/models/wavlm/modeling_wavlm.py
  transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
  transformers/tests/models/wavlm/test_modeling_wavlm.py

Any missing files or assumptions:
  modeling_wavlm.py is generated from modular_wavlm.py. For future Transformers edits, modular_wavlm.py is authoritative;
  for DinoML runtime behavior, modeling_wavlm.py was inspected because it contains the expanded Wav2Vec2-derived classes.
  Native WavLM uses Wav2Vec2FeatureExtractor; there is no wavlm-local processor or tokenizer file. Tokenizer files matter
  only for CTC checkpoints. No trust_remote_code source was required.
```

Representative primary configs inspected:

- [hf-internal-testing/tiny-random-wavlm](https://huggingface.co/hf-internal-testing/tiny-random-wavlm): `config.json`, `preprocessor_config.json`.
- [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base): `config.json`, `preprocessor_config.json`.
- [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus): `config.json`, `preprocessor_config.json`.
- [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large): `config.json`, `preprocessor_config.json`.
- [patrickvonplaten/wavlm-libri-clean-100h-base-plus](https://huggingface.co/patrickvonplaten/wavlm-libri-clean-100h-base-plus): `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`.
- [microsoft/wavlm-base-plus-sd](https://huggingface.co/microsoft/wavlm-base-plus-sd): `config.json`, `preprocessor_config.json`.
- [microsoft/wavlm-base-plus-sv](https://huggingface.co/microsoft/wavlm-base-plus-sv): `config.json`, `preprocessor_config.json`.

## 2. High-level architecture

WavLM is a wav2vec-style audio encoder with a raw-waveform convolutional feature extractor, feature projection, convolutional positional embedding, noncausal Transformer encoder, and WavLM-specific gated relative-position attention bias.

```text
CPU/data pipeline waveform load + pad/normalize
  -> input_values [B, samples], optional attention_mask [B, samples]
  -> Conv1d feature encoder [B, 512, T_feat]
  -> transpose to [B, T_feat, 512]
  -> LayerNorm + Linear(512 -> H)
  -> optional SpecAugment in training / mask_time_indices path
  -> conv positional embedding + encoder stack
  -> base hidden states or task head
```

Stage decomposition:

- CPU/data pipeline: mono waveform validation, sampling-rate check, padding/truncation, optional zero-mean/unit-variance normalization, attention mask creation.
- GPU/runtime core: 1D convolutional front end, feature projection, WavLM encoder, optional adapter.
- Independently validatable pieces: feature extractor length contract, one Conv1d layer, feature projection, gated relative-position bias, one encoder layer, then full encoder.
- Task heads: CTC logits, sequence-level pooling classifier, frame classifier, and XVector TDNN/statistic-pooling head. These can be staged after base encoder parity.
- No autoregressive prefill/decode, causal LM sampling, or KV cache exists for the primary target.

## 3. Important config dimensions

Source defaults from `WavLMConfig`:

| Field | Default |
|---|---:|
| `vocab_size` | 32 |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | 64, inferred as `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `feat_extract_norm` | `group` |
| `conv_dim` | `[512, 512, 512, 512, 512, 512, 512]` |
| `conv_stride` | `[5, 2, 2, 2, 2, 2, 2]`, product 320 |
| `conv_kernel` | `[10, 3, 3, 3, 3, 2, 2]` |
| `conv_bias` | false |
| `num_conv_pos_embeddings` | 128 |
| `num_conv_pos_embedding_groups` | 16 |
| `num_buckets` / `max_bucket_distance` | 320 / 800 |
| `do_stable_layer_norm` | false |
| `add_adapter` | false |
| `tdnn_dim/kernel/dilation` | `[512,512,512,512,1500]` / `[5,3,3,1,1]` / `[1,2,3,1,1]` |
| `xvector_output_dim` | 512 |

Representative checkpoint sweep:

| Checkpoint | Architecture | H | Layers | Heads | FFN | Front end | Stable LN | Processor normalize/mask | Head-significant fields |
|---|---|---:|---:|---:|---:|---|---|---|---|
| `hf-internal-testing/tiny-random-wavlm` | SequenceClassification | 16 | 4 | 2 | 20 | conv dim `[32,32,32]`, stride `[4,4,4]`, kernel `[8,8,8]` | true | normalize true, no attention mask default | debug-sized layer-norm front end |
| `microsoft/wavlm-base` | WavLMModel | 768 | 12 | 12 | 3072 | standard 7-layer Conv1d | false | normalize false, attention mask true | base encoder |
| `microsoft/wavlm-base-plus` | WavLMModel | 768 | 12 | 12 | 3072 | standard 7-layer Conv1d | false | normalize false, attention mask true | common production encoder |
| `microsoft/wavlm-large` | WavLMModel | 1024 | 24 | 16 | 4096 | standard 7-layer Conv1d, `feat_extract_norm="layer"` | true | normalize true, attention mask true | large stable-LN encoder |
| `patrickvonplaten/wavlm-libri-clean-100h-base-plus` | WavLMForCTC | 768 | 12 | 12 | 3072 | standard 7-layer Conv1d | false | normalize false, attention mask true | `vocab_size=31`, `pad_token_id=28` |
| `microsoft/wavlm-base-plus-sd` | AudioFrameClassification | 768 | 12 | 12 | 3072 | standard 7-layer Conv1d | false | normalize false, attention mask true | `use_weighted_layer_sum=true` |
| `microsoft/wavlm-base-plus-sv` | XVector | 768 | 12 | 12 | 3072 | standard 7-layer Conv1d | false | normalize false, attention mask true | `use_weighted_layer_sum=true`, 1000 labels in config |

Effective defaults to preserve when omitted from configs:

- `output_hidden_size` defaults to `hidden_size`.
- `max_bucket_distance` defaults to 800 when older configs omit it.
- `tdnn_*` and `xvector_output_dim` source defaults apply if an XVector head is instantiated without explicit checkpoint values.
- `mask_feature_min_masks` appears in checkpoint configs but is not declared in the current strict `WavLMConfig` field list; the modeling code reads it only if `mask_feature_prob > 0` and training. It is not required for eval inference.

## 3a. Family variation traps

- WavLM attention is not plain wav2vec2 attention. It adds bucketed relative-position bias and a hidden-state-dependent gate before calling PyTorch multi-head attention.
- Only layer 0 owns `rel_attn_embed`; later layers receive and reuse the same `position_bias` tensor. Each layer still has its own `gru_rel_pos_linear` and `gru_rel_pos_const` gate.
- The source sets `_supports_flash_attn = False`, `_supports_sdpa = False`, and `_supports_flex_attn = False`. A DinoML fused attention path must reproduce the WavLM additive gated bias explicitly.
- Base/base-plus use `feat_extract_norm="group"`: first feature Conv1d has GroupNorm with `num_groups=out_channels`; later convs have no norm. Large/tiny use `feat_extract_norm="layer"`: every feature Conv1d has LayerNorm over channels after transposing to `[B,T,C]`.
- Base uses post-attention/post-FFN LayerNorm; large/tiny stable mode uses pre-attention LayerNorm and final encoder LayerNorm.
- Feature extractor emits source layout `[B,C,T]`, then model transposes to `[B,T,C]`. Conv1d regions are axis-sensitive; a channel-last layout pass must be guarded locally.
- Sequence length is reduced by repeated Conv1d floor formulas, not by simple `ceil(input/320)` for all lengths.
- Processor `do_normalize` differs: base/base-plus configs set false; large and tiny set true.
- Checkpoint configs include historical fields such as `mask_channel_*`, `replace_prob`, `feat_quantizer_dropout`, and `freeze_feat_extract_train`. The inspected native source does not use them for eval runtime.
- `add_adapter=True` changes both output hidden width and time length through stride-2 GLU Conv1d layers. Sequence/audio-frame classification constructors reject adapters.
- CTC checkpoints need tokenizer/vocab behavior; base encoder checkpoints often advertise `tokenizer_class` but have no tokenizer files and no text decoding head.
- XVector TDNN stores each temporal convolution as an `nn.Linear` weight and reshapes it into Conv1d weights at runtime. Lowering must preserve that storage transform.
- Attention mask values are boolean/reduced masks on feature time steps; source passes them as `key_padding_mask = attention_mask.ne(1)` to `multi_head_attention_forward`, not as the same additive mask used in many decoder models.

## 4. Operator coverage checklist

Tensor/layout ops:

- `unsqueeze`, `transpose`, `permute`, `view`/`reshape`, `repeat`, `chunk`, `cat`, `stack`, `mean`, `std`, `sum`, `cumsum`, `flip`, boolean masks, masked assignment, arange indexing.
- Feature length math: repeated `floor((L - kernel) / stride) + 1`.
- Position-bucket matrix creation: `arange`, broadcast subtraction, abs, comparison, `where`, min, log, long cast, embedding gather.
- XVector variable-length pooling loops for masked batches, or a vectorized equivalent with explicit length masks.

Neural network primitives:

- Conv1d feature extractor:
  - Standard: `1 -> 512`, kernel 10, stride 5; then six `512 -> 512` convs with kernels `[3,3,3,3,2,2]`, strides all 2.
  - Bias is usually false from config.
  - GroupNorm first layer for base configs; LayerNorm after every conv for large/stable configs.
- Feature projection: `LayerNorm(512) -> Linear(512 -> H, bias=True)`.
- Positional convolution: grouped Conv1d `H -> H`, kernel 128, padding 64, groups 16, weight-normalized parameterization, SamePad removal of one trailing step for even kernels, GELU.
- Encoder Linear layers with bias:
  - Base attention Q/K/V/O: `Linear(768 -> 768)`, FFN `768 -> 3072 -> 768`.
  - Large attention Q/K/V/O: `Linear(1024 -> 1024)`, FFN `1024 -> 4096 -> 1024`.
  - Gate linear per layer/head: `Linear(head_dim -> 8)`.
- LayerNorm, GroupNorm, GELU, ReLU, GLU for adapters, dropout as eval no-op.
- Heads:
  - CTC: `Linear(output_hidden_size -> vocab_size)`.
  - Sequence classification: optional weighted layer sum, `Linear(H -> classifier_proj_size)`, mean/masked mean, `Linear(classifier_proj_size -> num_labels)`.
  - Frame classification: optional weighted layer sum, `Linear(H -> num_labels)` per frame.
  - XVector: `Linear(H -> tdnn_dim[0])`, TDNN Conv1d stack, statistic pooling, `Linear(2*tdnn_dim[-1] -> xvector_output_dim)`, `Linear(xvector_output_dim -> xvector_output_dim)`.

Attention primitives:

- Noncausal self-attention only.
- MHA, not GQA/MQA: KV heads equal query heads.
- Source calls `torch.nn.functional.multi_head_attention_forward` with separate q/k/v projection weights and concatenated q/k/v bias.
- Additive `attn_mask` argument is the gated relative-position bias shaped `[B*heads, T, T]`.
- Optional key padding mask from reduced feature attention mask.
- No causal mask, KV cache, cross-attention, sliding window, RoPE, ALiBi, or packed varlen support.

Preprocessing-coupled ops:

- Wav2Vec2FeatureExtractor mono audio checks, 16 kHz sampling-rate validation, padding, optional truncation, optional zero-mean/unit-variance normalization over unpadded samples.
- CTC tokenizer only for decoding/transcripts; GPU graph consumes `input_values` and optional `attention_mask`.

Parameter sharing / aliases:

- Layer 0 relative-position embedding is logically shared as a computed `position_bias` value, not as a module parameter shared across layers. Do not clone a `rel_attn_embed` into later layers.
- There are no tied token embeddings for WavLM CTC; CTC head is an independent linear projection.
- Positional Conv1d uses weight norm. Runtime loading may either materialize the normalized conv weight or preserve `weight_g`/`weight_v` and compute the effective weight before convolution.

## 5. Layer/block breakdown

Feature extractor for base/base-plus:

```text
x = input_values[:, None]                              # [B,1,S]
x = Conv1d(1 -> 512, k=10, s=5, bias=False)(x)
x = GroupNorm(num_groups=512, num_channels=512)(x)
x = GELU(x)
repeat 6:
  x = Conv1d(512 -> 512, k=[3,3,3,3,2,2], s=2, bias=False)(x)
  x = GELU(x)
return x                                               # [B,512,T_feat]
```

Feature extractor for large/stable configs:

```text
for each Conv1d layer:
  x = Conv1d(Cin -> Cout, kernel, stride, bias=conv_bias)(x)  # [B,C,T]
  x = transpose(x, -2, -1)
  x = LayerNorm(Cout)(x)
  x = transpose(x, -2, -1)
  x = GELU(x)
```

Base model stem:

```text
extract = feature_extractor(input_values).transpose(1, 2)  # [B,T_feat,512]
feature_attention_mask = reduce_mask(attention_mask) if present
hidden, extract_norm = LayerNorm(512) -> Linear(512 -> H) -> Dropout(extract)
hidden = optional SpecAugment/mask_time_indices path
hidden = encoder(hidden, feature_attention_mask)
hidden = optional adapter(hidden)
```

Encoder prelude:

```text
if attention_mask:
  hidden[~attention_mask[..., None].repeat(...)] = 0
pos = grouped_weightnorm_Conv1d(H -> H, k=128, pad=64, groups=16)(hidden.transpose(1,2))
pos = pos[:, :, :-1] for even kernel, GELU, transpose back
hidden = hidden + pos
hidden = LayerNorm(hidden) then dropout        # non-stable encoder
hidden = dropout(hidden)                       # stable encoder; final LN after stack
```

WavLM encoder block, base/post-LN variant:

```text
res = x
attn, position_bias = WavLMAttention(x, attention_mask, position_bias)
x = res + Dropout(attn)
x = LayerNorm(x)
x = x + FeedForward(x)
x = LayerNorm(x)
```

WavLM encoder block, stable/pre-LN variant:

```text
res = x
y = LayerNorm(x)
attn, position_bias = WavLMAttention(y, attention_mask, position_bias)
x = res + Dropout(attn)
x = x + FeedForward(LayerNorm(x))
```

Feed-forward block:

```text
ff = Linear(H -> I, bias=True)(x)
ff = GELU(ff)
ff = Dropout(ff)
ff = Linear(I -> H, bias=True)(ff)
ff = Dropout(ff)
```

Heads:

```text
CTC: logits = Linear(H or output_hidden_size -> vocab_size)(Dropout(hidden))

Sequence classification:
  hidden = weighted_layer_sum(all_hidden_states) or last_hidden_state
  hidden = Linear(H -> classifier_proj_size)(hidden)
  pooled = mean(hidden, dim=1) or masked_sum / valid_count
  logits = Linear(classifier_proj_size -> num_labels)(pooled)

Frame classification:
  hidden = weighted_layer_sum(...) or last_hidden_state
  logits = Linear(H -> num_labels)(hidden)

XVector:
  hidden = weighted_layer_sum(...) or last_hidden_state
  hidden = Linear(H -> tdnn_dim[0])(hidden)
  for each TDNN layer: Conv1d-from-linear-storage + ReLU
  pooled = cat(mean(hidden, dim=time), std(hidden, dim=time))
  embeddings = Linear(2*tdnn_dim[-1] -> xvector_output_dim)(pooled)
  logits = Linear(xvector_output_dim -> xvector_output_dim)(embeddings)
```

## 6. Attention requirements

- Variant: encoder-only noncausal self-attention.
- Heads: base `12 x 64`; large `16 x 64`; tiny `2 x 8`.
- Q/K/V/O projections are independent `nn.Linear(embed_dim, embed_dim)` with bias.
- Key/value heads equal query heads; no GQA/MQA repeat logic.
- Masking:
  - Reduced feature attention mask has shape `[B,T_feat]`, bool-like true for valid feature steps.
  - Attention implementation converts it to `key_padding_mask = attention_mask.ne(1)`.
  - The WavLM relative-position term is passed through PyTorch's `attn_mask` slot as an additive bias of shape `[B*heads,T,T]`.
- Relative bias/gate:
  - On the first layer call, layer 0 computes `position_bias` from bucketed relative positions using `rel_attn_embed`.
  - Each layer computes a per-token/per-head gate from current hidden states and multiplies the shared position bias before attention.
- Backend compatibility:
  - Current source explicitly disables FlashAttention/SDPA/FlexAttention capability flags.
  - A DinoML fused attention implementation must accept an arbitrary additive bias `[B,H,T,T]` after hidden-state-dependent gate computation and a key padding mask.
- Cache:
  - No autoregressive KV cache. The only cache-like optimization is precomputing relative bucket IDs or base `position_bias` for a fixed feature length.

Source math order to preserve:

```text
q,k,v are projected inside multi_head_attention_forward
scores = q @ k.T * head_dim^-0.5
scores += gated_position_bias
scores += key_padding_mask effect
softmax/dropout
out = scores @ v -> out_proj
```

## 7. Position encoding and custom math

WavLM uses both convolutional positional embeddings and gated relative-position bias.

Convolutional positional embedding:

```python
def wavlm_pos_conv(x_bth):
    y = x_bth.transpose(1, 2)
    y = weight_norm_conv1d(y, kernel_size=128, padding=64, groups=16)
    y = y[:, :, :-1]  # because kernel size is even
    y = gelu(y)
    return y.transpose(1, 2)
```

Relative bucket mapping:

```python
def relative_positions_bucket(relative_positions, num_buckets=320, max_distance=800):
    half = num_buckets // 2
    buckets = (relative_positions > 0).long() * half
    rel = abs(relative_positions)
    max_exact = half // 2
    large = max_exact + (
        torch.log(rel.float() / max_exact) / math.log(max_distance / max_exact)
        * (half - max_exact)
    ).long()
    large = torch.minimum(large, torch.full_like(large, half - 1))
    return buckets + torch.where(rel < max_exact, rel, large)
```

Gated relative-position bias:

```python
def wavlm_gated_bias(hidden_bth, position_bias_bhtt, gate_linear, gate_const):
    B, T, H = hidden_bth.shape
    A = gate_const.shape[1]
    D = H // A
    h = hidden_bth.view(B, T, A, D).permute(0, 2, 1, 3)  # [B,A,T,D]
    proj = gate_linear(h).view(B, A, T, 2, 4).sum(-1)    # [B,A,T,2]
    gate_a, gate_b = torch.sigmoid(proj).chunk(2, dim=-1)
    gate = gate_a * (gate_b * gate_const - 1.0) + 2.0   # [B,A,T,1]
    return gate * position_bias_bhtt                    # broadcasts over key length
```

What can be precomputed:

- Relative bucket matrix `[T,T]` for fixed feature length.
- Base position bias from layer-0 `rel_attn_embed`, if weights and `T` are fixed.
- Convolutional positional embedding cannot be precomputed because it depends on hidden states.
- Gated relative bias cannot be fully precomputed because it depends on each layer's hidden states and gate weights.

## 8. Preprocessing and input packing

Feature extractor contract from Wav2Vec2FeatureExtractor:

- Input is mono raw speech: one float per timestep. Batched numpy arrays with rank greater than 2 are rejected.
- Checkpoint sampling rate is 16,000 Hz; passing a mismatched `sampling_rate` raises.
- Padding value is `0.0`; padding side is right in all sampled configs.
- Output tensor is `input_values [B,S]` float32 and optional `attention_mask [B,S]` int32/long.
- Normalization is checkpoint-dependent:
  - base/base-plus/sd/sv/CTC sampled configs: `do_normalize=false`.
  - large and tiny configs: `do_normalize=true`.
- Normalization formula over valid samples is `(x - mean) / sqrt(var + 1e-7)`, with padded tail reset to padding value when an attention mask is available.
- No STFT, FFT, mel filterbank, or spectrogram feature extraction is part of WavLM preprocessing; the neural Conv1d front end consumes raw waveform samples.

Runtime input packing:

- GPU graph input should accept `input_values [B,S]` and optional `attention_mask [B,S]`.
- Reduced feature mask is computed after Conv1d length reduction:

```text
non_padded_lengths = attention_mask.cumsum(-1)[:, -1]
out_lengths = repeated floor((L - kernel) / stride) + 1
mask = zeros([B,T_feat]); mask[batch, out_lengths - 1] = 1
mask = flip(cumsum(flip(mask))).bool()
```

- For first inference integration, `mask_time_indices` and training-time SpecAugment can be rejected or ignored unless explicitly supplied. If supplied in eval, source applies masked-spec embedding by boolean indexed assignment.
- CTC postprocessing is outside the core graph: tokenizer decoding maps argmax/CTC collapsed ids to text using checkpoint vocab. The sampled CTC checkpoint has `vocab_size=31` and `pad_token_id=28`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature Conv1d front end to im2col/GEMM or cuDNN Conv1d

Preconditions:

- Source layout is `[B,C,T]` after `input_values[:, None]`.
- Conv1d has `dilation=1`, `groups=1`, and config kernels/strides.
- Bias presence follows `conv_bias`.
- Norm/activation after conv must remain in source order.

Replacement:

```text
Conv1d native provider
or
TemporalWindowFlatten -> GEMM(weight.reshape(out_channels, in_channels * kernel).T) -> BiasAdd -> Reshape
```

Shape equation:

```text
T_out = floor((T_in - kernel) / stride) + 1
```

Layout constraints:

- Do not globally translate `[B,C,T]` to `[B,T,C]` across feature convs unless every following norm and activation axis is rewritten.
- GroupNorm in base first layer normalizes channel groups over `[C,T]`; LayerNorm in large mode normalizes channels after transpose to `[B,T,C]`.

Failure cases:

- Non-default dilation/groups, unexpected padding, or adapter Conv1d with GLU should use separate guards.

Parity test sketch:

- Compare each conv layer output and length for several sample lengths, including non-multiples of 320.

### Rewrite: WavLM attention with gated relative bias to fused attention

Preconditions:

- Noncausal self-attention, `hidden_size % num_heads == 0`.
- Gated bias has been materialized as `[B,heads,T,T]` or `[B*heads,T,T]`.
- Key padding mask is available as `[B,T]` and applied with PyTorch-equivalent semantics.

Replacement:

```text
QKV linears -> reshape heads -> FusedAttention(q,k,v, additive_bias=gated_bias, key_padding_mask) -> out linear
```

Weight transform:

```python
w_qkv = torch.cat([w_q, w_k, w_v], dim=0)
b_qkv = torch.cat([b_q, b_k, b_v], dim=0)
```

Failure cases:

- Do not route through an attention backend that only supports causal masks or static per-head relative bias; WavLM bias is hidden-state gated.
- `output_attentions=True` in source returns averaged PyTorch attention weights broadcast back over heads; optimized inference can initially reject attention-weight outputs.

Parity test sketch:

- Compare one attention layer with and without padding mask for fixed `T`, then compare full encoder.

### Rewrite: precompute relative bucket IDs and base position bias

Preconditions:

- Feature length `T` is known or bucketed.
- Layer-0 `rel_attn_embed` weights are loaded.

Replacement:

```text
Runtime arange/log/bucket/gather -> cached position_bias[T,T,heads]
```

Failure cases:

- Batch size changes are fine; feature length changes need another cache entry.
- Gate remains dynamic per layer and cannot be removed.

### Rewrite: TDNN linear storage to Conv1d

Preconditions:

- XVector head only.
- `TDNNLayer.kernel.weight` shape is `[out_dim, in_dim * kernel]`.
- Source transform is `weight.view(out_dim, kernel, in_dim).transpose(1, 2)`.

Replacement:

```text
Linear-stored weight -> Conv1d weight [out_dim, in_dim, kernel] -> Conv1d(dilation=tdnn_dilation, stride=1) -> ReLU
```

Failure cases:

- LoRA on TDNNLayer is warned as unsupported by source optimization; DinoML should reject or bypass this rewrite if adapter modules alter the linear.

### Layout guard: source 1D axes

Candidate optimized layout: local channel-last `[B,T,C]` can help LayerNorm/Linear-heavy regions after feature projection. Guard the feature Conv1d stack, positional Conv1d, adapter Conv1d/GLU, and TDNN Conv1d as source-layout islands unless the pass rewrites:

- Conv1d channel axis `dim=1`.
- LayerNorm axes after explicit transposes.
- GroupNorm `num_channels` axis.
- GLU `dim=1` in adapter layers.
- TDNN `transpose(1,2)` and Conv1d weight transform.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d feature extractor provider path: it dominates early waveform processing and is shape/axis sensitive.
- Gated relative-position bias + attention: WavLM-specific correctness and performance hinge on avoiding slow materialized `[B,H,T,T]` handling where possible.
- Bias GEMM for Q/K/V/O and FFN linears.
- LayerNorm + residual patterns for both post-LN and stable pre-LN variants.

Medium priority:

- Feature projection `LayerNorm(512) + Linear(512 -> H)`.
- Grouped positional Conv1d + SamePad + GELU.
- QKV packed projection, guarded by exact independent-weight concat order.
- Sequence/head pooling with masks for classification and XVector.

Lower priority:

- GPU implementation of CPU feature extractor normalization/padding.
- Adapter Conv1d + GLU path because representative sampled configs set `add_adapter=false`.
- Training-only SpecAugment, Gumbel vector quantizer, contrastive loss, CTC loss.

## 11. Runtime staging plan

1. Parse `WavLMConfig`, load base encoder weights, and implement feature-length inference exactly.
2. Bring up Conv1d feature extractor parity for base `feat_extract_norm="group"` and large `feat_extract_norm="layer"`.
3. Implement feature projection and positional Conv1d parity.
4. Implement WavLM attention custom math: bucketed relative bias, per-layer gate, key padding mask.
5. Validate one encoder layer, then 12-layer base-plus encoder and 24-layer large stable-LN encoder.
6. Add CTC head with tokenizer-side decoding kept in CPU/postprocess.
7. Add sequence and frame classification heads, including optional weighted layer sum.
8. Add XVector head: TDNN transform, statistic pooling, embeddings/logits.
9. Optimize: Conv1d provider selection, QKV packing, fused gated-bias attention, LayerNorm/residual fusion.

Initial stubs/deferments:

- Reject `add_adapter=true` unless a checkpoint requiring adapters is selected.
- Reject `output_attentions=True` for optimized path.
- Treat preprocessing as CPU/data-pipeline work and accept already prepared tensors.

## 12. Parity and validation plan

- Processor parity: input padding, sampling-rate rejection, normalization true/false, attention-mask dtype and shape.
- Length parity: `_get_feat_extract_output_lengths` for scalar and batched lengths across short, exact, and non-multiple-of-320 waveforms.
- Feature extractor parity: each Conv1d layer output for base and large configs.
- Positional Conv1d parity: effective weight-norm Conv1d, SamePad removal, GELU.
- Relative bucket unit tests: compare bucket IDs and base bias for several `T` values.
- Gated bias unit tests: compare gate outputs and attention scores for random hidden states.
- Single attention and single encoder block parity with and without padding mask.
- Full encoder parity against `microsoft/wavlm-base-plus` and `microsoft/wavlm-large` using fixed eval mode.
- CTC logits parity for the sampled CTC checkpoint; decode parity can be a postprocess test.
- Frame classification and XVector parity using the sampled SUPERB-style checkpoints.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 `rtol=2e-2, atol=2e-2`, with attention softmax/gated bias preferably accumulated in fp32 for reduced precision.

## 13. Performance probes

- CPU/data preprocessing throughput: samples/sec for padding and optional normalization.
- Conv1d front-end throughput by waveform length and batch size.
- Feature length sweep: 1s, 5s, 10s, 30s audio; record `T_feat` and attention memory.
- Encoder-only throughput for base and large, split into attention, FFN, LayerNorm, and positional Conv1d.
- Gated relative-bias overhead: materialized bias time/memory versus fused bias generation.
- Attention backend comparison: eager matmul + bias + key padding mask versus DinoML fused attention.
- Head probes: CTC logits projection cost by vocab size, XVector TDNN/statistic pooling cost, weighted-layer-sum memory overhead.
- Peak memory probes for `[B,heads,T,T]` gated bias and attention probabilities at long audio lengths.

## 14. Skip/defer list

- Training, gradient checkpointing, LayerDrop/dropout train-mode behavior.
- SpecAugment random mask generation except explicit `mask_time_indices` if needed for pretraining parity.
- Gumbel vector quantizer and contrastive pretraining losses.
- CTC loss computation; inference needs logits and postprocess only.
- Adapter loading via `target_lang` and `add_adapter=true` until an adapter checkpoint is targeted.
- `output_attentions=True` exact PyTorch averaged-attention behavior.
- LoRA/PEFT-altered TDNN layers.
- Autoregressive generation, KV cache, beam search, speculative decoding: not applicable.
- Multi-GPU/tensor parallel and quantization for first integration.

## 15. Final implementation checklist

- [ ] Parse `WavLMConfig` and checkpoint processor fields.
- [ ] Implement Wav2Vec2FeatureExtractor-compatible CPU preprocessing or accept prepared `input_values`/`attention_mask`.
- [ ] Implement feature length and reduced attention-mask math.
- [ ] Load Conv1d, norm, projection, positional-conv, encoder, and head weights.
- [ ] Implement Conv1d feature extractor for group-norm and layer-norm variants.
- [ ] Implement weight-normalized grouped positional Conv1d with SamePad.
- [ ] Implement WavLM relative bucket computation.
- [ ] Implement hidden-state gated relative-position bias.
- [ ] Implement noncausal MHA with additive gated bias and key padding mask.
- [ ] Implement post-LN and stable pre-LN encoder layer variants.
- [ ] Add one-layer and full-encoder parity tests for base-plus and large.
- [ ] Add CTC logits head and tokenizer-side decode validation.
- [ ] Add sequence/frame classification weighted-layer-sum heads.
- [ ] Add XVector TDNN/statistic-pooling head.
- [ ] Add guarded Conv1d and attention rewrite passes.
- [ ] Benchmark preprocessing, Conv1d front end, encoder, gated bias, and heads separately.
