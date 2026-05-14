# UniSpeech-SAT Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/unispeech-sat-base, microsoft/unispeech-sat-large, microsoft/unispeech-sat-base-100h-libri-ft, microsoft/unispeech-sat-large-sd, microsoft/unispeech-sat-base-sv
Config source: downloaded config.json and preprocessor_config.json snapshots in this folder
Source files inspected:
- X:/H/transformers/src/transformers/models/unispeech_sat/configuration_unispeech_sat.py
- X:/H/transformers/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py
- X:/H/transformers/src/transformers/models/unispeech_sat/modular_unispeech_sat.py
- X:/H/transformers/src/transformers/models/wav2vec2/modeling_wav2vec2.py for inherited modular source context
Any missing files or assumptions:
- `modeling_unispeech_sat.py` is generated from `modular_unispeech_sat.py`; future source edits should start from the modular file, but this report uses the generated file for explicit in-library behavior.
- No gated/401 official configs were encountered. Tokenizer configs are absent (404) for base pretraining, large pretraining, large-sd, and base-sv; only the CTC checkpoint downloaded tokenizer_config.json.
- No DinoML tests were run.
```

Primary source URLs:
- [configuration_unispeech_sat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/unispeech_sat/configuration_unispeech_sat.py)
- [modeling_unispeech_sat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/unispeech_sat/modeling_unispeech_sat.py)
- [modular_unispeech_sat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/unispeech_sat/modular_unispeech_sat.py)

## 2. High-level architecture

UniSpeech-SAT is an audio encoder family derived from Wav2Vec2-style raw-waveform modeling. It is not an autoregressive decoder and has no decode-time KV cache. The useful first DinoML target should be encoder inference plus one head, preferably CTC or XVector depending on product need.

```text
raw mono waveform -> CPU feature extractor padding/normalization -> Conv1d feature encoder
  -> feature projection -> optional SpecAugment/masked_spec_embed in training
  -> convolutional positional embedding -> bidirectional Transformer encoder
  -> task head: CTC logits / sequence logits / frame logits / XVector embedding
```

Stage decomposition:
- CPU/data pipeline: audio decode, resample to 16 kHz, mono waveform production, right padding, optional zero-mean/unit-variance normalization, `attention_mask`.
- GPU/runtime stage 1: strided Conv1d feature extraction and feature-length mask downsampling.
- GPU/runtime stage 2: Transformer encoder with bidirectional self-attention.
- GPU/runtime stage 3: selected head. CTC emits frame-level vocabulary logits; XVector emits pooled embeddings and optional AM-Softmax training logits; frame classification emits per-frame labels.
- Independently cacheable outputs: encoder hidden states can be reused for different heads; XVector embeddings can be cached for verification/scoring. There is no autoregressive cache.

## 3. Important config dimensions

Source defaults from `UniSpeechSatConfig`:

| Field | Default |
| --- | --- |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | `hidden_size // num_attention_heads` = 64 |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `conv_dim` | `[512, 512, 512, 512, 512, 512, 512]` |
| `conv_stride` | `[5, 2, 2, 2, 2, 2, 2]`; total stride 320 |
| `conv_kernel` | `[10, 3, 3, 3, 3, 2, 2]` |
| `conv_bias` | false |
| `feat_extract_norm` | `group` |
| `num_conv_pos_embeddings` | 128 |
| `num_conv_pos_embedding_groups` | 16 |
| `do_stable_layer_norm` | false |
| `vocab_size` | 32 |
| `tdnn_dim/kernel/dilation` | `[512,512,512,512,1500]`, `[5,3,3,1,1]`, `[1,2,3,1,1]` |
| `xvector_output_dim` | 512 |
| `cache support` | no KV cache; attention backend can use eager/SDPA/Flash/Flex interfaces |

Representative checkpoint sweep:

| Checkpoint | Architecture | Hidden/layers/heads | Norm path | Conv bias | Head/operator variation |
| --- | --- | ---: | --- | --- | --- |
| `microsoft/unispeech-sat-base` | `UniSpeechSatForPreTraining` | 768 / 12 / 12 | group first conv; post-LN encoder | false | pretraining quantizer fields present; current `ForPreTraining.forward` returns extracted features and leaves full contrastive loss TODO |
| `microsoft/unispeech-sat-large` | `UniSpeechSatForPreTraining` | 1024 / 24 / 16 | per-conv LayerNorm; stable pre-LN encoder | false | larger codevector/projection dims 768 |
| `microsoft/unispeech-sat-base-100h-libri-ft` | `UniSpeechSatForCTC` | 768 / 12 / 12 | base path | false | CTC vocabulary size 32, tokenizer config present |
| `microsoft/unispeech-sat-large-sd` | `UniSpeechSatForAudioFrameClassification` | 1024 / 24 / 16 | large stable path | true | weighted layer sum enabled, frame classifier, `vocab_size` 256 |
| `microsoft/unispeech-sat-base-sv` | `UniSpeechSatForXVector` | 768 / 12 / 12 | base path | false | weighted layer sum, TDNN stack, statistic pooling, 512-d embedding |

For a 16,000-sample waveform, the default convolution stack emits 49 feature frames. The effective stride is 320 samples (20 ms at 16 kHz), and the inferred receptive field is 400 samples (25 ms) from source kernel/stride math.

## 3a. Family variation traps

- `do_stable_layer_norm` changes block ordering. Base uses attention -> residual -> LN -> FFN -> residual -> final LN. Large/stable uses LN -> attention -> residual -> final LN -> FFN -> residual, then a final encoder LN.
- `feat_extract_norm` changes Conv1d normalization. `group` normalizes only the first conv with `GroupNorm(num_groups=out_channels)`; `layer` applies per-conv `LayerNorm` after transposing to `[B,T,C]`.
- `conv_bias` is checkpoint-dependent: large-sd enables it while base/large pretraining snapshots do not.
- Attention is full MHA only. `hidden_size` must be divisible by `num_attention_heads`; no GQA/MQA field exists.
- `use_weighted_layer_sum` changes heads by requiring all hidden states, stacking `[B, layers+1, T, C]`, softmaxing learned layer weights, and summing across layers.
- CTC adapter support is present through `adapter_attn_dim`, `add_adapter`, `output_hidden_size`, `target_lang`, and external adapter files, but representative audited configs did not enable adapters. Sequence/frame classification explicitly reject `add_adapter=True`.
- `ForPreTraining` contains a Gumbel vector quantizer and speaker projection parameters, but the generated forward has a TODO and does not execute the full contrastive/speaker objective path for inference parity.
- XVector TDNN layers are stored as `nn.Linear` weights but executed with `F.conv1d` after reshaping/transposing the weight; a loader must preserve this layout.
- Layout trap: source Conv1d regions use `[B,C,T]` (`NCT`) after injecting a singleton channel, while encoder/head regions use `[B,T,C]`. Treat any channel-last/NCW rewrite as a local guarded optimization, not a semantic default.

## 4. Operator coverage checklist

Tensor/layout ops:
- Add singleton channel: `[B,L] -> [B,1,L]`.
- Transpose `[B,C,T] <-> [B,T,C]`.
- Reshape/view QKV into `[B,T,H,D]`, transpose to `[B,H,T,D]`, contiguous flatten back to `[B,T,C]`.
- Boolean mask construction, `cumsum`, `flip`, advanced indexed assignment for masks.
- Hidden-state stack and weighted sum for `use_weighted_layer_sum`.
- Mean/std reductions over time for classification/XVector pooling.

Neural primitives:
- Conv1d feature encoder: 7 layers, no padding, strides `[5,2,2,2,2,2,2]`, kernels `[10,3,3,3,3,2,2]`, channels `1 -> 512 -> ... -> 512`.
- Conv1d positional embedding: grouped Conv1d `hidden_size -> hidden_size`, kernel 128, padding 64, groups 16, weight normalization, trim one frame for even kernel, GELU.
- GroupNorm, LayerNorm, Linear, Dropout as inference identity, GELU, ReLU.
- MLP: `Linear(C -> intermediate) -> GELU -> Linear(intermediate -> C)`.

Attention primitives:
- Bidirectional self-attention with Q/K/V/O Linear(C,C) biases enabled by source attention default.
- Eager math is `matmul(q, k.T) * head_dim^-0.5`, additive mask, softmax over key length, matmul with V.
- Source dispatch also supports SDPA/Flash/Flex through `ALL_ATTENTION_FUNCTIONS`; DinoML can initially lower eager-equivalent full attention and later route to optimized bidirectional attention.

Position/custom math:
- Convolutional positional embedding only. No RoPE, ALiBi, learned absolute position table, relative bias, or cache position math.

Generation/cache ops:
- Not applicable. CTC decoding, if desired, is a postprocessing/controller step outside the neural graph.

Preprocessing-coupled ops:
- Wav2Vec2 feature extractor ABI: raw float waveform, 16 kHz, feature size 1, right padding, padding value 0/0.0, optional normalization, optional attention mask.
- Feature-length downsampling from sample mask to conv-frame mask.

Speaker/vector paths:
- XVector: projection `hidden_size -> tdnn_dim[0]`, TDNN Conv1d stack, statistic pooling concat mean/std, `Linear(3000 -> 512)`, `Linear(512 -> 512)` logits.
- AMSoftmax training loss normalizes features/weights, does `mm`, subtracts margin on target classes, scales by 30. Inference embeddings are `feature_extractor(statistic_pooling)`.

Quantized/packed metadata:
- No source-coupled inference quantized weight format. Pretraining quantizer is model math, not a deployment quantized weight format.

## 5. Layer/block breakdown

Feature encoder:

```text
input_values: [B,L]
x = input_values[:, None]                        # [B,1,L]
for i in 0..6:
  x = Conv1d(in_i, out_i=512, kernel_i, stride_i, bias=config.conv_bias)(x)
  if feat_extract_norm == "group" and i == 0: x = GroupNorm(512 groups)(x)
  if feat_extract_norm == "layer": x = LayerNorm(512)(transpose_to_BTC(x)); x = transpose_to_BCT(x)
  x = GELU(x)
extract_features = transpose(x, 1, 2)            # [B,Tc,512]
```

Feature projection:

```text
norm_extract = LayerNorm(512)(extract_features)
hidden = Linear(512 -> hidden_size)(norm_extract)
```

Encoder prelude:

```text
if attention_mask: zero padded hidden frames
attn_mask = bidirectional additive mask from reduced frame mask
pos = grouped_weight_norm_Conv1d(hidden_size -> hidden_size, kernel=128, groups=16)(hidden)
pos = trim_last_if_even_kernel(pos); pos = GELU(pos)
hidden = hidden + pos
if nonstable: hidden = LayerNorm(hidden); hidden = Dropout(hidden)
if stable: hidden = Dropout(hidden)
```

Non-stable encoder layer, repeated `num_hidden_layers`:

```text
res = hidden
q,k,v = Linear(hidden_size -> hidden_size)(hidden) each
attn = bidirectional_attention(q,k,v, mask)
hidden = res + Linear(hidden_size -> hidden_size)(attn)
hidden = LayerNorm(hidden)
hidden = hidden + MLP(hidden)
hidden = LayerNorm(hidden)
```

Stable encoder layer, repeated `num_hidden_layers`:

```text
res = hidden
hidden_norm = LayerNorm(hidden)
attn = bidirectional_attention(q,k,v from hidden_norm, mask)
hidden = res + Linear(hidden_size -> hidden_size)(attn)
hidden = hidden + MLP(LayerNorm(hidden))
if adapter_attn_dim is not None: hidden = hidden + AdapterLayer(hidden)
after all layers: hidden = LayerNorm(hidden)
```

CTC head:

```text
hidden = Dropout(encoder_hidden)
logits = Linear(output_hidden_size or hidden_size -> vocab_size)(hidden)
```

XVector head:

```text
hidden = optional weighted layer sum or last hidden
hidden = Linear(hidden_size -> tdnn_dim[0])(hidden)
for TDNN layer i:
  hidden = Conv1d(in=tdnn_dim[i-1 or 0], out=tdnn_dim[i], kernel=tdnn_kernel[i], dilation=tdnn_dilation[i])
  hidden = ReLU(hidden)
pooled = concat(mean(hidden, time), std(hidden, time))  # [B, 2 * tdnn_dim[-1]]
embedding = Linear(2 * tdnn_dim[-1] -> xvector_output_dim)(pooled)
logits = Linear(xvector_output_dim -> xvector_output_dim)(embedding)
```

## 6. Attention requirements

- Type: encoder-only bidirectional self-attention.
- Heads: MHA, with `num_key_value_heads == num_attention_heads` by construction.
- Head dim: base and large both 64 in audited configs.
- Projection widths: Q/K/V/O all `hidden_size -> hidden_size` with bias.
- Query/key/value length: same conv-frame length `Tc` for self-attention.
- Masking: reduced frame `attention_mask` is converted to an additive bidirectional mask by `create_bidirectional_mask`; padded hidden frames are zeroed before mask construction.
- Packed/varlen support: not present in source. Batch padding is represented by `attention_mask`.
- Sliding/local/block attention: none.
- Position interaction: no RoPE/relative bias; convolutional positional embedding is added before encoder layers.
- KV cache: none.
- Backend compatibility: `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are true. First DinoML parity can use dense full attention; optimized kernels must preserve bidirectional noncausal masking and source scaling order.

## 7. Position encoding and custom math

The only positional signal is a grouped convolution over hidden channels:

```python
def unispeech_sat_pos_conv(hidden_btc, conv_weight_normed):
    x = hidden_btc.transpose(1, 2)       # [B,C,T]
    x = conv1d(x, kernel=128, padding=64, groups=16)
    x = x[:, :, :-1]                    # because kernel 128 is even
    x = gelu(x)
    return x.transpose(1, 2)            # [B,T,C]
```

The conv uses PyTorch weight normalization on `dim=2`; materialized inference should either fold weight norm into a dense Conv1d weight at load time or implement the same normalization before launch. The positional convolution depends on runtime sequence length but not on absolute sample positions beyond the convolution.

Feature-length mask math:

```python
def conv_out_len(length, kernels, strides):
    for k, s in zip(kernels, strides):
        length = floor((length - k) / s) + 1
    return length
```

The reduced attention mask marks all feature frames before `output_lengths - 1` as valid via a one-hot index, reverse cumulative sum, and boolean conversion.

## 8. Preprocessing and input packing

Preprocessor snapshots use `Wav2Vec2FeatureExtractor`:

| Field | Observed value |
| --- | --- |
| `sampling_rate` | 16000 |
| `feature_size` | 1 |
| `do_normalize` | true |
| `padding_side` | right |
| `padding_value` | 0 or 0.0 |
| `return_attention_mask` | false in snapshots |

Runtime tensor ABI:
- `input_values`: float tensor `[B, L]`, raw mono waveform samples after decode/resample/normalization.
- `attention_mask`: optional integer/bool tensor `[B, L]` over raw samples. Even though preprocessor configs default `return_attention_mask=false`, model source supports it and it is required for exact padded pooling/CTC lengths.
- `mask_time_indices`: optional bool `[B, Tc]`, training/pretraining only. It should be rejected or treated as a training-only input for inference targets.

CPU/data pipeline should own audio decode, resampling, mono conversion, padding, and normalization. GPU graph should begin at `input_values` and optional `attention_mask`. No STFT, FFT, mel filterbank, windowing, or chunk/split/reassembly policy is present in this model family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature Conv1d stack to NCW kernels

Source pattern:
```text
[B,L] -> unsqueeze channel -> Conv1d/GELU stack -> transpose to [B,T,512]
```

Replacement: keep an internal NCW layout for all seven feature convs, fuse activation and first normalization where supported, then transpose once before feature projection.

Preconditions:
- Input is contiguous `[B,L]`.
- Conv config lengths match and use source no-padding semantics.
- `feat_extract_norm` path is known: `group` or `layer`.

Failure cases:
- Dynamic conv config not in manifest.
- `layer` norm path requires transpose or an NCW-aware LayerNorm over channels.

Parity test sketch: random waveform plus explicit padding lengths, compare every conv output and final `[B,T,512]` in fp32.

### Rewrite: non-overlap/small Conv1d to GEMM/im2col

Source pattern: each feature Conv1d and TDNN Conv1d.

Replacement:
```text
unfold temporal windows -> GEMM(weight.T) -> bias -> activation -> reshape
```

Preconditions:
- Static kernel/stride/dilation per layer.
- No source padding for feature/TDNN convs.
- Preserve PyTorch cross-correlation order, not mathematical convolution reversal.
- For TDNN, transform linear weight:
```python
w = linear.weight.view(out_dim, kernel, in_dim).transpose(1, 2)
```

Failure cases:
- High temporary memory from im2col at long audio lengths.
- TDNN dilation must be applied in window extraction.

### Rewrite: positional Conv1d weight norm fold

Source pattern: weight-normalized grouped Conv1d with kernel 128, same padding, trim one, GELU.

Replacement: fold `weight_g/weight_v` into a dense grouped Conv1d weight at load/materialization, then run ordinary grouped Conv1d.

Preconditions:
- Weight norm parameters are present or already removed by checkpoint loading.
- Fold uses PyTorch dim=2 normalization.

Failure cases:
- Fine-tuning/training with mutable weight norm is out of scope.

### Rewrite: separate Q/K/V linears to packed QKV GEMM

Source pattern: three independent `Linear(C,C)` projections with split as Q, K, V.

Replacement: concatenate weights in `[Q; K; V]` output-row order and run one GEMM producing `[B,T,3C]`, then split into Q/K/V.

Preconditions:
- All three projections share same input hidden states.
- Biases are present and packed in Q, K, V order.
- No adapter or external hook modifies individual projections.

Failure cases:
- Debug output requiring separate intermediate tensors.

### Rewrite: weighted layer sum

Source pattern: stack hidden states, softmax learned scalar weights, broadcast multiply, sum over layers.

Replacement: streaming weighted accumulation to avoid materializing `[B,Llayers,T,C]`.

Preconditions:
- All hidden states requested only for weighted sum, not returned to caller.

Failure cases:
- `output_hidden_states=True` as a public output requires preserving the tuple.

### Layout guidance

Initial graph translation should preserve source axes: feature/pos convs in `[B,C,T]`, encoder/head tensors in `[B,T,C]`. A layout pass may keep local conv islands in NCW or a backend-specific channel-last 1D layout, but axis-sensitive operations requiring guards include LayerNorm over channel, mean/std over `dim=1`, softmax over attention key `dim=-1`, CTC `log_softmax(dim=-1).transpose(0,1)`, and TDNN transpose/conv/transpose.

## 10. Kernel fusion candidates

Highest priority:
- Feature Conv1d + norm + GELU stack. This is the front-end bottleneck for raw waveform inference and has fixed small kernels/strides.
- LayerNorm kernels for `[B,T,C]`, especially large 1024-d stable path.
- QKV packed projection + bidirectional attention + output projection for base/large encoder blocks.
- MLP Linear/GELU/Linear with dropout removed for inference.

Medium priority:
- Positional grouped Conv1d with folded weight norm and GELU.
- Attention mask downsampling and additive mask construction as a compact runtime shape/mask helper.
- XVector TDNN Conv1d stack plus ReLU, statistic pooling mean/std.
- Weighted layer-sum streaming accumulation for classification/XVector/frame heads.

Lower priority:
- Pretraining Gumbel quantizer path, speaker projection, contrastive logits. Current inference reports can defer because source forward leaves full pretraining objective unfinished.
- Adapter loading and adapter attention layer path for multilingual CTC variants not present in audited representative configs.
- CTC loss; inference needs logits and can leave loss/postprocessing to a separate controller.

## 11. Runtime staging plan

Stage 1: config and preprocessing ABI loader. Parse conv/encoder/head fields, reject unsupported adapters initially, and load Wav2Vec2 feature extractor metadata.

Stage 2: feature encoder parity. Implement raw waveform `[B,L]` to `[B,T,512]` with attention-mask length reduction; validate base and large norm variants.

Stage 3: encoder block parity. Implement positional Conv1d, non-stable and stable blocks, full bidirectional attention, MLP, and hidden-state output option.

Stage 4: first head parity. Choose CTC for ASR logits or XVector for speaker embeddings. CTC is simpler after encoder; XVector exercises weighted layer sum, TDNN, and statistic pooling.

Stage 5: optimized lowering. Add QKV packing, attention backend selection, conv/GELU fusion, weight-norm folding, and TDNN Conv1d lowering.

Stage 6: optional heads and adapter path. Add sequence/frame classification, weighted layer sum public-output handling, CTC adapters only if an enabled checkpoint is in scope.

## 12. Parity and validation plan

- Config validation: ensure conv lists have equal length and `hidden_size % num_attention_heads == 0`; compare effective output lengths against source `_get_feat_extract_output_lengths`.
- Feature encoder: random fp32 waveforms with lengths `[16000, 16001, short boundary]`, compare after every Conv1d/norm/GELU layer.
- Mask reduction: random right-padded masks, compare reduced frame mask exactly.
- Positional Conv1d: compare folded-weight implementation against PyTorch weight-norm module.
- One encoder layer: base non-stable and large stable variants, compare fp32 with dropout disabled.
- Full encoder: base and large configs at short sequence lengths; tolerance fp32 `1e-4` absolute/relative, fp16/bf16 looser after layernorm/attention.
- CTC head: compare logits for `microsoft/unispeech-sat-base-100h-libri-ft`; exclude CTC loss from first inference parity.
- XVector head: compare embeddings and logits for `microsoft/unispeech-sat-base-sv`, including attention-mask bounded pooling.
- Frame classification: compare per-frame logits for `microsoft/unispeech-sat-large-sd`.

## 13. Performance probes

- CPU preprocessing throughput: decode/resample/normalize/pad, separated from GPU graph time.
- Feature Conv1d stack throughput versus waveform length and batch size.
- Encoder-only throughput for base and large: sweep `B` and post-conv frame count `Tc`.
- Attention backend comparison: eager-equivalent dense, SDPA-like, Flash-compatible bidirectional.
- Positional Conv1d grouped kernel time, with and without weight-norm folding.
- Weighted layer sum memory/time for heads requiring all hidden states.
- XVector TDNN/stat-pooling throughput, especially long utterances.
- End-to-end CTC logits throughput in audio seconds/sec and batch latency.
- Memory probes for hidden-state retention when `use_weighted_layer_sum=True`.

## 14. Skip/defer list

- Training losses: CTC loss, AMSoftmax loss, Gumbel-softmax training, contrastive/diversity pretraining.
- SpecAugment random masking and `mask_time_indices` execution for inference.
- LayerDrop and dropout randomness; inference uses deterministic disabled dropout.
- Adapter external file loading and `target_lang` unless an adapter checkpoint is explicitly targeted.
- Full `ForPreTraining` objective parity; current source forward is not a complete pretraining loss path.
- Beam search/CTC decoding and text normalization; keep as postprocessing.
- LoRA/PEFT special handling in TDNN.

## 15. Final implementation checklist

- [ ] Parse `UniSpeechSatConfig` and preprocessor metadata.
- [ ] Load Conv1d, LayerNorm/GroupNorm, Linear, positional conv weight-norm, and task-head weights.
- [ ] Implement feature Conv1d stack with exact output-length math.
- [ ] Implement reduced attention-mask construction.
- [ ] Implement feature projection and optional inference rejection for SpecAugment inputs.
- [ ] Implement positional grouped Conv1d with same-pad trim and GELU.
- [ ] Implement non-stable and stable encoder block variants.
- [ ] Implement bidirectional full self-attention with additive mask and no KV cache.
- [ ] Add QKV packing rewrite with Q/K/V row-order guard.
- [ ] Add positional weight-norm folding rewrite.
- [ ] Add CTC logits head.
- [ ] Add XVector TDNN/stat-pooling embedding head.
- [ ] Add weighted-layer-sum streaming optimization.
- [ ] Add parity tests for base, large, CTC, large-sd, and base-sv configs.
- [ ] Add performance probes for feature conv, encoder attention, and selected head.
