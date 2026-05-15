# Transformers Audit: `sew_d`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: asapp/sew-d-* public checkpoints, centered on asapp/sew-d-tiny-100k and asapp/sew-d-tiny-100k-ft-ls100h
Config source: local SEWDConfig plus public HF config/preprocessor/vocab/API metadata fetched 2026-05-13
Source files inspected:
- transformers/src/transformers/models/sew_d/configuration_sew_d.py
- transformers/src/transformers/models/sew_d/modeling_sew_d.py
- transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py
- transformers/src/transformers/feature_extraction_sequence_utils.py
- transformers/tests/models/sew_d/test_modeling_sew_d.py
- transformers/docs/source/en/model_doc/sew-d.md
Any missing files or assumptions: no gated/401 checkpoints encountered. HF snapshots are summarized in config_snapshots.md. The source has inherited/dead-looking branches (`ConvLayer`, adapter loading hooks, some config fields) not exercised by representative public SEW-D configs.
```

## 2. High-level architecture

SEW-D is an audio encoder with optional CTC or sequence-classification heads. It consumes normalized mono raw waveform samples, extracts features with a 13-layer strided `Conv1d` stack, projects to transformer hidden size, squeezes time by `squeeze_factor` before a noncausal encoder with DeBERTa-style disentangled relative position attention, then upsamples back to the pre-squeeze feature-frame rate.

```text
raw audio -> Wav2Vec2FeatureExtractor -> conv feature extractor -> feature LN/projection/dropout
  -> positional grouped Conv1d + AvgPool1d squeeze -> transformer encoder
  -> Linear+activation upsampling -> SEWDModel hidden states
  -> CTC logits or masked mean-pool classifier
```

Independently testable stages: CPU/audio preprocessing, convolutional feature extractor, squeezed transformer encoder, upsampling, CTC decoding/loss, and classifier pooling/head. There is no autoregressive prefill/decode stage.

## 3. Important config dimensions

| Field | Source default | Representative values |
| --- | ---: | --- |
| `model_type` | `sew-d` | all inspected checkpoints |
| `hidden_size` | 768 | 384 tiny, 512 small/mid, 768 base-plus |
| `num_hidden_layers` | 12 | 12 tiny/small, 24 mid/base-plus |
| `num_attention_heads` | 12 | 6, 8, 12 |
| `head_dim` | `hidden_size / heads` | 64 in inspected checkpoints |
| `intermediate_size` | 3072 | 1536, 2048, 3072 |
| `vocab_size` | 32 | 32 for CTC heads; base pretraining configs may omit it and use default |
| `conv_dim` | 13 layers ending 512 | base-plus scales to `[96, ..., 768]` |
| `conv_stride` | `(5,2,1,2,1,2,1,2,1,2,1,2,1)` | same inspected; product 320 |
| `conv_kernel` | `(10,3,1,3,1,3,1,3,1,2,1,2,1)` | same inspected |
| `squeeze_factor` | 2 | 2 |
| `num_conv_pos_embeddings` | 128 | 31, or 127 for `mid-k127` |
| `num_conv_pos_embedding_groups` | 16 | 16 |
| `relative_attention` | true | true |
| `position_buckets` | 256 | 256 |
| `max_position_embeddings` | 512 | 512 |
| `pos_att_type` | `("p2c","c2p")` | same |
| `share_att_key` | true | true |
| `feat_extract_norm` | group | group in inspected checkpoints |
| `feat_extract_activation` | gelu | gelu |
| processor sample rate | from preprocessor | 16000 Hz |
| cache support | n/a | no KV cache/generation |

Checkpoint sweep:

| Model | Head | Params metadata | Hidden/layers/heads | Conv dims | Pos conv kernel | Notes |
| --- | --- | ---: | --- | --- | ---: | --- |
| `asapp/sew-d-tiny-100k` | `SEWDModel` | 24.1M safetensors | 384/12/6 | ends 512 | 31 | feature extraction |
| `asapp/sew-d-tiny-100k-ft-ls100h` | `SEWDForCTC` | 24.1M safetensors | 384/12/6 | ends 512 | 31 | ASR, uppercase 32-token CTC vocab |
| `asapp/sew-d-small-100k` | `SEWDModel` | no safetensors in API | 512/12/8 | ends 512 | 31 | feature extraction |
| `asapp/sew-d-mid-400k` | `SEWDModel` | no safetensors in API | 512/24/8 | ends 512 | 31 | deeper encoder |
| `asapp/sew-d-mid-k127-400k-ft-ls100h` | `SEWDForCTC` | 80.4M safetensors | 512/24/8 | ends 512 | 127 | larger positional Conv1d kernel |
| `asapp/sew-d-base-plus-400k-ft-ls100h` | `SEWDForCTC` | 177.0M safetensors | 768/24/12 | ends 768 | 31 | scaled conv channels and hidden size |

## 3a. Family variation traps

- `hidden_size == num_heads * head_dim` in inspected configs, but source allows an explicit `attention_head_size` override after checking only `hidden_size % num_attention_heads == 0`; lowerers should read `attention_head_size` if present.
- Public checkpoint configs include `layerdrop`, `position_biased_input`, `feature_extractor_type`, and `tokenizer_class`; the inspected modeling code does not use `layerdrop` or `position_biased_input`.
- `feat_extract_norm="group"` means only the first feature conv has `GroupNorm(num_groups=out_channels)`; `feat_extract_norm="layer"` switches every conv layer to transpose -> `LayerNorm(C)` -> transpose.
- CTC checkpoints use `vocab_size=32` and uppercase tokenizer vocab. Base feature checkpoints can omit `vocab_size`; `SEWDConfig` default still supplies 32.
- `num_conv_pos_embeddings` is checkpoint-significant: `mid-k127` uses 127, while most ASAPP checkpoints use 31. The source default is 128, which would trigger same-pad removal because even kernels remove one timestep after the padded conv.
- Representative processors default `return_attention_mask=false`; if a caller supplies an attention mask, the model recomputes a reduced conv-frame mask and later a squeezed mask.
- The source has optional adapter-related hooks in `SEWDForCTC.tie_weights`, but the inspected configs do not define `add_adapter`/`adapter_attn_dim`.
- NCHW/NHWC analogue: audio tensors are `NCT` for `Conv1d` regions and `NTC` for transformer/linear regions. Axis-sensitive transposes must be guarded, especially around conv feature extraction, positional conv, LayerNorm, and AvgPool1d.

## 4. Operator coverage checklist

Tensor/layout ops:
- `unsqueeze`, `transpose`, `permute`, `contiguous`, `view`/`reshape`, `repeat`, `expand`, `stack`, `sum`, `mean`, boolean masking, scalar casts, padding, slicing/truncation.
- Audio layout switches: raw `input_values [B, T_audio]` -> feature conv `[B, 1, T]`/`[B, C, T_feat]` -> transformer `[B, T_feat, C]` -> squeezed `[B, T_feat / squeeze, H]`.

Neural network primitives:
- Feature `Conv1d`: 13 unpadded layers, bias usually false, first input channel 1, channel schedule from `conv_dim`, activation `gelu`.
- GroupNorm on first conv only for `feat_extract_norm="group"`: `num_groups=out_channels`, `num_channels=out_channels`.
- LayerNorm after conv feature extraction over `conv_dim[-1]`, optional `Linear(conv_dim[-1] -> hidden_size)`, dropout.
- Positional grouped `Conv1d(hidden_size -> hidden_size, kernel=num_conv_pos_embeddings, padding=floor(k/2), stride=squeeze_factor, groups=num_conv_pos_embedding_groups)` with weight norm, same-pad crop for even kernels, activation.
- `AvgPool1d(squeeze_factor, squeeze_factor)`.
- Transformer layers: Q/K/V `Linear(H -> H)` with bias, attention output `Linear(H -> H)`, FFN `Linear(H -> I)`, activation, `Linear(I -> H)`, dropout, residual adds, LayerNorm.
- Upsampling: `Linear(H -> H * squeeze_factor)`, activation, reshape `[B, S, squeeze, H] -> [B, S*squeeze, H]`.
- Heads: CTC `Linear(H -> vocab_size)`, classifier `Linear(H -> classifier_proj_size)` then masked mean pool then `Linear(classifier_proj_size -> num_labels)`.

Attention primitives:
- Noncausal self-attention, MHA, no KV cache, dense attention matrix.
- `torch.bmm` for QK, relative-bias Q/P and K/P products, and attention-value product.
- Masked softmax implemented by `XSoftmax`: masked fill with dtype minimum, `softmax(dim=-1)`, zero masked probabilities.

Position/relative-bias ops:
- Log-bucketed relative position IDs, embedding lookup of size `2 * position_buckets` when bucketing is active.
- Optional `LayerNorm` on relative embedding table when `norm_rel_ebd` includes `layer_norm`.
- Content-to-position and position-to-content gathered score terms.

Preprocessing-coupled ops:
- 16 kHz mono waveform validation by processor.
- Right padding with zero, optional `attention_mask`.
- Zero-mean/unit-variance normalization per sequence, using the unpadded prefix when a padding mask is present.

Training/loss ops:
- SpecAugment mask generation is NumPy/CPU and training-only unless explicit `mask_time_indices` are supplied.
- CTC training uses `log_softmax(..., dtype=float32).transpose(0,1)` and `ctc_loss(blank=pad_token_id)`, with cuDNN disabled around loss.

## 5. Layer/block breakdown

Feature extractor:

```text
x = input_values[:, None]                         # [B, 1, T]
for i in 13 conv layers:
  x = Conv1d(Cin -> conv_dim[i], kernel[i], stride[i], padding=0, bias=conv_bias)(x)
  if group norm and i == 0: x = GroupNorm(Cout groups)(x)
  if layer norm mode: x = transpose NCT->NTC; x = LayerNorm(Cout); x = transpose NTC->NCT
  x = activation(x)
x = transpose NCT->NTC                            # [B, T_feat, conv_dim[-1]]
x = LayerNorm(conv_dim[-1])
if conv_dim[-1] != hidden_size: x = Linear(conv_dim[-1] -> H)(x)
x = Dropout(feat_proj_dropout)(x)
```

Encoder wrapper:

```text
mask_feat = conv_length(attention_mask.sum(-1)) if provided
x[~mask_feat] = 0
pos = PosConv1d(stride=squeeze_factor)(transpose NTC->NCT x)
pooled = AvgPool1d(squeeze_factor)(transpose NTC->NCT x)
x = transpose((pooled[..., :min_len] + pos[..., :min_len]), NCT->NTC)
x = transformer_encoder(x, squeezed_mask)
x = Linear(H -> H*squeeze_factor)(x)
x = activation(x)
x = reshape channels_to_time back to [B, T_feat_or_less, H]
x = right-pad in time if shorter than original feature length
```

Transformer block repeated `num_hidden_layers` times:

```text
q = Linear(H -> H, bias=True)(x)
k = Linear(H -> H, bias=True)(x)
v = Linear(H -> H, bias=True)(x)
scores = bmm(q, k.T / sqrt(head_dim * scale_factor))
scores += disentangled_relative_bias(q, k, rel_embedding)   # if enabled
probs = masked_softmax(scores, pair_mask)
context = bmm(probs, v)
x_attn = LayerNorm(Dropout(Linear(H -> H)(context)) + x)
ff = activation(Linear(H -> intermediate_size)(x_attn))
x = LayerNorm(Dropout(Linear(intermediate_size -> H)(ff)) + x_attn)
```

Heads:

```text
CTC: logits = Linear(H -> vocab_size)(Dropout(final_dropout)(hidden_states))
classifier: hidden = optional weighted layer sum; proj = Linear(H -> classifier_proj_size);
            pooled = mean(proj, time) or mask-weighted mean; logits = Linear(classifier_proj_size -> num_labels)
```

## 6. Attention requirements

SEW-D requires noncausal dense self-attention only. It is encoder-style and has no autoregressive KV cache, no cross-attention, no packed/varlen metadata ABI, no sliding-window attention, no RoPE, no ALiBi, and no generation decode path.

Attention ABI:

- Input hidden states after squeeze: `[B, S, H]`.
- Q/K/V projected to `[B * heads, S, head_dim]`; representative `head_dim=64`.
- Scores reshaped to `[B, heads, S, S]`.
- Mask from `[B, S]` becomes pair mask `[B, 1, S, S]` by outer product of valid query/key positions.
- Relative position tensor is `[1, S, S]` before unsqueeze to attention-rank shape.
- Softmax is over key axis; dropout is `StableDropout` in attention paths.
- FlashAttention/SDPA compatibility is not direct because source adds DeBERTa-style c2p/p2c relative logits before softmax and uses a boolean pair mask; a fused backend would need a custom relative-bias prepass or integrated score callback.

## 7. Position encoding and custom math

SEW-D has two position mechanisms: a convolutional positional embedding before the squeezed transformer, and disentangled relative attention inside every transformer layer.

```python
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = where((relative_pos < mid) & (relative_pos > -mid),
                    tensor(mid - 1).type_as(relative_pos),
                    abs(relative_pos))
    log_pos = ceil(log(abs_pos / mid) / log((max_position - 1) / mid) * (mid - 1)) + mid
    return where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
```

```python
def build_relative_position(query_size, key_size, bucket_size, max_position):
    q_ids = arange(query_size)
    k_ids = arange(key_size)
    rel = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel = make_log_bucket_position(rel, bucket_size, max_position)
    return rel.long().unsqueeze(0)
```

Disentangled bias summary:

```text
att_span = position_buckets or max_relative_positions
rel_embeddings = rel_embedding_table[:2*att_span]
if share_att_key: project rel embeddings with the same q/k projections as content
c2p: bmm(query, pos_key.T), gather at clamp(relative_pos + att_span), divide by sqrt(head_dim * scale_factor)
p2c: bmm(key, pos_query.T), gather at clamp(-relative_pos + att_span), transpose, divide by same style scale
```

Relative position IDs depend only on squeezed sequence length and can be precomputed per `S`/device. Relative embedding projection depends on learned weights and layer; the projected relative Q/K tables can be cached per layer for static shapes in inference.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- Processor class is `Wav2Vec2FeatureExtractor`/`Wav2Vec2Processor` for CTC checkpoints.
- Input is mono raw speech as `float32` arrays or tensors; stereo/more-than-2D batched NumPy input is rejected.
- Representative checkpoints require `sampling_rate=16000`; passing a different sampling rate raises.
- Padding is right-side, value `0.0`. `pad_to_multiple_of` is available in the generic feature extractor.
- `do_normalize=true`: each sequence is normalized to `(x - mean) / sqrt(var + 1e-7)`. If padding and attention mask are present, mean/variance use only valid samples and padded tail is reset to zero.
- Public preprocessors set `return_attention_mask=false`; for group-norm feature extractors, docs advise zero padding without passing a mask. The model still supports a user-supplied mask and uses it for conv-frame masking, attention masking, CTC lengths, and classifier pooling.

GPU/runtime ABI:

- `input_values`: `[B, T_audio]`, float.
- Optional `attention_mask`: `[B, T_audio]`, integer/bool, right-padded lengths.
- Conv-frame length formula: repeated `floor((input_length - kernel) / stride) + 1` across all feature conv layers.
- With representative conv strides, nominal audio-to-logit ratio is 320 samples per feature frame; actual lengths use the floor formula and can differ for short or ragged inputs.
- CTC logits are `[B, T_feat, vocab_size]`; decode is external CTC argmax/collapse/tokenizer behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature Conv1d stack as provider-backed NCT region

Source pattern: repeated unpadded `Conv1d` in NCT layout with activation and optional normalization.

Replacement: keep a fused NCT conv pipeline or lower each layer to a 1D convolution provider; only cross the NCT/NTC boundary once after the stack.

Preconditions:
- `conv_kernel`, `conv_stride`, `conv_dim` lengths match.
- `conv_bias` known.
- Norm mode is either group-first or layer-every-layer.
- Preserve exact floor length formula.

Failure cases: do not translate layer-norm convs without preserving `LayerNorm(C)` over channel axis after NCT->NTC transpose.

### Rewrite: non-overlap `AvgPool1d(squeeze_factor)` as strided reduction

Source pattern: `AvgPool1d(k=s, stride=s)` on `[B, H, T_feat]`.

Replacement: reshape/gather windows of `squeeze_factor` then mean over window when `T_feat` has enough complete windows.

Preconditions:
- `padding=0`, `ceil_mode=false`, `count_include_pad=true` default semantics.
- Output length must match PyTorch floor pooling.

Failure cases: dynamic tails must be dropped exactly as PyTorch pooling does; later upsample may pad back to `n_input_timesteps`.

### Rewrite: SEW-D upsampling as PixelShuffle1D-style reshape

Source pattern:

```text
Linear(H -> H*s) -> activation -> reshape [B,S,s,H] -> reshape [B,S*s,H]
```

Replacement: a fused linear epilogue that writes directly to expanded time layout.

Preconditions:
- `src_embed_dim % squeeze_factor == 0` after projection; in source this is `H*s`.
- Activation matches config.

Failure cases: output may need right-padding if `S*s < original feature length`.

### Rewrite: relative position matrix cache

Source pattern: build `arange` difference and log buckets per forward.

Replacement: cache `[1,S,S]` bucketed integer matrix per `(S, position_buckets, max_position, device)`.

Preconditions: self-attention only with `query_size == key_size` for normal SEW-D encoder path.

Failure cases: keep generic path for `query_states`/rectangular attention inherited in source.

### Rewrite: CTC inference logit path

Source pattern: `Dropout(0 in eval)` then `Linear(H -> vocab)`.

Replacement: direct GEMM to logits, optionally last-stage argmax outside compiled graph.

Preconditions: inference/eval mode; no dropout.

Failure cases: training loss still needs float32 `log_softmax` and `ctc_loss`.

Layout guidance: do not treat audio `NCT` as NHWC. A layout pass should maintain named axes and only remove transposes when all adjacent consumers agree on channel/time semantics.

## 10. Kernel fusion candidates

Highest priority:
- Feature `Conv1d + norm + GELU` stack, because it is the front-end bottleneck and has many small 1D convolutions.
- Disentangled attention score construction: QK bmm plus c2p/p2c bmm/gather/bias addition before softmax.
- Masked softmax with dropout for `[B, heads, S, S]`.
- LayerNorm + residual after attention and FFN.

Medium priority:
- Positional grouped Conv1d + same-pad crop + GELU.
- Upsampling `Linear + GELU + reshape`.
- QKV projection packing for three independent `Linear(H -> H)` weights, preserving separate source tensors.
- Classifier masked mean pooling.

Lower priority:
- CTC `log_softmax`/loss for training.
- SpecAugment masking; source already says CPU/preprocessing and training-only.
- Weighted layer sum for classification, only when `use_weighted_layer_sum=true`.

## 11. Runtime staging plan

Stage 1: parse `SEWDConfig`, processor config, CTC vocab, and load weights for `asapp/sew-d-tiny-100k`.

Stage 2: implement raw waveform ABI and feature extractor parity: 13-layer Conv1d, group-norm mode, activation, conv length helper.

Stage 3: implement encoder wrapper parity: feature LayerNorm/projection, positional grouped Conv1d with weight norm materialization, AvgPool squeeze, upsample and right-padding.

Stage 4: implement one transformer layer with dense noncausal MHA and disentangled relative bias; validate after one layer and after all layers.

Stage 5: add CTC head and external greedy decode parity for `asapp/sew-d-tiny-100k-ft-ls100h`.

Stage 6: add config variation coverage: small/mid/base-plus dimensions, `num_conv_pos_embeddings=127`, optional layer-norm conv mode from synthetic configs.

Stage 7: optimize with NCT conv region, relative-position cache, fused masked attention, and upsample write fusion.

Initial stubs: training-only SpecAugment, CTC loss, gradient checkpointing, adapter loading, and classifier weighted layer sum can be deferred until inference parity is stable.

## 12. Parity and validation plan

- Processor parity: compare normalized/padded `input_values` and optional `attention_mask` against `Wav2Vec2FeatureExtractor` for single and ragged batches at 16 kHz.
- Conv length tests: random lengths through `_get_feat_extract_output_lengths`, including ragged attention masks and short edge cases.
- Feature extractor parity: each conv layer output and final `[B,T_feat,C]` output against PyTorch fp32, tolerance `rtol=1e-4, atol=1e-5`.
- Positional conv/pool/upsample parity: check squeezed lengths, `min_length` crop, and right-pad behavior.
- Relative position tests: exact integer bucket matrices for several `S`, including distances beyond `position_buckets // 2`.
- Attention parity: one layer with `relative_attention=true`, `pos_att_type=("p2c","c2p")`, both masked and unmasked.
- End-to-end encoder parity: `asapp/sew-d-tiny-100k` integration sample outputs, matching existing Transformers test slices.
- CTC parity: logits and greedy transcripts for `asapp/sew-d-tiny-100k-ft-ls100h`; fp32 tolerance `rtol=1e-3, atol=1e-3` for hidden/logit slices before decode.
- Config sweep smoke: tiny, small, mid, mid-k127, base-plus shape-only and selected numerical checks.

## 13. Performance probes

- Processor throughput: waveform normalization and padding samples/sec on CPU.
- Feature extractor throughput by audio length and batch size.
- Squeezed encoder throughput by `S = conv_len(T_audio) // squeeze_factor`.
- Attention backend comparison for dense MHA plus relative-bias prepass.
- Conv positional embedding kernel-size sweep: 31 vs 127.
- End-to-end ASR latency split into preprocessing, conv front-end, transformer encoder, CTC projection/decode.
- Memory probes for attention `[B, heads, S, S]` and relative bias temporaries.
- Batch-size and audio-duration sweep at 1s, 5s, 10s, 30s.

## 14. Skip/defer list

- Training-only SpecAugment and CTC loss.
- Gradient checkpointing.
- Adapter loading and multilingual adapter selection; representative ASAPP configs do not use it.
- Sequence-classification weighted layer sum until a target checkpoint requires it.
- Optional inherited `ConvLayer` path behind `conv_kernel_size`; not used by representative configs and appears to require fields absent from `SEWDConfig` defaults.
- Beam search or language-model decoding for CTC; greedy decode is enough for first parity.
- Quantization and packed weights; no source-coupled quantized format was found.
- Distributed/tensor-parallel execution.

## 15. Final implementation checklist

- [ ] Parse `SEWDConfig` including checkpoint-supplied fields and ignored-field warnings.
- [ ] Parse `Wav2Vec2FeatureExtractor` preprocessor config and CTC vocab.
- [ ] Implement raw waveform input ABI `[B,T_audio]` plus optional right-padding mask.
- [ ] Implement Conv1d feature extractor with group-first and layer-norm variants.
- [ ] Implement conv-frame length and feature-vector attention-mask helpers.
- [ ] Implement feature LayerNorm/projection/dropout inference path.
- [ ] Implement positional grouped Conv1d with weight-norm materialization and same-pad crop.
- [ ] Implement AvgPool squeeze and Linear/GELU upsampling.
- [ ] Implement dense noncausal MHA with SEW-D/DeBERTa disentangled relative bias.
- [ ] Add relative-position bucket cache.
- [ ] Implement CTC logits head and greedy decode harness.
- [ ] Implement sequence-classification pooling/head after encoder parity.
- [ ] Add tiny checkpoint layerwise and end-to-end parity tests.
- [ ] Add sweep tests for small/mid/base-plus and `mid-k127`.
- [ ] Benchmark preprocessing, conv extractor, squeezed encoder, CTC projection, and full ASR.
