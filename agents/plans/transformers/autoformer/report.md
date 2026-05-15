# Autoformer Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: huggingface/autoformer-tourism-monthly
Config source: official HF config plus accessible public representative configs
Source files inspected: src/transformers/models/autoformer/configuration_autoformer.py, modeling_autoformer.py, src/transformers/time_series_utils.py, tests/models/autoformer/test_modeling_autoformer.py, docs/source/en/model_doc/autoformer.md
Any missing files or assumptions: no processor/tokenizer file is model-coupled for the neural graph; time features are caller/data-pipeline inputs. No gated/401 configs encountered.
```

Primary source links:

- Transformers source: `transformers/src/transformers/models/autoformer/` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Official config: [huggingface/autoformer-tourism-monthly config.json](https://huggingface.co/huggingface/autoformer-tourism-monthly/blob/main/config.json)
- Docs: [Transformers Autoformer docs](https://huggingface.co/docs/transformers/model_doc/autoformer)
- Snapshots: `_sources/source_hotspots.md`, `_sources/config_sweep.md`

## 2. High-level architecture

Autoformer is a time-series encoder-decoder forecasting model, not a text generator. The primary DinoML target should be `AutoformerForPrediction.generate`: probabilistic multistep forecasting with full-horizon decoder execution and distribution sampling.

```text
past values + observed mask + static/time features
-> scaling + lag feature packing
-> encoder AutoCorrelation blocks over context
-> context decomposition into seasonal/trend init
-> decoder AutoCorrelation self/cross blocks over label+horizon window
-> seasonality projection + accumulated trend
-> distribution parameter projection
-> sampling: [batch, num_parallel_samples, prediction_length, input_size?]
```

The encoder can be validated independently because `AutoformerModel` returns `encoder_last_hidden_state` without requiring future values. The prediction head and sampling loop are separately stageable after base encoder-decoder parity.

## 3. Important config dimensions

| Field | Source default | Official tourism monthly |
| --- | ---: | ---: |
| `input_size` | 1 | 1 |
| `prediction_length` | required | 24 |
| `context_length` | defaults to `prediction_length` | 24 |
| `label_length` | 10 | 10 |
| `lags_sequence` | `[1,2,3,4,5,6,7]` | 16 lags up to 37 |
| `feature_size` | computed | 22 |
| `d_model` | 64 | 64 |
| encoder/decoder layers | 2 / 2 | 4 / 4 |
| encoder/decoder heads | 2 / 2 | 4 / 4 |
| FFN dims | 32 / 32 | 32 / 32 |
| `moving_average` | 25 | 25 |
| `autocorrelation_factor` | 3 | 3 |
| `distribution_output` | `student_t` | `student_t` |
| `scaling` | `True` meaning mean scaler | `mean` |
| `num_parallel_samples` | 100 | 100 |

Representative sweep:

| Config | `input_size` | context -> pred | `d_model` | layers enc/dec | heads enc/dec | lag count / max | feature notes |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| official tourism monthly | 1 | 24 -> 24 | 64 | 4 / 4 | 4 / 4 | 16 / 37 | 2 time features, one categorical id |
| traffic hourly | 1 | 48 -> 24 | 16 | 2 / 2 | 2 / 2 | 40 / 721 | no categorical id, 5 time features |
| electricity hourly | 1 | 96 -> 24 | 16 | 4 / 2 | 2 / 2 | 40 / 721 | categorical id cardinality 321 |
| exchange rate | 1 | 60 -> 30 | 16 | 2 / 2 | 2 / 2 | 29 / 780 | `scaling="std"` |
| EEG multivariate | 22 | 336 -> 96 | 512 | 4 / 2 | 16 / 8 | 20 / 20 | multivariate target, `feature_size=489` |

## 3a. Family variation traps

- `feature_size` is not `d_model`; it is `input_size * len(lags_sequence) + static/time/scaling features`. Decoder trend and seasonality projections output `feature_size`.
- `past_values` length must be at least `context_length + max(lags_sequence)`. Some configs have large max lags, such as 721 or 1093.
- `input_size > 1` changes target shape, distribution event shape, static loc/scale feature packing, and loss/sample output rank.
- `scaling` selects mean, std, or no-op scaler; `True` is mean scaling.
- `distribution_output` supports `student_t`, `normal`, and `negative_binomial` in source, though observed configs use `student_t`.
- AutoCorrelation is not SDPA/MHA. It uses FFT, top-k delay selection, and roll/gather aggregation.
- The test suite explicitly skips batching equivalence because the inference `tmp_delay` computation is suspected wrong and `topk` is unstable. DinoML should treat exact source parity as the first target before optimizing.
- `use_cache` exists through generic encoder-decoder cache plumbing, but `generate()` calls the decoder once over the whole horizon with `use_cache=False`. First integration should not implement token-by-token KV decode.
- `layer_norm_eps` appears in one historical official revision but was removed and is not read by the current config/modeling source.

## 4. Operator coverage checklist

Tensor/layout ops:

- Slice, concat, stack, reshape/view, transpose/permute, repeat, repeat_interleave, expand, unsqueeze/squeeze.
- Dynamic range/gather for AutoCorrelation inference delay aggregation.
- Lag subsequence extraction from trailing time windows.
- Mask shape validation and bidirectional mask creation for encoder/cross attention.

Neural primitives:

- Linear projections: value embedding `Linear(feature_size -> d_model, bias=False)`, Q/K/V/O `Linear(d_model -> d_model)`, FFN `Linear(d_model -> ffn_dim) -> activation -> Linear(ffn_dim -> d_model)`, decoder seasonality `Linear(d_model -> feature_size)`.
- `Conv1d(d_model -> feature_size, kernel=3, padding=1, padding_mode="circular", bias=False)` for trend projection.
- `LayerNorm(d_model)` plus Autoformer custom time-mean centering.
- Dropout/layerdrop are training-only for inference.
- Embedding for optional static categorical features.

AutoCorrelation primitives:

- `torch.fft.rfft`, complex conjugate multiply, `torch.fft.irfft`.
- Mean over head/channel, `topk`, `softmax`, `roll` in training-style path, `gather` over repeated values in eval path.
- Output attention tensors are autocorrelation tensors with shape conventions used by tests, not probability matrices over source positions.

Time-series ABI and distributions:

- Mean/std/no-op scaling using observed masks.
- `log1p(abs(loc))`, `log(scale)` static feature construction.
- Distribution parameter projection and domain maps: squareplus, clamp epsilon, StudentT df offset by 2, Normal positive scale, NegativeBinomial total_count/logits.
- Sampling from `torch.distributions`; deterministic graph parity and stochastic end-to-end parity need separate modes.

## 5. Layer/block breakdown

Encoder block, repeated `encoder_layers`:

```text
x = AutoCorrelationSelfAttention(x)
x = LayerNorm(residual + dropout(x))
x, _ = Decomposition(x)
y = Linear(d_model -> encoder_ffn_dim)
y = activation(y)
y = Linear(encoder_ffn_dim -> d_model)
x, _ = Decomposition(x + dropout(y))
x = LayerNorm(x) - mean_time(LayerNorm(x))
```

Decoder block, repeated `decoder_layers`:

```text
x = AutoCorrelationSelfAttention(x)
x, trend1 = Decomposition(residual + dropout(x))
x = LayerNorm(x)
x = AutoCorrelationCrossAttention(query=x, key_value=encoder_hidden)
x, trend2 = Decomposition(residual + dropout(x))
x = LayerNorm(x)
y = Linear(d_model -> decoder_ffn_dim) -> activation -> Linear(decoder_ffn_dim -> d_model)
x, trend3 = Decomposition(x + dropout(y))
x = LayerNorm(x) - mean_time(LayerNorm(x))
residual_trend = CircularConv1d((trend1 + trend2 + trend3).T).T
```

The decoder returns `seasonality_projection(x)` and accumulated `trend`.

## 6. Attention requirements

Autoformer attention is noncausal AutoCorrelation for encoder self-attention, decoder self-attention, and decoder cross-attention. It still projects Q/K/V with full MHA shapes:

```text
q,k,v: [B, T, d_model] -> [B, heads, T, head_dim]
flattened: [B * heads, T, head_dim]
```

It computes autocorrelation over the time axis with FFT, not `QK^T`:

```text
corr = irfft(rfft(q, n=tgt_len, dim=time) * conj(rfft(k, n=tgt_len, dim=time)), n=tgt_len)
top_k = int(autocorrelation_factor * log(time_length))
weights = softmax(mean(corr over head/channel) at top-k delays)
out = sum_i roll_or_gather(v, delay_i) * weights_i
```

Masks are accepted as `[B,1,tgt_len,src_len]`, but source autocorrelation output has trailing `head_dim`; for common self/cross forecasting shapes the mask path needs exact parity tests before relying on optimized kernels. No sliding-window, causal mask, RoPE, ALiBi, GQA, or MQA is present.

Cache note: generic `EncoderDecoderCache` support is wired into the attention module, but `generate()` disables cache and decodes the full prediction window once. Treat KV cache as deferred unless a caller explicitly uses low-level decoder cache APIs.

## 7. Position encoding and custom math

Autoformer uses learned-free sinusoidal positional embeddings plus external time features supplied by the caller. The sinusoidal table is initialized with sine in the first half and cosine in the second half, not interleaved.

Series decomposition:

```python
def decompose(x, kernel):
    pads = (kernel - 1) // 2
    padded = cat([x[:, :1].repeat(1, pads, 1), x, x[:, -1:].repeat(1, pads, 1)], dim=1)
    trend = avg_pool1d(padded.transpose(1, 2), kernel, stride=1).transpose(1, 2)
    return x - trend, trend
```

Autoformer layer norm:

```python
def autoformer_layernorm(x):
    y = layer_norm(x, normalized_shape=d_model)
    return y - y.mean(dim=1, keepdim=True).expand_as(y)
```

Distribution squareplus:

```python
positive = (x + sqrt(x * x + 4.0)) / 2.0
```

## 8. Preprocessing and input packing

The model expects precomputed numeric time features; it does not own calendar feature generation in the modeling file. CPU/data pipeline owns missing-value replacement, observed mask construction, static categorical ids, static real features, and future known covariates.

Runtime graph inputs:

- `past_values`: `[B, context_length + max(lags)]` or `[B, past, input_size]`.
- `past_observed_mask`: same target shape, optional; missing values use zeros in `past_values`.
- `past_time_features`: `[B, past, num_time_features + num_dynamic_real_features]`.
- `future_time_features`: `[B, prediction_length, same_feature_width]` for generation.
- `static_categorical_features`: `[B, num_static_categorical_features]`.
- `static_real_features`: `[B, num_static_real_features]`.

Packing path:

```text
context = last context_length past values
loc/scale = scaler(context, observed_context)
inputs = (past [+ future_values] - loc) / scale
static = [embedded categories?, static real?, log1p(abs(loc)), log(scale)]
features = concat(expanded_static, selected past/future time features)
lagged = stack(inputs[:, -lag-subseq_len : -lag] for lag in lags_sequence)
transformer_inputs = reshape(lagged, [B, seq, input_size * num_lags])
encoder_input = concat(transformer_inputs[:context], features[:context])
```

## 9. Graph rewrite / lowering opportunities

### Rewrite: moving average decomposition -> separable time pooling

Preconditions:

- Kernel is odd or source-compatible even behavior is explicitly reproduced.
- Padding repeats first/last timestep, not zero/reflect padding.
- Input layout remains `[B,T,C]` semantically.

Replacement:

```text
EdgeRepeatPadTime -> AvgPool1d over T per channel -> Subtract
```

Failure cases: even `moving_average` changes length because source pads `(k-1)//2` on each side; guard or reproduce exactly.

### Rewrite: static lag gather -> strided window pack

Preconditions:

- `lags_sequence` is compile-time constant.
- `past_length >= context_length + max(lags)`.
- Input is contiguous along time.

Replacement:

```text
GatherLagWindows -> FlattenLagFeature
```

Parity tests should include large lag values and `input_size > 1`.

### Rewrite: circular Conv1d(k=3) -> small GEMM/stencil

Preconditions:

- `kernel_size=3`, `stride=1`, `padding=1`, `padding_mode="circular"`, `groups=1`, `bias=False`.
- Source layout is `[B,T,d_model]` before projection and `[B,d_model,T]` inside convolution.

Replacement:

```text
CircularTimePad1 -> 3-tap channel projection -> [B,T,feature_size]
```

Failure cases: any non-default convolution attr or layout translation that changes time/channel axes.

### Rewrite: AutoCorrelation fused kernel

Preconditions:

- Fixed `heads`, `head_dim`, and bounded sequence lengths.
- Exact source top-k tie behavior accepted or deterministic policy documented.
- Eval path delay gather semantics reproduced.

Replacement:

```text
QKV GEMM -> batched RFFT/correlation/IRFFT -> topk delays -> weighted gather/roll -> output GEMM
```

Failure cases: unstable `topk`, mask path ambiguity, and source test skip for batching equivalence.

## 10. Kernel fusion candidates

Highest priority:

- Lag packing + feature concat: avoids repeated materialization of large lag stacks for long-lag configs.
- Decomposition moving average: repeated in every encoder and decoder layer; edge-repeat padding plus pool/subtract is a core cost.
- AutoCorrelation block: Q/K/V projections plus FFT/topk/gather dominate the architecture and cannot use normal attention kernels.

Medium priority:

- AutoformerLayernorm: LayerNorm plus time-axis mean subtract.
- Decoder trend circular Conv1d stencil.
- Distribution parameter projection and squareplus/clamp domain map.

Lower priority:

- Static categorical embedding and static feature expansion.
- Sampling loop optimization for many parallel samples; first pass can rely on framework distribution sampling outside DinoML compiled graph.

## 11. Runtime staging plan

1. Parse config and build shape ABI for `past_values`, observed masks, time features, static features, `loc/scale`, and output samples.
2. Implement preprocessing-in-graph subset: scaling, static feature concat, lag packing.
3. Run encoder-only parity for `AutoformerModel` on fixed random tensors.
4. Add decomposition, AutoformerLayernorm, and encoder block parity.
5. Add decoder block with full-horizon cross AutoCorrelation and trend accumulation.
6. Add `AutoformerForPrediction` parameter head for StudentT; defer Normal/NegativeBinomial until covered by tests.
7. Add generation wrapper with `num_parallel_samples`; allow CPU/framework distribution sampling initially if DinoML lacks distribution RNG.
8. Optimize lag packing, decomposition, circular trend projection, then AutoCorrelation.

## 12. Parity and validation plan

- Unit parity for scalers with observed masks, including all-missing rows and `input_size > 1`.
- Unit parity for lag subsequence extraction with max lag larger than context.
- Unit parity for decomposition with odd and even `moving_average`; source default is 25 but config permits other ints.
- AutoCorrelation parity on small `[B,T,d_model]` tensors in eval mode, with explicit tie/top-k cases.
- Single encoder layer and single decoder layer parity at fp32 tolerance around `1e-4`.
- Full `AutoformerModel` parity for official tourism config and test mini config.
- `AutoformerForPrediction` head parity for distribution params before stochastic sampling.
- Generation shape and seeded sample/mean checks for `[B,100,prediction_length]`; exact stochastic parity requires controlling PyTorch distribution RNG.

## 13. Performance probes

- Feature-packing throughput versus `max(lag)`, lag count, and `input_size`.
- Decomposition kernel sweep over `B`, `T`, `feature_size`, and `moving_average`.
- AutoCorrelation sweep over `T`, `heads`, `head_dim`, and `autocorrelation_factor`; split FFT cost from top-k/gather cost.
- Encoder-only throughput.
- Decoder full-horizon throughput.
- End-to-end generation throughput with `num_parallel_samples` sweep: 1, 10, 100.
- Memory probes for repeated batch expansion in `generate()`, especially multivariate EEG-like configs.
- Distribution sampling cost for StudentT versus Normal/NegativeBinomial.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- LayerDrop/dropout behavior except disabled inference parity.
- Generic low-level decoder cache and token-by-token decode.
- Output attentions as a production ABI; retain only for parity/debug initially.
- Normal and NegativeBinomial distribution heads until configs/tests require them.
- Processor/calendar feature generation; caller should provide time features first.
- Masked attention path optimization until exact mask semantics are tested.

## 15. Final implementation checklist

- [ ] Parse `AutoformerConfig` and recompute/validate `feature_size`.
- [ ] Load weights for encoder, decoder, static embeddings, trend conv, and distribution head.
- [ ] Implement mean/std/no-op scalers with observed masks.
- [ ] Implement lag subsequence packing.
- [ ] Implement edge-repeat moving-average decomposition.
- [ ] Implement AutoformerLayernorm.
- [ ] Implement AutoCorrelation eval path with FFT, top-k, and gather/roll aggregation.
- [ ] Implement encoder block parity.
- [ ] Implement decoder block parity with trend accumulation.
- [ ] Implement StudentT parameter projection and domain map.
- [ ] Add seeded generation/shape parity for official tourism monthly.
- [ ] Add multivariate config shape test.
- [ ] Benchmark lag packing, decomposition, AutoCorrelation, decoder horizon, and sampling.

