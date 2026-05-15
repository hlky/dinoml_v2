# Time Series Transformer Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: huggingface/time-series-transformer-tourism-monthly, plus public config sweep below
Config source: HF raw config.json and model API, fetched 2026-05-13
Source files inspected:
  transformers/src/transformers/models/time_series_transformer/configuration_time_series_transformer.py
  transformers/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py
  transformers/src/transformers/time_series_utils.py
  transformers/tests/models/time_series_transformer/test_modeling_time_series_transformer.py
Any missing files or assumptions: no processor/tokenizer; caller/data pipeline owns calendar/time features, missing-value imputation, static features, and dataset batching.
```

Snapshots were added under `_sources/`:

- `_sources/source_basis.md`
- `_sources/forecasting_abi.md`
- `_sources/config_sweep.json`

Primary URLs:

- [configuration_time_series_transformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/time_series_transformer/configuration_time_series_transformer.py)
- [modeling_time_series_transformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py)
- [time_series_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/time_series_utils.py)
- [test_modeling_time_series_transformer.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/tests/models/time_series_transformer/test_modeling_time_series_transformer.py)
- [huggingface/time-series-transformer-tourism-monthly](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly)
- [kashif/time-series-transformer-traffic-hourly](https://huggingface.co/kashif/time-series-transformer-traffic-hourly)
- [kashif/time-series-transformer-mv-traffic-hourly](https://huggingface.co/kashif/time-series-transformer-mv-traffic-hourly)
- [scirik/time-series-transformer-electricity-load-diagrams](https://huggingface.co/scirik/time-series-transformer-electricity-load-diagrams)
- [JLB-JLB/EEG_TimeSeriesTransformer_336_history_96_horizon](https://huggingface.co/JLB-JLB/EEG_TimeSeriesTransformer_336_history_96_horizon)

No gated/401/403 configs were encountered in the representative sweep.

## 2. High-level architecture

This is a probabilistic time-series forecasting encoder-decoder, not a text LM. The model converts lagged target values, static features, observed-mask-derived scale statistics, and caller-provided time features into a dense sequence. A BART-like Transformer encoder reads the context window; a causal decoder attends to the encoder output and emits distribution parameters for the forecast horizon.

```text
caller time-series preprocessing -> scaling + lag/static/time feature packing -> encoder context
  -> teacher-forced decoder for training/eval with future_values
  -> distribution parameter head + NLL loss

caller time-series preprocessing -> scaling + lag/static/time feature packing -> encoder context
  -> autoregressive Monte Carlo forecast loop
  -> sampled future values shaped [batch, num_parallel_samples, prediction_length, input_size?]
```

Stage decomposition:

- CPU/data-pipeline: missing-value replacement, observed masks, calendar/age/holiday features, static categorical IDs, static real features, train/validation windows.
- GPU/runtime stage 1: scale context, build lagged covariates, concatenate features.
- GPU/runtime stage 2: encoder over `context_length`.
- GPU/runtime stage 3 training/teacher forcing: decoder over `prediction_length`.
- GPU/runtime stage 3 forecast: sampling loop over `prediction_length`, repeated by `num_parallel_samples`.
- Independently cacheable: encoder output and static/scale features can be reused for one forecast request. Source generation does not exploit decoder KV cache in its inner loop.

## 3. Important config dimensions

Source defaults from `TimeSeriesTransformerConfig`:

| Field | Default / rule |
| --- | --- |
| `model_type` | `time_series_transformer` |
| `prediction_length` | required for model init |
| `context_length` | defaults to `prediction_length` |
| `input_size` | 1 |
| `lags_sequence` | `[1, 2, 3, 4, 5, 6, 7]` |
| `_past_length` | `context_length + max(lags_sequence)` |
| `distribution_output` | `student_t`; also `normal`, `negative_binomial` |
| `scaling` | `"mean"`; `True` maps to mean, `"std"` available, other/false is no-op |
| `num_time_features` | 0 |
| `num_dynamic_real_features` | 0 |
| `num_static_categorical_features` | 0 |
| `num_static_real_features` | 0 |
| `cardinality` | `[0]` if no static categorical features |
| `embedding_dimension` | `[min(50, (cat + 1) // 2)]` if not supplied and categorical features exist |
| `feature_size` | `input_size * len(lags_sequence) + sum(embedding_dimension) + dynamic + time + static_real + input_size * 2` |
| `d_model` | 64 |
| `encoder_layers` / `decoder_layers` | 2 / 2 |
| `encoder_attention_heads` / `decoder_attention_heads` | 2 / 2 |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 32 / 32 |
| `activation_function` | `gelu` |
| `use_cache` | true in config and decoder, but forecast loop does not pass cache to inner decoder calls |
| `num_parallel_samples` | 100 |

Representative checkpoint sweep:

| Model | Source | Horizon/context | Input size | d_model | Layers enc/dec | Heads enc/dec | Lags | Feature size | Static features | Scaling |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `huggingface/time-series-transformer-tourism-monthly` | HF config/API | 24 / 24 | 1 | 26 | 2 / 2 | 2 / 2 | 16 | 27 | cat 1, real 1 | `true` |
| `kashif/time-series-transformer-traffic-hourly` | HF config/API | 24 / 48 | 1 | 16 | 2 / 2 | 2 / 2 | 40 | 47 | none | `mean` |
| `kashif/time-series-transformer-mv-traffic-hourly` | HF config/API | 24 / 48 | 862 | 64 | 2 / 2 | 2 / 2 | 2 | 3453 | none | `mean` |
| `scirik/time-series-transformer-electricity-load-diagrams` | HF config/API | 24 / 48 | 1 | 32 | 4 / 4 | 2 / 2 | 40 | 49 | cat 1 | `mean` |
| `JLB-JLB/EEG_TimeSeriesTransformer_336_history_96_horizon` | HF config/API | 96 / 336 | 22 | 512 | 4 / 2 | 16 / 8 | 20 | 489 | none | `mean` |

## 3a. Family variation traps

- `feature_size` can be much larger than `d_model`; the value embedding is `Linear(feature_size -> d_model, bias=False)`.
- Multivariate configs change target rank and distribution event shape. `input_size=862` makes the Student-T head project three `[B,T,862]` parameter tensors and wraps the scalar distribution in an `Independent` event.
- `d_model` must be divisible by each attention head count, but encoder and decoder head counts can differ.
- `context_length` and `prediction_length` are independent. Forecast ABI requires `past_values` and `past_time_features` length `context_length + max(lags_sequence)`.
- `lags_sequence` may be long and sparse; hourly configs include lags up to 721, causing large required history even with a short context.
- Source generation uses `num_parallel_samples` by repeating the batch, not by vectorizing inside the distribution head only.
- Static categorical support is optional. If enabled, it is ordinary `nn.Embedding` over `static_categorical_features`, then concatenated into every time row.
- Time features are mandatory runtime inputs when configured. They are not learned position IDs; the model also adds fixed sinusoidal embeddings internally after value projection.
- `scaling=True` in older configs is source-equivalent to mean scaling.
- `distribution_output` branches change both output head arity and sampling math.
- Training/teacher forcing with `future_values` disables cache in `TimeSeriesTransformerForPrediction.forward`.
- Source decoder can create and update an `EncoderDecoderCache`, but `generate` does not feed `past_key_values` to the inner decoder loop. DinoML should not claim source-parity decode acceleration until it implements and validates an optimized equivalent.
- No NHWC/NCHW issue: tensors are sequence-major dense `[batch, time, channel/features]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Dense slicing with negative indices over time.
- `torch.cat` over time and feature axes.
- `torch.stack(lagged_values, dim=-1)`.
- `reshape` from `[B,S,input_size,num_lags]` to `[B,S,input_size * num_lags]`.
- `squeeze`, `unsqueeze`, `expand`, `repeat_interleave`.
- Zero tensor creation for no-future decoder bootstrap.
- Optional `torch.chunk(..., dim=-1)` for multiple static categorical features.

Feature/scaling ops:

- Observed-mask weighted sum/count over time.
- `clamp_min`, `where`, `abs`, `log1p`, `log`, `sqrt`, `square`, division.
- Mean scaler, std scaler, and no-op scaler.
- Missing values are not imputed in model source; caller must replace them and pass observed masks.

Neural primitives:

- `Embedding(cardinality_i -> embedding_dim_i)` for static categorical features.
- `Linear(feature_size -> d_model, bias=False)` value embedding in both encoder and decoder.
- Fixed sinusoidal positional embedding table of length `context_length + prediction_length`.
- `LayerNorm(d_model)` after value+position embedding and after each residual block.
- MHA self-attention in encoder and decoder.
- MHA cross-attention in decoder.
- FFN: `Linear(d_model -> encoder_ffn_dim/decoder_ffn_dim)`, activation, `Linear(ffn_dim -> d_model)`.
- Dropout/layerdrop are training only for first inference target.

Attention primitives:

- Encoder bidirectional self-attention over `context_length`.
- Decoder causal self-attention over teacher-forced horizon or generated prefix.
- Decoder cross-attention over encoder hidden states.
- Attention masks are produced by shared Transformers mask utilities and passed to `ALL_ATTENTION_FUNCTIONS`.
- Eager attention path is `QK^T * head_dim^-0.5`, additive mask, softmax over key length, `AV`.

Position encoding:

- Source uses caller-provided time features as forecasting covariates.
- Source also uses frozen sinusoidal positional embeddings. Decoder positions are offset by `context_length`.
- No RoPE, ALiBi, relative bias, or learned token embedding.

Generation/cache ops:

- Encoder output repeat by `num_parallel_samples`.
- Distribution parameter projection at each forecast step.
- `torch.distributions` sampling for Student-T/Normal/NegativeBinomial.
- Append normalized sample to repeated history.
- Optional decoder cache exists in generic decoder forward, but source forecast loop does not use it.

Preprocessing-coupled ops:

- Calendar/age/dynamic-real feature generation is out of model source and must be owned by caller or a separate processor.
- Observed masks are required for scale/loss parity when data has missing values.

## 5. Layer/block breakdown

Network input packing:

```text
context = last context_length rows of past_values
loc, scale = scaler(context, past_observed_mask context)
normalized_inputs = (past_values [+ future_values] - loc) / scale
lagged = gather lagged subsequences for each lag in lags_sequence
static_feat = [embedded static cats?, static real?, log1p(abs(loc)), log(scale)]
features = [static_feat broadcast over time, time features]
transformer_inputs = [flatten(lagged), features]
```

Encoder, repeated `encoder_layers`:

```text
x = Linear(feature_size -> d_model, bias=False)(encoder_inputs)
x = LayerNorm(x + sinusoidal_pos(context positions))
for layer:
  residual = x
  x = MHA_self(x, bidirectional_mask)
  x = LayerNorm(residual + dropout(x))
  residual = x
  x = Linear(d_model -> encoder_ffn_dim)
  x = activation(x)
  x = Linear(encoder_ffn_dim -> d_model)
  x = LayerNorm(residual + dropout(x))
```

Decoder, repeated `decoder_layers`:

```text
y = Linear(feature_size -> d_model, bias=False)(decoder_inputs)
y = LayerNorm(y + sinusoidal_pos(context_length offset))
for layer:
  residual = y
  y = MHA_self(y, causal_mask, optional self cache)
  y = LayerNorm(residual + dropout(y))
  residual = y
  y = MHA_cross(query=y, key/value=encoder_hidden, optional cross cache)
  y = LayerNorm(residual + dropout(y))
  residual = y
  y = Linear(d_model -> decoder_ffn_dim)
  y = activation(y)
  y = Linear(decoder_ffn_dim -> d_model)
  y = LayerNorm(residual + dropout(y))
```

Prediction head:

```text
student_t: 3 independent Linear(d_model -> input_size) heads for df, loc, scale
normal: 2 independent Linear(d_model -> input_size) heads for loc, scale
negative_binomial: 2 independent Linear(d_model -> input_size) heads for total_count, logits
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention.
- MHA only; no GQA/MQA.
- Shape: `query/key/value [B, encoder_heads, context_length, d_model / encoder_heads]`.
- Additive bidirectional mask from `create_bidirectional_mask`.

Decoder self-attention:

- Causal self-attention.
- MHA only.
- Teacher forcing length is `prediction_length` when future values are supplied.
- Source forecast loop length grows from 1 to `prediction_length`; it recomputes all prefix rows per step.
- Optional self-attention cache shape follows Transformers `DynamicCache` per layer: keys/values `[B, decoder_heads, T, head_dim]`.

Decoder cross-attention:

- Query length is decoder prefix/horizon length.
- Key/value length is `context_length`.
- MHA with decoder head count and `head_dim=d_model/decoder_attention_heads`.
- Optional cross-attention cache can store projected encoder keys/values, but the official forecast loop does not pass a cache.

Backend compatibility:

- SDPA/FlashAttention compatibility is mediated by `ALL_ATTENTION_FUNCTIONS` and mask utilities. For DinoML first parity, eager dense attention is sufficient.
- No packed/varlen, sliding-window, local, RoPE, relative bias, ALiBi, or KV-head repeat support is source-required.

## 7. Position encoding and custom math

There are two position-like sources:

- Caller-provided `past_time_features` and `future_time_features`, concatenated into the network input. These can include calendar Fourier features, age features, holidays, or dynamic real covariates known at prediction time.
- Internal fixed sinusoidal embeddings over projected feature rows. The implementation uses sin values in the first half and cos values in the second half rather than interleaving.

Custom math needed for parity:

```python
def squareplus(x):
    return (x + sqrt(x * x + 4.0)) / 2.0

def student_t_domain(df, loc, scale, eps):
    scale = clamp_min(squareplus(scale), eps)
    df = 2.0 + squareplus(df)
    return squeeze_last(df), squeeze_last(loc), squeeze_last(scale)
```

Mean scaling:

```python
ts_sum = sum(abs(data * observed), dim=time, keepdim=True)
num_observed = sum(observed, dim=time, keepdim=True)
scale = ts_sum / clamp(num_observed, min=1)
default_scale = batch fallback when num_observed == 0
scale = clamp(where(num_observed > 0, scale, default_scale), min=minimum_scale)
```

Std scaling:

```python
denom = clamp_min(sum(observed, dim=time, keepdim=True), 1)
loc = sum(data * observed, dim=time, keepdim=True) / denom
scale = sqrt(sum(((data - loc) * observed) ** 2, dim=time, keepdim=True) / denom + minimum_scale)
```

Precomputable:

- Sinusoidal table for `context_length + prediction_length`.
- Static categorical embeddings are normal weights.

Dynamic per request:

- Scaling `loc/scale`.
- Lag windows.
- Time features, because caller supplies them and they may vary by forecast start.

## 8. Preprocessing and input packing

No tokenizer, image processor, or built-in feature extractor exists for this family. Runtime inputs are:

| Input | Shape |
| --- | --- |
| `past_values` | `[B, context_length + max(lags)]` or `[B, context_length + max(lags), input_size]` |
| `past_time_features` | `[B, context_length + max(lags), num_time_features + num_dynamic_real_features]` |
| `past_observed_mask` | same as `past_values` or broadcast-compatible target shape |
| `static_categorical_features` | `[B, num_static_categorical_features]`, integer IDs |
| `static_real_features` | `[B, num_static_real_features]` |
| `future_values` | training/teacher forcing only, `[B, prediction_length]` or `[B, prediction_length, input_size]` |
| `future_time_features` | `[B, prediction_length, num_time_features + num_dynamic_real_features]` |
| `future_observed_mask` | loss mask, same target shape as `future_values` |

CPU/data-pipeline ownership:

- Missing values must be filled before model invocation.
- Observed masks must mark originally observed values.
- Calendar, age, holiday, and dynamic real covariates are caller-produced.
- Windowing must supply sufficient history for the largest lag.

GPU/runtime ownership for first DinoML integration:

- Scale computation from masks.
- Lagged subsequence extraction.
- Feature concatenation and static feature broadcast.

## 9. Graph rewrite / lowering opportunities

### Rewrite: static lag stack -> gather/copy kernel

Source pattern:

```text
for lag in lags_sequence:
  sequence[:, -lag - S : -lag]
stack(..., dim=-1)
reshape(B, S, input_size * num_lags)
```

Replacement:

```text
LagGather(sequence, lags_sequence, subsequences_length=S, shift) -> [B,S,input_size*num_lags]
```

Preconditions:

- `lags_sequence` is compile-time constant.
- `S` is known for teacher forcing or forecast step.
- Dense `[B,T,input_size]` or rank-2 univariate input normalized to rank-3 internally.
- `max(lag - shift) + S <= sequence_length`.

Failure cases:

- Dynamic lag list.
- Insufficient past length.
- Non-dense or irregularly-strided sequence.

Parity test sketch:

- Reproduce `test_create_network_inputs` lag assertions for univariate and multivariate shapes.

### Rewrite: feature packing -> fused pack kernel

Source pattern:

```text
scale context -> log features -> static broadcast -> time concat -> lag concat
```

Replacement:

```text
ScaleAndPack(past_values, observed_mask, static features, time features) -> transformer_inputs, loc, scale, static_feat
```

Preconditions:

- Fixed config feature layout.
- Scaling mode is one of source-supported `mean`, `std`, or no-op.
- Observed mask dtype/broadcast semantics are normalized.

Failure cases:

- Unknown historical config fields such as custom `scaling_dim`/`minimum_scale` not captured in manifest.
- Missing `future_time_features` for teacher-forced decoder.

Parity test sketch:

- Compare packed `transformer_inputs`, `loc`, and `scale` against PyTorch for each scaling mode and missing-mask corner cases.

### Rewrite: forecast loop prefix recompute -> validated cached decode

Source pattern:

```text
for k:
  decoder(inputs_embeds=prefix[0:k+1], encoder_hidden_states=repeated_enc)
```

Replacement:

```text
decoder_step(last_row, self_kv_cache, cross_kv_cache) after validating parity
```

Preconditions:

- Decoder input for step `k` can be produced as only the newest lag/features row.
- Sinusoidal decoder position offset equals `context_length + k`.
- Cross-attention projected K/V cache is identical to source recomputation.
- Random sampling uses a controlled RNG contract.

Failure cases:

- Source parity requires exact stochastic sample stream and cache changes alter call order/RNG order.
- Distribution sampling remains host/PyTorch-owned.

Parity test sketch:

- Force deterministic distribution parameters or fixed RNG seed, compare cached and source recompute step by step.

## 10. Kernel fusion candidates

Highest priority:

- Scale + observed-mask reductions: on every request, feeds all downstream values; needs correct missing-data behavior.
- Lag gather + feature packing: large lags and multivariate `input_size` can dominate preprocessing and memory traffic.
- Dense MHA/SDPA for encoder, decoder self-attention, and cross-attention: ordinary Transformer bottleneck.
- Distribution parameter projection: especially multivariate `input_size`, where three large heads are emitted for Student-T.

Medium priority:

- LayerNorm + residual regions in encoder/decoder blocks.
- FFN `Linear -> GELU -> Linear` for larger configs such as EEG.
- Forecast-loop cached decoder step, once stochastic parity and source-loop equivalence are proven.
- Repeat/expand elimination for `num_parallel_samples` by treating samples as a logical batch dimension.

Lower priority:

- Negative-binomial exact sampling kernels; uncommon in swept configs.
- Training-only dropout/layerdrop/loss acceleration.
- Alternative attention backends beyond dense SDPA; no source-required sparse pattern.

## 11. Runtime staging plan

Stage 1: config and ABI parser.

- Parse `prediction_length`, `context_length`, `input_size`, lags, static feature metadata, scaling, d_model/layers/heads, and distribution output.
- Reject unknown distribution outputs/losses and configs missing `prediction_length`.

Stage 2: source-parity packer.

- Implement mean/std/no-op scaling, static categorical embeddings, static real concatenation, time feature slicing, and lag packing.
- Validate against Transformers `create_network_inputs`.

Stage 3: teacher-forced encoder-decoder parity.

- Run `TimeSeriesTransformerModel` forward with supplied `future_values` and `future_time_features`.
- Implement encoder and decoder dense attention without relying on cache.

Stage 4: prediction head parity.

- Emit Student-T first because all swept configs use it.
- Add Normal and NegativeBinomial as optional branches.
- For inference, expose distribution parameters before sampling as a deterministic validation target.

Stage 5: forecast generation parity.

- Implement the source recompute loop first: repeat batch by `num_parallel_samples`, grow normalized history, sample per step.
- Add deterministic RNG controls for parity tests.

Stage 6: optimized forecast decode.

- Introduce validated decoder self/cross cache only after source-loop parity exists.
- Consider logical sample batching to reduce materialized repeats.

Stage 7: performance fusions.

- Fuse scale/lag/pack, attention blocks, FFNs, and distribution heads.

## 12. Parity and validation plan

- Config tests: effective `feature_size`, `_past_length`, target event shape, and distribution head arity.
- Scaler tests: mean/std/no-op with all observed, partially missing, and all missing context.
- Lag tests: reproduce source `test_create_network_inputs` parameterized lag cases and add multivariate `input_size > 1`.
- Packer tests: compare `transformer_inputs`, `loc`, `scale`, and `static_feat`.
- Single block parity: one encoder layer and one decoder layer with dropout disabled.
- Full model teacher-forcing parity: compare `last_hidden_state` and `encoder_last_hidden_state`.
- Head parity: compare Student-T `df/loc/scale` tensors after domain map; add Normal and NegativeBinomial synthetic configs.
- Sampling parity: initially compare output shapes and deterministic distribution params; exact sample values need RNG agreement.
- Official checkpoint smoke: `huggingface/time-series-transformer-tourism-monthly` expected encoder shape `[64, context_length, d_model]` and generation shape `[64, num_parallel_samples, prediction_length]`, matching source tests.
- Tolerances: fp32 `1e-4` for hidden-state slices as in Transformers integration tests; fp16/bf16 should use looser attention/head tolerances after separate calibration.

## 13. Performance probes

- Feature pack throughput versus batch, `context_length`, `max(lag)`, `len(lags_sequence)`, and `input_size`.
- Encoder throughput over context length.
- Teacher-forced decoder throughput over prediction length.
- Source forecast loop tokens/sec equivalent: forecast steps/sec with prefix recompute.
- Optimized cached forecast loop after validation: compare against source recompute.
- `num_parallel_samples` sweep: 1, 10, 100, and production-requested values.
- Multivariate head sweep: `input_size=1`, 22, 862.
- Lag pattern sweep: short dense lags `[1..7]`, hourly 40-lag seasonal list, sparse multivariate `[1,168]`.
- Distribution head comparison: Student-T versus Normal versus NegativeBinomial.
- Memory probes: repeated encoder hidden states and repeated history from `num_parallel_samples`.
- Attention backend comparison: eager dense, SDPA, and DinoML fused attention.

## 14. Skip/defer list

- Training loss and backprop as a first runtime target; keep NLL only for validation if needed.
- Dropout and layerdrop in inference.
- Beam search/text generation APIs; not applicable.
- Processor ownership for calendar/holiday feature engineering; caller/data pipeline first.
- Exact stochastic sample parity until RNG contract is specified; deterministic parameter parity first.
- Decoder cache optimization until source recompute generation is validated.
- NegativeBinomial and Normal heads can follow Student-T unless a target checkpoint needs them.
- Quantization, tensor parallelism, and distributed serving.

## 15. Final implementation checklist

- [ ] Parse `TimeSeriesTransformerConfig` including effective defaults and `feature_size`.
- [ ] Load value embedding, encoder/decoder, static categorical embeddings, and distribution projection weights.
- [ ] Implement observed-mask mean scaler.
- [ ] Implement observed-mask std scaler.
- [ ] Implement no-op scaler.
- [ ] Implement lag gather/pack with `shift=0` and generation `shift=1`.
- [ ] Implement static feature construction: embeddings, real features, `log1p(abs(loc))`, `log(scale)`.
- [ ] Implement encoder dense self-attention and FFN blocks.
- [ ] Implement decoder causal self-attention, cross-attention, and FFN blocks.
- [ ] Implement sinusoidal position table with decoder `context_length` offset.
- [ ] Implement Student-T parameter projection and domain map.
- [ ] Add optional Normal and NegativeBinomial distribution heads.
- [ ] Expose deterministic distribution parameters for forecast validation.
- [ ] Implement source-parity autoregressive sampling loop.
- [ ] Add packer parity tests against Transformers.
- [ ] Add full teacher-forcing parity on random configs.
- [ ] Add official tourism checkpoint smoke.
- [ ] Benchmark scale/lag/pack, encoder, decoder, generation loop, and `num_parallel_samples` sweeps.
