# Informer audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: informer
Primary runtime target: probabilistic time-series forecasting with InformerForPrediction.generate()
Config source: local configuration class plus public HF config.json files listed below
Source files inspected:
- transformers/src/transformers/models/informer/configuration_informer.py
- transformers/src/transformers/models/informer/modeling_informer.py
- transformers/src/transformers/models/informer/modular_informer.py
- transformers/src/transformers/time_series_utils.py
- transformers/tests/models/informer/test_modeling_informer.py
Snapshots:
- agents/plans/transformers/informer/_sources/
Missing files or assumptions: no processor class; time features and scaling inputs are caller/data-pipeline owned.
```

`modeling_informer.py` is generated from `modular_informer.py`; future source edits should target the modular file, but this audit treats generated `modeling_informer.py` as the import/runtime source. Public configs inspected:

- [huggingface/informer-tourism-monthly](https://huggingface.co/huggingface/informer-tourism-monthly/blob/main/config.json)
- [kashif/informer-traffic-hourly](https://huggingface.co/kashif/informer-traffic-hourly/blob/main/config.json)
- [kashif/informer-mv-traffic-hourly](https://huggingface.co/kashif/informer-mv-traffic-hourly/blob/main/config.json)
- [JLB-JLB/EEG_Informer_336_history_96_horizon](https://huggingface.co/JLB-JLB/EEG_Informer_336_history_96_horizon/blob/main/config.json)
- [shaddie/rocketpill_ts_informer_model](https://huggingface.co/shaddie/rocketpill_ts_informer_model/blob/main/config.json)
- [shaddie/rocketpill_thrustcurve_informer_model](https://huggingface.co/shaddie/rocketpill_thrustcurve_informer_model/blob/main/config.json)

Gated/unavailable gap: [hf-internal-testing/tiny-random-InformerModel](https://huggingface.co/hf-internal-testing/tiny-random-InformerModel) returned 401 for `config.json`; the test fixture in `test_modeling_informer.py` supplies the small/debug shape basis instead.

## 2. High-level architecture

Informer is an encoder-decoder time-series model, not a text LM. It consumes historical values, observed-value masks, time features, optional static real/categorical features, and optional future values for teacher-forced training.

```text
past/future time features + scaled target lags + static features
-> value projection + sinusoidal positions
-> Informer encoder with optional distilling Conv1d/BatchNorm/ELU/MaxPool
-> decoder self-attention + encoder cross-attention
-> distribution parameter projection
-> sampling loop for prediction_length future values
```

Stage decomposition:

- CPU/data pipeline: calendar/age/dynamic real feature construction, missing-value imputation to zero plus observed masks, batching variable series into fixed context/horizon.
- Runtime pre-transform: scaling from observed context, static feature assembly, lagged subsequence gather, concatenation.
- Encoder: cacheable per past window; distillation can shrink encoder length by repeated stride-2 pooling.
- Decoder/training: teacher-forced future values produce the full horizon in one forward pass.
- Decoder/generation: source implementation loops over horizon and samples from a distribution each step; this is stochastic and RNG-sensitive.

## 3. Important config dimensions

| Field | Source default | Operator effect |
| --- | ---: | --- |
| `prediction_length` | required | decoder horizon and sampling loop trip count |
| `context_length` | `prediction_length` | encoder input length before distillation |
| `input_size` | 1 | target variates; distribution event shape is `()` for 1 else `(input_size,)` |
| `lags_sequence` | `[1..7]` | lag gather count and required `past_values` length |
| `feature_size` | derived | `input_size * len(lags) + static/dynamic/time/scaler features` |
| `d_model` | 64 | transformer hidden width |
| `encoder_layers` / `decoder_layers` | 2 / 2 | repeated blocks; distil creates `encoder_layers - 1` conv layers |
| `encoder_attention_heads` / `decoder_attention_heads` | 2 / 2 | MHA heads; requires `d_model % heads == 0` |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 32 / 32 | FFN widths |
| `attention_type` | `prob` | `prob` uses custom ProbSparse; any other string currently selects full attention |
| `sampling_factor` | 5 | ProbSparse `u_part` and `u` sizes |
| `distil` | true | inserts circular Conv1d + BatchNorm1d + ELU + MaxPool1d after encoder layers except last |
| `scaling` | `mean` | mean/std/no-op target normalization |
| `distribution_output` | `student_t` | parameter head and sampling distribution |
| `num_parallel_samples` | 100 | batch expansion during generation |

Representative checkpoint sweep:

| Model/config | Context | Horizon | `input_size` | `d_model` | Enc/Dec layers | Heads | Lags | Dist | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| test fixture | 14 | 7 | 1 | 16 | 2/2 | 4/4 | 5 | false | small/debug, `sampling_factor=10` |
| `huggingface/informer-tourism-monthly` | 24 | 24 | 1 | 32 | 4/4 | 2/2 | 16 | true | one static categorical embedding |
| `kashif/informer-traffic-hourly` | 24 | 24 | 1 | 16 | 2/2 | 2/2 | 40 | true | long hourly lag set up to 721 |
| `kashif/informer-mv-traffic-hourly` | 48 | 24 | 862 | 64 | 2/2 | 2/2 | 2 | true | very wide multivariate target; `feature_size=3453` |
| `JLB-JLB/EEG...` | 336 | 96 | 22 | 512 | 4/2 | 16/8 | 20 | true | config has `attention_type="prop"` typo, which routes to full attention in current source |
| `shaddie/rocketpill_ts...` | 16 | 8 | 266 | 64 | 6/4 | 2/2 | 2 | true | wide target, short context |

## 3a. Family variation traps

- `attention_type` is not strictly validated in source. Only exact `"prob"` uses ProbSparse; a typo such as `"prop"` silently selects full attention.
- Historical configs may include `factor` or `attention_factor`; current source reads `sampling_factor`, so those fields are ignored unless duplicated.
- `feature_size` is derived by the config class, but several configs serialize it. DinoML should recompute or validate it against `input_size * len(lags_sequence) + _number_of_features`.
- Distillation changes encoder sequence length and attention output shapes. For `encoder_layers=4`, distillation pools after three layers, e.g. context 24 becomes 3 in integration tests.
- ProbSparse attention uses `torch.randint` inside forward, so batching equivalence and deterministic replay need RNG control or full-attention admission.
- Multivariate `input_size` changes distribution event shape, loss weighting, parameter projection widths, and output sequence rank.
- Time features are positional inputs. There is no learned positional index ABI for calendar semantics.
- Source tensor layout is `[batch, time, feature]`, except distilling conv temporarily permutes to `[batch, channel, time]`; layout translation should guard this local region.
- `use_cache` exists through decoder cache machinery, but `InformerForPrediction.generate()` does not pass a cache between loop iterations; it re-runs the decoder on the growing prefix.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank normalization for univariate `[B,T]` versus multivariate `[B,T,C]` target tensors.
- Slice/gather windows from trailing time axis for context and lags.
- `stack` lag slices, reshape `[B,S,C,num_lags] -> [B,S,C*num_lags]`.
- `cat`, `unsqueeze`, `expand`, `repeat_interleave`, `reshape`, `transpose`, `permute`, `contiguous`, indexed scatter/update for ProbSparse context.
- Top-k indices and advanced indexing for ProbSparse query selection.

Neural primitives:

- Linear projections: value embedding `feature_size -> d_model`; Q/K/V/O `d_model -> d_model`; FFN `d_model -> ffn_dim -> d_model`; distribution projection `d_model -> args_dim`.
- LayerNorm, BatchNorm1d, Dropout as inference no-op, GELU by default, ELU for distilling conv.
- Conv1d circular padding, kernel 3, channels `d_model -> d_model`; MaxPool1d kernel 3 stride 2 padding 1.

Attention primitives:

- Full MHA self-attention and encoder-decoder cross-attention.
- ProbSparse self-attention for encoder and decoder when `attention_type="prob"`.
- Causal mask for decoder self-attention; bidirectional masks for encoder and cross-attention.

Time-series/probabilistic ops:

- Mean/std/no-op scaling with observed masks.
- `log1p(abs(loc))`, `log(scale)` static scaler features.
- Embedding lookup for optional static categorical features.
- StudentT, Normal, and NegativeBinomial parameter domain maps and sampling/log-prob.
- Weighted average loss for training parity.

## 5. Layer/block breakdown

Network input construction:

```text
context = last context_length past_values
loc, scale = scaler(context, observed_context)
inputs = (past[,future] - loc) / scale
static_feat = [optional cat embedding, optional static real, log1p(abs(loc)), log(scale)]
features = expand(static_feat over time) concat time_features
lagged = stack(sequence[:, -lag-subseq_len:-lag] for lag in lags_sequence)
transformer_inputs = concat(flatten(lagged), features)
```

Encoder block, repeated `encoder_layers`:

```text
x = ValueLinear(feature_size -> d_model)(inputs)
x = LayerNorm(x + fixed sinusoidal positions)
for layer i:
  residual = x
  x = ProbSparseSelfAttention(x) or FullSelfAttention(x)
  x = LayerNorm(residual + x)
  residual = x
  x = Linear(d_model -> encoder_ffn_dim) -> activation -> Linear(encoder_ffn_dim -> d_model)
  x = LayerNorm(residual + x)
  if distil and i < encoder_layers - 1:
    x = Conv1dCircular(k=3) -> BatchNorm1d -> ELU -> MaxPool1d(k=3,s=2,p=1)
```

Decoder block, repeated `decoder_layers`:

```text
x = ValueLinear(feature_size -> d_model)(decoder_inputs)
x = LayerNorm(x + fixed positions offset by context_length)
for layer:
  x = LayerNorm(x + self_attention(x, causal_mask))
  x = LayerNorm(x + cross_attention(query=x, key/value=encoder_hidden))
  x = LayerNorm(x + FFN(x))
```

Distribution head:

```text
StudentT: Linear(d_model -> input_size) x 3 for df, loc, scale
Normal:   Linear(d_model -> input_size) x 2 for loc, scale
NegBin:   Linear(d_model -> input_size) x 2 for total_count, logits
```

All attention and FFN Linear layers use bias in the generated source.

## 6. Attention requirements

Full attention path:

- Encoder self-attention is bidirectional.
- Decoder self-attention is causal.
- Decoder cross-attention attends decoder query length to encoder output length.
- MHA only; no GQA/MQA. `head_dim = d_model // num_heads`.
- Full attention can dispatch through Transformers attention implementations for `InformerAttention`.

ProbSparse path:

- Q/K/V are projected to `[B * H, T, D]`.
- `u_part = min(factor * L_Q * ceil(log1p(L_K)), L_K)`.
- `u = min(factor * ceil(log1p(L_Q)), L_Q)`.
- Randomly samples `u_part` key positions with `torch.randint`, computes sampled QK scores, then selects top-`u` queries by `max(score) - mean(score)`.
- Computes dense attention only for selected queries, then writes selected outputs into an initial context.
- Encoder initial context is the mean of values expanded to all query positions.
- Decoder initial context is cumulative sum of values, cast through fp32 to avoid overflow.
- Attention masks are expected as additive masks of shape `[B,1,L_Q,L_K]` before expansion and top-query slicing.
- Returned attention weights for ProbSparse have shape `[B,H,u,L_K]`, not dense `[B,H,L_Q,L_K]`.

Cache distinction:

- Decoder layers accept `EncoderDecoderCache`, and cross-attention can cache K/V.
- The forecasting `generate()` loop currently does not feed `past_key_values` into successive decoder calls, so first integration can defer optimized cache reuse for source parity.

## 7. Position encoding and custom math

Informer uses fixed sinusoidal embeddings over `context_length + prediction_length`. Time-series calendar/age features are separate runtime inputs and are the semantic positional signal.

Concise custom snippets:

```python
def prob_sparse_sizes(L_q, L_k, factor):
    u_part = min(factor * L_q * ceil(log1p(L_k)), L_k)
    u = min(factor * ceil(log1p(L_q)), L_q)
    return u_part, u
```

```python
def lagged_subsequences(sequence, lags, S, shift=0):
    out = []
    for lag in [x - shift for x in lags]:
        begin = -lag - S
        end = -lag if lag > 0 else None
        out.append(sequence[:, begin:end, ...])
    return stack(out, dim=-1)
```

```python
def student_t_domain(df, loc, scale):
    scale = squareplus(scale).clamp_min(eps(scale.dtype))
    df = 2.0 + squareplus(df)
    return squeeze_last(df), squeeze_last(loc), squeeze_last(scale)
```

## 8. Preprocessing and input packing

No tokenizer or processor is implemented. Caller must supply:

- `past_values`: `[B, context_length + max(lags)]` or `[B, ..., input_size]`.
- `past_observed_mask`: same shape as `past_values`, optional; missing values should already be replaced by zero.
- `past_time_features`: `[B, past_length, num_time_features + num_dynamic_real_features]`.
- `future_time_features`: `[B, prediction_length, num_time_features + num_dynamic_real_features]`.
- Optional `static_categorical_features`: `[B, num_static_categorical_features]`, integer IDs.
- Optional `static_real_features`: `[B, num_static_real_features]`.
- Optional training `future_values` and `future_observed_mask`.

GPU graph ownership should start after these tensors are materialized. Calendar feature generation, frequency-specific lag selection, imputation, and batching are data-pipeline work.

## 9. Graph rewrite / lowering opportunities

### Rewrite: value projection to GEMM

Source pattern: `Linear(feature_size -> d_model, bias=False)` on `[B,T,F]`.

Replacement: flatten `[B*T,F] -> GEMM_RCR(weight [d_model,F]) -> reshape [B,T,d_model]`.

Preconditions: contiguous or stride-compatible `[B,T,F]`; static `feature_size`; no bias.

Parity test: compare encoder input embedding before positional add for univariate and multivariate configs.

### Rewrite: distilling Conv1d region

Source pattern: `permute(B,T,C)->Conv1d(circular,k=3,p=1)->BatchNorm1d->ELU->MaxPool1d->transpose`.

Replacement: preserve source NCT layout locally, or use a fused time-major kernel that implements circular left/right wrap and pool length exactly.

Preconditions: inference BatchNorm folded into Conv1d; channel count equals `d_model`; axis rewrite must keep time as the pooled dimension.

Failure cases: training mode, non-circular padding, dynamic time lengths without matching PyTorch pooling length.

### Rewrite: full attention to standard fused attention

Source pattern: Q/K/V Linears, reshape to `[B,H,T,D]`, additive mask, softmax, dropout, V matmul, output Linear.

Replacement: fused MHA/FlashAttention for full attention only.

Preconditions: `attention_type != "prob"`; inference dropout 0; additive mask semantics preserved; cross-attention rectangular shapes supported.

Failure cases: ProbSparse path, requested attention weight parity, non-eager backend drift.

### Rewrite: ProbSparse admission to deterministic full attention fallback

Source pattern: random key sampling + top-k query reduction.

Replacement: for first parity, reject/route ProbSparse to eager source or require fixed RNG and implement the sparse algorithm; optionally admit `attention_type != "prob"` full-attention configs first.

Preconditions: user accepts full-attention model/config or deterministic seed plumbing is defined.

Parity test: seeded single-layer ProbSparse against PyTorch with fixed `torch.manual_seed`.

## 10. Kernel fusion candidates

Highest priority:

- Network input assembly: scaling + static scaler features + lag gather + concatenate is a major non-transformer cost and is shape-sensitive.
- Full-attention MHA and cross-attention for `attention_type` typo/full configs.
- Distribution head plus domain map for StudentT/Normal/NegativeBinomial.

Medium priority:

- ProbSparse top-k/query-gather/update kernel family, especially for long contexts.
- Distilling Conv1d/BatchNorm/ELU/MaxPool inference fusion.
- FFN Linear + GELU + Linear with residual LayerNorm around it.

Lower priority:

- Decoder KV cache optimization for generation; source generate does not currently exploit it.
- Attention-output materialization for `output_attentions=True`; can be deferred for fast inference.

## 11. Runtime staging plan

1. Parse config and validate `feature_size`, lags, heads, distil, distribution output.
2. Load weights for `InformerModel` and run encoder/decoder training-style forward with `future_values`.
3. Implement network input assembly and one encoder/decoder block parity on small test fixture.
4. Support `attention_type!="prob"` full attention and cross-attention first.
5. Add distillation conv/pool and validate encoder length changes.
6. Add distribution parameter projections and deterministic distribution means/params parity.
7. Add source-style stochastic `generate()` sampling loop with explicit RNG contract.
8. Implement or route ProbSparse; require deterministic seed for parity tests.
9. Optimize lag assembly, distillation, attention, and sampling batch expansion.

Initially stubbable: training loss, `output_attentions`, LayerDrop/dropout training paths, cache reuse, non-StudentT distributions if configs in scope use StudentT only.

## 12. Parity and validation plan

- Config validation tests for default, tourism, traffic, multivariate, EEG typo/full-attention, and rocketpill configs.
- Lag gather tests over univariate and multivariate tensors with nontrivial `shift=1`.
- Scaler tests for mean/std/no-op with observed masks including all-missing rows.
- Single ProbSparse attention parity with fixed torch seed; include attention mask and decoder cumsum context.
- Full attention parity against source for encoder self, decoder self, and cross-attention.
- Distillation length parity for context lengths 16, 24, 48, 336.
- Distribution domain-map tests for StudentT/Normal/NegativeBinomial, including multivariate event shape.
- End-to-end `InformerForPrediction.forward()` on the test fixture and public tourism batch shape.
- Generation shape tests: `[B,num_parallel_samples,prediction_length]` and `[B,num_parallel_samples,prediction_length,input_size]`.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4` for deterministic paths; stochastic sampling should compare seeded samples or distribution parameters, not sample means, except with loose aggregate tolerances.

## 13. Performance probes

- Network-input assembly throughput versus context, lag count, and `input_size`.
- ProbSparse pipeline timing split: random index generation, sampled QK BMM, top-k, selected attention BMM, context update.
- Full attention versus ProbSparse for `L={24,48,336}` and head widths from configs.
- Distillation conv/pool throughput and encoder length shrink verification.
- Distribution sampling throughput for `num_parallel_samples={1,10,100}` and multivariate widths `{1,22,266,862}`.
- End-to-end generation loop timing by horizon, with and without decoder cache optimization.
- Memory probe for repeated batch expansion: effective batch `B * num_parallel_samples`.

## 14. Skip/defer list

- Training loss/backprop and LayerDrop/dropout behavior.
- `output_attentions=True` dense reconstruction for ProbSparse; source itself returns sparse selected-query weights.
- Optimized decoder KV-cache reuse in generate.
- Unsupported remote-code or non-`model_type=informer` repos returned by search, such as Llama text models named "informer".
- Processor/calendar feature generation.
- Quantized weights and multi-GPU tensor parallel.

## 15. Final implementation checklist

- [ ] Parse `InformerConfig` and reject invalid heads/lags/distributions.
- [ ] Recompute and validate `feature_size`.
- [ ] Load `InformerModel`/`InformerForPrediction` weights.
- [ ] Implement mean/std/no-op scalers with observed masks.
- [ ] Implement static feature assembly and categorical embeddings.
- [ ] Implement lagged subsequence gather and decoder `shift=1` path.
- [ ] Implement value projection, sinusoidal positions, LayerNorm, FFN.
- [ ] Implement full encoder/decoder/cross attention.
- [ ] Implement distilling Conv1d/BatchNorm/ELU/MaxPool.
- [ ] Decide ProbSparse policy: implement seeded sparse path or route/reject.
- [ ] Implement distribution projections and StudentT/Normal/NegativeBinomial domain maps.
- [ ] Implement source-style sampling loop with explicit RNG behavior.
- [ ] Add parity tests for config sweep and one-block/full-model paths.
- [ ] Add performance probes for lag assembly, attention, distillation, and sampling.
