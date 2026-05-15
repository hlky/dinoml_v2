# PatchTSMixer Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: primary representative `ibm/patchtsmixer-etth1-forecasting`; additional inspected configs from `ibm-research/patchtsmixer-etth1-pretrain`, `ibm/patchtsmixer-etth1-generate`, `ibm-granite/granite-timeseries-patchtsmixer`, and `sujayC66/PatchTSMixer_Stock_Data`.

Config source: official Hugging Face `config.json` URLs were accessible for all listed checkpoints. Snapshots are in `_sources/*.config.json`.

Source files inspected:

- `transformers/src/transformers/models/patchtsmixer/configuration_patchtsmixer.py`
- `transformers/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py`
- `transformers/tests/models/patchtsmixer/test_modeling_patchtsmixer.py`
- `transformers/src/transformers/time_series_utils.py` for distribution projection/domain-map behavior.
- `transformers/docs/source/en/model_doc/patchtsmixer.md` for task scope only; operator facts below are source-derived.

Snapshots created:

- `_sources/configuration_patchtsmixer.py`
- `_sources/modeling_patchtsmixer.py`
- `_sources/test_modeling_patchtsmixer.py`
- `_sources/ibm-research__patchtsmixer-etth1-pretrain.config.json`
- `_sources/ibm__patchtsmixer-etth1-forecasting.config.json`
- `_sources/ibm__patchtsmixer-etth1-generate.config.json`
- `_sources/ibm-granite__granite-timeseries-patchtsmixer.config.json`
- `_sources/sujayC66__PatchTSMixer_Stock_Data.config.json`

Any missing files or assumptions: no gated, 401, 403, or 404 configs were encountered for the selected representative checkpoints. The report targets inference parity for deterministic forecasting first. Pretraining masks, training losses, and stochastic sampling are documented as optional or deferred unless needed by a chosen deployment.

## 2. High-level architecture

PatchTSMixer is a time-series MLP-Mixer family, not an autoregressive text decoder. The core runtime contract consumes dense time-series values shaped `[batch, context_length, num_input_channels]`, scales them per channel, extracts temporal patches, projects each patch to `d_model`, applies repeated patch/feature mixer blocks, and then applies a task head.

Dataflow:

```text
past_values + observed_mask
  -> per-channel scaler
  -> temporal patchify/unfold
  -> patch Linear(patch_length -> d_model)
  -> optional positional encoding
  -> repeated mixer layers
  -> forecast/pretrain/classification/regression head
  -> optional inverse scaling or distribution construction
```

Stage decomposition:

- CPU/data-pipeline: assemble `past_values`, replace missing values as zeros, provide `observed_mask` when missingness matters.
- Runtime preprocessing: scaler over time dimension, fixed patch extraction, optional pretraining mask.
- Encoder: patch projection plus `num_layers` mixer layers.
- Heads: independently stageable deterministic forecast, masked pretraining reconstruction, classification, regression, and distributional forecast/regression.
- No KV cache, generation decode loop, tokenizer, attention mask packing, or text sampling controller exists. `generate()` only samples a PyTorch distribution from the forecast/regression head outputs.

## 3. Important config dimensions

Source defaults from `PatchTSMixerConfig`:

| Field | Default | Runtime effect |
| --- | ---: | --- |
| `context_length` | 32 | required input time length |
| `patch_length` | 8 | temporal window size per patch |
| `patch_stride` | 8 | temporal patch step; overlap if smaller than patch length |
| `num_patches` | derived | `(max(context_length, patch_length) - patch_length) // patch_stride + 1` |
| `num_input_channels` | 1 | time-series variables/channels |
| `d_model` | 8 | hidden width per patch |
| `expansion_factor` | 2 | MLP hidden factor for all mixer MLPs |
| `num_layers` | 3 | repeated mixer layers |
| `mode` | `common_channel` | no explicit channel mixer unless `mix_channel` |
| `gated_attn` | true | gated Linear+Softmax+multiply in mixer axes |
| `norm_mlp` | `LayerNorm` | `LayerNorm(d_model)` or `BatchNorm1d(d_model)` branch |
| `self_attn` | false | optional tiny self-attention across patches per channel |
| `self_attn_heads` | 1 | head count for optional patch attention |
| `use_positional_encoding` | false | optional `[num_patches, d_model]` add before mixer |
| `scaling` | `std` | `std`, `mean`, `True` as std in current code, or no-op |
| `loss` | `mse` | deterministic point head or distributional `nll` |
| `prediction_length` | 16 | forecast horizon |
| `prediction_channel_indices` | null | optional forecast channel slice |
| `num_targets` | 3 | classification label count or regression target count |
| `head_aggregation` | `max_pool` | classification/regression aggregation over patches |

Representative checkpoint sweep:

| Model | Architecture | context | patch/stride | channels | patches | d_model | layers | mode | loss | horizon/targets | scaling |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `ibm-research/patchtsmixer-etth1-pretrain` | `PatchTSMixerForPretraining` | 512 | 16/16 | 7 | 32 | 48 | 2 | common | mse | pretrain reconstruct patch len 16 | true |
| `ibm/patchtsmixer-etth1-forecasting` | `PatchTSMixerForPrediction` | 512 | 16/16 | 7 | 32 | 48 | 2 | common | mse | prediction 96 | true |
| `ibm/patchtsmixer-etth1-generate` | `PatchTSMixerForPrediction` | 512 | 16/16 | 7 | 32 | 48 | 2 | common | nll/student_t | prediction 96, samples 1 | true |
| `ibm-granite/granite-timeseries-patchtsmixer` | `PatchTSMixerForPrediction` | 512 | 16/16 | 7 | 32 | 48 | 2 | common | mse | prediction 96 | true |
| `sujayC66/PatchTSMixer_Stock_Data` | `PatchTSMixerForPrediction` | 140 | 8/8 | 1 | 17 | 48 | 3 | common | mse | prediction 14 | null |

## 3a. Family variation traps

- `config.scaling=True` is documented as mean scaling in the config docstring, but `PatchTSMixerModel.__init__` treats `True` as the std-scaler branch. DinoML should match source behavior for this commit and record the doc/source mismatch.
- `mode="common_channel"` still has channel behavior through shared weights and per-channel independent processing, but it does not run the explicit channel mixer. `mode="mix_channel"` adds channel-axis MLP/gating over `num_input_channels`.
- `self_attn=True` changes the block topology by adding noncausal self-attention across patches for each channel after reshaping `[B, C, P, D]` to `[B*C, P, D]`.
- `use_positional_encoding=True` only constructs an additive positional table when enabled. Otherwise the positional module is absent in `PatchTSMixerEncoder`, despite `PatchTSMixerPositionalEncoding` being able to hold zeros.
- `norm_mlp` switches between `LayerNorm` on last dim and `BatchNorm1d(d_model)` after flattening batch and channel into one batch dimension.
- `patch_stride < patch_length` creates overlapping patches through `Tensor.unfold`; no checkpoint inspected uses overlap, but source supports it.
- `prediction_channel_indices` sorts the list in-place in the forecast head and slices both prediction and target/scale/loc channels.
- Distributional `loss="nll"` returns tuples of distribution parameters, not one dense point tensor. Forecasting transposes every parameter tensor from `[B, C, prediction_length]` to `[B, prediction_length, C]`.
- Classification/regression heads reuse `num_targets`; classification interprets it as class count and uses cross entropy.
- `head_aggregation=None` flattens `[channels, patches, d_model]`; `use_last`, `max_pool`, and `avg_pool` reduce patch axis first.
- Configs may contain historical keys not declared in the current strict config class, such as `post_init`; loaded configs accepted them in the fetched JSON, but current source behavior should be based on fields read by the model.
- This is time-major source layout `[B, T, C]` at input and `[B, C, P, D]` internally. Treat any channel-last or alternative layout pass as a guarded optimization because `unfold`, reductions, `transpose`, and head flatten order are axis-sensitive.

## 4. Operator coverage checklist

Tensor/layout ops:

- Exact shape validation for `past_values.shape[-2] == context_length`.
- Slice tail window: `past_values[:, sequence_start:, :]` where `sequence_start = context_length - (patch_length + patch_stride * (num_patches - 1))`.
- Temporal `unfold` over axis `-2`, size `patch_length`, step `patch_stride`.
- `transpose`, `permute`, `reshape`, `view`, `contiguous`, `flatten`, `unsqueeze`, `repeat`, `cat`, and optional channel gather/slice.
- `max(dim=-1).values`, `mean(dim=-1)`, `sum(dim=1, keepdim=True)`, `clamp_min`, `where`.

Neural primitives:

- Linear patch projection: `Linear(patch_length -> d_model)`.
- Mixer MLPs with bias: `Linear(axis -> axis * expansion_factor) -> GELU -> dropout -> Linear(axis * expansion_factor -> axis) -> dropout`.
- Gated attention blocks: `Linear(axis -> axis) -> Softmax(dim=-1) -> elementwise multiply`.
- Residual adds after channel, patch, feature, and optional attention-normalization paths.
- `LayerNorm(d_model, eps=norm_eps)` or `BatchNorm1d(d_model, eps=norm_eps)`.
- Optional output-range restriction for regression: `sigmoid(x) * (hi - lo) + lo`.
- Distribution parameter projections from `time_series_utils.ParameterProjection`: separate Linear modules per distribution argument plus domain maps using `squareplus`, `sqrt`, `square`, `clamp_min`, and squeezes.

Attention primitives:

- Optional noncausal MHA over patches only: Q/K/V/O Linear(`d_model -> d_model`), reshape to `[B*C, heads, P, head_dim]`, dense attention scores `[B*C, heads, P, P]`, softmax over keys, and output projection.
- Source routes through `ALL_ATTENTION_FUNCTIONS` using `_attn_implementation`; eager fallback is ordinary scaled dot-product attention. No cache.

Preprocessing-coupled ops:

- Std scaler: weighted mean and variance over time axis using `observed_mask`, then `(data - loc) / sqrt(var + minimum_scale)`.
- Mean scaler: weighted average absolute value over time axis, batch-derived default scale for all-missing channels, clamp minimum, then `data / scale`.
- No-op scaler: returns `data`, zero loc, one scale shaped `[B, 1, C]`.
- Pretraining random/forecast masks use RNG, argsort/gather, randperm, masked_fill, and bool masks.

Task heads:

- Forecast point head: flatten `[P, D]` per channel to `P*D`, Linear(`P*D -> prediction_length`), transpose to `[B, prediction_length, C]`, inverse scale.
- Forecast distribution head: projection from `P*D` to distribution args for each channel/horizon, transpose each arg, construct StudentT/Normal/NegativeBinomial distribution.
- Pretraining head: Linear(`d_model -> patch_length`) per patch/channel.
- Classification/regression head: transpose to make patches last, aggregate or flatten, dropout, Linear(`C*D*mul_factor -> num_targets`), optional distribution output for regression.
- Scale injection for classification/regression when scaling is enabled: repeat loc/scale across patches, two small Linear stacks for stats and hidden+stats fusion.

## 5. Layer/block breakdown

Core model:

```text
past_values: [B, T=context_length, C]
observed_mask: [B, T, C] or ones_like
scaled, loc, scale = scaler(past_values, observed_mask)  # loc/scale [B, 1, C]
patch_input = unfold_time(scaled)                        # [B, C, P, patch_length]
hidden = Linear(patch_length -> D)(patch_input)           # [B, C, P, D]
hidden = optional position_enc[P, D] add
repeat num_layers:
  if mode == "mix_channel":
    residual = hidden
    y = Norm(hidden)
    y = permute(y, [B, D, P, C])
    y = optional Gate(C)(y)
    y = MLP(C -> C * expansion -> C)(y)
    hidden = permute(y, [B, C, P, D]) + residual
  residual = hidden
  y = Norm(hidden)
  if self_attn:
    a = reshape(y, [B*C, P, D])
    a = MHA_noncausal_patches(a)
    a = reshape(a, [B, C, P, D])
  y = transpose(y, patch_dim, feature_dim)                 # [B, C, D, P]
  y = MLP(P -> P * expansion -> P)(y)
  y = optional Gate(P)(y)
  y = transpose(y, [B, C, P, D])
  if self_attn:
    y = Norm(y + a)
  hidden = y + residual
  residual = hidden
  y = Norm(hidden)
  y = MLP(D -> D * expansion -> D)(y)
  y = optional Gate(D)(y)
  hidden = y + residual
```

Forecast head:

```text
h: [B, C, P, D]
h = flatten last two dims -> [B, C, P*D]
point = Linear(P*D -> prediction_length)(h)
point = transpose -> [B, prediction_length, C]
point = optional channel slice
prediction = point * scale + loc  # point-loss mode
```

Pretraining head:

```text
h: [B, C, P, D]
patch_reconstruction = Linear(D -> patch_length)(h)  # [B, C, P, patch_length]
```

Classification/regression head:

```text
h: [B, C, P, D]
if scaling enabled: h = InjectScalerStatistics4D(h, loc, scale)
h = transpose(-1, -2)  # [B, C, D, P]
h = use_last/max_pool/avg_pool over patch axis, or keep all patches
h = flatten channel/feature/(patch) dims
y = Linear(... -> num_targets)(dropout(h))
```

## 6. Attention requirements

Attention is optional and absent in all representative configs inspected. When `self_attn=True`, it is encoder-style, noncausal, self-attention across patch positions for each channel independently.

Required details for the optional path:

- Attention kind: noncausal self-attention, no cross-attention in PatchTSMixer call sites.
- MHA only. No MQA/GQA.
- Head count: `self_attn_heads`; `head_dim = d_model // self_attn_heads`; source rejects non-divisible widths.
- Query/key/value width: all `d_model`.
- Query and key length: `num_patches`.
- Masking: call site passes no attention mask.
- Cache: none.
- Position interaction: optional additive position encoding before mixer stack, not RoPE or relative bias.
- Backend: source uses `ALL_ATTENTION_FUNCTIONS` with eager fallback. Flash/SDPA compatibility is inherited only if the selected attention interface supports the shape and no mask.

For the primary deterministic forecast target, DinoML can reject `self_attn=True` initially or route it to a generic dense attention path.

## 7. Position encoding and custom math

There is no RoPE, ALiBi, learned absolute token table by default, or relative bias. Optional positional encoding is `[num_patches, d_model]` added to `[B, C, P, D]`.

Sincos initialization source behavior:

```python
position = arange(num_patches).unsqueeze(1)
div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)
pe = (pe - pe.mean()) / (pe.std() * 10)
```

Custom scaling math:

```python
den = observed_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
loc = (data * observed_mask).sum(dim=1, keepdim=True) / den
var = (((data - loc) * observed_mask) ** 2).sum(dim=1, keepdim=True) / den
scale = sqrt(var + minimum_scale)
scaled = (data - loc) / scale
```

Distribution domain maps from `time_series_utils`:

```python
squareplus(x) = (x + sqrt(x * x + 4.0)) / 2.0
student_t: scale = squareplus(scale).clamp_min(eps); df = 2.0 + squareplus(df)
normal: scale = squareplus(scale).clamp_min(eps)
negative_binomial: total_count = squareplus(total_count)
```

Position encodings are static after config/weight load. Scaling depends on runtime `past_values` and `observed_mask`.

## 8. Preprocessing and input packing

Runtime inputs:

- `past_values`: float tensor `[B, context_length, num_input_channels]`.
- `observed_mask`: optional float/bool-like tensor `[B, context_length, num_input_channels]`, values 1 observed and 0 missing. If omitted, source uses `torch.ones_like(past_values)`.
- Forecast `future_values`: optional training label `[B, prediction_length, C]`; not required for inference.
- Classification `target_values`: optional training label `[B]`; not required for inference.
- Regression `target_values`: optional training label `[B, num_targets]`; not required for inference.

Model-owned packing:

- Patch extraction is inside the graph, not a processor: tail crop, temporal `unfold`, transpose to `[B, C, P, patch_length]`.
- `sequence_start` can be nonzero if the stride/patch configuration does not exactly cover `context_length`; the model drops leading timesteps and keeps the newest covered window.
- Missing value replacement is caller-owned, but scaling uses `observed_mask` to ignore missing entries in loc/scale.

No tokenizer, OCR, image/audio/video processor, cu-seqlens metadata, placeholder scatter, or cacheable modality projector is involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: temporal non-overlap unfold + patch Linear -> strided batched GEMM

Source pattern:

```text
tail_slice([B,T,C]) -> unfold time to [B,C,P,L] -> Linear(L -> D)
```

Replacement:

```text
StridedWindowView/Copy [B*C*P, L] -> GEMM(weight[L,D]) + bias -> reshape [B,C,P,D]
```

Preconditions:

- `patch_stride == patch_length` for simple non-overlap view. Overlap can still lower through im2col/window copy but not a pure reshape.
- `sequence_start` must be applied exactly.
- Input layout is contiguous `[B, T, C]` or a layout-aware gather is available.
- Linear weight remains PyTorch layout `[D, L]`; GEMM consumes `x @ weight.T + bias`.

Failure cases:

- Overlapping patches require explicit gather/copy or strided view support.
- Dynamic `context_length`, `patch_length`, or stride are not source-supported by a loaded config.

Parity test sketch: compare patcher output for random `[B,T,C]` across non-overlap and overlap configs, including `sequence_start > 0`.

### Rewrite: axis MLP as grouped/batched GEMM

Source pattern:

```text
Linear applied along C, P, or D after transpose/permute
```

Replacement:

```text
reshape leading dims to batch rows -> GEMM -> GELU/dropout-disabled -> GEMM -> reshape back
```

Preconditions:

- Inference mode so dropout is identity.
- Axis order is known and transposes are either materialized or absorbed into accessors.
- For channel mixer, `mode == "mix_channel"` and channel count is static.

Failure cases:

- BatchNorm branch may require preserving reshape order before normalization.
- Distributional heads return tuples and need separate parameter projections.

Parity test sketch: single `PatchTSMixerLayer` parity for common and mix-channel modes.

### Rewrite: gated attention block fusion

Source pattern:

```text
attn_weight = softmax(Linear(x), dim=-1)
y = x * attn_weight
```

Replacement:

```text
Linear -> row-wise softmax -> elementwise multiply, optionally fused over the gated axis
```

Preconditions:

- Gated axis is the last logical dim after any transpose.
- No dropout in the gate.

Failure cases:

- Axis/layout rewrite must not move the softmax dimension.

Parity test sketch: random tensors for channel, patch, and feature gate shapes.

### Rewrite: classification/regression aggregation

Source pattern:

```text
transpose [B,C,P,D] -> [B,C,D,P] -> max/mean/last over P -> flatten -> Linear
```

Replacement:

```text
reduce over patch axis in source layout -> flatten [C,D] -> GEMM
```

Preconditions:

- `head_aggregation` is one of `use_last`, `max_pool`, `avg_pool`.
- Patch axis is identified before any layout translation.

Failure cases:

- `head_aggregation=None` must flatten all `[C,P,D]` in the exact source order.

### Rewrite: optional tiny self-attention -> standard dense attention

Source pattern:

```text
reshape [B,C,P,D] to [B*C,P,D] -> QKV/O projections -> dense noncausal attention
```

Replacement:

```text
standard MHA with batch = B*C, seq = P, heads = self_attn_heads
```

Preconditions:

- `d_model % self_attn_heads == 0`.
- No attention mask and no cache.
- Preserve source math order: scale scores before softmax, dropout identity in inference.

Failure cases:

- If selected attention implementation changes numerical behavior materially, keep eager parity path for validation.

## 10. Kernel fusion candidates

Highest priority:

- Patch extraction + patch projection, because it is the first layout-shaping operation and can avoid materializing `[B,C,P,L]` for non-overlap configs.
- Axis MLP blocks over patch and feature axes, because the model is mostly Linear/GELU/Linear/residual work.
- LayerNorm + feature MLP fusion for `[B,C,P,D]`, especially with small `D=48` representative configs.
- Forecast head flatten + Linear + transpose + inverse scale for deterministic forecasting.

Medium priority:

- Gated Linear+Softmax+multiply over channel/patch/feature axes.
- Classification/regression scale injection, because it adds several tiny Linear ops and repeats loc/scale across patches.
- Optional patch self-attention as dense attention with batch folded into `B*C`.
- Mean/std scaler reductions over `[T]`, especially when batch size and channel count are high.

Lower priority:

- Pretraining random/forecast masking. It is training/pretraining-oriented and RNG-heavy.
- Distributional sampling in `generate()`. Sampling belongs outside the deterministic compiled graph for first integration.
- BatchNorm branch unless a production config needs `norm_mlp` containing `batch`.

## 11. Runtime staging plan

Stage 1: parse `PatchTSMixerConfig`, load weights, and run deterministic `PatchTSMixerModel` encoder for `common_channel`, `LayerNorm`, `gated_attn=True`, `self_attn=False`, `scaling=True/std`, non-overlap patches.

Stage 2: implement `PatchTSMixerForPrediction` point forecast (`loss="mse"`) and inverse scaling. Validate `ibm/patchtsmixer-etth1-forecasting` and `ibm-granite/granite-timeseries-patchtsmixer`.

Stage 3: add layout/fusion rewrites for patch projection and axis MLPs. Keep source-layout fallback.

Stage 4: add `mode="mix_channel"` and `prediction_channel_indices` slicing.

Stage 5: add classification/regression heads, including `InjectScalerStatistics4D` and all `head_aggregation` modes.

Stage 6: add optional patch self-attention with a standard dense attention backend.

Stage 7: add distributional `nll` parameter heads and expose deterministic distribution parameters. Keep stochastic `.generate()` sampling in host/PyTorch-compatible code at first.

Stage 8: add pretraining mask/reconstruction only if model adaptation or masked-pretrain parity is a product goal.

## 12. Parity and validation plan

- Config tests: verify `num_patches`, `sequence_start`, `patch_last`, and strict handling of supported fields for default and fetched configs.
- Scaler tests: std, mean, no-op, all-missing channels, partial observed masks, and source mismatch for `scaling=True` as std behavior.
- Patchify tests: non-overlap `[512,16,16]`, stock `[140,8,8]`, and synthetic overlap cases.
- Single-block tests: patch mixer, feature mixer, channel mixer, gated attention enabled/disabled.
- Encoder tests: compare after 1 layer and after all layers for representative configs.
- Forecast head tests: deterministic point output shape `[B, prediction_length, C]`, channel slicing, inverse scaling.
- Classification/regression tests: all `head_aggregation` modes and `output_range`.
- Optional attention tests: `self_attn=True`, multiple head counts, source eager attention comparison.
- Distribution tests: StudentT, Normal, NegativeBinomial parameter shapes/domain maps; do not require random sample bit parity.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-5`; fp16/bf16 start with `rtol=1e-2, atol=1e-2` for fused reductions/softmax-like gates, tightened per kernel.

## 13. Performance probes

- Patchify + patch projection throughput for non-overlap and overlap strides.
- Encoder-only throughput by batch size, channel count, context length, and patch count.
- Axis MLP microbenchmarks by axis: channel (`C`), patch (`P`), feature (`D`).
- Gated attention overhead with and without fusion.
- Scaler reduction cost over large `context_length` and many channels.
- Forecast head throughput and memory traffic for `P*D -> prediction_length`.
- Optional patch self-attention sweep over `P`, heads, and `B*C`.
- End-to-end deterministic forecast throughput for `ibm-granite/granite-timeseries-patchtsmixer`.
- Distributional head overhead vs point forecast.
- Temporary memory probes for materialized unfold `[B,C,P,L]` vs fused/windowed lowering.

## 14. Skip/defer list

- Training losses and gradient behavior.
- Pretraining random/forecast masks for first deterministic forecast integration.
- Stochastic `.generate()` sampling inside the compiled graph.
- Distributional NLL heads until point forecast is stable.
- Optional patch self-attention if no target checkpoint enables it.
- BatchNorm branch unless a checkpoint requires it.
- `mix_channel` mode for the first IBM/granite forecast configs, which use `common_channel`.
- Dynamic context/patch lengths; Transformers source expects fixed config lengths.
- Any tokenizer, generation KV cache, beam search, or language-model scheduling work; not applicable.

## 15. Final implementation checklist

- [ ] Parse `PatchTSMixerConfig` and derive `num_patches`/`sequence_start`.
- [ ] Load Linear, LayerNorm/BatchNorm, optional positional, and head weights.
- [ ] Implement std/mean/no-op scalers with `observed_mask`.
- [ ] Implement source-faithful temporal patchify ABI.
- [ ] Implement patch projection `Linear(patch_length -> d_model)`.
- [ ] Implement feature and patch mixer MLP blocks with gated attention.
- [ ] Implement residual and normalization ordering exactly.
- [ ] Implement deterministic forecast head and inverse scaling.
- [ ] Add parity for `ibm/patchtsmixer-etth1-forecasting`.
- [ ] Add patchify/projection fusion rewrite with overlap fallback.
- [ ] Add axis-MLP lowering through batched GEMM.
- [ ] Add `mix_channel` channel mixer support.
- [ ] Add classification/regression heads and scale injection.
- [ ] Add optional patch self-attention support.
- [ ] Add distribution parameter heads and host-side sampling policy.
- [ ] Benchmark encoder, patchify/projection, forecast head, and end-to-end forecast.
