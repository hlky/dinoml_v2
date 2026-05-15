# PatchTST Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model family: `patchtst`.

Primary DinoML runtime target: `PatchTSTForPrediction` deterministic time-series forecasting. `PatchTSTForPretraining`, `PatchTSTForClassification`, `PatchTSTForRegression`, probabilistic sampling, and channel-attention variants are documented for staging.

Files inspected:

- `transformers/src/transformers/models/patchtst/configuration_patchtst.py`
- `transformers/src/transformers/models/patchtst/modeling_patchtst.py`
- `transformers/src/transformers/time_series_utils.py`
- `transformers/tests/models/patchtst/test_modeling_patchtst.py`
- `transformers/docs/source/en/model_doc/patchtst.md`

Pinned URLs:

- [configuration_patchtst.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/patchtst/configuration_patchtst.py)
- [modeling_patchtst.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/patchtst/modeling_patchtst.py)
- [time_series_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/time_series_utils.py)
- [test_modeling_patchtst.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/tests/models/patchtst/test_modeling_patchtst.py)

Configs inspected:

- [ibm-granite/granite-timeseries-patchtst](https://huggingface.co/ibm-granite/granite-timeseries-patchtst/raw/main/config.json), saved as `_sources/ibm-granite__granite-timeseries-patchtst.config.json`.
- [namctin/patchtst_etth1_pretrain](https://huggingface.co/namctin/patchtst_etth1_pretrain/raw/main/config.json), saved as `_sources/namctin__patchtst_etth1_pretrain.config.json`.
- [ibm/patchtst-etth1-regression-distribution](https://huggingface.co/ibm/patchtst-etth1-regression-distribution/raw/main/config.json), saved as `_sources/ibm__patchtst-etth1-regression-distribution.config.json`.
- [ibm-research/patchtst-etth1-pretrain](https://huggingface.co/ibm-research/patchtst-etth1-pretrain/raw/main/config.json), saved as `_sources/ibm-research__patchtst-etth1-pretrain.config.json`.
- [namctin/patchtst_etth1_forecast](https://huggingface.co/namctin/patchtst_etth1_forecast/raw/main/config.json) returned 401 Unauthorized. The Granite config is an open mirror-like copy with `_name_or_path: namctin/patchtst_etth1_forecast`; treat the original URL as a gated gap.

Local snapshots for inspected source slices, docs, tests, and configs live under `agents/plans/transformers/patchtst/_sources/`.

Assumptions: no processor or feature extractor is part of this family. Caller/data pipeline owns window selection, missing-value replacement, and `past_observed_mask`. No tests/import execution was required or run.

## 2. High-level architecture

PatchTST is an encoder-only time-series Transformer. It scales a `[batch, context_length, num_input_channels]` history, extracts temporal patches, embeds each patch as a token, runs Transformer encoder layers, and applies task-specific heads.

```text
past_values + optional observed mask
  -> per-channel scaling over time
  -> temporal patchify/unfold
  -> optional pretraining mask
  -> Linear(patch_length -> d_model)
  -> positional encoding + optional cls token
  -> encoder layers: temporal MHA, optional channel MHA, FFN
  -> forecast / pretrain / classification / regression head
```

There is no autoregressive prefill/decode split and no KV cache. Independently cacheable stages are limited to application-level reuse of an entire encoded history.

## 3. Important config dimensions

| Field | Default | Runtime meaning |
| --- | ---: | --- |
| `num_input_channels` | 1 | Input channel count `C`; checked against runtime input. |
| `context_length` | 32 | Required input time length `T`. |
| `patch_length` | 1 | Patch width `P`. |
| `patch_stride` | 1 | Patch stride `S`. |
| `num_patches` | derived | `(max(T, P) - P) // S + 1`. |
| `d_model` | 128 | Token width. |
| `num_attention_heads` | 4 | MHA heads; `d_model` must be divisible by heads. |
| `head_dim` | derived | `d_model // num_attention_heads`. |
| `num_hidden_layers` | 3 | Encoder depth. |
| `ffn_dim` | 512 | FFN intermediate size. |
| `norm_type` | `batchnorm` | `BatchNorm1d(d_model)` over transposed sequence, or `LayerNorm`. |
| `share_embedding` | `True` | Shared patch embedding across channels; false creates per-channel embeddings. |
| `channel_attention` | `False` | Adds attention across channels at each patch position. |
| `pre_norm` | `True` | Norm-before-subblock if true; post-norm if false. |
| `positional_encoding_type` | `sincos` | Fixed normalized sin/cos or learned random. |
| `use_cls_token` | `False` | Prepends a learned per-channel cls token. |
| `scaling` | `std` | `std`, `mean`/`True`, or no-op scaling. |
| `prediction_length` | 24 | Forecast horizon `H`. |
| `pooling_type` | `mean` | Head pooling: `mean`, `max`, or `None` where supported. |
| `share_projection` | `True` | Shared forecast head across channels; false creates per-channel heads. |
| `loss` | `mse` | MSE uses deterministic linear head; other values use distribution heads. |

Representative config sweep:

| Config | Architecture | T | C | P/S | Patches | Layers | D/heads | Head |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| `ibm-granite/granite-timeseries-patchtst` | `PatchTSTForPrediction` | 512 | 7 | 12/12 | 42 | 3 | 128/16 | cls token, `pooling_type=None`, shared projection, `H=96`, MSE. |
| `namctin/patchtst_etth1_pretrain` | `PatchTSTForPretraining` | 512 | 7 | 12/12 | 42 | 3 | 128/16 | forecast mask, reconstruct patches. |
| `ibm/patchtst-etth1-regression-distribution` | `PatchTSTForRegression` | 512 | 6 | 12/12 | 42 | 3 | 128/16 | mean pooling, normal distribution, `num_targets=1`. |
| `ibm-research/patchtst-etth1-pretrain` | historical config | 512 | 7 | 12/12 via legacy `stride` | 42 | legacy says 6 | legacy 128/16 | Uses old field names not read directly by current source. |

Common ETTh shape:

```text
past_values [B,512,7] -> patches [B,7,42,12]
-> embeddings [B,7,42,128]
-> cls variant [B,7,43,128]
-> forecast [B,96,7]
```

## 3a. Family variation traps

- Source input layout is `[B,T,C]`; internals become `[B,C,num_patches,features]`. Do not apply image NHWC/NCHW assumptions.
- Patch count and `sequence_start` depend on both `patch_length` and `patch_stride`; patchify can drop leading time steps.
- `share_embedding=False` and `share_projection=False` introduce per-channel `ModuleList` loops.
- `channel_attention=True` adds a second self-attention over `C` channels, batched as `B * L`.
- `norm_type=batchnorm` is a transposed `BatchNorm1d`, not LayerNorm.
- `pre_norm=False` changes residual/norm ordering.
- `use_cls_token=True` shifts patch-token indexing and changes head/pretraining behavior.
- `pooling_type=None` flattens all patch tokens for prediction; classification/regression reject `None`.
- `loss != "mse"` changes the head to distribution parameter projections, not just the loss.
- Historical configs may contain ignored names such as `encoder_layers`, `encoder_ffn_dim`, `encoder_attention_heads`, `stride`, `dropout_path`, `norm`, `shared_embedding`, `shared_projection`, `mask_ratio`, `mask_patches`, and `PatchTSTForMaskPretraining`. Current source reads modern names such as `num_hidden_layers`, `ffn_dim`, `num_attention_heads`, `patch_stride`, `norm_type`, `share_embedding`, and `share_projection`.
- Fetched configs contain `dropout`, but current modeling reads `attention_dropout`, `positional_dropout`, `path_dropout`, `ff_dropout`, and `head_dropout`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Exact rank/shape validation for `[B,T,C]`.
- `ones_like` observed-mask default.
- Slice, `unfold`/sliding-window extraction, transpose, reshape/view, contiguous copy, flatten, stack, concat.
- Mean and max reductions over the patch-token axis.
- Masked fill for pretraining.

Neural network primitives:

- Shared or per-channel `Linear(P -> D)` patch embedding.
- MHA Q/K/V/out `Linear(D -> D)` with optional bias.
- FFN `Linear(D -> ffn_dim) -> GELU/ReLU -> Linear(ffn_dim -> D)`.
- `BatchNorm1d(D)` over `[B*,D,L]` and optional `LayerNorm(D)`.
- Forecast, pretraining, classification, and regression linear heads.
- Dropout is an inference identity but matters for training-mode parity.

Attention primitives:

- Non-causal MHA over patch tokens.
- Optional non-causal MHA over channels.
- No source causal mask, cross-attention use, packed varlen, local attention, RoPE, ALiBi, or KV cache.

Preprocessing-coupled ops:

- Mean/std/no-op scaling with observed masks.
- Patch extraction is in the model graph.

Probabilistic head ops:

- Distribution parameter projection uses one linear per parameter.
- `squareplus`, epsilon clamp, affine transform by `loc`/`scale`, distribution `sample` and `log_prob` for probabilistic modes.

## 5. Layer/block breakdown

Base forward:

```text
past_observed_mask = ones_like(past_values) if omitted
scaled, loc, scale = scaler(past_values, past_observed_mask)
patches = patchify(scaled)                     # [B,C,N,P]
masked, mask = optional masking(patches)
x = patch_embedding(masked)                    # [B,C,N,D]
x = positional_encoding(x)                     # [B,C,N(+1),D]
x = encoder(x)
head(x)
```

Encoder layer:

```text
# temporal attention
x_time = view(x, [B*C,L,D])
if pre_norm:
    x_time = x_time + MHA(norm1(x_time))
else:
    x_time = norm1(x_time + MHA(x_time))
x = view(x_time, [B,C,L,D])

# optional channel attention
if channel_attention:
    x_ch = transpose/view(x, [B*L,C,D])
    x_ch = residual MHA with norm2
    x = view/transpose back to [B,C,L,D]

# FFN
x_ff = view(x, [B*C,L,D])
if pre_norm:
    x_ff = x_ff + Linear2(act(Linear1(norm3(x_ff))))
else:
    x_ff = norm3(x_ff + Linear2(act(Linear1(x_ff))))
x = view(x_ff, [B,C,L,D])
```

Common projection shapes for ETTh configs:

- Patch embedding: `Linear(12 -> 128)`.
- MHA per layer: Q/K/V/out `Linear(128 -> 128)`, 16 heads, head dim 8.
- FFN: `Linear(128 -> 512)` and `Linear(512 -> 128)`.
- Forecast head with cls token: shared `Linear(128 -> 96)` per channel.
- Regression head with mean pooling and 6 channels: flatten `[B,6,128] -> [B,768]`, then distribution projection.

## 6. Attention requirements

PatchTST uses encoder self-attention only.

| Requirement | Source behavior |
| --- | --- |
| Causal | No. |
| Self/cross | Self-attention; `PatchTSTAttention` has a cross-attention argument but PatchTST encoder does not use it. |
| MHA/MQA/GQA | MHA only. |
| Head dimensions | `head_dim = d_model // num_attention_heads`; Q/K/V width is `d_model`. |
| Masking | No encoder attention mask passed. Pretraining masks input patches, not attention scores. |
| Packed/varlen | Not implemented. |
| Sliding/local | Not implemented. |
| Position interaction | Additive patch positional encoding before encoder. |
| KV cache | Not applicable. |
| Backend | `ALL_ATTENTION_FUNCTIONS` dispatch with eager fallback: QK matmul, optional additive mask, softmax, dropout, AV matmul. |

Channel attention, when enabled, is attention over `C` channels:

```text
[B,C,L,D] -> transpose -> [B,L,C,D] -> view [B*L,C,D]
```

## 7. Position encoding and custom math

Sin/cos positional table:

```python
position = arange(num_patches).unsqueeze(1)
div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)
pe = (pe - pe.mean()) / (pe.std() * 10)
```

This can be precomputed per config. With `use_cls_token=True`, the table has `num_patches + 1` rows; row 0 is added to the learned cls token and rows 1.. are added to patch embeddings.

Std scaler:

```python
den = observed.sum(dim=1, keepdim=True).clamp_min(1)
loc = (data * observed).sum(dim=1, keepdim=True) / den
var = (((data - loc) * observed) ** 2).sum(dim=1, keepdim=True) / den
scale = sqrt(var + minimum_scale)
scaled = (data - loc) / scale
```

Distribution positive map:

```python
squareplus = (x + sqrt(x * x + 4.0)) / 2.0
```

StudentT maps `df = 2 + squareplus(df)` and positive `scale`; Normal maps positive `scale`; NegativeBinomial maps positive `total_count` and adjusts logits by `scale.log()` when scaled.

## 8. Preprocessing and input packing

No tokenizer, image processor, or feature extractor is involved. Runtime inputs are:

- `past_values`: float `[B, context_length, num_input_channels]`.
- `past_observed_mask`: optional observed indicator `[B, context_length, num_input_channels]`.
- Training-only labels: `future_values` or `target_values`.

Model-owned packing:

- Scale before patching.
- Patchify with `past_values[:, sequence_start:, :]`, `unfold(time, patch_length, patch_stride)`, then transpose to `[B,C,N,P]`.
- Pretraining masks patched inputs and returns `[B,C,N]` mask.

CPU/data-pipeline work:

- Choose the history window.
- Replace or handle missing values consistently with the observed mask.
- Any rolling-window or multi-horizon scheduling.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patchify unfold -> WindowExtract1D

Source pattern:

```text
x[:, sequence_start:, :].unfold(time, P, S).transpose(-2,-3).contiguous()
```

Replacement:

```text
out[b,c,n,p] = x[b, sequence_start + n*S + p, c]
```

Preconditions:

- `T == context_length`, `context_length > patch_length`, positive `patch_length` and `patch_stride`.
- Dense input with source layout `[B,T,C]`.

Failure cases:

- Dynamic `T` without buckets.
- Non-dense/strided inputs unless accessor support is explicit.

Parity test: compare against PyTorch `unfold` for overlapping and non-overlapping patches.

### Rewrite: shared patch embedding -> GEMM

Source pattern:

```text
[B,C,N,P] -> Linear(P,D)
```

Replacement:

```text
reshape [B*C*N,P] -> GEMM_RCR(weight [D,P]) + bias -> reshape [B,C,N,D]
```

Preconditions:

- `share_embedding=True`.
- Patch tensor has dense contiguous `P` dimension.

Failure cases:

- `share_embedding=False` requires per-channel GEMMs or grouped GEMM.

### Rewrite: BatchNorm over sequence -> BatchNorm1dTime

Source pattern:

```text
[B*,L,D] -> transpose [B*,D,L] -> BatchNorm1d(D) -> transpose back
```

Preconditions:

- Inference/eval mode with running statistics first.
- Rank-3 input.

Failure cases:

- Training-mode batch statistics deferred.

### Rewrite: forecast head -> channel-batched GEMM

Source pattern:

```text
pool/flatten [B,C,head_dim] -> Linear(head_dim,H) -> transpose [B,H,C]
```

Replacement:

```text
reshape [B*C,head_dim] -> GEMM -> reshape [B,C,H] -> transpose
```

Preconditions:

- `share_projection=True`.
- Deterministic `loss="mse"` first.

Failure cases:

- Distribution heads return tuples of parameter tensors.
- `share_projection=False` needs per-channel projections.

## 10. Kernel fusion candidates

Highest priority:

- Patchify + shared patch embedding, because it removes a large contiguous materialization before GEMM.
- Short-sequence encoder MHA for common `L=42/43`; launch overhead can dominate.
- BatchNorm1dTime with residual patterns, because `batchnorm` is common in reachable configs.
- Forecast projection plus inverse scaling `y * scale + loc`.

Medium priority:

- FFN `Linear -> GELU/ReLU -> Linear` with residual/norm ordering.
- Mean/max pooling heads.
- Channel attention for wider multivariate configs.
- Per-channel embedding/projection variants.

Lower priority:

- Probabilistic distribution maps and sampling.
- Pretraining mask generation with RNG/sort/gather.
- Output attention tensor materialization.

## 11. Runtime staging plan

1. Parse modern config fields and reject or translate historical fields.
2. Admit deterministic `PatchTSTForPrediction` with `loss="mse"`, `channel_attention=False`, `share_embedding=True`, `share_projection=True`.
3. Implement scaler, patchify, patch embedding, positional add/cls concat.
4. Implement temporal encoder attention, norm, FFN, residuals in eval mode.
5. Implement deterministic forecast head and inverse scaling.
6. Add classification/regression deterministic heads.
7. Add probabilistic parameter projections and decide whether sampling lives in runtime or host postprocessing.
8. Add pretraining mask/reconstruction if training/pretraining parity becomes a target.
9. Optimize with fused patchify+embedding, short-sequence attention, and guarded channel attention.

## 12. Parity and validation plan

- Patchify parity against PyTorch `unfold` for several `T/P/S` combinations.
- Scaler parity for std, mean, no-op, all-observed, partially observed, and no-observed channels.
- Positional encoding parity, including cls-token variant.
- One-layer encoder parity for `batchnorm`, `pre_norm=True`, `channel_attention=False`.
- Variant parity for `layernorm`, `pre_norm=False`, and `channel_attention=True`.
- End-to-end deterministic forecast parity for `[B,512,7] -> [B,96,7]`.
- Head parity for cls, mean pooling, max pooling, and no pooling.
- Regression/classification head parity.
- Distribution domain-map parity for StudentT, Normal, and NegativeBinomial.

Recommended tolerances: fp32 `rtol=1e-4`, `atol=1e-4` following upstream tests; fp16/bf16 start at `1e-2` and tighten per kernel.

## 13. Performance probes

- Patchify throughput over `B`, `C`, `T`, `patch_length`, and `patch_stride`.
- Encoder-only throughput for common `[B,512,7]`, `P=S=12`, `D=128`, `L=42/43`.
- Attention backend comparison for short non-causal sequences.
- Channel-attention sweep over `C`.
- BatchNorm vs LayerNorm cost over `[B*C,L,D]`.
- Forecast head plus inverse-scale bandwidth.
- End-to-end deterministic forecast batch-size sweep.
- Probabilistic head overhead for `num_parallel_samples`.
- Layout probe for `[B,T,C] -> [B,C,N,P/D]` copies and patchify+embedding fusion.
- Memory probe with `output_attentions=True`.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Training-mode dropout and BatchNorm batch statistics.
- Pretraining random/forecast masking and reconstruction loss.
- Output attentions.
- Probabilistic `generate` sampling and NLL.
- `share_embedding=False`, `share_projection=False`.
- `channel_attention=True`.
- Historical config compatibility unless a target checkpoint requires it.
- Classification and regression heads for first forecasting-only integration.
- Dynamic context length; source requires exact `context_length`.

## 15. Final implementation checklist

- [ ] Parse modern `PatchTSTConfig` and reject/translate legacy config names.
- [ ] Load weights for shared embedding, encoder layers, norms, positional constants, cls token, and deterministic forecast head.
- [ ] Implement std/mean/no-op scalers with observed masks.
- [ ] Implement `WindowExtract1D` patchify with `sequence_start`.
- [ ] Lower shared `Linear(patch_length -> d_model)` patch embedding.
- [ ] Implement fixed sin/cos positional table and optional cls concat.
- [ ] Implement non-causal temporal MHA without cache.
- [ ] Implement inference `BatchNorm1dTime` and `LayerNorm`.
- [ ] Implement FFN with GELU/ReLU, residuals, and pre/post norm.
- [ ] Implement deterministic prediction head and inverse scaling.
- [ ] Add one-block and end-to-end forecast parity tests.
- [ ] Add admission tests for deferred variants.
- [ ] Benchmark patchify, encoder, head, and end-to-end forecasting.
