# TimesFM Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/timesfm-2.0-500m-pytorch
Config source: https://huggingface.co/google/timesfm-2.0-500m-pytorch/raw/main/config.json
Source files inspected:
  transformers/src/transformers/models/timesfm/configuration_timesfm.py
  transformers/src/transformers/models/timesfm/modeling_timesfm.py
  transformers/src/transformers/models/timesfm/modular_timesfm.py
  transformers/src/transformers/models/timesfm/convert_timesfm_orignal_to_hf.py
  transformers/tests/models/timesfm/test_modeling_timesfm.py
  transformers/docs/source/en/model_doc/timesfm.md
Any missing files or assumptions:
  modeling_timesfm.py is generated from modular_timesfm.py. Runtime behavior was
  checked against the generated file; future Transformers source edits should
  target modular_timesfm.py. This audit scopes native in-library `timesfm`, not
  `timesfm2_5` or older custom remote-code wrappers.
```

Representative configs inspected from Hugging Face:

- `google/timesfm-2.0-500m-pytorch`: native `TimesFmModelForPrediction`.
- `huggingface/timesfm-tourism-monthly`: native small fine-tuned/debug-like config.
- `FinText/TimesFM_20M_2023_Global`, `FinText/TimesFM_8M_2021_US`, `FinText/TimesFM_8M_2000_Global`: open mirrors using `model_type: timesfm` but legacy/custom keys and architecture `TimesFMForHF`; not directly loadable by the pinned `TimesFmConfig` without translation.
- `google/timesfm-1.0-200m-pytorch`: raw `config.json` returned 404; a historical blob found by search used `model_type: t5`, so it is out of scope for this native `timesfm` report.
- `PartAI/FlaMinGo-timesfm`: raw config returned 401, so access would be needed before classifying it.
- `google/timesfm-2.5-200m-transformers`: `model_type: timesfm2_5`; out of scope and should receive a separate audit.

## 2. High-level architecture

TimesFM is a decoder-only time-series forecasting model over non-overlapping scalar time-series patches. The first useful DinoML target is `TimesFmModelForPrediction` inference for mean and quantile forecasts.

```text
list/ragged 1D time series
  -> CPU/data-pipeline pad-or-truncate, optional moving-average split
  -> [B, context_length] values + [B, context_length] padding + [B,1] freq
  -> patch to [B, N, patch_length]
  -> normalize from selected valid patch statistics
  -> concat value patch and padding patch
  -> residual input projection to hidden states
  -> optional sinusoidal positional embedding + frequency embedding
  -> causal decoder stack over patch tokens
  -> residual horizon projection per patch
  -> select last patch forecast, append autoregressively if needed
  -> mean_predictions [B, horizon_length], full_predictions [B, horizon_length, 1 + quantiles]
```

The core graph is patch-token inference, not text generation. There is no token vocabulary, LM head, RoPE, or KV-cache implementation in the native source. The Python wrapper does an autoregressive loop over horizon patches, but for the default config `horizon_length == output_patch_len`, so one decoder pass produces the full horizon.

Independently stageable pieces:

- Preprocess/pad/truncate: CPU/data-pipeline first.
- Patch normalization and padding mask construction: GPU graph candidate, but value-dependent indexing/gather must be validated carefully.
- Decoder block parity: standard causal self-attention over patch tokens.
- Horizon projection and output rescaling: graph-owned, easy to test independently.
- Optional `window_size` decomposition and `return_forecast_on_context`: defer from first runtime target.

## 3. Important config dimensions

Native `google/timesfm-2.0-500m-pytorch` config:

| Field | Value | Source |
|---|---:|---|
| `patch_length` | 32 | config.json |
| `context_length` | 2048 | config.json |
| patch tokens `N = context_length / patch_length` | 64 | inference |
| `horizon_length` | 128 | config.json |
| output channels per horizon point | 10 | 1 mean + 9 quantiles |
| `freq_size` | 3 | config.json |
| `hidden_size` | 1280 | config.json |
| `intermediate_size` | 1280 | config.json |
| `num_hidden_layers` | 50 | config.json |
| `num_attention_heads` | 16 | config.json |
| `head_dim` | 80 | config.json |
| Q/K/V width | 1280 | 16 * 80 |
| attention type | causal MHA | source |
| `rms_norm_eps` | 1e-6 | config.json |
| `attention_dropout` | 0.0 | config.json |
| `use_positional_embedding` | false | config.json |
| `pad_val` | 1123581321.0 | config.json |
| `tolerance` | 1e-6 | config.json |
| dtype | float32 | config.json |

Checkpoint sweep:

| Model/repo | Native class? | Context | Patch | Tokens | Layers | Hidden | Heads x dim | Horizon | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `google/timesfm-2.0-500m-pytorch` | yes | 2048 | 32 | 64 | 50 | 1280 | 16 x 80 | 128 | Production native target. |
| `huggingface/timesfm-tourism-monthly` | yes | 64 | 16 | 4 | 4 | 32 | 4 x 4 | 128 | Tiny native fine-tuned config; includes ignored `prediction_length: 24`. |
| `Ayushk44/timesfm-2.5-7m-pytorch-dummy` | partial | 16384 | 32 | 512 | 2 | 256 | 4 x 64 | 128 | `model_type: timesfm` but includes non-native `quantile_horizon_length`; dummy repo, not a representative production target. |
| `FinText/TimesFM_20M_2023_Global` | no | 512 | omitted | unknown | 9 | 432 | 6 x 72 | 128 | Uses legacy keys `context_len`, `num_layers`, `num_heads`, architecture `TimesFMForHF`. |
| `FinText/TimesFM_8M_2021_US` | no | 512 | omitted | unknown | 7 | 264 | 4 x 66 | 128 | Legacy/custom wrapper surface; reject or translate explicitly. |
| `google/timesfm-2.5-200m-transformers` | no | 16384 | 32 | 512 | 20 | 1280 | 16 x 80 | 128 | Separate `timesfm2_5` family with RoPE/QK-norm-like changes; do not fold into this audit. |

## 3a. Family variation traps

- Native `TimesFmConfig` expects `context_length`, `horizon_length`, `num_hidden_layers`, `num_attention_heads`, and `patch_length`. Repos with `context_len`, `horizon_len`, `num_layers`, or `num_heads` are not native without a key-translation adapter.
- `hidden_size == num_attention_heads * head_dim` for inspected native configs, but the source uses `num_heads * head_dim` explicitly for Q/K/V and output projection input width. Do not infer this from `hidden_size` alone.
- `use_positional_embedding` is configurable. The 500M native config disables it, but tests enable it; DinoML should support or reject it explicitly.
- `freq` shape is `[B,1]` after preprocessing, and `nn.Embedding` returns `[B,1,D]`; direct callers to `TimesFmModel` must preserve broadcast behavior against `[B,N,D]`.
- The source computes its own padding and causal masks. Generic external attention-mask tests are skipped in Transformers.
- There is no KV cache despite an autoregressive horizon loop. Each decode step recomputes the full current context prefix.
- Optional `window_size` uses `F.pad` + `conv1d` and doubles the logical batch into smoothed/residual streams, then recombines outputs. Keep it out of the first compiled graph.
- `return_forecast_on_context` changes output slicing and includes per-context patch forecasts. First integration can target default `False`.
- `truncate_negative` is a postprocessing clamp controlled by input minimum.
- `future_values` only drives loss computation; inference can defer it.
- `prediction_length` in `huggingface/timesfm-tourism-monthly` is not read by the pinned source. Forecast output length is `horizon_length`.
- No image/video layouts are present. Layout rewrite concerns are sequence axes only; a layout pass should protect patch/time axes and attention head axes with no-layout-translation guards.

## 4. Operator coverage checklist

Tensor/layout ops:

- Ragged/list input handling outside compiled graph, or an admission policy requiring pre-batched `[B,T]` values.
- Front pad/truncate to `context_length`; slice last `context_length` values.
- `view [B,T] -> [B,N,P]` with guard `T % patch_length == 0`.
- `view [B,T] -> [B,N,P]` for padding mask.
- `where`, `abs`, scalar comparison against `pad_val` and `tolerance`.
- `cat` over last dim: `[B,N,P] + [B,N,P] -> [B,N,2P]`.
- `min` over patch axis `P` to produce patch padding `[B,N]`.
- `sum` over `P`, `argmax` over patch token axis, `where` fallback index, advanced gather `inputs[bidx, patch_idx, :]`.
- `arange`, expand, modulo, `gather` for optional shifted positional embedding.
- `reshape/view` horizon output `[B,N,H*(1+Q)] -> [B,N,H,1+Q]`.
- Concatenate generated forecast patch to `final_out` along time axis in the Python horizon loop.

Neural network primitives:

- Input residual block: `Linear(2P -> I) -> SiLU -> Linear(I -> D)` plus residual `Linear(2P -> D)`, where 500M has `P=32`, `I=D=1280`.
- Frequency embedding: `Embedding(freq_size=3, D=1280)`, broadcast add to `[B,N,D]`.
- Decoder RMSNorm over last dim `D`, upcast to fp32 for variance.
- Decoder MLP: `LayerNorm(D, eps=1e-6) -> Linear(D -> I) -> ReLU -> Linear(I -> D) -> optional padding multiply -> residual add`.
- Horizon residual block: `Linear(D -> I) -> SiLU -> Linear(I -> H*(1+Q))` plus residual `Linear(D -> H*(1+Q))`; 500M output width is `128 * 10 = 1280`.
- Optional moving average: 1D constant-depthwise-like convolution over one scalar series with kernel `[1/window_size]`.
- Optional loss: MSE plus quantile pinball loss; training-only for first integration.

Attention primitives:

- Causal dense MHA over patch tokens: Q/K/V `Linear(D -> num_heads * head_dim)`.
- Q reshaped `[B,N,1280] -> [B,16,N,80]`.
- Per-dimension query scaling: `query *= softplus(scaling[80]) * 1.442695041 / sqrt(80)`.
- Additive 4D mask `[B,1,N,N]` using dtype min values, combining padding and upper-triangular causal mask with `minimum`.
- Attention softmax over key dimension in fp32, cast back to query dtype.
- Matmul attention weights by values, transpose back, output `Linear(1280 -> 1280)`.
- SDPA is supported by source dispatch; FlashAttention dispatch is not guaranteed by tests and is skipped as not yet supported because of masks.

Position/custom math:

- Optional absolute sinusoidal position embedding with timescale buffer `[D/2]`; not RoPE.
- Optional left-padding shift of positional table via modular gather.

Generation/cache ops:

- No KV cache. Horizon generation is a host/runtime loop over `ceil(horizon_length / output_patch_len)` passes; current native source uses `output_patch_len = horizon_length`, so one pass for inspected configs.
- Independently cacheable preprocessing outputs: padded context, padding mask, freq ids. Decoder hidden states are not cached across horizon steps in source.

Preprocessing-coupled ops:

- Per-sample variable-length sequence handling, left padding, truncation, frequency defaulting to high frequency id 0.
- Optional moving-average decomposition and recomposition.
- Optional nonnegative truncation based on input minimum.

## 5. Layer/block breakdown

Prediction forward, default inference:

```text
inputs = last forecast_context_len values from each 1D series
input_ts, input_padding, freq = _preprocess(inputs, freq)
final_out = input_ts
for horizon step:
  current_padding = input_padding[:, :final_out_length]
  input_ts = final_out[:, -forecast_context_len:]
  input_padding = current_padding[:, -forecast_context_len:]
  decoder_output = TimesFmModel(input_ts, input_padding, freq)
  fprop_outputs = horizon_ff_layer(decoder_output.last_hidden_state)
  fprop_outputs = fprop_outputs.view(B, N, horizon_length, 1 + num_quantiles)
  fprop_outputs = fprop_outputs * scale[:,None,None,None] + loc[:,None,None,None]
  new_ts = fprop_outputs[:, -1, :horizon_length, 0]
  append new_ts to final_out
return selected mean/full outputs
```

Decoder input transform:

```text
past_values [B,T], past_values_padding [B,T]
patched_inputs = view(B, N, P)
patched_pads = view(B, N, P)
patched_inputs = where(patched_pads == 1, 0, patched_inputs)
patched_pads = where(abs(patched_inputs - pad_val) < tolerance, 1, patched_pads)
mu, sigma = stats from first patch with >=3 valid elements
patched_inputs = normalize and preserve pad_val sentinels
concat_inputs = cat([patched_inputs * (1-patched_pads), patched_pads], -1)
x = ResidualBlock(2P -> I -> D, residual 2P -> D)
x += freq_embedding(freq)
attention_mask = causal + padding mask
```

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(D -> H*Hd)(x_norm).view(B,N,H,Hd).transpose(1,2)
q = q * softplus(per_dim_scaling[Hd]) * 1.442695041 / sqrt(Hd)
k = Linear(D -> H*Hd)(x_norm).view(B,N,H,Hd).transpose(1,2)
v = Linear(D -> H*Hd)(x_norm).view(B,N,H,Hd).transpose(1,2)
attn = causal_attention(q, k, v, additive_mask)
x = residual + Linear(H*Hd -> D)(attn.transpose/reshape)
x = x + Linear(I -> D)(ReLU(Linear(D -> I)(LayerNorm(x)))) * (1 - paddings[...,None])
```

Projection weights use normal PyTorch `nn.Linear` layout `[out_features, in_features]`. The conversion script shows original TimesFM stored a fused `qkv_proj` split by row blocks `[Q rows, K rows, V rows]`; the native HF module stores separate `q_proj`, `k_proj`, and `v_proj`.

## 6. Attention requirements

- Type: causal self-attention over patch tokens.
- Heads: MHA, no GQA/MQA in native `timesfm`; `num_key_value_heads` is not a native field.
- 500M shapes: `B x N x D = B x 64 x 1280`; Q/K/V are `B x 16 x 64 x 80`.
- Query/key/value widths are all `num_attention_heads * head_dim`, which equals 1280 for 500M.
- Attention mask: source builds `[B,1,N,N]` additive mask. Padding mask is `[B,N]` with 1 for padded patch; causal mask is strict upper triangle. Combined mask is `torch.minimum(padding_mask * min_dtype, causal_mask)`.
- Softmax: eager path explicitly computes softmax in fp32, casts back.
- Scaling: backend scaling is `1.0`; the model-specific scale is already multiplied per query dimension.
- Dropout: `0.0` at inference; config supports attention dropout during training.
- Packed/varlen: not implemented. Use dense patch sequence.
- Sliding/local/sparse: not implemented.
- Positional interaction: optional absolute sinusoidal embedding is added before attention; no RoPE/ALiBi.
- Cache: no KV cache or cache reorder.
- SDPA: source declares `_supports_sdpa=True` and tests eager vs SDPA parity; generic flash dispatch test is skipped because masks are not compile-ready.

## 7. Position encoding and custom math

Default 500M config has `use_positional_embedding=false`, so position embedding can be deferred for the first production checkpoint if admission rejects configs that enable it. Tests enable it, so parity coverage should eventually include it.

Sinusoidal embedding:

```python
num_timescales = hidden_size // 2
inv_timescales = min_timescale * exp(arange(num_timescales) * -log(max/min) / max(num_timescales - 1, 1))
scaled_time = position[..., None] * inv_timescales[None, None, :]
signal = cat([sin(scaled_time), cos(scaled_time)], dim=2)
signal = pad(signal, last_dim_to_hidden_size)
```

Per-dimension query scaling:

```python
scale = softplus(attn_scaling[head_dim]) * (1.442695041 / sqrt(head_dim))
query = query * scale[None, None, None, :]
```

Masked statistics choose the first patch with at least three non-padded points:

```python
valid_per_patch = sum(1 - padding, dim=2)
idx = argmax((valid_per_patch >= 3).int(), dim=1)
idx = where(no_patch_has_3_valid, N - 1, idx)
arr = inputs[batch_index, idx, :]
mask = 1 - padding[batch_index, idx, :]
mu = sum(arr * mask) / clamp(sum(mask), min=1)
sigma = sqrt(clamp(sum(((arr - mu) * mask) ** 2) / count, min=0))
```

Precomputable: sinusoidal inv-timescales and fixed causal mask per token bucket. Dynamic: padding-derived patch index, loc/scale, shifted positional table, and additive padding mask.

## 8. Preprocessing and input packing

Primary public forward accepts `past_values` as a sequence of 1D tensors. `_preprocess` pads or truncates each sequence to `context_length`:

- If shorter than context, left-pad values with zeros.
- Build padding of length `input_len + horizon_length`; left-pad it with ones for missing context.
- If longer than context, keep the final `context_length` values and the matching final padding span.
- Stack into `input_ts [B, context_length]` and `input_padding [B, context_length + horizon_length]`.
- Frequency defaults to `[0] * B` when omitted and is converted to int32 shape `[B,1]`.

For first DinoML integration, prefer a compiled core accepting already padded tensors:

```text
past_values: float [B, context_length]
past_values_padding: float/int [B, context_length]
freq: int [B,1] or [B] with normalized frontend
```

Then keep the list/ragged padding policy in Python/host glue. Optional `window_size` decomposition is CPU/data-pipeline work first: each series becomes `[moving_average, residual]`, batch doubles, and outputs are pairwise summed after model inference.

No tokenizer, placeholder tokens, pixel/audio processors, or multimodal scatter exist.

## 9. Graph rewrite / lowering opportunities

### Rewrite: static patch view and concat to input residual GEMM

Source pattern:

```text
past_values.view(B,N,P), padding.view(B,N,P)
where/mask normalize
cat([values, pads], -1)
ResidualBlock(2P -> I -> D)
```

Replacement:

```text
PatchView -> MaskedNormalize -> ConcatenateLastDim -> two GEMMs with SiLU residual block
```

Preconditions:

- `context_length % patch_length == 0`.
- Inputs are dense row-major `[B,T]`.
- Padding values are normalized to 0/1 before compiled graph or guarded.
- Preserve last-dim concat order `[normalized_values, patched_pads]`.

Failure cases:

- Ragged list inputs inside compiled graph.
- Non-contiguous tensors whose `.view` semantics would fail.
- Dynamic context lengths not divisible by patch length.

Parity test sketch: compare `TimesFmModel.forward` hidden state after `input_ff_layer` for random padded inputs and sentinel `pad_val` cases.

### Rewrite: Q/K/V separate linears to packed projection

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
view to [B,H,N,Hd]
```

Replacement:

```text
single GEMM D -> 3*(H*Hd), then split in Q,K,V row-block order
```

Weight transform:

```python
packed_weight = torch.cat([q.weight, k.weight, v.weight], dim=0)
packed_bias = torch.cat([q.bias, k.bias, v.bias], dim=0)
```

Preconditions:

- All three projections have same input, dtype, layout, and bias presence.
- Split order is Q then K then V. This matches the conversion script's original fused row blocks.
- `H * Hd` may differ from `hidden_size`; use explicit projection width.

Failure cases:

- Future configs with MQA/GQA or unequal K/V widths.
- Any source path that disables bias must be reflected in packed bias handling.

Parity test sketch: packed projection output split equals three separate projections before query scaling.

### Rewrite: query scale folded into Q projection

Source pattern:

```text
q = q_proj(x).view(..., Hd)
q *= softplus(scaling) * const
```

Replacement:

```text
scale Q projection output per head-dim, or pre-scale q_proj rows
```

Preconditions:

- Inference weights are fixed.
- The scale is shared across all heads for the same head dimension.
- Preserve dtype/upcast policy for `softplus` parameter materialization.

Weight transform:

```python
row_scale = scale.repeat(num_heads)
q_weight_scaled = q_weight * row_scale[:, None]
q_bias_scaled = q_bias * row_scale
```

Failure cases: training or mutable scaling parameter; backend attention that applies its own non-unit scaling.

### Rewrite: default one-pass horizon decode

Source pattern:

```text
num_decode_patches = ceil(horizon_length / output_patch_len)
output_patch_len = horizon_length
```

Replacement: compile a single decoder pass plus last-patch projection for configs where `output_patch_len == horizon_length`.

Preconditions:

- Native source currently sets `output_patch_len = config.horizon_length`.
- `return_forecast_on_context == False`.
- `window_size is None`.

Failure cases: future split output patch length, context forecast output, or longer horizon loop.

### Guarded no-layout-translation regions

- Patch/time axes `[B,N,P]`: do not swap `N` and `P`. Reductions over `P` and `N` are semantically different.
- Attention axes `[B,H,N,Hd]`: softmax must remain over key token axis `-1` of scores `[B,H,N,N]`.
- Horizon output `[B,N,Horizon,Q+1]`: selection uses last patch token `N=-1`, horizon point axis, and quantile axis.
- Positional shift gather axis is patch-token axis 1.

There is no NHWC/NCHW image layout opportunity.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over `D=1280` with fp32 variance and elementwise weight multiply.
- Dense causal attention over short patch sequences (`N=64` for 500M, `N=4` for tiny configs) with per-dim Q scaling. A fused prefill attention path can ignore KV-cache concerns.
- QKV packed projection + query scaling. This removes three GEMM launches and an elementwise multiply.
- MLP block `LayerNorm -> Linear -> ReLU -> Linear -> padding multiply -> residual add`.
- Residual blocks around input and horizon projections, especially 500M where input and horizon output widths are both 1280.

Medium priority:

- Masked patch statistics: value-dependent gather plus reduction is small but correctness-sensitive.
- Additive mask construction for fixed `N` buckets; precompute causal mask and combine padding cheaply.
- Horizon projection + reshape + rescale `(sigma, mu)` fusion.
- Optional positional embedding table generation/cache for configs that enable it.

Lower priority:

- Host-side ragged preprocessing in compiled runtime.
- Moving-average decomposition conv1d.
- Training loss and quantile pinball loss.
- `return_forecast_on_context` slicing path.

## 11. Runtime staging plan

Stage 1: config and weight admission.

- Accept native `TimesFmConfig` keys only.
- Reject or separately translate legacy/custom configs with `TimesFMForHF`, `context_len`, `num_layers`, or `num_heads`.
- Start with `window_size=None`, `return_forecast_on_context=False`, `future_values=None`, `truncate_negative=False`.

Stage 2: compiled decoder core with padded tensors.

- Inputs: `[B,context_length]`, `[B,context_length]` padding, `freq`.
- Implement patch view, normalization, input residual block, frequency embedding, attention mask, decoder layers, horizon projection, rescale.
- Validate one-pass default horizon.

Stage 3: block-level and full-model parity.

- One decoder layer random-weight parity.
- 4-layer tiny config parity (`huggingface/timesfm-tourism-monthly`-like).
- 50-layer 500M checkpoint output slice parity.

Stage 4: optimized attention/GEMM lowering.

- QKV packed GEMM and query-scale folding.
- SDPA/Flash-style dense causal attention for short patch sequences with additive padding mask.
- CUTLASS GEMM plans for residual blocks and MLP.

Stage 5: optional source features.

- Positional embedding and shift-gather.
- `truncate_negative`.
- `return_forecast_on_context`.
- `window_size` decomposition.

Stage 6: wrapper parity.

- Host preprocessing for list/ragged inputs and frequency defaults.
- Optional batch scheduler for many independent time series.

## 12. Parity and validation plan

- Unit tests for `_prepare_4d_attention_mask`: padding-only, causal-only, combined mask, dtype min behavior.
- Unit tests for `_timesfm_masked_mean_std`: all-padded, first valid patch, no patch with three valid points, `pad_val` sentinel.
- Random tensor parity for input residual block and horizon residual block, fp32 tolerance `1e-5`.
- Attention parity for eager math: compare Q scaling, mask addition, fp32 softmax, and output projection.
- Single decoder-layer parity with random weights and fixed padded inputs.
- Tiny native config full forward parity at fp32, then fp16/bf16 if runtime supports reduced precision. Use tolerances similar to Transformers tests: fp32 `1e-5`, fp16/bf16 `1e-3` for full predictions.
- 500M checkpoint smoke parity against the Transformers integration example: mean prediction shape `[3,128]` and known first 64-value slice within `1e-4`.
- Optional SDPA parity against eager for `mean_predictions`, `full_predictions`, and last hidden states.
- Regression tests for rejecting legacy/custom configs unless a translation adapter is implemented.

DinoML tests were intentionally not run for this audit.

## 13. Performance probes

- Preprocessing throughput: list/ragged pad/truncate and optional moving average.
- Decoder core throughput by batch size for `N=64,D=1280,L=50`.
- Attention backend comparison for short sequence lengths: eager, SDPA-like, fused dense causal attention.
- GEMM profile sweep for QKV/O/MLP/residual blocks; most 500M GEMMs are 1280-wide.
- Horizon projection throughput and output rescale bandwidth.
- Batch-size sweep for independent time series, especially small `N` where launch overhead may dominate.
- Context-length/token sweep: tiny `N=4`, 500M `N=64`, dummy long-context `N=512`.
- Optional feature probes: positional embedding shift gather, moving-average conv1d, and `return_forecast_on_context` output materialization.

## 14. Skip/defer list

- Training loss, quantile loss gradients, and dropout.
- `window_size` moving-average decomposition.
- `return_forecast_on_context`.
- `truncate_negative` until core forecast parity is stable.
- Legacy/custom `TimesFMForHF` wrappers using non-native config keys.
- `timesfm2_5` RoPE/continuous quantile head family.
- FlashAttention dispatch until mask compatibility is proven.
- KV cache and generation-controller work; native `timesfm` does not use KV cache.
- Quantization and packed weight formats; no source-coupled quantized storage exists in native `timesfm`.

## 15. Final implementation checklist

- [ ] Parse native `TimesFmConfig` and reject legacy/custom key-only configs.
- [ ] Load `TimesFmModelForPrediction` weights with standard Linear layout.
- [ ] Add host preprocessing for list inputs or require pre-padded tensors for stage 1.
- [ ] Implement patch view and padding/sentinel normalization.
- [ ] Implement masked first-valid-patch mean/std.
- [ ] Implement input residual block.
- [ ] Implement frequency embedding add.
- [ ] Implement causal plus padding additive mask.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement MHA with per-dim query scaling and fp32 softmax parity.
- [ ] Implement decoder MLP `LayerNorm -> Linear -> ReLU -> Linear -> padding mask -> residual`.
- [ ] Implement horizon residual projection, reshape, loc/scale rescale, and last-patch selection.
- [ ] Add QKV packed projection rewrite with Q/K/V split-order test.
- [ ] Add optional query-scale folding rewrite.
- [ ] Add single-layer, tiny-config, and 500M checkpoint parity tests.
- [ ] Add performance probes for decoder core, attention, and GEMM families.
- [ ] Defer optional moving average, context forecasts, and TimesFM 2.5 to separate tasks.
