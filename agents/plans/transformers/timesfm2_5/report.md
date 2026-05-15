# Transformers Audit: `timesfm2_5`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/timesfm-2.5-200m-transformers
Config source: official HF config.json at repo sha 5a9806b9b291fad9233b5249d88263f1846304d3
Source files inspected:
- transformers/src/transformers/models/timesfm2_5/configuration_timesfm2_5.py
- transformers/src/transformers/models/timesfm2_5/modular_timesfm2_5.py
- transformers/src/transformers/models/timesfm2_5/modeling_timesfm2_5.py
- transformers/src/transformers/models/timesfm2_5/convert_timesfm2_5_original_to_hf.py
- transformers/tests/models/timesfm2_5/test_modeling_timesfm2_5.py
Any missing files or assumptions:
- No processor/preprocessor files exist for this family; preprocessing is Python code in the modeling file.
- `modeling_timesfm2_5.py` and `configuration_timesfm2_5.py` are generated from `modular_timesfm2_5.py`. Runtime audit uses generated modeling source; upstream source edits should target modular.
- Only one official native `timesfm2_5` Transformers checkpoint was found. Other open repos are mirrors or original `timesfm`/MLX/dummy variants and are labeled below.
```

Evidence snapshot: `agents/plans/transformers/timesfm2_5/config_sweep.json`.

## 2. High-level architecture

Primary runtime target: time-series forecasting with `TimesFm2_5ModelForPrediction`, returning point/mean forecasts and quantile forecasts. This is not text generation. There is no tokenizer, vocabulary, LM head, or autoregressive decode loop.

Dataflow:

```text
list/rank-2 time series -> CPU/Python padding/truncation/window split -> global RevIN
-> patchify [B,T] to [B,N,patch_length] -> running patch RevIN + padding mask concat
-> residual input projection -> causal decoder-only transformer over patches
-> point and quantile residual projection heads -> patch-local denormalization
-> last patch selection -> continuous quantile adjustment -> global denormalization
-> optional flip-invariance average and nonnegative clamp -> forecasts
```

Independently stageable pieces:

- CPU/data-pipeline: list-of-1D input handling, left padding/truncation to `forecast_context_len`, optional moving-average decomposition, input-min check.
- GPU/runtime preprocessing: rank-2 padded tensor path, global mean/std, patch reshape, Welford running stats across patches, RevIN, mask concat, position ids and causal mask.
- Core neural body: residual input projection, 20 causal self-attention blocks, point/quantile heads.
- Postprocessing: continuous quantile stitch, flip-invariance second pass, nonnegative clamp, optional training loss.

## 3. Important config dimensions

Official `google/timesfm-2.5-200m-transformers` config:

| Field | Value | Source |
|---|---:|---|
| `model_type` | `timesfm2_5` | config.json |
| `architectures` | `TimesFm2_5ModelForPrediction` | config.json |
| `patch_length` | 32 | config.json |
| `context_length` | 16384 | config.json |
| `horizon_length` | 128 | config.json |
| `num_hidden_layers` | 20 | config.json |
| `hidden_size` | 1280 | config.json |
| `intermediate_size` | 1280 | config.json |
| `head_dim` | 80 | config.json/source |
| `num_attention_heads` | 16 | config.json |
| `num_key_value_heads` | 16 | config.json |
| Q/K/V widths | 1280/1280/1280 | source, derived from heads * head_dim |
| MLP activation | `swish` / SiLU | config.json/source |
| MLP/attention bias | `use_bias=false`, `attention_bias=false` | config.json |
| input projection bias | true | source overrides `use_bias=True` |
| quantiles | 9 configured + median slot = 10 outputs | config.json/source |
| `output_quantile_len` | 1024 | config.json |
| `decode_index` | 5 | config.json |
| RoPE | default, theta 10000.0 | config.json |
| dtype | float32 | config.json |
| cache support | no runtime decode cache for first target | source/test comments |

Representative config sweep:

| Repo | Scope | Key variation |
|---|---|---|
| `google/timesfm-2.5-200m-transformers` | official native target | 20 layers, H=1280, heads=16, head_dim=80, patch=32 |
| `machinadeusex/timesfm-2.5-200m-transformers` | mirror | same config blob and safetensors sha256 as official |
| `echo3700/timesfm-2.5-200m-transformers` / `PapaMoth/...` | mirrors | same native config fields as official |
| `google/timesfm-2.5-200m-pytorch` | original conversion input, not this native source | `model_type=timesfm`, `quantile_horizon_length=1024`, no native `timesfm2_5` RoPE/heads fields |
| `Ayushk44/timesfm-2.5-7m-pytorch-dummy` | nonofficial dummy, out of native scope | `hidden_size=256`, 2 layers, `model_type=timesfm` |
| `kunal732/timesfm-2.5-200m-transformers-mlx` | MLX mirror, out of native scope | `model_type=timesfm`, `patch_length=64` |

## 3a. Family variation traps

- Native source reads `num_key_value_heads`; official target is MHA (`16 == 16`), but source supports GQA/MQA if configs change.
- `head_dim` is explicit and should not be inferred blindly. Official `hidden_size == num_heads * head_dim`, but source uses `head_dim` directly.
- `intermediate_size == hidden_size` for official 2.5. This is a plain two-linear MLP, not gated SwiGLU.
- Input projection always has bias because `TimesFm2_5Model` passes `use_bias=True`; other residual blocks follow `config.use_bias`.
- Official config provides `rope_parameters`; source default in `configuration_timesfm2_5.py` is `None`, while modeling indexes `config.rope_parameters["rope_type"]`. DinoML should require/default `{rope_type: default, rope_theta: 10000.0}` rather than accepting absent RoPE metadata silently.
- Generated source has a likely broken optional path: `forward(window_size=...)` calls `self._timesfm_moving_average`, but the generated class defines `_timesfm2_5_moving_average`. First integration should reject `window_size is not None` unless parity against upstream confirms the intended alias.
- `past_values` public head input is a `Sequence[Tensor]`, but the core model consumes padded rank-2 `[B,T]`. DinoML should choose a bounded rank-2 tensor ABI first.
- `force_flip_invariance=True` doubles decoder work by running the model on `normalized_ts` and `-normalized_ts`.
- Continuous quantile head mutates selected quantile channels in a Python loop over configured quantiles; it is postprocessing, not neural body structure.
- The source creates its own causal mask from patch padding. External attention masks from generic Transformers tests are intentionally bypassed.

## 4. Operator coverage checklist

Tensor/layout ops:

- list-to-batch padding/truncation for variable-length 1D time series, first staged outside compiled graph.
- `view`/reshape `[B,T] -> [B,N,32]`, requiring `T % patch_length == 0` after preprocessing.
- slice last patch `[:, -1, ...]`, slice horizon, tensor clone/update, stack, cat, flip over quantile axis, optional `index_select`.
- broadcast `unsqueeze`, expand, elementwise where, maximum, clamp, min/max reductions.

Neural network primitives:

- Residual block: `Linear(input -> hidden) -> swish -> Linear(hidden -> output) + Linear(input -> output)`.
- Input projection: `Linear(64 -> 1280, bias=True)`, `Linear(1280 -> 1280, bias=True)`, residual `Linear(64 -> 1280, bias=True)`.
- Per-layer MLP: `Linear(1280 -> 1280, bias=False) -> swish -> Linear(1280 -> 1280, bias=False)`.
- Output point head: residual block `1280 -> 1280 -> 1280` because `horizon_length * (len(quantiles)+1) = 128 * 10`.
- Output quantile head: residual block `1280 -> 1280 -> 10240` because `output_quantile_len * 10 = 1024 * 10`.
- RMSNorm over last dim for hidden size 1280 and per-head dim 80.

Attention primitives:

- Causal self-attention over patch tokens `[B,N,H]`, official `N = context_length / patch_length = 512`.
- Q/K/V projections: `Linear(1280 -> 1280, bias=False)` each; O projection `Linear(1280 -> 1280, bias=False)`.
- Q/K reshape to `[B,16,N,80]`, RoPE, Q/K RMSNorm over 80, learnable per-dim query scale via `softplus(scale) * 1.442695041 / sqrt(80)`.
- Eager attention requires QK matmul, additive causal/padding mask, fp32 softmax on last dim, AV matmul. SDPA/Flash/Flex are declared supported, but first parity can use eager math.

Position/rotary/custom math:

- Default RoPE with `inv_freq = 1 / theta ** (arange(0, head_dim, 2) / head_dim)`.
- Position ids are patch positions shifted by count of masked patches: `arange(N) - num_masked`.

Time-series preprocessing-coupled ops:

- Welford-style running mean/std per batch across patch chunks with padding mask.
- RevIN normalize/denormalize with safe scale guard and optional mask-zeroing.
- Global mean/std before neural call and global denormalization after forecast.

Optional/deferred:

- Training loss: MSE plus quantile pinball loss.
- Optional moving-average decomposition: `pad` + `conv1d` if `window_size` path is fixed upstream.
- General Sequence-of-Tensor ABI and Python list output reassembly.

## 5. Layer/block breakdown

Core padded input path:

```text
past_values: [B,T], padding: [B,T]
patched_inputs = view([B,N,32])
patched_masks_bool = view([B,N,32]) >= 0.5
for i in 0..N-1:
  count, mean, std = Welford(count, mean, std, patched_inputs[:,i,:], mask[:,i,:])
context_mu/context_sigma = stack([B] per patch) -> [B,N]
normed_inputs = RevIN(patched_inputs, context_mu, context_sigma, mask) -> [B,N,32]
tokenizer_inputs = cat([normed_inputs, mask_float], dim=-1) -> [B,N,64]
input_embeddings = ResidualBlock(64,1280,1280) -> [B,N,1280]
```

Decoder block, repeated 20 times:

```text
residual = x
x = RMSNorm_1280(x)
q = Linear(1280 -> 1280, no bias)(x).view(B,N,16,80).transpose(1,2)
k = Linear(1280 -> 1280, no bias)(x).view(B,N,16,80).transpose(1,2)
v = Linear(1280 -> 1280, no bias)(x).view(B,N,16,80).transpose(1,2)
q,k = RoPE(q,k, cos/sin)
q = RMSNorm_80(q)
k = RMSNorm_80(k)
q = q * softplus(per_dim_scale)[None,None,None,:] * 1.442695041/sqrt(80)
a = causal_attention(q,k,v, mask, scaling=1.0)
x = RMSNorm_1280(Linear(1280 -> 1280, no bias)(a)) + residual
residual = x
x = RMSNorm_1280(x)
x = Linear(1280 -> 1280, no bias)(swish(Linear(1280 -> 1280, no bias)(x)))
x = RMSNorm_1280(x) + residual
```

Prediction heads:

```text
point_output = RevIN(ResidualBlock(1280,1280,1280)(hidden), context_mu, context_sigma, reverse=True)
quant_output = RevIN(ResidualBlock(1280,1280,10240)(hidden), context_mu, context_sigma, reverse=True)
point_forecast = point_output.view(B,N,128,10)[:, -1, :, :]
quantile_spreads = quant_output.view(B,N,1024,10)[:, -1, :, :]
```

## 6. Attention requirements

- Variant: causal self-attention over patch tokens.
- Official shape: `B x 512 x 1280` hidden, Q/K/V as `B x 16 x 512 x 80`.
- GQA/MQA: source supports `num_key_value_heads < num_attention_heads` through `repeat_kv`; official config has no repetition.
- Masking: `create_causal_mask(config, input_embeddings, padding_mask, past_key_values=None)` from patch padding. The padding mask is `1` for valid patches after `~patch_padding`.
- Position encoding: RoPE before Q/K RMSNorm and query per-dim scaling.
- Cache: the primary forecasting path passes `past_key_values=None` and processes full patch prefixes. Although attention code accepts `past_key_values.update(...)`, the model constructs masks with `past_key_values=None`; do not stage decode KV cache first.
- Backend compatibility: source advertises SDPA, FlashAttention, and FlexAttention support. Tests skip generic flash dispatch because forced masks do not fit the generic harness, then test direct eager-vs-SDPA/flash equivalence.
- Attention math order for parity: RoPE -> Q/K RMSNorm -> per-dim softplus query scale -> attention backend with `scaling=1.0` -> output projection.

## 7. Position encoding and custom math

RoPE basis:

```python
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2) / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos, sin = emb.cos(), emb.sin()
```

Application:

```python
def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

TimesFM-specific query scaling:

```python
scale = softplus(per_dim_scale) * (1.442695041 / sqrt(head_dim))
q = q * scale[None, None, None, :]
```

Running stats and RevIN are custom enough to test separately. Welford updates use masked counts, masked means/vars, and `sqrt(clamp(var, min=0))`. RevIN replaces scales below `1e-6` with `1` for normalization, and clamps final context sigma to at least `1e-6` for output scale.

## 8. Preprocessing and input packing

Public head ABI:

- `past_values`: `Sequence[Tensor]`, each 1D. Source truncates each to last `forecast_context_len`, pads shorter series on the left with zeros, and builds a padding vector.
- For a bounded DinoML graph, prefer accepting already padded `input_ts: [B,T]` and `input_padding: [B,T + horizon]` or `[B,T]` with `T` divisible by 32. Keep list handling in CPU/data pipeline first.
- `future_values` is training-only for first inference parity.

Patch packing:

- `patched_inputs = past_values.view(B, -1, 32)`.
- `patched_masks = past_values_padding[:, :seq_len].view(B, -1, 32)`.
- Patch token padding is determined by the last element of each patch: `patch_padding = patched_masks_bool[..., -1]`.
- Position ids shift left by the number of masked patches, so padded leading patches get negative positions.

Forecast postprocessing:

- `force_flip_invariance=True` runs the entire decoder/head twice, flips nonmedian quantiles, and averages signs.
- Continuous quantile head uses `quantile_spreads[..., idx] - quantile_spreads[..., median] + point_median` for nonmedian quantile channels where available.
- `infer_is_positive=True` clamps outputs only if the minimum input value across the request is nonnegative.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ResidualBlock to Fused GEMM Epilogue Chain

Source pattern:

```text
y = Linear_i(x)
y = swish(y)
y = Linear_o(y)
r = Linear_r(x)
out = y + r
```

Replacement:

```text
GEMM(input,input_layer) -> swish -> GEMM(output_layer) + GEMM(residual_layer)
```

Preconditions:

- Static last dimension and dense row-major `[B*N, K]` flattening.
- Bias presence must match each module. Input projection has bias; official transformer MLP/head residual blocks do not, except where source explicitly passes `use_bias=True`.
- Preserve activation exactness (`swish` from `ACT2FN`, normally SiLU).

Failure cases:

- Non-default `activation`, enabled biases, or non-contiguous hidden states.

Parity test sketch:

- Random `[B,N,in]` fp32 and bf16/fp16 tensors; compare block output before and after flattened GEMM lowering.

### Rewrite: Q/K/V Separate Projections to Packed Projection

Source pattern:

```text
q = Linear(H -> q_heads*D)(x)
k = Linear(H -> kv_heads*D)(x)
v = Linear(H -> kv_heads*D)(x)
```

Replacement:

```text
single GEMM H -> (q_heads + 2*kv_heads)*D, split [Q, K, V]
```

Preconditions:

- Same input `x`, same bias policy, compatible dtype/layout, no intervening ops.
- Packed weight rows must be `[Q rows][K rows][V rows]`; conversion script documents original fused QKV split using `chunk(3, dim=0)`.

Failure cases:

- GQA with unequal Q and KV widths still works but split sizes are `[num_heads*D, kv_heads*D, kv_heads*D]`, not equal thirds.

### Rewrite: Patchify View + Input Projection

Source pattern:

```text
[B,T] -> view [B,N,32]
cat(mask) -> [B,N,64]
ResidualBlock(64 -> 1280)
```

Replacement:

```text
bounded patch-pack kernel emits [B,N,64] then flattened GEMM residual block
```

Preconditions:

- `T % 32 == 0`; padding mask rank and dtype validated; patch flatten order is contiguous time order.
- No layout translation needed; this is rank-3 sequence data, not image/video layout.

### Guard: No Layout Translation Across Patch/Head Axes

There is no NCHW/NHWC region. Axis-sensitive ops include `dim=-1` reductions/norms, `transpose(1,2)` for `[B,N,Hd] -> [B,heads,N,D]`, `softmax(dim=-1)`, `flip(dims=(-1,))`, and `view(B,N,horizon,quantiles)`. Any layout pass should preserve semantic axes or mark these regions `no_layout_translation()`.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden size 1280 and head dim 80; used four hidden norms per layer plus Q/K norms.
- Fused QKV projection with split, then RoPE + Q/K RMSNorm + per-dim query scale.
- Causal attention over `N=512` patch tokens, MHA first; GQA repeat guards later.
- ResidualBlock GEMM/activation/add chain for input and output heads.

Medium priority:

- Welford running stats and RevIN over `[B,N,32]`; small patch dimension but latency-visible before every forecast.
- Last-patch-only projection slicing: avoid materializing unused forecast heads for all patches once parity is proven.
- Continuous quantile adjustment as a fused elementwise/update over `[B,horizon,10]`.

Lower priority:

- Optional moving-average `conv1d` path after source alias is resolved.
- Training quantile loss.
- Flash/Flex attention backends beyond eager/SDPA parity.

## 11. Runtime staging plan

1. Parse native `timesfm2_5` config and reject out-of-scope `model_type=timesfm` mirrors.
2. Load official dense safetensors and verify parameter names/shapes for one layer and the three residual heads.
3. Implement a bounded padded-tensor ABI: `past_values [B,T]`, `past_values_padding [B,T]`, with `T % patch_length == 0`.
4. Add standalone parity for Welford stats, RevIN, RoPE, Q/K RMSNorm, and query scaling.
5. Run one decoder block parity using eager dense attention.
6. Run full core model parity on `[B,512,1280]` patch embeddings or padded `[B,16384]` inputs.
7. Add point/quantile head and postprocessing parity without flip invariance.
8. Enable flip-invariance as a graph-level second pass.
9. Optimize QKV packing, RMSNorm, attention, and residual-block GEMM fusions.

Stub initially:

- Sequence-of-Tensor public convenience API.
- `window_size` moving-average path.
- training loss.
- GQA configs not represented by official checkpoint.

## 12. Parity and validation plan

- Custom op tests:
  - Welford update for all-valid, partially masked, all-masked patches.
  - RevIN forward/reverse with small sigma and broadcast loc/scale ranks.
  - RoPE cos/sin and application for negative shifted position ids.
  - per-dim query scale exactness against PyTorch `softplus`.
- Neural tests:
  - `TimesFm2_5ResidualBlock` random input parity for `64 -> 1280 -> 1280`, `1280 -> 1280 -> 1280`, and `1280 -> 1280 -> 10240`.
  - one attention layer eager parity with official dimensions and short `N`.
  - one decoder block parity after all four RMSNorm placements.
  - full model parity for small test config from Transformers tests.
  - official checkpoint inference parity for the integration sine inputs and expected first 64 mean predictions when weights are available.
- Tolerances:
  - fp32: `atol=1e-5` for intermediate blocks, `1e-4` for full forecast parity.
  - bf16/fp16: start at `1e-2` for attention-backed full forecasts, matching upstream test looseness.

DinoML tests were intentionally not run for this audit.

## 13. Performance probes

- CPU/list preprocessing throughput for variable-length sequences.
- Padded tensor preprocessing throughput: global stats, patch Welford, RevIN, mask concat.
- Decoder-only throughput over `B x 512 x 1280`.
- Attention backend comparison: eager GEMM-softmax-GEMM vs SDPA/Flash for causal padded patch masks.
- Batch-size sweep: `B = 1, 4, 16, 64`.
- Context sweep: `T = 1024, 4096, 16384` with `N=T/32`.
- Flip-invariance cost: one pass vs two passes.
- Head cost: all-patch projection vs last-patch-only optimized projection.
- Memory probe for full quantile head temporary `[B,N,10240]`.

## 14. Skip/defer list

- Training and `future_values` loss.
- `window_size` moving-average decomposition until generated source alias is clarified.
- Public list-of-1D `Sequence[Tensor]` ABI inside compiled runtime.
- Nonofficial `timesfm`/MLX/dummy configs.
- GQA/MQA variants not present in official native checkpoint.
- Dynamic/advanced RoPE variants beyond official default.
- Flash/Flex attention as a first backend.
- Any text-generation concepts: vocab, logits, sampling, KV-cache decode, beam search.

## 15. Final implementation checklist

- [ ] Parse `TimesFm2_5Config` and require native `model_type=timesfm2_5`.
- [ ] Require/default `rope_parameters={rope_type: default, rope_theta: 10000.0}`.
- [ ] Load official safetensors and validate dense parameter shapes.
- [ ] Add padded rank-2 time-series inference ABI.
- [ ] Implement patch reshape and padding-mask packing.
- [ ] Implement masked Welford running stats.
- [ ] Implement RevIN normalize/denormalize with safe-scale behavior.
- [ ] Implement residual block lowering and tests.
- [ ] Implement RMSNorm for hidden and per-head dims.
- [ ] Implement default RoPE with shifted patch position ids.
- [ ] Implement MHA attention with TimesFM query scaling.
- [ ] Add one-block and full-core parity tests.
- [ ] Add point and quantile projection heads.
- [ ] Add continuous quantile postprocessing and nonnegative clamp.
- [ ] Add optional flip-invariance second pass.
- [ ] Reject or fix `window_size` path before exposing it.
- [ ] Benchmark preprocessing, decoder, attention backend, and head projection bottlenecks.
