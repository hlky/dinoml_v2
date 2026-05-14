# OLMo3 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: allenai/Olmo-3-7B-Instruct, allenai/Olmo-3-7B-Think, allenai/Olmo-3-32B-Think
Config source: HF raw config/generation_config plus local Olmo3Config defaults
Source files inspected: modeling_olmo3.py, configuration_olmo3.py, modular_olmo3.py, convert_olmo3_weights_to_hf.py, shared rope/mask/cache/config helpers
Any missing files or assumptions: 32B Instruct and base repos returned 401 Unauthorized; tokenizer files were not needed for neural graph operator coverage
```

`modeling_olmo3.py` and `configuration_olmo3.py` are generated from `modular_olmo3.py`. For runtime parity, inspect generated files; for future Transformers source edits, modular is authoritative.

Primary DinoML target: `Olmo3ForCausalLM` text generation, with decoder prefill and decode. `Olmo3ForSequenceClassification` exists through a generic wrapper and should be deferred unless classification is requested.

## 2. High-level architecture

OLMo3 is a text-only decoder causal LM:

```text
token ids -> token embedding -> N decoder blocks -> final RMSNorm -> sliced LM head -> logits/sampling
```

Each decoder block contains causal self-attention and a SwiGLU MLP. The distinctive parts are:

- Hybrid attention schedule: three sliding-window causal layers followed by one full causal layer.
- Q/K post-projection RMSNorm before RoPE and attention.
- Branch RMSNorm after attention and after MLP, before each residual add.
- Optional GQA: 7B accessible configs are MHA; 32B Think is GQA with 40 Q heads and 8 KV heads.

Independently stageable regions:

- CPU/data pipeline: tokenization, chat template, generation sampling. Not part of the core graph.
- Prefill: embeddings, full prompt masks, RoPE, full/sliding causal attention mix, logits.
- Decode: one or small query lengths with per-layer hybrid KV cache.
- Logits head: can be last-token-only with `logits_to_keep`.

## 3. Important config dimensions

Source defaults in `Olmo3Config` differ from released configs. DinoML should always load checkpoint config values.

| Field | Source default | 7B Instruct / 7B Think | 32B Think |
|---|---:|---:|---:|
| `hidden_size` | 4096 | 4096 | 5120 |
| `num_hidden_layers` | 32 | 32 | 64 |
| `num_attention_heads` | 32 | 32 | 40 |
| `num_key_value_heads` | default to Q heads | 32 | 8 |
| inferred `head_dim` | 128 | 128 | 128 |
| Q projection width | 4096 | 4096 | 5120 |
| K/V projection width | 4096 | 4096 | 1024 |
| attention output width before `o_proj` | 4096 | 4096 | 5120 |
| `intermediate_size` | 11008 | 11008 | 27648 |
| `vocab_size` | 50304 | 100278 | 100278 |
| `max_position_embeddings` | 2048 | 65536 | 65536 |
| `sliding_window` | 4096 | 4096 | 4096 |
| layer schedule | S,S,S,F repeat | 24 S / 8 F | 48 S / 16 F |
| RoPE | default unless configured | YaRN, theta 500000, factor 8 | same |
| `rms_norm_eps` | 1e-5 | 1e-6 | 1e-6 |
| `hidden_act` | silu | silu | silu |
| `attention_bias` | false | false | false |
| checkpoint dtype | config-dependent | bf16 | bf16 |
| `use_cache` in config | true | false | false |

Representative checkpoint sweep:

| Checkpoint | Access | Operator-significant variation |
|---|---|---|
| `allenai/Olmo-3-7B-Instruct` | open | 32 layers, MHA, 100278 vocab, YaRN long context |
| `allenai/Olmo-3-7B-Think` | open | same graph shape as 7B Instruct; generation config differs only operationally |
| `allenai/Olmo-3-32B-Think` | open | 64 layers, GQA 40/8 heads, much larger MLP |
| `allenai/Olmo-3-32B-Instruct` | gated | raw config unavailable; likely same family but must verify before admitting |
| `allenai/Olmo-3-7B`, `allenai/Olmo-3-32B` | gated | base model configs unavailable; verify before assuming post-training variants cover them |

## 3a. Family variation traps

- Do not infer KV width from hidden size. Use explicit `num_key_value_heads` and `head_dim`.
- 32B requires GQA: KV cache uses 8 KV heads, not 40 repeated heads.
- `head_dim` is optional in config but source honors it if present; projection widths are `heads * head_dim`.
- Released configs use legacy `rope_scaling` plus `rope_theta`; `PreTrainedConfig` normalizes these to `rope_parameters`.
- Sliding-window layers and full layers coexist. Cache and masks are per-layer type.
- `use_cache=false` appears in released configs, but source supports `use_cache=True`; DinoML should expose cache as a runtime/generation choice rather than assuming the config default enables it.
- Attention and MLP linears are bias-free for accessible configs, but source supports attention bias via `attention_bias`.
- Residual order is branch-normalized post-attention/post-MLP, not the common pre-norm block.
- Q/K norms make naive fused QKV projection unsafe unless the split and per-branch norms are preserved.
- `lm_head.weight` is listed as tied to embeddings, but config has `tie_word_embeddings=false`; preserve actual checkpoint aliasing/untied status from loaded weights.
- Sequence classification is present but generic; causal LM is the first target.
- No NCHW/NHWC issue: this is text-only rank-2/rank-3 sequence work.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,T] -> [B,T,H]`.
- View/reshape `[B,T,heads*D] -> [B,T,heads,D]`.
- Transpose to attention layout `[B,heads,T,D]`.
- Contiguous/reshape back `[B,T,heads*D]`.
- Slice/index for `logits_to_keep`: last K tokens or tensor-selected positions.
- Optional cache reorder/index-select for beam search.

Neural network primitives:

- Bias-free Linear for Q/K/V/O, MLP gate/up/down, and LM head.
- Optional biased Q/K/V/O if `attention_bias=true`.
- RMSNorm over last dimension, fp32 variance math, weight scale, cast back.
- SiLU and elementwise multiply for SwiGLU.
- Residual add.

Attention primitives:

- Causal self-attention.
- Sliding-window causal self-attention with window 4096 for most layers.
- Full causal self-attention for every fourth layer.
- MHA and GQA/MQA-compatible attention.
- Softmax in fp32 for eager parity.
- Dropout can be fixed to zero for inference.

Position/rotary ops:

- RoPE over full `head_dim` unless `partial_rotary_factor` appears in future configs.
- YaRN RoPE for released configs: theta 500000, factor 8, original max 8192, attention scaling 1.2079441541679836.
- Dynamic RoPE update helper exists in source; first integration can require static/precomputed YaRN parameters for admitted configs.

Generation/cache ops:

- Per-layer KV cache with layer-type-specific storage.
- Full layers: K/V grow with sequence length.
- Sliding layers: persistent cache is bounded to recent tokens; effective attention window includes cached recent tokens plus current query.
- Cache update after RoPE, so cached keys are post-RoPE.
- `logits_to_keep` last-token or selected-token LM head.

Distributed/tensor-parallel metadata:

- Source declares TP plans for q/k/v/o, MLP, and LM head. DinoML can ignore for single-GPU first pass but should not bake in incompatible packed weights.

## 5. Layer/block breakdown

For `H=hidden_size`, `A=num_attention_heads`, `Kvh=num_key_value_heads`, `D=head_dim`, `I=intermediate_size`:

```text
Decoder block, repeated N times:
  residual = x                                      # [B,T,H]

  q = Linear(H -> A*D, bias=attention_bias)(x)
  k = Linear(H -> Kvh*D, bias=attention_bias)(x)
  v = Linear(H -> Kvh*D, bias=attention_bias)(x)
  q = RMSNorm(A*D)(q)
  k = RMSNorm(Kvh*D)(k)
  q = view/transpose(q)                             # [B,A,T,D]
  k = view/transpose(k)                             # [B,Kvh,T,D]
  v = view/transpose(v)                             # [B,Kvh,T,D]
  q, k = RoPE(q, k, cos, sin)
  k, v = cache.update(k, v, layer_idx)              # if cache present
  attn = causal_or_sliding_attention(q, k, v)
  attn = reshape(attn)                              # [B,T,A*D]
  attn = Linear(A*D -> H, bias=attention_bias)(attn)
  x = residual + RMSNorm(H)(attn)

  residual = x
  m = Linear(H -> I, bias=False)(x)                 # gate
  u = Linear(H -> I, bias=False)(x)                 # up
  m = SiLU(m) * u
  m = Linear(I -> H, bias=False)(m)
  x = residual + RMSNorm(H)(m)
```

Final model:

```text
x = RMSNorm(H)(x)
logits = Linear(H -> vocab_size, bias=False)(x[:, slice_indices, :])
```

Concrete projection shapes:

| Variant | Q | K | V | O | MLP gate/up | MLP down | LM head |
|---|---|---|---|---|---|---|---|
| 7B | 4096 -> 4096 | 4096 -> 4096 | 4096 -> 4096 | 4096 -> 4096 | 4096 -> 11008 | 11008 -> 4096 | 4096 -> 100278 |
| 32B | 5120 -> 5120 | 5120 -> 1024 | 5120 -> 1024 | 5120 -> 5120 | 5120 -> 27648 | 27648 -> 5120 | 5120 -> 100278 |

## 6. Attention requirements

Attention is autoregressive self-attention only.

| Requirement | OLMo3 behavior |
|---|---|
| Causal/noncausal | Causal |
| Cross-attention | None |
| MHA/GQA/MQA | MHA for 7B accessible configs; GQA for 32B Think |
| Head counts | 7B: Q=32 KV=32 D=128; 32B: Q=40 KV=8 D=128 |
| Q/K/V widths | Q=`A*D`; K/V=`Kvh*D`; value head dim equals key head dim |
| Masking | Per-layer full causal or sliding causal mask |
| Sliding/local | 3 out of 4 layers use `sliding_window=4096` |
| Packed/varlen | Shared mask helper can inspect packed `position_ids` when no attention mask/cache is present; first inference target can reject packed training-style position IDs |
| RoPE | Applied to Q/K before cache update |
| Cache | Per-layer K/V; cached keys are post-RoPE and unrepeated KV-head count |
| Backend compatibility | Source advertises flash-attn, SDPA, flex attention; eager path repeats KV explicitly |

For GQA, DinoML should keep cache as `[B,Kvh,T,D]` and avoid materializing repeated K/V. If a fallback attention path repeats, it is a semantic fallback, not the desired provider ABI.

Sliding cache nuance: `DynamicSlidingWindowLayer` stores recent K/V only. During decode, the persistent stored previous tokens are bounded, and the attention call receives stored recent tokens plus current query K/V, so the effective local attention window reaches `sliding_window` tokens.

## 7. Position encoding and custom math

Default source RoPE computes inverse frequencies from `rope_parameters["rope_theta"]` and `head_dim`. Released configs use YaRN through the shared RoPE utilities.

RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat([-x2, x1], dim=-1)

def apply_olmo3_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,T,D]
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Cos/sin generation:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = concat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Precompute opportunity: for static admitted configs, precompute `inv_freq` and optionally a cos/sin table up to profiled/max context. Runtime still needs position-id indexing, and decode can generate one position row at a time.

YaRN requirements for released configs:

- Normalize legacy `rope_scaling` and `rope_theta` into one rope parameter dict.
- Use `rope_type="yarn"`.
- Preserve `attention_factor`; do not silently use default scaling.
- Preserve `original_max_position_embeddings=8192` and `factor=8.0`.

## 8. Preprocessing and input packing

Core graph inputs:

- `input_ids`: `[B,T]` int token IDs, mutually exclusive with `inputs_embeds`.
- `inputs_embeds`: `[B,T,H]` optional direct embedding input.
- `attention_mask`: optional 2D padding mask or already prepared dict/4D mask.
- `position_ids`: optional `[B,T]`; if omitted, source creates monotonic positions offset by cache length.
- `past_key_values`: optional cache object.

CPU/data pipeline:

- Tokenizer/chat template and sampling are outside the neural graph.
- Generation configs set temperature/top-p/max tokens and sometimes multiple EOS IDs; these are controller ABI, not model ops.

No multimodal placeholder stitching, image/audio packing, or layout translation is required.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Last-token-only LM head

Source pattern:

```text
hidden_states[:, slice_indices, :] -> lm_head
```

Replacement:

```text
Select last K or explicit token indices -> GEMM(H -> vocab)
```

Preconditions:

- Inference without loss.
- `logits_to_keep` is static int, or tensor indices are validated and supported by gather.
- Output shape is `[B,K,V]` for int K or `[B,len(indices),V]` for tensor indices.

Failure cases: full logits requested (`logits_to_keep=0`), labels/loss requested, unsupported dynamic index tensor.

Parity test: compare full-logits slice against rewritten sliced-head output for K=1 and K>1.

### Rewrite: Q/K/V projection scheduling without unsafe QKV fusion

Source pattern:

```text
q = RMSNorm(q_proj(x)); k = RMSNorm(k_proj(x)); v = v_proj(x)
```

Replacement:

```text
three GEMMs, optional grouped scheduling, q/k RMSNorm, reshape/transpose
```

Preconditions:

- Preserve separate Q/K/V weights and optional biases.
- Apply q_norm over width `A*D` and k_norm over width `Kvh*D` before reshaping.
- Do not pack into one QKV output unless the packed layout preserves q/k norms before RoPE and v remains unnormalized.

Failure cases: assuming ordinary fused QKV split with no intermediate norms; assuming K/V widths equal Q width for GQA.

Parity test: one attention block up to pre-RoPE q/k/v tensors.

### Rewrite: GQA attention provider ABI

Source pattern:

```text
repeat_kv(k, A // Kvh); repeat_kv(v, A // Kvh); matmul attention
```

Replacement:

```text
GQA FlashAttention/SDPA provider using Q heads and KV heads directly
```

Preconditions:

- `A % Kvh == 0`.
- Q shape `[B,A,Tq,D]`, K/V shape `[B,Kvh,Tkv,D]`.
- Cache stores unrepeated K/V.
- Scaling is `D ** -0.5`.
- RoPE already applied to Q/K.

Failure cases: backend cannot represent sliding local causal window; backend requires repeated K/V materialization; mixed full/sliding layers not represented in execution plan.

Parity test: compare eager repeated-KV output and provider output for MHA and 32B-style GQA shapes.

### Rewrite: Sliding/full layer mask specialization

Source pattern:

```text
causal_mask_mapping = {"full_attention": full, "sliding_attention": local}
layer i chooses mask by layer_types[i]
```

Replacement:

```text
compile-time per-layer attention kind with full or local causal provider metadata
```

Preconditions:

- `layer_types` length equals `num_hidden_layers`.
- Only admit `full_attention` and `sliding_attention` for this report.
- Sliding layers have positive integer `sliding_window`.

Failure cases: custom mask functions, block sequence IDs, packed sequence masks, unknown layer type.

Parity test: mask truth table around the window boundary and decode with cache length greater than the window.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
two GEMMs -> fused SiLU/mul -> GEMM
```

Preconditions:

- Activation is `silu`.
- Gate/up/down are bias-free for accessible configs.
- Intermediate width is config-derived.

Failure cases: future config changes `hidden_act`, adds biases, or quantized packed weights require a different load/materialization path.

Parity test: random tensor MLP parity in fp32 and bf16/fp16 tolerances.

### Rewrite: Branch RMSNorm placement

Source pattern:

```text
x = residual + RMSNorm(attn_out)
x = residual + RMSNorm(mlp_out)
```

Replacement:

```text
fused branch RMSNorm + residual add
```

Preconditions:

- Norm is applied to branch output, not residual input.
- RMSNorm uses fp32 variance and config epsilon.
- Residual add happens after norm cast back to input dtype.

Failure cases: transforming to pre-norm, folding norm into surrounding linear without accounting for data-dependent variance.

Parity test: block-level parity with random inputs and isolated branch outputs.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm, including q_norm/k_norm widths and branch/final norms. OLMo3 is norm-heavy.
- GQA/sliding-window causal attention with KV cache. This is mandatory for 32B and long-context decode.
- RoPE + attention prefill/decode. Cached K is post-RoPE, so fusion must preserve cache write order.
- Last-token-only LM head. Vocab is 100278, so avoiding full-sequence logits matters.

Medium priority:

- SwiGLU MLP fusion.
- Grouped Q/K/V GEMM scheduling without unsafe semantic fusion.
- Cache update kernels for hybrid full/sliding layers.
- Bias-free linear + residual/norm scheduling for block throughput.

Lower priority:

- Sequence classification head.
- Beam-search cache reorder.
- Tensor-parallel plan support.
- YaRN dynamic update beyond static admitted configs, unless long context above precomputed tables is required early.

## 11. Runtime staging plan

Stage 1: Config and weight loader admission.

- Parse OLMo3 configs, normalize legacy RoPE fields, derive `head_dim`, projection widths, layer type counts.
- Admit only `full_attention`/`sliding_attention`, `hidden_act=silu`, `attention_bias=false` initially.

Stage 2: Single-block parity.

- Implement embeddings, linears, q/k RMSNorm, RoPE, branch RMSNorm placement, SwiGLU.
- Validate 7B MHA and 32B GQA shapes.

Stage 3: Prefill parity.

- Build full decoder with hybrid masks.
- Start with eager dense attention fallback if needed, then provider-backed full/sliding attention.

Stage 4: Decode with cache.

- Per-layer cache manifest: full layers grow; sliding layers bounded.
- Store post-RoPE K and raw V at KV-head count.

Stage 5: Optimized attention.

- Add GQA provider path with local causal window.
- Avoid materialized repeat_kv in optimized path.

Stage 6: Logits and generation controller.

- Implement `logits_to_keep` for last-token logits.
- Treat sampling/EOS/top-p as controller-side parity.

Stage 7: Production fusions.

- RMSNorm/residual fusion, SwiGLU fusion, precomputed RoPE tables, cache memory planning.

## 12. Parity and validation plan

- Config parsing tests for source defaults, 7B configs, 32B Think config, and gated-config rejection metadata.
- RMSNorm tests with fp32 accumulation and bf16/fp16 cast-back.
- RoPE tests for default and YaRN configs, including position offset from cache length.
- Q/K/V projection tests proving q/k norms happen before reshape/RoPE.
- Attention tests:
  - MHA 7B shape.
  - GQA 32B shape with `A=40`, `Kvh=8`.
  - Full causal mask.
  - Sliding boundary around 4096.
  - Cache decode after context exceeds sliding window.
- Single decoder block parity, then N-layer smoke parity.
- Prefill logits parity for short prompts and long prompt around full/sliding mask differences.
- Decode token parity for greedy one-token and multi-token loops.
- Last-token-only logits parity against full logits sliced after LM head.

Recommended tolerances: fp32 strict-ish (`rtol=1e-4`, `atol=1e-4` for block tests), bf16/fp16 looser (`rtol=5e-2`, `atol=5e-2` for full logits initially), then tighten per fused kernel.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 1k, 4k, 8k, 32k, 64k.
- Decode tokens/sec with cache length below and above 4096.
- Full vs sliding layer attention time split.
- GQA attention backend comparison on 32B shapes.
- KV cache memory by layer type and total model.
- RMSNorm kernel time share across q/k/branch/final norms.
- MLP GEMM plus SwiGLU time for 7B vs 32B.
- Last-token-only LM head vs full-sequence logits.
- RoPE table precompute/indexing vs on-the-fly generation.
- Batch-size sweep for prefill and decode.
- GGUF/dequant provider probes later: dense fallback vs dequant-before-GEMM for linears, especially LM head and MLP weights.

## 14. Skip/defer list

- Training, loss, gradient checkpointing.
- Sequence classification.
- Beam search and cache reorder beyond simple index-select tests.
- Tensor parallel and pipeline parallel execution.
- Quantized/packed weights unless a specific checkpoint requires them.
- Custom mask functions, packed sequence training masks, block sequence IDs.
- Remote-code variants and gated base/32B Instruct configs until accessible.
- General dynamic YaRN update beyond admitted max context if static tables cover target runs.

## 15. Final implementation checklist

- [ ] Parse `Olmo3Config` and normalize `rope_scaling`/`rope_theta` into RoPE parameters.
- [ ] Derive explicit `head_dim`, Q width, KV width, and attention output width.
- [ ] Reject unsupported `layer_types`, activations, or attention bias in the first admission pass.
- [ ] Load embeddings, untied/tied LM head according to actual checkpoint metadata.
- [ ] Implement OLMo3 RMSNorm with fp32 variance.
- [ ] Implement q/k post-projection RMSNorm.
- [ ] Implement default and YaRN RoPE tables/application.
- [ ] Implement hybrid full/sliding causal masks.
- [ ] Implement MHA and GQA attention without materialized KV repeat in optimized path.
- [ ] Implement full and sliding per-layer KV cache, with post-RoPE K storage.
- [ ] Implement SwiGLU MLP.
- [ ] Preserve branch RMSNorm-before-residual ordering.
- [ ] Implement `logits_to_keep` sliced LM head.
- [ ] Add single-block parity tests for 7B and 32B shapes.
- [ ] Add prefill and decode parity tests around sliding-window boundaries.
- [ ] Benchmark prefill, decode, cache memory, RMSNorm, MLP, and LM head slices.

