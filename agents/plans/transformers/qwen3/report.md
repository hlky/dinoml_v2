# Qwen3 Transformers Family Audit

Primary target: `Qwen3ForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen3
Primary task: causal LM prefill/decode/generation
Local source root: transformers
```

Source files inspected:

- `transformers/src/transformers/models/qwen3/configuration_qwen3.py`
- `transformers/src/transformers/models/qwen3/modeling_qwen3.py`
- `transformers/src/transformers/models/qwen3/modular_qwen3.py`
- Cross-checks: `src/transformers/models/qwen2/modeling_qwen2.py`, `src/transformers/models/qwen2/configuration_qwen2.py`, `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`, `src/transformers/models/qwen3_moe/configuration_qwen3_moe.py`, `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3/configuration_qwen3.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3/modeling_qwen3.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3/modular_qwen3.py`

Representative config and generation metadata inspected from Hugging Face raw files:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen3-32B`
- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-4B-Thinking-2507`
- Open tiny/debug mirrors: `llamafactory/tiny-random-qwen3`, `yujiepan/qwen3-tiny-random`, `optimum-intel-internal-testing/tiny-random-qwen3`

Small snapshots used by this report are under `agents/plans/transformers/qwen3/_sources/`: fetched `config.json` / selected `generation_config.json` files plus compact source line-hit text files.

Authoritative source note: `modeling_qwen3.py` is generated from `modular_qwen3.py`; future Transformers source edits should be checked in the modular file. This report uses the generated file for concrete expanded code and the modular file to identify intended inheritance: Qwen3 is Qwen2/Llama-like dense decoder attention with Q/K head RMSNorm added and Gemma-style SwiGLU MLP.

Missing files or assumptions:

- No remote code is required for the audited in-library class.
- Tokenization uses Qwen2 tokenizer metadata in sampled repos; no multimodal processor is consumed by `Qwen3ForCausalLM`.
- Official dense configs sampled use default RoPE with `rope_scaling: null` and inactive sliding window. Source supports generic RoPE utility plumbing and optional sliding-window layers; treat non-default RoPE and active sliding configs as staged admissions.
- `hf-internal-testing/tiny-random-Qwen3ForCausalLM` was gated at fetch time. Open tiny mirrors were used only as debug-shape examples.

## 2. High-level architecture

Qwen3 dense is a text-only decoder-only Transformer:

```text
tokenization/input_ids -> embedding -> N dense decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
prefill: full prompt causal attention + KV cache fill
decode: new token(s) + cache update + last-token logits
```

Stage decomposition:

- CPU/data pipeline: Qwen tokenizer, chat template, special-token handling, optional attention mask, generation sampling controls.
- GPU/runtime prefill: token embedding, shared RoPE cos/sin generation, repeated Qwen3 decoder blocks, full causal GQA attention, final norm, logits.
- GPU/runtime decode: one or more new tokens, position IDs from cache length, Q/K/V projections, Q/K head RMSNorm, RoPE, KV cache append, attention over cache, MLP, final norm, last-token logits.
- Generation controller: `GenerationMixin`; sampled repos provide `temperature`, `top_p`, `top_k`, EOS list, and pad token behavior in `generation_config.json`.

Implemented heads:

- Required for target: `Qwen3ForCausalLM`.
- Optional/deferred: base `Qwen3Model` for hidden-state outputs.
- Deferred for first causal-LM target: sequence classification, token classification, and question answering generic heads.

## 3. Important config dimensions

Source defaults from `Qwen3Config`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 151936 |
| `hidden_size` | 4096 |
| `intermediate_size` | 22016 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | If `None`, post-init sets it to `num_attention_heads`; source default is 32 |
| `head_dim` | Source default 128; attention uses explicit `head_dim` if present |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 32768 default; official sampled configs use 40960 or 262144 |
| `rms_norm_eps` | 1e-6 |
| `use_cache` | true |
| `tie_word_embeddings` | false source default; true for official 0.6B/1.7B/4B and 2507 4B |
| `rope_parameters` / raw `rope_theta` | Source rotary module reads standardized `config.rope_parameters["rope_theta"]`; sampled raw configs use `rope_theta` |
| `attention_bias` | false in source default and sampled official configs |
| `attention_dropout` | 0.0 |
| `use_sliding_window` | false |
| `sliding_window` | Post-init keeps it only when `use_sliding_window=true`; otherwise effective value is `None` |
| `layer_types` | If omitted, full attention for early layers and sliding attention for `i >= max_window_layers` only when effective `sliding_window` is set |

Representative checkpoint sweep. Dimensions are from fetched `config.json`; KV groups are derived.

| Model id | Layers | H | Heads/KV | Head dim | KV groups | MLP I | Vocab | Max pos | RoPE theta | Sliding active | Tied emb | Dtype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `llamafactory/tiny-random-qwen3` | 2 | 16 | 4/4 | 4 | 1 | 64 | 151936 | 32768 | 5e6 | no | true | fp32 metadata |
| `yujiepan/qwen3-tiny-random` | 2 | 64 | 2/1 | 32 | 2 | 128 | 151936 | 40960 | 1e6 | no | true | bf16 metadata |
| `optimum-intel-internal-testing/tiny-random-qwen3` | 2 | 32 | 2/2 | 8 | 1 | 128 | 151936 | 40960 | 1e6 | no | true | fp32 metadata |
| `Qwen/Qwen3-0.6B` | 28 | 1024 | 16/8 | 128 | 2 | 3072 | 151936 | 40960 | 1e6 | no | true | bf16 |
| `Qwen/Qwen3-1.7B` | 28 | 2048 | 16/8 | 128 | 2 | 6144 | 151936 | 40960 | 1e6 | no | true | bf16 |
| `Qwen/Qwen3-4B` | 36 | 2560 | 32/8 | 128 | 4 | 9728 | 151936 | 40960 | 1e6 | no | true | bf16 |
| `Qwen/Qwen3-8B` | 36 | 4096 | 32/8 | 128 | 4 | 12288 | 151936 | 40960 | 1e6 | no | false | bf16 |
| `Qwen/Qwen3-14B` | 40 | 5120 | 40/8 | 128 | 5 | 17408 | 151936 | 40960 | 1e6 | no | false | bf16 |
| `Qwen/Qwen3-32B` | 64 | 5120 | 64/8 | 128 | 8 | 25600 | 151936 | 40960 | 1e6 | no | false | bf16 |
| `Qwen/Qwen3-4B-Instruct-2507` | 36 | 2560 | 32/8 | 128 | 4 | 9728 | 151936 | 262144 | 5e6 | no | true | bf16 |
| `Qwen/Qwen3-4B-Thinking-2507` | 36 | 2560 | 32/8 | 128 | 4 | 9728 | 151936 | 262144 | 5e6 | no | true | bf16 |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| `Qwen3-0.6B`, `Qwen3-8B`, `Qwen3-32B` | `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-4B-Instruct-2507` | `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-4B-Thinking-2507` | `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |

## 3a. Family variation traps

- Qwen3 dense is not Qwen2 with only config changes. Qwen3 applies RMSNorm to Q and K on the per-head dimension immediately after projection reshape and before RoPE. The norm weights have length `head_dim`, not `hidden_size`.
- Qwen3 differs from Qwen2 projection bias defaults. Qwen2 in this pinned source has biased Q/K/V projections and bias-free O projection; Qwen3 gates Q/K/V/O bias on `config.attention_bias`, which is false in sampled official configs.
- Qwen3 differs from Qwen3-MoE by having dense Gemma-style MLP layers only. There is no router, top-k, expert grouping, packed expert tensor, aux loss, or expert-parallel metadata in dense Qwen3.
- Official dense Qwen3 configs use GQA: KV heads are 8 while query heads range from 16 to 64. Cache shape is KV-head count, not query-head count.
- `num_key_value_heads` may equal `num_attention_heads` in tiny/debug configs; avoid assuming GQA is always active.
- `hidden_size == num_attention_heads * head_dim` is true for official sampled configs but not guaranteed by source. One open tiny mirror uses `hidden_size=32`, `num_attention_heads=2`, `head_dim=8`, making attention output width 16 and O projection `16 -> 32`.
- `sliding_window` field presence is not enough. Effective sliding behavior requires `use_sliding_window=true` so post-init does not null the window, then `layer_types` must select `"sliding_attention"` layers.
- `layer_types` is operator-significant. Dense Qwen3 can mix full attention and sliding attention by layer index when enabled.
- Raw configs often contain `rope_theta`, while source reads standardized `rope_parameters`. DinoML config parsing should normalize these fields and preserve unsupported rope types for clear rejection.
- `rope_theta` changes from 1e6 in base dense models to 5e6 in sampled 2507 dense 4B models; max position also jumps from 40960 to 262144.
- `tie_word_embeddings` changes by checkpoint. 0.6B/1.7B/4B and sampled 2507 4B configs tie embeddings; 8B/14B/32B do not.
- Tokenizer configs include vision/tool/FIM marker tokens, but dense `Qwen3ForCausalLM` consumes only token IDs. Do not infer multimodal input tensors from token strings.
- Text layout is `[batch, sequence, hidden]`; attention transposes to `[batch, heads, sequence, head_dim]`. Layout translation is low value and risky around RMSNorm `dim=-1`, RoPE final-dimension pairing, softmax over key dimension, and cache layout.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- Shape/view ops for Q/K/V: `[B,S,out] -> [B,S,heads,D] -> [B,heads,S,D]`.
- `reshape`, `view`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `cat`, slicing/indexing for `logits_to_keep`.
- Residual adds and dtype casts.
- Optional cache append/update or static cache indexed update.

Neural network primitives:

- RMSNorm over hidden axis `[H]`, fp32 variance math, learned weight multiply.
- RMSNorm over Q/K head axis `[D]`, fp32 variance math, learned weight length `head_dim`.
- Bias-optional linears:
  - 0.6B: Q `1024 -> 2048`, K/V `1024 -> 1024`, O `2048 -> 1024`, MLP `1024 -> 3072 -> 1024`, LM `1024 -> 151936`.
  - 4B: Q `2560 -> 4096`, K/V `2560 -> 1024`, O `4096 -> 2560`, MLP `2560 -> 9728 -> 2560`, LM `2560 -> 151936`.
  - 8B: Q `4096 -> 4096`, K/V `4096 -> 1024`, O `4096 -> 4096`, MLP `4096 -> 12288 -> 4096`, LM `4096 -> 151936`.
  - 14B: Q `5120 -> 5120`, K/V `5120 -> 1024`, O `5120 -> 5120`, MLP `5120 -> 17408 -> 5120`, LM `5120 -> 151936`.
  - 32B: Q `5120 -> 8192`, K/V `5120 -> 1024`, O `8192 -> 5120`, MLP `5120 -> 25600 -> 5120`, LM `5120 -> 151936`.
- SiLU and elementwise multiply for SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

Attention primitives:

- Causal self-attention with RoPE.
- MHA/GQA without physical KV repeat in optimized path.
- Eager fallback semantics: repeat KV, QK matmul, scale by `head_dim**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout in training only, matmul with V.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, flex attention, or custom integrations.
- Optional sliding-window causal attention when effective `layer_types[i] == "sliding_attention"`.

Position/rotary ops:

- Default RoPE inverse frequency from `rope_theta` and `head_dim`.
- Apply RoPE to Q and K after Q/K head RMSNorm and before cache update.
- Dynamic/non-default RoPE plumbing exists through shared utilities; official sampled dense configs use default/no scaling.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache` is true and no cache is supplied.
- Cache `get_seq_length()` for default `position_ids`.
- KV cache update per layer after RoPE.
- Static/full and sliding cache semantics through shared cache classes.
- `logits_to_keep`: int or tensor index selection before LM head.

Preprocessing-coupled ops:

- Qwen tokenizer emits `input_ids` and optional `attention_mask`.
- Generation controller must handle EOS list, pad token, temperature, top-p, top-k, and chat-template/thinking-mode conventions outside the compiled core graph.

Distributed/tensor-parallel metadata:

- Config declares TP plan: Q/K/V/gate/up are column-wise, Q/K norms replicated, O/down row-wise, LM head column-wise gather output. First integration can be single-GPU, but the weight loader should preserve logical parameter names.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x0: [B,S,H]
residual = x0
x = RMSNorm_H(x0)
q = Linear_bias_optional(H -> QH*D)(x).view(B,S,QH,D)
k = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D)
v = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D)
q = RMSNorm_D(q).transpose(1, 2)  # [B,QH,S,D]
k = RMSNorm_D(k).transpose(1, 2)  # [B,KVH,S,D]
v = v.transpose(1, 2)             # [B,KVH,S,D]
cos,sin = shared RoPE(position_ids)
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal_or_sliding_GQA_attention(q,k,v,mask,scale=D**-0.5)
x = residual + Linear_bias_optional(QH*D -> H)(attn)

residual = x
x = RMSNorm_H(x)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Model tail:

```text
hidden = final RMSNorm_H(hidden)
slice_indices = slice(-logits_to_keep, None) if logits_to_keep is int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

All sampled official dense configs use `attention_bias=false`, so Q/K/V/O are bias-free for those checkpoints. Source can instantiate biasful attention if admitted by config.

## 6. Attention requirements

Variant: decoder causal self-attention with Qwen3 per-head Q/K RMSNorm, GQA, RoPE, optional per-layer sliding-window masking, and KV cache.

Shapes:

- Hidden input: `[B,S,H]`.
- Query after projection/norm/transpose: `[B,QH,S,D]`.
- Key/value after projection/transpose: `[B,KVH,S,D]`.
- Full cache per layer stores RoPE-applied K and raw V: key `[B,KVH,T,D]`, value `[B,KVH,T,D]`.
- Eager fallback repeats K/V to `[B,QH,T,D]`; optimized DinoML attention should not materialize this repeat.
- Attention output before O projection: `[B,S,QH*D]`, which may differ from `H` if config sets nonstandard `head_dim`.

Representative bf16 cache sizes:

- Qwen3-0.6B/1.7B: per token per layer `2 * 8 * 128 * 2 bytes = 4096 bytes`; 28 layers are about 112 KiB/token/batch element.
- Qwen3-4B/8B: same per-layer 4096 bytes; 36 layers are about 144 KiB/token/batch element.
- Qwen3-14B: same per-layer 4096 bytes; 40 layers are about 160 KiB/token/batch element.
- Qwen3-32B: same per-layer 4096 bytes; 64 layers are about 256 KiB/token/batch element.

Masking and cache:

- `Qwen3Model` builds a mask mapping with `"full_attention"` and, when any layer type is sliding, `"sliding_attention"`.
- Each decoder layer receives `causal_mask_mapping[self.config.layer_types[i]]`.
- Official sampled dense configs have `layer_types: null` in raw JSON and `use_sliding_window=false`, so post-init yields all full attention.
- If active, dynamic sliding layers store only recent window context while cumulative sequence length still drives `position_ids`.
- Default `position_ids` are `arange(S) + past_seen_tokens`, unsqueezed to `[1,S]`.
- Q/K are RoPE-applied before cache update, so cached K is stored post-RoPE.

Math order in eager fallback:

```text
q_proj/k_proj/v_proj -> q_norm/k_norm -> transpose -> RoPE(q,k) -> cache.update
repeat_kv(k/v) -> matmul(q, k^T) * scale -> add mask
softmax(dtype=float32) -> cast to query dtype -> dropout(training only) -> matmul(weights, v)
transpose/reshape -> o_proj
```

Backend compatibility:

- Source advertises FlashAttention, SDPA, flex attention, and generic attention backend support.
- First optimized DinoML target should be native GQA causal attention with KV cache shape `[B,KVH,T,D]` and no physical repeat.
- Attention output weights are optional diagnostics and can be deferred for optimized path.

## 7. Position encoding and custom math

Default RoPE:

```python
dim = config.head_dim or config.hidden_size // config.num_attention_heads
base = config.rope_parameters["rope_theta"]
inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_qwen3_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,S,D]
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunities:

- `inv_freq` is static for default RoPE and each config's `rope_theta`.
- Cos/sin depend on runtime `position_ids`, sequence length, and batch. Prefill computes `[B,S,D]` once and reuses it across all layers.
- Decode can compute only the new position row(s).
- Cached K is post-RoPE, so no cache-time reapplication is needed.
- Official sampled configs use default RoPE with `rope_scaling: null`; non-default rope types should be rejected or separately staged until parity is proven.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Sampled repos use `Qwen2Tokenizer` metadata and Qwen-style special tokens. GPU graph needs `input_ids`, optional `attention_mask`, optional `position_ids`, and optional cache.
- `input_ids` and `inputs_embeds` are mutually exclusive.
- `attention_mask` can be a 2D padding mask or an already prepared backend-compatible mask through shared masking utilities.
- `position_ids` may be omitted; source derives them from sequence length and cache length.
- `logits_to_keep` controls prompt-logit pruning. For generation, use `1` or explicit indices to avoid full-sequence vocab projection.
- Tokenizer configs include `<|vision_start|>`, `<|image_pad|>`, and related marker tokens, but dense Qwen3 has no image/video branch. They are plain token IDs here.

Generation-controller behavior outside compiled graph:

- Base and thinking configs use sampling defaults `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Instruct-2507 configs use `temperature=0.7`, `top_p=0.8`, `top_k=20`.
- EOS list is `[151645, 151643]` and pad token is `151643` in sampled generation configs.
- Chat templates, thinking-mode prompt conventions, tool-call formatting, and sampling processors can remain CPU/controller-side for first graph parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection with Q/K head RMSNorm

Source pattern:

```text
q = q_norm(q_proj(x).view(B,S,QH,D)).transpose(1,2)
k = k_norm(k_proj(x).view(B,S,KVH,D)).transpose(1,2)
v = v_proj(x).view(B,S,KVH,D).transpose(1,2)
```

Replacement pattern:

```text
grouped or packed projection -> split -> per-head RMSNorm on q/k -> transpose
```

Preconditions:

- Same normalized hidden input `x`.
- Bias handling matches `attention_bias`.
- Output widths are asymmetric under GQA: `[QH*D, KVH*D, KVH*D]`.
- Per-head RMSNorm weights are length `D`, applied before RoPE and after projection reshape.

Failure cases:

- Provider cannot expose split outputs before Q/K norm.
- Unsupported biasful attention configs.
- Tensor-parallel sharding plans require separate per-rank layout.

Parity test sketch:

- Compare Q/K/V tensors before RoPE for 0.6B, 4B, 8B, and 32B shapes, including exact q_norm/k_norm behavior.

### Rewrite: native GQA attention with RoPE-before-cache

Source pattern:

```text
RoPE(q,k) -> cache.update(k,v) -> repeat_kv -> attention
```

Replacement pattern:

```text
FusedGQAAttention(q, k, v, cos, sin, cache, mask_policy)
```

Preconditions:

- Causal self-attention.
- KV cache stores post-RoPE K and raw V as `[B,KVH,T,D]`.
- Backend supports representative query/KV groups: 2, 4, 5, and 8.
- Sliding-window policy is either inactive or explicitly supported per layer.

Failure cases:

- Non-default RoPE not admitted.
- Attention output weights requested from an optimized backend.
- Unsupported 4D/custom masks.
- Active sliding-window layer mix without matching cache/mask implementation.

Parity test sketch:

- Prefill then multi-step decode with omitted `position_ids`; compare cache growth, post-RoPE K tensors, and logits.

### Rewrite: dense SwiGLU gate/up fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement pattern:

```text
one packed gate_up GEMM H -> 2I -> split -> fused SiLU*multiply -> down GEMM
```

Preconditions:

- `gate_proj` and `up_proj` share input `x`.
- Both projections are bias-free as in source.
- Activation is exactly `silu`.
- Weight transform preserves split order `[gate, up]`.

Weight transform:

```python
packed_weight = cat([gate_proj.weight, up_proj.weight], dim=0)
```

Failure cases:

- Quantized or sharded weights whose packed layout cannot be represented by the selected provider.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement pattern:

```text
GatherLastTokenOrIndices -> GEMM(H -> vocab)
```

Preconditions:

- Generation only needs last-token logits or explicit prompt-logit subset.
- Labels/loss path disabled.

Failure cases:

- Prompt log-prob evaluation requiring all positions.

### Rewrite: inactive sliding-window elimination

Source pattern:

```text
raw config may mention sliding fields, but effective config has use_sliding_window=false
```

Replacement pattern:

```text
all layers use full_attention; no sliding mask/cache branch is lowered
```

Preconditions:

- Apply `Qwen3Config.__post_init__` semantics or equivalent normalization before graph lowering.

Failure cases:

- Caller supplies explicit `layer_types` with sliding layers or `use_sliding_window=true`.

## 10. Kernel fusion candidates

Highest priority:

- Native GQA attention with KV cache. Physical `repeat_kv` is unacceptable for production prefill/decode.
- Q/K per-head RMSNorm + RoPE + cache write. This is the defining Qwen3 attention delta versus Qwen2 and sits on the decode hot path.
- RMSNorm hidden-axis fused kernel. Each block has two hidden RMSNorms plus final norm.
- Gate/up packed MLP GEMM + SiLU multiply. Dense Qwen3 has no MoE, so MLP is a straightforward high-value dense fusion target.
- Last-token-only LM head. Vocab GEMM is large and should not run for every prompt token during decode.

Medium priority:

- QKV grouped projection with split outputs and Q/K norm hooks.
- Residual add fused into projection/down-projection epilogues where provider support exists.
- Optional sliding-window mask/cache branch for admitted future configs.
- Static cache indexed update for CUDA graph or fixed-shape decode scheduling.

Lower priority:

- Classification/QA/token-classification heads.
- Attention-weight output path.
- Training dropout, loss, and gradient checkpointing.
- Tensor-parallel execution.
- Non-default RoPE variants until a selected checkpoint requires them.

## 11. Runtime staging plan

1. Parse `Qwen3Config`, including explicit `head_dim`, `num_key_value_heads`, effective `layer_types`, effective `sliding_window`, tied embeddings, and standardized RoPE parameters.
2. Load weights for embeddings, attention projections, Q/K head norms, hidden RMSNorms, dense MLP projections, final norm, and LM head. Preserve tied embedding/lm-head aliasing where configured.
3. Implement one-block fp32/bf16 eager parity for a tiny config without cache: RMSNorms, Q/K norms, RoPE, GQA attention, MLP.
4. Implement full prefill parity for tiny and shape-load smoke for 0.6B/4B/8B/32B configs.
5. Add dynamic KV cache decode with post-RoPE K storage and last-token logits.
6. Replace eager attention with native GQA attention for prefill and decode.
7. Add packed gate/up MLP fusion and fused RMSNorm kernels.
8. Add optional static cache for production decode scheduling.
9. Add sliding-window per-layer support only after an active sliding config is selected.

Initially stub/defer:

- Sampling/chat templates can run outside DinoML.
- Classification, token-classification, and QA heads can be deferred.
- Tensor parallel/pipeline parallel can be deferred.
- Non-default RoPE can be rejected with a clear admission error until targeted.

## 12. Parity and validation plan

Custom op tests:

- Hidden RMSNorm and head-dim RMSNorm with eps `1e-6`, fp32 variance, fp16/bf16 storage.
- Default RoPE with `rope_theta` values `1e6` and `5e6`.
- Q/K norm + RoPE order regression: prove norm happens before RoPE and cache stores post-RoPE K.
- GQA attention for `QH/KVH = 16/8`, `32/8`, `40/8`, and `64/8`.
- Dense SwiGLU MLP parity for `H/I` pairs `1024/3072`, `2560/9728`, `4096/12288`, `5120/25600`.
- `logits_to_keep` int and tensor-index behavior.

Model tests:

- Single decoder layer parity on an open tiny Qwen3 config.
- Two-layer tiny prefill logits parity with `logits_to_keep=0`.
- Decode parity token by token with omitted `position_ids` and `DynamicCache`.
- `logits_to_keep=1` parity against full logits sliced to the final token.
- Tied and untied LM-head load tests.
- Config admission tests for inactive sliding window, explicit `head_dim`, default RoPE theta changes, and unsupported non-default RoPE rejection.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 full block/logits: start with `rtol=2e-2`, `atol=2e-2`; tighten per op where math order is identical.
- Optimized attention may need separate tolerances due to softmax accumulation order.

End-to-end:

- Raw token ID generation with deterministic sampling disabled or fixed greedy settings.
- Instruct/thinking prompt formatting and sampling can be controller-level tests after graph parity.

## 13. Performance probes

- Prefill tokens/sec by model scale and sequence length: 128, 2048, 8192, 40960, and 262144 where memory allows.
- Decode tokens/sec by batch size and cache length for 0.6B, 4B, 8B, and 32B.
- KV cache memory and bandwidth, verifying storage is `[B,8,T,128]` per layer for official dense scales.
- Attention backend comparison: eager repeat-KV, SDPA/FlashAttention-equivalent, DinoML native GQA.
- Q/K norm + RoPE fusion bandwidth impact.
- MLP throughput: separate gate/up versus packed gate_up; fused versus unfused SiLU multiply.
- LM-head cost: full prompt logits versus last-token-only logits.
- Weight-load time and memory for tied versus untied embeddings.
- Sliding-window synthetic benchmark only if active sliding layers are admitted.

Benchmark observations: none collected. These are source/config-derived probes.

## 14. Skip/defer list

Safe to defer for first causal LM integration:

- Training, labels/loss, dropout, and gradient checkpointing.
- Returning attentions and hidden states from optimized paths.
- Sequence classification, token classification, and question answering heads.
- Beam search and speculative decoding; standard logits are enough initially.
- Tensor parallel and pipeline parallel execution.
- Non-default RoPE variants and active sliding-window configs until targeted.
- Multimodal use of vision special tokens; this model family is text-only.

Do not defer:

- Q/K head RMSNorm before RoPE.
- GQA cache shape with KV heads only.
- RoPE-before-cache-update order.
- Effective config handling for inactive versus active sliding windows.
- Tied/untied embedding differences across checkpoints.
- Difference from Qwen3-MoE: no router or expert path in dense Qwen3.

## 15. Final implementation checklist

- [ ] Parse `Qwen3Config` and effective post-init fields.
- [ ] Normalize raw `rope_theta` into supported RoPE parameters; admit default RoPE first.
- [ ] Preserve explicit `head_dim`; do not assume `hidden_size == num_attention_heads * head_dim`.
- [ ] Load embeddings, attention projections, Q/K head norms, RMSNorms, MLP weights, final norm, and LM head.
- [ ] Support tied and untied `lm_head.weight`.
- [ ] Implement hidden-axis RMSNorm and head-dim Q/K RMSNorm.
- [ ] Implement bias-optional Q/K/V/O projections with asymmetric GQA outputs.
- [ ] Implement default RoPE and apply it after Q/K norm.
- [ ] Implement KV cache as `[B,KVH,T,D]` with post-RoPE keys.
- [ ] Implement native GQA causal attention without physical KV repeat.
- [ ] Implement optional per-layer sliding-window mask/cache admission.
- [ ] Implement dense SwiGLU MLP, preferably with packed gate/up fusion.
- [ ] Implement final RMSNorm, `logits_to_keep`, and LM head.
- [ ] Add one-block and tiny full-model parity tests.
- [ ] Add prefill/decode cache parity tests.
- [ ] Add config admission tests for 0.6B, 4B, 8B, 32B, 2507 long-context, and tiny mirrors.
- [ ] Benchmark attention, Q/K norm + RoPE, MLP, KV memory, and last-token logits.
