# ERNIE 4.5 audit for DinoML v2

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary dense text target: baidu/ERNIE-4.5-0.3B-PT and baidu/ERNIE-4.5-0.3B-Base-PT.
  Variation references only: baidu/ERNIE-4.5-21B-A3B-Base-PT,
  baidu/ERNIE-4.5-21B-A3B-PT, baidu/ERNIE-4.5-21B-A3B-Thinking,
  baidu/ERNIE-4.5-VL-28B-A3B-PT.

Config source:
  https://huggingface.co/baidu/ERNIE-4.5-0.3B-Base-PT/raw/main/config.json
  https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT/raw/main/config.json
  https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Base-PT/raw/main/config.json
  https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT/raw/main/config.json
  https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking/raw/main/config.json
  https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT/raw/main/config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/ernie4_5/configuration_ernie4_5.py
  X:/H/transformers/src/transformers/models/ernie4_5/modeling_ernie4_5.py
  X:/H/transformers/src/transformers/models/ernie4_5/modular_ernie4_5.py
  X:/H/transformers/src/transformers/models/ernie4_5/convert_ernie4_5_tokenizer.py
  Shared helpers searched: modeling_rope_utils.py, masking_utils.py, cache_utils.py

Any missing files or assumptions:
  This report targets the in-tree dense `model_type="ernie4_5"` implementation.
  Public MoE and VL checkpoints use `ernie4_5_moe` / `ernie4_5_moe_vl` and are
  listed as source gaps for this audit because their model sources are outside
  the requested `models/ernie4_5` directory. No quantized or packed-weight
  source path was present in the inspected dense implementation.
```

Snapshots/notes: `agents/plans/transformers/ernie4_5/_sources/source_notes.md`.

## 2. High-level architecture

ERNIE 4.5 dense text is a decoder-only causal language model with GQA, RMSNorm,
SwiGLU MLPs, full-dimension GLM-style RoPE, optional KV cache, and tied token
embedding / LM-head weights.

```text
Llama-style tokenizer/chat packing
  -> token embedding
  -> repeated pre-norm decoder blocks
  -> final RMSNorm
  -> optional last-token slice
  -> tied lm_head logits
```

Runtime path:

```text
input_ids or inputs_embeds + optional attention_mask/past_key_values
  -> create causal mask and position_ids
  -> prefill/decode decoder stack
  -> logits/sampling
```

Stageable units are tokenizer/chat-template packing, embedding lookup, one
decoder block parity, full prefill, decode with KV cache, and last-token-only
logits. There are no independent vision/audio/projector stages in the dense
source.

## 3. Important config dimensions

Worked example: `baidu/ERNIE-4.5-0.3B-PT`.

| Field | Value | Source |
|---|---:|---|
| model_type | ernie4_5 | config/source |
| architectures | Ernie4_5ForCausalLM | config.json |
| vocab_size | 103424 | config.json |
| hidden_size / H | 1024 | config.json |
| num_hidden_layers / L | 18 | config.json |
| num_attention_heads / A | 16 | config.json |
| num_key_value_heads / KV | 2 | config.json |
| num_key_value_groups | 8 | inferred: `A / KV` |
| head_dim / D | 128 | config.json |
| Q/O attention width | 2048 | inferred: `A * D` |
| K/V width | 256 | inferred: `KV * D` |
| intermediate_size / I | 3072 | config.json |
| activation | silu | config.json/source |
| RMSNorm eps | 1e-5 | config.json |
| max_position_embeddings | 131072 | config.json |
| rope_theta | 500000.0 | config.json/source default |
| rope_scaling | null | sampled dense configs |
| use_bias | false | config.json |
| use_cache | true | config.json |
| tie_word_embeddings | true | config.json/source |
| dtype | bfloat16 | config.json |

Representative checkpoint sweep:

| Model id | Source family | H | L | A | KV | D | Q width | K/V width | I | Vocab | Max pos | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baidu/ERNIE-4.5-0.3B-Base-PT` | dense | 1024 | 18 | 16 | 2 | 128 | 2048 | 256 | 3072 | 103424 | 131072 | base dense target |
| `baidu/ERNIE-4.5-0.3B-PT` | dense | 1024 | 18 | 16 | 2 | 128 | 2048 | 256 | 3072 | 103424 | 131072 | same geometry as base |
| `baidu/ERNIE-4.5-21B-A3B-Base-PT` | MoE gap | 2560 | 28 | 20 | 4 | 128* | 2560 | 512 | 12288 | 103424 | 131072 | `model_type=ernie4_5_moe`; `head_dim` omitted, inferred by current source style |
| `baidu/ERNIE-4.5-21B-A3B-PT` | MoE gap | 2560 | 28 | 20 | 4 | 128* | 2560 | 512 | 12288 | 103424 | 131072 | adds `num_nextn_predict_layers=1` |
| `baidu/ERNIE-4.5-21B-A3B-Thinking` | MoE gap | 2560 | 28 | 20 | 4 | 128* | 2560 | 512 | 12288 | 103424 | 131072 | config forces `_attn_implementation="eager"` and adds fused/top-k MoE fields |
| `baidu/ERNIE-4.5-VL-28B-A3B-PT` | VL MoE gap | 2560 | 28 | 20 | 4 | 128* | 2560 | 512 | 12288 | 103424 | 131072 | adds vision encoder, multimodal tokens, M-RoPE-like config |

`*` For MoE/VL rows, `head_dim` is not explicitly present in the sampled config
and is an inference from `H / A`; the dense source warns DinoML not to rely on
that relation because dense 0.3B has `H != A * D`.

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` in dense 0.3B. Q projection
  expands from 1024 to 2048, and O projection contracts from 2048 to 1024.
- GQA is mandatory for sampled dense checkpoints: `num_key_value_heads=2` and
  `num_attention_heads=16`.
- `head_dim` is a first-class config field and must not be inferred from
  `hidden_size // num_attention_heads` when present.
- Attention/MLP projection bias is controlled by `use_bias`; sampled configs set
  it false, but the source supports biased projections.
- RoPE is not Llama's half-split rotation. It uses even/odd `rotate_half` pairs
  and repeats the first half of cos/sin across adjacent elements.
- Cached K is stored after RoPE. Re-applying RoPE to cached keys during decode
  would be wrong.
- `rope_parameters` is the current config-class field; sampled JSONs use legacy
  `rope_theta` and `rope_scaling`, which `PreTrainedConfig` standardizes.
- `logits_to_keep` can be an integer tail count or a tensor of indices; this is
  a real graph-shape optimization for decode.
- Tied `lm_head.weight` and `model.embed_tokens.weight` must remain one logical
  parameter when `tie_word_embeddings=true`.
- MoE and VL public checkpoints are separate model types with routing, expert,
  vision, multimodal placeholder, and possible M-RoPE requirements not covered
  by the dense source.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding gather: `input_ids [B,S] -> [B,S,H]`.
- Shape/view/transpose/contiguous for attention: `[B,S,width] -> [B,heads,S,D]`.
- Slice/index for `logits_to_keep`: `hidden_states[:, slice_or_indices, :]`.
- Residual adds, elementwise multiply, dtype casts to/from fp32.
- Causal/padding mask construction from `attention_mask`, `position_ids`, and cache lengths.

Neural network primitives:

- RMSNorm over last dim with fp32 variance and learned weight `[H]`.
- Linear Q: `Linear(H -> A*D)`; dense 0.3B is `Linear(1024 -> 2048)`.
- Linear K/V: `Linear(H -> KV*D)`; dense 0.3B is `Linear(1024 -> 256)`.
- Linear O: `Linear(A*D -> H)`; dense 0.3B is `Linear(2048 -> 1024)`.
- SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`; dense 0.3B uses
  `Linear(1024 -> 3072)`, `Linear(1024 -> 3072)`, `Linear(3072 -> 1024)`.
- LM head: `Linear(H -> vocab_size, bias=False)` with tied embedding weight.

Attention primitives:

- Causal self-attention, GQA repeat from KV heads to query heads.
- Matmul score `[B,A,Q,D] @ [B,A,D,K]`, scale by `D ** -0.5`, mask add,
  fp32 softmax, dropout only in training, value matmul.
- Backend dispatch may use eager, SDPA, FlashAttention, or FlexAttention.

Position/rotary ops:

- Float32 inverse-frequency generation from `rope_theta` and `head_dim`.
- Position-dependent cos/sin generation for `[B,S,D]`.
- GLM-style interleaved RoPE on Q and K.
- Dynamic RoPE update hook for non-default rope types; first dense admission can
  require `rope_type="default"` and `rope_scaling=null`.

Generation/cache ops:

- Dynamic KV cache construction when `use_cache=True`.
- Per-layer cache update with K/V tensors shaped before repeat expansion:
  K/V `[B, KV, cached_length, D]`.
- Cache length feeds default `position_ids` and mask sizes.

Preprocessing-coupled ops:

- Llama tokenizer conversion script sets a chat template and special tokens.
- No image/audio/video preprocessing in dense `ernie4_5`.

Quantized/packed weight metadata ops:

- None found in inspected dense source. Quantized checkpoints would need an
  external loader/provider contract, not a source-derived runtime requirement.

Distributed/tensor-parallel ops:

- Config declares default TP plan: Q/K/V/gate/up are column-wise, O/down are
  row-wise, and LM head is column-wise gather output. DinoML can ignore TP for
  single-device parity but must preserve parameter shard semantics if enabled.

## 5. Layer/block breakdown

Decoder block, repeated `L` times:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> A*D, bias=use_bias)(x).view(B,S,A,D).transpose(1,2)
k = Linear(H -> KV*D, bias=use_bias)(x).view(B,S,KV,D).transpose(1,2)
v = Linear(H -> KV*D, bias=use_bias)(x).view(B,S,KV,D).transpose(1,2)
q,k = GLM-style RoPE(q,k, cos[position_ids], sin[position_ids])
k,v = cache.update(k,v, layer_idx) when cache is present
y = causal GQA attention(q,k,v, mask, scale=D**-0.5)
y = Linear(A*D -> H, bias=use_bias)(y)
x = residual + y
residual = x
x = RMSNorm(x)
x = Linear(I -> H)(silu(Linear(H -> I)(x)) * Linear(H -> I)(x))
x = residual + x
```

Final path:

```text
x = final RMSNorm(x)
x = x[:, logits_to_keep, :]  # optional tail/index slice
logits = tied Linear(H -> vocab_size, bias=False)(x)
```

## 6. Attention requirements

- Type: causal self-attention only for the dense source.
- Head form: GQA. Dense 0.3B has 16 query heads, 2 KV heads, 8 repeats, and
  `head_dim=128`.
- Projection widths: Q/O attention width is `num_attention_heads * head_dim`;
  K/V width is `num_key_value_heads * head_dim`.
- Query length and KV length can differ during decode: Q is current token/block,
  K/V include cached prefix plus current tokens.
- Masking: `create_causal_mask` consumes config attention implementation,
  optional 2D padding mask, cache sizes, and position IDs. Eager masks become an
  additive mask before softmax. Flash attention mask path may return `None` for
  all-valid padding.
- Packed/varlen: shared masking code can infer packed sequence segments from
  non-consecutive `position_ids` when no attention mask and no cache are used;
  this is a training/prefill-adjacent trap rather than a first decode target.
- Sliding/local/block sparse attention: none in dense source.
- Relative bias/ALiBi: none.
- KV cache: K/V are stored after projection; K is after RoPE, V is not rotated.
  Cache tensors remain `[B, KV, T, D]` before repeat expansion. Repeat expansion
  is a transient attention computation detail.
- Backend compatibility: source advertises FlashAttention, SDPA, and FlexAttention
  support. DinoML first parity can implement eager math; optimized admission
  should target GQA FlashAttention with pre-RoPE Q/K fused into the attention
  setup or a separate RoPE kernel.

## 7. Position encoding and custom math

RoPE parameter generation for the default dense configs:

```python
def ernie45_inv_freq(head_dim, rope_theta, device):
    i = arange(0, head_dim, 2, dtype=float32, device=device)
    return 1.0 / (rope_theta ** (i / head_dim))
```

Position embeddings are computed in float32:

```python
def ernie45_cos_sin(inv_freq, position_ids, attention_scaling=1.0):
    # inv_freq [D/2], position_ids [B,S]
    freqs = matmul(inv_freq[None, :, None], position_ids[:, None, :].float())
    freqs = transpose(freqs, 1, 2)       # [B,S,D/2]
    emb = concat([freqs, freqs], dim=-1) # [B,S,D]
    return cos(emb) * attention_scaling, sin(emb) * attention_scaling
```

Application uses even/odd pairs:

```python
def rotate_half_even_odd(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return stack([-x2, x1], dim=-1).flatten(-2)

def apply_ernie45_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    cos = repeat_interleave(cos[..., : cos.shape[-1] // 2], 2, dim=-1)
    sin = repeat_interleave(sin[..., : sin.shape[-1] // 2], 2, dim=-1)
    q_out = q.float() * cos + rotate_half_even_odd(q).float() * sin
    k_out = k.float() * cos + rotate_half_even_odd(k).float() * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)
```

Precomputable: `inv_freq` and, for fixed maximum position windows, cos/sin
tables. Dynamic inputs: `position_ids`, cache offset, and any non-default dynamic
RoPE update. First admission should require default RoPE and validate that
`rope_scaling`/`rope_parameters` do not request dynamic scaling.

## 8. Preprocessing and input packing

Dense runtime inputs are text only:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one must be supplied.
- Optional `attention_mask` is a 2D padding mask for shared mask creation.
- Optional `position_ids [B,S]`; if absent, source uses `arange(S) + past_seen_tokens` and unsqueezes to `[1,S]`.
- Tokenizer conversion wraps a Llama tokenizer, sets `add_bos_token=False`,
  `add_prefix_space=False`, `legacy=True`, model max length 131072, a default
  chat template using `User:` / `Assistant:`, and additional mask tokens.

CPU/data-pipeline work: chat template rendering, tokenization, padding/truncation,
and generation-controller sampling. GPU/runtime work: embedding lookup, mask and
position-id use, decoder, logits slice.

No multimodal placeholder/scatter path is present in the dense source. The VL
sampled config has image/video token IDs and vision settings, but that belongs
to `ernie4_5_moe_vl`, not this implementation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed projection

Source pattern:

```text
q = Linear(H -> A*D)(x)
k = Linear(H -> KV*D)(x)
v = Linear(H -> KV*D)(x)
```

Replacement:

```text
packed = Linear(H -> (A + 2*KV) * D)(x)
split packed as [q_rows, k_rows, v_rows]
```

Preconditions:

- Same input tensor, dtype, and batch/sequence layout.
- Same `use_bias` setting for all three projections.
- Weight rows concatenated in exact source order Q, then K, then V.
- Split sizes use explicit config widths, not `hidden_size`.
- No tensor-parallel shard layout is active, or shard metadata is rewritten
  consistently.

Shape equations:

- `q_width = num_attention_heads * head_dim`
- `kv_width = num_key_value_heads * head_dim`
- `packed_width = q_width + 2 * kv_width`

Failure cases:

- Any projection has a different input, bias setting, dtype, quantization
  format, or sharding policy.
- Consumer expects separate materialized Q/K/V tensors for debugging outputs.

Parity test sketch: compare separate and packed projections before RoPE for
random fp32/bf16 tensors and dense checkpoint weights.

### Rewrite: Q/K projection + RoPE fusion

Source pattern:

```text
q_proj/k_proj -> view/transpose -> apply_ernie45_rope
```

Replacement: projection kernel writes `[B, heads, S, D]` and applies
even/odd-pair RoPE for Q and K while writing output.

Preconditions:

- Default ERNIE 4.5 RoPE semantics exactly as above.
- Position IDs available on device and cache offset already included.
- K is written to cache after RoPE.
- V path remains unrotated.

Failure cases:

- Non-default dynamic RoPE scaling, packed sequence semantics that change
  position IDs per batch row, or a backend that expects unrotated cached K.

Parity test sketch: compare projected rotated Q/K and cached K for prefill and
single-token decode.

### Rewrite: GQA attention without materialized repeat

Source pattern:

```text
key = repeat_kv(key, groups)
value = repeat_kv(value, groups)
attention(q, key, value)
```

Replacement: use a GQA-capable attention kernel consuming Q heads and KV heads
directly.

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Attention kernel supports causal mask, padding mask, rectangular decode
  lengths, and scale `head_dim ** -0.5`.
- Cache tensors remain `[B, KV, T, D]`.

Failure cases:

- Kernel requires repeated dense K/V layout, unsupported mask form, or cannot
  preserve fp32 softmax parity where required.

Parity test sketch: eager repeat-KV attention versus GQA kernel over prefill,
decode, padding masks, and varied batch sizes.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement: slice hidden states before GEMM and launch a smaller logits GEMM.

Preconditions:

- `logits_to_keep` is a static positive integer or validated index tensor.
- Loss computation is absent, or labels are sliced consistently.
- Tied weight alias is preserved.

Failure cases:

- Full-sequence logits requested, training loss needs all shifted logits, or
  index tensor is dynamic without an indexed-GEMM/gather plan.

Parity test sketch: compare full logits slice with pre-sliced GEMM for
`logits_to_keep=1`, N-token tail, and tensor indices.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: twice per block plus final norm; fp32 variance and bf16 output need
  exact dtype behavior.
- GQA FlashAttention with KV cache: avoids materialized `repeat_kv` and is the
  main decode/prefill bottleneck.
- Q/K projection + ERNIE RoPE: source RoPE is model-specific and cache-sensitive.
- SwiGLU MLP: gate/up projections, SiLU, multiply, and down projection dominate
  dense block FLOPs.
- Last-token-only logits: vocab is 103424, so decode should avoid full-sequence logits.

Medium priority:

- Packed QKV projection: saves launch overhead and input reads, but must respect
  explicit unequal projection widths.
- Residual add + RMSNorm scheduling: useful memory-bandwidth reduction if norm
  parity is already locked down.
- Tied embedding/LM-head constant alias handling: more correctness than speed,
  but important for artifact-visible weights.

Lower priority:

- Tensor-parallel lowering from the config TP plan.
- Dynamic/non-default RoPE support.
- Training dropout/loss fusion.

## 11. Runtime staging plan

Stage 1: parse dense config, reject `model_type` other than `ernie4_5`, and load
weights with tied embedding/head alias metadata.

Stage 2: implement standalone custom ops for ERNIE RoPE, RMSNorm, repeat-free
GQA attention reference, and `logits_to_keep` slicing.

Stage 3: one-block parity in fp32 and bf16, including `H != A*D`.

Stage 4: full prefill parity with default RoPE and eager attention math.

Stage 5: decode with DynamicCache-compatible ABI: cached K/V `[B, KV, T, D]`,
cached K already rotated, and position IDs offset by cache length.

Stage 6: enable optimized GQA attention and Q/K RoPE fusion behind strict guards.

Stage 7: add optional packed QKV and last-token logits GEMM rewrites.

Stub initially: tokenizer/chat template, sampling controller, training loss,
tensor parallel, MoE/VL variants, and non-default RoPE scaling.

## 12. Parity and validation plan

- Config parser test: assert dense 0.3B geometry yields Q width 2048 and K/V
  width 256.
- RoPE unit tests: compare `rotate_half` even/odd behavior, cos/sin generation,
  and apply function for random `position_ids`, including nonzero cache offset.
- RMSNorm tests: fp32 accumulation, bf16/fp16 output cast, eps `1e-5`.
- Attention tests: eager GQA repeat-KV parity with masks, no mask, prefill, and
  rectangular decode lengths.
- Cache tests: after one prefill then decode token, cached K/V lengths and
  logits match a no-cache full forward within tolerance.
- One-layer parity: copied source weights, random inputs, fp32 tolerance around
  `1e-5` / `1e-4`; bf16 tolerance around `5e-2` for logits after a block.
- Full prefill logits: small sequence/batch, `logits_to_keep=0` and `1`.
- Decode token parity: greedy next-token equality against Transformers for a
  short prompt when using the same dtype/backend tolerance.
- Rewrite parity: packed QKV, fused RoPE, GQA kernel, and last-token logits each
  compared against the unfused graph before enabling together.

No code tests or imports were run during this audit, per request.

## 13. Performance probes

- Prefill throughput sweep over sequence lengths: 128, 512, 2048, 8192, and a
  long-context case near the target deployment bucket.
- Decode tokens/sec sweep over batch sizes and cache lengths.
- KV cache memory: `2 * L * B * KV * T * D * dtype_size`, using post-RoPE K.
- Attention backend comparison: eager reference, SDPA, FlashAttention/GQA kernel.
- RoPE kernel cost: separate RoPE versus fused Q/K projection + RoPE.
- MLP GEMM throughput: gate/up/down projections for `H=1024`, `I=3072`.
- LM-head cost: full-sequence logits versus last-token-only logits over vocab 103424.
- Weight loading/materialization time, especially tied embedding/head alias and
  optional external quantized formats if added later.

## 14. Skip/defer list

- Training, gradient checkpointing, dropout, and loss.
- MoE `ernie4_5_moe` and VL `ernie4_5_moe_vl` checkpoints.
- Quantization and packed-weight loaders not present in dense source.
- Tensor-parallel and pipeline-parallel execution.
- Non-default/dynamic RoPE scaling.
- FlexAttention packed-sequence training masks.
- Speculative decoding and `num_nextn_predict_layers`.
- Chat template/tokenizer conversion beyond documenting input IDs.

## 15. Final implementation checklist

- [ ] Parse dense `Ernie4_5Config` and reject unsupported `model_type`s.
- [ ] Preserve explicit `head_dim`, Q width, K/V width, and O input width.
- [ ] Load tied embedding / LM-head weights as one logical parameter.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement ERNIE 4.5 GLM-style interleaved RoPE.
- [ ] Implement causal GQA attention with cache tensors `[B, KV, T, D]`.
- [ ] Store cached K after RoPE and V unrotated.
- [ ] Implement SwiGLU MLP with optional projection bias.
- [ ] Implement `logits_to_keep` slicing before LM-head GEMM.
- [ ] Add one-block parity tests covering `hidden_size != A * D`.
- [ ] Add prefill logits parity tests.
- [ ] Add decode-with-cache parity tests.
- [ ] Add guarded packed-QKV rewrite.
- [ ] Add guarded Q/K projection + RoPE fusion.
- [ ] Add repeat-free GQA attention lowering.
- [ ] Add last-token-only logits lowering.
- [ ] Benchmark prefill, decode, KV memory, RoPE, MLP, and LM-head paths.
