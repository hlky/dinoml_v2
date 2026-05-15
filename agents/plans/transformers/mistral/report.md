# Transformers Mistral Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: mistralai/Mistral-7B-v0.1.
  Additional sizing references: mistralai/Mistral-7B-Instruct-v0.2,
  mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mistral-Nemo-Base-2407,
  mistralai/Ministral-8B-Instruct-2410.

Config source:
  https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json
  https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/tokenizer_config.json
  Additional configs fetched from the Hugging Face repos listed above.
  HF plugin metadata confirmed Mistral-family repos as text-generation
  `mistral` architecture models; some current Mistral repos are tagged for vLLM
  rather than transformers in repo metadata, but standard configs are readable.

Source files inspected:
  transformers/src/transformers/models/mistral/modeling_mistral.py
  transformers/src/transformers/models/mistral/modular_mistral.py
  transformers/src/transformers/models/mistral/configuration_mistral.py
  transformers/src/transformers/modeling_rope_utils.py
  Tokenizer behavior inferred from LlamaTokenizer inheritance/config because
  the mistral directory has no dedicated tokenization_mistral.py.

Any missing files or assumptions:
  No remote-code files are required for standard Mistral. The generated
  `modeling_mistral.py` says it is produced from `modular_mistral.py`; both
  were inspected. The report assumes inference-only CUDA GPU execution and
  prioritizes decoder-only text generation with KV cache.
```

## 2. High-level architecture

Mistral is a text-only decoder transformer closely related to Llama, with RMSNorm, RoPE, grouped-query attention, and a SiLU-gated MLP. The main source-level difference from Llama is sliding-window causal masking support: `MistralModel.forward` chooses `create_sliding_window_causal_mask` when `config.sliding_window` is not `None`, and attention backends receive a `sliding_window` kwarg.

```text
Llama-style/BPE tokenization
  -> token embedding
  -> decoder-only prefill/decode stack with RoPE, GQA, optional sliding-window mask, KV cache
  -> final RMSNorm
  -> lm_head logits, optionally logits_to_keep
  -> sampling
```

## 3. Important config dimensions

Worked example: `mistralai/Mistral-7B-v0.1`.

| Field | Mistral-7B-v0.1 value | Source |
|---|---:|---|
| architecture | MistralForCausalLM | HF repo/config |
| vocab_size / V | 32000 | config.json |
| hidden_size / H | 4096 | config.json |
| intermediate_size / I | 14336 | config.json |
| num_hidden_layers | 32 | config.json |
| num_attention_heads / A | 32 | config.json |
| num_key_value_heads / KvH | 8 | config.json |
| head_dim / D | 128 inferred | `H // A`; config omits `head_dim` |
| GQA group size / G | 4 | `A // KvH` |
| max_position_embeddings | 32768 | config.json |
| sliding_window | 4096 | config.json |
| rope_theta | 10000 | config.json |
| hidden_act | silu | config.json |
| rms_norm_eps | 1e-5 | config.json |
| tie_word_embeddings | false | config.json |
| torch_dtype | bfloat16 | config metadata |
| cache support | true inferred | MistralConfig default when omitted |

Representative checkpoint sweep:

| Checkpoint | H | I | layers | A | KvH | D | V | max pos | sliding_window | rope_theta | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mistralai/Mistral-7B-v0.1 | 4096 | 14336 | 32 | 32 | 8 | 128 inferred | 32000 | 32768 | 4096 | 10000 | bf16 |
| mistralai/Mistral-7B-Instruct-v0.2 | 4096 | 14336 | 32 | 32 | 8 | 128 inferred | 32000 | 32768 | null | 1000000 | bf16 |
| mistralai/Mistral-7B-Instruct-v0.3 | 4096 | 14336 | 32 | 32 | 8 | 128 inferred | 32768 | 32768 | null | 1000000 | bf16 |
| mistralai/Mistral-Nemo-Base-2407 | 5120 | 14336 | 40 | 32 | 8 | 128 explicit | 131072 | 131072 | null | 1000000 | bf16 |
| mistralai/Ministral-8B-Instruct-2410 | 4096 | 12288 | 36 | 32 | 8 | 128 explicit | 131072 | 32768 | 32768 | 100000000 | bf16 |

## 3a. Family variation traps

- Mistral is GQA by default: `num_key_value_heads=8` while `num_attention_heads=32`, so K/V width is `1024`, not `4096`, for 7B-class configs.
- `sliding_window` is not stable across checkpoints. v0.1 uses `4096`, later 7B instruct configs have `null`, and Ministral uses `32768`.
- Some Mistral-family configs include `layer_types` for alternating full/sliding attention. `MistralConfig.__post_init__` warns that such models should use AutoModel or Ministral classes instead. The plain `mistral` implementation inspected here does not branch per layer type.
- Vocab size changes from 32000 to 32768 to 131072 across the sampled repos, so LM head and tokenizer assumptions must be checkpoint-specific.
- RoPE theta varies widely: 10000, 1000000, and 100000000 in sampled configs.
- Nemo uses `H=5120`, 40 layers, 131K context, and 131K vocab; it is a different performance shape than 7B even with the same `mistral` model type.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,T] -> hidden[B,T,H]`.
- Projection reshapes:
  - Q `[B,T,A*D] -> [B,A,T,D]`.
  - K/V `[B,T,KvH*D] -> [B,KvH,T,D]`.
- Transpose/contiguous after attention.
- Optional K/V repeat fallback: `[B,KvH,T,D] -> [B,A,T,D]`.
- Causal/sliding-window additive mask construction.
- Slice/select hidden states for `logits_to_keep`.

### Neural network primitives

- Bias-free Linear only in inspected source:
  - 7B Q: `4096 -> 4096`; K/V: `4096 -> 1024`; O: `4096 -> 4096`.
  - 7B MLP gate/up: `4096 -> 14336`; down: `14336 -> 4096`.
  - Nemo LM head: `5120 -> 131072`.
- RMSNorm with fp32 accumulation and scale only.
- SiLU activation and elementwise multiply for gated MLP.
- Residual adds after attention and MLP.
- Dropout is present in attention but zero during inference.

### Attention primitives

- Causal decoder self-attention.
- GQA with native K/V head count smaller than Q head count.
- Optional sliding-window local causal attention through mask function and backend kwarg.
- RoPE applied to Q and K before cache update.
- Eager fallback: repeat_kv + matmul + additive mask + fp32 softmax + matmul.
- Optimized backend dispatch through `ALL_ATTENTION_FUNCTIONS`; model declares flash-attn, SDPA, and flex-attn support.

### Position/rotary ops

- Default RoPE inverse frequencies using `rope_theta` and `head_dim`.
- Current source supports non-default RoPE via `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`, though sampled configs use null `rope_scaling`/`rope_parameters`.
- Position IDs default to `arange(T) + past_seen_tokens`.
- Cos/sin computed in fp32 and cast to activation dtype.

### Generation/cache ops

- `DynamicCache(config=self.config)` allocation when use_cache is enabled.
- Per-layer K/V append; K is cached after RoPE.
- Cache tensor shape before repeat: `[B,KvH,past,D]`.
- Eager repeated K/V shape for attention math: `[B,A,past,D]`.
- For sliding-window decode, the logical mask limits attention to the recent window; the inspected DynamicCache call still stores updated K/V through the shared cache abstraction.
- `logits_to_keep` avoids full sequence LM head work.

### Preprocessing-coupled ops

- Mistral 7B uses LlamaTokenizer-style BOS/EOS defaults: BOS `<s>`, EOS `</s>`, add BOS true, add EOS false, no pad token.
- Instruct variants can include chat templates; this is data-pipeline work.
- Nemo uses a fast tokenizer with much larger vocab.

### Distributed/tensor-parallel ops

- Config declares the same TP plan shape as Llama: colwise Q/K/V/gate/up, rowwise O/down, colwise gather for LM head.
- Single-GPU parity can defer TP, but Nemo-scale LM head and 131K vocab make sharded output projection important.

## 5. Layer/block breakdown

Decoder block, repeated `N` times:

```text
residual = x                                           # [B,T,H]
y = RMSNorm(x)
q = Linear(H -> A*D, bias=False)(y)                    # [B,A,T,D]
k = Linear(H -> KvH*D, bias=False)(y)                  # [B,KvH,T,D]
v = Linear(H -> KvH*D, bias=False)(y)                  # [B,KvH,T,D]
cos,sin = rotary(position_ids)                         # [B,T,D]
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(layer,k,v) if cache enabled
mask = causal_mask or sliding_window_causal_mask
attn = GQAAttention(q,k,v,mask,scale=D**-0.5,sliding_window)
x = residual + Linear(A*D -> H, bias=False)(attn)

residual = x
y = RMSNorm(x)
gate = SiLU(Linear(H -> I, bias=False)(y))
up = Linear(H -> I, bias=False)(y)
x = residual + Linear(I -> H, bias=False)(gate * up)
```

Model head:

```text
x = embed_tokens(input_ids)
position_ids = arange(T) + cache_length
position_embeddings = rotary_emb(x, position_ids)
x = decoder_layers(x, mask, position_embeddings, cache)
x = RMSNorm(x)
logits = Linear(H -> V, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

- Causal self-attention only; no encoder/cross attention.
- GQA: `A=32`, `KvH=8`, `G=4` for all sampled Mistral configs.
- Head dim is explicit or inferred; sampled configs use `D=128`.
- Cache stores K/V as `[B,KvH,K,D]` after RoPE. Eager fallback repeats to `[B,A,K,D]` before score matmul.
- Masking is either full causal or sliding-window causal depending on `config.sliding_window`.
- Backend receives `sliding_window=getattr(config, "sliding_window", None)`.
- FlashAttention/SDPA/flex-attn compatibility: backend must support causal GQA and, for v0.1/Ministral-like configs, local/sliding-window attention. A backend that ignores `sliding_window` is only safe for configs where it is `None`.
- Packed/varlen support is an optimization; source path uses dense masks.
- Eager fallback is too slow for production and can materialize repeated K/V.

## 7. Position encoding and custom math

Mistral RoPE math matches the Llama-style implementation in the inspected source.

```python
def mistral_inv_freq(head_dim, rope_theta):
    i = arange(0, head_dim, 2, dtype=float32)
    return 1.0 / (rope_theta ** (i / head_dim))

def apply_mistral_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

Precompute:

- `inv_freq` is static for sampled configs.
- Cos/sin can be precomputed per `(rope_theta, max_position, D, dtype)` if memory is acceptable.
- Sliding-window masks can be generated from `position_ids`, cache length, and window size.

Dynamic:

- `position_ids` depends on cache length.
- Dynamic RoPE variants are source-supported but not required by sampled configs.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenization and optional chat-template rendering.
- BOS insertion and prompt formatting.
- Left padding/attention mask preparation when batching.

GPU/runtime work:

- Accept `input_ids[B,T]`, optional `attention_mask[B,T]`, optional `position_ids[B,T]`, optional cache.
- Build full causal or sliding-window causal masks depending on config.
- Prefill may use `logits_to_keep=1`.
- Decode usually has `T=1` and cache-derived position.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MistralRMSNorm -> RMSNormScaleOnly

Preconditions:

- Exact fp32 accumulation RMSNorm, last axis, scale only.
- Weight shape `[H]`, no bias.

Replacement:

```text
RMSNorm(x, weight, eps, axis=-1, fp32_accum=True)
```

Failure cases:

- Do not rewrite LayerNorm or norm variants with bias/mean subtraction.

Parity test sketch:

- Test `H={4096,5120}` and eps `1e-5` against HF.

### Rewrite: bias-free Linear -> GEMM_RCR

Preconditions:

- Source projection has `bias=False`.
- Dense row-major activation and weight `[N,K]`.

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR -> Reshape
```

Failure cases:

- Future custom classes with biases need a bias epilogue.
- TP shards need shard-aware lowering.

Parity test sketch:

- Compare Q/K/V/O, gate/up/down, and LM head projections.

### Rewrite: QKV grouped projection with unequal widths

Preconditions:

- Same normalized input feeds q/k/v.
- Bias-free projections.
- Output width is `(A + 2*KvH) * D`.

Replacement:

```text
ConcatenatedLinear(H -> (A + 2*KvH)*D) -> Split(q,k,v)
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
```

Failure cases:

- Do not assume q/k/v output widths match.
- Quantized block layouts may make grouped GEMM safer than concatenation.

Parity test sketch:

- Verify split q/k/v before RoPE for 7B and Nemo shapes.

### Rewrite: sliding-window mask -> local attention backend

Preconditions:

- `config.sliding_window is not None`.
- Backend implements causal local attention where key positions older than the window are masked.
- Cache/mask semantics agree with HF `create_sliding_window_causal_mask`.

Replacement:

```text
FullMaskAttention(q,k,v,additive_mask) -> LocalCausalAttention(q,k,v,window)
```

Failure cases:

- Unsafe when `sliding_window=None`.
- Unsafe for alternating layer-type configs unless per-layer attention type is represented.

Parity test sketch:

- Compare full mask output to local backend for prompts shorter, equal, and longer than window.

### Rewrite: eager repeat_kv -> native GQA attention

Preconditions:

- `A % KvH == 0`.
- Backend uses the same KV grouping order as HF `repeat_kv`.

Replacement:

```text
GQAAttention(q[B,A,Q,D], k/v[B,KvH,K,D], group_size=A/KvH)
```

Failure cases:

- Materializing repeated K/V loses memory benefits.

Parity test sketch:

- Compare native GQA to eager repeat for `G=4`.

### Rewrite: logits_to_keep -> sliced LM head

Preconditions:

- Generation does not need all sequence logits.
- Loss is not being computed.

Replacement:

```text
SliceHidden -> LMHeadGEMM
```

Failure cases:

- Full-token scoring or training requires all logits.

Parity test sketch:

- Compare `logits_to_keep=1` to full logits sliced at the end.

## 10. Kernel fusion candidates

Highest priority:

- Native GQA FlashAttention/paged attention with optional sliding-window mode.
- RMSNorm.
- RoPE application fused with Q/K layout path.
- SwiGLU/SiLU activation multiply between gate/up/down GEMMs.
- Last-token-only LM head, especially for 131K vocab Nemo/Ministral.

Medium priority:

- QKV grouped projection with unequal Q and KV widths.
- Sliding-window mask/local attention lowering.
- KV cache append/packing for `[B,KvH,T,D]`.
- Sharded LM head/top-k path for large vocab.

Lower priority:

- Non-default RoPE scaling variants.
- Classification/QA/token heads.
- Alternating full/sliding layer-type support in the plain `mistral` report; handle through a `ministral`-specific report if targeted.

## 11. Runtime staging plan

Stage 1: Parse MistralConfig and normalize defaults (`head_dim`, `use_cache`, old RoPE fields).

Stage 2: Load embeddings, decoder layers, RMSNorm, LM head.

Stage 3: Implement one-layer GQA parity without cache.

Stage 4: Add RoPE and default full causal prefill parity.

Stage 5: Add sliding-window mask parity for `Mistral-7B-v0.1`.

Stage 6: Add cached decode; ensure cached K/V is pre-repeat and post-RoPE.

Stage 7: Enable native GQA optimized attention and local attention when required.

Stage 8: Scale to Nemo/large-vocab LM head and long-context probes.

## 12. Parity and validation plan

- RMSNorm random tensor parity.
- RoPE cos/sin and rotated Q/K parity for theta `10000`, `1000000`, and `100000000`.
- GQA repeat parity for `A=32`, `KvH=8`.
- Sliding-window mask parity for prompt lengths below and above 4096; use small synthetic windows for cheap unit tests.
- Single decoder layer parity with and without sliding-window mask.
- Prefill logits parity with `logits_to_keep=1`.
- Cached decode parity for several tokens and cache shapes.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; bf16/fp16 `rtol=2e-2, atol=2e-2` initially.

## 13. Performance probes

- Prefill throughput for full causal vs sliding-window attention.
- Decode tokens/sec with GQA cache.
- KV cache memory: `2 * layers * B * KvH * T * D * dtype_bytes`.
- LM head cost for 32K vs 131K vocab.
- RoPE table generation vs cached table.
- Sliding-window length sweep.
- Long-context sweep for 32K and 131K context configs.

## 14. Skip/defer list

- Training/loss.
- Dropout and gradient checkpointing.
- Classification/QA/token heads.
- Beam search, speculative decoding, and continuous batching.
- Multi-GPU TP/PP for first parity.
- Quantization-specific paths.
- Per-layer alternating attention for Ministral unless that family is explicitly targeted.

## 15. Final implementation checklist

- [ ] Parse MistralConfig with `head_dim`, `num_key_value_heads`, and `sliding_window`.
- [ ] Normalize older RoPE fields and config defaults.
- [ ] Load embedding, layer, norm, and LM head weights.
- [ ] Implement RMSNormScaleOnly.
- [ ] Implement default RoPE and apply_rope.
- [ ] Implement bias-free GEMM lowering for projections and MLP.
- [ ] Implement GQA attention or safe repeat_kv fallback.
- [ ] Implement sliding-window causal mask/local attention.
- [ ] Implement DynamicCache append with pre-repeat K/V layout.
- [ ] Implement `logits_to_keep`.
- [ ] Add one-layer GQA parity.
- [ ] Add sliding-window parity tests.
- [ ] Add prefill and cached decode parity.
- [ ] Benchmark GQA prefill, decode, sliding-window attention, LM head, and KV memory.
