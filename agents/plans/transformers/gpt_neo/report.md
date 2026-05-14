# Transformers audit: `gpt_neo`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: EleutherAI/gpt-neo-125m, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
Config source: Hugging Face config.json plus local GPTNeoConfig defaults
Source files inspected: modeling_gpt_neo.py, configuration_gpt_neo.py, activations.py, cache_utils.py, masking_utils.py
Any missing files or assumptions: no model execution; tiny private config not used as authoritative; tokenizer coupling inspected from configs/source only
```

Primary target: `GPTNeoForCausalLM` autoregressive text generation on CUDA. Base model and classification/QA heads are secondary and can be deferred for first runtime parity.

Source links:
- Transformers source at commit: https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt_neo
- 125M config: https://huggingface.co/EleutherAI/gpt-neo-125m/blob/8741c104a4aae84316aa969cf1818c75ca44473e/config.json
- 1.3B config: https://huggingface.co/EleutherAI/gpt-neo-1.3B/raw/main/config.json
- 2.7B config: https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/5e1629a69b40344d3ba97e10662ef6593c5829f7/config.json

## 2. High-level architecture

GPT-Neo is a text-only decoder with learned token embeddings, learned absolute position embeddings, alternating global/local causal self-attention blocks, ungated GELU MLPs, final LayerNorm, and a tied LM projection.

```text
tokenize -> input_ids/attention_mask -> token+position(+token_type) embeddings -> decoder prefill -> decode with KV cache -> lm_head -> logits/sampling
```

Stage decomposition:
- CPU/data pipeline: GPT-2 byte-level BPE tokenizer, attention mask construction, generation controller sampling/beam logic.
- GPU/runtime graph: embeddings, decoder blocks, final norm, selected logits projection.
- Cacheable runtime state: per-layer autoregressive self-attention K/V tensors. Local layers still require source-parity handling because the current GPT-Neo config does not map `window_size` into shared `sliding_window` cache metadata.

## 3. Important config dimensions

| Field | Source/default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 50257 | Token embedding rows and LM head rows |
| `max_position_embeddings` | 2048 | Learned absolute position table and mask buffer extent |
| `hidden_size` | 768 / 2048 / 2560 | Residual width |
| `num_layers` | 12 / 24 / 32 | Decoder block count |
| `num_heads` | 12 / 16 / 20 | MHA heads |
| `head_dim` | derived | `hidden_size // num_heads`; source rejects non-divisible configs |
| `intermediate_size` | often `null` | Effective MLP width is `4 * hidden_size` |
| `attention_types` | alternating global/local | Expands to one type per layer; length must equal `num_layers` |
| `window_size` | 256 | Local attention attends to current plus previous `window_size - 1` positions in eager mask behavior |
| `activation_function` | `gelu_new` | tanh GELU approximation |
| `use_cache` | true | Uses Transformers `Cache` ABI |
| `bos/eos_token_id` | 50256 | GPT-2 end-of-text token used as BOS/EOS |
| `pad_token_id` | null | No default pad token; generation callers may still pass masks |

Representative sweep:

| Checkpoint/config | Layers | Hidden | Heads | Head dim | Effective MLP | Pattern | Window | Vocab |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Local test debug config | 2 | 32 | 4 | 8 | explicit 37 | global, local | 7 | 99 |
| `EleutherAI/gpt-neo-125m` | 12 | 768 | 12 | 64 | 3072 | `(global, local) x 6` | 256 | 50257 |
| `EleutherAI/gpt-neo-1.3B` | 24 | 2048 | 16 | 128 | 8192 | `(global, local) x 12` | 256 | 50257 |
| `EleutherAI/gpt-neo-2.7B` | 32 | 2560 | 20 | 128 | 10240 | `(global, local) x 16` | 256 | 50257 |

## 3a. Family variation traps

- Local/global attention is per-layer. Do not lower all layers as dense global attention unless local-window parity is intentionally disabled.
- `attention_types` expands to `attention_layers`; invalid lengths are config errors.
- `window_size` is GPT-Neo-specific. Shared cache utilities inspect `sliding_window`/`layer_types`, so DinoML should not infer that Transformers prunes GPT-Neo local-layer cache memory automatically.
- No RoPE, ALiBi, GQA, MQA, MoE, or gated MLP in native GPT-Neo.
- Q/K/V projections are separate weights, not GPT-2 `Conv1D` packed weights. Weight layout is standard PyTorch `nn.Linear`: `[out_features, in_features]`.
- `q_proj`, `k_proj`, and `v_proj` have no bias; `out_proj`, MLP linears, and LayerNorm have bias/affine terms.
- Causal LM ties `lm_head.weight` to `transformer.wte.weight`.
- `FlashAttention2` is source-advertised, but the inspected GPT-Neo FA2 call passes causal attention without an obvious local-window parameter. Gate FA2 for local layers until a parity test proves the same window mask.
- `logits_to_keep` allows last-token-only or indexed logits projection; this is a useful generation ABI and performance hook.
- Token type IDs, if provided, reuse the token embedding table and are added to hidden states.

## 4. Operator coverage checklist

Tensor/layout ops:
- Integer token lookup into `Embedding(vocab_size, hidden_size)`.
- Learned position lookup into `Embedding(max_position_embeddings, hidden_size)`.
- Optional token type lookup using the token embedding table.
- Elementwise add for embedding sum and residuals.
- View/reshape/permute/contiguous for `[B,S,H] <-> [B,heads,S,head_dim]`.
- Slice/index for `logits_to_keep` and optional last-token-only logits.

Neural primitives:
- LayerNorm over hidden width, epsilon `1e-5`.
- Linear projections:
  - attention Q/K/V: `Linear(H -> H, bias=False)`.
  - attention output: `Linear(H -> H, bias=True)`.
  - MLP: `Linear(H -> 4H or intermediate_size)`, `gelu_new`, `Linear(intermediate -> H)`.
  - LM head: `Linear(H -> vocab_size, bias=False)`, tied to token embeddings for CausalLM.
- Dropout can be compiled away for inference.

Attention primitives:
- Causal MHA with score matmul `[B,heads,Q,D] @ [B,heads,D,K]`.
- Score computation is upcast to float32 in eager source before mask/softmax.
- Additive external attention mask after local/causal bool mask.
- Softmax over key axis, cast probabilities back to value dtype, value matmul.
- Local causal mask for local layers.

Generation/cache ops:
- Per-layer K/V append/update.
- Cache reorder by batch/beam index for generation.
- Position IDs default to `arange(query_len) + past_seen_tokens`.
- `logits_to_keep` projection slice.

Preprocessing-coupled ops:
- GPT-2 tokenizer ABI with `bos_token_id=eos_token_id=50256`; no model-owned processor.

## 5. Layer/block breakdown

Decoder setup:

```text
inputs_embeds = wte(input_ids) or caller inputs_embeds
position_ids = arange(S) + past_seen_tokens unless provided
hidden = inputs_embeds + wpe(position_ids)
if token_type_ids: hidden += wte(token_type_ids)
```

Decoder block, repeated `num_layers`:

```text
residual = x
x_norm = LayerNorm(x)
q = Linear_no_bias(H -> H)(x_norm)
k = Linear_no_bias(H -> H)(x_norm)
v = Linear_no_bias(H -> H)(x_norm)
q,k,v = reshape to [B, heads, S, head_dim]
k,v = cache.update(k,v, layer_id) when cache is supplied
attn = causal_or_local_attention(q,k,v, attention_mask)
x = residual + Linear_bias(H -> H)(merge(attn))

residual = x
x = LayerNorm(x)
x = Linear(H -> intermediate)(x)
x = gelu_new(x)
x = Linear(intermediate -> H)(x)
x = residual + x
```

Final CausalLM:

```text
x = final LayerNorm(x)
logits = tied_lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

GPT-Neo uses causal self-attention only. It is MHA, not GQA/MQA: K/V head count equals Q head count. Query/key/value width is always `hidden_size`.

Attention mask and math order:
- Source eager path computes `attn_weights = q.float() @ k.float().transpose(-1,-2)`.
- It applies a bool module mask with `torch.where(mask, scores, finfo_min)`.
- It then adds the external additive attention mask if present.
- Softmax is over the key axis; probabilities are cast to `value.dtype`.
- No explicit `1 / sqrt(head_dim)` scale appears in the eager GPT-Neo attention path. FA2 passes `softmax_scale=1.0`.

Local/global pattern:
- Global layers use a full lower-triangular causal mask.
- Local layers keep a lower-triangular window. For query absolute row `i`, allowed keys are approximately `max(0, i - window_size + 1) ... i`. Local test evidence with `window_size=4` expects query index 5 to attend only to `[2,3,4,5]`.

Cache ABI:
- Cached K/V tensors use `[batch, heads, seq, head_dim]`.
- Eager local attention calls `layer_past.update(...)` and receives full K/V states, then uses the local mask to restrict attention.
- Because GPT-Neo config does not expose `layer_types` or shared `sliding_window`, `DynamicCache(config)` should be treated as full-growth for this family unless DinoML deliberately introduces a compatible local-cache policy.
- Beam reorder is batch-axis `index_select` on cached keys/values.

Backend compatibility:
- Dense prefill can use standard causal attention for global layers and local-window causal attention for local layers.
- Decode with `Q=1` for local layers only needs the last `window_size` keys semantically, but source cache behavior may keep full history. A pruned DinoML cache is an optimization only if position/mask parity is tested.
- FlashAttention2 for global layers may be viable. For local layers, require a no-FA2 guard or a local-window backend path because source dispatch does not visibly pass the GPT-Neo window.

## 7. Position encoding and custom math

GPT-Neo uses learned absolute position embeddings, not RoPE/ALiBi/relative bias.

```python
def gpt_neo_position_ids(query_len, past_seen_tokens, device):
    return torch.arange(query_len, device=device).unsqueeze(0) + past_seen_tokens
```

Local attention mask:

```python
def gpt_neo_local_mask(max_positions, window_size):
    causal = tril(ones(max_positions, max_positions, bool))
    too_old = tril(causal, -window_size)
    return causal ^ too_old
```

`gelu_new`:

```python
def gelu_new(x):
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))
```

Position IDs depend on cache length at runtime. The full local/global mask buffer can be precomputed per `max_position_embeddings` and `window_size`, or generated as index predicates in fused attention.

## 8. Preprocessing and input packing

Tokenizer/model coupling is GPT-2-style:
- Configs name `GPT2Tokenizer`; local tiny summary lists `GPT2Tokenizer` and `GPT2TokenizerFast`.
- `bos_token_id` and `eos_token_id` are 50256.
- No default pad token in official configs; padding requires caller/tokenizer policy plus `attention_mask`.

Runtime inputs:
- `input_ids`: `[B,S]`.
- `attention_mask`: typically `[B,total_seen_plus_S]`, consumed by shared mask utilities and/or additive eager mask.
- `position_ids`: optional `[B,S]`; generated if absent.
- `token_type_ids`: optional `[B,S]`; embedded through `wte` and added.
- `inputs_embeds`: optional `[B,S,H]`; mutually exclusive with `input_ids`.

No multimodal placeholder, processor, packed varlen, RoPE metadata, or image/audio preprocessing is required.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV GEMM

Source pattern:
```text
q = Linear_no_bias(H,H)(x)
k = Linear_no_bias(H,H)(x)
v = Linear_no_bias(H,H)(x)
```

Replacement:
```text
qkv = Linear_no_bias(H, 3H)(x)
split last dim into q,k,v in source order [q, k, v]
```

Preconditions:
- All three projections consume the exact same normalized hidden tensor.
- Weights are standard PyTorch linear layout `[out, in]`.
- No LoRA/adapters/quant wrappers mutate individual projections at runtime.
- Biases remain absent.

Failure cases:
- Quantized modules, adapter-injected layers, or nonstandard `nn.Linear` wrappers.

Parity sketch:
- Compare q/k/v tensors before attention for random `[B,S,H]` and all representative widths.

### Rewrite: local attention mask -> local-window attention backend

Source pattern:
```text
scores masked by GPTNeoSelfAttention.bias for local layers
```

Replacement:
```text
causal_local_attention(q, k, v, window_size)
```

Preconditions:
- Layer type is exactly `"local"`.
- Window predicate matches source inclusive window.
- External padding mask is added after local/causal mask.
- Decode path position offsets are validated against cache length.

Failure cases:
- FA2 path without local-window support.
- Sequence length beyond `max_position_embeddings` unless position/mask policy is extended.

### Rewrite: inference dropout removal

Dropout modules after embeddings, attention probs, attention output, and MLP output are identity in eval mode. Remove only for inference artifacts.

### Rewrite: last-token-only logits

Source pattern:
```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:
```text
project only requested positions, commonly S-1 during decode
```

Preconditions:
- `logits_to_keep` is an integer or static/index tensor known to generation controller.
- Loss is not computed inside graph.

## 10. Kernel fusion candidates

Highest priority:
- LayerNorm kernels for `[B,S,H]`, including final norm.
- Packed QKV GEMM plus split/reshape for prefill.
- Global causal attention and local-window causal attention with cache.
- MLP `Linear + gelu_new + Linear`, with GELU fused into GEMM epilogue where feasible.
- Last-token-only tied LM projection for decode.

Medium priority:
- Embedding sum fusion: token + position + optional token type.
- Residual add fused with output projection epilogues.
- Attention score upcast/mask/softmax/value matmul parity kernels for local windows.
- Cache append/reorder kernels, especially for batched generation.

Lower priority:
- Classification/QA heads.
- `output_attentions` dense materialization.
- Training losses/dropouts/gradient checkpointing.

## 11. Runtime staging plan

1. Parse config and reject invalid `attention_types` length, non-divisible `hidden_size/num_heads`, unsupported attention implementation, and unsupported adapters.
2. Load weights with tied `lm_head.weight` / `wte.weight` alias preserved.
3. Build one-block CPU/CUDA parity for embeddings, LayerNorm, separate linears, eager attention, MLP.
4. Implement full prefill with alternating global/local causal masks.
5. Implement decode with per-layer K/V cache; start with full-growth cache for source parity.
6. Add optional pruned local-layer cache with explicit parity tests.
7. Add packed QKV and attention backend fusions.
8. Add last-token-only logits and generation-controller integration.
9. Defer non-CausalLM heads until text generation is stable.

## 12. Parity and validation plan

- Config validation tests for expanded `attention_types`, `window_size`, and head divisibility.
- Random tensor tests for `gelu_new`, LayerNorm, embedding sum, Q/K/V reshape/merge.
- Attention unit tests:
  - global full causal prefill.
  - local window prefill with a small window like 4.
  - padding mask plus local mask ordering.
  - decode `Q=1` with growing cache.
- Single-block parity against Transformers in fp32, then fp16/bf16 with relaxed tolerances.
- Full-model prefill logits for 125M with short prompts.
- Decode token parity for greedy generation over 8-32 new tokens.
- Cache reorder parity for beam-style batch index selection.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 compare logits with `rtol=5e-2, atol=5e-2` initially, tighten after fused attention is stable.

## 13. Performance probes

- Prefill throughput sweep over `B in {1,4,8}` and `S in {128,512,1024,2048}`.
- Decode tokens/sec sweep over batch and prompt length.
- Global-vs-local layer attention timing split.
- Full-growth cache memory versus pruned local-cache memory.
- Last-token-only logits versus full-sequence logits projection.
- Separate MLP GEMM time by model size: 125M, 1.3B, 2.7B.
- QKV packed GEMM versus three separate GEMMs.
- Attention backend comparison: eager dense, local-window specialized, FlashAttention global-only.

## 14. Skip/defer list

- Training, loss computation, dropout, and gradient checkpointing.
- Sequence, token, and QA heads for first causal LM target.
- `output_hidden_states` and `output_attentions` materialization beyond debugging.
- FlashAttention2 for local layers until local-window parity is proven.
- Quantization/packed weight formats; no native source-coupled quantized format is required by GPT-Neo.
- Tensor parallel and multi-GPU.
- Speculative decoding and advanced generation processors.

## 15. Final implementation checklist

- [ ] Parse `GPTNeoConfig` and expand/validate `attention_types`.
- [ ] Load embeddings, LayerNorm, Q/K/V/out projections, MLP weights, and tied LM head.
- [ ] Implement learned absolute position IDs with cache-length offset.
- [ ] Implement `gelu_new`.
- [ ] Implement global causal attention.
- [ ] Implement GPT-Neo local-window causal attention.
- [ ] Implement full-growth K/V cache and beam reorder.
- [ ] Add guarded local-layer cache pruning.
- [ ] Add packed QKV rewrite.
- [ ] Add last-token-only logits projection.
- [ ] Add one-block, prefill, decode, and cache reorder parity tests.
- [ ] Benchmark prefill/decode, local/global layers, cache memory, and LM head projection.

