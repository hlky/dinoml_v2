# BLOOM Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: bigscience/bloom family, primary sweep: bloom-560m, bloom-1b1, bloom-3b, bloom-7b1, bloom
Config source: Hugging Face repo config.json/tokenizer_config.json fetched from official bigscience repos
Source files inspected:
  X:/H/transformers/src/transformers/models/bloom/configuration_bloom.py
  X:/H/transformers/src/transformers/models/bloom/modeling_bloom.py
  X:/H/transformers/src/transformers/models/bloom/__init__.py
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
  Local __init__.py imports tokenization_bloom, but tokenization_bloom.py/tokenization_bloom_fast.py is missing in the pinned checkout.
  Tokenizer facts below come from official HF tokenizer_config.json/special_tokens_map.json/tokenizer.json metadata, not local source.
```

Source URLs for future review:

- `modeling_bloom.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bloom/modeling_bloom.py
- `configuration_bloom.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bloom/configuration_bloom.py
- `cache_utils.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py
- `masking_utils.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/masking_utils.py
- Representative configs: https://huggingface.co/bigscience/bloom-560m/raw/main/config.json, https://huggingface.co/bigscience/bloom-1b1/raw/main/config.json, https://huggingface.co/bigscience/bloom-3b/raw/main/config.json, https://huggingface.co/bigscience/bloom-7b1/raw/main/config.json, https://huggingface.co/bigscience/bloom/raw/main/config.json

Primary runtime target: `BloomForCausalLM` generation. Other heads in source are optional/deferred: base `BloomModel` is required as the decoder body; sequence classification, token classification, and question answering are deferred.

## 2. High-level architecture

BLOOM is a text-only decoder-only causal LM with learned token embeddings, embedding LayerNorm, repeated Transformer decoder blocks, final LayerNorm, and a tied un-biased LM projection.

```text
tokenization/input_ids + 2D attention_mask
  -> word embedding
  -> embedding LayerNorm
  -> repeated decoder blocks with ALiBi causal self-attention and MLP
  -> final LayerNorm
  -> last-token or selected-token lm_head
  -> logits/sampling
```

Generation stage split:

```text
CPU tokenizer -> prefill(input_ids, attention_mask, empty cache) -> KV cache
             -> decode(one/new tokens, grown 2D attention_mask, KV cache)
             -> logits_to_keep/lm_head -> sampler/controller
```

The ALiBi tensor is runtime-derived from the 2D `attention_mask`, so the attention-mask preparation path is part of model parity, not only generation-controller plumbing.

## 3. Important config dimensions

Source defaults from `BloomConfig` at this commit: `vocab_size=250880`, `hidden_size=64`, `n_layer=2`, `n_head=8`, `layer_norm_epsilon=1e-5`, `use_cache=True`, `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=None`, `hidden_dropout=0.0`, `attention_dropout=0.0`, `pretraining_tp=1`, `slow_but_exact=False`, `tie_word_embeddings=True`. `n_embed` is a backward-compatible alias that overwrites `hidden_size`.

| Field | Meaning for lowering |
| --- | --- |
| `hidden_size` / `n_embed` | Model width `H`; many older configs use `n_embed` instead of `hidden_size`. |
| `n_layer` / `num_hidden_layers` | Number of decoder blocks. |
| `n_head` / `num_attention_heads` | Full MHA head count; no GQA/MQA field. |
| `head_dim` | Source computes `hidden_size // n_head` and requires exact divisibility. |
| `intermediate_size` | Fixed `4 * hidden_size`; `n_inner` appears in older configs but source MLP ignores it. |
| `vocab_size` | Embedding and LM head rows, usually 250880. |
| `max_position_embeddings` | Not used; BLOOM uses ALiBi and has no learned/rotary position embedding. |
| `activation` | BLOOM tanh GELU approximation. |
| `norm` | PyTorch `LayerNorm(eps=1e-5)` before attention, after attention, embedding, and final output. |
| `cache` | Dynamic/static HF `Cache`; per-layer K/V shape `[B, H, T, D]`. |
| `pretraining_tp` | Checkpoint hint for Megatron tensor-parallel merge. Runtime only changes math when `slow_but_exact=True`. |

Representative checkpoint sweep, from official `config.json`:

| Model id | Layers | Hidden | Heads | Head dim | MLP width | Vocab | `pretraining_tp` | dtype/config notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `bigscience/bloom-560m` | 24 | 1024 | 16 | 64 | 4096 | 250880 | 1 | Uses `n_embed`; pad=3; no generation_config at repo root. |
| `bigscience/bloom-1b1` | 24 | 1536 | 16 | 96 | 6144 | 250880 | 1 | Uses `n_embed`; same tokenizer metadata. |
| `bigscience/bloom-3b` | 30 | 2560 | 32 | 80 | 10240 | 250880 | 4 | `pretraining_tp=4`; source default `slow_but_exact=False`. |
| `bigscience/bloom-7b1` | 30 | 4096 | 32 | 128 | 16384 | 250880 | 1 | Uses `hidden_size`/`n_head`; `torch_dtype=float16`. |
| `bigscience/bloom` | 70 | 14336 | 112 | 128 | 57344 | 250880 | 4 | 176B config; source computes very large fused QKV and cache. |

`bigscience/bloomz-560m` has the same operator-significant dimensions as `bloom-560m` and adds `seq_length=2048` in config. It uses the same tokenizer class and special tokens.

## 3a. Family variation traps

- Older configs use `n_embed` and `num_attention_heads`; newer configs may use `hidden_size` and `n_head`. The config class maps names and applies `n_embed` in `__post_init__`.
- `pretraining_tp` is a checkpoint/training hint. The normal fast path ignores it; if `slow_but_exact=True`, attention output projection and MLP down projection are split into `pretraining_tp` slices and summed.
- No RoPE, no learned absolute positions, no GQA/MQA, no sliding window, no cross-attention, no MoE.
- ALiBi slopes depend on `num_heads`; ALiBi positions depend on `attention_mask.cumsum`, so left padding affects bias positions.
- Source attention is an eager explicit `baddbmm + mask + softmax(fp32) + bmm` path inside `BloomAttention`, not a direct call to SDPA/FlashAttention.
- Config fields from older training code such as `attention_softmax_in_fp32`, `masked_softmax_fusion`, `bias_dropout_fusion`, `skip_bias_add`, `skip_bias_add_qkv`, `offset_alibi`, and `n_inner` appear in HF configs but are not read by the inspected modeling source.
- Tokenizer metadata says `BloomTokenizerFast`, `padding_side="left"`, `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3`. Local tokenizer source is absent.
- Layout-sensitive graph pieces are all `[B, T, H]` text tensors and `[B, H, T, D]` cache tensors. There is no NCHW/NHWC issue, but head flattening to `[B*H, T, D]` must be guarded around attention rewrites.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab_size, hidden_size)` from `input_ids[int64] -> [B, T, H]`.
- Shape/view: `view`, `reshape`, `transpose`, `permute`, `contiguous`-equivalent layout materialization when needed.
- Slicing for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Concatenate for dynamic cache growth and static-cache attention-mask padding.
- `cumsum`, `arange`, `pow`, `cat`, broadcast multiply for ALiBi construction.

Neural network primitives:

- LayerNorm over last dim `H`, affine, epsilon `1e-5`.
- Fused QKV Linear `H -> 3H`, bias=True.
- Attention output Linear `H -> H`, bias=True.
- MLP up Linear `H -> 4H`, bias=True.
- BLOOM GELU tanh approximation.
- MLP down Linear `4H -> H`, bias=True.
- Residual add; dropout is inactive in inference because configs use 0.0 and model should run eval.
- LM head Linear `H -> vocab_size`, bias=False, weight tied to embeddings when `tie_word_embeddings=True`.

Attention primitives:

- Full causal self-attention MHA.
- `baddbmm`: `alibi[B*H,1,K] + (Q[B*H,Q,D] @ Kt[B*H,D,K]) / sqrt(D)`.
- Add 4D causal/padding mask `[B,1,Q,K]`.
- Softmax over `K` computed in fp32 and cast back to query dtype.
- `bmm`: `P[B*H,Q,K] @ V[B*H,K,D]`.

Generation/cache ops:

- HF `DynamicCache`/`StaticCache` equivalent.
- Per-layer cache update accepts key/value `[B,H,Q,D]` and returns full `[B,H,K,D]`; source then flattens heads.
- 2D attention mask length must be `past_length + current_length`, and static cache pads it to max cache length in `prepare_inputs_for_generation`.
- `logits_to_keep` optimization for last-token-only logits.

Tokenizer/preprocessing-coupled ops:

- Byte-level BPE tokenizer via `BloomTokenizerFast` metadata.
- Left padding is expected by tokenizer config; `attention_mask` must mark real tokens for ALiBi `cumsum`.
- Special token ids: `<unk>=0`, `<s>=1`, `</s>=2`, `<pad>=3` from repo configs/maps.

Distributed/tensor-parallel hints:

- Optional exact TP-merge emulation if `pretraining_tp > 1 and slow_but_exact`: slice input activation/weight along the projection input dimension and sum multiple `F.linear` results. First integration can reject or disable `slow_but_exact=True`.

## 5. Layer/block breakdown

Embedding and output:

```text
input_ids [B,T] -> word_embeddings [B,T,H]
hidden = LayerNorm(H)(embeddings)
for layer in N:
  hidden = BloomBlock(hidden, alibi, causal_mask, cache)
hidden = final LayerNorm(H)(hidden)
logits = lm_head(hidden[:, selected_positions, :]) -> [B,L_keep,V]
```

Decoder block, repeated `n_layer` times:

```text
ln1 = LayerNorm(hidden)
residual1 = hidden unless apply_residual_connection_post_layernorm else ln1
fused = Linear_qkv(ln1)                         # [B,T,3H], bias=True
q,k,v = view(fused, [B,T,Hd_count,3,D])         # each [B,Hd_count,T,D] after transpose
k,v = cache.update(k,v,layer_idx) if cache
scores = ALiBi + (q @ k.T) / sqrt(D)            # flattened to [B*heads,Q,K]
scores = scores.view([B,heads,Q,K]) + mask
probs = softmax(scores, dim=-1, dtype=fp32).to(model_dtype)
context = probs @ v                             # [B*heads,Q,D]
context = merge_heads(context)                  # [B,Q,H]
attn_out = Linear_o(context)                    # [B,Q,H], bias=True
hidden = residual1 + attn_out
ln2 = LayerNorm(hidden)
residual2 = hidden unless apply_residual_connection_post_layernorm else ln2
mlp = Linear_down(GELU(Linear_up(ln2)))          # H -> 4H -> H
hidden = residual2 + mlp
```

For `bloom-7b1`, this concretely means QKV `4096 -> 12288`, attention output `4096 -> 4096`, MLP `4096 -> 16384 -> 4096`. For `bigscience/bloom`, QKV is `14336 -> 43008`, MLP is `14336 -> 57344 -> 14336`.

## 6. Attention requirements

- Type: causal self-attention, full MHA.
- Heads: `num_key_value_heads == num_attention_heads`; no repeat-KV.
- Head dim: source-derived `D = hidden_size / n_head`.
- Masking: 4D additive float mask from `create_causal_mask`, shape `[B,1,Q,K]`, added after ALiBi + QK matmul.
- ALiBi: additive bias shape `[B*H,1,K]`, reshaped/broadcast against `[B*H,Q,K]` before mask addition.
- Cache storage: before flattening, keys and values are stored as `[B,H,T,D]`. Dynamic cache concatenates along `dim=-2`, so generated decode cache should append on sequence axis. Returned full K/V are then reshaped to `key [B*H,K,D].transpose(-1,-2)` and `value [B*H,K,D]`.
- Cached keys are stored before the attention flattening and after no positional transform; ALiBi is not baked into cache.
- Math order to preserve: QK matmul with `alpha=1/sqrt(D)` and `beta=1` into ALiBi via `baddbmm`; add mask; softmax in fp32; cast probabilities to query dtype; dropout; PV matmul.
- Source optimized backend dispatch: none inside `BloomAttention`; it always uses eager `baddbmm`/`bmm`. `create_causal_mask` may use mask utilities keyed by config `_attn_implementation`, but the attention kernel itself is not SDPA/FA2.
- FlashAttention compatibility inference: possible as an optimization only if ALiBi bias, padding, causal mask, fp32-softmax parity, and cache layout are supported. A direct FA2 replacement needs tests because source ALiBi uses mask cumsum and applies mask after ALiBi.

## 7. Position encoding and custom math

BLOOM uses ALiBi, not RoPE. Source implementation:

```python
def bloom_alibi(attention_mask, num_heads, dtype):
    B, K = attention_mask.shape
    closest = 2 ** floor(log2(num_heads))
    base = 2 ** (-(2 ** -(log2(closest) - 3)))
    slopes = base ** arange(1, closest + 1)
    if closest != num_heads:
        extra_base = 2 ** (-(2 ** -(log2(2 * closest) - 3)))
        extra = extra_base ** arange(1, 1 + 2 * min(closest, num_heads - closest), 2)
        slopes = cat([slopes, extra])
    positions = ((attention_mask.cumsum(-1) - 1) * attention_mask)[:, None, :]
    return (slopes[..., None] * positions).reshape(B * num_heads, 1, K).to(dtype)
```

Precompute opportunity: slopes are static per head count and can be cached. Positions depend on runtime `attention_mask`, especially padding, so full ALiBi cannot be a fixed constant unless there is a no-padding/full-prefix specialization.

BLOOM GELU:

```python
def bloom_gelu(x):
    return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
```

## 8. Preprocessing and input packing

CPU/data pipeline:

- Tokenizer class from official tokenizer metadata: `BloomTokenizerFast`.
- Tokenizer special tokens: `<unk>`, `<s>`, `</s>`, `<pad>`.
- Token ids from official configs/maps: unk 0, bos 1, eos 2, pad 3.
- Padding side from tokenizer config: left.
- No segment/token type ids and no position ids enter BLOOM source forward.

GPU/runtime inputs:

- `input_ids [B,T] int64` or mutually exclusive `inputs_embeds [B,T,H]`.
- `attention_mask [B,K]`, where `K = past_length + T` for decode. If omitted, source creates all ones.
- `past_key_values`, optional; created as `DynamicCache(config)` when `use_cache=True` and no cache is passed.

Generation-controller behavior:

- No repo-root `generation_config.json` found for inspected BLOOM checkpoints. Use generic Transformers generation defaults plus config `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=3`.
- `prepare_inputs_for_generation` has a BLOOM-specific static-cache path that pads the 2D mask to `max_cache_len` so ALiBi sees fixed mask length.
- `logits_to_keep` can limit LM head compute; first optimized generation path should set `logits_to_keep=1` for decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused BLOOM QKV projection

Source pattern:

```text
Linear(H -> 3H) -> view [B,T,heads,3,D] -> select q/k/v -> transpose
```

Replacement:

```text
single GEMM_RRR/RCR with fused bias -> structured Q/K/V views or split metadata
```

Preconditions:

- `query_key_value.weight` rows are ordered per head as `[q, k, v]` inside `view(B,T,heads,3,D)`, not simply all-Q then all-K then all-V.
- `hidden_size == heads * head_dim`.
- Downstream attention accepts either the fused storage layout or a lowered split with exact strides.

Failure cases:

- Any checkpoint conversion that changes QKV packing order.
- Layout pass that assumes `[Q_all, K_all, V_all]` row blocks.

Parity test sketch: compare projected Q/K/V tensors after `_reshape` for random hidden states against HF source.

### Rewrite: eager ALiBi attention to fused causal MHA

Source pattern:

```text
ALiBi.baddbmm(Q, K^T, beta=1, alpha=1/sqrt(D))
-> add 4D mask
-> softmax(fp32).to(dtype)
-> bmm(V)
```

Replacement:

```text
fused_prefill_attention_with_alibi(Q,K,V,mask,slopes)
fused_decode_attention_with_alibi(Q,K_cache,V_cache,mask,slopes)
```

Preconditions:

- Full MHA, no GQA/MQA.
- ALiBi positions match `attention_mask.cumsum`.
- Padding and causal masking are both represented.
- Softmax accumulation/upcast matches fp32 behavior within tolerance.

Shape equations:

- `Q [B,H,Q,D]`, `K/V [B,H,K,D]`, `scores [B,H,Q,K]`.
- Cache bytes per layer: `2 * B * H * max_T * D * sizeof(dtype)`.

Failure cases:

- Packed/custom 4D masks.
- Static-cache padded attention masks unless fused kernel can ignore padded cache slots correctly.

### Rewrite: last-token-only LM head

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden to [B,1,H] before vocab GEMM during decode
```

Preconditions:

- No training loss.
- Generation asks only next-token logits (`logits_to_keep=1` or equivalent).

Failure cases:

- Prompt scoring or logprobs over all prompt positions.
- Tensor-valued `logits_to_keep` requesting arbitrary positions.

### Rewrite: slow_but_exact TP slices to normal Linear or split GEMM

Source pattern when `pretraining_tp > 1 and slow_but_exact`:

```text
sum_i F.linear(x[..., i*s:(i+1)*s], W[:, i*s:(i+1)*s])
```

Replacement:

```text
normal Linear(x, W) if slow_but_exact=False
or split-K/sliced GEMM accumulation if exact mode is required
```

Preconditions:

- Normal path is allowed for `slow_but_exact=False`, which is true in inspected large configs.
- If exact mode is required, slice boundaries divide input width.

Failure cases:

- User explicitly requests exact Megatron TP parity for `slow_but_exact=True`.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm kernels for embedding LN, two per block, and final LN. BLOOM is LayerNorm-heavy and not RMSNorm.
- Fused QKV GEMM with correct BLOOM head-major QKV packing.
- ALiBi causal attention prefill and decode kernels with fp32 softmax and `[B,H,T,D]` cache.
- Last-token-only LM head for decode to avoid `[B,T,V]` vocab projection.

Medium priority:

- MLP fused bias + BLOOM GELU + down projection. Current DinoML has GEMM epilogues and GELU pieces, but BLOOM's two-GEMM MLP benefits from activation fusion.
- Residual-add fusion around attention output and MLP output; dropout can be removed in eval/zero-dropout configs.
- ALiBi slope/position generation specialization for no-padding batches.
- Static cache update and mask-padding path for compile/export-style decode.

Lower priority:

- `slow_but_exact` sliced projection mode.
- Classification/QA/token-classification heads.
- Tokenizer execution inside runtime; keep in CPU pipeline initially.

## 11. Runtime staging plan

1. Parse BLOOM config including `n_embed` alias, `n_head` alias, `pretraining_tp`, and tokenizer special ids.
2. Load weights and validate QKV packing with one small checkpoint (`bloom-560m`).
3. Implement one block parity in fp32 with eager unfused attention and no cache.
4. Add full prefill parity for `BloomForCausalLM`, including ALiBi from 2D mask and final LM head.
5. Add dynamic-cache decode parity with per-layer `[B,H,T,D]` K/V buffers.
6. Add `logits_to_keep=1` decode optimization.
7. Replace eager attention with fused ALiBi attention kernels after prefill/decode parity is stable.
8. Add GGUF/quantized weight loading only after dense weight parity; QKV and MLP weights are large enough to benefit from DinoML's GGUF runtime-dequant GEMM work.

Initial stubs: dropout as identity in eval; no training loss; no classification/QA heads; reject `slow_but_exact=True` until sliced GEMM parity is explicitly needed.

## 12. Parity and validation plan

- ALiBi unit tests: head counts 16, 32, 112; left-padded and unpadded masks; compare exact slopes/positions to HF.
- QKV packing test: random `[B,T,H]`, compare fused projection split Q/K/V tensors after `_reshape`.
- Single block fp32 parity: random hidden states and all-ones mask, no cache, tolerance around `1e-5`.
- Prefill LM parity: `bloom-560m` short prompts, fp32 or fp16 according to checkpoint, compare hidden states and logits.
- Decode parity: prefill then one-token and multi-token decode, compare cache shapes and logits.
- Left-padding parity: two prompts with shared suffix but different pad counts; verify ALiBi and logits match HF.
- `logits_to_keep=1` parity: compare last-token logits to full logits slice.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 `rtol=5e-3, atol=5e-3`, with tighter per-op tests where possible.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Prefill throughput sweep: `B in {1,4,8}`, `T in {128,512,2048}`.
- Decode tokens/sec sweep: cache lengths `{128,512,2048,4096}` and batch sizes `{1,4,16}`.
- ALiBi construction overhead: mask cumsum and bias materialization versus fused in-kernel ALiBi.
- Attention backend comparison: eager baddbmm/bmm baseline versus fused ALiBi attention.
- LM head cost: full prompt logits versus last-token-only logits.
- KV cache memory: `2 * layers * B * heads * T * head_dim * dtype_size`; for `bloom-7b1`, this is `2 * 30 * B * T * 4096 * dtype_size`.
- Weight GEMM profiling: QKV, attention output, MLP up/down, and LM head separately.
- Quantized/dequantized weight path probe for large `bloom`/`bloom-7b1` GEMMs.

## 14. Skip/defer list

- Training, gradients, dropout randomness, gradient checkpointing.
- `slow_but_exact=True` TP-sliced projection mode for first pass.
- Sequence classification, token classification, and QA heads.
- Beam search and advanced generation processors beyond standard causal LM sampling.
- Tokenizer implementation inside DinoML runtime.
- Multi-GPU/tensor-parallel execution, despite `pretraining_tp` metadata.
- StaticCache compile/export path until dynamic-cache generation works.
- Quantized KV cache classes and offloaded HF cache variants.

## 15. Final implementation checklist

- [ ] Parse `BloomConfig`, including `n_embed -> hidden_size` and `n_head`/`num_attention_heads` aliases.
- [ ] Load tokenizer metadata and special token ids from HF repo files.
- [ ] Load tied embedding/LM head weights.
- [ ] Implement BLOOM LayerNorm placements and epsilon.
- [ ] Implement BLOOM tanh GELU.
- [ ] Implement fused QKV projection with BLOOM row packing.
- [ ] Implement ALiBi slope and mask-cumsum position generation.
- [ ] Implement eager reference attention with fp32 softmax and additive causal/pad mask.
- [ ] Implement per-layer KV cache `[B,H,T,D]` update/readback.
- [ ] Implement prefill parity for `BloomForCausalLM`.
- [ ] Implement decode parity with `logits_to_keep=1`.
- [ ] Add rewrite for fused ALiBi prefill/decode attention.
- [ ] Add rewrite/fusion for last-token-only LM head.
- [ ] Add optional GGUF/dequant GEMM path for large dense projections.
- [ ] Add parity tests for left padding, cache growth, QKV split, ALiBi, one block, prefill logits, and decode logits.
- [ ] Add performance probes for prefill, decode, ALiBi overhead, LM head slicing, and KV memory.
