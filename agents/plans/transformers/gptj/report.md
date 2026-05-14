# GPT-J Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: gptj family; primary checkpoint EleutherAI/gpt-j-6b
Config source: local GPTJConfig plus official/open Hugging Face config.json, tokenizer_config.json, and repo file metadata
Source files inspected:
- X:/H/transformers/src/transformers/models/gptj/configuration_gptj.py
- X:/H/transformers/src/transformers/models/gptj/modeling_gptj.py
- X:/H/transformers/tests/models/gptj/test_modeling_gptj.py
- X:/H/transformers/src/transformers/models/auto/tokenization_auto.py
- X:/H/transformers/src/transformers/cache_utils.py, indirectly through Cache/DynamicCache usage
- X:/H/transformers/src/transformers/masking_utils.py, indirectly through create_causal_mask usage
Any missing files or assumptions: the GPT-J directory has no tokenizer file; AutoTokenizer maps `gptj` to GPT2Tokenizer. No remote-code files are required for the inspected in-library source. No standalone generation_config.json was present in inspected repos; generation defaults come from config/tokenizer metadata and GenerationMixin.
```

Representative Hub sources:

- `https://huggingface.co/EleutherAI/gpt-j-6b/raw/main/config.json`
- `https://huggingface.co/EleutherAI/gpt-j-6b/raw/main/tokenizer_config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-GPTJForCausalLM/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-GPTJModel/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-GPTJForQuestionAnswering/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-GPTJForSequenceClassification/raw/main/config.json`

Primary runtime target: `GPTJForCausalLM` decoder-only generation, including prefill, autoregressive decode, and KV cache. Base `GPTJModel` is required for block parity. Sequence classification and question answering heads are optional/deferred for generation-first integration.

## 2. High-level architecture

GPT-J is a text-only decoder-only Transformer with learned token embeddings, repeated pre-LayerNorm decoder blocks, causal MHA, GPT-J rotary position embeddings, a parallel residual MLP branch, final LayerNorm, and a bias-bearing LM projection.

```text
byte-level BPE tokenization -> input_ids/attention_mask
  -> token embedding (+ optional token_type embedding)
  -> N GPT-J decoder blocks
  -> final LayerNorm
  -> lm_head logits
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: GPT-2 byte-level BPE tokenization, padding policy, attention mask construction, generation loop and sampling.
- GPU prefill: embedding lookup, position IDs, full prompt causal attention, post-RoPE K/V cache population, logits for all or selected positions.
- GPU decode: one or more new tokens, position offset from cache length, cache append, attention over prior cache plus new tokens, normally last-token logits only.
- Independently validatable units: tokenizer IDs, GPT-J RoPE table/gather/apply, one decoder block, cache update/read, full prefill logits, one-step decode logits.

## 3. Important config dimensions

Current source defaults from `GPTJConfig`:

| Field | Default / meaning |
| --- | --- |
| `vocab_size` | 50400 |
| `n_positions` / `max_position_embeddings` | 2048 |
| `n_embd` / `hidden_size` | 4096 |
| `n_layer` / `num_hidden_layers` | 28 |
| `n_head` / `num_attention_heads` | 16 |
| `head_dim` | inferred as `n_embd // n_head`; source raises if not divisible |
| `rotary_dim` | 64; RoPE applies only to first 64 channels of each head by default |
| `n_inner` | `None`, so MLP width is `4 * n_embd` |
| `activation_function` | `gelu_new` |
| `layer_norm_epsilon` | `1e-5` |
| `attn_pdrop`, `resid_pdrop`, `embd_pdrop` | all `0.0` by default and no-op in eval |
| attention projection bias | false for q/k/v/out projections |
| MLP bias | true for both MLP linears |
| LM head bias | true in source `nn.Linear(config.n_embd, config.vocab_size)` |
| `use_cache` | true |
| `tie_word_embeddings` | false by default; source still declares LM-head tie metadata |
| attention backends | in-family classes for `eager` and `flash_attention_2`; no GPT-J SDPA class in this file |

Representative checkpoint sweep:

| Model id | Architecture | Layers | Hidden | Heads | Head dim | Rotary dim | MLP width | Vocab | Positions | Dtype/config notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hf-internal-testing/tiny-random-GPTJForCausalLM` | `GPTJForCausalLM` | 5 | 32 | 4 | 8 | 4 | 128 inferred from source, because `n_inner=null` | 1024 | 512 | `torch_dtype=float32`, pad token 1023 |
| `hf-internal-testing/tiny-random-GPTJModel` | `GPTJModel` | 5 | 32 | 4 | 8 | 4 | 128 inferred | 1024 | 512 | base body only; same operator shape as LM target except no LM head |
| `hf-internal-testing/tiny-random-GPTJForQuestionAnswering` | `GPTJForQuestionAnswering` | 5 | 32 | 4 | 8 | 4 | 128 inferred | 1024 | 512 | optional QA head `H -> num_labels` |
| `hf-internal-testing/tiny-random-GPTJForSequenceClassification` | `GPTJForSequenceClassification` | 5 | 32 | 4 | 8 | 4 | 128 inferred | 1024 | 512 | optional pooled last-token classifier |
| `EleutherAI/gpt-j-6b` | `GPTJForCausalLM` | 28 | 4096 | 16 | 256 | 64 | 16384 inferred | 50400 | 2048 | config has legacy fields `rotary=true`, `scale_attn_weights=true`; current source ignores those names and uses `rotary_dim` plus `scale_attn=sqrt(head_dim)` |

Omitted fields and effective defaults:

- `num_key_value_heads` is absent; GPT-J is full MHA with KV heads equal to query heads.
- `n_inner` is commonly null; current source uses `4 * n_embd`.
- `generation_config.json` was absent in inspected repos. `EleutherAI/gpt-j-6b` config carries `task_specific_params.text-generation` with sampling metadata, but this is generation-controller metadata, not module graph structure.
- Tiny configs carry historical fields such as `intermediate_size`, `hidden_act`, `hidden_dropout_prob`, `attention_probs_dropout_prob`, `type_vocab_size`, and `is_decoder`; current `GPTJConfig` does not define these and current GPT-J source does not use them for the model body.

## 3a. Family variation traps

- GPT-J uses separate `q_proj`, `k_proj`, and `v_proj` linears, not a packed QKV projection. All four attention projections are bias-free.
- RoPE is GPT-J-specific interleaved even/odd rotation, not GPT-NeoX half-split rotation. The source uses `rotate_every_two` over adjacent pairs.
- `rotary_dim` is absolute channels per head, not a fraction. Official GPT-J 6B has `head_dim=256` and `rotary_dim=64`, so 192 head channels pass through unchanged.
- If `rotary_dim=None`, source applies RoPE to the whole head and builds the sinusoidal table with `embed_dim`; first integration can reject this until needed.
- `create_sinusoidal_positions(max_positions, rotary_dim)` returns concatenated `[sin, cos]` halves; forward gathers by `position_ids`, splits into sin/cos, then repeats each sin/cos element twice to match even/odd channel pairs.
- Parallel residual is mandatory in this source: the same `ln_1(hidden)` feeds attention and MLP, then `hidden = attn_out + mlp_out + residual`. Do not lower GPT-J as a sequential attention-then-MLP block.
- Source has no RMSNorm, no gated MLP, no GQA/MQA, no sliding window, no ALiBi, and no cross-attention.
- `lm_head` is constructed with bias, unlike many decoder-only LM heads. Official config sets `tie_word_embeddings=false`; if a custom checkpoint ties weights, preserve aliasing explicitly.
- `token_type_ids`, if supplied, are embedded through the same `wte` token table and added to token embeddings. This is optional and uncommon for generation, but source supports it.
- Sequence classification has a padding trap: if `pad_token_id` is absent and batch size is greater than 1, source raises.
- Text tensor layout is `[batch, seq, hidden]`; head layout becomes `[batch, heads, seq, head_dim]`. There is no NCHW/NHWC issue, but view/permute/softmax axes need a no-layout-translation guard.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer `input_ids [B,S]`, optional `attention_mask [B,total_kv]`, optional `position_ids [1 or B,S]`, optional `token_type_ids [B,S]`
- embedding lookup `wte: [vocab_size,H] -> [B,S,H]`
- optional token type embedding through `wte(token_type_ids)` and add
- `arange`, cache-length add, `unsqueeze` for default position IDs
- sinusoidal table gather by position ID: table `[max_pos, 2 * rotary_dim/2]`, gathered `[B,S,rotary_dim]`
- split sin/cos, repeat-interleave on last dim, even/odd rotate, slice/concat rotary and pass-through channels
- reshape/view `[B,S,H] -> [B,S,A,D]`, permute to `[B,A,S,D]`, and reverse merge
- cache update/append for K/V
- hidden slice for `logits_to_keep`

Neural network primitives:

- LayerNorm over hidden axis: one per block plus final `ln_f`
- Linear Q/K/V `H -> H`, bias false
- Linear attention output `H -> H`, bias false
- MLP `Linear(H -> I)` with bias, `gelu_new`, `Linear(I -> H)` with bias
- residual 3-input add: `residual + attention + mlp`
- LM head `Linear(H -> vocab_size)` with bias true
- dropout as inference no-op

Attention primitives:

- causal self-attention MHA
- eager attention: fp32 Q/K matmul, divide by `sqrt(head_dim)`, add additive causal/padding mask, softmax over key axis, cast probabilities to value dtype, optional dropout, matmul with V
- FlashAttention2 path when `config._attn_implementation == "flash_attention_2"`
- no in-library GPT-J SDPA class at this commit; use eager-compatible fallback unless a DinoML fused attention replacement is admitted

Position/rotary ops:

- GPT-J sinusoidal table creation with theta fixed at `10000`
- GPT-J interleaved RoPE over `rotary_dim`, with pass-through tail when `rotary_dim < head_dim`
- cached keys are stored after RoPE

Generation/cache ops:

- `DynamicCache(config=config)` allocation when `use_cache=True` and no cache is supplied
- `past_key_values.get_seq_length()` for default position offset
- per-layer `cache.update(key, value, layer_idx)`
- cache tensors logically `[B, num_heads, cached_seq, head_dim]`
- `logits_to_keep` int or tensor selection before LM head

Tokenizer/preprocessing-coupled ops:

- AutoTokenizer maps `gptj` to GPT2Tokenizer.
- Byte-level BPE files: `vocab.json`, `merges.txt`, optionally `tokenizer.json`.
- `EleutherAI/gpt-j-6b` tokenizer config: `tokenizer_class=GPT2Tokenizer`, `model_max_length=2048`, `add_prefix_space=false`, `errors=replace`, unk/bos/eos all `<|endoftext|>`.

Optional heads:

- Sequence classification: `Linear(H -> num_labels, bias=false)` at every token, then gather rightmost non-pad token.
- Question answering: `Linear(H -> num_labels)` then split start/end logits and squeeze last dim.

## 5. Layer/block breakdown

Let `H=n_embd`, `A=n_head`, `D=H/A`, `R=rotary_dim`, `I=n_inner or 4H`, current token chunk `T`, and total KV length `Ktot`.

Base model:

```text
input_ids [B,T] or inputs_embeds [B,T,H]
hidden = wte(input_ids)                                      # [B,T,H]
if token_type_ids: hidden += wte(token_type_ids)
position_ids = arange(T) + past_seen_tokens                  # [1,T] unless caller supplies
causal_mask = create_causal_mask(...)
hidden = emb_dropout(hidden)                                 # eval no-op
for each block:
  hidden = GPTJBlock(hidden, causal_mask, position_ids, cache)
hidden = ln_f(hidden)
```

Decoder block, repeated `n_layer` times:

```text
residual = hidden
x = LayerNorm(hidden, eps=layer_norm_epsilon)

q = Linear_q(x)                                               # [B,T,H], no bias
k = Linear_k(x)                                               # [B,T,H], no bias
v = Linear_v(x)                                               # [B,T,H], no bias
q = view(q, [B,T,A,D])                                        # rotary layout
k = view(k, [B,T,A,D])
v = view(v, [B,T,A,D]).permute(0,2,1,3)                       # [B,A,T,D]
q, k = gptj_rope(q, k, position_ids, rotary_dim=R)            # still [B,T,A,D]
q = q.permute(0,2,1,3)                                        # [B,A,T,D]
k = k.permute(0,2,1,3)
k, v = cache.update(k, v, layer_idx) if cache else (k, v)     # [B,A,Ktot,D]
attn = causal_attention(q, k, v, causal_mask)                 # [B,A,T,D]
attn = merge_heads(attn)                                      # [B,T,H]
attn = Linear_out(attn)                                       # no bias

mlp = Linear_fc_in(x)                                         # [B,T,I], bias
mlp = gelu_new(mlp)
mlp = Linear_fc_out(mlp)                                      # [B,T,H], bias
hidden = residual + attn + mlp
```

LM head:

```text
selected = hidden[:, slice_indices, :]                        # `logits_to_keep`
logits = Linear(selected, H -> vocab_size, bias=true)
```

For GPT-J 6B shapes: Q/K/V/out projections are `4096 -> 4096`; MLP is `4096 -> 16384 -> 4096`; LM head is `4096 -> 50400`.

## 6. Attention requirements

Required variant: causal self-attention, full MHA.

| Property | GPT-J requirement |
| --- | --- |
| Query heads | `num_attention_heads` |
| KV heads | same as query heads; no GQA/MQA repeat |
| Head dim | `hidden_size // num_attention_heads` |
| Position encoding | GPT-J RoPE applied to Q/K before cache update |
| Cache layout | per layer K/V `[B, heads, seq, head_dim]` |
| Masking | causal mask plus optional padding mask from `attention_mask` |
| Sliding/local attention | absent |
| Packed/varlen metadata | absent in model source; FlashAttention helper may internally unpad for padding masks |

Eager math order from source:

```text
query_fp32 = query.to(float32)
key_fp32 = key.to(float32)
scores = query_fp32 @ key_fp32.transpose(-1, -2)
scores = scores / sqrt(head_dim)
scores = scores + attention_mask      # if present
probs = softmax(scores, dim=-1)
probs = probs.to(value.dtype)
context = probs @ value
```

Cache details:

```text
prefill q/k/v before cache: q,k,v [B,heads,S,D] after RoPE for q/k
prefill stored cache:       k,v [B,heads,S,D], with K already RoPE-applied
decode new q/k/v:           [B,heads,T,D]
decode stored cache after:  [B,heads,P+T,D]
attention consumes:         q [B,heads,T,D], k/v [B,heads,P+T,D]
```

Backend/fallback notes:

- `GPTJ_ATTENTION_CLASSES` contains only `"eager"` and `"flash_attention_2"`.
- `GPTJPreTrainedModel` declares `_supports_flash_attn = True`; it does not declare GPT-J-specific SDPA support in this file.
- FlashAttention2 path computes the same Q/K/V/RoPE/cache update, then transposes to `[B,T,heads,D]`, handles accidental fp32 input recast, and calls `_flash_attention_forward(..., is_causal=True)`.
- FlashAttention versions before 2.1 may use top-left causal masks; source carries a compatibility flag. DinoML should prefer its own bottom-right causal semantics and test `q_len != k_len` decode.
- `output_attentions=True` is not useful for first integration; source returns `attn_weights` for eager, while FlashAttention returns the attention output tensor under that variable name.

## 7. Position encoding and custom math

GPT-J sinusoidal table:

```python
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (arange(0, dim, 2).float() / dim))
    sinusoid = outer(arange(num_pos).float(), inv_freq)
    return cat([sin(sinusoid), cos(sinusoid)], dim=1)
```

GPT-J interleaved rotary apply:

```python
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return stack([-x2, x1], dim=-1).flatten(-2)

def apply_gptj_rope(t, sin, cos):
    sin = repeat_interleave(sin[:, :, None, :], 2, dim=3)
    cos = repeat_interleave(cos[:, :, None, :], 2, dim=3)
    return t * cos + rotate_every_two(t) * sin
```

Partial rotary path:

```python
q_rot, q_pass = q[..., :R], q[..., R:]
k_rot, k_pass = k[..., :R], k[..., R:]
q = cat([apply_gptj_rope(q_rot, sin, cos), q_pass], dim=-1)
k = cat([apply_gptj_rope(k_rot, sin, cos), k_pass], dim=-1)
```

Precompute opportunities:

- `embed_positions [max_positions, rotary_dim]` is config-dependent and can be stored as a constant for default GPT-J.
- For decode with monotonic positions, direct slice by cache offset is faster than gather by arbitrary `position_ids`.
- Caller-supplied arbitrary `position_ids` require the gather path.

Failure traps:

- Do not use GPT-NeoX half-rotation. GPT-J rotates adjacent pairs.
- `rotary_dim` must be even for the current table/repeat pattern.
- Rotary table length is capped by `n_positions`; long-context extension is not implemented in this source.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- Tokenizer is GPT-2 byte-level BPE, using `vocab.json` and `merges.txt` or `tokenizer.json`.
- `EleutherAI/gpt-j-6b` tokenizer metadata: `model_max_length=2048`, `add_prefix_space=false`, `errors=replace`, special tokens unk/bos/eos all `<|endoftext|>`.
- Model inputs are `input_ids` and optional `attention_mask`. Padding policy is external because GPT-J config has `pad_token_id=null` for the main checkpoint.

GPU/runtime inputs:

- Exactly one of `input_ids [B,T]` or `inputs_embeds [B,T,H]`.
- Optional `attention_mask [B,P+T]`; Transformers converts it through `create_causal_mask`.
- Optional `position_ids [B,T]`; default is `arange(T) + cache_length`.
- Optional `token_type_ids [B,T]`; source embeds them with `wte` and adds to hidden states.
- Optional `past_key_values`; source creates `DynamicCache(config)` when `use_cache=True`.

Generation-controller behavior outside the compiled graph:

- `GenerationMixin` owns greedy/sampling/beam loops, logits processors, stopping, and cache reorder.
- `task_specific_params.text-generation` in `EleutherAI/gpt-j-6b` config says `do_sample=true`, `max_length=50`, `temperature=1.0`; this should live outside the core DinoML module.
- First integration can validate core graph with greedy one-token decode and defer beam search, sampling processors, and cache reorder.

No multimodal placeholders, image/audio preprocessing, packed patch descriptors, or `cu_seqlens` are part of GPT-J.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> grouped attention projection schedule

Source pattern:

```text
q = Linear_q(x); k = Linear_k(x); v = Linear_v(x)
view each to heads; apply RoPE to q/k; attention
```

Replacement pattern:

```text
three GEMMs scheduled as a projection group, or a packed-QKV constant synthesized by loader under an explicit provenance flag
```

Preconditions:

- All three linears are `H -> H`, bias false.
- No consumer observes intermediate projection tensors.
- Weight packing, if used, records split order `[q, k, v]` and does not overwrite source checkpoint layout silently.

Shape equations:

- Input `[B,T,H]`, flattened GEMM M=`B*T`, K=`H`, N=`H` for each projection.
- Packed optional output `[B,T,3H]` splits into Q, K, V each `[B,T,H]`.

Failure cases:

- Treating GPT-J as source-packed QKV without loader-side transform.
- Tying or aliasing projection weights accidentally.

Parity test sketch: compare Q/K/V tensors after projection and head view for random hidden states in tiny and GPT-J 6B dimensions.

### Rewrite: GPT-J RoPE + cache write

Source pattern:

```text
gather sincos(position_ids) -> split sin/cos -> repeat_interleave
slice q/k rotary channels -> rotate_every_two -> concat pass-through -> cache.update(k,v)
```

Replacement pattern:

```text
fused_gptj_rope_qk kernel; decode path writes rotated K directly into session cache
```

Preconditions:

- `rotary_dim` is even and `0 < rotary_dim <= head_dim`.
- RoPE table uses GPT-J default theta 10000.
- Cache stores post-RoPE K.
- Position IDs are monotonic or the kernel supports arbitrary table gather.

Failure cases:

- `rotary_dim=None` full-head mode not implemented.
- GPT-NeoX-style rotate-half accidentally used.
- Position ID exceeds precomputed table length.

Parity test sketch: random Q/K with `rotary_dim=4` and `64`; compare rotated channels and pass-through tail against Transformers.

### Rewrite: causal attention -> fused prefill/decode attention

Source pattern:

```text
fp32(QK^T) / sqrt(D) + mask -> softmax -> cast -> probs @ V
```

Replacement pattern:

```text
DinoML fused causal MHA with optional padding mask and KV cache
```

Preconditions:

- `output_attentions=False`.
- Dropout disabled/eval.
- Full MHA, no GQA.
- Mask is pure causal or standard causal + 2D padding mask.
- Fused kernel numerical policy is admitted against fp32-score eager behavior.

Failure cases:

- Requested attention weights.
- Unsupported additive masks.
- FlashAttention top-left mask compatibility for old external kernels.

Parity test sketch: prefill and decode logits for no mask and padding mask; include `q_len=1`, `q_len>1`, and `past_len>0`.

### Rewrite: parallel residual branch fusion

Source pattern:

```text
x = ln_1(hidden)
hidden = residual + attn(x) + mlp(x)
```

Replacement pattern:

```text
reuse one LayerNorm result for both branches; fuse final 3-input residual add
```

Preconditions:

- Inference mode and dropout erased.
- `output_hidden_states`/debug observability does not require branch intermediates as public outputs.

Failure cases:

- Lowering as sequential block changes numerics and semantics.
- Training dropout.

Parity test sketch: one-block comparison with branch outputs enabled in a local harness before fused final add.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None) if int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

Replacement pattern:

```text
slice/gather selected hidden positions before vocab GEMM
```

Preconditions:

- Generation decode needs only newest token logits, usually `logits_to_keep=1`.
- No labels/loss path.

Failure cases:

- Full prompt logprobs or arbitrary tensor `logits_to_keep` not supported by the optimized path.

Parity test sketch: compare full logits last position to `logits_to_keep=1`; verify arbitrary tensor indices either match or are rejected with a clear diagnostic.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over hidden axis: one block norm plus final norm; feeds both branches, so reuse matters.
- GPT-J RoPE kernel: interleaved even/odd, partial rotary, on the decode critical path.
- Causal MHA prefill/decode with post-RoPE KV cache: dominant runtime cost.
- Parallel residual 3-input add and branch scheduling: required for correct block topology and useful for memory traffic.
- Last-token LM head GEMM with bias: avoids full `[B,S,V]` logits during decode.

Medium priority:

- Projection group for separate Q/K/V GEMMs; optional loader-packed QKV only with explicit provenance.
- Attention output projection plus residual epilogue candidates.
- MLP `Linear -> gelu_new -> Linear`; activation fusion or CUTLASS epilogue experiments.
- Cache append/write fused with K RoPE in decode.
- GGUF/runtime-dequant GEMM for large dense weights after fp16/bf16 dense parity, especially MLP and LM head.

Lower priority:

- Token type embedding add.
- Exact FlashAttention2 wrapper emulation; prefer DinoML fused attention once eager parity is proven.
- Sequence classification and QA heads.
- Beam cache reorder and advanced generation processors.

## 11. Runtime staging plan

1. Parse `GPTJConfig`, normalize/ignore legacy config-only fields with diagnostics, and load GPT-2 tokenizer metadata for BOS/EOS/max length.
2. Load weights for embeddings, LayerNorms, separate q/k/v/out projections, MLP, final norm, and bias-bearing LM head. Preserve untied embedding/LM-head weights unless checkpoint metadata proves aliasing.
3. Implement GPT-J RoPE unit parity and one decoder block in fp32/eval without cache.
4. Add full prefill parity for tiny GPT-J and GPT-J-6B-like dimensions, initially using eager-compatible attention.
5. Add DynamicCache-equivalent session KV storage with post-RoPE K and decode position offset.
6. Add `logits_to_keep=1` last-token LM head path for decode.
7. Replace eager attention with fused causal MHA under no-attention-output and supported-mask guards.
8. Add graph rewrites/fusions for RoPE+cache write, parallel residual add, MLP activation, and projection scheduling.
9. Add larger checkpoint loading and quantized/GGUF experiments only after dense generation parity is stable.

Initial stubs: dropout as identity in eval, no labels/loss, no `output_attentions`, no classification/QA heads, no beam search/cache reorder, no tokenizer execution inside compiled GPU graph.

## 12. Parity and validation plan

- Config tests: source defaults, GPT-J 6B config, tiny configs; assert `head_dim`, `rotary_dim`, MLP width, projection bias flags, LM-head bias.
- Tokenizer smoke: GPT-2 BPE encodes leading-space variants differently; BOS/EOS ID is 50256 for GPT-J 6B.
- RoPE primitive tests: compare `create_sinusoidal_positions`, gather by position IDs, `rotate_every_two`, partial rotary with pass-through tail.
- Projection tests: separate q/k/v/out linears with bias false and MLP/LM-head bias true.
- One-block parity: fp32 random tensors with no cache and with cache, including parallel residual.
- Prefill logits parity: tiny random GPT-J first, then a small prompt on `EleutherAI/gpt-j-6b` if weights are available.
- Decode parity: prefill prompt, then feed one and three new tokens with cache; compare to full recompute over concatenated input.
- Attention mask parity: no padding, right padding, and left padding if tokenizer/caller sets a pad token.
- Last-token logits: compare `logits_to_keep=1` with slicing full logits.
- Backend parity: eager vs DinoML fused attention under pure causal and causal+padding masks.

Recommended tolerances: fp32 eager `rtol=1e-4, atol=1e-4`; fp16/bf16 fused attention start around `rtol=5e-3, atol=5e-3` for hidden/logits, adjusting only with documented backend math differences.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Prefill throughput sweep by sequence length `{1,16,128,512,2048}` and batch size.
- Decode tokens/sec by batch size and cache length.
- KV cache memory: `2 * layers * B * heads * seq * head_dim * dtype_size`; GPT-J 6B fp16 at `B=1,S=2048` is about `2*28*16*2048*256*2 = 939,524,096` bytes.
- RoPE kernel time, including gather path vs monotonic-position slice path.
- Separate Q/K/V GEMM scheduling vs synthesized packed QKV projection.
- Attention backend comparison: eager matmul/softmax, FlashAttention-like, DinoML decode-specialized attention.
- MLP GEMM/activation/GEMM time and activation fusion benefit.
- LM-head full-sequence logits vs last-token-only logits, including bias cost.
- GGUF/runtime-dequant GEMM probe for MLP up/down and LM head if quantized deployment is targeted.

All probes above are proposed; no benchmark observations are included.

## 14. Skip/defer list

- Training, labels/loss, dropout randomness, and gradient checkpointing.
- `output_attentions=True` and full hidden-state recording.
- Sequence classification and question answering heads.
- Beam search, cache reorder, sampling processors, repetition penalties, and other generation-controller internals.
- Exact FlashAttention2 wrapper behavior and old top-left mask compatibility, beyond parity tests for the selected DinoML attention backend.
- `rotary_dim=None` full-head mode unless a real target config requires it.
- Historical config fields not read by current source, such as `rotary`, `scale_attn_weights`, `intermediate_size`, `hidden_act`, and `type_vocab_size`, except for compatibility diagnostics.
- Tensor parallel, pipeline parallel, quantized kernels, and GGUF/offload paths until dense generation works.
- Tokenizer execution inside DinoML runtime; keep byte-level BPE in CPU/data pipeline first.

## 15. Final implementation checklist

- [ ] Parse `GPTJConfig` and representative tokenizer/config metadata.
- [ ] Reject or document ignored historical config fields that current source does not read.
- [ ] Load embeddings, LayerNorms, separate q/k/v/out linears, MLP, and bias-bearing LM head.
- [ ] Preserve untied embedding and LM-head weights unless aliasing is explicit.
- [ ] Implement GPT-J interleaved partial RoPE with `rotary_dim`.
- [ ] Implement position ID generation from cache length and table gather.
- [ ] Implement LayerNorm and `gelu_new`.
- [ ] Implement GPT-J parallel residual decoder block.
- [ ] Implement eager causal MHA parity with fp32 Q/K score math.
- [ ] Implement artifact-visible KV cache storing post-RoPE K/V `[B,heads,seq,D]`.
- [ ] Implement decode cache append and position offset.
- [ ] Implement `logits_to_keep=1` last-token LM head with bias.
- [ ] Add RoPE, projection, one-block, prefill, and decode parity tests against Transformers.
- [ ] Add attention-mask parity tests with and without padding.
- [ ] Add fused attention lowering under strict mask/cache guards.
- [ ] Benchmark prefill, decode, KV memory, RoPE, Q/K/V GEMMs, MLP, and LM-head slicing.
