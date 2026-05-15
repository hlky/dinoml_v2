# CodeGen Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model family: `codegen`

Primary runtime target: `CodeGenForCausalLM` decoder-only text generation for code completion.

Model ids/configs inspected:

| Model id | Config snapshot | Local snapshot |
|---|---:|---|
| [Salesforce/codegen-350M-nl](https://huggingface.co/Salesforce/codegen-350M-nl) | `ecc2504ab5e8a03db45a271c078fb4b1ab5588ec` | `_sources/Salesforce_codegen-350M-nl/config.json` |
| [Salesforce/codegen-350M-mono](https://huggingface.co/Salesforce/codegen-350M-mono) | `d9107f71cca463240db1143f4a75a927a27fcb27` | `_sources/Salesforce_codegen-350M-mono/config.json` |
| [Salesforce/codegen-2B-mono](https://huggingface.co/Salesforce/codegen-2B-mono) | `0b1d0b33f26b1416f66f0ecf07cf6a29438c95ea` | `_sources/Salesforce_codegen-2B-mono/config.json` |
| [Salesforce/codegen-6B-mono](https://huggingface.co/Salesforce/codegen-6B-mono) | `62dfb58dbc7b5f04a3bc9b3ce0786fc82f1871b8` | `_sources/Salesforce_codegen-6B-mono/config.json` |
| [Salesforce/codegen-16B-multi](https://huggingface.co/Salesforce/codegen-16B-multi) | `049249f58352a8311dde257bded8e180d3f28f9b` | `_sources/Salesforce_codegen-16B-multi/config.json` |

Source files inspected:

- `transformers/src/transformers/models/codegen/modeling_codegen.py`
  - GitHub URL: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/codegen/modeling_codegen.py>
- `transformers/src/transformers/models/codegen/configuration_codegen.py`
  - GitHub URL: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/codegen/configuration_codegen.py>
- `transformers/src/transformers/models/codegen/tokenization_codegen.py`
  - GitHub URL: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/codegen/tokenization_codegen.py>

Any missing files or assumptions:

- There is no `tokenization_codegen_fast.py` in this pinned source directory.
- All inspected official Salesforce checkpoints were accessible without 401/403.
- Config snapshots and tokenizer snapshots were copied only under `agents/plans/transformers/codegen/_sources/`.
- The source is the in-library implementation, not remote code.

## 2. High-level architecture

CodeGen is a text-only, decoder-only causal LM. It is GPT-J-like in broad shape, but with CodeGen-specific QKV packing and a parallel residual block: one LayerNorm feeds both self-attention and MLP, then attention output, MLP output, and the residual are added together.

Dataflow:

```text
byte-level BPE preprocessing -> token embeddings -> N decoder blocks -> final LayerNorm -> LM head -> logits/sampling
```

Runtime stages:

```text
CPU/data pipeline:
  CodeGenTokenizer byte-level BPE, attention_mask construction, optional decode truncation helper

GPU/runtime:
  embedding lookup -> prefill decoder -> KV cache creation/update -> decode decoder step -> last-token or selected logits
```

Independently stageable pieces:

- Tokenizer parity can be validated separately from the GPU graph.
- One decoder block can be validated with synthetic hidden states, position ids, and masks.
- Prefill and decode can be validated separately because cache update happens inside each attention layer.
- The LM head can use `logits_to_keep` to avoid full-sequence logits in generation.

Implemented heads:

- `CodeGenModel`: required as the base decoder for the target.
- `CodeGenForCausalLM`: required for code generation.
- Training loss path in `CodeGenForCausalLM.forward(labels=...)`: optional/deferred for inference.

## 3. Important config dimensions

Source defaults from `CodeGenConfig`:

| Field | Default | Source effect |
|---|---:|---|
| `vocab_size` | 50400 | Embedding/LM head width if checkpoint omits it |
| `n_positions` | 2048 | Max positions for sinusoidal RoPE table |
| `n_ctx` | 2048 | Used only for a stored `CodeGenModel.rotary_dim = min(config.rotary_dim, config.n_ctx // n_head)` value; attention uses `config.rotary_dim` directly |
| `n_embd` | 4096 | Hidden size |
| `n_layer` | 28 | Decoder block count |
| `n_head` | 16 | Query/key/value head count |
| `head_dim` | inferred | `n_embd // n_head`; source rejects non-divisible cases |
| `rotary_dim` | 64 | First rotary dimensions of each q/k head |
| `n_inner` | `None` | Effective MLP width is `4 * n_embd` |
| `activation_function` | `gelu_new` | MLP activation through `ACT2FN` |
| `use_cache` | `True` | Creates `DynamicCache` when requested |
| `tie_word_embeddings` | `False` | Checkpoints declare untied embeddings/head, though `_tied_weights_keys` names a possible tie key for framework utilities |

Representative checkpoint sweep from official `config.json` plus tokenizer configs:

| Model | Dataset branch | Layers | Hidden | Heads | Head dim | MLP width | RoPE dim | Max pos | Vocab | Dtype |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `codegen-350M-nl` | NL | 20 | 1024 | 16 | 64 | 4096 | 32 | 2048 | 51200 | `float16` |
| `codegen-350M-mono` | Mono/Python | 20 | 1024 | 16 | 64 | 4096 | 32 | 2048 | 51200 | `float16` |
| `codegen-2B-mono` | Mono/Python | 32 | 2560 | 32 | 80 | 10240 | 64 | 2048 | 51200 | `float16` |
| `codegen-6B-mono` | Mono/Python | 33 | 4096 | 16 | 256 | 16384 | 64 | 2048 | 51200 | `float16` |
| `codegen-16B-multi` | Multi-language code | 34 | 6144 | 24 | 256 | 24576 | 64 | 2048 | 51200 | `float16` |

All swept checkpoints use `activation_function="gelu_new"`, `n_inner=null`, dropout probabilities `0.0`, `use_cache=true`, `bos_token_id=1`, `eos_token_id=50256`, and tokenizer `model_max_length=2048`.

## 3a. Family variation traps

- No MQA/GQA: `num_key_value_heads` is absent. Q, K, and V all use `n_head`.
- `head_dim` varies materially: 350M has 64, 2B has 80, and 6B/16B have 256.
- RoPE is partial for normal checkpoints: `rotary_dim < head_dim` for all swept configs. A fused attention path must rotate only the first `rotary_dim` q/k channels and concatenate the unrotated pass-through tail.
- QKV packing is not GPT-2's `q,k,v` and not a simple all-Q/all-K/all-V split. Source uses `qkv_proj: Linear(hidden -> 3*hidden, bias=False)`, reshapes to `(..., mp_num=4, -1)`, then splits each `mp_num` shard as `query, value, key`.
- The attention score order is `matmul -> mask add -> divide by sqrt(head_dim) -> softmax`, not the common `scale query before matmul` form. If masks are additive `-inf`-like values this is usually equivalent, but parity tests should preserve source order.
- Attention q/k matmul is upcast to fp32, then softmax output is downcast to `value.dtype` before multiplying V.
- The decoder block is parallel residual: MLP consumes the same normalized hidden states as attention, not the attention result.
- Projection biases: attention QKV and output projections are biasless. MLP and LM head use PyTorch `nn.Linear` defaults, so they have bias unless weights prove otherwise.
- `tie_word_embeddings=false` in configs. Treat input embedding and LM head as separate logical parameters for these checkpoints.
- Configs contain historical fields not read by this pinned modeling source: `scale_attn_weights`, `summary_*`, `gradient_checkpointing`, and `task_specific_params`. Do not turn them into runtime requirements for this in-library implementation.
- Config `tokenizer_class` says `GPT2Tokenizer`, while `tokenizer_config.json` says `CodeGenTokenizer`. The model coupling that matters is byte-level BPE, `<|endoftext|>` special token semantics, no prefix space by default, and max length 2048.
- No sliding-window/local attention, ALiBi, relative bias, long-context RoPE scaling, MoE, quantized/packed source weights, image/audio branches, packed varlen metadata, or tensor parallel runtime branches are implemented in this source.
- Source tensors are sequence-major logical PyTorch layouts such as `[batch, seq, hidden]` and attention `[batch, heads, query, key]`; no NHWC/channel-last rewrite is applicable.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `input_ids [B,S] -> hidden [B,S,H]`.
- Optional token-type embedding lookup and add using the same `wte`.
- Reshape/view: QKV reshape to `[B,S,4,3*H/4]`; head split/merge; final `view(-1, S, H)`.
- Permute/transpose: V to `[B,heads,S,D]`; Q/K to `[B,heads,S,D]`; K transpose for scores; attention output merge.
- Split and concatenate for Q/V/K packing and partial RoPE.
- Slice/index: `embed_positions[position_ids]`, `hidden_states[:, slice_indices, :]` for `logits_to_keep`.

Neural network primitives:

- `Embedding(vocab_size -> H)`.
- `LayerNorm(H, eps=1e-5)` before each block and final LayerNorm.
- Biasless attention QKV `Linear(H -> 3H)`.
- Biasless attention output `Linear(H -> H)`.
- MLP `Linear(H -> 4H)` with bias, `gelu_new`, then `Linear(4H -> H)` with bias.
- LM head `Linear(H -> vocab_size)` with bias in source.
- Dropout modules exist but are zero-probability in swept configs and disabled in inference.

Attention primitives:

- Dense causal self-attention only.
- Additive causal/padding mask from `create_causal_mask`.
- MHA, no MQA/GQA.
- fp32 QK score matmul and softmax, value-dtype AV matmul.
- KV cache update through `Cache.update(key, value, layer_idx)`.

Position/rotary ops:

- Precomputed sinusoidal table shaped `[max_position_embeddings, rotary_dim or hidden_size]`.
- Gather by `position_ids`.
- Split sin/cos halves.
- Repeat-interleave sin/cos across adjacent rotary pairs.
- `rotate_every_two` for q/k rotary channels.

Generation/cache ops:

- Dynamic cache allocation when `use_cache=True` and no cache is passed.
- Position ids offset by `past_key_values.get_seq_length()`.
- Per-layer cache reorder support comes from the common Transformers cache object, not this model file.
- `logits_to_keep` last-token/sparse logits slicing.

Preprocessing-coupled ops:

- Byte-level BPE with `add_prefix_space=false` by default.
- Optional `truncate_before_pattern` decode post-processing for code-generation demos; this is outside the model graph.

Not required:

- Sparse/local/block attention.
- Hash/sort/bucket attention.
- Cross-attention.
- Recurrent/state-space cache.
- Vision/audio preprocessing.
- Packed varlen `cu_seqlens`.
- Quantized source-specific kernels.

## 5. Layer/block breakdown

Embedding and input setup:

```text
input_ids [B,S] -> wte -> x [B,S,H]
optional token_type_ids [B,S] -> wte -> add into x
position_ids [1,S] or [B,S], offset by past length for decode
causal_mask = create_causal_mask(config, x, attention_mask, past_key_values, position_ids)
```

Decoder block, repeated `n_layer` times:

```text
residual = x
x_norm = LayerNorm(x)

qkv = Linear_no_bias(H -> 3H)(x_norm)
qkv = reshape(qkv, [B,S,4,3H/4])
query, value, key = split(qkv, local_dim=H/4, dim=-1)
query/key/value = split_heads(..., n_head, head_dim, mp_num=4)
value = permute(value, [B,heads,S,D])

sincos = embed_positions[position_ids]
sin, cos = split(sincos, rotary_dim/2)
query,key first rotary_dim channels = RoPE(query,key)
query,key pass-through tail = unchanged
query,key = permute to [B,heads,S,D]

key,value = cache.update(post_rope_key, value, layer_idx) if cache exists
scores = matmul(float32(query), float32(key).transpose(-1,-2))
scores = scores + additive_mask
probs = softmax(scores / sqrt(head_dim))
attn = matmul(probs.to(value.dtype), value)
attn = merge_heads(attn) -> [B,S,H]
attn = Linear_no_bias(H -> H)(attn)

mlp = Linear_bias(H -> 4H)(x_norm)
mlp = gelu_new(mlp)
mlp = Linear_bias(4H -> H)(mlp)

x = residual + attn + mlp
```

Final head:

```text
x = LayerNorm(x)
selected = x[:, -logits_to_keep:, :] or indexed selection
logits = Linear_bias(H -> vocab_size)(selected)
```

## 6. Attention requirements

Attention type:

- Causal decoder self-attention.
- Dense global attention over all cached/present tokens.
- MHA: `q_heads = k_heads = v_heads = n_head`.
- No GQA/MQA repeat expansion.
- No local/sliding-window admission rules.
- No ALiBi or learned relative bias.

Shapes:

```text
hidden_states: [B,S,H]
q,k,v before attention: [B,heads,S,D]
cache key/value per layer: [B,heads,T,D]
scores: [B,heads,S,T]
attn output before merge: [B,heads,S,D]
```

For prefill, `T=S` unless an external cache is passed. For decode with one new token, `S=1` and `T=past_length+1`.

Cache layout:

- The key stored in cache is after RoPE and after the `key.permute(0,2,1,3)` to `[B,heads,S,D]`.
- The source casts key to `hidden_states.dtype` before cache update. Value is already in that dtype.
- The cache object receives one update per layer via `layer_idx`; layer construction passes explicit indices.

Masking and math order:

```text
query_fp32 = query.to(float32)
key_fp32 = key.to(float32)
scores = query_fp32 @ key_fp32.transpose(-1, -2)
scores = scores + attention_mask
scores = scores / sqrt(head_dim)
probs = softmax(scores, dim=-1)
probs = probs.to(value.dtype)
output = probs @ value
```

This ordering is important for fused attention parity. A FlashAttention backend can be used only if it can reproduce partial RoPE, additive causal/padding masks, cache layout, fp32 score accumulation behavior, and the source's effective mask/scale order. The source itself uses eager matmul/softmax/matmul, not SDPA or FlashAttention dispatch.

## 7. Position encoding and custom math

CodeGen uses GPT-J-style rotary embeddings backed by a precomputed sinusoidal table. The table has shape `[max_position_embeddings, rotary_dim]` when `rotary_dim` is configured; otherwise it uses `hidden_size`.

Source-equivalent sketch:

```python
def codegen_sincos(max_pos, rotary_dim):
    inv_freq = 1.0 / (10000 ** (arange(0, rotary_dim, 2) / rotary_dim))
    sinusoid = outer(arange(max_pos), inv_freq)
    return concat([sin(sinusoid), cos(sinusoid)], dim=1)

def rotate_every_two(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return stack([-x2, x1], dim=-1).flatten(-2)

def apply_codegen_rope(t, sin, cos):
    sin = repeat_interleave(sin[:, :, None, :], repeats=2, dim=3)
    cos = repeat_interleave(cos[:, :, None, :], repeats=2, dim=3)
    return t * cos + rotate_every_two(t) * sin
```

For checkpoint configs, only the first `rotary_dim` channels of q/k are rotated:

```text
q = concat([rope(q[..., :rotary_dim]), q[..., rotary_dim:]], dim=-1)
k = concat([rope(k[..., :rotary_dim]), k[..., rotary_dim:]], dim=-1)
```

Precomputable:

- `embed_positions` sin/cos table up to `n_positions`.

Dynamic inputs:

- `position_ids`, including the decode offset from cache length.
- Device placement of the table. Source mutates the non-persistent buffer to the `position_ids` device.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Tokenization is byte-level BPE implemented through the `tokenizers` library.
- Tokenizer config uses `CodeGenTokenizer`, `add_prefix_space=false`, and `model_max_length=2048`.
- Special tokens map `bos_token`, `eos_token`, and `unk_token` to `<|endoftext|>`.
- Checkpoint `config.json` sets `eos_token_id=50256`, while `bos_token_id=1`; tokenizer special-token string still maps through vocabulary files. For generation parity, prefer loading tokenizer files instead of assuming BOS/EOS ids from config alone.
- `model_input_names` are `input_ids` and `attention_mask`; `token_type_ids` appear only if tokenizer construction asks for `return_token_type_ids`.

GPU graph inputs:

- Required: either `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one.
- Optional: `attention_mask`, `position_ids`, `token_type_ids`, and `past_key_values`.
- `token_type_ids`, if provided, are embedded through the same `wte` and added to token embeddings. This is not segment embedding with a separate table.

Generation-controller behavior outside the core graph:

- `logits_to_keep=1` or equivalent should be used for decode/last-token logits to avoid full `[B,S,V]` projection.
- The tokenizer `decode(..., truncate_before_pattern=...)` can remove extra generated code after patterns such as comments or repeated definitions, but this is post-processing and can be deferred.

## 9. Graph rewrite / lowering opportunities

### Rewrite: CodeGen QVK-packed projection split

Source pattern:

```text
qkv = Linear_no_bias(H -> 3H)(x)
qkv = reshape(qkv, [B,S,4,3H/4])
query, value, key = split(qkv, H/4, dim=-1)
split_heads each tensor with n_head/4 heads per mp shard
```

Replacement pattern:

```text
single GEMM H -> 3H -> layout-aware split into Q,V,K head tensors
```

Exact preconditions:

- `n_head % 4 == 0`.
- `H % n_head == 0`.
- `H * 3` divisible by `mp_num=4`.
- Source weight layout is PyTorch Linear `[out_features, in_features]`.
- Split order must remain `query, value, key` within each `mp_num` shard.

Shape equations:

```text
D = H / n_head
local_dim = H / 4
qkv_split shape = [B,S,4,3H/4]
each split = [B,S,4,H/4] -> [B,S,n_head,D]
```

Weight transform:

- Avoid physical transforms for first integration; emit source-faithful split.
- Optional future transform can reorder weights to canonical Q,K,V all-head blocks, but must also reorder checkpoint weights from per-shard Q,V,K layout.

Failure cases:

- Treating the projection as GPT-2-style `[Q all heads, K all heads, V all heads]` is wrong.
- Treating split order as Q,K,V is wrong.

Parity test sketch:

- Generate random `qkv_proj.weight`, run source split and rewritten split, compare Q/K/V tensors before RoPE for 350M, 2B, and 6B dimensions.

### Rewrite: partial RoPE plus attention prefill

Source pattern:

```text
slice rotary head channels -> apply RoPE -> concat pass-through tail -> attention
```

Replacement:

```text
fused QK partial-RoPE kernel feeding attention backend
```

Preconditions:

- `rotary_dim` is even and `0 < rotary_dim <= head_dim`.
- Position ids are monotonic or backend supports gather by arbitrary `position_ids`.
- Backend stores post-RoPE K in cache.

Failure cases:

- Full-head RoPE on 6B/16B would rotate 256 channels instead of 64.
- RoPE after cache update would store unrotated keys and break decode parity.

Parity test sketch:

- Compare q/k after RoPE for synthetic `position_ids` with and without past offset.

### Rewrite: parallel residual block fusion

Source pattern:

```text
x_norm = LayerNorm(x)
attn = Attention(x_norm)
mlp = MLP(x_norm)
y = x + attn + mlp
```

Replacement:

```text
LayerNorm once -> run attention and MLP branches -> fused residual add3
```

Preconditions:

- Both branches consume exactly the same normalized tensor.
- Inference mode dropout is disabled or `p=0`.

Failure cases:

- Rewriting as sequential GPT-2 style `x + attention`, then second norm/MLP is wrong.

Parity test sketch:

- One block parity with dropout disabled and fixed random tensors.

### Rewrite: last-token LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, -k:, :])
```

Replacement:

```text
slice/gather selected hidden rows -> GEMM(H -> vocab)
```

Preconditions:

- Generation path does not need all sequence logits.
- `logits_to_keep` is integer `1` or another known positive count, or a valid tensor index list.

Failure cases:

- Training loss path needs shifted full logits and labels; defer it.

Parity test sketch:

- Compare full logits slice to direct selected-row GEMM for `k=1` and `k>1`.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm for `[B,S,H]` with H up to 6144. It is on every block and final output.
- Biasless QKV GEMM plus CodeGen split. Correct split order is required before attention kernels can be trusted.
- Partial RoPE plus KV cache write. This removes small slice/concat overhead and prevents cache layout mistakes.
- Causal MHA prefill/decode with post-RoPE KV cache. This dominates runtime.
- MLP `Linear -> gelu_new -> Linear`; the first GEMM is 4H wide and profiler-visible.
- Last-token LM head for decode. Avoiding full-sequence logits matters at vocab 51200.

Medium priority:

- Fused residual add3 after parallel attention/MLP branches.
- LayerNorm + branch input staging to feed attention and MLP without extra materialization.
- QK fp32-score policy for reduced precision checkpoints.
- Token-type embedding add can compose with embedding output for the rare path.

Lower priority:

- Dropout kernels, because inference has dropout disabled and swept configs use zero probabilities.
- Decode truncation regex behavior, because it is CPU post-processing.
- Full-output attention weights, hidden states, and training loss.

## 11. Runtime staging plan

Stage 1: Config and tokenizer admission

- Parse `CodeGenConfig`.
- Reject unsupported variants where `n_head % 4 != 0`, `hidden_size % n_head != 0`, odd `rotary_dim`, or `rotary_dim > head_dim`.
- Load tokenizer files for end-to-end text parity, but keep tokenizer outside GPU graph.

Stage 2: One-block eager parity

- Implement embedding, LayerNorm, QKV projection split, partial RoPE, dense attention, MLP, and add3.
- Validate one block against Transformers with synthetic weights.

Stage 3: Full prefill parity

- Run full decoder without cache for short prompts.
- Compare final hidden states and logits for 350M first.

Stage 4: Decode cache parity

- Implement per-layer KV cache `[B,heads,T,D]`.
- Store post-RoPE K and V.
- Validate one-token decode after prefill.

Stage 5: Optimized attention and GEMM planning

- Swap eager attention chain for an optimized causal attention backend under strict parity guards.
- Profile GEMMs for QKV, MLP, output projection, and LM head.

Stage 6: Production generation path

- Add `logits_to_keep=1` fast path, batching, and memory accounting for KV cache.
- Defer beam search and training loss.

## 12. Parity and validation plan

Custom op tests:

- `create_sinusoidal_positions`, `rotate_every_two`, and partial RoPE against source for multiple `rotary_dim/head_dim` pairs.
- CodeGen Q/V/K split against source for 350M, 2B, 6B, and 16B dimensions.
- Attention math with additive mask and fp32 QK scores.

Single-layer parity:

- Random hidden states and checkpoint-like dimensions.
- Compare block output after `LayerNorm -> attention + MLP + residual`.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-5`; fp16/bf16 `atol=2e-2, rtol=2e-2` initially, tightened after kernel choices settle.

Full model parity:

- 350M prefill logits for short code prompts, batch 1 and batch >1.
- Decode token parity: prefill prompt, run one generated step through cache, compare logits.
- `logits_to_keep=1` parity against full logits last-token slice.

Tokenizer/end-to-end:

- Byte-level BPE examples with and without leading spaces.
- Code prompt completion with fixed greedy sampling or deterministic next-token comparison.

Negative/admission tests:

- Reject unsupported `rotary_dim > head_dim`.
- Reject configs with `n_head % 4 != 0` unless a source-faithful general split is implemented.
- Reject remote-code-only or non-`CodeGenForCausalLM` variants under this report scope.

## 13. Performance probes

- Prefill tokens/sec by model size: 350M, 2B, 6B, 16B shape proxies.
- Decode tokens/sec with KV cache for batch-size sweep.
- Sequence-length sweep up to 2048 for prefill.
- KV cache memory usage:

```text
bytes = layers * 2 * batch * heads * seq * head_dim * dtype_bytes
```

- Attention backend comparison: eager BMM/softmax/BMM vs fused attention, preserving partial RoPE and cache layout.
- GEMM provider profile for QKV, MLP up/down projections, output projection, and LM head.
- Last-token LM head vs full-sequence LM head.
- Tokenizer throughput for CPU/data-pipeline bottleneck separation.
- Cache update bandwidth and memory-fragmentation probes for long decode.

## 14. Skip/defer list

- Training loss and label shifting.
- Dropout and gradient checkpointing behavior.
- Output attentions and all hidden states.
- Beam search, sampling policies, and repetition penalties beyond core logits.
- Decode regex truncation helper.
- Remote-code variants and fine-tunes that change architecture.
- Quantization and packed weight formats.
- Multi-GPU/tensor parallel runtime. Source has a hardcoded `mp_num=4` packing artifact, but no runtime tensor-parallel execution.
- Fast tokenizer implementation, because none exists in this pinned source directory.

## 15. Final implementation checklist

- [ ] Parse `CodeGenConfig` and checkpoint tokenizer metadata.
- [ ] Add admission guards for `hidden_size % n_head`, `n_head % 4`, `rotary_dim`, and `n_positions`.
- [ ] Load untied `wte` and `lm_head` weights as separate logical parameters.
- [ ] Implement CodeGen Q/V/K packed projection split.
- [ ] Implement partial GPT-J-style RoPE with gathered `position_ids`.
- [ ] Implement dense causal MHA with source math order and fp32 QK scores.
- [ ] Implement post-RoPE KV cache layout `[B, heads, T, head_dim]`.
- [ ] Implement parallel residual decoder block.
- [ ] Implement `gelu_new` MLP.
- [ ] Implement final LayerNorm and `logits_to_keep` LM head path.
- [ ] Add one-block parity tests for 350M/2B/6B shape families.
- [ ] Add prefill logits parity for `Salesforce/codegen-350M-mono`.
- [ ] Add decode cache parity with one-token continuation.
- [ ] Benchmark prefill, decode, KV memory, and LM head slicing.
