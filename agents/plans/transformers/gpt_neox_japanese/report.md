# GPTNeoX Japanese Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: abeja/gpt-neox-japanese-2.7b
Config source: HF config.json plus local configuration defaults
Source files inspected:
  X:/H/transformers/src/transformers/models/gpt_neox_japanese/configuration_gpt_neox_japanese.py
  X:/H/transformers/src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py
  X:/H/transformers/src/transformers/models/gpt_neox_japanese/tokenization_gpt_neox_japanese.py
  X:/H/transformers/tests/models/gpt_neox_japanese/test_modeling_gpt_neox_japanese.py
Any missing files or assumptions:
  No modular source file was found for this family.
  The primary target is GPTNeoXJapaneseForCausalLM text generation.
  HF config access for abeja/gpt-neox-japanese-2.7b-ppo returned 401.
  Several Japanese GPT-NeoX-named checkpoints are model_type=gpt_neox and are out of scope for this report.
```

Primary source links:

- Transformers source at commit:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt_neox_japanese`
- In-scope production config:
  [abeja/gpt-neox-japanese-2.7b/config.json](https://huggingface.co/abeja/gpt-neox-japanese-2.7b/blob/main/config.json)
- Local notes:
  `agents/plans/transformers/gpt_neox_japanese/config_sweep.md`

Source anchors:

- Config defaults and RoPE legacy parameter conversion are in
  `configuration_gpt_neox_japanese.py:46-77`.
- RoPE implementation is in `modeling_gpt_neox_japanese.py:56-152`.
- Attention, QKV split, cache update, dense attention, and output projection
  are in `modeling_gpt_neox_japanese.py:155-289`.
- LayerNorm, MLP, residual, and last-layer attention bias are in
  `modeling_gpt_neox_japanese.py:292-380`.
- Model/prefill/decode position id and mask setup are in
  `modeling_gpt_neox_japanese.py:384-501`.
- Causal LM head, tied-weight declaration, and `logits_to_keep` slicing are in
  `modeling_gpt_neox_japanese.py:509-599`.
- Tokenizer text normalization, emoji mapping, and byte fallback are in
  `tokenization_gpt_neox_japanese.py:53-148` and `223-365`.

## 2. High-level architecture

Text-only decoder-only causal language model.

```text
CPU tokenizer -> input_ids/attention_mask
  -> token embedding
  -> N decoder blocks with pre-LN MHA + MLP
  -> final LayerNorm
  -> tied/un-tied LM projection
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: Japanese custom tokenizer loads `vocab.txt` and
  `emoji.json`, maps spaces/newlines/tabs to special strings, optionally cleans
  URL/email/tel/date/price/block characters, maps emoji to 12 groups, and falls
  back to UTF-8 byte tokens.
- GPU/runtime prefill: embedding lookup, causal mask construction, RoPE table
  generation for the current positions, all decoder layers, final LayerNorm,
  and LM head.
- GPU/runtime decode: one or more new tokens with a per-layer autoregressive KV
  cache; source uses Transformers `DynamicCache`.
- Independently cacheable state: per-layer K/V tensors after RoPE has been
  applied to keys. Tokenizer outputs and prompt input IDs are data-pipeline
  state, not KV cache.

Only `GPTNeoXJapaneseModel` and `GPTNeoXJapaneseForCausalLM` are implemented.
For the primary causal LM target, `GPTNeoXJapaneseModel` is required as the
base decoder. The feature-extraction output head is optional. Training loss,
hidden-state dumps, and attention dumps are deferred.

## 3. Important config dimensions

Effective production dimensions from
`abeja/gpt-neox-japanese-2.7b/config.json`:

| Field | Value | Source / note |
| --- | ---: | --- |
| `model_type` | `gpt_neox_japanese` | HF config |
| `architectures` | `GPTNeoXJapaneseForCausalLM` | HF config |
| `vocab_size` | 32000 | HF config |
| `hidden_size` | 2560 | HF config |
| `num_hidden_layers` | 32 | HF config |
| `num_attention_heads` | 32 | HF config |
| `head_dim` | 80 | inferred from source `hidden_size // num_attention_heads` |
| `intermediate_multiple_size` | 4 | HF config; MLP width = 10240 |
| `hidden_act` | `gelu` | HF config |
| `max_position_embeddings` | 2048 | HF config |
| `rotary_pct` / `partial_rotary_factor` | 1.0 | HF config -> normalized by config class |
| `rotary_emb_base` / `rope_theta` | 10000 | HF config -> normalized by config class |
| `layer_norm_eps` | 1e-5 | HF config |
| `attention_dropout` | default 0.1 if omitted | source default; ignored in eval |
| `hidden_dropout` | default 0.0 if omitted | source default |
| `use_cache` | true | HF config |
| `bos_token_id` | 31999 | HF config overrides source default 31996 |
| `eos_token_id` | 31999 | HF config |
| `tie_word_embeddings` | true by default | source default; source declares tied key |

Representative checkpoint/config sweep:

| Checkpoint | Scope | Hidden | Layers | Heads | Head dim | RoPE | MLP | Vocab | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| `hf-tiny-model-private/tiny-random-GPTNeoXJapaneseForCausalLM` | in-scope debug | 32 | 5 | 4 | 8 | pct 1.0, theta 10000 | x4 | 32000 | tiny random causal LM |
| `hf-tiny-model-private/tiny-random-GPTNeoXJapaneseModel` | in-scope base | 32 | 5 | 4 | 8 | pct 1.0, theta 10000 | x4 | 32000 | base model without LM head |
| `optimum-intel-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM` | in-scope debug | 32 | 5 | 4 | 8 | pct 1.0, theta 10000 | x4 | 32000 | same structure, older version metadata |
| `abeja/gpt-neox-japanese-2.7b` | in-scope production | 2560 | 32 | 32 | 80 | pct 1.0, theta 10000 | x4 | 32000 | production target |
| `stockmark/gpt-neox-japanese-1.4b` | out of scope | 2048 | 24 | 16 | 128 | pct 0.25, theta 10000 | generic `intermediate_size` | 50000 | `model_type: gpt_neox`, not this family |
| `rinna/japanese-gpt-neox-small` | out of scope | 768 | 12 | 12 | 64 | pct 1.0, theta 10000 | generic `intermediate_size` | 44416 | `model_type: gpt_neox`, T5 tokenizer |

## 3a. Family variation traps

- The source has no GQA/MQA: `num_key_value_heads` is absent and Q/K/V all use
  `num_attention_heads`.
- `head_dim` is not an explicit config field in this family; source computes
  `hidden_size // num_attention_heads`. Reject configs where hidden size is not
  divisible by head count.
- QKV projection is a single `Linear(hidden, 3 * hidden, bias=False)`, then
  per-head split order is `[q, k, v]` inside each head's contiguous `3 *
  head_size` block, not all-Q rows followed by all-K rows followed by all-V rows.
- All attention and MLP `Linear` modules are bias-free. The only learned
  non-norm bias is a separate `dense_bias` added after the last layer's
  attention output projection.
- MLP is ungated `Linear -> GELU -> Linear`; no SwiGLU/GEGLU.
- The in-scope production config omits `attention_dropout`,
  `hidden_dropout`, and `tie_word_embeddings`; source defaults apply.
- `bos_token_id` differs between source default and production config. Treat
  language/control IDs as tokenizer/generation ABI, not neural ops.
- Checkpoint names containing "Japanese GPT-NeoX" can route to generic
  `GPTNeoXForCausalLM` (`model_type: gpt_neox`) or GPT-2. Do not load them
  through this family without a separate audit.
- `rotary_pct` and `rotary_emb_base` are historical config keys normalized into
  `rope_parameters.partial_rotary_factor` and `rope_parameters.rope_theta`.
- The modeling code imports generic RoPE helpers, so non-default
  `rope_parameters.rope_type` could be accepted by current config/source even
  though the representative in-scope configs use default RoPE. Admit default
  RoPE first; route dynamic/scaled RoPE to a separate guard.
- Layout is sequence-major hidden-state `[B, S, H]`, attention `[B, heads, S,
  D]`, and no image/channel layout is present. NHWC/NCHW is not applicable.
  Layout rewrites should instead guard sequence/head permutation patterns.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token input `[B, S]`, optional attention mask `[B, S]`.
- Embedding lookup `Embedding(vocab_size -> hidden_size)`.
- `arange`, `unsqueeze`, broadcast, and add for position IDs with
  `past_seen_tokens`.
- Reshape/view `[B, S, 3H] -> [B, S, heads, 3D]`.
- Slice last dimension into Q/K/V per head: `[...,:D]`, `[...,D:2D]`,
  `[...,2D:]`.
- `permute(0,2,1,3)`, `contiguous`, `cat(dim=-1)`.
- Attention merge `permute(0,2,1,3).contiguous().view(B,S,H)`.
- `hidden_states[:, slice_indices, :]` for `logits_to_keep`.

Neural network primitives:

- LayerNorm over hidden dimension, epsilon 1e-5, affine parameters.
- Bias-free linear/GEMM:
  - token hidden QKV: `Linear(H -> 3H)`, production `2560 -> 7680`.
  - attention output: `Linear(H -> H)`, production `2560 -> 2560`.
  - MLP up: `Linear(H -> H * intermediate_multiple_size)`, production
    `2560 -> 10240`.
  - MLP down: `Linear(10240 -> 2560)`.
  - LM head: `Linear(H -> vocab_size, bias=False)`, production
    `2560 -> 32000`.
- GELU activation from Transformers `ACT2FN`.
- Residual add and optional bias-add for the last attention layer.
- Dropout exists in source but is identity in eval/inference.

Attention primitives:

- Dense causal self-attention only.
- MHA with Q/K/V head shape `[B, heads, S, D]`.
- QK scores via batched GEMM with scaling `1 / sqrt(D)`.
- Additive causal/padding mask broadcastable to `[B, heads, Q, K]`.
- Softmax over key dimension `dim=-1`.
- Attention-value matmul.
- KV cache update per layer through Transformers `Cache.update`.

Position/rotary ops:

- Default RoPE inverse-frequency table over `head_dim`, applied only to the
  leading `rotary_ndims = int(head_dim * partial_rotary_factor)` of Q and K.
- Production uses full-head RoPE because `rotary_pct=1.0`; generic source
  supports partial RoPE.
- Cos/sin computation is forced to fp32 before casting back to hidden dtype.

Generation/cache ops:

- Dynamic cache creation when `use_cache` and no past cache is supplied.
- `past_seen_tokens = past_key_values.get_seq_length()`.
- Per-layer cache entries store K/V after RoPE and before attention.
- Cache reorder for beam search is generic Transformers cache behavior, not
  implemented in this family file; defer for first single-sample greedy parity.

Preprocessing-coupled ops:

- Custom Japanese tokenizer and `emoji.json` are required for end-to-end parity.
- Tokenizer CPU work can be outside DinoML runtime at first.

Parameter aliasing:

- `GPTNeoXJapaneseForCausalLM` declares `embed_out.weight` tied to
  `gpt_neox_japanese.embed_in.weight`. Preserve one logical parameter when
  `tie_word_embeddings=True`.

## 5. Layer/block breakdown

Production symbols:

```text
B = batch
S = current sequence length
T = cached + current key length
H = hidden_size = 2560
heads = 32
D = head_dim = 80
I = intermediate_size = H * intermediate_multiple_size = 10240
```

Model:

```text
input_ids [B,S] -> embed_in -> hidden [B,S,H]
position_ids [1,S] or [B,S]
position_embeddings = rotary_emb(hidden, position_ids) -> cos/sin [B,S,rotary_ndims]
repeat decoder block 32 times
final_layer_norm(hidden) -> [B,S,H]
LM head on hidden[:, slice_indices, :] -> logits [B,S_keep,vocab]
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
ln = LayerNorm(x)
qkv = Linear(H -> 3H, bias=False)(ln)
qkv = view(qkv, [B,S,heads,3D])
q = qkv[..., :D].permute(0,2,1,3)
k = qkv[..., D:2D].permute(0,2,1,3)
v = qkv[..., 2D:].permute(0,2,1,3)
q_rot/q_pass = split(q, rotary_ndims)
k_rot/k_pass = split(k, rotary_ndims)
q_rot, k_rot = RoPE(q_rot, k_rot, cos, sin)
q = cat(q_rot, q_pass, dim=-1).contiguous()
k = cat(k_rot, k_pass, dim=-1).contiguous()
k, v = cache.update(k, v, layer_idx) when cache exists
attn = dense_causal_attention(q, k, v, mask)
attn = Linear(H -> H, bias=False)(merge_heads(attn))
if last layer: attn = attn + dense_bias[H] broadcast over [B,S,H]
x = residual + attn
mlp = Linear(I -> H, bias=False)(GELU(Linear(H -> I, bias=False)(LayerNorm(x))))
x = x + mlp
```

The last-layer attention bias is not inside `self.dense`; it is a separate
parameter expanded to `residual` shape before residual add.

## 6. Attention requirements

Required variant:

- Causal self-attention.
- Full dense attention, no local/sliding/block/random sparsity.
- MHA, not GQA/MQA.
- Query heads = key heads = value heads = `num_attention_heads`.
- Query/key/value width = `head_dim = hidden_size // num_attention_heads`.
- Query length is current `S`; key/value length is `T = past + S` when cache is
  active.
- Masking is produced by `create_causal_mask`, combining causal and
  caller-provided `attention_mask`.
- Source score math:
  `scores = baddbmm(zeros, q, k.transpose, beta=1, alpha=1/sqrt(D))`, reshape
  to `[B, heads, Q, K]`, add mask, softmax over `K`, dropout, cast to value
  dtype, then matmul with V.
- Cached keys are stored after RoPE. Values are stored directly from QKV split.
- Cache shape per layer before any backend packing:
  `key: [B, heads, T, D]`, `value: [B, heads, T, D]`.
- Source does not dispatch to SDPA/FlashAttention in this family file. DinoML
  can replace the baddbmm/softmax/matmul chain with a fused causal attention
  kernel under parity guards.

Unsupported or deferred:

- Cross-attention is not implemented even though tests set generic
  `add_cross_attention` in a smoke path.
- GQA/MQA/repeat-kv is absent.
- Packed/varlen sequence metadata is absent.
- `output_attentions=True` requires materializing dense attention weights;
  defer for optimized decode.

## 7. Position encoding and custom math

Default RoPE setup:

```python
base = config.rope_parameters["rope_theta"]
dim = config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Apply to the leading rotary dimensions:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_rope(q_rot, k_rot, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,S,R]
    sin = sin.unsqueeze(1)
    q = q_rot * cos + rotate_half(q_rot) * sin
    k = k_rot * cos + rotate_half(k_rot) * sin
    return q, k
```

Precompute opportunities:

- For default fixed theta and max position, inv_freq can be loaded or computed
  once.
- Cos/sin for prompt positions up to a bucketed max can be precomputed per
  dtype/device, but dynamic/decode position offsets require slicing by
  `past_seen_tokens`.
- Current source supports non-default rope types through generic
  `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; first integration should
  accept only `rope_type=default` unless a checkpoint requires another variant.

## 8. Preprocessing and input packing

Runtime neural inputs:

- `input_ids`: integer tensor `[B, S]`.
- `attention_mask`: optional mask `[B, S]` from tokenizer/generation pipeline.
- `position_ids`: optional integer tensor. If omitted, source creates
  `[0..S-1] + past_seen_tokens` and unsqueezes to `[1, S]`.
- `inputs_embeds`: alternate pre-embedded input. For first DinoML integration,
  prefer `input_ids` and reject `inputs_embeds` unless needed.

Tokenizer contract:

- Python tokenizer only; no fast tokenizer was inspected.
- Requires `vocab.txt` and `emoji.json`.
- Replaces ASCII/ideographic spaces, newlines, carriage returns, tabs, em dash,
  minus sign, and emoji before subword matching.
- Optional `do_clean_text` maps URL/email/tel/date/price/block-like glyphs to
  special tokens.
- Longest-match-ish search considers up to 3 characters normally, or up to
  `maxlen + 1` after `<`; when multiple candidates exist, the smallest token id
  is selected.
- Unknown single characters fall back to `<KIGOU>`, `<U2000U2BFF>`, or UTF-8
  byte tokens.

CPU/data-pipeline ownership is recommended for the tokenizer. The GPU graph
should begin at `input_ids`/`attention_mask`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed per-head QKV projection

Source pattern:

```text
qkv = Linear(H -> 3H)(x)
qkv.view(B,S,heads,3D)
q = qkv[..., :D]
k = qkv[..., D:2D]
v = qkv[..., 2D:]
```

Replacement:

```text
single GEMM -> fused per-head split views -> RoPE/attention
```

Preconditions:

- `hidden_size == num_attention_heads * head_dim`.
- Projection weight is loaded in PyTorch linear orientation
  `[3H, H]`; split is per-head `[q,k,v]` inside the output dimension after
  viewing to `[heads, 3D]`.
- No QKV bias.

Failure cases:

- Generic GPT-NeoX checkpoints may use different source classes or parallel
  residual behavior; route separately.

Parity test:

- Compare Q/K/V tensors after split and permute against Transformers for tiny
  and 2.7B-shaped random weights.

### Rewrite: dense attention chain -> fused causal attention

Source pattern:

```text
view B*heads -> baddbmm(q, kT, alpha=1/sqrt(D)) -> add mask
-> softmax(dim=-1) -> matmul(weights, v)
```

Replacement:

```text
fused MHA prefill/decode attention with additive mask and RoPE-applied cached K
```

Preconditions:

- Dense causal self-attention.
- No `output_attentions`.
- Dropout disabled or eval mode.
- Q/K/V shape `[B, heads, Q/K, D]`.
- Mask semantics match Transformers additive mask values.

Failure cases:

- Training, attention weight output, non-default mask rank unsupported by
  source tests, or non-default rope requiring dynamic scaling not implemented.

Parity test:

- Compare attention output before output projection over random masks,
  prefill, and decode with cache.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = embed_out(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden to requested positions before GEMM
```

Preconditions:

- `logits_to_keep` is an int or static tensor of positions.
- Loss is not computed.

Failure cases:

- Full-sequence logits requested, labels/loss path, or dynamic arbitrary
  position tensor not represented.

Parity test:

- Verify `logits_to_keep=1`, `0`, and a small static index tensor.

### Rewrite: eval bias/dropout/residual canonicalization

Source pattern:

```text
dropout(x + optional_bias, training=False) + residual
```

Replacement:

```text
residual_add(optional_bias_add(x))
```

Preconditions:

- Inference/eval mode.
- Dropout probability ignored because training is false.

Failure cases:

- Training or stochastic parity tests.

### Layout guidance: sequence/head layout only

There are no NHWC/NCHW tensors. Guarded layout work should target:

- eliminating redundant `permute(...).contiguous()` around attention when the
  fused attention provider accepts `[B,S,heads,D]` or `[B,heads,S,D]`;
- preserving source-visible logits/output layout `[B,S,V]`;
- keeping `dim=-1` softmax and last-dim RoPE axis intact.

No-layout-translation guard:

- Protect tokenizer/input IDs, position IDs, attention masks, and output logits
  from any channel-last style axis rewrite. They are sequence/token tensors.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,S,H]`: appears twice per block plus final norm.
- Bias-free GEMM coverage for QKV, attention output, MLP up/down, and LM head.
- Fused RoPE + attention prefill/decode: removes Q/K concat temporaries and the
  baddbmm/softmax/matmul chain.
- KV cache update/read with static per-layer slots: required for decode.
- Last-token-only LM head: important for decode throughput and vocab GEMM cost.

Medium priority:

- QKV GEMM + split + RoPE staging fusion, with correct packed per-head split.
- GELU MLP fusion: `Linear -> GELU -> Linear` can use optimized activation
  kernels and eventually fused epilogues.
- Last-layer attention bias + residual fusion.
- Attention mask construction and additive mask fusion into attention provider.

Lower priority:

- Tokenizer acceleration: useful for end-to-end serving but not a GPU runtime
  blocker.
- `output_hidden_states` / `output_attentions` materialization.
- Non-default RoPE variants through generic `ROPE_INIT_FUNCTIONS`.

## 11. Runtime staging plan

Stage 1: config/weights loader.

- Parse `GPTNeoXJapaneseConfig`, normalize `rotary_pct` and
  `rotary_emb_base`, reject generic `gpt_neox` configs.
- Load/tie `embed_in.weight` and `embed_out.weight` correctly.
- Validate hidden divisibility and QKV packed layout.

Stage 2: one-block parity without cache.

- Implement embedding, LayerNorm, packed QKV split, default RoPE, dense
  attention, MLP, residuals, and last-layer bias path.

Stage 3: full prefill parity.

- Run all layers, final norm, full logits, and `logits_to_keep`.
- Use dense attention first; fused attention can come after parity.

Stage 4: decode with KV cache.

- Allocate per-layer K/V cache `[B, heads, max_seq, D]`.
- Store keys after RoPE.
- Validate multi-token append against no-past recomputation.

Stage 5: optimized attention.

- Add fused causal prefill/decode attention with mask and output-attention
  rejection guard.

Stage 6: serving features.

- Add last-token LM head fast path, batching probes, and optional tokenizer
  pipeline integration.

Stub initially:

- Training/loss.
- Attention and hidden-state output tuples.
- Beam cache reorder.
- Non-default RoPE.
- Tokenizer inside GPU runtime.

## 12. Parity and validation plan

Concrete tests:

- Config parsing tests for production and tiny configs, including omitted
  defaults.
- QKV split parity with random tensors: compare q/k/v after `view`/slice/permute.
- RoPE parity for full-head and a synthetic partial-RoPE config.
- Single attention parity with and without padding mask, fp32 first.
- One decoder block parity, then after 2 layers, then full tiny model.
- Production-shape smoke with random weights for shape/memory only:
  `[B,S,H]=[1,16,2560]`, heads 32, D 80.
- Prefill logits parity on tiny random model with `logits_to_keep=0` and `1`.
- Decode parity: run prompt, cache, append 1 and 3 tokens; compare to no-past
  recomputation as the Transformers test does.
- Tied-weight test: modifying embedding weight changes LM head when config ties.
- Tokenizer golden tests for spaces, newline, emoji, heterograph vocab aliases,
  and byte fallback if end-to-end text parity is owned.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` for layer outputs; tighter for isolated ops.
- fp16/bf16: `rtol=1e-2`, `atol=1e-2` for full-layer outputs; attention may
  need separate score/output tolerances depending on backend accumulation.

## 13. Performance probes

- Tokenizer throughput for Japanese text length distribution.
- Prefill-only throughput by `[B,S]`: `S = 32, 128, 512, 2048`.
- Decode-only tokens/sec by batch and cache length.
- QKV/MLP/LM-head GEMM profile separately, especially LM head `[B,1,H] x
  [H,V]`.
- Attention backend comparison: dense baddbmm chain vs fused causal attention.
- KV cache memory and bandwidth by batch and max sequence.
- `logits_to_keep=1` vs full logits.
- Masked vs unmasked prefill.
- Tiny/debug model end-to-end latency for CI.
- Production-shape random-weight memory footprint and compile/profile time.
- Optional GGUF/quantized load and dequant provider probe only if a future
  checkpoint packaging path requires it; no source-coupled quantized format is
  present in this family source.

## 14. Skip/defer list

- Training, labels, and loss.
- Dropout stochastic behavior.
- `output_attentions=True` and dense attention weight returns.
- `output_hidden_states=True` unless needed for debugging parity.
- Beam search cache reorder and generation-controller policies beyond simple
  greedy/sampling.
- Non-default/scaled/dynamic RoPE variants.
- Generic `gpt_neox` Japanese checkpoints such as rinna/stockmark variants.
- Tokenizer implementation inside GPU/runtime.
- Multi-GPU tensor parallel.
- Quantization and packed weight formats; none are source-coupled here.
- NHWC/NCHW layout translation; not applicable to text-only tensors.

## 15. Final implementation checklist

- [ ] Parse `GPTNeoXJapaneseConfig` and normalize legacy RoPE keys.
- [ ] Reject non-`gpt_neox_japanese` configs in this integration path.
- [ ] Load embedding, packed QKV, output projection, MLP, LayerNorm, final norm,
      last-layer attention bias, and LM head weights.
- [ ] Preserve tied `embed_out.weight` / `embed_in.weight` aliasing when enabled.
- [ ] Implement embedding lookup for `[B,S]` input IDs.
- [ ] Implement LayerNorm over hidden dim.
- [ ] Implement packed per-head QKV split with source weight layout.
- [ ] Implement default RoPE and partial-RoPE guard.
- [ ] Implement dense causal MHA prefill reference path.
- [ ] Implement additive causal/padding mask compatibility.
- [ ] Implement MLP `Linear -> GELU -> Linear`.
- [ ] Implement last-layer attention bias add.
- [ ] Implement final norm and LM head with `logits_to_keep`.
- [ ] Implement per-layer KV cache storing RoPE-applied K and raw V.
- [ ] Add prefill logits parity tests.
- [ ] Add decode-with-cache parity tests.
- [ ] Add tokenizer golden tests if end-to-end text is in scope.
- [ ] Add fused attention rewrite behind parity guards.
- [ ] Add last-token-only LM head benchmark.
