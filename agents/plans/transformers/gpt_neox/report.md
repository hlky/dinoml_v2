# GPT-NeoX Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: gpt_neox family; primary checkpoint examples EleutherAI/gpt-neox-20b and EleutherAI/pythia-*
Config source: local configuration_gpt_neox.py plus official/open Hugging Face config.json files
Source files inspected:
- X:/H/transformers/src/transformers/models/gpt_neox/configuration_gpt_neox.py
- X:/H/transformers/src/transformers/models/gpt_neox/modeling_gpt_neox.py
- X:/H/transformers/src/transformers/models/gpt_neox/modular_gpt_neox.py
- X:/H/transformers/src/transformers/models/gpt_neox/tokenization_gpt_neox.py
- X:/H/transformers/src/transformers/cache_utils.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/integrations/sdpa_attention.py
- X:/H/transformers/src/transformers/integrations/flash_attention.py
- X:/H/transformers/src/transformers/modeling_rope_utils.py
- X:/H/transformers/src/transformers/activations.py
Any missing files or assumptions: no remote-code model files are required for the inspected family. modeling_gpt_neox.py is generated from modular_gpt_neox.py; future source edits should target the modular file, while the generated file is the exact expanded runtime source audited here.
```

Representative config sources:

- [EleutherAI/gpt-neox-20b config](https://huggingface.co/EleutherAI/gpt-neox-20b/raw/main/config.json)
- [EleutherAI/pythia-70m config](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json)
- [EleutherAI/pythia-160m config](https://huggingface.co/EleutherAI/pythia-160m/blob/main/config.json)
- [EleutherAI/pythia-1b config](https://huggingface.co/EleutherAI/pythia-1b/raw/main/config.json)
- [EleutherAI/pythia-12b config](https://huggingface.co/EleutherAI/pythia-12b/blob/main/config.json)
- [EleutherAI/gpt-neox-20b tokenizer_config](https://huggingface.co/EleutherAI/gpt-neox-20b/blob/main/tokenizer_config.json)

Primary runtime target for this report: `GPTNeoXForCausalLM` decoder-only generation, including prefill and autoregressive decode with KV cache. Other heads in source are optional/deferred: base `GPTNeoXModel` is useful for block parity, while sequence classification, token classification, and question answering are not required for first generation integration.

## 2. High-level architecture

GPT-NeoX is a text-only decoder stack:

```text
byte-level BPE tokenization -> token embedding -> N decoder blocks -> final LayerNorm -> LM projection -> logits/sampling
```

Each block has LayerNorm-preconditioned self-attention, RoPE on Q/K, a fused packed QKV projection, and an ungated two-linear MLP. The default block formulation uses parallel residuals:

```text
x -> LN_attn -> attention -> attn_out
x -> LN_mlp  -> MLP       -> mlp_out
output = x + attn_out + mlp_out
```

If `use_parallel_residual=false`, the block becomes sequential:

```text
attn_resid = x + attention(LN_attn(x))
output = attn_resid + MLP(LN_mlp(attn_resid))
```

Stage decomposition:

- CPU/data pipeline: byte-level BPE tokenization, attention-mask construction, generation loop controls, sampling.
- GPU prefill: embedding lookup, full causal attention over prompt, cache population, full or last-token logits.
- GPU decode: one or more new tokens, position IDs offset by cache length, cached K/V append, attention over prior cache plus new token, usually last-token logits only.
- Independently validatable units: tokenizer output IDs, RoPE cos/sin generation, one decoder block with and without cache, full prefill logits, one-step decode logits.

## 3. Important config dimensions

Current source defaults from `GPTNeoXConfig`:

| Field | Default / meaning |
| --- | --- |
| `vocab_size` | 50432 |
| `hidden_size` | 6144 |
| `num_hidden_layers` | 44 |
| `num_attention_heads` | 64 |
| `head_dim` | inferred as `hidden_size // num_attention_heads` |
| `intermediate_size` | 24576 |
| `hidden_act` | `gelu` by default; GPT-NeoX-20B uses `gelu_fast` |
| `max_position_embeddings` | 2048 |
| RoPE | standardized to `rope_parameters`; legacy `rotary_emb_base` -> `rope_theta`, `rotary_pct` -> `partial_rotary_factor` |
| default RoPE theta | `10000` when legacy config omits a standardized value |
| default partial rotary | `0.25` for GPT-NeoX conversion path |
| `attention_bias` | `true`; applies to QKV and attention output projections |
| MLP bias | always present in the two `nn.Linear` MLP layers |
| normalization | `nn.LayerNorm(hidden_size, eps=layer_norm_eps)` |
| `use_parallel_residual` | `true` |
| `tie_word_embeddings` | `false` |
| `use_cache` | `true` |
| attention backends | eager fallback, SDPA, FlashAttention 2/3/4, flex attention, and paged variants through Transformers interfaces |

Representative checkpoint sweep:

| Model | Hidden | Layers | Heads | Head dim | MLP | Vocab | Act | RoPE | Parallel residual | Dtype metadata |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `EleutherAI/pythia-70m` | 512 | 6 | 8 | 64 | 2048 | 50304 | `gelu` | theta 10000, rotary pct 0.25 | true | `float16` |
| `EleutherAI/pythia-160m` | 768 | 12 | 12 | 64 | 3072 | 50304 | `gelu` | theta 10000, rotary pct 0.25 | true | `float16` |
| `EleutherAI/pythia-1b` | 2048 | 16 | 8 | 256 | 8192 | 50304 | `gelu` | theta 10000, rotary pct 0.25 | true | `float16` |
| `EleutherAI/pythia-12b` | 5120 | 36 | 40 | 128 | 20480 | 50688 | `gelu` | theta 10000, rotary pct 0.25 | true | `float16` |
| `EleutherAI/gpt-neox-20b` | 6144 | 44 | 64 | 96 | 24576 | 50432 | `gelu_fast` | theta 10000, rotary pct 0.25 | omitted, source default true | `float16` |

Fields omitted in older configs but supplied by current source defaults include `attention_bias=true`, `attention_dropout=0.0`, `hidden_dropout=0.0`, `classifier_dropout=0.1`, and standardized `rope_parameters`. Some older configs store `attention_probs_dropout_prob` and `hidden_dropout_prob`; DinoML config loading should normalize or intentionally ignore these with a documented compatibility choice.

## 3a. Family variation traps

- QKV projection storage is packed per head, not all-Q/all-K/all-V row blocks. The linear output `[B,S,3H]` is viewed as `[B,S,num_heads,3*head_dim]`, transposed to `[B,num_heads,S,3*head_dim]`, then chunked into Q/K/V along the last dimension.
- No GQA/MQA in this source: `num_key_value_heads` is absent, and K/V cache heads equal attention heads.
- `hidden_size` must be divisible by `num_attention_heads`; `head_dim` is inferred unless a future config grows a custom field.
- RoPE is partial by default for official GPT-NeoX/Pythia configs: only the first `int(head_dim * partial_rotary_factor)` channels of Q and K are rotated. Remaining channels pass through unchanged.
- `partial_rotary_factor=0.25` yields rotary dimensions such as 16 for Pythia-70M, 32 for Pythia-12B, and 24 for GPT-NeoX-20B. Do not assume full-head RoPE.
- The RoPE implementation uses GPT-NeoX half-rotation (`[-x2, x1]`) over the rotary slice, not interleaved even/odd rotation.
- `use_parallel_residual` changes residual topology and fusion legality. It is true in official inspected configs, but the source supports false.
- Attention projection bias is configurable through `attention_bias`; MLP and LM head behavior are not controlled by that flag.
- MLP is ungated: `Linear -> activation -> Linear`, not SwiGLU/GEGLU.
- GPT-NeoX-20B uses `gelu_fast`; Pythia configs use standard PyTorch GELU.
- Embedding and LM-head weights are logically tieable by Transformers metadata, but official inspected configs set `tie_word_embeddings=false`; loaders should still preserve aliasing if a custom checkpoint ties them.
- `is_decoder` defaults false in config, but the model uses causal masking through `config.is_causal` default behavior and is used for causal LM. Do not gate generation support only on `is_decoder`.
- No layout translation is needed for text tensors. Protect token/sequence axes from layout rewrites: `[batch, seq, hidden]`, attention heads `[batch, heads, seq, head_dim]`, masks keyed by sequence axes.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer token input `[B,S]`
- embedding lookup `vocab_size -> hidden_size`
- `arange`, add scalar cache offset, `unsqueeze` for position IDs `[1,S]`
- reshape/view `[B,S,3H] -> [B,S,num_heads,3*head_dim]`
- transpose `[B,S,heads,3D] -> [B,heads,S,3D]`
- chunk/split QKV along last dim
- narrow/slice rotary and pass-through channels
- concatenate rotated and pass-through Q/K channels
- transpose attention output `[B,heads,S,D] -> [B,S,heads,D]`
- reshape `[B,S,heads,D] -> [B,S,H]`
- optional final hidden slice for `logits_to_keep`

Neural network primitives:

- LayerNorm over hidden axis, eps from config
- Linear QKV `H -> 3H`, bias controlled by `attention_bias`
- Linear attention output `H -> H`, bias controlled by `attention_bias`
- MLP `Linear(H -> intermediate_size)` with bias
- Activation `gelu` or `gelu_fast`
- MLP `Linear(intermediate_size -> H)` with bias
- LM head `Linear(H -> vocab_size)`, bias false
- residual adds: either 3-input add for parallel residual or two sequential residual adds
- dropout nodes are present but zero for inference and can be erased when not training

Attention primitives:

- causal self-attention, MHA
- Q/K/V shape `[B, num_heads, q_or_kv_len, head_dim]`
- eager attention math: `matmul(Q, K^T) * head_dim^-0.5`, add mask, `softmax(..., dtype=float32)`, cast to query dtype, dropout, matmul with V
- SDPA and FlashAttention-compatible fused attention paths
- optional 2D attention mask `[B, past+S]` from tokenizer/generation pipeline, converted by Transformers mask utilities

Position/rotary ops:

- inverse-frequency generation for rotary slice
- position-dependent `cos`/`sin`, computed in float32 under autocast disabled, cast to hidden dtype
- partial rotary apply to first rotary slice of Q/K
- support for standardized RoPE variants through `ROPE_INIT_FUNCTIONS` if a custom config uses non-default `rope_type`

Generation/cache ops:

- `DynamicCache(config=config)` creation when `use_cache` and no cache is supplied
- cache sequence length query for position offset
- per-layer cache append/update after RoPE, before attention backend
- cache tensors shaped `[B, num_heads, cached_seq, head_dim]`
- logits slicing via `logits_to_keep`; first integration can lower only last-token logits for decode

Preprocessing-coupled ops:

- GPT-NeoX tokenizer is byte-level BPE using `vocab.json`, `merges.txt`, or `tokenizer.json`
- tokenizer model inputs are `input_ids` and `attention_mask`
- default tokenizer special tokens from source are `<|endoftext|>` for unk/bos/eos and `<|padding|>` for pad; GPT-NeoX-20B tokenizer config sets unk/bos/eos to `<|endoftext|>`, `add_prefix_space=false`, `tokenizer_class=GPTNeoXTokenizer`

Distributed/tensor-parallel metadata:

- Source config declares TP hints: QKV and MLP input projections are columnwise; attention output and MLP output projections are rowwise; LM head is colwise-gather-output. This is optional for first single-GPU DinoML lowering.

## 5. Layer/block breakdown

Base model:

```text
input_ids [B,S] or inputs_embeds [B,S,H]
inputs_embeds = Embedding(input_ids)                 # [B,S,H]
position_ids = arange(S) + past_seen_tokens          # [1,S] unless caller supplies [B,S]
causal_mask = create_causal_mask(...)
hidden = emb_dropout(inputs_embeds)                  # inference no-op if dropout 0
cos, sin = rotary_emb(hidden, position_ids)          # [B or 1,S,rotary_dim]
for layer in layers:
  hidden = layer(hidden, causal_mask, cache, cos/sin)
hidden = final_layer_norm(hidden)
```

Decoder block, repeated `num_hidden_layers` times:

```text
attn_in = LayerNorm(hidden)
qkv = Linear(attn_in, H -> 3H, bias=attention_bias)
qkv = view(qkv, [B,S,num_heads,3*head_dim]).transpose(1,2)
q, k, v = chunk(qkv, 3, dim=-1)                      # each [B,heads,S,D]
q, k = partial_rope(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx)                 # if cache enabled
attn = attention(q, k, v, mask, scale=D^-0.5)
attn = Linear(reshape(attn), H -> H, bias=attention_bias)
```

Parallel residual path:

```text
mlp = Linear(LayerNorm(hidden), H -> I)
mlp = activation(mlp)
mlp = Linear(mlp, I -> H)
hidden = hidden + attn + mlp
```

Sequential residual path:

```text
attn_resid = hidden + attn
mlp = Linear(LayerNorm(attn_resid), H -> I)
mlp = activation(mlp)
mlp = Linear(mlp, I -> H)
hidden = attn_resid + mlp
```

LM head:

```text
logits = Linear(hidden[:, slice_indices, :], H -> vocab_size, bias=False)
```

For decode, `slice_indices` should normally keep only the newest token positions.

## 6. Attention requirements

GPT-NeoX attention is causal self-attention with standard MHA:

- Query heads: `num_attention_heads`
- KV heads: same as query heads
- Head dim: `hidden_size // num_attention_heads`
- Query/key/value cache after RoPE, shape per layer `[B, num_heads, cached_seq, head_dim]`
- No cross-attention in the inspected GPT-NeoX class despite some generic test configuration fields.
- No sliding-window/local attention in the GPT-NeoX config class. Generic cache utilities can support sliding caches for other models, but not required by this family.
- No ALiBi or relative attention bias.
- Masking is causal plus optional padding mask. For SDPA, the mask utility may return `None` when it can rely on the backend `is_causal` path; eager receives an additive mask with zero and negative infinity semantics.
- Eager path upcasts softmax computation to float32 and casts probabilities back to query dtype before the value matmul. Fused attention parity should preserve this numerically or set tolerances accordingly.
- Source backends: `_supports_flash_attn=True`, `_supports_sdpa=True`, `_supports_flex_attn=True`, `_supports_attention_backend=True`. Default attention selection in Transformers prefers SDPA when available, otherwise eager; explicit FlashAttention variants are also accepted.
- `output_attentions=True` forces users away from SDPA/FlashAttention in Transformers. DinoML first integration can omit attention weights and prioritize logits/cache parity.

Cache details:

```text
prefill input q/k/v: [B, heads, S_prompt, D]
prefill stored k/v: [B, heads, S_prompt, D]
decode input q/k/v: [B, heads, S_new, D]
decode stored k/v after update: [B, heads, S_prompt + S_new, D]
attention key/value consumed by backend: updated full cache [B, heads, total_kv, D]
```

Because there is no GQA, no repeat-KV expansion is needed for this family. Cached keys are stored after partial RoPE has already been applied.

## 7. Position encoding and custom math

Default GPT-NeoX RoPE:

```python
def gpt_neox_inv_freq(head_dim, partial_rotary_factor=0.25, theta=10000):
    rotary_dim = int(head_dim * partial_rotary_factor)
    return 1.0 / (theta ** (arange(0, rotary_dim, 2).float() / rotary_dim))
```

Runtime cos/sin:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
freqs = freqs.transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Partial RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def apply_gpt_neox_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)       # broadcast over heads
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k = cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q, k
```

Precompute opportunities:

- Static inverse frequencies are config-dependent and can be constant.
- For bounded maximum sequence length, cos/sin tables can be precomputed for default RoPE and sliced by `position_ids`.
- Dynamic, linear, yarn, longrope, llama3, or proportional RoPE variants should be gated behind config detection; the generic Transformers decorator can recompute frequencies when dynamic RoPE grows beyond cached length.

## 8. Preprocessing and input packing

Text preprocessing is byte-level BPE. Tokenizer output for runtime is:

```text
input_ids: int64-like token ids [B,S]
attention_mask: optional [B,S_total] with 1 for visible tokens and 0 for padding
```

There is no multimodal packing, no scatter of image/audio embeddings, no token type IDs used by the model body, and no packed sequence descriptor required for basic generation. Position IDs may be supplied by the caller, but Transformers creates `[1,S]` by default using `past_seen_tokens`.

Generation-controller behavior outside the core graph:

- `GenerationMixin` owns sampling, greedy/beam search, stopping criteria, and logits processors.
- First DinoML integration can validate core module parity with greedy one-token decode and leave beam search, sampling, repetition penalties, and server-style continuous batching to later layers.
- `logits_to_keep` is a source-level optimization knob; use it to avoid full-sequence LM-head work during decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed GPT-NeoX QKV projection

Source pattern:

```text
Linear(H -> 3H) -> view([B,S,heads,3D]) -> transpose(1,2) -> chunk(3, dim=-1)
```

Replacement pattern:

```text
single GEMM_RRR/GEMM_RCR -> fused or metadata view -> head-major Q/K/V accessors
```

Preconditions:

- `hidden_size == num_heads * head_dim`
- projection output features exactly `3 * hidden_size`
- packed row order is per-head `[q_head, k_head, v_head]`
- no consumer observes the raw packed tensor except view/transpose/chunk

Weight transform:

```python
w = qkv.weight.reshape(num_heads, 3, head_dim, hidden_size)
b = qkv.bias.reshape(num_heads, 3, head_dim) if bias is not None else None
```

Failure cases: all-Q/all-K/all-V assumptions, custom `head_dim`, nonstandard checkpoint remapping, requested output of intermediate packed QKV.

Parity test sketch: compare Q/K/V tensors after split against Transformers for random hidden states and official dimensions.

### Rewrite: partial-RoPE + cache update

Source pattern:

```text
slice rotary channels -> rotate_half -> mul/add -> concat pass-through -> cache append
```

Replacement pattern:

```text
fused partial_rope kernel for Q/K, writing rotated K directly into cache when decoding
```

Preconditions:

- default GPT-NeoX half-rotation layout
- `rotary_dim = int(head_dim * partial_rotary_factor)` and even
- cache stores post-RoPE K
- position IDs are monotonic or table-gather path is available

Failure cases: non-default RoPE scaling not implemented, interleaved RoPE assumption, externally supplied arbitrary position IDs without gather support.

Parity test sketch: random Q/K and position IDs for rotary pct 0.25, compare full output including pass-through tail.

### Rewrite: parallel residual block fusion

Source pattern:

```text
LN(x)->attention->proj plus LN(x)->MLP, then x + attn + mlp
```

Replacement pattern:

```text
schedule attention and MLP branches from same input; fuse final 3-input residual add
```

Preconditions:

- `use_parallel_residual=true`
- dropout disabled/inference
- no captured intermediate hidden states/attentions needed

Failure cases: `use_parallel_residual=false`, training/dropout, output-hidden-states debugging requiring exact observable intermediates.

Parity test sketch: one-block fp32 and fp16 comparisons against Transformers for both residual modes.

### Rewrite: decode last-token logits

Source pattern:

```text
hidden_states[:, slice_indices, :] -> embed_out
```

Replacement:

```text
gather last hidden token(s) -> GEMM(H -> vocab)
```

Preconditions:

- generation decode only needs newest token logits
- `logits_to_keep` is integer 1 or a validated tensor selecting decode positions

Failure cases: caller asks full prompt logits, training loss, arbitrary `logits_to_keep` tensor.

Parity test sketch: compare full logits last token vs sliced logits for prefill and decode.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over hidden size: twice per block plus final norm; required for parity and performance.
- Packed QKV GEMM with head-major output handling: removes avoidable copies and fixes GPT-NeoX-specific QKV layout.
- Partial RoPE kernel: small but on the critical path, especially decode; must respect pass-through tail.
- Causal MHA/FlashAttention prefill and decode with KV cache: dominant runtime cost.
- LM-head last-token GEMM: critical for decode and can avoid full `[B,S,V]` logits.

Medium priority:

- MLP `Linear -> GELU/gelu_fast -> Linear`: GEMM plus activation fusion or CUTLASS epilogue where available.
- Attention output projection plus residual add: useful epilogue candidate.
- Parallel residual final add of hidden, attention, and MLP branches.
- Cache append/write kernel that can combine K rotation and cache storage.

Lower priority:

- Token embedding lookup fusion with dropout erase.
- Full generation-controller kernels such as top-k/top-p sampling.
- Classification/QA heads.
- Tensor-parallel sharding plans.

## 11. Runtime staging plan

Stage 1: Parse config and load weights for `GPTNeoXForCausalLM`; normalize legacy RoPE keys and packed QKV metadata.

Stage 2: Run embedding, one block, and final LayerNorm parity in fp32 without cache. Stub generation and optimized attention with eager-compatible kernels.

Stage 3: Full prefill parity for small Pythia config, including partial RoPE, causal mask, packed QKV split, and full logits.

Stage 4: Decode with `DynamicCache`-equivalent K/V storage. Validate position offset, post-RoPE cache contents, and one-token logits.

Stage 5: Replace eager attention with DinoML fused attention/FlashAttention path for prefill and decode, preserving softmax scaling and mask behavior.

Stage 6: Add lowering rewrites/fusions for QKV, partial RoPE, MLP activation, residual epilogues, and last-token logits.

Stage 7: Add larger checkpoint loading, GGUF/quantized weight materialization if needed, and continuous batching/server scheduling.

Initial stubs: dropout as no-op in inference, attention weights outputs absent, non-generation heads absent, beam search/sampling external to compiled module.

## 12. Parity and validation plan

- Config normalization tests: legacy `rotary_pct`/`rotary_emb_base` and standardized `rope_parameters` produce the same rotary dim/theta.
- QKV split test: random hidden states, compare packed projection split tensors for Pythia-70M and GPT-NeoX-20B dimensions.
- RoPE unit test: compare partial rotated Q/K including pass-through channels for rotary pct 0.25 and custom full-rotary guard.
- LayerNorm and activation tests: `gelu` and `gelu_fast`, fp32/fp16 tolerances.
- One-block parity: fp32 tolerance around `rtol=1e-4, atol=1e-4`; fp16 around `rtol=1e-2, atol=1e-2` depending on attention backend.
- Prefill logits parity: Pythia-70M or a tiny GPTNeoX config with fixed prompt, compare last-token logits.
- Decode parity: prefill cache then feed 1-3 new tokens; compare decode logits to full recompute over concatenated sequence.
- Cache shape/content test: assert each layer stores K/V `[B,heads,total_seq,D]` after RoPE.
- Backend parity: eager vs SDPA/Fused attention for no-padding and padding-mask cases.
- End-to-end text generation smoke: greedy decode with `EleutherAI/pythia-70m` prompt, compare token IDs for a short fixed sequence.

## 13. Performance probes

- Prefill throughput by `(B,S)` for Pythia-70M, Pythia-1B-like, and GPT-NeoX-20B-like dimensions.
- Decode tokens/sec by batch size and cache length.
- KV cache memory usage: `2 * layers * B * heads * seq * head_dim * bytes`.
- QKV projection time before/after packed-layout rewrite.
- RoPE kernel time and cache-write fusion benefit.
- Attention backend comparison: eager, SDPA-like, FlashAttention-like, decode specialized attention.
- MLP GEMM/activation/GEMM time and activation fusion benefit.
- LM-head full-sequence vs last-token-only logits.
- Config sweep over head_dim 64, 96, 128, 256 because rotary dimensions and attention tile choices change.

## 14. Skip/defer list

- Training, labels, and loss computation.
- Dropout behavior other than inference no-op.
- Gradient checkpointing.
- Sequence classification, token classification, and QA heads.
- Attention weight materialization for `output_attentions=True`.
- Beam search, sampling processors, and logits warpers inside DinoML runtime.
- Tensor-parallel and pipeline-parallel plans.
- Paged attention/continuous batching until basic cache parity is stable.
- Non-default advanced RoPE variants unless an inspected target config requires them.
- Quantized/GGUF ingestion unless selected deployment checkpoints require it.

## 15. Final implementation checklist

- [ ] Parse `GPTNeoXConfig`, including legacy RoPE fields and old dropout field names.
- [ ] Load embedding, packed QKV, attention output, MLP, LayerNorm, and LM-head weights.
- [ ] Preserve QKV per-head `[q,k,v]` packed layout.
- [ ] Implement token embedding and position ID generation.
- [ ] Implement LayerNorm over hidden axis.
- [ ] Implement `gelu` and `gelu_fast`.
- [ ] Implement partial GPT-NeoX RoPE with half-rotation and pass-through tail.
- [ ] Implement causal MHA prefill.
- [ ] Implement KV cache append/update with post-RoPE K storage.
- [ ] Implement decode attention over cache.
- [ ] Implement parallel and sequential residual block variants.
- [ ] Implement LM-head last-token logits path.
- [ ] Add QKV split and RoPE unit parity tests.
- [ ] Add one-block, prefill, and decode parity tests against Transformers.
- [ ] Add attention backend/fallback parity tests with and without padding masks.
- [ ] Benchmark prefill, decode, cache memory, QKV/RoPE, MLP, and logits bottlenecks.
