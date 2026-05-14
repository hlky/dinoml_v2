# OPT Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: opt
Primary runtime target: OPTForCausalLM decoder-only generation, including prefill, decode, logits, and KV cache.
Dinoml assumptions: inference-only first, CUDA GPU target, preserve Transformers/PyTorch text axes, prioritize dense GEMM/LayerNorm/attention/cache lowering before optional heads.
```

Source files inspected:

- Local: `X:/H/transformers/src/transformers/models/opt/modeling_opt.py`
- Local: `X:/H/transformers/src/transformers/models/opt/configuration_opt.py`
- Local shared utilities: `X:/H/transformers/src/transformers/masking_utils.py`, `X:/H/transformers/src/transformers/cache_utils.py`
- Upstream source URL pattern at the pinned commit: `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/opt/...`

Representative official Hugging Face configs fetched from Hub raw files:

- `https://huggingface.co/facebook/opt-125m/raw/main/config.json`
- `https://huggingface.co/facebook/opt-350m/raw/main/config.json`
- `https://huggingface.co/facebook/opt-1.3b/raw/main/config.json`
- `https://huggingface.co/facebook/opt-6.7b/raw/main/config.json`
- `https://huggingface.co/facebook/opt-13b/raw/main/config.json`
- `https://huggingface.co/facebook/opt-30b/raw/main/config.json`
- `https://huggingface.co/facebook/opt-66b/raw/main/config.json`
- Matching `generation_config.json` and `tokenizer_config.json` for `opt-125m`, `opt-350m`, `opt-6.7b`, and `opt-30b`.

Missing files or assumptions:

- No local `tokenization_opt.py` exists in the inspected directory. Tokenizer facts below come from official Hub tokenizer metadata and OPT model-card behavior, not local tokenizer source.
- No remote code is required for the official `facebook/opt-*` checkpoints sampled here.
- Dtype and architecture dimensions are from `config.json`; behavior such as position offsets, normalization order, cache shapes, and logits slicing is from source.
- Optional heads in source: `OPTModel` is the required decoder body, `OPTForCausalLM` is required for this target, while sequence classification and question answering are deferred.

## 2. High-level architecture

OPT is a text-only decoder-only Transformer with learned token embeddings, learned absolute positional embeddings with an OPT-specific offset, causal self-attention, dense ReLU feed-forward blocks, LayerNorm, and a tied bias-free LM projection.

```text
byte-level BPE tokenizer -> input_ids + attention_mask
  -> token embedding (+ optional input projection)
  -> learned absolute position embedding with offset
  -> repeated causal decoder blocks
  -> optional final LayerNorm
  -> optional output projection
  -> tied LM head logits
  -> generation controller / sampling
```

Generation stage split:

```text
CPU tokenizer/padding -> GPU prefill(full prompt, empty cache)
                     -> per-layer KV cache
                     -> GPU decode(new token chunk, grown attention_mask, cache append)
                     -> logits_to_keep / lm_head -> sampler/controller
```

Independently stageable pieces:

- CPU/data pipeline: GPT-2 style byte-level BPE, `add_bos_token=true` in sampled tokenizer metadata, attention mask, left/right padding policy chosen by caller.
- GPU embedding/position stage: integer token lookup, position ID creation from the 2D mask, learned position lookup, optional `project_in`.
- GPU decoder block: LayerNorm order depends on config, then MHA, residual, FFN, residual.
- GPU generation cache: per-layer K/V append and causal/padding mask shape handling.
- Logits stage: optional last-token-only or selected-token LM projection through tied token embedding rows.

## 3. Important config dimensions

Source defaults from `OPTConfig`:

| Field | Default | Lowering effect |
| --- | ---: | --- |
| `vocab_size` | 50272 | Token embedding rows and tied LM head rows. |
| `hidden_size` | 768 | Decoder hidden width. |
| `word_embed_proj_dim` | `hidden_size` if omitted | Token embedding width and LM head input width; may differ from decoder hidden width. |
| `num_hidden_layers` | 12 | Decoder layer count. |
| `num_attention_heads` | 12 | Full MHA head count; no GQA/MQA field. |
| `head_dim` | `hidden_size / num_attention_heads` | Source requires exact divisibility. |
| `ffn_dim` | 3072 | FFN intermediate width. |
| `max_position_embeddings` | 2048 | Learned absolute positions before adding OPT offset. |
| `activation_function` | `relu` | FFN activation via `ACT2FN`. |
| `do_layer_norm_before` | `true` | Pre-LN for most checkpoints; `opt-350m` is post-LN. |
| `_remove_final_layer_norm` | `false` | Compatibility switch; removes decoder final LN only for old fine-tunes. |
| `enable_bias` | `true` | Attention and FFN linear bias switch. Official configs omit it, so source default applies. |
| `layer_norm_elementwise_affine` | `true` | LayerNorm gamma/beta switch. Official configs omit it, so source default applies. |
| `pad_token_id` / `bos_token_id` / `eos_token_id` | 1 / 2 / 2 | Padding and generation special-token defaults in current config class. |
| `use_cache` | `true` | Dynamic cache created when generation requests caching. |
| `tie_word_embeddings` | `true` | LM head weight is aliased to token embedding weight. |

Representative checkpoint sweep, from official `config.json`:

| Model id | Layers | Hidden | Embed dim | Heads | Head dim | FFN | Pre-LN | Max pos | dtype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `facebook/opt-125m` | 12 | 768 | 768 | 12 | 64 | 3072 | true | 2048 | float16 |
| `facebook/opt-350m` | 24 | 1024 | 512 | 16 | 64 | 4096 | false | 2048 | float16 |
| `facebook/opt-1.3b` | 24 | 2048 | 2048 | 32 | 64 | 8192 | true | 2048 | float16 |
| `facebook/opt-6.7b` | 32 | 4096 | 4096 | 32 | 128 | 16384 | true | 2048 | float16 |
| `facebook/opt-13b` | 40 | 5120 | 5120 | 40 | 128 | 20480 | true | 2048 | float16 |
| `facebook/opt-30b` | 48 | 7168 | 7168 | 56 | 128 | 28672 | true | 2048 | float16 |
| `facebook/opt-66b` | 64 | 9216 | 9216 | 72 | 128 | 36864 | true | 2048 | float16 |

Sampled generation metadata:

| Model ids sampled | `generation_config.json` token IDs |
| --- | --- |
| `opt-125m`, `opt-350m`, `opt-6.7b`, `opt-30b` | `_from_model_config=true`, `bos_token_id=2`, `eos_token_id=2`, `pad_token_id=1` |

Sampled tokenizer metadata:

| Field | Value in sampled `tokenizer_config.json` |
| --- | --- |
| `unk_token`, `bos_token`, `eos_token` | `</s>` |
| `pad_token` | `<pad>` |
| `add_bos_token` | `true` |
| `add_prefix_space` | `false` |
| `errors` | `replace` |

## 3a. Family variation traps

- `facebook/opt-350m` is the major structural exception: `do_layer_norm_before=false`, no decoder final LayerNorm, and `word_embed_proj_dim=512` while `hidden_size=1024`. It requires `project_in: 512 -> 1024`, `project_out: 1024 -> 512`, and an LM head from 512 to vocab.
- Most other official checkpoints are pre-LN and have `word_embed_proj_dim == hidden_size`.
- Learned absolute positions are not GPT-2-identical: `OPTLearnedPositionalEmbedding` allocates `max_position_embeddings + 2` rows and looks up `position_ids + 2`.
- Default position IDs are derived from `attention_mask.cumsum(dim=1) * attention_mask - 1`, then sliced by `past_seen_tokens`; padding locations become `-1` before adding the offset, so they map to row 1 if ever looked up.
- Causal mask creation is delegated to shared `create_causal_mask`, which may return a 4D additive mask, `None` for optimized SDPA/FlashAttention paths, or a backend-specific block mask.
- Cache is full MHA: cached key/value tensors are `[B, num_attention_heads, T, head_dim]`. There is no `num_key_value_heads`, no repeat-KV, no sliding window, no RoPE, no ALiBi, no MoE, and no gated MLP.
- Source scales queries before the attention backend call: `q_proj(hidden_states) * head_dim**-0.5`, then passes `scaling=1.0`. A fused attention path must preserve this math order.
- Linear biases are controlled by source default `enable_bias=true`; official sampled configs omit this field.
- `_remove_final_layer_norm` exists only for backward compatibility with old fine-tunes. Do not assume all pre-LN OPT checkpoints have a final decoder LayerNorm.
- `activation_dropout` and `output_projection` appear in some configs but are not read by the inspected current modeling source for inference.
- No layout translation is needed for text tensors. Guard `[B, T, H]`, LayerNorm last-dim, softmax last-dim, position `cumsum(dim=1)`, and cache `[B, heads, T, D]` against generic layout rewrites.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer `input_ids[int64]` shaped `[B, S]`, flattened with `.view(-1, S)` if supplied.
- Optional `attention_mask` shaped `[B, past + S]`; default all-ones mask is created when omitted.
- Optional externally supplied `inputs_embeds` shaped `[B, S, word_embed_proj_dim]`.
- Embedding lookup for `embed_tokens`: `[vocab_size, word_embed_proj_dim] -> [B, S, word_embed_proj_dim]`.
- `cumsum(attention_mask, dim=1)`, multiply by mask, subtract one, cast to integer, slice by cache length.
- Learned position embedding lookup: table `[max_position_embeddings + 2, hidden_size]`, index `position_ids + 2`.
- Add token and position embeddings, dropout as inference identity, residual add, reshape/view, transpose, contiguous.
- Optional `project_in` and `project_out` bias-free linear layers when embed dim differs from hidden size.
- Logits slicing by `logits_to_keep`: all positions when 0, last `k` positions for integer `k`, or tensor indices.

Neural network primitives:

- LayerNorm over hidden dim with affine optional by config; epsilon is PyTorch default unless inherited from global module defaults.
- Attention projections per layer: `q_proj`, `k_proj`, `v_proj`, `out_proj`, each `Linear(hidden_size -> hidden_size, bias=enable_bias)`.
- FFN per layer: `fc1 Linear(hidden_size -> ffn_dim, bias=enable_bias)`, ReLU, `fc2 Linear(ffn_dim -> hidden_size, bias=enable_bias)`.
- LM head: `Linear(word_embed_proj_dim -> vocab_size, bias=False)`, tied to token embedding weight.

Attention primitives:

- Dense MHA causal self-attention.
- Q/K/V reshape to `[B, heads, T, head_dim]`.
- Query pre-scale by `head_dim ** -0.5`.
- Attention mask addition before softmax for eager path.
- Softmax in fp32 then cast back to query dtype in eager path.
- Attention output matmul with V, transpose back to `[B, T, hidden_size]`, output projection.

Generation/cache ops:

- Dynamic cache append per layer.
- Per-layer cache tensors `[B, heads, T_total, head_dim]`.
- Cache sequence-length query for position slicing and mask sizes.
- Beam-search cache reordering via batch `index_select` is needed for beam generation parity, but can be deferred for greedy/sampling first.

Preprocessing-coupled ops:

- GPT-2 byte-level BPE tokenizer behavior is outside the GPU graph.
- Special tokens from sampled metadata: BOS/EOS/UNK all `</s>`, pad `<pad>`, generation uses BOS/EOS id 2 and pad id 1.

Optional/deferred heads:

- Sequence classification: per-token `Linear(word_embed_proj_dim -> num_labels, bias=False)` then rightmost non-pad pooling via mask/argmax.
- Question answering: per-token `Linear(word_embed_proj_dim -> 2)`, split/squeeze to start/end logits.

## 5. Layer/block breakdown

Embedding stage:

```text
input_ids: [B, S]
inputs_embeds = embed_tokens(input_ids)                  # [B, S, E]
attention_mask = ones([B, past + S]) if omitted
position_ids = cumsum(attention_mask, dim=1) * mask - 1
position_ids = position_ids[:, past_seen_tokens:]
pos = embed_positions(position_ids + 2)                  # [B, S, H]
if E != H: inputs_embeds = project_in(inputs_embeds)     # [B, S, H]
hidden = inputs_embeds + pos
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = hidden
if do_layer_norm_before:
    hidden = self_attn_layer_norm(hidden)

q = q_proj(hidden) * (head_dim ** -0.5)
k = k_proj(hidden)
v = v_proj(hidden)
q,k,v = view_transpose_to_heads(q,k,v)                   # [B, heads, T, D]
k,v = cache.update(k,v, layer_idx) if cache is present
attn = causal_attention(q, k, v, mask, scaling=1.0)
hidden = residual + out_proj(attn)

if not do_layer_norm_before:
    hidden = self_attn_layer_norm(hidden)

shape = hidden.shape
hidden2d = hidden.reshape(-1, H)
residual = hidden2d
if do_layer_norm_before:
    hidden2d = final_layer_norm(hidden2d)
hidden2d = fc2(relu(fc1(hidden2d)))
hidden = (residual + hidden2d).view(shape)

if not do_layer_norm_before:
    hidden = final_layer_norm(hidden)
```

Decoder output:

```text
if decoder.final_layer_norm exists:
    hidden = decoder.final_layer_norm(hidden)
if E != H:
    hidden = project_out(hidden)                         # [B, S, E]
logits = lm_head(hidden[:, slice_indices, :])            # [B, kept, vocab]
```

Bias rules:

- `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` use `bias=config.enable_bias`.
- `project_in`, `project_out`, and `lm_head` are bias-free.
- Official sampled configs rely on source default `enable_bias=true`.

## 6. Attention requirements

Required attention variant:

| Property | OPT requirement |
| --- | --- |
| Type | Decoder self-attention only. |
| Causality | Causal with optional padding mask. |
| Heads | Full MHA; Q heads = K heads = V heads. |
| KV head count | Same as `num_attention_heads`; no GQA/MQA. |
| Head dim | 64 for 125M/350M/1.3B, 128 for 6.7B and larger sampled official checkpoints. |
| Position interaction | Learned absolute positions are added before attention; cached K/V store already position-conditioned hidden projections. |
| Masking | Shared `create_causal_mask`; eager path adds additive mask to attention weights before softmax. |
| Cache | Dynamic cache appends new K/V along sequence dim `-2`. |
| Optimized backends | Source declares support for eager, SDPA, FlashAttention, and flex attention through shared attention/mask interfaces. |

Eager attention math:

```text
attn_weights = matmul(q_scaled, k.transpose(-1, -2)) + optional_mask
attn_probs = softmax(attn_weights, dim=-1, dtype=float32).to(q.dtype)
attn_out = matmul(attn_probs, v)
```

Cache shape before and after append for layer `i`:

```text
new_k, new_v: [B, heads, S_new, head_dim]
old cache:   [B, heads, T_past, head_dim]
updated:     [B, heads, T_past + S_new, head_dim]
```

Mask shape expectations:

- For eager fallback, additive masks must broadcast to attention weights `[B, heads, S_query, T_kv]`.
- For SDPA/FlashAttention, shared utilities may skip materializing the mask when there is no padding and causal semantics can be passed directly to the backend.
- For decoding with cache, `attention_mask` length should cover `past_seen_tokens + S_new`; if omitted, source creates an all-ones mask of that length.

## 7. Position encoding and custom math

OPT uses learned absolute positions with a hard-coded offset of 2. There is no RoPE, ALiBi, relative position bias, convolutional position encoding, or dynamic long-context scaling in the inspected source.

Essential position logic:

```python
def opt_position_ids(attention_mask, past_seen_tokens):
    position_ids = torch.cumsum(attention_mask, dim=1)
    position_ids = (position_ids * attention_mask - 1).long()
    return position_ids[:, past_seen_tokens:]

def opt_position_lookup(position_ids, embed_positions):
    return embed_positions(position_ids + 2)
```

What can be precomputed:

- The learned position embedding table is a normal constant with shape `[max_position_embeddings + 2, hidden_size]`.
- For fixed unpadded prefill buckets, position IDs are deterministic `[0..S-1]` then offset to rows `[2..S+1]`.

What depends on runtime inputs:

- Padding-aware positions depend on the 2D `attention_mask`.
- Decode positions depend on `past_seen_tokens` and the grown attention mask.
- External `position_ids`, if supplied, bypass the cumsum construction but still receive the `+2` lookup offset.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Tokenization is GPT-2 byte-level BPE style according to official OPT metadata/model cards.
- Sampled tokenizer metadata sets `add_bos_token=true`, `add_prefix_space=false`, `errors="replace"`.
- `</s>` is used for unk/bos/eos token content; `<pad>` is the pad token.

GPU/runtime graph inputs:

- `input_ids[int64]`: `[B, S]`.
- `attention_mask`: `[B, T_total]`, where `T_total = S` for prefill and `past + S` for decode.
- Optional `inputs_embeds`: `[B, S, word_embed_proj_dim]`; mutually exclusive with `input_ids`.
- Optional `position_ids`: `[B, S]`; if supplied, they should be already sliced to the new tokens in decode-like calls.
- Optional `past_key_values`: HF `Cache` object or Dinoml equivalent.

Generation-controller behavior outside the core graph:

- Sampling, beam search, stopping criteria, max-length control, and cache reordering are controller responsibilities.
- `generation_config.json` in sampled official repos only carries model-derived BOS/EOS/PAD ids.
- For first integration, greedy or sampling decode without beam cache reorder is enough; beam search adds batch index-select over every layer cache.

No multimodal/audio/vision packing, no packed sequence metadata, no placeholder scatter, and no processor-side tensor formats are required for OPT.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split linear projections to GEMM family

Source pattern:

```text
Linear([B,T,H] -> [B,T,H]) for q_proj/k_proj/v_proj/out_proj/fc1/fc2
```

Replacement:

```text
Flatten B*T -> GEMM_RCR/GEMM_RRR + bias epilogue -> reshape
```

Preconditions:

- Dense contiguous hidden states or known logical row-major layout.
- Static weight shape from config.
- Bias present only when `enable_bias=true`.

Shape equations:

- Attention projection: `M=B*T`, `K=H`, `N=H`.
- FFN up: `M=B*T`, `K=H`, `N=ffn_dim`.
- FFN down: `M=B*T`, `K=ffn_dim`, `N=H`.

Failure cases:

- Non-contiguous user-provided `inputs_embeds` without materialization support.
- Future checkpoints with `enable_bias=false` need no-bias GEMM variants.

Parity test sketch:

- Compare one projection and one full block against Transformers in fp32 and fp16 for random `[B,T,H]`.

### Rewrite: Q/K/V projection batching

Source pattern:

```text
q_proj(hidden), k_proj(hidden), v_proj(hidden)
```

Replacement:

```text
Either three independent GEMMs, or a packed/fused QKV GEMM after concatenating weights and biases.
```

Preconditions:

- Same input hidden tensor for all three projections.
- All three projections have matching `H -> H` shapes and same bias policy.
- Weight transform preserves split order `[q, k, v]`.

Weight transform:

```python
packed_w = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
packed_b = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)  # if bias
```

Failure cases:

- Weight aliasing or per-projection quantization metadata that must remain separate.
- Debug-first lowering can keep three GEMMs until a packed QKV provider exists.

### Rewrite: learned positions fast path for unpadded prefill/decode

Source pattern:

```text
cumsum(attention_mask) -> position_ids -> position embedding
```

Replacement:

```text
Range(start=past_seen_tokens, length=S) -> +2 -> position embedding
```

Preconditions:

- Attention mask is all ones.
- No external `position_ids`.
- No prefix/padding holes.

Failure cases:

- Left/right padded batch, partial masks, or caller-supplied positions.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Gather kept hidden positions before vocab GEMM.
```

Preconditions:

- `logits_to_keep` is integer 1 for decode or known small tail.
- Loss is not requested.

Shape equations:

- Full logits: `M=B*S`, `K=E`, `N=vocab`.
- Decode logits: `M=B`, `K=E`, `N=vocab`.

Failure cases:

- Training/loss path needs all shifted logits.
- Arbitrary tensor `logits_to_keep` requires gather/index-select support.

### Rewrite: cache-aware fused attention

Source pattern:

```text
q_scaled, k_cache_append, v_cache_append, causal/padding mask, softmax(fp32), matmul V
```

Replacement:

```text
Prefill FlashAttention/SDPA plus decode attention kernel with explicit K/V cache append.
```

Preconditions:

- Full MHA, no GQA expansion.
- Cache layout `[B, heads, T, D]` or provider-visible transform.
- Query is already scaled or kernel accepts scale=1.0 with pre-scaled Q.
- Mask backend can represent causal plus padding.

Failure cases:

- Additive mask semantics differ from backend boolean mask.
- Padding-aware decode with mixed sequence lengths needs exact causal/padding parity.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over hidden dim. OPT uses multiple LayerNorms per block, and pre/post-LN ordering must be exact.
- Dense GEMM with bias for attention and FFN projections. These dominate prefill and are already aligned with DinoML CUTLASS work.
- Causal attention with KV cache. Prefill attention and decode cache attention are required for generation throughput.
- Last-token-only LM head. Avoiding full-sequence vocab GEMM during decode is critical.

Medium priority:

- Packed QKV projection. Saves launches and memory traffic after correctness is stable.
- Linear + ReLU + linear FFN scheduling. ReLU itself is cheap, but fusing bias/ReLU around GEMM epilogues can reduce traffic.
- Position ID fast path for all-ones masks. Useful for common unpadded prefill/decode.
- Cache append plus attention kernel boundary. Avoid separate K/V concat materialization.

Lower priority:

- Sequence classification rightmost-non-pad pooling.
- Question-answering split/squeeze head.
- Dropout/layerdrop training behavior; inference treats dropout as identity and layerdrop inactive.
- Beam-search cache reorder. Needed for beam parity, not for first greedy/sampling runtime.

## 11. Runtime staging plan

Stage 1: Config and weights.

- Parse `OPTConfig`, including `word_embed_proj_dim`, `do_layer_norm_before`, `_remove_final_layer_norm`, `enable_bias`, and tokenizer/generation special IDs.
- Load tied token embedding / LM head as one logical parameter alias.
- Load `project_in`/`project_out` only when `word_embed_proj_dim != hidden_size`.

Stage 2: One-block eager parity.

- Implement embedding, OPT position IDs/offset, LayerNorm, linear+bias, ReLU, residuals, and eager attention for small shapes.
- Include both pre-LN (`opt-125m`) and post-LN/projected (`opt-350m`) block tests.

Stage 3: Full prefill.

- Run all decoder layers for `[B,S]` without cache first.
- Add causal mask and padding-mask parity.
- Validate logits for `logits_to_keep=0` and `logits_to_keep=1`.

Stage 4: Decode with dynamic cache.

- Add per-layer K/V cache allocation/append.
- Validate position slicing by `past_seen_tokens`.
- Run one-token and multi-token decode chunks.

Stage 5: Optimized kernels.

- Swap dense projections and FFN to CUTLASS GEMM epilogues.
- Add FlashAttention/SDPA-compatible prefill and decode paths preserving pre-scaled Q.
- Add last-token-only vocab GEMM.

Stage 6: Optional heads and controller features.

- Add beam cache reorder, sequence classification pooling, and QA head only after CausalLM is stable.

## 12. Parity and validation plan

Recommended tests:

- Config parser tests for `opt-125m`, `opt-350m`, `opt-6.7b`, and `opt-30b`; assert dimensions, LN order, projection presence, and tied embedding contract.
- Position ID tests:
  - unpadded prefill produces rows `[2..S+1]`;
  - padded masks match Transformers cumsum behavior;
  - decode with `past_seen_tokens > 0` slices positions correctly.
- Single operator tests for LayerNorm, linear+bias, ReLU FFN, and LM head tying.
- Single attention-layer parity with no cache, then with a populated cache; verify cache tensor shape `[B, heads, T, D]`.
- Single decoder block parity for pre-LN and post-LN/projected variants.
- Full prefill logits parity for small random configs and official-shaped configs with random weights.
- Decode parity: prefill prompt, decode one token with cache, compare logits and updated cache.
- Padding parity: left-padded and right-padded batches with explicit attention masks.
- End-to-end text smoke: tokenize a small prompt with official tokenizer metadata, run HF and Dinoml, compare next-token logits/top-k.

Suggested tolerances:

- fp32 eager: `rtol=1e-4`, `atol=1e-5` for block/logits; tighter for isolated LayerNorm/linear.
- fp16 optimized: `rtol=1e-2`, `atol=1e-2` initially, with attention softmax and large vocab logits compared carefully.
- For fused attention, compare both logits and intermediate attention outputs before enabling sampling parity.

## 13. Performance probes

- Tokenizer throughput: prompts/sec and tokens/sec in CPU data pipeline.
- Prefill throughput by model size: `B x S` sweep for `S={128,512,2048}`.
- Decode throughput: tokens/sec for `B={1,4,16,64}` and cache lengths `{128,512,2048}`.
- KV cache memory: `2 * layers * B * heads * T * head_dim * dtype_bytes`.
- OPT-350M projection overhead: measure `project_in/project_out` and smaller LM-head input width separately.
- Attention backend comparison: eager reference vs SDPA/FlashAttention provider for prefill and decode.
- LM head cost: all-token logits vs last-token-only logits for decode.
- GEMM provider probes: attention projection, FFN up/down, and vocab GEMM separately.
- Padding/mask overhead: all-ones mask fast path vs padded batch with materialized mask.

## 14. Skip/defer list

Safe to defer for first CausalLM integration:

- Training loss and label shifting.
- Dropout and layerdrop stochastic behavior.
- Gradient checkpointing.
- Sequence classification and question answering heads.
- Beam search cache reordering, unless beam generation is in the first product target.
- Quantization-specific checkpoint formats and tensor-parallel sharding.
- Cache offload and static cache export support.
- Flex-attention/block-mask special cases beyond normal causal plus padding masks.
- Remote-code or non-official OPT derivatives that alter architecture.

Do not defer:

- `opt-350m` pre/post-LN and projection behavior if the family target includes common official checkpoints.
- Learned position offset `+2`.
- Tied LM head / token embedding aliasing.
- Cache shape and query pre-scaling order.

## 15. Final implementation checklist

- [ ] Parse OPT config fields and effective defaults.
- [ ] Load OPT weights with tied `embed_tokens.weight` / `lm_head.weight` aliasing.
- [ ] Implement token embedding and tied LM head.
- [ ] Implement OPT learned absolute position IDs and `+2` offset.
- [ ] Implement optional `project_in` / `project_out` for `word_embed_proj_dim != hidden_size`.
- [ ] Implement pre-LN and post-LN decoder block variants.
- [ ] Implement MHA projections with query pre-scaling.
- [ ] Implement causal/padding mask parity for eager reference.
- [ ] Implement DynamicCache-compatible K/V append with `[B, heads, T, D]` layout.
- [ ] Implement FFN `Linear -> ReLU -> Linear`.
- [ ] Implement final decoder LayerNorm gated by `do_layer_norm_before` and `_remove_final_layer_norm`.
- [ ] Implement `logits_to_keep` slicing and last-token-only LM head optimization.
- [ ] Add config sweep tests for `opt-125m`, `opt-350m`, `opt-6.7b`, and `opt-30b`.
- [ ] Add position/padding/cached-decode parity tests.
- [ ] Add one-block and full-prefill logits parity tests.
- [ ] Add decode token parity and cache-shape tests.
- [ ] Benchmark prefill, decode, KV memory, attention backend, and LM-head variants.
