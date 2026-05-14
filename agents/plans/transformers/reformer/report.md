# Reformer Transformers Audit

## 1. Source Basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: primary target `google/reformer-crime-and-punishment`; additional configs `google/reformer-enwik8`, `hf-internal-testing/tiny-random-reformer`

Config source:

- `agents/plans/transformers/reformer/_sources/google__reformer-crime-and-punishment.config.json`
- `agents/plans/transformers/reformer/_sources/google__reformer-enwik8.config.json`
- `agents/plans/transformers/reformer/_sources/hf-internal-testing__tiny-random-reformer.config.json`
- tokenizer snapshots in the same `_sources/` directory where available

Source files inspected:

- `X:/H/transformers/src/transformers/models/reformer/configuration_reformer.py`
- `X:/H/transformers/src/transformers/models/reformer/modeling_reformer.py`
- `X:/H/transformers/src/transformers/models/reformer/tokenization_reformer.py`
- `X:/H/transformers/docs/source/en/model_doc/reformer.md`
- `X:/H/transformers/tests/models/reformer/test_modeling_reformer.py`

Primary source URLs for future re-check:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/reformer/modeling_reformer.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/reformer/configuration_reformer.py`
- `https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json`
- `https://huggingface.co/google/reformer-enwik8/resolve/main/config.json`

Any missing files or assumptions: no remote code is required for the inspected native source path. The report targets inference, primarily causal LM generation via `ReformerModelWithLMHead`; masked LM, classification, and QA heads are optional/deferred heads.

## 2. High-Level Architecture

Reformer is a text-only Transformer variant with axial position embeddings, reversible residual layers, and either local chunk attention or LSH chunk attention per layer.

Dataflow:

```text
tokenizer/data pipeline -> input_ids/attention_mask -> token embedding + axial/standard position embedding
-> reversible encoder stack with configured local/LSH layers
-> final LayerNorm over concatenated reversible streams
-> LM/classification/QA head -> logits/postprocessing
```

Primary runtime target:

```text
causal LM prefill -> ReformerDynamicCache(bucket/state cache) -> one-token decode -> logits/sampling
```

The encoder stack can be validated independently from the LM head. Attention kernels should be staged separately for dense fallback, local attention, and LSH attention because LSH requires random rotations, stable sorting, gathering, reverse sorting, and hash-round combination.

## 3. Important Config Dimensions

| Field | `crime-and-punishment` | `enwik8` | `tiny-random-reformer` |
|---|---:|---:|---:|
| architecture | `ReformerModelWithLMHead` | `ReformerModelWithLMHead` | config only |
| primary task | causal LM | causal LM | tiny tests |
| `is_decoder` | `true` | `true` | default `false` |
| `vocab_size` | 320 | 258 | 1000 |
| `hidden_size` | 256 | 1024 | 32 |
| `num_attention_heads` | 2 | 8 | 2 |
| `attention_head_size` | 64 | 128 | 64 |
| all-head size | 128 | 1024 | 128 |
| `feed_forward_size` | 512 | 4096 | 32 |
| layers from `attn_layers` | 6 | 12 | 4 |
| `attn_layers` | local/lsh alternating | mostly local, LSH at 2/6/10 | all local |
| `num_hashes` | 1 | 4 | 1 |
| `num_buckets` | `[64, 128]` | `512` | `None` |
| LSH chunk | 64 | 256 | `None` |
| local chunk | 64 | 128 | 4 |
| chunks before/after | source defaults or explicit `1/0` | `1/0` | local `1/0` |
| axial positions | yes | yes | yes |
| `axial_pos_shape` | `[512, 1024]` | `[128, 512]` | `[4, 25]` |
| `axial_pos_embds_dim` | `[64, 192]` | `[256, 768]` | `[16, 16]` |
| `max_position_embeddings` | 524288 | 65536 | 512 |
| FFN activation | ReLU | ReLU | GELU |
| cache support | `ReformerDynamicCache` | `ReformerDynamicCache` | source default |

Effective source defaults when omitted include `lsh_num_chunks_before=1`, `lsh_num_chunks_after=0`, `local_num_chunks_before=1`, `local_num_chunks_after=0`, `tie_word_embeddings=False`, `chunk_size_feed_forward=0`, `use_cache=True`, and `layer_norm_eps=1e-12`.

## 3a. Family Variation Traps

- `hidden_size` is not guaranteed to equal `num_attention_heads * attention_head_size`. The attention projection output size is `all_head_size`; attention output projects `all_head_size -> hidden_size`.
- `attn_layers` is the real layer count and schedule. Only `"lsh"` and `"local"` are implemented.
- Causal LM requires `config.is_decoder=True`; causal generation asserts no future chunks for both local and LSH attention (`*_num_chunks_after == 0`).
- LSH `num_buckets` can be an integer, a factor list, or `None`. If `None`, source mutates `config.num_buckets` at runtime based on sequence length.
- LSH hashing samples random rotations each forward unless `hash_seed` is set. DinoML needs a deterministic RNG contract, precomputed buckets, or an admission rule that requires fixed hash behavior.
- Axial embeddings require `sum(axial_pos_embds_dim) == hidden_size`; in training, `prod(axial_pos_shape) == sequence_length`, while inference only requires enough product for the used positions.
- Reversible layers produce a final hidden state of shape `[B, S, 2 * hidden_size]`; heads consume this doubled dimension.
- FFN and LM head chunking are memory optimizations via `apply_chunking_to_forward`, not semantic changes. They can be lowered as whole-tensor ops first.
- `tiny-random-reformer` is local-attention-only and not representative of LSH, cache, or causal LM behavior.
- No NHWC/channel-last issue applies; all tensors are token sequences `[batch, sequence, channels]`. Layout rewrites should be guarded around sequence-axis chunking and head-axis reshapes.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- `Embedding(input_ids)`, optional `inputs_embeds` bypass
- `arange`, `expand`, `cat`, `chunk`, `reshape/view`, `transpose`, `flatten`, `unsqueeze`, `squeeze`
- `pad`/concat padding to least common multiple of active chunk lengths
- `gather`, `scatter_`, `index_select`, `nonzero`, `argsort` with stable tie behavior
- integer arithmetic: `%`, `//`, `bit_length`-derived bucket sizing, sequence/chunk divisibility checks

Neural primitives:

- biasless attention projections: LSH `query_key: hidden -> heads * head_dim`, `value: hidden -> heads * head_dim`; local `query/key/value`
- biasless attention output `heads * head_dim -> hidden`
- FFN: `LayerNorm(hidden) -> Linear(hidden -> feed_forward_size, bias) -> activation -> Linear(feed_forward_size -> hidden, bias)`
- final encoder norm `LayerNorm(2 * hidden_size)`
- LM head `Linear(2 * hidden_size -> vocab_size, bias=False) + separate bias parameter in module`
- classification head: first-token pool, dropout, `Linear(2H -> H)`, tanh, `Linear(H -> labels)`
- QA head: `Linear(2H -> num_labels)` then split start/end

Attention primitives:

- Local chunk attention with adjacent chunk lookup and optional causal mask.
- LSH attention with shared query/key projection, vector length normalization, random rotations, bucket assignment, stable sort, gather-by-bucket, adjacent chunk lookup, self-mask, logsumexp softmax, reverse sort, and multi-hash weighted combine.
- Dense standard fallback when `sequence_length <= chunk_length`.

Position math:

- Axial position embedding materialization from factor weights and `position_ids`.
- Optional standard learned position embedding if `axial_pos_embds=False`.

Generation/cache ops:

- `ReformerDynamicCache` stores per-layer `(buckets, states)`, not KV tensors.
- Cache update concatenates states on sequence axis and LSH buckets on last axis.
- Beam reorder uses `index_select(0, beam_idx)` for both cached buckets and states.
- Decode with a non-empty cache only supports one new token per call.

Preprocessing-coupled ops:

- Tokenizer is BPE with metaspace pre-tokenizer/decoder; model inputs are `input_ids` and `attention_mask`.
- No token type IDs are used despite the embedding docstring.

## 5. Layer/Block Breakdown

Embedding:

```text
position_ids = arange(start_idx, start_idx + S) if absent
inputs = word_embedding(input_ids) or inputs_embeds
x = dropout(inputs) + position_embedding(position_ids)
```

Reversible encoder setup:

```text
y1, y2 = x, x
for each configured layer:
  y1 = y1 + attention_block(y2)
  y2 = y2 + feed_forward_block(y1)
out = LayerNorm(concat(y1, y2))  # shape [B, S, 2H]
```

Attention block:

```text
u = LayerNorm(hidden)
attn = LSHSelfAttention(u) or LocalSelfAttention(u)
return Linear(heads * head_dim -> H, bias=False)(attn)
```

FFN block:

```text
u = LayerNorm(y1)
u = Linear(H -> FF, bias=True)(u)
u = dropout(activation(u))
u = Linear(FF -> H, bias=True)(u)
```

For `crime-and-punishment`, local layers use Q/K/V `256 -> 128`, output `128 -> 256`, FFN `256 -> 512 -> 256`, and LM head `512 -> 320`. For `enwik8`, those are `1024 -> 1024`, `1024 -> 4096 -> 1024`, and `2048 -> 258`.

## 6. Attention Requirements

Local attention:

- Self-attention only.
- MHA with separate Q/K/V; no MQA/GQA.
- Chunked when `S > local_attn_chunk_length`; otherwise dense.
- For chunked mode, keys/values include `local_num_chunks_before` and `local_num_chunks_after` adjacent chunks via rolling concat.
- Decoder mode applies `query_index >= key_index` causal masking.
- Attention mask is boolean-expanded to attention logits shape; masked values use `-1e4` for fp16 and `-1e9` for fp32.

LSH attention:

- Self-attention only with tied Q/K projection named `query_key`.
- Query/key vectors are length-normalized and divided by `sqrt(head_dim)`.
- Hashing detaches vectors, optionally sets `torch.manual_seed(hash_seed)`, samples random rotations, assigns buckets by argmax over positive/negative rotations, and offsets bucket IDs by hash round.
- Padding tokens can be assigned an extra bucket.
- Sort order uses source `_stable_argsort`, implemented by scaling values with position offsets before `torch.argsort`.
- For chunked LSH, sorted tokens are gathered, chunked, attend over adjacent chunks, then reverse-sorted. Multiple hash rounds are combined by softmax over per-round logits.
- LSH always applies a self-mask that prevents attending to the same token except when masking leaves no useful alternative, approximated by finite negative self-mask values (`-1e3` fp16, `-1e5` fp32).

Cache/generation:

- This is not a standard KV cache. Per layer, cached states are `[B, cached_S, H]`; LSH cached buckets are `[B, heads, num_hashes, cached_S]`, while local layers store an empty bucket tensor.
- With past cache, source asserts the new sequence length is 1.
- Local decode recomputes K/V only over the relevant previous chunk tail plus the new token.
- LSH decode hashes the query, merges it with cached buckets, finds the relevant sorted bucket chunk(s), selects hidden states, and attends over that subset.
- `prepare_inputs_for_generation` drops the attention mask and passes `past_buckets_states` as `past_key_values`; position IDs are suppressed because Reformer computes them from cache length.
- No FlashAttention/SDPA dispatch is present. A first DinoML port should use custom local/LSH kernels or a faithful decomposition, not standard dense attention, except for the source fallback cases.

## 7. Position Encoding and Custom Math

Axial position encoding is required for both Google checkpoints. It stores one parameter per axis, broadcasts those factors, concatenates them on the channel axis, reshapes to a flat position table, and gathers by `position_ids`.

Concise source-equivalent inference sketch:

```python
def axial_position(position_ids, weights, axial_shape):
    # weights[i] has shape [1, ..., axial_shape[i], ..., 1, dim_i]
    batch = position_ids.shape[0]
    expanded = [w.expand((batch, *axial_shape, w.shape[-1])) for w in weights]
    max_pos = int(position_ids.max()) + 1
    cols = (max_pos + axial_shape[1] - 1) // axial_shape[1]
    table = torch.cat([w[:, :cols] for w in expanded], dim=-1)
    table = table.reshape(batch, -1, table.shape[-1])
    return torch.cat([table[i].index_select(0, position_ids[i])[None] for i in range(batch)], dim=0)
```

LSH hash sketch:

```python
def lsh_hash(vectors, rotations, num_hashes, num_buckets):
    rotated = torch.einsum("bmtd,mdhr->bmhtr", vectors.detach(), rotations)
    rotated = torch.cat([rotated, -rotated], dim=-1)
    buckets = torch.argmax(rotated, dim=-1)
    offsets = torch.arange(num_hashes, device=vectors.device).view(1, 1, -1, 1) * num_buckets
    return (buckets + offsets).flatten(2, 3)
```

Precomputable: axial weights are constants, but gathered position embeddings depend on `position_ids` and cache start index. LSH rotations are not model parameters in HF source; reproducible inference requires controlling or reproducing the source RNG sequence.

## 8. Preprocessing and Input Packing

CPU/data pipeline:

- ReformerTokenizer uses BPE with metaspace normalization. Runtime graph receives `input_ids: [B, S]` and optional `attention_mask: [B, S]`.
- No image/audio/multimodal packing exists.
- No token type IDs enter the graph.

GPU/runtime:

- If `position_ids` are absent, generate them from `start_idx_pos_encodings`, which is the cached state length during decode.
- In eval prefill, sequences longer than the minimum chunk length but not divisible by the LCM of active chunk lengths are padded. Attention mask is padded with zeros; outputs are sliced back to the original sequence length.
- For causal LM generation, source feeds only the last token after a past cache exists.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Chunked FFN/LM Head -> Whole-Tensor Linear

Source pattern: `apply_chunking_to_forward(forward_chunk, chunk_size, seq_dim=1, hidden_states)`.

Replacement: if `chunk_size == 0` or compile target has enough memory, lower as a single linear/activation/linear or LM projection.

Preconditions:

- Inference only or dropout disabled.
- No desired memory cap forcing chunked execution.
- Sequence-axis chunking has no cross-chunk dependency.

Failure cases: training/dropout checkpointing parity and strict memory-bounded inference.

Parity test sketch: compare whole-tensor and chunked FFN/LM logits for fixed random tensors in fp32/fp16.

### Rewrite: Dense Fallback Attention

Source pattern: local or LSH attention with `sequence_length <= chunk_length`.

Replacement: standard self-attention math with the model-specific masks and projection layouts.

Preconditions:

- Preserve LSH shared Q/K projection and self-mask behavior.
- Preserve local Q/K/V projection for local layers.
- Preserve finite mask values for dtype-sensitive parity.

Failure cases: chunked path, LSH multi-hash path, or generation with cached relevant chunks.

### Rewrite: Axial Position Gather -> Precomputed Position Table

Source pattern: broadcast axial weights, concatenate, flatten, index-select by position IDs.

Replacement: precompute the flat `[max_position_embeddings, hidden_size]` table at artifact build or module load for fixed axial weights.

Preconditions:

- Inference only, dropout disabled.
- `axial_pos_shape` product covers admitted maximum position.
- Memory budget accepts the dense table. For `crime-and-punishment`, dense table is large: `524288 * 256`.

Failure cases: very long contexts where dense table defeats axial memory savings, dynamic position shapes beyond the configured product.

### Rewrite: LSH Sort/Gather Attention Kernel

Source pattern: hash -> stable sort -> gather QK/V -> chunk attention -> reverse sort -> hash combine.

Replacement: custom provider-backed LSH attention family with explicit bucket/sort metadata.

Preconditions:

- Fixed `num_buckets`, `num_hashes`, chunk length, chunks before/after.
- Deterministic rotations or runtime RNG contract.
- Sequence length divisible by chunk LCM after source-compatible padding.

Failure cases: `num_buckets=None` runtime mutation, factorized buckets not implemented, `output_attentions=True` dense reconstruction requirements.

## 10. Kernel Fusion Candidates

Highest priority:

- Local chunk attention kernel: chunked QK, mask, softmax, and AV dominate local layers and are easier than LSH.
- LSH sort/gather attention pipeline: required for production Reformer behavior; off-the-shelf FlashAttention is not a drop-in replacement.
- Stable argsort/gather/scatter support: LSH cannot be faithful without bucket sorting and reverse sorting.
- LayerNorm + linear projection scheduling: every attention and FFN block is pre-norm.

Medium priority:

- Axial position gather kernel or precompute policy: important for long contexts.
- Reformer cache ABI: bucket/state cache is unique and must be explicit before generation parity.
- FFN fusion `Linear + ReLU/GELU + Linear` with optional chunk scheduling.
- Last-token-only logits using `logits_to_keep` to avoid projecting all prefill positions when generation only needs the tail.

Lower priority:

- Classification/QA heads.
- `output_attentions=True` reconstruction and storage.
- Training-only reversible backward mechanics and dropout seed replay.

## 11. Runtime Staging Plan

Stage 1: parse configs and load weights for `ReformerModel` and `ReformerModelWithLMHead`; reject unknown `attn_layers`, unsupported `num_buckets=None`, and nondeterministic hashing unless a policy is chosen.

Stage 2: implement embedding plus axial position parity and one no-cache local-attention block.

Stage 3: add dense fallback and chunked local attention for encoder/prefill.

Stage 4: add LSH prefill with fixed buckets, stable sort, gather/reverse-sort, adjacent chunks, self-mask, and multi-hash combine.

Stage 5: add `ReformerDynamicCache` state/bucket ABI and one-token decode for local layers, then LSH layers.

Stage 6: add LM head with `logits_to_keep`; validate `crime-and-punishment` prefill and decode logits.

Stage 7: optimize provider kernels for local attention, LSH pipeline, and axial positions; only then add optional heads.

Initially stub/defer: training, dropout, `output_attentions`, classification/QA, beam-search reorder beyond basic `index_select`, and runtime auto-selection for `num_buckets=None`.

## 12. Parity and Validation Plan

- Config parser tests for all three captured configs, including default-filled fields.
- Axial position tests for contiguous and offset `position_ids`, including decode start index.
- Stable argsort, factorized bucket hash, padding bucket, and reverse-sort unit tests.
- Single local-attention layer parity with and without causal mask.
- Single LSH layer parity with fixed `hash_seed`, `num_buckets` int, and `num_buckets` list.
- Chunk padding parity: input length not divisible by chunk LCM in eval should pad and slice back.
- Cache parity: prefill then one-token decode should match HF for local-only and mixed local/LSH configs.
- Full `ReformerModelWithLMHead` logits parity on `google/reformer-crime-and-punishment`, first fp32 then fp16/bf16 as available.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for decomposed math; fp16 `rtol=1e-2, atol=1e-2`, with stricter local tests if kernel math matches ordering.

## 13. Performance Probes

- Axial position materialization time and memory versus precomputed table.
- Local attention throughput sweep by `S`, chunk length, chunks before/after, batch, and heads.
- LSH pipeline breakdown: projection, hashing, sort, gather, chunk attention, reverse sort, hash combine.
- Prefill throughput for `S=512, 4096, 65536` where memory allows.
- Decode tokens/sec with cache lengths across chunk boundaries.
- Cache memory: per-layer states `[B, S, H]` plus LSH buckets `[B, heads, num_hashes, S]`.
- Last-token-only logits versus full logits.
- Factorized buckets versus integer buckets.

## 14. Skip/Defer List

- Training and reversible backward.
- Dropout and dropout seed replay.
- `output_attentions=True` except for debugging.
- Masked LM, sequence classification, and QA heads for first causal-LM integration.
- `num_buckets=None` runtime mutation unless explicitly admitted with an artifact-visible bucket plan.
- Beam search beyond basic cache reorder.
- Multi-GPU/tensor parallel.
- Quantization.

## 15. Final Implementation Checklist

- [ ] Parse `ReformerConfig` including `attn_layers`, axial fields, bucket fields, and source defaults.
- [ ] Load token embeddings, axial position weights, reversible encoder weights, and LM head weights.
- [ ] Implement axial position embedding gather or guarded precompute.
- [ ] Implement reversible forward shape contract with final `2 * hidden_size` norm.
- [ ] Implement local attention dense fallback and chunked adjacent-window path.
- [ ] Implement LSH hashing with deterministic rotations, stable sort, gather, reverse sort, and hash combine.
- [ ] Implement source-compatible masks, including causal, padding, LSH self-mask, and dtype-specific finite mask values.
- [ ] Define explicit `ReformerDynamicCache` ABI for buckets and states.
- [ ] Add one-token decode path and cache reorder.
- [ ] Add LM head with `logits_to_keep`.
- [ ] Add parity tests for axial positions, local attention, LSH attention, padding, cache decode, and full LM logits.
- [ ] Benchmark local attention, LSH pipeline phases, prefill, decode, logits, and cache memory.
