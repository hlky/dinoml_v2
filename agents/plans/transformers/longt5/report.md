# LongT5 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/long-t5-local-base, google/long-t5-local-large,
  google/long-t5-tglobal-base, google/long-t5-tglobal-large,
  google/long-t5-tglobal-xl, Stancld/longt5-tglobal-large-16384-pubmed-3k_steps
Config source: Hugging Face config.json snapshots under
  agents/plans/transformers/longt5/_sources/
Source files inspected:
  X:/H/transformers/src/transformers/models/longt5/configuration_longt5.py
  X:/H/transformers/src/transformers/models/longt5/modeling_longt5.py
  X:/H/transformers/src/transformers/models/longt5/__init__.py
  X:/H/transformers/src/transformers/masking_utils.py, for decoder causal mask call boundary
Any missing files or assumptions:
  No remote-code files are required for the in-library LongT5 family.
  Tokenizer files were not deeply audited; LongT5 uses T5/SentencePiece-style
  text tokenization and no model-coupled multimodal/audio preprocessing.
  The primary runtime target in this report is seq2seq conditional generation.
```

Primary source URLs at the pinned commit:

- `configuration_longt5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longt5/configuration_longt5.py
- `modeling_longt5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longt5/modeling_longt5.py
- `__init__.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/longt5/__init__.py

## 2. High-level architecture

LongT5 is a text-only encoder-decoder model. The decoder is T5-style dense causal self-attention plus dense encoder-decoder cross-attention. The encoder is the family-specific part: it uses either local sparse self-attention or transient-global sparse self-attention, selected by `encoder_attention_type`.

```text
SentencePiece/tokenization -> input_ids/attention_mask
-> shared token embedding
-> LongT5 encoder with local or transient-global sparse self-attention
-> decoder prefill/decode with dense causal self-attention and cross-attention
-> optional d_model**-0.5 scale when tie_word_embeddings=true
-> lm_head -> logits/sampling
```

Independently stageable pieces:

- CPU/data pipeline: SentencePiece tokenization, pad/eos handling, decoder start token construction.
- Encoder runtime: embedding, block padding, local/transient-global attention, RMSNorm, gated FFN.
- Decoder runtime: dense T5 attention, encoder-decoder cross-attention, cache update/reuse, LM head.
- Optional encoder-only target: `LongT5EncoderModel` exposes only the sparse encoder stack.

## 3. Important config dimensions

Shape symbols: `B=batch`, `S=encoder/source length`, `T=decoder target length`, `H=d_model`, `A=num_heads`, `D=d_kv`, `K=A*D`, `I=d_ff`, `V=vocab_size`, `R=local_radius`, `L=R+1`, `G=global_block_size`.

| Field | Source default | Runtime significance |
|---|---:|---|
| `vocab_size` | 32128 | Shared embedding and LM projection width. |
| `d_model` / `H` | 512 | Hidden width; official base configs use 768. |
| `d_kv` / `D` | 64 | Per-head Q/K/V dim. |
| `num_heads` / `A` | 8 | MHA only; no GQA/MQA. |
| `inner_dim` / `K` | `A * D` | Projection width; not required by source to equal `H`. |
| `d_ff` / `I` | 2048 | FFN intermediate width. |
| `num_layers` | 6 | Encoder layers and decoder default. |
| `num_decoder_layers` | defaults to `num_layers` | Decoder layer count can differ, though inspected checkpoints keep them equal. |
| `encoder_attention_type` | `local` | Must be `local` or `transient-global`; other values raise in `LongT5Block`. |
| `local_radius` / `R` | 127 | Encoder local block length is `L=128`; sequence is padded to a multiple of `L`. |
| `global_block_size` / `G` | 16 | Only used by transient-global encoder attention. |
| `relative_attention_num_buckets` | 32 | Learned per-head relative bias table rows. |
| `relative_attention_max_distance` | 128 | Log bucket saturation distance. |
| `feed_forward_proj` | `relu` | Source default differs from official configs, which use `gated-gelu`. |
| `tie_word_embeddings` | true | Official configs set false, so do not assume LM head aliases embeddings. |
| cache support | true for decoder | Encoder forcibly disables cache; decoder uses `EncoderDecoderCache`. |

Representative checkpoint sweep, from `config.json` snapshots:

| Model/config | Encoder attn | H | A | D | K | I | Enc/dec layers | FFN | Vocab | `n_positions` | Tie LM |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| `google/long-t5-local-base` | local | 768 | 12 | 64 | 768 | 2048 | 12/12 | gated-gelu | 32128 | 4096 | false |
| `google/long-t5-local-large` | local | 1024 | 16 | 64 | 1024 | 2816 | 24/24 | gated-gelu | 32128 | 4096 | false |
| `google/long-t5-tglobal-base` | transient-global | 768 | 12 | 64 | 768 | 2048 | 12/12 | gated-gelu | 32128 | 4096 | false |
| `google/long-t5-tglobal-large` | transient-global | 1024 | 16 | 64 | 1024 | 2816 | 24/24 | gated-gelu | 32128 | 4096 | false |
| `google/long-t5-tglobal-xl` | transient-global | 2048 | 32 | 64 | 2048 | 5120 | 24/24 | gated-gelu | 32128 | 4096 | false |
| `Stancld/...pubmed-3k_steps` | transient-global | 1024 | 16 | 64 | 1024 | 2816 | 24/24 | gated-gelu | 32100 | 4096 | false |

Observed config-only or historical fields: `n_positions` and `output_past` appear in checkpoint configs but are not used by the inspected LongT5 modeling code for tensor shapes. Treat `n_positions=4096` as tokenizer/training metadata, not an absolute position embedding length.

## 3a. Family variation traps

- Source defaults do not match official checkpoints: source default is `d_model=512`, `feed_forward_proj="relu"`, `tie_word_embeddings=true`; Google checkpoints use larger widths, `gated-gelu`, and untied LM heads.
- `encoder_attention_type` changes operator structure. `local` needs 3-block sparse local attention; `transient-global` additionally needs block-id construction, block aggregate creation, side/global K/V paths, and side relative bias.
- LongT5 encoder sparse attention is noncausal and cache-free. Decoder self/cross attention is dense T5 attention with cache.
- Only block 0 has learned relative attention bias modules; its computed `position_bias` is reused by later layers in a stack.
- `local_radius=127` implies block length 128. Inputs whose sequence length is not divisible by 128 are padded inside sparse attention helpers.
- `global_seq_len = S // global_block_size` in transient-global attention. Orphan tokens are assigned to the preceding full block; if `S < G`, the global side sequence can be length 0.
- All Q/K/V/O and FFN Linear modules are bias-free. DinoML should reject optional bias assumptions.
- `K=A*D` happens to equal `H` for inspected checkpoints, but the source computes projection width independently.
- Checkpoint vocab can differ (`Stancld` PubMed config uses 32100), so tokenizer/vocab pairing must be config-specific.
- No NCHW/NHWC layout translation applies; tensors are text-major `[B, sequence, hidden]`. Sparse attention uses axis-sensitive reshape, pad, cat, transpose, and einsum patterns that should be protected from generic layout rewrites.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(input_ids) -> [B,S,H]` and decoder embedding.
- Reshape/view, transpose, contiguous, slice, pad-to-multiple, concatenate, repeat, unsqueeze/squeeze.
- Boolean masks, `where`, logical-and, equality, comparison, `cumsum`, `floor`, `arange`, `max`, `sum`.
- One-hot block IDs and `einsum("...nd,...ng->...gd")` for transient global aggregates.

Neural network primitives:

- Bias-free Linear for Q/K/V/O, FFN, and LM head.
- LongT5/T5 RMSNorm: mean of squares over last dim, fp32 accumulation, rsqrt, learned scale only.
- ReLU FFN source path and gated-GELU FFN checkpoint path: `gelu_new(wi_0(x)) * wi_1(x) -> wo`.
- Residual adds and inference dropout elision.
- fp16 inf clamp after self-attention, cross-attention, and FFN when running half precision.

Attention primitives:

- Dense decoder self-attention and cross-attention: matmul scores, relative bias/mask add, fp32 softmax, dropout, matmul values, output projection.
- Encoder local sparse attention over 3 adjacent blocks with relative bias.
- Encoder transient-global sparse attention: local 3-block keys/values plus per-block global side keys/values.

Position/relative-bias ops:

- T5 bucketed relative position bias for dense attention.
- Local 3-block relative bias over `[L, 3L]`.
- Transient-global side relative bias over `[B, S, S//G]`.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` construction.
- Per-layer self-attention cache append.
- Per-layer cross-attention cache creation once and reuse via `is_updated`.
- Decoder start token is `pad_token_id` / `decoder_start_token_id=0` in inspected configs.

Preprocessing-coupled ops:

- SentencePiece tokenization and text padding are CPU/data-pipeline work.
- No image/audio/discrete-code processor or modality scatter path.

## 5. Layer/block breakdown

Encoder block, repeated `num_layers` times:

```text
x: [B,S,H]
y = RMSNorm(x)
if encoder_attention_type == "local":
  q,k,v = Linear(H -> K), reshape to [B,S,A,D]
  split q into [B,ceil(S/L),L,A,D]
  split k/v and concatenate previous/current/next blocks -> [B,ceil(S/L),3L,A,D]
  scores = einsum(q,k) -> [B,blocks,A,L,3L]
  scores += local relative bias + local mask
else:
  block_ids, global_segment_ids = fixed block ids from attention_mask
  global_inputs = RMSNorm(einsum(hidden_states, one_hot(block_ids)))
  local q,k,v as above
  side_k, side_v = Linear(global_inputs), repeated over local blocks
  scores over [3L + global_seq_len] keys
  scores += local relative bias/mask + side relative bias/mask
attn = softmax(scores.float(), dim=-1).type_as(scores)
x = x + Linear(K -> H)(attn @ values)
x = x + FFN(RMSNorm(x))
```

Decoder block, repeated `num_decoder_layers` times:

```text
x = x + DenseSelfAttention(RMSNorm(x), causal_mask, self_cache)
x = x + CrossAttention(RMSNorm(x), encoder_hidden_states, encoder_mask, cross_cache)
x = x + FFN(RMSNorm(x))
```

FFN variants:

- `relu`: `wi: H -> I`, activation, `wo: I -> H`.
- `gated-gelu`: `wi_0: H -> I`, `wi_1: H -> I`, `gelu_new(wi_0(x)) * wi_1(x)`, `wo: I -> H`.

Task heads:

- Required for primary target: `LongT5ForConditionalGeneration` LM head `[H,V]`, bias false.
- Optional: `LongT5Model` returns decoder hidden states without LM logits.
- Optional/deferred: `LongT5EncoderModel` for encoder-only feature extraction.

Parameter sharing:

- `LongT5Model` ties encoder and decoder token embeddings to `shared.weight`.
- `LongT5ForConditionalGeneration` also declares `lm_head.weight` tied to `shared.weight`, but inspected official configs set `tie_word_embeddings=false`; implementations must honor config/loaded weights rather than always aliasing.
- Relative bias modules are physically present only in layer 0 for encoder/decoder stacks, with computed bias tensors reused across layers.

## 6. Attention requirements

Dense decoder attention:

- Causal self-attention and noncausal cross-attention.
- MHA with `A` query heads and `A` KV heads, head dim `D`; no MQA/GQA.
- No explicit `1/sqrt(D)` scaling in source attention math.
- Mask/bias order: compute matmul scores, build or receive `position_bias`, add causal/encoder mask into bias, add bias to scores, softmax in fp32, cast back.
- Cache tensors are projected K/V in shape `[B,A,T,D]` for self-attention after transpose. Cross-attention K/V shape is `[B,A,S,D]` and is reused after first decode step.
- Cached keys are not position-encoded; LongT5 uses additive relative bias, not RoPE.

Encoder local attention:

- Noncausal sparse self-attention, no cache.
- Source sequence is padded to `L=local_radius+1`.
- Each query block attends previous/current/next key blocks, but `_mask_local_attention_mask` keeps only relative positions with `abs(relative_position) < L`.
- Attention tensor shape before flattening: `[B, ceil(S/L), A, L, 3L]`.
- Output attentions, if requested, are block-local tensors, not dense `[B,A,S,S]` reconstructions.

Encoder transient-global attention:

- Same local path plus side/global tokens.
- `global_inputs` are sums of hidden states per fixed block, then RMSNorm. They are transient: recomputed every layer, not learned persistent global tokens.
- Side keys/values are projected with the same `k`/`v` modules as local keys/values.
- Per-block key length is `3L + floor(S/G)` before masking/padding effects.
- Side bias combines a mask derived from token block IDs and learned global relative bias.

FlashAttention/SDPA compatibility:

- Decoder dense attention can be lowered to a T5-relative-bias-capable attention primitive or decomposed GEMM/softmax/GEMM first.
- Encoder sparse paths are not directly SDPA-compatible without a custom local/transient-global kernel or a guarded dense fallback. Dense fallback is acceptable only for small validation shapes because long-context memory grows as `S^2`.

## 7. Position encoding and custom math

LongT5 has no absolute position embeddings and no RoPE/ALiBi. It uses T5-style learned relative attention bias buckets.

Concise bucket math:

```python
def relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets += (relative_position > 0).long() * num_buckets
        relative_position = abs(relative_position)
    else:
        relative_position = -min(relative_position, 0)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + log(relative_position / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)
    large = min(large.long(), num_buckets - 1)
    return buckets + where(is_small, relative_position, large)
```

Local 3-block relative IDs:

```python
memory = arange(3 * L)
context = memory[L:-L]
relative_position = memory[None, :] - context[:, None]  # [L, 3L]
```

Transient-global block aggregates:

```python
one_hot = one_hot(where(block_ids >= 0, block_ids, global_seq_len), global_seq_len + 1)[..., :-1]
global_inputs = einsum("...nd,...ng->...gd", hidden_states, one_hot)
global_inputs = RMSNorm(global_inputs)
```

Precompute opportunities:

- Dense decoder relative bucket IDs for fixed `(query_length,key_length,past_seen_tokens)` cases can be cached per decode shape.
- Local `[L,3L]` bucket IDs are config-static.
- Transient side bucket IDs depend on `attention_mask`, block IDs, and dynamic `S`, so they are runtime shape/mask dependent.

## 8. Preprocessing and input packing

Runtime graph inputs:

- `input_ids: [B,S]` integer token IDs.
- `attention_mask: [B,S]` with 1 for valid tokens and 0 for padding.
- `decoder_input_ids: [B,T]` for prefill/decode, or generated from labels in training-only path.
- `decoder_attention_mask: [B,T]` optional; decoder causal mask is also applied.
- `encoder_outputs` may be supplied directly to skip encoder execution during generation.

Generation-controller behavior:

- Inspected generation configs contain `decoder_start_token_id=0`, `pad_token_id=0`, `eos_token_id=1`.
- No forced language/task IDs, timestamp rules, suppress-token processors, or multimodal placeholder expansion are model-specific.
- First integration can use externally supplied token IDs and simple greedy/logit parity; full `generate()` scheduling is a runtime/controller layer.

CPU/data-pipeline work:

- SentencePiece tokenization, padding/truncation, and special-token handling.
- Batch packing decisions. Sparse encoder kernels benefit from bucketing `S` to multiples of 128 and, for transient-global, useful multiples of 16.

## 9. Graph rewrite / lowering opportunities

### Rewrite: local attention blockization

Source pattern:

```text
pad_to_multiple -> reshape [B,S,A,D] to [B,blocks,L,A,D]
pad one block on both sides -> slice three shifted block tensors -> cat on key axis
```

Replacement pattern:

```text
LocalBlockGather(Q,K,V, block_len=L, radius=R) -> block sparse attention
```

Preconditions:

- Encoder-only self-attention, `encoder_attention_type="local"`.
- `L == local_radius + 1`; default `L=128`.
- Preserve source pad value 0 and locality mask `abs(relative_position) < L`.
- Output must be sliced back to original `S`.

Failure cases:

- Do not apply to decoder dense attention or cross-attention.
- Do not reconstruct dense attentions unless `output_attentions` parity requires it.

Parity test sketch: compare one encoder layer for `S` values below, equal to, and above multiples of 128 with mixed padding masks.

### Rewrite: transient-global side path

Source pattern:

```text
block_id construction -> one_hot -> einsum aggregate -> RMSNorm
-> shared K/V projection -> repeat over local blocks -> concat with local K/V
```

Replacement pattern:

```text
BlockAggregateSum + RMSNorm + SideKVProjection fused into transient-global attention kernel
```

Preconditions:

- `encoder_attention_type="transient-global"`.
- `global_block_size` positive; default 16.
- Block IDs must match source orphan-token and padding-token behavior exactly.

Failure cases:

- `global_seq_len=0` must not crash.
- Any attempt to average aggregates instead of summing before RMSNorm changes parity.

### Rewrite: gated FFN fusion

Source pattern:

```text
gelu_new(x @ wi_0.T) * (x @ wi_1.T) -> wo
```

Replacement:

```text
dual GEMM + fused gelu_new/mul + GEMM
```

Preconditions:

- `feed_forward_proj == "gated-gelu"` or other validated `gated-{act}`.
- Linear weights are PyTorch row-major `[out,in]`, bias absent.

Failure cases:

- Source default `relu` has only one input projection; do not force gated path.

### Rewrite: last-token logits

Source pattern:

```text
decoder hidden [B,T,H] -> optional scale -> lm_head [H,V]
```

Replacement:

```text
for decode step, project only [B,1,H] or selected last token
```

Preconditions:

- Generation decode step only; no full sequence logits requested.
- Preserve `tie_word_embeddings` scale rule.

## 10. Kernel fusion candidates

Highest priority:

- Local sparse encoder attention kernel for `[B,blocks,A,L,3L]`, including relative bias, mask add, fp32 softmax, and value matmul. This is the core LongT5 difference from T5.
- Decoder dense T5 attention with relative bias and cache. This can reuse T5 work and enables generation.
- RMSNorm with fp32 accumulation and learned scale only.
- Gated-GELU FFN dual projection fusion, because all inspected production configs use `gated-gelu`.

Medium priority:

- Transient-global attention fused kernel, including block aggregation and side K/V. It is needed for tglobal checkpoints but can follow local attention if staging starts with `google/long-t5-local-base`.
- Relative bias bucket generation/cache, especially local `[128,384]` static buckets.
- Cross-attention cache materialization/reuse to avoid reprojecting encoder K/V during decode.

Lower priority:

- Dense fallback encoder attention for tiny validation shapes.
- Output attention reconstruction or diagnostics.
- Training losses, dropout, and gradient-checkpointing behavior.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `google/long-t5-local-base`; implement embedding, RMSNorm, bias-free Linear, gated FFN, and relative bias tables.

Stage 2: one local encoder block parity with block padding, local masks, and first-layer relative bias reuse.

Stage 3: full local encoder parity and `LongT5EncoderModel` output parity.

Stage 4: decoder dense T5 block parity with self-attention cache disabled, then enabled for prefill/decode.

Stage 5: `LongT5ForConditionalGeneration` prefill logits parity, including untied LM head and no embedding-output alias when config says `tie_word_embeddings=false`.

Stage 6: transient-global encoder block parity, then tglobal full encoder and seq2seq parity.

Stage 7: optimized sparse attention kernels and generation scheduling probes.

Initially stub or defer: `output_attentions`, training loss, gradient checkpointing, dropout, and full HF `generate()` controller parity.

## 12. Parity and validation plan

- Unit-test `_split_into_blocks`, `_concatenate_3_blocks`, local mask construction, global block IDs, side relative positions, and block aggregates against PyTorch source helpers.
- Relative bucket parity for bidirectional encoder and unidirectional decoder cases, including distances around exact/log bucket thresholds.
- RMSNorm random tensor parity in fp32/fp16/bf16 with fp32 accumulation; tolerance `1e-5` fp32, `1e-3` fp16/bf16 initially.
- One-block local encoder parity for `S` in `{1, 16, 127, 128, 129, 256}` and masks with left/right padding.
- One-block transient-global parity for `S` in `{8, 16, 17, 128, 129, 4096}` to cover zero/one/multiple global blocks and orphan tokens.
- Decoder block prefill parity with cache off, then decode-step parity with cache on.
- Cross-attention cache parity: first decode step updates projected encoder K/V, second decode step reuses them.
- Full `google/long-t5-local-base` encoder logits or hidden-state parity on short and long inputs.
- End-to-end summarization smoke for local and tglobal checkpoints after graph parity is stable.

## 13. Performance probes

- Encoder-only throughput sweep by `S`: 512, 1024, 2048, 4096, 8192, 16384.
- Local attention kernel sweep over `B`, `A`, `L=128`, dtype, and padding percentage.
- Transient-global sweep over `global_block_size`, `S//G`, and mask sparsity.
- Prefill-only latency/throughput for decoder `T` with fixed encoder output.
- Decode tokens/sec with self-attention cache and cross-attention cache reuse.
- KV cache memory per decoder layer: self K/V grows with `T`; cross K/V fixed with `S`.
- LM head cost for full logits vs last-token logits.
- Dense fallback vs sparse kernel memory/time for small validation shapes.

## 14. Skip/defer list

- Training losses and `labels` shifting beyond decoder-start-token construction.
- Dropout behavior and gradient checkpointing.
- `output_attentions` dense reconstruction; source sparse attention outputs are block-shaped diagnostics.
- Beam search and advanced generation processors.
- Quantization and tensor parallelism.
- Remote-code or nonstandard LongT5 forks.
- Encoder sequence lengths beyond kernel-admitted block/memory limits until sparse kernels are validated.

## 15. Final implementation checklist

- [ ] Parse `LongT5Config`, including `encoder_attention_type`, `local_radius`, `global_block_size`, `feed_forward_proj`, and `tie_word_embeddings`.
- [ ] Load shared embeddings, optional untied LM head, Q/K/V/O, FFN, RMSNorm, and layer-0 relative bias weights.
- [ ] Implement LongT5 RMSNorm with fp32 accumulation.
- [ ] Implement T5 relative position bucket and learned bias gather.
- [ ] Implement local block split/3-block gather/mask semantics.
- [ ] Implement local sparse encoder attention or guarded dense fallback for tiny parity.
- [ ] Implement transient-global block IDs, aggregate sums, side bias, and side K/V path.
- [ ] Implement decoder dense self-attention and cross-attention with `EncoderDecoderCache` semantics.
- [ ] Preserve cross-attention cache reuse via per-layer updated flags.
- [ ] Add gated-GELU FFN lowering and relu FFN fallback.
- [ ] Honor untied LM head configs and the conditional `d_model**-0.5` output scale only when `tie_word_embeddings=true`.
- [ ] Add one-block, full-stack, prefill, decode, and end-to-end parity tests.
- [ ] Benchmark local and transient-global sparse attention separately from decoder decode.
