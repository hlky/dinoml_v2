# XLNet Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: xlnet
Primary runtime target: XLNetModel encoder-style hidden states plus XLNetLMHeadModel masked/permutation LM. Sequence classification, token classification, multiple choice, and QA heads are optional first-follow targets.
Config source: official Hugging Face config.json files listed below.
Source files inspected:
- transformers/src/transformers/models/xlnet/configuration_xlnet.py
- transformers/src/transformers/models/xlnet/modeling_xlnet.py
- transformers/src/transformers/models/xlnet/tokenization_xlnet.py
Any missing files or assumptions: no remote code is required for native XLNet. This report is inference-first and treats dropout/training losses as deferred except where they alter inference head shape.
```

Representative configs inspected:

- [xlnet/xlnet-base-cased config.json](https://huggingface.co/xlnet/xlnet-base-cased/blob/main/config.json)
- [xlnet/xlnet-large-cased config.json](https://huggingface.co/xlnet/xlnet-large-cased/blob/main/config.json)
- [sshleifer/tiny-xlnet-base-cased config.json](https://huggingface.co/sshleifer/tiny-xlnet-base-cased/blob/main/config.json)
- [hf-internal-testing/tiny-random-xlnet config.json](https://huggingface.co/hf-internal-testing/tiny-random-xlnet/blob/main/config.json)
- [Rostlab/prot_xlnet config.json](https://huggingface.co/Rostlab/prot_xlnet/blob/main/config.json)
- [TehranNLP/xlnet-base-cased-mnli config.json](https://huggingface.co/TehranNLP/xlnet-base-cased-mnli/blob/main/config.json)
- [AyushPJ/ai-club-inductions-21-nlp-XLNet config.json](https://huggingface.co/AyushPJ/ai-club-inductions-21-nlp-XLNet/blob/main/config.json)

## 2. High-level architecture

XLNet is a text-only Transformer-XL-derived stack with relative position attention, optional segment relative attention, optional recurrent hidden-state memory, and an unusual two-stream query path used when `target_mapping` is provided.

Dataflow:

```text
tokenizer/data pipeline -> input_ids/token_type_ids/attention_mask/perm_mask/target_mapping
-> token embeddings + optional mask query embedding
-> N x XLNetLayer(relative attention + FFN)
-> hidden states
-> optional LM/classification/token/MC/QA head
```

Runtime-relevant stages:

- CPU/data pipeline: SentencePiece/Unigram-style tokenization, left padding, special token construction, attention mask and optional XLNet-specific `perm_mask` / `target_mapping`.
- Core GPU graph: embeddings, layout transposes to sequence-major `[seq, batch, hidden]`, relative position embedding construction, recurrent hidden memory concat, relative attention, FFN, final transpose back to `[batch, seq_or_num_predict, hidden]`.
- Independently stageable heads: LM logits, sequence summary + classifier, per-token classifier, multiple-choice flatten/reshape, simple QA span projection, full SQuAD-style QA top-k/gather head.

## 3. Important config dimensions

Source defaults from `XLNetConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 32000 | embedding rows and LM head output |
| `d_model` | 1024 | hidden size |
| `n_layer` | 24 | repeated XLNet layers |
| `n_head` | 16 | MHA heads |
| `d_head` | `d_model // n_head` | per-head dim; strict validation requires equality |
| `d_inner` | 4096 | FFN hidden width |
| `ff_activation` | `gelu` | FFN activation; source also accepts `relu`, `silu`, `gelu_new` through `ACT2FN` |
| `attn_type` | `bi` | XLNet bidirectional attention; `uni` builds causal Transformer-XL-style mask |
| `mem_len` | 512 | memory length default in config class, but many released configs override to `null` |
| `reuse_len` | `None` | current-prefix length retained into memory before memory cutoff |
| `use_mems_eval` | `True` | returns memory in eval unless overridden |
| `use_mems_train` | `False` | memory disabled by default in train |
| `bi_data` | `False` | bidirectional relative-position batches during pretraining |
| `clamp_len` | -1 | optional relative distance clamp |
| `same_length` | `False` | alters `uni` causal mask window |
| `summary_type` | `last` | sequence summary pooling default |
| `start_n_top`, `end_n_top` | 5, 5 | full QA inference top-k widths |

Representative checkpoint sweep:

| Model id | Architecture in config | Layers | Hidden | Heads x head dim | FFN | Vocab | Activation | `mem_len` / `reuse_len` | Notes |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `xlnet/xlnet-base-cased` | `XLNetLMHeadModel` | 12 | 768 | 12 x 64 | 3072 | 32000 | gelu | null / null | common base; text-generation task params exist but model card positions XLNet mainly for fine-tuning |
| `xlnet/xlnet-large-cased` | `XLNetLMHeadModel` | 24 | 1024 | 16 x 64 | 4096 | 32000 | gelu | null / null | common large |
| `sshleifer/tiny-xlnet-base-cased` | `XLNetLMHeadModel` | 2 | 4 | 2 x 2 | 2 | 32000 | gelu | null / null | debug-sized dimensions |
| `hf-internal-testing/tiny-random-xlnet` | not specified | 5 | 32 | 4 x 8 | 128 | 1009 | gelu | 10 / 15 | exercises real memory reuse/cutoff behavior |
| `Rostlab/prot_xlnet` | null | 30 | 1024 | 16 x 64 | 4096 | 37 | relu | null / null | protein checkpoint; same body, different activation/vocab/layer count |
| `TehranNLP/xlnet-base-cased-mnli` | `XLNetForSequenceClassification` | 12 | 768 | 12 x 64 | 3072 | 32000 | gelu | null / null | sequence classification head, 3 labels |
| `AyushPJ/ai-club-inductions-21-nlp-XLNet` | `XLNetForQuestionAnsweringSimple` | 12 | 768 | 12 x 64 | 3072 | 32000 | gelu | null / null | simple QA span head |

## 3a. Family variation traps

- `d_head` is not independently flexible in native source: strict config validation requires `d_head == d_model // n_head`.
- Released base/large configs set `mem_len: null`, overriding the config class default of 512. If `use_mems=True` with `mem_len=None`, source returns all previous plus current hidden states, so memory grows with sequence length.
- `hf-internal-testing/tiny-random-xlnet` has `reuse_len=15` and `mem_len=10`; because `reuse_len` is larger than the tiny sequence in many tests, memory behavior can differ from production configs.
- `Rostlab/prot_xlnet` changes `ff_activation` to `relu`, `vocab_size` to 37, and `n_layer` to 30. Do not hard-code gelu/base/large assumptions.
- `attn_type="uni"` is implemented even though XLNet configs normally use `"bi"`. It changes masks and relative-position range.
- `bi_data=True` changes relative positional embedding construction by splitting batch into forward/backward halves. This should be rejected or separately tested unless a checkpoint requires it.
- `perm_mask` and `target_mapping` are not ordinary padding masks. They alter attention topology and trigger two-stream attention.
- Native generation is supported through `GenerationMixin`, but it is XLNet-specific: each generation step appends a dummy token, builds a dense permutation mask, predicts only the dummy position, and reuses hidden-state `mems`, not KV tensors.
- Internal layout is sequence-major `[seq, batch, hidden]`; all public inputs/outputs are batch-major. Layout passes must guard the boundary transposes and XLNet-specific einsums.
- Relative attention projection weights are raw parameters shaped `[d_model, n_head, d_head]`, not `nn.Linear` modules. Weight import/lowering must preserve or explicitly flatten them.
- `XLNetSequenceSummary(summary_type="attn")` raises `NotImplementedError`; configs using it should be rejected.

## 4. Operator coverage checklist

Tensor/layout ops:

- `transpose(0, 1)` and `permute(1, 2, 0)` input normalization.
- `contiguous`, `reshape`, `view`, `expand`, `squeeze`, `unsqueeze`, `cat`, `split`, `gather`, `index_select`, `topk`.
- `one_hot` for segment matrix construction.
- Batch-choice flatten and output reshape for multiple choice.

Neural network primitives:

- Embedding lookup: token embedding `[vocab_size, d_model]`.
- Parameter-backed linear/einsum projections:
  - Q/K/V/R: `[d_model] -> [n_head, d_head]`, no bias.
  - O: `[n_head, d_head] -> [d_model]`, no bias.
- FFN: `Linear(d_model -> d_inner)`, activation, `Linear(d_inner -> d_model)`.
- LayerNorm after attention residual and after FFN residual, epsilon `1e-12`.
- Optional heads:
  - LM: tied-weight `Linear(d_model -> vocab_size)` with bias.
  - Sequence classification: sequence summary projection, tanh, classifier.
  - Token classification: `Linear(d_model -> num_labels)`.
  - Simple QA: `Linear(d_model -> 2)` then split start/end.
  - Full QA: start projection, top-k, gather, end projection over expanded `[batch, seq, start_n_top, hidden]`, answerability classifier.

Attention primitives:

- Dense MHA with relative content, position, and optional segment score terms.
- `einsum` score forms:
  - content score: `ibnd,jbnd->bnij`
  - position score plus relative shift
  - segment score: `ibnd,snd->ibns` then `ijbs,ibns->bnij`
  - attention value: `bnij,jbnd->ibnd`
- Softmax over key dimension, inference dropout as no-op.
- Additive mask using `-1e30` for fp32/bf16-style paths and `-65500` for fp16.

Position/relative-bias ops:

- Sin/cos relative positional embeddings built from dynamic `qlen`, `klen`.
- Relative shift `rel_shift_bnij` implemented via reshape, slice/drop, reshape, and `index_select`.
- Learned per-head biases `r_w_bias`, `r_r_bias`, `r_s_bias`; learned `seg_embed[2, n_head, d_head]`.

Generation/cache ops:

- Hidden-state memory list length `n_layer`, each tensor `[mem_seq, batch, d_model]`.
- Memory concat before K/V projection; returned memory is detached current hidden states, not projected KV cache.
- Beam cache reorder: `index_select(dim=1, beam_idx)` for each memory tensor.
- Generation input preparation builds dense `perm_mask` `[batch, seq, seq]` and `target_mapping` `[batch, 1, seq]`.

Preprocessing-coupled ops:

- Tokenizer uses left padding and special-token template:
  - single: `$A <sep> <cls>`
  - pair: `$A <sep> $B <sep> <cls>`
- `token_type_ids` feed segment relative attention through a `[qlen, klen, batch, 2]` one-hot matrix.
- `attention_mask` is inverted to `input_mask = 1 - attention_mask`; `input_mask` and `attention_mask` are mutually exclusive.

## 5. Layer/block breakdown

Core shape convention inside `XLNetModel`: public `[B, S]` or `[B, S, H]` inputs are transposed to `[S, B]` / `[S, B, H]`. Let `M = memory length`, `Q = current query length`, `K = M + Q`, `H = d_model`, `N = n_head`, `D = d_head`.

Embedding and masks:

```text
input_ids[B,S] -> transpose -> input_ids[Q,B]
word_embedding -> h[Q,B,H]
if target_mapping[B,T,Q]:
  target_mapping -> [T,Q,B]
  mask_emb[1,1,H].expand(T,B,H) -> g[T,B,H]
else:
  g = None
token_type_ids[Q,B] + memory zeros[M,B] -> seg_mat[Q,K,B,2]
relative_positional_encoding(Q,K) -> r[R,B,H], where R depends on attn_type/bi_data
```

XLNet layer, repeated `n_layer` times:

```text
cat = concat(mems[i][M,B,H], h[Q,B,H]) or h
k = einsum(cat, W_k[d_model,n_head,d_head]) -> [K,B,N,D]
v = einsum(cat, W_v[d_model,n_head,d_head]) -> [K,B,N,D]
r_head = einsum(pos_emb, W_r[d_model,n_head,d_head]) -> [R,B,N,D]

h stream:
  q_h = einsum(h, W_q) -> [Q,B,N,D]
  attn_h = relative_attention(q_h, k, v, r_head, seg_mat, non_tgt_mask)
  h = LayerNorm(h + einsum(attn_h, W_o))
  h = LayerNorm(h + FFN(h))

g stream, only with target_mapping:
  q_g = einsum(g, W_q) -> [T,B,N,D]
  q_g_mapped = einsum("mbnd,mlb->lbnd", q_g, target_mapping) -> [Q,B,N,D]
  attn_g = relative_attention(q_g_mapped, k, v, r_head, seg_mat, attn_mask)
  attn_g = einsum("lbnd,mlb->mbnd", attn_g, target_mapping) -> [T,B,N,D]
  g = LayerNorm(g + einsum(attn_g, W_o))
  g = LayerNorm(g + FFN(g))
```

The output is `g` when present, otherwise `h`, then transposed back to `[B, T_or_Q, H]`.

## 6. Attention requirements

Attention type:

- Native XLNet uses self-attention only.
- `attn_type="bi"` starts with no causal mask; `perm_mask` and padding can still mask arbitrary query/key pairs.
- `attn_type="uni"` creates a causal Transformer-XL-style mask, with optional `same_length` behavior.
- MHA only: `n_head == n_key_value_heads`, no GQA/MQA.

Score math:

```text
AC = einsum(q + r_w_bias, k_content)
BD = rel_shift_bnij(einsum(q + r_r_bias, k_position))
EF = 0 or segment_einsum(q + r_s_bias, seg_embed, seg_mat)
score = (AC + BD + EF) / sqrt(d_head)
score += large_negative_mask
prob = softmax(score, dim=keys)
out = einsum(prob, v)
```

Masking:

- `perm_mask[B,Q,Q]` is transposed to `[Q,Q,B]`; `1` means cannot attend.
- Padding mask is `[Q,B]`, expanded across query positions.
- Memory positions are prepended with zeros in the data mask, so all memory tokens can be attended unless causal mask blocks them.
- `non_tgt_mask` masks self-attention for the h stream by adding a negative identity to the target positions when a data/causal mask exists.

Memory/cache:

- Cache tensors are hidden states, not K/V. Each layer consumes `mems[i]` by concatenating it with current h before K/V projection.
- Returned `new_mems[i]` is produced before layer `i` runs, from that layer's input h. Shape is `[new_mem_len, B, H]`.
- If `reuse_len > 0`, only the first `reuse_len` current states are candidates for memory.
- If `mem_len is None or 0`, cutoff is `0`, so returned memory contains the entire previous plus retained current sequence.
- If `mem_len > 0`, returned memory keeps the last `mem_len` entries after concatenation.
- Beam reorder is per-layer `mem.index_select(1, beam_idx)`.

FlashAttention/SDPA compatibility:

- A standard SDPA backend does not directly cover XLNet because scores include relative-shifted position terms and segment-relative terms.
- A staged implementation can use explicit score materialization first. A later fused kernel would need direct support for AC+BD+EF, relative shift, dynamic memory prefix, and dense arbitrary permutation masks.
- Generation decode is not standard one-token KV decode; source recomputes projections for `offset=2` plus dummy token and memory prefix each step.

## 7. Position encoding and custom math

Relative position embedding source equivalent:

```python
def xlnet_relative_positional_encoding(qlen, klen, d_model, attn_type, bi_data, clamp_len, bsz):
    inv_freq = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    if attn_type == "bi":
        beg, end = klen, -qlen
    elif attn_type == "uni":
        beg, end = klen, -1
    else:
        raise ValueError
    pos_seq = torch.arange(beg, end, -1.0).float()
    if clamp_len > 0:
        pos_seq = pos_seq.clamp(-clamp_len, clamp_len)
    sinusoid = torch.einsum("i,d->id", pos_seq, inv_freq)
    pos_emb = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
    return pos_emb[:, None, :].expand(-1, bsz, -1)
```

Relative shift source equivalent:

```python
def rel_shift_bnij(x, klen):
    # x: [batch, heads, query, relative_key]
    b, n, q, r = x.shape
    x = x.reshape(b, n, r, q)
    x = x[:, :, 1:, :]
    x = x.reshape(b, n, q, r - 1)
    return x.index_select(3, torch.arange(klen, device=x.device))
```

Precompute opportunities:

- `inv_freq` and the projection weight transforms are static per config.
- Position embeddings depend on `qlen`, `klen`, `attn_type`, `bi_data`, `clamp_len`, batch size, and device. They can be cached by `(qlen,klen,batch)` for fixed-shape runs.
- Segment matrix depends on runtime `token_type_ids` and memory length.

## 8. Preprocessing and input packing

Tokenizer/runtime coupling:

- XLNet tokenizer is Unigram-backed, uses left padding, and emits `[tokens] <sep> <cls>` for single sequences.
- Segment IDs matter when `token_type_ids` is provided because the model constructs relative segment attention. For paired inputs, A uses segment 0, B uses segment 1, and `<cls>` uses segment 2 at tokenizer level; the model itself converts unequal segment IDs to same/different one-hot classes.
- `attention_mask` follows HF convention where 1 means real token; model converts it to `input_mask` where 1 means masked padding.
- `perm_mask` and `target_mapping` are model inputs, not tokenizer defaults. They are required for masked-token prediction with two-stream attention and for native generation.

CPU/data-pipeline first:

- Text normalization/tokenization, special token insertion, left padding, `attention_mask`, and normal `token_type_ids`.
- For LM/generation parity, generation controller must create the dummy token, dense permutation mask, target mapping, and memory handoff exactly as `prepare_inputs_for_generation`.

GPU/runtime graph:

- Embedding lookup, optional mask query embedding expansion, segment one-hot construction if token type IDs are passed, attention masks, relative position embeddings, model body, and head.

## 9. Graph rewrite / lowering opportunities

### Rewrite: XLNet projection parameters to GEMM

Source pattern:

```text
einsum("ibh,hnd->ibnd", x, W)
```

Replacement:

```text
x[seq,batch,H] -> flatten [seq*batch,H]
GEMM with W_flat[H, N*D] -> reshape [seq,batch,N,D]
```

Preconditions:

- `W` is contiguous or copied to a packed `[H, N*D]` logical constant.
- `d_model == n_head * d_head`.
- Preserve no-bias behavior for Q/K/V/R projections.

Failure cases:

- Do not apply if a future checkpoint violates strict `d_head` validation through remote code.

Parity test sketch:

- Compare all five attention projection einsums against flattened GEMM for random `[seq,batch,H]`.

### Rewrite: XLNet output projection to GEMM

Source pattern:

```text
einsum("ibnd,hnd->ibh", attn_vec, W_o)
```

Replacement:

```text
attn_vec[seq,batch,N,D] -> flatten [seq*batch,N*D]
GEMM with W_o_flat[(N*D),H] -> [seq*batch,H] -> [seq,batch,H]
```

Preconditions:

- Weight transform maps `W_o[h,n,d]` to columns `h` over flattened `(n,d)`.
- Preserve residual + dropout(no-op inference) + LayerNorm order.

### Rewrite: target_mapping one-hot gather/scatter

Source pattern:

```text
q_head_g_mapped = einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
attn_vec_g = einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
```

Replacement:

```text
if target_mapping is one-hot per predicted token:
  gather selected query positions before attention
  scatter/gather back to prediction slots after attention
else:
  keep dense einsum
```

Preconditions:

- Runtime verifies each `[batch,predict,:]` row is one-hot or routes to dense path.
- Selection indices are in range and shape is `[B,T,Q]`.

Failure cases:

- Pretraining or custom callers may pass soft target mappings; those require dense matmul semantics.

### Rewrite: segment matrix avoid full one-hot

Source pattern:

```text
seg_mat = one_hot(token_type_ids[:,None] != cat_ids[None,:], 2)
ef = einsum("ijbs,ibns->bnij", seg_mat, ef)
```

Replacement:

```text
compute same/different segment score by selecting one of two learned segment embeddings
```

Preconditions:

- Token type IDs are integer and only equality/inequality matters.
- Preserve memory prefix segment ID zero behavior.

Failure cases:

- None for native source, but keep fallback until parity over mixed segment IDs is tested.

### Rewrite: LM head tied embedding

Source pattern:

```text
lm_loss(hidden_states) with weight tied to transformer.word_embedding.weight
```

Replacement:

```text
GEMM hidden @ embedding.T + lm_bias
```

Preconditions:

- Preserve logical alias between embedding and LM weight.
- Support `logits_to_keep` slicing before logits GEMM where possible.

Failure cases:

- Untied external checkpoints would need explicit detection, though native class declares tied keys.

## 10. Kernel fusion candidates

Highest priority:

- Relative attention score kernel: AC + BD relative shift + optional EF + mask + softmax dominates XLNet and is not a standard SDPA call.
- Projection GEMMs for Q/K/V/R/O: all XLNet layers use parameter-backed einsums that should canonicalize to GEMM before performance work.
- LayerNorm residual blocks: attention output + residual + LayerNorm and FFN output + residual + LayerNorm are repeated twice per layer.
- Memory concat plus K/V projection: avoid physically concatenating memory and current hidden states when possible.

Medium priority:

- FFN `Linear -> gelu/relu -> Linear` with activation variants. `Rostlab/prot_xlnet` makes relu coverage real.
- Target-mapping gather/scatter fast path for generation and masked-token prediction.
- Segment-relative attention without materializing `[Q,K,B,2]` one-hot.
- Last-token/target-only LM logits for generation and masked-token use.

Lower priority:

- Full QA inference head top-k/gather/end-logit pipeline.
- Multiple-choice flatten/reshape convenience fusion.
- `bi_data=True` position embedding specialization, unless a target checkpoint requires it.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse `XLNetConfig`, validate `d_model % n_head == 0` and `d_head == d_model // n_head`.
- Load embedding, raw attention parameters, FFN, LayerNorm, and selected head weights.
- First accepted configs: `attn_type="bi"`, `bi_data=False`, inference dropout disabled, `summary_type!="attn"`.

Stage 2: single-layer and encoder body parity without memory.

- Implement sequence-major internal lowering or guarded boundary transposes.
- Materialize dense relative attention scores explicitly.
- Validate `XLNetModel` hidden states for base/tiny shapes without `target_mapping`.

Stage 3: target mapping and two-stream attention.

- Add `mask_emb`, `g` stream, dense target-mapping einsums.
- Validate masked-token LM examples where output length is `num_predict`.

Stage 4: memory reuse.

- Add `mems` ABI as `n_layer` hidden-state tensors `[mem_seq,B,H]`.
- Implement `cache_mem`, `mem_len=None` growing-memory behavior, `mem_len>0` cutoff, and beam reorder.

Stage 5: heads.

- LM head with tied embedding and `logits_to_keep`.
- Sequence classification and token classification.
- Simple QA head.
- Defer full SQuAD top-k QA until top-k/gather paths are ready.

Stage 6: optimization.

- Rewrite projection einsums to GEMM.
- Add fused relative attention or score/softmax kernels.
- Add target-only logits and target-mapping one-hot fast paths.

## 12. Parity and validation plan

- Config validation tests:
  - base, large, tiny-random, and prot configs.
  - reject `summary_type="attn"` for classification heads.
- Custom math tests:
  - `relative_positional_encoding` for `bi`, `uni`, `clamp_len`, and `bi_data=False`.
  - `rel_shift_bnij` against Torch source for varying `qlen`, `klen`, and `mlen`.
  - segment matrix same/different behavior with memory prefix.
- Attention tests:
  - single `XLNetRelativeAttention` no memory/no segment.
  - with padding mask and `perm_mask`.
  - with `target_mapping` two-stream path.
  - with memory prefix.
- Model tests:
  - one-layer random config parity.
  - full tiny-random model parity with `mem_len=10`, `reuse_len=15`.
  - base/large smoke parity on short sequences.
- Head tests:
  - LM logits with and without `target_mapping`.
  - `prepare_inputs_for_generation` parity for first and later iteration.
  - sequence classification summary types `last`, `first`, `mean`, `cls_index`.
  - token classification and simple QA shapes.
- Suggested tolerances:
  - fp32 explicit-score path: `rtol=1e-4`, `atol=1e-5`.
  - fp16/bf16: start with `rtol=5e-2`, `atol=5e-2` for full model, then tighten per kernel after accumulation policy is fixed.

## 13. Performance probes

- Encoder-only throughput over batch sizes 1, 4, 16 and sequence lengths 32, 128, 512.
- Relative attention breakdown: projection GEMMs, score materialization, softmax, value matmul, output projection.
- Memory mode sweep:
  - `mem_len=None` growing memory.
  - fixed `mem_len` such as 10, 128, 512.
  - varying `reuse_len`.
- Target mapping sweep: `num_predict=1` generation-like, small masked-token batches, dense target mapping fallback.
- Segment attention overhead: token type IDs absent vs present.
- LM logits cost: full sequence logits vs `logits_to_keep=1`/target-only.
- QA full head overhead: start top-k/end top-k with default 5 x 5.
- Activation variant benchmark: gelu vs relu for `Rostlab/prot_xlnet`.

## 14. Skip/defer list

- Training losses, dropout randomness, and gradient checkpointing.
- `bi_data=True` pretraining pipeline unless a target checkpoint requires it.
- `attn_type="uni"` optimized path; implement correctness only after `bi` path.
- Full QA SQuAD 2.0 beam-style head can follow simple QA.
- Beam search controller and sampling processors; only implement cache reorder and generation input prep needed for parity harnesses first.
- Remote-code or non-native XLNet variants.
- Tensor-parallel and distributed inference.
- Fused relative attention kernel; start with explicit score tensors.

## 15. Final implementation checklist

- [ ] Parse `XLNetConfig` and enforce native strict dimension rules.
- [ ] Load XLNet raw attention weights `[d_model,n_head,d_head]` and define flatten transforms for GEMM lowering.
- [ ] Implement embedding, boundary transposes, and sequence-major model body.
- [ ] Implement relative positional embedding and `rel_shift_bnij`.
- [ ] Implement relative attention AC/BD/EF score path with fp16/fp32 mask constants.
- [ ] Implement padding mask, `perm_mask`, `target_mapping`, and segment matrix construction.
- [ ] Implement two-stream attention and `mask_emb`.
- [ ] Implement hidden-state `mems` ABI, `cache_mem`, and beam reorder.
- [ ] Implement FFN with gelu/relu/silu/gelu_new admission.
- [ ] Implement LM head with tied embedding alias and `logits_to_keep`.
- [ ] Implement sequence classification, token classification, multiple choice, and simple QA heads.
- [ ] Defer or separately stage full QA top-k head.
- [ ] Add projection-einsum-to-GEMM rewrites with weight transform tests.
- [ ] Add one-layer, tiny-random, base, memory, target-mapping, and generation-prep parity tests.
- [ ] Benchmark explicit attention, memory mode, target-only logits, and segment attention overhead.
