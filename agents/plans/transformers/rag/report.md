# Transformers RAG Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/rag-token-base, facebook/rag-token-nq, facebook/rag-sequence-base, facebook/rag-sequence-nq
Config source: raw Hugging Face Hub config.json files; see config_sweep.md
Source files inspected:
- transformers/src/transformers/models/rag/configuration_rag.py
- transformers/src/transformers/models/rag/modeling_rag.py
- transformers/src/transformers/models/rag/retrieval_rag.py
- transformers/src/transformers/models/rag/tokenization_rag.py
- transformers/src/transformers/models/rag/__init__.py
Any missing files or assumptions:
- RAG delegates neural bodies to AutoModel DPR question encoder and AutoModelForSeq2SeqLM generator. This report owns the RAG wrapper/retrieval/scoring/marginalization ABI, not full DPR or BART operator audits.
- Representative configs were public. No gated model repo was encountered.
- Generation config files were not present for the four checked repos.
```

Source URLs at the pinned commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/rag/modeling_rag.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/rag/retrieval_rag.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/rag/configuration_rag.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/rag/tokenization_rag.py

## 2. High-level architecture

RAG is a retrieval-augmented encoder-decoder generation wrapper:

```text
question tokens -> DPR question encoder -> CPU/FAISS or datasets retrieval
  -> retrieved title/text + original question -> generator tokenizer contexts
  -> BART/T5-style seq2seq generator encoder -> generator decoder
  -> per-document logits -> RAG-token or RAG-sequence marginalization/scoring
```

First useful DinoML target: `RagTokenForGeneration` or `RagSequenceForGeneration` with retrieval supplied externally as `context_input_ids`, `context_attention_mask`, `retrieved_doc_embeds`, and/or `doc_scores`. That lets DinoML validate the neural and RAG math without owning FAISS, Hugging Face Datasets, Python tokenization, or pickle-based legacy index loading.

Stage decomposition:

| Stage | Owner for first integration | Cacheability | Runtime tensors |
|---|---|---|---|
| Question tokenization | CPU/data pipeline | Per prompt | `input_ids [B,Q]`, `attention_mask [B,Q]` |
| Question encoder | Compose DPR audit | Query embedding can be cached per prompt | `question_hidden_states [B,768]` |
| Retrieval/index search | External service/CPU pipeline first | Retrieved docs and embeddings cacheable | `retrieved_doc_embeds [B,n_docs,768]`, ids/docs |
| Context construction/tokenization | CPU/data pipeline first | Context ids cacheable until docs/question change | `context_input_ids [B*n_docs,Lctx]`, `context_attention_mask [B*n_docs,Lctx]` |
| Generator encoder | Compose BART/T5 audit | Encoder outputs cacheable across decode | `last_hidden_state [B*n_docs,Lctx,Hgen]` |
| Generator decoder | DinoML via generator family | KV cache per beam/document | logits `[B*n_docs,T,V]` before marginalization |
| RAG scoring/marginalization | RAG wrapper-owned | Cheap deterministic post-op | `doc_scores [B,n_docs]`, marginalized logits/loss |

## 3. Important config dimensions

| Field | Source default | Representative value | DinoML relevance |
|---|---:|---:|---|
| `n_docs` | 5 | 5 | Multiplies generator batch: `B*n_docs`. |
| `max_combined_length` | 300 | 300 | Context sequence length for generator encoder after retriever tokenization. |
| `retrieval_vector_size` | 768 | 768 | Required width for query/document dot-product scores. |
| `retrieval_batch_size` | 8 | configs/default | CPU index search batching, not GPU graph. |
| `dataset` / `dataset_split` | `wiki_dpr` / `train` | same | External retrieval source. |
| `index_name` | `compressed` | `exact` or `legacy` | Selects FAISS/datasets path. |
| `do_marginalize` | `False` | mostly false/null | If true, output logits are logsumexp over docs. |
| `do_deduplication` | `True` | true/null | Generation-controller behavior for RAG-sequence. |
| Question encoder | required subconfig | DPR base | Separate encoder audit, output width 768. |
| Generator | required subconfig | BART large | Separate seq2seq audit, hidden 1024, vocab 50265. |

Representative checkpoint sweep:

| Model id | Variant | Retrieval | Question encoder | Generator | Operator-significant difference |
|---|---|---|---|---|---|
| `facebook/rag-token-base` | token marginalization base | `exact` wiki_dpr | DPR base | BART large | Base token path; exact HF dataset index. |
| `facebook/rag-token-nq` | token marginalization NQ | `legacy` wiki_dpr | DPR base | BART large | Legacy FAISS/pickle retrieval; top-level token ids set. |
| `facebook/rag-sequence-base` | sequence marginalization base | `exact` wiki_dpr | DPR base | BART large | Sequence rescoring path instead of per-token marginalized generation. |
| `facebook/rag-sequence-nq` | sequence marginalization NQ | `legacy` wiki_dpr | DPR base | BART large | Legacy retrieval plus sequence NLL ranking. |

## 3a. Family variation traps

- `RagConfig` cannot be instantiated without both `question_encoder` and `generator` subconfigs.
- RAG source can wrap any AutoModel question encoder and any AutoModelForSeq2SeqLM generator. First DinoML admission should allowlist DPR+BART from the official configs and reject un-audited generator families.
- The retrieval path is not a tensor op: it crosses to CPU NumPy float32, FAISS/HF Datasets, Python strings, and tokenizer calls.
- `index_name="legacy"` requires FAISS plus unpickling, guarded by `TRUST_REMOTE_CODE=True`; this should be external to DinoML runtime admission.
- Contexts are flattened as `B*n_docs`; generation cache reorder must preserve the hidden document dimension.
- `RagTokenForGeneration` and `RagSequenceForGeneration` have different marginalization/scoring behavior.
- `RagSequenceForGeneration.generate` loops over batch items, calls generator beams per document, deduplicates generated sequences with Python dict logic, then reruns RAG NLL scoring and top-k. Treat as generation controller, not one static graph.
- No vision/audio layout issues. The layout-sensitive parts are sequence reshapes/repeats and cache beam/document ordering.

## 4. Operator coverage checklist

RAG-wrapper required ops:

- Tensor/layout ops: `reshape/view`, `unsqueeze`, `squeeze`, `transpose(1,2)`, `repeat_interleave`, `expand`, `index_select`, `cat`, `stack`, dynamic pad/copy for `_cat_and_pad`.
- Neural primitives owned by wrapper: `bmm([B,1,768] x [B,768,n_docs]) -> [B,n_docs]` for `doc_scores`.
- Reductions/probability: `log_softmax` over docs and vocab, `logsumexp` over docs, `gather` by target token ids, `sum` over sequence, `masked_fill` on pad tokens.
- Selection/controller ops: `topk` for sequence candidate selection, Python deduplication for generated sequences, optional beam expansion.
- Generation/cache ops: delegate to generator; wrapper-specific cache reorder reshapes `[B*beam*n_docs,...] -> [B*beam,n_docs,...]`, indexes beam dim, then flattens.
- Retrieval/preprocessing coupled ops: external `get_top_docs`, document title/text string concatenation with separators, generator tokenizer padding/truncation to `max_combined_length`.

Delegated neural bodies:

- DPR question encoder: BERT-like encoder with token/type/position embeddings, MHA, GELU FFN, LayerNorm, pooling. Exact coverage belongs to a DPR audit.
- BART generator: encoder-decoder Transformer with learned positions, encoder self-attention, decoder causal self-attention, cross-attention, FFN, LayerNorm, LM head. Exact coverage belongs to a BART audit.

Parameter sharing:

- RAG wrapper exposes generator input/output embeddings through `get_input_embeddings` / `get_output_embeddings`; tied embedding/LM-head behavior belongs to the generator.
- RAG adds no large trainable parameters beyond delegated modules.

## 5. Layer/block breakdown

Forward path for `RagModel`:

```text
if retrieval is needed:
  question_hidden = question_encoder(input_ids, attention_mask)[0]      # [B,768]
  retrieval_query = detach(question_hidden).cpu().float().numpy()
  retriever returns:
    context_input_ids       # [B*n_docs,Lctx]
    context_attention_mask  # [B*n_docs,Lctx]
    retrieved_doc_embeds    # [B,n_docs,768]
  doc_scores = bmm(question_hidden[:,None,:], retrieved_doc_embeds.transpose(1,2)).squeeze(1)
else:
  caller must provide context_input_ids, context_attention_mask, doc_scores

decoder_input_ids = repeat_interleave(decoder_input_ids, n_docs, dim=0) if present
decoder_attention_mask = repeat_interleave(decoder_attention_mask, n_docs, dim=0) if present
gen_outputs = generator(
  input_ids=context_input_ids,
  attention_mask=context_attention_mask,
  encoder_outputs=optional precomputed generator encoder outputs,
  decoder_input_ids=decoder_input_ids,
  past_key_values=past_key_values,
)
return gen logits [B*n_docs,T,V], doc_scores [B,n_docs], generator caches/hidden states
```

`RagTokenForGeneration`:

```text
logits = gen_logits
if do_marginalize:
  seq_logprobs = log_softmax(logits, dim=-1).view(B,n_docs,T,V)
  doc_logprobs = log_softmax(doc_scores, dim=1)
  logits = logsumexp(seq_logprobs + doc_logprobs[:, :, None, None], dim=1)  # [B,T,V]
```

`RagSequenceForGeneration` loss/scoring:

```text
seq_logprobs = log_softmax(seq_logits, dim=-1).view(B,n_docs,T,V)
doc_logprobs = log_softmax(doc_scores, dim=1)[:, :, None, None]
rag_logprobs = cat([token0, token1 + doc_logprobs, token2_plus], dim=2)
ll = gather(rag_logprobs, target).mask_pad().sum(tokens).logsumexp(docs)
```

## 6. Attention requirements

RAG wrapper itself implements no attention kernel. It requires:

- Noncausal self-attention in the DPR question encoder.
- Seq2seq generator encoder self-attention, decoder causal self-attention, and decoder cross-attention from BART/T5-style generator.
- Autoregressive KV cache only in the generator decoder. RAG passes through `past_key_values` and implements document-aware beam cache reorder.

Wrapper-specific cache contract:

- Generator runs with effective batch `B*num_beams*n_docs` during RAG-token generation.
- Reorder reshapes each cache tensor to `[-1, n_docs, ...]`, applies `index_select(0, beam_idx)`, then flattens back.
- For `EncoderDecoderCache`, both self-attention and cross-attention key/value layers are reordered.
- Cached keys/values are generator-owned; RAG only changes the batch/beam/document ordering.

Masking:

- Question encoder uses caller `attention_mask [B,Q]`.
- Generator encoder uses `context_attention_mask [B*n_docs,Lctx]`.
- Decoder causal masking is delegated to generator.
- RAG wrapper adds no custom attention bias.

## 7. Position encoding and custom math

No RAG-specific position encoding exists. Position embeddings/RoPE/relative bias, if any, are delegated to the configured question encoder and generator.

Wrapper math to preserve:

```python
def rag_doc_scores(question_hidden, retrieved_doc_embeds):
    # question_hidden: [B, D], retrieved_doc_embeds: [B, n_docs, D]
    return torch.bmm(
        question_hidden.unsqueeze(1),
        retrieved_doc_embeds.transpose(1, 2),
    ).squeeze(1)

def rag_token_marginalize(seq_logits, doc_scores, n_docs):
    # seq_logits: [B*n_docs, T, V]
    seq_lp = torch.log_softmax(seq_logits, dim=-1).view(-1, n_docs, seq_logits.size(1), seq_logits.size(2))
    doc_lp = torch.log_softmax(doc_scores, dim=1)
    return torch.logsumexp(seq_lp + doc_lp[:, :, None, None], dim=1)
```

All retrieval-document embeddings are treated as float32 for scoring in the inspected source path.

## 8. Preprocessing and input packing

RAG has strong preprocessing coupling:

- `RagTokenizer` owns two tokenizers: `question_encoder_tokenizer` for input questions and `generator_tokenizer` for targets/output decoding.
- Retriever decodes question input ids with the question tokenizer, retrieves docs, then constructs strings as `prefix + title + title_sep + text + doc_sep + input_string`.
- Generator tokenizer emits `context_input_ids` and `context_attention_mask` with `padding="max_length"`, `truncation=True`, and `max_length=max_combined_length`.
- `context_input_ids` are ordered row-major by batch then doc: for each query `i`, docs `j=0..n_docs-1`.
- Retrieval returns `doc_ids [B,n_docs]` and `retrieved_doc_embeds [B,n_docs,D]`.

External boundaries for first DinoML target:

- CPU/data pipeline owns tokenization, string concatenation, FAISS/HF Datasets search, and legacy pickle/index loading.
- GPU/runtime receives already packed `context_input_ids`, masks, and either `doc_scores` or `retrieved_doc_embeds` plus `question_hidden`.
- The retriever can be represented as an external service ABI, not an op.

## 9. Graph rewrite / lowering opportunities

### Rewrite: document score BMM to row-dot GEMM

Source pattern:

```text
bmm(question_hidden[:,None,:], retrieved_doc_embeds.transpose(1,2)).squeeze(1)
```

Replacement:

```text
batched row-dot over D=768 -> doc_scores [B,n_docs]
```

Preconditions:

- `question_hidden` rank 2 `[B,D]`.
- `retrieved_doc_embeds` rank 3 `[B,n_docs,D]`.
- `D == retrieval_vector_size`.
- Dense contiguous or known-stride tensors.

Failure cases:

- Dynamic or mismatched retrieval width.
- Non-float doc embeddings.
- External retriever missing embeddings and caller only supplies `doc_scores`; then skip this rewrite.

Parity test sketch:

- Compare rewritten score op to PyTorch `bmm` for random `B`, `n_docs`, `D=768`.

### Rewrite: RAG-token marginalization fused reduction

Source pattern:

```text
log_softmax(seq_logits, -1).view(B,n_docs,T,V)
+ log_softmax(doc_scores, 1)[:, :, None, None]
-> logsumexp(dim=1)
```

Replacement:

```text
fused doc-marginalized logprob kernel over n_docs
```

Preconditions:

- `n_docs` small static or bounded, usually 5.
- Vocab axis last and contiguous.
- `seq_logits` ordered as flattened `[B*n_docs,T,V]` with doc-major chunks.

Failure cases:

- Sequence variant NLL uses different placement of doc scores.
- Caller requests unmarginalized logits.

Parity test sketch:

- Random logits/doc scores, compare fp32 to PyTorch within `1e-5`; fp16/bf16 with relaxed tolerance after upcast policy is fixed.

### Rewrite: context beam expansion as metadata view plus guarded materialization

Source pattern:

```text
reshape [B,n_docs,...] -> expand num_beams -> reshape [B*num_beams*n_docs,...]
```

Replacement:

```text
view/stride metadata if generator accepts broadcasted encoder outputs, else explicit copy
```

Preconditions:

- Consumer can handle repeated encoder output without mutation.
- Beam count known for generation invocation.

Failure cases:

- Generator backend requires contiguous encoder outputs.
- Cache reorder assumes flattened physical layout.

## 10. Kernel fusion candidates

Highest priority:

- Generator-family kernels from BART/T5 audit: encoder/decoder attention, cross-attention with cache, FFN/GELU, LayerNorm, logits.
- RAG-token marginalization fused reduction over docs for `[B,n_docs,T,V]`; it can otherwise become a large memory-bound logprob tensor.
- Document score row-dot for `D=768`, especially when retrieval embeddings are already on GPU.

Medium priority:

- Beam/document cache reorder with reshape + gather fused for generator cache tensors.
- Sequence NLL scoring kernel for RAG-sequence candidate reranking.
- Last-token-only RAG-token marginalization during decode, avoiding full `[T,V]` recomputation where generation only needs next-token scores.

Lower priority:

- `_cat_and_pad` dynamic sequence padding for final generated outputs; likely controller-side.
- FAISS/index acceleration inside DinoML; keep external until neural parity and ABI are proven.

## 11. Runtime staging plan

1. Parse `RagConfig`; enforce allowlist `question_encoder=dpr`, `generator=bart` for first official checkpoints.
2. Load delegated DPR and BART weights through their separate family loaders; preserve generator embedding/head tying.
3. Add RAG wrapper ABI that accepts external `context_input_ids`, `context_attention_mask`, and `doc_scores`.
4. Validate generator forward for flattened `B*n_docs` contexts and decoder ids.
5. Add optional wrapper-owned `doc_scores` computation from `question_hidden` plus `retrieved_doc_embeds`.
6. Add RAG-token marginalization for logits and decode-time next-token scores.
7. Add document-aware cache reorder and beam expansion for RAG-token generation.
8. Add RAG-sequence NLL/reranking as a controller path; keep Python dedup/stitch outside compiled graph initially.
9. Integrate an external retriever service contract; only later consider in-process FAISS/Datasets ownership.

Can stub initially:

- Full retrieval and tokenization by requiring precomputed contexts/scores.
- Training context encoder path.
- RAG-sequence thorough decoding loop.
- Legacy index loading.

## 12. Parity and validation plan

- Config parsing: round-trip the four representative configs and verify enforced allowlist/rejections.
- Doc score unit test: random `question_hidden [B,768]`, `retrieved_doc_embeds [B,5,768]` versus PyTorch `bmm`.
- RAG-token marginalization: random logits `[B*5,T,V]`, doc scores `[B,5]`; compare to source formula.
- Cache reorder: synthetic per-layer cache tensors with distinguishable batch/doc values; compare document-aware reorder.
- Forward parity with retrieval stubbed: run HF RAG with provided `context_input_ids`, `context_attention_mask`, and `doc_scores`, compare DinoML generator logits and optional marginalized logits.
- End-to-end controller parity later: fixed retriever outputs, greedy generation for `rag-token-nq`.
- RAG-sequence scoring: compare source `get_nll` on candidate labels, including pad masking and `exclude_bos_score`.

Suggested tolerances:

- fp32 wrapper math: `rtol=1e-5`, `atol=1e-6`.
- fp16/bf16 generator paths: follow delegated generator tolerances; wrapper reductions should upcast if parity requires it.

## 13. Performance probes

- Retriever latency: external FAISS/Datasets query throughput by `B`, `n_docs`, and index type.
- Context tokenization throughput: docs/sec for title/text/question construction and generator tokenizer.
- Question encoder throughput: `B x Q` sweep.
- Generator encoder throughput: `(B*n_docs) x Lctx` sweep, with `n_docs=1,5,10`.
- Decode throughput: tokens/sec for RAG-token with `B`, beams, `n_docs`, and cache size.
- RAG marginalization bandwidth: `[B,n_docs,T,V]` and last-token-only `[B,n_docs,V]`.
- Cache memory: generator KV cache size multiplied by `num_beams*n_docs`.
- Sequence reranking: candidate count after dedup versus NLL scoring cost.

## 14. Skip/defer list

- Training and context encoder training path.
- Legacy index ingestion inside DinoML, especially pickle-gated metadata.
- In-runtime Hugging Face Datasets/FAISS ownership.
- General AutoModel/AutoModelForSeq2SeqLM combinations beyond official DPR+BART until separately audited.
- Distributed training and distributed generation deduplication differences.
- Beam search variants outside sample/greedy/beam/beam-sample accepted by source.
- RAG-sequence full controller optimization; first support RAG-token or externally supplied candidates.

## 15. Final implementation checklist

- [ ] Parse `RagConfig` with nested question encoder and generator subconfigs.
- [ ] Add admission allowlist for official DPR+BART RAG checkpoints.
- [ ] Route DPR and BART bodies to separate audited loaders.
- [ ] Define external retriever/context ABI: `context_input_ids`, `context_attention_mask`, `doc_scores`, optional `retrieved_doc_embeds`.
- [ ] Implement/validate document score row-dot or BMM.
- [ ] Implement RAG-token marginalization.
- [ ] Implement document-aware generator cache reorder.
- [ ] Add flattened `B*n_docs` generator forward parity test.
- [ ] Add RAG-token decode parity with stubbed retrieval.
- [ ] Add RAG-sequence NLL/rerank parity test.
- [ ] Benchmark generator encoder/decode cost versus `n_docs`.
- [ ] Keep FAISS/datasets/tokenizer retrieval external until a separate provider contract exists.
