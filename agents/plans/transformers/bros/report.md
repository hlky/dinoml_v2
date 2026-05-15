# BROS DinoML Audit

## 1. Source Basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary sweep covers `jinho8345/bros-base-uncased`, `jinho8345/bros-large-uncased`, `naver-clova-ocr/bros-base-uncased`, `naver-clova-ocr/bros-large-uncased`, and `adamadam111/bros-funsd-finetuned`.

Config source: Hugging Face raw `config.json` files, captured in `_sources/config_sweep.md`.

Source files inspected:

- `transformers/src/transformers/models/bros/configuration_bros.py`
- `transformers/src/transformers/models/bros/modeling_bros.py`
- `transformers/src/transformers/models/bros/processing_bros.py`
- `transformers/src/transformers/models/bros/__init__.py`

Any missing files or assumptions: no BROS tokenizer implementation exists; `BrosProcessor` wraps a BERT tokenizer. No image processor or OCR pipeline exists in the source. The report target is encoder and token/relation classification inference. Training losses, document-classification checkpoints that name a non-existent in-library `BrosForDocumentClassification`, and optional decoder/cross-attention flags are out of scope.

## 2. High-Level Architecture

BROS is an encoder-only document text/layout model: BERT-style token embeddings and dense Transformer encoder blocks, with pairwise relative bounding-box sinusoidal features injected into every self-attention score matrix.

```text
caller OCR/text/boxes -> BertTokenizer/BrosProcessor -> input_ids + attention_mask + bbox
  -> word/segment/1D-position embeddings
  -> pairwise relative bbox sinusoid + projection
  -> N encoder blocks with bbox-aware dense self-attention
  -> sequence hidden states
  -> optional token classifier or SPADE relation heads
```

Stage decomposition:

- CPU/data pipeline: OCR/text extraction, box normalization, word-to-subword box expansion, BERT tokenization, special-token box assignment, overflow handling.
- GPU/runtime graph: token embedding lookup, bbox relative-position tensor construction, dense encoder, token/relation heads.
- Independently stageable: base `BrosModel` hidden-state parity, simple token classifier, then SPADE relation heads.

## 3. Important Config Dimensions

| Field | Base | Large | Source/default meaning |
|---|---:|---:|---|
| `vocab_size` | 30522 | 30522 | BERT uncased vocabulary |
| `hidden_size` | 768 | 1024 | encoder width |
| `num_hidden_layers` | 12 | 24 | repeated encoder blocks |
| `num_attention_heads` | 12 | 16 | dense MHA heads |
| `head_dim` | 64 | 64 | `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | 4096 | FFN expansion |
| `max_position_embeddings` | 512 | 512 | learned 1D token positions |
| `type_vocab_size` | 2 | 2 | segment embeddings |
| `dim_bbox` | 8 effective | 8 effective | four box corners as x/y coordinates |
| `bbox_scale` | 100.0 | 100.0 | multiply input boxes before relative sinusoids |
| `dim_bbox_sinusoid_emb_2d` | 192 | 256 | source derives `hidden_size // 4` |
| `dim_bbox_sinusoid_emb_1d` | 24 | 32 | source derives `dim_bbox_sinusoid_emb_2d // dim_bbox` |
| `dim_bbox_projection` | 64 | 64 | source derives `head_dim` |
| `hidden_act` | gelu | gelu | BERT FFN activation |
| `layer_norm_eps` | 1e-12 | 1e-12 | post-residual LayerNorm |
| `n_relations` | 1 | 1 | SPADE relation-head count |
| `torch_dtype` | float32 | float32 | config metadata |

Representative checkpoint sweep:

| Checkpoint | Architecture | Layers | Hidden | Heads | Head dim | Labels | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `naver-clova-ocr/bros-base-uncased` | `BrosModel` | 12 | 768 | 12 | 64 | n/a | older config omits `dim_bbox*`; current source derives them |
| `jinho8345/bros-base-uncased` | `BrosModel` | 12 | 768 | 12 | 64 | n/a | includes derived bbox fields matching current source |
| `naver-clova-ocr/bros-large-uncased` | `BrosModel` | 24 | 1024 | 16 | 64 | n/a | older config omits `dim_bbox*`; current source derives them |
| `jinho8345/bros-large-uncased` | `BrosModel` | 24 | 1024 | 16 | 64 | n/a | larger depth/width, same head dim |
| `adamadam111/bros-funsd-finetuned` | `BrosForTokenClassification` | 12 | 768 | 12 | 64 | 7 | standard token classifier head |
| `adamadam111/bros-docclass-finetuned` | `BrosForDocumentClassification` | 12 | 768 | 12 | 64 | 5 | config names a class absent from inspected source; reject/route separately |

## 3a. Family Variation Traps

- Current source derives bbox embedding dimensions from `hidden_size`, `num_attention_heads`, and `dim_bbox`; old configs omit those fields. DinoML should derive or validate consistency rather than trusting stale serialized values.
- `dim_bbox_sinusoid_emb_1d` must be even for the sin/cos split. With default derivation, base and large are safe; unusual `hidden_size`/`dim_bbox` combinations need an admission guard.
- Source accepts `bbox` last dimension 4 or 8. Four-value boxes are expanded to 8 coordinates; any other last dimension should be rejected.
- Source does not clamp or range-check boxes. DinoML should add explicit guards for expected normalized boxes if its processor contract assumes `[0,1]`, because the model multiplies by `bbox_scale=100.0`.
- BROS is not an image model at runtime. OCR and box generation are caller/data-pipeline responsibilities.
- `BrosProcessor` uses a BERT tokenizer and does not expand word boxes to subwords. That coupling must be owned by the integration wrapper.
- SPADE heads are axis-sensitive: hidden states are transposed to `[S,B,H]`, relation logits are `[B,S,S+1]` after squeezing `n_relations=1`, and the dummy node is appended on the key axis.
- The in-library source has encoder/decoder config fields, but sampled configs are encoder-only. Reject `is_decoder=True`/`add_cross_attention=True` for first integration.
- `BrosForDocumentClassification` appears in a finetuned config but is not implemented by the inspected source.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Embedding lookup for `input_ids`, `position_ids`, `token_type_ids`.
- Add, dropout-as-identity for inference, LayerNorm.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `cat`.
- Pairwise broadcast subtract for boxes: `[S,1,B,8] - [1,S,B,8] -> [S,S,B,8]`.
- Mask expansion via BERT-style attention mask to additive score mask.
- `masked_fill` for SPADE logits and invalid/self-token masks.
- Boolean `eye(S, S+1)` and concat with a dummy target column for relation heads.

Neural network primitives:

- Linear `hidden_size -> hidden_size` for Q/K/V/output.
- Linear `hidden_size -> intermediate_size`, GELU, Linear `intermediate_size -> hidden_size`.
- Pooler: take token 0, Linear `hidden_size -> hidden_size`, tanh.
- Token classifier: Linear `hidden_size -> num_labels`.
- SPADE initial classifier: Dropout, Linear `H -> H`, Dropout, Linear `H -> num_labels`.
- Relation extractor: query/key Linear `H -> n_relations * H`, learned dummy node `[1,H]`, batched matmul.

Attention primitives:

- Noncausal dense self-attention over sequence length `S`.
- MHA only; no GQA/MQA.
- Attention logits: `Q @ K^T + Q dot bbox_relative_embedding`, then divide by `sqrt(head_dim)`, add mask, softmax, `P @ V`.
- No KV cache for the primary target.

Position/layout math:

- Learned 1D token position embeddings.
- Pairwise relative 2-D bbox sin/cos features projected to head dimension.

Preprocessing-coupled ops:

- BERT tokenization outside graph.
- Caller-provided `bbox`, `attention_mask`, optional `token_type_ids`, optional `bbox_first_token_mask`.
- Integration wrapper must duplicate word boxes to all subword pieces and assign boxes for `[CLS]`, `[SEP]`, and padding.

## 5. Layer/Block Breakdown

Base embeddings:

```text
input_ids [B,S] -> word_embeddings [B,S,H]
token_type_ids [B,S] -> segment embeddings [B,S,H]
position_ids [1,S] or [B,S] -> learned position embeddings [B,S,H]
x = LayerNorm(word + segment + position)
```

Bbox embedding precompute per forward:

```text
bbox [B,S,4] -> [B,S,8] if needed
scaled_bbox = bbox * bbox_scale
bbox_t = transpose to [S,B,8]
relative = bbox_t[None,:,:,:] - bbox_t[:,None,:,:]  # [S,S,B,8]
sin/cos per coordinate -> [S,S,B,hidden_size//4]
projection -> [S,S,B,head_dim]
```

Encoder block, repeated `num_hidden_layers` times:

```text
q,k,v = Linear(H -> H)(x), split to [B,num_heads,S,head_dim]
scores = q @ k.transpose(-1,-2)
bbox_scores = einsum("bnid,bijd->bnij", q, bbox_rel_proj_as_[B,S,S,head_dim])
scores = (scores + bbox_scores) / sqrt(head_dim)
scores = scores + attention_mask
probs = softmax(scores, dim=-1)
attn = probs @ v
x = LayerNorm(Linear(attn) + residual)
ff = GELU(Linear(H -> intermediate)(x))
x = LayerNorm(Linear(intermediate -> H)(ff) + residual)
```

SPADE relation head:

```text
h = last_hidden_state.transpose(0,1)  # [S,B,H]
query = Linear(H -> n_relations*H)(h)
key_input = cat([h, dummy_node repeated as [1,B,H]], axis=0)
key = Linear(H -> n_relations*H)(key_input)
logits = matmul(query.permute(2,1,0,3), key.permute(2,1,3,0))  # [R,B,S,S+1]
```

## 6. Attention Requirements

Required attention is noncausal encoder self-attention. There is no generation cache, no causal mask, no local/sliding pattern, no packed varlen path, and no RoPE/ALiBi.

| Property | Requirement |
|---|---|
| Type | dense self-attention |
| Causality | noncausal |
| Heads | MHA, base 12 and large 16 |
| Head dim | 64 in sampled configs |
| Query/key/value width | all `hidden_size` |
| Query length vs key length | square `S x S` for primary target |
| Mask | additive attention mask broadcast to heads |
| Extra score term | `q_i dot projected_relative_bbox(i,j)` |
| Cache | none |
| FlashAttention compatibility | not direct unless backend supports an additive query-dependent relative-bbox score term |

The bbox score is the blocker for ordinary fused attention. A dense attention kernel can be reused only if DinoML materializes an additive score tensor `[B,heads,S,S]` or extends the attention provider to compute `Q dot R_ij` before softmax.

## 7. Position Encoding and Custom Math

BROS uses learned 1D token positions plus custom relative bbox sinusoids.

```python
def bros_expand_bbox(bbox):
    # bbox [B,S,4] as x1,y1,x2,y2
    return bbox[:, :, [0, 1, 2, 1, 2, 3, 0, 3]]

def bros_bbox_embedding(bbox8, bbox_scale, inv_freq, proj_weight):
    scaled = bbox8 * bbox_scale
    t = scaled.transpose(0, 1)                         # [S,B,8]
    rel = t[None, :, :, :] - t[:, None, :, :]           # [S,S,B,8]
    pieces = []
    for c in range(8):
        inp = rel[..., c, None] * inv_freq              # [S,S,B,D1/2]
        pieces.append(concat([sin(inp), cos(inp)], -1))
    emb = concat(pieces, -1)                            # [S,S,B,H/4]
    return linear_no_bias(emb, proj_weight)             # [S,S,B,head_dim]
```

Precomputable: `inv_freq`, learned projection weights, learned token position embeddings. Dynamic per request: pairwise bbox differences and sin/cos features because they depend on caller boxes.

## 8. Preprocessing and Input Packing

The source has no OCR. The caller must provide text tokens and boxes.

Input tensors for the neural graph:

- `input_ids`: `[B,S]` int token ids from a BERT uncased tokenizer.
- `bbox`: `[B,S,4]` or `[B,S,8]` float tensor. Source examples use small normalized coordinates; source does not enforce range.
- `attention_mask`: `[B,S]` or compatible BERT mask; default is all ones.
- `token_type_ids`: optional `[B,S]`; default zeros.
- `position_ids`: optional; default contiguous `[0..S-1]`.
- `bbox_first_token_mask`: optional for token-classification losses/SPADE post-loss filtering; for inference it can be omitted unless the integration exposes loss or first-token-only postprocessing.

Tokenizer/OCR coupling:

- `BrosProcessor` only wraps `BertTokenizer`; it does not call OCR.
- It defaults to `add_special_tokens=True`, `padding=False`, `stride=0`, and `return_overflowing_tokens=False`.
- For end-to-end document parity, the integration must define how each OCR word box is copied to subword tokens and what boxes are assigned to special/pad tokens.
- Since `max_position_embeddings=512`, long documents require chunking/overflow in the data pipeline. DinoML should not silently accept `S > 512`.

Layout guards:

- Require `input_ids.shape == attention_mask.shape == bbox.shape[:2]`.
- Require `bbox.shape[-1] in {4,8}`.
- Require `S <= max_position_embeddings`.
- Require `hidden_size % num_attention_heads == 0`.
- For optimized bbox math, require contiguous row-major `bbox` or add an explicit copy.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: BROS Q/K/V Projection Packing

Source pattern: three independent `Linear(H -> H)` projections for Q, K, V.

Replacement pattern: one packed GEMM producing `[B,S,3H]`, split in Q,K,V order.

Preconditions:

- Same input tensor for Q/K/V.
- All three projections have bias enabled, as PyTorch `nn.Linear` default.
- Weight pack order is all-Q rows, then all-K rows, then all-V rows.

Failure cases: cross-attention or checkpoint with nonstandard projection aliases.

Parity test sketch: compare packed projection split against three linears for base and large configs.

### Rewrite: Relative Bbox Score as Attention-Bias Producer

Source pattern:

```text
relative bbox -> sin/cos -> Linear(H/4 -> head_dim)
einsum(q, rel_bbox) -> [B,heads,S,S]
scores += bbox_scores
```

Replacement pattern: compile as a score-bias producer feeding dense attention, or fuse into a custom BROS attention kernel.

Preconditions:

- `dim_bbox=8`.
- `dim_bbox_projection == head_dim`.
- Source attention order preserved: add bbox score before scaling, then divide both content and bbox scores by `sqrt(head_dim)`, then add mask.

Failure cases: trying to use stock FlashAttention without supporting query-dependent score terms.

Parity test sketch: random boxes and hidden states, compare attention logits before softmax.

### Rewrite: 4-Coord Box Expansion

Source pattern: `bbox[:, :, [0,1,2,1,2,3,0,3]]`.

Replacement pattern: normalize all runtime input to `[B,S,8]` at graph boundary.

Preconditions:

- Input contract states boxes are `[x1,y1,x2,y2]`.
- Expansion order exactly matches source.

Failure cases: caller already supplies 8-point boxes; do not expand again.

### Rewrite: SPADE Relation Matmul to BMM/GEMM

Source pattern: transpose hidden to `[S,B,H]`, append dummy key, query/key linears, relation matmul.

Replacement pattern: represent as batched GEMM over batch and relation dimensions.

Preconditions:

- `n_relations=1` for first integration.
- Dummy node is a learned parameter appended after all sequence tokens.
- Output orientation `[B,S,S+1]` preserved.

Failure cases: `n_relations > 1` if output squeezing/layout differs from current heads.

## 10. Kernel Fusion Candidates

Highest priority:

- BROS bbox-relative attention score producer. This is the family-specific kernel gap and dominates the difference from BERT.
- Encoder LayerNorm + residual patterns. BROS has post-norm BERT blocks with repeated residual/LN.
- QKV packed projection. Reduces launch count and maps cleanly to existing GEMM work.

Medium priority:

- GELU FFN fusion: Linear/GELU/Linear epilogue opportunities after base parity.
- Attention mask and bbox score addition fusion before softmax.
- SPADE relation head batched GEMM plus mask fill for relation logits.

Lower priority:

- Pooler fusion, because token classification and relation extraction usually consume sequence states.
- Token classifier head fusion, useful but straightforward.

## 11. Runtime Staging Plan

Stage 1: parse `BrosConfig`, load base/large weights, and reject unsupported architectures/flags.

Stage 2: implement graph boundary ABI for `input_ids`, `bbox`, `attention_mask`, `token_type_ids`, and `position_ids`; add strict layout guards.

Stage 3: lower bbox expansion and pairwise relative bbox sinusoid/projection as explicit ops; validate a single attention layer.

Stage 4: run full `BrosModel` encoder parity for base config.

Stage 5: add `BrosForTokenClassification` head parity.

Stage 6: add SPADE EE/EL relation heads if needed by target workloads.

Stage 7: optimize QKV packing and bbox-aware attention score fusion.

Stub initially: dropout, training losses, hidden-state/attention output capture, pooler if only token classification is targeted.

## 12. Parity and Validation Plan

- Config admission tests: base, large, older configs without `dim_bbox*`, and rejection for unsupported document-classification architecture.
- Bbox expansion tests for `[B,S,4] -> [B,S,8]`.
- Bbox sinusoid/projection parity with random normalized and out-of-range boxes.
- Single self-attention layer parity including bbox score, mask, and softmax order.
- Encoder one-layer and full-depth parity with fixed random weights.
- Token classifier parity for logits `[B,S,num_labels]`.
- SPADE relation-head parity for logits `[B,S,S+1]`, dummy-node placement, invalid-token mask, and self-token mask.
- End-to-end sample parity using a BERT tokenizer, caller-supplied boxes, and `jinho8345/bros-base-uncased`.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5` for full encoder; fp16/bf16 should start with looser end-to-end tolerance after bbox sin/cos and softmax accumulation are validated in fp32.

## 13. Performance Probes

- Processor/OCR boundary throughput separately from GPU encoder throughput.
- Sequence length sweep: `S=64,128,256,512`, because pairwise bbox tensors are `O(S^2)`.
- Base vs large encoder latency and memory.
- Bbox score producer time and temporary memory footprint.
- Dense attention backend comparison: materialized bbox score tensor vs fused score computation.
- Token classification throughput.
- SPADE relation head throughput for `[B,S,S+1]` logits.
- Batch-size sweep with fixed `S=512`.

## 14. Skip/Defer List

- Training losses and gradient checkpointing.
- Decoder/cross-attention mode.
- `BrosForDocumentClassification` configs not backed by inspected source.
- General OCR/image preprocessing.
- Automatic long-document chunking and overflow aggregation, except as a wrapper-level later feature.
- General boolean scatter; BROS does not require it for encoder inference.
- FlashAttention replacement until bbox score fusion is available.

## 15. Final Implementation Checklist

- [ ] Parse `BrosConfig` and derive bbox dimensions from current source rules.
- [ ] Load BERT tokenizer metadata as a data-pipeline dependency, not a graph op.
- [ ] Define document ABI for `input_ids`, `bbox`, `attention_mask`, `token_type_ids`, and special-token boxes.
- [ ] Add guards for `S <= 512`, `bbox.shape[-1] in {4,8}`, and matching `[B,S]` dimensions.
- [ ] Implement 4-coordinate bbox expansion.
- [ ] Implement pairwise bbox relative subtraction.
- [ ] Implement BROS 1D/2D bbox sin/cos embedding.
- [ ] Implement bbox projection `Linear(hidden_size//4 -> head_dim, bias=False)`.
- [ ] Implement bbox-aware dense self-attention score path.
- [ ] Implement BERT-style embedding, encoder FFN, LayerNorm, and pooler.
- [ ] Add QKV packed projection rewrite with weight-pack parity tests.
- [ ] Add token classification head.
- [ ] Add SPADE relation extractor and masks if needed.
- [ ] Reject unsupported document-classification and decoder/cross-attention configs.
- [ ] Add single-layer, full-encoder, and end-to-end parity tests.
- [ ] Benchmark bbox score producer and dense attention at `S=512`.
