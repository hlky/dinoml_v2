# LayoutLMv2 DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/layoutlmv2-base-uncased, microsoft/layoutlmv2-large-uncased
Config source: HF config/preprocessor JSON snapshots plus local configuration defaults
Source files inspected:
- transformers/src/transformers/models/layoutlmv2/configuration_layoutlmv2.py
- transformers/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py
- transformers/src/transformers/models/layoutlmv2/processing_layoutlmv2.py
- transformers/src/transformers/models/layoutlmv2/image_processing_layoutlmv2.py
- transformers/src/transformers/models/layoutlmv2/image_processing_pil_layoutlmv2.py
- transformers/src/transformers/models/layoutlmv2/tokenization_layoutlmv2.py
Any missing files or assumptions: Detectron2 is a required runtime backend for native LayoutLMv2Model. Fine-tuned task checkpoints listed below returned 401 for raw files. Although docs annotate `bbox` as optional, the inspected forward path uses `bbox.dtype` before its fallback zero bbox assignment; DinoML should require `bbox` for parity or intentionally patch that behavior in a separate compatibility layer.
```

Snapshots written under `agents/plans/transformers/layoutlmv2/_sources/`.

Accessible configs:

- `microsoft/layoutlmv2-base-uncased`
- `microsoft/layoutlmv2-large-uncased`
- `microsoft/layoutlmv2-base-uncased`, revision `no_ocr`

401 gaps:

- [microsoft/layoutlmv2-base-uncased-finetuned-funsd](https://huggingface.co/microsoft/layoutlmv2-base-uncased-finetuned-funsd)
- [microsoft/layoutlmv2-base-uncased-finetuned-docvqa](https://huggingface.co/microsoft/layoutlmv2-base-uncased-finetuned-docvqa)
- [microsoft/layoutlmv2-base-uncased-finetuned-cord](https://huggingface.co/microsoft/layoutlmv2-base-uncased-finetuned-cord)

Access would confirm task-head `num_labels`, label maps, and processor/tokenizer files for those fine-tuned variants. The source implements task heads independently of those configs.

## 2. High-level architecture

LayoutLMv2 is a document text+image encoder with a Detectron2 ResNet-FPN visual backbone, BERT-style text embeddings, 2D layout embeddings, and bidirectional self-attention over concatenated text and visual tokens.

```text
document image + OCR/words/boxes
  -> image resize/BGR + WordPiece tokenization/bbox expansion
  -> Detectron2 FPN p2 -> adaptive 7x7 pooling -> 49 visual tokens
  -> text/layout embeddings + visual/layout embeddings
  -> concatenate text tokens then visual tokens
  -> bidirectional encoder with 1D and 2D relative attention bias
  -> text-only token/QA heads or CLS+visual pooled sequence head
```

Primary DinoML target: image+text encoder parity for token classification / document QA. Sequence classification is a close follow-up. Text-only staging is useful for subcomponents but is not the native forward contract because `LayoutLMv2Model.__init__` requires Detectron2 and `forward` always computes visual embeddings.

## 3. Important config dimensions

| Field | Base | Large | Notes |
|---|---:|---:|---|
| `vocab_size` | 30522 | 30522 | BERT WordPiece vocab |
| `hidden_size` | 768 | 1024 | encoder width |
| `num_hidden_layers` | 12 | 24 | encoder blocks |
| `num_attention_heads` | 12 | 16 | MHA |
| `head_dim` | 64 | 64 | `hidden_size / heads` |
| `intermediate_size` | 3072 | 4096 | GELU MLP |
| `max_position_embeddings` | 512 | 512 | text and visual position ids share table |
| `max_2d_position_embeddings` | 1024 | 1024 | bbox coords expected in 0..1000 |
| `coordinate_size` | 128 | 171 | x/y/left/right embedding width |
| `shape_size` | 128 | 170 | width/height embedding width |
| spatial concat width | 768 | 1024 | `4*coordinate_size + 2*shape_size` |
| `image_feature_pool_shape` | `[7,7,256]` | `[7,7,256]` | FPN `p2` -> 49 tokens |
| `fast_qkv` | true | false | packed QKV only in base config |
| `rel_pos_bins` / `max_rel_pos` | 32 / 128 | 32 / 128 | 1D relative bias |
| `rel_2d_pos_bins` / `max_rel_2d_pos` | 64 / 256 | 64 / 256 | x/y 2D relative bias |
| `has_relative_attention_bias` | default true | true | base config omits field, source default true |
| `has_spatial_attention_bias` | default true | true | base config omits field, source default true |
| `has_visual_segment_embedding` | default false | false | QA constructor overrides to true by default |
| image processor | resize 224, bilinear, OCR true | same | `no_ocr` revision sets OCR false |
| cache support | none | none | bidirectional encoder, no KV cache |

Historical/ignored config fields: accessible configs include `output_past: true`, but the inspected source has no past-key-value path and no generation cache.

## 4. Operator coverage checklist

### Tensor/layout ops

- Embedding lookup for word, 1D position, token type, x/y/h/w 2D bbox tables.
- `arange`, `expand`, `repeat`, `stack`, `cat(dim=1/-1)`, `view/reshape`, `transpose`, `permute`, `contiguous`.
- Bbox arithmetic: subtract width/height, integer floor divide, range generation, bucket indexing.
- Mask construction: unsqueeze to `[B,1,1,L_total]`, invert mask, fill with dtype minimum.
- Visual flatten: NCHW feature `[B,256,H_p2,W_p2] -> AdaptiveAvgPool2d(7,7) -> flatten(start_dim=2) -> transpose -> [B,49,256]`.

### Neural network primitives

- Linear:
  - base attention packed `Linear(768 -> 2304, bias=False)` plus learned q/v bias `[1,1,768]`.
  - large attention separate `Linear(1024 -> 1024)` Q/K/V with bias.
  - attention output `Linear(H -> H)`.
  - MLP `Linear(H -> 4H) -> GELU -> Linear(4H -> H)`.
  - visual projection `Linear(256 -> H)`.
  - pooler `Linear(H -> H) -> tanh`.
  - token/QA heads `Linear(H -> num_labels)`; sequence head `Linear(3H -> num_labels)`.
- LayerNorm eps `1e-12`, dropout inference no-op.
- Detectron2 ResNet-101 FPN ops: Conv2d, FrozenBatchNorm2d/possibly SyncBatchNorm conversion, ReLU, pooling, FPN lateral/top-down add/upsample.
- AdaptiveAvgPool2d or deterministic fallback AvgPool2d.

### Attention primitives

- Bidirectional dense MHA over `L_total = L_text + 49`.
- Eager matmul attention with scale applied to Q before QK.
- Additive 1D relative bias `[B, heads, L_total, L_total]`.
- Additive 2D spatial bias `[B, heads, L_total, L_total]`.
- Boolean/float padding mask conversion and masked fill.
- Softmax in fp32, cast back to value dtype.

### Position/relative-bias ops

- T5-style relative bucket function with bidirectional buckets and logarithmic tails.
- 1D bucket from `position_ids_i - position_ids_j`.
- 2D bucket from `bbox[:, :, 0]` and `bbox[:, :, 3]` only; source uses left x and lower y for relative spatial bias.
- Learned linear bias matrices used as indexed lookup via `weight.t()[bucket]`.

### Generation/cache ops

- None required. Reject/cache-ignore `output_past`.

### Preprocessing-coupled ops

- CPU/data pipeline: optional Tesseract OCR, empty-word filtering, bbox normalization to 0..1000, WordPiece tokenization, overflow mapping, subword bbox/label expansion.
- GPU/runtime: image channel contract is NCHW BGR pixel tensor, visual normalization by Detectron2 buffers, FPN, pooling, text-image concatenation.

### Distributed/tensor-parallel ops

- None for first integration. `synchronize_batch_norm` is training/distributed support and can be deferred.

## 5. Layer/block breakdown

Input packing:

```text
input_ids: [B, L_text]
bbox: [B, L_text, 4]
image: [B, 3, 224, 224] BGR, no model-side resize
visual tokens: [B, 49, H]
final sequence: [B, L_text + 49, H]
```

Text embedding:

```text
word = Embedding(vocab_size, H)(input_ids)
pos = Embedding(512, H)(position_ids)
spatial = cat(
  x(left), y(top), x(right), y(bottom), h(bottom-top), w(right-left)
)  # [B, L_text, H]
type = Embedding(type_vocab_size, H)(token_type_ids)
x_text = LayerNorm(word + pos + spatial + type)
```

Visual embedding:

```text
features = Detectron2FPN(image)["p2"]              # [B, 256, H2, W2]
features = AdaptiveAvgPool2d(7, 7)(features)
features = flatten_hw_transpose(features)          # [B, 49, 256]
visual = Linear(256 -> H)(features)
visual_bbox = generated 7x7 grid boxes in 0..1000
visual_pos_ids = arange(49)
x_visual = LayerNorm(visual + pos(visual_pos_ids) + spatial(visual_bbox) [+ visual_segment])
```

Encoder block, repeated `N` times:

```text
q,k,v = packed_or_separate_QKV(x)                  # [B, L_total, H]
q,k,v = reshape_to_heads([B, heads, L_total, 64])
scores = matmul(q / sqrt(64), k.T)
scores += rel_1d(position_ids)
scores += rel_2d(bbox)
scores = masked_fill(scores, mask_min)
probs = softmax(scores, dim=-1, dtype=fp32)
context = matmul(probs, v)
x = LayerNorm(Linear(context) + residual)
mlp = Linear(H -> intermediate) -> GELU -> Linear(intermediate -> H)
x = LayerNorm(mlp + residual)
```

Task heads:

- Token classification: slice text part `last_hidden_state[:, :L_text]`, dropout, `Linear(H -> num_labels)`.
- QA: slice text part, `Linear(H -> 2)`, split start/end.
- Sequence classification: compute initial visual embeddings separately, run full model, slice final text/visual, concatenate `[CLS_final, mean(initial_visual), mean(final_visual)]` -> `[B,3H]`, classify.

## 6. Attention requirements

- Type: bidirectional dense self-attention, not causal, no cross-attention.
- Head structure: MHA, no GQA/MQA. Base `12x64`, large `16x64`.
- Sequence length: `L_total = L_text + 49`; text max usually 512, so full attention can reach 561 if text is padded to 512.
- Masking: text attention mask is concatenated with all-ones visual mask; converted to additive min mask with shape `[B,1,1,L_total]`.
- Relative bias: both 1D and 2D biases are added before mask and softmax.
- Cache: none.
- FlashAttention/SDPA compatibility: possible only with an attention-bias backend that supports dense per-head `[B,H,L,L]` additive bias. Standard flash kernels without arbitrary bias cannot preserve parity.
- Eager fallback risk: dense bias materialization and softmax over `(L_text+49)^2` is acceptable for 512-token documents but should be benchmarked; Detectron2 FPN may dominate prefill-like latency.

## 7. Position encoding and custom math

2D absolute bbox embeddings:

```python
def spatial_bbox_embedding(bbox):
    left = x_embed(bbox[..., 0])
    top = y_embed(bbox[..., 1])
    right = x_embed(bbox[..., 2])
    bottom = y_embed(bbox[..., 3])
    height = h_embed(bbox[..., 3] - bbox[..., 1])
    width = w_embed(bbox[..., 2] - bbox[..., 0])
    return concat([left, top, right, bottom, height, width], dim=-1)
```

Relative bucket math:

```python
def rel_bucket(r, num_buckets, max_distance):
    half = num_buckets // 2
    sign = (r > 0) * half
    n = abs(r)
    exact = half // 2
    large = exact + log(n / exact) / log(max_distance / exact) * (half - exact)
    large = min(long(large), half - 1)
    return sign + where(n < exact, n, large)
```

2D relative bias:

```python
x_rel = rel_bucket(bbox[:, :, 0].unsqueeze(-2) - bbox[:, :, 0].unsqueeze(-1), 64, 256)
y_rel = rel_bucket(bbox[:, :, 3].unsqueeze(-2) - bbox[:, :, 3].unsqueeze(-1), 64, 256)
bias = x_bias_weight_T[x_rel].permute(0, 3, 1, 2)
bias += y_bias_weight_T[y_rel].permute(0, 3, 1, 2)
```

Visual bbox grid can be precomputed per `image_feature_pool_shape` and dtype/device, then repeated per batch:

```python
x = floor(arange(0, 1000 * (W + 1), 1000) / W)
y = floor(arange(0, 1000 * (H + 1), 1000) / H)
boxes = stack([x[:-1], y[:-1], x[1:], y[1:]] over HxW).reshape(H*W, 4)
```

The learned bias weights are static; bucket indices depend on runtime `position_ids` and `bbox`.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Image processor defaults to `do_resize=True`, bilinear resize to 224x224, then flips RGB to BGR.
- Default `apply_ocr=True` invokes Tesseract on CPU, filters empty words, and normalizes word boxes with `int(1000 * coord / width_or_height)`.
- `no_ocr` revision sets `apply_ocr=false`, allowing caller-provided `words`, `boxes`, and `word_labels`.
- Processor forbids user `boxes` or `word_labels` when `apply_ocr=True`.
- Tokenizer always encodes pretokenized words, expands word boxes to subword boxes, assigns special boxes:
  - CLS `[0,0,0,0]`
  - SEP `[1000,1000,1000,1000]`
  - PAD `[0,0,0,0]`
- Overflow mapping duplicates the corresponding image for each overflowed text chunk.

GPU/runtime work:

- Model consumes `image`, not `pixel_values`; processor output key is renamed to `image`.
- Treat `bbox` as required for native parity even though the signature marks it optional; source calls `_calc_visual_bbox(..., bbox, ...)` before the `bbox is None` fallback.
- Public image contract is NCHW. Treat the Detectron2 FPN as a no-layout-translation region at graph boundaries.
- Visual tokens are appended after text tokens; all task heads slice back to the first `L_text` tokens except sequence classification, which also pools visual tokens.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed LayoutLMv2 fast QKV -> split GEMMs or fused QKV

Source pattern:

```text
qkv = Linear(H -> 3H, bias=False)(x)
q,k,v = chunk(qkv, 3, dim=-1)
q += q_bias
v += v_bias
```

Replacement:

```text
FusedQKV(x, Wqkv, bias=[q_bias, zeros, v_bias]) -> split q,k,v
```

Preconditions:

- `config.fast_qkv == true`.
- Split order is exactly Q, K, V along output dim.
- K bias is zero.
- Input is `[B,L,H]` and output width is `3H`.

Failure cases: large config uses separate Q/K/V modules; do not apply packed rewrite there.

Parity test sketch: compare q/k/v tensors before reshape for random `[2,17,H]` and real checkpoint weights.

### Rewrite: visual bbox grid precompute

Source pattern: `_calc_visual_bbox` builds the same 7x7 normalized boxes every forward and repeats batch.

Replacement: constant `[49,4]` grid per pool shape, cast/repeat at runtime.

Preconditions:

- `image_feature_pool_shape[:2]` is static.
- `bbox.dtype` is integer and coordinate scale is 0..1000.
- Batch repeat preserves text+visual concatenation order.

Failure cases: dynamic pool shape or nonstandard coordinate scale.

### Rewrite: 2D bias as bucket lookup primitive

Source pattern:

```text
bucket(left_i - left_j) -> x_bias
bucket(bottom_i - bottom_j) -> y_bias
rel_2d = x_bias + y_bias
```

Replacement: fused `LayoutRelative2DBias(bbox, x_weight, y_weight, bins, max_distance)`.

Preconditions:

- Use source axes exactly: x uses `bbox[...,0]`, y uses `bbox[...,3]`.
- Output shape `[B, heads, L_total, L_total]`.
- Bias is added before mask.

Failure cases: bbox out of range for absolute embeddings, alternate coordinate convention, or attention backend cannot accept arbitrary additive bias.

### Rewrite: local NCHW Conv/FPN -> NHWC island

Source pattern: Detectron2 FPN NCHW conv/bn/relu/upsample stack.

Replacement: provider-selected NHWC conv island with entry/exit transposes.

Preconditions:

- Entire FPN subgraph is captured and all consumers are controlled through the final `[B,49,256]` token projection.
- Axis-sensitive ops are rewritten: channel flip `dim=1 -> dim=-1`, pixel mean/std broadcast, Conv2d, FrozenBN, upsample, FPN add, adaptive pool.
- Exit produces identical `[B,49,256]`.

Failure cases: partial graph capture, ImageList inputs with opaque layout, Detectron2 custom ops not lowered, or exposed intermediate FPN features.

## 10. Kernel fusion candidates

Highest priority:

- Detectron2 FPN conv/bn/relu/upsample path: visual branch is heavy and NCHW/NHWC placement determines most layout cost.
- Dense attention with relative 1D+2D bias: avoid separate `[B,H,L,L]` materialization where possible.
- Packed fast-QKV + q/v bias for base checkpoints.

Medium priority:

- Embedding sum + LayerNorm for text and visual embeddings.
- MLP `Linear -> GELU -> Linear` with residual LayerNorm.
- Visual `AdaptiveAvgPool2d(7,7) -> flatten/transpose -> Linear(256->H)`.

Lower priority:

- Pooler `Linear+tanh`.
- Task heads and last text-token slicing.
- Visual bbox grid generation once precomputed.

## 11. Runtime staging plan

1. Parse config and reject unsupported `output_past`/cache expectations.
2. Load text embeddings, 2D embeddings, encoder weights, and task heads; validate text/layout embedding and one encoder block with synthetic visual tokens.
3. Implement visual bbox grid and text+visual concatenation ABI.
4. Integrate Detectron2 FPN weights or provide a composed FPN lowering path; first parity target can keep NCHW throughout.
5. Run base encoder parity for `microsoft/layoutlmv2-base-uncased` with OCR disabled and supplied words/boxes.
6. Add task heads: token classification and QA first, sequence classification after initial/final visual pooling is checked.
7. Add large checkpoint support with separate Q/K/V path.
8. Optimize attention bias, visual branch NHWC islands, and fused QKV/MLP.

Stub initially: OCR, Tesseract, fine-tuned label maps, training losses, SyncBatchNorm conversion.

## 12. Parity and validation plan

- Unit-test bbox normalization and tokenizer bbox expansion against saved processor outputs.
- Random tensor tests for `spatial_bbox_embedding`, `relative_position_bucket`, 1D bias, 2D bias, and visual bbox grid.
- Single visual backbone probe: compare FPN `p2` pooled tokens before `visual_proj`.
- Single-layer encoder parity with fixed `[B,L_text+49,H]`, fixed bbox, and attention mask.
- Base full encoder parity with `apply_ocr=false`, deterministic supplied words/boxes, fp32 tolerance around `1e-5`.
- Large full encoder parity separately because QKV modules differ.
- Token classification/QA text-slice logits parity; sequence classification parity including initial and final visual means.
- Recommended tolerances: fp32 `1e-5` to `3e-5`; fp16/bf16 `1e-2` for visual+attention end-to-end after backend differences, tighter for isolated linear/embedding ops.

## 13. Performance probes

- OCR/preprocessing throughput separately from model runtime.
- Image resize/channel flip throughput and host-to-device transfer.
- Detectron2 FPN only, NCHW vs NHWC island, batch sweep.
- Encoder-only synthetic visual tokens to isolate attention/MLP.
- Attention backend comparison with arbitrary 1D+2D additive bias.
- Sequence length sweep: 128, 256, 512 text tokens plus fixed 49 visual tokens.
- Batch-size sweep for document QA and token classification.
- Memory usage for dense relative bias `[B,H,L,L]`.
- End-to-end document pages/sec with OCR disabled and enabled.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Tesseract OCR implementation inside GPU runtime; keep in CPU/data pipeline.
- Detectron2 SyncBatchNorm conversion and distributed training behavior.
- Sequence classification if first target is token classification/QA.
- Fine-tuned private/gated checkpoint label maps until access is available.
- Beam search, generation, KV cache, speculative decoding: not applicable.
- Remote/custom code: not required for inspected in-library LayoutLMv2.

## 15. Final implementation checklist

- [ ] Parse `LayoutLMv2Config` including base defaults omitted from old configs.
- [ ] Load text, 1D position, token type, 2D bbox, encoder, visual projection, and task-head weights.
- [ ] Implement absolute 2D bbox embedding concat.
- [ ] Implement visual 7x7 bbox grid generation/precompute.
- [ ] Implement relative bucket, 1D relative bias, and 2D spatial bias.
- [ ] Implement both packed fast-QKV and separate Q/K/V attention paths.
- [ ] Implement Detectron2 ResNet-FPN `p2` visual feature contract or equivalent lowered graph.
- [ ] Preserve NCHW model boundary; add guarded NHWC island only for fully captured FPN regions.
- [ ] Implement text+visual token concatenation and text-only output slicing for token/QA heads.
- [ ] Add processor fixture tests for OCR-disabled words/boxes and overflow image mapping.
- [ ] Add single-block, visual-backbone, full-encoder, token-head, QA-head parity tests.
- [ ] Benchmark visual branch, dense bias attention, and end-to-end pages/sec.
