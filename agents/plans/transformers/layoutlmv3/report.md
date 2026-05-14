# LayoutLMv3 DinoML Operator Assessment

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/layoutlmv3-base, microsoft/layoutlmv3-large
Config source: official HF config.json and preprocessor_config.json snapshots under _sources/hf_configs/
Source files inspected:
- X:/H/transformers/src/transformers/models/layoutlmv3/modeling_layoutlmv3.py
- X:/H/transformers/src/transformers/models/layoutlmv3/configuration_layoutlmv3.py
- X:/H/transformers/src/transformers/models/layoutlmv3/processing_layoutlmv3.py
- X:/H/transformers/src/transformers/models/layoutlmv3/image_processing_layoutlmv3.py
- X:/H/transformers/src/transformers/models/layoutlmv3/image_processing_pil_layoutlmv3.py
- X:/H/transformers/src/transformers/models/layoutlmv3/tokenization_layoutlmv3.py
Any missing files or assumptions:
- Native Transformers source implements base encoder, token classification, question answering, and sequence classification heads.
- No native `LayoutLMv3ForMaskedLM` or masked-image-modeling head is present in this source basis; tokenizer `<mask>` is preprocessing/tokenizer support only for this audit.
- `microsoft/layoutlmv3-base-finetuned-funsd`, `microsoft/layoutlmv3-base-finetuned-cord`, and `microsoft/layoutlmv3-base-finetuned-rvlcdip` raw config/preprocessor/tokenizer files returned 401. Access links are listed in the variation/gaps sections.
- `processor_config.json` is absent for public base/large repos; processor behavior is reconstructed from source plus `preprocessor_config.json` and `tokenizer_config.json`.
```

Primary runtime target: document AI multimodal encoder inference with text + layout boxes + page image, first for token classification and question answering. Sequence classification is a small additional head. Training losses, OCR execution, and any pretraining-only MIM/MLM behavior are out of first-runtime scope.

## 2. High-level architecture

LayoutLMv3 is a single-stream multimodal encoder. Text tokens and visual patch tokens are embedded separately, concatenated, then processed by a RoBERTa-like encoder with noncausal full self-attention. The attention scores receive both 1D relative position bias and 2D spatial relative bias derived from token/page bounding boxes.

```text
CPU image resize/OCR + tokenizer box expansion
  -> text embeddings + bbox embeddings
  -> NCHW image patch Conv2d + visual CLS/pos embeddings
  -> concat text tokens with visual CLS+patch tokens
  -> encoder with 1D relative bias + 2D spatial bias
  -> task head over text tokens, all tokens, or CLS token
```

Stage decomposition:

- CPU/data pipeline: optional Tesseract OCR, word bounding-box normalization to 0-1000, RoBERTa byte-level BPE tokenization, word-box expansion to subword boxes, image resize/rescale/normalize.
- GPU/runtime stem: text embedding sum, spatial embedding concat/add, NCHW patch Conv2d, visual position embedding add, visual/text concat.
- Encoder: repeated noncausal MHA + FFN, with attention masks and precomputed relative bias tensors shared across layers.
- Heads: token classification slices only text token outputs; QA projects every combined token to start/end logits; sequence classification uses token 0.

Independently stageable pieces: processor contract, patch embedding parity, relative/2D bias parity, one encoder block parity, then task-head parity.

## 3. Important config dimensions

| Field | `layoutlmv3-base` config.json | `layoutlmv3-large` config.json | Source default |
|---|---:|---:|---:|
| `hidden_size` | 768 | 1024 | 768 |
| `num_hidden_layers` | 12 | 24 | 12 |
| `num_attention_heads` | 12 | 16 | 12 |
| `head_dim` | 64 | 64 | `hidden_size / num_attention_heads` |
| `intermediate_size` | 3072 | 4096 | 3072 |
| `vocab_size` | 50265 | 50265 | 50265 |
| `max_position_embeddings` | 514 | 514 | 512 |
| `type_vocab_size` | 1 | 1 | 2 |
| `max_2d_position_embeddings` | 1024 | 1024 | 1024 |
| `coordinate_size` | 128 | 171 | 128 |
| `shape_size` | 128 | 170 | 128 |
| spatial concat width | `4*128 + 2*128 = 768` | `4*171 + 2*170 = 1024` | `4*coordinate_size + 2*shape_size` must equal hidden size |
| `input_size` | 224 | 224 | 224 |
| `patch_size` | omitted, effective 16 | omitted, effective 16 | 16 |
| visual token count at 224 | 197 = 1 + 14*14 | 197 | 197 |
| `num_channels` | omitted, effective 3 | omitted, effective 3 | 3 |
| `hidden_act` | gelu | gelu | gelu |
| `layer_norm_eps` | 1e-5 | 1e-5 | 1e-5 |
| visual final norm eps | 1e-6 source constant | 1e-6 source constant | 1e-6 |
| relative 1D bins/max | 32 / 128 | 32 / 128 | 32 / 128 |
| relative 2D bins/max | 64 / 256 | 64 / 256 | 64 / 256 |
| `has_relative_attention_bias` | true | true | true |
| `has_spatial_attention_bias` | true | true | true |
| dtype | float32 from config | float32 from config | not fixed by source |
| cache support | none | none | none |

Representative checkpoint sweep:

| Model id | Accessible raw config? | Operator-significant notes |
|---|---:|---|
| `microsoft/layoutlmv3-base` | yes | 12-layer 768-dim encoder, 12 heads, 14x14 visual grid at 224, `type_vocab_size=1`, preprocessor has `apply_ocr=true`. |
| `microsoft/layoutlmv3-large` | yes | Same operator structure with 24 layers and 1024 hidden size; spatial embedding split changes to `coordinate_size=171`, `shape_size=170`. |
| [`microsoft/layoutlmv3-base-finetuned-funsd`](https://huggingface.co/microsoft/layoutlmv3-base-finetuned-funsd) | 401 | Would confirm token-classification label count/head shape. Native head shape otherwise follows `num_labels`. |
| [`microsoft/layoutlmv3-base-finetuned-cord`](https://huggingface.co/microsoft/layoutlmv3-base-finetuned-cord) | 401 | Would confirm token-classification label count/head shape. |
| [`microsoft/layoutlmv3-base-finetuned-rvlcdip`](https://huggingface.co/microsoft/layoutlmv3-base-finetuned-rvlcdip) | 401 | Would confirm sequence-classification label count/head shape. |

## 3a. Family variation traps

- `max_position_embeddings` is 514 in public configs but 512 in source defaults; do not assume source defaults match pretrained weights.
- `type_vocab_size` is 1 in public configs, so token type IDs must be all zero for those weights. Source default is 2.
- `patch_size` and `num_channels` are omitted in public configs and come from source defaults: 16 and 3.
- The spatial embedding concatenation must exactly equal `hidden_size`: `4 * coordinate_size + 2 * shape_size`. Large uses asymmetric 171/170 sizes.
- Text and visual sequences are concatenated as `text tokens` then `visual CLS` then `visual patches`; this is not the ViT convention of a single leading CLS over all modalities.
- Token-classification logits slice only the first `text_seq_len` outputs. QA logits use the full combined sequence, including visual tokens.
- Visual bbox is generated from the configured pretraining grid size. Dynamic image sizes change patch count, but `calculate_visual_bbox` reuses the registered 14x14 bbox table in this source, so first integration should require processor size 224x224 unless a parity test approves other sizes.
- Patch Conv2d source is NCHW. NHWC is an optimization region only; concat axes, flatten order, and public `pixel_values` contract need no-layout-translation guards.
- Relative bias tensors are computed once per encoder forward and shared across all layers. They are not per-layer parameters.
- Source has `patch_height`/`patch_width` arguments into the encoder but this native code does not use them after passing them through.
- Public preprocessor `feature_extractor_type` is historical; current source class is `LayoutLMv3ImageProcessor`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B, 3, H, W]`, usually `[B, 3, 224, 224]`.
- Conv patch output flatten: `[B, C, Hp, Wp] -> [B, Hp*Wp, C]` via flatten spatial then transpose.
- `cat` along sequence axis for visual CLS + patches and text + visual tokens.
- `cat` along last axis for spatial embeddings.
- `view`, `transpose`, `permute`, `contiguous`, `reshape`, `expand`, `repeat`.
- `cumsum` over token axis for default text position IDs.
- `clip` for bbox width/height embedding indices.
- `arange`, integer division with truncation for visual bbox table.
- Attention mask expansion to additive broadcast shape `[B, 1, 1, S_total]`.

Neural network primitives:

- Embedding lookup: word `[vocab_size, hidden]`, token type `[type_vocab_size, hidden]`, 1D position `[max_position_embeddings, hidden]`.
- Spatial embeddings: x/y `[1024, coordinate_size]`, h/w `[1024, shape_size]`; concatenate six tensors to hidden width.
- Patch Conv2d: base `Conv2d(3 -> 768, kernel=16, stride=16, bias=True)`; large `Conv2d(3 -> 1024, kernel=16, stride=16, bias=True)`.
- Linear Q/K/V with bias: base `Linear(768 -> 768)`, large `Linear(1024 -> 1024)`.
- Attention output dense with bias: same hidden-to-hidden shape.
- FFN: base `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`; large `Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)`.
- LayerNorm eps 1e-5 for text/encoder, eps 1e-6 for visual-only `norm`.
- Dropout can be no-op in inference but shape placement matters for training parity if ever enabled.
- Head projections: token classification `Linear(hidden -> num_labels)` when `num_labels < 10`, else two-layer classification head; QA uses classification head to `num_labels` then split; sequence classification uses two-layer head over CLS.

Attention primitives:

- Noncausal encoder self-attention only.
- MHA: base 12 heads x 64, large 16 heads x 64. No MQA/GQA.
- Additive mask, 1D relative bias, 2D x/y spatial relative bias.
- CogView/PB-relax softmax variant: `softmax((scores / 32 - max(scores / 32)) * 32)`.
- No KV cache, no cross-attention, no generation decode path.

Position/relative-bias ops:

- Position ID generation for text: `cumsum(input_ids != pad_token_id) + pad_token_id`; visual position IDs start at 0.
- Relative position bucket with bidirectional exact/log buckets.
- Bias table lookup using linear weights transposed: `rel_pos_bias.weight.t()[bucket] -> [B, heads, S, S]`.
- 2D spatial bias from bbox left x and lower y (`bbox[...,0]` and `bbox[...,3]`), with separate x/y bias tables added.
- Visual absolute `pos_embed` parameter `[1, 197, hidden]`; patch position interpolation exists only if `patch_embed(..., position_embedding=...)` is called, but `forward_image` does not pass `position_embedding`.

Preprocessing-coupled ops:

- Optional Tesseract OCR and normalized boxes `[x0,y0,x1,y1]` in 0-1000 range.
- RoBERTa byte-level BPE tokenization with `add_prefix_space=True`.
- Word-level boxes expanded to subwords using tokenizer `word_ids`; question sequence boxes become pad boxes.
- Overflow image duplication based on `overflow_to_sample_mapping`.
- Image resize to 224 with bilinear resampling, rescale by 1/255 from backend defaults, normalize with mean/std `[0.5,0.5,0.5]` for public checkpoints.

Distributed/tensor-parallel ops: none required for the native source. Tensor parallel sharding can be deferred.

## 5. Layer/block breakdown

Text embedding path, for text length `T`:

```text
if position_ids missing:
  position_ids = cumsum(input_ids != pad_id, dim=1) + pad_id
token_type_ids default = zeros([B,T])
bbox default = zeros([B,T,4])
word = Embedding(vocab -> H)(input_ids)
type = Embedding(type_vocab -> H)(token_type_ids)
pos = Embedding(max_pos -> H)(position_ids)
spatial = concat(x0,y0,x1,y1,h,w embeddings)  # [B,T,H]
text_embeddings = LayerNorm(word + type + pos + spatial, eps=1e-5)
```

Visual embedding path, for image `[B,3,H_img,W_img]`:

```text
patches = Conv2d(3 -> H, kernel=16, stride=16)(pixel_values)  # [B,H,Hp,Wp]
patches = flatten_spatial_then_transpose(patches)             # [B,Hp*Wp,H]
visual = cat(cls_token.expand(B,1,H), patches, dim=1)          # [B,1+Hp*Wp,H]
visual = visual + pos_embed                                   # [1,197,H] for 224x224
visual = LayerNorm(visual, eps=1e-6)
```

Combined encoder input:

```text
x = cat(text_embeddings, visual_embeddings, dim=1) if both are present
x = LayerNorm(x, eps=1e-5)
x = dropout(x)
attention_mask = cat(text_mask, ones([B, visual_len]), dim=1)
final_bbox = cat(text_bbox, visual_bbox, dim=1)
final_position_ids = cat(text_position_ids_or_arange, visual_arange, dim=1)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,heads,64).transpose(1,2)
scores = matmul(q / sqrt(64), k.transpose(-1,-2))
scores += (rel_pos + rel_2d_pos) / sqrt(64)
scores += additive_attention_mask
prob = cogview_softmax(scores, alpha=32)
ctx = matmul(prob, v).transpose_to_BSH
x = LayerNorm(Linear(H -> H)(ctx) + residual, eps=1e-5)
y = GELU(Linear(H -> intermediate)(x))
x = LayerNorm(Linear(intermediate -> H)(y) + x, eps=1e-5)
```

Heads:

- Token classification: take `encoder[:, :T_text, :]`, dropout, then `Linear(H -> num_labels)` if `num_labels < 10`, otherwise dropout/dense/tanh/dropout/out projection.
- QA: classification head over all combined tokens, split final dim into start/end.
- Sequence classification: take `encoder[:,0,:]`, classification head.

## 6. Attention requirements

LayoutLMv3 needs encoder full self-attention only:

- Causal: no.
- Cross-attention: no.
- MHA/MQA/GQA: standard MHA; Q, K, V widths all equal hidden size; value width equals query/key width.
- Query length equals key/value length `S = T_text + 1 + Hp*Wp` for text+image, or `S = T_text`, or `S = 1+Hp*Wp`.
- Masking: additive broadcast mask from attention mask. Text masks are extended after appending all-one visual masks.
- Relative bias: source adds `(rel_pos + rel_2d_pos) / sqrt(head_dim)` after scaled QK matmul.
- Softmax: source uses CogView attention, not vanilla backend softmax. Dinoml fused attention must preserve this math order or route to an eager-compatible path.
- Packed/varlen: processor can produce overflow records, but model receives padded dense batches. No `cu_seqlens` ABI.
- Sliding/local/block sparse: none.
- KV cache: none. Encoder outputs may be cached by application code for retrieval/classification pipelines, but there is no source KV-cache ABI.
- FlashAttention/SDPA compatibility: raw full-attention math is compatible only if backend supports additive per-head `[B,heads,S,S]` bias and CogView softmax semantics. A vanilla FlashAttention path would not be parity-safe without a custom score modifier.

Likely too-slow fallback: materializing `[B, heads, S, S]` 1D and 2D bias plus full attention. For 224 images and 512 text tokens, `S` can be about 709, so this is manageable but still important for batching.

## 7. Position encoding and custom math

Relative bucket implementation to reproduce:

```python
def layoutlmv3_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = abs(relative_position)
    else:
        n = max(-relative_position, 0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (log(n.float() / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    large = min(large, num_buckets - 1)
    return ret + where(is_small, n, large)
```

2D bias generation:

```python
def layoutlmv3_2d_bias(bbox, x_weight_t, y_weight_t):
    rel_x = bbox[:, :, 0].unsqueeze(-2) - bbox[:, :, 0].unsqueeze(-1)
    rel_y = bbox[:, :, 3].unsqueeze(-2) - bbox[:, :, 3].unsqueeze(-1)
    bx = layoutlmv3_bucket(rel_x, num_buckets=64, max_distance=256)
    by = layoutlmv3_bucket(rel_y, num_buckets=64, max_distance=256)
    return x_weight_t[bx].permute(0, 3, 1, 2) + y_weight_t[by].permute(0, 3, 1, 2)
```

CogView/PB-relax softmax:

```python
def cogview_softmax(scores, alpha=32):
    scaled = scores / alpha
    return softmax((scaled - scaled.amax(dim=-1, keepdim=True)) * alpha, dim=-1)
```

Precomputable:

- Visual bbox table for a fixed grid.
- Visual position IDs for fixed image size.
- Relative 1D buckets for fixed sequence layouts and padding-free text lengths.

Dynamic:

- Text position IDs depend on padding and input IDs unless provided.
- 2D bias depends on OCR/tokenizer-provided bbox and visual bbox concatenation.
- Attention mask depends on text padding and visual length.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Image processor optionally applies Tesseract OCR. If `apply_ocr=True`, caller may not pass `boxes` or `word_labels`; OCR returns words and boxes.
- OCR boxes are normalized as `int(1000 * coord / image_dim)`.
- Empty OCR words are filtered before boxes are normalized.
- Images are resized, rescaled, and normalized before model input. Public configs use size 224, bilinear resampling, mean/std 0.5.
- Processor then tokenizes words with LayoutLMv3 tokenizer. For QA, `text` can be a question and OCR/word tokens become `text_pair`.
- Tokenizer requires word-level boxes unless OCR supplied them. It expands boxes to subword tokens via `word_ids`.
- RoBERTa special-token layouts: single sequence `<s> X </s>`; pair `<s> A </s></s> B </s>`.
- Special token, separator, and pad boxes default to `[0,0,0,0]`.
- If `return_overflowing_tokens=True`, pixel images are duplicated according to `overflow_to_sample_mapping`.

GPU/runtime inputs:

```text
input_ids:      [B, T]
bbox:           [B, T, 4], integer 0..1023 expected by embedding tables
attention_mask: [B, T], 1 for text tokens, 0 for pad
pixel_values:   [B, 3, H, W], NCHW, normally [B,3,224,224]
token_type_ids: optional [B,T], usually all zeros for public configs
```

Runtime packing:

- Model appends visual attention mask internally, all ones.
- Model appends visual bbox internally, including one visual CLS bbox followed by patch boxes.
- Text outputs remain at the front of the combined sequence, which makes token-classification slicing straightforward.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> window GEMM

Source pattern:

```text
Conv2d(C=3 -> H, kernel=patch_size, stride=patch_size, padding=0) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, kh=kw=16, stride=16) -> MatMul(weight_flat.T) -> BiasAdd -> [B, Hp*Wp, H]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input height and width divisible by patch size.
- Flatten order must match PyTorch NCHW Conv2d followed by row-major spatial flatten.

Shape equations:

- `Hp = H_img / patch_size`, `Wp = W_img / patch_size`, `Npatch = Hp * Wp`.
- `weight_flat = conv.weight.reshape(hidden_size, 3 * patch_size * patch_size)`.

Layout constraints:

- Direct NHWC translation is allowed only inside this local patch-extraction region if all consumers accept `[B,Npatch,H]`.
- Public `pixel_values` remains NCHW; a broader NHWC pass must rewrite Conv/window axes explicitly.

Failure cases:

- Nondivisible image shapes, non-default padding/dilation/groups, or dynamic `patch_size` not known at compile time.
- Any future config that passes `position_embedding` into `patch_embed` and relies on bicubic interpolation inside the patch module.

Parity sketch: compare patch embeddings before and after lowering for random `[B,3,224,224]` and representative weights.

### Rewrite: shared relative-bias precompute

Source pattern:

```text
encoder computes rel_pos and rel_2d_pos once, each layer adds same tensors
```

Replacement:

```text
PrecomputeBias(position_ids, bbox) -> AddBias inside each attention layer
```

Preconditions:

- `has_relative_attention_bias`/`has_spatial_attention_bias` unchanged across layers.
- Bias module weights are encoder-level shared tensors, not per-layer copies.
- `position_ids` and `bbox` fixed for the forward.

Failure cases:

- Future source with per-layer bias tables or mutation.

Parity sketch: compare bucket tensors and final `[B,heads,S,S]` bias exactly for random bbox ranges.

### Rewrite: embedding sum/final LayerNorm fusion

Source pattern:

```text
word + token_type + position + spatial -> LayerNorm -> optional concat visual -> LayerNorm
```

Replacement:

```text
FusedEmbeddingGatherAdd -> LayerNorm
```

Preconditions:

- Bbox coordinates are in range and spatial concat width equals hidden size.
- Token type vocab accepts all token type IDs.
- Preserve two LayerNorm placements when visual tokens are present: text embedding norm first, combined norm after concat.

Failure cases:

- `inputs_embeds` path bypasses word embedding gather.
- Missing text path or image-only path changes available tensors.

### Rewrite: no-layout-translation guard around combined sequence ABI

Source pattern:

```text
text [B,T,H] + visual [B,V,H] -> cat(dim=1) -> attention
```

Replacement:

```text
Protect sequence axis and hidden axis from vision-layout NHWC rewrites
```

Preconditions:

- Only image-local Conv/window extraction may use NHWC/channel-last internally.
- `cat(dim=1)`, `softmax(dim=-1)`, `bbox[:, :, i]`, and token slicing `[:, :T]` preserve source axes.

Failure cases:

- Any layout pass that rewrites sequence/hidden axes as if tensors were image maps.

Parity sketch: verify token-classification logits still ignore visual-token tail and QA logits still include it.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d/window GEMM + flatten transpose: important because every document image enters through this NCHW patch stem and NHWC can be useful only in this guarded local region.
- Relative/2D bias generation + attention score add: avoids repeated bucket/bias materialization overhead and keeps score-shape contracts explicit.
- Attention with per-head dense bias and CogView softmax: baseline attention is full `[S,S]`; preserving source math while fusing score/bias/mask/softmax is key.

Medium priority:

- Embedding gather/add + LayerNorm: many embedding sources feed one normalized token tensor.
- FFN dense + GELU + dense scheduling: standard encoder bottleneck, especially for large.
- Head slicing/projection fusion for token classification: avoid carrying visual token outputs into the head when only text logits are needed.

Lower priority:

- Image resize/rescale/normalize GPU fusion: useful only if preprocessing is moved from CPU.
- Visual bbox generation kernel: cheap and normally precomputable for fixed 224.
- Dropout removal/canonicalization for inference: simple graph cleanup.

## 11. Runtime staging plan

1. Parse config and reject unsupported combinations where `4*coordinate_size + 2*shape_size != hidden_size`.
2. Load base/large weights and implement text-only encoder parity with `visual_embed=False` or no `pixel_values`.
3. Implement patch embedding and visual-only encoder input path for `[B,3,224,224]`.
4. Implement combined text+image sequence construction, visual bbox/position IDs, and attention mask extension.
5. Implement 1D and 2D relative bias plus CogView attention parity for one layer, then all layers.
6. Add token classification, QA, and sequence classification heads. First useful target should be token classification over OCR/token boxes.
7. Add guarded Conv2d-to-GEMM and relative-bias precompute rewrites.
8. Add optimized attention backend if it supports dense per-head bias and CogView semantics; otherwise keep a parity backend.

Can be stubbed initially: Tesseract OCR, training losses, overflow handling, dynamic non-224 image support, and gated fine-tuned label maps.

## 12. Parity and validation plan

- Custom op tests: relative bucket exactness, 2D bias lookup shape/order, CogView softmax against source formula, visual bbox table for 14x14.
- Patch stem parity: Conv2d path vs GEMM lowering for base/large hidden sizes.
- Embedding parity: text embedding with known `input_ids`, `bbox`, `token_type_ids`, and generated position IDs; include pad-token position behavior.
- One-layer parity: random embeddings and masks, with/without relative and spatial bias.
- Encoder parity: after 1, 6, 12, and 24 layers depending on config.
- Combined input parity: verify total sequence `T + 197`, attention mask extension, final bbox concat, and token-classification text slice.
- Head parity: token classification logits `[B,T,num_labels]`, QA logits `[B,T+197]`, sequence logits `[B,num_labels]`.
- End-to-end processor parity: with `apply_ocr=False`, compare processor output fields from supplied words/boxes; OCR itself can be treated as CPU external.
- Recommended tolerances: fp32 `atol=1e-5, rtol=1e-5`; fp16/bf16 `atol=1e-2, rtol=1e-2`, with stricter isolated tests for integer bucket/box logic.

## 13. Performance probes

- CPU processor throughput: OCR on/off, tokenizer box expansion, overflow duplication.
- Image preprocessing throughput: resize/rescale/normalize CPU vs optional GPU path.
- Patch stem throughput: Conv2d vs GEMM lowering, NCHW baseline vs guarded NHWC local kernel.
- Encoder throughput by sequence length: text-only, image-only 197 tokens, and text+image around 709 tokens.
- Relative-bias generation cost: bucket lookup/materialization vs precomputed static plans.
- Attention backend comparison: eager dense bias + CogView softmax vs custom fused dense-bias attention.
- Batch-size sweep for base and large.
- Memory probe for `[B,heads,S,S]` score/bias tensors.
- Head-specific output cost: token classification text-slice only vs QA full-sequence logits.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Tesseract OCR implementation inside DinoML runtime; treat as CPU/data-pipeline dependency first.
- Dynamic image sizes beyond 224x224 until visual bbox and absolute position behavior are parity-tested.
- Masked language modeling or masked image modeling heads, because they are not implemented in the inspected native source.
- Multi-GPU tensor parallel and quantization.
- Beam search, KV cache, generation controllers: not applicable.
- Gated fine-tuned model label-map exactness until HF access resolves the 401 configs.

## 15. Final implementation checklist

- [ ] Parse `LayoutLMv3Config` including source defaults for omitted `patch_size` and `num_channels`.
- [ ] Validate spatial embedding width equation.
- [ ] Load text, visual, encoder, and task-head weights.
- [ ] Implement text embedding with bbox spatial embeddings.
- [ ] Implement visual patch embedding and visual CLS/position/norm path.
- [ ] Implement text+visual concat, visual bbox table, position IDs, and mask extension.
- [ ] Implement relative 1D bucket and 2D bbox bias.
- [ ] Implement CogView softmax attention with dense per-head bias.
- [ ] Implement token classification, QA, and sequence classification heads.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite.
- [ ] Add no-layout-translation guards for sequence concat, bbox axes, and token slicing.
- [ ] Add parity tests for custom position/bias math.
- [ ] Add one-layer, full-encoder, and head parity tests.
- [ ] Benchmark processor, patch stem, bias generation, encoder attention, and heads separately.
