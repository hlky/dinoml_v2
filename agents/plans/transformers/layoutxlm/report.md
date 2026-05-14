# LayoutXLM DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/layoutxlm-base plus accessible public fine-tunes/mirrors listed below
Config source: local Transformers source defaults plus HF raw config/preprocessor/tokenizer JSON snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/layoutxlm/configuration_layoutxlm.py
- X:/H/transformers/src/transformers/models/layoutxlm/modular_layoutxlm.py
- X:/H/transformers/src/transformers/models/layoutxlm/processing_layoutxlm.py
- X:/H/transformers/src/transformers/models/layoutxlm/tokenization_layoutxlm.py
- X:/H/transformers/src/transformers/models/layoutxlm/__init__.py
- X:/H/transformers/src/transformers/models/layoutlmv2/modeling_layoutlmv2.py, as source-proven neural reference
- X:/H/transformers/src/transformers/models/layoutlmv2/image_processing_layoutlmv2.py, as source-proven image processor reference
Any missing files or assumptions: There is no modeling_layoutxlm.py and no AutoModel mapping for model_type "layoutxlm" in this commit. LayoutXLM neural runtime facts below apply only when a checkpoint config routes to LayoutLMv2 source, which the official microsoft/layoutxlm-base config does.
```

Snapshots are under `agents/plans/transformers/layoutxlm/_sources/`.

Accessible configs/preprocessor assets:

| Repo | Source status | Runtime class signal |
|---|---|---|
| [microsoft/layoutxlm-base](https://huggingface.co/microsoft/layoutxlm-base) | official, accessible | `model_type: layoutlmv2`, tokenizer class `XLMRobertaTokenizer` in old config |
| [beomus/layoutxlm](https://huggingface.co/beomus/layoutxlm) | public fine-tune/mirror | `LayoutLMv2ForTokenClassification`, tokenizer class `LayoutXLMTokenizer` |
| [nielsr/layoutxlm-finetuned-xfund-fr](https://huggingface.co/nielsr/layoutxlm-finetuned-xfund-fr) | public fine-tune | `LayoutLMv2ForTokenClassification`, `LayoutXLMProcessor` |
| [Guigadal/layoutxlm-finetuned-xfund-es](https://huggingface.co/Guigadal/layoutxlm-finetuned-xfund-es) | public fine-tune | `LayoutLMv2ForTokenClassification`, `LayoutXLMProcessor` |
| [taprosoft/layoutxlm-no-visual](https://huggingface.co/taprosoft/layoutxlm-no-visual) | public custom variant | `LayoutLMv2ModelNoVisual`, not implemented in inspected source |
| [estyle/layoutxlm-base](https://huggingface.co/estyle/layoutxlm-base) | old mirror | `model_type: layout_xlm`, not implemented in inspected source |

401/availability gaps:

- [microsoft/layoutxlm-large](https://huggingface.co/microsoft/layoutxlm-large)
- [microsoft/layoutxlm-base-finetuned-funsd](https://huggingface.co/microsoft/layoutxlm-base-finetuned-funsd)
- [microsoft/layoutxlm-base-finetuned-xfund](https://huggingface.co/microsoft/layoutxlm-base-finetuned-xfund)
- [microsoft/layoutxlm-base-finetuned-docvqa](https://huggingface.co/microsoft/layoutxlm-base-finetuned-docvqa)

Access would confirm official large/fine-tuned label maps and processor metadata. It is not needed to establish the current in-library source contract.

## 2. High-level architecture

In this Transformers commit, `src/transformers/models/layoutxlm` is a document OCR/layout preprocessing and multilingual tokenizer family, not an independently implemented PyTorch model family. The source-proven delegation points are:

- `LayoutXLMConfig` in `modular_layoutxlm.py` subclasses `LayoutLMv2Config`; the generated `configuration_layoutxlm.py` materializes the same fields with `model_type = "layoutxlm"`.
- `LayoutXLMProcessor` is the LayoutLMv2 processor shape with LayoutXLM tokenizer coupling and output names `input_ids`, `bbox`, `attention_mask`, `image`.
- Auto image processor maps `layoutxlm` to `LayoutLMv2ImageProcessor` / PIL equivalent.
- Auto modeling maps include `layoutlmv2` but not `layoutxlm`.
- Public official LayoutXLM base config uses `model_type: layoutlmv2`, so its neural graph is LayoutLMv2 source, with multilingual tokenizer assets.

Dataflow for supported public checkpoints:

```text
document image + OCR/supplied words/boxes
  -> LayoutLMv2 image processor resize/BGR/OCR
  -> LayoutXLM tokenizer word-to-subword packing + bbox expansion
  -> LayoutLMv2 multimodal encoder over text tokens + 49 visual tokens
  -> token classification / QA / sequence classification heads
```

Primary DinoML target: route LayoutXLM checkpoints with `model_type: layoutlmv2` through the LayoutLMv2 encoder audit, while owning the LayoutXLM-specific tokenizer/processor ABI and checkpoint admission rules.

## 3. Important config dimensions

Source defaults in `LayoutXLMConfig` match LayoutLMv2 defaults unless overridden:

| Field | Source default | Official/accessed LayoutXLM base configs | Notes |
|---|---:|---:|---|
| `vocab_size` | 30522 | 250002 | XLM-R / SentencePiece-derived multilingual vocab, not BERT WordPiece |
| `hidden_size` | 768 | 768 | base width |
| `num_hidden_layers` | 12 | 12 | encoder layers |
| `num_attention_heads` | 12 | 12 | MHA, head dim 64 |
| `intermediate_size` | 3072 | 3072 | GELU FFN |
| `max_position_embeddings` | 512 | 514 | public LayoutXLM base/fine-tunes use 514 |
| `type_vocab_size` | 2 | 1 | XLM-R style all-zero token types in common configs |
| `pad_token_id` | 0 | 1 | XLM-R pad id |
| `layer_norm_eps` | 1e-12 | 1e-5 | public configs override |
| `max_2d_position_embeddings` | 1024 | 1024 | bbox indices must stay in range |
| `coordinate_size` / `shape_size` | 128 / 128 | 128 / 128 | bbox concat width = 768 |
| `image_feature_pool_shape` | `[7,7,256]` | `[7,7,256]` | 49 visual tokens |
| `fast_qkv` | true | false | public LayoutXLM base/fine-tunes use separate Q/K/V |
| `has_relative_attention_bias` | true | false | public LayoutXLM base/fine-tunes disable 1D relative bias |
| `has_spatial_attention_bias` | true | false | public LayoutXLM base/fine-tunes disable 2D attention bias |
| `has_visual_segment_embedding` | false | true | public LayoutXLM base/fine-tunes enable visual segment embedding |

Representative config sweep:

| Repo | model_type | architecture | labels | processor/OCR | operator-significant differences |
|---|---|---|---:|---|---|
| `microsoft/layoutxlm-base` | `layoutlmv2` | omitted/base | n/a | `LayoutLMv2FeatureExtractor`, OCR true | official base, no relative/spatial attention bias, separate Q/K/V, XLM-R vocab |
| `beomus/layoutxlm` | `layoutlmv2` | token classification | 7 | OCR true | explicit Detectron2 config, `LayoutXLMTokenizer` |
| `nielsr/layoutxlm-finetuned-xfund-fr` | `layoutlmv2` | token classification | 7 | OCR true, `ocr_lang: fra` | French OCR language metadata, label map |
| `Guigadal/layoutxlm-finetuned-xfund-es` | `layoutlmv2` | token classification | 7 | OCR true | newer `size: {height,width}` processor schema |
| `taprosoft/layoutxlm-no-visual` | `layoutlmv2` | `LayoutLMv2ModelNoVisual` | n/a | no preprocessor snapshot | custom architecture not implemented by official source |
| `estyle/layoutxlm-base` | `layout_xlm` | omitted | n/a | missing | historical model_type not recognized by current source |

## 3a. Family variation traps

- Do not instantiate `AutoModel` for `model_type: layoutxlm` in this commit; no modeling mapping exists.
- Do route official `microsoft/layoutxlm-base` through LayoutLMv2 because its config says `model_type: layoutlmv2`.
- `LayoutXLMConfig` source defaults are not the same as official base checkpoint values: QKV packing and relative/spatial attention bias flags differ.
- Multilingual tokenizer coupling changes vocab size, pad id, token type usage, and special-token boxes versus LayoutLMv2 English WordPiece checkpoints.
- `type_vocab_size=1` means any supplied token type id other than 0 is invalid for common LayoutXLM checkpoints, even though source defaults allow 2.
- Public configs include legacy `output_past` / `use_cache`; inspected LayoutLMv2 encoder source has no KV cache.
- `LayoutLMv2ModelNoVisual` and `model_type: layout_xlm` are custom/historical variants; reject or route to a separate audit.
- The visual branch is source-owned by Detectron2 through LayoutLMv2, not by `layoutxlm` source.
- OCR is a CPU/data-pipeline dependency. It is optional only when processor/config or caller supplies words/boxes.

## 4. Operator coverage checklist

LayoutXLM-owned preprocessing/tokenizer operators:

- Optional Tesseract OCR through LayoutLMv2 image processor: words plus normalized `[x0,y0,x1,y1]` boxes on a 0..1000 scale.
- Image resize to 224x224 by default, RGB-to-BGR channel flip, output key renamed to `image`.
- SentencePiece/Unigram tokenizer backend with XLM-R style special tokens.
- Pretokenized word encoding, pair handling, overflow flattening, and `overflow_to_sample_mapping`.
- Word bbox expansion to subword bbox.
- Special bbox assignment: CLS/PAD `[0,0,0,0]`, SEP `[1000,1000,1000,1000]`.
- Optional token labels with `only_label_first_subword=True` and padding label `-100`.

Neural operators for supported checkpoints are LayoutLMv2-owned:

- Embeddings: token, absolute 1D position, token type, 2D x/y/h/w bbox tables.
- Visual backbone: Detectron2 ResNet-FPN, pixel normalization, adaptive average pool to 7x7, flatten/transpose to 49 tokens, `Linear(256 -> hidden_size)`.
- Encoder: bidirectional MHA, separate Q/K/V for common LayoutXLM configs, residual LayerNorm, GELU FFN.
- Optional, config-gated relative 1D and 2D attention bias. Source default has them; common LayoutXLM public configs disable them.
- Heads: token classification, question answering, sequence classification when checkpoint routes to the corresponding LayoutLMv2 class.

No generation/cache ops are required.

## 5. Layer/block breakdown

Processor/tokenizer packing:

```text
image -> LayoutLMv2ImageProcessor -> image: [B,3,224,224] BGR
words/boxes -> LayoutXLMTokenizer -> input_ids: [B,L], bbox: [B,L,4], attention_mask: [B,L]
overflow chunks -> duplicate image by overflow_to_sample_mapping
```

LayoutLMv2 neural path for official base-style checkpoints:

```text
text_emb = word + position + spatial_bbox + token_type
visual_emb = Detectron2FPN(image)["p2"] -> pool 7x7 -> Linear(256,H)
visual_emb = visual + visual_position + visual_bbox [+ visual_segment]
x = concat(text_emb, visual_emb, dim=1)  # [B, L_text + 49, H]

repeat N layers:
  q = Linear(H,H)(x)
  k = Linear(H,H)(x)
  v = Linear(H,H)(x)
  scores = (q / sqrt(64)) @ k.T
  if enabled: scores += rel_1d + rel_2d
  scores = masked_fill(scores, padding_mask_min)
  context = softmax(scores, fp32) @ v
  x = LayerNorm(Linear(context) + residual)
  x = LayerNorm(Linear(GELU(Linear(x))) + residual)
```

For public LayoutXLM base/fine-tunes, `fast_qkv=false`, so do not use the packed QKV rewrite unless a different config explicitly enables it.

## 6. Attention requirements

- Attention type: bidirectional dense encoder self-attention.
- Heads: MHA, no GQA/MQA. Common base width is 12 heads x 64.
- Sequence length: `L_total = L_text + 49`; common max text length is effectively 514 in LayoutXLM configs.
- Masking: text attention mask plus all-ones visual mask; converted to additive min mask.
- Relative bias: source can add T5-style 1D relative bias and bbox-derived 2D spatial bias, but official LayoutXLM public configs inspected set both flags false.
- Cache: none. Ignore/reject legacy `output_past` and `use_cache`.
- Flash/SDPA opportunity: ordinary SDPA is straightforward when both relative-bias flags are false; arbitrary-bias attention support is needed for source-default `LayoutXLMConfig()` parity.

## 7. Position encoding and custom math

Absolute 2D bbox embedding, inherited from LayoutLMv2:

```python
spatial = concat([
    x_embed(bbox[..., 0]),
    y_embed(bbox[..., 1]),
    x_embed(bbox[..., 2]),
    y_embed(bbox[..., 3]),
    h_embed(bbox[..., 3] - bbox[..., 1]),
    w_embed(bbox[..., 2] - bbox[..., 0]),
], dim=-1)
```

Guard requirements:

- Bbox coordinates and derived width/height must index `[0, max_2d_position_embeddings)`.
- Common coordinate scale is 0..1000 from the processor.
- For LayoutXLM common configs, relative attention bias can be skipped because config disables it; source-default configs still need LayoutLMv2 relative bucket math.

## 8. Preprocessing and input packing

CPU/data-pipeline boundary:

- OCR, if enabled, is Tesseract via the LayoutLMv2 image processor. It is not a model graph op.
- The processor disallows caller-provided `boxes` or `word_labels` when OCR is enabled.
- OCR language can be checkpoint metadata, e.g. `ocr_lang: fra` for the nielsr French XFUN fine-tune.
- Images are resized and channel-flipped before model execution; output is NCHW BGR.
- Tokenization expects pretokenized words in document order. Pair mode treats first sequence as question text and second sequence as OCR words; first-sequence words get pad boxes.
- Overflow handling flattens variable overflow chunks and duplicates the source image for each chunk.

GPU/runtime boundary:

- Inputs for neural parity are `input_ids`, `bbox`, `attention_mask`, optional `token_type_ids`, and `image`.
- `bbox` should be treated as required for LayoutLMv2 parity even if some signatures mark it optional.
- Model output includes visual tokens after text tokens. Token classification and QA slice only the first `L_text` positions.

## 9. Graph rewrite / lowering opportunities

### Rewrite: LayoutXLM checkpoint admission to LayoutLMv2 graph

Source pattern:

```text
config.model_type == "layoutlmv2"
tokenizer/processor assets identify LayoutXLM
```

Replacement:

```text
Use LayoutLMv2 neural lowering + LayoutXLM tokenizer/processor ABI.
```

Preconditions:

- Config `model_type` is exactly `layoutlmv2`.
- Architecture is omitted or one of official LayoutLMv2 classes.
- Reject custom `LayoutLMv2ModelNoVisual` unless separately audited.

Parity test sketch: load `microsoft/layoutxlm-base` config and verify the DinoML admission route selects LayoutLMv2 operator coverage while retaining vocab size 250002 and LayoutXLM processor metadata.

### Rewrite: separate Q/K/V to provider QKV group

Source pattern for common LayoutXLM configs:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement: group three GEMMs or fuse into a single provider call only if weight packing preserves exact output order.

Preconditions:

- `fast_qkv == false`.
- All three projections have equal `in/out = hidden_size`.
- Biases are present in the LayoutLMv2 separate projection path.

Failure cases: source-default `fast_qkv=true` uses packed `qkv_linear` plus q/v-only learned bias and needs the LayoutLMv2 packed rewrite instead.

### Rewrite: skip relative-bias materialization for official LayoutXLM base

Source pattern: `has_relative_attention_bias=false` and `has_spatial_attention_bias=false`.

Replacement: use ordinary dense bidirectional attention without additive learned relative bias.

Preconditions: both config flags are explicitly false after defaults are resolved.

Failure cases: manually constructed `LayoutXLMConfig()` source defaults set both true; do not skip bias there.

### Rewrite: OCR/image preprocessing cache

Source pattern: OCR words/boxes and resized BGR image are deterministic for an input page and processor settings.

Replacement: cache preprocessor outputs outside the compiled neural graph.

Preconditions: same source image bytes, OCR language/config, resize settings, tokenizer version, and max length/stride.

Failure cases: stochastic or external OCR, changed Tesseract version/config, or different overflow settings.

## 10. Kernel fusion candidates

Highest priority:

- LayoutLMv2 encoder path without relative bias for public LayoutXLM base/fine-tunes: ordinary SDPA or fused attention is easier than LayoutLMv2 defaults.
- Detectron2 visual backbone, inherited from LayoutLMv2: likely the largest non-attention runtime cost.
- Embedding sum + LayerNorm for text/layout and visual/layout embeddings.

Medium priority:

- Separate Q/K/V grouped GEMM for `fast_qkv=false` LayoutXLM checkpoints.
- GELU FFN block fusion.
- Token-classification/QA slicing plus final linear heads.

Lower priority:

- OCR and tokenizer are CPU/data-pipeline work; optimize with caching before considering runtime ownership.
- Relative 1D/2D bias kernels for source-default `LayoutXLMConfig()` are lower priority than public checkpoint parity because representative public configs disable them.

## 11. Runtime staging plan

1. Add admission logic for LayoutXLM assets: if config `model_type=layoutlmv2`, route to LayoutLMv2 lowering with LayoutXLM tokenizer/processor metadata.
2. Reject `model_type=layoutxlm` for neural model loading in this Transformers commit unless a caller explicitly requests config-only/tokenizer-only behavior.
3. Reject known custom variants (`layout_xlm`, `LayoutLMv2ModelNoVisual`) until separately audited.
4. Implement/validate LayoutXLM tokenizer/processor fixtures: special boxes, subword bbox expansion, overflow image duplication, OCR-disabled supplied boxes.
5. Reuse LayoutLMv2 encoder parity for `microsoft/layoutxlm-base`; ensure overrides are honored: vocab 250002, pad id 1, max positions 514, eps 1e-5, type vocab 1, no relative/spatial bias, visual segment true.
6. Add token classification fine-tune parity using an accessible XFUN checkpoint.
7. Add processor OCR-language metadata handling as data-pipeline configuration, not graph ops.

## 12. Parity and validation plan

- Config admission tests:
  - official `microsoft/layoutxlm-base` routes to LayoutLMv2 graph.
  - source-default `LayoutXLMConfig()` is config-only or rejected for model lowering due no modeling class.
  - `layout_xlm` and `LayoutLMv2ModelNoVisual` are rejected with actionable messages.
- Tokenizer tests:
  - special token layout `<s> A </s>` and pair `<s> A </s></s> B </s>`.
  - all token type ids are zero for LayoutXLM tokenizer.
  - bbox expansion matches word ids and overflow sample mapping.
  - labels follow first-subword-only behavior.
- Processor tests:
  - OCR-enabled rejects caller boxes/labels.
  - OCR-disabled accepts supplied words/boxes and returns `image` key.
  - image duplication for overflow chunks.
- Neural tests:
  - reuse LayoutLMv2 single-block/full-encoder tests with LayoutXLM config overrides.
  - compare token classification logits for accessible XFUN fine-tune.
  - recommended fp32 tolerance `1e-5` to `3e-5`; visual branch may need looser tolerance if Detectron2 lowering differs.

## 13. Performance probes

- OCR throughput by language/config, separate from model runtime.
- Tokenizer throughput and overflow chunk count by page text length.
- Preprocessed image cache hit rate.
- LayoutLMv2 visual backbone throughput for LayoutXLM configs.
- Encoder attention speed with no relative/spatial bias versus LayoutLMv2 source-default bias path.
- Batch and text-length sweeps: `L_text` 128/256/512 plus 49 visual tokens.
- End-to-end pages/sec with OCR disabled and OCR enabled.

## 14. Skip/defer list

- Training losses, gradient checkpointing, and SyncBatchNorm conversion.
- Tesseract implementation inside DinoML GPU runtime.
- Official gated fine-tuned checkpoints until access is available.
- Historical `model_type: layout_xlm` mirror support.
- Custom no-visual `LayoutLMv2ModelNoVisual`.
- Generation, KV cache, beam search, speculative decoding: not applicable.
- Source-default `model_type: layoutxlm` neural loading unless Transformers adds a modeling mapping or DinoML defines an explicit alias policy.

## 15. Final implementation checklist

- [ ] Add LayoutXLM admission policy: processor/tokenizer family plus LayoutLMv2 neural route when config says `layoutlmv2`.
- [ ] Reject unsupported `layoutxlm`, `layout_xlm`, and custom no-visual model bodies for compiled neural runtime.
- [ ] Parse LayoutXLM tokenizer config: vocab size, pad id, special token boxes, first-subword label policy.
- [ ] Parse processor config: OCR enabled, OCR language/config, resize size schema variants.
- [ ] Add processor/tokenizer fixtures for bbox expansion, pair/question packing, overflow image duplication, and OCR-disabled supplied boxes.
- [ ] Reuse LayoutLMv2 operator coverage with LayoutXLM config overrides.
- [ ] Add no-relative-bias/no-spatial-bias attention parity for `microsoft/layoutxlm-base`.
- [ ] Add token classification parity with accessible XFUN-style checkpoint.
- [ ] Benchmark OCR, tokenization, visual backbone, encoder, and end-to-end pages/sec separately.
