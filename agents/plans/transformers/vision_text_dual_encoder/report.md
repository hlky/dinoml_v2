# VisionTextDualEncoder DinoML Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout X:/H/transformers
Model id: family-level audit for model_type="vision-text-dual-encoder"; representative checkpoints listed below
Config source: local configuration source plus Hugging Face config.json snapshots in _sources/
Source files inspected:
- X:/H/transformers/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
- X:/H/transformers/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py
- X:/H/transformers/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py
- X:/H/transformers/src/transformers/processing_utils.py for inherited processor ABI
Any missing files or assumptions: no remote-code body inspected; delegated vision/text backbones are owned by their own Transformers families and must be admitted through separate audits.
```

Pinned upstream source URLs for the inspected commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/processing_utils.py

Representative config snapshots:

| Checkpoint | Source | Notes |
| --- | --- | --- |
| `hf-internal-testing/tiny-random-VisionTextDualEncoderModel-vit-bert` | `_sources/hf-internal-testing_tiny-random-VisionTextDualEncoderModel-vit-bert_config.json` | Tiny ViT+BERT debug checkpoint plus preprocessor snapshot. |
| `ljnlonoljpiljm/webssl-mae700m-full2b-224-bert-base-uncased` | `_sources/ljnlonoljpiljm_webssl-mae700m-full2b-224-bert-base-uncased_config.json` | Large ViT-like vision encoder with BERT text encoder. |
| `koclip/koclip-base-pt` | `_sources/koclip_koclip-base-pt_config.json` | CLIP vision sub-config plus RoBERTa-like text. |
| `flavour/vtde-dinov2-small-multilingual-e5-small` | `_sources/flavour_vtde-dinov2-small-multilingual-e5-small_config.json` | DINOv2 vision plus multilingual E5/BERT-style text. |
| `ljnlonoljpiljm/CLIP-ViT-H-14-laion2B-s32B-b79K-384-xlm-roberta-large-tv` | `_sources/ljnlonoljpiljm_CLIP-ViT-H-14-laion2B-s32B-b79K-384-xlm-roberta-large-tv_config.json` | Larger CLIP vision plus XLM-R text, projection dim 1024. |
| `fabnem/UltraSoundCLIP` checkpoint `checkpoint-1650` | `_sources/fabnem_UltraSoundCLIP_checkpoint-1650_config.json` | CLIP vision plus BERT/SPLADE-origin text config. |

Primary runtime target: inference-time image/text retrieval scoring for already-trained `VisionTextDualEncoderModel` checkpoints. Training-only contrastive loss is deferred.

## 2. High-level architecture

This family is a wrapper/composition model:

```text
image processor + tokenizer
  -> pixel_values + input_ids/attention_mask/token_type_ids/position_ids
  -> delegated vision AutoModel/CLIPVisionModel -> vision pooler_output
  -> delegated text AutoModel -> text pooler_output
  -> wrapper-owned bias-free projection heads
  -> L2 normalize each projected embedding
  -> exp(logit_scale) * text_embeds @ image_embeds.T
  -> logits_per_text and transposed logits_per_image
```

Stage decomposition:

- CPU/data pipeline: tokenizer and image processor. The wrapper processor only merges the selected tokenizer/image processor outputs.
- Vision encoder: delegated to `CLIPVisionModel` when loading from a full CLIP config, otherwise `AutoModel` from `vision_config`.
- Text encoder: delegated to `AutoModel` from `text_config`.
- Wrapper-owned projection/similarity: `visual_projection`, `text_projection`, `logit_scale`, L2 normalization, similarity matmul, and orientation of output matrices.
- Independently cacheable outputs: projected and normalized image embeddings can be cached for image indexing; projected and normalized text embeddings can be cached for query batching. These are embedding caches, not KV caches.

## 3. Important config dimensions

Wrapper-level source defaults:

| Field | Effective source behavior |
| --- | --- |
| `model_type` | `vision-text-dual-encoder` |
| `projection_dim` | Default 512, checkpoint-overridable. |
| `logit_scale_init_value` | Default 2.6592, stored as scalar parameter and exponentiated at runtime. |
| `vision_config` | Required; no default-only construction. |
| `text_config` | Required; no default-only construction. |
| Projection biases | Always absent: both wrapper projections are `nn.Linear(..., bias=False)`. |
| Attention implementation | Wrapper advertises flash-attn/SDPA support, but actual attention belongs to delegated backbones. |
| Cache support | No wrapper KV cache. Any `use_cache` field in a sub-config is delegated and usually irrelevant for encoder-only retrieval. |

Representative config sweep:

| Checkpoint | Proj | Vision owner | Vision hidden/layers/heads | Image/patch | Text owner | Text hidden/layers/heads | Vocab |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| tiny random ViT+BERT | 512 | `vit` | 32 / 5 / 4 | 30 / 2 | `bert` | 32 / 5 / 4 | 1124 |
| webssl MAE700M + BERT | 512 | `vit` | 1280 / 32 / 16 | 224 / 14 | `bert` | 768 / 12 / 12 | 30522 |
| KoCLIP base | 512 | `clip_vision_model` | 768 / 12 / 12 | 224 / 32 | `roberta` | 1024 / 24 / 16 | 32000 |
| DINOv2 small + multilingual E5 | 384 | `dinov2` | 384 / 12 / 6 | 518 / 14 | `bert` | 384 / 12 / 12 | 250037 |
| CLIP ViT-H/14 + XLM-R large | 1024 | `clip_vision_model` | 1280 / 32 / 16 | 224 / 14 | `xlm-roberta` | 1024 / 24 / 16 | 250002 |
| UltraSoundCLIP checkpoint | 512 | `clip_vision_model` | 768 / 12 / 12 | 224 / 32 | `bert` | 768 / 12 / 12 | 30522 |

Other observed variation from configs:

| Field | Observed values |
| --- | --- |
| Text activation | `gelu` in sampled configs. |
| Vision activation | `gelu` or `quick_gelu`, depending on delegated vision owner. |
| Text `layer_norm_eps` | `1e-12` or `1e-5`. |
| Vision `layer_norm_eps` | `1e-12`, `1e-6`, or `1e-5`. |
| Text `type_vocab_size` | 1, 2, or 16. |
| Vision `qkv_bias` | Present/true for ViT/DINOv2-style configs; not always present in CLIP vision sub-config snapshots. |

## 3a. Family variation traps

- This report does not own fixed operator coverage for ViT, CLIPVision, SigLIPVision, ChineseCLIPVision, DINOv2, BERT, RoBERTa, XLM-R, E5, or SPLADE-derived text bodies. Those are delegated model families.
- The wrapper assumes both delegated models return a usable `pooler_output` at tuple/dataclass index 1. DinoML should reject sub-configs or delegated classes without a stable pooled output contract unless a separate audited pooling adapter is added.
- `configuration_vision_text_dual_encoder.py` maps `clip_vision_model`, `chinese_clip_vision_model`, and `siglip_vision_model` directly to their vision config classes. For other configs it uses `AutoConfig.for_model`; if the loaded config itself has `vision_config`, the wrapper extracts that nested `vision_config`.
- `from_vision_text_pretrained()` treats full CLIP configs specially: if `vision_config.model_type == "clip"`, it loads only `CLIPVisionModel` with the nested `vision_config`. The source comment explicitly leaves pretrained CLIP projection reuse unresolved; wrapper projection weights are newly initialized in this constructor path.
- Checkpoint projection dimensions are not inferred from hidden sizes. They are separate wrapper config fields and projection weight shapes must match `visual_projection.weight: [projection_dim, vision_hidden]` and `text_projection.weight: [projection_dim, text_hidden]`.
- Similarity orientation is source-defined: `logits_per_text = text_embeds @ image_embeds.T * scale`, shape `[B_text, B_image]`; `logits_per_image = logits_per_text.T`, shape `[B_image, B_text]`.
- `get_text_features()` and `get_image_features()` project the branch pooler outputs but do not L2-normalize or apply logit scale. The full `forward()` does normalize before similarity.
- Processor behavior is inherited from `ProcessorMixin`; this family adds no custom placeholder tokens, scatter, packing, or modality IDs.
- Image tensors from standard Transformers image processors are normally PyTorch-style `pixel_values` in `[B, C, H, W]`. NHWC/channel-last is an optimization candidate only inside an audited vision backbone region, not a semantic wrapper rewrite.
- Text special-token layout, padding side, segment IDs, and position IDs are tokenizer/backbone contracts. The wrapper simply forwards `input_ids`, `attention_mask`, `token_type_ids`, and `position_ids`.
- Training contrastive loss uses cross entropy on `logits_per_text` and its transpose. First inference integration can skip it.

## 4. Operator coverage checklist

Wrapper-owned required ops:

- Tensor/layout ops:
  - Pooler output selection from each delegated model output.
  - Matrix transpose for image embedding matrix in similarity.
  - Matrix transpose of `logits_per_text` for `logits_per_image`.
- Neural primitives:
  - `Linear(vision_hidden -> projection_dim, bias=False)`.
  - `Linear(text_hidden -> projection_dim, bias=False)`.
  - L2 norm over last dimension: square/sum/sqrt or vector norm.
  - Divide projected embeddings by norm with `keepdim=True`.
  - Scalar `exp(logit_scale)`.
  - GEMM/BMM-equivalent similarity: `[B_text, P] @ [P, B_image] -> [B_text, B_image]`.
  - Scalar multiply of similarity matrix by logit scale.
- Attention primitives:
  - None owned by the wrapper. Attention belongs to the admitted vision/text backbones.
- Preprocessing-coupled ops:
  - Processor merge of tokenizer and image processor outputs; no wrapper-specific tensor transform.
- Scatter/indexed update:
  - Not required. No placeholder embedding stitch.
- Position/cache/generation:
  - No wrapper KV cache, decode loop, RoPE, or generation controller.
- Optional training-only:
  - Cross entropy against diagonal labels for `logits_per_text` and `logits_per_text.T`.

Delegated operators depend entirely on the admitted sub-config pair. For sampled checkpoints, expect ViT/CLIP/DINOv2 vision encoder ops and BERT/RoBERTa/XLM-R text encoder ops, but those must be sourced from their own audits.

## 5. Layer/block breakdown

Wrapper forward path:

```text
vision_outputs = vision_model(pixel_values, output_attentions, output_hidden_states, return_dict)
text_outputs = text_model(input_ids, attention_mask, token_type_ids, position_ids, output_attentions, output_hidden_states, return_dict)

image_embeds = vision_outputs[1]
image_embeds = Linear(vision_hidden, projection_dim, bias=False)(image_embeds)

text_embeds = text_outputs[1]
text_embeds = Linear(text_hidden, projection_dim, bias=False)(text_embeds)

image_embeds = image_embeds / norm(image_embeds, dim=-1, keepdim=True)
text_embeds = text_embeds / norm(text_embeds, dim=-1, keepdim=True)

logit_scale = exp(scalar_parameter)
logits_per_text = matmul(text_embeds, transpose(image_embeds)) * logit_scale
logits_per_image = transpose(logits_per_text)
```

Shapes:

```text
pixel_values: [B_image, C, H, W] for typical image processors
input_ids: [B_text, S]
attention_mask: [B_text, S]
token_type_ids: optional [B_text, S]
position_ids: optional [B_text, S]
vision pooler_output: [B_image, vision_hidden]
text pooler_output: [B_text, text_hidden]
image_embeds/text_embeds after projection: [B_image or B_text, projection_dim]
logits_per_text: [B_text, B_image]
logits_per_image: [B_image, B_text]
```

Branch feature methods:

- `get_text_features(...)`: forwards to text model with `return_dict=True`, replaces `pooler_output` with projected pooled output, and returns the full text output object. No normalization.
- `get_image_features(...)`: forwards to vision model with `return_dict=True`, replaces `pooler_output` with projected pooled output, and returns the full vision output object. No normalization.

## 6. Attention requirements

No attention is wrapper-owned for the primary target. There is no autoregressive prefill/decode, no wrapper KV cache, no causal mask, and no cross-attention between image and text branches.

Backbone attention must be admitted separately:

- Vision branch may use noncausal encoder self-attention, patch embeddings, class tokens, and backbone-specific pooling.
- Text branch usually uses noncausal encoder self-attention with tokenizer padding masks and optional token type/position embeddings.
- `_supports_flash_attn` and `_supports_sdpa` on the wrapper are dispatch metadata, not proof that every delegated sub-model pair is supported by DinoML optimized attention.

Embedding caches:

- Image embeddings after projection+normalization can be precomputed for retrieval indices.
- Text embeddings after projection+normalization can be batched independently.
- These caches are dense embedding caches with shape `[N, projection_dim]`; they are not layer KV caches.

## 7. Position encoding and custom math

Wrapper custom math is limited to CLIP-style normalization and logit scaling:

```python
def wrapper_similarity(text_pool, image_pool, text_w, image_w, logit_scale):
    text = text_pool @ text_w.T
    image = image_pool @ image_w.T
    text = text / text.norm(dim=-1, keepdim=True)
    image = image / image.norm(dim=-1, keepdim=True)
    logits_per_text = (text @ image.T) * logit_scale.exp()
    return logits_per_text.T, logits_per_text
```

No wrapper RoPE, ALiBi, relative bias, or position table exists. Position embeddings are delegated to the text and vision encoders.

Numerical notes:

- Norm is over `projection_dim`.
- Source does not clamp norm or logit scale in this wrapper.
- `logit_scale` is a learned scalar parameter initialized from `logit_scale_init_value`.

## 8. Preprocessing and input packing

`VisionTextDualEncoderProcessor` only calls `ProcessorMixin(image_processor, tokenizer)`. The inherited `ProcessorMixin.__call__` forwards `images` to `image_processor`, `text` to `tokenizer`, merges their returned dictionaries, and returns a `BatchFeature`.

Expected runtime tensors:

| Tensor | Owner | Shape/meaning |
| --- | --- | --- |
| `pixel_values` | Image processor/backbone | Usually `[B_image, C, H, W]`, dtype float, normalized/resized by selected image processor. |
| `input_ids` | Tokenizer/text backbone | `[B_text, S]`, integer token IDs. |
| `attention_mask` | Tokenizer/text backbone | Optional `[B_text, S]`; forwarded directly. |
| `token_type_ids` | Tokenizer/text backbone | Optional `[B_text, S]`; meaningful for BERT-style configs, often absent or all zeros for RoBERTa-like configs. |
| `position_ids` | Caller/tokenizer/text backbone | Optional `[B_text, S]`; forwarded directly. |

Tiny preprocessor snapshot confirms an image-processor-driven shape path:

- `crop_size` 30x30
- `size.shortest_edge` 30
- `do_resize`, `do_center_crop`, `do_normalize`, `do_rescale` true
- `image_mean` and `image_std` are 3-element RGB lists

Branch contracts for retrieval:

- Image branch input can be batched independently from text branch input.
- Text and image batch sizes may differ at inference; final similarity is rectangular.
- Projection+normalization output is cacheable per branch.
- No wrapper token placeholder, boolean scatter, `cu_seqlens`, grid metadata, modality token type IDs, or packed sequence descriptor exists.

## 9. Graph rewrite / lowering opportunities

### Rewrite: projection plus L2 normalize

Source pattern:

```text
pooled -> Linear(hidden, projection_dim, bias=False) -> norm(dim=-1, keepdim=True) -> divide
```

Replacement:

```text
GEMM_RCR_OR_RRR -> row_l2_normalize
```

Preconditions:

- Projection weight is dense and bias-free.
- Input is rank 2 `[B, hidden]`.
- Norm axis is last dimension.
- No user requests unnormalized `get_*_features()` output from the same fused graph.

Shape equations:

```text
X: [B, H]
W: [P, H]
Y: [B, P]
norm: [B, 1]
Y_norm: [B, P]
```

Weight transform:

```python
# PyTorch Linear stores [out_features, in_features].
# GEMM choice must preserve Y = X @ W.T.
```

Failure cases:

- Missing or non-rank-2 pooler output.
- Zero norm rows if a backend wants reciprocal-norm clamping; source has no clamp.
- Need `get_image_features()`/`get_text_features()` parity, which returns projected but unnormalized pooler output.

Parity test sketch:

- Random pooled tensors and projection weights in fp32/fp16.
- Compare projected features and normalized features separately to PyTorch source ordering.

### Rewrite: similarity matmul orientation

Source pattern:

```text
logits_per_text = text_embeds @ image_embeds.T * exp(logit_scale)
logits_per_image = logits_per_text.T
```

Replacement:

```text
GEMM(text_embeds, image_embeds, B_transposed=True) -> scalar multiply -> optional transpose alias/copy
```

Preconditions:

- Both embeddings are normalized `[B_text, P]` and `[B_image, P]`.
- Both use the same `projection_dim`.
- Orientation is preserved exactly: text rows by image columns first.

Shape equations:

```text
T: [Bt, P]
I: [Bi, P]
T @ I.T: [Bt, Bi]
transpose: [Bi, Bt]
```

Layout constraints:

- Prefer a GEMM layout that consumes `image_embeds` as row-major logical `[Bi, P]` with transposed-B semantics.
- If `logits_per_image` is an ABI output, DinoML can either materialize the transpose or choose to compute both orientations when that is faster for downstream consumers. Source returns both.

Failure cases:

- Accidentally exposing `[B_image, B_text]` as `logits_per_text`.
- Treating batch sizes as equal; rectangular retrieval batches are valid.

Parity test sketch:

- Use `B_text != B_image` to force orientation correctness.
- Verify both logits matrices and the relationship `logits_per_image == logits_per_text.T`.

### Rewrite: cached branch embeddings

Source pattern:

```text
branch encoder -> projection -> normalize -> similarity
```

Replacement:

```text
offline or earlier-stage branch embedding cache -> similarity GEMM
```

Preconditions:

- Cache is tied to exact projection weights and `logit_scale` compatibility.
- Cached embeddings are already normalized if skipping branch normalization.
- Image/text processor versions and backbone checkpoints are part of cache provenance.

Failure cases:

- Caching unnormalized `get_*_features()` outputs and using them as normalized embeddings.
- Mixing embeddings from different projection heads.

### Layout candidate: vision NCHW to channel-last within audited backbone

Source pattern:

```text
processor pixel_values [B, C, H, W] -> delegated vision model
```

Replacement:

```text
guarded local channel-last vision region, only inside admitted vision-family lowering
```

Preconditions:

- The delegated vision family audit proves every axis-sensitive op is rewritten.
- Patch embedding/conv weights are transformed consistently.
- Pooling/class-token output ABI remains unchanged.

Failure cases:

- Applying NHWC at the wrapper boundary without rewriting the delegated vision model.
- Rewriting text/vision output axes; wrapper projection assumes rank-2 `[B, hidden]`.

## 10. Kernel fusion candidates

Highest priority:

- Projection GEMM plus row L2 normalization for each branch. This is the only wrapper-owned dense feature transform and sits on every inference path.
- Similarity GEMM with scalar scale and optional transpose handling. Retrieval workloads can be dominated by large rectangular score matrices.

Medium priority:

- Fused row norm implementation for reduced precision, computing norm in fp32 and storing normalized fp16/bf16 when selected.
- Avoid materializing both logits orientations when the caller needs only one. Source returns both in `forward()`, but retrieval APIs often consume `logits_per_image`.
- Branch embedding cache ABI with provenance for processor, backbone, projection weights, and normalization status.

Lower priority:

- Training contrastive loss fusion. Useful only if DinoML later supports training or loss evaluation.
- Backbone-specific fusions such as ViT patch embedding, BERT attention, or XLM-R LayerNorm belong to the corresponding family audits.

## 11. Runtime staging plan

Stage 1: wrapper config and weight loader

- Parse `VisionTextDualEncoderConfig`.
- Require both `vision_config` and `text_config`.
- Load wrapper weights: `visual_projection.weight`, `text_projection.weight`, `logit_scale`.
- Reject checkpoints where delegated body types are not on an audited allowlist.

Stage 2: projection/similarity parity from supplied pooled tensors

- Stub both encoders with explicit `vision_pooler_output` and `text_pooler_output`.
- Validate projection, normalization, logit scale, and similarity orientation.

Stage 3: compose one small audited pair

- Start with tiny or base ViT+BERT only after those families are admitted.
- Keep the wrapper as composition; do not duplicate backbone lowering inside this report's implementation.

Stage 4: branch feature APIs

- Support `get_image_features()` and `get_text_features()` semantics: projected but not normalized pooler outputs in returned branch output.
- Add a separate normalized embedding export if desired, but label it as DinoML-specific.

Stage 5: retrieval runtime

- Support separate image/text embedding cache build.
- Support rectangular text/image batches in final GEMM.

Stage 6: broader delegated allowlist

- Add CLIPVision+BERT/RoBERTa, ViT+BERT, DINOv2+BERT/E5, XLM-R text only after separate audits and parity fixtures exist.

## 12. Parity and validation plan

No DinoML tests were run for this audit.

Recommended parity tests:

- Config parse tests:
  - Missing `vision_config` or `text_config` rejects.
  - `projection_dim` and hidden sizes drive exact projection weight shapes.
  - Special nested CLIP/SigLIP/ChineseCLIP vision config extraction is represented.
- Wrapper math tests:
  - Random pooled fp32 test for projection+normalization+similarity.
  - Rectangular `B_text != B_image` test for logits orientation.
  - Scalar logit scale test with known `exp(2.6592)`.
  - Zero or near-zero pooled/projection rows documented against PyTorch behavior.
- Branch API tests:
  - `get_text_features()` returns projected, unnormalized pooler output.
  - `get_image_features()` returns projected, unnormalized pooler output.
- End-to-end composition tests:
  - Tiny ViT+BERT checkpoint with saved `_sources` config and processor snapshot.
  - One production-ish ViT+BERT or CLIPVision+RoBERTa checkpoint after sub-family audits.
- Tolerances:
  - fp32 wrapper-only: `rtol=1e-5`, `atol=1e-6`.
  - fp16/bf16 wrapper-only: use fp32 accumulation for norm/GEMM when possible; start around `rtol=1e-2`, `atol=1e-2` and tighten per backend.

## 13. Performance probes

- Processor throughput separately for image processor and tokenizer.
- Vision encoder throughput by image resolution and batch size.
- Text encoder throughput by sequence length and batch size.
- Projection+normalization throughput for image and text branches.
- Similarity matrix throughput for rectangular sweeps:
  - `B_text x B_image` = 1xN, Nx1, 32x1024, 1024x1024, larger retrieval batches.
- Memory use for cached normalized embeddings:
  - `[N_images, projection_dim]` and `[N_texts, projection_dim]`.
- GEMM backend comparison for final similarity:
  - row-major embeddings with transposed-B consumption versus pre-transposed image cache.
- End-to-end requests/hour split into preprocessing, branch encoder, projection, similarity, and post-softmax if caller requests probabilities.

## 14. Skip/defer list

- Training and `return_loss=True` contrastive loss.
- Any unaudited delegated backbone.
- Remote-code-only delegated bodies.
- Generation, beam search, KV cache, and causal decode paths; this wrapper is not a generation model.
- General NHWC/channel-last translation at the wrapper boundary.
- Reuse of pretrained CLIP projection heads when constructing from a full CLIP checkpoint via `from_vision_text_pretrained()`; source does not do this.
- Quantized/packed weight formats unless introduced by a delegated backbone audit.
- Multi-GPU/tensor parallel behavior.

## 15. Final implementation checklist

- [ ] Parse `VisionTextDualEncoderConfig` with required `vision_config` and `text_config`.
- [ ] Implement admission allowlist for delegated vision/text model families.
- [ ] Reject delegated models without stable rank-2 `pooler_output`.
- [ ] Load wrapper weights `visual_projection.weight`, `text_projection.weight`, and scalar `logit_scale`.
- [ ] Implement bias-free projection heads with source weight layout `[projection_dim, hidden]`.
- [ ] Implement row L2 normalization over `projection_dim`.
- [ ] Implement `exp(logit_scale)` scalar multiply.
- [ ] Implement similarity orientation: `logits_per_text = text @ image.T`, `logits_per_image = logits_per_text.T`.
- [ ] Add rectangular batch parity test for output orientation.
- [ ] Add wrapper-only pooled-tensor parity tests independent of backbones.
- [ ] Add `get_text_features()` and `get_image_features()` parity for projected but unnormalized branch outputs.
- [ ] Compose first audited ViT+BERT or CLIPVision+BERT pair.
- [ ] Add processor ABI fixture with tokenizer/image processor output names.
- [ ] Add branch embedding cache provenance design.
- [ ] Benchmark final similarity GEMM and optional pre-transposed image-cache layout.

## Gated gaps for DinoML admission

- Delegated backbone ownership is the main gate. The wrapper must not silently admit arbitrary `AutoModel` pairs.
- Pooler output compatibility is a hard gate. The native wrapper indexes `outputs[1]`; unsupported output ABIs should reject rather than guess pooling.
- Processor ABI is compositional. DinoML must route tokenizer/image processor behavior through the selected sub-components or require caller-supplied tensors.
- Similarity orientation must be tested with unequal image/text batch sizes.
- Branch feature methods and full forward differ by normalization. Treat projected branch features and normalized retrieval embeddings as separate ABI surfaces.
