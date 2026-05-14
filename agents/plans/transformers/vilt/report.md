# ViLT Transformers Audit

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: dandelin/vilt-b32-mlm, dandelin/vilt-b32-finetuned-vqa, dandelin/vilt-b32-finetuned-nlvr2, dandelin/vilt-b32-finetuned-coco, dandelin/vilt-b32-finetuned-flickr30k
Config source: official HF config.json, preprocessor_config.json, tokenizer_config.json, and special_tokens_map.json snapshots under agents/plans/transformers/vilt/_sources/
Source files inspected: src/transformers/models/vilt/configuration_vilt.py, modeling_vilt.py, image_processing_vilt.py, image_processing_pil_vilt.py, processing_vilt.py, __init__.py; auto/modeling_auto.py for VQA auto mapping.
Any missing files or assumptions: no gated/401/403 configs encountered. __init__.py has a TYPE_CHECKING import for feature_extraction_vilt.py, but that file is absent in this checkout; official checkpoint preprocessor configs still use historical feature_extractor_type=ViltFeatureExtractor while current source has ViltImageProcessor / ViltImageProcessorPil.
```

Primary links: [vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm), [vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), [vilt-b32-finetuned-nlvr2](https://huggingface.co/dandelin/vilt-b32-finetuned-nlvr2), [vilt-b32-finetuned-coco](https://huggingface.co/dandelin/vilt-b32-finetuned-coco), [vilt-b32-finetuned-flickr30k](https://huggingface.co/dandelin/vilt-b32-finetuned-flickr30k).

Report target: inference-only ViLT multimodal encoder parity for VQA / retrieval / MLM / NLVR heads. This is not an autoregressive generation family.

## 2. High-Level Architecture

ViLT is a single-stream image-text encoder. Text tokens use BERT-style word, token-type, and absolute position embeddings. Images use a non-overlapping Conv2d patch projection, learned 2D patch positions resized to the effective padded-image patch grid, a learned image CLS token, and modality type embeddings. Text and image token streams are concatenated before a BERT/ViT-like bidirectional Transformer encoder.

```text
CPU image/text preprocessing
  -> input_ids/attention_mask + NCHW pixel_values/pixel_mask
  -> text embeddings + image patch embeddings/patch mask/position interpolation
  -> concatenate text tokens and image tokens
  -> 12 bidirectional self-attention encoder blocks
  -> final LayerNorm + CLS pooler
  -> optional MLM / VQA / retrieval / NLVR / token-classification heads
```

Stage decomposition:

- CPU/data pipeline: BERT tokenization, image resize with aspect ratio preserved, size-divisor rounding, rescale, normalize, pad batch, pixel mask.
- Patch/token packing: Conv2d patch projection, pixel-mask downsampling to patch mask, stochastic/effective patch selection, learned position interpolation, text/image sequence concatenation.
- Encoder: independently stageable from the heads, but not independently cacheable by modality once text and image tokens are stitched.
- Heads: pooler-based VQA/retrieval/NLVR heads, token-only MLM/token-classification heads. NLVR runs the full encoder once per image and concatenates pooled outputs.

## 3. Important Config Dimensions

Source defaults from `configuration_vilt.py`:

| Field | Default |
| --- | ---: |
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| head_dim | 64 |
| intermediate_size | 3072 |
| vocab_size | 30522 |
| max_position_embeddings | 40 |
| type_vocab_size | 2 |
| modality_type_vocab_size | 2 |
| image_size | 384 |
| patch_size | 32 |
| image grid at config size | 12 x 12 = 144 patches |
| num_channels | 3 |
| qkv_bias | true |
| hidden_act | gelu |
| layer_norm_eps | 1e-12 |
| dropout / attention dropout | 0.0 / 0.0 |
| max_image_length | -1, meaning use all effective valid patches |
| cache support | none; encoder-only bidirectional attention |

Representative official checkpoint sweep:

| Model | Architecture in config | Hidden/layers/heads | Patch/image | Max text positions | Labels/head fields | Notable config facts |
| --- | --- | ---: | --- | ---: | --- | --- |
| `dandelin/vilt-b32-mlm` | `ViltForMaskedLM` | 768 / 12 / 12 | 32 / 384 | 40 | vocab 30522 | base MLM/pretraining head |
| `dandelin/vilt-b32-finetuned-vqa` | `ViltForVisualQuestionAnswering` | 768 / 12 / 12 | 32 / 384 | 40 | 3129 labels | historical architecture name; current source class is `ViltForQuestionAnswering` |
| `dandelin/vilt-b32-finetuned-nlvr2` | `ViltForImagesAndTextClassification` | 768 / 12 / 12 | 32 / 384 | 40 | 2 labels, `num_images=2`, `modality_type_vocab_size=3` | full encoder called separately for image 1 and image 2 |
| `dandelin/vilt-b32-finetuned-coco` | `ViltForImageAndTextRetrieval` | 768 / 12 / 12 | 32 / 384 | 40 | rank output 1 | pair scorer, not CLIP-style independent dual encoder |
| `dandelin/vilt-b32-finetuned-flickr30k` | `ViltForImageAndTextRetrieval` | 768 / 12 / 12 | 32 / 384 | 40 | rank output 1 | same operator structure as COCO retrieval |

Preprocessor snapshots use `feature_extractor_type=ViltFeatureExtractor`, `size=384`, `size_divisor=32`, bicubic resample id 3, and mean/std `[0.5, 0.5, 0.5]`. Current source defaults instead name `ViltImageProcessor` and use ImageNet standard mean/std unless checkpoint metadata overrides them. Tokenizer snapshots use `BertTokenizer`, `bert-base-uncased`, `do_lower_case=true`, `model_max_length=40`, and standard `[CLS] [SEP] [PAD] [MASK] [UNK]` tokens.

## 3a. Family Variation Traps

- `max_image_length` changes graph shape and selection behavior. `-1` uses `max(x_h * x_w)` effective patch count in the batch; positive values cap it and may sample valid patches down.
- Patch selection uses `torch.multinomial` over valid or padded patch indices. For strict parity, first integration should prefer `max_image_length=-1` and deterministic fully valid resized images, or model the sampling as a source-specific gather plan.
- `pixel_mask` is pixel-space `[B, H, W]`; source downsamples it with `interpolate` to patch grid, then flattens/selects patch tokens. It is not just a final attention mask.
- `position_embeddings` are learned for the config grid plus CLS, then bilinearly interpolated per sample to effective valid patch counts `(x_h, x_w)` and padded to the batch patch grid before selection.
- `ViltForImageAndTextRetrieval` is a cross-encoder ranker: every image-text pair runs jointly through ViLT and outputs one score. It does not produce independently cacheable normalized image/text embeddings or a CLIP-style similarity matrix.
- `ViltForImagesAndTextClassification` requires `num_images` to match the second image dimension. It loops over images, uses image modality ids `1..num_images`, concatenates pooler outputs along hidden dim, then classifies.
- VQA config uses historical `architectures=["ViltForVisualQuestionAnswering"]`; current `AutoModelForVisualQuestionAnswering` maps `vilt` to `ViltForQuestionAnswering`.
- `tie_word_embeddings` is forced true by `ViltConfig.__post_init__`; MLM decoder weight is tied to text word embeddings.
- Initial graph translation should preserve NCHW. NHWC/channel-last is an optimization opportunity only for local Conv2d/patch/attention regions with explicit axis rewrites.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- NCHW image input, pixel-space mask `[B,H,W]`, Conv2d patch projection, interpolate mask, bilinear interpolate position embeddings, pad, flatten, transpose, permute, contiguous, view/reshape.
- `torch.meshgrid`, `arange`, `stack`, `expand`, `nonzero`, unique-row grouping, per-row gather, cat along sequence/hidden axes, slice text portion from final sequence.
- Embedding lookup for word, text position, text token type, modality type, and learned image CLS/patch positions.

Neural network primitives:

- Patch projection: `Conv2d(3 -> 768, kernel=32, stride=32, bias=True)` in current source.
- Text embeddings: word `30522 x 768`, text position `40 x 768`, token type `2 x 768`, LayerNorm eps `1e-12`.
- Encoder block x12: LayerNorm, Q/K/V Linear `768 -> 768` with qkv bias, dense output `768 -> 768`, residual add, LayerNorm, FFN `768 -> 3072 -> 768`, GELU.
- Final LayerNorm and pooler `Linear(768 -> 768) + Tanh` on sequence token 0.
- MLM head: `Linear(768 -> 768) + GELU + LayerNorm + Linear(768 -> 30522)`, decoder weight tied to word embedding.
- VQA head: `Linear(768 -> 1536) + LayerNorm(1536) + GELU + Linear(1536 -> num_labels)`.
- Retrieval head: `Linear(768 -> 1)`.
- NLVR head for official config: concatenate two `768` pooler outputs, then `Linear(1536 -> 1536) + LayerNorm(1536) + GELU + Linear(1536 -> 2)`.
- Token classification: dropout then `Linear(768 -> num_labels)` over text-token slice only.

Attention primitives:

- Bidirectional dense self-attention only, MHA 12 heads, head dim 64, no causal mask, no KV cache, no cross-attention.
- Additive attention mask from concatenated text attention mask and image patch mask, broadcast to `[B,1,1,S]`, converted to dtype min for masked keys.
- Eager math order: Q/K/V projection, reshape to `[B,H,S,64]`, QK^T, scale by sqrt(64), add mask, softmax over keys, dropout, AV, transpose/contiguous/view.

Preprocessing-coupled ops:

- Aspect-ratio resize with shorter edge 384 by checkpoint config, longer-edge cap `int(1333/800*shorter)`, rounded to multiples of 32.
- Rescale and normalize. Checkpoint configs use mean/std `[0.5,0.5,0.5]`; current source defaults use ImageNet standard mean/std.
- Batch padding to max resized H/W and pixel mask with 1 for valid pixels and 0 for padding.

## 5. Layer / Block Breakdown

Image/text embedding path:

```text
text = word_embed(input_ids) + token_type_embed(token_type_ids) + position_embed(position_ids)
text = LayerNorm(text)

patch = Conv2d(pixel_values NCHW)                       # [B,768,Hp,Wp]
patch_mask = interpolate(pixel_mask[:,None], Hp,Wp)
effective_h/effective_w = sums over patch_mask
pos = bilinear_interpolate(learned_12x12_pos, effective_h/effective_w), pad to Hp/Wp
patch, patch_mask, pos = flatten/select to max_image_length
image = cat(cls_token, patch_tokens) + cat(cls_pos, pos)

text += modality_embed(0)
image += modality_embed(image_token_type_idx, usually 1)
hidden = cat(text, image, dim=1)
mask = cat(text_attention_mask, image_patch_mask, dim=1)
```

Encoder block, repeated 12 times:

```text
y = LayerNorm(hidden)
q,k,v = Linear(y), Linear(y), Linear(y)                 # all 768 -> 768, bias from qkv_bias
y = dense(SelfAttention(q,k,v, additive_mask))
hidden = hidden + y
y = LayerNorm(hidden)
y = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)
hidden = hidden + y
```

Final:

```text
sequence = LayerNorm(hidden)
pooled = Tanh(Linear(sequence[:,0]))
head-specific logits from pooled or text-token slice
```

## 6. Attention Requirements

ViLT requires encoder-style noncausal dense self-attention over the concatenated text-plus-image sequence.

| Property | Requirement |
| --- | --- |
| causal | no |
| self/cross | self-attention over stitched multimodal sequence |
| MHA/GQA/MQA | MHA only |
| heads / KV heads / head dim | 12 / 12 / 64 in inspected configs |
| sequence length | `text_len + 1 + selected_image_patches`; common 40 + 1 + 144 at square 384, but variable with aspect ratio and `max_image_length` |
| mask | additive key mask derived from text attention mask and selected patch mask |
| packed/varlen | no source varlen backend; padding mask is dense |
| sliding/local | none |
| position interactions | learned absolute text positions plus learned/interpolated image patch positions before attention |
| KV cache | not applicable |
| FlashAttention/SDPA | source uses explicit eager matmul/softmax; FlashAttention can be an optimization if additive padding mask and noncausal encoder semantics are preserved |

## 7. Position Encoding and Custom Math

Text uses learned absolute position ids sliced from `[0, max_position_embeddings)`.

Image positions are learned at config patch grid size and resized per sample:

```python
patch_grid = config.image_size // config.patch_size
spatial_pos = position_embeddings[:, 1:, :].transpose(1, 2).view(1, hidden, patch_grid, patch_grid)
per_sample_pos = interpolate(spatial_pos, size=(valid_patch_h, valid_patch_w),
                             mode="bilinear", align_corners=True)
per_sample_pos = pad(per_sample_pos, (0, Wp - valid_patch_w, 0, Hp - valid_patch_h))
pos_tokens = per_sample_pos.flatten(2).transpose(1, 2)
```

The valid patch dimensions depend on `pixel_mask` after interpolation to the patch grid. Text positions can be precomputed. Image spatial position interpolation depends on runtime image sizes/masks unless inputs are bucketed to fixed resized shapes.

## 8. Preprocessing and Input Packing

CPU/data pipeline:

- Text: BERT uncased tokenization, `[CLS] ... [SEP]`, padding/truncation controlled by tokenizer/processor kwargs. Official tokenizer configs set `model_max_length=40`.
- Image: resize preserving aspect ratio to shorter edge 384, cap longer edge to about 639 for that setting, round H/W down to a multiple of 32, bicubic resample, rescale, normalize, pad batch.
- Output tensors: `pixel_values` in NCHW float tensor `[B,3,Hpad,Wpad]`, `pixel_mask` int64 `[B,Hpad,Wpad]`, `input_ids`, `attention_mask`, optional `token_type_ids`.

GPU/runtime-coupled packing:

- `pixel_mask` is consumed inside the model to derive patch masks and selected patch indices, so it is part of graph parity, not just preprocessing metadata.
- The source can accept `image_embeds` directly as `[B,num_patches,hidden]`; this bypasses Conv2d and position interpolation and uses `pixel_mask.flatten(1)` as image mask. Treat this as a separate ABI, useful for staged tests but not enough for end-to-end image parity.
- NLVR processor use passes two images; model expects `pixel_values` shaped `[B,num_images,3,H,W]` after optional unsqueeze and loops over `num_images`.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Non-Overlap Conv2d Patch Projection -> Linear

Source pattern:

```text
Conv2d(C -> H, kernel=patch_size, stride=patch_size, padding=0, groups=1)
```

Replacement:

```text
WindowFlatten(NCHW, kh=kw=32, stride=32) -> Linear(C*kh*kw -> hidden) -> ReshapeToPatchGrid
```

Preconditions:

- `kernel_size == stride == config.patch_size`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W are divisible by patch size after preprocessing or guarded at runtime.
- Preserve PyTorch Conv2d flatten order: weight transform `conv.weight.reshape(out_channels, in_channels * kh * kw)`.
- Bias is present in current source and must be included.

Failure cases: arbitrary non-divisible image sizes, alternate channel counts, or source changes to padding/dilation/groups should fall back to Conv2d.

Parity test sketch: compare Conv2d output `[B,H,Hp,Wp]` against window-flatten linear output before flatten/transpose.

### Rewrite: Stitched Encoder Attention

Source pattern: separate text/image embedding construction followed by `cat(..., dim=1)` and ordinary encoder self-attention.

Replacement: canonical single encoder over `[B,S,H]` with an additive padding mask.

Preconditions: text/image sequence construction exactly matches source token order: all text tokens first, then image CLS, then selected image patches. Image modality ids must match source, especially NLVR `1` and `2`.

Failure cases: `max_image_length > 0` stochastic sampling, caller-provided `image_embeds`, or NLVR multi-image loop should use explicit guards/staged paths.

### Rewrite: NHWC / Channel-Last Patch Region

Treat NHWC as a guarded layout pass only.

Preconditions:

- Region starts at `pixel_values` and ends before flattened patch tokens.
- Rewrite Conv2d/window flatten axes from NCHW `[B,C,H,W]` to NHWC `[B,H,W,C]`.
- Rewrite padding/interpolation/mask axes explicitly; mask remains `[B,H,W]`.
- Downstream token tensors remain `[B,S,H]`.

Failure cases: any consumer requiring NCHW source semantics outside this local patch region, or direct exposure of intermediate patch maps.

### Rewrite: Pooler + Small Heads

Pooler `sequence[:,0] -> Linear -> Tanh` followed by small classifier/ranker MLPs can be fused as a low-priority epilogue chain once encoder parity is stable. Preserve Tanh before head layers.

## 10. Kernel Fusion Candidates

Highest priority:

- Conv patch embedding as GEMM/window-flatten path, because it is the first image bottleneck and has simple non-overlap guards.
- Encoder LayerNorm + QKV projections and dense attention backend for noncausal MHA with padding mask.
- FFN `Linear + GELU + Linear` for 768/3072 hidden widths.

Medium priority:

- Text/image embedding add + modality add + final concat/mask concat, mainly to reduce launch overhead around short sequences.
- Bilinear image position interpolation and patch-mask downsampling for fixed resolution buckets.
- Pooler plus VQA/NLVR classifier MLPs.

Lower priority:

- MLM prediction head fusion for text-token slice only.
- Retrieval rank head fusion; it is too small to matter until pair batching is solved.
- Token-classification slice/dropout/linear fusion; dropout is disabled in inference.

## 11. Runtime Staging Plan

1. Parse config and load weights for `ViltModel`; reject or stage separately caller-provided `image_embeds` and `max_image_length > 0`.
2. Implement deterministic end-to-end base encoder for square 384 inputs with full valid masks: text embeddings, patch Conv2d, learned position add, sequence stitch, encoder, final LayerNorm.
3. Add pixel-mask-aware position interpolation and patch mask handling for padded/aspect-ratio images.
4. Add pooler heads: retrieval ranker, VQA classifier, then NLVR two-image loop.
5. Add MLM/token-classification text-slice heads and tied embedding alias checks.
6. Optimize patch projection and encoder attention/FFN with guarded fusions.
7. Revisit stochastic patch selection for positive `max_image_length`; either reproduce RNG/select behavior or reject for first production path.

## 12. Parity and Validation Plan

- Processor parity: image resize/pad/mask for square, wide, tall, and mixed-batch inputs; compare `pixel_values` and `pixel_mask`.
- Patch embedding parity: Conv2d output, downsampled mask, interpolated positions, selected patch tokens for deterministic `max_image_length=-1`.
- Single-block parity: one encoder layer with fixed stitched embeddings and additive mask, fp32 tolerance around `1e-5`/`1e-4`.
- Full encoder parity: `last_hidden_state` and `pooler_output` for `dandelin/vilt-b32-mlm`.
- Head parity: VQA logits, retrieval scalar logits, NLVR logits for two-image input, MLM logits over text slice, token-classification logits.
- Layout rewrite parity: Conv2d-to-linear patch path and optional NHWC local pass against source NCHW outputs.
- Reduced precision: after fp32 parity, test fp16/bf16 with tolerances around `1e-2` for logits and validate attention mask min-value behavior.

## 13. Performance Probes

- CPU preprocessing throughput by image resolution/aspect ratio and batch size.
- Patch projection throughput for Conv2d versus window-flatten GEMM.
- Position interpolation and patch selection overhead for varied H/W buckets.
- Encoder-only throughput versus sequence length: text length 40 plus image patches 145, 241, and up to effective maximum around 1026 for 800x1333 style inputs.
- VQA/retrieval/NLVR end-to-end throughput; NLVR should report cost per pair because it runs the encoder once per image.
- Attention backend comparison: eager dense MHA versus fused noncausal attention with additive padding mask.
- Pair batching strategy for retrieval cross-encoder scoring.

## 14. Skip / Defer List

- Training losses and gradient checkpointing.
- Dropout behavior beyond inference-disabled paths.
- Positive `max_image_length` stochastic patch sampling in production admission, unless an exact RNG/select contract is implemented.
- Historical `ViltFeatureExtractor` class surface; current source image processors plus checkpoint metadata are enough for runtime parity.
- CLIP-style independent retrieval embedding cache; official ViLT retrieval head is a cross-encoder ranker.
- General NHWC translation outside the local patch/projection region.
- Quantization, tensor parallelism, and distributed inference.

## 15. Final Implementation Checklist

- [ ] Parse `ViltConfig` and checkpoint head class, including historical VQA architecture alias.
- [ ] Load/tie text word embedding and MLM decoder weights as one logical parameter.
- [ ] Implement image processor parity or require preprocessed `pixel_values`/`pixel_mask`.
- [ ] Implement NCHW Conv2d patch embedding with bias.
- [ ] Implement pixel-mask downsampling, patch-mask flatten/select, and image position interpolation.
- [ ] Implement text embeddings and modality embeddings.
- [ ] Implement stitched noncausal MHA encoder block with additive padding mask.
- [ ] Implement final LayerNorm and pooler.
- [ ] Add VQA, retrieval, NLVR, MLM, and token-classification heads as staged targets.
- [ ] Add guards for `max_image_length > 0`, `image_embeds`, nonstandard `num_images`, and layout rewrites.
- [ ] Add Conv2d-to-linear patch rewrite with parity tests.
- [ ] Benchmark preprocessing, patch projection, encoder, and head-specific throughput.
