# SAM3-LiteText Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: yonigozlan/sam3-litetext-s0, yonigozlan/sam3-litetext-s1, yonigozlan/sam3-litetext-l
Config source: HF raw config.json/tokenizer_config.json/processor_config.json fetched 2026-05-13
Source files inspected: sam3_lite_text configuration/modeling/modular/converter plus shared sam3 configuration/modeling
Any missing files or assumptions: weights and safetensors metadata were not downloaded; no imports/tests were run; tracker keys are ignored but no tracker implementation was present in this family.
```

Primary local files:

- `transformers/src/transformers/models/sam3_lite_text/configuration_sam3_lite_text.py`
- `transformers/src/transformers/models/sam3_lite_text/modeling_sam3_lite_text.py`
- `transformers/src/transformers/models/sam3_lite_text/modular_sam3_lite_text.py`
- `transformers/src/transformers/models/sam3_lite_text/convert_sam3_lite_text_to_hf.py`
- `transformers/src/transformers/models/sam3/configuration_sam3.py`
- `transformers/src/transformers/models/sam3/modeling_sam3.py`

Snapshot notes and reduced config snapshots are in `_sources/`.

## 2. High-level architecture

SAM3-LiteText is a vision-language segmentation model, not a text-only model. It keeps the SAM3 image encoder and detector/mask stack, then replaces SAM3's original text encoder with a lightweight MobileCLIP-style text encoder.

```text
image processor + CLIP tokenizer
  -> SAM3 ViT-H vision encoder + FPN
  -> MobileCLIP-style text tower
  -> optional box geometry encoder
  -> DETR encoder text/geometry fusion
  -> DETR query decoder with box refinement and presence token
  -> dot-product text/query scoring + mask decoder
  -> boxes, logits, masks, semantic segmentation
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize to 1008, rescale/normalize, CLIP BPE tokenization to length 77, optional normalized boxes and labels.
- Independently cacheable vision stage: `get_vision_features(pixel_values)` returns FPN hidden states and FPN positional encodings.
- Independently cacheable text stage: `get_text_features(input_ids, attention_mask)` returns full token features projected to detector hidden size.
- Prompt fusion: optional box prompt encoder concatenates geometry features with text features.
- Detector prefill equivalent: DETR encoder and decoder run non-autoregressively over vision tokens, text tokens, optional geometry prompts, and 200 learned queries.
- Output heads: box refinement, query/text score, mask dot product, semantic head.

There is no autoregressive decode loop and no KV cache. The relevant runtime state is reusable vision/text embeddings and optional prompt features.

## 3. Important config dimensions

| Component | Field | S0 | S1 | L |
|---|---:|---:|---:|---:|
| Text | hidden_size | 512 | 512 | 768 |
| Text | intermediate_size | 2048 | 2048 | 3072 |
| Text | projection_dim | 512 | 512 | 768 |
| Text | layers | 6 | 12 | 12 |
| Text | heads | 8 | 8 | 12 |
| Text | head_dim | 64 | 64 | 64 |
| Text | max_position_embeddings | 77 | 77 | 77 |
| Text | vocab_size | 49408 | 49408 | 49408 |
| Text | use_repmixer_blocks | true | false | false |
| Detector | hidden_size | 256 | 256 | 256 |
| DETR encoder/decoder | layers | 6 / 6 | 6 / 6 | 6 / 6 |
| DETR decoder | queries | 200 | 200 | 200 |
| Vision backbone | hidden/layers/heads | 1024 / 32 / 16 | same | same |
| Vision | image/patch/window | 1008 / 14 / 24 | same | same |
| Vision | global attention layers | 7, 15, 23, 31 | same | same |
| FPN | hidden_size | 256 | 256 | 256 |
| Processor | mask_size | 288x288 | likely same | likely same |

Representative checkpoint sweep:

| Model id | Text tower | Text params from HF docs | Operator-significant variation |
|---|---|---:|---|
| `yonigozlan/sam3-litetext-s0` | MobileCLIP-S0 | 42.54M | 6 text layers; RepMixer blocks at first/last layer; hidden 512 |
| `yonigozlan/sam3-litetext-s1` | MobileCLIP-S1 | 63.53M | 12 standard transformer text layers; hidden 512 |
| `yonigozlan/sam3-litetext-l` | MobileCLIP2-L | 123.80M | 12 standard transformer text layers; hidden 768, 12 heads |

The shared detector stack projects all text variants into 256-dim DETR hidden states through `Sam3LiteTextModel.text_projection`.

## 3a. Family variation traps

- S0 is not just a shallower transformer: with `use_repmixer_blocks=True`, layers 0 and `num_hidden_layers - 1` are RepMixer convolutional blocks operating through an NCHW-shaped `[B, C, 1, T]` view.
- S1 and L set `use_repmixer_blocks=False`, so every text layer is standard noncausal self-attention plus MLP.
- Text `hidden_size` is not the detector hidden size. The text tower emits 512 or 768 channels, then the full model projects each token to 256 channels for detector fusion.
- The text encoder's internal `pooler_output` is EOT-pooled by `input_ids.argmax(dim=-1)`, CLIP-style, but the full model's `get_text_features` overwrites `pooler_output` with projected full-sequence token features, not a pooled vector.
- The full model disables FlashAttention and FlexAttention support because DETR cross-attention can pass additive float relative-position bias masks. Text-only submodules advertise attention backend support.
- Tokenizer ABI is CLIP tokenizer compatible: lowercasing, `max_length=77`, `<|endoftext|>` as EOS/PAD/UNK, and `input_ids.argmax` pooling assumes EOT has the highest token id.
- The inspected family ignores unexpected tracker weights (`tracker_model`, `tracker_neck`) but does not implement a video tracker/session state ABI.
- Source uses NCHW for vision/FPN/ROI/mask paths and `[B, T, C]` for text/DETR sequence paths, with explicit flatten/transpose/permutation boundaries.
- `roi_align` casts bf16 vision features to fp16 because torchvision ROI align only supports fp16/fp32 in the source path.
- `spatial_shapes.shape[0] == 1` gates decoder box-RPB construction. Broader multi-level decoder RPB is not active in the inspected path.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `flatten(2)`, `transpose`, `permute`, `contiguous`, `cat`, `stack`, `pad`, `repeat`, `expand`, `clone`, boolean masks, `where`, `scatter`, `einsum`.
- Sequence/image layout bridges: `[B, C, H, W] -> [B, H*W, C]`; `[B, T, C] -> [B, C, 1, T] -> [B, T, C]`; `[B, T, C] -> [B, C, H, W]`.

Neural network primitives:

- Embedding tables for tokens, learned queries, reference boxes, presence token, box labels, CLS geometry token.
- Linear/GEMM: text Q/K/V/O, text MLP, text projection `text_hidden -> 256`, DETR attention projections, DETR MLPs, box heads, score projections.
- Conv2d: ViT patch embedding, FPN/neck, RepMixer depthwise conv, RepMixer pointwise MLP convs, mask pixel decoder, mask/semantic projections, geometry pooled projection.
- Norms: LayerNorm, BatchNorm2d in RepMixer, GroupNorm in mask pixel decoder.
- Activations/dropout: GELU, ReLU, softmax with fp32 accumulation, dropout, sigmoid, clamp, log/log2, sign, abs, sin/cos.
- Interpolation: text positional embedding uses bilinear interpolation for non-default text length; pixel decoder uses nearest upsample; vision positional tiling/interpolation exists in inherited SAM3 vision.

Attention primitives:

- Noncausal MHA self-attention in text and DETR paths.
- Cross-attention from geometry prompts to vision, vision tokens to prompts, decoder queries to text, decoder queries to vision, and mask decoder encoder states to prompts.
- Additive float mask support for RPB, plus boolean/bidirectional attention masks.

Preprocessing-coupled ops:

- CLIP BPE tokenizer, fixed 77 context, attention mask convention 1 valid / 0 padding.
- SAM3 image processor resize/rescale/normalize to NCHW pixel values.
- Box prompts in normalized `[0, 1]` cxcywh with label `-10` used as padding sentinel.

Detection/segmentation ops:

- `torchvision.ops.roi_align` for geometry box feature pooling.
- Box conversions `cxcywh -> xyxy`, inverse sigmoid refinement, learned query/reference boxes.
- Mask prediction `einsum("bqc,bchw->bqhw")`.
- Postprocessing is processor-owned and was not in the family modeling file.

## 5. Layer/block breakdown

Text embeddings:

```text
input_ids [B,T]
token_embedding -> [B,T,text_hidden]
learned_position_embedding(T) -> [1,T,text_hidden]
hidden = token + position
attention_mask -> create_bidirectional_mask
```

RepMixer text block, used only for S0 first and last layer:

```text
x [B,T,C] -> transpose/unsqueeze -> [B,C,1,T]
token_mixer = depthwise Conv2d(C groups, kernel 1xK, padding K//2) + BatchNorm2d + BatchNorm2d skip
x = x + layer_scale * (token_mixer(x) - reference_batchnorm(x))
ffn = depthwise Conv2d(C groups, 1xK) -> BatchNorm2d -> Conv2d(C->4C,1x1) -> GELU -> Conv2d(4C->C,1x1)
x = x + layer_scale * ffn
x -> squeeze/transpose -> [B,T,C]
```

Standard text encoder layer:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(C->C) split into H heads, head_dim 64
x = noncausal attention(q,k,v, additive/boolean mask)
x = residual + Linear(C->C)(x)
residual = x
x = LayerNorm(x)
x = Linear(C->4C) -> GELU -> Linear(4C->C)
x = residual + x
```

Text output:

```text
x = final LayerNorm(x)
pooled = x[batch, input_ids.argmax(-1)]
pooled = Linear(text_hidden -> projection_dim, bias=False)(pooled)
full model discards that pooled vector and projects all token states:
text_features = Linear(text_hidden -> 256)(last_hidden_state)
```

Detector fusion:

```text
vision FPN last level [B,256,H,W] -> flatten -> [B,H*W,256]
optional geometry boxes -> direct linear + ROIAlign pooled conv + sine coord projection + label embedding + CLS token
prompt_features = cat(text_features, geometry_features)
DETR encoder: vision self-attn with pos, then cross-attn to prompt features, then MLP
DETR decoder: learned presence+queries self-attn, text cross-attn, vision cross-attn with optional box RPB, MLP, box refinement
heads: query/text dot scoring, box xyxy, mask dot product, semantic conv
```

## 6. Attention requirements

Text tower:

- Noncausal self-attention.
- MHA, not GQA/MQA: Q/K/V all `hidden_size`, head_dim 64 for all published variants.
- Mask from `create_bidirectional_mask`; no causal mask and no KV cache.
- Eager attention applies `q @ k.T * head_dim^-0.5`, adds mask, softmax in fp32, casts back to query dtype, dropout, then `weights @ v`.
- SDPA/Flash/Flex can be eligible for text-only paths if mask/backend supports the exact mask semantics.

Detector:

- Noncausal self-attention and rectangular cross-attention.
- DETR encoder self-attention length is vision token count; cross-attention keys are prompt tokens.
- DETR decoder self-attention length is `num_queries + 1` because of the presence token.
- Decoder text cross-attention keys are prompt/text tokens.
- Decoder vision cross-attention keys are flattened vision tokens and may receive additive float RPB of shape `[B, heads, queries + 1, H*W]`.
- Full model should not globally force FlashAttention because additive RPB masks require SDPA/eager fallback.

No generation cache, encoder-decoder cache, or recurrent state is required for this family. Independently cacheable `vision_embeds` and `text_embeds` are ordinary model inputs and should be modeled as artifact-visible reusable tensors, not hidden session state.

## 7. Position encoding and custom math

Text position embedding is learned with source interpolation when runtime sequence length differs from trained length:

```python
def text_pos(position_embedding, seq_len):
    # source shape [1, 1, max_pos, C]
    if seq_len != position_embedding.shape[2]:
        position_embedding = interpolate(position_embedding, size=(seq_len, C), mode="bilinear", align_corners=False)
    return position_embedding.reshape(1, seq_len, C)
```

Sine box/grid position encodings use temperature-scaled sin/cos pairs. Box RPB uses a model-specific log transform:

```python
def box_rpb_delta(delta):
    z = delta * 8
    return sign(z) * log2(abs(z) + 1.0) / log2(8)
```

Decoder box refinement:

```python
def inverse_sigmoid(x, eps=1e-3):
    x = clamp(x, 0, 1)
    return log(clamp(x, min=eps) / clamp(1 - x, min=eps))

new_box = sigmoid(box_head(norm(query_hidden)) + inverse_sigmoid(reference_box))
```

Vision backbone inherited from SAM3 uses ViT absolute patch embeddings plus 2D RoPE in attention. That path should be audited with the SAM3 report if DinoML ports the full vision tower, because the LiteText-specific source only changes the text tower and detector wiring.

## 8. Preprocessing and input packing

Text ABI:

- Tokenizer class: CLIP tokenizer/fast backend.
- Vocabulary size: 49408.
- `model_max_length`: 77.
- BOS: `<|startoftext|>`, EOS/PAD/UNK: `<|endoftext|>`.
- `do_lower_case=true`, `add_prefix_space=false`.
- GPU graph input is `input_ids [B,T]` plus optional `attention_mask [B,T]`.
- CLIP EOT pooling inside `Sam3LiteTextTextModel` depends on `input_ids.argmax(-1)`, so tokenizer ID ordering is ABI-significant.

Image ABI:

- Processor produces NCHW `pixel_values [B,3,1008,1008]`.
- Rescale factor is `1/255`; mean/std are `[0.5, 0.5, 0.5]`.
- Mask postprocess target grid from processor metadata is 288x288 before final target-size handling.

Prompt ABI:

- Text prompts and box prompts are concatenated as prompt features after the text tower and optional geometry encoder.
- Box prompts are normalized cxcywh `[B,num_boxes,4]`.
- Box labels are `[B,num_boxes]`, with `1` positive, `0` negative, and `-10` padding/ignore.
- Geometry encoder appends a learned CLS prompt token and marks it valid.

Postprocessing:

- The model returns normalized xyxy boxes, masks, logits, presence logits, and semantic segmentation tensors.
- Thresholding, target/original size mapping, mask crop/resize, and variable-length instance records are processor postprocessing responsibilities and were not implemented in the inspected modeling file.

## 9. Graph rewrite / lowering opportunities

### Rewrite: RepMixer 1D depthwise Conv2d to Conv1d

Source pattern:

```text
[B,T,C] -> transpose -> [B,C,T] -> unsqueeze height=1
Conv2d(groups=C, kernel=(1,K), padding=(0,K//2)) -> BatchNorm2d
```

Replacement:

```text
[B,T,C] -> [B,C,T] -> depthwise Conv1d(C groups, kernel=K, padding=K//2) -> BatchNorm1d-equivalent -> [B,T,C]
```

Preconditions:

- Input height is always 1.
- Conv2d kernel height is 1.
- Groups equal channels.
- Padding height is 0.
- BatchNorm statistics/affine parameters are transformed exactly to the 1D channel form.
- No consumer observes the intermediate 4D tensor.

Failure cases:

- Any future RepMixer with height > 1, grouped-but-not-depthwise conv, nonzero height padding, or explicit NCHW consumer.

Parity test sketch:

- Random `[B,T,C]`, run source Conv2d+BN path and lowered Conv1d+BN path in eval mode for S0 sizes, compare fp32 within `1e-5`.

### Rewrite: 1x1 Conv2d to pointwise Linear/GEMM over flattened pixels

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding=0)
```

Replacement:

```text
[B,C,H,W] -> [B*H*W,C] -> GEMM(weight.T) + bias -> [B,Cout,H,W]
```

Preconditions:

- Kernel exactly 1x1, dilation 1, groups 1.
- Layout transform preserves NCHW channel order.
- Consumer can accept restored NCHW or a fused flattened layout.

Failure cases:

- Depthwise/grouped convs, 3x3 FPN convs, ROI pooled 7x7 conv, patch embedding 14x14 conv.

### Rewrite: non-overlap patch Conv2d to Linear

Source pattern:

```text
ViT patch embedding Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)
```

Replacement:

```text
unfold non-overlap 14x14 patches in NCHW order -> GEMM(weight.reshape(1024, 3*14*14).T)
```

Preconditions:

- Kernel equals stride, padding 0, dilation 1, groups 1.
- Image height/width divisible by 14.
- Patch flatten order matches PyTorch Conv2d NCHW kernel memory.

Failure cases:

- Dynamic image sizes not divisible by patch size; layout pass that silently changes channel/spatial flatten order.

### Rewrite: mask einsum to batched GEMM

Source pattern:

```text
pred_masks = einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)
```

Replacement:

```text
for each batch: [Q,C] x [C,H*W] -> [Q,H*W] -> [Q,H,W]
```

Preconditions:

- Instance embeddings are contiguous or represented with explicit strides after flattening.
- Batch dimension is not broadcast.
- Accumulation dtype matches source policy.

Failure cases:

- NHWC layout without axis rewrite; mixed precision accumulation differences not validated.

### Rewrite: text QKV load fusion

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
single packed GEMM C -> 3C, split as Q, K, V
```

Preconditions:

- Same input tensor and no side effects between projections.
- Packed weight row order is all-Q rows, then all-K rows, then all-V rows. The conversion script splits original `.in_proj_weight` with `torch.chunk(..., dim=0)` into this order.
- Bias packing follows the same order.

Failure cases:

- Original checkpoint names before conversion, or any backend that expects per-head interleaving.

## 10. Kernel fusion candidates

Highest priority:

- Text tower LayerNorm + QKV GEMM + noncausal attention for S1/L.
- RepMixer depthwise Conv1d/Conv2d + BatchNorm folding for S0 eval inference.
- Detector cross-attention with additive float RPB fallback policy.
- Mask einsum as batched GEMM.

Medium priority:

- MLP Linear/Conv + GELU/ReLU + Linear epilogues.
- FPN nearest upsample + add + 3x3 conv + GroupNorm + ReLU.
- Box RPB generation kernels for `[B,heads,Q,H*W]` to avoid Python-style temporaries.
- Geometry ROIAlign plus pooled projection, once ROIAlign admission is defined.

Lower priority:

- Text positional interpolation, because normal tokenizer length is fixed at 77.
- Dot-product scoring MLP/projection fusion.
- Semantic head fusion, usually small relative to vision and attention.

## 11. Runtime staging plan

Stage 1: parse config and tokenizer/processor metadata. Validate S0/S1/L text config differences and reject unsupported tokenizer changes.

Stage 2: implement text tower only. Cover standard transformer layers plus S0 RepMixer blocks; expose projected token features `[B,T,256]`.

Stage 3: integrate cached text features into a stubbed detector boundary. Validate `get_text_features` semantics and attention mask handling.

Stage 4: add detector encoder/decoder without masks first using dense SDPA/eager attention. Keep vision features as supplied tensors.

Stage 5: add box geometry prompts, ROIAlign, prompt concatenation, and box padding sentinel handling.

Stage 6: add mask decoder and output heads.

Stage 7: integrate SAM3 vision encoder/FPN or accept cached `vision_embeds` as the first production boundary.

Stage 8: optimize guarded rewrites and backend fusions after parity is stable.

Initial stubs: image processor, final postprocessing, full SAM3 vision tower, and geometry prompts can be deferred if first target is text-tower parity or cached-embed detector parity.

## 12. Parity and validation plan

- Tokenizer ABI: compare token IDs and masks for short prompts, padding, EOS, and lowercasing against HF tokenizer files.
- Text position interpolation: random learned position table at T=77 and alternate T values; compare interpolate/reshape path.
- RepMixer block parity: S0 shapes `[B,16,512]` and `[B,77,512]`, eval mode, fp32 tolerance `1e-5`.
- Standard text layer parity: S1/L layer with attention mask, fp32 `1e-5`, fp16/bf16 `1e-2` depending backend.
- Text tower parity: full text encoder last hidden state and projected `[B,T,256]` output.
- Prompt concatenation: text-only, boxes-only synthetic, and mixed prompt masks including `-10` labels.
- Detector layer parity: one DETR encoder layer and one decoder layer with and without RPB.
- Box math parity: `cxcywh->xyxy`, `inverse_sigmoid`, iterative box refinement.
- Mask head parity: mask embedder + pixel decoder + batched GEMM/einsum.
- End-to-end cached-embed parity: use HF-produced `vision_embeds` and DinoML text/detector path before full vision port.
- End-to-end model parity: image + text + optional boxes through processor, compare masks/logits/boxes before postprocess.

## 13. Performance probes

- Tokenization throughput for short prompts and batch prompt sets.
- Text tower throughput by variant: S0 RepMixer, S1 transformer, L transformer.
- Cached text feature reuse across multiple images.
- Cached vision feature reuse across multiple prompts.
- DETR encoder sequence length sweep over vision token count.
- Decoder query sweep around 200 queries and prompt length variation.
- RPB generation and attention backend comparison: eager/SDPA/fallback.
- ROIAlign throughput by number of boxes.
- Mask decoder resolution sweep and mask GEMM bandwidth.
- End-to-end segmentation throughput split into processor, vision, text, detector, mask, postprocess.

## 14. Skip/defer list

- Training, gradients, dropout-training parity, and gradient checkpointing.
- Video tracker/session state; source only ignores tracker keys in this family.
- Multi-GPU/tensor parallel.
- Quantized/packed weights. The inspected source and configs expose ordinary dense weights.
- Full postprocessing parity if first target is graph/runtime parity.
- Full SAM3 vision tower if first target can accept cached `vision_embeds`.
- FlashAttention for full model until additive float RPB mask dispatch is guarded.

## 15. Final implementation checklist

- [ ] Parse `Sam3LiteTextConfig` and nested SAM3 vision/detector configs.
- [ ] Load CLIP tokenizer ABI metadata and enforce max length/EOT/PAD contracts.
- [ ] Implement text embeddings and interpolated learned text position embeddings.
- [ ] Implement S1/L standard text encoder layers.
- [ ] Implement S0 RepMixer blocks with NCHW boundary guards.
- [ ] Implement `get_text_features`: project full token states to detector hidden size.
- [ ] Add text tower parity tests for S0, S1, and L configs.
- [ ] Define cached `text_embeds` and `vision_embeds` input contracts.
- [ ] Implement prompt concatenation and mask conventions.
- [ ] Implement geometry box encoder, including ROIAlign or gated fallback.
- [ ] Implement DETR encoder/decoder dense attention and RPB fallback policy.
- [ ] Implement box refinement and dot-product scoring.
- [ ] Implement mask decoder and `einsum`/batched-GEMM mask head.
- [ ] Add cached-embed detector parity tests before full vision integration.
- [ ] Audit/port inherited SAM3 ViT/FPN vision tower separately.
- [ ] Add guarded graph rewrites for RepMixer, QKV packing, 1x1 conv, patch conv, and mask GEMM.
