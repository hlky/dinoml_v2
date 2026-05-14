# DinoML Transformers Audit: SAM3

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/sam3 for intended production target; tiny-random/sam3 for open config snapshot only.
Config source: source defaults in configuration_sam3.py; tiny-random/sam3 config.json; official facebook/sam3 config/preprocessor reads returned 401.
Source files inspected: configuration_sam3.py, modeling_sam3.py, image_processing_sam3.py, processing_sam3.py, modular_sam3.py, convert_sam3_to_hf.py.
Any missing files or assumptions: video/tracker source is outside this sam3 directory; this report scopes Sam3Model image promptable concept segmentation.
```

`modeling_sam3.py` is generated from `modular_sam3.py`; upstream source edits should target the modular file. Snapshots and notes are under `_sources/`.

## 2. High-level architecture

SAM3 image inference is promptable concept segmentation: image encoder + CLIP text encoder + optional box prompt encoder + DETR-style prompt-fused encoder/decoder + mask head.

```text
image processor -> pixel_values[NCHW]
text/box processor -> input_ids, attention_mask, normalized cxcywh boxes
pixel_values -> ViT backbone -> FPN neck -> multiscale NCHW features
text ids -> CLIP text -> Linear(1024 -> 256) prompt features
optional boxes + FPN feature -> ROIAlign/box geometry encoder -> geometry prompt features
prompt features + FPN finest level -> DETR encoder -> DETR decoder queries/boxes/presence
decoder queries + encoder features + FPN features -> mask decoder -> pred_masks, semantic_seg
processor postprocess -> filtered/scaled boxes and resized binary masks
```

Independently stageable pieces: text features can be cached per prompt, vision FPN features can be cached per image, box geometry prompts can be recomputed per interactive prompt set, and postprocessing is separable CPU/GPU data-pipeline work. There is no autoregressive decode or KV cache in `Sam3Model`.

## 3. Important config dimensions

Source-default production-shaped dimensions:

| Component | Key dimensions |
| --- | --- |
| ViT backbone | image 1008, pretrain image 336, patch 14, tokens 72x72, hidden 1024, layers 32, heads 16, head dim 64, MLP 4736 |
| ViT attention | window size 24 except global layers `[7, 15, 23, 31]`, 2D axial RoPE theta 10000 |
| Vision neck | FPN hidden 256, scale factors `[4.0, 2.0, 1.0, 0.5]`; effective feature maps from 72x72 are about 288x288, 144x144, 72x72, 36x36, then `Sam3Model` drops the last |
| CLIP text | vocab 49408, hidden 1024, projection dim 512, layers 24, heads 16, max positions 32, GELU |
| Text projection | Linear(1024 -> 256) on CLIP last hidden state |
| Geometry encoder | hidden 256, layers 3, heads 8, MLP 2048, ROI size 7, ReLU |
| DETR encoder | hidden 256, layers 6, heads 8, MLP 2048, ReLU |
| DETR decoder | hidden 256, layers 6, heads 8, queries 200 plus 1 presence token, MLP 2048, ReLU |
| Mask decoder | hidden 256, upsampling stages 3, heads 8 for prompt cross-attn, GroupNorm groups 8 |

Representative config sweep:

| Config | Basis | Operator-significant differences |
| --- | --- | --- |
| `Sam3Config()` defaults | `configuration_sam3.py` | Full intended image detector shape above; official config gated so this is the source-default production proxy. |
| `tiny-random/sam3` | open HF config snapshot | hidden sizes 16, text layers 2, ViT layers 8, heads 2, MLP 32; keeps query count 200, ROI size 7, patch 14, image 1008, and same structural stages. |
| `sam3_video` wrapper configs | inferred from `Sam3Model.__init__` and ignored keys | may contain `detector_config`, `tracker_model.*`, `tracker_neck.*`; `Sam3Model` unwraps only detector config and ignores tracker weights. Route to separate video audit. |

## 3a. Family variation traps

- Official `facebook/sam3` is gated here; do not hard-code source defaults as verified checkpoint config until access resolves.
- Tiny-random has config fields like `box_rpb_mode`, `use_presence_token`, `qkv_bias`, `num_feature_levels`, `fpn_kernel_size`, and `fpn_stride` that current in-library source does not read. Treat them as ignored for this source basis.
- ViT code uses NCHW input, Conv2d patching, then NHWC `[B,H,W,C]` internal blocks for LayerNorm, window partition, and RoPE attention. FPN/ROI/mask heads are NCHW. Layout translation must be region-scoped.
- `window_size=24` local attention pads NHWC windows; global attention layers use full 72x72 attention. The window partition/unpartition shape math must stay paired.
- Geometry prompts use TorchVision `roi_align`; this is an external op boundary for first DinoML integration.
- `concat_padded_sequences` uses scatter to insert CLS after valid box prompts; this is bounded indexed row copy, not a general arbitrary scatter if masks are right-padded.
- DETR decoder applies a float additive box relative-position bias for vision cross-attention. Flash Attention is explicitly bypassed when additive float masks are present.
- No NMS in normal object/instance postprocess; NMS appears only in automatic crop mask-generation helper paths.
- Video/tracking state is out of this directory; do not infer memory tracker requirements from `Sam3Model`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- Flatten/transpose/view/reshape/permute for NCHW <-> sequence <-> NHWC <-> NCHW.
- NHWC window partition/unpartition with zero padding on H/W.
- `cat`, `stack`, `pad`, `tile`, `repeat`, `expand`, `clone`, `where`, boolean masks, indexed filtering.
- Bounded `scatter(dim=1)` for right-padded prompt concatenation.

Neural primitives:

- Linear/GEMM with bias for attention, MLPs, box heads, text projection, dot-product scoring.
- Conv2d 1x1/3x3, ConvTranspose2d 2x2 stride 2, MaxPool2d 2x2 stride 2.
- LayerNorm on sequence/NHWC last dimension, GroupNorm(8, 256), GELU/ReLU, dropout as inference no-op.
- ROIAlign on NCHW feature maps, output 7x7, followed by Conv2d(kernel=7) to collapse ROI.

Attention primitives:

- Noncausal MHA, hidden 1024/heads 16/head 64 in ViT and hidden 256/heads 8/head 32 elsewhere.
- ViT local/global self-attention with 2D RoPE on Q/K.
- Geometry self-attention over prompt tokens and cross-attention to vision tokens.
- DETR encoder vision self-attention and text/prompt cross-attention.
- DETR decoder query self-attention, text cross-attention, and vision cross-attention with optional additive box RPB.
- Mask decoder cross-attention from encoder features to combined prompts.

Position/custom math:

- 2D axial RoPE with pairwise rotation.
- Sine/cosine image position embeddings from cumulative non-mask coordinates.
- Box sine/cos encoding in `(y, x, h, w)` concatenation order.
- Inverse sigmoid box refinement with epsilon clamp.
- Box RPB log transform using signed `log2(abs(delta * 8) + 1) / log2(8)`.

Pre/postprocessing-coupled ops:

- Image resize/rescale/normalize/convert RGB to NCHW; original size tracking.
- Text tokenization padded to max length 32.
- Box normalization from absolute xyxy to normalized cxcywh; pad value `-10` masks boxes.
- Postprocess sigmoid scores, presence-score multiplication, box scaling, threshold filtering, mask sigmoid/resizing/binarization, optional non-overlap constraints, optional crop-generation NMS/RLE helpers.

## 5. Layer/block breakdown

ViT block, repeated 32 source-default layers:

```text
x[NCHW] -> Conv2d patch -> tokens[B,5184,1024] + tiled abs pos
tokens -> NHWC[B,72,72,1024]
for each block:
  residual = x
  x = LayerNorm(last dim)
  if local layer: x -> window_partition(B*nwin,24,24,1024)
  q,k,v = Linear(1024 -> 1024) each
  q,k = 2D RoPE(q,k)
  x = noncausal MHA(q,k,v) -> Linear(1024 -> 1024)
  if local layer: window_unpartition
  x = residual + x
  x = x + MLP(LN(x), 1024 -> 4736 -> 1024, GELU)
```

Vision neck:

```text
tokens[B,5184,1024] -> NCHW[B,1024,72,72]
scale 4: ConvTranspose2d 1024->512 -> GELU -> ConvTranspose2d 512->256 -> Conv1x1 -> Conv3x3
scale 2: ConvTranspose2d 1024->512 -> Conv1x1 512->256 -> Conv3x3
scale 1: Conv1x1 1024->256 -> Conv3x3
scale 0.5: MaxPool2d -> Conv1x1 1024->256 -> Conv3x3
```

`Sam3Model` uses `fpn_hidden_states[:-1]`, so the 0.5/downsampled level is not consumed by the image detector path.

Geometry encoder:

```text
boxes[B,N,4] normalized cxcywh
direct = Linear(4 -> 256)
pooled = ROIAlign(NCHW finest FPN, xyxy scaled to feature W/H, 7x7) -> Conv2d(256->256,k=7)
pos = sine(center x/y) + raw h/w -> Linear(258 -> 256)
prompt = direct + pooled + pos + label_embedding
prompt = concat right-padded CLS via bounded scatter
for 3 layers:
  prompt self-attn with padding mask
  prompt cross-attn to flattened vision features + pos
  MLP(256 -> 2048 -> 256)
```

DETR encoder/decoder:

```text
flatten finest FPN [B,256,H,W] -> [B,H*W,256]
encoder x 6:
  vision self-attn on x + pos
  cross-attn to combined text/geometry prompts
  MLP(256 -> 2048 -> 256)

decoder init:
  learned 200 query embeddings + learned 4D reference points + 1 presence token
decoder x 6:
  query_pos = MLP(box_sine(reference_boxes), 512 -> 256)
  self-attn over presence + queries
  text cross-attn
  vision cross-attn with optional box RPB if one spatial level
  MLP(256 -> 2048 -> 256)
  box delta = MLP(LN(query), 256 -> 256 -> 256 -> 4)
  reference = sigmoid(delta + inverse_sigmoid(reference)); detached for next layer
  presence = clamp(MLP(LN(presence)), -10, 10)
```

Mask/scoring heads:

```text
pred_logits = clamp((query_proj(decoder_layers) @ text_proj(mean_pooled_text)) / sqrt(256), -12, 12)
pixel_embed = FPN decoder(backbone features with finest replaced by encoder visual tokens)
instance_embeds = Conv1x1(pixel_embed)
mask_embeddings = MLP(query, 256 -> 256 -> 256 -> 256)
pred_masks = einsum("bqc,bchw->bqhw")
semantic_seg = Conv1x1(pixel_embed, 256 -> 1)
```

## 6. Attention requirements

All attention in `Sam3Model` is noncausal and inference cache-free.

| Attention | Query/key lengths | Width | Mask/bias | Backend note |
| --- | --- | --- | --- | --- |
| ViT local self-attn | each 24x24 window | 16 heads x 64 | none after padding | Flash/SDPA compatible if windowed tensors are materialized. |
| ViT global self-attn | 72x72 tokens | 16 x 64 | none | High cost; likely needs optimized attention. |
| Geometry self-attn | box prompts + CLS | 8 x 32 | bidirectional padding mask | Small sequence. |
| Geometry cross-attn | prompts -> finest FPN tokens | 8 x 32 | none | Prompt count small, KV image tokens. |
| DETR encoder self-attn | finest FPN tokens, usually 72x72 | 8 x 32 | none | Dense noncausal. |
| DETR encoder prompt cross-attn | vision tokens -> text/geometry prompt tokens | 8 x 32 | prompt padding mask | Rectangular attention. |
| DETR decoder self-attn | 201 query tokens | 8 x 32 | none | Small dense. |
| DETR decoder text cross-attn | 201 -> prompt len | 8 x 32 | prompt padding mask | Rectangular. |
| DETR decoder vision cross-attn | 201 -> H*W | 8 x 32 | additive box RPB with zero row for presence | Float additive mask forces SDPA/eager fallback if Flash requested. |
| Mask decoder prompt cross-attn | encoder tokens -> prompt len | 8 x 32 | prompt padding mask | Optional prompt-conditioned refinement. |

There is no KV cache, no autoregressive generation, no sliding-window causal attention, no GQA/MQA, and no packed varlen sequence ABI in this image path. Text encoder is CLIP and should be covered by a separate CLIP text audit or admitted as a composed sub-family.

## 7. Position encoding and custom math

2D ViT RoPE uses precomputed cos/sin per layer/window size. It rotates adjacent pairs, not Llama half-rotation:

```python
def rotate_pairwise(x):
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_sam3_2d_rope(q, k, cos, sin):
    qf = q.float()
    kf = k.float()
    return (qf * cos + rotate_pairwise(qf) * sin).to(q.dtype), (kf * cos + rotate_pairwise(kf) * sin).to(k.dtype)
```

Image sine position embeddings are generated from cumulative non-mask x/y coordinates, optionally normalized to `2*pi`, then concatenated as `[pos_y, pos_x]` and returned NCHW. Box position encodings concatenate `[pos_y, pos_x, pos_w, pos_h]` for decoder boxes, while geometry box coordinate encoding uses `[pos_y, pos_x, height, width]`.

Box RPB:

```text
xyxy = cxcywh_to_xyxy(reference_boxes)
delta_x = coord_w - [x0, x1]
delta_y = coord_h - [y0, y1]
encoded = sign(delta * 8) * log2(abs(delta * 8) + 1) / log2(8)
rpb = MLP_y(encoded_y) + MLP_x(encoded_x)
```

`inverse_sigmoid(x)` clamps `x` to `[0, 1]`, then clamps numerator/denominator by `eps=1e-3` before `log(x/(1-x))`.

## 8. Preprocessing and input packing

Image processor source defaults: RGB conversion, resize to 1008x1008, rescale, ImageNet mean/std normalize, NCHW channel-first tensor, and `original_sizes` recorded before resize. Segmentation labels, when provided, resize to mask size 288x288 with nearest interpolation and int64 output.

Processor text path tokenizes with `padding="max_length", max_length=32`. If text is `None` and boxes are present, it uses the default text prompt `"visual"`.

Box path requires images or `original_sizes`, accepts nested `[image][box][4]` boxes in absolute xyxy, pads missing entries with `-10`, normalizes x by original width and y by original height, preserves pad values, then converts xyxy to normalized cxcywh. Labels default to `1` for real boxes and `-10` for padded entries; model masks labels equal to `-10` and rewrites them to label `0` before embedding.

Postprocessing:

- Object detection: `scores = sigmoid(pred_logits)`, multiply by `sigmoid(presence_logits)` when available, optionally scale xyxy boxes by target `(height,width)`, filter `scores > threshold`. No NMS.
- Instance segmentation: same score/box path, `masks = sigmoid(pred_masks)`, optional bilinear resize with `align_corners=False`, then threshold to long binary masks. No NMS.
- Semantic segmentation: `sigmoid(semantic_seg)`, optional bilinear resize, threshold.
- Automatic mask-generation helpers include crop generation, stability scoring, mask-to-box, RLE conversion, and TorchVision `batched_nms`; treat this as optional/data-pipeline unless DinoML targets automatic crop generation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern: `Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size, bias=False)` followed by flatten/transpose.

Replacement: window flatten NCHW patches in PyTorch Conv2d order -> GEMM with `weight.reshape(hidden, 3*kh*kw).T` -> `[B,Hpatch*Wpatch,hidden]`.

Preconditions: kernel equals stride, padding 0, dilation 1, groups 1, input H/W divisible by patch, source NCHW or fully controlled NHWC rewrite with matching flatten order. Failure cases: dynamic image sizes not divisible by 14, nonstandard channel count, or layout pass cannot prove patch flatten order.

Parity sketch: random NCHW tensor, compare Conv2d+flatten+transpose to rewrite for fp32/fp16.

### Rewrite: mask einsum -> batched GEMM

Source pattern: `torch.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)`.

Replacement: reshape instance embeds to `[B,C,H*W]`, batched GEMM `[B,Q,C] x [B,C,HW] -> [B,Q,HW]`, reshape to `[B,Q,H,W]`.

Preconditions: dense contiguous or explicit strides handled; Q, C, H, W known or runtime shape-buffered; no layout translation inside the equation unless output axes are rewritten. Failure cases: non-contiguous NCHW feature map without accessor support.

### Rewrite: bounded padded prompt concat scatter -> indexed row copy

Source pattern: initialize zeros, copy seq1 prefix, scatter seq2 at `actual_seq1_lengths + arange(seq2_len)`.

Replacement: generated kernel per batch row copies valid right-padded prompt spans and appends CLS/geometry rows.

Preconditions: masks are right-padded and boolean True-valid; `seq2` length small/static; no arbitrary duplicate indices. Failure cases: non-right-padded prompt masks or external caller bypasses processor invariants.

### Rewrite: box RPB precompute per layer/shape

Source pattern: decoder recomputes coordinate grids and RPB MLPs for current reference boxes.

Replacement: precompute coordinate grids for fixed spatial shape; keep MLP dynamic on reference boxes.

Preconditions: single spatial level and fixed H/W per artifact or shape-guarded cache. Failure cases: multi-level spatial shapes or dynamic H/W without cache invalidation.

### Layout rewrite: local NHWC ViT block island

Source pattern: NCHW patch conv -> sequence -> NHWC ViT blocks -> sequence -> NCHW FPN.

Opportunity: preserve NHWC inside ViT LayerNorm/window attention and avoid extra contiguous transposes when attention and MLP kernels accept NHWC-last-dim semantics.

Guard boundaries: NCHW input Conv2d, FPN Conv2d/ConvTranspose/MaxPool, ROIAlign, mask decoder Conv2d/GroupNorm are no-layout-translation boundaries unless a whole subgraph is rewritten. Axis-sensitive attrs: LayerNorm last dim, `pad` tuple for NHWC windows, `flatten(2).transpose(1,2)` for NCHW maps, `interpolate(..., size=shape[-2:])`, GroupNorm channel axis.

## 10. Kernel fusion candidates

Highest priority:

- ViT LayerNorm + QKV projections + 2D RoPE + attention, especially 72x72 global layers and 24x24 local windows.
- FPN ConvTranspose/Conv/GELU and Conv2d neck paths; these dominate high-resolution NCHW feature preparation.
- Mask einsum as batched GEMM and optional Conv1x1 + mask GEMM scheduling.
- DETR decoder vision cross-attention with additive box RPB; preserve additive-bias math and backend fallback semantics.

Medium priority:

- Geometry ROIAlign + 7x7 projection path for interactive box prompts.
- Sine/cos position embedding generation and cached coordinate grids.
- MLP+residual+LayerNorm blocks in DETR encoder/decoder and geometry encoder.
- Dot-product scoring MLP + masked mean pooling + query/text projection.

Lower priority:

- Automatic crop-generation RLE/NMS helpers.
- Non-overlap constraints and connected-component-style postprocess; current source even marks connected components as TODO.
- Training losses, gradient checkpointing, dropout randomness.

## 11. Runtime staging plan

1. Parse `Sam3Config` and load weights for `Sam3Model` image detector only; reject or unwrap `sam3_video.detector_config` while ignoring tracker weights explicitly.
2. Compose or stub CLIP text encoder behind cached `text_embeds[B,32,256]`; first image-graph parity can accept precomputed text embeddings.
3. Implement ViT + FPN vision path with NCHW public ABI and guarded NHWC internal island.
4. Implement DETR encoder/decoder with dense noncausal attention and box RPB eager/SDPA fallback.
5. Implement mask decoder and postprocess object/instance/semantic outputs.
6. Add geometry prompt encoder with ROIAlign; before that, text-only prompts can be the first useful target.
7. Add optimized rewrites/fusions after source-faithful parity is stable.
8. Audit SAM3 Video separately for tracker/session state and frame ABI.

## 12. Parity and validation plan

- Unit parity for custom math: `rotate_pairwise`, 2D RoPE, sine position embeddings, `inverse_sigmoid`, box format conversions, box RPB matrix.
- Patch embedding rewrite parity against Conv2d for static 1008x1008 and at least one smaller guarded size divisible by 14.
- Window partition/unpartition round trip with non-divisible H/W to verify padding/crop.
- One ViT block parity for local and global attention layers.
- FPN level shape/value parity for scale factors 4, 2, 1, 0.5.
- Geometry encoder parity with 0 boxes, padded boxes, and valid boxes; include ROIAlign dtype behavior for bf16 -> fp16.
- DETR decoder one-layer and six-layer parity including reference-box detach/update behavior.
- Mask decoder parity for `einsum` rewrite and postprocess resizing/binarization.
- End-to-end image PCS parity against Transformers with fixed text prompt and optional boxes. Recommended tolerances: fp32 `1e-4` to `1e-5` for logits/masks before postprocess; fp16/bf16 relaxed around attention/ROIAlign/interpolate.

No DinoML tests or imports were run for this audit, per scope.

## 13. Performance probes

- Processor throughput: resize/normalize/tokenize/box normalization separately from model runtime.
- Vision encoder throughput and memory for 1008x1008, split by patch embed, local ViT blocks, global ViT blocks, FPN.
- DETR encoder/decoder throughput vs prompt length and query count 200.
- Geometry prompt latency vs number of boxes; isolate ROIAlign cost.
- Mask decoder cost vs Q=200 and mask resolution 288x288.
- Postprocess throughput for object, instance, semantic, and optional automatic mask-generation helpers.
- Attention backend comparison: eager/SDPA/Flash for no-mask attention and SDPA/eager for additive RPB.
- Layout probe: source-faithful NCHW/NHWC transitions vs guarded NHWC island.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- SAM3 Video tracker, session state, frame propagation, hotstart heuristics, and streaming quality tradeoffs.
- Automatic crop mask-generation as a first target; normal image PCS postprocess does not require its NMS/RLE path.
- General arbitrary scatter; only admit bounded right-padded prompt concat.
- General ROI/NMS provider maturity beyond the specific ROIAlign box prompt path.
- Multi-GPU/tensor parallel and quantized/packed weights; no source-coupled quantization in inspected image source.
- General NHWC rewrite of the whole model; keep layout changes guarded and local.

## 15. Final implementation checklist

- [ ] Parse `Sam3Config` plus source-default fallback fields.
- [ ] Reject unsupported/gated config surprises and route `sam3_video` to detector-only or separate video audit.
- [ ] Load image detector weights and preserve generated-source key mapping assumptions.
- [ ] Provide cached `text_embeds` ABI or compose audited CLIP text encoder.
- [ ] Implement NCHW image preprocessing metadata and original-size ABI.
- [ ] Implement ViT patch embedding, tiled abs position embeddings, NHWC blocks, window partition, and 2D RoPE.
- [ ] Implement FPN neck ConvTranspose/Conv/MaxPool and sine position embeddings.
- [ ] Implement DETR encoder/decoder dense attention, box RPB, inverse-sigmoid box refinement, presence logits.
- [ ] Implement mask decoder pixel FPN, GroupNorm, mask batched GEMM, semantic head.
- [ ] Implement object/instance/semantic postprocess with exact sigmoid, score, scaling, resize, and threshold rules.
- [ ] Add ROIAlign-backed geometry prompt encoder after text-only image PCS parity.
- [ ] Add safe rewrites: patch Conv2d->GEMM, mask einsum->BMM, bounded prompt concat copy, cached coordinate grids.
- [ ] Add parity tests at custom-op, block, stage, and end-to-end levels.
- [ ] Benchmark vision, DETR, geometry, mask, postprocess, and layout variants independently.
