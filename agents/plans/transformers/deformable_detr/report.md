# Deformable DETR DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: SenseTime/deformable-detr, plus structural variants listed below
Config source: Hugging Face config.json and preprocessor_config.json snapshots under _sources/hf_configs
Source files inspected:
- X:/H/transformers/src/transformers/models/deformable_detr/configuration_deformable_detr.py
- X:/H/transformers/src/transformers/models/deformable_detr/modeling_deformable_detr.py
- X:/H/transformers/src/transformers/models/deformable_detr/modular_deformable_detr.py
- X:/H/transformers/src/transformers/models/deformable_detr/image_processing_deformable_detr.py
- X:/H/transformers/src/transformers/models/deformable_detr/image_processing_pil_deformable_detr.py
Any missing files or assumptions: no gated/401/403 checkpoint gaps observed. model.safetensors.index.json and image_processor_config.json were absent/404 for sampled repos; preprocessor_config.json is present.
```

The generated `modeling_deformable_detr.py` states it is generated from `modular_deformable_detr.py`; future Transformers source edits should inspect the modular file first, while runtime parity should follow the generated modeling file.

Source URLs:
- [configuration_deformable_detr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deformable_detr/configuration_deformable_detr.py)
- [modeling_deformable_detr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deformable_detr/modeling_deformable_detr.py)
- [modular_deformable_detr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deformable_detr/modular_deformable_detr.py)
- [image_processing_deformable_detr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deformable_detr/image_processing_deformable_detr.py)

Representative configs inspected:
- [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr)
- [SenseTime/deformable-detr-with-box-refine](https://huggingface.co/SenseTime/deformable-detr-with-box-refine)
- [SenseTime/deformable-detr-with-box-refine-two-stage](https://huggingface.co/SenseTime/deformable-detr-with-box-refine-two-stage)
- [SenseTime/deformable-detr-single-scale](https://huggingface.co/SenseTime/deformable-detr-single-scale)
- [SenseTime/deformable-detr-single-scale-dc5](https://huggingface.co/SenseTime/deformable-detr-single-scale-dc5)
- [facebook/deformable-detr-detic](https://huggingface.co/facebook/deformable-detr-detic)
- [facebook/deformable-detr-box-supervised](https://huggingface.co/facebook/deformable-detr-box-supervised)

## 2. High-level architecture

Primary runtime target: object detection inference with `DeformableDetrForObjectDetection`.

Architecture: image preprocessor + ResNet/AutoBackbone feature pyramid + multi-scale deformable-attention encoder + learned-query decoder + classification and box heads.

```text
image resize/rescale/normalize/pad -> NCHW pixel_values + pixel_mask
  -> convolutional backbone feature maps + downsampled masks
  -> per-level Conv2d projection + GroupNorm + 2D position embeddings + level embeddings
  -> flatten/concat multi-scale tokens
  -> deformable-attention encoder
  -> learned query decoder or two-stage top-k proposal decoder
  -> per-decoder-layer class logits and box deltas
  -> final logits/pred_boxes -> top-k sigmoid postprocess -> boxes/scores/labels
```

The backbone feature contract is image-like NCHW feature maps with masks. The transformer contract is flattened `[batch, sum(H_l * W_l), d_model]` memory plus `spatial_shapes`, `level_start_index`, `valid_ratios`, and per-token masks.

## 3. Important config dimensions

| Field | Effective value in sampled configs | Source |
| --- | --- | --- |
| `d_model` / hidden size | 256 | config.json/source default |
| encoder layers | 6 | config.json/source default |
| decoder layers | 6 | config.json/source default |
| encoder heads / decoder heads | 8 / 8 | config.json/source default |
| head dim | 32 (`256 / 8`) | source equation |
| encoder FFN / decoder FFN | 1024 / 1024 | config.json/source default |
| activation | `relu` | config.json/source default |
| object queries | 300, except two-stage uses `two_stage_num_proposals=300` as decoder queries | config/source |
| feature levels | 4 baseline/refine/two-stage/DETIC, 1 single-scale/DC5 | config.json |
| sampled points | encoder 4, decoder 4 per head per feature level | config.json/source default |
| position embedding | sine, learned supported by source | config.json/source |
| backbone | legacy `backbone="resnet50"` or nested `backbone_config` ResNet with `out_features=["stage2","stage3","stage4"]` | config.json |
| dtype | `torch_dtype="float32"` in sampled configs | config.json |
| cache support | no autoregressive KV cache | source |
| custom kernel hook | `MultiScaleDeformableAttention` decorated for `kernels-community/deformable-detr`; Python fallback uses `grid_sample` | source |

Representative sweep:

| Model | Feature levels | Dilation | Box refine | Two-stage | Labels from `id2label` | Backbone contract |
| --- | ---: | --- | --- | --- | ---: | --- |
| `SenseTime/deformable-detr` | 4 | false | false | false | 91 | legacy `resnet50`; config default resolves selected stages |
| `SenseTime/deformable-detr-with-box-refine` | 4 | false | true | false | 91 | legacy `resnet50` |
| `SenseTime/deformable-detr-with-box-refine-two-stage` | 4 | false | true | true | 91 | legacy `resnet50` |
| `SenseTime/deformable-detr-single-scale` | 1 | false | false | false | 91 | legacy `resnet50`, one feature level |
| `SenseTime/deformable-detr-single-scale-dc5` | 1 | true | false | false | 91 | legacy `resnet50`, DC5 dilation |
| `facebook/deformable-detr-detic` | 4 | omitted/effective false | true | true | 1203 | nested ResNet `stage2,stage3,stage4` |
| `facebook/deformable-detr-box-supervised` | 4 | omitted/effective false | true | true | 1203 | nested ResNet `stage2,stage3,stage4` |

## 3a. Family variation traps

- `num_feature_levels=1` removes extra lower-resolution 3x3 stride-2 projected levels and changes deformable attention `n_levels`.
- `dilation=True` only applies when using a timm backbone per config docs; sampled DC5 config is a legacy `backbone="resnet50"` config, so DinoML should verify the resolved backbone contract before relying on output stride.
- `with_box_refine=True` changes decoder behavior by wiring `bbox_embed` into the decoder and updating reference points after each layer.
- `two_stage=True` requires `with_box_refine=True`; it replaces learned query embeddings with top-k encoder proposals and adds proposal-generation heads.
- Large-vocabulary DETIC-style configs keep the same operator body but widen `class_embed` to 1203 labels.
- `disable_custom_kernels` is a config field, and `DeformableDetrMultiscaleDeformableAttention` stores it, but the inspected Python fallback path calls `self.attn(...)` unconditionally. DinoML should treat custom-kernel admission as a runtime/provider decision, not just a model config fact.
- NCHW is the semantic source layout for image tensors, Conv2d, GroupNorm, pixel masks, and `grid_sample`; NHWC is a guarded optimization candidate only inside local convolution/deformable-sampling kernels.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW image tensors, batch padding, mask downsample by interpolate, flatten spatial `[B,C,H,W] -> [B,H*W,C]`, concat over sequence/feature-level axis.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `split`, `stack`, `cat`, `gather`, `topk`, `masked_fill`, boolean inversion.
- `spatial_shapes` int tensor `[L,2]`, `level_start_index` cumulative product, `valid_ratios` `[B,L,2]`.

Neural network primitives:
- Backbone ops from ResNet/AutoBackbone: Conv2d, frozen BatchNorm2d, residual blocks, pooling/activation.
- Input projections: Conv2d `C_l -> 256, kernel=1` for backbone levels; extra levels use Conv2d `256 -> 256, kernel=3, stride=2, padding=1`; GroupNorm(32, 256).
- Linear projections with bias: deformable value/output `Linear(256 -> 256)`, sampling offsets `Linear(256 -> heads * levels * points * 2)`, attention weights `Linear(256 -> heads * levels * points)`.
- Decoder self-attention q/k/v/o `Linear(256 -> 256)` with bias.
- FFN `Linear(256 -> 1024) -> ReLU -> Linear(1024 -> 256)`.
- Detection heads: class `Linear(256 -> num_labels)`; box MLP `256 -> 256 -> 256 -> 4`.

Attention primitives:
- Noncausal decoder self-attention, 8 heads, head dim 32.
- Multi-scale deformable attention for encoder self-attention and decoder cross-attention.
- Softmax over `num_feature_levels * n_points`, not over all memory tokens.

Position/custom math ops:
- 2D sine position embedding from valid mask cumulative sums.
- Optional learned row/column embeddings, with fixed embedding tables of length 50 in source.
- `inverse_sigmoid`, sigmoid box/reference refinement, proposal grid generation.

Preprocessing-coupled ops:
- Resize shortest edge 800 / longest edge 1333, rescale/normalize by ImageNet mean/std, pad to max batch height/width, emit `pixel_mask`.
- Object-detection postprocess: sigmoid class scores, flatten query-class scores, top-k, threshold, center-to-corner box conversion, optional scale by target sizes. No NMS in source postprocess.

## 5. Layer/block breakdown

Backbone and projections:

```text
pixel_values: [B, 3, Hpad, Wpad], pixel_mask: [B, Hpad, Wpad]
features = backbone(pixel_values) -> [(B, C_l, H_l, W_l), ...]
mask_l = interpolate(pixel_mask, size=(H_l, W_l)).bool()
source_l = Conv2d(C_l -> 256, k=1) or Conv2d(256 -> 256, k=3, stride=2, pad=1)
source_l = GroupNorm(32, 256)(source_l)
pos_l = sine_position(mask_l) -> [B, H_l*W_l, 256]
source_l = flatten NCHW -> [B, H_l*W_l, 256]
memory_input = cat_l(source_l)
mask_flatten = cat_l(mask_l.flatten(1))
```

Encoder layer, repeated 6 times:

```text
x: [B, S, 256], S=sum_l(H_l*W_l)
x_attn = MultiScaleDeformableAttention(
  hidden_states=x + pos,
  encoder_hidden_states=x,
  mask=[B,S],
  reference_points=[B,S,L,2],
  spatial_shapes=[L,2])
x = LayerNorm(x + Dropout(x_attn))
x = LayerNorm(x + Dropout(Linear(1024 -> 256)(Dropout(ReLU(Linear(256 -> 1024)(x))))))
```

Decoder layer, repeated 6 times:

```text
target: [B, Q, 256], query_pos: [B, Q, 256]
self_qkv = Linear(target + query_pos -> q,k), Linear(target -> v)
target = LayerNorm(target + MHA(q,k,v), noncausal 8 heads)
cross = MultiScaleDeformableAttention(
  hidden_states=target + query_pos,
  encoder_hidden_states=memory,
  mask=[B,S],
  reference_points=[B,Q,L,2 or 4],
  spatial_shapes=[L,2])
target = LayerNorm(target + cross)
target = LayerNorm(target + FFN(256 -> 1024 -> 256))
if with_box_refine: reference_points = sigmoid(bbox_embed[layer](target) + inverse_sigmoid(previous_reference))
```

Two-stage mode:

```text
object_query, proposals = gen_encoder_output_proposals(memory, ~mask_flatten, spatial_shapes)
proposal_logits = class_embed[-1](object_query)
proposal_boxes_logits = bbox_embed[-1](object_query) + proposals
topk = topk(proposal_logits[..., 0], two_stage_num_proposals)
reference_points = sigmoid(gather(proposal_boxes_logits, topk))
query_embed, target = split(LayerNorm(Linear(proposal_pos_embed(topk_boxes))), 256)
```

## 6. Attention requirements

Decoder self-attention:
- Noncausal MHA over object queries; no KV cache and no autoregressive decode.
- Q/K get `target + query_pos`; V gets `target`.
- Shapes: `[B,Q,256] -> [B,8,Q,32]`; attention scores `[B,8,Q,Q]`.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS` and supports eager/SDPA/Flash/Flex for this self-attention only.

Multi-scale deformable attention:
- Used as encoder self-attention with query length `S` and K/V source `S`, and as decoder cross-attention with query length `Q=300` and K/V source `S`.
- MHA-style value layout: `value_proj(memory) -> [B,S,8,32]`.
- Sampling offsets shape `[B,Nq,8,L,4,2]`; attention weights shape `[B,Nq,8,L,4]`.
- For 2D reference points: `sampling_locations = reference_points + offsets / [W_l,H_l]`.
- For 4D box references: `sampling_locations = ref_xy + offsets / n_points * ref_wh * 0.5`.
- Masking zeros padded values before sampling: `value.masked_fill(~attention_mask[...,None], 0)`.
- Python fallback uses per-level `grid_sample` with bilinear interpolation, zero padding, `align_corners=False`, then weighted sum over `L * points`.
- No sliding-window, ALiBi, RoPE, or KV cache.

Eager fallback risk: the per-level loop over `grid_sample` creates heavy temporary tensors `[B*heads, head_dim, Nq, points]` per level and is likely too slow for production. DinoML should prioritize a fused multi-scale deformable attention kernel.

## 7. Position encoding and custom math

Sine position embedding can be reproduced as:

```python
def deformable_detr_sine_pos(mask, num_feats=128, temperature=10000, scale=2 * pi):
    y = mask.cumsum(1, dtype=float)
    x = mask.cumsum(2, dtype=float)
    y = (y - 0.5) / (y[:, -1:, :] + 1e-6) * scale
    x = (x - 0.5) / (x[:, :, -1:] + 1e-6) * scale
    dim = temperature ** (2 * floor(arange(num_feats) / 2) / num_feats)
    pos_x = stack([sin((x[..., None] / dim)[..., 0::2]), cos((x[..., None] / dim)[..., 1::2])]).flatten(-2)
    pos_y = stack([sin((y[..., None] / dim)[..., 0::2]), cos((y[..., None] / dim)[..., 1::2])]).flatten(-2)
    return cat([pos_y, pos_x], -1).flatten_hw_to_sequence()
```

Reference point generation:

```python
def encoder_reference_points(spatial_shapes, valid_ratios):
    refs = []
    for level, (h, w) in enumerate(spatial_shapes):
        y, x = meshgrid(linspace(0.5, h - 0.5, h), linspace(0.5, w - 0.5, w))
        ref = stack([x.reshape(-1) / (valid_ratios[:, level, 0:1] * w),
                     y.reshape(-1) / (valid_ratios[:, level, 1:2] * h)], -1)
        refs.append(ref)
    return cat(refs, 1)[:, :, None] * valid_ratios[:, None]
```

Box refinement:

```python
def refine_reference(delta, reference):
    if reference.shape[-1] == 4:
        return sigmoid(delta + inverse_sigmoid(reference))
    out = delta.copy()
    out[..., :2] = delta[..., :2] + inverse_sigmoid(reference)
    return sigmoid(out)
```

Position grids depend on dynamic image/mask sizes and valid ratios. `level_embed`, learned query embeddings, learned row/column positions, class heads, and box heads are weights.

## 8. Preprocessing and input packing

CPU/data pipeline:
- Accept images, convert to tensor, resize using bilinear resampling with `size={"shortest_edge":800,"longest_edge":1333}` in sampled configs.
- Rescale by `1/255` when the preprocessor enables `do_rescale`; normalize by ImageNet mean/std.
- Pad every image in a batch to max resized height/width unless explicit `pad_size` is provided.
- Emit `pixel_values [B,3,Hpad,Wpad]` and `pixel_mask [B,Hpad,Wpad]`, where 1 means valid pixel and 0 means padding.

GPU/runtime graph:
- Downsample `pixel_mask` to each backbone/projected feature level.
- Compute valid ratios per feature level from masks.
- Build flattened memory and feature-level metadata.

Postprocess:
- Inputs: `outputs.logits [B,Q,num_labels]`, `outputs.pred_boxes [B,Q,4]` in normalized center format, optional `target_sizes [B,2]`.
- Applies sigmoid independently per class, flattens query-class pairs, top-k before threshold, gathers boxes, converts center to corners, scales by `(width,height,width,height)`, and filters by `score > threshold`.
- No source NMS. First DinoML parity should preserve duplicate boxes if source returns them.

Training-only annotation conversion, bipartite matching, panoptic masks, and losses are not required for first inference integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 input projection Conv2d -> per-pixel Linear

Preconditions:
- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Consumer either stays NCHW for GroupNorm or a guarded local layout region rewrites GroupNorm axes.

Replacement:

```text
NCHW -> NHWC/view [B*H*W,C] -> MatMul(weight.T) -> BiasAdd -> reshape
```

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels)
```

Failure cases: extra projected levels use 3x3 stride-2 Conv2d and require a general convolution or im2col lowering.

### Rewrite: FrozenBatchNorm2d -> affine scale/bias

Preconditions:
- Module is `DeformableDetrFrozenBatchNorm2d`, running stats and affine buffers are constants.
- NCHW semantic axis is channel axis 1 unless a guarded layout pass rewrites to NHWC axis -1.

Replacement:

```text
y = x * (weight / sqrt(running_var + 1e-5)) + (bias - running_mean * scale)
```

### Rewrite: top-k object detection postprocess as runtime plugin

Preconditions:
- Inference-only object detection output parity.
- No NMS is inserted.

Replacement:

```text
Sigmoid -> Flatten(query,class) -> TopK -> GatherBoxes -> CenterToCorners -> Scale -> Threshold
```

Failure cases: downstream applications expecting COCO-style NMS must add it outside parity mode.

### Layout guarded region: deformable attention sampling

Preconditions:
- DinoML kernel owns value projection, feature-level split, sampling, weighted sum, and output projection.
- `spatial_shapes`, level order, and `align_corners=False` are preserved exactly.
- Mask semantics remain 1 valid / 0 padding and value masking happens before sampling.

Replacement:

```text
value_proj + mask_zero + offsets/weights + bilinear multiscale gather + weighted sum + output_proj
```

Layout constraints:
- Source fallback reshapes to NCHW for `grid_sample`; an NHWC kernel may avoid NCHW materialization internally.
- Surrounding graph should use a conceptual `no_layout_translation()` guard unless all axis-sensitive `flatten(2)`, `transpose(1,2)`, `split`, and mask operations are controlled by the fused kernel.

## 10. Kernel fusion candidates

Highest priority:
- Fused multi-scale deformable attention: removes per-level `grid_sample` loop and temporary tensors; required for production throughput.
- Value projection + mask zeroing + multiscale sampling: avoids writing masked full memory and re-reading per level.
- Decoder box-refinement head + inverse sigmoid + sigmoid update: small but repeated and parity-sensitive for `with_box_refine`.

Medium priority:
- Conv2d projection + GroupNorm for feature levels.
- 2D sine position embedding and level embedding add for common image sizes.
- Decoder self-attention via existing noncausal MHA backend over `Q=300`.
- Detection postprocess top-k/gather/box-scale kernel for batched outputs.

Lower priority:
- FrozenBatchNorm folding into backbone Conv2d weights where backbone weights are static.
- Learned position embedding path; sampled configs use sine.
- Auxiliary outputs/loss paths for training diagnostics.

## 11. Runtime staging plan

1. Parse config and reject unsupported combinations only after recording `num_feature_levels`, `two_stage`, `with_box_refine`, `dilation`, and backbone contract.
2. Compose or reuse a separately audited ResNet/AutoBackbone implementation; first parity can use exported backbone feature tensors as stubs.
3. Implement projection, mask downsample, position embedding, flatten/metadata construction.
4. Implement Python-equivalent multi-scale deformable attention with `grid_sample` semantics for parity.
5. Run encoder-only parity on random feature maps and real preprocessed image metadata.
6. Add decoder learned-query path, then `with_box_refine`.
7. Add two-stage proposal generation/top-k path.
8. Add detection heads and postprocess parity.
9. Replace deformable attention fallback with optimized CUDA/NHWC-friendly fused kernel under strict guards.

## 12. Parity and validation plan

- Random tensor test for `inverse_sigmoid`, reference point generation, and 2D sine position embedding; fp32 tolerance `1e-5`.
- Random tensor test for multi-scale deformable attention against source fallback using varied `L`, `H_l/W_l`, `Q`, masks, 2D and 4D reference points; fp32 `1e-4`, fp16/bf16 relaxed around bilinear interpolation.
- Single encoder layer parity with fixed `spatial_shapes`, valid ratios, and masks.
- Single decoder layer parity for self-attention plus deformable cross-attention.
- End-to-end baseline `SenseTime/deformable-detr` logits and boxes on one resized/padded image.
- Variant parity for single-scale, box-refine, and two-stage configs.
- Postprocess parity: top-k, threshold, target-size scaling, and explicit no-NMS behavior.

No DinoML tests or Transformers execution were run for this audit, per scope.

## 13. Performance probes

- Preprocessor throughput: resize/rescale/normalize/pad and pixel-mask creation by image resolution.
- Backbone-only throughput for selected ResNet stages.
- Projection + flatten + position embedding throughput by batch and image resolution.
- Encoder deformable attention throughput by `S`, feature levels, heads, and points.
- Decoder throughput by query count and two-stage proposal count.
- Fallback `grid_sample` attention versus fused kernel comparison.
- Memory probe for temporary sampling tensors and flattened multi-scale memory.
- Postprocess latency for `B`, `Q=300`, and label counts 91 versus 1203.

## 14. Skip/defer list

- Training losses, Hungarian matching, auxiliary loss reporting, and annotation conversion.
- Panoptic/segmentation mask training helpers.
- Backbone fine-tuning controls and gradient checkpointing.
- Learned position embedding until a representative inference checkpoint requires it.
- Timm-specific DC5/dilation parity until a checkpoint with a resolved timm backbone contract is selected.
- Custom Hub kernel loading; DinoML should provide its own provider/kernel path.

## 15. Final implementation checklist

- [ ] Parse `DeformableDetrConfig` including `num_feature_levels`, `two_stage`, `with_box_refine`, `dilation`, and nested `backbone_config`.
- [ ] Load backbone and detection-head weights while preserving shared/tied head aliases for non-refine variants.
- [ ] Implement preprocessing contract for `pixel_values` and `pixel_mask`.
- [ ] Implement NCHW backbone feature contract and mask downsampling.
- [ ] Implement input projection Conv2d + GroupNorm levels.
- [ ] Implement 2D sine position embedding and `valid_ratios`.
- [ ] Implement `spatial_shapes` and `level_start_index` metadata.
- [ ] Implement multi-scale deformable attention fallback with `grid_sample` parity.
- [ ] Implement decoder self-attention and FFN blocks.
- [ ] Implement iterative box refinement and two-stage proposal path.
- [ ] Implement detection postprocess with sigmoid/top-k/threshold/box scaling and no NMS.
- [ ] Add guarded NHWC/fused deformable attention provider.
- [ ] Add parity tests for baseline, single-scale, box-refine, two-stage, and DETIC label-width configs.
- [ ] Benchmark preprocessing, backbone, deformable attention, decoder, and postprocess separately.
