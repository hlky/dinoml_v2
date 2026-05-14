# Transformers audit: deimv2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: deimv2 / Deimv2ForObjectDetection
Primary runtime target: image object detection inference, raw logits/boxes plus RT-DETR postprocess
Config source: native Deimv2Config defaults, one converted HF config, and official original-style Intellindust configs
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/deimv2/modeling_deimv2.py`
- `X:/H/transformers/src/transformers/models/deimv2/configuration_deimv2.py`
- `X:/H/transformers/src/transformers/models/deimv2/modular_deimv2.py`
- `X:/H/transformers/src/transformers/models/deimv2/convert_deimv2_original_pytorch_checkpoint_to_hf.py`
- `X:/H/transformers/src/transformers/models/rt_detr/image_processing_rt_detr.py`
- Backbone source touched for ABI only: `models/hgnet_v2/*`, `models/dinov3_vit/*`, and `backbone_utils.load_backbone`.

Local config snapshots written beside this report:

- `hf_config_harshal_hgnetv2_n_transformers.json`
- `hf_preprocessor_harshal_hgnetv2_n_transformers.json`
- `official_orig_hgnetv2_atto_config.json`
- `official_orig_hgnetv2_femto_config.json`
- `official_orig_hgnetv2_n_config.json`
- `official_orig_dinov3_s_config.json`
- `official_orig_dinov3_l_config.json`

External primary config URLs used:

- [harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers config](https://huggingface.co/harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers/raw/main/config.json)
- [harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers preprocessor](https://huggingface.co/harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers/raw/main/preprocessor_config.json)
- [Intellindust/DEIMv2_HGNetv2_ATTO_COCO config](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO/raw/main/config.json)
- [Intellindust/DEIMv2_HGNetv2_FEMTO_COCO config](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO/raw/main/config.json)
- [Intellindust/DEIMv2_HGNetv2_N_COCO config](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO/raw/main/config.json)
- [Intellindust/DEIMv2_DINOv3_S_COCO config](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO/raw/main/config.json)
- [Intellindust/DEIMv2_DINOv3_L_COCO config](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO/raw/main/config.json)

Missing files or assumptions:

- `modeling_deimv2.py` and `configuration_deimv2.py` are generated. `modular_deimv2.py` is authoritative for future source edits.
- Official Intellindust configs are original project configs, not directly `model_type="deimv2"` Transformers configs. The conversion script maps them into `Deimv2Config`; facts from those files are labeled original-config facts.
- No gated/401 config fetches were encountered. Large checkpoint tensors were not downloaded; weight metadata and exact parameter counts are not claimed.
- HGNetV2 and DINOv3ViT backbone internals are composed dependencies. This report records the DEIMv2 feature contract and high-risk backbone operator classes, but full backbone parity should use separate audits.

## 2. High-level architecture

DEIMv2 is a query-based object detector:

```text
RTDetrImageProcessor
  -> pixel_values NCHW
  -> backbone feature maps
  -> DEIMv2 encoder/FPN/PAN
  -> flatten multiscale maps
  -> anchor generation + top-k proposal selection
  -> query decoder with self-attention + multiscale deformable cross-attention
  -> class logits + normalized center boxes
  -> RT-DETR object-detection postprocess
```

Stage decomposition:

- CPU/data pipeline: image resize, optional rescale, optional normalization, optional padding/pixel mask, annotation conversion for training.
- Backbone: `load_backbone(config)` returns image-like NCHW feature maps. HGNetV2 and DINOv3ViT variants should be independently cacheable and independently validated.
- DEIMv2 encoder: conv projection, optional AIFI transformer on selected levels, top-down FPN, bottom-up PAN, all on NCHW maps.
- Proposal construction: flatten `[B,C,H,W] -> [B,H*W,C]`, concatenate levels, build `spatial_shapes` and `level_start_index`, generate anchors, score all memory positions, top-k query selection.
- Decoder: noncausal query self-attention, custom multiscale deformable cross-attention over the flattened feature memory, iterative box refinement.
- Postprocess: convert normalized center boxes to corner boxes, scale to target image size, score selection and threshold. Source postprocess does not run NMS.

First useful DinoML target: `Deimv2ForObjectDetection` inference for a converted HGNetV2-N style config with fixed `640x640` preprocessed NCHW images, no labels/training denoising, and source postprocess parity.

## 3. Important config dimensions

Native source defaults:

| Field | Effective default / rule | Source basis |
| --- | ---: | --- |
| `d_model` / `hidden_size` | 256 | `Deimv2Config` |
| `head_dim` | `d_model // decoder_attention_heads` | config post-init |
| `decoder_attention_heads` | 8 | `Deimv2Config` |
| `decoder_layers` | 6 | `Deimv2Config` |
| decoder layer applications | `decoder_layers + decoder_layers - eval_idx - 1` when `eval_idx >= 0`, otherwise normal negative-index equivalent | `Deimv2Decoder.__init__` |
| `decoder_ffn_dim` | 1024, but SwiGLU hidden width is `decoder_ffn_dim // 2` | source |
| `num_queries` | 300 | config |
| `num_feature_levels` | 3 | config |
| `decoder_n_points` | scalar 4 or list per level | config validation |
| `encoder_hidden_dim` | 256 | config |
| `encoder_layers` | 1 for hybrid, 0 in converted lite configs | conversion script |
| `encoder_attention_heads` | 8 | config |
| `max_num_bins` | 32, DFL output width `4 * (max_num_bins + 1)` | config/source |
| `reg_scale` | 4.0 | config |
| `top_prob_values` | 4 for LQE top-k per side | config |
| `use_gateway` | true | config |
| `encoder_type` | `"hybrid"` or `"lite"` | config |
| `decoder_method` | `"default"` or `"discrete"` | config/source |
| `anchor_image_size` | optional; if set, anchors/valid mask are cached at init | config/source |

Representative checkpoint sweep:

| Snapshot | Config kind | Backbone | Encoder | `d_model` | heads | layers | FFN | levels | points | queries | Strides / channels |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `hf_config_harshal_hgnetv2_n_transformers.json` | converted HF | HGNetV2 | hybrid | 128 | 8 | 3 | 512 | 2 | `[6,6]` | 300 | strides `[16,32]`, enc in `[512,1024]` |
| `official_orig_hgnetv2_atto_config.json` | original | HGNetV2 Atto | lite | 64 | inferred 8 default | 3 | 160 | 2 | `[4,2]` | 100 | stride `[16]`, enc in `[256]` |
| `official_orig_hgnetv2_femto_config.json` | original | HGNetV2 Femto | lite | 96 | inferred 8 default | 3 | 256 | 2 | `[4,2]` | 150 | stride `[16]`, enc in `[512]` |
| `official_orig_hgnetv2_n_config.json` | original | HGNetV2 B0 | hybrid | 128 | 8 | 3 | 512 | 2 | `[6,6]` | 300 | strides `[16,32]`, enc in `[512,1024]` |
| `official_orig_dinov3_s_config.json` | original | DINOv3 `vit_tiny` STA | hybrid | 192 | 8 | 4 | 512 | 3 | `[3,6,3]` | 300 | strides `[8,16,32]`, enc in `[192,192,192]` |
| `official_orig_dinov3_l_config.json` | original | DINOv3 `dinov3_vits16` STA | hybrid | 224 | 8 | 4 | 1792 | 3 | `[3,6,3]` | 300 | strides `[8,16,32]`, enc in `[224,224,224]` |

## 3a. Family variation traps

- Backbone is not fixed. `backbone_config.model_type == "dinov3_vit"` switches to `Deimv2DINOv3ConvEncoder`; otherwise source uses `Deimv2ConvEncoder` around `load_backbone`.
- Official original configs require conversion. Do not feed the raw `DEIMTransformer`/`HybridEncoder` JSON shape directly into native `Deimv2Config`.
- Lite encoders have no AIFI transformer layers and use avg-pool/adaptive-avg-pool fusion. Hybrid encoders add AIFI dense self-attention on configured feature levels.
- `decoder_n_points` can be a list per level. Deformable attention must support nonuniform per-level sample counts.
- `decoder_method="default"` uses bilinear `grid_sample` with coordinates mapped by `2 * loc - 1`; `"discrete"` uses integer clamp/gather. Treat `"discrete"` as a separate custom attention variant.
- `use_gateway=False` replaces gated cross-attention merge with residual add plus RMSNorm.
- `share_bbox_head=True` aliases bbox MLP modules across decoder layers. Preserve logical weight identity.
- Source supports `reference_points.shape[-1] == 2` and `== 4`, but current proposal construction produces 4D boxes. The 2D branch reshapes with `sequence_length` from encoder memory, so DinoML should initially admit only 4D reference boxes unless a parity test proves the 2D path.
- NCHW is the semantic source layout from processor through conv/feature maps. NHWC/channel-last is an optimization only for fully controlled conv/pool/norm regions.
- Axis-sensitive ops include `cat(..., dim=1)` for channels, `flatten(2).transpose(1,2)`, `softmax(dim=-1)`, `topk(dim=1 or -1)`, `gather(dim=1)`, `stack(dim=1)`, and RT-DETR postprocess `flatten(1)`.
- Training denoising path introduces random tensors, label/box noise, and attention masks. It can be rejected for inference.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW conv-map flatten: `[B,C,H,W] -> [B,H*W,C]`.
- `permute`, `transpose`, `contiguous`, `reshape/view`, `flatten`, `split`, `chunk`, `stack`, `cat`.
- Broadcast add/mul, scalar tensor constants, `where`, `masked_fill`, bool masks.
- `topk`, `max(dim=-1)`, `gather(dim=1)`, `repeat/tile`, `unsqueeze`.

Neural network primitives:

- `Conv2d` 1x1, 3x3 stride 1, 3x3 stride 2, HGNetV2/DINOv3-owned convs, all semantic NCHW.
- `BatchNorm2d`, `FrozenBatchNorm2d`, conv-bn-activation patterns.
- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, nearest/bilinear `interpolate`.
- `Linear` for projections, heads, MLPs, and DFL integral.
- `LayerNorm`, `RMSNorm`.
- Activations: SiLU, GELU, ReLU, sigmoid, softmax, clamp, log, exp, rsqrt, sin/cos.

Attention primitives:

- Dense noncausal MHA for encoder AIFI and decoder query self-attention.
- Custom multiscale deformable cross-attention over image-like feature levels.
- DINOv3 backbone attention and RoPE are composed dependency work, not owned by the DEIMv2 decoder.

Detection/postprocess ops:

- Anchor grid generation with `meshgrid`, normalize, concat, logit transform, valid-range mask.
- Top-k proposal selection over class-score maxima.
- DFL integral: softmax over `max_num_bins + 1`, linear projection by weighting function.
- Box math: `inverse_sigmoid`, `distance2bbox`, center-to-corners and corners-to-center conversion.
- LQE: softmax, top-k probabilities, mean, concat, MLP score adjustment.
- Postprocess: sigmoid/softmax score path, top-k, threshold mask, per-image variable-length records, no NMS.

Preprocessing-coupled ops:

- Resize to exact height/width or aspect-ratio modes from RT-DETR processor.
- Rescale by `1/255`.
- Optional normalize by ImageNet mean/std. Converted HGNetV2-N preprocessor sets `do_normalize=false`.
- Optional padding and `pixel_mask`; converted HGNetV2-N preprocessor sets `do_pad=false`.

## 5. Layer/block breakdown

Input processor:

```text
image -> resize -> rescale/optional normalize -> pixel_values [B,3,H,W]
optional pad -> pixel_mask [B,H,W]
```

Conv encoder, HGNetV2 path:

```text
pixel_values [B,3,H,W]
  -> load_backbone(config).feature_maps: list [B,C_i,H_i,W_i]
  -> per-level ConvNorm 1x1 to encoder_hidden_dim unless encoder_type == "lite"
```

Conv encoder, DINOv3 path:

```text
pixel_values [B,3,H,W]
  -> DINOv3ViTBackbone feature maps, patch-derived NCHW maps
  -> resize semantic maps by bilinear interpolate
  -> STA conv stem/detail maps
  -> cat semantic/detail on channels
  -> 1x1 ConvNorm to encoder_hidden_dim
```

Hybrid encoder:

```text
for configured encode_proj_layers:
  x [B,C,H,W] -> flatten/permute [B,H*W,C]
  pos = 2D sine/cos position embedding [1,H*W,C]
  repeat encoder_layers:
    x = LayerNorm/MHA/residual/LayerNorm/MLP/residual
  x -> [B,C,H,W]

top-down FPN:
  lateral 1x1 ConvNorm
  nearest upsample x2
  fuse by sum or channel concat
  RepNCSPELAN5 conv/CSP block

bottom-up PAN:
  SCDown 1x1 ConvNorm + 3x3 stride-2 ConvNorm
  fuse by sum or channel concat
  RepNCSPELAN5 conv/CSP block
```

Decoder input/proposal:

```text
sources = decoder_input_proj(feature_maps) plus optional extra stride-2 levels
source_flatten = cat_i(flatten_NCHW_to_BLC(source_i), dim=1)
anchors, valid_mask = generate_anchors(spatial_shapes)
memory = valid_mask * source_flatten
output_memory = Linear(d_model,d_model) + LayerNorm
enc_scores = Linear(d_model,num_labels)
enc_boxes_unact = MLP(d_model,d_model,4) + anchors
topk_ind = topk(max(enc_scores,-1), num_queries, dim=1)
target = gather(output_memory, topk_ind).detach() unless learn_initial_query
reference_points_unact = gather(enc_boxes_unact, topk_ind)
```

Decoder layer, repeated:

```text
query_pos = MLP(4,d_model,d_model).clamp(-10,10)
x = self_attention(x, q/k use x + query_pos, v uses x)
x = RMSNorm(residual + x)
cross_input = x + query_pos
cross = multiscale_deformable_attention(cross_input, source_flatten, ref_points)
if use_gateway:
  x = RMSNorm(sigmoid(Linear(cat(residual,cross))).chunk gates applied to residual/cross)
else:
  x = RMSNorm(residual + cross)
x = RMSNorm(x + SwiGLU_FFN(x))
```

Detection heads:

```text
class_embed_i: Linear(d_model,num_labels)
bbox_embed_i: MLP(hidden,hidden,4*(max_num_bins+1))
integral: softmax over bins -> F.linear(weighting_function) -> distance2bbox
LQE: softmax bins -> topk -> mean -> MLP -> add to class scores
```

## 6. Attention requirements

Dense self-attention:

- Noncausal MHA, self-attention only.
- Encoder AIFI: hidden width `encoder_hidden_dim`, heads `encoder_attention_heads`.
- Decoder query attention: hidden width `d_model`, heads `decoder_attention_heads`.
- Q/K projections consume `hidden_states + position_embeddings`; V consumes `hidden_states`.
- Source stores Q/K/V as separate `nn.Linear(hidden, hidden, bias=True)` and output `nn.Linear(hidden, hidden, bias=True)`.
- Shape: input `[B,L,C]`; Q/K/V become `[B,H,L,D]`; attention matmul `[B,H,L,L]`; output `[B,L,C]`.
- No KV cache, no causal mask, no decode loop. `ALL_ATTENTION_FUNCTIONS` can dispatch eager, SDPA, FlashAttention, or flex attention, but first parity can use eager matmul-softmax-matmul.

Multiscale deformable cross-attention:

- Query source: decoder query states `[B,Q,d_model]`.
- Key/value source: flattened multilevel memory `[B,S,d_model]`, where `S=sum_l H_l*W_l`.
- Value is reshaped to `[B,S,H,D]` and split by level in row-major flattened spatial order.
- Reference points for inference are 4D boxes `[B,Q,1,L,4]` after `sigmoid(reference_points_unact).unsqueeze(2)`.
- Sampling offsets: `Linear(d_model, heads * sum(points_per_level) * 2)` -> `[B,Q,H,P_total,2]`.
- Attention weights: `Linear(d_model, heads * P_total)` -> softmax over `P_total`.
- For 4D references:

```python
offset = sampling_offsets * (1 / points_per_level) * reference_points[..., 2:] * offset_scale
sampling_locations = reference_points[..., :2] + offset
```

- Default method maps to PyTorch grid coordinates with `grid = 2 * sampling_locations - 1` and `grid_sample(..., mode="bilinear", padding_mode="zeros", align_corners=False)`.
- Discrete method multiplies normalized coords by `[width,height]`, adds 0.5, casts to int64, clamps, and gathers feature values.
- Output sums sampled values weighted by attention weights, then returns `[B,Q,d_model]`.
- `level_start_index` is constructed but not consumed by the Python deformable attention core.

Admission recommendation: first admit 4D reference boxes, `decoder_method="default"`, fixed feature-level count, explicit per-level `(H,W)`, and no padding mask or mask all true. Add masked value zeroing and discrete mode as separate steps.

## 7. Position encoding and custom math

Hybrid AIFI 2D sine position encoding:

```python
pos_dim = embed_dim // 4
omega = 1.0 / temperature ** (arange(pos_dim) / pos_dim)
grid_h, grid_w = meshgrid(arange(H), arange(W), indexing="ij")
pos = cat([sin(grid_h*omega), cos(grid_h*omega),
           sin(grid_w*omega), cos(grid_w*omega)], dim=1)
```

This is cacheable by `(H,W,embed_dim,temperature,dtype,device)`. If `eval_size` is set and the model is in eval mode, source currently sets `pos_embed=None` inside `Deimv2AIFILayer`, so do not assume position embeddings are always present.

Anchor generation:

```python
grid_xy = stack([grid_x, grid_y], -1) + 0.5
grid_xy[..., 0] /= width
grid_xy[..., 1] /= height
wh = ones_like(grid_xy) * grid_size * (2.0 ** level)
anchors = concat([grid_xy, wh], -1).reshape(1, H*W, 4)
valid = ((anchors > 1e-2) & (anchors < 1 - 1e-2)).all(-1, keepdim=True)
anchors = log(anchors / (1 - anchors))
anchors = where(valid, anchors, finfo(dtype).max)
```

DFL weighting function is non-uniform and depends on `max_num_bins`, `up`, and `reg_scale`; it should be reproduced exactly rather than replaced with a uniform `[0..bins]` projection.

## 8. Preprocessing and input packing

Auto image processor mapping routes `model_type="deimv2"` to `RTDetrImageProcessor` / PIL equivalent.

Converted HGNetV2-N preprocessor snapshot:

```text
do_resize=true, size={"height":640,"width":640}
do_rescale=true, rescale_factor=1/255
do_normalize=false
do_pad=false
image_processor_type="RTDetrImageProcessor"
model_input_names=["pixel_values","pixel_mask"]
```

Source processor defaults differ slightly: `do_normalize=false`, `do_pad=false`, ImageNet mean/std fields are present but only used when normalization is enabled.

Runtime tensor contract:

- `pixel_values`: torch float tensor `[B,3,H,W]`, channel-first.
- `pixel_mask`: optional int64 `[B,H,W]`; DEIMv2 creates an all-ones mask if absent, but the current conv backbone call does not consume it. Deformable attention can still accept masks if provided through training paths.
- `target_sizes`: postprocess-only `[B,2]` in `(height,width)` order.

Postprocess ABI:

- Inputs: `outputs.logits [B,Q,num_labels]`, `outputs.pred_boxes [B,Q,4]` normalized center format.
- Convert boxes to corner format.
- If `target_sizes` provided, scale by `[img_w,img_h,img_w,img_h]`.
- Focal-loss path (`use_focal_loss=True`): sigmoid logits, flatten class/query axis, top-k `num_top_queries`, labels are `index % num_classes`, query indices are `index // num_classes`, then boxes are gathered on query axis.
- Non-focal path: softmax over classes, drop final no-object class, max over labels, optional top-k.
- Per image, return only rows with `score > threshold`.
- No NMS is present in source postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ConvNormAct inference fold

Source pattern:

```text
Conv2d(bias=False) -> BatchNorm2d/FrozenBatchNorm2d -> optional activation
```

Replacement:

```text
Conv2d(with folded weight/bias) -> activation
```

Preconditions:

- Eval mode; BN running stats fixed.
- Conv output channel equals BN feature count.
- Preserve source NCHW axes.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = w * scale[:, None, None, None]
b_fold = beta - running_mean * scale
```

Failure cases: training mode, mutable BN stats, unknown epsilon, channel-last rewrite not proven.

### Rewrite: 1x1 NCHW Conv2d -> GEMM

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `groups=1`, `dilation=1`.
- Input contiguous NCHW or a guarded layout pass owns the region.

Replacement:

```text
transpose/flatten [B,C,H,W] -> [B*H*W,C]
GEMM weight [out,C].T
reshape -> [B,out,H,W]
```

Layout constraints: channel-last optimized form is valid only if every producer/consumer in the conv island is rewritten consistently.

### Rewrite: non-overlap DINOv3 patch Conv2d -> Linear

Preconditions:

- DINOv3 backbone path only.
- `kernel_size == stride == patch_size`, `padding=0`, `groups=1`, input H/W divisible by patch size.

Replacement:

```text
PatchFlatten(NCHW row-major) -> MatMul(weight.reshape(out, C*P*P).T) -> bias -> tokens
```

Failure cases: this belongs to the DINOv3 audit if DinoML composes rather than owns that backbone.

### Rewrite: flatten/transposed multilevel maps as layout view

Source pattern:

```text
source.flatten(2).transpose(1,2)
cat(levels, dim=1)
```

Replacement: a materialized or view-backed `[B,S,C]` multilevel memory buffer with recorded level offsets.

Preconditions:

- Feature maps are dense NCHW.
- Downstream deformable attention receives the same row-major level order.

Parity test: compare gathered `source_flatten` and deformable attention output for several level shapes.

### Rewrite: deformable attention custom kernel

Source pattern:

```text
Linear offsets + Linear weights + softmax + per-level grid_sample + weighted sum
```

Replacement: one provider-backed multiscale deformable attention kernel.

Preconditions:

- 4D reference boxes.
- `decoder_method="default"`.
- Known level count, `points_per_level`, heads, and head dim.
- Bilinear interpolation with zeros padding and `align_corners=False`.

Failure cases: discrete mode, 2D references, noncontiguous level memory, padded masks not yet handled.

### Rewrite: postprocess top-k detector head

Source pattern:

```text
sigmoid(logits).flatten(1) -> topk -> label/index decode -> gather boxes -> threshold
```

Replacement: fused detector postprocess helper returning ragged per-image records.

Preconditions: fixed `num_top_queries`, focal-loss path, no NMS.

Failure cases: non-focal path, caller requires full unfiltered dense outputs.

## 10. Kernel fusion candidates

Highest priority:

- Conv-BN-activation folded or fused kernels for HGNetV2/DEIMv2 conv towers.
- Multiscale deformable attention provider kernel. The eager `grid_sample` loop is the main custom runtime risk.
- Dense MHA for AIFI and decoder query self-attention, starting with matmul-softmax-matmul and later SDPA/FlashAttention.
- Top-k + gather proposal selection for `[B,S,num_labels]` and `[B,S,4]`.

Medium priority:

- RMSNorm and LayerNorm kernels.
- SwiGLU FFN: `Linear gate`, SiLU, multiply, `Linear down`.
- DFL integral and LQE softmax/top-k/MLP sequence.
- NCHW nearest/bilinear resize/interpolate for FPN/DINOv3 STA.

Lower priority:

- Training denoising query construction.
- Postprocess ragged thresholding on GPU.
- DINOv3 backbone RoPE/attention if the first integration composes a separately audited backbone artifact.

## 11. Runtime staging plan

Stage 1: config and ABI admission.

- Parse converted `Deimv2Config`.
- Reject raw original-style configs unless routed through conversion.
- Admit one HGNetV2-N-like fixed image size and `decoder_method="default"`.

Stage 2: preprocessing/postprocess parity.

- Run RT-DETR processor equivalent for fixed resize/rescale.
- Implement postprocess without NMS and compare boxes/scores/labels.

Stage 3: backbone composition.

- Either import a separately audited HGNetV2 backbone or stub the backbone with recorded feature maps.
- Validate DEIMv2 encoder/decoder from feature maps first.

Stage 4: DEIMv2 encoder parity.

- Conv projection, hybrid/lite encoder variants, flatten/level metadata.
- Start with hybrid HGNetV2-N; add lite Atto/Femto after.

Stage 5: proposal and decoder parity.

- Anchor generation, top-k gather, query decoder self-attention, custom deformable attention.
- Validate one decoder layer, then full `eval_idx` output.

Stage 6: DINOv3 variant.

- Compose DINOv3ViT audit, STA fusion path, and three-level deformable attention with `[3,6,3]` points.

Stage 7: optimize.

- Fold conv-BN, add deformable attention kernel, add layout islands, add detector postprocess helper.

## 12. Parity and validation plan

- Processor parity: one PIL image, compare `pixel_values` shape/range and optional `pixel_mask` to Transformers for fixed 640x640.
- Anchor parity: exact anchors and valid mask for representative spatial shapes such as HGNetV2-N `[40x40,20x20]` and DINOv3 `[80x80,40x40,20x20]`.
- Deformable attention random test: compare eager PyTorch core for fixed `[B,Q,H,L,P,D]`; tolerances `fp32 rtol=1e-4/atol=1e-5`, `fp16 rtol=5e-2/atol=5e-3` around bilinear interpolation.
- DFL/integral test: random `pred_corners`, exact weighting function, compare boxes.
- Proposal test: compare `topk_ind`, gathered boxes, and gathered memory for deterministic scores with ties avoided.
- Single decoder-layer parity from random memory/reference points.
- Full decoder parity with backbone feature fixtures.
- End-to-end detector parity against `Deimv2ForObjectDetection` for one converted HF checkpoint when weights are accessible.
- Postprocess parity: verify no NMS, threshold behavior, target-size scaling, focal and non-focal branches.

DinoML tests were intentionally not run for this audit.

## 13. Performance probes

- Processor throughput: resize/rescale/optional pad for image size sweep.
- Backbone-only throughput by family: HGNetV2 lite/hybrid, DINOv3 STA.
- Encoder-only throughput: FPN/PAN convs and AIFI attention separately.
- Deformable attention microbench: vary batch, queries, heads, levels, points, spatial shapes, method default vs discrete.
- Proposal top-k/gather throughput for `S=sum(H_l*W_l)`.
- Decoder-only throughput by layers and `num_queries`.
- End-to-end object detection images/sec for fixed 640 and dynamic original sizes.
- Memory probe for multilevel flattened memory plus deformable-attention temporaries.
- Layout probe: NCHW baseline vs guarded channel-last conv islands, with flatten/deformable boundary costs measured separately.

## 14. Skip/defer list

- Training losses, Hungarian matching, dense one-to-one matching, denoising query construction.
- `labels` input path and random label/box noise.
- Raw original-config ingestion without conversion.
- 2D reference-point deformable attention until source behavior is validated.
- `decoder_method="discrete"` until default bilinear path is stable.
- DINOv3 backbone internals in the first HGNetV2 integration.
- Gated/remote-code checkpoints if any future repo requires `trust_remote_code=True`.
- NMS, because source postprocess does not perform it.
- Multi-GPU, quantization, and checkpoint tensor metadata beyond normal dense weights.

## 15. Final implementation checklist

- [ ] Parse converted `Deimv2Config` and reject unsupported raw original configs.
- [ ] Load RT-DETR image processor config and implement fixed resize/rescale NCHW ABI.
- [ ] Compose or stub the backbone feature-map contract.
- [ ] Implement NCHW Conv2d, BN/FrozenBN, pool, interpolate, concat/split/flatten/transpose coverage needed by encoder.
- [ ] Implement LayerNorm and RMSNorm.
- [ ] Implement dense noncausal MHA for AIFI/query self-attention.
- [ ] Implement anchor generation and valid-mask application.
- [ ] Implement top-k proposal selection and gather.
- [ ] Implement DEIMv2 multiscale deformable attention default bilinear path.
- [ ] Implement SwiGLU FFN and gateway merge.
- [ ] Implement DFL integral, weighting function, distance2bbox, LQE.
- [ ] Preserve bbox-head aliasing when `share_bbox_head=True`.
- [ ] Implement source postprocess: center-to-corner, target-size scaling, sigmoid/softmax score path, top-k, threshold, no NMS.
- [ ] Add one-layer, encoder, decoder, and end-to-end parity tests.
- [ ] Add performance probes for conv encoder, deformable attention, top-k/gather, and full detector.
