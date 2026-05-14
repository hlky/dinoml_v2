# DPT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: dpt family; representative checkpoints listed below
Config source: official Hugging Face config/preprocessor JSON from each model repo, main revision SHA recorded below
Source files inspected:
- X:/H/transformers/src/transformers/models/dpt/modeling_dpt.py
- X:/H/transformers/src/transformers/models/dpt/configuration_dpt.py
- X:/H/transformers/src/transformers/models/dpt/image_processing_dpt.py
- X:/H/transformers/src/transformers/models/dpt/image_processing_pil_dpt.py
- X:/H/transformers/src/transformers/models/dpt/modular_dpt.py
- X:/H/transformers/tests/models/dpt/test_modeling_dpt.py
- X:/H/transformers/tests/models/dpt/test_modeling_dpt_hybrid.py
- X:/H/transformers/tests/models/dpt/test_modeling_dpt_auto_backbone.py
- X:/H/transformers/tests/models/dpt/test_image_processing_dpt.py
Any missing files or assumptions: no remote-code files were required. AutoBackbone variants route into BEiT, SwinV2, DINOv2, or BiT implementations through Transformers backbone loading; this report treats those backbone internals as operator-significant variation but does not replace the separate family audits for those model bodies.
```

`modeling_dpt.py` is the runtime-authoritative PyTorch implementation. `modular_dpt.py` only covers the modular image processor source for the generated/new processor path in this checkout.

Representative config sources:

| Checkpoint | HF repo SHA | Task/head | Config/preprocessor notes |
|---|---:|---|---|
| `Intel/dpt-large` | `bc15f29aa3a80d532f2ed650b5e16ac48d8958f9` | depth | Native ViT DPT, 384 input, preprocessor `size=384`, mean/std 0.5. |
| `Intel/dpt-large-ade` | `5ca59e2439c8340340eafccba2377816903f444a` | segmentation | Native ViT DPT, ADE labels from `id2label` length 150, preprocessor `size=480`. |
| `Intel/dpt-hybrid-midas` | `11eaf7a1cf4bd70740697dbc216f98980c0aeb03` | depth | Hybrid BiT backbone plus ViT, `is_hybrid=true`. |
| `Intel/dpt-beit-base-384` | `e22f6ad701b672c643643c403a3fa7da27515222` | depth | AutoBackbone BEiT, relative position bias and layer scale inside backbone. |
| `Intel/dpt-swinv2-tiny-256` | `f7f350e1a10a5ea58671b68ab5f49dc18ec00483` | depth | AutoBackbone SwinV2, hierarchical window attention; DPT neck skips reassemble. |
| `facebook/dpt-dinov2-small-kitti` | `ae22d4d5332a62d4c6a24bf75ba8e77391c4238b` | depth | AutoBackbone DINOv2, patch 14, `add_projection=true`, preprocessor pads to divisor 14. |

## 2. High-level architecture

DPT is a dense prediction vision model, not a text generation model. The primary runtime targets are depth estimation and semantic segmentation.

```text
image preprocessing -> vision backbone / ViT encoder -> selected intermediate features
  -> DPT reassemble stage -> 3x3 channel projection convs -> feature fusion pyramid
  -> dense prediction head -> optional postprocess resize / argmax
```

Stage decomposition:

| Stage | Runtime contract | Independently testable? |
|---|---|---|
| CPU/data pipeline | Resize, optional aspect-ratio/multiple constraint, rescale, normalize, optional center pad, optional label reduction. Emits `pixel_values` in NCHW. | Yes, processor-only tests. |
| Backbone/encoder | Native ViT path, hybrid BiT+ViT path, or AutoBackbone feature-map path. Emits four intermediate features. | Yes, compare selected hidden states / feature maps. |
| DPT neck | Reassemble token sequences into NCHW feature maps unless using SwinV2 feature maps, then project channels to `fusion_hidden_size` and fuse top-down. | Yes, feed synthetic hidden-state lists. |
| Head | Depth head emits `[B,H,W]`; segmentation head emits `[B,num_labels,H,W]`. | Yes, head-only tests. |
| Postprocess | Depth: optional bicubic resize per image. Segmentation: optional bilinear resize logits then channel argmax. | Yes, CPU/Torch postprocess parity. |

## 3. Important config dimensions

Config defaults from `DPTConfig`:

| Field | Default / behavior |
|---|---|
| `model_type` | `dpt` |
| `image_size`, `patch_size`, `num_channels` | `384`, `16`, `3` for native DPT; may be `None` when `backbone_config` owns these. |
| `hidden_size`, `num_hidden_layers`, `num_attention_heads` | `768`, `12`, `12` for native DPT defaults. |
| `intermediate_size`, `hidden_act` | `3072`, `gelu`. |
| `qkv_bias` | `True` for native ViT attention. |
| `backbone_out_indices` | `(2,5,8,11)` unless AutoBackbone config is supplied, then set to `None`. |
| `readout_type` | `project`; valid values are `ignore`, `add`, `project`; hybrid requires `project`. |
| `reassemble_factors` | `(4,2,1,0.5)`. |
| `neck_hidden_sizes` | `(96,192,384,768)`. |
| `fusion_hidden_size` | `256`. |
| `head_in_index` | `-1`, the final fused feature. |
| `use_batch_norm_in_fusion_residual` | `False`; fusion residual conv bias defaults to `not use_batch_norm`. |
| `add_projection` | `False`; some depth checkpoints set `True`. |
| `use_auxiliary_head` | `True`, but auxiliary logits are training-oriented. |

Representative checkpoint sweep:

| Checkpoint | Backbone path | Input / patch | Encoder dims | Selected features | Neck/head variation | Processor |
|---|---|---|---|---|---|---|
| Tests native tiny | Native ViT | `32`, patch `16` | hidden `32`, layers `2`, heads `4`, MLP `37` | indices `[0,1,2,3]`, but tiny tests use 2 `neck_hidden_sizes` | fusion hidden `32`; depth and segmentation output `32x32` | synthetic tensors. |
| `Intel/dpt-large` | Native ViT | `384`, patch `16`, grid `24x24` | hidden `1024`, layers `24`, heads `16`, MLP `4096` | `[5,11,17,23]` | neck `[256,512,1024,1024]`, depth head | resize 384, mean/std `[0.5]*3`. |
| `Intel/dpt-large-ade` | Native ViT | `384`, patch `16`, grid `24x24` | same as dpt-large | `[5,11,17,23]` | segmentation head, ADE `id2label` length 150 | resize 480 in preprocessor. |
| `Intel/dpt-hybrid-midas` | BiT backbone + native ViT | `384`, patch `16` | ViT hidden `768`, layers `12`, heads `12`, MLP `3072` | BiT stage1/stage2 plus ViT indices `[8,11]` | reassemble `[1,1,1,0.5]`, neck `[256,512,768,768]` | resize 384, mean/std `[0.5]*3`. |
| `Intel/dpt-beit-base-384` | AutoBackbone BEiT | backbone image `384`, patch `16` | BEiT hidden `768`, layers `12`, heads `12`, MLP `3072` | BEiT out `[3,6,9,12]` | neck `[96,192,384,768]`; relative position bias inside backbone | resize 384, `ensure_multiple_of=32`. |
| `Intel/dpt-swinv2-tiny-256` | AutoBackbone SwinV2 | backbone image `256`, patch `4` | embed `96`, depths `[2,2,6,2]`, heads `[3,6,12,24]` | stages `[1,2,3,4]` | DPT reassemble disabled; hierarchical feature maps already NCHW | resize 256, resample bicubic enum 3. |
| `facebook/dpt-dinov2-small-kitti` | AutoBackbone DINOv2 | backbone image `518`, patch `14` | hidden `384`, layers `12`, heads `6`, MLP ratio `4` | `[3,6,9,12]` | neck `[48,96,192,384]`, `add_projection=true` before depth head | no resize/rescale, pad to divisor 14, ImageNet mean/std in 0-255 scale. |

## 3a. Family variation traps

- Native DPT and AutoBackbone DPT are structurally different. If `backbone_config` is present and `is_hybrid=false`, `DPTForDepthEstimation` uses `load_backbone(config)` directly instead of `DPTModel`; DPT's own ViT layers are not used.
- SwinV2 AutoBackbone is special-cased in `DPTNeck`: no reassemble stage, because feature maps are already image-like. A graph importer must not assume all DPT hidden states are `[B,seq,C]`.
- Hybrid mode uses BiT feature maps for the first two features and ViT token outputs for later features. It requires `readout_type="project"` and has `neck_ignore_stages=[0,1]`.
- Native ViT position embeddings are resized at runtime by bilinear interpolation of the 2D grid. The native patch embedding does not enforce input `height == image_size` in the current code path, so dynamic image sizes can work if the patch grid is compatible.
- `readout_type` changes token-to-feature math: `ignore` drops CLS, `add` adds CLS to each token channel, and `project` concatenates token+CLS then applies `Linear(2C -> C)+GELU`.
- `reassemble_factors` mix ConvTranspose2d upsampling, identity, and stride-2 Conv2d downsampling. They are not interchangeable with a pure resize unless weights are transformed or ignored.
- Segmentation `num_labels` may be implicit from `id2label`; `Intel/dpt-large-ade` has 150 labels even though `num_labels` is omitted in the JSON.
- Processor settings are checkpoint-significant. `facebook/dpt-dinov2-small-kitti` disables resize/rescale and pads to a multiple of 14 using 0-255 mean/std values; treating it like `Intel/dpt-large` changes input scale.
- NCHW is the semantic source layout throughout modeling code: Conv2d, ConvTranspose2d, BatchNorm2d, interpolation, and heads all use channel dimension 1. NHWC/channel-last optimization must be guarded to local conv/fusion regions and must rewrite axes.
- Axis-sensitive guards: token flatten/transpose uses `flatten(2).transpose(1,2)`, reassemble uses `[B,seq,C] -> [B,H,W,C] -> [B,C,H,W]`, segmentation argmax uses class channel `dim=0` per image or `dim=1` batched, depth squeeze removes channel dim 1.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input validation.
- Conv2d patch embedding with `kernel_size=stride=patch_size`, no padding for native DPT.
- Flatten spatial dims, transpose token/channel, concatenate CLS token, add positional embeddings.
- Sequence indexing and slicing: CLS token `hidden[:,0]`, patch tokens `hidden[:,1:]`.
- Reshape `[B,N,C] -> [B,H,W,C]`, permute to NCHW, contiguous.
- Feature list selection by backbone layer/stage indices.
- List reversal for top-down fusion.
- Optional `squeeze(dim=1)` for depth.

Neural network primitives:

- Linear projections: Q/K/V, attention output, MLP, readout projection, pooler.
- LayerNorm over hidden dimension.
- GELU, ReLU, Tanh pooler.
- Dropout can be removed or no-op for inference.
- Conv2d 1x1 and 3x3, ConvTranspose2d, BatchNorm2d in segmentation/fusion variants.
- Bilinear and bicubic interpolation with exact `align_corners` contracts.
- CrossEntropyLoss and auxiliary loss are training-only.

Attention primitives:

- Noncausal encoder self-attention.
- Standard MHA only for native DPT: `num_key_value_heads` is not used.
- Q/K/V Linear(`hidden_size -> hidden_size`) with optional bias `qkv_bias`.
- Attention tensors `[B,heads,seq,head_dim]`; eager math is `softmax((Q @ K^T) * head_dim**-0.5) @ V`.
- SDPA/Flash/Flex attention backend hooks exist through `ALL_ATTENTION_FUNCTIONS`; no mask is passed by DPT.

Backbone-dependent ops:

- BEiT: relative position bias, layer scale, ViT-like blocks, optional BEiT-specific backbone feature output.
- SwinV2: patch embedding, window attention, shifted/window hierarchy, patch merging; outputs image-like feature maps.
- DINOv2: ViT-like blocks with optional layer scale, patch 14, feature extraction at stages.
- BiT hybrid: ResNet-like convolutional bottleneck stages with group norm and dynamic padding.

Preprocessing-coupled ops:

- Resize output size calculation with `keep_aspect_ratio` and `ensure_multiple_of`.
- Rescale and normalize.
- Optional center pad to `size_divisor`.
- Segmentation label reduction: 0 becomes 255, then labels decrement by 1, 254 is restored to 255.

Postprocessing ops:

- Depth: optional bicubic resize from `[H,W]` to target size after unsqueeze to `[1,1,H,W]`.
- Segmentation: optional bilinear resize logits `[1,C,H,W]`, then argmax across classes.
- No NMS, boxes, masks, or generation cache.

## 5. Layer/block breakdown

Native DPT embedding:

```text
pixel_values: [B,3,H,W] NCHW
patch = Conv2d(3 -> hidden, kernel=patch, stride=patch)(pixel_values)  # [B,C,H/P,W/P]
tokens = flatten_spatial(patch).transpose(1,2)                         # [B,N,C]
tokens = concat(cls_token.expand(B,1,C), tokens)                       # [B,N+1,C]
pos = resize_position_embedding([1,N0+1,C], H/P, W/P)
hidden = dropout(tokens + pos)
```

Native ViT block, repeated `num_hidden_layers`:

```text
x_norm = LayerNorm(x)
q,k,v = Linear(C -> C, bias=qkv_bias)(x_norm), split into [B,heads,S,head_dim]
attn = noncausal_attention(q,k,v, mask=None, scale=head_dim**-0.5)
x = x + Linear(C -> C)(attn)
y = LayerNorm(x)
y = GELU(Linear(C -> intermediate)(y))
x = x + Linear(intermediate -> C)(y)
```

DPT reassemble for each selected native/BEiT/DINOv2 token feature:

```text
cls = feature[:,0]
patch_tokens = feature[:,1:]
grid = reshape patch_tokens to [B,patch_h,patch_w,C] then permute to [B,C,patch_h,patch_w]
if readout_type == "project":
    flat = grid.flatten(2).permute(0,2,1)
    flat = GELU(Linear(2C -> C)(concat(flat, cls expanded over tokens)))
    grid = flat.permute(0,2,1).reshape(original_nchw_shape)
elif readout_type == "add":
    grid = grid.flatten(2) + cls.unsqueeze(-1), then reshape
grid = Conv2d(C -> neck_hidden_size[i], kernel=1)(grid)
grid = ConvTranspose2d(..., kernel=stride=factor) if factor > 1
     | Identity if factor == 1
     | Conv2d(..., kernel=3, stride=int(1/factor), padding=1) if factor < 1
```

For `Intel/dpt-large`, selected hidden states are `[B,577,1024]` from layers 5, 11, 17, 23. Reassemble produces approximately:

```text
[B,256,96,96], [B,512,48,48], [B,1024,24,24], [B,1024,12,12]
```

DPT neck and fusion:

```text
features[i] = Conv2d(neck_hidden_sizes[i] -> fusion_hidden_size, kernel=3, padding=1, bias=False)
fused = reverse(features)
for feature in fused:
    if previous exists:
        residual = interpolate(feature, size=previous.hw, mode=bilinear, align_corners=False) if needed
        previous = previous + PreActResidual(residual)
    previous = PreActResidual(previous)
    previous = interpolate(previous, scale_factor=2, mode=bilinear, align_corners=True)
    previous = Conv2d(fusion_hidden_size -> fusion_hidden_size, kernel=1, bias=True)(previous)
```

Depth head:

```text
x = fused_features[head_in_index]
if add_projection:
    x = ReLU(Conv2d(256 -> 256, kernel=3, padding=1)(x))
x = Conv2d(F -> F/2, kernel=3, padding=1)(x)
x = Upsample(scale_factor=2, mode=bilinear, align_corners=True)(x)
x = Conv2d(F/2 -> 32, kernel=3, padding=1)(x)
x = ReLU(x)
x = Conv2d(32 -> 1, kernel=1)(x)
depth = ReLU(x).squeeze(dim=1)
```

Segmentation head:

```text
x = fused_features[head_in_index]
x = Conv2d(F -> F, kernel=3, padding=1, bias=False)(x)
x = BatchNorm2d(F)(x)
x = ReLU(x)
x = Dropout(p=semantic_classifier_dropout)(x)  # inference no-op
x = Conv2d(F -> num_labels, kernel=1)(x)
logits = Upsample(scale_factor=2, mode=bilinear, align_corners=True)(x)
```

## 6. Attention requirements

Native DPT attention is encoder-only, noncausal self-attention. There is no KV cache, generation path, attention mask, cross-attention, RoPE, ALiBi, sliding-window attention, or packed varlen sequence metadata in the native DPT blocks.

| Field | Native DPT |
|---|---|
| Type | Noncausal self-attention. |
| Heads | MHA, `head_dim = hidden_size / num_attention_heads`. |
| Q/K/V | Separate Linear layers, bias controlled by `qkv_bias`. |
| Mask | Always `None` in DPT source. |
| Scaling | `head_dim ** -0.5` passed to backend. |
| Dropout | `0.0` in eval, `attention_probs_dropout_prob` in training. |
| Backend | `ALL_ATTENTION_FUNCTIONS` dispatch; eager fallback is matmul, add mask if any, softmax, dropout, matmul. |
| Cache | Not applicable. |

For AutoBackbone variants, attention requirements belong to the selected backbone family:

- BEiT requires relative position bias and layer-scale behavior from the BEiT implementation.
- SwinV2 requires windowed/shifted attention and hierarchical feature maps.
- DINOv2 requires its ViT-style attention and layer-scale behavior.

DinoML should route these as backbone-specific audits rather than pretending they are native DPT attention.

## 7. Position encoding and custom math

Native DPT and hybrid ViT position embeddings are absolute learned position embeddings with runtime 2D bilinear resize:

```python
def resize_dpt_pos_embed(posemb, grid_h, grid_w, start_index=1):
    posemb_tok = posemb[:, :start_index]
    posemb_grid = posemb[0, start_index:]
    old = int(len(posemb_grid) ** 0.5)
    grid = posemb_grid.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    grid = interpolate(grid, size=(grid_h, grid_w), mode="bilinear")
    grid = grid.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
    return cat([posemb_tok, grid], dim=1)
```

Reassemble readout projection is model-specific and runtime-shape-sensitive:

```python
def project_readout(grid_nchw, cls, project):
    shape = grid_nchw.shape
    flat = grid_nchw.flatten(2).permute(0, 2, 1)
    readout = cls.unsqueeze(1).expand_as(flat)
    flat = project(cat([flat, readout], dim=-1))
    return flat.permute(0, 2, 1).reshape(shape)
```

What can be precomputed:

- Base position embedding weights and CLS token are constants.
- If input grid size is fixed per compiled artifact, resized position embeddings can be precomputed. If supporting dynamic `H/W`, the resize depends on runtime patch grid.
- Reassemble readout projection weights are constants, but CLS expansion and concat depend on batch and token count.

## 8. Preprocessing and input packing

Processor runtime tensor contract:

- Input images are converted to `pixel_values` in NCHW with shape `[B,3,H,W]`.
- Default processor uses bicubic resize, rescale by `1/255`, and ImageNet standard mean/std unless checkpoint JSON overrides them.
- `get_resize_output_image_size` computes scale to target height/width; if `keep_aspect_ratio=true`, it chooses the scale with least change and rounds each output dimension to `ensure_multiple_of`.
- Optional center padding pads height and width to a multiple of `size_divisor`.
- Segmentation maps are preprocessed with no normalize/rescale, squeezed to `[H,W]`, and converted to int64 labels.

Checkpoint-specific processor contracts:

- `Intel/dpt-large`: resize to 384, normalize with mean/std `[0.5,0.5,0.5]`.
- `Intel/dpt-large-ade`: resize to 480 even though model config has `image_size=384`; position embedding resize makes this viable.
- `Intel/dpt-beit-base-384`: resize 384, `ensure_multiple_of=32`.
- `Intel/dpt-swinv2-tiny-256`: resize 256, checkpoint preprocessor uses resample enum 3.
- `facebook/dpt-dinov2-small-kitti`: `do_resize=false`, `do_rescale=false`, `do_pad=true`, `size_divisor=14`, mean/std are 0-255 ImageNet values. DinoML should not insert a default 1/255 path for this checkpoint.

Postprocess:

- Depth output is raw `predicted_depth: [B,H,W]`. `post_process_depth_estimation` optionally bicubic-resizes each image to a caller-provided target `(height,width)` with `align_corners=False`, then returns a list of dictionaries.
- Segmentation output is raw logits `[B,num_labels,H,W]`. `post_process_semantic_segmentation` optionally bilinear-resizes each image's logits to target size with `align_corners=False`, then argmaxes over class channel. There is no NMS or mask thresholding.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Native patch Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(Cin -> hidden, kernel=patch_size, stride=patch_size, padding=0)
-> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, kh=kw=stride=patch_size) -> GEMM(weight_flat.T) -> BiasAdd -> token reshape
```

Preconditions:

- Native DPT patch embedding only, not BiT/SwinV2/backbone patch embeddings unless separately audited.
- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Runtime `H` and `W` divisible by patch size.
- Flatten order must match PyTorch NCHW convolution and subsequent `flatten(2).transpose(1,2)`.

Weight transform:

```python
w = conv.weight.reshape(hidden, in_channels * patch_h * patch_w)
```

Failure cases:

- Dynamic padding, non-divisible image sizes, AutoBackbone variants, or channel-last input without a guarded axis/layout rewrite.

Parity test sketch:

- Random NCHW images across fixed and bucketed `H/W`, compare Conv2d path to window-flatten GEMM before position embedding.

### Rewrite: Readout project as batched GEMM

Source pattern:

```text
grid [B,C,H,W] -> [B,H*W,C], concat expanded CLS -> Linear(2C -> C) -> GELU
```

Replacement:

```text
ConcatTokenReadout -> GEMM(2C,C) -> GELU -> reshape/permute
```

Preconditions:

- `readout_type == "project"`.
- Feature is token-originated `[B,seq,C]`; hybrid ignored stages 0 and 1 must bypass this rewrite.
- CLS token corresponds to the same hidden-state layer as the patch tokens.

Layout constraints:

- Source math is token-last `[B,N,C]` for Linear. If a channel-last layout pass keeps features NHWC, it must still feed Linear over the last dimension and restore the expected consumer layout.

### Rewrite: Reassemble 1x1 projection -> GEMM

Source pattern:

```text
Conv2d(hidden_size -> neck_channels, kernel=1) over NCHW feature map
```

Replacement:

```text
Flatten spatial to [B*H*W,C] -> GEMM -> reshape to NCHW
```

Preconditions:

- `kernel_size=1`, stride 1, no groups.
- Useful for token-originated features where a GEMM schedule is better than a small conv.

Failure cases:

- SwinV2 path where reassemble is disabled; fusion projection convs still exist but are 3x3.

### Rewrite: Fixed-size position embedding resize precompute

Source pattern:

```text
interpolate learned position grid to runtime patch grid
```

Replacement:

```text
artifact constant resized_pos_embed for known H/P,W/P
```

Preconditions:

- Compile target fixes input `H/W` or finite buckets.
- Interpolation mode and rounding match PyTorch bilinear default behavior.

Failure cases:

- Open dynamic image sizes, multiple processor resize policies, or runtime target sizes not known at compile time.

### Rewrite: Local NCHW conv/fusion region to channel-last kernels

Source pattern:

```text
NCHW Conv2d/BatchNorm/ReLU/interpolate/Conv2d in DPT neck/head
```

Replacement:

```text
guarded NCHW->NHWC boundary -> channel-last conv/fusion kernels -> NHWC->NCHW boundary before source-axis consumers
```

Preconditions:

- Region begins after token reassemble has produced image-like feature maps.
- All consumers inside the region are layout-rewritten together.
- Axis rewrites are explicit: channel dim `1 -> -1`, spatial dims `2/3 -> 1/2`, argmax/squeeze/postprocess remain protected unless rewritten.

Failure cases:

- Crossing into token `[B,S,C]` attention/MLP without a separate representation change.
- SwinV2/AutoBackbone feature maps whose layout contract comes from another family.
- Postprocess/logit consumers expecting NCHW class channel.

Recommended conceptual guards:

- `no_layout_translation()` around processor input semantics, token sequence blocks, `post_process_*`, and segmentation channel argmax unless the whole region is proven and rewritten.

## 10. Kernel fusion candidates

Highest priority:

- Native ViT block: LayerNorm + Q/K/V projections, attention, output projection, residual, MLP. This dominates native DPT-large.
- Conv/interpolate fusion in DPT neck. The top-down fusion repeatedly applies residual conv units, bilinear resize, and 1x1 projection on dense feature maps.
- Patch Conv2d to GEMM for native DPT when image size is fixed or bucketed.
- Depth/segmentation head conv chains, especially for fixed-resolution dense output.

Medium priority:

- Reassemble readout projection fusion: concat CLS + Linear + GELU + reshape/permute.
- Precompute or bucket-cache resized position embeddings.
- BatchNorm folding into Conv2d for inference segmentation/fusion heads.
- Processor resize/normalize throughput, if DinoML owns preprocessing.

Lower priority:

- Pooler support for `DPTModel` feature-extraction. It is not used by depth/segmentation heads.
- Training-only auxiliary head and losses.
- Attention backend variants beyond native noncausal encoder attention, unless selected AutoBackbone requires them.

## 11. Runtime staging plan

Stage 1: Parse config and processor JSON for native DPT, load weights, and validate patch embedding plus one ViT block on fixed 384 input.

Stage 2: Implement native encoder parity for `Intel/dpt-large` through selected hidden states. Stub postprocess initially and compare hidden states/features to PyTorch.

Stage 3: Implement DPT reassemble, neck, and depth head. Validate `Intel/dpt-large` end-to-end raw `predicted_depth` at 384.

Stage 4: Add processor and postprocess parity for depth, including target-size bicubic resize and dynamic/bucketed input sizes.

Stage 5: Add segmentation head for `Intel/dpt-large-ade`, including `num_labels` from `id2label`, BatchNorm folding, bilinear target resize, and argmax.

Stage 6: Add hybrid BiT path or route to a backbone abstraction. Validate `Intel/dpt-hybrid-midas` with mixed feature-map/token reassemble.

Stage 7: Add AutoBackbone DPT variants by composing already-audited BEiT, DINOv2, and SwinV2 backbones with DPT neck/head. SwinV2 must exercise the no-reassemble path.

Stage 8: Enable graph rewrites/fusions: patch Conv2d->GEMM, readout GEMM, channel-last conv neck/head, position embedding precompute for fixed buckets.

Can be stubbed initially:

- Pooler output for `DPTModel` if the first target is depth/segmentation.
- Training losses, auxiliary segmentation loss, gradient checkpointing, attention probabilities.
- AutoBackbone variants until the native DPT path is solid.

## 12. Parity and validation plan

- Processor parity: resize output size, `ensure_multiple_of`, `keep_aspect_ratio`, normalize/rescale, label reduction, and `size_divisor` padding. Include the DINOv2 KITTI no-rescale/no-resize processor.
- Custom op tests: position embedding resize, readout projection, reassemble factors `4`, `2`, `1`, and `0.5`.
- Single-block parity: native ViT block with random `[B,S,C]`, fp32 tolerance around `1e-5` absolute/relative.
- Native encoder parity: after selected layers 5/11/17/23 for `Intel/dpt-large`.
- Neck/head parity: synthetic feature list and real encoder features; compare fused feature shapes and values.
- Depth parity: raw `predicted_depth` for `Intel/dpt-large` and postprocessed resize to original image size.
- Segmentation parity: raw logits for `Intel/dpt-large-ade`, postprocess target resize, and argmax map.
- Hybrid parity: `Intel/dpt-hybrid-midas` mixed BiT feature maps plus ViT outputs.
- AutoBackbone parity: one checkpoint each for BEiT, SwinV2, and DINOv2 after those backbones are supported.
- Tolerances: fp32 `1e-4` for full model due interpolation/attention ordering; fp16/bf16 should use looser `1e-2` style tolerances and compare postprocess maps separately.

## 13. Performance probes

- Processor throughput: images/sec for resize/normalize/pad, split by checkpoint processor policy.
- Native encoder throughput: batch-size sweep for 384 and 480 images, report sequence length and attention backend.
- DPT neck/head throughput: isolate four feature maps into fusion and dense heads.
- End-to-end depth images/sec for `Intel/dpt-large`.
- End-to-end segmentation images/sec for `Intel/dpt-large-ade`, including argmax/postprocess.
- Resolution sweep: 256, 384, 480, 518/padded-to-14, and bucketed dynamic sizes.
- Layout probe: NCHW conv path versus guarded channel-last neck/head path.
- Rewrite probe: patch Conv2d native versus WindowFlatten+GEMM.
- AutoBackbone composition probes: BEiT, SwinV2, DINOv2 backbone-only versus DPT neck/head-only.

No benchmark observations are included here; these are source-derived recommended probes.

## 14. Skip/defer list

- Training, labels/losses, auxiliary segmentation loss.
- Gradient checkpointing and dropout behavior beyond inference no-op.
- Pooler output unless using `DPTModel` for feature extraction.
- Returning attentions/hidden states beyond what is needed for DPT neck selection.
- AutoBackbone internals until their owning family audits/operators are ready.
- Remote-code checkpoints, quantization, ONNX/export-specific paths.
- Unbounded NHWC translation across processor, token, and postprocess regions.

## 15. Final implementation checklist

- [ ] Parse `DPTConfig`, including implicit defaults and `id2label`-derived `num_labels`.
- [ ] Parse DPT processor JSON and preserve checkpoint-specific resize/rescale/normalize/pad settings.
- [ ] Load native DPT weights with CLS token and position embeddings.
- [ ] Implement native patch embedding Conv2d and position embedding resize.
- [ ] Implement native ViT encoder block: LayerNorm, QKV, noncausal attention, MLP, residuals.
- [ ] Capture/select intermediate hidden states by `backbone_out_indices`.
- [ ] Implement reassemble readout modes `ignore`, `add`, and `project`.
- [ ] Implement reassemble Conv2d/ConvTranspose2d/downsample Conv2d factors.
- [ ] Implement DPT neck projection convs and feature fusion residual blocks.
- [ ] Implement depth head and depth postprocess resize.
- [ ] Implement segmentation head, BatchNorm inference folding, logits resize, and argmax.
- [ ] Add native `Intel/dpt-large` depth parity tests.
- [ ] Add `Intel/dpt-large-ade` segmentation parity tests.
- [ ] Add hybrid BiT+DPT parity route or explicitly defer with config rejection.
- [ ] Add AutoBackbone composition path for BEiT, SwinV2, and DINOv2 or reject those configs clearly.
- [ ] Add guarded patch Conv2d->GEMM rewrite.
- [ ] Add guarded readout projection and 1x1 reassemble GEMM rewrites.
- [ ] Add guarded channel-last neck/head layout pass with no-layout guards around token and postprocess axes.
- [ ] Benchmark processor, encoder, neck/head, and end-to-end dense prediction throughput.
