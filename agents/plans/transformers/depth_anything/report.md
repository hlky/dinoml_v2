# Depth Anything Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: depth_anything family; representative checkpoints listed below
Config source: official Hugging Face config/preprocessor JSON from each model repo, with repo SHA captured in _sources
Source files inspected:
- X:/H/transformers/src/transformers/models/depth_anything/configuration_depth_anything.py
- X:/H/transformers/src/transformers/models/depth_anything/modeling_depth_anything.py
- X:/H/transformers/src/transformers/models/dpt/image_processing_dpt.py
- X:/H/transformers/src/transformers/models/dinov2/configuration_dinov2.py
- X:/H/transformers/src/transformers/models/dinov2/modeling_dinov2.py
- X:/H/transformers/src/transformers/backbone_utils.py
Any missing files or assumptions: no native image processor lives under depth_anything; official configs use DPTImageProcessor. No remote-code or gated official repos were required. This report composes the DINOv2 backbone audit instead of re-owning all DINOv2 operator coverage.
```

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/depth_anything/modeling_depth_anything.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/depth_anything/configuration_depth_anything.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dpt/image_processing_dpt.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/dinov2/modeling_dinov2.py

Local snapshots saved under `agents/plans/transformers/depth_anything/_sources/`, including source snapshots and official HF `config.json`, `preprocessor_config.json`, and `repo_info.json` for:

- `LiheYoung/depth-anything-small-hf`
- `LiheYoung/depth-anything-base-hf`
- `LiheYoung/depth-anything-large-hf`
- `depth-anything/Depth-Anything-V2-Small-hf`
- `depth-anything/Depth-Anything-V2-Base-hf`
- `depth-anything/Depth-Anything-V2-Large-hf`
- `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`
- `hf-internal-testing/tiny-random-DepthAnythingForDepthEstimation`

No checked repo returned 401/403 or `gated=true` through the HF API. Clickable model URLs above resolve access to all inspected configs.

## 2. High-level architecture

Depth Anything is monocular depth inference, not a generation model:

```text
DPT image preprocessing -> NCHW pixel_values
  -> DINOv2 AutoBackbone token feature maps
  -> DepthAnything reassemble neck
  -> top-down feature fusion pyramid
  -> convolutional depth head
  -> predicted_depth [B,H,W]
  -> optional DPTImageProcessor.post_process_depth_estimation resize
```

Stage decomposition:

| Stage | Runtime contract | Independently testable? |
|---|---|---|
| CPU/data pipeline | RGB conversion, resize to config `size`, aspect-ratio/multiple rounding, rescale, normalize, optional pad. Emits `pixel_values` as NCHW float tensor. | Yes, processor-only parity. |
| DINOv2 backbone | `AutoBackbone.from_config(backbone_config)` with `reshape_hidden_states=False`. Emits selected token sequences, not image maps: each feature is `[B, 1 + patch_h * patch_w, backbone_hidden]`. | Yes, use DINOv2 audit/operator parity. |
| Reassemble neck | Drops CLS, reshapes sequence to `[B, patch_h, patch_w, C]`, permutes to NCHW, 1x1 projects, then ConvTranspose2d/Identity/stride-2 Conv2d resizes four scales. | Yes, synthetic token features. |
| Fusion neck | 3x3 channel projection to `fusion_hidden_size`, reverse top-down fusion with residual conv units and bilinear interpolation. | Yes, synthetic NCHW feature maps. |
| Depth head | Conv3x3 -> bilinear upsample to preprocessed image size -> Conv3x3/ReLU -> Conv1x1 -> ReLU or Sigmoid scaled by `max_depth` -> squeeze channel. | Yes, head-only tests. |
| Postprocess | Per-image optional bicubic resize of `[H,W]` prediction to `target_size`. | Yes, CPU/Torch postprocess parity. |

Primary DinoML target: end-to-end `DepthAnythingForDepthEstimation` inference for relative depth first, with metric head as a small activation/scaling variation. Training labels/loss are not implemented in source and are out of scope.

## 3. Important config dimensions

Source defaults from `DepthAnythingConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `patch_size` | 14 | outer model computes `patch_height = pixel_height // patch_size`; must match backbone patch size. |
| `backbone_config` | default DINOv2 small | loaded through `AutoBackbone`; default hidden 384, heads 6, out indices `[9,10,11,12]`, `reshape_hidden_states=False`. |
| `reassemble_hidden_size` | 384 | input channels for every reassemble 1x1 conv; must equal selected backbone hidden size. |
| `reassemble_factors` | `[4,2,1,0.5]` | chooses ConvTranspose2d, Identity, or stride-2 Conv2d per selected feature. |
| `neck_hidden_sizes` | `[48,96,192,384]` | output channels of four reassemble projections. |
| `fusion_hidden_size` | 64 | common channel count for fusion and head input. |
| `head_in_index` | -1 | uses last fused output, the final highest-resolution map. |
| `head_hidden_size` | 32 | middle depth-head channel count. |
| `depth_estimation_type` | `relative` | `relative` uses final ReLU; `metric` uses Sigmoid. |
| `max_depth` | 1 effective | source sets missing/falsey value to 1; metric configs set 20 or 80. |

Representative checkpoint sweep:

| Checkpoint | Repo SHA | Depth type | Backbone H/L/heads | Selected stages | Neck -> fusion | Head | Processor |
|---|---|---|---|---|---|---|---|
| `hf-internal-testing/tiny-random-DepthAnythingForDepthEstimation` | `f6d7335...` | config omits, effective `relative`, max 1 | DINOv2 hidden 4, layers 2, heads 2, image 32, patch omitted in JSON | `[1,2]` | neck `[2,2]`, fusion 6 | conv1 `6->3`, conv2 `3->32`, conv3 `32->1` | DPT processor 518, aspect/multiple 14. |
| `LiheYoung/depth-anything-small-hf` | `25216a9...` | config omits, effective `relative`, max 1 | DINOv2 small: hidden 384, 12 layers by source default, 6 heads, patch 14 | `[9,10,11,12]` | `[48,96,192,384] -> 64` | `64->32->32->1` | resize 518, keep aspect, ensure multiple 14, ImageNet mean/std. |
| `LiheYoung/depth-anything-base-hf` | `fbce2ff...` | relative | hidden 768, 12 layers default, 12 heads, patch 14 | `[9,10,11,12]` | `[96,192,384,768] -> 128` | `128->64->32->1` | same processor. |
| `LiheYoung/depth-anything-large-hf` | `27ccb09...` | relative | hidden 1024, 24 layers, 16 heads, patch 14 | `[21,22,23,24]` | `[256,512,1024,1024] -> 256` | `256->128->32->1` | same processor. |
| `depth-anything/Depth-Anything-V2-Small-hf` | `5426e4f...` | relative | hidden 384, 12 layers default, 6 heads, patch 14 | `[3,6,9,12]` | `[48,96,192,384] -> 64` | `64->32->32->1` | same processor. |
| `depth-anything/Depth-Anything-V2-Base-hf` | `b1958af...` | relative | hidden 768, 12 layers default, 12 heads, patch 14 | `[3,6,9,12]` | `[96,192,384,768] -> 128` | `128->64->32->1` | same processor. |
| `depth-anything/Depth-Anything-V2-Large-hf` | `7581137...` | relative | hidden 1024, 24 layers, 16 heads, patch 14 | `[5,12,18,24]` | `[256,512,1024,1024] -> 256` | `256->128->32->1` | same processor. |
| `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | `8078d68...` | metric, max 20 | hidden 384, 12 layers default, 6 heads, patch 14 | `[3,6,9,12]` | `[48,96,192,384] -> 64` | final Sigmoid * 20 | same processor. |
| `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` | `fd2c220...` | metric, max 80 | hidden 384, 12 layers default, 6 heads, patch 14 | `[3,6,9,12]` | `[48,96,192,384] -> 64` | final Sigmoid * 80 | same processor. |

Fields commonly omitted in checkpoint JSON but supplied by source defaults:

- Depth Anything configs often omit `depth_estimation_type` and `max_depth`; effective behavior is relative depth with final ReLU and scale 1.
- DINOv2 backbone configs often omit `num_hidden_layers`, `mlp_ratio`, `hidden_act`, `apply_layernorm`, `layer_norm_eps`, `qkv_bias`, and `use_swiglu_ffn`; effective defaults are 12 layers, MLP ratio 4, GELU, apply selected-feature LayerNorm, eps `1e-6`, biased Q/K/V, no SwiGLU.
- Tiny random config omits backbone `patch_size`; the outer Depth Anything config still has `patch_size=14` by default unless the loaded full config overrides it. Treat tiny random as shape/debug-only, not production guidance.

## 3a. Family variation traps

- The backbone is a nested `backbone_config`, not owned directly by `DepthAnythingConfig`. DinoML should route DINOv2 operators through the separately audited `dinov2` family and keep this report focused on the neck/head/postprocess contract.
- `reshape_hidden_states=False` is mandatory for this native source path. If a checkpoint or caller supplies DINOv2 feature maps already reshaped to NCHW, `DepthAnythingReassembleStage` will incorrectly slice `[:,1:]` and reshape a rank-4 tensor. Add a no-layout/no-contract-translation guard around the backbone output ABI.
- Selected stages differ between Depth Anything v1 and v2 checkpoints. V1 small/base/large use late adjacent stages; v2 uses spread intermediate stages. Do not hard-code `[9,10,11,12]`.
- The model computes `patch_height` and `patch_width` from `pixel_values.shape[-2:] // config.patch_size`, not from backbone output shape. Runtime parity requires the processor to produce dimensions divisible by 14 or an importer must guard floor division and sequence length consistency.
- Reassemble factor `0.5` is a learned stride-2 Conv2d (`kernel=3,stride=2,pad=1`), not interpolation. Factor `4` and `2` are ConvTranspose2d with `kernel=stride=factor`.
- Fusion interpolation has two different `align_corners` contracts: residual resize to match current hidden state uses bilinear `align_corners=False`; top-down upsampling uses bilinear `align_corners=True`.
- Depth head interpolation to the preprocessed image size uses bilinear `align_corners=True`; postprocess resize to original target size uses bicubic `align_corners=False`.
- Metric depth changes only final activation and scaling: `Sigmoid() * max_depth`. Relative depth uses `ReLU() * 1`.
- Source tensor layout is NCHW through all conv, transpose conv, and interpolation modules. NHWC/channel-last can be an optimization for local conv-heavy regions only, with axis rewrites and consumer guards.
- Axis-sensitive ops needing guards: `hidden_state[:,1:]`, reshape `[B,patch_h,patch_w,C]`, `permute(0,3,1,2)`, Conv2d channel axis 1, `interpolate(size=(H,W))`, `squeeze(dim=1)`, postprocess `unsqueeze(0).unsqueeze(1)`.
- The source `DepthAnythingForDepthEstimation._no_split_modules = ["DPTViTEmbeddings"]` looks inherited/stale for this DINOv2-backed implementation; it does not create a runtime op requirement.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `pixel_values` input `[B,3,H,W]`.
- Backbone token feature ABI: selected feature maps are a tuple/list of `[B,1+patch_h*patch_w,Cb]`.
- Sequence slice `hidden[:,1:]`, reshape to `[B,patch_h,patch_w,Cb]`, permute to `[B,Cb,patch_h,patch_w]`, contiguous materialization.
- Reverse feature list for fusion, list indexing by `head_in_index`.
- Elementwise add for residual and fusion merges.
- `squeeze(dim=1)` from `[B,1,H,W]` to `[B,H,W]`.
- Processor/postprocess unsqueeze/squeeze for `[H,W] <-> [1,1,H,W]`.

Neural network primitives:

- DINOv2 backbone: Conv2d patch embedding, LayerNorm, noncausal MHA, Linear/GEMM, GELU, residual adds, LayerScale; see `dinov2` report.
- Reassemble 1x1 Conv2d with bias: `backbone_hidden -> neck_hidden[i]`.
- Reassemble resize:
  - ConvTranspose2d `neck_hidden[i] -> neck_hidden[i]`, `kernel=stride=4` or `2`, bias true.
  - Identity for factor 1.
  - Conv2d `neck_hidden[i] -> neck_hidden[i]`, `kernel=3,stride=2,padding=1`, bias true for factor 0.5.
- Neck projection Conv2d without bias: `neck_hidden[i] -> fusion_hidden_size`, `kernel=3,padding=1`.
- Fusion residual units: ReLU -> Conv2d `F->F`, `kernel=3,pad=1,bias=True` -> ReLU -> Conv2d `F->F`, repeated two residual units per fusion layer.
- Fusion projection Conv2d with bias: `F->F`, `kernel=1`.
- Depth head: Conv2d `F -> F/2`, Conv2d `F/2 -> head_hidden_size`, ReLU, Conv2d `head_hidden_size -> 1`, ReLU or Sigmoid, scalar multiply by `max_depth`.
- Bilinear interpolation and bicubic interpolation with exact `align_corners`.

Attention primitives:

- Required only inside the DINOv2 encoder backbone.
- Noncausal self-attention, standard MHA, no KV cache, no generation cache, no cross-attention.
- Q/K/V are independent `Linear(H -> H, bias=qkv_bias)`; head dim is `H / num_attention_heads` and is 64 for official small/base/large.

Preprocessing-coupled ops:

- DPTImageProcessor resize output-size calculation with `keep_aspect_ratio=True`, `ensure_multiple_of=14`.
- Bicubic image resize, rescale by `1/255`, ImageNet normalize, optional pad if enabled by a future config.
- Output `pixel_values` in channel-first format.

Postprocessing ops:

- Optional per-image bicubic resize of predicted depth to requested `(height,width)`.
- No NMS, boxes, masks, logits, beam search, or text generation.

## 5. Layer/block breakdown

Backbone feature contract:

```text
pixel_values: [B,3,H,W] NCHW
patch_h = H // 14
patch_w = W // 14
backbone(pixel_values).feature_maps[i]: [B, 1 + patch_h * patch_w, Cb]
```

Reassemble layer `i`:

```text
x = feature[i][:, 1:]                                   # drop CLS
x = reshape(x, [B, patch_h, patch_w, Cb])
x = permute(x, [0, 3, 1, 2]).contiguous()               # [B,Cb,patch_h,patch_w]
x = Conv2d(Cb -> neck_i, kernel=1, bias=True)(x)
if factor in {2,4}: x = ConvTranspose2d(neck_i -> neck_i, kernel=factor, stride=factor)(x)
elif factor == 1: x = x
elif factor == 0.5: x = Conv2d(neck_i -> neck_i, kernel=3, stride=2, padding=1)(x)
```

For 518x518 input (`patch_h=patch_w=37`) the reassemble spatial sizes are roughly `148`, `74`, `37`, and `19`.

Neck projection and fusion:

```text
features[i] = Conv2d(neck_i -> F, kernel=3, padding=1, bias=False)(reassembled[i])
features = reverse(features)
fused = None
for each hidden_state in reversed features:
    if fused is None:
        y = residual_unit2(hidden_state)
    else:
        if fused.shape != hidden_state.shape:
            hidden_state = interpolate(hidden_state, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        y = fused + residual_unit1(hidden_state)
        y = residual_unit2(y)
    y = interpolate(y, size=next_feature_size or scale_factor=2, mode="bilinear", align_corners=True)
    fused = Conv2d(F -> F, kernel=1, bias=True)(y)
```

Each residual unit:

```text
residual = x
x = ReLU(x)
x = Conv2d(F -> F, kernel=3, padding=1, bias=True)(x)
x = ReLU(x)
x = Conv2d(F -> F, kernel=3, padding=1, bias=True)(x)
return x + residual
```

Depth head:

```text
x = fused_outputs[head_in_index]                        # default final fused output
x = Conv2d(F -> F/2, kernel=3, padding=1, bias=True)(x)
x = interpolate(x, size=(patch_h * patch_size, patch_w * patch_size), mode="bilinear", align_corners=True)
x = Conv2d(F/2 -> head_hidden_size, kernel=3, padding=1, bias=True)(x)
x = ReLU(x)
x = Conv2d(head_hidden_size -> 1, kernel=1, bias=True)(x)
x = ReLU(x) * 1                 # relative depth
# or Sigmoid(x) * max_depth      # metric depth
predicted_depth = squeeze(x, dim=1)                     # [B,H_preprocessed,W_preprocessed]
```

## 6. Attention requirements

Depth Anything adds no attention outside the backbone.

Backbone attention summary for official configs:

| Property | Contract |
|---|---|
| Type | Encoder noncausal self-attention. |
| Heads | small 6, base 12, large 16. |
| Head dim | 64 for official configs. |
| Q/K/V | separate biased `Linear(H -> H)`. |
| Mask | none in normal depth inference. |
| Position | DINOv2 learned absolute positional embedding with bicubic interpolation when needed. |
| KV cache | not applicable. |
| SDPA/Flash | DINOv2 supports attention backend dispatch in Transformers; DinoML may first validate eager math and later route to optimized noncausal attention. |

No causal masks, packed/varlen sequence metadata, sliding-window attention, RoPE, ALiBi, cross-attention, or generation cache are required for this primary target.

## 7. Position encoding and custom math

Depth Anything itself has no custom position encoding. DINOv2 provides learned absolute positions:

```python
def dinov2_pos_for_image(position_embeddings, height, width, patch_size):
    cls_pos = position_embeddings[:, :1]
    patch_pos = position_embeddings[:, 1:]
    dim = patch_pos.shape[-1]
    n = int((patch_pos.shape[1]) ** 0.5)
    patch_pos = patch_pos.reshape(1, n, n, dim).permute(0, 3, 1, 2)
    patch_pos = interpolate(
        patch_pos.float(),
        size=(height // patch_size, width // patch_size),
        mode="bicubic",
        align_corners=False,
    ).to(position_embeddings.dtype)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return concat([cls_pos, patch_pos], dim=1)
```

What can be precomputed:

- For fixed preprocessed sizes such as 518x518, cache interpolated DINOv2 position tables per `(height,width,dtype)`.
- Reassemble/fusion target sizes are deterministic from `patch_h`, `patch_w`, and feature list sizes.

What stays dynamic:

- Processor `keep_aspect_ratio=True` can produce non-square multiples of 14.
- DINOv2 interpolation and Depth Anything head target size depend on runtime `pixel_values.shape[-2:]`.

## 8. Preprocessing and input packing

Official inspected configs use `DPTImageProcessor`:

| Field | Value |
|---|---|
| `image_processor_type` | `DPTImageProcessor` |
| `size` | `{"height": 518, "width": 518}` |
| `do_resize` | true |
| `resample` | bicubic through processor default/config |
| `keep_aspect_ratio` | true |
| `ensure_multiple_of` | 14 |
| `do_rescale` / factor | true / `1/255` |
| `do_normalize` | true |
| `image_mean` | `[0.485, 0.456, 0.406]` |
| `image_std` | `[0.229, 0.224, 0.225]` |
| `do_pad` | false in inspected official configs |
| Model input | `pixel_values`, NCHW `[B,3,H,W]` |

Resize math:

```text
scale_h = target_h / input_h
scale_w = target_w / input_w
if keep_aspect_ratio:
  use the scale closest to 1 for both axes
new_h = round(scale_h * input_h / ensure_multiple_of) * ensure_multiple_of
new_w = round(scale_w * input_w / ensure_multiple_of) * ensure_multiple_of
```

CPU/data-pipeline work:

- Image loading/RGB conversion.
- Resize, rescale, normalize, optional pad.
- Batch grouping by shape in the processor implementation.

GPU/runtime work:

- Model consumes only `pixel_values`.
- No placeholder tokens, token type IDs, masks, packed sequence descriptors, or `cu_seqlens` metadata.

Postprocessing:

```text
for each depth [H,W]:
  if target_size:
    depth = bicubic_interpolate(depth[None,None], size=target_size, align_corners=False).squeeze()
  return {"predicted_depth": depth}
```

End-to-end parity requires callers to pass original image sizes to `post_process_depth_estimation` when they want depth maps in original image resolution. The model output itself is in preprocessed image resolution.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap DINOv2 patch Conv2d -> Linear

Preconditions:

- Backbone is DINOv2 patch embedding.
- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input NCHW height/width are divisible by patch size or floor-conv semantics are explicitly preserved.
- The following flatten order remains PyTorch NCHW conv output flatten spatial then transpose to `[B,N,H]`.

Replacement:

```text
Extract non-overlap patches in row-major grid order
-> flatten each patch as [C,kh,kw]
-> MatMul(weight.reshape(out, C*kh*kw).T)
-> BiasAdd
-> token sequence [B,patch_h*patch_w,out]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Layout constraints/failure cases:

- If an NHWC image layout pass runs, it must rewrite extraction order and weight flattening or keep this region guarded as NCHW.
- Dynamic non-divisible images must either use Conv2d fallback or prove PyTorch floor output size and downstream `patch_h = H // P` match.

Parity test sketch: compare patch embedding tokens for random `[B,3,518,518]` and non-square multiples of 14 against PyTorch DINOv2.

### Rewrite: 1x1 Conv2d in reassemble/fusion projection -> channel Linear

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- NCHW semantic layout is preserved or NHWC pass rewrites channel axis and weight layout.

Replacement:

```text
NCHW -> optional local NHWC view/copy -> MatMul(Cin -> Cout) + bias -> restore expected consumer layout
```

Weight transform:

```python
w_linear = conv.weight[:, :, 0, 0]  # [Cout,Cin]
```

Failure cases:

- Do not apply across residual/fusion consumers unless all subsequent ops agree on optimized layout.
- For pure NCHW lowering, a small Conv2d kernel may be simpler and safer.

### Rewrite: depth/fusion Conv2d chains to cuDNN/channel-last region

Preconditions:

- Region is bounded from reassembled NCHW maps through Conv/ReLU/Add/Interpolate/Conv consumers.
- Every axis-sensitive op is rewritten: channel axis `1 -> -1`, spatial axes `2,3 -> 1,2`, `squeeze(dim=1) -> squeeze(dim=-1)` only if final depth channel is NHWC.
- Interpolation backend supports NHWC or explicit transposes are inserted at boundaries.

Replacement:

```text
NCHW feature maps -> guarded channel-last conv/fusion/head region -> NCHW or [B,H,W] output
```

Failure cases:

- Backbone token-to-map reshape is a no-layout-translation boundary unless the importer proves equivalent token order.
- DPT postprocess expects `[H,W]` per image, so final squeeze semantics must be preserved exactly.

Parity test sketch: run neck/head random-tensor parity with square 37x37 patch grids and non-square grids, relative and metric activations.

### Rewrite: final head interpolation target constant folding

Preconditions:

- Static or bucketed input image sizes.
- `patch_size` is static and equals backbone patch size.

Replacement:

```text
patch_h = H // patch_size
target_h = patch_h * patch_size
```

Fold target sizes into compile/profile buckets. For official processors the target is the preprocessed input shape because dimensions are multiples of 14.

Failure cases:

- If inputs are not multiples of patch size, target is floor-multiple and smaller than input.
- If outer `patch_size` diverges from backbone patch size, reject config.

## 10. Kernel fusion candidates

Highest priority:

- DINOv2 encoder kernels from the `dinov2` audit: LayerNorm, QKV projections, noncausal attention, MLP/GELU, residual/LayerScale.
- Conv2d/ConvTranspose2d/interpolate coverage for the neck/head. Depth Anything is convolution-heavy after the backbone, and DinoML currently has convolution/upsampling gaps in the v1 checklist.
- Bilinear/bicubic interpolation with exact `align_corners` and dtype behavior. This appears in backbone positions, fusion, head, and postprocess.

Medium priority:

- Conv+ReLU and residual-unit fusions in `DepthAnythingPreActResidualLayer`.
- 1x1 Conv2d as GEMM for reassemble projections and fusion projections.
- Channel-last guarded region for neck/head once Conv2d and interpolation layout contracts are stable.
- Processor throughput for large batches, especially resize/normalize with aspect-preserving multiple-of-14 rounding.

Lower priority:

- ConvTranspose2d special-case kernels for factors 2 and 4 if cuDNN coverage is not enough.
- Metric-depth Sigmoid/multiply fusion; it is small and only affects metric variants.
- Postprocess bicubic resize on GPU; CPU postprocess is acceptable for first graph parity if the runtime target is model-only depth.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `DepthAnythingConfig` and nested DINOv2 `backbone_config`.
- Reject or defer non-DINOv2/timm backbones for this family report unless separately audited.
- Validate `reshape_hidden_states=False`, matching `patch_size`, four neck features for production checkpoints, and `reassemble_hidden_size == backbone.hidden_size`.

Stage 2: compose DINOv2 backbone parity.

- Reuse DINOv2 encoder/backbone integration.
- Validate selected feature maps `[B,1+N,C]` for each configured stage.

Stage 3: neck-only parity.

- Implement reassemble, projection convs, feature fusion, interpolation contracts.
- Test synthetic selected hidden states before full backbone integration.

Stage 4: depth head parity.

- Implement relative head first.
- Add metric final Sigmoid and scale.

Stage 5: processor/postprocess parity.

- Confirm DPTImageProcessor config ingestion and end-to-end output resize behavior.
- Decide whether postprocess is CPU/data-pipeline or graph-owned for DinoML.

Stage 6: layout and kernel optimization.

- Add guarded NHWC/channel-last region only after NCHW semantic parity.
- Add Conv2d-to-GEMM and 1x1 Conv rewrites with explicit guards.

Initially stubbable:

- Training labels/loss, gradient checkpointing, hidden-state/attention returns, and postprocess GPU acceleration.

## 12. Parity and validation plan

Recommended tests:

- Config tests: parse all saved representative configs and verify effective defaults for omitted fields.
- Processor tests: compare `DPTImageProcessor` resize/rescale/normalize output shapes and values for square and non-square images.
- DINOv2 feature contract: run or fixture backbone selected features with `reshape_hidden_states=False`; assert list length, shapes, CLS presence, and selected stage names.
- Reassemble random tests: feed `[B,1+patch_h*patch_w,Cb]` features for `patch_h=patch_w=37` and non-square sizes; compare all four reassembled outputs.
- Fusion random tests: compare each fused output, especially shape mismatch branch using `align_corners=False` residual interpolation.
- Head random tests: relative and metric variants; verify final output `[B,patch_h*14,patch_w*14]` and channel squeeze.
- End-to-end image test: official small and one V2 metric small checkpoint, compare `predicted_depth` before and after postprocess.

Suggested tolerances:

- fp32 model graph: `rtol=1e-4`, `atol=1e-4`, tighten for isolated convs if possible.
- fp16/bf16 optimized graph: start `rtol=5e-3`, `atol=5e-3`, with separate interpolation tolerance checks.
- Postprocess bicubic resize can be backend-sensitive; test shape and value tolerance separately from model graph.

No DinoML tests were run for this audit, per user scope.

## 13. Performance probes

- Processor throughput: images/sec for resize to 518 with `keep_aspect_ratio=True` and multiple-of-14 rounding.
- Backbone throughput: DINOv2 selected feature extraction for small/base/large at 518 square and non-square multiples.
- Neck/head throughput: synthetic feature maps to isolate conv/interpolation cost from attention.
- Interpolation probes: bilinear `align_corners=True`, bilinear `align_corners=False`, bicubic `align_corners=False`, square and non-square grids.
- ConvTranspose2d probes: factor 2 and 4 reassemble layers by channel size.
- Layout probes: NCHW cuDNN vs guarded NHWC/channel-last for conv-heavy neck/head.
- Batch-size sweep: `B=1,2,4,8` at 518 square.
- Resolution sweep: 392, 518, 686 or other multiples of 14; include non-square aspect-preserved shapes.
- End-to-end requests/hour split into preprocessing, backbone, neck/head, postprocess.
- Memory probes: DINOv2 attention activation sizes at 37x37 patches (`S=1370`) and larger resolutions.

## 14. Skip/defer list

- Training and loss: source raises `NotImplementedError` when labels are supplied.
- Semantic segmentation and other DPT heads: not part of `depth_anything`.
- Non-DINOv2 or timm backbones: not present in inspected official Depth Anything configs.
- Hidden-state and attention outputs: useful for debugging but not required for primary depth inference output.
- Gated/private repo handling: no inspected official configs were gated; keep generic 401/403 report handling for future audits.
- GPU postprocess resize: model-only parity can return preprocessed-resolution depth first.
- Broad NHWC translation: defer until NCHW source parity and local guarded layout regions are proven.
- Quantization, multi-GPU tensor parallel, speculative decoding, KV cache, generation controllers: not applicable.

## 15. Final implementation checklist

- [ ] Parse `DepthAnythingConfig` and nested `backbone_config`.
- [ ] Validate `model_type="depth_anything"` and DINOv2 `backbone_config` admission.
- [ ] Validate `reshape_hidden_states=False`, matching patch sizes, selected feature count, and hidden/channel widths.
- [ ] Load DINOv2 backbone weights through the existing DINOv2 path.
- [ ] Implement selected DINOv2 backbone feature-map ABI `[B,1+N,C]`.
- [ ] Implement reassemble sequence slice/reshape/permute.
- [ ] Implement Conv2d 1x1/3x3, ConvTranspose2d factors 2/4, and stride-2 Conv2d factor 0.5.
- [ ] Implement bilinear interpolation with both `align_corners=True` and `False`.
- [ ] Implement feature fusion residual units and top-down fusion ordering.
- [ ] Implement depth head relative ReLU output.
- [ ] Implement metric Sigmoid plus `max_depth` scaling.
- [ ] Implement final `squeeze(dim=1)` and output shape reporting `[B,H,W]`.
- [ ] Implement or integrate `DPTImageProcessor` config handling for preprocessing.
- [ ] Implement postprocess bicubic resize or document CPU/data-pipeline ownership.
- [ ] Add parity tests for reassemble, fusion, head, and end-to-end small checkpoint.
- [ ] Add guarded Conv2d-to-GEMM and 1x1 Conv rewrites after source parity.
- [ ] Add NHWC/channel-last optimization only inside guarded conv/interpolation regions.
- [ ] Benchmark preprocessing, backbone, neck/head, interpolation, and end-to-end throughput.
