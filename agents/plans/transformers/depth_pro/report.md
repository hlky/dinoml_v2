# DepthPro DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: apple/DepthPro-hf
Config source: https://huggingface.co/apple/DepthPro-hf/raw/main/config.json at repo sha de816c8ce7168afcb231f96d501d72b869d0beda
Source files inspected:
- X:/H/transformers/src/transformers/models/depth_pro/modeling_depth_pro.py
- X:/H/transformers/src/transformers/models/depth_pro/configuration_depth_pro.py
- X:/H/transformers/src/transformers/models/depth_pro/image_processing_depth_pro.py
- X:/H/transformers/src/transformers/models/dinov2/modeling_dinov2.py, because DepthPro composes three AutoModel DINOv2 encoders by default/checkpoint config
- X:/H/transformers/src/transformers/models/dinov2/configuration_dinov2.py
Any missing files or assumptions:
- No native `processing_depth_pro.py` or tokenizer files are relevant.
- `image_processor_config.json` is absent for `apple/DepthPro-hf`; `preprocessor_config.json` is present.
- Only one native public Transformers checkpoint was found for `model_type="depth_pro"`. ONNX/CoreML/4-bit/community repos were not treated as separate native source variants.
- No gated/401/403 gap was observed for the official `apple/DepthPro-hf` config or preprocessor.
```

Snapshots are under `agents/plans/transformers/depth_pro/_sources/`.

## 2. High-level architecture

DepthPro is a monocular metric depth-estimation model with a multi-scale ViT encoder stack, DPT-like convolutional neck/fusion, a depth head, and an optional field-of-view head. The representative checkpoint uses three independent DINOv2-large style encoders: full-image encoder, shared patch encoder over scaled image patches, and FOV encoder.

```text
CPU image preprocessing -> NCHW pixel_values
  -> full-image DINOv2 encoder -> feature map
  -> scaled-image patch extraction -> shared patch DINOv2 encoder -> merged multi-scale feature maps
  -> neck upsample/projection -> DPT-like feature fusion -> inverse-depth head
  -> optional FOV encoder/head -> postprocess to metric depth/focal length
```

Independently stageable pieces:
- CPU/data pipeline: rescale/normalize before resize to 1536x1536.
- Encoder graph: DINOv2 full-image encoder plus patch encoder. DINOv2 itself is a separately audited family, but DepthPro owns the patch extraction/merge ABI and feature contracts.
- Decoder graph: convolution/transpose-convolution neck, fusion stage, depth head.
- Optional FOV graph: separate DINOv2 encoder plus convolutional scalar head. The official checkpoint config enables it.
- Postprocess: optional FOV-based depth scaling, resize to target image size, reciprocal/clamp conversion.

## 3. Important config dimensions

Representative checkpoint: `apple/DepthPro-hf`, `torch_dtype` from config is `float16`.

| Field | Value | Source |
|---|---:|---|
| `patch_size` | 384 | DepthPro config |
| preprocessor size | 1536 x 1536 | `preprocessor_config.json` |
| `fusion_hidden_size` | 256 | DepthPro config |
| `scaled_images_ratios` | `[0.25, 0.5, 1]` | DepthPro config |
| `scaled_images_overlap_ratios` | `[0.0, 0.5, 0.25]` | DepthPro config |
| `scaled_images_feature_dims` | `[1024, 1024, 512]` | DepthPro config |
| `intermediate_hook_ids` | `[11, 5]` | DepthPro config, indexes into patch encoder hidden states as `id + 1` |
| `intermediate_feature_dims` | `[256, 256]` | DepthPro config |
| `merge_padding_value` | 3 | DepthPro config |
| `use_fov_model` | true | official checkpoint config |
| `num_fov_head_layers` | 2 | DepthPro config |
| DINOv2 hidden size | 1024 | nested image/patch/FOV configs |
| DINOv2 layers | 24 each encoder | nested configs |
| DINOv2 heads / head dim | 16 / 64 | nested config + source equation |
| DINOv2 MLP width | 4096 | inference from DINOv2 source default `mlp_ratio=4`; omitted in checkpoint config |
| DINOv2 patch size | 16 | nested configs |
| DINOv2 image size | 384 | nested configs, enforced to match DepthPro `patch_size` |
| DINOv2 token length per 384 patch | 577 = CLS + 24 x 24 patches | source shape equation |
| cache support | none | encoder/depth model, no generation |

Checkpoint sweep:

| Repo | Native `depth_pro`? | Useful variation |
|---|---|---|
| `apple/DepthPro-hf` | yes | Main production checkpoint; FOV enabled; fp16 config; 1536 preprocessor. |
| `onnx-community/DepthPro-ONNX` | no, exported ONNX | Same base family but export/quant variants are provider/load concerns, not native Transformers source. |
| `CineAI/Depth-Pro-hf-4bit` | yes tag, community quantized | Potential source-coupled quantization/loading follow-up; not official native baseline. |
| `apple/DepthPro` / `apple/DepthPro-mixin` | no native Transformers or different mixin path | Original/non-Transformers packaging; useful for provenance only. |

## 3a. Family variation traps

- The nested DINOv2 configs omit many fields; effective defaults from `Dinov2Config` include `mlp_ratio=4`, `hidden_act="gelu"`, `layer_norm_eps=1e-6`, `qkv_bias=True`, `layerscale_value=1.0`, and `use_swiglu_ffn=False`.
- `DepthProConfig.__post_init__` rewrites dict sub-config `image_size` to equal DepthPro `patch_size`; preconstructed `PreTrainedConfig` objects with mismatched image size are rejected.
- The DINOv2 encoders are separate physical modules, not shared weights. Patch encoder is shared across all scaled patches inside one forward.
- The patch encoder passes all high/medium/low patches in one concatenated batch and then uses split/merge logic. Intermediate hidden features are taken from the combined patch batch and effectively use the high-resolution patch prefix after square-root truncation.
- Source layout is NCHW throughout convolutional and image-processing graph regions. NHWC should be a guarded layout/fusion optimization; axis-sensitive ops include `F.unfold`, `permute`, `torch.cat(..., dim=1)`, Conv2d/ConvTranspose2d channels, and `squeeze(dim=1)`.
- The model predicts a positive inverse-depth-like map from the head; postprocess returns metric depth as reciprocal after optional FOV scaling and target resize.
- FOV can be disabled via constructor/config, but official checkpoint config has `use_fov_model=true`; first parity should include both modes if weights are loaded from the official checkpoint.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW image tensors, shape checks, `F.interpolate` bilinear/bicubic with `align_corners=False`, `antialias=False` in preprocessing resize.
- `F.unfold` on NCHW scaled images with kernel `384 x 384` and strides derived from overlap.
- `permute`, `reshape`, `view`, `flatten`, `transpose`, `contiguous`, `split_with_sizes`, `torch.cat`, elementwise add, `squeeze(dim=1)`, clamp, reciprocal.
- Square-root/int shape computations and guard logic for square patch grids.

Neural network primitives:
- DINOv2 patch Conv2d `Conv2d(3 -> 1024, kernel=16, stride=16, bias=True)`.
- DINOv2 Linear Q/K/V `1024 -> 1024` with bias, output `1024 -> 1024`, MLP `1024 -> 4096 -> 1024` with GELU, LayerNorm eps `1e-6`, learned layer scale.
- DepthPro neck Conv2d/ConvTranspose2d:
  - image block ConvTranspose2d `1024 -> 1024`, kernel/stride 2.
  - scaled blocks Conv2d `1024 -> 1024`, `1024 -> 1024`, `1024 -> 512` with kernel 1, then ConvTranspose2d kernel/stride 2.
  - intermediate blocks Conv2d `1024 -> 256`, then 2 or 3 ConvTranspose2d `256 -> 256`.
  - global fuse Conv2d `2048 -> 1024`, kernel 1.
  - feature projections Conv2d `1024 -> 256`, `1024 -> 256`, `512 -> 256`, `256 -> 256`, kernel 3 where not identity.
- Fusion residual Conv2d `256 -> 256`, ReLU, optional BatchNorm2d disabled in representative config; fusion ConvTranspose2d `256 -> 256`, projection Conv2d `256 -> 256`.
- Depth head Conv2d `256 -> 128`, ConvTranspose2d `128 -> 128`, Conv2d `128 -> 32`, ReLU, Conv2d `32 -> 1`, ReLU.
- FOV head: Linear `1024 -> 128`, Conv2d `256 -> 128` stride 2, FOV Conv2d `128 -> 64` stride 2, Conv2d `64 -> 32` stride 2, final Conv2d `32 -> 1` kernel 6 for a 24x24 DINO grid with two stride-2 layers.

Attention primitives:
- Noncausal dense encoder self-attention, MHA, 16 heads, head dim 64, no mask for DepthPro calls.
- SDPA-compatible dispatch through DINOv2 attention interface; eager fallback is dense matmul-softmax-matmul.

Position/custom math:
- DINOv2 learned CLS token and learned absolute position table, bicubic interpolation if spatial grid changes.
- No RoPE, ALiBi, relative bias, KV cache, or generation.

Preprocessing-coupled ops:
- Rescale/normalize before resize, unlike many processors that resize first.
- Target-size postprocess and optional FOV-to-focal-length scaling.

## 5. Layer/block breakdown

Representative 1536x1536 input, batch `B`, NCHW:

```text
Preprocess:
  image -> rescale/normalize -> resize to [B,3,1536,1536]

Full image encoder:
  pixel_values -> bilinear resize [B,3,384,384]
  DINOv2 patch conv -> [B,576,1024], prepend CLS -> [B,577,1024]
  repeat 24 DINOv2 blocks -> [B,577,1024]
  drop CLS/reshape -> [B,1024,24,24]

Patch encoder:
  scaled 0.25 image [B,3,384,384] -> 1 patch/B
  scaled 0.5 image [B,3,768,768] -> 3x3 patches/B with stride 192
  scaled 1.0 image [B,3,1536,1536] -> 5x5 patches/B with stride 288
  concat high, medium, low patches -> [35B,3,384,384]
  shared DINOv2 patch encoder -> [35B,577,1024] plus hidden states
  reconstruct low/medium/high maps -> [B,1024,24,24], [B,1024,48,48], [B,1024,96,96]
  reconstruct intermediate maps from hidden states 12 and 6 -> [B,1024,96,96] each

Neck:
  upsample image -> [B,1024,48,48]
  upsample scaled maps -> [B,1024,48,48], [B,1024,96,96], [B,512,192,192]
  upsample intermediate maps -> [B,256,384,384], [B,256,768,768]
  cat image + low scaled on channels -> [B,2048,48,48] -> [B,1024,48,48]
  project all features to 256 channels -> five maps at 48,96,192,384,768

Fusion and depth:
  DPT-like progressive fusion/deconv -> final [B,256,768,768]
  depth head -> [B,1,1536,1536] -> squeeze dim 1 -> [B,1536,1536]
```

DINOv2 block repeated 24 times in each nested encoder:

```text
x = LayerNorm(x)
q,k,v = Linear(1024 -> 1024, bias=True), reshape [B,16,T,64]
x_attn = noncausal_attention(q,k,v)
x = x + layer_scale(Linear(1024 -> 1024)(x_attn))
y = LayerNorm(x)
y = Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
x = x + layer_scale(y)
```

## 6. Attention requirements

- Variant: encoder-only noncausal self-attention.
- Heads: MHA, 16 Q heads, 16 K/V heads, head dim 64, q/k/v widths 1024.
- Sequence lengths: 577 tokens for 384x384 patches/full image with DINO patch size 16. Position interpolation supports other grids, but DepthPro config forces 384 sub-encoder input size for representative path.
- Masking: DepthPro does not pass an attention mask to nested DINOv2 encoders. `bool_masked_pos` is not used by DepthPro calls.
- Cache: no autoregressive KV cache. Encoder outputs and patch features can be cached only as application-level image feature caches.
- Backend: DINOv2 dispatches through `ALL_ATTENTION_FUNCTIONS` and supports SDPA; eager fallback materializes dense `[B,16,T,T]` attention and is likely too slow for the 35B patch-encoder batch.

## 7. Position encoding and custom math

DINOv2 absolute position interpolation is inherited by all three nested encoders:

```python
def dinov2_pos_embed(table, tokens, height, width, patch_size):
    cls = table[:, :1]
    patch = table[:, 1:].reshape(1, S, S, C).permute(0, 3, 1, 2)
    patch = interpolate(patch.float(), size=(height // patch_size, width // patch_size),
                        mode="bicubic", align_corners=False).to(table.dtype)
    patch = patch.permute(0, 2, 3, 1).view(1, -1, C)
    return concat([cls, patch], dim=1)
```

DepthPro-specific patch extraction/merge math:

```python
def patch_stride(patch_size, overlap):
    return int(patch_size * (1 - overlap))

def postprocess_depth(raw, fov_deg, target_h, target_w):
    if fov_deg is not None:
        focal = 0.5 * target_w / tan(0.5 * deg2rad(fov_deg))
        raw = raw * target_w / focal
    raw = resize(raw[None, None], (target_h, target_w)).squeeze()
    return 1.0 / clamp(raw, 1e-4, 1e4)
```

Precomputable: DINOv2 position tables and fixed Conv/Linear weights. Dynamic: number of unfolded patches for arbitrary input size, merge crop sizes, target resize, and FOV-derived focal scaling.

## 8. Preprocessing and input packing

CPU/data-pipeline contract from `DepthProImageProcessor`:
- Input images are grouped by shape.
- Rescale and normalize are applied before resizing. Official preprocessor uses `rescale_factor=1/255`, `image_mean=0.5`, `image_std=0.5`, producing roughly `[-1, 1]`.
- Resize to 1536x1536 uses bilinear interpolation and `antialias=False`.
- Output is `pixel_values` in NCHW.

GPU/runtime packing:
- Patch extraction is part of the model graph, not the image processor. It uses scaled copies of `pixel_values`, `F.unfold`, and concatenates high-res patches first.
- For 1536x1536: patch counts are 1, 9, and 25 per image. The DINOv2 patch encoder sees `35 * B` images of shape `[3,384,384]`.
- Patch merge assumes a square patch grid and may truncate extra patches through `sqrt_n_patches_per_batch**2`.

Postprocessing:
- If `target_sizes` is supplied and FOV is present, compute focal length from target width and predicted horizontal FOV, scale raw prediction, resize to target size, then return reciprocal clamped depth.
- Output records per image: `predicted_depth`, `field_of_view`, `focal_length`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: DINOv2 patch Conv2d -> WindowFlatten + GEMM

Source pattern: `Conv2d(3 -> 1024, kernel=16, stride=16, padding=0)` followed by `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW WindowFlatten16x16 -> MatMul(weight_flat.T) -> BiasAdd -> [B,576,1024]
```

Preconditions:
- `kernel_size == stride == (16,16)`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input H/W divisible by 16.
- Preserve PyTorch NCHW flatten order unless a local NHWC region controls both producer and consumer.

Weight transform:

```python
w = conv.weight.reshape(1024, 3 * 16 * 16)
```

Failure cases: non-DINO sub-configs, changed patch sizes, nonzero padding, or non-divisible dynamic input.

### Rewrite: DepthPro patch extraction as explicit im2col

Source pattern: `F.unfold(pixel_values, kernel_size=384, stride=stride)` then `permute(2,0,1).reshape(-1,3,384,384)`.

Replacement:

```text
ScaledImage -> Unfold2D/NCHW im2col -> PatchBatchReshape
```

Preconditions:
- Square images or square grid assumption validated.
- Patch size and overlap match config.
- Preserve high/medium/low concatenation order and per-scale split sizes.

Layout constraints: this is a `no_layout_translation()` candidate unless the layout pass owns `unfold`, patch reshape, and downstream DINO patch embedding axes together.

### Rewrite: NCHW fusion block to NHWC local region

Source pattern: Conv2d/ReLU/ConvTranspose2d chains over fixed channel counts.

Replacement:

```text
NCHW input -> guarded NHWC conv/deconv fusion region -> NCHW output
```

Preconditions:
- All consumers inside the region are controlled.
- Rewrite channel concat `dim=1 -> dim=-1`, squeeze `dim=1 -> dim=-1`, and Conv weight layout.
- Reintroduce NCHW before public feature outputs or postprocess if consumers require source layout.

Failure cases: residual feature maps consumed outside the region, optional BatchNorm axis mismatch, or dynamic shape ops that assume NCHW.

### Rewrite: postprocess reciprocal-depth fusion

Source pattern: optional scalar FOV scale, resize, clamp, reciprocal.

Replacement:

```text
Resize -> Clamp(min=1e-4,max=1e4) -> Reciprocal
```

Preconditions:
- Same interpolation mode as processor mapping for bilinear.
- FOV scale is applied before resize as source does.
- Target size known per image or vectorized with per-sample dispatch.

## 10. Kernel fusion candidates

Highest priority:
- DINOv2 encoder attention/MLP kernels for the patch encoder batch of 35B sequences. This is the dominant compute.
- Patch Conv2d-to-GEMM and patch extraction/unfold packing, because the model repeatedly processes 384x384 windows.
- ConvTranspose2d-heavy neck/fusion/depth head kernels in channel-last optimized regions with strict axis guards.

Medium priority:
- LayerNorm + QKV projection fusion for DINOv2, preserving separate Q/K/V weights and qkv bias.
- GELU MLP fusion `1024 -> 4096 -> 1024`.
- DPT fusion residual Conv-ReLU-Conv residual chains.
- FOV branch scalar head if official checkpoint parity includes FOV.

Lower priority:
- Optional BatchNorm path in fusion residuals, disabled for official checkpoint.
- Position interpolation fast path for fixed 384 inputs; it can be precomputed for representative shape.
- Postprocess clamp/reciprocal fusion, usually memory-bandwidth-light versus encoders.

## 11. Runtime staging plan

1. Parse `DepthProConfig` and nested DINOv2 configs; reject unsupported non-DINO nested backbones initially.
2. Load weights for `apple/DepthPro-hf`; validate aliasing: three separate DINOv2 modules plus separate FOV module.
3. Implement/prevalidate DINOv2 encoder parity for one 384x384 tensor.
4. Implement DepthPro patch extraction and merge ABI for 1536x1536, then patch encoder feature parity.
5. Add full-image encoder and neck feature parity.
6. Add fusion stage and depth head parity.
7. Add optional FOV model parity and postprocess focal/depth conversion.
8. Add guarded NHWC/local conv and patch Conv2d-to-GEMM optimizations.
9. Benchmark batched images and memory pressure.

Initially stub: training loss, labels, optional hidden-state/attention outputs, non-DINO nested backbones, community quantized variants.

## 12. Parity and validation plan

- Unit parity for `split_to_patches` over 384, 768, 1536 inputs with overlaps 0, 0.5, 0.25.
- Unit parity for `merge_patches` with padding 0/3/6/12 and odd patch-count truncation behavior.
- DINOv2 single-block parity in fp32 and fp16/bf16.
- Patch encoder parity: compare split sizes and reconstructed feature shapes/values for 1536 input.
- Neck/fusion/depth-head parity from random feature maps with fixed shapes `[48,96,192,384,768]`.
- FOV parity with and without `use_fov_model`.
- End-to-end `predicted_depth` before postprocess, then postprocessed depth/focal length with a fixed target size.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` for full graph, tighter for isolated conv/linear where accumulation is controlled.

## 13. Performance probes

- Preprocessor throughput for rescale/normalize-before-resize at 1536x1536.
- Patch extraction/unfold bandwidth and temporary memory for `B * 35` patch batch.
- DINOv2 patch encoder throughput for `35B x 577 x 1024`.
- Full-image DINOv2 and FOV DINOv2 throughput separately.
- Neck/fusion/depth-head ConvTranspose2d throughput and peak activation memory.
- Batch-size sweep: B=1,2,4,8 if memory permits.
- Image-size sweep for non-1536 inputs to expose dynamic patch-count and position-interpolation costs.
- Attention backend comparison: eager vs SDPA/Flash-compatible dense encoder attention.
- NHWC conv-region comparison with explicit layout conversion overhead included.

## 14. Skip/defer list

- Training/loss path: source raises `NotImplementedError` when labels are provided.
- Gradient checkpointing and stochastic depth behavior for inference.
- Non-DINO nested backbone configs.
- Community 4-bit/bitsandbytes loading and ONNX/CoreML exported-provider contracts.
- Attention output materialization and full hidden-state output unless debugging requires them.
- Dynamic arbitrary aspect ratios beyond validated processor path, until patch merge square-grid assumptions are admitted.

## 15. Final implementation checklist

- [ ] Parse `DepthProConfig` and nested DINOv2 configs, including omitted DINOv2 defaults.
- [ ] Load `apple/DepthPro-hf` weights and keep image/patch/FOV encoders as distinct modules.
- [ ] Implement DINOv2 encoder block coverage for DepthPro shapes.
- [ ] Implement `split_to_patches`, patch ordering, split sizes, and `merge_patches`.
- [ ] Implement DepthPro neck Conv2d/ConvTranspose2d projections.
- [ ] Implement DPT-like fusion stage and depth head.
- [ ] Implement optional FOV encoder/head and focal-length postprocess.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite.
- [ ] Add guarded NCHW-to-NHWC conv/deconv fusion regions with axis rewrite checks.
- [ ] Add parity tests for patch merge, encoder features, depth logits, FOV, and postprocess.
- [ ] Benchmark patch encoder, full-image encoder, FOV encoder, and fusion/head separately.
