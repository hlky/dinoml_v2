# Transformers Audit: pp_ocrv5_server_det

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-OCRv5_server_det_safetensors
Config source: HF config/preprocessor plus original PaddleOCR config for cross-check
Primary runtime target: image text detector probability map + detector postprocess
Source files inspected: pp_ocrv5_server_det config/modeling/image_processing/modular files; hgnet_v2 config/modeling; model tests
Any missing files or assumptions: only one native Transformers server-det checkpoint config found; no imports, tests, or safetensors metadata inspection
```

The generated `configuration_*.py`, `image_processing_*.py`, and `modeling_*.py` files are generated from `modular_pp_ocrv5_server_det.py`; future source edits should use the modular file. The backbone is delegated through `load_backbone(config)` to `hgnet_v2`, so this report owns the detector neck/head/preprocess/postprocess and records the backbone feature contract, but a reusable HGNetV2 port deserves its own audit.

## 2. High-level architecture

This is a CNN OCR text detector, not a text Transformer and not an autoregressive model.

```text
CPU/image pipeline
  BGR image -> aspect-preserving resize to multiples of 32 -> rescale/normalize -> NCHW pixel_values

GPU/neural graph
  pixel_values [B,3,H,W]
  -> HGNetV2 backbone feature maps [stage1..stage4]
  -> large-kernel path-aggregation neck
  -> segmentation head + local refinement head
  -> probability map [B,1,H,W]

CPU/OpenCV postprocess
  probability map + original target_sizes
  -> threshold bitmap -> contours -> min-area boxes -> unclip -> score filter
  -> variable-length rotated boxes, scores, labels=0
```

Independently stageable parts are: image preprocessor, HGNetV2 backbone, detector neck, segmentation/local-refinement head, and CPU/OpenCV detector postprocess. First DinoML parity should target `PPOCRV5ServerDetForObjectDetection` logits/probability maps for NCHW inputs; end-to-end OCR detector parity additionally requires the OpenCV contour path.

## 3. Important config dimensions

| Field | Production safetensors config / source default | Runtime meaning |
| --- | --- | --- |
| `model_type` | `pp_ocrv5_server_det` | Native Transformers detector family |
| `backbone_config.model_type` | `hgnet_v2` | Delegated CNN backbone |
| `backbone_config.out_features` | `["stage1","stage2","stage3","stage4"]` | Neck consumes four image-like NCHW maps |
| HGNet stage channels | default `[128,512,1024,2048]` unless config overrides | Neck input channel list |
| `neck_out_channels` | `256` | Neck emits `256` channels before head |
| `reduce_factor` | `2` | Intraclass block reduces `64 -> 32` channels |
| `intraclass_block_number` | `4` | One intraclass block per fused level |
| `scale_factor_list` | `[1,2,4,8]` | Upsamples all neck levels to same spatial size |
| `scale_factor` | `2` | Local-refinement feature upsample |
| `hidden_act` | `relu` | Conv-BN activation wrapper |
| `kernel_list` | `[3,2,2]` | Head conv/deconv kernels |
| `interpolate_mode` | source default `nearest`; production config uses legacy `upsample_mode` | Modeling source reads only `interpolate_mode` |
| image resize | side limit 960, max side 4000, rounded to multiple of 32 | Dynamic H/W guard |
| postprocess thresholds | bitmap `0.3`, box score `0.6`, unclip `1.5`, max candidates `1000` | End-to-end detector ABI |

Representative config sweep:

| Source | Status | Operator-significant differences |
| --- | --- | --- |
| `PaddlePaddle/PP-OCRv5_server_det_safetensors` | native HF checkpoint | HGNetV2-L defaults, neck 256, four intraclass blocks, DB-style postprocess via image processor |
| Transformers unit-test synthetic config | source test only | Tiny HGNet channels `[16,32,64,128]`, neck 32, same topology, useful for shape tests |
| `PaddlePaddle/PP-OCRv5_server_det` | original PaddleOCR repo, not native HF weights | Confirms Paddle pre/postprocess and TRT dynamic shape envelope `[1,3,32,32]` to `[1,3,4000,4000]` |
| `PaddlePaddle/PP-OCRv5_mobile_det_safetensors` | out of scope | Separate `model_type=pp_ocrv5_mobile_det` and different backbone, do not merge into this audit |

## 3a. Family variation traps

- The production HF config contains Paddle-era fields (`mode`, `upsample_mode`, `use_lab`, `use_last_conv`, `class_expand`, `class_num`, `head_in_channels`) that the inspected Transformers detector source does not read. DinoML should ignore or reject these only according to native-source behavior, not Paddle expectations.
- `intraclass_block_config`, `scale_factor_list`, and `kernel_list` default to `None` in the config class but are indexed in modeling. Real checkpoints must provide them, or graph construction fails.
- `interpolate_mode` is the source-read field. The production config contains `upsample_mode`, so effective native behavior depends on whether config loading aliases unknown Paddle fields. Gate this before compile; do not silently assume `upsample_mode` is honored by modeling.
- The graph is NCHW throughout source modeling. NHWC/channel-last is an optimization candidate only inside fully controlled conv/BN/upsample/cat regions.
- Dynamic spatial shapes are real but bounded: H/W must be positive multiples of 32 for the backbone/neck scale ladder and top-down additions to align. Preprocessing enforces this; raw `pixel_values` callers must be guarded.
- Postprocess is variable-output and CPU/OpenCV heavy. It is not NMS and not a fixed `[B,N,4]` detector head.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW dense tensors, dynamic batch and dynamic H/W with multiple-of-32 guards.
- `torch.cat(..., dim=1)` for channel concatenation in HGNet stem/basic layers, neck final concat, and local refinement.
- `F.pad(..., (0,1,0,1))` in HGNetV2 stem.
- Elementwise add chains and scalar multiply/add: residuals, multi-branch conv sums, `0.5 * (a + b)`.

Neural network primitives:
- Conv2d NCHW with kernels 1x1, 2x2, 3x3, 5x5, 7x7, 9x9, asymmetric 7x1/1x7, 5x1/1x5, 3x1/1x3.
- Depthwise Conv2d in HGNetV2 light blocks and downsample paths (`groups=in_channels`).
- ConvTranspose2d NCHW with kernel/stride 2 for head upsampling.
- BatchNorm2d in inference mode, fusible into Conv2d/ConvTranspose2d where affine/running stats are frozen.
- ReLU and sigmoid.
- MaxPool2d kernel 2, stride 1, `ceil_mode=True` in HGNetV2 stem.
- Nearest-neighbor upsample by integer scale factors 2, 4, 8.

Preprocessing-coupled ops:
- Resize preserving aspect ratio, limit modes `max`, `min`, `resize_long`; round resized H/W to nearest multiple of 32 and clamp minimum 32.
- Rescale by `1/255`, channel normalization, BGR to RGB reorder in processor source, output `pixel_values` as NCHW.
- Return `target_sizes` as original `[height,width]` for postprocess scaling.

Postprocessing ops:
- Threshold probability map to bitmap.
- OpenCV `findContours`, `minAreaRect`, `boxPoints`, polygon mask fill, contour area/perimeter, polygon unclip, score filtering, coordinate scaling and clipping.
- Variable-length outputs per image: `boxes [N,4,2]` int16, `scores [N]`, `labels [N]` all zero. Source docstring claims corners `[N,4]`, but integration tests expect rotated quadrilateral boxes `[N,4,2]`.

Attention/generation/cache ops:
- No attention, no RoPE, no token cache, no generation controller, no KV cache.

## 5. Layer/block breakdown

HGNetV2 backbone contract, composed dependency:

```text
pixel_values [B,3,H,W] NCHW
-> stem conv/BN/ReLU + pad + maxpool/cat ladder
-> stage1 [B,128,H/4,W/4]
-> stage2 [B,512,H/8,W/8]
-> stage3 [B,1024,H/16,W/16]
-> stage4 [B,2048,H/32,W/32]
```

Detector neck:

```text
for each stage i:
  c_i = Conv2d(C_i -> 256, 1x1, bias=False)(stage_i)

top_down[3] = c_3
for i = 2..0:
  top_down[i] = c_i + nearest_upsample(top_down[i+1], scale=2)

for each i:
  p_i = Conv2d(256 -> 64, 9x9, padding=4, bias=False)(top_down[i] or c_3)

bottom_up[0] = p_0
for i = 1..3:
  bottom_up[i] = p_i + Conv2d(64 -> 64, 3x3, stride=2, padding=1)(bottom_up[i-1])

for each i:
  l_i = Conv2d(64 -> 64, 9x9, padding=4)(p_0 if i==0 else bottom_up[i])
  r_i = IntraclassBlock(l_i)

upsampled = [r0, up2(r1), up4(r2), up8(r3)]
neck_out = cat(reverse(upsampled), dim=1)  # [B,256,H/4,W/4]
```

Intraclass block for each 64-channel level:

```text
residual = x
y = Conv2d(64 -> 32, 1x1)(x)
y = Conv2d(7x7)(y) + Conv2d(7x1)(y) + Conv2d(1x7)(y)
y = Conv2d(5x5)(y) + Conv2d(5x1)(y) + Conv2d(1x5)(y)
y = Conv2d(3x3)(y) + Conv2d(3x1)(y) + Conv2d(1x3)(y)
y = Conv2d(32 -> 64, 1x1, bias=True) -> BatchNorm2d -> ReLU
return residual + y
```

Head:

```text
z = Conv2d+BN+ReLU(256 -> 64, 3x3)
z = ConvTranspose2d+BN+ReLU(64 -> 64, 2x2, stride=2)
feature = z
initial = sigmoid(ConvTranspose2d(64 -> 1, 2x2, stride=2))  # [B,1,H,W]
feature = nearest_upsample(feature, scale=2)                 # [B,64,H,W]
refined = cat([initial, feature], dim=1)
refined = Conv2d+BN+ReLU(65 -> 64, 3x3)
refined = sigmoid(Conv2d(64 -> 1, 1x1))
logits = 0.5 * (initial + refined)
```

## 6. Attention requirements

No attention is required for the primary target. There is no causal or noncausal attention, no packed/varlen metadata, no masks, no RoPE/relative bias, no FlashAttention/SDPA path, and no KV cache. Independently cacheable data is limited to static weights and, optionally, preprocessed image batches or backbone feature maps for debugging; these are not generation caches.

## 7. Position encoding and custom math

No position encoding exists in the detector graph. The custom math that matters is detector postprocess:

```python
mask = prediction > threshold
contours = cv2.findContours((mask * 255).astype(uint8), RETR_LIST, CHAIN_APPROX_SIMPLE)
box = cv2.boxPoints(cv2.minAreaRect(contour))
score = mean(prediction inside polygon mask)
offset_distance = polygon_area * unclip_ratio / polygon_perimeter
box_xy[:, 0] = round(box_xy[:, 0] * original_width / map_width)
box_xy[:, 1] = round(box_xy[:, 1] * original_height / map_height)
```

This belongs in a CPU/data-pipeline or postprocess component at first. A GPU implementation would need a separate variable-output contour/geometry provider contract.

## 8. Preprocessing and input packing

The processor groups same-shape images for batching, computes resize size per original shape, resizes with torchvision, then groups again for rescale/normalize. Source expects tensor images in channel-first form by the time `_preprocess` sees them; the HF preprocessor metadata also records Paddle-style decode as BGR/HWC followed by `ToCHWImage`.

Critical preprocessing contract:
- Input semantic image mode is BGR in the HF preprocessor metadata.
- Resize ratio depends on `limit_type`: `max`, `min`, or `resize_long`.
- Resized H/W are rounded to multiples of 32 with minimum side 32 and optional max side 4000.
- Pixel values are rescaled, normalized, then channel-reordered by `[:, [2,1,0], :, :]` in source.
- Model input is NCHW `pixel_values [B,3,H,W]`.
- `target_sizes` stores original image height/width and is required by `post_process_object_detection`.

Layout boundary:
- Preprocessor may begin from HWC/BGR, but model graph begins at NCHW. DinoML should not translate to NHWC globally unless it owns decode, resize, normalize, channel reorder, and every downstream conv/BN/upsample/cat axis rewrite.
- Channel-axis ops that must be guarded under layout translation: `dim=1` cats, channel reorder, BatchNorm2d channel stats, Conv2d weight layout, and depthwise `groups=in_channels`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d + ReLU -> fused inference conv

Preconditions:
- Inference mode with frozen BatchNorm running mean/var and affine parameters.
- Conv output feeds exactly the matching BN and activation, or safe local use.
- Preserve NCHW axis semantics unless a fully guarded NHWC layout island is selected.

Replacement:
```text
Conv2d(weight_fused, bias_fused) -> ReLU
```

Failure cases: training mode, mutable BN stats, shared intermediate output, unsupported grouped/depthwise conv, or unknown dtype tolerance.

Parity test sketch: compare each fused block on random NCHW tensors for fp32 and selected fp16/bf16 tolerances.

### Rewrite: ConvTranspose2d stride-2 2x2 -> nearest/zero-insert upsample plus Conv2d, or keep native provider

Preconditions:
- `kernel_size == stride == 2`, no padding/output padding/dilation changes.
- Weight transform and output-size equation exactly match PyTorch ConvTranspose2d.

Replacement:
```text
ConvTranspose2d provider
```

Preferred first step is a real ConvTranspose2d lowering/provider. Rewriting to upsample+conv is risky unless output geometry and weight placement are proven.

### Rewrite: nearest upsample + add in FPN neck

Preconditions:
- Scale factor is integer 2, mode is nearest, no align-corners behavior.
- Source and target feature maps have matching NCHW H/W after upsample.
- Input H/W divisible by 32.

Replacement:
```text
FusedNearestUpsampleAdd(top_down[i+1], channel_adjusted[i])
```

Failure cases: non-nearest interpolation, odd dynamic shapes, or NHWC pass that does not rewrite axes.

### Rewrite: large-kernel parallel intraclass conv sum

Preconditions:
- All branches consume the same reduced tensor and have same output shape/channels.
- Padding/stride from `intraclass_block_config` exactly preserves spatial size.

Replacement:
```text
ParallelConv2dSum3(branch kernels) repeated for 7/5/3 groups
```

Weight transform: none initially; future provider may fuse branch launches by concatenating output channels then reducing if profitable.

### Rewrite: NCHW conv island -> NHWC/channel-last optimized island

Preconditions:
- The island starts and ends at explicit layout conversion boundaries.
- Rewrite all axis-sensitive ops: `cat dim=1 -> dim=-1`, BatchNorm channel dimension, depthwise groups, channel reorder, Conv2d weight packing, and postprocess map interpretation.
- No external consumer expects intermediate NCHW tensors.

Failure cases: hidden-state outputs requested, direct feature-map debugging, unsupported ConvTranspose2d NHWC provider, or source `F.pad`/MaxPool ceil semantics not reproduced.

## 10. Kernel fusion candidates

Highest priority:
- NCHW Conv2d/DepthwiseConv2d + BatchNorm2d + ReLU inference fusion. Nearly the whole backbone and head are built from this.
- Nearest upsample + add/cat neck kernels. The detector neck has many small feature-map operations that can become launch-bound.
- ConvTranspose2d support for the head. Without it, end-to-end graph parity blocks.

Medium priority:
- Large-kernel 9x9 and asymmetric intraclass conv tuning. These dominate detector-specific neck cost after backbone support.
- Sigmoid + average final map fusion.
- CPU/OpenCV-compatible postprocess module, initially outside compiled DinoML graph.

Lower priority:
- Full NHWC/channel-last conv island. Useful only after NCHW parity is stable and every axis-sensitive boundary is guarded.
- GPU contour extraction/unclip. Valuable for high-throughput OCR but much more complex than fixed-shape neural graph lowering.

## 11. Runtime staging plan

Stage 1: parse config and reject unsupported/ambiguous fields. Require `intraclass_block_config`, `scale_factor_list`, `kernel_list`, `interpolate_mode="nearest"`, HGNetV2 backbone, and NCHW input H/W divisible by 32.

Stage 2: implement or compose HGNetV2 backbone parity as a separate dependency. Validate feature-map shapes and channels for stage1..stage4.

Stage 3: lower detector neck only with random backbone feature maps. Cover upsample/add/cat/intraclass blocks and dynamic H/W buckets.

Stage 4: lower segmentation/local-refinement head and produce `[B,1,H,W]` probability maps.

Stage 5: wire full `PPOCRV5ServerDetForObjectDetection` logits parity against Transformers for representative resized shapes.

Stage 6: add CPU postprocess parity using OpenCV-compatible routines. Keep variable-length outputs outside the fixed tensor graph at first.

Stage 7: add fusions/layout optimization under explicit guards.

## 12. Parity and validation plan

- Config admission tests: official safetensors config, unit-test tiny config, missing `intraclass_block_config`, missing `scale_factor_list`, non-nearest interpolation, non-HGNet backbone.
- Preprocessor shape tests: original image sizes around 31/32/33, large side over 960, max side over 4000, `max`/`min`/`resize_long` modes.
- Single-op tests: Conv2d, depthwise Conv2d, ConvTranspose2d, BatchNorm inference fusion, MaxPool2d ceil-mode, pad `(0,1,0,1)`, nearest interpolate.
- Neck parity: random four-stage feature maps with shapes `[H/4,H/8,H/16,H/32]`, compare neck output.
- Head parity: random `[B,256,H/4,W/4]`, compare probability map `[B,1,H,W]`.
- Full model parity: compare `last_hidden_state` on official checkpoint for at least square and rectangular resized inputs.
- Postprocess parity: fixed probability maps with simple rectangles, rotated contours, tiny boxes below `min_size`, low score boxes, and empty maps.
- Recommended tolerances: fp32 `rtol/atol` around `1e-4`; fp16/bf16 need looser per-layer and end-to-end tolerances due BN/conv accumulation.

## 13. Performance probes

- Preprocessor throughput by input size and batch grouping behavior.
- Backbone-only throughput for NCHW dynamic shapes: 736, 960, and max-stress 4000 where memory permits.
- Neck-only throughput with four feature-map inputs to isolate upsample/add/large-kernel conv cost.
- Head-only throughput, especially ConvTranspose2d.
- End-to-end neural graph throughput by batch size and aspect ratio.
- CPU postprocess latency versus number of contours and `max_candidates`.
- NCHW baseline versus guarded NHWC/channel-last island after parity is stable.
- Memory usage for dynamic max-shape artifacts and temporary feature maps.

## 14. Skip/defer list

- Training, loss functions, labels, gradient checkpointing, dropout behavior.
- Classification head from HGNetV2; not used by this detector target.
- Mobile detector family; it has a different model type/backbone.
- GPU/OpenCV contour replacement and variable-length output allocation inside compiled graph.
- Global NHWC translation as default semantics.
- PaddleOCR-only runtime fields not read by native Transformers source unless a separate Paddle compatibility target is requested.

## 15. Final implementation checklist

- [ ] Parse `PPOCRV5ServerDetConfig` and nested `HGNetV2Config`.
- [ ] Add admission guards for required detector config fields and NCHW H/W multiple-of-32 shapes.
- [ ] Implement/load HGNetV2 backbone feature contract or depend on separate HGNetV2 audit.
- [ ] Implement Conv2d, depthwise Conv2d, ConvTranspose2d, BatchNorm2d inference, ReLU, sigmoid, MaxPool2d ceil-mode, pad, nearest upsample, cat, add, scalar multiply.
- [ ] Add detector neck parity tests using random feature maps.
- [ ] Add segmentation/local-refinement head parity tests.
- [ ] Add full detector logits parity on official checkpoint.
- [ ] Add CPU/OpenCV-compatible postprocess parity for boxes/scores/labels.
- [ ] Add Conv+BN(+ReLU) fusion after unfused parity.
- [ ] Add guarded NCHW-to-NHWC layout island only after axis rewrite tests exist.
- [ ] Benchmark preprocessing, backbone, neck, head, postprocess, and end-to-end throughput separately.
