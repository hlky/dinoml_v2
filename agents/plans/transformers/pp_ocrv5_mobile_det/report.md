# PP-OCRv5 Mobile Det Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-OCRv5_mobile_det_safetensors
Config source: HF config/preprocessor plus original Paddle inference config snapshots in _sources/
Primary runtime target: OCR text-line detection probability-map model plus required DB postprocess
Source files inspected:
- transformers/src/transformers/models/pp_ocrv5_mobile_det/configuration_pp_ocrv5_mobile_det.py
- transformers/src/transformers/models/pp_ocrv5_mobile_det/modeling_pp_ocrv5_mobile_det.py
- transformers/src/transformers/models/pp_ocrv5_mobile_det/modular_pp_ocrv5_mobile_det.py
- transformers/src/transformers/models/pp_ocrv5_server_det/image_processing_pp_ocrv5_server_det.py
- transformers/src/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py
- transformers/src/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py
Any missing files or assumptions: pp_ocrv5_mobile_det has no family-local image processor; AutoImageProcessor maps it to PPOCRV5ServerDetImageProcessor. Modeling/config files are generated from modular source; modular_pp_ocrv5_mobile_det.py is authoritative for source edits.
```

HF/config snapshots saved under `agents/plans/transformers/pp_ocrv5_mobile_det/_sources/`:

- `hf_PaddlePaddle_PP-OCRv5_mobile_det_safetensors_config.json`
- `hf_PaddlePaddle_PP-OCRv5_mobile_det_safetensors_preprocessor_config.json`
- `hf_PaddlePaddle_PP-OCRv5_mobile_det_config.json`
- `hf_PaddlePaddle_PP-OCRv5_mobile_det_inference.yml`
- `hf_JoyCN_PaddleOCR-Pytorch_PP-OCRv5_mobile_det.yml`

Hub sources used: [safetensors config](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors/blob/main/config.json), [safetensors repo](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_safetensors), [original Paddle static repo](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det), and [JoyCN mirror](https://huggingface.co/JoyCN/PaddleOCR-Pytorch). JoyCN is an open mirror, not the native Transformers checkpoint basis.

## 2. High-level architecture

This is a CNN text detector, not an autoregressive Transformer. The model emits a single-channel segmentation/probability map for text regions; end-to-end OCR detection parity needs the processor DB postprocess to turn that map into variable-count quadrilateral boxes.

```text
BGR/HWC image decode -> resize to 32-multiple NCHW pixel_values -> normalize/channel reorder
  -> PP-LCNetV3 backbone feature maps stage2..stage5
  -> 1x1 channel projection -> RSE/FPN-style neck with nearest upsample/add/concat
  -> DB-style segmentation head with ConvTranspose2d upsampling
  -> sigmoid probability map [B, 1, H_resized, W_resized]
  -> CPU/OpenCV DB postprocess -> boxes/scores/labels
```

Stage boundaries:

- CPU/data pipeline: image decode, BGR handling, aspect-preserving resize, 32-multiple rounding, rescale/normalize, target-size tracking.
- GPU/runtime first target: NCHW `pixel_values -> last_hidden_state` probability map.
- CPU postprocess first target: threshold, contours, min-area boxes, box scoring, unclip expansion, scale to original image sizes.
- Deferred: text recognition and full OCR pipeline composition; this family only owns detection.

## 3. Important config dimensions

| Field | Native safetensors config / source default | Operator impact |
|---|---:|---|
| `model_type` | `pp_ocrv5_mobile_det` | AutoModelForObjectDetection route. |
| `backbone_config.model_type` | `pp_lcnet_v3` | Composed backbone; report includes required backbone ops. |
| `backbone_config.scale` | `0.75` | Scales PP-LCNetV3 channels. |
| `backbone_config.divisor` | `16` | Channel rounding via `make_divisible`. |
| `out_features` / `out_indices` | `stage2..stage5` / `[2,3,4,5]` | Feature strides inferred as `/4,/8,/16,/32` for standard input. |
| Backbone channels consumed | inferred `48,96,192,384` | From PP-LCNetV3 scale 0.75/divisor 16. |
| `layer_list_out_channels` | `[12,18,42,360]` | 1x1 projections before neck. |
| `neck_out_channels` | `96` | Neck internal concat output is 96 channels. |
| `reduction` | `4` | SE/RSE reduction. |
| `kernel_list` | `[3,2,2]` | Head conv 3x3, then two stride-2 deconvs. |
| `interpolate_mode` | `nearest` | Neck/top-down upsample ABI. |
| Processor `limit_side_len` | `960` | Default resize long/max side, depending config path. |
| Processor `max_side_limit` | `4000` | Runtime max side guard. |
| Processor normalization | scale `1/255`, mean/std RGB-equivalent | Source accepts BGR and reorders after normalization. |
| Postprocess thresholds | `thresh=0.3`, `box_thresh=0.6`, `max_candidates=1000`, `unclip_ratio=1.5` | Required for detection parity. |

Representative config sweep:

| Source | Native to current Transformers? | Neural config | Pre/postprocess config | Notes |
|---|---|---|---|---|
| `PaddlePaddle/PP-OCRv5_mobile_det_safetensors` | Yes | `pp_ocrv5_mobile_det`, PP-LCNetV3 x0.75, neck 96, kernels `[3,2,2]` | `preprocessor_config.json`: BGR, `limit_side_len=960`, `max_side_limit=4000`, mean `[0.406,0.456,0.485]` | Best first DinoML target. |
| `PaddlePaddle/PP-OCRv5_mobile_det` | No, Paddle static inference package | Static Paddle config with TensorRT dynamic shape min/opt/max `[1,3,32,32]`, `[1,3,736,736]`, `[1,3,4000,4000]` | DBPostProcess thresholds same as above; `DetResizeForTest.resize_long=960` | Useful ABI/postprocess reference; weights are Paddle format, not current HF safetensors. |
| `JoyCN/PaddleOCR-Pytorch` mirror | No, mirror/converted PyTorch package | YAML says `PPLCNetV3 scale=0.75`, `RSEFPN out_channels=96`, `DBHead k=50` | Same DBPostProcess thresholds; training config includes DB loss maps | Confirms upstream topology but not the in-library Transformers source. |

## 3a. Family variation traps

- `pp_ocrv5_mobile_det` delegates preprocessing/postprocessing to `PPOCRV5ServerDetImageProcessor`; do not look for a mobile-specific processor file.
- Current in-library mobile head returns only the sigmoid map. It does not implement the Paddle DB training threshold branch or adaptive binary map path advertised by some upstream YAML training configs.
- The source graph is NCHW throughout. NHWC is only an optimization candidate behind a fully guarded layout pass.
- The processor starts from BGR/HWC style inputs and returns NCHW `pixel_values`; channel order is axis-sensitive.
- `target_sizes` are original image sizes and are mandatory for postprocess scaling.
- Output boxes are variable-count quadrilaterals with shape `(N,4,2)` despite doc text saying corners `(N,4)`.
- No NMS is present in source postprocess; contour extraction plus thresholding/unclip is the selection path.
- `interpolate_mode` is config-dependent but defaults to nearest. Bilinear would need `align_corners` parity checked before admission.
- PP-LCNetV3 has learnable reparameterization branches in source. DinoML can either lower branch sums literally or admit only pre-fused/reparameterized checkpoints after verifying weight format.
- BatchNorm appears in inference source; production lowering should fold Conv+BatchNorm when weights are frozen.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW dense tensors, static or bucketed dynamic spatial dimensions divisible/rounded to 32.
- Feature tuple extraction from backbone stages.
- `torch.cat(..., dim=1)` for channel concat.
- Elementwise add for feature fusion and branch sums.
- Elementwise multiply for SE gating.
- Scalar affine `scale * x + bias`, `0.2*x + 0.5`, and `torch.clamp`.
- Sigmoid, ReLU, Hardswish, Hardsigmoid.

Neural network primitives:

- Conv2d NCHW: 1x1, 3x3, 5x5 depthwise, grouped/depthwise with `groups=in_channels`, stride 1/2, padding `kernel//2`.
- ConvTranspose2d NCHW: kernel 2, stride 2, no explicit padding in head.
- BatchNorm2d inference.
- AdaptiveAvgPool2d output size 1 for SE.
- Resize/upsample NCHW by scale factors `2,4,8`, default nearest.

Preprocessing-coupled ops:

- BGR decode or caller-supplied BGR tensor.
- Resize preserving aspect ratio, rounded to nearest 32 with min side 32 and optional max side 4000.
- Rescale `1/255`, normalize, CHW conversion, target-size side channel.

Detection postprocess ops:

- Threshold probability map to bitmap.
- OpenCV `findContours(RETR_LIST, CHAIN_APPROX_SIMPLE)`.
- `minAreaRect` and `boxPoints`, point ordering.
- Polygon score via `fillPoly` mask and mean over prediction map.
- Polygon unclip using area/perimeter offset-distance and line intersections.
- Scale box vertices from prediction map to original image size; clamp and round.

Attention/generation/cache ops:

- Not applicable. There is no attention, tokenizer, language generation, or KV cache for the detection target.

## 5. Layer/block breakdown

PP-LCNetV3 backbone, with input `pixel_values: [B,3,H,W]`:

```text
stem:
  Conv2d(3 -> make_divisible(16*0.75,16)=16, k=3, s=2, p=1) -> BN -> Identity

each PP-LCNetV3 block:
  depthwise LearnableRepLayer(groups=C, k=3 or 5, stride=1/2)
    optional identity BN
    optional 1x1 ConvBN branch
    4 symmetric kxk ConvBN branches
    sum -> learnable affine -> hardswish affine activation unless stride==2
  optional SE:
    AdaptiveAvgPool2d(1) -> Conv1x1(C -> C/4) -> ReLU -> Conv1x1(C/4 -> C) -> Hardsigmoid -> multiply
  pointwise LearnableRepLayer(k=1, stride=1)
```

Feature maps consumed by detector for the default config are inferred as:

```text
stage2: [B, 48, H/4,  W/4]
stage3: [B, 96, H/8,  W/8]
stage4: [B,192, H/16, W/16]
stage5: [B,384, H/32, W/32]
```

Mobile detector projection and neck:

```text
feature_maps[i] -> Conv1x1(backbone_C[i] -> [12,18,42,360][i])

insert_conv per level:
  Conv2d(Ci -> 96, k=1, bias=False) -> SE-style residual block

top-down:
  fused[2] += interpolate(fused[3], scale=2)
  fused[1] += interpolate(fused[2], scale=2)
  fused[0] += interpolate(fused[1], scale=2)

input_conv per level:
  Conv2d(96 -> 24, k=3, p=1, bias=False) -> SE-style residual block

processed:
  p2 scale 1, p3 scale 2, p4 scale 4, p5 scale 8
  cat([p5,p4,p3,p2], dim=1) -> [B,96,H/4,W/4]
```

Segmentation head:

```text
ConvBNAct(96 -> 24, k=3, p=1)
ConvTranspose2d(24 -> 24, k=2, s=2) -> BN -> ReLU
ConvTranspose2d(24 -> 1, k=2, s=2)
Sigmoid -> probability map [B,1,H,W]
```

## 6. Attention requirements

No attention is required. There is no causal/noncausal attention, no masks, no packed/varlen metadata, no positional encoding, no KV cache, and no generation controller for this model family.

## 7. Position encoding and custom math

There is no position encoding. Custom math that matters for parity is SE gating and DB postprocess geometry.

```python
def mobile_rse_gate(x, conv1, conv2):
    gate = adaptive_avg_pool2d(x, 1)
    gate = conv2(relu(conv1(gate)))
    gate = clamp(0.2 * gate + 0.5, 0.0, 1.0)
    return x + x * gate
```

Postprocess unclip is not a neural op; it uses polygon area/perimeter to compute an offset distance:

```python
offset_distance = contour_area(polygon) * unclip_ratio / arc_length(polygon)
```

The rest is line-normal shifting and adjacent shifted-line intersections, with a near-parallel fallback.

## 8. Preprocessing and input packing

Processor ABI:

- Input images are grouped by original spatial shape for batched resizing.
- Resize uses `limit_type` rules (`max`, `min`, or `resize_long`), then rounds both spatial dimensions to multiples of 32 and clamps each to at least 32.
- Source defaults: `limit_side_len=960`, `limit_type="max"` in class; original Paddle inference config uses `resize_long=960`.
- `max_side_limit=4000` is enforced before 32-multiple rounding.
- Rescale/normalize runs before explicit BGR-to-RGB channel reorder in the current server image processor.
- Output `pixel_values` are NCHW and fed to the model.
- Output side-channel `target_sizes` stores original `(height,width)` per image and is mandatory for `post_process_object_detection`.

Postprocess ABI:

- Input: `predictions.last_hidden_state` with shape `[B,1,H_map,W_map]`.
- `threshold=0.3` creates a bitmap.
- Per image, contours are processed up to `max_candidates=1000`.
- A candidate is rejected when the first min box short side is `< min_size`, mean polygon score is below `box_threshold=0.6`, or the unclipped min box short side is `< min_size + 2`.
- Output per image: dict with `"boxes"` as `int16` quadrilaterals `[N,4,2]`, `"scores"` as fp32, and `"labels"` all zero for text.
- No NMS, class softmax, or background label handling exists.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern:

```text
Conv2d(..., bias optional) -> BatchNorm2d -> activation
```

Replacement:

```text
Conv2d(folded_weight, folded_bias) -> activation
```

Preconditions:

- Inference mode with frozen BN running mean/var, gamma/beta, eps.
- Preserve groups/depthwise settings.
- Fold each branch independently before any branch-sum rewrite.

Failure cases:

- Training mode, missing BN buffers, unfrozen mutable stats.

Parity test sketch:

- Random NCHW tensors across stride/group/kernel cases; compare source module vs folded Conv2d in fp32 and fp16 tolerances.

### Rewrite: PP-LCNetV3 LearnableRepLayer branch fusion

Source pattern:

```text
optional identity BN + optional 1x1 ConvBN + N kxk ConvBN branches -> sum -> learnable affine -> activation
```

Replacement:

```text
single kxk Conv2d with padded/fused 1x1 and identity kernels -> learnable affine -> activation
```

Preconditions:

- Same stride, groups, in/out channels compatible with identity branch.
- All ConvBN branches are frozen and use same output shape.
- 1x1 branch can be zero-padded into kxk center.
- Identity BN can be represented as grouped identity kernel for depthwise or normal identity for matching in/out channels.

Failure cases:

- Dynamic branch count not supported by the fusion pass, missing BN state, stride/output mismatch.

Parity test sketch:

- One block per kernel size 3/5, groups 1 and depthwise, stride 1/2; compare before/after feature maps.

### Rewrite: Head ConvTranspose2d k2s2 to upsample-conv or direct deconv provider

Source pattern:

```text
ConvTranspose2d(Cin -> Cout, kernel=2, stride=2, padding=0)
```

Replacement options:

- Prefer native ConvTranspose2d provider first for parity.
- Optional guarded rewrite to nearest upsample + Conv2d only if weight pattern proves equivalent; do not assume generic deconv equals resize-conv.

Preconditions:

- Fixed kernel/stride/padding/dilation/output_padding.
- Weight transform formally validated.

Failure cases:

- Arbitrary learned deconv weights; use direct transposed-conv lowering.

### Rewrite: Guarded NCHW-to-NHWC convolution island

Source pattern:

```text
NCHW Conv/BN/activation/elementwise island with no axis-sensitive external consumer
```

Replacement:

```text
NHWC/channel-last internal island with rewritten channel axis ops
```

Preconditions:

- Rewrite `cat(dim=1)` to NHWC channel axis.
- Rewrite AdaptiveAvgPool/SE channel broadcasts.
- Rewrite all interpolation and ConvTranspose layout contracts.
- Entry/exit transposes are either eliminated by adjacent islands or justified by measured speedup.

Failure cases:

- Postprocess assumes NCHW output map; processor emits NCHW; partial islands around concat/resize are easy to get wrong.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm + activation folding for all backbone/neck/head convs; this is mandatory for efficient inference.
- Depthwise-separable Conv2d kernels, including grouped depthwise 3x3/5x5 and pointwise 1x1.
- Nearest upsample + add in the neck, because it appears repeatedly and is bandwidth-sensitive.
- ConvTranspose2d k2s2 head path, because it gates full probability-map parity.

Medium priority:

- SE block fusion: global avg pool + two 1x1 convs + clamp/hardsigmoid + channel multiply.
- LearnableRepLayer branch fusion to reduce multiple ConvBN branches into one inference conv.
- Channel concat of four same-resolution feature maps; can be fused with following head conv if layout and memory planner support it.

Lower priority:

- GPU DB postprocess. First integration can keep OpenCV/CPU postprocess, but GPU contour/box extraction may matter for high-throughput batched OCR.
- NHWC/channel-last convolution islands. Useful only after NCHW parity and axis guards are in place.

## 11. Runtime staging plan

Stage 1: Admit preprocessed `pixel_values` only. Load config/weights and run PP-LCNetV3 backbone plus detector on NCHW tensors without owning image decode.

Stage 2: Implement/fold Conv2d, grouped/depthwise Conv2d, BatchNorm inference, activations, AdaptiveAvgPool2d, elementwise add/mul/clamp, nearest resize, concat, and ConvTranspose2d. Validate probability-map parity.

Stage 3: Add the processor ABI outside the compiled graph: BGR/HWC decode boundary, resize-to-32-multiple, normalize, NCHW packing, and `target_sizes` propagation.

Stage 4: Implement CPU postprocess parity using an OpenCV-compatible helper or a bounded internal equivalent. Return variable-count quadrilateral records.

Stage 5: Add graph rewrites: Conv+BN fold, LearnableRepLayer branch fusion, optional concat/head fusion.

Stage 6: Profile layout alternatives: NCHW baseline vs guarded NHWC islands for convolution-heavy regions.

## 12. Parity and validation plan

- Config parse test for safetensors config, including effective PP-LCNetV3 defaults omitted from `backbone_config`.
- Processor shape tests for small, square, tall, wide, and max-side-limited images; verify resized dimensions are multiples of 32 and `target_sizes` are original sizes.
- Unit parity for PP-LCNetV3 ConvLayer, LearnableRepLayer, SE, depthwise-separable block.
- Neck parity on synthetic feature maps with expected strides/channels.
- Head parity for `[B,96,H/4,W/4] -> [B,1,H,W]`.
- Full probability-map parity against Transformers for batch 1 and grouped batch inputs.
- Postprocess parity on synthetic maps: empty map, one rectangle, rotated quadrilateral, low-score rejection, max-candidate truncation, min-size rejection.
- End-to-end detection parity on a small image set using target sizes from the processor.
- Recommended tolerances: fp32 probability map `atol=1e-5, rtol=1e-4`; fp16 after folded conv path likely `atol=1e-2, rtol=1e-2` pending backend convolution precision.

## 13. Performance probes

- Processor throughput: decode/resize/normalize separately from model runtime.
- Backbone-only latency and throughput over resized side lengths 320, 640, 960, 1280, and max-side stress.
- Neck/head latency split, especially resize/add/concat and ConvTranspose2d.
- Batch-size sweep for homogeneous image shapes and mixed shape groups.
- NCHW vs NHWC convolution island benchmark after parity.
- Conv+BN folded vs unfused branch-sum backbone benchmark.
- Postprocess CPU time vs number of contours/candidates; include empty and dense text maps.
- End-to-end OCR detection requests/sec with and without CPU postprocess.

## 14. Skip/defer list

- Training losses, DB shrink/threshold map construction, OHEM, and metrics.
- Paddle static inference loader for `inference.pdiparams`; first target is HF safetensors.
- Text recognition, angle classification, and full OCR pipeline orchestration.
- GPU contour extraction and polygon unclip.
- Bilinear `interpolate_mode` until a checkpoint/config requires it.
- NHWC as default semantic translation; keep it as guarded optimization.
- General object-detection NMS/classification heads; this detector is single-class text with DB contour postprocess.

## 15. Final implementation checklist

- [ ] Parse `PPOCRV5MobileDetConfig` plus nested `PPLCNetV3Config` defaults.
- [ ] Load HF safetensors weights and preserve backbone/detector key mapping.
- [ ] Implement NCHW Conv2d, grouped/depthwise Conv2d, ConvTranspose2d, BatchNorm inference, AdaptiveAvgPool2d.
- [ ] Implement/fuse ReLU, Hardswish, Hardsigmoid, sigmoid, clamp, add, multiply.
- [ ] Implement nearest `interpolate(scale_factor=2/4/8)` and channel concat.
- [ ] Add strict input guards: NCHW, 3 channels, spatial dimensions >=32 and compatible with backbone/neck strides.
- [ ] Add processor-side resize/normalize/target-size ABI or require callers to provide equivalent `pixel_values` plus `target_sizes`.
- [ ] Implement CPU DB postprocess returning variable-count `[N,4,2]` boxes, scores, labels.
- [ ] Add Conv+BN fold rewrite.
- [ ] Add LearnableRepLayer branch-fusion rewrite after branch parity.
- [ ] Add probability-map parity tests vs Transformers.
- [ ] Add postprocess parity tests vs `PPOCRV5ServerDetImageProcessor.post_process_object_detection`.
- [ ] Benchmark processor, model body, head, and postprocess separately.

## Gated gaps for DinoML

- `Conv2d`/`ConvTranspose2d`/`BatchNorm2d`/`AdaptiveAvgPool2d` are not currently complete in the tracked v2 checklist, so this family is blocked before model-body parity.
- NCHW layout must be preserved initially; NHWC/channel-last needs explicit axis rewrites for concat, SE broadcasts, resize, ConvTranspose, and output-map ABI.
- Postprocess is required for end-to-end detection parity and depends on OpenCV-style contour/min-area-rect/polygon operations plus variable-length outputs.
- The processor ABI has a channel-order trap: source inputs are BGR/HWC-style, while the model graph consumes NCHW `pixel_values`; DinoML should either own that preprocessing exactly or require preprocessed tensors with documented guards.
