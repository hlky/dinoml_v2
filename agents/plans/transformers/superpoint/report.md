# Transformers Audit: SuperPoint

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: magic-leap-community/superpoint, with mirror/wrapper configs sampled
Config source: local SuperPointConfig plus Hugging Face raw config/preprocessor files
Primary runtime target: SuperPointForKeypointDetection keypoints + descriptors
Assumptions: inference-only, CUDA first, preserve source NCHW semantics before optional layout rewrites
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/superpoint/configuration_superpoint.py`
- `X:/H/transformers/src/transformers/models/superpoint/modeling_superpoint.py`
- `X:/H/transformers/src/transformers/models/superpoint/image_processing_superpoint.py`
- `X:/H/transformers/src/transformers/models/superpoint/image_processing_pil_superpoint.py`
- `X:/H/transformers/src/transformers/models/superpoint/convert_superpoint_to_pytorch.py`
- `X:/H/transformers/tests/models/superpoint/test_modeling_superpoint.py`
- `X:/H/transformers/tests/models/superpoint/test_image_processing_superpoint.py`

Representative configs:

- [magic-leap-community/superpoint config](https://huggingface.co/magic-leap-community/superpoint/raw/main/config.json)
  and [preprocessor](https://huggingface.co/magic-leap-community/superpoint/raw/main/preprocessor_config.json).
- [stevenbucaille/superpoint config](https://huggingface.co/stevenbucaille/superpoint/raw/main/config.json)
  and [preprocessor](https://huggingface.co/stevenbucaille/superpoint/raw/main/preprocessor_config.json).
- [ETH-CVG/lightglue_superpoint config](https://huggingface.co/ETH-CVG/lightglue_superpoint/raw/main/config.json)
  and [preprocessor](https://huggingface.co/ETH-CVG/lightglue_superpoint/raw/main/preprocessor_config.json),
  used only for nested SuperPoint detector context.
- [stevenbucaille/lightglue_superpoint config](https://huggingface.co/stevenbucaille/lightglue_superpoint/raw/main/config.json)
  and [preprocessor](https://huggingface.co/stevenbucaille/lightglue_superpoint/raw/main/preprocessor_config.json),
  used only for nested/default SuperPoint detector context.
- [AXERA-TECH/superpoint config](https://huggingface.co/AXERA-TECH/superpoint/raw/main/config.json)
  is `model_type="ONNX"` and out of scope for native Transformers SuperPoint.

Missing files or assumptions: no remote code is needed for native SuperPoint. The canonical checkpoint has an
`architectures` field of `SuperPointModel`, but the inspected source exports `SuperPointForKeypointDetection`; treat the
architecture field as historical metadata, not a separate class requirement.

## 2. High-level architecture

SuperPoint is not a Transformer. It is a fully convolutional vision feature extractor with two heads:

```text
CPU/image pipeline -> pixel_values [B,3,H,W] -> first-channel slice [B,1,H,W]
  -> Conv encoder [B,128,H/8,W/8]
  -> detector head -> dense score map [B,H,W] -> NMS/threshold/top-k -> keypoints
  -> descriptor head -> descriptor map [B,256,H/8,W/8] -> grid_sample at keypoints
  -> padded variable-length keypoints/scores/descriptors/mask
  -> optional postprocess to original absolute image coordinates
```

Stage decomposition:

- CPU/data pipeline: image decode, optional resize, optional RGB-to-gray, rescale. The processor still emits
  3-channel `pixel_values`.
- GPU graph candidate: encoder convolutions and both convolutional heads.
- Structured-output postprocess: NMS, thresholding, variable-length keypoint selection, descriptor bilinear sampling,
  padding/mask, relative coordinate conversion.
- End-to-end postprocess: processor `post_process_keypoint_detection` multiplies relative `(x,y)` by original
  `(width,height)`, casts to `int32`, and filters padded rows using `mask`.

## 3. Important config dimensions

| Field | Default / canonical value | Operator impact |
| --- | ---: | --- |
| `encoder_hidden_sizes` | `[64, 64, 128, 128]` | Four Conv2d blocks; first three downsample by MaxPool2d. |
| `decoder_hidden_size` | `256` | Hidden channels for both detector and descriptor heads. |
| `keypoint_decoder_dim` | `65` | Detector logits: 64 cell positions plus dustbin. Source softmax drops dustbin. |
| `descriptor_decoder_dim` | `256` | Descriptor width per keypoint. |
| `keypoint_threshold` | `0.005` | Value-dependent `nonzero` selection. |
| `max_keypoints` | `-1` | `-1` keeps all thresholded/NMS points; nonnegative enables `topk`. |
| `nms_radius` | `4` | NMS implemented with max-pool kernel `2*r+1`, stride 1, padding `r`. |
| `border_removal_distance` | `4` | Lower-edge border filtering is active; upper-edge behavior follows source caveat below. |
| Processor `size` | `480x640` | Default input size; not a model config constraint. |
| Processor rescale | `1/255` | No mean/std normalization; `do_normalize=None`. |

Checkpoint sweep:

| Repo | Native family? | Detector config variation | Processor variation | Admission note |
| --- | --- | --- | --- | --- |
| `magic-leap-community/superpoint` | yes | canonical full defaults | no grayscale flag in saved preprocessor | first target |
| `stevenbucaille/superpoint` | yes | same as canonical | same as canonical | useful mirror |
| `ETH-CVG/lightglue_superpoint` | nested in LightGlue | full nested SuperPoint defaults | `do_grayscale=true`, LightGlue processor | compose after LightGlue audit |
| `stevenbucaille/lightglue_superpoint` | nested in LightGlue | only `model_type` saved; defaults fill the rest | `do_grayscale=true`, LightGlue processor | defaults-sensitive wrapper |
| `AXERA-TECH/superpoint` | no | ONNX/NPU export config | none found | reject for this native audit |

## 3a. Family variation traps

- The source expects NCHW tensors and immediately slices `pixel_values[:, 0, :, :]` to one channel.
- Processor grayscale is optional in native SuperPoint configs; source still uses only channel 0. If callers pass RGB
  without grayscale conversion, current behavior is "red channel only", not NTSC grayscale.
- Torchvision backend applies grayscale before resize; PIL backend resizes/rescales before grayscale. Treat exact
  processor parity as data-pipeline work.
- `keypoint_decoder_dim` must be `65` for the hard-coded score reshape into 64 sub-cell logits after dropping dustbin.
- Inputs should be multiples of 8 for full source intent. Non-multiple spatial sizes will be downsampled by pooling and
  produce score maps at `floor(H/8)*8` and `floor(W/8)*8`, while final relative scaling divides by original `H,W`.
- `_extract_keypoints` receives a full-resolution score map but passes `height * 8, width * 8` into border removal.
  Source-compatible lowering should preserve this unless a deliberate bug-fix mode is introduced.
- `torch.topk` tie ordering is not stable; upstream marks batching equivalence flaky for this reason.
- Batched output shape is the maximum keypoint count in the batch; this is a value-dependent shape/mask contract.
- LightGlue SuperPoint wrappers are composite `model_type="lightglue"` targets. The nested detector can use this report,
  but LightGlue matching/head behavior is owned by a separate family audit.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW slicing `pixel_values[:, 0:1, :, :]`.
- NCHW Conv2d, MaxPool2d, ReLU.
- `permute(0,2,3,1)`, reshape to `[B,Hc,Wc,8,8]`, `permute(0,1,3,2,4)`, reshape to `[B,Hc*8,Wc*8]`.
- Per-image loops over batch after encoder output.
- Zero allocation and dynamic padding to `[B,Kmax,2]`, `[B,Kmax]`, `[B,Kmax,256]`, `[B,Kmax]`.

Neural network primitives:

- Encoder: eight `3x3 stride=1 padding=1` Conv2d with bias, ReLU after every conv.
- Pooling: three `2x2 stride=2` MaxPool2d ops after encoder blocks 1-3.
- Detector head: Conv2d `128 -> 256`, `3x3`; ReLU; Conv2d `256 -> 65`, `1x1`.
- Descriptor head: Conv2d `128 -> 256`, `3x3`; ReLU; Conv2d `256 -> 256`, `1x1`.
- Channelwise L2 normalize descriptor map over `dim=1`.

Postprocess and structured-output ops:

- Softmax over detector channel axis `dim=1`; drop final dustbin channel.
- NMS via max-pool equality masks and two suppression iterations.
- Score threshold, `nonzero`, gather score values, border mask, optional `topk`.
- Coordinate flip from `(y,x)` to `(x,y)` and cast to score dtype.
- Descriptor sampling coordinate transform and `grid_sample(..., mode="bilinear", align_corners=True)`.
- L2 normalize sampled descriptors over descriptor channel axis.
- Final relative coordinate scaling by `[width,height]`; postprocessor multiplies by target `[width,height]`, casts
  keypoints to `int32`, and filters by `nonzero(mask)`.

Attention, generation, cache, RoPE, tokenizer, quantization, distributed, and packed/varlen sequence ops: not applicable.

## 5. Layer/block breakdown

Encoder for input `[B,1,H,W]`:

```text
Block 1: Conv2d(1 -> 64, 3x3, pad=1) -> ReLU -> Conv2d(64 -> 64, 3x3, pad=1) -> ReLU -> MaxPool2d(2)
Block 2: Conv2d(64 -> 64, 3x3, pad=1) -> ReLU -> Conv2d(64 -> 64, 3x3, pad=1) -> ReLU -> MaxPool2d(2)
Block 3: Conv2d(64 -> 128, 3x3, pad=1) -> ReLU -> Conv2d(128 -> 128, 3x3, pad=1) -> ReLU -> MaxPool2d(2)
Block 4: Conv2d(128 -> 128, 3x3, pad=1) -> ReLU -> Conv2d(128 -> 128, 3x3, pad=1) -> ReLU
Output: [B,128,floor(H/8),floor(W/8)]
```

Detector head, run per encoded image in current source:

```text
encoded [1,128,Hc,Wc]
  -> Conv2d(128 -> 256, 3x3, pad=1) -> ReLU
  -> Conv2d(256 -> 65, 1x1)
  -> softmax(dim=1)[:, :-1]
  -> pixel shuffle by reshape/permute to [1,Hc*8,Wc*8]
  -> simple_nms
  -> threshold/nonzero/border/topk
  -> keypoints [K,2] in x,y pixel order, scores [K]
```

Descriptor head, run per encoded image in current source:

```text
encoded [1,128,Hc,Wc]
  -> Conv2d(128 -> 256, 3x3, pad=1) -> ReLU
  -> Conv2d(256 -> 256, 1x1)
  -> L2Normalize(dim=1)
  -> coordinate normalization for keypoints
  -> grid_sample to [1,256,1,K]
  -> reshape [1,256,K] -> L2Normalize(dim=1) -> transpose to [K,256]
```

## 6. Attention requirements

No attention is required. There is no self-attention, cross-attention, KV cache, SDPA/FlashAttention path, causal mask,
relative bias, RoPE, packed sequence, or generation controller. The only "decoder" modules are convolutional detector
and descriptor heads.

## 7. Position encoding and custom math

No learned or sinusoidal position encoding is used. Spatial position enters through convolution locality, the 8x
detector cell expansion, and descriptor sampling coordinates.

Key source math to reproduce:

```python
scores = softmax(detector_logits, dim=1)[:, :-1]
scores = scores.permute(0, 2, 3, 1).reshape(B, Hc, Wc, 8, 8)
scores = scores.permute(0, 1, 3, 2, 4).reshape(B, Hc * 8, Wc * 8)
```

```python
xy = keypoints - scale / 2 + 0.5
xy = xy / [Wc * scale - scale / 2 - 0.5, Hc * scale - scale / 2 - 0.5]
grid = xy * 2 - 1
sampled = grid_sample(descriptor_map, grid.view(B, 1, K, 2), align_corners=True)
```

NMS is source-specific: equality to max-pooled scores, two rounds of suppression-mask expansion, and zeroing non-max
positions.

## 8. Preprocessing and input packing

Native processor defaults:

- Input images are resized to height `480`, width `640` by default.
- Values are rescaled by `1/255`.
- No mean/std normalization is applied.
- Processor output is `pixel_values` in channels-first `[B,3,H,W]`.
- `do_grayscale` exists but is not set in canonical native preprocessor config; LightGlue wrappers set it true.

Runtime graph boundary:

- DinoML first integration can accept preprocessed `[B,3,H,W]` and perform source-compatible first-channel extraction.
- Exact PIL/torchvision image resize and grayscale ordering belongs in CPU/data-pipeline parity unless DinoML later owns
  preprocessing.

Postprocess contract:

- Model returns relative keypoints padded to batch `Kmax`, scores, descriptors, and mask.
- Processor postprocess requires `target_sizes` shaped `[B,2]` in `(height,width)`, flips to `(width,height)`, multiplies
  relative keypoints, casts to `int32`, and removes padded rows using `mask`.
- There is no box NMS or class-label output; NMS is local keypoint score suppression before descriptor sampling.

## 9. Graph rewrite / lowering opportunities

### Rewrite: detector 64-channel cell logits -> pixel shuffle

Source pattern:

```text
softmax([B,65,Hc,Wc], dim=1)[:,0:64] -> permute/reshape/permute/reshape -> [B,Hc*8,Wc*8]
```

Replacement:

```text
channel-softmax+dustbin-drop -> depth-to-space(block=8) with source channel order
```

Preconditions: `keypoint_decoder_dim == 65`, NCHW logits, channel axis is detector class axis, block size 8, no channel
layout rewrite unless the softmax axis and depth-to-space order are rewritten together.

Failure cases: non-65 detector dim, NHWC tensors without axis rewrite, non-contiguous source assumptions, or desire to
reuse a generic pixel-shuffle with different channel ordering.

Parity sketch: compare logits-to-score-map against PyTorch for random `[B,65,Hc,Wc]`, including ties and dustbin-heavy
cases.

### Rewrite: static Conv2d heads/encoder to provider conv or im2col+GEMM

Preconditions:

- `stride=1`, `padding=1`, `dilation=1`, `groups=1` for 3x3 convs.
- `stride=1`, `padding=0`, `dilation=1`, `groups=1` for 1x1 heads.
- NCHW source semantics preserved, or a complete NHWC region rewrite covers conv, ReLU, pooling, softmax channel axis,
  descriptor normalization axis, and all consumers.

Replacement: cuDNN/oneDNN Conv2d preferred; 1x1 heads can lower to GEMM under channel/layout guards.

Failure cases: dynamic layouts crossing into score postprocess, non-contiguous views, or processor/model disagreement on
spatial size divisibility.

### Rewrite: value-dependent keypoint extraction as structured postprocess

Source pattern:

```text
score_map -> NMS -> threshold -> nonzero -> gather -> border mask -> optional topk
```

Replacement: keep as a detector postprocess operator family rather than decomposing into arbitrary public `nonzero` and
ragged gather for first integration.

Preconditions: one classless score map per image, fixed NMS radius, threshold scalar, optional bounded `max_keypoints`,
output padded with mask.

Failure cases: `max_keypoints=-1` with caller requiring static allocation, score ties requiring deterministic ordering,
or source bug-fix mode for far-border removal.

### Layout rewrite: NCHW encoder/head island to NHWC

Candidate safe island: first-channel slice through convolution/ReLU/pooling heads can be converted to NHWC if all Conv2d
weights are transformed and all channel-axis ops are rewritten.

Required axis rewrites:

- Softmax `dim=1` becomes channel-last `dim=-1`.
- Descriptor normalize `dim=1` becomes `dim=-1` before grid sampling or requires conversion back.
- Score reshape/permute must be replaced by channel-last depth-to-space logic.
- MaxPool2d NMS currently sees `[B,1,H,W]` implicitly through 3D input support in PyTorch max-pool helper; a layout pass
  should guard NMS/postprocess as no-layout-translation unless implemented deliberately.

No-layout-translation boundaries: `simple_nms`, `torch.nonzero` extraction, descriptor `grid_sample`, final padding/mask
assembly, and processor postprocess should initially remain source-axis faithful.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + ReLU blocks, especially encoder 3x3 pairs. This is most of dense compute and has static topology.
- Detector head softmax + dustbin drop + pixel shuffle. It creates a large score map and is axis-sensitive.
- NMS + threshold + compact keypoints as one structured postprocess kernel. This avoids exposing general dynamic
  `nonzero`/ragged gather too early.

Medium priority:

- Descriptor Conv2d + L2 normalization over channels.
- Descriptor coordinate transform + bilinear sampling + final L2 normalization.
- Batched padding/mask assembly with bounded `max_keypoints` mode.

Lower priority:

- CPU image preprocessing acceleration; useful for throughput but outside the neural graph.
- Exact top-k tie-stable behavior. Source itself does not guarantee stable ties, so this is mostly a reproducibility guard.

## 11. Runtime staging plan

Stage 1: config/weights and dense encoder-head graph.

- Parse SuperPoint config, load conv weights, run first-channel slice, encoder, detector logits, descriptor map.
- Stub keypoint extraction with exported dense score/descriptor maps if needed.

Stage 2: source-compatible detector postprocess.

- Implement detector softmax/pixel shuffle, simple NMS, threshold, border mask, and optional top-k.
- Prefer bounded `max_keypoints` admission first; for canonical `-1`, allocate from a configured upper bound or route to
  CPU/postprocess until value-dependent output shapes are ready.

Stage 3: descriptor sampling.

- Implement keypoint coordinate transform and bilinear `grid_sample(align_corners=True)` over descriptor map, then
  normalize and transpose to `[K,256]`.

Stage 4: batched variable-length ABI.

- Produce padded keypoints/scores/descriptors/mask with `Kmax` per batch. Wire postprocess to original target sizes.

Stage 5: layout/fusion optimization.

- Add guarded NCHW-to-NHWC conv island and depth-to-space rewrite after source parity is solid.

Can be stubbed initially: PIL/torchvision preprocessing, LightGlue wrapper composition, training labels/loss,
ONNX/NPU exports, and `output_hidden_states` beyond basic debug support.

## 12. Parity and validation plan

- Config/default tests: canonical and synthetic smaller config from upstream tests (`[32,32,64,64]`, decoder `128`).
- Encoder parity: random `[B,3,H,W]`, compare first-channel extraction and block outputs, fp32 tolerance `1e-5` to `1e-4`.
- Detector logits-to-score-map parity: compare softmax/drop/reshape and NMS on random and tie-heavy score maps.
- Keypoint extraction parity: threshold-only, NMS+threshold, border mask, and `max_keypoints` cases. Include equal-score
  tie cases but do not require stronger determinism than source.
- Descriptor sampling parity: fixed keypoints and descriptor maps, compare `grid_sample` with `align_corners=True`.
- End-to-end fixture parity: use upstream COCO sample expectation as a smoke test: canonical integration reports 568 and
  830 keypoints for two images, padded to 830, descriptors `[2,830,256]`.
- Postprocess parity: relative-to-absolute conversion, `int32` cast, mask filtering, and `(height,width)` target-size
  validation.
- Tolerances: fp32 `1e-4` for scores/descriptors; fp16 should be introduced only after dense conv and softmax tolerance
  is characterized, likely `1e-2` for descriptor values.

## 13. Performance probes

- Processor throughput: image decode/resize/grayscale/rescale, separate PIL and torchvision paths.
- Encoder+head dense throughput: batch-size sweep at `480x640`, plus smaller synthetic sizes.
- Detector postprocess throughput: NMS radius sweep, threshold density sweep, `max_keypoints=-1` versus bounded top-k.
- Descriptor sampling throughput: keypoint count sweep from 0 to dense high-count cases.
- End-to-end keypoint throughput: images/sec with and without CPU postprocess.
- Layout experiment: NCHW cuDNN versus guarded NHWC conv island, including conversion overhead and postprocess boundary
  copies.
- Memory probes: temporary score map `[B,H,W]`, descriptor map `[B,256,H/8,W/8]`, dynamic keypoint buffers, padded output.

## 14. Skip/defer list

- Training and labels: source raises if labels are provided.
- Attention, generation, caches, tokenizers: not part of this family.
- LightGlue matching: compose a separate LightGlue audit; only the nested SuperPoint detector is covered here.
- ONNX/NPU export repos: reject or route to import/export tooling, not native Transformers SuperPoint.
- General public `nonzero`/ragged tensor API: prefer a bounded SuperPoint postprocess op first.
- Full NHWC translation: defer until source-compatible NCHW parity exists.
- Exact PIL/torchvision preprocessing inside GPU runtime: keep in CPU/data pipeline initially.

## 15. Final implementation checklist

- [ ] Parse `SuperPointConfig` and reject unsupported `keypoint_decoder_dim != 65`.
- [ ] Load conv weights and preserve source parameter names.
- [ ] Implement first-channel extraction from `[B,3,H,W]` to `[B,1,H,W]`.
- [ ] Implement NCHW Conv2d/ReLU/MaxPool2d encoder path.
- [ ] Implement detector head Conv2d/ReLU/Conv2d.
- [ ] Implement channel softmax, dustbin drop, and 8x pixel-shuffle score expansion.
- [ ] Implement source-compatible simple NMS.
- [ ] Implement threshold, `nonzero`, border filter, and optional top-k as bounded detector postprocess.
- [ ] Implement descriptor head Conv2d/ReLU/Conv2d and L2 normalization.
- [ ] Implement descriptor coordinate transform and bilinear `grid_sample(align_corners=True)`.
- [ ] Implement padded variable-length output ABI with mask.
- [ ] Implement postprocess from relative to absolute keypoints using `target_sizes`.
- [ ] Add parity tests for each postprocess substep and end-to-end fixture outputs.
- [ ] Add performance probes for dense graph, NMS/extraction, and descriptor sampling.
- [ ] Add guarded NHWC/layout rewrite only after NCHW parity is proven.
