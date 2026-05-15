# YOLOS Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary inference target: hustvl/yolos-small, YolosForObjectDetection.
  Additional sweep: hustvl/yolos-tiny, hustvl/yolos-base,
  hustvl/yolos-small-300, hustvl/yolos-small-dwr,
  hf-internal-testing/tiny-random-YolosForObjectDetection.

Config source:
  https://huggingface.co/hustvl/yolos-small/raw/main/config.json
  https://huggingface.co/hustvl/yolos-small/raw/main/preprocessor_config.json
  Same raw paths for the additional sweep checkpoints above.

Source files inspected:
  transformers/src/transformers/models/yolos/configuration_yolos.py
  transformers/src/transformers/models/yolos/modeling_yolos.py
  transformers/src/transformers/models/yolos/modular_yolos.py
  transformers/src/transformers/models/yolos/image_processing_yolos.py
  transformers/src/transformers/models/yolos/image_processing_pil_yolos.py
  transformers/src/transformers/models/yolos/convert_yolos_to_pytorch.py

Any missing files or assumptions:
  No remote code is required for standard YOLOS. `image_processing_yolos.py`
  and `image_processing_pil_yolos.py` are generated from `modular_yolos.py`;
  future source edits should start from the modular file. This report targets
  CUDA inference for object detection from preprocessed image tensors first.
  CPU/PIL/torchvision preprocessing and postprocessing may remain outside the
  compiled graph initially. Training losses, Hungarian matching, annotation
  conversion, and panoptic/semantic/instance segmentation postprocess stubs are
  deferred.
```

## 2. High-level architecture

YOLOS is a ViT-style object detector. It patchifies the image with a
non-overlapping Conv2d, prepends a learned CLS token, appends learned detection
tokens, adds interpolated absolute position embeddings, runs a bidirectional
ViT encoder, then applies class and box MLP heads to the final detection-token
states.

```text
CPU image preprocessing
  -> pixel_values [B,3,H,W]
  -> patch Conv2d + flatten
  -> CLS + patch tokens + detection tokens + interpolated positions
  -> ViT encoder with optional mid-layer position embeddings
  -> final LayerNorm
  -> suffix detection-token slice
  -> class logits + normalized boxes
  -> postprocess to scores/labels/absolute xyxy boxes
```

Stage decomposition:

- CPU/data pipeline: decode, resize with aspect ratio, rescale, normalize, pad
  batch to common `H,W`, produce `pixel_values [B,3,H,W]` and `pixel_mask
  [B,H,W]`.
- Patch/embedding stage: source NCHW Conv2d with `kernel_size=stride=patch_size`;
  flatten spatial patches to `[B,N,Hid]`, concatenate CLS and detection tokens,
  interpolate initial position table to actual `H,W`, then add/dropout.
- Encoder stage: repeated noncausal ViT blocks. If
  `use_mid_position_embeddings=True`, a separate mid-layer position table is
  interpolated to actual `H,W` and added after every block except the last.
- Detection heads: select the last `num_detection_tokens` sequence positions,
  then run independent 3-layer MLPs for class logits and box coordinates.
- Postprocessing: softmax, ignore the final no-object class for score/label
  selection, threshold, convert `cxcywh` to `xyxy`, and scale by target image
  sizes. This is required for end-to-end detection parity.

The model class does not consume `pixel_mask`; padding only affects actual
image values and the processor/postprocess contract. That differs from DETR and
is important for graph inputs.

## 3. Important config dimensions

Worked example: `hustvl/yolos-small`.

| Field | Value | Source |
| --- | ---: | --- |
| primary task | object detection | config architecture/source |
| `model_type` | `yolos` | config.json |
| architecture | `YolosForObjectDetection` | config.json |
| `hidden_size` | 384 | config.json |
| `num_hidden_layers` | 12 | config.json |
| `num_attention_heads` | 6 | config.json |
| head dim | 64 | inferred `hidden_size / heads` |
| `intermediate_size` | 1536 | config.json |
| `hidden_act` | GELU | config.json |
| `image_size` | `[512,864]` | config.json |
| `patch_size` | 16 | config.json |
| config patch grid | `32 x 54 = 1728` | inferred from config |
| config sequence length | `1 + 1728 + 100 = 1829` | inferred/source formula |
| `num_detection_tokens` | 100 | config.json |
| `num_channels` | 3 | config.json |
| `qkv_bias` | true | config.json |
| `use_mid_position_embeddings` | true | config.json |
| class logits | `num_labels + 1` | source; extra final no-object class |
| COCO label ids | 91 `id2label` entries | config.json |
| preprocessor resize | shortest edge 800, longest edge 1333 | preprocessor_config |
| resize divisibility | output H/W floored to multiples of 16 | source |
| image mean/std | ImageNet `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | preprocessor_config |
| padding | source default `do_pad=True`, emits `pixel_mask` | image processor source |

Representative checkpoint sweep:

| Checkpoint | H | I | layers | heads | patch | config image | config seq | mid-pos | processor resize |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `hf-internal-testing/tiny-random-YolosForObjectDetection` | 32 | 37 | 5 | 4 | 2 | `[30,30]` | 236 | true | `[30,30]` legacy |
| `hustvl/yolos-tiny` | 192 | 768 | 12 | 3 | 16 | `[800,1333]` | 4251 | false | shortest 512, longest 1333 |
| `hustvl/yolos-small` | 384 | 1536 | 12 | 6 | 16 | `[512,864]` | 1829 | true | shortest 800, longest 1333 |
| `hustvl/yolos-small-300` | 384 | 1536 | 12 | 6 | 16 | `[512,864]` | 1829 | true | shortest 800, longest 1333 |
| `hustvl/yolos-small-dwr` | 330 | 1320 | 14 | 6 | 16 | `[512,864]` | 1829 | true | legacy `size=800,max_size=1333` |
| `hustvl/yolos-base` | 768 | 3072 | 12 | 12 | 16 | `[800,1344]` | 4301 | true | shortest 800, longest 1333 |

Notes:

- The config `image_size` defines the learned initial and mid position table
  shapes at load time. Runtime images may use different padded `H,W`; source
  interpolates patch-position slices every forward.
- `hustvl/yolos-small-dwr` is operator-significant because
  `hidden_size=330`, `num_heads=6`, so `head_dim=55`. DinoML attention and GEMM
  paths cannot assume `head_dim=64`.
- The current `YolosConfig` source default is base-like:
  `hidden_size=768`, `image_size=(512,864)`, `patch_size=16`,
  `num_detection_tokens=100`, `use_mid_position_embeddings=True`. Older configs
  may omit newer processor fields such as `image_processor_type`, but the
  current processor supplies defaults.

## 3a. Family variation traps

- Source model input layout is NCHW `pixel_values [B,C,H,W]`. Treat NHWC as a
  guarded internal optimization, not the source semantic contract.
- The processor emits `pixel_mask`, but `YolosModel.forward` only accepts and
  uses `pixel_values`. Do not thread DETR-style masks into attention for YOLOS
  parity.
- The processor resizes with aspect ratio and floors both resized dimensions to
  a multiple of 16. The model also works on padded runtime sizes that differ
  from `config.image_size` because it interpolates position tables.
- The initial position table is split as CLS / patch / detection suffix. Only
  the patch region is bicubic-interpolated; CLS and detection positions are not.
- Mid-position embeddings, when enabled, have shape
  `[num_hidden_layers - 1, 1, config_seq, hidden]`; they are interpolated once
  per forward and added after blocks `0..N-2`.
- Detection tokens are appended after patch tokens. Heads select
  `sequence_output[:, -num_detection_tokens:, :]`; any rewrite that changes
  token ordering must preserve the suffix contract.
- Attention is encoder-style, noncausal, and unmasked. There is no decoder
  cross-attention, no KV cache, no query/key/value head asymmetry, no RoPE, and
  no ALiBi.
- `hidden_size` is divisible by `num_attention_heads`, but the resulting
  `head_dim` is not always 64 (`small-dwr` uses 55).
- Classification and box heads are copied from DETR MLP heads but are applied
  to YOLOS detection-token states. Class output is `num_labels + 1`, where the
  final class is no-object and is excluded during postprocess scoring.
- No NMS is present in source postprocessing. Adding NMS would break parity.
- `YolosModel` has an optional CLS pooler, but `YolosForObjectDetection`
  constructs it with `add_pooling_layer=False`; pooler support is optional for
  the object-detection target.
- Training-only annotation conversion supports COCO detection and panoptic
  labels, but segmentation postprocess methods are explicit
  `NotImplementedError` stubs in YOLOS.
- Axis-sensitive layout traps:
  - Patch Conv2d consumes `[B,C,H,W]`; source flatten order is NCHW spatial
    row-major via `.flatten(2).transpose(1,2)`.
  - Position interpolation reshapes patch positions to
    `[1, hidden, patch_height, patch_width]`, interpolates over H/W, then
    flattens back to patch-token order.
  - Processor padding/masks are `[B,H,W]`; channel layout translation must not
    rewrite those axes.
  - Postprocess target sizes are `(height,width)`, but box scale order is
    `[width,height,width,height]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor validation for `pixel_values [B,3,H,W]`.
- Non-overlapping patch Conv2d:
  `Conv2d(num_channels -> hidden_size, kernel=patch, stride=patch, bias=True)`.
- Spatial flatten and transpose:
  `[B,Hid,Hp,Wp] -> [B,Hp*Wp,Hid]`.
- Learned parameter expand for CLS `[1,1,Hid] -> [B,1,Hid]` and detection
  tokens `[1,D,Hid] -> [B,D,Hid]`.
- Concatenate token sequence in exact order `[CLS, patches, detection]`.
- Position table slicing, transpose, view, bicubic interpolate,
  flatten/transpose, and concatenate.
- Optional mid-position table interpolation with leading layer dimension.
- Final detection suffix slice `[:, -D:, :]`.
- Postprocess boolean thresholding and variable-size per-image result records.

Neural network primitives:

- Linear with bias for Q/K/V/O projections, MLPs, pooler, and heads.
- LayerNorm over `hidden_size` with `eps=1e-12`.
- GELU from `ACT2FN` for transformer MLP.
- ReLU in detection class/box 3-layer MLP heads.
- Sigmoid for normalized `pred_boxes`.
- Tanh pooler activation, optional for `YolosModel` but not required for
  `YolosForObjectDetection`.
- Dropout appears in source but is zero in inspected production configs and
  disabled at inference.

Attention primitives:

- Noncausal self-attention over the whole token sequence.
- MHA only: `num_key_value_heads` is not present; no MQA/GQA.
- No attention mask in `YolosSelfAttention.forward`; source passes `None`.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` with eager fallback:
  `matmul(q,k^T) * scale -> softmax -> dropout -> matmul(weights,v)`.

Position/preprocessing-coupled ops:

- Bicubic interpolation with `align_corners=False` for learned patch position
  embeddings.
- Processor resize using YOLOS aspect-ratio rule and `mod_size=16`.
- Rescale, ImageNet normalize, right/bottom padding, and `pixel_mask`
  generation in CPU/data pipeline.

Detection/postprocessing ops:

- Class head per detection token:
  `Linear(H -> H) -> ReLU -> Linear(H -> H) -> ReLU -> Linear(H -> num_labels+1)`.
- Box head:
  `Linear(H -> H) -> ReLU -> Linear(H -> H) -> ReLU -> Linear(H -> 4) -> Sigmoid`.
- Object detection postprocess:
  softmax over classes, max over all but final no-object class, threshold,
  `cxcywh -> xyxy`, target-size scaling.
- No NMS, top-k, ROIAlign, or segmentation mask postprocess is required for the
  primary YOLOS object-detection target.

Training/loss deferred ops:

- Hungarian matching, classification/L1/GIoU losses, auxiliary loss wiring, and
  COCO annotation conversion are only used when `labels` are provided.

## 5. Layer/block breakdown

Processor output:

```text
pixel_values: [B,3,Hpad,Wpad] float, NCHW, normalized
pixel_mask:   [B,Hpad,Wpad] int64, 1 for valid pixels and 0 for padding
```

`pixel_mask` is part of processor output but not consumed by `YolosModel`.

Embedding path:

```text
patch = Conv2d(3 -> Hdim, kernel=P, stride=P)(pixel_values)
patch = patch.flatten(2).transpose(1,2)                 # [B,Hp*Wp,Hdim]
cls = cls_token.expand(B,1,Hdim)
det = detection_tokens.expand(B,D,Hdim)
x = concat(cls, patch, det, dim=1)                      # [B,1+Hp*Wp+D,Hdim]
pos = interpolate_initial_position_table(Hpad,Wpad)
x = dropout(x + pos)
```

Initial position interpolation:

```text
pos [1, 1+Ncfg+D, Hdim]
cls_pos = pos[:, 0:1, :]
patch_pos = pos[:, 1:-D, :].transpose(1,2).view(1,Hdim,Hcfg/P,Wcfg/P)
patch_pos = bicubic_interpolate(patch_pos, size=(Hpad/P,Wpad/P), align_corners=False)
patch_pos = patch_pos.flatten(2).transpose(1,2)
det_pos = pos[:, -D:, :]
interpolated = concat(cls_pos, patch_pos, det_pos, dim=1)
```

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
y = LayerNorm(x)
q = Linear(H -> H, bias=qkv_bias)(y).view(B,S,A,Dh).transpose(1,2)
k = Linear(H -> H, bias=qkv_bias)(y).view(B,S,A,Dh).transpose(1,2)
v = Linear(H -> H, bias=qkv_bias)(y).view(B,S,A,Dh).transpose(1,2)
y = noncausal_attention(q,k,v, mask=None)
y = Linear(H -> H)(y)
x = residual + dropout(y)

residual = x
y = LayerNorm(x)
y = Linear(H -> I)(y)
y = GELU(y)
y = Linear(I -> H)(y)
x = residual + dropout(y)

if use_mid_position_embeddings and layer_index < num_layers - 1:
    x = x + interpolated_mid_position[layer_index]
```

Final model output:

```text
last_hidden = LayerNorm(x)                              # [B,S,H]
pooler_output = Dense+Tanh(last_hidden[:,0])             # YolosModel only
```

Object detection head:

```text
det_hidden = last_hidden[:, -num_detection_tokens:, :]   # [B,D,H]
logits = MLP_3layer(H -> H -> H -> num_labels+1)(det_hidden)
pred_boxes = sigmoid(MLP_3layer(H -> H -> H -> 4)(det_hidden))
```

## 6. Attention requirements

- Attention type: noncausal encoder self-attention only.
- Head structure: MHA with `A=num_attention_heads`, `head_dim=hidden_size/A`.
  Representative head dims: tiny/small/base use 64; small-dwr uses 55; tiny
  random uses 8.
- Query/key/value projections: separate `nn.Linear` modules, each with
  `bias=config.qkv_bias`.
- Masking: none in model forward. Processor padding is not converted to an
  attention mask.
- Sliding/local attention: none.
- RoPE/ALiBi/relative bias: none.
- KV cache/generation: not applicable. YOLOS has no autoregressive decode.
- Packed/varlen support: none in source. Sequence length is dense
  `1 + floor(H/P) * floor(W/P) + D`.
- Backend compatibility: source advertises SDPA, FlashAttention, flex
  attention, and generic attention backend support. Eager math scales QK scores
  by `head_dim ** -0.5`, adds no mask, softmaxes on the key dimension, applies
  dropout only in training, then multiplies by V.

Sequence-length examples:

- `hustvl/yolos-small` config table length: `1 + 32*54 + 100 = 1829`.
- If preprocessing pads a batch to a typical `800 x 1328` multiple-of-16 size,
  runtime length is `1 + 50*83 + 100 = 4251`.
- `hustvl/yolos-base` config table length: `1 + 50*84 + 100 = 4301`.

The large image-token sequence makes prefill-style noncausal attention and MLP
throughput more important than the small fixed detection-token head.

## 7. Position encoding and custom math

YOLOS uses learned absolute position tables with bicubic interpolation over the
patch-token subgrid.

```python
def yolos_interpolate_initial(pos, image_hw, config):
    cls = pos[:, 0:1, :]
    det = pos[:, -config.num_detection_tokens:, :]
    patch = pos[:, 1:-config.num_detection_tokens, :]
    patch = patch.transpose(1, 2)
    patch = patch.view(1, config.hidden_size,
                       config.image_size[0] // config.patch_size,
                       config.image_size[1] // config.patch_size)
    new_h = image_hw[0] // config.patch_size
    new_w = image_hw[1] // config.patch_size
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.flatten(2).transpose(1, 2)
    return concat([cls, patch, det], dim=1)
```

Mid-position embeddings use the same split/interpolate/concat idea, with a
leading `depth=num_hidden_layers-1` dimension:

```text
mid_pos [depth,1,config_seq,H]
patch slice -> [depth,1,Ncfg,H] -> [depth,H,Hcfg/P,Wcfg/P]
interpolate -> [depth,1,Nruntime,H]
add mid_pos[i] after encoder block i for i < depth
```

Precompute opportunities:

- Initial interpolated position table can be cached per `(Hpad,Wpad,dtype)`.
- Mid-position table can be cached per `(Hpad,Wpad,dtype)` as
  `[num_layers-1,1,S,H]`.
- CLS and detection position slices are constant and should not be interpolated.
- If runtime buckets force `H,W` multiples of 16, cache keys can be compact and
  deterministic.

## 8. Preprocessing and input packing

Processor contract from `YolosImageProcessor` / `YolosImageProcessorPil`:

- Input images are converted to channel-first tensors before model handoff.
- Resize keeps aspect ratio with `size={"shortest_edge":..., "longest_edge":...}`.
- YOLOS-specific resize floors the resulting height and width to multiples of
  `mod_size=16`.
- Rescale and normalize use ImageNet mean/std in inspected configs.
- Batch images are padded on the bottom and right to the maximum resized
  height/width unless a `pad_size` is provided.
- `pixel_mask` is int64 `[B,Hpad,Wpad]`, 1 for original/resized valid pixels
  and 0 for padding.
- Model input names in the processor are `["pixel_values", "pixel_mask"]`, but
  the model graph only uses `pixel_values`.

Recommended first DinoML boundary:

```text
CPU processor owns decode/resize/rescale/normalize/pad.
DinoML runtime accepts:
  pixel_values: [B,3,Hpad,Wpad] float32/float16 NCHW, contiguous
  target_sizes: [B,2] optional postprocess input, CPU-side initially
Ignore or pass through pixel_mask for API compatibility; do not feed it into
YOLOS attention unless source behavior changes.
```

End-to-end object detection output:

```text
raw runtime outputs:
  logits:     [B,D,num_labels+1]
  pred_boxes: [B,D,4] normalized center_x, center_y, width, height in [0,1]

post_process_object_detection(threshold, target_sizes) returns list length B:
  scores: [Ni] max softmax probability excluding final no-object class
  labels: [Ni] argmax class id excluding final no-object class
  boxes:  [Ni,4] absolute xyxy in target image pixel coordinates
```

Postprocessing details:

- `prob = softmax(logits, -1)`.
- `scores, labels = prob[..., :-1].max(-1)`.
- `boxes = center_to_corners_format(pred_boxes)`.
- If target sizes are present, scale with `[img_w,img_h,img_w,img_h]`.
- Filter each image independently with `score > threshold`.
- No NMS or top-k pruning is part of HF YOLOS postprocess.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(C -> H, kernel=P, stride=P, padding=0)
  -> flatten(2)
  -> transpose(1,2)
```

Replacement:

```text
WindowFlatten -> GEMM_RCR_Bias(C*P*P -> H) -> Reshape[B,Hp*Wp,H]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Runtime `H,W` are at least one patch and divisible by `patch_size`, or the
  exact source floor behavior is explicitly preserved.
- Flatten order matches the selected activation layout.

Weight transform:

```python
# Source Conv2d weight is [out_channels, in_channels, kh, kw].
# For NHWC patch flatten order [kh, kw, c]:
w = conv.weight.permute(0, 2, 3, 1).reshape(out_channels, kh * kw * in_channels)
b = conv.bias
```

Layout constraints:

- Source-faithful NCHW flatten order is `[c,kh,kw]`.
- Preferred NHWC rewrite should be guarded at the public ABI boundary or paired
  with a known processor/runtime layout contract.

Failure cases:

- Non-divisible or non-multiple-of-16 image sizes if DinoML does not reproduce
  PyTorch Conv2d floor shape.
- Any future grouped/dilated/padded patch projection.

Parity test sketch: compare source Conv2d+flatten+transpose against rewritten
window-GEMM for fp32 and reduced precision over several padded `H,W` buckets.

### Rewrite: cache interpolated position embeddings

Source pattern:

```text
slice learned table -> bicubic interpolate patch grid -> concat CLS/patch/det
```

Replacement:

```text
CachedInitialPosition[H,W,dtype] and CachedMidPosition[H,W,dtype]
```

Preconditions:

- Position tables are constant for the loaded artifact.
- Runtime shape bucket `(H,W)` is known before execution.
- Bicubic implementation and `align_corners=False` match PyTorch.

Shape equations:

```text
Hp = H // patch_size
Wp = W // patch_size
S = 1 + Hp * Wp + num_detection_tokens
```

Failure cases:

- Arbitrary dynamic image sizes without shape-bucket cache management.
- Changing interpolation backend that shifts values enough to perturb
  detections near thresholds.

Parity test sketch: compare cached and source-interpolated initial and
mid-position tables for small, small-dwr, and base configs.

### Rewrite: QKV projections -> packed QKV

Source pattern:

```text
q = Linear(x_norm)
k = Linear(x_norm)
v = Linear(x_norm)
```

Replacement:

```text
PackedLinear(H -> 3H, bias=qkv_bias) -> Split(q,k,v)
```

Preconditions:

- Same normalized input tensor feeds q/k/v.
- Output widths and bias settings match.
- Split order is declared as Q, K, V.

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0) if bias else None
```

Failure cases:

- None in current YOLOS self-attention when source modules are standard.
- Do not reuse DETR's more constrained position-add rules here; YOLOS q/k/v
  all consume the same normalized sequence tensor.

Parity test sketch: compare per-layer q/k/v tensors and final attention output
before and after packing.

### Rewrite: detection-token-only head specialization

Source pattern:

```text
last_hidden -> suffix slice D tokens -> class MLP and box MLP
```

Replacement:

```text
SuffixSliceView -> two batched MLP/GEMM chains
```

Preconditions:

- Object-detection target only.
- Hidden sequence token order remains `[CLS, patches, detection]`.
- No caller requests all hidden states for a debugging/feature-extraction path.

Failure cases:

- Base `YolosModel` feature extraction or any future head consuming patch/CLS
  tokens after final LayerNorm.

Parity test sketch: compare raw `logits` and `pred_boxes` while materializing
only detection-token head inputs.

### Rewrite: guarded NHWC patch island

Source pattern: processor/model ABI is NCHW and patch Conv2d is OIHW.

Replacement:

```text
NCHW public ABI -> optional NHWC patch flatten/GEMM island -> token sequence
```

Required axis rewrites:

- Public input axis 1 is channels; NHWC channel axis becomes last.
- Conv weights transform from OIHW to `[O,kh,kw,I]` flatten order.
- Processor `pixel_mask` stays `[B,H,W]` and is not channel-translated.
- Position and postprocess target-size math stay in `(height,width)` order.

No-layout-translation guards:

- Public `pixel_values` ABI until a new processor/runtime contract is declared.
- Processor padding/mask generation.
- Position interpolation shape math and detection postprocess.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d -> GEMM/window-flatten rewrite. It removes an NCHW Conv2d and
  layout transpose at the source-to-token boundary.
- Packed QKV projection and noncausal attention for long image-token sequences
  (`S` commonly around 1800-4300 depending on config/runtime size).
- LayerNorm + residual scheduling around ViT blocks.
- MLP fusion `Linear -> GELU -> Linear` for transformer FFNs.
- Position interpolation cache per image-size bucket, especially when
  mid-position embeddings add a large `[layers-1,S,H]` table.

Medium priority:

- Detection head batched MLPs over only 100 detection tokens.
- Vectorized postprocess helper for softmax/max/box conversion/threshold.
- Processor-compatible resize bucket planning so runtime position-cache and
  attention profiles see stable sequence lengths.
- Dropout removal/folding for inference artifacts.

Lower priority:

- CLS pooler dense+tanh for base `YolosModel`.
- Training losses and auxiliary outputs.
- GPU image preprocessing. Useful later, but CPU processor handoff is enough
  for first parity.

## 11. Runtime staging plan

Stage 1: Config and processor handoff.

- Parse `YolosConfig`, preprocessor config, and label maps.
- Accept preprocessed `pixel_values [B,3,H,W]`; carry `pixel_mask` only as an
  optional API-compatible ignored input.
- Load weights for patch projection, token/position tables, encoder, final
  norm, and detection heads.

Stage 2: Patch and embedding parity.

- Implement source NCHW patch Conv2d or guarded Conv2d->GEMM rewrite.
- Implement CLS/detection token expansion, concatenation, and initial position
  interpolation.
- Validate output sequence shape and values for one fixed preprocessed image.

Stage 3: Encoder block parity.

- Implement one YOLOS block with LayerNorm-before-attention, residual ordering,
  GELU MLP, and optional mid-position add.
- Validate one block and then all blocks for tiny/small configs.

Stage 4: Detection heads and raw outputs.

- Implement final LayerNorm, detection-token suffix slice, class MLP, and box
  MLP with sigmoid.
- Validate `logits` and `pred_boxes` for `hustvl/yolos-small`.

Stage 5: Postprocessing parity.

- Implement HF-compatible object-detection postprocess initially as a CPU
  helper.
- Verify scores/labels/boxes for fixed thresholds and target sizes.

Stage 6: Config variation hardening.

- Add tiny, base, small-300, and small-dwr coverage.
- Ensure attention supports `head_dim=55` and runtime image shapes beyond
  config table shapes.

Stage 7: Optimizations.

- Add position-cache buckets, packed QKV, attention profiling, and NHWC patch
  island after source parity is stable.

## 12. Parity and validation plan

Random/operator tests:

- YOLOS resize shape rule: aspect-ratio resize with H/W floored to multiples of
  16.
- Patch Conv2d source path and rewritten window-GEMM path.
- Initial and mid-position interpolation with `align_corners=False`.
- Token order: CLS first, patches middle, detection suffix.
- MHA with non-64 head dims, especially `hidden=330, heads=6`.
- Center-to-corners box conversion and target-size scaling.

Model slice tests:

- Processor fixture: save HF `pixel_values` and `pixel_mask` for one or more
  images; use only `pixel_values` in model parity.
- Embedding output parity for small and base at both config and processor-sized
  image buckets.
- Single encoder block parity, with and without mid-position embeddings.
- Full `YolosModel.last_hidden_state` parity.
- Detection-token suffix head parity for class logits and sigmoid boxes.
- End-to-end postprocessed detection parity: compare number of kept boxes,
  labels, scores, and `xyxy` boxes for fixed `threshold` and `target_sizes`.

Suggested tolerances:

- fp32 source-faithful: `rtol=1e-4`, `atol=1e-5` per block; full model may need
  `rtol=2e-4` after long attention/MLP accumulation.
- fp16/bf16 optimized: compare against a reduced-precision PyTorch run; keep
  postprocess thresholds fixed and inspect detections near threshold
  separately.

## 13. Performance probes

- CPU image processor throughput: resize/normalize/pad images/sec and
  resulting `(H,W,S)` bucket distribution.
- Patch projection NCHW Conv2d vs window-GEMM/NHWC island.
- Position interpolation time with and without cached initial/mid tables.
- Encoder attention time as a function of sequence length `S`.
- Encoder MLP time for tiny/small/base/small-dwr hidden sizes.
- Packed QKV vs separate Q/K/V projection time.
- Detection heads and postprocess time, including variable retained detections.
- Batch-size sweep with mixed image sizes to measure padding waste.
- Attention backend comparison: eager/SDPA/Flash/flex equivalents for
  noncausal, unmasked attention.
- End-to-end images/sec with processor included and excluded.

No benchmark measurements are included; these are source-derived probe
recommendations.

## 14. Skip/defer list

- Training mode, dropout behavior, gradients, and gradient checkpointing.
- Hungarian matching, class/L1/GIoU losses, auxiliary loss outputs, and COCO
  annotation conversion.
- Segmentation/panoptic/semantic postprocess; YOLOS source stubs raise
  `NotImplementedError`.
- DETR-style pixel-mask attention, because YOLOS model source does not use it.
- CLS pooler output for the first `YolosForObjectDetection` target.
- GPU image decode/resize/normalize/pad preprocessing.
- NMS, because standard YOLOS postprocessing does not use it.
- Autoregressive generation, KV cache, beam search, speculative decoding, and
  tokenizer machinery; not applicable.
- Multi-GPU tensor parallelism and quantization.

## 15. Final implementation checklist

- [ ] Parse `YolosConfig`, including hidden size, layers, heads, patch size,
  detection token count, image size, `qkv_bias`, and mid-position flag.
- [ ] Parse `YolosImageProcessor` config: resize policy, mean/std, format, and
  padding/mask behavior.
- [ ] Load patch Conv2d, CLS token, detection tokens, initial position table,
  optional mid-position table, encoder blocks, final norm, and detection heads.
- [ ] Accept preprocessed `pixel_values [B,3,H,W]`; ignore `pixel_mask` in the
  model graph for parity.
- [ ] Implement source patch Conv2d + flatten + transpose.
- [ ] Add guarded patch Conv2d -> GEMM rewrite with layout-aware weight
  transform.
- [ ] Implement initial position interpolation and cache by `(H,W)`.
- [ ] Implement optional mid-position interpolation and post-block adds.
- [ ] Implement noncausal unmasked MHA, including non-64 head dimensions.
- [ ] Implement YOLOS LayerNorm/residual/MLP block ordering.
- [ ] Implement final LayerNorm and detection-token suffix slice.
- [ ] Implement class MLP and box MLP with sigmoid.
- [ ] Implement HF-compatible object-detection postprocess with no NMS.
- [ ] Add parity tests for processor handoff, embedding, position
  interpolation, one block, full encoder, raw outputs, and postprocessed boxes.
- [ ] Add variation tests for tiny, small, base, small-300, and small-dwr.
- [ ] Benchmark processor, patch projection, position interpolation, encoder
  attention/MLP, detection heads, and postprocess separately.
