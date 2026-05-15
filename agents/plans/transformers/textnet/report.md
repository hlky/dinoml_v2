# TextNet Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: czczup/textnet-base as the primary public checkpoint; czczup/textnet-tiny and czczup/textnet-small for variation.
Config source: Hugging Face config.json and preprocessor_config.json fetched with huggingface_hub; see evidence_config_sweep.json.
Source files inspected:
  transformers/src/transformers/models/textnet/configuration_textnet.py
  transformers/src/transformers/models/textnet/modeling_textnet.py
  transformers/src/transformers/models/textnet/image_processing_textnet.py
  transformers/src/transformers/models/textnet/image_processing_pil_textnet.py
  transformers/src/transformers/models/textnet/convert_textnet_to_hf.py
  transformers/tests/models/textnet/test_modeling_textnet.py
  transformers/tests/models/textnet/test_image_processing_textnet.py
  transformers/docs/source/en/model_doc/textnet.md
Any missing files or assumptions: no remote code is required. Official czczup checkpoints are open. This report targets inference for TextNetBackbone first, with TextNetModel pooling and TextNetForImageClassification as small follow-ons.
```

TextNet is not a text model despite the family name. It is a CNN vision backbone for text detection feature extraction, with optional image classification plumbing. The current in-library source is authoritative for DinoML; the conversion script only explains how original FAST checkpoints map into the Hugging Face module names.

## 2. High-level architecture

Runtime contract:

```text
RGB image preprocessing -> NCHW pixel_values -> stem Conv/BN/ReLU -> 4 CNN stages -> feature maps
                                                   -> optional adaptive pool 2x2
                                                   -> optional adaptive pool 1x1 + flatten + Linear classifier
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize shortest edge to the configured value, round height/width up to `size_divisor`, rescale by the generic image processor factor, normalize with ImageNet mean/std, emit `pixel_values` in channel-first format.
- GPU/runtime backbone: all neural work is NCHW `Conv2d`, `BatchNorm2d`, branch adds, and ReLU.
- Independently stageable outputs: backbone feature maps for `stage1..stage4`; pooled `TextNetModel` output; classifier logits when `num_labels > 0`.
- No text tokens, attention masks, sequence packing, generation, or KV cache exist for this family.

## 3. Important config dimensions

Representative checkpoint sweep:

| Checkpoint | Architecture | Image/preproc | Hidden sizes | Stage depths | Blocks | Kernel mix | Output features |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `czczup/textnet-tiny` | `TextNetBackbone` | shortest edge 640, divisor 32 | `[64,64,128,256,512]` | `[3,4,4,4]` | 15 | 9x `3x3`, 3x `1x3`, 3x `3x1` | `stage1..stage4` |
| `czczup/textnet-small` | `TextNetBackbone` | shortest edge 640, divisor 32 | `[64,64,128,256,512]` | `[2,8,8,5]` | 23 | 11x `3x3`, 6x `1x3`, 6x `3x1` | `stage1..stage4` |
| `czczup/textnet-base` | `TextNetBackbone` | shortest edge 640, divisor 32 | `[64,64,128,256,512]` | `[10,10,8,5]` | 33 | 19x `3x3`, 5x `1x3`, 9x `3x1` | `stage1..stage4` |
| `Raghavan/textnet-base` | `TextNetBackbone` | shortest edge 640, source default divisor 32 | `[64,64,128,256,512]` | `[10,10,8,5]` | 33 | same as base | `stage1..stage4` |
| `onnx-community/textnet-base` | `TextNetBackbone` | shortest edge 640, divisor 32 | `[64,64,128,256,512]` | `[10,10,8,5]` | 33 | same as base | `stage1..stage4` |

Core defaults from `TextNetConfig`:

| Field | Effective value or role |
| --- | --- |
| `model_type` | `textnet` |
| `stem_num_channels` | 3 |
| `stem_out_channels` | 64 |
| `stem_kernel_size`, `stem_stride` | 3, 2 |
| `hidden_sizes` | channels for stem and stages: `[64,64,128,256,512]` |
| `conv_layer_kernel_sizes` | per-stage NAS pattern of `3x3`, `1x3`, `3x1` |
| `conv_layer_strides` | one stride-2 block per stage in public checkpoints |
| `batch_norm_eps` | `1e-5` |
| `stem_act_func` | `relu` |
| `image_size` | config metadata, `[640,640]` in public checkpoints |
| `out_features/out_indices` | backbone feature selection; public checkpoints select all four stages |
| `cache support` | none |

For an input after preprocessing of `[B,3,H,W]` where `H` and `W` are divisible by 32, public checkpoint spatial shapes are:

```text
stem:   [B,  64, H/2,  W/2]
stage1: [B,  64, H/4,  W/4]
stage2: [B, 128, H/8,  W/8]
stage3: [B, 256, H/16, W/16]
stage4: [B, 512, H/32, W/32]
```

The integration test observes a real preprocessed COCO image producing `last_hidden_state = [1,512,20,27]`, implying a processor output of about `[1,3,640,864]`.

## 3a. Family variation traps

- TextNet is CNN-only and has no attention, MLP, RoPE, tokenizer, vocab, cache, or decoder surface.
- The main operator-significant variation is stage depth and kernel pattern. Tiny/small/base use the same channel widths but different counts of `3x3`, `1x3`, and `3x1` RepConv blocks.
- Every RepConv block is multi-branch in the current source: main conv+BN, optional vertical conv+BN, optional horizontal conv+BN, optional identity BN, branch sum, ReLU. Do not assume a single Conv2d per block unless weights have been explicitly reparameterized.
- Source layout is NCHW. NHWC/channel-last is only a guarded optimization. Axis-sensitive ops include `BatchNorm2d` channel axis 1, adaptive pooling over H/W, flatten after `1x1` pooling, backbone feature-map consumers, and processor output layout.
- `TextNetConvLayer` has a source-level trap for tuple stem kernels: the tuple-padding branch references `config.kernel_size`, which is not a declared config field. Public checkpoints use integer `stem_kernel_size=3`, so the bug is not hit.
- `Raghavan/textnet-base` omits `size_divisor` from `preprocessor_config.json`; current source default is 32. Treat this as an omitted-field default, not a different processor ABI.
- `TextNetBackbone` returns selected hidden states by stage name. `out_features=None` falls back through `BackboneConfigMixin` behavior and must be validated from source/tests before assuming all stages are returned.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW dense tensor input and output feature maps.
- Static/dynamic shape validation for `[B,3,H,W]`, with `H,W` preferably divisible by 32 for public processors.
- Tuple/list output packing for backbone `feature_maps`.
- `Flatten` after adaptive `1x1` pooling for classification.

Neural network primitives:

- `Conv2d` NCHW, bias false for all backbone convolutions.
- Kernel sizes `3x3`, `1x3`, `3x1`; stem `3x3`.
- Strides 1 and 2; padding is same-like: `((kh-1)//2, (kw-1)//2)`.
- `BatchNorm2d` inference: scale, bias, running mean, running variance, epsilon.
- Elementwise add for 2-4 branch sums.
- `ReLU`.
- `AdaptiveAvgPool2d((2,2))` for `TextNetModel.pooler_output`.
- `AdaptiveAvgPool2d((1,1))`, `Flatten`, `Linear(512 -> num_labels)` for `TextNetForImageClassification`.

Attention, position, generation, cache, quantized weight, packed sequence, distributed, tokenizer, and scatter/indexed-update ops:

- Not applicable for the primary target.

Preprocessing-coupled ops:

- RGB conversion.
- Resize shortest edge to 640 or configured `size.shortest_edge` with bilinear interpolation.
- Round resized H/W up to multiples of `size_divisor` before the final resize call.
- Rescale and normalize in channel-first format with ImageNet mean/std.

Layout rewrite notes:

- Initial translation should preserve NCHW.
- A guarded NHWC rewrite can cover local Conv/BN/ReLU regions only if all branch tensors in a RepConv block share NHWC and consumers accept NHWC or are translated back.
- Required axis rewrites: `BatchNorm2d` channel `dim=1 -> dim=-1`; adaptive pooling reductions over source H/W axes `2,3 -> 1,2` after NHWC; classifier flatten must see `[B,1,1,C] -> [B,C]` instead of `[B,C,1,1] -> [B,C]`.
- Protect backbone output boundaries with `no_layout_translation()` unless the downstream detector/head is audited with the same feature-map layout contract.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B,3,H,W]
x = Conv2d(3 -> 64, kernel=3x3, stride=2, padding=1, bias=False)(pixel_values)
x = BatchNorm2d(64, eps=1e-5)(x)
x = ReLU(x)
shape: [B,64,H/2,W/2]
```

RepConv block with `in_channels=Cin`, `out_channels=Cout`, `kernel=(kh,kw)`, `stride=s`:

```text
main = BN(Conv2d(Cin -> Cout, kernel=(kh,kw), stride=s, padding=((kh-1)//2,(kw-1)//2), bias=False)(x))
if kw != 1:
    vertical = BN(Conv2d(Cin -> Cout, kernel=(kh,1), stride=s, padding=((kh-1)//2,0), bias=False)(x))
    main = main + vertical
if kh != 1:
    horizontal = BN(Conv2d(Cin -> Cout, kernel=(1,kw), stride=s, padding=(0,(kw-1)//2), bias=False)(x))
    main = main + horizontal
if Cin == Cout and s == 1:
    main = main + BatchNorm2d(Cin)(x)
y = ReLU(main)
```

Stage `i`, repeated by config:

```text
stage_in = hidden_sizes[i]
stage_out = hidden_sizes[i+1]
first block consumes stage_in and emits stage_out
remaining blocks consume and emit stage_out
```

Public base stage details:

| Stage | Channels | Spatial scale after stage | Base block kernels/strides |
| --- | --- | --- | --- |
| stage1 | 64 -> 64 | `H/4,W/4` | 10 blocks; strides `[1,2,1,1,1,1,1,1,1,1]` |
| stage2 | 64 -> 128 | `H/8,W/8` | 10 blocks; first stride 2 |
| stage3 | 128 -> 256 | `H/16,W/16` | 8 blocks; first stride 2 |
| stage4 | 256 -> 512 | `H/32,W/32` | 5 blocks; first stride 2 |

Head paths:

```text
TextNetModel: last_hidden_state -> AdaptiveAvgPool2d((2,2)) -> pooler_output [B,512,2,2]
TextNetForImageClassification: last_hidden_state -> AdaptiveAvgPool2d((1,1)) -> Flatten -> Linear(512,num_labels)
```

## 6. Attention requirements

No attention is required. TextNet has no causal/noncausal self-attention, cross-attention, attention mask, packed/varlen metadata, sliding-window pattern, ALiBi/RoPE, KV cache, FlashAttention, or SDPA path. Backbone feature outputs are independently cacheable by a downstream detection stack, but that is an application-level feature cache, not a Transformer KV cache.

## 7. Position encoding and custom math

No position encoding exists in the model. Spatial information is carried by convolution and padding.

The only custom-ish math is RepConv branch aggregation:

```python
def repconv_inference_source(x, main, vertical=None, horizontal=None, identity_bn=None):
    y = main.conv_bn(x)
    if vertical is not None:
        y = y + vertical.conv_bn(x)
    if horizontal is not None:
        y = y + horizontal.conv_bn(x)
    if identity_bn is not None:
        y = y + identity_bn(x)
    return relu(y)
```

For an optimized deployment rewrite, each Conv+BN branch can be folded into conv weights/biases, then compatible branch kernels can be padded into a common `kh x kw` kernel and summed. That is an optimization and must be parity-tested against the unfused source graph.

## 8. Preprocessing and input packing

The processor emits only `pixel_values`; there are no masks, token IDs, boxes, OCR metadata, or packed descriptors.

CPU/data-pipeline steps:

```text
input image -> RGB -> channel-first tensor -> resize shortest edge -> ceil H/W to size_divisor -> bilinear resize
            -> rescale -> normalize -> pixel_values [B,3,H',W']
```

Defaults from public checkpoint configs:

- `size.shortest_edge = 640`
- `size_divisor = 32`
- `do_center_crop = false`
- `do_rescale = true`
- `do_normalize = true`
- `image_mean = [0.485,0.456,0.406]`
- `image_std = [0.229,0.224,0.225]`
- `do_convert_rgb = true`

DinoML can initially keep preprocessing outside the compiled graph. If processor parity is brought into DinoML later, the resize-to-divisor policy must be tested on non-square images because output shapes drive the final feature-map dimensions.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern:

```text
Conv2d(bias=False) -> BatchNorm2d(eps)
```

Replacement:

```text
Conv2d(bias=True) with folded weights
```

Preconditions:

- Inference mode using frozen BN running statistics.
- No training updates or batch-stat behavior.
- Conv output channel equals BN channel count.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = conv_w * scale[:, None, None, None]
b_fold = beta - running_mean * scale
```

Failure cases:

- Training mode, missing BN buffers, nonstandard BN behavior, or mutable weight sharing.

Parity test sketch:

- Random NCHW input, compare each folded branch output to PyTorch module in fp32 and fp16 tolerances.

### Rewrite: RepConv branch reparameterization

Source pattern:

```text
sum(ConvBN(kh,kw), optional ConvBN(kh,1), optional ConvBN(1,kw), optional IdentityBN) -> ReLU
```

Replacement:

```text
single folded Conv2d(kh,kw,bias=True) -> ReLU
```

Preconditions:

- Inference-only with all branch BN folded.
- Branch strides and output shapes match.
- Vertical/horizontal kernels are zero-padded into the main kernel footprint at the center column/row.
- Identity BN is only present when `Cin == Cout` and `stride == 1`; represent as a centered delta kernel before adding.

Shape equations:

```text
input [B,Cin,H,W] -> output [B,Cout,floor((H+2*ph-kh)/s)+1,floor((W+2*pw-kw)/s)+1]
```

Layout constraints:

- Source semantics are NCHW. NHWC version needs transformed weights `[kh,kw,Cin,Cout]` and channel-axis BN rewrite.

Failure cases:

- Dynamic branch enablement, non-identity padding semantics, training mode, tuple stem bug, or downstream code expecting individual branch activations.

Parity test sketch:

- Compare block-level outputs for all kernel patterns `3x3`, `1x3`, `3x1`, stride 1/2, identity present/absent.

### Rewrite: Local NCHW Conv/BN/ReLU island to NHWC

Source pattern:

```text
NCHW input -> repeated Conv2d/BN/Add/ReLU -> NCHW feature map
```

Replacement:

```text
NCHW->NHWC boundary transpose -> NHWC fused kernels -> NHWC->NCHW boundary transpose
```

Preconditions:

- Entire RepConv branch sum stays inside the NHWC island.
- Feature-map consumers either remain inside the island or receive an explicit translated boundary.
- Axis-sensitive attrs are rewritten: BN channel axis, pooling spatial axes, flatten order.

Failure cases:

- Exposed backbone feature maps consumed by unaudited detection heads, mixed-layout residual branches, or external users expecting NCHW tensors.

Parity test sketch:

- Stage-by-stage outputs with all `out_features` enabled, including non-square resized inputs.

### Rewrite: AdaptiveAvgPool2d fixed output

Source pattern:

```text
AdaptiveAvgPool2d((1,1)) or ((2,2))
```

Replacement:

```text
static grid average reductions over H/W bins
```

Preconditions:

- Positive H/W known at runtime.
- Match PyTorch adaptive pooling bin boundaries exactly.

Failure cases:

- Treating adaptive pooling as ordinary average pooling with fixed kernel/stride when H/W is not exactly divisible by output bins.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d+BN+ReLU inference fusion for NCHW. This is the dominant repeated pattern and is enough for a faithful first backend.
- RepConv reparameterization to single Conv+ReLU. This collapses 2-4 branches per block and removes branch-add overhead.
- Asymmetric `1x3`/`3x1` convolution kernels. They are common in all public variants and should not fall back to an inefficient generic path.

Medium priority:

- NHWC/channel-last convolution islands with explicit boundary guards. Useful for CUDA conv libraries, but only after NCHW parity is stable.
- Backbone multi-output feature-map materialization. Detection use cases need several feature levels; avoid unnecessary copies when returning `stage1..stage4`.
- Adaptive pooling kernels for `1x1` and `2x2` outputs.

Lower priority:

- Classification head fusion: `AdaptiveAvgPool2d(1) -> Flatten -> Linear`. Small compared with backbone convolution cost.
- Processor acceleration. Resize/normalize may matter for high-throughput serving, but first integration can leave it in the data pipeline.

## 11. Runtime staging plan

Stage 1: parse `TextNetConfig`, load safetensors weights, and run stem plus one RepConv block in NCHW with PyTorch parity.

Stage 2: implement unfused backbone parity for tiny config, returning all `feature_maps`.

Stage 3: add small/base config sweeps and non-square processor-shaped inputs such as `[1,3,640,864]`.

Stage 4: fold Conv+BN and validate folded weights branch-by-branch.

Stage 5: add RepConv single-conv reparameterization behind a graph rewrite flag.

Stage 6: add `TextNetModel` pooler output and `TextNetForImageClassification` head.

Stage 7: evaluate guarded NHWC islands and external conv provider choices once NCHW parity and feature-map ABI are boring.

Initially stub or defer preprocessing in compiled DinoML: accept already-normalized `pixel_values`.

## 12. Parity and validation plan

- Config parser tests for tiny/small/base depths, kernels, strides, hidden sizes, `out_features`, and omitted `size_divisor` default.
- Random tensor tests for `Conv2d+BN`, `RepConvLayer`, and `AdaptiveAvgPool2d((1,1)/(2,2))`.
- Single-block parity for each kernel pattern: `3x3`, `1x3`, `3x1`, stride 1 and stride 2.
- Stage parity after each stage, with backbone `hidden_states` and `feature_maps` enabled.
- Full `TextNetBackbone` parity for `czczup/textnet-tiny`, then `small`, then `base`.
- Known integration slice: `czczup/textnet-base` no-head inference should reproduce `last_hidden_state` shape `[1,512,20,27]` for the documented COCO image processor path.
- Classification head parity with synthetic `num_labels`, including `num_labels=0` identity behavior if admitted.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5` for module-level tests; folded/reparameterized fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 looser stage-level tolerances such as `rtol=1e-2, atol=1e-2` after source comparison.

## 13. Performance probes

- Processor throughput: image decode/RGB/resize/normalize images/sec for square and non-square inputs.
- Stem and per-stage latency with NCHW unfused kernels.
- RepConv branch count versus reparameterized single-conv speedup.
- Tiny/small/base backbone throughput at batch sizes 1, 4, 8, 16.
- Resolution sweep after processor rounding: shortest edge 320/640/960 and non-square aspect ratios.
- Feature-output cost: last hidden only versus all `stage1..stage4` feature maps.
- Layout probe: NCHW provider path versus guarded NHWC island, including transpose overhead.
- Classification head overhead separated from backbone.

## 14. Skip/defer list

- Training, loss computation, and batch-stat BatchNorm behavior.
- Original FAST detection neck/head and postprocessing; this Transformers family only owns the backbone/classifier.
- General OCR/text detection end-to-end postprocessing.
- Attention, KV cache, generation, tokenizer behavior, masks, and sequence packing.
- Quantization and packed weights; public safetensors are dense.
- Multi-GPU/tensor parallel support.
- NHWC as a default semantic graph; keep it as a guarded optimization.
- Tuple-valued `stem_kernel_size` until the source typo around `config.kernel_size` is resolved upstream or explicitly guarded.

## 15. Final implementation checklist

- [ ] Parse `TextNetConfig` and reject unsupported tuple stem bug cases or patch admission with an explicit guard.
- [ ] Load dense safetensors for `TextNetBackbone`.
- [ ] Accept preprocessed NCHW `pixel_values`.
- [ ] Implement NCHW `Conv2d` for kernels `3x3`, `1x3`, `3x1`, strides 1/2, same-like padding, bias false.
- [ ] Implement inference `BatchNorm2d`.
- [ ] Implement branch add and ReLU.
- [ ] Implement `TextNetRepConvLayer` source graph.
- [ ] Return selected backbone feature maps by `out_features/out_indices`.
- [ ] Add adaptive average pooling for `(2,2)` and `(1,1)`.
- [ ] Add classifier `Flatten + Linear(512 -> num_labels)` as optional.
- [ ] Add Conv+BN fold rewrite.
- [ ] Add RepConv reparameterization rewrite with branch-kernel padding.
- [ ] Add guarded NHWC island experiment with explicit axis rewrites and NCHW feature-map boundaries.
- [ ] Add parity tests for tiny/small/base configs and non-square processor outputs.
- [ ] Benchmark branch source graph versus reparameterized graph.
