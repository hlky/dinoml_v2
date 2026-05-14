# UVDoc Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/UVDoc_safetensors
Config source: https://huggingface.co/PaddlePaddle/UVDoc_safetensors/resolve/main/config.json
Source files inspected:
- X:/H/transformers/src/transformers/models/uvdoc/modular_uvdoc.py
- X:/H/transformers/src/transformers/models/uvdoc/modeling_uvdoc.py
- X:/H/transformers/src/transformers/models/uvdoc/configuration_uvdoc.py
- X:/H/transformers/src/transformers/models/uvdoc/image_processing_uvdoc.py
- X:/H/transformers/tests/models/uvdoc/test_modeling_uvdoc.py
- X:/H/transformers/src/transformers/models/pp_lcnet/modeling_pp_lcnet.py, for the modular inheritance source of Conv+BN+activation behavior
Representative configs:
- PaddlePaddle/UVDoc_safetensors config.json and preprocessor_config.json
- PaddlePaddle/UVDoc Paddle inference config.json, used only as a non-native deployment-shape reference
Any missing files or assumptions:
- Only one public Transformers-native UVDoc checkpoint config was found. The older PaddlePaddle/UVDoc repo is not a Transformers `model_type="uvdoc"` config and should not be treated as native source.
- The generated modeling/config/image-processing files state they are generated from modular_uvdoc.py. Future Transformers source edits should target modular_uvdoc.py; runtime behavior was checked against the generated files.
```

Primary DinoML target: inference-only `UVDocModel` document rectification on CUDA. This is a CNN + dense mesh prediction + postprocess grid-sampling pipeline, not a Transformer attention model.

## 2. High-level architecture

Dataflow:

```text
image processor -> NCHW pixel_values -> ResNet-like CNN downsampler -> parallel dilated bridge blocks -> channel concat -> CNN mesh head -> 2-channel rectification grid -> postprocess interpolate + grid_sample -> rectified image
```

Stage decomposition:

- CPU/data pipeline: image loading, tensor conversion, optional grouping by shape, rescale/normalize, RGB-to-BGR channel gather, bilinear resize to 712x488 with `align_corners=True`, and preservation of pre-resize `original_images`.
- GPU runtime graph: NCHW Conv2d/BatchNorm2d/activation blocks, residual adds, dilated convolutions, feature-map capture, `cat(dim=1)`, and final Conv2d mesh prediction.
- Postprocessing: per-image prediction upsample to each original image size, `permute` to NHW2 grid, `grid_sample(align_corners=True)`, CHW-to-HWC permute, scale by 255, channel flip, cast to uint8.
- Independently stageable pieces: backbone feature maps, bridge feature concat, head mesh output, and postprocess rectification can each be validated against PyTorch before full image parity.

## 3. Important config dimensions

Native `PaddlePaddle/UVDoc_safetensors` config:

| Field | Value | Operator impact |
| --- | --- | --- |
| `model_type` | `uvdoc` | `UVDocModel` |
| `kernel_size` | 5 | Main conv kernels are 5x5 except 1x1 connector |
| input layout | NCHW | Processor emits `channels_first` |
| processor resize | 712x488 | Default runtime input after preprocessing |
| `resnet_head` | `[3,32]`, `[32,32]` | Two stride-2 Conv+BN+ReLU layers |
| `resnet_configs` | 3 stages, 3/4/6 residual blocks | Stage 2 and 3 downsample spatially |
| backbone output channels | 128 | Final ResNet and every bridge block output width |
| `stage_configs` | 6 bridge blocks with dilation lists `[1]`, `[2]`, `[5]`, `[8,3,2]`, `[12,7,4]`, `[18,12,6]` | Parallel dilated Conv+BN+ReLU blocks over same ResNet feature map |
| `out_features` | `stage1` through `stage6` | Six 128-channel bridge outputs are concatenated |
| concat width | 768 | Head connector input channels = 128 * 6 |
| `bridge_connector` | `[128,128]` in config, multiplied by bridge count in source | 1x1 Conv+BN+ReLU from 768 to 128 |
| `out_point_positions2D` | `[128,32]`, `[32,2]` | Conv+BN+PReLU then Conv2d to two coordinate channels |
| `padding_mode` | `reflect` | Used only in `UVDocPointPositions2D`; most backbone convs use zero padding |
| `hidden_act` | `prelu` | Head conv_down activation has learned PReLU parameters |
| attention/cache | none | No MHA, RoPE, KV cache, or generation controller |

Shape sweep:

| Source/config | Input | ResNet output | Bridge concat | Mesh output | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| HF native preprocessor + config | `[B,3,712,488]` | `[B,128,45,31]` | `[B,768,45,31]` | `[B,2,45,31]` | Matches integration test expected mesh shape for batch size B |
| Transformers tiny test config | `[3,3,128,128]` | `[3,32,8,8]` | `[3,64,8,8]` | `[3,2,8,8]` | Uses reduced channels and 2 bridge outputs |
| PaddlePaddle/UVDoc inference config | dynamic `img`: min `[1,3,128,64]`, opt `[1,3,256,128]`, max `[8,3,512,256]` | Paddle artifact only | Paddle artifact only | Paddle artifact only | Useful deployment hint, not native Transformers config |

## 3a. Family variation traps

- The family is not attention-based despite living under `transformers.models`; do not require attention, masks, KV cache, token embeddings, logits, sampling, or language control.
- `UVDocBridge.forward` calls every bridge block on the same ResNet feature tensor and relies on `capture_outputs` to expose per-block hidden states. Do not lower it as a sequential stack where block 2 consumes block 1 output unless source changes.
- The head connector source computes input channels as `config.bridge_connector[0] * len(stage_configs)`. A config with different `out_features` must stay consistent with that width or be rejected.
- `dilation_values` appears in checkpoint config but the inspected source reads `backbone_config.stage_configs`, not the top-level `dilation_values` field.
- NCHW is semantic source layout. NHWC/channel-last is only an optimization candidate with guards for Conv2d, BatchNorm2d, `cat(dim=1)`, `F.interpolate`, `prediction[i:i+1]`, `permute(0,2,3,1)`, `grid_sample`, CHW/HWC output permutes, and channel gathers/flips.
- `padding_mode="reflect"` is not global. Backbone/head most convs use zero padding because their `UVDocConvLayer` calls omit the top-level padding mode. Reflect padding is used in `UVDocPointPositions2D`.
- Processor does RGB-to-BGR by channel indexing before resize and stores `original_images` after rescale/normalize and BGR conversion, before interpolation to 712x488.
- Public HF has both `PaddlePaddle/UVDoc_safetensors` and `PaddlePaddle/UVDoc`. The latter is a Paddle inference deployment repo with `.pdiparams` and dynamic-shape metadata, not the native Transformers config.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW dense tensor ABI for `[B,C,H,W]`.
- Channel gather `x[:, [2,1,0], :, :]` in processor.
- `torch.cat(feature_maps, dim=1)` for six `[B,128,H,W]` tensors into `[B,768,H,W]`.
- Per-image slice `prediction[i:i+1]`.
- `unsqueeze(0)`, `squeeze(0)`, `permute(0,2,3,1)`, `permute(1,2,0)`, `flip(dims=[-1])`.
- Scalar multiply and dtype cast to uint8 in postprocess.

Neural network primitives:

- Conv2d NCHW, mostly `kernel=5`, `stride=1/2`, dilation in `{1,2,3,4,5,6,7,8,12,18}`, zero padding or reflect padding, bias optional.
- Conv2d 1x1 connector, `768 -> 128`.
- BatchNorm2d inference folding or runtime BN for all `UVDocConvLayer` instances.
- ReLU activation.
- PReLU activation in the head, with learned slope parameter.
- Residual add and activation: `ReLU(conv_final + conv_down/input)`.
- Identity shortcut for non-downsample blocks.

Attention/generation primitives:

- Not required. No causal/noncausal attention, masks, packed sequence metadata, RoPE, ALiBi, KV cache, logits, or sampling path exists for the primary target.

Preprocessing-coupled ops:

- Torchvision-style image tensor conversion to channels-first.
- Rescale by `1/255`.
- Optional normalize hook exists in processor signature; native preprocessor config omits mean/std and uses default processor behavior.
- Bilinear resize/interpolate to `[712,488]` with `align_corners=True`.
- Shape grouping/reordering is a batching optimization in processor, not part of the neural graph.

Postprocessing ops:

- Bilinear interpolate mesh `[1,2,45,31] -> [1,2,H_original,W_original]` with `align_corners=True`.
- `grid_sample(original_image, grid, align_corners=True)` where grid is `[1,H_original,W_original,2]`.
- Output records are variable-size per original image; no NMS, thresholding, boxes, labels, or masks.

## 5. Layer/block breakdown

Default native forward, with input `[B,3,712,488]`:

```text
processor:
  image -> [B,3,H0,W0]
  rescale/normalize
  RGB-to-BGR channel gather
  original_images = pre-resize BGR tensors
  bilinear resize align_corners=True -> pixel_values [B,3,712,488]

resnet_head:
  Conv5x5 s2 p2 3 -> 32, BN, ReLU     -> [B,32,356,244]
  Conv5x5 s2 p2 32 -> 32, BN, ReLU    -> [B,32,178,122]

resnet_down stage 0:
  block 32 -> 32, dilation 1, no downsample
  block 32 -> 32, dilation 3, no downsample
  block 32 -> 32, dilation 3, no downsample
  output [B,32,178,122]

resnet_down stage 1:
  block 32 -> 64, dilation 1, downsample stride 2
  3 blocks 64 -> 64, dilation 3
  output [B,64,89,61]

resnet_down stage 2:
  block 64 -> 128, dilation 1, downsample stride 2
  5 blocks 128 -> 128, dilation 3
  output [B,128,45,31]

bridge, 6 captured blocks, each consumes the same ResNet output:
  block1: Conv5x5 dilation 1 -> [B,128,45,31]
  block2: Conv5x5 dilation 2 -> [B,128,45,31]
  block3: Conv5x5 dilation 5 -> [B,128,45,31]
  block4: Conv5x5 dilation 8 -> dilation 3 -> dilation 2 -> [B,128,45,31]
  block5: Conv5x5 dilation 12 -> dilation 7 -> dilation 4 -> [B,128,45,31]
  block6: Conv5x5 dilation 18 -> dilation 12 -> dilation 6 -> [B,128,45,31]

head:
  cat six bridge feature maps on channel axis -> [B,768,45,31]
  Conv1x1 768 -> 128, BN, ReLU -> [B,128,45,31]
  Conv5x5 reflect p2 128 -> 32, BN, PReLU -> [B,32,45,31]
  Conv5x5 reflect p2 32 -> 2 -> [B,2,45,31]
```

Residual block detail:

```text
residual = Conv5x5(stride=2 if downsample else 1, padding=2, bias=True, no activation)(x) or identity(x)
y = Conv5x5(stride=2 if downsample else 1, padding=2*dilation, dilation=dilation, bias=True, BN, ReLU)(x)
y = Conv5x5(stride=1, padding=2*dilation, dilation=dilation, bias=True, BN, no activation)(y)
out = ReLU(y + residual)
```

## 6. Attention requirements

No attention is required for the primary target. The source has:

- no self-attention or cross-attention,
- no MHA/MQA/GQA,
- no attention masks,
- no packed/varlen sequence descriptors,
- no sliding-window or sparse attention,
- no position encodings,
- no autoregressive or encoder-decoder KV cache.

The only cache-like boundary worth staging is independent reuse of preprocessed/resized `pixel_values`, backbone bridge feature maps, or final mesh predictions when a caller wants to postprocess the same prediction against multiple output policies.

## 7. Position encoding and custom math

There is no RoPE, ALiBi, relative bias, learned absolute embedding, or token position math.

Custom source math that matters for parity:

```python
def uvdoc_postprocess(prediction_i, original_image):
    if original_image.ndim == 3:
        original_image = original_image.unsqueeze(0)
    mesh = interpolate(prediction_i, size=original_image.shape[2:], mode="bilinear", align_corners=True)
    grid = mesh.permute(0, 2, 3, 1)
    rectified = grid_sample(original_image, grid, align_corners=True)
    image = rectified.squeeze(0).permute(1, 2, 0)
    image = (image * 255.0).flip(dims=[-1]).to(torch.uint8)
    return image
```

The grid coordinate convention is PyTorch `grid_sample`: normalized x/y coordinates in the last dimension, with `align_corners=True`. The model predicts those coordinates directly; DinoML should validate range behavior against PyTorch rather than clamp unless a separate postprocess policy requests it.

## 8. Preprocessing and input packing

The native preprocessor config emits:

```text
pixel_values: [B,3,712,488], channels_first, BGR order after channel gather, float tensor
original_images: list of [3,H_i,W_i] tensors, BGR order, rescaled/normalized, not resized
```

Processor steps:

1. Group images by shape for vectorized preprocessing.
2. Rescale by `0.00392156862745098` and normalize if requested.
3. Convert RGB to BGR with channel gather.
4. Save `original_images` after BGR/rescale/normalize and before resizing.
5. Resize to `[height=712,width=488]` with bilinear interpolation and `align_corners=True`.
6. Return `BatchFeature` with `pixel_values` and non-converted `original_images`.

End-to-end parity requires postprocess ownership. If DinoML initially owns only the neural graph, the runtime output contract is `[B,2,45,31]` mesh prediction and the caller must perform HF-equivalent postprocessing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern:

```text
Conv2d(weight,bias?) -> BatchNorm2d(running_mean,running_var,gamma,beta,eps) -> activation
```

Replacement:

```text
Conv2d(folded_weight, folded_bias) -> activation
```

Preconditions:

- Model is in eval/inference mode.
- BN running stats are materialized and stable.
- Input dtype tolerance covers folded-weight rounding for fp16/bf16.

Shape equations:

- Same Conv2d output shape as source.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
folded_weight = weight * scale.reshape(-1, 1, 1, 1)
folded_bias = beta + (bias_or_zero - running_mean) * scale
```

Failure cases:

- Training mode, missing BN stats, or dynamic mutation of BN parameters.

Parity test sketch:

- Compare one folded block against PyTorch Conv+BN for fp32 and fp16 on `[B,C,H,W]`.

### Rewrite: 1x1 Conv2d connector -> per-pixel GEMM

Source pattern:

```text
[B,768,H,W] -> Conv2d kernel=1 stride=1 padding=0 groups=1 -> [B,128,H,W]
```

Replacement:

```text
NCHWToNHWC or logical flatten [B*H*W,768] -> GEMM(weight.T) + bias -> reshape
```

Preconditions:

- Kernel is 1x1, stride 1, dilation 1, padding 0, groups 1.
- Layout transform is either explicit or fused with adjacent ops.

Failure cases:

- If head remains NCHW and surrounding conv kernels are not layout-aware, standalone transposes may erase the win.

### Rewrite: guarded NCHW CNN island -> NHWC/channel-last island

Source pattern:

```text
NCHW Conv2d/BN/ReLU residual island -> cat(dim=1) -> NCHW head convs
```

Replacement:

```text
NHWC Conv2d/BN/ReLU residual island -> cat(axis=-1) -> NHWC head convs -> convert only at external ABI/postprocess boundary
```

Preconditions:

- Entire island from `pixel_values` through mesh output is controlled.
- All Conv2d weights are transformed from `[O,I,KH,KW]` to provider layout as needed.
- `BatchNorm2d` channel axis rewrites from `dim=1` to `dim=-1`.
- `cat(dim=1)` rewrites to `cat(axis=-1)`.
- Postprocess either accepts NCHW mesh or explicitly converts NHWC mesh back before `interpolate`/`grid_sample`.

Failure cases:

- Mixed NCHW/NHWC consumers, feature-map capture ABI expecting NCHW, processor/postprocessor running in PyTorch, or any unrewritten axis-sensitive op.

Parity test sketch:

- One full neural graph on static `[1,3,712,488]`; compare mesh `[1,2,45,31]` after converting optimized layout output back to NCHW.

### Rewrite: postprocess as explicit grid-sampling stage

Source pattern:

```text
prediction [B,2,h,w], original_images list -> interpolate -> permute -> grid_sample -> HWC uint8
```

Replacement:

```text
MeshUpsample + GridSampleBilinearAlignCorners + HWCConvert + ChannelFlip + QuantizeUint8
```

Preconditions:

- Original image tensor is available in BGR, rescaled form.
- Per-image original sizes are known.
- `align_corners=True` is preserved.

Failure cases:

- Caller supplies RGB originals or unscaled uint8 originals without matching processor semantics.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + folded BatchNorm + activation for NCHW CNN inference. This covers almost every neural layer.
- Dilated 5x5 Conv2d coverage, including large dilations up to 18, because bridge quality depends on those receptive fields.
- `grid_sample` postprocess parity with `align_corners=True`, if DinoML owns end-to-end rectified-image output.

Medium priority:

- Residual block fusion around two Conv+activation ops and add+ReLU.
- 1x1 connector GEMM or optimized pointwise Conv2d.
- Static-shape specialization for default `[B,3,712,488] -> [B,2,45,31]`.

Lower priority:

- Processor grouping/reordering acceleration; useful for host throughput but not the core runtime bottleneck.
- Full NHWC island rewrite after NCHW parity is stable.
- Dynamic-shape CNN support beyond the default preprocessor size and the Paddle deployment hints.

## 11. Runtime staging plan

Stage 1: Config and weight loading.

- Parse `UVDocConfig` plus nested `UVDocBackboneConfig`.
- Reject non-native Paddle inference configs for the Transformers path.
- Preserve `out_features`/bridge count consistency.

Stage 2: Neural graph parity for static default input.

- Implement/fold Conv2d+BatchNorm+ReLU/PReLU, residual blocks, dilated convs, cat, and head.
- Target `[1,3,712,488] -> [1,2,45,31]`.

Stage 3: Backbone/bridge/head subgraph parity.

- Validate ResNet output, each bridge output, concat tensor, and final mesh independently.

Stage 4: End-to-end postprocess parity.

- Add mesh interpolate + `grid_sample` path or keep as host/PyTorch postprocess with a clear ABI.

Stage 5: Layout and fusion optimization.

- Add guarded NHWC CNN island rewrite only after axis rewrite tests exist.

Stage 6: Broader batching/dynamic shapes.

- Admit batch up to target deployment needs and optionally test Paddle hint shapes separately from native preprocessor defaults.

## 12. Parity and validation plan

- Unit parity: `UVDocConvLayer` with zero padding and reflect padding, with and without activation.
- Unit parity: residual block downsample and non-downsample cases for dilation 1, 3, 8, 18.
- Unit parity: PReLU head activation and learned slope loading.
- Single-stage parity: resnet_head, each resnet_down stage, each bridge block.
- Feature capture parity: verify six feature maps are produced from the same ResNet input and concatenated in `stage1..stage6` order.
- Full neural parity: compare `last_hidden_state` for `PaddlePaddle/UVDoc_safetensors` on the documented sample image; source test expects `[B,2,45,31]` and checks a 3x3 numeric slice.
- Postprocess parity: compare rectified image top-left pixels and full-image tolerance after `grid_sample`.
- Dtype parity: fp32 first; fp16/bf16 only after Conv/BN folding and grid-sample tolerances are decided. Transformers slow test exercises float32, float16, and bfloat16 model execution on accelerator.
- Recommended tolerances: fp32 neural mesh `rtol=2e-4, atol=2e-4` to match the source integration test; fp16/bf16 should use looser image-space checks after measuring PyTorch reference drift.

No DinoML tests were run for this audit, per task scope.

## 13. Performance probes

- Processor throughput: image decode + rescale/normalize + BGR gather + resize to 712x488.
- Neural graph throughput: Conv-heavy `[B,3,712,488] -> [B,2,45,31]` batch sweep.
- Bridge-only throughput: large-dilation Conv2d sweep over `[B,128,45,31]`.
- Postprocess throughput: mesh upsample + grid_sample for original image size sweep.
- Layout comparison: NCHW baseline versus guarded NHWC island, measured without host postprocess in the loop.
- Batch-size sweep: `B=1,2,4,8`.
- Dynamic input sweep only if bypassing the default processor resize or targeting Paddle deployment shapes.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Attention, KV cache, logits, generation, beam search.
- Tokenizer/text/OCR/layout-box handling; UVDoc consumes images only.
- Multi-GPU/tensor parallel.
- Quantized/packed weight formats; none are source-coupled in the native Transformers code.
- NHWC rewrite as a first milestone; start with faithful NCHW.
- End-to-end image postprocess can be deferred if first DinoML target is mesh prediction rather than rectified image bytes.

## 15. Final implementation checklist

- [ ] Parse `UVDocConfig` and nested `UVDocBackboneConfig`.
- [ ] Load `PaddlePaddle/UVDoc_safetensors` weights and preserve bridge/output feature order.
- [ ] Implement/fold Conv2d + BatchNorm2d inference.
- [ ] Implement ReLU and PReLU activations.
- [ ] Implement NCHW dilated Conv2d up to dilation 18.
- [ ] Implement residual add + activation blocks.
- [ ] Implement feature capture for six bridge outputs.
- [ ] Implement `cat(dim=1)` for bridge features.
- [ ] Implement head connector and mesh head.
- [ ] Add static neural parity test for `[1,3,712,488] -> [1,2,45,31]`.
- [ ] Add subgraph parity tests for ResNet, bridge, concat, and head.
- [ ] Decide whether DinoML owns postprocess or exposes mesh ABI.
- [ ] If owning postprocess, implement interpolate + grid_sample with `align_corners=True`.
- [ ] Add guarded NHWC rewrite tests covering `dim=1 -> dim=-1` axis rewrites.
- [ ] Benchmark NCHW baseline, dilated bridge, postprocess, and NHWC candidate.
