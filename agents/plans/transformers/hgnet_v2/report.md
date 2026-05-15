# Transformers audit: hgnet_v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: hgnet_v2
Primary runtime target: HGNetV2Backbone feature-map extraction; image classification head is optional.
Config source: src/transformers/models/hgnet_v2/configuration_hgnet_v2.py plus representative HF config.json files.
Source files inspected:
- transformers/src/transformers/models/hgnet_v2/modular_hgnet_v2.py
- transformers/src/transformers/models/hgnet_v2/modeling_hgnet_v2.py
- transformers/src/transformers/models/hgnet_v2/configuration_hgnet_v2.py
- transformers/src/transformers/backbone_utils.py
- transformers/src/transformers/models/rt_detr/image_processing_rt_detr.py
- transformers/docs/source/en/model_doc/hgnet_v2.md
- transformers/tests/models/hgnet_v2/test_modeling_hgnet_v2.py
Representative HF configs:
- https://huggingface.co/ustc-community/hgnet-v2/raw/main/config.json
- https://huggingface.co/ustc-community/hgnet-v2/raw/main/preprocessor_config.json
- https://huggingface.co/ustc-community/dfine-small-coco/raw/main/config.json
- https://huggingface.co/ustc-community/dfine-large-coco/raw/main/config.json
- https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO/raw/main/config.json
- https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO/raw/main/config.json
Any missing files or assumptions:
- There is no hgnet_v2-specific image processor file; the standalone HF checkpoint uses RTDetrImageProcessor.
- modeling_hgnet_v2.py and configuration_hgnet_v2.py are generated from modular_hgnet_v2.py; future source edits should target the modular file.
- Intellindust DEIMv2 configs are not native Transformers HGNetV2Config JSON. They are useful variation evidence only, not directly loadable by the inspected hgnet_v2 source.
- No 401/403 gated Hugging Face configs were encountered. The Intellindust DEIMv2 repos inspected returned config JSON but no `preprocessor_config.json` at `main`; treat that as a composed-model metadata gap, not an hgnet_v2 source gap.
```

## 2. High-level architecture

HGNetV2 is a convolutional image backbone, not a Transformer decoder. The native family owns:

```text
image preprocessing -> NCHW pixel_values -> conv stem -> 4 CNN stages -> selected NCHW feature maps
```

Optional native head:

```text
last stage map -> AdaptiveAvgPool2d(1,1) -> Flatten -> Linear(num_labels) -> logits
```

Composed detection/OCR/document models use HGNetV2 only as a backbone. Their detector, OCR, recognition, reading-order, or layout heads are separate families and should compose this audit rather than expanding hgnet_v2 ownership. First useful DinoML target is backbone feature extraction with selected `out_features`, because D-FINE, DEIMv2, PP-OCRV5, and PP-DocLayout variants consume NCHW stage maps.

Independently stageable regions:

- CPU/data pipeline: resize/rescale/optional annotation conversion in RTDetrImageProcessor.
- GPU/runtime: NCHW Conv2d/BatchNorm2d/activation/pad/pool/concat/residual CNN graph.
- Cacheable output: feature-map tuple selected by `out_features`/`out_indices`.
- Optional classifier: global average pool plus dense head.

## 3. Important config dimensions

Source defaults:

| Field | Default | Runtime significance |
| --- | --- | --- |
| `num_channels` | `3` | NCHW input channel guard: source checks `pixel_values.shape[1]`. |
| `embedding_size` | `64` | Reported stem feature channel count in `num_features`; not used to build stem convs. |
| `hidden_sizes` | `[256,512,1024,2048]` | Reported backbone channels. Must match effective stage outputs for composed users. |
| `hidden_act` | `relu` | Passed through `ACT2FN`; common configs use ReLU. |
| `stem_channels` | `[3,32,48]` | Stem conv channel plan. |
| `stem_strides` | `[2,1,1,2,1]` | Stem spatial reduction; first and fourth stem convs usually downsample. |
| `stage_in_channels` | `[48,128,512,1024]` | Per-stage input channels. |
| `stage_mid_channels` | `[48,96,192,384]` | Repeated block intermediate channels. |
| `stage_out_channels` | `[128,512,1024,2048]` | Actual per-stage output conv channels. |
| `stage_num_blocks` | `[1,1,3,1]` | Number of HGNetV2BasicLayer blocks per stage. |
| `stage_numb_of_layers` | `[6,6,6,6]` | Number of conv sublayers inside each basic layer. |
| `stage_downsample` | `[False,True,True,True]` | Adds depthwise stride conv before stages 2-4. |
| `stage_downsample_strides` | `[2,2,2,2]` | Downsample stride; source accepts scalar or tuple/list. |
| `stage_light_block` | `[False,False,True,True]` | Stages 3-4 use pointwise + depthwise light subblocks. |
| `stage_kernel_size` | `[3,3,5,5]` | Spatial conv kernels inside stage sublayers. |
| `use_learnable_affine_block` | `False` | Optional scalar `scale*x+bias` after activated conv layers. |
| `out_features`/`out_indices` | last stage if omitted | BackboneMixin chooses feature tuple ABI. |

Representative checkpoint sweep:

| Repo/config | Scope | Stem | Stage out channels | Blocks | Layers/block | LAB | Outputs | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ustc-community/hgnet-v2` | native HF backbone | `[3,32,64]` | `[128,512,1024,2048]` | `[1,2,5,2]` | `[6,6,6,6]` | false | stage2-4 | B4-like standalone backbone; preprocessor is RTDetrImageProcessor at 640x640. |
| `ustc-community/dfine-small-coco` | nested D-FINE backbone | `[3,16,16]` | `[64,256,512,1024]` | `[1,1,2,1]` | `[3,3,3,3]` | true | stage2-4 | Smaller detection backbone; source-owned head is D-FINE, not hgnet_v2. |
| `ustc-community/dfine-large-coco` | nested D-FINE backbone | `[3,32,48]` | `[128,512,1024,2048]` | `[1,1,3,1]` | `[6,6,6,6]` | false | stage2-4 | Close to source defaults. |
| `Intellindust/DEIMv2_HGNetv2_ATTO_COCO` | external/native-DEIM config | name `Atto` | not encoded as HF HGNetV2Config | external | external | true | return_idx `[2]` | Variation evidence only; not directly implemented by inspected native source. |
| `Intellindust/DEIMv2_HGNetv2_N_COCO` | external/native-DEIM config | name `B0` | external | external | external | true | return_idx `[2,3]` | Useful composed usage signal; route through a separate DEIMv2 audit/admission path. |

## 3a. Family variation traps

- This is CNN-only despite living under Transformers: no token sequence, attention, RoPE, KV cache, generation, or text ABI.
- Source layout is NCHW throughout. A NHWC/channel-last path is an optimization only and must rewrite `dim=1` concat, BatchNorm2d channel axis, Conv2d weight layout, MaxPool2d axes, adaptive pooling axes, and feature-map consumer contracts.
- `hidden_sizes` are metadata for `BackboneMixin.channels`; actual convolution widths come from `stage_out_channels`. Reject or repair configs where they disagree.
- `embedding_size` is metadata only for the stem feature in `num_features`; stem output is `stem_channels[2]`.
- `depths`, `layer_type`, `downsample_in_bottleneck`, and `downsample_in_first_stage` appear in representative configs but the inspected hgnet_v2 modeling source does not read them for graph construction.
- `stage_names` length is derived from `len(depths)` while encoder construction is derived from `len(stage_in_channels)`. Native configs keep these aligned; DinoML should reject mismatches because feature selection could silently drift.
- `use_learnable_affine_block=True` adds scalar affine parameters after activated conv layers only. Conv layers with `activation=None` do not apply LAB.
- `stage_downsample_strides` may be scalar or tuple/list; tuple strides require Conv2d stride tuple support and shape inference.
- Even-size stem convs (`kernel_size=2`) rely on explicit right/bottom `F.pad` before the conv, while `HGNetV2ConvLayer` itself uses padding `(kernel_size-1)//2`, which is zero for `2`.
- Light blocks use pointwise conv with no activation, then depthwise spatial conv with activation. Do not fuse them as a single dense conv unless using a depthwise+pointwise pattern with correct order.
- Backbone output feature maps are captured through hidden states: `(embedding, stage1, stage2, stage3, stage4)`. Composed models often expect only stage2-4 or stage1-4.
- Some public HF models named HGNetV2 are `timm`, Keras, or external-library checkpoints; they are out of scope for this native Transformers audit unless separately allowed.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input shape guard on channel dimension.
- Constant right/bottom `pad` for stem branches: `F.pad(x, (0,1,0,1))`.
- `torch.cat(..., dim=1)` for stem branch merge and dense block feature aggregation.
- Tuple/list output assembly for selected feature maps.
- `Flatten` after `[B,C,1,1]` for classification head.

Neural network primitives:

- Conv2d NCHW, bias false, padding `(k-1)//2`, stride scalar/tuple, groups 1.
- Depthwise Conv2d via `groups=in_channels` and `in_channels == out_channels` for stage downsample and light blocks.
- BatchNorm2d inference using running mean/var, gamma, beta, epsilon from PyTorch module defaults unless overridden by composed families.
- Activations via `ACT2FN`, primarily ReLU in inspected configs.
- Optional learnable affine block: scalar parameter multiply plus scalar bias broadcast over NCHW.
- MaxPool2d `kernel_size=2`, `stride=1`, `ceil_mode=True`.
- AdaptiveAvgPool2d output `(1,1)`.
- Linear classifier `hidden_sizes[-1] -> num_labels` when `num_labels > 0`.
- Residual add inside repeated basic-layer blocks after the first block in a stage.
- Dropout exists for `drop_path`, but native HGNetV2Stage always uses default `0.0`; inference can treat it as identity.

Preprocessing-coupled ops:

- RTDetrImageProcessor default: resize to exact 640x640, rescale by `1/255`, no normalization, no pad for the standalone checkpoint.
- Optional object-detection annotation conversion lives in processor/training/postprocessing, not in the HGNetV2 neural graph.

Detection/OCR composed usage:

- HGNetV2 produces image-like NCHW maps consumed by D-FINE/DEIM/PP-OCR/PP-DocLayout families.
- End-to-end detection postprocessing, query decoding, score filtering, OCR text decoding, or document layout logic are required by those composed models but deferred from the hgnet_v2-owned operator surface.

## 5. Layer/block breakdown

Stem, for input `[B,3,H,W]`:

```text
x1 = Conv3x3(stride=stem_strides[0], 3 -> stem_mid, bias=false) -> BN -> ReLU -> optional LAB
x1p = pad_right_bottom(x1, +1,+1)
branch = Conv2x2(stride=stem_strides[1], stem_mid -> stem_mid/2) -> BN -> ReLU -> optional LAB
branch = pad_right_bottom(branch, +1,+1)
branch = Conv2x2(stride=stem_strides[2], stem_mid/2 -> stem_mid) -> BN -> ReLU -> optional LAB
pool = MaxPool2d(kernel=2, stride=1, ceil_mode=True)(x1p)
merged = cat([pool, branch], channel)
x = Conv3x3(stride=stem_strides[3], 2*stem_mid -> stem_mid) -> BN -> ReLU -> optional LAB
x = Conv1x1(stride=stem_strides[4], stem_mid -> stem_out) -> BN -> ReLU -> optional LAB
```

Stage `i`, for `i=0..3`:

```text
if stage_downsample[i]:
  x = depthwise Conv3x3(stride=stage_downsample_strides[i], C_in -> C_in, activation=None) -> BN
repeat stage_num_blocks[i]:
  x = HGNetV2BasicLayer(...)
```

Basic layer:

```text
identity = x
outputs = [x]
for layer_index in range(stage_numb_of_layers[i]):
  if stage_light_block[i]:
    x = Conv1x1(C_prev -> C_mid, activation=None) -> BN
    x = depthwise ConvKxK(C_mid -> C_mid, groups=C_mid) -> BN -> ReLU -> optional LAB
  else:
    x = ConvKxK(C_prev -> C_mid) -> BN -> ReLU -> optional LAB
  outputs.append(x)
x = cat(outputs, channel)
x = Conv1x1(C_in + L*C_mid -> C_out/2) -> BN -> ReLU -> optional LAB
x = Conv1x1(C_out/2 -> C_out) -> BN -> ReLU -> optional LAB
if block_index != 0:
  x = x + identity
```

Classification head:

```text
x = final_stage
x = AdaptiveAvgPool2d((1,1))(x)
x = flatten(x)
logits = Linear(hidden_sizes[-1], num_labels)(x)
```

## 6. Attention requirements

No attention is required for the primary target. HGNetV2Backbone sets `has_attentions = False`; tests skip attention outputs; model outputs use `BackboneOutput` or `ImageClassifierOutputWithNoAttention`.

Not applicable for hgnet_v2: causal masks, packed/varlen attention, KV cache, RoPE/ALiBi, FlashAttention, generation decode, cross-attention, and recurrent state.

Attention appears only in composed detector/layout families after the HGNetV2 feature maps. Those attention blocks belong to the corresponding family audits.

## 7. Position encoding and custom math

HGNetV2 has no learned absolute position table, RoPE, ALiBi, or relative-position bias. Spatial position is implicit in convolution/pooling.

Custom math to preserve:

```python
def hgnetv2_learnable_affine(x, scale, bias):
    # scale and bias are scalar parameters broadcast over NCHW.
    return scale * x + bias
```

The explicit stem padding is also semantically important:

```python
def pad_right_bottom_one(x):
    # PyTorch F.pad argument order for NCHW: left, right, top, bottom.
    return pad(x, left=0, right=1, top=0, bottom=1)
```

BatchNorm2d inference can be pre-folded into Conv2d weights/bias when all BN statistics and affine parameters are constant.

## 8. Preprocessing and input packing

Standalone `ustc-community/hgnet-v2` ships `preprocessor_config.json` with `RTDetrImageProcessor`:

- `size={"height":640,"width":640}`
- `do_resize=true`
- `do_rescale=true`, `rescale_factor=1/255`
- `do_normalize=false`
- `do_pad=false`, so no `pixel_mask` is produced for the default standalone path
- output tensor: `pixel_values` stacked as NCHW `[B,3,640,640]`
- `format="coco_detection"` and annotation conversion fields exist because the processor class is shared with detectors; labels are not consumed by HGNetV2Backbone inference

The model forward only consumes `pixel_values`. For composed detection/OCR/document models:

- OCR ownership: text recognition/detection preprocessing and postprocessing belong to PP-OCR family reports. HGNetV2 only supplies stage maps.
- Detection ownership: box decode, score thresholding, top-k selection, target-size scaling, and absence/presence of NMS belong to D-FINE/DEIM/RT-DETR/PP-* family reports. HGNetV2 only supplies the backbone maps.
- Feature contract: native backbone returns NCHW image-like maps, not flattened token sequences.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern:

```text
Conv2d(bias=False) -> BatchNorm2d -> optional activation -> optional LAB
```

Replacement:

```text
Conv2d(with folded bias) -> optional activation -> optional LAB
```

Preconditions:

- Inference mode with frozen BN running mean/var, gamma, beta, epsilon.
- No training-time batch statistics.
- Conv output channel equals BN channel.

Weight transform:

```python
scale = gamma / sqrt(running_var + eps)
w_fold = conv_w * scale[:, None, None, None]
b_fold = beta - running_mean * scale
```

Failure cases: mutable BN, detector family replacing BatchNorm with nonstandard class whose semantics are not identical, or dynamic weight updates.

Parity test: compare each folded conv block and full backbone feature maps at fp32 tolerance `rtol=1e-4, atol=1e-5`.

### Rewrite: Conv + BN + ReLU fusion

Source pattern:

```text
Conv2d -> BatchNorm2d -> ReLU
```

Replacement:

```text
FusedConvBNReLU
```

Preconditions:

- Same as BN fold.
- `activation == "relu"`.
- LAB disabled or kept as a separate post-fusion scalar affine.

Layout constraints: source is NCHW; NHWC fusion requires explicit layout region and channel-axis rewrite.

Failure cases: `activation=None`, non-ReLU `hidden_act`, LAB fusion not implemented, grouped/depthwise conv provider missing.

### Rewrite: 1x1 Conv2d -> GEMM

Source pattern:

```text
NCHW [B,C,H,W] -> Conv1x1(Cin,Cout,stride=1,padding=0,groups=1)
```

Replacement:

```text
reshape/permute local image positions to [B*H*W,Cin] -> GEMM -> reshape back
```

Preconditions:

- `kernel_size=1`, `stride=1`, `padding=0`, `groups=1`.
- Controlled layout conversion around the op.
- BN fold either fused into output bias or applied separately.

Failure cases: grouped/depthwise conv, non-unit stride, global NHWC translation not accepted by downstream consumer.

### Rewrite: depthwise Conv2d provider path

Source pattern:

```text
Conv2d(groups=Cin, Cin=Cout, k=3 or 5, stride=1 or 2)
```

Replacement:

```text
DepthwiseConv2d specialized kernel, optionally fused with BN/ReLU
```

Preconditions:

- `groups == in_channels == out_channels`.
- Static or guarded kernel/stride/padding.
- Correct NCHW or guarded NHWC layout.

Failure cases: general grouped conv where `groups` is neither 1 nor depthwise.

### Rewrite: local NHWC/channel-last fusion region

Source pattern:

```text
NCHW conv-heavy stem/stage region with BN/ReLU/depthwise conv/concat
```

Replacement:

```text
NCHW input -> one guarded transpose to NHWC -> channel-last conv/pool/concat -> transpose selected outputs back to NCHW if consumers require it
```

Preconditions:

- Entire region controlled through selected output feature maps.
- Axis rewrites are applied: concat `dim=1 -> dim=-1`; BN channel axis; Conv2d weight OIHW to HWIO/provider format; pooling spatial axes; adaptive pool axes.
- Feature-map ABI either remains NCHW by transposing outputs back, or composed consumers are audited for NHWC.

Failure cases:

- Output feature consumed by an unaudited NCHW detector/OCR head.
- Branch merge around stem padding/pooling not rewritten consistently.
- Dynamic shapes without layout-aware shape functions.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm2d inference fold for every conv block. This is the dominant parity-preserving optimization and simplifies later fusion.
- Fused ConvBNReLU for NCHW and/or guarded NHWC. HGNetV2 is mostly this pattern.
- Depthwise ConvBN and Depthwise ConvBNReLU for downsample/light blocks. Stages 3-4 rely heavily on light blocks with 5x5 depthwise conv.

Medium priority:

- Channel concat + 1x1 aggregation conv planning. Basic layers concatenate `C_in + L*C_mid` channels, then immediately apply 1x1 convs.
- Stem branch fusion around explicit pad, maxpool, concat, and stride-2 conv. This region controls early high-resolution cost and is axis-sensitive.
- 1x1 Conv2d via GEMM/CUTLASS or a pointwise-conv provider for aggregation and stem4.

Lower priority:

- AdaptiveAvgPool2d + Linear classifier fusion for image classification. Useful only for classifier checkpoints.
- Optional LAB fusion into activation epilogue or folded scalar output transform. Only configs with `use_learnable_affine_block=true` need it.
- Dropout/drop-path elimination. Native inference already has zero drop path.

## 11. Runtime staging plan

1. Parse `HGNetV2Config` and reject unsupported config mismatches: list lengths, non-NCHW input, metadata/channel disagreement, non-ReLU activation until ACT2FN parity exists.
2. Load weights for Conv2d, BatchNorm2d, optional LAB, and optional classifier head.
3. Implement one stem parity path with explicit pads, maxpool ceil mode, channel concat, and shape checks.
4. Implement `HGNetV2BasicLayer` and `HGNetV2Stage` for dense and light blocks, including depthwise conv.
5. Return `BackboneOutput`-equivalent feature-map tuple for `out_features`.
6. Add optional classifier head.
7. Add BN folding/fused conv provider paths.
8. Compose with D-FINE/DEIM/PP-OCR only after their separate audits validate feature-map stride/channel contracts.

Stub initially:

- Training losses and labels.
- Processor annotation conversion.
- Composed detector/OCR/document heads.
- Non-ReLU activations if no representative checkpoint requires them.
- External Intellindust-style config translation.

## 12. Parity and validation plan

- Config-only validation: instantiate source-equivalent static shapes from representative configs and verify stage channel/stride expectations without running Transformers inside DinoML.
- Unit parity for custom pieces against PyTorch reference: right/bottom pad, MaxPool2d ceil mode, depthwise Conv2d, BN fold, LAB.
- Stem parity: random `[B,3,H,W]`, include odd/even H/W around 32, 224, 640 because explicit pad and ceil pooling affect shapes.
- BasicLayer parity: dense block with `light_block=false`, then `light_block=true`, including residual block index >0.
- Backbone parity: compare selected feature maps for `ustc-community/hgnet-v2`, D-FINE-small backbone config, and D-FINE-large backbone config.
- Classifier parity: compare logits for optional `HGNetV2ForImageClassification`.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/BF16 after fused kernels `rtol=5e-3, atol=5e-3`, with separate BN-fold error audit.

## 13. Performance probes

- Processor throughput: resize/rescale to 640x640, with and without normalization/padding.
- Stem-only latency and bandwidth at `[B,3,640,640]`.
- Per-stage latency for stage1-4; split dense vs light blocks.
- Depthwise 5x5 conv throughput in stages 3-4.
- ConvBNReLU fused vs unfused comparison.
- NCHW vs guarded NHWC/channel-last region comparison, including transpose overhead and output transpose-back cost.
- Feature-output sweep: last stage only vs stage2-4 vs stage1-4.
- Batch sweep: 1, 2, 4, 8, 16 at 640x640.
- Resolution sweep: 224, 320, 640, and detector-specific sizes.
- Composed-model probe boundary: backbone-only vs downstream detector/OCR head time, once those families are audited.

## 14. Skip/defer list

- Training losses, label handling, and annotation conversion.
- Detector/OCR/document postprocessing, including box decode, score filters, NMS policy, text decoding, and reading order.
- Attention, KV cache, generation, RoPE, and tokenizers; not applicable.
- External `timm`, Keras, or Intellindust-native HGNetV2 bodies.
- General grouped convolution beyond groups=1 and depthwise.
- Non-ReLU activation fusion until a native checkpoint requiring it is admitted.
- Global NHWC ABI for composed models; keep as guarded local optimization.

## 15. Final implementation checklist

- [ ] Parse `HGNetV2Config` and validate stage list lengths.
- [ ] Validate `hidden_sizes`/`embedding_size` metadata against actual stem/stage channel config.
- [ ] Load Conv2d, BatchNorm2d, optional LAB, and optional classifier weights.
- [ ] Implement NCHW Conv2d groups=1 and depthwise Conv2d.
- [ ] Implement BatchNorm2d inference and BN-fold rewrite.
- [ ] Implement ReLU and optional scalar LAB.
- [ ] Implement stem explicit right/bottom pads, MaxPool2d ceil mode, and channel concat.
- [ ] Implement BasicLayer dense and light-block variants.
- [ ] Implement Stage downsample and residual block add.
- [ ] Implement Backbone feature-map selection by `out_features`/`out_indices`.
- [ ] Implement optional AdaptiveAvgPool2d + Flatten + Linear classifier.
- [ ] Add parity tests for stem, basic layer, stage, full backbone, and classifier.
- [ ] Add performance probes for fused ConvBNReLU, depthwise conv, and NCHW/NHWC guarded layouts.
