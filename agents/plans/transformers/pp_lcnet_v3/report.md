# Transformers Audit: `pp_lcnet_v3`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: no released native Transformers checkpoint found; source docstring says PaddlePaddle/Not_yet_released
Config source: source defaults in configuration_pp_lcnet_v3.py; no Hub config.json found for pp_lcnet_v3
Source files inspected:
  X:/H/transformers/src/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py
  X:/H/transformers/src/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py
  X:/H/transformers/src/transformers/models/pp_lcnet_v3/modular_pp_lcnet_v3.py
  X:/H/transformers/src/transformers/models/pp_lcnet/image_processing_pp_lcnet.py
  X:/H/transformers/src/transformers/models/auto/auto_mappings.py
  X:/H/transformers/src/transformers/models/auto/modeling_auto.py
Any missing files or assumptions:
  No pp_lcnet_v3 image processor file exists. Auto image processor mapping only names pp_lcnet, not pp_lcnet_v3.
  No native classification head exists for pp_lcnet_v3 in this source; only PPLCNetV3Backbone is auto-mapped.
  modeling/configuration files are generated from modular_pp_lcnet_v3.py; the modular file is authoritative for future Transformers source edits.
```

Source URLs at the inspected commit:
- [configuration_pp_lcnet_v3.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py)
- [modeling_pp_lcnet_v3.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py)
- [modular_pp_lcnet_v3.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pp_lcnet_v3/modular_pp_lcnet_v3.py)

Hub sweep: HF connector search found only [longtc/PPLCNetV3](https://hf.co/longtc/PPLCNetV3), an unofficial PaddlePaddle-tagged repo with `.pdparams` files and no `config.json`. Snapshot: [hub_filelist_longtc_PPLCNetV3.txt](./hub_filelist_longtc_PPLCNetV3.txt). No gated official links were encountered; the official source placeholder is simply not released.

## 2. High-level architecture

Runtime contract: image CNN backbone feature extractor. This is not a Transformer attention model and not a text generation model.

```text
CPU/image preprocessing -> NCHW pixel_values -> stem Conv-BN ->
stage1..stage5 depthwise-separable reparameterized CNN blocks ->
selected NCHW feature maps
```

`PPLCNetV3Backbone` returns `BackboneOutput(feature_maps, hidden_states)`. The first useful DinoML target is backbone parity for `pixel_values -> feature_maps`, with configurable `out_features`/`out_indices`. Classification, OCR detection heads, and Paddle `.pdparams` conversion are outside this source family unless a later native head lands.

Stageable pieces:
- CPU/data pipeline: optional PP-LCNet image processor behavior if callers reuse the sibling `pp_lcnet` processor.
- GPU/runtime graph: pure NCHW CNN backbone.
- Independently validatable regions: stem, each stage, SE module, and output feature selection.

## 3. Important config dimensions

Source default snapshot: [source_default_config_snapshot.md](./source_default_config_snapshot.md).

| Field | Source default | Operator impact |
|---|---:|---|
| `model_type` | `pp_lcnet_v3` | AutoConfig key |
| `scale` | `1.0` | Multiplies channel widths before `make_divisible(..., divisor)` |
| `stem_channels` | `16` | Stem output channels before scaling |
| `stem_stride` | `2` | Initial spatial downsample |
| `block_configs` depths | `[1, 2, 2, 5, 4]` | 14 depthwise-separable blocks |
| `hidden_act` | `hardswish` | Used after most reparameterized branch sums |
| `reduction` | `4` | SE bottleneck channels are `C // 4` |
| `divisor` | `8` | Channel rounding guard |
| `conv_symmetric_num` | `4` | Four same-kernel Conv-BN branches per RepLayer |
| attention/cache fields | none | Not applicable |

Default 224x224, scale 1.0 feature-map sweep:

| Output | Channels | Spatial stride | Shape for `B=1` |
|---|---:|---:|---|
| stem | 16 | 2 | `[1,16,112,112]` |
| stage1 | 32 | 2 | `[1,32,112,112]` |
| stage2 | 64 | 4 | `[1,64,56,56]` |
| stage3 | 128 | 8 | `[1,128,28,28]` |
| stage4 | 256 | 16 | `[1,256,14,14]` |
| stage5 | 512 | 32 | `[1,512,7,7]` |

Representative config sweep is source-derived because no native checkpoint configs were available:

| Variant | Provenance | Scale | Stem C | Final C | Notes |
|---|---|---:|---:|---:|---|
| default source | source defaults | 1.0 | 16 | 512 | Only in-library default |
| x0.75 Paddle file name | unofficial Hub file name inference | 0.75 | 16 | 384 | Repo has `PPLCNetV3_x0_75_ocr_det.pdparams`, but no Transformers config |
| x0.5 template | source-supported scale inference | 0.5 | 8 | 256 | Channel widths rounded by divisor |
| x1.5 template | source-supported scale inference | 1.5 | 24 | 768 | Same topology, wider channels |

## 3a. Family variation traps

- `scale` and `divisor` mean channel widths are not simply the raw `block_configs` widths.
- `conv_symmetric_num` changes the number of parallel Conv-BN branches in every RepLayer. Default is 4, but source accepts other values.
- `stride == 2` RepLayers skip the final hardswish/learnable-affine activation block. Do not fuse activation unconditionally after branch sums.
- Depthwise RepLayers use `groups=in_channels`; pointwise RepLayers use `groups=1`.
- Kernel sizes are 3 or 5 in default blocks. All same-kernel branches use padding `kernel_size // 2`.
- SE exists only where `use_squeeze_excitation=True`, default first two stage5 blocks.
- The source graph is NCHW. NHWC/channel-last should be a guarded layout optimization, not semantic translation.
- `num_features` in `PPLCNetV3Backbone` uses `int(block[-1][2] * scale)` rather than `make_divisible`, while actual module channels use `make_divisible`. For non-integer or low scales, treat backbone metadata with caution and validate against actual tensor shapes.
- No native `PPLCNetV3ForImageClassification` class exists at this commit, despite sibling `pp_lcnet` having one.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW dense tensors.
- Feature tuple selection by stage name/index.
- Optional preprocessing resize, center crop, RGB-to-BGR channel reorder if DinoML owns sibling processor parity.

Neural network primitives:
- `Conv2d` NCHW, static weights, kernel 1/3/5, stride 1/2, padding 0/1/2, dilation 1.
- Depthwise `Conv2d` with `groups=C`.
- `BatchNorm2d` inference affine/mean/variance.
- Elementwise add for branch accumulation.
- Scalar learnable affine: `scale * x + bias`, where scale/bias are rank-1 one-element parameters.
- `hardswish`, `relu`, `hardsigmoid`.
- `AdaptiveAvgPool2d(output_size=1)`.
- Elementwise multiply for SE gating.

Attention, position, generation, cache, sparse, rotary, packed sequence, and KV-cache ops: not required.

Layout notes:
- Source tensors are NCHW from `pixel_values` through outputs.
- Candidate optimized layout: local NCHW->NHWC/channel-last regions around Conv-BN-activation chains, guarded so stage outputs returned to users preserve NCHW unless an ABI change is explicitly owned.

## 5. Layer/block breakdown

Stem:
```text
x: [B,3,H,W]
x = Conv2d(3 -> make_divisible(stem_channels*scale), k=3, stride=stem_stride, pad=1, bias=False)
x = BatchNorm2d(C)
```

Learnable RepLayer:
```text
branches = []
if out_channels == in_channels and stride == 1:
  branches += [BatchNorm2d(input)]
if kernel_size > 1:
  branches += [Conv2d(in -> out, k=1, stride=stride, pad=0, groups=groups, bias=False) + BN]
for i in range(conv_symmetric_num):
  branches += [Conv2d(in -> out, k=kernel_size, stride=stride, pad=(k-1)//2, groups=groups, bias=False) + BN]
y = sum(branches)
y = scalar_scale * y + scalar_bias
if stride != 2:
  y = scalar_scale2 * hardswish(y) + scalar_bias2
```

Depthwise-separable block:
```text
x = depthwise RepLayer(Cin -> Cin, groups=Cin, k=3/5, stride=1/2)
x = SE(x) or Identity(x)
x = pointwise RepLayer(Cin -> Cout, groups=1, k=1, stride=1)
```

SE module:
```text
s = AdaptiveAvgPool2d(x, 1)                  # [B,C,1,1]
s = Conv2d(C -> C//reduction, k=1, bias=True)
s = ReLU(s)
s = Conv2d(C//reduction -> C, k=1, bias=True)
s = Hardsigmoid(s)
y = x * s
```

## 6. Attention requirements

No attention is required. The primary target has:
- no causal or noncausal dense attention,
- no MHA/GQA/MQA,
- no masks,
- no packed/varlen attention metadata,
- no KV cache,
- no FlashAttention/SDPA path.

## 7. Position encoding and custom math

No positional encoding, RoPE, ALiBi, relative bias, or sequence position math exists.

Custom source math to preserve:

```python
def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
```

Inference-only branch reparameterization opportunity is custom but source does not pre-fuse it. DinoML can either execute the training-style multi-branch graph or fold branches offline under strict Conv-BN and padding guards.

## 8. Preprocessing and input packing

`pp_lcnet_v3` has no dedicated image processor mapping. If callers pair it with sibling `PPLCNetImageProcessor`, processor behavior is:
- input images converted to tensors in CHW order by the shared image backend,
- resize preserving aspect ratio using `resize_short=256`,
- optional `size_divisor` rounding,
- center crop to 224,
- rescale and normalize with mean `[0.406, 0.456, 0.485]` and std `[0.225, 0.224, 0.229]`,
- RGB to BGR by channel indexing `[2,1,0]`,
- returns `pixel_values`.

GPU graph input contract for the backbone remains `pixel_values: [B,3,H,W]` NCHW float tensor. There are no token IDs, placeholder tokens, packed patch rows, masks, or scatter updates.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference fold

Source pattern: `Conv2d(..., bias=False)` followed by `BatchNorm2d`.

Replacement:
```text
Conv2d(weight_fused, bias_fused)
```

Preconditions:
- inference/eval mode,
- frozen BN running mean/var, gamma, beta, epsilon,
- no training batch-stat dependence,
- Conv output consumed only by matching BN.

Weight transform:
```python
scale = gamma / sqrt(running_var + eps)
w_fused = conv_w * scale.reshape(-1, 1, 1, 1)
b_fused = beta - running_mean * scale
```

Parity test sketch: compare stem and one RepLayer before/after fold over random NCHW inputs, fp32 tolerance about `1e-5`.

### Rewrite: RepLayer branch fusion to one Conv2d

Source pattern: identity BN branch plus optional 1x1 Conv-BN branch plus N same-kernel Conv-BN branches, summed before learnable affine/activation.

Replacement:
```text
single Conv2d(k=kernel_size, stride=stride, padding=(k-1)//2, groups=groups) -> learnable affine -> optional activation affine
```

Preconditions:
- all branch Conv-BN pairs are folded,
- same stride/groups/output shape,
- 1x1 branch weights zero-padded into center of kxk kernel,
- identity BN branch converted to grouped identity kernel only when `in_channels == out_channels` and `stride == 1`,
- padding and dilation match source,
- dtype accumulation parity validated.

Failure cases:
- dynamic training BN,
- altered `num_conv_branches` with incompatible checkpoint names,
- non-default dilation or padding changes.

### Rewrite: SE 1x1 Conv on `[B,C,1,1]` to Linear

Preconditions:
- input spatial shape exactly 1x1 after adaptive average pool,
- NCHW layout known,
- groups=1, stride=1, padding=0.

Replacement: flatten `[B,C,1,1] -> [B,C]`, GEMM/linear, activation, GEMM/linear, hardsigmoid, reshape to `[B,C,1,1]`.

### Rewrite: guarded NHWC convolution island

Preconditions:
- entire Conv-BN-activation chain and its consumers are layout-controlled,
- stage output ABI either converts back to NCHW or declares channel-last output,
- axis-sensitive ops rewritten: BN channel axis `1 -> -1`, adaptive pool reduces H/W not C, SE channel gating broadcasts over H/W.

Failure cases:
- output `feature_maps` exposed mid-island in NCHW,
- caller-supplied strides/layout not controlled.

## 10. Kernel fusion candidates

Highest priority:
- Conv2d + BatchNorm2d fold for all branches. This removes many BN ops and simplifies the graph.
- RepLayer branch fusion. Default `conv_symmetric_num=4` means each RepLayer otherwise has 5 Conv-BN branches for k>1 plus optional identity BN.
- Depthwise Conv2d kernels for k=3/k=5, stride 1/2, NCHW first, with later NHWC optimization.

Medium priority:
- Conv + hardswish affine fusion after non-stride-2 RepLayers.
- SE pool + 1x1 conv/linear + activations + broadcast multiply.
- Pointwise 1x1 Conv lowering to GEMM where layout and batch/spatial flattening are controlled.

Lower priority:
- End-to-end channel-last island fusions.
- Feature-output tuple materialization elision.
- Paddle `.pdparams` conversion/loading.

## 11. Runtime staging plan

Stage 1: parse `PPLCNetV3Config` source-equivalent fields and validate stage/channel shapes.

Stage 2: load native Transformers weights when a real checkpoint appears; initially support source-default random-weight parity.

Stage 3: implement/fold Conv-BN, hardswish, scalar affine, depthwise/grouped conv, adaptive avg pool, and SE.

Stage 4: backbone parity for stem plus each stage with selected `out_features`.

Stage 5: RepLayer offline branch fusion and 1x1 Conv-to-GEMM rewrites.

Stage 6: layout optimization for guarded convolution islands.

Stage 7: add checkpoint-specific admission for Paddle-derived or native safetensors releases.

Stubbable initially: image processor parity, Paddle `.pdparams`, classification/detection heads, training paths.

## 12. Parity and validation plan

- Config parity: source default channel/spatial shape table for scale 1.0 and at least one non-default scale.
- Operator tests: Conv-BN fold, identity BN to kernel, 1x1 padding into kxk, hardswish, hardsigmoid, SE block.
- Single-block parity: one stride-1 block and one stride-2 block, with and without SE.
- Stage parity: compare outputs after stem and stages 1-5.
- Backbone parity: compare `feature_maps` selected by `out_features` and `out_indices`.
- Preprocessing parity only if DinoML owns processor: compare sibling `PPLCNetImageProcessor` output for RGB/BGR, resize, crop, normalize.
- Tolerances: fp32 `1e-5` to `1e-4`; fp16/bf16 expected looser, especially after BN folding and hardswish.

## 13. Performance probes

- Stem and each stage latency for NCHW input.
- Depthwise k=3 versus k=5 throughput, stride 1 versus 2.
- RepLayer before/after branch fusion.
- Conv-BN folded versus unfused graph load/run time.
- SE overhead at stage5 resolutions.
- Batch-size sweep: 1, 8, 32, 64.
- Resolution sweep: 224, 256, 320, dynamic non-square inputs if processor-free backbone admission allows them.
- NCHW versus guarded NHWC convolution island performance.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Attention, KV cache, generation, beam search.
- Native classification head for v3; absent in source.
- Paddle `.pdparams` import and OCR detection heads from unofficial Hub files.
- Full image processor ownership unless a native v3 processor/checkpoint requires it.
- Quantization and multi-GPU.
- Broad dynamic-shape image admission beyond validated convolution/pooling shape equations.

## 15. Final implementation checklist

- [ ] Parse `PPLCNetV3Config` fields: `scale`, `divisor`, `block_configs`, `stem_channels`, `stem_stride`, `reduction`, `hidden_act`, `conv_symmetric_num`
- [ ] Implement `make_divisible` channel derivation
- [ ] Admit NCHW `pixel_values`
- [ ] Load Conv2d, BatchNorm2d, scalar affine, and SE weights
- [ ] Implement/fold Conv2d + BatchNorm2d
- [ ] Implement depthwise/grouped Conv2d k=3/k=5 stride 1/2
- [ ] Implement hardswish and hardsigmoid
- [ ] Implement adaptive avg pool to 1x1
- [ ] Implement SE broadcast multiply
- [ ] Implement RepLayer branch-sum parity
- [ ] Add RepLayer branch-fusion rewrite with strict guards
- [ ] Add optional SE 1x1 Conv-to-Linear rewrite
- [ ] Add backbone `out_features`/`out_indices` selection parity
- [ ] Add stem/stage/backbone parity tests against Transformers
- [ ] Benchmark branch-fused versus unfused backbone
