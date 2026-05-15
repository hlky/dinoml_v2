# RegNet DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/regnet-y-040 as the default/common example; additional sampled checkpoints listed below.
Config source: local Transformers RegNetConfig plus official Hugging Face config.json/preprocessor_config.json snapshots.
Source files inspected:
- transformers/src/transformers/models/regnet/modeling_regnet.py
- transformers/src/transformers/models/regnet/configuration_regnet.py
- transformers/src/transformers/models/regnet/convert_regnet_to_pytorch.py
- transformers/src/transformers/models/regnet/convert_regnet_seer_10b_to_pytorch.py
- transformers/tests/models/regnet/test_modeling_regnet.py, for output-shape expectations only
Any missing files or assumptions: no RegNet-specific image processor exists in-tree; checkpoints reuse generic image processor configs. All sampled official configs were accessible; no gated/401/403 gaps found.
```

Source URLs at the inspected commit:

- `modeling_regnet.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/regnet/modeling_regnet.py
- `configuration_regnet.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/regnet/configuration_regnet.py
- `convert_regnet_to_pytorch.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/regnet/convert_regnet_to_pytorch.py

Local snapshots were written under `agents/plans/transformers/regnet/_sources/`.

Sampled HF configs:

- `facebook/regnet-x-002`
- `facebook/regnet-x-320`
- `facebook/regnet-y-040`
- `facebook/regnet-y-320`
- `facebook/regnet-y-320-seer`
- `facebook/regnet-y-10b-seer-in1k`

Primary runtime target for this report: image classification and image feature extraction. `RegNetModel` is required for feature extraction/backbone-style use. `RegNetForImageClassification` adds a required classifier head for ImageNet-style checkpoints. Training losses are out of scope.

## 2. High-level architecture

RegNet is a convolutional vision encoder, not an attention model. The source contract is NCHW throughout the model body:

```text
CPU image preprocessing -> pixel_values[N, 3, H, W] -> stride-2 conv stem
  -> 4 RegNet stages of bottleneck residual blocks
  -> last feature map[N, C4, H/32, W/32]
  -> adaptive average pool[N, C4, 1, 1]
  -> optional flatten + classifier logits[N, num_labels]
```

Stage decomposition:

- CPU/data pipeline: resize/normalize image to `pixel_values`; sampled checkpoints use ImageNet mean/std and sizes 224 or 384.
- GPU encoder: stem `Conv2d+BatchNorm2d+ReLU`, four residual bottleneck stages.
- Optional head: adaptive average pool, flatten, dense classifier.
- Independently optimizable region: local Conv/BN/activation blocks can be folded/fused. NHWC is a guarded layout optimization for convolutional blocks, but public inputs/outputs and hidden-state returns remain NCHW unless a caller explicitly opts into a channel-last internal plan.

## 3. Important config dimensions

Defaults from `RegNetConfig`:

| Field | Default / behavior |
|---|---|
| `num_channels` | 3 |
| `embedding_size` | 32 |
| `hidden_sizes` | `(128, 192, 512, 1088)` |
| `depths` | `(2, 6, 12, 2)` |
| `groups_width` | 64 |
| `layer_type` | `"y"`; `"x"` has no SE, `"y"` inserts SE |
| `hidden_act` | `"relu"` |
| `downsample_in_first_stage` | `True` |
| attention/cache/positions | not applicable |
| classifier input width | `hidden_sizes[-1]` |

Representative checkpoint sweep:

| Checkpoint | Arch | Layer | Depths | Hidden sizes | `groups_width` | 3x3 groups by stage | Processor size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `facebook/regnet-x-002` | classifier | x | 1/1/4/7 | 24/56/152/368 | 8 | 3/7/19/46 | 224 |
| `facebook/regnet-x-320` | classifier | x | 2/7/13/1 | 336/672/1344/2520 | 168 | 2/4/8/15 | 224 |
| `facebook/regnet-y-040` | classifier | y | 2/6/12/2 | 128/192/512/1088 | 64 | 2/3/8/17 | 224 |
| `facebook/regnet-y-320` | classifier | y | 2/5/12/1 | 232/696/1392/3712 | 232 | 1/3/6/16 | 224 |
| `facebook/regnet-y-320-seer` | base model | y | 2/5/12/1 | 232/696/1392/3712 | 232 | 1/3/6/16 | 384 |
| `facebook/regnet-y-10b-seer-in1k` | classifier | y | 2/7/17/1 | 2020/4040/11110/28280 | 1010 | 2/4/11/28 | 384 |

Shape equations for an input `NCHW = [B, 3, H, W]` with default `downsample_in_first_stage=True`:

- Stem: `[B, 32, ceil(H/2), ceil(W/2)]` for odd sizes under PyTorch `Conv2d(k=3,s=2,p=1)`.
- Stages 1-4: first block stride is `[2, 2, 2, 2]`, so usual even-size output is `[B, hidden_sizes[i], H/(4*2^i), W/(4*2^i)]`.
- For 224: final map is `7x7`; for 384: final map is `12x12`.

## 3a. Family variation traps

- `layer_type="x"` and `"y"` use the same bottleneck skeleton, but Y inserts an SE block after the grouped 3x3 and before the final 1x1 projection.
- Grouped 3x3 convolution groups are computed as `max(1, out_channels // groups_width)`, not a directly stored per-stage list. DinoML should compute and validate divisibility per stage.
- `RegNetYLayer` SE reduction uses `reduced_channels = round(in_channels / 4)`, where `in_channels` is the block input width, not necessarily `out_channels`. First blocks in a stage can therefore use a different SE bottleneck width than later same-stage blocks.
- `downsample_in_first_stage` affects total stride. Official sampled configs set it `true`; source defaults also set `true`. A config with `false` would make the first stage stride 1 and final stride 16 rather than 32.
- Checkpoints without `in1k` SEER can use `RegNetModel` with no classifier architecture. Do not assume every RegNet repo has logits.
- The source has no backbone wrapper that returns named feature maps, but `output_hidden_states=True` returns NCHW feature maps before every stage plus final output.
- No attention, sequence packing, positions, tokenizers, or caches exist. These sections should remain explicitly not applicable.
- Initial translation should preserve NCHW axes. Candidate NHWC regions require axis rewrites for channel concat/mul/add, `BatchNorm2d`, adaptive average pool channel position, and hidden-state output materialization.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input validation on `pixel_values.shape[1] == num_channels`.
- NCHW `Conv2d` layout, optionally internal NHWC/channel-last for fused regions.
- Residual add with matching `[B, C, H, W]`.
- Channel-wise multiply for SE attention broadcast `[B, C, 1, 1] * [B, C, H, W]`.
- Flatten `[B, C, 1, 1] -> [B, C]`.
- Optional tuple/dict output assembly with hidden states.

Neural network primitives:

- `Conv2d(3 -> embedding_size, k=3, stride=2, pad=1, groups=1, bias=False)`.
- `BatchNorm2d(C)` in inference mode using running mean/var and affine weight/bias.
- ReLU activation.
- Bottleneck 1x1 convs: `Conv2d(in -> out, k=1, stride=1, bias=False)` and `Conv2d(out -> out, k=1, stride=1, bias=False)`.
- Grouped spatial conv: `Conv2d(out -> out, k=3, stride={1|2}, pad=1, groups=max(1,out/groups_width), bias=False)`.
- Shortcut projection when `in != out` or `stride != 1`: `Conv2d(in -> out, k=1, stride={1|2}, bias=False) + BN`.
- Y-only SE: adaptive average pool to `1x1`, `Conv2d(out -> round(in/4), k=1, bias=True)`, ReLU, `Conv2d(round(in/4) -> out, k=1, bias=True)`, sigmoid, broadcast multiply.
- Adaptive average pool to `1x1`.
- Classifier: `Linear(hidden_sizes[-1] -> num_labels)` when `num_labels > 0`, otherwise identity.

Attention / cache / positions:

- None required.

Preprocessing-coupled ops:

- Processor snapshot uses resize to integer `size`, resample 3, normalize with ImageNet mean/std. No RegNet-specific packing metadata enters the GPU graph.

Distributed/tensor-parallel ops:

- None required by source.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: [B, 3, H, W] NCHW
x = Conv2d(3 -> embedding_size, k=3, s=2, p=1, bias=False)
x = BatchNorm2d(embedding_size)
x = ReLU(x)
```

RegNet X bottleneck block:

```text
input: [B, Cin, H, W]
groups = max(1, Cout // groups_width)
main = ConvBNReLU(Cin -> Cout, k=1, s=1, groups=1)
main = ConvBNReLU(Cout -> Cout, k=3, s=stride, p=1, groups=groups)
main = ConvBN(Cout -> Cout, k=1, s=1, groups=1, activation=None)
residual = input if Cin == Cout and stride == 1 else ConvBN(Cin -> Cout, k=1, s=stride)
output = ReLU(main + residual)
```

RegNet Y bottleneck block:

```text
input: [B, Cin, H, W]
groups = max(1, Cout // groups_width)
se_mid = round(Cin / 4)
main = ConvBNReLU(Cin -> Cout, k=1, s=1)
main = ConvBNReLU(Cout -> Cout, k=3, s=stride, p=1, groups=groups)
gate = AdaptiveAvgPool2d(main)                       # [B, Cout, 1, 1]
gate = Conv2d(Cout -> se_mid, k=1, bias=True)
gate = ReLU(gate)
gate = Conv2d(se_mid -> Cout, k=1, bias=True)
gate = Sigmoid(gate)
main = main * gate
main = ConvBN(Cout -> Cout, k=1, s=1, activation=None)
residual = input if Cin == Cout and stride == 1 else ConvBN(Cin -> Cout, k=1, s=stride)
output = ReLU(main + residual)
```

Encoder:

```text
x = stem(pixel_values)
stage0: RegNetStage(embedding_size -> hidden_sizes[0], depth=depths[0], first stride=2 if downsample_in_first_stage else 1)
stage1: RegNetStage(hidden_sizes[0] -> hidden_sizes[1], depth=depths[1], first stride=2)
stage2: RegNetStage(hidden_sizes[1] -> hidden_sizes[2], depth=depths[2], first stride=2)
stage3: RegNetStage(hidden_sizes[2] -> hidden_sizes[3], depth=depths[3], first stride=2)
```

Classifier head:

```text
last_hidden_state: [B, C4, Hout, Wout]
pooled = AdaptiveAvgPool2d((1, 1))(last_hidden_state) # [B, C4, 1, 1]
logits = Linear(C4 -> num_labels)(Flatten(pooled))
```

## 6. Attention requirements

No attention is required for RegNet. There is no causal/noncausal attention, no cross-attention, no MHA/MQA/GQA, no masks, no packed/varlen sequences, no RoPE/ALiBi/relative bias, no FlashAttention/SDPA path, and no KV cache.

Feature caches are ordinary image feature tensors if an application chooses to reuse `RegNetModel` outputs; they are not model-managed caches.

## 7. Position encoding and custom math

No learned or analytic position encoding is present. Spatial position is represented only through convolution, stride, padding, and pooling.

SE gate math to reproduce:

```python
def regnet_y_se(x, w1, b1, w2, b2):
    # x: [B, C, H, W]
    gate = x.mean(dim=(-2, -1), keepdim=True)
    gate = relu(conv1x1(gate, w1, b1))
    gate = sigmoid(conv1x1(gate, w2, b2))
    return x * gate
```

Conv/BN inference folding:

```python
def fold_conv_bn(w, bn_weight, bn_bias, running_mean, running_var, eps):
    scale = bn_weight / sqrt(running_var + eps)
    w_folded = w * scale.reshape(-1, 1, 1, 1)
    b_folded = bn_bias - running_mean * scale
    return w_folded, b_folded
```

The BN epsilon is the PyTorch `BatchNorm2d` default because the source does not pass a config epsilon.

## 8. Preprocessing and input packing

Sampled preprocessor configs contain:

- `do_resize: true`
- `size: 224` for non-SEER sampled checkpoints; `size: 384` for sampled SEER checkpoints.
- `resample: 3`
- `do_normalize: true`
- `image_mean: [0.485, 0.456, 0.406]`
- `image_std: [0.229, 0.224, 0.225]`

There is no patch packing, token packing, attention mask, position ID, modality metadata, or placeholder token handling. The GPU graph begins with already-normalized `pixel_values` in NCHW `[B, 3, H, W]`.

For end-to-end classification parity, postprocessing is only `argmax` or score handling outside the model. There is no NMS, box decode, mask resize, or structured output logic.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d inference folding

Source pattern:

```text
Conv2d(..., bias=False) -> BatchNorm2d(C) -> optional ReLU
```

Replacement:

```text
Conv2d(..., bias=True, folded weights) -> optional ReLU
```

Preconditions:

- Inference mode only.
- BN running mean/variance and affine parameters are loaded.
- No training-time batch-stat updates.
- Preserve PyTorch BN epsilon/default behavior.

Shape equations:

- Weight `[Cout, Cin/groups, kh, kw]`; folded scale is `[Cout]`.

Failure cases:

- Training, unfrozen BN, missing running stats, or provider requiring unfused BN for numeric debugging.

Parity test sketch:

- Compare a random NCHW tensor through original Conv+BN and folded Conv for every stem/block/shortcut/SE-free conv path.

### Rewrite: 1x1 Conv2d -> per-pixel GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, k=1, stride=1, pad=0, groups=1)
```

Replacement:

```text
Reshape NCHW/NHWC pixels -> MatMul([B*H*W, Cin], W.T) -> BiasAdd(optional) -> Reshape
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Layout pass knows whether channels are at dim 1 or dim -1.

Weight transform:

```python
w = conv.weight.reshape(cout, cin)
```

Layout constraints:

- NCHW translation must gather/scatter channel dim 1. NHWC optimized region can use contiguous channel-last pixels.

Failure cases:

- Shortcut 1x1 with stride 2 cannot use this exact rewrite unless strided spatial sampling is handled separately.

Parity test sketch:

- Compare both layouts on a block's first/final 1x1 conv and SE 1x1 convs.

### Rewrite: static 3x3 Conv2d lowering

Source pattern:

```text
Conv2d(C -> C, k=3, stride={1|2}, pad=1, groups=G, bias=False)
```

Replacement:

```text
Im2Col/GatherWindows -> grouped GEMM or direct convolution kernel -> reshape
```

Preconditions:

- Static weights.
- Preserve PyTorch padding and output-size equation.
- Group count exactly `max(1, C // groups_width)` and `C % G == 0`.

Layout constraints:

- NHWC region is attractive for CUDA conv kernels but must be guarded around hidden-state outputs and public NCHW contracts.

Failure cases:

- Dynamic shape without a provider supporting PyTorch's convolution output sizing, or unsupported group count.

Parity test sketch:

- Sweep stage widths/group counts from representative configs, including `G=1`, `G=17`, `G=28`, and stride 2.

### Rewrite: guarded NCHW -> NHWC residual stage region

Source pattern:

```text
NCHW stage input -> Conv/BN/ReLU/SE/residual blocks -> NCHW stage output
```

Replacement:

```text
Transpose once to NHWC -> channel-last fused blocks -> transpose once at stage boundary if needed
```

Preconditions:

- All consumers inside the region are controlled.
- Hidden-state outputs are either disabled or materialized back to NCHW at the exact source boundary.
- Axis-sensitive operations are rewritten: `BatchNorm2d` channel axis 1 to channel-last, SE mean over spatial axes, channel broadcast multiply, residual add, and classifier flatten after pool.

Failure cases:

- `output_hidden_states=True` with consumers expecting NCHW, external hooks, or mixed-layout residual branches.

Parity test sketch:

- Compare `RegNetModel(..., output_hidden_states=True)` with each hidden-state tensor restored to NCHW.

## 10. Kernel fusion candidates

Highest priority:

- Conv/BN/ReLU folding and fusion: every conv layer uses BN, and most have ReLU immediately after it.
- Grouped 3x3 bottleneck convolution kernels: this is the main compute path; group counts vary by stage and checkpoint.
- Residual block fusion around `main + shortcut + ReLU`: avoids extra memory traffic on large feature maps.

Medium priority:

- SE micro-fusion: global average pool + two 1x1 convs + sigmoid + channel multiply is small but repeated in every Y block.
- 1x1 Conv/GEMM optimization: bottleneck entry/exit convs dominate channel mixing, especially for large SEER widths.
- NHWC/channel-last stage regions: likely important for CUDA conv throughput, but only under explicit layout guards.

Lower priority:

- Classifier head fusion: adaptive pool + flatten + linear is cheap relative to the encoder.
- Hidden-state materialization optimization: useful for feature-extraction workloads, not needed for plain classification with hidden states disabled.

## 11. Runtime staging plan

1. Parse `RegNetConfig`, compute per-stage groups, strides, SE presence, and classifier presence.
2. Load weights for `RegNetModel` and `RegNetForImageClassification`; verify Conv/BN/Linear tensor names and shapes.
3. Implement NCHW stem plus one X block and one Y block parity.
4. Implement full four-stage encoder, including optional `output_hidden_states`.
5. Add pooler and classifier head parity for ImageNet checkpoints.
6. Add Conv/BN folding and direct/grouped convolution provider selection.
7. Add guarded NHWC stage-region lowering with NCHW boundary materialization.
8. Add performance sweeps for small, common, large, and SEER variants.

Can stub initially:

- Training losses and labels.
- Non-classification postprocessing.
- NHWC optimization, as long as NCHW parity exists first.

## 12. Parity and validation plan

- Config parser parity: compute groups and strides for sampled configs and reject invalid group divisibility.
- Random op tests: Conv+BN folding, grouped 3x3 stride 1/2, shortcut projection, SE gate.
- Single-block parity: one X block and one Y block with first-block `Cin != Cout` and later-block `Cin == Cout`.
- Stage parity: each stage output shape and values for NCHW inputs.
- Full encoder parity: `last_hidden_state`, `pooler_output`, and `hidden_states` tuple count/order.
- Classifier parity: logits for `facebook/regnet-y-040` and an X checkpoint.
- End-to-end image parity: same processor output and logits/top-1 label on a fixed RGB image.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-5`; fp16/bf16 start at `atol=2e-2, rtol=2e-2` for full-network logits, with tighter per-op tolerances where accumulation order is fixed.

## 13. Performance probes

- CPU preprocessing throughput for size 224 vs 384.
- Encoder-only throughput for `regnet-x-002`, `regnet-y-040`, `regnet-y-320`, and `regnet-y-10b-seer-in1k`.
- Stage-by-stage latency and bandwidth, especially grouped 3x3 convs.
- Batch-size sweep for classification: 1, 8, 32, 64 if memory permits.
- Resolution sweep: 224, 384, and dynamic non-multiple-of-32 sizes to confirm output-shape math.
- NCHW baseline vs guarded NHWC stage-region plan.
- Conv+BN folded vs unfused provider path.
- SE overhead split for X vs Y variants.
- Hidden-states disabled vs enabled memory traffic.

## 14. Skip/defer list

- Training losses and label-driven regression/multi-label problem type paths.
- Weight conversion scripts and timm/VISSL checkpoint conversion.
- Multi-GPU tensor parallelism.
- Quantization.
- Dynamic external backbone APIs; RegNet source itself does not implement a named-feature backbone wrapper.
- NHWC-first semantic translation. Use guarded layout/fusion regions after NCHW parity.

## 15. Final implementation checklist

- [ ] Parse `RegNetConfig` and sampled checkpoint fields.
- [ ] Compute per-stage stride, depth, width, and group counts.
- [ ] Load Conv2d, BatchNorm2d, SE Conv2d, Linear weights.
- [ ] Implement NCHW Conv2d/BN/ReLU stem.
- [ ] Implement X bottleneck block.
- [ ] Implement Y bottleneck block with SE gate.
- [ ] Implement shortcut projection and residual add.
- [ ] Implement four-stage encoder and hidden-state tuple order.
- [ ] Implement adaptive average pool, flatten, classifier head.
- [ ] Add Conv+BN folding rewrite.
- [ ] Add 1x1 Conv2d-to-GEMM rewrite with layout guards.
- [ ] Add grouped 3x3 provider admission tests.
- [ ] Add guarded NHWC stage-region optimization.
- [ ] Add block, stage, full-encoder, and classifier parity tests.
- [ ] Benchmark grouped conv, SE overhead, NCHW vs NHWC, and hidden-state materialization.
