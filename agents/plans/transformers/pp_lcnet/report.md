# PP-LCNet Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors plus variants below
Config source: local Transformers config defaults plus public Hugging Face config.json/preprocessor_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/pp_lcnet/modular_pp_lcnet.py
  X:/H/transformers/src/transformers/models/pp_lcnet/configuration_pp_lcnet.py
  X:/H/transformers/src/transformers/models/pp_lcnet/modeling_pp_lcnet.py
  X:/H/transformers/src/transformers/models/pp_lcnet/image_processing_pp_lcnet.py
  X:/H/transformers/tests/models/pp_lcnet/test_modeling_pp_lcnet.py
  X:/H/transformers/tests/models/pp_lcnet/test_image_processing_pp_lcnet.py
  X:/H/transformers/docs/source/en/model_doc/pp_lcnet.md
Any missing files or assumptions: no gated repos found; Hub repos are public. Generated files state that modular_pp_lcnet.py is authoritative for future edits.
```

Representative public Hub configs inspected:

| Model id | Repo SHA | Scope | Key variation |
|---|---:|---|---|
| `PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors` | `b2716954d2e321dd4436400b37584fab641027ab` | document orientation, 4 labels | source-default strides, 224 crop |
| `PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors` | `db46a3a25c8b3c88f86539fd28aceecc2a1b3ee1` | table type, 2 labels | source-default strides, 224 crop |
| `PaddlePaddle/PP-LCNet_x0_25_textline_ori_safetensors` | `1e6737131dedda1e87f7c01de171c20e81789c49` | text-line orientation, 2 labels | `scale=0.25`, asymmetric `[2, 1]` strides, 80x160 resize |
| `PaddlePaddle/PP-LCNet_x1_0_textline_ori_safetensors` | not separately fetched for metadata; config fetched | text-line orientation, 2 labels | `scale=1.0`, asymmetric `[2, 1]` strides, 80x160 resize |

Small config snapshots are in `config_snapshots.md`.

## 2. High-level architecture

PP-LCNet is a lightweight convolutional vision encoder with optional classification head. It is not an attention or language-generation model.

```text
CPU/image pipeline -> NCHW pixel_values -> stem Conv-BN-HardSwish ->
5 depthwise-separable convolution stages with optional SE ->
classification head: global average pool -> 1x1 conv -> HardSwish -> deterministic dropout scale -> flatten -> linear logits
```

Primary DinoML runtime target: `PPLCNetForImageClassification` for OCR-adjacent orientation/table classifiers. `PPLCNetBackbone` is independently stageable for feature-map extraction and downstream PaddleOCR-style composite models.

Stage decomposition:

| Stage | Owner | Runtime contract |
|---|---|---|
| Image decode/resize/crop/rescale/normalize/BGR reorder | CPU/data pipeline first | emits `pixel_values` as `[B, 3, H, W]` |
| Encoder | DinoML GPU/CUDA target | NCHW Conv2d/BatchNorm/activation/depthwise conv/SE |
| Classifier head | DinoML GPU/CUDA target | pooled logits in `last_hidden_state`, shape `[B, num_labels]` |
| Postprocessing | controller/app | argmax and label mapping; orientation rotation is outside model source |

## 3. Important config dimensions

Effective defaults come from `PPLCNetConfig.__post_init__` when omitted by checkpoint `config.json`.

| Field | Source default / observed | Runtime significance |
|---|---|---|
| `model_type` | `pp_lcnet` | native in-library source; no remote code required |
| `scale` | default/checkpoints `1.0`; x0.25 textline `0.25` | channels use `make_divisible(channel * scale, divisor)` |
| `divisor` | default `8`; omitted by sampled configs | channel rounding guard |
| `stem_channels` | default `16`; omitted by sampled configs | stem output channels, rounded |
| `stem_stride` | default `2`; omitted by sampled configs | first spatial downsample |
| `block_configs` | 5 stages; default or textline override | kernel, channels, stride, SE enable |
| `reduction` | `4` | SE bottleneck channels `C // 4` |
| `class_expand` | `1280` | 1x1 classifier expansion width |
| `hidden_act` | `hardswish` | Conv-BN activation and classifier activation |
| `hidden_dropout_prob` | `0.2` | source uses deterministic multiply by `0.8`, not stochastic dropout |
| `num_labels` | inferred from `id2label`: 2 or 4 | final `Linear(1280 -> num_labels)` |
| attention/cache fields | none | not applicable |

Default x1.0 encoder for 224x224 input:

| Feature | Shape after stage | Notes |
|---|---|---|
| input | `[B, 3, 224, 224]` | NCHW |
| stem | `[B, 16, 112, 112]` | 3x3 stride 2 |
| stage1 | `[B, 32, 112, 112]` | one 3x3 depthwise separable block |
| stage2 | `[B, 64, 56, 56]` | first block stride 2 |
| stage3 | `[B, 128, 28, 28]` | first block stride 2 |
| stage4 | `[B, 256, 14, 14]` | one 3x3 stride-2 block plus five 5x5 blocks |
| stage5 | `[B, 512, 7, 7]` | two 5x5 blocks, both with SE |
| logits | `[B, num_labels]` | stored as `last_hidden_state` |

Representative checkpoint sweep:

| Checkpoint | Input processor | Scale | Final encoder C | Final spatial inference | Labels |
|---|---|---:|---:|---|---:|
| x1.0 doc orientation | resize short 256, center crop 224 | 1.0 | 512 | 7x7 | 4 |
| x1.0 table classification | resize short 256, center crop 224 | 1.0 | 512 | 7x7 | 2 |
| x0.25 textline orientation | resize fixed 80x160, no crop | 0.25 | 128 | approximately 3x80 because later strides are `[2,1]` | 2 |
| x1.0 textline orientation | resize fixed 80x160, no crop | 1.0 | 512 | approximately 3x80 because later strides are `[2,1]` | 2 |

## 3a. Family variation traps

- Text-line checkpoints override `block_configs` with tuple strides `[2, 1]`; lowering must support Conv2d stride as an int or pair.
- `PPLCNetBackbone.num_features` uses `int(block[-1][2] * scale)` without `make_divisible`, while actual Conv2d channels use `make_divisible`; this differs for some scales. Trust tensor shapes, not only `num_features`.
- Source layout is NCHW throughout model code. NHWC/channel-last is only a guarded layout/fusion opportunity.
- Preprocessor configs say `image_mode=BGR` and `channel_first=false`, but source `_preprocess` returns tensor images indexed as `[C,H,W]` after RGB-to-BGR reorder. Treat model input as NCHW/BGR-normalized.
- `model_input_names` may include `original_image_size` in Hub processor config, but this source returns only `pixel_values` and the model consumes only `pixel_values`.
- Classification output ABI is `BaseModelOutputWithNoAttention.last_hidden_state`, not `ImageClassifierOutput.logits`; docs/tests use argmax on the tensor.
- `hidden_dropout_prob` is applied at inference as `x * (1 - p)`, not `nn.Dropout`.
- No attention, no token embeddings, no sequence length, no cache, no train/problem-type support in tests.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor input, optional output hidden-state capture.
- Tuple/int Conv2d stride handling, padding `kernel_size // 2`, dilation default 1.
- Flatten from `[B, 1280, 1, 1]` to `[B, 1280]`.
- Elementwise multiply for SE gating and deterministic dropout scaling.

Neural network primitives:

- `Conv2d(Cin -> Cout, k=3/5, stride=1/2 or [2,1], padding=k//2, groups=1, bias=False)`.
- Depthwise `Conv2d(C -> C, k=3/5, groups=C, bias=False)`.
- Pointwise `Conv2d(Cin -> Cout, k=1, bias=False)`.
- BatchNorm2d in inference mode; Conv+BN folding is high value.
- `HardSwish`, `ReLU`, `HardSigmoid`.
- `AdaptiveAvgPool2d(1)`.
- `Linear(1280 -> num_labels)`.

Preprocessing-coupled ops:

- Aspect-preserving resize to target short edge, optional fixed resize, center crop, rescale by `1/255`, normalize with BGR-ordered channels after RGB-to-BGR flip.

Attention/generation/position/cache/sparse/quantized ops:

- Not applicable for PP-LCNet primary target. There is no self-attention, cross-attention, RoPE, relative bias, KV cache, recurrent state, or generation controller.

Parameter aliasing:

- No tied embeddings or cross-layer sharing. BatchNorm `num_batches_tracked` is ignored on missing load for classification checkpoints.

## 5. Layer/block breakdown

Stem:

```text
x: [B, 3, H, W]
x = Conv2d(3 -> make_divisible(16*scale), k=3, stride=stem_stride, padding=1, bias=False)
x = BatchNorm2d(C)
x = HardSwish(x)
```

Depthwise separable block, repeated according to `block_configs`:

```text
for (k, cin, cout, stride, use_se):
  cin_s = make_divisible(cin * scale, divisor)
  cout_s = make_divisible(cout * scale, divisor)
  x = Conv2d(cin_s -> cin_s, k=k, stride=stride, padding=k//2, groups=cin_s, bias=False)
  x = BatchNorm2d(cin_s)
  x = HardSwish(x)
  if use_se:
    gate = AdaptiveAvgPool2d(1)(x)
    gate = Conv2d(cin_s -> cin_s//reduction, k=1, bias=True)(gate)
    gate = ReLU(gate)
    gate = Conv2d(cin_s//reduction -> cin_s, k=1, bias=True)(gate)
    gate = HardSigmoid(gate)
    x = x * gate
  x = Conv2d(cin_s -> cout_s, k=1, stride=1, padding=0, bias=False)
  x = BatchNorm2d(cout_s)
  x = HardSwish(x)
```

Classification head:

```text
x = AdaptiveAvgPool2d(1)(encoder_last)
x = Conv2d(final_c -> 1280, k=1, bias=False)(x)
x = HardSwish(x)
x = x * 0.8  # for hidden_dropout_prob=0.2
x = Flatten(start_dim=1)(x)
x = Linear(1280 -> num_labels)(x)
return BaseModelOutputWithNoAttention(last_hidden_state=x)
```

## 6. Attention requirements

No attention is required. PP-LCNet has no causal/noncausal attention, masks, packed/varlen metadata, sliding windows, relative position bias, RoPE, FlashAttention/SDPA dispatch, or KV cache. Backbone outputs are image feature maps, not token states for generation.

## 7. Position encoding and custom math

There is no position encoding. Spatial information is carried by convolution geometry.

Custom source math to reproduce:

```python
def make_divisible(value, divisor=8, min_value=None):
    min_value = divisor if min_value is None else min_value
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
```

SE and head math are ordinary bounded CNN math. HardSwish and HardSigmoid should match PyTorch semantics.

## 8. Preprocessing and input packing

CPU/data-pipeline first integration:

- Accept RGB images from PIL/torchvision-compatible inputs.
- Resize either by short edge (`resize_short=256`) or fixed `size={"height":80,"width":160}` when `resize_short=null`.
- Optional center crop: doc/table use 224; textline disables crop.
- Rescale by `0.00392156862745098`.
- Normalize with means `[0.406, 0.456, 0.485]` and stds `[0.225, 0.224, 0.229]`.
- Reorder RGB to BGR by channel index `[2, 1, 0]`.
- Emit `pixel_values` as `[B,3,H,W]`.

No token packing, placeholders, scatter, masks, segment IDs, position IDs, or multimodal metadata are present. Orientation correction and table/textline downstream use are postprocessing outside the model graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d + BatchNorm2d -> Conv2d folded weights

Preconditions:

- Inference mode with frozen BatchNorm running mean/variance/eps/affine.
- Applies to stem, depthwise, pointwise, and all non-SE conv layers followed immediately by BatchNorm2d.

Replacement:

```text
Conv2d(x, W, no_bias) -> BatchNorm(gamma,beta,mean,var,eps)
=> Conv2d(x, W_folded, b_folded)
```

Failure cases: training mode, unfrozen stats, or exporting hidden intermediates before BN.

Parity test sketch: random NCHW tensors for k=1/3/5, grouped and dense convs, compare fp32 at `rtol=1e-5, atol=1e-5`.

### Rewrite: 1x1 Conv2d at 1x1 spatial -> Linear

Preconditions:

- Input spatial dimensions are exactly 1x1 after `AdaptiveAvgPool2d(1)`.
- `groups=1`, `stride=1`, `padding=0`, `dilation=1`.

Replacement:

```text
[B,C,1,1] -> SqueezeHW -> MatMul(W.T) -> optional BiasAdd -> [B,Cout]
```

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels)
```

Failure cases: preserving Conv2d output rank for backbone consumers, dynamic spatial not pooled to 1x1.

### Rewrite: NCHW conv region -> channel-last fused region

Preconditions:

- Entire local region from input conv through stage/head is controlled.
- Axis-sensitive ops are rewritten: BatchNorm channel axis `1 -> -1`, AdaptiveAvgPool spatial axes `[2,3]`, Flatten after `[B,1,1,C]`, SE broadcast `[B,1,1,C]`.
- External ABI remains NCHW unless processor and consumers are also opted into channel-last.

Replacement: convert weights from OIHW to HWIO or provider-native conv layout and avoid transposes inside the region.

Failure cases: hidden-state outputs requested in NCHW, downstream `BackboneOutput` consumers expecting NCHW, or partial regions crossing unrewritten axis-sensitive ops. Use a conceptual `no_layout_translation()` guard at public model inputs/outputs and backbone feature-map outputs.

### Rewrite: fixed-size textline path shape specialization

Preconditions:

- Processor config exactly matches fixed 80x160 resize, no crop.
- `block_configs` contain `[2,1]` downsample strides.

Replacement: specialize output-shape planning and conv autotuning for height-downsampled, width-preserving feature maps.

Failure cases: caller overrides processor size or `block_configs`.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d + BatchNorm + HardSwish fusion for stem/depthwise/pointwise blocks; this is nearly the whole encoder.
- Depthwise Conv2d kernels for 3x3 and 5x5 NCHW/channel-last variants, including stride `[2,1]`.
- SE block fusion around global average pool, 1x1 convs, HardSigmoid, and channel multiply.

Medium priority:

- AdaptiveAvgPool2d(1) + 1x1 Conv + HardSwish + scalar multiply in classification head.
- Layout-specialized channel-last conv region with guarded NCHW public ABI.
- 1x1 Conv lowering to GEMM for pointwise convolutions and classifier expansion.

Lower priority:

- Backbone multi-output capture optimization for downstream composite models.
- Processor batching/group-by-shape throughput; useful end-to-end but outside core GPU graph.

## 11. Runtime staging plan

Stage 1: parse `PPLCNetConfig`, including default `block_configs`, `make_divisible`, and tuple strides.

Stage 2: load weights and run one Conv-BN-HardSwish block parity, including depthwise grouped conv.

Stage 3: implement full encoder parity for source-default x1.0 224x224 path.

Stage 4: implement classification head and output ABI as `[B,num_labels]` in `last_hidden_state`.

Stage 5: add text-line variants with fixed 80x160 processor and `[2,1]` stride support.

Stage 6: add `PPLCNetBackbone` feature-map output selection with NCHW feature maps.

Stage 7: add Conv+BN and Conv+BN+activation fusions, then optional channel-last region optimization.

Can be stubbed initially: image decode, orientation-rotation postprocess, `original_image_size`, training, hidden-state capture, and downstream PaddleOCR composite consumers.

## 12. Parity and validation plan

- Unit-test `make_divisible` for scales `0.25`, `1.0`, and edge channels where 10% rounding rule matters.
- Random op parity: Conv2d dense/depthwise with k=1/3/5, int stride and tuple stride; BatchNorm eval; HardSwish; HardSigmoid; AdaptiveAvgPool2d(1).
- Single block parity for one non-SE block and one SE block.
- Encoder parity after each stage for x1.0 224x224 and textline 80x160.
- Classification-head parity with random encoder outputs.
- End-to-end checkpoint parity against `PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors`; Transformers test expects logits shape `[1,4]` and approximate tensor `[[-0.3655, -1.0573, 2.4883, -1.0640]]` for the demo image with `rtol=2e-2, atol=2e-2`.
- Recommended tolerances: fp32 fused/unfused `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=2e-2, atol=2e-2` for end-to-end logits.

## 13. Performance probes

- Processor throughput: resize/crop/normalize/BGR reorder images/sec for doc/table and textline paths.
- Encoder-only throughput for batch sizes 1, 8, 32, 128 on 224x224 and 80x160 inputs.
- Stage-wise conv timing: stem, stage1-5, SE-only.
- Depthwise 3x3 vs 5x5 kernel timing, including stride `[2,1]`.
- NCHW vs guarded channel-last fused region comparison.
- Conv+BN folding speedup and numerical drift.
- Classification head timing separated from encoder.
- Backbone multi-output overhead when hidden states are requested.
- Memory bandwidth and activation footprint sweep for x0.25 and x1.0.

## 14. Skip/defer list

- Training, gradients, and gradient checkpointing.
- Attention, KV cache, generation, beam search, tokenization.
- General OCR pipeline behavior, orientation image rotation, table recognition, and text recognition after classification.
- `pp_lcnet_v3` and composite models that reuse PP-LCNet-like blocks; audit separately.
- Quantization/packed weights; sampled native checkpoints use safetensors but source has no custom quantized path.
- Multi-GPU/model parallelism; tests explicitly skip model parallelism.
- Full CPU image processing parity inside DinoML runtime; first integration can accept preprocessed `pixel_values`.

## 15. Final implementation checklist

- [ ] Parse `PPLCNetConfig` with defaults and strict 5-stage validation.
- [ ] Implement `make_divisible` channel rounding.
- [ ] Load Conv2d, BatchNorm2d, SE, 1x1 conv, and Linear weights.
- [ ] Implement NCHW Conv2d with dense and depthwise groups.
- [ ] Support tuple strides such as `[2, 1]`.
- [ ] Implement BatchNorm2d eval, HardSwish, HardSigmoid, ReLU.
- [ ] Implement AdaptiveAvgPool2d(1), channel multiply, flatten, and Linear head.
- [ ] Preserve classification output ABI as `last_hidden_state` logits.
- [ ] Add encoder stage parity tests for x1.0 224x224 and textline 80x160.
- [ ] Add checkpoint end-to-end parity for doc orientation and one textline model.
- [ ] Add Conv+BN folding rewrite with grouped-conv coverage.
- [ ] Add guarded NCHW-to-channel-last layout pass for fully controlled conv regions.
- [ ] Benchmark processor, encoder, head, and fused vs unfused kernels separately.
