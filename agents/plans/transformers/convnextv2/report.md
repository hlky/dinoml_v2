# ConvNeXtV2 Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/convnextv2-tiny-1k-224 as the common reference; sweep includes atto/tiny/base/large/huge variants.
Config source: official Hugging Face config.json and preprocessor_config.json files fetched from facebook/* repos.
Source files inspected:
- transformers/src/transformers/models/convnextv2/modeling_convnextv2.py
- transformers/src/transformers/models/convnextv2/configuration_convnextv2.py
- transformers/src/transformers/models/convnextv2/convert_convnextv2_to_pytorch.py
- transformers/src/transformers/models/convnext/image_processing_convnext.py
- transformers/src/transformers/models/convnext/image_processing_pil_convnext.py
- transformers/src/transformers/models/auto/image_processing_auto.py
- transformers/tests/models/convnextv2/test_modeling_convnextv2.py
Any missing files or assumptions: no gated/401/403 gaps were observed for the representative official repos. ConvNeXtV2 has no family-local image processor; AutoImageProcessor maps convnextv2 to the shared ConvNextImageProcessor.
```

Pinned source URLs:

- [modeling_convnextv2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/convnextv2/modeling_convnextv2.py)
- [configuration_convnextv2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/convnextv2/configuration_convnextv2.py)
- [image_processing_convnext.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/convnext/image_processing_convnext.py)

Local snapshots are under [agents/plans/transformers/convnextv2/_sources](H:/dinoml_v2/agents/plans/transformers/convnextv2/_sources).

## 2. High-level architecture

ConvNeXtV2 is a convolutional vision encoder, not an attention model. The primary DinoML target for this report is image classification inference with `ConvNextV2ForImageClassification`. `ConvNextV2Model` feature extraction and `ConvNextV2Backbone` feature-map output are useful optional targets.

```text
CPU/image pipeline -> pixel_values NCHW -> patch/stem Conv2d -> 4 ConvNeXtV2 stages
-> final spatial mean pool -> LayerNorm -> classifier Linear -> logits
```

Stage decomposition:

- CPU/data pipeline: resize, optional center crop depending on resolution path, rescale by `1/255`, normalize by ImageNet mean/std, emit `pixel_values` in channels-first layout.
- Stem: non-overlapping `Conv2d(Cin=3, Cout=C0, kernel=patch_size, stride=patch_size)`, then channels-first LayerNorm.
- Encoder stages: four stages. Stage 1 has no downsample; stages 2-4 apply channels-first LayerNorm plus stride-2 `Conv2d(kernel=2)` before repeated residual depthwise-conv blocks.
- Classification head: spatial mean over NCHW `H,W`, final `LayerNorm(C_last)`, `Linear(C_last -> num_labels)`.
- Backbone head: returns selected NCHW feature maps with per-output channels-first LayerNorm.

## 3. Important config dimensions

Source defaults from `ConvNextV2Config`:

| Field | Default | Source/runtime effect |
|---|---:|---|
| `model_type` | `convnextv2` | Auto model dispatch. |
| `num_channels` | 3 | Checked against `pixel_values.shape[1]`. |
| `patch_size` | 4 | Stem conv `kernel_size=stride=patch_size`. |
| `num_stages` | 4 | Encoder stage count; source expects hidden sizes/depths to match. |
| `hidden_sizes` | `[96, 192, 384, 768]` | Stage channel widths. |
| `depths` | `[3, 3, 9, 3]` | Residual block count per stage. |
| `hidden_act` | `gelu` | Activation between pointwise linears. |
| `layer_norm_eps` | `1e-12` | Final pooled LayerNorm only; internal LN uses hard-coded `1e-6`. |
| `drop_path_rate` | `0.0` | Training stochastic depth; inference is identity. |
| `image_size` | 224 | Metadata/default only for source forward; processor config controls actual image size. |
| `_out_features` / `_out_indices` | null | Backbone output selection. |

Representative checkpoint sweep from official `config.json` and `preprocessor_config.json`:

| Model id | Widths | Depths | Labels | Model `image_size` | Processor shortest edge | Notes |
|---|---:|---:|---:|---:|---:|---|
| [facebook/convnextv2-atto-1k-224](https://huggingface.co/facebook/convnextv2-atto-1k-224) | 40,80,160,320 | 2,2,6,2 | 1000 | 224 | 224 | Small/debug-friendly shape. |
| [facebook/convnextv2-tiny-1k-224](https://huggingface.co/facebook/convnextv2-tiny-1k-224) | 96,192,384,768 | 3,3,9,3 | 1000 | 224 | 224 | Common ImageNet-1k reference. |
| [facebook/convnextv2-base-22k-224](https://huggingface.co/facebook/convnextv2-base-22k-224) | 128,256,512,1024 | 3,3,27,3 | 1000 | 224 | 224 | Repo tag says 22k; inspected config head has 1000 labels. |
| [facebook/convnextv2-large-22k-384](https://huggingface.co/facebook/convnextv2-large-22k-384) | 192,384,768,1536 | 3,3,27,3 | 1000 | 224 | 384 | Larger processor resolution; config `image_size` remains 224. |
| [facebook/convnextv2-huge-22k-512](https://huggingface.co/facebook/convnextv2-huge-22k-512) | 352,704,1408,2816 | 3,3,27,3 | 1000 | 224 | 512 | Widest/highest-resolution sampled variant. |

All sampled preprocessors use `ConvNextImageProcessor`, bicubic resample id `3`, `do_resize=true`, `do_rescale=true`, `do_normalize=true`, `rescale_factor=1/255`, ImageNet mean/std, and `crop_pct=0.875`.

## 3a. Family variation traps

- `image_size` in sampled model configs is not the same as the processor's runtime shortest edge for 384/512 variants. Admission should use preprocessor/output tensor shape or caller tensor shape, not config `image_size` alone.
- The source tensor contract is NCHW at model boundaries and for feature-map outputs. NHWC is used only inside residual blocks after depthwise conv.
- Internal block LayerNorm and GRN operate on NHWC with channel axis `-1`; stem/downsample/backbone norms are channels-first wrappers that perform NCHW -> NHWC -> NCHW around `nn.LayerNorm`.
- `ConvNextV2GRN` is V2-specific custom math and should not be collapsed into ordinary LayerNorm or RMSNorm.
- `patch_size` accepts int/list/tuple by config type, but source passes it directly to `nn.Conv2d` for both kernel and stride. Non-square tuple patch sizes need shape equations that use per-axis divisibility.
- Downsample convs are `kernel=2, stride=2, padding=0`; source does not pad odd spatial dimensions. First integration should require spatial divisibility by 32 for common square inputs or match PyTorch floor conv shape exactly.
- `drop_path_rate` can be nonzero in config, but inference/eval path uses identity. Training stochastic behavior can be deferred.
- `out_features` changes `ConvNextV2Backbone` output maps and per-map normalization, but does not change encoder computation.
- No attention, tokens, RoPE, KV cache, masks, causal generation, or tokenizer coupling exists for the primary target.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation for `pixel_values` with channel count equal to `config.num_channels`.
- `permute(0,2,3,1)` and `permute(0,3,1,2)` around channels-last blocks.
- Mean reduction over spatial axes: source `last_hidden_state.mean([-2, -1])`.
- Optional hidden-state capture for `hidden_states` and backbone feature maps.

Neural network primitives:

- Stem `Conv2d(num_channels -> C0, kernel=patch_size, stride=patch_size, padding=0, groups=1, bias=True)`.
- Channels-first LayerNorm implemented as `NCHW -> NHWC -> LayerNorm(C) -> NCHW`; internal eps is `1e-6`.
- Stage downsample for stages 2-4: channels-first LayerNorm on `Cin`, then `Conv2d(Cin -> Cout, kernel=2, stride=2, padding=0, groups=1, bias=True)`.
- Depthwise conv per block: `Conv2d(C -> C, kernel=7, stride=1, padding=3, groups=C, bias=True)`.
- Channels-last LayerNorm per block: `LayerNorm(C, eps=1e-6)` over NHWC channel axis.
- Pointwise MLP per block as Linear over last dimension: `Linear(C -> 4C, bias=True)`, GELU, GRN, `Linear(4C -> C, bias=True)`.
- Residual add after optional drop path; inference drop path is identity.
- Final `LayerNorm(C_last, eps=config.layer_norm_eps)`.
- Classification head `Linear(C_last -> num_labels, bias=True)` or identity if `num_labels <= 0`.

Preprocessing-coupled ops:

- Resize policy: if target shortest edge `< 384`, resize shortest edge to `int(shortest_edge / crop_pct)` preserving aspect ratio, then center crop to square `shortest_edge`.
- If target shortest edge `>= 384`, resize directly to square `(shortest_edge, shortest_edge)` without the crop_pct branch.
- Rescale and normalize per channel using ImageNet constants.
- Emit `pixel_values` in channels-first shape `[B, 3, S, S]` for common sampled configs.

Attention/generation/cache ops: not applicable.

## 5. Layer/block breakdown

For input `pixel_values` shaped `[B, 3, H, W]`:

```text
Stem:
  x = Conv2d(3 -> C0, kernel=P, stride=P)(pixel_values)     # [B, C0, floor(H/P), floor(W/P)]
  x = LayerNorm_channels_first(C0, eps=1e-6)(x)

Encoder stage i, i=0:
  repeat depth[0] ConvNextV2Layer(C0)

Encoder stage i, i>0:
  x = LayerNorm_channels_first(C_{i-1}, eps=1e-6)(x)
  x = Conv2d(C_{i-1} -> C_i, kernel=2, stride=2)(x)
  repeat depth[i] ConvNextV2Layer(C_i)

ConvNextV2Layer(C):
  residual = x                                      # NCHW
  y = DepthwiseConv2d(C, kernel=7, padding=3)(x)    # NCHW
  y = permute(y, NCHW -> NHWC)
  y = LayerNorm(C, eps=1e-6)(y)
  y = Linear(C -> 4C)(y)
  y = GELU(y)
  y = GRN(4C)(y)
  y = Linear(4C -> C)(y)
  y = permute(y, NHWC -> NCHW)
  x = residual + y

Classification:
  pooled = mean(x, axes=[H,W])                      # [B, C_last]
  pooled = LayerNorm(C_last, eps=config.layer_norm_eps)(pooled)
  logits = Linear(C_last -> num_labels)(pooled)
```

Common shape examples:

- Tiny 224: `[B,3,224,224] -> [B,96,56,56] -> [B,192,28,28] -> [B,384,14,14] -> [B,768,7,7] -> [B,1000]`.
- Large 384: `[B,3,384,384] -> [B,192,96,96] -> [B,384,48,48] -> [B,768,24,24] -> [B,1536,12,12] -> [B,1000]`.
- Huge 512: `[B,3,512,512] -> [B,352,128,128] -> [B,704,64,64] -> [B,1408,32,32] -> [B,2816,16,16] -> [B,1000]`.

## 6. Attention requirements

No attention is required. ConvNeXtV2 has no causal/noncausal self-attention, cross-attention, MHA/MQA/GQA, attention masks, ALiBi/RoPE, packed sequence metadata, or KV cache. SDPA/FlashAttention compatibility is irrelevant for this family.

## 7. Position encoding and custom math

There is no positional embedding. Spatial position is represented implicitly by convolutions and pooling.

Custom GRN math from source, expressed compactly:

```python
def convnextv2_grn(x, gamma, beta):
    # x: NHWC, gamma/beta: [1, 1, 1, C]
    gx = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
    return gamma * (x * nx) + beta + x
```

GRN parameters are initialized to zero in the source init hook. At inference with trained weights, GRN requires `sqrt(sum(x*x) over H,W)`, mean over channels, division with epsilon, multiply, scale, bias, and residual add. In an NHWC optimized region, this is naturally channel-last; in a faithful NCHW graph it requires explicit axis translation or layout guards.

## 8. Preprocessing and input packing

The image processor is model-coupled for parity:

- Input images can arrive as PIL/NumPy/torch depending on backend; output is `pixel_values`.
- Torchvision backend groups images by shape for batched resize/normalization, then restores input order. This grouping is a CPU/preprocessing optimization and not part of the model graph.
- For 224 configs, shortest edge becomes `int(224 / 0.875) = 256`, then center crop to `224x224`.
- For 384 and 512 configs, source processor directly warps/resizes to square `384x384` or `512x512`; it does not run the crop_pct branch.
- Rescale and normalize are per-channel: `(image / 255 - mean) / std`.
- Runtime graph receives NCHW float `pixel_values`; no masks, grids, token IDs, packed rows, or sequence descriptors are consumed.

First DinoML integration can keep preprocessing outside the compiled graph and require ready-made NCHW `pixel_values`. End-to-end parity should include a separate processor parity test because 224 uses resize+crop while 384/512 use direct square resize.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Stem non-overlap Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(Cin -> C0, kernel=P, stride=P, padding=0, groups=1)
```

Replacement:

```text
WindowFlatten_NCHW_or_NHWC -> MatMul([Cin*P_h*P_w] -> C0) -> BiasAdd -> Reshape
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Spatial dimensions follow PyTorch conv floor semantics; for a pure patchify rewrite require divisibility by patch size, or keep guarded fallback.
- Weight transform: `w_flat = conv.weight.reshape(Cout, Cin * P_h * P_w)`; GEMM uses `w_flat.T` for row-major activations.
- Preserve source patch flatten order from NCHW convolution. For NHWC patch extraction, transform flatten order or weight layout accordingly.

Failure cases: tuple patch sizes without per-axis handling, non-divisible dynamic shapes if no floor-conv fallback exists, nonzero padding/dilation/groups.

Parity test sketch: random `[B,3,224,224]` and `[B,3,384,384]`, compare stem conv+LN against patchify-GEMM+LN in fp32, then fp16 with relaxed tolerance.

### Rewrite: Downsample Conv2d(kernel=2,stride=2) -> Linear/GEMM

Source pattern:

```text
LayerNorm_channels_first(Cin) -> Conv2d(Cin -> Cout, kernel=2, stride=2)
```

Replacement:

```text
LN -> 2x2 WindowFlatten -> MatMul([4*Cin] -> Cout) -> BiasAdd -> Reshape
```

Preconditions:

- `kernel=2`, `stride=2`, `padding=0`, `dilation=1`, `groups=1`.
- Spatial dimensions divisible by 2 for exact non-overlap windows, or guarded fallback matching conv floor output.
- Same flatten-order and weight transform rules as stem.

Failure cases: odd feature-map dimensions, alternate configs with different kernel/stride, consumers requiring NCHW layout immediately after downsample unless reshape returns NCHW or region remains guarded.

### Rewrite: Pointwise Linear on NHWC -> 1x1 Conv or Batched GEMM

Source pattern:

```text
NHWC LayerNorm -> Linear(C -> 4C) -> GELU -> GRN -> Linear(4C -> C)
```

Replacement:

```text
Flatten B*H*W rows -> GEMM -> GELU/GRN -> GEMM -> reshape NHWC
```

Preconditions:

- Input is truly NHWC contiguous or the lowering understands NHWC strides.
- Linear weights use PyTorch layout `[out_features, in_features]`.
- GRN remains between the two GEMMs and reduces over spatial axes, so it prevents fusing the full MLP into one GEMM chain.

Failure cases: translating this into NCHW without rewriting LayerNorm/GRN axes; dynamic `H,W` unsupported by reduction kernel; accidental channel-first LayerNorm.

### Layout optimization: Keep residual blocks NHWC internally

Source pattern:

```text
NCHW depthwise conv -> permute NHWC -> LN/Linear/GELU/GRN/Linear -> permute NCHW
```

Opportunity:

- If DinoML has NHWC depthwise conv and NHWC residual add, keep the whole block NHWC and remove both permutes.
- Stage boundaries can remain NCHW initially; later, entire stages could stay NHWC if downsample, hidden-state capture, pooling, and backbone output contracts are guarded.

Required axis rewrites:

- Depthwise conv: source NCHW `groups=C`; NHWC provider must treat channel axis as last.
- Block LayerNorm: source NHWC already normalizes axis `-1`; keep unchanged in an NHWC region.
- GRN: source `dim=(1,2)` and `mean(dim=-1)` already matches NHWC.
- Spatial global pool for classification: source NCHW mean axes `[-2,-1]`; NHWC rewrite must reduce axes `[1,2]`.
- Backbone feature maps and public `last_hidden_state`: source returns NCHW. Add a no-layout-translation guard or final NHWC->NCHW materialization for public outputs.

Failure cases:

- `output_hidden_states=True` or `ConvNextV2Backbone` selected outputs expose intermediate NCHW feature maps. Optimized NHWC internals must materialize these exactly.
- Any user-visible tensor shape assertions expect `[B,C,H,W]`.
- Channels-first LayerNorm wrappers in stem/downsample/backbone should be guarded unless replaced with an explicit NHWC equivalent plus matching output layout.

## 10. Kernel fusion candidates

Highest priority:

- DepthwiseConv2d 7x7 NHWC/NCHW provider. Every block uses it, and large variants have many stage-3 repeats.
- Channels-last LayerNorm + Linear(C -> 4C) for NHWC block tensors. This removes a memory pass before the main pointwise GEMM.
- GELU + GRN elementwise/reduction kernel. GRN is the distinctive ConvNeXtV2 cost beyond ConvNeXt; avoid decomposing it into many standalone kernels long-term.
- Pointwise GEMM/Linear for `B*H*W` rows. Stage-3 large/huge widths dominate arithmetic.

Medium priority:

- Stem/downsample non-overlap Conv2d-to-GEMM rewrites for providers that do not yet have strong conv coverage.
- Channels-first LayerNorm wrapper elimination inside guarded NHWC stage regions.
- Final spatial average pool + LayerNorm fusion for classification.

Lower priority:

- Classifier Linear fusion with final LayerNorm. It is one small `[B,C]` GEMM.
- DropPath support. It is identity for inference.
- Backbone multi-output normalization fusion. Useful for detector/segmentation composition, but optional for first image-classification parity.

## 11. Runtime staging plan

1. Parse `ConvNextV2Config` and load weights for `ConvNextV2ForImageClassification`; keep preprocessing external and accept NCHW `pixel_values`.
2. Implement faithful fp32 one-block parity: depthwise conv, NCHW/NHWC permutes, LayerNorm, Linear, GELU, GRN, residual add.
3. Run full encoder parity for tiny 224 with source layout, then add classifier head parity.
4. Add representative resolution/width coverage: atto 224, tiny 224, large 384, huge 512 shape-only or sampled layer parity.
5. Add `ConvNextV2Model` feature extraction outputs: `last_hidden_state`, `pooler_output`, optional `hidden_states`.
6. Add `ConvNextV2Backbone` only after feature-map output contracts and `out_features` selection are stable.
7. Introduce guarded NHWC optimizations for residual blocks/stages, with no-layout-translation guards around public NCHW outputs.
8. Add processor parity for end-to-end image classification if DinoML owns preprocessing; otherwise document processor as caller responsibility.

Initially stubbable: training loss, labels, stochastic DropPath, gradient checkpointing, backbone output selection beyond default, and CPU image preprocessing inside compiled artifacts.

## 12. Parity and validation plan

- GRN unit tests: random NHWC tensors for several `B,H,W,C`, compare custom implementation with PyTorch source in fp32 and fp16.
- Channels-first LayerNorm tests: compare wrapper `NCHW -> LN(C) -> NCHW` against source for stem/downsample/backbone shapes.
- Depthwise conv block test: one `ConvNextV2Layer` with random weights and inputs, compare output and intermediate NHWC shapes.
- Stage parity: stage 1 no-downsample and stage 2 downsample cases, including odd spatial dimensions if DinoML chooses PyTorch floor semantics instead of divisibility admission.
- Full model parity: `facebook/convnextv2-tiny-1k-224`, fixed image or random tensor, compare logits; source integration test expects logits shape `[1,1000]` and a known slice for the COCO fixture.
- Resolution sweep: 224, 384, 512 preprocessed `pixel_values`, validate final feature map sizes `/32`.
- Backbone parity: selected `out_features=["stage2","stage3","stage4"]`, verify tuple length, NCHW shapes, and normalized map values.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for logits/intermediate blocks; fp16/bf16 start with `rtol=5e-2, atol=5e-2` around GELU/GRN reductions, tighten after kernels are stable.

No DinoML tests were run for this audit by request.

## 13. Performance probes

- Processor throughput: resize+crop 224 versus direct square resize 384/512.
- Encoder-only throughput by model size: atto, tiny, base, large, huge.
- Block microbenchmarks by stage: depthwise 7x7, Linear C->4C, GRN, Linear 4C->C.
- Layout probe: faithful NCHW/NHWC permutes versus guarded NHWC block/stage execution.
- Resolution sweep: 224, 384, 512 at batch sizes 1, 8, 32.
- Feature-map output overhead: classification-only versus `output_hidden_states=True` and Backbone selected outputs.
- Conv provider comparison: direct Conv2d/depthwise provider versus im2col/GEMM rewrite for stem/downsample/depthwise.
- GRN decomposition versus fused reduction/elementwise kernel.

## 14. Skip/defer list

- Training losses and label handling.
- Stochastic DropPath behavior during training.
- Gradient checkpointing.
- Quantization and packed weights; source uses ordinary PyTorch dense weights.
- Attention, generation, cache, tokenizer, and sequence batching features; not part of this family.
- Backbone integration into DETR/MaskFormer-style downstream models until ConvNeXtV2 feature-map parity is done.
- Full preprocessing inside DinoML runtime; acceptable to start with caller-supplied `pixel_values`.
- Aggressive whole-model NHWC translation until public NCHW output guards are implemented.

## 15. Final implementation checklist

- [ ] Parse `ConvNextV2Config` fields: `num_channels`, `patch_size`, `hidden_sizes`, `depths`, `hidden_act`, `layer_norm_eps`, `num_labels`, `out_features/out_indices`.
- [ ] Load dense weights for stem conv, downsample convs, depthwise convs, LayerNorms, linears, GRN gamma/beta, final norm, and classifier.
- [ ] Implement NCHW Conv2d and depthwise Conv2d coverage needed by stem/stages.
- [ ] Implement channels-first LayerNorm wrapper and channels-last LayerNorm.
- [ ] Implement NHWC Linear over the last dimension or flatten-to-GEMM lowering.
- [ ] Implement GELU and residual add.
- [ ] Implement ConvNeXtV2 GRN with spatial L2 norm, channel mean, epsilon division, scale/bias, residual.
- [ ] Implement global spatial mean pool over NCHW `[H,W]`.
- [ ] Add one-block, one-stage, and full tiny-224 parity tests.
- [ ] Add resolution/width shape tests for 224/384/512 representative configs.
- [ ] Add image-classification logits parity against `facebook/convnextv2-tiny-1k-224`.
- [ ] Add optional `ConvNextV2Model` feature extraction output parity.
- [ ] Add optional `ConvNextV2Backbone` `out_features` NCHW feature-map parity.
- [ ] Add guarded stem/downsample Conv2d-to-GEMM rewrites.
- [ ] Add guarded NHWC residual-block layout optimization with public NCHW output materialization.
- [ ] Benchmark depthwise conv, pointwise GEMM, GRN, and layout translation overhead separately.
