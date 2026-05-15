# PoolFormer DinoML operator audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: sail/poolformer_s12, sail/poolformer_s24, sail/poolformer_s36, sail/poolformer_m36, sail/poolformer_m48
Config source: official Hugging Face config.json and preprocessor_config.json snapshots under _sources/hf_configs
Source files inspected:
- transformers/src/transformers/models/poolformer/configuration_poolformer.py
- transformers/src/transformers/models/poolformer/modeling_poolformer.py
- transformers/src/transformers/models/poolformer/image_processing_poolformer.py
- transformers/src/transformers/models/poolformer/image_processing_pil_poolformer.py
Any missing files or assumptions: no remote code or gated files were required. The report targets image classification first. No segmentation or backbone-specific PoolFormer head is implemented in the inspected source.
```

Source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/poolformer/modeling_poolformer.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/poolformer/configuration_poolformer.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/poolformer/image_processing_poolformer.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/poolformer/image_processing_pil_poolformer.py

## 2. High-level architecture

PoolFormer is an image-only convolutional/metaformer encoder with no attention. It keeps NCHW tensors through the model:

```text
image preprocessing -> NCHW pixel_values -> 4 Conv2d patch/downsample stages
  -> repeated PoolFormer blocks per stage -> final GroupNorm -> spatial mean
  -> Linear classifier -> logits
```

Primary runtime target: `PoolFormerForImageClassification`.

Optional outputs: hidden feature maps after each encoder stage when `output_hidden_states=True`. These are NCHW image-like maps, not token sequences.

Not applicable: text generation, prefill/decode, KV cache, attention masks, position IDs, tokenizers, segmentation postprocessing, detection postprocessing.

## 3. Important config dimensions

Source defaults:

| Field | Effective value / meaning |
|---|---|
| `num_channels` | 3 |
| `num_encoder_blocks` | 4 |
| `depths` | `(2, 2, 6, 2)` |
| `hidden_sizes` | `(64, 128, 320, 512)` |
| `patch_sizes` | `(7, 3, 3, 3)` |
| `strides` | `(4, 2, 2, 2)` |
| `padding` | `(2, 1, 1, 1)` |
| `pool_size` | 3 |
| `mlp_ratio` | 4.0 |
| `hidden_act` | `gelu` |
| `drop_path_rate` | 0.0 |
| `use_layer_scale` | true |
| `layer_scale_init_value` | 1e-5 default |
| `cache support` | none in model source; historical config `use_cache` is ignored |
| `attention` | none |

Representative checkpoint sweep:

| Checkpoint | Depths | Blocks | Hidden sizes | MLP widths | Layer scale | Processor crop pct |
|---|---:|---:|---|---|---:|---:|
| `sail/poolformer_s12` | 2/2/6/2 | 12 | 64/128/320/512 | 256/512/1280/2048 | 1e-5 | 0.90 |
| `sail/poolformer_s24` | 4/4/12/4 | 24 | 64/128/320/512 | 256/512/1280/2048 | 1e-5 | 0.90 |
| `sail/poolformer_s36` | 6/6/18/6 | 36 | 64/128/320/512 | 256/512/1280/2048 | 1e-6 | 0.90 |
| `sail/poolformer_m36` | 6/6/18/6 | 36 | 96/192/384/768 | 384/768/1536/3072 | 1e-6 | 0.95 |
| `sail/poolformer_m48` | 8/8/24/8 | 48 | 96/192/384/768 | 384/768/1536/3072 | 1e-6 | 0.95 |

For a 224x224 input, the four patch/downsample stages produce spatial sizes 56x56, 28x28, 14x14, and 7x7 with the official patch/stride/padding values.

## 3a. Family variation traps

- `num_hidden_layers` appears in old checkpoint configs as 12, but current source uses `sum(depths)` and ignores `num_hidden_layers`.
- `use_cache` appears in old checkpoint configs, but the inspected PoolFormer source has no cache ABI.
- All tensors in modeling code are NCHW. NHWC is an optimization/layout-pass candidate only inside guarded Conv/GroupNorm/AvgPool/1x1 regions.
- `GroupNorm(1, C)` is used instead of LayerNorm. It normalizes over all `C*H*W` values per sample, not just channels, so it is not equivalent to channel-last LayerNorm.
- The token mixer is `AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False) - x`; `count_include_pad=False` changes border math.
- The classification head uses `norm(sequence_output).mean([-2, -1])`, so an NHWC layout pass must rewrite spatial axes from `[-2, -1]` to `[1, 2]` or keep the source region guarded.
- `PoolFormerOutput` applies DropPath inside the MLP around the activation and second conv, and `PoolFormerLayer` may apply a separate residual DropPath. In eval, DropPath is identity.
- There is no implemented segmentation/backbone class in `modeling_poolformer.py`; feature-map outputs can be composed by downstream code but are not a separate HF head here.
- `PoolFormerFinalPooler` references `config.hidden_size` but is not used by `PoolFormerModel` or `PoolFormerForImageClassification`; do not treat it as required runtime surface for image classification.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW tensor input/output contract.
- Shape formulas for Conv2d output: `H' = floor((H + 2p - k) / s) + 1`, same for width.
- Mean reduction over spatial axes `[-2, -1]`.
- Optional tuple accumulation for NCHW stage hidden states.
- Concatenation is not used in the inspected source.

Neural network primitives:

- Conv2d with bias:
  - Stage 1: `Conv2d(3 -> C1, kernel=7, stride=4, padding=2)`.
  - Stages 2-4: `Conv2d(Ci-1 -> Ci, kernel=3, stride=2, padding=1)`.
  - MLP conv1: `Conv2d(C -> 4C, kernel=1, stride=1, padding=0)`.
  - MLP conv2: `Conv2d(4C -> C, kernel=1, stride=1, padding=0)`.
- GroupNorm with `num_groups=1`, affine weight/bias, default `eps=1e-5`, input NCHW.
- GELU from `ACT2FN`.
- AvgPool2d with `kernel=pool_size`, `stride=1`, symmetric padding, `count_include_pad=False`.
- Elementwise subtract, multiply, residual add.
- Per-channel layer-scale parameters shaped `[C]`, broadcast as `[1, C, 1, 1]`.
- Linear classifier: `Linear(C_last -> num_labels)` after spatial mean.

Attention primitives:

- None required.

Position/rotary/relative-bias ops:

- None required.

Generation/cache ops:

- None required. Reject or ignore historical `use_cache` for this source basis.

Preprocessing-coupled ops:

- RGB conversion if requested by frontend processor.
- Resize with BICUBIC resampling after scaling target size by `1 / crop_pct`.
- Center crop to 224x224.
- Rescale and ImageNet mean/std normalization.
- Output `pixel_values` as channel-first image tensors.

## 5. Layer/block breakdown

For input `x: [B, 3, H, W]`, each stage first applies patch embedding/downsampling:

```text
stage i:
  x = Conv2d(C_in -> C_i, k=patch_sizes[i], s=strides[i], p=padding[i], bias=True)(x)
  x = Identity(x)  # PoolFormerEmbeddings norm_layer is not supplied by PoolFormerEncoder
```

PoolFormer block, repeated `depths[i]` times at stage width `C` and spatial size `S_h x S_w`:

```text
if use_layer_scale:
  y = GroupNorm(1, C)(x)
  y = AvgPool2d(k=pool_size, s=1, p=pool_size//2, count_include_pad=False)(y) - y
  x = x + DropPath(layer_scale_1[:, None, None] * y)

  y = GroupNorm(1, C)(x)
  y = Conv2d(C -> 4C, k=1, bias=True)(y)
  y = GELU(y)
  y = DropPath(y)          # identity in eval
  y = Conv2d(4C -> C, k=1, bias=True)(y)
  y = DropPath(y)          # identity in eval
  x = x + DropPath(layer_scale_2[:, None, None] * y)
else:
  x = x + DropPath(Pooling(GroupNorm(1, C)(x)))
  x = x + DropPath(MLP(GroupNorm(1, C)(x)))
```

Classification head:

```text
x = GroupNorm(1, C_last)(x)
x = mean(x, axes=[H, W])      # source axes [-2, -1], shape [B, C_last]
logits = Linear(C_last -> num_labels, bias=True)(x)
```

## 6. Attention requirements

No attention variant is required. There is no causal/noncausal attention, no self/cross-attention, no MHA/MQA/GQA, no masks, no RoPE/ALiBi/relative bias, no packed/varlen sequence metadata, no FlashAttention/SDPA dispatch, and no KV cache.

The relevant "token mixer" is local average pooling over image grids, with subtraction of the input feature map.

## 7. Position encoding and custom math

No explicit position embeddings are used. Spatial information comes from Conv2d patch/downsample geometry and local pooling.

Pool token mixer math to reproduce:

```python
def poolformer_token_mixer(x, pool_size):
    # x: [B, C, H, W], source semantics
    pooled = avg_pool2d(
        x,
        kernel_size=pool_size,
        stride=1,
        padding=pool_size // 2,
        count_include_pad=False,
    )
    return pooled - x
```

Layer scale is a learned per-channel vector and can be folded into adjacent elementwise scale only when the whole residual branch is controlled:

```python
def apply_layer_scale(y, scale):
    # y: [B, C, H, W], scale: [C]
    return y * scale.reshape(1, -1, 1, 1)
```

Nothing depends on dynamic batch metadata besides normal image shape and batch size.

## 8. Preprocessing and input packing

Current torchvision and PIL processors implement the same high-level contract:

```text
input images -> optional RGB conversion -> resize with crop_pct-adjusted size
  -> center crop -> rescale -> normalize -> pixel_values
```

Official preprocessor snapshots use legacy fields:

- `feature_extractor_type`: `PoolFormerFeatureExtractor`.
- `size`: `224`.
- `crop_pct`: 0.90 for S checkpoints, 0.95 for M checkpoints.
- `resample`: 3, BICUBIC.
- `image_mean`: `[0.485, 0.456, 0.406]`.
- `image_std`: `[0.229, 0.224, 0.225]`.
- `do_resize_and_center_crop`: true.
- `do_normalize`: true.

Runtime graph input is already normalized `pixel_values` in channel-first layout. Resize/crop/normalization should initially stay in the CPU/data pipeline. There are no placeholder tokens, packed patch rows, attention masks, modality IDs, or sequence descriptors.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d MLP -> spatial batched GEMM

Source pattern:

```text
Conv2d(C -> 4C, k=1) -> GELU -> Conv2d(4C -> C, k=1)
```

Replacement:

```text
NCHW/NHWC view as B*H*W rows -> MatMul + Bias -> GELU -> MatMul + Bias -> restore layout
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Both conv weights are dense and bias handling is preserved.
- Layout pass owns producer and consumer or emits explicit transposes.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels)
```

Failure cases: non-1x1 conv, grouped conv, uncontrolled consumer expecting NCHW.

Parity sketch: compare one MLP branch on random `[B, C, H, W]` for S and M widths in fp32, fp16, bf16.

### Rewrite: patch/downsample Conv2d -> im2col/window GEMM

Source pattern:

```text
Conv2d(C_in -> C_out, k=7/s=4/p=2) or Conv2d(C_in -> C_out, k=3/s=2/p=1)
```

Replacement:

```text
Pad -> WindowFlatten(NCHW or NHWC) -> MatMul(weight_flat.T) -> BiasAdd -> Reshape
```

Preconditions:

- Static or guarded dynamic spatial shapes.
- `dilation == 1`, `groups == 1`, dense weight, bias preserved.
- Padding matches PyTorch zero padding.
- Flatten order matches source Conv2d correlation weight layout `[out, in, kh, kw]`.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Layout constraints: direct NHWC translation is only safe for a local fully-owned patch/downsample region. Otherwise preserve source NCHW and let a later layout/fusion pass choose NHWC.

Failure cases: unsupported dynamic padding, alternate data layout with unrewritten axes, nonzero dilation/groups.

### Rewrite: pooling mixer fusion

Source pattern:

```text
GroupNorm(1, C) -> AvgPool2d(k=3, s=1, p=1, count_include_pad=False) -> subtract input -> layer scale -> residual add
```

Replacement:

```text
fused_norm_pool_sub_scale_residual(x)
```

Preconditions:

- `pool_size` odd and padding exactly `pool_size // 2`.
- Preserve `count_include_pad=False` border divisor semantics.
- Eval mode or DropPath proven identity.
- Source layout axes known. For NHWC, channel axis and spatial axes must be rewritten.

Failure cases: training DropPath enabled, different pooling size/padding, consumer requires materialized normalized tensor.

Parity sketch: include border-heavy tiny shapes such as 1x1, 2x3, and 7x7 because `count_include_pad=False` is easiest to get wrong at edges.

### Rewrite: guarded NHWC conv block region

Source pattern:

```text
NCHW Conv2d/GroupNorm/AvgPool/1x1 Conv/GELU/residual chain
```

Replacement:

```text
layout-enter -> NHWC kernels/fusions -> layout-exit
```

Preconditions:

- All consumers inside the region are owned, or layout-exit materializes NCHW before external consumers and hidden-state outputs.
- Rewrite channel axis for GroupNorm and layer scale broadcast.
- Rewrite classifier spatial mean axes from source `[-2, -1]` to NHWC `[1, 2]` if the head stays inside NHWC.

Failure cases: `output_hidden_states=True` requiring public NCHW maps without a layout exit, downstream feature consumers with NCHW contract, or any unrewritten `dim=1`/spatial axes.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d patch/downsample kernels for stage transitions. These dominate early high-resolution work and define the memory layout contract.
- GroupNorm(1, C) + pooling mixer + residual scale/add. PoolFormer has many repeated blocks and no attention, so this is the key model-specific hot path.
- 1x1 Conv MLP fused as GEMM/GELU/GEMM over spatial positions.

Medium priority:

- Global GroupNorm + spatial mean + Linear classifier, especially for batch throughput.
- NHWC guarded region for conv/pool/MLP blocks, with explicit layout exits for hidden states.
- Layer-scale multiply fused into residual branch epilogues.

Lower priority:

- DropPath support beyond eval identity, only useful for training parity.
- Processor acceleration for resize/crop/normalize, useful end-to-end but separable from model graph.

## 11. Runtime staging plan

1. Parse PoolFormer config and reject/ignore unsupported historical fields such as `use_cache`.
2. Load weights for patch embeddings, GroupNorms, layer-scale vectors, 1x1 MLP convs, and classifier.
3. Implement one NCHW stage with Conv2d patch embedding and one PoolFormer block.
4. Validate full encoder feature map parity for S12 at 224x224.
5. Add classification head parity for S and M widths.
6. Add optional `output_hidden_states=True` public NCHW feature-map outputs.
7. Add guarded Conv2d/1x1 lowering and pooling mixer fusion.
8. Add NHWC layout-pass candidates only inside controlled regions with explicit axis rewrites and layout exits.

Initial stubs: training loss paths, DropPath randomness, and processor GPU acceleration can be stubbed for inference-only classification.

## 12. Parity and validation plan

- Random tensor tests for `AvgPool2d(..., count_include_pad=False) - x`, including small border-sensitive images.
- Random tensor tests for `GroupNorm(1, C)` against source axes on NCHW.
- Single-block parity for S width `C=64` and M width `C=96`.
- Stage parity after each patch/downsample plus block group on 224x224.
- Full `PoolFormerModel` last feature map parity.
- `PoolFormerForImageClassification` logits parity with `num_labels=1000`.
- Processor parity for crop_pct 0.90 and 0.95 if end-to-end image tests are in scope.
- Suggested tolerances: fp32 `atol=1e-5, rtol=1e-5`; fp16/bf16 `atol=2e-2, rtol=2e-2` for full model, tighter for isolated ops where accumulation order is controlled.

## 13. Performance probes

- CPU preprocessing throughput for resize/crop/normalize at crop_pct 0.90 vs 0.95.
- Encoder-only throughput by checkpoint family: S12, S36, M48.
- Stage-by-stage timing at 224x224 to isolate high-resolution stage 1 vs deep stage 3.
- Pooling mixer microbenchmarks for `C=64/96/128/192/320/384/512/768`.
- 1x1 MLP GEMM throughput over spatial sizes 56, 28, 14, 7.
- Batch-size sweep for 1, 8, 32, 64.
- Image resolution sweep for non-224 inputs if dynamic shapes are admitted.
- NCHW baseline vs guarded NHWC region with layout-enter/layout-exit overhead.
- Hidden-state output overhead when all four stage maps must be materialized as NCHW.

## 14. Skip/defer list

- Training and loss computation.
- DropPath randomness in training mode.
- Gradient checkpointing and weight initialization behavior.
- Multi-GPU/tensor parallelism.
- Quantization and packed weights; no source-coupled packed format is used here.
- Segmentation/detection heads; none are implemented in the inspected PoolFormer source.
- GPU preprocessing; CPU/data pipeline preprocessing is enough for first graph parity.

## 15. Final implementation checklist

- [ ] Parse `PoolFormerConfig`, including `depths`, `hidden_sizes`, patch/downsample tuples, `pool_size`, `mlp_ratio`, `use_layer_scale`, and `layer_scale_init_value`.
- [ ] Ignore or reject non-runtime historical fields such as `use_cache` and `num_hidden_layers`.
- [ ] Load Conv2d, GroupNorm, layer-scale, and classifier weights.
- [ ] Implement NCHW Conv2d patch/downsample stages.
- [ ] Implement `GroupNorm(1, C)` with NCHW semantics.
- [ ] Implement PoolFormer pooling mixer with `count_include_pad=False`.
- [ ] Implement 1x1 Conv MLP with GELU and eval DropPath identity.
- [ ] Implement classifier head: final GroupNorm, spatial mean, Linear.
- [ ] Preserve public NCHW hidden-state outputs.
- [ ] Add guarded 1x1 Conv -> GEMM rewrite.
- [ ] Add guarded patch Conv2d -> window GEMM rewrite.
- [ ] Add pooling mixer fusion with border parity tests.
- [ ] Add NHWC layout-pass candidate guarded by axis rewrites and layout exits.
- [ ] Add one-block, per-stage, full-encoder, and logits parity tests.
- [ ] Benchmark S12/S36/M48 encoder throughput and NCHW vs NHWC guarded regions.
