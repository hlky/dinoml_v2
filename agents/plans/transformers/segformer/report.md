# SegFormer Transformers Audit

## 1. Source basis

Transformers commit/version: local `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary target `nvidia/segformer-b0-finetuned-ade-512-512`; representative sweep also used `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`, `nvidia/segformer-b5-finetuned-ade-640-640`, `nvidia/mit-b0`, and `nvidia/mit-b5`.

Config source: official Hugging Face raw `config.json` and `preprocessor_config.json` for the model ids above. Label counts are inferred from `id2label` entries when `num_labels` is omitted.

Source files inspected:

- `transformers/src/transformers/models/segformer/configuration_segformer.py`
- `transformers/src/transformers/models/segformer/modeling_segformer.py`
- `transformers/src/transformers/models/segformer/modular_segformer.py`
- `transformers/src/transformers/models/segformer/image_processing_segformer.py`
- `transformers/src/transformers/models/segformer/image_processing_pil_segformer.py`

Any missing files or assumptions: `modeling_segformer.py`, `image_processing_segformer.py`, and `image_processing_pil_segformer.py` are generated from `modular_segformer.py`; future source edits should inspect the modular file first. This report targets inference for `SegformerForSemanticSegmentation`. Image classification via `SegformerForImageClassification` is documented as optional. Training losses, gradient checkpointing, and stochastic depth randomness are out of runtime scope.

## 2. High-level architecture

SegFormer is a vision-only Mix Transformer encoder plus task head. The semantic segmentation path is:

```text
image preprocessing -> NCHW pixel_values -> 4-stage MiT encoder -> multi-scale NCHW features -> all-MLP decode head -> low-resolution logits -> semantic segmentation postprocess
```

Encoder stages are independently structured:

```text
overlapping Conv2d patch embedding -> LayerNorm on tokens -> repeated efficient self-attention + Mix-FFN -> stage LayerNorm -> NCHW feature map
```

The decode head consumes all stage outputs. It projects each stage's channel dimension to `decoder_hidden_size`, upsamples every projected map to stage-0 resolution, concatenates features in reverse stage order, applies a 1x1 convolution plus BatchNorm plus ReLU, then a final 1x1 classifier. The model returns logits at approximately input resolution divided by 4. The processor postprocess optionally upsamples logits to per-image `target_sizes` and computes `argmax` class maps.

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize, rescale by `1/255`, ImageNet mean/std normalization, optional segmentation label reduction.
- Encoder: fully GPU-stageable after `pixel_values`; all stages can be validated independently because each emits one multi-scale feature map.
- Decode head: independently stageable once encoder hidden states are available.
- Postprocess: can run outside the compiled graph initially; it is deterministic bilinear resize plus `argmax`.

## 3. Important config dimensions

Source defaults from `SegformerConfig`:

| Field | Default |
| --- | --- |
| `num_channels` | 3 |
| `num_encoder_blocks` | 4 |
| `hidden_sizes` | `[32, 64, 160, 256]` |
| `depths` | `[2, 2, 2, 2]` |
| `num_attention_heads` | `[1, 2, 5, 8]` |
| `head_dim` | per stage `hidden_size / heads` |
| `sr_ratios` | `[8, 4, 2, 1]` |
| `patch_sizes` | `[7, 3, 3, 3]` |
| `strides` | `[4, 2, 2, 2]` |
| `mlp_ratios` | `[4, 4, 4, 4]` |
| `hidden_act` | `gelu` |
| `decoder_hidden_size` | 256 |
| `layer_norm_eps` | `1e-6` |
| `drop_path_rate` | `0.1` |
| `classifier_dropout_prob` | `0.1` |
| `reshape_last_stage` | `True` |
| `semantic_loss_ignore_index` | 255 |
| cache/generation | Not applicable |

Representative checkpoint sweep:

| Model id | Head | Labels | Hidden sizes | Depths | Heads | SR ratios | Decoder width | Processor resize | Dtype |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |
| `nvidia/segformer-b0-finetuned-ade-512-512` | semantic segmentation | 150 | 32/64/160/256 | 2/2/2/2 | 1/2/5/8 | 8/4/2/1 | 256 | 512 | float32 |
| `nvidia/segformer-b1-finetuned-cityscapes-1024-1024` | semantic segmentation | 19 | 64/128/320/512 | 2/2/2/2 | 1/2/5/8 | 8/4/2/1 | 256 | 1024 | float32 |
| `nvidia/segformer-b5-finetuned-ade-640-640` | semantic segmentation | 150 | 64/128/320/512 | 3/6/40/3 | 1/2/5/8 | 8/4/2/1 | 768 | 640 | float32 |
| `nvidia/mit-b0` | image classification | 1000 | 32/64/160/256 | 2/2/2/2 | 1/2/5/8 | 8/4/2/1 | 256 | not inspected | float32 |
| `nvidia/mit-b5` | image classification | 1000 | 64/128/320/512 | 3/6/40/3 | 1/2/5/8 | 8/4/2/1 | 768 | not inspected | float32 |

For the inspected segmentation processors, `do_resize=True`, `do_normalize=True`, ImageNet mean/std are `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`, and source defaults supply `do_rescale=True` with `rescale_factor=1/255` when the preprocessor config omits them.

## 3a. Family variation traps

- Stage dimensions are lists. `hidden_size`, heads, MLP size, patch embedding input channel count, and sequence reduction ratio are stage-specific.
- `hidden_size == num_heads * head_dim` in inspected configs, but DinoML should validate divisibility rather than assume it.
- Patch embeddings are overlapping convolutions with padding, not non-overlapping ViT patchify. Do not rewrite them to pure window flatten plus GEMM unless guarded for a different hypothetical config.
- Efficient self-attention reduces only key/value tokens when `sr_ratio > 1`; queries remain full-resolution for the stage.
- Stage 4 has `sr_ratio=1`, so no sequence-reduction conv is created there.
- Mix-FFN is not a plain two-linear MLP. It inserts a depthwise 3x3 Conv2d between `fc1` and GELU.
- Encoder layout alternates: stages receive NCHW feature maps, patch embedding emits token shape `(B, H*W, C)`, attention and MLP run on tokens, then stage output is usually restored to NCHW.
- `reshape_last_stage=False` is implemented and changes the last encoder output to token layout. The segmentation decode head has a compatibility branch that assumes square token maps via `int(sqrt(seq_len))`; non-square last-stage token maps would be unsafe for that branch.
- Segmentation decode head concatenates projected hidden states in reverse order with `torch.cat(all_hidden_states[::-1], dim=1)`. Channel order must be preserved for weight loading.
- Processor resize size is checkpoint-specific: common inspected values are 512, 640, and 1024.
- ADE-style `reduce_labels` may be enabled at preprocessing time for segmentation maps, not for runtime image inference. Background `0` is mapped through ignore index `255`.
- NCHW is the source semantic layout. NHWC/channel-last optimization is a guarded lowering choice only; all axis-sensitive `dim=1`, `flatten(2)`, `transpose(1, 2)`, Conv2d, BatchNorm2d, interpolate, and argmax axes must be rewritten together or protected by a no-layout-translation guard.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `pixel_values`.
- `flatten(start_dim=2)`, `transpose(1, 2)`, `permute(0, 3, 1, 2)`, `permute(0, 2, 3, 1)`, `reshape`, `view`, `contiguous`.
- Tuple/list capture of hidden states for decode head.
- `torch.cat(..., dim=1)` for channel concatenation.
- `mean(dim=1)` for optional image classification.
- `argmax(dim=0)` and `argmax(dim=1)` for semantic postprocess.

Neural network primitives:

- Conv2d patch embeddings:
  - Stage 0: `Conv2d(3 -> C0, kernel=7, stride=4, padding=3)`.
  - Stages 1-3: `Conv2d(Cprev -> Cstage, kernel=3, stride=2, padding=1)`.
- LayerNorm over token channel dimension with eps `1e-6`.
- Linear projections with PyTorch weight layout `[out_features, in_features]`.
- Depthwise Conv2d in Mix-FFN: `groups=hidden_features`, `kernel=3`, `stride=1`, `padding=1`.
- Sequence-reduction Conv2d: `Conv2d(C -> C, kernel=sr_ratio, stride=sr_ratio, padding=0, groups=1)`.
- GELU activation via `ACT2FN`.
- Dropout and DropPath are identity in inference.
- Decode head `Linear(Cstage -> decoder_hidden_size)` per stage.
- Bilinear `interpolate(..., align_corners=False)`.
- 1x1 Conv2d fuse: `decoder_hidden_size * 4 -> decoder_hidden_size`, bias false.
- BatchNorm2d on decoder channels, inference mode.
- ReLU.
- 1x1 Conv2d classifier: `decoder_hidden_size -> num_labels`.

Attention primitives:

- Non-causal multi-head self-attention.
- Separate Q, K, V, O Linear modules, all with default PyTorch bias enabled because `nn.Linear(...)` is called without `bias=False`.
- Sequence reduction before K/V projection for stages with `sr_ratio > 1`.
- Attention shape `(B, heads, query_tokens, kv_tokens)` and value matmul.
- Softmax over last dimension with fp32 softmax in eager path, cast back to query dtype.
- Optional backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, or FlexAttention compatibility.

Preprocessing-coupled ops:

- RGB conversion and input channel-order handling in image processor.
- Resize with bilinear interpolation for images.
- Rescale/normalize per channel.
- Optional segmentation map resize with nearest-exact interpolation and int64 labels.
- Optional label reduction for segmentation maps.

## 5. Layer/block breakdown

Let stage `s` have channels `C_s`, heads `A_s`, head dim `D_s = C_s / A_s`, spatial size `(H_s, W_s)`, and `N_s = H_s * W_s`.

Stage `s`, repeated 4 stages:

```text
input: stage 0 NCHW pixel_values, later stages NCHW feature maps
x_nchw = Conv2d(C_in -> C_s, kernel=patch_sizes[s], stride=strides[s], padding=patch_size//2)
H_s, W_s = spatial output shape
x = flatten spatial to (B, N_s, C_s)
x = LayerNorm(C_s)
for depth_s blocks:
  residual = x
  y = LayerNorm(C_s)(x)
  q = Linear(C_s -> C_s)(y).view(B, N_s, A_s, D_s).transpose(1, 2)
  if sr_ratio[s] > 1:
    kv = reshape y to NCHW
    kv = Conv2d(C_s -> C_s, kernel=sr, stride=sr)(kv)
    kv = flatten to (B, N_kv, C_s)
    kv = LayerNorm(C_s)(kv)
  else:
    kv = y
  k = Linear(C_s -> C_s)(kv).view(B, N_kv, A_s, D_s).transpose(1, 2)
  v = Linear(C_s -> C_s)(kv).view(B, N_kv, A_s, D_s).transpose(1, 2)
  attn = softmax((q @ k^T) * D_s^-0.5, dim=-1)
  y = (attn @ v).transpose(1, 2).reshape(B, N_s, C_s)
  y = Linear(C_s -> C_s)(y)
  x = residual + y
  residual = x
  y = LayerNorm(C_s)(x)
  y = Linear(C_s -> C_s * mlp_ratio[s])(y)
  y = reshape token map to NCHW
  y = depthwise Conv2d(kernel=3, stride=1, padding=1)(y)
  y = flatten to tokens
  y = GELU(y)
  y = Linear(C_s * mlp_ratio[s] -> C_s)(y)
  x = residual + y
x = LayerNorm(C_s)(x)
if stage is not last or reshape_last_stage:
  x = reshape(B, H_s, W_s, C_s).permute(0, 3, 1, 2).contiguous()
```

For B0 at 512x512 input, the stage spatial sizes are approximately `128x128`, `64x64`, `32x32`, and `16x16`. For B5 at 640x640, they are approximately `160x160`, `80x80`, `40x40`, and `20x20`. Exact output shape follows Conv2d floor arithmetic with padding.

Segmentation decode head:

```text
for each encoder feature map i:
  if last stage is token layout and reshape_last_stage is false:
    infer square H=W=sqrt(seq_len), reshape to NCHW
  tokens = flatten spatial and transpose to (B, H_i*W_i, C_i)
  tokens = Linear(C_i -> decoder_hidden_size)(tokens)
  map = transpose/reshape to (B, decoder_hidden_size, H_i, W_i)
  map = bilinear upsample to stage-0 spatial size
fused = cat([stage3, stage2, stage1, stage0], dim=1)
fused = Conv2d(decoder_hidden_size*4 -> decoder_hidden_size, kernel=1, bias=False)
fused = BatchNorm2d(decoder_hidden_size)
fused = ReLU(fused)
logits = Conv2d(decoder_hidden_size -> num_labels, kernel=1)
```

## 6. Attention requirements

SegFormer uses encoder-only, non-causal self-attention. There is no cross-attention, KV cache, generation mask, RoPE, ALiBi, relative bias, sliding-window attention, or decoder-time cache.

Attention is MHA with stage-specific head counts `[1, 2, 5, 8]`. It is not GQA/MQA. Q attends over all stage tokens. K/V optionally attend over spatially reduced tokens:

```text
query_tokens = H_s * W_s
kv_tokens = floor_conv(H_s, sr_ratio[s], stride=sr_ratio[s]) * floor_conv(W_s, sr_ratio[s], stride=sr_ratio[s])
```

For the inspected configs, `sr_ratios=[8,4,2,1]`, so the first three stages reduce K/V and the last stage does not. There is no attention mask in normal image inference. If an attention mask is supplied through generic kwargs, eager attention adds it before softmax.

Fused attention parity notes:

- Scaling is `head_dim ** -0.5`.
- Eager path computes `q @ k.transpose(-2, -1)`, adds mask if present, softmaxes in fp32, casts probabilities back to query dtype, applies dropout only in training, then multiplies by V.
- Source declares SDPA, FlashAttention, and FlexAttention support. DinoML can initially use unfused attention, then specialize efficient-attention kernels for reduced K/V sequence length.

## 7. Position encoding and custom math

There are no learned absolute position embeddings, RoPE, ALiBi, or relative bias tensors in the inspected source. Position information comes from convolutions:

- Overlapping patch embedding Conv2d at every stage.
- Mix-FFN depthwise 3x3 Conv2d between the two linear layers.

Short source-equivalent snippets:

```python
def sequence_reduce(tokens, height, width, conv, norm):
    b, n, c = tokens.shape
    x = tokens.transpose(1, 2).reshape(b, c, height, width)
    x = conv(x)  # kernel=stride=sr_ratio, no padding
    x = x.reshape(b, c, -1).transpose(1, 2)
    return norm(x)
```

```python
def mix_ffn(tokens, height, width, fc1, dwconv, act, fc2):
    x = fc1(tokens)
    b, n, c = x.shape
    x = x.transpose(1, 2).view(b, c, height, width)
    x = dwconv(x)  # depthwise 3x3
    x = x.flatten(2).transpose(1, 2)
    return fc2(act(x))
```

Convolution weights can be constant-loaded. Spatial sizes depend on runtime image size after preprocessing and Conv2d floor arithmetic.

## 8. Preprocessing and input packing

The runtime model input is `pixel_values` in NCHW format. The image processor handles CPU/data-pipeline work:

```text
image(s) -> convert/prepare image tensor -> resize to processor size -> rescale by 1/255 -> normalize by ImageNet mean/std -> batch as pixel_values
```

Observed processor sizes are `512`, `640`, and `1024` for the inspected segmentation checkpoints. Source defaults are square `512x512`, bilinear image resize, `do_rescale=True`, `do_normalize=True`, `image_mean=IMAGENET_DEFAULT_MEAN`, and `image_std=IMAGENET_DEFAULT_STD`.

For segmentation maps used with labels, processor behavior differs from images:

- Expected as 2D maps, no RGB conversion.
- Resize uses nearest-exact interpolation.
- No rescale or normalization.
- Convert/squeeze to int64.
- If `do_reduce_labels=True`, source maps label `0` to `255`, subtracts 1, then maps `254` back to `255`.

Postprocessing for end-to-end semantic segmentation:

```text
outputs.logits: (B, num_labels, H/4, W/4)
if target_sizes:
  for each image:
    logits_i = bilinear interpolate to (target_h, target_w), align_corners=False
    semantic_map_i = argmax(logits_i[0], dim=0)
else:
  semantic_maps = argmax(logits, dim=1), split by batch
```

There is no NMS, score thresholding, mask cropping, or variable-length detection output. Outputs are dense class-id maps.

## 9. Graph rewrite / lowering opportunities

### Rewrite: overlapping Conv2d patch embedding -> direct Conv2d provider

Source pattern: NCHW `Conv2d(C_in -> C_out, kernel=7 or 3, stride=4 or 2, padding=kernel//2)` followed by flatten/transpose and LayerNorm.

Replacement: lower as a real Conv2d provider, optionally fused with NCHW-to-token layout and LayerNorm.

Preconditions:

- `groups == 1`, `dilation == 1`.
- Source NCHW semantics preserved.
- Dynamic spatial shape uses Conv2d output floor formula.

Shape equations:

```text
H_out = floor((H_in + 2*pad - kernel) / stride) + 1
W_out = floor((W_in + 2*pad - kernel) / stride) + 1
tokens = H_out * W_out
```

Weight transform: none for Conv2d provider; PyTorch weight layout is `[C_out, C_in, KH, KW]`.

Layout constraints: an NHWC lowering may transform weights to provider layout, but the following `flatten(2).transpose(1,2)` and LayerNorm over channels must be rewritten consistently.

Failure cases: cannot rewrite to non-overlap patchify/GEMM because padding and overlap are required.

Parity test sketch: compare stage patch embedding output `(tokens, H, W)` against Transformers for odd and even input sizes, including 512 and 640.

### Rewrite: sequence-reduction Conv2d + K/V projection

Source pattern: token `(B,N,C)` -> NCHW -> `Conv2d(C -> C, kernel=stride=sr)` -> tokens -> LayerNorm -> K and V Linear.

Replacement: fused spatial reduction provider producing reduced tokens, followed by two GEMMs, or a conv-plus-two-projection fused kernel later.

Preconditions:

- `sr_ratio > 1`.
- Input tokens must correspond exactly to `(height,width)` passed alongside the block.
- Conv has `padding=0`, `groups=1`, `dilation=1`, `kernel=stride=sr_ratio`.

Shape equations:

```text
H_kv = floor((H - sr) / sr) + 1
W_kv = floor((W - sr) / sr) + 1
N_kv = H_kv * W_kv
```

Weight transform: Conv2d weight unchanged for Conv2d provider; K/V Linear use `[C, C]`.

Layout constraints: strongly axis-sensitive. Protect the token-to-NCHW reshape unless the whole block has a proven NHWC/channel-last plan.

Failure cases: `sr_ratio == 1` has no sequence reduction module; use original tokens for K/V.

Parity test sketch: stage-specific tests for SR ratios 8, 4, 2, and 1, checking attention K/V token counts.

### Rewrite: Mix-FFN depthwise conv region

Source pattern: `Linear(C -> C*mlp_ratio)` -> token-to-NCHW -> depthwise Conv2d 3x3 -> tokens -> GELU -> `Linear(C*mlp_ratio -> C)`.

Replacement: GEMM -> depthwise Conv2d provider -> fused GELU/GEMM epilogue where available.

Preconditions:

- Depthwise Conv2d has `groups == hidden_features`, `kernel=3`, `stride=1`, `padding=1`.
- Token count equals `height * width`.
- Dropout is disabled for inference.

Shape equations:

```text
hidden_features = C * mlp_ratio
tokens = H * W
```

Weight transform: Linear weights are PyTorch `[out, in]`; depthwise weights are `[hidden_features, 1, 3, 3]`.

Layout constraints: an NHWC plan can be attractive for depthwise conv, but LayerNorm/Linear token layout and residual add must agree.

Failure cases: non-GELU `hidden_act` must use the configured activation, not hard-coded GELU.

Parity test sketch: one block with random tensors and fixed weights for B0 and B5 channel sizes.

### Rewrite: decode-head per-stage projection

Source pattern: NCHW feature -> flatten/transpose -> Linear -> transpose/reshape -> bilinear upsample.

Replacement: 1x1 Conv2d equivalent for each stage projection, followed by bilinear upsample.

Preconditions:

- Projection is applied independently to every spatial position.
- Feature map is NCHW.
- Linear bias is preserved.

Shape equations:

```text
Linear(C_i -> D) over H_i*W_i positions == Conv2d(C_i -> D, kernel=1)
```

Weight transform:

```python
conv_w = linear_w.reshape(D, C_i, 1, 1)
conv_b = linear_b
```

Layout constraints: preserves NCHW decode-head flow and removes flatten/transpose pairs.

Failure cases: if `reshape_last_stage=False` and last hidden state is tokens, first recover a valid spatial map; current source assumes square.

Parity test sketch: compare decode-head logits before and after rewriting projections to 1x1 convolutions.

### Rewrite: NHWC/channel-last local conv islands

Source pattern: NCHW Conv2d/BatchNorm2d/interpolate-heavy regions with token conversions at boundaries.

Replacement: local NHWC/channel-last lowering for Conv2d, depthwise Conv2d, BatchNorm, and 1x1 Conv2d.

Preconditions:

- Entire local island is controlled by layout pass.
- Axis rewrites are explicit:
  - `dim=1` channel concat/argmax/BatchNorm channel handling must move to NHWC channel axis or be kept NCHW.
  - `flatten(2)` and `transpose(1,2)` must be replaced with equivalent NHWC tokenization.
  - Conv2d and interpolate consumers must agree on spatial axes.

Weight transform: provider-specific NCHW OIHW to NHWC-compatible filter layout if needed.

Failure cases: postprocess `argmax(dim=1)` and decode `cat(dim=1)` are incorrect under silent NHWC translation. Use no-layout-translation guards around unsupported regions.

Parity test sketch: whole-stage and decode-head layout-pass parity with non-square image sizes to catch axis mistakes.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d provider coverage for overlapping patch embeddings, sequence reduction, depthwise Mix-FFN conv, and decode 1x1 convs. SegFormer is convolution-heavy despite being a Transformer family.
- Token LayerNorm over channel dimension. It occurs after every patch embedding, sequence reduction, attention pre-norm, MLP pre-norm, and stage output.
- Efficient self-attention with reduced K/V length. Fusing Q/K/V matmuls is less direct because K/V may consume reduced tokens, but the attention matmul/softmax/value path is central.
- Decode-head projection-to-1x1-conv rewrite. It removes layout churn and lets existing Conv2d providers handle a large part of segmentation head work.

Medium priority:

- Mix-FFN GEMM + depthwise Conv2d + GELU scheduling. The inserted depthwise conv is operator-significant and can dominate small channel stages.
- Conv/BN/ReLU fusion for decode `linear_fuse + batch_norm + relu` in inference.
- Bilinear interpolate kernels for upsampling logits and intermediate decode features.
- Layout elimination around `flatten(2).transpose(1,2)` and token-to-NCHW round trips.

Lower priority:

- Classification head pooling and Linear for `mit-*` checkpoints.
- Fused postprocess resize plus argmax for dense maps.
- DropPath and dropout training behavior; inference treats them as identity.

## 11. Runtime staging plan

Stage 1: parse SegformerConfig, load weights, and run patch embedding plus one encoder block with NCHW Conv2d, LayerNorm, Linear, attention, and depthwise Conv2d parity.

Stage 2: implement full encoder output parity for B0 at processor size 512. Return all hidden states because semantic segmentation requires multi-scale features.

Stage 3: implement decode head with original source layout operations, bilinear upsample, BatchNorm2d inference, and classifier 1x1 Conv2d.

Stage 4: add processor-compatible postprocess outside compiled graph: optional target-size bilinear upsample and `argmax` maps.

Stage 5: optimize lowering: Conv2d provider selection, depthwise Conv2d, decode projection to 1x1 Conv2d, and reduced-K/V attention kernels.

Stage 6: add guarded NHWC/channel-last layout islands only after NCHW parity is stable.

Stage 7: broaden checkpoint coverage to B1/B5 and classification heads.

Initial stubs allowed: training losses, labels path, dropout/DropPath randomness, output attentions, image processor CPU pipeline inside DinoML runtime, and postprocess inside compiled graph.

## 12. Parity and validation plan

- Random op tests for Conv2d output shapes and values: patch embedding, sequence reduction, depthwise conv, 1x1 decode conv.
- LayerNorm parity with eps `1e-6` on token tensors.
- Attention parity for each stage shape, including SR ratios 8, 4, 2, and 1. Check fp32 softmax behavior in eager reference.
- Single `SegformerLayer` parity with fixed random weights for B0 and B5 channel sizes.
- Full encoder parity on `nvidia/segformer-b0-finetuned-ade-512-512` at 512x512.
- Decode head parity from saved encoder hidden states, then full segmentation logits parity.
- Postprocess parity for target sizes equal to original image size and omitted `target_sizes`.
- Non-square image smoke tests if dynamic shape support is claimed, especially for `reshape_last_stage=False` rejection/guard behavior.

Recommended tolerances: start with fp32 absolute/relative tolerance around `1e-4` for block outputs and logits. For fp16/bf16 optimized paths, compare against PyTorch reduced precision with looser tolerances, for example `5e-3` to `1e-2` depending on attention and interpolation accumulation.

## 13. Performance probes

- CPU preprocessing throughput by image size: 512, 640, 1024.
- Encoder-only latency and throughput per checkpoint scale B0, B1, B5.
- Per-stage profiling: patch embedding Conv2d, attention with SR, Mix-FFN depthwise conv.
- Decode-head profiling split into per-stage projections, bilinear upsample, 1x1 fuse/classifier.
- Batch-size sweep for 1, 2, 4, 8 at fixed processor size.
- Resolution sweep for 512, 640, 1024 to expose quadratic query attention cost and decode upsample cost.
- Attention backend comparison: eager matmul/softmax versus SDPA/Flash-compatible reduced-K/V attention.
- Layout-pass benchmark: source-faithful NCHW versus guarded NHWC/channel-last conv islands.
- End-to-end segmentation maps/sec including postprocess.

No benchmark observations are included here; these are proposed probes derived from source structure.

## 14. Skip/defer list

- Training losses: CrossEntropy/BCE label path, ignore-index loss behavior, and stochastic DropPath.
- Gradient checkpointing.
- Output attentions unless needed for debugging.
- In-graph CPU image preprocessing; keep processor as data-pipeline work first.
- In-graph semantic postprocess; start with logits parity and external postprocess.
- `reshape_last_stage=False` dynamic/non-square support; reject or route through a guarded compatibility path first.
- Classification heads for `mit-*` checkpoints, unless segmentation parity is complete.
- Quantization and multi-GPU partitioning.

## 15. Final implementation checklist

- [ ] Parse SegformerConfig including list-valued per-stage dimensions.
- [ ] Load SegFormer weights with Conv2d, depthwise Conv2d, LayerNorm, Linear, BatchNorm2d, and classifier parameters.
- [ ] Implement NCHW Conv2d lowering for overlapping patch embeddings.
- [ ] Implement sequence-reduction Conv2d plus LayerNorm for K/V tokens.
- [ ] Implement non-causal MHA with reduced K/V sequence length.
- [ ] Implement Mix-FFN with depthwise 3x3 Conv2d.
- [ ] Preserve multi-scale encoder hidden states for semantic segmentation.
- [ ] Implement decode head projections, upsample, reverse-order concat, Conv-BN-ReLU, and classifier.
- [ ] Implement external semantic segmentation postprocess: target-size bilinear resize plus argmax.
- [ ] Add one-block parity tests for all stage shapes.
- [ ] Add full B0 encoder and segmentation-logits parity tests.
- [ ] Add B1/B5 checkpoint shape and logits smoke tests.
- [ ] Add rewrite for decode Linear projections to equivalent 1x1 Conv2d.
- [ ] Add guarded NHWC/channel-last layout optimization tests with axis rewrite checks.
- [ ] Benchmark encoder, decode head, postprocess, batch-size, and resolution sweeps.
