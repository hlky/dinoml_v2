# OneFormer DinoML operator audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: representative official checkpoints:

- `shi-labs/oneformer_ade20k_swin_tiny`
- `shi-labs/oneformer_ade20k_swin_large`
- `shi-labs/oneformer_cityscapes_swin_large`
- `shi-labs/oneformer_coco_swin_large`
- `shi-labs/oneformer_ade20k_dinat_large`
- `shi-labs/oneformer_coco_dinat_large`

Config source: local snapshots under `agents/plans/transformers/oneformer/_sources/hf_configs/*/{config.json,preprocessor_config.json,tokenizer_config.json,vocab.json,merges.txt}`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/oneformer/configuration_oneformer.py`
- `X:/H/transformers/src/transformers/models/oneformer/modeling_oneformer.py`
- `X:/H/transformers/src/transformers/models/oneformer/image_processing_oneformer.py`
- `X:/H/transformers/src/transformers/models/oneformer/image_processing_pil_oneformer.py`
- `X:/H/transformers/src/transformers/models/oneformer/processing_oneformer.py`

Pinned source URLs:

- <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/oneformer/configuration_oneformer.py>
- <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/oneformer/modeling_oneformer.py>
- <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/oneformer/image_processing_oneformer.py>
- <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/oneformer/processing_oneformer.py>

Any missing files or assumptions: no gated/401/403 gaps found for the official SHI Labs configs sampled. `processor_config.json` is absent/404 for sampled repos; processor sequence lengths fall back to source defaults `max_seq_length=77`, `task_seq_length=77`. Backbones are composed through `load_backbone(config)` and are not fully re-audited here; this report owns the OneFormer head, pixel decoder, transformer decoder, task-token path, and postprocessing.

## 2. High-level architecture

OneFormer is a universal image segmentation model:

```text
RGB image + task string -> image resize/normalize/pad + task tokenization
-> nested vision backbone -> pixel decoder with multi-scale deformable attention + FPN
-> task-token-conditioned query transformer + masked transformer decoder
-> class logits + mask logits -> semantic/instance/panoptic postprocess
```

Runtime target: inference for `OneFormerForUniversalSegmentation` with semantic, instance, and panoptic segmentation. Training-only Hungarian matching, point sampling losses, and contrastive text-query losses can be deferred.

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize by shortest/longest edge, rescale/normalize, padding, pixel mask creation, BPE tokenization of `"the task is {semantic|instance|panoptic}"`.
- GPU/runtime: NCHW `pixel_values`, nested backbone feature maps, pixel decoder, query transformer/decoder, heads.
- Postprocess: semantic logit blend and resize; instance/panoptic thresholding, top-k, mask scoring, segment-id map construction, optional RLE conversion.

## 3. Important config dimensions

| Checkpoint | Dataset/classes | Backbone | Backbone shape | Queries | Decoder | Pixel encoder | Head dim | Resize |
| --- | ---: | --- | --- | ---: | --- | --- | ---: | --- |
| `oneformer_ade20k_swin_tiny` | 150 | Swin | embed 96, depths 2/2/6/2, heads 3/6/12/24, window 7 | 150 | 10 layers, 8 heads, hidden 256 | 6 deformable layers, 8 heads | 32 | 512-2048 |
| `oneformer_ade20k_swin_large` | 150 | Swin | embed 192, depths 2/2/18/2, heads 6/12/24/48, window 12 | 250 | same | same | 32 | 640-2560 |
| `oneformer_cityscapes_swin_large` | 19 | Swin large | same as above | 250 | same | same | 32 | 1024-2048 |
| `oneformer_coco_swin_large` | 133 | Swin large | same as above | 150 | same | same | 32 | 800-1333 |
| `oneformer_ade20k_dinat_large` | 150 | DiNAT | embed 192, depths 3/4/18/5, heads 6/12/24/48 | 250 | same | same | 32 | 640-2560 |
| `oneformer_coco_dinat_large` | 133 | DiNAT large | same as above | 150 | same | same | 32 | 800-1333 |

Other effective defaults/fields:

| Field | Value |
| --- | --- |
| `conv_dim`, `mask_dim`, `hidden_dim` | 256 |
| `encoder_feedforward_dim` | 1024 |
| `dim_feedforward` | 2048 |
| `num_attention_heads` | 8 |
| `query_dec_layers` | 2 |
| `strides` | `[4, 8, 16, 32]` |
| `common_stride` | 4 |
| `task_seq_len` | 77 |
| `text_encoder_width/layers/vocab/n_ctx` | 256 / 6 / 49408 / 16, training-only in current inference configs |
| `torch_dtype` | float32 in sampled configs |
| Cache support | none; encoder/decoder are non-autoregressive |

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors, padding, resize, channel-wise normalize in preprocessing.
- Backbone feature list contract: 4 NCHW maps at strides 4/8/16/32, consumed as `feature_maps`.
- `flatten(2)`, `transpose`, `permute`, `view`, `reshape`, `split`, `cat`, `stack`, `repeat`, `unsqueeze`, `argmax`, `topk`, boolean indexing.
- NCHW `Conv2d`, `GroupNorm(32, 256)`, bilinear and nearest-exact resize, mask flattening.

Neural network primitives:

- Nested backbone ops from Swin/DiNAT audits.
- Pixel decoder projections: 1x1 conv from backbone channels to 256 plus GroupNorm; FPN lateral 1x1 conv and 3x3 conv to 256.
- LayerNorm, GroupNorm, ReLU, GELU/quick GELU in training-only text path, Linear, Embedding.
- Task MLP: `Linear(77 -> 256)`, ReLU, `Linear(256 -> 256)`.
- Class head: `Linear(256 -> num_labels + 1)`.
- Mask head: MLP `Linear(256 -> 256)`, ReLU, `Linear(256 -> 256)`, ReLU, `Linear(256 -> 256)`.

Attention primitives:

- Pixel encoder multi-scale deformable attention: 6 layers, `d_model=256`, 8 heads, 3 levels, 4 points/level.
- Query transformer: 2 layers of noncausal self-attention over object queries and cross-attention to mask features, `d_model=256`, 8 heads.
- Main decoder: 9 layers (`decoder_layers - 1`) alternating masked cross-attention to 3 feature levels, self-attention over queries, FFN.
- Training-only text encoder/mapper: CLIP-like 6-layer text Transformer and text-query projector when `config.is_training=True`.

Position/custom ops:

- 2D sine/cosine NCHW position embeddings with cumulative sums over H/W.
- Learned per-level embeddings for pixel decoder and transformer module.
- Deformable attention reference points and `grid_sample` bilinear sampling.

Generation/cache ops: none.

Preprocessing/postprocessing-coupled ops:

- BPE tokenization with `attention_mask * input_ids`, fixed length 77.
- Semantic `einsum("bqc,bqhw->bchw")`, bilinear resize, argmax.
- Instance/panoptic softmax, sigmoid, top-k, thresholding, mask weighted argmax, segment map fill, optional COCO RLE.

## 5. Layer/block breakdown

Pixel level module:

```text
pixel_values: [B,3,H,W]
features = backbone(pixel_values).feature_maps  # 4 NCHW maps, strides 4/8/16/32
for last 3 feature levels in reverse:
  x_l = Conv2d(C_l -> 256, 1x1) -> GroupNorm(32)
  pos_l = 2D sine position embedding [B,256,H_l,W_l]
  flatten x_l/pos_l to [B,H_l*W_l,256]
source = concat(levels, dim=sequence)
source = 6 * (MSDeformAttn(source) + residual -> LayerNorm -> Linear(256->1024)->ReLU->Linear(1024->256) + residual -> LayerNorm)
split source back to 3 NCHW maps
top-down FPN over extra stride-4 level:
  lateral Conv2d(C->256,1x1,bias=False)->GroupNorm
  y = lateral + bilinear_resize(previous)
  y = Conv2d(256->256,3x3,pad=1,bias=False)->GroupNorm->ReLU
mask_features = Conv2d(256->256,1x1)(highest resolution)
```

Task/query transformer:

```text
task_inputs: [B,77] token-id values
task_token = MLP(77->256->256)              # [B,256]
query_embeddings = Embedding(num_queries,256)
object_query_pos = query_embeddings[:-1]    # Q-1 learned queries
query_features = sine_pos(mask_features)
memory = Conv2d(256->256,1x1)(mask_features).flatten -> [HW,B,256]
queries = repeat(task_token, Q-1) or zeros  # [Q-1,B,256]
queries = 2 * query-transformer layers over object queries and memory
queries = cat(object_queries, task_token[None])  # total Q == config.num_queries
```

Main transformer decoder, repeated `decoder_layers - 1` times after an initial prediction head:

```text
output: [Q,B,256]
class_logits = Linear(256 -> num_labels+1)(LayerNorm(output).T)
mask_embed = MLP(256 -> 256 -> 256 -> 256)
mask_logits = einsum("bqc,bchw->bqhw", mask_embed, mask_features)
attention_mask = bilinear_resize(mask_logits to selected level).sigmoid().flatten < 0.5
for each layer i:
  level = i % 3
  output = masked_cross_attention(output, feature_level[level], attn_mask)
  output = self_attention(output)
  output = FFN(256 -> 2048 -> 256)
  recompute class_logits, mask_logits, attention_mask for next level
```

Projection bias: Linear projections use bias by default. Pixel decoder lateral/output convs often set `bias=False` when followed by GroupNorm; mask/class/task MLP linears use bias.

## 6. Attention requirements

Pixel deformable attention:

- Noncausal self-attention over concatenated multi-scale feature tokens.
- MHA, 8 heads, `head_dim=32`.
- Learned offsets: `Linear(256 -> 8*3*4*2)`.
- Attention weights: `Linear(256 -> 8*3*4)`, softmax over 12 samples per query/head.
- Values: `Linear(256 -> 256)`, sampled through bilinear `grid_sample` with `align_corners=False`, zero padding.
- No cache, no FlashAttention/SDPA replacement unless lowered to a custom deformable-attention kernel.

Query transformer attention:

- Noncausal self-attention over `Q-1` object queries and cross-attention to flattened mask-feature grid.
- Uses PyTorch `nn.MultiheadAttention` sequence-major ABI `[S,B,C]`.
- No attention mask in normal inference for query transformer memory.

Main decoder attention:

- Noncausal masked cross-attention from Q queries to one feature level at a time.
- Self-attention over all Q queries, including appended task token.
- Masking style: boolean `attn_mask` of shape `[B*num_heads, Q, H_l*W_l]`, where `True` blocks attention. If an entire row is blocked, it is reset to all false before attention.
- Attention cycles through 3 feature levels with `index % 3`.
- SDPA can cover dense self/cross attention after layout/shape normalization, but the dynamic boolean mask and sequence-major PyTorch ABI need careful parity tests.

Likely slow eager fallbacks: Python-level deformable attention loop over levels with `grid_sample`; segment postprocessing loops over batch and masks.

## 7. Position encoding and custom math

2D sine position embedding:

```python
def oneformer_sine_pos(shape, mask=None, num_pos_feats=128, scale=2 * math.pi):
    if mask is None:
        mask = zeros([shape[0], shape[2], shape[3]], bool)
    y = (~mask).cumsum(1)
    x = (~mask).cumsum(2)
    y = y / (y[:, -1:, :] + 1e-6) * scale
    x = x / (x[:, :, -1:] + 1e-6) * scale
    dim = 10000 ** (2 * floor(arange(num_pos_feats) / 2) / num_pos_feats)
    return concat([sin_cos(y / dim), sin_cos(x / dim)]).permute(0, 3, 1, 2)
```

Multi-scale deformable attention:

```python
def ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
    grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(spatial_shapes):
        v = split_level(value, h*w).reshape(B*heads, head_dim, h, w)
        g = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    w = attention_weights.transpose(1, 2).reshape(B*heads, 1, Q, levels*points)
    return (stack(sampled, -2).flatten(-2) * w).sum(-1).view(B, heads*head_dim, Q).transpose(1, 2)
```

Precomputable: level embeddings, learned query embeddings, tokenizer vocab/merges. Dynamic: positional cumsums depend on feature map shape, reference points depend on spatial shapes/valid ratios, attention masks depend on predicted masks and selected feature level.

## 8. Preprocessing and input packing

Image preprocessing:

- Converts to RGB, resizes by `shortest_edge`/`longest_edge`, rescales by `1/255`, normalizes with ImageNet mean/std, pads batch to max H/W, returns NCHW `pixel_values` and `pixel_mask`.
- Current `OneFormerModel.forward` creates or accepts `pixel_mask`, but does not pass it into the pixel level module; pixel decoder internally creates all-false masks for feature levels. Treat `pixel_mask` as processor output but not model-graph-consumed in this source.

Task/text preprocessing:

- `task_inputs` must be `semantic`, `instance`, or `panoptic`.
- Processor builds strings `the task is semantic`, `the task is instance`, or `the task is panoptic`.
- Tokenizer pads/truncates to length 77 and returns `attention_mask * input_ids`; model casts this numeric vector to model dtype and applies the task MLP.
- Training with segmentation maps can emit `text_inputs` prompts like `a photo with a {class}` plus mask/class labels, but inference configs have `is_training=False`, so `text_mapper` is absent.

## 9. Graph rewrite / lowering opportunities

### Rewrite: local 1x1 Conv2d -> per-pixel Linear

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Consumer is immediately NCHW flatten/transpose or local NCHW conv/norm region whose layout is controlled.

Replacement:

```text
NCHW/NHWC local layout -> MatMul(Cin,Cout) -> BiasAdd -> reshape
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels)
```

Failure cases: backbone feature consumers shared outside OneFormer, GroupNorm axis not rewritten, or NHWC region escapes into PyTorch-compatible output contracts.

Parity sketch: compare pixel decoder projected features before flatten for several dynamic H/W.

### Rewrite: mask head einsum -> batched GEMM

Source pattern:

```text
mask_embed [B,Q,256], mask_features [B,256,H,W]
einsum("bqc,bchw->bqhw")
```

Replacement:

```text
flatten mask_features -> [B,256,H*W]
BatchedMatMul([B,Q,256], [B,256,H*W]) -> reshape [B,Q,H,W]
```

Preconditions: channel dimension equals `mask_dim`, no dtype-changing side effects, output layout is Q-first.

### Rewrite: semantic blend einsum -> batched GEMM

Source pattern:

```text
softmax(class_logits)[..., :-1] [B,Q,C] x sigmoid(mask_logits) [B,Q,H,W]
```

Replacement: `BatchedMatMul(transpose classes [B,C,Q], masks_flat [B,Q,H*W]) -> [B,C,H,W]`.

Failure cases: null class not removed, resize-before-blend variant, or class-label remapping outside metadata.

### Rewrite: guarded NHWC pixel-decoder island

Preconditions:

- Region starts after backbone feature materialization and ends before public NCHW mask logits/postprocess.
- All axis-sensitive ops are rewritten: Conv2d, GroupNorm channel axis, flatten spatial axes, concat over sequence, bilinear resize, `einsum`/matmul layout.
- No uncontrolled consumer observes intermediate NCHW tensors.

Replacement: keep backbone and outputs faithful, but lower local conv/norm/resize/GEMM islands in NHWC. Use `no_translation()` around deformable attention grid-sampling and postprocess unless explicitly lowered.

Failure cases: DiNAT/Swin backbone output layout ambiguity, feature-map metadata lost, or dynamic mask attention expects `[B*num_heads,Q,HW]` from NCHW flatten order.

### Rewrite: deformable attention custom op

Preconditions:

- `n_levels=3`, `n_points=4`, `num_heads=8`, `d_model=256`.
- `sampling_locations` normalized to `[0,1]` and converted to grid-sample coordinates by `2*x-1`.
- Bilinear interpolation with `align_corners=False`, zero padding.

Replacement: fused CUDA deformable attention kernel returning `[B,sequence,256]`.

Failure cases: reference point last dim 4 path, changed level count/points, output attentions required.

## 10. Kernel fusion candidates

Highest priority:

- Multi-scale deformable attention fused kernel: current source loops over feature levels and uses `grid_sample`; this is the distinctive cost center.
- Mask head batched GEMM: `Q x H*W` dominates high-resolution mask prediction and postprocess.
- Dense MHA/FFN decoder blocks: Q is small but repeated, and dynamic masks make a stable SDPA lowering useful.

Medium priority:

- Conv1x1 + GroupNorm projection islands in pixel decoder.
- Bilinear resize + threshold/sigmoid mask-attention preparation.
- Semantic postprocess softmax/sigmoid/GEMM/argmax path.

Lower priority:

- Task-token MLP and tokenizer output handling.
- Training-only text mapper and contrastive loss path.
- COCO RLE conversion and CPU segment loops.

## 11. Runtime staging plan

Stage 1: parse config and load weights for OneFormer head/pixel decoder; compose Swin first through an already-audited backbone path.

Stage 2: validate backbone feature contract: 4 NCHW maps, channels, strides, `out_features`.

Stage 3: implement pixel decoder in faithful NCHW, including deformable attention fallback.

Stage 4: implement task-token MLP and query/main transformer decoder; produce class/mask logits.

Stage 5: add semantic postprocess parity.

Stage 6: add instance and panoptic postprocess, including metadata-driven thing/stuff behavior.

Stage 7: optimize deformable attention, mask GEMMs, dense attention, and local NHWC islands.

Stub initially: training losses, `text_mapper`, auxiliary logits consumers, output attentions, DiNAT backbone if not yet admitted.

## 12. Parity and validation plan

- Custom op parity: sine position embedding for odd/even H/W; deformable attention against source math for random tensors with 3 levels and 4 points.
- Pixel decoder parity: compare `mask_features` and `multi_scale_features` given frozen backbone feature tensors.
- Query decoder parity: compare class logits and mask logits for random pixel decoder outputs and task inputs.
- Processor parity: verify task token IDs for all three tasks and image resize/pad shapes for ADE20K/COCO/Cityscapes configs.
- End-to-end logits parity: `oneformer_ade20k_swin_tiny` semantic input, fp32 tolerance about `1e-4` absolute/relative for logits.
- Postprocess parity: semantic map equality; instance/panoptic `segments_info` and segment-id maps under fixed thresholds.
- fp16/bf16 tolerances: use fp32 postprocess for thresholds/argmax-sensitive checks; logits can use about `5e-3` to `1e-2` depending on backbone attention backend.

## 13. Performance probes

- Processor throughput for resize/pad/tokenization at ADE20K, COCO, and Cityscapes sizes.
- Backbone-only throughput by checkpoint family: Swin tiny, Swin large, DiNAT large.
- Pixel decoder-only throughput and memory by input resolution.
- Deformable attention backend comparison: eager grid-sample vs fused kernel.
- Decoder-only throughput as `num_queries` changes 150 vs 250.
- Mask logits GEMM cost versus H/4 x W/4 mask grid.
- Semantic/instance/panoptic postprocess latency, especially top-k and segment loops.
- NHWC island probe for pixel decoder conv/norm/resizes with exact NCHW boundary materialization.

## 14. Skip/defer list

- Training losses, Hungarian matcher, random point sampling, contrastive loss.
- `config.is_training=True` text mapper path for first inference integration.
- Output attentions and hidden-state dumps.
- DiNAT backbone until nested backbone operator coverage is available.
- COCO RLE output unless an application requires it.
- Multi-GPU/tensor parallelism and quantization.

## 15. Final implementation checklist

- [ ] Parse OneFormer config, including nested `backbone_config`.
- [ ] Load OneFormer head/pixel-decoder weights and map `num_labels` from id2label/num_classes.
- [ ] Route Swin/DiNAT backbone through admitted backbone implementations.
- [ ] Implement task-token tokenizer ABI and `Linear(77->256->256)` task MLP.
- [ ] Implement 2D sine position embedding.
- [ ] Implement pixel decoder projections, FPN, and mask projection.
- [ ] Implement multi-scale deformable attention fallback.
- [ ] Add fused deformable attention kernel candidate.
- [ ] Implement query transformer and masked decoder attention.
- [ ] Implement mask/class heads and mask-attention generation.
- [ ] Implement semantic postprocess.
- [ ] Implement instance/panoptic postprocess and metadata handling.
- [ ] Add parity tests for pixel decoder, decoder logits, and postprocess.
- [ ] Benchmark backbone, pixel decoder, deformable attention, decoder, and postprocess separately.
