# Transformers VitDet Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Equivalent source URL:
  https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitdet

Model id:
  Source default/autodoc checkpoint: google/vitdet-base-patch16-224.
  Accessible representative nested-backbone configs:
  hustvl/vitmatte-base-composition-1k and hustvl/vitmatte-small-composition-1k.

Config source:
  transformers/src/transformers/models/vitdet/configuration_vitdet.py
  https://huggingface.co/hustvl/vitmatte-base-composition-1k/raw/main/config.json
  https://huggingface.co/hustvl/vitmatte-small-composition-1k/raw/main/config.json

Source files inspected:
  transformers/src/transformers/models/vitdet/modeling_vitdet.py
  transformers/src/transformers/models/vitdet/configuration_vitdet.py
  transformers/src/transformers/models/vitdet/__init__.py
  transformers/src/transformers/models/auto/auto_mappings.py
  transformers/src/transformers/models/auto/modeling_auto.py
  transformers/src/transformers/models/auto/image_processing_auto.py
  transformers/docs/source/en/model_doc/vitdet.md
  transformers/tests/models/vitdet/test_modeling_vitdet.py

Any missing files or assumptions:
  No VitDet image processor, detector head, segmentation head, NMS, mask
  postprocess, or class-score postprocess is present in this Transformers
  family. The docs explicitly say only the backbone is available. Google
  standalone VitDet config URLs returned HTTP 401 during this audit, so
  base/large sizing beyond source defaults is labeled as source-default or
  gated rather than checkpoint-confirmed.
```

## 2. High-level architecture

VitDet is a vision-only transformer backbone intended to feed detector or segmentation frameworks such as Mask R-CNN. The Transformers implementation provides `VitDetModel` and `VitDetBackbone`, both ending at feature maps, not object logits or masks.

```text
image preprocessing outside VitDet -> NCHW pixel_values -> patch Conv2d ->
NCHW patch feature map -> per-block NHWC transformer body with optional window
attention and relative position bias -> NCHW final/backbone feature maps
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize/padding is not owned by VitDet source in this family.
- Patch stage: source input is NCHW `[B,C,H,W]`; patch projection is non-overlapping `Conv2d(C -> hidden_size, kernel=patch, stride=patch)`.
- Encoder stage: blocks repeatedly convert NCHW to NHWC for LayerNorm, attention, MLP, and window partitioning, then return NCHW.
- Backbone extraction: `VitDetBackbone` collects selected hidden states by stage name; all feature maps stay `[B, hidden_size, Hp, Wp]`.
- Downstream heads: detection/segmentation heads, feature pyramid construction, NMS, box scaling, mask resize/crop/threshold are not in this family.

## 3. Important config dimensions

Source-default `VitDetConfig`:

| Field | Value | Source |
|---|---:|---|
| model_type | `vitdet` | config class |
| hidden_size | 768 | source default |
| num_hidden_layers | 12 | source default |
| num_attention_heads | 12 | source default |
| head_dim | 64 | inferred `hidden_size / heads` |
| mlp_ratio | 4 | source default |
| MLP hidden width | 3072 | inferred `hidden_size * mlp_ratio` |
| hidden_act | `gelu` | source default |
| dropout_prob | 0.0 | source default |
| layer_norm_eps | 1e-6 | source default |
| image_size | 224 | source default |
| pretrain_image_size | 224 | source default |
| patch_size | 16 | source default |
| num_channels | 3 | source default |
| qkv_bias | true | source default |
| use_absolute_position_embeddings | true | source default |
| use_relative_position_embeddings | false | source default |
| window_block_indices | `[]` | source default |
| residual_block_indices | `[]` | source default |
| window_size | 0 | source default |
| cache/generation | not applicable | source inspection |

Representative checkpoint/config sweep:

| Config | H | layers | heads | head_dim | image | channels | patch | rel pos | window blocks | residual blocks | dtype/source |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `VitDetConfig()` source default | 768 | 12 | 12 | 64 | 224 | 3 | 16 | false | none | none | source default |
| `hustvl/vitmatte-base-composition-1k` nested backbone | 768 | 12 default | 12 default | 64 | 512 | 4 | 16 default | true | 0,1,3,4,6,7,9,10 | 2,5,8,11 | HF config `float32` |
| `hustvl/vitmatte-small-composition-1k` nested backbone | 384 | 12 default | 6 | 64 | 512 | 4 | 16 default | true | 0,1,3,4,6,7,9,10 | 2,5,8,11 | HF config `float32` |
| `google/vitdet-base-patch16-224` | gated | gated | gated | gated | gated | gated | gated | gated | gated | gated | HTTP 401 |
| `google/vitdet-large-patch16-224` | gated | gated | gated | gated | gated | gated | gated | gated | gated | gated | HTTP 401 |

The accessible VitMatte configs omit many nested backbone fields; effective values above come from `VitDetConfig` defaults unless explicitly set.

## 3a. Family variation traps

- `num_channels` is not always 3. VitMatte nested configs use 4-channel inputs, so patch projection K is `4 * patch_h * patch_w`.
- `image_size` and `pretrain_image_size` can differ. Absolute position embeddings are initialized at `pretrain_image_size / patch_size` and bicubic-resized to runtime patch grid when needed.
- `hidden_size` must be divisible by `num_attention_heads`. Source uses integer `head_dim = hidden_size // num_attention_heads` and does not separately expose `head_dim`.
- Attention alternates by layer: indices in `window_block_indices` use local non-shifted window attention; other layers use global attention.
- Relative position embedding shapes depend on the attention region. Global blocks allocate `2 * (image_size / patch_size) - 1`; window blocks allocate `2 * window_size - 1`.
- Window partition pads the NHWC hidden map to a multiple of `window_size`, then unpartitions and slices back. Layout rewrites must preserve the pad/slice boundaries.
- Residual bottleneck blocks are optional per layer and operate in NCHW after the transformer block, with channel-wise custom `VitDetLayerNorm`.
- `VitDetBackbone` physical modules are not a feature pyramid. All stage outputs have the same channel count and patch-grid resolution unless downstream code changes them.
- No source attention mask, no causal mode, no KV cache, no generation controller.
- No source detector/segmentation postprocess exists despite the model name. Any boxes, masks, NMS, target sizes, or variable-length per-image records belong to downstream heads.

## 4. Operator coverage checklist

### Tensor/layout ops

- NCHW image input `[B,C,H,W]`.
- Non-overlapping patch Conv2d output `[B,Hid,Hp,Wp]`.
- NCHW to NHWC permute before each transformer block: `[B,Hid,Hp,Wp] -> [B,Hp,Wp,Hid]`.
- NHWC to NCHW permute after each block.
- Absolute position path: remove CLS position row, square reshape, NCHW bicubic interpolate, return NHWC position table, add to patch map.
- Window partition: NHWC pad on H/W, reshape `[B,Hp/ws,ws,Wp/ws,ws,C]`, transpose window axes, contiguous flatten to `[B*num_windows,ws,ws,C]`.
- Window unpartition: inverse reshape/transpose, crop padded H/W, contiguous.
- Optional hidden-state tuple accumulation for backbone output selection.

### Neural network primitives

- Patch `Conv2d(num_channels -> H, kernel=patch_size, stride=patch_size, bias=True)`.
- Fused QKV linear `Linear(H -> 3H, bias=qkv_bias)`, split order `[q,k,v]` through reshape to `[3,B,heads,N,head_dim]`.
- Output projection `Linear(H -> H, bias=True)`.
- MLP `Linear(H -> H*mlp_ratio) -> GELU -> Dropout -> Linear(H*mlp_ratio -> H) -> Dropout`.
- Standard `LayerNorm` over NHWC last axis with epsilon `layer_norm_eps`.
- Custom NCHW `VitDetLayerNorm` over channel axis for residual bottleneck blocks.
- Optional residual bottleneck: `1x1 Conv(H -> H/2, no bias) -> channel LN -> GELU -> 3x3 Conv(H/2 -> H/2, padding=1, no bias) -> channel LN -> GELU -> 1x1 Conv(H/2 -> H, no bias) -> channel LN -> residual add`.
- DropPath is inference identity unless training with nonzero `drop_path_rate`.

### Attention primitives

- Noncausal self-attention only.
- MHA only; no GQA/MQA.
- Global attention has sequence length `N = Hp * Wp`.
- Window attention has per-window sequence length `Nw = window_size * window_size`; batch is folded as `B * ceil(Hp/ws) * ceil(Wp/ws)`.
- Source eager math: scale queries before `Q @ K^T`, add optional decomposed relative positions, softmax over keys, then `P @ V`.
- `output_attentions=True` reshapes attention probabilities to `[B, heads, N, N]` for global or `[B*num_windows, heads, Nw, Nw]` for window blocks.

### Position/relative-bias ops

- Learned absolute position table with a stored CLS row even though no CLS token is emitted by VitDet embeddings.
- Bicubic 2D interpolation for absolute positions with `align_corners=False`.
- Learned decomposed relative position tables `rel_pos_h` and `rel_pos_w`.
- 1D linear interpolation of relative tables when query/key size differs from table length.
- Dynamic integer coordinate gather from relative table.
- Two `einsum` contractions from query tensor to height/width relative logits.

### Preprocessing-coupled ops

- VitDet source does not own image preprocessing or processors.
- GPU graph starts from already-normalized `pixel_values`.
- Shape guards needed: `pixel_values.shape[1] == config.num_channels`; ideally image H/W divisible by patch size for clean Conv2d patch grid. PyTorch Conv2d floors if not divisible, so exact parity either permits floor semantics or rejects with a bounded guard.

### Detection/segmentation postprocess

- Not present in `vitdet`.
- No class logits, boxes, masks, NMS, target/original size scaling, mask thresholding, or variable-length detection records.
- Downstream VitMatte uses a VitDet backbone but its trimap/image matting path is outside this family audit.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B,C,H,W]
x = Conv2d(C -> Hdim, kernel=P, stride=P)(pixel_values)
Hp = floor((H - P)/P) + 1
Wp = floor((W - P)/P) + 1
if absolute_pos:
  pos = position_embeddings[:, 1:, :] -> reshape/interpolate to [1,Hp,Wp,Hdim]
  x = permute_nchw_to_nhwc(x) + pos
  x = permute_nhwc_to_nchw(x)
```

Encoder block `i`, repeated `num_hidden_layers`:

```text
x_nchw: [B,Hdim,Hp,Wp]
x = permute_nchw_to_nhwc(x_nchw)
shortcut = x
x = LayerNorm_last_axis(x)
if i in window_block_indices:
  x, padded_hw = window_partition(x, window_size)
attn = MHA(x) with optional decomposed relative position
if i in window_block_indices:
  attn = window_unpartition(attn, window_size, padded_hw, original_hw)
x = shortcut + DropPath(attn)
x = x + DropPath(MLP(LayerNorm_last_axis(x)))
x_nchw = permute_nhwc_to_nchw(x)
if i in residual_block_indices:
  x_nchw = x_nchw + BottleneckConvBlock(x_nchw)
```

Backbone output:

```text
hidden_states = (stem_output, stage1_output, ..., stageN_output)
feature_maps = tuple(hidden_state for stage in out_features)
```

## 6. Attention requirements

VitDet attention is encoder-style, bidirectional, and noncausal. There is no mask, cross-attention, packed varlen ABI, sliding window with shifts, RoPE, ALiBi, or KV cache.

| Variant | Layers | Query/key/value shape | Relative positions | Notes |
|---|---|---|---|---|
| Global self-attention | layers not in `window_block_indices` | `Q,K,V: [B*heads, Hp*Wp, head_dim]` | optional, table length based on full patch grid | Quadratic in full image patch count. |
| Window self-attention | layers in `window_block_indices` | `Q,K,V: [B*num_windows*heads, ws*ws, head_dim]` | optional, table length based on `window_size` | Non-overlapping, no shift, pad then crop. |

Fused attention parity constraints:

- Preserve query scaling before matmul: `(queries * head_dim**-0.5) @ keys.T`.
- Add decomposed relative logits before softmax.
- Softmax axis is key/token dimension.
- For window attention, relative coordinates use window H/W, not original image H/W.
- Attention return tensors, if requested, are observability outputs and may block aggressive fused kernels unless separately materialized.

## 7. Position encoding and custom math

Absolute position interpolation:

```python
def vitdet_absolute_pos(pos, hp, wp):
    # pos: [1, num_patches + 1, hidden], first row is a stored CLS position.
    pos = pos[:, 1:, :]
    size = int(sqrt(pos.shape[1]))
    if size != hp or size != wp:
        pos = bicubic_interpolate(
            pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(hp, wp),
            align_corners=False,
        ).permute(0, 2, 3, 1)
    else:
        pos = pos.reshape(1, hp, wp, -1)
    return pos
```

Decomposed relative position math:

```python
def vitdet_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = 2 * max(q_size, k_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos = linear_interpolate_1d(rel_pos, size=max_rel_dist)
    q_coords = arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    idx = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[idx.long()]
```

```python
def add_vitdet_relative_logits(attn, q, rel_h, rel_w, q_hw, k_hw):
    qh, qw = q_hw
    kh, kw = k_hw
    q_grid = q.reshape(batch_heads, qh, qw, head_dim)
    rh = einsum("bhwc,hkc->bhwk", q_grid, vitdet_rel_pos(qh, kh, rel_h))
    rw = einsum("bhwc,wkc->bhwk", q_grid, vitdet_rel_pos(qw, kw, rel_w))
    return (attn.reshape(batch_heads, qh, qw, kh, kw)
            + rh[:, :, :, :, None]
            + rw[:, :, :, None, :]).reshape(batch_heads, qh * qw, kh * kw)
```

Precomputable pieces:

- Absolute position interpolation can be precomputed per admitted patch grid.
- Relative coordinate index tensors can be precomputed per `(q_size,k_size)` and per `window_size`.
- Relative logits still depend on runtime queries and need fused/gathered contraction support.

## 8. Preprocessing and input packing

VitDet does not define an image processor in this Transformers family. The runtime contract begins with:

```text
pixel_values: float tensor [B, num_channels, image_height, image_width]
```

Preprocessing outside this family may include image normalization, resize, padding, masks, trimaps, boxes, or segmentation prompts, but none of those enter `VitDetModel.forward` except as channels in `pixel_values`.

Detection/segmentation postprocess is absent. For downstream integrations that pair VitDet with detectors or matte/segmentation heads, document separately:

- original image sizes and padded sizes,
- feature pyramid construction from single-scale `[B,Hdim,Hp,Wp]`,
- box coordinate conventions and NMS behavior,
- mask upsample/crop/threshold rules,
- variable-length per-image output records,
- extra input channels such as VitMatte's 4-channel image/trimap tensor.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> patch GEMM

Source pattern:

```text
Conv2d(C -> Hdim, kernel=(Ph,Pw), stride=(Ph,Pw), padding=0, groups=1)
```

Replacement:

```text
NHWC window flatten [B,Hp,Wp,Ph*Pw*C] -> Linear(Ph*Pw*C -> Hdim) -> NHWC/NCHW as needed
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Source Conv2d floor semantics are either preserved or guarded with `H % Ph == 0` and `W % Pw == 0`.
- Weight transform: `w_flat = conv.weight.reshape(Hdim, C*Ph*Pw)` using PyTorch NCHW kernel flatten order.
- Bias is copied unchanged.

Failure cases:

- Non-divisible spatial dimensions if DinoML chooses strict patch divisibility.
- Channel-last rewrite must not silently change the PyTorch kernel flatten order.

Parity test sketch:

- Random NCHW inputs for `C=3` and `C=4`, patch 16, image 224/512.
- Compare Conv2d output to patch-flatten GEMM output before position add.

### Rewrite: NHWC transformer block keep-alive

Source pattern:

```text
NCHW -> NHWC -> LN/attention/MLP -> NCHW
```

Replacement:

```text
Keep encoder internal state NHWC across blocks; materialize NCHW only for residual conv blocks and requested backbone outputs.
```

Preconditions:

- No consumer between blocks requires NCHW.
- `output_hidden_states`/`VitDetBackbone` can either materialize NCHW snapshots at requested stages or expose layout metadata.
- Residual bottleneck layers still run with explicit NCHW/channel-axis normalization unless separately rewritten.

Failure cases:

- Stages selected by `out_features` expect source NCHW tensors.
- Residual block indices create NCHW islands after selected layers.

Parity test sketch:

- Compare block-by-block outputs for configs with no residual blocks and with residual blocks `[2,5,8,11]`.

### Rewrite: window partition/unpartition as layout metadata

Source pattern:

```text
pad NHWC -> reshape -> permute -> contiguous view -> attention -> inverse reshape/permute -> crop
```

Replacement:

```text
Windowed attention kernel reads NHWC tiles directly with pad guards and writes original H/W.
```

Preconditions:

- Window size is static and positive.
- Attention does not request materialized window tensors for external observation.
- Pad values are zero and only affect padded tokens inside attention; output is cropped.
- Relative position table uses `window_size`, not full image size.

Failure cases:

- Dynamic window size.
- `output_attentions=True` requires materialized per-window attention probability tensors.
- Nonzero or model-specific pad semantics.

### Rewrite: decomposed relative bias fusion

Source pattern:

```text
QK matmul -> reshape attention -> two einsums with Q and rel tables -> add -> softmax
```

Replacement:

```text
Fused attention pre-softmax bias callback or custom kernel that accumulates height/width relative logits.
```

Preconditions:

- Known `(qh,qw,kh,kw)` per attention call.
- Precomputed relative gather indices per axis.
- Same query scaling and dtype/upcast policy as source.

Failure cases:

- Fused backend only accepts static additive bias independent of Q; VitDet relative logits are query-dependent.
- Different q/k spatial sizes require interpolation and scaled coordinate indices.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d -> GEMM for `C*P*P -> Hdim`, especially 512x512 4-channel VitMatte backbones.
- NHWC LayerNorm + QKV projection, because source already enters attention in NHWC.
- Window attention with decomposed relative logits fused before softmax; avoids materializing padded windows and large bias tensors.
- Global attention with decomposed relative logits for the few cross-window/global propagation blocks.

Medium priority:

- MLP `Linear -> GELU -> Linear` fusion or at least activation/dropout elimination for inference.
- Absolute position interpolation precompute/cache per admitted image grid.
- Residual bottleneck NCHW conv block fusion for configs that use `[2,5,8,11]`.
- Channel-axis `VitDetLayerNorm` optimized for NCHW residual blocks.

Lower priority:

- DropPath training kernel; inference path is identity.
- `output_attentions` materialization; useful for debug, not first production path.
- Dynamic non-divisible image H/W floor-semantics support if first integration can guard to divisible patch sizes.

## 11. Runtime staging plan

Stage 1: parse `VitDetConfig`, load source-default and nested-backbone configs, validate patch/grid/head dimensions.

Stage 2: implement patch embedding and absolute position add parity for NCHW input, including 3-channel and 4-channel cases.

Stage 3: implement one encoder block with global attention and no relative positions; verify NCHW/NHWC boundary behavior.

Stage 4: add decomposed relative position math and parity tests for both full-grid and window-grid table shapes.

Stage 5: add window partition/unpartition with pad/crop and window attention.

Stage 6: add optional residual bottleneck blocks and custom channel-axis LayerNorm.

Stage 7: implement `VitDetBackbone` output selection with stage names and NCHW feature maps.

Stage 8: optimize layout keep-alive, patch GEMM, and fused attention kernels behind guards.

Stub initially:

- Training-only DropPath and dropout as identity for inference.
- `output_attentions` materialization unless debug parity requires it.
- Downstream detector/segmentation heads and postprocess, because they are not in this family.

## 12. Parity and validation plan

- Config parsing tests for source defaults and the two accessible VitMatte nested backbone configs.
- Patch Conv2d parity for `C=3,H=224,P=16,Hdim=768` and `C=4,H=512,P=16,Hdim=384/768`.
- Absolute position interpolation tests for same-size and resized grids; fp32 tolerance `atol=1e-5, rtol=1e-5`.
- `get_rel_pos` parity for equal and unequal q/k sizes, including table interpolation.
- Decomposed relative logits parity against source for small grids, e.g. `2x3`, `14x14`.
- Window partition/unpartition round trip for divisible and non-divisible H/W.
- Single global block parity with relative positions disabled and enabled.
- Single window block parity with `window_size=14`.
- Full 12-layer backbone parity for VitMatte-style config with residual blocks `[2,5,8,11]`.
- `VitDetBackbone` feature map selection parity for `out_features=["stage12"]`.
- Recommended tolerances: fp32 `1e-4` end-to-end, fp16/bf16 `1e-2` after attention/MLP; use tighter per-op tolerances where interpolation is not involved.

No model execution or imports were run for this audit.

## 13. Performance probes

- Processor/external preprocessing throughput, measured separately from VitDet.
- Patch embedding throughput: Conv2d baseline vs patch GEMM.
- Encoder global-block latency sweep by image grid: 14x14, 32x32, and any admitted 512/16 = 32 grid.
- Window-block latency sweep by `window_size` and padded/non-padded grids.
- Relative-position overhead: attention without rel pos vs with decomposed relative logits.
- Layout cost probe: per-block NCHW/NHWC permutes vs NHWC keep-alive with materialized NCHW outputs.
- Residual bottleneck overhead for configs with `[2,5,8,11]`.
- Memory probe for full-grid attention maps, especially `32x32` global blocks (`N=1024`) and `output_attentions=True`.
- Downstream integration probe: feature map production throughput before any detector/matting head.

## 14. Skip/defer list

- Detection heads, class logits, box heads, mask heads, NMS, and postprocess: not present in VitDet source.
- VitMatte convstream/fusion/matting head: downstream `vitmatte`, not this family.
- Training DropPath/dropout behavior and gradient checkpointing.
- `output_attentions=True` optimized materialization.
- Dynamic image sizes beyond a small admitted set of patch grids.
- Quantized/packed weights; no source-coupled quantization format was found.
- Multi-GPU/tensor parallel support.

## 15. Final implementation checklist

- [ ] Parse `VitDetConfig` including `window_block_indices`, `residual_block_indices`, `out_features`, and `out_indices`.
- [ ] Load patch, absolute position, QKV, projection, MLP, relative position, and residual bottleneck weights.
- [ ] Implement NCHW patch embedding with optional Conv2d -> GEMM rewrite.
- [ ] Implement absolute position resize/add with CLS-row removal.
- [ ] Implement NHWC LayerNorm, QKV split, noncausal MHA, and output projection.
- [ ] Implement `get_rel_pos` and decomposed relative position logits.
- [ ] Implement window partition/unpartition with pad/crop.
- [ ] Implement optional residual bottleneck block and channel-axis `VitDetLayerNorm`.
- [ ] Implement `VitDetBackbone` stage selection and NCHW feature map outputs.
- [ ] Add layout rewrite guards for NHWC keep-alive and no-layout-translation regions.
- [ ] Add parity tests for patch embedding, relative position math, window round trip, one block, and full backbone.
- [ ] Add performance probes for patch GEMM, window/global attention, relative-logit overhead, and layout permutes.

