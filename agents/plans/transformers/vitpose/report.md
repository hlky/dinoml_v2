# VitPose Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from transformers
Model id: usyd-community/vitpose-base-simple plus representative usyd-community/vitpose-base, vitpose-base-coco-aic-mpii, vitpose-plus-small/base/large/huge
Config source: HF config.json/preprocessor_config.json fetched 2026-05-13; source defaults from configuration_vitpose.py and configuration_vitpose_backbone.py
Source files inspected:
  transformers/src/transformers/models/vitpose/configuration_vitpose.py
  transformers/src/transformers/models/vitpose/modeling_vitpose.py
  transformers/src/transformers/models/vitpose/image_processing_vitpose.py
  transformers/src/transformers/models/vitpose/image_processing_pil_vitpose.py
  transformers/src/transformers/models/vitpose/convert_vitpose_to_hf.py
  transformers/src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py
  transformers/src/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py
Any missing files or assumptions: no official sampled configs were gated; no imports, tests, or model execution were run. Effective defaults are inferred from config class defaults when checkpoint JSON omits fields.
```

Source and config snapshots are summarized in `_sources/source_notes.md` and `_sources/hf_config_sweep.md`.

Representative HF model links: [`vitpose-base-simple`](https://huggingface.co/usyd-community/vitpose-base-simple), [`vitpose-base`](https://huggingface.co/usyd-community/vitpose-base), [`vitpose-base-coco-aic-mpii`](https://huggingface.co/usyd-community/vitpose-base-coco-aic-mpii), [`vitpose-plus-small`](https://huggingface.co/usyd-community/vitpose-plus-small), [`vitpose-plus-base`](https://huggingface.co/usyd-community/vitpose-plus-base), [`vitpose-plus-large`](https://huggingface.co/usyd-community/vitpose-plus-large), [`vitpose-plus-huge`](https://huggingface.co/usyd-community/vitpose-plus-huge).

## 2. High-level architecture

VitPose is a top-down human pose estimator:

```text
image + person boxes -> affine person crop preprocessing -> ViT-like image backbone -> pose decoder head -> heatmaps -> DARK/keypoint postprocess -> original-image keypoints
```

Stage decomposition:

```text
CPU/data pipeline:
  input image(s), per-person COCO-format boxes
  box -> center/scale -> affine crop to 256x192
  rescale + ImageNet normalize

GPU/runtime graph:
  pixel_values [B_person, 3, 256, 192]
  padded-stride Conv2d patch embedding -> 192 tokens
  absolute position add
  repeated noncausal ViT encoder blocks
  selected final stage LayerNorm
  token-to-grid reshape [B, S, C] -> [B, C, 16, 12]
  simple or classic decoder -> heatmaps [B, K, 64, 48]

CPU/postprocess:
  heatmap argmax
  DARK gaussian/log/Hessian coordinate refinement
  inverse center/scale mapping to image coordinates
  threshold/filter and per-input-image regrouping
```

The image processor and postprocessor can be validated independently from the neural graph. The backbone, head, and optional flip-test path should be validated as separate compiled graph slices because the output heatmaps are the neural boundary.

## 3. Important config dimensions

| Field | Source/default | Notes |
|---|---:|---|
| `image_size` | `(256, 192)` | height, width in backbone config |
| `patch_size` | `(16, 16)` | Conv2d kernel/stride, with `padding=2` |
| patch grid | `16 x 12` | source reshape uses `image_size // patch_size` |
| sequence length | `192` | no CLS token is forwarded; position table has `num_patches + 1` |
| `num_channels` | `3` | NCHW pixel values |
| `hidden_size` | `768` default | small/large/huge override |
| `num_hidden_layers` | `12` default | large 24, huge 32 |
| `num_attention_heads` | `12` default | large/huge 16 |
| `head_dim` | `hidden_size / heads` | base 64, small 32, huge 80 |
| `mlp_ratio` | `4` | hidden MLP width is `hidden_size * 4` |
| `hidden_act` | `gelu` | source uses `ACT2FN` |
| `qkv_bias` | `true` | Q/K/V linear projections have optional bias |
| `num_experts` | `1` default | VitPose+ configs use `6` |
| `part_features` | see sweep | MoE output suffix width |
| `num_labels` | `17` in sampled configs | COCO keypoints |
| `scale_factor` | `4` | simple decoder bilinear upsample |
| `use_simple_decoder` | `true` default | classic configs set false |
| dtype | `torch_dtype=float32` in sampled configs | from HF config metadata |
| cache support | none | encoder-only, no generation KV cache |

Representative checkpoint sweep:

| Model id | Decoder | Hidden | Layers | Heads | Experts | Part features | Out stage | Heatmap |
|---|---|---:|---:|---:|---:|---:|---|---|
| `usyd-community/vitpose-base-simple` | simple | 768 | 12 | 12 | 1 | 0 | `stage12` | 17 x 64 x 48 |
| `usyd-community/vitpose-base` | classic | 768 | 12 | 12 | 1 | 0 | `stage12` | 17 x 64 x 48 |
| `usyd-community/vitpose-base-coco-aic-mpii` | classic | 768 | 12 | 12 | 1 | 0 | `stage12` | 17 x 64 x 48 |
| `usyd-community/vitpose-plus-small` | classic | 384 | 12 | 12 | 6 | 96 | `stage12` | 17 x 64 x 48 |
| `usyd-community/vitpose-plus-base` | classic | 768 | 12 | 12 | 6 | 192 | `stage12` | 17 x 64 x 48 |
| `usyd-community/vitpose-plus-large` | classic | 1024 | 24 | 16 | 6 | 256 effective default | `stage24` | 17 x 64 x 48 |
| `usyd-community/vitpose-plus-huge` | classic | 1280 | 32 | 16 | 6 | 320 | `stage32` | 17 x 64 x 48 |

## 3a. Family variation traps

- The patch embedding is not standard ViT patchify: source uses `Conv2d(..., kernel_size=patch_size, stride=patch_size, padding=2)`. Any Conv2d-to-Linear rewrite needs a padding-aware guard or must be rejected.
- The source reshape assumes the selected feature map sequence length equals `(image_h // patch_h) * (image_w // patch_w)`. Configs with image/patch values that do not match the padded convolution output would break reshape/position-add parity.
- Position embeddings are added as `patch_tokens + pos[:, 1:] + pos[:, :1]`, not by prepending a CLS token.
- `out_indices` select backbone stages. The pose model uses only `outputs.feature_maps[-1]`, so multiple selected stages would still feed only the last selected map.
- Simple and classic heads require different operators: bilinear upsample plus 3x3 conv vs ConvTranspose2d/BatchNorm/ReLU blocks.
- VitPose+ MoE requires `dataset_index` at inference. Without it, the source raises when `num_experts > 1`.
- The MoE is not top-k routing. It is a dataset-indexed expert suffix with all expert linears computed in a Python loop and masked by equality.
- Official sampled configs often omit default backbone fields. DinoML config loading must fill defaults before shape planning.
- `vitpose-plus-large` omits `part_features`; effective source default is `256`, matching the converter intent for large.
- `vitpose-base-coco-aic-mpii` name suggests multi-dataset training, but the sampled HF config does not set `num_experts > 1`; with this source it follows the non-MoE MLP path.
- `flip_pairs` is an optional model input that mutates heatmap channel ordering and flips width axis. It is mostly a test-time augmentation helper, not part of ordinary single-pass heatmap generation.
- Processor `target_sizes` handling appears axis-sensitive: doc says `(height, width)`, but source assigns `image_width, image_height = target_sizes[i][0], target_sizes[i][1]`. Treat target-size coordinate handling as a parity-sensitive postprocess path.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors `[B_person, 3, 256, 192]`.
- Conv output flatten from `[B, C, 16, 12]` to `[B, C, 192]`, then transpose to `[B, 192, C]`.
- Broadcast add of position embeddings `[1, 192, C]` plus class-position bias `[1, 1, C]`.
- Token reshape after backbone: `[B, 192, C] -> permute [B, C, 192] -> reshape [B, C, 16, 12] -> contiguous`.
- Heatmap flip over width axis; channel-group reshape for optional flip-pair swap.

Neural network primitives:

- Patch embedding `Conv2d(3 -> C, kernel=16x16, stride=16x16, padding=2, bias=True)`.
- LayerNorm over hidden dimension with eps `1e-12`.
- Linear Q/K/V `(C -> C)` with configurable `qkv_bias`.
- Linear attention output `(C -> C)`.
- MLP base path: `Linear(C -> 4C) -> GELU -> Linear(4C -> C)`.
- MoE path: `Linear(C -> 4C) -> GELU -> Linear(4C -> C - part_features)` plus six `Linear(4C -> part_features)` expert suffix projections, concat on hidden dim.
- Residual adds after attention and MLP.
- Simple head: ReLU, bilinear upsample `scale_factor=4`, `align_corners=False`, `Conv2d(C -> K, 3x3, padding=1)`.
- Classic head: `ConvTranspose2d(C -> 256, 4x4, stride=2, padding=1, bias=False)`, BatchNorm2d(256), ReLU, second identical 256-to-256 block, `Conv2d(256 -> K, 1x1)`.

Attention primitives:

- Full noncausal self-attention over 192 patch tokens.
- MHA only; no GQA/MQA. Base: 12 heads x 64, small: 12 x 32, large: 16 x 64, huge: 16 x 80.
- Attention mask is always `None` in the inspected backbone call.
- Source supports eager/SDPA/FlashAttention interfaces through Transformers, but parity requires the same scaling `head_dim ** -0.5`, softmax axis `-1`, and dropout only in training.

Preprocessing-coupled ops:

- COCO box `[x, y, width, height]` to center/scale with aspect-ratio expansion and padding factor `1.25`.
- Unbiased affine warp matrix for each person crop.
- SciPy/OpenCV-like bilinear affine transform to `256x192`; channel layout goes through HWC NumPy in the processor.
- Rescale by `1/255` and ImageNet normalize.

Postprocessing ops:

- Heatmap flatten/argmax/amax per keypoint.
- Score gate: coordinates with score `<= 0` become `-1` before DARK refinement.
- DARK refinement: gaussian filter with sigma `0.8`, clip `[0.001, 50]`, log, edge pad, finite-difference derivative/Hessian, 2x2 inverse, coordinate update.
- Inverse coordinate mapping to original center/scale using output heatmap size and normalized scale factor `200`.
- COCO box to Pascal VOC box conversion with `x2=x+w-1`, `y2=y+h-1`.

## 5. Layer/block breakdown

Backbone embeddings:

```text
pixel_values: [B, 3, 256, 192]
x = Conv2d(3 -> C, kernel=16, stride=16, padding=2)(pixel_values)  # [B, C, 16, 12]
x = flatten spatial -> transpose                               # [B, 192, C]
x = x + position_embeddings[:, 1:] + position_embeddings[:, :1]
x = Dropout(p=0.0 in sampled inference configs)
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
y = LayerNorm(x)
q = Linear(C -> C, bias=qkv_bias)(y).view(B, S, H, D).transpose(1, 2)
k = Linear(C -> C, bias=qkv_bias)(y).view(B, S, H, D).transpose(1, 2)
v = Linear(C -> C, bias=qkv_bias)(y).view(B, S, H, D).transpose(1, 2)
attn = softmax((q @ k.transpose(-2, -1)) * D**-0.5)
y = (attn @ v).transpose(1, 2).reshape(B, S, C)
y = Linear(C -> C)(y)
x = residual + Dropout(y)
residual = x
y = LayerNorm(x)
y = MLP(y) or MoE_MLP(y, dataset_index)
x = residual + y
```

MoE MLP when `num_experts > 1`:

```text
h = GELU(Linear(C -> 4C)(x))
shared = Linear(4C -> C - part_features)(h)
expert_i = Linear_i(4C -> part_features)(h) * (dataset_index == i)
expert = sum_i(expert_i)
out = concat(shared, expert, dim=-1)
```

Pose head:

```text
selected_tokens = final selected backbone stage after LayerNorm
grid = selected_tokens.permute(0, 2, 1).reshape(B, C, 16, 12)

simple:
  heatmaps = Conv2d(C -> K, 3x3, padding=1)(Upsample(ReLU(grid), scale=4, bilinear, align_corners=False))

classic:
  y = ReLU(BatchNorm2d(ConvTranspose2d(C -> 256, 4x4, stride=2, padding=1)(grid)))
  y = ReLU(BatchNorm2d(ConvTranspose2d(256 -> 256, 4x4, stride=2, padding=1)(y)))
  heatmaps = Conv2d(256 -> K, 1x1)(y)
```

## 6. Attention requirements

VitPose attention is encoder-style full self-attention:

```text
causal or noncausal: noncausal
self-attention or cross-attention: self-attention only
MHA/MQA/GQA: MHA
query/key/value width: C = hidden_size
head count / head dim: config-dependent, e.g. 12 x 64 for base
query length and key/value length: S = 192 for sampled configs
masking style: no attention mask in source call
packed/varlen support: not required
sliding-window/local attention: not present
ALiBi/relative/RoPE: not present
KV cache: not applicable
FlashAttention/SDPA compatibility: source advertises support; eager fallback is dense scaled dot-product attention
```

There is no autoregressive generation, prefill, decode, or cache ABI. Independently cacheable work is at the application level: affine person crops, backbone feature maps, or heatmaps can be cached by callers, but the Transformers source does not expose a persistent neural cache.

## 7. Position encoding and custom math

Position math is absolute learned position addition:

```python
def vitpose_position_add(patch_tokens, position_embeddings):
    # patch_tokens: [B, 192, C]
    return patch_tokens + position_embeddings[:, 1:] + position_embeddings[:, :1]
```

The position parameter shape is `[1, num_patches + 1, C]`. The first slot acts like a broadcast offset; it is not prepended as a token. There is no source interpolation path for different image sizes.

Postprocess DARK coordinate refinement is custom math but processor-owned:

```python
def dark_refine(coords, heatmaps):
    hm = gaussian_filter(heatmaps, sigma=0.8)
    hm = log(clip(hm, 0.001, 50))
    dx_dy = centered_first_derivatives_at(coords, hm)
    hessian = second_derivatives_at(coords, hm)
    return coords - inv(hessian + eps * I) @ dx_dy
```

This can stay CPU/Numpy for first integration unless DinoML explicitly wants compiled end-to-end postprocess.

## 8. Preprocessing and input packing

Required processor input:

- Images plus `boxes`, where each box is COCO format `[top_left_x, top_left_y, width, height]`.
- For each input image, preprocessing emits one crop per box, so model batch dimension is number of persons, not number of original images.

Box-to-crop flow:

```text
box -> center [x + w/2, y + h/2]
aspect-ratio adjust to processor width/height ratio 192/256
scale = [adjusted_w, adjusted_h] / 200 * 1.25
warp matrix = get_warp_matrix(rotation=0, size_dst=[192, 256] - 1, size_target=scale * 200)
affine_transform per channel with bilinear interpolation
rescale + normalize
```

The processor defaults are `size=256x192`, `do_affine_transform=true`, `normalize_factor=200`, ImageNet mean/std. The inspected Torchvision processor calls SciPy in `affine_transform`; the PIL implementation is a parallel backend with equivalent postprocess helpers.

Input packing constraints:

- The model expects already-cropped/warped pixel values in NCHW.
- `target_sizes` only affect postprocess box scaling. They do not alter neural input shape.
- No attention masks, token type IDs, sequence packing metadata, or image placeholder tokens exist.

Postprocess output:

- `post_process_pose_estimation` returns `list[list[dict]]`, grouped by original image and input box.
- Each dict contains `keypoints [kept_K, 2]`, `scores [kept_K]`, `labels [kept_K]`, and `bbox` in Pascal VOC-like xyxy form.

## 9. Graph rewrite / lowering opportunities

### Rewrite: guarded padded patch Conv2d to GEMM/im2col

Source pattern:

```text
Conv2d(3 -> C, kernel=16x16, stride=16x16, padding=2)
```

Replacement:

```text
Pad2d(2) -> Unfold(kernel=16x16, stride=16x16) -> GEMM(weight_flat.T) -> BiasAdd -> TokenTranspose
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 2`
- `dilation == 1`
- `groups == 1`
- runtime input size exactly equals configured `image_size`
- unfolded patch count equals `(image_h // patch_h) * (image_w // patch_w)`

Failure cases:

- Do not use a padding-free Conv-to-Linear patchify rewrite.
- Reject configs where convolution output grid does not equal configured grid.

Parity test sketch:

- Random NCHW tensors for 256x192 and each hidden size; compare Conv2d output flattened to token sequence against rewrite.

### Rewrite: simple head upsample-conv fusion

Source pattern:

```text
ReLU -> bilinear Upsample(scale_factor=4, align_corners=False) -> Conv2d(C -> 17, 3x3, padding=1)
```

Replacement:

```text
fused ReLU + bilinear resize + small-output-channel conv
```

Preconditions:

- Static scale factor `4`
- `align_corners=False`
- input grid `[16, 12]`
- conv stride `1`, padding `1`, dilation `1`, groups `1`

Failure cases:

- Different scale factors or dynamic output sizes need a general resize op.
- Fusing must preserve PyTorch bilinear half-pixel coordinate convention.

### Rewrite: classic head deconv to optimized transposed-conv

Source pattern:

```text
ConvTranspose2d(k=4, stride=2, padding=1, bias=False) -> BatchNorm2d -> ReLU
```

Replacement:

```text
provider-backed ConvTranspose2d or lowered scatter/im2col-transpose path; optionally fold BatchNorm into deconv weights for inference
```

Preconditions:

- eval mode with frozen BatchNorm stats for BN folding
- exact transposed-conv output padding default `0`
- NCHW layout preserved or all axis rewrites explicit

Failure cases:

- Training mode BatchNorm cannot be folded.
- A normal Conv2d rewrite is not equivalent to ConvTranspose2d.

### Rewrite: dataset-index MoE expert selection

Source pattern:

```text
sum_i Linear_i(h) * (dataset_index == i)
```

Replacement:

```text
if dataset_index is batch-uniform: dispatch only selected expert
else compute/gather per sample by expert bucket
```

Preconditions:

- `num_experts > 1`
- `dataset_index` available and integer in `[0, num_experts)`
- preserve concat order `[shared, expert_suffix]`

Failure cases:

- Missing `dataset_index` must raise, matching source.
- Mixed batch expert routing needs stable sample regrouping to preserve order.

## 10. Kernel fusion candidates

Highest priority:

- Patch embedding Conv2d/im2col/GEMM: first large image-to-token cost and layout boundary; must preserve padding=2.
- Encoder LayerNorm + QKV projections: repeated in every layer and shape-stable at 192 tokens.
- Dense MHA over 192 tokens: small sequence length but many layers; SDPA/FlashAttention path can be used if overhead is reasonable.
- Classic head ConvTranspose2d blocks: required by most sampled checkpoints, including VitPose+.

Medium priority:

- MLP GEMMs with GELU: dominant per-layer dense compute.
- MoE expert suffix dispatch: important for VitPose+ to avoid computing all six experts when `dataset_index` is uniform.
- Simple head resize + conv: required by `vitpose-base-simple`, low output channel count but parity-sensitive.
- Token/grid layout fusion around final `permute -> reshape -> head`.

Lower priority:

- Heatmap flip-pair helper: optional test-time augmentation.
- CPU postprocess acceleration: useful for throughput with many persons, but can initially remain processor-owned.
- BatchNorm folding for classic head: straightforward inference optimization after ConvTranspose2d parity exists.

## 11. Runtime staging plan

Stage 1: Config and processor metadata loading.

- Fill effective backbone defaults from source classes.
- Admit only official 256x192, 16x16 patch configs initially.
- Keep affine crop/postprocess in Python/CPU.

Stage 2: Neural graph parity for base simple.

- Implement patch embedding with padding=2, position add, 12-layer encoder, simple head.
- Validate heatmaps for `vitpose-base-simple`.

Stage 3: Classic decoder parity.

- Add ConvTranspose2d and BatchNorm2d inference path.
- Validate `vitpose-base`.

Stage 4: VitPose+ MoE.

- Require `dataset_index`.
- Implement expert suffix selection and concat.
- Validate small/base/large/huge shape variants.

Stage 5: Optimized attention and fusion.

- Add SDPA/FlashAttention admission for fixed 192-token noncausal MHA.
- Add patch Conv2d rewrite and head fusions behind guards.

Stage 6: End-to-end pose pipeline.

- Integrate affine crop metadata, per-person regrouping, DARK postprocess, thresholding, and target-size behavior.

## 12. Parity and validation plan

- Processor parity: fixed image/box examples for `box_to_center_and_scale`, `get_warp_matrix`, affine crop, rescale/normalize. Tolerance should match float32 CPU/SciPy path; compare pixel tensors before model.
- Patch embedding parity: random `[B,3,256,192]`; compare Conv2d output and token order.
- Single encoder block parity: random tokens and weights for base, small, huge head dims; fp32 tolerance around `1e-5`, fp16 around `1e-2` depending backend.
- Full backbone parity: compare selected feature map `[B,192,C]`.
- Simple head parity: compare heatmaps `[B,17,64,48]` for bilinear `align_corners=False`.
- Classic head parity: compare both deconv blocks and final heatmaps; include BatchNorm eval stats.
- MoE parity: dataset indices for each expert and mixed batch; assert missing index raises.
- Flip helper parity: known `flip_pairs` over synthetic heatmaps to catch channel and width-axis mistakes.
- Postprocess parity: synthetic heatmaps with known peaks; validate argmax, DARK offsets, inverse mapping, threshold filtering, and grouped result structure.
- End-to-end parity: official image + boxes from source example; compare heatmaps first, then keypoints/scores after postprocess.

## 13. Performance probes

- Preprocessing throughput: images/sec and crops/sec for affine crop plus normalization, varied number of boxes per image.
- Backbone-only throughput: batch/person count sweep for 256x192 crops.
- Attention backend comparison: eager vs SDPA vs FlashAttention for S=192 and hidden sizes 384/768/1024/1280.
- Patch embedding lowering: native Conv2d vs im2col/GEMM rewrite.
- Head-only throughput: simple upsample+conv vs classic deconv blocks.
- MoE routing probe: compute-all masked expert source pattern vs selected-expert/bucketed lowering.
- End-to-end persons/sec split into preprocess, neural heatmaps, postprocess.
- Memory probe: activation memory across layers and hidden sizes; no KV cache memory applies.

## 14. Skip/defer list

- Training and losses are explicitly unsupported by the source forward.
- Gradient checkpointing.
- Dynamic image sizes and position embedding interpolation.
- Non-official keypoint label sets until configs with different `num_labels` are inspected.
- Combined-target heatmap mode; helper supports it, but current heads emit gaussian heatmaps.
- Fully compiled DARK postprocess; keep CPU/Numpy first.
- Multi-GPU/tensor parallel.
- Quantization and packed weights; no source-coupled quantized format appears in inspected configs.
- General timm/pretrained external backbone loading until a real VitPose config requires it.

## 15. Final implementation checklist

- [ ] Parse `VitPoseConfig` and nested `VitPoseBackboneConfig` with effective source defaults.
- [ ] Load weights and preserve stage/output selection metadata.
- [ ] Implement padded patch embedding `Conv2d(..., padding=2)`.
- [ ] Implement absolute position add `pos[:, 1:] + pos[:, :1]`.
- [ ] Implement noncausal MHA over fixed patch tokens.
- [ ] Implement base MLP and residual LayerNorm block.
- [ ] Implement simple decoder heatmap head.
- [ ] Implement classic ConvTranspose2d/BatchNorm/ReLU decoder.
- [ ] Implement VitPose+ `dataset_index` MoE suffix path.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite.
- [ ] Add heatmap flip-pair helper or explicitly keep it as a post/head helper.
- [ ] Keep affine crop preprocessing in CPU pipeline with parity tests.
- [ ] Keep DARK keypoint postprocess in CPU pipeline with parity tests.
- [ ] Validate base-simple heatmaps.
- [ ] Validate base classic heatmaps.
- [ ] Validate VitPose+ MoE shape and expert-selection parity.
- [ ] Benchmark preprocess, backbone, decoder, postprocess, and end-to-end persons/sec separately.
