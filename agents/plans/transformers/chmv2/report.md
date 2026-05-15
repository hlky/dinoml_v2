# CHMv2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/dinov3-vitl16-chmv2-dpt-head
Config source: official repo is gated; open ONNX mirror config used only as representative checkpoint snapshot.
Source files inspected:
- transformers/src/transformers/models/chmv2/configuration_chmv2.py
- transformers/src/transformers/models/chmv2/modeling_chmv2.py
- transformers/src/transformers/models/chmv2/modular_chmv2.py
- transformers/src/transformers/models/chmv2/image_processing_chmv2.py
- transformers/src/transformers/models/dinov3_vit/configuration_dinov3_vit.py
- transformers/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py
Any missing files or assumptions: facebook/dinov3-vitl16-chmv2-dpt-head config/preprocessor files returned gated/auth-required errors. HF API metadata was accessible and marks the repo gated="manual". The ONNX mirror is not authoritative for weights but matches the native architecture/config shape.
```

Primary links:

- Official gated checkpoint: [facebook/dinov3-vitl16-chmv2-dpt-head](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head)
- Open mirror config snapshot: [onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX](https://huggingface.co/onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX)
- Transformers docs page: [CHMv2](https://huggingface.co/docs/transformers/en/model_doc/chmv2)

`configuration_chmv2.py`, `modeling_chmv2.py`, and `image_processing_chmv2.py` are generated from `modular_chmv2.py`; future source edits should target the modular file. This report treats the generated files as the runtime source of truth and `modular_chmv2.py` as the authoring source.

## 2. High-level architecture

CHMv2 is an image-only dense depth/canopy-height estimator. It is not a language model and has no decode, logits sampling, token cache, or text generation ABI.

```text
CPU/image pipeline -> channels-first pixel_values
  -> DINOv3 ViT backbone feature stages + CLS tokens
  -> DPT-style reassemble/readout
  -> multi-scale convolutional feature fusion
  -> upsample convolutional depth-bin head
  -> depth-bin normalization -> predicted_depth [B,H,W]
  -> optional postprocess resize to target image size
```

Stage decomposition:

- CPU/data pipeline: RGB conversion inherited from image processor base, optional resize, rescale, normalize, center padding to multiples of 16.
- Backbone stage: DINOv3 ViT-L/16 produces selected feature maps from stages 6/12/18/24 and matching CLS tokens.
- Head stage: CHMv2-owned DPT-style reassembly, conv fusion, depth-bin conversion.
- Postprocess: optional bilinear resize of `predicted_depth` to caller target size.

The backbone should be composed with a separate `dinov3_vit` audit for full ViT operator coverage. This CHMv2 report owns the depth head, preprocessing, postprocessing, and the exact DINOv3 backbone feature contract it consumes.

## 3. Important config dimensions

Source defaults from `CHMv2Config`:

| Field | Default / effective value | Source |
|---|---:|---|
| `model_type` | `chmv2` | source default |
| `patch_size` | 16 | source default |
| `reassemble_factors` | `[4, 2, 1, 0.5]` | source default |
| `post_process_channels` | `[128, 256, 512, 1024]` | source default |
| `fusion_hidden_size` | 256 | source default |
| `head_hidden_size` | 128 | source default |
| `number_output_channels` | 256 depth bins | source default |
| `readout_type` | `project` | source default |
| `min_depth`, `max_depth` | `0.001`, `96.0` | source default |
| `bins_strategy` | `chmv2_mixlog` | source default |
| `norm_strategy` | `chmv2_mixlog` | source default |

Default nested backbone config injected by `CHMv2Config`:

| Field | Effective value | Runtime significance |
|---|---:|---|
| backbone type | `dinov3_vit` | delegated ViT body |
| `hidden_size` | 1024 | feature channel width |
| `intermediate_size` | 4096 | MLP width |
| `num_hidden_layers` | 24 | feature stages through `stage24` |
| `num_attention_heads` | 16 | head_dim = 64 |
| `num_register_tokens` | 4 | prefix length = CLS + 4 registers |
| `out_indices` | `[6, 12, 18, 24]` | selected feature stages |
| `return_class_token` | true | CHMv2 readout consumes CLS |
| `reshape_hidden_states` | true | backbone returns NCHW feature maps |
| `apply_layernorm` | true | stage maps are layernormed |
| `key_bias` | source default says true; mirror config says false | checkpoint-sensitive; see traps |
| `layer_norm_eps` | source default says `1e-6`; mirror config says `1e-5` | checkpoint-sensitive |

Representative checkpoint sweep:

| Repo/config | Access | Significant fields | Notes |
|---|---|---|---|
| `facebook/dinov3-vitl16-chmv2-dpt-head` | gated manual approval | architecture `CHMv2ForDepthEstimation`, F32 safetensors metadata has 336,876,800 parameters | Official source for weights/config, but raw files require auth. |
| `onnx-community/dinov3-vitl16-chmv2-dpt-head-ONNX` | open mirror | same `chmv2` architecture; DINOv3 ViT-L/16 backbone; `key_bias=false`, `layer_norm_eps=1e-5`, `dtype=float32` | Labeled mirror only. Config snapshot saved beside this report. |
| Source default `CHMv2Config()` | local source | DINOv3 ViT-L/16-ish defaults with `image_size=416`, `key_bias=true`, `layer_norm_eps=1e-6` | Effective defaults may differ from released checkpoint config. |

HF model search found only one native CHMv2 checkpoint and one ONNX mirror, so a 3-5 checkpoint sweep is not available from official open repos.

## 3a. Family variation traps

- `backbone_config` is nested and auto-loaded; DinoML should admit only audited backbones at first, starting with `dinov3_vit` ViT-L/16.
- Source default and mirror config differ in DINOv3 `key_bias` and `layer_norm_eps`; load-time config must drive weight shape/admission rather than hard-coded defaults.
- CHMv2 assumes `return_class_token=True` when `readout_type="project"` because each feature is paired with a CLS token.
- `reshape_hidden_states=True` makes the backbone return NCHW feature maps. If false, CHMv2 has a sequence-to-NCHW fallback, but the paired `(feature_map, cls_token)` path in current DINOv3 backbone returns image maps.
- `readout_type` changes graph shape: `project` uses concat + Linear(2H -> H) + GELU; `add` uses broadcast add; `ignore` skips CLS handling.
- `reassemble_factors` include transposed convolution for factors > 1 and stride-2 convolution for factor 0.5.
- Bilinear interpolation uses both `align_corners=False` and `align_corners=True` in different places; these are parity-sensitive.
- The depth conversion has four strategy combinations (`linear`, `softmax`, `sigmoid`, `chmv2_mixlog`) but released config uses `chmv2_mixlog`.
- NCHW is the source semantic layout for model conv/interpolate code. NHWC is only an optimization candidate behind guarded conv/interpolate layout rewrites.
- Training labels immediately raise `NotImplementedError`; first runtime should reject/ignore training loss paths.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor `[B,3,H,W]`, with `H,W` multiples of 16 after processor padding.
- Flatten patches, transpose, reshape, permute, contiguous.
- Slice/select prefix and patch tokens; `hidden_state[:, 1:]` fallback path.
- Tuple/list handling for `(feature_map, cls_token)` intermediate features.
- `cat` on last dim for readout projection.
- `expand_as` for CLS broadcast over patch positions.
- `squeeze(dim=1)` for final depth output.

Neural network primitives:

- DINOv3 patch embedding: Conv2d(3 -> 1024, kernel=16, stride=16).
- DINOv3 encoder: LayerNorm, Linear Q/K/V/O, noncausal MHA, GELU MLP Linear(1024 -> 4096 -> 1024), layer scale multiply, residual adds.
- CHMv2 readout projection: Linear(2048 -> 1024) + GELU per selected stage when `readout_type=project`.
- CHMv2 reassemble projection: Conv2d(1024 -> 128/256/512/1024, kernel=1).
- Reassemble resize: ConvTranspose2d for factors 4 and 2; Identity for factor 1; Conv2d(kernel=3, stride=2, padding=1) for factor 0.5.
- Head feature projection convs: Conv2d(128/256/512/1024 -> 256, kernel=3, padding=1, bias=False).
- Fusion residual conv unit: ReLU -> Conv2d(256 -> 256, 3x3, bias=True) -> ReLU -> Conv2d(256 -> 256, 3x3, bias=True) -> residual add.
- Fusion projection: Conv2d(256 -> 256, kernel=1, bias=True).
- Depth head: Conv2d(256 -> 128, 3x3) -> bilinear upsample x2 -> Conv2d(128 -> 128, 3x3) -> ReLU -> Conv2d(128 -> 256, 1x1).

Attention primitives:

- Backbone-only noncausal self-attention over prefix + patch tokens.
- MHA with 16 query heads and 16 key/value heads for the representative ViT-L/16 config; no GQA/MQA.
- RoPE applies only to patch tokens, not CLS/register prefix tokens.
- No autoregressive KV cache.

Position/depth custom math:

- 2D patch-center coordinate generation, RoPE cos/sin, rotate-half.
- `linspace`, `log`, `exp`, `relu`, `amin(dim=1)`, `clamp_min`, `clamp_max`, `nan_to_num`, reductions over bin/channel dimension, elementwise divide/multiply/sum.

Preprocessing/postprocessing-coupled ops:

- Image rescale/normalize with mean `[0.420,0.411,0.296]`, std `[0.213,0.156,0.143]`.
- Center padding to `size_divisor=16`.
- Optional processor resize with aspect-ratio/multiple-of guards.
- Optional output depth resize with bilinear interpolation, `align_corners=True`.

## 5. Layer/block breakdown

Backbone feature contract consumed by CHMv2:

```text
pixel_values [B,3,H,W]
patch grid: Hp = H // 16, Wp = W // 16
DINOv3 outputs:
  feature_maps: 4 tensors, each [B,1024,Hp,Wp] for stages 6/12/18/24
  cls_tokens: 4 tensors, each [B,1024]
```

CHMv2 reassemble stage, repeated for the four selected stages:

```text
feature [B,1024,Hp,Wp], cls [B,1024]
if readout_type == project:
  seq = feature.flatten(2).transpose(1,2)                 # [B,Hp*Wp,1024]
  cls = cls.unsqueeze(1).expand_as(seq)                   # [B,Hp*Wp,1024]
  seq = GELU(Linear(cat(seq, cls), 2048 -> 1024))
  feature = seq.permute(0,2,1).reshape([B,1024,Hp,Wp])
feature = Conv1x1(1024 -> C_i)
feature = resize_i(feature)
```

With representative factors/channels:

- Stage 6: `C=128`, ConvTranspose2d kernel/stride 4, output roughly `[B,128,4Hp,4Wp]`.
- Stage 12: `C=256`, ConvTranspose2d kernel/stride 2, output `[B,256,2Hp,2Wp]`.
- Stage 18: `C=512`, Identity, output `[B,512,Hp,Wp]`.
- Stage 24: `C=1024`, Conv2d 3x3 stride 2 padding 1, output roughly `[B,1024,ceil(Hp/2),ceil(Wp/2)]`.

Feature fusion:

```text
features_i = Conv3x3(C_i -> 256, bias=False)
reverse features from low-resolution to high-resolution
fused = fusion_layer0(features[0])
for each next feature:
  if sizes differ: residual = interpolate(residual, size=fused_hw, align_corners=False)
  fused = fused + ResidualConvUnit(residual)
  fused = ResidualConvUnit(fused)
  fused = interpolate(fused, scale_factor=2, align_corners=True)
  fused = Conv1x1(256 -> 256)
```

Depth-bin head:

```text
Conv3x3(256 -> 128)
BilinearUpsample(scale=2, align_corners=True)
Conv3x3(128 -> 128)
ReLU
Conv1x1(128 -> number_output_channels)
FeaturesToDepth
squeeze channel -> predicted_depth [B,Hout,Wout]
```

## 6. Attention requirements

Attention is required only inside the delegated DINOv3 ViT backbone.

- Type: encoder-style noncausal self-attention.
- Shape: hidden states `[B, T, 1024]`, where `T = 1 + num_register_tokens + Hp*Wp`; representative prefix length is 5.
- Heads: MHA, `num_attention_heads=16`, `head_dim=64`; key/value head count is also 16.
- Projections: separate Q/K/V/O linear layers. Representative mirror config has Q/V/O bias true and K bias false; source CHMv2 default injects K bias true.
- Masking: CHMv2 forward does not supply an attention mask. DINOv3 attention accepts one through generic kwargs.
- Backend: source dispatches through `ALL_ATTENTION_FUNCTIONS` and supports eager/SDPA/Flash/Flex attention via Transformers flags.
- Cache: no KV cache, no decode-time state, no causal mask.
- RoPE: Q/K patch tokens are rotated after projection/reshape and before attention. CLS/register tokens pass through unrotated.

## 7. Position encoding and custom math

DINOv3 uses 2D RoPE over patch-center coordinates. Coordinates are recomputed dynamically from runtime image size and patch size, in fp32, then cast to pixel dtype.

```python
def chmv2_dinov3_rope(q, k, cos, sin):
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix = num_tokens - num_patches
    q_prefix, q_patch = q.split((num_prefix, num_patches), dim=-2)
    k_prefix, k_patch = k.split((num_prefix, num_patches), dim=-2)
    q_patch = q_patch * cos + rotate_half(q_patch) * sin
    k_patch = k_patch * cos + rotate_half(k_patch) * sin
    return cat(q_prefix, q_patch, dim=-2), cat(k_prefix, k_patch, dim=-2)
```

Depth conversion for released config uses `chmv2_mixlog` bins plus `chmv2_mixlog` normalization:

```python
scaled_max = max_depth / 8
linear = linspace(min_depth, scaled_max, n_bins)
log = exp(linspace(log(min_depth), log(scaled_max), n_bins))
w = linspace(1, 0, n_bins)
bins = w * log + (1 - w) * linear

logits = relu(head_output)
shift = clamp((-amin(logits, dim=1)).clamp_min(0), max=1e-4) + 1e-8
weights = (logits + shift) / clamp_min(nan_to_num(sum(logits + shift, dim=1)), 1e-12)
depth = clamp_min(sum(weights * bins, dim=1, keepdim=True), 1e-12) * 8
```

Bins can be cached per `(n_bins, min_depth, max_depth, strategy, dtype, device)` for inference, except the tensor device/dtype must match runtime.

## 8. Preprocessing and input packing

Processor contract from source and open mirror:

- Input images are prepared into `pixel_values` with channels-first layout.
- Default source has `do_resize=False`, `do_rescale=True`, `do_normalize=True`, `do_pad=True`.
- Mean/std are canopy-specific: mean `[0.420, 0.411, 0.296]`, std `[0.213, 0.156, 0.143]`.
- Padding is centered so both height and width become multiples of `size_divisor=16`; pad split is floor/ceil around each spatial axis.
- Optional resize computes an aspect-ratio-preserving size rounded to `ensure_multiple_of=16` when enabled.
- Segmentation-map handling and `reduce_label` exist in image processor but are not used by `CHMv2ForDepthEstimation` inference.

Runtime input:

```text
pixel_values: float tensor [B,3,H,W], source NCHW, H%16==0, W%16==0 recommended/processor-guaranteed
```

Postprocessing:

- Model returns `predicted_depth` `[B,Hout,Wout]`.
- `post_process_depth_estimation` optionally resizes each depth map to a caller `target_size=(height,width)` using bilinear interpolation with `align_corners=True`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: DINOv3 patch Conv2d -> patch Linear

Preconditions:

- Conv2d kernel size equals stride equals `patch_size`.
- Padding and dilation are default zero/one.
- Groups is 1.
- Input is NCHW contiguous or a guarded layout-equivalent.
- Runtime `H,W` divisible by `patch_size`.

Replacement:

```text
NCHW image -> non-overlapping patch flatten [B,Hp*Wp,3*P*P]
  -> Linear(3*P*P -> hidden_size)
```

Weight transform:

```python
w_linear = conv.weight.reshape(hidden_size, 3 * patch_size * patch_size)
b_linear = conv.bias
```

Failure cases: non-divisible image size, nonzero padding, alternate patch size tuple without matching im2col order, or NHWC rewrite without explicit axis/order handling.

Parity sketch: compare patch embeddings before CLS/register concat for random `[B,3,416,416]` and odd-but-padded sizes.

### Rewrite: Conv1x1 -> per-pixel GEMM

Preconditions:

- Kernel size 1, stride 1, padding 0, dilation 1, groups 1.
- Source NCHW preserved or NHWC transform updates weight/layout and all consumers.

Replacement:

```text
[B,C,H,W] -> [B*H*W,C] GEMM W.T + bias -> reshape
```

Applies to reassemble projections and fusion projection.

### Rewrite: readout project as batched GEMM

Preconditions:

- `readout_type="project"`.
- Feature map is `[B,H,Hp,Wp]`, CLS `[B,H]`.
- CLS expansion is pure broadcast over patch positions.

Replacement:

```text
feature_NCHW -> sequence [B,Hp*Wp,H]
concat with broadcast CLS -> Linear(2H -> H) -> GELU -> NCHW
```

Can avoid materializing full expanded CLS by using a fused concat/bias GEMM or two GEMMs added before bias.

### Rewrite: bin weighted sum -> channel reduction

Preconditions:

- `number_output_channels` fixed for artifact or bins cached per n_bins.
- Strategy is admitted (`chmv2_mixlog` first).

Replacement:

```text
relu/shift/sum/div -> weighted channel sum with static bins vector
```

Failure cases: alternate `norm_strategy` should route to separately tested kernels.

### Layout rewrite: guarded NHWC conv/fusion region

Candidate region starts after backbone feature map production and can cover CHMv2 conv/interpolate-heavy head. Required guards:

- Rewrite every conv/interpolate/reduction axis consistently.
- Preserve depth-bin channel dimension for reductions (`dim=1` in NCHW becomes last channel in NHWC).
- Protect DINOv3 sequence/RoPE attention with a no-layout-translation boundary.
- Preserve `align_corners` values exactly for interpolation.

## 10. Kernel fusion candidates

Highest priority:

- DINOv3 LayerNorm + QKV projections + RoPE handoff, owned by `dinov3_vit` integration.
- Noncausal ViT Flash/SDPA attention with prefix-unrotated RoPE handling.
- CHMv2 depth-bin conversion for `chmv2_mixlog`: avoids many tiny elementwise/reduction kernels at the output.
- Conv2d + ReLU blocks in residual fusion head, especially 3x3 256-channel convs.

Medium priority:

- Patch Conv2d to GEMM or optimized patch embedding kernel.
- Readout projection fusion: CLS broadcast + concat + Linear + GELU.
- Conv1x1 per-pixel GEMM for projection-heavy parts.
- Bilinear upsample + following Conv2d fusion where layout and interpolation parity are controlled.

Lower priority:

- Alternate depth strategies (`linear`, `softmax`, `sigmoid`) unless configs require them.
- Segmentation-map preprocessing helpers.
- Training-only stochastic depth/drop path behavior.

## 11. Runtime staging plan

1. Parse CHMv2 config and nested `dinov3_vit` config; reject non-DINOv3 backbones initially.
2. Load weights with explicit namespaces for backbone, readout projections, reassemble layers, fusion layers, and depth head.
3. Compose an audited DINOv3 ViT-L/16 backbone that returns selected NCHW feature maps plus CLS tokens.
4. Implement CHMv2 head in faithful NCHW first: readout projection, reassemble resize ops, feature fusion, depth head.
5. Implement `chmv2_mixlog` depth conversion and postprocess resize.
6. Add guarded optimized rewrites for patch embedding, Conv1x1/GEMM, and CHMv2 head NHWC only after faithful parity.
7. Add optional optimized DINOv3 attention/fusion paths through the backbone audit.

Initial stubs: skip labels/training loss, segmentation postprocess, alternate depth strategies, and non-DINOv3 backbones.

## 12. Parity and validation plan

- Config parsing parity: source defaults vs mirror config, including `key_bias` and `layer_norm_eps`.
- Unit parity for `FeaturesToDepth` with all strategies, prioritizing `chmv2_mixlog`; fp32 tolerance `rtol=1e-5`, `atol=1e-5`.
- Unit parity for reassemble stage with synthetic four feature maps and CLS tokens; include `project`, `add`, and `ignore`.
- Conv/interpolate parity for `FeatureFusionLayer`, checking both `align_corners=False` residual resize and `align_corners=True` upsample.
- One-block/head parity: feed captured or random DINOv3-like feature maps into CHMv2 head and compare output depth bins.
- End-to-end checkpoint parity after gated access: compare `predicted_depth` against Transformers for representative padded images.
- Postprocess parity: resize predicted depth to multiple target sizes, including non-square.
- Recommended tolerances: fp32 `1e-5`/`1e-5`; fp16/bf16 head path likely needs `rtol=2e-2`, `atol=2e-2` around interpolation and reductions until kernels are stabilized.

## 13. Performance probes

- Processor throughput: resize/rescale/normalize/pad images per second.
- Backbone-only throughput over image sizes such as 384, 416, 512, and native padded non-square inputs.
- CHMv2 head-only throughput with fixed synthetic stage features.
- Depth conversion kernel time vs head conv time.
- End-to-end latency and throughput by batch size.
- Spatial-size sweep with `H,W` multiples of 16; include non-square satellite-like tiles.
- Attention backend comparison in DINOv3: eager vs SDPA vs Flash/Flex where supported.
- NCHW faithful head vs guarded NHWC conv/interpolate head.
- Memory probes: peak feature-map memory for selected stages and fusion temporaries.

## 14. Skip/defer list

- Training and labels loss path; source raises `NotImplementedError`.
- Generation, beam search, KV cache, tokenizer concerns; not applicable.
- Segmentation postprocess helper in image processor; not used by depth model.
- Non-DINOv3 backbones until separately audited and allowlisted.
- Alternate `readout_type` and depth strategies beyond the released config can be second-stage unless a target checkpoint uses them.
- DropPath/training-time position augmentation; inference path should run with eval mode.
- ONNX quantized mirror formats as weight-loading targets; treat separately from native Transformers safetensors.

## 15. Final implementation checklist

- [ ] Parse `CHMv2Config` and nested `DINOv3ViTConfig`.
- [ ] Gate first integration to `backbone_config.model_type == "dinov3_vit"`.
- [ ] Load official gated checkpoint config after access is available.
- [ ] Compose audited DINOv3 backbone feature contract: four NCHW feature maps plus CLS tokens.
- [ ] Implement CHMv2 readout projection modes.
- [ ] Implement reassemble Conv1x1 + ConvTranspose2d/Identity/stride-2 Conv2d.
- [ ] Implement CHMv2 feature projection and fusion layers.
- [ ] Implement depth-bin head and `chmv2_mixlog` conversion.
- [ ] Implement processor-compatible NCHW input guards and center-padding metadata.
- [ ] Implement optional postprocess depth resize.
- [ ] Add parity tests for `FeaturesToDepth`.
- [ ] Add head-only parity tests from synthetic/captured backbone features.
- [ ] Add end-to-end parity once gated weights are accessible.
- [ ] Benchmark backbone-only, head-only, depth-conversion-only, and end-to-end paths.
