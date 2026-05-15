# EfficientLoFTR Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: zju-community/efficientloftr
Config source: https://huggingface.co/zju-community/efficientloftr/raw/main/config.json
Source files inspected:
  transformers/src/transformers/models/efficientloftr/configuration_efficientloftr.py
  transformers/src/transformers/models/efficientloftr/modeling_efficientloftr.py
  transformers/src/transformers/models/efficientloftr/modular_efficientloftr.py
  transformers/src/transformers/models/efficientloftr/image_processing_efficientloftr.py
  transformers/src/transformers/models/efficientloftr/image_processing_pil_efficientloftr.py
  transformers/src/transformers/models/efficientloftr/convert_efficientloftr_to_hf.py
Any missing files or assumptions:
  No DinoML code was inspected or edited for this report.
  Only native Transformers EfficientLoFTR is in scope. Raw .ckpt/.pth/ONNX repos are not treated as native configs.
```

Primary task: image-pair keypoint matching / local feature matching, not text generation. Primary DinoML target should be `EfficientLoFTRForKeypointMatching` inference with image-pair input and ragged per-pair match output.

Generated-source note: the processor files are generated from `modular_efficientloftr.py` where applicable, and generated files carry the warning to edit modular source. The full modeling body is in `modeling_efficientloftr.py`.

Representative configs checked:

- [zju-community/efficientloftr](https://huggingface.co/zju-community/efficientloftr), repo sha `face1a79050ffa3e9da28720d1cf93aaf2e8f421`, `model_type=efficientloftr`, `EfficientLoFTRForKeypointMatching`, 16,050,816 F32 parameters from HF repo metadata, processor default `480x640`.
- [stevenbucaille/efficientloftr](https://huggingface.co/stevenbucaille/efficientloftr), repo sha `9aec0b3da50cdf02656b5ac2162a670d0e2013af`, native Transformers mirror with same model type and parameter metadata but historical config keys.
- Public raw artifact repos found but out of scope for native-source parity: [xmanifold/efficient_loftr](https://huggingface.co/xmanifold/efficient_loftr), [stevenbucaille/efficient_loftr_pth](https://huggingface.co/stevenbucaille/efficient_loftr_pth), [kornia/Efficient_LOFTR](https://huggingface.co/kornia/Efficient_LOFTR), [zahilaty/EfficientLoFTR-ONNX](https://huggingface.co/zahilaty/EfficientLoFTR-ONNX).

Additional notes are in `_sources/source_notes.md`.

## 2. High-level architecture

EfficientLoFTR is a detector-free local feature matcher. It consumes pairs of images and outputs matched keypoints plus matching scores. There is no tokenizer, no logits, no generation controller, and no KV cache.

Dataflow:

```text
image-pair preprocessing -> grayscale NCHW tensor -> RepVGG CNN pyramid
  -> coarse local feature transformer with aggregated self/cross attention
  -> coarse all-pairs matching and mutual-nearest filtering
  -> fine feature pyramid fusion and local window unfold
  -> matched-window fine correlations -> subpixel refinement
  -> normalized keypoints/match indices/scores -> optional postprocess to original pixels
```

Stage decomposition:

- CPU/data pipeline: validate image pairs, resize, rescale, grayscale conversion, stack pairs into `pixel_values`.
- GPU/runtime neural body: RepVGG backbone, aggregated attention, fine fusion convolutions and interpolation.
- Matching/postprocess-like body currently inside `forward`: coarse similarity, threshold/border masking/mutual-nearest filtering, window gathers by match indices, fine correlation, 3x3 softargmax-style expectation.
- External postprocess: rescale normalized keypoints to original image sizes and filter score/match validity. No NMS.

Independently stageable targets:

1. Backbone feature maps from `EfficientLoFTRModel`.
2. Coarse transformed features and coarse match scores/indices.
3. Fine fusion windows for known coarse indices.
4. End-to-end keypoints with ragged/padded output ABI.

## 3. Important config dimensions

Default and official checkpoint dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| `stage_num_blocks` | `[1, 2, 4, 14]` | config/default |
| `stage_stride` | `[2, 1, 2, 2]` | config/default |
| `out_features` | `[64, 64, 128, 256]` | config/default |
| input channels used by model | `1` | source selects first grayscale channel |
| `hidden_size` | `256` | config/default |
| `intermediate_size` | `512` | config post-init |
| `num_attention_layers` | `4` | config/default |
| `num_attention_heads` | `8` | config/default |
| `num_key_value_heads` | `8` | config post-init |
| `head_dim` | `32` | inferred from `256 / 8` |
| attention bias/dropout | `false` / `0.0` | config |
| Q aggregation | depthwise Conv2d `k=4,s=4` | config/source |
| KV aggregation | MaxPool2d `k=4,s=4` | config/source |
| coarse matching temperature/threshold | `0.1` / `0.2` | config |
| border removal | `2` coarse cells | config |
| fine window | `8x8` for image0, `10x10` padded window for image1 | config/source |
| fine slice dim | `8` | config |
| processor size | `480x640` | preprocessor config |
| default coarse map at processor size | `60x80` | inferred from strides |
| default attention token grid | `15x20` | inferred from coarse map and Q aggregation |
| dtype | `float32` | config/HF metadata |

Representative checkpoint/config sweep:

| Repo | Native Transformers config? | Operator-significant notes |
| --- | --- | --- |
| `zju-community/efficientloftr` | Yes | Current official config includes derived fields such as `fine_fusion_dims`, `stage_block_*`, `embedding_size=[15,20]`, and default processor config. |
| `stevenbucaille/efficientloftr` | Yes | Same architecture but historical keys include `stage_block_dims` and `stage_hidden_expansion`; current strict config class does not declare these fields. Treat as a config-admission trap unless loading path proves compatibility. |
| raw `.ckpt` / `.pth` repos | No | Need conversion or separate external-loader audit; not enough native config surface to admit automatically. |
| ONNX repo | No native config | Treat as an exported graph artifact, not a Transformers source basis. |

## 3a. Family variation traps

- Current config post-init forces `num_key_value_heads = num_attention_heads`; do not infer GQA/MQA support despite the generic attention helper.
- `hidden_size` must equal `out_features[-1]`; source validation rejects mismatch.
- `attention_bias=false` in representative configs. Bias-enabled projections are source-supported by config but not represented by the official checkpoint.
- The earlier mirror uses `stage_block_dims`, not current `out_features`. DinoML should reject or normalize this only with an explicit compatibility rule.
- Processor emits three grayscale-identical channels because of Transformers image pipeline constraints; model immediately extracts one channel. A direct DinoML ABI could accept `[B,2,1,H,W]` only if guarded as a different frontend contract.
- NCHW is the semantic source layout through Conv/BatchNorm/Unfold. NHWC only appears locally around LayerNorm and token attention after explicit permutes.
- Dynamic/ragged output count is core behavior. `matches`, `matching_scores`, and `keypoints` use the maximum match count per batch after coarse filtering.
- Coarse/fine matching contains value-dependent indexing, `max`, `where`, boolean masks, equality tests on scores, and advanced indexing. This is the main graph-lowering risk.
- `coarse_matching_skip_softmax` changes coarse score math from dual-softmax confidence to raw temperature-scaled similarity.
- `fine_kernel_size` must produce square window sizes used by `sqrt`; default `8`, debug tests use `2`.
- Input H/W should satisfy the stride/window assumptions. Tests explicitly use sizes divisible by early stage strides; first integration should require processor default or guarded multiples.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `permute`, `transpose`, `contiguous`, `stack`, `cat`, `split`, `repeat_interleave`, `expand`, `squeeze`, `unsqueeze`.
- Advanced indexing/gather: batch-indexed selection by coarse match indices, `torch.gather` into coordinate grids, boolean mask filtering in postprocess.
- `arange`, `linspace`, `meshgrid(indexing="ij")`, `cumsum`, integer `//`, `%`, casts to int/float.

Neural primitives:

- Conv2d NCHW, kernels `3x3`, `1x1`, and depthwise aggregation `4x4` with `groups=hidden_size`.
- BatchNorm2d inference folding opportunity.
- LayerNorm over channel-last feature dimension.
- ReLU and LeakyReLU.
- MaxPool2d `4x4,s=4`.
- Bilinear interpolate with `align_corners=False`.
- `unfold` / im2col for local fine windows.
- Linear projections: Q/K/V/O each `256 -> 256` without bias in official config; MLP `512 -> 512 -> 256` without bias.

Attention primitives:

- Noncausal self-attention and cross-attention on aggregated coarse token grids.
- MHA with `8` heads, `head_dim=32`, no masks, no cache.
- SDPA/FlashAttention-compatible in principle for token attention, but only after preserving local aggregation and 2D RoPE.

Position/custom math:

- 2D RoPE over aggregated coarse grid, using separate row/column construction and interleaved dimensions.
- `rotate_half` that pairs even/odd dimensions.

Matching/postprocess ABI:

- Coarse all-pairs similarity `[B, Hc*Wc, Hc*Wc]`, temperature divide, optional dual softmax.
- Threshold, border mask over `[B,H0,W0,H1,W1]`, mutual nearest-neighbor check.
- First fine stage: local correlation `[B,M,64,100] -> crop to [B,M,64,64]`, argmax over flattened window-pair dimension.
- Second fine stage: local correlation using the last 8 dims, gather 3x3 neighborhood, softmax, spatial expectation.
- Output normalization by width/height; postprocess denormalizes to original target sizes and filters variable-length records.

## 5. Layer/block breakdown

Backbone:

```text
pixel_values [B,2,3,H,W]
  -> reshape [2B,3,H,W]
  -> select grayscale channel [2B,1,H,W]
  -> stage0: 1 block, 1 -> 64, stride 2
  -> stage1: 2 blocks, 64 -> 64, stride 1
  -> stage2: 4 blocks, 64 -> 128, first stride 2
  -> stage3: 14 blocks, 128 -> 256, first stride 2
```

Each RepVGG block:

```text
y = Conv3x3+BN(x) + Conv1x1+BN(x) + optional BN identity(x)
y = ReLU(y)
```

Backbone returns stages 1..3. At default `480x640`: residual maps are approximately `240x320` at 64 channels and `120x160` at 128 channels; coarse map is `60x80` at 256 channels.

Local feature transformer, repeated 4 times:

```text
x [B,2,256,Hc,Wc]
self attention over both images reshaped as [2B,256,Hc,Wc]:
  q = depthwise Conv2d(k=4,s=4)
  kv = MaxPool2d(k=4,s=4)
  q,kv -> NHWC -> LayerNorm -> tokens [2B,300,256] at default size
  q,k,v = Linear(256 -> 256)
  q,k = 2D RoPE(q,k)
  attn = noncausal MHA(q,k,v)
  out = Linear(256 -> 256)
  out -> NCHW -> bilinear upsample scale 4
  mlp_input = concat(original, out) on channels -> NHWC [2B,Hc,Wc,512]
  mlp = Linear(512 -> 512) -> LeakyReLU -> Linear(512 -> 256) -> LayerNorm
  x = x + mlp_out
cross attention:
  image0 attends to image1, image1 attends to updated image0
```

Fine fusion:

```text
coarse [B,2,256,Hc,Wc] / sqrt(256)
-> reshape [2B,256,Hc,Wc]
-> Conv1x1 256 -> 256, interpolate x2
-> residual stage2 fusion: Conv1x1 128->256, add, Conv3x3+BN+LeakyReLU, Conv3x3 256->128, interpolate x2
-> residual stage1 fusion: Conv1x1 64->128, add, Conv3x3+BN+LeakyReLU, Conv3x3 128->64, interpolate x2
-> fine map [2B,64,Hf,Wf]
-> image0 unfold k=8,stride=(Hf/Hc),padding=0 -> [B,Hc*Wc,64,64]
-> image1 unfold k=10,stride=(Hf/Hc),padding=1 -> [B,Hc*Wc,100,64]
```

Matching:

```text
coarse features -> flatten [B,2,Hc*Wc,256] / sqrt(256)
similarity = image0 @ image1.T / temperature
confidence = softmax(sim, dim=1) * softmax(sim, dim=2) unless skip flag
matches = threshold + border removal + mutual nearest neighbors
fine windows are gathered at matched coarse indices
fine stage 1 uses first 56 dims by default, dual softmax, argmax
fine stage 2 uses last 8 dims, 3x3 softmax, spatial expectation
```

## 6. Attention requirements

Required attention is encoder-style, noncausal, no KV cache.

| Property | EfficientLoFTR |
| --- | --- |
| Attention sites | aggregated self-attention and cross-attention inside coarse feature transformer |
| Causal | No |
| Mask | `None` in source |
| Head form | MHA in current config/source |
| Heads / KV heads / dim | `8 / 8 / 32` |
| Query length | aggregated grid, default `15*20=300` |
| Key/value length | same as query length for default source; cross-attention code assumes same `seq_len` shape after aggregation |
| RoPE | applied to Q/K for self-attention path via passed position embeddings; cross-attention calls do not pass position embeddings in current source |
| Cache | None |
| Backend | `ALL_ATTENTION_FUNCTIONS` interface, eager fallback; `_supports_flash_attn` and `_supports_sdpa` are declared |

Important ABI detail: source computes `key_states = self.k_proj(current_states).view(batch_size, seq_len, -1, dim)`, using query `seq_len` and full hidden `dim` before reshaping to heads. This works for the configured equal-size aggregated self/cross grids. DinoML should guard equal Q/K token counts for this family unless the source is fixed or generalized.

## 7. Position encoding and custom math

2D RoPE is generated from image feature dimensions after aggregation:

```python
embed_h = (coarse_h - q_kernel) // q_stride + 1
embed_w = (coarse_w - q_kernel) // q_stride + 1
emb[..., 0::2] = row_index * inv_freq
emb[..., 1::2] = col_index * inv_freq
cos = repeat_interleave(cos(emb), 2, dim=-1)
sin = repeat_interleave(sin(emb), 2, dim=-1)
q = q * cos + rotate_even_odd(q) * sin
k = k * cos + rotate_even_odd(k) * sin
```

`inv_freq` uses `rope_theta=10000` and `partial_rotary_factor=4.0` in current configs. Since `head_dim=32`, the default rotary dimension becomes `128`, then the embedding is formed for `hidden_size=256`; this is source behavior and should be reproduced rather than simplified to LLM-style per-head RoPE without checking dimensions.

Precompute opportunity: for fixed `H,W`, `q_kernel`, `q_stride`, `hidden_size`, and dtype, cos/sin tables can be cached per coarse grid. Dynamic image sizes need regenerated tables.

## 8. Preprocessing and input packing

Processor contract:

- Input must be one pair of images or a list of image pairs.
- Default resize: height `480`, width `640`, bilinear.
- Default rescale: multiply by `1/255`.
- Default grayscale: true.
- Output `pixel_values` shape is `[B,2,3,H,W]` when returned as tensors. The three channels are grayscale-identical by design.
- The model requires `pixel_values.ndim == 5` and pair dimension `2`.

Postprocess contract:

- Inputs: model output and `target_sizes` with shape/list equivalent to `[B,2,2]`, each `(height,width)` for the original images.
- Model keypoints are normalized `[0,1]` by processed width/height.
- Postprocess multiplies by flipped original sizes `(width,height)`, casts to `int32`, filters `scores > threshold` and `matches > -1`, and returns per-pair variable-length dicts: `keypoints0`, `keypoints1`, `matching_scores`.

DinoML boundary recommendation:

- Stage 1 should let CPU/data pipeline own resize/rescale/grayscale and pass exact source ABI `[B,2,3,H,W]`.
- A later optimized ABI may accept `[B,2,1,H,W]` only with explicit adapter/parity tests.
- End-to-end parity needs target/original sizes for postprocess; these are not model graph inputs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv2d+BatchNorm inference fold

Source pattern: RepVGG and fine-fusion Conv2d followed by BatchNorm2d.

Replacement: folded Conv2d weight/bias.

Preconditions:

- Inference mode with frozen BN running stats.
- No training labels; source rejects labels for model forward.
- Preserve parallel RepVGG branches before any branch fusion.

Failure cases: training mode, unfrozen BN, or checkpoint conversion expecting separate keys.

Parity test sketch: compare each stage output before/after fold on default `480x640` and debug sizes with fp32 tolerance `1e-5`.

### Rewrite: RepVGG branch fuse

Source pattern:

```text
Conv3x3+BN + Conv1x1+BN + optional identity BN -> ReLU
```

Replacement: single Conv3x3 with folded branches plus ReLU.

Preconditions:

- Inference-only.
- Same stride and output channels.
- Convert `1x1` kernel into centered `3x3`.
- Identity branch exists only when channels equal and stride is 1.

Failure cases: unfused state dict roundtrip, nonstandard padding/stride, training.

### Rewrite: Aggregation attention layout island

Source pattern: NCHW -> depthwise Conv/MaxPool -> NHWC LayerNorm/tokens -> attention -> NCHW.

Replacement: local channel-last implementation for LayerNorm/Linear/attention while preserving NCHW external conv ABI.

Preconditions:

- Region is exactly bounded by the explicit source permutes.
- Axis rewrite: LayerNorm normalizes last channel axis, concat outside MLP is channel axis in NCHW then last axis in NHWC.
- Interpolate consumes NCHW.

Failure cases: global NHWC conversion without rewriting `dim=1` concat, Conv2d groups axis, MaxPool, Unfold, or border masks.

### Rewrite: Q aggregation Conv2d -> windowed depthwise linear

Source pattern: depthwise Conv2d `hidden_size` groups, kernel/stride usually `4`.

Replacement: fixed window gather + per-channel dot, or specialized depthwise kernel.

Preconditions:

- `groups == hidden_size`, `bias == false`, `padding == 0`.
- Source NCHW flatten order preserved.
- H/W large enough for `(H-k)//s+1`.

Failure cases: changed groups, padding, or global NHWC translation without weight/layout transform.

### Rewrite: Fine unfold -> matched-window gather

Source pattern unfolds every coarse window, then gathers only coarse matched indices.

Replacement: gather only matched local windows directly from fine maps.

Preconditions:

- Coarse matches are known before fine matching.
- Window sizes are fixed (`8` and `10`) and stride is `fine_height // coarse_height`.
- Exact PyTorch `unfold` flatten order is preserved.

Failure cases: need all windows for debugging/hidden outputs, dynamic windows, non-integer stride, or mismatched padding semantics.

### Rewrite: Coarse matching as postprocess/kernel boundary

Source comments identify coarse matching as parameter-free and postprocessable. DinoML can either lower it as graph ops or expose a custom output ABI from coarse features to a matcher kernel.

Preconditions:

- Need exact dual-softmax/threshold/border/mutual-nearest semantics.
- Need deterministic handling of ties from `max` and equality.
- Batch ragged output policy must match padded `matches`/`scores` shape.

Failure cases: replacing with approximate top-k, changing tie behavior, or dropping invalid `-1` entries.

## 10. Kernel fusion candidates

Highest priority:

- RepVGG inference branch fusion and Conv+BN fold: largest CNN cost and simplest deterministic win.
- Coarse matching custom kernel: all-pairs `[B,4800,4800]` at default processor size is expensive and value-dependent; dual softmax plus mutual-nearest filtering is a natural fused matcher ABI.
- Fine window gather/correlation/refinement: avoid materializing all unfolded windows when only matched windows are consumed.

Medium priority:

- Aggregated attention block fusion: depthwise aggregation, LayerNorm, QKV projections, RoPE, SDPA, O projection, bilinear upsample, and MLP have repeated layout churn.
- 2D RoPE table cache per image size.
- Fine 3x3 softargmax/expectation as a small fused postprocess kernel.

Lower priority:

- Generic NHWC/channel-last CNN conversion. Valuable only if the whole conv island is guarded; unsafe as default semantic translation.
- FlashAttention tuning for 300-token aggregated attention. Likely less dominant than coarse all-pairs matching and fine-window work at default sizes.

## 11. Runtime staging plan

Stage 1: Parse config and processor metadata. Admit only native `efficientloftr` configs with `hidden_size == out_features[-1]`, `num_key_value_heads == num_attention_heads`, default or explicitly supported image sizes, and `EfficientLoFTRForKeypointMatching`.

Stage 2: Load weights and run `EfficientLoFTRModel` backbone/coarse feature parity. Stub keypoint matching by returning feature maps or compare only `feature_maps`.

Stage 3: Add aggregated attention parity for one coarse transformer layer, including 2D RoPE and local layout islands.

Stage 4: Add full coarse transformer parity and coarse matching as a separate custom matcher/postprocess ABI.

Stage 5: Add fine fusion and matched-window extraction. Initially allow eager/unoptimized `unfold`; then replace with direct matched-window gather.

Stage 6: Add fine matching/refinement and full `matches`, `matching_scores`, `keypoints` ABI.

Stage 7: Add external postprocess to original image sizes and threshold filtering.

Stage 8: Optimize RepVGG fold, matcher kernels, fine correlation kernels, and layout islands.

## 12. Parity and validation plan

- Processor parity: PIL/torchvision paths for a pair and a batch of pairs; verify output shape `[B,2,3,480,640]`, grayscale channels, rescale, and resize.
- Config admission tests: official config, historical mirror config, malformed `hidden_size/out_features`, non-MHA KV heads, non-square fine windows.
- Backbone unit parity: random `[B,2,3,H,W]` input, compare stage feature maps.
- Aggregated attention layer parity: one layer with fixed coarse map, eager attention backend, compare fp32 outputs.
- 2D RoPE parity: fixed `60x80` coarse map and debug coarse map; compare cos/sin and rotated Q/K.
- Coarse matcher parity: hand-built score tensors covering threshold, border removal, ties, invalid `-1`, and `skip_softmax`.
- Fine matching parity: synthetic windows with known argmax and 3x3 expectation offsets.
- End-to-end parity: official checkpoint on known image pair, compare match count envelope, keypoints, scores. Suggested fp32 tolerance: `1e-4` for neural feature intermediates, stricter exact/int checks for indices, and pixel-level tolerance after int postprocess.
- Ragged ABI tests: batch with different match counts must preserve padded maximum shape and per-pair filtering.

## 13. Performance probes

- Preprocessing throughput: resize/rescale/grayscale pairs per second.
- Backbone-only latency/throughput for `480x640`, plus resolution sweep.
- Coarse transformer latency split: aggregation, attention, upsample+MLP.
- Coarse matching sweep over coarse grid sizes, threshold, and `skip_softmax`.
- Fine fusion and unfold materialization memory.
- Matched-window direct gather versus full `unfold`.
- Fine correlation/refinement cost versus number of coarse matches.
- End-to-end batch-size sweep, with separate reports for fixed match count and natural ragged outputs.
- Layout probe: source NCHW baseline versus guarded channel-last conv island.

## 14. Skip/defer list

- Training and labels; source rejects labels and no loss is produced.
- Tokenizer/generation/KV cache; not applicable.
- Raw `.ckpt`, `.pth`, Kornia, and ONNX loading unless separately audited.
- Non-default attention variants such as GQA/MQA.
- Global NHWC conversion.
- Approximate top-k matching that changes mutual-nearest/tie semantics.
- Visualization helper; keep outside runtime.

## 15. Final implementation checklist

- [ ] Parse `EfficientLoFTRConfig` and processor config.
- [ ] Add config admission guards for native source-supported combinations.
- [ ] Load official weights and preserve Conv/BN/Linear naming.
- [ ] Implement processor-compatible input ABI `[B,2,3,H,W]`.
- [ ] Implement NCHW RepVGG backbone.
- [ ] Add Conv+BN fold and optional RepVGG branch-fuse pass.
- [ ] Implement 2D RoPE table generation and rotation.
- [ ] Implement aggregated self/cross attention layout island.
- [ ] Implement coarse all-pairs matching with threshold, border removal, mutual nearest, invalid `-1` policy.
- [ ] Implement fine fusion pyramid and bilinear interpolation.
- [ ] Implement fine window extraction, first-stage argmax, second-stage 3x3 softargmax expectation.
- [ ] Implement normalized keypoint output ABI and target-size postprocess.
- [ ] Add parity tests for each stage and end-to-end keypoint matching.
- [ ] Benchmark processor, backbone, coarse matching, fine refinement, and end-to-end throughput.
