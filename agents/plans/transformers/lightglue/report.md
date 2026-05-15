# LightGlue Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ETH-CVG/lightglue_superpoint
Config source: https://huggingface.co/ETH-CVG/lightglue_superpoint/raw/main/config.json
Source files inspected:
  transformers/src/transformers/models/lightglue/configuration_lightglue.py
  transformers/src/transformers/models/lightglue/modeling_lightglue.py
  transformers/src/transformers/models/lightglue/modular_lightglue.py
  transformers/src/transformers/models/lightglue/image_processing_lightglue.py
  transformers/src/transformers/models/lightglue/image_processing_pil_lightglue.py
  transformers/src/transformers/models/superpoint/configuration_superpoint.py
  transformers/src/transformers/models/superpoint/modeling_superpoint.py
Any missing files or assumptions:
  modeling_lightglue.py/configuration_lightglue.py/image_processing_* are generated from modular_lightglue.py.
  Future Transformers edits should start from modular_lightglue.py, but DinoML import parity should follow the generated files.
  No model execution, imports, or DinoML tests were run.
```

Hub configs inspected:

| Model id | Scope | Important note |
|---|---:|---|
| [ETH-CVG/lightglue_superpoint](https://huggingface.co/ETH-CVG/lightglue_superpoint) | primary native target | `LightGlueForKeypointMatching` with in-library `SuperPoint` detector. |
| [ETH-CVG/lightglue_disk](https://huggingface.co/ETH-CVG/lightglue_disk) | separate/gated path | `trust_remote_code=true`; detector `model_type=disk` with `auto_map`, descriptor width 128. Current in-library audit should reject or route this until DISK remote code is separately audited. |
| [stevenbucaille/lightglue](https://huggingface.co/stevenbucaille/lightglue) | historical mirror | Older config fields such as `num_layers`, `num_heads`, `attention_probs_dropout_prob`, `rotary_value`; current source does not read several of these names. |
| [stevenbucaille/lightglue_superpoint](https://huggingface.co/stevenbucaille/lightglue_superpoint) | historical mirror | Native SuperPoint detector by default, but some historical fields should be ignored for this source basis. |
| [stevenbucaille/lightglue_minima](https://huggingface.co/stevenbucaille/lightglue_minima) | mirror/variant label | Config is structurally the same native SuperPoint stack as primary; name does not imply a different in-library model body. |

## 2. High-level architecture

LightGlue is an image-pair keypoint matching stack, not an autoregressive model. The first useful DinoML target is end-to-end keypoint matching for a pair of images using the native SuperPoint detector plus the LightGlue matcher.

```text
image pair preprocessing
  -> pixel_values [B, 2, C, H, W], source layout NCHW per image
  -> SuperPoint detector on [B*2, C, H, W]
  -> padded keypoints/descriptors/mask per image
  -> LightGlue descriptor/keypoint normalization and descriptor transformer
  -> match assignment / mutual nearest matching
  -> optional post_process_keypoint_matching to original image coordinates
```

Stage decomposition:

| Stage | Runtime contract | Cacheability / independence |
|---|---|---|
| Image preprocessing | Resize to 480x640 by default, rescale by 1/255, optional grayscale replicated to 3 channels, pair packing. | CPU/data-pipeline first; GPU resize/grayscale can be deferred. |
| SuperPoint detector | In-library CNN detector consumes `[B*2, 3, H, W]`, extracts first channel internally, emits padded keypoints, scores, descriptors, mask. | Can be audited and validated independently as `superpoint`; LightGlue consumes its output ABI. |
| LightGlue matcher | Operates on descriptor sequences and 2D keypoints for paired images; alternates self-attention and cross-attention. | Can support a feature-only entrypoint later: caller supplies keypoints/descriptors/mask and bypasses detector. |
| Match extraction | Builds assignment scores, mutual-nearest matches, score filtering, pruning restoration. | Post-score ABI is essential for parity, but can initially run as a bounded helper. |
| Postprocess | Converts relative keypoints to original image sizes and filters valid matches. | CPU postprocess is acceptable first. |

## 3. Important config dimensions

Primary `ETH-CVG/lightglue_superpoint` values from `config.json` unless noted:

| Field | Value | Source / operator significance |
|---|---:|---|
| `descriptor_dim` / `hidden_size` | 256 | LightGlue descriptor width. |
| `num_hidden_layers` | 9 | Repeated matcher layers; each has self-attn, self MLP, cross-attn, cross MLP, and one match-assignment layer. |
| `num_attention_heads` | 4 | Head dim = 64 by default. |
| `num_key_value_heads` | 4 | MHA for primary config. Source permits GQA/MQA if divisible. |
| `intermediate_size` | 512 | Effective source default is `descriptor_dim * 2`; MLP consumes concat width 512 and returns 256. |
| `attention_bias` | true | Q/K/V/O projections are biased. |
| `attention_dropout` | 0.0 | Inference dropout is 0. |
| `hidden_act` | `gelu` | MLP activation after LayerNorm on 512-wide states. |
| `depth_confidence` | 0.95 | Enables early stop when > 0. |
| `width_confidence` | 0.99 | Enables keypoint pruning when > 0. |
| `filter_threshold` | 0.1 | Match extraction threshold after exponentiated log scores. |
| SuperPoint `descriptor_decoder_dim` | 256 | No input projection needed for primary native config. |
| SuperPoint `max_keypoints` | -1 | Dynamic, data-dependent keypoint count; padded to max per `B*2`. |
| Processor size | 480x640 | Preprocessor default, not a model architectural limit. |
| Processor grayscale | true | SuperPoint path expects grayscale replicated into 3 channels. |

Representative config sweep:

| Model id | Detector | LightGlue dims | Detector descriptor dim | Processor grayscale | Native admission |
|---|---|---:|---:|---:|---|
| `ETH-CVG/lightglue_superpoint` | in-library `superpoint` | D=256, L=9, H=4, KVH=4 | 256 | true | Yes. |
| `ETH-CVG/lightglue_disk` | remote `disk` via `auto_map` | D=256, L=9, H=4, KVH=4 | 128 | false | No: route/reject pending DISK audit and remote-code policy. Requires `input_projection` 128 -> 256 if admitted. |
| `stevenbucaille/lightglue` | default/current source coerces dict to `superpoint` | D=256, historical `num_layers=9`, `num_heads=4` ignored by current source | SuperPoint default 256 | omitted, source default true | Use only after confirming weight keys/config compatibility. |
| `stevenbucaille/lightglue_superpoint` | default/current source coerces dict to `superpoint` | D=256, L=9, H=4, KVH=4 | SuperPoint default 256 | true | Likely native but historical ignored fields present. |
| `stevenbucaille/lightglue_minima` | in-library `superpoint` | D=256, L=9, H=4, KVH=4 | 256 | true | Same source shape as primary; variant meaning is not source-visible. |

## 3a. Family variation traps

- `keypoint_detector_config` is a sub-config. DinoML should admit only audited detector types. Primary SuperPoint is in-library; DISK is remote-code in the inspected Hub config.
- Current `LightGlueConfig` reads `num_hidden_layers` and `num_attention_heads`; older `num_layers` and `num_heads` fields are historical for this source basis and should not silently drive topology.
- `descriptor_dim % num_attention_heads == 0` is enforced by config validation.
- Source permits `num_key_value_heads < num_attention_heads` through `repeat_kv`; primary config is MHA. If DinoML admits GQA/MQA, require `num_attention_heads % num_key_value_heads == 0`.
- Attention class sets `is_causal=True`, but LightGlue passes detector masks, not a causal generation mask. Treat this as encoder-style attention over keypoints; do not add KV cache or autoregressive semantics.
- Keypoint count is dynamic and data-dependent from SuperPoint threshold/NMS/top-k. The transformer sequence length is not fixed by config.
- Early stop and pruning mutate active batch and sequence lengths between layers using boolean masks, list comprehensions, `pad_sequence`, and final index restoration. This is the main graph-capture trap.
- Source uses NCHW image tensors and SuperPoint CNNs. NHWC/channel-last is only a guarded optimization; semantic translation must preserve NCHW axes around conv, pooling, softmax dim=1, and `grid_sample`.
- Match extraction and postprocess use index/gather/scatter-like updates and variable-length output records. Do not reduce this to a plain dense classification head.

## 4. Operator coverage checklist

Tensor/layout ops:

- Pair packing/unpacking: `[B, 2, C, H, W] -> [B*2, C, H, W]`, then detector outputs back to `[B, 2, Kmax, ...]`.
- `reshape`, `view`, `transpose`, `permute`, `contiguous`, `flip(dim=1)`, `repeat_interleave`, `expand`, `stack`, `cat`, `gather`, `where`, `masked_fill`.
- Dynamic padding and ragged recovery: `pad_sequence`, boolean masked row selection, `torch.nonzero`, `topk` for detector when `max_keypoints >= 0`.
- `arange`, scalar tensor creation on device, dtype casts, shape-derived constants.

Neural network primitives:

- SuperPoint detector: Conv2d 3x3/1x1, ReLU, MaxPool2d, Softmax over detector class dim, MaxPool2d-based NMS, L2 normalize, `grid_sample(..., mode="bilinear", align_corners=True)`.
- LightGlue input projection: optional Linear(detector descriptor dim -> `descriptor_dim`), identity for SuperPoint primary.
- Positional encoder: Linear(2 -> `head_dim/2`, no bias), repeat-interleave to `head_dim`, cos, sin.
- Attention projections per self/cross block: Q Linear(256 -> 256), K Linear(256 -> 256), V Linear(256 -> 256), O Linear(256 -> 256), biased for primary.
- MLP after attention: concat `[x, attn]` to 512, Linear(512 -> 512), LayerNorm(512), GELU, Linear(512 -> 256).
- Match assignment: Linear(256 -> 256), scale by `descriptor_dim ** -0.25`, batched similarity GEMM `[B, K0, 256] @ [B, 256, K1]`, Linear(256 -> 1) matchability.
- Token confidence: Linear(256 -> 1), sigmoid, detached input.

Attention primitives:

- Dense self-attention over per-image keypoints with additive keypoint mask.
- Dense cross-attention between paired images via reshape/flip, same keypoint count after padding/pruning.
- MHA primary; repeat-KV GQA/MQA path is source-visible but not present in primary config.
- Eager attention math: `QK^T * head_dim**-0.5`, add mask, softmax in fp32, cast back, `P @ V`.
- Source advertises SDPA/FlashAttention support via Transformers attention dispatch; DinoML can initially implement the eager math and later map eligible masks/layouts to fused attention.

Position/custom math:

- Coordinate normalization by image size, then learned 2D projection to rotary cos/sin.
- LightGlue `rotate_half` pairs even/odd channels differently from LLaMA-style half split.

Preprocessing/postprocessing:

- Processor validates exact image pair structure and produces `pixel_values` as stacked pairs.
- Default resize to 480x640, rescale by 1/255, grayscale to 3 replicated channels for SuperPoint.
- `post_process_keypoint_matching` consumes `target_sizes [B,2,2]` as `(height,width)` per image, rescales keypoints, filters `(score > threshold) & valid match index`.

Quantized/packed weights:

- No source-coupled quantized or packed weight format in inspected native source.

Generation/cache:

- Not applicable. There is no decode loop, token cache, sampling, or logits head.

## 5. Layer/block breakdown

Detector branch for native SuperPoint:

```text
pixel_values [B*2, 3, H, W]
  -> take first channel [B*2, 1, H, W]
  -> 4 conv blocks, first 3 with max-pool stride 2, final no pool
  -> keypoint decoder:
       Conv2d(C=128 -> 256, 3x3, pad=1) -> ReLU
       Conv2d(256 -> 65, 1x1) -> softmax(dim=1) -> drop dustbin
       reshape 64 cell logits to full-resolution score map
       maxpool NMS -> threshold -> border removal -> optional top-k
  -> descriptor decoder:
       Conv2d(128 -> 256, 3x3, pad=1) -> ReLU
       Conv2d(256 -> descriptor_dim, 1x1) -> L2 normalize
       grid_sample at keypoints, scale=8, align_corners=True
       L2 normalize -> descriptors [K, descriptor_dim]
  -> pad all images to Kmax: keypoints [B*2,Kmax,2], scores [B*2,Kmax], descriptors [B*2,Kmax,Ddet], mask [B*2,Kmax]
```

LightGlue matcher:

```text
Input from detector:
  keypoints [B,2,Kmax,2] relative in [0,1]
  descriptors [B,2,Kmax,Ddet]
  mask [B,2,Kmax]

absolute_keypoints = keypoints * [width,height]
normalized_keypoints = (absolute_keypoints - [width,height]/2) / (max(width,height)/2)
descriptors = optional Linear(Ddet -> 256)
pos = Linear(2 -> head_dim/2, bias=False)
cos,sin = cos/sin(repeat_interleave(pos, 2, -1))

Repeated L=9 layers:
  self_out = MHA(descriptors, descriptors, descriptors, RoPE(q,k), mask)
  x = descriptors + MLP(cat(descriptors, self_out), 512 -> 256)
  paired = reshape x to [B,2,K,D], flip image dimension, reshape to [B*2,K,D]
  cross_out = MHA(x, paired, paired, paired mask)
  descriptors = x + MLP(cat(x, cross_out), 512 -> 256)
  optional confidence head -> early-stop image pairs
  optional matchability head -> prune keypoints, pad survivors

Match assignment at stopped/final layer:
  m = Linear(256 -> 256)(descriptors) / 256**0.25
  similarity = m0 @ m1.T
  masked positions = dtype min
  matchability = Linear(256 -> 1)
  scores = log_softmax(similarity,row) + log_softmax(similarity,col) + logsigmoid(matchability0)+logsigmoid(matchability1)
  add dustbin row/column for unmatched
  matches = mutual-nearest + threshold filtering
```

## 6. Attention requirements

LightGlue uses non-generative dense attention over keypoint sets.

| Field | Requirement |
|---|---|
| Causal? | Semantically noncausal for matching. `is_causal=True` is set on the module, but the forward path supplies masks over valid keypoints rather than generation masks. DinoML should not infer autoregressive behavior. |
| Self/cross | Both. Each layer runs self-attention per image, then cross-attention to the paired image. |
| MHA/MQA/GQA | Primary MHA: 4 query heads and 4 KV heads, head dim 64. Source has repeat-KV for `num_key_value_heads < num_attention_heads`. |
| Q/K/V widths | Primary Q/K/V all 256 total. If GQA admitted, K/V total width = `num_key_value_heads * head_dim`. |
| Query/key lengths | Self: K by K per image. Cross: K0 by K1 logically; source pads/prunes both images together to a common active K per batch step. |
| Masking | Detector mask becomes extended additive attention mask; cross mask is reshaped/flipped to the paired image. |
| Packed/varlen | No `cu_seqlens` ABI. Raggedness is represented by padding masks and source-level dynamic pruning. |
| Sliding/local | None. |
| Position | Learned 2D coordinate projection produces RoPE cos/sin and is applied to Q/K. Cross-attention does not pass `position_embeddings`, so only self-attention receives RoPE in the inspected source. |
| Cache | No KV cache. Independently cacheable artifacts are detector keypoints/descriptors and possibly LightGlue intermediate descriptors only for custom workflows, not source ABI. |
| Flash/SDPA | Source advertises both; fused backend must preserve additive mask, fp32 softmax behavior, and no causal mask unless verified through the selected Transformers backend path. |

## 7. Position encoding and custom math

Keypoint normalization:

```python
size = [width, height]
shift = size / 2
scale = max(width, height) / 2
normalized = (keypoints_xy - shift) / scale
```

LightGlue positional encoder and RoPE:

```python
proj = Linear(2, head_dim // 2, bias=False)(normalized_keypoints)
emb = repeat_interleave(proj, repeats=2, dim=-1)
cos, sin = cos(emb), sin(emb)

def rotate_half_lightglue(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return stack([-x2, x1], dim=-1).flatten(-2)

q = q.float() * cos[:, None] + rotate_half_lightglue(q.float()) * sin[:, None]
k = k.float() * cos[:, None] + rotate_half_lightglue(k.float()) * sin[:, None]
q, k = q.to(original_dtype), k.to(original_dtype)
```

The coordinate projection weights are learned. `cos/sin` depend on dynamic keypoints and cannot be a fixed sequence table.

## 8. Preprocessing and input packing

Processor ABI:

- Input must be a single pair of images or a list of image pairs. The processor flattens pairs, preprocesses images, then stacks each pair.
- Default output is `pixel_values` shaped `[B, 2, C, H, W]`. With defaults, `C=3`, `H=480`, `W=640`.
- Default torch processor groups images by shape before resize/rescale/grayscale, then restores order.
- `do_grayscale=true` turns RGB into 3 identical grayscale channels. SuperPoint then extracts channel 0 internally.
- `ETH-CVG/lightglue_disk` sets `do_grayscale=false`; this belongs with the DISK detector path and should not be admitted as native SuperPoint parity.

Model-coupled packing:

- Forward rejects inputs that are not rank 5 or where dimension 1 is not exactly 2.
- The detector receives `[B*2, C, H, W]`.
- SuperPoint outputs relative keypoints; LightGlue converts to absolute coordinates before normalization, then returns original relative `keypoints` in the output object.

Postprocess ABI:

- `target_sizes` must have one `(height,width)` pair for each image in each batch pair: list or tensor shape `[B,2,2]`.
- Keypoints are scaled by `target_sizes.flip(-1)` to convert `(x,y)` from relative to original image coordinate scale and cast to int32.
- Returned list entries contain variable-length `keypoints0`, `keypoints1`, and `matching_scores`. Only matches from image 0 are emitted after filtering against scores, `-1`, and out-of-bounds indices.

## 9. Graph rewrite / lowering opportunities

### Rewrite: feature-only LightGlue matcher entrypoint

Source pattern:

```text
pixel_values -> keypoint_detector -> LightGlue matcher
```

Replacement:

```text
caller-supplied keypoints/descriptors/mask -> LightGlue matcher -> matches/scores/prune
```

Preconditions:

- Detector output ABI must match source: keypoints `[B,2,K,2]` relative `(x,y)`, descriptors `[B,2,K,Ddet]`, mask `[B,2,K]`.
- Caller must provide original model image `height,width` used to make absolute coordinates.
- `Ddet == descriptor_dim` or include the source `input_projection`.

Failure cases:

- Remote detector configs, descriptor normalization differences, detector-specific keypoint ordering, or subpixel coordinate conventions.

Parity test sketch:

- Run Transformers SuperPoint once to capture keypoints/descriptors/mask, then compare DinoML matcher-only outputs layer-by-layer and final matches.

### Rewrite: static no-pruning/no-early-stop mode

Source pattern:

```text
if depth_confidence > 0: early stop
if width_confidence > 0: prune points
```

Replacement:

```text
set depth_confidence=0 and width_confidence=0 for static L-layer dense matcher
```

Preconditions:

- Config or runtime admission explicitly disables both knobs.
- User accepts parity only against the disabled-knob Transformers run, not default checkpoint behavior.

Shape equations:

- Active sequence length remains `Kmax` for all layers; no boolean row compaction or `pad_sequence`.

Failure cases:

- Default `ETH-CVG/lightglue_superpoint` uses both knobs, so this is not default parity.

### Rewrite: SuperPoint cell softmax reshape

Source pattern:

```text
Conv2d -> softmax(dim=1) over 65 -> drop dustbin -> permute/reshape 8x8 cells to full score map
```

Replacement:

```text
Conv2d -> softmax64plusdustbin -> pixel_shuffle-like layout transform
```

Preconditions:

- `keypoint_decoder_dim == 65`.
- Cell size is fixed at 8x8 by source reshape.
- NCHW axes preserved or fully rewritten with NHWC guards.

Failure cases:

- Alternate detector configs or remote-code detector bodies.

### Rewrite: match assignment as fused projection/GEMM/log-score

Source pattern:

```text
Linear descriptors -> scale -> pair split -> bmm similarity
matchability Linear -> logsigmoid
two log_softmax passes + dustbin row/col
```

Replacement:

```text
projection GEMM + similarity GEMM + fused double-log-softmax/matchability kernel
```

Preconditions:

- Equal padded `K` for both images at that assignment layer or explicit rectangular support.
- Mask semantics use dtype min before log-softmax.
- Output includes dustbin row/column for internal match extraction.

Failure cases:

- Dynamic pruning produces varying active K; CPU helper may be simpler first.

### Layout opportunity: NCHW detector, sequence matcher

Source region:

- SuperPoint CNN is NCHW with channel softmax and `grid_sample`.
- LightGlue matcher is sequence-major `[B*2,K,D]`.

Guard:

- Do not globally translate image tensors to NHWC unless all Conv2d, MaxPool2d, softmax(dim=1), `permute` reshapes, and `grid_sample` coordinate conventions are rewritten together.

## 10. Kernel fusion candidates

Highest priority:

- Dense attention over keypoints: Q/K/V GEMM + LightGlue RoPE + attention + O projection. K is often hundreds to thousands; attention dominates after detector.
- Match assignment kernel: descriptor projection + similarity GEMM + double log-softmax + matchability. This is core output ABI and likely memory-heavy.
- SuperPoint detector postprocess helpers: maxpool NMS, threshold/nonzero/top-k, descriptor `grid_sample`; without these the end-to-end native target cannot match source.

Medium priority:

- MLP block: concat + Linear(512->512) + LayerNorm + GELU + Linear(512->256). Straight GEMM/LN fusion opportunity.
- Pruning compaction: boolean mask row selection plus `pad_sequence`; useful for default performance but can be staged after static dense path.
- Postprocess filtering: mask/gather/valid-match filtering into compact output records.

Lower priority:

- Processor resize/grayscale on GPU. CPU pipeline is acceptable first.
- SDPA/FlashAttention routing. Start from source-eager parity, then admit fused attention for masks/layouts that match.
- GQA/MQA repeat-KV variants. Source-visible but not used by representative native configs.

## 11. Runtime staging plan

1. Parse `LightGlueConfig` and admit only native `superpoint` detector configs for first target.
2. Implement or compose SuperPoint detector parity separately: CNN encoder, score decoder, NMS/threshold/border/top-k, descriptor sampling.
3. Add matcher-only LightGlue parity using captured detector outputs with `depth_confidence=0,width_confidence=0` to avoid dynamic pruning in the first graph.
4. Add default match assignment and `get_matches_from_scores` ABI, including dustbin scores and mutual-nearest filtering.
5. Add default early-stop/pruning behavior as bounded runtime helpers or a graph region with explicit dynamic-shape support.
6. Add end-to-end `pixel_values -> outputs` native SuperPoint+LightGlue parity.
7. Optimize attention, MLP, match assignment, and detector postprocess kernels.
8. Route `lightglue_disk` and other detector variants to separate audits/admission policies.

Stubbable initially:

- Visualization helper.
- GPU image preprocessing.
- `output_hidden_states` and `output_attentions`.
- Dynamic pruning/early stop if the admitted config disables them for the first matcher-only target.

Not stubbable for default checkpoint parity:

- SuperPoint detector ABI.
- Match extraction ABI.
- Mask handling.
- Default early-stop/pruning, unless parity scope explicitly disables those config knobs.

## 12. Parity and validation plan

- Config admission tests:
  - Accept `ETH-CVG/lightglue_superpoint` native SuperPoint config.
  - Reject or route `ETH-CVG/lightglue_disk` because of remote `disk` detector and `trust_remote_code=true`.
  - Reject `descriptor_dim % num_attention_heads != 0`.
- Custom math tests:
  - `normalize_keypoints` for rectangular images, dtype/device preservation.
  - LightGlue `rotate_half` and RoPE against source for fp32/fp16.
  - `sigmoid_log_double_softmax` with masks and dustbin row/column.
  - `get_matches_from_scores` mutual-nearest filtering and threshold behavior.
- SuperPoint tests:
  - Conv encoder shape ladder for `[B,3,480,640]`.
  - Score-map reshape from `[B,64,H/8,W/8]` to `[B,H,W]`.
  - NMS/threshold/border/top-k edge cases.
  - `grid_sample` descriptor extraction with `align_corners=True`.
- Matcher tests:
  - One self-attention block with fixed synthetic keypoints/descriptors.
  - One full LightGlue layer including cross-image flip/reshape.
  - Full 9-layer matcher with pruning disabled.
  - Full default matcher with early-stop/pruning and final index restoration.
- End-to-end tests:
  - One known image pair through processor + model + postprocess.
  - Empty/no-keypoint case if detector threshold is forced high.
  - Batch with different per-image keypoint counts to exercise padding/masks.

Recommended tolerances:

- fp32 custom math and matcher-only: `rtol=1e-4, atol=1e-5` for scores/descriptors before argmax-sensitive extraction.
- fp16/bf16 optimized attention: compare intermediate logits/scores with looser `rtol=5e-3, atol=5e-3`; validate final matches by exact indices plus score tolerance.
- Match indices should be exact for deterministic fp32 parity; near-threshold cases need fixture margins away from `filter_threshold`.

## 13. Performance probes

- Processor throughput: image pair resize/rescale/grayscale, CPU vs optional GPU.
- SuperPoint detector throughput by image size and batch size.
- Detector postprocess cost split: score softmax/reshape, NMS, nonzero/top-k, descriptor `grid_sample`.
- Matcher-only sweep by `Kmax` with pruning disabled: K = 128, 256, 512, 1024, 2048.
- Default matcher sweep by scene difficulty: realized layer count from early stop, realized active K after pruning.
- Attention backend comparison: eager vs SDPA vs DinoML fused for padded masks.
- Match assignment sweep: similarity GEMM and double log-softmax time/memory.
- End-to-end batch-size sweep for `B` image pairs.
- Output compaction/postprocess throughput for variable match counts.
- Memory usage with and without pruning, especially attention matrices `[B*2,H,K,K]`.

## 14. Skip/defer list

- Training and loss: source rejects `labels`.
- Autoregressive generation, KV cache, beam search, sampling.
- `output_hidden_states` and `output_attentions` in optimized runtime.
- Visualization helper.
- Remote-code DISK and other detector variants until separately audited.
- GQA/MQA variants until representative native configs require them.
- GPU-native resize/grayscale preprocessing.
- Full dynamic pruning as graph IR if a bounded helper is used for first default parity.
- Quantization/packed weights; no source-native requirement.

## 15. Final implementation checklist

- [ ] Parse `LightGlueConfig` and nested detector config.
- [ ] Add admission: native `superpoint` first; route/reject `disk` remote-code configs.
- [ ] Load native SuperPoint + LightGlue weights with alias/shape checks.
- [ ] Implement SuperPoint Conv2d/ReLU/MaxPool/softmax/NMS/threshold/top-k/descriptor `grid_sample`.
- [ ] Implement LightGlue keypoint normalization and learned coordinate RoPE.
- [ ] Implement dense keypoint self-attention and cross-attention with additive masks.
- [ ] Implement LightGlue MLP block with concat, LayerNorm, GELU.
- [ ] Implement match assignment double log-softmax with dustbin row/column.
- [ ] Implement mutual-nearest `get_matches_from_scores`.
- [ ] Implement default early-stop and width-pruning ABI or explicitly disable with config guards.
- [ ] Implement postprocess to original image sizes.
- [ ] Add matcher-only parity fixtures from captured Transformers detector outputs.
- [ ] Add end-to-end processor/model/postprocess parity fixture for `ETH-CVG/lightglue_superpoint`.
- [ ] Benchmark detector, matcher attention, pruning, and match assignment separately.
