# DinoML Transformers Audit: SuperGlue

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: magic-leap-community/superglue_indoor, magic-leap-community/superglue_outdoor
Config source: official HF config.json/preprocessor_config.json plus local SuperGlueConfig defaults
Source files inspected:
- transformers/src/transformers/models/superglue/configuration_superglue.py
- transformers/src/transformers/models/superglue/modeling_superglue.py
- transformers/src/transformers/models/superglue/image_processing_superglue.py
- transformers/src/transformers/models/superglue/image_processing_pil_superglue.py
- transformers/src/transformers/models/superglue/convert_superglue_to_hf.py
- transformers/src/transformers/models/superpoint/configuration_superpoint.py
- transformers/src/transformers/models/superpoint/modeling_superpoint.py
Any missing files or assumptions: SuperGlue delegates keypoint detection to AutoModelForKeypointDetection, defaulting to SuperPoint. This report treats the SuperPoint boundary as required for end-to-end SuperGlue parity, but the SuperPoint body should be staged as its own owned detector sub-audit before broad operator admission.
```

Official checkpoint facts are captured in `evidence_configs.json`. Both official SuperGlue checkpoints are public and ungated:

- [magic-leap-community/superglue_indoor](https://huggingface.co/magic-leap-community/superglue_indoor)
- [magic-leap-community/superglue_outdoor](https://huggingface.co/magic-leap-community/superglue_outdoor)
- delegated detector default: [magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint)

## 2. High-level architecture

Runtime target: image-pair keypoint matching, not text generation. There is no autoregressive decode, vocabulary, logits, RoPE, KV cache, or generation controller.

Dataflow:

```text
image pair preprocessing -> SuperPoint keypoint detector/descriptor extractor -> keypoint coordinate+score MLP encoder -> 18-layer attentional GNN over padded keypoint sets -> final projection -> descriptor similarity -> log-space optimal transport -> mutual nearest match filtering -> postprocess to original image coordinates
```

Stage decomposition:

- CPU/data pipeline: validate image pairs; resize to 480x640 by default; rescale by 1/255; convert to grayscale while retaining 3 channels; pack as `pixel_values[B,2,3,H,W]`.
- Detector stage: SuperPoint consumes NCHW images after SuperGlue flattens pair axis to `[B*2,3,H,W]`, extracts one channel, runs conv/pool decoders, NMS, threshold/top-k, descriptor `grid_sample`, and returns padded keypoint rows.
- Matcher stage: pure keypoint/descriptor graph over `[B,2,K,hidden]`, where `K` is max detected keypoints in the batch.
- Postprocess: rescales relative keypoints to original target sizes, filters invalid or low-score matches, and returns variable-length match records.

## 3. Important config dimensions

| Field | Official value | Source/provenance | Operator impact |
|---|---:|---|---|
| `hidden_size` | 256 | SuperGlue config.json | descriptor width and all GNN projection widths |
| `num_attention_heads` | 4 | SuperGlue config.json | head dim = 64; standard MHA |
| `gnn_layers_types` | `["self","cross"] * 9` | SuperGlue config.json | 18 attentional propagation layers alternating self/cross |
| `keypoint_encoder_sizes` | `[32,64,128,256]` | SuperGlue config.json | MLP widths: 3->32->64->128->256->256 |
| `sinkhorn_iterations` | 100 | SuperGlue config.json | 100 loop iterations of logsumexp normalization |
| `matching_threshold` | 0.0 | SuperGlue config.json | post-Sinkhorn score filtering |
| `attention_probs_dropout_prob` | 0.0 | SuperGlue config.json | dropout is a no-op in inference |
| `torch_dtype` | float32 | HF config.json/model metadata | official weights are F32 |
| SuperPoint `encoder_hidden_sizes` | `[64,64,128,128]` | delegated SuperPoint default/config | NCHW conv stack |
| SuperPoint `descriptor_decoder_dim` | 256 | delegated SuperPoint default/config | descriptor width matches SuperGlue hidden size |
| SuperPoint `keypoint_decoder_dim` | 65 | delegated SuperPoint default/config | 8x8 cell classes plus dustbin |
| SuperPoint `keypoint_threshold` | 0.005 | delegated SuperPoint default/config | value-dependent `nonzero` output count |
| SuperPoint `max_keypoints` | -1 | delegated SuperPoint default/config | unbounded per-image keypoint count unless overridden |
| SuperPoint `nms_radius` | 4 | delegated SuperPoint default/config | NMS via max-pool radius 4 |
| Processor size | 480x640 | preprocessor_config.json | fixed default image size; dynamic override possible |

Representative checkpoint sweep:

| Checkpoint | HF revision inspected | Params | Processor | Operator-significant variation |
|---|---|---:|---|---|
| `magic-leap-community/superglue_indoor` | `413dc1d8a2b02aabea6ea6e24c62d9dcf1857705` | 13,343,553 F32 | resize 480x640, grayscale, rescale | baseline indoor weights |
| `magic-leap-community/superglue_outdoor` | `f4041f88aa6789c46558efaafc98316c6b58a382` | 13,343,553 F32 | same as indoor | same graph/config; outdoor weights |
| `magic-leap-community/superpoint` | `734450e9ffe229074f5998494ddc615475cdb20a` | 1,300,865 F32 | separate keypoint processor | delegated detector body used by SuperGlue default |

Only two official SuperGlue variants were found. They do not give the requested 3-5 checkpoint spread; the useful third row is the delegated SuperPoint detector config because it defines required end-to-end operators.

## 3a. Family variation traps

- `keypoint_detector_config` is a nested AutoModel contract. Current official configs only say `{"model_type":"superpoint"}`; DinoML should allowlist SuperPoint first and reject or separately audit any other detector body.
- `max_keypoints=-1` means the detector emits value-dependent, batch-padded `K = max(nonzero(scores > threshold))`; static graph admission needs a cap, bucketing, or external keypoint input mode.
- Source uses NCHW convs and spatial ops in SuperPoint. NHWC/channel-last should be a guarded layout optimization for the detector only; the matcher already works on rank-3 keypoint sequences.
- Processor returns grayscale values duplicated over 3 channels; SuperPoint immediately selects channel 0. A controlled preprocessing rewrite can emit `[B*2,1,H,W]` only if it also owns the processor-to-detector boundary.
- `hidden_size % num_attention_heads == 0` is validated; no GQA/MQA.
- GNN layer list can be config-edited. Only `"self"` and `"cross"` are valid, but length/order are not fixed by code.
- `sinkhorn_iterations` is config-driven and defaults to 100; lowering should not silently bake 100 unless config is fixed.
- Matching uses integer/boolean indexing, `gather`, `where`, `max(...).indices`, mutual consistency checks, and output `-1` sentinels; this is structured post-score logic, not a simple dense head.
- Official configs are float32. Reduced precision needs explicit parity because log-space Sinkhorn and thresholding are numerically sensitive.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape/view`, `transpose`, `permute`, `contiguous`, `flip`, `cat`, `clone`, `unsqueeze`, `expand`, `gather`, `masked_fill`, `where`, `logical_and`, boolean comparisons, `nonzero`, `topk`, `max(values+indices)`, advanced indexing, padding writes into zero-initialized batch tensors.
- Pair flatten/unflatten: `[B,2,3,H,W] -> [B*2,3,H,W]`; detector outputs `[B*2,K,*] -> [B,2,K,*]`.

Neural network primitives:

- SuperPoint conv blocks in NCHW: 3x3 Conv2d + ReLU + optional 2x2 MaxPool, channels `1->64->64->128->128`.
- SuperPoint score head: Conv2d `128->256` 3x3 + ReLU, Conv2d `256->65` 1x1, softmax over channel axis, dustbin drop, pixel shuffle-like reshape to full-resolution score map.
- SuperPoint descriptor head: Conv2d `128->256` 3x3 + ReLU, Conv2d `256->256` 1x1, L2 normalize over channel axis, bilinear `grid_sample`, L2 normalize again.
- SuperGlue keypoint MLP: Linear `3->32`, BatchNorm1d over channel after transpose, ReLU; Linear/BN/ReLU `32->64`, `64->128`, `128->256`; final Linear `256->256`.
- GNN MLP per layer: concat descriptor+attention output gives `512`; Linear/BN/ReLU `512->512`; final Linear `512->256`; residual add.
- Final projection: Linear `256->256` with bias.

Attention primitives:

- Noncausal self-attention over keypoints and noncausal cross-attention from one image's keypoints to the paired image's keypoints.
- Standard MHA: `H=4`, `D=64`, Q/K/V Linear `256->256` with bias; output Linear `256->256` with bias.
- Attention math: `Q @ K^T / sqrt(64)`, add extended mask, softmax over key axis, dropout no-op, `P @ V`.

Position/custom math:

- Keypoint coordinate normalization: subtract image center `[W,H]/2`, divide by `0.7 * max(W,H)`, concatenate keypoint score.
- No RoPE, ALiBi, learned absolute position table, or token position IDs.

Structured-output and postprocessing ops:

- Log-space optimal transport with dustbin row/column, `logsumexp` along rows/columns for `sinkhorn_iterations`.
- Mutual-nearest matching using max indices in both directions, `gather`, threshold, invalid match sentinel `-1`.
- Postprocess rescales relative keypoints by original `(width,height)`, masks padded rows, filters by score and bounds.

Preprocessing-coupled ops:

- Image pair validation/flattening, resize, rescale, grayscale conversion. End-to-end parity depends on processor behavior but these can be CPU/data-pipeline owned for first DinoML integration.

## 5. Layer/block breakdown

SuperGlue forward:

```text
pixel_values: [B,2,3,H,W], default H=480,W=640
flatten pairs -> [B*2,3,H,W]
SuperPoint -> keypoints [B*2,K,2], scores [B*2,K], descriptors [B*2,K,256], mask [B*2,K]
reshape -> [B,2,K,*]
absolute_keypoints = relative_keypoints * [W,H]
match image pair
```

SuperPoint detector, default:

```text
x: [B*2,3,H,W] -> x[:,0:1,:,:]
ConvBlock 1: Conv 1->64, ReLU, Conv 64->64, ReLU, MaxPool2d /2
ConvBlock 2: Conv 64->64, ReLU, Conv 64->64, ReLU, MaxPool2d /2
ConvBlock 3: Conv 64->128, ReLU, Conv 128->128, ReLU, MaxPool2d /2
ConvBlock 4: Conv 128->128, ReLU, Conv 128->128, ReLU
score head -> [1,65,H/8,W/8] -> softmax(channel) -> [1,H,W] scores -> NMS/nonzero/topk -> K keypoints
descriptor head -> [1,256,H/8,W/8] -> normalize -> grid_sample at K keypoints -> [K,256]
pad per-image variable K to batch max
```

SuperGlue matcher:

```text
keypoints [B,2,K,2], scores [B,2,K], descriptors [B,2,K,256]
reshape pair axis into batch: [B*2,K,*]
normalize keypoints by image size
keypoint_encoder(cat([x,y,score])) -> [B*2,K,256]
descriptors += encoded_keypoints
extended mask from [B*2,K] to attention additive mask
for 18 layers:
  if self: attention source = descriptors
  if cross: attention source = paired image descriptors via reshape [B,2,K,256] + flip pair axis
  q,k,v = Linear(256->256)
  attention -> [B*2,K,256]
  message = Linear(256->256)
  delta = MLP(cat([descriptors,message])) -> [B*2,K,256]
  descriptors = descriptors + delta
final Linear(256->256)
reshape -> desc0, desc1 [B,K,256]
scores = desc0 @ desc1^T / sqrt(256) -> [B,K,K]
mask invalid pair scores to dtype min
log_optimal_transport -> [B,K+1,K+1]
mutual-nearest filter -> matches/matching_scores [B,2,K]
```

## 6. Attention requirements

Attention is encoder-style graph attention over detected keypoints; no causal mask and no KV cache.

| Variant | Query | Key/value | Shape | Mask | Cache |
|---|---|---|---|---|---|
| Self | same image keypoint descriptors | same image descriptors | `[B*2,4,K,64] @ [B*2,4,64,K]` | additive mask over valid keypoints | none |
| Cross | each image descriptors | paired image descriptors | same Q length `K`, source K from flipped pair axis | paired-image additive mask | none |

FlashAttention/SDPA compatibility: mathematically compatible with dense noncausal MHA when `K` is statically padded and masks are additive. However, cross-attention source construction via pair-axis `reshape(...).flip(1)` and value-dependent `K` must be guarded before using a fused attention path. `output_attentions=True` requires returning dense attention probabilities and can be deferred for fast path.

## 7. Position encoding and custom math

No sequence positional embedding is used. Spatial information enters through normalized detected keypoint coordinates and detector descriptor sampling.

```python
def normalize_keypoints(keypoints, height, width):
    size = tensor([width, height])
    center = size / 2
    scaling = max(width, height) * 0.7
    return (keypoints - center) / scaling
```

Log-space optimal transport:

```python
def sinkhorn(log_cost, log_mu, log_nu, iterations):
    u = zeros_like(log_mu)
    v = zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - logsumexp(log_cost + v[:, None, :], dim=2)
        v = log_nu - logsumexp(log_cost + u[:, :, None], dim=1)
    return log_cost + u[:, :, None] + v[:, None, :]
```

The dustbin/bin score is a learned scalar parameter initialized to 1.0. `log_source_distribution` and `log_target_distribution` depend on runtime `K` and batch.

## 8. Preprocessing and input packing

Processor contract:

- Accepted input: a pair of images or a list of image pairs.
- Flattening: processor flattens pairs, processes each image, then stacks every pair into `pixel_values[B,2,C,H,W]`.
- Default image size: `[H,W]=[480,640]`, bilinear resize.
- Rescale: multiply by `1/255`.
- Grayscale: enabled. Torchvision path returns 3 identical channels; PIL path may convert PIL to L before later backend tensorization. The model's detector then keeps channel 0.
- Normalization mean/std: absent (`do_normalize=None`).

Postprocess contract:

- Input `target_sizes`: shape `[B,2,2]` or list of pair sizes in `(height,width)`.
- Rescale `outputs.keypoints` by flipped sizes to absolute `(x,y)`.
- Use `mask` to discard padded detector rows.
- Keep only matches from image 0 side where score exceeds threshold, match index is nonnegative, and match index is within valid image 1 keypoint rows.
- No source NMS is applied after matching; NMS is only in the SuperPoint detector.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor-owned grayscale channel collapse

Source pattern:

```text
RGB/duplicated-gray processor output [B*2,3,H,W] -> SuperPoint extract channel 0 -> [B*2,1,H,W]
```

Replacement:

```text
Processor emits [B*2,1,H,W] or graph inserts channel-select once at detector boundary
```

Preconditions: DinoML owns the processor-to-model ABI, `do_grayscale=True`, and detector is SuperPoint-compatible. Failure cases: custom detector needing 3 channels; user disables grayscale; remote processor emits non-identical RGB.

Parity test: compare SuperPoint outputs for duplicated grayscale 3-channel input and collapsed 1-channel path after first conv.

### Rewrite: SuperPoint score pixel shuffle canonicalization

Source pattern:

```text
Conv2d 256->65 -> softmax(channel) -> drop dustbin -> permute/reshape from [B,64,Hc,Wc] to [B,Hc*8,Wc*8]
```

Replacement:

```text
ChannelSoftmax -> DropLastChannel -> DepthToSpace(block=8, source order matching permute(0,2,3,1).reshape(...).permute(...))
```

Preconditions: `keypoint_decoder_dim == 65`, cell size 8, source NCHW layout preserved or layout pass rewrites channel/spatial axes exactly. Failure cases: different keypoint decoder dimension or detector replacement.

Parity test: randomized logits compare source permute/reshape against DepthToSpace lowering.

### Rewrite: conv detector NHWC region

Source pattern: NCHW Conv2d/ReLU/MaxPool chain and 1x1/3x3 decoder convs.

Replacement: guarded NHWC/channel-last conv region.

Preconditions: entire SuperPoint conv/softmax/pool/normalize/grid-sample coordinate path is rewritten together or an explicit boundary transpose is inserted before axis-sensitive ops. Required axis rewrites: softmax `dim=1 -> dim=-1`, L2 normalize `dim=1 -> dim=-1`, MaxPool spatial axes stay H/W, descriptor map layout for `grid_sample` must match the provider ABI. Failure cases: `grid_sample` provider only supports NCHW; score extraction with `nonzero` assumes source full-res map order.

Parity test: detector dense feature maps at each stage for fixed images, then keypoint/match parity.

### Rewrite: GNN attention to fused dense MHA

Source pattern: separate Q/K/V linears, reshape to heads, matmul/scale/mask/softmax/dropout/matmul, output linear.

Replacement: fused noncausal MHA for self/cross keypoint attention.

Preconditions: dropout probability 0 or inference mode, no `output_attentions`, static/bucketed padded `K`, additive mask semantics preserved, Q/K/V weights unpacked as independent Linear modules. Failure cases: returning attentions, nonzero dropout, unbounded `K`.

Parity test: per-layer attention output with random descriptors and masks; include cross layer pair-axis flip.

### Rewrite: Sinkhorn loop as custom fixed-iteration kernel

Source pattern: 100 Python iterations of two `logsumexp` reductions and broadcasts over `[B,K+1,K+1]`.

Replacement: custom optimal transport kernel or generated reduction loop.

Preconditions: fixed `sinkhorn_iterations`, bounded `K`, dtype policy chosen (prefer fp32 first). Failure cases: fp16 underflow/threshold drift, changing iteration count, very large K needing tiled reductions.

Parity test: random score matrices and masks, compare transport matrix and final mutual matches.

## 10. Kernel fusion candidates

Highest priority:

- SuperPoint conv blocks in NCHW or guarded NHWC: they dominate raw-image detector work and are required for end-to-end parity.
- Value-dependent keypoint extraction/NMS/top-k: this is the largest admission gap because it controls dynamic `K`, padding, masks, and downstream attention sizes.
- GNN MHA over keypoints: 18 dense attention layers over padded `K` with small hidden size; fused attention or batched GEMM dispatch matters once `K` approaches hundreds.
- Log-space Sinkhorn: 100 reduction iterations over `[B,K+1,K+1]`; a custom kernel avoids graph overhead and repeated temporary allocation.

Medium priority:

- Linear + BatchNorm1d + ReLU MLP fusion for keypoint encoder and propagation MLPs. BatchNorm is inference affine with running stats, so it can fold into preceding Linear weights when parameters are frozen.
- Final descriptor similarity GEMM plus scale and mask fill.
- Mutual match filtering as a structured post-score kernel using max values/indices and gather.

Lower priority:

- `output_hidden_states` and `output_attentions` materialization.
- Visualization and PIL drawing.
- Training/loss paths; source explicitly rejects labels.

## 11. Runtime staging plan

Stage 1: Parse SuperGlue and nested SuperPoint configs. Reject non-SuperPoint detector configs, non-float32 fast path, unbounded `max_keypoints` unless an integration cap/bucket policy is supplied.

Stage 2: Support a matcher-only entry that accepts precomputed `keypoints [B,2,K,2]`, `scores [B,2,K]`, `descriptors [B,2,K,256]`, and `mask [B,2,K]`. This isolates GNN, Sinkhorn, and match filtering without detector value-dependent ops.

Stage 3: Implement SuperPoint detector parity for fixed default 480x640 and static `max_keypoints` cap. Keep source NCHW first; add NHWC only as a guarded optimization.

Stage 4: Combine processor-owned preprocessing, detector, matcher, and postprocess for indoor/outdoor checkpoint parity.

Stage 5: Optimize: fused keypoint MHA, BN-folded MLPs, Sinkhorn custom kernel, guarded channel-last detector region.

Stage 6: Broaden admission: alternate `gnn_layers_types`, non-default image sizes, detector config overrides, reduced precision.

## 12. Parity and validation plan

- Unit test `normalize_keypoints` over multiple `(H,W)` and keypoint coordinates.
- Unit test SuperPoint score rearrange against source permute/reshape for random score logits.
- Unit test `simple_nms`, threshold/border removal/top-k, and descriptor `grid_sample` coordinate normalization.
- Single SuperPoint stage parity: fixed resized image, compare keypoints, scores, descriptors, and mask with source. Use exact or tight fp32 tolerances; keypoint indices should match exactly.
- Matcher-only parity: random bounded `K`, descriptors, scores, masks; compare per-layer GNN outputs, final scores, Sinkhorn matrix, matches, and matching scores.
- End-to-end official image-pair parity for indoor and outdoor checkpoints. The conversion script records reference output shapes of `[2,2,865]` for its sample images; use source-generated golden data at the inspected commit rather than relying on documentation.
- Suggested tolerances: fp32 `atol=1e-5` for dense blocks, `1e-4` for end-to-end matching scores. For fp16/bf16, validate match-index stability separately because thresholding and mutual-nearest indices are discontinuous.

## 13. Performance probes

- Processor throughput: resize/grayscale/rescale images/sec.
- SuperPoint detector throughput for 480x640, plus sweeps over H/W if dynamic sizes are admitted.
- Keypoint count sweep: `K=128,256,512,865,1024` for GNN attention, descriptor similarity, and Sinkhorn.
- Sinkhorn iteration sweep: 20/50/100/200 for kernel cost and numerical drift.
- Batch/pair sweep: `B=1,2,4,8`, tracking detector and matcher separately.
- NCHW vs guarded NHWC detector conv region, with and without boundary transposes.
- End-to-end matches/sec for indoor/outdoor sample pairs.
- Memory probes for padded `K`: attention matrices `[B*2,4,K,K]` and Sinkhorn `[B,K+1,K+1]`.

## 14. Skip/defer list

- Training; source rejects labels for SuperGlue and SuperPoint.
- `output_hidden_states=True` and `output_attentions=True` on optimized paths.
- Visualization helpers.
- Arbitrary nested keypoint detector families.
- Unbounded `max_keypoints=-1` production admission without an explicit dynamic output/bucketing plan.
- Reduced precision Sinkhorn and match filtering until fp32 parity is stable.
- Multi-GPU/distributed execution.

## 15. Final implementation checklist

- [ ] Parse `SuperGlueConfig` and nested `SuperPointConfig`.
- [ ] Load official indoor/outdoor F32 weights and preserve `bin_score`.
- [ ] Add admission guard: detector must be SuperPoint for first end-to-end target.
- [ ] Add matcher-only ABI for precomputed keypoints/descriptors/scores/mask.
- [ ] Implement/fuse Linear + folded BatchNorm1d + ReLU for keypoint/propagation MLPs.
- [ ] Implement noncausal self/cross MHA over padded keypoint sequences.
- [ ] Implement keypoint coordinate normalization.
- [ ] Implement log-space optimal transport with configurable fixed iterations.
- [ ] Implement mutual-nearest match filtering with `-1` invalid sentinel.
- [ ] Implement SuperPoint NCHW conv/pool score and descriptor heads.
- [ ] Implement SuperPoint score reshaping, NMS, threshold, border removal, top-k/cap.
- [ ] Implement descriptor bilinear `grid_sample` and L2 normalization.
- [ ] Add guarded NHWC detector rewrite notes/tests after NCHW parity.
- [ ] Add processor/postprocess parity for default 480x640 image pairs.
- [ ] Add indoor/outdoor end-to-end golden tests.
- [ ] Benchmark detector, matcher attention, Sinkhorn, and end-to-end matches/sec.
