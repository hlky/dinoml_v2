# EoMT-DINOv3 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: representative public converted checkpoints under `tue-mps/eomt-dinov3-*`; source docstring also names older conversion ids such as `tue-mps/coco_panoptic_eomt_large_640_dinov3`, but those returned 404 for `config.json`.

Config source: fetched `config.json` and `preprocessor_config.json` for:

- `tue-mps/eomt-dinov3-coco-panoptic-small-640`
- `tue-mps/eomt-dinov3-coco-panoptic-base-640`
- `tue-mps/eomt-dinov3-coco-panoptic-large-640`
- `tue-mps/eomt-dinov3-coco-panoptic-large-1280`
- `tue-mps/eomt-dinov3-coco-instance-large-640`
- `tue-mps/eomt-dinov3-coco-instance-large-1280`
- `tue-mps/eomt-dinov3-ade-semantic-large-512`

Source files inspected:

- `src/transformers/models/eomt_dinov3/modular_eomt_dinov3.py`
- `src/transformers/models/eomt_dinov3/modeling_eomt_dinov3.py`
- `src/transformers/models/eomt_dinov3/configuration_eomt_dinov3.py`
- `src/transformers/models/eomt_dinov3/convert_eomt_dinov3_to_hf.py`
- `src/transformers/models/dinov3_vit/modeling_dinov3_vit.py`
- `src/transformers/models/dinov3_vit/configuration_dinov3_vit.py`
- `src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py`
- `src/transformers/models/eomt/image_processing_eomt.py`
- `src/transformers/models/eomt/modeling_eomt.py` for inherited segmentation heads/loss/process shape context.

Any missing files or assumptions: official DINOv3 base backbone configs/weights at `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, and `facebook/dinov3-vitl16-pretrain-lvd1689m` returned 401 without gated access. The public converted EoMT-DINOv3 configs are accessible and already include the effective composed DINOv3 dimensions.

`modeling_eomt_dinov3.py` and `configuration_eomt_dinov3.py` are generated from `modular_eomt_dinov3.py`; future source edits should target the modular file. This report treats the generated modeling file as the expanded runtime source and the modular file as the authoritative edit source.

## 2. High-level architecture

Primary runtime target: inference for `EomtDinov3ForUniversalSegmentation`, producing mask-query logits and class-query logits for semantic, instance, or panoptic segmentation. Training losses, Hungarian matching, and point sampling are deferred.

Dataflow:

```text
image processor -> CHW pixel_values -> DINOv3 patch embedding + prefix tokens -> encoder-only ViT blocks
  -> insert learned query tokens before final N blocks -> mask-conditioned self-attention in final blocks
  -> query class head + query mask head + upscaled patch features -> segmentation postprocess
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize, optional patch splitting, optional right/bottom padding, rescale, ImageNet normalization, `patch_offsets` metadata.
- GPU model body: Conv2d patch embedding, dynamic 2D RoPE, noncausal self-attention over prefix/query/patch tokens, MLP blocks, mask attention updates, ConvTranspose/depthwise Conv2d upscale head.
- Postprocess: semantic `einsum` over query class probabilities and mask probabilities, optional patch merge, unpad/resize, argmax; instance/panoptic thresholding and segment record construction.

## 3. Important config dimensions

| Field | Source default | Observed sweep |
| --- | ---: | --- |
| `hidden_size` | 1024 | 384 small, 768 base, 1024 large |
| `num_hidden_layers` | 24 | 12 small/base, 24 large |
| `num_attention_heads` | 16 | 6, 12, 16 |
| `head_dim` | `hidden_size / heads` | 64 for all inspected checkpoints |
| `intermediate_size` | 4096 | 1536, 3072, 4096 |
| `patch_size` | 16 | 16 |
| `image_size` | 640 | 512, 640, 1280 |
| patch grid | `image_size / patch_size` | 32x32, 40x40, 80x80 |
| `num_queries` | 200 | 200 for COCO, 100 for ADE semantic |
| `num_register_tokens` | 4 | 4 |
| `num_blocks` | 4 | 3 for small/base COCO panoptic, 4 for large |
| `num_upscale_blocks` | 2 | 2 |
| labels | from `id2label` | 133 COCO panoptic, 80 COCO instance, 150 ADE |
| RoPE | default theta 100 | config has `rope_theta=100.0`, source uses default RoPE only |
| projection bias | q/v/o true, k false | same in sweep |
| MLP | GELU, biased, ungated default false | inspected checkpoints ungated |

Representative sweep details are in `config_sweep.md`.

## 3a. Family variation traps

- The public model IDs use `eomt-dinov3-*`; older underscore-style ids in the conversion catalog returned 404 for configs.
- Backbone DINOv3 repos are gated. Do not require direct backbone checkpoint access for converted EoMT-DINOv3 inference if the combined EoMT weights/config are present.
- `num_blocks` controls where learned query tokens are inserted: at `num_hidden_layers - num_blocks`. Small/base panoptic insert queries for 3 final blocks; large variants use 4.
- `num_queries` changes by task: COCO configs use 200; ADE semantic uses 100.
- `class_predictor` output is `num_labels + 1`; the final class is the no-object/null class and is removed in semantic/instance scoring.
- Attention mask semantics are unusual in final blocks: mask logits become a boolean query-to-patch visibility mask, then are expanded to `[B, heads, S, S]` and converted to additive dtype-min values for disallowed entries.
- Source model input is NCHW/CHW. Initial translation should preserve NCHW; NHWC is a guarded optimization only around local conv/norm/head regions.
- `LayerNorm2d` explicitly permutes NCHW to NHWC for channel LayerNorm and back. A layout pass can remove these permutes only if all surrounding ConvTranspose/Conv2d consumers are rewritten consistently.
- `grid_size` is computed from `config.image_size // patch_size` in model init and used in `predict()`. Dynamic input sizes must either match the configured grid or receive a guarded fix; source RoPE itself can compute dynamic patch grids, but mask prediction reshape uses the static config grid.
- `rope_type` other than default raises. Reject non-default RoPE configs.
- `use_gated_mlp=True` is implemented by inherited DINOv3 ViT blocks, but inspected EoMT-DINOv3 checkpoints set it false.
- Training-only fields (`train_num_points`, `oversample_ratio`, matcher/loss weights, random RoPE augmentation) are not first-inference requirements.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation; resize/pad metadata stays in processor/postprocess.
- Conv2d patch embedding: `[B,3,H,W] -> [B,C,H/16,W/16]`, kernel=stride=16, no explicit padding.
- `flatten(2)`, `transpose(1,2)`, `cat` for CLS/register/query/patch token sequences.
- `view`/`reshape` for `[B,S,C] -> [B,S,Hd,head_dim]`, then transpose to `[B,heads,S,head_dim]`.
- `split` and `cat` around RoPE prefix-vs-patch tokens.
- Static-grid reshape in `predict`: patch tokens `[B, Hgrid*Wgrid, C] -> [B,C,Hgrid,Wgrid]`.
- NCHW/NHWC/NCHW permutes in `LayerNorm2d`.

Neural network primitives:

- Linear projections `C -> C`: q bias true, k bias false, v bias true, o bias true.
- MLP: `Linear(C -> 4C)` + GELU + `Linear(4C -> C)`; optional gated MLP is `GELU(gate) * up -> down`.
- LayerNorm over sequence channel with `eps=1e-6`.
- LayerScale multiply by learned `[C]` vector.
- Learned embeddings: CLS `[1,1,C]`, register `[1,4,C]`, query `[num_queries,C]`.
- Segmentation class head: `Linear(C -> num_labels + 1)`.
- Mask head: three `Linear(C -> C)` layers with GELU after first two.
- Upscale head repeated twice: `ConvTranspose2d(C,C,kernel=2,stride=2)` -> GELU -> depthwise `Conv2d(C,C,kernel=3,pad=1,groups=C,bias=False)` -> channel LayerNorm.

Attention primitives:

- Noncausal dense self-attention, no KV cache, MHA only in inspected configs.
- Q/K RoPE applied only to patch-token suffix, not query/CLS/register prefix.
- Additive attention mask optional in final segmentation blocks; mask rank is `[B,heads,S,S]`.
- Source supports Transformers attention backends through `ALL_ATTENTION_FUNCTIONS`; eager path is matmul-scale-add-softmax-dropout-matmul.

Position/custom math:

- 2D patch-center RoPE from normalized patch center coordinates in `[-1,1]`.
- `cos`/`sin` computed in float32 and cast to input dtype.
- `rotate_half`, concat, cos/sin multiply-add.

Pre/postprocessing-coupled ops:

- Processor emits `pixel_values` as channels-first and optional `patch_offsets`.
- Semantic postprocess uses mask sigmoid, class softmax without null class, `einsum("bqc,bqhw->bchw")`, resize/merge, argmax.
- Instance/panoptic postprocess uses unpad/resize, thresholding, boolean masks, score multiplication, per-query loops, no NMS.

## 5. Layer/block breakdown

Patch/prefix embedding:

```text
pixel_values [B,3,H,W]
patch = Conv2d(3,C,k=16,s=16)(pixel_values)              # [B,C,H/16,W/16]
patch = flatten_hw_transpose(patch)                      # [B,Npatch,C]
hidden = cat(cls[1], register[R], patch[Npatch], dim=1)  # [B,1+R+Npatch,C]
```

Encoder block, repeated `num_hidden_layers` times:

```text
res = x
x = LayerNorm(x)
q = Linear(C,C,bias=True)(x)
k = Linear(C,C,bias=False)(x)
v = Linear(C,C,bias=True)(x)
q,k,v = reshape_to_heads([B,S,C] -> [B,H,S,64])
q_patch,k_patch = apply_2d_rope(q_patch,k_patch)
x = attention(q,k,v, optional additive mask)
x = Linear(C,C,bias=True)(x)
x = res + LayerScale(x)

res = x
x = LayerNorm(x)
x = Linear(C,4C,bias=True) -> GELU -> Linear(4C,C,bias=True)
x = res + LayerScale(x)
```

Query insertion and final mask-conditioned blocks:

```text
at layer index num_hidden_layers - num_blocks:
  query = Embedding(num_queries,C).expand(B,-1,-1)
  hidden = cat(query, cls/register/patch hidden, dim=1)

for each final block:
  if mask conditioning enabled:
    norm_hidden = LayerNorm(hidden)
    masks, classes = predict(norm_hidden)
    interp_masks = bilinear_resize(masks, grid_size).view(B,Q,Npatch)
    attn_mask = ones([B,S,S], bool)
    attn_mask[:, :Q, Q+prefix:] = interp_masks > 0
    attn_mask = expand_to_heads_and_additive(attn_mask)
  hidden = encoder_block(hidden, attn_mask)
```

Prediction head:

```text
query_tokens = hidden[:, :Q, :]
class_logits = Linear(C,num_labels+1)(query_tokens)

patch_tokens = hidden[:, Q+prefix:, :].transpose(1,2).reshape(B,C,Hgrid,Wgrid)
patch_features = upscale_block(patch_tokens)  # after 2 blocks, 4x spatial: [B,C,4Hgrid,4Wgrid]
query_features = MLP3(query_tokens)           # [B,Q,C]
mask_logits = einsum("bqc,bchw->bqhw", query_features, patch_features)
```

## 6. Attention requirements

EoMT-DINOv3 requires encoder-style, noncausal self-attention. There is no autoregressive generation, no prefill/decode split, and no KV cache.

- Attention type: self-attention over a mixed sequence of query tokens, CLS token, register tokens, and patch tokens.
- Head pattern: MHA, `num_attention_heads` heads, `head_dim=64` in inspected configs.
- Query/key/value width: all project `hidden_size -> hidden_size`; no GQA/MQA in source.
- Masking: none before query insertion; final blocks may use a per-query segmentation mask to restrict query-to-patch attention. Source uses `interpolated_logits > 0` as keep mask, then additive dtype minimum where false.
- Rectangular attention: not used; attention matrix is `[S,S]`.
- Flash/SDPA compatibility: source dispatches through Transformers attention backend. A fused backend must preserve RoPE-before-attention and additive mask semantics.
- Output attentions are optional library outputs, not required for first inference parity.

## 7. Position encoding and custom math

The custom RoPE is 2D and patch-only:

```python
def eomt_dinov3_rope(pixel_values, inv_freq, patch_size):
    h, w = pixel_values.shape[-2:]
    gh, gw = h // patch_size, w // patch_size
    yy = (torch.arange(0.5, gh) / gh) * 2.0 - 1.0
    xx = (torch.arange(0.5, gw) / gw) * 2.0 - 1.0
    coords = torch.stack(torch.meshgrid(yy, xx, indexing="ij"), dim=-1).flatten(0, 1)
    angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2).tile(2)
    return torch.cos(angles), torch.sin(angles)

def apply_patch_only_rope(q, k, cos, sin):
    n_patches = sin.shape[-2]
    n_prefix = q.shape[-2] - n_patches
    q0, qp = q.split((n_prefix, n_patches), dim=-2)
    k0, kp = k.split((n_prefix, n_patches), dim=-2)
    qp = qp * cos + rotate_half(qp) * sin
    kp = kp * cos + rotate_half(kp) * sin
    return torch.cat([q0, qp], dim=-2), torch.cat([k0, kp], dim=-2)
```

Inference can precompute cos/sin per `(grid_h, grid_w, dtype, device)` when input sizes are bucketed. Training-only coordinate shift/jitter/rescale can be deferred.

## 8. Preprocessing and input packing

EoMT image processor contract:

- Input images are prepared as channels-first tensors.
- Defaults: bilinear resize, rescale by `1/255`, ImageNet mean/std normalize, `data_format="channels_first"`.
- Public converted EoMT-DINOv3 preprocessors set `do_pad=true`, `do_split_image=false`, and resize/pad to square `512`, `640`, or `1280`.
- Padding is right/bottom zero padding after resize.
- Optional semantic splitting slices along the longer axis into overlapping square patches and emits `patch_offsets=[image_index,start,end]`; this is useful for semantic segmentation but not enabled in inspected converted configs.

Postprocess ABI:

- Inputs are model outputs `masks_queries_logits [B,Q,Hmask,Wmask]`, `class_queries_logits [B,Q,num_labels+1]`, `patch_offsets`, and `target_sizes [(Horig,Worig)]`.
- Semantic: resize masks to processor target size, `softmax(class)[...,:-1]`, `sigmoid(mask)`, class-mask einsum to `[B,num_labels,H,W]`, optional patch merge, resize to original size, argmax.
- Panoptic: resize/unpad masks, take max class including null, filter `label != num_labels` and `score > threshold`, weighted per-pixel argmax, segment validity checks, stuff-class merging.
- Instance: resize/unpad masks, remove null class, threshold by score, binary mask from `mask_pred > 0`, average sigmoid mask score over selected pixels, no NMS.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern: `Conv2d(3,C,kernel=patch_size,stride=patch_size,padding=0)` followed by `flatten(2).transpose(1,2)`.

Replacement: `WindowFlatten_NCHW(patch_size, patch_size) -> Linear(3*patch_size*patch_size -> C)`.

Preconditions: kernel equals stride, padding zero, dilation one, groups one, input height/width divisible by patch size or processor padding ensures divisibility, and flatten order matches PyTorch NCHW conv weight layout.

Weight transform: `W_linear = conv.weight.reshape(C, 3*ph*pw)`, `bias = conv.bias`.

Failure cases: dynamic unpadded sizes not divisible by 16; NHWC pass must explicitly rewrite window flatten order and weight layout.

Parity test sketch: compare patch embeddings for random `[B,3,512,512]`, `[B,3,640,640]`, `[B,3,1280,1280]` in fp32/fp16.

### Rewrite: Q/K/V projection packing

Source pattern: separate q/k/v linears with split biases: q and v have bias, k usually no bias.

Replacement: packed GEMM producing `[q,k,v]` blocks, with a zero synthetic bias block for k if the fused provider requires uniform bias.

Preconditions: all projections have identical in/out width; no weight sharing; split order must be q, k, v.

Failure cases: future configs with differing q/k/v widths or GQA/MQA are unsupported by current source anyway.

### Rewrite: mask einsum -> batched GEMM

Source pattern: `torch.einsum("bqc,bchw -> bqhw", query_tokens, prefix_tokens)`.

Replacement: reshape prefix to `[B,C,Hmask*Wmask]`, batched GEMM `[B,Q,C] x [B,C,HW] -> [B,Q,HW]`, reshape to `[B,Q,Hmask,Wmask]`.

Preconditions: prefix features contiguous or explicitly packed; query head output and prefix channels equal `C`.

### Rewrite: semantic postprocess einsum -> batched GEMM

Source pattern: `einsum("bqc,bqhw -> bchw")`.

Replacement: `[B,Cclass,Q] x [B,Q,HW] -> [B,Cclass,HW]`.

Preconditions: postprocess remains in GPU runtime or a dedicated tensor postprocess stage; class null channel already sliced off.

### Layout pass: NCHW upscale head local NHWC optimization

Candidate region: `ConvTranspose2d -> GELU -> depthwise Conv2d -> LayerNorm2d`.

Preconditions: entire upscale block and final mask GEMM consumer are under layout control. Axis rewrites include channel norm axis `C` from NCHW permute pattern to NHWC last dim, conv weight transforms, and final mask feature layout.

Failure cases: source-visible output of `prefix_tokens` as NCHW, `einsum("bqc,bchw")`, or postprocess expects NCHW masks. Use a no-layout-translation guard at model input and postprocess ABI boundaries.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,S,C]` and `LayerNorm2d` over channel, because they appear in every transformer block and every upscale block.
- QKV packed GEMM + RoPE preparation, preserving patch-only RoPE split.
- Dense noncausal MHA/SDPA with optional additive mask for final blocks.
- MLP GELU block with bias and residual/layerscale fusion.
- Batched GEMM for mask logits and semantic class-mask logits.

Medium priority:

- Conv2d patch embedding lowered to GEMM for fixed 16x16 patches.
- ConvTranspose2d and depthwise Conv2d upscale kernels, possibly NHWC internally.
- Attention-mask construction from mask logits: resize, compare `>0`, query-to-patch slice update, additive mask conversion.
- Bilinear resize for masks in final-block mask conditioning and postprocess.

Lower priority:

- Optional patch splitting/merging pipeline for semantic large images.
- Panoptic/instance variable-length segment assembly loops; can remain CPU/Python initially.
- Training losses, point sampling, Hungarian matching.

## 11. Runtime staging plan

Stage 1: parse config and load converted EoMT-DINOv3 weights for one small/base checkpoint. Reject gated standalone DINOv3 backbone composition unless full combined weights are provided.

Stage 2: implement patch embedding, prefix tokens, 2D RoPE, one DINOv3 ViT block parity with fixed square `pixel_values`.

Stage 3: run full encoder up to query insertion without mask conditioning; validate hidden states for small/base shapes.

Stage 4: add learned query insertion and final-block mask-conditioned attention mask construction.

Stage 5: add prediction head: class linear, mask MLP, upscale block, mask-logit GEMM.

Stage 6: add semantic postprocess first. Instance/panoptic postprocess can follow as separate ABI work because it includes thresholds and variable-length segment records.

Stage 7: optimize with patch Conv2d rewrite, packed QKV, fused attention, fused MLP/residual/layerscale, and batched GEMM mask heads.

Initial stubs: training loss, matcher, auxiliary per-layer loss outputs, random mask disabling in training, optional split-image patch merge, panoptic/instance segment assembly.

## 12. Parity and validation plan

- Config parser tests for all seven public converted configs; assert grid, labels, query count, final block count.
- RoPE unit tests for `[H,W] = 512,640,1280`, comparing cos/sin and patch-only q/k application.
- Patch embedding Conv2d-vs-linear rewrite tests in fp32 and fp16.
- Single block parity with random hidden states and no attention mask.
- Final block parity with synthetic mask-conditioned attention mask and query tokens.
- Full forward parity for `small-640` random pixels: compare `masks_queries_logits`, `class_queries_logits`, `last_hidden_state`.
- Postprocess parity for semantic output with fixed random logits and target sizes; compare argmax maps.
- Instance/panoptic postprocess parity with deterministic logits and thresholds before moving those loops into runtime.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for block/head parity; fp16/bf16 `rtol=5e-3, atol=5e-3` for full model logits, with stricter tests around pure indexing/reshape/mask construction.

## 13. Performance probes

- Processor throughput: resize+pad+normalize for 512/640/1280 and optional split-image path.
- Patch embedding rewrite throughput versus Conv2d.
- Encoder-only block throughput at sequence lengths: 1029 for 512 large (`1+4+1024`), 1605 before query and 1805 after query for 640 COCO, 6405 before query and 6605 after query for 1280 COCO.
- Final-block attention with additive mask versus no mask.
- Mask-head throughput: upscale ConvTranspose/depthwise Conv2d and `B,Q,C x B,C,HW`.
- Semantic postprocess GPU-vs-CPU: bilinear resize, sigmoid/softmax, class-mask GEMM, argmax.
- Batch-size sweep for `small-640`, `base-640`, `large-640`; separate 1280 because attention sequence length dominates.
- Memory probe for attention logits/masks in 1280: `[B,16,6605,6605]` additive masks are very large and need backend strategy scrutiny.

## 14. Skip/defer list

- Training, Hungarian matcher, point sampling, dice/BCE losses, auxiliary loss accumulation.
- Standalone DINOv3 backbone checkpoint conversion requiring gated Meta weights.
- Non-default RoPE.
- `use_gated_mlp=True` until an actual EoMT-DINOv3 checkpoint uses it.
- Dynamic arbitrary image sizes unless `predict()` static `grid_size` handling is fixed or guarded.
- Instance/panoptic variable-length record assembly for first semantic-only runtime.
- Split-image semantic patch merge unless targeting that processor mode.
- Multi-GPU/distributed and `accelerate` loss reduction.

## 15. Final implementation checklist

- [ ] Parse `EomtDinov3Config` and normalize `rope_theta`/`rope_parameters`.
- [ ] Load converted EoMT-DINOv3 weights and preserve q/k/v bias asymmetry.
- [ ] Implement NCHW image ABI and processor metadata guards.
- [ ] Implement Conv2d patch embedding or guarded Conv2d-to-linear rewrite.
- [ ] Implement CLS/register/query learned token expansion and concatenation.
- [ ] Implement 2D patch-center RoPE and patch-only q/k rotation.
- [ ] Implement noncausal MHA with optional additive `[B,H,S,S]` mask.
- [ ] Implement LayerNorm, LayerScale, GELU MLP, and residual ordering.
- [ ] Implement query insertion for the last `num_blocks` layers.
- [ ] Implement mask-conditioned attention-mask construction.
- [ ] Implement class predictor, 3-layer mask head, upscale block, and mask-logit batched GEMM.
- [ ] Implement semantic postprocess ABI: resize, softmax/sigmoid, class-mask GEMM, argmax.
- [ ] Add config sweep tests for small/base/large and 512/640/1280.
- [ ] Add block, full-forward, and semantic postprocess parity tests.
- [ ] Benchmark encoder attention, final mask-conditioned blocks, mask head, and postprocess separately.
