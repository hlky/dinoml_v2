# Transformers Audit: prompt_depth_anything

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: depth-anything/prompt-depth-anything-vitl-hf; also checked vits-hf and vits-transparent-hf
Config source: HF raw config/preprocessor JSON plus Transformers config defaults
Source files inspected:
  transformers/src/transformers/models/prompt_depth_anything/configuration_prompt_depth_anything.py
  transformers/src/transformers/models/prompt_depth_anything/modeling_prompt_depth_anything.py
  transformers/src/transformers/models/prompt_depth_anything/modular_prompt_depth_anything.py
  transformers/src/transformers/models/prompt_depth_anything/image_processing_prompt_depth_anything.py
  transformers/src/transformers/models/prompt_depth_anything/convert_prompt_depth_anything_to_hf.py
  transformers/src/transformers/models/dinov2/configuration_dinov2.py
  transformers/src/transformers/models/dinov2/modeling_dinov2.py
Any missing files or assumptions: no gated repos found; HF lookup for vitb-hf returned not found. modeling_prompt_depth_anything.py is generated from modular_prompt_depth_anything.py, so future source edits should inspect the modular file first.
```

Small snapshots are in `config_snapshot.md` in this folder.

## 2. High-level architecture

Prompt Depth Anything is an image depth-estimation model: CPU image/depth preprocessing, Dinov2 vision backbone, DPT-like reassemble/neck/fusion, optional prompt-depth conditioning at every fusion layer, and a convolutional depth head. The first DinoML target should be `PromptDepthAnythingForDepthEstimation` inference returning `predicted_depth`, not training or generation.

```text
RGB image + optional prompt depth
-> processor resize/rescale/normalize/prompt scaling
-> Dinov2 patch embedding + noncausal ViT encoder feature taps
-> token-to-feature reassembly
-> feature fusion + optional prompt-depth conv injection
-> depth head
-> optional postprocess resize to requested target size
```

Stageable pieces: processor parity, Dinov2 backbone parity, neck/head parity without prompt depth, prompt-conditioned parity, then postprocessing parity.

## 3. Important config dimensions

| field | vits-hf / transparent | vitl-hf | source/default notes |
|---|---:|---:|---|
| input layout | `[B,3,H,W]` | `[B,3,H,W]` | Processor emits channels-first tensors. |
| processor target | `756x756` aspect-preserving | `756x756` aspect-preserving | `ensure_multiple_of=14`, no pad. |
| patch_size | 14 | 14 | Used by Dinov2 patch conv and PromptDA head. |
| backbone | Dinov2 | Dinov2 | Report composes the Dinov2 source; a separate Dinov2 audit should own broad backbone coverage. |
| hidden_size | 384 | 1024 | From HF config. |
| num_hidden_layers | 12 effective | 24 | Small omits field, so Dinov2 default 12 applies. |
| num_attention_heads | 6 | 16 | Head dim is 64 in both. |
| qkv_bias | true effective | true effective | Dinov2 default when omitted. |
| mlp_ratio | 4 effective | 4 effective | MLP width 1536 small, 4096 large. |
| backbone out_indices | `[3,6,9,12]` | `[5,12,18,24]` | Four feature taps. |
| reshape_hidden_states | false | false | PromptDA reassembles 3D token features itself. |
| reassemble_hidden_size | 384 | 1024 | Must match backbone hidden size. |
| neck_hidden_sizes | `[48,96,192,384]` | `[256,512,1024,1024]` | Per-tap projection widths. |
| reassemble_factors | `[4,2,1,0.5]` | same | ConvTranspose, ConvTranspose, identity, stride-2 conv. |
| fusion_hidden_size | 64 | 256 | All fusion blocks run at this channel count. |
| head_hidden_size | 32 | 32 | Final conv stack channel. |
| depth_estimation_type | metric | metric | HF configs use sigmoid head then rescale by prompt-depth range if provided. |
| max_depth | 1 | 1 | Source sets missing/falsey max_depth to 1. |

Representative checkpoint sweep:

| checkpoint | availability | operator-significant variation |
|---|---|---|
| [depth-anything/prompt-depth-anything-vits-hf](https://hf.co/depth-anything/prompt-depth-anything-vits-hf) | open | Small Dinov2, 12 layers effective, fusion width 64. |
| [depth-anything/prompt-depth-anything-vits-transparent-hf](https://hf.co/depth-anything/prompt-depth-anything-vits-transparent-hf) | open | Same graph as small; treat as weight/data variant unless further processor metadata says otherwise. |
| [depth-anything/prompt-depth-anything-vitl-hf](https://hf.co/depth-anything/prompt-depth-anything-vitl-hf) | open | Large Dinov2, 24 layers, fusion width 256, wider neck. |
| `vitb/base` converter preset | source-only | Converter defines a base preset with Dinov2-base, fusion width 128, neck `[96,192,384,768]`; no HF-native `vitb-hf` repo was found. |

## 3a. Family variation traps

- The PromptDA report must compose a Dinov2 backbone. `backbone_config.model_type` can theoretically change through `AutoConfig`; first DinoML admission should allow only Dinov2 configs matching the audited feature contract.
- `reshape_hidden_states=false` is required for this source path. If true, PromptDA reassembly would receive 4D features but still removes `hidden_state[:, 1:]`, so reject or separately audit.
- Processor keeps aspect ratio and rounds to multiples of 14; dynamic `H,W` are possible and position embeddings may be bicubic-interpolated.
- Source tensors are NCHW after preprocessing. NHWC/channel-last is an optimization only for local conv regions and must rewrite axes around `permute`, `interpolate`, `Conv2d`, and `squeeze(dim=1)`.
- Prompt depth is optional. With prompt depth, source normalizes per sample with `min`/`max`, injects it at each fusion layer, then rescales output by the same range. Constant prompt depths are patched in the processor by adding `1e-6` to one element; callers that bypass the processor need a guard for `depth_max == depth_min`.
- `max_depth` is stored in the config/head but the inspected generated source does not multiply metric predictions by it; metric checkpoints use `Sigmoid` and prompt-depth range restoration when a prompt is supplied.
- Small HF config omits Dinov2 `num_hidden_layers`, `mlp_ratio`, `qkv_bias`, `layer_norm_eps`, and related defaults; DinoML config loading must materialize effective defaults.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW image tensors, channel-first prompt depth `[B,1,h,w]`.
- `reshape`, `permute(0,3,1,2)`, `contiguous`, `flatten(2)`, `transpose(1,2)`, `cat` for CLS token, slice `[:,1:]`, reverse feature list, `squeeze(dim=1)`.
- Per-sample `min`/`max` over flattened prompt depth, broadcast subtract/divide/multiply/add.

Neural primitives:
- Dinov2 patch `Conv2d(3 -> C, kernel=14, stride=14, padding=0)`.
- Dinov2 `LayerNorm(eps=1e-6)`, MHA projections `Linear(C -> C)` with bias, output `Linear(C -> C)`, MLP `Linear(C -> 4C)`, GELU, `Linear(4C -> C)`, layer-scale multiply, residual adds.
- Reassemble `Conv2d(C -> neck_i, kernel=1)`, `ConvTranspose2d(neck_i -> neck_i, kernel=stride=4 or 2)`, identity, and `Conv2d(neck_i -> neck_i, kernel=3, stride=2, padding=1)`.
- Neck `Conv2d(neck_i -> fusion, kernel=3, padding=1, bias=False)`.
- Fusion residual units: ReLU, `Conv2d(fusion -> fusion, kernel=3, padding=1)` twice, residual add.
- Prompt-depth layer: `Conv2d(1 -> fusion, 3x3)`, ReLU, `Conv2d(fusion -> fusion, 3x3)`, ReLU, `Conv2d(fusion -> fusion, 3x3)`.
- Bilinear `interpolate` with both `align_corners=False` for residual/prompt resize and `align_corners=True` for fusion/head upsample; postprocess uses bicubic `align_corners=False`.

Attention primitives:
- Noncausal dense self-attention only in Dinov2 backbone. No KV cache, RoPE, ALiBi, causal masks, sliding window, or cross-attention.

Position/custom math:
- Dinov2 learned absolute positional embeddings with bicubic interpolation in float32 when input patch grid differs from pretrained grid.

Preprocessing-coupled ops:
- Resize with aspect-ratio option and multiple-of-14 rounding, rescale by `1/255`, ImageNet normalize, optional center padding only if configured, prompt-depth meter scaling.

Postprocessing:
- Optional per-image resize of `[H,W]` depth to target size using bicubic interpolation and no NMS/thresholding.

## 5. Layer/block breakdown

Dinov2 backbone, repeated `L` times:

```text
x = LayerNorm(x)
q,k,v = Linear(C -> C, bias=True)(x), shaped [B,heads,S,64]
attn = softmax(q @ k^T * 1/sqrt(64))
x = residual + LayerScale(Linear(attn -> C))
y = LayerNorm(x)
y = Linear(C -> 4C) -> GELU -> Linear(4C -> C)
x = x + LayerScale(y)
```

PromptDA reassemble for four tapped states:

```text
tokens = hidden_state[:, 1:]
map = tokens.reshape(B, patch_h, patch_w, C).permute(0,3,1,2)
map = Conv1x1(C -> neck_i)(map)
map = resize_by_factor_i(map)
```

Fusion layer, applied from deepest to shallowest:

```text
if residual exists:
  residual = bilinear(residual, hidden_state.shape[2:], align_corners=False) if needed
  hidden_state = hidden_state + ResidualConvUnit(residual)
hidden_state = ResidualConvUnit(hidden_state)
if prompt_depth exists:
  p = bilinear(prompt_depth, hidden_state.shape[2:], align_corners=False)
  hidden_state = hidden_state + PromptDepthConv3x3Stack(p)
hidden_state = bilinear(hidden_state, next_size or scale_factor=2, align_corners=True)
hidden_state = Conv1x1(fusion -> fusion)(hidden_state)
```

Depth head:

```text
x = Conv3x3(fusion -> fusion/2)
x = bilinear(x, (patch_h * 14, patch_w * 14), align_corners=True)
x = Conv3x3(fusion/2 -> head_hidden_size)
x = ReLU(x)
x = Conv1x1(head_hidden_size -> 1)
x = Sigmoid(x) for metric checkpoints, or ReLU(x) for relative configs
predicted_depth = squeeze_channel(x)
```

## 6. Attention requirements

Attention is required only in the Dinov2 encoder branch:

- noncausal self-attention over `[CLS] + patch` tokens.
- MHA, not GQA/MQA. `num_attention_heads=6` or `16`; `head_dim=64`.
- No attention mask in the PromptDA forward path; `bool_masked_pos` is pretraining-only and not used by PromptDA examples.
- SDPA/Flash/Flex backend dispatch is supported by Dinov2 source, but eager semantics are dense `matmul -> add optional mask -> softmax(dim=-1) -> dropout -> matmul`.
- No KV cache or decode state. Backbone feature maps can be cached only as an encoder-output optimization if the same image is reused with different prompt-depth conditioning.

## 7. Position encoding and custom math

Dinov2 uses learned absolute position embeddings. If the patch grid differs, patch positions are reshaped to a square table, bicubic-interpolated in float32 to `(H//14,W//14)`, cast back, flattened, and concatenated with the CLS position.

```python
def dinov2_pos_embed(pos, height, width, patch_size):
    cls = pos[:, :1]
    patch = pos[:, 1:]
    n = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, n, n, -1).permute(0, 3, 1, 2)
    patch = interpolate(patch.float(), size=(height // patch_size, width // patch_size),
                        mode="bicubic", align_corners=False).to(pos.dtype)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
    return concat([cls, patch], dim=1)
```

Prompt conditioning math:

```python
depth_min = prompt_depth.reshape(B, -1).min(dim=1)
depth_max = prompt_depth.reshape(B, -1).max(dim=1)
p_norm = (prompt_depth - depth_min[:,None,None,None]) / (depth_max - depth_min)[:,None,None,None]
pred = pred_norm * (depth_max - depth_min)[:,None,None] + depth_min[:,None,None]
```

## 8. Preprocessing and input packing

CPU/data-pipeline owned work:
- Images are prepared without forced RGB conversion inside this processor path, then resized, rescaled, normalized, and optionally padded.
- HF processor config for open checkpoints uses bicubic resize to an aspect-preserving size near `756x756`, constrained to multiples of 14, no padding.
- Prompt depth is expected as 2D image-like input, scaled by `0.001`, converted to float32, and expanded to `[1,H,W]`. If constant, one pixel is nudged by `1e-6`.

GPU/runtime graph inputs:
- `pixel_values: [B,3,H,W]`, float, normalized, NCHW.
- Optional `prompt_depth: [B,1,h,w]`, float meters. Runtime can accept lower-resolution prompt depth because fusion layers interpolate it to each feature-map size.

Postprocessing:
- `post_process_depth_estimation` optionally resizes each predicted depth map to caller-provided `(height,width)` using bicubic interpolation, then returns `{"predicted_depth": depth}`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Dinov2 patch Conv2d -> Linear/GEMM

Source pattern: `Conv2d(3,C,kernel=14,stride=14,padding=0)` followed by `flatten(2).transpose(1,2)`.

Replacement: window flatten `[B,patch_h,patch_w,3*14*14] -> Linear(588 -> C) -> sequence`.

Preconditions: kernel equals stride, dilation 1, groups 1, no padding, `H,W` divisible by 14, NCHW flatten order preserved. Weight transform is `conv.weight.reshape(C, 3*14*14)`.

Failure cases: non-Dinov2 backbone, patch size tuple not `(14,14)`, layout pass changes spatial/channel order without matching weight transform.

### Rewrite: Reassemble token map layout elimination

Source pattern: `[B,S,C] -> reshape(B,ph,pw,C) -> permute NCHW -> Conv2d`.

Replacement: guarded NHWC/local channel-last conv region or direct token-to-im2col lowering.

Preconditions: token order exactly row-major patch order after dropping CLS; consumers remain inside controlled reassemble/neck region.

Failure cases: backbone returns already reshaped maps, `reshape_hidden_states=true`, or alternate backbone token ordering.

### Rewrite: Prompt-depth normalization as fused reduction + affine

Source pattern: flatten min/max, broadcast normalization, later broadcast denormalization.

Replacement: per-sample min/max reductions plus fused elementwise affine around prompt-depth branch and output.

Preconditions: prompt_depth present; denominator nonzero or processor guard reproduced.

Parity test: constant-depth prompt bypassing processor must reject or match the `1e-6` nudge policy explicitly.

### Rewrite: Bilinear upsample + Conv1x1

Source pattern: fusion layer ends with bilinear interpolate then `Conv2d(1x1)`.

Replacement: keep as separate initially; later consider fused resize-projection kernel for fixed fusion channel widths.

Preconditions: exact `align_corners=True`, scale/size known, NCHW axis semantics preserved.

## 10. Kernel fusion candidates

Highest priority:
- Dinov2 LayerNorm + QKV projection and dense noncausal attention for ViT feature extraction.
- Patch embedding Conv2d-as-GEMM for the 14x14 non-overlap stem.
- Neck/fusion Conv2d + ReLU residual units, because this family has many small spatial convs after the backbone.

Medium priority:
- Bilinear resize kernels with exact `align_corners` variants.
- Prompt-depth three-conv injection block, especially if prompt depth is common in target workloads.
- Head upsample + conv stack for output-resolution depth maps.

Lower priority:
- Postprocess bicubic resize; can live in CPU/data pipeline or a simple GPU helper at first.
- Training-only drop path/dropout; inference path uses identity/disabled dropout.

## 11. Runtime staging plan

1. Parse `PromptDepthAnythingConfig`, materialize Dinov2 defaults, and reject non-Dinov2 or `reshape_hidden_states=true` backbones for the first target.
2. Load weights and run processor-free random tensor parity for the PromptDA neck/head with synthetic Dinov2 feature taps.
3. Compose a separately validated Dinov2 backbone and verify `feature_maps` at configured `out_indices`.
4. End-to-end no-prompt inference parity for `vits-hf` on fixed `756x1008` or similar multiple-of-14 inputs.
5. Add prompt-depth input path: processor scaling, min/max normalization, prompt injection at each fusion layer, output denormalization.
6. Add postprocess resize parity.
7. Introduce local conv/layout and patch-conv rewrites behind guards.

Stubs: training labels/loss, hidden-state/attention optional returns, non-Dinov2 backbones, and unavailable vitb HF checkpoint.

## 12. Parity and validation plan

- Random op parity for Dinov2 positional interpolation at square and rectangular patch grids; fp32 tolerance `1e-5` to `1e-4`.
- Random neck/head parity with four synthetic feature maps for small and large configs; include prompt absent/present.
- Single Dinov2 layer parity with hidden size 384 and 1024, checking attention eager semantics.
- End-to-end checkpoint parity for `vits-hf` and `vitl-hf`: compare `predicted_depth` before postprocess on processor-produced tensors. Start fp32 with `atol=1e-4` to `1e-3`; relax for fused fp16 conv/attention.
- Processor parity: image resize dimensions, prompt depth scaling, constant-depth nudge, and target postprocess resize.
- Shape guards: reject non-multiple-of-14 `pixel_values` unless intentionally supporting floor patching behavior; reject prompt batch-size mismatch through input validation.

## 13. Performance probes

- Processor throughput: resize/normalize/prompt-depth preparation images/sec.
- Dinov2 backbone throughput by variant: small vs large, batch sweep, resolution sweep.
- Neck/head-only throughput with and without prompt depth.
- Bilinear resize cost split by `align_corners=False`, `align_corners=True`, and bicubic postprocess.
- End-to-end depth maps/sec for common resolutions after aspect-preserving resize.
- Memory probes for feature taps and fusion intermediates; large variant stores four `[B,S,1024]` feature maps before reassembly.
- Attention backend comparison: eager/SDPA/FlashAttention-compatible noncausal ViT attention.

## 14. Skip/defer list

- Training and `labels` loss path; source raises `NotImplementedError`.
- Gradient checkpointing, dropout, and stochastic depth as training behavior.
- Non-Dinov2 backbones through generic `AutoConfig`/`load_backbone`.
- Remote/original non-HF checkpoints except as conversion references.
- `bool_masked_pos` masked-image pretraining path.
- Hidden-state/attention materialization as required outputs; optional debug only.
- NHWC global translation; only guarded local layout regions should be attempted.

## 15. Final implementation checklist

- [ ] Parse PromptDepthAnything config and nested Dinov2 config with effective defaults.
- [ ] Add first-target admission guards: Dinov2 backbone, `reshape_hidden_states=false`, four `out_indices`, patch size 14.
- [ ] Load PromptDA and Dinov2 weights with correct Q/K/V split assumptions.
- [ ] Implement/compose Dinov2 patch embedding, absolute position interpolation, noncausal MHA, LayerNorm, GELU MLP, layer-scale residual blocks.
- [ ] Implement PromptDA reassemble layers including ConvTranspose2d and stride-2 conv.
- [ ] Implement fusion residual conv units and exact bilinear resize modes.
- [ ] Implement optional prompt-depth min/max normalization, prompt conv injection, and output denormalization.
- [ ] Implement metric/relative depth head activations and channel squeeze.
- [ ] Add processor/postprocess parity or define them as CPU pipeline boundaries.
- [ ] Add synthetic neck/head parity tests for small and large configs.
- [ ] Add end-to-end checkpoint parity for vits-hf and vitl-hf.
- [ ] Benchmark backbone, neck/head, prompt path, and postprocess separately.
