# BEiT Transformers audit for DinoML v2

## 1. Source basis

Transformers commit/version: local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: representative official Hugging Face checkpoints under `microsoft/beit-*`.

Config source:

- `https://huggingface.co/microsoft/beit-base-patch16-224/raw/main/config.json`
- `https://huggingface.co/microsoft/beit-large-patch16-224/raw/main/config.json`
- `https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/raw/main/config.json`
- `https://huggingface.co/microsoft/beit-large-patch16-224-pt22k/raw/main/config.json`
- `https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640/raw/main/config.json`
- Matching `preprocessor_config.json` files for the base, large, pretraining, and ADE segmentation checkpoints.

Source files inspected:

- `X:/H/transformers/src/transformers/models/beit/configuration_beit.py`
- `X:/H/transformers/src/transformers/models/beit/modular_beit.py`
- `X:/H/transformers/src/transformers/models/beit/modeling_beit.py`
- `X:/H/transformers/src/transformers/models/beit/image_processing_beit.py`
- `X:/H/transformers/src/transformers/models/beit/image_processing_pil_beit.py`
- `X:/H/transformers/src/transformers/masking_utils.py` for `create_bidirectional_mask`

`modeling_beit.py` is generated from `modular_beit.py`; future source edits should target `modular_beit.py`. Runtime behavior in this report is confirmed from the generated `modeling_beit.py`.

Any missing files or assumptions: no remote-code files are required. This is inference-only. Primary runtime target for DinoML should be `BeitForImageClassification` / base encoder first; masked image modeling, backbone extraction, and semantic segmentation heads are stageable follow-ups.

## 2. High-level architecture

BEiT is a ViT-like bidirectional vision encoder:

```text
image preprocessing -> NCHW pixel_values -> Conv2d patch embedding -> CLS/mask/position embedding
  -> repeated transformer encoder blocks with relative position bias
  -> pool or per-patch features -> task head
```

Main stages:

- CPU/data pipeline: resize, optional center crop, rescale, normalize, optional segmentation label reduction.
- Encoder: non-overlapping Conv2d patch embedding, CLS token prepend, optional mask-token replacement, optional absolute position embedding, bidirectional transformer layers.
- Image classification: mean-pool patch tokens plus LayerNorm when `use_mean_pooling=True`, then Linear classifier.
- Masked image modeling: encoder without pooler, final LayerNorm, Linear hidden-to-8192 visual-token logits for patch tokens only.
- Semantic segmentation: collect four hidden states, remove CLS, reshape sequence back to NCHW feature maps, FPN neck, UPer/PPM decode head, optional FCN auxiliary head.
- Backbone: return selected hidden states either as sequence tensors or reshaped NCHW feature maps, optionally through FPN.

Independently validatable units: image processor output, patch embedding, one encoder block, full encoder hidden states, pooler/classifier, MIM logits, segmentation feature-map reshape/FPN/decode head.

## 3. Important config dimensions

| Field | Source default | Base/224 cls | Large/224 cls | Base pt22k MIM | Large pt22k MIM | Base ADE 640 seg |
|---|---:|---:|---:|---:|---:|---:|
| architecture | source/config | `BeitForImageClassification` | `BeitForImageClassification` | `BeitForMaskedImageModeling` | `BeitForMaskedImageModeling` | `BeitForSemanticSegmentation` |
| image_size | 224 | 224 | 224 | 224 | 224 | 640 |
| patch_size | 16 | 16 | 16 | 16 | 16 | 16 |
| patch grid | inferred | 14x14 | 14x14 | 14x14 | 14x14 | 40x40 |
| sequence length incl. CLS | inferred | 197 | 197 | 197 | 197 | 1601 |
| hidden_size | 768 | 768 | 1024 | 768 | 1024 | 768 |
| num_hidden_layers | 12 | 12 | 24 | 12 | 24 | 12 |
| num_attention_heads | 12 | 12 | 16 | 12 | 16 | 12 |
| head_dim | inferred | 64 | 64 | 64 | 64 | 64 |
| intermediate_size | 3072 | 3072 | 4096 | 3072 | 4096 | 3072 |
| hidden_act | `gelu` | `gelu` | `gelu` | `gelu` | `gelu` | `gelu` |
| layer_norm_eps | 1e-12 | 1e-12 | 1e-12 | 1e-12 | 1e-12 | 1e-12 |
| layer_scale_init_value | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| use_absolute_position_embeddings | false | false | false | false | false | false |
| use_relative_position_bias | false | true | true | false | false | true |
| use_shared_relative_position_bias | false | false | false | true | true | false |
| use_mask_token | false | false | false | true | true | false |
| use_mean_pooling | true | true | true | true | true | true |
| vocab_size | 8192 | 8192 | 8192 | 8192 | 8192 | 8192 |
| num_labels | config metadata | 1000 inferred from `id2label` | 1000 inferred from `id2label` | not used | not used | 150 inferred from `id2label` |
| segmentation indices | config/source | n/a | n/a | n/a | n/a | `[3, 5, 7, 11]` via legacy `segmentation_indices` |
| pool_scales | `(1,2,3,6)` | n/a | n/a | n/a | n/a | `[1,2,3,6]` |

Processor sweep:

| Checkpoint | Resize | Size | Center crop | Normalize | Mean/std |
|---|---|---:|---|---|---|
| base/224 cls | true | 224 | false | true | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` |
| large/224 cls | true | 224 | false | true | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` |
| base pt22k MIM | true | 224 | false | true | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` |
| base ADE seg | true | 640 | false | true | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` |

Config fields omitted by several checkpoint configs fall back to source defaults: `pool_scales`, `use_auxiliary_head`, `add_fpn`, `reshape_hidden_states`, `drop_path_rate`, `hidden_dropout_prob`, and `attention_probs_dropout_prob` when absent.

## 3a. Family variation traps

- Fine-tuned classification/segmentation checkpoints use per-layer relative position bias; pt22k MIM checkpoints use one shared relative position bias at encoder input.
- Relative bias shape depends on runtime patch grid. At higher resolution the bias table is bilinearly interpolated and re-indexed.
- MIM requires `use_mask_token=True` and `bool_masked_pos` shaped `[B, num_patches]`; classification does not.
- Q and V projections have bias, K projection is `bias=False`.
- Attention is encoder-style bidirectional MHA, no KV cache, no causal decode path, no RoPE.
- `head_dim` is read as `config.head_dim` if present, else `hidden_size // num_attention_heads`. A custom config could make `num_heads * head_dim != hidden_size`.
- Pooling changes output semantics: mean-pool patch tokens plus LayerNorm when `use_mean_pooling=True`; otherwise final CLS token.
- Segmentation uses legacy `segmentation_indices`; `BeitConfig.__post_init__` maps it to `out_indices`.
- Segmentation and backbone reshape sequence `[B, 1+Hpatch*Wpatch, C]` to NCHW `[B, C, Hpatch, Wpatch]`. This is an axis-sensitive no-layout-translation boundary unless a whole local region is converted.
- Image processor outputs NCHW `pixel_values`. NHWC optimization is possible around patch embedding and segmentation conv heads, but source semantics and public tensors are NCHW.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation, shape read for `[B, C, H, W]`.
- Conv2d patch projection: `Conv2d(num_channels -> hidden_size, kernel=patch_size, stride=patch_size, padding=0, groups=1)`.
- Flatten spatial dimensions from `[B, hidden, Hp, Wp]` to `[B, hidden, Hp*Wp]`.
- Transpose to sequence `[B, Hp*Wp, hidden]`.
- Expand/broadcast CLS token and optional mask token.
- Masked select/blend for MIM embeddings: `embeddings * (1 - mask) + mask_token * mask`.
- Concatenate CLS plus patch sequence on `dim=1`.
- Add position embeddings or relative attention bias.
- Reshape/view/transpose for QKV and attention outputs.
- Mean over patch-token axis for pooling.
- Segmentation: slice `[:, 1:]`, transpose, reshape to NCHW, concat along channel axis, interpolate, argmax postprocess.

Neural network primitives:

- Linear/GEMM for Q, K, V, output projection, MLP, classifier, and MIM head.
- LayerNorm over hidden dimension with eps `1e-12`.
- GELU MLP activation.
- Elementwise residual add, layer scale multiply by learned `[hidden]` parameters.
- Dropout and DropPath are identity in inference.
- Segmentation only: Conv2d, ConvTranspose2d, BatchNorm2d in eval mode, GELU/ReLU, MaxPool2d, AdaptiveAvgPool2d, bilinear interpolate.

Attention primitives:

- Bidirectional self-attention, MHA, no cache.
- Eager path: matmul QK^T, scale by `head_dim**-0.5`, add mask/relative bias, softmax with fp32 accumulation, cast back, matmul AV.
- Source marks `_supports_sdpa=True`, `_supports_flash_attn=False`.

Position/relative-bias ops:

- Optional absolute position embedding interpolation with bicubic mode.
- BEiT relative position bias table gather by generated index.
- Bilinear interpolation of relative bias table for changed patch grid.
- Optional shared relative position bias applied once in `BeitModel`.
- Optional per-layer relative position bias applied inside every `BeitLayer`.

Preprocessing-coupled ops:

- Resize with bicubic default, rescale and normalize, optional center crop.
- Segmentation labels optionally reduce background labels to 255 then decrement.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B, 3, H, W]
patch = Conv2d(3 -> hidden, kernel=16, stride=16)(pixel_values)
patch = flatten(2).transpose(1, 2)              # [B, Hp*Wp, hidden]
if bool_masked_pos: patch = where(mask, mask_token, patch)
x = cat([cls_token.expand(B,1,H), patch], dim=1)
if absolute_pos: x = x + interpolated_pos_embed
```

Base encoder block, repeated `num_hidden_layers` times:

```text
if per_layer_relative_bias:
    attention_mask = attention_mask + rel_bias([Hp, Wp])
residual = x
y = LayerNorm(x)
q = Linear(hidden -> heads*head_dim, bias=True)(y)
k = Linear(hidden -> heads*head_dim, bias=False)(y)
v = Linear(hidden -> heads*head_dim, bias=True)(y)
y = bidirectional_attention(q, k, v, mask_or_bias)
y = Linear(heads*head_dim -> hidden, bias=True)(y)
x = residual + layer_scale_1 * y
residual = x
y = LayerNorm(x)
y = Linear(hidden -> intermediate)(y)
y = GELU(y)
y = Linear(intermediate -> hidden)(y)
x = residual + layer_scale_2 * y
```

Pooling/head variants:

- `BeitModel`: final `LayerNorm` is identity when mean pooling is enabled, else LayerNorm over the full sequence; pooler returns `LayerNorm(mean(x[:,1:]))` or `x[:,0]`.
- `BeitForImageClassification`: Linear `hidden_size -> num_labels` on pooled output.
- `BeitForMaskedImageModeling`: final LayerNorm over sequence, Linear `hidden_size -> vocab_size`, applied to patch tokens `[:,1:]`; labels/loss can be deferred.
- `BeitForSemanticSegmentation`: selected hidden states become NCHW feature maps, FPN neck changes scales, UPer head produces `[B, num_labels, Hout, Wout]`; loss and postprocess can be deferred.
- `BeitBackbone`: returns selected feature maps and can skip FPN unless `add_fpn=True`.

## 6. Attention requirements

Required attention for base/classification:

- Type: bidirectional encoder self-attention.
- Heads: base `12 x 64`, large `16 x 64`.
- Q/K/V: separate Linear modules; Q and V biased, K biasless.
- Masking: `create_bidirectional_mask`; for unpadded full attention this may return `None`, otherwise a 4D additive mask `[B,1,Q,K]`.
- Relative bias: additive mask term shaped `[1, heads, seq, seq]`.
- Packed/varlen support: not required by BEiT source.
- Sliding/local attention: not used.
- KV cache: not applicable.
- FlashAttention: source disables flash attention. SDPA is supported only when the selected attention backend can consume the additive bias/mask correctly.

Fused attention parity detail: eager softmax explicitly computes in `torch.float32` and casts to query dtype after softmax. Preserve scaling before mask/bias addition and dropout after softmax; dropout is zero in inference.

## 7. Position encoding and custom math

Relative position index:

```python
def beit_relative_position_index(hp, wp):
    coords = meshgrid(arange(hp), arange(wp), indexing="ij")
    coords = stack(coords).flatten(1)             # [2, hp*wp]
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
    rel[:, :, 0] += hp - 1
    rel[:, :, 1] += wp - 1
    rel[:, :, 0] *= 2 * wp - 1
    num_relative_distance = (2 * hp - 1) * (2 * wp - 1) + 3
    idx = zeros([hp * wp + 1, hp * wp + 1])
    idx[1:, 1:] = rel.sum(-1)
    idx[0, :] = num_relative_distance - 3
    idx[:, 0] = num_relative_distance - 2
    idx[0, 0] = num_relative_distance - 1
    return idx
```

The stored table has `((2*Hp0-1)*(2*Wp0-1)+3, heads)` rows. For runtime grids different from config grid, source interpolates the non-CLS table entries from `[old_width, old_height]` to `[new_height, new_width]`, then appends the three CLS-specific rows and gathers by the generated index.

Absolute position embeddings are usually disabled in inspected official configs. If enabled, patch position embeddings are reshaped to a square grid, permuted to NCHW, bicubic-interpolated to `[H/patch, W/patch]`, permuted back to sequence, and concatenated with the CLS position.

Precompute candidates:

- Fixed-resolution relative position index and full bias tensor can be precomputed per checkpoint and resolution.
- Shared relative bias can be computed once per forward and reused across layers.
- Per-layer relative bias has separate parameters per layer; only indices are reusable.

## 8. Preprocessing and input packing

Image processor contract:

- Input images are converted/prepared by Transformers image processor, resized to checkpoint size, rescaled, and normalized.
- Official BEiT processors observed use bicubic resize, no center crop, mean/std `[0.5, 0.5, 0.5]`.
- Runtime tensor is `pixel_values` in NCHW `[B, 3, H, W]`, usually `float32`.
- For ADE segmentation, labels are optional CPU-side inputs. `do_reduce_labels` can map background class handling before loss; inference only needs `pixel_values`.

MIM input packing:

- `bool_masked_pos` is `[B, Hp*Wp]`.
- Masked patch embeddings are replaced before CLS concatenation.
- Output logits are `[B, Hp*Wp, 8192]`; CLS is excluded.

There are no text tokens, tokenizer, placeholder stitching, modality token IDs, cu_seqlens, or generation controller rules.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding to GEMM

Source pattern: `Conv2d(C -> hidden, kernel=patch, stride=patch, padding=0, dilation=1, groups=1)` followed by `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW image -> patch extract in PyTorch flatten order -> GEMM([B*Hp*Wp, C*Kh*Kw] x [C*Kh*Kw, hidden]) -> [B, Hp*Wp, hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Runtime height/width divisible by patch size.
- Weight transform: `W_gemm = conv.weight.reshape(hidden, C*Kh*Kw).T`; bias is conv bias if present. BEiT patch projection uses PyTorch Conv2d default bias.
- Preserve NCHW patch flatten order unless an NHWC layout pass also transforms weight/layout.

Failure cases: non-divisible image size, non-default convolution attrs, or downstream consumers expecting spatial NCHW before flatten.

Parity test sketch: compare patch embedding output for random `[B,3,224,224]` and `[B,3,640,640]` against PyTorch Conv2d path at fp32/fp16 tolerances.

### Rewrite: shared relative bias materialization

Source pattern: `shared_position_bias(window_size)` in `BeitModel`, then same additive bias passed to every layer.

Replacement: precompute/generate one `[1, heads, seq, seq]` tensor per runtime patch grid and add it as the attention mask input to all layers.

Preconditions: `use_shared_relative_position_bias=True`, fixed or bucketed image size, no user-provided incompatible 4D attention mask.

Failure cases: dynamic arbitrary resolution without bias interpolation support; per-layer relative bias configs.

### Rewrite: per-layer relative bias into attention backend bias

Source pattern: `relative_position_bias + attention_mask` before attention.

Replacement: pass relative bias as the additive attention bias to a fused MHA/SDPA kernel.

Preconditions: backend supports noncausal additive per-head bias and fp32 softmax parity.

Failure cases: backend only supports boolean masks, causal masks, or no per-head bias.

### Rewrite: sequence-to-NCHW reshape region for segmentation

Source pattern: `hidden[:,1:].transpose(1,2).reshape(B,C,Hp,Wp)` followed by NCHW conv/pool/interpolate head.

Replacement opportunity: guarded NHWC local region:

```text
[B, S, C] -> [B, Hp, Wp, C] -> NHWC conv/pool/interpolate kernels -> final NCHW or NHWC logits as required
```

Preconditions: all consumers inside the segmentation head are layout-translated together; axis attrs rewrite `dim=1` channel concat to `dim=-1`; BatchNorm2d semantics become channel-last batch norm; output contract either converted back to NCHW or explicitly changed internally only.

Failure cases: returning intermediate feature maps, external hooks on NCHW tensors, unsupported ConvTranspose2d/AdaptiveAvgPool2d NHWC kernels.

### Rewrite: inference BatchNorm2d folding

Source pattern: segmentation `Conv2d -> BatchNorm2d -> activation`.

Replacement: fold BN affine/running stats into Conv2d weights/bias in eval mode.

Preconditions: inference mode, frozen running mean/var, no training updates.

Failure cases: training, unfrozen BN, missing running stats.

## 10. Kernel fusion candidates

Highest priority:

- Conv patch embedding lowered to GEMM or optimized non-overlap patch kernel. It is the first heavy op and exposes clean preconditions.
- Encoder LayerNorm plus Q/K/V GEMMs. BEiT repeats this 12 or 24 times and has regular ViT shapes.
- Noncausal attention with additive relative bias and fp32 softmax. Relative bias support is the main difference from plain ViT attention.
- MLP `Linear -> GELU -> Linear`, with residual/layer-scale epilogue fusion where possible.

Medium priority:

- Shared/per-layer relative bias precompute and gather/interpolate kernels.
- Pooler mean over patch tokens plus LayerNorm.
- MIM head `LayerNorm -> Linear` for patch tokens, useful for pt22k checkpoints.
- Segmentation NCHW conv/BN/activation and bilinear interpolate if ADE is in scope.

Lower priority:

- Absolute position embedding interpolation, since official inspected BEiT configs disable it.
- DropPath/dropout, identity in inference.
- Training losses, label reduction, and postprocess argmax/resize.

## 11. Runtime staging plan

Stage 1: parse `BeitConfig`, load weights, and run image processor parity externally. Support NCHW `pixel_values` with static 224 and 640 shapes.

Stage 2: implement patch embedding rewrite or Conv2d fallback and validate `BeitModel` embeddings including CLS and optional mask token.

Stage 3: one-block encoder parity with LayerNorm, Q/K/V, relative bias, attention, MLP, residual, and layer scale.

Stage 4: full encoder plus classification head for base/large 224 checkpoints.

Stage 5: MIM path for pt22k checkpoints: mask token, shared relative bias, final patch logits. Loss can be stubbed.

Stage 6: backbone/segmentation path: hidden-state capture, sequence-to-feature-map reshape, FPN, UPer head. Start with inference logits; defer loss.

Stage 7: optimize layout and fusions: patch embedding GEMM, fused attention with relative bias, NHWC-local segmentation head where guarded.

## 12. Parity and validation plan

- Random tensor patch embedding parity: Conv2d path versus GEMM rewrite for `[1,3,224,224]`, `[4,3,224,224]`, and `[1,3,640,640]`.
- Relative position bias parity: generated index and gathered bias for 14x14 and 40x40 grids, including shared and per-layer cases.
- One encoder layer parity against Transformers with fixed random weights, fp32 tolerance around `1e-5` absolute/relative.
- Full `BeitModel` parity for base 224 and pt22k MIM configs; compare last hidden state and pooler output.
- Image classification logits parity for base and large checkpoints; fp32 tolerance `1e-4`, fp16/bf16 tolerance chosen per fused attention accumulation.
- MIM logits parity with deterministic `bool_masked_pos`.
- Segmentation feature-map reshape and FPN/decode-head parity for ADE 640, first fp32.
- Processor parity can stay in CPU/data pipeline: compare `pixel_values` statistics and shape against HF processor.

## 13. Performance probes

- Processor throughput: resize/rescale/normalize images/sec for 224 and 640.
- Patch embedding throughput: Conv2d versus GEMM rewrite across batch sizes.
- Encoder-only throughput: base 12-layer and large 24-layer at 197 tokens.
- High-resolution encoder throughput: ADE 1601-token sequence to expose attention quadratic cost.
- Attention backend comparison: eager GEMM/softmax/GEMM versus fused SDPA with additive relative bias.
- MLP GEMM throughput and residual/layer-scale fusion impact.
- Segmentation head throughput split into FPN ConvTranspose, UPer pooling/interpolate, and conv bottlenecks.
- Memory probes: attention score/bias tensor size for 197 and 1601 tokens; relative bias cache size by heads/grid.

## 14. Skip/defer list

- Training losses, gradient checkpointing, DropPath randomness, and dropout behavior.
- AutoModelForMaskedImageModeling compatibility; source explicitly requires `BeitForMaskedImageModeling` for BEiT MIM.
- Arbitrary dynamic image sizes before relative-bias and absolute-position interpolation are validated.
- FlashAttention-specific path; source declares no flash attention support.
- Segmentation auxiliary loss and label preprocessing for first inference path.
- NHWC global translation; only guarded local layout regions should be optimized.
- Multi-GPU, quantization, pruning, and export-specific tracing branches.

## 15. Final implementation checklist

- [ ] Parse `BeitConfig`, including legacy `segmentation_indices -> out_indices`.
- [ ] Load BEiT weights and identify task head by `architectures`.
- [ ] Implement NCHW image input contract and processor parity metadata.
- [ ] Implement Conv2d patch embedding or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement CLS token concat and optional mask-token blend.
- [ ] Implement optional absolute position interpolation.
- [ ] Implement BEiT relative position index, table interpolation, gather, and shared/per-layer bias paths.
- [ ] Implement bidirectional MHA with additive per-head bias and fp32 softmax.
- [ ] Implement BEiT block LayerNorm, Q/K/V/O projections, GELU MLP, residuals, and layer-scale multiplies.
- [ ] Implement mean-pool/CLS pooler variants and image classification head.
- [ ] Implement MIM final LayerNorm and patch-token logits head.
- [ ] Add backbone hidden-state extraction and sequence-to-NCHW feature-map reshape.
- [ ] Add segmentation FPN/UPer/FCN head operators if ADE support is targeted.
- [ ] Add parity tests for patch embedding, relative bias, one layer, full encoder, classifier logits, MIM logits, and segmentation logits.
- [ ] Benchmark patch embedding, attention with relative bias, full encoder, and segmentation head.
