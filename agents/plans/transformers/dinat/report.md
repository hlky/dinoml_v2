# DiNAT Transformers Full Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: shi-labs/dinat-* official checkpoints
Config source: downloaded config.json and preprocessor_config.json snapshots in this folder
Source files inspected:
- transformers/src/transformers/models/dinat/configuration_dinat.py
- transformers/src/transformers/models/dinat/modeling_dinat.py
- transformers/src/transformers/models/vit/image_processing_vit.py
- transformers/src/transformers/models/auto/image_processing_auto.py
- transformers/src/transformers/backbone_utils.py
- transformers/docs/source/en/model_doc/dinat.md
- NATTEN operation docs: https://natten.org/operations/
Any missing files or assumptions: no gated/401 checkpoints encountered. NATTEN runtime source was not locally installed; operator ABI is inferred from Transformers call sites plus NATTEN public operation docs.
```

Snapshot files fetched under this directory:

- `shi-labs__dinat-tiny-in1k-224__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-mini-in1k-224__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-small-in1k-224__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-base-in1k-224__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-large-in22k-in1k-224__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-large-in22k-in1k-384__config.json`, `__preprocessor_config.json`
- `shi-labs__dinat-large-11x11-in22k-in1k-384__config.json`, `__preprocessor_config.json`

Primary runtime target for this report: image classification encoder parity, with backbone feature extraction as a required adjacent target. Training losses are deferred.

## 2. High-level architecture

DiNAT is a hierarchical vision encoder with convolutional patch embedding, four NHWC encoder stages, dilated 2-D neighborhood self-attention, MLP blocks, convolutional downsamplers between stages, final LayerNorm, global average pooling, and a linear image-classification head.

```text
image processor CPU path -> pixel_values [B,3,H,W]
-> two Conv2d stride-2 stem -> NHWC patch map [B,H/4,W/4,C0]
-> 4 encoder stages with DiNA blocks and 3 downsamplers
-> final NHWC map [B,H/32,W/32,C3]
-> LayerNorm(last dim) -> global average pool over spatial tokens -> classifier logits
```

Backbone mode stops at selected stage feature maps and returns NCHW maps after per-output LayerNorm. The neural body is owned by this family, except the neighborhood attention primitive is delegated to NATTEN. There is no autoregressive decode, KV cache, text tokenizer, or multimodal embedding stitch.

Independently stageable pieces:

- CPU/data pipeline: resize, rescale, normalize, NCHW tensor construction.
- Stem/downsamplers: static Conv2d + NHWC/NCHW layout transitions.
- Encoder block: LayerNorm + Q/K/V linears + NATTEN DiNA + MLP.
- Classification head: final norm, spatial average, dense classifier.
- Backbone ABI: selected feature maps, stage names, channels, NCHW output.

## 3. Important config dimensions

Source defaults from `DinatConfig`:

| Field | Default / behavior |
| --- | --- |
| `patch_size` | `4`; source rejects any other value |
| `num_channels` | `3` |
| `embed_dim` | `64` |
| `depths` | `(3,4,6,5)` |
| `num_heads` | `(2,4,8,16)` |
| `kernel_size` | `7` |
| `dilations` | default `[[1,8,1],[1,4,1,4],[1,2,1,2,1,2],[1,1,1,1,1]]` |
| `mlp_ratio` | `3.0` |
| `qkv_bias` | `True` |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-5` |
| `layer_scale_init_value` | `0.0`; if positive, two learned `[dim]` scales per block |
| `hidden_dropout_prob` / attention dropout | `0.0` by default; inference dropout is inactive |
| `drop_path_rate` | stochastic depth schedule; inference identity |
| cache support | not applicable |

Representative checkpoint sweep:

| Checkpoint | Input size from preprocessor | Stage dims | Depths | Heads | Head dim | MLP hidden | Kernel/dilations | Layer scale |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `dinat-mini-in1k-224` | 224 | 64/128/256/512 | 3/4/6/5 | 2/4/8/16 | 32 | 192/384/768/1536 | k7, default short stage3 | 0 |
| `dinat-tiny-in1k-224` | 224 | 64/128/256/512 | 3/4/18/5 | 2/4/8/16 | 32 | 192/384/768/1536 | k7 | 0 |
| `dinat-small-in1k-224` | 224 | 96/192/384/768 | 3/4/18/5 | 3/6/12/24 | 32 | 192/384/768/1536 | k7 | 0 |
| `dinat-base-in1k-224` | 224 | 128/256/512/1024 | 3/4/18/5 | 4/8/16/32 | 32 | 256/512/1024/2048 | k7 | `1e-5` |
| `dinat-large-in22k-in1k-224` | 224 | 192/384/768/1536 | 3/4/18/5 | 6/12/24/48 | 32 | 384/768/1536/3072 | k7 | 0 |
| `dinat-large-in22k-in1k-384` | 384 | 192/384/768/1536 | 3/4/18/5 | 6/12/24/48 | 32 | 384/768/1536/3072 | k7, larger dilations `[13,6,3]` | 0 |
| `dinat-large-11x11-in22k-in1k-384` | 384 | 192/384/768/1536 | 3/4/18/5 | 6/12/24/48 | 32 | 384/768/1536/3072 | k11 | 0 |

Config trap: downloaded `config.json` files contain `hidden_size: 512` even for small/base/large variants. The modeling source sizes all projections from per-stage `dim = embed_dim * 2**stage`, and `DinatModel.num_features` uses the same expression. DinoML should not size weights from JSON `hidden_size` for this family.

## 3a. Family variation traps

- `kernel_size` is either 7 or 11 in inspected checkpoints. This changes RPB shape `[heads, 2k-1, 2k-1]` and attention neighborhood size `k*k`.
- Dilation is per block. 384 checkpoints use larger dilations, so admission must read `config.dilations`, not derive from stage index.
- Stage widths come from `embed_dim`, not JSON `hidden_size`.
- Head dim is 32 for inspected official checkpoints, but source only requires `dim % num_heads == 0`.
- `mlp_ratio` changes: mini/tiny use 3.0, small/base/large use 2.0.
- `layer_scale_init_value > 0` is active for base and inserts two learned elementwise scales in every block.
- Patch size is hard rejected unless exactly 4.
- Image sizes are not stored in model config; they are in preprocessor configs as scalar `size: 224` or `384`.
- Source model internals are NHWC after the stem. Public inputs and backbone outputs are NCHW.
- Rectangular images are not a good first admission target. Source Q/K/V use `.view(B,H,W,heads,D).transpose(1,2)` before NATTEN and assume square-like spatial behavior in official use. Require `H == W` after preprocessing until parity is proven for rectangular shapes.
- NATTEN is mandatory. Transformers raises a backend requirement for `DinatModel`, `DinatForImageClassification`, and `DinatBackbone`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation against `num_channels`.
- `permute(0,2,3,1)` after stem Conv2d.
- NHWC LayerNorm over last dim.
- NHWC to NCHW permute for downsample Conv2d, then back to NHWC.
- `view`, `transpose(1,2)`, `permute(0,2,3,1,4)`, `contiguous`, spatial slicing after pad.
- `flatten(1,2)`, `transpose(1,2)`, `AdaptiveAvgPool1d(1)`, `torch.flatten(1)` for classifier pooling.
- Backbone path: NCHW hidden state -> NHWC flatten -> LayerNorm -> NCHW output.

Neural network primitives:

- Conv2d stem 1: `3 -> embed_dim/2`, kernel 3, stride 2, padding 1, bias true.
- Conv2d stem 2: `embed_dim/2 -> embed_dim`, kernel 3, stride 2, padding 1, bias true.
- Downsampler Conv2d per stages 0-2: `dim -> 2*dim`, kernel 3, stride 2, padding 1, bias false.
- Linear Q/K/V per block: `dim -> dim`, bias controlled by `qkv_bias` (`True` in official configs).
- Attention output Linear: `dim -> dim`, bias true.
- MLP Linear: `dim -> int(mlp_ratio*dim)` then GELU then `int(mlp_ratio*dim) -> dim`.
- Final classifier Linear: `last_dim -> num_labels` when `num_labels > 0`.

Attention primitives:

- Required custom sparse attention: 2-D noncausal dilated neighborhood attention.
- Score op: `natten2dqkrpb(q, k, rpb, kernel_size, dilation)`.
- Value op: `natten2dav(attention_probs, v, kernel_size, dilation)`.
- Softmax over neighborhood axis `dim=-1`.
- RPB parameter shape `[num_heads, 2*kernel_size-1, 2*kernel_size-1]`.

Preprocessing-coupled ops:

- Resize to square `224` or `384` with PIL bilinear (`resample: 3` in snapshots).
- Rescale is source-default true in `ViTImageProcessor`; old snapshots omit explicit `do_rescale`, so loader should apply processor defaults if using current processor class.
- Normalize RGB with ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- Emit `pixel_values` as `[B,3,H,W]`.

Not applicable: RoPE/ALiBi, KV cache, generation, tokenizer, packed sequence metadata, quantized packed weights, scatter stitch.

## 5. Layer/block breakdown

For input `pixel_values [B,3,S,S]`, official S is 224 or 384.

Stem:

```text
x = Conv2d(3 -> C0/2, k=3, s=2, p=1)(pixel_values)   # [B,C0/2,ceil(S/2),ceil(S/2)]
x = Conv2d(C0/2 -> C0, k=3, s=2, p=1)(x)             # [B,C0,ceil(S/4),ceil(S/4)]
x = permute NCHW -> NHWC                              # [B,S/4,S/4,C0]
x = LayerNorm(C0)(x)
```

DiNAT block, repeated by stage:

```text
shortcut = x                                           # [B,H,W,C]
y = LayerNorm(C)(x)
if H < kernel*dilation or W < kernel*dilation:
    y = pad right/bottom to at least kernel*dilation
q = Linear(C -> C, bias=qkv_bias)(y).view(B,Hpad,Wpad,heads,head_dim).transpose(1,2)
k = Linear(C -> C, bias=qkv_bias)(y).view(...).transpose(1,2)
v = Linear(C -> C, bias=qkv_bias)(y).view(...).transpose(1,2)
q = q / sqrt(head_dim)
scores = natten2dqkrpb(q, k, rpb, kernel_size, dilation)
p = softmax(scores, dim=-1)
ctx = natten2dav(p, v, kernel_size, dilation)
ctx = permute/view back to NHWC C
ctx = crop to original H,W if padded
ctx = Linear(C -> C)(ctx)
if layer_scale: ctx = scale_attn * ctx
x = shortcut + DropPath(ctx)
y = LayerNorm(C)(x)
y = Linear(C -> mlp_hidden)(y)
y = GELU(y)
y = Linear(mlp_hidden -> C)(y)
if layer_scale: y = scale_mlp * y
x = x + DropPath(y)
```

Downsampler after stages 0, 1, 2:

```text
y = permute NHWC -> NCHW
y = Conv2d(C -> 2C, k=3, s=2, p=1, bias=False)(y)
y = permute NCHW -> NHWC
y = LayerNorm(2C)(y)
```

Classifier:

```text
x = final LayerNorm(last_dim)(x)                       # [B,S/32,S/32,C3]
pooled = AdaptiveAvgPool1d(1)(x.flatten(1,2).transpose(1,2))
logits = Linear(C3 -> num_labels)(pooled.flatten(1))
```

## 6. Attention requirements

DiNAT uses encoder-only, noncausal, self-attention over 2-D image feature maps. There is no cross-attention and no KV cache.

| Property | Requirement |
| --- | --- |
| Pattern | 2-D neighborhood attention, optionally dilated |
| Query/key/value layout at source call | 5-D tensor derived from NHWC hidden states; official square inputs avoid H/W ambiguity |
| Heads | stage-specific MHA; inspected configs have Q heads = KV heads |
| Head dim | `dim / num_heads`, 32 in official snapshots |
| Q/K width | `dim`; V width `dim` |
| Masking | no attention mask; pad is physical zero padding only when a feature map is smaller than `kernel*dilation` |
| Local window | `kernel_size * kernel_size` sampled with `dilation` |
| Relative bias | learned table `[heads, 2k-1, 2k-1]` passed into score op |
| Backend | external NATTEN functions, not PyTorch SDPA |
| Attention outputs | if requested, returns dense-ish neighborhood probabilities, not full `[HW,HW]` dense attention |

NATTEN current docs describe `na2d` tensors as heads-last `[batch, X, Y, heads, head_dim]`, with `kernel_size` and `dilation` constrained by token layout dimensions. Transformers uses older functional names and pads only when feature maps are smaller than the requested receptive span. DinoML should implement a DiNAT-specific op contract first instead of lowering to generic dense attention.

## 7. Position encoding and custom math

There is no absolute position embedding, RoPE, or ALiBi. Spatial inductive bias comes from convolutional stem/downsamplers plus learnable relative positional bias inside NATTEN score generation.

Concise parity math:

```python
def dinat_attention(q, k, v, rpb, kernel_size, dilation):
    q = q / math.sqrt(q.shape[-1])
    scores = natten2dqkrpb(q, k, rpb, kernel_size, dilation)
    probs = softmax(scores, axis=-1)
    return natten2dav(probs, v, kernel_size, dilation)
```

Precomputable: RPB is a learned weight, no dynamic trigonometric tables. Dynamic: padding amount depends on feature map size and `kernel_size*dilation`.

## 8. Preprocessing and input packing

The model-coupled processor is mapped through ViT image processing. Snapshot preprocessor configs use old `feature_extractor_type: ViTFeatureExtractor`, but current auto mapping routes `dinat` to `ViTImageProcessor`.

CPU/data-pipeline contract:

- Decode/convert image to RGB outside the model graph.
- Resize to square `size` from checkpoint preprocessor, 224 or 384.
- Rescale pixel values if using current processor defaults.
- Normalize channelwise with ImageNet mean/std.
- Produce `pixel_values [B,3,S,S]` in NCHW.

GPU/runtime graph starts at `pixel_values`. There are no image masks, crop metadata, patch grids, placeholder tokens, token type IDs, or postprocessing beyond logits-to-label outside the model.

Backbone ABI:

- Default `out_features` is the last stage if not supplied.
- Valid stage names are `stem`, `stage1`, `stage2`, `stage3`, `stage4`.
- Feature maps are returned as NCHW after stage-specific LayerNorm.
- For S=224, feature map resolutions are approximately `stem 56`, `stage1 56`, `stage2 28`, `stage3 14`, `stage4 7`.
- For S=384, resolutions are `96`, `96`, `48`, `24`, `12`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: stem Conv2d pair as static convolution region

Source pattern:

```text
NCHW -> Conv3x3/s2/p1 -> Conv3x3/s2/p1 -> NHWC -> LayerNorm
```

Replacement pattern: keep as Conv2d kernels initially. Later, fuse NCHW input conversion, two convolutions, and NHWC output if the provider has an efficient channel-last convolution path.

Preconditions:

- `patch_size == 4`.
- `num_channels == 3` unless checkpoint proves otherwise.
- Kernel 3, stride 2, padding 1, dilation 1, groups 1.
- Preserve PyTorch Conv2d output-size formula `floor((in + 2p - k) / stride) + 1`.

Failure cases: arbitrary patch sizes, non-square admission without attention parity, grouped/depthwise conv not present in source.

Parity test sketch: compare stem output after two convs and LayerNorm for random `[1,3,224,224]` and `[2,3,384,384]`.

### Rewrite: downsampler layout fold

Source pattern:

```text
NHWC -> permute NCHW -> Conv2d(C,2C,k3,s2,p1,bias=False) -> permute NHWC -> LayerNorm(last)
```

Replacement: channel-last Conv2d or fused layout-aware Conv2d producing NHWC.

Preconditions:

- Region is local: only the downsampler consumes and produces the tensors.
- LayerNorm axis remains channel axis; rewrite `dim=-1` semantics, not `dim=1`.
- Weight layout remains PyTorch OIHW or is explicitly transformed for the provider.

Failure cases: backbone output requested before downsampling must preserve the pre-downsample NCHW feature map contract.

### Rewrite: independent Q/K/V linears to packed projection

Source pattern:

```text
q = Linear(C,C,bias=qkv_bias)
k = Linear(C,C,bias=qkv_bias)
v = Linear(C,C,bias=qkv_bias)
```

Replacement:

```text
qkv = Linear(C, 3*C)
split q,k,v in [q, k, v] order
```

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0) if bias else None
```

Preconditions: identical input tensor, no hooks requiring separate modules, split order `[Q,K,V]`, source bias setting equal for all three.

Failure cases: weight aliasing or quantized per-module metadata introduced later.

### Rewrite: DiNA fused attention op

Source pattern:

```text
Q/K/V reshape -> q scale -> natten2dqkrpb -> softmax -> dropout(inactive) -> natten2dav
```

Replacement: single fused `DinatNeighborhoodAttention2D` inference op.

Preconditions:

- Inference mode, dropout inactive.
- Noncausal self-attention, no additional keys/values.
- Q heads equal KV heads for current official checkpoints.
- `kernel_size`, `dilation`, and RPB shape match config.
- Square feature maps for first admission, or an exact rectangular parity proof.

Failure cases: `output_attentions=True` requires returning probabilities; training dropout/stochastic depth not supported in fused inference path.

### Rewrite: final global average pool simplification

Source pattern:

```text
x.flatten(1,2).transpose(1,2) -> AdaptiveAvgPool1d(1) -> flatten(1)
```

Replacement:

```text
ReduceMean over NHW spatial axes -> [B,C]
```

Preconditions: input is NHWC final map, pool output size is 1, no NaN-sensitive order constraints beyond normal tolerance.

Failure cases: if a future head changes pooling behavior.

## 10. Kernel fusion candidates

Highest priority:

- Fused 2-D dilated neighborhood attention with RPB: this is the defining runtime cost and cannot use dense SDPA efficiently.
- NHWC LayerNorm: appears before attention, before MLP, after stem, after downsamplers, and before pooling.
- QKV packed projection + reshape into heads-last NATTEN layout.

Medium priority:

- Downsampler Conv2d + layout conversion + LayerNorm.
- MLP Linear + GELU + Linear, especially for large stage3 with 18 blocks.
- Final spatial reduce mean + classifier.

Lower priority:

- Stem two-conv region fusion.
- Layer-scale multiply folded into residual path when enabled.
- Backbone output normalization/layout pack for detection/segmentation consumers.

## 11. Runtime staging plan

Stage 1: parse config and load weights. Compute stage dims from `embed_dim`; reject `patch_size != 4`; record `kernel_size`, per-layer dilations, layer-scale setting, and preprocessor size.

Stage 2: implement stem, downsampler, LayerNorm, MLP, pooling, classifier. Stub attention with a reference Python/NATTEN-compatible op if available.

Stage 3: implement source-faithful DiNA attention op for square inputs and no `output_attentions`.

Stage 4: full encoder parity for `DinatModel` and `DinatForImageClassification` on mini/tiny and one large 384 checkpoint.

Stage 5: implement `DinatBackbone` feature-map ABI, including `out_features/out_indices` validation and NCHW returned maps.

Stage 6: add layout/fusion rewrites: QKV pack, NHWC Conv2d regions, final reduce mean.

Stage 7: broaden admission after parity: kernel 11, larger 384 dilations, optional `output_attentions`, rectangular images only if exact source behavior is validated.

## 12. Parity and validation plan

No DinoML tests were run for this audit.

Recommended tests:

- Config-load tests for all snapshot configs: verify computed stage dims and reject trusting stale JSON `hidden_size`.
- Random tensor unit tests for `maybe_pad`: feature maps smaller than `kernel*dilation`, crop restores original H/W.
- Single block parity in fp32 with dropout/drop_path disabled for each stage width.
- DiNA op parity against NATTEN for kernel 7 and 11, dilations 1, 2, 3, 4, 6, 8, 13.
- Full encoder parity for `dinat-mini-in1k-224` and `dinat-large-in22k-in1k-384`.
- Classification logits parity on one real preprocessed image.
- Backbone parity for `out_features=["stage1","stage2","stage3","stage4"]`.

Tolerances: fp32 max/mean close to PyTorch reference (`rtol=1e-4`, `atol=1e-5`) for non-fused paths; fp16/bf16 fused attention should use looser layerwise tolerances and final-logit top-k agreement.

## 13. Performance probes

- Processor throughput for resize/rescale/normalize at 224 and 384.
- Stem/downsampler convolution throughput separately from attention.
- Encoder-only latency by stage; stage3 dominates for 18-layer variants.
- DiNA kernel sweep by `(S_stage, heads, kernel_size, dilation)`.
- Batch-size sweep for 224 and 384.
- Kernel 7 versus kernel 11 large checkpoint comparison.
- `output_attentions=False` fast path versus `True` materialization cost.
- NHWC native path versus explicit permute path memory bandwidth.
- Backbone multi-output extraction cost for detection/segmentation use.

## 14. Skip/defer list

- Training losses, stochastic depth randomness, dropout behavior.
- Rectangular/non-square image parity.
- `output_attentions=True` optimized path.
- Non-`patch_size=4` configs; source rejects them.
- Dense-attention fallback for DiNA except as a debug reference.
- Quantization and packed-weight loading.
- Multi-GPU/tensor parallel.
- Downstream detectors/segmenters that consume `DinatBackbone`; audit those wrapper families separately.

## 15. Final implementation checklist

- [ ] Parse `DinatConfig`; compute stage dims from `embed_dim`.
- [ ] Load downloaded/pretrained weights and ignore JSON `hidden_size` for projection sizing.
- [ ] Implement ViT-style image preprocessing contract or accept precomputed `[B,3,S,S]`.
- [ ] Implement stem Conv2d pair and NHWC transition.
- [ ] Implement NHWC LayerNorm.
- [ ] Implement per-block Q/K/V projections with Q/K/V split order.
- [ ] Implement 2-D dilated neighborhood attention with RPB.
- [ ] Implement padding/crop guards for small feature maps.
- [ ] Implement MLP GELU block and optional layer-scale multiply.
- [ ] Implement downsampler Conv2d NHWC/NCHW/NHWC path.
- [ ] Implement final LayerNorm, spatial mean pool, classifier.
- [ ] Implement `DinatBackbone` out feature selection and NCHW feature maps.
- [ ] Add QKV packing rewrite.
- [ ] Add guarded NHWC Conv2d/layout rewrite.
- [ ] Add final pooling reduce-mean rewrite.
- [ ] Add one-block, full-encoder, classification-logit, and backbone parity tests.
- [ ] Benchmark DiNA kernel sweep at 224/384 and kernel 7/11.
