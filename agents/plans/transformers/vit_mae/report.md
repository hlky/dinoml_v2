# Transformers Audit: vit_mae

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/vit-mae-base / facebook/vit-mae-large / facebook/vit-mae-huge, plus tiny random config
Config source: Hugging Face raw config.json and preprocessor_config.json, fetched 2026-05-13
Source files inspected:
  X:/H/transformers/src/transformers/models/vit_mae/configuration_vit_mae.py
  X:/H/transformers/src/transformers/models/vit_mae/modular_vit_mae.py
  X:/H/transformers/src/transformers/models/vit_mae/modeling_vit_mae.py
  X:/H/transformers/src/transformers/models/vit/modeling_vit.py
  X:/H/transformers/src/transformers/models/vit/image_processing_vit.py
  X:/H/transformers/src/transformers/models/vit/image_processing_pil_vit.py
  X:/H/transformers/src/transformers/models/auto/image_processing_auto.py
  X:/H/transformers/src/transformers/image_processing_backends.py
  X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions: no family-local image processor exists; AutoImageProcessor maps vit_mae to ViTImageProcessor / ViTImageProcessorPil. modeling_vit_mae.py is generated from modular_vit_mae.py; generated file is runtime source, modular file is authoritative for future source edits.
```

Primary runtime target for this report: `ViTMAEForPreTraining` image masked-autoencoding inference/reconstruction, including encoder, decoder, masking/restoration, logits, and optional loss math. Encoder-only `ViTMAEModel` is a clean stageable subset for image feature extraction.

Config URLs inspected:

- `https://huggingface.co/facebook/vit-mae-base/raw/main/config.json`
- `https://huggingface.co/facebook/vit-mae-large/raw/main/config.json`
- `https://huggingface.co/facebook/vit-mae-huge/raw/main/config.json`
- `https://huggingface.co/hf-tiny-model-private/tiny-random-ViTMAEForPreTraining/raw/main/config.json`
- `https://huggingface.co/facebook/vit-mae-base/raw/main/preprocessor_config.json`
- `https://huggingface.co/facebook/vit-mae-large/raw/main/preprocessor_config.json`
- `https://huggingface.co/facebook/vit-mae-huge/raw/main/preprocessor_config.json`

No 401/403 gated links were encountered for the configs above. The tiny random config was accessible despite the `hf-tiny-model-private` namespace name.

## 2. High-level architecture

ViT-MAE is an image masked autoencoder:

```text
image preprocessing -> NCHW pixel_values -> Conv2d patch embedding -> fixed 2D sin/cos pos add
-> per-sample random argsort mask/gather -> encoder ViT blocks over kept patches + CLS
-> decoder projection -> append learned mask tokens -> ids_restore gather -> decoder ViT blocks
-> patch pixel prediction -> optional patchify + masked MSE loss
```

Stage decomposition:

- CPU/data-pipeline: image load/convert, resize to 224, rescale, ImageNet normalize, output NCHW `pixel_values`.
- Encoder tokenization: non-overlap `Conv2d(C -> hidden, kernel=patch, stride=patch)`, flatten spatial grid row-major, add fixed position table without CLS, random masking.
- Encoder body: noncausal ViT self-attention blocks over `len_keep + 1` tokens.
- Decoder stitch: project encoder hidden width to decoder width, append repeated learned mask tokens, restore full patch order with `ids_restore`, add decoder position table.
- Decoder body and head: noncausal ViT self-attention blocks over all patch tokens + CLS, final linear predicts one flattened patch per position.
- Optional loss/postprocessing: `patchify(pixel_values)`, optional per-patch normalization, squared error averaged over patch dimension and masked positions. `unpatchify` is helper/postprocessing for visualization, not called by `forward`.

Independently validatable subsets: patch embedding + masking, one encoder block, full encoder, decoder restoration without blocks, full decoder logits, optional loss, and `unpatchify(logits)` visualization path.

## 3. Important config dimensions

Source-default config dimensions from `ViTMAEConfig`:

| Field | Default | Notes |
|---|---:|---|
| `hidden_size` | 768 | encoder width |
| `num_hidden_layers` | 12 | encoder block count |
| `num_attention_heads` | 12 | encoder MHA heads |
| `head_dim` | inferred 64 | source uses `getattr(config, "head_dim", hidden_size // heads)` |
| `intermediate_size` | 3072 | encoder MLP |
| `hidden_act` | `gelu` | ACT2FN lookup |
| `hidden_dropout_prob` | 0.0 | inactive for eval |
| `attention_probs_dropout_prob` | 0.0 | inactive for eval |
| `layer_norm_eps` | 1e-12 | pre-norm and final norms |
| `image_size` | 224 | int or pair accepted by patch embedding |
| `patch_size` | 16 | int or pair accepted in patch embedding, but some helper math assumes scalar/square |
| `num_channels` | 3 | NCHW channel dim |
| `qkv_bias` | true | separate Q/K/V projection biases |
| `decoder_hidden_size` | 512 | decoder width |
| `decoder_num_hidden_layers` | 8 | decoder block count |
| `decoder_num_attention_heads` | 16 | decoder MHA heads |
| `decoder_intermediate_size` | 2048 | decoder MLP |
| `mask_ratio` | 0.75 | `len_keep = int(seq_len * (1 - ratio))` |
| `norm_pix_loss` | false | optional pre-loss target normalization |

Representative checkpoint sweep, source/config-derived:

| Model | Encoder layers | Hidden | Heads x dim | MLP | Image/patch | Patches | Keep tokens | Encoder seq incl. CLS | Decoder |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| `facebook/vit-mae-base` | 12 | 768 | 12 x 64 | 3072 | 224 / 16 | 196 | 49 | 50 | 8 layers, 512 hidden, 16 x 32, MLP 2048 |
| `facebook/vit-mae-large` | 24 | 1024 | 16 x 64 | 4096 | 224 / 16 | 196 | 49 | 50 | same decoder size as base |
| `facebook/vit-mae-huge` | 32 | 1280 | 16 x 80 | 5120 | 224 / 14 | 256 | 64 | 65 | same decoder size as base, pred width 588 |
| `hf-tiny-model-private/tiny-random-ViTMAEForPreTraining` | 5 | 32 | 4 x 8 | 37 | 30 / 2 | 225 | 90 | 91 | 8 layers, 512 hidden, 16 x 32, MLP 2048 |

Preprocessor config for base/large/huge: resize enabled, size 224, bilinear resample id 2, rescale enabled, normalize enabled, ImageNet mean `[0.485, 0.456, 0.406]`, ImageNet std `[0.229, 0.224, 0.225]`.

## 3a. Family variation traps

- Encoder and decoder dimensions differ. The decoder always starts with `Linear(hidden_size -> decoder_hidden_size)`, then uses a copied config with decoder-specific hidden/layer/head/intermediate fields.
- `facebook/vit-mae-huge` uses `patch_size=14`, so reconstruction width is `14 * 14 * 3 = 588`; base/large use width 768.
- `head_dim` can be explicitly present even though representative configs omit it. Source projection width is `num_attention_heads * head_dim`, not blindly `hidden_size`.
- `patch_size` and `image_size` accept iterable values in patch embedding, but `patchify`, decoder predictor, and interpolation paths use scalar-style `config.patch_size`; first DinoML admission should require square scalar patch sizes.
- `mask_ratio` changes sequence length through `int(num_patches * (1 - mask_ratio))`. Non-round ratios floor, and `mask.sum()` can become zero for `mask_ratio=0`.
- Runtime masking is stochastic unless caller supplies `noise`. Production parity needs supplied deterministic `noise` or a DinoML-owned RNG contract.
- Encoder attention runs over kept tokens only; decoder attention runs over full restored patch sequence plus CLS. Do not benchmark encoder with full patch length for MAE pretraining parity.
- Position embeddings are fixed parameters initialized from 2D sin/cos, with a deliberate half-rotation to match an original MAE naming bug. Treat loaded position tables as weights; only reproduce initializer for random-weight tests.
- `interpolate_pos_encoding=True` introduces bicubic interpolation and dynamic image shape behavior; default path rejects mismatched H/W.
- `ViTMAEModel` has no pooler. `ViTMAEForPreTraining` has no classification head.
- Image processor normalizes to channels-first tensors. NHWC is only a guarded local optimization around processor/patch-embedding or patchify/unpatchify regions.
- Axis-sensitive no-layout-translation guards: model input `pixel_values[:, C, H, W]`, patch `Conv2d`, `flatten(2)`, `transpose(1,2)`, patchify reshape/permute, position interpolation permutes, and loss reductions over patch-feature dim `-1`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape guards for `pixel_values` rank 4 NCHW, channel equality, fixed/default H/W equality, divisibility by patch size when patchifying.
- `Conv2d` with `kernel_size=stride=patch`, `padding=0`, `groups=1`, `dilation=1`.
- `flatten(2)`, `transpose(1, 2)`, `reshape`, `view`, `permute`, `contiguous`, `cat`, `expand`, `repeat`, `gather`.
- `argsort` twice over `[B, num_patches]` noise for mask/restore ids.
- Scalar/fill ops: `ones`, slice assignment equivalent for mask construction, zero/one mask values.

Neural network primitives:

- Encoder patch projection: base `Conv2d(3 -> 768, 16x16 stride 16)`, large `Conv2d(3 -> 1024, 16x16 stride 16)`, huge `Conv2d(3 -> 1280, 14x14 stride 14)`.
- LayerNorm over last dim with eps `1e-12`.
- Linear Q/K/V/O projections with optional Q/K/V bias and always-biased output projection.
- GELU MLP: `Linear(H -> 4H-ish) -> gelu -> Linear(intermediate -> H)`.
- Residual adds and eval-mode dropout identity.
- Decoder embed `Linear(encoder_hidden -> decoder_hidden)`, decoder prediction `Linear(decoder_hidden -> patch_size^2 * channels)`.

Attention primitives:

- Noncausal encoder and decoder self-attention, MHA only.
- Q/K/V shaped `[B, heads, seq, head_dim]`.
- Eager math: `Q @ K^T * head_dim^-0.5`, additive mask if present, softmax in fp32 then cast back, `P @ V`, transpose/contiguous, output projection.
- Backend compatibility flags advertise SDPA, FlashAttention, and FlexAttention, but eager math is sufficient for parity.

Position/custom math:

- Fixed 2D sin/cos position table builder with `embed_dim % 4 == 0`.
- Optional bicubic interpolation for encoder position grid and decoder 1D-like patch positions.

Preprocessing-coupled ops:

- Resize/rescale/normalize in image processor, output `pixel_values`.
- Patchify target for loss and `unpatchify` helper.

Scatter/indexed update ops:

- No boolean scatter. Restoration is `gather` with `ids_restore` after concatenating visible tokens and repeated mask tokens.

Generation/cache ops:

- None. No autoregressive decode, no KV cache.

## 5. Layer/block breakdown

Patch embedding and masking:

```text
pixel_values: [B, C, H, W]
patch = Conv2d(C -> Henc, kernel=P, stride=P)(pixel_values)  # [B, Henc, Gh, Gw]
tokens = flatten_spatial(patch).transpose(1, 2)              # [B, L, Henc]
tokens = tokens + pos_embed[:, 1:, :]
ids_shuffle = argsort(noise, dim=1)
ids_restore = argsort(ids_shuffle, dim=1)
ids_keep = ids_shuffle[:, :int(L * (1 - mask_ratio))]
visible = gather(tokens, dim=1, ids_keep repeated over hidden)
mask = gather([0 for kept prefix, 1 otherwise], dim=1, ids_restore)
hidden = cat(cls_token + pos_embed[:, :1, :], visible, dim=1)
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
x = LayerNorm(x)
q = Linear(Henc -> heads * head_dim, bias=qkv_bias)(x)
k = Linear(Henc -> heads * head_dim, bias=qkv_bias)(x)
v = Linear(Henc -> heads * head_dim, bias=qkv_bias)(x)
x = noncausal_attention(q, k, v, optional bidirectional mask)
x = Linear(heads * head_dim -> Henc, bias=True)(x)
x = x + residual
residual = x
x = LayerNorm(x)
x = Linear(Henc -> intermediate)(x)
x = GELU(x)
x = Linear(intermediate -> Henc)(x)
x = x + residual
```

Final encoder norm:

```text
latent = LayerNorm(hidden)  # [B, len_keep + 1, Henc]
```

Decoder:

```text
x = Linear(Henc -> Hdec, bias=True)(latent)
mask_tokens = repeat(mask_token, [B, ids_restore_len + 1 - x_len, Hdec])
patch_stream = cat(x[:, 1:, :], mask_tokens, dim=1)
patch_stream = gather(patch_stream, dim=1, ids_restore repeated over Hdec)
x = cat(x[:, :1, :], patch_stream, dim=1)
x = x + decoder_pos_embed
for 8 decoder blocks: same ViT block at Hdec / decoder heads / decoder MLP
x = LayerNorm(x)
logits = Linear(Hdec -> P * P * C, bias=True)(x)
logits = logits[:, 1:, :]
```

Loss path:

```text
target = patchify(pixel_values)  # [B, L, P*P*C]
if norm_pix_loss:
    target = (target - mean(target, -1)) / sqrt(var(target, -1) + 1e-6)
loss = ((logits - target) ** 2).mean(dim=-1)
loss = (loss * mask).sum() / mask.sum()
```

## 6. Attention requirements

ViT-MAE uses only encoder-style bidirectional self-attention.

| Attribute | Requirement |
|---|---|
| Causal | no |
| Cross-attention | no |
| MHA/MQA/GQA | MHA; no GQA/MQA |
| Encoder heads | base 12, large 16, huge 16 |
| Encoder head dim | base/large 64, huge 80 |
| Decoder heads | 16 |
| Decoder head dim | 32 for representative checkpoints |
| Q/K/V width | `num_attention_heads * head_dim` |
| Value width | same as Q/K |
| Masking | bidirectional padding/custom mask support through `create_bidirectional_mask`; normally skipped/None when no padding/custom mask |
| Packed/varlen | not model-specific |
| Sliding/local | none |
| RoPE/ALiBi/relative bias | none |
| KV cache | none |
| Flash/SDPA | source can dispatch through Transformers attention interface; parity can start with eager dense attention |

Source math order for eager parity: project Q/K/V, view to `[B, seq, heads, head_dim]`, transpose to `[B, heads, seq, head_dim]`, multiply scores by `head_dim^-0.5`, add mask if present, softmax with fp32 accumulation, cast probabilities to query dtype, dropout in training only, multiply by V, transpose back, contiguous reshape, output projection.

## 7. Position encoding and custom math

Position embeddings are learned parameters with `requires_grad=False`, initialized from fixed 2D sin/cos tables. For loaded pretrained checkpoints, DinoML should load these tensors directly. For initializer parity:

```python
def vit_mae_2d_sincos(height, width, dim, temperature=10000.0, cls=True):
    assert dim % 4 == 0
    omega = 1.0 / temperature ** (arange(dim // 4, float64) / (dim // 4))
    grid_h, grid_w = meshgrid(arange(height), arange(width), indexing="ij")
    emb_h = flatten(grid_h).outer(omega)
    emb_w = flatten(grid_w).outer(omega)
    pos = cat([sin(emb_h), cos(emb_h), sin(emb_w), cos(emb_w)], dim=1)
    if cls:
        pos = cat([zeros([1, dim]), pos], dim=0)
    half = dim // 2
    return cat([pos[..., half:], pos[..., :half]], dim=-1)
```

The half-rotation is source-required for MAE pretrained weight layout. Encoder interpolation reshapes patch positions to `[1, sqrt(L), sqrt(L), dim]`, permutes to NCHW, bicubic interpolates to `[H/P, W/P]`, then returns to `[1, new_L, dim]`. Decoder interpolation treats patch positions as `[1, 1, L, dim]`, permutes to NCHW, bicubic-interpolates the last spatial extent to the restored token count, and reshapes back.

Dynamic dependencies: interpolation depends on runtime input H/W or restored token length. Fixed default-resolution inference can treat both encoder and decoder position embeddings as constants.

## 8. Preprocessing and input packing

Auto image processing for `vit_mae` maps to ViT image processors. The official base/large/huge preprocessors resize to 224, rescale, normalize with ImageNet stats, and return `pixel_values`. The Torchvision backend converts PIL/NumPy to tensors, converts channels-last inputs to channels-first, and performs resize/rescale/normalize in channels-first form. The PIL backend also emits channels-first by using image transforms with `data_format=ChannelDimension.FIRST`.

Runtime tensor contract:

```text
pixel_values: float tensor [B, 3, H, W], normally [B, 3, 224, 224]
noise: optional float tensor [B, num_patches], lower values are kept
attention_mask: optional 2D/4D attention mask accepted by shared masking utility
```

For first DinoML integration, own processor-to-model ABI as NCHW. NHWC can be an optimized internal region only if either the processor is controlled to emit NHWC and the patch embedding rewrite consumes NHWC, or an explicit layout conversion is inserted before the model boundary.

Patch packing order in `patchify`:

```text
[B, C, H, W]
-> reshape [B, C, Gh, P, Gw, P]
-> permute [B, Gh, Gw, P, P, C]
-> reshape [B, Gh*Gw, P*P*C]
```

This row-major patch order must match decoder logits and position table order.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> GEMM

Source pattern: `Conv2d(C -> H, kernel=P, stride=P, padding=0)` followed by `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW WindowFlatten row-major patches [B, Gh*Gw, C*P*P]
-> GEMM weight_flat.T + bias
-> [B, Gh*Gw, hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- input rank 4 NCHW with `C == config.num_channels`
- `H % P == 0` and `W % P == 0`
- flatten order must match PyTorch Conv2d over NCHW windows

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias
```

Layout constraints: source semantics are NCHW. An NHWC version needs a different window flatten and weight permutation. Failure cases: non-square patch helpers, interpolated dynamic sizes without divisibility guard, grouped/dilated/padded conv.

Parity test: compare source Conv2d+flatten+transpose against WindowFlatten+GEMM for base/huge patch sizes and random fp32/fp16 inputs.

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern: three independent `Linear(H -> heads * head_dim, bias=qkv_bias)`.

Replacement:

```text
packed = Linear(H -> 3 * heads * head_dim)
q, k, v = split(packed, order=[q, k, v], dim=-1)
```

Preconditions: same input tensor, same dtype/device, no per-projection side effects, config/source projection widths equal. Weight transform concatenates rows `[q_proj.weight; k_proj.weight; v_proj.weight]` and biases in the same `[q, k, v]` order.

Failure cases: explicit `head_dim` causing width not equal to `hidden_size` is allowed but must use actual projection width; missing QKV bias means no packed bias.

### Rewrite: fixed eval dropout removal

Source pattern: `Dropout(p)` after attention/MLP and attention probs. Representative configs use `p=0`; eval mode makes dropout identity.

Replacement: remove dropout nodes in inference graphs.

Preconditions: `model.training == False`. Failure case: training/pretraining optimization is out of scope.

### Rewrite: decoder mask-token restore as indexed row copy

Source pattern: repeat mask token, cat visible decoded patch tokens + mask tokens, gather by `ids_restore`.

Replacement:

```text
full = allocate [B, L, Hdec]
full[:, ids_restore positions] = rows from visible_or_mask_stream
```

Preconditions: `ids_restore` is a per-batch permutation produced by argsort(argsort(noise)); `ids_restore` values in `[0, L)`, no duplicates. Safer initial lowering can keep `gather`.

Failure cases: externally supplied malformed `ids_restore` if decoder is exposed standalone.

### Rewrite: patchify/unpatchify as metadata-aware reshape/transpose

Source pattern: reshape/permute/reshape only.

Replacement: generated shape-only view plus layout-aware copy if consumer requires contiguous flat patches.

Preconditions: NCHW contiguous input, square scalar patch, H/W divisible by patch. Failure cases: channel-last hidden layouts unless axes are rewritten together.

### Layout rewrite: local NHWC patch region

Candidate region: processor output -> patch embedding -> token sequence.

Preconditions:

- Whole region controlled from image tensor through patch-window flatten.
- Rewrite all axes: NCHW `[B,C,H,W]` to NHWC `[B,H,W,C]`; `flatten(2)` and patchify axes must be replaced, not reused.
- Consumers after patch embedding are sequence `[B,L,H]`, layout-neutral.

Failure cases requiring no-layout-translation guard: public `pixel_values` ABI, source `Conv2d`, `patchify` loss, `unpatchify`, position interpolation permutes, any direct comparison to PyTorch intermediate tensors.

## 10. Kernel fusion candidates

Highest priority:

- Patch embedding as WindowFlatten + GEMM: first operator bottleneck and easiest way to avoid generic Conv2d surface.
- LayerNorm + QKV GEMM packing: repeated in every encoder and decoder block.
- Dense noncausal attention for small/medium vision sequences: encoder MAE seq is short after masking, decoder seq is full but still 197/257 tokens for representative configs.
- GELU MLP fusion: Linear -> GELU -> Linear dominates block FLOPs.

Medium priority:

- Decoder restore gather/cat/mask-token fusion: removes small but awkward tensor traffic and makes reconstruction path cleaner.
- Position add + CLS prepend/cat fusion for encoder and decoder.
- Patchify + masked MSE loss fusion for pretraining loss parity.
- Final decoder prediction GEMM + logits slice removing CLS.

Lower priority:

- Bicubic position interpolation kernel. Default inference at checkpoint resolution avoids it.
- `unpatchify` visualization path.
- Attention mask materialization; normal no-padding path can skip mask.

## 11. Runtime staging plan

Stage 1: Parse `ViTMAEConfig`, load base checkpoint weights, and admit fixed 224x224 NCHW input with supplied deterministic `noise`.

Stage 2: Implement patch embedding + fixed position add + random masking parity. Initially require scalar square patch size and static image size.

Stage 3: Run one encoder `ViTMAELayer` parity, then full `ViTMAEModel` encoder parity over kept tokens. Use eager noncausal attention first.

Stage 4: Add decoder embed, mask-token append, `ids_restore` gather, fixed decoder pos add, one decoder block parity, then full decoder logits.

Stage 5: Add `patchify` and optional masked MSE loss. `norm_pix_loss=False` can land first because official configs use it.

Stage 6: Add graph rewrites: patch Conv2d to GEMM, packed QKV, eval dropout removal, GELU fusion, attention backend selection.

Stage 7: Broaden admissions: huge patch size 14, tiny/random debug configs, explicit `head_dim`, `norm_pix_loss=True`, optional `interpolate_pos_encoding=True`.

Can stub initially: output hidden states/attentions capture, training dropout, gradient checkpointing, `unpatchify`, position interpolation, external attention masks.

## 12. Parity and validation plan

- Config parser tests for base/large/huge/tiny sweep, including derived `num_patches`, `len_keep`, encoder/decoder sequence lengths, and prediction width.
- Patch embedding rewrite random tests: Conv2d path vs WindowFlatten+GEMM for patch 16 and 14.
- Position table initializer test against source helper for small grids; include the MAE half-rotation.
- Random masking deterministic tests with supplied `noise`: verify `ids_shuffle`, `ids_restore`, kept tokens, mask values, and `mask.sum`.
- Single encoder block parity in fp32 with fixed random weights and no dropout.
- Full encoder parity for `facebook/vit-mae-base` on one processed image and supplied `noise`.
- Decoder restore parity: compare hidden sequence immediately after `ids_restore` gather and position add.
- Full pretraining logits parity: source vs DinoML `logits [B, L, P*P*C]`.
- Loss parity with `norm_pix_loss=False`, then `True`.
- Tolerances: fp32 absolute/relative around `1e-4` for full graph, tighter for individual layout ops; fp16/bf16 should use relaxed tolerances around `1e-2` depending on attention/LayerNorm accumulation.

## 13. Performance probes

- Preprocessor throughput: PIL backend vs Torchvision backend, batch-size sweep.
- Patch embedding: Conv2d vs WindowFlatten+GEMM, patch 16/base and patch 14/huge.
- Encoder-only throughput with MAE kept-token sequence lengths 50/65 and batch sweep.
- Decoder-only throughput with full sequence lengths 197/257 and batch sweep.
- End-to-end pretraining reconstruction throughput split into preprocessing, patch/mask, encoder, decoder, loss.
- Attention backend comparison: eager dense vs SDPA/Flash-compatible noncausal attention at seq lengths 50, 65, 197, 257.
- Gather/restoration overhead probe for decoder stitch.
- MLP/GELU fusion impact for base/large/huge hidden sizes.
- Memory probe: temporary footprint for attention probabilities and decoder full sequence.

## 14. Skip/defer list

- Training and gradients.
- Dropout randomness and gradient checkpointing.
- Output hidden-states/attentions capture beyond debug parity.
- Classification/segmentation heads; Transformers source does not implement `ViTMAEForImageClassification`.
- Autoregressive generation, beam search, KV cache.
- Multi-GPU/tensor parallel.
- Quantized/packed weights.
- Dynamic/interpolated position encoding for first fixed-resolution pass.
- Non-square/tuple patch sizes in helper paths until admission guards and axis tests exist.
- General NHWC model translation; only local guarded patch-region rewrites.

## 15. Final implementation checklist

- [ ] Parse `ViTMAEConfig` and derive patch/grid/sequence dimensions.
- [ ] Load `ViTMAEForPreTraining` weights, preserving fixed position tables and mask token.
- [ ] Implement NCHW image input contract and processor snapshot metadata.
- [ ] Implement patch embedding Conv2d or guarded WindowFlatten+GEMM rewrite.
- [ ] Implement fixed position add, CLS prepend, and deterministic-noise masking.
- [ ] Implement `argsort`/`gather` masking and `ids_restore` ABI.
- [ ] Implement ViT encoder block: LayerNorm, Q/K/V/O Linear, noncausal attention, GELU MLP, residuals.
- [ ] Implement decoder projection, mask-token append, restoration gather, decoder blocks, decoder norm, prediction Linear.
- [ ] Implement `patchify` and masked MSE loss; add `norm_pix_loss` follow-up.
- [ ] Add source parity tests for base config one block, full encoder, decoder logits, and loss.
- [ ] Add patch Conv2d-to-GEMM rewrite tests for patch 16 and patch 14.
- [ ] Add QKV packing rewrite with `[q, k, v]` weight/bias concatenation.
- [ ] Add no-layout-translation guards around public NCHW ABI, patchify/unpatchify, and position interpolation.
- [ ] Benchmark preprocessing, patch embedding, encoder, decoder, attention backend, and reconstruction loss separately.
