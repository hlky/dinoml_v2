# Pixio Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from X:/H/transformers
Model id: pixio family; public mirror LiheYoung/pixio-vith16; official gated facebook/pixio-{vitb16,vitl16,vith16,vit1b16,vit5b16}
Config source: src/transformers/models/pixio/configuration_pixio.py, public LiheYoung config.json, converter size map
Source files inspected: configuration_pixio.py, modeling_pixio.py, modular_pixio.py by generation notice, convert_pixio_to_pytorch.py, tests/models/pixio/test_modeling_pixio.py, docs/source/en/model_doc/pixio.md, BitImageProcessor files, masking_utils.py
Any missing files or assumptions: official facebook configs and weights are gated; B/L/1B/5B checkpoint dimensions below are source-derived from the converter and HF repo metadata, not raw gated config.json files.
```

Small snapshots are in `config_snapshots.md` in this folder. The generated `modeling_pixio.py` and `configuration_pixio.py` are authoritative for runtime behavior in this checkout. `convert_pixio_to_pytorch.py` is useful for model-size names and original checkpoint QKV packing, but its target module names do not match the generated Pixio module names in this checkout, so treat it cautiously.

## 2. High-level architecture

Pixio is a vision-only ViT-style encoder and backbone. It has no text decoder, no autoregressive generation loop, no KV cache, and no multimodal embedding stitch. The model consumes preprocessed `pixel_values` in NCHW layout, applies non-overlapping patch embedding by `Conv2d`, prepends multiple learned class tokens, adds learned absolute position embeddings with bicubic interpolation for non-default image sizes, runs repeated bidirectional Transformer encoder blocks, then applies final LayerNorm and mean-pools the class tokens.

```text
image preprocessing -> NCHW pixel_values -> patch Conv2d -> CLS prepend + learned absolute position interpolation -> bidirectional ViT encoder -> final LayerNorm -> CLS mean pool / patch features
```

Stage decomposition:

```text
CPU/data pipeline: RGB conversion, resize/crop/rescale/normalize by BitImageProcessor
GPU/runtime stem: Conv2d patch embedding, flatten, transpose, CLS expand/concat, position add
Encoder: N repeated full bidirectional self-attention + MLP blocks
Backbone extraction: optional per-stage LayerNorm, remove CLS tokens, reshape to BCHW feature maps
```

The processor can be validated separately from the neural graph. Patch embedding and positional interpolation are independently testable. Encoder blocks can be validated one layer at a time, and `PixioBackbone` can be validated as a reshaping/extraction wrapper around `PixioModel`.

## 3. Important config dimensions

Default config from `PixioConfig`:

| Field | Default |
|---|---:|
| `hidden_size` | 1280 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 16 |
| `head_dim` | inferred as `hidden_size // num_attention_heads` unless `head_dim` exists |
| `mlp_ratio` | 4 |
| MLP intermediate | `hidden_size * mlp_ratio` |
| `hidden_act` | `gelu` |
| `image_size` | 256 |
| `patch_size` | 16 |
| `num_channels` | 3 |
| `n_cls_tokens` | 8 |
| `qkv_bias` | true |
| attention dropout | 0.0 |
| hidden dropout | 0.0 |
| stochastic depth | 0.0 |
| `layer_norm_eps` | 1e-6 |
| `apply_layernorm` for backbone | true |
| `reshape_hidden_states` for backbone | true |
| cache support | none; encoder only |
| RoPE/relative bias | none; learned absolute position embedding |

Representative checkpoint sweep:

| Model id/name | Config basis | hidden | layers | heads | head_dim | MLP hidden | Default sequence at 256x256 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `facebook/pixio-vitb16` | converter, gated repo metadata | 768 | 12 | 12 | 64 | 3072 | 264 | gated |
| `facebook/pixio-vitl16` | converter, gated repo metadata | 1024 | 24 | 16 | 64 | 4096 | 264 | gated |
| `LiheYoung/pixio-vith16` | public config | 1280 | 32 | 16 | 80 | 5120 | 264 | public mirror used by integration test |
| `facebook/pixio-vith16` | converter, gated repo metadata | 1280 | 32 | 16 | 80 | 5120 | 264 | gated official repo |
| `facebook/pixio-vit1b16` | converter, gated repo metadata | 1536 | 48 | 24 | 64 | 6144 | 264 | gated |
| `facebook/pixio-vit5b16` | converter, gated repo metadata | 3072 | 48 | 32 | 96 | 12288 | 264 | gated |

Sequence length is `n_cls_tokens + floor(H / patch_h) * floor(W / patch_w)`. At the standard 256x256, patch 16 gives `8 + 16 * 16 = 264`.

## 3a. Family variation traps

- `head_dim` is inferred from `hidden_size // num_attention_heads`, but H and 5B use 80 and 96 respectively, so do not assume 64.
- There are 8 class tokens by default, not one. Pooling averages all class tokens, and backbone reshape must drop all 8.
- Position embeddings are learned absolute embeddings and are interpolated with bicubic `align_corners=False` when runtime patch grid differs from the pretraining grid.
- `patch_size` and `image_size` can be scalar or iterable in the config, but `interpolate_pos_encoding` and backbone reshape use `height // self.patch_size` and `width // patch_size`; non-scalar patch sizes may need source-compatible guards or a bug-compatible fallback.
- `attention_mask` is accepted, but this is full bidirectional encoder attention. No causal mask or cache should be introduced.
- `qkv_bias` controls Q/K/V bias, while output projection and MLP projections always have bias in source.
- The source attention is MHA only. There is no GQA/MQA and no separate KV head count.
- `PixioBackbone` exposes selected stages through `BackboneMixin`; physical weights are one encoder stack, not separate stage modules.
- `apply_layernorm` and `reshape_hidden_states` change backbone output tensors without changing `PixioModel`.
- The original checkpoint converter says original QKV weights are packed as `[Q; K; V]`, but generated HF runtime uses separate `q_proj`, `k_proj`, and `v_proj`.
- Official Meta repos are gated and noncommercial-tagged; audits should not assume raw configs or weights are fetchable in CI.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input validation.
- `Conv2d(C -> hidden_size, kernel=patch_size, stride=patch_size, padding=0, groups=1)` for patch embedding.
- Flatten spatial patch grid, transpose NCHW projection to `(B, P, C)`.
- Learned class-token expand from `(1, n_cls_tokens, hidden)` to `(B, n_cls_tokens, hidden)`.
- Concatenate class and patch tokens on sequence axis.
- Add learned position embedding, with optional interpolation.
- Reshape/unflatten patch tokens to `(B, H/patch, W/patch, hidden)` and permute to `(B, hidden, H/patch, W/patch)` for backbone.
- Slice first `n_cls_tokens`, slice remaining patch tokens, mean over class-token axis.

Neural network primitives:

- LayerNorm over hidden dim, eps `1e-6`.
- Linear projections with shapes per block:
  - B: Q/K/V `768 -> 768`, O `768 -> 768`, MLP `768 -> 3072 -> 768`.
  - L: Q/K/V `1024 -> 1024`, O `1024 -> 1024`, MLP `1024 -> 4096 -> 1024`.
  - H: Q/K/V `1280 -> 1280`, O `1280 -> 1280`, MLP `1280 -> 5120 -> 1280`.
  - 1B: Q/K/V `1536 -> 1536`, O `1536 -> 1536`, MLP `1536 -> 6144 -> 1536`.
  - 5B: Q/K/V `3072 -> 3072`, O `3072 -> 3072`, MLP `3072 -> 12288 -> 3072`.
- GELU activation.
- Dropout and DropPath are inference identities for default eval path; training support requires stochastic per-sample mask.

Attention primitives:

- Dense bidirectional self-attention over all class and patch tokens.
- Q/K/V reshape to `(B, heads, S, head_dim)`.
- Matmul score, scale by `head_dim ** -0.5`, optional additive mask, softmax in fp32, dropout during training, value matmul.
- SDPA, FlashAttention, and Flex Attention are declared supported by Transformers, but eager math is the parity baseline.

Position ops:

- Learned absolute position table of shape `(1, n_cls_tokens + pretrain_patches, hidden)`.
- Bicubic interpolation of patch positions in BCHW layout at fp32 compute precision, then cast back.

Preprocessing-coupled ops:

- `BitImageProcessor`: convert RGB, resize, center crop, rescale, normalize, bicubic resampling.
- The public H config uses `image_size=256`; converter explicitly constructs a 256x256 Bit processor with no center crop and ImageNet mean/std for conversion validation. Repo processor snapshots should be verified per checkpoint.

Not applicable:

- No tokenizer, vocab, logits, LM head, generation sampling, KV cache, MoE, quantized source storage, recurrent/state-space cache, codebook, vocoder, or multimodal scatter.

## 5. Layer/block breakdown

Stem:

```text
pixel_values: (B, 3, H, W)
patch = Conv2d(3 -> hidden, kernel=16, stride=16)(pixel_values)
patch = flatten_spatial(patch).transpose(1, 2)        # (B, P, hidden)
cls = cls_token.expand(B, 8, hidden)
x = concat(cls, patch, dim=1)                         # (B, 8 + P, hidden)
x = dropout(x + interpolate_pos_encoding(x, H, W))
```

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
y = LayerNorm(x, eps=1e-6)
q = Linear(hidden -> heads * head_dim, bias=qkv_bias)(y)
k = Linear(hidden -> heads * head_dim, bias=qkv_bias)(y)
v = Linear(hidden -> heads * head_dim, bias=qkv_bias)(y)
attn = bidirectional_attention(q, k, v, optional_mask)
x = residual + DropPath(Dropout(Linear(heads * head_dim -> hidden, bias=True)(attn)))

residual = x
y = LayerNorm(x, eps=1e-6)
y = Linear(hidden -> hidden * mlp_ratio, bias=True)(y)
y = GELU(y)
y = Linear(hidden * mlp_ratio -> hidden, bias=True)(y)
x = residual + DropPath(Dropout(y))
```

Headless model output:

```text
last_hidden_state = LayerNorm(x, eps=1e-6)
pooler_output = mean(last_hidden_state[:, :n_cls_tokens, :], dim=1)
```

Backbone output:

```text
hidden_states = captured stem and layer outputs
for requested stages:
    optional LayerNorm
    optional drop class tokens
    optional reshape to (B, hidden, H/patch, W/patch)
```

## 6. Attention requirements

Pixio requires noncausal dense self-attention only.

| Requirement | Pixio behavior |
|---|---|
| causal/noncausal | noncausal bidirectional |
| attention type | self-attention |
| heads | equal Q, K, V head count |
| KV heads | same as query heads; no GQA/MQA |
| head_dim | inferred, varies by family |
| rectangular attention | no, Q length equals KV length in model path |
| mask style | optional 2D padding mask or prepared 4D additive mask via `create_bidirectional_mask` |
| packed/varlen | not source-required |
| sliding/local | no |
| ALiBi/RoPE | no |
| KV cache | no |
| backend compatibility | eager parity baseline; Transformers declares SDPA, FlashAttention, Flex Attention support |

For fused attention parity, preserve source math order: scale scores, add mask, softmax with fp32 accumulation, cast to query dtype, dropout if training, then value matmul.

## 7. Position encoding and custom math

Pixio uses learned absolute positional embeddings. Patch positions are interpolated when the runtime patch count differs from the learned table or when tracing forces interpolation.

```python
def pixio_position_embedding(position_embeddings, n_cls_tokens, x, height, width, patch_size):
    cls_pos = position_embeddings[:, :n_cls_tokens]
    patch_pos = position_embeddings[:, n_cls_tokens:]
    dim = x.shape[-1]
    old = int(patch_pos.shape[1] ** 0.5)
    new_h = height // patch_size
    new_w = width // patch_size
    patch_pos = patch_pos.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch_pos = interpolate(patch_pos.float(), size=(new_h, new_w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.to(position_embeddings.dtype).permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return concat([cls_pos, patch_pos], dim=1)
```

The source can skip interpolation when not tracing and the input matches the learned square grid. For optimized lowering, precompute the default 256x256 position add, but keep a dynamic interpolation path for other image sizes.

## 8. Preprocessing and input packing

The model graph starts at `pixel_values`. The image processor owns RGB conversion, resize/crop/rescale/normalize. Auto image processing maps Pixio to `BitImageProcessor` or `BitImageProcessorPil`.

Source layout is NCHW. Patch embedding expects `(batch, channels, height, width)` and raises when channel count differs from config. No token type ids, attention position ids, placeholder tokens, or multimodal scatter are used.

Processor details to pin per checkpoint:

- Generic Bit defaults: bicubic resize to shortest edge 224, center crop 224x224, rescale, normalize with CLIP mean/std, RGB conversion.
- Pixio converter validation processor: 256x256 size and crop size, `do_center_crop=False`, bicubic, ImageNet mean/std.
- Public mirror config gives neural `image_size=256`, but image processor settings live outside `config.json`; save processor snapshots when integrating a concrete checkpoint.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d to Linear

Source pattern:

```text
Conv2d(C -> hidden, kernel=(ph,pw), stride=(ph,pw), padding=0, dilation=1, groups=1)
flatten(2).transpose(1, 2)
```

Replacement:

```text
non-overlap WindowFlatten(NCHW, ph, pw) -> Linear(C * ph * pw -> hidden) -> sequence
```

Preconditions:

- `kernel_size == stride == patch_size`.
- Padding is zero, dilation is one, groups is one.
- Input height and width are divisible by patch height and width, or source-compatible floor behavior is accepted and tested.
- NCHW layout is preserved through window flatten.

Weight transform:

```python
linear_weight = conv_weight.reshape(hidden, channels * ph * pw)
linear_bias = conv_bias
```

Failure cases:

- Channel-last input without explicit layout rewrite.
- Non-scalar patch-size path unless audited against source bugs.
- Inputs where trailing pixels would be dropped differently by an alternative patch extractor.

Parity test sketch:

```text
random NCHW tensors over several H/W multiples of 16, compare Conv2d stem output after flatten/transpose to rewritten Linear sequence at fp32 tolerance.
```

### Rewrite: split or fuse QKV projections

Source pattern:

```text
q = q_proj(x); k = k_proj(x); v = v_proj(x)
```

Replacement:

```text
qkv = Linear(hidden -> 3 * hidden) then split [Q, K, V]
```

Preconditions:

- Same input tensor, dtype, bias enabled consistently.
- No intervening operations between projections.
- Packed output split order is `[Q, K, V]`.

Weight transform:

```python
packed_weight = concat([q_weight, k_weight, v_weight], dim=0)
packed_bias = concat([q_bias, k_bias, v_bias], dim=0)
```

Failure cases:

- Quantized or sharded weights with separate materialization policies.
- Future configs with different Q/K/V widths.

### Rewrite: backbone reshape as metadata when consumer supports BCHW

Source pattern:

```text
drop CLS -> reshape(B, H/patch, W/patch, hidden) -> permute(0, 3, 1, 2).contiguous()
```

Replacement:

```text
layout-aware view/transpose or direct BCHW producer
```

Preconditions:

- Consumer accepts BCHW and class tokens are not consumed.
- `reshape_hidden_states=True`.
- Axis rewrite is explicit: sequence patch index order is row-major over patch height, patch width.

Failure cases:

- Feature maps requested with `reshape_hidden_states=False`.
- Consumers needing token sequence layout.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over hidden: used twice per block plus final norm and optional backbone norms.
- Dense bidirectional attention with fp32 softmax: main encoder bottleneck at high sequence or large hidden sizes.
- QKV projection fusion: three GEMMs per block can become one packed GEMM.
- MLP `Linear + GELU + Linear`: large GEMM plus activation, especially 5B with `3072 -> 12288 -> 3072`.

Medium priority:

- Patch Conv2d to GEMM/Linear: useful for provider reuse and simplifying NCHW-to-sequence lowering.
- Position interpolation kernel or precomputed default path: default shape can fold, dynamic sizes need guarded interpolation.
- Residual add plus dropout/drop-path identity cleanup for inference.

Lower priority:

- Backbone feature extraction layout fusion: useful for dense downstream heads but not required for base model parity.
- Last-use class-token pooling fusion.

## 11. Runtime staging plan

Stage 1: parse `PixioConfig`, load public H checkpoint config, and instantiate random-weight graph with shape checks.

Stage 2: implement patch embedding, class token prepend, position add, and final pooling for the default 256x256 path.

Stage 3: implement one encoder block parity with eager bidirectional attention, LayerNorm, GELU MLP, and residuals.

Stage 4: scale to full `PixioModel` parity for H public mirror, eval mode only.

Stage 5: add positional interpolation parity for non-default image sizes.

Stage 6: add `PixioBackbone` stage extraction, optional layernorm, and BCHW reshape.

Stage 7: optimize with patch Conv2d rewrite, QKV fusion, attention backend selection, and MLP fusions.

Can stub initially: training dropout, stochastic depth, gradient checkpointing, output attentions, Flash/Flex-specific dispatch, and gated official checkpoint weight loading.

## 12. Parity and validation plan

- Config round-trip tests for scalar and tuple `image_size`/`patch_size`, with explicit guards for tuple cases if unsupported.
- Patch embedding parity: Conv2d output vs DinoML stem for fp32 random NCHW tensors.
- Position interpolation parity: default 256x256 skip path and at least one non-default size, comparing bicubic fp32 interpolation.
- Single-block parity: random hidden states and masks for eager attention and MLP at fp32.
- Full-model random parity: tiny config from Transformers tests, expected shape `(B, patches + n_cls_tokens, hidden)`.
- Public checkpoint parity: `LiheYoung/pixio-vith16`, COCO fixture or fixed image, expected output shape `(1, 264, 1280)` and slice tolerance matching Transformers test `rtol=1e-4`, `atol=1e-4`.
- Backbone parity: requested `out_features`, `apply_layernorm` true/false, `reshape_hidden_states` true/false.
- Suggested tolerances: fp32 `1e-4`; fp16/bf16 start at `5e-3` to `1e-2` for full model, stricter for isolated GEMMs and norms.

## 13. Performance probes

- Image processor throughput separately from GPU model time.
- Patch embedding/stem latency for batch and resolution sweeps.
- Encoder-only throughput for B/L/H/1B/5B dimensions at sequence 264.
- Sequence length sweep for dynamic image sizes: 224, 256, 384, 512.
- Attention backend comparison: eager, SDPA, FlashAttention, Flex where available.
- QKV fused vs separate GEMM timing.
- MLP GEMM throughput by family size.
- Backbone extraction overhead, especially reshape and contiguous permute.
- Memory footprint by checkpoint size, dtype, batch, and resolution.

## 14. Skip/defer list

- Training, gradients, stochastic depth randomness, and gradient checkpointing.
- Official gated Meta checkpoint downloads in automated validation unless access is configured.
- Classification, detection, depth, segmentation, and robotics heads; Pixio source exposes base model and backbone only.
- Quantized weights and packed runtime materialization beyond original QKV conversion metadata.
- Autoregressive generation, KV cache, beam search, tokenization, and multimodal inputs.
- Tensor parallel or distributed inference.
- Non-square or tuple `patch_size` support until source behavior is explicitly validated.

## 15. Final implementation checklist

- [ ] Parse `PixioConfig` including backbone fields and `n_cls_tokens`.
- [ ] Save concrete model and image processor snapshots for each target checkpoint.
- [ ] Load Pixio weights with separate Q/K/V HF layout and verify aliases are not accidentally cloned.
- [ ] Implement NCHW patch `Conv2d` stem and optional Conv2d-to-Linear rewrite.
- [ ] Implement multi-CLS token prepend and pooled mean over class tokens.
- [ ] Implement learned absolute position add and bicubic interpolation fallback.
- [ ] Implement bidirectional MHA with fp32 softmax parity.
- [ ] Implement GELU MLP and LayerNorm eps `1e-6`.
- [ ] Implement full encoder and final LayerNorm.
- [ ] Implement `PixioBackbone` stage capture, optional stage LayerNorm, CLS removal, and BCHW reshape.
- [ ] Add tiny-config random parity tests.
- [ ] Add public `LiheYoung/pixio-vith16` parity test for shape and fixed output slice.
- [ ] Add dynamic-size position interpolation parity test.
- [ ] Benchmark stem, attention, MLP, full encoder, and backbone extraction separately.
