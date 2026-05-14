# Transformers audit: metaclip_2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/metaclip-2-worldwide-huge-quickgelu, plus representative public Hub configs listed below
Config source: local configuration_metaclip_2.py and Hugging Face Hub config.json files
Source files inspected:
  X:/H/transformers/src/transformers/models/metaclip_2/configuration_metaclip_2.py
  X:/H/transformers/src/transformers/models/metaclip_2/modeling_metaclip_2.py
  X:/H/transformers/src/transformers/models/metaclip_2/modular_metaclip_2.py
  X:/H/transformers/src/transformers/models/metaclip_2/convert_metaclip_2_to_hf.py
  X:/H/transformers/src/transformers/models/auto/{auto_mappings,modeling_auto,processing_auto,tokenization_auto,image_processing_auto}.py
Any missing files or assumptions:
  No remote-code files are required for the audited native source path.
  The generated configuration/modeling files say future source edits should be made in modular_metaclip_2.py.
  CLIPImageProcessor and tokenizer implementations are composed from existing Transformers families, not owned by this model directory.
```

Representative public configs inspected from Hugging Face Hub on 2026-05-13:

- [facebook/metaclip-2-worldwide-s16](https://huggingface.co/facebook/metaclip-2-worldwide-s16)
- [facebook/metaclip-2-worldwide-b16](https://huggingface.co/facebook/metaclip-2-worldwide-b16)
- [facebook/metaclip-2-worldwide-b32](https://huggingface.co/facebook/metaclip-2-worldwide-b32)
- [facebook/metaclip-2-worldwide-l14](https://huggingface.co/facebook/metaclip-2-worldwide-l14)
- [facebook/metaclip-2-worldwide-huge-quickgelu](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu)
- [facebook/metaclip-2-worldwide-huge-378](https://huggingface.co/facebook/metaclip-2-worldwide-huge-378)
- [facebook/metaclip-2-worldwide-giant](https://huggingface.co/facebook/metaclip-2-worldwide-giant)
- [facebook/metaclip-2-worldwide-giant-378](https://huggingface.co/facebook/metaclip-2-worldwide-giant-378)
- [facebook/metaclip-2-mt5-worldwide-b32](https://huggingface.co/facebook/metaclip-2-mt5-worldwide-b32)

All checked links were public and not gated via `HfApi.model_info`. A compact config snapshot is in `checkpoint_sweep.json`.

## 2. High-level architecture

MetaClip2 is a CLIP-style dual encoder for image-text retrieval and contrastive scoring. Primary DinoML runtime target: `MetaClip2Model` inference, producing normalized image/text embeddings and similarity matrices.

```text
image preprocessing -> ViT vision encoder -> CLS pool -> visual projection -> L2 normalize
text tokenization -> causal text encoder -> EOS pool -> text projection -> L2 normalize
normalized text/image features -> logit_scale.exp() * text @ image.T -> logits_per_text/logits_per_image
```

Stage decomposition:

- CPU/data pipeline: image resize/crop/rescale/normalize, tokenizer-specific text tokenization, padding/truncation.
- Vision branch: NCHW `pixel_values` to non-overlapping patch Conv2d, class token, absolute learned positions, encoder stack, CLS pool.
- Text branch: token and learned absolute position embeddings, causal self-attention stack with optional padding mask, final LayerNorm, first-EOS pooling.
- Projection/similarity head: two bias-free Linear projections to shared `projection_dim`, vector norm division, matrix multiply, transpose.
- Independently cacheable outputs: image embeddings and text embeddings can be cached before the final similarity matrix. There is no autoregressive decode path or KV cache.

Implemented heads:

- `MetaClip2Model`: required for the primary contrastive target.
- `MetaClip2TextModel`, `MetaClip2VisionModel`: required as independently stageable branch targets.
- `MetaClip2TextModelWithProjection`, `MetaClip2VisionModelWithProjection`: optional wrappers useful for branch-only embedding parity.
- `MetaClip2ForImageClassification`: deferred unless DinoML wants classification fine-tune parity; it uses mean pooling over patch tokens, not CLS.

## 3. Important config dimensions

Source defaults from `configuration_metaclip_2.py`:

| Field | Text default | Vision default | Notes |
|---|---:|---:|---|
| `vocab_size` | 49408 | n/a | Hub worldwide configs override this to 901629 or 250000. |
| `hidden_size` | 512 | 768 | Must divide `num_attention_heads`. |
| `intermediate_size` | 2048 | 3072 | Plain MLP, not gated. |
| `projection_dim` | 512 | 512 | Top-level projection also defaults to 512. |
| `num_hidden_layers` | 12 | 12 | Branches can differ in Hub configs. |
| `num_attention_heads` | 8 | 12 | MHA only, no GQA/MQA. |
| `head_dim` | 64 | 64 | Computed as `hidden_size // heads`. |
| `max_position_embeddings` | 77 | n/a | Text hard guard. |
| `image_size` | n/a | 224 | Hub has 224 and 378. |
| `patch_size` | n/a | 32 | Hub has 14, 16, 32. |
| `hidden_act` | quick_gelu | quick_gelu | Hub has both `gelu` and `quick_gelu`. |
| `layer_norm_eps` | 1e-5 | 1e-5 | LayerNorm bias and weight present. |
| `attention_dropout` | 0.0 | 0.0 | Runtime dropout disabled for inference. |
| `logit_scale_init_value` | 2.6592 | n/a | Top-level scalar parameter, exponentiated at inference. |

Representative checkpoint sweep, from Hub `config.json`:

| Model | Text H/L/A/MLP | Vision H/L/A/MLP | Image/patch | Projection | Act | Tokenizer/EOS |
|---|---:|---:|---:|---:|---|---|
| worldwide-s16 | 384/12/6/1536 | 384/12/6/1536 | 224/16 | 384 | gelu | XLM-R, EOS 2 |
| worldwide-b32 | 512/12/8/2048 | 768/12/12/3072 | 224/32 | 512 | gelu | XLM-R, EOS 2 |
| worldwide-l14 | 768/12/12/3072 | 1024/24/16/4096 | 224/14 | 768 | gelu | XLM-R, EOS 2 |
| worldwide-huge-quickgelu | 1024/24/16/4096 | 1280/32/16/5120 | 224/14 | 1024 | quick_gelu | XLM-R, EOS 2 |
| worldwide-huge-378 | 1024/24/16/4096 | 1280/32/16/5120 | 378/14 | 1024 | gelu | XLM-R, EOS 2 |
| worldwide-giant-378 | 1280/32/20/5120 | 1664/48/16/8192 | 378/14 | 1280 | gelu | XLM-R, EOS 2 |
| mt5-worldwide-b32 | 512/12/8/2048 | 768/12/12/3072 | 224/32 | 512 | gelu | SiglipTokenizer, EOS 1 |

Patch-token counts:

- 224/32: 49 patches plus CLS = 50 positions.
- 224/16: 196 patches plus CLS = 197 positions.
- 224/14: 256 patches plus CLS = 257 positions.
- 378/14: 729 patches plus CLS = 730 positions.

Notable head dimensions:

- Most text branches use `head_dim=64`.
- Huge vision uses `1280 / 16 = 80`.
- Giant vision uses `1664 / 16 = 104`.

## 3a. Family variation traps

- Text and vision dimensions are not tied. Projection widths vary from 384 to 1280.
- Vision head dim is not always 64. Do not bake in CLIP-B assumptions.
- Patch size and position table length vary materially: 50, 197, 257, or 730 vision tokens in inspected configs.
- `hidden_act` can be `gelu` or `quick_gelu`; huge-quickgelu needs quick GELU parity.
- Text source uses causal attention, not a bidirectional text encoder.
- Text pooling finds the first position equal to `config.eos_token_id`. Native source does not preserve old CLIP argmax-token pooling compatibility for EOS 2.
- Config class defaults list CLIP-like token IDs, but released Hub configs set `bos_token_id=None`, `pad_token_id=None`, and `eos_token_id` to tokenizer-specific values.
- XLM-R checkpoints use `vocab_size=901629`; the mT5-named checkpoint uses `SiglipTokenizer`, `vocab_size=250000`, `eos_token_id=1`, and tokenizer `model_max_length=64`. The native neural body remains `MetaClip2Model`; the tokenizer ABI changes.
- `image_size` and `patch_size` are scalar in inspected configs. The config type hints allow list/tuple, but the current source computes `(self.image_size // self.patch_size) ** 2`, so first DinoML admission should require scalar square images and scalar patch sizes.
- `interpolate_pos_encoding=True` enables dynamic bicubic interpolation of absolute vision position embeddings. First integration can reject it and require exact processor image size.
- Source expects NCHW `pixel_values` and runs `Conv2d`. NHWC/channel-last is only an optimization region around patch extraction and normalization, not a semantic default.
- The conversion script documents original MetaCLIP packed `attn.in_proj_{weight,bias}` split order as Q, K, V. Native HF weights are already separate `q_proj`, `k_proj`, `v_proj`.
- Some Hub configs set `architectures` to `MetaCLIP2Model` with different capitalization than native class `MetaClip2Model`. DinoML should key admission on `model_type=metaclip_2`, not only that string.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape/view: `view`, `reshape`, `flatten(2)`, `transpose(1, 2)`, `contiguous`.
- `cat` along sequence axis for class token and patches.
- Token gather/index for EOS pooling: first equality match per row, then batch gather.
- Mean over patch-token axis for deferred image classification head.
- Matrix transpose for image/text similarity orientation.

Neural network primitives:

- Embedding lookup: token embedding `[vocab_size, text_hidden]`, text position embedding `[77, text_hidden]`, vision position embedding `[num_patches+1, vision_hidden]`.
- Conv2d patch embedding: `Conv2d(3 -> vision_hidden, kernel=patch_size, stride=patch_size, bias=False)`.
- LayerNorm with affine weight/bias, eps 1e-5.
- Linear with bias for Q/K/V/out and MLP fc1/fc2.
- Bias-free projection heads: visual `Linear(vision_hidden -> projection_dim)`, text `Linear(text_hidden -> projection_dim)`.
- Activation: GELU and quick GELU.
- L2 vector norm and division along feature dimension.
- Scalar `exp(logit_scale)` multiply.

Attention primitives:

- Dense MHA self-attention only.
- Separate Q, K, V projections, all `hidden_size -> hidden_size` with bias.
- Query/key/value reshaped to `[B, heads, seq, head_dim]`.
- Attention scores `q @ k.T * head_dim**-0.5`.
- Additive mask, softmax over key axis with fp32 softmax in eager path, dropout disabled in inference, `weights @ v`, output projection.
- Source declares SDPA, FlashAttention, and FlexAttention backend support through Transformers attention interfaces.

Position/custom math:

- Learned absolute text positions.
- Learned absolute vision class/patch positions.
- Optional bicubic interpolation of vision patch position grid.

Preprocessing-coupled ops:

- CLIP image processor: resize, center crop, rescale by `1/255`, normalize with CLIP mean/std, output NCHW.
- Tokenizer: XLMRobertaTokenizer for most worldwide configs; SiglipTokenizer for mT5-named configs.
- Padding and truncation happen outside the model. Text sequence must be no longer than 77 for native source defaults/configs.

Generation/cache ops:

- Not applicable. There is no LM head, no autoregressive decode loop, no KV cache, and no sampling.

Quantized/packed metadata:

- No native quantized or packed-weight path in inspected source.
- Conversion from original MetaCLIP packed QKV is a loader concern only; native HF state dict stores separate projection weights.

## 5. Layer/block breakdown

Text embeddings:

```text
input_ids [B,T] where T <= 77
token = Embedding(vocab_size, Ht)(input_ids) -> [B,T,Ht]
pos = Embedding(77, Ht)(position_ids[:T]) -> [1,T,Ht]
x = token + pos
```

Text encoder block, repeated `text_config.num_hidden_layers`:

```text
residual = x
x = LayerNorm(Ht)(x)
q = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,Dt).transpose(1,2)
k = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,Dt).transpose(1,2)
v = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,Dt).transpose(1,2)
x_attn = dense causal self-attention(q,k,v, additive_mask)
x = residual + Linear(Ht -> Ht, bias=True)(x_attn)
residual = x
x = LayerNorm(Ht)(x)
x = residual + Linear(I -> Ht, bias=True)(act(Linear(Ht -> I, bias=True)(x)))
```

Text head:

```text
x = final_layer_norm(x)
pool = x[batch_index, first_index(input_ids == eos_token_id), :]
text_embed_raw = Linear(Ht -> P, bias=False)(pool)
text_embed = text_embed_raw / norm(text_embed_raw, dim=-1, keepdim=True)
```

Vision embeddings:

```text
pixel_values [B,3,H,W] in NCHW
patch = Conv2d(3 -> Hv, kernel=patch, stride=patch, bias=False)(pixel_values)
patch = patch.flatten(2).transpose(1,2) -> [B,N,Hv]
cls = class_embedding.expand(B,1,Hv)
x = cat([cls, patch], dim=1) + learned_position_embedding
```

Vision encoder block, repeated `vision_config.num_hidden_layers`, is the same pre-norm MHA plus MLP structure, but noncausal and usually unmasked.

Vision head:

```text
x = pre_layrnorm(vision_embeddings)
x = encoder(x)
pool = post_layernorm(x[:,0,:])
image_embed_raw = Linear(Hv -> P, bias=False)(pool)
image_embed = image_embed_raw / norm(image_embed_raw, dim=-1, keepdim=True)
```

Similarity:

```text
logits_per_text = exp(logit_scale) * text_embed @ image_embed.T
logits_per_image = logits_per_text.T
```

## 6. Attention requirements

Text attention:

- Causal self-attention over token sequence. The source calls `create_causal_mask(..., past_key_values=None)` and passes `is_causal=True` into the encoder.
- Optional `attention_mask` from tokenizer is folded into the additive causal mask.
- MHA only: `num_key_value_heads` is absent and K/V heads equal query heads.
- Q/K/V widths are all `hidden_size`; value width equals key/query width.
- No cache. Every call recomputes all text token states.
- Maximum native sequence length is governed by learned positions, normally 77.

Vision attention:

- Noncausal dense self-attention over `[CLS] + patch_tokens`.
- No attention mask in normal image path.
- MHA only, no relative bias, no RoPE, no local/window sparsity.
- Sequence lengths are fixed by `image_size / patch_size` unless interpolation is enabled.

Backend compatibility:

- Eager path uses fp32 softmax then casts to query dtype.
- Source advertises SDPA, FlashAttention, and FlexAttention through `ALL_ATTENTION_FUNCTIONS`.
- For first DinoML parity, implement the eager math order and then allow optimized dense attention when masks and dtype match.

## 7. Position encoding and custom math

There is no RoPE, ALiBi, or relative position bias. Both branches use learned absolute positions.

Vision interpolation path, required only when admitting `interpolate_pos_encoding=True` or tracing with dynamic image size:

```python
def metaclip2_interpolate_pos(position_weight, embeddings, height, width, patch_size):
    cls = position_weight[None, :1]
    patch = position_weight[None, 1:]
    dim = embeddings.shape[-1]
    old = int((patch.shape[1]) ** 0.5)
    new_h = height // patch_size
    new_w = width // patch_size
    patch = patch.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return cat([cls, patch], dim=1)
```

Precomputable:

- Text position IDs and position embeddings for fixed `T`.
- Vision position embeddings for exact configured image size.
- For fixed non-configured image sizes, interpolated vision position embeddings can be precomputed if the admission policy allows those sizes.

Dynamic:

- EOS pooling index depends on tokenized input IDs.
- Padding mask and causal text mask depend on sequence length and batch padding.

## 8. Preprocessing and input packing

Image processor contract:

- Processor class in inspected configs: `CLIPImageProcessor`.
- Input images are resized and center-cropped to configured size, rescaled by `0.00392156862745098`, normalized with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Model input is `pixel_values` as `[B,3,H,W]`, NCHW.
- Huge-quickgelu preprocessor uses `size.shortest_edge=224` and center crop 224. Other inspected configs use explicit height/width resize/crop.

Text tokenizer contract:

- Most worldwide configs use `XLMRobertaTokenizer`, with `<s>`, `</s>`, `<pad>`, `<unk>` and `eos_token_id=2`.
- `facebook/metaclip-2-mt5-worldwide-b32` uses `SiglipTokenizer`, `vocab_size=250000`, tokenizer `model_max_length=64`, and `eos_token_id=1`.
- Native model receives only `input_ids`, optional `attention_mask`, and optional `position_ids`.
- The model pools at the first token equal to `config.eos_token_id`. Admission should guard that each row contains at least one EOS token, because `argmax` on an all-false row would silently select index 0.

No multimodal placeholder stitching is used. The branches are independent until the final similarity matrix.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed to GEMM

Source pattern:

```text
Conv2d(C=3 -> H, kernel=P, stride=P, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1,2)
```

Replacement:

```text
NCHW image -> non-overlapping patch extract in row-major h,w order
-> flatten each patch [3*P*P]
-> MatMul(weight_flat.T)
-> [B, N_patches, H]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- Input height and width exactly match config or an admitted interpolated-position size.
- Height and width divisible by patch size.
- Patch flatten order matches PyTorch Conv2d weight layout `[out_channels, in_channels, kh, kw]` over NCHW input.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_size * patch_size)
y = patch_matrix @ w.T
```

Layout constraints:

- Initial semantic graph remains NCHW.
- A guarded NHWC implementation must rewrite patch extraction and weight flatten order together and emit the same row-major patch token order expected by position embeddings.

Failure cases:

- Non-square/list image sizes until source behavior is separately verified.
- `interpolate_pos_encoding=True` with unvalidated dynamic sizes.

Parity test sketch:

- Compare Conv2d+flatten+transpose against patch-GEMM for random fp32 and fp16 inputs at 224/14, 224/16, 224/32, and 378/14.

### Rewrite: branch projection plus L2 normalize

Source pattern:

```text
pool -> Linear(..., bias=False) -> x / norm(x)
```

Replacement:

```text
MatMul -> vector_norm -> reciprocal multiply
```

Preconditions:

- Projection has no bias.
- Norm axis is final feature dimension and keepdim semantics are preserved.
- Avoid fusing across the projection when tests require raw branch embeddings.

Failure cases:

- Zero vectors. Source does not add epsilon; parity path should preserve that behavior.

### Rewrite: separate Q/K/V projection packing

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
```

Replacement:

```text
packed_qkv = MatMul(x, concat([Wq, Wk, Wv]).T) + concat([bq, bk, bv])
split packed_qkv as [q, k, v] along feature axis
```

Preconditions:

- Same input tensor and same dtype for all three projections.
- Split order is Q, K, V.
- Preserve independent parameter names for loading and debugging.

Failure cases:

- Source already converted from original packed weights; do not assume original checkpoint storage in HF state dict.

### Rewrite: similarity matrix orientation

Source pattern:

```text
logits_per_text = scale * text_embeds @ image_embeds.T
logits_per_image = logits_per_text.T
```

Replacement:

```text
one GEMM plus transpose/view for the opposite orientation
```

Preconditions:

- Both embeddings are already L2-normalized.
- `scale = exp(logit_scale)` is scalar.
- Preserve output naming: `logits_per_text` is `[B_text, B_image]`; `logits_per_image` is `[B_image, B_text]`.

### Layout opportunity: local vision patch path

Candidate optimized region:

```text
processor output or early GPU copy -> patch extraction -> patch projection -> [B,N,H] tokens
```

Axis-sensitive details:

- Source input is NCHW with channel axis 1.
- Processor normalization uses channel-wise mean/std.
- Patch tokens are ordered by row-major spatial grid after `flatten(2).transpose(1,2)`.
- Position embedding assumes that same patch order.

Guard:

- Treat image preprocessing plus patch embedding as a controlled local region before translating to NHWC/channel-last. Downstream transformer blocks are sequence-major `[B,N,H]` and independent of image layout.

## 10. Kernel fusion candidates

Highest priority:

- Dense MHA for short/medium encoder sequences: text length 77, vision length up to 730. Fused attention matters most for large giant vision stacks.
- LayerNorm plus Linear where provider supports it. Every encoder block has two pre-norms.
- QKV packed projection for both branches. Reduces three GEMM launches per layer.
- MLP Linear+activation+Linear, with GELU and quick GELU variants.
- Patch Conv2d to GEMM for non-overlap patches, especially 378/14 with 729 patches.

Medium priority:

- Projection plus L2 normalization for branch embedding production.
- Final similarity GEMM plus scalar multiply and paired output orientation.
- EOS pooling gather, including validation path for missing EOS.
- Vision position interpolation cache for admitted dynamic sizes.

Lower priority:

- Image classification mean-pool head.
- Training losses, contrastive loss, and classifier label losses.
- Output-attentions materialization if optimized attention fast path does not return dense weights.

## 11. Runtime staging plan

Stage 1: config and weight admission.

- Parse `model_type=metaclip_2` and nested `text_config`/`vision_config`.
- Admit scalar square `image_size`, scalar `patch_size`, MHA only, no interpolation initially.
- Load separate HF Q/K/V weights and branch projection weights.

Stage 2: branch primitive parity.

- Text embeddings, causal mask, one text block, EOS pooling.
- Vision patch embedding, class token, positions, one vision block, CLS pooling.

Stage 3: full branch parity.

- Run full text encoder and full vision encoder for b32/s16/l14 representative configs.
- Add GELU and quick GELU coverage.

Stage 4: contrastive head parity.

- Bias-free projections, L2 normalization, logit scale exponent, `text @ image.T`, output orientation.

Stage 5: optimized dense attention and QKV packing.

- Enable packed QKV GEMM and fused attention with strict mask/layout/dtype guards.

Stage 6: vision layout and patch GEMM.

- Add guarded Conv2d-to-GEMM patch rewrite and optional NHWC local implementation.

Stage 7: optional extensions.

- Admit 378-resolution giant/huge production variants.
- Add optional position interpolation.
- Add branch-only projection wrappers and image-classification head if needed.

## 12. Parity and validation plan

Random tensor tests:

- Conv2d patch embedding versus patch-GEMM for 224/32, 224/16, 224/14, 378/14.
- GELU and quick GELU activation parity.
- Attention eager math with causal text mask and unmasked vision attention.
- L2 normalization with no epsilon.
- EOS pooling gather with one EOS, multiple EOS, and missing-EOS rejection guard.

Single-layer parity:

- Text block at H=384/A=6/T=77 and H=1280/A=20/T=77.
- Vision block at H=768/A=12/N=50, H=1280/A=16/N=730, H=1664/A=16/N=730.

Full-model parity:

- `facebook/metaclip-2-worldwide-b32`: common small baseline.
- `facebook/metaclip-2-worldwide-huge-quickgelu`: quick GELU and H/14 baseline.
- `facebook/metaclip-2-worldwide-giant-378`: largest inspected sequence/width.
- `facebook/metaclip-2-mt5-worldwide-b32`: tokenizer/EOS/vocab ABI variation.

End-to-end checks:

- Processor output shape and dtype against Transformers for one image.
- Text/image embeddings before normalization and after normalization.
- `logits_per_text` and `logits_per_image` orientation and values.

Suggested tolerances:

- fp32: atol 1e-5 to 1e-4 depending on attention backend.
- fp16/bf16: atol 1e-2, rtol 1e-2 for full model, with tighter per-op tolerances where possible.

No DinoML tests were run for this audit, per scope.

## 13. Performance probes

- Image processor throughput versus model vision encoder throughput.
- Patch embedding Conv2d versus patch-GEMM at 224/32, 224/16, 224/14, 378/14.
- Vision encoder throughput by sequence length: 50, 197, 257, 730.
- Text encoder throughput at padded length 64 and 77.
- Fused dense attention versus unfused eager attention for head dims 64, 80, 104.
- Batch-size sweep for retrieval scoring: separate image embedding, text embedding, and similarity GEMM.
- Similarity matrix scaling with `B_text * B_image * projection_dim`.
- Memory use for giant-378 vision attention activations.
- GELU versus quick GELU kernel cost.
- Branch embedding cache hit scenario: precompute images and score many text batches, and vice versa.

## 14. Skip/defer list

- Training contrastive loss and classifier losses.
- Gradient checkpointing.
- Output attentions and hidden state capture, except as debug parity.
- Vision position interpolation for first admission.
- Image classification head unless specifically requested.
- Quantization and packed original MetaCLIP checkpoint conversion.
- Multi-GPU tensor parallelism.
- Any generation, beam search, sampling, or KV-cache scheduling: not applicable to this family.
- Non-square/list image configs until source semantics are verified.

## 15. Final implementation checklist

- [ ] Parse `MetaClip2Config` with nested text and vision configs.
- [ ] Admit/reject scalar `image_size`, scalar `patch_size`, `hidden_size % num_attention_heads == 0`.
- [ ] Load token, position, class, patch, encoder, projection, and `logit_scale` weights.
- [ ] Implement text token+position embeddings.
- [ ] Implement CLIP-style causal text mask with optional padding mask.
- [ ] Implement MHA encoder block with separate Q/K/V and fp32-softmax parity path.
- [ ] Implement GELU and quick GELU MLP.
- [ ] Implement EOS pooling by first `input_ids == eos_token_id` with missing-EOS guard.
- [ ] Implement NCHW vision patch embedding and class/position addition.
- [ ] Implement vision CLS pooling and post LayerNorm.
- [ ] Implement bias-free visual/text projections.
- [ ] Implement L2 normalization with source no-epsilon behavior.
- [ ] Implement `exp(logit_scale) * text @ image.T` and output transpose.
- [ ] Add patch Conv2d-to-GEMM rewrite with layout/order tests.
- [ ] Add packed QKV projection rewrite with Q,K,V split-order tests.
- [ ] Add branch-only parity tests for b32, huge-quickgelu, giant-378, and mt5-b32 configs.
- [ ] Add end-to-end similarity parity tests using Transformers processor/tokenizer outputs.
- [ ] Benchmark processor, branch encoders, attention backend, patch embedding, and similarity GEMM separately.
