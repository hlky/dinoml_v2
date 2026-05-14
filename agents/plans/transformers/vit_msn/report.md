# Transformers Audit: vit_msn

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/vit-msn-small, facebook/vit-msn-base, facebook/vit-msn-large, facebook/vit-msn-base-4, facebook/vit-msn-large-7
Config source: official Hugging Face config.json and preprocessor_config.json, summarized in config_sweep.json
Source files inspected:
  X:/H/transformers/src/transformers/models/vit_msn/configuration_vit_msn.py
  X:/H/transformers/src/transformers/models/vit_msn/modeling_vit_msn.py
  X:/H/transformers/src/transformers/models/vit_msn/modular_vit_msn.py
  X:/H/transformers/src/transformers/models/vit_msn/convert_msn_to_pytorch.py
  X:/H/transformers/src/transformers/models/vit/image_processing_vit.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/tests/models/vit_msn/test_modeling_vit_msn.py
Any missing files or assumptions: no remote code required; no gated files observed. modeling_vit_msn.py is generated from modular_vit_msn.py, but it is the runtime file inspected for exact operators.
```

Primary source links:

- [modeling_vit_msn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vit_msn/modeling_vit_msn.py)
- [configuration_vit_msn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vit_msn/configuration_vit_msn.py)
- [modular_vit_msn.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vit_msn/modular_vit_msn.py)
- [facebook/vit-msn-small](https://huggingface.co/facebook/vit-msn-small), [base](https://huggingface.co/facebook/vit-msn-base), [large](https://huggingface.co/facebook/vit-msn-large), [base-4](https://huggingface.co/facebook/vit-msn-base-4), [large-7](https://huggingface.co/facebook/vit-msn-large-7)

## 2. High-level architecture

ViT-MSN is an image-only, bidirectional Transformer encoder. First useful DinoML target: `ViTMSNModel` feature extraction and then `ViTMSNForImageClassification` CLS-head logits.

```text
CPU image preprocessing -> NCHW pixel_values
  -> Conv2d patch embedding -> flatten/transposed token sequence
  -> optional mask-token blend for masked image modeling
  -> prepend CLS + add learned absolute position embeddings
  -> N encoder blocks: pre-LN MHA + residual, pre-LN MLP + residual
  -> final LayerNorm
  -> optional classifier(sequence[:, 0, :]) -> logits
```

No text tokenizer, generation loop, KV cache, cross-attention, RoPE, ALiBi, MoE, or recurrent state is involved.

## 3. Important config dimensions

Source defaults from `ViTMSNConfig`:

| Field | Default |
|---|---:|
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| head_dim | inferred as `hidden_size // num_attention_heads` unless a nonstandard `head_dim` attr is injected |
| intermediate_size | 3072 |
| hidden_act | `gelu` |
| hidden_dropout_prob | 0.0 |
| attention_probs_dropout_prob | 0.0 |
| layer_norm_eps | 1e-6 |
| image_size | 224 |
| patch_size | 16 |
| num_channels | 3 |
| qkv_bias | true |

Representative official checkpoint sweep:

| Model id | H | Layers | Heads | Head dim | MLP | Patch | Tokens at 224 | Dropout | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `facebook/vit-msn-small` | 384 | 12 | 6 | 64 | 1536 | 16 | 197 | 0.0 | float32 |
| `facebook/vit-msn-base` | 768 | 12 | 12 | 64 | 3072 | 16 | 197 | 0.0 | float32 |
| `facebook/vit-msn-large` | 1024 | 24 | 16 | 64 | 4096 | 16 | 197 | 0.1 | float32 |
| `facebook/vit-msn-base-4` | 768 | 12 | 12 | 64 | 3072 | 4 | 3137 | 0.0 | float32 |
| `facebook/vit-msn-large-7` | 1024 | 24 | 16 | 64 | 4096 | 7 | 1025 | 0.1 | float32 |

Preprocessor configs are uniform: resize to 224 with resample `2`, rescale/normalize through the ViT image processor path, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, and NCHW `pixel_values` for the model.

## 3a. Family variation traps

- Patch size materially changes attention shape: `p=16` gives 197 tokens, `p=7` gives 1025, and `p=4` gives 3137. The `base-4` variant makes dense attention the dominant risk.
- The source accepts tuple/list `image_size` and `patch_size`; static square-only assumptions should be guarded.
- The source computes `head_dim = getattr(config, "head_dim", hidden_size // heads)`. Standard configs use 64, but a manually injected `head_dim` can make QKV output width differ from `hidden_size`.
- Q/K/V projections are separate HF weights, but `convert_msn_to_pytorch.py` splits original MSN/timm packed `qkv` weights in Q, K, V order.
- `qkv_bias` is configurable for Q/K/V only; output projection and MLP linears always have bias.
- Large variants keep dropout modules active in training. In inference, dropout is disabled but graph import should still reject or fold training mode explicitly.
- Position interpolation is optional. Without `interpolate_pos_encoding`, source rejects images whose runtime H/W differ from config image size.
- `bool_masked_pos` requires construction with `use_mask_token=True`; ordinary pretrained `ViTMSNModel(config)` has no mask token. First inference target can reject masked image modeling.
- Layout trap: model forward consumes NCHW and `Conv2d` is axis-sensitive. Treat NHWC/channel-last as a guarded optimized rewrite only for the processor-to-patch-embedding region.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW input validation on channel axis 1.
- `Conv2d(C -> H, kernel_size=patch, stride=patch, padding=0, groups=1)` for patch embedding.
- `flatten(2)` from `[B,H,Hp,Wp]` to `[B,H,N]`, `transpose(1,2)` to `[B,N,H]`.
- `expand` for CLS token and optional mask token.
- `cat(dim=1)` to prepend CLS.
- Elementwise add for position embeddings and residuals.
- Optional `reshape`, `permute`, bicubic `interpolate`, `view`, `cat` for position embedding interpolation.
- Slice/index `sequence_output[:, 0, :]` for classification.

Neural primitives:

- LayerNorm over last dim, eps `1e-6`.
- Linear with bias:
  - small: Q/K/V `384 -> 384`, O `384 -> 384`, MLP `384 -> 1536 -> 384`.
  - base/base-4: Q/K/V `768 -> 768`, O `768 -> 768`, MLP `768 -> 3072 -> 768`.
  - large/large-7: Q/K/V `1024 -> 1024`, O `1024 -> 1024`, MLP `1024 -> 4096 -> 1024`.
- GELU from `ACT2FN["gelu"]`.
- Classifier head when present: `Linear(hidden_size -> num_labels)`, usually 1000 for ImageNet fine-tuned heads. Official raw configs advertise `ViTMSNModel`, so classifier weights are a separate head target.

Attention primitives:

- Bidirectional self-attention, MHA only, no causal mask and no KV cache.
- Q/K/V reshape to `[B, S, heads, head_dim]`, transpose to `[B, heads, S, head_dim]`.
- Scores `Q @ K^T * head_dim^-0.5`, optional additive mask, softmax over last dim in fp32, cast back to query dtype, then `P @ V`.
- SDPA/Flash/Flex backends are supported by Transformers dispatch, but eager math above is the semantic reference.

Preprocessing-coupled ops:

- CPU/data pipeline owns image decode, RGB conversion when caller uses PIL, resize, rescale, normalize, and channel-first tensor packing.
- DinoML runtime first target should accept already-preprocessed `pixel_values: [B,3,224,224]`.

Gated/missing links:

- DinoML has bounded NCHW `avg_pool` but no general `conv2d` port in the checklist. Patch embedding must be admitted as a guarded `Conv2d -> Linear/GEMM` rewrite or wait for `conv2d`.
- LayerNorm is not listed as ported in the current op checklist. ViT-MSN requires it before useful parity.
- Dense encoder attention needs BMM/GEMM + softmax over `[B*heads,S,S]`; base p=16 is tractable, base-4 is much larger.
- Bicubic interpolate is not in the listed DinoML op surface. First integration should reject `interpolate_pos_encoding=True`.

## 5. Layer/block breakdown

Patch embedding:

```text
pixel_values: [B, C, H_img, W_img] in NCHW
conv: [B, hidden, H_img/patch_h, W_img/patch_w]
flatten+transpose: [B, N_patches, hidden]
optional mask blend: embeddings * (1 - mask) + mask_token * mask
prepend CLS: [B, N_patches + 1, hidden]
add position_embeddings: [1, N_patches + 1, hidden]
```

Encoder block, repeated `num_hidden_layers`:

```text
residual = x
x = LayerNorm(x)
q = Linear(hidden -> heads * head_dim, bias=qkv_bias)(x)
k = Linear(hidden -> heads * head_dim, bias=qkv_bias)(x)
v = Linear(hidden -> heads * head_dim, bias=qkv_bias)(x)
q,k,v = view/transpose to [B, heads, S, head_dim]
a = softmax((q @ k^T) * head_dim^-0.5 + optional_mask, dim=-1, fp32)
x = (a @ v).transpose(1,2).reshape(B,S,heads*head_dim)
x = Linear(heads * head_dim -> hidden, bias=True)(x)
x = x + residual
residual = x
x = LayerNorm(x)
x = Linear(hidden -> intermediate, bias=True)(x)
x = GELU(x)
x = Linear(intermediate -> hidden, bias=True)(x)
x = x + residual
```

Final:

```text
last_hidden_state = LayerNorm(x)
logits = classifier(last_hidden_state[:, 0, :])   # classification head only
```

## 6. Attention requirements

- Type: encoder self-attention, bidirectional, noncausal.
- Heads: standard MHA, no MQA/GQA. Official head_dim is 64 across inspected checkpoints.
- Query/key/value lengths: square attention with `S = num_patches + 1`; no rectangular cross-attention.
- Masking: `create_bidirectional_mask` may return `None` when no padding/additional mask is needed. If a 2D attention mask is supplied, it is converted through Transformers mask utilities into the backend mask format. First DinoML target can require no external `attention_mask`.
- Cache: no autoregressive KV cache.
- Flash/SDPA compatibility: source advertises SDPA, FlashAttention, and Flex support. DinoML can validate eager decomposition first, then add a dense bidirectional attention provider/fusion.

## 7. Position encoding and custom math

Absolute learned position embeddings are added after CLS concatenation. Normal fixed-size inference is just:

```python
embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
embeddings = embeddings + position_embeddings
```

Optional interpolation path:

```python
def interpolate_pos_encoding(pos, embeddings, height, width, patch_size):
    cls = pos[:, :1]
    patch = pos[:, 1:]
    dim = embeddings.shape[-1]
    old = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_size, width // patch_size), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).view(1, -1, dim)
    return cat([cls, patch], dim=1)
```

This should be deferred or handled as a CPU/offline position-table preparation path until DinoML has bicubic interpolate parity.

## 8. Preprocessing and input packing

The model graph expects `pixel_values` already normalized and packed as NCHW. Processor work belongs in the CPU/data pipeline for first integration:

```text
image -> resize 224 -> rescale -> normalize(mean/std) -> [B,3,224,224] float tensor
```

No placeholder tokens, modality token IDs, packed sequence metadata, OCR boxes, tokenizer metadata, or scatter-based multimodal stitching are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern:

```text
Conv2d(C -> H, kernel=(ph,pw), stride=(ph,pw), padding=0, dilation=1, groups=1)
-> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten NCHW patches to [B, N_patches, C*ph*pw]
-> GEMM_RCR/Linear(weight_flat: [H, C*ph*pw], bias: [H])
-> [B, N_patches, H]
```

Preconditions:

- Input is NCHW with channel count equal to `config.num_channels`.
- `height % patch_h == 0` and `width % patch_w == 0`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Flatten order must match PyTorch Conv2d receptive-field order over C, kh, kw.
- Bias is preserved.

Failure cases:

- Non-divisible image size, nonstandard tuple image/patch size not captured in guards, or any future grouped/dilated patch embedding.

Parity test sketch:

- Compare Conv2d path and unfolded/GEMM path for small/base/large and patch sizes 4, 7, 16 on random NCHW tensors.

### Rewrite: QKV separate linears -> packed GEMM

Source pattern:

```text
q = Linear(x, Wq, bq); k = Linear(x, Wk, bk); v = Linear(x, Wv, bv)
```

Replacement:

```text
qkv = Linear(x, concat_rows([Wq, Wk, Wv]), concat([bq, bk, bv]))
split last dim as [q, k, v]
```

Preconditions:

- Same input tensor, same dtype/layout, all Q/K/V biases either present or absent according to `qkv_bias`.
- Split order is Q, K, V. This matches both source forward and converter split semantics.

### Guarded layout rewrite: NCHW patch embedding to NHWC/channel-last

Candidate optimized region:

```text
processor output or runtime input -> patch embedding -> token sequence
```

NHWC rewrite requirements:

- Input axis rewrite `[B,C,H,W] -> [B,H,W,C]`.
- Conv weight transform from `[out,in,kh,kw]` to the provider's expected NHWC/filter layout.
- Flatten order must still produce patches in row-major spatial order and channel/kernel order equivalent to PyTorch.
- Downstream tokens are layout-neutral `[B,S,H]`; the layout translation should end before token sequence ops.

No-layout-translation guards:

- Source channel check uses `pixel_values.shape[1]`.
- Position interpolation uses NCHW `permute(0,3,1,2)` and bicubic interpolate; protect this path unless all axes are rewritten together.
- Attention and MLP operate on `[B,S,H]` and should not inherit image-layout tags.

## 10. Kernel fusion candidates

Highest priority:

- Patch embedding Conv2d-to-GEMM or direct patch kernel, because `conv2d` is currently a DinoML gap and every run needs this stage.
- LayerNorm over `[B,S,H]`, because each block has two plus a final norm.
- Dense bidirectional attention over encoder token lengths, especially p=4 and p=7 variants.
- Linear+GELU+Linear MLP path, using existing GEMM plus fused GELU where possible.

Medium priority:

- QKV packed projection GEMM to reduce three launches and simplify attention input packing.
- Fused attention output reshape+projection handoff.
- CLS pooling slice + classifier GEMM for image classification heads.

Lower priority:

- Position interpolation support, because fixed 224x224 inference does not need it.
- Mask-token blend for masked image modeling, because the public pretrained inference path normally does not construct `use_mask_token=True`.

## 11. Runtime staging plan

Stage 1: parse config and load `ViTMSNModel` weights for small/base p=16. Reject `interpolate_pos_encoding=True`, non-224 image sizes, external attention masks, masked image modeling, and injected nonstandard `head_dim`.

Stage 2: implement patch embedding via guarded Conv2d-to-GEMM rewrite and run embedding-only parity against PyTorch.

Stage 3: add one encoder block parity with LayerNorm, MHA, GELU MLP, residuals, and final LayerNorm.

Stage 4: run full encoder parity for small/base p=16, then add classifier head parity where classifier weights are present.

Stage 5: stage `large`, then p=7 and p=4 variants after dense attention memory/performance is measured.

Stage 6: optimize QKV packing, attention backend, LayerNorm fusion, and patch embedding layout rewrite.

## 12. Parity and validation plan

- Random patch embedding parity: Conv2d source vs DinoML rewrite for `[1,3,224,224]`, `[2,3,224,224]`, patch sizes 4/7/16.
- Position add parity with fixed learned table and no interpolation.
- Single-block parity for small/base/large dims, fp32 tolerance around `rtol=1e-4, atol=1e-4`.
- Attention parity with no mask and with an optional 2D bidirectional mask only after first target is stable.
- Full encoder parity against `facebook/vit-msn-small` on the COCO cats fixture used by Transformers tests.
- Classification parity only for checkpoints with classifier weights; Transformers integration expects logits shape `[1,1000]` and checks a small logits slice for `facebook/vit-msn-small`.
- Reduced precision validation can follow after fp32: use fp16/bf16 tolerances appropriate for accumulated attention and LayerNorm, not bitwise matching.

No DinoML tests were run for this audit.

## 13. Performance probes

- Processor throughput separately from runtime: image decode/resize/normalize images/sec.
- Patch embedding latency and bandwidth for p=16, p=7, p=4.
- Encoder block latency by component: LayerNorm, QKV GEMM, attention score, softmax, AV, output GEMM, MLP.
- Full encoder batch sweep for B=1, 4, 8, 16 at S=197, 1025, 3137.
- Attention memory usage and temporary allocation for p=4; scores are `[B, heads, 3137, 3137]`.
- Backend comparison: eager decomposed BMM/softmax/BMM vs fused dense bidirectional attention provider when available.
- QKV packed vs separate projection launch count and latency.

## 14. Skip/defer list

- Training, loss computation, and gradient checkpointing.
- Dropout behavior in training mode.
- Position embedding bicubic interpolation for non-config image sizes.
- Masked image modeling via `bool_masked_pos` and `mask_token`.
- External attention masks for initial fixed-image feature extraction.
- FlashAttention/FlexAttention-specific backend parity until eager decomposition passes.
- Quantization, packed weights, tensor parallelism, and multi-GPU scheduling.

## 15. Final implementation checklist

- [ ] Parse `ViTMSNConfig` and reject unsupported injected `head_dim`, interpolation, masks, and masked image modeling for stage 1.
- [ ] Load encoder weights, preserving separate Q/K/V logical weights and optional packed-QKV rewrite metadata.
- [ ] Add guarded Conv2d patch embedding admission or Conv2d-to-GEMM rewrite.
- [ ] Implement/validate LayerNorm over last dimension.
- [ ] Implement dense bidirectional MHA for `[B,S,H]`.
- [ ] Implement GELU MLP block and residual adds.
- [ ] Add final LayerNorm and optional CLS classifier head.
- [ ] Add small/base p=16 embedding, block, and full-encoder parity tests.
- [ ] Add p=7 and p=4 shape/performance probes before treating those variants as production-ready.
- [ ] Add guarded NHWC patch-embedding layout rewrite only after NCHW parity is stable.
