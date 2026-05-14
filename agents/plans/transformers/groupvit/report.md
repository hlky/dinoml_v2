# GroupViT Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: `nvidia/groupvit-gcc-yfcc` is the main source-doc checkpoint. `nvidia/groupvit-gcc-redcaps` is the second official converted checkpoint. Tiny/community configs were inspected only to expose config-shape traps.

Config source:

- `https://huggingface.co/nvidia/groupvit-gcc-yfcc/resolve/main/config.json`
- `https://huggingface.co/nvidia/groupvit-gcc-redcaps/resolve/main/config.json`
- `https://huggingface.co/nielsr/groupvit-gcc-yfcc-old/resolve/main/config.json`
- `https://huggingface.co/hf-tiny-model-private/tiny-random-GroupViTModel/resolve/main/config.json`
- `https://huggingface.co/onnx-internal-testing/tiny-random-GroupViTModel-ONNX/resolve/main/config.json`

Source files inspected:

- `X:/H/transformers/src/transformers/models/groupvit/modeling_groupvit.py`
- `X:/H/transformers/src/transformers/models/groupvit/configuration_groupvit.py`
- `X:/H/transformers/src/transformers/models/groupvit/convert_groupvit_nvlab_to_hf.py`
- `X:/H/transformers/tests/models/groupvit/test_modeling_groupvit.py`
- Auto mappings for processor/tokenizer/image processor: `CLIPProcessor`, `CLIPTokenizer`, `CLIPImageProcessor`.
- `X:/H/transformers/src/transformers/models/clip/image_processing_clip.py` for image preprocessing defaults.
- `X:/H/transformers/src/transformers/masking_utils.py` for `create_causal_mask`.

Small source/config snapshots saved under this folder:

- `config_sweep_summary.json`
- `nvidia__groupvit-gcc-yfcc__*.json`
- `nvidia__groupvit-gcc-redcaps__*.json`
- `nielsr__groupvit-gcc-yfcc-old__*.json`
- `hf-tiny-model-private__tiny-random-GroupViTModel__*.json`
- `onnx-internal-testing__tiny-random-GroupViTModel-ONNX__*.json`

Any missing files or assumptions: no gated/401 official GroupViT checkpoint was encountered. Only two official NVIDIA configs are available in the HF search results inspected; the old/tiny/ONNX repos are not production architecture variants. The current native source does not read the checkpoint `vision_config.qkv_bias` field; all `nn.Linear` projections are bias-enabled in source.

## 2. High-level architecture

GroupViT is a CLIP-like image/text dual encoder with an additional hierarchical grouping mechanism in the vision encoder. The first useful DinoML target should be feature extraction and image-text similarity; zero-shot segmentation is a second stage because it requires grouping attentions and post-head mask reconstruction.

Dataflow:

```text
CLIP image preprocessing -> NCHW pixel_values -> conv patch embedding -> grouped vision encoder -> mean pool -> projection -> L2 normalize
CLIP tokenization -> input_ids/attention_mask -> causal text encoder -> EOS pool -> projection -> L2 normalize
normalized features -> text @ image.T -> exp(logit_scale) -> logits_per_text/logits_per_image
optional segmentation -> grouping attentions -> resized grouping maps -> group/text logits -> per-label pixel logits
```

Stage decomposition:

- CPU/data pipeline: CLIP tokenizer; CLIP image resize/center-crop/RGB/rescale/normalize.
- Independently cacheable branch: text embeddings/features for a fixed label vocabulary can be cached before final image-text similarity.
- Vision encoder: patchify NCHW image, run three grouped stages, mean-pool final group tokens.
- Similarity head: two MLP projections with `BatchNorm1d` + ReLU, feature normalization, logit-scale multiply.
- Optional segmentation head: requires `output_attentions=True`, group assignment attentions, bilinear resizing, and group/text matrix products.

## 3. Important config dimensions

Source defaults:

| Field | Text default | Vision default |
|---|---:|---:|
| hidden_size | 256 | 384 |
| num_hidden_layers | 12 | 12, expected `sum(depths)` |
| depths | n/a | `[6, 3, 3]` |
| num_attention_heads | 4 | 6 |
| head_dim | 64 | 64 |
| intermediate_size | 1024 | 1536 |
| max_position_embeddings | 77 | n/a |
| vocab_size | 49408 | n/a |
| image_size | n/a | 224 |
| patch_size | n/a | 16 |
| num_group_tokens | n/a | `[64, 8, 0]` |
| num_output_groups | n/a | `[64, 8, 8]` |
| activation | `quick_gelu` | `gelu` |
| projection_dim | 256 | shared top-level |
| projection_intermediate_dim | 4096 | shared top-level |
| cache support | none | none |

Representative checkpoint sweep:

| Model id | Text shape | Vision shape | Image/patch | Groups | Projection | Notes |
|---|---|---|---|---|---|---|
| `nvidia/groupvit-gcc-yfcc` | 12 layers, H=256, heads=4, MLP=1024, max text=77 | depths `[6,3,3]`, H=384, heads=6, MLP=1536 | 224/16 | `[64,8,0] -> [64,8,8]` | 256 via 4096 | official production config, `torch_dtype=float32` |
| `nvidia/groupvit-gcc-redcaps` | same | same | same | same | same | official production config; operator-identical to YFCC |
| `nielsr/groupvit-gcc-yfcc-old` | same | same | same | same | same | older mirror; missing current processor metadata such as `do_convert_rgb` |
| `hf-tiny-model-private/tiny-random-GroupViTModel` | 5 layers, H=32, heads=4, MLP=37, max text=512 | depths `[6,3,3]`, H=32, heads=4, MLP=37 | 30/2 | `[64,8,0] -> [64,8,8]` | 64 via 4096 | tiny/debug; `num_hidden_layers` does not equal `sum(depths)` for vision in config snapshot |
| `onnx-internal-testing/tiny-random-GroupViTModel-ONNX` | H=32, 5 layers, MLP=37; some fields omitted | H=32, heads=4, MLP=37 | 30/2 | omitted | 64 via 4096 | incomplete/non-authoritative ONNX test config; route through effective source defaults or reject |

## 3a. Family variation traps

- The native source uses separate `q_proj`, `k_proj`, `v_proj` linears with bias. Historical/original checkpoints had packed QKV weights; conversion splits them in Q, K, V order.
- The official configs include `vision_config.qkv_bias`, but the inspected source does not read it. Do not make bias optional based on this field for native source parity.
- Text pooling has a CLIP compatibility branch: if `eos_token_id == 2`, it pools at `input_ids.argmax(dim=-1)`; otherwise it pools at the first explicit EOS position.
- Text attention is causal even though this is not a generation model. There is no KV cache or decode loop.
- Vision input semantics are NCHW. NHWC/channel-last is an optimization only around the local image preprocessing/patch embedding region with guarded axis rewrites.
- The grouping path changes sequence length between stages: 196 patch tokens -> 64 groups -> 8 groups for the official 224/16 setup.
- Segmentation is not a learned decoder head. It reuses group assignments, text embeddings, and bilinear-resized grouping maps.
- `hard_softmax` uses `max` + `scatter_`, and tests mark batching equivalence flaky because tie/index behavior is not stable. Inference needs deterministic tie expectations or tolerance around ties.
- `GroupViTMixerMLP` transposes `[B, tokens, C] -> [B, C, tokens]`, applies linear layers over the token axis, then transposes back. This is not a standard channel MLP.
- `BatchNorm1d` appears in both projection heads. For inference, it is affine + running-stat normalization over `[B, 4096]` or flattened group rows.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW `Conv2d(C=3 -> hidden, kernel=patch, stride=patch)`, then `flatten(2)` and `transpose(1,2)`.
- `reshape`, `view`, `transpose`, `permute`, `contiguous`, `flatten`, `cat(dim=1)`, `expand`, slicing along sequence axis.
- Advanced row gather for text EOS pooling: `last_hidden_state[arange(B), pooled_index]`.
- Reductions: `mean(dim=1)`, `sum(dim=-1, keepdim=True)`, `norm(dim=-1, keepdim=True)`, `argmax(dim=-1)`.
- Optional segmentation: bilinear `interpolate` on `[B, groups, H, W]`.

Neural network primitives:

- Embedding lookup for text token and learned position embeddings.
- Learned vision positional table add; optional bicubic interpolation exists in `GroupViTVisionEmbeddings.interpolate_pos_encoding`, but top-level `GroupViTVisionTransformer.forward` does not expose that flag.
- LayerNorm over hidden dimension.
- Linear with bias for Q/K/V/O, MLP fc1/fc2, assignment projections, and projection MLPs.
- GELU for vision MLPs; QuickGELU for text MLPs.
- BatchNorm1d + ReLU in projection heads.
- Dropout exists but inference should compile it as identity.

Attention primitives:

- Dense MHA via `bmm(q, k.T)`, softmax last dimension, dropout identity, `bmm(attn, v)`.
- Causal self-attention in text branch using a 4D mask from `create_causal_mask`.
- Noncausal self-attention in vision stages; no mask.
- Cross-attention in `GroupViTCrossAttentionLayer`: learned group queries attend to image tokens.
- Assignment attention in `GroupViTAssignAttention`: single-head scaled dot product over `[B, output_groups, input_tokens]`, hard or soft assignment over `dim=-2`, row renormalization over keys, then `attn @ value`.

Preprocessing-coupled ops:

- CLIP image processor: RGB conversion, resize shortest edge to 224 or 30 for tiny configs, center crop, rescale to float, normalize by CLIP mean/std, emit NCHW `pixel_values`.
- CLIP tokenizer: BPE/token IDs with max length 77 for official configs; BOS/EOS layout from CLIP tokenizer.

Scatter/indexed update ops:

- `hard_softmax` uses `zeros_like(...).scatter_(dim, index, 1.0)`. This appears in assignment attention. For inference/eval, it is still active because `get_attn(raw_attn)` defaults `hard=True`; gumbel sampling is disabled outside training.

Postprocessing/structured output:

- Optional segmentation returns `[B_image, B_text, logits_h, logits_w]`, where `logits_h/logits_w` match the resized grouping map, usually input image H/W when called from `forward` with `pixel_values.shape[2:]`.

## 5. Layer/block breakdown

Text branch, official shapes:

```text
input_ids [B_text, S<=77]
token_embedding -> [B_text, S, 256]
position_embedding -> [1, S, 256]
for 12 layers:
  residual = x
  x = LayerNorm(256)(x)
  q,k,v = Linear(256 -> 256, bias=True)(x), split into 4 heads, head_dim=64
  q *= 64^-0.5
  x = causal dense self-attention(q,k,v, attention_mask)
  x = residual + Linear(256 -> 256)(x)
  residual = x
  x = LayerNorm(256)(x)
  x = Linear(256 -> 1024) -> QuickGELU -> Linear(1024 -> 256)
  x = residual + x
x = final LayerNorm(256)
pooled = EOS/argmax row gather -> [B_text, 256]
text_projection = Linear(256 -> 4096) -> BatchNorm1d(4096) -> ReLU -> Linear(4096 -> 256)
```

Vision branch, official shapes:

```text
pixel_values [B_img, 3, 224, 224]
Conv2d(3 -> 384, kernel=16, stride=16) -> [B_img, 384, 14, 14]
flatten + transpose -> [B_img, 196, 384]
LayerNorm(384), add position_embeddings [1,196,384]

Stage 0, depth 6, group_token [1,64,384]:
  concat patch tokens + group tokens -> [B_img,260,384]
  repeat 6 encoder layers with noncausal MHA + GELU MLP
  split x [B_img,196,384], group [B_img,64,384]
  TokenAssign: project 64 group tokens to 64, cross-attend to 196 image tokens, assignment attention -> [B_img,64,384]

Stage 1, depth 3, group_token [1,8,384]:
  project previous 64 groups to 8 via MixerMLP over token axis
  concat current 64 image/group tokens + 8 group tokens -> [B_img,72,384]
  repeat 3 encoder layers
  TokenAssign: 8 output groups from 64 tokens -> [B_img,8,384]

Stage 2, depth 3, no new group token:
  repeat 3 encoder layers over [B_img,8,384]

LayerNorm(384)
pooled = mean(dim=1) -> [B_img,384]
visual_projection = Linear(384 -> 4096) -> BatchNorm1d(4096) -> ReLU -> Linear(4096 -> 256)
```

Similarity and optional segmentation:

```text
image_embeds = normalize(visual_projection(image_pool))
text_embeds = normalize(text_projection(text_pool))
logits_per_text = text_embeds @ image_embeds.T * exp(logit_scale)
logits_per_image = logits_per_text.T

if output_segmentation:
  image_group_embeds = visual_projection(last_hidden_state.reshape(-1, 384))
  grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])
  logits_per_image_group = image_group_embeds @ text_embeds.T * exp(logit_scale)
  seg_logits = (logits_per_image_group reshaped/permuted) @ flatten(grouping)
  seg_logits = reshape to [B_img, B_text, H, W]
```

## 6. Attention requirements

Text attention:

- Causal self-attention, MHA, 4 heads, head_dim 64 in official configs.
- Query, key, value width all equal `hidden_size`.
- Mask is created by `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values=None)`.
- No KV cache, no decode path, no generation controller.
- SDPA/FlashAttention could replace the explicit bmm-softmax-bmm if it preserves query scaling before matmul and the 4D causal/padding mask semantics.

Vision self-attention:

- Noncausal dense self-attention, MHA, 6 heads, head_dim 64 in official configs.
- Sequence lengths vary by stage: 260, 72, and 8 tokens in the official shape path.
- No attention mask.

Vision cross-attention:

- `GroupViTCrossAttentionLayer` uses group tokens as queries and image/group tokens as keys/values through the same MHA module.
- Query length can differ from key/value length, e.g. 64 queries over 196 keys in stage 0 and 8 queries over 64 keys in stage 1.

Assignment attention:

- Custom dense single-head attention, not standard MHA.
- `raw_attn = (q @ k.T) * hidden_size^-0.5`, shape `[B, output_groups, input_tokens]`.
- In eval, `attn = hard_softmax(raw_attn, dim=-2)` and `soft_attn = softmax(raw_attn, dim=-2)`.
- Hard assignment normalizes over the key/input-token dimension after hardmax: `attn = attn / (attn.sum(dim=-1, keepdim=True) + assign_eps)`.
- Output is `attn @ value`, then a projection.
- `soft_attn` is returned as the grouping tensor for segmentation. It is softmax over groups (`dim=-2`), so downstream grouping composition depends on this exact orientation.

## 7. Position encoding and custom math

Text position encoding is learned absolute position embedding `[max_position_embeddings, text_hidden]`.

Vision position encoding is learned absolute patch position embedding `[1, num_patches, vision_hidden]`. Optional interpolation code exists:

```python
def interpolate_groupvit_pos(pos, height, width, patch_size):
    npos, dim = pos.shape[1], pos.shape[-1]
    side = int(npos ** 0.5)
    pos = pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    pos = bicubic_interpolate(pos, size=(height // patch_size, width // patch_size), align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
```

Custom math snippets:

```python
def quick_gelu(x):
    return x * sigmoid(1.702 * x)

def hard_softmax_eval(logits, dim):
    y_soft = softmax(logits, dim)
    index = argmax(y_soft, dim=dim, keepdim=True)
    y_hard = zeros_like(logits).scatter(dim, index, 1.0)
    return y_hard - stop_gradient(y_soft) + y_soft

def grouping_from_attentions(attentions, hw):
    prev = None
    for attn in attentions:
        attn = permute(attn, [0, 2, 1])  # [B, input_groups_or_patches, output_groups]
        prev = attn if prev is None else prev @ attn
        cur = resize_attention_map(permute(prev, [0, 2, 1]), hw)
    return cur
```

For inference without gradients, `hard_softmax_eval` numerically behaves like one-hot argmax plus renormalization, while `soft_attn` remains true softmax and is needed for segmentation.

## 8. Preprocessing and input packing

Image preprocessing:

- Processor class: `CLIPProcessor`; image processor maps to `CLIPImageProcessor`.
- Official configs use resize/crop size 224; tiny configs use 30.
- Resize uses bicubic (`resample=3`) and CLIP defaults.
- RGB conversion is enabled in current official configs.
- Normalize mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`.
- Runtime tensor is NCHW `[B,3,H,W]`.

Text preprocessing:

- Tokenizer class: `CLIPTokenizer`.
- Official max length is 77.
- Tokenizer config from the official repos uses start/end text tokens and `pad_token` mapped to end-of-text in tokenizer metadata, while the model config has `pad_token_id=1`, `bos_token_id=0`, `eos_token_id=2` inherited from old CLIP conversion behavior.
- GPU graph consumes `input_ids`, optional `attention_mask`, and optional `position_ids`.

Dual-encoder cacheability:

- Text label embeddings can be precomputed/cached after `text_projection` + L2 normalization for a fixed label set.
- Image embeddings can be cached before similarity if the same image is scored against multiple vocabularies.
- Segmentation additionally needs grouping attentions from the image branch and text embeddings for the selected labels.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d to Linear

Source pattern:

```text
Conv2d(C -> H, kernel=(P,P), stride=(P,P), padding=0) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten_NCHW_to_tokens([B,C,H_img,W_img], P) -> GEMM([B*Npatch, C*P*P] x [C*P*P,H]) + bias -> [B,Npatch,H]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- input height/width match config unless the interpolation path is explicitly enabled
- height and width divisible by patch size
- flatten order must match PyTorch Conv2d NCHW window order

Failure cases: dynamic non-divisible spatial sizes, channel-last input without a full axis rewrite, nonzero padding, grouped convolution.

Parity test sketch: compare Conv2d path and lowered GEMM path for random `[1,3,224,224]` and `[B,3,30,30]` tiny configs, including bias.

### Rewrite: QKV separate Linear fusion

Source pattern:

```text
q = Linear(x); k = Linear(x or encoder); v = Linear(x or encoder)
```

Replacement:

```text
single packed GEMM -> split [q,k,v]
```

Preconditions:

- self-attention: q/k/v share the same input tensor and hidden width.
- cross-attention: k/v share encoder input, q uses query input; only fuse K/V unless a two-input grouped GEMM is available.
- packed weight row order should be Q, K, V to match converter split order for historical checkpoints.

Failure cases: cross-attention query/key source mismatch; source with custom per-projection bias toggles; non-contiguous packed output unsupported.

### Rewrite: projection head BatchNorm1d folding

Source pattern:

```text
Linear(in -> 4096) -> BatchNorm1d(4096, eval) -> ReLU -> Linear(4096 -> projection_dim)
```

Replacement:

```text
LinearFoldedBN -> ReLU -> Linear
```

Preconditions:

- inference/eval mode with frozen running mean/variance and affine parameters.
- input rank is `[B, in]` or flattened `[B*groups, in]`.

Failure cases: training mode, missing running stats, dynamic behavior from unfrozen BN.

### Rewrite: segmentation grouping composition

Source pattern:

```text
attn0 [B,64,196], attn1 [B,8,64]
permute/contiguous, matmul chain, reshape, bilinear interpolate
```

Replacement:

```text
batched matmul composition -> layout-aware resize -> segmentation matmul
```

Preconditions:

- attentions are emitted from assignment attention and preserve `[B, output_groups, input_tokens]`.
- patch grid can be recovered from `pixel_values.shape` and attention length.

Failure cases: missing `output_attentions`, non-square or unusual aspect ratio needs the exact `resize_attention_map` scale heuristic, arbitrary attention tuple lengths.

### Layout rewrite: guarded NCHW-to-NHWC patch region

Candidate region: image processor output through patch embedding only.

Required axis rewrites:

- Conv2d input `[B,C,H,W]` to NHWC `[B,H,W,C]`.
- Flatten patch-grid order must still emit `[B, H/P * W/P, hidden]`.
- Later sequence ops are layout-neutral `[B,N,C]`; do not propagate image NHWC assumptions beyond patchification.

No-layout-translation guards:

- `resize_attention_map` and segmentation logits operate on `[B, groups, H, W]` NCHW-like maps and call bilinear interpolate. Keep these axes explicit unless a full segmentation layout rewrite is implemented.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d-to-GEMM or optimized patchify+GEMM. It is the only spatial convolution and is easy to guard.
- LayerNorm + QKV projection + attention for short fixed sequence lengths. Vision has stage-specific small sequence sizes; text has S<=77.
- Assignment attention kernel: argmax/scatter/renormalize + `attn @ value` is unusual and likely graph-fragmenting.
- Projection head BN folding and Linear/ReLU/Linear fusion.

Medium priority:

- QuickGELU/GELU MLP fusion for encoder FFNs.
- L2 normalization + similarity matmul + logit-scale multiply for dual-encoder scoring.
- Grouping composition and segmentation matmul fusion for zero-shot segmentation.

Lower priority:

- Bicubic positional interpolation. Top-level source does not expose it in normal forward; keep as optional.
- Training-only gumbel softmax and contrastive loss.
- Dropout, because inference treats it as identity.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `GroupViTModel`, rejecting incomplete ONNX-style configs unless effective defaults can be reproduced.

Stage 2: implement text branch parity only: embeddings, causal mask, 12-layer encoder, EOS pooling, text projection.

Stage 3: implement vision branch parity without segmentation: NCHW patch embedding, position add, three grouped stages, assignment attention, mean pool, visual projection.

Stage 4: implement dual-encoder similarity: L2 normalize, `exp(logit_scale)`, logits orientation `[B_img, B_text]` and `[B_text, B_img]`.

Stage 5: add optional segmentation: capture assignment attentions, reconstruct grouping, resize maps, compute per-label segmentation logits.

Stage 6: enable guarded rewrites/fusions: patch Conv2d-to-GEMM, QKV fusion, BN folding, attention kernels, assignment-attention kernel.

Stage 7: production validation: cache text label features, benchmark image encoder and segmentation separately.

Initially stub/defer: training loss, gumbel sampling, gradient checkpointing, `return_loss`, model-card pipelines beyond feature extraction/segmentation.

## 12. Parity and validation plan

- Config parser tests for official YFCC/RedCaps and tiny configs; assert native source ignores `qkv_bias`.
- Random text branch parity against PyTorch for `[B, S]` with `S=7`, `S=77`, with and without attention mask.
- Text pooling tests for `eos_token_id==2` argmax behavior and non-2 first-EOS behavior.
- Random vision branch parity for tiny `[B,3,30,30]`, patch 2, H=32, including stage output shape `[B,8,32]`.
- Official-shape vision smoke parity for `[B,3,224,224]` through patch embedding and one stage, then full branch if memory allows.
- Assignment attention tests: hard one-hot path, soft attention return shape, `assign_eps` renormalization.
- Dual-encoder parity: logits orientation and `logit_scale.exp()`.
- Segmentation parity: compare `get_grouping_from_attentions`, bilinear resize, and final segmentation logits for synthetic attentions/text embeddings.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5` for branch internals; fp16/bf16 relaxed to `rtol=1e-2, atol=1e-2`, with stricter fp32 accumulation for softmax/matmul reductions where possible.

## 13. Performance probes

- CLIP image preprocessing throughput vs model runtime.
- Text encoder throughput for label vocabulary batches; measure feature cache hit path separately.
- Vision encoder throughput by stage: patch embedding, stage 0, stage 1, stage 2.
- Assignment attention microbenchmarks for `[B,64,196]` and `[B,8,64]`.
- Dense attention backend comparison for sequence lengths 260, 72, 77, and 8.
- Projection head BN-folded vs unfused runtime.
- Similarity matrix sweep over `B_img x B_text`, including large text-label vocabularies.
- Segmentation path benchmark: grouping reconstruction, bilinear resize to 224, final per-label matmul.
- Layout experiment: NCHW Conv2d baseline vs guarded NHWC/patchify+GEMM implementation.

## 14. Skip/defer list

- Training, contrastive loss, and gradient checkpointing.
- Gumbel-softmax sampling path; inference uses hard/soft deterministic paths.
- General image sizes with position interpolation, unless explicitly admitted with bicubic positional interpolation parity.
- General boolean scatter; only the bounded hard-softmax scatter pattern is needed.
- KV cache, decode, beam search, generation controllers: not applicable.
- Multi-GPU/tensor parallel and quantized packed weights: no source-coupled quantized format in native GroupViT.
- ONNX-internal incomplete config support, unless source-default filling is explicitly part of the loader.

## 15. Final implementation checklist

- [ ] Parse `GroupViTConfig`, `GroupViTTextConfig`, and `GroupViTVisionConfig`.
- [ ] Load official split Q/K/V weights and projection-head BatchNorm state.
- [ ] Implement CLIP tokenizer/image processor ABI handoff or document CPU pipeline ownership.
- [ ] Implement text embeddings, causal mask, text encoder, final norm, and EOS pooling.
- [ ] Implement NCHW patch embedding and learned vision position add.
- [ ] Implement vision grouped stages, including group-token expand/project and sequence concat/split.
- [ ] Implement dense MHA self-attention and cross-attention.
- [ ] Implement `GroupViTAssignAttention` hard assignment and soft grouping output.
- [ ] Implement GELU, QuickGELU, LayerNorm, BatchNorm1d inference, L2 norm.
- [ ] Implement dual projection heads and similarity logits with correct output orientation.
- [ ] Add optional segmentation grouping reconstruction and bilinear resize.
- [ ] Add guarded Conv2d patch embedding -> GEMM rewrite.
- [ ] Add QKV/KV fusion rewrite with converter-compatible Q,K,V order.
- [ ] Add BatchNorm1d folding for projection heads.
- [ ] Add parity tests for tiny random configs and official YFCC/RedCaps configs.
- [ ] Benchmark text-cache, vision encoder, assignment attention, similarity, and segmentation paths separately.
