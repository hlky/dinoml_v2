# ChineseCLIP family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: OFA-Sys/chinese-clip-vit-base-patch16 plus ViT-L/14, ViT-L/14-336px, ViT-H/14 variants
Config source: Hugging Face config/preprocessor files snapshotted under agents/plans/transformers/chinese_clip/_sources/
Source files inspected:
  transformers/src/transformers/models/chinese_clip/modeling_chinese_clip.py
  transformers/src/transformers/models/chinese_clip/modular_chinese_clip.py
  transformers/src/transformers/models/chinese_clip/configuration_chinese_clip.py
  transformers/src/transformers/models/chinese_clip/processing_chinese_clip.py
  transformers/src/transformers/models/chinese_clip/image_processing_chinese_clip.py
  transformers/src/transformers/models/chinese_clip/image_processing_pil_chinese_clip.py
Any missing files or assumptions:
  No tokenizer implementation is specific to chinese_clip; official repos expose vocab.txt and rely on BERT-style tokenization through AutoTokenizer.
  The generated modeling/config files state they are generated from modular_chinese_clip.py. Runtime facts below use modeling_chinese_clip.py because it is the actual imported native PyTorch implementation at the pinned commit; future upstream source edits should target modular_chinese_clip.py.
  OFA-Sys/chinese-clip-rn50 has no native config.json/preprocessor_config.json in the HF repo, only clip_cn_rn50.pt plus README, so the ResNet checkpoint is out of scope for this native ViT chinese_clip report.
```

Primary URLs:

- [modeling_chinese_clip.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/chinese_clip/modeling_chinese_clip.py)
- [configuration_chinese_clip.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/chinese_clip/configuration_chinese_clip.py)
- [processing_chinese_clip.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/chinese_clip/processing_chinese_clip.py)
- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)
- Out of scope RN50 repo: [OFA-Sys/chinese-clip-rn50](https://huggingface.co/OFA-Sys/chinese-clip-rn50)

No gated or 401/403 checkpoint was encountered. Missing `tokenizer_config.json` and `special_tokens_map.json` on the ViT repos returned 404; `vocab.txt` is present.

## 2. High-level architecture

ChineseCLIP is a non-generation dual encoder for contrastive image-text inference:

```text
text ids/masks/token types -> BERT-like text encoder -> first-token pool -> text projection -> L2 normalize
RGB image preprocessing -> ViT patch encoder -> CLS pool -> vision projection -> L2 normalize
normalized text/image embeddings -> text @ image.T -> exp(logit_scale) multiply -> logits_per_text/logits_per_image
```

Runtime target for first DinoML integration: inference-only contrastive retrieval / zero-shot image classification. There is no decoder, no autoregressive prefill/decode path, no KV cache, and no generation controller.

Stage decomposition:

- CPU/data pipeline: BERT WordPiece tokenization from `vocab.txt`; image resize/rescale/normalize/RGB conversion.
- Text encoder: independently stageable and independently cacheable text embedding branch.
- Vision encoder: independently stageable and independently cacheable image embedding branch.
- Projection/similarity head: small shared contrastive head consuming cached branch embeddings; output orientation matters.
- Training-only optional path: `return_loss=True` computes symmetric cross entropy and can be deferred.

## 3. Important config dimensions

Source defaults from `configuration_chinese_clip.py` differ from official checkpoints in a few practical ways: source text default vocab is `30522`, while official ViT checkpoints use `21128`; source vision default patch size is `32`, while official ViT-B uses `16` and larger checkpoints use `14`.

| Field | Source default | Official checkpoint behavior |
| --- | ---: | --- |
| `model_type` | `chinese_clip` | same |
| `projection_dim` | 512 | 512, 768, or 1024 |
| `logit_scale_init_value` | 2.6592 | 2.6592 |
| text `vocab_size` | 30522 | 21128 |
| text `max_position_embeddings` | 512 | 512 |
| text `hidden_act` | `gelu` | `gelu` |
| text `layer_norm_eps` | 1e-12 | 1e-12 |
| text `type_vocab_size` | 2 | 2 |
| vision `hidden_act` | `quick_gelu` | `quick_gelu` |
| vision `layer_norm_eps` | 1e-5 | 1e-5 |
| vision `attention_dropout` | 0.0 | 0.0 |
| cache support | none | none |

Representative checkpoint sweep:

| Model id | Text H/L/A/FFN | Text head dim | Vision H/L/A/FFN | Vision head dim | Image / patch | Tokens image | Projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `OFA-Sys/chinese-clip-vit-base-patch16` | 768 / 12 / 12 / 3072 | 64 | 768 / 12 / 12 / 3072 | 64 | 224 / 16 | 197 | 512 |
| `OFA-Sys/chinese-clip-vit-large-patch14` | 768 / 12 / 12 / 3072 | 64 | 1024 / 24 / 16 / 4096 | 64 | 224 / 14 | 257 | 768 |
| `OFA-Sys/chinese-clip-vit-large-patch14-336px` | 768 / 12 / 12 / 3072 | 64 | 1024 / 24 / 16 / 4096 | 64 | 336 / 14 | 577 | 768 |
| `OFA-Sys/chinese-clip-vit-huge-patch14` | 1024 / 24 / 16 / 4096 | 64 | 1280 / 32 / 16 / 5120 | 80 | 224 / 14 | 257 | 1024 |

The table values come from snapshotted `config.json` files. Dtype in those configs is `torch_dtype: "float32"`; mixed precision is an inference/runtime policy, not a checkpoint config requirement.

## 3a. Family variation traps

- Native source covers ViT-backed ChineseCLIP only. The official RN50 `.pt` repo is a different architecture and has no native `config.json`, so route it to a separate audit.
- `text_config_dict` / `vision_config_dict` are backward-compatibility mirrors. The config constructor updates `text_config` / `vision_config` from them, so loaders must respect those aliases if parsing raw JSON.
- Many generation-like fields appear inside checkpoint `vision_config` because it serialized a generic `PretrainedConfig`; native source does not use `num_beams`, `max_length`, `is_decoder`, `return_dict_in_generate`, etc. Treat them as ignored for this source basis.
- Official preprocessor configs use old `feature_extractor_type: ChineseCLIPFeatureExtractor`, `do_center_crop: false`, and `size: {"height": ..., "width": ...}`. Current source image processor class defaults say center crop true and `size: {"shortest_edge": 224}`. For checkpoint parity, prefer checkpoint preprocessor JSON over class defaults.
- `projection_dim` is not equal to both encoder hidden sizes in all variants. ViT-H uses text hidden 1024, vision hidden 1280, projection 1024.
- Text pooling for `ChineseCLIPModel` is first-token hidden state before projection, not the `ChineseCLIPTextPooler` tanh head and not EOS/EOT pooling.
- Vision pooling is CLS token after `post_layernorm`.
- Source tensor layout for images is NCHW `pixel_values`; NHWC is only an optimization candidate inside a guarded conv/patch region.
- Attention is noncausal MHA in both branches. No GQA/MQA, RoPE, ALiBi, local attention, sliding window, packed varlen metadata, or KV cache.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather for word, text position, token type, and vision position embeddings.
- Text position id slice `position_ids[:, :seq_length]`; token type buffer slice/expand.
- Add, residual add, concat along sequence dim, flatten from NCHW conv output, transpose/permute, reshape/view, contiguous.
- First-token/CLS indexing `[:, 0, :]`.
- Matrix transpose for `image_embeds.T` and output transpose for `logits_per_image`.

Neural network primitives:

- NCHW `Conv2d(C=3 -> H_v, kernel=patch_size, stride=patch_size, bias=False)` for patch embedding.
- Bias Linear layers in attention, output, and MLP blocks.
- Bias-free Linear projection heads: text `H_t -> projection_dim`, vision `H_v -> projection_dim`.
- LayerNorm with eps 1e-12 for text and 1e-5 for vision.
- Activations: `gelu` in text FFN, `quick_gelu` in vision FFN, `tanh` only for standalone `ChineseCLIPTextModel` pooler with `add_pooling_layer=True`.
- L2 norm via `pow(x, 2) -> sum(dim=-1, keepdim=True) -> pow(..., 0.5) -> div`.
- `exp(logit_scale)` scalar multiply.

Attention primitives:

- Dense noncausal scaled dot-product self-attention with query/key/value layout `[B, heads, S, head_dim]`.
- Mask add before softmax for text attention when `attention_mask` is present.
- Softmax over last dimension with fp32 softmax dtype, cast back to query dtype in eager path.
- Attention dropout is source-present but 0.0 in inference and official vision configs.

Preprocessing-coupled ops:

- CPU tokenizer produces `input_ids`, optional `attention_mask`, optional `token_type_ids`; missing token types default to zeros.
- CPU image processor produces float NCHW `pixel_values` after RGB conversion, resize, rescale, normalize.
- Checkpoint preprocessor `do_center_crop=false` means no crop for official ViT repos unless caller overrides.

Generation/cache ops:

- None required. Reject or ignore generation fields for this target.

## 5. Layer/block breakdown

Text embeddings:

```text
word = Embedding(input_ids)                       # [B, T, Ht]
type = Embedding(token_type_ids or zeros)         # [B, T, Ht]
pos = Embedding(position_ids or arange[:T])       # [1 or B, T, Ht]
x = LayerNorm(word + type + pos, eps=1e-12)
```

Text encoder block, repeated `text_config.num_hidden_layers`:

```text
q = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,D).transpose(1,2)
k = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,D).transpose(1,2)
v = Linear(Ht -> Ht, bias=True)(x).view(B,T,A,D).transpose(1,2)
a = softmax((q @ k.transpose(-2,-1)) * D**-0.5 + mask, dim=-1, fp32).to(q.dtype)
x_attn = (a @ v).transpose(1,2).reshape(B,T,Ht)
x = LayerNorm(Linear(Ht -> Ht)(x_attn) + residual, eps=1e-12)
ffn = GELU(Linear(Ht -> It)(x))
x = LayerNorm(Linear(It -> Ht)(ffn) + residual, eps=1e-12)
```

ChineseCLIPModel text feature path:

```text
pooled = last_hidden_state[:, 0, :]       # not tanh pooler
text_features = Linear(Ht -> P, bias=False)(pooled)
```

Vision embeddings:

```text
patch = Conv2d(3 -> Hv, kernel=Pch, stride=Pch, bias=False)(pixel_values)  # [B,Hv,Gh,Gw]
patch = patch.flatten(2).transpose(1, 2)                                   # [B,Gh*Gw,Hv]
cls = class_embedding.expand(B,1,Hv)
x = concat([cls, patch], dim=1) + position_embedding                       # [B,1+Gh*Gw,Hv]
```

Vision encoder block, repeated `vision_config.num_hidden_layers`:

```text
y = LayerNorm(x, eps=1e-5)
q,k,v = Linear(Hv -> Hv, bias=True)(y) separately
y = dense noncausal MHA(q,k,v)
x = x + Linear(Hv -> Hv, bias=True)(y)
y = LayerNorm(x, eps=1e-5)
y = Linear(Iv -> Hv)(quick_gelu(Linear(Hv -> Iv)(y)))
x = x + y
```

Vision feature path:

```text
pooled = post_layernorm(last_hidden_state[:, 0, :])
image_features = Linear(Hv -> P, bias=False)(pooled)
```

Contrastive head:

```text
image_embeds = image_features / sqrt(sum(image_features**2, dim=-1, keepdim=True))
text_embeds = text_features / sqrt(sum(text_features**2, dim=-1, keepdim=True))
logits_per_text = (text_embeds @ image_embeds.T) * exp(logit_scale)
logits_per_image = logits_per_text.T
```

## 6. Attention requirements

Both text and vision use dense noncausal self-attention:

| Branch | Self/cross | Causal | Heads | KV heads | Head dim | Mask |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Text | self | no | config `num_attention_heads` | same as Q heads | `hidden_size / heads` | bidirectional attention mask from `create_bidirectional_mask` |
| Vision | self | no | config `num_attention_heads` | same as Q heads | `hidden_size / heads` | normally none |

The eager attention math is:

```python
scores = torch.matmul(q, k.transpose(2, 3)) * scaling
if attention_mask is not None:
    scores = scores + attention_mask
probs = softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
out = torch.matmul(probs, v).transpose(1, 2).contiguous()
```

Transformers can dispatch through `ALL_ATTENTION_FUNCTIONS` via `config._attn_implementation`, so SDPA/FlashAttention-style backends may be used by PyTorch. DinoML parity should start with the eager math and then admit optimized attention only when it preserves noncausal mask semantics, fp32 softmax behavior, and `[B, H, S, D]` head layout.

KV cache requirements: none. No cached keys/values are accepted or returned. Branch embeddings after projection/normalization can be cached independently by application code for retrieval, but these are feature caches, not attention caches.

## 7. Position encoding and custom math

Text uses learned absolute position embeddings with default `position_ids = arange(max_position_embeddings)[:T]`.

Vision uses learned absolute position embeddings over `[CLS] + patch_grid`. For same-size inference, the table is gathered by fixed `position_ids`. If `interpolate_pos_encoding=True`, source interpolates patch positions:

```python
def chinese_clip_interpolate_pos(pos_weight, embeddings, height, width, patch_size):
    cls = pos_weight[:1]
    patch = pos_weight[1:]
    dim = embeddings.shape[-1]
    side = int((patch.shape[0]) ** 0.5)
    patch = patch.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    patch = interpolate(patch, size=(height // patch_size, width // patch_size),
                        mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return cat([cls.reshape(1, 1, dim), patch], dim=1)
```

First integration can require exact checkpoint image size and `interpolate_pos_encoding=False`. Interpolation needs bicubic resize, reshape/permute, and dynamic grid guards.

Custom math: L2 feature normalization intentionally avoids `torch.norm` in source and is implemented as `pow -> sum -> pow(0.5)` for exportability.

## 8. Preprocessing and input packing

Text branch:

- Tokenization is BERT-style WordPiece from `vocab.txt`; no chinese_clip-specific tokenizer code exists.
- Official repos lack `tokenizer_config.json` and `special_tokens_map.json`, so AutoTokenizer behavior comes from model/config and vocab conventions.
- Runtime tensors: `input_ids: int64 [B,T]`, `attention_mask: int/bool/float [B,T]` accepted by masking utility, optional `token_type_ids: int64 [B,T]`, optional `position_ids: int64 [1,T] or [B,T]`.
- Missing token type IDs become all zeros. Missing position IDs become `arange[:T]`.
- Pooling for contrastive model uses `last_hidden_state[:, 0, :]`, so CLS-first tokenizer layout is part of parity.

Image branch:

- Official checkpoint preprocessor JSON: `do_resize=true`, `do_normalize=true`, `do_center_crop=false`, `resample=3` (PIL bicubic), CLIP mean `[0.48145466, 0.4578275, 0.40821073]`, CLIP std `[0.26862954, 0.26130258, 0.27577711]`.
- Size is 224x224 for base/large/huge and 336x336 for large-336px.
- Runtime tensor: `pixel_values: float [B,3,H,W]` in NCHW.
- Source image processor class defaults include RGB conversion, rescale, normalize, center crop, and shortest-edge size, but the checkpoint preprocessor overrides center crop and uses explicit height/width. DinoML end-to-end parity should snapshot and honor the checkpoint preprocessor config.

Dual-encoder contract:

- `get_text_features` returns a `BaseModelOutputWithPooling` whose `pooler_output` has projected text features `[B_text, P]`, not normalized.
- `get_image_features` returns projected image features `[B_image, P]`, not normalized.
- `forward` normalizes both features and returns `logits_per_text [B_text, B_image]`; `logits_per_image [B_image, B_text]`.
- Text and image projected or normalized embeddings can be cached independently before final similarity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap ViT Conv2d patch embedding -> Linear

Source pattern:

```text
Conv2d(3 -> H, kernel_size=patch, stride=patch, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1,2)
```

Replacement:

```text
NCHW WindowFlatten in row-major patch order -> MatMul(W_flat.T) -> Reshape [B, Gh*Gw, H]
```

Preconditions:

- Input is NCHW.
- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- `height % patch_size == 0` and `width % patch_size == 0`.
- Weight transform: `w_flat = conv.weight.reshape(H, 3 * patch * patch)`.
- Flatten order must match PyTorch NCHW convolution window order and subsequent `flatten(2).transpose(1,2)`.

Failure cases: non-square or dynamic input sizes without patch divisibility guards; `interpolate_pos_encoding=False` still requires config image size equality.

Parity test sketch: compare patch tokens before class concat for random image and one checkpoint weight tensor in fp32.

### Rewrite: Q/K/V separate linear -> packed QKV GEMM

Source pattern: three independent bias linears from the same hidden state.

Replacement: one GEMM with concatenated output columns, then split `[q, k, v]` in that exact order.

Preconditions:

- All three inputs are identical and contiguous logical `[B,S,H]`.
- All q/k/v projections have same `in_features == out_features == hidden_size`.
- Biases preserved and split in `[q, k, v]`.
- Text source module names are `query`, `key`, `value`; vision names are `q_proj`, `k_proj`, `v_proj`.

Failure cases: checkpoints with missing bias or unequal dimensions, not observed in inspected configs.

### Rewrite: retrieval similarity GEMM

Source pattern:

```text
logits_per_text = text_embeds @ image_embeds.T
scale = exp(logit_scale)
logits_per_image = logits_per_text.T
```

Replacement:

```text
GEMM_RCR(text_embeds, image_embeds) with scalar epilogue multiply
optional materialized transpose only if caller requests logits_per_image
```

Preconditions: both embeddings already L2-normalized; output orientation must preserve `[B_text, B_image]` for `logits_per_text`.

### Layout candidate: NCHW patch-only region to NHWC

Source semantics are NCHW. A guarded layout pass may translate only the local patch embedding region if it also rewrites:

- Conv input channel axis from `dim=1` to `dim=-1`.
- Patch/window flatten order and weight layout.
- Output back to token-major `[B, patches, H]` before class concat.

Protect text embeddings, sequence attention, LayerNorm, and similarity head with no-layout-translation guards unless a larger sequence-layout lowering is explicitly implemented.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual add patterns in both encoders; text is post-norm after residual, vision is pre-norm plus residual.
- Dense noncausal attention: QKV packed projection, attention softmax in fp32, output projection.
- GEMM + bias + GELU/quick_gelu for FFNs.
- Conv patch embedding lowered to GEMM for ViT variants.
- L2 normalization and final scaled similarity GEMM, especially for large gallery batches.

Medium priority:

- Embedding sum + LayerNorm for text.
- Class token concat + position add for vision.
- Bias-free projection heads with adjacent L2 normalization.
- Optional materialized transpose elimination when only one logits orientation is consumed.

Lower priority:

- Bicubic position interpolation path; useful for dynamic image sizes but not required for fixed checkpoint parity.
- Training contrastive loss.
- Standalone `ChineseCLIPTextModel` tanh pooler if DinoML targets text model alone.

## 11. Runtime staging plan

Stage 1: Parse raw `ChineseCLIPConfig`, including backward-compatible `text_config_dict` and `vision_config_dict`; reject RN50/no-config repos for this target.

Stage 2: Load weights for one branch and run single-block parity for text and vision independently.

Stage 3: Implement full text encoder parity with CLS pooling and text projection. Tokenization can remain CPU-side.

Stage 4: Implement full vision encoder parity at fixed image size with NCHW patch embedding and no position interpolation.

Stage 5: Implement contrastive head: projection outputs, L2 normalize, scaled similarity, both output orientations.

Stage 6: Add optimized lowering: patch conv->linear, packed QKV, fused LayerNorm/residual/MLP, optimized noncausal attention.

Stage 7: Add optional image-position interpolation and end-to-end processor parity if DinoML owns preprocessing.

Stubs acceptable initially: `return_loss`, dropout/training, output attentions/hidden states, position interpolation, standalone text pooler, and all generation fields.

## 12. Parity and validation plan

- Config parser tests for all four snapshotted ViT configs, including token vocab size, image/patch size, projection dim, and ignored generic generation fields.
- Random tensor unit tests for `_get_vector_norm` equivalent with fp32/fp16 tolerances.
- Patch embedding parity: Conv2d path and lowered GEMM path on fixed NCHW random input.
- Single text layer parity with attention mask, token type IDs omitted and explicit zeros.
- Single vision layer parity with no mask.
- Full branch parity: `get_text_features` and `get_image_features` against Transformers in fp32.
- End-to-end contrastive parity: batched text/image logits, checking both `logits_per_text` and transposed `logits_per_image`.
- Processor parity if included: compare `pixel_values` from checkpoint preprocessor JSON for 224 and 336 variants.

Suggested tolerances: fp32 absolute/relative `1e-4`; fp16/bf16 branch parity `1e-2` after attention and FFN; logits tolerance should include scale multiply sensitivity from `exp(logit_scale)`.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- CPU preprocessing throughput: tokenizer and image resize/normalize separately.
- Text encoder throughput sweep over `B_text` and sequence length up to 512.
- Vision encoder throughput for 224/16, 224/14, and 336/14 token counts.
- Attention backend comparison by branch and sequence length.
- Similarity head sweep: `B_text x B_image` GEMM sizes for retrieval/gallery workloads.
- Feature-cache workflow: cached image embeddings plus fresh text queries; cached text embeddings plus fresh image batches.
- Patch embedding Conv2d vs lowered GEMM comparison.
- Memory probe for ViT-L/336 attention activations: 577 vision tokens substantially increase attention matrix size.

## 14. Skip/defer list

- Training and `return_loss=True` contrastive loss.
- Dropout and gradient checkpointing.
- Output hidden states and attentions unless required for debugging.
- Autoregressive generation, beam search, KV cache, speculative decoding: not applicable.
- RN50 checkpoint path: separate architecture audit.
- Position interpolation for non-checkpoint image sizes.
- Multi-GPU tensor parallelism and quantized weight formats.
- End-to-end tokenizer/image processor execution inside DinoML runtime; keep it CPU/data-pipeline first.

## 15. Final implementation checklist

- [ ] Parse `ChineseCLIPConfig`, `ChineseCLIPTextConfig`, and `ChineseCLIPVisionConfig`.
- [ ] Respect `text_config_dict` / `vision_config_dict` compatibility fields.
- [ ] Load text, vision, projection, logit_scale, embedding, and position weights.
- [ ] Implement text embedding sum + LayerNorm with default token type/position IDs.
- [ ] Implement dense noncausal text self-attention with bidirectional mask.
- [ ] Implement text FFN with GELU and post-residual LayerNorm.
- [ ] Implement contrastive text pooling from `last_hidden_state[:, 0, :]`.
- [ ] Implement NCHW ViT patch embedding and class/position embedding add.
- [ ] Implement dense noncausal vision attention and quick_gelu FFN.
- [ ] Implement vision CLS pooling with post LayerNorm.
- [ ] Implement bias-free text and visual projection heads.
- [ ] Implement L2 feature normalization and `exp(logit_scale)` similarity GEMM.
- [ ] Preserve `logits_per_text [B_text, B_image]` and `logits_per_image [B_image, B_text]`.
- [ ] Add fixed-size parity tests for base, large, large-336px, and huge configs.
- [ ] Add guarded Conv2d patch -> Linear rewrite.
- [ ] Add guarded packed QKV rewrite.
- [ ] Benchmark branch encoders and final similarity separately.
