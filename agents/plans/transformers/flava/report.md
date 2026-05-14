# FLAVA Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: facebook/flava-full, facebook/flava-image-codebook; gated/401 checked: facebook/flava-full-weights, facebook/flava-image, facebook/flava-text
Config source: HF config/preprocessor/tokenizer snapshots in agents/plans/transformers/flava/_sources/
Source files inspected: src/transformers/models/flava/configuration_flava.py, modeling_flava.py, image_processing_flava.py, image_processing_pil_flava.py, processing_flava.py
Any missing files or assumptions: official image/text/full-weights repos returned 401 in this environment. The report uses public facebook/flava-full and facebook/flava-image-codebook configs plus current source defaults for source-derived behavior.
```

Primary HF links: [facebook/flava-full](https://huggingface.co/facebook/flava-full), [facebook/flava-image-codebook](https://huggingface.co/facebook/flava-image-codebook), gated/401 links [facebook/flava-full-weights](https://huggingface.co/facebook/flava-full-weights), [facebook/flava-image](https://huggingface.co/facebook/flava-image), [facebook/flava-text](https://huggingface.co/facebook/flava-text).

Report target: inference-only `FlavaModel` parity for independent image/text encoders, optional multimodal encoder, and retrieval/contrastive embeddings. `FlavaForPreTraining` heads, masked modeling losses, and the image codebook are optional second-stage targets.

## 2. High-level architecture

FLAVA is a multimodal encoder family, not a decoder/generation model. The core model has a ViT-style image encoder, a BERT-style bidirectional text encoder, learned projections into a shared multimodal hidden width, and a third bidirectional multimodal encoder over concatenated image and text token sequences.

```text
CPU image/text preprocessing
  -> pixel_values NCHW + input_ids/attention_mask/token_type_ids
  -> image encoder and text encoder independently
  -> project image/text token states to multimodal width
  -> concat image tokens then text tokens
  -> optional multimodal encoder with extra CLS token
  -> optional heads: contrastive similarity, ITM, MLM, MIM, MMM
```

Stage decomposition:

- CPU/data pipeline: BERT tokenizer, image resize/center-crop/rescale/normalize, optional random MIM mask, optional codebook image preprocessing.
- Independently cacheable encoders: image and text encoders can run separately and their final hidden states can be cached before multimodal fusion.
- Multimodal fusion: projected image sequence and projected text sequence are concatenated; the multimodal model may prepend its own CLS token and uses dense bidirectional self-attention.
- Heads: `FlavaModel` exposes token-level image/text/multimodal embeddings. `FlavaForPreTraining` adds MLM/MIM/MMM/ITM/contrastive heads and loss-only training paths.

## 3. Important config dimensions

Source defaults from `configuration_flava.py`:

| Field | Image encoder | Text encoder | Multimodal encoder | Image codebook |
| --- | ---: | ---: | ---: | ---: |
| hidden size | 768 | 768 | 768 | 256 base channels |
| layers | 12 | 12 | 6 | 4 groups x 2 residual blocks |
| attention heads | 12 | 12 | 12 | none |
| head dim | 64 | 64 | 64 | n/a |
| intermediate size | 3072 | 3072 | 3072 | conv residual path hidden `out/4` |
| activation | GELU | GELU | GELU | ReLU |
| qkv bias | true | true | true | n/a |
| layer norm eps | 1e-12 | 1e-12 | 1e-12 | n/a |
| vocab size | 8192 image-code labels | 30522 text tokens | n/a | 8192 output channels |
| max positions | 197 image tokens at 224/16 | 512 | concatenated runtime length + optional CLS | output grid from conv/pool path |
| patch/image size | 16 / 224 | n/a | n/a | processor default 112 input |
| cache support | none | none for FLAVA use | none | none |

Top-level FLAVA defaults: `projection_dim=768`, `logit_scale_init_value=2.6592`, `init_codebook=True`, `return_loss=True`, `skip_unmasked_multimodal_encoder=True`, and all pretraining loss weights default to 1.0.

Representative checkpoint/config sweep:

| Model | Access | Architecture | Main dimensions | Processor/tokenizer facts |
| --- | --- | --- | --- | --- |
| `facebook/flava-full` | public | `FlavaForPreTraining` | source-default 12-layer image, 12-layer text, 6-layer multimodal, 768 width, image codebook 8192 vocab | CLIP mean/std image processor, 224 image path, 112 codebook path, BERT uncased tokenizer config |
| `facebook/flava-image-codebook` | public | `FlavaImageCodebook` | conv codebook only: hidden 256, 4 groups, 2 blocks/group, vocab 8192, float32 | no processor files in repo; use FLAVA processor codebook path defaults |
| `facebook/flava-full-weights` | 401 | expected FLAVA weights variant | inaccessible config | access would resolve whether dimensions or `init_codebook` differ |
| `facebook/flava-image` | 401 | expected image-only FLAVA | inaccessible config | access would resolve whether standalone image config differs |
| `facebook/flava-text` | 401 | expected text-only FLAVA | inaccessible config | access would resolve whether standalone text config differs |

## 3a. Family variation traps

- `FlavaModel.forward` requires `output_hidden_states=True`; it uses the final pre-layernorm hidden state from each unimodal encoder for the multimodal projections.
- `skip_unmasked_multimodal_encoder` defaults to true only in `FlavaForPreTraining`; plain `FlavaModel.forward` runs the multimodal encoder when both modalities are present unless the caller explicitly skips it.
- `get_text_features` and `get_image_features` assign `pooler_output = projection(last_hidden_state)`, producing projected per-token features, not a CLS-only vector. Contrastive pretraining uses `text_embeddings[:, 0, :]` and `image_embeddings[:, 0, :]` separately.
- Image and text hidden sizes can differ from multimodal hidden size in config, although public/default configs use 768 for all three. Keep `image_to_mm_projection` and `text_to_mm_projection` as real projections.
- The image encoder is NCHW with `Conv2d` patch embedding. Treat NHWC/channel-last as a guarded local optimization, not the semantic graph.
- Image position interpolation is optional via `interpolate_pos_encoding`; fixed 224 inputs can use learned constants directly.
- Text uses learned absolute position embeddings and BERT token type IDs; no RoPE, ALiBi, causal mask, KV cache, or generation path exists.
- The processor's random `FlavaMaskingGenerator` is training/pretraining input construction, not deterministic inference behavior.
- Codebook preprocessing has a separate 112x112 image path and `map_pixels(x) = 0.8*x + 0.1`; do not reuse the main 224 CLIP-normalized image tensor for codebook labels.
- Historical `*_config_dict` keys are merged over sub-configs for backward compatibility. Report source-derived defaults separately from checkpoint JSON values.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors `[B, 3, H, W]`, Conv2d patch embedding, `flatten(2)`, `transpose(1, 2)`, `view`, `reshape`, `permute`, `contiguous`, `cat`, `expand`.
- Bool mask flattening for image patch masks `[B, 14, 14] -> [B, 196]`.
- Additive attention mask expansion through `get_extended_attention_mask`.
- Indexed/boolean filtering only for optional pretraining losses, not first inference target.

Neural network primitives:

- Embedding lookups: text word `[30522, 768]`, text position `[512, 768]`, token type `[2, 768]`.
- Image patch Conv2d: `Conv2d(3 -> 768, kernel=16, stride=16, bias=True)` in current source.
- Learned image CLS, optional learned image mask token, learned image position table `[1, 197, 768]`.
- Linear with bias for separate Q/K/V/O projections, FFN `768 -> 3072 -> 768`, pooler `768 -> 768`, projection heads `768 -> 768`, and multimodal bridge projections.
- LayerNorm eps `1e-12`, GELU, Tanh, dropout as inference identity, L2 normalize, exp, dense matmul.

Attention primitives:

- Dense bidirectional MHA for image, text, and multimodal encoders.
- Separate Q/K/V Linear weights in standard PyTorch layout `[out_features, in_features]`.
- Head count 12, head dim 64 in public/default configs.

Preprocessing-coupled ops:

- Image resize, center crop, rescale by `1/255`, normalize with CLIP mean/std for main path.
- Codebook resize/crop to 112, rescale, normalize with mean `[0,0,0]`, std `[1,1,1]`, then pixel map `0.8*x + 0.1`.
- BERT uncased tokenization with `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, attention mask, token type IDs.

Discrete codebook/tokenizer ops:

- Optional `FlavaImageCodebook`: conv/ReLU/residual/max-pool network producing logits `[B, 8192, H', W']`; `get_codebook_indices` returns `argmax(dim=1)`.
- MIM/MMM heads predict 8192 image-code labels over selected image tokens.

Optional heads:

- MLM and MMM text: transform `Linear(768 -> 768) + GELU + LayerNorm + Linear(768 -> 30522)`.
- MIM and MMM image: same transform ending in `Linear(768 -> 8192)`.
- ITM: multimodal CLS pooler then `Linear(768 -> 2)`.
- Contrastive: CLS image/text projection, L2 normalize, `exp(logit_scale)`, image-text and text-image similarity matrices.

## 5. Layer/block breakdown

Image embedding:

```text
pixel_values [B, 3, 224, 224]
patch = Conv2d(3 -> 768, kernel=16, stride=16)(pixel_values)
patch = flatten(2).transpose(1, 2)                 # [B, 196, 768]
if bool_masked_pos: patch = where(mask, mask_token, patch)
x = cat(image_cls, patch, dim=1)                    # [B, 197, 768]
x = x + learned_or_interpolated_position_embedding
```

Text embedding:

```text
x = word_embedding(input_ids)
x = x + token_type_embedding(token_type_ids or zeros)
x = x + position_embedding(position_ids or arange)
x = LayerNorm(x)
```

FLAVA encoder block, repeated 12 times for image/text and 6 times for multimodal:

```text
y = LayerNorm(x)
q,k,v = Linear(y), Linear(y), Linear(y)
attn = softmax((q @ k.T) / sqrt(64) + optional_mask) @ v
x = x + Linear(attn)
y = LayerNorm(x)
y = GELU(Linear(768 -> 3072)(y))
x = x + Linear(3072 -> 768)(y)
```

Top-level multimodal path:

```text
image_states = image_model(..., output_hidden_states=True).hidden_states[-1]
text_states = text_model(..., output_hidden_states=True).hidden_states[-1]
image_mm = Linear(image_hidden -> mm_hidden)(image_states)
text_mm = Linear(text_hidden -> mm_hidden)(text_states)
mm_input = cat([image_mm, text_mm], dim=1)
if use_cls_token: mm_input = cat([mm_cls, mm_input], dim=1)
mm_output = multimodal_encoder(mm_input, dense_bidirectional_mask)
```

## 6. Attention requirements

All required attention is noncausal dense self-attention. There is no autoregressive prefill/decode, KV cache, sliding window, block sparsity, GQA/MQA, RoPE, or ALiBi.

| Site | Causal? | Q/K/V source | Heads | Head dim | Mask |
| --- | --- | --- | ---: | ---: | --- |
| image encoder | no | image token sequence | 12 | 64 | optional caller `attention_mask`; usually none |
| text encoder | no | text token sequence | 12 | 64 | additive extended mask from `[B, T]` attention mask |
| multimodal encoder | no | `[image tokens, text tokens]` plus optional CLS | 12 | 64 | all-ones image side concatenated with text attention mask when provided |

Math order: project Q/K/V, reshape to `[B, heads, seq, head_dim]`, scale QK scores by `1/sqrt(head_dim)`, add mask, softmax over key length, dropout if training, multiply by V, transpose/reshape back to `[B, seq, hidden]`, output Linear, then residual outside the attention module.

Dense SDPA/FlashAttention-style kernels are viable for inference if they preserve bidirectional additive-mask semantics and return token-major output parity. Output attention tensors can be deferred.

## 7. Position encoding and custom math

Image position interpolation is copied from ViT/DINO-style bicubic interpolation:

```python
def flava_interpolate_pos(position_embeddings, height, width, patch_size):
    cls = position_embeddings[:, :1]
    patch = position_embeddings[:, 1:]
    dim = patch.shape[-1]
    old = int((patch.shape[1]) ** 0.5)
    new_h, new_w = height // patch_size, width // patch_size
    patch = patch.reshape(1, old, old, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return cat([cls, patch], dim=1)
```

Text positions are simple learned absolute positions:

```python
position_ids = arange(max_position_embeddings)[None, :seq_len]
x = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
```

Codebook pixel mapping:

```python
def flava_codebook_map_pixels(x):
    return 0.8 * x + 0.1
```

Precompute fixed image/text position embeddings for standard 224 and sequence length <= 512. Bicubic interpolation depends on runtime image height/width and should be guarded behind explicit dynamic-image admission.

## 8. Preprocessing and input packing

`FlavaProcessor` is a thin `ProcessorMixin` wrapper around an image processor and tokenizer. Public `facebook/flava-full` processor snapshots use BERT uncased tokenizer settings and FLAVA image preprocessing.

Main image path:

- Input images are resized and center-cropped to 224 by default.
- Output `pixel_values` are channel-first `[B, 3, 224, 224]`.
- Rescale by `1/255`, normalize with OpenAI CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.

Optional codebook path:

- Enabled by `return_codebook_pixels=True`.
- Resize/crop to 112, rescale by `1/255`, normalize with mean `[0,0,0]` and std `[1,1,1]`, then apply `0.8*x + 0.1`.
- Output key is `codebook_pixel_values`; it feeds `FlavaImageCodebook.get_codebook_indices` for MIM labels.

Optional mask path:

- Enabled by `return_image_mask=True`.
- The processor emits `bool_masked_pos` from a random block masking generator over a 14x14 patch grid, default total 75 masked patches.
- This is pretraining data construction. First inference parity can require callers not to pass `bool_masked_pos`.

Text path:

- `input_ids [B, T]`, `attention_mask [B, T]`, optional `token_type_ids [B, T]`, optional `position_ids [B, T]`.
- Special-token behavior comes from BERT uncased tokenizer config. The model graph consumes token IDs, segment IDs, position IDs, and attention masks; tokenization itself can remain CPU/data-pipeline work.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d to Linear

Source pattern:

```text
Conv2d(C -> H, kernel=patch, stride=patch, padding=0, bias=True)
  -> flatten(2)
  -> transpose(1, 2)
```

Replacement:

```text
WindowFlatten(NCHW, patch_h x patch_w) -> GEMM(weight_flat.T) -> BiasAdd -> token sequence
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input height/width divisible by patch size, or source-compatible truncation/reject behavior is implemented.
- Preserve NCHW flatten order. If an NHWC layout pass is used, it must rewrite window extraction and weight flattening together.

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias
```

Failure cases: enabled interpolation with unsupported dynamic grids, non-224 images without explicit admission, grouped/dilated conv variants, or layout translation crossing out of the patch-embedding region.

### Rewrite: packed QKV projection

Source pattern: separate `query`, `key`, and `value` Linear modules, all bias-enabled in default/public configs.

Replacement: one packed GEMM with output split order `[query, key, value]`.

Preconditions:

- Same input tensor and same hidden width.
- `qkv_bias=True`, or the rewrite handles missing biases explicitly.
- Preserve PyTorch weight layout `[out_features, in_features]`.

Failure cases: future config with asymmetric Q/K/V widths, missing bias not covered by packed kernel, or cross-attention variants. Current FLAVA source has self-attention only, including the multimodal encoder.

### Rewrite: guarded NCHW to NHWC patch/attention region

NHWC may help Conv2d/window extraction and fused image preprocessing, but only inside a fully controlled local region. Axis-sensitive source ops include `flatten(2)`, `transpose(1, 2)`, image mask flattening, and `Conv2d` channel axis. A layout pass must either rewrite those axes or place a no-layout-translation guard around patch embedding and externally visible `pixel_values`.

### Rewrite: contrastive similarity

Source pattern:

```text
image_cls -> Linear -> L2Normalize
text_cls -> Linear -> L2Normalize
temperature = exp(logit_scale)
logits_i = image @ text_all.T * temperature
logits_t = text @ image_all.T * temperature
```

Replacement: fused normalize plus scaled GEMM for retrieval batches.

Preconditions: inference mode, single-process or explicit distributed gather policy. Failure cases: training distributed all-gather with gradient semantics.

## 10. Kernel fusion candidates

Highest priority:

- Dense bidirectional MHA for `[B, S, 768]` with head dim 64. It appears in all three encoders.
- Residual add + LayerNorm around attention and FFN blocks. LayerNorm eps is `1e-12`; preserve this for parity.
- Patch Conv2d to GEMM rewrite for fixed 224 image inputs.

Medium priority:

- Packed QKV projection and output projection scheduling.
- FFN `Linear + GELU + Linear` for 12+12+6 encoder blocks.
- Contrastive projection + L2 normalize + scaled similarity matrix.
- MLM/MIM head transform `Linear + GELU + LayerNorm` before large vocab GEMM.

Lower priority:

- Codebook conv residual network. It is only needed for automatic MIM labels or standalone codebook use.
- Bicubic position interpolation for non-224 images.
- Training loss filtering, boolean masked token compaction, and distributed contrastive gather.

## 11. Runtime staging plan

Stage 1: parse `FlavaConfig` and sub-configs; load public `facebook/flava-full` weights; reject gated/unknown variants until configs are available.

Stage 2: run image encoder and text encoder independently from already-tokenized/normalized tensors. Validate fixed 224 image path and text sequence <= 512.

Stage 3: add `FlavaModel` multimodal fusion: projection layers, concat order `[image, text]`, optional multimodal CLS, dense mask construction, and 6 multimodal blocks.

Stage 4: add contrastive inference outputs: CLS projection, L2 normalization, `exp(logit_scale)`, image/text similarity matrices. Treat distributed all-gather as deferred.

Stage 5: add optional ITM head for paired image/text scoring.

Stage 6: add optional MLM/MIM/MMM heads and masked-position filtering if pretraining parity is requested.

Stage 7: add `FlavaImageCodebook` only if automatic MIM label generation or standalone codebook inference matters.

Stage 8: optimize with patch-to-GEMM, packed QKV, fused residual LayerNorm, and optimized dense attention.

## 12. Parity and validation plan

- Config tests: instantiate source-default configs and public `facebook/flava-full`/`facebook/flava-image-codebook` snapshots; assert image tokens 197, text max positions 512, multimodal layers 6, codebook vocab 8192.
- Processor-boundary tests: compare HF processor output shapes/values for one image and one text prompt; keep CPU preprocessing separate from DinoML runtime initially.
- Patch embedding parity: compare Conv2d -> flatten -> transpose tokens before CLS/position add.
- Single-block parity: image/text/multimodal `FlavaLayer` with random masks; fp32 tolerance around `1e-5` where accumulation order matches.
- Encoder parity: compare image-only and text-only last hidden states and pooled outputs for fixed inputs.
- Multimodal parity: compare `FlavaModel` outputs for one paired image/text input with and without `skip_multimodal_encoder`.
- Head parity: contrastive logits `[B_image, B_text]` and `[B_text, B_image]`, ITM logits `[B,2]`, optional MLM/MIM logits.
- Codebook parity: compare codebook logits `[B,8192,14,14]` and `argmax(dim=1)` labels from 112x112 `codebook_pixel_values`.
- Reduced precision: start with fp32 parity; for fp16/bf16 full-encoder parity use relaxed tolerances around `1e-2` after layerwise checks pass.

No DinoML tests or imports were run for this docs-only audit.

## 13. Performance probes

- CPU processor throughput: tokenizer, main image preprocessing, codebook preprocessing, random mask generation.
- Image encoder throughput for batch and resolution sweeps; patch Conv2d path versus GEMM rewrite.
- Text encoder throughput over sequence length 16/77/128/512.
- Multimodal encoder throughput over text length and image token count.
- Attention backend comparison: eager dense attention versus fused SDPA-style kernel for encoder masks.
- Contrastive retrieval matrix throughput for batch-size sweeps and cached branch embeddings.
- Optional codebook conv throughput and memory footprint.
- Hidden-state memory probe when `FlavaModel` requires `output_hidden_states=True`.

## 14. Skip/defer list

- Training losses, dropout, gradient checkpointing, and distributed contrastive all-gather.
- Automatic MIM label generation and `FlavaImageCodebook` unless the target is pretraining parity.
- Random mask generation inside DinoML runtime.
- Dynamic image sizes and bicubic position interpolation.
- Output attention tensors and hidden-state tuple materialization beyond what `FlavaModel` needs internally.
- Gated checkpoint variants until access resolves their actual configs.
- Quantized or packed weights; no source-coupled quantized storage format was found.

## 15. Final implementation checklist

- [ ] Parse `FlavaConfig`, `FlavaImageConfig`, `FlavaTextConfig`, `FlavaMultimodalConfig`, and optional `FlavaImageCodebookConfig`.
- [ ] Load public FLAVA weights and preserve submodule names for image/text/multimodal/codebook separation.
- [ ] Implement text embeddings with token type and learned absolute position embeddings.
- [ ] Implement image patch embedding, CLS/mask token handling, and learned image positions for fixed 224 inputs.
- [ ] Implement dense bidirectional MHA encoder blocks with pre-LayerNorm ordering.
- [ ] Implement image/text encoders and final LayerNorm/pooler.
- [ ] Implement image/text-to-multimodal projections, concat order, multimodal CLS, and multimodal encoder.
- [ ] Implement contrastive CLS projection, L2 normalize, `exp(logit_scale)`, and similarity matrices.
- [ ] Add optional ITM head.
- [ ] Add optional MLM/MIM/MMM heads and masked-token filtering.
- [ ] Add optional image codebook conv network and `argmax(dim=1)` labels.
- [ ] Add patch Conv2d to GEMM rewrite with NCHW/layout guards.
- [ ] Add packed QKV rewrite with split-order tests.
- [ ] Add fused residual LayerNorm and dense attention kernels.
- [ ] Validate source-default and `facebook/flava-full` parity against Transformers.
- [ ] Benchmark processor, branch encoders, multimodal fusion, contrastive similarity, and optional codebook separately.
