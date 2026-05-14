# BridgeTower Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: BridgeTower/bridgetower-base, BridgeTower/bridgetower-base-itm-mlm, BridgeTower/bridgetower-large-itm-mlm, BridgeTower/bridgetower-large-itm-mlm-itc, BridgeTower/bridgetower-large-itm-mlm-gaudi
Config source: HF config.json and preprocessor_config.json snapshots in agents/plans/transformers/bridgetower/_sources/
Source files inspected: src/transformers/models/bridgetower/configuration_bridgetower.py, modeling_bridgetower.py, image_processing_bridgetower.py, image_processing_pil_bridgetower.py, processing_bridgetower.py, tests/models/bridgetower/test_modeling_bridgetower.py, docs/source/en/model_doc/bridgetower.md
Any missing files or assumptions: no gated or 401/403 checkpoint configs were encountered. processor_config.json/tokenizer_config.json returned 404 for inspected official repos; the processor is composed from the image processor plus a RoBERTa tokenizer named in preprocessor_config.json.
```

Primary HF links: [bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base), [base-itm-mlm](https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm), [large-itm-mlm](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm), [large-itm-mlm-itc](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc), [large-itm-mlm-gaudi](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-gaudi).

Report target: inference-only multimodal encoder/fusion parity for `BridgeTowerModel`, then optional ITM, MLM, and contrastive heads. This is not a text generation family.

## 2. High-level architecture

BridgeTower is a dual-stream vision/text encoder with cross-modal fusion. It uses a CLIP/ViT-like vision transformer, a RoBERTa-like bidirectional text transformer, bridge/link layers from late unimodal layers, and paired cross-modal layers for text-to-image and image-to-text fusion.

```text
CPU image/text preprocessing
  -> pixel_values NCHW + input_ids/attention_mask
  -> early text encoder layers and early vision encoder layers
  -> project each modality into shared hidden size and add modality type embeddings
  -> 6 paired cross-modal layers with link-tower residual bridges
  -> pooled text CLS + image CLS concatenation
  -> optional ITM / MLM / contrastive heads
```

Stage decomposition:

- CPU/data pipeline: image resize, center crop, rescale, CLIP mean/std normalize, optional padding mask, RoBERTa tokenization.
- Independently stageable encoders: early text and vision layers before the first cross-modal layer. For base this is 7 layers; for large this is 19 layers because `split_index = text_layers - cross_layers + 1`.
- Fusion stage: 6 paired cross-modal layers. Each layer has self-attention, cross-attention, FFN, and bridge/link LayerNorm input from the corresponding late unimodal layer.
- Heads: `BridgeTowerForImageAndTextRetrieval` returns 2-way ITM logits, `BridgeTowerForMaskedLM` returns vocab logits for every text token, and `BridgeTowerForContrastiveLearning` returns normalized text/image/cross embeddings plus pairwise similarity matrices for loss.

## 3. Important config dimensions

Effective source defaults from `configuration_bridgetower.py`:

| Field | Default |
| --- | ---: |
| fusion hidden size | 768 |
| fusion layers | 6 |
| fusion attention heads | 12 |
| text hidden size | 768 |
| text layers | 12 |
| text heads | 12 |
| text head dim | 64 |
| text intermediate size | 3072 |
| vocab size | 50265 |
| max position embeddings | 514 |
| token type vocab size | 1 in text embeddings, plus separate 2-entry multimodal type embedding |
| vision hidden size | 768 |
| vision layers | 12 |
| vision patch/image size | 16 / 288 |
| activation | GELU in text/fusion FFN, QuickGELU in vision FFN |
| cache support | not required for BridgeTowerModel; copied text model can use decoder cache only if configured as decoder |

Representative checkpoint sweep from HF `config.json` snapshots:

| Model | Hidden | Text layers | Vision layers | Cross layers | Heads | Patch/image | Image tokens | Head-specific fields |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `BridgeTower/bridgetower-base` | 768 | 12 | 12 | 6 | 12 | 16 / 288 | 325 | base model |
| `BridgeTower/bridgetower-base-itm-mlm` | 768 | 12 | 12 | 6 | 12 | 16 / 288 | 325 | ITM and MLM weights |
| `BridgeTower/bridgetower-large-itm-mlm` | 1024 | 24 | 24 | 6 | 16 | 14 / 294 | 442 | ITM and MLM weights |
| `BridgeTower/bridgetower-large-itm-mlm-itc` | 1024 | 24 | 24 | 6 | 16 | 14 / 294 | 442 | `contrastive_hidden_size=512`, `logit_scale_init_value=2.6592` |
| `BridgeTower/bridgetower-large-itm-mlm-gaudi` | 1024 | 24 | 24 | 6 | 16 | 14 / 294 | 442 | same model dimensions as large ITM/MLM |

Preprocessor snapshot variation: base uses `tokenizer="roberta-base"` and size 288 where present; large uses `tokenizer="roberta-large"` and size 294. Text config still allows 514 positions, while preprocessor configs advertise `max_text_len=50`; treat that as a data-pipeline truncation policy, not a model graph limit.

## 3a. Family variation traps

- Base and large differ in hidden width, head count, FFN width, number of unimodal text/vision layers, patch size, and image token count. Do not hardcode 325 image tokens.
- `split_index = text_layers - cross_layers + 1` means the number of early unimodal layers changes with the text/vision depth. Cross-modal layers remain 6 in inspected official configs.
- `share_cross_modal_transformer_layers=True` makes one text projection and one image projection module reused at every cross layer. If false, source constructs per-layer `ModuleList`s. The contrastive head source assumes the shared case because it calls `cross_modal_image_transform(...)` directly.
- `share_link_tower_layers=False` in inspected configs creates one link tower per cross layer after the first. If true, the source assigns a single `BridgeTowerLinkTower`, but `BridgeTowerModel.forward` indexes it as a list; DinoML should reject or separately verify this config combination before claiming support.
- `link_tower_type` can be `add`, `scaled_add`, or `interpolate`; only `add` appears in inspected official configs. The other two add learned scalar multiply/blend parameters.
- `pixel_mask` is accepted but the main `BridgeTowerModel.forward` overwrites the cross-modal image mask with all ones after image embeddings are produced. Do not rely on processor `pixel_mask` for the fusion stage unless source behavior changes.
- Historical config/preprocessor keys such as `drop_rate`, `head_hidden_scale`, `input_image_embed_size`, `input_text_embed_size`, `max_text_len`, `mlp_ratio`, `vit_remove_last`, and `init_layernorm_from_vit` are present in some snapshots but not read by the inspected in-library modeling source.
- Base `config.json` uses historical `vit_remove_last`; current `BridgeTowerVisionConfig` reads `remove_last_layer`. Effective source default is `remove_last_layer=False` if the current config loader does not map the old key.
- `position_embedding_type="absolute"` appears in text configs and RoPE/ALiBi are absent.
- `use_cache=True` in text config is inherited from RoBERTa/BERT-style code, but BridgeTower's multimodal forward path is encoder/fusion only and does not expose generation.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input, Conv2d patch embedding, flatten spatial axes, transpose to token-major and back.
- `permute`, `transpose`, `reshape/view`, `contiguous`, `stack`, `cat`, `expand`, `gather`, `cumsum`, boolean/int mask construction, dtype casts.
- Token/index ops for word, position, text token type, and modality type embeddings.

Neural network primitives:

- Embedding lookup: vocab 50265 x hidden, learned absolute text positions 514 x hidden, learned vision positions `(grid^2 + 1) x hidden`, modality embeddings 2 x hidden.
- Linear with bias for text/fusion Q/K/V/O, FFN, poolers, link projections, ITM, contrastive heads.
- Conv2d patch embedding: base `Conv2d(3 -> 768, kernel=16, stride=16, bias=False)`; large `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- LayerNorm epsilon `1e-5`, dropout in source but disabled at inference, GELU, QuickGELU, Tanh, L2 normalize, exponent, matmul.
- MLM head: `Linear(H -> H) + GELU + LayerNorm + Linear(H -> 50265, bias=False) + separate bias`. Decoder weight is tied to text token embedding by `_tied_weights_keys` for the MLM class.
- ITM head: `Linear(2H -> 2)`.
- Contrastive head: `Linear(H -> 512)` for text and image, `Linear(2H -> 512)` for cross embeddings, L2 normalization, three batch similarity matrices scaled by `exp(logit_scale)`.

Attention primitives:

- Vision self-attention via `nn.MultiheadAttention` in token-major `[L, B, H]`, noncausal, MHA, head dim 64.
- Text self-attention via separate Q/K/V Linear, bidirectional mask, MHA, head dim 64.
- Cross-modal paired layers use self-attention plus cross-attention in each stream: text queries attend to image keys/values, and image queries attend to text keys/values.
- The source routes text/fusion attention through `ALL_ATTENTION_FUNCTIONS` with eager fallback. Eager math is QK^T scale, additive mask, softmax, dropout, AV.

Position/custom math:

- Text learned absolute positions generated from non-pad token cumulative counts.
- Vision learned class token plus learned patch positions, with optional bicubic interpolation for non-config image sizes.

Generation/cache ops:

- No generation, sampling, or decoder KV cache is required for BridgeTowerModel or the implemented heads.
- The standalone copied `BridgeTowerTextModel` can instantiate `EncoderDecoderCache` only when `is_decoder=True`, but inspected BridgeTower configs are encoder-only and the multimodal model does not use this path.

Preprocessing-coupled ops:

- Resize shortest edge to configured size with max side `int(1333 / 800 * shortest_edge)`, round, then floor height/width to a multiple of `size_divisor=32`.
- Center crop to square `crop_size.shortest_edge`.
- Rescale and normalize with OpenAI CLIP mean/std.
- Pad batch images and produce `pixel_mask`, though the main fusion model ignores the input mask later.
- RoBERTa tokenization with special tokens, attention mask, and default no padding unless requested.

## 5. Layer/block breakdown

Vision embedding:

```text
pixel_values [B, 3, H, W]
patch = Conv2d(3 -> Hv, kernel=patch, stride=patch, bias=False)
patch = flatten(2).transpose(1, 2)          # [B, G, Hv]
tokens = cat(class_embedding, patch)         # [B, G+1, Hv]
tokens = tokens + learned/interpolated positions
tokens = LayerNorm(tokens)
tokens = permute to [L, B, Hv]
```

Vision residual attention block, repeated `vision_config.num_hidden_layers`:

```text
y = x + MultiheadAttention(LayerNorm(x), key_padding_mask=None)
x = y + Linear(4Hv -> Hv)(QuickGELU(Linear(Hv -> 4Hv)(LayerNorm(y))))
```

Text embedding and encoder block:

```text
position_ids = cumsum(input_ids != pad_id) + pad_id
x = word_embedding + token_type_embedding + position_embedding
x = LayerNorm(x)
repeat text_config.num_hidden_layers:
  q,k,v = Linear(H -> H) separately, with bias
  attn = bidirectional MHA(q,k,v, attention_mask)
  x = LayerNorm(Linear(attn) + residual)
  mlp = Linear(H -> I) -> GELU -> Linear(I -> H)
  x = LayerNorm(mlp + residual)
```

Cross-modal fusion, repeated 6 times after early unimodal layers:

```text
first fusion layer:
  text = LayerNorm(Linear(text_hidden -> H)(text) + text_modality_type)
  image = LayerNorm(Linear(vision_hidden -> H)(image) + image_modality_type)
  text = CrossLayer(text, encoder_hidden_states=image)
  image = CrossLayer(image, encoder_hidden_states=text_initial)

later fusion layers:
  text_unimodal = next text encoder layer(text_unimodal)
  image_unimodal = next vision block(image_unimodal)
  text_bridge = LinkTower(Linear(text_unimodal -> H) + text_type, previous_cross_text)
  image_bridge = LinkTower(Linear(image_unimodal -> H) + image_type, previous_cross_image)
  text = CrossLayer(text_bridge, encoder_hidden_states=image_bridge)
  image = CrossLayer(image_bridge, encoder_hidden_states=text_bridge)
```

`CrossLayer` itself is:

```text
self_attn = MHA(hidden, hidden, hidden)
x = LayerNorm(Linear(self_attn) + hidden)
cross_attn = MHA(query=x, key/value=encoder_hidden_states)
x = LayerNorm(Linear(cross_attn) + x)
ffn = Linear(H -> I) -> GELU -> Linear(I -> H)
x = LayerNorm(ffn + x)
```

Pool/head path:

```text
text_cls = tanh(Linear(text_features[:, 0]))
image_cls = tanh(Linear(image_features[:, 0]))
pooler_output = cat(text_cls, image_cls)      # [B, 2H]
```

## 6. Attention requirements

BridgeTower requires dense, bidirectional encoder attention only for the primary runtime target.

| Attention site | Causal? | Type | Heads | Head dim | Masking | Cache |
| --- | --- | --- | ---: | ---: | --- | --- |
| vision residual blocks | no | self MHA | `Hv / 64` | 64 | optional key padding in standalone path, unused in main model | none |
| text encoder | no | self MHA | 12 base, 16 large | 64 | additive bidirectional attention mask from text attention mask | none |
| cross-modal text layers | effectively noncausal in eager path | self MHA then cross MHA to image | 12/16 | 64 | text self mask and image all-ones cross mask | none |
| cross-modal image layers | effectively noncausal in eager path | self MHA then cross MHA to text | 12/16 | 64 | image all-ones self mask and text cross mask | none |

Source-specific math order for text/fusion attention: project Q/K/V, reshape to `[B, heads, S, head_dim]`, compute QK^T scaled by `head_dim**-0.5`, add mask, softmax over key length, dropout if training, multiply by V, transpose back, output Linear, dropout, residual add, LayerNorm.

No sliding-window, block-sparse, local, ALiBi, RoPE, GQA, MQA, packed varlen, or FlashAttention-only requirement exists. SDPA-style backends are possible only if they preserve the same dense mask semantics and output orientation.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings. Source creates pad-aware positions:

```python
def bridgetower_text_position_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = (input_ids != padding_idx).int()
    incremental = (cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental.long() + padding_idx
```

Vision positions are learned absolute patch embeddings with optional bicubic interpolation:

```python
def interpolate_vision_pos(pos_weight, cls_count, old_grid, new_h, new_w):
    cls = pos_weight[:cls_count]
    patch = pos_weight[cls_count:].reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
    return cat([cls[None, :, :], patch], dim=1)
```

For fixed official image sizes, position embeddings can be loaded as constants. Interpolation depends on dynamic input height/width and should be a guarded optional path behind `interpolate_pos_encoding=True`.

## 8. Preprocessing and input packing

Processor outputs:

- `input_ids [B, T]`, `attention_mask [B, T]`, optional tokenizer `token_type_ids` from RoBERTa-style tokenizer. The model's top-level `BridgeTowerModel.forward` currently ignores its `token_type_ids` argument when creating text embeddings and calls `self.text_model.embeddings(input_ids=input_ids)`.
- `pixel_values [B, 3, S, S]` after resize, center crop, rescale, and CLIP normalization. Official S is 288 for base and 294 for large.
- `pixel_mask [B, S, S]` may be produced by the image processor when padding, but the main model builds an all-ones token mask of shape `[B, image_tokens]` after patch embedding.

Image pipeline details are CPU/data-pipeline work for first integration. The GPU graph should start from normalized NCHW `pixel_values` and token tensors. Candidate runtime-side preprocessing can be deferred unless DinoML wants end-to-end processor parity.

There is no multimodal placeholder-token scatter into a text sequence. Text and image remain separate streams until cross-attention. Image embeddings can be supplied directly through `image_embeds`, but source expects `[B, image_tokens, H]` and permutes to token-major internally.

For contrastive/retrieval usage, text, image, and cross-modal pooled embeddings can be cached independently only after their relevant stages:

- Early unimodal features can be cached before cross-modal fusion for repeated pair scoring.
- Final `image_embeds` used by the contrastive head are derived from hidden image states plus modality embedding/projection and do not equal raw vision-only pooled features.
- ITM logits are pair-specific because they depend on cross-modal fusion, not only independent branch embeddings.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d to Linear

Source pattern:

```text
Conv2d(C -> H, kernel=patch, stride=patch, padding=0, bias=False) -> flatten(2) -> transpose(1, 2)
```

Replacement:

```text
WindowFlatten(NCHW, patch x patch, row-major spatial order) -> GEMM(weight_flat.T) -> token sequence
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- Input height and width divisible by patch size.
- Preserve NCHW semantic axis order unless a local layout region rewrites the Conv/window extraction and all consumers.

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases: dynamic image sizes not divisible by patch, enabled position interpolation with unexpected grid, grouped/bias conv variants, or channel-last layout pass without matching weight/window permutation. Parity test: compare patch tokens before class-token concat for base and large configs.

### Rewrite: QKV projection packing

Source pattern: separate `query`, `key`, and `value` Linear modules with standard PyTorch weights `[out_features, in_features]`.

Replacement: one packed GEMM producing `[Q, K, V]` in all-Q/all-K/all-V row-block order, then split into three tensors.

Preconditions:

- Same input tensor, same dtype/device, same `hidden_size`, all projections have bias.
- Preserve split order `query`, `key`, `value`.
- Apply separately for text self-attention, cross-modal self-attention, and cross-attention K/V from encoder hidden states.

Failure cases: cross-attention Q input differs from K/V input, so only K/V can be packed together there unless the compiler emits two GEMMs.

### Rewrite: token-major vision attention to batch-major dense attention

Source pattern: vision blocks operate `[L, B, H]` because `nn.MultiheadAttention` default is not batch-first.

Replacement: keep sequence tensors `[B, L, H]` across the vision tower and rewrite attention axes internally.

Preconditions:

- Every consumer in the controlled region is vision block attention/MLP/LayerNorm.
- Rewrite `permute(1,0,2)` and `permute(1,0,2)` boundaries consistently.
- Key padding mask semantics are preserved for standalone `BridgeTowerVisionModel`.

Failure cases: output hidden-state stacking expects layer list tensors in source orientation; use a no-layout-translation guard around externally visible hidden states.

### Rewrite: link tower add/scaled/interpolate as fused residual LayerNorm

Source pattern:

```text
LayerNorm(a + b)
LayerNorm(a * scalar + b)
LayerNorm(a * (1 - beta) + b * beta)
```

Replacement: fused elementwise blend plus LayerNorm.

Preconditions: same shape `[B, S, H]`, scalar parameter broadcast, inference mode. Failure cases: unsupported `link_tower_type` or scalar dtype mismatch.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm plus residual add for text, vision, link, and fusion blocks. It appears at every block boundary and dominates graph count.
- Dense MHA for bidirectional encoder attention. Required for text self-attention and both cross-modal streams; head dim is consistently 64.
- Patch Conv2d to GEMM rewrite for base and large. It converts the only convolution into a standard projection.

Medium priority:

- QKV packed projection for self-attention and K/V packing for cross-attention.
- FFN fused Linear + GELU/QuickGELU + Linear scheduling, especially large `1024 -> 4096 -> 1024`.
- MLM head fusion: transform Linear + GELU + LayerNorm before vocab GEMM.
- Contrastive head: projection + L2 normalize + scaled similarity matrix for batch retrieval workloads.

Lower priority:

- Vision position interpolation. Official inference can use fixed precomputed positions unless dynamic image admission is needed.
- Training losses and dropout paths.
- Standalone decoder/cache behavior inherited by `BridgeTowerTextModel`, because it is not used by official BridgeTower configs.

## 11. Runtime staging plan

Stage 1: parse configs and load base/large weights, rejecting unsupported stale or inconsistent config combinations. Stub heads initially.

Stage 2: implement processor boundary tensors and run patch embedding plus one vision/text block parity from normalized tensors.

Stage 3: implement full unimodal text and vision encoders with fixed official image sizes and bidirectional dense masks.

Stage 4: implement six paired cross-modal layers and bridge/link towers. Validate `BridgeTowerModel` `text_features`, `image_features`, and `pooler_output`.

Stage 5: add ITM head and MLM head. Preserve MLM decoder weight alias with text embeddings.

Stage 6: add contrastive head for `large-itm-mlm-itc`, including L2 normalization, stacked embeddings, `exp(logit_scale)`, and three similarity matrices.

Stage 7: add optimizations: patch-conv rewrite, packed QKV, fused residual LayerNorm, optimized dense attention.

Stage 8: optional dynamic image path with position interpolation and processor parity.

## 12. Parity and validation plan

- Config/load tests: load all five inspected config snapshots; assert base image token count 325, large count 442, split indices 7 and 19.
- Custom op tests: text position ID creation with pad tokens, link tower variants, vision position interpolation against PyTorch for one non-square or larger grid.
- Single-block parity: vision residual block, text encoder block, cross-modal layer with random masks and fp32 tolerances around `1e-5` absolute/relative.
- Encoder parity: compare early text/vision features at split boundary and final `BridgeTowerModel` outputs against Transformers for base and large.
- Head parity: ITM logits `[B,2]`, MLM logits `[B,T,50265]`, contrastive embeddings `[B,512]` and similarity matrices `[B,B]`.
- End-to-end smoke: use HF processor snapshots for one image/text pair and compare top ITM class or masked token id.
- Recommended reduced precision tolerances: start with fp32 graph parity; for fp16/bf16 use layerwise tolerances around `1e-2` for full model outputs, with stricter checks on embeddings/projections where accumulation order is controlled.

No DinoML runtime tests were run for this docs-only audit.

## 13. Performance probes

- Processor throughput split by image resize/crop/normalize and tokenizer.
- Patch embedding throughput for Conv2d path versus GEMM rewrite.
- Text-only early encoder latency for base T=50 and larger T up to 514.
- Vision-only early encoder latency for base 325 and large 442 tokens.
- Cross-modal fusion latency sweep over text length and image token count.
- ITM pair scoring throughput: one image against many texts, with and without cached early unimodal features.
- Contrastive batch similarity throughput for batch-size sweeps.
- Attention backend comparison: eager dense attention versus SDPA/custom kernel for text self-attention and cross-attention.
- Memory probe for hidden-state capture because contrastive output forces `output_hidden_states=True`.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout behavior beyond inference-disabled identity.
- Standalone `BridgeTowerTextModel` decoder/cache mode.
- Dynamic image sizes and bicubic position interpolation, unless a deployment requires it.
- Processor execution inside DinoML runtime.
- Unsupported or source-inconsistent config combinations: `share_link_tower_layers=True`, contrastive head with `share_cross_modal_transformer_layers=False`, non-`add` link tower until explicitly tested.
- Historical remote/training preprocessor fields that the current source does not read.
- Quantized or packed weights; no source-coupled quantized format was found.

## 15. Final implementation checklist

- [ ] Parse `BridgeTowerConfig`, `BridgeTowerTextConfig`, and `BridgeTowerVisionConfig`.
- [ ] Reject or route unsupported config traps: shared link tower, non-shared cross-modal transforms in contrastive head, stale `vit_remove_last` without explicit mapping.
- [ ] Load text, vision, cross-modal, pooler, MLM, ITM, and contrastive weights.
- [ ] Preserve MLM decoder weight alias to text token embeddings.
- [ ] Implement text embedding position ID generation.
- [ ] Implement fixed-size vision patch embedding and learned class/position embeddings.
- [ ] Implement vision QuickGELU MHA blocks.
- [ ] Implement RoBERTa-style bidirectional text blocks.
- [ ] Implement paired cross-modal layers with self-attention, cross-attention, FFN, and link towers.
- [ ] Implement pooler concat output.
- [ ] Add ITM, MLM, and contrastive heads.
- [ ] Add patch Conv2d to GEMM rewrite with strict guards.
- [ ] Add packed QKV/KV projection rewrite with split-order tests.
- [ ] Add fused residual LayerNorm kernels.
- [ ] Validate base and large BridgeTowerModel parity against Transformers.
- [ ] Validate ITM, MLM, and contrastive head parity.
- [ ] Benchmark processor, unimodal encoders, cross-modal fusion, and retrieval batch scoring separately.
