# KOSMOS-2 Transformers Audit

## 1. Source basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: `microsoft/kosmos-2-patch14-224`.

Config source: `https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/config.json`, repo API sha `e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c`; preprocessor config from the same repo. The Microsoft repo exposes in-library `transformers` weights/configs, not custom remote-code files. The `ydshieh/kosmos-2-patch14-224` mirror advertises historical custom code; that remote-code path is out of scope for this native-source audit.

Source files inspected:

- `src/transformers/models/kosmos2/configuration_kosmos2.py`
- `src/transformers/models/kosmos2/modeling_kosmos2.py`
- `src/transformers/models/kosmos2/processing_kosmos2.py`
- `src/transformers/models/clip/image_processing_clip.py`, because the checkpoint preprocessor declares `CLIPImageProcessor`
- `tests/models/kosmos2/test_modeling_kosmos2.py`
- HF files: `config.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `generation_config.json`

Any missing files or assumptions: there is no native `image_processing_kosmos2.py` or tokenizer implementation in the current local source; KOSMOS-2 composes `CLIPImageProcessor` and `XLMRobertaTokenizer`. I found one official Microsoft checkpoint for this family, so the checkpoint sweep separates the official config from source defaults and test/debug configs.

Primary runtime target: multimodal image-to-text generation with grounding-token postprocessing via `Kosmos2ForConditionalGeneration`.

## 2. High-level architecture

KOSMOS-2 is a vision encoder plus causal text decoder. Image features are not cross-attended by every text layer. Instead, the image path is projected into `latent_query_num` text-width vectors, and those vectors are inserted into reserved token positions before the causal decoder prefill.

Dataflow:

```text
image/text preprocessing -> CLIP-style pixel_values + input_ids + image mask
-> ViT vision encoder -> post LayerNorm -> L2 normalize
-> image-to-text projection cross-attention -> 64 image embeddings
-> masked embedding stitch into text token embeddings
-> causal decoder prefill -> cached autoregressive decode -> logits/sampling
-> optional grounding cleanup and patch-index-to-box postprocessing
```

Stage decomposition:

- CPU/data pipeline: resize/crop/rescale/normalize RGB images; tokenize text; insert 64 fake image tokens between `<image>` and `</image>`; build `image_embeds_position_mask`; optionally insert `<patch_index_XXXX>` tokens from supplied boxes.
- Independently cacheable image path: `pixel_values -> vision_model -> image_to_text_projection -> image_embeds` yields `[B, 64, 2048]` and can be passed directly on later calls instead of `pixel_values`.
- Prefix construction: text embeddings are gathered, image embeddings are indexed into positions where `image_embeds_position_mask == 1`, then sinusoidal positions are added.
- Prefill: causal decoder processes the full prompt including image placeholders and returns a self-attention KV cache.
- Decode: `prepare_inputs_for_generation` drops `image_embeds` and `image_embeds_position_mask` after the first cached iteration; decode is text-only with the image information already in cached hidden/KV state.

## 3. Important config dimensions

Official checkpoint dimensions from `config.json` unless noted:

| Component | Field | Value |
|---|---:|---:|
| global | `latent_query_num` | 64 |
| global | `torch_dtype` | `float32` |
| text | `vocab_size` | 65037 |
| text | `embed_dim` / hidden size | 2048 |
| text | `layers` | 24 |
| text | `attention_heads` | 32 |
| text | head dim | 64 |
| text | KV heads | 32, no GQA/MQA |
| text | `ffn_dim` | 8192 |
| text | `max_position_embeddings` | 2048 |
| text | position encoding | learned-free sinusoidal buffer, offset 2 |
| text | activation | `gelu` |
| text | dropout / attention dropout | 0.1 / 0.1 |
| text | `scale_embedding` | true, multiply token embeddings by `sqrt(2048)` |
| text | `use_cache` | true |
| text | tied embeddings | true, LM head aliases token embedding |
| vision | `hidden_size` | 1024 |
| vision | `num_hidden_layers` | 24 |
| vision | `num_attention_heads` | 16 |
| vision | head dim | 64 |
| vision | `intermediate_size` | 4096 |
| vision | `image_size` | 224 |
| vision | `patch_size` | 14 |
| vision | patches | 16 x 16 + CLS = 257 tokens |
| vision | activation | `quick_gelu` |
| image processor | output layout | NCHW `pixel_values`, `[B, 3, 224, 224]` |

Representative sweep:

| Source | Text dims | Vision dims | Image/query dims | Operator-significant notes |
|---|---:|---:|---:|---|
| `microsoft/kosmos-2-patch14-224` | 24L, 2048 hidden, 32 heads, 8192 FFN, vocab 65037 | 24L, 1024 hidden, 16 heads, 4096 FFN, 224/14 | 257 vision tokens, 64 latent queries | Production target; float32 config; tied LM head |
| source defaults | same as official | same as official | same | Defaults in config classes mirror the official checkpoint |
| unit-test tiny | 2L, 32 hidden, 4 heads, 37 FFN, vocab 99 | 2L, 32 hidden, 4 heads, 37 FFN, 32/4 | 65 vision tokens, 3 latent queries | Debug only; pipeline tests skipped because processor defaults still use 64 image tokens |
| integration resize path | official text | official vision with processor size/crop 180 | 145 vision tokens at 180/14 floor grid plus CLS | Requires `interpolate_pos_encoding=True`; otherwise source raises on image-size mismatch |

## 3a. Family variation traps

- The official native family has one published Microsoft checkpoint; mirrors may include old remote-code files, but native Transformers owns the current in-library path.
- The processor's `num_image_tokens` must match `config.latent_query_num`. Source defaults both to 64, but test tiny configs use `latent_query_num=3`; this mismatch is explicitly noted in skipped pipeline tests.
- Text uses full MHA, not GQA. `embed_dim == attention_heads * head_dim` for official text and vision.
- Text FFN is not the most common `Linear -> GELU -> Linear` only: it inserts `LayerNorm(ffn_dim)` between activation/dropout and `fc2`.
- Text self-attention adds an inner `LayerNorm(embed_dim)` on attention output before `out_proj`.
- Image features are stitched by boolean indexed assignment into token embeddings. Naive concatenation is wrong.
- `Kosmos2ForConditionalGeneration.generate` precomputes image embeddings before calling `text_model.generate`; image tensors are not repeatedly sent through the decoder during cached decode.
- KOSMOS-2 declares `_supports_sdpa=True` and `_supports_flash_attn=False`; FlashAttention is disabled in source comments due to CUDA device errors.
- Padding support is fragile in tests: several common padding/attention-backend tests are skipped with "KOSMOS-2 doesn't support padding", while a custom left-padding generation test exists for the image mask.
- Vision source is NCHW. NHWC is an optimization candidate only around controlled Conv2d/attention/MLP regions; source flatten order after Conv2d must be preserved.
- Vision input size guard rejects sizes other than config image size unless `interpolate_pos_encoding=True`; interpolation changes position embedding shape dynamically.
- Grounding boxes are textual patch-index tokens, not a detection head. End-to-end grounding parity includes processor postprocessing, not runtime NMS.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor handling, dtype cast to patch Conv2d weight dtype.
- Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)`.
- `flatten(2)`, `transpose(1, 2)`, `permute`, `view`, `reshape`, `contiguous`, `cat`, `expand`, `index_select`.
- Boolean mask conversion and indexed assignment: `inputs_embeds[mask.bool()] = image_embeds.view(-1, hidden)`.
- `cumsum`, `ne`, integer position-id creation for padded input ids.
- L2 normalize over `dim=-1` for image sequence features.
- Optional bicubic interpolation for vision position embeddings.

Neural network primitives:

- Embedding lookup for tokens, positions, and vision learned positions.
- LayerNorm at text hidden size 2048, text FFN size 8192, vision hidden size 1024.
- Linear projections with bias for all Q/K/V/out projections and MLP/FFN projections except LM head bias is false.
- GELU and QuickGELU.
- Dropout is inactive for inference but present in source.
- LM head `Linear(2048 -> 65037, bias=False)` tied to token embedding when `tie_word_embeddings=True`.

Attention primitives:

- Vision noncausal MHA: `[B, 16, S_v, 64]`, `S_v=257` for 224x224.
- Projection cross-attention: query latent `[B, 64, 2048]`; K/V from `[vision_tokens + 64] = 321` text-width states; 32 heads, head dim 64.
- Text causal self-attention with cache: 24 layers, 32 heads, head dim 64.
- Optional dormant text cross-attention if `text_config.add_cross_attention=True`; official config has false.
- Additive attention masks using `torch.finfo(dtype).min`.

Position/custom ops:

- Text sinusoidal positions with offset 2 and pad-token-aware position ids.
- Vision learned absolute embeddings; optional bicubic interpolation of patch embeddings.

Generation/cache ops:

- `DynamicCache` for text self-attention; `EncoderDecoderCache` only if optional cross-attention is enabled.
- Cache update per layer stores K/V after projection and head reshape.
- `logits_to_keep` last-token or indexed logits slicing before LM head.
- Generation-specific image input elision after the first cached iteration.

Preprocessing-coupled ops:

- CLIP image preprocessing: RGB convert, resize shortest edge 224, center crop 224x224, rescale by 1/255, normalize by CLIP mean/std.
- KOSMOS token insertion for `<image>`, `</image>`, `<grounding>`, phrase/object tags, delimiter, and 1024 patch-index tokens.
- `image_embeds_position_mask` construction and left/right padding adjustment.
- Grounding text cleanup and patch-index-to-normalized-coordinate conversion.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [B, 3, H, W] NCHW
patch = Conv2d(3 -> 1024, kernel=14, stride=14, bias=False)
patch = flatten spatial -> transpose -> [B, (H/14)*(W/14), 1024]
cls = learned [1024] expanded to [B, 1, 1024]
x = cat(cls, patch, dim=1) + learned/interpolated pos [1, S_v, 1024]
```

Vision block, repeated 24 times:

```text
res = x
x = LayerNorm(1024)(x)
q,k,v = Linear(1024 -> 1024, bias=True)(x)
q,k,v -> [B, 16, S_v, 64]
x = attention(q, k, v, noncausal)
x = Linear(1024 -> 1024, bias=True)(x)
x = res + x
res = x
x = LayerNorm(1024)(x)
x = Linear(1024 -> 4096) -> QuickGELU -> Linear(4096 -> 1024)
x = res + x
```

Image-to-text projection:

```text
vision_seq = post_layernorm(vision_last_hidden_state)      # [B, 257, 1024]
vision_seq = normalize(vision_seq, dim=-1)
h = Linear(1024 -> 2048)(vision_seq)                       # [B, 257, 2048]
latent = learned [64, 2048] expanded to [B, 64, 2048]
kv = cat(h, latent, dim=1)                                  # [B, 321, 2048]
image_embeds = cross_attention(query=latent, key_value=kv)  # [B, 64, 2048]
```

Text embedding and decoder block, repeated 24 times:

```text
tok = Embedding(input_ids) or inputs_embeds
tok[image_mask.bool()] = image_embeds.view(-1, 2048)
tok = tok * sqrt(2048)
x = tok + sinusoidal_positions(input_ids or inputs_embeds, past_len)
x = dropout(x)

res = x
x = LayerNorm(2048)(x)
q,k,v = Linear(2048 -> 2048, bias=True)(x)
q,k,v -> [B, 32, T, 64]
k,v = cache.update(k, v, layer_idx) if cache present
x = causal_attention(q, k, v, additive_mask)
x = LayerNorm(2048)(x)          # inner attention LN
x = Linear(2048 -> 2048)(x)
x = res + dropout(x)

res = x
x = LayerNorm(2048)(x)
x = Linear(2048 -> 8192) -> GELU -> dropout -> LayerNorm(8192)
x = Linear(8192 -> 2048) -> dropout
x = res + x
```

LM head:

```text
x = final LayerNorm(2048)
logits = tied Linear(2048 -> 65037, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

Vision attention is noncausal self-attention, MHA 16 heads, head dim 64. It has no KV cache. Source attention math for eager backend is `matmul(q, k.T) * scale`, add mask, softmax in current dtype, dropout, `matmul(weights, v)`. The source comment explicitly says KOSMOS-2 eager attention does not cast attention weights to fp32.

Projection attention is cross-attention implemented by `KosmosTextAttention` with `is_decoder=False`, 32 heads, head dim 64. It receives latent queries as hidden states and concatenated image/text-width features plus latent queries as encoder states. It uses no cache in the image projector.

Text attention is causal decoder self-attention, MHA 32 heads, head dim 64. Official checkpoint has no sliding window, ALiBi, RoPE, GQA/MQA, or packed varlen metadata. The additive causal mask shape is `[B, 1, T_q, T_k]`, with previous cache length prepended as zeros. Padding masks, when supplied, are expanded to `[B, 1, T_q, S]` and added to the causal mask.

KV cache:

- Per text layer key/value before expansion: `[B, 32, cache_seq, 64]`.
- There is no repeat-KV step because KV heads equal query heads.
- Cached keys are plain projected keys; there is no RoPE or relative position transform to store before/after.
- During prefill with image placeholders, cache length includes the image-token positions in `input_ids`.
- During decode, image embeddings are not restitched; generated token K/V extends the cached text stream.
- Optional cross-attention cache exists only if a non-default config enables text cross-attention.

Backend compatibility: `_supports_sdpa=True`, `_supports_flash_attn=False`. A DinoML first pass can implement eager/SDPA-compatible dense attention; FlashAttention should remain guarded until parity is proven for KOSMOS-2 masks and dtype behavior.

## 7. Position encoding and custom math

Text uses M2M100-style sinusoidal embeddings with `offset = 2`. For `input_ids`, position ids skip padding and start at `padding_idx + 1`; with pad id 1, first non-pad position is 2. With `inputs_embeds`, positions are sequential because pad tokens cannot be inferred.

```python
def kosmos2_position_ids(input_ids, padding_idx, past_len):
    mask = (input_ids != padding_idx).int()
    incremental = (cumsum(mask, dim=1) + past_len) * mask
    return incremental.long() + padding_idx
```

The sinusoidal table can be precomputed up to `max_position_embeddings + 2`, but source expands it dynamically if `padding_idx + 1 + seq_len + past_len` exceeds the current buffer.

Vision positions are learned absolute embeddings with an optional interpolation path:

```python
def interpolate_vision_pos(pos, height, width, patch_size):
    cls = pos[:, :1]
    patch = pos[:, 1:].reshape(1, sqrt_n, sqrt_n, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_size, width // patch_size), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return cat([cls, patch], dim=1)
```

Grounding coordinate math is processor postprocessing, not model graph math:

```python
cell = 1.0 / 32
ul_x, ul_y = ul_idx % 32, ul_idx // 32
lr_x, lr_y = lr_idx % 32, lr_idx // 32
# same row/col boxes use cell boundaries; diagonal boxes use cell centers
```

## 8. Preprocessing and input packing

Image preprocessing from `preprocessor_config.json` and `CLIPImageProcessor`:

- Convert to RGB.
- Resize with bicubic resampling; official config has `size.shortest_edge=224` and `use_square_size=true`, which the CLIP processor converts to square 224x224 sizing.
- Center crop to 224x224.
- Rescale by `0.00392156862745098`.
- Normalize by mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Return `pixel_values` in channel-first layout `[B, 3, 224, 224]`.

Text/processor coupling:

- Base tokenizer is `XLMRobertaTokenizer`; BOS id 0, pad id 1, EOS id 2, UNK id 3.
- Processor adds non-special tag tokens and patch-index tokens. In the official tokenizer config these include `<image>` id 64003, `</image>` 64004, `<phrase>` 64007, `</phrase>` 64008, `<object>` 64009, `</object>` 64010, delimiter 64011, `<grounding>` 64012, and `<patch_index_0000>` through `<patch_index_1023>` at 64013 through 65036.
- If images are present, text is prefixed with `<image>` plus 64 fake `<image>` tokens plus `</image>`.
- The processor rewrites the 64 fake image token ids to `range(first_image_token_id, first_image_token_id + num_image_tokens)`. If unset, `first_image_token_id = tokenizer.unk_token_id + 1`, which is 4 for the official tokenizer.
- `image_embeds_position_mask` has zeros for BOS, the real BOI token, and EOI token; ones for the 64 image embedding slots.
- Padding is custom-adjusted for batched image/text inputs and must pad `input_ids`, `attention_mask`, and `image_embeds_position_mask` consistently.

Embedding stitch:

```text
inputs_embeds = embed_tokens(input_ids)
inputs_embeds[image_embeds_position_mask.bool()] = image_embeds.view(-1, 2048)
inputs_embeds *= sqrt(2048)
inputs_embeds += sinusoidal_positions(...)
```

Shape guard needed in DinoML: for each batch, the true count of ones in `image_embeds_position_mask` must equal `latent_query_num`; globally it must equal `B * latent_query_num`, or the indexed assignment will fail or misalign.

Generation controller behavior:

- `Kosmos2ForConditionalGeneration.generate` accepts `pixel_values` or `inputs`; if `image_embeds` is absent it computes `image_embeds` once, then calls `text_model.generate`.
- `Kosmos2TextForCausalLM.prepare_inputs_for_generation` removes image inputs on cached non-first iterations and removes generated `position_ids` so the model's sinusoidal embedding module creates the KOSMOS-specific offset positions.
- `generation_config.json` only supplies BOS/EOS/PAD ids; no forced decoder ids or suppress-token processors are present.

Postprocessing:

- `post_process_generation` drops text before the last `</image>`.
- `clean_text_and_extract_entities_with_bboxes` removes tags, extracts `<object><patch_index_XXXX><patch_index_YYYY></object>` pairs, maps them to normalized coordinates on a 32x32 grid, and adjusts character spans after tag removal.
- There is no NMS, score thresholding, detection head, or mask resizing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding to GEMM

Source pattern: `Conv2d(3 -> 1024, kernel=14, stride=14, padding=0, dilation=1, groups=1, bias=False)` followed by flatten spatial and transpose to token sequence.

Replacement:

```text
NCHW image -> unfold non-overlapping 14x14 patches in row-major spatial order
-> MatMul(patch_flat, weight_flat.T)
-> reshape [B, H/14 * W/14, 1024]
```

Preconditions:

- Input is NCHW and contiguous or lowered with equivalent strides.
- `H % 14 == 0`, `W % 14 == 0`.
- Kernel equals stride, padding 0, dilation 1, groups 1.
- Preserve PyTorch Conv2d flatten order: channel, kernel_y, kernel_x within each output position.
- Bias is absent for official patch embedding.

Failure cases: interpolated-position dynamic sizes still allow this rewrite if divisibility holds; non-divisible resize/crop sizes need fallback or explicit guard.

Parity test sketch: compare Conv2d output after `flatten(2).transpose(1, 2)` with unfolded GEMM at 224 and 180 input sizes.

### Rewrite: image projection as static cross-attention block

Source pattern: `dense(vision_seq)`, expand learned latent query, concatenate `[vision_seq_projected, latent]`, then MHA with latent as query and concat as K/V.

Replacement: keep as one attention subgraph with fixed query length 64 and key/value length `S_v + 64`.

Preconditions:

- `latent_query_num` fixed for compiled artifact or bucketed.
- No cache and no causal mask.
- Hidden size 2048, 32 heads for official checkpoint.

Failure cases: changing `latent_query_num`, enabling attention dropout in training, or altered text head count.

Parity test sketch: random vision features through native projection versus lowered dense + attention subgraph.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before `lm_head`.

Replacement: for decode, apply tied LM head only to `[B, 1, 2048]`.

Preconditions:

- Caller requests `logits_to_keep=1` or generation passes last token only.
- No loss computation requiring full shifted logits.

Failure cases: training/loss path, scoring full prompt, tensor-valued arbitrary `logits_to_keep`.

### Rewrite: guarded image-embedding precompute

Source pattern: `pixel_values -> vision_model -> post_layernorm -> normalize -> image_to_text_projection`.

Replacement: expose `image_embeds` as a cacheable input to decoder prefill.

Preconditions:

- Same image, same processor options, same vision/projection weights.
- `image_embeds` shape `[B, latent_query_num, text_hidden]` and dtype matches text embedding path.

Failure cases: image augmentation, interpolated position encoding differences, or batch/image ordering changes.

### Layout guard: NCHW vision region

Source region is NCHW through Conv2d, then token-major `[B, S, C]`. A channel-last optimization can be local to patch extraction/Conv2d if it rewrites Conv2d weight layout and preserves output token order. Do not globally translate axes for LayerNorm, attention softmax, `normalize(dim=-1)`, or text embedding stitch.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder LayerNorm + QKV projection + attention for 24 layers; this dominates prefill/decode.
- Cached causal MHA with `[B, 32, T, 64]` K/V and additive causal mask.
- GELU FFN with intermediate LayerNorm: `Linear(2048 -> 8192) -> GELU -> LayerNorm(8192) -> Linear(8192 -> 2048)`.
- Last-token-only tied LM head for decode.

Medium priority:

- Vision patch Conv2d lowered to GEMM.
- Vision MHA/MLP fusion for 24 layers over short sequence 257.
- Projection dense + fixed-query cross-attention, since it is independently cacheable and has fixed query length 64.
- L2 normalize over image feature dim.

Lower priority:

- Bicubic position interpolation path for non-224 images.
- Processor postprocessing acceleration; it is CPU string/regex work and not a GPU runtime bottleneck.
- Dropout/layerdrop training paths.

## 11. Runtime staging plan

Stage 1: parse nested `Kosmos2Config`, load tied text embeddings/LM head once, and reject remote-code/custom variants for this native target.

Stage 2: implement the vision encoder at 224x224 only, including patch Conv2d, learned positions, LayerNorm, MHA, QuickGELU MLP, and post LayerNorm parity.

Stage 3: implement image-to-text projection and validate `image_embeds` parity independently.

Stage 4: implement text embedding stitch, sinusoidal positions, one decoder block, then full prefill logits.

Stage 5: implement `DynamicCache` decode for text self-attention and `prepare_inputs_for_generation` behavior that drops image embeddings after prefill.

Stage 6: add processor-compatible input contracts and grounding postprocessing in the host layer.

Stage 7: enable optimized attention/GEMM/fusion paths with layout guards and last-token logits.

Initially stubbable: training loss, labels path, optional `add_cross_attention=True`, beam search beyond using generic generation controller, position interpolation for non-224 images, and full tokenizer implementation if inputs are supplied as tensors.

## 12. Parity and validation plan

- Random tensor tests for `_expand_mask`, `_make_causal_mask`, KOSMOS sinusoidal position ids, and patch-index coordinate conversion.
- Patch embedding parity for 224x224 NCHW input and a guarded 180x180 interpolation case.
- Single vision layer parity in fp32 and reduced precision if supported.
- Full vision encoder parity: `last_hidden_state [B, 257, 1024]` and post-layernorm normalized sequence.
- Image projector parity: `[B, 64, 2048]` image embeddings and optional projection attention weights.
- Embedding stitch parity with a mask containing exactly 64 ones per sample; include left-padding mask case.
- Single text block and full text prefill logits parity with `use_cache=False` and `use_cache=True`.
- Decode parity: prefill with image embeddings, then one and several generated tokens with cached K/V; verify image embeddings are not re-applied.
- End-to-end official snowman prompt parity using generated ids and grounding postprocessing from the Transformers integration test.

Suggested tolerances: fp32 `rtol=1e-5, atol=1e-5` for logits slices; fp16/bf16 `rtol=1e-2, atol=1e-2` for hidden states/logits unless attention backend changes require a documented relaxation.

## 13. Performance probes

- CPU preprocessing throughput: image resize/crop/normalize plus tokenizer/image-mask construction.
- Vision encoder throughput for batch size sweep at 224x224.
- Image projector throughput separately from vision encoder.
- Text prefill throughput versus prompt length, with and without image embeddings already precomputed.
- Decode tokens/sec with KV cache, batch size sweep, and last-token LM head enabled.
- KV cache memory: `24 layers * 2 * B * heads(32) * seq * head_dim(64) * dtype_size`.
- Attention backend comparison: eager-compatible, SDPA, and any future fused backend with KOSMOS dtype/mask parity.
- Layout rewrite probe: NCHW Conv2d versus unfolded GEMM patch embedding.
- End-to-end requests/hour split into preprocessing, vision/projector, prefill, decode, and postprocessing.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, layerdrop, and gradient checkpointing.
- Optional `text_config.add_cross_attention=True`; official checkpoint disables it.
- FlashAttention enablement; source marks it unsupported.
- Historical remote-code mirror implementations.
- Non-224 image interpolation for first integration.
- Beam search-specific optimization; correctness can use generic generation first.
- Quantization and multi-GPU tensor parallelism.
- Full detection-style postprocessing; KOSMOS-2 grounding is token/regex based with no NMS.

## 15. Final implementation checklist

- [ ] Parse `Kosmos2Config` with nested text and vision configs.
- [ ] Load official weights with tied `text_model.lm_head.weight` and `text_model.model.embed_tokens.weight`.
- [ ] Implement CLIP-style image preprocessing contract or accept preprocessed NCHW `pixel_values`.
- [ ] Implement vision patch Conv2d and optional guarded Conv2d-to-GEMM rewrite.
- [ ] Implement vision MHA, QuickGELU MLP, and LayerNorm stack.
- [ ] Implement post-layernorm plus `normalize(dim=-1)` image feature step.
- [ ] Implement image-to-text projection dense + fixed-query cross-attention.
- [ ] Implement token embedding stitch with `image_embeds_position_mask` shape/count guard.
- [ ] Implement KOSMOS sinusoidal position ids and dynamic table growth or explicit max guard.
- [ ] Implement causal text decoder prefill and `DynamicCache` decode.
- [ ] Implement additive causal/padding masks with dtype-min fill.
- [ ] Implement `logits_to_keep` and last-token-only LM head lowering.
- [ ] Implement generation input handling that precomputes image embeddings once and drops them during cached decode.
- [ ] Implement grounding postprocessing for patch-index tokens.
- [ ] Add parity tests for vision, projector, stitch, prefill, decode, and snowman-style end-to-end generation.
- [ ] Benchmark preprocessing, vision/projector, prefill, decode, cache memory, and attention backend variants.
