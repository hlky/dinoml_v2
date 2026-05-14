# PaliGemma Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/paligemma-3b-pt-224, google/paligemma-3b-mix-224, google/paligemma-3b-pt-448, google/paligemma2-3b/10b/28b-style variants by converter naming
Config source: local Transformers config/converter defaults; direct google/* config fetches returned 401/gated
Source files inspected:
- X:/H/transformers/src/transformers/models/paligemma/configuration_paligemma.py
- X:/H/transformers/src/transformers/models/paligemma/modeling_paligemma.py
- X:/H/transformers/src/transformers/models/paligemma/processing_paligemma.py
- X:/H/transformers/src/transformers/models/paligemma/convert_paligemma_weights_to_hf.py
- X:/H/transformers/src/transformers/models/paligemma/convert_paligemma2_weights_to_hf.py
- X:/H/transformers/src/transformers/models/siglip/modeling_siglip.py
- X:/H/transformers/src/transformers/models/siglip/image_processing_siglip.py
- X:/H/transformers/src/transformers/models/gemma/modeling_gemma.py
- X:/H/transformers/src/transformers/models/gemma/configuration_gemma.py
- X:/H/transformers/src/transformers/models/gemma2/modeling_gemma2.py
- X:/H/transformers/src/transformers/models/gemma2/configuration_gemma2.py
Any missing files or assumptions: official google checkpoint configs/processors are gated from this environment. Representative checkpoint dimensions below come from in-library converter/config defaults, not downloaded config.json.
```

Gemma and Gemma2 generated files state they are generated from `modular_gemma.py` and `modular_gemma2.py`; future Transformers source edits should inspect the modular files first. This report owns the PaliGemma composition layer and records the composed Gemma/SigLIP requirements needed for integration planning.

## 2. High-level architecture

PaliGemma is a multimodal conditional generation wrapper:

```text
CPU image/text preprocessing -> SigLIP vision encoder -> linear multimodal projector
  -> image-token embedding stitch into Gemma/Gemma2 token embeddings
  -> decoder prefill with bidirectional image/prefix mask support
  -> autoregressive decode with KV cache -> tied/untied LM logits
```

Stage decomposition:

```text
Data pipeline:
  resize/rescale/normalize RGB image to NCHW pixel_values
  expand each <image> placeholder into image_seq_length repeated token IDs
  prepend image placeholders + BOS when text omits explicit placeholders

Cacheable vision/projector stage:
  pixel_values [num_images_total, 3, H, W]
  SigLIP patch tokens [num_images_total, image_seq_length, 1152]
  projector [1152 -> text_hidden] to image_features

Prefill:
  embed text tokens, replace image placeholder embeddings with image_features,
  construct causal/full/sliding masks, run Gemma/Gemma2 over the packed sequence

Decode:
  omit pixel_values after the first generation iteration when use_cache=True;
  continue with cached self-attention keys/values and 1-indexed positions
```

The SigLIP tower plus projector can be validated independently from the text decoder. The embedding stitch and PaliGemma mask creation are the model-specific bridge and need their own parity tests.

## 3. Important config dimensions

Source-derived defaults and converter-derived representative variants:

| Variant source | Text body | Image size | Image tokens | Vision hidden/layers/heads | Text hidden/layers/heads/KV/head_dim | MLP size | Projection | Notes |
|---|---:|---:|---:|---|---|---:|---:|---|
| `PaliGemmaConfig()` default | Gemma | 224 | 256 | 1152 / 27 / 16 | 2048 / 18 / 8 / 1 / 256 | 16384 | 2048 | Source config default has vision intermediate 4096; converter uses 4304 for released PaliGemma1. |
| converter `3b-224px` | Gemma | 224 | 256 | 1152 / 27 / 16 | 2048 / 18 / 8 / 1 / 256 | 16384 | 2048 | `image_token_id=257152`, vocab 257152. |
| converter `3b-448px` | Gemma | 448 | 1024 | 1152 / 27 / 16 | 2048 / 18 / 8 / 1 / 256 | 16384 | 2048 | Same operators, larger patch sequence. |
| converter `3b-896px` | Gemma | 896 | 4096 | 1152 / 27 / 16 | 2048 / 18 / 8 / 1 / 256 | 16384 | 2048 | Prefill attention and stitch cost dominate. |
| converter `2b-224` PaliGemma2 | Gemma2 | 224 | 256 | 1152 / 27 / 16 | 2304 / 26 / 8 / 4 / 256 | 9216 | 2304 | Alternating sliding/full layer types by Gemma2 default. |
| converter `9b-448` PaliGemma2 | Gemma2 | 448 | 1024 | 1152 / 27 / 16 | 3584 / 42 / 16 / 8 / 256 | 14336 | 3584 | GQA, larger hidden. |
| converter `27b-896` PaliGemma2 | Gemma2 | 896 | 4096 | 1152 / 27 / 16 | 4608 / 46 / 32 / 16 / 128 | 36864 | 4608 | Head dim differs; query scale is `4608 // 32`. |

Common fields:

| Field | Effective behavior |
|---|---|
| `image_token_index` / `image_token_id` | PaliGemma wrapper aliases these; image token may be outside text vocab and is replaced with PAD before token embedding lookup. |
| `text_config.use_bidirectional_attention` | If unset, PaliGemma forces it to `True`; token type IDs then allow bidirectional image/prefix blocks and causal suffix behavior. |
| `rope_parameters` | Config default supplies normal RoPE parameters through Transformers defaults; Gemma/Gemma2 code expects `rope_parameters["rope_theta"]` and `rope_type`. |
| `use_cache` | Text decoder cache is supported; vision/projector outputs are not a KV cache but can be precomputed before prefill. |
| `tie_word_embeddings` | PaliGemmaForConditionalGeneration ties `lm_head.weight` to `model.language_model.embed_tokens.weight` when config tie policy is active. |

## 3a. Family variation traps

- Same PaliGemma wrapper can instantiate Gemma or Gemma2 through `AutoModel.from_config`; DinoML should dispatch on `text_config.model_type`.
- PaliGemma2 uses Gemma2-only attention softcapping, final logit softcapping, pre/post attention and FFN RMSNorms, GQA, and alternating `sliding_attention` / `full_attention` layer types unless `layer_types` overrides them.
- PaliGemma1 Gemma has MQA-like `num_key_value_heads=1`; do not assume MHA.
- `hidden_size` does not always equal `num_attention_heads * head_dim` in Gemma2 2B/9B converter variants: Q/O width is `num_heads * head_dim`, while residual hidden width can be different. Lowering must use projection shapes from config, not infer all widths from hidden size.
- `hidden_act` versus `hidden_activation` differs between Gemma and Gemma2.
- Image token placeholders are repeated by the processor. The model checks placeholder count equals flattened image feature elements before `masked_scatter`.
- Processor can handle multiple images per prompt by expanding one `<image>` per image into `image_seq_length` tokens; this makes prefill sequence length data-dependent on prompt image count.
- Source image tensors are NCHW. NHWC/channel-last is an optimization candidate only inside guarded vision regions; the text embedding stitch and decoder sequence axes must be protected from layout translation.
- SigLIP `interpolate_pos_encoding` changes positional embedding math for non-square or non-trained resolutions; released PaliGemma processors resize to fixed square sizes, so first integration can reject interpolation.
- Official checkpoint configs are gated here; report facts about released variants are from converter defaults unless otherwise stated.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input, Conv2d patch embed with `kernel=stride=14`, `padding=valid`, `groups=1`.
- Flatten spatial patch map, transpose `[B, C, Gh, Gw] -> [B, Gh*Gw, C]`.
- Embedding lookup, scalar multiply by `sqrt(hidden_size)` in Gemma/Gemma2 token embeddings.
- Boolean compare for image token IDs, clone/index replacement of OOV image IDs with PAD.
- `masked_scatter` / indexed copy from projected image features into token embeddings.
- Arange/unsqueeze/add for 1-indexed PaliGemma `position_ids`.
- Reshape/view/transpose/contiguous for QKV and attention outputs.
- Slice last logits via `logits_to_keep`.

Neural network primitives:

- SigLIP vision: LayerNorm, Linear with bias, GELU/tanh GELU, residual add.
- PaliGemma projector: Linear `[1152 -> text_hidden]` with bias.
- Gemma/Gemma2 decoder: RMSNorm with `(1 + weight)` scale, biasless Q/K/V/O and MLP projections, gated GELU MLP `down(act(gate(x)) * up(x))`.
- LM head: Linear `[text_hidden -> vocab_size]`, bias=False, tied to input embedding when enabled.
- Gemma2 extras: attention logit softcap `softcap * tanh(score / softcap)` and final logit softcap.

Attention primitives:

- SigLIP noncausal encoder self-attention: MHA, 16 heads, head dim 72 for hidden 1152.
- Gemma causal/bidirectional self-attention with RoPE, MQA/GQA, KV cache.
- Gemma2 full and sliding-window causal masks, GQA, softcapped attention logits.
- Backend dispatch supports eager, SDPA, FlashAttention, and FlexAttention through Transformers attention interfaces.

Preprocessing-coupled ops:

- RGB conversion, resize, rescale, normalize with SigLIP image processor defaults.
- Placeholder expansion: each `<image>` becomes `image_seq_length` token occurrences.
- Token type IDs and labels: processor always requests token type IDs, masks labels where `token_type_ids == 0`, and can optionally return multimodal token type IDs.

Layout guard notes:

- Vision Conv2d-to-GEMM or NHWC rewrite must rewrite flatten/transpose consumers and preserve patch order.
- Decoder axes are `[batch, seq, hidden]`; attention head axes are `[batch, heads, seq, head_dim]`. Do not apply image layout rules past the projector output.

## 5. Layer/block breakdown

SigLIP vision stage, 27 layers:

```text
pixel_values [B_img, 3, H, W]
patch = Conv2d(3 -> 1152, kernel=stride=14)(pixel_values)
x = flatten_spatial(patch).transpose(1, 2) + learned_position[Gh*Gw]
repeat 27:
  y = LayerNorm(x)
  q,k,v = Linear(1152 -> 1152, bias=True)(y)
  y = noncausal MHA(q,k,v)
  x = x + Linear(1152 -> 1152, bias=True)(y)
  y = LayerNorm(x)
  y = Linear(1152 -> 4304, bias=True)(y)
  y = GELU/tanh-GELU variant
  y = Linear(4304 -> 1152, bias=True)(y)
  x = x + y
x = post LayerNorm(x)
image_features = Linear(1152 -> text_hidden, bias=True)(x)
```

Gemma decoder block, repeated `N`:

```text
x0 = x
y = RMSNorm(x)
q = Linear(hidden -> num_heads * head_dim, bias=False)(y)
k = Linear(hidden -> num_kv_heads * head_dim, bias=False)(y)
v = Linear(hidden -> num_kv_heads * head_dim, bias=False)(y)
q,k = RoPE(q,k, 1-indexed PaliGemma positions)
k,v = KVCache.update(k,v, layer)
y = attention(q, repeat_kv(k), repeat_kv(v), mask)
x = x0 + Linear(num_heads * head_dim -> hidden, bias=False)(y)
x0 = x
y = RMSNorm(x)
y = down_proj(act(gate_proj(y)) * up_proj(y))
x = x0 + y
```

Gemma2 block differs:

```text
y = input_rmsnorm(x)
y = attention_with_optional_sliding_mask_and_softcap(y)
y = post_attention_rmsnorm(y)
x = residual + y
y = pre_ffn_rmsnorm(x)
y = gated_mlp(y)
y = post_ffn_rmsnorm(y)
x = residual + y
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch sequence only.
- MHA, no KV cache, dropout disabled for inference.
- Mask is normally absent for PaliGemma vision.

Text attention:

- Autoregressive generation path with self-attention KV cache.
- PaliGemma wrapper sets 1-indexed positions before entering Gemma/Gemma2.
- PaliGemma1 released converter dims: 8 query heads, 1 KV head, head_dim 256. Cache per layer stores K/V before repeat expansion as `[batch, num_kv_heads, cached_seq, head_dim]`, i.e. `[B, 1, S, 256]`.
- PaliGemma2 converter dims: 2B `[B, 4, S, 256]`, 9B `[B, 8, S, 256]`, 27B `[B, 16, S, 128]` before repeat expansion.
- Cached keys are stored after RoPE has been applied, because attention applies RoPE before `past_key_values.update`.
- PaliGemma can use bidirectional prefix/image attention via `token_type_ids`; the wrapper passes block sequence IDs where `token_type_ids == 0` maps to bidirectional group `0` and suffix maps to causal `-1`.
- Gemma2 needs both full and sliding masks when `sliding_window` is set, and selects per layer using `layer_types`.
- Eager fallback repeats KV heads, materializes attention weights, applies optional mask, softmax in fp32, downcasts, then computes AV. This is too slow for production prefill/decode at 448/896 image-token lengths; fused GQA attention with cache and sliding-window support is a priority.

## 7. Position encoding and custom math

Gemma/Gemma2 RoPE is source-compatible at the operation level:

```python
def apply_gemma_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

PaliGemma-specific position trap:

```text
position_ids = arange(seq_len) + past_seen_tokens
position_ids = position_ids[None, :] + 1
```

RoPE cos/sin computation is forced to fp32 internally and cast back to hidden dtype. It can be precomputed for static max positions and reused by decode if position IDs are monotonic. Dynamic RoPE variants are technically supported through `rope_parameters["rope_type"]`, but no PaliGemma converter variant inspected requires a non-default type.

SigLIP position encoding is learned 2D patch-position embedding flattened to sequence. Interpolation uses bicubic resize of the learned grid and should be deferred unless dynamic image sizes are admitted.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- `PaliGemmaProcessor` requires `images`; text defaults to `""` for captioning.
- `SiglipImageProcessor` defaults: convert RGB, resize to configured square size, rescale, normalize with ImageNet standard mean/std, output `pixel_values` in channels-first format.
- Processor requires `image_processor.image_seq_length`; converters set it to `vision_config.num_image_tokens`.
- If text lacks `<image>`, the processor infers images per prompt and constructs:

```text
"<image>" * image_seq_length * num_images + bos_token + prompt + "\n"
```

- If text contains `<image>`, each occurrence is expanded to `image_seq_length` consecutive tokens and BOS is inserted after the final image token run.
- Tokenizer adds `<image>` if missing, adds 1024 `<loc0000>...` tokens and 128 `<seg000>...` tokens, and disables automatic BOS/EOS.
- Suffix is training-oriented: suffix text gets EOS appended and labels mask prefix/image tokens via token type IDs. First inference can ignore suffix/labels.

GPU/runtime packing:

- `input_ids` include repeated image placeholder IDs. If `image_token_id >= vocab_size`, those positions are copied to `0` before embedding lookup.
- `image_features` shape is `[num_images_total, image_seq_length, text_hidden]`.
- `special_image_mask` is `[batch, seq, hidden]`, expanded from `input_ids == image_token_id`.
- `inputs_embeds.masked_scatter(special_image_mask, image_features)` performs the stitch. DinoML can lower this as a guarded indexed copy where total placeholder tokens equals `num_images_total * image_seq_length`.
- Multiple images are flattened through the image batch dimension; prompt grouping is recovered only by placeholder locations in `input_ids`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP patch Conv2d -> Linear

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, `dilation == 1`, `groups == 1`.
- NCHW source tensor has height/width divisible by patch size.
- Consumer is exactly flatten spatial then transpose to `[B, patches, hidden]`.

Replacement:

```text
WindowFlatten_NCHW(patch=14, stride=14) -> GEMM(input_flat, weight_flat.T) -> BiasAdd -> Reshape [B, patches, hidden]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases: non-square dynamic images without positional interpolation support, changed padding/dilation/groups, or downstream consumers needing the 4D conv map.

Parity test sketch: compare patch embeddings before position add for random NCHW fp32/fp16 tensors at 224 and 448.

### Rewrite: image placeholder masked_scatter -> deterministic indexed copy

Preconditions:

- Placeholder mask comes from `input_ids == image_token_id`.
- Placeholder count equals `num_images_total * image_seq_length`.
- Image placeholders appear in processor order.
- `image_features` is contiguous `[num_images_total, image_seq_length, hidden]`.

Replacement:

```text
embeds = token_embedding(llm_input_ids)
positions = nonzero(input_ids == image_token_id)
embeds[positions, :] = image_features.reshape(-1, hidden)
```

Failure cases: caller supplies `inputs_embeds` without `input_ids`, nonstandard placeholder embedding equality path, or mismatched image token count.

### Rewrite: last-token-only logits

Preconditions:

- `logits_to_keep == 1` or decode step only needs the newest token.
- No loss computation over full labels.

Replacement:

```text
hidden[:, -1:, :] -> LM head
```

Failure cases: training loss, scoring full sequences, tensor-valued `logits_to_keep`.

### Layout rewrite: channel-last vision island

Preconditions:

- Entire Conv2d/LayerNorm/attention/MLP vision tower is lowered in one controlled region.
- Patch flatten order and learned position index order are preserved.
- Outputs return to `[B_img, patches, hidden]` before projector.

Required guards: no layout translation across processor boundary, placeholder stitch, decoder, or sequence-axis ops.

## 10. Kernel fusion candidates

Highest priority:

- GQA/MQA FlashAttention with KV cache, bidirectional prefix mask, Gemma2 sliding-window masks, and softcap support.
- RMSNorm `(1 + weight)` fusion for Gemma/Gemma2.
- Gated MLP fusion: `gate_proj`, `up_proj`, activation, multiply, `down_proj`.
- Placeholder stitch as a fused indexed embedding update to avoid materializing huge `[B,S,H]` boolean masks.
- Last-token logits GEMM for decode.

Medium priority:

- SigLIP patch Conv2d as GEMM and vision MLP GEMM epilogues.
- SigLIP LayerNorm + QKV projection fusion for large image-token prefill.
- RoPE generation/application fused with Q/K projection or attention prefill.
- Gemma2 softcap fused into attention score path and final logits.

Lower priority:

- Position embedding interpolation for dynamic image sizes.
- Processor-side token type/label generation on GPU.
- Multi-image batching layout optimizations beyond the deterministic flatten order.

## 11. Runtime staging plan

1. Parse PaliGemma config and reject unsupported text backbones except `gemma` and `gemma2`.
2. Load weights with alias preservation for tied input embedding / LM head.
3. Implement processor-compatible input contract using fixed image sizes and one or more images per prompt.
4. Validate SigLIP vision tower plus multimodal projector independently.
5. Implement embedding stitch and PaliGemma 1-indexed position IDs.
6. Run one Gemma/Gemma2 decoder block parity with stitched embeddings.
7. Run full prefill logits parity for 224 variants.
8. Add decode with KV cache, omitting `pixel_values` after first cached iteration.
9. Add optimized attention and last-token logits.
10. Broaden to 448/896 and PaliGemma2 9B/27B only after memory and sliding-window paths are stable.

Initially stub/defer suffix labels, training losses, image-position interpolation, and generation-controller extras beyond greedy/sampling parity.

## 12. Parity and validation plan

- Processor parity: compare `input_ids`, `token_type_ids`, `attention_mask`, `labels`, and `pixel_values` for explicit `<image>`, implicit `<image>`, batched images, and two images per prompt.
- Patch embedding rewrite parity at 224 and 448, fp32 tolerance `1e-5`, reduced precision tolerance `1e-2`.
- SigLIP layer parity after 1, 3, and 27 layers.
- Projector parity for `[num_images, image_seq_length, 1152] -> [num_images, image_seq_length, hidden]`.
- Placeholder stitch parity including OOV image token replacement and mismatch error path.
- Gemma/Gemma2 one-block parity with synthetic stitched embeddings and masks.
- Prefill logits parity for 224 PaliGemma1 and PaliGemma2, with `logits_to_keep=1` and full logits.
- Decode parity for 4-8 generated tokens, checking cache sequence length and K/V shapes before repeat expansion.
- Multi-image parity using processor packing and flattened image batch order.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=2e-2, atol=2e-2` for full model logits, with tighter per-op tolerances where accumulation is fp32.

## 13. Performance probes

- CPU processor throughput: images/sec resize/normalize and tokens/sec placeholder expansion.
- Vision-only throughput for 224/448/896 and batch sweep.
- Projector throughput and memory bandwidth.
- Prefill throughput versus total sequence length: text-only, 1 image, 2 images, 224/448/896.
- Decode tokens/sec with cache for batch sizes 1, 4, 16 and prompt lengths with image prefixes.
- KV cache memory per layer and total: PaliGemma1 MQA versus PaliGemma2 GQA.
- Attention backend comparison: eager, SDPA, FlashAttention, sliding/full Gemma2.
- Last-token logits versus full-sequence logits.
- Placeholder stitch cost with boolean `masked_scatter` emulation versus indexed copy.

## 14. Skip/defer list

- Training, suffix loss generation, gradient checkpointing.
- Beam search and advanced generation processors.
- Quantization-specific kernels.
- Tensor parallel and pipeline parallel plans.
- Dynamic/non-square image sizes and SigLIP positional interpolation.
- Remote-code-only or nonstandard checkpoints not accepted by in-library `PaliGemmaConfig`.
- Full detection/segmentation postprocessing for `<loc*>` and `<seg*>` tokens; those are downstream decoding conventions, not core model graph work.
- Multi-GPU offload and cache paging.

## 15. Final implementation checklist

- [ ] Parse `PaliGemmaConfig` and nested `vision_config` / `text_config`.
- [ ] Route `text_config.model_type` to Gemma or Gemma2 lowering.
- [ ] Load and alias tied token embedding / LM head weights.
- [ ] Implement fixed-size SigLIP image preprocessing contract or import CPU pipeline outputs.
- [ ] Lower SigLIP patch embedding, encoder layers, and post LayerNorm.
- [ ] Lower multimodal projector `Linear(1152 -> text_hidden)`.
- [ ] Implement image-token OOV replacement before embedding lookup.
- [ ] Implement processor-compatible placeholder expansion in integration tests.
- [ ] Lower image embedding stitch as guarded indexed copy.
- [ ] Implement PaliGemma 1-indexed `position_ids`.
- [ ] Implement Gemma MQA attention with RoPE and KV cache.
- [ ] Implement Gemma2 GQA attention with softcap and full/sliding masks.
- [ ] Implement Gemma/Gemma2 RMSNorm and gated MLP.
- [ ] Implement `logits_to_keep` and last-token LM head optimization.
- [ ] Add processor, vision tower, stitch, prefill, and decode parity tests.
- [ ] Benchmark processor, vision, prefill, decode, cache memory, and attention backends.
