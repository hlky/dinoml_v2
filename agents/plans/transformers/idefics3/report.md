# Idefics3 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: HuggingFaceM4/Idefics3-8B-Llama3, plus tiny/debug, fine-tune, and quantized config variants
Config source: Hugging Face Hub raw config/preprocessor/tokenizer files, May 13 2026
Source files inspected:
  X:/H/transformers/src/transformers/models/idefics3/modeling_idefics3.py
  X:/H/transformers/src/transformers/models/idefics3/configuration_idefics3.py
  X:/H/transformers/src/transformers/models/idefics3/processing_idefics3.py
  X:/H/transformers/src/transformers/models/idefics3/image_processing_idefics3.py
  X:/H/transformers/src/transformers/models/idefics3/image_processing_pil_idefics3.py
  X:/H/transformers/src/transformers/models/llama/modeling_llama.py
  X:/H/transformers/src/transformers/models/llama/configuration_llama.py
  X:/H/transformers/src/transformers/models/idefics2/modeling_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/configuration_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/processing_idefics2.py
Any missing files or assumptions:
  Native source is authoritative for this report; no modular_idefics3.py exists in this checkout.
  The production config carries legacy/remote-code fields such as text_config.perceiver_config and text_config.use_resampler, but the inspected native Idefics3 source does not read them.
  processor_config.json is absent for HuggingFaceM4/Idefics3-8B-Llama3; processor behavior comes from processing_idefics3.py defaults plus tokenizer/preprocessor config.
```

Primary source URLs:

- Transformers native source at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics3`
- Nested Llama source at the same commit:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/llama`
- Idefics2 comparison source at the same commit:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics2`
- Hub config roots:
  `https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3`,
  `https://huggingface.co/trl-internal-testing/tiny-Idefics3ForConditionalGeneration`,
  `https://huggingface.co/optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration`,
  `https://huggingface.co/Mantis-VL/mantis-8b-idefics3_16384`,
  `https://huggingface.co/ronantakizawa/idefics3-8b-llama3-awq`,
  `https://huggingface.co/qgallouedec/idefics3-8B-tiny`

Representative configs inspected:

| Model id | Scope | Notes |
|---|---|---|
| `HuggingFaceM4/Idefics3-8B-Llama3` | common production | bf16, Llama3-style text config, SigLIP SO400M-like vision config |
| `trl-internal-testing/tiny-Idefics3ForConditionalGeneration` | small/debug | reduced hidden sizes/layers but keeps Idefics3 structure |
| `optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration` | tiny/random | ONNX/testing shape edge case with `hidden_size=16`, `head_dim=128` in text config |
| `Mantis-VL/mantis-8b-idefics3_16384` | fine-tune variant | same main dimensions as production; config adds top-level `hidden_size` |
| `ronantakizawa/idefics3-8b-llama3-awq` | quantized variant | compressed-tensors 4-bit Linear weights; vision and connector are explicitly ignored |
| `qgallouedec/idefics3-8B-tiny` | tiny variant | tiny text/vision dimensions with production-like token ids |

Primary runtime target: multimodal conditional generation, with image feature extraction, multimodal prefill, and cached autoregressive decode.

## 2. High-level architecture

Idefics3 is a SigLIP-like variable-resolution vision encoder plus a Llama-family causal decoder. Images are encoded into patch tokens, spatially downsampled by a connector pixel shuffle, projected into the Llama hidden size, and stitched into text token embeddings at `<image>` placeholder positions.

```text
CPU image/text preprocessing
  -> NCHW pixel batch + pixel masks + expanded image placeholder tokens
  -> variable-resolution vision encoder
  -> pixel-shuffle connector + Linear projection
  -> masked_scatter into text embeddings
  -> Llama causal LM prefill
  -> cached decode
  -> lm_head logits / generation controller
```

Stage decomposition:

| Stage | Runtime contract | Independently cacheable? |
|---|---|---|
| Processor | Builds prompt strings with row/column image tags, resizes/splits/pads/normalizes images, emits `input_ids`, `attention_mask`, `pixel_values`, `pixel_attention_mask` | CPU/data pipeline; processed tensors can be cached per request |
| Vision encoder | Consumes only nonzero real image slots after dropping padded images; outputs `[real_images, patch_seq, vision_hidden]` | Yes, per processed image batch |
| Connector | Pixel-shuffles square patch grids by `scale_factor`, then applies a bias-free Linear projection to text hidden size | Yes, output is accepted as `image_hidden_states` |
| Embedding stitch | Replaces every `<image>` token embedding with one projected image feature row using `masked_scatter` | Validate independently; exact count/order matters |
| Text prefill | Llama decoder over mixed text/image embedding sequence | Standard AR KV cache |
| Decode | Llama one-token or short-step decode; source drops `pixel_values` after the first generation iteration | Image encoder/connector should not be recomputed |

The main stageable boundary is `image_hidden_states`: Transformers lets callers pass it instead of `pixel_values`, and generation preparation clears pixels when cache is active after the first iteration.

## 3. Important config dimensions

Production values below are from `HuggingFaceM4/Idefics3-8B-Llama3/config.json` unless marked as source default or inference from source.

| Field | Production value | Provenance |
|---|---:|---|
| model type | `idefics3` | config.json |
| text model | `llama` via `AutoModel.from_config` | config.json/source |
| text hidden size | 4096 | config.json |
| text layers | 32 | config.json |
| text attention heads / KV heads | 32 / 8 | config.json |
| text head dim | 128 | inferred from hidden/heads; explicit in newer variants |
| text MLP intermediate | 14336 | config.json |
| text activation | `silu` | config.json |
| text attention/mlp bias | false / false | config.json |
| vocab size | 128259 | config.json |
| max positions | 131072 | config.json |
| RoPE | Llama3 scaling, factor 8, original max 8192, theta 500000 | config.json |
| cache support | `use_cache=True`, `DynamicCache` when needed | config/source |
| top-level dtype | `bfloat16` | config.json |
| image token id | 128257 | config.json/tokenizer |
| fake/image/eou token ids | 128256 / 128257 / 128258 | tokenizer_config.json |
| pad token id | 128002 | config/tokenizer |
| EOS token ids | `[128001, 128008, 128009]` | config/tokenizer |
| tokenizer padding/truncation side | left / left | tokenizer_config.json |
| vision hidden size | 1152 | config.json |
| vision layers | 27 | config.json |
| vision heads / head dim | 16 / 72 | config.json/inference |
| vision MLP intermediate | 4304 | config.json |
| vision activation | `gelu_pytorch_tanh` | config.json |
| vision image size / patch size | 364 / 14 | config.json |
| vision position grid | `26 x 26`, 676 learned positions | config/source |
| connector scale factor | 2 | config.json |
| image seq len per processed image | `((364 // 14) ** 2) / 4 = 169` | config/source |
| connector projection | `Linear(1152 * 4 -> 4096, bias=False)` | source/config |
| image processor size | longest edge 1456 | preprocessor_config.json |
| image split max size | longest edge 364 | source default; not serialized in production preprocessor config |
| image mean/std | `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]` | preprocessor_config.json |

Representative checkpoint sweep:

| Model id | Text hidden/layers/heads/KV | Vision hidden/layers/patch | Image seq len | dtype | Operator-significant variation |
|---|---|---|---:|---|---|
| `HuggingFaceM4/Idefics3-8B-Llama3` | 4096 / 32 / 32 / 8 | 1152 / 27 / 14 | 169 | bf16 | production target |
| `Mantis-VL/mantis-8b-idefics3_16384` | 4096 / 32 / 32 / 8 | 1152 / 27 / 14 | 169 | bf16 | fine-tune; same native graph |
| `ronantakizawa/idefics3-8b-llama3-awq` | 4096 / 32 / 32 / 8 | 1152 / 27 / 14 | 169 | bf16 | compressed-tensors 4-bit Linear; ignores vision/connector/lm_head |
| `trl-internal-testing/tiny-Idefics3ForConditionalGeneration` | 64 / 8 / 4 / 1 | 64 / 4 / 14 | 169 | float32 top-level | debug-sized native graph |
| `optimum-intel-internal-testing/tiny-random-Idefics3ForConditionalGeneration` | 16 / 2 / 4 / 2, `head_dim=128` | 16 / 2 / 14 | 169 | bf16 | config has `hidden_size != heads * head_dim`; source Llama projections use explicit `head_dim` |
| `qgallouedec/idefics3-8B-tiny` | 16 / 2 / 4 / 2, `head_dim=128` | 16 / 2 / 14 | 169 | bf16 | tiny shape stress case |

## 3a. Family variation traps

- Idefics3 is not Idefics2 with renamed files. Native Idefics3 removes the Idefics2 Perceiver resampler and instead uses `Idefics3Connector.pixel_shuffle` followed by a single bias-free projection.
- The production config still contains `text_config.perceiver_config` and `text_config.use_resampler=false`. The inspected native Idefics3 model never reads either field. DinoML should not implement Perceiver behavior for native Idefics3 based on those stale fields.
- `image_seq_len` is processor-coupled and must match `int(((image_size // patch_size) ** 2) / scale_factor**2)`. For production this is 169, not Idefics2's 64.
- Processor image splitting changes one logical image into multiple processed images plus a global image. Placeholder text includes row/column tags for split tiles plus a global image block.
- `pixel_shuffle` assumes the vision patch sequence length is a perfect square and that height equals width after preprocessing/splitting. It reconstructs `height = width = int(seq**0.5)` and has no explicit runtime guard for nonsquare patch grids.
- Tiny configs may set `hidden_size != num_attention_heads * head_dim`. The current Llama source uses explicit `head_dim` for projection sizes, so DinoML importers must not assume output Q width equals hidden size.
- Text is GQA when `num_key_value_heads < num_attention_heads`; production has 8 KV heads vs 32 Q heads.
- Vision attention is full MHA with learned absolute/bucketized positions, not RoPE.
- Source tensors for pixels are NCHW inside `[batch, max_images, channels, height, width]`. Layout optimization must guard `Conv2d`, mask `unfold`, `flatten(2).transpose(1,2)`, and connector reshape/permutation order.
- The tokenizer uses left padding and left truncation. That affects causal positions and cache parity for batched generation.
- `prepare_inputs_for_generation` clears `pixel_values` and `pixel_attention_mask` when `image_hidden_states` is supplied or when cached decode is past the first iteration.
- `inputs_merger` with `input_ids=None` compares `inputs_embeds` to the learned image-token embedding. First integration should require `input_ids` and defer that embedding-equality path.
- AWQ/compressed-tensors configs quantize generic `Linear` targets but explicitly ignore all vision encoder linears, the connector projection, and `lm_head`. Dense native graph parity should land first.

## 4. Operator coverage checklist

Tensor/layout ops:

- Text embedding lookup and final LM-head GEMM.
- NCHW `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` patch embedding.
- `flatten`, `transpose`, `view`, `reshape`, `permute`, `contiguous`, `expand`, boolean indexing, indexed assignment, and `masked_scatter`.
- Pixel mask `unfold` over height and width with `size=stride=patch_size`, sum over patch windows, compare `> 0`.
- Drop padded image slots by detecting images whose all values are zero.
- Connector pixel shuffle over `[B, seq, C]` with `seq = h * w`, `scale_factor=2`.
- Processor-derived sequence/count validation for split images and placeholder runs.

Neural network primitives:

- LayerNorm for vision blocks and post-vision norm.
- RMSNorm for nested Llama decoder.
- Dense Linear/GEMM with and without bias.
- Vision MLP: `Linear(1152 -> 4304, bias=True)` + `gelu_pytorch_tanh` + `Linear(4304 -> 1152, bias=True)`.
- Connector projection: `Linear(4608 -> 4096, bias=False)`.
- Text Llama MLP: SwiGLU `gate_proj`, `up_proj`, `down_proj`, production `4096 -> 14336 -> 4096`, bias false.
- LM head: `Linear(4096 -> 128259, bias=False)`.

Attention primitives:

- Vision bidirectional MHA, production 16 heads x 72 head dim.
- Text causal GQA, production 32 Q heads, 8 KV heads, 128 head dim.
- Attention masks from `create_bidirectional_mask` for vision and `create_causal_mask` for Llama.
- Eager attention math uses additive masks, scaling by `head_dim**-0.5`, fp32 softmax, dtype downcast, dropout, and value matmul.
- Native source advertises eager, SDPA, FlashAttention, and FlexAttention backend dispatch.

Position/rotary/cache ops:

- Vision learned position embedding indexed by fractional-coordinate bucketization.
- Llama3 RoPE with `rope_theta=500000`, factor 8, low/high frequency factors, original max 8192.
- `DynamicCache` for text KV cache; keys are cached after RoPE and before repeat-KV expansion.
- `logits_to_keep` slicing for last-token-only or selected-token logits.

Preprocessing-coupled ops:

- Image resize/split/pad/normalize in the processor.
- Prompt rewrite with `<fake_token_around_image>`, `<row_i_col_j>`, `<global-img>`, and repeated `<image>` tokens.
- Optional `mm_token_type_ids` generation from fake-token spans; this is processor metadata, not consumed by `Idefics3Model.forward`.

Parameter sharing:

- Top-level config has `tie_word_embeddings=false`, and `lm_head` is constructed separately from token embeddings.
- The class still declares `_tied_weights_keys = {"lm_head.weight": "model.text_model.embed_tokens.weight"}` for optional tying machinery. DinoML should preserve the config-driven untied default and only alias if the loaded checkpoint/state dict proves tied storage.

## 5. Layer/block breakdown

Vision patch embedding:

```text
pixel_values: [R, 3, H, W] NCHW, where R is nonzero processed image slots
patch_embeds = Conv2d(3, 1152, kernel=stride=14)(pixel_values)  # [R, 1152, H/14, W/14]
x = patch_embeds.flatten(2).transpose(1, 2)                     # [R, P, 1152]
position_ids = bucketize(valid patch fractional h/w into 26 x 26 learned grid)
x = x + position_embedding(position_ids)
```

Vision encoder block, repeated 27 times in production:

```text
residual = x
x = LayerNorm(eps=1e-6)(x)
q,k,v = Linear(1152 -> 1152, bias=True)(x)
x = bidirectional MHA(q,k,v, patch mask)
x = residual + Linear(1152 -> 1152, bias=True)(x)
residual = x
x = LayerNorm(eps=1e-6)(x)
x = Linear(1152 -> 4304, bias=True) -> gelu_pytorch_tanh -> Linear(4304 -> 1152, bias=True)
x = residual + x
```

Vision output and connector:

```text
x = LayerNorm(eps=1e-6)(vision_last_hidden_state)
x = pixel_shuffle(x, scale_factor=2)
  # [R, 26*26, 1152] -> [R, 13*13, 1152*4]
image_features = Linear(4608 -> 4096, bias=False)(x)
```

Text/image stitch:

```text
inputs_embeds = embed_tokens(input_ids)
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, image_features.reshape(-1, 4096))
```

Llama decoder block, repeated 32 times in production:

```text
residual = x
x = RMSNorm(eps=1e-5)(x)
q = Linear(4096 -> 32 * 128, bias=False)(x)
k = Linear(4096 -> 8 * 128, bias=False)(x)
v = Linear(4096 -> 8 * 128, bias=False)(x)
q,k = apply_rope(q,k, cos, sin)
k,v = DynamicCache.update(k,v, layer_idx) if cache is active
x = causal GQA(q,k,v, attention_mask)
x = residual + Linear(4096 -> 4096, bias=False)(x)
residual = x
x = RMSNorm(eps=1e-5)(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

LM head:

```text
logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch tokens.
- Production MHA: 16 heads, hidden 1152, head dim 72.
- Mask is a bidirectional additive/backend-specific mask created from `patch_attention_mask.view(R, -1)`.
- No KV cache. Cached image features are a connector output boundary, not an attention KV cache.

Text attention:

- Causal decoder self-attention through native Llama source.
- Production GQA: Q heads 32, KV heads 8, head dim 128, repeat factor 4 for eager math.
- RoPE is applied to Q/K before cache update.
- Cache tensor shapes per layer before repeat expansion are effectively:
  - K: `[batch, 8, cached_seq, 128]`
  - V: `[batch, 8, cached_seq, 128]`
  Attention backends may materialize or logically repeat to `[batch, 32, cached_seq, 128]`.
- The source Llama path computes `position_ids` from `past_key_values.get_seq_length()` when not supplied.
- Eager order is Q/K/V projection -> RoPE -> cache update -> repeat KV -> `q @ k.T * scale` -> additive mask -> fp32 softmax -> dropout -> `attn @ v` -> output projection.
- FlashAttention/SDPA/FlexAttention compatibility flows through `ALL_ATTENTION_FUNCTIONS`; DinoML should first match eager math, then lower to fused GQA attention under backend guards.

Generation/cache semantics:

- `Idefics3Model.forward` creates a `DynamicCache(config=self.config)` when `use_cache` is true and no cache is passed, then forwards it into the text model. The nested Llama source also creates a `DynamicCache` if it receives `use_cache` and no cache.
- `prepare_inputs_for_generation` passes multimodal inputs through the parent generation helper, then clears pixels whenever `image_hidden_states` is already supplied or when `use_cache` is true and `is_first_iteration` is false.
- During decode, generated `<image>` tokens should not trigger image feature insertion unless actual image features are supplied; the source comment explicitly avoids replacing generated image-token ids with nonexistent images.

## 7. Position encoding and custom math

Vision variable-resolution position IDs use learned absolute buckets. Production `image_size=364` and `patch_size=14`, so the trained square grid is 26 by 26.

```python
def idefics3_patch_position_ids(patch_attention_mask, num_patches_per_side=26):
    nb_h = patch_attention_mask[:, :, 0].sum(1)
    nb_w = patch_attention_mask[:, 0, :].sum(1)
    step_h = 1.0 / nb_h
    step_w = 1.0 / nb_w
    h = arange(max_patches_h) * step_h[:, None]
    w = arange(max_patches_w) * step_w[:, None]
    h = clamp(h, max=1.0 - 1e-6)
    w = clamp(w, max=1.0 - 1e-6)
    bucket_h = bucketize(h, boundaries=arange(1/26, 1.0, 1/26), right=True)
    bucket_w = bucketize(w, boundaries=arange(1/26, 1.0, 1/26), right=True)
    return bucket_h[:, :, None] * 26 + bucket_w[:, None, :]
```

Connector pixel shuffle:

```python
def idefics3_pixel_shuffle(x, scale_factor=2):
    bsz, seq, embed_dim = x.shape
    h = w = int(seq ** 0.5)
    x = x.view(bsz, h, w, embed_dim)
    x = x.view(bsz, h, w // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, w // scale_factor, h // scale_factor, embed_dim * scale_factor**2)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(bsz, seq // scale_factor**2, embed_dim * scale_factor**2)
```

Llama RoPE:

```python
def apply_llama_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RoPE cos/sin can be precomputed for fixed ranges, but position IDs depend on cache length and left-padding behavior during generation.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Text may contain `<image>` markers. Processor counts markers and requires the number of markers to match the number of supplied logical images.
- For each logical image, image preprocessing may split it into `rows * cols` tiles plus a global image. The prompt replacement includes:
  - for split tiles: `<fake_token_around_image><row_i_col_j>` plus `image_seq_len` repeated `<image>` tokens for each tile, row-separated by newlines;
  - then a global block: `<fake_token_around_image><global-img>` plus `image_seq_len` repeated `<image>` tokens plus a closing fake token.
- For nonsplit images, the replacement is `<fake_token_around_image><global-img><image> * image_seq_len <fake_token_around_image>`.
- Production tokenizer special ids: fake image token 128256, image token 128257, end-of-utterance 128258. Row/column tags are converted from tokenizer vocab for 6 by 6 possible row/col labels.
- Image processor defaults in source: resize longest edge `4 * 364 = 1456`, split max `364`, convert RGB, rescale, normalize, pad, image splitting enabled.
- Production preprocessor config overrides mean/std to `[0.5, 0.5, 0.5]`; source defaults are ImageNet standard mean/std if a config omits them.
- Runtime tensors after processor with padding: `pixel_values [B, max_num_processed_images, 3, Hmax, Wmax]`, `pixel_attention_mask [B, max_num_processed_images, Hmax, Wmax]`.

GPU/runtime:

- Model flattens image slots to `[B * max_images, 3, H, W]`, drops all-zero slots, and applies the same filter to `pixel_attention_mask`.
- Patch mask is produced by unfolding pixel masks over patch-size windows and marking a patch valid if the summed mask is greater than zero. This differs from an all-valid requirement and should be matched.
- Vision encoder returns `[real_processed_images, 676, 1152]` for production 364 x 364 processed images.
- Connector returns `[real_processed_images, 169, 4096]`; flattened row count must equal the count of `<image>` token ids in `input_ids`.
- `masked_scatter` uses row-major order over the expanded boolean mask, so image feature row order must match processor split/global ordering exactly.
- `image_hidden_states` can be supplied directly to skip pixel processing in the model. DinoML should expose this as an explicit precomputed image-feature input for decode-heavy integration.

Generation-controller behavior:

- Chat template is tokenizer/processor-owned and outside the core GPU graph.
- EOS is `[128001, 128008, 128009]` for production.
- Generation should avoid recomputing pixels after the first prefill; this is native source behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Source pattern:

```text
Conv2d(C=3, out=1152, kernel=14, stride=14, padding=valid)
flatten(2).transpose(1, 2)
```

Replacement:

```text
PatchExtract/WindowFlatten [R, P, 3*14*14] -> GEMM(weight.T) -> BiasAdd -> [R, P, 1152]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, dilation 1, groups 1.
- H and W are divisible by 14 or the lowering exactly preserves Conv2d floor output dimensions.
- Source NCHW flatten order is preserved.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
y = windows @ w.T + conv.bias
```

Failure cases: NHWC-translated tensors without matching window order, dynamic nonsquare split tiles, grouped/dilated future configs.

Parity test: compare patch embeddings before positional add for padded rectangular images and split/global images.

### Rewrite: connector pixel shuffle -> reshape/permute schedule

Source pattern: `Idefics3Connector.pixel_shuffle` over `[R, seq, vision_hidden]`.

Replacement: explicit view/permute/reshape schedule or a fused kernel feeding the connector projection.

Preconditions:

- `seq == (image_size // patch_size) ** 2`.
- `sqrt(seq)` is an integer.
- `sqrt(seq) % scale_factor == 0`.
- Source row-major patch order from `flatten(2).transpose(1,2)` is preserved.

Shape equation:

```text
[R, h*w, C] -> [R, (h/scale)*(w/scale), C*scale^2]
production: [R, 676, 1152] -> [R, 169, 4608]
```

Failure cases: variable rectangular patch grids, disabled processor resizing/splitting that produces nonsquare grids, or future scale factors that do not divide the patch grid.

### Rewrite: image feature precompute

Source pattern:

```text
pixel_values -> vision_model -> connector -> image_hidden_states -> masked_scatter
```

Replacement: compile vision+connector as an independently callable subgraph and let the text prefill graph accept `image_hidden_states`.

Preconditions:

- Same processor settings, dtype, and weights.
- Prompt placeholders exactly match feature row count.
- No training/dropout.

Parity test: one forward with `pixel_values`, one with `image_hidden_states=get_image_features(...).pooler_output`, compare logits.

### Rewrite: masked_scatter image stitch -> indexed copy

Source pattern: `inputs_embeds.masked_scatter(input_ids == image_token_id, image_hidden_states)`.

Replacement: compute image-token positions and perform indexed row copy into an embedding buffer.

Preconditions:

- `input_ids` is present.
- `sum(input_ids == image_token_id) == image_hidden_states.numel() / hidden_size`.
- Feature rows are already flattened in processor/model order.

Failure cases: `inputs_embeds`-only path, generated image tokens during decode, mismatched split/global prompt expansion.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` with `logits_to_keep`.

Replacement: run vocab GEMM only for final or selected token positions.

Preconditions: no loss computation; `logits_to_keep` is an integer final-token count or a supported index tensor.

Parity test: compare selected rows from full logits with optimized logits.

### Layout optimization candidate: guarded NCHW vision island

Initial lowering should preserve PyTorch source axes. A later NHWC/channel-last pass must rewrite or guard:

- `pixel_values [B, I, C, H, W]`.
- Conv weight layout `[out, in, kh, kw]`.
- Pixel mask axes `[R, H, W]` and `unfold(dimension=1/2)`.
- `patch_embeds.flatten(2).transpose(1,2)` token order.
- Pixel shuffle reshape/permutation order over `[seq, hidden]`.

The processor boundary, patch-mask construction, and embedding stitch should be protected by no-layout-translation guards until a local vision-layout pass proves exact token ordering.

## 10. Kernel fusion candidates

Highest priority:

- Llama RMSNorm + QKV projection preparation, because text prefill/decode dominates once image features are cached.
- Llama3 GQA FlashAttention with KV cache, including bf16 and `num_key_value_heads < num_attention_heads`.
- Llama SwiGLU MLP fusion for `silu(gate) * up`.
- Placeholder indexed-copy stitch, because `masked_scatter` is dynamic and easy to replace under exact count guards.
- Connector pixel shuffle + projection, because it is Idefics3-specific and must be correct before multimodal prefill.

Medium priority:

- Patch Conv2d -> GEMM for non-overlap NCHW patches.
- Vision LayerNorm + MHA/MLP block fusions for image-heavy batch throughput.
- Vision position bucketization helper, especially `bucketize` and indexed assignment.
- Last-token-only logits for decode.

Lower priority:

- GPU processor resize/split/normalize. Keep CPU/data-pipeline first.
- AWQ/compressed-tensors lowering. The dense graph and DinoML GGUF/encoded-constant path should be validated first.
- `mm_token_type_ids`; processor can emit it but native Idefics3 forward does not consume it.

## 11. Runtime staging plan

1. Parse Idefics3 config and nested Llama config; reject native-ignored Perceiver/use_resampler fields as non-operative for this family.
2. Load text-only Llama submodule and verify pure text prefill/decode with the production tokenizer ids.
3. Implement the prompt placeholder count validator outside the compiled graph.
4. Implement connector-only parity from random/recorded vision hidden states through pixel shuffle and projection.
5. Implement vision embedding parity: NCHW patch Conv2d, patch mask, and position bucket IDs.
6. Implement the full vision encoder and post-layernorm.
7. Expose `image_hidden_states` as a precomputed image-feature input to the text prefill graph.
8. Implement full multimodal prefill with `pixel_values` after vision/connector parity is stable.
9. Implement cached decode, verifying pixels are not recomputed after first iteration.
10. Add optimized attention, connector, patch-embed, and last-token logits rewrites under guards.
11. Add quantized/AWQ variants only after dense bf16 parity is routine.

Initially stub or externalize: chat template rendering, image URL loading, torchvision/PIL preprocessing, training loss, beam search, compressed-tensors quantized kernels, and arbitrary `inputs_embeds`-only multimodal stitch.

## 12. Parity and validation plan

- Config test: production and tiny configs parse nested text/vision dimensions, including explicit `head_dim` tiny cases.
- Processor contract test: count image tokens for nonsplit and split images; verify row/column/global prompt ordering.
- Image preprocessing metadata test: `get_number_of_image_patches` and prompt token counts for multiple aspect ratios.
- Patch mask test: padded image masks produce the same `patch_attention_mask` as Transformers (`sum > 0` per patch).
- Vision embedding test: Conv2d + position IDs for a random image/mask.
- Pixel shuffle test: compare connector reshapes for `[R, 676, C] -> [R, 169, 4C]`.
- Connector test: pixel shuffle + Linear projection for production and tiny hidden sizes.
- Stitch test: indexed-copy replacement equals `masked_scatter` for multiple images per batch.
- Text test: pure Llama prefill and two cached decode steps; verify KV shapes `[B, 8, S, 128]` for production.
- Multimodal prefill test: compare logits with `pixel_values` vs precomputed `image_hidden_states`.
- End-to-end smoke: one image QA prompt, greedy generation, compare generated token ids over a short fixed decode.

Suggested tolerances: fp32 unit tests `atol=1e-4, rtol=1e-4`; bf16/fp16 block tests `atol=2e-2, rtol=2e-2`; full-model logits may need `atol=3e-2, rtol=3e-2` when attention/GEMM backends differ but should use tighter tolerances for unfused reference paths.

## 13. Performance probes

- CPU processor throughput: images/sec for resize/split/pad/normalize and prompt rewriting.
- Vision encoder throughput by number of processed images and image resolution.
- Connector throughput for `[R, 676, 1152] -> [R, 169, 4096]`.
- Multimodal prefill tokens/sec with 0, 1, 5, and 10 processed images.
- Decode tokens/sec after cached multimodal prefill.
- KV cache memory: `32 layers * 2 tensors * B * 8 KV heads * seq * 128 * dtype_size`.
- Attention backend comparison: eager vs SDPA/Flash/Flex-compatible GQA.
- Last-token-only logits speedup over full-sequence logits.
- AWQ/compressed-tensors GEMM probe after dense parity.

Any benchmark numbers should be recorded as DinoML measurements, not inferred from source.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Beam search and complex generation policies beyond greedy/sampling parity.
- GPU implementation of PIL/torchvision preprocessing.
- `inputs_embeds`-only image-token embedding comparison path.
- `mm_token_type_ids`, because native forward does not consume it.
- Legacy Idefics2 Perceiver behavior and config-advertised `perceiver_config` fields in Idefics3 text config.
- AWQ/compressed-tensors import/lowering for first dense graph.
- Multi-GPU tensor parallel/pipeline plans.
- General NHWC/channel-last vision translation until exact patch order, mask axes, and pixel shuffle guards are proven.

## 15. Final implementation checklist

- [ ] Parse Idefics3 config and nested Llama/Vision configs.
- [ ] Reject or ignore native-unused Perceiver/use_resampler fields with an audit note.
- [ ] Load text, vision, connector, and LM-head weights with config-driven untied embedding policy.
- [ ] Implement Llama decoder ops: RMSNorm, Llama3 RoPE, GQA, causal mask, KV cache, SwiGLU.
- [ ] Implement `logits_to_keep` final-token LM-head slicing.
- [ ] Implement NCHW patch Conv2d or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement patch-mask unfolding and vision position bucketization.
- [ ] Implement bidirectional vision attention and LayerNorm MLP blocks.
- [ ] Implement connector pixel shuffle and `Linear(vision_hidden * scale_factor^2 -> text_hidden)`.
- [ ] Implement image placeholder count validation and indexed embedding stitch.
- [ ] Support `image_hidden_states` as a precomputed image-feature input.
- [ ] Add processor placeholder/splitting contract tests.
- [ ] Add vision embedding, vision encoder, connector, stitch, multimodal prefill, and cached decode parity tests.
- [ ] Benchmark processor, vision, connector, prefill, decode, logits, and KV memory separately.
