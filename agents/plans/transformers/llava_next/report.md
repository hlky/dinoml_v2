# LLaVA-NeXT Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `transformers`.

Model id: family `llava_next`; representative checkpoints inspected from Hugging Face:

- `llava-hf/llava-v1.6-mistral-7b-hf`
- `llava-hf/llava-v1.6-vicuna-7b-hf`
- `llava-hf/llava-v1.6-vicuna-13b-hf`
- `llava-hf/llava-v1.6-34b-hf`
- `optimum-internal-testing/tiny-random-llava-next-mistral`

Config source: each checkpoint `config.json`; LLaVA-NeXT v1.6 `preprocessor_config.json`; `llava-v1.6-mistral-7b-hf` `processor_config.json`. Hub config URLs used include:

- `https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/llava-v1.6-vicuna-13b-hf/raw/main/config.json`
- `https://huggingface.co/llava-hf/llava-v1.6-34b-hf/raw/main/config.json`
- `https://huggingface.co/optimum-internal-testing/tiny-random-llava-next-mistral/raw/main/config.json`

Source files inspected:

- `src/transformers/models/llava_next/configuration_llava_next.py`
- `src/transformers/models/llava_next/modeling_llava_next.py`
- `src/transformers/models/llava_next/processing_llava_next.py`
- `src/transformers/models/llava_next/image_processing_llava_next.py`
- `src/transformers/models/llava_next/image_processing_pil_llava_next.py`
- Delegated source for representative towers/decoders: `clip/modeling_clip.py`, `llama/modeling_llama.py`, `mistral/modeling_mistral.py`, and their configuration files.

Any missing files or assumptions: no remote-code files were required for the in-library `llava_next` class. The modeling file delegates the vision tower and language model through `AutoModel.from_config`, so decoder details are owned by the selected text family. This report targets inference-time multimodal image+text generation.

## 2. High-level architecture

LLaVA-NeXT is a vision encoder plus multimodal projector plus causal text decoder. Compared with base LLaVA, the wrapper adds AnyRes image patch packing, spatial unpadding, learned image-newline tokens, and variable-length placeholder expansion.

Dataflow:

```text
image/text processor -> AnyRes pixel patches + expanded input_ids
pixel patches -> vision tower hidden states -> feature selection -> projector
projected patch features -> spatial pack/unpad/newline -> image token sequence
input_ids -> token embeddings
image placeholder mask -> masked_scatter(image features into text embeddings)
stitched inputs_embeds -> decoder prefill -> logits / KV cache
cached decode -> text decoder only -> logits / sampling
```

Stage decomposition:

- CPU/data-pipeline: RGB conversion, best-resolution selection, aspect-preserving resize, padding to the selected grid, division into 336x336 patches, base-image resize/crop, rescale/normalize, tokenization, chat template, and expansion of each `<image>` marker into a variable run of image tokens.
- Cacheable vision prefix: `pixel_values + image_sizes -> vision_tower -> selected hidden states -> projector -> pack_image_features`. This can be cached per image and feature-selection policy.
- Prefix construction: projected image features are copied into the token embedding stream at repeated image-token positions. This is a small but strict indexed-stitch stage.
- Prefill: text decoder consumes one multimodal embedding sequence and creates the autoregressive KV cache.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values` and `image_sizes` only on the first generation iteration, or when cache is disabled. Subsequent cached decode is text-only.

The independently testable boundaries are image preprocessing, vision/projector, AnyRes packing, placeholder stitch, decoder prefill, and cached decode.

## 3. Important config dimensions

Source defaults from `LlavaNextConfig`: `image_token_index=32000`, `image_seq_length=576`, `projector_hidden_act="gelu"`, `vision_feature_select_strategy="default"`, `vision_feature_layer=-2`, `multimodal_projector_bias=True`, `tie_word_embeddings=False`, and default `image_grid_pinpoints=[[336,672],[672,336],[672,672],[1008,336],[336,1008]]`. If omitted, `vision_config` defaults to CLIP vision hidden 1024, 24 layers, 16 heads, patch 14, image 336. If omitted, `text_config` defaults to Llama.

| Checkpoint | Text model | Text dims | KV heads | Context / RoPE / sliding | Vision tower | Image/token settings | Dtype source |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny-random llava-next-mistral | Mistral | hidden 8, 2 layers, 4 heads, head_dim 2, MLP 32 | 2 | max pos 2048, rope theta 10000, sliding 4096 | CLIP hidden 16, 2 layers, 4 heads, patch 14, image 336 | image token 32000, image_seq_length 576, default strategy | `config.json` float32 |
| llava-v1.6-vicuna-7b | Llama/Vicuna | config omits hidden/layers/heads; effective current Llama defaults are hidden 4096, 32 layers, 32 heads, MLP 11008, head_dim 128 | effective 32 | max pos 4096, Llama default rope theta | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 32000, default strategy, AnyRes pinpoints | `config.json` float16 |
| llava-v1.6-vicuna-13b | Llama/Vicuna | hidden 5120, 40 layers, 40 heads, MLP 13824, head_dim 128 | 40 | max pos 4096 | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 32000, default strategy, AnyRes pinpoints | `config.json` float16 |
| llava-v1.6-mistral-7b | Mistral | config omits hidden/layers/heads; effective Mistral defaults are hidden 4096, 32 layers, 32 heads, MLP 14336, head_dim 128 | 8 | max pos 32768, rope theta 1000000, `sliding_window=null` | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 32000, default strategy, AnyRes pinpoints | top `torch_dtype` float16; fetched raw config currently includes text `torch_dtype` bfloat16 |
| llava-v1.6-34b | Llama/Yi-style | hidden 7168, 60 layers, 56 heads, MLP 20480, head_dim 128 | 8 | max pos 4096, rope theta 5000000, text `use_cache=false` in config | CLIP hidden 1024, 24 layers, 16 heads, patch 14, image 336 | image token 64000, vocab 64064, default strategy, AnyRes pinpoints | top `torch_dtype` float16; text config bfloat16 |

Preprocessor settings from v1.6 checkpoints: `image_processor_type="LlavaNextImageProcessor"`, `size.shortest_edge=336`, `crop_size=336x336`, BICUBIC resample (`3`), RGB conversion, rescale by `1/255`, OpenAI CLIP mean/std, AnyRes `image_grid_pinpoints`, and usually `do_pad=true` except the fetched 34B preprocessor omitted `do_pad` and therefore relies on source/default or caller behavior. `processor_config.json` for Mistral sets `image_token="<image>"`, `num_additional_image_tokens=1`, `patch_size=14`, and `vision_feature_select_strategy="default"`.

## 3a. Family variation traps

- Text model is delegated. LLaVA-NeXT v1.6 includes Llama/Vicuna, Mistral, and Yi-style Llama configs; operator coverage must follow the delegated decoder, not a single hardcoded LLaMA shape.
- Several configs omit text dimensions and rely on current source defaults. Effective hidden/layer/head values should be resolved by instantiating the delegated config class, not by reading only the raw checkpoint JSON.
- `num_key_value_heads` can be smaller than `num_attention_heads`: Mistral 7B uses 8 KV heads for 32 attention heads; 34B uses 8 KV heads for 56 attention heads.
- The 34B config sets `image_token_index=64000` and `vocab_size=64064`; most 7B/13B configs use image token 32000 and vocab 32064.
- LLaVA-NeXT placeholder count is image-size dependent. The source `image_seq_length=576` is a nominal base grid value, not the final token count for AnyRes images.
- Processor/model coupling is strict: `patch_size`, `num_additional_image_tokens`, and `vision_feature_select_strategy` must match the model's feature selection. A mismatch trips the model's placeholder count check.
- `vision_feature_select_strategy="default"` removes token 0 from each selected vision hidden state before the projector. For CLIP this removes CLS. `"full"` keeps it.
- `vision_feature_layer` may be an int or list. A list concatenates hidden states on the feature dimension before the projector, changing `linear_1` input to `vision_hidden * len(layers)`.
- The source wrapper only supports the `spatial_unpad` merge behavior. Other historical LLaVA merge types should be rejected or routed to a separate audit.
- AnyRes patch packing is NCHW and axis-sensitive: patch division, padding, crop/resize, CLIP Conv2d, feature view/permute/flatten, and unpadding all assume channel-first tensors until the sequence becomes `[tokens, hidden]`.
- Generation cache behavior differs from ordinary text-only decode: image tensors enter only before the image features are merged and cached.
- `inputs_embeds`-only mode finds image placeholders by comparing embeddings to the image-token embedding. This is fragile for compiled runtimes and should be deferred unless explicitly needed.
- `use_image_newline_parameter` appears in checkpoint configs but the in-library source always creates and uses `self.image_newline`; do not gate newline behavior only on that historical field.

## 4. Operator coverage checklist

Tensor/layout ops:

- Image preprocessing: RGB conversion, best-resolution selection from `image_grid_pinpoints`, aspect-preserving resize, zero padding, division into fixed 336x336 patches, base-image resize, optional crop, rescale, normalize, NCHW stack, optional patch-count padding for batch.
- Vision embedding: CLIP `Conv2d(3 -> vision_hidden, kernel=stride=patch, bias=False)`, flatten spatial axes, transpose `[B, C, Gh, Gw] -> [B, Gh*Gw, C]`, concat class embedding, add learned positions.
- Feature selection: hidden-state tuple indexing, optional multi-layer concat on last dim, CLS/token-0 slice, projector, `torch.split` by per-image patch count.
- AnyRes packing: list iteration, view `[num_patch_h,num_patch_w,H,W,D]`, permute to `[D,num_patch_h,H,num_patch_w,W]`, flatten into an image-like feature map, unpad by original aspect ratio, append newline column, flatten to tokens, concat base image features.
- Placeholder stitch: embedding lookup, equality mask, `unsqueeze`, `expand_as`, count/numel check, `masked_scatter`.
- Logits: optional `logits_to_keep` slice and `Linear(text_hidden -> vocab_size, bias=False)`.

Neural network primitives:

- CLIP vision: LayerNorm, full self-attention, Linear Q/K/V/O with biases, GELU/QuickGELU MLP, residual adds.
- Projector: `Linear(vision_hidden * num_feature_layers -> text_hidden)`, GELU, `Linear(text_hidden -> text_hidden)`, bias controlled by `multimodal_projector_bias`.
- Text decoders: RMSNorm, Linear Q/K/V/O, RoPE, causal attention, GQA repeat for eager math, gated SiLU MLP, residual adds, final norm, LM head.

Attention primitives:

- Vision full noncausal MHA over base/patch image tokens.
- Text causal self-attention with compact KV cache, RoPE on Q/K before cache update, optional GQA, optional sliding-window mask only when the delegated decoder config has a non-null sliding window.
- Source wrapper advertises FlashAttention, SDPA, flex attention, and attention backend support; exact dispatch is inherited from the delegated models.

Generation/cache ops:

- DynamicCache allocation in delegated text models when `use_cache=True` and no cache is provided.
- Per-layer cache update with post-RoPE K and raw V in compact KV-head shape.
- `prepare_inputs_for_generation` image-input guard: include `pixel_values`/`image_sizes` only on first iteration or when cache is disabled.
- `logits_to_keep` for last-token-only decode logits.

Preprocessing-coupled ops:

- Placeholder expansion in `LlavaNextProcessor`.
- `image_sizes [num_images,2]` as required metadata for patch count, unpadding, and newline count.
- Optional `mm_token_type_ids` creation in the processor; the model forward does not consume it.

## 5. Layer/block breakdown

Image processor and AnyRes patch creation:

```text
image [C,H,W]
best = select_best_resolution((H,W), image_grid_pinpoints)
resized = aspect_preserving_resize(image, best)
padded = zero_pad_to(best)
patches = divide_to_patches(padded, patch_size=336)
base = resize original image to 336x336
pixel_values_for_image = [base] + patches
per patch: resize/crop/rescale/normalize -> [3,336,336]
batch pad on patch-count dimension if do_pad
```

CLIP vision block, repeated `vision_config.num_hidden_layers`:

```text
pixel_values [total_patches,3,336,336]
patch = Conv2d(3 -> 1024, kernel=stride=14, bias=False)
tokens = flatten(2).transpose(1,2)           # [total_patches,576,1024]
tokens = concat(class_token, tokens) + pos   # [total_patches,577,1024]
x = pre_layernorm(tokens)
for block:
  y = LayerNorm(x)
  q,k,v = Linear(1024 -> 1024)
  y = full self-attention(q,k,v)
  x = x + Linear(1024 -> 1024)
  y = LayerNorm(x)
  y = Linear(1024 -> 4096) -> activation -> Linear(4096 -> 1024)
  x = x + y
```

Feature select/project/pack:

```text
selected = hidden_states[layer] or cat(hidden_states[layers], dim=-1)
if strategy == "default": selected = selected[:, 1:]
projected = Linear(... -> text_hidden) -> GELU -> Linear(text_hidden -> text_hidden)
per image split by image_size_to_num_patches
base = projected[0]
patch_grid = projected[1:].view(num_patch_h, num_patch_w, 24, 24, text_hidden)
patch_grid = permute/flatten to [text_hidden, feature_h, feature_w]
patch_grid = unpad_image(patch_grid, original_image_size)
patch_grid = cat newline embedding as an extra feature column
patch_tokens = flatten spatial -> [variable_tokens, text_hidden]
image_tokens = cat(base, patch_tokens)
```

Text decoder block, repeated according to delegated config:

```text
x = RMSNorm(x)
q = Linear(hidden -> num_attention_heads * head_dim)
k,v = Linear(hidden -> num_key_value_heads * head_dim)
q,k = RoPE(q,k)
k,v = cache.update(k,v, layer_idx)
attn = causal_attention(q,k,v, mask, scaling=head_dim**-0.5)
x = residual + o_proj(attn)
y = RMSNorm(x)
y = down_proj(silu(gate_proj(y)) * up_proj(y))
x = residual + y
```

Projection bias details: LLaVA-NeXT projector biases default to enabled. Mistral attention projections are bias-free. Current Llama source uses `attention_bias` and `mlp_bias`, both defaulting false unless a delegated config overrides them.

## 6. Attention requirements

Vision attention:

- Full noncausal self-attention, no KV cache.
- CLIP Q/K/V shapes: `[total_patches, vision_heads, 577, head_dim]` for the 336/14 CLIP tower.
- Eager math order: `q @ k^T * scale`, add mask if present, fp32 softmax, cast to query dtype, dropout in training, `@ v`, transpose/reshape, output projection.

Text attention:

- Causal self-attention inherited from Llama or Mistral.
- Query shape before attention: `[B, num_attention_heads, q_len, head_dim]`.
- Cached K/V shape per layer before repeat expansion: `[B, num_key_value_heads, kv_seq_len, head_dim]`.
- Eager fallback repeats K/V only for attention math; DinoML should store compact KV-head cache.
- Cached keys are stored after RoPE because the cache update happens after `apply_rotary_pos_emb`.
- Mistral passes `sliding_window=getattr(config, "sliding_window", None)` to the attention backend; the representative v1.6 Mistral checkpoint explicitly has `sliding_window=null`, while the tiny-random checkpoint uses 4096.
- 34B has GQA (`56` heads, `8` KV heads), so repeat factor is 7.
- `prepare_inputs_for_generation` prevents image reprocessing during cached decode. Prefill cache includes the expanded multimodal sequence; decode appends text tokens only.

Optimized attention compatibility: first DinoML support should target prefill/decode for Llama/Mistral-style RoPE + compact GQA cache. FlashAttention/SDPA parity must preserve query scaling, mask semantics, fp32 softmax behavior for eager fallback comparisons, and post-RoPE cache storage.

## 7. Position encoding and custom math

Vision positions are learned CLIP position embeddings for a fixed 336 image patch grid. Standard AnyRes processing still sends each 336x336 patch through the CLIP tower, so CLIP position interpolation is not required for the representative v1.6 path.

Text RoPE is inherited from Llama/Mistral. The source pattern is:

```python
def apply_llama_like_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

AnyRes unpadding and newline insertion are custom multimodal math:

```python
def unpad_feature_map(x, original_hw):
    # x is [hidden, padded_feature_h, padded_feature_w]
    original_h, original_w = original_hw
    current_h, current_w = x.shape[1:]
    if original_w / original_h > current_w / current_h:
        new_h = int(round(original_h * (current_w / original_w), 7))
        pad = (current_h - new_h) // 2
        return x[:, pad: current_h - pad, :]
    new_w = int(round(original_w * (current_h / original_h), 7))
    pad = (current_w - new_w) // 2
    return x[:, :, pad: current_w - pad]
```

Precomputable: CLIP learned positions, text RoPE cos/sin tables per RoPE parameter set, and static patch-grid metadata for each supported original image bucket. Dynamic: `image_sizes`, selected best resolution, unpadded feature height/width, placeholder count, and position IDs when continuing from an existing cache.

## 8. Preprocessing and input packing

Processor output contract:

- `input_ids [B, S_expanded]` and usually `attention_mask [B, S_expanded]`.
- `pixel_values [B, max_num_patches, 3, 336, 336]` when `do_pad=True`, or a list / unpadded patch dimension when batching is disabled.
- `image_sizes [num_images, 2]` in original `(height, width)` order.

Placeholder expansion:

- For each textual `<image>`, the processor computes `num_image_tokens = unpadded_features + newline_features + base_features`.
- `base_features = (336 / patch_size)^2 + num_additional_image_tokens`; with patch 14 and CLIP CLS this is `576 + 1`.
- If `vision_feature_select_strategy == "default"`, the processor subtracts one token to match CLS removal.
- `unpadded_features` and `newline_features` are derived from original image aspect ratio and the selected AnyRes grid. This is the major LLaVA-NeXT difference from fixed-token LLaVA.

Model-side stitch:

- `get_image_features` recomputes per-image patch counts from `image_sizes`, `image_grid_pinpoints`, and `vision_config.image_size`.
- If `pixel_values` is 5D, it slices off batch-padded patch rows according to those counts and concatenates real patches before the vision tower.
- It projects and packs image features, concatenates all image token rows, and builds `special_image_mask = input_ids == image_token_id`.
- It checks that `inputs_embeds[special_image_mask].numel() == image_features.numel()` and calls `inputs_embeds.masked_scatter(special_image_mask, image_features)`.

CPU/data-pipeline work can own PIL/torchvision image preprocessing and tokenization initially. GPU/runtime work should own CLIP tower, projector, packing if image features are produced on GPU, indexed placeholder copy, decoder prefill, and decode.

Layout guards:

- Source image tensors are channel-first. A channel-last optimization must rewrite per-channel normalize, pad/crop/resize axes, patch extraction, Conv2d weight layout, and feature-map unpadding axes.
- The AnyRes `view -> permute -> flatten -> unpad -> newline -> flatten` region should be treated as `no_layout_translation()` until a dedicated NHWC-aware equivalent is proven.
- Text/projector tensors are `[tokens, hidden]` or `[batch, seq, hidden]`; do not apply image layout translation after sequence formation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap CLIP Conv2d patch embed -> GEMM

Source pattern: CLIP `Conv2d(3 -> hidden, kernel_size=patch_size, stride=patch_size, bias=False)` followed by flatten spatial axes and transpose to token-major.

Replacement:

```text
NCHW image -> WindowFlatten([patch,patch,3]) -> GEMM(weight_flat.T) -> [B, Gh*Gw, hidden]
```

Preconditions:

- `kernel_size == stride == patch_size`
- padding is zero, dilation is 1, groups is 1
- input is NCHW or an explicitly transformed NHWC equivalent
- Conv2d floor-grid behavior is preserved for non-divisible dimensions, though representative processor emits 336x336 patches
- flatten order matches PyTorch Conv2d weight layout `[out_channels, in_channels, kh, kw]`

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases: non-CLIP vision tower, position interpolation path, image sizes that bypass standard 336 patch preprocessing, or an NHWC path without a matching weight transform.

Parity test sketch: compare CLIP patch Conv2d+flatten+transpose and GEMM rewrite for `[N,3,336,336]` in fp32/fp16.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern: `input_ids == image_token_id`, expand to hidden dimension, `inputs_embeds.masked_scatter(mask, image_features)`.

Replacement:

```text
indices = nonzero(input_ids == image_token_id)
copy image_features.reshape(-1, hidden) into inputs_embeds[indices]
```

Preconditions:

- Placeholder token count exactly equals packed image feature count.
- Processor order and image feature order match.
- `input_ids` path is used; defer embedding-comparison placeholder detection.

Failure cases: `inputs_embeds`-only mode, multiple images with mismatched processor order, or batched requests without explicit per-sample image descriptors.

Parity test sketch: one image, two images in one prompt, two batched prompts with different image token counts, and no-image text.

### Rewrite: cacheable AnyRes vision prefix

Source pattern: `pixel_values + image_sizes -> vision_tower -> projector -> pack_image_features`.

Replacement:

```text
vision_prefix_graph(pixel_values, image_sizes) -> packed_image_features
prefill_graph(input_ids, attention_mask, packed_image_features, image_token_positions)
```

Preconditions:

- Same image preprocessing output and `image_sizes`.
- Same `vision_feature_layer`, select strategy, projector weights, and dtype.
- The prefill graph receives explicit image token positions or validates them from `input_ids`.

Failure cases: dynamic runtime changes to feature layer strategy, unsupported merge type, or processors that do not expose `image_sizes`.

### Rewrite: fixed-grid pack/unpad specialization

Source pattern: per-image Python list packing and dynamic `unpad_image`.

Replacement:

```text
for known image bucket:
  reshape/permute/flatten with static grid_h/grid_w
  slice static unpadded feature ranges
  append newline token per row
```

Preconditions: bucketed original image sizes or precomputed unpad extents; grid pinpoints fixed; CLIP 336/14 patch grid.

Failure cases: arbitrary unbucketed image sizes, changed grid pinpoints, or processor/model patch-size mismatch.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder RMSNorm, RoPE, causal/GQA FlashAttention with compact KV cache, and SwiGLU MLP fusion. These dominate prefill/decode cost.
- Indexed placeholder copy replacing broad `masked_scatter`; it is correctness-critical and avoids an awkward expanded boolean mask over `[B,S,H]`.
- AnyRes pack/unpad/newline kernel or bucket-specialized lowering after CLIP/projector. This is unique to LLaVA-NeXT and otherwise becomes Python/list glue.

Medium priority:

- CLIP patch Conv2d-as-GEMM and CLIP LayerNorm/attention/MLP kernels.
- Projector Linear-GELU-Linear fusion over image tokens.
- Last-token-only logits via `logits_to_keep=1` for decode.
- Cacheable vision-prefix scheduling so repeated prompts over the same image do not rerun CLIP/projector.

Lower priority:

- GPU image preprocessing. Keep CPU preprocessing first unless it becomes the bottleneck.
- CLIP position interpolation for nonstandard paths.
- `inputs_embeds`-only placeholder comparison.
- Beam-search-specific generation controllers.

## 11. Runtime staging plan

Stage 1: parse `LlavaNextConfig`, resolve delegated CLIP and text configs, and load weights for the tiny-random checkpoint. Stub tokenizer/image preprocessing with recorded tensors.

Stage 2: implement projector and placeholder stitch parity using random image features and fixed expanded `input_ids`.

Stage 3: implement CLIP 336/14 vision tower parity for total patch batches; select hidden layer `-2`, remove CLS for default strategy, and project.

Stage 4: implement AnyRes `pack_image_features` parity for bucketed `image_sizes`, including unpadding and newline insertion.

Stage 5: run decoder prefill parity from stitched `inputs_embeds` for Vicuna/Llama and Mistral variants.

Stage 6: implement cached decode with compact KV cache and the generation guard that omits image tensors after first cached iteration.

Stage 7: enable optimized attention/GEMM rewrites behind config guards, then split vision-prefix and prefill graphs for production scheduling.

Initially stub: chat-template/tokenizer exactness, CPU image preprocessing inside DinoML, beam search, training loss, embedding-comparison placeholder mode, and arbitrary unbucketed image sizes.

## 12. Parity and validation plan

- Config parity: instantiate checkpoint configs and verify effective delegated defaults for omitted Vicuna/Mistral fields.
- Processor parity: compare `pixel_values` shape, `image_sizes`, and expanded image-token counts for square, wide, and tall images.
- Patch-count parity: compare `image_size_to_num_patches` and processor `_get_number_of_features` for each representative grid bucket.
- Projector parity: random selected vision states through Linear-GELU-Linear in fp32/fp16.
- AnyRes packing parity: synthetic projected features through `pack_image_features`, including one-patch and multi-patch cases with newline.
- Placeholder stitch parity: compare `masked_scatter` and indexed copy for one image, multiple images, batched prompts, and mismatch error handling.
- Vision parity: CLIP hidden state `-2` default crop for 336/14 patches.
- Text parity: one Llama block, one Mistral block with GQA, full prefill logits, and one-token/multi-token cached decode.
- End-to-end parity: one image+prompt through Transformers and DinoML with greedy next-token logits.
- Recommended tolerances: fp32 isolated ops around `1e-4`; fp16/bf16 full logits around `1e-2` unless accumulation/layout differs; stricter tolerances for pure copies and index/stitch checks.

No DinoML tests or benchmarks were run for this docs-only audit.

## 13. Performance probes

- CPU preprocessing images/sec by original resolution and selected grid bucket.
- CLIP vision throughput over total patch count: 3, 5, 7, and larger padded patch batches.
- Projector throughput over packed image token lengths.
- AnyRes pack/unpad/newline latency and memory bandwidth.
- Placeholder stitch latency versus sequence length and image-token count.
- Prefill tokens/sec with expanded multimodal sequence lengths.
- Decode tokens/sec with `logits_to_keep=1`, batch-size sweep, and compact GQA cache.
- KV cache memory: `layers * 2 * batch * num_key_value_heads * seq * head_dim * dtype_size`.
- Attention backend comparison: eager, SDPA, FlashAttention for Llama MHA, Mistral GQA, and 34B GQA.
- End-to-end requests/hour with and without cached vision prefix.

All probes are proposed source-derived work items; no measurements are included.

## 14. Skip/defer list

- Training, labels/loss, gradients, and gradient checkpointing.
- Beam search and advanced generation controllers beyond basic sampling hooks.
- `inputs_embeds`-only image placeholder detection.
- Remote-code or non-`llava_next` variants.
- Non-`spatial_unpad` merge types.
- Arbitrary image grid pinpoints beyond guarded/bucketed support.
- GPU image preprocessing for first integration.
- Multi-GPU tensor parallel, quantization, and speculative decoding.
- CLIP position interpolation for nonstandard direct image sizes.

## 15. Final implementation checklist

- [ ] Parse `LlavaNextConfig` plus delegated `vision_config` and `text_config`.
- [ ] Resolve omitted delegated config defaults before planning shapes.
- [ ] Load CLIP vision, projector, image_newline, text decoder, and LM head weights.
- [ ] Implement processor fixtures for `pixel_values`, `image_sizes`, expanded `input_ids`, and `attention_mask`.
- [ ] Implement `image_size_to_num_patches` and `_get_number_of_features` parity.
- [ ] Implement CLIP 336/14 vision tower path.
- [ ] Implement feature selection: layer int/list, default CLS removal, full strategy, concat on feature dim.
- [ ] Implement projector Linear -> GELU -> Linear with configurable bias.
- [ ] Implement AnyRes pack/unpad/newline for `spatial_unpad`.
- [ ] Implement placeholder count validation and indexed-copy stitch.
- [ ] Implement Llama/Mistral prefill from `inputs_embeds`.
- [ ] Implement compact per-layer KV cache `[B, kv_heads, seq, head_dim]`.
- [ ] Add decode path that omits `pixel_values` and `image_sizes` after the first cached iteration.
- [ ] Add strict Conv2d patch-embed-to-GEMM rewrite guard.
- [ ] Add cacheable vision-prefix/projector graph boundary.
- [ ] Add parity tests for processor counts, projector, AnyRes packing, stitch, prefill logits, and decode logits.
- [ ] Benchmark preprocessing, vision/projector, pack/stitch, prefill, decode, and KV memory separately.
