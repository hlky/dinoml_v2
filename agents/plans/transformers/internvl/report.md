# InternVL Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: OpenGVLab/InternVL3-1B-hf, plus 2B/8B/14B/38B official -hf variants
Config source: HF config/tokenizer/preprocessor/generation snapshots saved under _sources/
Source files inspected:
- transformers/src/transformers/models/internvl/configuration_internvl.py
- transformers/src/transformers/models/internvl/modeling_internvl.py
- transformers/src/transformers/models/internvl/modular_internvl.py
- transformers/src/transformers/models/internvl/processing_internvl.py
- transformers/src/transformers/models/internvl/video_processing_internvl.py
- transformers/src/transformers/models/got_ocr2/image_processing_got_ocr2.py
- transformers/src/transformers/models/qwen2/modeling_qwen2.py
- transformers/src/transformers/models/qwen2/configuration_qwen2.py
Any missing files or assumptions: no gated official -hf configs encountered. Qwen2 decoder internals are composed from the separately owned `qwen2` family and should share that audit for full text-only coverage.
```

Commit-pinned source URLs:

- [configuration_internvl.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/internvl/configuration_internvl.py)
- [modeling_internvl.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/internvl/modeling_internvl.py)
- [modular_internvl.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/internvl/modular_internvl.py)
- [processing_internvl.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/internvl/processing_internvl.py)
- [video_processing_internvl.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/internvl/video_processing_internvl.py)
- [GotOcr2 image processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/got_ocr2/image_processing_got_ocr2.py)
- [Qwen2 modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2/modeling_qwen2.py)

`modeling_internvl.py` is generated from `modular_internvl.py`; future Transformers source edits should start in the modular file, but the generated modeling file is the installed behavior audited here.

Representative HF artifacts inspected:

| Repo | Link | Local snapshots | Notes |
|---|---|---|---|
| `OpenGVLab/InternVL3-1B-hf` | [HF](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | `_sources/OpenGVLab__InternVL3-1B-hf--*` | Smallest official in-library config; Qwen2 text hidden 896. |
| `OpenGVLab/InternVL3-2B-hf` | [HF](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | `_sources/OpenGVLab__InternVL3-2B-hf--*` | Qwen2 text hidden 1536. |
| `OpenGVLab/InternVL3-8B-hf` | [HF](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | `_sources/OpenGVLab__InternVL3-8B-hf--*` | Common larger dense decoder; Qwen2 text hidden 3584. |
| `OpenGVLab/InternVL3-14B-hf` | [HF](https://huggingface.co/OpenGVLab/InternVL3-14B-hf) | `_sources/OpenGVLab__InternVL3-14B-hf--*` | More layers and wider Qwen2 decoder. |
| `OpenGVLab/InternVL3-38B-hf` | [HF](https://huggingface.co/OpenGVLab/InternVL3-38B-hf) | `_sources/OpenGVLab__InternVL3-38B-hf--*` | Larger vision tower, RMSNorm vision, Q/K norm enabled; config uses historical `image_token_index`. |

## 2. High-level architecture

Primary runtime target: multimodal image/video-to-text conditional generation through `InternVLForConditionalGeneration`.

```text
CPU image/video/text preprocessing
  -> GotOcr2 image processor / InternVL video processor: RGB, resize, rescale, normalize, NCHW frames
  -> InternVLProcessor: expand <IMG_CONTEXT> placeholders and concatenate image/video patches
  -> InternVLVisionModel: Conv2d patch embedding + bidirectional ViT encoder
  -> pixel_shuffle downsample + MLP projector to Qwen2 hidden size
  -> masked_scatter projected vision tokens into text token embeddings
  -> Qwen2 causal decoder prefill/decode with KV cache
  -> LM head logits/sampling
```

Stage decomposition:

| Stage | Runtime/cache contract |
|---|---|
| CPU/data pipeline | Owns image/video decode, resize, optional crop-to-patches, frame sampling, placeholder expansion, tokenizer padding. Output `pixel_values` is channel-first. |
| Vision tower | Accepts `pixel_values[Bv,3,H,W]`; produces patch sequence with optional CLS token. No KV cache. Can be cached per image patch/frame. |
| Pixel shuffle/projector | Converts selected vision sequence to spatial grid, downsamples by `downsample_ratio`, then projects to text hidden size. Can be cached with vision output. |
| Prefix construction | Broad `masked_scatter` replaces all image-placeholder token embeddings. Processor guarantees repeated `<IMG_CONTEXT>` spans in media order, but the model accepts arbitrary positions if counts match. |
| Text prefill/decode | Qwen2 autoregressive decoder. `prepare_inputs_for_generation` forwards `pixel_values` only on the first generation iteration or when `use_cache=False`. |

## 3. Important config dimensions

Effective dimensions for `OpenGVLab/InternVL3-1B-hf`:

| Field | Value | Provenance |
|---|---:|---|
| image token id | 151667 | `config.json` |
| image sequence length per visual patch | 256 | `config.json`; processor constructor default matches |
| downsample ratio | 0.5 | `config.json` |
| vision image size / patch size | 448x448 / 14x14 | `config.json` |
| raw vision patch grid | 32x32 = 1024 patches | derived from config |
| projected tokens per patch/frame | 16x16 = 256 | derived from `downsample_ratio=0.5` and source `pixel_shuffle` |
| vision hidden/layers/heads/intermediate | 1024 / 24 / 16 / 4096 | `config.json` |
| vision norm / QK norm | LayerNorm / disabled | `config.json` |
| text model | Qwen2 | `config.json` |
| text hidden/layers/heads/KV heads | 896 / 24 / 14 / 2 | `config.json` |
| text head dim | 64 | inferred from Qwen2 source default `hidden_size // num_attention_heads`; config omits `head_dim` |
| text MLP intermediate | 4864 | `config.json` |
| vocab size | 151674 | `config.json` |
| max positions | 32768 | `config.json` |
| RoPE | dynamic, factor 2.0, theta 1000000 | `config.json`, Qwen2 RoPE implementation |
| dtype | bfloat16 | `config.json` |
| cache support | yes | Qwen2 `use_cache=True`, `DynamicCache` |

Checkpoint sweep:

| Repo | Vision shape | Vision norm/QK norm | Text hidden | Text layers | Heads/KV heads | Head dim | MLP | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `InternVL3-1B-hf` | 1024 x 24 | LayerNorm / no QK norm | 896 | 24 | 14 / 2 | 64 | 4864 | Smallest useful target. |
| `InternVL3-2B-hf` | 1024 x 24 | LayerNorm / no QK norm | 1536 | 28 | 12 / 2 | 128 | 8960 | GQA factor 6. |
| `InternVL3-8B-hf` | 1024 x 24 | LayerNorm / no QK norm | 3584 | 28 | 28 / 4 | 128 | 18944 | GQA factor 7. |
| `InternVL3-14B-hf` | 1024 x 24 | LayerNorm / no QK norm | 5120 | 48 | 40 / 8 | 128 | 13824 | More decoder depth. |
| `InternVL3-38B-hf` | 3200 x 45 | RMSNorm / QK norm enabled | 5120 | 64 | 40 / 8 | 128 | 27648 | Larger vision tower; `image_token_index` historical key present. |

Common processor/tokenizer snapshot across inspected repos:

| Field | Value | Provenance |
|---|---|---|
| image processor | `GotOcr2ImageProcessorFast` | `preprocessor_config.json`; implementation maps to GotOCR processor behavior |
| resize size | 448x448 | `preprocessor_config.json` |
| data format | `channels_first` | `preprocessor_config.json` |
| crop-to-patches | false in config; processor call default sets images `crop_to_patches=True` unless overridden | config plus `InternVLProcessorKwargs` |
| min/max image patches | 1 / 12 | `preprocessor_config.json` |
| image mean/std | ImageNet mean/std | `preprocessor_config.json` |
| tokenizer class | `Qwen2Tokenizer` | `tokenizer_config.json` |
| media tokens | `<img>`, `</img>`, `<IMG_CONTEXT>`, `<video>` | `tokenizer_config.json` |
| generation BOS/EOS | 151643 / 151645 | `generation_config.json` |

## 3a. Family variation traps

- Qwen2 is a composed text decoder, not InternVL-owned source. DinoML should share Qwen2 decode/cache/operator coverage and keep this report focused on vision/projector/processor/stitch integration.
- The 38B config uses `image_token_index` while current `InternVLConfig` reads `image_token_id`; source defaults still give 151667, but loaders should normalize or warn on the historical field instead of treating it as a separate ABI.
- Vision tower shape changes materially at 38B: hidden 3200, 45 layers, 25 heads, RMSNorm, and `use_qk_norm=True`. Projector input width becomes `3200 * 4 = 12800` before projection.
- All inspected text configs use GQA (`num_key_value_heads < num_attention_heads`), but the grouping factor varies from 5 to 7. KV cache memory and attention kernels must use per-checkpoint KV heads.
- `head_dim` is omitted in inspected Qwen2 configs and comes from source default. Do not infer from hidden size alone for future configs that may set explicit `head_dim`.
- Processor call defaults are not identical to preprocessor JSON: `InternVLProcessorKwargs` sets image `crop_to_patches=True`, while saved preprocessor configs say `crop_to_patches=false`. The call path wins unless caller overrides.
- Image patch count is dynamic when crop-to-patches is enabled. Each image can become `num_blocks + thumbnail` patches, bounded by max 12 plus thumbnail behavior.
- Video path flattens `[B_video, T, C, H, W]` into `pixel_values[(B_video*T), C, H, W]`; the model has no separate video token id in config and uses image placeholders per frame.
- Placeholder stitch source is generic `masked_scatter`. Processor-created prompts produce ordered repeated `<IMG_CONTEXT>` spans bracketed by `<img>`/`</img>`, but model-side validation only checks total element count.
- Vision source is NCHW through Conv2d, then token sequence, then NHWC-like temporary `[B,H,W,C]` for `pixel_shuffle`. Layout passes need explicit no-translation guards around axis-sensitive reshapes/permutes.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW resize/rescale/normalize in data pipeline; first DinoML target can keep this outside compiled graph.
- Conv2d patch embedding `Conv2d(3 -> vision_hidden, kernel=14, stride=14, padding=0)`.
- `flatten(2)`, `transpose(1,2)`, `reshape/view`, `permute`, `contiguous`, `cat`, `expand`, `slice`, `masked_scatter` or bounded indexed row copy.
- `pixel_shuffle` source pattern on `[B,32,32,C] -> [B,16,16,4C]` for `downsample_ratio=0.5`.
- Optional bicubic interpolate for absolute position embeddings if input patch grid differs from trained grid.

Neural primitives:

- Vision LayerNorm or RMSNorm over last dim; 38B requires vision RMSNorm and Q/K RMSNorm.
- Qwen2 RMSNorm over text hidden.
- Linear/GEMM with bias for vision Q/K/V/O, vision MLP, projector; Qwen2 attention Q/K/V have bias, Qwen2 O/MLP/lm_head are bias-free.
- Activations: GELU in vision/projector; SiLU in Qwen2 gated MLP.
- Elementwise residual add, layer-scale multiply (`lambda_1`, `lambda_2`) in every vision layer.

Attention primitives:

- Vision noncausal dense MHA, no cache, heads = 16 or 25, head_dim = 64 or 128.
- Qwen2 causal GQA with RoPE and KV cache; all inspected checkpoints use dynamic RoPE scaling.
- FlashAttention/SDPA-compatible backend is advertised by source, but eager parity path is matmul-softmax-matmul.

Preprocessing-coupled ops:

- GotOCR dynamic tiling: aspect-ratio grid search, resize to tiled canvas, crop fixed 448x448 patches, optional thumbnail.
- Video frame sampling can be CPU-owned; default `do_sample_frames=False`, but helper supports uniform sampling from metadata.

Scatter/indexed update ops:

- Source: `inputs_embeds.masked_scatter(special_image_mask, image_features)`.
- Bounded rewrite target: validate placeholder count, derive ordered row indices from `input_ids == image_token_id`, then copy flattened projected features into embedding rows. Reject arbitrary `inputs_embeds` equality-mask mode for first integration unless input ids are present.

Generation/cache ops:

- Qwen2 `DynamicCache` per layer; keys/values stored after RoPE for keys.
- `prepare_inputs_for_generation` must suppress `pixel_values` after first cached iteration.
- `logits_to_keep` slices hidden states before LM head; last-token-only logits is required for efficient decode.

## 5. Layer/block breakdown

Vision patch and embeddings:

```text
pixel_values[Bv,3,H,W] in NCHW
  -> Conv2d(3, Hv, kernel=14, stride=14)
  -> flatten spatial, transpose to [Bv, P, Hv]
  -> prepend CLS [Bv, 1+P, Hv]
  -> add absolute position embedding, interpolated if grid differs
```

Vision block, repeated `vision_layers`:

```text
x_norm = LayerNorm/RMSNorm(x)
q,k,v = Linear(Hv -> Hv, bias=attention_bias)
optional q,k = RMSNorm(q), RMSNorm(k) before head reshape
attn = noncausal MHA(q,k,v)
x = x + lambda_1 * Linear(Hv -> Hv)(attn)
y = LayerNorm/RMSNorm(x)
y = Linear(Hv -> Iv) -> GELU -> Linear(Iv -> Hv)
x = x + lambda_2 * y
```

Projector:

```text
vision_features = last_hidden_state or selected hidden_states[layer]
if strategy == "default": drop CLS token
reshape [Bv, 1024, Hv] -> [Bv, 32, 32, Hv]
pixel_shuffle(scale=0.5) -> [Bv, 16, 16, 4*Hv]
reshape -> [Bv, 256, 4*Hv]
LayerNorm(4*Hv) -> Linear(4*Hv -> Ht) -> GELU -> Linear(Ht -> Ht)
```

Qwen2 decoder block, repeated `text_layers`:

```text
res = x
x = RMSNorm(x)
q = Linear(Ht -> n_heads*head_dim, bias=True)
k,v = Linear(Ht -> n_kv_heads*head_dim, bias=True)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v, layer)
x = causal GQA(q,k,v, mask) -> Linear(n_heads*head_dim -> Ht, bias=False)
x = res + x
res = x
x = RMSNorm(x)
x = down_proj(SiLU(gate_proj(x)) * up_proj(x))
x = res + x
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over `1 + patch_count` tokens before feature selection. For 448x448 and patch 14, sequence length is 1025 including CLS.
- MHA only; no KV cache, no RoPE, no relative bias. Attention weights are not upcast in the local eager implementation.
- Source dispatches through Transformers attention backend interface and advertises SDPA/Flash/Flex support. DinoML can first lower dense matmul-softmax-matmul or route to an attention provider with noncausal support.

Text attention:

- Qwen2 causal self-attention, GQA. Inspected configs:
  - 1B: 14 query heads, 2 KV heads, head dim 64.
  - 2B/8B/14B/38B: head dim 128, KV heads 2/4/8/8.
- Q/K receive RoPE before cache update, so cached keys are post-RoPE.
- Eager path repeats KV heads to query-head count before matmul; optimized attention should avoid materializing repeat where possible.
- Masks are generated by Qwen2 `create_causal_mask`; sliding-window mask is available in source but disabled in inspected configs (`use_sliding_window=false`, `sliding_window=null`).
- Per-layer cache shape before repeat is `[B, num_key_value_heads, cached_seq, head_dim]`; query shape is `[B, num_attention_heads, q_len, head_dim]`.

## 7. Position encoding and custom math

Vision absolute position interpolation:

```python
def internvl_pos_embed(pos, height, width, patch_size):
    cls = pos[:, :1]
    patch = pos[:, 1:].reshape(1, S, S, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_size[0], width // patch_size[1]))
    return concat(cls, patch.permute(0, 2, 3, 1).reshape(1, -1, dim), dim=1)
```

For fixed 448x448 inputs this can be precomputed and interpolation avoided. Dynamic image sizes or tracing require bicubic interpolation.

Qwen2 dynamic RoPE:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

Cos/sin depend on `position_ids`, current maximum sequence for dynamic scaling, dtype, and device. Prefill can precompute tables per admitted max length; decode needs position-dependent rows.

## 8. Preprocessing and input packing

Image ABI:

- Processor returns `pixel_values[N_visual_patches, 3, 448, 448]` in channels-first format.
- For an image with `num_patches = p`, text placeholder expansion is:
  - `<img>` start token,
  - `<IMG_CONTEXT>` repeated `image_seq_length * p`,
  - `</img>` end token.
- Only `<IMG_CONTEXT>` positions are replaced by projected features. Start/end image tokens remain normal text embeddings.
- With crop-to-patches enabled, GotOCR tiling chooses a tile grid within `[min_patches, max_patches]`, crops each tile, and adds a thumbnail if more than one tile. `num_patches` therefore includes tile patches plus thumbnail.

Video ABI:

- Video processor produces `pixel_values_videos[B, T, C, H, W]` after optional sampling, resize, center crop, rescale, normalize.
- InternVL processor flattens video frames to `[B*T, C, H, W]`, sets one visual patch per frame, and emits text like `Frame1: <img>... </img>` for each frame.
- The model receives all images and video frames through the same `pixel_values` argument and the same `image_token_id` stitch path.

Placeholder stitch:

- `get_placeholder_mask` supports two modes:
  - preferred: `input_ids == config.image_token_id`;
  - fallback: equality against the image-token embedding when only `inputs_embeds` are supplied.
- It validates total element count by comparing selected embedding elements against `image_features.numel()`.
- First DinoML admission should require `input_ids`, reject fallback embedding-equality masks, and require placeholder indices to be ordered exactly as processor outputs were concatenated.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input is semantic NCHW and height/width divisible by patch size.
- Output immediately follows `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW WindowFlatten(row-major patch order) -> MatMul(weight.reshape(out, 3*14*14).T) -> BiasAdd -> [B, P, Hv]
```

Failure cases: dynamic non-divisible image sizes, altered padding/dilation/groups, or a layout pass that changes patch flatten order without transforming weights.

### Rewrite: InternVL pixel_shuffle downsample -> reshape/transpose canonical op

Preconditions:

- Input shape `[B, S, S, C]`, square `S`, `scale_factor=0.5`.
- `S * scale_factor` and `C / scale_factor` integer; inspected models have `S=32`.
- Consumer is the projector LayerNorm over last dim.

Replacement:

```text
[B,32,32,C] -> view/permute/view/permute -> [B,16,16,4C] -> reshape [B,256,4C]
```

This is not PyTorch `pixel_shuffle`/`pixel_unshuffle` exactly; preserve source axis order.

### Rewrite: masked_scatter stitch -> indexed row copy

Preconditions:

- `input_ids` present.
- Mask is exactly `input_ids == image_token_id`.
- Number of mask positions equals `Bv * projected_seq`.
- Placeholder positions are in processor media order; for first integration require each media span to be contiguous repeated `<IMG_CONTEXT>` tokens.

Replacement:

```text
inputs_embeds = embedding(input_ids)
row_indices = nonzero(input_ids == image_token_id) in row-major order
copy image_features.reshape(-1, Ht) into inputs_embeds[row_indices]
```

Failure cases: caller supplies arbitrary `inputs_embeds`, reordered placeholders, count mismatch, or mixed tokenizer conventions.

### Rewrite: last-token-only logits

Preconditions:

- In decode or generation with `logits_to_keep=1`.
- No loss computation.

Replacement:

```text
hidden[:, -1:, :] -> lm_head
```

Failure cases: training labels, `logits_to_keep` tensor selecting non-tail positions, or full prefill logits requested.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm, GQA RoPE attention with KV cache, and SwiGLU MLP. These dominate decode and can reuse Qwen2 work.
- Vision Conv2d patch embedding to GEMM plus LayerNorm/attention/MLP stack for prefill throughput.
- Placeholder indexed row copy to avoid admitting general boolean `masked_scatter`.

Medium priority:

- Projector `LayerNorm -> Linear -> GELU -> Linear`, with `Linear(4096 -> Ht)` for 1B/2B/8B/14B and `Linear(12800 -> 5120)` for 38B.
- Vision QKV projection packing when `attention_bias=false`; 38B still has q/k post-projection RMSNorm.
- Last-token-only LM head for decode.

Lower priority:

- Bicubic position interpolation for non-448 images; can be precomputed or guarded out initially.
- Video processor GPU preprocessing; CPU/data pipeline is acceptable at first.
- Dropout/training/gradient checkpointing paths.

## 11. Runtime staging plan

1. Parse `InternVLConfig`, normalize `image_token_index` to `image_token_id` when present, and compose Qwen2 config parsing.
2. Load text-only Qwen2 weights and run text prefill/decode parity using existing Qwen2 staging.
3. Implement fixed 448x448 vision tower parity for `InternVL3-1B-hf` with one image patch and no position interpolation.
4. Implement projector and source-order `pixel_shuffle`; validate `get_image_features` against Transformers.
5. Implement bounded placeholder indexed-copy stitch requiring `input_ids`.
6. Run one-image multimodal prefill logits parity with `use_cache=True`.
7. Add decode generation parity: first iteration includes `pixel_values`, subsequent cached iterations omit it.
8. Expand to dynamic image tiling and video frame flattening after the fixed-image path is stable.
9. Add 38B-specific vision RMSNorm/QK-norm and larger projector dimensions.

Stubbable initially: image/video preprocessing, chat template construction, crop-to-patches, video sampling, full-logit prefill output, beam search, and training losses.

## 12. Parity and validation plan

- Unit parity for `pixel_shuffle` with random `[B,32,32,Hv]` and scale 0.5.
- Unit parity for bounded stitch: tokenizer-produced placeholders, multiple images, count mismatch rejection, and arbitrary-position rejection.
- Vision patch embedding parity on fixed 448x448 NCHW inputs.
- One vision block parity for LayerNorm variant and RMSNorm/QK-norm variant.
- `get_image_features` parity for 1B and 38B configs: output `[Bv,256,Ht]`.
- Qwen2 text prefill/decode parity delegated to Qwen2 audit tests, including GQA cache shapes.
- End-to-end one image prompt logits parity for `InternVL3-1B-hf`.
- Video smoke parity with a small fixed number of frames: processor flattening and placeholder count, then shared image path.

Recommended tolerances: fp32 custom/layout ops `1e-5` absolute; bf16/fp16 end-to-end hidden/logit comparisons should use relaxed `1e-2` style tolerances and compare stage outputs separately before full generation.

## 13. Performance probes

- CPU preprocessing throughput: resize/tiling/tokenization separately from model time.
- Vision tower throughput over `N_visual_patches` sweep: 1, 2, 4, 8, 12+thumbnail, and video frame counts.
- Projector throughput for `Hv=1024` and `Hv=3200`.
- Prefill throughput by text length and total image tokens inserted.
- Decode tokens/sec with and without cached visual prefix.
- KV cache memory by checkpoint and batch size; separate text KV cache from cacheable vision/projector outputs.
- Attention backend comparison for vision noncausal MHA and Qwen2 causal GQA.
- LM head last-token-only versus full-prefill logits.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, dropout behavior.
- Beam search and speculative decoding.
- Arbitrary `inputs_embeds` image-token equality-mask stitch.
- General boolean `masked_scatter`.
- Dynamic non-448 vision position interpolation for first fixed-image path.
- GPU-owned image/video preprocessing and video decode.
- Sliding-window Qwen2 attention; inspected configs disable it.
- Quantized/packed weights; inspected configs are bf16 logical weights.
- Multi-GPU tensor parallel plans.

## 15. Final implementation checklist

- [ ] Parse `InternVLConfig` and normalize historical `image_token_index`.
- [ ] Compose Qwen2 config/weights through the Qwen2 integration path.
- [ ] Load InternVL vision tower weights and projector weights.
- [ ] Implement NCHW Conv2d patch embedding or guarded Conv2d-to-Linear rewrite.
- [ ] Implement vision LayerNorm/RMSNorm, optional Q/K RMSNorm, dense noncausal attention, GELU MLP, and layer-scale residuals.
- [ ] Implement absolute position embedding fast path for fixed 448x448 and guarded interpolation fallback/defer.
- [ ] Implement InternVL `pixel_shuffle(scale=0.5)` exactly.
- [ ] Implement projector `LayerNorm -> Linear -> GELU -> Linear`.
- [ ] Implement bounded placeholder indexed row copy from `input_ids == image_token_id`.
- [ ] Enforce processor ABI guards: placeholder count/order, `pixel_values` NCHW, projected tokens per visual patch.
- [ ] Implement generation staging so `pixel_values` are used only on first cached iteration.
- [ ] Add parity tests for vision tower, projector, stitch, multimodal prefill, and cached decode.
- [ ] Benchmark preprocessing, vision/projector, prefill, decode, and KV/visual-cache memory separately.
