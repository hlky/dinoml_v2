# Transformers Audit: qianfan_ocr

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
  v4.50.3-DeepSeek-3-4398-gb75feb2af6
Model id:
  baidu/Qianfan-OCR
Config source:
  https://huggingface.co/baidu/Qianfan-OCR/raw/main/config.json
  https://huggingface.co/baidu/Qianfan-OCR/raw/main/processor_config.json
  https://huggingface.co/baidu/Qianfan-OCR/raw/main/preprocessor_config.json
  https://huggingface.co/baidu/Qianfan-OCR/raw/main/tokenizer_config.json
  https://huggingface.co/baidu/Qianfan-OCR/raw/main/generation_config.json
  https://huggingface.co/api/models/baidu/Qianfan-OCR
Source files inspected:
  transformers/src/transformers/models/qianfan_ocr/configuration_qianfan_ocr.py
  transformers/src/transformers/models/qianfan_ocr/modeling_qianfan_ocr.py
  transformers/src/transformers/models/qianfan_ocr/modular_qianfan_ocr.py
  transformers/src/transformers/models/qianfan_ocr/processing_qianfan_ocr.py
  transformers/src/transformers/models/qwen3/modeling_qwen3.py
  transformers/src/transformers/models/qwen3/configuration_qwen3.py
  transformers/src/transformers/models/got_ocr2/image_processing_got_ocr2.py
Any missing files or assumptions:
  Public checkpoint was accessible. No weights were loaded and no imports/tests were run.
  The report treats Qwen3 text behavior as required because the representative config selects text_config.model_type=qwen3.
```

Additional source notes are in `_sources/source_notes.md` and `_sources/config_snapshot.md`.

## 2. High-level architecture

Qianfan-OCR is an image-text-to-text multimodal generation model:

```text
CPU/chat/image preprocessing -> tiled 448x448 image patches -> Qianfan vision ViT
  -> pixel-shuffle downsample -> MLP multimodal projector
  -> masked scatter into Qwen3 token embeddings
  -> Qwen3 causal prefill/decode -> lm_head logits -> text/markup tokens
```

Stage decomposition:

| Stage | Owner | Cacheable independently | Notes |
| --- | --- | --- | --- |
| Chat template and placeholder expansion | CPU processor/tokenizer | Yes | Expands each `<image>` into `<img>` + `256 * num_patches` `<IMG_CONTEXT>` + `</img>`. |
| Image resize/crop/rescale/normalize | CPU or preprocessing runtime | Yes | GotOcr2 image processor emits NCHW `pixel_values` and per-image `num_patches`. |
| Vision encoder | GPU graph | Yes | Noncausal ViT over one 448x448 tile at a time, 1025 tokens including CLS. |
| Pixel shuffle + projector | GPU graph | Yes | Converts 1024 patch tokens to 256 text-width image features per tile. |
| Multimodal stitch | GPU graph | No, tied to prompt layout | `masked_scatter` replaces only `<IMG_CONTEXT>` token embeddings. |
| Text prefill/decode | GPU graph | Yes | Qwen3 causal decoder with optional KV cache. Checkpoint config sets `use_cache=false`, but generation can request cache. |
| Structured OCR output parsing | Postprocess | Yes | Source has no structured parser; coordinates/boxes are emitted as tokens and decoded by tokenizer/application code. |

## 3. Important config dimensions

Representative checkpoint `baidu/Qianfan-OCR`:

| Dimension | Value | Source |
| --- | --- | --- |
| dtype | `bfloat16` | HF `config.json` |
| parameters | 4,741,408,256 BF16 | HF API safetensors metadata |
| image token id | `151671` | HF `config.json`; overrides source default `151667` |
| image seq length | `256` per processed tile | HF `processor_config.json` |
| vision hidden/layers/heads | 1024 / 24 / 16 | HF `config.json` |
| vision head dim | 64 | inference from 1024 / 16 |
| vision patch/image | 14 / 448 | HF `config.json` |
| vision patches per tile | 32 x 32 = 1024 plus CLS | inference from patch/image |
| vision MLP | 1024 -> 4096 -> 1024, GELU | HF config and source |
| vision norm | LayerNorm eps 1e-6 | HF config |
| downsample ratio | 0.5 | HF config |
| projector | LayerNorm(4096), Linear(4096 -> 2560), GELU, Linear(2560 -> 2560) | source + config |
| text model | Qwen3 causal decoder | HF `text_config.model_type` |
| text hidden/layers | 2560 / 36 | HF config |
| text heads / KV heads / head dim | 32 / 8 / 128 | HF config |
| text query width | 4096 | 32 * 128 |
| text KV width | 1024 | 8 * 128 |
| text intermediate | 9728 | HF config |
| vocab | 153678 | HF config |
| max positions | 32768 | HF config |
| RoPE theta | 5000000 | HF config |
| sliding window | disabled (`use_sliding_window=false`, `sliding_window=null`) | HF config |
| cache support | Source supports `Cache`; checkpoint default `use_cache=false` | source + HF config |

Representative checkpoint sweep:

| Model id | Revision/API status | Operator-significant shape | Notes |
| --- | --- | --- | --- |
| `baidu/Qianfan-OCR` | public, HF API sha `623bf5d20d446abdb36606aa4547cd0c18886fe5` | 24-layer ViT + 36-layer Qwen3 GQA decoder; up to 12 dynamic image tiles by processor metadata | Only public Qianfan-OCR checkpoint found/inspected. |

## 3a. Family variation traps

- The source default `image_token_id=151667` is not the checkpoint value. The representative tokenizer maps `151667` to `<think>` and `151671` to `<IMG_CONTEXT>`, so loaders must use checkpoint config/tokenizer values.
- `qianfan_ocr` source config reads `vision_config.attention_bias`; checkpoint contains `vision_config.qkv_bias=true`. If `attention_bias` is omitted, strict source defaults make QKV biased. Treat `qkv_bias` as historical/ignored for this source basis unless conversion maps it.
- Processor defaults request `crop_to_patches=True`, while the checkpoint `preprocessor_config.json` has `crop_to_patches=false`. End-to-end parity must verify merge precedence.
- `config.json` contains historical processor fields (`dynamic_image_size`, `force_image_size`, `min_dynamic_patch`, `max_dynamic_patch`, `use_thumbnail`, `select_layer`, `template`, `ps_version`) that inspected modeling code does not read directly.
- Text `hidden_size != num_attention_heads * head_dim`: 2560 versus 4096 query/output attention width. Do not infer projection widths from hidden size.
- Qwen3 uses GQA: 32 query heads, 8 KV heads, repeat factor 4.
- Qwen3 applies RMSNorm on each Q/K head after projection and before RoPE.
- Qwen3 MLP is SwiGLU-style, not a plain FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Vision path is NCHW at patch Conv2d input, then sequence-major `[B, S, C]`, then NHWC-like `[B, H, W, C]` for pixel shuffle. Layout passes need guards at these axis-sensitive transitions.
- Source assumes square vision feature grids by `feature_size = int(channels ** 0.5)`. This is safe for 448/14 tiles after dropping CLS; unsafe for non-square feature sequences unless preprocessing keeps square tiles.
- Image patch count controls both `pixel_values` rows and placeholder token count. Mismatch is a hard runtime error before language model execution.
- Qianfan-OCR does not support video input in its processor, even though inherited InternVL paths mention video.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor ingest, dtype cast to model dtype.
- Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=14, stride=14, padding=0)`.
- Flatten spatial patches, transpose to `[B, 1024, 1024]`, prepend CLS.
- Position embedding add with optional bicubic interpolation for non-default tile sizes.
- Reshape `[B, 1024, 1024] -> [B, 32, 32, 1024]`.
- Pixel shuffle downsample at scale 0.5 using view/permute/contiguous/view/permute to `[B, 16, 16, 4096]`, then flatten to `[B, 256, 4096]`.
- Boolean/equality mask from `input_ids == image_token_id`, unsqueeze/expand, `masked_scatter`.
- Dynamic arange position IDs for decode when absent.
- Last-token or selected-index logits via `logits_to_keep`.

Neural network primitives:

- Embedding lookup for Qwen3 tokens.
- LayerNorm for vision/projector and RMSNorm for Qwen3 plus optional vision QK norm.
- Linear/GEMM families:
  - Vision attention Q/K/V/O: 1024 -> 1024 with bias for checkpoint-effective source behavior.
  - Vision MLP: 1024 -> 4096 -> 1024.
  - Projector: 4096 -> 2560 -> 2560.
  - Qwen3 attention: Q 2560 -> 4096, K/V 2560 -> 1024, O 4096 -> 2560, no bias.
  - Qwen3 MLP: gate/up 2560 -> 9728, down 9728 -> 2560, no bias.
  - LM head: 2560 -> 153678, no bias.
- GELU for vision/projector, SiLU for Qwen3 MLP, dropout/drop path can be disabled in inference.

Attention primitives:

- Vision noncausal MHA, 16 heads, head dim 64, optional SDPA/Flash/Flex backend, no cache.
- Text causal GQA, 32 query heads, 8 KV heads, head dim 128, repeat KV factor 4, optional SDPA/Flash/Flex backend.
- Causal mask generation and optional sliding mask code path from Qwen3, but checkpoint disables sliding attention.

Position/rotary ops:

- Vision learned absolute position embeddings with bicubic interpolation.
- Qwen3 RoPE over `head_dim=128`, theta 5,000,000, float32 frequency matmul/cos/sin then cast back.

Generation/cache ops:

- Transformers `Cache`/`DynamicCache` compatible Qwen3 KV cache.
- `prepare_inputs_for_generation` forwards `pixel_values` only for the first generation iteration, or whenever `use_cache=False`.

Preprocessing-coupled ops:

- GotOcr2 image processor: RGB conversion, resize, rescale, normalize, optional dynamic tiling and thumbnail append.
- Placeholder/token construction with `<img>`, `</img>`, and repeated `<IMG_CONTEXT>`.

Structured postprocess:

- No source-side parser. OCR/layout outputs are generated tokens including coordinate specials such as `<quad>`, `<box>`, and `<COORD_###>` from tokenizer metadata. DinoML should treat parsing as application/tokenizer postprocess unless a separate parser is added.

## 5. Layer/block breakdown

Vision tile path:

```text
pixel_values: [num_tiles, 3, 448, 448]
patch = Conv2d(3 -> 1024, k=14, s=14) -> [num_tiles, 1024, 32, 32]
x = flatten_hw_transpose(patch) -> [num_tiles, 1024, 1024]
x = concat(cls_token, x) -> [num_tiles, 1025, 1024]
x = x + learned_abs_pos_or_interpolated_pos
```

Vision block, repeated 24 times:

```text
res = x
x = LayerNorm(x)
q,k,v = Linear(1024 -> 1024, bias=True)
q,k,v = reshape to [B, 16, S, 64]
x = noncausal_attention(q, k, v)
x = Linear(1024 -> 1024)(x)
x = res + layer_scale_1 * x
res = x
x = LayerNorm(x)
x = Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
x = res + layer_scale_2 * x
```

Image feature projection:

```text
if select_strategy == "default": drop CLS -> [tiles, 1024, 1024]
reshape -> [tiles, 32, 32, 1024]
pixel_shuffle(scale=0.5) -> [tiles, 16, 16, 4096]
flatten -> [tiles, 256, 4096]
LayerNorm(4096) -> Linear(4096 -> 2560) -> GELU -> Linear(2560 -> 2560)
```

Multimodal stitch and decoder:

```text
text_embeds = Embedding(input_ids) -> [B, T, 2560]
mask = input_ids == image_token_id
guard: mask.count == num_tiles * 256
inputs_embeds = masked_scatter(text_embeds, mask, image_features)
Qwen3 decoder runs on [B, T, 2560]
lm_head selected hidden states -> logits [B, kept_T, 153678]
```

Qwen3 decoder block, repeated 36 times:

```text
res = x
x = RMSNorm(x)
q = Linear(2560 -> 4096, bias=False).view(B,T,32,128)
k = Linear(2560 -> 1024, bias=False).view(B,T,8,128)
v = Linear(2560 -> 1024, bias=False).view(B,T,8,128)
q,k = per-head RMSNorm(q,k)
q,k = RoPE(q,k)
k,v = cache.update(k,v) if cache is active
x = causal GQA(q,k,v), repeating KV heads 4x in eager path
x = Linear(4096 -> 2560, bias=False)(x)
x = res + x
res = x
x = RMSNorm(x)
x = Linear(9728 -> 2560)(SiLU(Linear(2560 -> 9728)(x)) * Linear(2560 -> 9728)(x))
x = res + x
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA: 16 heads, head dim 64.
- Query/key/value lengths are equal to 1025 when CLS is present.
- No KV cache, no RoPE, no relative bias.
- Attention mask is optional; normal inference has no image mask.
- Source eager path does not upcast attention weights before softmax.
- SDPA/Flash/Flex may be selected through Transformers attention interface.

Text attention:

- Causal self-attention.
- GQA: 32 Q heads, 8 KV heads, head dim 128, repeat factor 4.
- Q width and attention output width are 4096; residual hidden width is 2560.
- Q/K per-head RMSNorm occurs after projection and before RoPE.
- RoPE is applied before cache update, so cached keys are stored after position encoding.
- Cache tensor shape per layer before repeat: keys and values `[B, 8, cache_len, 128]`.
- Eager attention repeats KV to `[B, 32, cache_len, 128]`, computes scaled matmul, adds causal mask, softmaxes in fp32, casts back, then matmul with values.
- Checkpoint disables sliding-window attention. Qwen3 source can create sliding masks if config has sliding layers, but that is not required for this checkpoint.
- `prepare_inputs_for_generation` avoids re-running the vision tower after the first cached iteration; if `use_cache=False`, pixel values are forwarded each iteration.

## 7. Position encoding and custom math

Vision absolute position interpolation:

```python
def vision_pos_embed(position_embeddings, height, width, patch_h=14, patch_w=14):
    cls = position_embeddings[:, :1]
    patch = position_embeddings[:, 1:]
    side = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_h, width // patch_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, patch.shape[1])
    return concat(cls, patch, dim=1)
```

Qwen3 RoPE:

```python
def qwen3_rope(q, k, position_ids, theta=5000000, head_dim=128):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = matmul(inv[None, :, None], position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat(freqs, freqs, dim=-1)
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    return (q * cos[:, None]) + (rotate_half(q) * sin[:, None]), (k * cos[:, None]) + (rotate_half(k) * sin[:, None])
```

Precomputable: base inverse frequencies and default learned vision positions. Dynamic: RoPE position IDs with cache offsets, interpolated vision positions for non-448 tile sizes.

## 8. Preprocessing and input packing

Qianfan-OCR does not invoke an external OCR engine in the processor. The caller supplies images and text/chat messages; the model generates OCR/document markup.

Processor ABI:

- Inputs: `images`, `text`; `videos` raises `ValueError`.
- Image processor fetches images, flattens image lists, and returns `pixel_values` plus `num_patches`.
- Text must contain one `<image>` placeholder per input image after chat templating.
- Each placeholder is replaced by `<img>` + `image_seq_length * num_patches` repetitions of `<IMG_CONTEXT>` + `</img>`.
- Returned fields: tokenizer fields such as `input_ids`, `attention_mask`; image field `pixel_values`; optional `mm_token_type_ids`.
- `model_input_names` are tokenizer input names plus image processor input names.

Image preprocessing:

- Checkpoint processor uses `GotOcr2ImageProcessor`, channels-first output, RGB conversion, resize to 448x448, rescale by 1/255, normalize with CLIP mean/std.
- Optional tiling path chooses a grid with up to `max_patches=12`, resizes the image to `grid_w*448` by `grid_h*448`, slices 448x448 tiles, and appends a thumbnail if more than one tile.
- The number of image features is `sum(num_patches) * 256`; the number of `<IMG_CONTEXT>` ids must match exactly.

Runtime graph ABI:

- `input_ids`: `[B, T]`, contains `<IMG_CONTEXT>` ids at image feature insertion positions.
- `pixel_values`: `[sum_tiles, 3, 448, 448]` for checkpoint default tile size.
- `attention_mask`: standard text attention mask.
- `position_ids`: optional; Qwen3 creates arange plus cache offset if absent.
- `past_key_values`: optional Qwen3 cache.
- `image_sizes` appears in forward signature but is not consumed by inspected source.

Layout guards:

- Guard `pixel_values` as NCHW, channel count 3, and tile H/W divisible by patch size 14.
- Guard vision patch sequence after dropping CLS is a square number before pixel shuffle.
- Guard pixel shuffle `downsample_ratio=0.5` and `height,width` compatible with source view equations.
- Guard placeholder count equals projected image feature count.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 1024, kernel=14, stride=14, padding=0) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten_NCHW_to_patch_rows -> GEMM(weight_flat.T) -> BiasAdd -> sequence [B, H/14*W/14, 1024]
```

Preconditions:

- `kernel_size == stride == (14,14)`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input NCHW and channels exactly 3.
- Runtime H and W divisible by 14.
- Patch flatten order matches PyTorch Conv2d cross-correlation over NCHW.

Weight transform:

```python
w = conv.weight.reshape(1024, 3 * 14 * 14)
b = conv.bias
```

Failure cases: dynamic non-divisible image sizes, alternate channel layouts, grouped conv, or a layout pass that changes patch flatten order.

Parity test sketch: compare Conv2d path and rewritten GEMM path on bf16/fp32 random `[1,3,448,448]` and one tiled batch.

### Rewrite: image token stitch to indexed copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[..., None].expand_as(inputs_embeds), image_features)
```

Replacement:

```text
positions = nonzero(input_ids == image_token_id)
guard len(positions) == image_features.numel / hidden_size
scatter_rows(inputs_embeds, positions, image_features.reshape(-1, hidden_size))
```

Preconditions:

- `image_features` row order matches processor placeholder order.
- Token id comes from loaded checkpoint config.
- No layout translation across the hidden dimension.

Failure cases: duplicate/missing placeholders are okay only if exact count/order matches; otherwise hard error.

### Rewrite: pixel shuffle downsample as reshape-permute

Source pattern is already views/permutes on `[B, 32, 32, 1024]`.

Preconditions:

- `downsample_ratio == 0.5`.
- Square feature grid 32x32 for checkpoint default.
- Channels divisible by 2 and then 4 according to source equations.
- Preserve source axis order; do not replace with PyTorch `pixel_shuffle` without proving identical NHWC semantics.

### Rewrite: Qwen3 fused attention

Source pattern:

```text
q/k/v linear -> q/k per-head RMSNorm -> RoPE -> cache update -> causal GQA attention -> o_proj
```

Replacement: one Dinoml attention provider for GQA prefill/decode plus separate Q/K RMSNorm and RoPE, then later fused QKV/RoPE/attention when parity is established.

Preconditions:

- `head_dim=128`, `q_heads=32`, `kv_heads=8`, no sliding window for checkpoint.
- Cache keys stored after RoPE.
- Softmax dtype and mask addition order match source/backend target.

Failure cases: config enables sliding attention, dynamic rope scaling, or non-default attention backend behavior with different numerical order.

## 10. Kernel fusion candidates

Highest priority:

- Qwen3 RMSNorm and per-head Q/K RMSNorm: appears twice per layer plus inside attention; direct parity and performance wins.
- Qwen3 GQA prefill/decode attention with KV cache: dominant decoder cost, requires exact cache shape `[B,8,L,128]`.
- GEMM + SwiGLU for Qwen3 MLP: 36 layers with large 2560/9728 projections.
- Vision patch Conv2d lowered to GEMM: simple guarded rewrite unlocks existing GEMM providers.
- Image feature stitch indexed-copy: avoids generic boolean masked scatter as a first-class slow path.

Medium priority:

- Vision noncausal attention over fixed 1025 tokens.
- Vision LayerNorm + linear projection sequences.
- Projector LayerNorm + two GEMMs + GELU.
- Last-token-only logits for decode using `logits_to_keep`.

Lower priority:

- Vision position interpolation for non-448 shapes; processor normally emits 448 tiles.
- DropPath/dropout training paths; inference can elide.
- Structured coordinate-token parsing; not a core graph kernel.

## 11. Runtime staging plan

Stage 1: config/processor ABI

- Parse nested Qianfan + Qwen3 config.
- Load tokenizer special token ids, especially `<IMG_CONTEXT>`.
- Implement or wrap processor-side image tiling and placeholder expansion.

Stage 2: vision tile encoder

- Run one 448x448 tile through patch embedding and one/few vision layers.
- Add guarded Conv2d-to-GEMM rewrite.

Stage 3: projector and stitch

- Implement pixel shuffle, projector, placeholder count guard, and indexed-copy replacement.
- Validate image feature rows against processor-expanded token spans.

Stage 4: Qwen3 prefill

- Reuse/implement Qwen3 RMSNorm, RoPE, GQA, SwiGLU, causal mask, and logits.
- First parity can use `use_cache=false` because checkpoint default disables cache.

Stage 5: decode with cache

- Add cache-aware generation path and ensure image tower runs only on first iteration.
- Store keys after RoPE in `[B,8,L,128]`.

Stage 6: optimized kernels

- Enable FlashAttention/SDPA-style GQA and fixed-size vision attention.
- Add logits-to-keep and GEMM fusion/provider selections.

Stubs initially acceptable: chat template rendering, coordinate-token parser, beam search, training/dropout.

## 12. Parity and validation plan

- Processor tests:
  - One image and one `<image>` placeholder: verify `pixel_values` rows, `num_patches`, and exact count of `<IMG_CONTEXT>` ids.
  - Mismatched placeholders/images: verify hard error.
  - Tiling enabled for wide/tall image: verify patch count and token count.
- Vision ops:
  - Patch Conv2d rewrite parity on `[1,3,448,448]`.
  - Vision position interpolation parity on one non-default tile size if supported.
  - Single vision block parity in fp32 and bf16.
- Projector/stitch:
  - Pixel shuffle parity for `[B,32,32,1024]`.
  - Projector parity and masked scatter/indexed-copy parity.
  - Placeholder mismatch guard test.
- Qwen3:
  - RoPE and per-head RMSNorm random tensor parity.
  - One decoder layer parity with and without cache.
  - 36-layer prefill logits parity on text-only prompt.
  - Multimodal prefill logits parity after image stitch.
  - Decode token parity for one greedy step with cache.
- End to end:
  - Image + OCR prompt output smoke against Transformers for a small `max_new_tokens`.
  - Suggested tolerances: fp32 custom ops `rtol=1e-4, atol=1e-5`; bf16/fp16 layer/block `rtol=2e-2, atol=2e-2`, with logits/token parity checked by top-k agreement where exact equality is too strict.

## 13. Performance probes

- Processor throughput: resize/tiling/token expansion images/sec.
- Vision encoder throughput by tile count: 1, 2, 4, 8, 12 tiles.
- Projector + stitch latency separately from vision.
- Text prefill latency by total sequence length including image tokens.
- Decode tokens/sec with and without KV cache.
- KV cache memory: `36 layers * 2 * B * 8 * L * 128 * dtype_size`.
- Attention backend comparison: eager versus SDPA/Flash for text GQA and vision MHA.
- Last-token logits versus full-sequence logits.
- GEMM provider sweep for Qwen3 projections, especially 2560x4096, 2560x1024, 2560x9728, 9728x2560.
- Dynamic tiling sweep by image aspect ratio and patch count.

## 14. Skip/defer list

- Training, gradients, dropout, stochastic depth behavior.
- Video input: QianfanOCRProcessor explicitly rejects it.
- Beam search and advanced generation controllers.
- Structured OCR parser beyond tokenizer decode.
- Sliding-window attention for this checkpoint.
- Dynamic/advanced RoPE scaling variants; representative config has `rope_scaling=null`.
- Quantized/packed weights; checkpoint metadata sampled is BF16 safetensors.
- Multi-GPU tensor parallel despite Qwen3 TP plan metadata.
- Non-448 direct vision tiles, after basic interpolation tests are documented.

## 15. Final implementation checklist

- [ ] Parse QianfanOCRConfig with nested QianfanOCRVisionConfig and Qwen3Config.
- [ ] Use checkpoint/tokenizer `image_token_id=151671`, not source default.
- [ ] Implement processor-compatible image resize/normalize and optional GotOcr2 tiling.
- [ ] Implement placeholder expansion and image-token count guards.
- [ ] Load vision, projector, Qwen3, and LM head weights with tied-weight alias awareness.
- [ ] Implement NCHW patch Conv2d or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement vision learned absolute position embedding and interpolation.
- [ ] Implement 24-layer noncausal vision ViT.
- [ ] Implement Qianfan pixel shuffle downsample with axis/layout guards.
- [ ] Implement projector LayerNorm/GELU/GEMMs.
- [ ] Implement indexed image embedding stitch replacement for `masked_scatter`.
- [ ] Implement Qwen3 RMSNorm, RoPE, GQA, SwiGLU, causal mask, and logits.
- [ ] Implement optional Qwen3 KV cache with keys stored after RoPE.
- [ ] Add parity tests for processor token counts, pixel shuffle, stitch, one vision block, one decoder block, prefill logits, and one decode step.
- [ ] Benchmark processor, vision, prefill, decode, cache memory, and attention backend variants.

