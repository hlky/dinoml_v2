# GOT-OCR2 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: native in-library checkpoint [`stepfun-ai/GOT-OCR-2.0-hf`](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf). Historical/custom-code contrast configs inspected: [`stepfun-ai/GOT-OCR2_0`](https://huggingface.co/stepfun-ai/GOT-OCR2_0), [`abhinand/GOT-OCR-2.0-unofficial`](https://huggingface.co/abhinand/GOT-OCR-2.0-unofficial), and [`impactframes/GOT-OCR2_0`](https://huggingface.co/impactframes/GOT-OCR2_0).

Config source: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, and `generation_config.json` fetched from the official HF repo. Snapshots are under `_sources/`.

Source files inspected:

- `src/transformers/models/got_ocr2/configuration_got_ocr2.py`
- `src/transformers/models/got_ocr2/modeling_got_ocr2.py`
- `src/transformers/models/got_ocr2/modular_got_ocr2.py`
- `src/transformers/models/got_ocr2/processing_got_ocr2.py`
- `src/transformers/models/got_ocr2/image_processing_got_ocr2.py`
- `src/transformers/models/got_ocr2/image_processing_pil_got_ocr2.py`
- `src/transformers/models/qwen2/modeling_qwen2.py` and `configuration_qwen2.py` for the delegated text decoder topology.

Any missing files or assumptions: native `modeling_got_ocr2.py` is generated from `modular_got_ocr2.py`; future source edits should use the modular file. No gated official files were encountered. Older `GOT-OCR2_0` configs use remote/custom code and `model_type` values such as `GOT` or `qwen2_vl`; they are not native `got_ocr2` scope.

## 2. High-level architecture

Primary runtime target: image-conditioned autoregressive OCR generation.

Dataflow:

```text
CPU image load/resize/normalize/prompt construction
-> NCHW pixel_values
-> SAM-like vision encoder
-> convolutional projector to a spatial sequence of Qwen2-width tokens
-> masked scatter into <imgpad> token embeddings
-> Qwen2 causal decoder prefill/decode
-> lm_head logits
-> generation controller stops on <|im_end|>
```

Stage decomposition:

- CPU/data pipeline: image loading, optional crop-to-patches tiling, CLIP mean/std normalization, OCR prompt construction, box normalization to `[0, 1000]`, and tokenizer handling.
- Vision stage: one or more independent `pixel_values` entries shaped `[num_images_or_patches, 3, 1024, 1024]` for the official preprocessor.
- Projector stage: converts each image/patch to language hidden-size embeddings. Source math with the official 1024 preprocessor size produces 256 embeddings per image/patch; the official config advertises `image_seq_length=576`, but the inspected native model does not read that field in forward.
- Prefix stitch: `masked_scatter` replaces every `<imgpad>` placeholder token with projector features. Token count must equal feature count.
- Text stage: delegated Qwen2 causal decoder with RoPE, MHA for official config, DynamicCache-compatible KV cache, and tied output embedding/lm head.

## 3. Important config dimensions

Official `main` config facts:

| Field | Value |
|---|---:|
| `model_type` | `got_ocr2` |
| `torch_dtype` | `bfloat16` |
| `image_token_index` / `<imgpad>` | `151859` |
| `image_seq_length` | `576` |
| text `model_type` | `qwen2` |
| text hidden size | `1024` |
| text layers | `24` |
| text attention heads / KV heads | `16 / 16` |
| text head dim | `64` inferred from `1024 / 16` |
| text intermediate size | `2816` |
| text vocab size | `151860` |
| text RoPE theta | `1000000.0` |
| text max positions | omitted in main config; effective Qwen2 default from `GotOcr2Config` is `32768` |
| text sliding window | omitted in main config; effective default is disabled |
| vision image size / patch size | default `1024 / 16` |
| vision patch grid | `64 x 64` |
| vision hidden / output channels | `768 / 256` |
| vision layers / heads | `12 / 12` |
| vision window size | `14`, except global layers |
| vision global attention indexes | `[2, 5, 8, 11]` |
| projector output grid | source math gives `16 x 16 = 256` for 1024 input; config advertises `image_seq_length=576` but forward uses actual feature shape |

Representative checkpoint/config sweep:

| Source | Scope | Key differences |
|---|---|---|
| `stepfun-ai/GOT-OCR-2.0-hf` `main` | Native official | `model_type=got_ocr2`, nested `text_config`, bf16, `image_seq_length=576`, preprocessor size 1024. |
| `stepfun-ai/GOT-OCR-2.0-hf` commit `813b588` | Native official historical | Same 608-byte config as main in the fetched snapshot. |
| `stepfun-ai/GOT-OCR-2.0-hf` commit `5015eba` | Native official historical | Flatter config with text fields at top level, `torch_dtype=float32`, `image_token_id` alias, `sliding_window=32768` but `use_sliding_window=false`; native config maps `image_token_id` to `image_token_index`. |
| `stepfun-ai/GOT-OCR2_0` | Remote-code original | `model_type=GOT`, `auto_map` to `modeling_GOT`, token ids `im_start=151857`, `im_end=151858`, `im_patch=151859`, `image_token_len=256`; route separately. |
| `impactframes/GOT-OCR2_0` | Mirror/custom | `model_type=qwen2_vl` with custom code tags; not the native `got_ocr2` implementation. |

## 3a. Family variation traps

- Native `got_ocr2` owns the vision tower, projector, processor, placeholder stitching, and lm head wrapper, but delegates the text body to `AutoModel.from_config(text_config)`. The official config selects Qwen2; other `text_config.model_type` values would change the decoder audit owner.
- `vision_config` in the official config only contains `"model_type": ""`; all operator-significant vision dimensions come from `GotOcr2VisionConfig` defaults.
- Official processor default `num_image_tokens=256` differs from official model `image_seq_length=576`; the inspected native forward path checks placeholders against actual projector features and does not use `image_seq_length`. For native parity, DinoML should make processor placeholder count match actual projector features or reject mismatches before graph execution.
- Optional crop-to-patches multiplies placeholders by `num_patches`; each tile still produces 576 features in the native model. This is a prompt/packing growth issue, not a different vision graph.
- The vision encoder internally moves between NCHW, NHWC, window-packed NHWC, and NCHW. Treat NHWC/channel-last as a guarded local optimization only.
- Qwen2 config supports GQA/MQA and sliding attention in general. The official GOT-OCR2 config has `num_key_value_heads == num_attention_heads` and disabled sliding attention, but DinoML should keep the delegated Qwen2 admission guard explicit.
- Historical remote-code configs expose `image_token_len=256`, `use_im_start_end`, and different class names. These are not read by the inspected native source.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW resize/rescale/normalize in preprocessing, or an explicit external processor boundary.
- Conv patch embed: `Conv2d(3 -> 768, kernel=16, stride=16)` from `[B,3,1024,1024]` to `[B,768,64,64]`, then `permute(0,2,3,1)` to NHWC.
- NHWC `LayerNorm(768)`, residual add, reshape, permute, contiguous, padding, window partition/unpartition, flatten.
- Vision neck: NHWC to NCHW, `Conv2d(768 -> 256, kernel=1, bias=False)`, channels-first LayerNorm via NCHW<->NHWC permutes, `Conv2d(256 -> 256, kernel=3, padding=1, bias=False)`.
- Projector: `Conv2d(256 -> 512, kernel=3, stride=2, padding=1, bias=False)`, `Conv2d(512 -> 1024, kernel=3, stride=2, padding=1, bias=False)`, flatten spatial to sequence, `Linear(1024 -> 1024)`.
- Embedding lookup and boolean mask/scatter replacement for image placeholders.

Neural primitives:

- Vision MLP `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`.
- Qwen2 RMSNorm, biasful Q/K/V projections, biasless output projection, SwiGLU MLP `gate/up/down`, tied output projection.
- LM head `Linear(1024 -> 151860, bias=False)` with `logits_to_keep` slicing.

Attention primitives:

- Vision dense/window attention over NHWC spatial tokens with packed QKV `Linear(768 -> 2304, bias=True)`, relative-position bias, fp32 softmax, dropout disabled for inference.
- Qwen2 causal self-attention with RoPE, cache update, causal mask, and official MHA (`16` Q heads, `16` KV heads, head dim `64`).

Position/custom math:

- Vision absolute position table `[1,64,64,768]`.
- Vision decomposed relative position for height/width using linear interpolation, coordinate indexing, and two einsums.
- Qwen2 RoPE with theta `1000000.0`, fp32 cos/sin construction, apply to Q/K before cache update.

Preprocessing-coupled ops:

- Optional crop tiling based on closest aspect-ratio canvas, thumbnail append, and `num_patches` metadata.
- Prompt template emits `<img>` + repeated `<imgpad>` + `</img>` plus OCR task text, optional color, box, format, multi-page, and patch-reference phrases.

## 5. Layer/block breakdown

Vision patch stem:

```text
pixel_values [B,3,1024,1024]
-> Conv2d 16x16/16 to [B,768,64,64]
-> NHWC [B,64,64,768]
-> add abs_pos [1,64,64,768]
```

Vision block, repeated 12 times:

```text
x = LayerNorm_NHWC(x)
if local layer: x_windows = pad + window_partition(window=14)
qkv = Linear(768 -> 2304, bias=True)
q,k,v = reshape to [B_or_windows * 12, H*W, 64]
scores = (q * 1/sqrt(64)) @ k.T
scores += decomposed_relative_position_bias
x = softmax(scores, fp32).to(dtype) @ v
x = Linear(768 -> 768)
if local layer: x = window_unpartition + crop padding
x = residual + x
x = x + Linear(3072 -> 768)(GELU(Linear(768 -> 3072)(LayerNorm_NHWC(x))))
```

Global attention layers are indexes `2, 5, 8, 11`; the other layers use 14x14 local windows.

Vision neck and projector:

```text
[B,64,64,768] -> NCHW [B,768,64,64]
-> Conv1x1 768->256 -> LayerNorm_channels_first
-> Conv3x3 256->256 padding=1 -> LayerNorm_channels_first
-> Conv3x3 stride=2 256->512 padding=1 -> [B,512,32,32]
-> Conv3x3 stride=2 512->1024 padding=1 -> [B,1024,16,16]
-> flatten/permute -> [B,256,1024]
-> Linear(1024->1024)
```

Note: source default `image_size=1024`, official preprocessor size `1024`, and two stride-2 projector convs from a 64x64 neck yield 16x16 = 256 features. The official config still advertises `image_seq_length=576`; this field is not consumed by inspected native forward or processor defaults. This is a high-priority admission check: DinoML should validate loaded checkpoint/config/processor agreement instead of trusting `image_seq_length` alone. If a variant truly uses 1536 input, the patch grid would be 96x96 and the projector would yield 24x24 = 576.

Qwen2 decoder layer, repeated 24 times:

```text
x = residual + o_proj(causal_attention(RMSNorm(x)))
x = residual + down_proj(silu(gate_proj(RMSNorm(x))) * up_proj(RMSNorm(x)))
```

Projection shapes for the official config:

- `q_proj`: `1024 -> 1024`, bias true.
- `k_proj`: `1024 -> 1024`, bias true.
- `v_proj`: `1024 -> 1024`, bias true.
- `o_proj`: `1024 -> 1024`, bias false.
- MLP gate/up/down: `1024 -> 2816`, `1024 -> 2816`, `2816 -> 1024`, all bias false.

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- Local window attention for most layers and full 64x64 global attention for indexes `2, 5, 8, 11`.
- MHA with 12 heads, head dim 64.
- Relative position bias is decomposed over height and width and added before softmax.
- No KV cache; vision features are independently cacheable as a complete encoder/projector result.

Text attention:

- Causal self-attention from Qwen2.
- Official config is MHA (`num_key_value_heads=16`, `num_attention_heads=16`), but the source supports GQA through `repeat_kv`.
- Cache ABI uses Transformers `Cache`; `Qwen2Model` creates `DynamicCache(config)` when `use_cache=True` and no cache is supplied. Per layer stores key/value after RoPE, before repeat expansion, shaped `[B, num_key_value_heads, cache_len, head_dim]`.
- Masking uses Qwen2 causal mask mapping. Sliding masks exist in Qwen2 source but are inactive for official GOT-OCR2 because `use_sliding_window=false`.
- Flash/SDPA compatibility belongs to Qwen2; `GotOcr2PreTrainedModel` itself disables flash/sdpa/flex flags in the modular source, so first DinoML integration should use explicit dense causal attention unless a delegated Qwen2 optimized path is admitted.

## 7. Position encoding and custom math

Vision relative position:

```python
rel = interpolate(rel_pos.T[None], size=2 * max(q, k) - 1, mode="linear")
q_coords = arange(q)[:, None] * max(k / q, 1.0)
k_coords = arange(k)[None, :] * max(q / k, 1.0)
idx = (q_coords - k_coords) + (k - 1) * max(q / k, 1.0)
selected = rel.T[idx.long()]
```

Then:

```python
q_hw = q.reshape(batch_heads, q_h, q_w, head_dim)
rel_h = einsum("bhwc,hkc->bhwk", q_hw, rel_pos_h)
rel_w = einsum("bhwc,wkc->bhwk", q_hw, rel_pos_w)
scores += reshape(rel_h[..., None] + rel_w[..., None, :])
```

Qwen2 RoPE:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = concat(freqs, freqs, dim=-1)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

Vision abs-pos and rel-pos tables are weights. RoPE cos/sin depend on sequence positions and past length but can be cached per batch/position range.

## 8. Preprocessing and input packing

Official preprocessor config:

- `GotOcr2ImageProcessor`, RGB conversion, resize, rescale by `1/255`, CLIP mean/std normalization.
- Official fetched `preprocessor_config.json` uses size `1024 x 1024`; source class default is `384 x 384`. Use repo config, not source default, for the official checkpoint.
- Output `pixel_values` are channel-first tensors and `num_patches` metadata.

Optional tiling:

- `crop_to_patches=True` chooses an aspect-ratio grid with `min_patches` to `max_patches` (default 1 to 12), resizes to `tile_size * grid`, slices tiles, and appends a thumbnail when more than one tile exists.
- Multi-page input flattens pages for image processing, then sums per-page patch counts when constructing one prompt.

Prompt/placeholder contract:

- Special strings: `<|im_start|>`, `<|im_end|>`, `<img>`, `</img>`, `<imgpad>`.
- Tokenizer config maps `<img>` to `151857`, `</img>` to `151858`, and `<imgpad>` to `151859`.
- Processor constructs one prompt per logical sample containing `num_image_tokens * num_patches` `<imgpad>` tokens.
- Model verifies placeholder token count against `image_features.shape[0] * image_features.shape[1]`, then uses `masked_scatter`.
- OCR output postprocess is generation-side decoding; there is no model-owned OCR parser. End-to-end examples use `stop_strings="<|im_end|>"` and decode tokens after the prompt.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d to linear

Source pattern: `Conv2d(3 -> 768, kernel=16, stride=16, padding=0)` followed by NHWC permute.

Replacement: block/window flatten in NCHW order -> GEMM with flattened conv weights -> NHWC view.

Preconditions: static 1024x1024 or admitted dynamic dimensions divisible by 16; groups 1; dilation 1; no padding; source NCHW layout preserved through flatten order.

Failure cases: any non-default image size, non-contiguous input, or layout pass that changes spatial flatten order without transforming weights.

Parity test: compare patch stem output on random bf16/fp32 image tensors against PyTorch conv for one image and tiled batch.

### Rewrite: local window attention packing

Source pattern: pad NHWC -> reshape/permute into `[B*num_windows, 14, 14, C]` -> attention -> reverse.

Replacement: explicit window partition op or fused local-attention kernel over 14x14 windows.

Preconditions: layer not in global indexes, window size 14, NHWC hidden states, padding/crop semantics identical.

Failure cases: global attention layers, changed image grid, or layout translation that does not rewrite window axes.

### Rewrite: channels-first LayerNorm around conv neck

Source pattern: NCHW -> NHWC -> LayerNorm(C) -> NCHW.

Replacement: channels-first layernorm kernel or NHWC-conv local region if both convs are lowered in channel-last.

Preconditions: normalized dimension is channel only, no consumers observe the intermediate layout, conv weights transformed if using NHWC kernels.

Failure cases: arbitrary strided tensors or debug outputs for intermediate hidden states.

### Rewrite: image placeholder masked_scatter to indexed copy

Source pattern: `inputs_embeds.masked_scatter(image_token_mask, image_features)`.

Replacement: gather placeholder positions then indexed copy of `[total_image_tokens, hidden]`.

Preconditions: exact placeholder count equals image feature tokens; mask is generated from `input_ids == 151859`; feature ordering is batch-major and patch-major.

Failure cases: caller supplies `inputs_embeds` without `input_ids`; equality-to-embedding fallback is more fragile and should be rejected or normalized.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm + QKV projection + RoPE + causal attention, inherited from Qwen2 and dominant during text decode.
- Qwen2 SwiGLU MLP fusion: gate/up projections, SiLU multiply, down projection.
- Vision window attention kernel with relative-position add for 14x14 windows.
- Placeholder indexed-copy primitive, because it sits on the multimodal ABI boundary and avoids generic boolean masked scatter.

Medium priority:

- Conv patch embedding to GEMM or optimized Conv2d.
- Channels-first LayerNorm in the neck without explicit layout ping-pong.
- Projector conv stack and flatten/projection fusion.
- Last-token-only lm head using `logits_to_keep=1` for decode.

Lower priority:

- Processor tiling acceleration; this can remain CPU/data-pipeline work initially.
- Vision global attention optimization. Only four layers use 4096-token dense vision attention; useful, but the text decoder will dominate long generation.
- Formatting/color/box prompt helpers; they are string/tokenizer ABI, not neural kernels.

## 11. Runtime staging plan

1. Parse native `got_ocr2` config and reject remote-code `GOT`/`qwen2_vl` variants for this integration.
2. Load official weights and verify effective vision image size/projector token count against `image_seq_length=576`.
3. Implement processor boundary contract or require caller-supplied `pixel_values`, `input_ids`, and `attention_mask` with exact placeholder count.
4. Bring up projector-only parity from saved/torch vision embeddings.
5. Bring up full vision encoder parity, including local/global attention and relative position.
6. Compose image feature scatter with an already-audited Qwen2 decoder prefill.
7. Add decode with Qwen2 KV cache; pass `pixel_values` only on the first generation iteration.
8. Add optimized attention, layout rewrites, and last-token logits once parity is stable.

## 12. Parity and validation plan

- Config admission tests for official main, official historical flat config, and remote-code rejects.
- Processor tests for single image, `crop_to_patches`, multi-page flattening, color, box, and `format=True` prompt text.
- Unit parity for `get_optimal_tiled_canvas` and `get_number_of_image_patches`.
- Patch embedding, window partition/unpartition, relative-position interpolation, and one vision layer random-tensor parity.
- Vision encoder/projector parity against HF for one image; include the effective token count check.
- Placeholder stitch parity: verify exact token/features mismatch errors and matching indexed-copy output.
- Qwen2 single-layer and full-prefill parity can reuse the Qwen2 audit tolerances; recommend fp32 `1e-4` absolute/relative for isolated ops and bf16 looser end-to-end tolerance.
- Decode parity: fixed prompt/image, greedy generation, stop on `<|im_end|>`, compare generated token ids over a short horizon.

## 13. Performance probes

- CPU preprocessing throughput for resize/normalize and crop-to-patches, split by number of tiles.
- Vision encoder throughput for batch of 1, 4, 8 images/patches at effective checkpoint image size.
- Projector throughput and image-feature cache memory.
- Prefill throughput as a function of prompt length plus image placeholder count.
- Decode tokens/sec for greedy generation with cached image prefix.
- KV cache memory for 24-layer Qwen2 at prompt lengths including multi-page/tiled prompts.
- Vision local attention vs global attention layer timings.
- Layout strategy comparison: source NCHW/NHWC transitions versus guarded channel-last kernels for patch/neck/projector regions.
- LM head probe for full logits versus last-token-only `logits_to_keep=1`.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Remote-code `GOTQwenForCausalLM` and mirror `qwen2_vl` variants.
- Beam search and sampling policy beyond greedy/token parity.
- GPU implementation of image loading, PIL parity, and string prompt construction.
- Quantized/8-bit/bitsandbytes forks.
- Multi-GPU tensor parallel plans.
- Non-Qwen2 text backbones unless a native config with another `text_config.model_type` is explicitly admitted.

## 15. Final implementation checklist

- [ ] Parse `GotOcr2Config` and nested Qwen2/GotOcr2Vision configs.
- [ ] Reject remote-code `GOT` and mirror `qwen2_vl` configs in native path.
- [ ] Verify checkpoint effective vision image size and projector token count equals `image_seq_length`.
- [ ] Load tied embeddings/lm head without breaking aliasing.
- [ ] Implement or externalize official image processor ABI.
- [ ] Implement patch conv, NHWC LayerNorm, window partition/unpartition, vision attention, and relative-position bias.
- [ ] Implement vision neck and projector conv stack.
- [ ] Replace placeholder `masked_scatter` with guarded indexed copy.
- [ ] Compose with Qwen2 prefill/decode and DynamicCache ABI.
- [ ] Add processor, vision, projector, stitch, prefill, and short decode parity tests.
- [ ] Benchmark preprocessing, vision, prefill, decode, and cache memory separately.
