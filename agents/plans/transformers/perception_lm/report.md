# Transformers Audit: `perception_lm`

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: official family ids are [`facebook/Perception-LM-1B`](https://huggingface.co/facebook/Perception-LM-1B), [`facebook/Perception-LM-3B`](https://huggingface.co/facebook/Perception-LM-3B), and [`facebook/Perception-LM-8B`](https://huggingface.co/facebook/Perception-LM-8B). All are manual-gated; raw `config.json` and processor files returned 401 without accepted access. Hub API metadata was readable and confirmed `model_type=perception_lm`, `PerceptionLMForConditionalGeneration`, file lists, manual gating, and BF16 parameter metadata. Detailed checkpoint dimensions below use open mirrors and are marked as mirror-derived.

Config source:
- Source defaults: `src/transformers/models/perception_lm/configuration_perception_lm.py`.
- Gated official metadata: Hugging Face API for the three `facebook/Perception-LM-*` repos.
- Mirror configs/processors: `Dhruvil03/Perception-LM-1B-fp16`, `PIA-SPACE-LAB/Perception-LM-1B`, `PIA-SPACE-LAB/Perception-LM-3B`, `Dhruvil03/Perception-LM-8B-Int4-NotBNB`, and one quantization variant `Dhruvil03/Perception-LM-1B-Int4bit`.
- Compact snapshot: `agents/plans/transformers/perception_lm/config_sweep.md`.

Source files inspected:
- `src/transformers/models/perception_lm/modular_perception_lm.py` - authoritative source for future edits.
- `src/transformers/models/perception_lm/modeling_perception_lm.py` - generated runtime file.
- `src/transformers/models/perception_lm/configuration_perception_lm.py`.
- `src/transformers/models/perception_lm/processing_perception_lm.py`.
- `src/transformers/models/perception_lm/image_processing_perception_lm.py`.
- `src/transformers/models/perception_lm/video_processing_perception_lm.py`.
- `src/transformers/models/perception_lm/convert_perception_lm_weights_to_hf.py`.
- Delegated body references: `src/transformers/models/llama/modeling_llama.py`, `configuration_llama.py`, and `src/transformers/models/timm_wrapper/modeling_timm_wrapper.py`, `configuration_timm_wrapper.py`.

Any missing files or assumptions:
- `image_processing_perception_lm_fast.py` is referenced by converted processor configs but was not present in the local family folder; the available source is the slow/torchvision processor plus mirror config values.
- The neural vision tower is delegated to timm via `TimmWrapperModel`; this report does not audit timm Perception Encoder internals beyond the wrapper contract and conversion-selected architecture names.
- The text decoder is delegated to `LlamaModel`; this report summarizes the source-required Llama operators but should compose a separate Llama audit for full decoder maturity.

## 2. High-level architecture

PerceptionLM is a multimodal image/video-conditioned causal LM:

```text
text/chat template + image/video preprocessing
  -> token placeholder expansion + pixel_values / pixel_values_videos
  -> timm vision tower per tile/frame
  -> PerceptionLM projector + optional 2D token pooling
  -> masked-scatter replacement of <|image|>/<|video|> token embeddings
  -> Llama causal decoder prefill/decode
  -> lm_head logits
```

Stage decomposition:
- CPU/data pipeline: chat template inserts `<|image|>` and `<|video|>` markers; image processor resizes, tiles, rescales, normalizes; video processor resizes/rescales/normalizes provided frames.
- Prefix construction: processor expands each media marker into exactly the number of placeholder tokens implied by processed tile/frame dimensions, patch size, and projector pooling ratio.
- Independently cacheable encoder/projector: image and video pixel batches can be run through `vision_tower -> projector -> pooling` before decoder prefill. The resulting feature rows can be cached per media item as long as placeholder order and dtype/device are preserved.
- Prefill: text embeddings are formed, media feature rows replace placeholder token embeddings, and the Llama decoder builds self-attention KV cache.
- Decode: `prepare_inputs_for_generation` forwards pixel tensors only on the first generation iteration, or when `use_cache=False`; subsequent cached decode is text-only Llama.

First useful DinoML runtime target: image-text conditional generation with precomputed/compiled image encoder+projector and Llama prefill/decode. Video should share most mechanics but needs separate frame-sampling ABI decisions.

## 3. Important config dimensions

Source defaults from `PerceptionLMConfig`:

| Field | Default/source behavior |
| --- | --- |
| `model_type` | `perception_lm` |
| `vision_config` | `TimmWrapperConfig()` if omitted |
| `text_config` | Llama config if omitted or if dict lacks `model_type` |
| `vision_use_cls_token` | `True`; drop token 0 from vision `last_hidden_state` before projector |
| `projector_pooling_ratio` | `1` in source default; representative checkpoints use `2` |
| `image_token_id` / `video_token_id` | `128002` / `128003` |
| `tie_word_embeddings` | Inherits `text_config.tie_word_embeddings` if omitted |

Representative checkpoint sweep, with detailed values mirror-derived:

| Checkpoint basis | Text hidden | Layers | Q/KV heads | Head dim | MLP | Vocab | Context | Vision tower | Projector |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1B mirror | 2048 | 16 | 32 / 8 | 64 | 8192 | 128256 | 11520 | `vit_pe_core_large_patch14_336`, embed 1024, depth 23 | 1024 -> 2048 -> 2048, pool 2 |
| 3B mirror | 3072 | 28 | 24 / 8 | 128 | 8192 | 128256 | 11520 | same large PE, embed 1024, depth 23 | 1024 -> 3072 -> 3072, pool 2 |
| 8B mirror/variant | 4096 | 32 | 32 / 8 | 128 | 14336 | 128256 | 11520 | `vit_pe_core_gigantic_patch14_448`, embed 1536, depth 47 | 1536 -> 4096 -> 4096, pool 2 |

Common processor values from open configs:

| Field | Value |
| --- | --- |
| image layout | channels-first tensors |
| image/video size | 448 x 448 |
| image tiling | `thumb+tile`, up to 36 tiles plus one thumbnail |
| patch/pooling | patch 14, pooling ratio 2 |
| per tile tokens after pooling | `(448 / 14 / 2)^2 = 256` |
| max image placeholders | `(36 + 1) * 256 = 9472` tokens for `thumb+tile` |
| normalization | rescale by 1/255, mean/std `[0.5, 0.5, 0.5]` in mirror configs |

## 3a. Family variation traps

- PerceptionLM source is a wrapper/composition model. Do not infer a fixed vision operator set from this folder alone; admit exact `TimmWrapperConfig.architecture` values or route to a separate timm/Perception Encoder audit.
- The text config can be any `CONFIG_MAPPING` model type in source, though checkpoints use Llama. DinoML should initially require `text_config.model_type == "llama"`.
- `hidden_size != num_attention_heads * inferred default` should not be assumed away. Source Llama has explicit `head_dim`; 3B uses 24 heads with head dim 128, so Q width is 3072, while KV width is 1024.
- GQA is required: representative configs use `num_key_value_heads=8` with more query heads.
- Official configs are gated; mirror configs can diverge. The 8B open mirror observed has `rope_scaling=null`, while 1B/3B mirrors use Llama-3 RoPE scaling. Recheck official 8B before hardcoding RoPE policy.
- `projector_pooling_ratio=2` requires the vision token count after CLS removal to be a square and divisible by 2 per side.
- `vision_use_cls_token=True` removes the first vision token. If a future vision tower omits CLS or returns image-like maps instead of token sequences, this source path is unsafe.
- Placeholder replacement uses broad boolean `masked_scatter`, but the processor creates a stricter ordered row-copy pattern. DinoML should lower under processor-derived count/order guards, not admit general boolean scatter as the first path.
- Image preprocessing is NCHW/channels-first. Any NHWC optimization must rewrite image resize/tile/vision-tower regions together or stop at a no-layout-translation boundary before source axis-sensitive `view`, `permute`, and patch embedding assumptions.
- `logits_to_keep` changes final LM head cost and slicing. Last-token-only logits is required for efficient decode.
- Quantized mirror variants may advertise BitsAndBytes config. That is a loading/provider policy, not native PerceptionLM graph behavior.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup for text tokens and image/video token embeddings.
- Equality against token ids, boolean masks, `sum`, `unsqueeze`, `expand_as`.
- Guarded row replacement for media features into `inputs_embeds`; source is `masked_scatter`.
- Flatten leading axes: `pixel_values.flatten(0, 1)` for `[B, media_count, C, H, W] -> [B*media_count, C, H, W]`.
- Projector transposes/permutes: `NLD -> LND -> NLD`.
- Adaptive pooling path: `NLD -> NCHW`, `adaptive_avg_pool2d`, flatten, transpose.
- Logit slice by integer or tensor `logits_to_keep`.

Neural network primitives:
- Timm vision tower for exact admitted PE architectures. Wrapper expects NCHW pixel values and returns token `last_hidden_state`.
- Projector: `Linear(vision_embed_dim -> text_hidden, bias=True)`, GELU, `Linear(text_hidden -> text_hidden, bias=True)`.
- Llama RMSNorm, GQA attention, SwiGLU MLP, residual adds, final RMSNorm, LM head.
- Tied embedding/LM head alias when `text_config.tie_word_embeddings` is true.

Attention primitives:
- Causal Llama self-attention with RoPE.
- GQA KV repeat for eager path or native GQA support in fused attention.
- SDPA/FlashAttention compatible backend path if masks/cache match source semantics.

Position/rotary/custom math:
- Default or Llama-3 scaled RoPE depending on `text_config.rope_parameters`.
- Position ids advance by `past_key_values.get_seq_length()` when omitted.

Generation/cache ops:
- Dynamic per-layer KV cache for Llama: keys/values stored after RoPE for K and before GQA repeat.
- `prepare_inputs_for_generation` must drop pixel tensors after the first cached iteration.
- Generation config uses BOS 128000 and EOS list `[128001, 128009]` in open configs.

Preprocessing-coupled ops:
- Image resize to thumbnail and tile canvas, tile split via `view -> permute -> contiguous -> view`, concat thumbnail plus tiles.
- Rescale/normalize channels-first images/videos.
- Placeholder expansion count based on processed media tensor height/width, patch size, pooling ratio, and number of tiles/frames.

Packed/varlen metadata:
- No cu_seqlens-style metadata in this family source. Attention is ordinary padded causal decoder attention plus HF cache.

Quantized/packed weight metadata:
- Not in native source. BitsAndBytes mirror config should be treated as an optional loader fallback or rejected for first DinoML integration.

## 5. Layer/block breakdown

Image/video feature path:

```text
pixel_values: [B, N_media, C, H, W]
flat_pixels = flatten(0, 1) -> [B*N_media, C, H, W]
vision_outputs = timm_vision_tower(flat_pixels).last_hidden_state
if vision_use_cls_token: tokens = vision_outputs[:, 1:, :]
features = Linear(vision_dim -> text_hidden, bias)
features = GELU(features)
features = Linear(text_hidden -> text_hidden, bias)
if projector_pooling_ratio > 1:
  require token_count = h*h
  features = adaptive_avg_pool2d over [B*N_media, text_hidden, h, h]
return [B*N_media, pooled_tokens, text_hidden]
```

The source briefly transposes features to `LND` for the two linear layers and then transposes back. For DinoML this is a layout artifact; the linear math is per token and can stay as `[N, L, D]` if weight layout and output parity are preserved.

Text/image/video fusion:

```text
inputs_embeds = embed_tokens(input_ids)
image_features = get_image_features(pixel_values).to(inputs_embeds dtype/device)
mask = input_ids == image_token_id, expanded over hidden dimension
require num masked scalar slots == image_features.numel()
inputs_embeds = masked_scatter(inputs_embeds, mask, image_features)
repeat for video_token_id / pixel_values_videos
```

Decoder block, repeated `text_config.num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)
q = Linear(hidden -> num_heads * head_dim, no bias)
k = Linear(hidden -> num_kv_heads * head_dim, no bias)
v = Linear(hidden -> num_kv_heads * head_dim, no bias)
q, k = RoPE(q, k, position_ids)
k, v = cache_update(k, v) if cache exists
x = causal GQA attention(q, k, v, mask)
x = Linear(num_heads * head_dim -> hidden, no bias)
x = residual + x
residual = x
x = RMSNorm(x)
x = Linear(hidden -> intermediate, no bias) -> SiLU
x = x * Linear(hidden -> intermediate, no bias)
x = Linear(intermediate -> hidden, no bias)
x = residual + x
```

Output head:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

Required for representative configs:
- Causal self-attention only in the text decoder.
- GQA: 1B has 32 Q heads / 8 KV heads / head dim 64; 3B has 24 / 8 / 128; 8B has 32 / 8 / 128.
- No cross-attention. Media enters as replaced prefix embeddings before the causal decoder.
- Query/key/value lengths are rectangular during decode: query length may be 1 while key/value length includes cached media+text prefix.
- Masking is Llama causal mask plus caller `attention_mask`.
- KV cache is Llama `DynamicCache` when `use_cache=True` and no cache is supplied. Cache update receives RoPE-applied K and raw V before KV repeat.
- Eager fallback repeats KV heads then computes matmul, adds mask, softmax in fp32, dropout, matmul V.
- Source advertises Llama support for FlashAttention, SDPA, and flex attention. DinoML can use fused GQA attention when mask/cache/position semantics match.
- No local/sliding/block-sparse attention in PerceptionLM source.

Cache distinction:
- Autoregressive KV cache belongs only to the Llama decoder.
- Vision/projector outputs are independently cacheable media features, not KV cache.
- Processor-derived placeholder counts and tile/frame tensors are data-pipeline metadata.

## 7. Position encoding and custom math

PerceptionLM itself adds no new positional math. The delegated Llama text decoder uses RoPE. Source-equivalent core:

```python
def apply_llama_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

For 1B/3B mirror configs, RoPE uses `rope_theta=500000` and Llama-3 scaled parameters with factor 32, low frequency factor 1, high frequency factor 4, and original max positions 8192. The 8B open mirror observed omits `rope_scaling`; this is a mirror-derived trap to recheck against official gated config.

Projector pooling custom guard:

```python
def projector_pool(features, ratio):
    b, num_tokens, c = features.shape
    h = int(math.sqrt(num_tokens))
    assert h * h == num_tokens
    y = features.permute(0, 2, 1).reshape(b, c, h, h)
    y = adaptive_avg_pool2d(y, (h // ratio, h // ratio))
    return y.flatten(2).transpose(1, 2)
```

The processor can precompute media placeholder counts from processed tile sizes. RoPE cos/sin can be precomputed per position bucket/cache length, but Llama-3 scaled RoPE parameters must match official config.

## 8. Preprocessing and input packing

Text and chat:
- Processor requires `text`.
- Chat template places all image markers, then video markers, then text content for each message in open Hub metadata.
- Tokenizer uses Llama-3-style tokens in open generation configs: BOS 128000, EOS 128001 and 128009. Image/video ids are 128002/128003.

Image processor:
- Input images are converted to RGB, resized with bicubic, rescaled by 1/255, normalized with mean/std `[0.5, 0.5, 0.5]` in open configs, and returned as channels-first.
- `thumb+tile` mode creates a 448x448 thumbnail plus a tiled canvas with up to 36 tiles. Tile ordering is row-major over height then width after `_split`: `[batch, nch, ncw, C, tile_h, tile_w] -> [batch, nch*ncw, C, tile_h, tile_w]`.
- For one 448 tile with patch 14 and pooling 2, placeholders per tile are 256. Max image placeholders are 9472 with 36 tiles plus thumbnail.

Video processor:
- Source processor is simpler: resize to 448x448, no center crop, rescale/normalize, RGB conversion.
- Open configs leave `num_frames`, `fps`, and `do_sample_frames` null. Video decode/frame sampling ownership is therefore outside the model graph for first integration; require callers or a data pipeline to provide frames.
- Model reuses `get_image_features` for `pixel_values_videos`, so the same vision tower/projector handles frames or video chunks. Placeholder id is 128003.

Embedding stitch:
- Source uses `masked_scatter` over an expanded boolean mask. Processor guarantees stricter row order if text was expanded by `_expand_media_tokens`.
- DinoML lowering should implement `ordered_placeholder_row_copy` with guards:
  - token ids equal `image_token_id` or `video_token_id`;
  - placeholder count times hidden size equals media feature scalar count;
  - feature flatten order matches source `masked_scatter` row-major order;
  - image and video replacement are applied in source order, image first then video.

## 9. Graph rewrite / lowering opportunities

### Rewrite: media `masked_scatter` -> ordered row copy

Source pattern:

```text
mask = (input_ids == media_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, media_features)
```

Replacement:

```text
positions = nonzero(input_ids == media_token_id) in row-major order
copy media_features.reshape(-1, hidden) into inputs_embeds[positions, :]
```

Preconditions:
- `input_ids` is available, not only `inputs_embeds`.
- Processor-expanded prompts or equivalent guard prove placeholder order/count.
- `media_features.numel() == count(media_token_id) * hidden`.
- No overlapping image/video token ids.

Failure cases: arbitrary `inputs_embeds` with no `input_ids`, non-row-major token replacement, or count mismatch.

Parity test sketch: random embeddings and random media rows; compare source `masked_scatter` with indexed row copy for multiple batch sizes and interleaved text/media positions.

### Rewrite: projector transpose-linear-transpose elimination

Source pattern:

```text
features [N,L,Dv] -> permute [L,N,Dv] -> two Linear ops -> permute [N,L,H]
```

Replacement: run both linears over the last dimension of `[N,L,D]` directly.

Preconditions: linears are per-token, no consumer observes intermediate strides, GELU is pointwise, and output layout is restored to `[N,L,H]`.

Failure cases: stride-sensitive custom hooks or exported intermediate hidden states.

Parity test sketch: compare random features through both forms for 1B/3B/8B projector dimensions.

### Rewrite: fixed 448 patch embedding Conv2d -> GEMM

This belongs to the timm/PE audit, not the PerceptionLM wrapper, but it is likely important.

Preconditions:
- Exact admitted PE timm architecture.
- Patch embedding has `kernel_size == stride == 14`, padding 0, dilation 1, groups 1.
- Input is channels-first 448x448 or an explicitly rewritten NHWC region.

Replacement:

```text
WindowFlatten([C,14,14]) -> GEMM(weight.reshape(out, C*14*14).T) -> bias -> token sequence
```

Failure cases: timm architecture changes, overlapping patches, dynamic non-448 inputs, or layout pass that does not rewrite all axis consumers.

### Rewrite: last-token-only logits

Source supports `logits_to_keep`. For decode, compile an LM-head path that computes only the last token or specified token indices.

Preconditions: loss is not requested and generation only needs selected logits.

Failure cases: training loss, full-sequence logits required for diagnostics.

## 10. Kernel fusion candidates

Highest priority:
- Llama RMSNorm and residual patterns. Every decoder block has two RMSNorms plus residual adds.
- GQA FlashAttention/SDPA with RoPE and KV cache. Decoder prefill/decode dominates after media features are stitched.
- SwiGLU MLP: gate/up projections, SiLU, multiply, down projection.
- Media row-copy stitch. Avoid general boolean scatter in production.
- Last-token-only LM head for decode.

Medium priority:
- Projector `Linear + GELU + Linear`, with optional direct `[N,L,D]` layout.
- Projector adaptive average pooling over square token grids.
- Vision patch embedding and PE block fusions, after a separate timm/PE audit.
- Image processor resize/tile throughput in CPU/data pipeline, especially `thumb+tile` for high tile counts.

Lower priority:
- Full video frame batching and frame-sampling integration.
- Quantized mirror loading paths such as BitsAndBytes; first use dense BF16/FP16 weights or DinoML's own quantized constant policy.
- Return hidden states/attentions parity.

## 11. Runtime staging plan

Stage 1: parse `PerceptionLMConfig` and mirror/official config fixtures. Require `text_config.model_type == "llama"` and allowlist `vision_config.architecture`.

Stage 2: implement PerceptionLM glue parity with synthetic vision features: token embedding, placeholder count guards, ordered row-copy stitch, call into a stub Llama block or one audited Llama block.

Stage 3: projector parity: `Linear -> GELU -> Linear -> optional adaptive pool`, with random source comparison for 1B/3B/8B dimensions.

Stage 4: compose image encoder path for one exact PE/timm architecture or accept precomputed vision tokens as an external extractor boundary. For first integration, precomputed `last_hidden_state` is a valid stub.

Stage 5: Llama prefill parity with stitched media embeddings, full logits for a short prompt.

Stage 6: decode parity with KV cache and `prepare_inputs_for_generation` behavior proving pixel tensors are used only on first cached iteration.

Stage 7: optimized attention, last-token logits, projector fusions, and media feature cache.

Stage 8: video path with explicit frame decode/sampling ownership and placeholder counts.

## 12. Parity and validation plan

- Config validation: load official gated configs when access is granted; until then test mirror configs and reject unsupported text/vision model types.
- Processor count tests: image sizes across aspect ratios, `thumb+tile` vs single tile, verify `_get_num_multimodal_tokens` and actual expanded placeholders.
- Image preprocessing tests: compare resize/tile order, NCHW layout, normalization, and max tile count for synthetic images.
- Projector tests: random `[N,L,vision_dim]` tensors for 1024/1536 vision dims and 2048/3072/4096 hidden dims; fp32 tolerance `1e-5`, fp16/bf16 tolerance around `5e-2` for long fused chains.
- Pooling tests: square token counts 1024 -> 256 with ratio 2; reject non-square token counts.
- Stitch tests: compare `masked_scatter` with ordered row copy for image-only, video-only, and mixed prompts.
- Llama block tests: one-layer and after-N-layer parity with GQA and RoPE configs.
- Prefill logits: small batch with one image feature prefix and text suffix, compare logits.
- Decode token parity: prefill with media, then one-step decode with cache and no pixel tensors.
- End-to-end smoke: one official or mirror image-text prompt after access/weights are available.

## 13. Performance probes

- Processor throughput: images/sec by input resolution and tile count.
- Vision tower throughput: tiles/sec for PE-large and PE-gigantic.
- Projector throughput: tokens/sec for 256, 1024, and 9472 media token cases.
- Prefill tokens/sec by text length plus media placeholder count.
- Decode tokens/sec with and without last-token-only logits.
- KV cache memory by batch, media token prefix length, and generated length.
- Media feature cache hit rate and memory footprint.
- Attention backend comparison: eager, SDPA, Flash/GQA fused.
- Layout probe: NCHW source path versus guarded NHWC/channel-last vision patch region.
- Quantized loading/dequant probe only after dense parity is stable.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- General `inputs_embeds` media replacement without `input_ids`; reject or fallback because source finds placeholders by embedding equality.
- Arbitrary `text_config` model types beyond Llama.
- Arbitrary timm architectures beyond exact allowlisted PE variants.
- Full general boolean `masked_scatter`.
- Official gated config assumptions until access is granted.
- Video decode and frame sampling inside DinoML runtime; require predecoded frames first.
- Hidden states/attentions outputs.
- Beam search, speculative decoding, tensor parallel, and distributed runtime.
- BitsAndBytes/native mirror quantization as first-class graph behavior.

## 15. Final implementation checklist

- [ ] Add config parser for `PerceptionLMConfig` with `text_config.model_type == "llama"` guard.
- [ ] Add official gated config recheck task for `facebook/Perception-LM-{1B,3B,8B}`.
- [ ] Add allowlist for PE timm architectures or external precomputed-vision-token boundary.
- [ ] Implement projector `Linear -> GELU -> Linear`.
- [ ] Implement projector square-token adaptive average pooling with rejection guards.
- [ ] Implement ordered image/video placeholder row-copy lowering.
- [ ] Compose Llama decoder audit/runtime for GQA RoPE causal LM.
- [ ] Preserve tied embedding/LM-head alias when config requires it.
- [ ] Implement `prepare_inputs_for_generation` media-on-first-iteration behavior.
- [ ] Add processor placeholder-count parity tests.
- [ ] Add stitch parity tests against `masked_scatter`.
- [ ] Add projector and pooling parity tests for 1B/3B/8B dimensions.
- [ ] Add prefill logits and one-step cached decode parity tests.
- [ ] Benchmark processor, vision tower, projector, prefill, decode, and KV memory separately.

