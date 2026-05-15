# Cohere2 Vision Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  transformers @ b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model id:
  CohereLabs/command-a-vision-07-2025, official repo listed on Hugging Face but gated for raw config access.
  mlx-community/command-a-vision-07-2025-4bit, open mirror used only for compact config/preprocessor/tokenizer snapshots.

Config source:
  agents/plans/transformers/cohere2_vision/_sources/mlx-community_command-a-vision-07-2025-4bit_config.json
  agents/plans/transformers/cohere2_vision/_sources/mlx-community_command-a-vision-07-2025-4bit_preprocessor_config.json
  agents/plans/transformers/cohere2_vision/_sources/mlx-community_command-a-vision-07-2025-4bit_tokenizer_config.json
  agents/plans/transformers/cohere2_vision/_sources/mlx-community_command-a-vision-07-2025-4bit_generation_config.json

Source files inspected:
  transformers/src/transformers/models/cohere2_vision/configuration_cohere2_vision.py
  transformers/src/transformers/models/cohere2_vision/modeling_cohere2_vision.py
  transformers/src/transformers/models/cohere2_vision/modular_cohere2_vision.py
  transformers/src/transformers/models/cohere2_vision/image_processing_cohere2_vision.py
  transformers/src/transformers/models/cohere2_vision/processing_cohere2_vision.py
  transformers/src/transformers/models/cohere2/configuration_cohere2.py
  transformers/src/transformers/models/cohere2/modeling_cohere2.py
  transformers/src/transformers/models/aya_vision/configuration_aya_vision.py
  transformers/src/transformers/models/aya_vision/modeling_aya_vision.py
  transformers/src/transformers/models/aya_vision/processing_aya_vision.py
  transformers/src/transformers/models/siglip/modeling_siglip.py
  transformers/src/transformers/models/siglip/configuration_siglip.py
  transformers/src/transformers/models/siglip2/modeling_siglip2.py, for contrast only

Any missing files or assumptions:
  Official CohereLabs raw config files returned 401 without credentials. The mirror config is a 4-bit MLX conversion, so quantization fields are mirror-specific unless confirmed elsewhere. Architecture dimensions match the public commit diff and model-card claims, but the report treats source code plus the open snapshot as the concrete basis.
```

The generated `modeling_cohere2_vision.py` and `image_processing_cohere2_vision.py` say future source edits should be made in `modular_cohere2_vision.py`; the generated files are still the runtime code audited here.

## 2. High-level architecture

`cohere2_vision` is a multimodal projector plus LLM generation model:

```text
CPU image tiling/normalization + text placeholder expansion
  -> SigLIP-style vision tower over 512x512 tiles
  -> pixel-shuffle + SwiGLU MLP projector
  -> masked scatter into Cohere2 token embeddings
  -> Cohere2 decoder prefill
  -> cached Cohere2 text-only decode
  -> lm_head logits
```

Stage decomposition:

- CPU/data pipeline: tile selection by nearest supported aspect ratio, optional thumbnail tile, resize/rescale/normalize to channels-first `pixel_values`, and prompt expansion with image placeholder tokens.
- Independently stageable vision path: each 512x512 tile can be encoded by the vision tower and projected before LM prefill. The projected image token embeddings are cacheable for a fixed image and processor config.
- Prefix construction: replace only `<|IMG_PATCH|>` token embeddings by projected features using `masked_scatter`; BOI/EOI/line-break tokens remain normal text embeddings.
- Prefill: Cohere2 decoder consumes the mixed embedding sequence and builds the autoregressive KV cache.
- Decode: subsequent generation iterations omit `pixel_values` when cache is used because image information is already in the cached prefix.

First useful DinoML target: image-text prefill with projected image embeddings stitched into text embeddings, then cached text decode. Vision encoder-only parity and projector parity can be validated before full LM parity.

## 3. Important config dimensions

Representative production/open-mirror dimensions:

| Field | Value | Source |
| --- | ---: | --- |
| `model_type` | `cohere2_vision` | mirror `config.json` |
| text model | `cohere2` | mirror `config.json` |
| vision model | `siglip_vision_model` | mirror `config.json` and source default |
| `vocab_size` | 256000 | mirror `text_config` |
| text `hidden_size` | 12288 | mirror `text_config` |
| text `num_hidden_layers` | 64 | mirror `text_config` |
| text `num_attention_heads` | 96 | mirror `text_config` |
| text `num_key_value_heads` | 8 | mirror `text_config` |
| text `head_dim` | 128 | mirror `text_config` |
| text `intermediate_size` | 36864 | mirror `text_config` |
| text `max_position_embeddings` | 500000 | mirror `text_config`; model card says configured context is 32k |
| `rope_theta` / `rope_scaling` | 50000 / null | mirror `text_config` |
| `sliding_window` | 4096 | mirror `text_config` |
| layer pattern | 3 sliding layers, then 1 full layer, repeated 16 times | mirror `layer_types` |
| text `logit_scale` | 1.0 | mirror `text_config` |
| `image_token_id` | 255036 | mirror and source default |
| vision `hidden_size` | 1152 | mirror `vision_config` |
| vision layers / heads | 27 / 16 | mirror `vision_config` |
| vision patch/image size | 16 / 512 | mirror `vision_config` |
| vision tokens per tile before projector | 32 x 32 = 1024 | inferred from config |
| `downsample_factor` | 2 | mirror and source default |
| projected image tokens per tile | 16 x 16 = 256 | source projector and processor |
| max crop tiles | 12 plus thumbnail when tile count > 1 | processor source and preprocessor config |
| max visual embeddings per image | 13 x 256 = 3328 | source-derived |
| projector `linear_1` | `Linear(4608 -> 36864, bias=True)` | mirror config + source |
| projector `linear_2` | `Linear(18432 -> 12288, bias=True)` | mirror config + source |
| generation cache | `cache_implementation: hybrid` in generation config; source uses `DynamicCache(config=...)` when missing | mirror + Cohere2 source |

Checkpoint sweep:

| Checkpoint/config | Availability | Operator-significant notes |
| --- | --- | --- |
| `CohereLabs/command-a-vision-07-2025` | official repo gated for raw files | Public card says 112B, F16, 32k configured context, SigLIP2-patch16-512 wording, up to 12 tiles plus thumbnail. Use as product target but require authenticated config snapshot for final loader tests. |
| `mlx-community/command-a-vision-07-2025-4bit` | open mirror snapshot | Same `cohere2_vision` architecture fields, but includes MLX 4-bit `quantization` fields. Treat packed weight format as mirror-specific, not native Transformers behavior. |
| Source defaults `Cohere2VisionConfig()` | in-library fallback, not a checkpoint | Vision defaults differ from mirror: hidden 1152, intermediate 3072, image 512, 27 layers, 12 heads; text defaults are smaller Cohere2 8k-hidden/40-layer unless overridden. Useful only for tiny construction tests, not product parity. |

No small/debug official `cohere2_vision` checkpoint was found during this audit.

## 3a. Family variation traps

- The public model card says SigLIP2, but the inspected config/source route through `model_type="siglip_vision_model"` and therefore the SigLIP Conv2d patch-embedding implementation, not the SigLIP2 linear packed-patch implementation.
- `num_key_value_heads=8` and `num_attention_heads=96`, so text attention is GQA with 12 query groups per KV head.
- Cohere2 text source computes `head_dim = hidden_size // num_attention_heads`; do not infer other projection widths.
- Full and sliding attention layers use different masks. The production pattern is 48 sliding-window layers and 16 full-attention layers.
- In the inspected Cohere2 source, RoPE is applied only when `self.sliding_window is not None`; full-attention layers receive unrotated keys/queries. This is source-derived and should be parity-tested because it is surprising.
- Standalone `Cohere2ForCausalLM` multiplies logits by `config.logit_scale`; `Cohere2VisionForConditionalGeneration` owns its own `lm_head` and does not apply that multiply. The mirror sets `logit_scale=1.0`, so this is harmless for that checkpoint but a family admission trap.
- Processor text token count and image feature count are not the same: each tile inserts 256 `<|IMG_PATCH|>` placeholders plus one `<|IMG_LINE_BREAK|>`, wrapped by BOI/EOI. Only patch placeholders are replaced by projected features.
- Projector `pixel_shuffle` assumes the vision sequence length is a perfect square and divisible by `downsample_factor`.
- `image_sizes` appears in the forward signature but is not read by the inspected model path.
- Mirror config contains historical fields such as `alignment_activation_fn`, `enable_adapter_layer_norm`, `adapter_layer_norm_eps`, `vision_feature_layer`, and `vision_feature_select_strategy` that this `cohere2_vision` source does not read. AyaVision reads feature selection fields; Cohere2Vision does not.
- `tie_word_embeddings=True` and `_tied_weights_keys` aliases `lm_head.weight` to `model.language_model.embed_tokens.weight`; weight loading/lowering must preserve that logical alias.

## 4. Operator coverage checklist

Tensor/layout ops:

- Channels-first image tensors `[tiles_total, 3, 512, 512]`; no default NHWC semantic translation.
- Vision patch Conv2d `Conv2d(3 -> 1152, kernel=16, stride=16, bias=True)` for SigLIP basis.
- Flatten/transpose patch grid to `[tiles_total, 1024, 1152]`.
- Projector reshape/permute sequence implementing pixel shuffle from `[B, 1024, 1152]` to `[B, 16, 16, 4608]`.
- `masked_scatter` or indexed copy from projected image features into `[batch, seq, 12288]` token embeddings.
- `slice(-logits_to_keep, None)` or tensor gather before LM head.

Neural network primitives:

- SigLIP encoder: LayerNorm, dense Q/K/V/O projections, dense MLP with `gelu_pytorch_tanh`, residual adds, post LayerNorm.
- Projector: `Linear(4608 -> 36864, bias=True)`, chunk last dim into `x, gate`, `SiLU(gate) * x`, then `Linear(18432 -> 12288, bias=True)`.
- Cohere2 decoder: embedding lookup, custom mean/variance LayerNorm, Q/K/V/O projections, SwiGLU MLP `gate/up/down`, final norm, tied LM head `Linear(12288 -> 256000, bias=False)`.

Attention primitives:

- Vision self-attention is noncausal MHA, 16 heads, head dim 72 for the mirror vision config.
- Text self-attention is causal GQA, 96 query heads, 8 KV heads, head dim 128, with both 4096-token sliding-window and full causal masks.
- Source dispatch supports eager, SDPA, FlashAttention, and FlexAttention through Transformers attention interfaces.

Position/rotary ops:

- Vision absolute learned position embedding over the 32x32 patch grid; interpolation exists in SigLIP source but Cohere2Vision calls the tower without passing `interpolate_pos_encoding`, so default is false.
- Cohere2 interleaved RoPE uses `repeat_interleave(freqs, 2)` and `rotate_half` over even/odd pairs, applied only to sliding layers in inspected source.

Generation/cache ops:

- Per-layer KV cache stores key/value before repeat expansion: `[batch, 8, cached_seq, 128]` per K and V for production config.
- Attention repeat expansion is transient to `[batch, 96, key_seq, 128]`.
- Decode should not resend `pixel_values` when using cache after first iteration.

Preprocessing-coupled ops:

- Aspect-ratio grid enumeration for tile counts where `cols * rows <= max_patches`.
- Resize to selected tiled canvas, crop into 512x512 tiles, append thumbnail if more than one tile, then resize/rescale/normalize.
- Placeholder expansion must match processor exactly: BOI, per-tile 256 patch placeholders and one line-break token, EOI.

Quantized/packed weight metadata:

- Native Transformers source audited here uses regular PyTorch modules. The mirror snapshot adds MLX 4-bit quantization fields (`bits=4`, `group_size=64`); DinoML should treat those as a separate loader/provider contract, with dense F16 as the safe source-equivalent fallback.

## 5. Layer/block breakdown

Image preprocessing per original image:

```text
choose (cols, rows), cols * rows <= 12
resize original to (rows * 512, cols * 512)
crop row-major 512x512 tiles
if tile_count > 1: append resized 512x512 thumbnail
rescale by 1/255 and normalize channels
flatten all tiles across batch into pixel_values
```

Vision tile encoder, repeated for each tile:

```text
pixel_values [T, 3, 512, 512]
Conv2d patch embedding -> [T, 1152, 32, 32]
flatten + transpose -> [T, 1024, 1152]
add learned absolute positions
repeat 27 SigLIP encoder layers:
  x = x + SelfAttention(LayerNorm(x))
  x = x + MLP(LayerNorm(x))
x = post LayerNorm(x)
```

Projector:

```text
image_features [T, 1024, 1152]
pixel_shuffle -> [T, 16, 16, 4608]
h = Linear(4608 -> 36864, bias=True)
x, gate = chunk(h, 2, dim=-1)
h = SiLU(gate) * x
h = Linear(18432 -> 12288, bias=True)
flattened by masked_scatter into patch-token slots
```

Cohere2 decoder block, repeated 64 times:

```text
residual = x
n = Cohere2LayerNorm(x)  # mean/variance LN, no bias
q = Linear(12288 -> 12288, bias=False)(n).view(B, S, 96, 128).transpose(1, 2)
k = Linear(12288 -> 1024, bias=False)(n).view(B, S, 8, 128).transpose(1, 2)
v = Linear(12288 -> 1024, bias=False)(n).view(B, S, 8, 128).transpose(1, 2)
if sliding layer: q, k = Cohere2 RoPE(q, k)
k, v = cache.update(k, v)
attn = causal/sliding GQA attention(q, k, v)
attn = Linear(12288 -> 12288, bias=False)(attn)
mlp = down_proj(SiLU(gate_proj(n)) * up_proj(n))
x = residual + attn + mlp
```

Final:

```text
x = Cohere2LayerNorm(x)
logits = lm_head(x[:, logits_to_keep, :])
# no Cohere2Vision wrapper logit_scale multiply in inspected source
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over 1024 patch tokens per tile.
- MHA with 16 heads and 72-dim heads for the representative config.
- No KV cache.
- Dense attention is enough for first parity; SDPA/Flash can be an optimization.

Text attention:

- Causal self-attention with mixed layer masks.
- GQA: 96 query heads, 8 KV heads, 12-way repeat expansion.
- Sliding layers use a 4096-token sliding-window causal mask and RoPE.
- Full layers use full causal mask. The source does not apply RoPE in full layers.
- Cache tensors before repeat expansion are K and V shaped `[batch, 8, seq, 128]` per layer. After repeat, attention consumes `[batch, 96, seq, 128]`.
- Eager fallback computes `matmul(q, k.T) * head_dim**-0.5`, adds mask, softmax in fp32, casts back, dropout, then `matmul(attn, v)`. Production should prefer a fused GQA attention path with sliding-window support.

Cache distinction:

- Vision/projector outputs are independently cacheable prefix embeddings, not KV cache.
- The decoder KV cache is the only autoregressive state. Once image embeddings are included in prefill, decode can be text-only.

## 7. Position encoding and custom math

Cohere2 RoPE differs from common LLaMA-style implementations by interleaving frequencies and rotating even/odd pairs:

```python
def cohere2_rope(q, k, position_ids, inv_freq):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = freqs.repeat_interleave(2, dim=-1)
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    return q * cos[:, None] + rotate_half(q) * sin[:, None], k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Precompute candidates:

- Vision absolute position embeddings are static for 512x512 tiles.
- Cohere2 inverse frequencies are static for default RoPE; cos/sin depend on runtime `position_ids` and cache length.
- Sliding/full layer mask mapping depends on sequence length, attention mask, cache length, and layer type.

## 8. Preprocessing and input packing

CPU/data-pipeline contract:

- Images are converted to RGB, resized bicubic, rescaled by `1/255`, and normalized with mirror snapshot mean/std `[0.5, 0.5, 0.5]`. Source class defaults use OpenAI CLIP mean/std, so checkpoint preprocessor config must override source defaults.
- Tile canvas selection enumerates `(cols, rows)` with `cols * rows <= max_patches`, sorted by area, and chooses the least-upscaling or least-downscaling grid.
- A thumbnail tile is appended only when the crop grid produced more than one tile.
- Processor output `pixel_values` is flattened over all tiles/images, effectively `[total_tiles, 3, 512, 512]` for tensor outputs.

Text/placeholder contract:

- Tokenizer special IDs from snapshot: BOI `<|START_OF_IMG|>` = 255033, EOI `<|END_OF_IMG|>` = 255034, line break `<|IMG_LINE_BREAK|>` = 255035, patch `<|IMG_PATCH|>` = 255036.
- For each image with `N` processed tiles, the processor replaces one image marker with:

```text
BOI + repeat N times (256 * IMG_PATCH + IMG_LINE_BREAK) + EOI
```

- Number of inserted image-region tokens is `2 + N * 257`; number of projected features to scatter is `N * 256`.
- The model checks feature/token agreement through the expanded boolean mask and `masked_scatter`.
- `return_mm_token_type_ids` can be requested from the processor, but the model forward does not consume token type IDs.

Generation controller:

- `prepare_inputs_for_generation` forwards `pixel_values` on the first generation iteration or when cache is disabled. With cache enabled, later decode steps omit image tensors.
- Mirror generation config uses sampling defaults `do_sample=true`, `temperature=0.3`, `top_p=0.75`; those are generation policy, not core graph operators.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed 512 SigLIP Conv2d patch embedding -> Linear

Source pattern:

```text
Conv2d(3 -> 1152, kernel=16, stride=16, padding=0) on [T, 3, 512, 512]
```

Replacement:

```text
WindowFlatten([16,16] NCHW patches) -> MatMul(weight_flat.T) -> BiasAdd -> [T, 1024, 1152]
```

Preconditions:

- `kernel_size == stride == 16`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Input is channels-first and spatial dimensions are exactly divisible by 16.
- Preserve source flatten order: row-major patch order after Conv2d output flatten and transpose.

Weight transform:

```python
w = conv.weight.reshape(1152, 3 * 16 * 16)
b = conv.bias
```

Failure cases: dynamic non-512 vision sizes with positional interpolation, channel-last layout without axis rewrite, grouped or padded conv variants.

Parity test: compare Conv2d path and flattened GEMM path on random `[2,3,512,512]` fp32/fp16 inputs before position add.

### Rewrite: projector pixel-shuffle + Linear -> batched GEMM over 16x16 tokens

Source pattern:

```text
[T, 1024, 1152] -> reshape/permute -> [T, 16, 16, 4608] -> Linear(4608 -> 36864)
```

Replacement:

```text
GuardedPixelShuffle2x2 -> flatten [T*256, 4608] -> GEMM + bias -> reshape
```

Preconditions:

- Vision sequence length is square.
- `sqrt(seq_len) % downsample_factor == 0`.
- Downsample factor is 2.
- Permute order exactly matches source.

Failure cases: non-square feature maps, non-2 downsample, alternate vision tower sequence formats.

### Rewrite: image embedding stitch -> indexed copy

Source pattern:

```text
inputs_embeds.masked_scatter(input_ids == image_token_id expanded over hidden, image_features)
```

Replacement:

```text
Compute patch token positions -> flatten projected image features -> IndexedCopyRows
```

Preconditions:

- `input_ids` are available or equivalent placeholder positions are provided.
- Number of patch-token positions equals `total_tiles * 256`.
- Feature flatten order follows projector output memory order.

Failure cases: `inputs_embeds`-only path comparing embeddings to the image-token embedding, mismatched prompt expansion, batched variable image counts without per-sample descriptors.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Gather selected hidden rows -> GEMM with tied embedding/lm_head weight
```

Preconditions: `logits_to_keep` is an int or explicit tensor known at runtime; preserve no-logit-scale behavior for this wrapper.

## 10. Kernel fusion candidates

Highest priority:

- Cohere2 LayerNorm: custom mean/variance LN without bias appears before every attention/MLP pair and at final norm.
- GQA attention with mixed full/sliding masks and KV cache: main decode/prefill bottleneck.
- Projector pixel-shuffle + `Linear(4608 -> 36864)` + SwiGLU + `Linear(18432 -> 12288)`: high cost for up to 3328 visual tokens per image.
- Image-token indexed copy: required for clean dynamic multimodal batching.

Medium priority:

- SigLIP Conv2d patch embedding as GEMM for 512 fixed tiles.
- SigLIP encoder LayerNorm + attention/MLP fusions for vision-only throughput.
- Cohere2 SwiGLU MLP fusion and large GEMM provider selection.
- Last-token-only logits for decode.

Lower priority:

- Vision absolute-position add folding into patch embeddings for fixed 512 tiles.
- Processor tiling acceleration on GPU; CPU pipeline is acceptable for first integration.
- Quantized MLX mirror loading; not part of native Transformers source parity.

## 11. Runtime staging plan

1. Parse config and processor snapshots, reject unsupported remote/mirror-only flags unless explicitly implemented.
2. Load dense F16 weights with tied `lm_head`/embedding alias preserved; separately classify mirror 4-bit weights as deferred provider work.
3. Implement processor-equivalent placeholder expansion metadata and an indexed-copy stitch op.
4. Validate projector alone from random `[T,1024,1152]` features, including pixel-shuffle ordering.
5. Validate SigLIP vision tower tile encoder or compose with a separately audited SigLIP implementation.
6. Run mixed-embedding Cohere2 prefill without cache optimization, comparing hidden states/logits.
7. Add DynamicCache-compatible decode for Cohere2 with mixed full/sliding masks.
8. Add optimized GQA full/sliding attention backends and last-token logits.
9. Add batching support for multiple images and variable tile counts with explicit per-sample tile descriptors.

Initially stub: sampling policy, chat template policy, beam search, mirror quantization, and GPU image tiling.

## 12. Parity and validation plan

- Processor parity: fixed image sizes covering 1 tile, multi-tile without max, and max 12 plus thumbnail; compare `num_patches`, token counts, and `pixel_values` shape.
- Aspect-ratio selection tests for wide/tall/square images against `get_optimal_tiled_canvas`.
- Projector random tensor parity in fp32 and fp16; assert output flatten order matches `masked_scatter` order.
- Stitch parity: construct prompts with one and multiple images, compare full `inputs_embeds` after scatter.
- SigLIP tile encoder parity for one tile and multiple flattened tiles; tolerance fp32 `1e-4`, fp16/bf16 `5e-3` initially.
- Cohere2 single-layer parity with both sliding and full layer types; explicitly test that full layers follow source RoPE behavior.
- Prefill logits parity on text-only and image-text inputs with `logits_to_keep=1` and `0`.
- Decode parity for one and several tokens using cache; validate cache tensor shapes `[B,8,S,128]` before repeat.
- End-to-end smoke: one image prompt, deterministic greedy decode for a few tokens. Treat sampling differences as controller-level, not graph parity.

## 13. Performance probes

- CPU processor throughput by original image resolution and tile count.
- Vision tower throughput per tile and per image for 1, 4, 8, and 13 tiles.
- Projector throughput for `T in {1, 4, 13}` tiles.
- Prefill throughput versus text length plus visual token count, especially 3328 visual embeddings.
- Decode tokens/sec for cache lengths around 4k, 32k, and larger configured contexts.
- Sliding-window versus full-attention layer timings.
- KV cache memory for 64 layers with `[B,8,S,128]` K and V, separated by full/sliding cache behavior.
- Last-token logits GEMM cost for vocab 256000.
- Dense F16 versus any future packed/quantized weight materialization path.

## 14. Skip/defer list

- Training, loss parity beyond basic label path, and gradient checkpointing.
- Beam search, speculative decoding, and safety/prompt policy.
- GPU implementation of image tiling/resizing.
- Quantized MLX 4-bit weight loading/provider support.
- Multi-GPU tensor parallel and pipeline-parallel plans.
- SigLIP2-specific packed-patch path unless an authenticated official config proves `siglip2_vision_model`.
- `inputs_embeds`-only placeholder detection path can be deferred behind `input_ids`-based multimodal serving.
- Vision positional interpolation for non-512 tile inputs, because the processor normalizes tiles to 512.

## 15. Final implementation checklist

- [ ] Parse `Cohere2VisionConfig`, nested `Cohere2Config`, nested SigLIP vision config, and processor config.
- [ ] Reject or separately route configs whose vision model is not `siglip_vision_model`.
- [ ] Preserve tied `lm_head.weight` and `model.language_model.embed_tokens.weight`.
- [ ] Implement image tile-count metadata and placeholder token expansion.
- [ ] Implement channels-first 512 tile preprocessing contract or accept preprocessed `pixel_values`.
- [ ] Implement/project SigLIP vision tower contract for `[tiles,3,512,512] -> [tiles,1024,1152]`.
- [ ] Implement projector pixel-shuffle, SwiGLU MLP, and flatten order tests.
- [ ] Implement indexed image embedding stitch for `<|IMG_PATCH|>` positions.
- [ ] Implement Cohere2 decoder block with custom LayerNorm, GQA, mixed full/sliding masks, and source RoPE behavior.
- [ ] Implement KV cache ABI `[batch, num_kv_heads, seq, head_dim]` per layer.
- [ ] Implement wrapper LM head without extra logit scaling unless a future source/config path requires it.
- [ ] Add processor, projector, stitch, single-layer, prefill, decode, and end-to-end image-text parity tests.
- [ ] Benchmark processor, vision tower, projector, prefill, decode, logits, and cache memory separately.
