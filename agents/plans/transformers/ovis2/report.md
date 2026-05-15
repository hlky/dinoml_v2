# Ovis2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: thisisiron/Ovis2-1B-hf, thisisiron/Ovis2-2B-hf for in-library scope; AIDC-AI/Ovis2-{1B,2B,4B,8B,16B,34B} inspected as remote-code/original-config contrast.
Config source: HF config/preprocessor/processor JSON snapshots in _sources/.
Source files inspected:
  transformers/src/transformers/models/ovis2/configuration_ovis2.py
  transformers/src/transformers/models/ovis2/modeling_ovis2.py
  transformers/src/transformers/models/ovis2/modular_ovis2.py
  transformers/src/transformers/models/ovis2/processing_ovis2.py
  transformers/src/transformers/models/ovis2/image_processing_ovis2.py
  transformers/src/transformers/models/ovis2/image_processing_pil_ovis2.py
  transformers/src/transformers/models/qwen2/configuration_qwen2.py
  transformers/src/transformers/models/qwen2/modeling_qwen2.py
Any missing files or assumptions: larger converted thisisiron/*-hf configs returned 401; original AIDC-AI/Ovis2-* repos are model_type=ovis with auto_map remote code and are out of scope for the in-library Ovis2 class unless explicitly routed to a separate remote-code audit.
```

`modeling_ovis2.py` is generated from `modular_ovis2.py`; future source edits should target `modular_ovis2.py`. The in-library implementation is registered in Auto mappings for `model_type="ovis2"` and does not require top-level `trust_remote_code`. Some converted configs retain nested AIMv2 `backbone_config.auto_map` metadata, but the in-library source builds the Ovis2 vision transformer directly and does not instantiate that remote backbone.

Small snapshots written under `_sources/`:

- `thisisiron__Ovis2-{1B,2B}-hf.config.json`
- `thisisiron__Ovis2-{1B,2B}-hf.preprocessor_config.json`
- `thisisiron__Ovis2-{1B,2B}-hf.processor_config.json`
- `thisisiron__Ovis2-{1B,2B}-hf.added_tokens.json`
- `thisisiron__Ovis2-{1B,2B}-hf.special_tokens_map.json`
- `AIDC-AI__Ovis2-{1B,2B,4B,8B,16B,34B}.config.json`
- `AIDC-AI__Ovis2-{1B,2B,4B,8B,16B,34B}.modeling_ovis.py`

## 2. High-level architecture

Primary runtime target: multimodal conditional generation with image-conditioned Qwen2 causal decoding.

```text
CPU image preprocessing/token expansion
  -> Ovis2 vision tokenizer over NCHW image tiles
  -> soft visual-token distribution to text-width embedding projection
  -> indexed embedding stitch into Qwen2 token embeddings
  -> Qwen2 prefill/decode with delegated DynamicCache
  -> tied LM head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: resize/crop/tile images, normalize to NCHW `pixel_values`, compute per-image tile grids, expand `<image>` into `<IMG_START>`, `<IMG_ATOM>`, grid separators, and `<IMG_END>`.
- Vision tokenizer/projector: local ViT-like encoder produces one soft visual vocabulary distribution per visual token, pads five indicator-token columns, then multiplies by `visual_embeddings_table.weight` to produce decoder-width image embeddings.
- Prefix construction: text token embeddings are created by the Qwen2 embedding table, `<IMG_ATOM>` positions are replaced with image embeddings via `masked_scatter`, and indicator token ids are replaced with learned visual indicator embeddings.
- Prefill: the stitched `inputs_embeds` are passed to `AutoModel.from_config(config.text_config)`, concretely Qwen2Model for inspected configs.
- Decode: `prepare_inputs_for_generation` forwards `pixel_values` only on the first iteration or when cache is disabled; later decode relies on the Qwen2 KV cache.

The vision tokenizer output can be validated and cached independently from the decoder KV cache. The stitched prefix embeddings can also be cached as decoder prefill inputs, but the source does not expose a separate prefix-KV construction API.

## 3. Important config dimensions

In-library defaults from `configuration_ovis2.py`:

| Field | Default |
| --- | --- |
| `hidden_size` | 1536 |
| `vocab_size` | 151643 default class, 151936 in inspected converted checkpoints |
| `image_token_id` | 151665 (`<IMG_ATOM>` in inspected tokenizer snapshots) |
| `visual_indicator_token_ids` | 151666..151670 (`<IMG_START>`, `<IMG_GRID>`, `<IMG_COL>`, `<IMG_ROW>`, `<IMG_END>`) |
| text config | Qwen2Config |
| vision hidden / layers / heads | 1024 / 24 / 8 |
| vision intermediate | 2816 |
| vision image / patch | 224 default; 448 / 14 in inspected checkpoints |
| vision hidden_stride | 1 default; 2 in inspected checkpoints |
| vision vocab | 16384 default; 65536 in inspected checkpoints |
| visual tokenize function | `softmax` default |
| vision qkv/mlp bias | false / false |
| tie word embeddings | true |

In-library converted checkpoint sweep:

| Model | Text hidden | Layers | Q heads | KV heads | Head dim | MLP | Max pos | RoPE | Vision | Visual tokens/tile | Vocab |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `thisisiron/Ovis2-1B-hf` | 896 | 24 | 14 | 2 | 64 | 4864 | 32768 | default theta 1e6 | 1024d, 24L, 8H, 448/14, stride 2 | 256 | 151936 |
| `thisisiron/Ovis2-2B-hf` | 1536 | 28 | 12 | 2 | 128 | 8960 | 32768 | default theta 1e6 | 1024d, 24L, 8H, 448/14, stride 2 | 256 | 151936 |

Official original remote-code config contrast, out of in-library scope:

| Model | `model_type` | Requires remote code | Text hidden/layers/heads/kv | Vision backbone | Vision stride | Visual vocab |
| --- | --- | --- | --- | --- | ---: | ---: |
| `AIDC-AI/Ovis2-1B` | `ovis` | yes, `configuration_ovis.OvisConfig`, `modeling_ovis.Ovis` | 896 / 24 / 14 / 2 | `apple/aimv2-large-patch14-448` | 2 | 65536 |
| `AIDC-AI/Ovis2-2B` | `ovis` | yes | 1536 / 28 / 12 / 2 | `apple/aimv2-large-patch14-448` | 2 | 65536 |
| `AIDC-AI/Ovis2-4B` | `ovis` | yes | 2048 / 36 / 16 / 2 | `apple/aimv2-huge-patch14-448` | 2 | 65536 |
| `AIDC-AI/Ovis2-8B` | `ovis` | yes | 3584 / 28 / 28 / 4 | `apple/aimv2-huge-patch14-448` | 2 | 65536 |
| `AIDC-AI/Ovis2-16B` | `ovis` | yes | 5120 / 48 / 40 / 8 | `apple/aimv2-huge-patch14-448` | 2 | 65536 |
| `AIDC-AI/Ovis2-34B` | `ovis` | yes | 5120 / 64 / 40 / 8 | `apple/aimv2-1B-patch14-448` | 2 | 65536 |

## 3a. Family variation traps

- In-library `ovis2` and original `ovis` are not the same runtime contract. `AIDC-AI/Ovis2-*` configs use `auto_map` remote files and should be rejected or routed to a separate remote-code audit for native Ovis2 integration.
- The in-library source ignores nested `vision_config.backbone_config` behavior. DinoML should use the flattened Ovis2VisionConfig fields actually read by source, not the historical AIMv2 remote-code hints.
- `hidden_size == num_attention_heads * head_dim` for inspected Qwen2 configs, but head dim varies: 64 for 1B and 128 for 2B. Larger original configs also use GQA with `num_key_value_heads < num_attention_heads`.
- Vision token count is not simply number of patches. With 448 image size, 14 patch, and `hidden_stride=2`, each processed image tile produces `(448/14/2)^2 = 256` visual embeddings.
- Processor expansion for a tiled image produces one thumbnail block plus one block per tile when `row * col > 1`, so image atom count is `256` for a 1x1 grid and `(1 + row * col) * 256` otherwise.
- `image_token_id` is `<IMG_ATOM>`, not literal `<image>` in inspected converted processors. Literal `<image>` is only the user prompt marker before processor expansion.
- Visual indicator tokens are handled separately from atom image embeddings. They are replaced after the image `masked_scatter`, one id at a time.
- Vision source tensor layout is NCHW. Any NHWC/channel-last optimization must guard Conv2d, flatten/transpose, attention sequence layout, and the hidden-stride reshape/permute.
- Qwen2 text attention uses biased q/k/v projections, biasless output projection, biasless MLP projections, RMSNorm, RoPE, and GQA. Do not infer all projection widths from `hidden_size` alone.
- `use_sliding_window=false` in inspected configs, so Qwen2 creates full causal masks only. Source supports sliding layer types through Qwen2Config but not used here.
- The vision tokenizer supports `softmax`, `st_argmax`, and `gumbel_argmax`; inspected inference configs use `softmax`. `gumbel_argmax` is stochastic/training-like and should be rejected for deterministic inference unless explicitly needed.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image resize/crop/stack/normalize in data pipeline.
- Conv2d patch embedding: `Conv2d(3 -> vision_hidden, kernel=patch_size, stride=patch_size, padding=valid)`.
- Flatten spatial patches and transpose from `[B, C, H', W']` to `[B, H'*W', C]`.
- Hidden-stride pooling-by-reshape path when `hidden_stride > 1`: pad sequence square grid, reshape to six dimensions, permute, reshape.
- `torch.cat` over vocab dimension to append zero indicator columns.
- `arange` for visual indicator indices and decoder position ids.
- `masked_scatter` / indexed update into decoder embeddings for `<IMG_ATOM>`.
- Boolean masks and indexed assignment for five visual indicator token ids.
- Slice last logits via `logits_to_keep`.

Neural network primitives:

- Vision RMSNorm with fp32 variance and input dtype restore.
- Vision noncausal MHA: q/k/v/out Linear, softmax attention, optional SDPA/Flash/Flex backend.
- Vision SwiGLU MLP: `down(silu(gate(x)) * up(x))`, qkv/mlp bias false in inspected configs.
- Vision `head_linear`: `Linear(vision_hidden * hidden_stride^2 -> visual_vocab - 5, bias=False)`, e.g. `4096 -> 65531` for 1024 hidden, stride 2.
- Vision `LayerNorm(visual_vocab - 5)`.
- Softmax over visual vocabulary logits.
- Visual embedding projection: soft one-hot/distribution `[N, 256, visual_vocab] @ visual_embeddings_table.weight [visual_vocab, text_hidden]`.
- Qwen2 decoder stack: RMSNorm, GQA attention, SwiGLU MLP, final RMSNorm.
- LM head: `Linear(text_hidden -> vocab_size, bias=False)`, tied to decoder token embeddings by `_tied_weights_keys`.

Attention primitives:

- Vision encoder: noncausal MHA, no RoPE, no cache.
- Qwen2 decoder: causal self-attention, RoPE before cache update, GQA repeat from KV heads to Q heads in eager fallback, DynamicCache support.
- Attention masks: full causal mask for inspected configs; sliding-window masks are source-supported through Qwen2 but disabled in configs.

Position/rotary/custom math:

- Vision learned absolute position embedding of length `(image_size / patch_size)^2`, added after patch RMSNorm.
- Qwen2 default RoPE with theta 1e6 in inspected configs, cos/sin computed in fp32 then cast to hidden dtype.

Preprocessing-coupled ops:

- Crop-to-patches grid selection, thumbnail insertion for multi-tile images, BICUBIC resize, rescale by 1/255, OpenAI CLIP mean/std normalization.
- Prompt expansion from `<image>` to Ovis2 special-token sequence.

Generation/cache ops:

- Delegated Qwen2 DynamicCache per layer. Ovis2 itself only controls whether `pixel_values` are forwarded on generation iterations.

Quantized/packed weights:

- No in-library Ovis2-specific quantized or packed weight format found. Original repos are safetensors/remote-code; quantized loading is not part of this source basis.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values: [num_processed_images, 3, 448, 448] NCHW
patch = Conv2d(3 -> 1024, kernel=14, stride=14)(pixel_values)  # [N, 1024, 32, 32]
x = flatten(2).transpose(1, 2)                                 # [N, 1024, 1024]
x = RMSNorm(x)
x = x + position_embedding(position_ids)                        # learned absolute positions
```

Vision encoder block, repeated 24 times in inspected configs:

```text
y = RMSNorm(x)
q = Linear(1024 -> 1024, bias=False)(y).view(N, 1024, 8, 128).transpose(1, 2)
k = Linear(1024 -> 1024, bias=False)(y).view(...).transpose(1, 2)
v = Linear(1024 -> 1024, bias=False)(y).view(...).transpose(1, 2)
a = noncausal_attention(q, k, v, scale=128**-0.5)
x = x + Linear(1024 -> 1024, bias=False)(a)
y = RMSNorm(x)
x = x + Linear(2816 -> 1024, bias=False)(silu(Linear(1024 -> 2816)(y)) * Linear(1024 -> 2816)(y))
```

Vision tokenizer head for inspected configs:

```text
x = final RMSNorm(x)                                            # [N, 1024, 1024]
x = hidden_stride_2_pack(x)                                     # [N, 256, 4096]
logits = Linear(4096 -> 65531, bias=False)(x)
logits = LayerNorm(65531)(logits)
visual_probs = softmax(logits, dim=-1)                          # [N, 256, 65531]
visual_probs = cat([visual_probs, zeros([N, 256, 5])], dim=2)    # [N, 256, 65536]
image_features = visual_probs @ visual_embeddings_table.weight  # [N, 256, text_hidden]
indicator_features = embedding([65531..65535])                  # [5, text_hidden]
```

Embedding stitch:

```text
inputs_embeds = qwen2.embed_tokens(input_ids)
mask_atoms = input_ids == image_token_id  # 151665 / <IMG_ATOM>
assert count(mask_atoms) * text_hidden == image_features.numel()
inputs_embeds = inputs_embeds.masked_scatter(mask_atoms[..., None].expand_as(inputs_embeds), image_features)
for each visual_indicator_token_id:
    inputs_embeds[input_ids == token_id] = corresponding indicator_features[i]
```

Qwen2 decoder block, repeated per `text_config.num_hidden_layers`:

```text
y = RMSNorm(x)
q = Linear(hidden -> q_heads * head_dim, bias=True)(y)
k = Linear(hidden -> kv_heads * head_dim, bias=True)(y)
v = Linear(hidden -> kv_heads * head_dim, bias=True)(y)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache is enabled
a = causal_attention(q, k, v, GQA repeat if eager, mask, scale=head_dim**-0.5)
x = x + Linear(q_heads * head_dim -> hidden, bias=False)(a)
y = RMSNorm(x)
x = x + down_proj(silu(gate_proj(y)) * up_proj(y))              # all Qwen2 MLP projections bias=False
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over visual token sequences.
- MHA, not GQA: 8 heads, head_dim 128 for inspected 1024-hidden vision tower.
- No positional rotation inside attention; learned absolute position embedding is added before the encoder.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS` with eager fallback, and declares FlashAttention/SDPA/Flex support.
- No KV cache for vision branch.

Text decoder attention:

- Causal self-attention, Qwen2Model delegated through `AutoModel.from_config`.
- GQA in inspected configs: 1B uses 14 Q heads / 2 KV heads / 64 head_dim; 2B uses 12 Q heads / 2 KV heads / 128 head_dim.
- q/k/v projection widths are explicit: q width `num_attention_heads * head_dim`, k/v width `num_key_value_heads * head_dim`, output width `hidden_size`.
- RoPE is applied to q/k before `past_key_values.update`, so cached keys are post-RoPE.
- Cache tensors before GQA repeat have shape `[batch, num_key_value_heads, cache_seq, head_dim]`; eager fallback repeats to `[batch, num_attention_heads, cache_seq, head_dim]` for attention math.
- Full causal masks for inspected configs. Sliding-window Qwen2 source path exists but configs set `use_sliding_window=false` and `sliding_window=null`.
- Softmax is computed in fp32 in eager fallback, cast back to query dtype, then dropout is applied only during training.

## 7. Position encoding and custom math

Vision learned positions are static for a fixed image size and patch size. For 448/14 the sequence length is 1024 before hidden-stride packing.

Qwen2 default RoPE can be reproduced as:

```python
def qwen2_default_rope(position_ids, head_dim, theta=1_000_000.0):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = position_ids[:, :, None].float() * inv[None, None, :]
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)

def apply_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Vision hard-softmax variant, source-supported but not used by inspected inference configs:

```python
def hard_softmax(logits, dim):
    y_soft = softmax(logits, dim)
    index = argmax(y_soft, dim, keepdim=True)
    y_hard = zeros_like(logits).scatter(dim, index, 1.0)
    return y_hard - detach(y_soft) + y_soft
```

Precompute opportunities:

- Vision position ids and learned position lookup are static per image size.
- RoPE inverse frequencies are static per text config; cos/sin depend on runtime `position_ids` and cache length.
- Visual indicator embeddings are static per model and can be cached.

## 8. Preprocessing and input packing

Image processor contract from converted checkpoints:

- `crop_to_patches=true`, `min_patches=1`, `max_patches=12`, `use_covering_area_grid=true`.
- Resize/crop target size 448x448, BICUBIC resample, RGB conversion, rescale factor `1/255`, OpenAI CLIP mean/std.
- Source output is NCHW tensors. The processor returns `pixel_values` and an internal `grids` value; `Ovis2Processor` pops `grids` before returning model inputs and uses it only to expand text.

Crop-to-patches behavior:

- For each image, choose `(num_rows, num_columns)` grid by minimal covering grid or closest aspect ratio.
- Resize the original image to `num_columns * 448` by `num_rows * 448`, slice tiles, and if more than one tile exists, insert a thumbnail resized to 448x448 before the tiles.
- Final `pixel_values` is flattened across all processed images/tiles. The decoder does not receive grid metadata.

Text placeholder expansion:

```text
<image>
  -> <IMG_START> + 256 * <IMG_ATOM> + <IMG_GRID>
     + optional tiled rows/cols each with 256 * <IMG_ATOM>
     + <IMG_END>
```

For a 1x1 grid this creates 256 `<IMG_ATOM>` tokens. For a multi-tile grid it creates `(1 + rows * cols) * 256` atom tokens, with `<IMG_COL>` between columns and `<IMG_ROW>` between rows.

Special token ids from snapshots:

| Token | Id | Runtime use |
| --- | ---: | --- |
| `<IMG_ATOM>` | 151665 | replaced by image features |
| `<IMG_START>` | 151666 | replaced by visual indicator embedding 0 |
| `<IMG_GRID>` | 151667 | replaced by visual indicator embedding 1 |
| `<IMG_COL>` | 151668 | replaced by visual indicator embedding 2 |
| `<IMG_ROW>` | 151669 | replaced by visual indicator embedding 3 |
| `<IMG_END>` | 151670 | replaced by visual indicator embedding 4 |

GPU/runtime first target should accept already-tokenized `input_ids`, `attention_mask`, optional `position_ids`, and preprocessed `pixel_values`; CPU image tiling/token expansion can stay outside the compiled graph initially.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d to Linear

Source pattern:

```text
Conv2d(C -> D, kernel=P, stride=P, padding=valid) -> flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten_NCHW(PxP non-overlap) -> GEMM([C*P*P] x D) -> reshape [B, H/P*W/P, D]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == valid` / 0
- `dilation == 1`
- `groups == 1`
- input height/width divisible by patch size
- preserve PyTorch NCHW flatten order

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias  # absent in current source
```

Failure cases: dynamic image sizes without divisible guards, NHWC layout rewrite without matching window flatten order, non-square future patch configs.

Parity test sketch: compare Conv2d+flatten+transpose against window flatten + GEMM for fp32 and bf16/fp16 weights at 448/14.

### Rewrite: vision tokenizer projection as GEMM

Source pattern:

```text
softmax(head_norm(head_linear(x))) -> cat zero indicator columns -> matmul(visual_probs, visual_embeddings_table.weight)
```

Replacement:

```text
Softmax visual logits -> GEMM(visual_vocab) with appended zero columns or sliced weight
```

Preconditions:

- `tokenize_function == "softmax"`
- indicator columns are exactly zero before matmul
- visual embedding table shape `[visual_vocab, text_hidden]`

Weight transform: use `visual_embeddings_table.weight[:visual_vocab - num_indicator_tokens]` for image atoms and a separate embedding lookup for indicator ids.

Failure cases: `st_argmax` or `gumbel_argmax`, changed indicator count, nonzero padded columns.

### Rewrite: hidden_stride pack

Source pattern:

```text
seq [N, L, D] -> square grid -> pad -> reshape -> permute -> reshape [N, L/stride^2, D*stride^2]
```

Replacement:

```text
2D block pack over visual token grid
```

Preconditions:

- `L` is a perfect square before padding.
- `hidden_stride` positive integer.
- Padding uses zero values before packing.
- Consumer is the following `head_linear`.

Failure cases: non-square visual sequence, layout-translated sequence with different row-major order.

### Rewrite: multimodal embedding stitch to indexed copy

Source pattern:

```text
inputs_embeds.masked_scatter(expand(input_ids == image_token_id), image_features)
for each indicator id: inputs_embeds[mask] = indicator_feature
```

Replacement:

```text
TokenEmbedding -> IndexedWrite(atom_positions, image_features.flatten(0,1)) -> IndexedWrite(indicator_positions, indicator_features)
```

Preconditions:

- Atom placeholder count equals flattened image feature count.
- Processor has expanded placeholders in source order.
- Indicator id order matches config `visual_indicator_token_ids`.

Failure cases: `inputs_embeds` supplied without `input_ids`, malformed prompt expansion, multiple images with mismatched feature order.

### Layout candidate: NCHW vision island

Keep the initial semantic lowering as NCHW. A channel-last optimization can be guarded over the patch Conv2d plus local vision encoder only if it rewrites Conv2d input layout, flatten axes, attention input `[B, T, C]`, and hidden-stride square-grid assumptions. The embedding stitch and Qwen2 decoder are sequence-major and should be protected from image-layout translation.

## 10. Kernel fusion candidates

Highest priority:

- Qwen2 RMSNorm, RoPE, GQA FlashAttention with KV cache: required for practical prefill/decode throughput.
- Qwen2 SwiGLU MLP fusion: repeated in every decoder layer and maps to existing GEMM plus activation multiply patterns.
- Multimodal embedding stitch indexed-copy kernel: `masked_scatter` and per-token indicator assignment are graph-hostile but simple once positions are known.
- Vision tokenizer head GEMM/softmax/GEMM: large `4096 -> 65531` projection plus visual embedding matmul is distinctive and expensive.

Medium priority:

- Vision RMSNorm + noncausal attention over 1024 tokens and 8 heads.
- Patch Conv2d to GEMM rewrite for non-overlap patches.
- Hidden-stride pack kernel for the fixed 32x32 to 16x16 visual grid.
- Last-token-only logits via `logits_to_keep`.

Lower priority:

- Full CPU/GPU preprocessing in graph. Keep image tiling and text expansion in the data pipeline first.
- `st_argmax` / `gumbel_argmax` visual tokenization. Inspected inference configs use softmax.
- Sliding-window Qwen2 attention. Source supports it, but inspected configs disable it.

## 11. Runtime staging plan

Stage 1: parse converted `ovis2` configs and load weights, rejecting `model_type="ovis"` remote-code configs for this path.

Stage 2: run Qwen2 text-only path through `inputs_embeds` or `input_ids`, with prefill logits and decode cache parity.

Stage 3: implement Ovis2 vision tokenizer standalone: patch embedding, 24-layer vision encoder, hidden-stride pack, visual vocab projection, visual embedding table matmul.

Stage 4: implement processor-compatible prefix construction outside graph: produce `input_ids`, `attention_mask`, `pixel_values`, and expected atom/indicator positions.

Stage 5: implement embedding stitch and full multimodal prefill parity.

Stage 6: decode with delegated Qwen2 KV cache, forwarding `pixel_values` only on first iteration or cache-disabled runs.

Stage 7: add optimized kernels/fusions: patch Conv2d GEMM, vision tokenizer head, indexed stitch, Qwen2 FlashAttention, last-token logits.

Initially stub: CPU image preprocessing, chat template, sampling policy, beam search, and remote-code `AIDC-AI/Ovis2-*` behavior.

## 12. Parity and validation plan

- Config loader tests: accept converted `model_type="ovis2"`; reject or route `model_type="ovis"` with top-level `auto_map`.
- Processor tests: given synthetic grid values, verify `<image>` expansion atom counts and indicator ordering for 1x1, 1x2, and 3x4 grids.
- Patch embedding rewrite test: Conv2d path vs WindowFlatten+GEMM at 448/14.
- Vision hidden-stride test: source reshape/pad/permute path vs custom pack kernel, including square size divisible by 2 and an artificial non-divisible square.
- Visual tokenizer head test: `head_linear -> LayerNorm -> softmax -> visual_embeddings_table` parity.
- Embedding stitch test: random image features and token ids with multiple images; verify exact position replacement and mismatch failure.
- Qwen2 one-layer parity: RMSNorm, RoPE, GQA attention with cache update, MLP.
- Full prefill logits parity against Transformers for 1B and 2B converted configs on one image prompt.
- Decode parity: first step with image, subsequent step without `pixel_values`, compare logits and cache sequence length.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 end-to-end `rtol=2e-2, atol=2e-2`, with stricter per-op tolerances where accumulation is fp32.

## 13. Performance probes

- CPU preprocessing throughput: crop-to-patches grid selection, resize, normalization, token expansion.
- Vision tokenizer throughput by number of processed image tiles: 1, 2, 5, 13 processed images per sample.
- Vision head breakdown: transformer encoder vs `4096 -> 65531` head vs visual embedding matmul.
- Prefill-only latency/throughput with image prefix lengths: text only, 256 atoms, 768 atoms, 3328 atoms.
- Decode tokens/sec with cached image prefix, batch size sweep.
- KV cache memory by model size and sequence length.
- Attention backend comparison for Qwen2: eager, SDPA, FlashAttention equivalent.
- Indexed stitch cost versus precomputed `inputs_embeds`.
- Last-token logits versus full-sequence logits.
- Weight loading/projection memory: visual embedding table `[65536, hidden]`, LM head/token embedding tied alias.

## 14. Skip/defer list

- Training and loss parity except basic label-shift smoke tests.
- Gradient checkpointing.
- Beam search and advanced generation controllers.
- Original `AIDC-AI/Ovis2-*` remote-code `model_type="ovis"` implementation.
- `gumbel_argmax` visual tokenization for deterministic inference.
- Sliding-window Qwen2 attention unless a converted config enables it.
- Multi-GPU tensor parallel plans.
- Full image preprocessing inside DinoML runtime.
- Quantized/packed weight ingestion; no Ovis2-specific packed format was found in the in-library source.

## 15. Final implementation checklist

- [ ] Parse `Ovis2Config`, `Ovis2VisionConfig`, and nested Qwen2Config fields.
- [ ] Reject or separately route `model_type="ovis"` remote-code configs.
- [ ] Load tied token embedding / LM head as one logical parameter.
- [ ] Implement Qwen2 text-only prefill/decode with GQA RoPE cache.
- [ ] Implement Ovis2 NCHW patch embedding and learned vision positions.
- [ ] Implement Ovis2 vision RMSNorm, noncausal MHA, and SwiGLU blocks.
- [ ] Implement hidden-stride pack for visual tokens.
- [ ] Implement visual tokenizer head: Linear, LayerNorm, softmax.
- [ ] Implement visual embedding table matmul and indicator embedding lookup.
- [ ] Implement processor-compatible `<image>` expansion checks.
- [ ] Implement indexed embedding stitch for `<IMG_ATOM>` and indicator ids.
- [ ] Add patch Conv2d-to-GEMM guarded rewrite.
- [ ] Add visual tokenizer projection rewrite/fusion.
- [ ] Add full multimodal prefill parity test for 1B/2B converted configs.
- [ ] Add decode parity with image forwarded only on first generation step.
- [ ] Benchmark preprocessing, vision tokenizer, prefill, decode, and cache memory separately.
