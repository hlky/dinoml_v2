# Transformers Llama4 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream source URLs use the same commit under
  https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary intended targets: meta-llama/Llama-4-Scout-17B-16E,
  meta-llama/Llama-4-Scout-17B-16E-Instruct,
  meta-llama/Llama-4-Maverick-17B-128E-Instruct.
  Official Meta repos were gated from this environment, so fetched config
  snapshots come from open mirrors and are labeled below.

Config source:
  Open snapshots under agents/plans/transformers/llama4/_sources/:
    unsloth_Llama-4-Scout-17B-16E-Instruct_config.json
    unsloth_Llama-4-Scout-17B-16E-Instruct-unsloth_config.json
    unsloth_Llama-4-Maverick-17B-128E-Instruct_config.json
    mlx-community_meta-llama-Llama-4-Scout-17B-16E-6bit_config.json
    mlx-community_Llama-4-Maverick-17B-128E-Instruct-6bit_config.json
    unsloth_Llama-4-Scout-17B-16E-Instruct_preprocessor_config.json
    unsloth_Llama-4-Scout-17B-16E-Instruct_processor_config.json
    unsloth_Llama-4-Scout-17B-16E-Instruct_tokenizer_special_tokens_summary.json

Source files inspected:
  transformers/src/transformers/models/llama4/modeling_llama4.py
  transformers/src/transformers/models/llama4/configuration_llama4.py
  transformers/src/transformers/models/llama4/processing_llama4.py
  transformers/src/transformers/models/llama4/image_processing_llama4.py
  transformers/src/transformers/models/llama4/convert_llama4_weights_to_hf.py
  transformers/src/transformers/modeling_rope_utils.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/cache_utils.py
  transformers/docs/source/en/model_doc/llama4.md
  transformers/tests/models/llama4/test_modeling_llama4.py
  transformers/tests/models/llama4/test_processing_llama4.py
  transformers/tests/models/llama4/test_image_processing_llama4.py

Any missing files or assumptions:
  The in-library source is authoritative for this report. Official Meta config
  URLs returned 401 without credentials, so mirror configs are used for sizing.
  Quantized mirror metadata is not treated as native runtime behavior. DinoML
  assumptions: inference-only first, CUDA GPU target, prefill/decode generation,
  multimodal image+text support, and guarded graph rewrites instead of broad
  layout translation.
```

## 2. High-level architecture

Llama4 is an early-fusion multimodal autoregressive MoE decoder. The image path
is a standalone vision transformer that returns projected patch features. The
text path is a Llama-like causal decoder with GQA, mixed RoPE/NoPE behavior,
chunked attention masks, and MoE feed-forward layers. Unlike `mllama`, there is
no text-to-image cross-attention: image features are projected to text hidden
size and inserted directly into the token embedding stream with `masked_scatter`.

```text
CPU image preprocessing + text placeholder expansion
  -> pixel_values[N_tiles,3,336,336] and input_ids[B,T]
  -> vision patch Unfold+Linear -> vision transformer -> pixel shuffle MLP
  -> multimodal projector 4096 -> 5120
  -> masked_scatter projected image features into text embeddings at <|patch|> ids
  -> MoE causal decoder prefill/decode with mixed chunked/full masks
  -> final RMSNorm -> lm_head/logits_to_keep -> sampling
```

Stage decomposition:

- CPU/data pipeline: image fetch, resize/pad/tile, bfloat16 rescale+normalize,
  placeholder expansion, tokenization, attention-mask construction.
- Independently cacheable vision/projector stage: `pixel_values` can be encoded
  once for an image prompt. Resulting projected image token embeddings can be
  cached independently from decoder KV cache.
- Prefix construction: token embeddings plus `masked_scatter` image embeddings.
- Prefill: decoder processes the multimodal prefix and builds per-layer KV
  cache.
- Decode: `pixel_values` are not forwarded after the first generation
  iteration; cached text KV state carries the image-conditioned prefix.

## 3. Important config dimensions

Worked production-shaped Scout Instruct mirror, from
`unsloth_Llama-4-Scout-17B-16E-Instruct_config.json`:

| Field | Value | Source/provenance |
|---|---:|---|
| architecture | `Llama4ForConditionalGeneration` | config.json |
| text hidden size | 5120 | config.json |
| text layers | 48 | config.json |
| attention heads / KV heads / head dim | 40 / 8 / 128 | config.json |
| GQA repeat factor | 5 | inferred from heads/KV heads |
| vocab size | 202048 | config.json |
| max positions | 10485760 | config.json, mirror |
| text intermediate expert size | 8192 | config.json |
| dense MLP intermediate size | 16384 | config.json |
| num local experts / top-k | 16 / 1 | config.json |
| MoE layers | 48 listed | config.json |
| attention chunk size | 8192 | config.json |
| RoPE | llama3 scaling, factor 16, original max 8192, theta 500000 | config.json |
| Q/K L2 norm | enabled | config.json/source |
| attention bias | false | config/source default |
| cache implementation | hybrid | config.json |
| image size / patch size | 336 / 14 | vision config and preprocessor |
| vision hidden / layers / heads | 1408 / 34 / 16 | config.json |
| vision output dim | 4096 | config.json |
| pixel shuffle ratio | 0.5 | config/source |
| projector | `Linear(4096 -> 5120, bias=False)` | source/config |
| image token ids | start 200080, end 200081, patch/image index 200092 | config/source |
| dtype | bfloat16 | config.json |

Representative checkpoint/config sweep:

| Snapshot | Official? | Text H/L/A/KV | Experts | MoE pattern | Max pos | RoPE | QK norm | Vision | Notes |
|---|---|---:|---:|---|---:|---|---|---|---|
| `unsloth/Llama-4-Scout-17B-16E-Instruct` | mirror | 5120/48/40/8 | 16 top1 | 48 listed layers | 10,485,760 | llama3 factor 16 | yes | 336, 34L, H=1408 | Open mirror; `cache_implementation=hybrid`. |
| `unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth` | mirror | 5120/48/40/8 | 16 top1 | 48 listed layers | 10,485,760 | llama3 factor 16 | yes | same | Similar open mirror, no explicit pad id in top-level summary. |
| `mlx-community/meta-llama-Llama-4-Scout-17B-16E-6bit` | mirror, quantized | 5120/48/40/8 | 16 top1 | omitted, source defaults would matter | 262,144 | llama3 factor 8 | yes | same | Quantized mirror; source MoE/layer defaults must be re-canonicalized before admission. |
| `unsloth/Llama-4-Maverick-17B-128E-Instruct` | mirror | 5120/48/40/8 | 128 top1 | 24 odd layers | 1,048,576 | theta 500000 default | no | same | `interleave_moe_layer_step=2`; `cache_implementation=hybrid`. |
| `mlx-community/Llama-4-Maverick-17B-128E-Instruct-6bit` | mirror, quantized | 5120/48/40/8 | 128 top1 | 24 odd layers | 1,048,576 | theta 500000 default | no | same | `attn_temperature_tuning` appears as integer `4`; treat as truthy but audit mirror drift. |

There is no small public Llama4 checkpoint in the inspected source. The local
tests are slow integration tests against gated Scout, plus image/processor unit
tests with tiny processor settings; they are useful for preprocessing behavior,
not model operator coverage.

## 3a. Family variation traps

- Official Meta repos are gated; open mirrors can contain quantization,
  compressor, or stale historical fields. DinoML should canonicalize configs
  through the in-library `Llama4Config` behavior or explicitly reject unclear
  mirror-only fields.
- Llama4 is not Llama, Llava, or mllama. It uses early fusion by token embedding
  replacement, not cross-attention.
- The text decoder is MoE. Scout uses 16 local experts; Maverick uses 128. The
  current source top-k is configurable but sampled configs use top-1.
- MoE layer placement differs: Scout mirror lists every layer; Maverick lists
  odd layers only. Do not assume every layer is sparse.
- The source has two MLP widths: sparse expert intermediate `8192`, dense/shared
  MLP intermediate `16384`.
- GQA is mandatory: 40 Q heads, 8 KV heads, head dim 128.
- Source `no_rope_layers` naming is confusing. In `Llama4TextAttention`,
  `self.use_rope = config.no_rope_layers[layer_idx]`, while `layer_types` maps
  truthy entries to `"chunked_attention"` and false entries to
  `"full_attention"`. Report/runtime logic should follow the effective source
  fields, not the name.
- Chunked attention and RoPE are coupled by source defaults/mirrors: most layers
  use the chunked mask, while every fourth layer in sampled mirrors is full
  attention without RoPE and can receive attention-temperature scaling.
- `attn_temperature_tuning` appears as `True`, `None`, and `4` in mirrors. The
  source treats it as a boolean guard and uses `attn_scale`/`floor_scale` for
  math; DinoML should normalize to enabled/disabled plus explicit parameters.
- Text RoPE uses complex tensors (`torch.polar`, `view_as_complex`) rather than
  Llama's cos/sin pair helper.
- Vision RoPE is 2D complex RoPE over patch coordinates plus CLS token, not the
  text RoPE table.
- Vision preprocessing and modeling consume NCHW tiles. NHWC/channel-last is
  only a local optimization candidate around patch extraction and GEMM.
- The processor expands one user `<|image|>` placeholder into many special
  tokens, including `<|patch|>` tokens. The model replaces token id
  `image_token_index` (200092 in configs) with projected image features.
- `processor_config.json` may name `image_token="<|image|>"`, while
  `Llama4Config.image_token_index` points at `<|patch|>` in the tokenizer
  snapshot. The model uses the numeric config field, so placeholder expansion
  must be tested end to end.
- `image_features.numel()` must equal selected placeholder embedding slots. A
  mismatch is a hard error.
- Weight conversion has nonstandard expert packing: expert `gate_up_proj` is
  stored as `[num_experts, hidden, 2*expert_dim]` and `down_proj` as
  `[num_experts, expert_dim, hidden]`, used through `bmm`, not `nn.Linear`.
- The source advertises SDPA/Flex support but sets `_supports_flash_attn=False`
  in the base model, while docs mention FlashAttention examples. Treat source
  support flags as the audit basis.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather for text tokens: `input_ids[B,T] -> [B,T,5120]`.
- Image placeholder mask: compare `input_ids == image_token_index`, expand to
  `[B,T,5120]`, validate selected element count.
- `masked_scatter` or indexed copy into embedding tensor.
- View/reshape/flatten/transpose/permute/contiguous for attention head splits:
  Q `[B,T,40,128] -> [B,40,T,128]`, K/V `[B,T,8,128] -> [B,8,T,128]`.
- `repeat_kv` fallback from `[B,8,T,128]` to `[B,40,T,128]` if backend lacks GQA.
- Causal full mask and chunked causal mask construction with left-padding
  awareness and `attention_chunk_size=8192`.
- Vision NCHW Unfold, `permute(0,2,1)`, concat CLS token, reshape between
  `[N_tiles,P,Hv]`, `[N_tiles,1,P,Hv]`, and `[N_tiles,T,Hv]`.
- Pixel shuffle reshapes and permutes: `[N,576,1408] -> [N,144,5632]` for
  336/14 patches with ratio 0.5.

Neural network primitives:

- Bias-free text linear projections:
  - Q `5120 -> 5120`, K/V `5120 -> 1024`, O `5120 -> 5120`.
  - Dense/shared MLP gate/up `5120 -> 16384`, down `16384 -> 5120`.
  - LM head `5120 -> 202048`, bias false.
- Optional text attention bias exists in config, sampled native configs use
  false.
- RMSNorm over hidden size with fp32 accumulation and learned scale.
- L2Norm over Q/K head dimension when enabled, fp32 accumulation, no scale.
- SiLU and elementwise multiply for text MLPs.
- Vision Unfold+Linear patch embedding: `3*14*14=588 -> 1408`, bias false.
- Vision LayerNorm, Linear with bias for Q/K/V/O and MLP:
  - Q/K/V/O `1408 -> 1408`.
  - MLP `1408 -> 5632 -> 1408`, GELU.
- Vision adapter MLP after pixel shuffle:
  - `5632 -> 4096`, GELU, dropout, `4096 -> 4096`, GELU.
- Multimodal projector: `4096 -> 5120`, bias false.

Attention primitives:

- Text causal GQA with mixed full/chunked masks.
- Vision noncausal MHA over patch tokens with 16 heads, head dim 88.
- Eager text attention does not upcast attention weights to fp32 before
  softmax; preserve this difference from standard Llama reports.
- SDPA/Flex dispatch through `ALL_ATTENTION_FUNCTIONS`; FlashAttention source
  flag is false for this model at this commit.

MoE/routing ops:

- Router linear `5120 -> num_local_experts`, bias false.
- `topk(router_logits, k=top_k, dim=1)`.
- `full_like(-inf)` and `scatter_` to keep only selected logits.
- `sigmoid(router_scores.float()).to(dtype)`.
- Token repeat by expert count/top-k, score multiply, expert `bmm`, sum over
  top-k, and add shared expert output.
- Expert packed BMMs:
  - `hidden.view(E, -1, 5120) bmm gate_up_proj[E,5120,16384]`.
  - chunk gate/up each `8192`, apply SiLU(gate)*up.
  - `bmm(..., down_proj[E,8192,5120])`.

Position/rotary ops:

- Text complex RoPE with `torch.polar(ones, freqs)`.
- Llama3 RoPE scaling for Scout-like configs.
- NoPE layers skip RoPE and may apply query temperature scaling.
- Vision 2D complex RoPE over patch x/y coordinates and CLS zeroing.

Generation/cache ops:

- `DynamicCache(config=...)` when `use_cache=True` and no cache is passed.
- Hybrid cache/layer types: chunked layers behave like sliding-window storage
  from cache class perspective; full layers retain full KV.
- KV cache update after RoPE/QK norm/temperature scaling ordering.
- `prepare_inputs_for_generation` forwards `pixel_values` only first iteration
  or when cache is disabled.
- `logits_to_keep` slices hidden states before LM head.

Preprocessing-coupled ops:

- Image resize without distortion, pad to supported canvas, split to tiles, add
  global tile if more than one tile.
- Rescale+normalize in bfloat16 when both enabled.
- Placeholder expansion using `<|image_start|>`, tile separators, `<|image|>`,
  repeated `<|patch|>`, `<|image_end|>`.
- Left padding by processor default.

Distributed/tensor-parallel ops:

- Config declares TP plans for text Q/K/V/O, dense/shared MLPs, packed experts,
  router, vision patch embedding, and projector. Single-GPU parity can defer TP,
  but real Llama4 sizes require sharded GEMM and grouped expert planning.

## 5. Layer/block breakdown

Vision preprocessing/model block:

```text
pixel_values: [N_tiles, 3, 336, 336]
x = Unfold(kernel=14,stride=14) -> [N_tiles, 588, 576]
x = permute -> Linear(588 -> 1408, bias=False) -> [N_tiles,576,1408]
x = cat learned CLS -> [N_tiles,577,1408]
x = add learned positional_embedding_vlm[577,1408]
x = LayerNorm(1408)
freqs = 2D complex vision RoPE table [577,44]
repeat 34 layers:
  residual = x
  x = LayerNorm(x)
  q,k,v = Linear(1408 -> 1408, bias=True)
  q,k = vision_complex_rope(q,k,freqs)
  x = noncausal MHA(q,k,v)
  x = residual + Linear(1408 -> 1408, bias=True)(x)
  residual = x
  x = LayerNorm(x)
  x = Linear(1408 -> 5632, bias=True) -> GELU -> Linear(5632 -> 1408, bias=True)
  x = residual + x
x = LayerNorm(x)
x = drop CLS -> [N_tiles,576,1408]
x = pixel_shuffle(ratio=0.5) -> [N_tiles,144,5632]
x = Linear(5632 -> 4096, bias=False) -> GELU -> Dropout(0) -> Linear(4096 -> 4096, bias=False) -> GELU
```

Multimodal stitch:

```text
image_features = vision_model(pixel_values).last_hidden_state  # [N_tiles,144,4096]
flat = image_features.view(-1,4096)
projected = Linear(4096 -> 5120, bias=False)(flat)
mask = input_ids == image_token_index
inputs_embeds = token_embedding(input_ids)
inputs_embeds = masked_scatter(mask[...,None].expand_as(inputs_embeds), projected)
```

Text decoder layer, repeated 48 times:

```text
residual = x
y = RMSNorm(x)
q = Linear(5120 -> 5120, bias=attention_bias)(y).view(B,T,40,128)
k = Linear(5120 -> 1024, bias=attention_bias)(y).view(B,T,8,128)
v = Linear(5120 -> 1024, bias=attention_bias)(y).view(B,T,8,128)
if use_rope[layer]:
  q,k = complex_rope(q,k,freqs_cis)
if use_qk_norm and use_rope[layer]:
  q = L2Norm(q, axis=-1); k = L2Norm(k, axis=-1)
if attn_temperature_tuning and not use_rope[layer]:
  q *= log1p(floor((positions+1)/floor_scale))*attn_scale + 1
q,k,v = transpose to [B,heads,T,D]
k,v = cache.update(k,v,layer) if cache
x = residual + Linear(5120 -> 5120)(GQA(q,k,v, full_or_chunked_mask))

residual = x
y = RMSNorm(x)
if layer is MoE:
  routed = router/topk/sigmoid + packed expert BMMs
  shared = dense SwiGLU MLP(5120 -> 16384 -> 5120)
  x = residual + routed + shared
else:
  x = residual + dense SwiGLU MLP(5120 -> 16384 -> 5120)
```

LM head:

```text
x = final RMSNorm(x)
logits = Linear(5120 -> 202048, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

Text attention:

- Causal self-attention only; no cross-attention.
- GQA: Q heads 40, KV heads 8, head dim 128, repeat factor 5.
- Head shapes before backend: Q `[B,40,Q,128]`, K/V `[B,8,K,128]`.
- Full and chunked causal masks are both required. `create_chunked_causal_mask`
  uses `attention_chunk_size=8192` and accounts for left padding.
- Sampled `no_rope_layers` pattern starts `[1,1,1,0,1,1,1,0,...]`: source
  applies RoPE/QK norm on truthy entries and chunked masks for those layer
  types; false entries receive full attention and optional temperature scaling.
- Cached keys are stored after RoPE and optional Q/K L2 norm. NoPE layers cache
  temperature-scaled K? In source, temperature scaling applies to Q only, before
  transpose/cache update, so cached K is not temperature-scaled.
- Eager attention order: repeat KV, matmul, multiply by `head_dim**-0.5`, add
  mask, softmax in current dtype, dropout, matmul V. This differs from Llama's
  fp32 softmax fallback.
- FlashAttention: source base flag `_supports_flash_attn=False`; SDPA and Flex
  are supported. If adding a fused backend, require explicit parity checks for
  chunked masks and dtype softmax behavior.

Vision attention:

- Noncausal self-attention only.
- MHA, 16 heads, head dim `1408/16=88`.
- Vision RoPE is applied to Q/K before transpose.
- The `attention_mask` argument is currently ignored in the vision attention
  call path (`None` is passed to backend). Processor tiling handles padding by
  forming real padded pixels, not an attention mask.
- No KV cache for vision.

Cache layout:

- Dynamic cache stores per text layer K/V before repeat, shape
  `[B,8,seen_tokens,128]`.
- Static/hybrid cache should allocate full layers and chunked/sliding-style
  layers differently. From `cache_utils`, `"chunked_attention"` maps to sliding
  window cache classes with window `attention_chunk_size`; the mask is what
  distinguishes chunked from sliding semantics.
- Decode first iteration can include image pixel work; subsequent iterations
  must omit pixels and use only decoder cache.

## 7. Position encoding and custom math

Text complex RoPE:

```python
def llama4_text_freqs(position_ids, inv_freq, attention_scaling=1.0):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    return torch.polar(torch.ones_like(freqs), freqs) * attention_scaling

def llama4_apply_complex_rope(q, k, freqs_cis):
    q_c = view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    q = view_as_real(q_c * freqs_cis[:, :, None, :]).flatten(3).type_as(q)
    k = view_as_real(k_c * freqs_cis[:, :, None, :]).flatten(3).type_as(k)
    return q, k
```

Scout-like configs use Llama3 RoPE scaling. The Transformers implementation
standardizes legacy `rope_scaling` into `rope_parameters`, then computes
frequency-dependent scaling: wavelengths above the low-frequency threshold use
`inv_freq/factor`, wavelengths below high-frequency threshold are unchanged,
and the middle region is smoothed.

NoPE attention temperature:

```python
positions = arange(T) + past_seen_tokens
scale = log1p(floor((positions + 1) / floor_scale)) * attn_scale + 1.0
q = q * scale.view(1, T, 1, 1)
```

Vision 2D RoPE:

```python
idx = image_size // patch_size
coords = arange(idx * idx).reshape(idx * idx, 1)
coords = cat([coords, coords[:1]], dim=0)  # CLS row
coords[-1] = -2
freq_dim = hidden_size // num_heads // 2
rope_freq = 1 / rope_theta ** (arange(0, freq_dim, 2)[:freq_dim//2] / freq_dim)
freqs_x = ((coords % idx) + 1)[..., None] * rope_freq
freqs_y = ((coords // idx) + 1)[..., None] * rope_freq
freqs = cat([repeat_interleave(freqs_x,2), repeat_interleave(freqs_y,2)], -1)[..., ::2]
freqs = masked_fill(coords < 0, 0)  # CLS
freqs_ci = view_as_complex(stack([cos(freqs), sin(freqs)], -1))
```

Precomputable:

- Vision RoPE table for a fixed image/patch size.
- Text inverse frequencies for fixed RoPE settings; cos/complex tables can be
  precomputed for bounded max positions, but 10M-token Scout contexts make full
  tables impractical.

Dynamic inputs:

- Text positions depend on cache length or explicit `position_ids`.
- Dynamic RoPE variants may update inverse frequencies through
  `dynamic_rope_update`, although sampled Llama4 configs use default or llama3.

## 8. Preprocessing and input packing

Image processor contract, from source and Scout preprocessor snapshot:

- Input images are converted to RGB, resized, padded, split to NCHW tiles, then
  rescaled and normalized.
- Defaults/snapshot: `size={"height":336,"width":336}`, `max_patches=16`,
  mean/std `[0.5,0.5,0.5]`, rescale factor `1/255`, normalize enabled.
- Supported canvases are factor pairs up to `max_patches`, multiplied by tile
  size. The best fit preserves aspect ratio and minimizes padding unless
  `resize_to_max_canvas=True`.
- If an image uses more than one tile, a global resized tile is appended. Test
  coverage expects 17 tiles for `max_patches=16`.
- Output `pixel_values` is flattened over images/tiles as
  `[total_tiles,3,336,336]`; `aspect_ratios` is returned to the processor for
  prompt expansion, not to the model forward.

Processor/token packing:

- Text is required. If images are present, the processor fetches/flat-lists
  images and requires total text placeholders to equal image count.
- It computes `num_patches_per_chunk =
  (image_height/patch_size) * (image_width/patch_size) / downsample_ratio`.
  For 336, patch 14, ratio 0.5, this is `24*24/4 = 144`.
- For multi-tile images, `_prompt_split_image` emits per-tile patch runs plus
  `<|tile_x_separator|>`/`<|tile_y_separator|>`, then always emits a global
  `<|image|>` plus 144 `<|patch|>` tokens before `<|image_end|>`.
- Runtime model replacement uses `config.image_token_id`/`image_token_index`,
  which is 200092 in sampled configs and corresponds to `<|patch|>` in the
  tokenizer snapshot.
- Left padding is default in `Llama4ProcessorKwargs`.

GPU/runtime graph inputs:

- `input_ids[B,T]`, `attention_mask[B,T]`, optional `position_ids[B,T]`,
  optional `pixel_values[N_tiles,3,336,336]`, optional cache.
- First multimodal iteration must run vision/projector and scatter. Decode
  iterations should pass no `pixel_values` when cache is active.

No `cu_seqlens`, explicit modality token type IDs, image codebook, logits masks,
or separate cross-attention masks are present in this native source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Llama4 vision Unfold+Linear -> patch GEMM

Source pattern:

```text
torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
-> permute(0,2,1)
-> Linear(3*patch*patch -> vision_hidden, bias=False)
```

Replacement:

```text
NCHW non-overlap WindowFlatten -> GEMM_RCR(weight=[1408,588]) -> [N_tiles,576,1408]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- padding 0, dilation 1, groups 1.
- input height/width divisible by patch size.
- preserve PyTorch NCHW flatten order.

Layout constraints:

- Initial semantic graph should remain NCHW. A channel-last optimized patch
  extractor must locally rewrite flatten order or pre-permute weights.

Failure cases:

- Non-square patch configs, runtime image sizes not divisible by patch size, or
  processor changes that emit already-tokenized patches.

Parity test sketch:

- Compare source Unfold+Linear with lowered patch GEMM for 336 and test 20
  image sizes in fp32/bf16.

### Rewrite: projected image scatter -> indexed embedding replacement

Source pattern:

```text
inputs_embeds = embed(input_ids)
mask = input_ids == image_token_index
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None].expand_as(inputs_embeds), projected_vision_flat)
```

Replacement:

```text
EmbeddingGather -> IndexedRowsWrite(token_positions, projected_features)
```

Preconditions:

- Number of selected token rows exactly equals `projected_features.shape[0]`.
- Replacement order follows row-major order of `masked_scatter`.
- No aliasing between input and output embeddings.

Failure cases:

- Mismatched placeholder expansion or custom processor/tokenizer id drift.

Parity test sketch:

- Build prompts with one image, multiple images, and no image. Compare selected
  embedding rows before decoder.

### Rewrite: text QKV projections -> grouped GEMM

Source pattern: same normalized input feeds Q/K/V linears.

Replacement:

```text
GroupedGEMM([5120->5120, 5120->1024, 5120->1024]) -> split/head reshape
```

Preconditions:

- Same input tensor and dtype.
- Bias settings match or are handled as separate epilogues.
- GQA output widths are unequal; do not pack assuming 3*hidden.

Failure cases:

- Tensor-parallel shards or quantized formats with incompatible packing.

Parity test sketch:

- Compare Q/K/V tensors for Scout and Maverick configs before RoPE.

### Rewrite: complex RoPE -> real fused RoPE kernel

Source pattern: `view_as_complex`, complex multiply, `view_as_real`.

Replacement:

```text
ApplyComplexPairRoPE(q,k,freqs_real,freqs_imag, pair_layout=adjacent)
```

Preconditions:

- Last dim is even.
- Pair layout matches `reshape(...,-1,2)`, not Llama rotate-half layout.
- Frequency table dtype/cast order preserves fp32 angle math and output cast.

Failure cases:

- Accidentally using Llama rotate-half RoPE will be wrong for this source.

Parity test sketch:

- Random Q/K parity against `apply_rotary_emb` for head_dim 128 and vision head
  dim 88.

### Rewrite: chunked/full attention dispatch

Source pattern: one model creates both full and chunked masks, then each layer
selects `causal_mask_mapping[layer_types[i]]`.

Replacement:

```text
LayerTypedAttention(layer_type, q,k,v, cache, attention_mask)
```

Preconditions:

- `layer_types` length equals layer count.
- Chunked layers use chunk size 8192 and left-padding offsets.
- Full layers keep normal causal visibility.

Failure cases:

- FlashAttention backend that cannot respect chunked mask beyond chunk size.
- Static cache that treats chunked layers as full cache and overreads old keys.

Parity test sketch:

- Compare layer attention outputs for sequence lengths below, equal to, and
  above 8192 using an artificial small chunk size in a tiny config.

### Rewrite: MoE packed BMM -> grouped expert GEMM

Source pattern: router top-k, score multiply, packed `bmm` over expert axis.

Replacement:

```text
RouterTopK -> TokenExpertPartition -> GroupedGEMM(gate_up) -> SwiGLU -> GroupedGEMM(down) -> ScatterAdd + shared MLP
```

Preconditions:

- Top-k and expert count are known.
- Preserve sigmoid router score, not softmax.
- Preserve packed expert weight orientation:
  `gate_up_proj[E,H,2I]`, `down_proj[E,I,H]`.

Failure cases:

- Unsorted source fallback uses dense repeat rather than true token partition;
  optimized grouped GEMM must reproduce top-k ordering and score application.

Parity test sketch:

- Router/top-k parity, expert-only parity, and full MoE layer parity with top-1
  and a synthetic top-2 config.

### Rewrite: pixel shuffle MLP shape fold

Source pattern: `[N,576,1408]` pixel shuffle ratio 0.5 -> `[N,144,5632]`
followed by MLP.

Replacement:

```text
PatchGrid2x2ChannelFold -> Linear(5632 -> 4096) -> GELU -> Linear(4096 -> 4096) -> GELU
```

Preconditions:

- Number of patches is a square grid.
- `shuffle_ratio=0.5`.
- Channel dimension divisible by `ratio^-2`.

Failure cases:

- Different ratio or non-square patch grid.

Parity test sketch:

- Compare pixel shuffle output indices using a monotonic tensor, then MLP output
  with random weights.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm and L2 Q/K norm: many per block, fp32 accumulation, simple
  shape contracts.
- GQA attention with layer-typed full/chunked masks and KV cache: central to
  both Scout and Maverick prefill/decode.
- Complex RoPE fused with Q/K projection layout transform: source is not the
  standard Llama rotate-half path.
- MoE router + grouped expert GEMM: without this, Maverick/Scout sparse layers
  are prohibitively slow and memory-heavy.
- Vision patch Unfold+Linear lowering: turns NCHW patch extraction into GEMM.
- `masked_scatter` image embedding replacement: correctness-critical for
  multimodal prompts.

Medium priority:

- SwiGLU fusion for dense/shared MLP and expert activation multiply.
- Pixel shuffle + first projector GEMM shape specialization.
- Last-token-only logits through `logits_to_keep`.
- KV cache append/pack for chunked and full layer types.
- Vision MHA with 2D RoPE and noncausal SDPA.

Lower priority:

- TP/sharded expert collectives.
- Quantized mirror-specific formats and FP8/6bit loaders.
- Full processor acceleration beyond deterministic CPU pipeline.
- Training/router auxiliary loss.

## 11. Runtime staging plan

Stage 1: Parse and canonicalize `Llama4Config`, including nested text/vision
configs, legacy `rope_scaling`, `layer_types`, `no_rope_layers`, MoE placement,
and tokenizer special ids.

Stage 2: Implement text-only dense path with embeddings, RMSNorm, complex
RoPE, GQA full attention, dense MLP, final norm, and `logits_to_keep`. Use a
synthetic dense/no-MoE config first.

Stage 3: Add chunked/full layer-type masks and cache layout. Validate small
chunk sizes, then production 8192 chunk metadata.

Stage 4: Add Q/K L2 norm and NoPE temperature scaling. Validate Scout and
Maverick behavior separately.

Stage 5: Add MoE router and a slow reference grouped expert path, then replace
with provider-backed grouped GEMM.

Stage 6: Implement vision tower independently from text: preprocessing stub
with already-tiled pixels, Unfold+Linear, vision RoPE/MHA/MLP, pixel shuffle
adapter.

Stage 7: Implement multimodal projector and embedding stitch. Validate prompt
placeholder counts and image/no-image branches.

Stage 8: End-to-end multimodal prefill, then decode where pixels are only used
on the first iteration.

Stage 9: Optimize attention, MoE, patch embedding, pixel shuffle, and sharded
large-checkpoint execution.

Initially stub or defer chat-template rendering, sampling policy, official
gated weight download automation, quantized mirror kernels, and multi-GPU TP.

## 12. Parity and validation plan

- Config canonicalization tests for Scout and Maverick mirrors: effective
  expert count, MoE layer indices, layer types, RoPE params, image token id.
- Text complex RoPE parity against source for head dim 128 and several
  positions, including long positions.
- Vision 2D RoPE parity for image size 336, patch 14, head dim 88, including
  CLS row.
- RMSNorm and L2Norm parity in fp32/bf16.
- GQA attention parity for full and chunked masks. Use a tiny config with
  `attention_chunk_size=4` to make mask behavior observable.
- Cache parity for two decode steps: verify K/V shape `[B,8,T,128]` and that
  pixels are not required after first iteration.
- MoE router parity: logits, top-k indices, sigmoid router scores, sparse score
  matrix, and full layer output.
- Expert BMM parity with packed weights.
- Vision Unfold+Linear patch parity for 336 and processor-test 20 tiles.
- Pixel shuffle parity with monotonic index tensor, then random activation MLP.
- Processor parity: single image, multi-tile image, two images, placeholder
  mismatch error, left padding.
- Multimodal stitch parity: compare `inputs_embeds` after `masked_scatter`.
- Full tiny synthetic model prefill logits parity, then decode token parity for
  2-4 generated tokens.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block tests initially
  `rtol=2e-2, atol=2e-2`, tightened after fused attention/MoE kernels stabilize.

## 13. Performance probes

- CPU image preprocessing throughput by original resolution, tile count, and
  `resize_to_max_canvas`.
- Vision tower throughput for `[N_tiles,3,336,336]` tile counts 1, 4, 16, 17.
- Patch Unfold+Linear vs lowered patch GEMM.
- Pixel shuffle adapter throughput and memory traffic.
- Projector throughput over image token counts.
- Text prefill tokens/sec split by full vs chunked layers and sequence length.
- Decode tokens/sec with cache length sweep and full/chunked cache memory.
- MoE router+expert grouped GEMM throughput by batch tokens, expert count 16
  vs 128, and top-k.
- Attention backend comparison: eager vs SDPA/Flex vs future native chunked GQA.
- KV cache memory:
  `2 * layers * B * KV_heads * effective_cache_tokens(layer) * 128 * dtype_bytes`.
- LM head full logits vs `logits_to_keep=1`.
- End-to-end multimodal request latency split into preprocessing, vision,
  prefill, decode, and logits.

## 14. Skip/defer list

- Training losses, router auxiliary loss, dropout, and gradient checkpointing.
- Official gated repository access automation.
- Quantized/FP8/6bit mirror-specific kernels and loaders.
- Multi-GPU tensor parallel, expert parallel, and pipeline parallel execution.
- Beam search, speculative decoding, and continuous batching beyond cache
  shape compatibility.
- General NHWC graph-wide layout translation. Keep image input semantics NCHW
  and optimize local patch extraction only.
- FlashAttention admission until chunked mask and source support flags are
  reconciled.
- Remote-code or derivative checkpoints that use different model classes.

## 15. Final implementation checklist

- [ ] Parse nested `Llama4Config`, `Llama4TextConfig`, and `Llama4VisionConfig`.
- [ ] Canonicalize `rope_scaling`/`rope_parameters`, `no_rope_layers`, and `layer_types`.
- [ ] Validate tokenizer/config image token ids and placeholder expansion.
- [ ] Load text embeddings, LM head, text norms, Q/K/V/O, dense MLP, packed expert weights, router, vision, and projector weights.
- [ ] Implement text RMSNorm and Q/K L2Norm.
- [ ] Implement Llama4 complex text RoPE and Llama3 scaling.
- [ ] Implement NoPE query temperature scaling.
- [ ] Implement causal GQA full attention and chunked attention masks.
- [ ] Implement per-layer KV cache with full/chunked layer types.
- [ ] Implement dense/shared SwiGLU MLP.
- [ ] Implement MoE router, sigmoid top-k routing, packed expert BMM/grouped GEMM, and scatter/sum.
- [ ] Implement vision image processor parity or accept preprocessed `pixel_values`.
- [ ] Implement NCHW Unfold+Linear patch embedding and optional guarded GEMM rewrite.
- [ ] Implement vision 2D complex RoPE, MHA, LayerNorm, GELU MLP, and pixel shuffle adapter.
- [ ] Implement multimodal projector and ordered indexed embedding replacement.
- [ ] Implement `prepare_inputs_for_generation` behavior: pixels only on first cached iteration.
- [ ] Add parity tests for config, RoPE, masks/cache, MoE, vision tower, image stitch, prefill, and decode.
- [ ] Benchmark preprocessing, vision, MoE, prefill, decode, cache memory, and logits separately.
