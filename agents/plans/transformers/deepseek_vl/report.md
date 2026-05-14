# Transformers Audit: `deepseek_vl`

## 1. Source basis

```text
Transformers commit/version:
- Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
- Representative configs report transformers_version 4.53.0.dev0

Model id:
- Primary: deepseek-community/deepseek-vl-1.3b-chat
- Variation trap inspected: deepseek-community/deepseek-vl-7b-chat

Config source:
- agents/plans/transformers/deepseek_vl/_sources/deepseek-community__deepseek-vl-1.3b-chat__config.json
- agents/plans/transformers/deepseek_vl/_sources/deepseek-community__deepseek-vl-1.3b-chat__preprocessor_config.json
- agents/plans/transformers/deepseek_vl/_sources/deepseek-community__deepseek-vl-1.3b-chat__processor_config.json
- agents/plans/transformers/deepseek_vl/_sources/deepseek-community__deepseek-vl-1.3b-chat__tokenizer_config.json
- agents/plans/transformers/deepseek_vl/_sources/deepseek-community__deepseek-vl-1.3b-chat__generation_config.json
- Matching files for deepseek-community/deepseek-vl-7b-chat, used only to identify the hybrid family boundary.

Source files inspected:
- X:/H/transformers/src/transformers/models/deepseek_vl/configuration_deepseek_vl.py
- X:/H/transformers/src/transformers/models/deepseek_vl/modeling_deepseek_vl.py
- X:/H/transformers/src/transformers/models/deepseek_vl/processing_deepseek_vl.py
- X:/H/transformers/src/transformers/models/deepseek_vl/image_processing_deepseek_vl.py
- X:/H/transformers/src/transformers/models/deepseek_vl/image_processing_pil_deepseek_vl.py
- X:/H/transformers/src/transformers/models/deepseek_vl/modular_deepseek_vl.py
- Supporting backbone source: siglip/modeling_siglip.py, siglip/configuration_siglip.py, llama/modeling_llama.py, llama/configuration_llama.py
- Boundary check only: deepseek_vl_hybrid/modeling_deepseek_vl_hybrid.py, image_processing_deepseek_vl_hybrid.py, processing_deepseek_vl_hybrid.py

Any missing files or assumptions:
- No tests or imports were run by request.
- The primary audited source basis is plain model_type "deepseek_vl". The public 7B chat config is model_type "deepseek_vl_hybrid"; that requires the separate deepseek_vl_hybrid implementation and is gated below.
- HF config files were accessible. No safetensors metadata was fetched.
```

Links:
- [deepseek-community/deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-community/deepseek-vl-1.3b-chat)
- [deepseek-community/deepseek-vl-7b-chat](https://huggingface.co/deepseek-community/deepseek-vl-7b-chat)

## 2. High-level architecture

Architecture type: multimodal projector + LLM.

Plain `deepseek_vl` stages:

```text
CPU image/text preprocessing
  -> SigLIP vision encoder over NCHW pixel_values
  -> DeepseekVLAligner Linear(vision_hidden -> text_hidden) + GELU + Linear(text_hidden -> text_hidden)
  -> reshape image sequence and masked_scatter into text token embeddings at image_token_id positions
  -> Llama decoder prefill/decode with RoPE and self-attention KV cache
  -> tied/aliasable LM head logits and sampling
```

Stage decomposition:

| Stage | Owner | Cacheability | Notes |
|---|---|---:|---|
| Text prompt expansion | Processor/CPU | No | Replaces each tokenizer `image_token` string with 576 repeated placeholders before tokenization. |
| Image resize/pad/normalize | Processor/CPU or preprocessing graph | Recomputable | Emits NCHW `pixel_values`; no tiling for plain `deepseek_vl`. |
| Vision encoder | SigLIP vision model | Yes | Independent image stage; output sequence length is 576 for 384/16 patches when `vision_use_head=false`. |
| Aligner/projector | DeepSeek-VL | Yes | Two dense projections with GELU map 1024-d vision tokens to 2048-d text tokens for 1.3B. |
| Embedding stitch | DeepSeek-VL model forward | No | `masked_scatter` requires placeholder count exactly equals flattened image features. |
| Text prefill | Llama decoder | KV cache | Prefill includes image embeddings only in first generation iteration when cache is used. |
| Decode | Llama decoder | KV cache | Later decode steps omit `pixel_values` when cache is enabled. |

Validation can be split into processor ABI, vision encoder output parity, aligner parity, stitch parity, text prefill logits, and decode token parity.

## 3. Important config dimensions

Primary 1.3B config:

| Field | Value | Source |
|---|---:|---|
| `model_type` | `deepseek_vl` | config |
| `architectures` | `DeepseekVLForConditionalGeneration` | config |
| dtype | `float16` | config |
| `image_token_id` | 100015 | config/tokenizer |
| `num_image_tokens` | 576 | processor_config |
| text model | `llama` | config |
| text hidden size | 2048 | config |
| text layers | 24 | config |
| text attention heads | 16 | config |
| text KV heads | 16 | config |
| text head dim | 128 | config |
| text MLP intermediate | 5632 | config |
| vocab size | 102400 | config |
| max positions | 16384 | config/tokenizer |
| RoPE theta | 10000.0 | config |
| text activation | `silu` | config |
| text norm eps | `1e-6` RMSNorm | config |
| attention bias / MLP bias | false / false | config |
| cache support | `use_cache=true` | config |
| vision model | `siglip_vision_model` | config |
| vision hidden size | 1024 | config |
| vision layers | 24 | config |
| vision heads | 16 | config |
| vision MLP intermediate | 4096 | config |
| image size / patch size | 384 / 16 | config |
| vision sequence length | 576 | inferred from 24 x 24 patches |
| vision head | `vision_use_head=false` | config |
| image mean/std | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` | preprocessor_config |
| resize/pad | longest side to 384, square pad | source + preprocessor_config |

Representative checkpoint sweep:

| Model id | model_type | Architecture | Text dims | Vision dims | Processor fields | DinoML admission |
|---|---|---|---|---|---|---|
| `deepseek-community/deepseek-vl-1.3b-chat` | `deepseek_vl` | `DeepseekVLForConditionalGeneration` | 24L, H=2048, 16 Q / 16 KV heads, I=5632 | SigLIP 24L, H=1024, 16 heads, 384/16 -> 576 patches | `pixel_values`, 576 placeholders | Primary target |
| `deepseek-community/deepseek-vl-7b-chat` | `deepseek_vl_hybrid` | `DeepseekVLHybridForConditionalGeneration` | 30L, H=4096, 32 Q / 32 KV heads, I=11008 | SigLIP low-res plus SAM high-res branch | `pixel_values`, `high_res_pixel_values`, 576 placeholders | Gate as separate family/source |

## 3a. Family variation traps

- `deepseek_vl` and `deepseek_vl_hybrid` are distinct model types in the inspected Transformers checkout. Do not load 7B chat through the plain `DeepseekVLConfig`.
- The 7B chat config has `high_res_vision_config` with SAM-like vision, relative position attention, window attention, and a second processor output `high_res_pixel_values`. That is outside plain `deepseek_vl`.
- Plain `deepseek_vl` has no tiling or crop packing. It relies on a fixed 384 square image after aspect-preserving resize and padding.
- The processor ABI is token-string based: one `<image_placeholder>` in the prompt expands to 576 repeated placeholder tokens. The model only checks total placeholder count versus flattened image features, not per-image grouping.
- `vision_use_head=false` matters. DeepSeek-VL consumes SigLIP `last_hidden_state` aligned tokenwise, not SigLIP pooled output.
- Placeholder ID must remain 100015 and tokenizer extra special token content must remain `<image_placeholder>`.
- Text config uses full MHA for inspected plain and hybrid checkpoints (`num_key_value_heads == num_attention_heads`), but DinoML should still parse the Llama config fields rather than assume that family-wide.
- Llama `head_dim` is explicit in configs; do not infer projection widths from `hidden_size` alone.
- `logits_to_keep` slices logits in `DeepseekVLForConditionalGeneration.forward`; last-token-only logits can be a graph/runtime optimization.
- Tied-weight aliasing: `_tied_weights_keys` maps `lm_head.weight` to `model.language_model.embed_tokens.weight`; lowerers must preserve a single logical parameter when weights are tied.
- Layout guard: vision source is NCHW through Conv2d, flatten, transpose to `[B, S, C]`. Any NHWC optimization must rewrite Conv2d, LayerNorm/attention axes, flatten order, and position addition explicitly.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B, 3, H, W]`.
- Resize/pad/rescale/normalize in preprocessing; may start outside compiled runtime.
- Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=16, stride=16, padding=valid)` for 1.3B.
- Flatten spatial patches `patch_embeds.flatten(2).transpose(1, 2)` to `[B, 576, 1024]`.
- Reshape image embeddings `[B_img, 576, H_text] -> [-1, H_text]`.
- Boolean equality mask for `input_ids == image_token_id`.
- Unsqueeze/expand mask to embedding width.
- `masked_scatter` or equivalent indexed copy into `[B, T, H_text]` embeddings.
- Slice logits by integer or tensor `logits_to_keep`.

Neural network primitives:

- Embedding lookups for tokens, position IDs, and SigLIP patch positions.
- Linear layers:
  - Aligner 1.3B: `Linear(1024 -> 2048)` with bias, GELU, `Linear(2048 -> 2048)` with bias.
  - Llama attention: q/k/v/o projections with no bias, widths `2048 -> 2048`.
  - Llama MLP: gate/up `2048 -> 5632`, down `5632 -> 2048`, no bias, SiLU-gated multiply.
  - LM head `Linear(2048 -> 102400, bias=False)`, tied to embeddings when weight tying is active.
  - SigLIP vision attention/MLP: q/k/v/o `1024 -> 1024` with bias; MLP `1024 -> 4096 -> 1024`.
- RMSNorm for Llama text (`eps=1e-6`).
- LayerNorm for SigLIP vision (`eps=1e-6`).
- GELU in SigLIP/aligner, SiLU in Llama MLP.
- Residual adds.

Attention primitives:

- SigLIP vision encoder: noncausal dense MHA, 16 heads, head_dim 64, optional SDPA/eager backend, no cache.
- Llama decoder: causal self-attention, 16 Q heads, 16 KV heads, head_dim 128, RoPE before cache update, KV cache support.
- FlashAttention/SDPA compatibility is advertised by the wrapper; parity must preserve mask and RoPE/cache order.

Position/rotary ops:

- SigLIP absolute patch position embedding length 576.
- SigLIP optional bicubic position interpolation exists in source but default DeepSeek-VL call does not pass `interpolate_pos_encoding=True`.
- Llama RoPE with theta 10000.0 and explicit `head_dim=128`.

Generation/cache ops:

- DynamicCache initialization when `use_cache` and no cache is passed.
- Cache update per Llama layer stores key/value after RoPE.
- `prepare_inputs_for_generation` passes `pixel_values` only on first iteration or when `use_cache=False`.

Preprocessing-coupled ops:

- Tokenizer prompt replacement: `prompt.replace(image_token, image_token * 576)`.
- Aspect-preserving resize: longest side becomes target size, shorter side rounded and floored by `min_size=14`.
- Square pad with background color from image mean, then rescale and normalize.

Scatter/indexed update ops for multimodal embedding stitch:

- Must implement a shape-checked `masked_scatter` equivalent. The source verifies `inputs_embeds[special_image_mask].numel() == image_features.numel()`.
- A production lowering should prefer an indexed-copy ABI with explicit placeholder offsets to avoid compiling a broad boolean masked-scatter path first.

Quantized/packed weight metadata ops:

- None in inspected source/configs. Any quantized checkpoint support would be a loader/provider layer, not model-source behavior.

## 5. Layer/block breakdown

Plain DeepSeek-VL image branch:

```text
pixel_values: [B_img, 3, 384, 384]
patch = Conv2d(3 -> 1024, kernel=16, stride=16)(pixel_values)
patch = flatten HW then transpose -> [B_img, 576, 1024]
patch = patch + learned_position[0:576]
repeat 24 SigLIP encoder layers:
  y = LayerNorm(x)
  q,k,v = Linear(1024 -> 1024, bias=True)(y), split to 16 heads x 64
  y = dense noncausal attention(q,k,v)
  x = x + Linear(1024 -> 1024, bias=True)(y)
  y = LayerNorm(x)
  y = Linear(1024 -> 4096, bias=True) -> GELU -> Linear(4096 -> 1024, bias=True)
  x = x + y
x = LayerNorm(x)
```

Plain DeepSeek-VL aligner:

```text
vision_tokens: [B_img, 576, 1024]
image_embeds = Linear(1024 -> H_text, bias=True)
image_embeds = GELU(image_embeds)
image_embeds = Linear(H_text -> H_text, bias=True)
```

For 1.3B, `H_text=2048`; for the hybrid 7B trap, plain aligner does not apply because the source uses `DeepseekVLHybridAligner`.

Embedding stitch:

```text
inputs_embeds = token_embedding(input_ids)  # [B_text, T, H_text]
image_features = image_embeds.reshape(-1, H_text)
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
assert selected_scalar_count == image_features.numel()
inputs_embeds = masked_scatter(inputs_embeds, mask, image_features)
```

Llama decoder block, repeated 24 times for 1.3B:

```text
residual = x
x = RMSNorm(x)
q = Linear(2048 -> 16*128, bias=False)(x)
k = Linear(2048 -> 16*128, bias=False)(x)
v = Linear(2048 -> 16*128, bias=False)(x)
q,k = RoPE(q,k, position_ids/cache_position)
k,v = cache.update(k,v, layer_idx) if cache enabled
x_attn = causal_attention(q,k,v, mask)
x = residual + Linear(2048 -> 2048, bias=False)(x_attn)
residual = x
x = RMSNorm(x)
x_mlp = down_proj(SiLU(gate_proj(x)) * up_proj(x))
x = residual + x_mlp
```

Final:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch tokens.
- MHA, not GQA/MQA, for inspected SigLIP config: 16 heads, head_dim 64.
- Query length equals key/value length: 576 for 384 x 384 images with patch 16.
- No KV cache.
- Source supports eager attention or registered attention interface. Eager path applies `matmul(q, k.T) * scale`, adds mask if present, softmax in fp32, dropout during training, then `matmul(weights, v)`.
- DeepSeek-VL vision call does not provide an attention mask in the plain path.

Text attention:

- Causal autoregressive self-attention.
- Inspected 1.3B is MHA: 16 Q heads, 16 KV heads, head_dim 128. The Llama source still contains repeat-KV logic for GQA/MQA when configs differ.
- RoPE applies to q/k before cache update.
- Cache shape before repeat expansion, per layer: key/value `[B, num_key_value_heads, S_cache, head_dim]`; for 1.3B `[B, 16, S, 128]`.
- For GQA variants, source repeats cached/updated KV to Q-head count for attention using `repeat_kv`.
- Masking is causal plus any attention mask from generation utilities.
- Source advertises FlashAttention and SDPA support. DinoML can start with dense causal attention and add optimized prefill/decode kernels behind strict mask/cache guards.

Multimodal generation cache distinction:

- Vision encoder/projector outputs are independently cacheable image features but are not KV cache.
- During generation with cache, `pixel_values` are consumed on the first iteration and omitted later; their effect persists through text self-attention KV cache.

## 7. Position encoding and custom math

Llama RoPE, source-equivalent sketch:

```python
def apply_llama_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Precompute candidates:

- SigLIP absolute position IDs and embeddings are constant for fixed 384/16.
- Llama inverse frequencies/cos/sin can be precomputed up to `max_position_embeddings=16384` for fixed RoPE theta, or generated for dynamic cache positions.
- SigLIP bicubic position interpolation can be deferred for first integration because the primary processor fixes 384 and the default call does not request interpolation.

Custom preprocessing math:

```python
def deepseek_vl_resize_shape(height, width, target=384, min_size=14):
    scale = target / max(height, width)
    return max(round(height * scale), min_size), max(round(width * scale), min_size)
```

Padding is centered to a square with background color from image mean. For the downloaded 1.3B preprocessor config, mean/std are both `[0.5, 0.5, 0.5]`, so the background before rescale/normalize is `[127,127,127]`.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Text must be a string or list of strings.
- Processor expands every occurrence of `<image_placeholder>` to 576 repeated placeholder tokens before tokenization.
- Tokenizer class is `LlamaTokenizerFast`; `add_bos_token=true`, `add_eos_token=false`, `model_max_length=16384`, pad token is EOS.
- Image processor accepts images, converts to tensor through the backend, groups by shape for batched resizing, resizes, pads, rescales, normalizes, and returns `pixel_values`.

Plain image ABI:

```text
input images -> pixel_values: [num_images, 3, 384, 384] after resize/pad
input_ids: [B, T] containing exactly num_images * 576 placeholder token positions
attention_mask: tokenizer output
position_ids: optional; otherwise Llama defaults apply
```

There is no image tiling/packing in plain `deepseek_vl`. The only packing is placeholder expansion and embedding scatter into the text stream.

GPU/runtime work:

- Vision encoder can run as a separate compiled subgraph returning `[B_img, 576, vision_hidden]`.
- Aligner can run with the vision encoder or as a cacheable projector subgraph.
- Stitching into text embeddings needs a deterministic placeholder-offset ABI. A boolean mask implementation is faithful, but an explicit offset list is easier to lower and validate.

Hybrid boundary:

- `deepseek_vl_hybrid` processor returns both `pixel_values` and `high_res_pixel_values`.
- The high-res branch resizes to 1024, pads, normalizes with CLIP mean/std, runs SAM vision, projects/downsamps high-res maps, combines low/high-res features, then scatters 576 projected tokens. This should be a separate audit before admitting 7B.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP non-overlap Conv2d patch embed -> Linear

Source pattern:

```text
Conv2d(C=3 -> H_v, kernel=P, stride=P, padding=valid)
flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten NCHW patches [B, H/P, W/P, C*P*P]
MatMul(weight_flat.T) + bias
Reshape [B, num_patches, H_v]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid` or zero.
- `dilation == 1`.
- `groups == 1`.
- NCHW source layout or an explicit NHWC rewrite with weight permutation.
- Height and width divisible by patch size after preprocessing.

Shape equations:

- `grid_h = H // P`, `grid_w = W // P`, `S = grid_h * grid_w`.
- For primary config: `H=W=384`, `P=16`, `S=576`.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b_linear = conv.bias
```

Failure cases:

- Dynamic non-divisible image sizes.
- Position interpolation enabled without matching dynamic position rewrite.
- NHWC layout pass changes patch flatten order without a tested permutation.

Parity test sketch:

- Random NCHW `[2,3,384,384]`, compare Conv2d+flatten+transpose to WindowFlatten+Linear in fp32 and fp16 tolerance.

### Rewrite: placeholder masked_scatter -> indexed embedding copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = masked_scatter(inputs_embeds, mask.expand_as(inputs_embeds), image_features.reshape(-1))
```

Replacement:

```text
offsets = nonzero(input_ids == image_token_id) in row-major order
checked_indexed_copy(inputs_embeds, offsets, image_features)
```

Preconditions:

- Placeholder count equals `num_images * 576`.
- Row-major nonzero order matches PyTorch masked-scatter traversal.
- Image features are flattened in `[image, patch, channel]` order.
- No duplicate offsets.

Failure cases:

- Prompt has missing/extra placeholders.
- Multiple images per prompt without an explicit mapping policy. Source total-count check catches only scalar count; DinoML should additionally validate intended grouping if it exposes a stricter ABI.

Parity test sketch:

- Generate two prompts with known placeholder offsets and compare source masked-scatter to indexed copy.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])
```

Replacement:

```text
if logits_to_keep == 1: gather final hidden token -> GEMM to vocab
```

Preconditions:

- No loss computation requiring all shifted logits.
- Generation path asks for only final or selected logits.

Failure cases:

- `labels` are provided.
- `logits_to_keep` is a tensor with arbitrary positions; requires gather semantics.

### Layout guard: NCHW vision island

Source pattern:

```text
NCHW preprocessing -> Conv2d -> flatten spatial -> [B,S,C] transformer
```

Optimized candidate:

- Keep preprocessing/Conv2d NCHW, then switch to `[B,S,C]` row-major transformer tokens.
- An NHWC Conv2d pass is allowed only with explicit axis and weight rewrites.

Required guards:

- Patch flatten order unchanged.
- Position embedding index order unchanged: row-major spatial order after Conv2d.
- LayerNorm axes remain channel-last only after tokenization; hybrid has channel-first LayerNorm in the SAM neck and must be gated separately.

## 10. Kernel fusion candidates

Highest priority:

- Llama RMSNorm, q/k/v projections, RoPE, causal attention prefill/decode, and KV cache update.
- Llama SwiGLU MLP: `SiLU(gate) * up` plus down projection.
- Multimodal indexed-copy stitch, because broad `masked_scatter` is awkward and can become a bottleneck at prefill.
- Last-token-only logits for decode.

Medium priority:

- SigLIP patch Conv2d-to-GEMM rewrite for fixed 384 images.
- SigLIP LayerNorm + QKV projections and dense noncausal attention over 576 tokens.
- Aligner MLP fusion: Linear + GELU + Linear on `[B_img, 576, H]`.
- Precompute/fuse SigLIP position embedding add for fixed 576 patches.

Lower priority:

- Processor resize/pad/normalize on GPU. It is important for throughput, but can be outside first compiled model parity.
- SigLIP position interpolation, because primary config does not require dynamic image size.
- Hybrid high-res SAM branch and relative/window attention, because it is a distinct model type.

## 11. Runtime staging plan

Stage 1: Config and processor ABI

- Parse `DeepseekVLConfig`, nested SigLIP vision config, nested Llama config, processor config, and tokenizer placeholder metadata.
- Validate that model_type is exactly `deepseek_vl` for this path.
- Stub image preprocessing as precomputed `pixel_values` if needed.

Stage 2: Vision encoder parity

- Implement fixed-shape SigLIP vision encoder for `[B,3,384,384]`.
- Start with dense attention and explicit position embedding add.

Stage 3: Aligner and embedding stitch

- Implement aligner MLP.
- Add checked placeholder-offset/indexed-copy op equivalent to source `masked_scatter`.

Stage 4: Text decoder prefill

- Reuse/adapt Llama decoder support with RoPE, RMSNorm, SwiGLU, causal attention, and vocabulary projection.
- Validate multimodal prefill logits.

Stage 5: Decode with KV cache

- Ensure `pixel_values` are omitted after first generation iteration when cache is enabled.
- Validate cache tensor shapes and cache update order.

Stage 6: Optimized kernels and rewrites

- Add Conv2d patch rewrite, attention kernels, last-token logits, and MLP fusions behind guards.

Stage 7: Hybrid as separate admission

- Audit `deepseek_vl_hybrid` before accepting 7B chat. Required extras: high-res processor, SAM vision branch, relative/window attention, channel-first LayerNorm, downsample projections, and hybrid aligner.

## 12. Parity and validation plan

- Processor parity:
  - Text with one and two `<image_placeholder>` occurrences; assert tokenized placeholder count is `576 * occurrences`.
  - Image aspect ratios wide/tall/square; assert resize shape, center padding, mean/std normalization, and NCHW output.
- Patch embedding rewrite:
  - Random `[B,3,384,384]`, compare Conv2d+flatten+transpose against linearized patch path.
- SigLIP single layer:
  - Random hidden states `[B,576,1024]`, compare LayerNorm, attention, MLP block in fp32.
- Vision encoder:
  - Compare final `last_hidden_state` `[B,576,1024]` for one image.
- Aligner:
  - Compare `[B,576,1024] -> [B,576,2048]` for 1.3B.
- Stitch:
  - Compare source `masked_scatter` to DinoML indexed copy with exact placeholder offsets.
- Llama block:
  - Single-layer and after-N-layer parity with RoPE/cache disabled and enabled.
- Prefill logits:
  - End-to-end image+text prefill logits, `logits_to_keep=0` and `logits_to_keep=1`.
- Decode token:
  - One-step decode with cache after multimodal prefill; assert `pixel_values` are not required.

Recommended tolerances:

- fp32 source parity: `atol=1e-4`, `rtol=1e-4` for blocks, tighter for preprocessing/embedding integer logic.
- fp16 runtime parity: `atol=2e-2`, `rtol=2e-2` for end-to-end logits; use component-level tolerances where kernels are deterministic.

## 13. Performance probes

- Processor throughput: images/sec for resize/pad/normalize across aspect ratios.
- Vision encoder throughput: `[B,3,384,384]` batch sweep.
- Aligner throughput: `[B,576,1024] -> [B,576,2048]`.
- Stitch cost: boolean masked-scatter versus indexed-copy offsets for prompt lengths up to 16K.
- Prefill tokens/sec with and without image tokens.
- Decode tokens/sec at batch sizes 1, 4, 8 with cache.
- KV cache memory: 24 layers x 2 tensors x `[B,16,S,128]` x dtype bytes for 1.3B.
- Last-token logits GEMM time versus full-sequence logits.
- Attention backend comparison: dense eager/SDPA/FlashAttention-compatible prefill and decode.
- Patch Conv2d versus im2col/GEMM rewrite for fixed 384 images.

## 14. Skip/defer list

- Training, gradient checkpointing, dropout behavior, and loss-first training paths.
- `deepseek_vl_hybrid` / 7B chat admission until separately audited.
- SAM high-res branch, relative/window attention, high-res image processor outputs.
- SigLIP position interpolation for non-384 images.
- Beam search and generation-controller policy beyond standard cache handoff.
- Quantized weight loading/provider support, unless a specific checkpoint format requires it.
- Multi-GPU tensor parallel.
- GPU image preprocessing in first parity stage.

## 15. Final implementation checklist

- [ ] Parse `DeepseekVLConfig` and reject non-`deepseek_vl` model types in this path.
- [ ] Parse processor/tokenizer ABI: `<image_placeholder>`, `image_token_id=100015`, `num_image_tokens=576`.
- [ ] Load nested SigLIP vision weights and Llama text weights with tied LM-head alias handling.
- [ ] Implement or stub CPU image preprocessing to produce NCHW `[B,3,384,384]`.
- [ ] Implement SigLIP patch embedding, position embedding, 24-layer vision encoder, and final LayerNorm.
- [ ] Implement DeepseekVLAligner `Linear -> GELU -> Linear`.
- [ ] Implement checked placeholder-offset indexed copy equivalent to `masked_scatter`.
- [ ] Implement Llama decoder prefill with RoPE and causal mask.
- [ ] Implement Llama decode with KV cache and first-iteration image handoff.
- [ ] Add Conv2d patch-embed-to-linear rewrite with NCHW/layout guards.
- [ ] Add last-token-only logits rewrite for generation.
- [ ] Add processor, vision, aligner, stitch, prefill, and decode parity tests.
- [ ] Add performance probes for processor, vision encoder, prefill, decode, stitch, and logits.

## Gated gaps

- `deepseek_vl_hybrid` checkpoints, including the 7B chat config, are gated as a separate family despite the similar name.
- A general boolean `masked_scatter` lowering is not required if DinoML introduces a stricter checked placeholder-offset copy op.
- NHWC/channel-last optimization for the vision path is gated behind explicit axis/weight/flatten-order rewrites.
- Dynamic image sizes and SigLIP position interpolation are deferred for the fixed 384 primary path.
- Quantized/packed weight handling is not source-required for the inspected configs and should enter through loader/provider contracts only.
