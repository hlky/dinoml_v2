# DeepSeek-VL-Hybrid Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout X:/H/transformers
Model id: deepseek-community/deepseek-vl-7b-base and deepseek-community/deepseek-vl-7b-chat
Config source: HF config/preprocessor/processor/tokenizer pages plus local source defaults
Source files inspected: configuration_deepseek_vl_hybrid.py, modeling_deepseek_vl_hybrid.py, modular_deepseek_vl_hybrid.py, processing_deepseek_vl_hybrid.py, image_processing_deepseek_vl_hybrid.py, image_processing_pil_deepseek_vl_hybrid.py, convert_deepseek_vl_hybrid_weights_to_hf.py; composed llama/siglip/sam source files
Any missing files or assumptions: official 7B base/chat were accessible and identical architecturally; no distinct small native hybrid checkpoint was found. The 1.3B DeepSeek-VL config is non-hybrid and is only a contrast.
```

Generated files in this family say source edits should be made in `modular_deepseek_vl_hybrid.py`; `modeling_*.py`, config, processing, and image-processing files are generated from it.

Primary DinoML target: image-text-to-text autoregressive generation with one or more still images, first focusing on cached prefill/decode parity for the 7B hybrid configs.

## 2. High-level architecture

DeepSeek-VL-Hybrid is a multimodal projector plus LLaMA CausalLM:

```text
CPU image/text preprocessing
  -> low-res SigLIP vision encoder [B,3,384,384] -> [B,576,1024]
  -> high-res SAM vision encoder [B,3,1024,1024] -> projected [B,576,1024]
  -> aligner concat/project -> [B,576,4096]
  -> replace 576 image placeholder token embeddings per image
  -> LLaMA prefill/decode with KV cache
  -> lm_head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: chat template, image placeholder expansion, resize/pad/rescale/normalize, tokenizer.
- Cacheable vision stage: SigLIP low-res and SAM high-res encoders plus DeepSeek high-res projection can be run once per image.
- Prefix construction: image features replace placeholder embeddings in the prompt.
- Prefill: full multimodal prompt through LLaMA with causal mask and RoPE.
- Decode: text-only token steps using LLaMA KV cache; pixel tensors are passed only on the first generation iteration unless `use_cache=False`.

## 3. Important config dimensions

| Field | 7B base/chat value | Source |
| --- | ---: | --- |
| text hidden size | 4096 | HF config |
| text layers | 30 | HF config |
| text heads / KV heads / head dim | 32 / 32 / 128 | HF config |
| text intermediate | 11008 | HF config |
| vocab size | 102400 | HF config |
| max positions | 16384 | HF config |
| RoPE | theta 10000, no scaling | HF config |
| low-res vision | SigLIP, 24 layers, 1024 hidden, 16 heads | HF config |
| low-res image/patch/tokens | 384, patch 16, 576 tokens | HF config/source |
| high-res vision | SAM vision, 12 layers, 768 hidden, 12 heads | HF config |
| high-res image/patch/grid | 1024, patch 16, 64 x 64 | HF config/source |
| SAM output channels | 256 | HF config |
| high-res projected tokens | 576 x 1024 | source inference from conv/proj |
| image token id/count | 100015 / 576 per image | HF tokenizer/processor |
| dtype | float16 | HF config |
| cache | LLaMA `Cache`, `use_cache=True` | HF config/source |

Representative sweep:

| Checkpoint | model_type | text | low-res vision | high-res vision | ABI note |
| --- | --- | --- | --- | --- | --- |
| `deepseek-community/deepseek-vl-7b-base` | `deepseek_vl_hybrid` | 30L, 4096, MHA | SigLIP 384 -> 576 tokens | SAM 1024 -> projected 576 tokens | native target |
| `deepseek-community/deepseek-vl-7b-chat` | `deepseek_vl_hybrid` | same | same | same | same graph; chat template matters |
| `deepseek-community/deepseek-vl-1.3b-base` | `deepseek_vl` | 24L, 2048 | SigLIP only in inspected config | none | out of scope for this hybrid report |

## 3a. Family variation traps

- Native hybrid configs compose three model families; DinoML should admit exact `text_config.model_type=llama`, `vision_config.model_type=siglip_vision_model`, and `high_res_vision_config.model_type=sam_vision_model` first.
- The 1.3B DeepSeek-VL configs are not `deepseek_vl_hybrid`; do not silently route them to this graph.
- Low-res and high-res paths must produce the same 24 x 24 token grid before alignment. Official 7B does this through `384/16 = 24` and high-res downprojection from 64 x 64 to 24 x 24.
- The processor expands the image token string, not a separate image-grid tensor. The model validates only total placeholder count against flattened image features, so DinoML needs stricter guards for per-sample counts/order.
- SAM uses NHWC internally for transformer layers, then NCHW neck/projection convs. This is a layout-sensitive boundary, not a free global NHWC conversion.
- SAM alternates local window attention and global attention by `global_attn_indexes`; local windows pad/unpad NHWC tensors.
- LLaMA here is MHA, not GQA, because `num_key_value_heads == num_attention_heads`; source still supports GQA if future configs change.
- `logits_to_keep` is a runtime efficiency ABI on the conditional-generation wrapper.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup, masked/indexed replacement, reshape/view, flatten, transpose/permute, contiguous, concat along hidden axis, dynamic slicing for `logits_to_keep`.
- NCHW image tensors from processor; SigLIP converts conv output to sequence; SAM uses NCHW patch conv -> NHWC transformer -> NCHW neck -> sequence.

Neural primitives:

- LLaMA: RMSNorm, bias-free Linear, SwiGLU `down(silu(gate(x)) * up(x))`, residual add.
- SigLIP: Conv2d patch embedding, learned position embedding, LayerNorm, dense noncausal MHA, GELU MLP.
- SAM: Conv2d patch embedding, learned absolute position, LayerNorm, qkv Linear with bias, GELU MLP, Conv2d neck/projection, bilinear interpolate, scalar multiply by `high_res_vision_alpha`.
- Aligner: `Linear(1024 -> 2048)` low-res, `Linear(1024 -> 2048)` high-res, concat to 4096, GELU, `Linear(4096 -> 4096)`.

Attention primitives:

- Causal LLaMA self-attention with RoPE, cache update, SDPA/FlashAttention-compatible path.
- SigLIP dense noncausal self-attention, no cache.
- SAM local window attention and global attention with decomposed relative position bias.

Preprocessing-coupled ops:

- Resize longest side to target, center pad to square, rescale by `1/255`, normalize, and emit `pixel_values` plus `high_res_pixel_values`.
- Placeholder token expansion from one `<image_placeholder>` to 576 repeated tokens.

Scatter/indexed update:

- Source uses boolean `masked_scatter` over expanded `[B,S,H]` mask. DinoML should lower first to ordered row copy with guards: exactly 576 placeholders per image, placeholder positions stable after tokenization, flattened image features row-major `[image, token, hidden]`.

## 5. Layer/block breakdown

Low-res SigLIP vision, 24 layers:

```text
pixel_values [B,3,384,384]
patch = Conv2d(3 -> 1024, kernel=stride=16) -> [B,1024,24,24]
x = flatten/transpose + learned position -> [B,576,1024]
repeat 24:
  x = x + MHA(LayerNorm(x))          # 16 heads, head_dim 64, q/k/v/out all 1024
  x = x + MLP(LayerNorm(x))          # Linear 1024->4096, GELU, Linear 4096->1024
x = final LayerNorm(x)
```

High-res SAM branch:

```text
high_res_pixel_values [B,3,1024,1024]
x = Conv2d(3 -> 768, kernel=stride=16) -> [B,64,64,768] NHWC
x = x + abs_pos
repeat 12:
  x = LayerNorm(x)
  if layer not in [2,5,8,11]: partition NHWC windows of 14 with pad/unpad
  qkv = Linear(768 -> 2304, bias=True)
  scores = q @ k.T * scale + decomposed_rel_pos
  x = residual + Linear(attn)
  x = x + MLP(LayerNorm(x))          # 768->3072->768, GELU
sam_last = neck Conv1x1 768->256, LN channels_first, Conv3x3 256->256, LN
deepseek_last = interpolate to 96x96, Conv3x3 stride2 256->512, Conv3x3 stride2 512->1024
global_hidden = same neck/proj applied to hidden_states[global_attn_index + 1]
high = deepseek_last + global_hidden * high_res_vision_alpha
high = NCHW -> [B,576,1024]
```

Aligner and language:

```text
low = Linear(1024 -> 2048)(siglip_tokens)
high = Linear(1024 -> 2048)(sam_tokens)
img = Linear(4096 -> 4096)(GELU(concat([high, low], dim=-1)))
inputs_embeds = token_embedding(input_ids)
inputs_embeds[image_token_mask] = img.reshape(-1,4096)
repeat 30 LLaMA layers:
  x = x + o_proj(attn(RMSNorm(x), RoPE, causal mask, cache))
  x = x + down_proj(silu(gate_proj(RMSNorm(x))) * up_proj(...))
logits = lm_head(x[:, slice_indices, :])
```

## 6. Attention requirements

Text decoder:

- Causal self-attention only; no cross-attention.
- Official configs are MHA: 32 Q heads, 32 KV heads, head dim 128.
- Q/K/V projections are separate bias-free Linear layers: 4096 -> 4096.
- RoPE is applied to Q/K before cache update. Cached keys are post-RoPE.
- Cache is Transformers `Cache`, updated per layer with `[B, heads, seq, head_dim]` K/V tensors.
- Prefill can use FlashAttention/SDPA if mask semantics match; decode needs KV append and last-token logits.

Low-res SigLIP:

- Noncausal dense self-attention over 576 tokens, 16 heads, head dim 64.
- No KV cache and no local/sparse pattern.

High-res SAM:

- 12 layers on a 64 x 64 grid, 12 heads, head dim 64.
- Layers not in `[2,5,8,11]` use 14 x 14 local windows with padding to a multiple of 14.
- Global layers use full 4096-token spatial attention.
- Adds decomposed relative position bias computed from query and height/width relative tables. SDPA path passes it as `attn_mask`.

## 7. Position encoding and custom math

LLaMA RoPE follows the standard LLaMA source:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = torch.cat((freqs, freqs), dim=-1)
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

SigLIP uses a learned absolute position embedding for the fixed 24 x 24 patch grid. Interpolation exists in source but the DeepSeek wrapper does not pass `interpolate_pos_encoding`; first admission should require 384 x 384 low-res inputs.

SAM uses learned absolute NHWC position plus decomposed relative position bias:

```python
rel_h = einsum("bhwc,hkc->bhwk", query.reshape(B,H,W,C), rel_pos_h)
rel_w = einsum("bhwc,wkc->bhwk", query.reshape(B,H,W,C), rel_pos_w)
scores += reshape(rel_h[..., None] + rel_w[..., None, :])
```

The high-res branch also has a learned scalar `high_res_vision_alpha`, initialized to zero by source initialization but loaded from checkpoint for inference.

## 8. Preprocessing and input packing

Processor ABI:

- Text must contain the tokenizer's `image_token` string, `<image_placeholder>` for inspected configs.
- The processor replaces each occurrence with 576 repeated `<image_placeholder>` strings before tokenization.
- Tokenizer config maps `<image_placeholder>` to id 100015.
- `processor_config.json` records `num_image_tokens=576`.

Image ABI:

- Emits `pixel_values` and `high_res_pixel_values`.
- Official preprocessor low-res path: resize longest side to 384, center-pad square when needed, rescale, normalize with mean/std `[0.5,0.5,0.5]`.
- High-res path: resize longest side to 1024, center-pad square, rescale, normalize with CLIP mean/std.
- Torchvision/PIL implementations both produce channel-first tensors for model input.

Runtime packing:

- Model requires `pixel_values` and `high_res_pixel_values` together.
- Image features flatten as `[num_images_or_batch, 576, hidden] -> [-1, hidden]`.
- Source accepts arbitrary placeholder positions through `masked_scatter`, but processor-produced prompts usually contain contiguous runs of 576 tokens per image. DinoML should reject arbitrary boolean scatter until a stricter indexed-copy ABI is validated.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed patch Conv2d -> Linear

Source pattern: SigLIP/SAM patch embedding `Conv2d(C -> D, kernel=stride=patch, padding=0)`.

Replacement:

```text
NCHW fixed image -> non-overlap patch flatten -> GEMM(weight_flat.T) -> add bias if present -> sequence/grid reshape
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`
- input spatial dims equal config image size and divisible by patch size
- preserve source flatten order: row-major patches, channel/kernel order matching PyTorch Conv2d

Failure cases: dynamic image sizes, interpolated SigLIP positions, grouped convs, or layout pass that changes patch flatten order.

### Rewrite: multimodal masked_scatter -> guarded indexed row copy

Source pattern: expanded mask `[B,S,H]` from `input_ids == image_token_id`, then `masked_scatter`.

Replacement:

```text
positions = where(input_ids == image_token_id)
guard count == num_images * 576
copy image_features.reshape(-1,H) into inputs_embeds at ordered positions
```

Preconditions: source prompt expansion, stable row-major order, no extra image-token ids outside image spans.

### Rewrite: SAM NCHW/NHWC layout region

SAM source uses NCHW convs, NHWC transformer attention/MLP, then NCHW neck/projection. A guarded layout pass can keep SAM transformer activations NHWC and avoid repeated transposes around layer norms. Do not translate the whole model to NHWC; LLaMA and SigLIP sequence layouts are independent.

### Rewrite: high-res projection interpolate + stride convs

The high-res projection always interpolates SAM neck output to `4 * output_size = 96`, then two stride-2 convs produce 24 x 24. This is a bounded region for provider-backed bilinear resize and conv fusion. Guard on official `output_size=24`, input channels 256, output channels 1024.

## 10. Kernel fusion candidates

Highest priority:

- LLaMA RMSNorm, RoPE + attention prefill/decode, SwiGLU MLP, and last-token-only `lm_head`.
- Multimodal indexed embedding stitch, because general boolean scatter is broader than needed.
- SAM window attention with relative bias and pad/unpad, because it is the unusual operator surface.

Medium priority:

- SigLIP patch embedding as GEMM and fixed 576-token dense attention.
- SAM Conv/LN neck and DeepSeek projection conv stack.
- Aligner GEMM + GELU + GEMM for 576 tokens.

Lower priority:

- CPU/GPU image preprocessing. First parity can keep this in the data pipeline.
- `output_attentions` and hidden-state recording.

## 11. Runtime staging plan

Stage 1: parse config and reject unsupported variants. Load weights for the exact 7B hybrid graph.

Stage 2: implement processor ABI validation: 576 placeholders per image, paired low/high tensors, fixed sizes.

Stage 3: run SigLIP low-res encoder parity in isolation.

Stage 4: run SAM high-res encoder/projection parity in isolation, including local/global attention layers.

Stage 5: run aligner and embedding-stitch parity, replacing `masked_scatter` with guarded indexed copy.

Stage 6: run LLaMA text-only prefill/decode with KV cache.

Stage 7: end-to-end multimodal prefill logits, then cached decode token parity.

Stage 8: add optimized attention/conv/GEMM providers and layout-local rewrites.

## 12. Parity and validation plan

- Processor parity: one image with aspect ratios square/wide/tall; verify low/high tensor shapes, pad placement, mean/std, and token count.
- SigLIP single-layer and full-encoder parity at fp32/fp16.
- SAM layer parity for one local-window layer and one global layer, including relative-bias math.
- High-res projection parity: SAM neck output -> 576 x 1024 tokens.
- Aligner parity on random `[B,576,1024]` low/high inputs.
- Embedding stitch parity for contiguous and deliberately invalid placeholder layouts.
- LLaMA block, prefill logits, and single-token decode with cache.
- End-to-end image-text prompt logits and generated first token against Transformers.

Suggested tolerances: fp32 `1e-4` to `1e-5` for isolated ops; fp16 `1e-2` to `5e-2` for full vision branches, with tighter tolerances for GEMM-only slices.

## 13. Performance probes

- CPU preprocessing images/sec by resolution/aspect ratio.
- Low-res SigLIP encoder throughput.
- High-res SAM encoder throughput split by local-window versus global layers.
- High-res projection conv/interpolate time.
- Aligner time.
- Prefill tokens/sec with 576 image tokens plus text length sweep.
- Decode tokens/sec with KV cache and last-token logits.
- KV cache memory for 30 layers at sequence length 576 + prompt + generated tokens.
- Layout-pass A/B probe for SAM NHWC-preserving path versus explicit permutes.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and output attentions.
- Non-hybrid `deepseek_vl` 1.3B routing.
- Arbitrary `AutoConfig` substitutions for text/vision/high-res branches.
- Dynamic low/high image sizes and SigLIP position interpolation.
- General boolean scatter beyond processor-guaranteed image-token runs.
- Beam search, speculative decoding, and multi-GPU tensor parallel.
- Quantized/packed loading unless a separate weight-format audit requests it.

## 15. Final implementation checklist

- [ ] Parse `DeepseekVLHybridConfig` and admit only LLaMA + SigLIP vision + SAM vision first.
- [ ] Load 7B hybrid weights with tied `lm_head`/embedding alias preserved.
- [ ] Implement processor ABI guards for paired image tensors and 576 placeholders per image.
- [ ] Implement low-res SigLIP encoder operators.
- [ ] Implement high-res SAM local/global attention with decomposed relative position bias.
- [ ] Implement high-res projection conv/interpolate path.
- [ ] Implement aligner projections and GELU.
- [ ] Lower image embedding stitch to guarded indexed row copy.
- [ ] Implement LLaMA prefill/decode with RoPE and KV cache.
- [ ] Add layout guards around SAM NHWC/NCHW boundaries.
- [ ] Add parity tests for processor, branch encoders, aligner, stitch, prefill, and decode.
- [ ] Benchmark preprocessing, low-res encoder, high-res encoder, prefill, decode, and cache memory.
