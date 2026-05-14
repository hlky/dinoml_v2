# MiniCPM-V 4.6 (`minicpmv4_6`) DinoML Audit

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary: openbmb/MiniCPM-V-4.6, alias openbmb/MiniCPM-V-4_6.

Config source:
  HF config snapshot from openbmb/MiniCPM-V-4.6, repo sha c83e202c69261e37ab3df63177047f36d6841931.
  Saved under agents/plans/transformers/minicpmv4_6/_sources/.

Source files inspected:
  minicpmv4_6/configuration_minicpmv4_6.py
  minicpmv4_6/modeling_minicpmv4_6.py
  minicpmv4_6/modular_minicpmv4_6.py
  minicpmv4_6/processing_minicpmv4_6.py
  minicpmv4_6/image_processing_minicpmv4_6.py
  minicpmv4_6/image_processing_pil_minicpmv4_6.py
  minicpmv4_6/video_processing_minicpmv4_6.py
  qwen3_5/configuration_qwen3_5.py
  qwen3_5/modeling_qwen3_5.py
  cache_utils.py

Any missing files or assumptions:
  No imports/tests were run. Report is source/config audit only.
  modeling_minicpmv4_6.py is generated from modular_minicpmv4_6.py; future HF source edits should target modular.
  The primary runtime target is multimodal causal generation: image/video/text prefix construction, prefill, decode, logits.
```

## 2. High-level architecture

MiniCPM-V 4.6 is a multimodal projector plus hybrid causal LM:

```text
CPU image/video processor + chat template
  -> NaViT-packed pixel_values / pixel_values_videos + target_sizes
  -> MiniCPMV4_6VisionModel
  -> window attention merger and downsample MLP projector
  -> masked placeholder replacement in text embeddings
  -> Qwen3.5 text decoder prefill/decode
  -> tied/untied LM head logits
```

Stage decomposition:

- CPU/data pipeline: resize, optional slicing, rescale/normalize, patchify into `[1, C, patch_size, packed_len]`, text placeholder expansion, tokenization.
- Vision encoder/projector: independently cacheable per prompt image/video; consumes `target_sizes` and emits one feature row per placeholder token.
- Prefix construction: embeds text tokens, replaces image/video placeholder token rows with visual features via `masked_scatter`.
- Text prefill/decode: nested `qwen3_5_text` model with 18 linear-attention Gated Delta Net layers and 6 full-attention layers for the primary config.
- Logits: `lm_head(hidden_states)`, with `_tied_weights_keys` declaring `lm_head.weight` aliases `model.language_model.embed_tokens.weight`; primary HF config also sets `tie_word_embeddings=true`.

## 3. Important config dimensions

Primary `openbmb/MiniCPM-V-4.6` dimensions from `config.json`:

| Field | Value | Source |
|---|---:|---|
| model_type | `minicpmv4_6` | config |
| dtype / params | BF16, 1,300,428,016 params | HF safetensors metadata |
| image_token_id / video_token_id | 248056 / 248057 | config |
| text vocab_size | 248094 | config `text_config` |
| text hidden_size | 1024 | config `text_config` |
| text layers | 24 | config `text_config` |
| text layer pattern | 3 linear attention, then 1 full attention, repeated | config `layer_types` |
| full attention heads / KV heads | 8 / 2 | config |
| full attention head_dim | 256 | config; hidden_size != heads * head_dim |
| Q projection width | `8 * 256 * 2 = 4096`, split into q and gate | source + config |
| K/V projection width | `2 * 256 = 512` each | source + config |
| linear key/value heads | 16 / 16 | config |
| linear key/value head dim | 128 / 128 | config |
| linear conv kernel | 4 | config |
| MLP intermediate | 3584 | config |
| activation | `silu` text, `gelu_pytorch_tanh` vision | config |
| max_position_embeddings | 262144 | config |
| RoPE | default, theta 10000000, partial_rotary_factor 0.25 | config |
| vision hidden / layers / heads | 1152 / 27 / 16 | config |
| vision intermediate | 4304 | config |
| vision patch_size | 14 | config |
| vision image_size | 980 in vision config; top-level image_size 1120 | config |
| insert_layer_id | 6 | config/source |
| merge_kernel_size | default `(2, 2)` unless omitted config default | source default |
| merger_times | default `1` unless omitted config default | source default |

Representative checkpoint sweep:

| Checkpoint | Scope | Runtime class | Key variation |
|---|---|---|---|
| `openbmb/MiniCPM-V-4.6` | primary | in-library `MiniCPMV4_6ForConditionalGeneration` | Qwen3.5 hybrid text, NaViT packer, image/video tokens |
| `openbmb/MiniCPM-V-4_6` | alias observed | same config contents | same as primary |
| `openbmb/MiniCPM-V-4` | historical trap | remote-code `MiniCPMV`, `model_type=minicpmv` | LongRoPE, different vocab/hidden size, remote code, not this source |
| `openbmb/MiniCPM-V` | historical trap | remote-code `MiniCPMV` | older SigLIP/timm-style config, not this source |

The primary family currently has one official accessible in-library checkpoint. Older MiniCPM-V configs are useful only to reject accidental routing into `minicpmv4_6`.

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` in text full attention: `1024 != 8 * 256`. Q/O projection widths must come from config/source, not hidden size.
- Full attention Q projection is nonstandard: `q_proj` emits `num_heads * head_dim * 2`, then splits into query and an output gate.
- Text stack is hybrid. `layer_types` drives either `linear_attention` Gated Delta Net or `full_attention`; do not assume every layer has KV cache.
- The nested text model supports optional FlashAttention/SDPA/eager interfaces for full attention, but linear-attention layers require FLA/causal-conv1d fast paths or expensive torch fallbacks.
- `attn_output_gate`, `mamba_ssm_dtype`, `mlp_only_layers`, and MTP fields appear in the checkpoint config but are not read by the inspected `qwen3_5_text` source path for this wrapper. Treat them as ignored by this source basis unless future source changes consume them.
- The MiniCPM wrapper does not use Qwen3.5-VL-style `image_grid_thw`, `video_grid_thw`, or `mm_token_type_ids`; it passes visual embeddings through placeholder replacement and then calls `Qwen3_5TextModel`.
- Vision input is already patchified by the processor into `[1, C, patch_size, packed_len]`; the source `Conv2d` patch embedding sees height `patch_size` and width equal to packed patch rows, not ordinary `[B, C, H, W]` images.
- Vision attention is variable-length over packed visual units using `cu_seqlens` for FlashAttention or Python `split`/`cat` fallback.
- The window merger gathers/scatters by computed `window_index`, sorts back with `argsort`, and requires target heights/widths divisible by `(2, 2)`.
- Placeholder replacement uses broad `masked_scatter`, but processor-generated placeholders are stricter: repeated `<|image_pad|>` or `<|video_pad|>` token runs in text order, counts equal `prod(target_size) // divisor`.
- `downsample_mode="4x"` skips the intermediate ViT merger and changes visual token divisor from 16 to 4.
- Image/video preprocessing is axis-sensitive NCHW/packed layout. NHWC/channel-last is only a guarded optimization inside controlled patchify/conv regions.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape`, `view`, `flatten`, `transpose`, `permute`, `contiguous`, `cat`, `concat`, `split`, `repeat`, `repeat_interleave`, `expand`, `unsqueeze`, `squeeze`, `argsort`, `arange`, `bucketize`, `cumsum`, `pad`, `index_select`/advanced gather.
- Dynamic integer shape math from `target_sizes`: products, cumulative sums, max segment length, divisibility guards.

Neural primitives:

- Vision `Conv2d(3 -> 1152, kernel=14x14, stride=14x14, padding=valid)` over packed image tensor.
- Vision `LayerNorm(1152)`, `Linear(1152 -> 1152)` Q/K/V/O with bias, `Linear(1152 -> 4304 -> 1152)` MLP with `gelu_pytorch_tanh`.
- Window merger: varlen attention over 2x2 windows, `LayerNorm(1152)`, `LayerNorm(4608)`, `Linear(4608 -> 17216 -> 1152)`.
- Projector/downsample MLP: `LayerNorm(4608)`, `Linear(4608 -> 4608 -> 1024)` for primary `merger_times=1`.
- Text RMSNorm with `(1 + weight)` scale initialized at zero.
- Text SwiGLU MLP: `Linear(1024 -> 3584)` gate/up, `silu(gate) * up`, `Linear(3584 -> 1024)`.
- LM head `Linear(1024 -> 248094, bias=False)`, tied to embeddings for primary config.

Attention primitives:

- Vision noncausal MHA with varlen packed segments, 16 heads, head_dim 72 (`1152 / 16`).
- Text full causal GQA with 8 Q heads, 2 KV heads, head_dim 256, KV repeat factor 4, q/k RMSNorm, RoPE, sigmoid output gate.
- Text linear attention Gated Delta Net: depthwise causal conv1d, sigmoid/softplus/exp state math, chunk rule for prefill, recurrent rule for single-token decode.

Position/rotary/custom math:

- Vision learned position embedding with fractional coordinate `bucketize` against a 70x70 table for `image_size=980`, `patch_size=14`.
- Text RoPE with partial rotary dim `256 * 0.25 = 64`; source uses a 3-row M-RoPE-style frequency interleave even when MiniCPM wrapper supplies ordinary positions.

Generation/cache ops:

- Full attention dynamic KV cache for full-attention layers only.
- Linear attention fixed state cache per linear layer: `conv_states [B, conv_dim, 4]` where `conv_dim=6144`, and recurrent state `[B, 16, 128, 128]`.
- Cache reorder for beam search must reorder both KV layers and linear attention states.
- `prepare_inputs_for_generation` sends pixel tensors only on first iteration or when `use_cache=False`; target sizes are deliberately not beam-expanded.

Preprocessing-coupled ops:

- Bicubic resize, RGB conversion, rescale/normalize, optional high-res slicing, patchify/unfold, video sampling, optional sub-frame canvas concatenation.
- Chat template maps image/video content to `<|image_pad|>` and `<|video_pad|>` before processor expansion.

Scatter/indexed update:

- `inputs_embeds.masked_scatter(mask, image_features)` and same for video. DinoML should lower this to validated row copy over placeholder positions, not admit general boolean scatter.

## 5. Layer/block breakdown

Vision embedding:

```text
pixel_values [1, 3, 14, packed_len]
  -> Conv2d kernel=stride=14 -> [1, 1152, 1, num_patches_total]
  -> flatten(2).transpose(1,2) -> [1, num_patches_total, 1152]
  -> add per-unit learned position embeddings selected by target_sizes
```

Vision encoder layer, repeated 27 times with merger inserted after layer index 6 in `"16x"` mode:

```text
x = x + MHA(LayerNorm(x), cu_seqlens, noncausal)
x = x + Linear(gelu_tanh(Linear(LayerNorm(x))))
```

ViT window merger at insert layer:

```text
window_index = 2x2 window gather from each target_size
x_window = x[:, window_index, :]
x_window = x_window + MHA(LayerNorm(x_window), window_cu_seqlens)
x = restore_original_order(x_window) + residual
for each visual unit:
  x = view/permute 2x2 windows -> [num_windows, 4608]
  residual = mean over 4 window tokens -> [num_windows, 1152]
  x = Linear(17216 -> 1152, gelu_tanh(Linear(LayerNorm(4608))))
  x = x + residual
```

Projector merger:

```text
for each target_size:
  x = view/permute 2x2 windows -> [num_tokens, 4608]
  x = LayerNorm(4608)
  x = Linear(4608 -> 4608)
  x = GELU
  x = Linear(4608 -> 1024)
```

Text linear-attention layer:

```text
r = x
x = RMSNorm(x)
mixed_qkv = Linear(1024 -> 6144)(x)
z = Linear(1024 -> 2048)(x)
b = Linear(1024 -> 16)(x).sigmoid()
a = Linear(1024 -> 16)(x)
mixed_qkv = depthwise causal Conv1d(groups=6144, kernel=4) + SiLU
q, k, v = split(2048, 2048, 2048), reshape to 16 heads of 128
g = -exp(A_log) * softplus(a + dt_bias)
out = gated_delta_rule(q, k, v, g, beta=b, recurrent_state)
out = RMSNormGated(out, z)
x = r + Linear(2048 -> 1024)(out)
x = x + SwiGLU(RMSNorm(x))
```

Text full-attention layer:

```text
r = x
x = RMSNorm(x)
q, gate = split(Linear(1024 -> 4096)(x), last_dim, 2)
k = Linear(1024 -> 512)(x)
v = Linear(1024 -> 512)(x)
q = RMSNorm(head_dim=256)(q); k = RMSNorm(head_dim=256)(k)
q, k = RoPE(q, k)
k, v = KV-cache update; repeat KV heads by 4 for eager attention
attn = causal scaled dot product attention
attn = attn * sigmoid(gate)
x = r + Linear(2048 -> 1024)(attn)
x = x + SwiGLU(RMSNorm(x))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA, no KV cache.
- Input packed as one batch row with concatenated visual units; `cu_seqlens` splits independent units.
- FlashAttention path consumes `cu_seq_lens_q/k` and max lengths; fallback loops over segment splits and concatenates outputs.
- No attention mask in source for normal vision path.

Text full attention:

- Causal self-attention with GQA.
- Q heads 8, KV heads 2, head_dim 256, Q/K/V widths 2048/512/512, output projection input width 2048.
- KV cache stores post-RoPE K and raw V per full-attention layer after `past_key_values.update`.
- Mask comes from `create_causal_mask`; padding attention mask is respected.
- Eager path upcasts softmax to fp32, then returns to query dtype.
- FlashAttention/SDPA can be used for the full-attention layers if DinoML reproduces q/k norm, RoPE, gate, and mask order.

Text linear attention:

- Not KV attention. It owns fixed-size conv and recurrent states.
- Prefill/chunk mode may use FLA `chunk_gated_delta_rule`; decode single-token mode may use fused recurrent rule.
- Torch fallback contains loops and triangular/chunk temporaries; too slow for production but useful as parity spec.
- Padding mask for linear layers is a 2D mask that zeroes hidden states unless cached or all-ones.

Cache manifest for primary config:

- Layers 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22: linear attention state.
- Layers 3,7,11,15,19,23: full attention KV cache.
- Beam reorder must apply to both families.

## 7. Position encoding and custom math

Vision learned position ids:

```python
boundaries = arange(1 / 70, 1.0, 1 / 70)  # 980 / 14
bucket_h = bucketize(arange(0, 1 - 1e-6, 1 / target_h), boundaries, right=True)
bucket_w = bucketize(arange(0, 1 - 1e-6, 1 / target_w), boundaries, right=True)
pos_ids = (bucket_h[:, None] * 70 + bucket_w).flatten()
embeddings += position_embedding(pos_ids)
```

Text RoPE:

```python
rotary_dim = int(head_dim * partial_rotary_factor)  # 64 for primary config
freqs = inv_freq @ position_ids[temporal,height,width]
freqs_t = freqs[0]
freqs_t[..., 1:33:3] = freqs[1, ..., 1:33:3]
freqs_t[..., 2:30:3] = freqs[2, ..., 2:30:3]
cos, sin = cos(freqs_t), sin(freqs_t)
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

For this MiniCPM wrapper, `position_ids` usually defaults inside `Qwen3_5TextModel` to four identical sequential rows; the first is used for causal mask positions, and the remaining three feed RoPE. Precompute cos/sin for common decode buckets, but preserve dynamic update for long contexts.

## 8. Preprocessing and input packing

Image ABI:

- Processor accepts images, converts RGB, resizes/slices on CPU, rescales and normalizes.
- Default HF preprocessor snapshot: `max_slice_nums=9`, `scale_resolution=448`, `patch_size=14`, `slice_mode=true`, `use_image_id=true`, mean/std `[0.5, 0.5, 0.5]`.
- Output `pixel_values`: `[1, 3, 14, packed_len]`.
- Output `target_sizes`: `int32[num_visual_units, 2]`, patch grid height/width for source image and optional slices.
- Extra processor-only metadata: `grids`, `num_patches_per_image` used to expand placeholders.

Video ABI:

- Processor can decode/sample frames when metadata includes duration/fps.
- Default main frame cap 128; if `stack_frames > 1`, sub-frames are arranged into per-second composite canvases with 6-pixel separators.
- Each visual unit is processed like an image; output `pixel_values_videos` is `[1, 3, 14, packed_len]` from the processor.
- Model `get_video_features` expects effective first dim as frame/beam dimension, then repacks with `permute(1,2,0,3).reshape(1, C, patch_size, -1)`. This path is shape-sensitive and should be parity-tested carefully.

Placeholder stitch:

- Tokenizer special tokens include `<|image_pad|>`, `<|video_pad|>`, `<image>`, `</image>`, `<slice>`, `</slice>`, `<image_id>`, `</image_id>`.
- For each image placeholder, processor replaces one `<|image_pad|>` marker with:
  `image_id? + <image> + N source image pad tokens + </image> + optional slice rows`.
- `N = target_h * target_w // divisor`, divisor is 16 for `"16x"` and 4 for `"4x"`.
- The model validates only total mask element count against feature element count, then uses `masked_scatter`. DinoML should add stricter guards: placeholder IDs match, positions are in processor order, total rows equal concatenated feature rows, and no arbitrary interleaved mask pattern unless explicitly allowed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed patch Conv2d -> Linear

Source pattern:

```text
Conv2d(C=3, out=1152, kernel=(14,14), stride=(14,14), padding=valid)
on pixel_values [1,3,14,packed_len]
```

Replacement:

```text
Processor patchify produces columns of 3*14*14 values
GEMM(patches, conv.weight.reshape(1152, 588).T) + bias
```

Preconditions:

- `pixel_values.shape[2] == patch_size`.
- Packed length divisible by `patch_size`.
- Conv kernel/stride equal patch size, groups 1, dilation 1, valid padding.
- Preserve processor flatten order: torch path uses unfold then `reshape(C, P, P, -1).permute(0,1,3,2).reshape(C,P,-1)`.

Failure cases:

- Ordinary NCHW images not pre-patchified.
- Alternate patch size or channel count without weight transform.

Parity sketch:

- Compare Conv2d output against Linear over processor-produced patches for random packed tensors and real preprocessor outputs.

### Rewrite: placeholder `masked_scatter` -> indexed row copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = inputs_embeds.masked_scatter(mask[...,None], image_features)
```

Replacement:

```text
positions = nonzero(input_ids == token_id) in row-major order
copy image_features rows to inputs_embeds[positions]
```

Preconditions:

- `input_ids` available.
- Number of positions equals feature rows.
- Mask expanded only over embedding dimension.
- Positions generated by MiniCPM processor or accepted by explicit row-major guard.

Failure cases:

- `inputs_embeds`-only placeholder detection through embedding equality.
- Non-row-major or duplicated/arbitrary scatter semantics.

### Rewrite: 2x2 visual merge -> static window gather + GEMM

Source pattern:

```text
view(merged_h,2,merged_w,2,C).permute(0,2,1,3,4).reshape(-1,4*C)
```

Replacement:

```text
layout-aware gather/window-pack -> LayerNorm -> GEMM -> GELU -> GEMM
```

Preconditions:

- `target_h` and `target_w` divisible by 2 at each merge round.
- Source is contiguous segment per visual unit.
- No global layout translation crosses segment boundaries.

Failure cases:

- Dynamic target sizes without divisibility guards.
- `"4x"` mode skips the ViT merger but still uses final projector merge.

### Rewrite: full-attention Q projection split

Source pattern:

```text
q_gate = Linear(1024 -> 4096)
q, gate = chunk(q_gate.view(..., heads, head_dim*2), 2, dim=-1)
```

Replacement:

```text
single GEMM -> view -> split q/gate -> q RMSNorm/RoPE; gate saved for output sigmoid multiply
```

Preconditions:

- Weight layout is PyTorch linear `[out_features, in_features]`.
- Split is per-head in the last dimension, not all-Q rows then all-gate rows globally.

Failure cases:

- Treating projection as `hidden_size -> hidden_size` due to hidden size.

## 10. Kernel fusion candidates

Highest priority:

- Qwen3.5 linear-attention Gated Delta Net fast path: this dominates 18/24 text layers and requires custom state ABI.
- Full-attention fused path: Q/K RMSNorm + partial RoPE + GQA attention + sigmoid output gate.
- RMSNorm and SwiGLU MLP fusion for text layers.
- Placeholder indexed row copy, because general boolean scatter is broader than DinoML should admit.
- Vision varlen attention with `cu_seqlens`, because image/video prefix cost is front-loaded and independently cacheable.

Medium priority:

- Packed patch Conv2d-to-GEMM lowering.
- 2x2 merger pack + projector MLP fusion.
- Vision LayerNorm + QKV projections.
- Last-token-only logits for decode.

Lower priority:

- CPU/GPU processor acceleration. First integration can keep resize/slicing/patchify on CPU.
- `"4x"` high-token mode optimization after `"16x"` parity.
- Beam-search pixel re-encoding avoidance beyond source-compatible first-iteration handling.

## 11. Runtime staging plan

Stage 1: parse config/weights and reject non-`minicpmv4_6` remote-code MiniCPM variants.

Stage 2: CPU/data pipeline compatibility: accept processor-produced `input_ids`, `attention_mask`, `pixel_values`, `target_sizes`, and optionally bypass in-runtime preprocessing.

Stage 3: vision encoder/projector parity for fixed image sizes in `"16x"` mode, including `target_sizes`, `cu_seqlens`, insert-layer window merger, and final projector.

Stage 4: placeholder stitch as indexed row copy with strict guards.

Stage 5: text single-block parity for both layer types: one Gated Delta Net layer and one full-attention layer.

Stage 6: full prefill parity with hybrid cache object initialized but decode disabled.

Stage 7: decode parity with mixed cache: KV for full-attention layers, conv/recurrent state for linear layers, plus beam reorder.

Stage 8: optimized attention/linear-attention kernels, last-token logits, visual feature caching, and batching.

Initially stub or defer video sampling, `"4x"` mode, processor-in-runtime resize, beam search, and non-primary historical configs.

## 12. Parity and validation plan

- Processor ABI snapshot tests: run HF processor offline and assert shapes, target sizes, placeholder token counts, and text replacement for single image, multi-image, sliced image, and one short video.
- Vision embedding test: Conv2d output plus learned position embedding for fixed `target_sizes`.
- Vision varlen attention test: two visual units with different patch counts, compare FlashAttention-compatible varlen path or eager split fallback.
- Window merger test: small divisible grids, validate gather -> attention -> argsort restore -> 2x2 merge.
- Projector test: `target_sizes` list produces feature row counts matching placeholders.
- Placeholder copy test: compare DinoML indexed row copy to HF `masked_scatter`; include rejection for count mismatch and arbitrary masks.
- Text block tests: one linear-attention layer prefill/decode with state update; one full-attention layer prefill/decode with KV update.
- End-to-end prefill logits: one image prompt and text-only prompt.
- Decode token parity: greedy 1, 2, 8 token decode with cache.
- Tolerances: fp32 custom math `1e-4` absolute/relative; bf16/fp16 logits around `5e-2` initially, tighten per kernel.

## 13. Performance probes

- CPU preprocessing throughput by image resolution, slice count, and video frame count.
- Vision encoder/projector throughput by total packed visual tokens and number of visual units.
- Prefill-only throughput by text length plus visual token count.
- Decode tokens/sec by batch size with hybrid cache enabled.
- Linear-attention kernel comparison: torch fallback vs custom chunk/recurrent provider.
- Full-attention backend comparison: eager/SDPA/FlashAttention-equivalent with GQA and gate.
- Cache memory probe: 6 KV layers plus 18 linear state layers; include beam expansion/reorder cost.
- Placeholder stitch cost by number of visual rows.
- LM head cost and last-token-only logits speedup.

## 14. Skip/defer list

- Training, losses, gradient checkpointing.
- Historical remote-code `minicpmv` checkpoints.
- Processor-owned video decode/frame sampling inside DinoML runtime; accept preprocessed tensors first.
- `"4x"` downsample mode until `"16x"` parity is stable.
- General boolean scatter.
- General dynamic image slicing on GPU.
- Tensor parallel/distributed plans.
- Cache offloading/prefetch.
- Speculative decoding and beam search beyond preserving source cache reorder ABI.
- Quantized/packed weight loading; no source-coupled quantized format is required by the inspected native source.

## 15. Final implementation checklist

- [ ] Parse `MiniCPMV4_6Config` plus nested `qwen3_5_text` and vision configs.
- [ ] Reject or route `model_type=minicpmv` remote-code historical checkpoints.
- [ ] Load/tie text embeddings and LM head according to `_tied_weights_keys`.
- [ ] Implement processor ABI intake for `pixel_values`, `target_sizes`, `pixel_values_videos`, `target_sizes_videos`.
- [ ] Add packed patch Conv2d lowering or direct Conv2d support for `[1,3,14,L]`.
- [ ] Implement vision learned position bucketization.
- [ ] Implement varlen noncausal vision attention with `cu_seqlens`.
- [ ] Implement 2x2 window attention merger gather/restore and 2x2 projector merge.
- [ ] Implement guarded placeholder indexed row copy for image/video tokens.
- [ ] Implement Qwen3.5 RMSNorm `(1 + weight)` semantics.
- [ ] Implement Qwen3.5 full-attention Q/gate projection split, q/k norm, partial RoPE, GQA, output gate.
- [ ] Implement Gated Delta Net prefill and decode kernels/state ABI.
- [ ] Implement mixed cache manifest and beam reorder for KV plus linear states.
- [ ] Add single-block, vision/projector, prefill-logit, and decode-token parity tests.
- [ ] Benchmark preprocessing, vision/projector, prefill, decode, cache memory, and LM head.
