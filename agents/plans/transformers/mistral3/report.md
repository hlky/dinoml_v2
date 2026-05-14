# DinoML Transformers Audit: mistral3

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary first target is `mistralai/Mistral-Small-3.1-24B-Instruct-2503`; representative sweep also checked 3.1 base, 3.2 instruct, `mistralai/Mistral-Medium-3.5-128B`, and `mistralai/Ministral-3-3B-Instruct-2512`.

Config source: Hub `config.json` / processor JSON snapshots saved beside this report:

- `mistral-small-3.1-24b-instruct-config.json`
- `mistral-small-3.1-24b-base-config.json`
- `mistral-small-3.2-24b-instruct-config.json`
- `mistral-medium-3.5-128b-config.json`
- `ministral-3-3b-instruct-config.json`
- `mistral-small-3.1-processor-config.json`
- `mistral-small-3.1-preprocessor-config.json`

Source files inspected:

- `src/transformers/models/mistral3/configuration_mistral3.py`
- `src/transformers/models/mistral3/modular_mistral3.py`
- `src/transformers/models/mistral3/modeling_mistral3.py`
- `src/transformers/models/mistral3/convert_mistral3_weights_to_hf.py`
- Delegated vision: `src/transformers/models/pixtral/configuration_pixtral.py`, `modeling_pixtral.py`, `processing_pixtral.py`, `image_processing_pixtral.py`, `image_processing_pil_pixtral.py`
- Delegated Small 3.x text core: `src/transformers/models/mistral/configuration_mistral.py`, `modeling_mistral.py`
- Brief guard check for newer text core: `src/transformers/models/ministral3/configuration_ministral3.py`, `modeling_ministral3.py`
- Docs reference: `docs/source/en/model_doc/mistral3.md`

Any missing files or assumptions: no remote-code files are needed for the inspected native source. `modeling_mistral3.py` is generated from `modular_mistral3.py`; future source edits should target the modular file, while runtime parity should match the generated file. Hub metadata for the checked official repos reported `gated: false`; the internal tiny random checkpoint returned 401 and is not used. The 3.2 instruct repo exposed `config.json` and `generation_config.json`, but `processor_config.json`, `preprocessor_config.json`, and `tokenizer_config.json` returned 404 from the raw main URL during this audit.

## 2. High-level architecture

`mistral3` is a multimodal image-text causal generation wrapper:

```text
CPU image/text preprocessing -> Pixtral vision encoder -> patch merger/projector
  -> replace [IMG] placeholders in text embeddings -> Mistral/Ministral causal decoder prefill
  -> autoregressive decode with text KV cache -> lm_head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: image decode, RGB conversion, resize to patch-aligned dimensions, rescale/normalize, right/bottom pad to batch max, chat template, `[IMG]` expansion into `[IMG]`, `[IMG_BREAK]`, `[IMG_END]` token runs.
- Independently cacheable vision/projector stage: `pixel_values[N,3,Hpad,Wpad]`, `image_sizes[num_images,2]` -> per-image projected features with width `text_hidden_size`.
- Prefix construction: token embeddings are built by the text decoder embedding table, then image feature rows replace placeholder token embedding rows via `masked_scatter`.
- Prefill: causal decoder consumes the stitched prefix and creates per-layer KV cache.
- Decode: source intentionally forwards `pixel_values` only on the first generation iteration, or when `use_cache=False`.

For first DinoML integration, target Mistral Small 3.1/3.2 native image-text and text-only generation. Treat `model_type=mistral3` configs whose `text_config.model_type` is `ministral3` as a separate admission class until the `ministral3` decoder audit is accepted.

## 3. Important config dimensions

Default `Mistral3Config` matches the Small 3.x shape: Pixtral vision + Mistral decoder, `image_token_index=10`, `spatial_merge_size=2`, `projector_hidden_act=gelu`, `multimodal_projector_bias=false`, `tie_word_embeddings=true`.

| Field | Mistral Small 3.1/3.2 value | Source |
| --- | ---: | --- |
| Text hidden size | 5120 | `config.json` |
| Text layers | 40 | `config.json` |
| Text attention heads / KV heads | 32 / 8 | `config.json` |
| Text head dim | 128 | `config.json` |
| Text intermediate size | 32768 | `config.json` |
| Text vocab size | 131072 | `config.json` |
| Text max positions | 131072 | `config.json` |
| Text RoPE theta | 1000000000.0 | `config.json` |
| Text sliding window | null | `config.json` |
| Vision hidden size | 1024 | `config.json` |
| Vision layers | 24 | `config.json` |
| Vision heads / head dim | 16 / 64 | `config.json` |
| Vision intermediate size | 4096 | `config.json` |
| Vision image size / patch size | 1540 / 14 | `config.json` |
| Spatial merge size | 2 | `config.json` / processor |
| Effective image token grid | `(H/28) * (W/28)` feature rows; text placeholders include break/end tokens | source + processor |
| dtype | bfloat16 | `config.json` |

Representative checkpoint sweep:

| Checkpoint | Wrapper | Text core | Text h/layers/heads/KV | Context/RoPE | Vision h/layers/patch | Quantization | Scope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Mistral-Small-3.1-24B-Instruct-2503` | mistral3 | mistral | 5120 / 40 / 32 / 8 | 131072, default theta 1e9 | 1024 / 24 / 14 | none in config | primary |
| `Mistral-Small-3.1-24B-Base-2503` | mistral3 | mistral | same | same | same | none in config | same ops, base weights |
| `Mistral-Small-3.2-24B-Instruct-2506` | mistral3 | mistral | same | same | same | none in config | same native ops; processor files absent from raw main during audit |
| `Mistral-Medium-3.5-128B` | mistral3 | ministral3 | 12288 / 88 / 96 / 8 | 262144, YaRN | 1664 / 48 / 14 | fp8 compressed-tensors metadata | wrapper only here; defer text core |
| `Ministral-3-3B-Instruct-2512` | mistral3 | ministral3 | 3072 / 26 / 32 / 8 | 262144, YaRN | 1024 / 24 / 14 | fp8 compressed-tensors metadata | wrapper only here; defer text core |

## 3a. Family variation traps

- Wrapper `model_type=mistral3` does not guarantee the text body is `mistral`; newer official configs use `text_config.model_type=ministral3`.
- `hidden_size` is not always equal to attention output width by inference alone; use explicit `head_dim`. Small text uses `32 * 128 = 4096` Q/O width into a 5120 hidden model, so Q projection is `5120 -> 4096`, K/V are `5120 -> 1024`, and O is `4096 -> 5120`.
- GQA is required: `num_key_value_heads=8 < num_attention_heads=32` for Small, repeat factor 4.
- The projector does not replace break/end tokens. The processor expands one image marker into image-token rows plus row-break/end markers, but the model only scatters features into token id 10 (`[IMG]`).
- Vision attention is block-diagonal by image in eager/SDPA style paths. Flash attention uses `position_ids` and `attention_mask=None`; DinoML should not accidentally permit cross-image attention unless the backend has equivalent varlen/block separation.
- Vision source is NCHW: `pixel_values[B,3,H,W]`, `Conv2d`, then flatten. NHWC is only a guarded fusion/layout opportunity around patch embedding.
- `Mistral3PatchMerger` uses `unfold` on a `C,H,W` grid with `kernel_size=stride=spatial_merge_size`; H and W after vision patching must be divisible by `spatial_merge_size`. The processor enforces this by using `patch_size * spatial_merge_size` during resizing.
- Config key drift: Small configs use top-level `torch_dtype`; newer configs use `dtype` and may include `quantization_config`.
- Newer configs include FP8 quantization metadata with `modules_to_not_convert` for vision tower, projector, and `lm_head`; that is a loading/provider contract, not an operator in the unquantized graph.
- `vision_feature_layer` can be an int or list. A list concatenates selected hidden states along the last dim before projector `linear_1`.
- Weight conversion permutes Q/K projection weights for the RoPE formulation. DinoML loaders should treat HF weight layout as already converted, but native Mistral-format import must preserve that transform.

## 4. Operator coverage checklist

Tensor/layout ops:

- Text embedding lookup, reshape/view, transpose, contiguous, concat/split, tensor indexing, arange, mask creation.
- Vision NCHW `Conv2d(3 -> vision_hidden, kernel=patch, stride=patch, bias=false)`.
- Vision crop of padded patch maps using `image_sizes // patch_size`, flatten each image as `p.flatten(1).T`, concatenate images into one sequence with batch size 1.
- Patch merger: split by per-image token counts, `view(h,w,d)`, `permute(2,0,1)`, `unsqueeze`, `unfold(kernel=stride=spatial_merge_size)`, transpose, concat.
- Placeholder mask: `input_ids == image_token_id`, expand to embedding shape, count/numel guard, `masked_scatter`.

Neural primitives:

- RMSNorm with fp32 variance accumulation for text, vision, and projector.
- Bias-free Linear projections unless `multimodal_projector_bias=true`.
- Small text MLP: `gate_proj Linear(5120 -> 32768)`, `up_proj Linear(5120 -> 32768)`, SiLU, multiply, `down_proj Linear(32768 -> 5120)`.
- Vision MLP: `gate/up Linear(1024 -> 4096)`, activation from config (`silu` in checkpoint, `gelu` in source default), multiply, `down Linear(4096 -> 1024)`.
- Projector: RMSNorm(vision hidden), patch merge Linear(`vision_hidden * merge^2 -> vision_hidden`), `linear_1(vision_hidden * num_feature_layers -> text_hidden)`, GELU, `linear_2(text_hidden -> text_hidden)`.
- LM head: `Linear(text_hidden -> vocab_size, bias=false)` with tied weight alias to text token embeddings when `tie_word_embeddings=true`.

Attention primitives:

- Text causal GQA with RoPE, KV cache, optional sliding-window mask, backend dispatch to eager/SDPA/Flash/Flex.
- Vision noncausal full self-attention per image block, no KV cache.
- Eager attention softmax upcasts to fp32 before downcast.

Position/rotary/custom math:

- Text 1D RoPE from `MistralRotaryEmbedding`, default or configured `rope_parameters`.
- Pixtral 2D mesh RoPE table indexed by `row * max_width + col`, with separate row/column frequencies interleaved into head dim.

Generation/cache ops:

- Dynamic KV cache per text layer. Small cache shape per layer before repeat: K/V `[batch, 8, seq, 128]`; attention backend sees repeated K/V as `[batch, 32, seq, 128]` in eager.
- `prepare_inputs_for_generation` must drop `pixel_values` after first cached iteration.
- `logits_to_keep` supports last-token-only or tensor index slicing.

Preprocessing-coupled ops:

- BICUBIC resize, RGB conversion, rescale by `1/255`, CLIP mean/std normalize, right/bottom zero pad, channel-first output.
- Prompt expansion using `[IMG]`, `[IMG_BREAK]`, `[IMG_END]`; tokenizer special token ids are part of the ABI.

Quantized/packed metadata:

- Native Small configs have no quantization metadata. Newer official configs advertise FP8 with dense exclusions. DinoML should reject or route these until compressed-tensors FP8 loading and dequant/materialization policy is audited.

## 5. Layer/block breakdown

Vision patch + transformer:

```text
pixel_values[B,3,Hpad,Wpad]
patch = Conv2d(3 -> V, kernel=P, stride=P, bias=false)
patch_i = patch[i, :, :Hi/P, :Wi/P]
tokens = concat_i(flatten_C_HW(patch_i).T)[None, total_patches, V]
tokens = RMSNorm(tokens)
position_ids = row * (image_size/P) + col for each valid patch
for 24 vision layers:
  residual = tokens
  tokens = RMSNorm(tokens)
  q,k,v = Linear(V -> V), bias=false
  q,k = Pixtral 2D RoPE(q,k)
  tokens = residual + Attention(noncausal block-per-image)
  residual = tokens
  tokens = RMSNorm(tokens)
  tokens = residual + down(act(gate(tokens)) * up(tokens))
```

Small 3.x projector:

```text
features[sum(Hi/P*Wi/P), 1024]
features = RMSNorm(features)
for each image:
  grid = features.view(Hi/P, Wi/P, 1024).permute(2,0,1)[None]
  windows = unfold(grid, kernel=2, stride=2).T       # [Hi/28 * Wi/28, 4096]
merged = Linear(4096 -> 1024, bias=false)(windows)
hidden = Linear(1024 -> 5120, bias=false)(merged)
hidden = GELU(hidden)
hidden = Linear(5120 -> 5120, bias=false)(hidden)
```

Small 3.x decoder block, repeated 40 times:

```text
x = residual stream [B,T,5120]
h = RMSNorm(x)
q = Linear(5120 -> 4096, bias=false)(h).view(B,T,32,128).transpose(1,2)
k = Linear(5120 -> 1024, bias=false)(h).view(B,T,8,128).transpose(1,2)
v = Linear(5120 -> 1024, bias=false)(h).view(B,T,8,128).transpose(1,2)
q,k = 1D RoPE(q,k)
k,v = cache.update(k,v,layer)
attn = causal GQA(q,k,v, mask, repeat_kv=4)
x = x + Linear(4096 -> 5120, bias=false)(attn)
h = RMSNorm(x)
mlp = Linear(32768 -> 5120)(SiLU(Linear(5120 -> 32768)(h)) * Linear(5120 -> 32768)(h))
x = x + mlp
```

Final:

```text
x = RMSNorm(x)
logits = Linear(5120 -> 131072, bias=false)(x[:, slice_indices, :])
```

## 6. Attention requirements

Text attention:

- Causal self-attention only; no cross-attention.
- GQA for Small: 32 query heads, 8 KV heads, head dim 128, repeat factor 4.
- Q width 4096, K/V width 1024, O input width 4096, hidden width 5120.
- Mask is created by `create_causal_mask` when `sliding_window=None`; if a supported config sets `sliding_window`, source uses `create_sliding_window_causal_mask` and passes `sliding_window` to the backend.
- RoPE is applied before cache update; cached keys are post-RoPE.
- Eager fallback repeats K/V before matmul and softmaxes in fp32. Optimized backends should preserve scaling `head_dim ** -0.5`, mask addition before softmax, and dropout only in training.
- KV cache per layer stores K/V before repeat expansion. For Small, per-layer K/V are `[B,8,S,128]`.

Vision attention:

- Noncausal self-attention, MHA not GQA: 16 heads, head dim 64 for Small.
- Attention spans patches within each original image only. Eager mask is `[1,1,total_patches,total_patches]` with dtype minimum outside same-image blocks.
- Flash attention path uses no dense mask in source. Admission should require backend-equivalent image sequence boundary handling or fall back to dense/block mask.
- No KV cache for vision; vision/projector outputs can be cached as prefix embeddings independent of text KV.

## 7. Position encoding and custom math

Text RoPE is standard Mistral/Llama-style:

```python
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = torch.cat((freqs.transpose(1, 2), freqs.transpose(1, 2)), dim=-1)
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
q = q * cos[:, None] + rotate_half(q) * sin[:, None]
k = k * cos[:, None] + rotate_half(k) * sin[:, None]
```

Pixtral vision RoPE is source-specific 2D mesh indexing:

```python
max_side = image_size // patch_size
freqs = 1.0 / (theta ** (arange(0, head_dim, 2) / head_dim))
freqs_h = outer(arange(max_side), freqs[::2])
freqs_w = outer(arange(max_side), freqs[1::2])
table = concat(row_terms, col_terms, dim=-1).reshape(-1, head_dim // 2)
table = concat(table, table, dim=-1)
position_id = row * max_side + col
cos, sin = cos(table[position_id]), sin(table[position_id])
```

Precompute opportunities:

- Text default RoPE inverse frequencies and cos/sin tables can be cached by sequence length and dtype/device. Dynamic/YaRN configs from `ministral3` must use that family's rope parameters.
- Vision mesh RoPE table is bounded by `vision_config.image_size // patch_size`; per-request dynamic work is selecting valid patch positions for each image.

## 8. Preprocessing and input packing

Image processor ABI for Small 3.1 snapshots:

- Input images are converted to RGB, resized with bicubic resampling while preserving aspect ratio, and aligned to `patch_size * spatial_merge_size = 28`.
- Longest edge is capped at 1540.
- Pixel values are rescaled by `0.00392156862745098`, normalized with mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- Output layout is channel-first `pixel_values[B,3,Hpad,Wpad]`.
- `image_sizes` records each resized unpadded `(height,width)`.
- Batching pads each image to max resized height/width with zeros on bottom/right.

Processor/token packing:

- `PixtralProcessor` has `image_token="[IMG]"`, `image_break_token="[IMG_BREAK]"`, `image_end_token="[IMG_END]"`, `patch_size=14`, `spatial_merge_size=2`.
- For each `[IMG]` in text, processor replaces it with `(num_width_tokens * [IMG] + [IMG_BREAK])` repeated for each merged grid row, and changes the final token to `[IMG_END]`.
- `num_height_tokens = resized_height // 28`, `num_width_tokens = resized_width // 28`.
- The model replaces only `[IMG]` rows, not break/end rows, so feature rows per image are `num_height_tokens * num_width_tokens`.
- `masked_scatter` flatten order is row-major through the expanded text sequence and feature tensor. DinoML can lower to ordered indexed row copy if it verifies placeholder count equals feature row count and preserves tokenizer expansion order.

CPU/data pipeline owns image decode and resizing. GPU/runtime owns vision encoder, projector, embedding stitch, decoder, logits, and cache.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Pixtral patch Conv2d -> Linear

Source pattern: `Conv2d(3 -> V, kernel=P, stride=P, padding=0, dilation=1, groups=1, bias=false)` on NCHW.

Replacement:

```text
NCHW PatchExtract(P,P,stride=P) -> WindowFlatten(C*P*P) -> MatMul(W_flat.T)
  -> Reshape/flatten to token sequence
```

Preconditions: input height/width are multiples of `P`; no padding/dilation/groups/bias; flatten order matches PyTorch `Conv2d` NCHW kernel storage `[out,in,kh,kw]`; downstream valid-region crop by `image_sizes // P` is preserved.

Failure cases: dynamic partial patches, NHWC without matching weight permutation, or a backend that cannot preserve padded-region crop before sequence concat.

Parity sketch: random NCHW images with varied valid `image_sizes`; compare patch maps before and after crop/flatten.

### Rewrite: Patch merger unfold -> block gather + GEMM

Source pattern: `view(h,w,d) -> permute(d,h,w) -> unfold(kernel=M,stride=M) -> Linear(d*M*M -> d)`.

Replacement:

```text
GridBlockGather(row-major MxM neighborhoods) -> Flatten([d,M,M] in PyTorch unfold order)
  -> MatMul(merging_layer.weight.T)
```

Preconditions: `h % M == 0`, `w % M == 0`, `M=spatial_merge_size`, no padding, no overlap, source flatten order exactly matches `torch.nn.functional.unfold` on `[1,d,h,w]`.

Failure cases: processor not used, image sizes not aligned to `patch_size * M`, or a future config using `M != 2` without tested gather order.

### Rewrite: placeholder masked_scatter -> ordered indexed row copy

Source pattern: expanded boolean mask over `[B,T,H]` and `masked_scatter(mask, image_features)`.

Replacement: compute `indices = where(input_ids == image_token_id)` in row-major order, then copy feature rows to `inputs_embeds[indices]`.

Preconditions: count of `[IMG]` tokens equals feature rows; image feature concat order follows processor image order; no caller-supplied arbitrary `inputs_embeds` mask mode unless embedding-equality fallback is implemented.

Failure cases: user supplies `inputs_embeds` without `input_ids`, tokenizer special token ids mismatch, or prompts not expanded by `PixtralProcessor`.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode, materialize only final hidden row before LM head.

Preconditions: `logits_to_keep=1` or equivalent last-token slice; no loss computation over full labels.

### Layout opportunities and guards

- NCHW-to-NHWC may be profitable only inside the patch conv/normalization/pre-attention projection region if all axis rewrites are explicit. Source semantic graph should remain NCHW at boundaries.
- Guard `position_ids_in_meshgrid`, block attention mask generation, patch crop, and `unfold` as axis-sensitive; these are not safe for blind layout translation.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm, because it appears twice per decoder block plus final norm and controls decoder throughput.
- GQA FlashAttention/paged KV cache for `32q/8kv/head_dim128`; this dominates prefill/decode.
- Q/K/V projection + RoPE preparation, preserving non-square Q/K widths and post-RoPE cache storage.
- SwiGLU MLP fusion for `5120 -> 32768 -> 5120`.
- Last-token-only logits for decode against `vocab_size=131072`.

Medium priority:

- Vision patch Conv2d-to-GEMM and patch flatten/crop fusion.
- Vision noncausal block attention for many image patches; avoid dense cross-image work in batched multi-image prompts.
- Patch merger gather + Linear fusion.
- Projector RMSNorm + Linear/GELU/Linear.

Lower priority:

- Processor CPU throughput optimization.
- Multi-feature-layer projector path.
- FP8 compressed-tensors loading for newer Medium/Ministral checkpoints.

## 11. Runtime staging plan

Stage 1: parse `Mistral3Config`, nested Pixtral/Mistral configs, processor metadata, and reject unsupported `text_config.model_type` values unless routed.

Stage 2: load Small 3.x dense weights, including tied embedding/LM-head alias and converted HF Q/K layouts.

Stage 3: vision encoder parity on one resized image: patch conv, mesh RoPE, block attention, hidden state output.

Stage 4: projector parity: patch merger + MLP projection, including per-image split sizes.

Stage 5: embedding stitch parity using processor-expanded prompts; initially lower scatter as guarded indexed row copy.

Stage 6: text-only and multimodal prefill logits parity with dense/eager attention.

Stage 7: decode with DynamicCache-compatible KV ABI and first-iteration-only `pixel_values`.

Stage 8: enable optimized GQA attention, patch conv lowering, patch merger lowering, and last-token logits.

Stub initially: training loss, hidden-state/attention output capture, quantized FP8 configs, `ministral3` text core, 3.2 missing processor repo files if not supplied by caller.

## 12. Parity and validation plan

- Unit test Pixtral resize token-count helper on fixed image sizes: verify resized dims are multiples of 28 and placeholder counts match feature rows plus break/end tokens.
- Random tensor test RMSNorm fp32 accumulation for bf16/fp16/fp32.
- Random patch conv lowering parity for several padded images and valid `image_sizes`.
- Patch merger parity with small synthetic grids and `spatial_merge_size=2`; include non-square grids.
- Pixtral 2D RoPE parity by comparing cos/sin table indices for varied image sizes.
- Single vision layer parity, then full 24-layer vision parity on one small generated image tensor.
- Single decoder layer parity with GQA and cache disabled/enabled.
- Prefill logits parity for text-only prompt.
- Prefill logits parity for one image prompt after processor expansion.
- Decode token parity: first step with image, second step from cache without `pixel_values`.
- Recommended tolerances: fp32 `atol=1e-5, rtol=1e-5`; bf16/fp16 layer outputs `atol=2e-2, rtol=2e-2`, logits evaluated with top-k/token agreement plus numeric tolerance appropriate to backend.

## 13. Performance probes

- Processor throughput: images/sec for varied original sizes, measuring resize/pad/token expansion separately.
- Vision encoder throughput by total valid patch count and padded batch shape.
- Patch merger/projector throughput by number of merged image tokens.
- Prefill throughput for text-only versus image-text prompts at equal final sequence length.
- Decode tokens/sec for batch sizes 1, 4, 16 with cache.
- KV cache memory for Small: layers * 2 * B * KV_heads * seq * head_dim * dtype.
- Attention backend comparison: eager, SDPA, Flash, and DinoML fused GQA.
- Vision block-attention probe: single large image versus many small images with same total patch count.
- LM-head probe for full logits versus last-token-only logits.
- FP8 load/dequant probe only after quantization admission exists.

## 14. Skip/defer list

- Training, gradient checkpointing, and loss parity.
- Beam search and speculative decoding.
- Returning dense attention tensors or all hidden states in production path.
- `ministral3` text-core checkpoints inside the `mistral3` wrapper until separately admitted.
- FP8/compressed-tensors official newer checkpoints.
- Arbitrary caller-supplied `inputs_embeds` image placeholder detection by embedding equality.
- Non-default vision RoPE: Pixtral source raises for non-default vision `rope_type`.
- Multi-GPU tensor parallel and pipeline parallel plans.

## 15. Final implementation checklist

- [ ] Parse wrapper config and nested `vision_config` / `text_config`.
- [ ] Add admission guard for `text_config.model_type in {"mistral"}` for first Small 3.x target.
- [ ] Load dense weights and preserve tied `lm_head.weight` / `embed_tokens.weight` alias when configured.
- [ ] Implement Pixtral image processor ABI or require preprocessed `pixel_values` + `image_sizes`.
- [ ] Implement Pixtral patch embedding and valid-region crop/flatten.
- [ ] Implement Pixtral 2D mesh RoPE.
- [ ] Implement vision block self-attention.
- [ ] Implement patch merger unfold/gather + Linear.
- [ ] Implement multimodal projector.
- [ ] Implement guarded `[IMG]` indexed row copy.
- [ ] Implement Mistral GQA causal decoder with RoPE and KV cache.
- [ ] Implement `prepare_inputs_for_generation` pixel-values first-iteration behavior.
- [ ] Add text-only prefill/decode parity tests.
- [ ] Add single-image and multi-image prefill parity tests.
- [ ] Add patch conv and patch merger rewrite parity tests.
- [ ] Benchmark vision, prefill, decode, and LM-head slices separately.
