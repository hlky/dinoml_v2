# Pixtral Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model ids/configs inspected:

| Source | File | Scope notes |
| --- | --- | --- |
| `mistralai/Pixtral-12B-Base-2409` | `_sources/mistralai_Pixtral-12B-Base-2409_params.json` | Official Mistral-format `params.json`, no HF `config.json` in repo listing. |
| `mistralai/Pixtral-12B-2409` | `_sources/mistralai_Pixtral-12B-2409_params.json` | Official Mistral-format instruct params; same operator-significant dimensions as base in the snapshot. |
| `mistralai/Pixtral-Large-Instruct-2411` | `_sources/mistralai_Pixtral-Large-Instruct-2411_config.json` | Official HF `LlavaForConditionalGeneration` config snapshot at commit `6ca6050117ca15da33ab3071bfa888107d589f50`; repo API listing also exposes `params.json`. |
| `unsloth/Pixtral-12B-2409` | `_sources/unsloth_Pixtral-12B-2409_{config,preprocessor_config,processor_config}.json` | Open HF-format mirror; used only to inspect converted HF config/processor metadata. |
| `mistral-community/pixtral-12b` | `_sources/mistral-community_pixtral-12b_{config,preprocessor_config,processor_config}.json` | Open HF-format community conversion; some fields rely on Transformers defaults. |

Source files inspected:

- `src/transformers/models/pixtral/configuration_pixtral.py`
- `src/transformers/models/pixtral/modeling_pixtral.py`
- `src/transformers/models/pixtral/processing_pixtral.py`
- `src/transformers/models/pixtral/image_processing_pixtral.py`
- `src/transformers/models/pixtral/image_processing_pil_pixtral.py`
- `src/transformers/models/pixtral/convert_pixtral_weights_to_hf.py`
- Delegated wrapper/decoder files: `models/llava/{configuration_llava.py,modeling_llava.py}` and `models/mistral/{configuration_mistral.py,modeling_mistral.py}`.

Any missing files or assumptions: Pixtral itself only registers `PixtralVisionModel` under `model_type="pixtral"`. HF multimodal generation is represented as `model_type="llava"` with `vision_config.model_type="pixtral"` and `text_config.model_type="mistral"`. The Mistral modeling file is generated from `modular_mistral.py`; future source edits should inspect the modular file too. Official Pixtral-12B repos are Mistral-format, so converted HF metadata is partly from mirrors and the local conversion script.

## 2. High-level architecture

Primary runtime target: multimodal autoregressive generation.

Dataflow:

```text
CPU image/text preprocessing
  -> Pixtral vision tower over packed variable-size images
  -> LLaVA multimodal projector
  -> masked-scatter image embeddings into token embeddings
  -> delegated Mistral decoder prefill
  -> delegated Mistral decode with KV cache
  -> LM head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize longest edge to at most 1024, patch-aligned output size, rescale/normalize, pad images in batch to max `H,W`, tokenize text after expanding `[IMG]` placeholders into row-major image token strings.
- Vision tower: noncausal Pixtral ViT-like encoder. It accepts padded NCHW `pixel_values` plus `image_sizes`, crops convolution outputs back to each image's true patch grid, concatenates all image patches into one sequence with batch dimension 1, and uses block-diagonal attention so images do not attend across image boundaries.
- Projector: LLaVA two-layer MLP maps vision hidden size to text hidden size.
- Prefix construction: LLaVA concatenates projected image features and inserts them into `inputs_embeds` at `image_token_id` positions via `masked_scatter`.
- Decoder: Mistral causal LM owns text RoPE, GQA, optional sliding-window attention, DynamicCache, final RMSNorm, and untied `lm_head`.

Independently stageable units: image processor token-count logic; Pixtral vision tower; projector; image-token scatter; Mistral prefill/decode; final logits slicing. Vision/projector outputs can be cached independently for a fixed image prompt, but the native generation cache is the delegated Mistral self-attention KV cache after image embeddings have been stitched into the prefix.

## 3. Important config dimensions

| Field | Pixtral-12B official params | Pixtral-12B HF mirror | Pixtral Large HF config | Source |
| --- | ---: | ---: | ---: | --- |
| wrapper architecture | Mistral-format conversion target | `LlavaForConditionalGeneration` | `LlavaForConditionalGeneration` | config/params |
| text hidden size | 5120 | 5120 | 12288 | config/params |
| text layers | 40 | 40 | 88 | config/params |
| text attention heads | 32 | 32 | 96 | config/params |
| text KV heads | 8 | 8 | 8 | config/params |
| text head dim | 128 | 128 | inferred 128 unless explicit `head_dim` absent | config/source default |
| text intermediate | 14336 | 14336 | 28672 | config/params |
| vocab size | 131072 | 131072 | 32768 | config/params |
| max positions | 131072 in official params, 1024000 in conversion defaults/mirrors | 1024000 | 131072 | config/params |
| text RoPE theta | 1e9 | 1e9 | 1e9 | config/params |
| text sliding window | absent/null for 12B | null | 4096 | config |
| vision hidden size | 1024 | 1024 | 1408 | config/params |
| vision layers | 24 | 24 | 40 | config/params |
| vision attention heads | 16 | 16 | source default would be 16 if omitted; `head_dim=88` implies 16 heads for hidden 1408 | config/source inference |
| vision head dim | 64 | 64 | 88 | config/inference |
| vision intermediate | 4096 | 4096 | 6144 | config/params |
| image size / patch | 1024 / 16 | 1024 / 16 | 1024 / 16 | config/params |
| vision activation | official params omit, conversion script fills `silu`; one mirror says `gelu` | `gelu` in unsloth, `silu` in community | `silu` | config/conversion |
| projector | two Linear layers with GELU | same | same | LLaVA source |
| projector bias | `adapter_bias` default true in converter | config default true unless overridden | config default true unless omitted | source default/config |
| dtype | not in official params | `bfloat16` | `bfloat16` | config |

Effective defaults to watch: `PixtralVisionConfig` defaults `hidden_act="gelu"`, `num_attention_heads=16`, `num_hidden_layers=24`, `hidden_size=1024`, `intermediate_size=4096`, `num_channels=3`, and derives `head_dim = hidden_size // num_attention_heads`. `MistralConfig` defaults `sliding_window=4096`, but converted 12B HF configs set it to null; do not assume sliding-window attention unless the active text config has it.

## 3a. Family variation traps

- The owned `pixtral` model is vision-only. End-to-end generation uses the LLaVA wrapper and Mistral decoder; route those delegated subgraphs through their own coverage rather than treating them as Pixtral-native code.
- Official 12B repos use Mistral-format `params.json`; HF `LlavaConfig` snapshots may be converted by community/mirror code. Label 12B HF conversion fields separately from official params when they disagree, especially `max_position_embeddings` and `hidden_act`.
- `image_seq_length=1` in LLaVA config is not the real number of image patch embeddings. Pixtral expands placeholders dynamically from resized image sizes: `(num_width_patches + 1) * num_height_patches` text tokens, but the model scatters only `[IMG]` feature positions into embeddings. `[IMG_BREAK]` and `[IMG_END]` are text tokens, not vision features.
- Variable-size image support is processor plus vision-forward behavior: padded NCHW tensors are cropped by `image_sizes` after Conv2d patching; patch sequences are concatenated into a single batch-1 sequence.
- Vision attention is block-diagonal across images for eager/SDPA-style paths. Flash-attention requested path sets the explicit mask to `None` and relies on `position_ids`; DinoML should validate whether backend varlen/block semantics are needed before enabling a no-mask flash path.
- Pixtral vision RoPE supports only `rope_type="default"` and precomputes a 2D frequency table for `image_size // patch_size` squared positions. Non-default vision RoPE config should be rejected for this source path.
- Text decoder uses GQA: `num_key_value_heads < num_attention_heads` for all inspected configs. KV cache is stored before repeat expansion.
- Large config has `sliding_window=4096`; 12B configs/mirrors set null. This changes causal-mask construction and optimized attention requirements.
- Source image tensors are NCHW. NHWC/channel-last is an optimization candidate only inside controlled Conv2d/patching and normalization regions; placeholder expansion, crop axes, flatten order, and block-size math are axis-sensitive.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image pad to max batch `H,W`; per-image crop of patch Conv2d output by `image_sizes // patch_size`.
- Conv2d patch embedding: `in_channels=3`, `out_channels=vision_hidden`, `kernel_size=stride=patch_size=16`, `bias=False`.
- Flatten/transpose per image: `[C,Hpatch,Wpatch] -> [Hpatch*Wpatch,C]`; concatenate all images over sequence; unsqueeze batch to `[1,total_patches,C]`.
- Token embedding lookup; boolean mask creation from `input_ids == image_token_id`; `masked_scatter`/indexed copy of projected image features into `inputs_embeds`.
- Logits slicing via `logits_to_keep`.

Neural network primitives:

- RMSNorm with fp32 variance and epsilon `1e-5` in Pixtral vision and Mistral configs inspected.
- Bias-free Linear Q/K/V/O projections in vision attention.
- Vision gated MLP: `down_proj(act(gate_proj(x)) * up_proj(x))`; activation config-dependent (`silu` for official conversion defaults/large, `gelu` in one mirror).
- LLaVA projector: `Linear(vision_hidden * num_selected_layers -> text_hidden, bias=multimodal_projector_bias) -> GELU -> Linear(text_hidden -> text_hidden, bias=...)`.
- Mistral decoder RMSNorm, GQA projections, SwiGLU MLP, final RMSNorm, untied LM head.

Attention primitives:

- Vision noncausal MHA with full Q/K/V head count and 2D patch RoPE.
- Optional block-diagonal additive mask for concatenated images.
- Text causal GQA with RoPE, optional sliding window, DynamicCache, and backend dispatch through `ALL_ATTENTION_FUNCTIONS`.

Position/custom math:

- Pixtral 2D RoPE table indexed by `h * max_width + w`.
- Mistral 1D RoPE with `rope_theta` from text config.

Preprocessing-coupled ops:

- Resize preserving aspect ratio to longest edge <= 1024, then round up dimensions to patch multiples.
- Rescale by `1/255`, normalize by CLIP-style mean/std, RGB conversion.
- Placeholder token expansion with `[IMG]` repeated per patch row, `[IMG_BREAK]` at row ends, final token replaced by `[IMG_END]`.

Generation/cache ops:

- Mistral `DynamicCache` per decoder layer, shape logically `[batch, num_key_value_heads, cached_seq, head_dim]` for K and V before repeat to query heads.
- `prepare_inputs_for_generation` forwards `pixel_values` only on the first generation iteration, or when `use_cache=False`.

## 5. Layer/block breakdown

Pixtral vision forward:

```text
pixel_values: [B,3,Hpad,Wpad], image_sizes: list[(Hi,Wi)]
patch = Conv2d(3 -> V, kernel=16, stride=16, bias=False)
patch_i = patch[i, :, :Hi/16, :Wi/16]
seq = concat_i(patch_i.flatten(1).T) -> [total_patches,V]
x = RMSNorm(seq.unsqueeze(0))
position_ids = concat_i(meshgrid(Hi/16, Wi/16, id=h*64+w))
cos,sin = Pixtral2DRoPE(position_ids)
mask = block_diagonal(total_patches per image) unless flash attention requested
repeat L_v times:
  residual = x
  x = RMSNorm(x)
  q,k,v = Linear(V -> V, bias=False)
  q,k = 2D RoPE(q,k)
  x = residual + Attention(q,k,v, block_mask)
  residual = x
  x = RMSNorm(x)
  x = residual + Linear(I -> V)(act(Linear(V -> I)(x)) * Linear(V -> I)(x))
return last_hidden_state: [1,total_patches,V]
```

Projector and stitch:

```text
selected_image_feature = vision_hidden_states[-1]  # strategy "full" for Pixtral configs
image_features = Linear(V -> T) -> GELU -> Linear(T -> T)
if image_sizes provided: split by prod(image_sizes // patch_size)
flat_features = cat(split_features, dim=0)
text_embeds = embed_tokens(input_ids)
special_image_mask = input_ids == image_token_id
inputs_embeds = masked_scatter(text_embeds, special_image_mask[...,None], flat_features)
```

Delegated Mistral decoder block, repeated `num_hidden_layers`:

```text
residual = x
x = RMSNorm(x)
q = Linear(T -> n_heads*head_dim)
k,v = Linear(T -> n_kv_heads*head_dim)
q,k = 1D RoPE(q,k)
k,v = cache.update(k,v, layer_idx) when caching
x = residual + GQA_Attention(q,k,v, causal/sliding mask)
residual = x
x = RMSNorm(x)
x = residual + Linear(I -> T)(silu(Linear(T -> I)(x)) * Linear(T -> I)(x))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention, MHA, no KV cache.
- Shapes: q/k/v `[1, num_heads, total_patches, head_dim]`; for 12B vision `heads=16`, `head_dim=64`; for Large inferred `heads=16`, `head_dim=88`.
- Eager math: `matmul(q,k^T) * head_dim^-0.5`, add block mask, softmax in fp32, cast back, dropout only in training, matmul with V, output projection.
- Mask semantics: concatenated images attend only within their own patch block. First integration should implement explicit block masks; enabling flash-attention without a mask needs a separate parity check for multi-image prompts.

Text attention:

- Causal self-attention, GQA. Cache stores K/V as `[batch, n_kv_heads, seq, head_dim]`; eager attention repeats to `[batch, n_heads, seq, head_dim]`.
- RoPE is applied to q/k before `past_key_values.update`, so cached keys are already position encoded.
- Large requires sliding-window causal masking (`sliding_window=4096` in inspected HF config). 12B converted configs may use full causal masking.
- Optimized dispatch is through Transformers attention interfaces: eager fallback, SDPA, FlashAttention, FlexAttention where supported.

## 7. Position encoding and custom math

Pixtral 2D RoPE precompute, source-equivalent sketch:

```python
max_side = image_size // patch_size
freqs = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs_h = outer(arange(max_side), freqs[::2])
freqs_w = outer(arange(max_side), freqs[1::2])
table = cat([
    repeat(freqs_h[:, None, :], width=max_side),
    repeat(freqs_w[None, :, :], height=max_side),
], dim=-1).reshape(max_side * max_side, head_dim // 2)
inv_freq = cat([table, table], dim=-1)
position_id = h * max_side + w
cos, sin = cos(inv_freq[position_id]), sin(inv_freq[position_id])
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

Precomputable: 2D vision RoPE table for a fixed `image_size`, `patch_size`, `head_dim`, `rope_theta`; text RoPE inverse frequencies. Dynamic inputs: actual `position_ids` from each image patch grid; text `position_ids` offset by cache length.

## 8. Preprocessing and input packing

Image processor contract:

- Input images become torch tensors in channel-first layout.
- Resize preserves aspect ratio if either side exceeds `size["longest_edge"]`; dimensions are then rounded up to patch multiples using ceil-like patch token counts.
- Pixel values are rescaled, normalized, then padded on right/bottom to batch max `H,W`.
- Outputs: `pixel_values` `[B,3,Hmax,Wmax]` and `image_sizes` list/tensor of true resized `(height,width)` per image.

Processor placeholder contract:

- `processor_config`: `image_token="[IMG]"`, `image_break_token="[IMG_BREAK]"`, `image_end_token="[IMG_END]"`, `patch_size=16`, `spatial_merge_size=1` unless overridden.
- For each `[IMG]` in text, the processor consumes the next resized image size. Replacement order is row-major:

```text
for each patch row:
  [IMG] repeated num_width_patches, then [IMG_BREAK]
replace final row break with [IMG_END]
```

- `_get_num_multimodal_tokens` reports `(num_width_patches + 1) * num_height_patches`, including break/end text tokens. The model's feature scatter expects only the `[IMG]` positions to match projected patch features.
- Image features/prefix embeddings can be cached independently from decoder KV for repeated prompts with the same image and projector weights.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap patch Conv2d -> GEMM

Source pattern: `Conv2d(C -> V, kernel=patch, stride=patch, padding=0, dilation=1, groups=1, bias=False)` followed by true-size crop, flatten, transpose.

Replacement:

```text
PatchExtractNCHW(row-major patches) -> MatMul(flat_patch, W.T) -> concatenate sequences
```

Preconditions: NCHW contiguous or explicit strides supported; `H,W` after processor are multiples of patch size; crop uses `image_sizes // patch_size`; no bias; no overlapping patches. Weight transform: `conv.weight.reshape(V, C * patch_h * patch_w)`. Failure cases: arbitrary Conv2d, nonzero padding/dilation/groups, channel-last tensors without a verified patch flatten order.

Parity test: random padded batch with different `image_sizes`, compare source Conv2d+crop+flatten against rewrite for each image and concatenated sequence.

### Rewrite: projector MLP canonicalization

Source pattern: `Linear(V->T,bias) -> GELU -> Linear(T->T,bias)`.

Replacement: normal DinoML GEMM bias GELU epilogue where bias exists, then GEMM bias. Preconditions: selected vision layer count known; feature select strategy `full`; dtype tolerance set for bf16/fp16. Failure cases: multiple selected layers concatenate hidden states, different activation, missing projector bias.

### Rewrite: image placeholder stitch -> indexed copy

Source pattern: boolean mask from `input_ids == image_token_id`, expanded to embed shape, `masked_scatter`.

Replacement: precompute image token positions, validate count equals `sum((Hi/16)*(Wi/16))`, then indexed row copy into token embeddings. Preconditions: mask is 1D over sequence per batch; image features are already flat in prompt order. Failure cases: `inputs_embeds` path without `input_ids`, duplicate special-token embedding comparison path, batched prompts with independent image counts until packing metadata is explicit.

### Layout optimization: guarded NCHW -> NHWC patch region

Candidate region: image processor output through patch projection only. Required axis rewrites: Conv2d layout, pad dimensions, crop dimensions, flatten order, and patch-extract weight transform. Consumer contract after rewrite must still produce `[total_patches,V]`. Protect placeholder/token logic, sequence concat, masks, and decoder with a conceptual no-layout-translation guard until they have layout-agnostic lowering.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for both vision and Mistral decoder, fp32 variance with reduced-precision storage.
- GQA FlashAttention/paged attention with RoPE and KV cache for delegated Mistral, including sliding-window large variant.
- Vision patch Conv2d-to-GEMM or direct patch projection kernel with per-image crop/pack.
- Indexed image embedding stitch replacing `masked_scatter`.

Medium priority:

- Vision QKV projection + 2D RoPE preparation.
- Vision block-diagonal attention for multi-image prefill.
- SwiGLU/GELU-gated MLP fusion for vision and text.
- Projector GEMM+GELU fusion.
- Last-token-only LM head via `logits_to_keep`.

Lower priority:

- CPU-side resize/normalize acceleration inside DinoML runtime. This can remain data pipeline initially.
- Vision encoder output caching across repeated prompts.
- Multi-selected-layer projector support; inspected Pixtral configs use `vision_feature_layer=-1`.

## 11. Runtime staging plan

Stage 1: parse official Mistral-format params and HF `LlavaConfig`; load Pixtral vision weights and projector weights with clear provenance.

Stage 2: implement image processor parity and placeholder expansion/token-count validation outside compiled runtime.

Stage 3: run Pixtral vision tower parity for one image and multi-image variable-size batches using explicit block masks.

Stage 4: add projector and indexed image-feature stitch into delegated Mistral token embeddings.

Stage 5: compose with existing/parallel Mistral decoder prefill parity, including GQA and full causal cache.

Stage 6: add decode with KV cache; ensure `pixel_values` is only used on the first generation iteration when cache is enabled.

Stage 7: enable optimized attention paths: text FlashAttention/paged cache first, then vision block/varlen attention and patch projection rewrite.

Stub initially: sampling controller, chat template, Mistral-common tokenizer implementation, training losses, multi-GPU tensor parallel, and CPU image preprocessing inside DinoML.

## 12. Parity and validation plan

- Processor tests: resize output sizes for odd aspect ratios; verify dimensions are patch multiples; verify placeholder expansion for one image, multiple images, and mismatched image-token count.
- Patch projection tests: random image tensors with padded batch and `image_sizes`; compare Conv2d crop/flatten sequence.
- 2D RoPE tests: compare `position_ids_in_meshgrid`, cos/sin table indexing, and `apply_rotary_pos_emb` on random q/k.
- Vision block tests: one Pixtral attention layer and full vision tower against Transformers for fp32, then bf16/fp16 tolerances.
- Projector/stitch tests: compare projected features and indexed-copy replacement against `masked_scatter`; include `[IMG_BREAK]` and `[IMG_END]` positions to prove they are not replaced by image features.
- Prefill logits: one image prompt through HF LLaVA/Pixtral wrapper vs DinoML-composed graph.
- Decode token parity: first decode step with `past_key_values`; confirm no second vision forward when cache is enabled.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=5e-2, atol=5e-2` initially, tighten after fused kernels are stable.

## 13. Performance probes

- CPU preprocessing throughput by image resolution and batch image count.
- Patch projection throughput for variable `H,W`, separated from vision transformer time.
- Vision encoder throughput for one image vs many images in one prompt; compare explicit block mask vs backend varlen path.
- Projector + stitch time as a function of total image patches and text length.
- Mistral prefill tokens/sec for text-only vs image-augmented prompt lengths.
- Decode tokens/sec with KV cache, including large sliding-window config.
- KV cache memory: `layers * 2 * batch * n_kv_heads * seq * head_dim * dtype_size`, separated from independently cacheable image embeddings.
- LM head cost with `logits_to_keep=1` vs full prefix logits.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Beam search and advanced generation processors beyond basic greedy/sampling parity.
- Mistral-common tokenizer/chat-template exactness inside compiled runtime.
- Multi-GPU tensor parallel and pipeline parallel execution.
- Non-default Pixtral vision RoPE; source rejects it.
- Arbitrary `vision_feature_layer` lists until a checkpoint requires multi-layer concatenation.
- NHWC global layout translation; use guarded local patch-region optimization only.
- Direct support for official Mistral-format runtime loading can be staged after HF-converted weight loading unless product requirements prioritize native Mistral repos.

## 15. Final implementation checklist

- [ ] Parse Pixtral official params and HF LLaVA/Pixtral configs with provenance labels.
- [ ] Load Pixtral vision, projector, and delegated Mistral weights without breaking tied/untied LM-head assumptions.
- [ ] Implement Pixtral image resize/pad/normalize metadata path.
- [ ] Implement placeholder expansion and image-token count validation.
- [ ] Implement NCHW patch projection plus per-image crop/flatten/concatenate.
- [ ] Implement Pixtral 2D RoPE table and meshgrid position IDs.
- [ ] Implement vision block-diagonal attention mask.
- [ ] Implement Pixtral RMSNorm, noncausal MHA, and gated MLP parity.
- [ ] Implement LLaVA projector and image-feature indexed stitch.
- [ ] Compose with delegated Mistral prefill and decode cache.
- [ ] Add text GQA cache tests, including sliding-window large config.
- [ ] Add guarded Conv2d-to-GEMM patch rewrite.
- [ ] Add no-layout-translation guards around placeholder/scatter/sequence regions.
- [ ] Benchmark preprocessing, vision tower, projector/stitch, prefill, decode, and logits separately.
