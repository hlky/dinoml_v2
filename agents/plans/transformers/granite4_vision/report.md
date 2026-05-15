# granite4_vision Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ibm-granite/granite-4.0-3b-vision
Config source: https://huggingface.co/ibm-granite/granite-4.0-3b-vision/raw/main/config.json
Processor sources:
  https://huggingface.co/ibm-granite/granite-4.0-3b-vision/raw/main/processor_config.json
  https://huggingface.co/ibm-granite/granite-4.0-3b-vision/raw/main/preprocessor_config.json
  https://huggingface.co/ibm-granite/granite-4.0-3b-vision/raw/main/tokenizer_config.json
Source files inspected:
  transformers/src/transformers/models/granite4_vision/configuration_granite4_vision.py
  transformers/src/transformers/models/granite4_vision/modeling_granite4_vision.py
  transformers/src/transformers/models/granite4_vision/processing_granite4_vision.py
  transformers/src/transformers/models/granite4_vision/modular_granite4_vision.py
  transformers/src/transformers/models/siglip/modeling_siglip.py
  transformers/src/transformers/models/blip_2/modeling_blip_2.py
Any missing files or assumptions:
  Only one official granite4_vision checkpoint was found. No gated repos were encountered.
  The Hub repo is tagged custom_code and has auto_map entries, but this report scopes to the in-library source at the pinned commit.
  Modeling/config/processing files are generated from modular_granite4_vision.py; generated files are runtime truth, modular is the upstream edit source.
```

Small operator-significant snapshot: `agents/plans/transformers/granite4_vision/config_snapshot.json`.

## 2. High-level architecture

Primary runtime target: multimodal image-text-to-text causal generation with one or more images in the prefill prompt, then text-only autoregressive decode.

Dataflow:

```text
CPU image resize/pad/crop + tokenizer placeholder expansion
-> SigLIP vision encoder over NCHW 384x384 crops
-> 8 Window-QFormer downsampler/projectors
-> pack/unpad/add image_newline features
-> zero text image-token embeddings
-> layerwise masked_scatter additive image-feature injection
-> causal text decoder prefill/decode
-> tied LM head logits / logits_scaling
```

Stage decomposition:

- CPU/data pipeline: LLaVA-NeXT image processor chooses any-resolution grid pinpoints, emits `pixel_values`, `image_sizes`, text `input_ids`, and expanded `<image>` placeholders.
- Vision encoder: SigLIP vision model consumes flattened crops as `[sum_patches, 3, 384, 384]`; outputs all hidden states.
- Projectors: four deepstack projectors consume selected SigLIP layers and four spatial projectors consume one layer, each producing LLM-width features.
- Prefix construction: features are packed back per source image, unpadded according to original image size, and newline embeddings are appended along image rows.
- Prefill: base token embeddings at `<image>` positions are zeroed; each configured decoder layer adds projected image features into the same placeholder positions.
- Decode: `prepare_inputs_for_generation` forwards image tensors only on the first iteration or when cache is disabled; later decode should reuse the text decoder cache and omit vision work.

Independently stageable validation units: SigLIP crop encoder, one Window-QFormer projector, pack/unpad/newline feature packing, placeholder injection, one decoder block, full prefill logits, cache decode logits.

## 3. Important config dimensions

| Field | Official `ibm-granite/granite-4.0-3b-vision` value | Provenance |
|---|---:|---|
| architecture | `Granite4VisionForConditionalGeneration` | config.json |
| dtype | `bfloat16` | config.json |
| parameters | 3997.2M | HF repo metadata |
| text source class | `Granite4VisionTextModel` | modeling source |
| text nested `model_type` | `granitemoehybrid` | config.json |
| text hidden size | 2560 | config.json |
| text layers | 40 | config.json |
| text attention heads / KV heads | 40 / 8 | config.json |
| head dim | 64 | inferred from source default `hidden_size // num_attention_heads` |
| GQA repeat | 5 | inferred from 40 / 8 |
| text intermediate size | 8192 | config.json |
| vocab size | 100353 | config.json |
| image token id | 100352 | config/tokenizer_config |
| max positions | 131072 | config.json |
| RoPE theta | 10000000 | config.json |
| embedding / attention / residual / logit multipliers | 12 / 0.015625 / 0.22 / 10 | config.json + source |
| text attention bias / MLP bias | false / source default false | config + source default |
| tied LM head | true | config + `_tied_weights_keys` |
| vision family | `siglip_vision_model` | config.json |
| vision image / patch size | 384 / 16 | config.json |
| vision patch tokens per crop | 576 | inferred 24 * 24 |
| vision hidden / layers / heads | 1152 / 27 / 16 | config.json |
| vision MLP width / activation | 4304 / `gelu_pytorch_tanh` | config.json |
| projector type | Window Q-Former downsampler | source |
| model downsample rate | `4/8` | config.json |
| processor downsample rate | `2/4` | processor_config, equivalent fraction |
| projector output tokens per 384 crop | 144 | inferred from 24x24 -> 12x12 |
| qformer default layers / heads / hidden | 1 / 18 / 1152 | config class defaults derived from vision hidden |
| deepstack map | `[-19->9, -13->6, -7->3, -1->0]` | config.json |
| spatial target layers | `[12, 15, 18, 21]` | config.json |
| image grid pinpoints | 27 resolutions up to 3840 on long side | config.json |

Representative checkpoint sweep:

| Checkpoint | Status | Operator-significant notes |
|---|---|---|
| `ibm-granite/granite-4.0-3b-vision` | official, open | Only official `granite4_vision` repo found by HF search. Uses SigLIP 384/16, 8 Window-QFormer projectors, pure attention Granite4Vision text implementation. |
| Source default `Granite4VisionConfig()` | source default, not a checkpoint | Defaults to SigLIP vision, Llama text config, image token id 32000, default grid pinpoints, and qformer derived from vision hidden size. Useful for parser defaults only, not production parity. |
| Hypothetical configs with explicit `qformer_config` | source-supported | May change Q-Former layers/heads/activation/dropout/cross-attention frequency; must not be assumed identical to official checkpoint. |

## 3a. Family variation traps

- Native source ignores the checkpoint's `granitemoehybrid` text topology fields for this family. `Granite4VisionModel` constructs `Granite4VisionTextModel(config.text_config)`, whose layers are pure causal self-attention + SwiGLU MLP. Do not import Mamba, MoE, router, or hybrid-cache requirements from the nested config unless auditing a separate remote-code path.
- `text_config.use_cache` is false in the checkpoint, but the forward path accepts `past_key_values`; generation can still pass a cache. Admission should be explicit about supported cache mode.
- `hidden_size == num_attention_heads * head_dim` for the official checkpoint, but source allows explicit `head_dim`; parser should not infer projection widths solely from hidden size.
- GQA is required: 40 query heads, 8 KV heads, 5x KV repeat.
- Attention scaling is `attention_multiplier`, not `1/sqrt(head_dim)`, in the Granite4Vision text decoder.
- Embeddings are multiplied by 12; residual branch outputs are multiplied by 0.22 before residual add; logits are divided by 10.
- The processor expands one textual `<image>` marker into many repeated `<image>` tokens. Model stitching uses a broad boolean mask, but the processor guarantees a stricter placeholder-count pattern that DinoML can guard.
- `vision_feature_select_strategy="full"` for the official checkpoint. The source supports `"default"` by dropping the first vision token; official SigLIP vision embeddings have no CLS token, so `"full"` should be the first admission target.
- Source layout is NCHW for pixel crops and `[B, tokens, C]` for vision/projector/text sequences. NHWC/channel-last is an optimization candidate only within guarded patch embedding/vision blocks.
- Pack/unpad/newline logic is axis-sensitive and depends on original `image_sizes`, selected grid resolution, and row-major feature order.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `flatten`, `split`, `cat`, `expand`, `unsqueeze`, `squeeze`, `arange`.
- Boolean compare, `masked_fill`, `masked_scatter`, boolean indexing, count/sum for placeholder validation.
- Dynamic shape integer math for image grid selection, patch counts, unpad extents, `Fraction(downsample_rate)`.

Vision preprocessing-coupled ops:

- CPU image resize/center-crop/pad/rescale/normalize/RGB conversion via `LlavaNextImageProcessor`.
- Any-resolution grid selection using `image_grid_pinpoints`.
- `pixel_values` layout: padded processor output can be `[batch, num_patches, 3, 384, 384]`; model flattens valid patches to `[sum_patches, 3, 384, 384]`.

Vision encoder primitives:

- Conv2d patch embedding: `Conv2d(3 -> 1152, kernel=16, stride=16, padding=valid)` over NCHW.
- Learned absolute position embedding `[576, 1152]`, added to patch tokens.
- 27 SigLIP encoder layers: LayerNorm, dense noncausal MHA with 16 heads of dim 72, MLP `Linear(1152 -> 4304)`, `gelu_pytorch_tanh`, `Linear(4304 -> 1152)`, residual adds.
- Final post LayerNorm exists in SigLIP, but Granite4Vision consumes `vision_outputs.hidden_states`, not pooler output.

Window Q-Former/projector primitives:

- Eight `Granite4VisionWindowQFormerDownsampler` modules for the official checkpoint.
- LayerNorm over vision hidden, area interpolation downsample for deepstack projectors, 2x2 spatial offset sampling for spatial projectors.
- Window raster/unraster reshapes: 24x24 tokens -> 3x3 windows of 8x8 keys; downsampled 12x12 -> 3x3 windows of 4x4 queries.
- Learned query `[1, 16, 1152]` and learned image positions `[1, 64, 1152]`.
- BLIP-2 QFormer, default one layer: query self-attention, cross-attention from 16 query tokens to 64 window tokens, LayerNorm residuals, MLP `Linear(1152 -> 3072) -> gelu -> Linear(3072 -> 1152)`.
- Projector output `Linear(1152 -> 2560, bias=True)`.

Text decoder primitives:

- Embedding table `[100353, 2560]`, tied to LM head.
- 40 decoder layers with RMSNorm, GQA causal self-attention, RoPE, SwiGLU MLP, residual scaling.
- Projections per layer: Q `Linear(2560 -> 2560, bias=False)`, K/V `Linear(2560 -> 512, bias=False)`, O `Linear(2560 -> 2560, bias=False)`.
- MLP per layer: gate/up `Linear(2560 -> 8192, bias=False)`, down `Linear(8192 -> 2560, bias=False)`, activation `silu`, elementwise multiply.
- Final RMSNorm and LM head `Linear(2560 -> 100353, bias=False)`, then divide logits by 10.

Position/cache ops:

- RoPE cos/sin generated in fp32 from position ids, cast to hidden dtype.
- Causal mask creation via `create_causal_mask`.
- KV cache update stores post-RoPE keys and values per text layer.

Scatter/indexed update ops:

- Initial `inputs_embeds.masked_fill(vision_mask, 0.0)`.
- At configured decoder layers, `hidden_states = hidden_states.masked_scatter(mask, hidden_states[mask] + features.flatten())`.
- Safe DinoML lowering target: guarded indexed row add/copy into placeholder rows, with exact placeholder token count and row-major flatten order checks.

## 5. Layer/block breakdown

SigLIP vision crop encoder, repeated 27 times after patch embedding:

```text
pixel_values [P,3,384,384]
-> Conv2d k16/s16 -> [P,1152,24,24]
-> flatten/transpose + learned pos -> [P,576,1152]

Vision block:
  y = LayerNorm(x)
  q,k,v = Linear(1152 -> 1152) each, 16 heads x 72
  y = noncausal attention(q,k,v)
  x = x + Linear(1152 -> 1152)(y)
  y = LayerNorm(x)
  y = Linear(1152 -> 4304) -> gelu_pytorch_tanh -> Linear(4304 -> 1152)
  x = x + y
```

Window Q-Former projector, official dimensions:

```text
image_features [P,576,1152]
-> LayerNorm
-> window keys [P*9,64,1152] + image_positions
-> downsample:
   deepstack: area interpolate 24x24 -> 12x12
   spatial: pick one corner from each 2x2 block -> 12x12
-> query [P*9,16,1152] = learned_query + downsampled_window
-> QFormer self/cross attention over query and key window
-> unraster [P,144,1152]
-> Linear(1152 -> 2560, bias=True)
```

Text decoder block, repeated 40 times:

```text
if layer_idx has image features:
  x[image_placeholder_rows] += projected_features

residual = x
x = RMSNorm(x)
q = Linear(2560 -> 2560)(x) -> [B,40,T,64]
k = Linear(2560 -> 512)(x) -> [B,8,T,64]
v = Linear(2560 -> 512)(x) -> [B,8,T,64]
q,k = RoPE(q,k)
k,v = cache.update(k,v) if cache is present
y = causal GQA attention(q,k,v), scaling=0.015625
x = residual + Linear(2560 -> 2560)(y) * 0.22
residual = x
x = RMSNorm(x)
y = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + y * 0.22
```

LM head:

```text
hidden = final_RMSNorm(x)
logits = tied_lm_head(hidden[:, selected_positions, :]) / 10
```

## 6. Attention requirements

Text decoder attention:

- Causal self-attention.
- GQA: 40 Q heads, 8 KV heads, 5 KV repeats, head dim 64.
- Q width 2560; K/V width 512 each; attention output width 2560.
- RoPE is applied to Q and K before cache update.
- Cached K/V shapes before repeat: `[batch, 8, seq, 64]`; backend repeat expands to `[batch, 40, seq, 64]` logically.
- Masking comes from `create_causal_mask`; eager path adds mask before fp32 softmax and casts probabilities back to query dtype.
- FlashAttention/SDPA/FlexAttention are source-supported through `ALL_ATTENTION_FUNCTIONS`. Fused attention must preserve the model's custom scaling value.

Vision attention:

- SigLIP encoder uses dense noncausal self-attention, 16 heads, head dim 72, no cache.
- Mask is bidirectional/optional; no local/sparse pattern.

Q-Former attention:

- Query-driven non-autoregressive attention, not generation decode.
- Query source: learned query table plus downsampled visual features, shape `[P*9,16,1152]`.
- Key/value source for cross-attention: windowed image features plus image position embeddings, shape `[P*9,64,1152]`.
- Default QFormer has one layer and `cross_attention_frequency=1`, so every layer has cross-attention.
- The BLIP-2 QFormer source disables attention backend support; first DinoML parity should use explicit dense matmul/softmax.

## 7. Position encoding and custom math

Text RoPE:

```python
def granite4_vision_rope(position_ids, head_dim=64, theta=10000000):
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Cos/sin can be precomputed for static or bucketed positions. Dynamic decode depends on `past_key_values.get_seq_length()`.

Vision position:

- SigLIP uses learned absolute patch position embeddings for a 24x24 grid.
- Interpolation exists in SigLIP source but Granite4Vision calls the vision tower without passing `interpolate_pos_encoding`; official preprocessing keeps 384x384 crops, so first admission can reject other crop sizes.

Projector position:

- Learned `image_positions` are added to each 8x8 key window.
- Learned `image_newline` is appended during feature packing; it is a parameter, not processor metadata.

## 8. Preprocessing and input packing

Processor contract:

- Image processor: `LlavaNextImageProcessor`, `do_resize`, `do_center_crop`, `do_pad`, `do_rescale`, `do_normalize`, `do_convert_rgb`.
- Normalization: mean/std `[0.5, 0.5, 0.5]`, rescale factor `1/255`.
- Crop size: 384x384. Patch size: 16.
- Any-resolution grid pinpoints: 27 `[height,width]` choices. The model also recomputes selected grid shapes from `image_sizes`.
- Tokenizer: GPT2Tokenizer with left padding; `<image>` id 100352.

Placeholder expansion:

- For every `<image>` substring in text, processor computes `num_image_tokens` from original size, selected grid, processed crop size, patch size, and downsample rate, then replaces one textual marker with that many repeated `<image>` tokens.
- If `vision_feature_select_strategy == "default"`, token count is reduced by one. Official config uses `"full"`.

Model packing:

- `pixel_values` may be rank 5 `[batch, num_patches, 3, 384, 384]`; model slices each sample to valid patch count and concatenates to rank 4.
- For multi-patch any-resolution images, `pack_image_features` separates base image features from grid image features, reshapes `[num_patch_h, num_patch_w, h, w, C]`, permutes to channel-first image-map order, unpads according to original image size, appends newline feature per image row, flattens back to token sequence, and prepends base image features.
- For one-patch images, it appends one newline token after the base feature sequence.

Scatter contract:

- `get_placeholder_mask` checks that `num_masked_elements == image_features.numel()`, not just token count. The mask is expanded over hidden dimension.
- Processor-generated prompt gives contiguous repeated `<image>` token runs per image, but source would accept any positions as long as counts match. DinoML should first require processor-like contiguous runs and reject arbitrary boolean scatter.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 1152, kernel=16, stride=16, padding=valid)
-> flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten_NCHW_16x16_row_major -> GEMM([P*576, 768] x [768,1152]) -> add bias -> [P,576,1152]
```

Preconditions:

- NCHW input, height and width exactly 384 for first target.
- `kernel_size == stride == 16`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Preserve PyTorch Conv2d flatten order: output grid row-major after `flatten(2).transpose(1,2)`.

Weight transform:

```python
w = conv.weight.reshape(1152, 3 * 16 * 16).T
b = conv.bias
```

Failure cases: non-384 crops unless position interpolation and dynamic grid are admitted; NHWC input unless a guarded layout pass rewrites axes and weight layout.

Parity test sketch: compare Conv2d path and lowered Linear path for random `[2,3,384,384]` fp32/bf16 inputs before and after position add.

### Rewrite: placeholder masked_scatter -> indexed row add

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds = masked_fill(expanded_mask, 0)
hidden_states = masked_scatter(expanded_mask, hidden_states[mask] + features.flatten())
```

Replacement:

```text
validate image_token rows and feature rows
zero selected embedding rows
for each selected row i: hidden[i, :] += features[row_i, :]
```

Preconditions:

- `input_ids` are available, not only `inputs_embeds`.
- Placeholder rows are processor-generated contiguous runs per image.
- Feature row count equals placeholder token count for each packed image and layer.
- Same feature order as `torch.cat(packed_features, dim=0)`.

Failure cases: arbitrary scattered image tokens, inputs-only embedding comparison path, mismatched image feature counts.

### Rewrite: Window raster/downsample/QFormer batching

Source pattern:

```text
[P,576,C] -> [P*9,64,C] keys
[P,144,C] -> [P*9,16,C] queries
```

Replacement:

```text
static reshape/transpose kernels around dense QFormer attention and MLP
```

Preconditions:

- Official 384/16 grid, `downsample_rate=1/2`, `window_side=8`, `query_side=4`.
- `24 % 8 == 0`; no default-CLS drop for first target.

Failure cases: alternate qformer config, alternate crop/grid size, non-equivalent downsample rate.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :]) / logits_scaling
```

Replacement:

```text
Gather selected decode rows -> tied embedding GEMM -> divide by 10
```

Preconditions: `logits_to_keep` is int 1 or a static tensor of selected positions; tied weight alias preserved.

## 10. Kernel fusion candidates

Highest priority:

- Text RMSNorm with fp32 variance and bf16 output.
- GQA causal attention with RoPE and custom scaling, prefill and decode cache variants.
- SwiGLU MLP: two GEMMs plus `silu(gate) * up` fused activation/multiply before down GEMM.
- Placeholder indexed row zero/add for image feature injection, replacing general boolean scatter.
- SigLIP patch embedding as im2col/window flatten + GEMM or direct Conv2d provider.

Medium priority:

- SigLIP LayerNorm + QKV projection + dense noncausal attention for 576-token fixed crops.
- QFormer 16-query/64-key cross-attention microkernel or batched dense attention.
- Window raster/unraster and spatial offset downsample fused reshape/gather kernels.
- Pack/unpad/newline GPU kernel if end-to-end image prefill becomes a bottleneck.

Lower priority:

- Area interpolation downsample specialized 24x24 -> 12x12.
- Dynamic any-resolution packing in GPU runtime; initially keep in CPU/data pipeline.
- Alternate attention backends beyond SDPA/FlashAttention parity.

## 11. Runtime staging plan

Stage 1: parse official config and load weights with an admission rule that rejects Mamba/MoE interpretation for this family; preserve tied embedding/LM-head alias.

Stage 2: implement text-only `Granite4VisionTextModel` parity: embedding multiplier, 40 attention decoder blocks, custom RoPE theta/scaling, residual multiplier, final RMSNorm, logits scaling.

Stage 3: add cache decode for text-only prompts, with GQA KV cache `[B,8,T,64]`.

Stage 4: implement SigLIP vision encoder for fixed 384x384 crops and validate selected hidden states.

Stage 5: implement one Window-QFormer projector and then all eight projector applications with official deepstack/spatial layer maps.

Stage 6: implement processor-compatible feature packing and guarded indexed image-token injection for prefill.

Stage 7: run full multimodal prefill logits parity, then decode parity where image tensors are omitted after first iteration.

Stage 8: add optimized kernels/fusions: patch conv rewrite, fixed-shape vision attention, QFormer attention, GQA FlashAttention, last-token logits.

Stub initially: training/loss, arbitrary `inputs_embeds` image matching path, non-official qformer configs, non-384 vision crops, `"default"` vision feature strategy.

## 12. Parity and validation plan

- Config parser test: official config yields pure attention Granite4Vision text layers despite nested `granitemoehybrid` fields.
- Unit test `interpolate_downsample` and `spatial_offset_downsample` for `[P,576,1152]`.
- Unit test `_windowed_raster` and `_unwindowed_raster` round trip for 24/8 and 12/4 shapes.
- SigLIP patch embedding parity: Conv2d vs lowered GEMM for fp32 and bf16.
- SigLIP one-layer and 27-layer hidden-state parity against Transformers.
- QFormer projector parity for one image crop, then multi-crop split/pack parity.
- Placeholder injection parity: compare masked source path to indexed row add under processor-generated contiguous image tokens.
- Text decoder one-layer parity with and without image feature injection.
- Text-only prefill logits parity and one-token decode parity.
- Full image+text prefill logits parity on one 384x384 image and one any-resolution multi-patch image.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 `rtol=3e-2, atol=3e-2` for full graph, tighter per-kernel tolerances where accumulation is fp32.

## 13. Performance probes

- Processor throughput: images/sec for resize/pad/grid selection and placeholder expansion.
- Vision encoder throughput: crop batch sweep for `[P,3,384,384]`.
- Projector throughput: QFormer windows/sec for 1, 2, 4, and 8 images.
- Packing/injection cost: feature rows/sec and scatter/indexed-copy bandwidth.
- Prefill throughput: text length x image token count sweep.
- Decode tokens/sec: batch size sweep with cached image prefix.
- KV cache memory: 40 layers x 2 x 8 KV heads x 64 head dim x sequence length x dtype.
- Last-token logits GEMM cost with vocab 100353.
- Compare eager dense attention, SDPA, FlashAttention for text prefill/decode and SigLIP 576-token attention.

## 14. Skip/defer list

- Training, gradient checkpointing, and loss parity.
- Arbitrary remote-code behavior that diverges from pinned in-library source.
- Mamba/SSM/MoE fields present in nested `text_config`; they are out of scope for this native source path.
- `"default"` vision feature strategy unless a checkpoint requires it.
- Non-384 crop sizes and SigLIP position interpolation.
- Arbitrary boolean `masked_scatter` image-token positions; require processor-generated placeholder runs first.
- Multi-GPU tensor parallel plan.
- Quantized or packed weight loading beyond normal safetensors dtype handling.
- Beam search and generation-controller extras beyond first-token/greedy parity.

## 15. Final implementation checklist

- [ ] Parse `Granite4VisionConfig` and official processor/tokenizer metadata.
- [ ] Reject or separately route configs requiring remote-code behavior outside pinned source.
- [ ] Preserve tied `embed_tokens.weight` / `lm_head.weight` alias.
- [ ] Implement Granite4Vision text decoder: embedding multiplier, RMSNorm, GQA, RoPE, SwiGLU, residual multiplier, logits scaling.
- [ ] Implement text KV cache with `[B,8,T,64]` K/V per layer.
- [ ] Implement SigLIP 384/16 vision encoder and selected hidden-state export.
- [ ] Implement Window-QFormer downsampler for official 24x24 -> 12x12 windows.
- [ ] Implement deepstack and spatial projector schedules.
- [ ] Implement image feature pack/unpad/newline logic.
- [ ] Lower placeholder scatter to guarded indexed row zero/add.
- [ ] Add Conv2d patch embedding -> GEMM rewrite.
- [ ] Add one-layer, projector, text-only, multimodal prefill, and decode parity tests.
- [ ] Benchmark processor, vision, projector, prefill, decode, and logits stages separately.
