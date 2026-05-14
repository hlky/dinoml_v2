# Idefics2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: HuggingFaceM4/idefics2-8b, plus base/chatty/AWQ config variants
Config source: Hugging Face Hub raw config/preprocessor/processor/tokenizer/generation files, May 13 2026
Source files inspected:
  X:/H/transformers/src/transformers/models/idefics2/modeling_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/configuration_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/processing_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/image_processing_idefics2.py
  X:/H/transformers/src/transformers/models/idefics2/image_processing_pil_idefics2.py
  X:/H/transformers/src/transformers/models/mistral/modeling_mistral.py
  X:/H/transformers/src/transformers/models/mistral/configuration_mistral.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/generation/utils.py
Any missing files or assumptions:
  No official small/debug Idefics2 checkpoint config was found in the local source scan. The report uses 8B production variants and source defaults.
  modeling_idefics2.py is the authoritative native implementation; no modular_idefics2.py exists in this checkout.
```

Primary source URLs:

- Transformers native source at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics2`
- Mistral nested decoder source at the same commit:
  `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mistral`
- Hub config roots:
  `https://huggingface.co/HuggingFaceM4/idefics2-8b`,
  `https://huggingface.co/HuggingFaceM4/idefics2-8b-base`,
  `https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty`,
  `https://huggingface.co/HuggingFaceM4/idefics2-8b-AWQ`

Hub configs inspected:

| Model id | Hub revision sha from API | Scope |
|---|---:|---|
| `HuggingFaceM4/idefics2-8b` | `2c42686c57fe21cf0348c9ce1077d094b72e7698` | instructed/common production |
| `HuggingFaceM4/idefics2-8b-base` | `e37a0b376ca55a497c68de86505c60a0f8d7d713` | base model, no image splitting by default |
| `HuggingFaceM4/idefics2-8b-chatty` | `8e65868b394317b973bd61db3b08e6478ebeedbf` | chat/instruction variant |
| `HuggingFaceM4/idefics2-8b-AWQ` | `83af3b1fa68533f74e0248127efa3dfa0a005416` | 4-bit AWQ text-weight variant |

Primary runtime target: multimodal conditional generation, with image encoder plus connector prefill and Mistral causal decode.

## 2. High-level architecture

Idefics2 is a vision encoder + Perceiver connector + Mistral decoder model:

```text
CPU image/text preprocessing
  -> NCHW pixel batch + pixel masks + expanded <image> token placeholders
  -> SigLIP-like ViT over variable-resolution patch grids
  -> gated MLP modality projection
  -> Perceiver resampler to 64 image tokens per real image
  -> masked scatter into token embeddings at <image> positions
  -> Mistral causal LM prefill
  -> cached autoregressive decode
  -> lm_head logits / generation sampling
```

Stage decomposition:

| Stage | Runtime contract | Independently cacheable? |
|---|---|---|
| Processor | Builds text placeholders, resizes/splits/pads/normalizes images, emits `input_ids`, `attention_mask`, `pixel_values`, `pixel_attention_mask` | CPU/data pipeline; cache processed tensors per request |
| Vision encoder | Consumes real nonzero images only after dropping padded image slots | Yes, per image batch and preprocessing settings |
| Connector | Projects vision hidden states and Perceiver-resamples each real image to 64 vectors | Yes, `image_hidden_states` can be passed instead of `pixel_values` |
| Embedding stitch | Replaces exactly the `<image>` token slots with connector vectors using `masked_scatter` | Validate independently; shape mismatch should fail early |
| Text prefill | Mistral decoder over full mixed text/image sequence | Standard AR KV cache |
| Decode | Mistral one-token or short-step decode using text KV cache | Image pixels should not be recomputed |

## 3. Important config dimensions

The 8B Hub configs are sparse. Many operator-significant text/perceiver values come from `MistralConfig` and `Idefics2PerceiverConfig` defaults after `Idefics2Config.__post_init__`.

| Field | Effective 8B value | Provenance |
|---|---:|---|
| text model | `mistral` via `AutoModel.from_config` | config.json |
| text hidden size | 4096 | Mistral source default |
| text layers | 32 | Mistral source default |
| text attention heads / KV heads | 32 / 8 | Mistral source default |
| text head dim | 128 | inferred by MistralConfig |
| text MLP intermediate | 14336 | Mistral source default |
| text activation | `silu` | Mistral source default |
| max positions | 32768 | config.json override |
| sliding window | 4096 | Mistral source default |
| RoPE | default RoPE, theta 10000 | rotary config default |
| vocab size | 32002 or 32003 | config.json variant |
| image token id | 32001 | config.json |
| vision hidden size | 1152 | config.json |
| vision layers | 27 | config.json |
| vision heads | 16 | config.json |
| vision MLP intermediate | 4304 | config.json |
| vision patch size | 14 | config.json |
| trained image size / position grid | 980 -> 70 x 70 learned buckets | config.json + source |
| Perceiver latents | 64 per real image | Perceiver default + processor config |
| Perceiver depth | 3 | Perceiver source default |
| Perceiver heads / KV heads / head dim | 16 / 4 / 96 | Perceiver source default |
| cache support | `use_cache=True`, DynamicCache for text decoder | config/source |

Representative checkpoint sweep:

| Model id | Vocab | Image splitting | dtype field | Generation EOS | Quantization/operator change |
|---|---:|---|---|---|---|
| `idefics2-8b-base` | 32002 | false | float32 | `2` | none |
| `idefics2-8b` | 32003 | true | float32 | `[2, 32002]` | none |
| `idefics2-8b-chatty` | 32003 | true | float32 | `[2, 32002]` | none |
| `idefics2-8b-AWQ` | 32003 | true | float16 | `[2, 32002]` | AWQ 4-bit GEMM; excludes vision model, modality projection, Perceiver |

## 3a. Family variation traps

- `perceiver_config` in the Hub configs only says `model_type: idefics2`; all Perceiver dimensions are source defaults, then `hidden_size`/`rms_norm_eps` are synchronized to the text config if needed.
- Processor `image_seq_len` must equal `config.perceiver_config.resampler_n_latents` (64). A mismatch breaks the placeholder/scatter contract.
- `do_image_splitting=True` changes one logical image into 5 processed images: 4 quadrants plus the original. The processor also expands one `<image>` marker into 5 runs of 64 `<image>` tokens.
- Base has vocab 32002 and no `<end_of_utterance>` in the tokenizer config; instructed/chatty/AWQ use vocab 32003 and `32002` as an extra EOS.
- Text decoder is Mistral GQA with sliding-window causal masks. Do not assume full-context Llama attention despite similar blocks.
- Vision tensors are source NCHW, wrapped as `[batch, max_num_images, 3, H, W]`. Layout passes need guards around `Conv2d`, `unfold`, `flatten(2).transpose(1,2)`, and mask axes.
- The Perceiver is GQA cross-attention over `concat(context, latents)`, not a simple pooling layer.
- Normal source generation blocks `<fake_token_around_image>` and `<image>` through `bad_words_ids`; relying on generated image tokens during decode is out of scope.
- AWQ quantizes text-side GEMMs but explicitly does not convert vision, modality projection, or Perceiver modules.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text tokens.
- NCHW `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` patch embed.
- `flatten`, `transpose`, `view`, `reshape`, `contiguous`, `expand`, `cat`, boolean indexing, `masked_scatter`.
- Pixel mask `unfold` over height and width with `size=stride=patch_size`, sum over patch windows, equality to full-patch area.
- Patch position math: `arange`, `bucketize`, clamp, indexed assignment into `position_ids`.
- Drop padded images by detecting all-zero image slots.

Neural network primitives:

- LayerNorm for vision blocks.
- RMSNorm for text and Perceiver.
- Dense Linear/GEMM with and without bias.
- Vision MLP: `Linear(1152 -> 4304)` + `gelu_pytorch_tanh` + `Linear(4304 -> 1152)`.
- Text MLP: SwiGLU `gate/up/down` with `4096 -> 14336 -> 4096`, bias false.
- Connector projection: gated MLP `1152 -> 14336 -> 4096`, bias false.
- Perceiver MLP: gated MLP `4096 -> 16384 -> 4096`, bias false.
- LM head `Linear(4096 -> vocab_size)`, bias false; config sets `tie_word_embeddings=false`, although `_tied_weights_keys` still names the optional alias.

Attention primitives:

- Vision bidirectional self-attention, MHA 16 heads, head dim 72.
- Perceiver bidirectional GQA cross-attention, Q heads 16 x 96, KV heads 4 x 96; KV length is patch sequence plus 64 latents.
- Mistral causal GQA self-attention, 32 Q heads, 8 KV heads, head dim 128, sliding window 4096.
- Eager path uses matmul, additive mask, fp32 softmax, dropout, matmul. Source also advertises SDPA, FlashAttention, and FlexAttention through Transformers backend dispatch.

Position/cache/preprocessing ops:

- Vision learned absolute position embedding selected by fractional bucketized coordinates.
- Mistral RoPE on Q/K before cache update.
- Text `DynamicCache` per decoder layer; cache stores post-RoPE keys and values before KV repetition.
- Processor-coupled placeholder expansion and `masked_scatter` image embedding stitch.
- Generation input preparation that disables `pixel_values` after first step or when precomputed `image_hidden_states` are supplied.

## 5. Layer/block breakdown

Vision patch embedding:

```text
pixel_values: [R, 3, H, W] NCHW, where R = number of nonzero real image rows
patch_embeds = Conv2d(3, 1152, kernel=stride=14)(pixel_values)  # [R, 1152, H/14, W/14]
x = flatten(2).transpose(1, 2)                                  # [R, P, 1152]
position_ids = bucketize(valid patch fractional h/w into 70 x 70 grid)
x = x + position_embedding(position_ids)
```

Vision encoder block, repeated 27 times:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(1152 -> 1152, bias=True)
x = bidirectional MHA(q,k,v, patch mask)
x = residual + Linear(1152 -> 1152, bias=True)(x)
residual = x
x = LayerNorm(x)
x = Linear(1152 -> 4304, bias=True) -> gelu_pytorch_tanh -> Linear(4304 -> 1152, bias=True)
x = residual + x
```

Connector:

```text
image_hidden: [R, P, 1152]
x = SwiGLU-style MLP(1152 -> 14336 -> 4096, bias=False)
x = PerceiverResampler(context=x, mask=patch_mask_flat)
output: [R, 64, 4096], flattened to [R * 64, 4096]
```

Perceiver layer, repeated 3 times:

```text
latents: [R, 64, 4096]
context: [R, P, 4096]
attention_mask over [context_len + 64]
latents_norm = RMSNorm(latents)
context_norm = RMSNorm(context)
q = Linear(4096 -> 1536, bias=False)(latents_norm)      # 16 * 96
k = Linear(4096 -> 384, bias=False)(cat(context, latents))  # 4 * 96
v = Linear(4096 -> 384, bias=False)(cat(context, latents))
latents = residual + Linear(1536 -> 4096, bias=False)(GQA(q,k,v))
latents = latents + SwiGLU MLP(4096 -> 16384 -> 4096)
```

Text decoder block, repeated 32 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(4096 -> 4096, bias=False)
k,v = Linear(4096 -> 1024, bias=False)
q,k = RoPE(q,k)
k,v = DynamicCache.update(k,v, layer_idx) if cache is active
x = causal/sliding-window GQA(q,k,v)
x = residual + Linear(4096 -> 4096, bias=False)(x)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over valid patch tokens.
- MHA, 16 heads, hidden 1152, head dim 72.
- Mask is bidirectional and derived from `patch_attention_mask.view(R, -1)`.
- No KV cache.

Perceiver attention:

- Noncausal cross/self hybrid: queries are latents; keys/values are `concat(context, latents)`.
- GQA: 16 query heads, 4 KV heads, head dim 96, KV repeated by 4 in eager attention.
- Attention mask covers context tokens plus latent tokens. It is built with `create_bidirectional_mask`.
- Normal forward does not pass `past_key_values`; the source has cache plumbing but the attention constructor currently sets `self.layer_idx = None`, so DinoML should treat Perceiver caching as unsupported unless separately validated.

Text attention:

- Causal decoder self-attention through Mistral.
- GQA: 32 query heads, 8 KV heads, head dim 128, KV repeated by 4 for eager math.
- Sliding-window causal mask when `config.sliding_window` is not `None`; effective default is 4096.
- RoPE applied to Q/K before cache update. Cache tensor shapes per layer are effectively `[batch, 8, cached_seq, 128]` for K and V before repeat expansion; attention sees repeated `[batch, 32, seq, 128]` in eager fallback.
- Backend dispatch can use eager, SDPA, FlashAttention, or FlexAttention. DinoML parity should preserve source order: Q/K/V projection, RoPE, cache update, attention backend with scaling `head_dim**-0.5`, output projection.

## 7. Position encoding and custom math

Vision variable-resolution position IDs use learned 70 x 70 position buckets for a max trained `image_size=980` and `patch_size=14`.

```python
def idefics2_patch_position_ids(patch_attention_mask, num_patches_per_side=70):
    nb_h = patch_attention_mask[:, :, 0].sum(1)
    nb_w = patch_attention_mask[:, 0, :].sum(1)
    step_h = 1.0 / nb_h
    step_w = 1.0 / nb_w
    h = arange(max_patches_h) * step_h[:, None]
    w = arange(max_patches_w) * step_w[:, None]
    h = clamp(h, max=1.0 - 1e-6)
    w = clamp(w, max=1.0 - 1e-6)
    bucket_h = bucketize(h, boundaries=arange(1/70, 1.0, 1/70), right=True)
    bucket_w = bucketize(w, boundaries=arange(1/70, 1.0, 1/70), right=True)
    return bucket_h[:, :, None] * 70 + bucket_w[:, None, :]
```

Mistral RoPE:

```python
def apply_mistral_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RoPE cos/sin can be precomputed per position range and dtype/device, but position IDs depend on cache length during decode.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Text must contain `<image>` markers. Processor replaces each marker with `<fake_token_around_image>` + 64 repeated `<image>` tokens + `<fake_token_around_image>`.
- If `do_image_splitting=True`, the replacement is repeated 5 times and image preprocessing produces 4 quadrants plus the original image.
- Images are converted to RGB, resized by shortest/longest edge, rescaled, normalized, and padded to the batch max image count and max H/W.
- Default source image processor values are shortest edge 378, longest edge 980, bilinear resize, rescale factor `1/255`; Hub Idefics2 configs use mean/std `[0.5, 0.5, 0.5]`.
- Runtime tensors: `pixel_values [B, max_images, 3, Hmax, Wmax]`, `pixel_attention_mask [B, max_images, Hmax, Wmax]`.

GPU/runtime:

- Model flattens image slots to `[B * max_images, 3, H, W]`, drops all-zero padded images, and carries the matching flattened pixel mask.
- Patch mask is true only if every pixel in the patch-size window is valid.
- Connector output is `[real_images, 64, 4096]`, then flattened to `[real_images * 64, 4096]`.
- `inputs_merger` builds a boolean mask where `input_ids == image_token_id`, expands it over hidden size, and uses `masked_scatter` to insert image vectors into `inputs_embeds`.
- DinoML should validate that `count(input_ids == 32001) == real_images * 64`; with splitting this is `logical_images * 5 * 64`.
- The model can accept precomputed `image_hidden_states` instead of pixels. This is the clean first cache boundary for image-heavy generation.

Generation-controller behavior:

- Generation configs block token IDs `32000` and `32001` as bad words.
- Instruct/chatty/AWQ use EOS `[2, 32002]`; base uses EOS `2`.
- Chat template is processor-owned, not part of the core GPU graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Source pattern:

```text
Conv2d(C=3, out=1152, kernel=14, stride=14, padding=valid)
flatten(2).transpose(1, 2)
```

Replacement:

```text
PatchExtract/WindowFlatten [R, P, 3*14*14] -> GEMM(weight.T) -> BiasAdd -> [R, P, 1152]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, dilation 1, groups 1.
- H and W are divisible by 14 after preprocessing or fallback handles floor patch count exactly as Conv2d.
- Source NCHW flatten order is preserved.

Weight transform:

```python
w = conv.weight.reshape(1152, 3 * 14 * 14)
y = windows @ w.T + conv.bias
```

Failure cases: dynamic image sizes not divisible by patch size, NHWC-translated tensors without matching window flatten order, or future grouped/dilated patch embeds.

Parity test: compare patch embeddings before positional add for rectangular padded images with mixed valid masks.

### Rewrite: image connector precompute

Source pattern:

```text
pixel_values -> vision_model -> modality_projection -> perceiver -> image_hidden_states -> masked_scatter
```

Replacement: expose `image_hidden_states` as a compiled subgraph output/input boundary.

Preconditions:

- Same processor settings, dtype, and model weights.
- Text prompt placeholders match precomputed image count.
- No training/dropout.

Failure cases: image splitting mismatch, processor mean/std mismatch, or prompts with different numbers of image placeholders.

Parity test: one forward with `pixel_values`, one forward with `image_hidden_states` from `get_image_features`, compare prefill logits.

### Rewrite: placeholder stitch to indexed copy

Source pattern: `inputs_embeds.masked_scatter(input_ids == image_token_id, image_hidden_states)`.

Replacement: compute image-token indices and perform deterministic indexed copy into an embedding buffer.

Preconditions:

- Mask count exactly equals source row count.
- Source row order matches processor/image flatten order.
- No generated image token during decode.

Shape equation:

```text
num_image_rows = sum(input_ids == image_token_id)
image_hidden_states.shape == [num_image_rows, hidden_size]
```

Failure cases: user-provided `inputs_embeds` path compares embeddings to the image-token embedding rather than checking `input_ids`; prefer requiring `input_ids` for first integration.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` controlled by `logits_to_keep`.

Replacement: for decode, run vocab GEMM only for final token.

Preconditions: no loss calculation and `logits_to_keep` is 1 or a final-token slice.

Parity test: compare full logits final row with last-token-only logits.

### Layout optimization candidate: local NCHW vision island

Initial lowering should preserve NCHW semantics. A later layout pass may translate the patch embed and vision block internals, but it must rewrite:

- `pixel_values` axes `[B, I, C, H, W]`.
- `unfold(dimension=1/2)` after image-slot flattening of masks `[R, H, W]`.
- Conv weight layout `[out, in, kh, kw]`.
- `flatten(2).transpose(1,2)` sequence construction.

Protect processor boundary, patch-mask construction, and placeholder stitch with no-layout-translation guards until those rewrites are proven.

## 10. Kernel fusion candidates

Highest priority:

- Mistral RMSNorm + QKV GEMM preparation, because it dominates prefill/decode and matches other audited Mistral-family work.
- GQA FlashAttention/sliding-window decode with KV cache, because text decode is the production bottleneck after image features are cached.
- SwiGLU/MLP fused activation multiply for text and connector GEMMs.
- Placeholder indexed-copy stitch, because `masked_scatter` is awkward for static lowering and easy to validate.

Medium priority:

- Patch Conv2d -> GEMM for NCHW non-overlap patches.
- Vision LayerNorm + attention + MLP block fusions for image-heavy batch throughput.
- Perceiver GQA attention over `[patches + 64]` KV length; important for high-resolution/image-splitting workloads.
- Vision position bucketize/indexing helper, especially if preprocessing leaves variable aspect ratios.

Lower priority:

- MultiheadAttentionPoolingHead, because it is defined but not used by the main Idefics2 forward path.
- AWQ-specific text GEMM lowering; useful later but the unquantized graph should land first.
- Processor GPU offload for resize/normalize; CPU pipeline is acceptable for first parity.

## 11. Runtime staging plan

1. Parse native Idefics2 config and fill Mistral/Perceiver defaults explicitly.
2. Load text-only Mistral submodule and verify pure text generation with no `pixel_values`.
3. Implement placeholder expansion validation in the importer but keep processor outside DinoML runtime.
4. Run connector-only parity from random/recorded vision hidden states through modality projection and Perceiver.
5. Run vision encoder parity for one rectangular image batch, including patch mask and position bucket IDs.
6. Implement image feature precompute boundary: accept `image_hidden_states` into the compiled text prefill graph.
7. Implement full prefill with `pixel_values` after vision/connector parity is stable.
8. Implement decode with text KV cache and verify no image recomputation after first iteration.
9. Add optimized attention and GEMM/fusion rewrites under guards.
10. Add AWQ/quantized text weights only after dense Idefics2 parity is routine.

Initially stub or externalize: chat template, image URL loading, PIL/torchvision preprocessing, beam search, training loss, and AWQ quantization.

## 12. Parity and validation plan

- Config/default test: sparse Hub config expands to expected text/perceiver defaults.
- Processor contract test: count expanded `<image>` tokens for base (`1 * 64`) and splitting variants (`5 * 64`).
- Patch mask test: padded rectangular image produces the same `patch_attention_mask` as Transformers.
- Vision embedding test: compare Conv2d + bucketized position embedding for a small random image/mask.
- Vision block test: one encoder layer, then 27-layer encoder parity.
- Connector test: modality projection + 3 Perceiver layers for random `[R, P, 1152]` and masks.
- Stitch test: indexed-copy replacement equals `masked_scatter`, including multiple images per batch.
- Text prefill test: pure text and multimodal prompt logits.
- Decode test: first-token prefill plus two cached decode steps; verify cache shapes `[B, 8, S, 128]` per Mistral layer.
- End-to-end smoke: one image QA prompt with fixed `max_new_tokens`, comparing generated token IDs under greedy decoding.

Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4` for blocks; fp16/bf16 `atol=2e-2, rtol=2e-2` for full model logits, with tighter per-op tolerances where kernels accumulate in fp32.

## 13. Performance probes

- CPU processor throughput: images/sec for resize/split/pad/normalize.
- Vision encoder throughput by image resolution and by `do_image_splitting`.
- Connector throughput vs patch count and number of real images.
- Prefill tokens/sec for mixed sequences with 0, 1, 5, and 10 real processed images.
- Decode tokens/sec after cached image/text prefill.
- KV cache memory: `32 layers * 2 tensors * B * 8 KV heads * seq * 128 * dtype_size`.
- Perceiver attention backend comparison: eager vs SDPA/Flash/Flex equivalent.
- Last-token-only logits speedup for decode.
- AWQ dense-vs-quantized text GEMM probe after dense parity.

## 14. Skip/defer list

- Training loss and label masking.
- Gradient checkpointing.
- Beam search and complex generation policies beyond greedy/sampling parity.
- GPU implementation of PIL/torchvision preprocessing.
- AWQ import/lowering for first dense graph.
- Multi-GPU tensor parallel and pipeline plans.
- Perceiver KV caching; normal source path does not require it.
- Remote-code or historical config flags not read by native source.
- General NHWC vision translation until local patch/vision layout guards exist.

## 15. Final implementation checklist

- [ ] Parse Idefics2 config and materialize sparse Mistral/Perceiver defaults.
- [ ] Load text, vision, connector, and LM-head weights with correct alias policy.
- [ ] Implement Mistral decoder ops: RMSNorm, RoPE, GQA, sliding-window causal mask, KV cache.
- [ ] Implement vision patch Conv2d or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement vision variable-resolution position bucketization.
- [ ] Implement bidirectional vision attention and LayerNorm MLP blocks.
- [ ] Implement connector gated MLP and Perceiver GQA resampler.
- [ ] Implement image placeholder count validation and indexed embedding stitch.
- [ ] Support `image_hidden_states` as a precomputed image-feature input.
- [ ] Add text-only prefill/decode parity tests.
- [ ] Add processor placeholder and image-splitting contract tests.
- [ ] Add vision encoder, connector, stitch, multimodal prefill, and cached decode parity tests.
- [ ] Benchmark processor, vision, connector, prefill, decode, and KV memory separately.
