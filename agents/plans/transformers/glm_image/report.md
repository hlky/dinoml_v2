# glm_image Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: zai-org/GLM-Image/vision_language_encoder
Config source: Hugging Face raw files under zai-org/GLM-Image, plus open mirrors/quantized variants
Primary runtime target: GlmImageForConditionalGeneration, the AR image-token generator
Any missing files or assumptions: full image synthesis requires the Diffusers GLM-Image pipeline; this report covers the native Transformers AR component and source-image encoder/VQ tokenization only.
```

Source files inspected:

- `transformers/src/transformers/models/glm_image/modular_glm_image.py` is the authoritative future-edit source.
- `configuration_glm_image.py`, `modeling_glm_image.py`, `image_processing_glm_image.py`, `image_processing_pil_glm_image.py`, `processing_glm_image.py` are generated from the modular source and were used for exact runtime behavior.
- `transformers/docs/source/en/model_doc/glm_image.md`.
- Local snapshots saved in this folder:
  - `zai-org_GLM-Image_vision_language_encoder_config.json`
  - `zai-org_GLM-Image_vision_language_encoder_generation_config.json`
  - `zai-org_GLM-Image_processor_preprocessor_config.json`
  - `zai-org_GLM-Image_processor_tokenizer_config.json`
  - `xcreates_GLM-Image_vision_language_encoder_config.json`
  - `Disty0_GLM-Image-SDNQ-4bit-dynamic_vision_language_encoder_config.json`
  - `Disty0_GLM-Image-SDNQ-4bit-dynamic_vision_language_encoder_quantization_config.json`
  - `Intel_GLM-Image-int4-AutoRound_vision_language_encoder_config.json`
  - `Intel_GLM-Image-int4-AutoRound_vision_language_encoder_quantization_config.json`

HF repository notes: `zai-org/GLM-Image` is open, not gated in the HF API, but tagged as a Diffusers `text-to-image` repo. The Transformers model lives in the `vision_language_encoder` subfolder. No gated links were needed for this audit.

## 2. High-level architecture

GLM-Image in Transformers is a multimodal autoregressive image-token model:

```text
processor text/optional source images
  -> packed source-image patches + image_grid_thw + target grid prompt
  -> optional source vision encoder
  -> optional VQ codebook replacement of source image placeholders
  -> text decoder prefill with 3D/M-RoPE positions
  -> autoregressive decode of discrete image tokens
  -> lm_head over 16,512 image-token vocabulary
```

Stage decomposition:

- CPU/data pipeline: image resize, rescale, normalize, patch flattening, prompt expansion with `<sop>H W<eop>`, image placeholder expansion, tokenizer.
- Source-image encoder stage: `GlmImageVisionModel` converts packed source image patches to rank-2 hidden states. This is independently cacheable for image-to-image inputs before VQ tokenization.
- VQ tokenization stage: source image hidden states are reshaped to `[T, C, H, W]`, projected by a 1x1 conv to codebook width, L2-normalized, nearest-codebook indexed, and copied into text token IDs.
- Prefix/prefill stage: text plus source image code IDs produce embeddings and 3D position IDs.
- Decode stage: causal GQA decoder generates target-grid image tokens with KV cache.
- Outside this report: Diffusers DiT decoder, scheduler, VAE image synthesis, and text encoder in the pipeline repo.

## 3. Important config dimensions

Official `zai-org/GLM-Image/vision_language_encoder` config:

| Field | Value | Source |
| --- | ---: | --- |
| text hidden size | 4096 | config.json |
| text layers | 40 | config.json |
| text attention heads | 32 | config.json |
| text KV heads | 2 | config.json |
| text head dim | 128 | inferred from source default `hidden_size // heads` |
| text intermediate size | 13696 | config.json |
| text vocab size | 168064 | config.json |
| image-token output vocab | 16512 | config.json |
| max position embeddings | 131072 | config.json |
| text activation | SiLU | config.json |
| text attention bias | effective `True` | omitted in official config, source default |
| text dtype | bfloat16 | config.json nested `text_config.dtype` |
| RoPE | default, theta 10000, partial rotary 0.5, M-RoPE section `[8,12,12]` | config.json |
| cache | enabled | config.json |
| vision hidden size | 1536 | config.json |
| vision depth | 40 | config.json |
| vision heads/head dim | 16 / 96 | config.json + inference |
| vision intermediate size | 6144 | config.json |
| vision patch size | 16 | config.json |
| vision spatial merge | effective 1 | omitted in official config, source default |
| vision activation | GELU | config.json |
| VQ embed/codebook | 2048 / 16384 | config.json |
| VQ latent channels | 1536 | config.json |

Representative checkpoint sweep:

| Repo/subfolder | Role | Shape variation | Quant/loading variation | Notes |
| --- | --- | --- | --- | --- |
| `zai-org/GLM-Image/vision_language_encoder` | official native AR component | 40L text, 40L vision, GQA 32q/2kv, patch 16 | dense bf16 config | Official public source basis. |
| `xcreates/GLM-Image/vision_language_encoder` | open mirror | same operator dimensions as official | dense bf16 config | Same 1473-byte native config snapshot. |
| `kostakoff/GLM-Image/vision_language_encoder` | open mirror checked by raw config | same headline dimensions | dense bf16 config | Not saved locally; config returned 200 and matched headline fields. |
| `models123/GLM-Image/vision_language_encoder` | open mirror checked by raw config | same headline dimensions | dense bf16 config | Not saved locally; config returned 200 and matched headline fields. |
| `Disty0/GLM-Image-SDNQ-4bit-dynamic/vision_language_encoder` | quantized mirror | same operator dimensions | SDNQ mixed `uint4`/`int5`, dynamic/static quant metadata | Loading contract differs; source has no native SDNQ kernels. |
| `Intel/GLM-Image-int4-AutoRound/vision_language_encoder` | quantized mirror | same operator dimensions | AutoRound int4, group size 128, symmetric | Loading/provider contract differs; source has no native AutoRound kernels. |

## 3a. Family variation traps

- `hidden_size == heads * head_dim` for observed configs, but source allows explicit `head_dim`; do not infer projection widths from hidden size alone.
- Text attention is GQA: 32 query heads, 2 KV heads, repeat factor 16. Cache stores unexpanded KV heads.
- Official config omits `text_config.attention_bias` and `vision_config.spatial_merge_size`; source defaults make text Q/K/V biased and vision merge size 1.
- `GlmImageForConditionalGeneration.lm_head` outputs only `vision_vocab_size` logits, not full `vocab_size`.
- `pad_token_id` differs between model config and generation config/tokenizer: model text config has 167841, generation config uses 167855, tokenizer pad token text is `<|dit_token_16385|>`. Treat padding/generation metadata as ABI-sensitive.
- Source image processing replaces `<|image|>` placeholders in `input_ids` using `masked_scatter`, but the processor makes a strict count/order pattern that can be guarded and lowered to indexed copy.
- Processor defaults in source differ from official processor snapshot: source class defaults patch 14, temporal patch 2, merge 2, CLIP mean/std; official snapshot overrides patch 16, temporal patch 1, merge 1, mean/std `[0.5,0.5,0.5]`.
- Vision position embeddings use `grid_sample` over a learned square table; this is not ordinary RoPE.
- Text M-RoPE consumes rank-3 or rank-4 packed position IDs. Mask creation may use the leading text-position plane when `position_ids.shape[0] == 4`.
- The public repo is a composite Diffusers pipeline. Native Transformers coverage is not enough for end-to-end RGB image output.
- Quantized mirrors advertise external formats; DinoML should reject or fallback unless the loader/provider admits SDNQ/AutoRound explicitly.
- Layout traps: source image processor emits flattened patch rows, vision patch embed re-views to NCHW `[N, C, patch, patch]`, VQ path permutes `[T,H,W,C] -> [T,C,H,W]`, and `grid_sample` expects NCHW position table plus `[N,H_out,W_out,2]` grid. Protect these regions from broad NHWC translation unless all axes are rewritten.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `permute`, `transpose`, `contiguous`, `cat`, `split`, `chunk`, `stack`, `pad`, `repeat`, `repeat_interleave`, `expand`, `masked_scatter`, boolean masks, `where`, `diagonal`.
- Dynamic sequence metadata: `image_grid_thw`, `images_per_sample`, cumulative `cu_seqlens`, per-sample source/target grid splits.

Neural primitives:

- Embedding `168064 -> 4096`.
- LM head `Linear(4096 -> 16512, bias=False)`.
- Vision patch conv `Conv2d(3 -> 1536, kernel=16, stride=16)` after packed patch re-view.
- Vision LayerNorm over 1536, MLP `Linear(1536 -> 6144) -> GELU -> Linear(6144 -> 1536)`.
- Text RMSNorm over 4096.
- Text MLP `Linear(4096 -> 27392, bias=False) -> chunk(gate,up) -> SiLU(gate) * up -> Linear(13696 -> 4096, bias=False)`.
- VQ `Conv2d(1536 -> 2048, 1x1)`, L2 normalize, codebook nearest neighbor against `Embedding(16384, 2048)`, argmin.

Attention primitives:

- Vision noncausal MHA on packed variable-length image sequences: fused QKV `Linear(1536 -> 4608, bias=True)`, 16 heads, head dim 96, output projection `Linear(1536 -> 1536, bias=True)`.
- Text causal GQA: Q `Linear(4096 -> 4096, bias=True)`, K/V `Linear(4096 -> 256, bias=True)` each, O `Linear(4096 -> 4096, bias=False)`, 32 query heads, 2 KV heads, head dim 128.

Position/custom math:

- Vision learned 2D position interpolation via `grid_sample(mode="bilinear", align_corners=False, padding_mode="border")`.
- Text default RoPE with partial rotary factor 0.5, then M-RoPE section interleaving.

Generation/cache ops:

- DynamicCache-compatible KV update per text layer.
- `prepare_inputs_for_generation` drops `pixel_values` after first cached iteration.
- `logits_to_keep` slicing before LM head.
- Beam/input expansion with special packed visual tensor handling.

Preprocessing-coupled ops:

- Smart resize with aspect-ratio guard <= 4, min/max pixels, patch/merge divisibility.
- Patch flatten ABI from image processor: official snapshot emits `pixel_values` shaped `[sum(grid_h*grid_w), 3*1*16*16] = [N,768]`.
- Target-grid prompt construction: T2I adds two target grids; I2I adds source grids plus one target grid.

Quantized/packed metadata:

- Dense source has standard PyTorch row-major `nn.Linear` weights.
- SDNQ and AutoRound snapshots are loader/provider contracts only; current Transformers `glm_image` source does not implement those formats itself.

## 5. Layer/block breakdown

Vision branch:

```text
pixel_values: [total_patches, C * temporal_patch * patch * patch]
x = view(-1, 3, 16, 16)
x = Conv2d(3 -> 1536, kernel=16, stride=16)(x).view(-1, 1536)
pos = grid_sample(position_embedding_table, h/w coords from grid_thw)
x = x + pos
repeat 40:
  r = x
  x = LayerNorm(1536)(x)
  q,k,v = Linear(1536 -> 4608, bias=True).reshape(seq, 3, 16, 96)
  x = noncausal attention per image sequence using cu_seqlens
  x = r + Linear(1536 -> 1536, bias=True)(x)
  r = x
  x = LayerNorm(1536)(x)
  x = Linear(1536 -> 6144) -> GELU -> Linear(6144 -> 1536)
  x = r + x
```

VQ source-image tokenization:

```text
image_embeds split by grid_thw.prod
hs = view(T, H, W, 1536).permute(0, 3, 1, 2)
z = Conv2d(1536 -> 2048, kernel=1)(hs)
z_flat = normalize(z.permute(0,2,3,1).view(-1, 2048))
codebook = normalize(embedding.weight)
dist = ||z||^2 + ||e||^2 - 2 z e^T
image_ids = argmin(dist, dim=1)
input_ids = masked_scatter(input_ids == image_token_id, image_ids)
```

Text decoder block, repeated 40:

```text
r = x
x = RMSNorm(4096)(x)
q = Linear(4096 -> 4096, bias=True).view(B,S,32,128)
k = Linear(4096 -> 256, bias=True).view(B,S,2,128)
v = Linear(4096 -> 256, bias=True).view(B,S,2,128)
q,k = M-RoPE(q,k, position_ids[3,B,S])
k,v = cache_update(k,v)  # stored as [B,2,S,128]
x = causal GQA(q,k,v, mask)  # K/V repeat to 32 heads in eager path
x = Linear(4096 -> 4096, bias=False)(x)
x = RMSNorm(4096)(x)
x = r + x
r = x
x = RMSNorm(4096)(x)
gate_up = Linear(4096 -> 27392, bias=False)(x)
gate, up = chunk(gate_up, 2)
x = SiLU(gate) * up
x = Linear(13696 -> 4096, bias=False)(x)
x = RMSNorm(4096)(x)
x = r + x
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention, MHA, no KV cache.
- Input packed as one rank-2 sequence across images. `cu_seqlens` splits each image/frame sequence.
- Flash path passes `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, `is_causal=False`.
- Non-flash path splits Q/K/V per image sequence and concatenates outputs.
- No RoPE in vision attention; learned 2D positional embeddings are added before blocks.

Text attention:

- Causal self-attention, GQA, 32 query heads / 2 KV heads / head dim 128.
- Q width 4096; K/V width 256 each; attention output width 4096.
- KV cache shape before repeat: `[batch, 2, cached_seq, 128]` per layer. Eager attention repeats to `[batch, 32, cached_seq, 128]` only for attention math.
- Cached keys are stored after M-RoPE is applied.
- Mask comes from `create_causal_mask`; packed FA-like masking can use text-position IDs when a 4-plane position tensor is supplied.
- Source-specific eager math: matmul scale, add mask, softmax in fp32, cast back to query dtype, dropout, matmul V, transpose/contiguous.
- FlashAttention/SDPA are advertised by `_supports_flash_attn` and `_supports_sdpa`; parity must preserve M-RoPE-before-cache and GQA repeat semantics.

## 7. Position encoding and custom math

Text RoPE uses a partial rotary dimension: `head_dim=128`, `partial_rotary_factor=0.5`, so `rotary_dim=64`. M-RoPE combines temporal, height, and width frequency planes by sections.

```python
def glm_image_mrope(position_ids, inv_freq, mrope_section=(8, 12, 12)):
    # position_ids: [3, batch, seq]
    freqs = matmul(inv_freq[None, None, :, None].expand(3, batch, -1, 1),
                   position_ids[:, :, None, :]).transpose(2, 3)
    chunks = freqs.split(mrope_section, dim=-1)
    freqs = cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)
```

Vision positional adaptation:

```python
def glm_image_vision_pos(pos_table, h_coords, w_coords, target_h, target_w):
    table = pos_table.view(orig, orig, hidden).permute(2, 0, 1)[None].float()
    norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
    norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
    grid = stack((norm_w, norm_h), dim=-1)[None, :, None, :]
    return grid_sample(table, grid, mode="bilinear",
                       align_corners=False, padding_mode="border").squeeze()
```

Precomputable: base inverse frequencies and the square learned position table. Dynamic: source/target grid-derived position IDs, decode position cache, grid-sample coordinates.

## 8. Preprocessing and input packing

Official processor snapshot:

- Resize with `min_pixels=262144`, `max_pixels=4194304`, patch size 16, temporal patch size 1, merge size 1.
- Normalize RGB with mean/std `[0.5, 0.5, 0.5]`, rescale factor `1/255`.
- Emits `pixel_values`, `image_grid_thw`, `images_per_sample`.
- For each resized image, `grid_h=resized_h/16`, `grid_w=resized_w/16`, and flattened patches have width `3 * 1 * 16 * 16 = 768`.

Prompt/token ABI:

- `image_token_id=167855`, token text `<|image|>`.
- `image_start_token_id=16384`, `image_end_token_id=16385`.
- Grid text uses `<sop>` and `<eop>`.
- Text-to-image appends target grids for output and preview, then BOS. Image-to-image appends source grids and one target grid.
- `images_per_sample` is `[source_images + target_grids]` per sample.

Placeholder replacement:

- Processor expands each source `<|image|>` to exactly `grid_h * grid_w` placeholders.
- Model counts source images from non-padding `image_end_token_id`, tokenizes source images through vision+VQ, verifies placeholder count equals VQ token count, then uses `masked_scatter`.
- DinoML can lower this to a guarded indexed row copy if it verifies equal counts, processor ordering, and that replacement values are flattened in source-grid order.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed patch Conv2d -> Linear

Source pattern: `pixel_values.view(-1, 3, 16, 16) -> Conv2d(3,1536,k=16,s=16) -> view(-1,1536)`.

Replacement:

```text
MatMul(pixel_values, conv.weight.reshape(1536, 768).T) -> BiasAdd
```

Preconditions:

- Official or admitted config has `patch_size=16`, `temporal_patch_size=1`, `spatial_merge_size=1`.
- Conv `kernel_size == stride == patch_size`, padding 0, dilation 1, groups 1.
- Processor flatten order matches source NCHW re-view `[C, patch_h, patch_w]`.
- No NHWC rewrite crosses this boundary without a matching weight permutation.

Failure cases: alternate processor temporal patch size, merge size, channel-last flatten order, or non-source processor input.

Parity test: compare conv and linear outputs on random packed patches with official config.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern: boolean mask over `input_ids == image_token_id`, then `masked_scatter` of flattened `image_ids`.

Replacement:

```text
positions = nonzero(input_ids == image_token_id)
input_ids[positions] = image_ids
```

Preconditions: placeholder count equals image token count, positions are generated by `GlmImageProcessor`, and row-major flatten order is preserved.

Failure cases: user-supplied arbitrary `input_ids`, missing source grids, or count mismatch.

### Rewrite: text gate_up projection split

Source pattern: one dense `Linear(4096 -> 27392)` then `chunk(2)`.

Replacement: preserve packed weight as `[gate_rows; up_rows]` and fuse `SiLU(gate) * up`.

Preconditions: `intermediate_size=13696`, `hidden_act=silu`, bias false.

### Layout guard: vision/VQ NCHW regions

Protect `patch_embed`, vision positional `grid_sample`, and VQ `permute(0,3,1,2)`/1x1 conv from generic NHWC translation unless the pass rewrites axes for conv, grid coordinates, normalize dim, and downstream view contracts together.

## 10. Kernel fusion candidates

Highest priority:

- Text GQA FlashAttention with KV cache: dominates decode; must support 32q/2kv and post-RoPE cached K.
- RMSNorm and residual/RMSNorm placement: four RMSNorms per text block plus final norm.
- Gate-up SwiGLU fusion: one large `4096 -> 27392` projection per layer.
- Last-token-only logits via `logits_to_keep`: avoid full sequence `4096 -> 16512` where generation only needs next token.

Medium priority:

- M-RoPE generation/application fused with Q/K layout.
- Vision variable-length attention using `cu_seqlens`.
- Packed patch Conv2d-to-GEMM rewrite.
- VQ nearest-codebook GEMM/argmin for image-to-image source tokenization.

Lower priority:

- Vision GELU MLP fusion.
- Position `grid_sample` specialization or precompute for common target/source grids.
- Generation visual tensor expansion helpers for beam search.

## 11. Runtime staging plan

Stage 1: parse dense `GlmImageConfig`, load text weights, and run one text decoder block with random tensors.

Stage 2: implement M-RoPE position generation and text prefill logits for text-to-image prompts with no source images.

Stage 3: add decode with DynamicCache-compatible KV tensors and `logits_to_keep=1`.

Stage 4: add processor ABI or a strict external preprocessor contract for `input_ids`, `attention_mask`, `image_grid_thw`, and `images_per_sample`.

Stage 5: add source-image vision encoder and VQ token replacement for image-to-image.

Stage 6: add optimized GQA attention, gate-up fusion, RMSNorm fusion, and patch-conv lowering.

Stage 7: decide quantized mirror policy: reject, dense-dequant fallback, or provider-specific SDNQ/AutoRound admission.

Stub initially: Diffusers decoder, RGB image synthesis, source-image path, quantized loaders, beam search, training losses.

## 12. Parity and validation plan

- Config defaults: verify omitted `attention_bias` and `spatial_merge_size` resolve to source defaults.
- Random op tests: RMSNorm, partial RoPE/M-RoPE, vision position `grid_sample`, VQ normalize/distance/argmin.
- Single text layer parity in fp32, then bf16 tolerance.
- Text stack prefill parity on a small prompt with official tokenizer/processor output, comparing hidden states and `lm_head` logits.
- Decode parity for several generated steps, verifying cached KV length and position IDs.
- Vision branch parity on synthetic packed patches and `image_grid_thw`.
- Placeholder replacement parity with one and two source images.
- End-to-end AR token parity: compare generated image-token IDs before Diffusers decoding.

Suggested tolerances: fp32 `atol=1e-5, rtol=1e-5`; bf16/fp16 block outputs `atol=2e-2, rtol=2e-2`, with token parity checked by logits ranking where exact sampling is RNG-dependent.

## 13. Performance probes

- Processor throughput by resolution and batch size.
- Vision encoder throughput by total patches and number of images.
- VQ tokenization time versus codebook size 16384 and token count.
- Text prefill tokens/sec for prompt plus target-grid metadata.
- Decode tokens/sec across target grids: default 1152x768 gives `token_h=36`, `token_w=24`, plus preview in T2I.
- KV cache memory for 40 layers, 2 KV heads, head dim 128, bf16.
- Attention backend comparison: eager, SDPA, FlashAttention with GQA.
- Patch-conv lowered GEMM versus Conv2d.
- LM head `logits_to_keep` full-sequence versus last-token.
- Quantized loading/dequant/provider probes for SDNQ and AutoRound mirrors if admitted.

## 14. Skip/defer list

- Training, losses, gradient checkpointing.
- Full Diffusers DiT decoder, scheduler, VAE RGB synthesis.
- Quantized SDNQ/AutoRound execution unless separately admitted.
- Beam search and complex visual tensor expansion.
- Video paths; source code has temporal grid conventions, but official GLM-Image processor snapshot uses temporal patch size 1 for images.
- Alternate RoPE types beyond observed default unless a config requires them.
- Multi-GPU tensor parallel; source provides a TP plan, but first integration can be single-device dense.

## 15. Final implementation checklist

- [ ] Parse `GlmImageConfig` with nested text, vision, and VQ configs.
- [ ] Apply source defaults for omitted config fields.
- [ ] Load dense text weights and `lm_head`.
- [ ] Implement text RMSNorm.
- [ ] Implement partial M-RoPE with `[8,12,12]` sections.
- [ ] Implement causal GQA attention with 32 query heads and 2 KV heads.
- [ ] Implement DynamicCache-compatible decode.
- [ ] Implement `logits_to_keep` slicing before `lm_head`.
- [ ] Add text prefill and decode parity tests.
- [ ] Implement processor ABI validation for `image_grid_thw` and `images_per_sample`.
- [ ] Implement guarded placeholder indexed-copy replacement.
- [ ] Implement vision patch Conv2d or guarded packed-patch GEMM rewrite.
- [ ] Implement vision variable-length noncausal attention over `cu_seqlens`.
- [ ] Implement vision `grid_sample` position interpolation.
- [ ] Implement VQ normalize/distance/argmin tokenization.
- [ ] Add source-image parity tests.
- [ ] Benchmark prefill, decode, vision encoder, VQ, and LM head separately.
- [ ] Reject or explicitly route SDNQ/AutoRound quantized checkpoints until provider support exists.
