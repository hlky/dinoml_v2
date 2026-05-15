# ColQwen2 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: primary native checkpoint [`vidore/colqwen2-v1.0-hf`](https://huggingface.co/vidore/colqwen2-v1.0-hf). Historical ColQwen2 repos such as [`vidore/colqwen2-base`](https://huggingface.co/vidore/colqwen2-base) and [`vidore/colqwen2-v1.0-merged`](https://huggingface.co/vidore/colqwen2-v1.0-merged) are noted as legacy/ColPali-format references, not the native `model_type=colqwen2` target.

Config source: HF `config.json`, `preprocessor_config.json`, and `tokenizer_config.json` for `vidore/colqwen2-v1.0-hf`; Qwen2-VL 2B/7B/72B configs used to expose delegated-backbone variation. A small local sweep is saved in `config_sweep.md`.

Source files inspected:

- `src/transformers/models/colqwen2/configuration_colqwen2.py`
- `src/transformers/models/colqwen2/modular_colqwen2.py`
- `src/transformers/models/colqwen2/modeling_colqwen2.py`
- `src/transformers/models/colqwen2/processing_colqwen2.py`
- `src/transformers/models/colpali/modeling_colpali.py`
- `src/transformers/models/qwen2_vl/configuration_qwen2_vl.py`
- `src/transformers/models/qwen2_vl/modeling_qwen2_vl.py`
- `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`

Any missing files or assumptions: no gated repos were required. `modeling_colqwen2.py` and `processing_colqwen2.py` are generated from `modular_colqwen2.py`; future source edits should target the modular file. The report target is inference for visual document retrieval and text-query retrieval embeddings, not autoregressive generation.

## 2. High-level architecture

ColQwen2 is a multimodal retrieval wrapper around a Qwen2-VL backbone:

```text
image/text preprocessing -> optional Qwen2-VL vision encoder -> image-token embedding stitch
  -> Qwen2-VL causal text decoder run as an embedding encoder
  -> Linear(hidden_size -> embedding_dim=128) -> L2 normalize -> attention-mask zeroing
  -> late-interaction MaxSim scoring outside the model module
```

Stage decomposition:

- CPU/data pipeline: image resize/rescale/normalize, patch flattening, visual prompt construction, image placeholder expansion, query prefix/suffix construction, tokenizer padding.
- Vision stage: Qwen2-VL patch embedding, 32 vision transformer blocks, patch merger to text hidden width. Can be validated independently from text queries.
- Prefix/stitch stage: replace `<|image_pad|>` token embeddings with vision features using a placeholder mask. Processor guarantees contiguous repeated image placeholders for the standard image prompt.
- Text/retrieval stage: Qwen2-VL decoder body over the whole prompt/query sequence; no LM head for ColQwen2 retrieval.
- Scoring stage: processor helper computes `einsum("bnd,csd->bcns").max(dim=3).sum(dim=2)` for query/passages. This is an optional retrieval service primitive, not part of `ColQwen2ForRetrieval.forward`.

## 3. Important config dimensions

| Field | Native `vidore/colqwen2-v1.0-hf` value | Source |
|---|---:|---|
| top-level `model_type` | `colqwen2` | config.json |
| top-level architecture | `ColQwen2ForRetrieval` | config.json |
| embedding projection dim | 128 | config.json / `ColQwen2Config.embedding_dim` |
| dtype | `bfloat16` | config.json |
| VLM body | Qwen2-VL 2B-style | config.json |
| text hidden size | 1536 | `vlm_config.text_config` |
| text layers | 28 | `vlm_config.text_config` |
| attention heads / KV heads | 12 / 2 | `vlm_config.text_config` |
| head dim | 128 inferred from `1536 / 12`; source requires divisibility | source/config |
| MLP intermediate | 8960 | `vlm_config.text_config` |
| activation | text `silu`; vision `quick_gelu` | config.json |
| vocab size | 151936 | config.json |
| max positions | 32768 | config.json |
| RoPE theta | 1000000.0 | config.json |
| M-RoPE section | `[16, 24, 24]` | config.json |
| `use_cache` | true in config, optional/deferred for retrieval | config.json/source |
| vision depth / width | 32 blocks, `embed_dim=1280`, output hidden `1536` | config.json |
| vision heads / head dim | 16 / 80 | config/source |
| patch / temporal patch / merge | 14 / 2 / 2 | config/preprocessor |
| image processor max pixels | 602112 for ColQwen2, 12845056 for base Qwen2-VL 2B | preprocessor configs |

Representative checkpoint sweep:

| Checkpoint | Native target? | Text width/layers/heads/KV | Vision output width | Vocab | Operator-significant variation |
|---|---:|---|---:|---:|---|
| `vidore/colqwen2-v1.0-hf` | yes | 1536 / 28 / 12 / 2 | 1536 | 151936 | Adds retrieval projection and ColQwen2 processor image padding/unpadding. |
| `Qwen/Qwen2-VL-2B-Instruct` | delegated body | 1536 / 28 / 12 / 2 | 1536 | 151936 | Same backbone scale; generation head exists in Qwen2-VL but is out of scope here. |
| `Qwen/Qwen2-VL-7B-Instruct` | no | 3584 / 28 / 28 / 4 | 3584 | 152064 | Larger GEMMs, untied embeddings, different vocab. |
| `Qwen/Qwen2-VL-72B-Instruct` | no | 8192 / 80 / 64 / 8 | 8192 | 152064 | Much deeper decoder and wider projection/MLP. |
| `vidore/colqwen2-base` / `v1.0-merged` | no, legacy | 1536 / 28 / 12 / 2 | partial legacy config | 151936 | `model_type=qwen2_vl` with `architectures=["ColQwen2"]`; route separately or convert. |

## 3a. Family variation traps

- Native ColQwen2 source wraps `AutoModel.from_config(config.vlm_config)`, so the neural body is owned by Qwen2-VL. DinoML should compose a Qwen2-VL audit rather than treating ColQwen2 as a wholly separate transformer.
- The primary runtime contract is multi-vector retrieval embeddings. Do not add or require LM logits for first ColQwen2 parity.
- ColQwen2 manually computes `inputs_embeds`, runs the visual encoder, applies `masked_scatter`, then calls the VLM with `input_ids=None` and without forwarding image tensors. This differs from normal Qwen2-VL multimodal generation paths.
- The standard processor expands one `<|image_pad|>` placeholder into `prod(image_grid_thw) / merge_size**2` repeated tokens. Lowering can guard this stricter pattern instead of admitting arbitrary boolean scatter.
- Processor pads `pixel_values` to `[batch, max_num_patches, patch_vector]`; model immediately unpads with `offsets = image_grid_thw[:,1] * image_grid_thw[:,2]`. The source comment says `image_grid_thw=(num_patches_h,num_patches_w,temporal_patch_size)`, but the Qwen2-VL processor emits `[1, grid_h, grid_w]`; treat the code equation as authoritative.
- Text config has GQA (`num_key_value_heads < num_attention_heads`).
- Qwen2-VL supports sliding-window layer metadata, but inspected ColQwen2 config has `use_sliding_window=false`, so first integration can reject sliding layers.
- Historical configs may use `rope_scaling.type="mrope"` while current config standardization maps `mrope` to default RoPE with `mrope_section`; do not require a separate rope type if the current source normalizes it.
- Vision input starts as flattened patch vectors, then `PatchEmbed` views them into `[N, C, temporal_patch, patch, patch]` for a stride-equals-kernel `Conv3d`. This is a specialized packed-patch ABI, not ordinary NCHW image input inside the model.
- Layout-sensitive regions: image processor patch flatten order, vision `rot_pos_emb` merged spatial order, `masked_scatter` row order, M-RoPE channel sections, late-interaction max over passage sequence.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs.
- Boolean comparisons to image token ID and attention masks.
- `arange`, unsqueeze, expand, broadcast, `torch.split`, boolean mask gather for image unpadding.
- `masked_scatter` or guarded indexed row copy from image features into token embeddings.
- Reshape/view/permute/transpose/contiguous for QKV and patch layouts.
- Attention-mask multiplication of final embeddings by `attention_mask.unsqueeze(-1)`.

Neural network primitives:

- Text RMSNorm over hidden size 1536 with fp32 variance and eps `1e-6`.
- Vision LayerNorm over 1280 and PatchMerger LayerNorm.
- Linear projections: text Q `1536 -> 1536` bias, K/V `1536 -> 256` bias, O `1536 -> 1536` no bias; MLP gate/up `1536 -> 8960` no bias, down `8960 -> 1536` no bias.
- Vision QKV `1280 -> 3840` bias, output `1280 -> 1280`, MLP `1280 -> 5120 -> 1280`.
- PatchMerger `LayerNorm(1280) -> view groups of 4 -> Linear(5120 -> 5120) -> GELU -> Linear(5120 -> 1536)`.
- Retrieval projection `Linear(1536 -> 128)` followed by L2 normalization.

Attention primitives:

- Text causal self-attention with GQA and optional KV cache.
- Vision noncausal self-attention over variable-length image chunks; FlashAttention path uses `cu_seqlens`, non-flash path splits each image/video chunk.
- Softmax in fp32 for eager attention.

Position/rotary ops:

- Text multimodal RoPE applies temporal/height/width channel sections via `mrope_section=[16,24,24]`.
- ColQwen2 retrieval path can initially use text model default sequential 3D-expanded positions unless caller supplies `position_ids`.
- Vision rotary position embedding over merged spatial grid order.

Preprocessing-coupled ops:

- `smart_resize` divisibility by `patch_size * merge_size = 28`, min/max pixels, CLIP mean/std normalization, RGB conversion.
- Patch flatten order `[B, C, H, W] -> [B, grid_h*grid_w, C*temporal_patch*patch*patch]`.
- Text image prompt prefix and query prefix/suffix.

Scatter/indexed update ops:

- Image feature insertion into positions where `input_ids == image_token_id`.
- Required guard: number of selected embedding scalars must equal image feature scalar count.

Retrieval scoring ops:

- Padded variable-length query/passages, batched dot products, max over passage tokens, sum over query tokens, output orientation `[n_queries, n_passages]`.

## 5. Layer/block breakdown

ColQwen2 forward:

```text
if pixel_values:
  unpad pixel_values from [B,max_patches,1176] to [sum(grid_h*grid_w),1176]
inputs_embeds = token_embedding(input_ids)
if pixel_values:
  image_embeds = vlm.visual(pixel_values, image_grid_thw).pooler_output
  inputs_embeds = masked_scatter(inputs_embeds, input_ids == image_token_id, image_embeds)
hidden = vlm(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=...)
emb = Linear(hidden_size -> 128)(hidden)
emb = emb / norm(emb, dim=-1, keepdim=True)
if attention_mask: emb *= attention_mask[...,None]
```

Vision encoder, repeated 32 times:

```text
patch_vectors [total_patches, 1176]
  -> view [-1,3,2,14,14] -> Conv3d(3 -> 1280, kernel=stride=(2,14,14), no bias)
  -> tokens [total_patches,1280]
for each block:
  x = x + VisionAttention(LayerNorm(x), cu_seqlens, vision_rope)
  x = x + Linear(5120 -> 1280)(quick_gelu(Linear(1280 -> 5120)(LayerNorm(x))))
PatchMerger:
  LayerNorm(1280) -> group 2x2 spatial tokens -> Linear(5120 -> 5120) -> GELU -> Linear(5120 -> 1536)
```

Text decoder block, repeated 28 times:

```text
res = x
x = RMSNorm(x)
q = Linear(1536 -> 1536, bias=True)(x).view(B,S,12,128)
k = Linear(1536 -> 256, bias=True)(x).view(B,S,2,128)
v = Linear(1536 -> 256, bias=True)(x).view(B,S,2,128)
q,k = multimodal_rope(q,k, mrope_section=[16,24,24])
attn = causal GQA attention(q,k,v, mask/cache)
x = res + Linear(1536 -> 1536, bias=False)(attn)
res = x
x = RMSNorm(x)
x = res + Linear(8960 -> 1536)(silu(Linear(1536 -> 8960)(x)) * Linear(1536 -> 8960)(x))
```

## 6. Attention requirements

Text attention:

- Causal self-attention, not cross-attention.
- GQA: 12 query heads, 2 KV heads, 6 query heads per KV head, head dim 128.
- Q/K/V widths are 1536/256/256 for the native 2B-scale config.
- Eager path repeats KV heads before matmul; optimized path should use native GQA FlashAttention/SDPA without materializing repeat if possible.
- Masking uses Transformers causal mask utilities. Sliding-window masks exist in source but are not active for `vidore/colqwen2-v1.0-hf`.
- KV cache is implemented by the delegated Qwen2-VL text model, but retrieval embedding parity can initially run prefill/full-sequence only and reject `past_key_values`.

Vision attention:

- Noncausal self-attention over image patch sequences.
- Source packs all image patches into one sequence and uses `cu_seqlens` to delimit images for FlashAttention; non-flash source splits Q/K/V by chunk length.
- Native dimensions: 16 heads, head dim 80, no KV grouping.

Packed/varlen support:

- Vision `cu_seqlens` is required for efficient batching.
- Text packed-position support exists in Qwen2-VL, but ColQwen2 processor path does not emit `mm_token_type_ids` and ColQwen2 wrapper does not forward `**kwargs` to the VLM, so first native ColQwen2 lowering should avoid relying on Qwen2-VL generation-style packed M-RoPE.

## 7. Position encoding and custom math

Text M-RoPE source shape:

```python
def colqwen2_text_rope(q, k, cos, sin, mrope_section):
    sections = mrope_section * 2
    cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
    sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Vision position order:

```python
h_ids, w_ids = meshgrid_after_2x2_merge_reorder(h, w)
pos_ids = stack([h_ids, w_ids]).repeat(t, 1)
freqs = vision_inv_freq_outer(max(h,w))[pos_ids].flatten(1)
cos, sin = cos(freqs_cat), sin(freqs_cat)
```

Precompute candidates:

- Text inverse frequencies and vision inverse frequencies are static per config.
- Vision `pos_ids`, `cu_seqlens`, and text placeholder counts depend on image sizes and prompt packing.
- Retrieval-only ColQwen2 can precompute/capture image embeddings for a document page independently from query embeddings, but final MaxSim scores remain query-passage pair dependent.

## 8. Preprocessing and input packing

Image preprocessing:

- Accepted raw image ownership is CPU/data pipeline.
- Resize with `smart_resize(height,width,factor=28,min_pixels=3136,max_pixels=602112)` for ColQwen2 v1.0-hf.
- Rescale by `1/255`, normalize with CLIP mean/std, RGB conversion enabled.
- Processor emits channels-first flattened patch vectors. For each image: `grid_h = resized_h / 14`, `grid_w = resized_w / 14`, `image_grid_thw=[1,grid_h,grid_w]`, `pixel_values` rows have width `3*2*14*14 = 1176` in Qwen2-VL processor logic. In the ColQwen2 model, rows are viewed back through `PatchEmbed` as `[3,2,14,14]`.
- ColQwen2 pads per-image patch rows to `[batch,max_grid_h_grid_w,row_width]`; model unpads to total valid rows using `grid_h * grid_w`.

Text/query preprocessing:

- Images and text are mutually exclusive processor inputs.
- Image prompt default: `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>`.
- Query prompt default: `"Query: " + query + pad_token * 10`.
- Tokenizer config uses left padding, pad token `<|endoftext|>`, eos `<|im_end|>`, image token `<|image_pad|>` id `151655`.

Embedding stitch:

- Processor replaces the single image token in the visual prompt with repeated placeholders, then tokenizes them back as repeated image-token IDs.
- Model inserts vision features with `masked_scatter` over all image-token positions. DinoML can lower to indexed copy if it validates count equality and preserves row-major feature order.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ColQwen2 image masked_scatter -> guarded indexed row copy

Source pattern: `image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)` followed by `inputs_embeds.masked_scatter(image_mask, image_embeds)`.

Replacement: gather flattened token positions for `image_token_id`, require `len(positions) == image_embeds.shape[0]`, then row-copy `image_embeds[i]` into `inputs_embeds[b,pos,:]`.

Preconditions: image tokens are processor-generated repeated placeholders; hidden width matches VLM text hidden size; no interleaving beyond the known prompt pattern unless indexed copy supports arbitrary validated positions.

Failure cases: caller-provided custom `input_ids` with mismatched image-token count; multiple images per sample without matching `image_grid_thw` order; `inputs_embeds` path where token IDs are unavailable.

Parity test sketch: compare full `masked_scatter` to indexed copy for single image, batched different image sizes, and a deliberate mismatch rejection.

### Rewrite: packed patch Conv3d -> Linear

Source pattern: `pixel_values.view(-1,3,2,14,14)` then `Conv3d(3 -> 1280, kernel=stride=(2,14,14), bias=False)` producing one token per patch row.

Replacement:

```text
Linear(3*2*14*14 -> 1280, bias=False)
```

Weight transform: flatten conv weights as `[out_channels, 3*2*14*14]`.

Preconditions: input is already flattened exactly in Qwen2-VL processor order; kernel equals stride; padding/dilation/groups default; temporal patch size 2 and spatial patch size 14 match config.

Failure cases: raw NCHW images inside graph, altered patch flatten order, patch/temporal sizes not matching the compiled transform.

### Rewrite: late-interaction scoring to tiled GEMM + reductions

Source pattern: `einsum("bnd,csd->bcns").max(dim=3).sum(dim=2)`.

Replacement: tile query/passages, compute dot blocks over dim 128, max-reduce over passage sequence, sum-reduce over query sequence.

Preconditions: L2-normalized embeddings; padded rows are zeroed by attention mask; output orientation `[queries, passages]`.

Failure cases: nonzero padded embeddings, variable lengths without masks, output-device/dtype differences required by processor helper.

### Layout rewrite: processor/vision patch region

Source semantic layout is channels-first and flattened-patch order. A channel-last optimization is only safe for the local processor-to-patch-linear region if the flatten order and weight permutation are rewritten together. Put a no-layout-translation guard around vision `rot_pos_emb`, `PatchMerger.view(-1, spatial_merge^2*embed_dim)`, and image-token stitch unless all downstream axis contracts are explicitly updated.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for Qwen2-VL text blocks; required in every decoder block.
- GQA causal FlashAttention for `heads=12`, `kv_heads=2`, `head_dim=128`, with optional cache later.
- Text MLP SwiGLU fusion: gate/up projections plus SiLU multiply and down projection.
- Vision patch Conv3d-as-linear and 2x2 PatchMerger linear region.
- Retrieval projection plus L2 normalization and mask zeroing.

Medium priority:

- Vision variable-length noncausal attention with `cu_seqlens`.
- Q/K projection plus M-RoPE application.
- Guarded image-feature indexed stitch.
- Late-interaction MaxSim tiled scoring kernel.

Lower priority:

- Sliding-window attention, because native ColQwen2 config disables it.
- Decode/KV-cache optimized path, useful only if serving query/document embedding incrementally.
- Larger Qwen2-VL 7B/72B variants until a native ColQwen2 checkpoint wraps them.

## 11. Runtime staging plan

Stage 1: parse native `ColQwen2Config`, load top-level retrieval projection plus nested Qwen2-VL 2B-scale weights; reject legacy `model_type=qwen2_vl` ColQwen2 repos unless converted.

Stage 2: implement query-only text embedding path: token embeddings, 28 text blocks, final RMSNorm, retrieval projection, L2 normalization, attention-mask zeroing.

Stage 3: implement image processor ABI and vision encoder independently; validate `pixel_values + image_grid_thw -> image_embeds`.

Stage 4: implement image prompt embedding stitch and full document image embedding path.

Stage 5: implement retrieval scoring helper or service-side MaxSim primitive.

Stage 6: add optimized attention/fusions: GQA FlashAttention, vision varlen attention, SwiGLU, patch linear rewrite.

Stage 7: optional KV-cache/decode and larger delegated-backbone variants if future use cases need generation-like serving.

## 12. Parity and validation plan

- Config tests: parse `vidore/colqwen2-v1.0-hf`, confirm nested Qwen2-VL text/vision dimensions and projection dim.
- Processor parity: raw image sizes around min/max/divisibility boundaries; verify `image_grid_thw`, placeholder count, padded/unpadded `pixel_values`.
- Custom op tests: RMSNorm fp32 variance, M-RoPE sectioning, vision rotary ordering, patch Conv3d-to-linear rewrite.
- Single text block parity: random hidden states, masks, no cache, bf16/fp32 tolerances.
- Vision block parity: random packed patches with 1 and 2 image chunks, eager attention first, then FlashAttention/varlen path.
- Full query path parity: processor query inputs -> embeddings; compare Transformers output.
- Full image path parity: processor image inputs -> document embeddings; compare Transformers output.
- Retrieval score parity: list and padded embeddings; verify `[n_queries,n_passages]` orientation.
- Suggested tolerances: fp32 `1e-4` absolute for block/unit tests; bf16/fp16 end-to-end compare with relaxed `1e-2` to `3e-2` absolute depending on attention backend.

## 13. Performance probes

- Processor throughput by image resolution and `max_pixels` cap.
- Vision encoder throughput by total valid patches and number of images per batch.
- Text query-only throughput by batch size and sequence length.
- Document image embedding throughput split into preprocess, vision, stitch, text decoder, projection/norm.
- GQA attention backend comparison: eager/SDPA/FlashAttention for query path.
- Vision varlen attention comparison: split-per-image fallback versus packed `cu_seqlens`.
- MaxSim scoring throughput by `n_queries`, `n_passages`, query length, passage token count.
- Embedding cache memory: 128-d bf16/fp32 vectors per token for passages.
- Load-time/projection GEMM weight dtype impact for bf16.

## 14. Skip/defer list

- Training loss and labels.
- Gradient checkpointing.
- Autoregressive LM head and text generation.
- Video path in Qwen2-VL; ColQwen2 processor explicitly removes video input names.
- Sliding-window attention unless a config enables it.
- Beam search and generation helper expansion.
- PEFT adapter merge and legacy `colpali` library checkpoint loading, except as a conversion/import tool.
- Multi-GPU tensor parallel plans.
- General boolean scatter; prefer guarded indexed copy for processor-generated image placeholders.

## 15. Final implementation checklist

- [ ] Parse `ColQwen2Config` with nested `vlm_config`.
- [ ] Reject or route legacy `architectures=["ColQwen2"]`, `model_type="qwen2_vl"` repos.
- [ ] Load Qwen2-VL vision/text weights and ColQwen2 `embedding_proj_layer`.
- [ ] Implement query-only retrieval embedding path.
- [ ] Implement Qwen2-VL RMSNorm, M-RoPE, GQA attention, and SwiGLU MLP.
- [ ] Implement ColQwen2 image processor ABI or require precomputed `pixel_values`/`image_grid_thw`.
- [ ] Implement vision packed patch linear/Conv3d, vision RoPE, varlen noncausal attention, PatchMerger.
- [ ] Replace image `masked_scatter` with guarded indexed row copy.
- [ ] Implement retrieval projection, L2 normalization, and attention-mask zeroing.
- [ ] Add MaxSim scoring primitive or document service-side scoring contract.
- [ ] Add processor, single-block, vision-encoder, query-path, image-path, and scoring parity tests.
- [ ] Benchmark processor, vision encoder, text path, full document embedding, and MaxSim scoring.
