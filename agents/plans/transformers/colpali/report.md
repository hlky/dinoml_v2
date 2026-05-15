# ColPali Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from transformers.

Model id:
  Primary native checkpoints: vidore/colpali-v1.2-hf, vidore/colpali-v1.3-hf.
  Adapter/original repos inspected for variation: vidore/colpali-v1.2,
  vidore/colpali-v1.3, vidore/colpali2-3b-pt-448.
  Debug/nonstandard snapshot: michaelfeil/colpali-v12-random-testing.

Config source:
  HF raw config/preprocessor/tokenizer snapshots saved under
  agents/plans/transformers/colpali/_sources/.

Source files inspected:
  transformers/src/transformers/models/colpali/configuration_colpali.py
  transformers/src/transformers/models/colpali/modeling_colpali.py
  transformers/src/transformers/models/colpali/processing_colpali.py
  transformers/src/transformers/models/colpali/modular_colpali.py
  transformers/src/transformers/models/paligemma/configuration_paligemma.py
  transformers/src/transformers/models/paligemma/modeling_paligemma.py
  transformers/src/transformers/models/paligemma/processing_paligemma.py
  transformers/src/transformers/models/siglip/modeling_siglip.py
  transformers/src/transformers/models/gemma/modeling_gemma.py
  transformers/tests/models/colpali/test_modeling_colpali.py
  transformers/tests/models/colpali/test_processing_colpali.py

Any missing files or assumptions:
  processing_colpali.py is generated from modular_colpali.py; future source
  edits should target modular_colpali.py. Native ColPali neural coverage composes
  PaliGemma, which itself composes SigLIP vision and Gemma text. This report owns
  the ColPali retrieval wrapper and required runtime ABI; full PaliGemma/SigLIP/
  Gemma kernel coverage should be shared with their own audits.
```

## 2. High-level architecture

Primary runtime target: visual document retrieval with separate document-image and text-query embedding passes, then late-interaction scoring. This is not an autoregressive generation target, although the delegated PaliGemma/Gemma body can expose `past_key_values`.

```text
document image -> ColPaliProcessor image path -> PaliGemma VLM -> projection -> L2/mask -> page token embeddings
text query -> ColPaliProcessor text path -> PaliGemma/Gemma text path -> projection -> L2/mask -> query token embeddings
query embeddings + page embeddings -> MaxSim late interaction -> scores [num_queries, num_pages]
```

Stage decomposition:

- CPU/data pipeline: image fetch/RGB conversion/resize/rescale/normalize and tokenizer prompt construction.
- Vision encoder/projector: SigLIP patch encoder plus PaliGemma multimodal projector; independently cacheable for document pages.
- Text/prefix encoder: Gemma language backbone over query text or image-token prompt sequence.
- ColPali head: `Linear(text_hidden_size -> embedding_dim)`, L2 normalize over embedding dimension, zero padded tokens.
- Retrieval scorer: pad variable-length multi-vector embeddings, compute query-page MaxSim scores.

## 3. Important config dimensions

Native v1.2/v1.3 HF configs are effectively identical in operator structure.

| Field | v1.2-hf / v1.3-hf value | Provenance |
| --- | ---: | --- |
| `model_type` | `colpali` | HF `config.json` |
| architecture | `ColPaliForRetrieval` | HF `config.json` |
| wrapper `embedding_dim` | 128 | HF `config.json` |
| dtype | `bfloat16` top-level, text config `float32` | HF `config.json`; runtime casts wrapper inputs to module dtype |
| VLM model type | `paligemma` | HF `config.json` |
| text hidden size | 2048 | HF `config.json` |
| text layers | 18 | HF `config.json` |
| text attention heads | 8 | HF `config.json` |
| text KV heads | 1 | HF `config.json`; GQA/MQA-style |
| text head dim | 256 effective | Gemma source default because omitted in checkpoint config |
| text intermediate size | 16384 | HF `config.json` |
| text activation | `gelu_pytorch_tanh` effective | Gemma source default because omitted |
| text attention bias | false effective | Gemma source default because omitted |
| text max positions | 8192 effective | Gemma source default because omitted |
| text RoPE | default, theta 10000 effective | Gemma source default path |
| PaliGemma positions | 1-indexed `position_ids` | PaliGemma source |
| bidirectional prefix attention | true effective for PaliGemma when unset | PaliGemma config source |
| vision model | `siglip_vision_model` | HF `config.json` |
| vision hidden size | 1152 | HF `config.json` |
| vision layers / heads | 27 / 16 | HF `config.json` |
| vision MLP size | 4304 | HF `config.json` |
| image size / patch | 448 / 14 | HF `config.json` |
| image tokens | 1024 | HF `config.json`; `(448/14)^2` |
| image token id | 257152 | HF `config.json` |
| vocab size | 257216 | HF `config.json` |
| preprocessor output layout | channels-first `[B,3,448,448]` | processor defaults and HF `preprocessor_config.json` |

Representative checkpoint sweep:

| Model | Native DinoML target? | Config shape | Image ABI | Notes |
| --- | --- | --- | --- | --- |
| `vidore/colpali-v1.2-hf` | yes | `model_type=colpali`, PaliGemma/Gemma/SigLIP, 128-d embeddings | `[B,3,448,448]`, 1024 image tokens | Official native Transformers checkpoint. |
| `vidore/colpali-v1.3-hf` | yes | same operator dimensions as v1.2-hf | same | Common production checkpoint; different weights. |
| `michaelfeil/colpali-v12-random-testing` | no, debug only | `model_type=paligemma`, `architectures=["ColPali"]` | `[B,3,224,224]`, 256 image tokens | Nonstandard/debug snapshot; not native `ColPaliConfig`. |
| `vidore/colpali-v1.2` | adapter route | no root `config.json`; PEFT LoRA adapter | `[B,3,448,448]`, 1024 image tokens | `adapter_config.json` targets language projections and `custom_text_proj`. |
| `vidore/colpali-v1.3` | adapter route | no root `config.json`; PEFT LoRA adapter | likely same 448 path | Requires merge/adapters or original `colpali` library. |
| `vidore/colpali2-3b-pt-448` | separate audit | no root `config.json`; PEFT adapter over PaliGemma2 base | `[B,3,448,448]`, 1024 image tokens | PaliGemma2/Gemma2 body changes are not covered by this native ColPali target. |

## 3a. Family variation traps

- ColPali is a wrapper over `AutoModel(config.vlm_config)`. Do not infer a fixed neural body from the ColPali directory alone.
- Native HF checkpoints have `text_config` repeated both top-level and under `vlm_config`; source uses `config.vlm_config.text_config` for model construction and projection width.
- Popular `vidore/colpali-v1.2` and `vidore/colpali-v1.3` repos are PEFT/original-style adapters, not native `ColPaliForRetrieval` checkpoints. First DinoML admission should reject or require a pre-merged HF checkpoint.
- The processor processes images or text, not both in one call. This differs from vanilla PaliGemma.
- Page/document embeddings include prompt/text/image-token positions and are zeroed by `attention_mask`; scoring code assumes padded tokens are zero vectors.
- Retrieval score orientation is `[queries, passages]`. Swapping query/page axes silently changes downstream ranking.
- Vision source is NCHW: processor emits channels-first, SigLIP patch embedding consumes `[B,C,H,W]`, then flattens to `[B,num_patches,hidden]`. NHWC is only a guarded optimization region.
- PaliGemma uses `masked_scatter` to replace image-token embeddings with vision features; placeholder count must equal image feature count.
- Gemma uses GQA/MQA-style text attention for native configs: 8 Q heads, 1 KV head, head dim 256.
- PaliGemma sets bidirectional/prefix-aware masking behavior via token type IDs on the first iteration. Retrieval does not need generation decode, but it does need this mask parity for image/text prefix embedding.

## 4. Operator coverage checklist

Wrapper-owned ColPali ops:

- `Linear(2048 -> 128)` with bias for `embedding_proj_layer`.
- L2 norm over last dim: `embeddings / norm(embeddings, dim=-1, keepdim=True)`.
- `attention_mask.unsqueeze(-1)` and elementwise multiply.
- Variable-length padding of embedding lists for scoring.
- Late interaction score: batched dot products `[Bq,Sq,128] x [Bp,Sp,128] -> [Bq,Bp,Sq,Sp]`, `max` over passage sequence, `sum` over query sequence.

Delegated PaliGemma/SigLIP/Gemma ops required for native v1.2/v1.3:

- Token embedding with image-token replacement guard; for OOV image token, PaliGemma maps image token id to pad before embedding.
- `masked_scatter`/indexed copy from image features into text embeddings.
- 1-indexed position ids for Gemma.
- Prefix-aware causal/bidirectional masks from attention mask and token type IDs.
- SigLIP patch `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` on NCHW.
- SigLIP learned 2D position embedding flattened to patch sequence; optional bicubic interpolation if enabled.
- SigLIP encoder repeated 27 times: LayerNorm, MHA, residual, LayerNorm, MLP `1152 -> 4304 -> 1152`, residual.
- PaliGemma projector `Linear(1152 -> 2048)` with bias.
- Gemma decoder repeated 18 times: RMSNorm, GQA self-attention, residual, RMSNorm, gated MLP `2048 -> 16384 -> 2048`, residual.
- Gemma attention projections: `q_proj 2048 -> 2048`, `k_proj 2048 -> 256`, `v_proj 2048 -> 256`, `o_proj 2048 -> 2048`, all bias-free for effective defaults.
- RoPE on Q/K with head dim 256.
- Attention backend compatibility: eager/SDPA/Flash/Flex advertised in source; DinoML can start with dense attention parity and later use FlashAttention.

Preprocessing-coupled ops:

- Image fetch/flatten list, `process_image`, RGB conversion, resize to 448, rescale by `1/255`, normalize mean/std `[0.5,0.5,0.5]`, channels-first output.
- Query prompt packing: `bos + "Question: " + query + pad_token*10 + "\n"`, default max length 50.
- Document prompt packing: `"<image>" * image_seq_length + bos + "Describe the image." + "\n"`.
- Token type IDs and labels are produced by processor; labels are training-only for retrieval inference.

Quantized/packed weight metadata:

- Native v1.2/v1.3 HF checkpoints are bf16 safetensors with ordinary dense names including `embedding_proj_layer.*`, `vlm.multi_modal_projector.linear.*`, and delegated backbone weights.
- Adapter repos use LoRA PEFT metadata; native DinoML runtime should either premerge or reject until adapter loading is explicitly admitted.

## 5. Layer/block breakdown

ColPali forward:

```text
optional pixel_values = pixel_values.to(model dtype)
vlm_output = PaliGemmaModel(input_ids, pixel_values, attention_mask, output_hidden_states=True, ...)
last_hidden = vlm_output[0]                         # [B,S,2048]
emb = Linear(2048 -> 128)(last_hidden.to(proj_dtype))
emb = emb / l2_norm(emb, dim=-1, keepdim=True)      # [B,S,128]
if attention_mask: emb = emb * attention_mask[...,None]
return embeddings, optional past_key_values, hidden_states, attentions, image_hidden_states
```

PaliGemma image+text embedding path used by document pages:

```text
pixel_values [B,3,448,448]
SigLIP patch conv -> [B,1152,32,32] -> flatten/transpose -> [B,1024,1152]
add learned patch positions
27 x SigLIP encoder block
post LayerNorm -> [B,1024,1152]
Linear(1152 -> 2048) -> image_features [B,1024,2048]
token_embed(input_ids with image ids mapped to pad if OOV) -> [B,S,2048]
masked_scatter image_features into image-token slots
Gemma 18-layer text model with prefix-aware mask -> [B,S,2048]
```

Gemma block:

```text
residual = x
y = RMSNorm(x)
q = Linear(2048 -> 8*256)(y)
k = Linear(2048 -> 1*256)(y)
v = Linear(2048 -> 1*256)(y)
q,k = RoPE(q,k, 1-indexed positions)
attn = GQA(q,k,v, mask, optional cache)
x = residual + Linear(2048 -> 2048)(attn)
residual = x
y = RMSNorm(x)
y = Linear(16384 -> 2048)(gelu_pytorch_tanh(Linear(2048 -> 16384)(y)) * Linear(2048 -> 16384)(y))
x = residual + y
```

SigLIP vision block:

```text
residual = x
y = LayerNorm(x)
y = dense noncausal MHA(y)
x = residual + y
residual = x
y = LayerNorm(x)
y = Linear(4304 -> 1152)(gelu_pytorch_tanh(Linear(1152 -> 4304)(y)))
x = residual + y
```

## 6. Attention requirements

ColPali wrapper itself has no attention. Required attention comes from delegated PaliGemma.

Vision attention:

- Noncausal dense self-attention over 1024 patch tokens.
- MHA with 16 heads, hidden 1152, head dim 72.
- No KV cache for retrieval.
- Mask is usually absent in the vision encoder.

Text attention:

- Gemma self-attention over the full packed sequence.
- Native configs use 8 query heads, 1 KV head, head dim 256, repeat factor 8.
- Q/K receive RoPE before cache update.
- Source eager path applies `matmul(q,k^T) * head_dim^-0.5`, adds mask, softmax in fp32, casts back, then matmul with V.
- PaliGemma mask is prefix-aware when `token_type_ids` are present in first iteration: image/prefix tokens can attend bidirectionally and suffix tokens use causal structure.
- Retrieval target does not require autoregressive decode or KV cache for first integration, but the model output may expose delegated `past_key_values`; reject cache-enabled runs initially unless Gemma cache ABI is already supported.
- Sliding attention is not present for native PaliGemma/Gemma1 configs; source has a conditional for text configs with `sliding_window`, mostly relevant to PaliGemma2/Gemma2-style variants.

## 7. Position encoding and custom math

SigLIP vision uses learned absolute patch positions. For 448/14 native configs, the fixed table shape is `[1024,1152]`. Optional interpolation path reshapes to `[1,Hpos,Wpos,D]`, permutes to NCHW, bicubic-interpolates, then returns `[1,new_h*new_w,D]`; first integration should guard exact image size and disable interpolation.

PaliGemma/Gemma text positions are 1-indexed:

```python
position_ids = arange(seq_len, device=device) + past_seen_tokens
position_ids = position_ids.unsqueeze(0) + 1
```

Gemma default RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos(), emb.sin()
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

ColPali custom math:

```python
emb = proj(last_hidden)
emb = emb / emb.norm(dim=-1, keepdim=True)
emb = emb * attention_mask.unsqueeze(-1)
score = einsum("bnd,csd->bcns", q_emb, p_emb).max(dim=3).values.sum(dim=2)
```

## 8. Preprocessing and input packing

Image/document path:

- Caller supplies pages as images; no OCR or layout boxes are required.
- Processor fetches URLs if needed, flattens image lists, applies SigLIP image preprocessing, and emits `pixel_values`.
- Native v1.2/v1.3 preprocessor config: resize to 448 x 448, rescale by `0.00392156862745098`, normalize with mean/std 0.5, channels-first. `do_convert_rgb` is a processor default even when omitted in fetched config.
- Text string is built as 1024 `<image>` placeholders, BOS token, visual prompt prefix `"Describe the image."`, newline.
- `input_ids`, `attention_mask`, `token_type_ids`, `labels`, and `pixel_values` are returned for images. `labels` are not required for inference.

Query path:

- Processor accepts string or list of strings.
- Default string: `bos + "Question: " + query + pad_token*10 + "\n"`.
- Default query `max_length` is 50 unless overridden.
- No `pixel_values` are emitted.

Embedding ABI:

- Model outputs `embeddings` as `[B,sequence_length,128]`.
- For a list scorer, each item can be `[sequence_length_i,128]`; padded tensor form is `[N,max_sequence,128]`.
- Document/page and query embeddings can be cached independently before final scoring.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed SigLIP patch Conv2d to GEMM

Source pattern:

```text
Conv2d(C=3, out=1152, kernel=14, stride=14, padding=valid) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten NCHW patches [B,1024,3*14*14] -> GEMM(weight.T) -> BiasAdd -> [B,1024,1152]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == valid`, dilation 1, groups 1.
- Input rank 4 and layout NCHW unless an NHWC pass owns the whole region.
- Height and width exactly divisible by patch size; first target can require 448 x 448.
- Weight transform: `conv.weight.reshape(out_channels, in_channels * kh * kw)`.

Failure cases: interpolated position encodings, dynamic image sizes without patch divisibility metadata, non-NCHW caller tensors, or altered SigLIP variants.

Parity test sketch: compare patch embedding output before position add for random `[2,3,448,448]` fp32/bf16 inputs.

### Rewrite: ColPali scorer to tiled GEMM + reductions

Source pattern:

```text
einsum("bnd,csd->bcns").max(dim=3).sum(dim=2)
```

Replacement:

```text
For query batch tile Bq and passage tile Bp:
  scores4d = batched GEMM(q [Bq,Sq,D], p^T [Bp,D,Sp])
  max over Sp
  sum over Sq
```

Preconditions:

- Both embedding tensors are L2-normalized and padded tokens are zero.
- Embedding dim exactly 128 for native ColPali.
- Output orientation remains `[num_queries,num_passages]`.
- If padding masks are retained separately, scorer must ignore padded query tokens or rely on zeroed query embeddings matching source behavior.

Failure cases: unnormalized external embeddings, mixed dtypes/devices, score output requested on CPU while inputs are GPU, or very long variable sequences without tiling.

### Rewrite: image feature masked_scatter to indexed copy

Source pattern:

```text
inputs_embeds.masked_scatter(special_image_mask, image_features)
```

Replacement:

```text
Validate count(input_ids == image_token_id) == B * num_image_tokens
Copy image_features rows into known image-token slots.
```

Preconditions:

- Processor-generated document prompts put all image placeholders at the start.
- `num_images == 1` for first admission.
- Image token id and sequence length are static from config/preprocessor.

Failure cases: user-supplied custom prompt with image tokens not contiguous, multiple images per prompt, or `inputs_embeds` path without `input_ids`.

### Layout rewrite: NCHW patch region to NHWC/channel-last

Treat as guarded optimization only.

Preconditions:

- Processor-to-patch-embedding region is controlled.
- Rewrite `Conv2d`/window flatten weight layout and flatten order together.
- Downstream output must return token sequence `[B,num_patches,hidden]`, so the layout change is local.

No-layout guard regions:

- Token sequence tensors `[B,S,H]`.
- Gemma attention/head axes.
- ColPali L2 normalization `dim=-1`.
- Retrieval scorer axes `bnd,csd->bcns`, `max(dim=3)`, `sum(dim=2)`.

## 10. Kernel fusion candidates

Highest priority:

- ColPali projection + L2 normalize + attention-mask zeroing. Small but always on both query and page embedding paths.
- Gemma RMSNorm and gated MLP fusion inherited from PaliGemma/Gemma.
- GQA attention with RoPE and prefix-aware masks for the Gemma text body.
- SigLIP patch Conv2d-to-GEMM or optimized patch embedding for document indexing throughput.
- Tiled MaxSim scorer: GEMM plus max/sum reductions, preserving `[queries,passages]`.

Medium priority:

- PaliGemma image-token indexed copy replacing general masked scatter for processor-generated prompts.
- SigLIP LayerNorm + attention/MLP fusions for page indexing.
- Vision position add fused with patch projection output.
- Cache document image embeddings and optionally projected page embeddings before scoring.

Lower priority:

- Optional SigLIP positional interpolation.
- Autoregressive PaliGemma generation heads and LM logits.
- Adapter/LoRA merge at runtime.

## 11. Runtime staging plan

Stage 1: native config and weight loading for `vidore/colpali-v1.2-hf` / `v1.3-hf`; reject PEFT adapter repos and non-`model_type=colpali` debug snapshots.

Stage 2: implement ColPali wrapper parity using a delegated/stubbed PaliGemma output tensor: projection, L2 norm, mask zeroing, output ABI.

Stage 3: compose audited PaliGemma/SigLIP/Gemma encoder parity for image and query embedding passes at fixed 448 resolution, no cache, no generation.

Stage 4: implement retrieval scorer kernel or reference path with exact source orientation and variable-length padding behavior.

Stage 5: add graph rewrites for fixed patch embedding and image-token scatter, gated by processor-derived prompt/image invariants.

Stage 6: optimize document-index throughput and scorer tiling; add optional bf16/fp16 paths and document embedding cache workflows.

Stage 7: separately admit adapters, PaliGemma2/ColPali2, dynamic image sizes, and generation-style cache paths if product scope needs them.

## 12. Parity and validation plan

- Processor ABI tests: image path returns `input_ids`, `attention_mask`, `token_type_ids`, `labels`, `pixel_values`; query path returns no `pixel_values`; image+text in one call rejects.
- Wrapper random tests: feed random `[B,S,2048]` delegated hidden states through `Linear -> L2 -> mask` and compare fp32/bf16 against PyTorch.
- L2 edge test: masked padded positions become zero; non-padded vectors have norm near 1.
- PaliGemma image stitch test: generated prompt with 1024 image tokens copies projected vision features into the exact prefix slots.
- Single SigLIP patch embedding parity: NCHW `[1,3,448,448]` through patch conv/flatten/pos add.
- One Gemma layer parity with GQA/RoPE/prefix mask.
- End-to-end embedding parity on one document image and one query from the HF integration dataset.
- Scorer parity: list and padded tensor inputs with unequal sequence lengths; assert output `[num_queries,num_passages]` and diagonal ranking on the small HF test dataset.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4` for wrapper/scorer; bf16 end-to-end scores need looser absolute tolerance similar to upstream test, initially `atol<=1` for retrieval score matrix.

## 13. Performance probes

- Processor throughput: images/sec for resize/normalize/tokenization.
- SigLIP page encoder throughput by batch size and resolution.
- Gemma query encoder throughput for query lengths 16/32/50.
- Document embedding cache size: pages x sequence length x 128 x dtype.
- ColPali projection+normalization bandwidth.
- MaxSim scorer throughput over query/page tile sizes, sequence length, dtype, and CPU-vs-GPU output device.
- Patch Conv2d vs patch-GEMM rewrite at 448 and batch sizes 1/8/32.
- Dense attention backend comparison for SigLIP 1024-token noncausal attention and Gemma 1K-plus prefix sequences.
- Adapter merge/load overhead if PEFT repos are later admitted.

## 14. Skip/defer list

- Training losses and `labels`.
- PaliGemma LM head and text generation.
- KV cache/decode path for retrieval.
- Multiple images per prompt and arbitrary user-authored image token placement.
- Dynamic image sizes and SigLIP positional interpolation.
- PEFT/LoRA adapter runtime loading or merge.
- ColPali2/PaliGemma2/Gemma2 sliding-window variants.
- Multi-GPU tensor parallel and distributed scoring.
- Quantized or GGUF weight loading beyond ordinary dense safetensors.

## 15. Final implementation checklist

- [ ] Parse native `ColPaliConfig` and reject non-native/adapters for first target.
- [ ] Load dense safetensors including `embedding_proj_layer.*` and delegated `vlm.*` weights.
- [ ] Implement ColPali wrapper projection, L2 normalization, and mask zeroing.
- [ ] Preserve output ABI `embeddings [B,S,128]` plus optional hidden/image states as deferred metadata.
- [ ] Compose PaliGemma fixed-resolution image embedding path.
- [ ] Compose Gemma query/text embedding path with prefix-aware masks and 1-indexed RoPE positions.
- [ ] Implement or lower image-token feature stitch with count/contiguity guards.
- [ ] Implement MaxSim scorer with `[queries,passages]` orientation.
- [ ] Add guarded patch Conv2d-to-GEMM rewrite for NCHW 448/14 inputs.
- [ ] Add no-layout-translation guards around token sequence, normalization, attention, and scoring axes.
- [ ] Add processor ABI tests for image-only, text-only, and rejection of combined calls.
- [ ] Add wrapper, scorer, one-block, and end-to-end retrieval parity tests.
- [ ] Benchmark page indexing, query embedding, and scorer tile throughput.

## Gated gaps for DinoML admission

- Full PaliGemma/SigLIP/Gemma operator coverage must exist or be explicitly composed; ColPali source alone is not enough.
- `masked_scatter`/indexed image-token stitching needs a bounded implementation or guarded rewrite.
- Prefix-aware PaliGemma mask construction from `token_type_ids` is required for document embeddings.
- GQA attention with 1 KV head, RoPE, and bf16-friendly RMSNorm/MLP are required through the delegated Gemma body.
- NCHW processor/patch embedding is the semantic source layout; NHWC/channel-last is optimization-only behind guards.
- MaxSim scorer needs variable-length padding semantics and fixed output orientation `[num_queries,num_passages]`.
- PEFT adapter repos and ColPali2/PaliGemma2 variants should be rejected or routed to separate audits until adapter merge and Gemma2/sliding-window contracts are admitted.
