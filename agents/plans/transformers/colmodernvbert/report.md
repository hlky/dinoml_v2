# Transformers Family Audit: `colmodernvbert`

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ModernVBERT/colmodernvbert-merged, plus related ModernVBERT/colmodernvbert and ModernVBERT/colmodernvbert-base
Config source: local source defaults plus Hub JSON snapshots saved beside this report
Source files inspected:
- transformers/src/transformers/models/colmodernvbert/modular_colmodernvbert.py
- transformers/src/transformers/models/colmodernvbert/configuration_colmodernvbert.py
- transformers/src/transformers/models/colmodernvbert/modeling_colmodernvbert.py
- transformers/src/transformers/models/colmodernvbert/processing_colmodernvbert.py
- transformers/src/transformers/models/modernvbert/modeling_modernvbert.py
- transformers/src/transformers/models/modernvbert/configuration_modernvbert.py
- transformers/src/transformers/models/modernbert/modeling_modernbert.py
- transformers/src/transformers/models/siglip/modeling_siglip.py
- transformers/src/transformers/models/idefics3/image_processing_idefics3.py
Any missing files or assumptions:
- No gated checkpoint was encountered.
- Native `colmodernvbert` source is generated from `modular_colmodernvbert.py`; future source edits should inspect the modular file first.
- Official Hub snapshots found for `ModernVBERT/colmodernvbert-base` and `ModernVBERT/colmodernvbert-merged` ship `model_type: modernvbert`, not `colmodernvbert`.
- `ModernVBERT/colmodernvbert` is a PEFT LoRA adapter with no `config.json`; it points at `ModernVBERT/colmodernvbert-base`.
```

Small snapshots saved under this directory:

- `ModernVBERT__colmodernvbert-base/config.json`
- `ModernVBERT__colmodernvbert-merged/config.json`
- `ModernVBERT__colmodernvbert/adapter_config.json`
- `ModernVBERT__colmodernvbert*/preprocessor_config.json`
- `ModernVBERT__colmodernvbert*/processor_config.json`
- mirror/adapter snapshots for `sjoerdgunneweg/colmodernvbert_reproduction` and `zimble/colmodernvbert-ru-swap-rumodernbert`

## 2. High-Level Architecture

Primary DinoML target: visual document retrieval embeddings and late-interaction scoring, not autoregressive generation.

The native wrapper is a ColPali-style retrieval model:

```text
image/text preprocessing -> ModernVBert VLM -> token-level hidden states -> Linear(hidden -> embedding_dim) -> L2 normalize -> attention-mask zeroing -> multi-vector embeddings
query embeddings + passage embeddings -> MaxSim late interaction score matrix
```

The delegated VLM body is `AutoModel.from_config(config.vlm_config)`. For the official accessible checkpoints, that body is `modernvbert`: a SigLIP vision encoder plus a ModernBERT bidirectional text encoder with image-token embedding replacement.

Stage decomposition:

- CPU/data pipeline: image resize/split/pad/rescale/normalize, tokenization, image placeholder expansion, optional `mm_token_type_ids`.
- Vision encoder: SigLIP patch embedding and 12 noncausal ViT blocks over each real image crop.
- Image connector: pixel-shuffle-style token reduction from 1024 SigLIP patches to 64 image tokens for 512x512 crops, then bias-free projection to text hidden size.
- Text encoder: ModernBERT bidirectional encoder over text plus inserted image embeddings.
- Retrieval head: per-token projection to 128 dims, L2 normalization, mask multiplication.
- Scoring: batch-padded ColBERT MaxSim: `einsum("bnd,csd->bcns").max(dim=3).sum(dim=2)`.

Independently cacheable pieces: passage document embeddings can be precomputed and stored; query embeddings are separate. Vision features are cacheable before text fusion only if the exact prompt/image-token layout is fixed.

## 3. Important Config Dimensions

Official base and merged config snapshots are identical in operator-significant fields and are `modernvbert` configs, not wrapper configs.

| Field | Value | Source |
|---|---:|---|
| top-level model_type | `modernvbert` | `config.json` |
| image_token_id | 50407 | `config.json` |
| pixel_shuffle_factor | 4 | `config.json` |
| text hidden_size | 768 | `config.json` |
| text num_hidden_layers | 22 | `config.json` |
| text num_attention_heads | 12 | `config.json` |
| text head_dim | 64 | inferred from source: `768 / 12` |
| text intermediate_size | 1152 | `config.json` |
| text activation | `gelu` | `config.json` |
| text attention bias | false | `config.json` |
| text MLP bias | false | `config.json` |
| text norm bias | false | `config.json` |
| text max_position_embeddings | 7999 | `config.json` |
| text local_attention | 128 | `config.json` |
| text sliding half-window | 64 | source property `local_attention // 2`; attention passes `65` inclusive window |
| RoPE theta | 160000 for full and sliding | `config.json` |
| vocab_size | 50408 | `config.json` |
| vision model_type | `siglip_vision_model` | `config.json` |
| vision image_size | 512 | `config.json` |
| vision patch_size | 16 | `config.json` |
| vision patches per crop | 1024 | inferred `(512 / 16)^2` |
| vision hidden_size | 768 | `config.json` |
| vision num_hidden_layers | 12 | `config.json` |
| vision num_attention_heads | 12 | `config.json` |
| vision intermediate_size | 3072 | `config.json` |
| vision activation | `gelu_pytorch_tanh` | `config.json` |
| retrieval embedding_dim | 128 | native wrapper source default |
| processor image_seq_len | 64 | `processor_config.json` |

Representative checkpoint sweep:

| Repo | Hub role | Config basis | Operator-significant notes |
|---|---|---|---|
| `ModernVBERT/colmodernvbert-base` | base dense model | `model_type: modernvbert` | 252.1M params from Hub metadata; SigLIP + ModernBERT; no wrapper `embedding_dim` in config. |
| `ModernVBERT/colmodernvbert-merged` | merged dense model | `model_type: modernvbert` | Same config as base; docstring names this checkpoint for native wrapper but Hub config routes to ModernVBert. |
| `ModernVBERT/colmodernvbert` | LoRA adapter | no `config.json`; `adapter_config.json` | PEFT LoRA rank 32, alpha 32, dropout 0.1, target regex hits `model.text_model` `Wo/Wqkv/Wi` and `custom_text_proj`; base is `ModernVBERT/colmodernvbert-base`. |
| `sjoerdgunneweg/colmodernvbert_reproduction` | mirror/reproduction adapter | no `config.json`; `adapter_config.json` | PEFT adapter over `ModernVBERT/modernvbert`; same LoRA target family. |
| `zimble/colmodernvbert-ru-swap-rumodernbert` | mirror/variant adapter | no `config.json`; `adapter_config.json` | Adapter over `ModernVBERT/colmodernvbert-base`; Russian tokenizer/processor metadata but no dense config fetched. |

## 3a. Family Variation Traps

- Native `ColModernVBertConfig` defaults `vlm_config` to `modernvbert`, but accessible official dense configs are already `modernvbert`. DinoML should route these as ModernVBert bodies plus retrieval/PEFT handling, or require an actual wrapper config.
- `ColModernVBertConfig` in the generated file contains duplicated `vlm_config is None` / dict conversion logic from ColQwen2 inheritance. Treat source behavior as current truth; do not rely on the second Qwen2 branch being reachable after the first branch initializes ModernVBert.
- The neural body is delegated through `AutoModel.from_config`; unsupported `vlm_config.model_type` values must be allowlisted or rejected.
- Text attention alternates full and sliding attention: layers 0,3,6,9,12,15,18,21 are full; all others are sliding.
- This is bidirectional retrieval encoding. There is no autoregressive decode loop or KV cache requirement for the first target.
- Placeholder replacement uses boolean masks, cumsum, block indexing, `zeros_like`, indexed assignment, and `torch.where`; DinoML can specialize this because processor-generated image tokens are contiguous blocks of length `image_seq_len`.
- Processor image splitting creates row/column tags and a global image crop. Sequence length per image can exceed 64 when the image is split.
- Source processor defaults to `channels_first`; initial semantic lowering should preserve NCHW. NHWC is an optimization only for local Conv2d/attention regions with axis guards.
- SigLIP position interpolation exists but the ModernVBert path normally resizes crops to 512, matching the static position table.
- LoRA adapter repos add a loading contract, not a core op contract: rank-32 low-rank deltas on ModernBERT `Wqkv`, `Wi`, `Wo`, and a projection module name advertised as `custom_text_proj`.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Embedding lookup for token IDs `[B, S] -> [B, S, 768]`.
- View/reshape/transpose/permute/contiguous for QKV packing, SigLIP patch flattening, ModernVBert pixel shuffle, attention layouts.
- `flatten(2).transpose(1,2)` after Conv2d patch embedding.
- Boolean comparisons and reductions for real-image filtering: `(pixel_values == 0).sum(...) != nb_values`.
- Boolean mask cumsum, modulo, integer division, cumsum offsets, indexed gather/scatter-like assignment for image-token replacement.
- `where(mask.unsqueeze(-1), image_embeds, inputs_embeds)`.
- Pad and stack in processor output path; this can remain CPU/data pipeline initially.

Neural primitives:

- Conv2d patch embedding: `Conv2d(3 -> 768, kernel=16, stride=16, padding=valid)` on NCHW.
- Dense GEMMs/Linear:
  - SigLIP Q/K/V/O: `768 -> 768`, bias enabled by SigLIP source defaults.
  - SigLIP MLP: `768 -> 3072 -> 768`.
  - ModernVBert connector: `Linear(12288 -> 768, bias=False)` because `768 * 4^2 = 12288`.
  - ModernBERT Wqkv: `768 -> 2304`, bias false.
  - ModernBERT Wo: `768 -> 768`, bias false.
  - ModernBERT gated MLP Wi: `768 -> 2304`, split into activation and gate, bias false.
  - ModernBERT MLP Wo: `1152 -> 768`, bias false.
  - Retrieval projection: `768 -> 128`, bias true in native wrapper/ColPali superclass.
- LayerNorm with eps `1e-6` for SigLIP and `1e-5` for ModernBERT; ModernBERT norms have no bias.
- GELU and tanh-approx GELU (`gelu_pytorch_tanh`).
- Residual adds, dropout as identity for inference, elementwise multiply for gated MLP and mask application.
- L2 normalization: `x / norm(x, dim=-1, keepdim=True)`.

Attention primitives:

- SigLIP dense noncausal self-attention over 1024 patch tokens per crop, 12 heads, head dim 64.
- ModernBERT dense bidirectional full attention and bidirectional sliding-window attention, 12 heads, head dim 64.
- RoPE on ModernBERT Q/K before attention.
- Attention masks for padding and sliding windows. No causal mask for the retrieval target.

Preprocessing-coupled ops:

- Idefics3 image resize/split/pad/rescale/normalize in CPU/data pipeline.
- Prompt expansion with `<fake_token_around_image>`, `<image>`, `<global-img>`, row/col tags.
- Optional `mm_token_type_ids` constructed from fake image token spans.

Quantized/packed weights:

- No native quantized weight format in source. PEFT adapters are source-coupled loading metadata; dense fallback is base/merged checkpoint loading.

Position/rotary:

- SigLIP learned 2D patch position embedding flattened to sequence.
- ModernBERT RoPE per layer type with `rope_theta=160000`.

Retrieval scoring:

- Pad variable-length query/passage embeddings.
- Batched similarity tensor `[Bq, Bp, Sq, Sp]`, max over passage tokens, sum over query tokens.

## 5. Layer/Block Breakdown

ModernVBert image branch:

```text
pixel_values [B, Nimg, 3, H, W]
flatten real images -> [Nreal, 3, H, W]
patch_mask = unfold(pixel_attention_mask, patch=16, stride=16).sum > 0
SigLIP:
  Conv2d patch embed -> [Nreal, 768, 32, 32]
  flatten/transpose + learned position -> [Nreal, 1024, 768]
  repeat 12:
    x = x + MHA(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
  post LayerNorm -> [Nreal, 1024, 768]
connector:
  pixel shuffle token rearrange -> [Nreal, 64, 12288]
  Linear(12288 -> 768, no bias) -> [Nreal, 64, 768]
```

ModernVBert text branch:

```text
input_ids [B, S]
tok_embedding + LayerNorm + dropout -> [B, S, 768]
replace each block of 64 <image> token embeddings with connector output
ModernBERT encoder, repeated 22:
  if layer 0: attention input norm is Identity else LayerNorm
  qkv = Linear(768 -> 2304, no bias)
  q,k,v = reshape/split to [B, 12, S, 64]
  q,k = RoPE(q,k)
  attn = full bidirectional or sliding bidirectional attention
  x = residual + Linear(attn, 768 -> 768, no bias)
  y = LayerNorm(x)
  u, gate = Linear(y, 768 -> 2304, no bias).chunk(2)
  x = x + Linear(GELU(u) * gate, 1152 -> 768, no bias)
final LayerNorm -> [B, S, 768]
```

Retrieval head:

```text
emb = Linear(768 -> 128, bias=True)(last_hidden_state.to(proj_dtype))
emb = emb / norm(emb, dim=-1, keepdim=True)
if attention_mask: emb = emb * attention_mask[..., None]
return [B, S, 128]
```

Physical/logical sharing:

- The native wrapper subclasses ColPali and owns one delegated VLM plus one projection layer.
- The dense official configs are ModernVBert configs; adding ColModernVBert wrapper semantics around them must avoid cloning the VLM weights.
- LoRA adapters logically modify selected base weights at load time.

## 6. Attention Requirements

Primary target attention is encoder-style only.

- Causal or noncausal: noncausal/bidirectional.
- Self-attention or cross-attention: self-attention only.
- Head geometry: MHA, 12 Q heads, 12 K/V heads, head dim 64 for both SigLIP and ModernBERT.
- Query/key/value widths: all 768 total width.
- Rectangular attention: not in core source; query/key lengths match per branch.
- Masking:
  - SigLIP can consume patch attention masks generated from pixel padding.
  - ModernBERT builds one full bidirectional mask and one bidirectional sliding-window mask.
  - Sliding attention uses local attention 128; source passes `sliding_window = local_attention // 2 + 1 = 65`.
- Packed/varlen: source removes padded images before the vision model, but token attention itself is ordinary padded sequence attention.
- KV cache: not required. `past_key_values` should be rejected or ignored for first retrieval parity unless a delegated body unexpectedly exposes it.
- FlashAttention/SDPA: source marks support. A DinoML first pass can use dense attention for parity, then add full/sliding bidirectional FlashAttention-style kernels.

## 7. Position Encoding and Custom Math

ModernBERT RoPE:

```python
def modernbert_rope(position_ids, head_dim=64, theta=160000.0):
    inv = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = (inv[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate = lambda x: cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
```

Precompute opportunity: for fixed sequence buckets, precompute cos/sin for each layer type. Full and sliding layer types use the same theta in inspected configs, so they can share the table for these checkpoints.

SigLIP position encoding: learned position embedding for a 32x32 patch grid. Bicubic interpolation is source-supported when `interpolate_pos_encoding=True`; first integration can require fixed 512x512 crops and reject interpolation.

ModernVBert pixel shuffle:

```text
[B, 1024, 768] -> view [B,32,32,768]
-> group 4x4 spatial neighborhoods into channel dimension
-> [B,8,8,12288] -> [B,64,12288]
```

This is a fixed reshape/permute pattern for 512x512 crops and `pixel_shuffle_factor=4`.

## 8. Preprocessing and Input Packing

Processor defaults:

- `return_tensors="pt"`.
- Text padding: longest.
- Images: `Idefics3ImageProcessor`, `do_convert_rgb=True`, `do_resize=True`, `do_image_splitting=True`, `do_pad=True`, `do_rescale=True`, `do_normalize=True`.
- Resize longest edge to 2048 before splitting; split/resize crops to `max_image_size.longest_edge=512`.
- Output data format: channels-first.
- Normalize with mean/std `[0.5, 0.5, 0.5]`; rescale by `1/255`.
- Model graph receives `pixel_values [B, max_images, 3, H, W]` and optionally `pixel_attention_mask [B, max_images, H, W]`.

Image placeholder ABI:

- Special tokens include `<image>` id 50407, `<fake_token_around_image>` id 50406, `<end_of_utterance>` id 50405, `<global-img>` id 50368, and 6x6 row/col tags ids 50369-50404 in the official tokenizer snapshot.
- `image_seq_len=64` should equal `(image_size / patch_size)^2 / pixel_shuffle_factor^2`.
- For unsplit single image: fake token, global tag, 64 image tokens, fake token.
- For split images: each crop gets row/col tag plus 64 image tokens, then a global crop block. The processor records row/col counts and sequence lengths.
- `create_mm_token_type_ids` marks spans starting at fake-image-token positions as image modality spans.

Image embedding stitch:

- Source accepts arbitrary prompt positions but processor-generated image token runs are contiguous blocks.
- The model checks that the number of `<image>` tokens per sample is divisible by `patch_size` where `patch_size` is actually the connector output sequence length (64).
- It computes block offsets from per-sample token counts and assigns image features into the masked positions.
- DinoML can lower this as a guarded indexed row copy: require processor-produced contiguous image-token blocks, exact image token count equals `num_image_blocks * 64`, and `image_hidden_states` block order matches prompt order.

Query path:

- `process_queries` prepends `query_prefix` and appends 10 `<end_of_utterance>` augmentation tokens by default.
- No image tensors are used for pure text queries.

Postprocessing/scoring:

- `score_retrieval` is not a neural module forward; it pads ragged query/passage embedding lists and computes a `[n_queries, n_passages]` score matrix.
- Output orientation is queries as rows and passages as columns.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: SigLIP Patch Conv2d -> Linear/GEMM

Source pattern:

```text
Conv2d(3 -> 768, kernel=16, stride=16, padding=valid) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten(NCHW, 16x16 non-overlap) -> Linear(768 input pixels -> 768) -> [B, 1024, 768]
```

Preconditions:

- `kernel_size == stride == 16`.
- `padding == valid`, `dilation == 1`, `groups == 1`.
- Input crop height/width divisible by 16.
- Preserve NCHW semantic order unless a guarded NHWC pass rewrites both window flatten and weights.

Weight transform:

```python
w = conv.weight.reshape(768, 3 * 16 * 16)
b = conv.bias
```

Failure cases: non-512 crop with interpolated positions, non-divisible spatial dims, grouped conv, layout uncertainty.

Parity sketch: compare patch embeddings for random NCHW crops before and after rewrite.

### Rewrite: ModernVBert Pixel Shuffle + Linear -> Gathered GEMM

Source pattern: fixed reshape/permute regrouping of `[B,1024,768]` into `[B,64,12288]`, followed by bias-free linear.

Replacement: either materialize the gathered 4x4 patch neighborhood and GEMM, or transform the projection weight into 16 per-offset GEMMs accumulated into 64 output tokens.

Preconditions:

- `image_size=512`, `patch_size=16`, `pixel_shuffle_factor=4`.
- SigLIP patch grid is square 32x32.
- Output token order matches source permute sequence.

Failure cases: interpolated/non-square patch grids or changed shuffle factor.

### Rewrite: ModernBERT Fused QKV Projection

Source pattern:

```text
Wqkv(hidden) -> view [B,S,3,12,64] -> unbind q,k,v
```

Replacement: one GEMM `768 -> 2304` with split order `[q, k, v]` along the `dim=-3` view.

Preconditions: `hidden_size == num_heads * head_dim`, attention bias false for official configs, source weight layout preserved as PyTorch Linear `[out, in]`.

### Rewrite: Gated MLP Fusion

Source pattern:

```text
Wi(x).chunk(2) -> GELU(input) * gate -> Wo(...)
```

Replacement: fused biasless GEMM plus GELU-multiply epilogue feeding second GEMM.

Preconditions: `hidden_activation="gelu"`, `mlp_bias=False`, inference dropout zero.

### Rewrite: Image Masked Scatter -> Guarded Indexed Copy

Source pattern:

```text
image_mask = input_ids == image_token_id
image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask]]
where(image_mask[...,None], image_embeds, inputs_embeds)
```

Replacement: copy each 64-row image feature block into known contiguous token ranges, then use copied embedding buffer as text input.

Preconditions:

- Input produced by `ColModernVBertProcessor`.
- Every image token block is contiguous length 64.
- Number/order of blocks equals real processed images.
- No arbitrary user-inserted `<image>` tokens outside processor expansion.

Failure cases: arbitrary mask positions, missing/fake token span mismatch, token count not divisible by 64.

### Rewrite: Retrieval MaxSim to Tiled GEMM + Reductions

Source pattern:

```text
einsum("bnd,csd->bcns") -> max over passage tokens -> sum over query tokens
```

Replacement: for each query/passsage batch tile, GEMM query embeddings against passage embeddings transposed, reduce max over passage dimension, then reduce sum over query dimension.

Preconditions: embeddings are L2-normalized, padded rows are zeroed by attention mask, output orientation `[queries, passages]`.

## 10. Kernel Fusion Candidates

Highest priority:

- ModernBERT LayerNorm + QKV GEMM + RoPE preparation. This dominates 22 text layers and feeds both full/sliding attention.
- Bidirectional sliding-window attention. Two thirds of text layers use sliding attention; dense fallback will overcompute on long document prompts.
- Gated MLP GELU-multiply. The official intermediate size is small but repeated 22 times; a fused activation multiply avoids extra memory traffic.
- Image embedding stitch as guarded indexed copy. General boolean scatter is too broad for DinoML; a processor-guarded copy is much safer.

Medium priority:

- SigLIP patch Conv2d-to-GEMM and 12-block dense vision attention. Passage encoding cost matters for indexing pipelines.
- Pixel-shuffle + connector projection specialization.
- Retrieval MaxSim tiled scoring kernel for large query/passage batches.
- L2 normalization and attention-mask multiply fusion after retrieval projection.

Lower priority:

- SigLIP position interpolation; fixed 512 crops can be required first.
- Processor image resize/split GPU acceleration; CPU pipeline is acceptable for initial parity.
- PEFT LoRA folded-weight materialization; useful for adapter repos but not required for merged dense checkpoints.

## 11. Runtime Staging Plan

Stage 1: admission and config routing.

- Accept dense `modernvbert` configs as the delegated body for this family.
- Reject native `colmodernvbert` configs whose `vlm_config.model_type` is not allowlisted.
- Decide whether first target is dense merged checkpoint only or PEFT adapter plus base.

Stage 2: ModernVBert encoder parity without retrieval head.

- Validate SigLIP image features, connector output, image-token stitch, and ModernBERT last hidden state on small synthetic inputs.
- Stub processor with fixed 512x512 one-image inputs and already-expanded token IDs.

Stage 3: retrieval head parity.

- Add `Linear(768 -> 128)`, L2 normalize, attention-mask zeroing.
- Compare embeddings for text-only queries and single-image document prompts.

Stage 4: processor-compatible image packing.

- Support `pixel_values [B,N,3,H,W]`, `pixel_attention_mask`, real-image filtering, and multiple image blocks.
- Add guards for contiguous 64-token image blocks.

Stage 5: scoring parity.

- Implement or call a MaxSim scoring helper with query-row/passage-column orientation.
- Allow passage embeddings to be cached independently.

Stage 6: optimized kernels.

- Replace dense sliding attention with local attention kernels.
- Add Conv2d-to-GEMM, QKV/RoPE fusion, gated MLP fusion, and tiled MaxSim.

Stage 7: adapter support.

- Fold LoRA adapters into dense weights at load time for the official adapter repo, then reuse the dense path.

## 12. Parity and Validation Plan

- Config parse tests:
  - Native wrapper default config.
  - Dense `ModernVBERT/colmodernvbert-base` / merged `modernvbert` config routed as delegated body.
  - PEFT adapter metadata recognized but not treated as a native graph config.
- Unit tests:
  - Pixel shuffle: random `[B,1024,768]` versus Transformers source function.
  - Image-token stitch: contiguous blocks, multiple images per sample, malformed block count rejection.
  - ModernBERT RoPE for full/sliding layer types.
  - Gated MLP and retrieval L2 normalization.
- Single-block parity:
  - One SigLIP encoder layer.
  - One ModernBERT full-attention layer and one sliding-attention layer.
- End-to-end graph parity:
  - Text-only query embeddings.
  - Single 512x512 image document embeddings.
  - Split-image document prompt with row/col tags and global image block.
- Scoring parity:
  - Ragged lists and padded tensors.
  - Output shape `[n_queries, n_passages]`.
  - Zeroed padded rows do not affect MaxSim.
- Suggested tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for blocks; slightly looser after full model.
  - fp16/bf16: `rtol=5e-2`, `atol=5e-2` initially for full encoder, then tighten by kernel.

## 13. Performance Probes

- Processor throughput: images/sec for resize/split/pad/normalize at common document resolutions.
- Vision encoder throughput: crops/sec for 512x512 NCHW inputs; sweep number of crops per document.
- Connector throughput: pixel shuffle + projection microbench.
- Text encoder throughput: sequence-length sweep, especially sequences with many image tokens.
- Full vs sliding attention comparison: dense fallback versus local kernel for `S` near 1024, 2048, 4096, 7999.
- Retrieval head throughput: projection + L2 normalize for `[B,S,768]`.
- MaxSim throughput: query count x passage count x sequence length sweep; memory traffic of `[Bq,Bp,Sq,Sp]` materialization versus tiled reduction.
- Cache probe: precomputed passage embedding storage size and scoring throughput from cached embeddings.
- PEFT probe: one-time LoRA merge cost versus per-run low-rank delta application.

## 14. Skip/Defer List

- Training losses and labels, including suffix label construction.
- Masked LM, sequence classification, token classification, QA, and multiple-choice heads from delegated ModernVBert/ModernBERT source.
- Autoregressive generation, beam search, and KV cache.
- Arbitrary user-provided image token masks; require processor-compatible contiguous blocks first.
- SigLIP interpolated position embeddings for non-512 crops.
- GPU implementation of image resize/split preprocessing.
- General PEFT runtime application; fold supported LoRA adapters into dense weights first.
- Non-ModernVBert delegated `vlm_config` values until separately audited.
- NHWC/channel-last global layout translation. Keep NCHW semantics unless a local guarded rewrite owns all axis changes.

## 15. Final Implementation Checklist

- [ ] Parse native `ColModernVBertConfig` and dense `modernvbert` checkpoint configs with explicit routing.
- [ ] Load dense ModernVBert weights without cloning delegated VLM parameters.
- [ ] Decide first adapter policy: reject PEFT, fold LoRA, or require merged dense checkpoint.
- [ ] Implement SigLIP NCHW patch embedding and encoder parity.
- [ ] Implement ModernVBert pixel shuffle connector.
- [ ] Implement guarded image-token embedding stitch.
- [ ] Implement ModernBERT full and sliding bidirectional attention with RoPE.
- [ ] Implement ModernBERT gated MLP.
- [ ] Implement retrieval projection, L2 normalization, and attention-mask zeroing.
- [ ] Implement/call MaxSim scoring with `[queries, passages]` output orientation.
- [ ] Add fixed 512x512 single-image parity tests.
- [ ] Add split-image processor-layout parity tests.
- [ ] Add text-only query embedding parity tests.
- [ ] Add MaxSim ragged/padded scoring parity tests.
- [ ] Benchmark vision encoder, text encoder, retrieval head, and MaxSim independently.
