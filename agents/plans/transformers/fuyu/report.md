# Fuyu Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: adept/fuyu-8b
Config source: https://huggingface.co/adept/fuyu-8b/raw/main/config.json
Source files inspected:
- X:/H/transformers/src/transformers/models/fuyu/configuration_fuyu.py
- X:/H/transformers/src/transformers/models/fuyu/modeling_fuyu.py
- X:/H/transformers/src/transformers/models/fuyu/image_processing_fuyu.py
- X:/H/transformers/src/transformers/models/fuyu/image_processing_pil_fuyu.py
- X:/H/transformers/src/transformers/models/fuyu/processing_fuyu.py
- X:/H/transformers/src/transformers/models/persimmon/configuration_persimmon.py
- X:/H/transformers/src/transformers/models/persimmon/modeling_persimmon.py
- X:/H/transformers/src/transformers/activations.py
Local snapshots:
- agents/plans/transformers/fuyu/_sources/adept_fuyu-8b_config.json
- agents/plans/transformers/fuyu/_sources/adept_fuyu-8b_preprocessor_config.json
- agents/plans/transformers/fuyu/_sources/adept_fuyu-8b_generation_config.json
- agents/plans/transformers/fuyu/_sources/adept_fuyu-8b_tokenizer_config.json
- agents/plans/transformers/fuyu/_sources/adept_fuyu-8b_model.safetensors.index.json
Any missing files or assumptions: only one official native Fuyu checkpoint was found. Historical revisions of
adept/fuyu-8b were checked for config drift; they are architecturally equivalent, with one revision omitting
`text_config` and relying on the FuyuConfig default Persimmon text model.
```

The current in-library implementation is authoritative for this report. Fuyu composes the `persimmon` model family via
`AutoModel.from_config(config.text_config)`, so decoder operator coverage belongs to Persimmon while Fuyu owns the
image patch projection, placeholder construction, multimodal embedding replacement, and generation handoff.

## 2. High-level architecture

Fuyu is a multimodal causal decoder with no standalone vision encoder. The image branch is direct patch projection:
CPU/data pipeline image preprocessing produces flattened non-overlapping image patches, a single learned linear layer
maps each patch vector into the text hidden size, and those projected patch embeddings replace special image placeholder
token embeddings inside the decoder input sequence.

```text
image/text preprocessing -> flattened image patches + image placeholder token stream
-> token embedding lookup + patch Linear(2700 -> 4096)
-> masked_scatter replacement in input embeddings
-> Persimmon causal decoder prefill
-> decode with self-attention KV cache; image tensors omitted after first cached step
-> LM head logits/sampling
```

Stage decomposition:

- CPU/data pipeline: resize-if-needed, pad to processor canvas, rescale/normalize, crop back to patch-aligned unpadded
  bounds, NCHW patchify, placeholder/newline token construction, coordinate token scaling for box/point prompts, left
  padding and attention mask construction.
- Independently cacheable projection: image patches can be projected once to `[batch, num_patches, hidden]` before
  embedding replacement. There is no image encoder KV cache.
- Prefix construction: text embeddings plus patch embeddings are stitched into a single causal sequence. This stage is
  shape-sensitive because variable image sizes change the number of placeholder tokens.
- Prefill: full multimodal sequence through Persimmon decoder.
- Decode: generated token steps use Persimmon self-attention cache only. `prepare_inputs_for_generation` drops
  `image_patches` and `image_patches_indices` after the first iteration when cache is enabled.

The image projection, embedding stitch, Persimmon single block, prefill logits, and cached decode can be validated
independently.

## 3. Important config dimensions

Effective `adept/fuyu-8b` values:

| Field | Value | Source |
| --- | ---: | --- |
| architecture | `FuyuForCausalLM` | HF config |
| text backbone | `persimmon` | HF `text_config` / FuyuConfig default |
| vocab_size | 262144 | HF config |
| hidden_size | 4096 | HF config |
| num_hidden_layers | 36 | HF config |
| num_attention_heads | 64 | HF config |
| num_key_value_heads | 64 inferred | source: Persimmon has MHA, no GQA field |
| head_dim | 64 | source equation `4096 / 64` |
| intermediate_size | 16384 | HF config |
| activation | `relu2` = `square(relu(x))` | HF config + activations.py |
| layer_norm_eps | 1e-5 | HF config |
| qk_layernorm | true | HF config |
| attention/dropout | 0.0 / 0.0 | HF config |
| max_position_embeddings | 16384 | HF config |
| rope_theta | 25000.0 | HF config historical field, normalized by config machinery |
| partial_rotary_factor | 0.5 | HF config / source BC default |
| rotary_ndims | 32 | `head_dim * partial_rotary_factor` |
| image_size | 300 | HF model config; not the current processor canvas |
| processor canvas | 1080 x 1920 | HF preprocessor config and source defaults |
| patch_size | 30 x 30 | HF config/source processor |
| patch vector width | 2700 | `30 * 30 * 3` |
| full-canvas patches | 2304 | `36 * 64` for 1080 x 1920 |
| full-canvas image tokens | 2340 | 2304 placeholders + 36 newline tokens |
| image_token_id | 71011 | FuyuConfig default; processor derives from `|SPEAKER|` tokenizer encoding |
| image_newline_id | tokenizer-derived | processor derives from `|NEWLINE|` |
| eos_token_id | 71013 | HF config/generation config |
| dtype | bfloat16 | HF config |
| use_cache | true | HF config |
| tie_word_embeddings | false | HF config |

Representative checkpoint/config sweep:

| Checkpoint/config | Dimensions | Operator-significant notes |
| --- | --- | --- |
| `adept/fuyu-8b` main | 36 layers, H=4096, heads=64, patch=30, bf16 | Current official native Fuyu config; includes `text_config: {"model_type":"persimmon"}`. |
| `adept/fuyu-8b` rev `d804d9c8` | same | Historical config omits `text_config`; current `FuyuConfig` supplies a default Persimmon text config from top-level fields. |
| `adept/fuyu-8b` rev `b5c3f725` | same | Same native architecture as main. |
| `FuyuConfig()` source defaults | same base dimensions, `image_size=300` | Source defaults are similar to checkpoint but processor source defaults use 1080 x 1920 canvas. |
| `PersimmonConfig()` composed text default | same decoder dimensions | Useful for decoder parity because Fuyu delegates to Persimmon `AutoModel`. |

## 3a. Family variation traps

- The current source has no vision transformer. Treat image handling as patchify + `Linear(patch_dim -> hidden)`, not
  an encoder/projector stack.
- `image_patches_indices` is produced by the processor and an index-copy helper exists, but current `FuyuModel.forward`
  does not use it. It uses `input_ids == image_token_id` or an embedding equality fallback to build a boolean mask,
  then `inputs_embeds.masked_scatter(...)`.
- Placeholder count must equal flattened patch feature element count. Newline tokens are in `input_ids` but are not
  replaced by image features.
- Variable image sizes are supported by cropping padded images to `ceil(unpadded_h / patch_h) * patch_h` and
  `ceil(unpadded_w / patch_w) * patch_w`, then adding one newline token per patch row.
- Processor source and model config disagree in presentation: model config has `image_size=300`, while current
  processor config/source uses a 1080 x 1920 canvas. Runtime should rely on actual processor outputs, not `image_size`.
- Text decoder is Persimmon MHA, not GQA/MQA. There is no `num_key_value_heads` field in the current source.
- Attention uses one packed `query_key_value` linear with split order `[q, k, v]` inside shape
  `[B, S, num_heads, 3, head_dim]`.
- Q/K LayerNorm is per-head over `head_dim` after QKV split and before head transpose/RoPE.
- RoPE is partial: only the first 32 of 64 head dimensions are rotated for the checkpoint.
- Activation is ungated squared ReLU, not SwiGLU/GEGLU.
- All dense projections in Persimmon attention/MLP have bias; LM head has no bias.
- `tie_word_embeddings=false`; do not assume LM head and token embedding aliasing even though `_tied_weights_keys`
  metadata exists for tie-capable loading.
- Source modeling is sequence-major hidden tensors `[B, S, H]` and attention `[B, heads, S, D]`. Processor image tensors
  are NCHW before patchification. NHWC is only a guarded local optimization around patch extraction/projection.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW resize/pad/rescale/normalize in data pipeline, or equivalent CPU preprocessor parity.
- Non-overlap patch extraction from `[B, C, H, W]` with `patch_h=patch_w=30`, stride 30, no overlap.
- `unfold`/view/permute/reshape equivalent producing patch vectors in order `[patch_row, patch_col, y, x, channel]`.
- Embedding lookup `[B, S] -> [B, S, 4096]`.
- Boolean compare, unsqueeze, expand, masked scatter/indexed replacement.
- Left padding, concatenate, stack, arange, nonzero/count_nonzero for processor parity if brought into runtime.
- Final logits slice for `logits_to_keep`.

Neural network primitives:

- `Linear(2700 -> 4096)` image projection with bias.
- Token embedding `[262144, 4096]`.
- Per layer: packed QKV `Linear(4096 -> 12288)` with bias, output `Linear(4096 -> 4096)` with bias.
- Per layer: MLP `Linear(4096 -> 16384)` with bias, `relu2`, `Linear(16384 -> 4096)` with bias.
- LayerNorm with affine params and eps 1e-5 for input LN, post-attn LN, final LN, Q LN, K LN.
- Residual adds; dropout is configured 0.0 for inference.
- LM head `Linear(4096 -> 262144)` without bias.

Attention primitives:

- Causal self-attention only, MHA with 64 Q/K/V heads and head_dim 64.
- Packed QKV split `[q, k, v]`.
- Per-head Q/K LayerNorm before RoPE.
- Partial RoPE on first 32 dimensions of Q/K.
- KV cache update after RoPE; cached keys are post-QK-LN and post-RoPE.
- Backend-compatible attention interface: eager, SDPA, FlashAttention, and flex-attention are advertised by source.

Position/rotary ops:

- `position_ids` default arange starts at `past_key_values.get_seq_length()` during cached decode.
- RoPE cos/sin computed in fp32 then cast to hidden dtype.
- Dynamic RoPE framework is present through generic rope utilities, but checkpoint uses default rope with theta 25000.

Generation/cache ops:

- DynamicCache creation when `use_cache=True` and no cache is passed.
- Per-layer key/value cache shape before repeat expansion is `[B, 64, cached_seq, 64]`; there is no repeat expansion.
- First-step multimodal prefill; later cached decode drops image patch tensors.
- `logits_to_keep=0` means all logits because `slice(0, None)` is used; positive integer keeps trailing tokens.

Preprocessing-coupled ops:

- Resize only when either image dimension exceeds canvas; preserve aspect ratio.
- Pad NCHW image to canvas with constant value 1.0, then rescale by 1/255 and normalize with mean/std 0.5.
- Patch-aligned crop based on unpadded resized dimensions during variable-sized token construction.
- Placeholder token rows terminated by newline tokens.
- Optional box/point prompt coordinate scaling and post-generation coordinate de-scaling.

Scatter/indexed update ops:

- Required current model path: boolean mask and `masked_scatter` from projected patch embeddings into token embeddings.
- Optional/deferred source helper: index-based copy using `image_patches_indices`, nonnegative positions, and per-batch
  source indices. This is not invoked by current `forward`.

Packed/varlen metadata:

- No `cu_seqlens` or packed attention metadata in source. Variable image sizes manifest as variable sequence lengths and
  placeholder counts before normal causal attention.

## 5. Layer/block breakdown

Image prefix construction:

```text
input_ids: [B, S]
image_patches: [B, P, 2700]
inputs_embeds = Embedding(input_ids) -> [B, S, 4096]
patch_embeddings = Linear(2700 -> 4096)(image_patches) -> [B, P, 4096]
special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand([B, S, 4096])
inputs_embeds = masked_scatter(inputs_embeds, special_image_mask, patch_embeddings)
```

The scatter is flatten-order sensitive: `masked_scatter` consumes `patch_embeddings` in row-major flattened order and
therefore requires processor placeholder order to match patch order.

Persimmon decoder block, repeated 36 times:

```text
residual = x
x = LayerNorm(4096, eps=1e-5)(x)
qkv = Linear(4096 -> 12288, bias=True)(x)
q, k, v = view(qkv, [B, S, 64, 3, 64]).select(q/k/v)
q = LayerNorm(64, eps=1e-5)(q)   # if qk_layernorm
k = LayerNorm(64, eps=1e-5)(k)
q, k, v = transpose to [B, 64, S, 64]
q_rot, q_pass = split(q, [32, 32])
k_rot, k_pass = split(k, [32, 32])
q_rot, k_rot = RoPE(q_rot, k_rot, cos, sin)
q = concat(q_rot, q_pass, dim=-1)
k = concat(k_rot, k_pass, dim=-1)
k, v = cache.update(k, v, layer_idx)  # when cache provided
attn = causal_attention(q, k, v, mask, scale=1/sqrt(64))
x = residual + Linear(4096 -> 4096, bias=True)(attn)
residual = x
x = LayerNorm(4096, eps=1e-5)(x)
x = Linear(4096 -> 16384, bias=True)(x)
x = square(relu(x))
x = Linear(16384 -> 4096, bias=True)(x)
x = residual + x
```

Final:

```text
x = final LayerNorm(4096)
logits = Linear(4096 -> 262144, bias=False)(selected x positions)
```

## 6. Attention requirements

Fuyu requires the Persimmon causal self-attention variant:

- causal: yes.
- self-attention: yes; no cross-attention.
- heads: 64 query heads, 64 key/value heads, head_dim 64.
- MHA/MQA/GQA: MHA only in inspected source.
- mask: `create_causal_mask` combines causal masking, attention mask, cache length, and position ids. Processor produces
  left-padded attention masks for batched multimodal prompts.
- packed/varlen: no varlen kernel descriptors in source.
- sliding/local attention: not present.
- RoPE: partial RoPE before cache update.
- KV cache: per layer K/V stored as `[B, 64, T, 64]`; K is stored after QK LayerNorm and RoPE, V after transpose and
  before attention. No KV-head repeat.
- SDPA/FlashAttention compatibility: source advertises SDPA, FlashAttention, attention backend, and flex attention.
  Eager fallback computes `q @ k.T * scale`, adds mask, softmax in fp32, casts to query dtype, dropout, then `@ v`.

For Fuyu generation, image patch embeddings are part of the prefill token sequence and are not a separate KV cache
type. After first iteration with cache, the generation wrapper omits `image_patches` and `image_patches_indices`.

## 7. Position encoding and custom math

Persimmon RoPE computes inverse frequencies with:

```python
dim = int(head_dim * partial_rotary_factor)
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos, sin = emb.cos(), emb.sin()
```

Application is only to the rotary prefix of Q/K:

```python
def fuyu_partial_rope(q, k, cos, sin, rotary_ndims=32):
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q_rot, q_pass], -1), cat([k_rot, k_pass], -1)
```

`position_ids` are ordinary absolute token positions over the combined image/text stream. Image patches and newline
tokens consume position ids like text tokens. Cos/sin for default RoPE can be precomputed up to max position for fixed
theta and rotary dimension, but dynamic rope utilities in the source mean non-default future configs should be routed
through a separate audit before assuming static tables.

Custom activation:

```python
def relu2(x):
    return torch.square(torch.relu(x))
```

## 8. Preprocessing and input packing

Runtime tensors produced by the current processor:

- `input_ids`: `[B, S]`, left-padded when batching samples with different lengths.
- `attention_mask`: `[B, S]`, zeros for left padding and ones for real placed tokens.
- `image_patches`: usually tensor `[B, P, 2700]` for single-image samples; batching behavior can leave a list when
  multiple samples have different patch counts, so DinoML should define a padded/bucketed contract for compiled runs.
- `image_patches_indices`: `[B, S]` with patch source indices at image placeholder positions and `-1` elsewhere.
  Current model forward ignores it; it remains useful metadata for an indexed-copy rewrite.
- Optional `mm_token_type_ids` can be returned by processor but is not consumed by `FuyuModel.forward`.

Image flow:

```text
input image -> channel-first tensor
-> resize if larger than 1080x1920 while preserving aspect ratio
-> pad bottom/right to 1080x1920 with value 1.0
-> rescale and normalize: x = (x / 255 - 0.5) / 0.5
-> crop to ceil(unpadded_h / 30) * 30 by ceil(unpadded_w / 30) * 30
-> patchify NCHW into [num_patch_rows * num_patch_cols, 2700]
-> create [placeholder... newline] rows in token stream
```

For a resized unpadded image with patch grid `Gh = ceil(h / 30)`, `Gw = ceil(w / 30)`:

- patch count `P = Gh * Gw`.
- image token count `P + Gh` because each row gets one newline token.
- only the `P` placeholder positions are replaced by projected patch embeddings.

Coordinate prompt coupling:

- `<box>...</box>` and `<point>...</point>` are rewritten to special tokens.
- Coordinates are divided by 2 and scaled to transformed image space before tokenization.
- Postprocessing can convert generated coordinate token spans back to text box/point markup and multiply by 2 after
  inverse scaling. This is end-to-end task behavior, not part of the decoder graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: direct patch projection as WindowFlatten + GEMM

Source pattern:

```text
NCHW image -> unfold(2, 30, 30) -> unfold(3, 30, 30) -> contiguous
-> view(B, C, P, 30, 30) -> permute(0, 2, 3, 4, 1)
-> reshape(B, P, 2700) -> Linear(2700 -> 4096)
```

Replacement:

```text
PatchExtractNHWCOrder(NCHW, kh=kw=stride=30) -> GEMM_RCR/GEMM_RRR + bias -> [B, P, 4096]
```

Preconditions:

- `patch_height == patch_width == stride == 30`.
- no padding inside the patch extraction; image already cropped/padded to patch divisibility.
- `groups == 1`, `dilation == 1`, RGB channels = 3.
- flattened order exactly matches source: patch row-major, then within-patch y, x, channel.
- dynamic image dims must be multiples of 30 after processor crop.

Weight transform:

```python
# PyTorch linear: y = x @ weight.T + bias, weight [4096, 2700]
gemm_rhs = weight.T  # or keep as RCR RHS depending on DinoML GEMM layout
```

Layout constraints:

- Initial semantic graph should preserve NCHW processor output.
- A local NHWC patch kernel is safe only if it explicitly produces the same flattened order as source. Do not globally
  translate the processor/image region to NHWC unless resize/pad/normalize consumers are covered.

Failure cases:

- non-RGB images, non-30 patch sizes, disabled padding with non-divisible dims, or an alternate processor backend whose
  patch order differs.

Parity test sketch:

- random `[1, 3, 60, 90]` and `[2, 3, 1080, 1920]` tensors; compare source `patchify_image` plus PyTorch linear against
  lowered patch GEMM in fp32 and bf16.

### Rewrite: masked_scatter image replacement to indexed copy/scatter

Source pattern:

```text
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, patch_embeddings)
```

Replacement:

```text
dst_positions = nonzero(input_ids == image_token_id)
inputs_embeds[dst_positions, :] = patch_embeddings.reshape(-1, hidden)
```

Preconditions:

- number of placeholder tokens equals `B * P` for tensorized image patches, or equals sum of per-sample patch counts for
  ragged handling.
- placeholder order is exactly processor patch order.
- newline tokens are not selected.
- no `inputs_embeds` equality fallback is needed when `input_ids` is available.

Shape equations:

```text
input_ids: [B, S]
patch_embeddings: [B, P, H] or packed [sum_P, H]
selected elements: count(input_ids == image_token_id) * H == numel(patch_embeddings)
```

Failure cases:

- generation decode steps with no image patches, truncated prompts where processor and image tensor counts no longer
  match, or caller passes only `inputs_embeds` and no `input_ids`.

Parity test sketch:

- build synthetic `input_ids` with placeholders and newlines; compare masked_scatter and indexed assignment for uneven
  image row counts and left-padded batches.

### Rewrite: packed QKV linear split

Source pattern:

```text
Linear(4096 -> 12288) -> view(B, S, 64, 3, 64) -> select q/k/v
```

Replacement:

```text
single packed GEMM -> Q/K/V views with split order per head [q, k, v]
```

Preconditions:

- hidden size divisible by heads.
- storage split is interleaved per head, not all-Q/all-K/all-V row blocks.
- Q/K LayerNorm consumes `[B, S, heads, head_dim]` before transpose.

Weight transform:

- Prefer no transform initially; preserve packed output and view/split.
- If splitting into three GEMMs, rows must be gathered with `view(out=64, 3, 64)` order, not contiguous thirds.

Failure cases:

- a checkpoint converted to a different packed layout.

### Rewrite: QK LayerNorm + partial RoPE + attention fusion

Source pattern:

```text
QKV -> split -> per-head LN(Q/K) -> transpose -> split rotary/pass dims -> RoPE -> concat -> attention
```

Replacement:

```text
QKV GEMM -> fused per-head QK LN -> fused partial RoPE -> FlashAttention prefill/decode
```

Preconditions:

- `qk_layernorm=true`.
- head_dim static 64, rotary_ndims static 32.
- cache stores post-RoPE K.
- backend attention supports MHA with `[B, heads, S, D]`, causal mask, and external position ids.

Failure cases:

- non-default rope types, disabled qk_layernorm, nonzero attention dropout in training, attention backend requiring a
  different mask convention.

## 10. Kernel fusion candidates

Highest priority:

- Image `Linear(2700 -> 4096)` as batched GEMM over all patches. Full-canvas prefill has 2304 patch rows per image.
- Packed QKV GEMM with correct interleaved split order.
- Per-head Q/K LayerNorm plus partial RoPE feeding FlashAttention for prefill and decode.
- Causal MHA FlashAttention with KV cache for `[B, 64, S, 64]`.
- Last-token-only or `logits_to_keep` LM head GEMM to avoid full vocabulary projection for unneeded positions.

Medium priority:

- LayerNorm kernels for hidden size 4096 and head size 64.
- `relu2` MLP activation fused with the MLP up/down GEMM epilogue where practical.
- Embedding replacement indexed scatter kernel for multimodal prefix construction.
- Patch extraction + projection fused kernel for fixed patch size and RGB.

Lower priority:

- Coordinate token preprocessing/postprocessing acceleration; keep CPU initially.
- PIL backend parity; the Torchvision backend is more relevant for tensorized preprocessing.
- Full dynamic/ragged batching for variable patch counts; start with static buckets.

## 11. Runtime staging plan

Stage 1: parse Fuyu config, resolve `text_config` to Persimmon defaults, load weights, and expose image projection plus
one Persimmon block.

Stage 2: implement processor-compatible patchify/projection parity for fixed image tensors. Stub coordinate
postprocessing and multi-sample ragged batching.

Stage 3: implement embedding stitch using `input_ids == image_token_id` and a strict placeholder-count check. Prefer an
indexed-copy lowering while preserving masked_scatter parity.

Stage 4: run multimodal prefill through N decoder layers with eager/composed attention.

Stage 5: add decode with Persimmon KV cache and generation rule that drops image tensors after the first cached step.

Stage 6: enable optimized attention and QK-LN/RoPE fusions under guards.

Stage 7: add bucketed variable image shapes and batching policy for patch counts/sequence lengths.

Stage 8: add end-to-end coordinate postprocessing and broader processor compatibility.

Initially stub/defer PIL-specific preprocessing, multi-image/subsequence generalization, training loss, and advanced
generation modes beyond greedy/sampling parity.

## 12. Parity and validation plan

- Patchify unit tests: compare source `FuyuImageProcessor.patchify_image` for several NCHW shapes divisible by 30,
  including 30x30, 60x90, 1080x1920, and cropped variable-size cases.
- Image projection tests: random patches through `vision_embed_tokens`, fp32 tolerance `1e-5/1e-5`, bf16 tolerance
  around `2e-2/2e-2`.
- Scatter tests: compare source masked_scatter with DinoML indexed replacement for placeholder/newline streams,
  left-padded batches, and count mismatch failures.
- RoPE tests: compare cos/sin and partial application for position ids starting at 0 and at a nonzero cache offset.
- Single-layer Persimmon tests: one decoder layer with QK LayerNorm, partial RoPE, and eager attention.
- After-N-layer tests: 1, 2, 36 layers on short text-only sequences, then multimodal prefixes.
- Prefill logits: `adept/fuyu-8b` prompt/image fixture through Transformers vs DinoML for selected logits.
- Decode token parity: first decode step with cache from multimodal prefill; verify image tensors are not required.
- End-to-end image-text output smoke: bus/chart/skateboard fixtures from the HF repo, allowing token-level parity for a
  deterministic generation config before looser text checks.

## 13. Performance probes

- CPU preprocessing throughput: resize/pad/normalize/patchify split by image resolution.
- Patch projection throughput: patches/sec for `P` sweeps: 1, 100, 576, 2304.
- Prefix stitch throughput: masked_scatter/indexed-copy cost versus sequence length and patch count.
- Prefill-only latency/tokens per second for text-only and image+text prompts.
- Decode-only tokens/sec with cache for batch sizes 1, 2, 4 and context lengths after image prefix.
- KV cache memory: `layers * 2 * B * heads * seq * head_dim * dtype_size`; for bf16 Fuyu this is roughly
  `36 * 2 * B * 64 * seq * 64 * 2` bytes.
- Attention backend comparison: eager composed vs SDPA/FlashAttention equivalent for long image prefixes.
- LM head probe: full logits for all prefix positions versus `logits_to_keep=1`.
- Variable image shape sweep: patch grids such as 1x1, 10x16, 18x32, 36x64 to separate patch projection, sequence
  length, and cache effects.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and dropout behavior.
- Beam search and advanced generation controllers beyond ordinary cache-aware generation.
- Multi-image and multi-subsequence prompt packing beyond what current processor hardcodes for common use.
- PIL backend acceleration; keep as CPU/reference path.
- Runtime coordinate parser/postprocessor acceleration.
- Non-default/dynamic RoPE variants unless a real Fuyu checkpoint requires them.
- Quantization and GGUF ingestion.
- Multi-GPU tensor parallelism.
- Global NHWC conversion. Only guarded local patch extraction/projection layout rewrites should be considered first.

## 15. Final implementation checklist

- [ ] Parse `FuyuConfig` and synthesize Persimmon `text_config` defaults when omitted.
- [ ] Load Fuyu/Persimmon weights, preserving untied embedding and LM head semantics.
- [ ] Implement NCHW patchify parity for 30x30 non-overlap patches.
- [ ] Implement `Linear(2700 -> 4096)` image patch projection.
- [ ] Implement placeholder-count validation and image embedding replacement.
- [ ] Decide first compiled contract for variable patch counts: static buckets or padded packed patches.
- [ ] Implement Persimmon packed QKV split with per-head `[q, k, v]` layout.
- [ ] Implement per-head Q/K LayerNorm.
- [ ] Implement partial RoPE with theta 25000 and rotary dimension 32.
- [ ] Implement causal MHA prefill and decode KV cache.
- [ ] Implement `relu2` MLP activation.
- [ ] Implement `logits_to_keep` / last-token LM head optimization.
- [ ] Add patchify, scatter, RoPE, single-layer, prefill, and decode parity tests.
- [ ] Add guarded WindowFlatten+GEMM rewrite for image projection.
- [ ] Add guarded QK-LN/RoPE/attention fusion path.
- [ ] Benchmark preprocessing, patch projection, prefill, decode, LM head, and KV memory.
