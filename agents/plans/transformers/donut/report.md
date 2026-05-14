# Donut DinoML Operator Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: naver-clova-ix/donut-base; naver-clova-ix/donut-base-finetuned-cord-v2; naver-clova-ix/donut-base-finetuned-docvqa; naver-clova-ix/donut-base-finetuned-rvlcdip; naver-clova-ix/donut-proto
Config source: HF config.json/preprocessor_config.json/tokenizer_config.json/special_tokens_map.json snapshots under _sources/hf_configs
Source files inspected: configuration_donut_swin.py, modeling_donut_swin.py, image_processing_donut.py, processing_donut.py, image_processing_pil_donut.py; delegated composition sources modeling_vision_encoder_decoder.py and modeling_mbart.py
Any missing files or assumptions: no processor_config.json or generation_config.json exists in sampled repos; decoder implementation is delegated to MBartForCausalLM through VisionEncoderDecoderModel, not owned by the donut source directory.
```

Pinned source URLs:

- [configuration_donut_swin.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/donut/configuration_donut_swin.py)
- [modeling_donut_swin.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/donut/modeling_donut_swin.py)
- [image_processing_donut.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/donut/image_processing_donut.py)
- [processing_donut.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/donut/processing_donut.py)
- [modeling_vision_encoder_decoder.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py)
- [modeling_mbart.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/mbart/modeling_mbart.py)

No sampled official config returned 401/403. `processor_config.json` and `generation_config.json` returned 404 for the sampled official repos, so generation defaults are taken from `config.json` and the generic `GenerationMixin`/VisionEncoderDecoder path.

## 2. High-level architecture

Primary target: document image-to-structured-text generation using a DonutSwin vision encoder plus a delegated causal MBART decoder inside `VisionEncoderDecoderModel`.

```text
CPU image preprocessing -> DonutSwin encoder -> encoder hidden sequence cache -> MBART decoder prefill/decode with cross-attention -> token ids -> DonutProcessor.token2json
```

Stage decomposition:

- CPU/data pipeline: RGB image orientation alignment, resize-to-shortest-edge, aspect-preserving thumbnail, centered padding, rescale and normalize to `pixel_values`.
- Encoder: DonutSwin patch Conv2d, shifted-window self-attention stages, patch merging, final token sequence. Encoder output can be cached per image/question pair before decoder generation.
- Prefix/input construction: task prompt tokens such as `<s_cord-v2>`, `<s_docvqa>`, question markup, or class prompt are tokenizer-side inputs. The processor does not build these prompts automatically beyond tokenization.
- Decoder prefill/decode: MBART causal self-attention with encoder-decoder cross-attention and KV cache.
- Postprocessing: text decode and `DonutProcessor.token2json` parse generated `<s_key>...</s_key>` spans and categorical `<label/>` leaves into JSON-like Python structures.

The `donut` source directory owns only DonutSwin and processor behavior. End-to-end Donut requires composing already-auditable `vision_encoder_decoder` and `mbart` behavior.

## 3. Important config dimensions

Source defaults are Swin-like and not representative of production Donut: `image_size=224`, `embed_dim=96`, `depths=(2,2,6,2)`, `num_heads=(3,6,12,24)`, `window_size=7`, `patch_size=4`, `mlp_ratio=4`, `qkv_bias=True`, `use_absolute_embeddings=False`, `layer_norm_eps=1e-5`.

Representative checkpoint sweep:

| Checkpoint | Scope | Encoder type | Image HxW | Patch grid | Final grid | Depths | Heads | Window | Decoder | Decoder layers | d_model | Vocab | Max target positions | Processor effective HxW | Align long axis |
|---|---:|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `naver-clova-ix/donut-base` | common base | `donut-swin` | 2560x1920 | 640x480 | 80x60 | 2,2,14,2 | 4,8,16,32 | 10 | MBART causal LM | 4 | 1024 | 57525 | 1536 | 2560x1920 | true |
| `naver-clova-ix/donut-base-finetuned-cord-v2` | receipt extraction | `donut-swin` | 1280x960 | 320x240 | 40x30 | 2,2,14,2 | 4,8,16,32 | 10 | MBART causal LM | 4 | 1024 | 57580 | 768 | 1280x960 | false |
| `naver-clova-ix/donut-base-finetuned-docvqa` | document QA | `donut-swin` | 2560x1920 | 640x480 | 80x60 | 2,2,14,2 | 4,8,16,32 | 10 | MBART causal LM | 4 | 1024 | 57532 | 128 | 2560x1920 | false |
| `naver-clova-ix/donut-base-finetuned-rvlcdip` | document classification | `donut-swin` | 2560x1920 | 640x480 | 80x60 | 2,2,14,2 | 4,8,16,32 | 10 | MBART causal LM | 4 | 1024 | 57544 | 8 | 2560x1920 | false |
| `naver-clova-ix/donut-proto` | legacy/prototype | `swin` | 2048x1536 | 512x384 | 64x48 | 2,2,18,2 | 4,8,16,32 | 8 | MBART causal LM | 4 | 1024 | 57524 | 768 | 2048x1536 | true |

Source-derived encoder dimensions for the common base/CORD/docvqa checkpoints:

| Stage | Channels | Heads | Head dim | Blocks | Window tokens | MLP hidden |
|---|---:|---:|---:|---:|---:|---:|
| Patch embedding | 128 | n/a | n/a | n/a | n/a | n/a |
| Stage 1 | 128 | 4 | 32 | 2 | 100 | 512 |
| Stage 2 | 256 | 8 | 32 | 2 | 100 | 1024 |
| Stage 3 | 512 | 16 | 32 | 14 | 100 | 2048 |
| Stage 4 | 1024 | 32 | 32 | 2 | 100 | 4096 |

## 3a. Family variation traps

- `donut-proto` uses `encoder.model_type="swin"`, not `donut-swin`; route it through the Swin audit or treat it as a legacy variant, not native DonutSwin.
- HF processor `size` appears as `[width, height]` in older JSON snapshots; current `DonutImageProcessor.__init__` reverses tuple/list input. Effective target size matches `encoder.image_size` as HxW.
- Production Donut image grids are huge. A `2560x1920` page creates 307,200 patch tokens before downsampling and 4,800 final encoder tokens for cross-attention.
- `max_position_embeddings` varies by task: 8 for RVL-CDIP classification, 128 for DocVQA, 768 for CORD/proto, 1536 for base.
- Prompt/task special tokens differ per fine-tune and are tokenizer/postprocessing contracts, not model graph modules.
- `use_absolute_embeddings` exists in source but sampled Donut configs use relative position bias only. If enabled, absolute embeddings include a class-position row even though DonutSwin patch embeddings do not prepend a CLS token; require parity investigation before admitting such configs.
- Shifted-window logic uses NHWC-like internal tensors inside attention blocks even though model input/output contracts are NCHW image tensors and `[B, S, C]` token tensors.
- `always_partition` and `interpolate_pos_encoding` are optional source paths. First integration can reject or defer them for sampled Donut checkpoints.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image input `[B,3,H,W]`.
- Pad right/bottom to patch-size divisibility before patch Conv2d.
- Conv2d patch embedding `Conv2d(3 -> 128, kernel=4, stride=4, bias=True)` for common Donut.
- Flatten spatial `NCHW -> [B,H'W',C]` via `flatten(2).transpose(1,2)`.
- Reshape token sequence to `[B,H,W,C]`, pad H/W to window multiples, cyclic `roll` on H/W, window partition/reverse, crop padded windows, contiguous/view.
- Patch merging gather order: concatenate `[row::2, col::2]` for `col in 0..1`, `row in 0..1` along channel axis, then `LayerNorm(4C)` and `Linear(4C -> 2C, bias=False)`.
- Optional hidden-state output reshape `[B,S,C] -> [B,C,H,W]`.
- Decoder token embeddings, position IDs, last-token logit slicing.

Neural network primitives:

- LayerNorm over channel/hidden dimensions, eps `1e-5`.
- Linear Q/K/V per window: `Linear(C -> C, bias=qkv_bias)` separately for q, k, v.
- Attention output `Linear(C -> C, bias=True)`.
- MLP `Linear(C -> 4C) -> GELU -> Linear(4C -> C)`.
- Residual adds; dropout/drop-path is identity in inference.
- Adaptive average pool over encoder token sequence only for `DonutSwinForImageClassification`, optional/deferred for primary Donut generation.
- MBART decoder: embedding scale by `sqrt(d_model)` when `scale_embedding=True`, learned positions, LayerNorm, Linear `1024 -> 4096 -> 1024`, tied LM head `Linear(1024 -> vocab, bias=False)`.

Attention primitives:

- DonutSwin noncausal local MHA inside fixed windows, with shifted-window mask for alternating blocks.
- Relative position bias table indexed by precomputed `relative_position_index`.
- MBART causal self-attention with cache.
- MBART encoder-decoder cross-attention over DonutSwin final tokens, cacheable cross K/V.
- SDPA/Flash/Flex backend compatibility is source-owned by MBART attention; DonutSwin encoder attention uses eager matmul/softmax in this source.

Position/relative-bias ops:

- DonutSwin 2-D relative position index and per-head bias add.
- Optional absolute position embedding and bicubic interpolation path for unsupported/admitted configs.
- MBART learned positional embeddings with offset from `past_key_values_length`.

Generation/cache ops:

- `VisionEncoderDecoderModel` encoder output cache: `[B, S_enc, 1024]`, where `S_enc=4800` for 2560x1920 and `S_enc=1200` for 1280x960.
- MBART self-attention KV cache per decoder layer: `[B, 16, T, 64]`.
- MBART cross-attention cache per decoder layer: `[B, 16, S_enc, 64]`, updated once and reused through `EncoderDecoderCache`.
- Causal mask and optional decoder padding mask.

Preprocessing-coupled ops:

- Optional `rot90(..., k=3)` long-axis alignment.
- Resize shortest edge to `min(target_h,target_w)` with bilinear resampling.
- Aspect-preserving thumbnail clamped to target box.
- Center padding to target HxW.
- Rescale by `1/255` through backend defaults and normalize with checkpoint mean/std `[0.5,0.5,0.5]`.
- XLM-R tokenizer for prompt/labels; special token markup for JSON/class leaves.

## 5. Layer/block breakdown

Image processor:

```text
image CHW/RGB tensor
if do_align_long_axis: rotate 90 degrees clockwise when input and target orientations differ
if do_resize: resize shortest edge to min(target_h, target_w)
if do_thumbnail: shrink larger side to fit target while preserving aspect
if do_pad: centered pad to target_h x target_w
rescale and normalize -> pixel_values [B,3,H,W]
```

DonutSwin patch embedding:

```text
pixel_values [B,3,H,W]
x = pad right/bottom to divisible by 4
x = Conv2d(3 -> 128, kernel=4, stride=4, bias=True)
grid = [H/4, W/4]
x = flatten spatial -> transpose -> [B, grid_h * grid_w, 128]
x = LayerNorm(128)
```

DonutSwin stage, repeated by `depths[i]` with alternating shift `0, window_size//2`:

```text
x_seq [B, H*W, C]
shortcut = x_seq
x = LayerNorm(C)(x_seq)
x = reshape [B,H,W,C]
x = pad H/W to multiples of window_size
if shifted: x = roll(x, shifts=(-shift, -shift), dims=(H,W))
windows = partition -> [B*num_windows, window_size^2, C]
q,k,v = Linear(C -> C, bias=True) separately
scores = q @ k.T / sqrt(C/heads)
scores += relative_position_bias[heads, window_tokens, window_tokens]
if shifted: scores += window attention mask values {0, -100}
attn = softmax(scores, dim=-1)
context = attn @ v
context = Linear(C -> C, bias=True)
x = reverse windows, reverse roll, crop padding, reshape [B,H*W,C]
x = shortcut + x
y = LayerNorm(C)(x)
y = Linear(C -> 4C, bias=True) -> GELU -> Linear(4C -> C, bias=True)
x = x + y
```

Patch merging after stages 1-3:

```text
x [B,H*W,C] -> [B,H,W,C]
x = pad bottom/right if H or W odd
x = concat(x[:,0::2,0::2], x[:,1::2,0::2], x[:,0::2,1::2], x[:,1::2,1::2], dim=-1)
x = reshape [B, ceil(H/2)*ceil(W/2), 4C]
x = LayerNorm(4C)
x = Linear(4C -> 2C, bias=False)
```

VisionEncoderDecoder + MBART decoder:

```text
encoder_hidden = DonutSwin(pixel_values).last_hidden_state  # [B,S_enc,1024]
decoder_inputs = token_embedding(input_ids) * sqrt(1024) + learned_positions
for each of 4 decoder layers:
  h = LayerNorm(h)
  h = h + causal_self_attention(h, cache)
  h = LayerNorm(h)
  h = h + cross_attention(query=h, key_value=encoder_hidden, cross_cache)
  h = LayerNorm(h)
  h = h + Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
h = final LayerNorm
logits = tied_lm_head(h[:, logits_to_keep:, :])
```

## 6. Attention requirements

DonutSwin shifted-window attention:

- Noncausal encoder self-attention.
- MHA, not GQA/MQA.
- Per-stage heads/head dim: 4/32, 8/32, 16/32, 32/32.
- Query/key/value widths all equal stage channel width.
- Query length and key/value length are fixed per window: `window_size^2`, usually 100.
- Alternating blocks use cyclic shifts of 5 pixels/tokens for window size 10, then an additive mask with `0` for same shifted partition and `-100` for disallowed pairs.
- Relative position bias is added before mask and softmax.
- No KV cache, no causal mask, no packed/varlen support.
- Eager matmul/softmax path only in `modeling_donut_swin.py`; this is likely slow for large page batches without a fused window attention kernel.

MBART decoder attention:

- Causal self-attention: MHA 16 heads, head dim 64, q/k/v width 1024, cache `[B,16,T,64]`.
- Cross-attention: decoder queries length `T_dec`, key/value length `S_enc` from DonutSwin final tokens. For base/docvqa/rvlcdip `S_enc=4800`; for CORD `S_enc=1200`.
- Cross-attention K/V are cached once in `EncoderDecoderCache.cross_attention_cache`; cached tensors are after linear projection and head reshape.
- `encoder_attention_mask` is `None` in `VisionEncoderDecoderModel.forward`; there is no image padding mask for DonutSwin final tokens.
- MBART dispatch can use `_attn_implementation` through `ALL_ATTENTION_FUNCTIONS`; preserve query scaling, mask creation, and cache update order.

## 7. Position encoding and custom math

DonutSwin relative position index:

```python
def donut_relative_position_index(window_h, window_w):
    coords = meshgrid(arange(window_h), arange(window_w))  # [2, Wh, Ww]
    flat = flatten(coords, start_dim=1)
    rel = flat[:, :, None] - flat[:, None, :]
    rel = permute(rel, (1, 2, 0))
    rel[:, :, 0] += window_h - 1
    rel[:, :, 1] += window_w - 1
    rel[:, :, 0] *= 2 * window_w - 1
    return rel.sum(-1)
```

Shifted-window mask:

```python
def shifted_window_mask(height_pad, width_pad, window, shift):
    img_mask = zeros([1, height_pad, width_pad, 1])
    # fill 3x3 regions around window and shift boundaries with ids 0..8
    windows = window_partition(img_mask, window).view(-1, window * window)
    mask = windows[:, None, :] - windows[:, :, None]
    return where(mask != 0, -100.0, 0.0)
```

Precompute per window size:

- `relative_position_index` and the gather order into `relative_position_bias_table`.
- Static shifted-window masks for fixed padded stage grids and fixed batch-independent shapes.

Dynamic per input:

- Processor resize/orientation/padding.
- Stage H/W after patch embedding and patch merging when admitting dynamic image sizes.
- MBART position IDs depend on `past_key_values_length`.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- `DonutImageProcessor` groups images by shape, then applies orientation alignment, resize, thumbnail, padding, rescale, and normalization.
- `do_align_long_axis=True` rotates images clockwise when input long axis disagrees with target long axis.
- `do_resize=True` first resizes shortest edge to target shortest edge.
- `do_thumbnail=True` then preserves aspect ratio while shrinking to fit the target HxW box.
- `do_pad=True` pads to exact target size with centered padding. Source has `random_padding` support in helper, but `_preprocess` passes `False`.
- Raw checkpoint JSON uses list `size` values like `[1920, 2560]`; current processor reverses list/tuple input, so effective HxW is `[2560, 1920]`.

GPU/runtime work:

- `pixel_values` enters model as NCHW `[B,3,H,W]`.
- No image attention mask or grid metadata is emitted by the processor.
- No multimodal scatter into text embeddings. The image is encoded separately and consumed by decoder cross-attention.
- Prompt tokens are ordinary decoder input IDs. Example special tokens include `<s_cord-v2>`, `<s_docvqa>`, `<s_question>`, `<s_answer>`, `<s_rvlcdip>`, and categorical `<invoice/>`-style tokens.

Postprocessing:

- `DonutProcessor.token2json` parses generated strings by looking for `<s_key>...</s_key>` spans.
- Nested fields are parsed recursively if content contains more `<s_...>` tags.
- Leaf fields split on `<sep/>`.
- Categorical leaves that exactly match an added vocab token like `<invoice/>` are converted to `invoice`.
- If no structured tags are found, output is `{"text_sequence": tokens}`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> WindowFlatten + GEMM

Source pattern: `Conv2d(3 -> embed_dim, kernel=patch_size, stride=patch_size, padding=0, groups=1)`.

Replacement:

```text
PadToMultiple(H,W,patch) -> WindowFlatten(NCHW, patch_h, patch_w, stride=patch) -> MatMul(weight_flat.T) -> BiasAdd -> Reshape [B, H/patch*W/patch, C]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Preserve DonutSwin source right/bottom padding before flattening.
- Flatten order must match PyTorch Conv2d NCHW channel/kernel order.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Layout constraints:

- This is a safe local NHWC candidate only when the flatten/GEMM/reshape region is fully controlled.
- Public graph input remains NCHW; a layout pass must not globally rewrite `dim=1` channel assumptions.

Failure cases:

- Dynamic sizes not divisible by patch without applying source padding.
- Nonstandard patch tuple not reflected in flatten order.

Parity sketch: compare Conv2d output before flatten to GEMM output for random NCHW images with divisible and non-divisible H/W.

### Rewrite: patch merging gather -> fused 2x2 merge GEMM

Source pattern: `view [B,H,W,C]`, pad odd H/W, concatenate four parity slices on channel, `LayerNorm(4C)`, `Linear(4C -> 2C, bias=False)`.

Replacement:

```text
PadOddBottomRight -> Gather2x2Channels(order=[00,10,01,11]) -> LayerNorm -> MatMul
```

Preconditions:

- Gather order exactly follows `for col in range(2) for row in range(2)`.
- Padding only bottom/right.
- Consumer expects token sequence `[B, ceil(H/2)*ceil(W/2), 2C]`.

Layout constraints:

- A channel-last internal layout is natural here, but the output is a sequence. Protect surrounding NCHW/public-output reshapes with `no_layout_translation()` unless all consumers are controlled.

Failure cases:

- Reordering slices as `[00,01,10,11]` silently changes weights.
- Omitting odd-size padding changes sequence length.

Parity sketch: random sequence/grid tests for even and odd H/W, compare merged tokens.

### Rewrite: shifted-window partition/roll/mask canonicalization

Source pattern: `LayerNorm -> [B,H,W,C] -> pad -> roll -> window_partition -> attention -> window_reverse -> roll back -> crop`.

Replacement:

```text
WindowAttention2D(input, H, W, window, shift, rel_bias, mask_policy=donut_swin)
```

Preconditions:

- Noncausal MHA with equal q/k/v width.
- Fixed square `window_size`.
- Mask values exactly `0` and `-100.0`.
- Relative bias added before shift mask and softmax.

Layout constraints:

- This is the highest-value NHWC fusion region because source already uses `[B,H,W,C]` internally.
- Do not rewrite the initial NCHW image processor/model boundary or hidden-state output `permute(0,3,1,2)` unless all consumers are layout-aware.

Failure cases:

- `min(input_resolution) <= window_size` mutates window/shift behavior unless `always_partition=True`.
- Dynamic shapes require recomputing padded H/W and shifted masks.

Parity sketch: one block parity for shift 0 and shift 5, with H/W divisible and non-divisible by window size.

### Rewrite: last-token-only logits

Source pattern: MBART causal LM computes `lm_head(hidden_states[:, slice_indices, :])` with `logits_to_keep`.

Replacement:

```text
SliceDecodePositions -> MatMul(tied_embedding.T)
```

Preconditions:

- Generation path only requires last token logits.
- LM head weight tied to decoder token embedding.

Failure cases:

- Training/loss parity or beam scoring that needs full prefix logits.

Parity sketch: compare full logits last column vs sliced logits for random hidden states.

## 10. Kernel fusion candidates

Highest priority:

- DonutSwin shifted-window attention kernel: document images create many 10x10 windows; fusing pad/roll/partition/QKV/relative-bias/mask/softmax/reverse removes large temporary traffic.
- Patch embedding Conv2d-to-GEMM: first operation touches very large pages and is a clean local lowering.
- Patch merging fused gather + LayerNorm + Linear: repeated three times with large token grids; order-sensitive but profitable.
- MBART cross-attention with cached encoder K/V: base images produce 4,800 encoder tokens, so decode cost is dominated by repeated cross-attention.

Medium priority:

- Encoder LayerNorm + Q/K/V projections per stage.
- Relative position bias gather/add fused into window attention score setup.
- MBART decoder self-attention Flash/SDPA with cache.
- Last-token-only LM head and tied embedding weight reuse.

Lower priority:

- Absolute position interpolation path; sampled configs do not require it.
- Classification head pooling for `DonutSwinForImageClassification`.
- DropPath/dropout kernels, because inference treats them as identity.

## 11. Runtime staging plan

Stage 1: parse composite `VisionEncoderDecoderConfig`; reject `donut-proto` or route to Swin because it uses `encoder.model_type="swin"`.

Stage 2: implement Donut image processor parity on CPU/data pipeline for fixed checkpoint target sizes.

Stage 3: load DonutSwin weights and run patch embedding plus one shifted-window block parity.

Stage 4: full DonutSwin encoder parity for base and CORD sizes, returning `[B,S_enc,1024]`.

Stage 5: compose with MBART decoder using unfused generic attention, no beam search, greedy decode.

Stage 6: add encoder-output cache and MBART self/cross KV cache for decode.

Stage 7: add window attention, patch embedding, patch merging, and last-token LM-head fusions.

Stage 8: implement task prompt helpers and `token2json` postprocessing for end-to-end extraction/classification parity.

Initial stubs: beam search, training loss, classification head, absolute embedding interpolation, and `donut-proto` legacy Swin routing.

## 12. Parity and validation plan

- Image processor parity: landscape/portrait images with and without `do_align_long_axis`; verify output HxW, centered padding, bilinear resize, normalization.
- Patch Conv2d parity: random NCHW tensors for exact and non-divisible H/W.
- Window partition/reverse parity: random `[B,H,W,C]`, window 10, shifted and unshifted.
- Shifted-window mask parity: compare generated masks for stage grids 640x480, 320x240, 160x120, 80x60 and CORD 320x240, 160x120, 80x60, 40x30.
- Single DonutSwin block parity at stage channels 128 and 1024.
- Full encoder parity: compare `last_hidden_state` for `donut-base` and CORD sizes.
- Decoder prefill logits parity: given fixed prompt IDs and encoder hidden states.
- Decode token parity: greedy one-token and multi-token with self/cross caches enabled.
- End-to-end parity: generated text and `token2json` for CORD, DocVQA, and RVL-CDIP prompts.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 encoder block `rtol=5e-2, atol=5e-2`; generated token parity should be exact under deterministic greedy decode.

## 13. Performance probes

- CPU preprocessing images/sec for 1280x960 and 2560x1920, split by rotate/resize/thumbnail/pad/normalize.
- Patch embedding throughput and memory bandwidth.
- DonutSwin stage-by-stage latency and temporary memory; separate shifted and unshifted blocks.
- Window attention backend comparison: eager vs fused, batch-size sweep.
- Encoder-only throughput for `S_final=1200` and `S_final=4800`.
- Decoder prefill latency by prompt length and encoder length.
- Decode tokens/sec with and without cross-attention K/V cache.
- KV memory: self cache `[layers,B,16,T,64]` plus cross cache `[layers,B,16,S_enc,64]`.
- End-to-end requests/hour by task checkpoint and image size.
- NHWC fusion probe: measure partition/roll/permute elimination inside guarded Swin blocks.

## 14. Skip/defer list

- Training, labels/loss, stochastic depth/dropout behavior.
- Beam search and sampling; greedy decode is enough for first parity.
- `DonutSwinForImageClassification` unless targeting ImageNet-like classification.
- `bool_masked_pos` masked-image-modeling path.
- Absolute position embeddings/interpolation unless a checkpoint actually sets `use_absolute_embeddings=True`.
- `always_partition` tracing/export path.
- `donut-proto` native support in this report because its config uses `swin`, not `donut-swin`.
- Multi-GPU tensor parallel and quantization.

## 15. Final implementation checklist

- [ ] Parse `VisionEncoderDecoderConfig` with `encoder.model_type="donut-swin"` and delegated MBART decoder config.
- [ ] Load DonutSwin, MBART decoder, tied LM-head/embedding weights, and special-token metadata.
- [ ] Implement Donut image preprocessing: long-axis align, shortest-edge resize, thumbnail, centered pad, normalize.
- [ ] Implement DonutSwin patch Conv2d with right/bottom patch padding.
- [ ] Implement window partition/reverse, cyclic shift, shifted-window mask, and relative position bias.
- [ ] Implement DonutSwin stage blocks and patch merging with exact 2x2 gather order.
- [ ] Implement encoder-output cache for `[B,S_enc,1024]`.
- [ ] Compose MBART causal decoder self-attention and cross-attention with `EncoderDecoderCache`.
- [ ] Add greedy generation path with task prompt input IDs.
- [ ] Implement `token2json`-compatible postprocessing.
- [ ] Add guarded Conv2d-to-GEMM patch embedding rewrite.
- [ ] Add guarded shifted-window attention fusion candidate with NHWC internal layout only.
- [ ] Add patch merging gather/LayerNorm/Linear fusion candidate.
- [ ] Add processor, encoder, prefill, decode, and end-to-end parity tests.
- [ ] Benchmark preprocessing, encoder stages, cross-attention cache, and decode tokens/sec.
