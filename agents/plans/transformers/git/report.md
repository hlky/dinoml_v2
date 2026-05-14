# GIT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/git-base as base reference; sweep includes base/large, COCO, VQA/TextVQA, VATEX/video variants.
Config source: Hugging Face config.json, preprocessor_config.json, generation_config.json fetched 2026-05-13 from microsoft/* GIT repos.
Source files inspected:
- X:/H/transformers/src/transformers/models/git/configuration_git.py
- X:/H/transformers/src/transformers/models/git/modeling_git.py
- X:/H/transformers/src/transformers/models/git/processing_git.py
- X:/H/transformers/src/transformers/models/git/convert_git_to_pytorch.py
Source URLs:
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/git/configuration_git.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/git/modeling_git.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/git/processing_git.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/git/convert_git_to_pytorch.py
Any missing files or assumptions: no remote code is required for inspected checkpoints. Processor behavior composes GitProcessor with CLIPImageProcessor for image checkpoints and VideoMAEImageProcessor for video checkpoints.
```

Representative checkpoint config URLs inspected:

- https://huggingface.co/microsoft/git-base/resolve/main/config.json
- https://huggingface.co/microsoft/git-base-coco/resolve/main/config.json
- https://huggingface.co/microsoft/git-large/resolve/main/config.json
- https://huggingface.co/microsoft/git-large-coco/resolve/main/config.json
- https://huggingface.co/microsoft/git-base-textvqa/resolve/main/config.json
- https://huggingface.co/microsoft/git-large-textvqa/resolve/main/config.json
- https://huggingface.co/microsoft/git-base-vatex/resolve/main/config.json
- https://huggingface.co/microsoft/git-large-vatex/resolve/main/config.json
- https://huggingface.co/microsoft/git-base-msrvtt-qa/resolve/main/config.json

Representative processor configs inspected from the same repos at
`preprocessor_config.json`; image checkpoints use `CLIPImageProcessor`, while
VATEX/MSRVTT video checkpoints use `VideoMAEImageProcessor`. No
`processor_config.json` was present in the inspected repos.

Primary runtime target for this report: `GitForCausalLM` multimodal generation. `GitModel` is required as the body. `GitVisionModel` is required as an independently stageable image/video-frame encoder. Training loss is optional. Vision-only use is optional but useful for encoder parity.

## 2. High-level architecture

GIT is a vision encoder + causal text decoder model. The vision side is CLIP-like ViT over image patches. Its sequence output is projected into the text hidden size, prepended to text token embeddings, and consumed by a BERT-shaped decoder stack running with a causal/block mask.

```text
CPU image/video preprocessing + BERT tokenization
  -> NCHW pixel_values
  -> GitVisionModel patch encoder
  -> Linear+LayerNorm visual projection
  -> visual prefix concat with text embeddings
  -> causal decoder prefill
  -> KV-cache decode
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: resize, center crop, rescale, normalize, convert RGB where configured, tokenize text with BERT tokenizer conventions.
- Vision encoder: NCHW image tensor to patch sequence. For video, source loops over frames and calls the same image encoder per frame.
- Projector/prefix: `Linear(vision_hidden -> 768)` then `LayerNorm(768)`, then concat before text embeddings. These visual prefix embeddings can be cached independently from tokenization and, for first integration, can be validated as a standalone prefix tensor.
- Prefill: decoder sees `[visual_prefix, text_tokens]`, image prefix tokens attend bidirectionally within the image block, text attends causally to prefix and previous text.
- Decode: `prepare_inputs_for_generation` passes `pixel_values` only on the first generation iteration when cache is enabled. Later steps rely on cached visual/text keys and values.

## 3. Important config dimensions

Source defaults:

| Field | GitConfig default | GitVisionConfig default | Notes |
|---|---:|---:|---|
| `hidden_size` | 768 | 768 | Text hidden is always 768 in inspected checkpoints. |
| `num_hidden_layers` | 6 | 12 | Decoder is shallow; large variants enlarge only vision encoder. |
| `num_attention_heads` | 12 | 12 | Text head dim 64. |
| `intermediate_size` | 3072 | 3072 | GELU text FFN, quick_gelu vision FFN. |
| `vocab_size` | 30522 | n/a | BERT uncased vocabulary in official conversion. |
| `max_position_embeddings` | 1024 | n/a | Absolute text positions. |
| `image_size` | n/a | 224 | Some checkpoints use 420 or 480. |
| `patch_size` | n/a | 16 | Large vision variants use 14. |
| `num_image_with_embedding` | `None` | n/a | Set to 6 for video checkpoints. |
| `use_cache` | `True` | n/a | DynamicCache used when `use_cache` and no cache is passed. |
| dtype | `torch_dtype` often `float32` in config | same | From HF config, not safetensors metadata. |

Checkpoint sweep:

| Checkpoint | Task flavor | Text layers/heads/hidden | Vision layers/heads/hidden | Image size/patch | Prefix tokens | Processor |
|---|---|---:|---:|---:|---:|---|
| `microsoft/git-base` | pretraining/base | 6 / 12 / 768 | 12 / 12 / 768 | 224 / 16 | 197 | CLIPImageProcessor |
| `microsoft/git-base-coco` | image caption | 6 / 12 / 768 | 12 / 12 / 768 | 224 / 16 | 197 | CLIPImageProcessor |
| `microsoft/git-large` | larger vision | 6 / 12 / 768 | 24 / 16 / 1024 | 224 / 14 | 257 | CLIPImageProcessor |
| `microsoft/git-base-textvqa` | high-res VQA | 6 / 12 / 768 | 12 / 12 / 768 | 480 / 16 | 901 | CLIPImageProcessor |
| `microsoft/git-large-textvqa` | high-res large VQA | 6 / 12 / 768 | 24 / 16 / 1024 | 420 / 14 | 901 | CLIPImageProcessor |
| `microsoft/git-base-vatex` | video caption | 6 / 12 / 768 | 12 / 12 / 768 | 224 / 16 | 1182 | VideoMAEImageProcessor |
| `microsoft/git-large-vatex` | large video caption | 6 / 12 / 768 | 24 / 16 / 1024 | 224 / 14 | 1542 | VideoMAEImageProcessor |

Prefix tokens are `(image_size // patch_size)^2 + 1`, multiplied by `num_image_with_embedding` for video. `git-large-r*` shares the large vision dimensions in the in-library config; the audited source does not add a separate retrieval architecture.

## 3a. Family variation traps

- The text decoder is architecturally BERT-like but semantically causal. Do not route it as a bidirectional BERT encoder.
- Large checkpoints increase vision hidden size/layers/heads but keep text hidden size at 768, so the visual projection is required.
- VQA/TextVQA checkpoints use much larger image grids: 480/16 or 420/14 gives 900 patch tokens plus class token.
- Video is implemented in-library with `num_image_with_embedding=6`, per-frame image encoding, learned frame temporal embeddings in vision hidden size, and sequence concat. There is no 3D/video attention kernel in this source.
- Text attention only registers `"eager"` in `GIT_SELF_ATTENTION_CLASSES`; non-eager `_attn_implementation` would not be a valid text decoder path in this source. Vision attention may dispatch through `ALL_ATTENTION_FUNCTIONS`.
- No RoPE, ALiBi, GQA, MQA, sliding window, or cross-attention.
- `tie_word_embeddings` is false in official configs, even though `_tied_weights_keys` names a potential output/embedding alias. Treat output projection and token embedding as separate unless config/loaded weights prove tying.
- Image/video prefix is not represented by placeholder token IDs. The model prepends projected visual embeddings internally.
- Source pixel layout is NCHW for images and `[B, F, C, H, W]` for batched video. NHWC is an optimization candidate only inside controlled patch-embedding regions.
- Axis-sensitive layout traps: Conv2d assumes channel axis 1; patch flatten uses `flatten(2).transpose(1, 2)`; attention uses `[B, heads, seq, head_dim]`; visual/text concat is on sequence dim 1.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW Conv2d patch embedding, flatten from `[B, C, Gh, Gw]` to `[B, C, Gh*Gw]`, transpose to `[B, Gh*Gw, C]`.
- `cat` on sequence dimension for class token, visual/text prefix, and per-frame video features.
- `repeat` visual prefix across embedding batch when text batch is a multiple of visual batch.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `slice`, integer position slicing, rank checks.
- `torch.where` and mask construction for block sequence IDs.

Neural primitives:

- Embedding lookup for word embeddings and absolute text positions.
- Linear with bias for text Q/K/V/out, text MLP, LM head.
- Linear without bias? Patch embedding Conv2d has `bias=False`; other inspected Linear modules use default bias.
- LayerNorm eps `1e-12` for text, `1e-5` for vision and visual projection.
- GELU text FFN; quick_gelu vision FFN.
- Dropout is source-present but disabled in inference.

Attention primitives:

- Text causal MHA: Q/K/V `Linear(768 -> 768)`, 12 heads, head dim 64, eager matmul-softmax-matmul, KV cache.
- Vision noncausal MHA: base `768/12/64`, large `1024/16/64`, no cache, optional backend dispatch.
- Additive attention masks with large negative masked values from `create_causal_mask`.

Generation/cache ops:

- DynamicCache allocation/update per decoder layer.
- Cache-aware `prepare_inputs_for_generation`.
- Last-token logits via `logits_to_keep` slicing.
- Attention mask expansion during decode when pixel values are absent and cached prefix exists.

Preprocessing-coupled ops:

- CLIPImageProcessor or VideoMAEImageProcessor image normalization/resizing contract.
- BERT tokenizer IDs: config BOS/CLS 101, EOS/SEP 102, PAD 0.

Scatter/indexed update ops:

- No multimodal scatter into token embeddings. Visual embeddings are prepended by concat.

Packed/varlen metadata:

- None. No `cu_seqlens`, no packed attention descriptors.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values [B, 3, H, W]
patch = Conv2d(3 -> Vh, kernel=patch, stride=patch, bias=False) -> [B, Vh, Gh, Gw]
patch = flatten spatial -> transpose -> [B, Gh*Gw, Vh]
class = learned [Vh] expanded to [B, 1, Vh]
x = cat(class, patch, dim=1)
x = x + absolute vision position embedding, optionally bicubic-interpolated
```

Vision encoder block, repeated 12 base or 24 large times:

```text
res = x
x = LayerNorm(x, eps=1e-5)
q,k,v = Linear(Vh -> Vh, bias=True)
x = noncausal MHA(q,k,v)
x = res + Linear(Vh -> Vh, bias=True)
res = x
x = LayerNorm(x, eps=1e-5)
x = Linear(Vh -> 4*Vh) -> quick_gelu -> Linear(4*Vh -> Vh)
x = res + x
```

Vision wrapper:

```text
x = embeddings(pixel_values)
x = pre_layrnorm(x)
x = vision_encoder(x)
x = post_layernorm(x)
```

Projection/prefix:

```text
visual_features [Bv, P, Vh]
if video: per frame add learned [1,1,Vh] temporal embedding before frame concat
projected = Linear(Vh -> 768) -> LayerNorm(768, eps=vision_eps)
projected = repeat to text batch if needed
decoder_input = cat(projected, text_embeddings, dim=1)
```

Text embeddings:

```text
token = Embedding(vocab_size=30522, hidden=768, padding_idx=0)
pos = Embedding(max_position_embeddings=1024, hidden=768)
x = token + pos
x = LayerNorm(eps=1e-12)
```

Decoder block, repeated 6 times:

```text
q,k,v = Linear(768 -> 768, bias=True)
k,v = cache.update(k,v, layer_idx) if cache is active
scores = matmul(q, k.T) / sqrt(64)
scores = scores + causal/block mask
prob = softmax(scores, dim=-1)
ctx = matmul(prob, v)
x = LayerNorm(Linear(ctx) + residual, eps=1e-12)
mlp = Linear(768 -> 3072) -> gelu -> Linear(3072 -> 768)
x = LayerNorm(mlp + residual, eps=1e-12)
```

LM head:

```text
logits = Linear(768 -> 30522, bias=True)(hidden_states[:, slice_indices, :])
```

## 6. Attention requirements

Text decoder attention:

- Causal self-attention with multimodal block-prefix exception.
- MHA only: 12 query heads, 12 KV heads, head dim 64. No GQA/MQA repeat expansion.
- Q/K/V tensors after projection are `[B, heads, T, 64]`.
- Cache tensors per decoder layer store keys and values as `[B, 12, cache_seq, 64]`.
- Cached keys are after absolute position embeddings have already been added to hidden states; there is no separate RoPE state.
- Attention math order: QK matmul, divide by `sqrt(head_dim)`, add mask, softmax in default dtype, dropout, PV matmul.
- Text decoder does not use SDPA/FlashAttention in this source because only `"eager"` is registered.

Vision attention:

- Noncausal self-attention over patch/class tokens.
- MHA base `[B, 12, P, 64]`; large `[B, 16, P, 64]`.
- No KV cache.
- Eager fallback multiplies by scale before mask, softmaxes with `dtype=torch.float32`, casts back to query dtype, then dropout and PV matmul.
- `ALL_ATTENTION_FUNCTIONS` may select SDPA-like implementations for vision, but parity should start from eager semantics.

Masking:

- `create_causal_mask` receives `block_sequence_ids=group_ids`.
- Image prefix positions get group id 0, text positions get -1. This makes image tokens bidirectional within the image block while preserving causal suffix behavior.
- During cached single-token decode without `pixel_values`, the source expands `attention_mask` with ones for cached image tokens because GIT does not use placeholder image tokens in `input_ids`.

## 7. Position encoding and custom math

Text uses learned absolute position embeddings. In normal prefill, position IDs are sliced from `[0..max_position_embeddings)`. In cached single-token decode without pixel input, source adjusts externally supplied `position_ids` by cache length.

Vision uses learned absolute position embeddings over class + square patch grid. Optional interpolation is required only when `interpolate_pos_encoding=True` or tracing dynamic shapes.

```python
def git_interpolate_vision_pos(position_embedding, embeddings, height, width, patch_size):
    num_patches = embeddings.shape[1] - 1
    num_positions = position_embedding.shape[0] - 1
    if num_patches == num_positions and height == width:
        return position_embedding[None, :, :]
    cls = position_embedding[:1]
    patch = position_embedding[1:]
    dim = embeddings.shape[-1]
    side = int(num_positions ** 0.5)
    patch = patch.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_size, width // patch_size), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return concat([cls[None, :, :], patch], dim=1)
```

Precomputable:

- Text position embedding table.
- Vision position embeddings for fixed image sizes.
- Causal/block masks for static prefix/text lengths and batch shape, modulo decode cache lengths.

Dynamic:

- Interpolated vision positions for non-configured image sizes.
- Decode attention mask expansion from current cache and caller attention mask.

## 8. Preprocessing and input packing

Image checkpoints:

- `GitProcessor` wraps `CLIPImageProcessor` and tokenizer.
- `preprocessor_config.json`: resize shortest edge to configured size, center crop to square, rescale by `1/255`, normalize by CLIP mean/std `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`, convert RGB where present.
- Runtime tensor contract: `pixel_values` `[B, 3, H, W]`, float, NCHW.

Video checkpoints:

- `preprocessor_config.json` uses `VideoMAEImageProcessor`, resize/crop 224, rescale by `1/255`, normalize mean/std `[0.5, 0.5, 0.5]`.
- Model code accepts `pixel_values` rank 5 as `[B, F, 3, H, W]`, but HF examples often create frames as a list. Integration should normalize to batched rank 5 before graph entry.
- `F` should equal `config.num_image_with_embedding` for learned temporal embeddings; official video configs use 6.

Text:

- Tokenizer is BERT-base-uncased in conversion script; model IDs provide empty tokenizer_config but configs use BOS/CLS 101, EOS/SEP 102, PAD 0.
- Prompt text embeddings are appended after visual prefix. There are no image placeholder token IDs or scatter positions.

Generation controller:

- First generation step includes `pixel_values`; later cached steps do not.
- `logits_to_keep` can restrict logits to last tokens and should be used for decode efficiency.
- Beam search and sampling live in generic Transformers generation, not the core GIT graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear

Source pattern:

```text
Conv2d(C -> Vh, kernel=P, stride=P, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1, 2)
```

Replacement:

```text
NCHW PatchExtract(PxP) -> WindowFlatten(C*P*P) -> MatMul(weight_flat.T) -> Reshape [B, Gh*Gw, Vh]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- `H % patch_size == 0`, `W % patch_size == 0`.
- Flatten order must match PyTorch Conv2d NCHW kernel storage `[out, in, kh, kw]`.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Layout constraints:

- Faithful graph is NCHW. NHWC patch extraction is allowed only if the image processor/runtime region and weight flatten permutation are controlled.
- Required NHWC weight transform would reorder from `[out, C, kh, kw]` to match window layout `[kh, kw, C]` or the chosen lowering order.

Failure cases:

- Interpolated dynamic image sizes without divisibility.
- Non-square or list-valued config not tested by official GIT configs.

Parity test sketch:

- Compare patch embeddings before and after rewrite for random NCHW inputs at 224, 420, and 480 sizes in fp32 and fp16.

### Rewrite: visual projection prefix cache

Source pattern:

```text
image_encoder(pixel_values) -> Linear(Vh, 768) -> LayerNorm -> cat(text_embeddings)
```

Replacement:

```text
precompute visual_prefix [B, P, 768] once -> decoder prefill consumes prefix
```

Preconditions:

- `pixel_values`, `interpolate_pos_encoding`, vision weights, projector weights fixed for request.
- Batch repeat rule preserved: visual prefix may be repeated when text batch is an integer multiple of visual batch.
- For video, preserve frame order and temporal embedding addition before projection.

Failure cases:

- Training/dropout enabled.
- Dynamic image augmentation or prompt-dependent image processing.

Parity test sketch:

- Compare full `GitModel` hidden states to a split run that injects cached projected prefix.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = output(hidden_states[:, slice_indices, :])
```

Replacement:

```text
if decode or logits_to_keep == 1: select last hidden -> GEMM(768, vocab)
```

Preconditions:

- No loss computation.
- Generation only needs next-token logits.

Failure cases:

- Caller requests full logits or training labels.

### Layout guard: vision NCHW to NHWC local region

Candidate region:

- Processor output through patch embedding and perhaps vision MLP/attention if a full channel-last ViT path is implemented.

Required axis rewrites:

- Conv input channel `dim=1` becomes last channel.
- Patch flatten/transposes must preserve output `[B, P, Vh]`.
- LayerNorm stays on hidden last dim after patch sequence formation.

No-layout-translation guard:

- Decoder and attention sequence tensors should remain `[B, seq, hidden]`.
- Concats on sequence dim 1 and attention reshapes/transposes must not be rewritten as image layout operations.

## 10. Kernel fusion candidates

Highest priority:

- Text decoder MHA with KV cache: Q/K/V GEMMs, mask add, softmax, PV, output GEMM. This dominates decode and prefill after vision prefix is computed.
- LayerNorm + residual patterns in decoder and vision blocks. GIT has post-norm BERT-style residual blocks.
- Vision patch Conv2d -> GEMM rewrite for predictable non-overlap patches.
- Last-token LM head GEMM for decode.

Medium priority:

- Vision encoder attention using SDPA/FlashAttention-style noncausal MHA.
- MLP fusions: Linear + GELU/quick_gelu + Linear, or activation fused into GEMM epilogue where practical.
- Visual projection Linear + LayerNorm, especially for high-resolution VQA/video prefix sizes.
- Mask construction/cache-aware mask expansion as a precomputed metadata path.

Lower priority:

- Bicubic position interpolation; required only for non-configured image sizes or tracing.
- Video frame loop batching: stack frames as `B*F` through image encoder, then reshape and add temporal embeddings.
- Training loss shift and cross-entropy.

## 11. Runtime staging plan

Stage 1: parse `GitConfig` and load weights for `GitVisionModel`, `GitProjection`, decoder embeddings, decoder blocks, and LM head. Reject non-eager text `_attn_implementation` for first pass.

Stage 2: vision-only parity for fixed image size, NCHW input, no interpolation. Cover base 224/16 and large 224/14.

Stage 3: visual projection and prefix construction parity. Validate image prefix token counts 197, 257, 901 and video counts 1182/1542.

Stage 4: decoder prefill parity with projected prefix plus text tokens. Implement causal/block mask exactly.

Stage 5: decode parity with `DynamicCache`-equivalent per-layer KV cache and no `pixel_values` after first step.

Stage 6: optimized attention and last-token logits. Keep eager attention as reference.

Stage 7: video variant: batch frame encoder calls, temporal embeddings, frame concat, prefix cache.

Can stub initially:

- Training loss.
- Vision position interpolation.
- Beam search/sampling internals beyond accepting generation step inputs.
- Vision SDPA dispatch; eager attention is sufficient for parity.

## 12. Parity and validation plan

- Config parsing tests for base, large, TextVQA, VATEX.
- Random tensor tests for patch Conv2d rewrite at `[1,3,224,224]`, `[1,3,420,420]`, `[1,3,480,480]`.
- Vision embeddings parity before and after position add.
- One vision block parity, then full vision encoder parity.
- Visual projector parity for `Vh=768` and `Vh=1024`.
- Mask parity: compare generated block causal mask for image prefix + text suffix and cached single-token decode.
- One decoder layer parity with and without cache.
- Full prefill logits parity for `microsoft/git-base-coco` and `microsoft/git-large-coco`.
- Decode token parity for two to five generation steps with cache.
- Video prefix parity for `microsoft/git-base-vatex`: encode 6 frames, add temporal embeddings, concat.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for block outputs; tighter for isolated linear/LayerNorm.
- fp16/bf16: `rtol=5e-2`, `atol=5e-2` for full logits unless using identical accumulation order; inspect max token-rank changes.

## 13. Performance probes

- Processor throughput: images/sec for CLIPImageProcessor and VideoMAEImageProcessor separately.
- Vision encoder throughput by image size: 224, 420, 480.
- Prefix construction throughput: projection + LayerNorm + concat.
- Prefill throughput split by prefix length and text length.
- Decode tokens/sec with cached prefix for batch sizes 1, 4, 16.
- KV cache memory: 6 layers * 2 tensors * `[B, 12, seq, 64]` * dtype bytes.
- Vision attention backend comparison: eager vs SDPA where available.
- Last-token-only logits vs full logits for generation.
- Video throughput: frame loop vs batched `B*F` image encoder.

## 14. Skip/defer list

- Training loss and labels.
- Gradient checkpointing.
- Generic non-eager text attention until source supports it.
- Vision position interpolation for arbitrary image sizes.
- Beam search, sampling policy, and generic generation processors beyond one-step logits parity.
- Multi-GPU/tensor parallel.
- Quantization.
- Remote-code variants; none are required for inspected official checkpoints.
- NHWC global translation. Only local, guarded vision patch/layout passes should be attempted first.

## 15. Final implementation checklist

- [ ] Parse `GitConfig` and nested `GitVisionConfig`.
- [ ] Load official GIT weights and preserve untied LM head unless config/weights prove tying.
- [ ] Implement CLIPImageProcessor/VideoMAEImageProcessor tensor contract or require preprocessed NCHW input.
- [ ] Implement vision patch embedding Conv2d and optional guarded Conv2d->Linear rewrite.
- [ ] Implement vision learned class/position embeddings and fixed-size path.
- [ ] Implement vision MHA/MLP/LayerNorm blocks.
- [ ] Implement visual projection `Linear(Vh -> 768) + LayerNorm`.
- [ ] Implement image/video prefix concat, repeat rule, and temporal embeddings.
- [ ] Implement text embeddings with absolute positions and `LayerNorm(eps=1e-12)`.
- [ ] Implement decoder causal/block mask with bidirectional image prefix.
- [ ] Implement text decoder MHA eager reference and KV cache shape `[B, 12, S, 64]`.
- [ ] Implement cached decode path with no `pixel_values` after first step.
- [ ] Implement `logits_to_keep` and last-token LM head optimization.
- [ ] Add parity tests for base, large, high-res VQA, and video configs.
- [ ] Benchmark processor, vision, prefill, decode, and video frame batching separately.
