# SmolVLM Transformers Audit

## 1. Source basis

Transformers commit/version:
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id:
Primary native-source target is `HuggingFaceTB/SmolVLM2-2.2B-Instruct`; checkpoint sweep also inspected `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, and historical `HuggingFaceTB/SmolVLM-{256M,500M,Instruct}` configs.

Config source:
Raw Hugging Face `config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer_config.json`, `generation_config.json`, and model API metadata snapshots saved under `agents/plans/transformers/smolvlm/_sources/`.

Source files inspected:
- `src/transformers/models/smolvlm/configuration_smolvlm.py`
- `src/transformers/models/smolvlm/modeling_smolvlm.py`
- `src/transformers/models/smolvlm/modular_smolvlm.py`
- `src/transformers/models/smolvlm/processing_smolvlm.py`
- `src/transformers/models/smolvlm/image_processing_smolvlm.py`
- `src/transformers/models/smolvlm/image_processing_pil_smolvlm.py`
- `src/transformers/models/smolvlm/video_processing_smolvlm.py`
- delegated decoder basis: `src/transformers/models/llama/modeling_llama.py`, `configuration_llama.py`
- comparison points: `src/transformers/models/idefics3/modeling_idefics3.py`, `src/transformers/models/llava/modeling_llava.py`

Any missing files or assumptions:
`modeling_smolvlm.py` is generated from `modular_smolvlm.py`; future source edits should target the modular file. Older SmolVLM 256M/500M/2.2B configs use `model_type="idefics3"` and `Idefics3ForConditionalGeneration`; this report treats them as compatibility checkpoints, not native `smolvlm` source behavior. Native `smolvlm` uses `AutoModel.from_config(text_config)`, so decoder internals are delegated to the Llama family and should compose that audit for full decoder coverage.

## 2. High-level architecture

SmolVLM is a compact multimodal generation stack:

```text
CPU image/video preprocessing -> NCHW pixel_values + pixel_attention_mask
  -> SigLIP-like vision encoder
  -> pixel-shuffle connector + biasless projection
  -> indexed replacement of <image> token embeddings
  -> delegated Llama causal decoder prefill/decode
  -> lm_head logits/sampling
```

Stage decomposition:
- CPU/data pipeline: fetch/convert RGB, resize, optional image splitting, pad, normalize, produce row/column split metadata, expand `<image>` or `<video>` placeholders into repeated `<image>` tokens.
- Vision runtime: noncausal ViT over NCHW image tensors with patch mask-derived bidirectional attention mask.
- Connector runtime: square-grid pixel shuffle reduces patch sequence length by `scale_factor^2` and expands channel width by the same factor, then projects to decoder hidden size.
- Prefix construction: `inputs_merger` replaces placeholder-token embeddings with image features using per-sample block offsets.
- Decoder prefill/decode: normal delegated Llama causal self-attention with DynamicCache; image encoder/projector output can be provided as `image_hidden_states` and reused independently from KV cache.

First useful DinoML target: image-text generation prefill + decode for one or more already-processed images. Video is a processor-level frame-prompt expansion over the same image encoder path and can be staged later.

## 3. Important config dimensions

Native source defaults from `SmolVLMConfig`/`SmolVLMVisionConfig`: `model_type="smolvlm"`, `text_config` defaults to Llama, `vision_config` defaults to hidden 1152, 12 layers, 16 heads, image size 224, patch size 32, `scale_factor=2`, `image_token_id=128257`, `use_cache=True`. Llama defaults fill omitted fields such as `num_attention_heads=32`, `num_key_value_heads=num_attention_heads`, `hidden_act="silu"`, `attention_bias=False`, `mlp_bias=False`, and `head_dim=hidden_size // num_attention_heads` unless explicitly present.

Representative checkpoint sweep:

| Checkpoint | Source scope | Text hidden/layers | Heads/KV/head_dim | MLP | Vision hidden/layers/heads | Image/patch/scale | Image seq len | Processor | Cache flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SmolVLM2-256M-Video-Instruct | native `smolvlm` | 576 / 30 | 9 / 3 / 64 | 1536 | 768 / default 12 / 12 | 512 / 16 / 4 | 64 | split images, video 64 frames @ 1 fps | config omits/uses source default unless inherited |
| SmolVLM2-500M-Video-Instruct | native `smolvlm` | 960 / 32 | 15 / 5 / 64 | 2560 | 768 / default 12 / 12 | 512 / 16 / 4 | 64 | split images, video 64 frames @ 1 fps | config omits/uses source default unless inherited |
| SmolVLM2-2.2B-Instruct | native `smolvlm` | 2048 / 24 | omitted -> Llama default 32 / 32 / 64 | 8192 | default 1152 / 27 / default 16 | 384 / 14 / 3 | 81 | split images, video 64 frames @ 1 fps | `use_cache=false` in config |
| SmolVLM-256M-Instruct | historical `idefics3` | 576 / 30 | 9 / 3 / 64 | 1536 | 768 / 12 / 12 | 512 / 16 / 4 | 64 | split images, no video sampling config | `bfloat16` metadata |
| SmolVLM-500M-Instruct | historical `idefics3` | 960 / 32 | 15 / 5 / 64 | 2560 | 768 / 12 / 12 | 512 / 16 / 4 | 64 | split images, no video sampling config | `bfloat16` metadata |
| SmolVLM-Instruct | historical `idefics3` | 2048 / 24 | 32 / 32 / 64 | 8192 | 1152 / 27 / 16 | 384 / 14 / 3 | 81 | split images, no video sampling config | `bfloat16` metadata |

Config-derived image token IDs vary: SmolVLM2-2.2B uses `49190`; source default is `128257`; older configs use their own tokenizer vocab/id space. Do not hardcode placeholder IDs.

## 3a. Family variation traps

- Native `smolvlm` and historical `idefics3` checkpoints share very similar vision/connector math, but their class names and some merger behavior differ. Route `model_type="idefics3"` checkpoints through an Idefics3 compatibility path unless explicitly testing native SmolVLM2 behavior.
- `hidden_size` is not enough to infer projections. Use `num_attention_heads`, `num_key_value_heads`, and `head_dim`; 256M/500M use GQA (`KV < Q`), while 2.2B native JSON omits heads/KV and therefore falls back to Llama defaults.
- `image_seq_len = ((image_size // patch_size) ** 2) // scale_factor**2`. For 384/14/3 this is `81`; for 512/16/4 this is `64`.
- The connector assumes a square patch sequence before pixel shuffle: `height = width = int(seq**0.5)`. Any dynamic admission must enforce square sequence length and divisibility by `scale_factor`.
- Image splitting changes both number of image blocks and prompt token expansion. A split image emits row/column-local image token blocks plus one global-image block.
- Source tensor layout for vision is NCHW. Layout rewrites must guard axis-sensitive code: Conv2d input `NCHW`, `flatten(2).transpose(1,2)`, pixel mask `unfold(dimension=1/2)`, image splitting `unfold(height_dim=2,width_dim=3)`, padding mask shape `[B, Nimg, H, W]`.
- The model stitches image embeddings by indexed replacement, not cross-attention. This differs from cross-attention VLMs and from simple LLaVA-style global masked scatter because SmolVLM groups image feature blocks per sample.
- The decoder cache is delegated to `text_model` Llama `DynamicCache`; image features are independent prefix data, not a vision KV cache.
- `logits_to_keep` can restrict lm_head projection to the last tokens and is important for decode performance.
- Processor requires `num2words` for text/video prompt construction; this is CPU pipeline behavior, not a GPU runtime op.

## 4. Operator coverage checklist

Tensor/layout ops:
- NCHW Conv2d patch embedding with `kernel_size=stride=patch_size`, `padding=valid`.
- Flatten patches, transpose `[B,C,Hp,Wp] -> [B, Hp*Wp, C]`.
- `view`, `reshape`, `permute`, `contiguous`, boolean flattening, `unfold` over pixel masks, `cat`, `where`, `zeros_like`.
- Boolean masks, cumsum, pad of cumulative counts, integer floor-div/mod, boolean indexing/scatter.

Neural network primitives:
- Vision LayerNorm, noncausal attention, GELU/tanh-GELU activation, MLP Linear -> activation -> Linear.
- Connector pixel shuffle on token grids and biasless Linear `(vision_hidden * scale_factor^2) -> text_hidden`.
- Decoder Llama RMSNorm, biasless Q/K/V/O projections, SwiGLU MLP, residual adds, final norm, lm_head Linear.

Attention primitives:
- Vision bidirectional self-attention over patch tokens, MHA, fp32 softmax in eager path.
- Decoder causal self-attention, MHA/GQA depending on text config, RoPE before cache update, DynamicCache.
- Backend compatibility through Transformers `ALL_ATTENTION_FUNCTIONS` for eager/SDPA/Flash/Flex attention.

Position/rotary/relative-bias ops:
- Vision variable-resolution positional bucketing with `bucketize` over fractional patch coordinates and learned position embedding lookup.
- Decoder Llama RoPE with `rope_theta`/`rope_parameters`, cos/sin computed in fp32 and cast back.

Generation/cache ops:
- DynamicCache creation if `use_cache` and no past cache is supplied.
- Per-layer KV cache stores K/V after RoPE and before repeat-KV expansion; shapes are `[batch, num_key_value_heads, cache_seq, head_dim]`.
- `prepare_inputs_for_generation` drops `pixel_values`/`pixel_attention_mask` after first cached iteration or when `image_hidden_states` is supplied.

Preprocessing-coupled ops:
- Image resize/rescale/normalize/pad, optional split into max-size square crops plus global image.
- Video sampling, timestamps, square-frame resize, per-frame prompt expansion; video frames reuse the image path.
- Placeholder expansion: `<image>` becomes fake-token wrappers, optional `<row_i_col_j>` markers, one or more runs of repeated `<image>` tokens.

Scatter/indexed update ops for multimodal embedding stitch:
- `image_mask = input_ids == image_token_id`.
- Verify `num_image_tokens % patch_size == 0`, where `patch_size` is actually the connector output sequence length per image block.
- Compute per-sample `blocks_per_sample`, `block_offset`, `chunk_idx`, `local_idx`, and assign `image_hidden_states[block_idx, local_idx, :]` into zero-like embedding buffer.
- `torch.where(image_mask[..., None], image_embeds, inputs_embeds)`.

## 5. Layer/block breakdown

Vision preprocessing to patch tokens:

```text
pixel_values: [B, Nimg, 3, H, W]
flatten real images -> [Breal, 3, H, W]
pixel_attention_mask -> unfold H/W by patch_size -> patch_attention_mask [Breal, Hp, Wp]
Conv2d(3 -> vision_hidden, kernel=stride=patch) -> [Breal, vision_hidden, Hp, Wp]
flatten + transpose -> [Breal, Hp*Wp, vision_hidden]
position bucket ids from valid patch rows/cols -> embedding add
```

Vision encoder layer, repeated `vision_config.num_hidden_layers`:

```text
x = x + SelfAttention(LayerNorm(x), bidirectional patch mask)
x = x + Linear2(activation(Linear1(LayerNorm(x))))
post_layernorm(x)
```

Connector:

```text
x: [Breal, seq, vision_hidden], seq must be square
x = pixel_shuffle_token_grid(x, scale_factor)
x: [Breal, seq / scale_factor^2, vision_hidden * scale_factor^2]
x = Linear(x, bias=False) -> [Breal, image_seq_len, text_hidden]
```

Decoder block, delegated Llama repeated `text_config.num_hidden_layers`:

```text
x = x + OProj(Attention(RoPE(Q(RMSNorm(x))), RoPE(K(...)), V(...), cache, causal_mask))
x = x + Down(SiLU(Gate(RMSNorm(x))) * Up(RMSNorm(x)))
```

LM head:

```text
hidden[:, slice(-logits_to_keep, None), :] -> Linear(text_hidden -> vocab_size, bias=False)
```

Weight aliasing:
`SmolVLMForConditionalGeneration._tied_weights_keys` declares `lm_head.weight` tied to `model.text_model.embed_tokens.weight`, while config `tie_word_embeddings` is usually `false`. DinoML should preserve alias intent if loading tied checkpoints, but not assume all configs tie by default.

## 6. Attention requirements

Vision attention:
- Noncausal self-attention.
- MHA with `vision_hidden / vision_heads` head dim; source validates divisibility.
- Mask comes from patch-validity mask via `create_bidirectional_mask`.
- No KV cache; output is an independently cacheable image-prefix tensor.

Decoder attention:
- Causal self-attention from delegated Llama.
- MHA or GQA: Q heads = `num_attention_heads`, KV heads = `num_key_value_heads`, groups = Q/KV.
- Projections are separate Linear layers with source weight layouts `[out_features, in_features]`; no packed QKV source format.
- RoPE is applied to Q/K before `past_key_values.update`.
- Cache tensors are stored before repeat expansion as `[B, num_key_value_heads, T, head_dim]`; attention backends may repeat/broadcast to Q head count internally.
- Eager fallback computes `(Q @ K^T) * head_dim^-0.5`, adds mask, softmax with `dtype=torch.float32`, casts to query dtype, dropout, then `P @ V`.
- Flash/SDPA/Flex are advertised by source flags and delegated through Transformers attention registry; first DinoML parity can use unfused eager math, but production should target GQA FlashAttention with cache.

Packed/varlen support:
No explicit `cu_seqlens` or packed sequence descriptors are produced by SmolVLM. Padding/attention masks follow normal tokenizer and image patch mask conventions.

## 7. Position encoding and custom math

Vision position bucketing is the nonstandard piece. It adapts a fixed learned square position table to variable image aspect ratios:

```python
def smolvlm_patch_position_ids(patch_attention_mask, num_patches_per_side):
    boundaries = arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
    nb_h = patch_attention_mask[:, :, 0].sum(1)
    nb_w = patch_attention_mask[:, 0, :].sum(1)
    h = arange(max_h)[None, :] * (1.0 / nb_h)[:, None]
    w = arange(max_w)[None, :] * (1.0 / nb_w)[:, None]
    bh = bucketize(clamp(h, max=1.0 - 1e-6), boundaries, right=True)
    bw = bucketize(clamp(w, max=1.0 - 1e-6), boundaries, right=True)
    return (bh[:, :, None] * num_patches_per_side + bw[:, None, :]).reshape(batch, -1)
```

Decoder RoPE is standard Llama:

```python
cos = cos[position_ids].unsqueeze(1)
sin = sin[position_ids].unsqueeze(1)
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin
```

Precomputable:
Vision learned position embedding weights and bucket boundaries for fixed `image_size/patch_size`; decoder inverse frequencies for fixed RoPE settings. Dynamic:
valid patch counts, bucketized position IDs, text position IDs, and cache sequence length.

## 8. Preprocessing and input packing

Image processor contract:
- Input images are converted RGB, resized with longest edge `size.longest_edge`, optionally split, rescaled, normalized with ImageNet mean/std, and padded.
- Default native processor values are `size.longest_edge = 4 * 364`, `max_image_size.longest_edge = 364`, `do_image_splitting=True`; checkpoint configs override to 512/2048 or 384/1536 families.
- Output for padded tensors is `pixel_values [B, max_num_images_or_frames, 3, Hmax, Wmax]` and `pixel_attention_mask [B, max_num_images_or_frames, Hmax, Wmax]`.
- With splitting, each oversized image is resized to multiples of the encoder max size, split into square crops, and appended with a resized global image. `rows` and `cols` are returned for prompt expansion.

Prompt packing:
- A single image placeholder expands to `<fake_token_around_image><global-img><image>*image_seq_len<fake_token_around_image>`.
- Split images expand by row/column crop markers plus image-token runs for every crop, then a global-image run.
- The number and order of `<image>` placeholders must match connector output blocks exactly.

Video processor contract:
- Videos are mutually exclusive with images at processor call level.
- Frames are resized to square `max_image_size`; source notes videos are always processed without image splitting.
- Optional sampling uses metadata, target fps, `max_frames`, and center-skip logic; official SmolVLM2 video configs use 64 frames at 1 fps.
- Prompt expansion describes frame count/duration and inserts timestamped single-image prompt strings per frame.

Runtime packing:
`prepare_inputs_for_generation` avoids re-encoding images during cached decode by setting pixel inputs to `None` when cache is active after the first iteration or when `image_hidden_states` is supplied. DinoML can expose an API that separately runs/caches image features and feeds them to text prefill.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern:
`Conv2d(3, vision_hidden, kernel_size=patch, stride=patch, padding=valid)` followed by `flatten(2).transpose(1,2)`.

Replacement:

```text
NCHW image -> non-overlap patch flatten [B, Hp*Wp, 3*patch*patch]
  -> GEMM(weight_flat.T) + bias -> [B, Hp*Wp, vision_hidden]
```

Preconditions:
`kernel_size == stride == patch_size`, `padding == 0/valid`, `dilation == 1`, `groups == 1`, input H/W divisible by patch size or already padded/rounded by processor, NCHW layout or explicit layout transform.

Weight transform:
`conv.weight.reshape(out_channels, in_channels * patch * patch)`.

Failure cases:
Non-NCHW tensors, non-divisible spatial dimensions, altered conv groups/dilation/padding, or a future overlapping patch config.

Parity test sketch:
Random NCHW image tensors at 384/512 and padded rectangular sizes; compare Conv2d output after flatten/transpose against rewritten GEMM.

### Rewrite: connector pixel shuffle + projection -> grouped reshape/GEMM

Source pattern:
Token grid reshape/permutation merges each `scale_factor x scale_factor` patch neighborhood into channels, then applies one biasless projection.

Replacement:
Static reshape/permute/copy or layout-aware gather into `[B, seq/s^2, vision_hidden*s^2]` followed by GEMM.

Preconditions:
`seq` is a perfect square, both grid dimensions divisible by `scale_factor`, source patch order is row-major after vision flatten/transpose, projection weight shape matches `vision_hidden*s^2 -> text_hidden`.

Failure cases:
Dynamic non-square sequences, local crop sizes that do not match the configured square token grid, or a layout pass that changes patch flatten order.

Parity test sketch:
Known ascending-token tensor through source `pixel_shuffle`; compare output ordering and projected result.

### Rewrite: multimodal placeholder stitch -> indexed copy

Source pattern:
Boolean mask, cumsum, per-sample block offsets, image feature lookup, `torch.where`.

Replacement:
Dedicated indexed-copy op from `[Bblocks, image_seq_len, hidden]` into `[B, T, hidden]` at `image_token_id` positions.

Preconditions:
Per-sample image token count divisible by `image_seq_len`; total blocks equals image feature block count; no generated `<image>` tokens should be interpreted as image placeholders during cached decode.

Failure cases:
Mismatched processor/model `image_seq_len`, missing row/column expansion, tokenizer changes image token ID, decode step accidentally includes image token ID as normal generated text.

Parity test sketch:
Batch with unequal image counts and split-image rows/cols; compare source `inputs_merger` to DinoML indexed copy.

### Rewrite: last-token-only logits

Source pattern:
`logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])`.

Replacement:
Project only selected decode positions.

Preconditions:
No loss computation requiring full logits; `logits_to_keep` is positive int or explicit index tensor.

Failure cases:
Training/loss path or caller requests full sequence logits.

## 10. Kernel fusion candidates

Highest priority:
- Llama RMSNorm + QKV projections and RMSNorm + SwiGLU MLP for decoder throughput.
- GQA FlashAttention prefill/decode with KV cache stored as KV heads, not expanded Q heads.
- Connector pixel shuffle + projection; this is compact but easy to make memory-bound if implemented as several materialized layout copies.
- Placeholder indexed copy; correctness-sensitive and avoids expensive boolean advanced indexing.

Medium priority:
- Vision Conv patch embedding lowered to GEMM or optimized conv.
- Vision LayerNorm + MHA + MLP, especially for 27-layer 2.2B vision encoder.
- Vision position-bucket ID generation; small but shape-sensitive.
- Last-token lm_head projection for decode.

Lower priority:
- CPU/video prompt construction and sampling optimizations.
- Fusing image rescale/normalize/pad in runtime; first integration can keep this in CPU/data pipeline.
- Historical Idefics3 compatibility path after native SmolVLM2 parity.

## 11. Runtime staging plan

1. Parse native `smolvlm` config and route `text_config.model_type="llama"` to the existing/delegated Llama loader.
2. Load weights and validate naming/aliasing for vision encoder, connector, text model, and lm_head.
3. Implement processor-compatible static image path for already-normalized/padded `pixel_values` plus `pixel_attention_mask`.
4. Run vision encoder + connector parity independently and expose `image_hidden_states` cache.
5. Implement placeholder stitch and one full multimodal prefill logits parity.
6. Add delegated Llama decode with DynamicCache-compatible KV ABI and skip image re-encode after first step.
7. Add optimized attention, connector fusion, and last-token logits.
8. Add video frame prompt path as a processor/prefix feature, reusing image encoder blocks.
9. Decide whether to support old `idefics3` SmolVLM checkpoints here or leave them to the Idefics3 audit.

Stubbable initially:
CPU prompt construction, image resizing/splitting, video sampling, loss computation, attention output tensors, training, and beam search.

## 12. Parity and validation plan

- Random tensor tests for vision position bucketing, including rectangular masks and padded regions.
- Pixel shuffle ordering tests for `scale_factor=3` and `scale_factor=4`.
- Conv patch embedding rewrite parity for 384/14 and 512/16 families.
- Single vision layer parity, then full vision encoder + connector parity.
- Placeholder stitch tests with unequal image counts per batch, split-image blocks, and image-token count mismatch errors.
- Decoder single-layer and full delegated Llama prefill parity should come from the Llama audit; SmolVLM-specific tests should verify image prefix embeddings feed the decoder identically.
- Prefill logits parity for one text-only prompt, one single-image prompt, and one split-image prompt.
- Decode token parity with image features cached and `pixel_values=None` after first iteration.
- Video prefix parity for sampled/preprocessed frame tensors and timestamp prompt expansion.

Recommended tolerances:
fp32: `rtol=1e-4, atol=1e-5`; fp16/bf16: start with `rtol=5e-2, atol=5e-2` end-to-end and tighten per-op where stable. Source uses fp32 softmax/RMSNorm/RoPE intermediates in key places; preserve those before comparing reduced precision.

## 13. Performance probes

- Processor throughput: resize/split/pad/normalize images per second.
- Video CPU path: frame sampling + timestamp prompt construction + frame preprocessing.
- Vision encoder throughput by resolution and number of split blocks.
- Connector-only bandwidth and GEMM time by `scale_factor`.
- Placeholder stitch time versus batch size, sequence length, and image blocks.
- Prefill-only tokens/sec with image prefix length included.
- Decode-only tokens/sec with cached image features and KV cache.
- KV cache memory by batch, layers, KV heads, head dim, and sequence length.
- Last-token lm_head versus full-sequence lm_head.
- Attention backend comparison: eager, SDPA, Flash/Flex-compatible path.
- Checkpoint sweep: 256M/500M/2.2B image_seq_len 64 vs 81 and GQA vs MHA.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Beam search, speculative decoding, logits processors beyond normal generation.
- Video sampling and prompt text generation in the compiled GPU runtime.
- Processor image resize/split in GPU runtime; keep CPU pipeline first.
- Attention weight outputs and hidden-state recording.
- Historical `idefics3` checkpoint routing unless explicitly selected.
- Quantized/packed weights; no source-coupled quantized format is implemented in native SmolVLM source.
- Multi-GPU tensor parallel plans.
- NHWC/channel-last vision layout translation until patch masks, unfold axes, flatten order, and connector ordering are guarded.

## 15. Final implementation checklist

- [ ] Parse `SmolVLMConfig`, nested `vision_config`, and delegated `text_config`.
- [ ] Reject or route `model_type="idefics3"` SmolVLM checkpoints to Idefics3 compatibility.
- [ ] Load vision Conv2d/LayerNorm/attention/MLP weights.
- [ ] Load connector projection and enforce square/divisible pixel-shuffle contract.
- [ ] Load delegated Llama weights and lm_head with alias checks.
- [ ] Implement NCHW patch embedding and patch attention mask generation.
- [ ] Implement vision variable-resolution position bucketing.
- [ ] Implement vision bidirectional attention path.
- [ ] Implement connector pixel shuffle and projection.
- [ ] Implement placeholder indexed-copy stitch with mismatch checks.
- [ ] Implement image feature cache input path.
- [ ] Compose delegated Llama prefill/decode with KV cache.
- [ ] Implement `logits_to_keep` last-token projection optimization.
- [ ] Add single-block, encoder/projector, prefill, and decode parity tests.
- [ ] Add performance probes for processor, vision, connector, prefill, and decode.
