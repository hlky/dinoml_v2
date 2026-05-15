# ModernVBert family audit for DinoML

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ModernVBERT/modernvbert
Config source: local Transformers config plus Hub config snapshots
Primary runtime target: ModernVBertForMaskedLM visual masked-token inference
DinoML assumptions: inference-only first, CUDA GPU target, faithful PyTorch axes first, optimize layout/fusion only behind guards
```

Source files inspected:

- `transformers/src/transformers/models/modernvbert/modular_modernvbert.py`
- `transformers/src/transformers/models/modernvbert/configuration_modernvbert.py`
- `transformers/src/transformers/models/modernvbert/modeling_modernvbert.py`
- `transformers/src/transformers/models/modernbert/configuration_modernbert.py`
- `transformers/src/transformers/models/modernbert/modeling_modernbert.py`
- `transformers/src/transformers/models/siglip/configuration_siglip.py`
- `transformers/src/transformers/models/siglip/modeling_siglip.py`
- `transformers/src/transformers/models/idefics3/processing_idefics3.py`
- `transformers/src/transformers/models/idefics3/image_processing_idefics3.py`
- `transformers/tests/models/modernvbert/test_modeling_modernvbert.py`
- `transformers/docs/source/en/model_doc/modernvbert.md`

Source URLs at the pinned commit:

- [configuration_modernvbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/modernvbert/configuration_modernvbert.py)
- [modeling_modernvbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/modernvbert/modeling_modernvbert.py)
- [modular_modernvbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/modernvbert/modular_modernvbert.py)
- [modeling_modernbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/modernbert/modeling_modernbert.py)
- [modeling_siglip.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/siglip/modeling_siglip.py)
- [Idefics3 processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics3/processing_idefics3.py)
- [Idefics3 image processor](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/idefics3/image_processing_idefics3.py)

Hub configs and processor snapshots written beside this report:

- `ModernVBERT__modernvbert__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `ModernVBERT__modernvbert-embed__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `ModernVBERT__bimodernvbert__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `ModernVBERT__colmodernvbert-base__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`

Hub metadata checked with `huggingface_hub`: all four ModernVBERT repos above are public and not gated as of 2026-05-13. `paultltc/modernvbert_hf`, used by the local slow integration test, returned 404/RepositoryNotFound to this environment.

Missing files or assumptions:

- `src/transformers/models/modernvbert/__init__.py` lists `processing_modernvbert.py` and image-processing files, but those files do not exist at this commit. AutoProcessor maps `modernvbert` to `Idefics3Processor`.
- `modular_modernvbert.py` is the authoritative source for future Transformers edits; `configuration_modernvbert.py` and `modeling_modernvbert.py` are generated.
- `ModernVBERT/modernvbert-embed` and `ModernVBERT/bimodernvbert` configs declare `architectures: ["BiModernVBert"]`, but no `BiModernVBert` class exists in this checkout. Treat those as out-of-scope for the native ModernVBert report unless a separate remote-code or future-source audit is done.

## 2. High-level architecture

ModernVBert is a vision-language encoder. It does not implement autoregressive generation or a decoder KV cache. The inspected first target is visual masked LM:

```text
CPU image/text preprocessing
  -> Idefics3 image processor and prompt expansion
  -> SigLIP vision encoder over real image tiles
  -> ModernVBert pixel-shuffle connector
  -> indexed replacement of <image> token embeddings
  -> ModernBert bidirectional text encoder
  -> prediction head + tied/aliased LM projection
  -> masked-token logits
```

Stage decomposition:

- CPU/data pipeline: image resize/split/pad/rescale/normalize; prompt construction with `<fake_token_around_image>`, `<global-img>`, row/column tags, and repeated `<image>` tokens; tokenizer and attention mask construction.
- Vision encoder: SigLIP `SiglipVisionModel` consumes flattened real image tiles shaped `[real_images, 3, H, W]` and emits patch tokens `[real_images, (H/16)*(W/16), 768]`.
- Connector: pixel-shuffle-like sequence rearrangement turns each 512x512 image tile's 1024 patch tokens into 64 projected image tokens of width 768.
- Prefix/embedding construction: the model replaces positions whose `input_ids == image_token_id` with image features. This is not a general user-facing scatter; the processor guarantees repeated contiguous image-token spans.
- Text encoder: ModernBert bidirectional encoder with mixed full and sliding-window self-attention layers.
- Heads: masked LM is required for this target. Sequence and token classification heads are implemented but optional/deferred for first integration.

Independently cacheable pieces:

- Preprocessed image tiles are CPU-pipeline artifacts.
- Vision+connector image features can be cached per image/tile batch before text embedding stitch.
- ModernBert hidden states are not decode-cacheable in the autoregressive sense; it is an encoder.

## 3. Important config dimensions

Representative official configs are operator-equivalent for the native `modernvbert` class. The "embed" and "bimodernvbert" repos use `model_type: modernvbert` but point to an unavailable `BiModernVBert` architecture, so their config dimensions are useful but their heads are not covered by this report.

| Field | ModernVBERT/modernvbert | ModernVBERT/colmodernvbert-base | ModernVBERT/modernvbert-embed | ModernVBERT/bimodernvbert |
|---|---:|---:|---:|---:|
| model_type | modernvbert | modernvbert | modernvbert | modernvbert |
| architectures | omitted | omitted | BiModernVBert | BiModernVBert |
| image_token_id | 50407 | 50407 | 50407 | 50407 |
| pixel_shuffle_factor | 4 | 4 | 4 | 4 |
| text model_type | modernbert | modernbert | modernbert | modernbert |
| text vocab_size | 50408 | 50408 | 50408 | 50408 |
| text hidden_size | 768 | 768 | 768 | 768 |
| text layers | 22 | 22 | 22 | 22 |
| text heads / head_dim | 12 / 64 | 12 / 64 | 12 / 64 | 12 / 64 |
| text intermediate_size | 1152 | 1152 | 1152 | 1152 |
| max_position_embeddings | 7999 | 7999 | 7999 | 7999 |
| local_attention | 128 | 128 | 128 | 128 |
| layer pattern | full, slide, slide repeated | same | same | same |
| rope_theta full/sliding | 160000 / 160000 | same | same | same |
| attention_bias / mlp_bias | false / false | false / false | false / false | false / false |
| norm_eps / norm_bias | 1e-5 / false | same | same | same |
| decoder_bias | true | true | true | true |
| vision model_type | siglip_vision_model | siglip_vision_model | siglip_vision_model | siglip_vision_model |
| vision image_size / patch_size | 512 / 16 | 512 / 16 | 512 / 16 | 512 / 16 |
| vision hidden_size | 768 | 768 | 768 | 768 |
| vision layers / heads | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| vision intermediate_size | 3072 | 3072 | 3072 | 3072 |
| image_seq_len from processor | 64 | 64 | 64 | 64 |

Effective source defaults when omitted:

- Top-level `classifier_pooling` defaults to `"cls"`, `classifier_dropout` to `0.0`, `classifier_bias` to `False`, and `tie_word_embeddings` to `False` in `ModernVBertConfig`.
- ModernBert defaults `max_position_embeddings=8192`, but the official ModernVBert configs set `7999`.
- SigLIP vision defaults `image_size=224`, but official configs set `512`.

## 3a. Family variation traps

- Native class composition is delegated: `text_config` must resolve to `ModernBertConfig`, and `vision_config` must resolve to `SiglipVisionConfig` for this report.
- `ModernVBERT/modernvbert-embed` and `ModernVBERT/bimodernvbert` advertise `BiModernVBert`, which is absent from the inspected source. DinoML should reject those architecture values for the native `ModernVBertForMaskedLM` path or route them to a separate audit.
- Processor class varies by repo (`Idefics3Processor`, `BiModernVBertProcessor`, `ColModernVBertProcessor`), while the local `modernvbert` directory has no processor implementation. For first integration, use the Idefics3 processor ABI only.
- ModernBert attention alternates full and sliding-window attention. Layers 0, 3, 6, ..., 21 are full attention; intervening layers use bidirectional sliding attention with `local_attention=128`.
- The text model uses a single packed `Wqkv` projection, split as `[q, k, v]` by viewing `[B, S, 3, heads, head_dim]` and unbinding on the `3` axis.
- MLP is gated: `Wi: 768 -> 2304`, chunked into two 1152 tensors, then `gelu(input) * gate`.
- Vision is NCHW in source. NHWC/channel-last is an optimization candidate only around patch embedding/vision blocks and requires axis guards.
- The connector assumes the vision token count is a perfect square and divisible by `pixel_shuffle_factor ** 2`; official shape is `1024 -> 64`.
- The image-token stitch is broad boolean indexing in source, but processor-produced positions are structured repeated spans. DinoML can lower a stricter indexed row copy with guards.
- SigLIP `vision_use_head` is absent in official configs, so source default enables the multihead pooling head, but ModernVBert ignores `pooler_output` from SigLIP and uses `last_hidden_state`; the SigLIP head can be skipped if the delegated model build allows it or dead-code elimination proves it unused.
- Classification head pooling can be `"cls"` or `"mean"`. Mean pooling uses `attention_mask.sum(dim=1)`, so it needs a divide-by-count guard.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs, position/row/col special token IDs in preprocessing.
- Reshape/view/flatten/transpose/permute/contiguous for QKV, patch embeddings, connector pixel shuffle, and attention outputs.
- Boolean comparisons, `sum` over image masks and attention masks, cumulative sum, pad, floor-div, modulo, and indexed gather for image-token mapping.
- `where` for final merge: `torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)`.
- Boolean mask indexing/scatter-like update: `image_embeds[image_mask] = image_hidden_states[...]`. Prefer a guarded indexed copy.

Neural primitives:

- NCHW `Conv2d(3 -> 768, kernel=16, stride=16, padding=valid)` for SigLIP patch embedding.
- Dense Linear projections:
  - SigLIP attention q/k/v/out: `768 -> 768`, with bias.
  - SigLIP MLP: `768 -> 3072 -> 768`, with GELU tanh approximation.
  - ModernBert Wqkv: `768 -> 2304`, no bias in official configs.
  - ModernBert Wo: `768 -> 768`, no bias.
  - ModernBert gated MLP Wi/Wo: `768 -> 2304`, chunk to `1152 + 1152`, then `1152 -> 768`, no bias.
  - Connector projection: `12288 -> 768`, no bias.
  - Prediction head: `768 -> 768`, activation GELU, LayerNorm.
  - LM head: `768 -> 50408`, bias true; weight tied/aliased to `model.text_model.embeddings.tok_embeddings.weight` by `_tied_weights_keys`.
- LayerNorm with eps `1e-6` for SigLIP and `1e-5` for ModernBert, usually no bias in ModernBert.
- Dropout appears in source but is disabled in eval/inference.

Attention primitives:

- SigLIP vision dense bidirectional MHA, 12 heads, head_dim 64, no causal mask.
- ModernBert bidirectional self-attention with full and sliding attention masks, 12 heads, head_dim 64, RoPE on q/k.
- Backends advertised by source: eager, SDPA, FlashAttention, Flex attention. DinoML can start with dense attention for small tests and then add full/sliding optimized paths.

Position/rotary/custom math:

- SigLIP absolute patch position embedding table `[1024, 768]` for 512x512/16x16 images.
- Optional SigLIP position interpolation if `interpolate_pos_encoding=True`; ModernVBert does not pass this flag explicitly.
- ModernBert RoPE for both full and sliding layer types, with config-specific theta `160000`.

Preprocessing-coupled ops:

- Resize longest edge to 2048, optional split into 512-square tiles plus a global resized tile, pad to batch max, rescale by `1/255`, normalize with mean/std `[0.5, 0.5, 0.5]`.
- Processor prompt expansion into image-token spans, row/column tags, and fake wrapper tokens.
- `pixel_attention_mask` to `patch_attention_mask` via `unfold(..., size=16, step=16)` and any-valid-pixel reduction.

Generation/cache ops:

- Not applicable for the primary target. This is an encoder and masked LM, not causal generation.

Optional/deferred heads:

- `ModernVBertModel`: required as base.
- `ModernVBertForMaskedLM`: required for first target.
- `ModernVBertForSequenceClassification`: optional; adds pooling/dropout/classifier.
- `ModernVBertForTokenClassification`: optional; adds per-token classifier.
- Training losses are deferred.

## 5. Layer/block breakdown

Vision path, per real image tile:

```text
pixel_values: [R, 3, 512, 512]
patch = Conv2d(3 -> 768, k=16, s=16)(pixel_values)
tokens = flatten_hw(patch).transpose(1, 2)      # [R, 1024, 768]
tokens += position_embedding[0:1024]

repeat 12 SigLIP layers:
  y = LayerNorm(tokens, eps=1e-6)
  q = Linear(768 -> 768, bias=True)(y)
  k = Linear(768 -> 768, bias=True)(y)
  v = Linear(768 -> 768, bias=True)(y)
  y = dense_bidirectional_attention(q, k, v)
  tokens = tokens + Linear(768 -> 768, bias=True)(y)
  y = LayerNorm(tokens, eps=1e-6)
  y = Linear(768 -> 3072)(y)
  y = gelu_pytorch_tanh(y)
  y = Linear(3072 -> 768)(y)
  tokens = tokens + y

tokens = LayerNorm(tokens, eps=1e-6)
```

Connector:

```text
image_hidden_states: [R, 1024, 768]
reshape to [R, 32, 32, 768]
pixel_shuffle_factor = 4
rearrange 4x4 local neighborhoods into channel dim
image_features: [R, 64, 12288]
image_features = Linear(12288 -> 768, bias=False)(image_features)
```

Text path:

```text
inputs_embeds = token_embedding(input_ids) or caller-provided embeddings
if images exist:
  replace every image_token_id embedding with ordered image_features rows

x = LayerNorm(token/image embeddings, eps=1e-5, bias=False)

repeat 22 ModernBert layers:
  attention type = full on layers 0,3,6,...,21, sliding otherwise
  a = Identity(x) for layer 0 else LayerNorm(x, eps=1e-5, bias=False)
  qkv = Linear(768 -> 2304, bias=False)(a)
  q,k,v = view(qkv, [B,S,3,12,64]).unbind(axis=3way)
  q,k = RoPE(q,k, theta per layer type)
  a = bidirectional_attention(q,k,v, mask=full_or_sliding)
  x = x + Linear(768 -> 768, bias=False)(a)
  m = LayerNorm(x, eps=1e-5, bias=False)
  input, gate = Linear(768 -> 2304, bias=False)(m).chunk(2, dim=-1)
  x = x + Linear(1152 -> 768, bias=False)(gelu(input) * gate)

x = final LayerNorm(x, eps=1e-5, bias=False)
```

Masked LM head:

```text
h = LayerNorm(gelu(Linear(768 -> 768, bias=False)(x)), eps=1e-5, bias=False)
logits = Linear(768 -> 50408, bias=True)(h)
```

## 6. Attention requirements

ModernVBert requires two encoder-style attention families:

SigLIP vision attention:

- Noncausal self-attention over image patch tokens.
- MHA, not GQA/MQA: 12 query heads and 12 key/value heads, head_dim 64.
- Query/key/value length is the patch token count, normally 1024 for 512x512 tiles.
- No RoPE; uses learned absolute position embeddings before attention.
- No KV cache.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS`, with eager fallback equal to matmul, additive mask, fp32 softmax, dropout, matmul.

ModernBert text attention:

- Noncausal bidirectional self-attention over the mixed text/image-token sequence.
- MHA, not GQA/MQA: 12 query heads and 12 key/value heads, head_dim 64.
- QKV projection is packed in one linear weight.
- RoPE is applied to q and k before attention.
- Full attention layers use a bidirectional mask.
- Sliding layers use a bidirectional sliding-window mask. Official `local_attention=128`, and the module passes `sliding_window=config.sliding_window + 1`, where `config.sliding_window = local_attention // 2`.
- No KV cache; no autoregressive decode path.

For first DinoML admission:

- Accept dense full attention for all layers for functional bringup only if explicitly labeled as a debug fallback, because sliding layers are source-visible behavior.
- Production parity should implement both full bidirectional and local bidirectional attention masks.
- Attention output tensors are optional diagnostics; hidden-state parity does not require materializing dense attention weights unless `output_attentions=True`.

## 7. Position encoding and custom math

ModernBert RoPE:

```python
def modernbert_rope(position_ids, head_dim, theta):
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate_half = lambda x: cat([-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Source computes cos/sin in fp32 and casts back to the model dtype. Official configs use the same `rope_theta=160000.0` for full and sliding attention, even though ModernBert source defaults differ when not overridden.

SigLIP vision positions:

- Learned absolute table of length `(image_size // patch_size) ** 2`.
- For official 512/16 configs, table length is 1024.
- If `interpolate_pos_encoding=True`, the table is reshaped to square grid, permuted to NCHW, bicubic-interpolated with `align_corners=False`, then flattened back. The ModernVBert forward path does not pass that flag explicitly, so first integration can require fixed 512x512 tiles and reject interpolation.

Connector pixel shuffle:

```python
def connector_shuffle(x, factor):
    b, seq, c = x.shape
    h = w = int(seq ** 0.5)
    x = x.view(b, h, w, c)
    x = x.view(b, h, w // factor, c * factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b, w // factor, h // factor, c * factor * factor)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(b, seq // (factor * factor), c * factor * factor)
```

This exact order matters for weight/layout rewrites.

## 8. Preprocessing and input packing

Processor contract, from the official `ModernVBERT/modernvbert` snapshots:

- `processor_class`: `Idefics3Processor`
- `image_processor_type`: `Idefics3ImageProcessor`
- `image_seq_len`: 64
- Image preprocessing: convert RGB, resize longest edge to 2048, split/pad to 512, rescale by `0.00392156862745098`, normalize with mean/std 0.5, pad to batch max.
- Output tensors: `pixel_values` shaped `[B, max_num_images, 3, H, W]`; `pixel_attention_mask` shaped `[B, max_num_images, H, W]`.
- Tokenizer special tokens include `<global-img>`, row/column tags up to 6x6, `<fake_token_around_image>`, `<image>`, and `<end_of_utterance>`.

Image splitting:

- If the resized image exceeds 512 on either side, the processor splits it into `ceil(H/512) * ceil(W/512)` square crops and appends a global 512x512 image.
- Prompt expansion inserts row/column tags plus 64 `<image>` tokens per crop/global image.
- Model-side `get_image_features` drops all-zero padded images before the vision encoder; if every image is padding, it keeps one empty image.

Embedding stitch:

- Source computes `image_mask = input_ids == image_token_id`.
- It checks each sample's image token count is divisible by `patch_size`, where `patch_size` is actually `image_hidden_states.shape[1]` after connector, normally 64.
- It computes per-sample block offsets from cumulative counts, then copies image feature rows in row-major order into matching token positions.
- Processor guarantees stricter structure than the raw boolean assignment: one or more contiguous spans of exactly 64 image tokens per image block, surrounded by fake/global/row-col text tokens.

Layout notes:

- Processor and source use NCHW image tensors.
- A channel-last optimization can cover resize/normalize/Conv2d only if the entire image tensor region is controlled and all Conv/flatten axes are rewritten. Otherwise mark the vision input region `no_layout_translation()`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: SigLIP non-overlap patch Conv2d to Linear

Source pattern:

```text
Conv2d(3 -> 768, kernel=16, stride=16, padding=valid)
flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten(NCHW, kh=16, kw=16, stride=16) -> Linear(768 input elems -> 768) -> AddPosition
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == valid`, `dilation == 1`, `groups == 1`
- Input height and width divisible by 16 after processor padding/resizing
- Flatten order exactly matches PyTorch NCHW Conv2d output then `flatten(2).transpose(1, 2)`

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * kh * kw)
b_linear = conv.bias
```

Failure cases: non-512 tile sizes with position interpolation, non-NCHW layout without a verified permutation, or dynamic image sizes not divisible by patch size.

Parity test sketch: compare patch tokens before position add for random `[2,3,512,512]` inputs in fp32 and fp16.

### Rewrite: ModernBert packed QKV

Source pattern:

```text
qkv = Linear(768 -> 2304)(x)
qkv.view(B, S, 3, 12, 64)
query, key, value = unbind(dim=-3)
```

Replacement:

```text
single GEMM -> split metadata views -> RoPE(q,k) -> attention
```

Preconditions:

- `hidden_size == num_heads * head_dim`
- Packed order is all-Q block, all-K block, all-V block as produced by view/unbind above.
- Weight is dense row-major in PyTorch linear convention `[2304, 768]`.

Failure cases: checkpoint with `attention_bias=True`, alternate head_dim, or non-ModernBert text_config needs separate admission.

### Rewrite: image-token boolean scatter to segmented row copy

Source pattern:

```text
image_embeds = zeros_like(inputs_embeds)
image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]
merged = where(image_mask[...,None], image_embeds, inputs_embeds)
```

Replacement:

```text
for each verified image span:
  copy image_features[image_block, :, :] into inputs_embeds[token_start:token_start+64, :]
```

Preconditions:

- Processor-owned prompt expansion or validated input IDs.
- Each image span contains exactly `image_seq_len` contiguous image tokens.
- Total spans equals first dimension of `image_hidden_states`.
- No arbitrary interleaving of image tokens.

Failure cases: caller-provided arbitrary `inputs_embeds` with no `input_ids`, malformed image token counts, or non-contiguous image tokens.

### Rewrite: connector shuffle plus projection

Source pattern:

```text
[R, 1024, 768] -> pixel_shuffle(factor=4) -> [R, 64, 12288] -> Linear(12288 -> 768)
```

Replacement:

```text
layout-aware gather/reshape -> GEMM
```

Preconditions:

- Vision token count is square and equals `(image_size // patch_size) ** 2`.
- Factor divides both grid dimensions.
- Preserve exact reorder shown in section 7.

Failure cases: non-square vision token sequence, dynamic interpolated image grids, or factor mismatch.

## 10. Kernel fusion candidates

Highest priority:

- ModernBert LayerNorm + packed QKV + RoPE preparation. This is repeated 22 times and feeds attention directly.
- Bidirectional full/sliding attention kernels. Sliding layers are two-thirds of the text stack and should not be silently densified for production.
- ModernBert gated MLP fusion: `Linear -> chunk -> gelu -> multiply -> Linear`. This is a major per-layer cost and maps cleanly to fused activation multiply plus GEMM.
- Connector pixel shuffle plus projection. The `12288 -> 768` projection is large for each 64-token image block; preserving shuffle order while fusing layout movement avoids a bulky temporary.

Medium priority:

- SigLIP patch Conv2d-to-GEMM or direct optimized patch embedding.
- SigLIP vision MHA and MLP fusions. Vision is only 12 layers over 1024 tokens per real tile, so it is still material for multi-image documents.
- Image-token segmented copy. Replacing boolean scatter with guarded row copy reduces unsupported indexing pressure.
- Prediction head plus LM projection, including tied weight handling.

Lower priority:

- SigLIP multihead pooling head, because ModernVBert does not consume SigLIP `pooler_output`.
- Sequence/token classification heads.
- Bicubic position interpolation, unless first integration admits non-512 image tiles.

## 11. Runtime staging plan

Stage 1: config and weights admission

- Parse `ModernVBertConfig`, nested ModernBert and SigLIP configs.
- Reject unavailable `architectures: ["BiModernVBert"]` for the native masked-LM path.
- Preserve tied `lm_head.weight` alias with token embeddings.

Stage 2: isolated block parity

- Run SigLIP patch embedding and one encoder layer on fixed 512 tiles.
- Run ModernBert embeddings, RoPE, one full-attention layer, one sliding-attention layer, and gated MLP.
- Run connector shuffle/projection on `[R,1024,768]`.

Stage 3: encoder-only ModernVBert parity with precomputed `image_hidden_states`

- Bypass image preprocessing and vision encoder.
- Validate image-token stitch plus ModernBert hidden states.

Stage 4: full visual masked-LM parity

- Accept `pixel_values`, `pixel_attention_mask`, `input_ids`, `attention_mask`.
- Run SigLIP vision, connector, stitch, text encoder, prediction head, LM head.

Stage 5: optimized attention and fusions

- Replace dense debug attention with full/sliding specialized attention.
- Add packed QKV/RoPE fusion and gated MLP fusion.

Stage 6: optional heads and processor variants

- Add sequence classification and token classification.
- Separately audit BiModernVBert and ColModernVBert processors/classes if those are product targets.

Initially stub or reject:

- Training losses.
- `output_attentions=True`.
- `inputs_embeds` image-token detection path without `input_ids`.
- Non-512/interpolated vision tiles.
- BiModernVBert/ColModernVBert heads.

## 12. Parity and validation plan

Unit parity:

- Connector shuffle parity for random `[1,1024,768]` and `[3,1024,768]`.
- Image-token stitch parity for processor-valid spans: one image, multiple images in one sample, and mixed image counts across batch.
- RoPE parity for full and sliding layer types, fp32 and fp16/bf16 cast-back.
- ModernBert packed QKV split order parity.
- Sliding mask parity for `local_attention=128`, including short sequences below the window.

Layer parity:

- SigLIP first layer output after patch embedding and after one encoder layer.
- ModernBert layer 0 full attention and layer 1 sliding attention.
- Full ModernBert text encoder with random `inputs_embeds`.

End-to-end:

- Reproduce the local integration test shape: image plus text `"This [MASK] is on the wall."`, compare top-5 masked-token logits/probabilities against Transformers.
- Image-free text path should still run as a ModernBert-like encoder/masked LM.
- Precomputed `image_hidden_states` path should match `pixel_values` path at the text encoder boundary.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-4` for logits after full model, stricter for isolated linear/LN blocks.
- fp16/bf16: start with `rtol=5e-2`, `atol=5e-2` for full logits and tighten per fused kernel after backend selection.

## 13. Performance probes

- Image processor throughput: resize/split/pad/rescale/normalize images/sec and produced tile count distribution.
- Vision encoder throughput: real image tiles/sec for `[R,3,512,512]`.
- Connector throughput: `[R,1024,768] -> [R,64,768]`, with and without fused shuffle/projection.
- Text encoder throughput by sequence length, especially sequence lengths from prompt-expanded image tokens.
- Full versus sliding attention layer timing; sweep sequence length and image token count.
- Masked LM logits cost: all-token logits versus mask-position-only logits as an optional optimization.
- Memory probes: image features cache size, attention temporary size for full layers, sliding attention temporary size.
- End-to-end document batch throughput: sweep batch size, images per sample, and split rows/cols.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Sequence classification and token classification heads until masked-LM path is stable.
- `BiModernVBert`, `ColModernVBert`, and retrieval heads.
- General boolean scatter; use guarded image-span copy first.
- Arbitrary image sizes with SigLIP position interpolation.
- Returning dense attention weights.
- Multi-GPU/model parallel quirks; the local test suite already skips one data-parallel case due to dtype/device behavior.
- Quantization or packed weight formats; no source-coupled quantized path was found in the native ModernVBert source.

## 15. Final implementation checklist

- [ ] Parse top-level `ModernVBertConfig` and nested `ModernBertConfig`/`SiglipVisionConfig`.
- [ ] Reject missing `BiModernVBert` architecture for the native masked-LM target.
- [ ] Load token embedding, tied LM head, SigLIP, ModernBert, connector, and head weights with alias preservation.
- [ ] Implement fixed-shape SigLIP patch embedding and learned position add.
- [ ] Implement SigLIP encoder layer parity.
- [ ] Implement connector pixel shuffle and `12288 -> 768` projection.
- [ ] Implement processor-valid image-token segmented row copy.
- [ ] Implement ModernBert embeddings, LayerNorm, packed QKV, RoPE, full attention, sliding attention, gated MLP, and final norm.
- [ ] Implement ModernVBert prediction head and LM head.
- [ ] Add unit parity for connector, RoPE, QKV split, image stitch, and sliding mask.
- [ ] Add one-layer SigLIP and ModernBert parity tests.
- [ ] Add full visual masked-LM parity against `ModernVBERT/modernvbert`.
- [ ] Benchmark processor, vision encoder, connector, text encoder, and full masked-LM throughput separately.
