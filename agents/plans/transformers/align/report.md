# ALIGN audit for DinoML v2

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: kakaobrain/align-base
Config source: https://huggingface.co/kakaobrain/align-base/raw/main/config.json
Source files inspected:
  transformers/src/transformers/models/align/configuration_align.py
  transformers/src/transformers/models/align/modeling_align.py
  transformers/src/transformers/models/align/processing_align.py
  transformers/src/transformers/models/align/convert_align_tf_to_hf.py
  transformers/src/transformers/models/efficientnet/image_processing_efficientnet.py
  transformers/src/transformers/models/bert/tokenization_bert.py
Snapshots:
  agents/plans/transformers/align/_sources/
Any missing files or assumptions:
  No ALIGN-specific tokenizer or image processor exists; AlignProcessor wraps a BERT tokenizer and EfficientNetImageProcessor.
  kakaobrain/coyo-align-b7-base is visible in the HF API but raw config/preprocessor/tokenizer URLs returned 404, so it is recorded as an inaccessible representative gap.
```

Representative URLs:

- [kakaobrain/align-base](https://huggingface.co/kakaobrain/align-base)
- [kakaobrain/coyo-align-b7-base](https://huggingface.co/kakaobrain/coyo-align-b7-base) returned 404 for raw config assets during this audit.

## 2. High-level architecture

ALIGN is a dual-encoder contrastive model:

```text
CPU image preprocessing + BERT tokenization
  -> EfficientNet-derived vision encoder -> global pool -> image feature
  -> BERT-like text encoder -> first-token slice -> text projection
  -> L2 normalize both branches -> text/image similarity matrix -> optional contrastive loss
```

Stage decomposition:

- CPU/data pipeline: image resize/crop/rescale, BERT WordPiece tokenization, padding to 64 by processor default.
- Vision branch: NCHW EfficientNet-B7-like convolutional encoder, independently cacheable per image.
- Text branch: encoder-only BERT stack, independently cacheable per text.
- Contrastive head: normalize image/text features, compute `text_embeds @ image_embeds.T / temperature`, transpose for `logits_per_image`.

There is no autoregressive decode, no KV cache, and no cross-modal embedding stitch. The final similarity matrix is the only cross-branch interaction.

## 3. Important config dimensions

Source defaults versus checkpoint config differ in a few operator-significant places.

| Field | Source default | `kakaobrain/align-base` checkpoint | Notes |
|---|---:|---:|---|
| projection_dim | 640 | 640 | Only text branch has explicit `nn.Linear(768 -> 640)` in `AlignModel`. |
| temperature_init_value | 1.0 | 1.0 | Runtime parameter divides logits; not an exponentiated logit scale. |
| text hidden_size | 768 | 768 | BERT-base width. |
| text layers | 12 | 12 | Encoder-only. |
| text heads / head_dim | 12 / 64 | 12 / 64 | MHA, no GQA/MQA. |
| text intermediate_size | 3072 | 3072 | GELU FFN. |
| vocab_size | 30522 | 30522 | BERT uncased tokenizer assets. |
| max_position_embeddings | 512 | 512 | Processor defaults to max_length 64. |
| type_vocab_size | 2 | 2 | Token type IDs default to zeros. |
| vision image_size | 600 | 289 | Checkpoint aligns with center-crop size. |
| vision width/depth coefficient | 2.0 / 3.1 | 2.0 / 3.1 | EfficientNet-B7-like scaling. |
| vision hidden_dim | 2560 | 640 | Used as final pooling kernel size, not a projection width. |
| vision actual final channels | 640 | 640 | Derived from `round_filters(320)`. |
| vision `num_hidden_layers` | 64 | 64 | Misleading for execution: actual block count is 55. |
| vision activation | swish | swish | Applies in stem, expansion, depthwise, SE reduce. |
| dtype | source init dtype | float32 | From checkpoint config. |

Representative checkpoint sweep:

| Model id | Accessible config | Text encoder | Vision branch | Processor/config variation |
|---|---:|---|---|---|
| `kakaobrain/align-base` | yes | BERT-base: 12x768, 12 heads | EfficientNet-style, 55 MBConv blocks, 640 final channels | resize 346, center crop 289, rescale offset to `[-1, 1]`, max text length 64 |
| `kakaobrain/coyo-align-b7-base` | no, 404 | unknown from config | likely ALIGN/B7 from metadata, but not source-confirmed | raw config/preprocessor/tokenizer URLs 404 |

For `align-base`, rounded vision stages are:

| Stage | Blocks | In channels first block | Out channels | Kernel | First stride | Expand |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 64 | 32 | 3 | 1 | 1 |
| 2 | 7 | 32 | 48 | 3 | 2 | 6 |
| 3 | 7 | 48 | 80 | 5 | 2 | 6 |
| 4 | 10 | 80 | 160 | 3 | 2 | 6 |
| 5 | 10 | 160 | 224 | 5 | 1 | 6 |
| 6 | 13 | 224 | 384 | 5 | 2 | 6 |
| 7 | 4 | 384 | 640 | 3 | 1 | 6 |

## 3a. Family variation traps

- `vision_config.num_hidden_layers` is not the forward loop count. The encoder builds `ceil(depth_coefficient * num_block_repeats[i])` blocks per stage.
- `AlignVisionConfig.hidden_dim` is used as an `AvgPool2d` or `MaxPool2d` kernel size. Do not treat it as a linear projection dimension.
- Source defaults set `vision_config.image_size=600` and `hidden_dim=2560`; the public checkpoint sets `image_size=289` and `hidden_dim=640`.
- The full model projects text features only. Image features are the pooled vision channels and must match `projection_dim` by architecture/config.
- `get_text_features()` returns the projected first-token feature without L2 normalization; full `forward()` normalizes both branches before logits.
- The source has a BERT pooler, but `AlignModel.forward()` ignores the pooler output and slices `text_outputs[0][:, 0, :]`.
- The vision source is NCHW PyTorch Conv2d/BatchNorm2d. NHWC/channel-last is an optimization candidate only under guarded layout rewrites.
- `temperature` is a learned scalar divisor. CLIP-style `exp(logit_scale)` is not the ABI here.
- Training loss uses cross entropy with label smoothing `0.1`; inference does not need it.
- No ALIGN-specific tokenizer exists. The checkpoint tokenizer config says `BertTokenizer`, lower-case WordPiece, model max length 64.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape, reshape, transpose, contiguous, first-token slice `[:, 0, :]`.
- Broadcast add for attention mask and residual connections.
- L2 norm over feature axis with keepdim, elementwise divide.
- Matrix transpose and GEMM for contrastive logits: `[T, 640] @ [640, I] -> [T, I]`.

Neural network primitives:

- Embedding tables: word `[30522, 768]`, position `[512, 768]`, token type `[2, 768]`.
- LayerNorm over hidden dim with eps `1e-12`.
- Dropout is present but can be disabled for inference.
- Linear projections with bias: BERT Q/K/V/O `768 -> 768`, FFN `768 -> 3072 -> 768`, pooler `768 -> 768`, text projection `768 -> 640`.
- Conv2d NCHW, BatchNorm2d, ZeroPad2d, Swish, GELU, tanh, sigmoid.
- Depthwise Conv2d with `groups=in_channels`.
- AdaptiveAvgPool2d output size 1 for squeeze-excite.
- AvgPool2d or MaxPool2d final pooling with `kernel_size=config.hidden_dim`, `ceil_mode=True`.

Attention primitives:

- Encoder noncausal dense self-attention only.
- Q/K/V are separate linear layers, shape `[B, S, 768] -> [B, 12, S, 64]`.
- Attention scores use `query @ key.transpose(-2, -1) * head_dim**-0.5`.
- Add extended attention mask, softmax in fp32, cast back to query dtype, dropout in training, multiply by V.
- No RoPE, ALiBi, relative bias, sliding window, cache, or cross-attention.

Preprocessing-coupled ops:

- Image: resize to 346x346, center crop to 289x289 for checkpoint, rescale by `1/127.5`, subtract 1 when `rescale_offset=True`.
- Text: BERT WordPiece with `[CLS] ... [SEP]`, lower-case, padding default `max_length=64`.

Parameter sharing:

- No cross-layer sharing.
- Text embeddings are not tied to a decoder head because there is no LM head.

## 5. Layer/block breakdown

Text embeddings:

```text
input_ids/token_type_ids/position_ids
  -> word + token_type + position embeddings
  -> LayerNorm(eps=1e-12)
  -> dropout
```

Text layer, repeated 12 times:

```text
x0 = x
q,k,v = Linear(768 -> 768)(x), separate weights/biases
q,k,v = reshape to [B, 12, S, 64]
a = softmax((q @ k.T) / sqrt(64) + attention_mask)
x = a @ v
x = reshape to [B, S, 768]
x = LayerNorm(Dropout(Linear(768 -> 768)(x)) + x0)
y = GELU(Linear(768 -> 3072)(x))
x = LayerNorm(Dropout(Linear(3072 -> 768)(y)) + x)
```

Vision stem:

```text
pixel_values [B, 3, H, W]
  -> ZeroPad2d(right=1, bottom=1)
  -> Conv2d(3 -> 64, kernel=3, stride=2, bias=False)
  -> BatchNorm2d
  -> Swish
```

Vision MBConv block:

```text
residual = x
if expand_ratio != 1:
  x = Conv2d(C -> C*expand, kernel=1, bias=False) -> BatchNorm2d -> Swish
if stride == 2:
  x = ZeroPad2d(correct_pad(kernel))
x = DepthwiseConv2d(groups=C, kernel=3 or 5, stride=1 or 2, bias=False)
x = BatchNorm2d -> Swish
se = AdaptiveAvgPool2d(1)(x)
se = Conv2d(C -> max(1, input_stage_channels*0.25), kernel=1) -> Swish
se = Conv2d(... -> C, kernel=1) -> Sigmoid
x = x * se
x = Conv2d(C -> out_channels, kernel=1, bias=False) -> BatchNorm2d
if stride == 1 and block is not the first repeat in its stage:
  x = Dropout(drop_connect_rate * block_index / num_blocks)(x) + residual
```

Full contrastive head:

```text
image_embeds = vision_pooler(vision_last_hidden_state).reshape([B_image, 640])
text_embeds = text_projection(text_last_hidden_state[:, 0, :])  # [B_text, 640]
image_embeds = image_embeds / ||image_embeds||_2
text_embeds = text_embeds / ||text_embeds||_2
logits_per_text = text_embeds @ image_embeds.T / temperature
logits_per_image = logits_per_text.T
```

## 6. Attention requirements

ALIGN text attention is BERT-style encoder MHA:

- Noncausal self-attention.
- 12 query heads, 12 key/value heads, head dim 64.
- Query/key/value widths are all 768.
- Query length equals key/value length for normal text input.
- Mask is an extended additive attention mask broadcastable to `[B, heads, S, S]`.
- No packed/varlen source path.
- No local/sliding/block sparse pattern.
- No ALiBi, RoPE, relative position bias, or cache.
- SDPA-style lowering is compatible if it preserves additive mask semantics, fp32 softmax parity expectations, dropout disabled in inference, and noncausal mode.

## 7. Position encoding and custom math

Text positions use learned absolute embeddings. Position IDs default to `arange(max_position_embeddings)[:seq_length]` and are not rotary or sinusoidal.

Source-specific contrastive math:

```python
def align_logits(text_hidden_0, image_pooled, text_projection, temperature):
    text_embeds = text_projection(text_hidden_0)
    image_embeds = image_pooled
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_text = text_embeds @ image_embeds.t() / temperature
    return logits_per_text, logits_per_text.t()
```

EfficientNet rounding for channel dimensions:

```python
def round_filters(num_channels, width_coefficient=2.0, divisor=8):
    scaled = num_channels * width_coefficient
    new_dim = max(divisor, int(scaled + divisor / 2) // divisor * divisor)
    if new_dim < 0.9 * scaled:
        new_dim += divisor
    return int(new_dim)
```

The rounded channel plan is config-dependent and should be computed during model admission.

## 8. Preprocessing and input packing

Image processor contract for `align-base`:

- Processor class: `AlignProcessor`.
- Image processor: `EfficientNetImageProcessor`.
- Resize to `{"height": 346, "width": 346}` with `resample=2` from checkpoint config; conversion script used PIL bilinear.
- Center crop to `289x289`.
- `do_rescale=True`, `rescale_factor=1/127.5`, `rescale_offset=True`, so tensor values become approximately `image * (1/127.5) - 1`.
- `do_normalize=False`, `include_top=False` for the checkpoint.
- Processor emits `pixel_values`, conventionally NCHW for PyTorch models.

Text processor contract:

- Tokenizer class: `BertTokenizer`.
- Lowercase WordPiece, Chinese char splitting enabled, special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`.
- `AlignProcessor` defaults text kwargs to `padding="max_length"` and `max_length=64`.
- Token type IDs are accepted; if omitted, the model uses an all-zero buffer.

Dual-encoder ABI:

- Image branch input: `pixel_values [B_image, 3, 289, 289]` for the representative checkpoint.
- Text branch input: `input_ids`, `attention_mask`, optional `token_type_ids`, optional `position_ids`.
- Branch outputs can be cached independently as pooled image features and projected text features. Full `forward()` normalization must be included before final logits.
- Output orientation is explicit: `logits_per_text [B_text, B_image]`; `logits_per_image [B_image, B_text]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv2d to GEMM

Source pattern:

```text
Conv2d(Cin -> Cout, kernel=1, stride=1, padding="same") on NCHW
```

Replacement:

```text
transpose/reshape NCHW -> [B*H*W, Cin] -> GEMM(weight.T) -> reshape back
```

Preconditions:

- Kernel size exactly 1.
- Groups exactly 1.
- Stride 1 and no spatial padding effect.
- Bias handling matches source module; many expansion/projection convs are bias-free, SE convs have bias.
- Layout pass must preserve NCHW-visible consumers or rewrite all consumer axes.

Failure cases:

- Depthwise convs, stride-2 convs, or any consumer requiring source NCHW strides without a layout guard.

Parity test sketch:

- Compare each pointwise conv module against PyTorch for random NCHW tensors across fp32/fp16 tolerances and multiple spatial sizes.

### Rewrite: NCHW EfficientNet region to channel-last guarded fusion

Source pattern:

```text
Conv2d/DepthwiseConv2d -> BatchNorm2d -> Swish, plus SE 1x1 convs and sigmoid multiply
```

Replacement:

```text
NHWC/channel-last internal layout kernels across a whole vision block or stage
```

Preconditions:

- Rewrite enters at a layout boundary and exits with the same semantic tensor expected by downstream source code.
- All axis-sensitive ops are rewritten: BatchNorm channel axis, AdaptiveAvgPool spatial axes, final pooling, depthwise group channel mapping.
- ZeroPad2d `left/right/top/bottom` semantics are preserved.
- Fallback remains available for unusual image sizes or pooling modes.

Failure cases:

- Partial layout conversion that leaves BatchNorm/pooling using NCHW axes.
- `pooling_type="max"` untested path if only mean pooling is validated.

Parity test sketch:

- Stage-by-stage vision parity with fixed random weights and random images, including odd crop size 289 to exercise `correct_pad`.

### Rewrite: contrastive head fusion

Source pattern:

```text
feature_norm(image) + feature_norm(text) + matmul + scalar divide + transpose
```

Replacement:

```text
normalize rows -> GEMM -> scalar scale; materialize transpose only if both orientations are requested
```

Preconditions:

- Inference mode or no need to reproduce loss gradients.
- `temperature` scalar is loaded and nonzero.
- Consumers declare whether they need `logits_per_text`, `logits_per_image`, or both.

Failure cases:

- Returning unnormalized `get_text_features()`/`get_image_features()` APIs through the fused full-forward path.

## 10. Kernel fusion candidates

Highest priority:

- Vision Conv2d/BatchNorm/Swish and depthwise Conv2d/BatchNorm/Swish regions. These dominate the image encoder.
- BERT encoder attention with additive mask, using SDPA/FlashAttention-style kernels for noncausal self-attention.
- Contrastive normalize + GEMM + scale for batched retrieval scoring.

Medium priority:

- BERT FFN Linear/GELU/Linear with residual LayerNorm boundaries.
- SE block fusion: adaptive average pool, two 1x1 convs, swish/sigmoid, channel multiply.
- NHWC/channel-last vision stage kernels under strict layout guards.

Lower priority:

- Training-only label-smoothed contrastive loss.
- BERT pooler tanh path, because `AlignModel.forward()` does not consume it.
- Final logits transpose elision when only one output orientation is requested.

## 11. Runtime staging plan

Stage 1: parse `AlignConfig`, instantiate text/vision sub-configs, compute real vision block plan from rounded repeats and filters.

Stage 2: load weights and run independent text encoder parity for fixed padded token batches.

Stage 3: implement/validate EfficientNet-derived vision encoder in NCHW with checkpoint preprocessing.

Stage 4: wire contrastive ABI: text first-token projection, image pooled features, L2 normalization, temperature division, both output orientations.

Stage 5: add branch-level caching for image and text embeddings before the similarity GEMM.

Stage 6: add guarded vision layout/fusion passes and attention backend selection.

Training loss, dropout behavior, and inaccessible `coyo-align-b7-base` variants can be stubbed or deferred for first inference integration.

## 12. Parity and validation plan

- Config parser test: assert `align-base` builds 55 vision blocks, not 64.
- Text embedding parity: token IDs, token type omission, position ID defaults, attention mask extension.
- Single text layer parity in fp32, then 12-layer text encoder parity with dropout disabled.
- Vision stem and per-stage parity in fp32 for `[1, 3, 289, 289]`, including stride-2 padding.
- SE block parity with fixed weights and dynamic batch sizes.
- Full vision encoder parity to pooled `[B, 640]`.
- Contrastive head parity: verify `logits_per_image == logits_per_text.T` and temperature division.
- End-to-end zero-shot smoke using checkpoint processor outputs.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16 after optimized kernels `rtol=5e-3, atol=5e-3`, with fp32 accumulation for softmax/norm where practical.

## 13. Performance probes

- CPU preprocessing throughput: resize/crop/tokenization separately from GPU execution.
- Vision encoder latency by stage, NCHW versus channel-last guarded path.
- Text encoder latency by sequence length: 16, 32, 64, 128, 512.
- Similarity head throughput for image/text matrix sizes, e.g. `I,T` in `{1, 8, 64, 1024}`.
- Branch embedding cache hit/miss benchmark for retrieval workloads.
- Attention backend comparison for BERT text branch at max_length 64.
- Batch-size sweep for images at 289x289 and any dynamic image-size admission variants.

## 14. Skip/defer list

- Autoregressive generation and KV cache: not applicable.
- Cross-attention and multimodal token stitching: not applicable.
- Training dropout and label-smoothed contrastive loss for first inference path.
- `pooling_type="max"` until a checkpoint requiring it is found.
- Inaccessible `kakaobrain/coyo-align-b7-base` specifics until config assets are available.
- Quantized/packed weights: no source-coupled quantized format in inspected implementation.
- Multi-GPU/tensor parallel execution.

## 15. Final implementation checklist

- [ ] Parse `AlignConfig`, `AlignTextConfig`, and `AlignVisionConfig`.
- [ ] Load BERT tokenizer/processor metadata separately from the runtime graph.
- [ ] Implement BERT-like text embeddings, MHA, FFN, residual LayerNorm stack.
- [ ] Implement EfficientNet-derived NCHW stem and MBConv blocks with SE.
- [ ] Compute rounded vision filters and rounded repeat counts from config.
- [ ] Add checkpoint preprocessing contract for resize/crop/rescale offset.
- [ ] Implement text first-token projection `Linear(768 -> 640)`.
- [ ] Preserve image pooled feature ownership without adding a nonexistent image projection.
- [ ] Implement L2 feature normalization and temperature-divided similarity GEMM.
- [ ] Return both `logits_per_text [T, I]` and `logits_per_image [I, T]`.
- [ ] Add parity tests for text branch, vision branch, contrastive head, and full processor path.
- [ ] Add guarded NHWC/channel-last fusion experiments only after NCHW parity is stable.
