# AltCLIP model-family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: BAAI/AltCLIP family
Config source: HF config/preprocessor/tokenizer JSON snapshots under agents/plans/transformers/altclip/_sources/
Source files inspected:
- X:/H/transformers/src/transformers/models/altclip/modeling_altclip.py
- X:/H/transformers/src/transformers/models/altclip/configuration_altclip.py
- X:/H/transformers/src/transformers/models/altclip/modular_altclip.py
- X:/H/transformers/src/transformers/models/altclip/processing_altclip.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/activations.py
- X:/H/transformers/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py
Any missing files or assumptions:
- modeling_altclip.py and configuration_altclip.py are generated from modular_altclip.py. Treat generated files as runtime source basis and modular_altclip.py as future source-edit basis.
- No remote code is required for the inspected public BAAI checkpoints.
- xxai/AltCLIP exposes HF repo metadata but no standard config/preprocessor/tokenizer files, so it is not used for dimensions.
```

HF configs inspected:

- [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP), repo sha `17788d7d45af0e41d0f10682fe91d655ac96435c`
- [BAAI/AltCLIP-m9](https://huggingface.co/BAAI/AltCLIP-m9), repo sha `41d03cb9f2d9f904e7c63e90817d4bbed0d4a3a8`
- [BAAI/AltCLIP-m18](https://huggingface.co/BAAI/AltCLIP-m18), repo sha `4e7c13dba5f09db51efb59429de18d3c81473b0d`
- [hf-tiny-model-private/tiny-random-AltCLIPModel](https://huggingface.co/hf-tiny-model-private/tiny-random-AltCLIPModel), repo sha `8251bf655999b87ecce4d64f64457f27820d4dc0`; despite the namespace, config/preprocessor/tokenizer JSON was accessible during this audit.

Primary runtime target: inference-only CLIP-like dual encoder for image-text retrieval / zero-shot classification. Training contrastive loss is optional. There is no autoregressive generation, decode loop, or KV cache.

## 2. High-level architecture

AltCLIP is a two-tower contrastive model:

```text
text tokenizer -> AltRoberta text encoder -> text transformation -> text projection -> L2 normalize
image processor -> ViT vision encoder -> visual projection -> L2 normalize
normalized text/image features -> text @ image.T -> exp(logit_scale) multiply -> logits_per_text/logits_per_image
```

Stages that can be validated independently:

- CPU/data pipeline: XLM-R tokenizer and CLIP-style image resize/crop/rescale/normalize.
- Text tower: XLM-R/RoBERTa-like bidirectional encoder, first-token pooling after a text transformation.
- Vision tower: ViT patch embedding and bidirectional encoder, class-token pooling.
- Projection/similarity head: bias-free text and visual projections, L2 normalization, similarity matrix, scalar logit scale.

No prefix construction, prefill, decode, sampling, or cache state is part of the primary runtime.

## 3. Important config dimensions

Source defaults:

| Field | Text default | Vision default | Notes |
|---|---:|---:|---|
| hidden_size | 1024 | 768 | Checkpoints override vision to 1024 or 1280. |
| num_hidden_layers | 24 | 12 | Checkpoints override vision depth. |
| num_attention_heads | 16 | 12 | Head dim must divide hidden size. |
| head_dim | 64 | 64 | Derived from hidden_size / num_attention_heads for inspected configs. |
| intermediate_size | 4096 | 3072 | Standard non-gated MLP. |
| vocab_size | 250002 | n/a | XLM-R tokenizer family. |
| max_position_embeddings | 514 | n/a | Text positions start at pad_token_id + 1 for non-pad tokens. |
| image_size | n/a | 224 | Tiny random uses 30. |
| patch_size | n/a | 32 default | BAAI checkpoints use 14; tiny uses 2. |
| hidden_act | gelu | quick_gelu | QuickGELU is `x * sigmoid(1.702*x)`. |
| projection_dim | top-level 768 | vision sub-config default 512 | Top-level controls final contrastive feature size. |
| text project_dim | 768 | n/a | Text encoder maps hidden_size to project_dim before top-level projection. |
| logit_scale_init_value | 2.6592 | n/a | Runtime uses `exp(logit_scale)`. |
| cache support | none | none | Encoder-only bidirectional attention. |

Representative checkpoint sweep:

| Checkpoint | Text layers/hidden/heads | Text project_dim | Vision layers/hidden/heads | Patch/grid/tokens | Final projection | Dropout notes |
|---|---:|---:|---:|---:|---:|---|
| BAAI/AltCLIP | 24 / 1024 / 16 | 768 | 24 / 1024 / 16 | 14, 16x16, 257 tokens | 768 | text hidden/attn dropout 0.1; vision attention dropout 0.0 |
| BAAI/AltCLIP-m9 | 24 / 1024 / 16 | 768 | 24 / 1024 / 16 | 14, 16x16, 257 tokens | 768 | text dropout fields 0.0 |
| BAAI/AltCLIP-m18 | 24 / 1024 / 16 | 1024 | 32 / 1280 / 16 | 14, 16x16, 257 tokens | 1024 | larger vision tower and projection |
| tiny-random-AltCLIPModel | 5 / 32 / 4 | 32 | 5 / 32 / 4 | 2, 15x15, 226 tokens | 64 | test-sized odd MLP intermediate 37 |

Processor/tokenizer contract from snapshots:

| Checkpoint | Image preprocessing | Tokenizer |
|---|---|---|
| BAAI/AltCLIP | resize shortest_edge 224, center crop 224x224, rescale 1/255, normalize CLIP mean/std | XLMRobertaTokenizer, model_max_length 512, `<s>`/`</s>` special tokens, pad id 1 |
| BAAI/AltCLIP-m9 | older preprocessor JSON uses scalar size/crop_size 224 and omits explicit rescale fields | XLMRobertaTokenizer, model_max_length 512 |
| BAAI/AltCLIP-m18 | same structured image processor as BAAI/AltCLIP | XLMRobertaTokenizer, model_max_length 512 |
| tiny-random | resize/crop 30, rescale 1/255, CLIP mean/std | XLMRobertaTokenizer, model_max_length 512 |

## 3a. Family variation traps

- The final contrastive dimension is top-level `projection_dim`, not `vision_config.projection_dim`. `AltCLIPModel.__init__` constructs `visual_projection(hidden -> projection_dim)` and `text_projection(project_dim -> projection_dim)`.
- Text `project_dim` can differ from top-level `projection_dim`; m18 uses text `project_dim=1024` and final `projection_dim=1024`, while tiny uses text `project_dim=32` and final `projection_dim=64`.
- The text branch is not OpenAI CLIP text pooling. It is XLM-R/RoBERTa-like, bidirectional, uses first-token pooling after `pre_LN` and a transformation linear over every token.
- `get_text_features()` differs slightly from `AltCLIPTextModel.forward()` naming: it reads `text_outputs.last_hidden_state[:, 0, :]` after the text model has already transformed sequence states, then applies the top-level projection.
- Vision config `image_size` and `patch_size` are scalar in inspected configs. The generated config allows list/tuple, but the modeling code computes `(image_size // patch_size) ** 2`, so non-square/list configs should be rejected or separately audited.
- Vision source consumes NCHW `pixel_values`; NHWC is only an optimization candidate around patch embedding and local vision encoder regions.
- `interpolate_pos_encoding=True` admits higher resolutions by bicubic interpolation of 2D patch position embeddings. Default `False` requires exact `height == image_size` and `width == image_size`.
- The attention backend can be eager, SDPA, FlashAttention, or flex through Transformers common `ALL_ATTENTION_FUNCTIONS`; all are noncausal encoder attention.
- Checkpoint config fields inherited from broader configs, such as `is_decoder`, `cross_attention_hidden_size`, or tokenizer_class fields inside sub-configs, are not used by this AltCLIP source for generation or cross-attention.
- Training-only dropout and contrastive loss exist in source, but inference should run with dropout disabled and can omit the loss path.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape checks for exact image size unless interpolation is enabled.
- NCHW Conv2d patch embedding, flatten spatial `[B, C, Gh, Gw] -> [B, Gh*Gw, C]`, transpose, concat class token.
- Text gather for default token type IDs using position IDs.
- Text position ID construction: `ne(pad)`, `cumsum(dim=1)`, mask multiply, long cast, add `pad_token_id`.
- View/reshape/transpose for attention heads: `[B, S, H] -> [B, heads, S, head_dim]`.
- First-token slicing for text and image pooling.
- Matrix transpose for image feature matrix in contrastive logits.

Neural network primitives:

- Embedding: text word, token type, text position, vision position.
- Conv2d: vision patch embedding, `kernel_size=stride=patch_size`, `bias=False`.
- Linear with bias: text Q/K/V, text attention output, text MLP, text transformation, vision Q/K/V/out, vision MLP.
- Linear without bias: top-level `visual_projection`, `text_projection`.
- LayerNorm with eps 1e-5: text embeddings, text post-encoder pre_LN, text post-attention/MLP, vision pre/post and per-block norms.
- GELU for text MLP, QuickGELU for vision MLP.
- Residual add, scalar multiply, softmax, dropout as training-only.
- L2 norm implemented as pow(2) -> sum(dim=-1, keepdim=True) -> pow(0.5), then divide.
- `exp(logit_scale)` scalar multiply.

Attention primitives:

- Dense bidirectional self-attention in both branches.
- MHA only; no MQA/GQA.
- Additive attention mask for padded text tokens; vision usually no mask.
- Eager attention order: `q @ k.T * scale`, add mask, softmax in fp32, cast back, dropout, `attn @ v`.
- SDPA/Flash/flex admissible as optimized noncausal alternatives if mask semantics match.

Position/relative ops:

- Learned absolute text positions with pad-aware position IDs.
- Learned vision absolute positions, optional bicubic 2D interpolation for patch positions.
- No RoPE, ALiBi, relative position bias, or convolutional position encoding.

Generation/cache ops:

- Not applicable. No causal mask, KV cache, beam search, logits processor, or token sampling.
- Independently cacheable artifacts are final text/image embeddings before similarity, not autoregressive caches.

Preprocessing-coupled ops:

- XLM-R tokenizer with `<s> $A </s>` single sequence and `<s> A </s> </s> B </s>` pair template.
- Image resize/center-crop/rescale/normalize emits NCHW `pixel_values`.

## 5. Layer/block breakdown

Text embedding:

```text
input_ids [B, S]
position_ids = cumsum(input_ids != pad_id) * (input_ids != pad_id) + pad_id
token_type_ids = zeros/gathered buffer unless provided
x = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
x = LayerNorm(x)
```

Text encoder block, repeated `text_config.num_hidden_layers`:

```text
residual = x
q = Linear(hidden -> hidden, bias=True)(x).view(B, S, heads, head_dim).transpose(1, 2)
k = Linear(hidden -> hidden, bias=True)(x).view(...).transpose(1, 2)
v = Linear(hidden -> hidden, bias=True)(x).view(...).transpose(1, 2)
attn = noncausal_attention(q, k, v, additive_padding_mask)
x = Linear(hidden -> hidden, bias=True)(attn.reshape(B, S, hidden))
x = LayerNorm(x + residual)
residual = x
x = Linear(hidden -> intermediate, bias=True)(x)
x = GELU(x)
x = Linear(intermediate -> hidden, bias=True)(x)
x = LayerNorm(x + residual)
```

Text head:

```text
sequence = pre_LN(sequence)
projection_state = Linear(text_hidden -> text_project_dim, bias=True)(sequence)
pooled_text = projection_state[:, 0]
text_embeds = Linear(text_project_dim -> projection_dim, bias=False)(pooled_text)
```

Vision embedding:

```text
pixel_values [B, 3, H, W]
patch = Conv2d(3 -> vision_hidden, kernel=patch_size, stride=patch_size, bias=False)
patch = patch.flatten(2).transpose(1, 2)  # [B, Gh*Gw, vision_hidden]
class = class_embedding.expand(B, 1, vision_hidden)
x = concat(class, patch, dim=1) + learned/interpolated position embeddings
x = pre_layernorm(x)
```

Vision encoder block, repeated `vision_config.num_hidden_layers`:

```text
residual = x
x_norm = LayerNorm(x)
attn = noncausal_attention(Q/K/V Linear(x_norm), mask=None)
x = residual + Linear(attn)
residual = x
x = LayerNorm(x)
x = Linear(hidden -> intermediate)(x)
x = QuickGELU(x)
x = Linear(intermediate -> hidden)(x)
x = residual + x
```

Vision head:

```text
pooled_image = post_layernorm(last_hidden_state[:, 0, :])
image_embeds = Linear(vision_hidden -> projection_dim, bias=False)(pooled_image)
```

Similarity:

```text
image_embeds = image_embeds / sqrt(sum(image_embeds**2, dim=-1, keepdim=True))
text_embeds = text_embeds / sqrt(sum(text_embeds**2, dim=-1, keepdim=True))
logits_per_text = (text_embeds @ image_embeds.T) * exp(logit_scale)
logits_per_image = logits_per_text.T
```

## 6. Attention requirements

- Type: bidirectional self-attention in both towers.
- Causality: noncausal (`is_causal=False` in text and vision attention modules).
- Heads: MHA, with `head_dim = hidden_size / num_attention_heads`.
- BAAI text: 16 heads, 64 head dim. BAAI/AltCLIP and m9 vision: 16 heads, 64 head dim. m18 vision: 16 heads, 80 head dim.
- Masking: text uses `create_bidirectional_mask` from a 2D attention mask `[B, S]`, producing backend-specific masks; eager form is additive `[B, 1, Q, K]` with 0 for valid and dtype min for masked. Vision passes no mask in normal use.
- Packed/varlen: no source-specific packed sequence metadata. FlashAttention path may internally unpad from mask, but the model ABI is plain padded text.
- Sliding/local/block attention: none.
- Position interaction: attention receives already embedded absolute positions; no Q/K position transform.
- KV cache: none. Cached keys/values are not produced or consumed.
- Fused attention parity detail: eager softmax is computed with `dtype=torch.float32` and cast back to query dtype.

## 7. Position encoding and custom math

Text position IDs:

```python
def altclip_text_position_ids(input_ids, pad_id):
    mask = (input_ids != pad_id).int()
    incremental = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental.long() + pad_id
```

Vision interpolation:

```python
def interpolate_vision_positions(position_weight, height, width, patch_size):
    cls = position_weight[:1]
    patch = position_weight[1:]
    old = int((patch.shape[0]) ** 0.5)
    patch = patch.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(height // patch_size, width // patch_size), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, position_weight.shape[-1])
    return concat(cls.reshape(1, 1, -1), patch, dim=1)
```

QuickGELU:

```python
def quick_gelu(x):
    return x * sigmoid(1.702 * x)
```

L2 vector norm is intentionally expanded for exportability:

```python
norm = pow(sum(pow(x, 2), dim=-1, keepdim=True), 0.5)
```

Precomputable:

- Learned text/vision position tables and class embedding are constants.
- Fixed image-size vision position lookup can be folded into an add constant.
- Interpolated vision position embeddings depend on dynamic image height/width when `interpolate_pos_encoding=True`.
- Text position IDs depend on runtime padding pattern unless caller provides them.

## 8. Preprocessing and input packing

Text:

- Tokenizer class: `XLMRobertaTokenizer` backed by tokenizers Unigram/SentencePiece assets.
- Single sequence template: `<s> tokens </s>`.
- Pair template: `<s> A </s> </s> B </s>`.
- Model inputs: `input_ids`, `attention_mask`; `token_type_ids` is optional and defaults to zeros because `type_vocab_size=1`.
- `model_max_length=512` in snapshots. Text config max position embeddings is 514 to include pad offset/special-token positions.
- Padding token id is 1; position IDs leave pad tokens at position 1.

Image:

- Image processor emits `pixel_values` in NCHW float layout.
- BAAI/AltCLIP and m18: resize shortest edge to 224, center crop 224x224, rescale by 1/255, normalize with CLIP mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`.
- m9 uses an older scalar processor schema for size/crop and omits explicit rescale fields; runtime import should normalize this through Transformers image processor defaults or require a parity check.
- The GPU graph should start from already prepared `pixel_values`; image decoding, resize, crop, and tokenization are CPU/data-pipeline work for first integration.

Dual-encoder output contract:

- Text feature shape `[text_batch, projection_dim]`.
- Image feature shape `[image_batch, projection_dim]`.
- `logits_per_text` shape `[text_batch, image_batch]`.
- `logits_per_image` shape `[image_batch, text_batch]`.
- Text/image embeddings can be cached independently before the final similarity matrix.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear

Source pattern:

```text
Conv2d(C -> H, kernel=P, stride=P, padding=0, groups=1, bias=False)
flatten(2).transpose(1, 2)
```

Replacement:

```text
WindowFlatten_NCHW(PxP, row-major spatial order) -> MatMul(weight_flat.T) -> Reshape [B, Gh*Gw, H]
```

Preconditions:

- `kernel_size == stride == patch_size`
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`
- Input height/width divisible by patch size.
- Preserve NCHW flatten order unless a guarded layout pass rewrites both window extraction and weight layout.

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * patch_size * patch_size)
```

Failure cases: interpolation with non-divisible image sizes, list/tuple config not supported by current source arithmetic, or upstream NHWC without explicit weight/layout rewrite.

Parity test sketch: compare Conv2d patch output after flatten/transpose against WindowFlatten+GEMM for random NCHW images at 224 and tiny 30.

### Rewrite: encoder QKV projections -> packed GEMM

Source pattern:

```text
q = Linear(x); k = Linear(x); v = Linear(x)
```

Replacement:

```text
packed_qkv = GEMM(x, concat(Wq, Wk, Wv), concat(bq, bk, bv))
split order: q, k, v
```

Preconditions:

- Same input tensor and output hidden width.
- Bias present for all three projections.
- Preserve split order `q, k, v`; text module attribute order is query/key/value, vision attribute order in construction is k/v/q but forward computes q/k/v. Use forward split order.

Failure cases: weight aliases or quantized packed formats not present in source; none observed in inspected configs.

Parity test sketch: one block attention input, compare individual projections against packed projection split before head reshape.

### Rewrite: fixed-size vision position add folding

Source pattern:

```text
concat(class_embedding, patch_embeddings) + position_embedding(position_ids)
```

Replacement:

```text
concat(class + cls_pos, patch + patch_pos)
```

Preconditions:

- `interpolate_pos_encoding=False`.
- Runtime image size equals config image size.
- Position IDs are the default contiguous buffer.

Failure cases: dynamic higher-resolution path with bicubic interpolation.

### Rewrite: contrastive normalization + logits

Source pattern:

```text
x / sqrt(sum(x*x, -1, keepdim=True))
y / sqrt(sum(y*y, -1, keepdim=True))
(x @ y.T) * exp(logit_scale)
```

Replacement:

```text
L2Normalize(text) + L2Normalize(image) -> GEMM_RCR -> scalar multiply
```

Preconditions:

- Last-dim feature width equals top-level projection_dim.
- Keep orientation: text rows by image rows gives `logits_per_text`; transpose gives `logits_per_image`.
- Preserve fp32 accumulation policy for norms and GEMM if parity requires it.

Failure cases: zero vectors need source-equivalent divide behavior.

### Layout pass candidate: local NCHW patch region

Source layout is NCHW through image processor and Conv2d. A guarded NHWC/channel-last optimization can be local to patch embedding and vision encoder if:

- Axis-sensitive ops are rewritten: Conv2d input channels axis, flatten spatial axes, transpose into token sequence, LayerNorm over final hidden dim.
- Consumers after patch embedding operate on `[B, tokens, hidden]`, so the layout pass should end before token sequence form.
- Do not reinterpret text `[B, S, H]` as image layout.

## 10. Kernel fusion candidates

Highest priority:

- Encoder LayerNorm + GEMM coverage for text/vision blocks: dominates both towers and maps to existing dense primitives.
- SDPA/Flash-compatible noncausal MHA with additive padding mask: both branches are attention-heavy; no cache complexity.
- Patch Conv2d-to-GEMM rewrite for ViT: simplifies vision input into GEMM-friendly patch projection.
- Contrastive head kernel: L2 normalize plus GEMM_RCR plus scalar multiply is the end-to-end task output.

Medium priority:

- Packed QKV projection GEMM for text and vision.
- Fused bias + GELU / QuickGELU + second GEMM epilogue where profitable.
- First-token pooling plus projection for text/image heads.
- Bicubic position interpolation only if higher-resolution inference is admitted.

Lower priority:

- Training contrastive loss.
- Dropout paths.
- Output attentions/hidden state capture.
- Flex attention backend parity before basic eager/SDPA parity.

## 11. Runtime staging plan

1. Parse AltCLIPConfig and reject unsupported config shapes: non-scalar image/patch sizes, hidden not divisible by heads, decoder/cross-attention flags, unsupported attention backend requests.
2. Load weights for BAAI/AltCLIP and tiny random; materialize both towers independently.
3. Implement text tower parity with padded tokenizer outputs, including pad-aware position IDs and first-token projection.
4. Implement vision tower parity at fixed image size with NCHW patch Conv2d or Conv2d-to-GEMM rewrite.
5. Implement projection/similarity head and validate logits orientation.
6. Add m9/m18 sweep: handle dropout config differences, 1280-wide/32-layer vision, projection_dim 1024.
7. Add optimized SDPA/Flash-style noncausal attention and packed QKV.
8. Optionally admit `interpolate_pos_encoding=True` with a guarded bicubic interpolation implementation.

Initial stubs:

- Return no loss unless `return_loss=True` is explicitly in scope.
- Ignore `output_attentions` and `output_hidden_states` for first runtime target.
- Require preprocessed `pixel_values` and tokenized text inputs.

## 12. Parity and validation plan

- Unit test text position ID generation against source for padded/unpadded sequences.
- Unit test QuickGELU against Transformers activation.
- Unit test L2 normalization expanded math and zero-vector edge behavior.
- Patch embedding rewrite parity: compare source Conv2d+flatten+transpose against lowered patch GEMM.
- Single text layer parity with random inputs and padding mask.
- Single vision layer parity with no mask.
- Full text tower parity on tiny random checkpoint with tokenizer-produced inputs.
- Full vision tower parity on tiny random checkpoint with random/preprocessed image tensor.
- End-to-end logits parity for tiny random and BAAI/AltCLIP: compare `text_embeds`, `image_embeds`, `logits_per_text`, and `logits_per_image`.
- m18 stress parity for `vision_hidden=1280`, `vision_head_dim=80`, `projection_dim=1024`.

Recommended tolerances:

- fp32 eager: `atol=1e-4`, `rtol=1e-4` for tower outputs; logits may need `atol=2e-4` after full depth.
- fp16/bf16 optimized attention: compare features/logits with looser `atol=2e-2`, `rtol=2e-2`, and track cosine similarity separately.

## 13. Performance probes

- CPU preprocessing throughput: tokenizer batch size and image resize/crop throughput separately.
- Text tower throughput by batch and sequence length: 16, 64, 128, 512.
- Vision tower throughput by batch at 224x224 and optional higher resolution.
- Attention backend comparison: eager vs SDPA vs DinoML fused noncausal attention.
- Patch embedding Conv2d vs GEMM rewrite timing.
- Projection/similarity matrix timing for many text labels vs image batch, including cached image/text features.
- End-to-end zero-shot classification throughput with cached text embeddings.
- Memory probe for m18 vision tower activations and temporary attention matrices.

## 14. Skip/defer list

- Training and `return_loss=True` contrastive loss.
- Dropout behavior outside eval mode.
- Gradient checkpointing.
- Autoregressive generation, KV cache, beam search, sampling.
- Cross-attention/decoder modes implied by inherited config fields.
- `output_attentions` and `output_hidden_states` materialization.
- Dynamic higher-resolution image interpolation until fixed-size parity is stable.
- Multi-GPU, tensor parallel, quantized or packed weight formats; none are source-required for inspected checkpoints.

## 15. Final implementation checklist

- [ ] Parse AltCLIPConfig and processor/tokenizer metadata.
- [ ] Reject unsupported non-scalar image_size/patch_size configs for this source basis.
- [ ] Load text, vision, projection, class, position, and logit_scale weights.
- [ ] Implement XLM-R tokenizer data contract or require tokenized inputs.
- [ ] Implement pad-aware text position IDs and zero token type IDs.
- [ ] Implement text embedding + 24-layer bidirectional encoder parity.
- [ ] Implement vision NCHW patch embedding + ViT encoder parity.
- [ ] Implement first-token pooling for text and vision.
- [ ] Implement text transformation and top-level bias-free projections.
- [ ] Implement L2 feature normalization and `exp(logit_scale)` similarity head.
- [ ] Preserve logits orientation: text-by-image and image-by-text.
- [ ] Add Conv2d patch -> GEMM rewrite with strict guards.
- [ ] Add packed QKV rewrite with `q,k,v` split order tests.
- [ ] Add tiny random end-to-end parity test.
- [ ] Add BAAI/AltCLIP fixed-size image/text parity test.
- [ ] Add BAAI/AltCLIP-m18 dimension stress test.
- [ ] Benchmark cached text/image embedding retrieval and full zero-shot scoring.
