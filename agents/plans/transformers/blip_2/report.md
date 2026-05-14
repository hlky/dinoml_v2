# BLIP-2 Transformers Family Audit

Primary target: inference-time multimodal generation with `Blip2ForConditionalGeneration`. The first DinoML path should stage the BLIP-2 vision encoder, Q-Former, language projection, image-token embedding stitch, and delegated language model prefill/decode separately. `Blip2ForImageTextRetrieval` is a useful follow-on because it exercises Q-Former text input and projection/similarity heads.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  blip_2 family; representative checkpoints listed below
Config source:
  Hugging Face config.json, preprocessor_config.json, processor_config.json, tokenizer_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/blip_2/modeling_blip_2.py
  X:/H/transformers/src/transformers/models/blip_2/configuration_blip_2.py
  X:/H/transformers/src/transformers/models/blip_2/processing_blip_2.py
  X:/H/transformers/src/transformers/models/blip/image_processing_blip.py
  X:/H/transformers/src/transformers/models/opt/modeling_opt.py
  X:/H/transformers/src/transformers/models/t5/modeling_t5.py
Any missing files or assumptions:
  No remote-code files are required for the inspected public BLIP-2 checkpoints. BLIP-2 delegates the language model through AutoModelForCausalLM or AutoModelForSeq2SeqLM, so this report covers the BLIP-2 wrapper and the representative OPT/T5 contracts, not every possible text_config family. HF configs were fetched from public model repos on 2026-05-13.
```

Pinned source URLs:

- `modeling_blip_2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip_2/modeling_blip_2.py
- `configuration_blip_2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip_2/configuration_blip_2.py
- `processing_blip_2.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip_2/processing_blip_2.py
- `image_processing_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/image_processing_blip.py
- `modeling_opt.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/opt/modeling_opt.py
- `modeling_t5.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/t5/modeling_t5.py

Representative HF configs fetched:

- `Salesforce/blip2-opt-2.7b`: https://huggingface.co/Salesforce/blip2-opt-2.7b/resolve/main/config.json
- `Salesforce/blip2-opt-6.7b`: https://huggingface.co/Salesforce/blip2-opt-6.7b/resolve/main/config.json
- `Salesforce/blip2-flan-t5-xl`: https://huggingface.co/Salesforce/blip2-flan-t5-xl/resolve/main/config.json
- `Salesforce/blip2-flan-t5-xxl`: https://huggingface.co/Salesforce/blip2-flan-t5-xxl/resolve/main/config.json
- `Salesforce/blip2-itm-vit-g`: https://huggingface.co/Salesforce/blip2-itm-vit-g/resolve/main/config.json
- `hf-internal-testing/tiny-random-Blip2ForConditionalGeneration`: https://huggingface.co/hf-internal-testing/tiny-random-Blip2ForConditionalGeneration/resolve/main/config.json

Processor/tokenizer configs were fetched from the same repos where available.

## 2. High-level architecture

BLIP-2 is a multimodal composition:

```text
image processor -> pixel_values[N,3,H,W]
  -> BLIP-2 ViT-G-style vision encoder -> image tokens[N,image_seq,vision_hidden]
  -> Q-Former query tokens with cross-attention to image tokens -> query states[N,Q,768]
  -> language_projection[768 -> text_hidden]
  -> masked_scatter into <image> token embedding slots
  -> delegated OPT/T5 language model -> logits/generate
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, bicubic resize to configured square size, rescale by `1/255`, CLIP mean/std normalization, tokenizer, and insertion of `num_query_tokens` image placeholder tokens when the processor has `num_query_tokens`.
- Cacheable vision stage: `pixel_values -> vision_model.last_hidden_state`. Common Salesforce checkpoints use image size 224, patch 14, sequence length `1 + 16*16 = 257`, hidden 1408.
- Cacheable Q-Former/projector stage: learned query tokens `[1,Q,768]` expand per batch, attend to image tokens, and project to the language hidden size. This can be validated independently of the LLM.
- Prefix construction: projected query features replace `<image>` token embeddings by `masked_scatter`; count and order are parity-critical.
- Prefill/decode: delegated language model owns autoregressive or encoder-decoder cache semantics. OPT variants are decoder-only; FLAN-T5 variants are encoder-decoder.

Implemented heads:

- Required for primary target: `Blip2ForConditionalGeneration`.
- Optional but close: `Blip2Model` feature helpers and `get_image_features`.
- Deferred/follow-on: `Blip2ForImageTextRetrieval`, `Blip2TextModelWithProjection`, `Blip2VisionModelWithProjection`.

## 3. Important config dimensions

Source defaults from `configuration_blip_2.py`:

| Field | Vision default | Q-Former default | Top/text default | Notes |
| --- | ---: | ---: | ---: | --- |
| `hidden_size` | 1408 | 768 | OPT default if omitted | Salesforce generation configs omit most vision/Q-Former fields and rely on these defaults. |
| `intermediate_size` | 6144 | 3072 | delegated | Vision/Q-Former use GELU FFN. |
| `num_hidden_layers` | 39 | 12 | delegated | Q-Former cross-attn frequency controls which layers see image tokens. |
| `num_attention_heads` | 16 | 12 | delegated | Vision and Q-Former head dim defaults are 88 and 64. |
| `image_size` / `patch_size` | 224 / 14 | n/a | n/a | Source computes `(image_size // patch_size) ** 2 + 1`. |
| `qkv_bias` | true | n/a | delegated | Vision packed QKV has Q and V bias, K bias zero-filled. |
| `cross_attention_frequency` | n/a | 2 | n/a | Q-Former layers `0,2,4,...` add cross-attention over image tokens. |
| `encoder_hidden_size` | n/a | overwritten to vision hidden | n/a | `Blip2Config.__post_init__` forces Q-Former cross-attn K/V input width to vision hidden. |
| `num_query_tokens` | n/a | n/a | 32 | Tiny random uses 10. |
| `image_token_index` | n/a | n/a | checkpoint-specific or `None` | OPT uses 50265, T5 uses 32100, tiny uses 1024. ITM checkpoint omits it. |
| `use_decoder_only_language_model` | n/a | n/a | derived | True when `text_config.model_type` maps to causal LM. |

Representative checkpoint sweep:

| Checkpoint | Head | Vision dims | Q-Former dims | Text backend | Text dims | Image token | Query tokens | Operator-significant notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `hf-internal-testing/tiny-random-Blip2ForConditionalGeneration` | generation | H32/L5/heads4/patch2/image30 | H32/L6/heads4/FFN37 | T5 | d_model32, enc5/dec5, heads4, d_kv8 | 1024 | 10 | Debug odd FFN sizes and small image grid `15*15+1`. |
| `Salesforce/blip2-opt-2.7b` | generation | source defaults H1408/L39/heads16 | source defaults H768/L12/heads12 | OPT causal LM | H2560/L32/heads32/FFN10240 | 50265 | 32 | Projector is `Linear(768 -> 2560)`, decoder-only cache. |
| `Salesforce/blip2-opt-6.7b` | generation | source defaults | source defaults | OPT causal LM | H4096/L32/heads32/FFN16384 | 50265 | 32 | Same BLIP-2 front-end, larger projector/LLM hidden. |
| `Salesforce/blip2-flan-t5-xl` | generation | source defaults | source defaults | T5 seq2seq | d_model2048, enc24/dec24, heads32, d_kv64, gated-gelu FFN5120 | 32100 | 32 | Projected image tokens feed T5 encoder input embeddings. |
| `Salesforce/blip2-flan-t5-xxl` | generation | source defaults | source defaults | T5 seq2seq | d_model4096, enc24/dec24, heads64, d_kv64, gated-gelu FFN10240 | 32100 | 32 | Similar to XL but doubled model width/head count. |
| `Salesforce/blip2-itm-vit-g` | retrieval | source defaults | H768/L12/heads12 with text input | BERT-like Q-Former text path | qformer vocab 30523, `use_qformer_text_input=true` | `None` | 32 | No `processor_config.json`; uses Q-Former token embeddings and projection/similarity heads, not LLM generation. |

## 3a. Family variation traps

- BLIP-2 is a wrapper around a delegated LLM. OPT and T5 variants have different cache, mask, position-bias, and generation contracts.
- Salesforce configs omit most vision/Q-Former dimensions; effective values come from source defaults. Do not infer those fields are absent at runtime.
- `config.qformer_config.encoder_hidden_size` is overwritten to `vision_config.hidden_size`, so cross-attention K/V are `vision_hidden -> 768`.
- `num_query_tokens` must match both the learned query parameter and processor placeholder expansion. A mismatch becomes a `masked_scatter` shape failure.
- `image_token_index` is checkpoint-specific and may be absent for retrieval. Source also accepts the legacy alias `image_token_id`.
- Processor prepends image placeholders before the text tokenizer's normal BOS/special tokens. `max_length` is reduced by `num_query_tokens`.
- `Blip2ForConditionalGeneration.generate` fabricates `[<image> * Q, bos]` when no `input_ids` are provided, but only if `image_token_index` is set.
- Q-Former is kept in fp32 by `_keep_in_fp32_modules`; outputs are downcast to the vision/LLM dtype before projection or scatter.
- Vision input is NCHW. Any NHWC/channel-last optimization must be a guarded vision-local rewrite covering resize/normalize/Conv2d/flatten axis changes.
- Vision position interpolation uses a temporary `[1,D,h,w]` bicubic path and should be a no-layout-translation guard or precomputed per bucket.
- Q-Former does not advertise FlashAttention/SDPA/flex attention support; source eager attention includes attention-map hooks and additive masks.
- T5 FLAN checkpoints use relative attention bias and encoder-decoder caches; OPT checkpoints use learned absolute positions and decoder-only caches.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor ingest and dtype cast to patch embedding weight dtype.
- Conv2d patch embedding: `Conv2d(3 -> 1408, kernel=14, stride=14, padding=0)` for common checkpoints.
- Flatten/transpose: `[N,1408,16,16] -> [N,256,1408]`.
- CLS parameter expand, concat on sequence axis, learned position add.
- Optional bicubic interpolation of position embeddings: sequence -> square grid -> NCHW interpolate -> sequence.
- LayerNorm over last dim, residual adds, reshape/view/permute for attention heads.
- Query token parameter expand `[1,Q,768] -> [N,Q,768]`.
- Attention mask creation/inversion: ones masks, 2D/3D broadcast to `[N,1,1,S]` or `[N,1,Q,S]`, additive `-10000.0`.
- Placeholder mask: `input_ids == image_token_id`, unsqueeze/expand to embedding shape, `masked_scatter`.
- Retrieval path: text slicing to drop image placeholders when `image_token_index` exists, mean over query dimension, transpose similarity.

Neural network primitives:

- Embedding lookup for LLM input tokens; Q-Former text embeddings for ITM.
- Vision packed QKV `Linear(1408 -> 4224)` with Q/V bias and zero K bias, output projection `Linear(1408 -> 1408)`.
- Vision MLP `Linear(1408 -> 6144) -> GELU -> Linear(6144 -> 1408)`.
- Q-Former self-attention Q/K/V/O `Linear(768 -> 768)` with bias.
- Q-Former cross-attention K/V `Linear(1408 -> 768)`, query `Linear(768 -> 768)`.
- Q-Former FFN `Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768)`.
- Language projection `Linear(768 -> text_hidden)`.
- OPT or T5 delegated primitives; for representative targets, OPT uses decoder MHA/FFN and T5 uses RMS-style T5 LayerNorm, relative bias attention, and gated-gelu FFN.
- Retrieval heads: `vision_projection(768 -> 256)`, `text_projection(768 -> 256)`, `itm_head(768 -> 2)`, L2 normalize, matmul similarity, max over query tokens.

Attention primitives:

- Vision noncausal MHA over image sequence, no mask.
- Q-Former noncausal self-attention over query-only or query+text sequence.
- Q-Former cross-attention on query slice only in layers where `layer_idx % cross_attention_frequency == 0`.
- OPT causal self-attention with DynamicCache, learned absolute positions, query scaling before backend call.
- T5 encoder bidirectional self-attention and decoder causal self/cross-attention with relative position bias and EncoderDecoderCache.

Generation/cache ops:

- Independently cacheable image embeddings and projected query embeddings before LLM prefill.
- OPT prefill cache contains the image placeholder positions after they have been replaced by projected query embeddings.
- T5 encoder output is cacheable as the multimodal prefix representation; decoder cache has self-attn K/V and cross-attn K/V to encoder outputs.
- Underlying language model `generate` owns beam search, sampling, logits processors, and cache update behavior.

Preprocessing-coupled ops:

- `BlipImageProcessor`: RGB, bicubic resize, rescale, CLIP mean/std normalize, output channels-first `pixel_values`.
- `Blip2Processor`: adds `<image>` special token if tokenizer lacks one, disables token type IDs, prepends exactly `num_query_tokens` image-token IDs when both image and text are present.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values[N,3,H,W]
patch = Conv2d(3 -> V, kernel=P, stride=P)(pixel_values.to(weight_dtype))
patch = patch.flatten(2).transpose(1,2)         # [N,(H/P)*(W/P),V]
cls = class_embedding.expand(N,1,V)
x = concat([cls, patch], dim=1)
x = x + position_embedding[:, :x.seq, :]
```

Vision block, repeated `vision_layers` times:

```text
residual = x
x = LayerNorm(x, eps=1e-6)
qkv = Linear(V -> 3V)(x).reshape(N,S,3,heads,head_dim).permute(2,0,3,1,4)
x = softmax((q @ k.T) * head_dim**-0.5) @ v
x = Linear(V -> V)(merge_heads(x)) + residual
residual = x
x = LayerNorm(x, eps=1e-6)
x = Linear(V -> I) -> GELU -> Linear(I -> V)
x = x + residual
```

Q-Former layer, repeated 12 times by default:

```text
x = self_attention(x, attention_mask)
if query_length > 0:
  q = x[:, :query_length, :]
  if layer_idx % cross_attention_frequency == 0:
    q = cross_attention(q, encoder_hidden_states=image_embeds)
  q = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> LayerNorm(residual)
  if text tail exists:
    text = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> LayerNorm(residual)
    x = concat([q, text], dim=1)
else:
  x = text FFN path
```

Generation wrapper:

```text
image_embeds = vision_model(pixel_values).last_hidden_state
image_mask = ones([N, image_seq])
query_tokens = learned_query_tokens.expand(N,Q,768)
query_output = qformer(query_tokens, encoder_hidden_states=image_embeds, encoder_attention_mask=image_mask)
projected = language_projection(query_output.to(image_dtype))  # [N,Q,text_hidden]
inputs_embeds = language_model.embed_tokens(input_ids)
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = masked_scatter(inputs_embeds, mask, projected.to(inputs_embeds.dtype))
outputs = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...)
```

## 6. Attention requirements

Vision attention:

- Noncausal full self-attention.
- MHA only; common default `heads=16`, `head_dim=88`.
- No mask in normal path.
- Packed QKV order is all-Q, all-K, all-V along the last linear dimension.
- Source eager math: QK matmul, multiply by scaling, optional additive mask, softmax without fp32 upcast in BLIP-2 eager helper, dropout, AV matmul, output projection.

Q-Former attention:

- Noncausal self-attention over query tokens, or over query+text tokens in ITM mode.
- Cross-attention every `cross_attention_frequency` layers, only for the first `query_length` tokens.
- Default MHA `heads=12`, `head_dim=64`.
- Cross-attention K/V input width is `vision_hidden`, output per-head width is Q-Former head dim.
- Additive masks use `(1 - mask) * -10000.0` cast to Q-Former dtype.
- No KV cache; Q-Former output can be cached as a whole for repeated prompts over the same image.

OPT delegated attention:

- Decoder-only causal MHA; `Salesforce/blip2-opt-2.7b` uses `heads=32`, `head_dim=80`, and `hidden=2560`; 6.7B uses `head_dim=128`.
- Query projection is multiplied by `head_dim**-0.5` before the backend call; backend scaling is `1.0`.
- Cache stores projected K/V after token+position embeddings, shaped `[N, heads, seq, head_dim]`.
- Prefill sequence includes the image placeholder positions after `masked_scatter`, so decode cache already contains image-conditioned prefix K/V.
- Source supports attention backends through the OPT implementation, but parity must preserve query pre-scaling and causal mask construction.

T5 delegated attention:

- Encoder bidirectional self-attention over the stitched multimodal embeddings.
- Decoder causal self-attention plus encoder-decoder cross-attention.
- Relative position bias is computed in the first block and shared through the stack; bucket count is 32 for Salesforce FLAN configs.
- Cache uses `EncoderDecoderCache`: decoder self-attn K/V grow with generated tokens; cross-attn K/V to encoder hidden states are reused after first update.
- Cache K/V shape is `[N, heads, seq, d_kv]`, e.g. XL `[N,32,seq,64]`, XXL `[N,64,seq,64]`.

No BLIP-2 wrapper path requires RoPE, ALiBi, sliding-window attention, GQA, MQA, packed varlen metadata, or `cu_seqlens`.

## 7. Position encoding and custom math

BLIP-2 vision uses learned absolute 2D patch positions flattened into sequence order. Q-Former has no explicit position embeddings for query-only generation mode. In retrieval text mode, `Blip2TextEmbeddings` adds learned token positions before concatenating query embeddings.

Vision interpolation when `interpolate_pos_encoding=True`:

```python
def blip2_interpolate_pos_embed(pos_embed, height, width, patch_size):
    cls = pos_embed[:, :1]
    patch = pos_embed[:, 1:]
    d = patch.shape[-1]
    old = int(patch.shape[1] ** 0.5)
    new_h, new_w = height // patch_size, width // patch_size
    patch = patch.reshape(1, old, old, d).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_h * new_w, d)
    return concat([cls, patch], dim=1)
```

T5 relative position bucket math is delegated to T5:

```python
def t5_relative_bucket(relative_position, bidirectional, num_buckets, max_distance):
    # exact buckets for small distances, logarithmic buckets up to max_distance
    # decoder mode clamps positive relative positions by using -min(relative_position, 0)
    return bucket_ids
```

Precompute opportunities: fixed 224x224 vision position embeddings need no interpolation; interpolated positions can be precomputed by image bucket. Q-Former query tokens are learned constants expanded at runtime. OPT position IDs and T5 relative bias depend on sequence length/cache position.

## 8. Preprocessing and input packing

Image processor contract:

- Source `BlipImageProcessor` defaults to bicubic resize, CLIP mean/std, RGB conversion, rescale, normalize, and channels-first output.
- Inspected Salesforce BLIP-2 processors specify `size: {height:224,width:224}`, `resample:3`, `rescale_factor:0.00392156862745098`, and CLIP mean/std.
- Tiny random uses `size: 30x30`.
- The model destructures `pixel_values` as `[batch, channels, height, width]`.

Text/processor contract:

- `Blip2Processor` forces `tokenizer.return_token_type_ids = False`.
- If tokenizer lacks `image_token`, it adds special token `<image>`.
- When images and text are both present and `num_query_tokens` is set, it tokenizes the normal text, separately tokenizes `"<image>" * num_query_tokens` with no special tokens/padding/truncation, and prepends those IDs to every text sample.
- `max_length` is reduced by `num_query_tokens`, so the final tokenized sequence can still include the image placeholders.
- OPT tokenizer config uses GPT2Tokenizer-style tokens with `<image>` id 50265 and effectively unbounded `model_max_length`.
- FLAN-T5 tokenizer config uses T5Tokenizer with `<image>` id 32100 and `model_max_length=512`.
- ITM checkpoint uses BertTokenizer and has no `processor_config.json`; source retrieval paths may manually drop the first `num_query_tokens` text IDs when `image_token_index` is set.

Multimodal stitch:

- Projected image/query features have shape `[N,Q,text_hidden]`.
- Placeholder mask has shape `[N,S,text_hidden]` after expansion.
- `masked_scatter` writes flattened projected features into placeholder positions in row-major token order. There is no explicit source shape check before scatter, so DinoML should add one for clearer failures.
- Image/Q-Former/projector outputs can be precomputed for repeated prompts over the same image, but final LLM KV caches are prompt-specific.

Generation controller notes:

- `Blip2ForConditionalGeneration.generate` recomputes vision and Q-Former outputs before calling `language_model.generate`.
- If `input_ids` is absent, it creates `[image_token_index] * num_query_tokens + [bos_token_id]` for every image.
- For decoder-only LMs, it passes both `inputs_embeds` and `input_ids` into `language_model.generate`; for encoder-decoder LMs, it omits `input_ids` and passes `inputs_embeds` as encoder inputs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input is NCHW and `height >= patch_size`, `width >= patch_size`.
- For exact source parity, output grid uses Conv2d floor semantics; common 224/14 and tiny 30/2 are divisible.

Replacement:

```text
WindowFlatten_NCHW(pixel_values, P, P) -> GEMM(weight_flat.T) -> BiasAdd -> [N,grid_h*grid_w,V]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
b = conv.bias
```

Failure cases: non-divisible dynamic images without floor-grid handling, interpolated position embeddings not bucketed, NHWC input without matching patch flatten/weight transform.

Parity test sketch: compare Conv2d+flatten+transpose with rewritten GEMM for 224/14 and 30/2, fp32 and fp16.

### Rewrite: Q-Former cross-attention K/V precompute

Source pattern: in layers `0,2,4,...`, K/V are projected from fixed `image_embeds`.

Replacement: precompute per-cross-attention-layer image K/V once after vision output and reuse for all query tokens.

Preconditions: image embeddings immutable, Q-Former weights fixed, no attention output recording requiring source module hooks.

Failure cases: training, attention map gradient hooks, dynamic image embeddings inside one request.

### Rewrite: placeholder masked_scatter -> indexed copy

Source pattern:

```text
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = masked_scatter(inputs_embeds, mask, projected_image_features)
```

Replacement:

```text
positions = nonzero(input_ids == image_token_id)
copy projected.reshape(N*Q,H) into inputs_embeds[positions]
```

Preconditions: each batch row has exactly `Q` placeholders and placeholder order matches projected query order.

Failure cases: `inputs_embeds`-only placeholder detection by embedding equality, mixed rows with incorrect placeholder count, missing `image_token_index`.

### Rewrite: cacheable multimodal prefix

Source pattern: vision -> Q-Former -> projection runs on every `generate` call.

Replacement: expose a cacheable subgraph returning `[N,Q,text_hidden]`, then run LLM prefill with text IDs and image-token positions.

Preconditions: same image tensor, same preprocessing, same model weights, same dtype policy.

Failure cases: generation APIs that expect hidden states/attentions from the vision/Q-Former path in the same output object.

### Layout rewrite: guarded NCHW -> token sequence vision island

Candidate region: image normalization and patch embedding. Axis-sensitive rewrites:

- Channel normalization axis `C=1` in NCHW becomes last channel in NHWC.
- Conv2d patch flatten order must preserve PyTorch `[out,in,kh,kw]`.
- Source `flatten(2).transpose(1,2)` can be eliminated if the patch GEMM emits `[N,grid_h*grid_w,V]`.
- Position interpolation should remain source NCHW bicubic or be precomputed outside the translated region.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d patch embed lowered to GEMM: fixed non-overlap patch projection is present in every path.
- Q-Former attention and cross-attention: small but unavoidable bridge; cross-attn K/V precompute removes repeated image K/V projection.
- Placeholder indexed copy: tiny operation, but central to generation parity and useful for clear runtime checks.
- Delegated LLM attention/cache kernels: OPT/T5 prefill/decode dominate total latency after the image prefix.

Medium priority:

- Vision packed QKV + attention for sequence length 257 and hidden 1408.
- Language projection `Linear(768 -> text_hidden)` fused with dtype cast where safe.
- T5 relative-position bias generation/cache and gated-gelu FFN for FLAN variants.
- Last-token-only logits in decoder-only OPT generation.

Lower priority:

- Bicubic vision position interpolation on GPU; precompute fixed buckets first.
- Retrieval projection normalization/similarity fusion.
- Output hidden-state/attention recording.

## 11. Runtime staging plan

1. Parse `Blip2Config`, instantiate source-defaulted vision/Q-Former configs, and instantiate delegated OPT or T5 config.
2. Load weights for tiny random and one production OPT checkpoint; preserve LLM embedding/LM-head aliasing according to delegated family rules.
3. Validate image processor boundary with recorded `pixel_values`, then implement vision patch embedding plus one vision block.
4. Run full vision encoder parity for `[N,3,224,224] -> [N,257,1408]`.
5. Implement Q-Former query-only path with cross-attention to vision tokens; validate one layer and full 12-layer output.
6. Implement language projection and placeholder stitch with explicit count checks.
7. For OPT: run multimodal prefill logits, then cached decode token parity.
8. For T5: run encoder from stitched embeddings, decoder prefill/decode with self and cross cache parity.
9. Add `generate` controller parity for image-only prompt and text prompt.
10. Add retrieval head staging after generation path is stable.
11. Enable guarded rewrites/fusions: patch GEMM, indexed stitch, Q-Former K/V precompute, optimized LLM attention.

Initially stub/defer losses, gradient checkpointing, output attentions/hidden states, beam/sampling policy beyond greedy parity, accelerate device-map hacks, quantization, and arbitrary-resolution interpolation.

## 12. Parity and validation plan

- Processor parity: compare HF processor `pixel_values`, `input_ids`, and `attention_mask` for OPT, T5, tiny, and ITM processors; assert exactly `num_query_tokens` image placeholders.
- Patch embed parity: Conv2d+flatten+transpose vs DinoML lowering for 224/14 and 30/2.
- Vision parity: one block, then full encoder, checking CLS token and all patch tokens.
- Q-Former parity: query-only self-attn, cross-attn layer with random image embeddings, then full Q-Former with real vision output.
- Stitch parity: random `input_ids` with image placeholders; verify indexed-copy rewrite matches `masked_scatter` and fails clearly on count mismatch.
- OPT prefill parity: logits for `Salesforce/blip2-opt-2.7b` with image-only and prompted inputs.
- OPT decode parity: one-token and multi-token cached decode vs full recompute; cache shape `[N,32,total_seq,80]` for 2.7B.
- T5 parity: encoder output from stitched embeddings; decoder logits and cache behavior for FLAN-T5 XL.
- Retrieval parity: ITM head logits, contrastive similarity orientation, query-token max over image features.

Recommended tolerances: fp32 isolated ops `rtol=1e-4, atol=1e-5`; fp16/bf16 full-block checks `rtol=1e-2, atol=1e-2`; greedy generated token parity should be exact when using identical generation settings.

## 13. Performance probes

- CPU preprocessing throughput: resize/rescale/normalize and tokenizer placeholder insertion.
- Vision encoder throughput for batch sizes 1/4/16 at 224x224.
- Q-Former bridge throughput split into self-attn, cross-attn, and FFN.
- Language projection and placeholder stitch latency.
- OPT prefill tokens/sec with sequence length `Q + prompt_len`.
- OPT decode tokens/sec and KV memory: `layers * 2 * batch * heads * seq * head_dim * dtype_size`.
- T5 encoder throughput on stitched embeddings and decoder tokens/sec with cross-attention cache.
- End-to-end split: preprocessing, vision, Q-Former/projector, prefill, decode.
- Cacheable image-prefix reuse: repeated prompts over one image with and without re-running vision/Q-Former.
- Attention backend comparison for delegated OPT/T5 paths.

No DinoML tests or benchmarks were run for this docs-only audit.

## 14. Skip/defer list

- Training losses, gradient checkpointing, gradient attention hooks, and output recording.
- Beam search, nucleus sampling, suppress-token processors, and advanced generation controllers beyond first greedy parity.
- Accelerate multi-device placement hacks and tensor parallelism.
- Quantization/int8/bitsandbytes paths.
- Arbitrary image sizes and GPU bicubic position interpolation.
- `inputs_embeds`-only image placeholder detection by embedding equality.
- Remote-code or nonstandard text backends not represented by OPT/T5 until separately audited.
- Tokenizer execution inside DinoML runtime; keep tokenization and image preprocessing in the CPU/data pipeline first.

## 15. Final implementation checklist

- [ ] Parse `Blip2Config`, source-defaulted `Blip2VisionConfig`, `Blip2QFormerConfig`, and delegated `text_config`.
- [ ] Load vision, Q-Former, language projection, and delegated LLM weights.
- [ ] Preserve delegated LLM tied-weight aliases where applicable.
- [ ] Implement or fixture the BLIP image processor tensor contract.
- [ ] Implement processor placeholder count checks for `<image>` tokens.
- [ ] Implement NCHW Conv2d patch embedding or guarded patch-GEMM rewrite.
- [ ] Implement BLIP-2 vision encoder block and final post layer norm.
- [ ] Implement Q-Former self-attention, periodic image cross-attention, and query/text FFN split.
- [ ] Implement fp32 Q-Former island with explicit output cast.
- [ ] Implement `language_projection(768 -> text_hidden)`.
- [ ] Implement image-placeholder indexed copy equivalent to `masked_scatter`.
- [ ] Implement OPT multimodal prefill and cached decode path.
- [ ] Implement T5 multimodal encoder plus decoder self/cross cache path.
- [ ] Add parity tests for processor, patch embed, vision, Q-Former, stitch, OPT prefill/decode, and T5 encoder-decoder.
- [ ] Add retrieval head parity for ITM/projection as a follow-on.
- [ ] Benchmark preprocessing, vision, Q-Former/projector, stitch, prefill, decode, and cache memory separately.
