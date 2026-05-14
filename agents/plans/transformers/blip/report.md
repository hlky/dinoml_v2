# BLIP Transformers Family Audit

Primary target: multimodal BLIP inference for image captioning, visual question answering, and image-text retrieval. The first DinoML integration should prioritize the shared vision encoder plus the text decoder cross-attention path used by `BlipForConditionalGeneration`; VQA and ITM are close follow-ons.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  blip family; representative checkpoints listed below
Config source:
  Hugging Face config.json, preprocessor_config.json, tokenizer_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/blip/modeling_blip.py
  X:/H/transformers/src/transformers/models/blip/modeling_blip_text.py
  X:/H/transformers/src/transformers/models/blip/configuration_blip.py
  X:/H/transformers/src/transformers/models/blip/processing_blip.py
  X:/H/transformers/src/transformers/models/blip/image_processing_blip.py
  X:/H/transformers/src/transformers/image_processing_backends.py
  X:/H/transformers/tests/models/blip/test_modeling_blip.py
Any missing files or assumptions:
  No remote-code files are required for the inspected Salesforce and internal-testing BLIP checkpoints. Local source is authoritative. Official HF configs were fetched from public model repos on 2026-05-13.
```

Pinned source URLs:

- `modeling_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/modeling_blip.py
- `modeling_blip_text.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/modeling_blip_text.py
- `configuration_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/configuration_blip.py
- `processing_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/processing_blip.py
- `image_processing_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/image_processing_blip.py

Representative HF configs fetched:

- `hf-internal-testing/tiny-random-BlipModel`: https://huggingface.co/hf-internal-testing/tiny-random-BlipModel/resolve/main/config.json
- `Salesforce/blip-image-captioning-base`: https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/config.json
- `Salesforce/blip-image-captioning-large`: https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/config.json
- `Salesforce/blip-vqa-base`: https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/config.json
- `Salesforce/blip-itm-base-coco`: https://huggingface.co/Salesforce/blip-itm-base-coco/resolve/main/config.json

Processor/tokenizer configs were fetched from the same repos using `preprocessor_config.json` and `tokenizer_config.json`.

## 2. High-level architecture

BLIP is a multimodal family with a ViT-style vision encoder and BERT/MED-style text modules. The implemented heads are:

- `BlipForConditionalGeneration`: vision encoder plus causal text decoder with cross-attention to image tokens; required for captioning.
- `BlipForQuestionAnswering`: vision encoder, question text encoder with image cross-attention, then answer text decoder with cross-attention to question embeddings; optional after captioning but architecturally important.
- `BlipForImageTextRetrieval`: vision encoder plus text encoder; either image-text matching classifier over fused CLS or normalized projection similarity; optional for first captioning target.
- Deprecated `BlipModel`: dual encoder/projection contrastive path plus a cross-attended multimodal feature helper; useful as a retrieval/projection reference.

Dataflow:

```text
image -> processor -> pixel_values[N,3,H,W]
      -> Conv2d patch embed -> CLS + learned pos -> pre-norm ViT encoder -> image tokens

captioning:
prompt tokens -> token+position+LN -> causal decoder self-attn + image cross-attn -> LM head -> logits/generate

VQA:
question tokens + image tokens -> cross-attended question encoder
answer BOS token(s) + question embeddings -> causal decoder -> answer logits/generate

ITM/retrieval:
text tokens + optional image cross-attn -> CLS pool/projection or ITM classifier -> score
```

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize to square target, optional pad, rescale, CLIP mean/std normalization, BERT tokenization with no token type IDs by processor default.
- Independently cacheable vision stage: `pixel_values[N,3,H,W] -> image_embeds[N, 1 + H/P * W/P, vision_hidden]`. This is shared by captioning, VQA, and ITM.
- Captioning prefill/decode: text decoder cross-attends to fixed image tokens. Cross-attention K/V can be cached after the first decode step.
- VQA staging: question encoder cross-attends to image tokens and can be validated separately from answer decoding.
- Retrieval staging: image projection and text projection/ITM head can be run independently after branch encoders.

## 3. Important config dimensions

Source defaults from `configuration_blip.py`:

| Field | Text default | Vision default | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 30524 | n/a | BERT tokenizer vocabulary. |
| `hidden_size` | 768 | 768 | Text `encoder_hidden_size` is overwritten to vision hidden size in `BlipConfig.__post_init__`. |
| `intermediate_size` | 3072 | 3072 | GELU FFN. |
| `num_hidden_layers` | 12 | 12 | Text layers include cross-attention when `is_decoder=True`. |
| `num_attention_heads` | 8 default in source text config, 12 in common checkpoints | 12 | Head dim is `hidden_size / heads`; checkpoint values matter. |
| `max_position_embeddings` | 512 | n/a | Learned absolute positions. |
| `image_size` | n/a | 384 | Square defaults; list/tuple accepted by config type but source assumes scalar division in patch count. |
| `patch_size` | n/a | 16 | Non-overlapping Conv2d patchify. |
| `projection_dim` | 768 text config, 512 top-level/common | 512 vision config | `BlipModel` projection uses top-level `projection_dim`; ITM uses `image_text_hidden_size`. |
| `image_text_hidden_size` | n/a | n/a | Top-level default 256 for ITM projection heads. |
| `logit_scale_init_value` | n/a | n/a | Top-level default 2.6592 for deprecated contrastive `BlipModel`. |
| `hidden_act` | gelu | gelu | Plain FFN, not gated. |
| `layer_norm_eps` | 1e-12 | 1e-5 | Text and vision differ. |
| `is_decoder` / `use_cache` | true / true | n/a | Text cache is used only when called as decoder/generation. |

Representative checkpoint sweep:

| Checkpoint | Architecture | Text H/L/heads/FFN | Vision H/L/heads/FFN | Image/Patch | Vocab | Operator-significant notes |
| --- | --- | --- | --- | --- | ---: | --- |
| `hf-internal-testing/tiny-random-BlipModel` | `BlipModel` | 32 / 5 / 4 / 37 | 32 / 5 / 4 / 37 | 30 / 2 | 1124 | Debug-size dual/projection model; odd FFN size catches shape assumptions. |
| `Salesforce/blip-image-captioning-base` | `BlipForConditionalGeneration` | 768 / 12 / 12 / 3072 | 768 / 12 / 12 / 3072 | 384 / 16 | 30524 | Main captioning target; image seq len 577. |
| `Salesforce/blip-image-captioning-large` | `BlipForConditionalGeneration` | 768 / 12 / 12 / 3072 | 1024 / 24 / 16 / 4096 | 384 / 16 | 30524 | Vision hidden differs from text hidden; cross-attn K/V project 1024 -> 768. |
| `Salesforce/blip-vqa-base` | `BlipForQuestionAnswering` | 768 / 12 / 12 / 3072 | 768 / 12 / 12 / 3072 | 384 / 16 | 30524 | Adds question encoder plus answer decoder. |
| `Salesforce/blip-itm-base-coco` | `BlipForImageTextRetrieval` | 768 / 12 / 12 / 3072 | 768 / 12 / 12 / 3072 | 384 / 16 | 30524 | ITM classifier `[H -> 2]` and optional projection similarity `[H -> 256]`. |

For 384x384, patch size 16 produces `24 * 24 = 576` patch tokens plus one CLS token.

## 3a. Family variation traps

- Text source defaults say 8 heads, but common checkpoints use 12 heads. Use checkpoint config, not class defaults.
- Captioning-large has vision hidden 1024 while text hidden remains 768. Cross-attention key/value weights are shaped `[text_hidden, vision_hidden]`; do not assume shared hidden size.
- Text `is_decoder=True` creates a cross-attention module in every text layer even when a call uses text-only self-attention.
- `BlipTextModel.forward` has an `is_decoder` argument defaulting to `False`; generation paths pass `is_decoder=True` through `BlipTextLMHeadModel`. Calls without decoder mode disable cache and use noncausal masks.
- Vision encoder is pre-norm residual; text encoder/decoder is BERT-style post-norm residual with an embedding LayerNorm. Fusion opportunities differ.
- `BlipVisionModel.forward` applies the same `post_layernorm` to the full sequence and then again to `last_hidden_state[:,0,:]`; this is source behavior and should be preserved.
- `BlipForConditionalGeneration.generate` mutates prompt handling: if no prompt is provided it creates `[bos, eos]`, forces `input_ids[:,0] = bos`, passes `input_ids[:, :-1]`, and sets generation `eos_token_id` to `sep_token_id` rather than config `eos_token_id`.
- Processor disables `token_type_ids`; tokenizer configs are BERT tokenizers but segment IDs are not a model input through `BlipProcessor`.
- Image processor outputs NCHW `pixel_values`. Any NHWC/channel-last optimization must be a guarded local vision-region rewrite with Conv2d and flatten/transpose axis rewrites.
- Positional interpolation path uses NHWC temporary layout for patch position embeddings: `[1, S, D] -> [1,h,w,D] -> NCHW for bicubic interpolate -> back to sequence`.
- `BlipModel` is deprecated in the source warning, but its projection/logit scale path remains useful for retrieval parity.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor ingest, dtype cast to patch weight dtype.
- Conv2d patch embedding: `Conv2d(3 -> vision_hidden, kernel=patch_size, stride=patch_size, padding=0)`.
- Flatten spatial dims then transpose: `[N,D,H/P,W/P] -> [N,S,D]`.
- CLS parameter expand, concat on sequence axis, learned position add.
- Sequence slicing/indexing for CLS (`[:,0,:]`), prompt truncation (`input_ids[:, :-1]`), `logits_to_keep` slicing, transpose for similarity matrices.
- Reshape/view/permute for attention heads, cache tensors, and interpolation.
- Optional bicubic resize for position embeddings when `interpolate_pos_encoding=True`.

Neural network primitives:

- Embedding lookup for token IDs and absolute positions.
- LayerNorm with separate eps values: text `1e-12`, vision `1e-5`.
- Linear with bias for Q/K/V in text, FFNs, output projections, pooler, LM transform/head, ITM head, ITM projections.
- Linear without bias for `BlipModel.visual_projection` and `BlipModel.text_projection`.
- GELU, tanh pooler, dropout as no-op for inference, residual add.
- L2 normalize over last dim, scalar `exp(logit_scale)`, matmul similarity.
- LM head: dense -> GELU -> LayerNorm -> tied decoder linear plus bias. Tied output weight aliases token embedding.

Attention primitives:

- Vision self-attention: packed QKV `Linear(D -> 3D)`, reshape to `[3,N,H,S,head_dim]`, noncausal softmax attention, output projection.
- Text self-attention: separate Q/K/V linears, optional causal+padding additive mask, cache update, output dense + dropout + residual + LayerNorm.
- Text cross-attention: Q from decoder/question hidden, K/V from encoder hidden with `encoder_hidden_size -> all_head_size`, additive encoder mask, cross-attn cache reuse.
- No RoPE, ALiBi, relative bias, sliding-window, GQA, or MQA.

Generation/cache ops:

- HF `DynamicCache` and `EncoderDecoderCache` shape contract per layer: self-attn keys/values `[batch, heads, decoded_len, head_dim]`; cross-attn keys/values `[batch, heads, encoder_seq, head_dim]`.
- Cache conversion from self-attention-only `DynamicCache` into `EncoderDecoderCache` when `use_cache=True`.
- Cross-attention K/V `is_updated[layer_idx]` flag to reuse image/question K/V after first generated token.
- Generation controller quirks: BOS forcing, SEP as EOS, optional prompt slicing, `logits_to_keep` last-token optimization.

Preprocessing-coupled ops:

- Image processor: RGB, resize to square size, optional padding in some configs, rescale by `1/255`, normalize with CLIP mean/std, output NCHW `pixel_values`.
- Text processor: BERT tokenizer, lower-case true in inspected configs, no `token_type_ids`, default no padding unless requested.
- Runtime-created image attention masks: all ones with shape `[N, image_seq]`.

## 5. Layer/block breakdown

Vision patch embedding:

```text
pixel_values[N,3,H,W]
patch = Conv2d(3 -> V, kernel=P, stride=P)(pixel_values)  # [N,V,H/P,W/P]
patch = flatten(2).transpose(1,2)                         # [N,S,V]
tokens = concat(class_embedding.expand(N,1,V), patch, dim=1)
tokens = tokens + position_embedding[:, :S+1, :]
```

Vision encoder layer, repeated `vision_layers` times:

```text
residual = x
x = LayerNorm(x, eps=vision_eps)
qkv = Linear(V -> 3V)(x).reshape(N,S,3,Hd,head_dim).permute(2,0,3,1,4)
x = softmax((q @ k.T) / sqrt(head_dim)) @ v
x = Linear(V -> V)(merge_heads(x)) + residual
residual = x
x = LayerNorm(x, eps=vision_eps)
x = Linear(V -> I) -> GELU -> Linear(I -> V)
x = x + residual
```

Vision output:

```text
last_hidden_state = post_layernorm(sequence)
pooler_output = post_layernorm(last_hidden_state[:, 0, :])
```

Text embedding:

```text
inputs = word_embedding(input_ids) + position_embedding(position_ids[past_len:past_len+S])
inputs = LayerNorm(inputs, eps=text_eps)
```

Text layer, repeated `text_layers` times:

```text
self_attn = Q(hidden), K(hidden/cache), V(hidden/cache)
self_attn = softmax((q @ k.T) / sqrt(head_dim) + attention_mask) @ v
x = LayerNorm(Linear(self_attn) + hidden)

if encoder_hidden_states is not None:
  cross = Q(x), K(encoder_hidden/cache), V(encoder_hidden/cache)
  cross = softmax((q @ k.T) / sqrt(head_dim) + encoder_mask) @ v
  x = LayerNorm(Linear(cross) + x)

ffn = Linear(H -> I) -> GELU -> Linear(I -> H)
x = LayerNorm(ffn + x)
```

LM head:

```text
hidden[:, selected_positions, :]
-> Linear(H -> H) -> GELU -> LayerNorm -> Linear(H -> vocab)
```

The LM decoder weight is tied to `bert.embeddings.word_embeddings.weight`; keep it one logical parameter.

## 6. Attention requirements

Vision attention:

- Noncausal full self-attention.
- MHA only; no GQA/MQA. Base: 12 heads x 64 dim. Large vision: 16 heads x 64 dim.
- No attention mask in normal vision path.
- Source eager math order: QK matmul, multiply by `head_dim**-0.5`, softmax, dropout, AV matmul, output projection.
- FlashAttention/SDPA-compatible for inference if the exact scaling and no-mask path are preserved.

Text self-attention:

- MHA with separate Q/K/V linears.
- Encoder-like calls (`is_decoder=False`) use padding masks only and disable cache.
- Decoder calls (`is_decoder=True`) combine causal mask and padding mask. Additive mask values are `(1 - mask) * -10000.0` cast to model dtype.
- Cache stores keys after linear projection and head transpose. No position encoding is applied beyond absolute embedding before projection.
- Cached self-attn shapes are `[N, heads, past_or_total_len, head_dim]`; new decode query is usually `[N, heads, 1, head_dim]`.

Text cross-attention:

- Present in every text layer when config `is_decoder=True`, but active only when `encoder_hidden_states` is passed.
- K/V source hidden size is `config.encoder_hidden_size`, which is set from vision hidden size in `BlipConfig.__post_init__`.
- Captioning cross-attn K/V shape before cache: `[N, heads, image_seq, head_dim]`.
- VQA answer decoder cross-attends to question encoder outputs, so K/V shape is `[N, heads, question_seq, head_dim]`.
- `EncoderDecoderCache.is_updated[layer_idx]` controls reuse of cross-attention K/V after the first generated ID.

Packed/varlen, sliding-window/local attention, ALiBi, RoPE, and relative bias are not required.

## 7. Position encoding and custom math

BLIP uses learned absolute text positions and learned absolute vision position embeddings. No rotary or relative position math is present.

Vision position interpolation, used only when `interpolate_pos_encoding=True`:

```python
def blip_interpolate_pos_embed(pos_embed, height, width, patch_size):
    cls = pos_embed[:, :1]
    patch = pos_embed[:, 1:]
    d = patch.shape[-1]
    old = int((patch.shape[1]) ** 0.5)
    new_h, new_w = height // patch_size, width // patch_size
    patch = patch.reshape(1, old, old, d).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(patch, size=(new_h, new_w), align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_h * new_w, d)
    return concat([cls, patch], dim=1)
```

Precompute/freeze opportunities:

- Standard 384x384 checkpoints can use the learned vision `position_embedding` directly.
- Interpolated position embeddings can be precomputed per image resolution if dynamic image sizes are bucketed.
- Text `position_ids` are deterministic from `past_key_values_length` and sequence length.

## 8. Preprocessing and input packing

Image processor contract:

- `BlipImageProcessor` defaults from source: bicubic resize, CLIP mean `[0.48145466, 0.4578275, 0.40821073]`, CLIP std `[0.26862954, 0.26130258, 0.27577711]`, square size 384, RGB conversion, resize, rescale, normalize.
- Fetched Salesforce configs mostly specify `size: {height:384,width:384}`, `resample: 3`, `rescale_factor: 1/255`, and `do_pad: true` for large/VQA/ITM. Base captioning config omits `do_rescale`/`rescale_factor` but source defaults supply them.
- Processor output is `pixel_values[N,3,H,W]`, channels first. The model code destructures `batch_size, _, height, width = pixel_values.shape`.

Text processor contract:

- `BlipProcessor` wraps a BERT tokenizer and forces `tokenizer.return_token_type_ids = False`.
- Defaults: add special tokens, no padding unless requested, no token type IDs, no offsets/lengths. Inspected tokenizer configs use `BertTokenizer`, `do_lower_case=True`, `model_max_length=512`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`.
- Config token IDs: `bos_token_id=30522`, `sep_token_id=102`, `pad_token_id=0`, `eos_token_id=2`. Generation uses `sep_token_id` as EOS for stopping.

Multimodal stitching:

- BLIP does not scatter image embeddings into placeholder text tokens. It passes image/question embeddings as `encoder_hidden_states` to text cross-attention.
- Runtime-created encoder masks are all-ones integer tensors matching encoder token count; no grid metadata or `cu_seqlens` are used.
- Image encoder outputs can be precomputed and cached for repeated caption prompts or ITM texts.

Generation controller behavior:

- Captioning without prompt creates `[BOS, EOS]` for each image, forces first token to BOS, decodes from all but the final prompt token, and stops on SEP.
- VQA generation first encodes question+image, creates one BOS token per question, then decodes answer with SEP as EOS.
- Beam search/sampling processors are inherited from HF `GenerationMixin` and can be staged after greedy parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> GEMM

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input is NCHW and `H % patch_size == 0`, `W % patch_size == 0`.
- Patch flatten order matches PyTorch Conv2d NCHW storage.

Replacement:

```text
WindowFlatten_NCHW(pixel_values, P, P) -> GEMM(weight_flat.T) -> BiasAdd -> Reshape[N,S,V]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
b = conv.bias
```

Failure cases: dynamic image sizes without divisibility guards, interpolated position embeddings not bucketed, NHWC-translated input without matching window flatten rewrite.

Parity sketch: compare Conv2d patch output before flatten against flattened-GEMM output for base and large weights, fp32 and fp16.

### Rewrite: vision QKV packed projection

Source pattern:

```text
Linear(V -> 3V) -> reshape(N,S,3,heads,head_dim) -> permute(2,0,3,1,4)
```

Replacement: one GEMM with packed output, then either fused split/head view or direct attention backend consuming packed QKV.

Preconditions: packed Q/K/V order is all-Q, all-K, all-V from the linear output split along the last dimension; no bias transform beyond normal linear bias.

### Rewrite: text cross-attention K/V precompute

Source pattern: every decode step can recompute or reuse cross-attn K/V through `EncoderDecoderCache`.

Replacement: precompute per-layer cross-attention K/V for fixed image/question encoder states during prefill.

Preconditions: encoder hidden states are immutable for the generated sequence; `is_updated[layer]` is true after first update; batch and beam expansion rules are handled by generation.

### Rewrite: last-token-only LM logits

Source pattern: `logits_to_keep` slices hidden states before the LM head.

Replacement: for decode, run LM transform/head only on `[N,1,H]`.

Preconditions: no loss computation; caller requests only next-token logits. Failure case: training or full-sequence logit return.

### Layout rewrite: local NCHW -> NHWC vision island

Candidate optimized region: image patch embed through vision encoder GEMMs/attention. Required axis rewrites:

- Conv2d input NCHW to NHWC only if patch extraction and weight layout are transformed.
- Flatten/transpose source `[N,V,Hp,Wp] -> [N,Hp*Wp,V]` becomes simpler if patch projection emits `[N,Hp,Wp,V]`.
- LayerNorm, attention, MLP already operate over last hidden dim and are layout-neutral once tokens are `[N,S,V]`.

Guard: image processor/model boundary and position interpolation should remain source-faithful unless the entire local region is translated. Position interpolation uses NCHW bicubic internally and should be protected by a no-layout-translation guard or rewritten as a separate precompute.

## 10. Kernel fusion candidates

Highest priority:

- Conv2d patch embed lowered to GEMM: first operator in every path; large fixed patch size makes this easy to validate.
- LayerNorm + QKV/linear preparation: both text and vision are norm-heavy; text uses post-norm residual, vision uses pre-norm, so keep separate fusion templates.
- Attention prefill/decode with cross-attention cache: captioning performance depends on image cross-attn and causal self-attn cache.
- LM head last-token path: dense -> GELU -> LayerNorm -> vocab GEMM can dominate decode when vocab is 30524.

Medium priority:

- Vision QKV packed attention kernel for image seq 577 and large seq 577 x hidden 1024.
- Cross-attention K/V preprojection cache for image tokens in captioning and question embeddings in VQA.
- GELU MLP fusion around two GEMMs where epilogue fusion is available.
- L2 normalize + similarity matmul + logit scale for retrieval.

Lower priority:

- Bicubic position interpolation on GPU; precompute per resolution first.
- ITM classifier fusion; small head, not a first bottleneck.
- Full generation controller parity beyond greedy/beam basics.

## 11. Runtime staging plan

1. Parse BLIP config and load weights, preserving tied LM output/input embeddings and text/vision hidden-size asymmetry.
2. Implement/validate vision patch embedding plus one vision block for base shape `[N,3,384,384] -> [N,577,768]`.
3. Run full vision encoder parity and cache `image_embeds`.
4. Implement text decoder self-attention, cross-attention, and LM head for captioning prefill without cache.
5. Add decode with self-attention KV cache and cross-attention K/V reuse; match `generate` greedy output for captioning.
6. Add VQA question encoder cross-attending to image tokens, then answer decoder.
7. Add ITM/retrieval heads: classifier path and projection similarity path.
8. Add guarded optimizations: patch Conv2d->GEMM, attention kernels, cross-attn preprojection, last-token LM head.

Initially stub/defer losses, dropout, output hidden/attention recording, beam search internals, and interpolation for non-384 image sizes.

## 12. Parity and validation plan

- Processor parity: compare `pixel_values` for the HF test image and a synthetic RGB image against HF processor; check NCHW shape and normalized channel statistics.
- Patch embed parity: Conv2d path vs DinoML lowering for base and large, fp32 tolerance `1e-5`, fp16 tolerance `1e-2`.
- Vision block parity: one block, then full encoder, checking CLS and all tokens.
- Text block parity: self-attn-only, cross-attn with random encoder states, and causal mask with padding.
- Cache parity: decode one token at a time with `use_cache=True` vs full prefill logits for the same sequence.
- Captioning E2E: HF integration expected tokens for `Salesforce/blip-image-captioning-base` image-only prompt are `[30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]`.
- Captioning with prompt `"a picture of"`: expected tokens `[30522, 1037, 3861, 1997, 1037, 2450, 1998, 2014, 3899, 2006, 1996, 3509, 102]`.
- VQA E2E: `Salesforce/blip-vqa-base`, question `"how many dogs are in the picture?"`, expected `[30522, 1015, 102]`.
- ITM E2E: `Salesforce/blip-itm-base-coco`, expected softmax near `[[0.0029, 0.9971]]` for ITM head and similarity near `[[0.5162]]`.

Recommended tolerances: fp32 hidden-state checks `rtol=1e-4, atol=1e-5`; fp16/BF16 block checks `rtol=1e-2, atol=1e-2`; generated token parity should be exact under greedy decoding.

## 13. Performance probes

- Image preprocessing throughput: CPU/PIL vs torchvision backend; include resize+normalize and batch-size sweep.
- Vision encoder throughput for base and large at 384x384, batch sizes 1/4/16.
- Captioning prefill split: vision encoder, text prefill, first LM logits.
- Decode tokens/sec with and without cross-attention K/V cache.
- VQA split: vision encoder, question encoder, answer decode.
- Retrieval split: image feature cache throughput, text feature throughput, ITM classifier and similarity matrix throughput.
- LM head cost for full logits vs `logits_to_keep=1`.
- Memory probes: per-layer self KV cache, cross-attn image K/V cache, and large checkpoint vision activations.

## 14. Skip/defer list

- Training losses, label smoothing, and gradient-only attention map hooks.
- Gradient checkpointing and output recording for hidden states/attentions.
- Beam search, sampling, and advanced generation processors beyond what is needed for first greedy parity.
- Dynamic arbitrary image sizes and bicubic position interpolation; support fixed 384 first.
- Deprecated `BlipModel` contrastive loss.
- Quantization and multi-GPU/tensor parallelism.
- Tokenizer implementation inside DinoML runtime; keep tokenization in CPU/data pipeline.

## 15. Final implementation checklist

- [ ] Parse `BlipConfig`, `BlipTextConfig`, and `BlipVisionConfig`.
- [ ] Load tied text embedding/LM head weights without cloning logical parameters.
- [ ] Implement BLIP image processor contract or define CPU-pipeline boundary for `pixel_values`.
- [ ] Implement NCHW Conv2d patch embedding or guarded Conv2d-to-GEMM rewrite.
- [ ] Implement vision pre-norm encoder block and post layer norm/CLS pooling.
- [ ] Implement text embeddings with position offset for cached decode.
- [ ] Implement text self-attention masks, causal masks, and additive mask values.
- [ ] Implement text cross-attention with `encoder_hidden_size` possibly different from text hidden size.
- [ ] Implement `EncoderDecoderCache`-equivalent self and cross K/V cache behavior.
- [ ] Implement LM prediction head and `logits_to_keep`.
- [ ] Implement captioning generate prompt/BOS/SEP behavior.
- [ ] Add VQA question encoder and answer decoder staging.
- [ ] Add ITM classifier and projection-similarity retrieval heads.
- [ ] Add parity tests for processor, one block, full vision, cross-attn, cache decode, captioning, VQA, and ITM.
- [ ] Benchmark preprocessing, vision encoder, prefill, decode, VQA, and retrieval branches.
