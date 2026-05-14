# InstructBLIP Transformers Family Audit

Primary target: multimodal image-to-text generation for `InstructBlipForConditionalGeneration`. First integration should treat InstructBLIP as a composed runtime: BLIP-style ViT vision encoder, instruction-aware Q-Former, a learned query-to-LLM bridge, then either a T5 seq2seq LM or a LLaMA/Vicuna causal LM. The instruction-aware Q-Former and image-token masked scatter are the main family-specific pieces; the large language model should compose separately audited T5/LLaMA coverage where possible.

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  instructblip family; representative checkpoints listed below
Config source:
  Hugging Face config.json, generation_config.json, processor_config.json,
  preprocessor_config.json, tokenizer_config.json
Source files inspected:
  X:/H/transformers/src/transformers/models/instructblip/modeling_instructblip.py
  X:/H/transformers/src/transformers/models/instructblip/configuration_instructblip.py
  X:/H/transformers/src/transformers/models/instructblip/processing_instructblip.py
  X:/H/transformers/src/transformers/models/instructblip/convert_instructblip_original_to_pytorch.py
  X:/H/transformers/src/transformers/models/blip/image_processing_blip.py
  X:/H/transformers/docs/source/en/model_doc/instructblip.md
  X:/H/transformers/tests/models/instructblip/test_modeling_instructblip.py
Any missing files or assumptions:
  No remote-code files are required for the inspected Salesforce checkpoints.
  Local source is authoritative. Official HF configs were fetched from public
  model repos on 2026-05-13. Importing the local checkout was blocked by an
  installed huggingface_hub API mismatch, so effective omitted defaults were
  taken from configuration_instructblip.py rather than AutoConfig execution.
```

Pinned source URLs:

- `modeling_instructblip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/instructblip/modeling_instructblip.py
- `configuration_instructblip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/instructblip/configuration_instructblip.py
- `processing_instructblip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/instructblip/processing_instructblip.py
- `image_processing_blip.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/blip/image_processing_blip.py

Representative HF configs fetched:

- `Salesforce/instructblip-flan-t5-xl`: https://huggingface.co/Salesforce/instructblip-flan-t5-xl/resolve/main/config.json
- `Salesforce/instructblip-flan-t5-xxl`: https://huggingface.co/Salesforce/instructblip-flan-t5-xxl/resolve/main/config.json
- `Salesforce/instructblip-vicuna-7b`: https://huggingface.co/Salesforce/instructblip-vicuna-7b/resolve/main/config.json
- `Salesforce/instructblip-vicuna-13b`: https://huggingface.co/Salesforce/instructblip-vicuna-13b/resolve/main/config.json

Processor/preprocessor/tokenizer configs were fetched from the same repos where available. No small official tiny-random InstructBLIP checkpoint was found; the local tests synthesize tiny configs instead.

## 2. High-level architecture

InstructBLIP is a multimodal generation family:

```text
image + prompt text
  -> CPU image/text processors
  -> ViT vision encoder
  -> instruction-aware Q-Former over [learned queries, qformer text tokens]
     with query-prefix cross-attention to vision tokens
  -> Linear(qformer_hidden -> language_hidden)
  -> replace <image> placeholder embeddings in the LM prompt
  -> T5 encoder-decoder generation or LLaMA/Vicuna causal generation
```

Implemented heads:

- `InstructBlipForConditionalGeneration`: required. Owns vision encoder, Q-Former, `language_projection`, masked image-token embedding replacement, and generation wrapper.
- `InstructBlipModel`: optional base model using `AutoModel` instead of a generation LM head. Useful for feature parity but not the first product target.
- `InstructBlipVisionModel`: optional independently stageable vision encoder.
- `InstructBlipQFormerModel`: required as a family-specific stage.

Stage decomposition:

- CPU/data pipeline: RGB conversion, resize to 224x224 by common checkpoints, rescale by `1/255`, CLIP mean/std normalization, tokenizer prompt packing, and separate Q-Former tokenization.
- Cacheable vision stage: `pixel_values[N,3,H,W] -> image_embeds[N, 1 + H/P * W/P, 1408]` for default vision config. At 224/14 this is 257 tokens.
- Cacheable instruction-aware image feature stage: `image_embeds + qformer_input_ids -> language_model_inputs[N, num_query_tokens, text_hidden]`. This depends on the instruction text, so it is cacheable only for the same image and Q-Former prompt.
- Prefix construction: text embeddings contain `num_query_tokens` `<image>` placeholders. `masked_scatter` replaces those embedding positions with projected query outputs.
- LM prefill/decode: delegated to nested T5 or LLaMA generation. For causal LLaMA, the image features live inside the prefill prefix and then normal KV-cache decode begins. For T5, the image features are part of encoder `inputs_embeds`; decoder generation uses T5 encoder-decoder cache behavior.

## 3. Important config dimensions

Source defaults from `configuration_instructblip.py`:

| Component | Field | Default/effective behavior |
| --- | --- | --- |
| Top-level | `num_query_tokens` | 32 |
| Top-level | `image_token_index` | Optional in source defaults; real checkpoints set it. Required for placeholder replacement. |
| Vision | `hidden_size` / layers / heads | 1408 / 39 / 16 |
| Vision | `intermediate_size` | 6144 |
| Vision | `image_size`, `patch_size` | 224, 14 |
| Vision | `hidden_act`, `layer_norm_eps` | `gelu`, `1e-6` |
| Vision | `qkv_bias` | true; source stores q and v bias, k bias zero |
| Q-Former | `hidden_size` / layers / heads | 768 / 12 / 12 |
| Q-Former | `intermediate_size` | 3072 |
| Q-Former | `max_position_embeddings` | 512 text positions; query embeddings are prepended without position embeddings |
| Q-Former | `cross_attention_frequency` | 2, including layer 0 |
| Q-Former | `encoder_hidden_size` | Forced to `vision_config.hidden_size` in top-level config post-init |
| Text | `text_config` | Any `AutoModelForSeq2SeqLM` or `AutoModelForCausalLM` config; current Salesforce checkpoints use T5 or LLaMA/Vicuna |

Representative checkpoint sweep:

| Checkpoint | LM branch | Query tokens | Image token id | LM hidden | LM layers | LM heads / KV heads | FFN | Vocab | Positions | dtype source |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `Salesforce/instructblip-flan-t5-xl` | T5 seq2seq | 32 | 32100 | 2048 | 24 enc / 24 dec | 32 / n/a, `d_kv=64` | 5120 gated-gelu | 32128 | `n_positions=512`; relative buckets 32/max 128 | config `float32` |
| `Salesforce/instructblip-flan-t5-xxl` | T5 seq2seq | 32 | 32100 | 4096 | 24 enc / 24 dec | 64 / n/a, `d_kv=64` | 10240 gated-gelu | 32128 | relative buckets 32/max 128 | config `float32` |
| `Salesforce/instructblip-vicuna-7b` | LLaMA causal | 32 | 32001 | 4096 | 32 | 32 / 32, head_dim 128 | 11008 SwiGLU | 32064 | `max_sequence_length=2048`, RoPE theta 10000 | config `float16` |
| `Salesforce/instructblip-vicuna-13b` | LLaMA causal | 32 | 32001 | 5120 | 40 | 40 / 40, head_dim 128 | 13824 SwiGLU | 32064 | `max_sequence_length=2048`, RoPE theta 10000 | config `float16` |

All four inspected configs omit most vision and Q-Former dimensions, relying on the InstructBLIP source defaults above. Their `qformer_config.vocab_size` is 30523, one above the BERT base vocabulary because conversion adds a `[DEC]` BOS token for the Q-Former tokenizer.

## 3a. Family variation traps

- T5 and Vicuna branches are structurally different. T5 uses encoder-decoder relative position bias and decoder cross-attention; Vicuna uses causal LLaMA RoPE, RMSNorm, SwiGLU, and autoregressive KV cache.
- `use_decoder_only_language_model` is derived from `text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`; do not infer from `is_encoder_decoder` alone.
- The source also has a historical `config.image_token_id` alias mapped to `image_token_index`, while code paths use both names. DinoML should normalize this at config load.
- Processor packing is required. It prepends exactly `num_query_tokens` `<image>` token placeholders to the text tokenizer input when images are present. Missing or truncated placeholders cause `masked_scatter` shape failures or silent semantic corruption.
- The Q-Former receives instruction text, unlike BLIP-2. Q-Former attention mask length is `num_query_tokens + qformer_prompt_len`, and only the query prefix is cross-attended to vision tokens and later projected to the LM.
- Q-Former layers split FFNs: query tokens use `intermediate_query/output_query`; text tokens use `intermediate/output`. A naive shared-FFN BERT lowering is wrong.
- Q-Former cross-attention exists on layers where `layer_idx % cross_attention_frequency == 0`, including layer 0.
- Vision patch embedding source layout is NCHW. NHWC/channel-last is an optimization only inside the Conv2d/flatten/transpose region and must preserve token order.
- Vision attention QKV is a single packed `Linear(hidden, 3*hidden)` with a q/k/v reshape order `[B,S,3,H,D] -> [3,B,H,S,D]`; q and v can have bias while k bias is zero.
- `InstructBlipQFormerModel` explicitly disables attention backends. Its attention adds masks and stores cross-attention maps in eager code; first DinoML parity should implement eager MHA.
- The vision model advertises SDPA/Flash/Flex attention support through `ALL_ATTENTION_FUNCTIONS`, but the custom eager path does not upcast attention weights to fp32.
- The common Vicuna processor config currently reports `processor_class: InstructBlipVideoProcessor` even for the image checkpoint, while `preprocessor_config.json` and model source are image InstructBLIP. Treat this as a loading/metadata trap, not a requirement to support video in this report.
- `generate()` builds a default prompt only when `input_ids` is absent: `[image_token_index] * num_query_tokens + [text_config.bos_token_id]`. Processor-based calls normally provide packed `input_ids`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input, dtype cast to Conv2d weight dtype.
- Conv2d patch embed `3 -> 1408`, kernel=stride=14, bias true.
- Flatten spatial `[B,C,H/P,W/P] -> [B,C,L]`, transpose to `[B,L,C]`, concat CLS at dim 1.
- Learned positional add; optional bicubic interpolation for non-default image sizes.
- LayerNorm over last dim with eps `1e-6` for vision, `1e-12` for Q-Former.
- View/reshape/permute/contiguous for QKV and attention heads.
- `torch.cat` for query/text masks and Q-Former embeddings.
- `masked_scatter` or equivalent indexed embedding replacement from `[B,32,text_hidden]` into LM token embeddings.
- Boolean/equality placeholder mask generation, unsqueeze, expand_as.

Neural network primitives:

- Vision packed QKV Linear `1408 -> 4224`, projection `1408 -> 1408`, MLP `1408 -> 6144 -> 1408`, GELU.
- Q-Former embeddings: word embedding `30523 x 768`, position embedding `512 x 768`, add, LayerNorm, dropout disabled for inference.
- Q-Former self-attention Linear Q/K/V `768 -> 768`, output Linear `768 -> 768`, FFNs `768 -> 3072 -> 768`.
- Q-Former cross-attention on query tokens: Q `768 -> 768`, K/V `1408 -> 768`, output `768 -> 768`.
- Language bridge Linear `768 -> text_hidden`: 2048/4096 for FLAN-T5, 4096/5120 for Vicuna.
- Nested LM operators from T5 or LLaMA reports: T5 relative attention/gated-GELU/RMS-style layer norm; LLaMA RoPE/RMSNorm/SwiGLU/causal attention.

Attention primitives:

- Vision encoder: noncausal MHA, 16 heads, head_dim 88, no attention mask in normal path.
- Q-Former self-attention: noncausal MHA over `[query tokens + qformer text tokens]`, 12 heads, head_dim 64, additive mask `(1-mask)*-10000`.
- Q-Former cross-attention: noncausal query-to-vision MHA over only the first `num_query_tokens` hidden states, 12 heads, K/V from vision tokens.
- Nested LM attention: T5 encoder/decoder/cross-attention or LLaMA causal self-attention with cache.

Generation/cache ops:

- Generation wrapper with `inputs_embeds` instead of only `input_ids`.
- For causal LM branch, pass `input_ids` alongside `inputs_embeds` to generation when available.
- For encoder-decoder branch, omit `input_ids` and pass `inputs_embeds` plus `attention_mask` to the T5 encoder.
- LM KV cache is delegated to nested LM. Vision/Q-Former features are precomputed once per generate call and are not autoregressive KV cache entries themselves.

Preprocessing-coupled ops:

- Blip image processor: RGB conversion, resize to 224x224 using resample id 3, rescale factor `0.00392156862745098`, normalize by CLIP mean `[0.48145466,0.4578275,0.40821073]` and std `[0.26862954,0.26130258,0.27577711]`.
- Text tokenizer and Q-Former tokenizer are separate. Q-Former tokenizer is BERT base uncased plus `[DEC]` in conversion; LM tokenizer is T5 or LLaMA with truncation side left in conversion/checkpoint configs.
- Processor reduces text `max_length` by `num_query_tokens` before adding image placeholders.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values[B,3,H,W]
patch = Conv2d(3, 1408, kernel=14, stride=14)(pixel_values)
patch = flatten(2).transpose(1,2)                       # [B, (H/14)*(W/14), 1408]
x = cat([class_embedding.expand(B,1,1408), patch], dim=1)
x = x + position_embedding[:, :x_len, :]
```

Vision encoder layer, repeated 39 times by default:

```text
res = x
x = LayerNorm(eps=1e-6)(x)
qkv = Linear(1408 -> 4224, packed q/k/v)(x)
q,k,v = reshape to [B,16,S,88]
x = softmax((q @ k.T) * sqrt(88)^-1) @ v
x = Linear(1408 -> 1408)(x) + res
res = x
x = LayerNorm(eps=1e-6)(x)
x = Linear(1408 -> 6144) -> GELU -> Linear(6144 -> 1408)
x = x + res
```

Vision post:

```text
last_hidden_state = LayerNorm(eps=1e-6)(x)
pooler_output = LayerNorm(eps=1e-6)(last_hidden_state[:,0,:])
```

The pooled CLS output is not used by `InstructBlipForConditionalGeneration`; Q-Former consumes all vision tokens.

Q-Former embeddings:

```text
text = word_embedding(qformer_input_ids) + position_embedding(position_ids)
x = cat([query_embeds[B,32,768], text], dim=1)
x = LayerNorm(eps=1e-12)(x)
```

Q-Former layer, repeated 12 times:

```text
x = SelfAttention(x, additive_mask) -> Linear -> dropout -> LayerNorm(residual)
query_part = x[:, :32, :]
if layer_idx % 2 == 0:
    query_part = CrossAttention(query_part, vision_tokens, vision_mask)
query_part = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> LayerNorm(residual)
if text_part exists:
    text_part = Linear(768 -> 3072) -> GELU -> Linear(3072 -> 768) -> LayerNorm(residual)
    x = cat([query_part, text_part], dim=1)
else:
    x = query_part
```

Language bridge and LM:

```text
query_output = qformer_last_hidden[:, :32, :]
image_features = Linear(768 -> text_hidden)(query_output)
inputs_embeds = language_model.embed_tokens(input_ids)
mask = input_ids == image_token_index
inputs_embeds = masked_scatter(mask.expand_as(inputs_embeds), image_features)
logits = language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, ...)
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention over patch plus CLS tokens.
- MHA only; no GQA/MQA.
- Heads: 16, head_dim 88 for default hidden 1408.
- No padding mask in normal source path.
- Source may dispatch through eager/SDPA/Flash/Flex interfaces, but eager math is `matmul -> scale -> optional mask add -> softmax -> dropout -> matmul` with no fp32 upcast.
- No KV cache.

Q-Former self-attention:

- Noncausal self-attention over query prefix and instruction tokens.
- Heads: 12, head_dim 64.
- Additive mask from 2D or 3D attention mask, converted to model dtype and `(1.0 - mask) * -10000.0`.
- Query tokens are visible to instruction tokens and vice versa through the self-attention mask unless the caller supplies a custom 3D mask.
- No KV cache in source.
- Source disables optimized attention backend support for Q-Former; implement eager parity first.

Q-Former cross-attention:

- Cross-attention occurs in layers 0, 2, 4, 6, 8, 10 with default frequency 2.
- Only query prefix hidden states cross-attend to image embeddings; instruction text tokens skip cross-attention and use the text FFN path.
- K/V projection input dim is `encoder_hidden_size=1408`; Q/output hidden dim is 768.
- Vision attention mask defaults to all ones over image tokens and is inverted to additive mask.
- Cross-attention outputs are not cached across different instructions; they can be cached for repeated decode using the same image and Q-Former prompt because they are computed before LM generation.

Nested LM attention:

- FLAN-T5 checkpoints require encoder self-attention with relative position bucket bias, decoder causal self-attention with cache, and decoder cross-attention to the T5 encoder output. T5 caches decoder self-attention K/V and encoder-decoder K/V according to the T5 implementation, not InstructBLIP source.
- Vicuna checkpoints require LLaMA causal self-attention, RoPE before cache storage in the nested LLaMA implementation, full MHA (`num_key_value_heads == num_attention_heads` in inspected configs), and generation KV cache.

## 7. Position encoding and custom math

Vision positional interpolation is the only InstructBLIP-specific position math outside the nested LM:

```python
def interpolate_vision_pos(pos, embeddings, height, width, patch_size):
    cls = pos[:, :1]
    patch = pos[:, 1:]
    dim = embeddings.shape[-1]
    src = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, src, src, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(
        patch,
        size=(height // patch_size, width // patch_size),
        align_corners=False,
    )
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return concat([cls, patch], dim=1)
```

Default processor sizes avoid interpolation. If `interpolate_pos_encoding=False`, source slices learned position embeddings to current sequence length and assumes the grid matches the trained square size.

Q-Former text positions use absolute learned embeddings for only `qformer_input_ids`. Learned query embeddings are prepended after text token+position addition and do not receive position embeddings.

Nested LM position math is delegated:

- T5: relative attention bucket bias, no absolute token position embeddings.
- Vicuna/LLaMA: RoPE with `rope_theta=10000`, no scaling in inspected configs.

## 8. Preprocessing and input packing

Processor output tensors for image+text calls:

```text
pixel_values: [B, 3, 224, 224] float, normalized NCHW
input_ids: [B, 32 + prompt_len_with_specials] long, with 32 leading <image> tokens
attention_mask: same length as input_ids
qformer_input_ids: [B, qformer_prompt_len] long
qformer_attention_mask: same length as qformer_input_ids
```

Image preprocessing should remain in the CPU/data pipeline initially. Runtime graph starts at `pixel_values`.

Text packing contracts:

- Processor adds a special `<image>` token to the LM tokenizer if missing.
- If `max_length` is provided, processor subtracts `num_query_tokens` before tokenizing text, then prepends image placeholders without special tokens or padding.
- Q-Former tokenization uses the original instruction text independently and does not include `<image>` placeholders.
- Conversion code uses `google-bert/bert-base-uncased` for Q-Former tokenizer, adds `[DEC]` as BOS, and sets truncation side left. LM tokenizers are T5 or LLaMA with left truncation; Vicuna conversion adds `[PAD]`.

Embedding stitch:

```text
special_image_mask = input_ids == image_token_index
special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

The flattened true count in `special_image_mask` must equal `B * num_query_tokens * text_hidden`. DinoML should validate this explicitly before scatter.

Generation-controller behavior:

- `generate()` precomputes image features once, builds or embeds the LM prompt, replaces placeholders, and calls nested `language_model.generate`.
- If no `input_ids` are supplied, source creates `[image_token_index] * num_query_tokens + [bos_token_id]`, repeated for batch.
- Generation options such as beams, sampling, length penalty, and repetition penalty are standard nested LM generation behavior. They can be stubbed at first if DinoML validates logits/prefill/decode parity separately.

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision patch Conv2d -> Linear

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input layout is source NCHW or a fully guarded NHWC local region.
- Runtime `height` and `width` divisible by `patch_size`.
- Weight is standard Conv2d `[out_channels, in_channels, kh, kw]`; bias optional.

Replacement:

```text
PatchExtract(NCHW, kh=kw=P, stride=P)
  -> WindowFlatten in PyTorch conv order
  -> GEMM(weight.reshape(out, 3*P*P).T)
  -> BiasAdd
  -> Reshape [B, H/P * W/P, hidden]
```

Failure cases: non-default interpolation with dynamic H/W still works if patch extraction emits the same grid order, but positional interpolation must also be lowered. Any layout pass must preserve the flatten order that feeds token positions.

Parity test sketch: compare Conv2d patch embeddings plus flatten/transpose against rewritten GEMM on random NCHW images for 224 and one larger divisible size with interpolated positions disabled for the patch-only test.

### Rewrite: image-token masked scatter -> indexed copy

Preconditions:

- Placeholder count per sample equals `num_query_tokens`.
- Placeholder positions are known from `input_ids == image_token_index`; common processor places them contiguously at the front.
- `image_features.shape == [B, num_query_tokens, text_hidden]`.

Replacement:

```text
inputs_embeds = token_embedding(input_ids)
for each batch row:
  copy image_features[row, :, :] into placeholder positions
```

Optimized special case: if placeholders are contiguous prefix positions, replace with a slice assignment into `inputs_embeds[:, :num_query_tokens, :]`.

Failure cases: caller-supplied prompts may place image tokens outside the prefix or pass `inputs_embeds` without `input_ids`; keep generic indexed copy fallback.

### Rewrite: Q-Former query/text split FFN

Preconditions:

- `query_length == config.num_query_tokens`.
- Layer has default Q-Former structure.
- Query and text FFN weights are distinct and must not be merged.

Replacement:

```text
self_attention over full sequence
query = slice(x, :Q)
text = slice(x, Q:)
query = optional_cross_attention(query, vision)
query = query_ffn(query)
text = text_ffn(text)
x = concat(query, text)
```

This rewrite makes the cross-attention-only-query behavior explicit and prevents accidental BERT-block canonicalization.

### Rewrite: precompute image features for decode

Preconditions:

- Same `pixel_values`, `qformer_input_ids`, and `qformer_attention_mask`.
- Same model weights and dtype policy.
- Generation only changes autoregressive decoded tokens after the initial packed prompt.

Replacement:

```text
vision + qformer + language_projection -> image_features cache
image_features + prompt embeddings -> LM prefill
decode uses nested LM KV cache only
```

Failure cases: changing the instruction prompt changes Q-Former outputs even for the same image.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm kernels for vision and Q-Former. These are on every block and use small eps variants.
- Packed vision QKV projection plus reshape. Default shape `1408 -> 4224` over 257 tokens, repeated 39 times.
- Q-Former attention and cross-attention eager MHA. Cross-attention K/V projection from 1408 to 768 is family-specific and latency-visible.
- Image-token indexed copy. This removes generic `masked_scatter` overhead and adds strong shape validation.
- Delegate/fuse nested LM hot paths according to T5/LLaMA reports: T5 relative attention and gated-GELU; LLaMA RMSNorm, RoPE attention, SwiGLU, KV-cache decode.

Medium priority:

- Vision patch Conv2d to GEMM or specialized patchify GEMM.
- Vision MLP GELU fusion.
- Q-Former FFN fusion, separately for query and text paths.
- Last-token-only logits for causal Vicuna branch when generating.
- T5 encoder input prefix construction from image features and prompt embeddings.

Lower priority:

- Bicubic positional interpolation on GPU. Default checkpoint preprocessing avoids it.
- Full SDPA/Flash attention for vision. Useful later, but eager parity is easier and Q-Former still needs a custom path.
- Beam-search controller in DinoML. Standard generation can be hosted outside the compiled module at first.

## 11. Runtime staging plan

1. Parse top-level config and normalize nested defaults: source defaults for omitted vision/Q-Former configs, `image_token_index`, `num_query_tokens`, and LM branch type.
2. Load weights for vision encoder, Q-Former, query tokens, language projection, and route nested LM weights to existing T5/LLaMA loaders.
3. Implement standalone vision encoder parity for `pixel_values -> last_hidden_state`.
4. Implement Q-Former parity with instruction tokens, full self-attention, query-prefix cross-attention every second layer, and split query/text FFNs.
5. Implement `get_image_features`: vision + Q-Former + language projection, validating `[B,32,text_hidden]`.
6. Implement LM prefix embedding replacement with a prefix-slice fast path and generic indexed-copy fallback.
7. For FLAN-T5 checkpoints, run T5 encoder from `inputs_embeds` and validate decoder logits/decode through the T5 path.
8. For Vicuna checkpoints, run LLaMA causal prefill from `inputs_embeds`, then cached decode.
9. Add optimized kernels and layout rewrites only after stage-level fp32 parity is stable.

Stubbable initially: labels/loss, dropout/training, gradient checkpointing, attention recording hooks, accelerate device-map handling, beam search, quantization, and non-default image-size interpolation.

## 12. Parity and validation plan

- Config tests: verify omitted nested configs resolve to source defaults; verify T5 vs LLaMA branch routing.
- Processor contract tests: with a real processor output, assert `input_ids` has exactly 32 image placeholders, `qformer_input_ids` does not, and `pixel_values` are NCHW normalized.
- Patch embedding rewrite tests: Conv2d vs GEMM rewrite for random fp32 inputs.
- Vision single-layer and full-encoder parity: fp32 tolerance `1e-4` absolute/relative; fp16 tolerance around `5e-3` depending on attention backend.
- Q-Former layer parity: separately test layers with and without cross-attention, including text length zero and nonzero.
- `get_image_features` parity: compare projected query outputs for one image and prompt.
- Embedding stitch parity: source `masked_scatter` vs DinoML indexed copy for prefix and non-prefix placeholder layouts.
- T5 branch parity: prefill encoder hidden states/logits for `instructblip-flan-t5-xl` config with small random weights, then one decode token.
- Vicuna branch parity: causal prefill logits and one cached decode token.
- End-to-end smoke: one fixed public image and prompt, compare generated token IDs under greedy decoding for a small max length after stage parity is established.

## 13. Performance probes

- CPU preprocessing throughput: images/sec for resize/rescale/normalize and tokenizer packing.
- Vision encoder throughput: batch sweep over 1, 2, 4, 8 at 224x224.
- Q-Former throughput: sweep qformer prompt lengths while holding image token count 257 and query count 32.
- Image-feature cache hit probe: repeated decode with same image/prompt to isolate LM-only generation.
- LM prefill throughput: T5 encoder input length `32 + prompt_len`; Vicuna causal prefix length `32 + prompt_len`.
- Decode tokens/sec with nested LM cache for Vicuna 7B/13B and T5 decoder.
- Memory probes: vision activations, Q-Former activations, image feature prefix, and nested LM KV cache separately.
- Scatter/indexed-copy cost probe: generic `masked_scatter` vs prefix slice assignment.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, gradient checkpointing.
- Attention map saving and backward hooks.
- Accelerate multi-device `_hf_hook` behavior.
- Quantized loading and bitsandbytes paths.
- InstructBLIPVideo and `video_token_index`; this report is image InstructBLIP only.
- Non-default image resolution with positional interpolation, after default 224 path is stable.
- Full generation algorithms beyond greedy/sampling handoff: beam search can remain in a controller layer.
- Vision FlashAttention/SDPA optimization until eager parity exists.
- Remote-code or non-Salesforce checkpoints that alter the Q-Former or processor contract.

## 15. Final implementation checklist

- [ ] Parse `InstructBlipConfig` and apply source defaults for omitted vision/Q-Former fields.
- [ ] Normalize `image_token_id`/`image_token_index` and validate `num_query_tokens`.
- [ ] Load Blip image processor settings and document CPU preprocessing boundary.
- [ ] Load separate LM tokenizer and Q-Former tokenizer contracts.
- [ ] Implement/load vision patch embedding, CLS token, positional add, and ViT encoder.
- [ ] Add guarded Conv2d patch embedding -> GEMM rewrite.
- [ ] Implement Q-Former embeddings with query-token prepend.
- [ ] Implement Q-Former self-attention eager path and additive mask.
- [ ] Implement Q-Former query-only cross-attention on configured layer interval.
- [ ] Implement separate query/text FFN paths in Q-Former layers.
- [ ] Implement `language_projection` and `get_image_features` parity.
- [ ] Implement image placeholder validation and indexed embedding replacement.
- [ ] Route FLAN-T5 branch to seq2seq LM from `inputs_embeds`.
- [ ] Route Vicuna branch to causal LM prefill/decode from `inputs_embeds`.
- [ ] Add stage parity tests for vision, Q-Former, image features, and embedding stitch.
- [ ] Add branch parity tests for T5 and LLaMA/Vicuna one-step generation.
- [ ] Benchmark preprocessing, vision, Q-Former, prefill, decode, and cache memory separately.
