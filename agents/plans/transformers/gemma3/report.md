# Transformers Gemma3 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model family / architecture:
  gemma3

Primary runtime target:
  Generation. Cover both Gemma3TextForCausalLM text-only generation and
  Gemma3ForConditionalGeneration image+text generation.

Source files inspected:
  transformers/src/transformers/models/gemma3/modular_gemma3.py
  transformers/src/transformers/models/gemma3/modeling_gemma3.py
  transformers/src/transformers/models/gemma3/configuration_gemma3.py
  transformers/src/transformers/models/gemma3/processing_gemma3.py
  transformers/src/transformers/models/gemma3/image_processing_gemma3.py
  transformers/src/transformers/models/gemma3/image_processing_pil_gemma3.py
  transformers/src/transformers/models/siglip/modeling_siglip.py
  transformers/src/transformers/models/siglip/configuration_siglip.py
  transformers/src/transformers/cache_utils.py

Pinned source URLs for future review:
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma3/modular_gemma3.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma3/modeling_gemma3.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma3/configuration_gemma3.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma3/processing_gemma3.py
  https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma3/image_processing_gemma3.py

Representative configs and processor configs:
  Official google/gemma-3-* repos were gated in this environment, returning
  HTTP 401 for raw config files. Open Hugging Face mirrors were used and are
  labeled as such:
  https://huggingface.co/twm-opensource/gemma-3-1b-it/raw/main/config.json
  https://huggingface.co/agentlans/gemma-3-1b-it-multilingual-ner/raw/main/config.json
  https://huggingface.co/Gunulhona/Gemma-3-4B/raw/main/config.json
  https://huggingface.co/Runware/gemma-3-12b-pt/raw/main/config.json
  https://huggingface.co/msp5382/gemma-3-12b-capstone-ft/raw/main/config.json
  https://huggingface.co/Gunulhona/Gemma-3-27B/raw/main/config.json
  https://huggingface.co/ZySec-AI/gemma-3-4b-document-writer-lora/raw/main/processor_config.json
  https://huggingface.co/ZySec-AI/gemma-3-4b-document-writer-lora/raw/main/preprocessor_config.json

Any missing files or assumptions:
  modeling_gemma3.py and configuration_gemma3.py are generated from
  modular_gemma3.py. The generated modeling file is the exact runtime source in
  this checkout; modular_gemma3.py is authoritative for future Transformers
  source edits. No remote-code files are required for standard Gemma3 classes.
  This report assumes inference-only CUDA execution. No DinoML tests were run.
```

## 2. High-level architecture

Gemma3 has two generation surfaces in this source:

- Text-only decoder: `Gemma3TextModel` plus `Gemma3ForCausalLM`.
- Image+text conditional generation: SigLIP vision tower plus Gemma3 multimodal projector plus `Gemma3TextModel` language model and LM head.

Text-only dataflow:

```text
tokenizer -> input_ids/attention_mask
  -> scaled token embedding
  -> decoder prefill/decode stack with local/full RoPE attention and KV cache
  -> final RMSNorm
  -> lm_head logits, optionally logits_to_keep
  -> sampling
```

Multimodal dataflow:

```text
CPU image processor + tokenizer/placeholder expansion
  -> pixel_values + input_ids + token_type_ids
  -> SigLIP vision encoder
  -> average-pool + RMSNorm + projection to text hidden size
  -> masked_scatter image embeddings into text embedding stream
  -> decoder prefill/decode with image-aware masks and KV cache
  -> lm_head logits
```

Stage decomposition for image+text:

- CPU/data pipeline: fetch images, RGB conversion, resize/rescale/normalize, optional pan-and-scan crop construction, chat template/tokenization, image placeholder expansion.
- Vision encoder: `pixel_values -> SigLIP last_hidden_state`. This stage is independently testable and cacheable per image.
- Multimodal projector: reshape 64x64 SigLIP patch sequence to NCHW, 4x4 average pool to 16x16, RMSNorm, `1152 -> text_hidden` matmul. This is independently testable.
- Prefix construction: replace 256 `image_token_index` placeholders per image/crop with projected image embeddings. This must be validated before language prefill.
- Prefill: full mixed image+text sequence through hybrid full/sliding attention.
- Decode: text-only token steps with cached K/V. `prepare_inputs_for_generation` passes `pixel_values` only on the first iteration when cache is used, and drops `token_type_ids` on later iterations.

## 3. Important config dimensions

Source defaults from `Gemma3TextConfig` and `Gemma3Config`:

| Field | Source default | Runtime significance |
|---|---:|---|
| `vocab_size` | 262208 | text embedding and LM head size |
| `hidden_size` | 2304 | text hidden width |
| `intermediate_size` | 9216 | gated MLP width |
| `num_hidden_layers` | 26 | decoder layers |
| `num_attention_heads` | 8 | query heads |
| `num_key_value_heads` | 4 | KV heads; GQA required |
| `head_dim` | 256 | explicit; do not infer only from `H/A` |
| `max_position_embeddings` | 131072 | RoPE cache horizon default |
| `hidden_activation` | `gelu_pytorch_tanh` | gated MLP activation |
| `rms_norm_eps` | 1e-6 | text RMSNorm eps |
| `attention_bias` | false | Q/K/V/O projections are bias-free in sampled configs |
| `query_pre_attn_scalar` | 256 | attention scale is `query_pre_attn_scalar**-0.5`, not always `head_dim**-0.5` |
| `sliding_window` | 4096 source default | sampled production configs use 512 or 1024 |
| `sliding_window_pattern` | 6 | every 6th layer is full attention when layer_types omitted |
| local RoPE theta | 10000 | for `sliding_attention` |
| global RoPE theta | 1000000 | for `full_attention` |
| `mm_tokens_per_image` | 256 | image placeholders and projector output tokens |
| `image_token_index` | 262144 | placeholder token id, may be outside 1B vocab |
| `boi_token_index` / `eoi_token_index` | 255999 / 256000 | processor image wrapper tokens |

Representative checkpoint sweep from open mirrors:

| Config source | Class | H | I | Layers | Q heads | KV heads | D | KV groups | V | Max pos | Sliding window | RoPE | Vision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `twm-opensource/gemma-3-1b-it` mirror | `Gemma3ForCausalLM` | 1152 | 6912 | 26 | 4 | 1 | 256 | 4 | 262144 | 32768 | 512 | default, local theta 10000, full theta 1000000 | none |
| `agentlans/...1b...` mirror | `Gemma3ForCausalLM` | 1152 | 6912 | 26 | 4 | 1 | 256 | 4 | 262144 | 32768 | 512 | explicit layer_types, no rope scaling | none |
| `Gunulhona/Gemma-3-4B` mirror | `Gemma3ForConditionalGeneration` | 2560 | 10240 | 34 | 8 | 4 | 256 | 2 | 262208 | 131072 | 1024 | linear factor 8 for full attention | SigLIP 896/14, H=1152, 27 layers |
| `Runware/gemma-3-12b-pt` mirror | `Gemma3ForConditionalGeneration` | 3840 | 15360 | 48 | 16 | 8 | 256 | 2 | 262208 | 131072 | 1024 | linear factor 8 for full attention | SigLIP 896/14, H=1152, 27 layers |
| `msp5382/gemma-3-12b...` mirror | `Gemma3ForCausalLM` | 3840 | 15360 | 48 | 16 | 8 | 256 | 2 | 262208 | 131072 | 1024 | linear factor 8 | none |
| `Gunulhona/Gemma-3-27B` mirror | `Gemma3ForConditionalGeneration` | 5376 | 21504 | 62 | 32 | 16 | 128 | 2 | omitted | omitted in mirror | 1024 | linear factor 8; source fills missing defaults | SigLIP 896/14, H=1152, 27 layers |

Vision config for sampled multimodal mirrors:

| Field | Value | Notes |
|---|---:|---|
| `vision_config.model_type` | `siglip_vision_model` | loaded through `AutoModel.from_config` |
| `image_size` | 896 | processor mirror also uses 896x896 |
| `patch_size` | 14 | 64x64 patch grid, 4096 patch tokens |
| `hidden_size` | 1152 | projector input width |
| `intermediate_size` | 4304 | SigLIP MLP width |
| `num_hidden_layers` | 27 | non-causal vision transformer layers |
| `num_attention_heads` | 16 | head_dim 72 |
| `vision_use_head` | false | Gemma3 consumes last_hidden_state, no SigLIP pool head |
| `layer_norm_eps` | 1e-6 | SigLIP norms and projector RMSNorm |

## 3a. Family variation traps

- `gemma3_text` and `gemma3` are distinct config/model types. Text-only `Gemma3ForCausalLM` has no vision tower or placeholder scatter.
- 1B text configs use `vocab_size=262144`, while multimodal configs use `text_config.vocab_size=262208` and `image_token_index=262144`. The multimodal source replaces image ids with PAD before embedding if `image_token_index >= vocab_size`.
- `hidden_size != num_attention_heads * head_dim` for the 1B mirror: `1152 != 4 * 256`. Q/O projection widths are `num_attention_heads * head_dim = 1024`, not `hidden_size`. Do not assume attention output width equals H before `o_proj`.
- GQA is required. Sampled configs use `num_key_value_heads < num_attention_heads`; 1B is MQA-like with one KV head.
- Attention scale is `query_pre_attn_scalar**-0.5`. For 27B mirror, `query_pre_attn_scalar=168` while `head_dim=128`.
- Layer pattern is hybrid. If `layer_types` is omitted, `Gemma3TextConfig` marks layers 5, 11, 17, ... as `full_attention` and all others as `sliding_attention`.
- Local and global attention use different RoPE parameter buckets. Sliding attention defaults to theta 10000; full attention defaults to theta 1000000 and may receive `rope_scaling`/linear factor 8 from configs.
- `sliding_window` differs by checkpoint: 512 for 1B mirrors, 1024 for 4B/12B/27B mirrors, 4096 source default.
- `use_bidirectional_attention=True` changes masks and halves/offsets the configured sliding window in config post-init. It is not used in sampled generation configs but is a source-supported trap.
- Gemma3 RMSNorm stores a zero-initialized parameter and multiplies by `1.0 + weight`, not by `weight` directly.
- Decoder blocks use four RMSNorms: input, post-attention, pre-FFN, post-FFN. This differs from simpler two-norm decoders.
- Multimodal `token_type_ids` are semantic. They build block sequence ids so image blocks can attend bidirectionally within image regions while preserving causal text behavior.
- Processor pan-and-scan can insert additional image placeholders and crops. First integration can disable it only if the processor contract explicitly sets `do_pan_and_scan=False`.
- Source image tensors are channels-first NCHW. NHWC/channel-last may optimize SigLIP and pooling, but initial translation should preserve NCHW axes.

## 4. Operator coverage checklist

### Tensor/layout ops

- Embedding gather: `input_ids[B,T] -> [B,T,H]`, scaled by `sqrt(hidden_size)`.
- Leading-dim flatten for Linear/GEMM and restore `[B,T,*]`.
- Projection reshapes:
  - Q: `[B,T,A*D] -> [B,A,T,D]`.
  - K/V: `[B,T,KvH*D] -> [B,KvH,T,D]`.
- `transpose`, `reshape`, `contiguous` around attention.
- `repeat_kv` fallback: `[B,KvH,T,D] -> [B,A,T,D]` for eager attention.
- Causal and sliding-window mask construction with additive mask broadcast.
- Dict attention mask dispatch by layer type: `causal_mask_mapping["full_attention"|"sliding_attention"]`.
- Token id replacement for out-of-vocab image token: clone/scatter image ids to pad id 0 before embedding.
- Multimodal placeholder mask: `input_ids == image_token_index`, unsqueeze/expand to embedding shape.
- `masked_scatter` to replace image-token embeddings with projected image features.
- `token_type_ids` comparison, pad, cumsum, `where` for image block ids.
- `logits_to_keep` slicing before LM head.

### Neural network primitives

- Bias-free text Linear/GEMM for sampled configs:
  - 1B: Q `1152 -> 1024`, K/V `1152 -> 256`, O `1024 -> 1152`, gate/up `1152 -> 6912`, down `6912 -> 1152`, LM head `1152 -> 262144`.
  - 4B: Q `2560 -> 2048`, K/V `2560 -> 1024`, O `2048 -> 2560`, gate/up `2560 -> 10240`, down `10240 -> 2560`, LM head `2560 -> 262208`.
  - 12B: Q `3840 -> 4096`, K/V `3840 -> 2048`, O `4096 -> 3840`, gate/up `3840 -> 15360`, down `15360 -> 3840`.
  - 27B: Q `5376 -> 4096`, K/V `5376 -> 2048`, O `4096 -> 5376`, gate/up `5376 -> 21504`, down `21504 -> 5376`.
- RMSNorm with fp32 accumulation and `(1 + weight)` scale.
- Gated GELU MLP: `down(gelu_tanh(gate(x)) * up(x))`.
- Residual adds after post-attention norm and post-FFN norm.
- Optional logits tanh softcap for text-only class when `final_logit_softcapping` is configured.
- Optional attention logit softcap in eager helper if `softcap` is supplied by backend dispatch. In the local `Gemma3Attention.forward`, `attn_logit_softcapping` is stored but not explicitly passed to the selected interface.

### Attention primitives

- Decoder causal self-attention.
- Hybrid local/global attention by layer.
- MQA/GQA with native KV head count.
- RoPE on Q/K before cache update.
- KV cache update per layer; cache class depends on `layer_types`.
- Eager fallback: repeat K/V, matmul, optional softcap, additive mask, fp32 softmax, dropout, matmul V.
- Optimized backend dispatch through `ALL_ATTENTION_FUNCTIONS`.

### Vision and projector primitives

- Image processor emits channels-first `pixel_values`.
- SigLIP patch embedding: `Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)` for 896x896 -> `[B,1152,64,64]`.
- SigLIP learned absolute position embedding add over 4096 tokens.
- SigLIP 27-layer non-causal MHA transformer:
  - Q/K/V/O `1152 -> 1152` with bias.
  - LayerNorm eps 1e-6.
  - MLP `1152 -> 4304 -> 1152` with `gelu_pytorch_tanh`.
- SigLIP post-layernorm.
- Projector reshape path:
  - `[B,4096,1152] -> transpose -> [B,1152,4096] -> reshape [B,1152,64,64]`.
  - AvgPool2d kernel/stride 4 -> `[B,1152,16,16]`.
  - flatten+transpose -> `[B,256,1152]`.
  - RMSNorm eps from vision config.
  - Matmul `[B,256,1152] x [1152,H] -> [B,256,H]`.

### Preprocessing-coupled ops

- RGB conversion, resize, rescale, normalize.
- Optional pan-and-scan crop generation.
- Text replacement of each begin-image token with `"\n\n" + boi + image_token * 256 + eoi + "\n\n"`.
- `token_type_ids` creation through processor multimodal token utility.
- Placeholder count check: total image-token embedding slots must equal projected feature elements.

## 5. Layer/block breakdown

Text model setup:

```text
input_ids[B,T] or inputs_embeds[B,T,H]
inputs_embeds = Embedding(V,H)(input_ids) * sqrt(H)
if use_cache and no past: past_key_values = DynamicCache(config)
position_ids = arange(T) + past_seen_tokens
mask_map = {
  full_attention: create_causal_mask(...),
  sliding_attention: create_sliding_window_causal_mask(...)
}
position_embeddings[layer_type] = rotary_emb(hidden_states, position_ids, layer_type)
```

Decoder block, repeated `N` times:

```text
residual = x
y = RMSNorm(input)(x)
q = Linear(H -> A*D, bias=attention_bias)(y).view(B,T,A,D).transpose(1,2)
k = Linear(H -> KvH*D, bias=attention_bias)(y).view(B,T,KvH,D).transpose(1,2)
v = Linear(H -> KvH*D, bias=attention_bias)(y).view(B,T,KvH,D).transpose(1,2)
q = RMSNorm(q_norm over D)(q)
k = RMSNorm(k_norm over D)(k)
q,k = apply_rope(q,k,cos[layer_type],sin[layer_type])
k,v = cache.update(k,v,layer_idx) if cache exists
a = attention(q,k,v, mask_map[layer_type], scale=query_pre_attn_scalar**-0.5,
              sliding_window=config.sliding_window only for sliding layers)
a = reshape to [B,T,A*D]
a = Linear(A*D -> H, bias=attention_bias)(a)
a = RMSNorm(post_attention)(a)
x = residual + a

residual = x
y = RMSNorm(pre_feedforward)(x)
y = Linear(I -> H)(gelu_tanh(Linear(H -> I)(y)) * Linear(H -> I)(y))
y = RMSNorm(post_feedforward)(y)
x = residual + y
```

Text LM head:

```text
x = final RMSNorm(x)
logits = Linear(H -> V, bias=False)(x[:, slice_indices, :])
if final_logit_softcapping: logits = tanh(logits / softcap) * softcap
```

Multimodal prefix stitch:

```text
pixel_values[Nimg,3,896,896] -> SigLIP last_hidden_state[Nimg,4096,1152]
image_features = projector(last_hidden_state)       # [Nimg,256,H]
inputs_embeds = text_embedding(llm_input_ids)        # [B,T,H]
mask = (input_ids == image_token_index)[..., None].expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
language_model(inputs_embeds=inputs_embeds, token_type_ids-derived masks, ...)
```

## 6. Attention requirements

Text attention:

- Causal decoder self-attention; no cross-attention module.
- GQA/MQA: Q heads `A`, KV heads `KvH`, group size `A / KvH`.
- Q shape `[B,A,Q,D]`; K/V shape `[B,KvH,K,D]` before repeat or native GQA.
- Scale is `query_pre_attn_scalar**-0.5`.
- Q/K per-head RMSNorm happens before RoPE.
- RoPE is applied before cache update, so cached keys are already rotated.
- Layer type selects both mask and RoPE parameter bucket:
  - `sliding_attention`: local mask, local theta by default 10000, sliding cache layer.
  - `full_attention`: full causal mask, global theta by default 1000000, full dynamic cache layer.
- `DynamicCache(config)` inspects `layer_types`; sliding layers store `[B,KvH,min(seq_len,sliding_window),D]` while full layers grow to `[B,KvH,seq_len,D]`.
- Eager fallback repeats K/V to `[B,A,K,D]`, applies additive mask, softmax in fp32, casts back to query dtype, applies dropout, and matmuls by V.
- FlashAttention/SDPA compatibility: suitable only if backend supports native GQA, per-layer sliding windows, additive masks, pre-rotated K cache, and the Gemma3 scale. A fallback can materialize repeated K/V but is too costly for production decode.

Multimodal masking:

- Processor returns `token_type_ids`; source maps `token_type_ids == 1` to image tokens.
- `get_block_sequence_ids_for_mask` starts a new image block when an image token follows a non-image token and assigns cumulative image block ids.
- These block ids are passed into `create_causal_mask` and `create_sliding_window_causal_mask`, allowing image block behavior to differ from normal text causal masking. DinoML should treat this as required for image+text parity.

Vision attention:

- SigLIP vision encoder uses non-causal MHA, no cache.
- Q/K/V shapes `[B,16,S,72]` for sampled Gemma3 vision configs (`S=4096` at 896x896).
- No RoPE; learned absolute positions are added before encoder layers.
- Dropout is configured but zero in sampled configs.

## 7. Position encoding and custom math

Gemma3 RoPE has separate parameter sets by layer type:

```python
def gemma3_inv_freq(config, layer_type):
    base = config.rope_parameters[layer_type]["rope_theta"]
    dim = config.head_dim
    i = arange(0, dim, 2, dtype=float32)
    return 1.0 / (base ** (i / dim))

def gemma3_rope(position_ids, inv_freq, attention_scaling, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], axis=-1)
    cos = cos(emb) * attention_scaling
    sin = sin(emb) * attention_scaling
    return cos.to(dtype), sin.to(dtype)

def apply_gemma3_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Important details:

- RoPE computation forces fp32 under disabled autocast, then casts cos/sin to activation dtype.
- The source supports non-default RoPE types through `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; sampled 4B/12B/27B mirrors use `rope_scaling={"rope_type":"linear","factor":8.0}` on the full-attention bucket.
- `rope_scaling` is backward-compatible input. `Gemma3TextConfig.convert_rope_params_to_dict` updates `rope_parameters["full_attention"]` and separately fills `sliding_attention`.
- Position IDs default to `arange(current_seq) + past_key_values.get_seq_length()`.
- Cos/sin can be cached per `(layer_type, position, dtype)` for default/linear fixed parameters. Dynamic RoPE variants need explicit guards.

RMSNorm:

```python
def gemma3_rms_norm(x, weight, eps):
    y = x.float() * rsqrt(mean(x.float() ** 2, axis=-1, keepdims=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)
```

## 8. Preprocessing and input packing

Text-only generation:

- Runtime graph inputs: `input_ids[B,T]`, optional `attention_mask[B,T]`, optional `position_ids[B,T]`, optional `past_key_values`.
- Tokenizer/chat template is data-pipeline work.
- Processor is not used for `Gemma3ForCausalLM`.

Image+text processor contract:

- `Gemma3Processor` accepts `images` and `text`.
- Defaults from source processor kwargs:
  - tokenizer: `padding=False`, `return_mm_token_type_ids=True`.
  - images: `do_convert_rgb=True`, `do_pan_and_scan=False`, pan-and-scan min crop 256, max crops 4, activation ratio 1.2.
- Processor config mirror: `image_seq_length=256`, `processor_class="Gemma3Processor"`.
- Preprocessor mirror: `size={"height":896,"width":896}`, `rescale_factor=1/255`, `image_mean=[0.5,0.5,0.5]`, `image_std=[0.5,0.5,0.5]`, `image_processor_type="Gemma3ImageProcessor"`.
- Source class defaults are 224 and ImageNet mean/std, but checkpoint preprocessor config overrides them. DinoML must read checkpoint preprocessor config rather than source defaults for real Gemma3 multimodal.
- Output keys: `input_ids`, `attention_mask` as tokenizer provides, `token_type_ids` when requested, and `pixel_values`. `num_crops` is produced by the image processor but removed from public model inputs by `Gemma3Processor.model_input_names`.
- Pixel tensor shape for sampled multimodal configs: `[num_images_plus_crops, 3, 896, 896]`.

Placeholder conventions:

- Tokenizer supplies `boi_token`, `image_token`, `eoi_token`, and `image_token_id`.
- Each begin-image marker in text is expanded to:

```text
"\n\n" + boi_token + image_token repeated image_seq_length times + eoi_token + "\n\n"
```

- With `image_seq_length=256`, one image produces 256 placeholder positions in the language sequence.
- If no text is supplied, processor creates prompts from begin-image tokens.
- If pan-and-scan inserts crops, the processor rewrites the prompt with text describing original image and crops and adds more begin-image tokens. First integration can require `do_pan_and_scan=False` to avoid this prompt mutation.

Runtime stitching:

- Vision features are generated as `[Nimg,256,H]`.
- Placeholder mask is made from `input_ids == image_token_index` or, when only `inputs_embeds` are given, by comparing embeds to the image token embedding.
- The total number of placeholder embedding elements must equal `image_features.numel()`, otherwise the source raises.
- `masked_scatter` writes flattened image features into placeholder slots in sequence order.
- Image features can be precomputed and cached for repeated prompts if the exact placeholder insertion order is preserved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Gemma3RMSNorm -> RMSNormOnePlusWeight

Source pattern:

```text
x.float() * rsqrt(mean(x.float()**2, dim=-1) + eps)
-> multiply by (1 + weight.float())
-> cast to input dtype
```

Replacement:

```text
RMSNorm(axis=-1, eps, fp32_accum=True, scale=(1 + weight))
```

Preconditions:

- Weight shape matches normalized last dimension.
- No bias term.
- Preserve fp32 accumulation and `(1 + weight)` semantics.

Failure cases:

- Do not share a generic RMSNorm kernel that assumes raw `weight` scale unless the weight is transformed at load time and provenance records it.

Parity test sketch:

- Compare text hidden norms for H 1152/2560/3840/5376 and head norms for D 128/256 in fp32/fp16/bf16.

### Rewrite: text Linear -> GEMM_RCR / bias GEMM

Source pattern:

```text
nn.Linear(in_features, out_features, bias=config.attention_bias or False)
```

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR(weight[out,in]) -> optional bias epilogue -> Reshape
```

Preconditions:

- Dense row-major activation.
- Weight layout is PyTorch `[out_features, in_features]`.
- Bias absent for sampled text configs; guard for `attention_bias=True`.

Shape equations:

- Q: `out=A*D`, K/V: `out=KvH*D`, O: `in=A*D`, not necessarily H.

Failure cases:

- Assuming Q projection output equals H breaks 1B, 4B, and 27B.

Parity test sketch:

- Projection parity per sampled config, including 27B where `D=128` and `query_pre_attn_scalar=168`.

### Rewrite: separate Q/K/V -> grouped or concatenated projection

Source pattern:

```text
q_proj(normed_x), k_proj(normed_x), v_proj(normed_x)
```

Replacement:

```text
GroupedGEMM(q,k,v) or ConcatenatedLinear(H -> (A + 2*KvH) * D) -> split
```

Preconditions:

- Same input tensor, same dtype/device, compatible quantization/residency policy.
- Bias settings match or are handled in split epilogues.

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
```

Failure cases:

- Do not require equal Q/K/V output widths under GQA.
- Quantized constants with different storage policies may need grouped rather than concatenated GEMM.

### Rewrite: eager repeat_kv attention -> native GQA attention

Source pattern:

```text
repeat_kv(k, A/KvH), repeat_kv(v, A/KvH), matmul/softmax/matmul
```

Replacement:

```text
GQAAttention(q[B,A,Q,D], k[B,KvH,K,D], v[B,KvH,K,D], group_size=A/KvH)
```

Preconditions:

- `A % KvH == 0`.
- Backend preserves Gemma3 scale, mask semantics, RoPE-before-cache, and fp32 softmax behavior.

Failure cases:

- Materializing repeated K/V for decode defeats the memory benefit.
- Sliding layers need local-cache/window semantics; full layers need unbounded past.

### Rewrite: SigLIP patch Conv2d -> WindowFlatten+GEMM

Source pattern:

```text
Conv2d(3 -> 1152, kernel=14, stride=14, padding=valid)
-> flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlattenNCHW([B,3,H,W], 14x14 non-overlap)
-> GEMM([B*Npatch, 3*14*14] x [3*14*14, 1152])
-> BiasAdd
-> Reshape([B,Npatch,1152])
```

Preconditions:

- `kernel_size == stride == patch_size`, padding valid/0, dilation 1, groups 1.
- Preserve NCHW flatten order.
- Input height/width divisible by patch size for the optimized no-tail path. PyTorch Conv2d floors tails; if arbitrary sizes are admitted, add a floor/tail guard.

Failure cases:

- NHWC rewrite requires explicit axis and weight-layout transform; keep NCHW semantic graph first.

### Rewrite: projector AvgPool2d -> token-grid reduction

Source pattern:

```text
[B,4096,1152] -> [B,1152,64,64] -> AvgPool2d(4,4) -> [B,1152,16,16] -> [B,256,1152]
```

Replacement:

```text
Reshape tokens to [B,16,4,16,4,1152] -> mean over two 4 axes -> [B,256,1152]
```

Preconditions:

- `image_size=896`, `patch_size=14`, `mm_tokens_per_image=256`, so 64 patches per side and 16 projected tokens per side.
- `kernel_size = patches_per_image / sqrt(mm_tokens_per_image)` is integer.

Failure cases:

- Different `mm_tokens_per_image` or image/patch size must recompute grid and pooling kernel.

### Rewrite: multimodal masked_scatter -> indexed embedding update

Source pattern:

```text
mask = (input_ids == image_token_index)[...,None].expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
```

Replacement:

```text
ImageTokenPositions(input_ids) -> ScatterRows(inputs_embeds, positions, image_features.reshape(-1,H))
```

Preconditions:

- Placeholder count equals `Nimg * mm_tokens_per_image`.
- Scatter order matches row-major scan order used by `masked_scatter`.

Failure cases:

- Inputs provided as `inputs_embeds` without `input_ids` need embedding-equality fallback or should be out of scope initially.

## 10. Kernel fusion candidates

Highest priority:

- RMSNormOnePlusWeight: six normalization uses per text layer if counting Q/K head norms plus four block norms, plus final norm.
- Native hybrid GQA attention with KV cache: required for text-only and multimodal generation; sliding layers need bounded cache.
- RoPE + Q/K head norm + layout staging: every attention layer uses Q/K norm before RoPE.
- Gated GELU MLP fusion: `gelu_tanh(gate) * up` is a large activation path.
- Last-token-only logits: source already supports `logits_to_keep`; avoid full `[B,T,V]` logits during generation.

Medium priority:

- QKV grouped projection with unequal Q/K/V widths.
- SigLIP patch embedding Conv2d-to-GEMM/window flatten.
- SigLIP non-causal flash/SDPA over 4096 image tokens.
- Projector pooling+RMSNorm+projection pipeline.
- Multimodal indexed scatter of image features into token embeddings.
- Image-aware mask construction from `token_type_ids`.

Lower priority:

- Dynamic/pan-and-scan preprocessing inside runtime.
- Attention/logit softcap paths if no target checkpoint enables them.
- Bidirectional attention mode.
- Sequence classification heads.
- SigLIP variable-resolution positional interpolation; sampled Gemma3 uses fixed 896.

## 11. Runtime staging plan

Stage 1: Text config and one-block skeleton.

- Parse `Gemma3TextConfig`, including `head_dim`, `query_pre_attn_scalar`, `layer_types`, `sliding_window`, and old `rope_scaling` fields.
- Load embeddings, norms, projections, MLP, and LM head.
- Implement Gemma3 RMSNorm and text Linear lowering.

Stage 2: Text prefill without cache.

- Implement local/global RoPE buckets and masks.
- Run one decoder block, then full text-only prefill for 1B-style shapes.
- Use eager/repeat-KV fallback first if needed.

Stage 3: Text decode with hybrid cache.

- Implement per-layer cache classes or equivalent metadata:
  - full layers store full `[B,KvH,T,D]`;
  - sliding layers keep `min(T, sliding_window)`.
- Validate position offset, RoPE-before-cache, and `logits_to_keep=1`.

Stage 4: Optimized text attention.

- Add native GQA attention and sliding-window attention backend.
- Add last-token logits and MLP fusion.

Stage 5: Multimodal preprocessing contract.

- Parse processor/preprocessor configs.
- Require `do_pan_and_scan=False` initially.
- Accept preprocessed `pixel_values`, `input_ids`, `attention_mask`, `token_type_ids`.

Stage 6: Vision/projector parity.

- Implement SigLIP patch embedding, learned position add, non-causal encoder, post-layernorm.
- Implement projector pooling/RMSNorm/projection.
- Validate `image_features[Nimg,256,H]`.

Stage 7: Multimodal prefill/decode.

- Implement placeholder count validation and indexed scatter.
- Implement image-aware mask construction from token_type ids.
- Run multimodal prefill logits parity and cached decode where later iterations omit `pixel_values`.

Stage 8: Optimization and production scheduling.

- Conv2d-to-GEMM, native vision attention, image-feature caching, QKV grouping, paged KV cache, continuous batching.

## 12. Parity and validation plan

- Config parser tests:
  - text-only 1B mirror;
  - multimodal 4B/12B mirrors;
  - 27B mirror with omitted defaults filled by source config.
- RMSNorm parity for hidden dims and head dims, checking `(1 + weight)` semantics.
- RoPE parity for `sliding_attention` and `full_attention` buckets, including linear full-attention scaling when configured.
- `layer_types` pattern test: omitted layer types produce 5 sliding + 1 full repeating.
- Projection shape tests for 1B, 4B, 12B, 27B, proving Q/O widths can differ from H.
- GQA attention parity against HF eager fallback for group sizes 2 and 4.
- DynamicCache shape/lifecycle parity for sliding versus full layers over decode steps longer than the sliding window.
- Single decoder layer parity, then N-layer and full text-only prefill parity.
- Cached text decode parity for 2-4 generated tokens with `logits_to_keep=1`.
- Processor contract smoke: one image marker expands to 256 image tokens and produces `token_type_ids`.
- Image processor parity for fixed 896 config: RGB/resize/rescale/normalize to `[N,3,896,896]`.
- SigLIP vision encoder parity at 896 for a small fixed batch.
- Projector parity from synthetic `[N,4096,1152]` and real SigLIP output.
- Placeholder scatter parity, including count mismatch failure.
- Multimodal prefill logits parity and one-step cached decode parity.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-5` for custom math; fp16/bf16 initial `rtol=2e-2, atol=2e-2`, then tighten layerwise. Vision attention over 4096 tokens may need looser end-to-end tolerance until backend math order is fixed.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Text prefill tokens/sec by sequence length and batch size for 1B/4B/12B/27B shapes.
- Decode tokens/sec by cache length, separating sliding and full layers.
- KV cache memory:
  - full layer: `2 * B * KvH * T * D * dtype_bytes`;
  - sliding layer: `2 * B * KvH * min(T, sliding_window) * D * dtype_bytes`.
- Native GQA attention versus repeat-KV fallback memory and latency.
- RoPE generation versus cached cos/sin lookup for local and global buckets.
- RMSNorm/head-norm bandwidth.
- MLP GEMM and gated GELU activation bandwidth.
- LM head full logits versus `logits_to_keep=1`.
- Image processor throughput separately from GPU runtime.
- SigLIP encoder throughput at 896x896, especially patch embedding and 4096-token attention.
- Projector throughput and memory traffic.
- Multimodal end-to-end split: processor, vision, projector, scatter/prefix, language prefill, decode.
- Image-feature caching benefit for repeated prompts over same image.

These are proposed probes, not measurements.

## 14. Skip/defer list

Safe to defer for first generation integration:

- Training, labels/loss, dropout, gradient checkpointing.
- Sequence classification heads.
- Pan-and-scan prompt/crop expansion; require `do_pan_and_scan=False` at first.
- `inputs_embeds` multimodal placeholder detection by embedding equality; require `input_ids` for first multimodal path.
- `use_bidirectional_attention=True` except as a documented config guard.
- Attention/logit softcapping unless a selected checkpoint enables it.
- Dynamic arbitrary image sizes and SigLIP position interpolation beyond fixed 896.
- Multi-GPU TP/PP plans.
- Quantized weights and GGUF integration.
- Beam search/speculative decoding/controller features outside core model graph.

## 15. Final implementation checklist

- [ ] Parse `Gemma3TextConfig` and `Gemma3Config`.
- [ ] Normalize `rope_scaling`/`rope_parameters` into full and sliding RoPE buckets.
- [ ] Generate default `layer_types` when omitted.
- [ ] Load text embeddings, projections, norms, MLP weights, and LM head.
- [ ] Implement Gemma3 RMSNorm with `(1 + weight)`.
- [ ] Implement text Linear/GEMM with `A*D` and `KvH*D` widths.
- [ ] Implement gated `gelu_pytorch_tanh` MLP.
- [ ] Implement local/global RoPE and apply_rope.
- [ ] Implement causal and sliding-window masks.
- [ ] Implement GQA attention fallback and native optimized path.
- [ ] Implement hybrid full/sliding KV cache.
- [ ] Implement `logits_to_keep`.
- [ ] Add text one-block, prefill, and cached decode parity tests.
- [ ] Parse Gemma3 processor and image processor configs.
- [ ] Implement fixed-896 image preprocessing contract or require preprocessed `pixel_values`.
- [ ] Implement SigLIP vision encoder subset used by Gemma3.
- [ ] Add guarded SigLIP patch Conv2d -> GEMM rewrite.
- [ ] Implement multimodal projector pooling/RMSNorm/projection.
- [ ] Implement image placeholder count validation.
- [ ] Implement indexed scatter of image features into text embeddings.
- [ ] Implement token_type_ids image block mask metadata.
- [ ] Add projector, scatter, multimodal prefill, and multimodal decode parity tests.
- [ ] Benchmark text prefill/decode, KV memory, vision encoder, projector, and end-to-end multimodal generation.
