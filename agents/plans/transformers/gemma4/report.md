# Transformers Gemma4 Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Representative official configs inspected from google/gemma-4-E2B,
  google/gemma-4-E2B-it, google/gemma-4-E4B, google/gemma-4-E4B-it,
  google/gemma-4-26B-A4B, google/gemma-4-26B-A4B-it, google/gemma-4-31B,
  google/gemma-4-31B-it, plus tiny-random/gemma-4-moe.

Config source:
  Local source defaults from configuration_gemma4.py.
  Downloaded HF config snapshots are stored in _sources/.

Source files inspected:
  transformers/src/transformers/models/gemma4/configuration_gemma4.py
  transformers/src/transformers/models/gemma4/modeling_gemma4.py
  transformers/src/transformers/models/gemma4/modular_gemma4.py
  transformers/src/transformers/models/gemma4/processing_gemma4.py
  transformers/src/transformers/models/gemma4/image_processing_gemma4.py
  transformers/src/transformers/models/gemma4/image_processing_pil_gemma4.py
  transformers/src/transformers/models/gemma4/video_processing_gemma4.py
  transformers/src/transformers/models/gemma4/feature_extraction_gemma4.py
  transformers/src/transformers/models/gemma4/convert_gemma4_weights.py

Any missing files or assumptions:
  modeling_gemma4.py is the generated runtime source in this checkout.
  modular_gemma4.py should be treated as the future source-edit basis.
  No remote-code files were required for standard Gemma4 classes.
  This is an inference-only CUDA planning audit. No DinoML imports or tests were run.
```

Pinned source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/modeling_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/configuration_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/processing_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/image_processing_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/video_processing_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/feature_extraction_gemma4.py

## 2. High-level architecture

Primary runtime target: any-to-any/multimodal autoregressive generation through `Gemma4ForConditionalGeneration`. Text-only `Gemma4ForCausalLM` is also implemented and is the most useful first bring-up target.

```text
text/image/video/audio preprocessing
  -> token ids with expanded multimodal placeholders
  -> optional image/video patch encoder -> soft vision tokens
  -> optional audio feature extractor + audio encoder -> soft audio tokens
  -> masked_scatter into text embedding stream
  -> Gemma4 text decoder prefill/decode with hybrid sliding/full attention
  -> tied LM head -> optional final logit softcap -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer/chat template, placeholder expansion, image/video resize and patchification, video frame sampling/timestamps, audio waveform to log-mel features.
- Independently cacheable encoders: vision encoder for image/video patches; audio encoder for `input_features`.
- Prefix construction: text embedding, optional per-layer embedding construction, multimodal soft-token replacement, and multimodal block ids for vision-aware masks.
- Prefill: mixed text/image/video/audio sequence through the text decoder.
- Decode: text-only incremental decoding with `past_key_values`; processor-derived multimodal tensors are passed only on the first cached iteration.

## 3. Important config dimensions

Source defaults:

| Field | Source default | Runtime significance |
|---|---:|---|
| text vocab | 262144 | embedding and LM head |
| text hidden | 2304 | source default only; real checkpoints vary |
| text intermediate | 9216 | dense MLP width |
| text layers | 30 | decoder depth |
| query heads / KV heads | 8 / 4 | GQA/MQA required |
| text head dim | 256 | explicit; do not infer from hidden/head count |
| `global_head_dim` | 512 | full-attention layers may use a wider head dim |
| `num_global_key_value_heads` | null | can change KV width for full layers |
| `attention_k_eq_v` | false | if true, full-attention V reuses K projection |
| `num_kv_shared_layers` | 0 | shared-KV layers own no K/V weights |
| `sliding_window` | 512 | local text attention |
| layer pattern | 5 sliding + 1 full, last forced full | when `layer_types` omitted |
| text RoPE | sliding default theta 10000; full proportional theta 1000000, partial factor 0.25 | separate RoPE buckets |
| final logit softcap | null source default; 30 in sampled configs | `tanh(logits / cap) * cap` |
| PLE dim | 256 source default | per-layer embeddings when nonzero |
| MoE | disabled source default | enabled in 26B-A4B |
| vision patch / pool | 16 / 3 | patch pixels are pre-flattened; output soft tokens = patches / 9 |
| vision default H / layers / heads | 768 / 16 / 12 | E2B/E4B; 26B/31B use larger vision H=1152, layers=27 |
| audio default H / layers / heads | 1024 / 12 / 8 | E2B/E4B configs include audio; large configs sampled omit audio_config |

Representative checkpoint sweep:

| Config | Text H | I | Layers | Q/KV heads | D local | Full D | SW | PLE | MoE | Vision | Audio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `google/gemma-4-E2B` | 1536 | 6144 | 35 | 8 / 1 | 256 | 512 | 512 | 256 | no | H=768, L=16, A=12, D=64 | H=1024, L=12 |
| `google/gemma-4-E2B-it` | 1536 | 6144 | 35 | 8 / 1 | 256 | 512 | 512 | 256 | no | same as E2B | same as E2B |
| `google/gemma-4-E4B` | 2560 | 10240 | 42 | 8 / 2 | 256 | 512 | 512 | 256 | no | H=768, L=16, A=12, D=64 | H=1024, L=12 |
| `google/gemma-4-E4B-it` | 2560 | 10240 | 42 | 8 / 2 | 256 | 512 | 512 | 256 | no | same as E4B | same as E4B |
| `google/gemma-4-26B-A4B` | 2816 | 2112 | 30 | 16 / 8 | 256 | 512 | 1024 | 0 | 128 experts, top-8, expert I=704 | H=1152, L=27, A=16, D=72 | omitted |
| `google/gemma-4-31B` | 5376 | 21504 | 60 | 32 / 16 | 256 | 512 | 1024 | 0 | no | H=1152, L=27, A=16, D=72 | omitted |
| `tiny-random/gemma-4-moe` | 8 | 64 | 4 | 8 / 4 | 32 | 512 from source default | 1024 | 0 | 128 experts, top-8, expert I=32 | tiny H=8, L=2 | omitted |

Processor snapshots for `google/gemma-4-{E2B,E4B,31B}-it` agree on:

| Processor field | Value |
|---|---:|
| image soft tokens | 280 |
| image patch size / pool | 16 / 3 |
| image preprocessing | RGB, resize, rescale `1/255`, no normalize |
| video soft tokens per frame | 70 |
| video frames | 32 |
| video preprocessing | RGB, resize, rescale `1/255`, normalize with mean 0/std 1 |
| audio max soft tokens | 750 |
| audio ms/token | 40 |
| audio sampling rate | 16000 |
| audio features | 128 log-mel bins, frame 320, hop 160, FFT 512, mel floor 0.001 |

## 3a. Family variation traps

- Gemma4 is not just Gemma3 with a renamed config. It adds pre-flattened dynamic image/video patches, audio, optional PLE, optional MoE, full-attention global head width, optional K=V projection sharing, and optional shared-KV layers.
- `hidden_size != num_attention_heads * head_dim` in E2B and 26B-A4B. Q/O width is `num_attention_heads * layer_head_dim`, where full layers may use `global_head_dim=512`.
- Full attention and sliding attention can have different K/V head count and head dim. `attention_k_eq_v=True` is source-supported for full attention and removes `v_proj`.
- `num_kv_shared_layers > 0` is source-supported. Shared layers do not own `k_proj`, `v_proj`, `k_norm`, or `v_norm` and consume side-channel `shared_kv_states`.
- E2B/E4B use per-layer embeddings (`hidden_size_per_layer_input=256`); 26B-A4B/31B disable them (`0`).
- 26B-A4B enables MoE after the dense MLP. The dense `intermediate_size=2112` is not the expert width; experts use `moe_intermediate_size=704`, `num_experts=128`, `top_k_experts=8`.
- Large sampled configs set `use_bidirectional_attention="vision"`, so multimodal block ids affect mask construction. Smaller sampled configs use conventional causal masks.
- Vision encoder consumes patch tensors `[B,max_patches,3*16*16]`, not raw NCHW images. Image/video patchification is processor-owned but can be moved into runtime only with exact flatten-order guards.
- Image/video token counts are dynamic per aspect ratio: `num_soft_tokens = num_patches / pooling_kernel_size^2`, capped by supported token budgets.
- Video placeholder expansion includes timestamps and one soft-token block per sampled frame.
- Audio placeholder count is dynamic from waveform duration and capped at 750. The model strips encoder padding before scatter.
- `Gemma4RMSNorm` uses direct `weight`, unlike Gemma3's `(1 + weight)` convention.
- Vision/audio `Gemma4ClippableLinear` may clamp input/output using checkpoint buffers when `use_clipped_linears=True`.
- Initial graph translation should preserve source patch axes and position-id semantics. NHWC/channel-last rewrites need guarded processor-to-encoder ownership.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding gather with scale: text `Embedding(V,H) * sqrt(H)`; PLE `Embedding(V,L*P) * sqrt(P)`.
- Reshape/view/transpose for Q/K/V: `[B,T,H] -> [B,T,A,D] -> [B,A,T,D]`.
- Per-layer embedding reshape: `[B,T,L*P] -> [B,T,L,P]`.
- Boolean masks, `where`, `cumsum`, `roll`, comparisons, `masked_fill`, `masked_scatter`.
- `topk`, `softmax`, expert `one_hot`, `where`, and `index_add_` for MoE fallback/reference.
- Patchification flatten order:
  - image `[C,H,W] -> [patch_h*patch_w, patch*patch*C]` through reshape/permute.
  - video `[F,C,H,W] -> [F,patch_h*patch_w, patch*patch*C]`.
- `pad` for patch and position padding; `-1` position padding guard.
- `unfold`/gather for audio local attention windows and audio feature extraction.
- Last-token/suffix logits slicing through `logits_to_keep`.

Neural network primitives:

- Text dense projections:
  - E2B Q `1536 -> 2048`, K/V local `1536 -> 256`, full K/V if no global override default `1536 -> 512`, O local `2048 -> 1536`, O full `4096 -> 1536`.
  - E4B Q local `2560 -> 2048`, K/V local `2560 -> 512`, O local `2048 -> 2560`, O full `4096 -> 2560`.
  - 26B-A4B Q local `2816 -> 4096`, K/V local `2816 -> 2048`, O local `4096 -> 2816`, O full `8192 -> 2816`.
  - 31B Q local `5376 -> 8192`, K/V local `5376 -> 4096`, O local `8192 -> 5376`, O full `16384 -> 5376`.
- Text MLP: `down(gelu_tanh(gate(x)) * up(x))`, optionally double-wide for shared-KV layers.
- RMSNorm with fp32 accumulation and optional scale-less mode.
- PLE path: model projection, RMSNorm over PLE dim, combine with token PLE, per-layer gate/projection residual.
- MoE path: router RMSNorm without scale, router GEMM, softmax, top-k, weight renormalization, per-expert scale, grouped expert gate/up/down GEMMs, weighted scatter-add.
- LM head tied to text embedding by key contract; final logit softcap when configured.
- Vision patch Linear: `3*16*16 -> vision_hidden`, bias-free.
- Vision transformer: Q/K/V/O clippable linears, Q/K scaled RMSNorm, scale-less V norm, 2D RoPE, gated MLP.
- Vision pooler: position-aware average pooling to requested soft-token length, multiply by `sqrt(vision_hidden)`.
- Multimodal embedder: scale-less RMSNorm then `vision/audio_hidden -> text_hidden`.
- Audio SSCP Conv2d blocks, LayerNorm over channel, ReLU, Linear to audio hidden.
- Audio conformer-like blocks: FFN, local relative attention with softcap, causal depthwise Conv1d + GLU, RMSNorms, output projection.

Attention primitives:

- Text causal self-attention with hybrid sliding/full masks.
- GQA/MQA with native KV heads; avoid materialized repeat for production.
- Full layers may have larger head dim than sliding layers.
- Optional `attention_k_eq_v` and shared-KV layer side channel.
- Vision non-causal attention over patch tokens with 2D RoPE and padding masks.
- Audio blocked local attention over `[B, heads, blocks, chunk, context]` with relative position term and softcap.

Position/rotary/relative-bias ops:

- Text RoPE: separate layer-type buckets; sliding default theta 10000, full proportional theta 1000000 with `global_head_dim` key.
- Vision 2D RoPE: independent frequencies per x/y dimension, concatenated cos/sin.
- Vision learned x/y position table lookup via one-hot matmul; padding positions zeroed.
- Audio sinusoidal relative position table and relative shift.

Generation/cache ops:

- `DynamicCache(config)` for text.
- Per-layer cache update only for non-shared text layers.
- `shared_kv_states` side-channel for shared layers and last non-shared layer per layer type.
- `prepare_inputs_for_generation` drops multimodal tensors after first cached iteration and clears `mm_token_type_ids` during cached decode.

Preprocessing-coupled ops:

- Image/video aspect-ratio-preserving resize computed from `patch_size`, `max_soft_tokens`, and `pooling_kernel_size`.
- Image/video patchify and position-id generation are model-coupled.
- Audio feature extraction is custom NumPy STFT/log-mel, not generic HF spectrogram.
- Placeholder expansion for image/video/audio and `mm_token_type_ids` creation.

Quantized/packed metadata:

- No in-library source-coupled quantized runtime path was found in `modeling_gemma4.py`.
- Conversion source declares variants and MoE packed expert weight mapping. GGUF/GPTQ/AWQ/NVFP4 mirrors should be treated as separate loader/provider audits, not as native Gemma4 source behavior.

## 5. Layer/block breakdown

Text setup:

```text
input_ids[B,T] -> embed_tokens[B,T,H] * sqrt(H)
if PLE:
  token_ple = embed_tokens_per_layer(input_ids).reshape(B,T,L,P)
  model_ple = RMSNorm(Linear(H -> L*P)(inputs_embeds) / sqrt(H)).reshape(B,T,L,P)
  per_layer_inputs = (token_ple + model_ple) / sqrt(2)
if use_cache and no past: past_key_values = DynamicCache(config)
position_ids = arange(T) + past_seen_tokens
mask_map = {full_attention, sliding_attention}
rope_map = {layer_type: rotary_emb(hidden_states, position_ids, layer_type)}
```

Text decoder layer:

```text
residual = x
y = RMSNorm(x)
q = Linear(H -> A*D_layer)(y).view(B,T,A,D_layer)
q = RMSNorm(q)
q = RoPE(q).transpose(1,2)
if shared_kv_layer:
  k,v = shared_kv_states[layer_type]
else:
  k = Linear(H -> KVH*D_layer)(y).view(B,T,KVH,D_layer)
  v = Linear(H -> KVH*D_layer)(y).view(B,T,KVH,D_layer) or k when attention_k_eq_v
  k = RMSNorm(k); k = RoPE(k).transpose(1,2)
  v = RMSNorm_no_scale(v).transpose(1,2)
  k,v = cache.update(k,v,layer_idx)
  maybe shared_kv_states[layer_type] = k,v
a = GQA(q,k,v, mask_for_layer_type, scale=1.0, sliding_window=maybe)
a = Linear(A*D_layer -> H)(a)
a = RMSNorm(post_attention)(a)
x = residual + a

residual = x
y = RMSNorm(pre_ffn)(x)
dense = MLP(y)
if MoE:
  dense = RMSNorm_1(dense)
  route_input = residual.reshape(-1,H)
  topk = Router(route_input)
  expert = Experts(RMSNorm(route_input), topk).reshape(B,T,H)
  expert = RMSNorm_2(expert)
  y = dense + expert
y = RMSNorm(post_ffn)(y)
x = residual + y

if PLE:
  x = x + RMSNorm(Linear(P -> H)(act(Linear(H -> P)(x)) * per_layer_input_i))
x = x * layer_scalar
```

Vision path:

```text
pixel_values[B,max_patches,768] already patchified and padded
position_ids[B,max_patches,2], padding where (-1,-1)
patch_embed = Linear(768 -> Hv)(2 * (pixel_values - 0.5))
pos = one_hot(x/y positions) @ position_embedding_table, summed over x/y
x = patch_embed + zero_padded(pos)
for Lv layers:
  RMSNorm -> noncausal 2D-RoPE attention -> RMSNorm -> residual
  RMSNorm -> gated MLP -> RMSNorm -> residual
pool by patch positions to output_length = max_patches / 9
strip padding tokens
RMSNorm_no_scale -> Linear(Hv -> H_text)
```

Audio path:

```text
waveform CPU pipeline -> input_features[B,T,128], input_features_mask[B,T]
unsqueeze -> [B,1,T,128]
Conv2d 1->128 stride 2 + LayerNorm(channel) + ReLU
Conv2d 128->32 stride 2 + LayerNorm(channel) + ReLU
flatten frequency/channel -> Linear((128/4)*32 -> 1024)
for 12 layers:
  FFN residual with 0.5 scale
  RMSNorm -> blocked local relative attention with tanh softcap 50 -> RMSNorm residual
  causal depthwise Conv1d + GLU residual
  FFN residual with 0.5 scale
  RMSNorm
Linear(1024 -> 1536)
RMSNorm_no_scale -> Linear(1536 -> H_text)
strip invalid padding positions before scatter
```

## 6. Attention requirements

Text attention:

- Causal self-attention only; no cross-attention module.
- Hybrid layer schedule from config: E2B uses 4 sliding then 1 full repeated; E4B/26B/31B use 5 sliding then 1 full repeated.
- Q shape is `[B,A,Q,D_layer]`; K/V shape is `[B,KVH,K,D_layer]`.
- Scaling is exactly `1.0`, not `head_dim^-0.5`.
- Q/K RMSNorm and RoPE occur before cache update. Cached keys are already normalized and rotated.
- V uses RMSNorm with `with_scale=False`.
- Sliding layers pass `sliding_window`; full layers do not.
- Full layers use `global_head_dim` when set. This changes Q/O widths and RoPE dimensions.
- If `attention_k_eq_v` is true for a non-sliding layer, value projection is absent and V starts from K projection output before scale-less V norm.
- Shared-KV layers consume `shared_kv_states[layer_type]` even during decode because a sliding cache may not retain full states. DinoML needs a cache manifest separating physical cache owners from logical layers.
- Eager fallback repeats K/V to Q head count; production should use native GQA.
- FlashAttention/SDPA compatibility requires native GQA, hybrid masks, sliding-window support, scale `1.0`, already-rotated K cache, and support for block-sequence ids when `use_bidirectional_attention="vision"`.

Vision attention:

- Non-causal self-attention over patch tokens with padding mask.
- Uses Q/K/V clippable linear projections, Q/K RMSNorm, scale-less V norm, 2D RoPE, scale `1.0`, and generic attention backend.
- No KV cache.

Audio attention:

- Encoder local attention, not autoregressive KV-cache generation.
- Chunk size 12, left context 12 (`attention_context_left=13`), right context 0 by default.
- Q/K/V heads are 8 with D=`hidden_size/heads` = 128 for E2B/E4B audio.
- Query is scaled by `(head_dim^-0.5 / log(2)) * softplus(per_dim_scale)`; key by `log(1+e)/log(2)`.
- Relative position logits are added before tanh softcap.
- Mask is converted from 4D bidirectional/local mask to blocked 5D format.
- Softmax is fp32 and output is projected by clippable linear.

## 7. Position encoding and custom math

Text RoPE:

```python
def gemma4_text_rope(config, layer_type, position_ids, dtype):
    params = config.rope_parameters[layer_type]
    dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim
    # proportional full-attention routes through ROPE_INIT_FUNCTIONS with head_dim_key="global_head_dim"
    inv = init_rope(params, dim)
    freqs = (inv[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
```

Vision 2D RoPE:

```python
def gemma4_vision_rope(position_ids_xy, head_dim, theta=100.0):
    spatial_dim = head_dim // 2
    inv = 1.0 / (theta ** (arange(0, spatial_dim, 2).float() / spatial_dim))
    parts = []
    for axis in [0, 1]:
        freqs = inv[None, :, None] @ position_ids_xy[:, None, :, axis].float()
        parts.append(cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1))
    emb = cat(parts, dim=-1)
    return emb.cos(), emb.sin()
```

Audio relative attention:

```python
def gemma4_audio_logits(q, k_context, rel_k, softcap):
    content = q @ k_context.transpose(-1, -2)
    rel = relative_shift(q.reshape(B, H, -1, D) @ rel_k.transpose(-1, -2))
    logits = content + rel
    return torch.tanh(logits / softcap) * softcap
```

Precomputable:

- Text inverse frequencies and fixed cos/sin tables per layer type when RoPE params are static.
- Vision x/y position embeddings for bounded patch grids and the 2D RoPE tables.
- Audio relative sinusoidal table and projected relative keys for fixed weights/context.

Dynamic:

- Text decode position offsets.
- Image/video patch positions and padding masks.
- Audio frame masks and block context masks.

## 8. Preprocessing and input packing

Text:

- Inputs are `input_ids`, optional `attention_mask`, optional `position_ids`, optional `past_key_values`.
- Tokenizer/chat template is outside the neural graph.

Image:

- Processor resizes preserving aspect ratio under a patch/token budget, rescales by `1/255`, does not normalize in sampled processor configs.
- Output `pixel_values` is `[num_images, max_patches, 768]`, already patchified in channel-last patch-flatten order from source `permute(1,3,2,4,0)`.
- `image_position_ids` is `[num_images, max_patches, 2]`, padded with `(-1,-1)`.
- Default max patches are `280 * 3^2 = 2520`; soft tokens per image are actual patches divided by 9.
- Prompt replacement is `boi + image_token * n + eoi`.

Video:

- Processor samples 32 frames by default, resizes/pads each frame like images with max 70 soft tokens per frame.
- Output `pixel_values_videos` is `[num_videos, num_frames, max_patches, 768]`.
- `video_position_ids` is `[num_videos, num_frames, max_patches, 2]`.
- Prompt replacement includes `mm:ss` timestamps and one block per sampled frame.
- Model flattens video and frame axes before reusing the vision tower.

Audio:

- Feature extractor runs in NumPy CPU pipeline.
- Sampling rate 16 kHz, max 480000 samples, pad to multiple of 128.
- Semicausal left pad of 160 samples before framing.
- Frame length 320, hop 160, FFT 512, 128 mel bins, HTK mel, log with floor 0.001.
- Output `input_features[B,T,128]` and `input_features_mask[B,T]`.
- Placeholder count is computed from waveform length by matching mel frames and two stride-2 SSCP convs, capped at 750.

Multimodal stitch:

- Placeholder masks come from `input_ids == image/video/audio_token_id`, or an exact embedding equality fallback if only `inputs_embeds` is supplied.
- Multimodal token ids are replaced with PAD before text embedding.
- PLE, when enabled, is computed from PAD-replaced ids before soft-feature scatter.
- `masked_scatter` writes flattened image/video/audio features in row-major mask order.
- Count checks compare `inputs_embeds[mask].numel()` with feature tensor `numel()`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Gemma4 RMSNorm

Source pattern:

```text
x.float() * pow(mean(x.float()**2) + eps, -0.5) * optional weight.float()
```

Replacement:

```text
RMSNorm(axis=-1, eps, fp32_accum=True, scale=weight or none)
```

Preconditions:

- Weight shape equals normalized dimension.
- Preserve scale-less mode for V norm, router norm, and multimodal pre-projection norm.

Failure cases:

- Do not apply Gemma3 `(1 + weight)` transform; Gemma4 uses direct `weight`.

Parity test sketch:

- Hidden, head, PLE, and scale-less norm parity in fp32/fp16/bf16.

### Rewrite: Q/K/V projections to grouped GEMM

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
GroupedGEMM or packed Linear -> split(Q,K,V)
```

Preconditions:

- Same input tensor and dtype.
- Account for layer type: full layers may have larger `D`; `attention_k_eq_v` omits V projection.
- Shared-KV layers must not expect K/V weights.

Failure cases:

- Concatenating local and full layer weights under one static width is wrong.
- Assuming `A*D == H` is wrong for sampled configs.

### Rewrite: native GQA attention

Source pattern:

```text
repeat_kv(k,v) -> q @ k.T -> mask -> fp32 softmax -> @ v
```

Replacement:

```text
GQA(q[B,A,Q,D], k/v[B,KVH,K,D], group=A/KVH)
```

Preconditions:

- `A % KVH == 0`.
- Preserve scale `1.0`, RoPE-before-cache, layer mask, and fp32 softmax.

Failure cases:

- Shared-KV and sliding-window caches need explicit state ownership; do not hide this in generic KV cache.

### Rewrite: patchified image/video Linear as patch GEMM

Source pattern:

```text
processor patchify -> Linear(768 -> Hv)
```

Replacement:

```text
optional runtime PatchifyNCHW/NFCHW -> GEMM_RCR
```

Preconditions:

- Input is controlled from processor through patch embedder.
- `patch_size=16`, `channels=3`, flatten order `[patch_y, patch_x, channel]` inside each patch.
- Height/width divisible by patch size after processor resize.

Failure cases:

- Do not silently swap to NHWC without rewriting patch flatten order and position ids.

### Rewrite: multimodal masked_scatter to indexed row copy

Source pattern:

```text
mask = (input_ids == token_id)[...,None].expand_as(inputs_embeds)
inputs_embeds.masked_scatter(mask, features)
```

Replacement:

```text
positions = nonzero(input_ids == token_id)
ScatterRows(inputs_embeds, positions, features.reshape(-1,H))
```

Preconditions:

- Placeholder count exactly equals feature rows.
- Row-major nonzero order matches source mask flatten order.
- First integration should require `input_ids`, not embedding-equality fallback.

Failure cases:

- Ragged mixed modality prompts with dynamic token counts need capacity and ordering guards.

### Rewrite: MoE expert loop to grouped expert GEMM

Source pattern:

```text
router -> topk -> Python expert loop -> index_add_
```

Replacement:

```text
TopKRouter -> token/expert bucketization -> grouped GEMM gate_up -> activation*multiply -> grouped down -> scatter_add
```

Preconditions:

- `enable_moe_block=True`.
- Expert weights layout: `gate_up_proj[E, 2*moe_I, H]`, `down_proj[E, H, moe_I]`.
- Top-k weights are normalized and then multiplied by `per_expert_scale[top_k_index]`.

Failure cases:

- Dense E2B/E4B/31B should not pay MoE routing overhead.

## 10. Kernel fusion candidates

Highest priority:

- Text GQA attention with hybrid full/sliding cache and scale `1.0`; repeat-KV fallback is too expensive.
- RMSNorm variants, including scale-less norm and head norms.
- Text MLP SwiGLU/GELU gated path.
- Last-token-only LM head plus final softcap.
- PLE path for E2B/E4B: projection, norm, per-layer gate/projection residual.
- MoE grouped expert path for 26B-A4B.

Medium priority:

- QKV grouped projection with layer-type-specific widths.
- Vision patch Linear and 2D RoPE attention over padded dynamic patch sequences.
- Vision pooler plus multimodal embedder.
- Multimodal indexed scatter and block-id mask construction.
- Audio local attention with relative shift and tanh softcap.
- Audio SSCP conv + projection.

Lower priority:

- Moving image/video/audio preprocessing into GPU runtime.
- Embedding-equality fallback for `inputs_embeds`-only multimodal calls.
- Returning full attention tensors for optimized attention.
- Clipped-linear clamp fusion; needed only if target checkpoints use non-infinite clamp buffers.

## 11. Runtime staging plan

Stage 1: text-only dense decoder skeleton.

- Parse `Gemma4TextConfig`, layer types, head dims, RoPE params, final softcap.
- Load text embeddings, norms, dense projections, dense MLP, LM head.
- Implement RMSNorm and GEMM lowering with explicit Q/O widths.

Stage 2: text prefill without cache.

- Implement RoPE buckets, full/sliding masks, GQA attention fallback, and suffix logits.
- Validate E2B/E4B dense configs first.

Stage 3: decode cache.

- Implement hybrid cache, full/sliding layer ownership, RoPE offset, and `logits_to_keep=1`.
- Add shared-KV admission but reject nonzero `num_kv_shared_layers` until implemented.

Stage 4: E2B/E4B PLE.

- Add token PLE, context PLE projection, per-layer residual gate/projection.

Stage 5: multimodal prefix with precomputed features.

- Accept projected image/video/audio features and implement placeholder validation, indexed scatter, and vision-aware mask ids.

Stage 6: owned vision encoder.

- Implement patchified input ABI, learned 2D positions, 2D RoPE vision encoder, position-aware pooling, and projection.

Stage 7: owned audio encoder.

- Keep feature extraction CPU first; implement SSCP conv, blocked local relative attention, light conv, and output projection.

Stage 8: MoE 26B-A4B.

- Implement router/top-k and grouped expert GEMMs.

Stage 9: optimized kernels and production scheduling.

- Native attention kernels, MoE token dispatch, feature caching, paged KV cache, and continuous batching.

## 12. Parity and validation plan

- Config parser tests for E2B, E4B, 26B-A4B, 31B, and tiny-random MoE snapshots.
- Shape tests proving local/full projection widths differ and `A*D` can differ from hidden size.
- RMSNorm parity for scaled and scale-less modes.
- Text RoPE parity for sliding default and full proportional buckets.
- Full/sliding mask parity including `use_bidirectional_attention="vision"` block ids.
- Dense one-layer text parity, then full prefill parity for E2B/E4B.
- Cached decode parity over several tokens with `logits_to_keep=1`.
- PLE parity for E2B/E4B against source `get_per_layer_inputs` and `project_per_layer_inputs`.
- Multimodal scatter parity for image, video, audio, and mixed prompts with count mismatch failures.
- Image processor contract parity for aspect-ratio resize, patch flatten order, position ids, and soft-token count.
- Vision encoder block and pooler parity on synthetic patch tensors.
- Audio feature extractor parity on fixed waveforms, then SSCP conv/audio layer parity.
- MoE router and expert parity for tiny-random and 26B-A4B shapes.
- Suggested tolerances: fp32 unit ops `rtol=1e-5, atol=1e-5`; fp16/bf16 block/logit parity initially `rtol=2e-2, atol=2e-2`, tightening after attention math order is fixed.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Text prefill throughput by batch and sequence length for E2B/E4B/31B.
- Decode tokens/sec by cache length, split by sliding and full layers.
- KV memory by physical owner layer; compare shared-KV manifest versus naive per-layer allocation when nonzero sharing is admitted.
- Native GQA versus repeated-KV fallback.
- PLE overhead for E2B/E4B.
- Full-vocab logits versus `logits_to_keep=1`.
- 26B-A4B router/top-k and grouped expert GEMM dispatch overhead.
- Image processor throughput, patch count distribution, and vision encoder throughput by soft-token budget.
- Video throughput split by frame sampling/patchify, vision tower, and prefix insertion.
- Audio feature extraction CPU throughput and audio encoder throughput by waveform duration.
- End-to-end multimodal split: preprocessing, encoders, scatter/prefix, text prefill, decode.
- Quantized loader/provider comparison for external GGUF/NVFP4/AWQ mirrors once a loader target is chosen.

## 14. Skip/defer list

Safe to defer for first integration:

- Training, loss, dropout, gradient checkpointing.
- Beam search cache reorder beyond generic cache support.
- `inputs_embeds`-only reverse embedding/placeholder fallback.
- Full audio/video preprocessing on GPU.
- Returning exact attention weights from optimized attention.
- Shared-KV configs if no selected checkpoint sets `num_kv_shared_layers > 0`; keep an explicit reject guard.
- `attention_k_eq_v=True` until a target checkpoint requires it; keep an explicit reject guard.
- Clipped-linears with finite clamp buffers until checkpoint inspection proves they matter.
- Quantized/GGUF/NVFP4/AWQ loader paths; route to separate provider audits.
- Multi-GPU TP/EP/PP plans.

## 15. Final implementation checklist

- [ ] Parse `Gemma4Config`, `Gemma4TextConfig`, `Gemma4VisionConfig`, and `Gemma4AudioConfig`.
- [ ] Normalize layer-type-specific RoPE parameters.
- [ ] Preserve explicit local/full head dims and KV head counts.
- [ ] Load tied token embedding / LM head without breaking alias identity.
- [ ] Implement Gemma4 RMSNorm with optional scale-less mode.
- [ ] Implement scaled token embeddings.
- [ ] Implement text Linear/GEMM with `A*D_layer` and `KVH*D_layer` widths.
- [ ] Implement gated text MLP.
- [ ] Implement PLE lookup/projection/residual path.
- [ ] Implement text RoPE before cache update.
- [ ] Implement full/sliding masks and vision block-id masks.
- [ ] Implement native GQA attention fallback and optimized path.
- [ ] Implement hybrid full/sliding KV cache.
- [ ] Add guards for shared-KV and `attention_k_eq_v` until supported.
- [ ] Implement final norm, tied LM head, `logits_to_keep`, and final softcap.
- [ ] Implement multimodal placeholder count validation and indexed scatter.
- [ ] Implement image/video patch tensor ABI and position-id guards.
- [ ] Implement Gemma4 vision encoder, 2D RoPE, pooler, and multimodal projection.
- [ ] Implement CPU audio feature extraction contract or require precomputed features.
- [ ] Implement audio SSCP conv, blocked local attention, light conv, and projection.
- [ ] Implement MoE router/top-k/grouped experts for 26B-A4B.
- [ ] Add config, one-block, prefill, decode, multimodal prefix, vision, audio, and MoE parity tests.
- [ ] Benchmark text prefill/decode, KV memory, PLE, MoE, vision, audio, and end-to-end multimodal generation.

