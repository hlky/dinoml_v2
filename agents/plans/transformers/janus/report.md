# Janus Transformers family audit

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id:
  Native in-library scope: deepseek-community/Janus-Pro-1B and deepseek-community/Janus-Pro-7B
  Out-of-scope historical/remote-code examples: deepseek-ai/Janus-Pro-1B,
    onnx-community/Janus-Pro-1B-ONNX
Config source:
  agents/plans/transformers/janus/_sources/*.json
Source files inspected:
  X:/H/transformers/src/transformers/models/janus/configuration_janus.py
  X:/H/transformers/src/transformers/models/janus/modeling_janus.py
  X:/H/transformers/src/transformers/models/janus/modular_janus.py
  X:/H/transformers/src/transformers/models/janus/processing_janus.py
  X:/H/transformers/src/transformers/models/janus/image_processing_janus.py
  X:/H/transformers/src/transformers/models/janus/image_processing_pil_janus.py
  X:/H/transformers/src/transformers/models/llama/modeling_llama.py
Any missing files or assumptions:
  modeling_janus.py is generated from modular_janus.py. Future Transformers edits
  should usually be checked against modular_janus.py, but runtime behavior was
  confirmed in generated modeling_janus.py.
```

Raw config URLs used:

- https://huggingface.co/deepseek-community/Janus-Pro-1B/raw/main/config.json
- https://huggingface.co/deepseek-community/Janus-Pro-7B/raw/main/config.json
- https://huggingface.co/deepseek-community/Janus-Pro-1B/raw/main/preprocessor_config.json
- https://huggingface.co/deepseek-community/Janus-Pro-7B/raw/main/preprocessor_config.json
- https://huggingface.co/deepseek-community/Janus-Pro-1B/raw/main/generation_config.json
- https://huggingface.co/deepseek-community/Janus-Pro-7B/raw/main/generation_config.json
- https://huggingface.co/deepseek-ai/Janus-Pro-1B/raw/main/config.json
- https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX/raw/main/config.json

Primary runtime target for this report: JanusForConditionalGeneration inference
covering image-text understanding/text generation first, then text-to-image VQ
code generation and VQ decoder image reconstruction.

## 2. High-level architecture

Janus is a multimodal decoder stack:

```text
image preprocessing -> Janus vision encoder -> MLP aligner -> placeholder scatter
  -> Llama decoder prefill/decode -> text logits/sampling

text prompt with BOI -> Llama decoder with CFG batch doubling -> VQ-code head
  -> 576 autoregressive image codes -> VQ decoder -> image postprocess
```

Stage decomposition:

- CPU/data pipeline: tokenizer/chat template, placeholder expansion, image
  resize/pad/rescale/normalize, output postprocess to uint8/PIL.
- Independently cacheable understanding branch: vision encoder plus aligner
  produces `[B, 576, text_hidden]` image features for fixed 384x384 inputs.
- Prefix construction: text embeddings are masked-scattered at positions whose
  `input_ids == config.image_token_id`; the number of placeholders must exactly
  match image features.
- Decoder prefill/decode: delegated to the nested Llama `AutoModel`; Janus
  itself does not implement decoder attention.
- Text output: `lm_head(hidden)` over text vocab.
- Image output: custom `generate(generation_mode="image")` bypasses `lm_head`,
  uses a VQ-code head of size 16384, feeds sampled code embeddings back into
  the Llama cache, then decodes codes through the VQ-VAE decoder.

## 3. Important config dimensions

| Field | Janus-Pro-1B | Janus-Pro-7B | Source |
|---|---:|---:|---|
| text hidden size | 2048 | 4096 | config.json |
| text layers | 24 | 30 | config.json |
| text attention heads | 16 | 32 | config.json |
| text KV heads | 16 | 32 | config.json |
| text head dim | 128 | 128 | config.json |
| text intermediate size | 5632 | 11008 | config.json |
| text vocab size | 102400 | 102400 | config.json |
| max positions | 16384 | 16384 | config.json |
| RoPE theta/scaling | 10000 / null | 10000 / null | config.json |
| dtype | bfloat16 | bfloat16 | config.json |
| image token id | 100581 | 100594 | config.json |
| BOI token id for image generation | 100003 | 100016 | generation_config.json |
| pad token id | 100002 | 100015 | generation_config.json |
| guidance scale | 5 | 5 | generation_config.json |
| vision image size / patch | 384 / 16 | 384 / 16 | config.json |
| vision tokens | 576 | 576 | config.json |
| vision hidden / layers / heads | 1024 / 24 / 16 | 1024 / 24 / 16 | config.json |
| vision aligner | 1024->2048, depth 2 | 1024->4096, depth 2 | config.json/source |
| VQ codebook size / dim | 16384 / 8 | 16384 / 8 | config.json |
| VQ latent grid | 24x24 | 24x24 | derived by JanusConfig |
| image generation head | 2048->2048->16384 | 4096->4096->16384 | config/source |

Representative checkpoint sweep:

| Checkpoint/config | Native model_type | Operator-significant notes | Scope |
|---|---|---|---|
| deepseek-community/Janus-Pro-1B main | janus | Native JanusForConditionalGeneration, full expanded config, 2048 text hidden | in scope |
| deepseek-community/Janus-Pro-7B main | janus | Same vision/VQ topology, 4096 text hidden, different image/BOI/pad token IDs | in scope |
| deepseek-community/Janus-Pro-1B initial config | janus | Omits many fields; current config defaults supply JanusVisionConfig and JanusVQVAEConfig fields | compatibility note |
| deepseek-ai/Janus-Pro-1B | multi_modality | Remote-code-era `aligner_config`, `gen_vision_config`, `language_config`; not read by native JanusConfig | out of scope for native source |
| onnx-community/Janus-Pro-1B-ONNX | multi_modality | Legacy processor `VLChatProcessor`; useful for export comparison but not native Janus source | out of scope for native source |

## 3a. Family variation traps

- Native Janus uses `model_type: "janus"` with `text_config`,
  `vision_config`, and `vq_config`. Remote-code-era `model_type:
  "multi_modality"` configs are structurally different and should be rejected
  or routed to a separate audit.
- `image_token_id` differs between 1B and 7B and is not the same as the BOI
  token used to start image generation.
- The processor hardcodes `num_image_tokens = 576`; native configs also carry
  576 through `vision_config.num_image_tokens`. Treat mismatch as an admission
  error.
- The VQ latent grid is derived from `vision_config.image_size // patch_size`,
  so the code count is `num_patches ** 2`. For current checkpoints this is
  24x24 = 576.
- Image VQ codes are not normal text-vocabulary logits. Image generation uses a
  separate `generation_head` with 16384 outputs and a separate
  `generation_embeddings` table.
- Text decoder is delegated to Llama. Decoder cache, RoPE, RMSNorm, SwiGLU, and
  causal attention parity belong to the Llama backend contract.
- The vision branch is NCHW in source. NHWC/channel-last can be a local
  optimization around conv/attention/MLP regions only if every permute, flatten,
  and downstream consumer is guarded.
- VQ decoder is convolutional NCHW and uses GroupNorm plus swish-like
  `x * sigmoid(x)`, nearest upsample, asymmetric pad before stride-2 conv, and
  spatial BMM attention at the lowest resolution.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensors, `permute`, `flatten(2)`, `transpose(1,2)`, reshape/view,
  masked scatter into token embeddings, boolean equality mask, repeat, cat,
  argmax, multinomial for sampling path, clip/cast for postprocess.
- Layout-sensitive VQ decode: NCHW conv body returns NCHW, then `permute(0,2,3,1)`
  for `decode_image_tokens`.

Neural network primitives:

- Embedding for text tokens, positions, and image-code feedback.
- Conv2d patch embedding: `Conv2d(3 -> 1024, kernel=16, stride=16, padding=valid)`.
- Linear projections: vision MHA `1024 -> 1024` Q/K/V/O with bias; aligner
  `1024 -> H -> H`; Llama projections from the nested text config; VQ head
  `H -> H -> 16384`; VQ aligner `8 -> H -> H`.
- LayerNorm in vision, RMSNorm in Llama, GroupNorm(32) in VQ-VAE.
- GELU in vision/aligners/VQ head; SiLU/SwiGLU in Llama; explicit
  `x * sigmoid(x)` in VQ blocks.
- Conv2d 3x3/1x1 VQ encoder/decoder, nearest upsample, constant pad.

Attention primitives:

- Vision encoder noncausal full self-attention, MHA, 16 heads, head dim 64.
- Llama causal self-attention, MHA for current configs because KV heads equal
  attention heads, head dim 128, RoPE before cache update.
- VQ spatial attention at lowest latent resolution via BMM softmax BMM over
  `height * width`.

Generation/cache ops:

- Delegated Llama `Cache` with dynamic/static variants via Transformers.
- Text generation uses standard GenerationMixin path.
- Image generation uses custom loop of exactly 576 steps, static cache by
  default, CFG batch doubling, VQ-code head, softmax/multinomial or argmax.

Scatter/indexed update ops for multimodal embedding stitch:

- `input_ids == image_token_id` mask.
- Expand mask over hidden dimension.
- `inputs_embeds.masked_scatter(mask, image_features)`.
- Runtime check that placeholder element count equals image feature element count.

Discrete codebook / tokenizer ops:

- VQ codebook embedding `Embedding(16384, 8)`.
- Decode maps code indices `[B, 576]` to `[B, 8, 24, 24]` after L2
  normalization of codebook vectors.
- Image code logits are over codebook indices 0..16383, not text token IDs.

Preprocessing-coupled ops:

- Resize longest side to 384 with bicubic, min side at least 14.
- Pad to square with background color 127/127/127.
- Rescale by 1/255 and normalize by mean/std `[0.5, 0.5, 0.5]`.
- Processor expands each `<image_placeholder>` to
  `<begin_of_image>` + 576 placeholders + `<end_of_image>`.

## 5. Layer/block breakdown

Vision input path:

```text
pixel_values [B,3,384,384] NCHW
  -> Conv2d(3,1024,k=16,s=16) -> [B,1024,24,24]
  -> flatten/transpose -> [B,576,1024]
  -> learned absolute position add
  -> 24 x JanusVisionEncoderLayer
  -> JanusVisionAlignerMLP -> [B,576,H]
```

Vision encoder layer, repeated 24 times:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(1024 -> 1024, bias=True)
q,k optional LayerNorm when use_qk_norm=True; current configs false
x = noncausal attention(q,k,v)
x = residual + Linear(1024 -> 1024)(x)
residual = x
x = LayerNorm(x)
x = Linear(1024 -> 4096) -> GELU -> Linear(4096 -> 1024)
x = residual + x
```

Text decoder block is the nested Llama block, repeated 24 or 30 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> heads*128, bias=False)
k,v = Linear(H -> kv_heads*128, bias=False)
q,k = RoPE(q,k)
k,v = cache.update(k,v) when cache is present
x = causal_attention(q,k,v, mask)
x = residual + Linear(heads*128 -> H, bias=False)(x)
residual = x
x = RMSNorm(x)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
x = residual + x
```

Understanding forward:

```text
inputs_embeds = token_embedding(input_ids)
image_features = aligner(vision_model(pixel_values).last_hidden_state)
inputs_embeds = masked_scatter(inputs_embeds, input_ids == image_token_id, image_features)
hidden = language_model(inputs_embeds, attention_mask, position_ids, cache)
logits = lm_head(hidden[:, requested_logits_positions, :])
```

Image generation step:

```text
conditional/unconditional prompt embeddings -> language_model(cache)
hidden = last hidden state
scores = Linear(H -> H) -> GELU -> Linear(H -> 16384)
scores = CFG/logits processors(scores)
next_code = sample_or_argmax(scores)
inputs_embeds = Embedding(16384, 8)(next_code) -> Linear(8 -> H) -> GELU -> Linear(H -> H)
```

VQ decode:

```text
codes [B,576] -> codebook embedding [B,576,8] -> L2 normalize
  -> view [B,24,24,8] -> permute [B,8,24,24]
  -> 1x1 post_quant_conv 8->256
  -> convolutional decoder with ResNet/attention/upsample blocks
  -> [B,3,384,384] -> NHWC for postprocess
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention only.
- MHA, no GQA/MQA; 16 heads, head dim 64.
- Optional Q/K LayerNorm exists in source but current configs set
  `use_qk_norm=false`.
- Uses Transformers attention interface, so SDPA/eager backend selection can
  apply if admitted. Dropout is zero for inference.

Text decoder attention:

- Causal self-attention delegated to Llama.
- Current 1B and 7B configs are MHA: `num_key_value_heads == num_attention_heads`.
- Cache tensors before repeat expansion are `[B, kv_heads, cache_len, 128]`.
  In image generation, Janus doubles batch for CFG, so cache batch is `2*B`.
- Keys are cached after RoPE, because Llama applies RoPE before
  `past_key_values.update`.
- No sliding window, local attention, ALiBi, or packed/varlen metadata appears
  in native Janus config/source.
- Fused attention parity must preserve Llama order: projection, view/transpose,
  RoPE in fp-compatible dtype, cache update, mask addition/backend attention,
  output reshape/contiguous, output projection.

VQ spatial attention:

- Noncausal dense attention inside the VQ-VAE lowest-resolution blocks.
- Source uses conv Q/K/V, reshape to spatial sequence, `torch.bmm`, scale by
  `channels ** -0.5`, `softmax(dim=2)`, second `bmm`, conv output projection.
- This is not a decoder KV cache and should not share the Llama cache ABI.

## 7. Position encoding and custom math

Vision position encoding:

- Learned absolute position embedding of length 576 for 384x384 and patch 16.
- Optional interpolation path reshapes `[1, 576, 1024]` to square grid,
  permutes to NCHW, bicubic interpolates, then flattens back. Current processor
  fixes 384x384, so first integration can reject interpolation.

Llama RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

VQ codebook decode:

```python
z = embedding(image_tokens)          # [B,576,8]
z = normalize(z, p=2, dim=-1)
z = z.view(B,24,24,8).permute(0,3,1,2)
image = decoder(post_quant_conv(z))
```

VQ swish-like activation:

```python
x = group_norm(x)
x = x * sigmoid(x)
```

## 8. Preprocessing and input packing

Image preprocessing for understanding:

- Input images are converted to tensors in NCHW processor convention.
- Resize preserves aspect ratio so the largest side becomes 384; output side is
  at least 14.
- Pad to square with background color derived from mean, current config
  `[127, 127, 127]`.
- Rescale by `0.00392156862745098` and normalize with mean/std `[0.5]*3`.
- Runtime tensor entering Janus is `pixel_values`, expected by model as
  `[B, 3, 384, 384]` for current configs.

Prompt packing:

- Processor expands every tokenizer image placeholder string into
  begin-of-image token, 576 placeholder tokens, end-of-image token.
- For text generation with default system prompt enabled, the processor prepends
  `DEFAULT_SYSTEM_PROMPT`.
- For image generation, processor appends the image start token to the prompt
  and does not process `images`.

Embedding stitch:

- Source computes text embeddings first.
- If `pixel_values` is present, Janus computes vision features and reshapes to
  `[B*576, H]`.
- Placeholder positions are selected by `input_ids == image_token_id` or, when
  only embeddings are provided, equality to the image-token embedding.
- The scatter is hidden-dimension expanded; mismatched token/feature counts
  raise an error.

Postprocess for generated images:

- `decode_image_tokens` returns decoded image as NHWC float tensor.
- Image processor postprocess reverses normalization/rescale, clips to `[0,255]`,
  casts to uint8, and can return PIL images.

## 9. Graph rewrite / lowering opportunities

### Rewrite: vision patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> 1024, kernel=16, stride=16, padding=0) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
WindowFlatten NHWC-or-NCHW patches -> GEMM(weight_flat.T) -> BiasAdd -> [B,576,1024]
```

Preconditions:

- `kernel_size == stride == 16`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input height and width divisible by 16 after preprocessing.
- Flatten order matches PyTorch Conv2d NCHW semantics.

Failure cases:

- Interpolated/dynamic image sizes without divisibility proof.
- Any layout pass that changes patch flatten order without transforming weights.

Parity test sketch: compare Conv2d patch embeddings and rewritten GEMM output
for random NCHW `[B,3,384,384]`, bf16/fp32 tolerances, including bias.

### Rewrite: placeholder scatter as indexed copy

Source pattern:

```text
mask = input_ids == image_token_id
inputs_embeds.masked_scatter(mask[...,None], image_features)
```

Replacement:

```text
validate count(mask) == B*num_image_tokens
indexed_copy(inputs_embeds, nonzero(mask), image_features)
```

Preconditions:

- Exactly 576 placeholders per image and ordering matches processor expansion.
- No duplicate/ambiguous image positions after batching.

Failure cases:

- User supplies `inputs_embeds` without `input_ids`; source falls back to
  embedding equality, which is fragile for optimized lowering. Prefer requiring
  `input_ids` for first integration.

### Rewrite: VQ spatial attention to standard BMM attention

Source pattern:

```text
1x1 conv Q/K/V -> reshape spatial -> BMM -> scale -> softmax -> BMM -> 1x1 conv
```

Replacement: canonical dense noncausal attention over `S = H*W`, preserving NCHW
conv projections as 1x1 GEMMs.

Preconditions:

- Static spatial shape in VQ decoder/encoder blocks.
- No masks, no dropout.

Failure cases:

- Layout translation must rewrite spatial flatten order and restore NCHW conv
  consumer layout.

### Rewrite: image-code generation head as codebook logits

Source pattern:

```text
last_hidden -> Linear(H,H) -> GELU -> Linear(H,16384)
```

Replacement: standard MLP head with output vocabulary equal to VQ codebook size,
not text vocab.

Preconditions:

- `generation_mode == "image"`.
- Output dimension equals `vq_config.num_embeddings`.

Failure cases:

- Applying text-vocab logits masks or token processors to image-code logits.

## 10. Kernel fusion candidates

Highest priority:

- Llama RMSNorm + QKV GEMM + RoPE + causal attention with cache, inherited from
  the Llama path. This dominates text prefill/decode and image-code generation.
- Placeholder indexed-copy/scatter for `[B,576,H]` image features into text
  embeddings.
- Vision patch Conv2d-to-GEMM and vision MHA/MLP fusions for understanding
  throughput.
- VQ-code generation head and code embedding/aligner, because image generation
  runs it 576 times.

Medium priority:

- VQ decoder GroupNorm + swish + Conv2d fusion.
- VQ nearest upsample + Conv2d fusion.
- VQ spatial BMM attention at low resolution.
- Last-token-only hidden/logits path for image-code decode loop.

Lower priority:

- Vision positional interpolation; current processor/config can reject dynamic
  image size first.
- Training-only VQ encode/loss path.
- PIL/uint8 postprocessing on GPU.

## 11. Runtime staging plan

Stage 1: Config and weight loading admission.

- Accept native `model_type: janus` only.
- Reject `multi_modality` remote-code configs for this path.
- Load nested Llama, Janus vision, aligner, VQ codebook/decoder, generation head.

Stage 2: Understanding prefill parity.

- Implement processor-compatible pixel tensor contract or accept preprocessed
  `pixel_values`.
- Run vision encoder + aligner + placeholder stitch + one Llama prefill.
- Validate logits for image-text-to-text prompts.

Stage 3: Text decode parity.

- Use Llama cache ABI directly.
- Keep image features only in first prefill; subsequent steps are text-only
  cache decode.

Stage 4: Image-code generation parity.

- Implement custom Janus image generation loop with CFG batch doubling,
  static cache sizing, 576 code steps, code logits, sampling/argmax, and
  generation-code embeddings.
- First useful target can use greedy decode before multinomial parity.

Stage 5: VQ decode parity.

- Decode `[B,576]` image code indices through codebook + post_quant_conv +
  convolutional decoder.
- Postprocess to image tensors; PIL conversion can stay outside DinoML.

Stage 6: Optimize.

- Add patch Conv2d rewrite, fused Llama attention, fused VQ blocks, and
  last-token-only image-code loop.

## 12. Parity and validation plan

- Config admission tests for native 1B/7B plus rejection of remote-code
  `multi_modality` configs.
- Processor packing tests: one image placeholder expands to exactly 576 model
  placeholders between BOI/EOI; image generation appends BOI and no image tensor.
- Vision embedding parity: random/preprocessed `[B,3,384,384]` through patch
  embedding, position add, one vision layer, all vision layers.
- Placeholder stitch parity: compare source `masked_scatter` with DinoML indexed
  copy for one and multiple images.
- Llama delegated parity: one decoder block, prefill logits, one decode token,
  cache shape/update checks.
- Text end-to-end: image-text prompt to next-token logits and short generate.
- Image-code loop: greedy 1 step, 2 steps, then all 576 code tokens with fixed
  seed for sampling tests.
- VQ decode: codebook lookup + L2 normalize shape test, one decoder block, full
  decoded image tensor.
- Suggested tolerances: fp32 `1e-4` for isolated ops; bf16/fp16 `1e-2` to
  `3e-2` for full blocks, with stricter logits checks when using identical
  attention backend.

## 13. Performance probes

- Image preprocessing throughput separately from GPU model runtime.
- Vision encoder+aligner throughput for batch sizes 1, 2, 4, 8.
- Llama prefill throughput for prompt lengths with and without 576 image tokens.
- Text decode tokens/sec with cache for 1B and 7B.
- Image generation code tokens/sec for 576-step loop, split by Llama decode,
  generation head, sampling, and code embedding/aligner.
- CFG overhead: batch B versus doubled 2B image generation path.
- VQ decoder throughput for generated code batches.
- KV cache memory for text generation and image generation static cache
  `max(max_length, prompt_len + 576)`.
- Compare eager attention, SDPA, and DinoML fused attention for Llama and
  vision branches.
- Conv/GEMM provider probe for patch embedding and VQ conv-heavy decoder.

## 14. Skip/defer list

- Training, losses, gradient checkpointing, and VQ encode commitment loss.
- Remote-code `multi_modality` checkpoints until separately audited.
- Beam search for image generation; source rejects image generation modes other
  than greedy or sampling.
- Dynamic/interpolated image sizes; current native checkpoints use 384x384.
- Multinomial sampling can be deferred behind greedy image-code parity.
- Multi-GPU/tensor parallel and quantized weight formats; no native Janus
  source-coupled quantized storage was found.
- `inputs_embeds`-only multimodal placeholder detection by embedding equality;
  first DinoML integration should require `input_ids`.

## 15. Final implementation checklist

- [ ] Parse native JanusConfig with nested text/vision/VQ configs.
- [ ] Reject remote-code `multi_modality` configs in the native Janus path.
- [ ] Load/tie text embedding and LM head weights.
- [ ] Load Janus vision encoder, aligner, generation embeddings, generation aligner, generation head, and VQ decoder weights.
- [ ] Implement NCHW image preprocessing contract or require preprocessed `pixel_values`.
- [ ] Implement vision patch embedding, learned position add, vision MHA/MLP blocks.
- [ ] Implement placeholder count validation and indexed embedding stitch.
- [ ] Delegate Llama decoder blocks, RoPE, RMSNorm, SwiGLU, attention, and cache through the existing Llama path.
- [ ] Add text prefill/decode parity tests.
- [ ] Implement image generation loop with CFG batch doubling and 576 VQ-code steps.
- [ ] Implement VQ codebook lookup, L2 normalize, post_quant_conv, and VQ decoder.
- [ ] Add generated image-code and VQ decode parity tests.
- [ ] Add patch Conv2d-to-GEMM rewrite with layout/order guards.
- [ ] Benchmark vision, text prefill, text decode, image-code decode, and VQ decoder separately.
