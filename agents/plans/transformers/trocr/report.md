# TrOCR DinoML operator/runtime audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/trocr-base-handwritten as the common OCR reference; sweep also covers small/base/large printed and large handwritten.
Config source: official Hugging Face config.json, preprocessor_config.json, tokenizer_config.json, generation_config.json snapshots under _sources/hf_configs/.
Source files inspected:
- transformers/src/transformers/models/trocr/modeling_trocr.py
- transformers/src/transformers/models/trocr/configuration_trocr.py
- transformers/src/transformers/models/trocr/processing_trocr.py
- transformers/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py for wrapper ownership boundary.
Any missing files or assumptions: no gated/401/403 gaps found for sampled Microsoft checkpoints. Vision encoder operator coverage composes existing ViT/DeiT audits rather than being re-owned by this TrOCR decoder audit.
```

Source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/trocr/modeling_trocr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/trocr/configuration_trocr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/trocr/processing_trocr.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py

Local snapshots:

- `agents/plans/transformers/trocr/_sources/modeling_trocr.py`
- `agents/plans/transformers/trocr/_sources/configuration_trocr.py`
- `agents/plans/transformers/trocr/_sources/processing_trocr.py`
- `agents/plans/transformers/trocr/_sources/modeling_vision_encoder_decoder.py`
- `agents/plans/transformers/trocr/_sources/hf_configs/*`

## 2. High-level architecture

Primary runtime target: end-to-end OCR generation through `VisionEncoderDecoderModel` using a ViT/DeiT image encoder and a TrOCR causal decoder.

Architecture:

```text
CPU image preprocessing -> ViT/DeiT encoder -> encoder hidden states
decoder start token -> TrOCR decoder prefill/decode with cross-attention -> vocab logits -> text generation -> tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: image load/convert, resize to 384, rescale if enabled by processor defaults, normalize with mean/std, channel-first `pixel_values`.
- Vision encoder: external ViT/DeiT family returns `encoder_outputs[0]` as `[batch, encoder_seq, encoder_hidden]`; for 384x384 patch16 checkpoints this is normally 577 tokens including CLS.
- Optional bridge: generic wrapper only applies `enc_to_dec_proj` when encoder hidden size differs from decoder hidden size and decoder `cross_attention_hidden_size` is absent. Sampled TrOCR configs set `cross_attention_hidden_size`, so TrOCR cross-attention directly projects encoder width.
- Decoder: autoregressive text decoder with causal self-attention, encoder-decoder cross-attention, MLP, learned or sinusoidal positions, and untied or tied LM projection depending on checkpoint.
- Generation controller: `decoder_start_token_id=2`, `eos_token_id=2`, `pad_token_id=1`, tokenizer decode with skipped special tokens.

Independently cacheable stages: image encoder outputs can be cached per image; during decode, cross-attention K/V can be cached per layer separately from growing decoder self-attention K/V.

## 3. Important config dimensions

Representative checkpoint sweep from official configs:

| Checkpoint | Encoder | Encoder dim/layers/heads | Decoder dim/layers/heads | Head dim | FFN | Vocab | Max pos | Cross-attn K/V width | Decoder activation | Position path | Embed scale | Weight tie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `microsoft/trocr-small-printed` | DeiT | 384 / 12 / 6 | 256 / 6 / 8 | 32 | 1024 | 64044 | 512 | 384 | relu | learned | yes | false |
| `microsoft/trocr-base-printed` | ViT | 768 / 12 / 12 | 1024 / 12 / 16 | 64 | 4096 | 50265 | 512 | 768 | gelu | config omits, effective default learned | no | true |
| `microsoft/trocr-base-handwritten` | ViT | 768 / 12 / 12 | 1024 / 12 / 16 | 64 | 4096 | 50265 | 512 | 768 | gelu | config omits, effective default learned | no | true |
| `microsoft/trocr-large-printed` | ViT | 1024 / 24 / 16 | 1024 / 12 / 16 | 64 | 4096 | 50265 | 1024 | 1024 | relu | sinusoidal | yes | false |
| `microsoft/trocr-large-handwritten` | ViT | 1024 / 24 / 16 | 1024 / 12 / 16 | 64 | 4096 | 50265 | 512 | 1024 | gelu | config omits, effective default learned | no | true |

Common processor config across sampled checkpoints:

| Field | Value |
|---|---|
| image processor | `ViTImageProcessor` |
| resize | enabled |
| size | 384 |
| normalize | enabled |
| mean/std | `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]` |
| source layout after processor | `pixel_values` as `[batch, 3, 384, 384]` |

Source defaults that may fill omitted old checkpoint fields:

| Field | Source default |
|---|---:|
| `use_learned_position_embeddings` | true |
| `layernorm_embedding` | true |
| `scale_embedding` | false |
| `use_cache` | true in `TrOCRConfig`, but sampled official checkpoint configs serialize decoder `use_cache=false` |
| `tie_word_embeddings` | true in `TrOCRConfig`; top-level `VisionEncoderDecoderConfig.tie_word_embeddings=false` |

## 3a. Family variation traps

- Small printed is not a scaled-down base only: it uses a DeiT encoder, decoder hidden 256, vocab 64044, ReLU, embedding scale, and no output/input weight tie.
- Large printed changes max decoder positions to 1024, uses sinusoidal decoder positions, ReLU, embedding scale, and untied LM projection.
- Base/handwritten configs omit `use_learned_position_embeddings` and `layernorm_embedding`; under the inspected in-library `TrOCRConfig`, missing fields default to learned positions plus embedding LayerNorm.
- `cross_attention_hidden_size` is operator-significant. It sets cross-attention K/V input width, avoiding the generic wrapper projection even when encoder hidden size differs from decoder hidden size.
- Sampled decoder configs set `use_cache=false`; source implements self-attention and cross-attention cache. First parity should respect config defaults, then optimized decode can explicitly enable cache if generation parity is checked.
- TrOCR source is decoder-only. End-to-end OCR composition and image encoder ownership live in `VisionEncoderDecoderModel` plus ViT/DeiT source.
- Vision tensor source contract is NCHW. NHWC is a guarded optimization inside the ViT/DeiT patch/attention path, not a semantic rewrite at the TrOCR wrapper boundary.
- The wrapper sets `encoder_attention_mask = None`, so cross-attention is normally unmasked over all encoder tokens.
- Output projection can be tied to `model.decoder.embed_tokens.weight`; lowering must preserve aliasing when `tie_word_embeddings=true`.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B,3,384,384]` from processor.
- ViT/DeiT encoder sequence output `[B,Senc,Henc]`, commonly `[B,577,768]` or `[B,577,1024]`.
- Decoder token reshape/view from `[B,T]`, embedding lookup, add token+position, optional LayerNorm.
- Attention reshape/transposes: `[B,T,Hdec] -> [B,heads,T,head_dim] -> [B*heads,T,head_dim]`.
- Optional last-token slicing in generation controller for cached decode.
- Axis-sensitive layout guard: do not translate sequence hidden tensors to NHWC; only vision encoder internals may use guarded channel-last fusion.

Neural network primitives:

- Token embedding `Embedding(vocab_size,Hdec,padding_idx=1)` with optional scale `sqrt(Hdec)`.
- Learned position embedding table `[max_position_embeddings+2,Hdec]` or sinusoidal index select.
- LayerNorm over hidden dim after embeddings when enabled.
- Per decoder layer self-attention projections with bias:
  - small: Q/K/V/O `Linear(256 -> 256)`
  - base/large: Q/K/V/O `Linear(1024 -> 1024)`
- Cross-attention projections with bias:
  - small: Q `Linear(256 -> 256)`, K/V `Linear(384 -> 256)`, O `Linear(256 -> 256)`
  - base: Q `Linear(1024 -> 1024)`, K/V `Linear(768 -> 1024)`, O `Linear(1024 -> 1024)`
  - large: Q/K/V/O all `Linear(1024 -> 1024)` when encoder width is 1024.
- MLP with bias:
  - small: `Linear(256 -> 1024)` -> ReLU -> `Linear(1024 -> 256)`
  - base/handwritten: `Linear(1024 -> 4096)` -> GELU -> `Linear(4096 -> 1024)`
  - large printed: `Linear(1024 -> 4096)` -> ReLU -> `Linear(4096 -> 1024)`
- LM projection `Linear(Hdec -> vocab_size, bias=False)`.

Attention primitives:

- Decoder causal MHA self-attention, no GQA/MQA, no RoPE.
- Decoder cross-attention MHA with rectangular `[Tdec,Senc]` scores.
- Additive masks shaped `[B,1,T,S]`.
- Softmax over source length; dropout is inference-disabled.
- Source implementation uses eager `bmm`, not SDPA/FlashAttention dispatch.

Position ops:

- Learned absolute decoder positions with offset 2.
- Sinusoidal positions with padding-aware cumulative positions for checkpoints that set `use_learned_position_embeddings=false`.
- Vision encoder absolute position interpolation belongs to ViT/DeiT reports.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache self, DynamicCache cross)` when cache is enabled with encoder states.
- Self-attention cache grows keys/values `[B,heads,Tpast,head_dim]`.
- Cross-attention cache stores projected encoder K/V `[B,heads,Senc,head_dim]` and uses an `is_updated[layer_idx]` flag.
- Beam/cache reorder support comes from generic cache/generation infrastructure, not custom TrOCR code.

Preprocessing-coupled ops:

- Resize image to 384, convert/rescale/normalize according to ViTImageProcessor.
- Tokenizer for labels/text decode; no image placeholders or multimodal token stitching.

Parameter aliasing:

- `_tied_weights_keys` maps `output_projection.weight` to `model.decoder.embed_tokens.weight`. Honor this only when the resolved config ties weights.

## 5. Layer/block breakdown

Vision encoder, composed from ViT/DeiT audit:

```text
pixel_values [B,3,384,384]
  -> patch embedding Conv2d(kernel=16,stride=16) -> patch tokens [B,576,Henc]
  -> add CLS/absolute positions -> encoder blocks -> encoder_hidden_states [B,577,Henc]
```

TrOCR decoder embedding:

```text
input_ids [B,T]
token = Embedding(input_ids) * (sqrt(Hdec) if scale_embedding else 1)
pos = learned_or_sinusoidal_positions(input_ids, past_len)
x = token + pos
x = LayerNorm(x) if layernorm_embedding else x
```

Decoder block, repeated `decoder_layers` times:

```text
residual = x
q = Linear(Hdec -> Hdec, bias)(x) * head_dim**-0.5
k,v = Linear(Hdec -> Hdec, bias)(x or cache append)
x = MHA(q,k,v, causal_mask)
x = LayerNorm(residual + Linear(Hdec -> Hdec, bias)(x))

residual = x
q = Linear(Hdec -> Hdec, bias)(x) * head_dim**-0.5
k,v = Linear(Henc/cross_attention_hidden_size -> Hdec, bias)(encoder_hidden_states or cached cross K/V)
x = MHA(q,k,v, encoder_mask=None)
x = LayerNorm(residual + Linear(Hdec -> Hdec, bias)(x))

residual = x
x = Linear(Hdec -> FFN, bias)(x)
x = GELU or ReLU
x = Linear(FFN -> Hdec, bias)(x)
x = LayerNorm(residual + x)
```

LM head:

```text
logits = Linear(Hdec -> vocab_size, bias=False)(x)
```

## 6. Attention requirements

Self-attention:

- Causal autoregressive decoder self-attention.
- MHA only. `num_key_value_heads == num_attention_heads`.
- Shapes:
  - small: heads 8, head_dim 32, Q/K/V width 256.
  - base/large: heads 16, head_dim 64, Q/K/V width 1024.
- Query is scaled before matmul by `head_dim**-0.5`.
- Mask is created by `create_causal_mask` and added before softmax.
- Cached self-attention K/V are stored after projection and after `[B,heads,T,head_dim]` transpose; there is no RoPE.

Cross-attention:

- Noncausal encoder-decoder MHA.
- Query source is decoder hidden states `[B,Tdec,Hdec]`.
- Key/value source is vision encoder hidden states `[B,Senc,Henc]`; for 384 patch16 ViT/DeiT configs, `Senc` is normally 577 including CLS.
- Rectangular attention scores `[B*heads,Tdec,Senc]`.
- Cross K/V projection input width is `cross_attention_hidden_size` where set: 384, 768, or 1024 in sampled configs.
- Wrapper normally passes `encoder_attention_mask=None`, so no cross mask is used.
- With `EncoderDecoderCache`, cross K/V are computed once, stored per layer, and reused after `is_updated[layer_idx]=True`.

Backend compatibility:

- Source path is eager projection + reshape + `torch.bmm` + softmax + `torch.bmm`.
- A fused attention backend can replace it if it preserves query pre-scaling, additive mask order, rectangular cross-attention, and cache shapes.
- The eager fallback is likely too slow for batched OCR decode because every token does per-layer self-attention and cross-attention in Python-level modules.

## 7. Position encoding and custom math

Learned positions:

```python
def trocr_learned_positions(input_ids, past_len, table):
    bsz, seq_len = input_ids.shape[:2]
    pos = arange(past_len, past_len + seq_len).expand(bsz, seq_len)
    return table[pos + 2]
```

Sinusoidal positions:

```python
def trocr_sinusoidal_positions(input_ids, pad_id, past_len, table):
    mask = (input_ids != pad_id).int()
    pos = (cumsum(mask, dim=1) + past_len) * mask + pad_id
    return table.index_select(0, pos.reshape(-1)).reshape(input_ids.shape[0], input_ids.shape[1], -1)
```

Embedding scale:

```python
token_emb = embedding(input_ids) * (sqrt(hidden_size) if scale_embedding else 1.0)
```

Precomputable:

- Learned position table and sinusoidal table up to max decode length.
- Encoder hidden states and cross-attention K/V for a fixed image.

Dynamic:

- Decoder position IDs depend on `past_key_values_length`.
- Sinusoidal position IDs depend on padding pattern and past length.
- Self-attention causal mask depends on current target length and cache length.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- `TrOCRProcessor` delegates images to `ViTImageProcessor`.
- Sampled processor configs resize to 384, normalize with mean/std 0.5, and emit `pixel_values`.
- `TrOCRProcessor` delegates text to tokenizer. With both images and text, it returns image inputs plus `labels=tokenizer(...).input_ids`.
- No layout metadata, grid metadata, placeholder expansion, or multimodal scatter appears in TrOCR processor.

GPU/runtime work:

- `VisionEncoderDecoderModel.forward(pixel_values=...)` invokes the encoder unless `encoder_outputs` are already supplied.
- Encoder outputs feed decoder cross-attention directly.
- For labels, wrapper computes decoder inputs by shifting labels right, replacing `-100` with pad token, and prepending `decoder_start_token_id`.
- Generation starts from `decoder_start_token_id=2` in sampled generation configs and stops on `eos_token_id=2`.

Postprocessing:

- Tokenizer `batch_decode(..., skip_special_tokens=True)` is required for end-to-end OCR text parity.
- No structured boxes, NMS, or segmentation masks are part of TrOCR source behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: ViT/DeiT patch Conv2d -> Linear

Source pattern:

```text
NCHW pixel_values -> Conv2d(in=3,out=Henc,kernel=16,stride=16,padding=0) -> flatten patches
```

Replacement:

```text
WindowFlatten([16,16,3]) -> MatMul(weight_flat.T) -> BiasAdd -> token reshape
```

Preconditions:

- Encoder is ViT/DeiT patch embedding with `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Image height/width divisible by patch size after processor resize.
- Consumer is the local patch-token reshape in the encoder.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Layout constraints:

- This is a local fully-contained NCHW-to-window pattern. It is safe to lower to NHWC window flatten only if the following token order matches source flatten order.
- Protect the wrapper boundary and decoder sequence tensors with a conceptual `no_layout_translation()`.

Failure cases:

- Dynamic image sizes without matching position interpolation audit.
- Non-16 patch size or nonzero padding/dilation/groups.

Parity test sketch: compare patch embeddings before encoder position add for random `[B,3,384,384]`.

### Rewrite: separate Q/K/V projections -> packed projection

Source pattern:

```text
q = Linear(x) * scale
k = Linear(src)
v = Linear(src)
```

Replacement:

```text
PackedQKVMatMul -> split [Q,K,V] -> reshape heads
```

Preconditions:

- Same input tensor for self-attention; for cross-attention, Q input differs from K/V input, so only pack K/V together unless backend supports two-input QKV.
- All projections have bias and output `Hdec`.
- Preserve split order as Q, K, V for any newly packed tensor; source weights are stored as separate modules.

Weight transform:

```python
packed_w = concat([q.weight, k.weight, v.weight], axis=0)
packed_b = concat([q.bias, k.bias, v.bias], axis=0)
```

Failure cases:

- Cross-attention with `Henc != Hdec` cannot use a single input packed QKV.
- Quantized or sharded weights need separate admission.

Parity test sketch: compare projected Q/K/V before reshape and after cache update.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = output_projection(hidden_states[:, :, :])
```

Replacement:

```text
decode step: output_projection(hidden_states[:, -1:, :])
```

Preconditions:

- Autoregressive generation only.
- No loss computation and caller only consumes next-token logits.
- Preserve full logits for prefill parity when requested.

Parity test sketch: compare last token logits from full projection and sliced projection.

### Rewrite: cross-attention K/V precompute

Source pattern:

```text
for each generated token: K,V = Linear(encoder_hidden_states) unless cross cache is updated
```

Replacement:

```text
after encoder: per-layer CrossKVProject -> cache [B,heads,Senc,head_dim]
decode: reuse cached K/V
```

Preconditions:

- Encoder hidden states are fixed for the request.
- Decoder layer weights are fixed.
- No encoder attention mask changes per decode step.

Failure cases:

- Beam expansion/reorder must also reorder cross-cache batch dimension.
- If encoder outputs are recomputed due to dynamic augmentation, invalidate cache.

Parity test sketch: compare logits for two decode steps with source cache and precomputed cross K/V.

## 10. Kernel fusion candidates

Highest priority:

- ViT/DeiT patch embedding Conv2d-to-GEMM plus channel-last vision internals: OCR requests pay this once per image, and NHWC can remove local image-layout churn in the vision encoder.
- Decoder self-attention and cross-attention fused kernels: TrOCR does two attentions per layer per token; source eager `bmm` path is the main decode hot path.
- Cross-attention K/V projection cache: encoder sequence length is fixed, so avoiding repeated K/V projection is high value.
- Last-token-only LM projection: vocab projection is large, especially 50k to 64k vocab.

Medium priority:

- LayerNorm + residual fusion after attention and MLP.
- MLP GEMM + activation fusion for ReLU/GELU.
- Embedding lookup + position add + optional LayerNorm fusion for prefill.
- Packed self-attention QKV projection when input is shared.

Lower priority:

- Dropout removal in inference graphs.
- Sinusoidal position table precompute for large printed.
- Beam-search specific cache reorder kernels; defer until greedy/batch decode is stable.

## 11. Runtime staging plan

1. Parse `VisionEncoderDecoderConfig`, nested encoder config, nested TrOCR decoder config, processor config, and generation config.
2. Load weights, preserving tied `output_projection`/token embedding aliases where applicable.
3. Compose an already-supported ViT/DeiT encoder and verify encoder hidden-state shape and dtype for `[B,3,384,384]`.
4. Implement TrOCR decoder block parity without cache using full target sequence and cross-attention.
5. Add end-to-end prefill logits parity through `VisionEncoderDecoderModel` semantics with `encoder_attention_mask=None`.
6. Add decode loop with optional self-attention cache and cross-attention cache; first respect checkpoint `use_cache=false`, then benchmark explicit cache enablement.
7. Enable optimized attention, cross-K/V precompute, and last-token logits.
8. Add guarded vision layout rewrites/fusions. Keep decoder sequence tensors in source layout.

Stubs acceptable initially:

- Beam search, sampling variants, and cache reorder.
- Training loss/label shifting if first target is inference-only OCR.
- Rare sinusoidal large-printed path if base/handwritten is the first milestone, provided config admission rejects it until implemented.

## 12. Parity and validation plan

- Random tensor tests:
  - learned position offset and sinusoidal position IDs with pad tokens.
  - attention reshape and mask shape validation for self and cross attention.
  - cross-attention K/V projection for `Henc != Hdec`.
- Single-layer parity:
  - one TrOCR decoder layer with random hidden states and random encoder states.
  - test small-like dimensions and base-like dimensions.
- After-N-layer parity:
  - decoder stack without cache, full sequence.
  - decoder stack with cache for two or more incremental steps.
- Encoder/projector parity:
  - compare external ViT/DeiT encoder output shape and values using already audited encoder path.
  - verify no `enc_to_dec_proj` is inserted when `cross_attention_hidden_size` is set.
- Prefill logits parity:
  - `pixel_values + decoder_input_ids` -> logits for base handwritten.
- Decode token parity:
  - greedy generate a short OCR sequence with and without explicit cache, noting sampled configs default cache off.
- End-to-end OCR parity:
  - same image processor output and tokenizer decode as HF for printed and handwritten checkpoints.
- Recommended tolerances:
  - fp32: `atol=1e-5`, `rtol=1e-5`.
  - fp16/bf16: `atol=1e-2`, `rtol=1e-2` for logits; use token parity for generation.

## 13. Performance probes

- CPU preprocessing throughput: image resize/normalize per second.
- Encoder-only throughput for `[B,3,384,384]`, split by ViT/DeiT variant.
- Decoder prefill throughput by target length and batch size.
- Decode-only tokens/sec with cache disabled vs enabled.
- Cross-attention K/V precompute cost and memory by batch/layer.
- End-to-end OCR requests/hour for small, base, and large.
- Batch-size sweep with fixed 384 images.
- Target sequence-length sweep up to 512 and 1024 for large printed.
- KV cache memory usage: self cache plus cross cache.
- Attention backend comparison: eager bmm vs fused self-attention vs fused rectangular cross-attention.
- LM projection benchmark: full sequence logits vs last-token-only logits.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- LayerDrop behavior; inference should disable it.
- Beam search and beam cache reorder for the first greedy OCR target.
- Sampling controls beyond greedy decode.
- Multi-GPU tensor parallel.
- Quantization and packed weight formats; not present in inspected source/configs.
- Remote-code variants; none required for sampled Microsoft checkpoints.
- General `enc_to_dec_proj` path for non-TrOCR composites where `cross_attention_hidden_size` is absent.

## 15. Final implementation checklist

- [ ] Parse top-level `VisionEncoderDecoderConfig` and nested TrOCR decoder config.
- [ ] Parse ViT/DeiT image processor config and tokenizer/generation special IDs.
- [ ] Load encoder and decoder weights, preserving tied embedding/LM-head aliases.
- [ ] Compose ViT/DeiT encoder output `[B,Senc,Henc]`.
- [ ] Implement TrOCR learned position offset and sinusoidal position path.
- [ ] Implement decoder embedding scale and optional embedding LayerNorm.
- [ ] Implement causal self-attention MHA with additive mask.
- [ ] Implement rectangular cross-attention with K/V input width from `cross_attention_hidden_size`.
- [ ] Implement decoder MLP with GELU/ReLU selected by config.
- [ ] Implement LM projection and last-token decode option.
- [ ] Implement optional `EncoderDecoderCache` self and cross K/V ABI.
- [ ] Add guarded ViT/DeiT patch Conv2d-to-GEMM rewrite.
- [ ] Add packed self-attention QKV rewrite under shape/layout guards.
- [ ] Add cross-attention K/V precompute optimization.
- [ ] Add parity tests for small, base-handwritten, and large-printed config variants.
- [ ] Benchmark preprocessing, encoder, prefill, decode, cache memory, and logits projection separately.
