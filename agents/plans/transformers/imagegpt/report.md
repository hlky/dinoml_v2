# ImageGPT Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openai/imagegpt-small, openai/imagegpt-medium, openai/imagegpt-large
Config source: official HF config.json and preprocessor_config.json, plus in-source ImageGPTConfig defaults
Source files inspected:
- X:/H/transformers/src/transformers/models/imagegpt/modeling_imagegpt.py
- X:/H/transformers/src/transformers/models/imagegpt/configuration_imagegpt.py
- X:/H/transformers/src/transformers/models/imagegpt/image_processing_imagegpt.py
- X:/H/transformers/src/transformers/models/imagegpt/image_processing_pil_imagegpt.py
- X:/H/transformers/src/transformers/pytorch_utils.py for Conv1D ABI
- X:/H/transformers/src/transformers/activations.py for quick_gelu
Any missing files or assumptions: no tokenizer file; image tokenization is owned by ImageGPTImageProcessor. openai/imagegpt-small-32x32 and openai/imagegpt-small-cifar10 raw URLs returned 401 and are treated as gaps.
```

Primary runtime target for DinoML: `ImageGPTForCausalImageModeling`, i.e. autoregressive generation over discrete color-cluster image tokens. `ImageGPTModel` is required as the shared body. `ImageGPTForImageClassification` is optional/deferred for first integration.

## 2. High-level architecture

ImageGPT is a GPT-2-style causal decoder over rasterized image-code tokens, not a patch-embedding vision transformer. The dataflow is:

```text
CPU image resize/rescale/normalize -> nearest color-cluster tokenization -> flattened [B, H*W] input_ids
-> token embedding + learned absolute position embedding
-> N causal decoder blocks with KV cache
-> final no-mean LayerNorm
-> LM head over 512 image code tokens
-> sampling -> cluster-id sequence -> CPU cluster lookup back to RGB image
```

Stage decomposition:

- CPU/data pipeline: image resize to 32x32 for public checkpoints, channel-first normalization to roughly `[-1, 1]`, nearest-cluster quantization, row-major flattening.
- GPU/runtime prefill: embed SOS/context/image-token prefix, run full causal decoder.
- GPU/runtime decode: one or more new image-code tokens with per-layer KV cache.
- CPU/postprocess: map generated cluster IDs `0..511` through processor `clusters`, then convert normalized cluster RGB by `round(127.5 * (cluster + 1.0))`.

The color-code processor output is independently testable and cacheable. Decoder KV cache is separate from the image codebook; generated image codes can be stored as token IDs, while KV cache stores projected hidden states after attention projection and before attention matmul reuse.

## 3. Important config dimensions

| Field | Source default | Public checkpoint values | Runtime significance |
|---|---:|---:|---|
| `vocab_size` | 513 | 513 | embedding table includes 512 color codes plus SOS token |
| LM output size | `vocab_size - 1` | 512 | logits exclude SOS; generated image pixels are cluster IDs only |
| `n_positions` | 1024 | 1024 | 32x32 image token grid; generation examples use max length 1025 with initial SOS |
| `n_embd` | 512 | 512 / 1024 / 1536 | hidden width |
| `n_layer` | 24 | 24 / 36 / 48 | decoder block count |
| `n_head` | 8 | 8 / 8 / 16 | MHA heads |
| `head_dim` | `n_embd / n_head` | 64 / 128 / 96 | no GQA/MQA; source rejects non-divisible widths |
| `n_inner` | null | null | MLP width is `4 * n_embd` |
| activation | `quick_gelu` | `quick_gelu` | `x * sigmoid(1.702*x)` |
| layer norm eps | `1e-5` | `1e-5` | no mean subtraction; RMSNorm-like but learned weight only |
| cache | true | true | DynamicCache supported |
| cross attention | false | omitted/effective false | source can instantiate cross-attn but public ImageGPT checkpoints do not use it |

Representative checkpoint sweep:

| Model id | n_layer | hidden | heads | head_dim | MLP | positions | codebook |
|---|---:|---:|---:|---:|---:|---:|---:|
| `openai/imagegpt-small` | 24 | 512 | 8 | 64 | 2048 | 1024 | 512 RGB clusters |
| `openai/imagegpt-medium` | 36 | 1024 | 8 | 128 | 4096 | 1024 | 512 RGB clusters |
| `openai/imagegpt-large` | 48 | 1536 | 16 | 96 | 6144 | 1024 | 512 RGB clusters |

The HF API reports all three official repos as public Apache-2.0 ImageGPT models trained on ImageNet-21k. That metadata is repo metadata, not modeling-source behavior.

## 3a. Family variation traps

- `vocab_size=513` does not mean the LM head emits 513 tokens. `lm_head = Linear(n_embd, vocab_size - 1, bias=False)`, so SOS ID `512` is input-only.
- `tie_word_embeddings=false` in configs, but the class declares `_tied_weights_keys={"lm_head.weight": "transformer.wte.weight"}`. Because shapes differ (`[512, H]` vs `[513, H]`), DinoML should not assume a valid tied alias for this head without checkpoint inspection.
- The custom layer norm does not subtract the mean and has no bias. It is `x / sqrt(mean(x^2)+eps) * weight`.
- Projections use Transformers `Conv1D` weights stored as `[in_features, out_features]` and executed with `addmm(bias, x_flat, weight)`, not PyTorch `Linear` storage `[out, in]`.
- Public configs set `reorder_and_upcast_attn=false`, but source implements an alternate `baddbmm` fp32 attention path. First integration can reject or defer configs with this flag true.
- Public configs omit `pad_token_id`, `bos_token_id`, and `eos_token_id`; generation examples manually seed input with `vocab_size - 1`.
- Source supports optional cross-attention when `add_cross_attention=true`; public ImageGPT generation does not require it.
- Processor defaults in current source say size 256, but public checkpoint preprocessors say `size=32`. Checkpoint preprocessor config must override source defaults.
- Layout translation trap: preprocessing temporarily converts NCHW tensors to NHWC only for color quantization. Decoder consumes flattened `[B, S]` IDs, so NHWC/NCHW optimization is processor-local, not part of the decoder graph.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs `[B, S] -> [B, S, H]`.
- Embedding lookup for learned positions `[1, S] -> [1, S, H]`.
- Optional token type embedding reuse through `wte(token_type_ids)`.
- Reshape/view, transpose/permute, contiguous before merge heads.
- Split packed QKV projection output along last dim into equal `H` chunks.
- Mean reduction over sequence for classification head only.

Neural primitives:

- Conv1D-as-linear with weight layout `[in, out]` and bias `[out]`.
- No-mean RMS-style layer norm over hidden dim.
- Residual add.
- Dropout is inactive for inference but appears in source.
- QuickGELU: `x * sigmoid(1.702*x)`.
- LM GEMM `H -> 512`, bias false.

Attention primitives:

- Dense causal self-attention, MHA only.
- QKV packed projection `H -> 3H`; split order is `query, key, value`.
- Output projection `H -> H`.
- Matmul QK^T, scale by `1/sqrt(head_dim)`, optional layer-index scale, causal mask, optional additive attention mask, softmax, matmul with V.
- KV cache update and read per layer, shapes `[B, n_head, past_seq, head_dim]`.

Position/codebook/preprocessing ops:

- Learned absolute position embedding, no RoPE/ALiBi.
- Color quantization by nearest cluster using squared Euclidean distance against `clusters[512,3]`.
- Processor image path: resize, rescale, normalize, NCHW to NHWC for quantization, flatten row-major to IDs.

## 5. Layer/block breakdown

Decoder body:

```text
input_ids [B,S] or inputs_embeds [B,S,H]
positions [1,S] = arange(S) + past_seen_tokens
x = wte(input_ids) + wpe(positions)
optional x += wte(token_type_ids)
repeat N blocks:
  r = x
  y = no_mean_layer_norm(x)
  qkv = Conv1D_H_to_3H(y)              # weight [H,3H], bias [3H]
  q,k,v = split(qkv, [H,H,H])
  q,k,v -> [B, heads, S_or_K, head_dim]
  k,v = cache.update(k,v) if cache enabled
  a = causal_attention(q,k,v, additive_mask)
  a = merge_heads(a) -> [B,S,H]
  x = r + Conv1D_H_to_H(a)
  r = x
  y = no_mean_layer_norm(x)
  y = Conv1D_H_to_4H(y)
  y = quick_gelu(y)
  y = Conv1D_4H_to_H(y)
  x = r + y
x = final no_mean_layer_norm(x)
logits = matmul(x, lm_head.weight.T)   # [B,S,512], no bias
```

For public checkpoints:

- Small: `H=512`, MLP `2048`, QKV `512 -> 1536`, 24 blocks.
- Medium: `H=1024`, MLP `4096`, QKV `1024 -> 3072`, 36 blocks.
- Large: `H=1536`, MLP `6144`, QKV `1536 -> 4608`, 48 blocks.

Classification head, optional: average hidden states over `dim=1`, then `Linear(H -> num_labels, bias=False)`.

## 6. Attention requirements

Required for causal image generation:

- Causal self-attention.
- MHA, not GQA/MQA.
- Query/key/value width all equal `n_embd`; value head dim equals query/key head dim.
- Attention mask is optional additive mask prepared from `[B, S_total]` as `[B,1,1,S_total]` with masked positions set to `torch.finfo(dtype).min`.
- Causal mask comes from a lower-triangular bool buffer of shape `[1,1,max_pos,max_pos]`. During decode, source slices `bias[:, :, key_length - query_length : key_length, :key_length]`.
- KV cache stores keys and values after projection/reshape and before attention. Shape per layer is `[B, n_head, T, head_dim]`; after appending one token in decode, `T` grows by one.
- No sliding window, local attention, block sparsity, ALiBi, RoPE, packed varlen, or FlashAttention dispatch is implemented in this source.

The source has an alternate `reorder_and_upcast_attn` path:

```text
q [B,Hd,Q,D] and k^T [B,Hd,D,K] are reshaped to baddbmm batches.
scores are computed in fp32 with alpha scale_factor.
softmax stays fp32, then downcasts to value dtype before AV.
```

First DinoML target can require `reorder_and_upcast_attn=false`; public official checkpoints satisfy that.

Cross-attention is source-implemented but not needed for official ImageGPT checkpoints. If admitted later, it adds separate `q_attn H->H`, `c_attn H_enc->2H` for K/V, and an encoder-decoder cache with cross-attention updated once per layer.

## 7. Position encoding and custom math

Position encoding is learned absolute lookup:

```python
past_seen = cache.get_seq_length() if cache is not None else 0
position_ids = arange(seq_len) + past_seen
hidden = wte(input_ids) + wpe(position_ids[None, :])
```

Custom no-mean layer norm:

```python
def imagegpt_norm(x, weight, eps):
    return x / sqrt(mean(x * x, axis=-1, keepdims=True) + eps) * weight
```

QuickGELU:

```python
def quick_gelu(x):
    return x * sigmoid(1.702 * x)
```

Color-code decode:

```python
rgb_uint8 = round(127.5 * (clusters[token_ids] + 1.0)).astype(uint8)
image = rgb_uint8.reshape(height, width, 3)
```

Position tables and causal mask can be precomputed for `n_positions=1024`. Decode position IDs depend on cache length.

## 8. Preprocessing and input packing

Processor contract for public checkpoints:

- Input: RGB images, resized to 32x32 using `resample=2` from preprocessor config.
- Processor normalizes with mean/std `[0.5,0.5,0.5]`, so cluster matching happens in normalized RGB space.
- Torch processor groups images by shape for batched transforms, then quantizes `stacked_images.permute(0,2,3,1).reshape(-1,3)`.
- PIL processor similarly stacks channel-first arrays, transposes to `[B,H,W,C]`, then flattens.
- Color code IDs are nearest-cluster argmin over 512 clusters.
- Final model input is flattened row-major `[B, H*W]`, normally `[B,1024]`.
- Unconditional generation starts with context `[[512]]`, the SOS token. The output sequence includes SOS first; generated image tokens are `output[:,1:]`.

Discrete codebook ABI:

- Codebook source: `preprocessor_config.json` `clusters`, shape `[512,3]`, float values in normalized RGB space.
- Token IDs `0..511` map directly to cluster rows.
- Token ID `512` is SOS/input-only for generation seeding.
- LM logits shape is `[B,S,512]`; sampling must never request ID 512 from the model head.
- There is no placeholder expansion or masked scatter. Image code order is row-major raster order.
- Image codes can be cached as compact integer sequences separately from decoder KV cache.

CPU/data-pipeline ownership is recommended for resize/rescale/normalize/nearest-cluster quantization initially. A GPU color-quantization kernel is optional and should be treated as preprocessing acceleration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Conv1D to GEMM

Source pattern: `Conv1D(nf, nx)` uses `torch.addmm(bias, x.view(-1,nx), weight[nx,nf])`.

Replacement:

```text
Flatten leading dims -> GEMM_RRR(A [B*S,nx], W [nx,nf]) -> BiasAdd -> Reshape [B,S,nf]
```

Preconditions: input last dim is `nx`, weight is stored `[nx,nf]`, dense contiguous or ABI-contiguous flatten. Weight transform only needed if DinoML GEMM expects `[out,in]`; otherwise preserve `[in,out]` as RHS row-major. Failure cases: importing as `nn.Linear` without transposing will silently invert weights.

### Rewrite: packed QKV projection

Source pattern: `c_attn: H -> 3H`, then split order `query,key,value`.

Replacement: single GEMM producing `[B,S,3H]`, then view/split. Optional later fusion with Q/K/V reshape and attention prefill.

Preconditions: self-attention only, `add_cross_attention=false`, split size exactly `H`. Failure cases: cross-attention uses `q_attn` plus K/V-only `c_attn`.

### Rewrite: no-mean norm as RMSNorm family

Source pattern: `x / sqrt(mean(x^2)+eps) * weight`.

Replacement: RMSNorm-like op with no bias and no mean centering.

Preconditions: final axis normalized, learned weight shape `[H]`, fp32 accumulation preferred for fp16/bf16. Failure cases: do not substitute LayerNorm.

### Rewrite: processor color quantization

Source pattern:

```text
x [B,3,H,W] -> x_nhwc_flat [B*H*W,3]
d = x^2 - 2*x@clusters.T + clusters^2
ids = argmin(d, dim=1)
ids.reshape(B,H*W)
```

Replacement: CPU vectorized nearest-centroid or GPU tiled `argmin_512x3`.

Preconditions: cluster shape `[512,3]`, normalized RGB input, row-major flatten. Layout constraints: NCHW input from processor; NHWC is local only. Failure cases: `do_color_quantize=false` returns pixel values, which the decoder cannot consume as `input_ids`.

### Rewrite: last-token-only logits

Source pattern computes LM logits for all hidden states.

Replacement for decode: apply LM head only to `hidden[:, -1:, :]`.

Preconditions: generation controller needs only next-token logits and no full-sequence logits. Failure cases: loss computation, full prefill parity tests, or APIs requesting all logits.

## 10. Kernel fusion candidates

Highest priority:

- Conv1D/GEMM coverage with correct `[in,out]` storage, because all projections depend on it.
- No-mean RMSNorm kernel with fp32 accumulation.
- Causal MHA prefill and decode with KV cache for `S<=1024`; attention is the main runtime cost.
- QuickGELU MLP fusion: GEMM -> QuickGELU -> GEMM, with activation fused where possible.
- Last-token-only LM head for decode to avoid `[B,S,512]` logits work.

Medium priority:

- QKV projection + split + head reshape fusion.
- Cache append plus causal attention layout standardization.
- CPU or GPU optimized color quantization for preprocessing throughput.
- Fused residual add around attention/MLP output.

Lower priority:

- Optional `reorder_and_upcast_attn=true` fp32 baddbmm path.
- Cross-attention path.
- Classification average-pool head.
- Training loss and dropout.

## 11. Runtime staging plan

Stage 1: parse ImageGPT config and processor config, enforce public-checkpoint subset: `add_cross_attention=false`, `reorder_and_upcast_attn=false`, `vocab_size=513`, `n_positions=1024`, clusters `[512,3]`.

Stage 2: import weights with Conv1D layout preserved. Add one-block parity for embeddings, no-mean norm, packed QKV, attention, MLP.

Stage 3: run full prefill parity for `ImageGPTModel` and `ImageGPTForCausalImageModeling` on short token prefixes and full 1024-token image sequences.

Stage 4: implement decode with DynamicCache-compatible per-layer K/V tensors, starting from SOS and generated prefixes.

Stage 5: wire processor-owned CPU color quantization and RGB decode for end-to-end image-token parity.

Stage 6: add optimized attention/GEMM fusions and last-token logits.

Stage 7: optional classification head and optional preprocessing acceleration.

Initially stub/defer losses, dropout, gradient checkpointing, cross-attention, output attentions, and hidden-state collection.

## 12. Parity and validation plan

- Unit test color quantization against HF processor for small synthetic RGB arrays and one real image. Include cluster tie behavior as `argmin` first index.
- Unit test no-mean norm versus PyTorch source in fp32 and fp16.
- Unit test Conv1D import: random `x`, random `[in,out]` weight, compare `addmm`.
- Single block parity for small/medium/large shapes with cache disabled.
- Full model prefill logits parity on random valid token IDs `0..512` for prefixes and `0..511` for image tokens.
- Decode parity: seed with SOS `[512]`, append a few sampled or fixed IDs, compare next-token logits and cache lengths after each step.
- End-to-end processor + model smoke: HF processor `input_ids` into DinoML and HF model, compare logits.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 relaxed around attention/softmax, e.g. `rtol=1e-2, atol=1e-2`, with fp32 accumulation in norm/softmax.

## 13. Performance probes

- CPU processor throughput: images/sec for resize/normalize/nearest-cluster quantization.
- GPU optional color-quantization throughput: `B*1024` pixels against 512 clusters.
- Prefill latency/throughput for sequence lengths 1, 64, 256, 1024.
- Decode tokens/sec with KV cache for batch sizes 1, 4, 16 and past lengths up to 1024.
- Attention backend comparison: naive dense, fused causal, and cache decode kernels.
- GEMM throughput per projection shape for small/medium/large.
- LM head last-token-only versus full-sequence logits.
- KV cache memory: `2 * layers * B * heads * T * head_dim * dtype_size`.
- End-to-end images/sec including sampling and CPU RGB decode.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Dropout behavior in training mode.
- Cross-attention and encoder-decoder cache.
- `reorder_and_upcast_attn=true` unless a checkpoint requires it.
- Output attentions and all hidden states.
- Image classification head for first generation target.
- Beam search; first integration can use greedy/top-k sampling controller outside the compiled graph.
- Gated/401 checkpoint variants until access is available.
- GPU preprocessing unless CPU processor is a measured bottleneck.

## 15. Final implementation checklist

- [ ] Parse `ImageGPTConfig` and checkpoint `preprocessor_config.json`.
- [ ] Validate/disallow unsupported config flags for first target.
- [ ] Load `clusters[512,3]` and define SOS/input-only token ID 512.
- [ ] Implement or bind CPU processor parity for resize/rescale/normalize/color quantization.
- [ ] Import Conv1D weights preserving `[in,out]` orientation.
- [ ] Implement ImageGPT no-mean norm.
- [ ] Implement QuickGELU.
- [ ] Lower packed QKV projection and split order `q,k,v`.
- [ ] Implement causal MHA prefill with additive mask.
- [ ] Implement per-layer KV cache decode.
- [ ] Implement LM head `H -> 512` with no bias and last-token-only decode optimization.
- [ ] Add one-block parity tests.
- [ ] Add full prefill logits parity tests for small and one larger config shape.
- [ ] Add decode cache parity tests from SOS.
- [ ] Add processor/codebook round-trip tests.
- [ ] Benchmark prefill, decode, processor, and cache memory.
