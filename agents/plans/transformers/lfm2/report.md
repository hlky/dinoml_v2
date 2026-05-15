# LFM2 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: LiquidAI/LFM2-1.2B as the docstring/default representative; config sweep includes 350M, 700M, 1.2B, 2.6B, and LFM2.5 variants.
Config source: HF `config.json` files fetched from raw `main` on 2026-05-13 plus `Lfm2Config` defaults.
Source files inspected:
- transformers/src/transformers/models/lfm2/configuration_lfm2.py
- transformers/src/transformers/models/lfm2/modeling_lfm2.py
- transformers/src/transformers/models/lfm2/modular_lfm2.py
- transformers/src/transformers/cache_utils.py for hybrid KV/conv cache behavior
- transformers/src/transformers/configuration_utils.py and modeling_rope_utils.py for RoPE config normalization
Any missing files or assumptions: tokenizer/generation configs, safetensors metadata, remote-code variants, GGUF/ONNX/MLX conversions, and `lfm2_moe`/`lfm2_vl` are out of scope.
```

Primary source URLs at the inspected commit:

- [configuration_lfm2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2/configuration_lfm2.py)
- [modeling_lfm2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2/modeling_lfm2.py)
- [modular_lfm2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2/modular_lfm2.py)

`modeling_lfm2.py` is generated from `modular_lfm2.py`; future source edits should treat `modular_lfm2.py` as authoritative and `modeling_lfm2.py` as the concrete generated runtime source.

## 2. High-level architecture

LFM2 is a text-only causal language model with a hybrid decoder stack. Each decoder layer is either:

- `full_attention`: causal GQA self-attention with RoPE, q/k per-head RMSNorm, dynamic KV cache, and an MLP.
- `conv`: gated short causal depthwise Conv1d with fixed-size conv state, and an MLP.

Dataflow:

```text
input_ids -> token embedding -> hybrid decoder layers -> final RMSNorm -> tied/untied lm_head -> logits/sampling
```

Runtime decomposition:

- CPU/data pipeline: tokenization, chat template, generation controller, attention mask construction inputs.
- GPU prefill: token embedding, full prompt through hybrid layers, attention-layer KV cache fill, conv-layer state fill.
- GPU decode: one token through attention layers using KV cache, conv layers using fixed `conv_states`, final logits.
- Independently optimizable blocks: RMSNorm, GQA attention, short-conv operator, SwiGLU MLP, last-token logits.

## 3. Important config dimensions

Source defaults from `Lfm2Config`:

| Field | Default / behavior | Runtime significance |
|---|---:|---|
| `vocab_size` | 65536 | embedding and LM head width |
| `hidden_size` | 2560 | residual width |
| `intermediate_size` | 12288 | MLP source value before optional adjustment |
| `block_auto_adjust_ff_dim` | true | if true, effective MLP hidden = `ceil_to_multiple(block_ffn_dim_multiplier * int(2 * intermediate_size / 3), block_multiple_of)` |
| `num_hidden_layers` | 32 | decoder depth |
| `num_attention_heads` | 32 | query heads |
| `num_key_value_heads` | 8 | GQA KV heads |
| `head_dim` | absent by default | effective `hidden_size // num_attention_heads` unless config adds `head_dim` |
| `max_position_embeddings` | 128000 | RoPE/cache position range |
| `rope_parameters` / `rope_theta` | default theta 1000000.0 | standardized by config utilities |
| `conv_L_cache` | 3 | short-conv kernel/state length |
| `conv_bias` | false | published configs use no conv/projection bias |
| `layer_types` | derived from `full_attn_idxs` | per-layer attention vs conv admission |
| `use_cache` | true | hybrid KV plus conv-state cache |
| `tie_word_embeddings` | true | LM head may alias token embedding |

Representative checkpoint sweep from HF config files:

| Model | Hidden | Layers | Q heads / KV heads | Head dim | Full-attn layers | Conv layers | Effective MLP hidden | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `LiquidAI/LFM2-350M` | 1024 | 16 | 16 / 8 | 64 | 6 | 10 | 4608 | bf16 |
| `LiquidAI/LFM2-700M` | 1536 | 16 | 24 / 8 | 64 | 6 | 10 | 6912 | bf16 |
| `LiquidAI/LFM2-1.2B` | 2048 | 16 | 32 / 8 | 64 | 6 | 10 | 8192 | bf16 |
| `LiquidAI/LFM2-2.6B` | 2048 | 30 | 32 / 8 | 64 | 8 | 22 | 10752 | bf16 |
| `LiquidAI/LFM2.5-350M` | 1024 | 16 | 16 / 8 | 64 | 6 | 10 | 4608 | bf16 |
| `LiquidAI/LFM2.5-1.2B-Thinking` | 2048 | 16 | 32 / 8 | 64 | 6 | 10 | 8192 | bf16 |

The 350M/700M/1.2B configs use `full_attn_idxs`; newer configs may spell the same pattern as explicit `layer_types`. DinoML should normalize both before graph construction.

## 3a. Family variation traps

- `layer_types` is structural. Conv layers are not attention layers and do not own KV tensors.
- If `layer_types` is absent and `full_attn_idxs` is absent, the config default makes every layer `full_attention`; published LFM2 checkpoints usually do not use that all-attention default.
- `intermediate_size` is not always the actual MLP width. When `block_auto_adjust_ff_dim=true`, source recomputes it with the `2/3` rule and `block_multiple_of` rounding.
- GQA is required: `num_key_value_heads < num_attention_heads` in representative configs.
- Source supports optional `head_dim`; do not infer `hidden_size == num_attention_heads * head_dim` until normalized.
- q/k/v projections are separate dense weights, not source-packed QKV. Safe fusion must define a packed weight transform.
- q and k get per-head RMSNorm after projection and before RoPE; value does not.
- Attention projections and MLP projections are biasless. Short-conv can have bias if `conv_bias=true`, although representative configs use false.
- Legacy configs use `rope_theta`; new configs may use `rope_parameters`. DinoML should accept the normalized effective rope dict, not only one spelling.
- `theta`, `num_heads`, `block_dim`, `conv_dim`, `block_norm_eps`, and similar historical fields are present in some configs but not read by the inspected LFM2 modeling path.
- `LiquidAI/LFM2-24B-A2B` and `LiquidAI/LFM2-8B-A1B` are `lfm2_moe`, not this family.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[B, S] -> [B, S, H]`.
- Reshape/view: projection outputs `[B, S, heads * D] -> [B, S, heads, D]`.
- Transpose: attention `[B, S, heads, D] <-> [B, heads, S, D]`; short-conv `[B, S, 3H] -> [B, 3H, S]`.
- Chunk/split: short-conv `3H -> B, C, x`; optional fused QKV split.
- Contiguous/materialization after transpose before `out_proj`.
- Slice last tokens for `logits_to_keep`; support integer and tensor indices if matching HF API.

Neural primitives:

- RMSNorm over last dim for residual width `H` and per-head width `D`, fp32 variance accumulation.
- Linear `H -> QH*D`, `H -> KVH*D`, `H -> KVH*D`, and `QH*D -> H` for attention layers.
- Linear `H -> 3H` and `H -> H` for short-conv layers.
- Depthwise causal Conv1d: input `[B, H, S]`, groups `H`, kernel/state length `conv_L_cache`, padding `L-1`, output cropped to `S`.
- SwiGLU MLP: `w2(silu(w1(x)) * w3(x))`, with `H -> I`, `H -> I`, `I -> H`.
- Final LM head `H -> vocab_size`; tied with token embedding when `tie_word_embeddings=true`.

Attention primitives:

- Causal self-attention only, no cross-attention.
- GQA repeat or grouped attention with Q heads `num_attention_heads`, KV heads `num_key_value_heads`, group count `QH / KVH`.
- Additive causal/padding mask for full-attention layers.
- SDPA/Flash/Flex attention-compatible backend path, plus eager fallback.

Position/cache ops:

- RoPE cos/sin generation from dynamic `position_ids`.
- Apply RoPE to q/k before cache update.
- Dynamic KV cache for attention layers: key/value `[B, KVH, T, D]`.
- Fixed conv cache for conv layers: `conv_states` `[B, H, L_cache]`.
- Beam reorder must index both KV layers and conv-state layers on batch dim.

Preprocessing-coupled ops:

- 2D `attention_mask` is used directly for conv layers during prefill, but skipped for one-token decode by setting `linear_attention=None`.
- Tokenizer/generation prompt behavior was not inspected; keep it outside the first neural graph target.

## 5. Layer/block breakdown

Full-attention decoder block, repeated for indices where `layer_types[i] == "full_attention"`:

```text
residual = x
u = RMSNorm_H(x)
q = Linear(H -> num_attention_heads * D, bias=False)(u)
k = Linear(H -> num_key_value_heads * D, bias=False)(u)
v = Linear(H -> num_key_value_heads * D, bias=False)(u)
q = RMSNorm_D(q.view(B,S,QH,D)).transpose(1,2)
k = RMSNorm_D(k.view(B,S,KVH,D)).transpose(1,2)
v = v.view(B,S,KVH,D).transpose(1,2)
q,k = RoPE(q,k, cos[position_ids], sin[position_ids])
k,v = cache.update(k,v, layer_idx) if cache enabled
a = causal_gqa_attention(q,k,v, mask, scale=D**-0.5)
x = residual + Linear(QH * D -> H, bias=False)(a.transpose(1,2).reshape(B,S,QH*D))
x = x + Linear(I -> H, bias=False)(silu(Linear(H -> I)(RMSNorm_H(x))) * Linear(H -> I)(RMSNorm_H(x)))
```

Conv decoder block, repeated for indices where `layer_types[i] == "conv"`:

```text
residual = x
u = RMSNorm_H(x)
u = u * attention_mask[..., None] during multi-token masked prefill, if mask exists
BCx = Linear(H -> 3H, bias=conv_bias)(u).transpose(-1,-2)
B, C, x_proj = chunk(BCx, 3, channel_dim)
Bx = B * x_proj
conv_out = causal_depthwise_conv1d_or_state_update(Bx, weight[H, L_cache], bias?)
y = C * conv_out
x = residual + Linear(H -> H, bias=conv_bias)(y.transpose(-1,-2).contiguous())
x = x + SwiGLU_MLP(RMSNorm_H(x))
```

## 6. Attention requirements

- Variant: causal self-attention in selected layers only.
- Head pattern: GQA, representative configs use `D=64`, `KVH=8`, Q heads 16/24/32.
- Projection widths: q width `num_attention_heads * head_dim`; k/v width `num_key_value_heads * head_dim`; attention output width `num_attention_heads * head_dim`; output projection returns `hidden_size`.
- Masking: full-attention layers use `create_causal_mask`; eager path adds mask before softmax. Conv layers receive a 2D padding mask for prefill, not the causal 4D mask.
- RoPE: applied to q/k before cache update; cached keys are stored after RoPE.
- Cache: full-attention cache grows with sequence length as `[B, KVH, T, D]`; conv cache is fixed `[B, H, L_cache]`.
- Decode: one-token decode should not pass conv padding mask (`linear_attention=None`) and should use conv state update instead of full convolution.
- Backend compatibility: source declares FlashAttention, SDPA, and Flex attention support, but eager fallback exactly repeats KV heads before matmul. DinoML can avoid explicit repeat in a grouped-attention kernel if results match.
- Packed/varlen: no explicit cu_seqlens or packed varlen source path in this family.
- Sliding/local attention: none in `lfm2`; do not confuse conv layers with sliding-window attention.

## 7. Position encoding and custom math

Default RoPE:

```python
def lfm2_inv_freq(head_dim, rope_theta):
    return 1.0 / (rope_theta ** (arange(0, head_dim, 2, fp32) / head_dim))

def lfm2_rope_tables(position_ids, inv_freq, dtype):
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def lfm2_apply_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Cos/sin can be precomputed for static position buckets, but decode still needs position offset from cache length. Advanced `rope_type` values are delegated through shared Transformers RoPE utilities; first DinoML admission should allow `rope_type="default"` and reject or route others until audited.

## 8. Preprocessing and input packing

The neural graph inputs are ordinary text tensors:

- `input_ids`: `[B, S]`, exactly one of `input_ids` or `inputs_embeds`.
- `inputs_embeds`: optional `[B, S, H]`.
- `attention_mask`: expected as a 2D token mask for caller input; full-attention layers receive a causal additive mask built from it, conv layers use the 2D mask directly during multi-token prefill.
- `position_ids`: optional `[B, S]`; if absent, source builds `arange(S) + past_seen_tokens`.
- `past_key_values`: hybrid cache with attention KV plus conv state.

No image/audio/video processors, placeholder scatter, packed patch metadata, or modality token ABI exists in this `lfm2` source. `lfm2_vl` and audio ONNX wrappers are separate audit targets.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Separate Q/K/V projections -> packed projection

Source pattern:

```text
q = Linear(H -> QH*D)(x)
k = Linear(H -> KVH*D)(x)
v = Linear(H -> KVH*D)(x)
```

Replacement:

```text
qkv = Linear(H -> (QH + 2*KVH)*D)(x)
q,k,v = split(qkv, [QH*D, KVH*D, KVH*D], last_dim)
```

Preconditions:

- All three projections are biasless or have compatible bias packing.
- Same normalized input tensor and dtype.
- Packed row order is exactly `[q rows, k rows, v rows]` using PyTorch Linear weight layout `[out_features, in_features]`.
- Keep q/k per-head RMSNorm after split; do not normalize v.

Failure cases: quantized or sharded weights without a defined packed storage plan, mismatched input aliases, or config-specific projection bias.

Parity test sketch: compare q/k/v tensors before RoPE for random `[B,S,H]`, then one-layer logits with packed vs unpacked projection.

### Rewrite: ShortConv prefill -> fused gated causal depthwise FIR

Source pattern:

```text
in_proj -> split B,C,x -> B*x -> depthwise causal Conv1d -> C*conv -> out_proj
```

Replacement:

```text
single fused kernel over [B,S,H] computing projection split, gated product, causal depthwise FIR, output gate, optional out_proj handoff
```

Preconditions:

- `groups == hidden_size`, `kernel_size == conv_L_cache`, `padding == conv_L_cache - 1`, dilation 1, stride 1.
- Crop source conv output to `[..., :seqlen]`.
- Preserve prefill mask multiplication before `in_proj`.
- Conv weight is transformed from `[H,1,L]` to `[H,L]`.

Failure cases: non-depthwise conv, different padding/crop semantics, dynamic kernel length without provider support.

Parity test sketch: compare full prefill conv block for masked and unmasked inputs, including prompt length shorter/equal/longer than `L_cache`.

### Rewrite: ShortConv decode -> state update kernel

Source pattern:

```text
conv_state = roll/append(Bx_token)
conv_out = sum(conv_state * weight[H,L])
```

Replacement:

```text
static-address conv_state[B,H,L] update + per-channel dot
```

Preconditions:

- Decode query length is 1.
- Cache state already initialized or initialized from padded prefill state.
- Batch reorder updates conv state on dim 0.

Failure cases: multi-token decode step without rolling semantics, cache missing layer type metadata, external cache object not matching LFM2 `layer_types`.

### Rewrite: Last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for `logits_to_keep=1`, gather last hidden row before GEMM and run `gemm_rrr`/`gemm_rcr` on `[B,H] x [vocab,H]`.

Preconditions: generation target only needs final token logits; loss/training path disabled.

Failure cases: caller requests full logits or tensor-valued `logits_to_keep`.

## 10. Kernel fusion candidates

Highest priority:

- Hybrid cache ABI: represent per-layer KV-vs-conv state explicitly before lowering.
- RMSNorm for `H` and `D`, fp32 accumulation with bf16/fp16 storage.
- GQA FlashAttention/SDPA for full-attention layers, with cached RoPE keys and no explicit KV repeat.
- ShortConv fused prefill and decode state update; this is the distinctive LFM2 operator.
- SwiGLU MLP fused activation/multiply between two GEMMs.

Medium priority:

- Packed QKV projection plus q/k per-head RMSNorm and RoPE handoff.
- Last-token-only logits and tied embedding/LM-head alias handling.
- Masked prefill fast path for conv layers where padding mask is all ones.

Lower priority:

- Advanced RoPE variants beyond default.
- Tensor-parallel plan support for `lm_head`.
- Training loss, output attentions, hidden-state capture.

## 11. Runtime staging plan

1. Parse config and normalize aliases: `layer_types`, `rope_parameters`, effective MLP width, `tie_embedding`.
2. Load weights for embeddings, RMSNorms, attention projections, short-conv weights, MLPs, and LM head with alias preservation for tied weights.
3. Build one-block parity fixtures for one attention layer and one conv layer.
4. Implement full prefill for a small representative config with hybrid cache creation.
5. Implement one-token decode with KV append and conv-state update.
6. Enable optimized GQA attention and short-conv provider paths.
7. Add graph rewrites: packed QKV, fused short-conv, last-token logits, optional MLP fusion.

Initially stub or defer tokenizer/chat template, beam search, tensor parallel, advanced RoPE, and training loss.

## 12. Parity and validation plan

- RMSNorm random tests for `[B,S,H]` and `[B,S,heads,D]`, fp32/fp16/bf16 tolerances.
- RoPE table and apply parity across position offsets, including decode position after nonzero cache length.
- ShortConv operator parity:
  - prefill without mask,
  - prefill with padding mask,
  - prompt length `< L_cache`, `== L_cache`, and `> L_cache`,
  - one-token decode after prefill.
- Attention layer parity for GQA with and without cache, comparing pre-RoPE q/k, post-cache k/v shapes, and output.
- Single full decoder layer parity for attention and conv layer types.
- N-layer parity for the published 16-layer pattern using random weights or a tiny checkpoint.
- Prefill logits parity with `logits_to_keep=0` and `logits_to_keep=1`.
- Decode token parity for several incremental steps and batch reorder.

Suggested tolerances: fp32 `1e-5` absolute/relative for small blocks; bf16/fp16 block-level `1e-2` style tolerance after fused attention/conv paths, with source eager fp32-accumulating RMSNorm/softmax preserved.

## 13. Performance probes

- Prefill-only throughput by sequence length for hybrid 16-layer and 30-layer patterns.
- Decode tokens/sec with batch sweep, separating attention layers from conv layers.
- ShortConv prefill kernel time vs generic depthwise Conv1d lowering.
- ShortConv decode state update time and memory bandwidth.
- GQA attention backend comparison: eager repeat, grouped SDPA/Flash, and cache layout variants.
- KV cache memory plus conv-state memory by model size and context length.
- MLP GEMM throughput for effective intermediate sizes 4608, 6912, 8192, 10752.
- Last-token-only logits GEMM cost for vocab 65536 and tied/untied weights.
- Packed QKV projection vs separate GEMMs.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Output attentions and hidden-state capture.
- Beam search beyond cache reorder validation.
- Tensor-parallel and pipeline-parallel plans.
- `lfm2_moe`, `lfm2_vl`, audio, ONNX, MLX, GGUF-specific loading, and quantized packed weight formats.
- Advanced RoPE types unless a config uses them and they are admitted with explicit tests.
- General Conv1d support beyond the guarded depthwise causal short-conv pattern.

## 15. Final implementation checklist

- [ ] Parse `Lfm2Config` aliases and normalize `layer_types`.
- [ ] Compute effective MLP intermediate width exactly.
- [ ] Parse/normalize `rope_parameters`; admit default RoPE first.
- [ ] Load tied embedding/LM-head weights without breaking alias semantics.
- [ ] Implement RMSNorm over residual width and head width.
- [ ] Implement LFM2 RoPE table/apply math.
- [ ] Implement GQA causal attention with KV cache for `full_attention` layers.
- [ ] Implement short-conv prefill and decode state ABI for `conv` layers.
- [ ] Implement hybrid cache manifest with per-layer KV or conv-state entries.
- [ ] Add safe packed-QKV rewrite with `[q,k,v]` row order.
- [ ] Add guarded short-conv fused rewrite/provider.
- [ ] Add last-token-only logits rewrite.
- [ ] Validate one attention block, one conv block, full prefill, and multi-step decode.
- [ ] Benchmark prefill, decode, short-conv, attention backend, and logits bottlenecks.
