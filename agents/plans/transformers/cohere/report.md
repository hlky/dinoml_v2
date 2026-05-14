# Cohere Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: cohere
Primary runtime target: causal LM prefill/decode for text generation
Config source: local Transformers source defaults plus HF config snapshots; production CohereForAI repos were gated from this environment.
Source files inspected:
- X:/H/transformers/src/transformers/models/cohere/configuration_cohere.py
- X:/H/transformers/src/transformers/models/cohere/modeling_cohere.py
- X:/H/transformers/src/transformers/models/cohere/modular_cohere.py
- X:/H/transformers/src/transformers/models/cohere/tokenization_cohere.py
- X:/H/transformers/src/transformers/models/cohere2/configuration_cohere2.py
- X:/H/transformers/src/transformers/models/cohere2/modeling_cohere2.py
Snapshots written under: agents/plans/transformers/cohere/_sources/
Any missing files or assumptions: official production `CohereForAI/*` config URLs returned 401; open HF mirrors/quant repos are labeled as mirrors below. `modeling_cohere.py` is generated from `modular_cohere.py`; future source edits should target the modular file.
```

Primary source URLs:
- [configuration_cohere.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere/configuration_cohere.py)
- [modeling_cohere.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere/modeling_cohere.py)
- [modular_cohere.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere/modular_cohere.py)
- [tokenization_cohere.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere/tokenization_cohere.py)
- [configuration_cohere2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere2/configuration_cohere2.py)
- [modeling_cohere2.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere2/modeling_cohere2.py)

Config snapshots:
- Official small/debug: [hf-internal-testing/cohere-random config](https://huggingface.co/hf-internal-testing/cohere-random/raw/main/config.json)
- Open mirror for Command-R v01: [second-state/C4AI-Command-R-v01-GGUF config](https://huggingface.co/second-state/C4AI-Command-R-v01-GGUF/blob/main/config.json)
- Open mirror for Command-R+: [alpindale/c4ai-command-r-plus-GPTQ config](https://huggingface.co/alpindale/c4ai-command-r-plus-GPTQ/blob/main/config.json)
- Open mirror for Aya-23-8B: [mlx-community/aya-23-8B-8bit config](https://huggingface.co/mlx-community/aya-23-8B-8bit/commit/c956faef84ad39e0e39029dc7db83170e18b101c)
- Open mirror for Aya-23-35B: [Zoyd/CohereForAI_aya-23-35B-2_5bpw_exl2 config](https://huggingface.co/Zoyd/CohereForAI_aya-23-35B-2_5bpw_exl2/commit/c883c447a0cc75b232633463cabc67c222a9ab0e)

## 2. High-level architecture

Cohere is a text-only decoder causal LM. The runtime graph is:

```text
tokenizer/left padding -> token embedding -> N decoder blocks -> final layer norm -> lm_head -> logit_scale -> logits/sampling
```

Each decoder block is a parallel-residual transformer block: one input layer norm feeds both self-attention and SwiGLU MLP, then the residual, attention output, and MLP output are added together. There is no encoder, cross-attention, vision branch, audio branch, MoE, recurrence, or hidden-state memory beyond autoregressive KV cache.

Independently stageable parts:
- CPU/data pipeline: Cohere byte-level BPE tokenizer, NFC normalization, digit splitting, left padding, chat/tool/RAG templates.
- GPU prefill: embeddings, full causal self-attention, MLP, final norm, logits.
- GPU decode: one-token or short-step attention with per-layer KV cache.
- Optional optimization: last-token-only logits via `logits_to_keep`.

## 3. Important config dimensions

Source defaults from `CohereConfig`:

| Field | Source default | Runtime significance |
|---|---:|---|
| `vocab_size` | 256000 | Embedding and LM head width. |
| `hidden_size` | 8192 | Decoder hidden width. |
| `intermediate_size` | 22528 | MLP gate/up width. |
| `num_hidden_layers` | 40 | Decoder block count. |
| `num_attention_heads` | 64 | Query heads. |
| `num_key_value_heads` | `None -> num_attention_heads` | MHA by default; configs may set GQA. |
| `head_dim` | inferred as `hidden_size // num_attention_heads` | Projection output width; not stored by CohereConfig. |
| `max_position_embeddings` | 8192 | RoPE/cache admission baseline. |
| `default_theta` | 500000.0 | Used when legacy configs omit `rope_theta`. |
| `rope_parameters` | `None`, standardized by config base | Modeling reads `config.rope_parameters["rope_type"]` and `["rope_theta"]`. |
| `logit_scale` | 0.0625 | Required post-LM-head multiply. |
| `attention_bias` | false | Q/K/V/O projections are biasless by default. |
| `use_qk_norm` | false | Optional per-head LayerNorm on Q/K before transpose/RoPE/cache. |
| `tie_word_embeddings` | true | LM head weight aliases token embedding when tied. |
| `use_cache` | true | Dynamic autoregressive KV cache enabled. |
| `hidden_act` | `silu` | SwiGLU activation. |

Representative config sweep:

| Checkpoint/config source | Basis | `model_type` | Layers | Hidden | Heads/KV | Head dim | MLP | RoPE theta | QK norm | Logit scale | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `hf-internal-testing/cohere-random` | official debug config | `cohere` | 2 | 8192 | 64/64 | 128 | 22528 | 10000 | omitted -> false | 0.0625 | Tiny layer count but production-sized widths. |
| `CohereForAI/c4ai-command-r-v01` via second-state mirror | open mirror of gated original | `cohere` | 40 | 8192 | 64/64 | 128 | 22528 | 8000000 | omitted -> false | 0.0625 | MHA, tied embeddings present in mirror. |
| `CohereForAI/c4ai-command-r-plus` via alpindale GPTQ mirror | open mirror of gated original | `cohere` | 64 | 12288 | 96/8 | 128 | 33792 | 75000000 | true | 0.8333333333 | GQA plus QK norm; quant metadata is mirror-specific loading concern. |
| `CohereForAI/aya-23-8B` via mlx mirror | open mirror of gated original | `cohere` | 32 | 4096 | 32/8 | 128 | 14336 | 10000 | false | 0.0625 | GQA, smaller hidden/MLP. |
| `CohereForAI/aya-23-35B` via Zoyd EXL2 mirror | open mirror of gated original | `cohere` | 40 | 8192 | 64/64 | 128 | 22528 | 8000000 | false | 0.0625 | MHA; EXL2 quant metadata is not native Transformers runtime behavior. |

## 3a. Family variation traps

- `num_key_value_heads` varies. Command-R v01 and Aya-23-35B are MHA (`KV heads == Q heads`), while Command-R+ and Aya-23-8B use GQA (`KV heads < Q heads`). Cache and attention kernels must not assume MHA.
- `use_qk_norm` is config-dependent and materially changes attention. When enabled, Q and K are normalized over `(heads, head_dim)` / `(kv_heads, head_dim)` before transpose, RoPE, and cache update.
- `rope_theta` varies widely: 10000, 8000000, and 75000000 in inspected configs. Current source standardizes legacy `rope_theta`/`rope_scaling` into `rope_parameters`; DinoML should preserve the effective standardized values.
- Cohere RoPE differs from Llama: frequencies are `repeat_interleave`d, and `rotate_half` splits even/odd elements rather than first/second halves.
- `logit_scale` is not cosmetic. Source multiplies logits after the LM head; Command-R+ mirror uses `0.8333333333`, much larger than the default `0.0625`.
- `tie_word_embeddings` defaults true and `CohereForCausalLM` declares `lm_head.weight` tied to `model.embed_tokens.weight`. Lowering must preserve alias identity.
- Some mirrors include `quantization`, `quantization_config`, `auto_map`, `pretraining_tp`, or old `model_max_length`. Native source does not read those in the forward path; treat them as loader/admission metadata, not core graph ops.
- Tokenizer is left-padding and model input names are only `input_ids` and `attention_mask`. No token type IDs enter the model.
- Cohere2 is not a drop-in variant of Cohere: it has `sliding_window`, generated `layer_types`, sliding-window causal masks, explicit `head_dim`, and no QK norm. In the inspected Cohere2 source, RoPE is applied only when `self.sliding_window is not None`; Cohere applies RoPE in every layer.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- View/reshape: projected Q/K/V `[B,S,*] -> [B,S,heads,head_dim]`.
- Transpose: Q/K/V `[B,S,heads,D] -> [B,heads,S,D]`.
- Contiguous/reshape after attention: `[B,heads,S,D] -> [B,S,H]`.
- Slicing/gather for `logits_to_keep`: integer tail slice or tensor indices on sequence dimension.
- Left-padding attention mask ingestion.

Neural network primitives:
- Biasless Linear for Q/K/V/O unless `attention_bias=true`.
- Biasless MLP `gate_proj`, `up_proj`, `down_proj`.
- SiLU and multiply for SwiGLU: `down(silu(gate(x)) * up(x))`.
- Cohere LayerNorm: mean/variance over last dim, fp32 compute, learned weight, no bias. Used for input norm, final norm, and optional Q/K norm.
- Residual add of three tensors: `residual + attention + mlp`.

Attention primitives:
- Causal self-attention only.
- MHA or GQA depending `num_key_value_heads`.
- Query scaling by `head_dim ** -0.5`.
- Mask addition before softmax.
- Softmax in fp32, cast back to query dtype.
- Dropout is training-only; inference uses zero.
- Backend dispatch supports eager, SDPA, FlashAttention, and FlexAttention through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:
- Standardized RoPE parameter parsing.
- Cohere interleaved RoPE frequency construction.
- Cohere even/odd `rotate_half`.
- Dynamic RoPE update decorator for advanced RoPE types, although inspected production configs use default/legacy theta fields.

Generation/cache ops:
- Dynamic cache creation when `use_cache` and no cache is supplied.
- Position IDs from `past_key_values.get_seq_length()`.
- Per-layer cache update stores K/V after optional QK norm and RoPE, before GQA repeat.
- Cache tensor logical shape before repeat: K/V `[B, num_key_value_heads, cached_seq, head_dim]`.
- Eager repeat expansion only for attention compute: `[B, num_attention_heads, cached_seq, head_dim]`.
- Beam reorder through Transformers cache abstraction.

Preprocessing-coupled ops:
- Byte-level BPE tokenizer with NFC normalization.
- Digit splitting before ByteLevel pre-tokenization.
- Left padding and attention mask.
- Chat/tool/RAG template rendering is CPU/controller work, not GPU graph work.

Quantized/packed weight metadata:
- Native `cohere` source has dense `nn.Linear` modules. Mirror GPTQ/EXL2/MLX metadata should be admitted as loading/provider work only if DinoML implements those formats; otherwise require dense/safetensors weights or a known dequant path.
- Transformers GGUF helper maps `model_type == "cohere"` to GGUF architecture name `command-r`.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = CohereLayerNorm(x)                         # [B,S,H]

q = q_proj(x_norm).view(B,S,num_heads,head_dim)
k = k_proj(x_norm).view(B,S,num_kv_heads,head_dim)
v = v_proj(x_norm).view(B,S,num_kv_heads,head_dim)
if use_qk_norm:
  q = CohereLayerNorm((num_heads, head_dim))(q)
  k = CohereLayerNorm((num_kv_heads, head_dim))(k)
q,k,v = transpose to [B,heads_or_kv,S,D]
q,k = CohereRoPE(q,k, cos, sin)
k,v = cache.update(k,v,layer_idx)                   # cache stores post-RoPE K
attn = causal_attention(q,k,v, repeat_kv if needed)
attn = o_proj(attn.reshape(B,S,num_heads*D))

mlp = down_proj(silu(gate_proj(x_norm)) * up_proj(x_norm))
x = residual + attn + mlp
```

Final model:

```text
hidden = embed_tokens(input_ids)
position_ids = arange(S) + cache_seen_tokens
cos,sin = rotary_emb(hidden, position_ids)
hidden = decoder_blocks(hidden, causal_mask, cos/sin, cache)
hidden = final CohereLayerNorm(hidden)
logits = lm_head(hidden[:, logits_to_keep, :]) * logit_scale
```

Projection shapes for common configs:
- Command-R v01/Aya-23-35B: Q/K/V/O are effectively `8192 -> 8192`, MLP `8192 -> 22528 -> 8192`.
- Command-R+: Q `12288 -> 12288`, K/V `12288 -> 1024`, O `12288 -> 12288`, MLP `12288 -> 33792 -> 12288`.
- Aya-23-8B: Q `4096 -> 4096`, K/V `4096 -> 1024`, O `4096 -> 4096`, MLP `4096 -> 14336 -> 4096`.

## 6. Attention requirements

Cohere requires causal self-attention for prefill and decode.

| Requirement | Cohere behavior |
|---|---|
| Causal/noncausal | Causal only. |
| Self/cross | Self-attention only. |
| MHA/GQA | Both, config-dependent. `num_key_value_groups = num_attention_heads // num_key_value_heads`. |
| Head dim | Source uses `getattr(config, "head_dim", hidden_size // num_attention_heads)`; inspected configs derive 128. |
| Masking | `create_causal_mask` builds mask; eager path adds it to scores before fp32 softmax. |
| Sliding/local | Not in `cohere`; this is a Cohere2 feature. |
| RoPE/cache order | Q/K projections -> optional QK norm -> transpose -> RoPE -> cache update. Cached K is post-RoPE. |
| Cache layout | Per layer K/V `[B, num_key_value_heads, T_cache, head_dim]`; repeat to query heads is compute-local. |
| Backend compatibility | Source declares FlashAttention, SDPA, FlexAttention support and dispatches by `_attn_implementation`. |
| Eager fallback | Matmul QK, mask add, fp32 softmax, dropout, matmul V. Fine for parity; too slow for production prefill/decode. |

Admission rules for DinoML:
- Require `hidden_size % num_attention_heads == 0`.
- Require `num_attention_heads % num_key_value_heads == 0`.
- Preserve optional QK norm order before RoPE/cache.
- Preserve post-RoPE cache layout; do not cache pre-RoPE K.
- Reject Cohere2 configs in this family path unless routed to a separate `cohere2` audit/runtime.

## 7. Position encoding and custom math

Cohere RoPE is the largest parity trap. It does not use Llama's first-half/second-half rotation.

```python
def cohere_inv_freq(theta, head_dim, device):
    i = torch.arange(0, head_dim, 2, dtype=torch.float, device=device)
    return 1.0 / (theta ** (i / head_dim))

def cohere_cos_sin(inv_freq, position_ids, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def cohere_rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def cohere_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    return ((qf * cos) + (cohere_rotate_half(qf) * sin)).to(q.dtype), \
           ((kf * cos) + (cohere_rotate_half(kf) * sin)).to(k.dtype)
```

Precompute candidates:
- `inv_freq` depends on `rope_theta` and `head_dim`.
- `cos/sin` depend on runtime `position_ids`, cache length, dtype, and device. They can be cached per sequence position bucket but must honor decode offset.
- Dynamic/non-default RoPE types flow through `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; first integration can admit only default standardized `rope_type="default"` unless a checkpoint requires otherwise.

## 8. Preprocessing and input packing

Tokenizer/runtime inputs:
- `CohereTokenizer` is byte-level BPE with NFC normalization, digit splitting, and ByteLevel pre-tokenization.
- Padding side is left.
- Model inputs are `input_ids` and `attention_mask`; no segment IDs.
- Default special token IDs in config: pad `0`, BOS `5`, EOS `255001`.
- `position_ids` are optional; if omitted, source constructs `[0..S-1] + past_seen_tokens`.
- Chat, RAG, and tool-use templates affect prompt text/token IDs but are controller/data-pipeline work.

GPU graph inputs for first integration:
- `input_ids [B,S]` or precomputed `inputs_embeds [B,S,H]`, exactly one required.
- `attention_mask` accepted and converted to a causal mask.
- Optional `past_key_values` with per-layer K/V.
- Optional `position_ids [1 or B,S]`; if omitted, runtime may synthesize them from cache length.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV/GQA projection

Source pattern:
```text
q = Linear(H -> num_heads*D)
k = Linear(H -> num_kv_heads*D)
v = Linear(H -> num_kv_heads*D)
```

Replacement:
```text
packed = Linear(H -> (num_heads + 2*num_kv_heads)*D)
split packed as [Q_all, K_all, V_all]
```

Preconditions:
- Same input tensor and dtype for all three projections.
- All three projections have matching bias policy.
- Weight transform concatenates rows in exact source order Q then K then V.
- Preserve optional QK norm after split and before transpose/RoPE.

Failure cases:
- Quantized/provider formats whose packing metadata cannot be transformed safely.
- Tensor-parallel sharding where source row partitions are already externally managed.

Parity test sketch:
- Random hidden states, random weights, both MHA and GQA configs, compare Q/K/V tensors before RoPE.

### Rewrite: Cohere attention to fused GQA attention

Source pattern:
```text
Q/K/V projections -> optional QK norm -> Cohere RoPE -> cache update -> repeat_kv -> matmul/softmax/matmul
```

Replacement:
```text
FusedCausalGQAAttention(q, k_cache_post_rope, v_cache, mask, scale)
```

Preconditions:
- Cache stores post-RoPE K in `[B,KV,T,D]`.
- Fused kernel supports GQA without materializing repeat, fp32 softmax accumulation, and additive causal/padding mask.
- RoPE implementation is Cohere even/odd/interleaved variant.

Failure cases:
- Non-default RoPE not implemented.
- QK norm enabled but not fused or correctly placed.
- Attention backend requires dense attention output tensors for debugging.

Parity test sketch:
- Prefill and decode one step for MHA and GQA, with/without QK norm, compare hidden states and cache tensors.

### Rewrite: SwiGLU MLP fusion

Source pattern:
```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:
```text
dual GEMM gate/up -> fused SiLU*multiply -> down GEMM
```

Preconditions:
- Biasless gate/up/down as in source, or explicit bias support if config changes.
- Intermediate size static for provider planning.

Parity test sketch:
- Compare MLP output for fp32/fp16/bf16 with tolerances; include Aya-23-8B and Command-R+ dimensions.

### Rewrite: last-token-only logits

Source pattern:
```text
logits = lm_head(hidden_states[:, slice_indices, :]) * logit_scale
```

Replacement:
```text
slice hidden before GEMM; compute only requested sequence positions
```

Preconditions:
- `logits_to_keep` is an integer tail count or concrete tensor indices.
- Sampling only needs final/token-selected logits.
- Preserve tied weight alias and `logit_scale`.

Failure cases:
- Caller requests full logits, loss, or arbitrary hidden-state consumers.

## 10. Kernel fusion candidates

Highest priority:
- Cohere LayerNorm: used once per block plus final norm and optional Q/K norm; fp32 mean/variance is required.
- Cohere RoPE: parity-sensitive interleaved/even-odd variant; fuse with Q/K layout conversion where possible.
- GQA causal attention with KV cache: needed for Command-R+, Aya-23-8B, and efficient decode.
- SwiGLU MLP: dominant GEMM/elementwise pattern.
- Last-token-only LM head with logit scale: avoids full-vocab GEMM over all prefill positions during decode.

Medium priority:
- Packed QKV projection: reduces launch overhead and improves memory locality.
- Parallel residual block fusion around two branches where memory planning allows.
- Optional QK norm fusion into attention pre-processing.
- Tied embedding/LM-head loader alias preservation.

Lower priority:
- Training-only dropout/loss paths.
- Debug attention weights reconstruction.
- Native GPTQ/EXL2/MLX quant metadata support from mirrors; useful later but not required for dense Transformers parity.

## 11. Runtime staging plan

Stage 1: Parse `CohereConfig`, normalize legacy `rope_theta` into an effective RoPE parameter record, load dense weights, preserve tied embedding alias.

Stage 2: Single-block parity in fp32 with no cache, MHA, default RoPE, no QK norm.

Stage 3: Full prefill parity for Command-R v01/Aya-23-35B style MHA, including final norm, LM head, and `logit_scale`.

Stage 4: Add GQA and KV cache decode for Aya-23-8B/Command-R+ shapes. Verify cache stores post-RoPE K and unexpanded K/V heads.

Stage 5: Add optional QK norm for Command-R+ configs.

Stage 6: Enable optimized fused attention, packed QKV, SwiGLU fusion, and last-token-only logits.

Stage 7: Add loader/provider support for accepted quantized formats or GGUF `command-r` mapping if needed; otherwise reject those configs clearly.

Initially stub/defer: training loss, dropout, gradient checkpointing, output attentions, tensor parallel plans, GPTQ/EXL2 execution, and Cohere2 sliding-window behavior.

## 12. Parity and validation plan

- Config parsing tests: source defaults, legacy `rope_theta`, `num_key_value_heads=None`, `tie_word_embeddings`, `use_qk_norm`.
- Custom op tests: Cohere LayerNorm vs PyTorch source; Cohere RoPE vs source for even/odd rotation; repeat_kv for MHA/GQA.
- Single-layer tests: random weights for MHA and GQA; compare post-attention, MLP, and block output.
- Cache tests: prefill cache shape `[B,KV,S,D]`; decode appends one token; compare K after RoPE and V after projection.
- Full model smoke: `hf-internal-testing/cohere-random` logits, including `logit_scale`.
- Representative dimension tests: Aya-23-8B GQA, Command-R+ GQA+QK norm, Command-R v01 MHA.
- Tied weight test: embedding and LM head share one logical parameter when `tie_word_embeddings=true`.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 hidden/logits `rtol=1e-2, atol=1e-2`, with tighter custom-op fp32 references where practical.

## 13. Performance probes

- Prefill tokens/sec sweep over `B,S` for MHA and GQA configs.
- Decode tokens/sec sweep with cache lengths 1K, 8K, 32K, 128K where config/model admission permits.
- KV cache memory usage: MHA Command-R v01 vs GQA Command-R+/Aya-23-8B.
- Attention backend comparison: eager parity, SDPA baseline, fused GQA kernel, FlashAttention-style path.
- QK norm overhead for Command-R+.
- MLP throughput by hidden/intermediate shape.
- LM-head throughput with full logits vs last-token-only logits.
- RoPE generation/cache overhead for dynamic positions.
- Dense loader vs GGUF/quant dequant/provider path if non-dense weights are admitted.

## 14. Skip/defer list

- Training, loss, gradients, gradient checkpointing.
- Dropout behavior outside inference.
- Beam-search controller internals beyond cache reorder support.
- Output attentions and dense attention weight materialization.
- Tensor parallel and pipeline parallel plans.
- GPTQ/EXL2/MLX quantized execution from mirror configs.
- Cohere2 sliding-window/full-attention alternation; route to separate family support.
- Cohere2 vision and Cohere ASR families.
- Non-default dynamic/YARN/longrope RoPE unless a target checkpoint requires it.

## 15. Final implementation checklist

- [ ] Parse `CohereConfig` and effective `rope_parameters`.
- [ ] Reject or route `model_type=cohere2` separately.
- [ ] Load dense embeddings, linears, norms, and tied LM head alias.
- [ ] Implement Cohere LayerNorm with fp32 mean/variance and no bias.
- [ ] Implement Cohere interleaved RoPE and even/odd `rotate_half`.
- [ ] Implement MHA/GQA attention with post-RoPE K cache layout `[B,KV,T,D]`.
- [ ] Implement optional QK norm before transpose/RoPE/cache.
- [ ] Implement SwiGLU MLP.
- [ ] Implement parallel residual block add.
- [ ] Implement final norm, LM head, `logit_scale`, and `logits_to_keep`.
- [ ] Add single-block MHA/GQA parity tests.
- [ ] Add decode cache append/reorder parity tests.
- [ ] Add full small-checkpoint logits parity using `hf-internal-testing/cohere-random`.
- [ ] Add Command-R+/Aya dimension compile/load tests.
- [ ] Benchmark prefill, decode, KV memory, MLP, and LM-head slicing.
