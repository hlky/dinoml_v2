# SmolLM3 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: HuggingFaceTB/SmolLM3-3B
Config source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B/raw/main/config.json
Source files inspected:
  transformers/src/transformers/models/smollm3/configuration_smollm3.py
  transformers/src/transformers/models/smollm3/modeling_smollm3.py
  transformers/src/transformers/models/smollm3/modular_smollm3.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/modeling_rope_utils.py
  transformers/tests/models/smollm3/test_modeling_smollm3.py
Any missing files or assumptions:
  No processor/image/audio files exist for this family. The report targets native
  in-library PyTorch SmolLM3ForCausalLM, not ONNX, GGUF, or remote-code variants.
```

`configuration_smollm3.py` and `modeling_smollm3.py` are generated from `modular_smollm3.py`; future Transformers source edits should be made in the modular file, while DinoML should lower the generated modeling file behavior.

Representative configs inspected:

- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), main `config.json`, `generation_config.json`, `tokenizer_config.json`, and `model.safetensors.index.json`.
- [HuggingFaceTB/SmolLM3-3B-Base](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base), main `config.json`.
- [HuggingFaceTB/SmolLM3-3B-ONNX](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-ONNX), main `config.json`.
- [h-d-h/smollm3-sft](https://huggingface.co/h-d-h/smollm3-sft), main `config.json`; this is an open fine-tune/mirror, not an official base checkpoint.

No gated access was encountered for the inspected configs. GGUF and ONNX repos are conversion/export artifacts and should be admitted through separate loader/provider audits.

## 2. High-level architecture

SmolLM3 is a text-only autoregressive decoder with GQA self-attention, RMSNorm, SwiGLU MLPs, tied token embedding and LM head weights, and a per-layer choice between RoPE and NoPE. The primary DinoML runtime target should be `SmolLM3ForCausalLM`.

```text
tokenizer/CPU input_ids + attention_mask
  -> token embedding [B,S] -> [B,S,2048]
  -> 36 decoder blocks with causal self-attention
  -> final RMSNorm
  -> tied LM head [B,S,2048] x [2048,128256]
  -> logits / sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, BOS/EOS/pad handling, attention mask construction inputs.
- GPU prefill: embedding, full causal attention over prompt, RoPE table application on RoPE-enabled layers, logits for selected positions.
- GPU decode: one or more new tokens with per-layer KV cache update.
- Generation controller: sampling settings from `generation_config.json` such as temperature and top-p are outside the core neural graph.

Classification, token-classification, and QA heads are implemented through generic Transformers wrappers. They are optional/deferred for a causal-LM-first integration.

## 3. Important config dimensions

Main public checkpoints share the same neural dimensions:

| Field | Source default | SmolLM3-3B / Base / ONNX | DinoML note |
| --- | ---: | ---: | --- |
| `vocab_size` | 128256 | 128256 | Embedding rows and LM head rows. |
| `hidden_size` | 2048 | 2048 | Residual width. |
| `num_hidden_layers` | 36 | 36 | Decoder blocks. |
| `num_attention_heads` | 16 | 16 | Query heads. |
| `num_key_value_heads` | 4 | 4 | GQA, 4x repeat to 16 query heads. |
| `head_dim` | inferred 128 | inferred 128 | Source may honor explicit `head_dim` if present. |
| `intermediate_size` | 11008 | 11008 | Gate/up width. |
| `max_position_embeddings` | 32768 | 65536 | Official configs override source default. |
| `rope_theta` | 2000000.0 default theta | 5000000.0 | Legacy `rope_theta` is standardized into `rope_parameters`. |
| `hidden_act` | `silu` | `silu` | SwiGLU: `silu(gate) * up`. |
| `rms_norm_eps` | 1e-6 | 1e-6 | RMSNorm fp32 variance. |
| `attention_bias` | false | false | Q/K/V/O bias absent. |
| `mlp_bias` | false | false | MLP bias absent. |
| `tie_word_embeddings` | true | true except h-d-h omits field | Missing field defaults to true. |
| `torch_dtype` / `dtype` | unset | bf16 for official, float32 in h-d-h config | Metadata/loading concern, not source math change. |
| `use_cache` | true | 3B/ONNX false, Base/h-d-h true | Source supports cache either way; config default controls forward/generation defaults. |

Representative checkpoint sweep:

| Checkpoint | Kind | Cache default | BOS/EOS/pad | Layer types | Export/extra fields | Operator-significant variation |
| --- | --- | --- | --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM3-3B` | Official instruct | `use_cache=false` | 128000 / 128012 / 128004 | 36 full attention | `pretraining_tp`, `max_window_layers`, `rope_scaling=null` | Same graph; generation config enables sampling. |
| `HuggingFaceTB/SmolLM3-3B-Base` | Official base | `use_cache=true` | null / 128001 / 128004 | omitted, computed full attention | `transformers.js_config` | Same graph; tokenizer/control differs. |
| `HuggingFaceTB/SmolLM3-3B-ONNX` | Official ONNX export | `use_cache=false` | 128000 / 128012 / 128004 | 36 full attention | `transformers.js_config` q4 metadata | Native source ignores export-specific quant metadata. |
| `h-d-h/smollm3-sft` | Open fine-tune/mirror | `use_cache=true` | null / 128001 / null | 36 full attention | `dtype=float32`, `transformers.js_config` | Same graph; not official source of family behavior. |

The safetensors index for `HuggingFaceTB/SmolLM3-3B` reports `total_parameters=3075098624`, `total_size=6150197248`, and no separate `lm_head.weight` entry, consistent with tied embeddings.

## 3a. Family variation traps

- `hidden_size == num_attention_heads * head_dim` for inspected configs, but source computes `head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)`. Do not assume equality for future configs without checking.
- GQA is required: `num_key_value_heads=4 < num_attention_heads=16`; cache is stored before repeat expansion as `[B,4,T,128]`.
- `no_rope_layers` is semantically inverted by name: source sets `self.use_rope = config.no_rope_layers[layer_idx]`. Value `1` means RoPE is applied, value `0` means NoPE.
- Default layer pattern is RoPE on layers 0,1,2, NoPE on layer 3, repeating every 4 layers. Official configs explicitly provide that pattern.
- `use_sliding_window=false` and `sliding_window=null` in inspected official configs. Source can create sliding attention only if both are enabled and the layer is a NoPE layer. DinoML should reject or separately stage sliding-window configs until local attention masks are implemented.
- `layer_types` in official configs are all `full_attention`; if absent, source computes all full attention for the inspected configs.
- `pretraining_tp`, `max_window_layers`, `rope_scaling=null`, and `transformers.js_config` appear in some configs but are not read by the inspected native modeling path.
- `use_cache=false` in the main instruct config does not remove the model cache ABI; it only changes default behavior.
- No NHWC/NCHW image layout exists. Relevant layout risk is sequence/head transpose and contiguous/view assumptions, not channel-last rewriting.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,2048]`.
- Rank-preserving reshape/view from projection output `[B,S,out] -> [B,S,heads,128]`.
- Transpose `[B,S,H,D] -> [B,H,S,D]` for Q/K/V and `[B,H,S,D] -> [B,S,H,D]` for attention output.
- Contiguous materialization before flattening attention output.
- Slice logits by sequence positions: `hidden_states[:, slice_indices, :]`.
- Residual adds on `[B,S,2048]`.

Neural primitives:

- RMSNorm over last dim: `[B,S,2048]`, fp32 variance, scale weight `[2048]`.
- Linear Q: `2048 -> 2048`, no bias.
- Linear K/V: `2048 -> 512`, no bias each.
- Linear O: `2048 -> 2048`, no bias.
- MLP gate/up: `2048 -> 11008`, no bias each.
- MLP down: `11008 -> 2048`, no bias.
- SiLU activation and elementwise multiply for SwiGLU.
- LM head: `2048 -> 128256`, no bias, tied to `model.embed_tokens.weight`.

Attention primitives:

- Causal self-attention, GQA/MQA-style repeat from 4 KV heads to 16 query heads.
- Eager fallback: `MatMul(Q,K^T) * 1/sqrt(128)`, add mask, fp32 softmax on last dim, dropout during training only, `MatMul(P,V)`.
- Backend dispatch through Transformers attention interface for eager, SDPA, FlashAttention, and flex attention. DinoML should treat backend choice as an optimization surface with eager math as parity reference.
- Optional sliding-window causal mask for non-default configs.

Position/rotary:

- Default RoPE with `rope_theta=5000000.0` on RoPE-enabled layers, head dim 128, cos/sin computed in fp32 then cast to model dtype.
- NoPE layers skip Q/K rotary application but still receive the same attention mask family.

Generation/cache:

- DynamicCache or compatible cache object. Per layer K/V update receives new `[B,4,S_new,128]` tensors and returns concatenated or indexed cache tensors.
- Cache reorder for beam search is inherited from Transformers cache/generation machinery; beam search can be deferred initially.

Preprocessing-coupled:

- Tokenizer emits `input_ids` and `attention_mask` only. There are no pixel, audio, region, codebook, scatter, or packed multimodal tensors.
- Chat-template and tool-use markers affect token IDs but not neural operators.

Quantized/packed metadata:

- Native source has no quantized weight format. ONNX `transformers.js_config` and GGUF repos are external conversion/provider contracts.

## 5. Layer/block breakdown

For `B=batch`, `S=query sequence length`, `T=cache length after update`, `D=128`, `Hq=16`, `Hkv=4`, hidden `C=2048`, intermediate `I=11008`:

```text
Embedding:
  input_ids [B,S] -> embed_tokens [B,S,2048]

Decoder block, repeated 36 times:
  residual = x [B,S,2048]
  h = RMSNorm(x, dim=-1)
  q = Linear(2048 -> 2048, bias=False)(h).view(B,S,16,128).transpose(1,2)
  k = Linear(2048 -> 512, bias=False)(h).view(B,S,4,128).transpose(1,2)
  v = Linear(2048 -> 512, bias=False)(h).view(B,S,4,128).transpose(1,2)
  if layer has RoPE:
      q,k = apply_rotary_pos_emb(q,k,cos,sin)
  if cache:
      k,v = cache.update(k,v,layer_idx)  # [B,4,T,128]
  attn = causal_gqa_attention(q, k, v, mask)
  attn = attn.transpose(1,2).reshape(B,S,2048)
  x = residual + Linear(2048 -> 2048, bias=False)(attn)

  residual = x
  h = RMSNorm(x, dim=-1)
  m = Linear(11008 -> 2048, bias=False)(
        silu(Linear(2048 -> 11008, bias=False)(h))
        * Linear(2048 -> 11008, bias=False)(h)
      )
  x = residual + m

Final:
  x = RMSNorm(x, dim=-1)
  logits = Linear(2048 -> 128256, bias=False, tied_weight=embed_tokens)(x selected positions)
```

## 6. Attention requirements

SmolLM3 requires causal self-attention for the primary CausalLM target. There is no cross-attention and no encoder branch.

| Attribute | Requirement |
| --- | --- |
| Pattern | Causal self-attention. |
| Head mode | GQA: 16 query heads, 4 KV heads, repeat factor 4. |
| Head dim | 128 for inspected configs. |
| Q/K/V widths | Q 2048, K 512, V 512, output 2048. |
| Prefill shapes | Q `[B,16,S,128]`, K/V `[B,4,S,128]`, attention scores logically `[B,16,S,S]`. |
| Decode shapes | Q `[B,16,S_new,128]`, cached K/V `[B,4,T,128]`, scores `[B,16,S_new,T]`. |
| Mask | Full causal mask by default; optional local causal mask if sliding config is enabled. |
| RoPE | Applied to Q/K before cache update on RoPE-enabled layers, so cached K is already position-encoded. |
| Softmax math | Eager path upcasts softmax to fp32 then casts to query dtype. |
| Packed/varlen | Native source does not require cu-seqlens-style inputs; backend FlashAttention may internally optimize. |
| Sliding-window | Not required for inspected official configs; source passes `sliding_window` to attention backend when active. |
| Cache ABI | Store per-layer K and V before KV-head repeat, shape `[B,4,T,128]` each. |

For first integration, implement dense full-causal GQA prefill and decode. Reject `use_sliding_window=true` with non-null `sliding_window` until a local causal attention path and parity tests exist.

## 7. Position encoding and custom math

RoPE inv frequencies are computed from standardized `config.rope_parameters`:

```python
dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2).float() / dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
emb = cat([freqs, freqs], dim=-1).transpose(1, 2)
cos = cos(emb).to(dtype)
sin = sin(emb).to(dtype)
```

Q/K rotary application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def smollm3_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunity: for static maximum context, precompute `cos/sin [max_pos,128]` per dtype/device for default RoPE. Dynamic RoPE update exists in the shared utility for advanced RoPE types, but inspected configs use default RoPE with no scaling.

NoPE layers must skip rotary entirely. They are not all layers; official pattern skips layers 3, 7, 11, 15, 19, 23, 27, 31, and 35.

## 8. Preprocessing and input packing

The model-coupled runtime inputs are:

- `input_ids [B,S]` int64/int32-compatible token indices.
- Optional `attention_mask [B,S]` from tokenizer or caller.
- Optional `position_ids [B,S]`; if omitted, source builds `arange(S) + past_seen_tokens` and unsqueezes to batch 1.
- Optional `inputs_embeds [B,S,2048]`; first DinoML integration can reject this and require `input_ids`.

Tokenizer notes from `HuggingFaceTB/SmolLM3-3B/tokenizer_config.json`:

- tokenizer class: `PreTrainedTokenizerFast`.
- model input names: `input_ids`, `attention_mask`.
- `model_max_length=131072`, which exceeds neural `max_position_embeddings=65536`; DinoML should guard neural max position separately.
- main instruct special IDs include BOS `<|begin_of_text|>` 128000, EOS `<|im_end|>` 128012, and pad token configured as `<|im_end|>` in tokenizer config while model config uses pad id 128004. Treat tokenizer/model config mismatch as generation ABI, not a graph op.

There is no image/audio/video processor, placeholder token scatter, packed patch metadata, or modality embedding stitch.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split Q/K/V projection pack

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
single GEMM x @ packed_qkv_weight.T -> split [2048,512,512]
```

Preconditions:

- All three projections are bias-free or all biases are handled in the packed epilogue.
- Same input tensor, same dtype, same leading dimensions.
- Packed row order is exactly all Q rows, then all K rows, then all V rows.
- Output split widths are `[num_attention_heads*head_dim, num_key_value_heads*head_dim, num_key_value_heads*head_dim]`.

Shape equations:

- input `[M,2048]`, packed weight `[3072,2048]`, output `[M,3072]`, split to Q `[M,2048]`, K `[M,512]`, V `[M,512]`.

Failure cases:

- Future configs with projection bias, nonstandard head_dim, quantized separate storage, or tensor-parallel sharding not normalized first.

Parity test sketch:

- Compare packed projection splits to independent PyTorch modules for random fp32 and bf16 tensors.

### Rewrite: RMSNorm fusion

Source pattern:

```text
to_fp32 -> square -> mean(dim=-1) -> add eps -> rsqrt -> multiply -> cast -> scale
```

Replacement:

```text
RMSNorm(hidden=2048, eps=1e-6, fp32_accum=True)
```

Preconditions:

- Reduction axis is the last dimension.
- Weight shape is `[2048]`.
- No bias.

Failure cases:

- Layout pass changes the last logical hidden axis. Protect with a no-layout-translation guard unless the norm axis is explicitly rewritten.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
dual GEMM 2048->11008 + fused silu_mul + GEMM 11008->2048
```

Preconditions:

- `hidden_act == "silu"`.
- Gate/up projections share input and have no bias for inspected configs.
- Gate/up split order must be tracked if packed.

Failure cases:

- Other activation functions or MLP bias.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather selected sequence rows -> GEMM(2048 -> vocab)
```

Preconditions:

- `logits_to_keep` is 1 or a bounded small integer/tensor of selected positions.
- Loss is not being computed.

Failure cases:

- Training labels or full-sequence logits requested for parity tests.

### Layout rewrite notes

No NHWC/NCHW rewrite applies. Guarded sequence-layout optimizations may remove transpose/contiguous pairs around attention only if all consumers agree on one internal attention layout:

- Source semantic hidden layout is `[B,S,C]`.
- Attention internal layout is `[B,H,S,D]`.
- Axis-sensitive ops requiring guards: RMSNorm `dim=-1`, softmax `dim=-1` over key positions, attention mask broadcast axes, RoPE cos/sin unsqueeze at dim 1, logits slice on sequence axis 1.
- Do not globally translate `[B,S,C]` to another layout without rewriting `view(...,-1,head_dim)` and transpose axes.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 accumulation, because it appears twice per block plus final norm.
- GQA FlashAttention/SDPA equivalent for prefill and decode with KV cache stored as `[B,4,T,128]`.
- QKV packed projection plus RoPE for RoPE-enabled layers.
- SwiGLU activation multiply between gate/up GEMMs.
- Last-token-only LM head GEMM to avoid full `[B,S,128256]` logits in decode.

Medium priority:

- Decode cache update plus attention kernel launch with RoPE already applied to new K.
- Residual add fused with output projection epilogue where DinoML epilogue coverage supports it.
- Embedding lookup plus optional position-id construction outside the hot loop.

Lower priority:

- Sliding-window local attention, because inspected official configs disable it.
- Classification/QA heads.
- Export-specific ONNX/Transformers.js quantized metadata.

## 11. Runtime staging plan

Stage 1: parse `SmolLM3Config`, normalize legacy `rope_theta`/`rope_scaling` into effective `rope_parameters`, and load tied embedding/LM weights.

Stage 2: one-block fp32 parity without cache using eager dense attention, including RoPE-enabled and NoPE layer cases.

Stage 3: full prefill logits parity for short prompts, initially full-sequence logits and then last-token logits.

Stage 4: dynamic KV cache decode parity with cache tensors `[B,4,T,128]` and cached K stored after RoPE for RoPE layers.

Stage 5: optimized GQA attention backend and packed QKV projection.

Stage 6: bf16/fp16 loading and numerical tolerances against Transformers.

Stage 7: optional local/sliding attention admission if a real config requires it.

Stub/defer initially: training loss, dropout, gradient checkpointing, classification heads, beam search cache reorder, ONNX/GGUF loader semantics, and export-specific quantization.

## 12. Parity and validation plan

- Config normalization test: source defaults versus `SmolLM3-3B` config, including effective `no_rope_layers`, `layer_types`, `rope_theta=5000000.0`, and tied embeddings.
- RoPE unit test: compare cos/sin and Q/K rotation for random `[B,16,S,128]` and `[B,4,S,128]`, including a NoPE layer where tensors must remain unchanged.
- RMSNorm random tensor test with fp32 accumulation and bf16/fp16 output.
- Single decoder layer parity for layer 0 and layer 3 to cover RoPE and NoPE.
- Prefill logits parity on a short prompt. Transformers test uses prompt IDs `[1,306,4658,278,6593,310,2834,338]` and checks logits slices for the 3B checkpoint.
- Decode parity: prefill N tokens, decode one token with cache, compare to full-prefix recomputation.
- Mask parity: left/right padding attention-mask cases if batching padded prompts is admitted.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for unit ops; bf16/fp16 model logits `rtol=1e-2, atol=1e-2` initially, tighten per backend.

DinoML tests were not run for this audit, per user instruction.

## 13. Performance probes

- Prefill throughput sweep over `B={1,4,8}` and `S={128,1024,4096,8192,65536 guard}`.
- Decode tokens/sec sweep for `B={1,8,32}` with cache lengths `{128,1024,4096,16384,65536}`.
- KV cache memory probe: `36 layers * 2 tensors * B * 4 heads * T * 128 * dtype_size`.
- Attention backend comparison: eager reference, SDPA-like, FlashAttention-like GQA.
- QKV packed versus separate GEMMs.
- RMSNorm and SwiGLU fused versus unfused.
- LM head full logits versus last-token-only logits.
- Weight-loading probe for tied embedding/LM head alias preservation and optional GGUF conversion as a separate provider path.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Dropout behavior in training mode.
- Sequence classification, token classification, and question answering heads.
- Beam search and cache reorder, after greedy decode works.
- Sliding-window/local attention configs, because inspected official configs disable them.
- ONNX/Transformers.js q4 export metadata.
- GGUF quantized loader and runtime dequantization for this family, unless explicitly selected as a separate GGUF integration target.
- Multi-GPU tensor parallel and pipeline plans.
- Advanced RoPE types beyond default RoPE.
- Speculative/assisted generation controller paths.

## 15. Final implementation checklist

- [ ] Parse SmolLM3 config and normalize effective RoPE parameters.
- [ ] Reject unsupported sliding-window configs or implement local causal masks.
- [ ] Load tied `model.embed_tokens.weight` / `lm_head.weight` as one logical parameter.
- [ ] Implement embedding lookup for `[B,S] -> [B,S,2048]`.
- [ ] Implement RMSNorm with last-dim fp32 accumulation.
- [ ] Implement bias-free Q/K/V/O linear projections with shapes `2048->2048`, `2048->512`, `2048->512`, `2048->2048`.
- [ ] Implement default RoPE and per-layer NoPE guard.
- [ ] Implement GQA causal attention with KV repeat factor 4.
- [ ] Implement KV cache ABI `[B,4,T,128]` per layer for K and V.
- [ ] Implement SwiGLU MLP `2048->11008->2048`.
- [ ] Implement final RMSNorm and tied LM head `2048->128256`.
- [ ] Add QKV packing rewrite with explicit split order.
- [ ] Add last-token-only logits rewrite.
- [ ] Add one-block RoPE and NoPE parity tests.
- [ ] Add prefill logits and one-token decode parity tests.
- [ ] Benchmark prefill, decode, KV memory, QKV packing, RMSNorm, and LM-head slicing.
