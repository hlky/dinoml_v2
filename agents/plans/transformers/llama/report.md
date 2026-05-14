# Transformers Llama Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary worked example: NousResearch/Llama-2-7b-hf.
  Additional sizing references: NousResearch/Llama-2-13b-hf,
  NousResearch/Meta-Llama-3-8B, TinyLlama/TinyLlama-1.1B-Chat-v1.0,
  codellama/CodeLlama-7b-hf.

Config source:
  https://huggingface.co/NousResearch/Llama-2-7b-hf/raw/main/config.json
  https://huggingface.co/NousResearch/Llama-2-7b-hf/raw/main/tokenizer_config.json
  Additional configs fetched from the Hugging Face model repos listed above.
  HF plugin metadata confirmed the sampled repos as transformers llama
  AutoModelForCausalLM text-generation models.

Source files inspected:
  X:/H/transformers/src/transformers/models/llama/modeling_llama.py
  X:/H/transformers/src/transformers/models/llama/configuration_llama.py
  X:/H/transformers/src/transformers/models/llama/tokenization_llama.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py

Any missing files or assumptions:
  No remote-code files are required for standard Llama. Official Meta Llama 3.2
  configs were gated in this environment, so open mirrors/checkpoints were used
  for sizing. The report assumes inference-only CUDA GPU execution and
  prioritizes decoder-only text generation with KV cache.
```

## 2. High-level architecture

Llama is a text-only decoder transformer. Each block is pre-norm causal self-attention followed by a gated SiLU MLP. The attention path uses RoPE on Q/K. Newer Llama-family checkpoints often use grouped-query attention, so KV head count can be smaller than query head count.

```text
BPE/SentencePiece-like tokenization
  -> token embedding
  -> decoder-only prefill/decode stack with RoPE and KV cache
  -> final RMSNorm
  -> lm_head logits, optionally logits_to_keep
  -> sampling
```

## 3. Important config dimensions

The first table is the worked Llama-2 7B example. Symbols: `B=batch`, `T=sequence length`, `H=hidden_size`, `A=num_attention_heads`, `KvH=num_key_value_heads`, `D=head_dim`, `I=intermediate_size`, `V=vocab_size`, `G=A/KvH`.

| Field | NousResearch/Llama-2-7b-hf value | Notes |
|---|---:|---|
| architecture | LlamaForCausalLM | Decoder-only LM |
| vocab_size / V | 32000 | LM head is bias-free and untied by config |
| hidden_size / H | 4096 | Hidden width |
| num_hidden_layers | 32 | Decoder blocks |
| num_attention_heads / A | 32 | Query heads |
| num_key_value_heads / KvH | 32 | MHA for Llama 2 7B |
| head_dim / D | 128 inferred | `H // A` when absent |
| intermediate_size / I | 11008 | SwiGLU/SiLU gated MLP |
| hidden_act | silu | `act(gate_proj(x)) * up_proj(x)` |
| max_position_embeddings | 4096 | Llama 2 context length |
| rope_theta | 10000 inferred/default | Config omits it; current default path uses base theta |
| rms_norm_eps | 1e-5 | Source default is 1e-6, checkpoint uses 1e-5 |
| attention_bias | false inferred | Older configs omit; LlamaConfig default false |
| mlp_bias | false inferred | Older configs omit; LlamaConfig default false |
| tie_word_embeddings | false | Separate LM head weight |
| torch_dtype | float16 | Checkpoint metadata |
| use_cache | true inferred | DynamicCache path |

Representative real checkpoint ranges:

| Checkpoint | H | I | layers | A | KvH | D | G | V | max pos | RoPE | dtype |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2048 | 5632 | 22 | 32 | 4 | 64 | 8 | 32000 | 2048 | theta 10000 | bf16 |
| NousResearch/Llama-2-7b-hf | 4096 | 11008 | 32 | 32 | 32 | 128 | 1 | 32000 | 4096 | default theta | fp16 |
| NousResearch/Llama-2-13b-hf | 5120 | 13824 | 40 | 40 | 40 | 128 | 1 | 32000 | 4096 | default theta | fp16 |
| NousResearch/Meta-Llama-3-8B | 4096 | 14336 | 32 | 32 | 8 | 128 | 4 | 128256 | 8192 | theta 500000 | bf16 |
| codellama/CodeLlama-7b-hf | 4096 | 11008 | 32 | 32 | 32 | 128 | 1 | 32016 | 16384 | theta 1000000 | bf16 |

Important inference: Dinoml must not assume Llama means MHA. Llama 3 and TinyLlama use GQA, so K/V projection output width is `KvH*D`, not `H`, and attention kernels must either support GQA natively or repeat/gather K/V safely.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,T] -> hidden[B,T,H]`.
- Flatten leading dims for Linear/GEMM and restore `[B,T,*]`.
- View projection outputs:
  - Q `[B,T,A*D] -> [B,A,T,D]`.
  - K/V `[B,T,KvH*D] -> [B,KvH,T,D]`.
- Transpose/contiguous around attention output.
- KV repeat/expand fallback: `[B,KvH,T,D] -> [B,A,T,D]` when backend lacks native GQA.
- Causal mask construction and additive mask broadcast to `[B,A,Q,K]`.
- Slice/select logits through `logits_to_keep`; source computes `lm_head(hidden_states[:, slice_indices, :])`.

### Neural network primitives

- Bias-free Linear by default:
  - Llama-2 7B Q/K/V/O: `4096 -> 4096`.
  - Llama-3 8B Q: `4096 -> 4096`, K/V: `4096 -> 1024`, O: `4096 -> 4096`.
  - TinyLlama Q: `2048 -> 2048`, K/V: `2048 -> 256`, O: `2048 -> 2048`.
  - MLP: gate/up `H -> I`, down `I -> H`.
  - LM head: `H -> V`.
- Optional attention and MLP bias exists in config fields, but sampled checkpoints use no bias or omit fields that default false.
- RMSNorm: fp32 accumulation, scale only, no bias.
- SiLU activation and elementwise multiply for gated MLP.
- Residual adds after attention and MLP.
- Softmax in fp32 for eager attention.
- Dropout is present but zero for inference.

### Attention primitives

- Causal self-attention only.
- MHA and GQA.
- RoPE applied to Q and K before cache update.
- DynamicCache append for K/V.
- Scaling by `head_dim**-0.5`.
- Attention backends selected through `ALL_ATTENTION_FUNCTIONS`; eager fallback is repeat_kv + matmul + additive mask + fp32 softmax + matmul.

### Position/rotary ops

- Default RoPE inverse frequencies from `rope_theta` and `head_dim`.
- Position IDs default to `arange(T) + past_seen_tokens`.
- `LlamaRotaryEmbedding` computes cos/sin in fp32 then casts to activation dtype.
- Supported source-level rope types include default, linear, dynamic NTK, yarn, longrope, and llama3 through `modeling_rope_utils.py`.
- Llama 3.1-style RoPE has low/high frequency smoothing parameters in current Transformers, even though the open Llama-3 8B mirror sampled here uses base `rope_theta=500000` without a `rope_scaling` payload.

### Generation/cache ops

- DynamicCache allocation when `use_cache=True` and no cache is provided.
- Per-layer K/V append with shape `[B,KvH,past,D]`.
- Cache length query for default position IDs.
- `logits_to_keep` support to avoid full prefill logits.

### Preprocessing-coupled ops

- LlamaTokenizer uses byte-level BPE with byte fallback, metaspace pre-tokenization, no normalization, and left padding.
- Tokenizer defaults: unk `<unk>` id 0, BOS `<s>` id 1, EOS `</s>` id 2 for Llama 1/2-style tokenizers.
- Llama 3-style tokenizers use different special strings and larger vocab; chat templates are data-pipeline work.

### Distributed/tensor-parallel ops

- LlamaConfig declares a default tensor-parallel plan:
  - Q/K/V, gate/up projections: colwise.
  - O/down projections: rowwise.
  - LM head: colwise gather output.
- Single-GPU first parity can defer TP, but large production checkpoints will need sharded GEMM and sharded KV-cache planning.

## 5. Layer/block breakdown

Decoder block, repeated `N` times:

```text
residual = x                                           # [B,T,H]
y = RMSNorm(x)
q = Linear(H -> A*D, bias=attention_bias)(y)           # [B,A,T,D]
k = Linear(H -> KvH*D, bias=attention_bias)(y)         # [B,KvH,T,D]
v = Linear(H -> KvH*D, bias=attention_bias)(y)         # [B,KvH,T,D]
cos,sin = rotary(position_ids)                         # [B,T,D]
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(layer,k,v) if cache enabled
attn = CausalAttention(q,k,v,mask,scale=D**-0.5,GQA)
x = residual + Linear(A*D -> H, bias=attention_bias)(attn)

residual = x
y = RMSNorm(x)
gate = SiLU(Linear(H -> I, bias=mlp_bias)(y))
up = Linear(H -> I, bias=mlp_bias)(y)
x = residual + Linear(I -> H, bias=mlp_bias)(gate * up)
```

Model head:

```text
x = embed_tokens(input_ids)
position_ids = arange(T) + cache_length
position_embeddings = rotary_emb(x, position_ids)
x = decoder_layers(x, causal_mask, position_embeddings, cache)
x = RMSNorm(x)
logits = Linear(H -> V, bias=False)(x[:, slice_indices, :])
```

## 6. Attention requirements

- Causal decoder self-attention only; no cross-attention.
- MHA when `A == KvH`; GQA when `A > KvH`.
- Head dim is `config.head_dim` if provided, else `H // A`.
- Q shape `[B,A,Q,D]`; K/V shape `[B,KvH,K,D]`.
- Native GQA backend should avoid materializing repeated K/V. Eager fallback repeats by expansion/reshape to `[B,A,K,D]`.
- Masking style is additive causal mask plus caller attention mask, passed to attention backend.
- RoPE is applied before cache update, so cached K is already position-rotated.
- KV cache stores only `KvH` heads, not repeated query heads.
- FlashAttention compatibility: causal FlashAttention with RoPE already applied is suitable if it supports MHA/GQA and additive masks. For decode, a paged KV-cache backend is the useful production target.
- Eager fallback is likely too slow because it materializes repeated K/V for GQA and uses separate matmul/softmax/matmul.

## 7. Position encoding and custom math

Default RoPE:

```python
def llama_default_inv_freq(head_dim, rope_theta, device):
    i = arange(0, head_dim, 2, dtype=float32, device=device)
    return 1.0 / (rope_theta ** (i / head_dim))

def llama_rotary_embedding(position_ids, inv_freq, dtype):
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], axis=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat([-x2, x1], axis=-1)

def apply_llama_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Llama 3.1 RoPE scaling in current Transformers smooths inverse frequencies by wavelength. Dinoml can initially support default RoPE and add extended variants behind explicit config guards.

What can be precomputed:

- `inv_freq` is static for default RoPE.
- Cos/sin can be precomputed for static max position and dtype if memory allows.
- Decode cos/sin for one new position can be computed cheaply from cache length.

Dynamic inputs:

- `position_ids` depends on `past_seen_tokens` unless supplied.
- Dynamic/longrope variants may update inverse frequencies based on maximum position.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- Tokenization and chat template application.
- BOS/EOS insertion policy.
- Left padding and attention mask construction for batched prompts.

GPU/runtime graph work:

- Accept `input_ids[B,T]`, optional `attention_mask[B,T]`, optional `position_ids[B,T]`, optional cache.
- For prefill, process all prompt tokens and optionally keep only last logits.
- For decode, process `T=1` new token with cache length-derived position.
- No image/audio/video packing, modality IDs, or placeholder-token replacement in standard Llama.

## 9. Graph rewrite / lowering opportunities

### Rewrite: LlamaRMSNorm -> RMSNormScaleOnly

Preconditions:

- Exact form: cast input to fp32, compute mean of squares on last dim, multiply by `rsqrt(var + eps)`, cast normalized tensor to input dtype, multiply by weight.
- Weight shape is `[H]`; no bias.

Replacement:

```text
RMSNorm(x, weight, eps, axis=-1, fp32_accum=True)
```

Failure cases:

- Do not rewrite generic LayerNorm.
- Preserve source cast order for reduced precision parity.

Parity test sketch:

- Compare random tensors for `H={2048,4096,5120}` and eps `1e-5`.

### Rewrite: bias-free Linear -> GEMM_RCR

Preconditions:

- `attention_bias == false` or `mlp_bias == false` for the matched projection.
- Input is dense row-major after flattening leading dims.
- Weight stored `[out_features,in_features]`.

Replacement:

```text
FlattenLeadingDims -> GEMM_RCR(A=[M,K], B=[N,K]) -> Reshape([*,N])
```

Failure cases:

- If a future checkpoint sets bias true, use bias epilogue GEMM.
- Sharded TP weights need shard-aware lowering.

Parity test sketch:

- Compare Q/K/V/O, gate/up/down, and LM head projections independently.

### Rewrite: QKV projection grouping for self-attention

Preconditions:

- Same normalized input tensor feeds q/k/v.
- Bias settings match and are either all absent or all present.
- Output widths can differ: Q width `A*D`, K/V width `KvH*D`.

Replacement:

```text
GroupedGEMM([q_proj,k_proj,v_proj]) or ConcatenatedLinear(H -> (A+2*KvH)*D) -> Split
```

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
```

Failure cases:

- Do not assume equal Q/K/V output widths under GQA.
- Quantized weights with incompatible block layouts may need grouped rather than concatenated GEMM.

Parity test sketch:

- Test MHA and GQA configs and compare split tensors before RoPE.

### Rewrite: eager repeat_kv attention -> native GQA attention

Preconditions:

- `A % KvH == 0`.
- Attention semantics are equivalent to repeating each KV head `G=A/KvH` times.
- Backend consumes Q heads and KV heads separately.

Replacement:

```text
GQAAttention(q[B,A,Q,D], k[B,KvH,K,D], v[B,KvH,K,D], group_size=G)
```

Failure cases:

- Backends requiring materialized `[B,A,K,D]` cannot claim memory savings.
- Incorrect KV head grouping order breaks parity.

Parity test sketch:

- Compare native GQA output against HF eager `repeat_kv` for `G={1,4,8}`.

### Rewrite: RoPE cos/sin generation -> cached RoPE table

Preconditions:

- Default RoPE or supported fixed scaling variant.
- `rope_theta`, `head_dim`, dtype, and maximum position are fixed.

Replacement:

```text
StaticCosSinTable[position_ids] -> ApplyRoPE
```

Failure cases:

- Dynamic NTK or longrope variants may require runtime inv_freq updates.
- Llama 3.1 smoothing must be represented exactly if enabled.

Parity test sketch:

- Compare cos/sin and rotated Q/K against HF for sampled positions including long-context positions.

### Rewrite: logits_to_keep -> last-token LM head

Preconditions:

- Generation needs only the final token logits or explicit selected positions.
- No loss computation requiring all logits.

Replacement:

```text
SliceHidden([B,T,H] -> [B,K,H]) -> LMHeadGEMM(H -> V)
```

Failure cases:

- Training/loss and perplexity evaluation require full logits.

Parity test sketch:

- Compare `logits_to_keep=1` and explicit tensor indices against full-logits slicing.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: two per block plus final norm; bandwidth-sensitive and easy to validate.
- Native RoPE + Q/K layout transform: required every layer and decode step.
- GQA FlashAttention/paged attention with KV cache: essential for Llama 3/TinyLlama and production decode.
- SwiGLU/SiLU MLP fusion: `SiLU(gate) * up` is a large activation-memory path between GEMMs.
- Last-token-only logits: reduces LM head work during decode and matches source `logits_to_keep`.

Medium priority:

- QKV grouped projection with unequal GQA output widths.
- Fused residual add with RMSNorm input staging where numerically acceptable.
- KV cache append/pack kernels for `[B,KvH,T,D]`.
- Sharded LM head and top-k/top-p sampling kernels.

Lower priority:

- Bias epilogues for rare biased Llama configs.
- Extended RoPE variants beyond default/base theta.
- Sequence classification, QA, and token classification heads.

## 11. Runtime staging plan

Stage 1: Parse LlamaConfig including old `rope_theta`/`rope_scaling` and current `rope_parameters` forms.

Stage 2: Load embedding, decoder layers, RMSNorm weights, LM head, and untied/tied weight metadata.

Stage 3: Implement RMSNorm and default RoPE parity.

Stage 4: Run one decoder layer parity without cache for an MHA config.

Stage 5: Add GQA parity using TinyLlama or Llama-3 8B-style shapes.

Stage 6: Run prefill logits parity with `logits_to_keep=1`.

Stage 7: Add cached decode with K/V append and RoPE position offset.

Stage 8: Enable optimized attention/paged KV cache.

Stage 9: Add extended RoPE variants only when a target checkpoint requires them.

## 12. Parity and validation plan

- RMSNorm random tensor parity for fp32/fp16/bf16.
- Default RoPE cos/sin and `apply_rotary_pos_emb` parity for multiple positions and dtypes.
- GQA `repeat_kv` equivalence tests for `G=1,4,8`.
- Single attention parity for MHA and GQA.
- Single decoder layer parity for Llama-2 7B-like and Llama-3 8B-like dimensions.
- Full small-model prefill parity, preferably TinyLlama for accessible weights.
- Cached decode parity for 2-4 steps, checking logits and cache shapes.
- `logits_to_keep` parity for last-token and explicit indices.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 `rtol=2e-2, atol=2e-2` initially, then tighten per backend.

## 13. Performance probes

- Prefill throughput over batch and sequence length.
- Decode tokens/sec over batch size and cache length.
- MHA vs GQA attention backend comparison.
- KV cache memory usage: `2 * layers * B * KvH * T * D * dtype_bytes`.
- RoPE generation vs precomputed table overhead.
- MLP GEMM/activation bandwidth split.
- LM head full logits vs `logits_to_keep=1`.
- Long-context sweep for CodeLlama-style 16K context.
- Batch padding/left-padding overhead.

## 14. Skip/defer list

- Training and loss.
- Dropout and gradient checkpointing.
- Sequence classification, QA, token classification heads.
- Beam search and speculative decoding.
- Multi-GPU TP/PP for first parity.
- Quantization-specific formats.
- Dynamic NTK, YaRN, LongRoPE, and Llama 3.1 RoPE scaling until a selected checkpoint requires them.
- Chat template rendering; keep it in preprocessing.

## 15. Final implementation checklist

- [ ] Parse LlamaConfig with head_dim and num_key_value_heads defaults.
- [ ] Normalize old RoPE config fields into explicit rope parameters.
- [ ] Load embedding, layers, norms, and LM head.
- [ ] Implement RMSNormScaleOnly.
- [ ] Implement default RoPE cos/sin and apply_rope.
- [ ] Implement bias-free Linear/GEMM lowering for Q/K/V/O, MLP, and LM head.
- [ ] Implement SwiGLU/SiLU MLP.
- [ ] Implement causal MHA attention.
- [ ] Implement native GQA attention or safe repeat_kv fallback.
- [ ] Implement DynamicCache K/V append.
- [ ] Implement `logits_to_keep` lowering.
- [ ] Add one-layer MHA and GQA parity tests.
- [ ] Add prefill logits parity.
- [ ] Add cached decode parity.
- [ ] Benchmark prefill, decode, RoPE, MLP, LM head, and KV memory.
