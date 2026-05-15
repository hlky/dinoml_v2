# CodeLlama Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: codellama/CodeLlama-* HF checkpoints, implemented by model_type="llama"
Config source: downloaded config/tokenizer/generation JSON snapshots in _sources/
Source files inspected:
- transformers/src/transformers/models/code_llama/tokenization_code_llama.py
- transformers/src/transformers/models/code_llama/__init__.py
- transformers/src/transformers/models/llama/configuration_llama.py
- transformers/src/transformers/models/llama/modeling_llama.py
- transformers/src/transformers/modeling_rope_utils.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/cache_utils.py
Any missing files or assumptions:
- There is no CodeLlama modeling or configuration file in this Transformers commit. CodeLlama model configs use model_type="llama" and architectures=["LlamaForCausalLM"], so runtime coverage is shared with Llama.
- meta-llama CodeLlama repos returned 401 without credentials: [7B](https://huggingface.co/meta-llama/CodeLlama-7b-hf), [7B Python](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf), [13B Instruct](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf), [34B](https://huggingface.co/meta-llama/CodeLlama-34b-hf), [70B](https://huggingface.co/meta-llama/CodeLlama-70b-hf). Accepted gated access plus an HF token would resolve this.
- Accessible snapshots came from the open [codellama](https://huggingface.co/codellama) namespace. Treat these as representative configs, not as proof that every gated meta-llama repo is byte-identical.
```

Representative snapshots saved:

- `_sources/codellama__CodeLlama-7b-hf.config.json`
- `_sources/codellama__CodeLlama-7b-Python-hf.config.json`
- `_sources/codellama__CodeLlama-13b-Instruct-hf.config.json`
- `_sources/codellama__CodeLlama-34b-hf.config.json`
- `_sources/codellama__CodeLlama-70b-hf.config.json`
- `_sources/codellama__CodeLlama-70b-Instruct-hf.config.json`
- `_sources/hf-internal-testing__tiny-random-LlamaForCausalLM.config.json`
- Matching tokenizer/generation/special-token JSON for the codellama checkpoints.

## 2. High-level architecture

Primary runtime target: causal LM prefill and autoregressive decode.

Dataflow:

```text
CPU tokenizer/infill prompt packing -> input_ids/attention_mask
-> token embedding -> repeated Llama decoder blocks
-> final RMSNorm -> tied/aliasable LM projection -> logits/sampling
```

The CodeLlama-specific directory owns tokenizer behavior only. The neural model is a text-only decoder: embedding table, causal self-attention with RoPE, gated SwiGLU MLP, residuals, final RMSNorm, and LM head. No encoder, cross-attention, vision/audio branch, MoE, state-space layer, or sliding-window layer is implemented by this source basis.

Stage decomposition:

- CPU/data pipeline: BPE tokenizer, optional CodeLlama infill split on `<FILL_ME>`, prompt template construction for instruct use.
- Prefill: full causal self-attention over prompt tokens; cache writes one K/V tensor pair per layer.
- Decode: one or more new tokens, position IDs offset by cache length, cached K/V reused.
- Logits: `logits_to_keep` can restrict output projection to last tokens, useful for decode and last-token-only prefill.

Other heads in shared Llama source: base `LlamaModel` is required as the backbone; `LlamaForCausalLM` is required for this target; sequence classification, token classification, and QA heads are optional/deferred.

## 3. Important config dimensions

Source defaults from `LlamaConfig` if omitted by a checkpoint:

| Field | Effective default |
| --- | --- |
| `model_type` | `llama` |
| `hidden_act` | `silu` |
| `num_key_value_heads` | `num_attention_heads` |
| `head_dim` | `hidden_size // num_attention_heads` |
| `attention_bias` | `False` |
| `mlp_bias` | `False` |
| `use_cache` | `True` |
| `tie_word_embeddings` | `False` |
| `rope_parameters` | standardized to `{"rope_type": "default", "rope_theta": rope_theta}` |

Representative checkpoint sweep:

| Checkpoint | h | Layers | Q heads | KV heads | Head dim | MLP | Vocab | Max pos | RoPE theta | Attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hf-internal-testing/tiny-random-LlamaForCausalLM | 16 | 2 | 4 | 4 | 4 | 64 | 32000 | 2048 | 10000 | MHA debug |
| codellama/CodeLlama-7b-hf | 4096 | 32 | 32 | 32 | 128 | 11008 | 32016 | 16384 | 1000000 | MHA |
| codellama/CodeLlama-7b-Python-hf | 4096 | 32 | 32 | 32 | 128 | 11008 | 32000 | 16384 | 1000000 | MHA |
| codellama/CodeLlama-13b-Instruct-hf | 5120 | 40 | 40 | 40 | 128 | 13824 | 32016 | 16384 | 1000000 | MHA |
| codellama/CodeLlama-34b-hf | 8192 | 48 | 64 | 8 | 128 | 22016 | 32000 | 16384 | 1000000 | GQA, repeat 8 |
| codellama/CodeLlama-70b-hf | 8192 | 80 | 64 | 8 | 128 | 28672 | 32016 | 16384 | 1000000 | GQA, repeat 8 |
| codellama/CodeLlama-70b-Instruct-hf | 8192 | 80 | 64 | 8 | 128 | 28672 | 32016 | 4096 | 10000 | GQA, repeat 8 |

Config-derived dtype is `bfloat16` for the accessible CodeLlama configs and `float32` for the tiny random config. The current source does not read CodeLlama-specific long-context flags beyond normal Llama `rope_theta`/standardized `rope_parameters`.

## 3a. Family variation traps

- CodeLlama is not a separate model class here. Route neural graph lowering to Llama; use CodeLlama tokenizer only when tokenizer parity matters.
- 34B and 70B use GQA: `num_key_value_heads=8` while `num_attention_heads=64`. DinoML must store cache as KV heads, not expanded Q heads.
- 7B/13B are full MHA: `num_key_value_heads == num_attention_heads`.
- 70B Instruct accessible config differs from the other long-context configs: `max_position_embeddings=4096`, `rope_theta=10000`.
- `vocab_size` varies between 32000 and 32016. Do not hard-code CodeLlama vocab size.
- Checkpoints commonly omit `attention_bias`, `mlp_bias`, `head_dim`, and `rope_parameters`; source defaults make attention/MLP projections bias-free and head_dim 128 for real CodeLlama sizes.
- `pretraining_tp` appears in configs but the inspected modeling source does not branch on it.
- Infill is tokenizer-side prompt packing, not a separate decoder architecture. The GPU graph still consumes normal `input_ids`, `attention_mask`, optional `position_ids`, and optional cache.
- No NCHW/NHWC layout issue exists for the text decoder. Layout-sensitive axes are sequence/head axes in reshapes/transposes and last-dim normalization/reductions.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `[B, S] -> [B, S, hidden_size]`.
- View/reshape/transposes: Q/K/V `[B, S, proj] -> [B, heads, S, head_dim]`; attention output transpose `[B, heads, S, head_dim] -> [B, S, hidden]`.
- Contiguous/materialized copies after transpose where required by lowering.
- Slice/index for `logits_to_keep`, usually last-token slice.
- Optional cache batch index select/reorder for beam search later.

Neural network primitives:

- RMSNorm over last dim with fp32 variance accumulation and weight multiply.
- Bias-free linear projections by default:
  - 7B: Q 4096->4096, K/V 4096->4096, O 4096->4096, gate/up 4096->11008, down 11008->4096.
  - 13B: Q/K/V 5120->5120, O 5120->5120, gate/up 5120->13824, down 13824->5120.
  - 34B: Q 8192->8192, K/V 8192->1024, O 8192->8192, gate/up 8192->22016, down 22016->8192.
  - 70B: Q 8192->8192, K/V 8192->1024, O 8192->8192, gate/up 8192->28672, down 28672->8192.
- SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
- Residual adds.
- LM head `hidden_size -> vocab_size`, bias-free. `LlamaForCausalLM` declares tied weight keys for `lm_head.weight` and `model.embed_tokens.weight`; source config default says `tie_word_embeddings=False`, so loading must preserve actual checkpoint aliasing if present rather than infer tying from architecture alone.

Attention primitives:

- Causal self-attention.
- MHA or GQA depending on `num_key_value_heads`.
- RoPE applied to Q and K before cache update.
- Cache update stores post-RoPE K and raw V.
- Eager fallback: QK matmul, mask add, fp32 softmax, dropout only in training, AV matmul.
- Optimized source path dispatches through `ALL_ATTENTION_FUNCTIONS` for eager/SDPA/Flash/Flex depending on `_attn_implementation`.

Position/rotary ops:

- Default RoPE with `rope_theta` from config, head_dim-derived full rotary dimension.
- Current source can support standard HF RoPE variants through `rope_parameters`, but accessible CodeLlama configs use default RoPE only.

Generation/cache ops:

- Dynamic cache or static cache backend ABI.
- Per-layer key/value tensors shaped `[B, num_key_value_heads, S_cache, head_dim]`.
- GQA repeat/broadcast to Q heads is a compute view before attention, not cache storage.
- Position IDs default to `arange(query_len) + past_seen_tokens`.

Preprocessing-coupled ops:

- CodeLlama BPE tokenizer with metaspace pre-tokenizer, byte fallback decoder, left padding.
- Optional infill processor inserts prefix/suffix/middle tokens around two text segments.

Not required:

- MoE routing, cross-attention, sliding-window/local attention, packed varlen metadata beyond normal HF attention-mask handling, quantized/packed weight metadata, multimodal scatter, recurrent/state-space cache.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(hidden -> num_attention_heads * head_dim, bias=attention_bias)(x_norm)
k = Linear(hidden -> num_key_value_heads * head_dim, bias=attention_bias)(x_norm)
v = Linear(hidden -> num_key_value_heads * head_dim, bias=attention_bias)(x_norm)
q,k,v = view to [B, heads_or_kv_heads, S, head_dim]
q,k = RoPE(q,k, cos(position_ids), sin(position_ids))
k,v = cache.update(k,v, layer_idx) if cache is present
attn = causal_attention(q,k,v, mask, scaling=head_dim**-0.5)
x = residual + Linear(num_attention_heads * head_dim -> hidden, bias=attention_bias)(attn)
residual = x
x = RMSNorm(x)
x = residual + down_proj(silu(gate_proj(x)) * up_proj(x))
```

Model wrapper:

```text
input_ids -> Embedding(vocab, hidden)
shared RoPE cos/sin computed once per forward from hidden_states dtype/device and position_ids
decoder blocks
final RMSNorm
LM head on hidden_states[:, slice_indices, :]
```

Default projection biases are absent. If a future checkpoint sets `attention_bias` or `mlp_bias`, DinoML should either admit and lower bias variants or reject clearly.

## 6. Attention requirements

Required variant: causal decoder self-attention.

- MHA for 7B/13B, GQA for 34B/70B.
- Head dim: 128 for representative CodeLlama production configs.
- Q heads: 32, 40, or 64. KV heads: 32, 40, or 8.
- Scaling: multiply QK scores by `head_dim ** -0.5`.
- Masking: `create_causal_mask` combines causal mask, optional 2D padding mask, cache offsets, and backend-specific representation. For pure SDPA paths, the mask helper may skip materializing a mask when it can use backend `is_causal`.
- Eager math order: repeat KV to Q head count, `matmul(Q, K^T) * scaling`, add mask, softmax in fp32, cast to query dtype, dropout if training, `matmul(weights, V)`.
- Cache layout: store each layer as K and V tensors `[B, num_key_value_heads, S_cache, head_dim]`; DynamicLayer appends along `dim=-2`.
- Cached K is post-RoPE because RoPE is applied before `past_key_values.update`.
- Decode position IDs are offset by `past_key_values.get_seq_length()`.

No sliding window is configured by representative CodeLlama configs. No cross-attention or encoder-decoder cache is present.

FlashAttention/SDPA compatibility:

- Source declares `_supports_flash_attn`, `_supports_sdpa`, `_supports_flex_attn`, and `_supports_attention_backend`.
- DinoML first parity can use explicit dense causal/GQA attention; optimized path should target fused prefill and decode kernels that consume unexpanded KV heads.

## 7. Position encoding and custom math

Default RoPE source behavior:

```python
def code_llama_default_rope(position_ids, head_dim, rope_theta):
    inv = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
    freqs = matmul(inv[None, :, None], position_ids[:, None, :]).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

The current Llama source computes RoPE in fp32 under autocast-disabled scope, then casts cos/sin to the hidden-state dtype. Cos/sin depend on runtime `position_ids`, device, dtype, head_dim, and `rope_theta`; they can be precomputed per admitted context length for default RoPE, or generated on GPU for dynamic cache offsets.

Advanced RoPE variants (`linear`, `dynamic`, `yarn`, `longrope`, `llama3`, `proportional`) are available in shared rope utilities, but the representative CodeLlama configs inspected do not require them. DinoML should not silently accept non-default `rope_parameters` for CodeLlama until each variant is separately validated.

## 8. Preprocessing and input packing

CPU/data pipeline:

- Tokenizer class for most accessible CodeLlama configs: `CodeLlamaTokenizer`; 70B configs report `LlamaTokenizer`.
- `CodeLlamaTokenizer` is BPE with byte fallback, metaspace replacement `"_"`-like U+2581 in source, and no text normalization beyond the configured metaspace/infilling processor.
- Model input names: `input_ids`, `attention_mask`; padding side is left.
- Defaults: BOS `<s>` added, EOS not added, no pad token.
- Infill path:
  - If `text` contains `fill_token` (`<FILL_ME>`) and no explicit suffix, split into prefix/suffix.
  - Normal infill order: `<BOS>? <PRE> prefix <SUF> suffix <MID>`.
  - `suffix_first=True` order: `<BOS>? <PRE> <SUF> suffix <MID> prefix`.
  - This only changes token IDs and special-token ordering; the model graph remains standard causal LM.

GPU/runtime inputs:

- `input_ids` or `inputs_embeds`, exactly one.
- Optional 2D `attention_mask` matching padded tokens and current/cache length.
- Optional `position_ids`; if omitted, source constructs them from cache length.
- Optional `past_key_values`.

Generation-controller behavior such as sampling, stopping, instruct chat formatting, and fill-token prompt construction can be kept outside the first DinoML compiled graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV projection

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
GEMM(x, concat_rows_or_cols(q,k,v)) -> split [Q, K, V]
```

Preconditions:

- Same input `x`, same dtype, same batch/sequence layout.
- Bias handling matches `attention_bias` exactly. Representative configs use no bias.
- Split sizes are `[num_attention_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim]`.
- Packed weight transform must respect PyTorch Linear storage `[out_features, in_features]`.

Failure cases:

- Non-default bias without fused bias support.
- Tensor parallel weight shards or remote-code packed layouts not matching separate HF weights.

Parity test sketch: compare packed projection splits against individual modules for MHA and GQA configs.

### Rewrite: GQA repeat_kv -> grouped attention kernel

Source pattern:

```text
expand [B, KVH, S, D] -> [B, KVH, groups, S, D] -> reshape [B, QH, S, D]
```

Replacement: fused attention accepts Q heads and unexpanded KV heads with `groups = QH // KVH`.

Preconditions: `num_attention_heads % num_key_value_heads == 0`; cache stores unexpanded KV heads.

Failure cases: attention backend that requires physical repeated K/V for debugging outputs.

### Rewrite: RoPE + cache update + attention prefill/decode

Source pattern:

```text
RoPE(q,k) -> cache append -> causal attention
```

Replacement: fused attention entrypoint that applies RoPE to Q/K and appends post-RoPE K to cache before attention.

Preconditions: default RoPE, known head_dim, contiguous Q/K/V layouts, cache slot addresses stable.

Failure cases: non-default rope variant, user-supplied unusual `position_ids`, attention output/weight debugging.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
silu(gate_proj(x)) * up_proj(x) -> down_proj
```

Replacement: two GEMMs plus fused activation/mul epilogue, then down GEMM.

Preconditions: matching intermediate shape and dtype; no MLP bias or supported bias.

Failure cases: hidden_act not `silu`, quantized/non-dense weights without provider support.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, -k:, :])`.

Replacement: slice hidden states before GEMM, or run GEMM only for selected token rows.

Preconditions: `logits_to_keep` is int or static tensor indices; no loss computation requiring all logits.

Failure cases: training loss, arbitrary dynamic index tensor not supported by lowering.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: every block has two RMSNorms plus final norm; fp32 accumulation parity matters.
- GQA FlashAttention with KV cache: 34B/70B depend on efficient unexpanded KV heads.
- QKV projection + RoPE + attention handoff: removes reshapes/transposes and avoids materializing repeated KV.
- SwiGLU: large intermediate GEMMs plus activation multiply dominate block cost.
- Last-token-only logits: avoids full-sequence vocab GEMM during decode and common prefill sampling.

Medium priority:

- Fused residual add around attention/MLP outputs.
- RoPE table/precompute cache for admitted max contexts.
- Packed QKV projection for MHA and GQA.
- BF16 GEMM/RMSNorm/attention path because configs report `bfloat16`.

Lower priority:

- Sequence/token/QA/classification heads from shared Llama source.
- Beam-search cache reorder.
- Advanced RoPE variants not used by inspected CodeLlama configs.
- Tensor parallel execution plan from source `_tp_plan`.

## 11. Runtime staging plan

Stage 1: shared Llama config and weight loader for CodeLlama configs. Admit default RoPE only, no projection biases, no nonstandard remote code.

Stage 2: one-block parity for MHA 7B shape and GQA 34B/70B shape using random weights.

Stage 3: full prefill parity for tiny random and one real-size shape skeleton with random weights; validate logits with `logits_to_keep=1`.

Stage 4: decode with DynamicCache-compatible ABI: per-layer K/V `[B, KVH, max_seq, D]`, append along sequence, position offset handling.

Stage 5: optimized attention path for MHA and GQA, then BF16 production path.

Stage 6: tokenizer/data-pipeline parity for CodeLlama infill prompts outside the compiled graph.

Stage 7: optional generation-controller integration: sampling, stopping, instruct templates, beam reorder.

Can stub initially: tokenizer execution, sampling, beam search, all non-causal-LM heads, remote/gated repo access, advanced RoPE variants.

## 12. Parity and validation plan

- Config parser tests:
  - omitted `attention_bias`, `mlp_bias`, `head_dim`, and `rope_parameters` produce source-equivalent defaults.
  - 34B/70B configs preserve `KVH=8`, not inferred as `QH`.
- Custom op tests:
  - RMSNorm fp32 accumulation against PyTorch for fp32/fp16/bf16.
  - RoPE default theta 10000 and 1000000, multiple position offsets.
  - `repeat_kv` equivalence for GQA groups 8.
  - SwiGLU MLP parity.
- Single-layer parity:
  - random MHA 7B-like block and GQA 34B-like block, fp32 first, then bf16 tolerance.
- Prefill parity:
  - tiny random checkpoint, no cache and with cache, attention mask with left padding.
- Decode parity:
  - prefill N tokens, decode one token, compare logits and cache contents after append.
- End-to-end text parity:
  - CPU tokenizer produces expected infill token sequence; DinoML graph consumes IDs and matches HF logits for tiny/small available weights.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=2e-2, atol=2e-2` for full block logits, tighter for isolated GEMM/RMSNorm where practical.

## 13. Performance probes

- Prefill tokens/sec by batch and sequence length: 512, 2048, 4096, 16384.
- Decode tokens/sec by batch with cache lengths 1k, 4k, 16k.
- MHA 7B/13B vs GQA 34B/70B attention backend comparison.
- KV cache memory: `layers * 2 * B * KVH * max_seq * head_dim * dtype_size`.
- QKV projection packing vs separate GEMMs.
- SwiGLU fused epilogue vs unfused activation/mul.
- Last-token-only LM head vs full-sequence logits.
- BF16 vs FP16 vs FP32 fallback throughput.
- Tokenizer/infill CPU throughput separately from GPU graph.
- Optional GGUF or other quantized weight load/dequant provider comparison, if later weights are ingested in encoded form. This is a loading/provider probe, not a source requirement.

## 14. Skip/defer list

- Training, loss, dropout, gradient checkpointing.
- Sequence classification, token classification, and question answering heads.
- Beam search and cache reorder for first greedy/sampling integration.
- Tensor parallel and pipeline parallel plans.
- Advanced/non-default RoPE variants unless a specific admitted checkpoint uses them.
- Remote-code-only or historical config flags not read by the inspected source.
- Quantization, GGUF, GPTQ/AWQ, or packed-weight loading unless selected by DinoML as a separate weight-ingestion target.
- Chat template/instruct safety behavior beyond producing token IDs.

## 15. Final implementation checklist

- [ ] Route CodeLlama model configs to shared Llama runtime lowering.
- [ ] Parse config defaults for `head_dim`, `num_key_value_heads`, `attention_bias`, `mlp_bias`, and standardized default RoPE.
- [ ] Reject unsupported non-default `rope_parameters` or projection biases until implemented.
- [ ] Load embedding, per-layer Q/K/V/O, RMSNorm, MLP, final norm, and LM head weights with alias tracking.
- [ ] Implement RMSNorm with fp32 accumulation.
- [ ] Implement default RoPE with configurable `rope_theta` and position offsets.
- [ ] Implement MHA and GQA causal attention without expanding stored cache.
- [ ] Define cache ABI as per-layer K/V `[B, KVH, max_seq, head_dim]`, post-RoPE K, append along sequence.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep`/last-token LM projection.
- [ ] Add tokenizer-side infill prompt parity tests outside compiled graph.
- [ ] Add one-block MHA and GQA parity tests.
- [ ] Add prefill and one-token decode parity with cache.
- [ ] Benchmark prefill, decode, KV memory, QKV packing, SwiGLU, and LM-head slicing.
