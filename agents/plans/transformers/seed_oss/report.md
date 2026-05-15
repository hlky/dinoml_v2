# Seed-OSS Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Public checkpoint configs report transformers_version 4.55.0.

Model id:
  Primary: ByteDance-Seed/Seed-OSS-36B-Instruct
  Representative configs: ByteDance-Seed/Seed-OSS-36B-Instruct,
  ByteDance-Seed/Seed-OSS-36B-Base, ByteDance-Seed/Seed-OSS-36B-Base-woSyn.

Config source:
  Raw Hugging Face config.json files plus source configuration defaults.

Source files inspected:
  transformers/src/transformers/models/seed_oss/configuration_seed_oss.py
  transformers/src/transformers/models/seed_oss/modular_seed_oss.py
  transformers/src/transformers/models/seed_oss/modeling_seed_oss.py
  transformers/src/transformers/modeling_rope_utils.py
  transformers/src/transformers/configuration_utils.py

Any missing files or assumptions:
  No model-specific processor/image/audio files exist. `modeling_seed_oss.py`
  is generated from `modular_seed_oss.py`; future Transformers source edits
  should target modular_seed_oss.py, but DinoML runtime behavior should audit
  the generated modeling file shipped in the package. No gated links were hit.
```

Source URLs:

- [configuration_seed_oss.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/seed_oss/configuration_seed_oss.py)
- [modular_seed_oss.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/seed_oss/modular_seed_oss.py)
- [modeling_seed_oss.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/seed_oss/modeling_seed_oss.py)
- [Seed-OSS-36B-Instruct](https://hf.co/ByteDance-Seed/Seed-OSS-36B-Instruct)
- [Seed-OSS-36B-Base](https://hf.co/ByteDance-Seed/Seed-OSS-36B-Base)
- [Seed-OSS-36B-Base-woSyn](https://hf.co/ByteDance-Seed/Seed-OSS-36B-Base-woSyn)

## 2. High-level architecture

Primary DinoML runtime target: causal language model prefill and decode for `SeedOssForCausalLM`.

Seed-OSS is a text-only decoder stack: token embedding -> repeated RMSNorm/GQA/RoPE/SwiGLU decoder blocks -> final RMSNorm -> untied LM head. It has no vision/audio branch, no MoE, no cross-attention, and no source-owned preprocessing beyond tokenizer/chat-template behavior.

```text
tokenizer/chat template -> input_ids/attention_mask -> embedding
  -> 64x decoder block with causal GQA + RoPE + SwiGLU
  -> final RMSNorm -> lm_head -> logits/sampling
```

Independently stageable pieces:

- CPU/data pipeline: tokenizer, chat template, thinking-budget/tool tokens.
- GPU prefill: embedding, full causal attention, full logits or last-token logits.
- GPU decode: one/new-token step with per-layer KV cache.
- Generation controller: sampling, temperature/top-p, chat/thinking-budget conventions. This is ABI/control logic, not a neural op.

Other heads in source: base `SeedOssModel`, sequence classification, token classification, and question answering. For this report they are deferred; they reuse the decoder body but require task-specific pooling/span logits.

## 3. Important config dimensions

Hub 36B config dimensions:

| Field | Value | Provenance |
|---|---:|---|
| `vocab_size` | 155136 | `config.json` |
| `hidden_size` | 5120 | `config.json` |
| `num_hidden_layers` | 64 | `config.json` |
| `num_attention_heads` | 80 | `config.json` |
| `num_key_value_heads` | 8 | `config.json` |
| `num_key_value_groups` | 10 | inferred from source math |
| `head_dim` | 128 | `config.json` |
| Q projection width | 10240 | inferred: `80 * 128` |
| K/V projection width | 1024 each | inferred: `8 * 128` |
| attention output input width | 10240 | inferred: `80 * 128` |
| `intermediate_size` | 27648 | `config.json` |
| `hidden_act` | `silu` | `config.json` |
| `max_position_embeddings` | 524288 | `config.json` |
| RoPE | default, theta `1e7` | `config.json` legacy `rope_scaling`/`rope_theta`, normalized by config utils |
| `rms_norm_eps` | `1e-6` | `config.json` |
| attention Q/K/V bias | true | `config.json` and source |
| attention output bias | false | `config.json` and source |
| MLP bias | false | `config.json` and source |
| `tie_word_embeddings` | false | `config.json` |
| checkpoint dtype | bfloat16 | `config.json`; index size is consistent |
| parameters | 36,151,104,512 | safetensors index metadata |

Representative checkpoint sweep is in [config_sweep.md](H:/dinoml_v2/agents/plans/transformers/seed_oss/config_sweep.md). The three public ByteDance-Seed 36B variants share the same architecture config; Base vs Instruct differences are tokenizer/template/weights/generation behavior, not operator structure.

## 3a. Family variation traps

- Source defaults are not the public 36B shape. `SeedOssConfig` defaults to `hidden_size=4096`, while all public 36B configs inspected use `hidden_size=5120`. DinoML should require a checkpoint config for production shape planning.
- `hidden_size != num_attention_heads * head_dim` for public configs: `5120 != 80 * 128 = 10240`. Q/O attention width is twice the residual hidden width.
- GQA is required: `num_key_value_heads=8`, Q heads `80`, repeat factor `10`.
- Q/K/V projections are separate modules, not a single packed QKV weight in source.
- Q/K/V projection biases exist; O projection and MLP projections are biasless for public configs.
- Hub configs use legacy `rope_scaling` and `rope_theta`; current config utilities normalize these into `rope_parameters`.
- `tie_word_embeddings=false`; LM head and embedding are separate physical tensors despite `_tied_weights_keys` metadata existing for optional tying.
- Long context is native by config (`524288`), but source attention is ordinary causal attention plus backend dispatch. No sliding-window/local sparse attention field is read.
- Dropout fields are nonzero in configs, but inference mode makes dropout inactive.
- Thinking-budget/tool tokens are tokenizer/chat-template behavior. They do not add neural graph ops.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,5120]`.
- View/reshape/transpose for attention: `[B,S,10240] -> [B,80,S,128]`, `[B,S,1024] -> [B,8,S,128]`.
- Contiguous/reshape after attention: `[B,80,S,128] -> [B,S,10240]`.
- Slice/select for `logits_to_keep`: full sequence by default, last `K` tokens for efficient decode/prefill logits.
- Causal mask construction from `attention_mask`, `position_ids`, and cache length.

Neural primitives:

- RMSNorm over last dim with fp32 variance, weight scale.
- Linear: embedding hidden to Q `[5120 -> 10240]` with bias.
- Linear: embedding hidden to K/V `[5120 -> 1024]` with bias.
- Linear: attention output `[10240 -> 5120]` without bias.
- SwiGLU MLP: gate/up `[5120 -> 27648]`, `silu(gate) * up`, down `[27648 -> 5120]`.
- Residual adds around attention and MLP.
- LM head `[5120 -> 155136]` without bias, untied.

Attention primitives:

- Causal self-attention, GQA/MQA-style repeat of KV heads from 8 to 80.
- RoPE on Q and K before cache update.
- KV cache update per layer.
- Backend-compatible SDPA/Flash/Flex attention; eager fallback repeats KV then does matmul, mask add, fp32 softmax, dropout, value matmul.

Position/rotary:

- Default RoPE over full `head_dim=128`, base theta `1e7`.
- Cos/sin computed in fp32 from `position_ids`, then cast to hidden dtype.

Generation/cache:

- DynamicCache if `use_cache=True` and no cache is provided.
- Per-layer K/V tensors before KV repeat, shape `[B,8,T,128]`.
- `position_ids = arange(current S) + past_seen_tokens` when omitted.

Distributed/tensor-parallel metadata:

- Source config declares TP hints: Q/K/V/gate/up colwise, O/down rowwise, LM head colwise gather. First DinoML target can ignore distributed execution but should preserve weight names and output orientation.

## 5. Layer/block breakdown

Decoder block, repeated `64` times for public checkpoints:

```text
x: [B,S,5120]
residual = x
x = RMSNorm(x)
q = Linear(x, 5120 -> 10240, bias=True).view(B,S,80,128).transpose(1,2)
k = Linear(x, 5120 -> 1024, bias=True).view(B,S,8,128).transpose(1,2)
v = Linear(x, 5120 -> 1024, bias=True).view(B,S,8,128).transpose(1,2)
q,k = RoPE(q,k, cos/sin[position_ids])
k,v = cache.update(k,v, layer_idx) when cache is present
attn = causal_attention(q,k,v, GQA repeat 8 -> 80)
attn = attn.reshape(B,S,10240)
x = residual + Linear(attn, 10240 -> 5120, bias=False)
residual = x
x = RMSNorm(x)
mlp = Linear(silu(Linear(x, 5120 -> 27648)) * Linear(x, 5120 -> 27648), 27648 -> 5120)
x = residual + mlp
```

Model wrapper:

```text
input_ids -> Embedding(155136,5120)
decoder blocks
final RMSNorm
lm_head selected hidden states -> [B,K_or_S,155136]
```

## 6. Attention requirements

- Variant: causal decoder self-attention.
- Head structure: GQA with `num_attention_heads=80`, `num_key_value_heads=8`, repeat factor `10`, `head_dim=128`.
- Query width: `10240`; key/value width before repeat: `1024`.
- Masking: causal mask from Transformers `create_causal_mask`; additive mask in eager path before softmax.
- Cache: K/V stored after RoPE and before KV repeat. Per layer each of K and V is `[B,8,T,128]`; after repeat for eager attention it becomes `[B,80,T,128]`.
- Rectangular attention: decode uses query length usually `1`, key length `past + 1`; prefill uses `S x S`.
- Packed/varlen support: not model-specific in source. FlashAttention/SDPA compatibility is advertised by `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn`.
- Sliding/local attention: not implemented for Seed-OSS source; reject configs that introduce such fields unless a separate audit proves support.
- Output attentions: eager path can return dense attention weights `[B,80,Q,K]`; optimized backends may not need this for first inference target.

## 7. Position encoding and custom math

Source RoPE computes inverse frequencies from `rope_parameters["rope_theta"]` and `head_dim`, then forms cos/sin from `position_ids`.

```python
def seed_oss_rope(position_ids, head_dim=128, theta=10000000.0):
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
    emb = cat((freqs.transpose(1, 2), freqs.transpose(1, 2)), dim=-1)
    return cos(emb), sin(emb)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate = lambda x: cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
```

Precompute opportunity: for static max context/buckets, `inv_freq` is constant and cos/sin tables can be cached by position range and dtype. Dynamic decode still needs position offset from cache length.

## 8. Preprocessing and input packing

Model graph inputs are text-only:

- `input_ids: [B,S] int64` or `inputs_embeds: [B,S,5120]`, exactly one required.
- Optional `attention_mask`; source passes it to causal mask construction.
- Optional `position_ids: [B,S]`; if omitted, source derives it from current sequence and cache length.
- Optional `past_key_values`.

Tokenizer/control ABI:

- Fast tokenizer, special IDs `bos=0`, `pad=1`, `eos=2`.
- Instruct chat templates may insert thinking-budget and tool-call tokens such as `<seed:think>`, `<seed:cot_budget_reflect>`, and `<seed:tool_call>`.
- These tokens are ordinary token IDs to the neural graph. Budget enforcement and direct-answer behavior are generation-controller/template responsibilities.

No image/audio/video preprocessing, placeholder scatter, packed patch metadata, or channel-layout translation is applicable.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> grouped projection schedule

Source pattern:

```text
q = Linear(x, Wq, bq)
k = Linear(x, Wk, bk)
v = Linear(x, Wv, bv)
```

Replacement: one scheduled GEMM group or packed GEMM producing three output ranges, followed by view/transpose.

Preconditions:

- Same input `x`, same dtype/device, no intervening consumers that require materialized partial outputs.
- Preserve separate weight storage names for loading.
- Packed output order must be `[q, k, v]` with widths `[10240, 1024, 1024]`.

Failure cases: tensor parallel sharding, partial materialization for debugging, or configs with changed biases.

Parity test: compare unpacked Q/K/V tensors before RoPE for one layer.

### Rewrite: GQA FlashAttention

Source pattern: RoPE Q/K, cache update, backend attention with GQA.

Replacement: fused causal attention that accepts Q `[B,80,Q,128]`, K/V `[B,8,K,128]`, avoids explicit KV repeat, and updates/reads cache.

Preconditions:

- Causal self-attention only.
- RoPE applied before cache write.
- Additive/padding mask semantics match source.
- Head repeat factor `80 / 8` integer.

Failure cases: requested dense attention weights, unsupported custom 4D masks, or backend not supporting GQA directly.

### Rewrite: SwiGLU fused MLP

Source pattern:

```text
down(silu(gate(x)) * up(x))
```

Replacement: two input GEMMs plus fused `silu*multiply`, then down GEMM. If weight packing exists, gate/up can be grouped.

Preconditions: `hidden_act == "silu"` and `mlp_bias` matches loaded config.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: compile/runtime option to project only last token or selected token indices.

Preconditions: caller sets `logits_to_keep=1` or a static/index tensor; loss is not computed.

Failure cases: full prefill logits needed for evaluation/loss.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: twice per block plus final norm; fp32 variance with reduced-precision output.
- GQA FlashAttention with RoPE and KV cache: dominant prefill/decode cost; must avoid materializing repeated KV.
- Q/K/V projection scheduling: large asymmetric projections with biases.
- SwiGLU MLP fusion: huge intermediate width `27648`, activation multiply is bandwidth-sensitive.
- Last-token LM head: vocab `155136` makes full prefill logits expensive.

Medium priority:

- Residual add fused with projection output where artifact scheduling allows.
- RoPE table/cache generation for long context buckets.
- GEMM provider coverage for `bfloat16` with bias on Q/K/V and no-bias O/MLP/LM head.
- KV cache layout kernels for append/reorder and beam/cache gather if generation support expands.

Lower priority:

- Classification/QA heads.
- Training loss and dropout.
- Distributed/tensor-parallel execution from source TP hints.

## 11. Runtime staging plan

1. Parse config and reject unsupported variants: require causal `seed_oss`, no MoE/sliding attention, known RoPE types initially `default`.
2. Load weights with exact names and shapes; support untied embedding/LM head and Q/K/V biases.
3. Single-block eager parity in fp32/bf16 with no cache, small `B,S`.
4. Full prefill parity for logits on small sequence, then long-context smoke with smaller hidden test config if available.
5. Decode parity with DynamicCache-equivalent per-layer K/V state.
6. Enable optimized GQA attention and last-token logits.
7. Add graph rewrites for grouped Q/K/V and SwiGLU.
8. Add production probes for batch/sequence/cache memory and bf16 throughput.

Initially stub/defer: sampling controller, chat template, classification/QA heads, tensor parallel, quantized loading.

## 12. Parity and validation plan

- Config parser tests: public 36B configs normalize `rope_scaling`/`rope_theta` into effective default RoPE theta `1e7`.
- Operator tests: RMSNorm fp32 accumulation, RoPE, KV repeat/GQA attention, SwiGLU.
- Weight-shape tests: assert Q `[10240,5120]`, K/V `[1024,5120]`, O `[5120,10240]`, gate/up `[27648,5120]`, down `[5120,27648]`, LM head `[155136,5120]`.
- Single-layer parity: random small config including `hidden_size != num_heads * head_dim`.
- Full-model small-config parity: prefill logits and hidden states against Transformers.
- Decode parity: prefill cache then one-token decode equals full recompute next-token logits.
- Tolerances: fp32 `1e-4` to `3e-4`; bf16/fp16 attention logits `1e-2` to `3e-2` depending on backend, with stricter per-op tolerances before fused attention.

## 13. Performance probes

- Prefill tokens/sec sweep: `S={1K,4K,16K,64K}` subject to memory.
- Decode tokens/sec sweep: batch size and active cache length.
- KV cache memory: `layers * 2 * B * KV_heads * T * head_dim * dtype_size`.
- Attention backend comparison: eager repeat-KV vs SDPA/Flash-style direct GQA.
- MLP throughput: gate/up/down GEMM time and activation-multiply bandwidth.
- LM head: full sequence logits vs last-token-only logits.
- Weight loading memory: bf16 dense load footprint from safetensors index, plus future quantized/GGUF alternatives if introduced separately.

## 14. Skip/defer list

- Training, gradients, dropout behavior, and loss.
- Sequence/token classification and QA heads for the first causal-LM target.
- Beam search/cache reorder unless generation controller work begins.
- Tensor parallel / pipeline parallel execution despite source metadata.
- BitsAndBytes 4-bit/8-bit model-card examples; no source-coupled packed format is implemented in `seed_oss` modeling code.
- Tool-call parser, thinking-budget policy, and chat-template rendering inside DinoML runtime.
- Dense attention weight outputs for optimized path unless explicitly requested.

## 15. Final implementation checklist

- [ ] Parse `SeedOssConfig`, including legacy `rope_scaling` + `rope_theta`.
- [ ] Require checkpoint config; do not use class defaults as 36B shape facts.
- [ ] Load embeddings, untied LM head, Q/K/V biases, and biasless O/MLP weights.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement default RoPE with theta `1e7` and cache-position offsets.
- [ ] Implement causal GQA attention with KV cache stored before repeat.
- [ ] Implement Seed-OSS decoder block with attention width `num_heads * head_dim`.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep`/last-token LM-head lowering.
- [ ] Add Q/K/V grouped projection rewrite with shape/order guards.
- [ ] Add GQA FlashAttention lowering with no explicit KV repeat.
- [ ] Add single-op, single-block, prefill, and decode parity tests.
- [ ] Benchmark prefill, decode, KV memory, MLP, and LM-head variants.
