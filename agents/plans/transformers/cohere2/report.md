# Cohere2 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Cohere2 model_type; primary public bases appear to be CohereLabs/c4ai-command-r7b-12-2024 and CohereLabs/c4ai-command-a-03-2025, but official configs were gated from this environment.
Config source: pinned Transformers config source plus open HF mirror config.json files listed below.
Source files inspected:
- transformers/src/transformers/models/cohere2/configuration_cohere2.py
- transformers/src/transformers/models/cohere2/modeling_cohere2.py
- transformers/src/transformers/models/cohere2/modular_cohere2.py
- transformers/src/transformers/models/cohere/modeling_cohere.py, for Cohere differences
- transformers/src/transformers/masking_utils.py and cache_utils.py, for sliding masks/cache behavior
Any missing files or assumptions: official CohereLabs config.json downloads returned 401; representative checkpoint facts below use open mirrors and are labeled as such.
```

Upstream source URLs at the pinned commit:

- `configuration_cohere2.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere2/configuration_cohere2.py>
- `modeling_cohere2.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere2/modeling_cohere2.py>
- `modular_cohere2.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/cohere2/modular_cohere2.py>

Local config snapshots are under `agents/plans/transformers/cohere2/_sources/`.

Representative open mirror configs inspected:

- `mlx-community/c4ai-command-r7b-12-2024-8bit`, mirror of gated `CohereLabs/c4ai-command-r7b-12-2024`, sha `d1aa131fdad4a916c298c43ddf6aca4455b6d43a`.
- `Niazi/c4ai-command-r7b-12-2024-w4a16`, quantized mirror of `c4ai-command-r7b-12-2024`, sha `8440202011f48f812343a531453b1136b2c1dfd0`.
- `mlx-community/c4ai-command-a-03-2025-bf16`, mirror/fine-tune tag of gated `CohereLabs/c4ai-command-a-03-2025`, sha `8eda3c9ce9f5f3a331d7b4b8cbc3e168070c3339`.
- `mlx-community/c4ai-command-a-03-2025-8bit`, quantized mirror of `c4ai-command-a-03-2025`, sha `93ec895d3203243637cf79a6c7ba76b9e7a6d1a7`.
- `Firworks/c4ai-command-a-03-2025-nvfp4`, quantized mirror of `c4ai-command-a-03-2025`, sha `49579aace523ca3fc32a2d4843a96ad639fff921`.

The generated `modeling_cohere2.py` says it is generated from `modular_cohere2.py`; future source edits should target the modular file. Runtime parity should use the generated modeling file as the exact executed implementation.

## 2. High-level architecture

Cohere2 is a text-only decoder-only causal LM for autoregressive generation.

```text
token ids / input embeddings
-> token embedding
-> repeated Cohere2 decoder blocks
   -> input LayerNorm
   -> self-attention branch and gated MLP branch in parallel from the same normalized hidden state
   -> single residual add: residual + attention + mlp
-> final LayerNorm
-> tied/untied LM head
-> logits * logit_scale
-> sampling/generation controller
```

Primary runtime target for this report: `Cohere2ForCausalLM` prefill and decode. `Cohere2Model` is required as the base body. Training loss, gradient checkpointing, output attentions, and tensor/pipeline parallel plans are optional/deferred.

Stage decomposition:

- CPU/data pipeline: tokenization, chat template, padding, attention mask creation inputs.
- GPU runtime prefill: embeddings, full/sliding causal masks, RoPE, N decoder layers, final norm, selected logits.
- GPU runtime decode: same block with per-layer hybrid KV cache; sliding layers keep bounded context, full layers grow normally.
- Generation controller: `logits_to_keep`, sampling, EOS handling, and tokenizer-specific chat behavior live outside the core graph.

## 3. Important config dimensions

Source defaults from `Cohere2Config`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 256000 |
| `hidden_size` | 8192 |
| `intermediate_size` | 22528 |
| `num_hidden_layers` | 40 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | defaults to `num_attention_heads` if omitted |
| `head_dim` | computed as `hidden_size // num_attention_heads` in `__post_init__` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 8192 |
| `layer_norm_eps` | 1e-5 |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `sliding_window` | 4096 |
| `layer_types` | if omitted, every 4th layer is `full_attention`; others are `sliding_attention` |
| `rope_parameters` | standardized from legacy `rope_theta`/`rope_scaling`; source reads `config.rope_parameters["rope_type"]` |
| `logit_scale` | 0.0625 default in source; mirrors use 0.25 |
| `tie_word_embeddings` | true by source default |
| `use_cache` | true |

Representative config sweep from open mirrors:

| Mirror config | Source basis | Layers | Hidden | Heads / KV | Head dim | FFN | Max pos | Window / schedule | Logit scale | Dtype / quant |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| `mlx-community/c4ai-command-r7b-12-2024-8bit` | mirror of gated 7B | 32 | 4096 | 32 / 8 | 128 | 14336 | 8192 | `4096`, pattern 4 | 0.25 | bf16 source, MLX 8-bit fields |
| `Niazi/c4ai-command-r7b-12-2024-w4a16` | quantized 7B mirror | 32 | 4096 | 32 / 8 | 128 | 14336 | 132096 | `4096`, pattern 4 | 0.25 | compressed-tensors W4A16, ignores `lm_head` |
| `mlx-community/c4ai-command-a-03-2025-bf16` | mirror of 111B Command A | 64 | 12288 | 96 / 8 | 128 | 36864 | 131072 | `4096`, pattern 4 | 0.25 | bf16 |
| `mlx-community/c4ai-command-a-03-2025-8bit` | quantized Command A mirror | 64 | 12288 | 96 / 8 | 128 | 36864 | 131072 | `4096`, pattern 4 | 0.25 | MLX 8-bit fields |
| `Firworks/c4ai-command-a-03-2025-nvfp4` | quantized Command A mirror | 64 | 12288 | 96 / 8 | 128 | 36864 | 131072 | explicit 48 sliding + 16 full | 0.25 | NVFP4 compressed-tensors |

Fields present in mirror configs but not read by the current in-library `modeling_cohere2.py` include `position_embedding_type`, `rotary_pct`, `order_of_interleaved_layers`, `use_embedding_sharing`, `use_gated_activation`, `use_parallel_block`, `use_parallel_embedding`, `layer_switch`, and `cache_implementation`. They may matter to older remote code or loader tools, but DinoML should not treat them as native-source runtime requirements for this pinned source.

## 3a. Family variation traps

- GQA is the common production shape: `num_key_value_heads` is much smaller than `num_attention_heads` in mirrors. KV cache stores KV heads, not repeated query heads.
- `hidden_size == num_attention_heads * head_dim` in inspected configs, but source computes `head_dim` and projection widths explicitly. DinoML should validate, not infer silently.
- Hybrid attention schedule is critical. If `layer_types` is omitted, `sliding_window_pattern=4` from legacy configs means layers 4, 8, 12, ... are full attention, while the first three in each group are sliding attention.
- `sliding_window=None` disables sliding-window masks for layers typed as sliding only if admission routes carefully; source mask creation raises if a sliding mask is requested without a window.
- RoPE is not Llama-style concatenated frequency duplication. Cohere2 uses `repeat_interleave` and `rotate_half` over even/odd pairs.
- RoPE is applied only when `self.sliding_window is not None` inside `Cohere2Attention.forward`. Under the default hybrid schedule, full-attention layers do not apply RoPE in this pinned source. This is surprising and must be parity-tested before any "fixup" rewrite.
- Cohere2 removes Cohere's `use_qk_norm` path. There are no Q/K norms in Cohere2 source, even if older Cohere-family docs mention them.
- Decoder block uses a parallel residual form: attention and MLP both consume the same input-normalized hidden state, then `residual + attention + mlp`. This differs from sequential Llama blocks.
- LM logits are multiplied by `config.logit_scale` after the head. Production mirrors use `0.25`, while source default is `0.0625`.
- `lm_head.weight` is tied to `model.embed_tokens.weight` by default; lowering must preserve aliasing as one logical parameter unless `tie_word_embeddings=false`.
- Quantized mirror configs advertise MLX, EXL2, compressed-tensors, or NVFP4 metadata. Native Transformers Cohere2 source itself uses normal `nn.Linear`; quantized formats are loader/provider contracts, not model graph ops.
- No vision/audio/layout pass concerns. Tensors are source `[batch, seq, hidden]` and attention internal `[batch, heads, seq, head_dim]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `[B, S] -> [B, S, H]`.
- View/reshape for projections: `[B, S, heads * D] -> [B, S, heads, D]`.
- Transpose between `[B, S, heads, D]` and `[B, heads, S, D]`.
- Contiguous/reshape after attention: `[B, heads, S, D] -> [B, S, heads * D]`.
- Slice for `logits_to_keep`: integer suffix slice or tensor index over sequence dimension.
- Broadcast, expand, reshape for `repeat_kv`: `[B, KVH, S, D] -> [B, QH, S, D]` in eager fallback.

Neural network primitives:

- Biasless/bias-optional Linear:
  - Q: `Linear(H -> num_attention_heads * head_dim)`.
  - K/V: `Linear(H -> num_key_value_heads * head_dim)`.
  - O: `Linear(num_attention_heads * head_dim -> H)`.
  - MLP gate/up: `Linear(H -> intermediate_size)`, no bias.
  - MLP down: `Linear(intermediate_size -> H)`, no bias.
  - LM head: `Linear(H -> vocab_size)`, no bias, commonly tied with embedding.
- Cohere2 LayerNorm: mean/variance LayerNorm with learned weight, no bias, compute in fp32 then cast back.
- SiLU and elementwise multiply for gated MLP: `down(silu(gate(x)) * up(x))`.
- Parallel residual add: three-input add.
- Final logits scale: multiply by scalar `logit_scale`.

Attention primitives:

- Causal self-attention.
- Hybrid full/sliding attention per layer.
- GQA/MQA-compatible KV head repeat or native GQA backend.
- Softmax in fp32 for eager fallback, cast back to query dtype.
- Attention masks from `create_causal_mask` and `create_sliding_window_causal_mask`.
- SDPA/Flash/Flex attention dispatch through `ALL_ATTENTION_FUNCTIONS`, with eager fallback.

Position/rotary ops:

- RoPE inverse-frequency generation from `rope_parameters["rope_theta"]` and `head_dim`.
- Dynamic RoPE update support for non-default RoPE types via shared Transformers `ROPE_INIT_FUNCTIONS`.
- Interleaved GPT-J-style cos/sin expansion and even/odd `rotate_half`.

Generation/cache ops:

- `DynamicCache(config=config)` creates per-layer cache classes from `config.layer_types`.
- Full layers store `[B, KVH, total_seq, D]`.
- Sliding layers store bounded `[B, KVH, min(seq_len, sliding_window), D]` and update with sliding eviction.
- Cache reorder/reset for generation/beam search can be deferred for first greedy/sampling parity.

Quantized/packed weight metadata ops:

- Required only if DinoML wants to load quantized mirrors directly. Native source sees dense `nn.Linear` weights.
- `compressed-tensors` mirror names quantized `Linear` targets and may ignore `lm_head`; MLX/EXL2/NVFP4 mirrors need separate loader/provider admission.
- Safe first path: load dense/bf16 weights or dequantize external format into dense logical tensors before lowering.

## 5. Layer/block breakdown

For `B=batch`, `S=query length`, `H=hidden_size`, `QH=num_attention_heads`, `KVH=num_key_value_heads`, `D=head_dim`, `I=intermediate_size`:

```text
Embedding:
  hidden = embed_tokens(input_ids)  # [B, S, H]
  position_ids = arange(S) + past_seen_tokens, shape [1, S], unless supplied
  cos, sin = rotary_emb(hidden, position_ids)  # [B or 1, S, D]

Decoder block i, repeated N times:
  residual = hidden
  x = LayerNorm(hidden)  # fp32 mean/var, no bias

  q = q_proj(x).view(B, S, QH, D).transpose(1, 2)
  k = k_proj(x).view(B, S, KVH, D).transpose(1, 2)
  v = v_proj(x).view(B, S, KVH, D).transpose(1, 2)
  if layer i is sliding_attention:
      q, k = interleaved_rope(q, k, cos, sin)
  k, v = cache.update(k, v, layer_idx=i) if cache is enabled
  attn = attention(q, k, v, mask_for_layer_type, scaling=D**-0.5, sliding_window=maybe_4096)
  attn = o_proj(attn.reshape(B, S, QH * D))

  mlp = down_proj(silu(gate_proj(x)) * up_proj(x))
  hidden = residual + attn + mlp

Final:
  hidden = LayerNorm(hidden)
  logits = lm_head(hidden[:, slice_indices, :]) * logit_scale
```

Projection biases are controlled by `attention_bias`; inspected configs set false. MLP and LM head are biasless in source.

## 6. Attention requirements

Cohere2 attention is causal self-attention with hybrid full/sliding layer types.

| Requirement | Detail |
| --- | --- |
| Causality | causal decoder self-attention |
| Head pattern | GQA when `KVH < QH`; common mirrors use 32/8 or 96/8 |
| Head dim | 128 in inspected mirrors; source computes `H // QH` |
| Scaling | scores multiplied by `D**-0.5` |
| Masking | full layers use standard causal mask; sliding layers use causal window mask |
| Sliding schedule | default legacy pattern: 3 sliding layers then 1 full layer |
| Sliding mask rule | `kv_idx > q_idx - sliding_window` combined with causal mask |
| Backend | dispatches via configured attention implementation; supports FlashAttention, SDPA, Flex, eager |
| Eager fallback | repeats KV to query heads, matmul, add mask, fp32 softmax, dropout, matmul V |
| Dropout | zero in inference; source passes 0.0 when not training |
| Cache | per-layer KV cache; sliding layers bounded by window, full layers grow |
| Packed/varlen | not explicitly modeled in Cohere2 source; backend may handle packed internals |

Cache shapes before GQA repeat:

```text
new key/value per layer: [B, KVH, S_new, D]
full layer cache after update: [B, KVH, S_total, D]
sliding layer cache after update: [B, KVH, min(S_total, sliding_window), D]
eager repeated K/V for compute only: [B, QH, K_effective, D]
```

Cached keys are stored after the conditional RoPE application for sliding layers; full layers cache unrotated keys in this pinned source. That distinction is an admission trap for unified cache kernels.

## 7. Position encoding and custom math

Cohere2 RoPE source facts:

- `rope_parameters["rope_theta"]` feeds inverse frequencies.
- Default mirror configs carry legacy `rope_theta=50000`; `PreTrainedConfig` standardizes that into `rope_parameters`.
- Cos/sin are computed in fp32 under disabled autocast and cast to hidden dtype.
- Frequencies are duplicated with `torch.repeat_interleave(freqs, 2, dim=-1)`, not `cat(freqs, freqs)`.
- `rotate_half` pairs even/odd dimensions.

Implementation sketch:

```python
def cohere2_rope_tables(inv_freq, position_ids, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = repeat_interleave(freqs, repeats=2, dim=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)

def cohere2_rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return stack([-x_odd, x_even], dim=-1).flatten(-2)

def cohere2_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qf, kf = q.float(), k.float()
    return (qf * cos + cohere2_rotate_half(qf) * sin).to(q.dtype), (
        kf * cos + cohere2_rotate_half(kf) * sin
    ).to(k.dtype)
```

Precompute opportunities:

- `inv_freq` is static per config/rope type.
- Cos/sin for fixed decode positions can be cached by position range and dtype.
- Dynamic RoPE variants from `ROPE_INIT_FUNCTIONS` must preserve Transformers' update semantics; first integration should admit only default RoPE unless a checkpoint requires scaling.

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids: [B, S]` or `inputs_embeds: [B, S, H]`, exactly one required.
- Optional `attention_mask` accepted as a tensor or as a dict mapping `"full_attention"`/`"sliding_attention"` to prebuilt masks.
- Optional `position_ids: [B or 1, S]`. If omitted, source uses contiguous positions offset by `past_key_values.get_seq_length()`.
- Optional `past_key_values` cache.
- `logits_to_keep` can request only suffix logits for memory efficiency; default `0` means all logits in Python slice semantics.

CPU/data-pipeline work:

- Tokenizer/chat template and padding are outside `modeling_cohere2.py`.
- There is no model-coupled image/audio preprocessing.

GPU/runtime work:

- Attention mask creation may be lowered or supplied as precomputed runtime metadata. Hybrid masks are keyed by layer type.
- Position IDs and RoPE tables can be generated on GPU or passed/precomputed.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GQA attention without physical `repeat_kv`

Source pattern:

```text
K/V [B, KVH, K, D] -> expand/reshape repeat -> [B, QH, K, D] -> attention
```

Replacement:

```text
native grouped-query attention with QH/KVH grouping
```

Preconditions:

- `QH % KVH == 0`.
- No output attentions required, or dense attention weights can be reconstructed only for debug.
- Backend supports causal and sliding-window masks with KV cache.

Failure cases:

- Eager parity tests requesting `attn_weights` may require materialized repeated-head weights.

Parity test sketch:

- Compare one layer eager attention and native GQA for full and sliding masks, with cache lengths below/equal/above `sliding_window`.

### Rewrite: fused parallel decoder block

Source pattern:

```text
x = LayerNorm(hidden)
attn = Attention(x)
mlp = Down(SiLU(Gate(x)) * Up(x))
out = hidden + attn + mlp
```

Replacement:

```text
LayerNorm -> parallel attention and SwiGLU MLP -> fused three-input residual add
```

Preconditions:

- Same normalized `x` feeds attention and MLP.
- No hooks/output captures requiring branch intermediates.

Failure cases:

- Do not rewrite to sequential Llama-style `hidden + attn` then norm/MLP.

Parity test sketch:

- Single block with random weights, compare fp32 and bf16 tolerances.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden[:, slice_indices, :] -> lm_head -> scale
```

Replacement:

```text
gather/slice final hidden first, then GEMM only selected positions
```

Preconditions:

- `logits_to_keep` is integer suffix or static token index tensor.
- Loss is not being computed.

Failure cases:

- Full-sequence logits requested for scoring/perplexity.

Parity test sketch:

- Compare `logits_to_keep` values `0`, `1`, `N`, and tensor index.

### Rewrite: tied embedding/LM head alias

Source pattern:

```text
embed_tokens.weight is lm_head.weight when tie_word_embeddings=true
```

Replacement:

```text
one logical constant with two views/uses
```

Preconditions:

- Config has `tie_word_embeddings=true` or HF loader confirms alias.

Failure cases:

- Untied checkpoint or quantized format excluding `lm_head` from quantization.

Parity test sketch:

- Validate pointer/weight identity at load and logits after changing shared weight in a small synthetic model.

## 10. Kernel fusion candidates

Highest priority:

- Cohere2 LayerNorm: every block plus final norm; fp32 mean/variance and biasless affine.
- GQA FlashAttention/SDPA with hybrid full/sliding masks and KV cache. This is the central throughput/memory path.
- RoPE + attention prefill/decode, specifically interleaved GPT-J-style RoPE and conditional application for sliding layers.
- SwiGLU MLP: `silu(gate) * up` plus down projection.
- Last-token-only logits with scalar `logit_scale`.

Medium priority:

- Fused QKV projection as three GEMMs or packed GEMM with split order `[q, k, v]` only after weight packing is explicit. Source weights are separate `q_proj`, `k_proj`, `v_proj`.
- Parallel block scheduling: overlap or fuse residual accumulation from attention and MLP.
- Hybrid cache allocator: full layers grow, sliding layers use bounded windows.

Lower priority:

- Output attentions reconstruction.
- Training dropout/loss.
- Direct support for external quantized mirror formats before dense fallback is stable.

## 11. Runtime staging plan

Stage 1: parse config and dense weights.

- Admit native `Cohere2ForCausalLM` only.
- Reject or dequantize external quantization formats outside the graph.
- Preserve tied embedding/head alias.

Stage 2: one-block and full-prefill dense parity.

- Implement Cohere2 LayerNorm, interleaved RoPE, parallel residual, and SwiGLU.
- Use dense/eager attention first with full and sliding masks.

Stage 3: hybrid cache decode parity.

- Implement cache manifest per layer type.
- Validate sliding layers bounded to `sliding_window`; full layers grow.

Stage 4: optimized attention.

- Replace repeat-KV eager path with native GQA attention.
- Add full and sliding window backend variants.

Stage 5: logits and generation integration.

- Add `logits_to_keep`, scalar logit scale, sampler-facing output.

Stage 6: quantized/offload loading.

- Add provider-specific admission for compressed-tensors/MLX/EXL2/NVFP4 only after dense path and current DinoML GGUF loading contracts are stable.

## 12. Parity and validation plan

- Config parsing tests:
  - omitted `num_key_value_heads` defaults to `num_attention_heads`;
  - omitted `layer_types` with `sliding_window_pattern=4` produces 3 sliding + 1 full schedule;
  - legacy `rope_theta` becomes effective `rope_parameters["rope_theta"]`.
- Custom op tests:
  - Cohere2 LayerNorm versus PyTorch source for fp32/bf16;
  - interleaved RoPE versus source snippet;
  - sliding causal mask around boundary positions `sliding_window-1`, `sliding_window`, `sliding_window+1`.
- Single-layer parity:
  - full-attention layer and sliding-attention layer separately;
  - GQA shapes `QH=32, KVH=8` and `QH=96, KVH=8`.
- Full prefill parity:
  - small synthetic config, all logits;
  - production-shaped config with random weights if memory allows, selected logits only.
- Decode parity:
  - token-by-token decode after prefill;
  - cache length crosses sliding window;
  - mixed full/sliding layers.
- End-to-end text parity:
  - compare next-token logits for known prompts on a small/dequantized checkpoint.
- Tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5`;
  - bf16/fp16 optimized attention: start with `rtol=2e-2`, `atol=2e-2`, then tighten per backend.

## 13. Performance probes

- Prefill-only tokens/sec across sequence lengths: 1k, 4k, 8k, 32k, 128k where config admits.
- Decode tokens/sec versus batch size with mixed full/sliding layers.
- KV cache memory split by full layers and sliding layers.
- Sliding-window attention backend comparison: eager, SDPA, FlashAttention-compatible local window, custom.
- GQA repeat materialization overhead versus native grouped attention.
- MLP GEMM throughput and SwiGLU activation bandwidth.
- LM head cost with all logits versus `logits_to_keep=1`.
- Dense bf16 load time and memory footprint versus quantized mirror dequant paths.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Output attentions unless debugging requires them.
- Beam-search cache reorder for first greedy/sampling integration.
- Tensor/pipeline parallel plans.
- Non-default dynamic/yarn/longrope variants unless a selected checkpoint requires them.
- Direct MLX/EXL2/NVFP4/compressed-tensors execution; use dense fallback first.
- Any remote-code-only behavior advertised by historical config fields but not read by pinned native source.

## 15. Final implementation checklist

- [ ] Parse `Cohere2Config`, including legacy `sliding_window_pattern` and RoPE parameter standardization.
- [ ] Load dense weights and preserve tied `embed_tokens.weight` / `lm_head.weight`.
- [ ] Implement Cohere2 biasless LayerNorm.
- [ ] Implement interleaved Cohere2 RoPE and `rotate_half`.
- [ ] Implement hybrid layer schedule and mask creation.
- [ ] Implement GQA attention with native KV-head cache layout.
- [ ] Implement sliding-window cache layers and full-attention cache layers in one cache manifest.
- [ ] Implement parallel decoder residual block.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep` and final `logit_scale`.
- [ ] Add single-layer full/sliding parity tests.
- [ ] Add prefill logits parity.
- [ ] Add decode/cache parity crossing the sliding-window boundary.
- [ ] Add performance probes for prefill, decode, KV memory, and logits slicing.
- [ ] Add optional dense-fallback loader path for quantized mirror configs before any direct quantized provider path.
