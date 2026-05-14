# DinoML Transformers Audit: StableLM

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: stabilityai/stablelm-3b-4e1t as primary; stablelm-zephyr-3b, stablelm-2-1_6b, tiny-random-stablelm-2 as config sweep
Config source: Hugging Face raw config.json files, plus StableLmConfig defaults
Source files inspected:
  X:/H/transformers/src/transformers/models/stablelm/configuration_stablelm.py
  X:/H/transformers/src/transformers/models/stablelm/modeling_stablelm.py
  X:/H/transformers/tests/models/stablelm/test_modeling_stablelm.py
  X:/H/transformers/docs/source/en/model_doc/stablelm.md
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/modeling_layers.py
Any missing files or assumptions:
  No modular StableLM source exists in this checkout; modeling_stablelm.py is authoritative.
  Tokenizer coupling is light: docs/auto mappings identify GPTNeoXTokenizer/GPTNeoXTokenizerFast for StableLM 3B.
  GGUF mirrors were inspected only for config availability; packed weight metadata must be read from GGUF files at load time.
```

Primary DinoML target: `StableLmForCausalLM` inference, covering prefill logits and autoregressive decode with KV cache. `StableLmModel` is required as the body. Sequence and token classification heads are optional/deferred because they use generic Transformers heads on top of the same decoder body.

## 2. High-level architecture

StableLM is a text-only decoder-only causal LM:

```text
tokenizer/input_ids -> token embedding -> N decoder blocks -> final LayerNorm -> LM head -> logits/sampling
```

Decode stage decomposition:

```text
CPU tokenizer/generation controller -> GPU prefill -> per-layer KV cache -> GPU one-token decode -> last-token logits -> sampler
```

There is no vision/audio branch, no encoder-decoder cross-attention, no MoE, and no state-space recurrence. Independently stageable pieces are embeddings/config+weight loading, one decoder block parity, full prefill, KV-cache decode, and optional head wrappers.

## 3. Important config dimensions

Config defaults come from `StableLmConfig`; representative checkpoint values come from `config.json`.

| Field | Source default | 3B 4E1T / Zephyr 3B | StableLM 2 1.6B | tiny random StableLM 2 |
| --- | ---: | ---: | ---: | ---: |
| `hidden_size` | 2560 | 2560 | 2048 | 512 |
| `num_hidden_layers` | 32 | 32 | 24 | 8 |
| `num_attention_heads` | 32 | 32 | 32 | 16 |
| `num_key_value_heads` | 32 | 32 | 32 | 4 |
| `head_dim` | `hidden_size / heads` | 80 | 64 | 32 |
| `rotary_ndims` | `head_dim * partial_rotary_factor` | 20 | 16 | 8 |
| `intermediate_size` | 6912 | 6912 | 5632 | 1536 |
| `vocab_size` | 50304 | 50304 | 100352 | 100352 |
| `max_position_embeddings` | 4096 | 4096 | 4096 | 4096 |
| `hidden_act` | `silu` | `silu` | `silu` | `silu` |
| `layer_norm_eps` | `1e-5` | `1e-5` | `1e-5` | `1e-5` |
| `use_qkv_bias` | false | false | true | false |
| `qk_layernorm` | false | omitted/effective false | omitted/effective false | true |
| `use_parallel_residual` | false | omitted/effective false | omitted/effective false | true |
| `tie_word_embeddings` | false | false | false | false |
| `torch_dtype` | config-dependent | bfloat16 | float16 | bfloat16 |
| `use_cache` | true | true | true | true |

Checkpoint sweep:

| Model id | Purpose | Structural variation |
| --- | --- | --- |
| `stabilityai/stablelm-3b-4e1t` | common base production checkpoint and Transformers integration-test target | 32-layer MHA, no QKV bias, BF16, GPT-NeoX tokenizer family |
| `stabilityai/stablelm-zephyr-3b` | instruction-tuned sibling | same neural dimensions as 3B 4E1T; generation/chat template differences are outside the core module graph |
| `stabilityai/stablelm-2-1_6b` | smaller production checkpoint | FP16, larger vocab, QKV bias enabled |
| `stabilityai/tiny-random-stablelm-2` | debug/checkpoint fixture | GQA (`kv_heads=4`), per-head Q/K layernorm, parallel residual path |
| `afrideva/stablelm-3b-4e1t-GGUF` | GGUF mirror referenced by Transformers tests | raw config only has `model_type`; require original config or GGUF metadata for shape admission |

## 3a. Family variation traps

- `num_key_value_heads` can be smaller than `num_attention_heads`; DinoML must support GQA/MQA-style KV repeat or native GQA attention. The tiny random StableLM 2 fixture has `heads=16`, `kv_heads=4`.
- `use_qkv_bias` is checkpoint-significant. StableLM 2 1.6B enables bias on Q/K/V projections; 3B 4E1T does not. `o_proj`, MLP projections, and LM head are bias-free in source.
- `qk_layernorm` is optional. When true, source applies one independent `LayerNorm(head_dim)` module per Q head and per KV head after projection/reshape and before RoPE.
- `use_parallel_residual` changes block equations. The non-parallel path has two LayerNorms; the parallel path reuses `input_layernorm(x)` for attention and MLP, then adds `residual + attn + mlp`.
- RoPE is partial: only the first `int(head_dim * partial_rotary_factor)` channels of Q/K rotate; the pass-through channels are concatenated afterward.
- `hidden_size` must be divisible by `num_attention_heads`; source enforces this. `head_dim` is inferred as integer division, not an independent config field.
- No sliding-window config is read by `StableLmModel`; any historical config advertising sliding/local attention should be rejected for this source basis unless routed through a different audited implementation.
- `rope_scaling` and `rotary_scaling_factor` appear in the tiny random config, but the inspected source reads `config.rope_parameters`; old scalar fields are compatibility inputs to config normalization, not direct modeling operations.
- `tie_word_embeddings=false` in representative configs. Source declares a tied-weights key for the LM head, but config does not tie by default; weight identity should follow `tie_word_embeddings`.
- There is no NCHW/NHWC tensor path in this text-only decoder. Layout rewrites are about sequence/head layouts, not image channel-last translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `input_ids[B,S] -> hidden[B,S,H]`.
- Reshape/view/transpose for Q/K/V: `[B,S,H] -> [B,heads,S,D]` and `[B,kv_heads,S,D]`.
- Slice and concatenate on last dim for partial RoPE: `rotary_ndims` and `head_dim - rotary_ndims`.
- KV cache append/concatenate along sequence dim `-2`.
- Optional `repeat_kv`: `[B,kv_heads,T,D] -> [B,heads,T,D]` if not using native GQA attention.
- Last-token or indexed logits slice: `hidden_states[:, slice_indices, :]`.

Neural network primitives:

- `LayerNorm(H)` with affine weight/bias, eps `1e-5`, used before attention, optionally after attention, and at final norm.
- Optional per-head `LayerNorm(D)` modules for Q and K when `qk_layernorm=true`.
- Linear GEMMs:
  - 3B: Q `2560 -> 2560`, K `2560 -> 2560`, V `2560 -> 2560`, O `2560 -> 2560`, gate/up `2560 -> 6912`, down `6912 -> 2560`, LM head `2560 -> 50304`.
  - 1.6B: Q/K/V/O `2048 -> 2048`, gate/up `2048 -> 5632`, down `5632 -> 2048`, LM head `2048 -> 100352`; Q/K/V have bias.
  - Tiny GQA fixture: Q `512 -> 512`, K/V `512 -> 128`, O `512 -> 512`, gate/up `512 -> 1536`, down `1536 -> 512`.
- SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Residual add patterns: serial two-add or parallel three-term add.
- Dropout appears in source but is inactive for inference.

Attention primitives:

- Causal self-attention, MHA or GQA.
- Dense prefill attention over `[B,heads,Q,K]`.
- Decode attention with per-layer KV cache shape `[B,kv_heads,T,D]`.
- Additive causal/padding mask before softmax.
- Softmax over key dimension with fp32 accumulation in eager source.
- Optional backend dispatch to SDPA or FlashAttention 2 via `ALL_ATTENTION_FUNCTIONS`.

Position/rotary/custom math:

- RoPE inv-frequency table from `rope_theta` and partial rotary dimension.
- Float32 position-frequency matmul and `cos`/`sin`; output cast to hidden dtype.
- `rotate_half`, elementwise multiply/add, and concat with pass-through Q/K channels.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache=True` and no cache is passed.
- Per-layer cache update with keys/values stored before any `repeat_kv` expansion.
- Position IDs default to `[past_seen_tokens, ..., past_seen_tokens + S - 1]`.
- Generation controller can set `logits_to_keep=1` to avoid full-sequence LM-head GEMM during decode.

Optional heads:

- Sequence classification: generic full-sequence linear `H -> num_labels`, then rightmost non-pad token selection via mask/argmax. Defer for causal LM target.
- Token classification: dropout plus linear `H -> num_labels`. Defer for causal LM target.

Quantized/packed weight metadata:

- Native StableLM source uses dense PyTorch linear weights. Transformers GGUF tests reference StableLM GGUF mirrors; DinoML should treat GGUF as a weight-loading/provider contract with dense fallback or bounded runtime dequant, not as a StableLM graph op.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x                                      # x [B,S,H]
x_norm = LayerNorm(H)(x)
q = Linear(H -> heads*D, bias=use_qkv_bias)(x_norm)
k = Linear(H -> kv_heads*D, bias=use_qkv_bias)(x_norm)
v = Linear(H -> kv_heads*D, bias=use_qkv_bias)(x_norm)
q = view_transpose(q)                             # [B,heads,S,D]
k = view_transpose(k)                             # [B,kv_heads,S,D]
v = view_transpose(v)                             # [B,kv_heads,S,D]
if qk_layernorm:
  q = per_head_LayerNorm(D)(q)
  k = per_head_LayerNorm(D)(k)
q_rot, q_pass = split(q, [rotary_ndims, D-rotary_ndims])
k_rot, k_pass = split(k, [rotary_ndims, D-rotary_ndims])
q_rot, k_rot = partial_RoPE(q_rot, k_rot, cos, sin)
q = concat(q_rot, q_pass, dim=-1)
k = concat(k_rot, k_pass, dim=-1)
k, v = cache_update(k, v)                         # if cache object exists
attn = causal_attention(q, k, v, mask, scale=D^-0.5)
attn = Linear(H -> H, bias=False)(reshape(attn))
if use_parallel_residual:
  mlp = Linear(intermediate -> H)(silu(gate(x_norm)) * up(x_norm))
  x = residual + attn + mlp
else:
  mid = residual + attn
  mlp = Linear(intermediate -> H)(silu(gate(LayerNorm(H)(mid))) * up(LayerNorm(H)(mid)))
  x = mid + mlp
```

After all blocks:

```text
x = final LayerNorm(H)(x)
logits = Linear(H -> vocab_size, bias=False)(x[:, logits_to_keep_slice, :])
```

## 6. Attention requirements

- Type: causal decoder self-attention only.
- Heads: MHA when `num_key_value_heads == num_attention_heads`; GQA when smaller.
- Projection widths: Q output is `num_attention_heads * head_dim == hidden_size`; K/V output is `num_key_value_heads * head_dim`.
- Head dim: inferred from `hidden_size // num_attention_heads`.
- Scaling: attention scores multiplied by `head_dim ** -0.5`.
- Masking: `create_causal_mask` combines causal and optional padding mask. Eager path adds mask to attention scores before softmax.
- Softmax: eager implementation requests `dtype=torch.float32`, then casts probabilities back to query dtype.
- Cache: per layer, keys and values are stored as `[B,kv_heads,T,D]` after RoPE for keys and before any repeat-to-query-head expansion. `DynamicCache` appends along sequence dimension.
- Position IDs: default single row `[S]` offset by current cache length; callers may pass `[B,S]`.
- Packed/varlen: source passes `position_ids` through to attention interface for FlashAttention 2, but no StableLM-specific packed sequence ABI is present.
- Sliding/local attention: not implemented by StableLM source.
- Backend parity: initial DinoML can implement an eager-equivalent dense attention path, then lower MHA/GQA to FlashAttention/SDPA-style kernels when masks, RoPE placement, and cache shape match.

Cache shapes:

```text
prefill input q,k,v:
  q [B, heads, S, D]
  k/v [B, kv_heads, S, D]
cache after prefill:
  K/V layer_i [B, kv_heads, S_total, D]
decode step:
  q [B, heads, 1, D]
  K/V read [B, kv_heads, S_total, D]
  attention output [B, heads, 1, D] -> [B,1,H]
```

## 7. Position encoding and custom math

StableLM uses partial RoPE. Only a prefix of the head dimension rotates; the remaining channels pass through unchanged.

```python
def stablelm_partial_rope(q, k, position_ids, rope_theta, partial_rotary_factor):
    # q: [B, heads, S, D], k: [B, kv_heads, S, D]
    rotary_ndims = int(D * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (arange(0, rotary_ndims, 2) / rotary_ndims))
    freqs = position_ids[:, None, :].float().transpose_like_matmul(inv_freq)
    emb = concat(freqs, freqs, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    q_rot = q_rot * cos[:, None, :, :] + rotate_half(q_rot) * sin[:, None, :, :]
    k_rot = k_rot * cos[:, None, :, :] + rotate_half(k_rot) * sin[:, None, :, :]
    return concat(q_rot, q_pass, -1), concat(k_rot, k_pass, -1)
```

`rotate_half(x)` splits the last dim in half and returns `concat(-x2, x1)`. Inv-frequencies can be precomputed for default RoPE up to a max context; dynamic RoPE variants can update buffers through shared Transformers RoPE utilities and should be rejected initially unless a config sweep proves they are required.

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids[B,S]` int token IDs, or mutually exclusive `inputs_embeds[B,S,H]`.
- Optional `attention_mask[B,T]` where `T` includes seen cache tokens plus current query length; 4D prebuilt masks are accepted by shared mask utilities.
- Optional `position_ids[B,S]`; otherwise source creates a single row offset by cache length.
- Optional `past_key_values` cache and `use_cache`.

CPU/data-pipeline work:

- Tokenization uses GPT-NeoX tokenizer family for StableLM 3B per docs/auto mapping.
- Chat formatting for Zephyr/instruct variants belongs to tokenizer/generation controller, not the neural graph.

GPU/runtime work:

- Embedding lookup, causal mask handling or backend causal metadata, RoPE, decoder blocks, LM head.
- No multimodal placeholder scatter, no image/audio processor tensors, no NHWC/NCHW axis translation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Split Q/K/V projections into grouped GEMM batch

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
three independent GEMMs scheduled together, or packed QKV GEMM when weights are repacked offline
```

Preconditions:

- Same input `x_norm[B,S,H]`.
- Bias handling matches `use_qkv_bias`.
- Packed weight output order must be exactly `[Q all rows, K all rows, V all rows]` if using one fused GEMM; StableLM source stores separate modules, so packing is a DinoML load-time transform.

Shape equations:

- Q rows `heads*D`.
- K/V rows `kv_heads*D`.
- Packed output width `(heads + 2*kv_heads) * D`.

Failure cases:

- Per-head Q/K layernorm cannot be moved before Q/K projection.
- Weight tying or external quantized formats must preserve original parameter provenance.

Parity sketch:

- Compare separate PyTorch projections against packed GEMM split outputs for MHA and GQA configs, with and without QKV bias.

### Rewrite: Partial RoPE + attention prefill fusion

Source pattern:

```text
split q/k -> RoPE on prefix -> concat -> attention
```

Replacement:

```text
attention backend that applies partial RoPE while loading Q/K tiles, or a pre-attention fused RoPE kernel
```

Preconditions:

- Default RoPE or implemented dynamic RoPE variant.
- `rotary_ndims` is even and known for the compiled profile.
- Cache stores post-RoPE keys, matching Transformers update order.

Failure cases:

- Unsupported RoPE scaling types.
- Attention backend that expects full-head RoPE.

Parity sketch:

- Random Q/K tests over `D=80, rotary_ndims=20` and `D=64, rotary_ndims=16`; compare cached decode logits.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
two GEMMs -> fused silu/mul -> down GEMM, or fused CUTLASS epilogue for gate/up when provider supports it
```

Preconditions:

- Activation is `silu`.
- Gate and up projections share input and have same output width.

Failure cases:

- Different activation in future configs; reject or fall back.

Parity sketch:

- Single MLP parity over BF16/FP16 accumulation tolerances for 3B and 1.6B dimensions.

### Rewrite: Last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
for decode, materialize only `[B,1,H] -> [B,1,V]`; for prefill, honor `logits_to_keep`
```

Preconditions:

- `logits_to_keep` is int or validated tensor indices.
- Generation only needs the final token logits.

Failure cases:

- Full sequence logits requested for loss/evaluation.

Parity sketch:

- Compare full LM-head slice against pre-sliced hidden-state GEMM.

### Layout notes: sequence/head layout only

There is no NHWC/NCHW image region. Candidate layout rewrites are guarded transformations among `[B,S,H]`, `[B,S,heads,D]`, and `[B,heads,S,D]`. Protect any attention backend boundary with a no-layout-translation guard unless all producers/consumers agree on head-major layout and cache layout.

Axis-sensitive operations:

- `transpose(1, 2)` after Q/K/V view.
- `softmax(dim=-1)` over key length.
- `split/concat(dim=-1)` for partial RoPE.
- `repeat_kv(..., dim=1)` over KV heads.
- Cache append along `dim=-2`.
- Sequence classification rightmost-token pooling over sequence axis.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm and optional per-head LayerNorm: StableLM uses LayerNorm rather than RMSNorm, so existing LLaMA-style RMSNorm coverage is not enough.
- GQA/MHA causal attention with partial RoPE and KV cache: this is the decode bottleneck and the most parity-sensitive path.
- Q/K/V projection scheduling plus RoPE: repeated every layer and shape-stable.
- SwiGLU MLP fused elementwise: large intermediate widths make the activation multiply bandwidth-visible.
- Last-token-only LM head: avoids `[B,S,V]` work during decode.

Medium priority:

- Bias/no-bias GEMM variants for Q/K/V based on config.
- Parallel residual fused add path for StableLM 2-style configs.
- Mask creation/elision for backend causal attention: avoid materializing dense 4D masks when backend can encode causality and padding.
- GGUF dense fallback and bounded runtime dequant for StableLM GGUF mirrors.

Lower priority:

- Sequence classification pooling/indexing.
- Token classification dropout/head.
- Training-only dropout/loss paths.
- Exotic RoPE variants not present in representative production configs.

## 11. Runtime staging plan

Stage 1: Parse `StableLmConfig`, normalize legacy RoPE fields into `rope_parameters`, load dense weights, and reject unsupported config flags such as sliding/local attention.

Stage 2: Single-block parity without cache for MHA no-bias 3B dimensions. Include LayerNorm, QKV projections, partial RoPE, eager causal attention, O projection, serial residual, and SwiGLU.

Stage 3: Full prefill parity for `StableLmForCausalLM`, initially using dense attention and full logits.

Stage 4: Decode with `DynamicCache` equivalent: post-RoPE K cache and V cache per layer, stored as `[B,kv_heads,T,D]`.

Stage 5: Add GQA fixture parity, QKV bias, per-head Q/K LayerNorm, and parallel residual using `tiny-random-stablelm-2` and `stablelm-2-1_6b` configs.

Stage 6: Optimized attention and fusion: native GQA FlashAttention-style prefill/decode, fused partial RoPE, SwiGLU fusion, last-token-only logits, and guarded mask elision.

Stage 7: Weight-provider staging: GGUF load/dequant admission for mirrors with incomplete `config.json`, preserving dense fallback and explicit provenance.

Optional later stage: sequence/token classification heads.

## 12. Parity and validation plan

- Config parser tests: defaults plus four representative configs; verify effective `head_dim`, `rotary_ndims`, bias flags, GQA groups, and residual path.
- Custom op tests: `rotate_half`, partial RoPE, per-head LayerNorm, `repeat_kv`, and cache append shapes.
- Single-layer parity: random hidden states for 3B no-bias MHA, 1.6B bias MHA, and tiny GQA+QK-LN+parallel-residual.
- Full-model prefill parity: compare logits for `stabilityai/tiny-random-stablelm-2` and, when weights are available, the integration-test prompt for `stablelm-3b-4e1t`.
- Decode parity: prefill N tokens, decode one token with cache, compare against full recompute logits at the same position.
- Last-token LM-head parity: compare full logits slice versus `logits_to_keep=1`.
- GGUF parity: if using GGUF mirrors, compare decoded dense weights or runtime dequant output against an original dense checkpoint or Transformers GGUF load path.

Suggested tolerances:

- FP32 custom math: `rtol=1e-5`, `atol=1e-6`.
- FP16/BF16 block parity: start with `rtol=2e-2`, `atol=2e-2` for full logits, tighten per op where accumulation policy is controlled.
- Cache decode logits should be compared against full recompute with identical position IDs and mask semantics.

## 13. Performance probes

- Prefill throughput by `(B,S)` for 3B dimensions: `S=128,512,2048,4096`.
- Decode tokens/sec by batch and cache length: `B=1,4,16`, `T=128..4096`.
- Attention backend comparison: eager dense, SDPA-style, FlashAttention-style with native GQA.
- KV cache memory footprint: MHA 3B versus GQA tiny fixture; report bytes per token per layer.
- QKV+RoPE pipeline timing: projection, RoPE, cache update, attention separately.
- MLP timing: gate/up GEMMs, SwiGLU elementwise, down GEMM.
- LM head timing: full sequence logits versus last-token-only.
- GGUF provider probe: load-time full dequant versus runtime pre-GEMM dequant for projection/MLP matrices.
- Batch-size sweep with padding masks to validate mask-elision and padding behavior.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, and gradient checkpointing.
- Beam search and advanced generation controllers beyond cache reorder/select semantics.
- Sequence and token classification heads for first causal-LM integration.
- Quantization schemes other than explicit GGUF dense fallback/runtime-dequant experiments.
- Multi-GPU tensor parallel and continuous batching.
- Unsupported RoPE scaling variants until representative configs require them.
- Sliding-window/local/block-sparse attention, because current StableLM source does not implement it.

## 15. Final implementation checklist

- [ ] Parse and validate StableLM config, including legacy RoPE field normalization.
- [ ] Load dense weights with correct bias/no-bias projection contracts.
- [ ] Implement/tokenize or accept `input_ids`, `attention_mask`, `position_ids`, and `inputs_embeds` ABI.
- [ ] Implement embedding lookup and final untied LM head.
- [ ] Implement LayerNorm and optional per-head Q/K LayerNorm.
- [ ] Implement Q/K/V/O GEMM shapes for MHA and GQA.
- [ ] Implement partial RoPE with pass-through head channels.
- [ ] Implement causal attention with fp32 softmax semantics and additive mask parity.
- [ ] Implement KV cache update/read as `[B,kv_heads,T,D]`.
- [ ] Implement serial and parallel residual decoder block variants.
- [ ] Implement SwiGLU MLP and fusion tests.
- [ ] Add full prefill logits parity tests.
- [ ] Add decode-with-cache parity tests.
- [ ] Add last-token-only logits lowering.
- [ ] Add optimized attention/fusion path with fallback guards.
- [ ] Add GGUF admission/loading plan for StableLM mirrors with incomplete configs.
