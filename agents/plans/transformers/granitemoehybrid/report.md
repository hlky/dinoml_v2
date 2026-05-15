# GraniteMoeHybrid Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: granitemoehybrid
Primary runtime target: causal language model generation
Transformers source root: transformers
Config snapshots: agents/plans/transformers/granitemoehybrid/_sources/
```

Source files inspected:

- `src/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py`
- `src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py`
- `src/transformers/models/granitemoehybrid/modular_granitemoehybrid.py`
- `src/transformers/cache_utils.py`
- Contrast files: `models/granite`, `models/granitemoe`, `models/granitemoeshared`, `models/mamba2`
- Tests: `tests/models/granitemoehybrid/test_modeling_granitemoehybrid.py`

Pinned source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/granitemoehybrid/modular_granitemoehybrid.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py`

Representative HF configs inspected:

- `https://huggingface.co/tiny-random/granite-4.0-h/resolve/main/config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-h-micro-base/resolve/main/config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-h-small/resolve/main/config.json`
- `https://huggingface.co/ibm-granite/granite-4.0-h-1b/resolve/main/config.json`

The shipped `modeling_granitemoehybrid.py` is generated from `modular_granitemoehybrid.py`; future HF source edits should be made in the modular file, but DinoML should match the generated runtime file.

Any missing files or assumptions: no remote-code files were needed; the report scopes native Transformers source only. Safetensor index metadata was not downloaded because operator structure and dimensions were available from config/source.

## 2. High-level architecture

GraniteMoeHybrid is a text-only causal decoder with per-layer selection between Mamba2-style state-space blocks and causal self-attention blocks. Each decoder layer also has a shared gated MLP, and may additionally have sparse MoE experts.

Dataflow:

```text
GPT2 tokenizer/input_ids -> token embedding * embedding_multiplier
-> repeated hybrid decoder layers:
   RMSNorm -> Mamba state-space OR causal attention -> residual * residual_multiplier
   RMSNorm -> optional MoE + shared SwiGLU MLP -> residual * residual_multiplier
-> final RMSNorm -> tied/untied LM head -> logits / logits_scaling
```

Generation has a mixed cache ABI:

```text
attention layers: growing KV cache [B, kv_heads, cached_seq, head_dim]
mamba layers: fixed conv state [B, conv_dim, d_conv] + recurrent state [B, mamba_heads, mamba_head_dim, d_state]
```

Stages that can be validated independently:

- Config/tokenizer loading and token IDs.
- One Mamba layer in prefill mode, with and without cache initialization.
- One Mamba layer in single-token decode mode from prefilled conv/recurrent states.
- One attention layer with GQA KV cache.
- MoE router and grouped expert matmuls.
- Full prefill logits, then decode token parity with mixed cache state.

## 3. Important config dimensions

Source defaults from `GraniteMoeHybridConfig`:

| Field | Default |
|---|---:|
| `vocab_size` | 32000 |
| `hidden_size` | 4096 |
| `intermediate_size` | 11008 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | defaults to `num_attention_heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 2048 |
| `position_embedding_type` | `None` |
| `rope_parameters` | `None` |
| `attention_bias` / `mamba_proj_bias` | `False` / `False` |
| `mamba_conv_bias` | `True` |
| `num_local_experts` / `num_experts_per_tok` | 8 / 2 |
| `mamba_n_heads`, `mamba_n_groups`, `mamba_d_state` | 128, 1, 256 |
| `mamba_d_head` | `"auto" = mamba_expand * hidden_size // mamba_n_heads` |
| `mamba_d_conv`, `mamba_expand`, `mamba_chunk_size` | 4, 2, 256 |
| `use_cache` | `True` |

Representative checkpoint sweep from `config.json`:

| Model | Layers | Layer pattern | Hidden | Attn heads/KV | Attn head dim | Mamba heads/head/state | Experts/top-k | Shared MLP | Max pos | Dtype |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| tiny-random/granite-4.0-h | 2 | 1 mamba, 1 attention | 32 | 2/2 | 16 | 128/64/128 | 72/10 | 128 | 131072 | omitted |
| granite-4.0-h-350m | 32 | 28 mamba, 4 attention | 768 | 12/4 | 64 | 48/32/128 | 0/0 | 2048 | 32768 | bf16 |
| granite-4.0-h-micro-base | 40 | 36 mamba, 4 attention | 2048 | 32/8 | 64 | 64/64/128 | 0/0 | 8192 | 131072 | bf16 |
| granite-4.0-h-small | 40 | 36 mamba, 4 attention | 4096 | 32/8 | 128 | 128/64/128 | 72/10 | 1536 | 131072 | bf16 |
| granite-4.0-h-1b | 40 | 36 mamba, 4 attention | 1536 | 12/4 | 128 | 48/64/128 | 0/0 | 4096 | 131072 | bf16 |

Attention layer indices in sampled configs:

- 350m: `[10, 13, 17, 27]`
- micro-base/small/1b: `[5, 15, 25, 35]`
- tiny-random: `[1]`

Sampled configs set `position_embedding_type` to `"nope"` and omit `rope_parameters`, so RoPE is inactive for these checkpoints even though the source implements it. They also omit `time_step_min`, `time_step_max`, and `time_step_limit`; source defaults apply as `0.001`, `0.1`, and `(0.0, inf)`.

## 3a. Family variation traps

- `layer_types` is architecture-defining. DinoML must build a cache manifest per layer, not assume all layers are attention or all layers are Mamba.
- `layers_block_type` is an alias for `layer_types`; source reads `config.layers_block_type[i]`.
- `position_embedding_type == "rope"` is the only source path that creates rotary embeddings. The inspected IBM Granite 4.0 hybrid configs use `"nope"`, so no RoPE or position-dependent attention transform is required for that subset.
- GQA is common: sampled production configs have `num_key_value_heads < num_attention_heads`.
- Mamba dimensions are independent of attention dimensions. `mamba_expand * hidden_size == mamba_n_heads * mamba_d_head` is validated by config.
- `hidden_size != num_attention_heads * source default head_dim` can happen if `head_dim` is explicitly added in future configs; source computes attention head dim with `getattr(config, "head_dim", hidden_size // num_attention_heads)`.
- Dense variants exist: `num_local_experts == 0` disables sparse MoE and uses only `shared_mlp`.
- MoE variants can have many experts and large top-k; `granite-4.0-h-small` uses 72 experts and top-k 10.
- Shared MLP is always constructed in this source, unlike `granitemoeshared`, where `shared_intermediate_size == 0` can disable it.
- Embeddings and LM head are tied in inspected configs (`tie_word_embeddings: true`), although source declares an untied linear head and relies on HF tying.
- Attention scaling is `config.attention_multiplier`, not the usual `1 / sqrt(head_dim)`. Sampled configs use small explicit constants such as `0.0078125` or `0.015625`.
- `embedding_multiplier`, `residual_multiplier`, and `logits_scaling` are checkpoint-significant numeric transforms.
- Fast Mamba kernels are optional runtime dependencies from `mamba-ssm` and `causal-conv1d`. Source falls back to a PyTorch implementation, but that path is expensive.
- Padding masks are handled differently for Mamba and attention. Mamba masks zero hidden states for pads, but cached decode or all-ones masks skip this zeroing.
- `seq_idx`/padding-free support requires the fast Mamba path; slow path raises `NotImplementedError` if `seq_idx` is supplied.
- Layout is ordinary PyTorch row-major token layout. No NCHW/NHWC region exists; layout-sensitive axes are token axis `dim=1`, hidden axis `dim=-1`, attention heads after `transpose(1, 2)`, MoE token flatten axis `B*S`, and Mamba conv channel axis after `transpose(1, 2)`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- Elementwise multiply/divide by scalar config values.
- Reshape/view/transpose/contiguous for attention heads and Mamba conv.
- Split/chunk along hidden axis.
- Pad on sequence/conv time axis.
- Cumsum, tril masks, masked fill, repeat/repeat_interleave, expand.
- TopK, sort, scatter, one-hot, index gather, index_add for MoE routing.
- Dynamic list/split by per-expert token counts, or a fused MoE replacement that avoids Python lists.

Neural primitives:

- Biasless/bias Linear for projections.
- RMSNorm with fp32 variance.
- Gated RMSNorm: `RMSNorm(silu(gate) * hidden)`.
- SiLU/SwiGLU: `act(a) * b`.
- Depthwise causal Conv1d with `groups == conv_dim`, `kernel_size == mamba_d_conv`, `padding == d_conv - 1`, then truncate to original sequence length.
- BMM for single-token recurrent output projection by `C`.

Attention primitives:

- Causal self-attention only.
- GQA/MQA repeat from KV heads to query heads.
- Optional RoPE before cache update when enabled.
- Additive causal mask from `create_causal_mask`.
- Softmax in fp32 then cast to query dtype on eager path.
- SDPA/Flash/Flex dispatch through `ALL_ATTENTION_FUNCTIONS`.

Mamba/state-space primitives:

- In-projection `Linear(H -> mamba_expand*H + conv_dim + mamba_n_heads)`.
- `A = -exp(A_log.float())`.
- `dt = softplus(dt + dt_bias)`, then clamp to `time_step_limit`.
- Chunked SSD scan for prefill, with `chunk_size = mamba_chunk_size`.
- Single-token state update:
  `state = state * exp(dt * A) + dt * B * x`.
- State-to-output with `C`, plus `D` skip, gated RMSNorm, output projection.
- Static-address in-place cache updates for conv and recurrent states.

MoE primitives:

- Router `Linear(H -> E, bias=False)` in fp32 logits.
- TopK over experts, softmax over selected logits.
- Expert weights stored `[num_experts, out_features, in_features]`.
- Expert input grouped by sorted expert id.
- Per-expert input linear `H -> 2*intermediate_size`, SwiGLU, output linear `intermediate_size -> H`.
- Gate multiply and `index_add` back to original flattened token index.

Generation/cache ops:

- Mixed `DynamicCache(config)` construction from `layer_types`.
- Attention `Cache.update(k, v, layer_idx)` with KV growth.
- Mamba `update_conv_state` and `update_recurrent_state`.
- `get_seq_length()` must find the first attention layer when layer 0 is Mamba.
- Beam reorder indexes batch dimension for KV, conv state, and recurrent state.
- Reset zeroes fixed states and resets `has_previous_state`.

Preprocessing-coupled ops:

- Auto tokenizer mapping uses GPT2Tokenizer.
- Generation configs inspected use `bos_token_id = eos_token_id = 100257`, `pad_token_id = 100256`.
- No image/audio processor path.

## 5. Layer/block breakdown

Attention layer:

```text
x0 = hidden_states                         [B, S, H]
x = RMSNorm(x0)
q = Linear(H -> n_heads*head_dim, no bias) [B, S, n_heads*D] -> [B, n_heads, S, D]
k = Linear(H -> kv_heads*head_dim)         [B, kv_heads, S, D]
v = Linear(H -> kv_heads*head_dim)         [B, kv_heads, S, D]
if RoPE enabled: q,k = RoPE(q,k,cos,sin)
if cache: k,v = cache.update(k,v,layer_idx)
attn = causal_attention(q,k,v, scale=attention_multiplier)
x = Linear(n_heads*D -> H)(attn)
x = x0 + x * residual_multiplier
y0 = x
y = RMSNorm(y0)
if num_local_experts > 0:
    y = MoE(y) + shared_SwiGLU_MLP(y)
else:
    y = shared_SwiGLU_MLP(y)
out = y0 + y * residual_multiplier
```

Mamba layer:

```text
x0 = hidden_states                         [B, S, H]
x = RMSNorm(x0)
p = in_proj(x)                             [B, S, intermediate + conv_dim + mamba_heads]
gate, xBC, dt = split(p)
xBC -> depthwise causal conv over token axis
split xBC into x, B, C
prefill: chunked SSD scan, save final conv/recurrent states
decode: roll/update conv state, update recurrent state in place
x = gated RMSNorm(scan_or_step_output, gate)
x = out_proj(x)                            [B, S, H]
x = x0 + x * residual_multiplier
then same post block RMSNorm + MoE/shared MLP + residual
```

For `granite-4.0-h-small`, concrete major shapes:

- Attention Q: `Linear(4096 -> 4096)`, K/V: `Linear(4096 -> 1024)`, O: `Linear(4096 -> 4096)`, no bias.
- Mamba `intermediate_size = 8192`, `conv_dim = 8192 + 2*1*128 = 8448`.
- Mamba in-proj: `Linear(4096 -> 8192 + 8448 + 128 = 16768)`, no bias.
- Mamba conv: depthwise `Conv1d(8448 -> 8448, kernel=4, groups=8448, bias=True)`.
- Mamba out-proj: `Linear(8192 -> 4096)`.
- Sparse MoE expert input: per expert `Linear(4096 -> 1536)` because config `intermediate_size=768` and input expert projection emits `2*intermediate`.
- Shared MLP: `Linear(4096 -> 3072) -> SwiGLU -> Linear(1536 -> 4096)`.

## 6. Attention requirements

Attention is causal self-attention, not cross-attention. The source supports eager, SDPA, FlashAttention, and Flex attention through Transformers backend dispatch. Only layers with `layer_type != "mamba"` instantiate attention.

Requirements:

- Query shape before attention: `[B, num_attention_heads, S_q, head_dim]`.
- KV shape before GQA repeat: `[B, num_key_value_heads, S_kv, head_dim]`.
- KV cache stores keys after RoPE when RoPE is active.
- Eager path repeats KV to `[B, num_attention_heads, S_kv, head_dim]`.
- Attention scores are `q @ k^T * attention_multiplier`, then additive mask, then fp32 softmax, dropout during training only, then `weights @ v`.
- There is no sliding-window/local attention in this source.
- `position_ids` default to `arange(S) + past_key_values.get_seq_length()`. With all-Mamba configs, `get_seq_length()` would fail because no attention layer tracks length; sampled configs have attention layers.

Cache shapes for production sampled configs:

| Model | Attention KV cache per attention layer |
|---|---|
| 350m | `[B, 4, cached_seq, 64]` keys and values |
| micro-base | `[B, 8, cached_seq, 64]` keys and values |
| small | `[B, 8, cached_seq, 128]` keys and values |
| 1b | `[B, 4, cached_seq, 128]` keys and values |

## 7. Position encoding and custom math

RoPE is conditional. `GraniteMoeHybridModel.__init__` constructs `rotary_emb` only if `position_embedding_type == "rope"`. Inspected IBM Granite 4.0 hybrid configs use `"nope"`, so the sampled production path has no rotary math.

When enabled, RoPE uses `rope_parameters["rope_theta"]`, computes inv frequencies in fp32, and returns cos/sin cast to the hidden dtype. It also supports non-default rope types through `ROPE_INIT_FUNCTIONS`.

Relevant custom math:

```python
def granite_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

Mamba single-token decode math:

```python
dt = softplus(dt + dt_bias)
dt = clamp(dt, time_step_limit[0], time_step_limit[1])
A = -exp(A_log.float())
dA = exp(dt[..., None] * A[None, :, :, :])
dB = dt[..., None] * B[..., None, :]
state = state * dA + dB * x[..., None]
y = matmul(state, C[..., None]).squeeze(-1) + x * D
y = RMSNorm(silu(gate) * y)
```

Prefill uses chunked SSD scan with triangular/segment-sum structure. This is not equivalent to dense attention, even though the slow code forms attention-like chunk matrices.

## 8. Preprocessing and input packing

Runtime input contract is text-only:

- `input_ids`: `[B, S]` int token IDs.
- Optional `attention_mask`: `[B, S]`, used as an additive causal mask for attention and as a multiplicative padding mask for Mamba hidden states.
- Optional `position_ids`: `[B, S]`; only matters when RoPE is active.
- Optional `inputs_embeds`: `[B, S, H]`, mutually exclusive with `input_ids`.

Tokenizer/config coupling:

- Auto tokenizer maps `granitemoehybrid` to GPT2Tokenizer.
- Downloaded tokenizer configs use a very large `model_max_length` sentinel, so the model config `max_position_embeddings` is the more meaningful runtime admission limit.
- Generation configs use pad token 100256 and BOS/EOS 100257 for sampled checkpoints.

Padding-free packed training/inference metadata is exposed through `GraniteFlashAttentionKwargs`: `cu_seq_lens_q`, `cu_seq_lens_k`, `max_length_q`, `max_length_k`, and `seq_idx`. For DinoML first integration this should be rejected or routed to a specialized fast-Mamba path because the slow path raises when `seq_idx` is provided.

## 9. Graph rewrite / lowering opportunities

### Rewrite: attention projection canonicalization

Source pattern:

```text
Linear -> view(*, heads, head_dim) -> transpose(1,2)
```

Replacement:

```text
GEMM_RCR/Linear -> HeadReshapeTranspose metadata -> fused attention
```

Preconditions:

- Dense contiguous hidden states `[B, S, H]`.
- `out_features == heads * head_dim`.
- Bias flag matches config.

Failure cases:

- Future configs with non-divisible head dimensions must be rejected before lowering.
- If RoPE is active, fused Q/K path must preserve RoPE before KV cache write.

Parity test sketch: compare Q/K/V tensors and pre-softmax scores for one attention layer.

### Rewrite: depthwise causal Conv1d decode update

Source pattern:

```text
roll conv_state -> insert new xBC -> sum(conv_state * weight, dim=-1) + bias -> activation
```

Replacement:

```text
Static ring-buffer update + depthwise dot over d_conv
```

Preconditions:

- `groups == conv_dim`, `kernel_size == d_conv`, `padding == d_conv - 1`.
- Decode `seq_len == 1`.
- Cache state already initialized with shape `[B, conv_dim, d_conv]`.

Failure cases:

- Multi-token decode must use prefill/update path.
- Non-SiLU activation can use generic activation; fast kernel assumptions only cover silu/swish.

### Rewrite: Mamba prefill fused scan

Source pattern:

```text
in_proj -> split -> causal depthwise conv -> split x/B/C/dt -> chunked SSD scan -> gated RMSNorm -> out_proj
```

Replacement:

```text
Mamba2 fused prefill kernel with final recurrent state output
```

Preconditions:

- `mamba_n_groups` divides `mamba_n_heads`.
- `mamba_n_heads * mamba_d_head == mamba_expand * hidden_size`.
- `time_step_limit` supported.
- The kernel returns both hidden output and final recurrent state for cache.

Failure cases:

- Packed `seq_idx` requires explicit support.
- Masks with padding require either source-equivalent zeroing before/after conv or fallback.

### Rewrite: MoE grouped expert GEMM

Source pattern:

```text
router -> topk -> sort by expert -> per-expert F.linear loops -> gate multiply -> index_add
```

Replacement:

```text
TopKRouter + token permutation + grouped GEMM(input experts) + SwiGLU + grouped GEMM(output experts) + scatter-add
```

Preconditions:

- Expert weights use `[E, out, in]`.
- Router top-k fixed by config.
- No training aux loss required for inference.

Failure cases:

- `expert_size.tolist()` is data-dependent; compiled graph needs dynamic grouped GEMM or a padded capacity plan.
- Top-k can be high (`10`) and experts can be many (`72`), so naive per-expert loops will be slow.

### Rewrite: tied LM head alias

Source pattern:

```text
embed_tokens.weight and lm_head.weight tied by HF when tie_word_embeddings=true
```

Replacement:

```text
One logical constant used by embedding and final projection
```

Preconditions:

- Config `tie_word_embeddings == true`.
- Weight loader confirms alias or equal storage.

Failure cases:

- Untied checkpoints must keep two constants.

## 10. Kernel fusion candidates

Highest priority:

- Mamba prefill fused conv+scan+state-write. This dominates most layers in sampled configs.
- Mamba decode fused conv-state update + selective state update + gated RMSNorm.
- MoE top-k routing + grouped expert GEMM + scatter-add for `granite-4.0-h-small`.
- GQA attention prefill/decode with KV cache for the four attention layers.
- RMSNorm and gated RMSNorm in fp32 accumulation.

Medium priority:

- Linear + split for Mamba `in_proj`.
- SwiGLU shared MLP and expert MLP epilogues.
- Last-token-only LM head for decode/prefill with `logits_to_keep`.
- Scalar multiplier folds: embedding, residual, logits scaling, attention multiplier.

Lower priority:

- RoPE fusion, because sampled production configs disable it.
- Training aux router loss.
- Flex attention and padding-free `seq_idx` metadata.
- CPU slow Mamba parity beyond debugging.

## 11. Runtime staging plan

Stage 1: parse config and build a per-layer manifest with `mamba` versus `attention`, dimensions, multipliers, tied embedding/head status, and MoE enabled/disabled.

Stage 2: load weights and run embedding, RMSNorm, shared MLP, and LM head parity.

Stage 3: implement attention layer parity with GQA cache for the attention-only subset.

Stage 4: implement one Mamba layer prefill in slow/reference form, including final conv/recurrent state materialization.

Stage 5: implement Mamba decode state update with fixed state buffers and static-address mutation semantics.

Stage 6: integrate full mixed-layer prefill and decode.

Stage 7: add MoE grouped expert lowering for sparse variants; dense `num_local_experts == 0` checkpoints can start earlier.

Stage 8: replace slow Mamba and MoE paths with fused kernels and add continuous batching/cache scheduling.

Initial stubs:

- Reject `seq_idx`/padding-free mode.
- Reject RoPE-enabled configs until the attention path is proven, or support default RoPE only.
- Skip router aux loss and labels for inference.
- Route `num_local_experts > 0` to a slow reference or reject until grouped MoE is ready.

## 12. Parity and validation plan

Recommended tests:

- Config parser test for all sampled configs, including attention indices and cache layer classes.
- Random tensor parity for RMSNorm and gated RMSNorm with fp32 accumulation.
- Random tensor parity for Mamba conv-state update over `seq_len == 1`.
- Random tensor parity for Mamba recurrent update and state-to-output BMM.
- One Mamba layer prefill parity for short `S < chunk_size`, exact `S == chunk_size`, and uneven `S`.
- One attention layer prefill and decode parity with GQA cache.
- MoE router parity: top-k indices, sorted grouping, expert sizes, gate values, scatter-add output.
- Full model prefill logits parity on tiny-random and one IBM checkpoint.
- Decode parity for 1, 2, and 8 generated tokens after a prefill.

Tolerances:

- fp32 reference: `rtol=1e-4`, `atol=1e-5` for primitive tests.
- bf16/fp16 model path: `rtol=1e-2`, `atol=1e-2` for logits and hidden states, with tighter checks for integer router indices.

No DinoML tests were run for this docs-only audit.

## 13. Performance probes

- Mamba prefill throughput by `(B, S)` and `chunk_size`.
- Mamba decode tokens/sec with fixed state buffers.
- Attention prefill and decode throughput for only attention layers.
- KV cache memory versus Mamba state memory for 32k and 131k context settings.
- MoE router time, token permutation time, grouped GEMM time, scatter-add time.
- Dense variants (`num_local_experts=0`) versus sparse variants (`72` experts, top-k `10`).
- Last-token-only logits versus full sequence logits.
- Fast Mamba kernel parity/perf against source fallback.
- Batch-size sweep for conv/recurrent state update.
- Sequence-length sweep around chunk boundaries: 255, 256, 257, 512, 1024.

## 14. Skip/defer list

Safe to defer for first causal inference integration:

- Training, labels, gradient checkpointing, router auxiliary loss.
- Beam search, except cache `reorder_cache` should be documented before claiming beam support.
- Quantization and compressed-tensors variants.
- Tensor parallel plans.
- Flex attention and padding-free `seq_idx` path.
- RoPE variants if first target is IBM Granite 4.0 hybrid configs with `"nope"`.
- CPU high-performance Mamba kernels.
- Continuous batching.

Do not defer for production Granite 4.0 hybrid:

- Mixed cache ABI.
- Mamba prefill/decode states.
- GQA attention for attention layers.
- Config multipliers and logits scaling.
- MoE if targeting `granite-4.0-h-small`.

## 15. Final implementation checklist

- [ ] Parse `GraniteMoeHybridConfig`, including `layer_types`/`layers_block_type`.
- [ ] Reject invalid layer type lists and invalid Mamba dimension equations.
- [ ] Load tied embeddings/LM head as one logical constant when configured.
- [ ] Implement embedding multiplier and logits scaling.
- [ ] Implement RMSNorm and gated RMSNorm with fp32 accumulation.
- [ ] Implement shared SwiGLU MLP.
- [ ] Implement GQA causal attention with `attention_multiplier`.
- [ ] Implement optional default RoPE before KV cache update, or reject RoPE configs initially.
- [ ] Build mixed cache manifest: attention KV, Mamba conv state, Mamba recurrent state.
- [ ] Implement Mamba prefill scan and state writes.
- [ ] Implement Mamba single-token decode state update.
- [ ] Implement Mamba padding-mask zeroing semantics.
- [ ] Implement MoE router top-k, grouping, grouped expert GEMMs, and scatter-add.
- [ ] Add one-layer Mamba parity tests.
- [ ] Add one-layer attention parity tests.
- [ ] Add MoE routing parity tests.
- [ ] Add full prefill logits parity.
- [ ] Add decode token parity with cache reorder/reset coverage.
- [ ] Benchmark Mamba prefill/decode, attention, MoE, and LM head separately.
