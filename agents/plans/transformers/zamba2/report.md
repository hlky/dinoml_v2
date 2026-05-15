# Transformers Audit: zamba2

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in local checkout `transformers`.

Model id: primary source examples point at `Zyphra/Zamba2-2.7B` and `Zyphra/Zamba2-7B-v1`; representative public configs inspected for `Zyphra/Zamba2-1.2B`, `Zyphra/Zamba2-2.7B`, `Zyphra/Zamba2-2.7B-instruct`, and `Zyphra/Zamba2-7B-Instruct`.

Config source:

- `transformers/src/transformers/models/zamba2/configuration_zamba2.py`
- Public HF config URLs listed in `evidence_configs.md`.

Source files inspected:

- `transformers/src/transformers/models/zamba2/modeling_zamba2.py`
- `transformers/src/transformers/models/zamba2/modular_zamba2.py`
- `transformers/src/transformers/models/zamba2/configuration_zamba2.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/tests/models/zamba2/test_modeling_zamba2.py`

Any missing files or assumptions:

- `modeling_zamba2.py` is generated from `modular_zamba2.py`; future upstream source edits should inspect `modular_zamba2.py` first, while runtime behavior is visible in the generated file.
- `Zyphra/Zamba2-7B` and `Zyphra/Zamba2-7B-v1` public `config.json` fetches returned 401. The 7B config variation in this report uses `Zyphra/Zamba2-7B-Instruct`.
- The source optionally imports external hub kernels `causal-conv1d` and `mamba-ssm`; DinoML should treat those as provider requirements or use the explicit torch fallback for parity.

## 2. High-level architecture

Zamba2 is a text-only causal decoder for next-token generation. It is a hybrid Mamba2 plus transformer decoder: most layers are Mamba2 state-space layers, and selected `hybrid` layers run a shared transformer attention/MLP block, project it through a linear bridge, then inject that result into the following Mamba layer.

Dataflow:

```text
tokenizer/input_ids -> token embedding -> repeated mamba or hybrid blocks -> final RMSNorm -> LM head -> logits/sampling
```

Primary DinoML target: `Zamba2ForCausalLM` prefill and decode. `Zamba2Model` feature extraction is useful for block parity. `Zamba2ForSequenceClassification` is optional/deferred; it reuses the base model and adds last-non-pad-token pooling plus one linear score head.

Stage decomposition:

- CPU/data pipeline: Llama/Mistral-style tokenizer, optional ChatML-style prompt templating for instruct checkpoints.
- GPU/runtime prefill: token embedding, full-sequence Mamba2 scan, sparse set of full causal attention blocks, final norm, optional last-token-only logits.
- GPU/runtime decode: one-token Mamba recurrent update plus convolution state update on every layer; hybrid layers also update attention KV cache.
- Independently optimizable regions: Mamba2 mixer, hybrid attention block, SwiGLU/GELU-gated MLP, final LM projection, last-token logits.

## 3. Important config dimensions

Source defaults from `Zamba2Config`:

| Field | Default / derived value | Runtime meaning |
| --- | --- | --- |
| `vocab_size` | 32000 | Embedding and LM head width |
| `hidden_size` | 2560 | Main residual width `H` |
| `attention_hidden_size` | `2 * hidden_size` | Hybrid attention input after concat with original embeddings |
| `num_hidden_layers` | 54 | Entries in `layers_block_type` |
| `layers_block_type` | 45 `mamba`, 9 `hybrid` for default | Per-layer cache/operator type |
| `num_attention_heads` | 32 | Query heads in hybrid attention |
| `num_key_value_heads` | defaults to `num_attention_heads` | MHA by default, no GQA unless config changes |
| `attention_head_dim` | `2 * hidden_size // num_attention_heads` | Q/K/V head dimension for hybrid attention |
| `intermediate_size` | `4 * hidden_size` unless set | MLP gated width |
| `mamba_expand` | 2 | Mamba intermediate width `I = 2H` |
| `n_mamba_heads` | 8 default; checkpoints use 64/80/112 | Mamba SSM head count |
| `mamba_headdim` | `mamba_expand * H // n_mamba_heads` | Mamba per-head channel width |
| `mamba_d_state` | 64 default | SSM state width |
| `mamba_d_conv` | 4 | Depthwise causal conv kernel |
| `mamba_ngroups` | 1 default; 7B instruct uses 2 | B/C group count |
| `chunk_size` | 256 | Full-sequence scan chunking |
| `hidden_act` | `gelu` | MLP gate activation |
| `use_mem_rope` | false default | RoPE only in hybrid attention if true |
| `use_shared_attention_adapter` | false default | Optional LoRA-like adapter additions to q/k/v |
| `num_mem_blocks` | 1 default | Number of unique shared transformer blocks before weight tying repeats |
| `add_bias_linear` | false | Bias on Mamba in/out and MLP projections only if true |
| `use_cache` | true | Hybrid cache: attention KV plus Mamba conv/recurrent state |
| `num_logits_to_keep` | 1 | Generation only computes last prompt logits by default |

Representative checkpoint sweep:

| Checkpoint | Hidden | Layers | Mamba / hybrid | Attention Q heads / KV heads / head dim | Mamba heads / head dim / groups | MLP width | Notable config behavior |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| `Zyphra/Zamba2-1.2B` | 2048 | 38 | 32 / 6 | 32 / 32 / 128 | 64 / 64 / 1 | 8192 | `use_mem_rope=true`, shared attention adapters enabled, `num_mem_blocks=1`, `mamba_d_state=128` |
| `Zyphra/Zamba2-2.7B` | 2560 | 54 | 45 / 9 | 32 / 32 / 160 | 80 / 64 / 1 | 10240 | no mem RoPE, no adapters, `num_mem_blocks=2` |
| `Zyphra/Zamba2-2.7B-instruct` | 2560 | 54 | 45 / 9 | 32 / 32 / 160 | 80 / 64 / 1 | 10240 | same neural structure as 2.7B base; tokenizer has ChatML-style template |
| `Zyphra/Zamba2-7B-Instruct` | 3584 | 81 | 68 / 13 | 32 / 32 / 224 | 112 / 64 / 2 | 14336 | mem RoPE enabled, `mamba_ngroups=2`, 13 hybrid layers |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * attention_head_dim`; attention uses `attention_hidden_size = 2H`, so `attention_head_dim = 2H / heads`.
- Hybrid attention inputs concatenate current hidden states with the original token embeddings, then RMSNorm over width `2H`.
- `num_key_value_heads` defaults to `num_attention_heads`; source supports GQA/MQA through `repeat_kv`, but inspected public configs are MHA.
- `use_mem_rope` gates RoPE entirely. Some checkpoints use no RoPE in hybrid attention.
- RoPE config may be absent in `config.json`; `PreTrainedConfig` standardizes missing `rope_parameters` to default RoPE parameters when the config class has the field.
- Mamba state dimensions vary: 1.2B uses `mamba_d_state=128`, 2.7B and 7B instruct use 64; 7B instruct uses two B/C groups.
- `use_shared_attention_adapter` adds adapter projection lists to Q/K/V only; the MLP always builds adapter lists, selecting `Identity` for non-owned tied blocks.
- Weight tying is not only embedding/LM-head tying. Hybrid transformer blocks are tied according to `num_mem_blocks`; source tests skip some offload paths because mixed layers and tied weights can corrupt shapes.
- There is no MoE router or expert dispatch in `zamba2` source. The MLP is gated, not mixture-of-experts. Cache utilities mention `"moe"` as a linear-attention-shaped placeholder, but this family does not instantiate MoE modules.
- External Mamba kernels are optional. If unavailable, source falls back to a long eager torch implementation with chunked scan, triangular masks, `cumsum`, `exp`, `repeat_interleave`, and `bmm`.
- FlashAttention/SDPA/Flex attention are supported for hybrid attention, but tests call out no support for right padding plus `use_cache` with FlashAttention 2.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, T] -> [B, T, H]`.
- Clone of `inputs_embeds` to preserve `original_hidden_states`.
- Concatenate `[B, T, H] + [B, T, H] -> [B, T, 2H]` in each hybrid layer.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `split`, `chunk`, `unsqueeze`, `expand`, `repeat_interleave`, `pad`, `roll`, `index_select`, `argmax`, advanced gather for classification pooling.
- Causal mask creation from attention mask, position ids, and cache length.

Neural primitives:

- RMSNorm over last dim: standard width `H`, hybrid attention input width `2H`.
- Grouped gated RMSNorm in Mamba output: optional gate branch applies `silu(gate)` before per-group RMS normalization.
- Linear projections:
  - Mamba `in_proj`: `H -> I + (I + 2 * G * S) + M`, where `I=mamba_expand*H`, `G=mamba_ngroups`, `S=mamba_d_state`, `M=n_mamba_heads`.
  - Mamba `out_proj`: `I -> H`.
  - Hybrid attention `q_proj`: `2H -> A_heads * A_dim = 2H`.
  - Hybrid attention `k_proj` and `v_proj`: `2H -> KV_heads * A_dim`.
  - Hybrid attention `o_proj`: `2H -> H`.
  - Hybrid MLP `gate_up_proj`: `H -> 2 * intermediate_size`.
  - Hybrid MLP `down_proj`: `intermediate_size -> H`.
  - Hybrid bridge linear: `H -> H`, bias false.
  - LM head: `H -> vocab_size`, bias false, tied to embeddings if `tie_word_embeddings` is effective.
- Activations: `silu` for Mamba conv/gate; `gelu` by inspected configs for MLP gate.
- Depthwise causal Conv1d over `[B, conv_dim, T]` with groups=`conv_dim`, kernel=`mamba_d_conv`, padding=`mamba_d_conv-1`, truncate to sequence length.

Attention primitives:

- Causal self-attention only, in `hybrid` layers.
- MHA/GQA via Q `[B, heads, Q, A_dim]`, K/V `[B, kv_heads, K, A_dim]`, repeat KV to query heads if needed.
- Scale is `(attention_head_dim / 2) ** -0.5`, not the ordinary `head_dim ** -0.5`.
- Optional RoPE on Q/K only when `use_mem_rope`.
- Backend-dispatched attention through eager/SDPA/FlashAttention/Flex attention interfaces.

State-space/cache ops:

- Hybrid cache manifest per layer type:
  - `mamba` layers: Mamba conv state plus recurrent SSM state, no KV cache.
  - `hybrid` layers: both Mamba conv/recurrent state and dynamic attention K/V cache.
- Conv state shape from tests/source: `[B, I + 2 * G * S, mamba_d_conv]`.
- Recurrent state shape from tests/source: `[B, n_mamba_heads, mamba_headdim, mamba_d_state]`.
- Attention KV cache shape: `[B, num_key_value_heads, past_T, attention_head_dim]` for hybrid layers only.
- Decode update must mutate static-address cache tensors in place for cudagraph compatibility.

Position/rotary ops:

- Position IDs are generated as `arange(T) + past_seen_tokens`.
- Default RoPE computes inverse frequencies with base `rope_theta`, dimension from config head dim; cos/sin are computed in fp32 and cast to input dtype.
- Long-context flag sets `max_position_embeddings=16384` in config post-init, but inspected public configs have `use_long_context=false`.

Preprocessing-coupled ops:

- Tokenization uses `LlamaTokenizer` in representative configs.
- Instruct checkpoints use ChatML-like `<|im_start|>role\ncontent<|im_end|>` prompt template; this is controller/tokenizer ABI, not a GPU graph op.

## 5. Layer/block breakdown

Mamba decoder layer, repeated for every `mamba` entry:

```text
residual = x                                  # [B,T,H]
x = RMSNorm(x)
proj = Linear(H -> I + conv_dim + M)(x)
gate, hidden_B_C, dt = split(proj)
hidden_B_C = depthwise causal Conv1d(conv_dim, kernel=d_conv) + SiLU
hidden, B, C = split(hidden_B_C, [I, G*S, G*S])
y = Mamba2 selective scan/update(hidden, dt, A=-exp(A_log), B, C, D)
y = grouped gated RMSNorm(y, gate)
y = Linear(I -> H)(y)
out = residual + y
```

Hybrid layer, repeated at config-selected positions:

```text
attn_in = concat(x, original_token_embeddings)   # [B,T,2H]
attn_in = RMSNorm(attn_in)
q,k,v = Linear(2H -> 2H), Linear(2H -> KV*A_dim), Linear(2H -> KV*A_dim)
q,k = optional RoPE(q,k)
attn = causal self-attention(q,k,v, cache)
attn = Linear(2H -> H)(attn)
ff = RMSNorm(attn)
gate, up = split(Linear(H -> 2*intermediate)(ff))
ff = GELU(gate) * up
ff = Linear(intermediate -> H)(ff)
bridge = Linear(H -> H, bias=False)(ff)
x = MambaDecoderLayer(x, transformer_hidden_states=bridge)
```

Important semantic detail: the shared transformer block output is not added as its own residual. It is linearly projected and added to the input of the Mamba layer before that layer's input RMSNorm.

## 6. Attention requirements

Attention is present only in `hybrid` layers. The primary target still needs attention for generation parity, but not every layer owns KV state.

| Requirement | Source behavior |
| --- | --- |
| Type | Causal self-attention |
| Layer ownership | `hybrid` layers only |
| Input width | `attention_hidden_size = 2H` after concat with original embeddings |
| Query heads | `num_attention_heads` |
| KV heads | `num_key_value_heads`; source supports GQA by repeat expansion |
| Head dim | `attention_head_dim = 2H / num_attention_heads` |
| Scaling | `(head_dim / 2) ** -0.5` |
| Mask | `create_causal_mask` result passed as `causal_mask` to attention; Mamba receives the original 2D attention mask for pad zeroing |
| RoPE | Optional, controlled by `use_mem_rope` |
| Packed/varlen | Not explicitly in Zamba2 source; backend attention may support it internally |
| Sliding/local attention | None in this source |
| KV cache | Dynamic attention cache only for hybrid layers; K/V stored after optional RoPE because cache update follows RoPE |
| Backend compatibility | eager, SDPA, FlashAttention, Flex through `ALL_ATTENTION_FUNCTIONS`; FA2 right padding plus cache is called out as unsupported in tests |

Decode cache state ABI must include both attention KV for hybrid layers and fixed-size Mamba states for every layer. The cache object must also answer `get_seq_length()` using the first attention layer, while Mamba states do not grow with sequence length.

## 7. Position encoding and custom math

RoPE is optional and only applies to hybrid attention:

```python
def zamba2_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RoPE cos/sin can be precomputed for a maximum position bucket when using default RoPE. Dynamic RoPE variants would depend on sequence length, but inspected configs use absent/default `rope_parameters`.

Mamba2 custom math is the largest gap. Prefill either calls `mamba_chunk_scan_combined` or the eager fallback. The fallback chunks sequence length to `chunk_size`, builds lower-triangular segment sums, exponentiates cumulative A terms, computes intra-chunk and inter-chunk low-rank contributions, then returns final SSM state. Decode uses a one-token selective state update:

```python
dt = softplus(dt + dt_bias).clamp_min(time_step_min)
dA = exp(dt[..., None] * A)
dB = dt[..., None] * B[..., None, :]
state = state * dA + dB * x[..., None]
y = matmul(state, C[..., None]).squeeze(-1) + D * x
```

## 8. Preprocessing and input packing

Runtime graph inputs:

- `input_ids`: `[B,T]`, integer token IDs.
- `attention_mask`: optional `[B,T_total]` with `1` for valid tokens. It affects causal attention and also zeroes Mamba hidden states for pad tokens.
- `position_ids`: optional `[B,T]`; if omitted, source derives from cache sequence length.
- `inputs_embeds`: optional `[B,T,H]`; mutually exclusive with `input_ids`.

CPU/tokenizer ABI:

- Base checkpoints use `LlamaTokenizer` with Mistral-like tokens (`<s>`, `</s>`).
- Instruct checkpoints use tokenizer special tokens `<|im_start|>` and `<|im_end|>` plus a ChatML-style template.
- No multimodal placeholder scatter, image/audio packing, or packed sequence descriptors are present.

Generation controller behavior:

- `prepare_inputs_for_generation` sets `logits_to_keep = config.num_logits_to_keep`.
- Default `num_logits_to_keep=1` avoids full prompt logits during generation.

## 9. Graph rewrite / lowering opportunities

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice(-logits_to_keep, None), :])
```

Replacement:

```text
SliceLastK(hidden_states, K) -> GEMM(H -> vocab)
```

Preconditions: inference without loss; `logits_to_keep` is integer or validated index tensor; generation path defaults to `1`.

Failure cases: full-logit prefill parity tests, teacher-forcing/loss, caller-provided tensor indices.

Parity test sketch: compare full HF logits slice against DinoML sliced GEMM for `K=1`, `K=4`, and full `T`.

### Rewrite: packed gated MLP projection

Source pattern:

```text
gate_up = Linear(H -> 2I)
gate, up = chunk(gate_up, 2)
out = Linear(I -> H)(gelu(gate) * up)
```

Replacement:

```text
GEMM(H -> 2I) -> split -> GeluMul -> GEMM(I -> H)
```

Preconditions: weight rows are split as first `I` gate, second `I` up; activation equals checkpoint `hidden_act`; optional adapter addition is fused or disabled.

Failure cases: non-`gelu` configs, active adapter branches, nonstandard bias settings.

### Rewrite: Mamba depthwise causal Conv1d decode update

Source pattern:

```text
conv_state = roll_or_copy(previous_conv_state, new_hidden_B_C)
hidden_B_C = sum(conv_state * conv_weight, dim=-1) + bias
hidden_B_C = silu(hidden_B_C)
```

Replacement:

```text
StateAppend -> depthwise dot(kernel=4) -> bias -> SiLU
```

Preconditions: decode token count small, kernel size fixed at 4, groups equal channels, contiguous `[B,C,K]` state.

Failure cases: prefill/full convolution path, kernel size variants, missing bias behavior if `use_conv_bias=false` is ever honored differently from source's Conv1d construction.

### Rewrite: hybrid attention QKV packing

Source pattern: separate Q, K, V linears from the same `[B,T,2H]` tensor.

Replacement: one packed GEMM producing QKV with row-block order `[Q, K, V]`, followed by views/splits.

Preconditions: adapters disabled or folded into packed weights; same dtype/layout; K/V widths known.

Failure cases: `use_shared_attention_adapter=true` with per-hybrid-layer adapter lists, tied block aliases that must not be duplicated incorrectly.

### Guarded no-layout-translation region: Mamba scan

The Mamba fallback uses axis-sensitive `transpose`, chunking on sequence dim, lower-triangular masks over chunk dim, and state shapes `[B, heads, head_dim, state]`. Initial DinoML lowering should preserve source axes. Layout optimization belongs inside a dedicated Mamba provider, not a generic NHWC/channel-last pass.

## 10. Kernel fusion candidates

Highest priority:

- Mamba2 prefill scan provider: this dominates non-attention layers and currently maps to external Triton kernels in source.
- Mamba2 decode state update: fixed-size conv and recurrent state update are required for usable generation latency.
- RMSNorm and grouped gated RMSNorm: frequent, simple, and shape-stable.
- Hybrid attention with optional RoPE and KV cache: fewer layers than Mamba, but essential for parity.

Medium priority:

- Mamba `in_proj` split plus depthwise conv staging.
- GELU-gated MLP fused activation multiply.
- Last-token-only LM head.
- QKV packed projection when adapters are disabled.

Lower priority:

- Adapter fusion for 1.2B shared attention adapters.
- Classification pooling/head.
- Eager fallback chunked scan as generic graph ops; useful for reference, poor as production lowering.

## 11. Runtime staging plan

Stage 1: parse config and instantiate exact layer schedule, including `mamba` versus `hybrid`, tied hybrid block aliases, and tokenizer/generation metadata.

Stage 2: load weights and run embedding, RMSNorm, linear, MLP, and one Mamba layer in eager/reference form. Validate state shapes from config.

Stage 3: implement a bounded Mamba2 provider for prefill and one-token decode. Stub hybrid attention initially only for mamba-only slices or tiny synthetic configs.

Stage 4: add hybrid attention blocks with causal mask, optional RoPE, and KV cache. Validate mixed cache manifest layer by layer.

Stage 5: run full prefill logits parity for 1.2B/2.7B style configs with `use_mamba_kernels=false` reference, then optimized provider parity.

Stage 6: implement decode loop parity with static-address Mamba states and hybrid KV growth.

Stage 7: add fusions: QKV packing, MLP fused gate, last-token logits, Mamba projection/conv/scan fusion.

Stage 8: add production scheduling constraints: continuous batching must manage both growing attention KV and fixed Mamba state per request.

## 12. Parity and validation plan

- Config parser tests for source defaults and public config rows in `evidence_configs.md`.
- Random tensor RMSNorm and gated grouped RMSNorm parity in fp32/fp16/bf16; suggested tolerances fp32 `1e-5`, fp16/bf16 `2e-2` around reductions.
- Mamba conv-state update parity for prefill lengths `<4`, `=4`, `>4`, and decode one-token append.
- Mamba recurrent update parity against the torch fallback for small `B,T,H,heads,state,groups`.
- Single Mamba decoder layer parity with `use_mamba_kernels=false`.
- Single hybrid layer parity with and without `use_mem_rope`, and with GQA synthetic config where `num_key_value_heads < num_attention_heads`.
- Full small synthetic model parity over a mixed `["mamba", "hybrid", "mamba"]` schedule.
- Prefill logits parity for public-style configs using random weights, then real checkpoint smoke when weights are available.
- Decode token parity for one step and multi-step continuation, including cache reorder/reset tests.
- Classification head parity: last non-pad token selection and pooled logits, optional.

## 13. Performance probes

- Mamba prefill throughput sweep over `B`, `T`, `chunk_size`, `mamba_d_state`, and `mamba_ngroups`.
- Mamba decode tokens/sec with fixed state update and depthwise conv kernel size 4.
- Hybrid attention prefill/decode comparison: eager, SDPA/Flash-style, and DinoML fused path.
- Cache memory probe split by Mamba fixed states versus hybrid attention KV.
- Last-token LM head time versus full prompt logits.
- 1.2B adapter-enabled versus adapter-disabled hybrid layer cost.
- Sequence length sweep at 512, 2048, 4096, and long-context 16384 synthetic config.
- End-to-end generation throughput with mixed cache manifest and batch growth.

## 14. Skip/defer list

- Training, gradients, and gradient checkpointing.
- External HF hub kernel loading mechanics; DinoML should own or explicitly depend on equivalent providers.
- CPU/GPU offload parity for tied hybrid blocks; upstream tests skip offload due to mixed layer/tied-weight issues.
- Sequence classification head for first causal LM target.
- Beam-search reorder beyond cache `index_select` parity; basic reorder should still be validated before production generation.
- Long-context variants unless a config with `use_long_context=true` is admitted.
- Quantization/packed weight loading; no source-coupled quantized format is required by this implementation.
- MoE routing, because Zamba2 source has no experts.

## 15. Final implementation checklist

- [ ] Parse `Zamba2Config`, including derived `attention_hidden_size`, `attention_head_dim`, `mamba_headdim`, and `hybrid_layer_ids`.
- [ ] Preserve `layers_block_type` as an artifact-visible mixed cache/lowering manifest.
- [ ] Load tied token embedding/LM head and tied hybrid transformer blocks without cloning logical aliases.
- [ ] Implement RMSNorm and grouped gated RMSNorm.
- [ ] Implement Mamba2 projection split, depthwise causal Conv1d, prefill scan, and decode state update.
- [ ] Implement Mamba cache states `[B, conv_dim, d_conv]` and `[B, n_mamba_heads, mamba_headdim, d_state]`.
- [ ] Implement hybrid attention with concat-original-embedding input, optional RoPE, custom scale `(head_dim/2)^-0.5`, and KV cache.
- [ ] Implement hybrid MLP gated GELU and bridge linear injection into Mamba.
- [ ] Implement causal mask and attention mask pad-zeroing semantics.
- [ ] Implement last-token-only LM head path controlled by `logits_to_keep`.
- [ ] Add one-layer Mamba, one hybrid layer, mixed-cache, prefill-logit, and decode-continuation parity tests.
- [ ] Benchmark Mamba provider, hybrid attention provider, cache memory, and last-token logits separately.
