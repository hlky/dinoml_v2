# DeepSeek-V4 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4, local checkout transformers
Model id: deepseek-ai/DeepSeek-V4-Flash-Base, deepseek-ai/DeepSeek-V4-Flash, deepseek-ai/DeepSeek-V4-Pro-Base, deepseek-ai/DeepSeek-V4-Pro
Config source: HF config.json snapshots copied under _sources/
Source files inspected:
  transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py
  transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
  transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
  transformers/docs/source/en/model_doc/deepseek_v4.md
  transformers/src/transformers/activations.py
Any missing files or assumptions: no tokenizer-specific coupling was needed for the neural graph. No small/debug checkpoint was found; official representatives are very large production checkpoints.
```

`modeling_deepseek_v4.py` is generated from `modular_deepseek_v4.py`; future source edits should target the modular file, but runtime behavior was checked against the generated file. HF configs inspected from:

- [DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json)
- [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
- [DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base/blob/main/config.json)
- [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)
- Mirror quantization sample: [mlx-community/DeepSeek-V4-Flash-4bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit/blob/main/config.json)

## 2. High-level architecture

Primary DinoML target: text-only causal LM inference, first prefill and decode logits for `DeepseekV4ForCausalLM`.

Dataflow:

```text
token ids -> embedding -> hc_mult residual streams -> repeated decoder blocks
  -> hyper-head stream collapse -> RMSNorm -> lm_head -> logits/sampling
```

Each decoder block is a MoE decoder with manifold-constrained hyper-connections, shared-KV multi-query attention, optional compressed long-range KV branches, grouped low-rank output projection, and routed plus shared SwiGLU experts. There is no vision/audio branch.

Independently stageable pieces:

- Embedding + final RMSNorm + LM head.
- mHC stream machinery, because hidden state rank is `[B, S, hc_mult, D]` through the stack.
- One decoder block with local sliding attention only.
- CSA/HCA compressor state and cache ABI.
- MoE routing and expert dispatch.
- Quantized weight loading/provider handling, separate from dense graph semantics.

## 3. Important config dimensions

| Field | Flash / Flash-Base | Pro / Pro-Base | Source |
|---|---:|---:|---|
| `hidden_size` | 4096 | 7168 | config.json |
| `num_hidden_layers` | 43 | 61 | config.json |
| `num_attention_heads` | 64 | 128 | config.json |
| `num_key_value_heads` | 1 | 1 | config.json |
| `head_dim` | 512 | 512 | config.json |
| Q output width | 32768 | 65536 | inferred from heads * head_dim |
| KV width | 512 | 512 | source uses shared K=V MQA |
| `q_lora_rank` | 1024 | 1536 | config.json |
| `qk_rope_head_dim` | 64 | 64 | legacy config, folded into `partial_rotary_factor` |
| `moe_intermediate_size` | 2048 | 3072 | config.json |
| `n_routed_experts` | 256 | 384 | config.json |
| `num_experts_per_tok` | 6 | 6 | config.json |
| `n_shared_experts` | 1 | 1 | config.json; source has one shared MLP, not a loop |
| `sliding_window` | 128 | 128 | config.json |
| `compress_rates` | CSA 4, HCA 128 | CSA 4, HCA 128 | config defaults / legacy ratios |
| `index_n_heads`, `index_head_dim` | 64, 128 | 64, 128 | config.json |
| `index_topk` | 512 | 1024 | config.json |
| `o_groups`, `o_lora_rank` | 8, 1024 | 16, 1024 | config.json |
| `hc_mult` | 4 | 4 | config.json |
| `max_position_embeddings` | 1048576 | 1048576 | config.json |
| RoPE | YaRN factor 16, original 65536, main theta 10000, compress theta 160000 | same | config.json + config post-init |
| dtype | `bfloat16` | `bfloat16` | config.json |
| cache | dynamic only, custom per-layer state | dynamic only, custom per-layer state | source |

Representative checkpoint sweep:

| Checkpoint | Role | Size knobs | Attention schedule after source truncation | Expert dtype / quant metadata |
|---|---|---|---|---|
| `DeepSeek-V4-Flash-Base` | base | 43L, D=4096, 256 experts | 2 sliding, 21 CSA, 20 HCA | `expert_dtype=fp8`, `quantization_config.quant_method=fp8` |
| `DeepSeek-V4-Flash` | instruct/chat | same as Flash-Base | same | `expert_dtype=fp4`, but quant config still says fp8 |
| `DeepSeek-V4-Pro-Base` | base | 61L, D=7168, 384 experts | 30 CSA, 31 HCA, no used sliding | `expert_dtype=fp8`, fp8 quant config |
| `DeepSeek-V4-Pro` | instruct/chat | same as Pro-Base | same | `expert_dtype=fp4`, but quant config still says fp8 |
| `mlx-community/DeepSeek-V4-Flash-4bit` | open mirror | Flash dims | same as Flash | mirror-specific `quantization` map, not native Transformers behavior |

Official configs carry legacy `compress_ratios` one entry longer than `num_hidden_layers`; current source truncates schedules to `num_hidden_layers`.

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim`: attention Q width is much larger than model hidden size, followed by grouped output projection.
- `num_key_value_heads` is effectively fixed to 1 in source behavior; K and V are the same tensor.
- Flash uses two sliding bootstrap layers; Pro has no used sliding layer after schedule truncation.
- CSA and HCA have different cache state and compressor math; they cannot be lowered as ordinary dense causal attention.
- `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are false in source. Reasons include `head_dim=512`, attention sinks, and compressor KV-length mutation.
- Static cache/fullgraph compile is disabled. `DynamicCache(config=...)` is required to instantiate `DeepseekV4HCACache` / `DeepseekV4CSACache`.
- mHC changes the residual tensor rank to `[B, S, hc_mult, D]` inside the model.
- First few MoE layers are `hash_moe`, using checkpoint buffer `tid2eid[input_ids]`; later layers use top-k learned routing.
- Router score is `sqrt(softplus(logits))`, not softmax. `topk_method`, `norm_topk_prob`, and `expert_dtype` are config metadata; current in-library source does not branch on `topk_method` or `expert_dtype`.
- Instruct configs advertise `expert_dtype=fp4` while `quantization_config` still says fp8. Treat as a weight/provider contract, not a graph-op change.
- Mirror 4-bit configs use non-Transformers module names such as `attn.wq_a` and `ffn.switch_mlp`; do not load as native HF weight layout without an explicit mapping.
- No NHWC/NCHW layout issue in the neural graph; all tensors are token sequences. Layout guards are instead about `[B,S,H,D]`, `[B,H,S,D]`, compressor windows, and grouped-head flatten order.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,D]`.
- `unsqueeze`, `expand`, `contiguous`, `view`/`reshape`, `transpose`, `flatten`, `chunk`, `cat`, `pad`, `gather`, `index_add_`, `one_hot`, `where`, `nonzero`, `topk`, `arange`.
- Dynamic shape guards for sequence length, cache length, compressor window divisibility, and output logits slicing.

Neural primitives:

- RMSNorm weighted and unweighted with fp32 variance.
- Linear/GEMM without bias for attention and most projections.
- Grouped block-diagonal linear implemented as BMM over groups.
- SwiGLU: `silu(gate) * up`; routed experts additionally clamp gate/up.
- Sigmoid, softplus, sqrt, ReLU, softmax, rsqrt, matmul, reductions.
- Sinkhorn loop for mHC: 20 row/column normalization iterations by default.

Attention primitives:

- Causal shared-KV MQA with K=V, `head_dim=512`.
- Sliding-window causal mask, local branch length 128.
- Per-head attention sink appended to attention logits before softmax, then dropped from value matmul.
- Partial interleaved RoPE on trailing 64 channels, with conjugate output rotation.
- CSA/HCA compressed KV concatenated onto local KV axis.

MoE and routing:

- Hash router: `tid2eid[input_ids] -> [B*S, top_k]`.
- Learned router: fp32 `Linear(D -> E)`, `sqrtsoftplus`, correction-bias top-k, gather, weight normalization, scale.
- Expert tensor layout: `gate_up_proj[E, 2*I, D]`, `down_proj[E, D, I]`, split order `[gate, up]`.
- Dispatch fallback in source loops over hit experts, gathers tokens, runs two expert GEMMs, and `index_add_` accumulates weighted output.
- Shared expert MLP: dense `D -> I -> D`.

Quantized/packed metadata:

- Official `quantization_config`: fp8, dynamic activations, e4m3, ue8m0 scales, block `[128,128]`.
- `expert_dtype` flags fp8/fp4 but is not read by native modeling code.
- MLX mirror 4-bit config is external packed-weight metadata and requires separate loader admission.

Generation/cache:

- `DynamicCache(config=...)` with layer-specific cache classes.
- Cache reorder/rollback-sensitive generation paths should be rejected initially because source marks `_is_stateful=True`.
- `logits_to_keep` supports last-token or indexed logits.

## 5. Layer/block breakdown

Model setup:

```text
inputs_embeds = Embedding(input_ids)                    [B,S,D]
hidden_streams = expand(inputs_embeds, hc_mult)         [B,S,4,D]
position_embeddings = main RoPE cos/sin                 [B,S,32] before repeat_interleave
```

Decoder block, repeated `num_hidden_layers`:

```text
post, comb, collapsed = attn_hc(hidden_streams)         collapsed [B,S,D]
x = RMSNorm(collapsed)
q_residual = RMSNorm(Linear(D -> q_lora_rank)(x))
q = Linear(q_lora_rank -> num_heads*head_dim)
q = view/transpose -> [B,H,S,512]
q = unweighted RMSNorm(q)
q = interleaved trailing-slice RoPE(q, main)
kv = RMSNorm(Linear(D -> 512)(x)) -> [B,1,S,512]
kv = interleaved trailing-slice RoPE(kv, main)
kv = dynamic sliding cache update, K == V
if CSA/HCA: compressed_kv = compressor(...); kv = cat(local_kv, compressed_kv, dim=sequence)
attn = eager attention with sink logits and causal/sliding mask
attn = conjugate RoPE(attn, main)
attn = grouped output projection: grouped BMM -> Linear(o_groups*o_lora_rank -> D)
hidden_streams = post * attn + comb @ hidden_streams

post, comb, collapsed = ffn_hc(hidden_streams)
y = RMSNorm(collapsed)
y = routed_experts(y, input_ids) + shared_swiglu(y)
hidden_streams = post * y + comb @ hidden_streams
```

Final:

```text
hidden = RMSNorm(hc_head(hidden_streams))                [B,S,D]
logits = Linear(D -> vocab_size)(hidden[:, slice, :])
```

Projection shapes:

- Flash Q path: `4096 -> 1024 -> 32768`; KV: `4096 -> 512`; grouped out: 8 groups of 4096 to 1024, then `8192 -> 4096`.
- Pro Q path: `7168 -> 1536 -> 65536`; KV: `7168 -> 512`; grouped out: 16 groups of 4096 to 1024, then `16384 -> 7168`.

## 6. Attention requirements

Attention is causal self-attention only. It is not standard FlashAttention/SDPA-compatible in the current source.

Common attention ABI:

- Query shape: `[B, num_attention_heads, q_len, head_dim]`.
- KV shape before repeat: `[B, 1, kv_len, head_dim]`.
- KV is repeated across query heads for eager attention.
- K and V are the same tensor after RoPE; value uses RoPE and output applies inverse rotation.
- Mask is sliding-window causal and is padded with zeros when compressed KV entries are appended.
- Attention logits include an extra sink column per head; sink probability is removed before multiplying values.

Cache ABI:

- Sliding branch cache stores one rolling K=V tensor per layer with effective window `sliding_window - 1` retained before adding current tokens.
- HCA cache adds `buffer_kv["compressor"]`, `buffer_gate["compressor"]`, `compressed_kv["compressor"]`, and `entry_count["compressor"]`.
- CSA cache additionally adds `"indexer"` entries and overlap state `overlap_kv` / `overlap_gate` for both compressor and indexer.
- Cached compressed entries are stored after compressed-branch RoPE.
- `StaticCache` is not compatible because compressor-specific methods are required.

CSA:

- Compress rate 4.
- Compressor emits overlapping compressed entries from two series Ca/Cb, using softmax over `2 * compress_rate`.
- Lightning indexer builds its own compressed keys at `index_head_dim=128`, scores `[B,S,index_n_heads,T]`, reduces over heads with learned weights, and gathers top `index_topk` compressed entries per query.
- Gathered compressed KV output shape is reshaped to `[B,1,S*topk,head_dim]`, so compressed keys are query-expanded rather than a simple shared `[B,1,T,D]` block.

HCA:

- Compress rate 128.
- Non-overlapping compressor emits one compressed entry per full window and concatenates all running entries to the local KV branch.
- No indexer and no overlap state.

## 7. Position encoding and custom math

RoPE is interleaved-pair, partial, and applied to the trailing slice:

```python
def deepseek_v4_rope(x, cos, sin):
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    x1, x2 = rope[..., 0::2], rope[..., 1::2]
    rotated_half = stack((-x2, x1)).flatten(-2)
    return cat([nope, rope.float() * cos + rotated_half.float() * sin], dim=-1).to(x.dtype)
```

Main branch uses `rope_theta=10000`. Compressed branch uses `compress_rope_theta=160000`. YaRN parameters from legacy `rope_scaling` are folded into nested `rope_parameters` for `main` and `compress`.

Custom math to reproduce:

- `sqrtsoftplus(x) = sqrt(softplus(x))` for router scores.
- mHC Sinkhorn projection on a `[B,S,hc_mult,hc_mult]` matrix.
- Attention sink: concatenate learnable sink logits to attention logits, softmax, drop sink probabilities before value matmul.
- CSA overlap construction with `-inf` gates for unavailable prior Ca slots.

Position-dependent and dynamic:

- Query/main RoPE depends on `position_ids`.
- Compressor positions are deterministic window starts: `entry_count * compress_rate + window_index * compress_rate`.
- Position caches must survive prefill/decode boundaries through `entry_count`.

## 8. Preprocessing and input packing

Text preprocessing is standard tokenizer-owned `input_ids` plus optional `attention_mask`. The model graph consumes:

- `input_ids [B,S]` for embeddings and hash-MoE routing.
- Optional `inputs_embeds [B,S,D]`; hash-MoE layers still need `input_ids`, so DinoML should reject `inputs_embeds` without `input_ids` for hash layers unless a hash-route tensor is supplied.
- Optional `position_ids [B,S]`; if omitted, source creates an arange offset by cache length.
- Optional `attention_mask [B,S]`, converted to a sliding-window causal mask.
- `logits_to_keep`, which is generation-controller ABI for last-token-only logits.

No modality packing, placeholder scatter, image/video/audio preprocessing, or channel layout translation is involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: grouped output projection to grouped GEMM plus GEMM

Source pattern:

```text
attn_output [B,S,H,Dh] -> reshape [B,S,o_groups,H*Dh/o_groups]
DeepseekV4GroupedLinear -> flatten -> o_b_proj
```

Replacement:

```text
GroupBMM/BatchedGEMM over o_groups -> reshape [B,S,o_groups*o_lora_rank] -> GEMM
```

Preconditions:

- `num_attention_heads * head_dim % o_groups == 0`.
- Source flatten order `[groups, heads_per_group, head_dim]` is preserved.
- Weight layout matches `DeepseekV4GroupedLinear.weight.view(o_groups, out_per_group, in_per_group)`.

Failure cases: arbitrary tensor-parallel sharding or non-contiguous head layout.

### Rewrite: dense local attention with sink

Source pattern:

```text
QK^T * scale + mask
cat(sink_logits)
softmax
drop sink column
P @ V
```

Replacement: custom attention kernel with a virtual sink column and no materialized sink value.

Preconditions:

- Inference dropout is zero.
- K=V and MQA repetition is handled inside kernel or by grouped indexing.
- Dense tensor attention only for local sliding branch plus already-materialized compressed entries.

Failure cases: using SDPA/FA directly loses sink behavior.

### Rewrite: HCA compressor window pooling

Source pattern:

```text
kv = Linear(D -> head_dim)
gate = Linear(D -> head_dim) + position_bias
softmax(gate, dim=window)
sum(kv * weights)
RMSNorm
compressed RoPE
```

Replacement: fused windowed projection/pool kernel or projection GEMM plus fused softmax-weighted reduction.

Preconditions:

- Full windows only; remainders stay in cache buffer.
- Non-overlap for HCA, overlap disabled.
- `compress_rate` static per layer type.

### Rewrite: CSA indexed compressed attention

Replacement: two-stage provider path:

```text
compress windows -> indexer scores/topk -> gather compressed entries -> attention
```

Preconditions:

- Top-k is along compressed-entry axis.
- Gather order matches `torch.gather` over expanded `[B,1,S,T,D]`.
- `topk = min(index_topk, compressed_count)`; zero compressed entries must be handled.

Failure cases: treating compressed KV as a fixed shared sequence loses query-dependent gather.

### Rewrite: MoE expert loop to grouped GEMM dispatch

Source pattern:

```text
topk -> one_hot/where per expert -> expert gate_up GEMM -> clamp/SwiGLU -> down GEMM -> weighted index_add
```

Replacement: token bucketing by expert, grouped GEMM for `gate_up` and `down`, weighted scatter-add.

Preconditions:

- Top-k indices are int64/int32 and bounded by expert count.
- Expert tensor split order is `[gate, up]`.
- Accumulation order tolerances are defined for reduced precision.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden[:, slice_indices, :])`.

Replacement: lower only requested logits rows when `logits_to_keep` is static last-token or small index tensor.

Failure cases: training loss or full-sequence logits requested.

## 10. Kernel fusion candidates

Highest priority:

- Dynamic cache/state ABI for sliding + CSA/HCA. Without this, decode parity is gated.
- RMSNorm and unweighted RMSNorm, including fp32 accumulation.
- Sink attention kernel with MQA, `head_dim=512`, and virtual sink column.
- MoE routing + grouped expert GEMMs + scatter-add.
- mHC Sinkhorn/mix kernels, because they wrap every attention and MLP site.

Medium priority:

- RoPE fused into Q/KV projections and conjugate output rotation.
- HCA compressor projection + window softmax reduction.
- CSA compressor/indexer/topk/gather pipeline.
- Grouped output projection as grouped GEMM plus dense GEMM.
- Last-token-only LM head.

Lower priority:

- Full router auxiliary loss; training-only for first inference target.
- Returned attention tensors; useful for debugging, expensive for production.
- Mirror-specific 4-bit packed-weight execution.

## 11. Runtime staging plan

Stage 1: parse config and enforce admission guards.

- Accept official Flash/Pro dense graph configs.
- Reject static cache, SDPA/FA/Flex lowering, missing `input_ids` for hash layers, and unknown quant formats.

Stage 2: dense one-block parity without cache.

- Implement mHC, RMSNorm, grouped output projection, eager attention with sink, and shared expert.
- Start with a sliding-attention Flash bootstrap block.

Stage 3: MoE parity.

- Add hash routing, learned top-k routing, expert packed tensor loading, and grouped GEMM dispatch.

Stage 4: prefill CSA/HCA.

- Implement compressor window pooling, CSA indexer, top-k gather, and compressed-KV attention concatenation.

Stage 5: decode cache parity.

- Add dynamic sliding/HCA/CSA cache manifests with buffers, overlap state, compressed entries, entry counts, and reorder restrictions.

Stage 6: quantized/provider loading.

- Add fp8 official weight materialization/admission first. Treat `expert_dtype=fp4` and MLX 4-bit mirror as separate provider contracts.

Stage 7: production scheduling.

- Continuous batching only after state rollback/reorder policy is explicit; source marks the model stateful.

## 12. Parity and validation plan

- Unit tests for interleaved partial RoPE against source tensors, including inverse output rotation.
- RMSNorm and unweighted RMSNorm fp32-accumulation tests for bf16/fp16/fp32.
- mHC tests with small `hc_mult` and fixed Sinkhorn iteration count.
- Attention sink tests comparing eager attention to source for local-only blocks.
- HCA compressor tests for remainder buffering and `entry_count`.
- CSA compressor/indexer tests for overlap state, `topk=min(index_topk,T)`, and gather order.
- MoE router tests for hash lookup, sqrtsoftplus top-k, correction bias, weight normalization, and expert scatter-add.
- Single-layer parity: sliding, HCA, and CSA layer types separately.
- Prefill logits parity at short, window-boundary, and compressor-boundary lengths.
- Decode token parity across prefill/decode calls, including cache buffers.
- Recommended tolerances: fp32 `1e-5`/`1e-4`; bf16/fp16 attention and MoE use looser per-stage tolerances, then end-to-end logits tolerance based on HF bf16 eager baseline.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 4096, 65536, and compressor-boundary cases.
- Decode tokens/sec with cache warm states and different compressed-entry counts.
- CSA indexer time split: compressor, index scores, top-k, gather, attention.
- HCA compressor time split: projection, window softmax pooling, compressed attention.
- MoE routing/expert grouped GEMM occupancy by batch and sequence length.
- KV/cache memory: local sliding cache, compressed pools, CSA query-expanded gathered KV.
- mHC overhead as percent of block time.
- LM head full logits vs last-token-only logits.
- fp8/fp4 dequant/materialization vs dense bf16 weights, separately from graph execution.

## 14. Skip/defer list

- Training loss and router auxiliary loss.
- Gradient checkpointing.
- Beam search, assisted/speculative generation, prompt lookup, and contrastive search; source state cannot be rewound safely.
- Static cache/fullgraph compile.
- FlashAttention/SDPA/FlexAttention lowering until custom sink/compressor-compatible providers exist.
- Tensor parallel/expert parallel beyond metadata-aware loading.
- MLX mirror 4-bit packed execution.
- Remote-code or non-HF weight key aliases without explicit mapping.

## 15. Final implementation checklist

- [ ] Parse `DeepseekV4Config`, including legacy `compress_ratios`, `num_hash_layers`, `qk_rope_head_dim`, and nested RoPE conversion.
- [ ] Add admission guards for dynamic cache only, eager/custom attention only, `num_key_value_heads == 1`, K=V, and `head_dim == 512`.
- [ ] Load dense/bf16 weights and preserve tied LM-head metadata even though `tie_word_embeddings=false` in official configs.
- [ ] Implement weighted and unweighted RMSNorm.
- [ ] Implement interleaved trailing-slice RoPE and inverse output rotation.
- [ ] Implement mHC hyper-connection and hyper-head.
- [ ] Implement sink-aware MQA attention with sliding causal mask.
- [ ] Implement grouped output projection.
- [ ] Implement hash router and learned sqrtsoftplus top-k router.
- [ ] Implement routed expert grouped GEMM and weighted scatter-add.
- [ ] Implement HCA compressor and cache state.
- [ ] Implement CSA compressor, indexer, top-k gather, and cache overlap state.
- [ ] Add dynamic cache manifest/state ABI and reject state rollback generation modes.
- [ ] Add fp8 quantized weight admission/materialization plan.
- [ ] Add one-block, prefill, and decode parity tests.
- [ ] Benchmark prefill, decode, MoE, compressor, cache memory, and logits slicing.

## Gated gaps for DinoML

- Dynamic stateful cache ABI for HCA/CSA is the primary gate; ordinary KV cache is insufficient.
- Sink-aware attention is required; stock SDPA/FlashAttention parity is explicitly disabled by source.
- CSA query-dependent compressed gather is required for production configs.
- MoE expert dispatch needs grouped GEMM plus scatter-add; current DinoML grouped GEMM surface is not yet listed as complete.
- `head_dim=512` and `hidden_size != heads * head_dim` require nonstandard attention projection/output contracts.
- mHC introduces `[B,S,hc_mult,D]` hidden-state layout and Sinkhorn loops around every sublayer.
- Official quantization metadata is fp8, while instruct configs also flag fp4 experts; loader/provider policy must gate these before weight import.
