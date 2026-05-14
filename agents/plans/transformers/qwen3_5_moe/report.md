# Transformers audit: qwen3_5_moe

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Qwen/Qwen3.5-35B-A3B, Qwen/Qwen3.5-122B-A10B, Qwen/Qwen3.5-397B-A17B, plus FP8/GPTQ variants
Config source: Hugging Face config.json via /raw/main/config.json
Source files inspected: configuration_qwen3_5_moe.py, modeling_qwen3_5_moe.py, modular_qwen3_5_moe.py, __init__.py, cache_utils.py
Any missing files or assumptions: processor/tokenizer source was not inspected; quantized safetensors metadata was not downloaded.
```

`modeling_qwen3_5_moe.py` is generated from `modular_qwen3_5_moe.py`. For exact runtime behavior, this report uses the generated file. For future upstream source edits, `modular_qwen3_5_moe.py` is the authoritative source file.

Primary DinoML target: inference-only conditional generation, staged as text-only prefill/decode first, then multimodal image/video prefix construction. The shipped representative checkpoints use `Qwen3_5MoeForConditionalGeneration`, but the source also implements `Qwen3_5MoeForCausalLM` for text-only use.

## 2. High-level architecture

Qwen3.5-MoE is a multimodal autoregressive decoder with:

- A packed vision encoder for image/video patches.
- A hybrid text decoder with 3 linear-attention layers followed by 1 full-attention layer, repeated.
- A per-layer sparse MoE feed-forward block plus a dense shared expert.
- Multimodal RoPE positions for text/image/video token sequences.
- A bias-free LM head for conditional generation.

Dataflow:

```text
processor/tokenizer -> input_ids + mm_token_type_ids + grid_thw + packed pixels
  -> token embeddings
  -> optional vision encoder -> placeholder embedding stitch
  -> M-RoPE position ids
  -> hybrid MoE decoder prefill/decode
  -> last-token or selected-token lm_head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: chat template, tokenization, image/video decode/resizing/patch packing, `mm_token_type_ids`, `image_grid_thw`, `video_grid_thw`.
- Independently cacheable vision prefix: visual encoder outputs one feature row per image/video placeholder token after spatial merge.
- Prefix construction: replace image/video placeholder token embeddings with visual features, then compute M-RoPE position IDs.
- Text prefill: all layers run; full-attention layers populate KV cache, linear-attention layers populate fixed conv/recurrent states.
- Decode: full-attention layers append KV, linear-attention layers update fixed-size states, and logits can be restricted with `logits_to_keep`.

## 3. Important config dimensions

Source defaults for `Qwen3_5MoeTextConfig`:

| Field | Default |
|---|---:|
| `vocab_size` | 248320 |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 40 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `max_position_embeddings` | 32768 default, 262144 in representative checkpoints |
| `linear_conv_kernel_dim` | 4 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 32 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `hidden_act` | `silu` |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `use_cache` | true |

Representative checkpoint sweep:

| Checkpoint | hidden | layers | linear/full | attn heads | KV heads | head dim | linear K/V heads | experts/top-k | MoE/shared int | vision out | quant |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen3.5-35B-A3B` | 2048 | 40 | 30/10 | 16 | 2 | 256 | 16/32 | 256/8 | 512/512 | 2048 | none |
| `Qwen/Qwen3.5-122B-A10B` | 3072 | 48 | 36/12 | 32 | 2 | 256 | 16/64 | 256/8 | 1024/1024 | 3072 | none |
| `Qwen/Qwen3.5-397B-A17B` | 4096 | 60 | 45/15 | 32 | 2 | 256 | 16/64 | 512/10 | 1024/1024 | 4096 | none |
| `Qwen/Qwen3.5-122B-A10B-FP8` | 3072 | 48 | 36/12 | 32 | 2 | 256 | 16/64 | 256/8 | 1024/1024 | 3072 | FP8 |
| `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` | 3072 | 48 | 36/12 | 32 | 2 | 256 | 16/64 | 256/8 | 1024/1024 | 3072 | GPTQ int4 |

Shared checkpoint RoPE: `rope_type="default"`, `rope_theta=10000000`, `partial_rotary_factor=0.25`, `mrope_section=[11,11,10]`, `mrope_interleaved=true`.

Vision config defaults: `depth=27`, `hidden_size=1152`, `num_heads=16`, `intermediate_size=4304`, `patch_size=16`, `temporal_patch_size=2`, `spatial_merge_size=2`, `num_position_embeddings=2304`. `out_hidden_size` tracks text hidden size in representative configs.

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for all inspected text configs. Full-attention output width is `num_attention_heads * head_dim`, then projected back to hidden size.
- `q_proj` is not a standard Q projection: it outputs `2 * num_attention_heads * head_dim`, split into query and attention-output gate.
- GQA is strong: inspected configs use only 2 KV heads for 16 or 32 Q heads.
- The decoder is hybrid. Most layers are not KV-cache attention layers; they are Gated DeltaNet linear-attention layers with fixed conv/recurrent state.
- Linear attention projection widths are independent from full attention: `linear_num_key_heads * linear_key_head_dim` and `linear_num_value_heads * linear_value_head_dim`.
- MoE routing differs by size: the 397B config has 512 experts and top-10 routing, while 35B/122B use 256 experts and top-8.
- Expert weights are packed as 3D tensors; `gate_up_proj` split order is `gate, up`.
- Text RMSNorm uses zero-initialized weight and applies `(1 + weight)` at runtime, not a plain multiplicative weight initialized to one.
- M-RoPE uses 3D position IDs for multimodal sequences and an interleaved `mrope_section` layout. Text-only paths still expand position IDs to four rows and then pass rows 1:3 to RoPE.
- Multimodal insertion uses broad `masked_scatter`; DinoML should lower only under stricter placeholder-count/order guards from the processor.
- FP8/GPTQ configs advertise source-coupled quantization policies. Treat these as loader/provider contracts, not ordinary dense dtype changes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,H]`.
- Reshape/view/transpose/permute/contiguous for attention and vision packing.
- `torch.split`, `torch.chunk`, `torch.cat`, `torch.stack`.
- Boolean masks, equality against placeholder IDs, `masked_fill`, `masked_scatter` for multimodal stitch.
- `repeat`, `repeat_interleave`, `expand`, `index_select` for cache reorder and position construction.
- `topk`, `one_hot`, `where`, `nonzero`, `index_add_` for MoE routing fallback.

Neural primitives:

- RMSNorm with `(1 + weight)` scale for text.
- RMSNormGated for linear attention output: RMSNorm then multiply by `silu(z)`.
- LayerNorm in vision blocks and patch merger.
- Dense Linear and bias/no-bias variants.
- SiLU, GELU tanh approximation, sigmoid, softplus, exp, rsqrt.
- Depthwise causal Conv1d over linear-attention QKV channels.
- Vision Conv3d patch embed with `kernel=stride=[temporal_patch_size, patch_size, patch_size]`.

Attention primitives:

- Causal full self-attention with GQA, RoPE on Q/K, per-head Q/K RMSNorm, output gate, KV cache.
- Gated DeltaNet linear attention with causal conv state and recurrent `[B, V_heads, K_dim, V_dim]` state.
- Vision noncausal packed variable-length self-attention with `cu_seqlens`.

Position/rotary:

- Default RoPE with partial rotary factor 0.25.
- Text M-RoPE interleaving across T/H/W frequency sections.
- Vision 2D RoPE plus learned position embedding interpolation over patch grids.

Generation/cache:

- Hybrid `DynamicCache(config=...)` with per-layer cache type.
- Full-attention KV cache: `[B, num_key_value_heads, past_seq, head_dim]`.
- Linear-attention conv state: `[B, conv_dim, linear_conv_kernel_dim]`.
- Linear-attention recurrent state: `[B, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`.
- Beam reorder must index both KV and linear states along batch.
- `logits_to_keep` for last-token-only or selected-token logits.

Quantized/packed weight metadata:

- FP8 config: dynamic activation scheme, non-per-tensor weight/activation flags, `[128,128]` weight block size, many modules excluded.
- GPTQ int4 config: group size 128, symmetric true, `desc_act=false`, attention/shared/visual/MTP excluded by dynamic patterns.
- First DinoML integration should reject or route quantized variants unless the loader can prove exact tensor metadata and dense fallback.

## 5. Layer/block breakdown

Text decoder layer, repeated `num_hidden_layers`:

```text
residual = x
x = RMSNorm(hidden_size)(x)
if layer_type == "linear_attention":
    qkv = Linear(hidden -> 2 * linear_key_dim + linear_value_dim, bias=False)(x)
    z = Linear(hidden -> linear_value_dim, bias=False)(x)
    b = Linear(hidden -> linear_num_value_heads, bias=False)(x)
    a = Linear(hidden -> linear_num_value_heads, bias=False)(x)
    qkv = depthwise causal Conv1d + SiLU(qkv)
    q, k, v = split(qkv, [key_dim, key_dim, value_dim])
    beta = sigmoid(b)
    g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
    q,k = repeat heads to value-head count if needed
    y, recurrent_state = gated_delta_rule(q,k,v,g,beta,state)
    y = RMSNormGated(head_v_dim)(y, z)
    x = residual + Linear(value_dim -> hidden, bias=False)(y)
else:
    q, gate = split(Linear(hidden -> 2 * num_heads * head_dim, bias=attention_bias)(x))
    k = Linear(hidden -> kv_heads * head_dim, bias=attention_bias)(x)
    v = Linear(hidden -> kv_heads * head_dim, bias=attention_bias)(x)
    q = per-head RMSNorm(head_dim)(q)
    k = per-head RMSNorm(head_dim)(k)
    q,k = RoPE(q,k)
    y = causal GQA(q,k,v,cache)
    y = y * sigmoid(gate)
    x = residual + Linear(num_heads * head_dim -> hidden, bias=attention_bias)(y)
residual = x
x = RMSNorm(hidden_size)(x)
x = residual + SparseMoE(x) + gated_shared_expert(x)
```

MoE block:

```text
tokens = x.view(B*S, H)
router_logits = Linear(H -> num_experts, bias=False)(tokens)
router_probs = softmax(router_logits, fp32)
top_values, top_indices = topk(router_probs, K)
top_values = top_values / sum(top_values)
expert_y = sum_selected_experts(SwiGLU(tokens @ gate_up[e].T) @ down[e].T * top_values)
shared_y = sigmoid(Linear(H -> 1)(tokens)) * DenseSwiGLU(H -> shared_intermediate -> H)(tokens)
y = expert_y + shared_y
```

Vision block:

```text
packed patches -> Conv3d patch embed -> learned pos interpolation add
repeat depth:
  x = x + varlen noncausal attention(LayerNorm(x), cu_seqlens, vision RoPE)
  x = x + MLP(LayerNorm(x))
patch merger: group spatial_merge_size^2 tokens -> LayerNorm -> Linear -> GELU -> Linear(text_hidden)
```

## 6. Attention requirements

Full text attention:

- Causal self-attention only.
- GQA: `num_attention_heads / num_key_value_heads` groups.
- Q width: `num_attention_heads * head_dim`.
- K/V width: `num_key_value_heads * head_dim`.
- Attention output width before output projection: `num_attention_heads * head_dim`.
- Q/K get RMSNorm over `head_dim` before RoPE.
- RoPE is applied before cache update; cached keys are post-RoPE.
- Masking uses Transformers `create_causal_mask`.
- Source dispatches through `ALL_ATTENTION_FUNCTIONS`, so SDPA/FlashAttention can be used when requested; eager fallback repeats KV with `repeat_kv`, computes scaled scores, mask add, softmax upcast to fp32, dropout, and value matmul.

Linear attention:

- Not a KV cache. It is a stateful Gated DeltaNet with a depthwise conv over Q/K/V channels and fixed recurrent state.
- Prefill/chunk decode path uses a chunked gated delta rule with chunk size 64 in the fallback.
- Single-token cached decode uses recurrent update path and `causal_conv1d_update`.
- Linear-attention mask is only used for left-padding zeroing before projection; it is cleared once cache has previous state or all tokens are unmasked.
- Cache state ABI:
  - conv state: `[B, 2 * key_dim + value_dim, conv_kernel]`
  - recurrent state: `[B, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]`
  - state tensors have static addresses in Transformers cache for compile/CUDA graph friendliness.

Vision attention:

- Noncausal self-attention over packed image/video patch tokens.
- Uses `cu_seqlens` for FlashAttention path; non-Flash path splits by packed sample lengths and concatenates outputs.
- QKV are packed in one dense `Linear(dim -> 3*dim, bias=True)` and split in Q,K,V order.

## 7. Position encoding and custom math

Text RoPE:

```python
dim = int(head_dim * partial_rotary_factor)
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2) / dim))
freqs = inv_freq[None, None, :, None] @ position_ids[:, :, None, :]
freqs = freqs.transpose(2, 3)
freqs_t = freqs[0]
for dim_id, offset in [(1, 1), (2, 2)]:
    length = mrope_section[dim_id] * 3
    freqs_t[..., offset:length:3] = freqs[dim_id, ..., offset:length:3]
emb = cat([freqs_t, freqs_t], dim=-1)
cos, sin = cos(emb), sin(emb)
```

Apply RoPE:

```python
q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
k = (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))
```

Position IDs:

- Text-only `Qwen3_5MoeTextModel` creates `[4, B, S]` position IDs, uses row 0 for causal-mask position bookkeeping, and rows 1:3 for RoPE.
- Multimodal `Qwen3_5MoeModel` computes `[3, B, S]` M-RoPE IDs from `mm_token_type_ids` plus image/video grids.
- Video grids are repeated per frame and then treated as temporal slices with `T=1`.
- `rope_deltas` are cached on the model object for incremental generation.

Vision positional math:

- Learned 2D position embedding table is bilinearly interpolated over H/W grid indices.
- Vision RoPE builds 2D row/column rotary frequencies, flattened in spatial-merge-aware order.
- These position paths depend on dynamic `grid_thw`; precompute per admitted grid bucket when possible.

## 8. Preprocessing and input packing

Source model inputs:

- Text: `input_ids`, optional `attention_mask`, optional `position_ids`, optional `inputs_embeds`.
- Multimodal metadata: `mm_token_type_ids` with text `0`, image `1`, video `2`; `image_grid_thw`; `video_grid_thw`.
- Image/video tensors: source docstring says `pixel_values` and `pixel_values_videos`, but the vision patch embed expects processor-packed patch rows that can be viewed as `[-1, C, temporal_patch_size, patch_size, patch_size]`.

Placeholder stitch:

- `image_token_id=248056`, `video_token_id=248057`.
- Source validates that placeholder mask element count equals feature tensor elements.
- Source then uses `inputs_embeds.masked_scatter(mask, features)`.
- DinoML should not admit general boolean scatter for first integration. Use a guarded indexed-row copy:
  - processor must guarantee placeholder token count equals feature row count;
  - placeholder order must match `torch.cat(split_features, dim=0)`;
  - mask must cover complete hidden rows, not arbitrary elements;
  - reject mixed or reordered placeholders unless processor audit proves order.

CPU/data-pipeline ownership:

- Tokenizer/chat template and image/video resizing/packing should remain outside first GPU graph.
- `grid_thw`, `mm_token_type_ids`, and placeholder positions are graph inputs or compile-time bucket metadata.

## 9. Graph rewrite / lowering opportunities

### Rewrite: full-attention q-proj gate split

Source pattern:

```text
Linear(H -> 2 * QW) -> view(..., head_dim * 2) -> chunk(2, dim=-1) -> q, gate
```

Replacement:

```text
PackedGemm -> split q/gate -> q_norm/RoPE/attention -> multiply sigmoid(gate)
```

Preconditions:

- `q_proj.out_features == 2 * num_attention_heads * head_dim`.
- Split is along the last per-head dimension after view, equivalent to paired `[q, gate]` per head.
- Preserve the gate multiply before `o_proj`.

Failure cases: do not rewrite as QKV; K and V are separate projections.

### Rewrite: MoE expert packed gate/up GEMM

Source pattern:

```text
gate, up = linear(tokens_for_expert, gate_up_proj[e]).chunk(2, dim=-1)
expert = down_proj[e](silu(gate) * up)
```

Replacement:

```text
token dispatch by expert -> grouped GEMM packed gate/up -> SwiGLU -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- `gate_up_proj` layout is `[E, 2*I, H]`.
- `down_proj` layout is `[E, H, I]`.
- Top-k weights are renormalized selected router probabilities.
- Scatter-add is along flattened token index.

Failure cases: sparse routing ties, nondeterministic top-k tie behavior, or unsupported top-k values should fall back to eager/reference.

### Rewrite: linear-attention prefill/decode split

Source pattern:

```text
depthwise causal conv -> q/k/v split -> gated delta rule
```

Replacement:

```text
Prefill/chunk kernel for chunked gated delta rule; decode-step kernel for recurrent update
```

Preconditions:

- `linear_conv_kernel_dim` known, normally 4.
- `linear_num_value_heads % linear_num_key_heads == 0` for Q/K repeat.
- State tensors are explicitly allocated and updated in layer order.

Failure cases: left-padding masks in prefill require zeroing hidden states before projections; cache reorder/reset must update both state tensors.

### Rewrite: multimodal masked_scatter to row-copy

Source pattern:

```text
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

Replacement:

```text
placeholder_indices = where(input_ids == image_token_id)
copy_rows(inputs_embeds[placeholder_indices], image_embeds)
```

Preconditions:

- Mask expands complete hidden rows.
- `image_embeds.shape[0] == number_of_placeholder_tokens`.
- Source order of placeholders matches feature concatenation order.
- Same for video.

Failure cases: `inputs_embeds` without `input_ids` uses embedding equality to detect placeholders; reject this path initially.

### Rewrite: vision patch Conv3d to patch Linear

Source pattern:

```text
view(-1, C, Tpatch, Hpatch, Wpatch) -> Conv3d(kernel=stride=patch)
```

Replacement:

```text
flatten patch row -> Linear(C*Tpatch*Hpatch*Wpatch -> vision_hidden) + bias
```

Preconditions:

- Processor emits non-overlapping patch rows in the same flatten order as the source view.
- `kernel_size == stride == [temporal_patch_size, patch_size, patch_size]`.
- `groups == 1`, `dilation == 1`, no padding.

Failure cases: raw image/video tensors with different layout must use faithful Conv3d or processor-owned packing.

## 10. Kernel fusion candidates

Highest priority:

- Hybrid cache ABI: full-attention KV plus linear-attention conv/recurrent states. Without this, decode parity is not meaningful.
- Gated DeltaNet kernels: depthwise causal conv update, chunked prefill, recurrent decode update, RMSNormGated.
- MoE routing and expert GEMM: router softmax/top-k/renorm, grouped expert GEMM, weighted scatter-add.
- Full-attention GQA FlashAttention with post-RoPE cached keys and output gate.

Medium priority:

- RMSNorm `(1 + weight)` and per-head Q/K RMSNorm fusion.
- Q/gate projection fusion and sigmoid-gate multiply before output projection.
- Last-token-only `lm_head` for decode via `logits_to_keep`.
- Vision patch linearization and packed varlen vision attention.

Lower priority:

- Full multimodal M-RoPE grid construction on GPU; this can start as CPU/preprocessing metadata.
- FP8/GPTQ quantized loaders and providers.
- Training-only router auxiliary loss.

## 11. Runtime staging plan

Stage 1: Parse configs and load dense BF16 text-only weights. Reject quantized variants and multimodal pixel inputs.

Stage 2: Single text decoder layer parity for both layer types: one full-attention layer and one linear-attention layer, no cache.

Stage 3: Text-only prefill parity across N layers, including MoE routing and final RMSNorm/lm_head.

Stage 4: Decode parity with hybrid cache state. Implement explicit per-layer cache manifest with `full_attention` KV entries and `linear_attention` conv/recurrent entries.

Stage 5: Optimize kernels: GQA attention, Gated DeltaNet prefill/decode, MoE grouped GEMM.

Stage 6: Add multimodal prefix path with processor-owned packed patches, vision encoder, guarded row-copy stitch, M-RoPE IDs.

Stage 7: Admit quantized variants only after a source-coupled quantization audit and provider plan.

Can stub initially:

- Vision branch, by rejecting `pixel_values`/`pixel_values_videos`.
- Router auxiliary loss.
- Beam search, while still implementing cache reorder before generation parity is marked complete.
- FP8/GPTQ weight formats.

## 12. Parity and validation plan

- Unit parity for RMSNorm `(1 + weight)` against Transformers tensors.
- Unit parity for text RoPE and M-RoPE interleaving with fixed `position_ids`.
- Unit parity for full-attention projection split: confirm `q_proj` gate split and output gate placement.
- Unit parity for Gated DeltaNet fallback math on small shapes, including prefill and single-token decode with carried state.
- Unit parity for linear-attention state updates: conv state and recurrent state shapes/content after prefill and decode.
- Unit parity for router softmax/top-k/renorm, including top-k `8` and `10`.
- Single MoE block parity with deterministic small expert weights and repeated expert hits.
- One full decoder block parity for a full-attention layer and one for a linear-attention layer.
- Text-only prefill logits parity on 35B-shaped small random config.
- Decode token parity for 2-4 steps with cache reorder/reset checks.
- Multimodal row-copy parity once processor audit is available: placeholder counts, feature ordering, `rope_deltas`.

Suggested tolerances:

- fp32 custom math: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 layer parity: `rtol=2e-2`, `atol=2e-2` initially, tighten per kernel.
- MoE routing indices should match exactly for non-tie logits.

## 13. Performance probes

- Text prefill tokens/sec sweep by sequence length: 1k, 8k, 32k, 128k where memory allows.
- Decode tokens/sec sweep by batch size with hybrid cache memory tracking.
- Full-attention layer benchmark separated from linear-attention layer benchmark.
- Gated DeltaNet prefill chunk-size sensitivity and decode-step latency.
- MoE router plus expert dispatch benchmark: top-k 8 vs 10, experts 256 vs 512, token counts per expert.
- Grouped expert GEMM utilization and scatter-add overhead.
- KV cache memory vs linear state memory by layer type.
- `lm_head` full logits vs last-token-only logits.
- Vision encoder throughput by grid size and packed sample count.
- Quantized load/dequant/provider probes for FP8/GPTQ only after admission.

## 14. Skip/defer list

- Training loss and router auxiliary loss for first inference target.
- Quantized FP8/GPTQ checkpoint execution.
- Multimodal processor internals until tokenizer/processor audit.
- General boolean `masked_scatter`.
- Beam search and cache reordering can be deferred for greedy first-token parity, but must be implemented before generation parity.
- Tensor-parallel and pipeline-parallel plans.
- Cache offload/prefetch.
- Vision/video path for text-only first milestone.

## 15. Final implementation checklist

- [ ] Parse `Qwen3_5MoeConfig`, `text_config`, and `vision_config`.
- [ ] Reject unsupported quantized configs unless a loader/provider is selected.
- [ ] Represent `layer_types` in the DinoML model manifest.
- [ ] Load text embeddings and LM head; preserve source weight-key handling and reject unexpected alias assumptions.
- [ ] Implement RMSNorm with `(1 + weight)`.
- [ ] Implement full-attention Q/gate, K, V, Q/K norm, RoPE, GQA attention, output gate, and `o_proj`.
- [ ] Implement hybrid cache manifest: full-attention KV plus linear-attention conv/recurrent states.
- [ ] Implement Gated DeltaNet prefill and decode state update.
- [ ] Implement MoE router softmax/top-k/renorm.
- [ ] Implement packed expert gate/up and down grouped GEMM path.
- [ ] Implement shared expert and shared expert gate.
- [ ] Add text-only one-layer parity tests for full and linear layers.
- [ ] Add prefill logits parity.
- [ ] Add decode cache parity.
- [ ] Audit processor/tokenizer for multimodal placeholder ordering.
- [ ] Add guarded multimodal row-copy stitch.
- [ ] Add vision patch embed/vision attention/merger parity if multimodal is admitted.
- [ ] Benchmark prefill, decode, MoE dispatch, and linear-attention kernels separately.
