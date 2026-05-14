# Laguna Transformers Family Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary public checkpoint [`poolside/Laguna-XS.2`](https://huggingface.co/poolside/Laguna-XS.2). Quantized representative variants: [`poolside/Laguna-XS.2-FP8`](https://huggingface.co/poolside/Laguna-XS.2-FP8), [`poolside/Laguna-XS.2-NVFP4`](https://huggingface.co/poolside/Laguna-XS.2-NVFP4), [`poolside/Laguna-XS.2-INT4`](https://huggingface.co/poolside/Laguna-XS.2-INT4). No gated configs were encountered.

Config source: raw Hub `config.json` snapshots saved beside this report:

- `poolside-Laguna-XS.2-config.json`
- `poolside-Laguna-XS.2-FP8-config.json`
- `poolside-Laguna-XS.2-NVFP4-config.json`
- `poolside-Laguna-XS.2-INT4-config.json`
- tokenizer/generation snapshots for the BF16 checkpoint are also saved.

Source files inspected:

- `src/transformers/models/laguna/modular_laguna.py`: authoritative source for future Transformers edits.
- `src/transformers/models/laguna/modeling_laguna.py`: generated materialized implementation used for runtime behavior.
- `src/transformers/models/laguna/configuration_laguna.py`: generated config class.
- `docs/source/en/model_doc/laguna.md`: official model notes.
- `tests/models/laguna/test_modeling_laguna.py`: source-level coverage and skipped integration status.
- Inherited/generated source references from Qwen2-MoE, Qwen3-MoE, Qwen3.5-MoE, Gemma3 RoPE, Llama attention helpers, and Transformers masking/cache utilities as pulled into `modeling_laguna.py`.

Any missing files or assumptions: The Hub checkpoint includes `auto_map` entries for remote `configuration_laguna.py` and `modeling_laguna.py`, but this report is scoped to the native in-library source at the pinned commit. Treat remote-code drift as out of scope until separately diffed. No processor/image/audio files are relevant; this is a text causal LM.

## 2. High-level architecture

Laguna is a decoder-only MoE causal language model for text generation.

Dataflow:

```text
tokenizer/chat template -> input_ids/attention_mask -> token embedding
  -> 40 decoder blocks with full or sliding causal attention and dense/sparse MLP
  -> final RMSNorm -> LM head -> logits -> generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, BOS/EOS/PAD handling, optional FIM/tool/thinking special tokens.
- Prefill: full sequence embeddings, causal or sliding masks, per-layer RoPE, attention, MoE routing, LM head.
- Decode: one or a few new tokens with per-layer KV cache. Full-attention layers attend to all prior tokens; sliding layers use a local causal window.
- Independently optimizable regions: RMSNorm, Q/K/V projections, RoPE, attention, attention-output head gate, MoE router, grouped experts, shared expert, LM head.

Primary DinoML target for this report: `LagunaForCausalLM` inference, prefill plus decode with cache. Training loss, router auxiliary loss, gradient checkpointing, and output recording are optional/deferred.

## 3. Important config dimensions

Base source defaults from `LagunaConfig` are similar to XS.2 but differ in RoPE and layer pattern defaults: config defaults use all full attention and first layer dense/rest sparse unless checkpoint config overrides them.

| Field | XS.2 BF16 config value | Source default / notes |
| --- | ---: | --- |
| `vocab_size` | 100352 | same default |
| `hidden_size` | 2048 | same default |
| `num_hidden_layers` | 40 | same default |
| `num_attention_heads` | 48 | fallback when per-layer list omitted |
| `num_attention_heads_per_layer` | 10 layers use 48, 30 layers use 64 | length must equal 40 |
| `num_key_value_heads` | 8 | fixed KV cache head count |
| `head_dim` | 128 | explicit; do not infer from hidden size |
| Q projection width | 6144 on full layers, 8192 on sliding layers | `num_heads[layer] * head_dim` |
| K/V projection width | 1024 | `num_key_value_heads * head_dim` |
| Attention output width before `o_proj` | 6144 or 8192 | layer-dependent |
| `intermediate_size` | 8192 | dense SwiGLU layer 0 |
| `num_experts` | 256 | sparse layers |
| `num_experts_per_tok` | 8 | top-k router |
| `moe_intermediate_size` | 512 | each routed expert |
| `shared_expert_intermediate_size` | 512 | sparse block shared expert |
| `mlp_layer_types` | 1 dense, 39 sparse | source default same pattern |
| `layer_types` | 10 full, 30 sliding in 1:3 pattern | source default is all full |
| `sliding_window` | 512 | used only by sliding layers |
| `max_position_embeddings` | 131072 | source default same |
| Full-layer RoPE | YaRN, theta 500000, factor 32, partial rotary 0.5 | checkpoint override |
| Sliding-layer RoPE | default, theta 10000, partial rotary 1.0 | checkpoint override/default |
| `attention_bias` | false | Q/K/V/O no bias; gate no bias |
| `torch_dtype` | bfloat16 | config metadata |
| `tie_word_embeddings` | false | source still declares possible tied key; checkpoint untied |
| `use_cache` | true | decode cache expected |

Representative checkpoint sweep:

| Model | Structural dims | Quantization/config variation | DinoML admission note |
| --- | --- | --- | --- |
| `poolside/Laguna-XS.2` | 40 layers, 2048 hidden, 8 KV heads, 48/64 query heads, 256 experts, top-8 | BF16 weights; no `quantization_config` | First dense-weight parity target. |
| `poolside/Laguna-XS.2-FP8` | Same neural structure | `compressed-tensors`, `float-quantized`, 8-bit float weights/activation scheme, KV cache FP8 scheme, Hadamard transform metadata | Requires compressed-tensors/FP8 loader or reject/fallback. |
| `poolside/Laguna-XS.2-NVFP4` | Same neural structure | `compressed-tensors`, `nvfp4-pack-quantized`, 4-bit float weights/activations, KV cache FP8 scheme | Requires NVFP4 packed provider; reject initially. |
| `poolside/Laguna-XS.2-INT4` | Same neural structure | `compressed-tensors`, `pack-quantized`; layers 1-30 target INT4 MLP/expert weights, layers 31-39 INT8; attention and selected modules ignored; Hadamard transform metadata | Weight names/ignore regexes need exact source mapping before support. |

## 3a. Family variation traps

- `hidden_size != num_attention_heads_per_layer * head_dim`. The residual stream is 2048, but attention inner width is 6144 or 8192.
- Query heads vary by layer, while KV heads remain 8. Full layers use GQA group count 6; sliding layers use group count 8.
- `layer_types` controls both mask construction and RoPE parameter choice.
- Source defaults are all full attention; public XS.2 config is mixed full/sliding in a repeating 1 full + 3 sliding pattern.
- `rope_parameters` are nested by layer type. Generic flat `rope_scaling` handling is intentionally skipped in tests.
- Full-attention checkpoint RoPE is YaRN with partial rotary factor 0.5; sliding RoPE is default with full rotary factor 1.0.
- Attention output has a per-head `softplus(g_proj(x))` gate after attention and before `o_proj`.
- MLP layer 0 is dense SwiGLU; layers 1-39 are sparse MoE plus shared expert.
- Router uses sigmoid scores, top-k on score plus correction bias, gather, sum normalization, and no softmax for inference routing.
- `moe_apply_router_weight_on_input=True` raises during config validation and should be rejected.
- `gating`, `partial_rotary_factor`, and `use_bidirectional_attention` appear in checkpoint configs but are not read by the inspected native source as top-level runtime controls. RoPE partial factors are read only inside nested `rope_parameters`.
- Quantized variants advertise compressed-tensors formats that are loading/provider contracts, not new neural graph ops.
- No NHWC/NCHW layout translation is relevant for the text graph. Axis-sensitive ops are sequence/head/hidden reshapes and `dim=-1` reductions/top-k/gather.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup `[B,T] -> [B,T,2048]`.
- Shape/view/reshape for `[B,T,width] -> [B,T,H,D]`, transposes between `[B,T,H,D]` and `[B,H,T,D]`, contiguous materialization after attention.
- `arange`, unsqueeze, broadcast, cat for RoPE and masks.
- Slice/index for `logits_to_keep`, last-token-only logits, and cache position.

Neural network primitives:

- RMSNorm over last dim for hidden size 2048 and head dim 128, fp32 variance/rsqrt, output cast back to input dtype.
- Linear/GEMM no-bias for embeddings/LM head, attention Q/K/V/O/gate, dense MLP, shared MLP, router, expert packed projections.
- SwiGLU: `silu(gate_proj(x)) * up_proj(x) -> down_proj`.
- Softplus attention gate: `softplus(g_proj(x).float()).to(dtype)`.
- Residual adds.

Attention primitives:

- Causal dense self-attention.
- Sliding-window causal self-attention with window 512.
- GQA/MQA-style KV repeat from 8 KV heads to 48 or 64 query heads.
- RoPE on Q/K before cache update.
- KV cache update per layer, keys/values stored as `[B,8,T,128]` before repeat.
- SDPA/FlashAttention/flex-compatible path; eager fallback uses matmul, mask add, fp32 softmax, dropout disabled in inference, matmul with V.

MoE/routing ops:

- Router linear `[tokens,2048] x [256,2048]`.
- Optional tanh softcap on router logits.
- Sigmoid, add correction bias for selection only, `topk(k=8)`, gather, normalize by selected-score sum.
- One-hot/per-expert token dispatch in eager source; optimized DinoML should lower to grouped expert GEMM/scatter-add.
- Expert packed `gate_up_proj[expert]` layout `[2 * 512, 2048]`, split as gate then up; `down_proj[expert]` layout `[2048,512]`.
- Weighted `index_add` back to token order.
- Shared expert `Linear(2048 -> 512)`, `Linear(2048 -> 512)`, `Linear(512 -> 2048)` added to routed expert output.

Position/rotary ops:

- Per-layer-type inv frequency tables.
- Full layer: rotary dimension `128 * 0.5 = 64`; sliding layer: rotary dimension 128.
- YaRN support required for public BF16 full layers; default RoPE required for sliding layers.

Generation/cache ops:

- Dynamic cache construction when `use_cache` is true.
- Position id generation from cache length.
- Per-layer cache update and beam/cache reorder later through Transformers cache ABI.
- LM head `Linear(2048 -> 100352)` no bias; untied for XS.2.

Quantized/packed weight metadata ops:

- `compressed-tensors` FP8/NVFP4/INT4 configs should be treated as weight-loading/provider formats. Initial DinoML should reject quantized variants unless the compressed-tensors transform, ignored-module regexes, scales, packed data, and optional Hadamard transforms are explicitly supported.

Preprocessing-coupled ops:

- No image/audio processor.
- Text tokenizer has custom special tokens, FIM tokens, assistant/tool/thinking marker tokens, EOS ids `[2,24]`, pad id `9`, and generation defaults `temperature=0.7`, `top_p=0.9`, `max_new_tokens=2048`.

## 5. Layer/block breakdown

Decoder block, repeated 40 times:

```text
residual = x
x = RMSNorm_2048(x)
q = Linear(2048 -> num_heads[layer] * 128, bias=False)(x)
k = Linear(2048 -> 8 * 128, bias=False)(x)
v = Linear(2048 -> 8 * 128, bias=False)(x)
q = RMSNorm_128(q.view(B,T,Hq,128)).transpose(1,2)
k = RMSNorm_128(k.view(B,T,8,128)).transpose(1,2)
v = v.view(B,T,8,128).transpose(1,2)
q,k = partial_or_full_RoPE(q,k, layer_type)
k,v = cache.update(k,v, layer_idx) if cache enabled
attn = causal_or_sliding_attention(q,k,v, mask, scale=1/sqrt(128))
attn = attn.reshape(B,T,Hq*128)
gate = softplus(Linear(2048 -> Hq, bias=False)(x).float()).to(dtype)
attn = (attn.view(B,T,Hq,128) * gate[...,None]).reshape(B,T,Hq*128)
x = residual + Linear(Hq*128 -> 2048, bias=False)(attn)

residual = x
x = RMSNorm_2048(x)
if dense layer:
    x = Linear(8192 -> 2048)(silu(Linear(2048 -> 8192)(x)) * Linear(2048 -> 8192)(x))
else:
    shared = SwiGLU(2048 -> 512 -> 2048)(x)
    router = sigmoid(Linear(2048 -> 256)(x).float())
    experts = top8_grouped_experts(x, router)
    x = experts * moe_routed_scaling_factor + shared
x = residual + x
```

Full layers in XS.2: indices 0, 4, 8, ..., 36 use `Hq=48`, Q/O inner width 6144, full causal mask, YaRN partial RoPE over 64 dims.

Sliding layers in XS.2: remaining 30 layers use `Hq=64`, Q/O inner width 8192, sliding causal mask with window 512, default RoPE over 128 dims.

## 6. Attention requirements

- Causal self-attention only; no cross-attention.
- Mixed dense/full and sliding-window causal attention by layer.
- GQA with variable query-head count: query heads 48 or 64, KV heads 8, head dim 128.
- Query/key/value widths: Q is 6144 or 8192, K/V are 1024, attention output before `o_proj` is 6144 or 8192.
- Masking: Transformers `create_causal_mask` for full layers and `create_sliding_window_causal_mask` for sliding layers. Masks are added before softmax in eager path.
- Cache: per-layer K/V tensors hold un-repeated KV heads after RoPE and before GQA repeat. Effective cache shape per layer is `[B,8,total_seq,128]` for both layer types.
- Sliding-window admission: generated attention must enforce `sliding_window=512` and layer-type mask semantics. Full layers must not be accidentally truncated to the local window.
- FlashAttention/SDPA compatibility: source dispatches through `ALL_ATTENTION_FUNCTIONS`; attention-output gating is outside the backend call. DinoML can initially use an eager-equivalent dense/local attention path, then substitute optimized attention if mask/window/cache semantics match.
- Dropout is zero for inference.

## 7. Position encoding and custom math

RoPE is computed once per layer type per forward from shared `position_ids` and then reused by layers of that type. Full and sliding layers can have different frequency tables and rotary dimensions.

Short implementation sketch:

```python
def laguna_rope_inv_freq(config, layer_type):
    params = config.rope_parameters[layer_type]
    dim = int(config.head_dim * params.get("partial_rotary_factor", 1.0))
    base = params["rope_theta"]
    return 1.0 / (base ** (arange(0, dim, 2).float() / dim))

def apply_laguna_rope(q, k, cos, sin):
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q_embed, q_pass], -1), cat([k_embed, k_pass], -1)
```

Custom math to preserve:

- Full-layer YaRN RoPE from Transformers `ROPE_INIT_FUNCTIONS["yarn"]`, not just default theta.
- Sliding-layer default RoPE with theta 10000 and full head dim.
- Q/K RMSNorm happens before RoPE and after reshape to head dimension.
- Router softcap if `moe_router_logit_softcapping > 0`:

```python
router_logits = tanh(router_logits / cap) * cap
```

- Router selection uses `sigmoid(router_logits) + e_score_correction_bias`; routing weights gather from raw sigmoid scores and normalize only selected scores.
- Attention gate:

```python
gate = softplus(g_proj(hidden_states).float()).to(attn_output.dtype)
attn_output = attn_output.view(B,T,Hq,128) * gate[..., None]
```

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- `PreTrainedTokenizerFast`, custom chat template, and special token handling.
- Config/tokenizer BOS and EOS both use token id `2` for `〈|EOS|〉`; generation EOS ids are `[2,24]`, where `24` is `</assistant>`.
- PAD id is `9`; UNK id is `0`; FIM tokens include ids 6, 7, and 11.
- Generation config defaults to sampling enabled, temperature 0.7, top-p 0.9, max new tokens 2048.

GPU/runtime graph inputs:

- `input_ids: int64 [B,T]` or `inputs_embeds: [B,T,2048]`, exactly one.
- Optional `attention_mask`; source can also accept a precomputed dict mapping layer types to masks, but DinoML should initially own mask generation from ordinary attention masks.
- Optional `position_ids`; if absent, generated from sequence length plus cache length.
- Optional `past_key_values` cache.

No multimodal placeholder stitch, no masked scatter, no processor-derived grid metadata, and no layout translation candidate.

## 9. Graph rewrite / lowering opportunities

### Rewrite: attention projections as independent GEMMs

Source pattern: Q, K, V are separate no-bias linear layers with different output widths.

Replacement pattern: three GEMMs or a fused multi-output projection only if weights are packed explicitly by DinoML.

Preconditions:

- `attention_bias == False`.
- Layer-local `Hq` known from `num_attention_heads_per_layer`.
- K/V output width fixed at 1024.

Shape equations:

- `Q: [B*T,2048] x [2048,Hq*128]`.
- `K,V: [B*T,2048] x [2048,1024]`.

Failure cases: do not assume one uniform Q width across all layers; do not pack QKV using hidden-size-derived head count.

Parity test sketch: compare Q/K/V tensors after reshape and Q/K RMSNorm for both a full layer and sliding layer.

### Rewrite: eager MoE loop to grouped expert GEMM

Source pattern: top-k routing, expert-wise token selection, per-expert gate/up/down projections, weighted `index_add`.

Replacement pattern: route tokens into grouped expert batches, run packed `gate_up` grouped GEMM, SiLU/mul, grouped down GEMM, weighted scatter-add.

Preconditions:

- `num_experts=256`, `top_k=8`, expert hidden/intermediate dimensions match config.
- Expert packed weight layout is `[expert, 2*intermediate, hidden]`, split gate then up.
- Router semantics match sigmoid + correction-bias selection + selected-score normalization.

Failure cases: `moe_apply_router_weight_on_input=True` unsupported; quantized expert weights require separate provider.

Parity test sketch: random small config with deterministic router weights; compare selected expert ids, routing weights, and sparse block output.

### Rewrite: attention-output gate fusion

Source pattern: attention backend output `[B,T,Hq,128]`, `softplus(g_proj(x).float())`, multiply per head, flatten, `o_proj`.

Replacement pattern: fuse softplus and per-head multiply into attention-output epilogue or pre-`o_proj` elementwise kernel.

Preconditions:

- `g_proj` output width equals layer query-head count.
- Gate input is the normalized attention input hidden state, not attention output.

Failure cases: if attention backend emits flattened `[B,T,Hq*128]`, the gate multiply must preserve head grouping.

Parity test sketch: compare pre-`o_proj` gated attention tensor for Hq 48 and 64.

### Rewrite: RoPE table precompute by layer type

Source pattern: compute cos/sin for each distinct layer type per forward.

Replacement pattern: cache or precompute inv frequencies and build cos/sin once per layer type and sequence/cache position.

Preconditions:

- Respect dynamic RoPE types such as YaRN; position IDs and max sequence changes are inputs.
- Full and sliding layer types must remain separate.

Failure cases: using full-layer partial rotary factor on sliding layers, or default RoPE for full YaRN layers.

Parity test sketch: compare cos/sin and post-RoPE Q/K for positions around 4096 and long-context positions.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])` where `logits_to_keep` can be integer or tensor indices.

Replacement pattern: for decode and common generation, only run LM head on last token.

Preconditions:

- `logits_to_keep == 1` or equivalent last-token slice.
- No loss computation requiring all logits.

Failure cases: prompt logprob/evaluation paths asking for full sequence logits.

Parity test sketch: compare logits for `logits_to_keep=1`, `0`, and explicit tensor indices.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden and head-dim cases. It is on every residual path and Q/K projection path.
- GQA/sliding FlashAttention with RoPE-applied K cache and variable query head count. This is the decode throughput center.
- Grouped MoE expert GEMM and scatter-add. Eager per-expert loops are not production viable.
- SwiGLU dense/shared expert fusion.
- Router top-k pipeline: linear, optional softcap, sigmoid, bias-add selection, top-k, gather, normalize.

Medium priority:

- Attention-output softplus gate fused with reshape/flatten before `o_proj`.
- Q/K/V projection scheduling with per-layer Q width awareness.
- RoPE kernel supporting partial rotary and YaRN/default tables.
- Last-token-only LM head and optional vocab sharding.
- Sliding-window mask/cache trimming for decode.

Lower priority:

- Router auxiliary loss and router logits output recording.
- Flex attention parity beyond dense/full and local-window causal paths.
- Quantized compressed-tensors providers for FP8/NVFP4/INT4 variants.
- Tensor-parallel plans from Transformers metadata.

## 11. Runtime staging plan

Stage 1: Parse config and load BF16 dense weights for `poolside/Laguna-XS.2`; reject quantized variants with a clear compressed-tensors unsupported message.

Stage 2: Implement small-config block parity with dense MLP only, RMSNorm, variable-width attention projections, RoPE, and eager dense causal attention.

Stage 3: Add mixed full/sliding attention masks and per-layer RoPE parameter selection; validate one full layer and one sliding layer.

Stage 4: Add sparse MoE parity with top-k router, grouped expert lowering, shared expert, and routed scaling.

Stage 5: Prefill parity for the full 40-layer BF16 checkpoint at short sequence lengths.

Stage 6: Decode parity with `DynamicCache`-equivalent K/V tensors, position-id offset, full versus sliding cache behavior, and last-token logits.

Stage 7: Replace eager attention with optimized full/sliding attention backends and add grouped expert provider tuning.

Stage 8: Add optional compressed-tensors admission for FP8/NVFP4/INT4 only after storage, scale, ignored-module, transform, and KV-cache quantization contracts are explicit.

Initially stubbable: training loss, router auxiliary loss, output hidden states/attentions/router logits, tensor parallelism, beam-search reorder, quantized checkpoints.

## 12. Parity and validation plan

- Config validation tests: list lengths, `moe_apply_router_weight_on_input=True` rejection, per-layer head divisibility by KV heads.
- RMSNorm random tensor tests for `[B,T,2048]` and `[B,T,H,128]`, fp32/fp16/bf16.
- RoPE tests for full YaRN partial-64 and sliding default full-128, including long positions beyond 4096.
- Attention projection and Q/K norm tests for a full layer (`Hq=48`) and sliding layer (`Hq=64`).
- Attention tests: eager full causal, sliding window 512, prefill and one-token decode with cache.
- Router tests: top-k ids, routing weights, score correction bias effect, optional logit softcap.
- Expert tests: packed `gate_up_proj` split order and weighted scatter-add.
- Single decoder layer parity: layer 0 dense MLP and layer 1 sparse MoE.
- After-N-layer parity: 2-layer mixed full/sliding mini config, then all 40 layers on tiny sequence if memory allows.
- Prefill logits parity for BF16 checkpoint with `logits_to_keep=1` and full logits.
- Decode token parity for greedy generation for a short prompt.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 block-level `rtol=2e-2, atol=2e-2` initially, tightened per fused kernel once accumulation order is fixed. MoE routing id parity should be exact except for intentional tie cases.

## 13. Performance probes

- Tokenizer/chat-template throughput separately from model runtime.
- Prefill throughput by sequence length: 512, 4096, 32768, 131072 where feasible.
- Decode tokens/sec with cache for batch sizes 1, 4, 16.
- Full-layer versus sliding-layer attention backend comparison.
- Sliding window sweep around 512 and short-context fallback behavior.
- KV cache memory by batch and generated length: 40 layers * 2 tensors * 8 heads * 128 head dim.
- Router/top-k latency and selected-expert load distribution.
- Grouped expert GEMM occupancy by batch/sequence and active expert histogram.
- Shared expert versus routed expert time split.
- LM head time for full logits versus last-token-only.
- Dense BF16 load time and memory footprint.
- Quantized provider probe later: FP8/NVFP4/INT4 load/dequant/GEMM paths versus BF16 baseline.

## 14. Skip/defer list

- Training loss, router auxiliary loss, and gradient checkpointing.
- Output recording for hidden states, attentions, and router logits.
- Quantized compressed-tensors checkpoints until provider/storage support exists.
- Tensor parallelism and pipeline parallelism.
- Beam-search cache reorder beyond a simple decode cache ABI.
- Flex attention-specific behavior if SDPA/FlashAttention-compatible full and sliding causal paths are available.
- Remote-code checkpoint drift from the native source basis.
- General layout translation work; this family is text-only.

## 15. Final implementation checklist

- [ ] Parse `LagunaConfig`, including nested `rope_parameters`, `layer_types`, `mlp_layer_types`, and `num_attention_heads_per_layer`.
- [ ] Reject unsupported config flags such as `moe_apply_router_weight_on_input=True`.
- [ ] Load BF16 dense XS.2 weights with untied embedding and LM head.
- [ ] Add clear rejection/admission policy for `compressed-tensors` quantized variants.
- [ ] Implement RMSNorm for hidden size 2048 and head dim 128.
- [ ] Implement per-layer Q/K/V/O projections with variable Q/O widths.
- [ ] Implement Laguna RoPE with full/sliding layer-type parameter dispatch and partial rotary.
- [ ] Implement causal full attention and sliding-window causal attention.
- [ ] Implement KV cache shape `[B,8,T,128]` per layer and decode position offsets.
- [ ] Implement attention-output softplus head gate.
- [ ] Implement dense SwiGLU MLP for layer 0.
- [ ] Implement sigmoid top-k MoE router with correction bias and selected-score normalization.
- [ ] Implement packed expert grouped GEMM and weighted scatter-add.
- [ ] Implement shared expert and routed scaling factor.
- [ ] Implement LM head with `logits_to_keep` optimization.
- [ ] Add one-block, two-block, prefill, and decode parity tests.
- [ ] Benchmark full/sliding attention, MoE router/experts, LM head, and KV cache memory.
