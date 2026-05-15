# GPT-OSS Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openai/gpt-oss-20b, openai/gpt-oss-120b; tiny-random/gpt-oss as debug-sized config only
Config source: HF config.json snapshots saved under _sources/
Source files inspected:
- transformers/src/transformers/models/gpt_oss/configuration_gpt_oss.py
- transformers/src/transformers/models/gpt_oss/modular_gpt_oss.py
- transformers/src/transformers/models/gpt_oss/modeling_gpt_oss.py
- transformers/src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py
- transformers/src/transformers/integrations/mxfp4.py
- transformers/src/transformers/quantizers/quantizer_mxfp4.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/cache_utils.py
Any missing files or assumptions: no remote-code model file is required for the official OpenAI configs. The generated modeling file is runtime-authoritative, but its header says future source edits must be made in modular_gpt_oss.py.
```

Snapshots written for this audit:

- `_sources/configuration_gpt_oss.py`
- `_sources/modular_gpt_oss.py`
- `_sources/modeling_gpt_oss.py`
- `_sources/openai_gpt-oss-20b_config.json`
- `_sources/openai_gpt-oss-120b_config.json`
- `_sources/tiny-random_gpt-oss_config.json`
- `_sources/*generation_config.json`
- `_sources/*model.safetensors.index.json`
- `_sources/*original_dtypes.json`

The official OpenAI checkpoint configs include `quantization_config.quant_method = "mxfp4"` and exclude attention, router, embeddings, and lm head from quantization. The in-library source implements the dense BF16 expert module and a separate MXFP4 replacement path in `integrations/mxfp4.py`; DinoML should treat MXFP4 as a source-coupled weight/storage path, not a normal dense Linear variant.

## 2. High-level architecture

GPT-OSS is a text-only causal LM with a decoder-only MoE Transformer body:

```text
token ids -> token embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Each decoder block is:

```text
RMSNorm -> GQA self-attention with RoPE, attention sinks, full/sliding causal mask, KV cache
residual add
RMSNorm -> top-k router -> sparse MoE expert MLP -> residual add
```

Primary runtime target for DinoML should be `GptOssForCausalLM` prefill and decode. `GptOssModel` is the base decoder. Sequence and token classification heads exist through generic wrappers, but they are optional/deferred for the generation target.

Independently stageable pieces:

- CPU/data pipeline: tokenizer/chat template, `input_ids`, optional `attention_mask`, `position_ids`.
- GPU/runtime prefill: embedding, alternating sliding/full causal masks, RoPE, GQA attention, MoE routing/expert dispatch, logits.
- GPU/runtime decode: one-token or short-token forward with mixed full/sliding cache layers.
- Optimization-only: MXFP4 expert kernels, FlashAttention/flex/paged attention with sink support, last-token-only logits.

## 3. Important config dimensions

| Field | Source default | openai/gpt-oss-20b | openai/gpt-oss-120b | tiny-random/gpt-oss |
| --- | ---: | ---: | ---: | ---: |
| `hidden_size` | 2880 | 2880 | 2880 | 32 |
| `num_hidden_layers` | 36 | 24 | 36 | 2 |
| `num_attention_heads` | 64 | 64 | 64 | 2 |
| `num_key_value_heads` | 8 | 8 | 8 | 1 |
| `head_dim` | 64 | 64 | 64 | 32 |
| Q projection width | 4096 | 4096 | 4096 | 64 |
| K/V projection width each | 512 | 512 | 512 | 32 |
| GQA repeat factor | 8 | 8 | 8 | 2 |
| `intermediate_size` | 2880 | 2880 | 2880 | 64 |
| dense expert count | 128 | 32 | 128 | 32 |
| experts per token | 4 | 4 | 4 | 4 |
| `vocab_size` | 201088 | 201088 | 201088 | 201088 |
| `max_position_embeddings` | 131072 | 131072 | 131072 | 131072 |
| `sliding_window` | 128 | 128 | 128 | 128 |
| layer types | alternating sliding/full | 24 alternating | 36 alternating | 2 alternating |
| attention bias | true | true | true | true |
| tied embeddings | false | false | false | true |
| quantization | source default none | MXFP4 experts | MXFP4 experts | none in config |

RoPE settings from official configs:

| Field | Value |
| --- | --- |
| `rope_theta` | 150000 |
| `rope_scaling.rope_type` | `yarn` |
| `factor` | 32.0 |
| `beta_fast` | 32.0 |
| `beta_slow` | 1.0 |
| `original_max_position_embeddings` | 4096 |
| `truncate` | false |

Generation config facts:

| Model | `bos_token_id` | `eos_token_id` | `pad_token_id` | Sampling |
| --- | ---: | --- | ---: | --- |
| OpenAI 20B/120B | 199998 | `[200002, 199999, 200012]` | 199999 | `do_sample=true` |
| tiny-random | 199998 | `[200002, 199999]` | 199999 | `do_sample=true`, `trust_remote_code=true` in generation config only |

Note: the official config JSON uses `experts_per_token` and `num_experts_per_tok`; the source reads `num_experts_per_tok`. The JSON also uses `rope_scaling`, while the config class default field is named `rope_parameters`; the converter handles old/new original formats, and the runtime RoPE object reads `config.rope_parameters`.

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim`: hidden is 2880 but attention output width before `o_proj` is 4096. Do not infer Q/O widths from hidden size alone.
- GQA is required: K/V heads are 8 while Q heads are 64 for OpenAI configs.
- Alternating attention is required: layer 0 is `sliding_attention`, layer 1 is `full_attention`, and so on.
- Sliding-window KV cache layers are not the same shape/lifetime as full cache layers.
- Attention has learned per-head sink logits. Fused attention must include the sink in softmax normalization and then drop the sink probability before value matmul.
- The dense expert source stores `gate_up_proj` as `[num_experts, hidden_size, 2 * intermediate_size]` and splits output by interleaved even/odd columns, not contiguous gate/up halves.
- MXFP4 checkpoints store expert weights as `_blocks` and `_scales` tensors plus BF16/float biases. The runtime replacement module swizzles those into Triton-kernel layout.
- Router `topk` uses logits directly, then softmax over the selected top-k values only. It does not softmax over all experts first in inference.
- Official OpenAI configs set `router_aux_loss_coef=0.9`, while source default is `0.001`; this matters only when training/loss or router aux outputs are enabled.
- `swiglu_limit=7.0` is in official configs and the MXFP4 module reads it; dense source uses hard-coded `limit=7.0`.
- `tie_word_embeddings=false` for official OpenAI configs despite `_tied_weights_keys` being present on the LM class. Do not alias embed/lm_head unless config enables it.
- `_supports_sdpa = False`; source optimized paths are eager fallback, FlashAttention/flex/paged backends with sink support, or hub kernels.
- No NCHW/NHWC layout translation is relevant; this is a text-only sequence model. Protect sequence/head/layout reshapes from generic layout passes.

## 4. Operator coverage checklist

Tensor/layout ops:

- token embedding lookup `[B, S] -> [B, S, H]`
- reshape `[B, S, H] -> [B*S, H]` for router/expert dispatch
- view/transpose for attention `[B, S, heads*D] -> [B, heads, S, D]`
- concat/split/chunk on last dim for RoPE and expert gate/up split
- `index_add_`/scatter-add for dense fallback MoE
- top-k indices, one-hot, nonzero, where/gather/scatter for fallback routing
- last-token or selected-token slicing for `logits_to_keep`

Neural primitives:

- RMSNorm with fp32 variance accumulation and output cast back to input dtype
- Linear with bias for Q/K/V/O and router
- bias-free Linear for LM head
- residual add
- clamp, sigmoid, multiply, add for custom gated expert activation
- softmax over router top-k values

Attention primitives:

- causal self-attention
- GQA repeat K/V from 8 KV heads to 64 Q heads
- alternating full causal and sliding-window causal masks
- RoPE on Q/K before cache update
- attention sink logits per head
- KV cache update/reorder for mixed full/sliding cache layers
- FlashAttention/flex/paged candidates must support `s_aux` sink and local window metadata

Position/rotary ops:

- YaRN RoPE parameter initialization through `ROPE_INIT_FUNCTIONS`
- default RoPE fallback with `rope_theta`
- fp32 position/frequency matmul, cos/sin, cast to model dtype
- half-split rotation, not pairwise-even/odd rotation

MoE/routing ops:

- router Linear `hidden_size -> num_local_experts`
- `topk(k=4)` over experts
- softmax over selected top-k values
- per-token expert dispatch
- grouped expert GEMM for `hidden -> 2 * intermediate`
- interleaved gate/up split
- custom gated activation
- grouped expert GEMM for `intermediate -> hidden`
- weighted accumulation back to token order

Quantization/packed metadata ops:

- MXFP4 expert `_blocks`/`_scales` load path
- FP4 LUT decode with scale exponent when dequantizing
- Triton swizzle path for native MXFP4 expert GEMM
- dense BF16 dequant fallback when MXFP4 kernels are unavailable or explicitly disabled

Generation/cache ops:

- DynamicCache construction from `config.layer_types`
- per-layer full/sliding cache update
- position id generation from `past_key_values.get_seq_length()`
- cache reorder for beam/search paths
- `logits_to_keep` integer or tensor slicing

Distributed/tensor-parallel ops, optional/deferred:

- source declares expert-parallel and tensor-parallel plans for router/grouped GEMMs and lm_head.
- `routing_torch_dist` handles local expert ranges under initialized `torch.distributed`.

## 5. Layer/block breakdown

For OpenAI 20B/120B layer with `H=2880`, `A=64`, `KV=8`, `D=64`, `I=2880`, `E=32 or 128`, `K=4`:

```text
x: [B, S, 2880]

residual = x
x = RMSNorm(x)
q = Linear(2880 -> 4096, bias=True)(x).view(B, S, 64, 64).transpose(1, 2)
k = Linear(2880 -> 512, bias=True)(x).view(B, S, 8, 64).transpose(1, 2)
v = Linear(2880 -> 512, bias=True)(x).view(B, S, 8, 64).transpose(1, 2)
q, k = half-split RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache is present
attn = causal/sliding GQA attention(q, k, v, mask, sinks)
x = residual + Linear(4096 -> 2880, bias=True)(attn)

residual = x
x = RMSNorm(x)
flat = x.reshape(B*S, 2880)
router_logits = Linear(2880 -> E, bias=True)(flat)
router_values, router_indices = topk(router_logits, k=4)
router_scores = softmax(router_values, dim=-1)
for selected experts:
    gate_up = flat_selected @ gate_up_proj[e] + gate_up_proj_bias[e]
    gate = gate_up[..., 0::2]
    up = gate_up[..., 1::2]
    gate = clamp(gate, max=7.0)
    up = clamp(up, min=-7.0, max=7.0)
    y = (up + 1) * gate * sigmoid(1.702 * gate)
    y = y @ down_proj[e] + down_proj_bias[e]
    scatter_add(y * router_score)
x = residual + routed.reshape(B, S, 2880)
```

Weight/layout details:

- Dense `GptOssExperts.gate_up_proj`: `[E, 2880, 5760]`, used as `current_state @ gate_up_proj[e]`.
- Dense `GptOssExperts.down_proj`: `[E, 2880, 2880]`, used as `gated_output @ down_proj[e]`.
- MXFP4 `Mxfp4GptOssExperts.gate_up_proj` initial parameter placeholder: `[E, 5760, 90, 16]` uint8 for OpenAI configs.
- MXFP4 `down_proj` placeholder: `[E, 2880, 90, 16]` uint8 for OpenAI configs.
- MXFP4 `_blocks`/`_scales` are reshaped/swizzled and assigned as Triton tensor wrappers before forward.

## 6. Attention requirements

GPT-OSS requires autoregressive self-attention only. There is no cross-attention or encoder cache for the primary target.

Attention facts:

- Causal: yes.
- Self-attention: yes.
- MHA/MQA/GQA: GQA, 64 query heads, 8 KV heads, repeat factor 8 for OpenAI configs.
- Head dim: 64 for OpenAI configs.
- Scaling: `head_dim ** -0.5`.
- Dropout: 0.0 for inference.
- Masking: source builds a mapping with both `full_attention` and `sliding_attention` masks, then selects by `layer_types[i]`.
- Sliding window: 128 for sliding layers.
- RoPE: Q/K receive RoPE before cache update, so cached keys are already position-encoded.
- Sink logits: `sinks` shape `[num_attention_heads]`, expanded to `[B, heads, q_len, 1]`, concatenated to attention logits, included in max-subtract and softmax, then removed before value matmul.
- SDPA: source marks unsupported.
- Optimized backends: `_supports_flash_attn=True`, `_supports_flex_attn=True`, compatible flash implementations include `kernels-community/vllm-flash-attn3` and `flash_attention_4`.

Cache ABI for OpenAI configs:

- Full layer K/V cache logical shape before repeat: `[B, 8, T, 64]` each.
- Sliding layer K/V cache logical shape before repeat: bounded by the sliding-window cache layer; source cache docs describe `[B, heads, min(seq_len, sliding_window), head_dim]`.
- Attention compute after repeat uses K/V as `[B, 64, T_or_window, 64]`.
- Cache update happens after RoPE and before K/V repetition.
- `DynamicCache(config)` uses `config.layer_types` to build full or sliding cache layers.

Fused attention parity risks:

- A normal FlashAttention call without sink support is wrong.
- Sliding layers need local left-window semantics matching `create_sliding_window_causal_mask`.
- If a fused backend returns probabilities/attentions, sink-normalized probabilities exclude the sink column after softmax.
- Position ids use `past_key_values.get_seq_length()` from the cache object; mixed sliding/full cache implementations must preserve the reported global sequence length, not just resident sliding length.

## 7. Position encoding and custom math

RoPE computation:

```python
def gpt_oss_rope(inv_freq, position_ids, attention_scaling, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    cos = freqs.cos() * attention_scaling
    sin = freqs.sin() * attention_scaling
    return cos.to(dtype), sin.to(dtype)
```

Rotation:

```python
def gpt_oss_apply_rope(x, cos, sin):
    first, second = torch.chunk(x, 2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
```

This is half-split RoPE. DinoML should not substitute an even/odd pair rotation unless weights and reference prove equivalence.

Expert activation:

```python
def gpt_oss_expert_gate(gate_up):
    gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
    gate = gate.clamp(max=7.0)
    up = up.clamp(min=-7.0, max=7.0)
    return (up + 1) * gate * torch.sigmoid(1.702 * gate)
```

Attention sink eager math:

```python
scores = (q @ repeat_kv(k).transpose(-2, -1)) * scale
scores = scores + mask
combined = torch.cat([scores, sinks[..., None]], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
probs = softmax(combined, dim=-1)
scores_without_sink = probs[..., :-1]
out = scores_without_sink @ repeat_kv(v)
```

Precomputable:

- RoPE inverse frequencies and attention scaling for a config.
- Dense causal/sliding mask templates for static buckets, though dynamic attention masks and cache length still affect runtime.
- MXFP4 swizzled expert weights after load, if DinoML adopts the same packed format.

Dynamic:

- `position_ids` if caller provides them or cache length changes.
- routing top-k indices and per-token expert grouping.
- sliding/full mask materialization with batch padding masks.

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids: [B, S]` or `inputs_embeds: [B, S, H]`, exactly one required.
- `attention_mask` optional. If not already a dict, source constructs both full and sliding causal masks from it.
- `position_ids` optional. If missing, source creates `[1, S]` as `arange(S) + past_seen_tokens`.
- `past_key_values` optional. If `use_cache=True` and missing, source creates `DynamicCache(config)`.

CPU/data-pipeline:

- Tokenization and chat template are outside the model graph.
- Official generation config has BOS 199998, PAD 199999, and multiple EOS ids.
- HF API metadata for the official repos includes an OpenAI harmony chat template; that is tokenizer/controller behavior, not a GPU graph operator.

GPU/runtime:

- token embedding lookup
- optional externally supplied embeddings path
- dynamic mask generation or pre-packed mask mapping
- logits slicing with `logits_to_keep`

No vision/audio/discrete-code preprocessing applies.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V Linear canonicalization

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x) -> view/transpose -> RoPE(q,k)
```

Replacement:

```text
Grouped or fused QKV GEMM -> split [Q, K, V] -> reshape heads -> RoPE
```

Preconditions:

- Same input tensor, same dtype, same batch/sequence axes.
- Bias enabled for all Q/K/V or handled per projection.
- Output split sizes are `[num_attention_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim]`.
- Do not assume Q/K/V are equal widths.

Weight transform:

```python
w = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
b = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
```

Failure cases:

- checkpoint stored as packed original `qkv.weight` must be converted with exact Q/K/V split order before this rewrite.
- tensor-parallel sharding can change local split boundaries.

Parity test:

- Compare q/k/v tensors before RoPE on random `[B,S,2880]` for 20B and tiny configs.

### Rewrite: dense fallback MoE -> grouped expert GEMM

Source pattern:

```text
topk router -> per-expert loop -> expert matmul -> index_add
```

Replacement:

```text
topk router -> stable token grouping -> grouped GEMM gate_up -> fused gate -> grouped GEMM down -> scatter-add
```

Preconditions:

- `num_experts_per_tok` fixed for compiled graph or admitted as runtime metadata.
- Expert weights are either dense BF16 or a supported packed MXFP4 provider.
- Routing order and duplicate expert behavior match PyTorch `topk`.
- Accumulation order tolerance is defined; scatter-add can be nondeterministic if parallelized.

Shape equations:

- router logits `[B*S, E]`
- selected indices/scores `[B*S, K]`
- gate_up per selected token `[tokens_for_expert, 2I]`
- down output `[tokens_for_expert, H]`

Failure cases:

- distributed expert-parallel route where only a subset of experts is local.
- unsupported MXFP4 packed layout.

Parity test:

- Force router logits to select known experts, compare dense fallback and grouped implementation after one layer.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
Slice hidden states first, then GEMM only selected positions.
```

Preconditions:

- `logits_to_keep` is int > 0 or a known tensor index set.
- No loss computation requiring all shifted logits.

Failure cases:

- labels provided for training/loss.
- caller requests full logits with `logits_to_keep=0`.

Parity test:

- Compare logits for `logits_to_keep=1`, `logits_to_keep=N`, and tensor indices.

### Rewrite: RoPE precompute and fused apply

Source pattern:

```text
rotary_emb(hidden_states, position_ids) -> apply_rotary_pos_emb(q,k)
```

Replacement:

```text
cached cos/sin gather for positions -> fused half-split rotate Q/K
```

Preconditions:

- Same YaRN/default RoPE parameters.
- Position ids monotonic or gatherable.
- Half-split rotation semantics preserved.

Failure cases:

- dynamic RoPE update changes inverse frequencies.
- custom position ids exceed precomputed table.

Parity test:

- Compare Q/K after RoPE for prefill and decode position ids around 4096 and long-context ranges.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: every block has two plus final norm, fp32 accumulation is required.
- GQA attention with RoPE, sink logits, KV cache, and sliding/full variants: this is the central decode/prefill bottleneck and has nonstandard sink math.
- MoE router plus grouped expert GEMM: expert dispatch dominates model compute; dense per-expert Python loop is not viable.
- MXFP4 expert GEMM/dequant path: official OpenAI checkpoints are MXFP4 for experts, with `_blocks`/`_scales` storage.
- last-token-only logits: avoids full-vocab GEMM for all prefill positions during generation.

Medium priority:

- Fused QKV projection with unequal Q/K/V widths.
- RoPE apply fused with Q/K layout transform.
- fused expert activation `(up + 1) * gate * sigmoid(alpha * gate)` with clamps.
- mask/cache specialization for alternating sliding/full layers.

Lower priority:

- router auxiliary loss and router logits output capture; not needed for inference-first.
- sequence/token classification heads.
- distributed expert-parallel routing; useful later for large model serving, but not needed for single-device parity.

## 11. Runtime staging plan

Stage 1: config and dense one-block parity.

- Parse config including `layer_types`, GQA dimensions, RoPE, sink weights, and MoE dimensions.
- Load a tiny/random dense checkpoint or synthetic weights.
- Validate embedding -> one decoder block -> final norm without MXFP4.

Stage 2: prefill dense parity.

- Implement full/sliding causal masks and DynamicCache-compatible no-cache prefill.
- Implement dense fallback MoE grouped enough to avoid Python loops.
- Compare full logits for short prompts.

Stage 3: decode cache parity.

- Add mixed full/sliding KV cache ABI.
- Verify cached keys are post-RoPE.
- Compare step-by-step decode logits against Transformers.

Stage 4: official MXFP4 load path.

- Admit `_blocks`/`_scales` expert tensors and BF16 non-expert tensors.
- First fallback can dequantize MXFP4 to BF16 at load time if memory permits.
- Optimized path should keep packed weights and lower expert GEMMs through an explicit provider.

Stage 5: optimized attention.

- Add fused attention backend with sink logits and sliding windows.
- Validate prefill and decode separately for full and sliding layers.

Stage 6: production routing and batching.

- Token grouping, grouped GEMM, scatter-add, cache memory planning.
- Add continuous batching only after cache and routing metadata are artifact-visible.

Initially safe to stub:

- training loss and router auxiliary loss
- sequence/token classification heads
- distributed expert parallelism
- beam search cache reorder beyond basic cache API tests
- native MXFP4 kernel if dense dequant fallback is explicitly marked as a memory-heavy first path

## 12. Parity and validation plan

Custom op tests:

- RMSNorm fp32 accumulation vs PyTorch for fp32/fp16/bf16 inputs.
- half-split RoPE against source helper for random q/k and long position ids.
- attention sink softmax against eager source, including masks with `-inf`.
- expert gate activation with clamp limits and interleaved split.
- MXFP4 unpack/dequant on small hand-built `_blocks`/`_scales`.

Layer tests:

- One attention layer, no cache, full attention.
- One attention layer, no cache, sliding attention.
- One attention layer, decode with cache update.
- One MoE block with deterministic router logits selecting fixed experts.
- Full decoder layer parity with random weights.

Model tests:

- tiny-random/gpt-oss prefill logits.
- tiny-random/gpt-oss decode token-by-token vs full prefill where applicable.
- OpenAI config shape/load dry run without full execution.
- Official checkpoint partial-load metadata test: verify MXFP4 names and excluded dense modules.

Suggested tolerances:

- fp32 dense unit ops: `rtol=1e-5`, `atol=1e-5`.
- bf16/fp16 end-to-end: start with `rtol=5e-2`, `atol=5e-2` for full blocks, tighten per op.
- MXFP4 dequant/native: compare against Transformers dequant path with dtype-specific tolerances; do not compare directly to dense source weights unless using the same dequant routine.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 4096, 8192, 32768.
- Decode tokens/sec with mixed full/sliding cache at batch sizes 1, 4, 16, 64.
- KV cache memory split by full layers vs sliding layers.
- Attention backend comparison: eager, DinoML fused, FlashAttention-like sink-capable implementation.
- MoE routing overhead: top-k + grouping + scatter separate from expert GEMMs.
- Expert GEMM throughput: dense BF16, load-time dequant BF16, packed MXFP4 native.
- Router distribution stress: uniform random routes vs all tokens to same experts.
- Last-token-only logits vs full-sequence logits.
- Weight load latency and peak memory for MXFP4 dequant fallback vs packed provider.

## 14. Skip/defer list

- Training and gradient checkpointing.
- Router auxiliary loss in the runtime fast path.
- Sequence classification and token classification.
- Distributed expert parallel routing and tensor parallel plans.
- Beam search beyond basic `reorder_cache` compatibility.
- CPU native MXFP4 inference path.
- Chat template/controller parity except token ids and EOS/PAD handling.
- Speculative decoding or assistant models.
- Saving/re-serializing MXFP4 weights.

## 15. Final implementation checklist

- [ ] Parse `GptOssConfig` including `layer_types`, GQA dimensions, RoPE/YaRN, sliding window, sinks, and MoE fields.
- [ ] Reject unsupported config combinations explicitly: missing `layer_types` with sliding cache ambiguity, unsupported `rope_type`, unsupported MXFP4 native path, or unsupported distributed expert mode.
- [ ] Load dense weights and preserve official non-tied embed/lm_head behavior.
- [ ] Load MXFP4 `_blocks`/`_scales` expert metadata or dequantize with a documented fallback.
- [ ] Implement RMSNorm with fp32 accumulation.
- [ ] Implement half-split RoPE with YaRN parameter support.
- [ ] Implement GQA attention with 64 Q heads / 8 KV heads shape support.
- [ ] Implement attention sink logits in eager and fused attention.
- [ ] Implement full causal and sliding-window causal masks.
- [ ] Implement mixed full/sliding KV cache layout and global position accounting.
- [ ] Implement router Linear + top-k + selected softmax.
- [ ] Implement grouped dense MoE expert GEMMs and scatter-add.
- [ ] Implement custom expert gate activation with interleaved gate/up split.
- [ ] Add last-token-only logits lowering.
- [ ] Add tiny-random prefill parity test.
- [ ] Add one-layer attention sink parity test.
- [ ] Add sliding-cache decode parity test.
- [ ] Add MXFP4 metadata/load/dequant parity test.
- [ ] Benchmark prefill, decode, MoE routing, expert GEMM, logits, and cache memory separately.
