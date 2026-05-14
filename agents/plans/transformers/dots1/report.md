# dots1 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: rednote-hilab/dots.llm1.base, rednote-hilab/dots.llm1.inst, rednote-hilab/dots.llm1.inst-FP8-dynamic
Config source: Hugging Face raw config.json snapshots saved beside this report
Source files inspected:
  X:/H/transformers/src/transformers/models/dots1/modular_dots1.py
  X:/H/transformers/src/transformers/models/dots1/modeling_dots1.py
  X:/H/transformers/src/transformers/models/dots1/configuration_dots1.py
  X:/H/transformers/src/transformers/masking_utils.py
  X:/H/transformers/src/transformers/cache_utils.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py
  X:/H/transformers/src/transformers/modeling_flash_attention_utils.py
  X:/H/transformers/src/transformers/integrations/moe.py
Any missing files or assumptions:
  modeling_dots1.py and configuration_dots1.py are generated from modular_dots1.py. Future Transformers source edits should target modular_dots1.py.
  No processor/image/audio files exist for dots1. Runtime target is text causal LM.
  Official base/instruct/FP8 repos inspected were public and not gated as of this audit. No small/debug checkpoint was found; the sweep uses base, instruct, and official FP8-dynamic.
```

Local snapshots:

- `config.dots.llm1.base.json`
- `config.dots.llm1.inst.json`
- `config.dots.llm1.inst-FP8-dynamic.json`
- `generation_config.dots.llm1.inst.json`

Primary external links:

- [rednote-hilab/dots.llm1.base](https://huggingface.co/rednote-hilab/dots.llm1.base)
- [rednote-hilab/dots.llm1.inst](https://huggingface.co/rednote-hilab/dots.llm1.inst)
- [rednote-hilab/dots.llm1.inst-FP8-dynamic](https://huggingface.co/rednote-hilab/dots.llm1.inst-FP8-dynamic)

## 2. High-level architecture

dots1 is a text-only decoder-only MoE causal language model. The generated implementation combines Qwen3-style decoder scaffolding/attention with DeepSeek-V3-style dense/MoE blocks, but the generated `modeling_dots1.py` is the runtime source basis for DinoML.

```text
tokenizer/chat template -> input_ids/attention_mask
  -> token embedding
  -> 62 decoder blocks:
       RMSNorm -> causal self-attention with q/k head RMSNorm + RoPE -> residual
       RMSNorm -> dense SwiGLU for early layers or MoE(shared + routed experts) -> residual
  -> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, padding/attention mask, generation-controller EOS handling.
- GPU prefill: embeddings, full causal self-attention, RoPE, dense first layer, routed MoE layers, final logits.
- GPU decode: same blocks with per-layer autoregressive KV cache; official configs use full attention only.
- Independently optimizable: RMSNorm, RoPE generation/application, attention backend, MoE router/top-k/expert GEMMs, last-token-only logits.

Primary DinoML target: `Dots1ForCausalLM` prefill and decode for text generation.

Other implemented heads: `Dots1Model` bare hidden-state model is optional; no sequence classification, multimodal, encoder, or task-specific heads are implemented in this family.

## 3. Important config dimensions

Official checkpoint dimensions from `config.json`:

| Field | base | instruct | FP8-dynamic |
|---|---:|---:|---:|
| `architectures` | `Dots1ForCausalLM` | `Dots1ForCausalLM` | `Dots1ForCausalLM` |
| `hidden_size` | 4096 | 4096 | 4096 |
| `num_hidden_layers` | 62 | 62 | 62 |
| `num_attention_heads` | 32 | 32 | 32 |
| `num_key_value_heads` | 32 | 32 | 32 |
| effective `head_dim` | 128 | 128 | 128 |
| `intermediate_size` | 10944 | 10944 | 10944 |
| `moe_intermediate_size` | 1408 | 1408 | 1408 |
| `n_routed_experts` | 128 | 128 | 128 |
| `n_shared_experts` | 2 | 2 | 2 |
| `num_experts_per_tok` | 6 | 6 | 6 |
| `first_k_dense_replace` | 1 | 1 | 1 |
| `norm_topk_prob` | true | true | true |
| `routed_scaling_factor` | 2.5 | 2.5 | 2.5 |
| `n_group` | omitted, effective 1 | omitted, effective 1 | 1 |
| `topk_group` | omitted, effective 1 | omitted, effective 1 | 1 |
| `vocab_size` | 152064 | 152064 | 152064 |
| `max_position_embeddings` | 32768 | 32768 | 32768 |
| RoPE | `rope_theta=10000000`, `rope_scaling=null` | same | same |
| `rms_norm_eps` | 1e-5 | 1e-5 | 1e-5 |
| `hidden_act` | `silu` | `silu` | `silu` |
| `attention_bias` | false | false | false |
| `attention_dropout` | 0.0 | 0.0 | 0.0 |
| `sliding_window` | null | null | null |
| `use_cache` | true | true | true |
| dtype / weight format | BF16 safetensors | BF16 + small F32 metadata | compressed-tensors FP8 for Linear, BF16/F32 exceptions |

Source-default dimensions from `Dots1Config` differ from the released checkpoints: `hidden_size=4608`, `max_position_embeddings=2048`, `rms_norm_eps=1e-6`, `sliding_window=4096`, `n_routed_experts=None`, `n_shared_experts=None`, and `num_experts_per_tok=None`. DinoML should not instantiate runnable MoE from bare source defaults without filling required expert fields.

Representative checkpoint sweep:

| Checkpoint | Source | Operator-significant notes |
|---|---|---|
| `rednote-hilab/dots.llm1.base` | official config + Hub metadata | 142,774,381,696 BF16 parameters from Hub safetensors metadata; EOS token 151643; neural graph is 1 dense layer + 61 MoE layers. |
| `rednote-hilab/dots.llm1.inst` | official config + generation config | Same neural dimensions; EOS token 151649 in config, generation config uses EOS list `[151643, 151649]`; chat template changes ABI only. |
| `rednote-hilab/dots.llm1.inst-FP8-dynamic` | official config | Same neural dimensions; compressed-tensors FP8 dynamic activations/channel-wise FP8 weights for `Linear`, `lm_head` ignored; quantized loading/provider contract differs. |

## 3a. Family variation traps

- Official configs use old-style `rope_theta`/`rope_scaling`; current source standardizes these into `config.rope_parameters`. DinoML config parsing should support both encodings.
- `head_dim` is not an explicit official config key. Source computes `hidden_size // num_attention_heads` unless an extra `head_dim` attribute exists. Do not infer projection width from `hidden_size` alone when user configs include `head_dim`.
- Current official checkpoints are MHA (`num_key_value_heads == num_attention_heads`), but source supports GQA/MQA through `num_key_value_heads` and `repeat_kv`.
- Attention and MLP projection biases are config-dependent. Official checkpoints set `attention_bias=false`; MLP and expert projections are bias-free in source.
- `first_k_dense_replace` changes block structure. Official checkpoints have layer 0 dense SwiGLU, layers 1-61 MoE.
- `n_group`, `topk_group`, `norm_topk_prob`, `routed_scaling_factor`, and `num_experts_per_tok` change MoE routing. Official configs are top-6 from 128 experts with one group and normalized probabilities.
- `config.num_local_experts` is read by source expert storage, while `Dots1Config.attribute_map` aliases it to `n_routed_experts`. Loaders must preserve this alias.
- `sliding_window` and `layer_types` can introduce sliding attention, but official configs set `sliding_window=null`; all official layers are full causal attention.
- Historical config keys `moe_layer_freq`, `pretraining_tp`, `scoring_func`, `topk_method`, and `use_sliding_window` appear in released configs but are not read by `modeling_dots1.py`. Treat them as ignored for the current in-library source basis.
- `tie_word_embeddings=false` in official configs, despite `_tied_weights_keys` declaring a possible alias name. Do not tie LM head and embeddings unless config asks for it.
- FP8-dynamic changes storage/dequant/provider requirements but not graph topology.
- No NCHW/NHWC or image/video layout concerns exist for dots1.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab_size=152064, hidden=4096)`.
- Reshape/view `[B,S,4096] -> [B,S,heads,128]`, transpose to `[B,heads,S,128]`, transpose back, contiguous, flatten.
- Mask preparation from 2D attention masks to causal attention backend masks.
- Optional top-k index/gather/scatter/index-add paths in MoE.

Neural network primitives:

- RMSNorm over hidden: `RMSNorm(4096, eps=1e-5)` for block norms and final norm.
- Per-head RMSNorm: `RMSNorm(128, eps=1e-5)` on q and k after projection and before transpose/RoPE.
- Dense attention projections, official shapes:
  - `q_proj`: Linear `4096 -> 4096`, bias false.
  - `k_proj`: Linear `4096 -> 4096`, bias false.
  - `v_proj`: Linear `4096 -> 4096`, bias false.
  - `o_proj`: Linear `4096 -> 4096`, bias false.
- Dense layer-0 MLP:
  - `gate_proj`: Linear `4096 -> 10944`, bias false.
  - `up_proj`: Linear `4096 -> 10944`, bias false.
  - `silu(gate) * up`.
  - `down_proj`: Linear `10944 -> 4096`, bias false.
- MoE layers 1-61:
  - Router `F.linear(float32 hidden, weight[128,4096])`.
  - Expert `gate_up_proj`: `[128, 2816, 4096]`, chunk into gate/up of `1408` each.
  - Expert `down_proj`: `[128, 4096, 1408]`.
  - Shared experts as dense SwiGLU with `intermediate = 1408 * 2 = 2816`.
- `lm_head`: Linear `4096 -> 152064`, bias false.

Attention primitives:

- Causal self-attention.
- MHA for official checkpoints; source also supports GQA/MQA.
- Eager fallback uses `matmul(q, k^T) * head_dim^-0.5`, add mask, fp32 softmax, dropout, matmul with v.
- Backend dispatch supports eager/SDPA/Flash/Flex/Paged via `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:

- Default RoPE over full head dim 128.
- `inv_freq = 1 / rope_theta ** (arange(0, dim, 2) / dim)`, with official `rope_theta=10000000`.
- cos/sin computed in float32 and cast back to hidden dtype.
- `rotate_half` concatenates `[-second_half, first_half]`.

Generation/cache ops:

- `DynamicCache(config)` by default when `use_cache` and no cache is supplied.
- Per-layer key/value cache stores post-RoPE keys and raw values with shape `[B, num_key_value_heads, T, head_dim]`; official `[B,32,T,128]`.
- Decode position IDs are `arange(new_tokens) + past_seen_tokens`.
- `logits_to_keep` may restrict LM head to last tokens or an index tensor.

Preprocessing-coupled ops:

- Tokenization and chat templates only. No image/audio processor, placeholder scatter, packed multimodal metadata, or layout translation.
- Instruct generation config sets sampling defaults (`temperature=0.7`, `top_p=0.8`) and EOS list `[151643,151649]`; this is controller ABI, not graph math.

Quantized/packed weight metadata ops:

- FP8-dynamic config uses `quant_method=compressed-tensors`, `format=float-quantized`, dynamic token FP8 input activations, channel-wise FP8 Linear weights, and ignores `lm_head`. DinoML should route this through a quantized provider contract or dequantize to dense fallback before using the standard graph.

Distributed/tensor-parallel ops:

- Source config declares TP plans: attention q/k/v colwise, o rowwise, expert `gate_up_proj` packed colwise, expert down rowwise, shared experts col/rowwise, lm_head colwise gather. This is not required for single-GPU parity but matters for production sharding.

## 5. Layer/block breakdown

Decoder block, official layer 0 dense:

```text
x: [B,S,4096]
r = x
x = RMSNorm_4096(x)
q = Linear(4096 -> 4096, no bias)(x).view(B,S,32,128)
k = Linear(4096 -> 4096, no bias)(x).view(B,S,32,128)
v = Linear(4096 -> 4096, no bias)(x).view(B,S,32,128)
q = RMSNorm_128(q).transpose(1,2)
k = RMSNorm_128(k).transpose(1,2)
v = v.transpose(1,2)
q,k = RoPE(q,k, cos,sin)
k,v = cache.update(k,v, layer_idx) when cache is present
a = causal_attention(q,k,v, scale=1/sqrt(128), mask)
x = r + Linear(4096 -> 4096, no bias)(a)
r = x
x = RMSNorm_4096(x)
x = Linear(10944 -> 4096)(silu(Linear(4096 -> 10944)(x)) * Linear(4096 -> 10944)(x))
x = r + x
```

Decoder block, official layers 1-61 MoE:

```text
attention path is identical
r = x
x = RMSNorm_4096(x)
router_logits = Linear(4096 -> 128, fp32, no bias)(x.reshape(B*S,4096))
router_scores = sigmoid(router_logits)
choice_scores = router_scores + e_score_correction_bias[128]
group_scores = top2(choice_scores.view(tokens,n_group,experts_per_group)).sum(-1)
active_groups = topk(group_scores, topk_group)
topk_indices = topk(masked choice_scores, k=6, sorted=False)
topk_weights = gather(router_scores, topk_indices)
topk_weights = topk_weights / (sum(topk_weights) + 1e-20)
topk_weights = topk_weights * 2.5
routed = sum_over_selected_experts(Linear_1408_to_4096(silu(gate) * up) * topk_weight)
shared = DenseSwiGLU(4096 -> 2816 -> 4096)
x = r + routed + shared
```

Model tail:

```text
x = final RMSNorm_4096(x)
logits = Linear(4096 -> 152064, no bias)(x[:, slice_indices, :])
```

## 6. Attention requirements

Required for official checkpoints:

- Causal self-attention only.
- MHA: 32 query heads, 32 key/value heads, 128 dim per head.
- Query/key/value widths are all 4096; attention output width is 4096.
- Masking: lower-triangular causal mask plus optional padding mask. 4D masks may be passed through if already prepared by generation helpers.
- RoPE is applied to q and k before cache update; cached keys are stored after RoPE.
- Softmax is fp32 in eager attention and cast back to query dtype.
- Dropout is 0.0 in inference.
- Official configs do not require sliding-window attention.
- KV cache per layer before any repeat expansion: `[B,32,T,128]` key and `[B,32,T,128]` value. With GQA configs, cache remains `[B,num_key_value_heads,T,head_dim]` and only attention computation repeats K/V to query heads.

Source-supported variants DinoML may admit later:

- GQA/MQA when `num_key_value_heads < num_attention_heads`.
- Sliding causal attention when `config.layer_types[i] == "sliding_attention"` and `config.sliding_window` is not null.
- FlashAttention/SDPA/Flex/Paged dispatch via `config._attn_implementation`; source passes `sliding_window` through backend kwargs.

Packed/varlen support is backend-specific in Transformers attention integrations, not dots1-specific source logic. First integration can require dense padded batches and full attention.

## 7. Position encoding and custom math

RoPE basis:

```python
def dots1_inv_freq(head_dim=128, theta=10_000_000):
    return 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
```

RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat((-x2, x1), dim=-1)

def apply_dots1_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precomputable:

- `inv_freq` for fixed `head_dim` and `rope_theta`.
- cos/sin tables up to the maximum admitted sequence length when using default RoPE.

Dynamic inputs:

- `position_ids` depend on cache length during decode.
- Source decorates RoPE with `dynamic_rope_update` for advanced RoPE types; official configs use default RoPE only.

Custom MoE routing math:

```python
scores = sigmoid(router_logits)
choice = scores + e_score_correction_bias
group_scores = topk(choice.view(tokens, n_group, experts_per_group), k=2, dim=-1).values.sum(-1)
group_idx = topk(group_scores, k=topk_group, sorted=False).indices
choice = choice.masked_fill(groups_not_selected, -inf)
expert_idx = topk(choice, k=num_experts_per_tok, sorted=False).indices
weights = gather(scores, expert_idx)
if norm_topk_prob:
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
weights = weights * routed_scaling_factor
```

Top-k tie ordering is not guaranteed because source uses `sorted=False`; parity tests should avoid exact-tie router logits or compare against captured PyTorch outputs.

## 8. Preprocessing and input packing

Neural graph inputs:

- `input_ids`: `[B,S]` long, or mutually exclusive `inputs_embeds`: `[B,S,4096]`.
- `attention_mask`: optional 2D `[B, past+S]` padding mask or already prepared mask mapping.
- `position_ids`: optional `[B,S]`; if omitted, source uses `arange(S) + past_seen_tokens` and unsqueezes to `[1,S]`.
- `past_key_values`: optional Transformers `Cache`.
- `use_cache`: controls whether cache is returned.

Tokenizer/controller ABI:

- Base repo tokenizer metadata uses `<|endoftext|>` as EOS/pad in Hub metadata and config EOS id `151643`.
- Instruct config EOS id is `151649`; generation config uses `[151643,151649]`.
- Chat templates build system/user/assistant delimiters. This affects input tokens and stopping behavior, not GPU operator coverage.

No model-coupled image/audio/video preprocessing, placeholder embeddings, `masked_scatter`, grid metadata, `cu_seqlens`, token-type IDs, or structural side inputs are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate q/k/v projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(x, q_w)
k = Linear(x, k_w)
v = Linear(x, v_w)
```

Replacement:

```text
qkv = MatMul(x, concat_rows(q_w, k_w, v_w).T)
split qkv as [Q rows, K rows, V rows]
```

Preconditions:

- Same input tensor and dtype.
- Same bias policy; official checkpoints have no attention bias.
- Split order must be Q, K, V with widths `[num_attention_heads*head_dim, num_key_value_heads*head_dim, num_key_value_heads*head_dim]`.
- Preserve per-head q/k RMSNorm after split and before RoPE.

Failure cases:

- Bias mismatch, quantized storage that cannot be concatenated losslessly, provider requiring separate sharded weights, or custom `head_dim` not reflected in split widths.

Parity sketch:

- Compare q/k/v tensors after q/k RMSNorm and reshape for random `[B,S,4096]`.

### Rewrite: dense SwiGLU fusion

Source pattern:

```text
down(silu(gate(x)) * up(x))
```

Replacement:

```text
packed_gate_up_gemm -> split -> silu_mul -> down_gemm
```

Preconditions:

- `hidden_act == "silu"`.
- `gate_proj` and `up_proj` are bias-free or have compatible packed bias support.
- Official dense layer 0 uses intermediate 10944; shared experts use 2816.

Weight transform:

```python
packed = concat([gate_proj.weight, up_proj.weight], dim=0)
```

### Rewrite: MoE expert packed GEMM

Source pattern:

```text
for selected expert:
  gate, up = linear(tokens_for_expert, gate_up_proj[expert]).chunk(2)
  y = linear(silu(gate) * up, down_proj[expert])
  index_add(output, token_idx, y * topk_weight)
```

Replacement:

```text
router_topk -> token/expert bucketing -> grouped GEMM gate_up -> silu_mul -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- `gate_up_proj` layout is `[expert, 2*moe_intermediate, hidden]` with `[gate; up]` concatenation.
- `down_proj` layout is `[expert, hidden, moe_intermediate]`, used by `F.linear(input, down_proj[expert])`.
- Token order and accumulation must match source within floating tolerance.
- Router top-k and normalization must be source-equivalent.

Failure cases:

- `n_group > 1` and `topk_group` not implemented, tie-sensitive top-k tests, quantized expert storage without provider support, or non-silu activation.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather needed positions -> lm_head
```

Preconditions:

- `logits_to_keep` is known as int `1` or a valid index tensor.
- Loss computation is not requested.

Failure cases:

- Full logits requested, labels/loss requested, or generation controller needs prompt logits.

### Rewrite: RoPE table precompute

Source pattern:

```text
freqs = inv_freq @ position_ids
cos/sin per forward
```

Replacement:

```text
lookup precomputed cos/sin[position_ids]
```

Preconditions:

- Default RoPE, fixed theta/head dim, admitted max position bounded.
- Position IDs are monotonic or arbitrary lookup supported.

Failure cases:

- Dynamic/linear/yarn/longrope configs, sequence length beyond table, or per-layer RoPE parameters.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm hidden and per-head q/k RMSNorm. It appears twice per block plus q/k norms in every attention layer.
- Packed QKV projection + q/k RMSNorm + reshape scheduling. This is on every token in prefill and decode.
- RoPE + attention prefill/decode. Preserve source order: q/k norm, transpose, RoPE, cache update, attention.
- MoE router + grouped top-k expert GEMMs. 61 of 62 layers are MoE; naive per-expert loops are too slow for production.
- SwiGLU gate/up fusion for dense layer and shared experts.

Medium priority:

- Last-token-only logits for decode.
- FP8 Linear provider path for official FP8-dynamic checkpoint.
- Attention backend comparison: dense prefill Flash/SDPA and paged decode.
- Weighted scatter-add/index-add fusion for MoE accumulation.

Lower priority:

- Sliding-window attention path, because official configs do not use it.
- Tensor-parallel sharding plans.
- Full logits/loss path, training-only loss, and output attentions.

## 11. Runtime staging plan

Stage 1: Config and weights

- Parse old and new RoPE config fields.
- Load BF16 official base/instruct dense weights.
- Reject or explicitly route FP8 compressed-tensors until provider support exists.
- Confirm embedding/LM head are not tied for official configs.

Stage 2: Dense-only skeleton

- Implement embeddings, RMSNorm, RoPE, MHA attention, first dense SwiGLU layer, final norm, LM head.
- Validate layer 0 parity on random tensors and captured checkpoint weights.

Stage 3: MoE eager parity

- Implement router sigmoid, group/top-k selection, probability normalization, routed scaling, shared experts, and source-layout expert weights.
- Start with small token batches and deterministic non-tie router fixtures.

Stage 4: Full prefill parity

- Run full 62-layer prefill for short prompts and compare logits.
- Add `logits_to_keep` support.

Stage 5: Decode with KV cache

- Implement cache shape `[B,32,T,128]` per layer, position offset, and one-token decode.
- Validate multi-step token parity against Transformers.

Stage 6: Optimized kernels

- Packed QKV, fused RMSNorm, Flash/SDPA attention, grouped MoE GEMM, last-token logits.

Stage 7: Production controls

- Continuous batching, paged cache, quantized FP8 path, tensor parallel, long-context memory probes.

Initially stub/defer: training loss, output attentions, gradient checkpointing, sliding-window configs, quantized loading, and distributed sharding.

## 12. Parity and validation plan

- Config parser tests:
  - official old-style `rope_theta`/`rope_scaling` maps to default RoPE parameters.
  - omitted `n_group`/`topk_group` defaults to 1.
  - ignored historical keys do not affect graph admission.
- Random tensor custom-op tests:
  - RMSNorm fp32 accumulate/cast-back parity.
  - RoPE cos/sin and `rotate_half` parity.
  - Router top-k with controlled non-tie logits.
  - Expert weight layout `[expert, 2*I, H]` chunk order parity.
- Single-layer parity:
  - layer 0 dense block.
  - one MoE block with a small synthetic config and then official layer weights.
- After-N-layer parity:
  - first 2 layers, then first 8 layers, then all 62.
- Prefill logits parity:
  - short prompt, padded batch, long-context smoke prompt.
- Decode parity:
  - prefill + 1 token, then 8 iterative decode steps with cache.
- End-to-end text parity:
  - base EOS and instruct EOS-list generation controller cases.
- Suggested tolerances:
  - fp32 custom ops: max abs `1e-5`, relative `1e-5`.
  - bf16 full-model logits: start with max abs `5e-2`/relative `5e-2`, tighten per kernel once attention/MoE order is fixed.
  - Top-k indices must match exactly on non-tie tests.

No DinoML tests were run for this audit; the task scope was source/config inspection only.

## 13. Performance probes

- Tokenizer/chat-template throughput separately from GPU runtime.
- Prefill throughput over sequence lengths 128, 512, 2048, 8192, 32768.
- Decode tokens/sec over batch sizes 1, 4, 16, 64.
- KV cache memory: `62 * 2 * B * T * 32 * 128 * dtype_size` for official full attention.
- Router time vs expert GEMM time vs scatter-add time.
- Expert load-balance histogram for real prompts and synthetic random prompts.
- Dense first layer vs MoE layer latency breakdown.
- Attention backend comparison: eager, SDPA, FlashAttention, paged decode if available.
- Last-token-only logits vs full logits at vocab 152064.
- FP8-dynamic load/dequant/provider comparison against BF16.
- Continuous batching probe with mixed prefill/decode workloads.

## 14. Skip/defer list

- Training, loss parity, gradients, gradient checkpointing.
- Output attentions/hidden-state recording unless needed for debugging.
- Sliding-window attention configs; official released configs are full attention.
- GQA/MQA admission beyond config parser guards; official checkpoints are MHA.
- Quantized FP8 provider path for first BF16 parity.
- Tensor parallel and pipeline parallel.
- Beam search/speculative decoding; first target can use greedy/sampling controller.
- Remote/hub custom kernels; treat them as optional optimized providers after dense parity.

## 15. Final implementation checklist

- [ ] Parse `Dots1Config`, including legacy `rope_theta`/`rope_scaling` and source-default omissions.
- [ ] Admit official BF16 base/instruct configs and reject incomplete bare defaults.
- [ ] Load embeddings, 62 decoder layers, final norm, and untied LM head.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement default RoPE with `theta=10000000` and position offset from cache length.
- [ ] Implement MHA causal attention with cache shape `[B,32,T,128]`.
- [ ] Implement q/k per-head RMSNorm before RoPE.
- [ ] Implement dense SwiGLU for layer 0 and shared experts.
- [ ] Implement MoE router sigmoid, correction bias, group top-k, expert top-k, normalization, and scaling.
- [ ] Implement expert storage layout `[E,2*I,H]` and `[E,H,I]`.
- [ ] Implement source-equivalent scatter/index-add accumulation.
- [ ] Add `logits_to_keep` support.
- [ ] Add config guards for ignored historical fields and unsupported sliding/GQA/quantized variants.
- [ ] Add single-op parity tests for RMSNorm, RoPE, router, and expert GEMM layout.
- [ ] Add layer 0 parity, MoE layer parity, full prefill logits parity, and decode cache parity.
- [ ] Benchmark prefill, decode, MoE routing/expert GEMMs, logits, and KV memory.
