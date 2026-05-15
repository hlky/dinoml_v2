# PhiMoE Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/Phi-3.5-MoE-instruct
Config source: HF config.json plus native PhimoeConfig defaults
Primary runtime target: causal LM prefill/decode for text generation
Source files inspected:
  transformers/src/transformers/models/phimoe/configuration_phimoe.py
  transformers/src/transformers/models/phimoe/modeling_phimoe.py
  transformers/src/transformers/models/phimoe/modular_phimoe.py
  transformers/src/transformers/modeling_rope_utils.py
  transformers/src/transformers/configuration_utils.py
Any missing files or assumptions:
  No DinoML code or tests were run. No commits were made.
```

Source URLs:

- [configuration_phimoe.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/phimoe/configuration_phimoe.py)
- [modeling_phimoe.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/phimoe/modeling_phimoe.py)
- [modular_phimoe.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/phimoe/modular_phimoe.py)
- [microsoft/Phi-3.5-MoE-instruct config](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct/raw/main/config.json)

`modeling_phimoe.py` is generated from `modular_phimoe.py`; future source edits should inspect the modular file first. The official HF repo still carries remote-code `auto_map` entries, but this report is scoped to native in-library `Phimoe*` behavior at the pinned Transformers commit. A small config sweep snapshot is in `config_sweep_snapshot.json`.

## 2. High-level architecture

PhiMoE is a text-only decoder-only MoE causal LM:

```text
tokenizer/chat template -> token ids/attention mask -> embedding
  -> repeated decoder blocks with GQA self-attention + sparse MoE MLP
  -> final LayerNorm -> LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: Llama tokenizer, left padding, chat-template insertion of `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, and generation stop IDs.
- Prefill: full causal or sliding-window causal attention over prompt, LongRoPE position embedding, MoE routing/expert execution per token.
- Decode: one or more new tokens, `DynamicCache` KV update per layer, optional last-token-only logits through `logits_to_keep`.
- Independently optimizable regions: tokenizer outside DinoML runtime, one decoder block parity, attention+KV cache, MoE router+expert dispatch, final logits slice.

Implemented heads:

- `PhimoeModel`: required as the decoder body.
- `PhimoeForCausalLM`: required for the target.
- `PhimoeForSequenceClassification`: optional/deferred.
- Training loss and router auxiliary loss: deferred for inference.

## 3. Important config dimensions

| Field | Native default | `microsoft/Phi-3.5-MoE-instruct` |
|---|---:|---:|
| `vocab_size` | 32064 | 32064 |
| `hidden_size` | 4096 | 4096 |
| `num_hidden_layers` | 32 | 32 |
| `num_attention_heads` | 32 | 32 |
| `num_key_value_heads` | 8 | 8 |
| inferred `head_dim` | 128 | 128 |
| `intermediate_size` | 6400 | 6400 |
| `num_local_experts` | 16 | 16 |
| `num_experts_per_tok` | 2 | 2 |
| `hidden_act` | `silu` | `silu` |
| `max_position_embeddings` | 131072 | 131072 |
| original context | from RoPE params when set | 4096 |
| RoPE theta | default class says 1000000.0 | config has `rope_theta=10000.0` |
| RoPE type | default unless supplied | LongRoPE via historical `rope_scaling` |
| `sliding_window` | `None` | 131072 |
| `attention_bias` | `False` | `True` |
| `lm_head_bias` | `False` | `True` |
| `tie_word_embeddings` | `False` | `False` |
| `torch_dtype` | source default not dtype-specific | `bfloat16` |
| `use_cache` | `True` | `True` |

Representative checkpoint sweep:

| Checkpoint | Kind | Operator-significant notes |
|---|---|---|
| [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) | official production | 32 layers, 4096 hidden, GQA 32/8 heads, 16 experts, top-2 routing, LongRoPE, bfloat16, projection and LM head biases enabled. |
| [yujiepan/phi-3.5-moe-tiny-random](https://huggingface.co/yujiepan/phi-3.5-moe-tiny-random) | open tiny/debug | 2 layers, hidden 16, MHA-style 4 KV heads, still 16 experts/top-2. Good structural smoke target, not a performance proxy. |
| [mlx-community/Phi-3.5-MoE-instruct-4bit](https://huggingface.co/mlx-community/Phi-3.5-MoE-instruct-4bit) | open mirror/quantized | Same graph dimensions in config; MLX quantized weights are a loading/provider issue, not native PyTorch operator surface. |
| [1024m/Phi-3.5-MoE-4bit-nf4](https://huggingface.co/1024m/Phi-3.5-MoE-4bit-nf4) | open bitsandbytes mirror | Same graph dimensions, `quantization_config.quant_method=bitsandbytes`, `torch_dtype=float16`. Route to quantized loading audit or dense fallback. |
| [trl-internal-testing/tiny-random-PhiMoEForCausalLM](https://huggingface.co/trl-internal-testing/tiny-random-PhiMoEForCausalLM) | internal tiny | Raw config returned 401 Unauthorized during this audit. Access would resolve whether it is useful as another tiny fixture. |

Search hits such as `fmshahata/phi-moe-*` were inspected enough to reject for this report: their configs use `model_type=phi` / `PhiForCausalLM`, not native `phimoe`.

## 3a. Family variation traps

- `hidden_size != num_key_value_heads * head_dim`: official K/V width is 1024 while Q/O attention width is 4096.
- GQA is required for the production checkpoint: `num_key_value_heads=8`, `num_attention_heads=32`, repeat factor 4.
- Projection bias is config-dependent. Native defaults are no bias, official Phi-3.5-MoE enables attention and LM-head bias.
- Source uses `nn.LayerNorm`, despite the config field name `rms_norm_eps`. Do not lower these norms as RMSNorm.
- Official configs use historical `rope_scaling`; native config standardization exposes this as `rope_parameters`.
- LongRoPE changes inv-frequencies and applies `short_mscale` or `long_mscale` depending on `seq_len > original_max_position_embeddings`.
- `prepare_inputs_for_generation` may invalidate the cache when crossing `original_max_position_embeddings + 1`.
- `sliding_window` is source-read. Official value equals max context, but smaller-window configs would require local causal attention admission.
- `num_experts_per_tok` should be admitted as 2 first; the source `sparsemixer` path is specialized around selecting first and second experts.
- Expert weights are packed 3D tensors, not separate `nn.Linear` modules per expert.
- Inference routing is deterministic max, threshold-mask, softmax, gather, second max after scatter-masking the first expert. Training-only Gumbel and custom autograd can be ignored for inference.
- Quantized mirrors need a separate source-coupled loading/provider contract. Do not treat bitsandbytes or MLX weight storage as normal dense weights.
- The official repo has remote-code files and `auto_map`, but native Transformers support exists at the pinned commit. DinoML should prefer native source basis unless explicitly auditing remote code parity.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,H]`.
- `view/reshape`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `cat`, `chunk`.
- Slice for `logits_to_keep`, typically last token in decode.
- Causal/sliding mask construction and additive mask application.
- `nonzero`, `where`, `one_hot`, `permute`, `gather`, `scatter` or masked fill for routing.
- `index_add_` equivalent for accumulating expert outputs back to token rows.

Neural primitives:

- `LayerNorm(H=4096, eps=1e-5, affine=True)` for input, post-attention, final norm.
- Dense linear projections:
  - `q_proj: 4096 -> 4096`, bias per config.
  - `k_proj: 4096 -> 1024`, bias per config.
  - `v_proj: 4096 -> 1024`, bias per config.
  - `o_proj: 4096 -> 4096`, bias per config.
  - Router: `4096 -> 16`, bias false.
  - LM head: `4096 -> 32064`, bias per config.
- Expert packed GEMMs:
  - `gate_up_proj[e]: 4096 -> 12800`, split into gate/up `6400 + 6400`.
  - activation multiply: `silu(gate) * up`.
  - `down_proj[e]: 6400 -> 4096`.
  - route weight multiply per selected token/expert.

Attention primitives:

- Causal self-attention with GQA.
- RoPE on Q/K before cache update.
- KV cache update with post-RoPE K and V.
- FlashAttention/SDPA/Flex Attention compatible backend hook, with eager fallback.
- Softmax in fp32 in eager path, cast back to query dtype.

Position/rotary ops:

- LongRoPE inv-frequency selection using `short_factor` or `long_factor`.
- `cos/sin` computed in fp32 from `[B,S]` `position_ids`, then cast to hidden dtype.
- `rotate_half` split-concat convention.

Generation/cache ops:

- `DynamicCache` allocation/update/reorder equivalent.
- Position IDs from `past_seen_tokens + arange(S)`.
- Cache reset guard when crossing original context boundary for LongRoPE.
- Last-token-only logits path.

Preprocessing-coupled ops:

- Tokenizer and chat template are CPU/data pipeline work.
- Left-padding attention mask enters GPU mask construction.
- Stop tokens from generation config are controller behavior, not graph ops.

Quantized/packed weight metadata:

- Packed MoE expert tensors are required.
- bitsandbytes/MLX quantized checkpoints are deferred unless DinoML has an explicit loader/provider path.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x0 = hidden_states                         # [B,S,4096]
x = LayerNorm(x0)
q = Linear(x, Wq) -> view [B,S,32,128] -> transpose [B,32,S,128]
k = Linear(x, Wk) -> view [B,S,8,128] -> transpose [B,8,S,128]
v = Linear(x, Wv) -> view [B,S,8,128] -> transpose [B,8,S,128]
q,k = LongRoPE(q,k, position_embeddings)
k,v = cache.update(k,v, layer_idx)         # if cache supplied
attn = GQA_causal_attention(q,k,v, mask)
x = x0 + Linear(attn.reshape[B,S,4096], Wo)

y0 = x
y = LayerNorm(y0)
flat = reshape(y, [B*S,4096])
router_logits = Linear(flat, Wr)           # [B*S,16]
weights, expert_ids = sparsemixer(router_logits)
expert_out = sparse routed SwiGLU experts
x = y0 + reshape(expert_out, [B,S,4096])
```

Model wrapper:

```text
input_ids -> embed_tokens -> blocks -> final LayerNorm -> LM head slice -> logits
```

Weights are not tied for the official config (`tie_word_embeddings=false`), but `PhimoeForCausalLM` declares tied-weight keys for loader compatibility. DinoML should preserve the actual config and checkpoint aliasing rather than assuming tying.

## 6. Attention requirements

- Type: decoder causal self-attention.
- Head structure: GQA for official checkpoint, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, repeat factor 4.
- Query width: 4096. Key/value projection width: 1024 each. Attention output width before `o_proj`: 4096.
- Masking: additive causal mask from `create_causal_mask` or `create_sliding_window_causal_mask` when `sliding_window` is not `None`.
- Sliding window: source-supported. Official value is 131072, effectively full-window for max context, but admission should reject or route smaller local-window configs until local attention is implemented.
- Cache: per layer K shape `[B,8,T,128]`, V shape `[B,8,T,128]`; K is stored after RoPE.
- Eager math order: repeat K/V to 32 heads, `matmul(q, k^T) * head_dim^-0.5`, add mask, fp32 softmax, cast to query dtype, dropout only in training, `matmul(weights, value)`.
- Backend dispatch: `ALL_ATTENTION_FUNCTIONS` can select eager, SDPA, FlashAttention, or Flex Attention. DinoML first target should own one exact eager-compatible path plus a later optimized GQA flash path.
- Packed/varlen: no model-specific cu-seqlens ABI in this source. Backend-specific packed attention may be an optimization, not source semantics.

## 7. Position encoding and custom math

RoPE is source-specific because PhiMoE wraps LongRoPE scaling and an mscale choice:

```python
def phimoe_rope(position_ids, rope_params, head_dim, dtype):
    seq_len = max(position_ids) + 1
    factor = rope_params["long_factor"] if seq_len > rope_params["original_max_position_embeddings"] else rope_params["short_factor"]
    mscale = rope_params["long_mscale"] if seq_len > rope_params["original_max_position_embeddings"] else rope_params["short_mscale"]
    inv_freq = longrope_inv_freq(theta=rope_params["rope_theta"], dim=head_dim, factor=factor)
    freqs = matmul(inv_freq[None, :, None].float(), position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat(freqs, freqs, dim=-1)
    return (cos(emb) * mscale).to(dtype), (sin(emb) * mscale).to(dtype)
```

Apply convention:

```python
def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunity: for static max context and fixed RoPE params, cosine/sine tables can be precomputed or cached. Dynamic decode still needs correct indexing and the LongRoPE boundary reset behavior.

## 8. Preprocessing and input packing

Tokenizer/runtime ABI:

- Official tokenizer class in `tokenizer_config.json`: `LlamaTokenizer`.
- `model_max_length=131072`, `padding_side="left"`.
- `pad_token="<|endoftext|>"`, `pad_token_id=32000`.
- Generation config uses EOS IDs `[32000, 32001, 32007]` and BOS ID `1`.
- Chat template inserts role tags and `<|end|>` separators; this is controller/data-pipeline work.

GPU graph inputs:

- `input_ids` or `inputs_embeds`, exactly one.
- `attention_mask` `[B,S]` optional but important for padding.
- `position_ids` optional; if absent source creates `[1,S]` from cache length.
- `past_key_values` optional cache object.

No image/audio/video processors, placeholder scatters, packed multimodal descriptors, or channel-layout translation are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections to grouped GEMM

Source pattern:

```text
q = linear(x, Wq, bq)
k = linear(x, Wk, bk)
v = linear(x, Wv, bv)
```

Replacement:

```text
one grouped GEMM or fused projection with split outputs [Q, K, V]
```

Preconditions:

- Same input `x`, same dtype/device, compatible bias presence.
- Preserve split widths exactly: official `[4096, 1024, 1024]`.
- Weight packing order must be `[q, k, v]`, not a uniform 3-way split.
- Output reshapes must preserve `[B,S,heads,head_dim]`.

Failure cases: configs with unusual `head_dim`, missing bias on only some projections, or weight-only quantization that requires separate provider handling.

Parity test: compare q/k/v tensors before RoPE on random `[B,S,H]` for fp32 and bf16 tolerances.

### Rewrite: expert packed GEMM by expert buckets

Source pattern:

```text
top_k_pos, token_idx = where(expert_mask[e])
current = hidden[token_idx]
gate, up = linear(current, gate_up_proj[e]).chunk(2)
out = linear(silu(gate) * up, down_proj[e])
final.index_add_(token_idx, out * routing_weight)
```

Replacement:

```text
route tokens -> per-expert compact batches -> expert GEMMs -> weighted scatter-add
```

Preconditions:

- `top_k=2`.
- No token dropping or capacity cap.
- Stable accumulation semantics for duplicate token rows from the two experts.
- Expert weight layout is `[E, out, in]` for both packed tensors.

Failure cases: training jitter/Gumbel path, unsupported `top_k`, quantized expert storage without provider support.

Parity test: deterministic router logits producing known expert assignments, compare final hidden states and selected expert IDs.

### Rewrite: last-token logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement: for decode, only apply LM head to `[B,1,H]` last token.

Preconditions: generation caller does not request full logits or arbitrary tensor `logits_to_keep`.

Failure cases: training loss, scoring full prompts, or user-requested hidden-logit slices.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm affine for `[B,S,4096]`. PhiMoE uses LayerNorm throughout; this is in every block.
- GQA FlashAttention with post-RoPE K cache. Prefill/decode performance depends on avoiding repeated K/V expansion.
- Sparse MoE routing plus expert bucket GEMMs. This is the family-defining cost and needs a provider-visible plan.
- SwiGLU expert fusion: `linear gate_up -> chunk -> silu(gate) * up -> down`.

Medium priority:

- Q/K/V grouped projection plus RoPE fusion.
- Last-token-only LM head and optional top-k sampling boundary.
- Causal mask generation fused with attention admission, especially for sliding-window variants.
- Expert scatter-add accumulation with deterministic duplicate row behavior.

Lower priority:

- Router auxiliary-loss path, training jitter, and custom autograd.
- Sequence classification head.
- Remote-code parity with older official repo files after native source parity is stable.

## 11. Runtime staging plan

Stage 1: parse native `PhimoeConfig`, normalize historical `rope_scaling` into DinoML RoPE metadata, load dense tiny-random weights.

Stage 2: one-block no-cache fp32/bf16 parity with eager attention and dense fallback MoE loop.

Stage 3: full tiny model prefill parity, including LongRoPE and official bias flags.

Stage 4: decode parity with `DynamicCache` equivalent, post-RoPE K storage, position ID generation, and cache reset at LongRoPE original-context crossing.

Stage 5: production-shape dense checkpoint loading with bf16 and last-token logits.

Stage 6: optimized GQA attention and provider-backed MoE expert batching.

Stage 7: quantized checkpoint admission: either reject, dequantize to dense at load, or add explicit bitsandbytes/GGUF/MLX provider contracts.

Initially stub/defer: router aux loss, training jitter, sequence classification, remote-code-only differences, bitsandbytes native kernels.

## 12. Parity and validation plan

- Config parser tests for native defaults versus official config overrides.
- RoPE tests around positions `4095`, `4096`, `4097` to catch short/long factor and cache reset behavior.
- Attention unit parity for `[B,S,H]` with `B=1/2`, `S=1/7/128`, with and without cache.
- Router tests with fixed logits for threshold masking, first expert selection, scatter-masked second selection, weight gather, and selected expert concat order.
- Expert tests for packed `gate_up_proj` and `down_proj` layouts against PyTorch source.
- One decoder layer parity in fp32, then bf16.
- Tiny-random end-to-end prefill logits parity.
- Decode token parity over several generated steps with cache.
- Production config shape-only compile/admission tests for max context, sliding-window value, biases, and LongRoPE factor lengths.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5` for isolated ops; bf16/fp16 `rtol=2e-2, atol=2e-2` initially for full-block parity, tightened per fused kernel.

## 13. Performance probes

- Prefill throughput sweep over `S={128,512,2048,4096,8192}` and batch sizes.
- Decode tokens/sec with cache for `B={1,4,16}` and long cached contexts.
- Attention backend comparison: eager-compatible, SDPA-like, GQA flash, and sliding-window local if admitted.
- MoE router overhead versus expert GEMM time, separated by token count and expert imbalance.
- Expert bucket occupancy histogram and tail-latency probes.
- Last-token LM head cost versus full-sequence logits.
- KV cache memory footprint for `[layers=32, kv_heads=8, head_dim=128]`.
- Dense load versus quantized mirror dequant/load time once a quant provider exists.

No benchmark observations were taken in this audit.

## 14. Skip/defer list

- Training, gradients, gradient checkpointing, router auxiliary loss.
- Router jitter and Gumbel sampling paths.
- Sequence classification head.
- Multi-GPU tensor parallel and pipeline parallel plans.
- bitsandbytes, MLX, or other quantized mirror loading until provider contracts exist.
- Remote-code parity unless native in-library behavior proves insufficient.
- Beam search and sampling processors beyond EOS/pad/tokenizer ABI.
- Smaller sliding-window local attention configs until a local attention implementation is explicitly admitted.

## 15. Final implementation checklist

- [ ] Parse `PhimoeConfig` and normalize `rope_scaling`/`rope_parameters`.
- [ ] Load tokenizer/generation metadata needed by controller.
- [ ] Load dense weights with packed expert tensor layout.
- [ ] Implement LayerNorm affine parity.
- [ ] Implement LongRoPE cos/sin and `rotate_half` application.
- [ ] Implement GQA causal attention with post-RoPE KV cache.
- [ ] Implement cache reset guard at original context crossing.
- [ ] Implement router sparsemixer inference path.
- [ ] Implement packed expert execution and weighted `index_add` accumulation.
- [ ] Implement LM head with `logits_to_keep`.
- [ ] Add Q/K/V grouped projection rewrite with split-width guards.
- [ ] Add MoE expert bucket GEMM lowering plan.
- [ ] Add tiny-random one-block and full-model parity.
- [ ] Add production config admission tests.
- [ ] Add prefill/decode and MoE occupancy benchmarks.
