# FlexOlmo Transformers Audit

## 1. Source basis

```text
Transformers commit/version:
  transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model id:
  Primary: allenai/FlexOlmo-7x7B-1T
  Additional in-family configs: allenai/FlexOlmo-7x7B-1T-RT,
  allenai/Flex-reddit-2x7B-1T, allenai/Flex-code-2x7B-1T

Config source:
  Local source defaults in configuration_flex_olmo.py.
  Raw Hub config.json files listed in config_sweep.md.

Source files inspected:
  transformers/src/transformers/models/flex_olmo/modular_flex_olmo.py
  transformers/src/transformers/models/flex_olmo/configuration_flex_olmo.py
  transformers/src/transformers/models/flex_olmo/modeling_flex_olmo.py
  transformers/src/transformers/integrations/moe.py
  transformers/src/transformers/conversion_mapping.py
  transformers/src/transformers/cache_utils.py
  transformers/src/transformers/masking_utils.py

Any missing files or assumptions:
  No processor or image/audio preprocessor is involved. Tokenizer was inspected
  only for text ABI coupling. Public representative repos were not gated in the
  Hub API/raw-config checks. modular_flex_olmo.py is the authoritative editable
  source; modeling_flex_olmo.py and configuration_flex_olmo.py are generated.
```

Primary runtime target: `FlexOlmoForCausalLM` inference, including prefill logits and decode with autoregressive self-attention KV cache. Training loss, router aux loss, and diagnostic router logits are optional/deferred.

Primary source links:

- <https://huggingface.co/allenai/FlexOlmo-7x7B-1T>
- <https://huggingface.co/allenai/FlexOlmo-7x7B-1T/raw/main/config.json>
- <https://huggingface.co/docs/transformers/model_doc/flex_olmo>

## 2. High-level architecture

FlexOlmo is a text-only causal decoder with dense self-attention and sparse MoE feed-forward blocks. It composes OLMo2-style attention/RMSNorm/RoPE behavior with OLMoE/Mixtral-style sparse expert routing, but FlexOlmo changes the decoder block ordering: attention output is normalized before residual add, and MoE output is normalized before residual add.

Dataflow:

```text
GPT2-style tokenization/chat template
  -> token ids [B, S]
  -> token embedding [B, S, H]
  -> shared RoPE cos/sin for position_ids [B, S, D]
  -> N decoder blocks
  -> final RMSNorm
  -> LM head over kept token positions
  -> logits [B, K, vocab]
  -> external generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: GPT2 tokenizer, chat template, padding/truncation, `attention_mask`, optional caller-provided `position_ids`.
- GPU prefill: embeddings, full causal attention over prompt, MoE routing/expert execution, final logits.
- GPU decode: one or more new tokens, per-layer KV cache update, causal attention against cached keys/values, usually last-token logits only.
- Independently optimizable units: token embedding/LM head, one decoder layer, attention prefill/decode, MoE router plus expert GEMMs, final logits slicing.

## 3. Important config dimensions

Representative source/default and checkpoint dimensions:

| Field | Source default | Main 7x7B | Domain 2x7B examples | Runtime significance |
| --- | ---: | ---: | ---: | --- |
| `vocab_size` | 100352 | 100352 | 100352 | Embedding and LM head width |
| `hidden_size` | 4096 | 4096 | 4096 | Residual width |
| `num_hidden_layers` | 32 | 32 | 32 | Decoder block count |
| `num_attention_heads` | 32 | 32 | 32 | Query heads |
| `num_key_value_heads` | defaults to heads | 32 | 32 | MHA in observed configs; GQA possible by config |
| `head_dim` | inferred 128 | inferred 128 | inferred 128 | `hidden_size // num_attention_heads`; source also honors explicit `head_dim` if present |
| `intermediate_size` | 11008 | 11008 | 11008 | Expert hidden width |
| `num_experts` | 7 | 7 | 2 | Operator-significant MoE width |
| `num_experts_per_tok` | 5 | 7 | 2 | Top-k routes per token |
| `max_position_embeddings` | 4096 | 4096 | 4096 | RoPE cache/original max |
| `rope_theta` | 500000 | 500000 | 500000 | Default RoPE base |
| `hidden_act` | `silu` | `silu` | `silu` | SwiGLU expert activation |
| `attention_bias` | false | false | false | Observed projections are bias-free |
| `attention_dropout` | 0.0 | 0.0 | 0.0 | Inference dropout off |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | false | false | false | LM head is not tied despite `_tied_weights_keys` metadata |
| `use_cache` | true | true | true | Decode KV cache enabled |

Checkpoint sweep details are in `config_sweep.md`.

## 3a. Family variation traps

- `num_experts` and `num_experts_per_tok` are the main observed family variation: 7x7B checkpoints route to all 7 experts, while domain 2x7B checkpoints route to both 2 experts.
- Source default `num_experts_per_tok=5` does not match the observed main or 2-expert checkpoints; use checkpoint config, not source default.
- `num_key_value_heads` defaults to `num_attention_heads`, but source supports lower KV head counts. DinoML should admit MHA first, and gate GQA/MQA by `num_attention_heads % num_key_value_heads == 0`.
- Source honors explicit `head_dim` via `getattr(config, "head_dim", hidden_size // heads)`. Do not infer projection widths from `hidden_size` alone for future checkpoints.
- No sliding-window attention is used by FlexOlmoModel, even though it inherits Mixtral lineage.
- `rope_scaling` appears in configs as `null`; source uses `rope_parameters`/default RoPE machinery. Treat non-default RoPE types as a separate admission path unless verified.
- Public checkpoints store split per-expert weights (`experts.E.gate_proj`, `up_proj`, `down_proj`), while current source module expects packed expert tensors (`experts.gate_up_proj`, `down_proj`). Transformers conversion mapping merges these at load time. DinoML needs a weight-load conversion contract.
- Historical config field `clip_qkv: null` appears in 2x7B configs, but `modeling_flex_olmo.py` does not read it. Ignore/reject non-null `clip_qkv` for this source basis.
- Tokenizer `model_max_length=8192` exceeds model `max_position_embeddings=4096`; first integration should enforce the model-side position limit unless adding explicit long-context RoPE support.
- `output_router_logits` and aux load-balancing loss are training/diagnostic paths, not required for first inference parity.
- No vision/audio layout translation applies. Tensor axes are text sequence axes; protect `[B, S, H]`, `[B, heads, S, D]`, and `[tokens, experts]` MoE routing axes from layout passes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `Embedding(vocab_size=100352, hidden_size=4096, padding_idx=100277)`.
- Reshape/view: `[B, S, H] -> [B, S, heads, D] -> [B, heads, S, D]`.
- Transpose and contiguous around attention output.
- Slice/gather for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Flatten tokens for routing: `[B, S, H] -> [B*S, H]`.
- Expert selection/indexing: top-k indices, one-hot/where fallback in eager source, packed grouping/gather for optimized path.
- Accumulation over top-k expert outputs into `[B*S, H]`.

Neural primitives:

- Bias-free linear projections:
  - Q: `Linear(4096 -> num_attention_heads * head_dim)`, observed `4096 -> 4096`.
  - K/V: `Linear(4096 -> num_key_value_heads * head_dim)`, observed `4096 -> 4096`.
  - O: `Linear(num_attention_heads * head_dim -> 4096)`, observed `4096 -> 4096`.
  - Router: `Linear(4096 -> num_experts)`, no bias.
  - Expert gate/up packed logical GEMM: `Linear(4096 -> 2 * 11008)` per selected expert, no bias.
  - Expert down: `Linear(11008 -> 4096)` per selected expert, no bias.
  - LM head: `Linear(4096 -> 100352)`, no bias.
- RMSNorm over last dim with fp32 variance accumulation, then cast back to input dtype.
- SiLU and multiply for SwiGLU.
- Residual add after post-attention norm and after post-FFN norm.
- Softmax for router in fp32; `topk`.

Attention primitives:

- Causal dense self-attention for prefill and decode.
- MHA/GQA repeat-kv support: repeat KV heads from `[B, kv_heads, S, D]` to query heads if needed.
- Softmax over key length in fp32, cast to query dtype.
- FlashAttention/SDPA/FlexAttention compatible backend dispatch in Transformers; eager fallback is explicit matmul/softmax/matmul.

Position/rotary ops:

- RoPE cos/sin generation in fp32 from `position_ids`.
- Apply RoPE to Q and K before cache update.

Generation/cache ops:

- `DynamicCache` per layer with K/V tensors shaped `[B, num_key_value_heads, total_seq, head_dim]` before any repeat expansion.
- Cache stores position-encoded keys because RoPE is applied before `past_key_values.update`.
- Position id default: `arange(current_seq) + past_seen_tokens`, shape `[1, S]`.

Quantized/packed weight metadata:

- No quantized checkpoint format is required by source. Weight packing is logical MoE source-key conversion, not quantization.

Preprocessing-coupled ops:

- GPT2Tokenizer, special ids, chat template. No model-internal token-type ids, segment ids, image/audio placeholders, packed sequence descriptors, or `cu_seqlens` ABI in this family.

Optional/deferred:

- Training loss, aux router loss, output router logits capture.
- Tensor parallel plans.
- Alternative experts implementations from `transformers.integrations.moe.py` beyond a first deterministic grouped/packed path.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
Input x: [B, S, H]

Attention branch:
  residual = x
  q = RMSNorm(q_proj(x))            # [B, S, QH * D]
  k = RMSNorm(k_proj(x))            # [B, S, KVH * D]
  v = v_proj(x)                     # [B, S, KVH * D]
  q = view(q, [B, S, QH, D]).transpose(1, 2)
  k = view(k, [B, S, KVH, D]).transpose(1, 2)
  v = view(v, [B, S, KVH, D]).transpose(1, 2)
  q, k = apply_rope(q, k, cos, sin)
  k, v = cache.update(k, v, layer_idx) if cache is present
  attn = causal_attention(q, k, v, mask, scale=D**-0.5)
  attn = attn.transpose(1, 2).reshape([B, S, QH * D])
  attn = o_proj(attn)
  x = residual + RMSNorm(attn)

MoE branch:
  residual = x
  tokens = reshape(x, [B*S, H])
  router_logits = tokens @ router_weight.T     # [B*S, E]
  router_probs = softmax(router_logits, fp32)
  weights, indices = topk(router_probs, K)
  optional normalize top-k weights if norm_topk_prob
  for each selected expert:
      gate, up = linear(tokens_for_expert, gate_up_proj[expert]).chunk(2)
      y = down_proj[expert](silu(gate) * up)
      y *= route_weight
  moe = sum selected expert outputs per token
  x = residual + RMSNorm(reshape(moe, [B, S, H]))
```

Observed main shapes: `H=4096`, `QH=KVH=32`, `D=128`, `E=7`, `K=7`, `intermediate=11008`. Domain 2x7B examples use `E=2`, `K=2`.

Final head:

```text
x = final_rms_norm(x)
logits = lm_head(x[:, slice_indices, :])  # [B, logits_to_keep or S, vocab]
```

## 6. Attention requirements

- Type: causal self-attention only.
- Cross-attention: none.
- Observed head pattern: MHA with `num_key_value_heads == num_attention_heads == 32`.
- Source-supported head pattern: GQA/MQA if `num_key_value_heads < num_attention_heads`; `num_attention_heads // num_key_value_heads` is used for repeat expansion.
- Head dim: inferred 128 for representative configs; explicit `head_dim` must override the inference if present.
- Masking: `create_causal_mask` combines causal masking and optional 2D padding mask. For optimized attention backends, mask may be skipped when causal semantics and backend allow it.
- Packed/varlen: no explicit packed sequence or `cu_seqlens` interface in FlexOlmo source.
- Sliding/local attention: not used by FlexOlmoModel.
- Relative bias/ALiBi: none.
- RoPE: Q/K only, before cache update.
- Cache: per layer K/V are stored after RoPE with shape `[B, KVH, total_seq, D]`; repeat to Q heads is an attention computation detail and should not be stored in cache.
- Attention math order in eager fallback: repeat KV, `query @ key.T`, multiply by `D**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout if training, `weights @ value`.

## 7. Position encoding and custom math

Default RoPE inverse frequency:

```python
def flex_olmo_inv_freq(head_dim, rope_theta=500000.0):
    i = arange(0, head_dim, 2, dtype=float32)
    return 1.0 / (rope_theta ** (i / head_dim))
```

Per-call cos/sin generation:

```python
def flex_olmo_rope_tables(inv_freq, position_ids):
    freqs = matmul(inv_freq[None, :, None], position_ids[:, None, :].float()).transpose(1, 2)
    emb = concat([freqs, freqs], dim=-1)
    return cos(emb), sin(emb)
```

RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat([-x2, x1], dim=-1)

def apply_flex_olmo_rope(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precomputable: default `inv_freq` and bounded cos/sin tables for static maximum positions. Dynamic input: `position_ids`, especially during decode with cache offset. The source computes cos/sin in fp32 and returns them as fp32; Q/K are cast back to original dtype after rotation.

## 8. Preprocessing and input packing

Text ABI:

- Input is either `input_ids [B, S]` or caller-provided `inputs_embeds [B, S, H]`, exactly one required.
- Optional `attention_mask [B, seen_plus_S]` handles padding.
- If `position_ids` is absent, source creates `[1, S]` positions offset by cached sequence length.
- Tokenizer is GPT2Tokenizer with vocab size 100352 and special ids around 100257-100277.
- Chat template is tokenizer-side text construction, not a GPU graph op.

No multimodal packing:

- No `pixel_values`, `input_features`, grid metadata, placeholder scatter, modality token type ids, or sequence-packing metadata.
- No NCHW/NHWC layout issue. Layout passes should not rewrite sequence/head dimensions.

Generation-controller behavior:

- Sampling, beam search, stopping, chat prompt construction, and suppression rules live outside the core model graph.
- First DinoML target can accept already-tokenized input ids and return logits plus updated cache.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Q/K/V projection grouping

Source pattern:

```text
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
q = RMSNorm(q)
k = RMSNorm(k)
```

Replacement pattern:

```text
GroupedLinear(x, [Wq, Wk, Wv]) -> split -> q_norm/k_norm -> v
```

Preconditions:

- All projections are bias-free or all compatible bias handling is implemented.
- `q_norm` and `k_norm` remain separate after split; V is not normalized.
- Weight output widths are known: `QH*D`, `KVH*D`, `KVH*D`.

Failure cases:

- Future checkpoint with attention biases requires bias split.
- Explicit `head_dim` causing `QH*D != hidden_size` must be honored.

Parity sketch: compare q/k/v pre-RoPE tensors against Transformers for random `[B, S, 4096]`.

### Rewrite: RoPE plus attention prefill/decode

Source pattern:

```text
view/transpose q,k,v -> apply_rope(q,k) -> cache update -> attention backend
```

Replacement:

```text
FusedRotaryQK -> FlashAttention/SDPA prefill or decode kernel
```

Preconditions:

- RoPE table dtype/order matches source fp32 cos/sin then cast to Q/K dtype.
- Cache stores post-RoPE K.
- Dense causal full attention, no sliding window.
- Mask either absent or standard padding+causal supported by backend.

Failure cases:

- Non-default RoPE types, custom block masks, or unsupported GQA head grouping.

### Rewrite: MoE split checkpoint weights to packed expert tensors

Source checkpoint pattern:

```text
experts.E.gate_proj.weight [I, H]
experts.E.up_proj.weight   [I, H]
experts.E.down_proj.weight [H, I]
```

Runtime source pattern:

```text
gate_up_proj [E, 2*I, H] = concat([gate_proj, up_proj], dim=0) per expert
down_proj    [E, H, I]   = stack(down_proj per expert)
```

Preconditions:

- All experts present and have equal shapes.
- No expert bias tensors.
- Config `num_experts` matches loaded expert count.

Failure cases:

- Already-packed checkpoint tensors; quantized or transposed expert formats; missing experts.

Parity sketch: after conversion, compare one selected expert's `linear(...).chunk(2)` against split-weight eager computation.

### Rewrite: MoE eager routing to grouped GEMM

Source pattern:

```text
softmax(router_logits) -> topk -> per-expert loop -> index_add_
```

Replacement:

```text
TopKRouter -> token/expert pair sort or grouped offsets -> grouped GEMM gate_up -> SiLU*up -> grouped GEMM down -> weighted segment sum
```

Preconditions:

- Inference only; deterministic tie policy documented.
- `num_experts_per_tok <= num_experts`.
- Route weights match source fp32 softmax and optional `norm_topk_prob`.
- Accumulation order/tolerance accepted for low precision.

Failure cases:

- Need exact eager `index_add_` ordering in fp32/fp16; unsupported expert implementation selection; tensor-parallel expert sharding.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
if logits_to_keep == 1: gather final hidden row -> GEMM [B, H] x [H, vocab]
```

Preconditions:

- Generation only needs last-token logits.
- Caller does not request full sequence logits or tensor indices.

Failure cases:

- Prompt logprob scoring or training loss needs broader logits.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 reduction and dtype cast. FlexOlmo has q_norm, k_norm, post-attention norm, post-FFN norm, and final norm.
- Causal attention prefill/decode with RoPE and KV cache. This dominates decode and must store post-RoPE K/V exactly.
- MoE router plus grouped expert GEMMs. Top-k all-expert routing for 7x7B means active compute is large; a Python/eager per-expert loop is not viable.
- SwiGLU expert fusion: `gate_up GEMM -> chunk -> silu(gate) * up -> down GEMM`.
- Last-token-only LM head GEMM for decode.

Medium priority:

- Grouped QKV projection launch, while preserving q/k normalization.
- Router softmax+topk kernel for `[B*S, E]`, with small E (2 or 7 observed).
- Fused residual add after post-norm, if it composes with RMSNorm output.
- Weight-load conversion and packed expert materialization with artifact-visible provenance.

Lower priority:

- Aux router loss and router logits capture.
- Non-default attention backend variants (`flex_attention`) beyond core dense causal semantics.
- Tensor-parallel plans.

## 11. Runtime staging plan

Stage 1: config and weight admission.

- Parse `flex_olmo` configs.
- Admit bias-free MHA observed configs first: `num_key_value_heads == num_attention_heads`, `rope_scaling == null`, `attention_bias == false`.
- Implement split-to-packed expert weight conversion and validate expert count/shape.

Stage 2: single-block and dense operators.

- Implement embeddings, RMSNorm, RoPE, q/k/v/o linears, SwiGLU expert math for manually selected routes.
- Validate one layer without cache and with small random tensors.

Stage 3: MoE routing parity.

- Implement router softmax/topk and a deterministic grouped or batched expert path.
- Compare eager source and DinoML for 2-expert and 7-expert configs.

Stage 4: prefill CausalLM parity.

- Run full 1-block then N-block prefill logits with full causal mask.
- Support `logits_to_keep` for last-token-only output.

Stage 5: decode cache.

- Add per-layer KV cache ABI `[B, KVH, T, D]`, stored after RoPE.
- Validate prompt prefill plus one-token and multi-token decode against Transformers.

Stage 6: optimized kernels.

- Enable FlashAttention/SDPA-equivalent path, grouped expert GEMMs, and fused RMSNorm/SwiGLU kernels.

Stage 7: production scheduling.

- Add batch/sequence sweeps, cache memory planning, optional continuous batching, and weight-residency policies for 33B-size all-fp32 checkpoints.

## 12. Parity and validation plan

- Config tests: reject unsupported non-null `clip_qkv`, mismatched expert counts, unsupported RoPE variants, and tokenizer/model position-limit confusion.
- Weight conversion tests: split checkpoint expert weights -> packed tensors; verify shapes `[E, 2*I, H]` and `[E, H, I]`.
- RMSNorm random tensor parity: fp32, fp16, bf16; tolerance fp32 `1e-5`, fp16/bf16 around `2e-2` depending accumulation.
- RoPE parity: random Q/K plus position ids with cache offset; verify post-RoPE K is what enters cache.
- Attention parity: eager dense attention for MHA first, then GQA if admitted.
- Router parity: softmax/topk outputs for E=2/K=2 and E=7/K=7, including tie behavior documentation.
- Expert parity: selected expert computation for one token, multiple tokens, duplicate expert assignments, and all-expert routing.
- Single-block parity: compare hidden states after attention branch, MoE branch, and final block output.
- Full prefill parity: logits for short prompts and padded batches.
- Decode parity: prefill cache then generate one token; compare logits and cache shapes after each step.
- End-to-end text smoke: tokenizer chat template -> logits -> one greedy token, but sampling itself can remain external.

## 13. Performance probes

- Tokenization/chat-template throughput separately from model runtime.
- Prefill latency and throughput over `S = 128, 512, 2048, 4096`.
- Decode tokens/sec for `B = 1, 4, 16, 32` and cache lengths up to 4096.
- Attention backend comparison: eager matmul, SDPA/FlashAttention-equivalent, GQA fallback if future configs need it.
- MoE routing probe: router softmax/topk time for E=2 vs E=7.
- Expert grouped GEMM probe: active top-k pairs per token for K=2 and K=7; compare sorted grouped GEMM vs batched gather GEMM.
- LM head probe: full sequence logits vs last-token-only logits.
- KV cache memory usage: `[layers, B, KVH, T, D, 2]` with dtype sweep.
- Weight residency/loading probe: 33B fp32 checkpoint load time and memory, plus future quantized/GGUF conversion if added.

No benchmark measurements are included here; these are source-derived probe recommendations.

## 14. Skip/defer list

- Training, labels, and aux router load-balancing loss.
- Gradient checkpointing.
- Output router logits capture unless needed for diagnostics.
- Beam search and sampling controllers; keep outside first graph target.
- Tensor parallel and pipeline parallel plans.
- Non-default RoPE scaling/types.
- GQA/MQA runtime if initial checkpoint target is observed MHA only.
- Quantized, FP8, MXFP4, or GGUF formats; current public config/source basis is fp32 safetensors.
- Alternative MoE expert implementations unless selected as DinoML's optimized provider path.
- Sequence lengths beyond model `max_position_embeddings=4096`.

## 15. Final implementation checklist

- [ ] Parse `FlexOlmoConfig` and checkpoint overrides.
- [ ] Enforce first-stage admission: causal LM, no sliding window, no non-default RoPE, no attention bias unless implemented.
- [ ] Load token embedding, final norm, and LM head weights.
- [ ] Implement split expert checkpoint conversion to packed `gate_up_proj` and `down_proj`.
- [ ] Implement RMSNorm with fp32 variance accumulation.
- [ ] Implement default RoPE cos/sin and Q/K rotation.
- [ ] Implement bias-free Q/K/V/O projections with q/k post-projection RMSNorm.
- [ ] Implement dense causal attention prefill.
- [ ] Implement KV cache update storing post-RoPE K/V.
- [ ] Implement decode attention against cache.
- [ ] Implement router softmax/topk with optional `norm_topk_prob`.
- [ ] Implement grouped or batched expert GEMMs plus weighted top-k accumulation.
- [ ] Implement post-attention and post-FFN norm-before-residual ordering.
- [ ] Implement final `logits_to_keep` handling.
- [ ] Add config/weight-conversion parity tests.
- [ ] Add RMSNorm/RoPE/attention/MoE unit parity tests.
- [ ] Add one-block, full-prefill, and decode parity tests.
- [ ] Benchmark prefill, decode, MoE grouped GEMM, LM head, and cache memory.
