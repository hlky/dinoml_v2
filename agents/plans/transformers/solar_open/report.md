# DinoML Transformers Audit: solar_open

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: upstage/Solar-Open-100B
Config source: https://huggingface.co/upstage/Solar-Open-100B/raw/main/config.json
Source files inspected:
- X:/H/transformers/src/transformers/models/solar_open/configuration_solar_open.py
- X:/H/transformers/src/transformers/models/solar_open/modeling_solar_open.py
- X:/H/transformers/src/transformers/models/solar_open/modular_solar_open.py
- X:/H/transformers/docs/source/en/model_doc/solar_open.md
- X:/H/transformers/tests/models/solar_open/test_modeling_solar_open.py
Representative configs inspected:
- upstage/Solar-Open-100B config.json and generation_config.json
- SSON9/solar-open-tiny-dummy config.json
Evidence snapshots written:
- agents/plans/transformers/solar_open/solar_open_100b_config_snapshot.json
- agents/plans/transformers/solar_open/solar_open_tiny_dummy_config_snapshot.json
Any missing files or assumptions:
- Full 100B safetensors were not downloaded. Weight tensor shapes below are derived from source modules plus config dimensions.
- The public config uses legacy rope_scaling fields; Transformers normalizes these into rope_parameters during config construction.
- HF access for config files was public at inspection time. No gated config blocker was encountered.
```

The generated source says `configuration_solar_open.py` and `modeling_solar_open.py` are generated from `modular_solar_open.py`; treat `modular_solar_open.py` as the editing basis for upstream parity notes.

## 2. High-level architecture

SolarOpen is a text-only causal decoder with per-layer MoE feed-forward blocks. It is Llama-like in residual/RMSNorm structure, but the attention projection widths are not the usual `hidden_size == num_heads * head_dim` shape, and the FFN is a routed plus shared MoE.

```text
token ids -> embedding -> repeated decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

```text
CPU tokenizer/chat template -> GPU embedding/prefill -> autoregressive decode with KV cache -> logits_to_keep LM head -> generation controller
```

Independently cacheable stages:

- Tokenization and chat template are CPU/data-pipeline work.
- Decoder prefill emits per-layer self-attention KV cache.
- Decode reuses per-layer KV cache. Cached keys are stored after RoPE is applied; values are stored after V projection.
- LM head can be last-token-only using `logits_to_keep`.

## 3. Important config dimensions

| Field | Solar-Open-100B config | Source default | Operator significance |
| --- | ---: | ---: | --- |
| `hidden_size` | 4096 | 4096 | Residual width and embedding width |
| `num_hidden_layers` | 48 | 48 | Decoder block count |
| `num_attention_heads` | 64 | 64 | Q/O attention head count |
| `num_key_value_heads` | 8 | 8 | GQA KV head count |
| `head_dim` | 128 | 128 | Per-head width |
| Q/O width | 8192 | 8192 | `64 * 128`; larger than hidden size |
| K/V width | 1024 | 1024 | `8 * 128` |
| `vocab_size` | 196608 | 196608 | Embedding and LM head rows |
| `max_position_embeddings` | 131072 | 131072 | RoPE/cache length basis |
| `rope_theta` | 1000000 | 1000000 | Default base |
| RoPE scaling | yarn factor 2, original max 65536 | config-normalized | Long-context behavior |
| `moe_intermediate_size` | 1280 | 1280 | Routed expert inner width |
| `n_routed_experts` | 128 | 128 | Expert tensor leading dimension |
| `n_shared_experts` | 1 | 1 | Shared SwiGLU width multiplier |
| `num_experts_per_tok` | 8 | 8 | Top-k experts per token |
| `n_group` / `topk_group` | omitted in 100B config, effective 1/1 | 1/1 | Group-limited routing |
| `hidden_act` | omitted in 100B config, effective `silu` | `silu` | SwiGLU activation |
| `attention_bias` | omitted in 100B config, effective false | false | Q/K/V bias gate |
| `tie_word_embeddings` | false | false | LM head and embedding are separate physical parameters |
| `use_cache` | true | true | Generation cache supported |
| dtype | `torch_dtype`: bfloat16 | not fixed by source | Main checkpoint storage/inference dtype |

Representative checkpoint sweep:

| Model | Layers | Hidden | Q heads / KV heads / head dim | Q width | MoE experts / top-k | Context | RoPE | dtype |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | --- |
| `upstage/Solar-Open-100B` | 48 | 4096 | 64 / 8 / 128 | 8192 | 128 / 8 | 131072 | yarn factor 2, original 65536, theta 1e6 | bf16 |
| `SSON9/solar-open-tiny-dummy` | 24 | 2048 | 16 / 4 / 128 | 2048 | 16 / 4 | 4096 | yarn factor 2, original 2048, theta 1e6 | bf16 |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim` for the 100B config: residual width is 4096, Q/O attention width is 8192. Do not derive Q projection width from hidden size.
- GQA is required: `num_key_value_heads < num_attention_heads`, with 8 query groups per KV head in the 100B config.
- Q/K/V attention bias is config-gated; public 100B omits it, so source default `attention_bias=False` applies.
- The routed MoE uses packed expert weights: `gate_up_proj` stores gate and up rows together as `[expert, 2 * moe_intermediate, hidden]`, split in gate/up order.
- `intermediate_size=10240` exists in the public 100B config, but the executed layer MLP is `SolarOpenMoE`; routed experts use `moe_intermediate_size=1280`, and shared experts are constructed with `moe_intermediate_size * n_shared_experts`.
- The source supports grouped expert routing through `n_group` and `topk_group`; public 100B config omits both, so defaults 1/1 apply.
- RoPE may arrive as legacy `rope_scaling` in checkpoint config or as normalized `rope_parameters`; first integration should support normalized config and preserve legacy conversion parity.
- The test suite explicitly checks that partial RoPE init with yarn keeps `partial_rotary_factor=1.0` and default theta `1_000_000`.
- `use_qk_norm` and `first_k_dense_replace` appear in representative configs but are not read by the inspected SolarOpen modeling source. Treat them as ignored for this source basis, not required runtime ops.
- `lm_head.weight` is listed as a tied-weight key, but `tie_word_embeddings=false` in representative configs, so first parity should treat embedding and LM head as separate weights unless a checkpoint says otherwise.
- No NCHW/NHWC convolutional layout exists in the core model. Layout-sensitive rewrites are mainly `transpose`, `view`, `reshape`, `contiguous`, expert packed-weight row order, and attention `[B, H, S, D]` versus `[B, S, H, D]` semantics.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab=196608, hidden=4096)`.
- `view` / `reshape` token flattening: `[B, S, H] -> [B*S, H]` for router and experts.
- Projection reshape: Q `[B,S,8192] -> [B,S,64,128] -> [B,64,S,128]`; K/V `[B,S,1024] -> [B,S,8,128] -> [B,8,S,128]`.
- `transpose(1, 2)` and `contiguous()` around attention output.
- `unsqueeze`, `expand`, `reshape` for KV repeat: `[B,8,S,128] -> [B,8,8,S,128] -> [B,64,S,128]`.
- `one_hot`, `permute(2,1,0)`, `where`, `nonzero`, `index_add_` for eager MoE dispatch fallback.

Neural network primitives:

- RMSNorm over last dim, fp32 variance: hidden 4096.
- Dense linear projections: Q `4096 -> 8192`, K `4096 -> 1024`, V `4096 -> 1024`, O `8192 -> 4096`.
- Router linear in fp32: `4096 -> 128` for 100B.
- Routed expert packed gate/up linear per active expert: `[tokens_e,4096] x [2560,4096] -> [tokens_e,2560]`, chunk into two `[tokens_e,1280]` tensors.
- Routed down linear per active expert: `1280 -> 4096`.
- Shared expert SwiGLU: gate `4096 -> 1280`, up `4096 -> 1280`, down `1280 -> 4096` for `n_shared_experts=1`.
- LM head: `4096 -> 196608`, with `logits_to_keep` slice before projection.

Attention primitives:

- Causal self-attention, GQA, prefill and decode.
- Backend dispatch through Transformers attention interface: eager, SDPA, FlashAttention, and flex attention are declared supported by the model class.
- Eager path: matmul QK^T, additive causal mask, fp32 softmax over key length, dropout during training, matmul with V.

MoE routing ops:

- `sigmoid(router_logits)`.
- Add fp32 `e_score_correction_bias` of shape `[n_routed_experts]`.
- Group scores: reshape to `[tokens, n_group, n_routed_experts // n_group]`, top-2 per group, sum, top-`topk_group` groups.
- `scatter_` group mask, mask fill with `-inf`, top-k experts, `gather` selected sigmoid weights.
- Optional top-k probability normalization with epsilon `1e-20`, multiply by `routed_scaling_factor`.

Position/rotary ops:

- RoPE inverse frequency generation from theta, partial rotary factor, and head dim.
- Dynamic RoPE update decorator for advanced rope types such as yarn/dynamic.
- Cos/sin computed in fp32 with autocast disabled, then cast to hidden dtype.
- `rotate_half`: split last dimension into halves and concatenate `[-x2, x1]`.

Generation/cache ops:

- DynamicCache creation when `use_cache=True` and no cache is supplied.
- Cache update per layer after RoPE is applied to K.
- Causal mask creation with `input_embeds`, optional attention mask, position ids, and cache.
- `logits_to_keep` as int slice or tensor indices to reduce LM head work.

Preprocessing-coupled ops:

- Tokenization/chat template is outside the core model. Docs use `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- Generation config supplies `do_sample=true`, `temperature=0.8`, `top_p=0.95`; docs also recommend `top_k=50`.

Distributed/tensor-parallel ops:

- Source config includes a default TP plan: Q/K/V colwise, O rowwise, expert gate/up packed colwise, expert down rowwise, routed experts as MoE TP experts, LM head colwise gather output.
- DinoML first parity can ignore TP if loading single-rank weights, but weight names and packed expert layout must not conflict with future sharding.

## 5. Layer/block breakdown

Decoder block, repeated 48 times for Solar-Open-100B:

```text
x: [B, S, 4096]
residual = x
x = RMSNorm(x)                                      # fp32 variance, weight [4096]
q = Linear(x, 4096 -> 8192, bias=attention_bias)    # [B, S, 64, 128]
k = Linear(x, 4096 -> 1024, bias=attention_bias)    # [B, S, 8, 128]
v = Linear(x, 4096 -> 1024, bias=attention_bias)    # [B, S, 8, 128]
q,k = RoPE(q,k, position_ids)                       # q [B,64,S,128], k [B,8,S,128]
k,v = cache.update(k,v, layer_idx)                  # if cache supplied
attn = causal GQA(q,k,v, mask, scale=1/sqrt(128))
attn = Linear(attn, 8192 -> 4096, bias=False)
x = residual + attn
residual = x
x = RMSNorm(x)
router_logits = Linear_fp32(x, 4096 -> 128)
topk_indices, topk_weights = grouped sigmoid top-k router
routed = sum_experts(SwiGLU_expert(x) * topk_weights)
shared = Linear(SiLU(Linear(x, 4096 -> 1280)) * Linear(x, 4096 -> 1280), 1280 -> 4096)
x = residual + routed + shared
```

Final head:

```text
x = final RMSNorm(x)                                # [B, S, 4096]
x = x[:, logits_to_keep, :]                         # usually last token for decode
logits = Linear(x, 4096 -> 196608, bias=False)
```

## 6. Attention requirements

- Variant: causal decoder self-attention.
- Head structure: GQA. Solar-Open-100B has 64 Q heads, 8 KV heads, head dim 128, repeat factor 8.
- Query width: 8192. Key/value projection widths: 1024 each. Attention output width before O projection: 8192.
- Masking: `create_causal_mask` combines causal, padding/attention mask, position ids, and cache length. Eager attention adds mask to QK scores before softmax.
- Cache: autoregressive self-attention KV cache, one key tensor and one value tensor per layer. Key is cached after RoPE. Shape before repeat is `[B, 8, cached_S, 128]`; eager attention repeats to `[B, 64, cached_S, 128]` before matmul.
- Prefill: Q length and K/V length are both current sequence length before cache append; with a cache, K/V length becomes previous plus current.
- Decode: Q length is usually 1; K/V length is full cached length.
- Softmax math order in eager path: `matmul * scale`, add mask, softmax with `dtype=torch.float32`, cast to query dtype, dropout, matmul V.
- Optimized backend compatibility: source declares FlashAttention, SDPA, and flex attention support. DinoML parity should first match eager math, then admit fused attention with explicit GQA, post-RoPE-cache, additive mask, and fp32 softmax equivalence conditions.
- No cross-attention, ALiBi, sliding-window/local attention, packed/varlen metadata, or block sparse attention appears in the inspected SolarOpen source.

## 7. Position encoding and custom math

RoPE is applied to Q and K after projection/transpose and before KV cache update. The source computes default inverse frequencies with:

```python
base = config.rope_parameters["rope_theta"]
dim = int(head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_solar_open_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

For yarn or other non-default RoPE types, `ROPE_INIT_FUNCTIONS[rope_type]` supplies inverse frequencies and an `attention_scaling` multiplier. The `dynamic_rope_update` decorator can update cached frequencies for advanced RoPE types, so the lowering contract must not freeze RoPE tables solely at static max length unless the selected rope type admits that precomputation.

Precomputable:

- Default RoPE inv_freq/cos/sin up to a bounded max position if position IDs are monotonic and max length is fixed.

Dynamic:

- Position IDs can be caller-supplied or generated from cache length.
- Yarn/dynamic RoPE can change effective frequency buffers through the Transformers helper.

## 8. Preprocessing and input packing

Core GPU inputs:

- `input_ids: [B, S]` or `inputs_embeds: [B, S, 4096]`, exactly one required.
- Optional `attention_mask`.
- Optional `position_ids: [B, S]`; otherwise generated as `arange(S) + past_seen_tokens` and unsqueezed to `[1, S]`.
- Optional `past_key_values`.

CPU/data-pipeline work:

- Tokenizer and chat template construction. The docs example uses messages, `apply_chat_template`, and `add_generation_prompt=True`.
- Sampling policy from generation config/docs: `temperature=0.8`, `top_p=0.95`, docs also mention `top_k=50`.

There is no image/audio/video branch, placeholder-token scatter, external processor grid metadata, or multimodal embedding stitch in this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: grouped QKV projection pack

Source pattern:

```text
q = Linear(x, Wq)
k = Linear(x, Wk)
v = Linear(x, Wv)
reshape/transposes to [B,H,S,D]
```

Replacement pattern:

```text
single or grouped GEMM producing packed [q, k, v] buffers -> split with explicit widths
```

Preconditions:

- Same input tensor, same dtype, same bias policy.
- Preserve split order exactly: Q rows first if weights are physically packed by rewrite, then K, then V.
- Widths are `[8192, 1024, 1024]` for 100B, not equal thirds.
- Fallback when `attention_bias=True` and bias packing is not implemented.

Parity test sketch:

- Random one-layer Q/K/V projection with 100B dimensions reduced to small fixtures where `hidden_size != q_width`.
- Compare split tensors before RoPE.

### Rewrite: RoPE plus attention prefill/decode fusion

Source pattern:

```text
q,k projection -> RoPE(q,k) -> cache update -> attention backend
```

Replacement:

```text
fused RoPE-aware GQA attention
```

Preconditions:

- Cached keys must be stored post-RoPE.
- Additive mask semantics and fp32 softmax must match selected tolerance.
- Query head count and KV head count must be passed separately; no materialized repeat is required inside an optimized kernel if GQA is native.
- Dynamic/yarn RoPE tables must be current for the runtime position IDs.

Failure cases:

- Non-default attention backend with incompatible mask ABI.
- Rope type requiring dynamic update not represented in the compiled artifact.

### Rewrite: MoE packed expert GEMM

Source pattern:

```text
for active experts:
  gate, up = linear(tokens_e, gate_up_proj[expert]).chunk(2)
  y = down(silu(gate) * up)
  index_add_(token_idx, y * topk_weight)
```

Replacement:

```text
router top-k -> token/expert bucketing -> grouped GEMM gate_up -> fused SiLU multiply -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- Expert weight layout `[expert, 2 * moe_intermediate, hidden]` and gate/up chunk order preserved.
- Top-k indices and weights are source-equivalent, including group mask, `sorted=False` top-k tie behavior tolerance, and optional normalization.
- Accumulation into output must support repeated tokens across top-k experts.

Failure cases:

- Unstable `topk` tie ordering may change exact expert choices for equal scores.
- Layout rewrite that transposes expert storage without updating packed chunk axis.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None) or tensor indices
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
carry only requested hidden positions into LM head GEMM
```

Preconditions:

- `logits_to_keep` is statically or runtime-known.
- For tensor indices, preserve PyTorch indexing semantics.

### Guarded layout rewrite notes

- The model is not NCHW/NHWC, so no convolution channel-last pass is needed.
- Add a conceptual `no_layout_translation()` guard around:
  - RoPE last-dimension split/concat; changing the head-dim axis breaks parity.
  - Attention tensors while transposed as `[B,H,S,D]`; softmax axis must stay key sequence.
  - Packed expert `gate_up_proj`; the split axis is row/output dimension.
  - Router group reshape `[tokens, n_group, experts_per_group]`; expert axis order is semantic.
- Candidate layout optimizations may eliminate transposes only when all consumers agree on `[B,S,H,D]` versus `[B,H,S,D]` and the attention backend ABI is updated accordingly.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over 4096 with fp32 variance: two per block plus final norm.
- GQA FlashAttention/SDPA prefill and decode with native KV heads, avoiding materialized `repeat_kv`.
- MoE router and grouped expert GEMMs: the routed MoE dominates compute and requires token bucketing/scatter-add support.
- Fused SwiGLU for routed and shared experts.
- Last-token-only LM head for decode.

Medium priority:

- Q/K/V projection packing with unequal Q and KV widths.
- RoPE table generation/application fused with Q/K layout transform.
- Causal mask creation and attention mask canonicalization for static/paged decode.
- Expert top-k route normalization fused after router logits.

Lower priority:

- Training dropout, loss, and gradient-checkpointing paths.
- Tensor-parallel sharding plans.
- Sampling controller kernels; start with CPU/controller parity.

## 11. Runtime staging plan

Stage 1: config and weight schema admission.

- Parse normalized and legacy RoPE config forms.
- Load embedding, per-layer attention, RMSNorm, packed expert tensors, router weights/bias, shared experts, final norm, LM head.
- Reject unsupported quantized/packed formats with clear messages.

Stage 2: eager one-block parity.

- Implement RMSNorm, Q/K/V/O projections with unusual Q width, RoPE, eager GQA attention, MoE eager routing, and shared expert.
- Use tiny config dimensions for first tests.

Stage 3: full prefill parity.

- Run all layers without cache on tiny dummy and a synthetic reduced config.
- Validate logits for fixed prompts.

Stage 4: decode with KV cache.

- Store post-RoPE K and V per layer as `[B, num_kv_heads, S_cache, head_dim]`.
- Validate one-token incremental decode against full prefill continuation.

Stage 5: optimized attention.

- Add GQA fused attention with native KV heads, additive mask, fp32-softmax parity envelope, and RoPE/cache ABI guards.

Stage 6: optimized MoE.

- Lower router/top-k/group mask and grouped GEMMs.
- Add expert bucketing and scatter-add provider contracts.

Stage 7: production scheduling.

- Paged KV cache, batching, and tensor parallel can follow after single-rank numerical parity is stable.

Initial stubs:

- Sampling can use existing host generation logic.
- Training loss/dropout can be deferred.
- Expert implementation can start eager/bounded before adding optimized grouped kernels.

## 12. Parity and validation plan

- Config parsing tests:
  - Public 100B legacy `rope_scaling` normalizes to yarn `rope_parameters`.
  - Partial yarn config preserves `partial_rotary_factor=1.0` and theta `1e6`.
- Operator tests:
  - RMSNorm fp32 variance against PyTorch for fp32/fp16/bf16.
  - RoPE default and yarn table/application against Transformers for random position IDs.
  - Q/K/V projection shape test where `hidden_size != num_heads * head_dim`.
  - Eager GQA attention with and without cache; verify cached K is post-RoPE.
  - Router top-k/group selection for `n_group=1` and synthetic `n_group>1`.
  - Packed expert gate/up split and weighted `index_add` against source eager path.
- Single-layer parity:
  - Reduced config with deterministic weights, no cache, then with one-step cache.
- Full-model parity:
  - `SSON9/solar-open-tiny-dummy` prefill logits and greedy generation for a short prompt.
  - Optional 100B shape-only manifest/load validation without materializing full weights.
- Recommended tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for block outputs.
  - bf16/fp16 eager: start `rtol=3e-2`, `atol=3e-2` for full-layer logits; tighten per fused kernel once accumulation policies are fixed.
- Decode parity:
  - Compare full prefill logits for prefix plus next token against prefill + cached one-token decode.

Do not mark optimized fused attention or MoE complete until both prefill and decode/cache parity pass.

## 13. Performance probes

- Tokenizer/chat-template throughput versus GPU prefill.
- Prefill-only latency sweep over sequence lengths: 1k, 4k, 16k, 64k, 131k if memory permits.
- Decode tokens/sec sweep over batch size and cache length.
- KV cache memory: `layers * 2 * B * num_kv_heads * S * head_dim * dtype_size`.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible, DinoML fused GQA.
- Router/top-k latency and selected expert histogram.
- MoE grouped GEMM throughput by tokens per expert and top-k.
- Shared expert versus routed expert compute split.
- LM head last-token-only versus full-sequence logits.
- Weight load and optional quantized/dequant provider comparison if GGUF or other converted weights are introduced later.

## 14. Skip/defer list

- Training, loss parity, dropout, and gradient checkpointing.
- Tensor-parallel and pipeline-parallel execution, despite source TP/PP plans.
- Multi-GPU expert parallelism.
- Quantized or GGUF converted weights unless explicitly selected as the weight source.
- Sampling quality parity beyond generation-controller parameter checks.
- Rare non-default rope types beyond yarn/default until config coverage requires them.
- Flex attention backend parity until eager/SDPA or FlashAttention path is stable.

## 15. Final implementation checklist

- [ ] Parse `SolarOpenConfig`, including legacy `rope_scaling` and normalized `rope_parameters`.
- [ ] Load embeddings, attention weights, packed expert weights, router weights/bias, shared expert weights, final norm, and LM head.
- [ ] Add shape guards for `hidden_size`, `num_attention_heads`, `num_key_value_heads`, and explicit `head_dim`.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement RoPE default/yarn table generation and application.
- [ ] Implement causal GQA attention eager path with post-RoPE KV cache.
- [ ] Implement `logits_to_keep` LM head slicing.
- [ ] Implement router sigmoid/group/top-k/normalization path.
- [ ] Implement packed expert gate/up split, SiLU multiply, down projection, top-k weighting, and scatter-add.
- [ ] Add guarded QKV packing rewrite with unequal width split tests.
- [ ] Add guarded fused GQA attention rewrite without materialized KV repeat.
- [ ] Add guarded MoE grouped-GEMM rewrite preserving expert packed layout.
- [ ] Add tiny dummy config prefill and decode parity tests.
- [ ] Add one-block randomized parity tests for source-specific operators.
- [ ] Benchmark prefill, decode, attention backend, MoE routing/grouped GEMM, and last-token LM head.

