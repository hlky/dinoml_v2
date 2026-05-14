# DeepSeek V3 Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary scope is native `deepseek_v3` in-library inference for `DeepseekV3ForCausalLM`. Representative configs inspected:

- `deepseek-ai/DeepSeek-V3` config snapshot: `_sources/deepseek-ai__DeepSeek-V3.config.json`
- `deepseek-ai/DeepSeek-V3-Base` config snapshot: `_sources/deepseek-ai__DeepSeek-V3-Base.config.json`
- `deepseek-ai/DeepSeek-V3-0324` config snapshot: `_sources/deepseek-ai__DeepSeek-V3-0324.config.json`
- `deepseek-ai/DeepSeek-V3.1` and `deepseek-ai/DeepSeek-V3.1-Base` config snapshots: `_sources/deepseek-ai__DeepSeek-V3.1*.config.json`
- `bzantium/tiny-deepseek-v3` config snapshot: `_sources/bzantium__tiny-deepseek-v3.config.json`
- `deepseek-ai/DeepSeek-V3.2*` configs were fetched only to classify scope. They use `model_type="deepseek_v32"` and are out of scope for this `deepseek_v3` report.

Config source:

- Local config class: `X:/H/transformers/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py`
- HF raw configs fetched from `https://huggingface.co/<repo>/raw/main/config.json`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py`
- `X:/H/transformers/src/transformers/models/deepseek_v3/modular_deepseek_v3.py`
- `X:/H/transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`
- Comparison only: `X:/H/transformers/src/transformers/models/deepseek_v2/{configuration,modeling,modular}_deepseek_v2.py`
- Remote-code snapshot for divergence checks only: `_sources/deepseek-ai__DeepSeek-V3.{configuration_deepseek.py,modeling_deepseek.py}`

Any missing files or assumptions:

- `modeling_deepseek_v3.py` is generated from `modular_deepseek_v3.py`; future source edits should target the modular file, while DinoML parity should follow the generated runtime code.
- Official configs still include `auto_map` entries for remote code and older `transformers_version` values. This report is scoped to the pinned native Transformers implementation, not the legacy remote-code class.
- Official configs contain fields that the native source does not read, including `ep_size`, `moe_layer_freq`, `num_nextn_predict_layers`, `scoring_func`, `topk_method`, and training-only/tiny fields such as `aux_loss_alpha`/`seq_aux`. Treat those as config provenance/traps unless a separate remote-code audit owns them.
- Official V3/V3.1 configs advertise FP8 quantization metadata. The native modeling file describes tensor math, not the full quantized weight-loading backend; DinoML should route FP8/dequant handling through its own constant/provider plan.

## 2. High-level architecture

DeepSeek V3 is a text-only causal decoder with MLA-style latent attention projections and DeepSeekMoE feed-forward blocks. The first `first_k_dense_replace` decoder layers use dense SwiGLU MLPs; later layers use routed experts plus a shared expert MLP. The primary runtime target is autoregressive causal LM inference.

Dataflow:

```text
token ids / embeddings -> token embedding -> N decoder blocks -> final RMSNorm
  -> optional logits_to_keep slice -> LM head -> logits -> sampling/controller
```

Decoder block dataflow:

```text
RMSNorm -> MLA self-attention with RoPE and cache -> residual
RMSNorm -> dense SwiGLU or routed MoE + shared expert -> residual
```

Stage decomposition:

- CPU/data pipeline: tokenizer/chat-template, attention mask construction inputs, position id preparation if not supplied.
- GPU/runtime prefill: embeddings, all decoder layers, causal mask, RoPE cos/sin, cache writes, logits.
- GPU/runtime decode: one or few new tokens, cache read/update per layer, last-token logits via `logits_to_keep`.
- Independently optimizable regions: RMSNorm, MLA projection/RoPE/attention/cache path, MoE routing and expert GEMMs, final last-token-only LM head.

Heads implemented by native source:

- `DeepseekV3ForCausalLM`: required for this report.
- `DeepseekV3Model`: required as the base decoder.
- `DeepseekV3ForSequenceClassification`, `DeepseekV3ForTokenClassification`: optional/deferred for causal-LM integration.

## 3. Important config dimensions

Effective native defaults from `DeepseekV3Config` unless noted:

| Field | Default/native value | Operator significance |
|---|---:|---|
| `vocab_size` | 129280 | Embedding and LM head width. |
| `hidden_size` | 7168 | Residual width. |
| `num_hidden_layers` | 61 | Decoder block count. |
| `num_attention_heads` | 128 | Q/O projection head count. |
| `num_key_value_heads` | 128 | No GQA expansion in official configs; native supports `num_attention_heads // num_key_value_heads`. |
| `qk_nope_head_dim` | 128 | Non-rotary q/k head slice. |
| `qk_rope_head_dim` | 64 | Rotary q/k slice and native `head_dim`. |
| `qk_head_dim` | 192 | Computed as `qk_nope_head_dim + qk_rope_head_dim`; note `hidden_size != heads * qk_head_dim`. |
| `v_head_dim` | 128 | V and attention-output head dim. |
| `q_lora_rank` | 1536 | Query latent rank; `None` switches to direct Q projection. |
| `kv_lora_rank` | 512 | KV latent rank. |
| `intermediate_size` | 18432 | Dense early-layer SwiGLU intermediate. |
| `moe_intermediate_size` | 2048 | Per routed expert intermediate. |
| `n_routed_experts` | 256 | Routed expert count in production configs. |
| `n_shared_experts` | 1 | Shared expert MLP has `moe_intermediate_size * n_shared_experts`. |
| `num_experts_per_tok` | 8 | Routed top-k experts per token. |
| `n_group` / `topk_group` | 8 / 4 | Group-limited routing before expert top-k. |
| `first_k_dense_replace` | 3 | Layers `0..2` dense MLP, layers `3..60` MoE. |
| `norm_topk_prob` | true | Normalize selected routing weights before scaling. |
| `routed_scaling_factor` | 2.5 | Multiplies selected routing weights. |
| `max_position_embeddings` | 4096 default class, 163840 official configs | Long context through YaRN config in checkpoints. |
| `rope_scaling` / `rope_parameters` | official `type/yarn`, factor 40 | PreTrainedConfig converts legacy `rope_scaling` to `rope_parameters`; native RoPE uses standardized dict. |
| `rope_interleave` | true native default | Chooses interleaved RoPE application path. Official configs omit it, so native default applies. |
| `hidden_act` | `silu` | SwiGLU gate activation. |
| `attention_bias` | false | Q-a, KV-a, O projection bias disabled in official configs; native supports true. |
| `attention_dropout` | 0.0 | Inference dropout is 0. |
| `tie_word_embeddings` | false | LM head is a separate logical parameter despite `_tied_weights_keys` metadata. |
| `torch_dtype` | bf16 in configs | Config provenance, not a source math guarantee. |
| `quantization_config` | FP8 in official large configs | Weight format provenance, not implemented in the modeling graph itself. |

Representative checkpoint sweep:

| Repo/config | Scope | Layers | Routed experts | Groups/topk group | Context | Quantization | Notes |
|---|---|---:|---:|---:|---:|---|---|
| `bzantium/tiny-deepseek-v3` | debug/native-compatible | 6 | 8 | 2 / 2 | 163840 | none in config | Same hidden size/head dims as production; useful for shape parity without 61 layers. |
| `deepseek-ai/DeepSeek-V3-Base` | production/base | 61 | 256 | 8 / 4 | 163840 | FP8 e4m3, block 128x128 | Same native graph shape; base weights. |
| `deepseek-ai/DeepSeek-V3` | production/chat | 61 | 256 | 8 / 4 | 163840 | FP8 e4m3, block 128x128 | Config has `transformers_version=4.33.1` and remote-code `auto_map`; native source now exists. |
| `deepseek-ai/DeepSeek-V3-0324` | production/chat update | 61 | 256 | 8 / 4 | 163840 | FP8 e4m3, block 128x128 | Same operator structure as V3 config. |
| `deepseek-ai/DeepSeek-V3.1` | production/chat update | 61 | 256 | 8 / 4 | 163840 | FP8 e4m3 with `scale_fmt=ue8m0` | Still `model_type=deepseek_v3`; same native graph shape. |
| `deepseek-ai/DeepSeek-V3.2*` | out of scope | 61 | 256 | 8 / 4 | 163840 | FP8 | `model_type=deepseek_v32`, `architectures=DeepseekV32ForCausalLM`; route to separate audit. |

## 3a. Family variation traps

- `hidden_size` is not the QK projection width. Production QK head width is `192`, so `128 * 192 = 24576`, not `7168`.
- `head_dim` in config is set to `qk_rope_head_dim` (`64`) for RoPE frequency generation, while attention QK math uses `qk_head_dim=192` and V uses `v_head_dim=128`.
- `num_key_value_heads` defaults to `num_attention_heads` if omitted. Native eager attention can repeat KV for GQA, but official V3 configs use full 128 KV heads.
- `q_lora_rank=None` changes query projection structure from `q_a -> RMSNorm -> q_b` to one direct `q_proj`.
- `attention_bias=True` adds bias to `q_a_proj`, `kv_a_proj_with_mqa`, and `o_proj`, not to `q_b_proj`/`kv_b_proj`.
- Official configs omit `rope_interleave`; native default `True` applies and changes the RoPE tensor shuffle before applying `rotate_half`.
- Official configs use legacy `rope_scaling`; native config standardization maps it into `rope_parameters` with `rope_type="yarn"` and `rope_theta`.
- Official configs contain `num_nextn_predict_layers=1`, but native source does not implement MTP heads and ignores unexpected `model.layers.61.*` on load. Do not require MTP for the native causal-LM target.
- Official configs contain `topk_method="noaux_tc"` and `scoring_func="sigmoid"`, but native V3 routing is hard-coded to sigmoid, group-limited top-k, and correction bias; it does not branch on those strings.
- Tiny config uses the same full hidden width but fewer layers/experts, so it is a good operator smoke test but not a memory/performance proxy for 256 experts.
- V3.2 configs are structurally similar but have a different `model_type`; do not silently feed them into this report's class.
- No vision/audio/channel layout translation is involved. Tensor layout traps are sequence/head layout transposes around attention and FlashAttention compatibility, not NHWC/NCHW.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `[batch, seq] -> [batch, seq, 7168]`.
- `view`, `reshape`, `transpose(1, 2)`, `contiguous`, `split`, `cat`, `expand`, `slice`.
- Causal mask creation from `attention_mask`, `position_ids`, and cache length.
- `logits_to_keep` integer or tensor indexing on sequence dimension before LM head.

Neural network primitives:

- RMSNorm over last dim with fp32 variance and cast back.
- Dense linear/GEMM without bias for embeddings/MLP/down/projection defaults.
- Optional bias on Q-a/KV-a/O if config requests it.
- SwiGLU: `down(silu(gate(x)) * up(x))`.
- LM head: `Linear(7168 -> 129280, bias=False)`.

Attention primitives:

- MLA query path: `Linear(7168 -> 1536)` + RMSNorm + `Linear(1536 -> 128 * 192)`.
- MLA KV path: `Linear(7168 -> 512 + 64)` split into latent KV and rotary K; RMSNorm on latent KV; `Linear(512 -> 128 * (128 + 128))`.
- RoPE on only 64 dims of Q/K, then concat with 128 non-rotary dims.
- Causal self-attention with cache update. Native supports eager, SDPA, FlashAttention, and flex attention through `ALL_ATTENTION_FUNCTIONS`.
- FlashAttention requested path pads V from 128 to 192 when QK/V head dims differ, then slices output back to 128.

Position/rotary ops:

- RoPE cos/sin generation in fp32 with YaRN support through shared `ROPE_INIT_FUNCTIONS`.
- Interleaved RoPE tensor rearrangement when `rope_interleave=True`.
- YaRN attention scaling can additionally modify attention scale by `mscale^2`.

MoE/routing ops:

- Router GEMM in fp32: `Linear(7168 -> n_routed_experts)` with weight `[experts, hidden]`.
- Sigmoid routing scores plus fp32 `e_score_correction_bias`.
- Group score: reshape `[tokens, n_group, experts_per_group]`, top-2 per group, sum.
- Top groups: `topk(group_scores, topk_group)`, scatter group mask, masked fill inactive experts to `-inf`.
- Expert top-k: top `num_experts_per_tok`; gather uncorrected sigmoid weights; optional normalization; scale by `routed_scaling_factor`.
- Expert execution: one-hot expert mask, per-hit token gather, per-expert packed gate/up GEMM, SiLU multiply, down GEMM, multiply route weight, `index_add_` into token buffer.
- Shared expert: dense SwiGLU with intermediate `moe_intermediate_size * n_shared_experts`.

Generation/cache ops:

- `DynamicCache(config=config)` when `use_cache=True` and no cache is supplied.
- Per-layer cache update stores post-RoPE full key states `[batch, kv_heads, cached_seq, 192]` and values `[batch, kv_heads, cached_seq, 128]` for native source.
- Cache length drives generated `position_ids`.
- Beam cache reorder is inherited from `GenerationMixin`/base cache utilities, not custom in this file.

Distributed/tensor-parallel metadata:

- `base_model_tp_plan` marks expert tensors as packed/expert-parallel and shared/dense MLP projections as column/row-wise.
- Native single-process graph still contains all experts logically. DinoML should preserve parameter identity and expert axis layout if adding TP or expert parallel later.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(7168, eps=1e-6)(x)
attn = DeepseekV3Attention(x, causal_mask, position_embeddings, cache)
x = residual + attn

residual = x
x = RMSNorm(7168, eps=1e-6)(x)
if layer_idx < first_k_dense_replace:
    x = dense SwiGLU: down(silu(gate(x)) * up(x))
else:
    x = routed MoE(x) + shared SwiGLU(x)
x = residual + x
```

Dense MLP layers `0..2` for production configs:

```text
gate_proj: Linear(7168 -> 18432, bias=False)
up_proj:   Linear(7168 -> 18432, bias=False)
down_proj: Linear(18432 -> 7168, bias=False)
```

MoE layers `3..60` for production configs:

```text
router: Linear(7168 -> 256, fp32, bias=False)
routed experts:
  gate_up_proj: [256, 4096, 7168] packed as 2 * moe_intermediate_size rows
  down_proj:    [256, 7168, 2048]
shared expert:
  gate/up: Linear(7168 -> 2048, bias=False)
  down:    Linear(2048 -> 7168, bias=False)
```

Attention per layer in production configs:

```text
q_a_proj: Linear(7168 -> 1536, bias=False)
q_a_layernorm: RMSNorm(1536)
q_b_proj: Linear(1536 -> 24576, bias=False)  # 128 heads * 192 qk dim

kv_a_proj_with_mqa: Linear(7168 -> 576, bias=False)  # 512 latent + 64 rotary k
kv_a_layernorm: RMSNorm(512)
kv_b_proj: Linear(512 -> 32768, bias=False)  # 128 heads * (128 k_nope + 128 v)

o_proj: Linear(16384 -> 7168, bias=False)  # 128 heads * 128 v dim
```

## 6. Attention requirements

Required variant: causal self-attention with MLA projections and RoPE.

- Causality: causal decoder mask from `create_causal_mask`.
- Head structure: production configs have 128 Q heads and 128 KV heads. Native supports GQA/MQA mathematically through `repeat_kv` if `num_key_value_heads < num_attention_heads`, but official configs do not exercise it.
- Q/K dims: Q and K have 192 dims per head after concat: 128 no-RoPE + 64 RoPE.
- V dim: 128 dims per head.
- Masking: additive causal/padding mask passed to attention backend. Eager path adds mask before softmax.
- Softmax math: eager path computes `matmul(q, k.T) * scaling`, adds mask, softmaxes in fp32, casts to query dtype, applies dropout, then matmuls V.
- Scaling: default `192 ** -0.5`; for non-default RoPE with `mscale_all_dim`, native V3 multiplies by `yarn_get_mscale(factor, mscale_all_dim) ** 2`.
- Cache: native stores expanded full key/value states after RoPE and after KV latent expansion, not compressed MLA latents. Per layer, production cache shapes are:
  - key: `[batch, 128, total_seq, 192]`
  - value: `[batch, 128, total_seq, 128]`
- FlashAttention/SDPA compatibility: native dispatch goes through `ALL_ATTENTION_FUNCTIONS`. When FlashAttention is requested and QK/V dims differ, V is padded to 192 before attention and output is sliced back to 128 before O projection. A DinoML fused attention kernel can avoid physical padding if it supports asymmetric QK/V dimensions.
- Packed/varlen support: current native generated file delegates to backend kwargs; explicit varlen unpadding lives in older remote code, not in the pinned native V3 implementation.

Eager fallback is too slow for production because it materializes full attention weights `[batch, heads, query, key]` and repeats KV for GQA if present. First optimized path should target causal prefill/decode attention with asymmetric QK/V dims and cache writes.

## 7. Position encoding and custom math

RoPE is applied only to the `qk_rope_head_dim=64` suffix of Q and to the standalone K rotary projection. The non-rotary 128-dim slices bypass RoPE.

Native default/interleaved RoPE sketch:

```python
def apply_deepseek_v3_rope(q_rot, k_rot, cos, sin, interleave=True):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    if interleave:
        q_rot = q_rot.view(B, H, S, D // 2, 2).transpose(4, 3).reshape(B, H, S, D)
        k_rot = k_rot.view(B, 1, S, D // 2, 2).transpose(4, 3).reshape(B, 1, S, D)
    q_out = q_rot * cos + rotate_half(q_rot) * sin
    k_out = k_rot * cos + rotate_half(k_rot) * sin
    return q_out, k_out
```

YaRN scale helper:

```python
def yarn_get_mscale(scale=1, mscale=1):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0
```

Precomputable:

- Inverse frequencies and static cos/sin tables for fixed max positions and dtype can be precomputed or cached.
- Position-specific cos/sin depends on runtime `position_ids`, cache length, and dynamic RoPE update behavior for advanced rope types.

Dynamic inputs:

- `position_ids` default to `arange(seq) + past_seen_tokens`.
- Cache length changes RoPE positions during decode.

## 8. Preprocessing and input packing

Text inputs:

- `input_ids` shape `[batch, seq]` or `inputs_embeds` shape `[batch, seq, 7168]`; exactly one must be supplied.
- Tokenizer/chat template is outside the model graph.
- `attention_mask`, if supplied, enters causal mask construction. DinoML can accept a prebuilt canonical additive mask initially, then reproduce `create_causal_mask` later.
- `position_ids`, if omitted, are generated from current sequence length and cache length.

No multimodal preprocessing, image/audio processors, placeholder scatter, discrete image codebook, or `cu_seqlens` metadata is required for native DeepSeek V3 causal LM.

Generation-controller behavior outside core graph:

- Sampling, temperature/top-p, stop tokens, chat templates, and endpoint behavior are outside `modeling_deepseek_v3.py`.
- `logits_to_keep` is in the model forward and should be implemented early to avoid full-sequence vocab GEMMs in decode.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MLA projection decomposition to explicit GEMMs

Source pattern:

```text
q = q_b(RMSNorm(q_a(x)))
compressed_kv = kv_a(x)
k_latent, k_rot = split(compressed_kv, [kv_lora_rank, qk_rope_head_dim])
kv = kv_b(RMSNorm(k_latent))
k_nope, v = split(kv, [qk_nope_head_dim, v_head_dim])
```

Replacement pattern:

```text
GEMM q_a -> RMSNorm -> GEMM q_b -> reshape/split
GEMM kv_a -> split -> RMSNorm -> GEMM kv_b -> reshape/split
```

Preconditions:

- Static config dims known.
- Preserve `attention_bias` on q-a/kv-a when enabled.
- Preserve fp32 RMSNorm math.

Shape equations:

- `q_b_out = [B, S, H * (D_nope + D_rope)]`
- `kv_a_out = [B, S, R_kv + D_rope]`
- `kv_b_out = [B, S, H * (D_nope + D_v)]`

Failure cases:

- Unknown `q_lora_rank=None` needs direct-Q branch.
- Config with `num_key_value_heads < num_attention_heads` must verify reshape/repeat semantics.

Parity test sketch:

- Random one-layer attention with fixed position ids and no cache; compare q/k/v tensors before attention.

### Rewrite: interleaved RoPE canonicalization

Source pattern:

```text
view(..., D/2, 2) -> transpose(last two small dims) -> reshape -> rotate_half RoPE
```

Replacement:

```text
single fused RoPE kernel with interleave flag
```

Preconditions:

- `rope_interleave=True`.
- Last dimension even and equals `qk_rope_head_dim`.
- Inputs contiguous or layout handled by kernel.

Failure cases:

- `rope_interleave=False` must use standard half-rotation order.
- Dynamic RoPE types must preserve attention scaling and frequency update.

Parity test sketch:

- Compare fused kernel against Python function for multiple `position_ids`, including decode offset.

### Rewrite: asymmetric FlashAttention without V padding

Source pattern:

```text
if flash and qk_head_dim != v_head_dim:
    value = pad(value, [0, qk_head_dim - v_head_dim])
attention(...)
output = output[..., :v_head_dim]
```

Replacement:

```text
Attention(Q dim 192, K dim 192, V dim 128) -> output dim 128
```

Preconditions:

- Backend supports different QK and V dimensions.
- Causal mask/cache semantics match source.
- Scaling matches native YaRN-adjusted scaling.

Failure cases:

- Backend requires equal head dims and cannot avoid padding.
- Attention backend returns weights for user-visible `output_attentions`; can defer weights.

Parity test sketch:

- Prefill and decode comparisons against native eager attention for small shapes.

### Rewrite: routed expert loop to grouped expert GEMM

Source pattern:

```text
one_hot(topk_indices) -> per expert token gather -> gate_up GEMM -> silu*up
-> down GEMM -> route weight multiply -> index_add
```

Replacement:

```text
router top-k -> token/expert bucketing -> grouped GEMM gate_up -> activation
-> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- Expert axis is weight dimension 0.
- `gate_up_proj[e]` row order is `[gate rows, up rows]`; split by `moe_intermediate_size`.
- Preserve unsorted top-k behavior only insofar as output is sum-reduction; deterministic tie parity may need source-compatible top-k.
- Accumulation order differences tolerated within reduced-precision tolerance.

Failure cases:

- Distributed expert parallel (`ep_size`) is config-only in native source but may matter for sharded checkpoints.
- Very small batch/decode may prefer persistent per-token expert kernels over full grouped GEMM.

Parity test sketch:

- Random router outputs with ties avoided; compare top-k indices/weights and final MoE output for tiny expert count, then production-like expert count with sparse hit sets.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None) if int else logits_to_keep
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather/slice hidden states -> GEMM hidden_to_vocab only for kept positions
```

Preconditions:

- `logits_to_keep` known as int or validated tensor index.
- For decode default, use last token only.

Failure cases:

- Training/loss requires labels over more positions.

Parity test sketch:

- Compare logits for `logits_to_keep=0`, `1`, `N`, and tensor indices.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 reduction: appears twice per block plus q/kv latent norms; cheap but very frequent.
- MLA projection + RoPE + attention prefill/decode: dominant non-MoE cost and cache ABI hinge.
- Causal attention with QK dim 192 and V dim 128: avoids source FlashAttention V pad/slice overhead.
- MoE routing + grouped expert GEMMs: production has 58 MoE layers, 256 experts, top-8; Python-style per-expert loops are not viable.
- Last-token-only LM head: reduces decode logits work from full sequence to kept positions.

Medium priority:

- Dense SwiGLU fusion for first three layers and shared experts.
- Router top-k/group mask kernel: sigmoid, correction bias, group top-2, group scatter, expert top-k, normalize/scale.
- Expert scatter-add with route-weight multiply.
- Cache update fused with K/V layout write.

Lower priority:

- Sequence/token classification heads.
- `output_attentions=True` materialization.
- Full remote-code legacy varlen FlashAttention unpadding path.
- MTP/next-token prediction auxiliary modules until a separate scope requires them.

## 11. Runtime staging plan

Stage 1: config/weights parser and tiny checkpoint graph construction.

- Parse native config, standardize `rope_scaling` to `rope_parameters`.
- Reject `model_type=deepseek_v32`.
- Load dense bf16/fp32 debug weights first; mark FP8 production weights as a separate constant/dequant milestone.

Stage 2: one dense decoder block parity.

- Implement RMSNorm, dense SwiGLU, MLA projection decomposition, RoPE, eager attention, O projection.
- Use no-cache prefill with small shapes.

Stage 3: MoE layer parity.

- Implement router top-k and naive expert execution first.
- Add grouped expert GEMM lowering after correctness.

Stage 4: full prefill parity.

- Run tiny 6-layer checkpoint and then 61-layer config with random/synthetic weights.
- Support causal mask and `logits_to_keep`.

Stage 5: decode with cache.

- Implement per-layer K cache `[B, H_kv, T, 192]`, V cache `[B, H_kv, T, 128]`.
- Validate position-id offset and cache update order.

Stage 6: optimized attention and MoE.

- Replace eager attention with asymmetric FlashAttention-style kernel.
- Replace naive MoE with token bucketing + grouped GEMM + scatter-add.

Stage 7: production weight formats and scheduling.

- Add FP8 block dequant/load policy and performance probes.
- Consider expert-parallel and tensor-parallel plans only after single-device parity.

Initially stub/defer:

- Sequence/token classification.
- Training/loss.
- `output_attentions`.
- MTP weights and `model.layers.61.*`.
- Remote-code-only behavior not present in native source.

## 12. Parity and validation plan

Custom op tests:

- RMSNorm fp32 variance against PyTorch for fp32/bf16/fp16 inputs.
- Standard and interleaved RoPE for random positions and decode offsets.
- YaRN scaling: compare inv_freq/cos/sin and attention scaling for official `rope_scaling`.
- Router: compare sigmoid, correction bias, group top-k mask, expert top-k indices/weights.
- Expert packed MLP: verify `gate_up_proj` split order and scatter-add output.

Single-layer parity:

- Dense layer `layer_idx=0`: no cache prefill, then decode with cache.
- MoE layer `layer_idx=3`: fixed router weights with no top-k ties; compare full block output.

After-N-layer parity:

- Tiny checkpoint first 1, 2, 6 layers with fixed input ids.
- Production-shaped random model for shape/cache stress without loading full weights.

Prefill/decode parity:

- Prefill logits with `logits_to_keep=0` and `logits_to_keep=1`.
- Decode one token after prefill; compare cache lengths and next-token logits.
- Attention backend parity: eager reference versus optimized asymmetric attention.

Tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for block-level; tighter for simple ops.
- bf16/fp16: start with `rtol=2e-2`, `atol=2e-2` for full blocks, then tune per kernel.
- MoE grouped GEMM/scatter may need relaxed tolerance because source accumulation order is per-expert loop with `index_add_`.

End-to-end:

- Tiny checkpoint greedy decode for a fixed prompt.
- Production config load smoke with FP8 path disabled/rejected clearly until implemented.

## 13. Performance probes

- Config/load probe: parse config and materialize parameter metadata, including expert tensor sizes and FP8 block metadata.
- Prefill sequence sweep: `S = 128, 512, 2048, 8192+`, batch sweep `B = 1, 4, 8`.
- Decode tokens/sec: `B = 1, 8, 32`, cache lengths `T = 1k, 16k, 64k, 160k`.
- Cache memory probe: per layer K/V bytes with K dim 192 and V dim 128; compare against hypothetical compressed latent cache only as a future optimization, not source parity.
- Attention backend comparison: eager, SDPA, FlashAttention-style asymmetric kernel, with and without V padding.
- MoE routing throughput: router top-k time, token bucketing time, expert GEMM time, scatter-add time.
- Expert load balance probe: random/router-real distributions, number of hit experts per batch, tokens per expert, grouped GEMM occupancy.
- LM head probe: full-sequence logits versus `logits_to_keep=1`.
- FP8/dequant probe: load-time dequant, pre-GEMM dequant, and resident dense bf16 memory pressure, labeled separately from source-derived graph facts.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Sequence and token classification heads.
- `output_attentions=True` materialized attention weights.
- Beam search and cache reorder beyond basic cache ABI smoke.
- Remote-code legacy details unless native parity fails for official checkpoints.
- MTP/`num_nextn_predict_layers` auxiliary modules and `model.layers.61.*` weights.
- FP8 production weight execution for first graph parity; require explicit constant/dequant plan before claiming production checkpoint support.
- Expert/tensor parallel execution from `ep_size` or TP metadata.
- V3.2 / `deepseek_v32`.

## 15. Final implementation checklist

- [ ] Parse `DeepseekV3Config`, including legacy `rope_scaling` normalization.
- [ ] Reject or route `deepseek_v32` configs separately.
- [ ] Load embeddings, decoder weights, final norm, and LM head with correct untied LM-head semantics.
- [ ] Decide and implement FP8 checkpoint constant/dequant policy before production checkpoint claims.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement MLA q/kv projection graph and shape checks.
- [ ] Implement YaRN/default RoPE and `rope_interleave=True` path.
- [ ] Implement causal mask and position-id/cache-length handling.
- [ ] Implement eager/reference attention with QK dim 192 and V dim 128.
- [ ] Implement KV cache ABI: K `[B,H,T,192]`, V `[B,H,T,128]`.
- [ ] Implement FlashAttention-style optimized attention without mandatory V padding.
- [ ] Implement router sigmoid/group/top-k/normalize/scale.
- [ ] Implement packed expert gate/up split and routed expert down projection.
- [ ] Implement shared expert and routed/shared sum.
- [ ] Add grouped expert GEMM + scatter-add rewrite.
- [ ] Implement `logits_to_keep` sequence slicing before LM head.
- [ ] Add one-block dense parity test.
- [ ] Add one-block MoE parity test.
- [ ] Add tiny-checkpoint prefill parity.
- [ ] Add decode-with-cache parity.
- [ ] Add prefill/decode/MoE/cache performance probes.
