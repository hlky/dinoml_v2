# HYV3 Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tencent/Hy3-preview
Config source: raw Hugging Face Hub config.json snapshots, saved beside this report
Source files inspected:
  transformers/src/transformers/models/hy_v3/configuration_hy_v3.py
  transformers/src/transformers/models/hy_v3/modeling_hy_v3.py
  transformers/src/transformers/models/hy_v3/modular_hy_v3.py
  transformers/src/transformers/models/hy_v3/__init__.py
  transformers/src/transformers/conversion_mapping.py
  transformers/tests/models/hy_v3/test_modeling_hy_v3.py
Any missing files or assumptions: no processor/image/audio files exist for this family; tokenizer work is CPU/generation-controller ABI, not neural graph work.
```

Primary source URLs at the pinned commit:

- `configuration_hy_v3.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/hy_v3/configuration_hy_v3.py
- `modeling_hy_v3.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/hy_v3/modeling_hy_v3.py
- `modular_hy_v3.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/hy_v3/modular_hy_v3.py

`modeling_hy_v3.py` and `configuration_hy_v3.py` are generated from `modular_hy_v3.py`; future Transformers source edits should target the modular file. Reported runtime behavior is from the generated in-library source, with the modular file used to identify intentional deltas from inherited Llama/Apertus/MiniMax/Mixtral/Qwen3-MoE pieces.

Hub/config snapshots saved here:

- `config_tencent_Hy3-preview.json`
- `config_tencent_Hy3-preview-Base.json`
- `config_hf-internal-testing_HYV3-tiny-random.json`
- `config_tiny-random_hy-v3.json`
- `config_yujiepan_hy-v3-tiny-random.json`
- `generation_config_tencent_Hy3-preview.json`
- `tokenizer_specials_tencent_Hy3-preview.md`

Hub access note: official Tencent repos inspected here are public and not gated as of 2026-05-13. Quantized MLX/community derivatives exist, but their storage/runtime contracts are out of scope for the first DinoML in-library source audit.

## 2. High-level architecture

HYV3 is a text-only causal decoder LM with a dense first block and sparse MoE blocks after that. The primary DinoML target should be `HYV3ForCausalLM` prefill/decode logits for text generation.

```text
tokenizer/chat template -> input_ids/attention_mask
  -> token embedding
  -> 80 decoder blocks: causal GQA attention + dense-or-MoE MLP
  -> final RMSNorm
  -> untied LM head
  -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, tool/reasoning prompt tokens, generation config.
- GPU/runtime graph: embeddings, RoPE, causal GQA attention with KV cache, RMSNorms, dense MLP, sparse MoE routing/expert execution, final LM head.
- Independently cacheable state: per-layer self-attention KV cache. No vision/audio/projector branch exists.
- Optional/stub-first heads: only causal LM is implemented; training loss and router-logit capture can be deferred for inference.

## 3. Important config dimensions

| Field | Official Hy3-preview | Source default | Runtime significance |
|---|---:|---:|---|
| `vocab_size` | 120832 | 120832 | embedding and LM head width |
| `hidden_size` | 4096 | 4096 | residual width |
| `num_hidden_layers` | 80 | 80 | decoder depth |
| `mlp_layer_types` | omitted, effective `dense` then 79 `sparse` | same | layer 0 dense, layers 1-79 MoE |
| `num_attention_heads` | 64 | 64 | query heads |
| `num_key_value_heads` | 8 | 8 | GQA KV heads |
| `head_dim` | 128 | 128 | q/k/v head width |
| Q projection | 4096 -> 8192 | same | no bias by default |
| K/V projection | 4096 -> 1024 each | same | no bias by default |
| O projection | 8192 -> 4096 | same | no bias by default |
| `intermediate_size` | 13312 | 13312 | dense layer-0 SwiGLU width |
| `moe_intermediate_size` | 1536 | 1536 | routed and shared expert SwiGLU width |
| `num_experts` | 192 | 192 | routed experts per sparse layer |
| `num_experts_per_tok` | 8 | 8 | top-k routing |
| `num_shared_experts` | 1 | 1 | always-active shared MLP width = 1536 |
| `router_scaling_factor` | 2.826 | 2.826 | scales normalized top-k weights |
| `rms_norm_eps` | 1e-5 | 1e-5 | all RMSNorms |
| `rope_parameters` | default RoPE, theta 11158840.0 | standardized default | full-head RoPE |
| `max_position_embeddings` | 262144 | 131072 | config/source variation; official is 256K |
| `attention_bias` | omitted/effective false | false | q/k/v/o bias admission |
| `mlp_bias` | omitted/effective false | false | dense/shared MLP bias admission |
| `tie_word_embeddings` | false | false | LM head is separate from token embedding |
| `use_cache` | true | true | DynamicCache KV path |
| checkpoint dtype | BF16 weights plus small F32 tensors from safetensors metadata | unspecified | source itself is dtype-generic |

Representative checkpoint sweep:

| Repo | Official? | Purpose | Layers | Hidden | Heads/KV | Experts/top-k | Context | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `tencent/Hy3-preview` | yes | instruct/chat | 80 | 4096 | 64/8 | 192/8 | 262144 | public, 112 shards, tokenizer has reasoning/tool chat template |
| `tencent/Hy3-preview-Base` | yes | base model | 80 | 4096 | 64/8 | 192/8 | 262144 | public, 99 shards, older `generate_config.json` spelling |
| `hf-internal-testing/HYV3-tiny-random` | no | Transformers integration test | 8 | 768 | 12/4 | 16/2 | 4096 | test parity target, `enable_moe_fp32_combine=true` |
| `tiny-random/hy-v3` | no | public tiny/debug | 4 | 8 | 8/4 | 192/8 | 262144 | deliberately odd: `hidden_size != num_heads * head_dim` |
| `yujiepan/hy-v3-tiny-random` | no | public tiny/debug | 4 | 8 | 8/4 | 192/8 | 262144 | same shape trap as `tiny-random/hy-v3` |

## 3a. Family variation traps

- Do not infer `q_proj` output from `hidden_size`; source uses `num_attention_heads * head_dim`, and debug configs can have `hidden_size != num_attention_heads * head_dim`.
- GQA is required: official checkpoints use 64 query heads and 8 KV heads, so attention either repeats KV by 8 or uses a native GQA backend.
- Source always constructs q/k RMSNorm modules before RoPE. Config fields like `qk_norm=true` match official configs but are not used as optional switches by the inspected source.
- `mlp_layer_types` is generated if omitted: first layer dense, all later layers sparse. DinoML should validate length and supported values.
- Sparse expert weights are packed tensors, not one module per expert: `gate_up_proj[num_experts, 2 * moe_intermediate, hidden]`, split as gate then up; `down_proj[num_experts, hidden, moe_intermediate]`.
- Router uses sigmoid, top-k over `routing_weights + e_score_correction_bias`, gathers raw sigmoid weights, normalizes selected weights, then multiplies by `router_scaling_factor`.
- `e_score_correction_bias` is a per-sparse-layer FP32 buffer initialized to zeros and kept in FP32; upstream weight conversion can map `mlp.expert_bias` to it.
- Current in-library source ignores or hardcodes several config-advertised flags: `enable_attention_fp32_softmax`, `enable_lm_head_fp32`, `moe_router_enable_expert_bias`, `moe_router_use_sigmoid`, `route_norm`, `qk_norm`, `use_grouped_mm`, `first_k_dense_replace`, `expert_hidden_dim`, and `num_nextn_predict_layers`. Treat them as metadata unless a future source revision reads them.
- MTP is explicitly not supported by the in-library model: unexpected keys matching `model.layers.80.*` are ignored. Do not include MTP in first DinoML parity.
- No sliding window: the model calls the normal causal mask helper and comments that there is no sliding window.
- Tensor-parallel plans exist in config metadata, including `packed_colwise` and `moe_tp_experts`, but single-GPU DinoML can defer distributed sharding.
- No NCHW/NHWC layout issue: this is text-only rank-2/3 sequence math.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token embedding lookup `[B,S] -> [B,S,4096]`.
- `arange`, add past length, unsqueeze for `position_ids [1,S]`.
- Reshape/view from `[B,S,*]` to `[B,S,H,D]`, transpose to `[B,H,S,D]`, contiguous reshape back.
- Slice/index for `logits_to_keep`: `hidden_states[:, slice_indices, :]`.
- Gather/scatter-like MoE token routing: `topk`, `gather`, `one_hot`, `permute`, `where`, expert-specific token gather, `index_add_`.

Neural primitives:

- RMSNorm over last dim with FP32 variance and cast back.
- Linear/GEMM without bias for official q/k/v/o, MLP, experts, LM head.
- Optional linear bias if `attention_bias` or `mlp_bias` configs are admitted.
- SwiGLU: `silu(gate) * up`, then down projection.
- Residual adds, dtype casts for optional FP32 MoE combine.

Attention primitives:

- Causal self-attention, GQA/MQA-capable.
- RoPE applied to q/k after q/k RMSNorm and before cache update.
- Eager fallback: q @ k^T * `head_dim^-0.5`, mask add, FP32 softmax, dropout in training only, attn @ v.
- Production path can use Transformers attention backends (`flash_attention`, SDPA, flex attention) if parity preserves math order.

Position/rotary ops:

- Default RoPE with theta 11158840.0 over `head_dim`.
- Cos/sin computed in FP32 then cast to hidden dtype.
- Rotate-half convention splits first half/second half of head dimension.

Generation/cache ops:

- DynamicCache or equivalent static cache with per-layer K/V tensors `[B, num_key_value_heads, T, head_dim]`.
- Cache stores post-qk-norm, post-RoPE keys; values are unrotated V projections.
- Cache update appends per layer before attention backend sees key/value.

MoE routing and expert ops:

- Router GEMM `hidden -> num_experts`.
- Sigmoid router probabilities, expert-bias add for choice only, top-k unsorted.
- Per-token top-k normalization and scaling.
- Grouped expert GEMM or segmented token batches per expert.
- Shared expert MLP always active in sparse layers.

Preprocessing-coupled ops:

- Tokenizer/chat template controls BOS/user/assistant/EOS/tool/reasoning tokens. This is outside the core graph but required for chat parity.

Quantized/packed weight metadata:

- In-library source has no quantized weight path. Community MLX/FP8/FP4 derivatives should be separate loading/provider audits.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)
q = Linear(hidden_size -> num_attention_heads * head_dim)(x)
k = Linear(hidden_size -> num_key_value_heads * head_dim)(x)
v = Linear(hidden_size -> num_key_value_heads * head_dim)(x)
q = reshape/transposed to [B, Hq, S, D]
k = reshape/transposed to [B, Hkv, S, D]
v = reshape/transposed to [B, Hkv, S, D]
q = RMSNorm(q) over D
k = RMSNorm(k) over D
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx) if cache is present
a = causal GQA attention(q, k, v, mask)
x = residual + Linear(Hq * D -> hidden_size)(a)
residual = x
x = RMSNorm(x)
x = dense SwiGLU MLP if layer type is dense, else HYV3 MoE
x = residual + x
```

Official layer 0 dense MLP:

```text
gate_proj: 4096 -> 13312
up_proj:   4096 -> 13312
down_proj: 13312 -> 4096
activation: silu
bias: false unless admitted by config
```

Official sparse layers 1-79:

```text
router: 4096 -> 192, FP32 input/weight
routed experts:
  gate_up_proj: [192, 3072, 4096], split last output as gate/up of 1536 each
  down_proj:    [192, 4096, 1536]
shared expert:
  gate_proj/up_proj: 4096 -> 1536
  down_proj: 1536 -> 4096
combine: routed + shared, FP32 combine only if config enables it
```

Final head:

```text
final_norm: RMSNorm(4096)
lm_head: Linear(4096 -> 120832), bias false, not tied to embedding
```

## 6. Attention requirements

HYV3 requires autoregressive self-attention only.

- Pattern: causal self-attention.
- Head form: GQA. Official `Hq=64`, `Hkv=8`, `groups=8`, `D=128`.
- Query width: 8192 for official checkpoint.
- Key/value width: 1024 each for official checkpoint.
- Output width before `o_proj`: 8192.
- Mask: causal mask from `create_causal_mask`, plus user `attention_mask` if supplied. No sliding window.
- Packed/varlen: generic Transformers masking utilities may detect packed sequences from `position_ids`, but HYV3 source does not implement custom packed metadata. First DinoML pass can reject packed/varlen.
- RoPE: q/k RMSNorm first, then RoPE, then cache update.
- Cache: per-layer K/V stored before repeat expansion, shape `[B, 8, T, 128]` official. Attention backend may logically repeat to `[B,64,T,128]`.
- Eager softmax: FP32 softmax regardless of config flag, then cast to query dtype.
- FlashAttention/SDPA compatibility: source advertises flash attention, SDPA, flex attention, and attention backend dispatch. DinoML should first match eager math, then add GQA flash prefill/decode kernels.

## 7. Position encoding and custom math

RoPE inverse frequency:

```python
base = config.rope_parameters["rope_theta"]
dim = config.head_dim
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

Runtime RoPE table generation:

```python
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Cos/sin can be precomputed for bounded static positions or generated per run. Position IDs include `past_seen_tokens` during decode. Dynamic/non-default RoPE types are technically accepted by shared RoPE helpers if config changes, but official configs use `rope_type="default"`; first DinoML admission should accept only default HYV3 RoPE.

## 8. Preprocessing and input packing

Neural graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one required.
- Optional `attention_mask`, typically `[B,S]` with padding.
- Optional `position_ids [B,S]`; if omitted, generated as contiguous positions plus cache length.
- Optional `past_key_values`; if `use_cache=true` and none is supplied, source creates `DynamicCache`.

Tokenizer/generation ABI:

- Official instruct tokenizer has BOS `<｜hy_begin▁of▁sentence｜>`, pad `<｜hy_▁pad▁｜>`, user `<｜hy_User｜>`, assistant `<｜hy_Assistant｜>`, EOS `<｜hy_eos｜>`, plus reasoning/tool XML-like tokens in the chat template.
- `generation_config_tencent_Hy3-preview.json` sets `do_sample=true`, `temperature=0.9`, `top_k=-1`, `top_p=1`, EOS 120025, pad 120002.
- Reasoning/tool prompt construction is generation-controller/tokenizer work. It should not enter the GPU graph except as token IDs.

There are no image/audio/video processors, placeholder embedding stitches, `cu_seqlens`, or modality grids.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> packed QKV projection

Source pattern:

```text
q = Linear(x, q_proj)
k = Linear(x, k_proj)
v = Linear(x, v_proj)
```

Replacement:

```text
packed = Linear(x, concat_rows(q, k, v))
split as [Hq*D, Hkv*D, Hkv*D]
```

Preconditions:

- Same input tensor, dtype, device, and no per-projection side effects.
- Bias settings equal or bias tensors packed in the same split order.
- Preserve source split order `q, k, v`; do not assume Qwen2-MoE conversion order unless weight loader has already materialized HF keys.

Failure cases:

- Tensor-parallel sharded weights without a manifest-level packing plan.
- Quantized/packed storage formats not decoded to dense compatible rows.

Parity test sketch: compare q/k/v tensors before q/k RMSNorm on random `[B,S,H]` for official and tiny configs.

### Rewrite: q/k RMSNorm + RoPE fusion

Source pattern:

```text
q = RMSNorm(q); k = RMSNorm(k); q,k = RoPE(q,k,cos,sin)
```

Replacement: fused per-head norm and rotate kernel.

Preconditions:

- RMSNorm epsilon matches config.
- Normalize over head dimension only.
- RoPE full `head_dim`, default rotate-half layout.
- Cos/sin dtype/cast order preserved within tolerances.

Failure cases:

- Non-default RoPE types or partial rotary factors.
- Debug shapes with very small hidden sizes must still use projected head width.

### Rewrite: MoE expert loop -> grouped segmented GEMM

Source pattern:

```text
top_k_index = topk(sigmoid(router) + bias)
for expert in expert_hit:
    tokens = where(expert_mask[expert])
    y = down(silu(gate(tokens)) * up(tokens))
    final.index_add_(tokens, y * selected_weight)
```

Replacement:

```text
route -> token sort/bucket by expert -> grouped GEMM gate_up -> SwiGLU -> grouped GEMM down -> weighted scatter-add
```

Preconditions:

- `top_k` static and small.
- Expert tensors use HF packed layout `[E, 2I, H]` with gate then up.
- `topk(sorted=False)` order does not affect final sum except floating-point accumulation order; define tolerance.
- Preserve expert-bias use for choice only, not gathered probability.

Failure cases:

- General `index_add_` with arbitrary duplicate semantics admitted without deterministic segmented reductions.
- Quantized/community weights with different packing.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: for decode or `logits_to_keep=1`, apply LM head only to selected final positions.

Preconditions:

- Caller does not request full logits.
- Generation controller accepts sliced logits.

Failure cases:

- Training loss or full-sequence perplexity.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm, including small per-head q/k RMSNorm.
- GQA causal attention with default RoPE, prefill and decode KV cache.
- MoE router + top-k + segmented grouped GEMM path.
- SwiGLU MLP and expert `gate_up -> silu*up -> down` fusion.
- Last-token LM head for decode.

Medium priority:

- Packed QKV GEMM with split metadata.
- RoPE table generation/cache for long-context positions.
- Shared expert MLP plus routed-output combine, including optional FP32 combine.
- Causal mask generation and padding-mask fusion into attention backend.

Lower priority:

- Router logits output capture for diagnostics.
- Distributed/tensor-parallel plan support.
- Non-default attention backends after eager parity is stable.

## 11. Runtime staging plan

Stage 1: config/weight loading.

- Parse HYV3 config, standardize RoPE defaults, reject unsupported MTP execution.
- Load dense HF weights, including packed expert tensors and `e_score_correction_bias`.
- Validate `mlp_layer_types`.

Stage 2: one-block and tiny-model eager parity.

- Implement embeddings, RMSNorm, q/k/v/o GEMMs, RoPE, causal eager attention, dense SwiGLU.
- Use `hf-internal-testing/HYV3-tiny-random` for manageable parity.

Stage 3: sparse MoE parity.

- Implement router sigmoid/top-k/normalization/scaling and a simple segmented expert execution path.
- Validate routed output plus shared expert combine.

Stage 4: full prefill logits.

- Run all layers without cache for short sequences, compare logits.
- Keep attention eager/dense initially.

Stage 5: decode with KV cache.

- Add K/V append cache `[B,Hkv,T,D]`.
- Validate one-token and multi-token decode against Transformers.

Stage 6: optimized kernels.

- Introduce GQA FlashAttention decode/prefill, grouped MoE GEMM, packed QKV, last-token logits.

Stage 7: production scheduling.

- Batch/paged KV cache, long-context memory probes, optional tensor parallel, quantized loading/provider paths.

## 12. Parity and validation plan

- RMSNorm random tensors: FP32, BF16, FP16; tolerance `1e-5` fp32, `1e-2` reduced precision.
- RoPE parity: compare q/k after q/k RMSNorm and RoPE for short, offset, and long position IDs.
- Attention single-layer parity: no cache prefill and cache decode; compare attention output before `o_proj`.
- Dense layer-0 block parity on random hidden states.
- MoE router parity: compare logits, top-k indices as sets/order-sensitive where stable, top-k weights.
- Expert parity: compare routed expert output and final `index_add_` accumulation for repeated token/expert hits.
- Tiny model full logits: `hf-internal-testing/HYV3-tiny-random`, batched padded inputs, BF16.
- Decode token parity: greedy `max_new_tokens` against Transformers integration-test pattern.
- Official smoke: load config/weights metadata and run a very short prefill if memory permits; otherwise validate shard/index and per-layer tensor shapes.

## 13. Performance probes

- Prefill-only tokens/sec across sequence lengths: 1K, 4K, 16K, 64K, 256K if memory allows.
- Decode tokens/sec for batch sizes 1, 4, 16 with KV cache length sweep.
- KV cache memory: `[layers, B, 2, Hkv, T, D]` BF16 footprint.
- MoE routing overhead: router/top-k/scatter time versus grouped expert GEMM time.
- Expert load balance: token counts per expert for representative prompts and batch sizes.
- Dense versus packed QKV projection time.
- Last-token logits versus full-sequence logits time and memory.
- BF16 eager attention versus FlashAttention/SDPA backend parity and speed.
- Weight loading time and memory residency for 298.8B-parameter BF16 checkpoint.
- Quantized derivative load/dequant/provider probes only after a separate quantized storage audit.

## 14. Skip/defer list

- Training loss, labels, gradient checkpointing.
- Router auxiliary loss: source returns `aux_loss=None`.
- MTP/speculative layer execution: current source ignores `model.layers.80.*`.
- Tensor parallel and pipeline parallel execution.
- Non-default RoPE variants.
- General packed/varlen sequence support.
- Community MLX/FP8/FP4/NVFP4 quantized checkpoints.
- Tool/reasoning chat-template rendering inside DinoML runtime; keep in tokenizer/controller.
- Full 256K production scheduling before short-context prefill/decode parity.

## 15. Final implementation checklist

- [ ] Parse HYV3 config and standardize omitted defaults.
- [ ] Reject unsupported source/config combinations: MTP execution, non-default RoPE, unsupported `mlp_layer_types`, quantized derivative formats.
- [ ] Load embeddings, q/k/v/o, RMSNorms, LM head, dense MLP weights.
- [ ] Load packed expert weights `[E,2I,H]` and `[E,H,I]` plus FP32 `e_score_correction_bias`.
- [ ] Implement RMSNorm and q/k head RMSNorm.
- [ ] Implement default HYV3 RoPE.
- [ ] Implement causal GQA attention eager path.
- [ ] Implement KV cache update/read with stored `[B,Hkv,T,D]` K/V.
- [ ] Implement dense layer-0 SwiGLU MLP.
- [ ] Implement HYV3 router sigmoid/top-k/normalize/scale.
- [ ] Implement sparse expert execution and weighted scatter-add.
- [ ] Implement shared expert combine with config-controlled FP32 combine.
- [ ] Implement sliced LM head for `logits_to_keep`.
- [ ] Add single-block, MoE, prefill, and decode parity tests against Transformers.
- [ ] Benchmark prefill, decode, MoE routing, grouped expert GEMM, KV cache memory, and logits slicing.
