# Transformers Audit: `hunyuan_v1_moe`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tencent/Hunyuan-A13B-Instruct
Config source: Hugging Face config.json plus native HunYuanMoEV1Config defaults
Source files inspected:
  transformers/src/transformers/models/hunyuan_v1_moe/configuration_hunyuan_v1_moe.py
  transformers/src/transformers/models/hunyuan_v1_moe/modeling_hunyuan_v1_moe.py
  transformers/src/transformers/models/hunyuan_v1_moe/modular_hunyuan_v1_moe.py
  transformers/tests/models/hunyuan_v1_moe/test_modeling_hunyuan_v1_moe.py
  transformers/src/transformers/modeling_rope_utils.py
Any missing files or assumptions:
  modeling_hunyuan_v1_moe.py is generated; modular_hunyuan_v1_moe.py is authoritative for future source edits.
  This report scopes native in-library `hunyuan_v1_moe`, not remote-code `hunyuan` quantized variants.
```

Representative Hub configs inspected: `tencent/Hunyuan-A13B-Instruct`, `tencent/Hunyuan-A13B-Instruct-GPTQ-Int4`, `tencent/Hunyuan-A13B-Instruct-FP8`, `bullerwins/Hunyuan-A13B-Instruct-hf`, and `bullerwins/Hunyuan-A13B-Instruct-GGUF` metadata. See `config_sweep.md` and `source_notes.md`.

## 2. High-level architecture

Primary runtime target: causal language-model inference for `HunYuanMoEV1ForCausalLM`.

```text
tokenizer/chat template -> token embeddings -> N causal MoE decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

The body is a Llama-like decoder with GQA, RoPE, per-head Q/K RMSNorm, and MoE feed-forward blocks. Each decoder layer has one causal self-attention block and one MoE block. The MoE block is not sparse-only: it adds a dense shared SwiGLU MLP branch to routed expert output.

Independently stageable pieces: tokenizer/chat-template ABI, embedding and one decoder block, prefill with full causal mask, decode with KV cache, MoE routing/expert dispatch, final logits. Sequence classification exists but is optional for the text-generation target.

## 3. Important config dimensions

| Field | Native default | A13B Instruct config | Runtime meaning |
| --- | ---: | ---: | --- |
| `hidden_size` | 4096 | 4096 | Decoder width |
| `num_hidden_layers` | 32 | 32 | Decoder blocks |
| `num_attention_heads` | 32 | 32 | Query heads |
| `num_key_value_heads` | defaults to heads | 8 | GQA with 4 query groups per KV head |
| `head_dim` | inferred 128 | 128 | Q/K/V head width |
| `intermediate_size` | 11008 | 3072 | Shared MLP and native expert intermediate |
| `num_experts` | 1 | 64 | Routed local experts |
| `moe_topk` | 1 | list of 8 for all 32 layers | Experts selected per token |
| `vocab_size` | 290943 | 128167 | Embedding and LM-head rows |
| `max_position_embeddings` | 2048 | 32768 | Model RoPE/cache length |
| `rope` | `rope_parameters` | legacy `rope_scaling` plus `rope_theta` | Standardized by shared config BC into `rope_parameters` |
| `attention_bias` | false | false | Q/K/V/O projections are biasless |
| `attention_dropout` | 0.0 | 0.1 | Disabled in inference |
| `torch_dtype` | unspecified | bf16 | Checkpoint metadata |
| `tie_word_embeddings` | false | true | Native class declares tied LM-head/embed alias key |

Checkpoint variation traps:

- Official dense A13B uses native `model_type: hunyuan_v1_moe`, but keeps `auto_map` to remote code.
- GPTQ and FP8 official variants use `model_type: hunyuan`, not `hunyuan_v1_moe`; they require separate source and quantized-weight admission.
- The native source ignores many remote config fields: `use_cla`, `use_mla`, LoRA-rank fields, vision fields, `mlp_bias`, `moe_intermediate_size`, grouped-routing fields, and classification-pool fields.
- Tokenizer `model_max_length` is 262144, larger than the model config context of 32768; DinoML should enforce model context separately.

## 3a. Family variation traps

- `hidden_size == num_attention_heads * head_dim` for A13B, but source reads explicit `head_dim`; do not infer from hidden size alone.
- `num_key_value_heads < num_attention_heads` is required for A13B GQA.
- RoPE is applied before Q/K RMSNorm. Moving Q/K norm before RoPE is not source-equivalent.
- Cached keys are stored after RoPE and Q/K RMSNorm; values are cached after V projection and reshape.
- MoE routing is token-dependent, top-k sparse, and uses fp32 gate logits/softmax.
- Expert weights are packed by expert in 3D tensors, not separate `Linear` modules.
- `tie_word_embeddings=true` in A13B config means embedding and LM-head aliasing should be preserved as one logical parameter when weights are tied.
- Native source supports attention backend dispatch through Transformers attention interfaces, but DinoML should first own an explicit dense/GQA path.
- Remote-code `hunyuan` configs with quantization, MLA, CLA, or multimodal fields should be rejected or routed to separate audits.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup `[B,S] -> [B,S,4096]`.
- Reshape/view/transposes for `[B,S,H] -> [B,heads,S,D]` and back.
- `contiguous` after attention transpose, slice for `logits_to_keep`.
- Gather/scatter/index ops for MoE token routing: `one_hot`, `nonzero`, `where`, row gather, `index_add_`.

Neural primitives:
- RMSNorm over last dim for hidden size and head dim, fp32 accumulation.
- Biasless Linear: Q `4096 -> 4096`, K/V `4096 -> 1024`, O `4096 -> 4096`.
- Shared SwiGLU: `4096 -> 3072` gate, `4096 -> 3072` up, elementwise SiLU/mul, `3072 -> 4096` down.
- Expert packed SwiGLU: per selected expert `4096 -> 2*3072`, chunk gate/up, SiLU/mul, `3072 -> 4096`.
- LM head `4096 -> 128167`, biasless, preferably last-token-only in decode.

Attention primitives:
- Causal self-attention with GQA repeat or native grouped attention.
- Additive causal/padding mask, fp32 softmax in eager path, dropout disabled for inference.
- KV cache update per layer.

Position/rotary:
- RoPE cos/sin generated in fp32 from `inv_freq @ position_ids`, duplicated over half dims, cast to activation dtype.
- DynamicNTKAlphaRotary special case when `rope_type == "dynamic"` and `alpha` is present.

Generation/cache:
- Dynamic KV cache, `position_ids = arange(seq) + past_seen_tokens`.
- `logits_to_keep` for prefill/decode output slicing.
- Beam cache reorder is provided by generic generation/cache layers, not custom native code here.

Quantized/packed weights:
- Native dense family has no GPTQ/FP8 dequant path.
- Expert packed weights are source-owned layout, not a quantization format.

## 5. Layer/block breakdown

Decoder block, repeated 32 times for A13B:

```text
residual = x
x = RMSNorm(x)                              # [B,S,4096]
q = Linear(x, 4096 -> 32*128) -> [B,32,S,128]
k = Linear(x, 4096 -> 8*128)  -> [B,8,S,128]
v = Linear(x, 4096 -> 8*128)  -> [B,8,S,128]
q,k = RoPE(q,k, cos, sin)
q = RMSNorm(q over D=128)
k = RMSNorm(k over D=128)
k,v = cache.update(k,v, layer_idx) if cache enabled
attn = causal GQA attention(q,k,v, mask)
x = residual + Linear(attn, 4096 -> 4096)

residual = x
x = RMSNorm(x)
shared = down(silu(gate(x)) * up(x))        # dense shared MLP
router = Linear(fp32 x, 4096 -> 64)
weights, experts = topk(softmax(router), 8)
routed = sum_experts(weights[e] * expert_e(x))
x = residual + shared + routed
```

Projection biases are false in the representative config and native default.

## 6. Attention requirements

Attention is causal self-attention only for the primary target. There is no cross-attention, encoder cache, local/sliding window, ALiBi, or block-sparse pattern in the native source.

Required shape contract:

```text
query: [B, Q, 32, 128] or [B,32,Q,128] internally
key/value current: [B, 8, Q, 128]
key/value cached: [B, 8, K_total, 128]
attention output: [B, Q, 32, 128] -> [B,Q,4096]
```

The eager implementation repeats KV heads to 32 attention heads before matmul. A production DinoML backend should use GQA directly and avoid materializing repeated KV when possible. Masking is causal plus optional user attention mask through `create_causal_mask`. Softmax is explicitly computed in fp32 in eager mode.

FlashAttention/SDPA compatibility: the source advertises flash, SDPA, flex attention, and generic attention backend support. DinoML parity should first validate eager math order, then add GQA FlashAttention with cached KV.

## 7. Position encoding and custom math

RoPE generation:

```python
if rope_type == "dynamic" and alpha is present:
    base = rope_theta * alpha ** (head_dim / (head_dim - 2))
    inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))
else:
    inv_freq = rope_init_fn(config)
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat(freqs, freqs, dim=-1)
cos, sin = cos(emb), sin(emb)
```

Application:

```python
def rotate_half(x):
    return cat((-x[..., D//2:], x[..., :D//2]), dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

Cos/sin depend on runtime `position_ids` and can be precomputed per position bucket or generated on GPU. For decode, only the current step positions are needed.

## 8. Preprocessing and input packing

Native model input ABI:

- `input_ids: [B,S]` or `inputs_embeds: [B,S,4096]`, exactly one required.
- Optional `attention_mask`, consumed by causal mask creation.
- Optional `position_ids`; otherwise source derives them from cache length.
- Tokenizer/chat template is CPU/data-pipeline work.

The official tokenizer config uses a chat template that injects special text markers. DinoML does not need to lower this into the GPU graph, but end-to-end parity needs tokenizer/template ownership. Although the official config includes image/video token IDs and vision fields, native `hunyuan_v1_moe` modeling does not consume image/video tensors or stitch multimodal embeddings.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GQA attention without KV repeat

Source pattern: repeat KV heads from `[B,8,K,128]` to `[B,32,K,128]`, then dense attention.

Replacement: grouped-query attention kernel with `num_query_heads=32`, `num_kv_heads=8`, group size 4.

Preconditions: `num_attention_heads % num_key_value_heads == 0`, causal self-attention, same `head_dim` for Q/K/V, no attention output requested. Failure cases: need dense attention weights output, unsupported attention backend flags, mismatched head dims.

Parity test sketch: compare eager repeat-KV attention against grouped kernel over prefill and one-step decode with padding masks.

### Rewrite: packed expert GEMM batching

Source pattern: per hit expert, gather token rows, run packed `gate_up_proj[e]`, chunk, activation multiply, `down_proj[e]`, weighted `index_add_`.

Replacement: segmented/grouped GEMM by selected expert, followed by weighted scatter-add.

Preconditions: static `num_experts`, static `top_k`, packed expert weights, deterministic top-k routing, no dropped tokens. Shape equations: tokens `T=B*S`, routed rows `T*top_k`, expert intermediate `I=3072`. Failure cases: quantized experts without provider, grouped-routing remote flags, capacity/drop-token flags.

Parity test sketch: fixed random small config with 4 experts/top2, compare selected experts, weights, routed output, and full block output.

### Rewrite: shared SwiGLU as fused epilogue chain

Source pattern: `down(silu(gate(x)) * up(x))`.

Replacement: two input GEMMs, fused SiLU/mul, final GEMM, optionally provider-fused where available.

Preconditions: biasless linears, same intermediate size for gate/up. Failure cases: `mlp_bias` remote variants or non-SiLU activation.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: during decode, bind only last hidden row to LM head.

Preconditions: `logits_to_keep == 1` or decode controller requests final token only. Failure cases: training loss, full-sequence logits, arbitrary tensor `logits_to_keep`.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm hidden and per-head Q/K RMSNorm, because every block uses three RMSNorm applications.
- GQA FlashAttention with post-RoPE/post-QK-norm inputs and KV cache.
- MoE routing plus grouped expert GEMMs; the eager loop/scatter path is the main throughput risk.
- Shared SwiGLU fusion for dense branch in every layer.
- Last-token-only LM head for decode.

Medium priority:
- Q/K/V projection packing into one or two GEMMs. K/V can be packed together; Q width differs only by output rows, not input.
- RoPE generation/application fused with Q/K reshape and Q/K RMSNorm where parity allows.
- Expert weight loading and layout transform into provider-friendly grouped-GEMM storage.

Lower priority:
- Sequence classification pooling/head.
- Full attention-weight materialization.
- Remote-code MLA/CLA and multimodal fields, since native source does not implement them.

## 11. Runtime staging plan

1. Parse native `HunYuanMoEV1Config`, with explicit rejection for `model_type=hunyuan`, GPTQ/FP8 quantization, MLA/CLA, and multimodal remote-code paths.
2. Load dense A13B weights and preserve tied embedding/LM-head aliasing where present.
3. One-block parity with embeddings, RMSNorm, RoPE, Q/K norm, eager GQA attention, shared MLP, and eager MoE.
4. Full prefill parity for short sequences with dense/eager MoE.
5. Decode parity with dynamic KV cache and `position_ids` offset by cache length.
6. Replace attention with optimized GQA FlashAttention.
7. Replace eager MoE with grouped/segmented expert GEMM and scatter-add.
8. Add logits slicing and decode scheduler probes.
9. Consider quantized/GGUF/GPTQ/FP8 variants only under separate provider contracts.

## 12. Parity and validation plan

- Unit tests for RMSNorm fp32 accumulation against PyTorch.
- RoPE tests for default, linear, and dynamic-alpha config forms, including exact `base = theta * alpha ** (D/(D-2))`.
- Attention parity: eager repeat-KV vs DinoML GQA for prefill and decode, with and without padding mask.
- MoE router parity: fp32 gate logits, softmax/topk, selected-weight renormalization.
- Expert parity: packed expert tensor layout, gather/scatter-add, top-k accumulation.
- Single decoder block parity in fp32, then bf16 tolerance.
- Full model short-prompt prefill logits parity against Transformers.
- Decode token parity for a fixed prompt and greedy/top-k=1 generation.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5` for block-level math; bf16 `rtol=3e-2, atol=3e-2` initially, tightened per kernel after accumulation choices are fixed.

## 13. Performance probes

- Prefill throughput sweep: batch, sequence length, and context near 2K/8K/32K.
- Decode tokens/sec sweep with KV cache sizes.
- GQA attention backend comparison: eager, SDPA-style, FlashAttention.
- MoE routing overhead: top-k and routing scatter time independent of expert GEMM.
- Expert load-balance histograms and grouped-GEMM occupancy by batch/sequence.
- Shared MLP vs routed expert time per layer.
- LM-head full logits vs last-token-only logits.
- KV cache memory by batch and context.
- Weight-load/provider comparison for dense bf16 and future quantized formats.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, and gradient checkpointing.
- Sequence classification head.
- Full returned attentions.
- Beam search cache reorder beyond generic cache index-select parity.
- Remote-code `hunyuan` variants, MLA, CLA, multimodal image/video fields.
- GPTQ, FP8, GGUF, and other quantized checkpoints.
- Tensor-parallel plans except preserving logical weight orientation and aliases.

## 15. Final implementation checklist

- [ ] Parse native `hunyuan_v1_moe` config and normalize legacy RoPE fields.
- [ ] Reject or route `model_type=hunyuan`, GPTQ, FP8, MLA, CLA, and multimodal remote-code configs.
- [ ] Load embeddings, LM head, Q/K/V/O, RMSNorm, shared MLP, gate, and packed expert weights.
- [ ] Preserve tied embedding/LM-head aliasing when config/checkpoint ties weights.
- [ ] Implement Hunyuan RMSNorm with fp32 accumulation.
- [ ] Implement DynamicNTKAlphaRotary and standard RoPE application.
- [ ] Implement causal GQA prefill and decode with KV cache.
- [ ] Implement per-head Q/K RMSNorm after RoPE.
- [ ] Implement shared SwiGLU MLP.
- [ ] Implement MoE router softmax/topk/renormalization.
- [ ] Implement packed expert eager reference path.
- [ ] Add grouped expert GEMM and scatter-add optimized path.
- [ ] Add one-block, full-prefill, and decode parity tests.
- [ ] Add performance probes for attention, MoE routing, grouped experts, and logits slicing.
