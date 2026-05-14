# Transformers Audit: longcat_flash

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: meituan-longcat/LongCat-Flash-Chat native path, plus config sweep variants
Config source: local LongcatFlashConfig plus HF raw config.json files
Source files inspected: configuration_longcat_flash.py, modeling_longcat_flash.py, modular_longcat_flash.py, tests/docs/support utilities listed in _sources/source_notes.md
Any missing files or assumptions: LongCat Flash Lite uses LongcatFlashNgram remote-code classes not present in this native source audit
```

`modeling_longcat_flash.py` is generated from `modular_longcat_flash.py`; the generated file is the runtime source basis, while `modular_longcat_flash.py` is authoritative for upstream edits. This report targets inference for `LongcatFlashForCausalLM`: causal text prefill, decode with KV cache, and logits. Training, loss, tensor parallel execution, and remote-code N-gram variants are not first-stage runtime targets.

## 2. High-Level Architecture

LongCat Flash is a text-only causal decoder with MLA attention and shortcut-connected MoE.

```text
token ids / inputs_embeds
  -> token embedding
  -> 28 logical decoder layers
       -> each logical layer runs 2 attention+dense-MLP sublayers
       -> first post-attention branch also computes shortcut MoE
       -> second sublayer residual adds shortcut MoE output
  -> final RMSNorm
  -> lm_head
  -> logits / generation controller
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, attention-mask construction inputs.
- GPU/runtime prefill: embedding, full causal mask, RoPE tables, 56 attention sublayers, 56 dense SwiGLU MLPs, 28 shortcut MoE applications, logits.
- GPU/runtime decode: one new token, position offset from cache length, causal mask or backend causal flag, update 56 KV cache layers.
- Independently optimizable pieces: RoPE/cos-sin generation, MLA Q/K/V projection region, attention backend, dense MLP, MoE router plus grouped expert GEMMs, last-token-only logits.

## 3. Important Config Dimensions

Native source defaults:

| Field | Default / effective value | Runtime significance |
| --- | ---: | --- |
| `vocab_size` | 131072 | embedding and LM head width |
| `hidden_size` | 6144 | residual width |
| `num_layers` | 28 | logical blocks |
| `num_hidden_layers` | 56 after model init | cache layer count, two attention sublayers per logical block |
| `num_attention_heads` | 64 | Q heads |
| `num_key_value_heads` | defaults to 64 | no GQA by default, but source computes `num_key_value_groups` |
| `q_lora_rank` | 1536 | query low-rank first projection width |
| `kv_lora_rank` | 512 | compressed KV pass width |
| `qk_nope_head_dim` | 128 | non-RoPE Q/K per-head width |
| `qk_rope_head_dim` | 64 | RoPE Q/K per-head width |
| `qk_head_dim` | 192 if omitted | attention Q/K per-head width |
| `v_head_dim` | 128 | value/output per-head width |
| `head_dim` | 64 | RoPE default frequency dim in source, not Q/K width |
| `ffn_hidden_size` | 12288 | dense MLP intermediate |
| `expert_ffn_hidden_size` | 2048 | per-expert MoE intermediate |
| `n_routed_experts` | 512 | computed experts |
| `zero_expert_num` | 256 | identity experts |
| `moe_topk` | 12 | experts selected per token |
| `routed_scaling_factor` | 6.0 | multiplies routed weights |
| `max_position_embeddings` | 131072 | Chat long context |
| `rope_theta` | 10000000.0 | default RoPE base for Chat |
| `attention_bias` | false | source still supports optional bias |
| `hidden_act` | `silu` | SwiGLU activation |
| `tie_word_embeddings` | false | LM head is a separate logical weight |
| `use_cache` | true | generation cache enabled |

Representative HF config sweep:

| Checkpoint | Native class? | Hidden | Logical layers | Attn sublayers | Heads | QK/V dims | Experts | Context | RoPE | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `meituan-longcat/LongCat-Flash-Chat` | yes | 6144 | 28 | 56 | 64 | 192 / 128 | 512 + 256 zero, top-12 | 131072 | default theta 1e7 | raw config omits `model_type` |
| `meituan-longcat/LongCat-Flash-Thinking` | yes | 6144 | 28 | 56 | 64 | 192 / 128 | same | 131072 | default theta 1e7 | same native operator shape |
| `meituan-longcat/LongCat-Flash-Chat-FP8` | yes, quantized loading separate | 6144 | 28 | 56 | 64 | 192 / 128 | same | 131072 | default theta 1e7 | FP8 config is loader/provider work |
| `meituan-longcat/LongCat-Flash-Omni` | text config yes | 6144 | 28 | 56 | 64 | 192 / 128 | same | 131072 | default theta 1e7 | multimodal wrapper not covered here |
| `meituan-longcat/LongCat-Flash-Lite` | no, remote N-gram class | 3072 | 14 | likely 28 | 32 | 192 / 128 | 256 + 128 zero, top-12 | 327680 | YaRN factor 10 | requires `LongcatFlashNgram*` audit |
| `tiny-random/longcat-flash` | yes debug | 8 | 2 | 4 | 4 | 256 / 64 | 32 + 16 zero, top-12 | 131072 | default | intentionally odd dims for testing |

## 3a. Family Variation Traps

- `hidden_size != num_attention_heads * qk_head_dim`: Chat is `6144 != 64 * 192`. Do not infer Q/K projection dimensions from hidden size.
- `head_dim` is not the attention Q/K head width. Source uses `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` for attention and `head_dim` for default RoPE frequency construction.
- `qk_head_dim != v_head_dim` for Chat. Flash attention path pads values from 128 to 192 and slices attention output back to 128.
- `num_hidden_layers` is mutated at model init to `2 * num_layers` so DynamicCache has one KV cache layer per attention sublayer.
- Lite/Lite-FP8 configs are not covered by native `LongcatFlashForCausalLM`; route to remote-code audit or reject for this source basis.
- FP8 config fields describe loading/provider policy, not native graph ops in `modeling_longcat_flash.py`.
- MoE includes identity zero experts. Expert ids `>= n_routed_experts` bypass expert GEMMs and add weighted hidden states.
- Router computes logits in fp32 with `F.linear(hidden_states.float(), weight.float())`, ignores `router_bias` in forward even if configured, and keeps classifier weight in fp32 modules.
- Dense MLP and expert MLP use SwiGLU, but expert weights are packed as `gate_up_proj[expert, 2 * intermediate, hidden]` then split gate/up.
- Native source supports optional `num_key_value_heads < num_attention_heads`; default configs do not use it.
- No sliding-window/local attention fields are read by this model source.
- Output attentions require eager attention; optimized attention interfaces do not return full weights.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,hidden]`.
- `view`, `reshape`, `transpose`, `contiguous`, `split`, `cat`, `expand`, `pad`, slice, gather, `index_add_`.
- Dynamic position id arange with cache offset.
- Last-token or indexed logits slicing via `logits_to_keep`.

Neural primitives:

- RMSNorm over last dim with fp32 variance, for hidden 6144, q rank 1536, kv rank 512, final norm.
- Dense Linear:
  - `q_a_proj`: `6144 -> 1536`, optional bias.
  - `q_b_proj`: `1536 -> 64 * 192 = 12288`, no bias.
  - `kv_a_proj_with_mqa`: `6144 -> 512 + 64 = 576`, optional bias.
  - `kv_b_proj`: `512 -> 64 * (128 + 128) = 16384`, no bias.
  - `o_proj`: `64 * 128 = 8192 -> 6144`, optional bias.
  - Dense MLP per sublayer: `6144 -> 12288`, `6144 -> 12288`, `12288 -> 6144`, no bias.
  - Router: `6144 -> 768` for Chat total experts, fp32.
  - LM head: `6144 -> 131072`, no bias.
- SiLU, elementwise multiply, residual add, weighted add.

Attention primitives:

- Causal self-attention, MLA projection structure, optional MHA/GQA repeat, KV cache update per attention sublayer.
- FlashAttention/SDPA/Flex/eager backend interface with Q/K dim 192 and V dim 128 padded to 192 for flash.

MoE primitives:

- Softmax over experts, top-k unsorted, gather selected weights, one-hot expert mask, per-expert token selection, grouped expert GEMM, identity expert branch, `index_add_` accumulation.
- First integration should avoid admitting arbitrary `where`/one-hot scatter as a general op; use a bounded MoE router/expert dispatch abstraction.

Position/rotary ops:

- Default RoPE and optional configured RoPE variants from Transformers rope utilities.
- LongCat-specific interleaved RoPE transform before `rotate_half`.
- YaRN scaling is needed only if admitting Lite or any native config with `rope_scaling`.

Generation/cache ops:

- DynamicCache construction from config, `get_seq_length`, `update(key,value,layer_idx)`, 56 cache layers for Chat.
- Beam/cache reorder should be inherited from Transformers cache semantics before production generation.

Quantized/packed weight metadata:

- FP8 checkpoints require loader/provider support for `quantization_config`; source graph has dense logical weights.
- Tensor-parallel plan includes MLA-specific `kv_a_proj_with_mqa` handling, but TP is optional/deferred for first single-GPU/runtime parity.

## 5. Layer/Block Breakdown

One logical decoder layer, repeated `num_layers` times:

```text
residual = x
x = RMSNorm_0(x)
x_attn = MLA_0(x, RoPE, causal_mask, cache[layer_idx*2])
x = residual + x_attn

residual = x
x = RMSNorm_post_0(x)
shortcut = MoE(x)
x = DenseSwiGLU_0(x)
x = residual + x

residual = x
x = RMSNorm_1(x)
x_attn = MLA_1(x, RoPE, causal_mask, cache[layer_idx*2+1])
x = residual + x_attn

residual = x
x = RMSNorm_post_1(x)
x = DenseSwiGLU_1(x)
x = residual + x + shortcut
```

MLA sublayer for Chat shapes:

```text
q = Linear(6144 -> 1536) -> RMSNorm(1536) -> Linear(1536 -> 12288)
q -> [B,64,S,192] -> split q_pass[128], q_rot[64]

compressed_kv = Linear(6144 -> 576)
split k_pass_compressed[512], k_rot[64]
k_pass = RMSNorm(512) -> Linear(512 -> 16384)
k_pass -> [B,64,S,256] -> split k_pass[128], v[128]
k_rot -> [B,1,S,64] -> RoPE -> expand to [B,64,S,64]

query = cat(q_pass, q_rot) -> [B,64,S,192]
key = cat(k_pass, k_rot) -> [B,64,S,192]
value = [B,64,S,128]
attention -> [B,S,64,128] -> reshape [B,S,8192] -> Linear(8192 -> 6144)
```

MoE block for Chat:

```text
router logits = Linear_fp32(6144 -> 768)
scores = softmax(logits)
topk indices = topk(scores + e_score_correction_bias, k=12, sorted=False)
weights = gather(scores, topk_indices) * 6.0
for each hit expert:
  if expert_id < 512:
    gate_up = Linear(hidden, gate_up_proj[expert]) -> split [2048,2048]
    y = silu(gate) * up
    y = Linear(y, down_proj[expert])
  else:
    y = identity(hidden)
  final.index_add_(token_idx, y * selected_weight)
```

## 6. Attention Requirements

- Type: causal autoregressive self-attention.
- Projection family: MLA with low-rank Q and compressed KV, not plain QKV.
- Heads: Chat Q heads 64, KV heads default 64. Source supports grouped repeat if `num_key_value_heads < num_attention_heads`.
- Q/K width: `qk_head_dim=192`, split into non-RoPE 128 and RoPE 64.
- V width: `v_head_dim=128`.
- Scaling: `qk_head_dim ** -0.5`; if non-default RoPE has `mscale_all_dim`, multiply by `yarn_get_mscale(factor, mscale_all_dim)^2`.
- Masking: `create_causal_mask` from Transformers, combining causal and optional 2D attention mask. No LongCat-specific sliding window in native source.
- Cache: key stored after RoPE/cat as `[B, num_key_value_heads, T, 192]`; value stored as `[B, num_key_value_heads, T, 128]`. Chat has 56 cache entries.
- Flash attention contract: if requested and Q/K width differs from V width, pad V last dim to QK width before backend call, then slice output back to V width. DinoML fused attention must either support unequal QK/V dimensions natively or apply the same pad/slice guard.
- Eager fallback order: repeat KV, matmul QK^T, multiply scaling, add mask, softmax in fp32, cast to query dtype, dropout, matmul with V, transpose to `[B,S,H,Dv]`.
- Packed/varlen: no LongCat-owned packed sequence ABI in model forward; mask utilities may derive packed masks from `position_ids` only in broader Transformers paths.

## 7. Position Encoding and Custom Math

Default Chat uses RoPE with `rope_theta=10000000.0`. Source standardizes legacy `rope_theta` into `rope_parameters`. The native default computes inverse frequencies using `head_dim`, which is 64 for Chat, matching `qk_rope_head_dim` but not `qk_head_dim`.

Core LongCat interleaved RoPE:

```python
def apply_longcat_interleaved_rope(q_rot, k_rot, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q_rot.view(B, H, S, D // 2, 2).transpose(4, 3).reshape(B, H, S, D)
    k = k_rot.view(B, H, S, D // 2, 2).transpose(4, 3).reshape(B, H, S, D)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

RoPE cos/sin are computed in fp32 under autocast-disabled context and cast back to hidden dtype. For decode, `position_ids` are offset by `past_key_values.get_seq_length()`. Cos/sin can be precomputed for static max context in default RoPE, but dynamic/non-default RoPE may update `inv_freq` based on sequence length.

YaRN is not required for native Chat, Thinking, or Omni text configs observed, but Lite configs include YaRN. If DinoML admits a native config with `rope_type="yarn"`, preserve both the rope utility's `attention_scaling` for cos/sin and LongCat MLA's separate attention scaling adjustment when `mscale_all_dim` is set.

## 8. Preprocessing and Input Packing

Primary runtime inputs:

- `input_ids`: `[B,S]` int token ids, or exactly one `inputs_embeds`: `[B,S,6144]`.
- `attention_mask`: optional 2D mask `[B, seen+S]` or already prepared backend mask.
- `position_ids`: optional `[B,S]`; if omitted, source creates `[0..S-1] + past_seen_tokens` and unsqueezes to `[1,S]`.

Tokenizer/chat template lives outside the neural graph. Source docs and tests use `AutoTokenizer.apply_chat_template` for Chat; DinoML can treat tokenization and generation-controller prompt formatting as CPU pipeline work.

No multimodal placeholder stitch is present in this native `longcat_flash` graph. `LongCat-Flash-Omni` uses this text config but any multimodal projector/packing belongs to a separate Omni audit.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: MLA projection region

Source pattern:

```text
q_a -> RMSNorm -> q_b -> reshape/transpose/split
kv_a -> split -> kv_norm/kv_b and direct k_rot path -> reshape/split
scales -> RoPE -> cat -> attention -> o_proj
```

Replacement:

```text
Explicit MLA op group with separate q_pass, q_rot, k_pass, k_rot, value tensors and cache update
```

Preconditions:

- `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, and head counts are explicit.
- Preserve scale factors `(hidden_size / q_lora_rank)^0.5` and `(hidden_size / kv_lora_rank)^0.5`.
- Preserve RoPE only on `q_rot`/`k_rot`.

Failure cases: `q_lora_rank=None` is partially scaffolded in init but forward assumes `q_a_proj`; reject until implemented.

Parity test sketch: compare one MLA sublayer eager output and cache tensors for fp32/bf16 at prefill and decode.

### Rewrite: unequal-value FlashAttention

Source pattern:

```text
if flash and qk_head_dim != v_head_dim:
    value = pad(value, [0, qk_head_dim - v_head_dim])
attention(...)
output = output[..., :v_head_dim]
```

Replacement: native attention ABI with separate QK dim and V dim, or guarded pad/slice wrapper.

Preconditions: `qk_head_dim >= v_head_dim`; for Chat 192 >= 128.

Failure cases: configs with `v_head_dim > qk_head_dim` cannot use this pad path.

### Rewrite: dense SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement: grouped GEMM or fused two-input activation-multiply epilogue before down GEMM.

Preconditions: no bias in all three dense MLP projections for native configs.

### Rewrite: MoE expert dispatch

Source pattern: one-hot expert mask, `where`, per-expert loop, `index_add_`.

Replacement: router top-k plus token sorting/grouped expert GEMM plus weighted scatter-add; identity experts handled by weighted residual add.

Preconditions:

- Top-k is unsorted; output is mathematically commutative only through addition, but tie/index order affects determinism.
- Expert ids `[0,n_routed_experts)` use packed gate/up and down weights; ids after that are identity.
- Router weights use softmax scores, while top-k choice uses scores plus correction bias.

Failure cases: if correction bias is mutable or nonzero, top-k choice and gathered weight source differ; preserve exactly.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, slice_indices, :])`.

Replacement: emit logits only for requested final tokens in decode/prefill.

Preconditions: `logits_to_keep` int or tensor is known at lowering/runtime and no loss computation is requested.

### Rewrite: FP8 weight materialization

Source pattern: HF `quantization_config` declares FP8 block quantized weights, but model graph uses logical dense linears.

Replacement: loader/provider contract that dequantizes or directly feeds FP8 GEMM providers.

Preconditions: exact quant metadata present, ignored layers respected, dense fallback available.

Failure cases: absent provider, unsupported ignored-layer naming, remote-code checkpoint layout drift.

## 10. Kernel Fusion Candidates

Highest priority:

- RMSNorm kernels for hidden 6144, q rank 1536, kv rank 512.
- MLA prefill/decode attention with explicit QK/V dims and KV cache update.
- RoPE interleave plus Q/K concatenation feeding attention.
- Dense SwiGLU MLP fusion for 56 sublayers.
- MoE router plus grouped expert GEMM and identity expert accumulation.
- Last-token-only LM head for decode.

Medium priority:

- Fused `q_a + RMSNorm + q_b` and `kv_a split + kv_norm + kv_b` regions.
- FlashAttention pad/slice elimination via separate V dimension.
- Top-k router specialization for 768 or 384 experts and k=12.
- Cache allocation/layout tuned for 56 sublayers and long context.
- FP8 dequant/GEMM provider path for FP8 checkpoints.

Lower priority:

- Tensor parallel sharding plans and MLA-specific `kv_a_proj_with_mqa` TP behavior.
- Full logits for prefill when only sampling next token is needed.
- Output attentions in eager mode.
- YaRN support unless Lite or future native configs are admitted.

## 11. Runtime Staging Plan

1. Parse native `LongcatFlashConfig`; reject `LongcatFlashNgram*`, `q_lora_rank=None`, unsupported FP8 provider policy, and non-default rope variants initially.
2. Load dense bf16/fp16 weights for Chat-shaped native checkpoints; preserve untied `embed_tokens` and `lm_head`.
3. Implement single-block eager parity with RMSNorm, MLA projections, default RoPE, eager attention, dense MLP, and MoE reference dispatch.
4. Scale to all layers prefill without cache optimization; validate final hidden/logits on short prompts.
5. Add DynamicCache-compatible decode with 56 KV entries and position offset.
6. Replace eager attention with optimized attention supporting QK=192 and V=128.
7. Add MoE grouped expert provider and identity expert fast path.
8. Add last-token logits and batching/scheduling.
9. Add optional FP8 loading/provider path.
10. Revisit Lite N-gram remote-code family as a separate target.

## 12. Parity and Validation Plan

- Config parser tests: Chat, Thinking, Chat-FP8 metadata, tiny-random, and Lite rejection.
- Unit tests for `apply_rotary_pos_emb_interleave` against Transformers for odd batch/sequence and decode offsets.
- MLA single-sublayer parity: compare query/key/value/cache tensors and output for fp32 and bf16.
- Eager attention parity: causal mask, padded tokens, prefill rectangular query/key lengths, decode length 1.
- Cache parity: after prefill and two decode steps, assert 56 key/value layer shapes and contents.
- MoE parity: router top-k with zero and nonzero correction bias, identity expert ids, duplicate token routing, and deterministic tie policy.
- Dense MLP and final norm parity.
- End-to-end short prompt logits parity using `hf-internal-testing/LongCat-ShortCat` if accessible; otherwise random tiny config.
- Recommended tolerances: fp32 `1e-4` absolute for isolated ops; bf16/fp16 `1e-2` to `3e-2` for full layer/logits depending on attention backend.

No DinoML tests or model imports were run for this audit.

## 13. Performance Probes

- Prefill tokens/sec versus sequence length: 1k, 8k, 32k, 128k.
- Decode tokens/sec and KV memory with 56 sublayers, batch sweep.
- Attention backend comparison: eager/SDPA/Flash-like with QK=192, V=128.
- Cache layout probe: store keys `[B,H,T,192]`, values `[B,H,T,128]`, update bandwidth at decode.
- Router/top-k latency for total experts 768 and top-12.
- MoE dispatch balance: active computed experts versus identity experts per batch/sequence.
- Grouped expert GEMM throughput for expert hidden 2048 and token-count histogram.
- Dense MLP GEMM throughput for `6144x12288` and `12288x6144`.
- Last-token logits versus full-sequence logits over vocab 131072.
- FP8 load/dequant/provider probe, separated from graph parity.

## 14. Skip/Defer List

- Training, labels/loss, gradient checkpointing.
- Tensor parallel and pipeline parallel execution.
- Output attentions outside eager debug mode.
- Beam search details beyond cache reorder conformance.
- Lite/Lite-FP8 N-gram architecture and YaRN long-context path.
- Omni multimodal wrapping and projector/prefix packing.
- FP8 direct kernels unless the provider path is explicitly selected.
- CPU offload/disk offload parity; upstream tests skip these because router uses direct `.type()` weight casts.
- Speculative decoding / MTP modules; source ignores unexpected `model.mtp.*` load keys.

## 15. Final Implementation Checklist

- [ ] Parse native LongCat Flash config and normalize legacy `rope_theta` / `rope_scaling`.
- [ ] Reject or route `LongcatFlashNgram*` checkpoints to a separate audit.
- [ ] Load dense weights with explicit dimensions and untied LM head.
- [ ] Implement RMSNorm fp32-accumulation kernels.
- [ ] Implement LongCat interleaved RoPE and position-id cache offset.
- [ ] Implement MLA projections with explicit QK/V dims and LoRA scaling factors.
- [ ] Implement causal attention with cache and unequal QK/V dimensions.
- [ ] Implement FlashAttention-compatible pad/slice fallback or native separate-V attention ABI.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement bounded MoE router, top-k, identity expert path, grouped expert GEMM, and scatter-add.
- [ ] Implement 56-layer DynamicCache-compatible KV manifest.
- [ ] Implement last-token-only logits.
- [ ] Add single-block, prefill, decode, cache, and MoE parity tests.
- [ ] Add performance probes for attention, MoE dispatch, dense MLP, logits, and KV memory.
- [ ] Add optional FP8 loading/provider admission after dense parity.
