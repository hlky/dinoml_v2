# Qwen3-MoE Transformers Family Audit

Primary target: `Qwen3MoeForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen3_moe
Primary task: causal LM prefill/decode/generation
Local source root: X:/H/transformers
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py`
- `X:/H/transformers/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`
- `X:/H/transformers/src/transformers/models/qwen3_moe/modular_qwen3_moe.py`
- Cross-checks: `src/transformers/models/qwen3/modeling_qwen3.py`, `src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`, `src/transformers/models/qwen2_moe/configuration_qwen2_moe.py`, `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_moe/configuration_qwen3_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen3_moe/modular_qwen3_moe.py`

Representative config and generation metadata inspected from Hugging Face raw files:

- `https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-Qwen3MoeForCausalLM/raw/main/config.json`

Tokenizer metadata sampled:

- `https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/tokenizer_config.json`
- `https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/raw/main/tokenizer_config.json`
- `https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/raw/main/tokenizer_config.json`

Authoritative source note: `modeling_qwen3_moe.py` is generated from `modular_qwen3_moe.py`; future Transformers source edits should be checked in the modular file. This report uses the generated file for concrete expanded code and the modular file to identify intended inheritance/differences: Qwen3-MoE combines Qwen3 attention with Qwen2-MoE-style packed experts, but omits the Qwen2-MoE shared expert path.

Missing files or assumptions:

- No multimodal processor is involved for this text-only family. Tokenizer special tokens include vision/tool/FIM markers, but the audited `qwen3_moe` model does not consume image/video tensors.
- Official FP8 configs add `quantization_config`; source architecture is still `Qwen3MoeForCausalLM`. Treat FP8 weight loading as a separate quantization/runtime admission task.
- Current official configs sampled use `rope_scaling: null` and inactive sliding window. The source supports generic RoPE utilities and can enable sliding-window masks if `use_sliding_window=true` survives config post-init.

## 2. High-level architecture

Qwen3-MoE is a text-only decoder-only sparse MoE language model:

```text
tokenization/input_ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
                                      | each block: Qwen3 GQA attention + top-8 sparse MoE MLP
prefill: full prompt causal attention + KV cache fill
decode: new token(s) + cache update + MoE per token + last-token logits
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, special tokens, attention-mask construction, generation sampling controls.
- GPU/runtime prefill: embeddings, shared RoPE cos/sin generation, repeated decoder blocks, full causal GQA attention, top-k expert routing/execution, final norm, logits.
- GPU/runtime decode: one or few new tokens, position IDs from cache length, Q/K/V projections, Q/K RMSNorm, RoPE, KV cache append, attention over cache, MoE routing/expert execution, last-token logits.
- Generation controller: standard `GenerationMixin`; `generation_config.json` supplies sampling defaults such as temperature, top-p, top-k, EOS list, and pad token.

Implemented heads:

- Required for target: `Qwen3MoeForCausalLM`.
- Optional/deferred: base `Qwen3MoeModel` for hidden states.
- Deferred for first causal-LM target: sequence classification, token classification, and question answering generic heads.

## 3. Important config dimensions

Source defaults from `Qwen3MoeConfig`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 151936 |
| `hidden_size` | 2048 |
| `intermediate_size` | 6144 for dense fallback MLP layers |
| `moe_intermediate_size` | 768 per expert by default |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 4 |
| `head_dim` | Source attention uses explicit `head_dim` when present, otherwise `hidden_size // num_attention_heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 32768 default; official sampled configs use 40960 or 262144 |
| `rms_norm_eps` | 1e-6 |
| `use_cache` | true |
| `tie_word_embeddings` | false |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `decoder_sparse_step` | 1 |
| `mlp_only_layers` | `[]` after post-init if omitted |
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `norm_topk_prob` | false source default; true in official sampled configs |
| `output_router_logits` | false |
| `router_aux_loss_coef` | 0.001 |
| `sliding_window` | Set to configured value only if `use_sliding_window=true`; otherwise post-init sets it to `None` |

Representative checkpoint sweep. Dimensions are from `config.json`; active parameters in names such as A3B/A22B are repo metadata/name intent, not computed from source.

| Model id | Layers | H | Heads/KV | Head dim | KV groups | Dense I | Expert I | Experts/top-k | Max pos | RoPE theta | Sliding active | Dtype/quant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM` | 2 | 64 | 4/2 | 16 | 2 | 128 | 64 | 4/2 | 40960 | 1e6 | no | bf16 metadata |
| `Qwen/Qwen3-30B-A3B` | 48 | 2048 | 32/4 | 128 | 8 | 6144 | 768 | 128/8 | 40960 | 1e6 | no | bf16 |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 48 | 2048 | 32/4 | 128 | 8 | 6144 | 768 | 128/8 | 262144 | 1e7 | no | bf16 |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | 48 | 2048 | 32/4 | 128 | 8 | 6144 | 768 | 128/8 | 262144 | 1e7 | no | bf16 |
| `Qwen/Qwen3-235B-A22B` | 94 | 4096 | 64/4 | 128 | 16 | 12288 | 1536 | 128/8 | 40960 | 1e6 | no | bf16 |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | 94 | 4096 | 64/4 | 128 | 16 | 12288 | 1536 | 128/8 | 262144 | 5e6 | no | bf16 |
| `Qwen/Qwen3-30B-A3B-FP8` | 48 | 2048 | 32/4 | 128 | 8 | 6144 | 768 | 128/8 | 40960 | 1e6 | no | fp8 config, bf16 logical dtype |
| `Qwen/Qwen3-235B-A22B-FP8` | 94 | 4096 | 64/4 | 128 | 16 | 12288 | 1536 | 128/8 | 40960 | 1e6 | no | fp8 config, bf16 logical dtype |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| `Qwen3-30B-A3B` | `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-30B-A3B-Instruct-2507` | `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-30B-A3B-Thinking-2507` | `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `Qwen3-235B-A22B` | same sampling family as 30B base |
| `Qwen3-235B-A22B-Instruct-2507` | same sampling family as 30B instruct |

## 3a. Family variation traps

- This is MoE in every official sampled layer: `decoder_sparse_step=1`, `mlp_only_layers=[]`, and `num_experts > 0` make every decoder MLP a sparse MoE block. Source can express dense-only exception layers; lowering must inspect `mlp_only_layers` and `decoder_sparse_step`.
- Qwen3-MoE uses Qwen3 attention, not Qwen2/Qwen2-MoE attention: Q and K projections are RMS-normalized per head before RoPE. The `q_norm` and `k_norm` weights have length `head_dim`, not `hidden_size`.
- Official sampled configs use GQA with only 4 KV heads. 30B has 8 query heads per KV head; 235B has 16 query heads per KV head. Cache shape uses KV heads, never expanded query heads.
- All sampled official configs use top-8 routing over 128 experts. This is much heavier than Mixtral top-2 over 8 experts and changes router/top-k/scatter costs materially.
- `norm_topk_prob=true` in official sampled configs. The selected expert probabilities are renormalized across the top-8 experts before weighting expert outputs.
- There is no Qwen2-MoE shared expert in `Qwen3MoeSparseMoeBlock`. Source only has routed experts plus optional dense MLP layers selected by `mlp_only_layers`; do not port Qwen2-MoE's `shared_expert` and `shared_expert_gate` into Qwen3-MoE.
- Expert weights are packed as 3D tensors: `gate_up_proj[E, 2*moe_intermediate_size, hidden_size]` and `down_proj[E, hidden_size, moe_intermediate_size]`. This differs from three separate per-expert projections.
- `attention_bias=false` in sampled configs, but source gates all Q/K/V/O biases behind `config.attention_bias`. Reject or support biasful configs explicitly.
- `sliding_window` can exist in source defaults but is inactive when `use_sliding_window=false`; official sampled configs set `sliding_window: null`. Do not infer active local attention from default `4096`.
- `max_window_layers` is present in configs but not consumed by this generated qwen3_moe source path for per-layer type construction. Sliding behavior is global via effective `config.sliding_window`.
- `head_dim` is explicit in representative configs. Although sampled configs satisfy `hidden_size == num_attention_heads * head_dim`, lowering should use explicit `head_dim`.
- FP8 configs add quantization metadata and long `modules_to_not_convert` lists. Native source still describes the same graph; DinoML should route FP8 through a separate weight-admission path.
- Text layout is `[batch, sequence, hidden]`; attention transposes to `[batch, heads, sequence, head_dim]`. Layout translation is low value and risky around `dim=-1` RMSNorm, top-k over experts, softmax over keys, expert scatter-add, and cache layout.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- Shape/view ops for Q/K/V: `[B,S,out] -> [B,S,heads,D] -> [B,heads,S,D]`.
- `reshape`, `view`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `cat`, `chunk`, slicing/indexing for `logits_to_keep`.
- Residual adds and dtype casts.
- MoE route indexing: `one_hot`, `permute`, `sum`, `greater`, `nonzero`, `where`, token gather, `index_add_`.

Neural network primitives:

- RMSNorm over hidden axis `[H]`, fp32 variance math, weight multiply.
- Per-head RMSNorm over Q/K head axis `[D]`, fp32 variance math, weight length `head_dim`.
- Bias-optional linears:
  - 30B: Q `2048 -> 4096`, K/V `2048 -> 512`, O `4096 -> 2048`.
  - 235B: Q `4096 -> 8192`, K/V `4096 -> 512`, O `8192 -> 4096`.
  - Router 30B `2048 -> 128`; router 235B `4096 -> 128`.
  - Routed expert 30B gate/up packed `2048 -> 1536`, split to two `768` halves, down `768 -> 2048`.
  - Routed expert 235B gate/up packed `4096 -> 3072`, split to two `1536` halves, down `1536 -> 4096`.
  - Dense fallback MLP if present: 30B `2048 -> 6144 -> 2048`; 235B `4096 -> 12288 -> 4096`.
  - LM head: 30B `2048 -> 151936`; 235B `4096 -> 151936`.
- SiLU and elementwise multiply for SwiGLU.

Attention primitives:

- Causal self-attention with RoPE.
- GQA without physical KV repeat in optimized path.
- Eager fallback semantics: repeat KV, QK matmul, scale by `head_dim**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout in training only, matmul with V.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, flex attention, or custom integrations.

Position/rotary ops:

- Default RoPE inverse frequency from `rope_theta` and `head_dim`.
- Dynamic RoPE update support exists through shared utilities for non-default rope types; official sampled configs are default/no scaling.
- Apply RoPE to Q and K after per-head RMSNorm and before cache update.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache` is true and no cache is supplied.
- Cache `get_seq_length()` for default `position_ids`.
- KV cache update per layer after RoPE.
- Static/full cache support should follow shared Transformers cache semantics even though source imports only the base `Cache` type directly.
- `logits_to_keep`: int or tensor index selection before LM head.

MoE/router ops:

- Router linear, fp32 softmax over `E=128`, top-k with `K=8`, optional top-k renormalization.
- Per-expert token grouping, packed gate/up GEMM, chunk, SiLU multiply, down GEMM, route-weight multiply, and scatter-add back to flattened token order.
- Router logits capture path for diagnostics/aux loss.

Distributed/tensor/expert parallel metadata:

- Config declares TP and EP plans: attention Q/K/V colwise, Q/K norms replicated, O rowwise, packed expert gate/up colwise/grouped GEMM, expert down rowwise/grouped GEMM, router as `ep_router`, and experts as `moe_tp_experts`. First DinoML integration can be single GPU, but weight layout metadata should preserve these contracts.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x0: [B,S,H]
residual = x0
x = RMSNorm_H(x0)
q = Linear_bias_optional(H -> QH*D)(x).view(B,S,QH,D)
k = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D)
v = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D)
q = RMSNorm_D(q).transpose(1, 2)  # [B,QH,S,D]
k = RMSNorm_D(k).transpose(1, 2)  # [B,KVH,S,D]
v = v.transpose(1, 2)             # [B,KVH,S,D]
cos,sin = shared RoPE(position_ids)
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal_GQA_attention(q,k,v,mask,scale=D**-0.5,sliding_window?)
x = residual + Linear_bias_optional(QH*D -> H)(attn)

residual = x
x = RMSNorm_H(x)
if sparse layer:
    flat = x.view(B*S,H)
    router_logits = Linear(H -> E)(flat)
    probs = softmax(router_logits.float(), dim=-1)
    top_values, top_indices = topk(probs, K, dim=-1)
    if norm_topk_prob: top_values = top_values / sum(top_values, dim=-1, keepdim=True)
    for expert e with assigned tokens:
        gate, up = Linear(H -> 2*I_moe, expert=e)(tokens).chunk(2, dim=-1)
        y = Linear(I_moe -> H, expert=e)(silu(gate) * up)
        scatter_add(token_idx, y * selected_route_weight)
else:
    y = Linear(I_dense -> H)(silu(Linear(H -> I_dense)(x)) * Linear(H -> I_dense)(x))
x = residual + y.reshape(B,S,H)
```

Model tail:

```text
hidden = final RMSNorm_H(hidden)
slice_indices = slice(-logits_to_keep, None) if logits_to_keep is int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

Example production shapes:

- Qwen3-30B-A3B: `H=2048`, `QH=32`, `KVH=4`, `D=128`, `E=128`, `K=8`, `I_moe=768`, `I_dense=6144`. K/V projection width is `512`; O consumes `4096`.
- Qwen3-235B-A22B: `H=4096`, `QH=64`, `KVH=4`, `D=128`, `E=128`, `K=8`, `I_moe=1536`, `I_dense=12288`. K/V projection width is still `512`; O consumes `8192`.

## 6. Attention requirements

Variant: decoder causal self-attention with Qwen3 per-head Q/K RMSNorm, GQA, RoPE, optional global sliding-window masking, and KV cache.

Shapes:

- Hidden input: `[B,S,H]`.
- Query after projection/norm/transpose: `[B,QH,S,D]`.
- Key/value after projection/transpose: `[B,KVH,S,D]`.
- Full cache per layer stores RoPE-applied K and raw V: key `[B,KVH,T,D]`, value `[B,KVH,T,D]`.
- Eager fallback repeats K/V to `[B,QH,T,D]`; optimized DinoML attention should not materialize this repeat.
- Attention output before O projection: `[B,S,QH*D]`.

Representative cache sizes in bf16:

- 30B: per token per layer `2 * 4 * 128 * 2 bytes = 2048 bytes`; all 48 layers are about 96 KiB/token/batch element.
- 235B: per token per layer is still 2048 bytes because KV heads and head dim match 30B; all 94 layers are about 188 KiB/token/batch element.

Masking and cache:

- `Qwen3MoeModel` chooses `create_causal_mask` when effective `config.sliding_window is None`; otherwise `create_sliding_window_causal_mask`.
- Official sampled configs are full causal attention. A config with `use_sliding_window=true` would apply a global sliding-window mask/cache policy through `config.sliding_window`, not Qwen3 dense model's per-layer `layer_types`.
- Default `position_ids` are `arange(S) + past_seen_tokens`, unsqueezed to `[1,S]`.
- Q/K are RoPE-applied before cache update, so cached K is stored post-RoPE.

Math order in eager fallback:

```text
q_proj/k_proj/v_proj -> q_norm/k_norm -> transpose -> RoPE(q,k) -> cache.update
repeat_kv(k/v) -> matmul(q, k^T) * scale -> add mask
softmax(dtype=float32) -> cast to query dtype -> dropout(training only) -> matmul(weights, v)
transpose/reshape -> o_proj
```

Backend compatibility:

- Source advertises FlashAttention, SDPA, flex attention, and generic attention backend support.
- First optimized DinoML target should be native GQA causal attention with KV cache shape `[B,KVH,T,D]` and no physical repeat.
- Attention output weights are optional diagnostics and can be deferred for optimized path.

## 7. Position encoding and custom math

Default RoPE:

```python
dim = config.head_dim or config.hidden_size // config.num_attention_heads
base = config.rope_parameters["rope_theta"]
inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = emb.cos() * attention_scaling
sin = emb.sin() * attention_scaling
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_qwen3_moe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,S,D]
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunities:

- `inv_freq` is static for default RoPE and each config's `rope_theta`.
- Cos/sin depend on runtime `position_ids`, sequence length, and batch. Prefill computes `[B,S,D]` once and reuses it across all layers.
- Decode can compute only the new position row(s).
- Official sampled configs use default RoPE with `rope_scaling: null`; non-default rope types should be rejected or separately staged until parity is proven.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Tokenizer is Qwen-style text BPE metadata outside the `qwen3_moe` directory. GPU graph needs `input_ids`, optional `attention_mask`, optional `position_ids`, and optional cache.
- Sampled tokenizer configs define special tokens including `<|endoftext|>` id 151643, `<|im_start|>` id 151644, `<|im_end|>` id 151645, plus tool/FIM/vision marker tokens. These are token IDs only for this text model; no image/video tensors are consumed by `Qwen3MoeForCausalLM`.
- `input_ids` and `inputs_embeds` are mutually exclusive.
- `attention_mask` can be a 2D padding mask or an already prepared backend-compatible mask through shared masking utilities.
- `logits_to_keep` controls prompt-logit pruning. For generation, use `1` or explicit indices to avoid full-sequence vocab projection.

Generation-controller behavior outside compiled graph:

- Base and thinking configs use sampling defaults `temperature=0.6`, `top_p=0.95`, `top_k=20`.
- Instruct-2507 configs use `temperature=0.7`, `top_p=0.8`, `top_k=20`.
- EOS list is `[151645, 151643]` and pad token is `151643` in sampled generation configs.
- Chat templates, thinking-mode prompt conventions, tool-call formatting, and sampling processors can remain CPU/controller-side for first graph parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection with Q/K head RMSNorm

Source pattern:

```text
q = q_norm(q_proj(x).view(B,S,QH,D)).transpose(1,2)
k = k_norm(k_proj(x).view(B,S,KVH,D)).transpose(1,2)
v = v_proj(x).view(B,S,KVH,D).transpose(1,2)
```

Replacement pattern:

```text
grouped or packed projection -> split -> per-head RMSNorm on q/k -> transpose
```

Preconditions:

- Same normalized hidden input `x`.
- Bias handling matches `attention_bias`.
- Output widths are asymmetric under GQA: `[QH*D, KVH*D, KVH*D]`.
- Per-head RMSNorm weights are length `D`, applied before RoPE and after projection reshape.

Failure cases:

- Provider cannot expose split outputs before Q/K norm.
- Unsupported biasful attention configs.
- Tensor-parallel sharding plans require separate per-rank layout.

Parity test sketch:

- Compare Q/K/V tensors before RoPE for 30B and 235B shapes, including exact q_norm/k_norm behavior.

### Rewrite: native GQA attention with RoPE-before-cache

Source pattern:

```text
RoPE(q,k) -> cache.update(k,v) -> repeat_kv -> attention
```

Replacement pattern:

```text
FusedGQAAttention(q, k, v, cos, sin, cache, mask_policy)
```

Preconditions:

- Causal self-attention.
- KV cache stores post-RoPE K and raw V as `[B,KVH,T,D]`.
- Backend supports `QH/KVH` grouping of 8 or 16 for representative official configs.
- Sliding-window policy is either inactive or explicitly supported.

Failure cases:

- Non-default RoPE not admitted.
- Attention output weights requested from an optimized backend.
- Unsupported 4D/custom masks.

Parity test sketch:

- Prefill then multi-step decode with omitted `position_ids`; compare cache growth and logits.

### Rewrite: top-8 MoE routing to sorted expert batches

Source pattern:

```text
router_logits = linear(flat, router_weight)
probs = softmax(router_logits.float(), dim=-1)
values, indices = topk(probs, 8, dim=-1)
values = values / values.sum(dim=-1, keepdim=True)
for expert with hits:
    gather tokens -> packed gate/up -> silu*up -> down -> weighted index_add
```

Replacement pattern:

```text
RouterTopK -> expert assignment sort/histogram -> grouped expert GEMM -> weighted scatter-add
```

Preconditions:

- Inference mode, no router jitter.
- `norm_topk_prob` behavior exactly represented.
- Provider supports dynamic per-expert row counts and total routed rows `M*K`.
- Top-k tie behavior is acceptable or test inputs avoid ties.

Shape equations:

- `M = B*S`.
- Router logits `[M,128]`.
- Selected expert ids and weights `[M,8]`.
- Total routed rows before grouping: `8*M`.
- Expert gate/up: `[M_e,H] x [H,2*I_moe] -> [M_e,2*I_moe]`.
- Expert down: `[M_e,I_moe] x [I_moe,H] -> [M_e,H]`.

Weight transform:

- Source `gate_up_proj[e]` is an `F.linear` weight shaped `[2*I_moe,H]`.
- Source `down_proj[e]` is shaped `[H,I_moe]`.
- A DinoML `gemm_rcr`-style provider can consume these as row-major linear weights or prepack them per expert/provider manifest.

Failure cases:

- Requesting bitwise parity with nondeterministic scatter-add order.
- Empty experts must be handled without invalid launches.
- FP8 quantized expert weights need a separate quantized grouped-GEMM admission path.

Parity test sketch:

- Compare router logits, selected expert indices, normalized selected weights, per-expert outputs, and final scatter-add for small `E=4,K=2` and production `E=128,K=8`.

### Rewrite: dense fallback MLP when `mlp_only_layers` is non-empty

Source pattern:

```text
down(silu(gate_proj(x)) * up_proj(x))
```

Replacement pattern:

```text
fused gate/up GEMM -> SiLU*multiply -> down GEMM
```

Preconditions:

- Layer index is in `mlp_only_layers` or fails sparse-layer condition.
- Bias-free dense projections.
- Activation is exactly SiLU.

Failure cases:

- Assuming every layer is MoE without checking config.

### Rewrite: last-token-only LM head

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement pattern:

```text
GatherLastTokenOrIndices -> GEMM(H -> vocab)
```

Preconditions:

- Generation only needs last-token logits or explicit prompt-logit subset.
- Labels/loss path disabled.

Failure cases:

- Prompt log-prob evaluation requiring all positions.

## 10. Kernel fusion candidates

Highest priority:

- Native GQA attention with KV cache. GQA groups are large, especially 235B with 16 query heads per KV head; physical repeat is a production nonstarter.
- Q/K per-head RMSNorm + RoPE + cache write. This is the defining Qwen3 attention delta versus Qwen2/Mixtral and sits directly on decode hot path.
- MoE routing and grouped expert GEMMs. Top-8 over 128 experts dominates the Qwen3-MoE-specific runtime challenge.
- Packed expert gate/up GEMM + SiLU multiply. Source already packs gate/up; preserve that layout.
- Weighted scatter-add for top-8 contributions. Each token receives eight expert outputs, not Mixtral's two.
- Last-token-only LM head. Vocab GEMM is large and should not run for every prompt token during decode.

Medium priority:

- Router softmax/top-k specialized for `E=128,K=8`.
- QKV grouped projection with split outputs and Q/K norm hooks.
- RMSNorm hidden-axis fused kernel for pre/post attention and final norm.
- Expert residency/offload scheduling. 128 experts per layer makes weight movement more important than dense Qwen2 and different from Mixtral's 8 experts.
- Optional sliding-window mask/cache branch for non-official or future configs.

Lower priority:

- Aux load-balancing loss and router-logit diagnostic output.
- Generic classification/QA/token-classification heads.
- FP8-specific kernels until quantized weight admission is designed.
- Tensor/expert parallel execution. Preserve metadata early; implement after single-GPU parity.

## 11. Runtime staging plan

1. Parse `Qwen3MoeConfig`, including explicit `head_dim`, `num_key_value_heads`, sparse-layer selection, `norm_topk_prob`, effective `sliding_window`, and RoPE parameters.
2. Load weights for embeddings, attention projections, q/k head norms, hidden RMSNorms, packed experts, router weights, final norm, and LM head. Reject FP8 configs unless the quantization path is selected.
3. Implement one-block fp32/bf16 eager parity for a tiny config without cache: attention, Q/K norms, RoPE, router, top-k, and expert scatter.
4. Implement full prefill parity for tiny and shape-load smoke for 30B/235B configs.
5. Add dynamic KV cache decode with post-RoPE K storage and last-token logits.
6. Replace eager attention with native GQA attention for prefill and decode.
7. Replace Python-style MoE loop with sorted-token grouped expert GEMM and weighted scatter-add.
8. Add optional dense fallback MLP layer support driven by `mlp_only_layers`.
9. Add production scheduling: continuous batching, paged/cache allocation, expert residency/offload planning, and later tensor/expert parallel.

Initially stub/defer:

- Sampling/chat templates can run outside DinoML.
- Aux load-balancing loss can be computed only for diagnostics or deferred.
- FP8 and other quantized formats can be rejected by config admission until a provider-backed path exists.
- Sequence/token classification and QA heads can be deferred.

## 12. Parity and validation plan

Custom op tests:

- Hidden RMSNorm and head-dim RMSNorm with eps `1e-6`, fp32 variance, fp16/bf16 storage.
- Default RoPE with `rope_theta` values `1e6`, `5e6`, and `1e7`.
- Q/K norm + RoPE order regression: prove norm happens before RoPE and cache stores post-RoPE K.
- GQA attention for `QH/KVH = 32/4` and `64/4`.
- Router softmax/top-k/renormalization for `E=128,K=8` and tiny `E=4,K=2`.
- Packed expert gate/up/down with empty experts and multiple contributions per token.
- Scatter-add accumulation with top-8 route weights.

Model tests:

- Single decoder layer parity on `hf-internal-testing/tiny-random-Qwen3MoeForCausalLM`.
- Two-layer tiny prefill logits parity with `logits_to_keep=0`.
- Decode parity token by token with omitted `position_ids` and DynamicCache.
- `logits_to_keep=1` parity against full logits sliced to the final token.
- Config admission tests for inactive sliding window, explicit head_dim, dense fallback `mlp_only_layers`, and FP8 rejection/route.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 full block/logits: start with `rtol=2e-2`, `atol=2e-2`; tighten per op where math order is identical.
- Optimized attention and grouped MoE scatter may need separate tolerances due to softmax and accumulation order. Router selected indices should be exact on non-tie test data.

End-to-end:

- Raw token ID generation with deterministic sampling disabled or fixed greedy settings.
- Instruct/thinking prompt formatting and sampling can be controller-level tests after graph parity.

## 13. Performance probes

- Prefill tokens/sec by model scale and sequence length: 128, 2048, 8192, 40960, and 262144 where memory allows.
- Decode tokens/sec by batch size and cache length for 30B and 235B.
- KV cache memory and bandwidth, verifying storage is `[B,4,T,128]` per layer for both official scales.
- Attention backend comparison: eager repeat-KV, SDPA/FlashAttention-equivalent, DinoML native GQA.
- Router/top-k latency for `M` tokens with `E=128,K=8`.
- Expert token distribution and grouped GEMM utilization for prefill versus decode.
- Expert weight residency/offload probes: all-resident bf16 baseline, GGUF/dequant-before-GEMM exploratory path, FP8 config rejection or future admission path.
- Last-token LM head versus full prompt logits.
- Q/K norm + RoPE fusion bandwidth impact.

Benchmark observations: none collected. These are source/config-derived probes.

## 14. Skip/defer list

Safe to defer for first causal LM integration:

- Training, labels/loss, gradient checkpointing, dropout, and aux load-balancing loss.
- Returning attentions and router logits from optimized paths.
- Sequence classification, token classification, and question answering heads.
- Beam search and speculative decoding; standard logits are enough initially.
- FP8 quantized loading/kernels, except for explicit admission rejection or routing.
- Tensor parallel, expert parallel, pipeline parallel.
- Non-default RoPE variants and active sliding-window configs until targeted.
- Multimodal use of vision special tokens; this model family is text-only.

Do not defer:

- Q/K head RMSNorm before RoPE.
- GQA cache shape with KV heads only.
- RoPE-before-cache-update order.
- Top-8 routing with optional selected-probability renormalization.
- Absence of Qwen2-MoE shared expert.
- Packed expert weight layout.

## 15. Final implementation checklist

- [ ] Parse `Qwen3MoeConfig` and effective post-init fields.
- [ ] Validate/admit default RoPE first; preserve theta and explicit `head_dim`.
- [ ] Load embeddings, attention projections, Q/K head norms, RMSNorms, routers, packed experts, final norm, and LM head.
- [ ] Reject or separately route FP8 configs through quantization admission.
- [ ] Implement hidden-axis RMSNorm and head-dim Q/K RMSNorm.
- [ ] Implement bias-optional Q/K/V/O projections with asymmetric GQA outputs.
- [ ] Implement default RoPE and apply it after Q/K norm.
- [ ] Implement KV cache as `[B,KVH,T,D]` with post-RoPE keys.
- [ ] Implement native GQA causal attention without physical KV repeat.
- [ ] Implement router linear, fp32 softmax, top-8, and `norm_topk_prob`.
- [ ] Implement packed expert gate/up, SiLU multiply, down, route weighting, and scatter-add.
- [ ] Handle `mlp_only_layers` dense MLP fallback.
- [ ] Implement final RMSNorm, `logits_to_keep`, and LM head.
- [ ] Add one-block and tiny full-model parity tests.
- [ ] Add prefill/decode cache parity tests.
- [ ] Add config admission tests for 30B, 235B, instruct/thinking, tiny, and FP8 metadata.
- [ ] Benchmark attention, MoE router, expert grouped GEMMs, KV memory, and last-token logits.
