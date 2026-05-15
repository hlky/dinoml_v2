# Qwen2-MoE Transformers Family Audit

Primary target: `Qwen2MoeForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: qwen2_moe
Primary task: causal LM prefill/decode/generation
Local source root: transformers
```

Source files inspected:

- `transformers/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py`
- `transformers/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`
- `transformers/src/transformers/models/qwen2_moe/modular_qwen2_moe.py`
- Cross-checks: `src/transformers/models/mixtral/modeling_mixtral.py`, `src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`, `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2_moe/configuration_qwen2_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2_moe/modular_qwen2_moe.py`

Representative config and metadata inspected from Hugging Face raw/API files:

- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/raw/main/tokenizer_config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/raw/main/generation_config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/raw/main/tokenizer_config.json`
- `https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-Qwen2MoeForCausalLM/raw/main/config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-Qwen2MoeForCausalLM/raw/main/generation_config.json`

Authoritative source note: `modeling_qwen2_moe.py` is generated from `modular_qwen2_moe.py`; future Transformers source edits should be checked in the modular file. This report uses the generated file for expanded concrete code and the modular file for inheritance intent. The implementation composes Llama/Qwen2-style attention, Mixtral-style packed experts, and a Qwen2-MoE-specific shared expert.

Missing files or assumptions:

- No multimodal processor is involved. Tokenizer chat templates and special tokens are controller/data-pipeline concerns.
- Official base/chat checkpoints are ungated. The GPTQ Int4 config is official and has the same logical graph plus `quantization_config`; DinoML should route it through a separate quantized-weight admission path.
- Community repos named like Qwen2-MOE that were sampled use `model_type: qwen2` and `architectures: Qwen2ForCausalLM`, not native `qwen2_moe`; they are out of scope for this report.

## 2. High-level architecture

Qwen2-MoE is a text-only decoder-only sparse MoE language model:

```text
tokenization/input_ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
                                      | each block: causal self-attention + sparse routed experts + gated shared expert
prefill: full prompt causal attention + KV cache fill
decode: new token(s) + cache update + MoE/shared expert + last-token logits
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, special-token handling, padding/attention-mask construction, sampling controls.
- GPU/runtime prefill: embeddings, shared RoPE cos/sin generation, repeated decoder blocks, causal or per-layer sliding-window attention, router/top-k, routed expert GEMMs, shared-expert MLP, final norm, logits.
- GPU/runtime decode: derive `position_ids` from cache length when omitted, Q/K/V projections for new tokens, RoPE, cache append or sliding-window cache update, attention over cached KV, MoE/shared expert execution, last-token logits.
- Generation controller: standard `GenerationMixin`; generation configs only affect sampling/EOS/pad behavior and can remain outside the compiled graph initially.

Implemented heads:

- Required for target: `Qwen2MoeForCausalLM`.
- Optional/deferred: base `Qwen2MoeModel` hidden-state output.
- Deferred for first causal-LM target: sequence classification, token classification, and question answering generic heads.

## 3. Important config dimensions

Source defaults from `Qwen2MoeConfig`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 151936 |
| `hidden_size` | 2048 |
| `intermediate_size` | 5632 for dense fallback MLP and default shared expert |
| `moe_intermediate_size` | 1408 per routed expert |
| `shared_expert_intermediate_size` | 5632 |
| `num_hidden_layers` | 24 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 16 by default, so official Qwen1.5-MoE is MHA rather than GQA |
| `head_dim` | Source attention uses explicit `head_dim` if present, else `hidden_size // num_attention_heads`; official configs omit it, effective `128` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 32768 default; official base config uses 8192, chat uses 32768 |
| `rope_theta` / `rope_parameters` | Official configs use top-level `rope_theta: 1000000.0`; current config normalizes to `rope_parameters` |
| `rms_norm_eps` | 1e-6 |
| `use_cache` | true |
| `tie_word_embeddings` | false in official configs; source has tied-weight key metadata but does not tie unless config requests it |
| `qkv_bias` | true source default; official older configs omit it, so current source default makes Q/K/V bias active |
| `attention_dropout` | 0.0 |
| `decoder_sparse_step` | 1 |
| `mlp_only_layers` | `[]` after post-init if omitted |
| `num_experts` | 60 |
| `num_experts_per_tok` | 4 |
| `norm_topk_prob` | false in official configs |
| `output_router_logits` | false |
| `router_aux_loss_coef` | 0.001 |
| `use_sliding_window` | false in official configs |
| `sliding_window` | Post-init sets to configured value if `use_sliding_window=true`, else `0` in current source |
| `layer_types` | If omitted, generated per layer: odd 1-based layers below `max_window_layers` become `sliding_attention` only when `use_sliding_window=true`; otherwise all `full_attention` |

Representative checkpoint sweep:

| Model id | Layers | H | Heads/KV | Head dim | KV groups | Dense/shared I | Expert I | Experts/top-k | Max pos | Sliding active | QKV bias | Dtype/quant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM` | 2 | 64 | 4/2 | 16 | 2 | 128/64 | 64 | 4/2 | 32768 | no; `layer_types` full | true | bf16 metadata |
| `Qwen/Qwen1.5-MoE-A2.7B` | 24 | 2048 | 16/16 | 128 | 1 | 5632/5632 | 1408 | 60/4 | 8192 | no | omitted, source default true | bf16 |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 24 | 2048 | 16/16 | 128 | 1 | 5632/5632 | 1408 | 60/4 | 32768 | no | omitted, source default true | bf16 |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` | 24 | 2048 | 16/16 | 128 | 1 | 5632/5632 | 1408 | 60/4 | 32768 | no | omitted, source default true | GPTQ int4, float16 logical metadata |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| `Qwen1.5-MoE-A2.7B` | `bos_token_id=151643`, `pad_token_id=151643`, `eos_token_id=[151645,151643]` |
| `Qwen1.5-MoE-A2.7B-Chat` | `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`, `eos_token_id=[151645,151643]`, `pad_token_id=151643` |
| `tiny-random-Qwen2MoeForCausalLM` | `_from_model_config=true`, `bos_token_id=151643`, `eos_token_id=151645` |

## 3a. Family variation traps

- Qwen2-MoE is not simply Mixtral. It has a gated dense shared expert in every sparse layer: `shared_expert(x) * sigmoid(shared_expert_gate(x))`, added to the routed expert output.
- Qwen2-MoE is not Qwen3-MoE. It does not apply Q/K per-head RMSNorm before RoPE, and official Qwen1.5-MoE has 16 KV heads for 16 query heads, not 4 KV heads. Qwen3-MoE dropped the shared expert path and uses top-8 over 128 experts; Qwen2-MoE uses top-4 over 60 experts plus shared expert.
- Sparse/dense layer schedule is config-dependent. A layer is sparse only when `layer_idx not in mlp_only_layers`, `num_experts > 0`, and `(layer_idx + 1) % decoder_sparse_step == 0`.
- Official configs omit `qkv_bias`, but current source default is `qkv_bias=True`. DinoML config normalization should record the effective default and load Q/K/V bias tensors when present.
- Official base and chat configs set `use_sliding_window=false`; current post-init sets `sliding_window=0`, but `layer_types` still records all full-attention layers. If active sliding-window configs appear, only alternating odd 1-based layers below `max_window_layers` become sliding attention by default.
- `num_key_value_heads` can be smaller than query heads, as in the tiny config. Do not hard-code official MHA behavior.
- Expert weights are packed as 3D tensors in current source: `gate_up_proj[E, 2*moe_intermediate_size, hidden_size]` and `down_proj[E, hidden_size, moe_intermediate_size]`. The official GPTQ config lists older per-expert module names such as `mlp.experts.0.up_proj`; that is quantization metadata, not the current generated module's physical parameter layout.
- Router top-k probabilities are softmaxed in fp32. `norm_topk_prob=false` for sampled official configs, unlike Qwen3-MoE official configs where it is true.
- Layout translation is low value for this text model. Protect attention and MoE regions with no-layout-translation guards around sequence/head transposes, `dim=-1` softmax/top-k/RMSNorm, `chunk(2, dim=-1)`, and token scatter-add.
- `tie_word_embeddings=false` in official configs. If a future config sets it true, embedding and LM-head weight identity must be preserved.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- `reshape`, `view`, `transpose`, `contiguous`, `unsqueeze`, `expand`, `cat`, `chunk`, slicing/indexing for `logits_to_keep`.
- Residual adds and dtype casts.
- MoE route indexing: `one_hot`, `permute`, reduction over mask dimensions, `greater`, `nonzero`, `where`, token gather, `index_add_`.

Neural network primitives:

- RMSNorm over hidden axis `[H]`, fp32 variance math, weight multiply.
- Q/K/V linears with optional bias:
  - Official Qwen1.5-MoE: Q/K/V each `2048 -> 2048`, O `2048 -> 2048`.
  - Tiny: Q `64 -> 64`, K/V `64 -> 32`, O `64 -> 64`.
- Router linear:
  - Official: `2048 -> 60`.
  - Tiny: `64 -> 4`.
- Routed expert packed gate/up:
  - Official: `2048 -> 2816`, split into two `1408` halves.
  - Tiny: `64 -> 128`, split into two `64` halves.
- Routed expert down:
  - Official: `1408 -> 2048`.
  - Tiny: `64 -> 64`.
- Shared expert MLP and gate:
  - Official shared gate/up/down: `2048 -> 5632 -> 2048`; scalar gate `2048 -> 1`.
  - Tiny shared gate/up/down: `64 -> 64 -> 64`; scalar gate `64 -> 1`.
- Dense fallback MLP if selected by schedule:
  - Official: `2048 -> 5632 -> 2048`.
  - Tiny source default/config: `64 -> 128 -> 64`.
- SiLU, elementwise multiply, sigmoid scalar gate, route-weight multiply, expert/shared output add.
- LM head: official `2048 -> 151936`; tiny `64 -> 151936`.

Attention primitives:

- Causal self-attention with RoPE.
- MHA for official configs; GQA support required for tiny and future configs.
- Eager fallback semantics: repeat KV when `num_key_value_heads < num_attention_heads`, QK matmul, scale by `head_dim**-0.5`, add mask, fp32 softmax, cast to query dtype, dropout in training only, matmul with V.
- Backend dispatch through `ALL_ATTENTION_FUNCTIONS` for eager, SDPA, FlashAttention, flex attention, or integrations.
- Full causal and optional per-layer sliding-window causal masks.

Position/rotary ops:

- Default RoPE inverse frequency from `rope_theta` and `head_dim`.
- Dynamic RoPE update support exists through shared utilities for non-default RoPE types; sampled configs use default/no scaling.
- Apply RoPE to Q and K before cache update.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache` is true and no cache is supplied.
- Cache `get_seq_length()` for default `position_ids`.
- Per-layer cache classes follow `config.layer_types`: full attention uses unbounded dynamic layer, sliding attention uses sliding-window layer.
- KV cache update per layer after RoPE.
- `logits_to_keep`: int or tensor index selection before LM head.

MoE/router ops:

- Router linear, fp32 softmax over experts, top-k, optional selected-probability renormalization.
- Routed expert grouping, packed gate/up GEMM, chunk, SiLU multiply, down GEMM, route-weight multiply, and scatter-add back to flattened token order.
- Shared expert dense SwiGLU MLP plus scalar sigmoid gate, added to routed output.
- Router logits capture path for diagnostics/aux loss.

Distributed/tensor-parallel metadata:

- Config TP plan includes attention Q/K/V colwise, O rowwise, dense MLP gate/up colwise, down rowwise. The current config does not expose the richer Qwen3-MoE expert-parallel plan, but first DinoML integration should still keep expert weights grouped and provider-visible.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x0: [B,S,H]
residual = x0
x = RMSNorm_H(x0)
q = Linear_bias_optional(H -> QH*D)(x).view(B,S,QH,D).transpose(1,2)
k = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D).transpose(1,2)
v = Linear_bias_optional(H -> KVH*D)(x).view(B,S,KVH,D).transpose(1,2)
cos,sin = shared RoPE(position_ids)
q,k = apply_rope(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = causal_or_sliding_attention(q,k,v,mask,scale=D**-0.5)
x = residual + Linear_bias_false(QH*D -> H)(attn)

residual = x
x = RMSNorm_H(x)
if sparse layer:
    flat = x.view(B*S,H)
    shared = down_shared(silu(gate_shared(flat)) * up_shared(flat))
    shared = sigmoid(shared_expert_gate(flat)) * shared
    router_logits = Linear(H -> E)(flat)
    probs = softmax(router_logits.float(), dim=-1)
    top_values, top_indices = topk(probs, K, dim=-1)
    if norm_topk_prob: top_values = top_values / sum(top_values, dim=-1, keepdim=True)
    for expert e with assigned tokens:
        gate, up = Linear(H -> 2*I_moe, expert=e)(tokens).chunk(2, dim=-1)
        routed = Linear(I_moe -> H, expert=e)(silu(gate) * up)
        scatter_add(token_idx, routed * selected_route_weight)
    y = routed_output + shared
else:
    y = down_dense(silu(gate_dense(x)) * up_dense(x))
x = residual + y.reshape(B,S,H)
```

Model tail:

```text
hidden = final RMSNorm_H(hidden)
slice_indices = slice(-logits_to_keep, None) if logits_to_keep is int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

Official production shape summary:

- `H=2048`, `QH=16`, `KVH=16`, `D=128`, `E=60`, `K=4`, `I_moe=1408`, `I_shared=5632`, `I_dense=5632`.
- Q/K/V widths are all `2048`; O consumes `2048`.
- Every official sampled layer is sparse because `decoder_sparse_step=1`, `mlp_only_layers=[]`, and `num_experts=60`.

## 6. Attention requirements

Variant: decoder causal self-attention with RoPE, optional per-layer sliding-window masking/cache, MHA/GQA depending on config, and KV cache.

Shapes:

- Hidden input: `[B,S,H]`.
- Query after projection/transpose: `[B,QH,S,D]`.
- Key/value after projection/transpose: `[B,KVH,S,D]`.
- Full cache per layer stores RoPE-applied K and raw V: key `[B,KVH,T,D]`, value `[B,KVH,T,D]`.
- Eager fallback repeats K/V to `[B,QH,T,D]` when `KVH < QH`; optimized DinoML attention should avoid materializing this repeat.
- Attention output before O projection: `[B,S,QH*D]`.

Representative cache sizes in bf16:

- Official Qwen1.5-MoE: per token per layer `2 * 16 * 128 * 2 bytes = 8192 bytes`; all 24 layers are about 192 KiB/token/batch element.
- Tiny random: per token per layer `2 * 2 * 16 * 2 bytes = 128 bytes`; all 2 layers are 256 bytes/token/batch element.

Masking and cache:

- `Qwen2MoeModel` builds both mask entries when the provided `attention_mask` is not already a dict: `"full_attention": create_causal_mask(...)` and `"sliding_attention": create_sliding_window_causal_mask(...)`.
- Each layer selects `causal_mask_mapping[config.layer_types[i]]`.
- Official configs have inactive sliding windows. A future active config would alternate sliding/full attention according to `layer_types`, not use one global policy for all layers.
- Default `position_ids` are `arange(S) + past_seen_tokens`, unsqueezed to `[1,S]`.
- Q/K are RoPE-applied before cache update, so cached K is stored post-RoPE.

Math order in eager fallback:

```text
q_proj/k_proj/v_proj -> reshape/transpose -> RoPE(q,k) -> cache.update
repeat_kv(k/v) -> matmul(q, k^T) * scale -> add mask
softmax(dtype=float32) -> cast to query dtype -> dropout(training only) -> matmul(weights, v)
transpose/reshape -> o_proj
```

Backend compatibility:

- Source advertises FlashAttention, SDPA, flex attention, and generic attention backend support.
- First optimized DinoML target can treat official Qwen1.5-MoE as MHA, but the implementation should preserve a GQA-capable path because the tiny config and config class allow `num_key_value_heads < num_attention_heads`.
- Returning attention weights can be deferred for optimized attention.

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

def apply_qwen2_moe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B,1,S,D]
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunities:

- `inv_freq` is static for default RoPE and each config's `rope_theta`.
- Cos/sin depend on runtime `position_ids`, sequence length, batch, dtype/device, and dynamic RoPE variants.
- Prefill computes `[B,S,D]` once and reuses it across all layers.
- Decode can compute only the new position row(s).
- Official sampled configs use default theta-only RoPE; non-default RoPE types should be separately staged or explicitly rejected.

## 8. Preprocessing and input packing

Tokenizer/runtime contract:

- Tokenizer emits `input_ids` and optional `attention_mask`. GPU graph needs `input_ids` or `inputs_embeds`, optional `attention_mask`, optional `position_ids`, optional cache, and optional `logits_to_keep`.
- Special tokens sampled: `<|endoftext|>` id `151643`, `<|im_start|>` id `151644`, `<|im_end|>` id `151645`.
- Base tokenizer uses `<|endoftext|>` as EOS; chat tokenizer uses `<|im_end|>` as EOS and `<|endoftext|>` as pad.
- Chat template inserts role-delimited text with `<|im_start|>` and `<|im_end|>`. This is controller/data-pipeline work, not compiled GPU graph work.
- `input_ids` and `inputs_embeds` are mutually exclusive.
- `attention_mask` can be a 2D padding mask or an already prepared per-layer dict matching `layer_types`.
- `logits_to_keep` controls prompt-logit pruning. For generation, use `1` or explicit indices to avoid full-sequence vocab projection.

Generation-controller behavior outside compiled graph:

- Base generation config supplies BOS/pad/EOS only.
- Chat generation config adds sampling defaults: temperature `0.7`, top-p `0.8`, top-k `20`, repetition penalty `1.05`.
- Sampling processors, chat formatting, and EOS handling can stay CPU/controller-side for first graph parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection group

Source pattern:

```text
q = q_proj(x).view(B,S,QH,D).transpose(1,2)
k = k_proj(x).view(B,S,KVH,D).transpose(1,2)
v = v_proj(x).view(B,S,KVH,D).transpose(1,2)
```

Replacement pattern:

```text
GroupedGEMM or packed projection -> split [Q, K, V] -> reshape/transposes
```

Preconditions:

- Same normalized hidden input `x`.
- Bias handling matches effective `qkv_bias`.
- Output widths may be asymmetric when GQA is configured.
- Provider returns split outputs before RoPE/cache update.

Failure cases:

- Assuming official MHA sizes for all configs.
- Missing Q/K/V bias tensors in configs where effective source default requires them.
- Tensor-parallel sharding plans require separate per-rank layout.

Parity test sketch:

- Compare Q/K/V tensors before RoPE for official MHA shape and tiny GQA shape, with and without bias.

### Rewrite: native attention with RoPE-before-cache

Source pattern:

```text
RoPE(q,k) -> cache.update(k,v) -> repeat_kv -> attention
```

Replacement pattern:

```text
FusedCausalAttention(q, k, v, cos, sin, cache, layer_mask_policy)
```

Preconditions:

- Decoder self-attention.
- KV cache stores post-RoPE K and raw V as `[B,KVH,T,D]`.
- Layer mask policy is full causal or supported sliding-window causal.
- Backend supports MHA and GQA without physical KV repeat.

Failure cases:

- Active alternating sliding-window layers not represented in cache/mask manifest.
- Non-default RoPE not admitted.
- Attention output weights requested from optimized backend.

Parity test sketch:

- Prefill then multi-step decode with omitted `position_ids`; compare cache growth, attention output, and logits.

### Rewrite: top-4 routed experts plus shared expert

Source pattern:

```text
shared = shared_down(silu(shared_gate(flat)) * shared_up(flat))
shared = sigmoid(shared_expert_gate(flat)) * shared
router_logits = linear(flat, router_weight)
probs = softmax(router_logits.float(), dim=-1)
values, indices = topk(probs, 4, dim=-1)
for expert with hits:
    gather tokens -> packed gate/up -> silu*up -> down -> weighted index_add
output = routed + shared
```

Replacement pattern:

```text
RouterTopK -> expert assignment sort/histogram -> grouped expert GEMM -> weighted scatter-add
SharedSwiGLU + scalar sigmoid gate -> add routed/shared outputs
```

Preconditions:

- Inference mode, no aux loss needed on hot path.
- `norm_topk_prob` behavior exactly represented, including the official false case.
- Provider supports dynamic per-expert row counts and total routed rows `M*K`.
- Shared expert is executed for every token regardless of routing.

Shape equations:

- `M = B*S`.
- Router logits `[M,60]`; selected expert ids and weights `[M,4]`.
- Total routed rows before grouping: `4*M`.
- Expert gate/up: `[M_e,H] x [H,2*I_moe] -> [M_e,2*I_moe]`.
- Expert down: `[M_e,I_moe] x [I_moe,H] -> [M_e,H]`.
- Shared expert: `[M,H] x [H,2*I_shared] -> [M,2*I_shared]`, then `[M,I_shared] x [I_shared,H] -> [M,H]`.
- Shared scalar gate: `[M,H] x [H,1] -> [M,1]`.

Weight transform:

- Source `gate_up_proj[e]` is an `F.linear` weight shaped `[2*I_moe,H]`.
- Source `down_proj[e]` is shaped `[H,I_moe]`.
- Shared expert uses normal separate dense MLP weights: `gate_proj[I_shared,H]`, `up_proj[I_shared,H]`, `down_proj[H,I_shared]`, plus `shared_expert_gate[1,H]`.

Failure cases:

- Dropping the shared expert, which would make Qwen2-MoE behave like Qwen3-MoE/Mixtral and break parity.
- Requesting bitwise parity when scatter-add order changes.
- Empty experts must be handled without invalid launches.
- GPTQ expert metadata uses older per-expert names and needs separate quantized loader mapping.

Parity test sketch:

- Compare router logits, selected expert indices, selected weights, routed expert output, shared expert output, scalar shared gate, and final sum for tiny and official shapes.

### Rewrite: sparse/dense layer schedule admission

Source pattern:

```text
if layer_idx not in mlp_only_layers and num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0:
    use sparse MoE block
else:
    use dense SwiGLU MLP
```

Replacement pattern:

```text
ConfigLowering emits layer_type = sparse_moe or dense_mlp per decoder layer
```

Preconditions:

- `mlp_only_layers` normalized before lowering.
- `decoder_sparse_step` is positive and layer indices are zero-based.

Failure cases:

- Assuming every layer is MoE because official sampled configs are.
- Forgetting dense fallback weights when a future config includes `mlp_only_layers`.

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

- MoE routed expert scheduler plus shared expert. Qwen2-MoE-specific runtime cost is routed top-4 over 60 experts plus a dense shared SwiGLU per token.
- Packed expert gate/up GEMM + SiLU multiply. Source already packs routed expert gate/up weights; preserve that layout.
- Weighted scatter-add for top-4 contributions. Necessary once expert batches are grouped.
- Shared expert gate fusion: shared SwiGLU output multiplied by `sigmoid(linear(x))`; this is absent from Qwen3-MoE and Mixtral.
- Attention with KV cache. Official configs are MHA, but the same implementation must support GQA for tiny/future configs and avoid physical KV repeat.
- Last-token-only LM head. Vocab projection is large and should not run for every prompt token during decode.

Medium priority:

- Router softmax/top-k specialized for official `E=60,K=4`.
- QKV grouped projection with optional bias.
- RMSNorm hidden-axis fused kernel for pre/post attention and final norm.
- RoPE + cache write fusion.
- Alternating sliding-window mask/cache support for future active configs.
- Expert residency/offload scheduling. 60 routed experts per layer plus shared dense expert makes weight movement a first-order production issue.

Lower priority:

- Aux load-balancing loss and router-logit diagnostic output.
- Generic sequence/token classification and QA heads.
- GPTQ/quantized kernels until quantized weight admission is designed.
- Multi-GPU tensor/expert parallel execution.

## 11. Runtime staging plan

1. Parse `Qwen2MoeConfig`, including effective defaults for `qkv_bias`, `rope_parameters`, `layer_types`, sparse layer schedule, `norm_topk_prob`, and sliding-window activation.
2. Load weights for embeddings, attention projections/biases, RMSNorms, router weights, packed routed experts, shared expert weights/gate, final norm, and LM head. Reject GPTQ configs unless the quantized path is selected.
3. Implement one-block fp32/bf16 eager parity for the tiny config without cache: attention, RoPE, router, routed experts, shared expert, and scatter.
4. Implement full prefill parity for tiny and shape/load smoke for official base/chat configs.
5. Add dynamic KV cache decode with post-RoPE K storage and last-token logits.
6. Replace eager attention with native MHA/GQA attention for prefill and decode.
7. Replace Python-style MoE loop with sorted-token grouped expert GEMM and weighted scatter-add while preserving shared expert execution.
8. Add dense fallback MLP layer support driven by `mlp_only_layers` and `decoder_sparse_step`.
9. Add optional active sliding-window layer support or explicit config rejection.
10. Add production scheduling: continuous batching, paged cache allocation, expert residency/offload planning, and later quantized expert support.

Initially stub/defer:

- Sampling/chat templates can run outside DinoML.
- Aux load-balancing loss can be computed only for diagnostics or deferred.
- GPTQ and other quantized formats can be rejected by config admission until a provider-backed path exists.
- Sequence classification, token classification, and QA heads can be deferred.

## 12. Parity and validation plan

Custom op tests:

- RMSNorm with eps `1e-6`, fp32 variance, fp16/bf16 storage.
- Default RoPE with `rope_theta=1e6` and decode offsets.
- MHA and GQA attention: official `QH/KVH=16/16` and tiny `4/2`.
- Q/K/V projection parity with effective QKV bias.
- Router softmax/top-k for official `E=60,K=4` and tiny `E=4,K=2`; include `norm_topk_prob=false` and true synthetic cases.
- Packed expert gate/up/down with empty experts and multiple route contributions per token.
- Shared expert SwiGLU plus scalar sigmoid gate.
- Routed/shared output add and scatter-add accumulation.

Model tests:

- Single decoder layer parity on `hf-internal-testing/tiny-random-Qwen2MoeForCausalLM`.
- Two-layer tiny prefill logits parity with full logits and `logits_to_keep=1`.
- Decode parity token by token with omitted `position_ids` and `DynamicCache`.
- Config admission tests for official base, chat, GPTQ Int4, tiny GQA, inactive sliding window, and a synthetic active sliding-window layer schedule.
- Shape-only/load smoke for official base/chat weights if full checkpoints are available locally.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 full block/logits: start with `rtol=2e-2`, `atol=2e-2`; tighten per op where math order is identical.
- Optimized attention and grouped MoE scatter may need separate tolerances due to softmax and accumulation order. Router selected indices should be exact on non-tie test data.

End-to-end:

- Raw token ID generation with deterministic sampling disabled or fixed greedy settings.
- Chat prompt formatting and sampling can be controller-level tests after graph parity.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 128, 512, 2048, 8192 for base; include 32768 for chat if memory allows.
- Decode tokens/sec by batch size and cache length.
- KV cache memory and bandwidth, verifying official cache uses `[B,16,T,128]` per layer while tiny/future GQA uses KV heads only.
- Attention backend comparison: eager repeat-KV, SDPA/FlashAttention-equivalent, DinoML native MHA/GQA.
- Router/top-k latency for `M` tokens with `E=60,K=4`.
- Expert token distribution and grouped GEMM utilization for prefill versus decode.
- Shared expert cost versus routed expert cost; Qwen2-MoE has dense shared expert work every sparse layer.
- Expert weight residency/offload probes: all-resident bf16 baseline, GGUF/dequant-before-GEMM exploratory path, GPTQ rejection or future admission path.
- Last-token LM head versus full prompt logits.

Benchmark observations: none collected. These are source/config-derived probes.

## 14. Skip/defer list

Safe to defer for first causal LM integration:

- Training, labels/loss, gradient checkpointing, dropout, and aux load-balancing loss.
- Returning attentions and router logits from optimized paths.
- Sequence classification, token classification, and question answering heads.
- Beam search and speculative decoding; standard logits are enough initially.
- GPTQ/quantized loading and kernels, except for explicit admission rejection or routing.
- Tensor parallel, expert parallel, and pipeline parallel.
- Non-default RoPE variants until targeted.
- Active sliding-window optimized kernel for official base/chat parity, because sampled official configs disable it.

Do not defer:

- Effective Q/K/V bias handling.
- RoPE-before-cache-update order.
- KV cache shape with KV heads only for GQA-capable configs.
- Sparse/dense layer schedule from `decoder_sparse_step` and `mlp_only_layers`.
- Top-4 routing over 60 experts for official configs.
- The gated shared expert path.
- Packed routed expert weight layout.

## 15. Final implementation checklist

- [ ] Parse `Qwen2MoeConfig` and effective post-init fields.
- [ ] Normalize top-level `rope_theta` into default RoPE parameters.
- [ ] Validate/admit default RoPE first; preserve explicit or inferred `head_dim`.
- [ ] Parse sparse/dense layer schedule from `decoder_sparse_step`, `mlp_only_layers`, and `num_experts`.
- [ ] Parse `layer_types` and active/inactive sliding-window policy.
- [ ] Load embeddings, attention projections and Q/K/V biases, RMSNorms, router weights, packed routed experts, shared expert weights/gate, final norm, and LM head.
- [ ] Reject or separately route GPTQ configs through quantization admission.
- [ ] Implement hidden-axis RMSNorm.
- [ ] Implement bias-optional Q/K/V projections and bias-free O projection.
- [ ] Implement default RoPE and apply it before cache update.
- [ ] Implement KV cache as `[B,KVH,T,D]` with post-RoPE keys.
- [ ] Implement native causal MHA/GQA attention without physical KV repeat.
- [ ] Implement optional sliding-window mask/cache branch or explicit unsupported-config rejection.
- [ ] Implement router linear, fp32 softmax, top-k, and optional `norm_topk_prob`.
- [ ] Implement packed routed expert gate/up, SiLU multiply, down, route weighting, and scatter-add.
- [ ] Implement shared expert SwiGLU plus scalar sigmoid gate and add to routed output.
- [ ] Implement dense fallback MLP for non-sparse scheduled layers.
- [ ] Implement final RMSNorm, `logits_to_keep`, and LM head.
- [ ] Add one-block and tiny full-model parity tests.
- [ ] Add prefill/decode cache parity tests.
- [ ] Add config admission tests for base, chat, GPTQ, tiny, and synthetic active sliding-window configs.
- [ ] Benchmark attention, router/top-k, routed expert grouped GEMMs, shared expert, KV memory, and last-token logits.
