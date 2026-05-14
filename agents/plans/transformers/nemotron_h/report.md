# Nemotron-H Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary native target is `nemotron_h` / `NemotronHForCausalLM`. Representative configs inspected:

- [nvidia/Nemotron-H-4B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K) -> `nvidia__Nemotron-H-4B-Base-8K.config.json`
- [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K) -> `nvidia__Nemotron-H-8B-Base-8K.config.json`
- [nvidia/Nemotron-H-56B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K) -> `nvidia__Nemotron-H-56B-Base-8K.config.json`
- [nvidia/Nemotron-H-47B-Reasoning-128K](https://huggingface.co/nvidia/Nemotron-H-47B-Reasoning-128K) -> `nvidia__Nemotron-H-47B-Reasoning-128K.config.json`
- [nvidia/Nemotron-H-47B-Reasoning-128K-FP8](https://huggingface.co/nvidia/Nemotron-H-47B-Reasoning-128K-FP8) -> `nvidia__Nemotron-H-47B-Reasoning-128K-FP8.config.json`
- [dmax123/tiny-nemotron-dummy-weights](https://huggingface.co/dmax123/tiny-nemotron-dummy-weights) at revision `081dbac3061bb16c0c458c1798b1d9d7bc135c95` -> `dmax123__tiny-nemotron-dummy-weights.config.json`
- [kurogane/Nemotron-H-micro-test03](https://huggingface.co/kurogane/Nemotron-H-micro-test03) -> `kurogane__Nemotron-H-micro-test03.config.json`; third-party mirror/debug config, useful only as a variation warning.

Config source: raw Hub `config.json` files saved beside this report. The NVIDIA links were accessible during audit; no gated config fetch was encountered. Some model cards/license terms may still restrict weights.

Source files inspected:

- `X:/H/transformers/src/transformers/models/nemotron_h/configuration_nemotron_h.py`
- `X:/H/transformers/src/transformers/models/nemotron_h/modular_nemotron_h.py`
- `X:/H/transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py`
- Delegated/generated source references: `models/zamba2/modeling_zamba2.py`, `models/jamba/modeling_jamba.py`, `models/deepseek_v3/modeling_deepseek_v3.py`, `models/nemotron/modeling_nemotron.py`, `cache_utils.py`, `activations.py`
- Tests/docs: `tests/models/nemotron_h/test_modeling_nemotron_h.py`, `docs/source/en/model_doc/nemotron_h.md`

Any missing files or assumptions: `modeling_nemotron_h.py` is generated from `modular_nemotron_h.py`; future source edits belong in the modular file, but DinoML should audit the generated file because it contains the exact native runtime body. Checkpoint configs contain `auto_map` entries for remote-code-era files; this report scopes native in-library source at the pinned commit, not arbitrary remote code. The native source does not implement MTP heads despite accepting MTP config fields and ignoring unexpected `mtp.*` weights on load.

## 2. High-level architecture

Nemotron-H is a text-only causal decoder with a hybrid per-layer mixer. Each block is:

```text
token ids -> embedding -> repeated hybrid blocks -> final RMSNorm -> LM head -> logits
```

Each hybrid block chooses one mixer from `layers_block_type`: Mamba2 SSM, causal self-attention, dense MLP, or MoE. The first useful DinoML target is causal LM prefill/decode for native `NemotronHForCausalLM`.

Stage decomposition:

```text
CPU tokenizer -> input_ids/attention_mask
GPU embedding + position_ids
prefill: Mamba/MLP/MoE/attention hybrid stack with attention KV + Mamba state cache
decode: one/new-token recurrent Mamba update + attention KV append + optional MoE routing
last-token logits -> generation controller/sampling
```

The Mamba and attention states can be validated independently from dense MLP/MoE blocks. There is no vision/audio processor, image placeholder stitch, cross-attention branch, or postprocessing path.

## 3. Important config dimensions

Native source defaults include `hidden_size=4096`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, `intermediate_size=21504`, `mamba_num_heads=128`, `mamba_head_dim=64`, `ssm_state_size=128`, `n_groups=8`, `conv_kernel=4`, `chunk_size=128`, `n_routed_experts=8`, `num_experts_per_tok=2`, and `mlp_hidden_act="relu2"`.

Representative checkpoint sweep, with layer counts derived from `layers_block_type` or `hybrid_override_pattern`:

| Config | Layers | Mixers | H | Attn heads/KV/head_dim | Mamba heads/head_dim/state | MLP/MoE | Context | Notes |
|---|---:|---|---:|---|---|---|---:|---|
| `dmax123/tiny` | 7 | 3 Mamba, 3 MoE, 1 Attn | 288 | 40/2/128 | 128/64/128 | dense 384, MoE 384, shared 5376, latent 576 | 8192 | Debug config has `hidden_size != num_heads * head_dim`; likely not production-shaped. |
| `nvidia 4B Base` | 52 | 24 Mamba, 24 MLP, 4 Attn | 3072 | 32/8/effective 96 from config key `attention_head_dim` | 112/64/128 | dense 12288 | 8192 | Native config class has `head_dim` default 128; admission should verify loaded effective `head_dim`. |
| `nvidia 8B Base` | 52 | 24 Mamba, 24 MLP, 4 Attn | 4096 | 32/8/128 | 128/64/128 | dense 21504 | 8192 | Common base target. |
| `nvidia 56B Base` | 118 | 54 Mamba, 54 MLP, 10 Attn | 8192 | 64/8/128 | 256/64/256 | dense 32768 | 8192 | Large dense-hybrid target. |
| `nvidia 47B Reasoning` | 98 | 45 Mamba, 48 MLP, 5 Attn | 8192 | 64/8/128 | 256/64/256 | dense 30720 | 131072 | Long-context config; no source RoPE use observed. |
| `nvidia 47B Reasoning FP8` | 98 | 45 Mamba, 48 MLP, 5 Attn | 8192 | 64/8/128 | 256/64/256 | dense 30720 | 131072 | Config advertises FP8 quantization/KV cache metadata; native modeling source does not read it. |
| `kurogane micro` | 12 | 6 Mamba, 5 MLP, 1 Attn | 1536 | 12/4/effective 128 | 80/80/source defaults may differ | dense 6144 | 131072 | Third-party config uses non-native names like `mamba_num_groups`, `mamba_state_dim`; route separately or reject until verified. |

For production NVIDIA configs, `num_hidden_layers` is legacy metadata; native `num_hidden_layers` is computed from the converted layer-type list.

## 3a. Family variation traps

- Layer topology is config-driven. `hybrid_override_pattern` maps `M -> mamba`, `E -> moe`, `* -> attention`, `- -> mlp`; native configs may omit `layers_block_type`.
- Attention is GQA when `num_key_value_heads < num_attention_heads`; KV heads are repeated only inside eager attention.
- `head_dim` must be admitted explicitly. Some configs use `attention_head_dim`; native source reads `config.head_dim`. A bad mapping changes `q/o` widths and can make `hidden_size != num_attention_heads * head_dim`.
- Native `NemotronHAttention` defines RoPE helpers but does not apply RoPE in `forward`; do not add rotary position math for this source basis without a separate remote-code audit.
- Mamba state is not KV cache: it has depthwise conv state plus recurrent SSM state, updated in-place through `DynamicCache`.
- Mamba fast path depends on optional external packages `causal-conv1d` and `mamba-ssm`. Native eager fallback is mathematically explicit but expensive.
- MoE experts are not gated SwiGLU experts: each routed expert is `down_proj(act(up_proj(x)))`, then weighted and `index_add_` accumulated.
- Optional `moe_latent_size` changes routed expert input/output width and adds `hidden -> latent -> hidden` projections.
- FP8 config metadata is a loading/provider contract, not native graph math. Source does not implement dequant, FP8 KV cache, or ModelOpt kernels in `forward`.
- MTP config fields and unexpected `mtp.*` weights are ignored by this native class for inference.
- Tensor layout is sequence-major `[B, L, C]` for model body; Mamba depthwise conv temporarily transposes to `[B, conv_dim, L]`. Protect Mamba conv/scan internals from naive NHWC-style layout rewriting.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,L] -> [B,L,H]`; arange/unsqueeze for position ids.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `split`, `cat`, `pad`, `expand`, `repeat_interleave`, `masked_fill`, `where`, `index_select`/beam reorder.
- Dynamic sequence padding to `chunk_size` for Mamba prefill.

Neural primitives:

- Linear: attention `q: H -> num_attention_heads*head_dim`, `k/v: H -> num_key_value_heads*head_dim`, `o: num_attention_heads*head_dim -> H`.
- Mamba `in_proj: H -> mamba_intermediate + conv_dim + mamba_num_heads`; source split actually has two unused `d_mlp` slots for compatibility when present.
- Mamba depthwise Conv1d: channels `conv_dim = mamba_num_heads*mamba_head_dim + 2*n_groups*ssm_state_size`, kernel `conv_kernel`, groups `conv_dim`, causal crop to sequence length.
- Mamba `out_proj: mamba_num_heads*mamba_head_dim -> H`.
- RMSNorm and gated RMSNorm; ReLU-squared (`relu2`), SiLU, softplus, clamp, exp, rsqrt.
- Dense MLP `H -> intermediate_size -> H`, no gate.
- MoE router and expert GEMMs, top-k selection, sigmoid router scores, one-hot/scatter masks, per-expert gather, `index_add_`.

Attention primitives:

- Causal self-attention, GQA/MQA repeat, mask add, softmax in fp32, dropout only in training, BMM/matmul for scores and values.
- Cache append/update for attention KV; no source RoPE application in native `forward`.

Position/custom math:

- Position ids are created and fed to mask creation; no native rotary call in attention. `max_position_embeddings` still affects mask/cache admission.

Generation/cache ops:

- `DynamicCache` with heterogeneous layers: attention layers store growing K/V; Mamba layers store fixed conv and recurrent states.
- Cache reorder for beam search must index both attention K/V and Mamba conv/recurrent state on batch dimension.
- `logits_to_keep` slicing before LM head; default generation path sets it from `config.num_logits_to_keep`.

Preprocessing-coupled ops:

- Text tokenization only. No model-coupled processor beyond `input_ids`, `attention_mask`, `position_ids`.

Quantized/packed weight metadata:

- FP8 `quantization_config` appears in one checkpoint and names ignored conv/lm_head modules, but the native source treats layers as normal PyTorch modules. DinoML should handle FP8 as a separate loader/provider admission decision.

## 5. Layer/block breakdown

Common block, repeated `len(layers_block_type)`:

```text
residual = x
x = RMSNorm(x.to(norm_weight_dtype))
x = mixer_by_layer_type(x, mask/cache)
x = residual + x
```

Attention block:

```text
q = Linear(H -> A*D, bias=False).view(B,L,A,D).transpose(1,2)
k = Linear(H -> K*D, bias=False).view(B,L,K,D).transpose(1,2)
v = Linear(H -> K*D, bias=False).view(B,L,K,D).transpose(1,2)
k,v = cache.update(k,v, layer_idx)  # if cache present
y = causal_attention(q,k,v, mask, scale=D^-0.5)
y = y.transpose(1,2).reshape(B,L,A*D)
out = Linear(A*D -> H, bias=False)(y)
```

Mamba block, prefill fallback:

```text
p = Linear(H -> projection_size)(x)
_, _, gate, hbc, dt = split(p, [d_mlp, d_mlp, mamba_intermediate, conv_dim, mamba_heads])
hbc = depthwise_conv1d(hbc.transpose(1,2), causal padding)[:, :, :L].transpose(1,2)
h, B, C = split(hbc, [mamba_intermediate, n_groups*state, n_groups*state])
dt = softplus(dt + dt_bias).clamp(min=time_step_min)
chunked SSD scan with A=-exp(A_log), B/C group repeat, D residual
y = Zamba2RMSNormGated(y, gate)
out = Linear(mamba_intermediate -> H)(y)
```

Mamba block, decode with previous state:

```text
p = in_proj(x_new)
conv_state = roll/update fixed [B, conv_dim, conv_kernel]
hbc = sum(conv_state * conv_weight) + bias; hbc = activation(hbc)
dt = softplus(dt + dt_bias)
recurrent_state = recurrent_state * exp(dt*A) + dt*B*x
y = recurrent_state @ C + D*x
y = gated_rms_norm(y, gate)
out = out_proj(y)
```

Dense MLP block:

```text
out = Linear(intermediate -> H, bias=mlp_bias)(relu2(Linear(H -> intermediate, bias=mlp_bias)(x)))
```

MoE block:

```text
router_logits = Linear(H -> n_routed_experts, fp32)
scores = sigmoid(router_logits)
group_scores = top2_per_group(scores + correction_bias).sum(-1)
selected_groups = topk(group_scores, topk_group)
selected_experts = topk(masked_scores, num_experts_per_tok)
weights = gather(scores, selected_experts); optional normalize; scale
x_flat = optional Linear(H -> latent)
for active expert:
    y_i = Linear(moe_intermediate -> input_dim)(act(Linear(input_dim -> moe_intermediate)(tokens_i)))
    final.index_add_(token_idx, y_i * route_weight)
out = optional Linear(latent -> H)(final).view(B,L,H) + shared_dense_mlp(residual)
```

## 6. Attention requirements

Attention is causal self-attention only, with GQA. For a production `8B` config: `A=32`, `K=8`, `D=128`, so Q/O width is 4096 and KV width is 1024. For `4B`, admission must ensure effective head dim is 96 if using `attention_head_dim`; otherwise native default `head_dim=128` would imply Q/O width 4096 against `hidden_size=3072`.

Masking uses `create_causal_mask` with `attention_mask`, `past_key_values`, and `position_ids`. Eager attention repeats KV heads before score matmul, adds the mask, softmaxes with fp32 accumulation, casts back to query dtype, then multiplies values. Native source advertises FlashAttention, SDPA, and flex attention support through `ALL_ATTENTION_FUNCTIONS`; optimized backends must preserve GQA, causal mask shape, scale order, and cache update semantics.

Cache shapes before repeat:

```text
key/value per attention layer: [B, num_key_value_heads, cached_seq, head_dim]
query per step: [B, num_attention_heads, query_len, head_dim]
after repeat for eager math: [B, num_attention_heads, cached_seq, head_dim]
```

Cached keys/values are stored without RoPE because native attention does not apply RoPE. There is no cross-attention cache.

## 7. Position encoding and custom math

Native `NemotronHAttention.forward` does not call `apply_rotary_pos_emb`, despite defining the helper. Position ids are used for mask/cache positioning only in this source basis.

Custom math DinoML must reproduce for native parity:

```python
def rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * y.to(x.dtype)

def relu2(x):
    return torch.relu(x).square()

def mamba_decode_state(prev, x, dt, A_log, B, C, D):
    A = -torch.exp(A_log.float())
    dt = torch.nn.functional.softplus(dt)
    dA = torch.exp(dt[..., None] * A[..., None, None])
    state = prev * dA + (dt[..., None] * B[..., None, :] * x[..., None])
    y = state.to(C.dtype).matmul(C[..., None]).squeeze(-1) + D[..., None] * x
    return y, state
```

Mamba prefill also uses chunked segment sums with lower-triangular masks and `exp(segment_sum(A))`; these are dynamic and sequence-length dependent. A/D/dt bias are weights; masks, chunk padding, and per-token B/C are runtime values.

## 8. Preprocessing and input packing

Runtime inputs are tokenized text:

```text
input_ids: [B,L] int tokens, or inputs_embeds: [B,L,H]
attention_mask: optional [B,L], used for causal mask and to zero Mamba padded tokens
position_ids: optional [B,L], otherwise arange + past length
```

CPU/data-pipeline work: tokenizer, chat template, padding side, sampling policy. GPU/runtime work: embedding lookup, position-id arange if omitted, causal mask construction, Mamba mask simplification. If `attention_mask` is all ones, Mamba receives `None`; if decoding with previous Mamba state, Mamba mask is also `None`.

No placeholder tokens, scatter stitches, multimodal grid metadata, packed `cu_seqlens`, image/audio features, or codebooks are part of this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: last-token-only logits

Source pattern: `hidden_states[:, slice_indices, :] -> lm_head -> float`.

Replacement: gather/slice the requested final positions before GEMM. Preconditions: inference without loss, `logits_to_keep` is positive integer or explicit static index tensor. Shape: `[B,L,H] -> [B,K,H] -> [B,K,V]`. Failure cases: training labels, arbitrary dynamic index tensor, callers requesting all logits.

### Rewrite: dense Linear -> CUTLASS GEMM/BMM

Source pattern: `nn.Linear` and `F.linear` on `[B,L,H]` or flattened `[tokens,H]`.

Replacement: flatten leading dims to M, run `gemm_rcr`/bias variant, reshape. Preconditions: dense row-major activation, static or bounded K/N, compatible dtype. Weight layout: PyTorch linear stores `[out_features, in_features]`; DinoML RCR can consume transposed logical RHS without physical transpose. Failure cases: FP8 packed weights without admitted dequant/provider path.

### Rewrite: GQA attention

Source pattern: project Q/K/V, optional cache update, repeat KV, matmul-softmax-matmul.

Replacement: GQA FlashAttention/SDPA that consumes unexpanded KV heads. Preconditions: native no-RoPE source basis, causal mask supported, dropout zero for inference, fp32 softmax parity tolerance documented. Failure cases: output attentions requested in non-eager backend, incompatible mask, unsupported `head_dim`.

### Rewrite: Mamba depthwise Conv1d decode

Source pattern: cache roll/update plus depthwise kernel dot for one token.

Replacement: fixed-size state update kernel over `[B, conv_dim, conv_kernel]`. Preconditions: `conv_kernel` small positive, `groups == conv_dim`, causal update, activation in admitted set. Failure cases: prefill shorter than conv kernel edge semantics, training fast path, non-SiLU activation if relying on external causal-conv kernel.

### Rewrite: MoE grouped expert GEMM

Source pattern: one-hot expert mask, Python loop over active experts, `F.linear`, weighted `index_add_`.

Replacement: top-k routing -> token bucketing -> grouped GEMM up/down -> weighted scatter-add. Preconditions: fixed expert count, static top-k, stable top-k tie policy accepted, no gated expert branch, latent projection handled. Failure cases: exact PyTorch `topk(sorted=False)` tie parity required, empty experts, dynamic token counts without allocation plan.

### Layout guard

The model body is `[B,L,C]`. Mamba conv is explicitly `[B,conv_dim,L]`; chunk scan has many axis-sensitive `permute`, `reshape`, `sum`, and `cumsum` operations. Use a conceptual no-layout-translation guard around Mamba scan and cache update unless every axis rewrite is proven.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm and gated RMSNorm, because every block uses pre-norm and every Mamba block uses gated norm.
- GQA attention prefill/decode with cache, because native eager repeat-KV is wasteful.
- Mamba decode update: in-proj split, causal depthwise state update, selective recurrent state update, gated norm, out-proj.
- Last-token-only LM head, because `num_logits_to_keep=1` is default.

Medium priority:

- Dense MLP `Linear + relu2 + Linear`.
- Mamba prefill chunk scan fused kernels or delegated `mamba-ssm` equivalent.
- MoE router top-k plus grouped expert GEMM and weighted scatter-add.
- Linear bias-free GEMM variants for attention and Mamba projections.

Lower priority:

- Causal mask construction fusion.
- Beam cache reorder.
- FP8 loading/dequant path for FP8 checkpoints, after dense BF16 parity is solid.

## 11. Runtime staging plan

Stage 1: parse native config, convert `hybrid_override_pattern`, reject or explicitly map legacy key names such as `attention_head_dim`.

Stage 2: load dense BF16/FP16 weights, embeddings, RMSNorm, dense MLP, and LM head; validate one all-MLP or isolated block if available.

Stage 3: implement attention blocks with GQA causal prefill and KV cache decode.

Stage 4: implement Mamba eager fallback parity for one layer, including fixed conv/recurrent cache ABI.

Stage 5: implement full hybrid stack prefill/decode for dense non-MoE checkpoints (`4B/8B/56B` base patterns are Mamba+MLP+attention).

Stage 6: add MoE routed expert path for configs containing `moe`, including latent projection if present.

Stage 7: optimize with FlashAttention, Mamba fused scan/update, last-token logits, and grouped GEMM.

Stage 8: evaluate FP8 checkpoints as a separate provider/loading feature. Stub initially by rejecting FP8 `quantization_config` unless a dense materialization path is available.

## 12. Parity and validation plan

- Config tests: `hybrid_override_pattern` conversion, layer counts, `num_hidden_layers` derived length, `head_dim`/`attention_head_dim` admission.
- Operator tests: RMSNorm fp32 accumulation, `relu2`, gated RMSNorm, Mamba segment-sum/chunk scan, depthwise causal conv update, router group top-k.
- Single-layer parity: one attention, one Mamba, one MLP, one MoE layer with random weights and fixed seeds.
- Cache parity: prefill then one-token decode equals full recompute for attention and Mamba layers; verify Mamba conv state `[B, conv_dim, conv_kernel]` and recurrent state `[B, mamba_num_heads, mamba_head_dim, ssm_state_size]`.
- Stack parity: after N layers for the real layer pattern, then full `NemotronHModel`.
- Causal LM parity: prefill logits and decode-token logits with `logits_to_keep=1`.
- Tolerances: fp32 custom ops around `1e-5` absolute/relative; bf16/fp16 model parity around `1e-2` to `3e-2` for full stack, with stricter per-op tolerances where fp32 accumulation is preserved.

No DinoML code/tests were run for this audit by request.

## 13. Performance probes

- Config/load probe: dense BF16 load time and peak resident weight memory.
- Prefill throughput sweep by sequence length: 1K, 8K, 32K, 128K where config admits it.
- Decode tokens/sec sweep by batch size and cache length, split by attention layers vs Mamba layers.
- Mamba prefill scan probe by `chunk_size`, `ssm_state_size`, `mamba_num_heads`, and sequence length.
- Mamba decode update probe for conv/recurrent state bandwidth and fixed-state memory.
- Attention backend comparison: eager repeat-KV, SDPA, FlashAttention GQA.
- MoE routing probe: top-k/router time, token bucketing, grouped expert GEMM occupancy, scatter-add time.
- LM head probe: all logits vs `logits_to_keep=1`.
- KV plus Mamba state memory probe; attention cache grows with sequence, Mamba state remains fixed-size per layer.
- FP8 provider probe only after admission: dequant/materialization time, dense fallback memory, FP8 KV cache support if required.

## 14. Skip/defer list

- Training, loss, dropout, gradient checkpointing.
- MTP/next-token prediction layers; native inference class ignores `mtp.*` weights.
- Remote-code-specific behavior not present in pinned native source.
- FP8 quantized runtime and FP8 KV cache until loader/provider policy is designed.
- Beam search optimization beyond correct cache reorder.
- Tensor parallel/distributed expert parallel.
- Mamba external fast-path exact kernel integration; start with native eager parity, then admit fused kernels.
- Output attentions for optimized attention backends.

## 15. Final implementation checklist

- [ ] Parse `NemotronHConfig` and convert `hybrid_override_pattern` to `layers_block_type`.
- [ ] Add admission guards for `head_dim`/`attention_head_dim`, `hidden_size`, layer types, and unsupported legacy config keys.
- [ ] Load embeddings, RMSNorm, dense linear weights, LM head, Mamba weights, and optional MoE expert tensors.
- [ ] Implement RMSNorm and Zamba2-style gated RMSNorm.
- [ ] Implement `relu2`, SiLU, softplus/clamp/exp custom math paths.
- [ ] Implement dense MLP block parity.
- [ ] Implement GQA causal attention prefill/decode and KV cache.
- [ ] Implement Mamba conv state and recurrent state ABI.
- [ ] Implement Mamba eager prefill scan and one-token decode update.
- [ ] Implement `DynamicCache`-style heterogeneous cache manifest and beam reorder.
- [ ] Implement `logits_to_keep` slice before LM head.
- [ ] Add MoE router, top-k/group routing, non-gated expert MLP, latent projection, and weighted scatter-add.
- [ ] Reject or separately admit FP8 `quantization_config`.
- [ ] Add single-layer, stack, prefill, and decode parity tests.
- [ ] Benchmark attention, Mamba scan/update, MoE grouped GEMM, and last-token logits.
