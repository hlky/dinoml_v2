# Transformers Audit: zamba

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Zyphra/Zamba-7B-v1; Zyphra/Zamba-7B-v1-phase1
Config source: Hugging Face config.json for both model ids, plus local configuration_zamba.py defaults
Source files inspected:
  transformers/src/transformers/models/zamba/configuration_zamba.py
  transformers/src/transformers/models/zamba/modeling_zamba.py
  transformers/tests/models/zamba/test_modeling_zamba.py
  transformers/src/transformers/cache_utils.py
  transformers/src/transformers/masking_utils.py
  transformers/docs/source/en/model_doc/zamba.md
Any missing files or assumptions:
  No modular_zamba.py exists for this family at the inspected commit.
  Only official zamba checkpoints found were the 7B v1 and phase1 configs; Zamba2 is a separate model_type and out of scope.
  No DinoML tests were run.
```

Primary report target: `ZambaForCausalLM` text generation. `ZambaModel` feature extraction is required as the body. `ZambaForSequenceClassification` is optional/deferred for first integration.

Primary online config links inspected:

- [Zyphra/Zamba-7B-v1 config.json](https://huggingface.co/Zyphra/Zamba-7B-v1/blob/main/config.json)
- [Zyphra/Zamba-7B-v1-phase1 config.json](https://huggingface.co/Zyphra/Zamba-7B-v1-phase1/blob/main/config.json)
- [Transformers Zamba docs](https://huggingface.co/docs/transformers/model_doc/zamba)

## 2. High-level architecture

Zamba is a text-only causal decoder, but not a normal all-attention decoder. Every layer owns a Mamba mixer. A subset of layers are `hybrid`: they first run a shared/tied attention+MLP branch on `concat(current_hidden, original_token_embedding)`, linearly project that branch output, add it into the Mamba input path, then run the layer-local Mamba mixer.

```text
tokenizer/input_ids -> token embedding
  -> keep original embedding clone for all hybrid layers
  -> 76 decoder layers:
       mamba layer: RMSNorm -> multi-head Mamba mixer -> residual add
       hybrid layer: shared attention+MLP on concat(hidden, original_embedding)
                     -> linear(hidden_size -> hidden_size)
                     -> add into Mamba input
                     -> RMSNorm -> multi-head Mamba mixer -> residual add
  -> final RMSNorm -> last-token-or-selected-token LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: Llama tokenizer, BOS insertion by tokenizer config, padding mask construction.
- GPU/runtime prefill: embeddings, all Mamba sequence scans, hybrid causal attention prefill, final norm, optional limited logits.
- GPU/runtime decode: one-token Mamba conv/state update per layer plus KV-cache update only for hybrid attention layers.
- Independently stageable pieces: Mamba mixer parity, hybrid attention branch parity, cache ABI, LM head/logits slicing.

## 3. Important config dimensions

For the official 7B configs:

| Field | Value | Source / runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 32000 | token embedding and LM head rows |
| `hidden_size` | 3712 | decoder hidden width |
| `num_hidden_layers` | 76 | all layers include Mamba |
| `layers_block_type` | 63 `mamba`, 13 `hybrid` | official configs include explicit list |
| hybrid layer ids | 2, 7, 13, ..., 73 | from config list/default generation |
| `attention_hidden_size` | 7424 | source default `2 * hidden_size`; input is concat hidden + original embedding |
| `num_attention_heads` | 16 | hybrid attention Q heads |
| `num_key_value_heads` | 16 | no GQA for official configs |
| `attention_head_dim` | 464 | `2 * hidden_size // num_attention_heads`; Q/K/V width 7424 |
| `intermediate_size` | 14848 | gated MLP width in attention branch |
| `mamba_expand` | 2 | Mamba intermediate width is 7424 |
| `n_mamba_heads` | 2 | Mamba SSM heads |
| Mamba head dim | 3712 | `mamba_expand * hidden_size / n_mamba_heads` |
| `mamba_d_state` | 16 | SSM recurrent state width |
| `mamba_d_conv` | 4 | depthwise causal conv kernel/cache length |
| `mamba_dt_rank` | 232 | official config; source auto default is `ceil(hidden_size / 16)` |
| activations | MLP `gelu`, Mamba `silu` | config |
| `rms_norm_eps` | 1e-5 | config |
| `max_position_embeddings` | 4096 | used by config/mask machinery, not by RoPE in this source |
| `rope_theta` | 10000 | present in checkpoint config, not read by current zamba modeling source |
| `sliding_window` | null | present in checkpoint config, no local sliding attention implementation observed |
| `torch_dtype` | bfloat16 | checkpoint config |
| `use_cache` | true | mixed cache: attention KV plus Mamba conv/recurrent states |
| `use_mamba_kernels` | true | fast path requires `mamba-ssm`, `causal-conv1d`, CUDA |
| `num_logits_to_keep` | 1 | generation path computes only selected trailing logits by default |

Representative checkpoint sweep:

| Model id | `model_type` | Layers | Hybrid layers | Hidden | Attention heads/KV | Mamba heads/state/conv | dtype | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `Zyphra/Zamba-7B-v1` | `zamba` | 76 | 13 | 3712 | 16 / 16, head_dim 464 | 2 / 16 / 4 | bf16 | main production config |
| `Zyphra/Zamba-7B-v1-phase1` | `zamba` | 76 | 13 | 3712 | 16 / 16, head_dim 464 | 2 / 16 / 4 | bf16 | same runtime structure |
| local test fixture | `zamba` | 5 | 2 by test expectation | 64 | 4 / 4 | 2 / 16 / 4 | default torch | `use_mamba_kernels=False`; useful debug shape |

## 3a. Family variation traps

- `attention_hidden_size != hidden_size`: hybrid attention consumes `concat(hidden_states, original_hidden_states)` with width `2H`.
- `attention_head_dim` is based on `attention_hidden_size`, so official attention output width before `o_proj` is 7424, not 3712.
- Official zamba has no GQA (`num_key_value_heads == num_attention_heads`), but source supports `num_key_value_heads < num_attention_heads` through `repeat_kv`.
- Checkpoint config advertises `rope_theta` and `sliding_window`, but the inspected `modeling_zamba.py` does not apply RoPE and does not implement local/sliding attention in `ZambaAttention`.
- Hybrid attention weights are tied/shared by `_tied_weights_keys`; lowering must preserve this logical aliasing instead of cloning the shared transformer branch independently.
- `use_mamba_kernels=True` is not just an optimization flag in source: if fast kernels are unavailable or the module is not on CUDA, `forward()` raises. A DinoML importer should either route to a native Mamba provider or force/rewrite an admitted slow-reference mode.
- Mamba cache is fixed-size and stateful, not a KV cache. It requires conv state `[B, 7424, 4]` and recurrent state `[B, 2, 3712, 16]` per Mamba layer.
- In `cuda_kernels_forward`, `use_precomputed_states` checks `cache_params.has_previous_state` without calling it. The slow path calls `cache_params.has_previous_state(self.layer_idx)`. Match source behavior deliberately during parity investigation.
- `attention_probs_dropout_prob` appears in tests but current config field is `attention_dropout`; treat legacy names as ignored unless config loading maps them.
- Zamba2 configs and source are materially different (`model_type: zamba2`, RoPE, Mamba2/gated norms). They need a separate audit.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,3712]`.
- Clone/carry original embeddings for all hybrid layers.
- Concatenate on last dim: `[B,S,3712] + [B,S,3712] -> [B,S,7424]`.
- Reshape/view/transpose/contiguous around attention heads and Mamba heads.
- Split/chunk projected tensors: Mamba `Linear(3712 -> 14848)` split into hidden/gate each `[B,S,7424]`; SSM params split into dt/B/C widths `[232,16,16]`.
- Causal/padding mask creation for attention; 2D attention mask also gates Mamba conv inputs.
- Last-token or index-select logits slicing from hidden states.

Neural primitives:

- RMSNorm with fp32 variance, shape `[*, 3712]` and `[*, 7424]`.
- Linear/GEMM without bias for embeddings-projection, attention Q/K/V/O, MLP, hybrid projection, LM head.
- MLP SwiGLU-like pattern but using GELU gate: `down(gelu(gate(x)) * up(x))`, shapes `3712 -> 14848 -> 3712`.
- Elementwise add/mul, `rsqrt`, `exp`, `softplus`, SiLU, GELU.
- Depthwise causal Conv1d: groups=7424, kernel=4, padding=3, bias=true, activation SiLU, truncate to input sequence length.

Attention primitives:

- Causal self-attention in hybrid layers only.
- Q projection `Linear(7424 -> 7424)`, K/V `Linear(7424 -> 7424)` for official configs.
- Head shape `[B,16,S,464]`; attention scale is `(head_dim / 2) ** -0.5`, not the standard `head_dim ** -0.5`.
- Eager math: `Q @ K^T`, add mask, fp32 softmax, dropout only in training, `P @ V`, output projection `Linear(7424 -> 3712)`.
- Optional `repeat_kv` required if non-official configs use fewer KV heads.
- Source declares `_supports_flash_attn=False`, `_supports_sdpa=False`, but uses `ALL_ATTENTION_FUNCTIONS.get_interface`; tests exercise FA2 separately with caveats. DinoML should first target eager-equivalent attention and then admit optimized attention behind parity guards.

Recurrent/state-space cache ops:

- Per Mamba layer conv cache update with static address semantics and rolling update.
- Per Mamba layer recurrent SSM state update, shape `[B,2,3712,16]`.
- Selective scan over sequence for prefill; selective state update for decode.
- Beam/cache reorder over batch dimension for both Mamba states and attention KV.

Packed/varlen metadata ops:

- No packed sequence or `cu_seqlens` path is explicit in Zamba source. Standard Transformers mask helpers may generate backend-specific mask forms; first DinoML path should admit dense padded batches.

Quantized/packed weight metadata ops:

- No source-coupled packed weight format in `modeling_zamba.py`. BitsAndBytes appears only in tests for loading/FA2 behavior, not as required architecture.

Preprocessing-coupled ops:

- Llama tokenizer. Tokenizer config uses `add_bos_token=true`, `add_eos_token=false`, pad token id 0, BOS 1, EOS 2. Tokenization is CPU/data-pipeline work.

## 5. Layer/block breakdown

Mamba decoder layer, repeated for every layer:

```text
residual = x                                      # [B,S,3712]
x_for_mamba = x + transformer_hidden if hybrid else x
x_norm = RMSNorm_3712(x_for_mamba)
proj = Linear(3712 -> 14848, bias=mamba_proj_bias false)(x_norm)
hidden, gate = split(proj, 7424, 7424)
hidden = depthwise_causal_conv1d(hidden, groups=7424, kernel=4, bias=true)
hidden = silu(hidden)
ssm_params = per-mamba-head matmul x_proj_weight [2,264,3712] @ hidden_head
dt, B, C = split(ssm_params, [232,16,16])
dt = softplus(dt_proj_weight [2,3712,232] @ dt + dt_bias [2,3712])
A = -exp(A_log [2,3712,16])
scan/update SSM state with D [2,3712] and silu(gate)
out = Linear(7424 -> 3712, bias=false)(scan_output)
x = residual + out
```

Hybrid layer extra branch before Mamba:

```text
attn_input = concat(x, original_embedding)        # [B,S,7424]
attn_input = RMSNorm_7424(attn_input)
q = Linear(7424 -> 7424, bias=false)
k = Linear(7424 -> 7424, bias=false)
v = Linear(7424 -> 7424, bias=false)
attn = causal_attention(q,k,v, scale=(464/2)^-0.5)
attn = Linear(7424 -> 3712, bias=false)(attn)
ff = RMSNorm_3712(attn)
ff = Linear(14848 -> 3712)(gelu(Linear(3712 -> 14848)(ff)) * Linear(3712 -> 14848)(ff))
transformer_hidden = Linear(3712 -> 3712, bias=false)(ff)
Mamba layer consumes x + transformer_hidden before RMSNorm
```

Final:

```text
x = RMSNorm_3712(x)
logits = Linear(3712 -> 32000, bias=false)(x[:, selected_positions, :])
```

Aliasing contract:

- `ZambaForCausalLM` ties `lm_head.weight` to `model.embed_tokens.weight`.
- `ZambaModel` declares `_tied_weights_keys` for hybrid `shared_transf` modules so later hybrid attention+MLP branches share the first hybrid branch weights logically.

## 6. Attention requirements

Attention is causal self-attention and exists only in hybrid layers. Official config has 13 attention applications among 76 layers.

| Requirement | Zamba value |
| --- | --- |
| Causality | causal decoder self-attention |
| Query source | current hidden concatenated with original embeddings |
| Key/value source | same concatenated tensor |
| Q heads | 16 |
| KV heads | 16 official, source supports repeat for fewer KV heads |
| head dim | 464 |
| Q/K/V projected width | 7424 / 7424 / 7424 official |
| output width | 7424 attention result -> `o_proj` to 3712 |
| mask | additive causal/padding mask from `create_causal_mask` |
| softmax | fp32 softmax, cast back to query dtype |
| scale | `(head_dim / 2) ** -0.5` |
| RoPE/ALiBi | none in inspected zamba source |
| cache | attention KV cache only for hybrid layers, plus Mamba state cache for all layers |

Cache shape before KV repeat:

```text
key_states/value_states per hybrid layer: [B, num_key_value_heads, T, head_dim]
official: [B, 16, T, 464]
after repeat_kv if GQA/MQA config: [B, num_attention_heads, T, 464]
```

Right padding caveat: tests note Zamba does not support right padding plus `use_cache` with FlashAttention 2. First DinoML attention path should reject that optimized combination or canonicalize to eager-equivalent dense attention.

## 7. Position encoding and custom math

The inspected zamba implementation does not apply RoPE, ALiBi, or learned absolute position embeddings. `position_ids` are generated and passed to `create_causal_mask`, but Q/K are not position-rotated.

Custom attention scaling:

```python
def zamba_attention_scores(q, k, mask):
    scores = (q @ k.transpose(-2, -1)) * ((q.shape[-1] / 2) ** -0.5)
    if mask is not None:
        scores = scores + mask
    probs = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
    return probs
```

Custom Mamba slow-reference math, schematic:

```python
A = -exp(A_log.float())
dt = softplus(dt_proj_weight @ time_step + dt_proj_bias)
discrete_A = exp(A * dt)
discrete_B = dt * B
state = discrete_A[t] * state + discrete_B[t] * hidden[t]
y_t = state @ C[t]
y_t = (y_t + hidden[t] * D) * silu(gate[t])
```

Precomputable: `A = -exp(A_log.float())`, depthwise conv weights, and static SSM parameters `D`; not precomputable: dt/B/C because they are token-dependent.

## 8. Preprocessing and input packing

Runtime graph inputs:

- `input_ids`: `[B,S]` int token ids, or `inputs_embeds` `[B,S,3712]`.
- `attention_mask`: `[B,S_seen + S]` with 1 for valid tokens and 0 for padding. The same mask participates in attention mask creation and Mamba hidden-state gating around the depthwise conv.
- `position_ids`: optional; source creates `[1,S] + past_seen_tokens` if omitted.

Tokenizer/data pipeline:

- `tokenizer_class` is `LlamaTokenizer`.
- BOS token is added by tokenizer config; EOS is not automatically added.
- Special ids: pad/unk 0, BOS 1, EOS 2.
- No multimodal placeholder, scatter, image/audio packing, or packed sequence metadata exists in this family.

GPU/runtime owns embedding lookup, masks, model body, cache updates, and logits. Tokenization and text generation controller policy can stay outside DinoML for first parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Mamba in-projection split

Source pattern:

```text
Linear(H -> 2I) -> transpose -> view -> chunk(hidden, gate)
```

Replacement:

```text
single GEMM producing packed [hidden, gate], then views/slices consumed by conv and gate activation
```

Preconditions:

- Weight layout is standard PyTorch linear `[2I, H]`.
- Split order is hidden first, gate second after `.view(batch, -1, 2, seq).chunk(2, dim=2)`.
- Bias is governed by `mamba_proj_bias`; official config false.

Parity test sketch: compare hidden/gate tensors before conv for random `[B,S,3712]` in fp32 and bf16.

### Rewrite: depthwise causal Conv1d -> specialized short causal conv

Source pattern:

```text
Conv1d(channels=I, groups=I, kernel=4, padding=3)(x)[..., :S] -> silu
```

Replacement:

```text
channelwise causal rolling-window kernel with optional conv cache update and fused SiLU
```

Preconditions:

- `groups == in_channels == out_channels == 7424`.
- `kernel_size == mamba_d_conv == 4`.
- Padding/truncation exactly preserves causal alignment.
- Attention mask multiplication before and after conv must be preserved.

Failure cases: non-4 kernel, disabled conv bias, packed sequence masks, or shorter-than-kernel prefill need explicit tests because cache comments call out edge behavior.

### Rewrite: Mamba selective scan provider

Source pattern:

```text
token-dependent dt/B/C projection -> softplus/exp discretization -> recurrent scan -> gate multiply
```

Replacement:

```text
provider-backed selective_scan for prefill and selective_state_update for decode
```

Preconditions:

- Fixed `mamba_d_state=16`, `n_mamba_heads=2`, official `mamba_head_dim=3712`.
- Preserve fp32 math around `A_log`, `D`, dt bias, and softplus.
- Preserve per-head independent scan and concatenation order.

Failure cases: using only dense elementwise loops will be too slow; source fast path requires external kernels and CUDA.

### Rewrite: hybrid attention branch fusion

Source pattern:

```text
concat -> RMSNorm_7424 -> Q/K/V projections -> causal attention -> O projection -> RMSNorm_3712 -> GELU-gated MLP
```

Replacement:

```text
fused concat/RMSNorm, packed QKV GEMM if weights are transformed, optimized causal attention, fused GELU gate MLP
```

Preconditions:

- Shared transformer branch weight aliasing preserved.
- Q/K/V split order must match separate PyTorch modules.
- Attention scale is Zamba-specific `(head_dim/2)^-0.5`.
- No RoPE should be inserted.

Failure cases: treating it like Llama/Mistral attention with RoPE or standard scale breaks parity.

### Rewrite: last-token-only logits

Source pattern:

```text
slice_indices = slice(-logits_to_keep, None)
lm_head(hidden[:, slice_indices, :])
```

Replacement:

```text
gather selected hidden rows before vocab GEMM
```

Preconditions:

- `labels is None` for generation.
- `logits_to_keep` is integer or supported tensor index.

## 10. Kernel fusion candidates

Highest priority:

- Mamba selective scan/update provider, including conv and recurrent cache ABI. This dominates correctness and performance.
- RMSNorm for widths 3712 and 7424 with fp32 variance and dtype-preserving output.
- Depthwise causal conv kernel fused with mask and SiLU for kernel size 4.
- Gated MLP/GELU multiply for hybrid attention branch.
- Causal attention for `[B,16,S,464]` with Zamba scale and optional KV cache.

Medium priority:

- Packed QKV projection for hybrid layers after weight transform.
- Mamba `x_proj_weight` and `dt_proj_weight` batched/headed small-K GEMMs.
- Last-token-only LM head for generation.
- Cache reorder/reset kernels for Mamba state plus attention KV.

Lower priority:

- Sequence classification pooling/indexing head.
- Training losses/dropout.
- BitsAndBytes/4-bit external loading behavior.
- FlashAttention 2 parity; source does not declare native support and tests carry caveats.

## 11. Runtime staging plan

Stage 1: parse config and weight manifest. Reject `model_type != "zamba"` and route Zamba2 separately. Preserve tied token embedding/LM head and shared transformer branch aliases.

Stage 2: implement CPU/PyTorch-like reference for one Mamba mixer with `use_mamba_kernels=False` semantics. Validate shapes and slow selective scan.

Stage 3: implement one full Mamba decoder layer parity including conv/recurrent cache prefill and decode.

Stage 4: implement one hybrid layer parity, including concat with original embeddings, shared attention branch, hybrid projection, and Mamba add-in.

Stage 5: build full prefill parity for `ZambaModel` at small debug config and official dimensions with dense masks.

Stage 6: add decode cache parity with mixed cache manifest:

```text
for every layer: conv_state [B,7424,4], recurrent_state [B,2,3712,16]
for hybrid layers only: key/value [B,16,T,464]
```

Stage 7: replace slow scan with provider-backed selective scan/update, then add optimized attention.

Stage 8: enable production generation path with last-token logits and batching constraints.

Can be stubbed initially: sequence classification head, training losses, dropout, offload, FlashAttention, quantized external loaders.

## 12. Parity and validation plan

- Config parity: load official configs and assert derived dimensions, layer type counts, hybrid indices, tied weight declarations.
- RMSNorm random tensor parity for `[2,7,3712]` and `[2,7,7424]`; fp32 tolerance `1e-6`, bf16/fp16 tolerance around `1e-2`.
- Mamba projection split parity: compare hidden/gate split order after `in_proj`.
- Depthwise conv parity: random prefill sequences and one-token decode with cache, including attention masks with padding.
- Slow selective scan parity: single Mamba head and full two-head mixer against Transformers slow path with `use_mamba_kernels=False`.
- Hybrid attention parity: verify scale `(464/2)^-0.5`, no RoPE, mask addition, fp32 softmax.
- One Mamba layer parity, then one hybrid layer parity, then after-N-layer parity at local debug config.
- Full prefill logits parity for official model with `logits_to_keep=1`; compare last-token logits.
- Decode parity: prefill then one and several token decode; verify both Mamba states and attention KV update/reorder.
- Tied-weight parity: ensure shared transformer branch and tied LM head are not duplicated in state dict/import.
- Recommended tolerances: fp32 `1e-4` end-to-end for small tests; bf16 `1e-2` to `3e-2` for logits depending on scan provider and attention backend.

## 13. Performance probes

- Mamba prefill selective-scan throughput by sequence length: 128, 512, 2048, 4096.
- Mamba decode selective-state-update tokens/sec with cache for batch sizes 1, 4, 16.
- Depthwise conv update bandwidth and launch overhead for `[B,7424,4]`.
- Hybrid attention prefill throughput at the 13 attention layers, separated from Mamba time.
- KV cache plus Mamba state memory:
  - Mamba fixed state per layer per batch: `7424*4 + 2*3712*16` elements.
  - Attention KV grows only for 13 hybrid layers: `2*16*T*464` elements per hybrid layer per batch.
- LM head cost with `logits_to_keep=1` versus all logits.
- End-to-end prefill/decode split and batch-size sweep.
- Provider comparison: slow reference scan versus native selective-scan/update provider.

## 14. Skip/defer list

- Training, labels, and dropout.
- Sequence classification and zero-shot pipelines.
- FlashAttention 2 and SDPA special handling.
- Right-padding plus cache optimized-attention combinations.
- Quantized BitsAndBytes loading.
- Multi-GPU/offload.
- Zamba2, Mamba2, RoPE-enabled variants.
- General packed/varlen sequence metadata.
- Beam search reorder beyond basic cache index-select parity for first generation path.

## 15. Final implementation checklist

- [ ] Parse `ZambaConfig` and derive `attention_hidden_size`, `attention_head_dim`, `mamba_dt_rank`, and `layers_block_type`.
- [ ] Reject or separately route `model_type=zamba2` and config fields not implemented by zamba source.
- [ ] Load/tie token embedding and LM head weights.
- [ ] Preserve shared transformer branch aliases for hybrid layers.
- [ ] Implement RMSNorm for 3712 and 7424 widths.
- [ ] Implement Mamba `in_proj` split with exact hidden/gate order.
- [ ] Implement depthwise causal Conv1d kernel/cache update with mask and SiLU.
- [ ] Implement Mamba dt/B/C projections and slow selective scan reference.
- [ ] Add provider-backed selective scan and selective state update.
- [ ] Implement Mamba cache manifest and lifecycle: conv state, recurrent state, reset, reorder.
- [ ] Implement hybrid attention branch with Zamba-specific scale and no RoPE.
- [ ] Implement optional KV repeat for non-official GQA configs.
- [ ] Implement MLP GELU gate fusion candidate.
- [ ] Implement final RMSNorm and last-token-only LM head.
- [ ] Add single-block, hybrid-block, prefill-logit, and decode-token parity tests.
- [ ] Benchmark Mamba prefill/decode, hybrid attention, cache memory, and LM-head slicing.
