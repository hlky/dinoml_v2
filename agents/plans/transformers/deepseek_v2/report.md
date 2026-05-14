# DeepSeek-V2 Transformers Family Audit

Primary target: `DeepseekV2ForCausalLM` inference and generation on CUDA. This is a source/config audit only; no DinoML runtime code was edited, no DinoML tests were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: deepseek_v2
Primary task: causal LM prefill/decode/generation
Local source root: X:/H/transformers
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py`
- `X:/H/transformers/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py`
- `X:/H/transformers/src/transformers/models/deepseek_v2/modular_deepseek_v2.py`
- Cross-checks: `src/transformers/models/qwen2_moe/modeling_qwen2_moe.py`, `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`, `src/transformers/modeling_rope_utils.py`, `src/transformers/configuration_utils.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deepseek_v2/configuration_deepseek_v2.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/deepseek_v2/modular_deepseek_v2.py`

Representative config and generation metadata inspected from Hugging Face raw files, snapshotted under `H:/dinoml_v2/agents/plans/transformers/deepseek_v2/_sources/`:

- `https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V2.5/raw/main/config.json`
- `https://huggingface.co/yujiepan/deepseek-v2-tiny-random/raw/main/config.json`
- Matching `generation_config.json` files for all six repos above.

Authoritative source note: `modeling_deepseek_v2.py` and `configuration_deepseek_v2.py` are generated from `modular_deepseek_v2.py`; future Transformers source edits should target the modular file. This report uses the generated files for concrete expanded implementation details and the modular file for intended inheritance notes. The modular source derives from Llama-style model classes, Qwen2-MoE packed experts, and custom DeepSeek MLA attention.

Missing files or assumptions:

- No processor, image/audio preprocessor, or tokenizer coupling was required for the core graph. Tokenization is text-only and outside this directory.
- Public checkpoint configs include `auto_map` entries for older remote-code files named `configuration_deepseek`/`modeling_deepseek`. The tiny-random mirror also points its `auto_map` back to the DeepSeek-V2-Chat remote-code files. This report scopes required runtime behavior to the pinned in-library `deepseek_v2` source. Remote-code-only differences should be audited separately if DinoML loads those exact remote files.
- Raw configs use legacy `rope_scaling` plus `rope_theta`; the pinned config class exposes `rope_parameters`. `PreTrainedConfig`/RoPE utilities normalize legacy fields into `rope_parameters`, including `rope_type="yarn"` for sampled DeepSeek configs.

## 2. High-level architecture

DeepSeek-V2 is a text-only decoder-only sparse MoE language model with MLA-style low-rank attention projections:

```text
token ids -> embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
                         | each block: MLA-style causal self-attention + dense or MoE SwiGLU MLP
prefill: full prompt causal attention + KV cache fill
decode: new token(s) + cache update + last-token logits
```

Stage decomposition:

- CPU/data pipeline: tokenizer, BOS/EOS/chat formatting, padding/attention mask construction, generation sampling controls.
- GPU/runtime prefill: embeddings, shared YaRN/default RoPE complex frequency generation, repeated decoder blocks, full causal self-attention, top-k MoE routing/expert execution, final norm, logits.
- GPU/runtime decode: position IDs from cache length, low-rank Q/KV projections for new token(s), RoPE on partial Q/K dimensions, KV cache append, attention over cache, MoE, last-token logits.
- Generation controller: standard `GenerationMixin`; sampled generation configs set BOS/EOS and sampling defaults.

Implemented heads:

- Required for target: `DeepseekV2ForCausalLM`.
- Optional/deferred: base `DeepseekV2Model` for hidden states.
- Deferred for first causal-LM target: `DeepseekV2ForSequenceClassification`, implemented through a generic classification head.

The defining differences from already audited Mixtral/Qwen patterns are:

- Attention is not ordinary Q/K/V projection. It uses query LoRA for full models, compressed KV plus separate shared RoPE key channels, and key dimensions split into no-position (`qk_nope_head_dim`) and RoPE (`qk_rope_head_dim`) parts.
- The native source still caches expanded per-head K/V tensors, not the latent KV representation. DinoML can optimize to latent-cache MLA later, but source parity starts with post-expansion cache tensors.
- MoE resembles Qwen2-MoE packed experts, not Mixtral's separate expert modules, and includes a shared expert MLP branch added to routed experts. Unlike Qwen2-MoE, DeepSeek does not have a sigmoid gate on the shared expert branch.

## 3. Important config dimensions

Source defaults from `DeepseekV2Config`:

| Field | Default / behavior |
| --- | --- |
| `vocab_size` | 32000 source default; sampled official configs use 102400 |
| `hidden_size` | 4096 default |
| `intermediate_size` | 11008 dense MLP default |
| `moe_intermediate_size` | 1407 per routed expert default |
| `num_hidden_layers` | 32 default |
| `num_attention_heads` | 32 default |
| `num_key_value_heads` | set to `num_attention_heads` if omitted; sampled configs use equality, so no GQA grouping despite MQA-like projection naming |
| `head_dim` | post-init sets `head_dim = qk_rope_head_dim`; attention actually uses `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` |
| `qk_nope_head_dim` | 128 default, no-position key/query sub-dimension |
| `qk_rope_head_dim` | 64 default, RoPE-applied sub-dimension |
| `qk_head_dim` | derived in attention as 192 for sampled configs |
| `v_head_dim` | 128 default, value sub-dimension |
| `q_lora_rank` | 1536 default; `None` means direct Q projection |
| `kv_lora_rank` | 512 default |
| `first_k_dense_replace` | 0 default; sampled configs use 1, so layer 0 is dense MLP and later layers are MoE |
| `n_routed_experts` | 64 default |
| `n_shared_experts` | 2 default |
| `num_experts_per_tok` | `None` default; sampled configs use 6 |
| `topk_method` | `"greedy"` default; full models use `"group_limited_greedy"` |
| `n_group` / `topk_group` | optional group-limited routing fields |
| `routed_scaling_factor` | 1.0 default; full models use 16.0 |
| `norm_topk_prob` | field exists, but current DeepSeek route code does not renormalize selected probabilities |
| `hidden_act` | `silu` |
| `rms_norm_eps` | `1e-6` |
| `use_cache` | true |
| `attention_bias` / `mlp_bias` | false in defaults and sampled configs |
| `rope_parameters` | standardized RoPE dict; sampled legacy configs normalize `rope_scaling` type `yarn` plus `rope_theta` |
| `tie_word_embeddings` | false |

Representative checkpoint sweep. Dimensions are from `config.json`; effective RoPE source behavior comes from normalized `rope_parameters`.

| Model id | Layers | H | Heads/KV | Q LoRA | KV LoRA | QK no/rope | V dim | Dense I | Expert I | Experts/shared/top-k | Routing | Max pos | RoPE | Dtype |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `yujiepan/deepseek-v2-tiny-random` | 2 | 8 | 2/2 | 2 | 2 | 2/2 | 2 | 16 | 4 | 160/2/6 | group-limited, 8 groups, top 3 groups, scale 16 | 163840 | YaRN factor 40, theta 10000, mscale 0.707 | bf16 metadata |
| `deepseek-ai/DeepSeek-V2-Lite` | 27 | 2048 | 16/16 | none | 512 | 128/64 | 128 | 10944 | 1408 | 64/2/6 | greedy, scale 1 | 163840 | YaRN factor 40, theta 10000, mscale 0.707 | bf16 |
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | 27 | 2048 | 16/16 | none | 512 | 128/64 | 128 | 10944 | 1408 | 64/2/6 | greedy, scale 1 | 163840 | same as Lite | bf16 |
| `deepseek-ai/DeepSeek-V2` | 60 | 5120 | 128/128 | 1536 | 512 | 128/64 | 128 | 12288 | 1536 | 160/2/6 | group-limited, 8 groups, top 3 groups, scale 16 | 163840 | YaRN factor 40, theta 10000, mscale 0.707 | bf16 |
| `deepseek-ai/DeepSeek-V2-Chat` | 60 | 5120 | 128/128 | 1536 | 512 | 128/64 | 128 | 12288 | 1536 | 160/2/6 | same as V2 | 163840 | same as V2 | bf16 |
| `deepseek-ai/DeepSeek-V2.5` | 60 | 5120 | 128/128 | 1536 | 512 | 128/64 | 128 | 12288 | 1536 | 160/2/6 | group-limited, scale 16 | 163840 | YaRN factor 40, theta 10000, mscale 1.0 | bf16 |

Generation config sweep:

| Model id | Generation fields observed |
| --- | --- |
| all sampled DeepSeek-V2/V2.5 plus tiny-random configs | `bos_token_id=100000`, `eos_token_id=100001`, `do_sample=true`, `temperature=0.3`, `top_p=0.95` |

## 3a. Family variation traps

- `head_dim` is misleading if treated like Llama/Qwen. The config post-init sets it to `qk_rope_head_dim` (64), but attention score dimension is `qk_nope_head_dim + qk_rope_head_dim` (192). Scaling uses `192 ** -0.5`.
- `hidden_size != num_attention_heads * qk_head_dim` for full models. Example V2 has `5120 != 128 * 192`. Q/O projection widths are independent of `hidden_size`.
- `num_key_value_heads` equals query heads in sampled configs, so source attention does not use GQA grouping. The "MQA" in `kv_a_proj_with_mqa` refers to one shared RoPE key slice before expansion, not to KV cache heads fewer than query heads.
- Native source caches expanded `key_states [B, heads, T, qk_nope+qk_rope]` and `value_states [B, heads, T, v_head_dim]`. It does not expose a latent KV cache ABI even though projections are MLA-style.
- FlashAttention path pads values from `v_head_dim=128` to `qk_head_dim=192`, calls the attention backend, then crops the attention output back to 128 value dims before `o_proj`.
- Lite configs set `q_lora_rank=null`, so Q is a direct `q_proj`. Full V2/V2.5 use `q_a_proj -> RMSNorm(rank) -> q_b_proj`.
- Public raw configs include remote-code `auto_map`, `aux_loss_alpha`, `moe_layer_freq`, `scoring_func`, `seq_aux`, and sometimes `ep_size`. The pinned in-library source does not consume these fields in forward graph construction; do not require them unless auditing remote code.
- Current DeepSeek source ignores `norm_topk_prob` in routing. It softmaxes over all experts, selects top-k, then multiplies selected weights by `routed_scaling_factor`; it does not divide by the selected-probability sum.
- Full models use `group_limited_greedy`: group scores are max expert probabilities per group, top groups are selected, non-selected groups are masked to zero, then expert top-k is taken. Lite uses plain greedy over all experts.
- `first_k_dense_replace=1` in sampled configs, so layer 0 uses dense `DeepseekV2MLP`; layers 1..N-1 use `DeepseekV2Moe`.
- DeepSeek MoE includes shared experts as a dense SwiGLU MLP with intermediate size `moe_intermediate_size * n_shared_experts`. This branch is added directly to routed expert output. This differs from Qwen2-MoE's sigmoid-gated shared expert and Qwen3-MoE's absence of shared experts.
- YaRN RoPE is required for official sampled configs. A default-RoPE-only implementation is not sufficient for checkpoint parity.
- Layout translation is low value for text-only DeepSeek-V2. Protect attention and MoE routing with no-layout-translation guards around head/sequence axes, top-k expert axis, `softmax(dim=-1)`, complex RoPE final-dim pairing, and scatter-add token order.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- Shape/view ops: `view`, `reshape`, `transpose`, `contiguous`, `split`, `cat`, `expand`, slicing for `logits_to_keep`.
- Complex-view RoPE helpers: `reshape(..., -1, 2)`, `view_as_complex`, complex multiply, `view_as_real`, flatten.
- MoE route indexing: `softmax`, `topk`, `max`, `zeros_like`, `scatter_`, boolean mask, `masked_fill`, `one_hot`, `permute`, `sum`, `nonzero`, `where`, token gather, `index_add_`.
- Residual adds and dtype casts.

Neural network primitives:

- RMSNorm over hidden dimension, Q-LoRA rank dimension, and KV-LoRA rank dimension with fp32 variance math and output cast back to input dtype.
- Bias-optional linears:
  - Lite direct Q: `2048 -> 16 * 192 = 3072`, bias false.
  - Full Q LoRA: `5120 -> 1536`, RMSNorm, then `1536 -> 128 * 192 = 24576`.
  - KV A: Lite/full `H -> kv_lora_rank + qk_rope = 512 + 64 = 576`.
  - KV B: Lite `512 -> 16 * (128 + 128) = 4096`; full `512 -> 128 * 256 = 32768`.
  - O: Lite `16 * 128 = 2048 -> 2048`; full `128 * 128 = 16384 -> 5120`.
  - Dense layer-0 MLP: Lite `2048 -> 10944 -> 2048`; full `5120 -> 12288 -> 5120`.
  - Router: Lite `2048 -> 64`; full `5120 -> 160`.
  - Routed expert packed gate/up: Lite `2048 -> 2816` split into two 1408 halves; full `5120 -> 3072` split into two 1536 halves.
  - Routed expert down: Lite `1408 -> 2048`; full `1536 -> 5120`.
  - Shared expert MLP: Lite intermediate `1408 * 2 = 2816`; full intermediate `1536 * 2 = 3072`.
  - LM head: Lite `2048 -> 102400`; full `5120 -> 102400`.
- SiLU and multiply for SwiGLU.

Attention primitives:

- Causal self-attention with query/key dimension 192 and value dimension 128.
- No-position and RoPE-position split for Q/K.
- Shared single-head RoPE key slice expanded across all heads after RoPE.
- Eager fallback: optional `repeat_kv`, QK matmul, scale by `192 ** -0.5`, add mask, fp32 softmax, dropout only in training, attention-value matmul.
- Flash/SDPA/flex attention dispatch through `ALL_ATTENTION_FUNCTIONS`; Flash path requires V padding/cropping when value dim differs from score dim.

Position/rotary ops:

- YaRN RoPE inverse-frequency generation through `ROPE_INIT_FUNCTIONS["yarn"]`.
- DeepSeek custom RoPE apply uses complex multiply on half-size complex pairs, not the common `rotate_half` formulation in audited Qwen2/Mixtral reports.
- Position embeddings are computed once in `DeepseekV2Model.forward` and shared across layers.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache` is true and no cache is passed.
- Cache `get_seq_length()` for default `position_ids`.
- KV cache update per layer after full key construction. Cached K is post-RoPE and expanded across heads.
- `logits_to_keep`: int or tensor index selection before LM head.

Distributed/tensor-parallel metadata:

- Config declares TP plans: `q_b_proj` colwise, KV A special `mla_kv_a_proj`, KV B colwise, O rowwise, packed experts colwise/down rowwise, experts as `moe_tp_experts`, shared experts colwise/rowwise, and LM head colwise gather output. First DinoML integration can be single-GPU, but the MLA-specific KV A plan should be preserved in metadata.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
x0: [B,S,H]
residual = x0
x = RMSNorm_H(x0)

if q_lora_rank is None:
    q = Linear(H -> heads * 192, bias=False)(x)
else:
    q = Linear(q_lora_rank -> heads * 192, bias=False)(
            RMSNorm_rank(Linear(H -> q_lora_rank, bias=attention_bias)(x)))
q = q.view(B,S,heads,192).transpose(1,2)
q_nope, q_pe = split(q, [128,64], dim=-1)

compressed_kv = Linear(H -> kv_lora_rank + 64, bias=attention_bias)(x)
latent_kv, k_pe = split(compressed_kv, [512,64], dim=-1)
kv = Linear(512 -> heads * (128 + 128), bias=False)(RMSNorm_512(latent_kv))
kv = kv.view(B,S,heads,256).transpose(1,2)
k_nope, v = split(kv, [128,128], dim=-1)
k_pe = k_pe.view(B,1,S,64)

q_pe, k_pe = complex_rope(q_pe, k_pe, freqs_cis)
k_pe = expand(k_pe, B, heads, S, 64)
q_full = cat(q_nope, q_pe, dim=-1)      # [B,heads,S,192]
k_full = cat(k_nope, k_pe, dim=-1)      # [B,heads,S,192]
k_full, v = cache.update(k_full, v, layer_idx) if cache enabled
attn = causal_attention(q_full, k_full, v, mask, scale=192^-0.5)
if flash_backend: pad v/attention output to 192 then crop back to 128
x = residual + Linear(heads * 128 -> H, bias=attention_bias)(attn)

residual = x
x = RMSNorm_H(x)
if layer_idx < first_k_dense_replace:
    y = Linear(I_dense -> H)(silu(Linear(H -> I_dense)(x)) * Linear(H -> I_dense)(x))
else:
    router_logits = Linear(H -> E, bias=False)(x.float())
    probs = softmax(router_logits, dim=-1, dtype=float32)
    selected_experts, selected_weights = greedy_or_group_limited_topk(probs)
    selected_weights = selected_weights * routed_scaling_factor
    routed = packed_expert_loop(x, selected_experts, selected_weights)
    shared = Linear(I_shared -> H)(silu(Linear(H -> I_shared)(x)) * Linear(H -> I_shared)(x))
    y = routed + shared
x = residual + y
```

Model tail:

```text
hidden = final RMSNorm_H(hidden)
slice_indices = slice(-logits_to_keep, None) if logits_to_keep is int else logits_to_keep
logits = lm_head(hidden[:, slice_indices, :])
```

## 6. Attention requirements

Variant: decoder causal self-attention with MLA-style low-rank projections, partial RoPE, optional backend dispatch, and standard autoregressive KV cache.

Shapes:

- Hidden input: `[B,S,H]`.
- Query after projection: `[B,heads,S,192]`, split as `[128 nope, 64 rope]`.
- KV A output: `[B,S,576]`, split as latent KV `[B,S,512]` and shared RoPE key `[B,S,64]`.
- KV B output after latent norm/projection: `[B,heads,S,256]`, split as key-nope `[B,heads,S,128]` and value `[B,heads,S,128]`.
- RoPE key slice is `[B,1,S,64]`, then expanded to `[B,heads,S,64]`.
- Native source cache per layer stores key `[B,heads,T,192]` and value `[B,heads,T,128]`.
- FlashAttention path pads value to `[B,heads,T,192]`, then crops attention output back to value dim 128 before the O projection.

Representative cache sizes in bf16 under source-expanded cache:

- Lite: per token per layer `(16 * 192 + 16 * 128) * 2 bytes = 10240 bytes`; all 27 layers are about 270 KiB/token/batch element.
- V2/V2.5: per token per layer `(128 * 192 + 128 * 128) * 2 bytes = 81920 bytes`; all 60 layers are about 4.7 MiB/token/batch element.

Inference: these cache estimates are source-parity expanded cache sizes. An optimized latent-cache MLA design could be much smaller, but it would be a graph/runtime rewrite beyond the current HF cache ABI.

Math order in source:

```text
RMSNorm -> Q low-rank/direct projection
RMSNorm -> KV A projection -> split latent KV and RoPE K
latent KV RMSNorm -> KV B projection -> split K_nope and V
complex RoPE on Q_rope and K_rope
expand K_rope across heads -> concatenate full Q/K
cache.update(full K, V)
attention backend(q, k, v, mask, scale=192^-0.5)
O projection over heads * v_head_dim
```

Masking and cache:

- `DeepseekV2Model.forward` always calls `create_causal_mask`, not a sliding-window mask.
- Default `position_ids` are `arange(S) + past_seen_tokens`, unsqueezed to `[1,S]`.
- Cached keys are post-RoPE and already concatenated `[nope, rope]`.
- There is no cross-attention and no encoder-decoder cache.

Backend compatibility:

- Source advertises FlashAttention, SDPA, flex attention, and generic attention backend support.
- A first DinoML fused attention provider can implement the source-expanded Q/K/V path and avoid only obvious materialized expansions. A later MLA provider should target latent KV cache and fused reconstruction as an optimization with its own parity tests.

## 7. Position encoding and custom math

DeepSeek RoPE generation returns complex frequencies, not separate cos/sin tensors:

```python
def deepseek_v2_freqs_cis(inv_freq, position_ids, attention_scaling):
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    return torch.polar(torch.ones_like(freqs), freqs) * attention_scaling
```

Apply RoPE:

```python
def apply_deepseek_v2_rope(q_pe, k_pe, freqs_cis):
    q_complex = view_as_complex(q_pe.float().reshape(*q_pe.shape[:-1], -1, 2))
    k_complex = view_as_complex(k_pe.float().reshape(*k_pe.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(1).to(q_complex.device)  # [B,1,S,rope_dim/2]
    q_out = view_as_real(q_complex * freqs).flatten(3).type_as(q_pe)
    k_out = view_as_real(k_complex * freqs).flatten(3).type_as(k_pe)
    return q_out, k_out
```

YaRN requirements from sampled configs:

- `rope_type` normalizes from legacy `"type": "yarn"`.
- `rope_theta=10000`, `factor=40`, `original_max_position_embeddings=4096`, `beta_fast=32`, `beta_slow=1`.
- V2/V2-Lite use `mscale=0.707`, `mscale_all_dim=0.707`; V2.5 uses `mscale=1.0`, `mscale_all_dim=1.0`.
- RoPE utility computes a correction range, mixes extrapolation and interpolation inverse frequencies, and derives an attention scaling factor. DinoML should either reuse an equivalent YaRN helper or reject these configs until YaRN parity is implemented.

Precompute opportunities:

- `inv_freq` and YaRN correction factors are static per config and device/dtype.
- `freqs_cis` depends on runtime `position_ids`, sequence length, and batch. Prefill computes `[B,S,32]` complex frequencies for `qk_rope_head_dim=64`; decode can compute only new positions.
- Position embeddings are shared across layers in source and should not be recomputed per layer.

## 8. Preprocessing and input packing

Text-only runtime contract:

- Required: `input_ids [B,S]` or `inputs_embeds [B,S,H]`, exactly one.
- Optional: `attention_mask`, `position_ids`, `past_key_values`, `use_cache`, `logits_to_keep`.
- If `inputs_embeds` is absent, source applies token embedding.
- If `position_ids` is absent, source derives it from cache length.
- If `use_cache` and no cache is supplied, source creates `DynamicCache(config)`.

Generation-controller behavior outside compiled graph:

- Sampled generation configs set BOS `100000`, EOS `100001`, `do_sample=true`, `temperature=0.3`, and `top_p=0.95`.
- Chat templates and tokenizer behavior are outside this model directory and can remain CPU/controller-side for first graph parity.
- `logits_to_keep` is source-supported and important for generation: use `1` or explicit indices to avoid a full prompt vocabulary projection.

No multimodal tensors, placeholder tokens, packed image grids, audio features, or `cu_seqlens` metadata are model-coupled for this family.

## 9. Graph rewrite / lowering opportunities

### Rewrite: DeepSeek MLA source-expanded attention region

Source pattern:

```text
Q low-rank/direct projection
KV A projection -> latent KV + shared K_rope
KV B projection -> per-head K_nope + V
complex RoPE -> expand K_rope -> cat full Q/K -> cache.update -> attention
```

Replacement pattern:

```text
DeepSeekMLAAttentionSourceParity(q_path, kv_a, kv_b, freqs_cis, cache, mask)
```

Preconditions:

- `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128` or provider declares exact configured dimensions.
- Cache ABI stores expanded full K and V as native source does.
- YaRN/default RoPE type is supported.
- Flash path value padding/cropping is preserved when using a backend requiring equal QK/V dims.

Shape equations:

- `q_width = heads * (qk_nope + qk_rope)`.
- `kv_a_width = kv_lora_rank + qk_rope`.
- `kv_b_width = heads * (qk_nope + v_head)`.
- `o_input_width = heads * v_head`.

Failure cases:

- Treating `head_dim` as attention score dim.
- Assuming GQA cache heads fewer than query heads.
- Caching latent KV while claiming source parity.

Parity test sketch:

- Compare Q parts, KV parts, RoPE outputs, cache tensors, attention output before O projection, and final attention block output for Lite and full shapes.

### Rewrite: latent-cache MLA optimization

Source pattern:

```text
cache.update(concat(k_nope, expanded_k_pe), value_states)
```

Replacement pattern:

```text
cache latent_kv and k_pe separately; reconstruct/fuse K_nope/V inside attention
```

Preconditions:

- Dedicated cache ABI records latent KV `[B,T,kv_lora_rank]` plus K_rope `[B,1,T,qk_rope]` or equivalent packed form.
- Attention kernel can apply `kv_b_proj` or a mathematically equivalent transformed query-side projection without materializing full expanded K/V.
- O projection and value path preserve source math within accepted tolerance.

Failure cases:

- This is not a source-level graph rewrite in current HF code; it changes cache tensors and kernel math.
- Quantized/offloaded weights need provider-visible provenance.

Parity test sketch:

- Start with no-cache prefill equality, then prefill+decode cache equality against source-expanded cache for several sequence lengths. Track memory separately from logits parity.

### Rewrite: group-limited MoE routing to sorted expert batches

Source pattern:

```text
router_probs = softmax(router_logits.float(), dim=-1)
if group_limited:
    group_scores = router_probs.view(M, n_group, -1).max(dim=-1)
    group_idx = topk(group_scores, topk_group)
    score_mask = scatter/expand group mask to experts
    tmp_scores = router_probs.masked_fill(~score_mask, 0.0)
    weights, expert_idx = topk(tmp_scores, top_k)
else:
    weights, expert_idx = topk(router_probs, top_k)
weights *= routed_scaling_factor
```

Replacement pattern:

```text
RouterTopKGrouped -> assignment sort/histogram -> grouped packed expert GEMM -> weighted scatter-add
```

Preconditions:

- Inference mode, no aux loss output required.
- `norm_topk_prob` remains ignored for DeepSeek source parity unless a future source revision uses it.
- Expert count is divisible by `n_group` for group-limited routing.
- Provider supports dynamic per-expert row counts and top-k=6.

Shape equations:

- Flat tokens `M = B * S`.
- Router logits `[M,E]`.
- Selected experts/weights `[M,6]`.
- Total routed rows before grouping: `6*M`.
- Expert gate/up: `[M_e,H] x [H,2*I_moe] -> [M_e,2*I_moe]`.
- Expert down: `[M_e,I_moe] x [I_moe,H] -> [M_e,H]`.

Weight transform:

- Source `gate_up_proj[e]` is an `F.linear` weight shaped `[2*I_moe,H]`.
- Source `down_proj[e]` is shaped `[H,I_moe]`.

Failure cases:

- Top-k tie determinism.
- Empty experts.
- Accidentally importing Qwen2-MoE shared-expert sigmoid gate.

Parity test sketch:

- Compare group scores, selected groups, selected experts, selected weights after scaling, per-expert outputs, and final scatter-add.

### Rewrite: shared expert dense branch

Source pattern:

```text
shared = down(silu(gate_proj(x)) * up_proj(x))
moe_out = routed_experts(x) + shared
```

Replacement pattern:

```text
fused shared gate/up GEMM -> SiLU*multiply -> down GEMM -> add routed output
```

Preconditions:

- `n_shared_experts is not None`.
- Shared intermediate equals `moe_intermediate_size * n_shared_experts`.
- No sigmoid gate is applied in DeepSeek-V2.

Failure cases:

- Confusing this with Qwen2-MoE's `shared_expert_gate`.

### Rewrite: dense first-layer MLP

Source pattern:

```text
layer_idx < first_k_dense_replace -> DeepseekV2MLP(intermediate_size)
```

Replacement pattern:

```text
fused dense SwiGLU MLP for early layers
```

Preconditions:

- Layer index is below `first_k_dense_replace`.
- Activation is `silu`; biases follow `mlp_bias`.

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

- Generation only needs last-token logits or a known subset.
- Loss/training path disabled.

## 10. Kernel fusion candidates

Highest priority:

- DeepSeek MLA source-parity attention block. Q/KV projection shapes are nonstandard and must be correct before any fused attention is trusted.
- YaRN complex RoPE + cache write. Official configs require YaRN and source stores post-RoPE expanded K.
- Native attention with unequal QK/V dims. Avoid generic fallbacks that pad/crop through extra memory unless the backend demands it.
- MoE routing with group-limited top-k. Full models use 160 experts, 8 groups, top 3 groups, top 6 experts, and scale 16.
- Packed expert gate/up GEMM + SiLU multiply + down GEMM. This dominates MoE runtime after attention.
- Weighted scatter-add for top-6 expert contributions.
- Last-token-only LM head.

Medium priority:

- Q-LoRA and KV-LoRA projection fusion: `linear -> RMSNorm -> linear` for Q and latent KV paths.
- Shared expert MLP fusion, separate from routed expert execution.
- Dense first-layer MLP fusion.
- Router softmax/top-k specialization for `E=64/160`, `K=6`.
- Latent-cache MLA optimization once source-expanded parity is established.
- Expert residency/offload planning. Full V2 has 160 experts per MoE layer and 59 MoE layers; this interacts directly with DinoML's GGUF/offload work.

Lower priority:

- Sequence classification head.
- Returning attention weights from optimized backends.
- Training losses, aux/router losses, gradient checkpointing, dropout.
- Tensor/expert/pipeline parallel execution. Preserve TP metadata early; implement after single-GPU parity.
- Remote-code-only fields not consumed by pinned in-library source.

## 11. Runtime staging plan

1. Parse `DeepseekV2Config`, including legacy RoPE normalization, MLA dimensions, dense-vs-MoE layer split, routed/shared expert fields, and generation defaults.
2. Load weights for embeddings, attention low-rank projections, RMSNorms, dense MLP, routed packed experts, shared experts, final norm, and LM head. Preserve packed expert layout.
3. Implement one-block parity for Lite-style direct Q attention without cache and dense layer-0 MLP.
4. Implement one-block parity for full V2 Q-LoRA attention and MoE routing, including group-limited routing.
5. Implement YaRN complex RoPE parity and source-expanded DynamicCache decode.
6. Run tiny/synthetic full prefill parity, then shape-load smoke for Lite and V2 configs.
7. Add source-expanded fused attention for prefill/decode with value padding/cropping behavior where needed.
8. Replace eager MoE loop with sorted-token grouped expert GEMM and weighted scatter-add.
9. Add last-token logits lowering and generation controller integration for raw token IDs.
10. Explore latent-cache MLA as a separate optimized provider/cache ABI after source parity is stable.

Initially stub/defer:

- Sampling/chat templates can run outside DinoML.
- Sequence classification can be deferred.
- Remote-code-only aux loss fields can be ignored for causal LM inference.
- Multi-GPU TP/EP/PP can be metadata-only.

## 12. Parity and validation plan

Custom op tests:

- RMSNorm over hidden, Q-LoRA rank 1536, and KV-LoRA rank 512 with fp32 variance.
- YaRN frequency generation for Lite/V2/V2.5 config parameters, including `mscale=0.707` and `mscale=1.0`.
- Complex RoPE apply on `[B,heads,S,64]` Q and `[B,1,S,64]` K.
- MLA projection split tests for Lite direct Q and full Q-LoRA paths.
- Attention source-expanded cache update: K shape `[B,heads,T,192]`, V shape `[B,heads,T,128]`, post-RoPE K.
- Flash-style value pad/crop parity for `qk_head_dim != v_head_dim`.
- Router greedy and group-limited routing, including group mask and scaling factor.
- Packed expert gate/up/down and shared expert branch.
- Scatter-add with top-6 contributions and empty experts.

Model tests:

- Single dense first layer parity for Lite and full synthetic configs.
- Single MoE layer parity with greedy routing (Lite) and group-limited routing (full).
- Two-layer tiny synthetic model covering dense layer 0 plus MoE layer 1.
- Prefill logits parity with `logits_to_keep=0`.
- Decode parity token by token with omitted `position_ids` and cache reuse.
- `logits_to_keep=1` parity against full logits sliced to final token.
- Config admission tests for remote-code `auto_map`, legacy `rope_scaling`, `q_lora_rank=None`, tiny-random dimensions, group-limited routing, and V2.5 mscale values.

Tolerance guidance:

- fp32 isolated ops: `rtol=1e-4`, `atol=1e-5`.
- bf16/fp16 block/logits: start with `rtol=2e-2`, `atol=2e-2`; tighten where math order is identical.
- Router selected expert indices should be exact on non-tie test data.
- Optimized MoE scatter and fused attention may need slightly looser logits tolerances due to accumulation order.

End-to-end:

- Raw token ID greedy or fixed-sampling controller tests after graph parity.
- Chat formatting should be separate tokenizer/controller parity, not part of the compiled model graph.

## 13. Performance probes

- Prefill tokens/sec by sequence length: 128, 2048, 8192, 32768, 163840 where memory allows.
- Decode tokens/sec by batch size and cache length for Lite and full V2.
- Source-expanded KV cache memory versus experimental latent-cache MLA memory.
- Attention backend comparison: eager source, SDPA/FlashAttention-compatible source-expanded path, DinoML fused MLA path.
- Value pad/crop overhead when using FlashAttention-style equal-dim kernels.
- Router/top-k latency for `E=64,K=6` and `E=160,K=6` with group-limited routing.
- Expert token distribution and grouped GEMM utilization for prefill versus decode.
- Shared expert branch cost versus routed expert cost.
- Weight-loading/offload probes for 160-expert full model: dense bf16 baseline, GGUF/dequant-before-GEMM exploratory path, future expert residency scheduling.
- Last-token LM head versus full prompt logits.
- RoPE generation bandwidth and benefit of per-position decode computation.

Benchmark observations: none collected. These are source/config-derived probes.

## 14. Skip/defer list

Safe to defer for first causal LM integration:

- Training, labels/loss, dropout, gradient checkpointing, aux/router loss fields, and `seq_aux`.
- Sequence classification head.
- Returning attentions from optimized attention.
- Beam search and speculative decoding; standard logits are enough initially.
- Tensor parallel, expert parallel, and pipeline parallel execution.
- Quantized weight formats beyond DinoML's separate GGUF/runtime-dequant admission.
- Remote-code-only behavior not present in pinned in-library source.
- Latent-cache MLA optimization, as long as source-expanded cache parity is implemented first.

Do not defer:

- YaRN RoPE for official sampled configs.
- `qk_nope_head_dim + qk_rope_head_dim` score dimension and `v_head_dim` output dimension distinction.
- Source cache shape correctness.
- Q-LoRA versus direct-Q config branch.
- Group-limited top-k routing and `routed_scaling_factor`.
- Shared expert dense branch without Qwen2-style sigmoid gate.
- Packed expert weight layout.

## 15. Final implementation checklist

- [ ] Parse `DeepseekV2Config` and normalize legacy `rope_scaling`/`rope_theta` to `rope_parameters`.
- [ ] Admit YaRN RoPE for sampled official configs or reject clearly until implemented.
- [ ] Load embeddings, MLA attention projections, RMSNorms, dense MLP, packed routed experts, shared experts, final norm, and LM head.
- [ ] Preserve packed expert weights: `gate_up_proj[E,2I,H]`, `down_proj[E,H,I]`.
- [ ] Implement DeepSeek RMSNorm with fp32 variance.
- [ ] Implement complex YaRN/default RoPE frequency generation and apply function.
- [ ] Implement Lite direct-Q and full Q-LoRA attention projection branches.
- [ ] Implement KV A/B latent projection path and split K-nope, K-rope, and V correctly.
- [ ] Implement source-expanded KV cache `[B,heads,T,192]` and `[B,heads,T,128]`.
- [ ] Implement causal attention with unequal QK/V dims and Flash-style pad/crop where required.
- [ ] Implement dense first-layer SwiGLU MLP.
- [ ] Implement router linear, fp32 softmax, greedy and group-limited top-6 routing, and routed scaling.
- [ ] Implement routed packed expert execution, route weighting, and scatter-add.
- [ ] Implement shared expert SwiGLU branch and add it to routed output.
- [ ] Implement final RMSNorm, `logits_to_keep`, and LM head.
- [ ] Add one-block Lite and full V2 parity tests.
- [ ] Add prefill/decode cache parity tests.
- [ ] Add config admission tests for tiny-random, Lite, V2, V2-Chat, V2.5, and remote-code metadata.
- [ ] Benchmark attention, KV memory, routing, grouped expert GEMMs, shared expert branch, and last-token logits.
- [ ] Stage latent-cache MLA as a separate optimization after source-expanded parity.
