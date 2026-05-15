# Transformers family audit: glm4_moe_lite

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: zai-org/GLM-4.7-Flash for the primary open production config
Config source: HF config.json files listed below plus local configuration_glm4_moe_lite.py defaults
Primary runtime target: Glm4MoeLiteForCausalLM, text-only autoregressive generation
DinoML assumptions: inference-only first, CUDA GPU target, prefill/decode parity before optimized MoE/attention providers
```

Source files inspected:

- `transformers/src/transformers/models/glm4_moe_lite/configuration_glm4_moe_lite.py`
- `transformers/src/transformers/models/glm4_moe_lite/modeling_glm4_moe_lite.py`
- `transformers/src/transformers/models/glm4_moe_lite/modular_glm4_moe_lite.py`
- `transformers/src/transformers/integrations/moe.py`
- `transformers/src/transformers/integrations/sdpa_attention.py`
- `transformers/src/transformers/configuration_utils.py`
- `transformers/src/transformers/modeling_rope_utils.py`

Authoritative source note: `configuration_glm4_moe_lite.py` and `modeling_glm4_moe_lite.py` are generated from `modular_glm4_moe_lite.py`; future source edits should target the modular file, but DinoML should audit the generated modeling file because it is the concrete runtime implementation.

Representative configs inspected:

- Official BF16: [zai-org/GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash/raw/main/config.json), model repo is ungated/public.
- Tiny/random smoke config: [tiny-random/glm-4-moe-lite](https://huggingface.co/tiny-random/glm-4-moe-lite/raw/main/config.json), public.
- Pruned/compressed variant: [cerebras/GLM-4.7-Flash-REAP-23B-A3B](https://huggingface.co/cerebras/GLM-4.7-Flash-REAP-23B-A3B/raw/main/config.json), public.
- Community FP8: [marksverdhei/GLM-4.7-Flash-FP8](https://huggingface.co/marksverdhei/GLM-4.7-Flash-FP8/raw/main/config.json), public.
- Community NVFP4: [GadflyII/GLM-4.7-Flash-NVFP4](https://huggingface.co/GadflyII/GLM-4.7-Flash-NVFP4/raw/main/config.json), public.
- Small normalized snapshot: `agents/plans/transformers/glm4_moe_lite/config_sweep_snapshot.json`.

No gated config was needed for this audit. The official model card reports the model as GLM-4.7-Flash, a 30B-A3B text-generation MoE model with Transformers/safetensors metadata and MIT license; parameter size and license are repo metadata, not source-code facts.

## 2. High-level architecture

`glm4_moe_lite` is a text-only decoder-only MoE language model. The primary forward path is:

```text
tokenizer/chat template -> input_ids/attention_mask
-> token embedding
-> repeated causal decoder layers with MLA-style low-rank Q/KV projections, RoPE, causal attention, dense-or-MoE MLP
-> final RMSNorm
-> lm_head
-> logits/sampling controller
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, left padding, special-token control, generation stopping IDs.
- GPU prefill: embed full prompt, build causal mask, compute RoPE cos/sin, run all decoder layers, produce logits and per-layer KV cache.
- GPU decode: one or more new tokens, update position IDs from cache length, run attention against cached K/V, compute last-token logits.
- Independently stageable providers: RMSNorm, dense GEMMs, RoPE, causal attention, MoE routing, grouped expert GEMMs, final logits.

Only `Glm4MoeLiteModel` and `Glm4MoeLiteForCausalLM` are implemented. There are no encoder, multimodal, classification, or sequence-classification heads in this family. Training loss is present through generic `loss_function` when labels are provided, but it is deferred for DinoML inference parity.

## 3. Important config dimensions

Primary production config values below come from `zai-org/GLM-4.7-Flash/config.json`; effective defaults come from `Glm4MoeLiteConfig` when omitted.

| Field | Production value | Source/runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 154880 | Token embedding rows and LM head rows |
| `hidden_size` | 2048 | Residual stream width |
| `num_hidden_layers` | 47 | Decoder layer count |
| `num_attention_heads` | 20 | Query/output attention heads |
| `num_key_value_heads` | 20 | KV heads; no GQA expansion for production, but source supports `num_attention_heads / num_key_value_heads` |
| `q_lora_rank` | 768 | Query low-rank first projection width; `None` switches to direct Q projection |
| `kv_lora_rank` | 512 | Compressed KV width before KV expansion |
| `qk_nope_head_dim` | 192 | Non-rotary Q/K head dim |
| `qk_rope_head_dim` | 64 | Rotary Q/K head dim |
| Effective `qk_head_dim` | 256 | Source computes `qk_nope_head_dim + qk_rope_head_dim` |
| `v_head_dim` | 256 | Value head dim and attention output per head |
| Attention output width | 5120 | `num_heads * v_head_dim`; projected back to 2048 |
| `intermediate_size` | 10240 | Dense MLP hidden width |
| `moe_intermediate_size` | 1536 | Per-routed-expert FFN hidden width |
| `n_routed_experts` | 64 | Routed expert count |
| `n_shared_experts` | 1 | Shared expert multiplier; shared MLP width = 1536 |
| `num_experts_per_tok` | 4 | Top-k routed experts per token |
| `n_group`, `topk_group` | 1, 1 | Grouped routing selection; grouping is source-visible even if degenerate here |
| `routed_scaling_factor` | 1.8 | Multiplies normalized top-k weights |
| `norm_topk_prob` | true | Normalize selected expert probabilities before scaling |
| `hidden_act` | `silu` | SwiGLU/SiLU gate activation |
| `max_position_embeddings` | 202752 | RoPE/cache maximum |
| `rope_theta` | 1000000 | Legacy config field standardized into `rope_parameters.rope_theta` |
| `rope_interleave` | default true | Interleaved RoPE transform path |
| `attention_bias` | false | Bias on `q_a_proj`, `kv_a_proj_with_mqa`, and `o_proj`; Q/KV second projections are biasless |
| `attention_dropout` | 0.0 | Inference dropout is 0 |
| `use_cache` | true | Dynamic KV cache enabled |
| `tie_word_embeddings` | false | Source declares tied-weight key but production config uses separate LM head |
| `dtype` | bfloat16 | Checkpoint metadata; source upcasts selected math to fp32 |

Representative checkpoint sweep:

| Model | H | Layers | Heads/KV | Q rank | KV rank | QK dims | V dim | Dense/MoE FFN | Experts/top-k | Notable variation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| `zai-org/GLM-4.7-Flash` | 2048 | 47 | 20/20 | 768 | 512 | 192+64 | 256 | 10240/1536 | 64/4 | Official BF16; legacy RoPE keys |
| `tiny-random/glm-4-moe-lite` | 8 | 2 | 4/4 | 32 | 384 | 64+192 | 64 | 32/32 | 64/4 | Shape smoke only; hidden size is not representative |
| `cerebras/GLM-4.7-Flash-REAP-23B-A3B` | 2048 | 47 | 20/20 | 768 | 512 | 192+64 | 256 | 10240/1536 | 48/4 | Fewer routed experts; no nextn layers in config metadata |
| `marksverdhei/GLM-4.7-Flash-FP8` | 2048 | 47 | 20/20 | 768 | 512 | 192+64 | 256 | 10240/1536 | 64/4 | FP8 serialized weights; ignore `lm_head` |
| `GadflyII/GLM-4.7-Flash-NVFP4` | 2048 | 47 | 20/20 | 768 | 512 | 192+64 | 256 | 10240/1536 | 64/4 | compressed-tensors NVFP4; ignores embedding/gate/self-attn/lm_head |

## 3a. Family variation traps

- Do not infer attention dimensions from `hidden_size`. Query/key width is `num_heads * (qk_nope_head_dim + qk_rope_head_dim)` and value/output-attention width is `num_heads * v_head_dim`; production uses 5120-wide attention output before `o_proj`.
- `head_dim` in `attribute_map` maps to `qk_rope_head_dim`, not the full QK head dimension. Some configs carry both `head_dim` and `qk_head_dim`; source computes `qk_head_dim` from nope+rope dims.
- `q_lora_rank is None` changes Q projection structure from two GEMMs plus RMSNorm to one direct GEMM.
- `attention_bias` affects only selected projections, not all linear layers.
- `mlp_layer_types` controls dense versus sparse layer bodies. If omitted, source defaults to one dense layer then all sparse layers.
- Configs include historical `first_k_dense_replace`, `topk_method`, `num_nextn_predict_layers`, `rope_scaling`, and sometimes `partial_rotary_factor`/`rope_theta`. The inspected modeling source does not read `first_k_dense_replace`, `topk_method`, or `num_nextn_predict_layers`; DinoML should treat them as metadata unless a separate remote/runtime path is audited.
- RoPE parameters are standardized by `PreTrainedConfig`/`RotaryEmbeddingConfigMixin`; current official configs use legacy `rope_theta` plus `rope_scaling: null` rather than explicit `rope_parameters`.
- `rope_interleave=True` changes the layout math around RoPE and introduces view/transpose/reshape before rotation.
- FlashAttention requested with `qk_head_dim != v_head_dim` pads values up to QK width and slices attention output back. Production has `qk_head_dim == v_head_dim == 256`, so this branch is not active for the official config but remains source-required for variants.
- MoE source has eager, `batched_mm`, `grouped_mm`, and `sonicmoe` expert dispatch via `config._experts_implementation`. DinoML should choose an explicit provider strategy rather than assuming the eager Python loop.
- Quantized configs advertise FP8/NVFP4/compressed-tensors formats. The inspected modeling source does not implement those packed formats; they are weight-loading/provider contracts.
- Tokenizer contains multimodal-looking special tokens, but this model source has no image/audio/video branch. Treat those as tokenizer vocabulary only for this report.
- No NCHW/NHWC layout concern exists in the neural graph. Layout-sensitive axes are sequence/head/channel axes in `view`, `transpose`, `split`, `cat`, `topk`, and attention masks; protect them from generic layout translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token embedding lookup: `[B, S] -> [B, S, 2048]`.
- `arange`, `unsqueeze`, broadcast/add for position IDs when absent.
- `view`/`reshape`, `transpose`, `contiguous`, `split`, `cat`, `expand`, `pad` for attention tensor plumbing.
- `slice`/advanced slice for `logits_to_keep`: final hidden states `[:, slice_indices, :]`.
- `masked_fill`, `scatter_`, `gather`, `one_hot`, `where`, `nonzero`, `index_add_` for eager MoE path; optimized MoE can replace much of this with routing metadata plus grouped GEMM.
- `topk`, `sort`, `histc`, `cumsum`, `permute`, `repeat_interleave` or equivalent for optimized expert routing/grouped GEMM.

Neural network primitives:

- RMSNorm over last dim: hidden 2048, q rank 768, kv rank 512.
- Dense GEMM/Linear:
  - token embedding and LM head `2048 -> 154880` with no bias.
  - Q low-rank `2048 -> 768` optional bias, RMSNorm, then `768 -> 20 * 256 = 5120` no bias.
  - Direct Q alternate `2048 -> 5120` no bias when `q_lora_rank=None`.
  - KV low-rank plus rotary `2048 -> 512 + 64 = 576` optional bias.
  - KV expansion `512 -> 20 * (192 + 256) = 8960` no bias.
  - Attention output `5120 -> 2048` optional bias.
  - Dense MLP `2048 -> 10240`, `2048 -> 10240`, `10240 -> 2048`, no biases.
  - Shared expert MLP `2048 -> 1536 * n_shared_experts`, `2048 -> 1536 * n_shared_experts`, down to 2048.
  - Routed expert packed `gate_up_proj`: `[E, 2 * 1536, 2048]`; `down_proj`: `[E, 2048, 1536]`.
- SiLU activation and gated multiply for dense/shared/routed MLP.
- Residual adds after attention and MLP.

Attention primitives:

- Causal self-attention with dynamic cache.
- MLA-style projection decomposition: Q has low-rank path; K/V use compressed KV plus separate rotary K component.
- MHA/GQA-compatible interface. Production is MHA (`20/20`) but source supports repeat-KV when KV heads are fewer.
- Eager attention: QK matmul, mask add, fp32 softmax, dropout, AV matmul.
- SDPA/Flash/Flex backend dispatch via `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:

- RoPE frequency generation from standardized `rope_parameters`.
- Interleaved and non-interleaved RoPE application.
- Optional YaRN-style attention scaling when non-default RoPE has `factor`/`mscale_all_dim`.

Generation/cache ops:

- Dynamic KV cache update per layer after RoPE and K/V construction.
- Position ID offset from `past_key_values.get_seq_length()`.
- Causal mask creation from attention mask, position IDs, and cache state.
- Last-token-only logits through `logits_to_keep`.

Quantized/packed weight metadata ops:

- No source-native quantized math in `modeling_glm4_moe_lite.py`.
- FP8/NVFP4/compressed-tensors checkpoints require a separate loader/dequant/provider policy and dense fallback. Community configs may ignore modules such as `lm_head`, embeddings, router gate, or self-attention.

Distributed/tensor-parallel metadata:

- Config declares TP plans for Q/KV/o projections, dense MLP, packed expert weights, and LM head. DinoML single-GPU parity can ignore TP first, but loader should not confuse packed expert layout with sharded layout.

## 5. Layer/block breakdown

Model prologue:

```text
input_ids [B,S] -> embed_tokens -> hidden [B,S,2048]
if position_ids absent: arange(S) + cached_length -> [1,S]
causal_mask = create_causal_mask(...)
cos,sin = rotary_emb(hidden, position_ids) -> [B,S,64]
```

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm_2048(x)

if q_lora_rank is not None:
  q = Linear(2048 -> 768, bias=attention_bias)
  q = RMSNorm_768(q)
  q = Linear(768 -> 5120, bias=False)
else:
  q = Linear(2048 -> 5120, bias=False)
q = view [B,S,20,256] -> transpose [B,20,S,256]
q_pass,q_rot = split [192,64]

compressed_kv = Linear(2048 -> 576, bias=attention_bias)
k_pass_seed,k_rot = split [512,64]
k_pass_value = RMSNorm_512(k_pass_seed)
k_pass_value = Linear(512 -> 8960, bias=False)
k_pass_value = view [B,S,20,448] -> transpose [B,20,S,448]
k_pass,value = split [192,256]
k_rot = view [B,1,S,64]
q_rot,k_rot = RoPE(q_rot,k_rot,cos,sin)
k_rot = expand to [B,20,S,64]
query = cat(q_pass,q_rot) -> [B,20,S,256]
key = cat(k_pass,k_rot) -> [B,20,S,256]
key,value = cache.update(key,value,layer_idx) when cache is enabled
attn = causal_attention(query,key,value,mask,scale=1/sqrt(256))
attn = reshape [B,S,5120]
attn = Linear(5120 -> 2048, bias=attention_bias)
x = residual + attn

residual = x
x_norm = RMSNorm_2048(x)
if mlp_layer_types[layer_idx] == "sparse":
  router_logits = Linear(2048 -> E, fp32)
  route top-k experts and weights
  routed = grouped/loop expert SwiGLU MLP per token
  shared = SwiGLU MLP(2048 -> 1536*n_shared_experts -> 2048)
  mlp = routed + shared
else:
  mlp = SwiGLU MLP(2048 -> 10240 -> 2048)
x = residual + mlp
```

Model epilogue:

```text
x = RMSNorm_2048(x)
logits = Linear(2048 -> vocab_size, bias=False) over requested final positions
```

## 6. Attention requirements

Attention type: causal decoder self-attention. There is no cross-attention.

Head contract:

- Query heads: `num_attention_heads`.
- KV heads: `num_key_value_heads`; source computes `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- Production: 20 query heads, 20 KV heads, 256 QK dim, 256 V dim.
- General source: Q/K head dim can differ from V head dim. FlashAttention path pads V to QK width and slices back when needed.

Masking:

- `create_causal_mask` builds a causal mask from config, input embeddings, optional padding attention mask, cache, and position IDs.
- Eager path adds mask to attention scores before softmax.
- SDPA path may use `is_causal` when query length > 1 and no explicit mask is passed.
- Decode single-token path should not blindly set SDPA `is_causal=True`; source disables that through shape logic.

Cache:

- Per-layer cache stores full `key_states [B, KV_heads, T, qk_head_dim]` and `value_states [B, KV_heads, T, v_head_dim]` after RoPE and concatenation.
- With production MHA, cached K/V are `[B,20,T,256]` each.
- For GQA variants, cache stores KV heads before repeat expansion; eager/SDPA repeats or enables GQA at attention execution.
- Cached keys are stored after RoPE.

Backend compatibility:

- Eager attention is the semantic fallback and returns attention weights.
- SDPA is source-supported and returns no attention weights.
- Flash/Flex are source-supported through Transformers attention interface. DinoML can initially implement eager-equivalent dense causal attention, then replace with a fused causal attention provider.

## 7. Position encoding and custom math

RoPE frequencies are computed in fp32 from `rope_parameters.rope_theta` and a rotary dimension derived from `head_dim`/`qk_rope_head_dim` plus `partial_rotary_factor`. Cos/sin are generated from `position_ids` and cast back to the hidden dtype.

Non-interleaved RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

def rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Interleaved RoPE first reshapes the last dim into pairs and swaps pair axes before the same rotate-half formula:

```python
q = q.view(B, H, S, D // 2, 2).transpose(4, 3).reshape(B, H, S, D)
k = k.view(B, H, S, D // 2, 2).transpose(4, 3).reshape(B, H, S, D)
q, k = rope(q, k, cos, sin)
```

The official production config uses default RoPE with `rope_theta=1000000` and `partial_rotary_factor=1.0`. Non-default RoPE variants are source-supported through `ROPE_INIT_FUNCTIONS`; if `rope_type != "default"` and `mscale_all_dim` is set, attention scaling is multiplied by `yarn_get_mscale(factor, mscale_all_dim) ** 2`.

Precompute opportunity: `inv_freq` is static per config; cos/sin can be cached per position bucket for prefill/decode, but dynamic position IDs and cache offsets must remain explicit.

## 8. Preprocessing and input packing

The model source consumes text token IDs only:

- `input_ids [B,S]` or `inputs_embeds [B,S,2048]`, exactly one required.
- `attention_mask` is optional and feeds causal mask construction.
- `position_ids` is optional; if absent, source constructs monotonically increasing positions offset by cache length.

Official tokenizer config observations:

- `tokenizer_class`: `PreTrainedTokenizer`.
- `model_max_length`: 128000, while model config max positions are 202752.
- `padding_side`: left.
- Pad/eos token is `<|endoftext|>` id 154820.
- Generation EOS list from config/generation config: `[154820, 154827, 154829]`.
- Chat/template, reasoning/tool-parser behavior, and special tokens are generation-controller/tokenizer ABI, not neural graph operators.

There is no processor, image/audio/video tensor, placeholder scatter, or multimodal embedding stitch in this family source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MLA Q projection canonicalization

Source pattern:

```text
Linear(H -> q_lora_rank, optional bias) -> RMSNorm(q_lora_rank) -> Linear(q_lora_rank -> heads*qk_head_dim)
```

Replacement:

```text
two explicit GEMMs with intervening RMSNorm
```

Preconditions:

- `q_lora_rank is not None`.
- Preserve RMSNorm in fp32 accumulation.
- Do not fold the two linear layers across RMSNorm.

Failure cases: `q_lora_rank=None` uses direct `Linear(H -> heads*qk_head_dim)`.

Parity test sketch: compare q projection output before split/reshape for random bf16/fp32 tensors.

### Rewrite: packed KV projection split

Source pattern:

```text
Linear(H -> kv_lora_rank + qk_rope_head_dim) -> split([kv_lora_rank, qk_rope_head_dim])
```

Replacement:

```text
single GEMM with two logical consumers, or two pre-split views of one GEMM output
```

Preconditions:

- Preserve output order: compressed KV rank first, rotary K tail second.
- `k_rot` is one-head `[B,1,S,qk_rope_head_dim]` then expanded across heads after RoPE.

Failure cases: any checkpoint with packed weight order different from source should be rejected.

### Rewrite: KV expansion packed split

Source pattern:

```text
RMSNorm(kv_lora_rank) -> Linear(kv_lora_rank -> heads*(qk_nope_head_dim+v_head_dim))
-> view/transpose -> split([qk_nope_head_dim, v_head_dim])
```

Replacement:

```text
GEMM -> structured split into K-nope and V head tensors
```

Preconditions: exact split order is `[k_pass, value]`; value width may differ from Q/K width.

### Rewrite: dense SwiGLU MLP

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
dual GEMM -> fused SiLU-multiply -> GEMM
```

Preconditions: `hidden_act == "silu"`, no projection biases in source MLP, static known intermediate size.

Parity test sketch: random hidden states through dense layer 0.

### Rewrite: routed expert eager loop -> grouped expert GEMM

Source pattern:

```text
router topk -> one_hot/where per expert -> per-expert two GEMMs -> weighted index_add
```

Replacement:

```text
topk routing -> token/expert pair list sorted by expert -> grouped GEMM gate_up -> fused SiLU-mul -> grouped GEMM down -> weighted per-token reduce
```

Preconditions:

- Expert weight layout: `gate_up_proj[E, 2*I, H]` with chunk order `[gate, up]`; `down_proj[E, H, I]`.
- Top-k count fixed by `num_experts_per_tok`.
- Preserve router sigmoid, correction bias, group mask, `topk(sorted=False)` semantics, optional top-k normalization, and routed scaling.
- For first integration, admission can require `n_group=topk_group=1`.

Failure cases: expert parallel sentinels, non-default expert implementations, quantized packed experts without loader support.

### Rewrite: final logits slicing

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
last-token or selected-token GEMM only
```

Preconditions: `logits_to_keep` is integer or static index tensor known at compile/run boundary.

Failure cases: callers requesting all logits for long prefill or arbitrary dynamic index tensors.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over 2048/768/512 with fp32 accumulation. It appears before every attention/MLP sublayer and inside MLA projections.
- Dense GEMM coverage for non-square attention widths, especially `2048 -> 768`, `768 -> 5120`, `2048 -> 576`, `512 -> 8960`, `5120 -> 2048`, and LM head.
- RoPE plus Q/K split/cat plumbing. Interleaved RoPE has awkward view/transpose/reshape overhead that should be fused or avoided with a layout-aware kernel.
- Causal attention prefill/decode with KV cache, preserving `qk_head_dim` and `v_head_dim` separation.
- MoE routing and grouped expert GEMM. The eager Python loop is not viable for production; route/topk/sort/grouped-GEMM/reduce is the core performance path.

Medium priority:

- SwiGLU fusion for dense/shared/routed MLPs.
- Router pipeline fusion: fp32 linear, sigmoid, correction bias, group topk, expert topk, gather, normalize, scale.
- Last-token-only logits GEMM to avoid full `[B,S,V]` projection during decode/prefill when not needed.
- Quantized weight materialization/dequant providers for FP8/NVFP4 variants.

Lower priority:

- Attention backend variants beyond dense causal and SDPA-like semantics, such as Flex/Flash-specific masks.
- Tensor-parallel execution plans.
- Output attentions parity; SDPA source does not support returning attention weights.

## 11. Runtime staging plan

Stage 1: config and weight metadata admission.

- Parse `Glm4MoeLiteConfig`, standardize legacy RoPE keys into explicit `rope_parameters`.
- Reject unsupported `model_type`, unknown expert implementations, unsupported quantization configs, and unsupported `mlp_layer_types`.
- Load dense BF16 official/tiny weights first; quantized checkpoints are deferred to provider-specific loading.

Stage 2: one-block dense-path parity.

- Implement embedding, RMSNorm, MLA projections, RoPE, causal dense attention without cache, dense MLP, residuals.
- Use a tiny/random config or synthetic one-layer config.

Stage 3: prefill parity for production BF16.

- Add all layers, causal mask, final RMSNorm, logits slicing.
- Initially allow eager/reference MoE for correctness only on tiny sizes; production should be guarded off until grouped expert path exists.

Stage 4: optimized MoE provider.

- Implement router/topk/grouped expert GEMM/reduce with source-exact weight layout and normalization.
- Validate dense layer 0 plus sparse layers separately.

Stage 5: decode with KV cache.

- Add dynamic cache ABI with per-layer K/V tensors after RoPE.
- Validate position offset and single-token decode logits.

Stage 6: optimized attention and logits.

- Replace dense attention fallback with fused causal attention provider for prefill/decode.
- Add last-token logits and selected-token logits path.

Stage 7: quantized checkpoints and production scheduling.

- Add explicit provider contracts for FP8/NVFP4/compressed-tensors or reject with clear messages.
- Add batching, cache paging/offload, and TP only after single-GPU BF16 parity is stable.

## 12. Parity and validation plan

- Config parity: load official config and assert effective `qk_head_dim=256`, standardized RoPE dict, layer-type pattern, and ignored historical metadata classification.
- RoPE parity: compare interleaved and non-interleaved functions on random `[B,H,S,64]` q/k tensors and nonzero position IDs.
- RMSNorm parity: fp32, bf16, fp16 random inputs; tolerance fp32 `1e-5`, bf16/fp16 `2e-2` relative/absolute depending on accumulation.
- Attention projection parity: compare q/k/v tensors before attention for random hidden states, including `q_lora_rank=None` synthetic config.
- Attention parity: eager causal attention prefill and decode with cache; test rectangular `[Q=1, K=T]` decode.
- MoE router parity: check top-k indices/weights for source router, including group mask and normalization.
- Expert parity: tiny expert weights, compare eager loop and grouped-GEMM replacement; include duplicate token expert assignments and empty experts.
- Single decoder layer parity: dense layer and sparse layer separately.
- Full tiny model parity: `tiny-random/glm-4-moe-lite` logits for a short prompt, with and without cache.
- Production smoke: official config shape-only compile/admission plus selected synthetic BF16 weight shard if full weights are too large.
- End-to-end text parity: fixed prompt, greedy decode for a few tokens after BF16 path is available.

## 13. Performance probes

- Prefill throughput by sequence length: 1k, 4k, 16k, 64k, 128k tokens if memory allows.
- Decode tokens/sec by batch size and cache length.
- KV cache memory: production per token per layer is approximately `2 * 20 * 256` BF16 values = 20 KiB/layer/token, about 940 KiB/token across 47 layers before allocator overhead.
- Router latency and top-k throughput for `[B*S, 64]` logits.
- Grouped expert GEMM throughput by tokens per expert distribution; include worst-case skew and uniform routing.
- Dense/shared MLP GEMM throughput and SwiGLU fusion benefit.
- Attention backend comparison: eager/reference, SDPA-like, fused prefill, fused decode.
- LM head probe: full logits versus last-token-only logits.
- Quantized load/dequant probe for FP8/NVFP4 variants once providers exist.

Benchmark facts above are proposed probes, not measured results.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing, and output attentions.
- Beam search and advanced generation controllers beyond greedy/sampling inputs/outputs.
- Tensor parallel and pipeline parallel execution, despite config metadata.
- `sonicmoe` and PyTorch-specific `grouped_mm` dispatch exactness; DinoML should implement its own provider or a bounded equivalent.
- FP8/NVFP4/compressed-tensors checkpoints until explicit loader/dequant/provider contracts exist.
- Speculative decoding / `num_nextn_predict_layers`; config metadata exists but inspected source has no next-token prediction head.
- Tool/reasoning parsers and chat template policies as neural graph work; keep in tokenizer/generation layer.
- Multimodal special token behavior; no model branch consumes image/audio/video tensors in this family.

## 15. Final implementation checklist

- [ ] Parse `Glm4MoeLiteConfig` and standardize legacy RoPE fields.
- [ ] Add source-basis admission for ignored historical fields: `first_k_dense_replace`, `topk_method`, `num_nextn_predict_layers`.
- [ ] Load dense BF16 weights with expert layout checks.
- [ ] Implement embedding and final LM head, including `tie_word_embeddings=false` default.
- [ ] Implement RMSNorm with fp32 accumulation.
- [ ] Implement MLA Q/KV projection graph and exact split/cat order.
- [ ] Implement interleaved and non-interleaved RoPE.
- [ ] Implement causal attention prefill fallback.
- [ ] Implement KV cache update/read ABI with post-RoPE K storage.
- [ ] Implement decode attention against cache.
- [ ] Implement dense SwiGLU MLP.
- [ ] Implement MoE router: sigmoid, correction bias, group topk, expert topk, gather, normalize, scale.
- [ ] Implement grouped expert GEMM provider for `gate_up_proj[E,2I,H]` and `down_proj[E,H,I]`.
- [ ] Add shared expert MLP and routed-plus-shared combine.
- [ ] Add last-token/selective logits lowering.
- [ ] Add tiny config one-layer and full-model parity tests.
- [ ] Add official-config shape/admission tests.
- [ ] Add prefill and decode logits parity tests.
- [ ] Add MoE routing and grouped expert parity tests.
- [ ] Add performance probes for attention, MoE, LM head, and KV memory.
- [ ] Add explicit reject/defer path for FP8/NVFP4/compressed-tensors until provider support lands.
