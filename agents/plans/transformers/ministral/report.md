# Transformers family audit: `ministral`

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: source docstring and HF docs point at `mistralai/Ministral-8B-Instruct-2410`, but its current `config.json` says `model_type: "mistral"` and `architectures: ["MistralForCausalLM"]`. Treat that checkpoint as a compatibility trap for this `ministral` audit, not as proof that the `ministral` class owns the live weights.

Config source:
- Source defaults in `src/transformers/models/ministral/configuration_ministral.py`.
- Fetched snapshots saved beside this report:
  - `mistralai__Ministral-8B-Instruct-2410__config.json`
  - `mistralai__Ministral-3-3B-Base-2512__config.json`
  - `mistralai__Ministral-3-3B-Instruct-2512__config.json`
  - `mistralai__Ministral-3-8B-Base-2512__config.json`
  - `mistralai__Ministral-3-8B-Instruct-2512__config.json`
  - `mistralai__Ministral-3-14B-Base-2512__config.json`
  - `mistralai__Ministral-3-14B-Instruct-2512__config.json`
  - matching small `generation_config.json` snapshots for the instruct checkpoints.
  - `tokenizer_generation_summary.md` for tokenizer special-token and generation metadata.

Source files inspected:
- `X:/H/transformers/src/transformers/models/ministral/configuration_ministral.py`
- `X:/H/transformers/src/transformers/models/ministral/modeling_ministral.py`
- `X:/H/transformers/src/transformers/models/ministral/modular_ministral.py`
- `X:/H/transformers/src/transformers/cache_utils.py`
- `X:/H/transformers/src/transformers/masking_utils.py`

Source URLs:
- https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/ministral
- https://huggingface.co/docs/transformers/model_doc/ministral
- https://huggingface.co/mistralai/Ministral-8B-Instruct-2410

Any missing files or assumptions:
- `modeling_ministral.py` and `configuration_ministral.py` are generated from `modular_ministral.py`; future source edits should start in the modular file.
- `mistralai/Ministral-8B-Base-2410` returned 401 for `config.json`; report notes it as gated/unavailable. Access would resolve whether it also uses `model_type: mistral`.
- Official `Ministral-3-*` repos are `model_type: mistral3` multimodal wrappers with nested text config `model_type: ministral3`; they are out of scope for this `ministral` text-only report.
- HF docs say the architecture alternates 1 full attention layer followed by 3 sliding-window layers, but the inspected `MinistralConfig.__post_init__` only auto-fills all layers as sliding when `sliding_window` is not `None`. An alternating pattern must therefore come from explicit checkpoint `layer_types`, as in the 2410 `mistral` checkpoint.

## 2. High-level architecture

Primary runtime target: causal language modeling with `MinistralForCausalLM`.

Architecture: text-only decoder with token embedding, repeated pre-norm decoder blocks, final RMSNorm, and untied LM head by default.

Dataflow:

```text
tokenizer/input_ids -> token embedding -> decoder prefill/decode blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Stage decomposition:
- CPU/data pipeline: tokenization, chat template handling, `input_ids`, optional 2D padding mask.
- GPU/runtime prefill: embeddings, position IDs, RoPE tables, causal or sliding causal attention masks, all decoder layers, logits.
- GPU/runtime decode: one or more new tokens, per-layer KV cache update, same RoPE math using absolute positions.
- Generation controller: sampling, stopping on EOS, chat prompt formatting. This is not part of the neural graph.

Independently stageable pieces:
- Decoder block without cache.
- RoPE and mask generation.
- GQA attention with full causal mask.
- Sliding-window GQA attention and sliding cache.
- LM head with `logits_to_keep` last-token slicing.

## 3. Important config dimensions

Source default `MinistralConfig`:

| Field | Value | Provenance |
|---|---:|---|
| `model_type` | `ministral` | source default |
| `vocab_size` | 32000 | source default |
| `hidden_size` | 4096 | source default |
| `intermediate_size` | 14336 | source default |
| `num_hidden_layers` | 32 | source default |
| `num_attention_heads` | 32 | source default |
| `num_key_value_heads` | 8 | source default |
| `head_dim` | `None`, effective 128 | source default/inference from source |
| `hidden_act` | `silu` | source default |
| `max_position_embeddings` | 131072 | source default |
| `rms_norm_eps` | 1e-6 | source default |
| `rope_parameters` | required by generated model as dict with at least `rope_type`/`rope_theta`; docs describe optional | source behavior |
| `sliding_window` | 4096 | source default |
| `attention_dropout` | 0.0 | source default |
| `use_cache` | true | source default |
| `tie_word_embeddings` | false | source default |

Representative checkpoint/config sweep:

| Repo | Scope for this report | Top/model text type | Layers | Hidden | Heads/KV/Dim | MLP | Context/window | RoPE | Notes |
|---|---|---|---:|---:|---|---:|---|---|---|
| source defaults | in scope | `ministral` | 32 | 4096 | 32/8/effective 128 | 14336 | 131072/4096 | config-dependent | All layers auto become `sliding_attention` when no `layer_types` are supplied and `sliding_window` is set. |
| `mistralai/Ministral-8B-Instruct-2410` | route to `mistral` audit | `mistral` | 36 | 4096 | 32/8/128 | 12288 | 32768/32768 | `rope_theta=1e8` | Explicit repeating `full, sliding, sliding, sliding` layer pattern; `torch_dtype=bfloat16`, vocab 131072. |
| `mistralai/Ministral-3-3B-{Base,Instruct}-2512` | out of scope, `mistral3` | nested `ministral3` text | 26 | 3072 | 32/8/128 | 9216 | 262144/null | YaRN, theta 1e6, factor 16 | Multimodal wrapper with Pixtral vision tower; instruct adds FP8 quantization config. |
| `mistralai/Ministral-3-8B-{Base,Instruct}-2512` | out of scope, `mistral3` | nested `ministral3` text | 34 | 4096 | 32/8/128 | 14336 | 262144/null | YaRN, theta 1e6, factor 16 | Multimodal wrapper; not served by `modeling_ministral.py`. |
| `mistralai/Ministral-3-14B-{Base,Instruct}-2512` | out of scope, `mistral3` | nested `ministral3` text | 40 | 5120 | 32/8/128 | 16384 | 262144/null | YaRN, theta 1e9, factor 16 | Multimodal wrapper; instruct adds FP8 quantization config. |

## 3a. Family variation traps

- `head_dim` is explicit when present. Do not infer projection width from `hidden_size` alone; `q_proj` output is `num_attention_heads * head_dim`, and `o_proj` input is the same.
- GQA is default: `num_key_value_heads=8` and `num_attention_heads=32`, so KV heads repeat by factor 4 in eager attention.
- Attention projections have no bias in inspected source. MLP and LM head are also bias-free.
- `layer_types` controls full versus sliding attention per layer. Source defaults make all layers sliding when `sliding_window` is set; official 2410 checkpoint uses an alternating pattern but routes to `mistral`.
- `sliding_window=None` changes source-generated `layer_types` to all `full_attention`.
- `rope_parameters` is used by `MinistralRotaryEmbedding`; source expects `config.rope_parameters["rope_type"]`. A config that only has legacy `rope_theta` would need normalization before this class.
- Live official `Ministral-3-*` configs are not this model family. They include vision tower, projector, image token, FP8 quantization metadata, and nested `ministral3`; reject or route separately.
- `MinistralForCausalLM` declares tied weight keys, but `tie_word_embeddings` default is false. Preserve aliasing only when config enables tying or loaded weights are shared.
- `logits_to_keep` can be an int suffix count or tensor indices; first integration can require `0` or `1` and reject arbitrary index tensors.
- `attention_mask` may be a dict precomputed by generation; graph import should either own mask creation or admit a per-layer mask mapping ABI.
- No vision/audio/layout translation applies to the in-scope `ministral` source.

## 4. Operator coverage checklist

Tensor/layout ops:
- Token embedding lookup: `[B, T] -> [B, T, H]`.
- Shape/view ops for projections: `[B, T, H] -> [B, T, heads, head_dim] -> [B, heads, T, head_dim]`.
- Transpose, contiguous, reshape back to `[B, T, heads * head_dim]`.
- Slice logits: `hidden_states[:, slice_indices, :]`.
- Optional concat or rolling-window copy for KV cache updates.

Neural network primitives:
- Bias-free Linear:
  - `q_proj`: `H -> num_attention_heads * head_dim`
  - `k_proj`: `H -> num_key_value_heads * head_dim`
  - `v_proj`: `H -> num_key_value_heads * head_dim`
  - `o_proj`: `num_attention_heads * head_dim -> H`
  - `gate_proj`: `H -> intermediate_size`
  - `up_proj`: `H -> intermediate_size`
  - `down_proj`: `intermediate_size -> H`
  - `lm_head`: `H -> vocab_size`
- RMSNorm with fp32 variance and cast back to input dtype.
- SiLU and elementwise multiply for SwiGLU-style MLP.
- Residual adds.

Attention primitives:
- Causal self-attention.
- GQA/MQA-style KV repeat when `num_key_value_heads < num_attention_heads`.
- Sliding-window causal attention when layer type is `sliding_attention`.
- Dense full causal attention when layer type is `full_attention`.
- Softmax in fp32 then cast to query dtype in eager path.
- Dropout is present in code but disabled for inference.

Position/rotary ops:
- RoPE table generation from inverse frequencies.
- `rotate_half`, concat, broadcasted cos/sin multiply/add.
- Optional non-default RoPE through `ROPE_INIT_FUNCTIONS` if config requests it.

Generation/cache ops:
- DynamicCache/StaticCache per layer.
- Cache tensor ABI: `[B, num_key_value_heads, cached_T, head_dim]` before any repeat-to-query-head expansion.
- Sliding cache for sliding layers uses bounded length based on `sliding_window`.
- Cache reorder for beam search can be deferred for first greedy/sampling integration.

Quantized/packed weight metadata ops:
- None in native `ministral` source. FP8 metadata observed in `Ministral-3-*` instruct configs belongs to `mistral3` and should be audited there.

Preprocessing-coupled ops:
- Tokenizer/chat template only; no model-side multimodal stitch.

Distributed/tensor-parallel ops:
- Config declares TP plans for projections and LM head. DinoML can ignore for single-GPU first target and preserve as future sharding metadata.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers`:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(H -> A * D, bias=False)(x_norm).view(B,T,A,D).transpose(1,2)
k = Linear(H -> K * D, bias=False)(x_norm).view(B,T,K,D).transpose(1,2)
v = Linear(H -> K * D, bias=False)(x_norm).view(B,T,K,D).transpose(1,2)
q, k = RoPE(q, k, cos[position_ids], sin[position_ids])
k, v = cache.update(k, v, layer_idx) if cache enabled
attn = causal_or_sliding_GQA(q, k, v, mask, scale=D ** -0.5)
x = residual + Linear(A * D -> H, bias=False)(attn.transpose/reshape)

residual = x
x_norm = RMSNorm(x)
mlp = Linear(I -> H, bias=False)(silu(Linear(H -> I)(x_norm)) * Linear(H -> I)(x_norm))
x = residual + mlp
```

For source defaults: `A=32`, `K=8`, `D=128`, `H=4096`, `I=14336`. For `head_dim=None`, `D=H // A`.

Final causal LM path:

```text
x = Embedding(input_ids)
x = decoder_layers(x)
x = RMSNorm(x)
logits = Linear(H -> vocab_size, bias=False)(x[:, slice_indices, :])
```

Sequence classification, token classification, and question-answering heads are inherited generic heads. They are optional/deferred for the primary causal-LM target.

## 6. Attention requirements

Attention type:
- Decoder-only causal self-attention.
- Full causal and sliding-window causal variants selected per layer by `config.layer_types[i]`.
- GQA when `num_key_value_heads < num_attention_heads`; MHA if equal; MQA if KV heads equals 1.

Shapes:
- Query before attention: `[B, A, Tq, D]`.
- Key/value before repeat: `[B, K, Tkv, D]`.
- Eager repeat for matmul: `[B, A, Tkv, D]`, where `A = K * num_key_value_groups`.
- Scores: `[B, A, Tq, Tkv]`.
- Output before `o_proj`: `[B, Tq, A * D]`.

Masking:
- `MinistralModel.forward` builds both `full_attention` and `sliding_attention` masks unless the caller passes a dict.
- Sliding causal mask uses `create_sliding_window_causal_mask` and passes `local_size=sliding_window` to SDPA-style mask interfaces.
- Padding masks are 2D `[B, seen_tokens + Tq]` before mask expansion.
- Packed sequence and blockwise overlay hooks exist in the shared mask utility, but `ministral` does not add model-specific packed metadata in its forward signature.

Cache:
- `DynamicCache(config=self.config)` builds per-layer cache classes from `layer_types`.
- For sliding layers, dynamic cache storage is bounded to `[B, K, min(seq_len, sliding_window), D]` by the shared cache layer.
- Cached keys are stored after RoPE, because RoPE is applied before `past_key_values.update`.
- `position_ids` default to `arange(Tq) + past_key_values.get_seq_length()`.

Backend compatibility:
- Source advertises FlashAttention, SDPA, flex attention, and eager fallback through `ALL_ATTENTION_FUNCTIONS`.
- First DinoML path can implement dense eager math for full attention and a bounded sliding-window attention kernel/mask path. Optimized FlashAttention-style kernels should preserve scaling before softmax, mask addition before softmax, fp32 softmax, and output dtype cast.

## 7. Position encoding and custom math

Default RoPE inverse frequencies:

```python
def ministral_default_inv_freq(config):
    base = config.rope_parameters["rope_theta"]
    dim = config.head_dim or config.hidden_size // config.num_attention_heads
    return 1.0 / (base ** (arange(0, dim, 2, fp32) / dim))
```

Runtime cos/sin:

```python
freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Application:

```python
def apply_ministral_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat((-x2, x1), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Precompute opportunity:
- `inv_freq` is static for default RoPE.
- Cos/sin can be precomputed up to the admitted max sequence length for fixed RoPE. For dynamic/non-default RoPE, the shared `dynamic_rope_update` decorator may mutate/update buffers, so DinoML should gate on supported `rope_type`.

## 8. Preprocessing and input packing

In-scope `ministral` consumes text only:
- `input_ids`: `[B, T]` int token IDs.
- `attention_mask`: optional `[B, T_seen + T]`, values 1 for valid and 0 for padding before mask conversion.
- `position_ids`: optional `[B, T]`; if absent, generated from cache length.
- `inputs_embeds`: optional `[B, T, H]`; mutually exclusive with `input_ids`.

Tokenizer/generation observations from snapshots:
- `Ministral-8B-Instruct-2410`: tokenizer class `LlamaTokenizer`, BOS `<s>`, EOS `</s>`, no pad token in tokenizer config, chat template present, generation BOS=1/EOS=2.
- `Ministral-3-*` instruct snapshots: tokenizer class `TokenizersBackend`, BOS `<s>`, EOS `</s>`, PAD `<pad>`, generation pad token id 11. These are out-of-scope multimodal wrapper repos.

No model-coupled image/audio/video tensors, placeholder token scatter, `cu_seqlens`, or packed patch metadata exists in `modeling_ministral.py`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: bias-free Linear to GEMM

Source pattern:

```text
nn.Linear(in, out, bias=False)(x)
```

Replacement:

```text
GEMM row-major x [B*T, in] with weight.T [in, out] -> reshape [B,T,out]
```

Preconditions:
- Weight is dense and not source-coupled quantized metadata.
- Input is contiguous or lowered with explicit strides.
- Preserve dtype accumulation policy for fp16/bf16.

Failure cases:
- Tensor-parallel sharded weights unless sharding is admitted.
- Runtime-set `inputs_embeds` with unsupported strides.

Parity test sketch: compare each projection and LM head against PyTorch over static and decode `T=1` shapes.

### Rewrite: Q/K/V projections into packed GEMM

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
single GEMM with packed columns [Q | K | V], then split
```

Preconditions:
- All three projections share same input tensor and dtype.
- No bias.
- Packed output split sizes are `[A*D, K*D, K*D]`.
- Weight transform concatenates projection weights along output-feature dimension.

Failure cases:
- Different quantization/materialization policy per projection.
- Tensor-parallel partitioning already applied differently.

Parity test sketch: compare packed split outputs before RoPE.

### Rewrite: RMSNorm fusion

Source pattern:

```text
x_fp32 = x.to(fp32)
variance = mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
y = weight * (x_fp32 * rsqrt(variance + eps)).to(input_dtype)
```

Replacement: fused RMSNorm kernel.

Preconditions:
- Normalize last axis.
- Weight shape `[H]`.
- Epsilon from config.

Failure cases:
- Non-last-axis normalization or non-contiguous hidden dimension.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
two packed input GEMMs -> fused SiLU/multiply -> down GEMM
```

Preconditions:
- `hidden_act == "silu"` for this optimized path.
- `gate_proj` and `up_proj` are bias-free and share input.
- Split order `[gate, up]` is preserved.

Failure cases:
- Config uses another `hidden_act`; source allows any `ACT2FN` key.

### Rewrite: last-token-only logits

Source pattern: `lm_head(hidden_states[:, -1:, :])` when `logits_to_keep=1`.

Replacement: gather/slice last hidden token before GEMM.

Preconditions:
- No loss computation requiring all logits.
- Generation controller only needs next-token logits.

Failure cases:
- `logits_to_keep=0` all logits or tensor index selection.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm: used twice per block plus final norm; fp32 variance behavior matters.
- GQA attention with RoPE and KV cache: core prefill/decode cost and correctness surface.
- Sliding-window attention/cache: family-defining behavior when `layer_types` includes `sliding_attention`.
- Bias-free GEMM family coverage for Q/K/V/O/MLP/LM head.
- SwiGLU fused activation multiply between two GEMMs.

Medium priority:
- Packed QKV projection with split sizes `[A*D, K*D, K*D]`.
- RoPE application fused into Q/K projection output layout.
- Last-token-only LM head.
- Mask generation/canonicalization for full and sliding causal masks.

Lower priority:
- Generic inherited classification and QA heads.
- Tensor-parallel plans.
- Beam-search cache reorder.
- Dynamic/non-default RoPE variants beyond source defaults until a checkpoint requires them.

## 11. Runtime staging plan

Stage 1: parse `MinistralConfig`, normalize `rope_parameters`, reject live checkpoints whose `model_type` is not `ministral`.

Stage 2: load dense weights and run embedding, RMSNorm, bias-free Linear, RoPE, and MLP parity for one block without cache.

Stage 3: implement full causal GQA prefill for a small static sequence and compare block/model hidden states.

Stage 4: add sliding-window causal attention and per-layer `layer_types` dispatch.

Stage 5: add Dynamic/Static KV cache ABI for decode, including sliding-layer bounded cache behavior.

Stage 6: add `MinistralForCausalLM` logits path with `logits_to_keep=1`.

Stage 7: enable optimized attention/GEMM fusions and packed QKV after dense parity is stable.

Initially stub/defer:
- Classification/token/QA heads.
- Beam search and cache reorder.
- Tensor parallelism.
- Non-default RoPE scaling unless a concrete `ministral` config requires it.

## 12. Parity and validation plan

Recommended tolerances:
- fp32: `rtol=1e-4`, `atol=1e-5` for block-level tests.
- bf16/fp16: `rtol=5e-2`, `atol=5e-2` for logits initially; tighten by op after accumulation policy is fixed.

Tests:
- RMSNorm random tensor parity for `[B,T,H]`, including reduced precision input and fp32 variance.
- RoPE parity for generated `position_ids`, including decode offset after cache length.
- GQA eager attention parity with `A=32`, `K=8`, `D=128`, full causal mask.
- Sliding causal attention parity for `T > sliding_window` on a reduced toy config.
- Cache update parity: prefill then decode one token; verify cached K/V shapes and logits.
- One decoder layer parity on a small config with source-random weights.
- N-layer reduced model parity for both all-sliding and explicit alternating `layer_types`.
- `logits_to_keep=1` parity against full logits last-token slice.
- Config admission tests: reject `mistral`, `mistral3`, nested `ministral3`, and FP8 quantized Mistral3 snapshots for this family.

## 13. Performance probes

- Prefill tokens/sec sweep over `B` and `T`: 128, 512, 2048, 4096, and admitted long-context buckets.
- Decode tokens/sec with cache for `B=1,4,16`, comparing full versus sliding layers.
- KV cache memory versus sequence length for full and sliding layer mixes.
- Attention backend comparison: eager dense, SDPA/Flash-compatible full attention, sliding-window kernel.
- GEMM profile for projection shapes, especially `H -> A*D`, `H -> K*D`, `H -> I`, `I -> H`, and `H -> vocab`.
- Packed QKV projection speedup versus three separate GEMMs.
- Last-token LM head versus all-token LM head.
- RoPE generation/application cost as sequence length grows.

## 14. Skip/defer list

- Training, loss, gradient checkpointing.
- Sequence classification, token classification, and QA heads for first causal-LM target.
- Beam search, cache reorder, assisted/speculative decoding.
- Tensor-parallel and pipeline-parallel execution.
- Quantized/FP8 checkpoint loading; not native to this source basis.
- `mistral3` multimodal vision/projector/image-token path; separate audit required.
- Arbitrary `logits_to_keep` tensor indices.
- Non-default/dynamic RoPE variants unless an admitted `ministral` checkpoint uses them.

## 15. Final implementation checklist

- [ ] Add `ministral` config parser with `rope_parameters` normalization.
- [ ] Add admission rule rejecting `mistral`, `mistral3`, and nested `ministral3` configs for this family.
- [ ] Load dense embedding, decoder, norm, and LM head weights.
- [ ] Implement bias-free Linear lowering for projection/MLP/LM head shapes.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement default RoPE table generation and application.
- [ ] Implement GQA repeat or grouped attention without materializing repeat when optimized.
- [ ] Implement full causal prefill attention.
- [ ] Implement sliding-window causal attention and mask/cache rules.
- [ ] Implement KV cache ABI storing post-RoPE K/V as `[B, kv_heads, T, head_dim]`.
- [ ] Implement decode with cache offset-derived `position_ids`.
- [ ] Implement SwiGLU MLP and optional packed gate/up rewrite.
- [ ] Implement `logits_to_keep=1` LM head fast path.
- [ ] Add one-block and reduced-model parity tests.
- [ ] Add prefill/decode logits parity tests.
- [ ] Benchmark GEMM, attention, RoPE, cache memory, and last-token logits.
