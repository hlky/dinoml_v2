# HunYuanDenseV1 Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: primary native checkpoints are `tencent/Hunyuan-0.5B-Pretrain`, `tencent/Hunyuan-1.8B-Instruct`, `tencent/Hunyuan-4B-Instruct`, `tencent/Hunyuan-7B-Instruct`, and `tencent/Hunyuan-MT-7B`.

Config source: raw Hugging Face Hub `config.json`, `tokenizer_config.json`, `generation_config.json`, and where present `model.safetensors.index.json`. Small snapshots were saved under this folder in per-repo subdirectories.

Source files inspected:

- `src/transformers/models/hunyuan_v1_dense/configuration_hunyuan_v1_dense.py`
- `src/transformers/models/hunyuan_v1_dense/modular_hunyuan_v1_dense.py`
- `src/transformers/models/hunyuan_v1_dense/modeling_hunyuan_v1_dense.py`
- `src/transformers/modeling_rope_utils.py`
- `src/transformers/integrations/sdpa_attention.py`
- `src/transformers/cache_utils.py`
- `src/transformers/masking_utils.py`
- `tests/models/hunyuan_v1_dense/test_modeling_hunyuan_v1_dense.py`
- `docs/source/en/model_doc/hunyuan_v1_dense.md`

Authoritative source note: `modeling_hunyuan_v1_dense.py` is generated from `modular_hunyuan_v1_dense.py`; future source edits should use the modular file. The generated file was still inspected because it includes expanded inherited Llama code.

Missing files or assumptions: no gated official dense checkpoints were encountered in the selected sweep. Legacy `tencent/Hunyuan-7B-*-0124` repos advertise custom code and older tags; they are out of scope for this native `hunyuan_v1_dense` report unless separately audited.

## 2. High-level architecture

This is a text-only causal decoder for autoregressive generation. The primary DinoML target should be `HunYuanDenseV1ForCausalLM`: token embedding, repeated decoder blocks, final RMSNorm, tied or untied LM projection depending on checkpoint config, then generation-controller sampling outside the graph.

Dataflow:

```text
tokenizer/chat template -> input_ids/attention_mask -> embeddings -> decoder prefill -> KV cache -> decode -> logits -> sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, language/task prompt conventions for MT/instruct variants, generation config.
- GPU/runtime prefill: embedding lookup, causal mask, full-sequence decoder, logits for selected positions.
- GPU/runtime decode: one or more new tokens, position id offset from cache length, per-layer KV cache update, last-token logits.
- Independently validatable pieces: RMSNorm, dynamic-alpha RoPE, Q/K head RMSNorm, GQA attention, SwiGLU MLP, tied LM head, cache update/reorder.

`HunYuanDenseV1ForSequenceClassification` is implemented through a generic sequence-classification wrapper. It is optional/deferred for the generation target.

## 3. Important config dimensions

Source defaults from `configuration_hunyuan_v1_dense.py`: `vocab_size=290943`, `hidden_size=4096`, `intermediate_size=11008`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=None` then defaulted to attention heads, `head_dim=None`, `hidden_act="silu"`, `max_position_embeddings=2048`, `rms_norm_eps=1e-5`, `attention_bias=False`, `attention_dropout=0.0`, `use_cache=True`, `tie_word_embeddings=False`.

Representative checkpoint sweep from saved Hub configs:

| checkpoint | params metadata | H | layers | Q heads | KV heads | head_dim | Q width | KV width | MLP I | vocab | max pos | dtype | cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `tencent/Hunyuan-0.5B-Pretrain` | 539,010,048 BF16 | 1024 | 24 | 16 | 8 | 128 | 2048 | 1024 | 3584 | 120818 | 262144 | bf16 | true |
| `tencent/Hunyuan-1.8B-Instruct` | Hub metadata not recorded here | 2048 | 32 | 16 | 4 | 128 | 2048 | 512 | 6144 | 120818 | 262144 | bf16 | true |
| `tencent/Hunyuan-4B-Instruct` | Hub metadata not recorded here | 3072 | 36 | 32 | 8 | 128 | 4096 | 1024 | 8192 | 120818 | 262144 | bf16 | true |
| `tencent/Hunyuan-7B-Instruct` | sharded safetensors, Hub used storage ~15.0 GB | 4096 | 32 | 32 | 8 | 128 | 4096 | 1024 | 14336 | 128167 | 32768 | bf16 | true |
| `tencent/Hunyuan-MT-7B` | translation tuned | 4096 | 32 | 32 | 8 | 128 | 4096 | 1024 | 14336 | 128256 | 32768 | bf16 | true |
| `tiny-random/hunyuan-dense-v1` | open tiny/debug | 16 | 2 | 2 | 1 | 32 | 64 | 32 | 64 | 128167 | 32768 | bf16 | true |
| `tencent/Hunyuan-7B-Instruct-FP8` | quantized | 4096 | 32 | 32 | 8 | 128 | 4096 | 1024 | 14336 | 128167 | 32768 | bf16 logical | config says `use_cache=false` |
| `tencent/Hunyuan-7B-Instruct-GPTQ-Int4` | quantized | 4096 | 32 | 32 | 8 | 128 | 4096 | 1024 | 14336 | 128167 | 32768 | bf16 logical | true |

RoPE fields in Hub configs use legacy `rope_scaling` plus top-level `rope_theta`. The inspected Transformers config standardizes this into `rope_parameters`. All selected official configs use dynamic RoPE with `rope_theta=10000`; alpha is `1000.0` except `Hunyuan-MT-7B`, which uses `100000.0`.

## 3a. Family variation traps

- Do not infer attention projection width from `hidden_size`. Source uses `q_proj: hidden_size -> num_attention_heads * head_dim`, `k_proj/v_proj: hidden_size -> num_key_value_heads * head_dim`, and `o_proj: num_attention_heads * head_dim -> hidden_size`. The 0.5B and 4B configs have widened Q/O widths.
- GQA is standard in official configs: KV heads are fewer than Q heads. Runtime attention must either repeat KV logically or use a backend with native GQA.
- Source always creates Q and K per-head RMSNorm after RoPE. Historical config flags such as `use_qk_norm` should not be treated as disabling this path for the inspected native source.
- Source always creates RoPE and MLP projections without MLP bias. Historical fields `use_rotary_pos_emb`, `mlp_bias`, `norm_type`, `dense_list`, `cla_share_factor`, `pool_type`, and classification flags are not read by this modeling source for CausalLM.
- `attention_bias` is read and should guard projection bias admission, though representative configs set it false.
- `attention_dropout` exists; inference passes dropout 0.0 because `self.training` is false.
- Official 0.5B/1.8B/4B use vocab 120818 and long max position 262144. 7B uses vocab 128167 and max position 32768. MT-7B uses vocab 128256 and a different RoPE alpha.
- Tokenizer/chat-template conventions differ between the newer small models and 7B/MT. Treat language/task prompting as generation ABI, not model graph.
- Quantized variants advertise compressed-tensors FP8, GPTQ int4, and AWQ int4 repos. These are loading/provider contracts, not new neural graph ops.
- `tie_word_embeddings` is true in representative official configs, while source defaults false. Weight loading must preserve aliasing between `model.embed_tokens.weight` and `lm_head.weight` when tied.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `input_ids -> [B,T,H]`.
- `view`/reshape from `[B,T,width]` to `[B,T,heads,D]`.
- Transpose to attention layout `[B,heads,T,D]`.
- `contiguous` after attention output transpose.
- Slice/index for `logits_to_keep`: int keeps last N tokens; tensor index selects arbitrary token positions.
- `arange`, add scalar cache offset, unsqueeze for default `position_ids`.
- Causal/padding mask construction compatible with Transformers `create_causal_mask`.
- Sequence-classification optional: rightmost non-pad token selection and gather.

Neural network primitives:

- RMSNorm over last dim, fp32 variance and rsqrt, multiply by learned weight.
- Linear/GEMM without bias for MLP, LM head, and representative attention configs.
- Optional Linear bias for attention projections only when `attention_bias=true`.
- SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Residual adds after attention and MLP.
- Final RMSNorm.

Attention primitives:

- Causal self-attention.
- MHA/GQA with `num_key_value_groups = num_attention_heads // num_key_value_heads`.
- Q/K/V projections with nonstandard widths as listed in the sweep table.
- RoPE on Q/K, then per-head Q/K RMSNorm.
- KV cache update after RoPE and Q/K norm; cached keys are already position-encoded and normalized.
- Eager fallback: `matmul(Q, K^T) * head_dim^-0.5`, add mask, fp32 softmax, cast to query dtype, dropout in training, `matmul(P, V)`.
- SDPA/Flash/Flex paths are advertised by the source; DinoML can start with its own fused causal GQA attention path and use eager math as parity reference.

Position/rotary ops:

- Dynamic-alpha RoPE inverse-frequency initialization.
- Runtime cos/sin generation in fp32 using `inv_freq @ position_ids`.
- Rotate-half in split-half convention, not interleaved-pair convention.

Generation/cache ops:

- Dynamic per-layer KV cache, shape before KV repeat: keys/values `[B, num_key_value_heads, cached_T, head_dim]`.
- Cache reorder for beam search can be deferred but the cache ABI should not preclude it.
- Last-token-only logits via `logits_to_keep=1` should be first-class for decode.

Quantized/packed weight metadata ops:

- FP8 compressed-tensors: config targets `Linear`, ignores `lm_head` and `model.embed_tokens`, no KV-cache quantization scheme.
- GPTQ int4: bits 4, group size 128, symmetric, `desc_act=true`, `static_groups=true`, checkpoint format `gptq`.
- AWQ int4 is present in official repo search results but not deeply inspected here; route to a quantized-weight audit before native admission.

Preprocessing-coupled ops:

- Tokenizer and chat template are CPU-side. No multimodal scatter, image/audio packing, or processor tensors are part of this family.
- MT-7B generation parity requires translation prompt/template handling outside the graph.

## 5. Layer/block breakdown

Model:

```text
input_ids -> Embedding(vocab, H)
position_ids = arange(T) + cached_length when omitted
position_embeddings = DynamicAlphaRoPE(position_ids, H/D config)
repeat N decoder layers
final RMSNorm(H)
lm_head(H -> vocab)
```

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(H)(x)
q = Linear(H -> Q_heads * D, bias=attention_bias)(x).view(B,T,Q_heads,D).transpose(1,2)
k = Linear(H -> KV_heads * D, bias=attention_bias)(x).view(B,T,KV_heads,D).transpose(1,2)
v = Linear(H -> KV_heads * D, bias=attention_bias)(x).view(B,T,KV_heads,D).transpose(1,2)
q,k = RoPE(q,k, cos, sin)
q = RMSNorm(D)(q)
k = RMSNorm(D)(k)
k,v = cache.update(k,v, layer_idx) when cache is provided
attn = causal_attention(q,k,v, mask, scale=D^-0.5)
x = residual + Linear(Q_heads * D -> H, bias=attention_bias)(attn)
residual = x
x = RMSNorm(H)(x)
x = residual + Linear(I -> H)(silu(Linear(H -> I)(x)) * Linear(H -> I)(x))
```

Representative block widths:

- 0.5B: `H=1024`, `Q=2048`, `KV=1024`, `I=3584`.
- 1.8B: `H=2048`, `Q=2048`, `KV=512`, `I=6144`.
- 4B: `H=3072`, `Q=4096`, `KV=1024`, `I=8192`.
- 7B/MT: `H=4096`, `Q=4096`, `KV=1024`, `I=14336`.

## 6. Attention requirements

Attention is causal self-attention with GQA in representative checkpoints. There is no encoder cross-attention, no sliding window, no ALiBi, no block-sparse attention, and no multimodal prefix branch in the inspected source.

Required fields:

- Query length: prefill `Tq=T`; decode commonly `Tq=1`.
- Key/value length: prefill `Tk=T`; decode `Tk=past_T + Tq`.
- Q shape: `[B, num_attention_heads, Tq, head_dim]`.
- K/V cache shape before repeat: `[B, num_key_value_heads, Tk, head_dim]`.
- Eager GQA expands K/V to query head count with `expand` then `reshape`.
- Source SDPA path may use PyTorch `enable_gqa=True` when mask is absent and runtime supports it; otherwise it repeats KV.
- Masking style: additive attention mask in eager path; SDPA receives either mask or `is_causal` depending on conditions.
- Softmax: eager path computes softmax in fp32 and casts to query dtype.
- Dropout: training only. Inference dropout is zero even when config has `attention_dropout=0.1`.
- Flash/SDPA/Flex compatibility: source declares support. DinoML first integration can target dense causal GQA attention and later add FlashAttention-compatible prefill/decode kernels.

Cache detail: keys are cached after RoPE and Q/K RMSNorm. Values are cached after V projection and reshape/transpose. Therefore a DinoML cache update must not store raw pre-RoPE keys.

## 7. Position encoding and custom math

The model uses RoPE with a Hunyuan-specific dynamic-alpha branch. Configs provide `rope_scaling` using old key `type="dynamic"`; Transformers standardizes this to `rope_parameters["rope_type"]="dynamic"` and carries `rope_theta`.

Short parity snippet:

```python
def hunyuan_inv_freq(head_dim, rope_theta, alpha):
    base = rope_theta * alpha ** (head_dim / (head_dim - 2))
    return 1.0 / (base ** (arange(0, head_dim, 2).float() / head_dim))

def hunyuan_rope(q, k, cos, sin):
    # q/k: [B, heads, T, D], cos/sin: [B, T, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Runtime cos/sin generation:

```text
freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
emb = concat(freqs, freqs, dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

For dynamic-alpha configs, `attention_scaling=1.0`. `inv_freq` can be precomputed per model/config and regenerated if supporting dynamic RoPE update paths beyond the current alpha branch. Cos/sin depend on runtime `position_ids`, batch, sequence length, and cache offset.

## 8. Preprocessing and input packing

The model consumes token IDs or caller-supplied input embeddings. It has no image/audio/video processor.

CPU/data pipeline:

- `AutoTokenizer` is `PreTrainedTokenizerFast` for official configs.
- Newer 0.5B/1.8B/4B tokenizer snapshots use Hunyuan-specific placeholder-style BOS/EOS/PAD tokens and a longer tool/chat template.
- 7B and MT snapshots use start/eos/pad style tokens and shorter templates.
- Generation configs commonly set sampling defaults: `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `repetition_penalty=1.05`; MT uses `top_p=0.6`.
- Some generation configs use multiple EOS ids, including EOD. Generation controller must respect this separately from graph execution.

GPU/runtime graph inputs:

- `input_ids: int64 [B,T]` or `inputs_embeds: float [B,T,H]`, exactly one.
- `attention_mask` is optional and consumed only by mask construction.
- `position_ids: int64 [B,T]` optional; when omitted, source creates a single-row `[1,T]` sequence offset by cached length.
- `past_key_values` optional cache object.

No placeholder scatter, modality token IDs, packed patch metadata, or `cu_seqlens` are required by the native source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: attention QKV projection grouping

Source pattern:

```text
q = Linear(H -> QW)(x)
k = Linear(H -> KW)(x)
v = Linear(H -> KW)(x)
```

Replacement:

```text
single grouped GEMM or packed QKV GEMM -> split [Q, K, V]
```

Preconditions:

- All three projections have same input tensor and dtype.
- Bias handling matches `attention_bias`; representative configs have no bias.
- Packed output split sizes are `[num_attention_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim]`.
- Do not assume Q/K/V widths are equal.

Weight transform:

```text
packed_weight = concat(q_proj.weight, k_proj.weight, v_proj.weight, dim=0)
```

Failure cases: quantized checkpoints with source-coupled packing, projection bias mismatch, or nonstandard checkpoint key layout should disable the rewrite until loader support proves equivalence.

Parity test sketch: compare separate projections vs packed projection and split for 0.5B-like `H=1024,Q=2048,KV=1024` and 4B-like `H=3072,Q=4096,KV=1024`.

### Rewrite: RoPE + QK RMSNorm attention prelude

Source pattern:

```text
q,k = RoPE(q,k)
q = RMSNorm(D)(q)
k = RMSNorm(D)(k)
```

Replacement: fused per-head pre-attention kernel producing normalized Q/K in attention layout.

Preconditions:

- Last dim is `head_dim`.
- Split-half RoPE convention.
- fp32 norm variance and rsqrt are preserved.
- Cache stores post-RoPE, post-QK-norm keys.

Failure cases: alternate rope type without alpha branch, custom remote code, or layout pass changing the last-dim rotation convention.

### Rewrite: SwiGLU MLP fusion

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement: fused two-input activation multiply epilogue between two GEMMs, then down GEMM. Later, use grouped GEMM for gate/up.

Preconditions:

- `hidden_act == "silu"`.
- `gate_proj` and `up_proj` have same output width `intermediate_size`.
- No MLP bias in native source.

Parity test sketch: random BF16/FP32 hidden states, compare unfused PyTorch path against DinoML fused epilogue for all representative H/I widths.

### Rewrite: tied LM head alias

Source pattern: separate `Embedding(vocab,H)` and `Linear(H,vocab,bias=False)` modules with `_tied_weights_keys`.

Replacement: one logical weight with two views/usages.

Preconditions:

- Config or loaded state indicates tied weights.
- Weight dtype/layout supports both embedding gather and GEMM RHS use.

Failure cases: source defaults `tie_word_embeddings=false`, quantized variants may exclude `lm_head` or embeddings from quantization, and checkpoints can diverge. Validate from state dict metadata.

### Rewrite: last-token-only logits

Source pattern: `logits = lm_head(hidden_states[:, slice_indices, :])`.

Replacement: during decode use only final hidden row, avoiding full `[B,T,vocab]`.

Preconditions:

- `logits_to_keep` is int 1 or a contiguous suffix.
- Loss is not requested.

Failure cases: arbitrary tensor `logits_to_keep`, training loss, or callers requesting all logits.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for hidden states and per-head Q/K norm. It appears multiple times per layer and uses fp32 accumulation.
- Causal GQA attention with RoPE/QK-norm prelude and KV cache update. This is the core prefill/decode bottleneck.
- GEMM coverage for nonstandard projection widths, especially 0.5B and 4B widened Q/O projections.
- SwiGLU activation multiply. It is in every decoder layer and maps cleanly to existing elementwise/fused epilogue work.
- Last-token-only logits, especially for large vocab 120k/128k.

Medium priority:

- Packed QKV projection and grouped gate/up projection.
- RoPE cos/sin generation cache or fused generation for decode.
- Quantized RHS dequant-to-GEMM for official FP8/GPTQ/AWQ variants, after dense parity.
- Tied embedding/LM-head residency and alias-aware loading.

Lower priority:

- Sequence-classification pooling head.
- FlexAttention-specific block-mask path; native source supports it, but dense causal attention is enough for first parity.
- Training/dropout/loss paths.

## 11. Runtime staging plan

Stage 1: parse config and load dense BF16 weights for the tiny-random checkpoint and one official small checkpoint. Canonicalize legacy `rope_scaling` into DinoML RoPE metadata.

Stage 2: implement single-block parity with random weights for RMSNorm, Q/K RoPE+norm, GQA attention, residuals, and SwiGLU. Include widened Q/O width cases.

Stage 3: full prefill parity for `HunYuanDenseV1Model` without KV cache. Start with `attention_mask=None` and then add padding masks.

Stage 4: CausalLM logits parity with tied LM head and `logits_to_keep=1`.

Stage 5: decode parity with per-layer KV cache. Cache tensors are post-RoPE/post-QK-norm keys and raw projected values.

Stage 6: optimized attention and fusions: packed QKV, fused RoPE+QK norm, FlashAttention-style GQA prefill/decode, SwiGLU fusion.

Stage 7: quantized variants as separate provider/load-path work: compressed-tensors FP8 first if metadata is straightforward, then GPTQ/AWQ.

Stubs acceptable initially: tokenizer/chat template, generation sampling, sequence classification, beam cache reorder, quantized checkpoint loading.

## 12. Parity and validation plan

- RMSNorm random tensor tests over `[B,T,H]` and `[B,heads,T,D]`, requiring fp32 variance behavior. Suggested tolerance: fp32 `1e-5`, bf16 `2e-2`.
- RoPE tests for dynamic-alpha base: compare inv_freq, cos/sin, and rotated Q/K for alpha `1000.0` and `100000.0`.
- Projection-shape tests for 0.5B and 4B traps where Q width differs from hidden size.
- Single attention-layer parity with no cache, then with cache for prefill+one-token decode. Verify cached key placement after RoPE/QK norm.
- MLP SwiGLU parity for all representative `H/I` widths.
- Full tiny-random end-to-end prefill logits parity.
- Official 0.5B short-prompt parity for prefill logits and greedy next token.
- Decode token parity for 8 to 32 generated tokens with fixed sampling disabled.
- Optional MT prompt parity should validate tokenizer/generation-controller formatting, not just graph logits.
- Quantized variants: first validate loader materialization against dequantized dense reference for selected linear weights before graph parity.

## 13. Performance probes

- Prefill throughput sweep: batch `1,2,4,8`, sequence `128,512,2048,8192`, separate 32k/262k position-id overhead.
- Decode tokens/sec sweep: batch `1,4,16,64`, cache length `128,2k,8k,32k`.
- Attention backend comparison: eager dense GQA, SDPA-like GQA, FlashAttention-style GQA, paged decode when available.
- QKV packing benefit for widened 0.5B/4B projections.
- RoPE+QK-norm fused prelude cost in prefill and decode.
- LM-head logits cost with all logits vs `logits_to_keep=1`.
- KV cache memory: per layer `2 * B * KV_heads * T * head_dim * dtype_bytes`; for 7B BF16 this is `2 * B * 8 * T * 128 * 2` bytes per layer.
- Quantized load/dequant probes: compressed FP8/GPTQ/AWQ load time, dequant-to-GEMM cost, and memory residency compared with dense BF16.
- Tokenizer/chat-template throughput separately from model runtime for high-request-rate serving.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing, dropout behavior.
- Sequence classification head and pooling.
- Beam search cache reorder and complex generation processors beyond EOS handling.
- FlexAttention block-mask-specific lowering.
- Legacy custom-code `*-0124` checkpoints.
- Quantized official variants until dense graph parity and explicit quantized provider admission are ready.
- Multi-GPU tensor parallel and pipeline plans, though source includes simple `_tp_plan`/`_pp_plan` metadata for `lm_head`.
- Full 262144-token production admission until memory planning, KV cache policy, and RoPE/cos-sin handling are profiled.

## 15. Final implementation checklist

- [ ] Parse `HunYuanDenseV1Config`, including legacy `rope_scaling` to standardized RoPE metadata.
- [ ] Load dense BF16 weights and preserve tied embedding/LM-head aliases.
- [ ] Implement embedding lookup and final LM projection with `logits_to_keep`.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement dynamic-alpha Hunyuan RoPE and split-half rotation.
- [ ] Implement attention projections with explicit Q/K/V widths.
- [ ] Implement per-head Q/K RMSNorm after RoPE.
- [ ] Implement causal GQA attention with padding-mask support.
- [ ] Implement per-layer KV cache storing post-RoPE/post-QK-norm keys.
- [ ] Implement SwiGLU MLP.
- [ ] Add packed QKV rewrite guarded by split widths and bias.
- [ ] Add fused RoPE+QK-norm prelude.
- [ ] Add tied LM-head alias handling.
- [ ] Add tiny-random full-model parity.
- [ ] Add official 0.5B prefill and decode parity.
- [ ] Add widened-projection parity tests for 0.5B and 4B configs.
- [ ] Benchmark prefill, decode, logits, KV memory, and RoPE/QK-norm overhead.
- [ ] Audit FP8 compressed-tensors, GPTQ, and AWQ checkpoints as separate load/provider tasks.
