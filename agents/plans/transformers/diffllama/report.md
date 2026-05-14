# DiffLlama Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: native scope is model_type="diffllama"; representative configs from kajuma/DiffLlama-0.3B-handcut, kajuma/DiffLlama-1B, reyllama/DiffLlama-375M checkpoint-64434, reyllama/diffllama_300m checkpoint-12000.
Config source: local Transformers configuration plus Hub config.json snapshots under agents/plans/transformers/diffllama/snapshots/.
Source files inspected: X:/H/transformers/src/transformers/models/diffllama/configuration_diffllama.py, modeling_diffllama.py, modular_diffllama.py.
Any missing files or assumptions: modeling_diffllama.py is generated from modular_diffllama.py; exact runtime behavior was read from the generated file, future upstream edits should target modular_diffllama.py. No gated model links were encountered. amazingvince/diff-llama uses model_type="diff_llama" plus remote code and is out-of-scope for native DiffLlama admission.
```

Small snapshots saved:

- `snapshots/kajuma__DiffLlama-0.3B-handcut/config.json`, tokenizer and generation config.
- `snapshots/kajuma__DiffLlama-1B/config.json`, tokenizer and generation config.
- `snapshots/reyllama__DiffLlama-375M/checkpoint-64434/config.json`, tokenizer and generation config.
- `snapshots/reyllama__diffllama_300m/checkpoint-12000/config.json`, tokenizer and generation config.
- `snapshots/amazingvince__diff-llama/config.json` and `modeling_diff_llama.py` only as an out-of-scope remote-code contrast.

Key source anchors: config defaults and post-init are in `configuration_diffllama.py:47-77`; MLP is `modeling_diffllama.py:56-69`; RoPE helpers are `modeling_diffllama.py:72-166`; differential attention is `modeling_diffllama.py:186-279`; FlashAttention2 and SDPA variants are `modeling_diffllama.py:281-486`; RMSNorm is `modeling_diffllama.py:488-506`; decoder/model/LM head are `modeling_diffllama.py:515-730`.

Pinned source URLs:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/diffllama/configuration_diffllama.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/diffllama/modeling_diffllama.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/diffllama/modular_diffllama.py`

Representative Hub config URLs:

- `https://huggingface.co/kajuma/DiffLlama-0.3B-handcut/blob/main/config.json`
- `https://huggingface.co/kajuma/DiffLlama-1B/blob/main/config.json`
- `https://huggingface.co/reyllama/DiffLlama-375M/blob/main/checkpoint-64434/config.json`
- `https://huggingface.co/reyllama/diffllama_300m/blob/main/checkpoint-12000/config.json`
- `https://huggingface.co/amazingvince/diff-llama/blob/main/config.json` out-of-scope remote-code contrast.

## 2. High-level architecture

DiffLlama is a text-only causal decoder LM. It is Llama-shaped, but replaces ordinary self-attention output mixing with a Differential Transformer-style two-branch value/output subtraction controlled by learned per-layer lambda vectors.

```text
tokenizer/input_ids -> token embedding -> repeated decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenization, chat template, attention mask construction inputs, generation controller.
- GPU runtime prefill: embeddings, causal decoder stack, RoPE, differential self-attention, SwiGLU MLP, logits.
- GPU runtime decode: one-token or small-token decode with per-layer KV cache, same differential attention math after cache update.
- Independently stageable: tokenizer ABI, one decoder block, RoPE/lambda math, prefill logits, decode cache update, last-token-only logits.

Primary DinoML target for this report: `DiffLlamaForCausalLM` prefill and decode. Sequence classification, token classification, and question answering wrappers are implemented but optional/deferred for this target.

## 3. Important config dimensions

Source defaults from `DiffLlamaConfig`:

| Field | Default |
|---|---:|
| vocab_size | 32000 |
| hidden_size | 2048 |
| intermediate_size | 8192 |
| num_hidden_layers | 16 |
| num_attention_heads | 32 |
| num_key_value_heads | defaults to num_attention_heads |
| head_dim | defaults to hidden_size // num_attention_heads |
| max_position_embeddings | 2048 |
| hidden_act | silu |
| rms_norm_eps | 1e-5 |
| attention_bias | false |
| attention_dropout | 0.0 |
| lambda_std_dev | 0.1 |
| use_cache | true |
| tie_word_embeddings | false |

Representative checkpoint sweep, from saved `config.json` files:

| Checkpoint | Scope | Vocab | H | Layers | Q heads | KV heads | Head dim | MLP | Context | RoPE | Tied emb | Dtype |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `kajuma/DiffLlama-0.3B-handcut` | native | 128256 | 2048 | 16 | 32 | 8 | 64 | 8192 | 131072 | llama3, theta 500000, factor 32 | true | float32 |
| `kajuma/DiffLlama-1B` | native | 102400 | 2048 | 16 | 32 | 8 | 64 | 8192 | 8192 | default, theta 10000 | false | bfloat16 |
| `reyllama/DiffLlama-375M/checkpoint-64434` | native | 128256 | 1024 | 16 | 16 | 4 | 64 | 4096 | 131072 | llama3, theta 500000, factor 32 | true | bfloat16 |
| `reyllama/diffllama_300m/checkpoint-12000` | native | 128256 | 2048 | 16 | 32 | 8 | 64 | 8192 | 131072 | llama3, theta 500000, factor 32 | true | bfloat16 |
| `amazingvince/diff-llama` | remote-code only contrast | 32768 | 768 | 24 | 16 | 16 | 48 | 2304 | 2048 | linear rope_scaling factor 1 | false | float32 |

## 3a. Family variation traps

- Native family name is `diffllama`; `diff_llama` remote-code repos should not be silently admitted through this report.
- `hidden_size == num_attention_heads * head_dim` in sampled native configs, but source permits explicit `head_dim`; admission should validate projection widths from config, not infer blindly.
- Native checkpoints use GQA (`num_key_value_heads < num_attention_heads`) in representative Hub configs, even though config defaults to MHA.
- Long-context configs use legacy `rope_scaling`/`rope_theta` fields that Transformers standardizes into `rope_parameters`; DinoML should preserve llama3 RoPE behavior.
- `tie_word_embeddings` varies. If true, `lm_head.weight` and `model.embed_tokens.weight` are one logical parameter alias.
- `mlp_bias`, `pretraining_tp`, and `patch_size` appear in some configs but are not read by the inspected native modeling source.
- `attention_bias` exists and changes Q/K/V/O Linear bias materialization, although sampled native configs set it false.
- FlashAttention2 path rejects `StaticCache`; SDPA/eager can use cache abstractions.
- Differential attention splits/repeats values and chunks attention outputs across head dimension/head groups; ordinary Llama attention fusion is not parity-safe without these extra operations.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,H]`.
- View/reshape/transpose/contiguous for `[B,S,H] <-> [B,heads,S,D]`.
- Chunk/split, concat, repeat/expand/reshape for KV repeat and differential value preparation.
- Slice/index for `logits_to_keep`, position id generation, and mask slicing.

Neural network primitives:

- Linear Q: `H -> num_attention_heads * head_dim`, optional bias.
- Linear K/V: `H -> num_key_value_heads * head_dim`, optional bias.
- Linear O: `num_attention_heads * head_dim -> H`, optional bias.
- SwiGLU MLP: `down(silu(gate(x)) * up(x))`, with `H -> I`, `H -> I`, `I -> H`, no bias in native source.
- RMSNorm with fp32 variance and affine weight for model norms; non-affine RMSNorm for attention groupnorm.
- Final LM head `H -> vocab_size`, no bias; optional tied alias to embeddings.

Attention primitives:

- Causal self-attention, MHA/GQA.
- Eager matmul-softmax-matmul path with fp32 softmax.
- SDPA path with pre-expanded KV and differential V layout.
- FlashAttention2 path called twice, once per split value branch.

Position/rotary ops:

- RoPE cos/sin generation from `position_ids`, `rope_parameters`, `head_dim`.
- `rotate_half`, apply RoPE to Q/K before cache update.
- Dynamic RoPE update decorator for advanced RoPE types, including llama3 configs.

Generation/cache ops:

- Dynamic cache construction when `use_cache=True` and no cache is provided.
- Per-layer cache update stores RoPE-applied K and original V shape `[B,kv_heads,T,D]`.
- Cache reorder/reset behavior is inherited from Transformers cache utilities, not implemented in this model file.

Preprocessing-coupled ops:

- Tokenizer/chat template only. No image/audio/video processors.

Quantized/packed weight metadata:

- No native packed weight format in source. Quantized configs should route through generic Transformers quantization/loading policy, not DiffLlama-specific graph ops.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers`:

```text
x0 = hidden_states                       # [B,S,H]
x = RMSNorm_affine(x0)
q = Linear(H -> QH*D)(x)                 # [B,S,QH*D]
k = Linear(H -> KVH*D)(x)                # [B,S,KVH*D]
v = Linear(H -> KVH*D)(x)                # [B,S,KVH*D]
q = view(q, [B,QH,S,D]); k/v = view(... [B,KVH,S,D])
q,k = RoPE(q,k, position_embeddings)
k,v = cache.update(k,v, layer_idx)       # optional, before repeat_kv
k = repeat_kv(k, QH/KVH); v = repeat_kv(v, QH/KVH)
v = concat(chunk(v, 2, dim=heads), dim=D).repeat(heads x2)
scores = q @ k.T / sqrt(D) + causal_mask
p = softmax(scores, fp32).to(q.dtype)
y = p @ v                                # [B,QH,S,2D]
y1,y2 = chunk(y, 2, dim=heads)
lambda = exp(sum(lambda_q1*lambda_k1)) - exp(sum(lambda_q2*lambda_k2)) + lambda_init(layer)
y = y1 - lambda * y2                     # [B,QH/2,S,2D] -> same flattened H when QH*D == H
y = (1 - lambda_init) * RMSNorm_no_affine(y, normalized_shape=2D)
y = Linear(QH*D -> H)(reshape(y, [B,S,QH*D]))
hidden_states = x0 + y
x1 = hidden_states
x = RMSNorm_affine(x1)
x = down_proj(silu(gate_proj(x)) * up_proj(x))
hidden_states = x1 + x
```

Final model path:

```text
hidden = token_embedding(input_ids)
position_ids = arange(S) + past_seen_tokens unless supplied
causal_mask = create_causal_mask(...)
shared position_embeddings = rotary_emb(hidden, position_ids)
hidden = decoder_stack(hidden, causal_mask, position_embeddings, cache)
hidden = final RMSNorm(hidden)
logits = lm_head(hidden[:, logits_to_keep_slice, :])
```

## 6. Attention requirements

- Type: causal self-attention only for primary CausalLM target.
- Heads: Q heads from `num_attention_heads`; KV heads from `num_key_value_heads`; GQA when KV heads are fewer.
- Widths: Q projection width `QH * D`; K/V width `KVH * D`; O input width `QH * D`.
- Masking: Transformers `create_causal_mask` produces the additive causal/padding mask for eager/SDPA; SDPA slices mask to current KV length.
- Cache: cached K is stored after RoPE; cached V is stored before `repeat_kv` and before differential value concat/repeat.
- Eager math order: QK matmul, divide by `sqrt(head_dim)`, add mask, fp32 softmax, cast back, dropout, PV matmul, differential subtraction/groupnorm/O projection.
- FlashAttention2: static cache is rejected. The path transposes to `[B,S,H,D]`, splits V along head axis into two halves, repeats each half to full head count, runs flash attention twice, concatenates branch outputs, then applies lambda/groupnorm/O projection.
- Sliding window: source passes `getattr(self, "sliding_window", None)` to FlashAttention2 only; `DiffLlamaConfig` does not define it. Treat as absent unless a future subclass/config adds it.
- Dense attention outputs for `output_attentions` are naturally available only from eager attention; optimized paths return `None`.

## 7. Position encoding and custom math

RoPE is standard Llama-style, with config-standardized variants from `ROPE_INIT_FUNCTIONS` when `rope_type != "default"`. Cos/sin are generated in fp32 then cast to the model dtype. For llama3 configs, the source relies on shared Transformers rope utilities.

Short custom math sketch:

```python
def apply_diffllama_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

def lambda_full(layer_idx, q1, k1, q2, k2):
    init = 0.8 - 0.6 * exp(-0.3 * layer_idx)
    return exp(sum(q1 * k1, fp32)) - exp(sum(q2 * k2, fp32)) + init
```

Precompute candidates: inverse frequency tables and, for bounded context buckets, cos/sin tables. Dynamic inputs: `position_ids`, past length, and any dynamic RoPE scaling update for long-context variants.

## 8. Preprocessing and input packing

Runtime tensors:

- `input_ids`: `[B,S]` int64/token IDs, or `inputs_embeds`: `[B,S,H]`; exactly one must be supplied.
- `attention_mask`: optional tokenizer/padding mask consumed by `create_causal_mask`.
- `position_ids`: optional `[B,S]`; if absent, generated from sequence length plus cache length.
- `past_key_values`: optional Transformers `Cache`.

Tokenizer observations from sampled configs:

- 128k-vocab native configs use Llama-3-like special tokens such as `<|begin_of_text|>`, `<|eot_id|>`, and right padding.
- `kajuma/DiffLlama-1B` uses `vocab_size=102400`, so vocab/tokenizer must be checkpoint-specific.
- Chat templating and forced generation behavior are controller/tokenizer ABI, not model graph ops.

No multimodal placeholder scatter, processor grid metadata, audio features, OCR boxes, or layout-sensitive image/video tensors are present.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV separate Linear pack for GEMM batching

Source pattern: three independent Linear ops from the same normalized hidden state.

Replacement: one packed GEMM producing `[q, k, v]` plus split views.

Preconditions:

- Same input tensor, dtype, and batch/sequence flattening.
- Bias handling matches `attention_bias`.
- Packed output order must be exactly Q then K then V.
- Projection widths are `QH*D`, `KVH*D`, `KVH*D`; do not assume equal widths.

Failure cases: quantized per-module loading with incompatible packing metadata; checkpoints requiring independent weight aliasing; custom attention bias variants.

Parity test sketch: compare Q/K/V tensors before RoPE for random `[B,S,H]`, with and without attention bias.

### Rewrite: Differential attention fused epilogue

Source pattern: attention output branch split, scalar lambda computation, `y1 - lambda*y2`, non-affine RMSNorm over `2*head_dim`, scale by `1-lambda_init`.

Replacement: fused post-attention kernel over `[B,S,QH/2,2D]`.

Preconditions:

- Differential value layout exactly matches source chunk/concat/repeat path.
- Lambda vectors are per-layer constants and reduction is fp32.
- Groupnorm has `elementwise_affine=False` and normalized shape `2*D`.

Failure cases: output attentions requested from eager path, training/dropout enabled, head counts not divisible by 2.

Parity test sketch: random attention probabilities and V for each sampled config; compare fused output before O projection.

### Rewrite: last-token-only logits

Source pattern: `hidden_states[:, slice_indices, :] -> lm_head`.

Replacement: for decode or generation with `logits_to_keep=1`, GEMM only the last token row.

Preconditions: caller does not request full logits or loss; `logits_to_keep` is static int 1 or validated index tensor.

Failure cases: training labels/loss, full-sequence logit consumers.

### Layout notes

This is text-only rank-3/rank-4 tensor code. There is no NCHW/NHWC region. Protect attention head/sequence axes from generic layout translation: source uses `[B,heads,S,D]` for eager/SDPA and `[B,S,heads,D]` for FlashAttention2 calls.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm affine and non-affine RMSNorm: every block uses two affine RMSNorms plus attention groupnorm.
- GQA causal attention with RoPE and KV cache: core prefill/decode bottleneck.
- Differential attention postprocess: lambda reductions, branch subtraction, non-affine RMSNorm, and scale are unique to this family.
- SwiGLU MLP: `silu(gate) * up -> down` is the largest GEMM-heavy block after attention.
- Last-token-only logits: important for decode with large vocab.

Medium priority:

- Packed QKV projection GEMM.
- RoPE application fused with Q/K reshape/cache staging.
- KV repeat avoidance inside attention provider for GQA.
- Shared cos/sin table cache for long-context buckets.

Lower priority:

- FlashAttention2 twin-call parity path; useful as an optimization model, but DinoML can first lower eager/SDPA-equivalent math.
- Classification/QA/token heads.
- Full output-attentions materialization.

## 11. Runtime staging plan

1. Parse native `DiffLlamaConfig`, including legacy `rope_scaling`/`rope_theta` normalization into a DinoML rope manifest.
2. Load weights for embeddings, per-layer Q/K/V/O, lambda vectors, RMSNorms, MLP, final norm, and LM head with tied-weight alias support.
3. Implement one-block fp32/bf16 parity with eager attention and no cache.
4. Implement full prefill parity for `DiffLlamaForCausalLM`, returning logits and optional cache.
5. Add decode with DynamicCache-compatible K/V shapes; store K after RoPE, V before repeat/differential transform.
6. Add optimized GQA attention provider and differential post-attention fusion.
7. Add last-token logits and batching/continuous decode scheduling.
8. Add optional wrapper heads only after CausalLM is stable.

Stub initially: FlashAttention2-specific dispatch, output attentions, training loss, gradient checkpointing, remote-code `diff_llama` repos.

## 12. Parity and validation plan

- Config parsing tests for defaults, GQA, explicit `head_dim`, llama3 RoPE, tied embeddings.
- Unit tests for `rotate_half`, RoPE cos/sin, and `lambda_init_fn`.
- Random tensor tests for differential attention postprocess in fp32 and bf16/fp16.
- Single decoder layer parity against Transformers with `attn_implementation="eager"`, no cache.
- Full prefill logits parity for sampled small configs or reduced synthetic configs.
- Decode parity: prefill N tokens, decode one token, compare logits and cache lengths.
- Tied embedding parity: assert one logical parameter/storage alias when config says true.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 prefill `rtol=2e-2, atol=2e-2` initially, tighten after fused kernels are stable.

## 13. Performance probes

- Prefill tokens/sec across sequence lengths 128, 512, 2048, 8192, and long-context buckets for llama3 RoPE configs.
- Decode tokens/sec with cache lengths 128 through 131072 where memory allows.
- Attention backend comparison: eager baseline, SDPA-equivalent, DinoML fused GQA, twin-flash-style differential path.
- Differential postprocess kernel timing separate from QK/PV attention.
- MLP GEMM throughput and SwiGLU fusion benefit.
- Last-token logits versus full-sequence logits for vocab 102400 and 128256.
- KV cache memory usage: per layer `K,V = [B,KVH,T,D]`; compare GQA storage before repeat versus expanded heads.
- RoPE table generation/cache overhead for long-context dynamic position IDs.

## 14. Skip/defer list

- Training, dropout, labels/loss, gradient checkpointing.
- Sequence classification, token classification, question answering heads.
- Output attentions on optimized attention paths.
- FlashAttention2 exact external-kernel dispatch and StaticCache rejection behavior beyond admission checks.
- Remote-code `model_type="diff_llama"` checkpoints.
- Tensor parallel pipeline plans and multi-GPU sharding.
- Generic quantization integration unless provided by shared DinoML weight-loading policy.

## 15. Final implementation checklist

- [ ] Parse `DiffLlamaConfig` and reject non-native `model_type` values unless separately audited.
- [ ] Normalize legacy RoPE config fields into explicit rope provider metadata.
- [ ] Load embeddings, LM head, tied aliases, Q/K/V/O, MLP, RMSNorm, and lambda vectors.
- [ ] Implement DiffLlama RMSNorm affine and non-affine variants.
- [ ] Implement RoPE helpers with llama3 RoPE parity.
- [ ] Implement eager differential GQA attention.
- [ ] Implement KV cache ABI: K after RoPE, V before repeat/differential transform.
- [ ] Implement SwiGLU MLP.
- [ ] Add last-token-only logits lowering.
- [ ] Add one-block parity tests.
- [ ] Add prefill logits parity tests.
- [ ] Add decode cache/logits parity tests.
- [ ] Add GQA/fused attention performance probes.
- [ ] Add differential postprocess fusion after baseline parity is stable.
