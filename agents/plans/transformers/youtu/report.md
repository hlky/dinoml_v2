# DinoML Transformers Audit: `youtu`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: tencent/Youtu-LLM-2B and tencent/Youtu-LLM-2B-Base
Config source: local Transformers defaults plus HF config.json files fetched from the two public repos
Source files inspected:
  transformers/src/transformers/models/youtu/configuration_youtu.py
  transformers/src/transformers/models/youtu/modeling_youtu.py
  transformers/src/transformers/models/youtu/modular_youtu.py
Any missing files or assumptions:
  No image/audio processor exists for the in-library youtu family. Youtu-VL and Youtu-Parsing are remote-code multimodal families and are out of scope for this report.
```

`modeling_youtu.py` and `configuration_youtu.py` are generated from `modular_youtu.py`; future Transformers source edits should target the modular file. The generated files are still the best audit basis because they show the exact current runtime classes after expansion.

HF configs inspected:

- [tencent/Youtu-LLM-2B config.json](https://huggingface.co/tencent/Youtu-LLM-2B/raw/main/config.json), public, current `model_type: "youtu"`.
- [tencent/Youtu-LLM-2B-Base config.json](https://huggingface.co/tencent/Youtu-LLM-2B-Base/raw/main/config.json), public, current `model_type: "youtu"`.
- Historical [tencent/Youtu-LLM-2B-Base config at 3cf6899](https://huggingface.co/tencent/Youtu-LLM-2B-Base/raw/3cf689939d7b5d0c6d58688c5e59e253ea844002/config.json), public, remote-code `model_type: "youtu_llm"` and legacy `rope_theta` / `rope_scaling` fields.
- HF model API metadata for `tencent/Youtu-LLM-2B`: public, safetensors parameter count `1,961,560,064` BF16.

## 2. High-level architecture

Youtu is a text-only dense decoder-only causal LM with Multi-Latent Attention-style low-rank Q and KV projections, Qwen3-style SwiGLU MLP, Llama-style RMSNorm/residual block structure, and tied input/output embeddings.

```text
tokenization/chat template -> input_ids/attention_mask -> token embedding
  -> repeated decoder block with MLA self-attention + SwiGLU MLP
  -> final RMSNorm -> tied LM head -> logits -> generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: fast tokenizer, optional chat template, left truncation, attention mask construction.
- GPU/runtime: embedding, RoPE table computation or lookup, decoder prefill/decode, final logits.
- Cacheable state: per-layer autoregressive KV cache after RoPE/key construction; tokenized prompts and chat-template output are data-pipeline cache candidates.
- First useful DinoML target: `YoutuForCausalLM` prefill logits, then single-token decode with cache.

No NHWC/NCHW image or video layout translation is applicable in the in-library `youtu` model. All neural tensors are text sequence tensors, primarily `[batch, seq, hidden]` and attention tensors `[batch, heads, seq, dim]`.

## 3. Important config dimensions

Current public 2B configs match the in-library defaults except `rope_parameters.rope_theta` is explicitly `1600000` in HF configs.

| Field | Current HF 2B value | Source default | Operator impact |
|---|---:|---:|---|
| `vocab_size` | 128256 | 128256 | Embedding and tied LM head `[128256, 2048]`. |
| `hidden_size` | 2048 | 2048 | Main residual width. |
| `num_hidden_layers` | 32 | 32 | Repeated decoder blocks. |
| `num_attention_heads` | 16 | 16 | Q heads and KV heads in current configs. |
| `num_key_value_heads` | 16 | 16 | No GQA repeat for current 2B, but source supports repeat when lower. |
| `q_lora_rank` | 1536 | 1536 | Two-step Q projection `2048 -> 1536 -> 3072`. |
| `kv_lora_rank` | 512 | 512 | Compressed KV projection/norm path. |
| `qk_nope_head_dim` | 128 | 128 | Non-RoPE Q/K slice per head. |
| `qk_rope_head_dim` | 64 | 64 | RoPE Q/K slice per head. |
| `qk_head_dim` | 192 | computed | Attention score width. |
| `v_head_dim` | 128 | 128 | Value and output-per-head width. |
| `intermediate_size` | 6144 | 6144 | SwiGLU gate/up/down width. |
| `max_position_embeddings` | 131072 | 131072 | Long-context RoPE/cache bound. |
| `rope_parameters` | `{"rope_type":"default","rope_theta":1600000}` | `None` then normalized by base config machinery | RoPE frequencies; legacy configs use `rope_theta`. |
| `rope_interleave` | true | true | Source applies extra view/transpose before RoPE. |
| `attention_bias` | false | false | Q/KV low-rank A projections and output projection can be biased if enabled. |
| `hidden_act` | `silu` | `silu` | SwiGLU activation. |
| `rms_norm_eps` | `1e-6` | `1e-6` | RMSNorm epsilon. |
| `dtype` | `bfloat16` | not a source default | HF config/model metadata dtype. |
| `use_cache` | true | true | DynamicCache support in source. |
| `tie_word_embeddings` | true | true | `lm_head.weight` aliases `embed_tokens.weight`. |

Representative checkpoint/config sweep:

| Repo/config | Scope | Shape variation | Notes |
|---|---|---|---|
| `tencent/Youtu-LLM-2B` main | Instruct text generation | Same 2B dimensions above | Public, current native in-library `model_type: "youtu"`; tokenizer has chat/tool template. |
| `tencent/Youtu-LLM-2B-Base` main | Base text generation | Same 2B dimensions above | Public, current native in-library `model_type: "youtu"`; no architectural delta. |
| `tencent/Youtu-LLM-2B-Base` historical `3cf6899` | Legacy remote-code config | Same neural dimensions | `model_type: "youtu_llm"`, `auto_map` remote code, `torch_dtype`, `rope_theta`, `rope_scaling`; route to this native report only after config migration. |
| `tencent/Youtu-VL-4B-Instruct` | Out of scope | Multimodal `YoutuVLForConditionalGeneration` | Remote-code `youtu_vl` with image/video tokens; requires separate audit. |
| `tencent/Youtu-Parsing` | Out of scope | Multimodal parsing | Remote-code `youtu_vl`/custom processor; requires separate audit. |

## 3a. Family variation traps

- `head_dim` in config is set to `qk_rope_head_dim` (`64`) for RoPE frequency computation, while attention score width is `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192`. Do not infer attention width from `head_dim`.
- Q and K use width 192 per head, V uses width 128 per head. FlashAttention path pads V/output to QK width and slices back.
- Current 2B has `num_key_value_heads == num_attention_heads`, but source has `repeat_kv` for GQA/MQA if future configs use fewer KV heads.
- `q_lora_rank` may be `None`; then source uses a direct Q projection `hidden -> heads * qk_head_dim` instead of low-rank Q A/RMSNorm/B.
- `attention_bias` gates bias on `q_a_proj`, `kv_a_proj_with_mqa`, and `o_proj`; current configs set false.
- `mlp_bias` appears in HF configs but this generated source does not read it; MLP Linear layers are bias-free.
- Legacy remote-code configs use `model_type: "youtu_llm"`, `auto_map`, `rope_theta`, `rope_scaling`, and `torch_dtype`. The inspected native source expects `model_type: "youtu"` and `rope_parameters`.
- `rope_interleave=True` changes RoPE input memory transformation through `view(..., d//2, 2).transpose(4, 3).reshape(...)`; a fused RoPE kernel must match this ordering.
- No multimodal placeholder/scatter, NHWC/NCHW, packed image/video, or processor-derived grid metadata belongs to this in-library family.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `input_ids [B,S] -> [B,S,2048]`.
- Reshape/view/transpose/contiguous around attention: `[B,S,H*D] -> [B,S,H,D] -> [B,H,S,D]`.
- Split and concat on last dimension: Q split `[128,64]`, compressed KV split `[512,64]`, projected KV split `[128,128]`, Q/K concat to 192.
- Broadcast/expand `k_rot [B,1,S,64] -> [B,16,S,64]`.
- Optional `repeat_kv` expand/reshape for future GQA/MQA.
- Last-token or indexed logits slicing via `logits_to_keep`.

Neural primitives:

- RMSNorm over last dimension for hidden width 2048, Q rank 1536, KV rank 512.
- Bias-free or config-gated Linear/GEMM:
  - `embed_tokens`: `[128256,2048]`, tied to LM head.
  - Q low-rank current path: `Linear(2048 -> 1536)` then RMSNorm then `Linear(1536 -> 3072)`.
  - Direct Q fallback if `q_lora_rank=None`: `Linear(2048 -> 3072)`.
  - KV A/MQA: `Linear(2048 -> 576)`, split into 512 latent and 64 RoPE key.
  - KV B: `Linear(512 -> 4096)`, split into K-nope `[16,128]` and V `[16,128]`.
  - Output projection: `Linear(2048 -> 2048)`.
  - SwiGLU MLP: gate/up `Linear(2048 -> 6144)`, `silu(gate) * up`, down `Linear(6144 -> 2048)`.
  - LM head: `Linear(2048 -> 128256)`, bias false, tied weight.
- Elementwise add residuals, multiply, SiLU, rsqrt, pow/mean for RMSNorm.

Attention primitives:

- Causal self-attention with rectangular prefill/decode masks from `create_causal_mask`.
- QK matmul `[B,16,Q,192] x [B,16,K,192]`, scale by `192 ** -0.5` plus optional YaRN mscale for non-default RoPE.
- Softmax in float32 then cast back.
- Attention-value matmul with V width 128, or padded width 192 for FlashAttention compatibility.
- KV cache update stores post-RoPE K and V per layer.

Position/rotary ops:

- Default RoPE frequency generation from `rope_parameters["rope_theta"]` and dim 64.
- Interleaved RoPE path is required for current configs.
- Dynamic RoPE update decorator may affect non-default RoPE variants; default 2B does not need dynamic update behavior beyond position-id-dependent cos/sin.

Generation/cache ops:

- `DynamicCache(config)` creation when `use_cache=True`.
- Position IDs default to `arange(seq) + past_seen_tokens`.
- Per-layer cache update and generation reorder support through Transformers cache classes.
- Generation config uses `eos_token_id=128001`, `pad_token_id=128001`, sampling defaults `top_k=20`, `top_p=0.95`; the HF generation config sets `use_cache=false`, while model config says `use_cache=true`, so controller policy must choose explicitly.

Preprocessing-coupled ops:

- Fast tokenizer only; model inputs are `input_ids` and `attention_mask`.
- Instruct repo chat template inserts `<|User|>`, `<|Assistant|>`, tool-call XML-like spans, optional `<think>` sections; this is tokenizer/controller ABI, not neural graph.

Quantized/packed, multimodal, sparse/local, state-space, distributed ops:

- No native quantized or packed weight format in source.
- No multimodal scatter/indexed update in source.
- No sliding-window/local/block attention in source.
- Tensor-parallel plan metadata exists for MLP and LM head, but source forward is single-process PyTorch.

## 5. Layer/block breakdown

Decoder block, repeated 32 times for current 2B:

```text
x: [B,S,2048]
r = x
x = RMSNorm_2048(x)

q_latent = Linear(2048 -> 1536, bias=attention_bias)(x)
q = Linear(1536 -> 3072, bias=false)(RMSNorm_1536(q_latent))
q = view [B,S,16,192] -> transpose [B,16,S,192]
q_nope, q_rot = split(q, [128,64])

compressed_kv = Linear(2048 -> 576, bias=attention_bias)(x)
kv_latent, k_rot = split(compressed_kv, [512,64])
kv = Linear(512 -> 4096, bias=false)(RMSNorm_512(kv_latent))
kv = view [B,S,16,256] -> transpose [B,16,S,256]
k_nope, v = split(kv, [128,128])
k_rot = view [B,1,S,64]
q_rot, k_rot = interleaved RoPE(q_rot, k_rot, cos, sin)
k_rot = expand to [B,16,S,64]
q = concat(q_nope, q_rot) -> [B,16,S,192]
k = concat(k_nope, k_rot) -> [B,16,S,192]
k,v = cache.update(k,v,layer) if cache is present
attn = causal_attention(q,k,v, scale=192^-0.5)
attn = reshape [B,S,2048]
x = r + Linear(2048 -> 2048, bias=attention_bias)(attn)

r = x
x = RMSNorm_2048(x)
mlp = Linear(6144 -> 2048, bias=false)(silu(Linear(2048 -> 6144)(x)) * Linear(2048 -> 6144)(x))
x = r + mlp
```

Model/head:

```text
input_ids -> embedding [B,S,2048]
position_ids default from cache length
cos,sin = rotary_emb(hidden_states, position_ids)
32 decoder blocks
final RMSNorm_2048
logits = tied lm_head(hidden_states[:, logits_to_keep, :]) -> [B,S_keep,128256]
```

## 6. Attention requirements

- Type: autoregressive causal self-attention only.
- Dense/sparse: dense full causal attention; no source sliding window, block sparsity, local/global pattern, or cross-attention.
- Head structure: MHA in current 2B (`16 Q heads`, `16 KV heads`), source supports GQA by repeating KV heads if `num_key_value_heads < num_attention_heads`.
- Projection widths: Q/K score width 192 per head, V width 128 per head, output width 2048.
- Query length vs key length: prefill uses `Q=K=S`; decode uses `Q=1`, `K=past+1` through cache.
- Masking: `create_causal_mask` receives attention mask, position IDs, input embeddings, and cache. DinoML should preserve Transformers mask semantics before replacing with optimized attention.
- RoPE/cache ordering: K stored in cache after concatenating RoPE and non-RoPE key slices, so cached keys are post-position-encoding.
- FlashAttention/SDPA/Flex: source advertises all three. FlashAttention has a special value padding path when `qk_head_dim != v_head_dim`; eager attention does not pad V.
- Softmax order in eager path: matmul, scale, add mask, float32 softmax, cast to query dtype, dropout, matmul with V.
- Cache tensor shapes for current 2B after update: per layer K `[B,16,T,192]`, V `[B,16,T,128]` for eager/SDPA-style cache. FlashAttention transiently pads V to 192 only around backend call.

## 7. Position encoding and custom math

Default current configs use RoPE with theta 1,600,000 and rotary dim 64. The generated config sets `head_dim = qk_rope_head_dim`, so frequency tables are length 64, not 192.

Important source-equivalent snippets:

```python
def youtu_default_inv_freq(theta: float, rope_dim: int = 64):
    return 1.0 / (theta ** (arange(0, rope_dim, 2).float() / rope_dim))

def youtu_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat([-x2, x1], dim=-1)

def youtu_interleaved_rope(q, k, cos, sin):
    # q,k are [B,H,S,64]; cos/sin are [B,S,64].
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q = q.view(B, H, S, 32, 2).transpose(4, 3).reshape(B, H, S, 64)
    k = k.view(B, H_or_1, S, 32, 2).transpose(4, 3).reshape(B, H_or_1, S, 64)
    return q * cos + youtu_rotate_half(q) * sin, k * cos + youtu_rotate_half(k) * sin
```

Cos/sin can be precomputed by maximum position bucket for default RoPE, but the source computes them per forward from `position_ids` and casts to input dtype. Non-default `rope_parameters` should be gated until DinoML has parity for `ROPE_INIT_FUNCTIONS`, dynamic RoPE update, and YaRN mscale scaling.

## 8. Preprocessing and input packing

The neural model consumes either exactly one of:

- `input_ids [B,S]` integer tokens, plus optional `attention_mask`.
- `inputs_embeds [B,S,2048]`, plus optional `attention_mask`.

If `position_ids` is absent, source builds `[1,S]` arange offset by cache length. If `use_cache=True` and no cache is supplied, source creates `DynamicCache`.

Tokenizer/controller facts from HF repo metadata:

- Tokenizer class: `PreTrainedTokenizerFast`.
- Model input names: `input_ids`, `attention_mask`.
- `model_max_length`: 131072.
- `truncation_side`: left.
- BOS/EOS: `<|begin_of_text|>` id 128000, `<|end_of_text|>` id 128001.
- Pad token in tokenizer/generation config is EOS id 128001, while model config `pad_token_id` is null.
- Instruct chat template is a data-pipeline template with role tags and tool-call markup; no model-side scatter or modality placeholder is used.

No image/audio/video decode, frame sampling, OCR/layout boxes, multimodal placeholder expansion, packed sequence descriptors, or `cu_seqlens`-style metadata is part of this in-library target.

## 9. Graph rewrite / lowering opportunities

### Rewrite: low-rank Q projection fusion

Source pattern:

```text
q_b_proj(RMSNorm(q_a_proj(x)))
```

Replacement pattern:

```text
GEMM(2048 -> q_lora_rank) -> fused RMSNorm_1536 -> GEMM(q_lora_rank -> heads*qk_head_dim)
```

Preconditions:

- `q_lora_rank` is not `None`.
- Q A bias handling matches `attention_bias`.
- RMSNorm epsilon equals config `rms_norm_eps`.

Failure cases: direct Q projection path when `q_lora_rank=None`; nonstandard rank; bias-enabled configs without bias parity.

Parity test sketch: random `[B,S,2048]`, compare low-rank Q states before Q split and after RoPE for fp32/bf16.

### Rewrite: compressed KV projection fusion

Source pattern:

```text
compressed = kv_a_proj_with_mqa(x)
kv_latent, k_rot = split(compressed, [kv_lora_rank, qk_rope_head_dim])
kv = kv_b_proj(RMSNorm(kv_latent))
k_nope, v = split(view(kv), [qk_nope_head_dim, v_head_dim])
```

Replacement pattern:

```text
single GEMM for compressed KV -> split
RMSNorm_512 + GEMM(512 -> 4096) -> split into K-nope and V
```

Preconditions: split sizes exactly match config; `kv_lora_rank=512`, `qk_rope_head_dim=64` for first optimized kernel; preserve `k_rot` from the A projection, not B projection.

Failure cases: future configs with different KV head count/ranks; bias-enabled A projection.

### Rewrite: interleaved RoPE canonicalization

Source pattern:

```text
view [B,H,S,d//2,2] -> transpose last two dims -> reshape [B,H,S,d] -> rotate_half RoPE
```

Replacement pattern:

```text
InterleavedRoPE(q_rot, k_rot, cos, sin)
```

Preconditions: `rope_interleave=True`, `qk_rope_head_dim` even, contiguous last dimension or an accessor-aware kernel that exactly matches source flatten order.

Failure cases: `rope_interleave=False`, non-default RoPE variants without parity, attempting to reuse standard half-split RoPE kernel.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement pattern:

```text
select last token or caller-provided token rows -> GEMM(2048 -> vocab)
```

Preconditions: `logits_to_keep` is integer 1 for decode or an explicit static/validated index tensor; output shape is `[B,S_keep,128256]`.

Failure cases: training loss path, arbitrary dynamic tensor indices before gather parity.

### Layout rewrite notes

No NCHW/NHWC rewrite applies. Attention layout rewrites are sequence/head layout only:

- Source semantic hidden layout is `[B,S,H]`; attention kernels may use `[B,H,S,D]`.
- A layout pass may remove adjacent `transpose(1,2).contiguous().reshape` only inside a fully controlled attention region.
- Guard against rewriting split axes: Q/K/V split is always last dimension before or after head reshape, not channel axis from vision layouts.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm for widths 2048, 1536, 512: used multiple times per block and directly on low-rank attention paths.
- Low-rank Q A/RMSNorm/B and KV A/RMSNorm/B scheduling: MLA-style projection dominates graph shape and temporary traffic.
- Interleaved RoPE + Q/K concat: current RoPE path has extra view/transpose/reshape and must be parity-exact.
- Dense causal attention with QK 192 and V 128, including FlashAttention-compatible value padding/slicing.
- SwiGLU MLP: two `2048 -> 6144` GEMMs plus SiLU/multiply/down projection.
- Last-token LM head for decode: avoid full-sequence vocab projection.

Medium priority:

- Fused residual add around attention and MLP.
- Cache update/read kernels with post-RoPE K and narrower V shape.
- Static cos/sin table materialization for default theta 1,600,000 and 128k context.
- GQA repeat-free backend path, even though current 2B is MHA; this protects future configs.

Lower priority:

- Non-default RoPE/YaRN variants from generic Transformers hooks.
- Bias-enabled attention projection variants.
- Tensor-parallel plan consumption.
- Training loss and labels.

## 11. Runtime staging plan

Stage 1: parse native `youtu` config, reject legacy `youtu_llm` unless migrated, load tied embedding/LM head and one decoder block.

Stage 2: single-block fp32 parity for RMSNorm, low-rank Q/KV projections, interleaved RoPE, and eager attention without cache.

Stage 3: full prefill parity for `YoutuModel` and `YoutuForCausalLM` with `logits_to_keep=0` and then `logits_to_keep=1`.

Stage 4: decode parity with DynamicCache-equivalent per-layer K `[B,16,T,192]` and V `[B,16,T,128]`.

Stage 5: optimized attention backend with explicit admission for QK width 192, V width 128, causal mask, and default RoPE.

Stage 6: production fusions: RMSNorm, SwiGLU, Q/KV projection scheduling, RoPE, last-token LM head.

Stage 7: generation-controller integration for tokenizer chat template, EOS/PAD handling, sampling settings, and batching.

Initially stub/defer: labels/loss, attentions/hidden-state recording, gradient checkpointing, tensor parallel, non-default RoPE, remote-code `youtu_llm`, and Youtu-VL/Parsing.

## 12. Parity and validation plan

- Unit parity for `YoutuRMSNorm` widths 512, 1536, 2048; fp32 tolerance `1e-5`, bf16 tolerance roughly `2e-2` relative/absolute depending on accumulation.
- Unit parity for interleaved RoPE against source for random Q/K `[B,16,S,64]` and `[B,1,S,64]`, including nonzero position offsets.
- Projection parity for Q path and KV path separately; compare split tensors `q_nope`, `q_rot`, `k_nope`, `k_rot`, `value_states`.
- Attention parity eager path with small `[B=2,S=8]`, causal masks, no cache; compare attention output before `o_proj`.
- One-block parity with random weights and hidden states.
- N-layer smoke parity for 2-4 layers before full 32-layer checkpoint.
- Full prefill logits parity on public 2B checkpoint with short prompts, `logits_to_keep=1` and full logits.
- Decode parity: prefill N tokens, decode one token with cache, compare logits and cache shapes.
- End-to-end tokenizer/controller parity for base prompt and instruct chat template, but keep sampling nondeterminism out of graph parity by comparing logits/greedy next token.

## 13. Performance probes

- Tokenizer/chat-template throughput for long prompts near 128k.
- Prefill throughput sweep: `S = 128, 512, 2048, 8192, 32768`, batch 1 and throughput batches.
- Decode tokens/sec sweep with cache lengths `T = 128, 2048, 8192, 32768, 131072`.
- Attention backend comparison for QK 192/V 128: eager reference, SDPA, FlashAttention-style padded V path.
- KV cache memory probe: per token per layer current 2B stores `16*(192+128)` elements; BF16 gives about 10,240 bytes/token across 32 layers.
- Projection/MLP GEMM profiling: Q low-rank, KV compressed, MLP gate/up/down, LM head last-token.
- Last-token versus full-sequence logits projection benchmark.
- RoPE implementation benchmark: source-equivalent interleaved reshape path versus fused kernel.
- Weight-load and tied-embedding alias validation; ensure LM head does not duplicate tied storage unnecessarily.

## 14. Skip/defer list

- Training loss and labels.
- Gradient checkpointing.
- Attention/hidden-state output recording.
- Beam search, speculative decoding, and advanced generation processors beyond EOS/PAD and sampling defaults.
- Tensor-parallel/distributed execution plans.
- Non-default/dynamic RoPE and YaRN mscale variants.
- Bias-enabled attention variants until a representative checkpoint requires them.
- Legacy remote-code `model_type: "youtu_llm"` unless explicitly migrated.
- Youtu-VL, Youtu-Parsing, image/video tokens, multimodal processors, placeholder scatter.
- Quantized or packed weight loading; source has no such format.

## 15. Final implementation checklist

- [ ] Parse `YoutuConfig` native `model_type: "youtu"` and normalize `rope_parameters`.
- [ ] Reject or migrate legacy `model_type: "youtu_llm"` configs with `rope_theta`/`rope_scaling`.
- [ ] Load tied `embed_tokens.weight` / `lm_head.weight` as one logical parameter.
- [ ] Implement RMSNorm for widths 2048, 1536, and 512.
- [ ] Implement low-rank Q projection path and direct-Q fallback.
- [ ] Implement compressed KV A/RMSNorm/B projection path and exact split shapes.
- [ ] Implement interleaved RoPE with theta 1,600,000 and rotary dim 64.
- [ ] Implement causal self-attention with QK head dim 192 and V head dim 128.
- [ ] Implement KV cache shape contract K `[B,16,T,192]`, V `[B,16,T,128]`.
- [ ] Implement FlashAttention admission or fallback for `qk_head_dim != v_head_dim`.
- [ ] Implement SwiGLU MLP `2048 -> 6144 -> 2048`.
- [ ] Implement `logits_to_keep` last-token/indexed LM head path.
- [ ] Add single-op parity tests for RMSNorm, RoPE, Q/KV projection, attention.
- [ ] Add one-block, prefill, and decode-cache parity tests against Transformers.
- [ ] Benchmark prefill, decode, RoPE, attention backend, MLP, and last-token logits.
