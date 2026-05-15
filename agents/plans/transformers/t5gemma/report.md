# DinoML Transformers Audit: t5gemma

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in local checkout `transformers`.

Model id: primary examples are `google/t5gemma-2b-2b-prefixlm-it`, `google/t5gemma-2b-2b-ul2`, `google/t5gemma-9b-2b-ul2`, `google/t5gemma-9b-9b-ul2`, plus T5-sized `s/b/l/ml/xl` variants. First DinoML target for this report: `T5GemmaForConditionalGeneration` text-to-text generation.

Config source:

- Local source defaults: `src/transformers/models/t5gemma/configuration_t5gemma.py`.
- Official Google `config.json` files are license-gated: raw fetches for representative Google repos returned `401 Unauthorized`. HF model API metadata was accessible and confirmed `model_type="t5gemma"`, `architectures=["T5GemmaForConditionalGeneration"]`, gated status, tokenizer metadata, and safetensors parameter counts.
- Open mirror/raw configs inspected:
  - `https://huggingface.co/deb-cmd/t5gemma-simplifier/resolve/main/config.json` (`model_type="t5gemma"`, full encoder/decoder config, T5-sized 768 hidden variant; mirror, not official Google).
  - `https://huggingface.co/RE-N-Y/t5gemma-2b-2b-prefixlm-it/resolve/main/config.json` and `.../t5gemma-2b-2b-ul2-it/...` (`model_type="t5_gemma_module"`, module-only 2B-style config; mirror, not official Google).

Source files inspected:

- `src/transformers/models/t5gemma/modular_t5gemma.py` is the authoritative source for future edits.
- `src/transformers/models/t5gemma/modeling_t5gemma.py` is generated from the modular file and is the concrete runtime surface inspected.
- `src/transformers/models/t5gemma/configuration_t5gemma.py` is generated from the modular file and defines config defaults.
- `docs/source/en/model_doc/t5gemma.md` for official family description and published checkpoint families.
- `tests/models/t5gemma/test_modeling_t5gemma.py` for source-supported asymmetric encoder/decoder behavior, SDPA equivalence coverage, generation/cache cases, and encoder-only head coverage.

Any missing files or assumptions:

- No `processing_*` or tokenizer source exists under `models/t5gemma`; tokenizer is routed through Gemma tokenizer auto mappings. Chat templates and tokenizer configs are checkpoint artifacts, not model code.
- Official raw configs would resolve after accepting the Gemma license on Hugging Face. Until then, checkpoint dimension sweeps below distinguish source defaults, open mirrors, HF API metadata, and inference from naming/docs.
- No DinoML tests were run, per task scope.

## 2. High-level architecture

T5Gemma is a text-only encoder-decoder Transformer: T5-style encoder/decoder topology with Gemma 2-style RMSNorm, RoPE, gated MLP, grouped-query attention, attention logit softcapping, final logit softcapping, and alternating sliding/full self-attention layers.

Dataflow:

```text
tokenizer/chat template -> encoder token embeddings -> encoder blocks -> encoder hidden states
decoder input ids / shifted labels -> decoder token embeddings -> decoder self-attn + cross-attn blocks
-> final decoder norm -> LM head -> optional final logit softcap -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: tokenization, chat template, input/decoder input construction, right-shift labels for training-style calls, generation control.
- Encoder: independently cacheable for fixed source prompt; outputs `[B, Senc, Henc]`.
- Decoder prefill: causal/sliding self-attention over decoder prompt plus cross-attention over encoder hidden states.
- Decode: decoder self-attention KV cache grows; cross-attention K/V can be computed once per layer and reused through `EncoderDecoderCache`.
- Heads: conditional generation LM head is required for the primary target. Sequence and token classification heads are optional/deferred.

## 3. Important config dimensions

Source-default `T5GemmaModuleConfig`:

| Field | Default / source behavior |
|---|---:|
| `vocab_size` | 256000 |
| `hidden_size` | 2304 |
| `intermediate_size` | 9216 |
| `num_hidden_layers` | 26 per module |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | 4 |
| `head_dim` | 256 |
| Q/O attention width | `num_attention_heads * head_dim = 2048`, not `hidden_size` |
| K/V width | `num_key_value_heads * head_dim = 1024` |
| MLP activation | `gelu_pytorch_tanh` |
| norm eps | `1e-6` |
| max positions | 8192 |
| RoPE | default RoPE, `rope_theta=10000.0` in inspected configs |
| attention scale | `query_pre_attn_scalar ** -0.5`; default `1 / sqrt(256)` |
| sliding window | 4096 on `sliding_attention` layers |
| layer pattern | odd-numbered layers by 1-based index are sliding, even are full |
| attention bias | false |
| dropout / attention dropout | 0.0 defaults |
| cache | decoder `use_cache=True`; encoder self-attn has no cache |
| final logit softcap | 30.0 |
| attention logit softcap | 50.0 |
| token IDs | pad 0, eos 1, bos 2 by source default |
| tied LM weights | config default true; tied key is LM head to decoder embeddings |

Representative checkpoint/config sweep:

| Checkpoint | Source | Status | Architecture | Key dimensions / metadata |
|---|---|---|---|---|
| `google/t5gemma-s-s-prefixlm` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 safetensors total 312,517,632 params; docs describe T5-sized small family. |
| `google/t5gemma-b-b-prefixlm` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 total 591,490,560 params. |
| `google/t5gemma-ml-ml-prefixlm` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 total 2,200,345,344 params. |
| `google/t5gemma-xl-xl-prefixlm` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 total 3,766,980,608 params; sharded safetensors. |
| `google/t5gemma-2b-2b-prefixlm-it` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | conversational tag/chat template metadata; sharded BF16 weights. |
| `google/t5gemma-9b-2b-ul2` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 total 12,292,375,296 params; asymmetric size by model name/docs. |
| `google/t5gemma-9b-9b-ul2` | HF API | gated raw config | `T5GemmaForConditionalGeneration` | BF16 total 20,333,401,088 params; sharded safetensors. |
| `deb-cmd/t5gemma-simplifier` | open mirror config | accessible, non-official | `T5GemmaForConditionalGeneration` | encoder and decoder both `H=768`, `I=2048`, `L=12`, `A=12`, `KV=12`, `Dh=64`, `Smax=8192`, sliding window 4096, eos `[1,107]`. |
| `RE-N-Y/t5gemma-2b-2b-prefixlm-it` | open mirror config | module-only, non-official | `T5GemmaEncoder` in config | module `H=2304`, `I=9216`, `L=26`, `A=8`, `KV=4`, `Dh=256`, `Smax=8192`, BF16. Treat as module evidence, not full seq2seq config. |

## 3a. Family variation traps

- `hidden_size` is not necessarily `num_attention_heads * head_dim`. The 2B-style module has `hidden_size=2304` and attention Q/O width `2048`; lowering must use explicit `head_dim`.
- Encoder and decoder can be asymmetric. Source config sets `decoder.cross_attention_hidden_size = encoder.hidden_size`; cross-attention K/V read encoder width but emit decoder KV width.
- `num_key_value_heads` may be less than `num_attention_heads` (GQA) or equal (MHA). `repeat_kv` is required for eager attention unless using a backend that natively supports GQA.
- Layer type alternates `sliding_attention` and `full_attention`. Sliding attention is used in both encoder bidirectional masks and decoder causal masks.
- Attention has two nonstandard softcaps: tanh softcap on attention scores before mask addition, and tanh softcap on final logits after LM head.
- Attention scaling is `query_pre_attn_scalar ** -0.5`, not necessarily `head_dim ** -0.5`.
- Source supports FlashAttention/SDPA/Flex attention dispatch, but Transformers tests skip FA2 numerical equivalence because T5Gemma eager/FA2 outputs are expected to differ. DinoML parity should start with eager math.
- Raw Google configs are gated; do not infer exact per-size dimensions from HF API parameter counts alone.
- No NCHW/NHWC tensors are present. Layout translation work is limited to sequence/head reshapes and transposes; protect axis-sensitive sequence/head ops from image-style layout passes.
- Configs may serialize legacy `rope_theta` while generated modeling reads normalized `config.rope_parameters["rope_type"]` and `["rope_theta"]`; config loading must normalize this before runtime.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: encoder and decoder `Embedding(vocab_size, hidden_size)` with padding index.
- Scalar multiply embedding output by `sqrt(hidden_size)`.
- Shape/view ops: `view(..., -1, head_dim)`, `transpose(1,2)`, `transpose(2,3)`, `reshape`, `contiguous`.
- `torch.arange`, `unsqueeze`, `expand`, slice/index for `position_ids`, `logits_to_keep`, and classification pooling.
- Concatenate on last dimension for RoPE frequency duplication and `rotate_half`.
- Mask construction from `input_ids != pad_token_id`, plus causal, bidirectional, sliding causal, and bidirectional sliding masks.

Neural network primitives:

- RMSNorm over last dimension with fp32 internal math and weight represented as `(1 + weight)`.
- Dense linears:
  - Self-attn Q: `Linear(H -> A * Dh, bias=attention_bias)`.
  - Self-attn K/V: `Linear(H -> KV * Dh, bias=attention_bias)`.
  - Self-attn O: `Linear(A * Dh -> H, bias=attention_bias)`.
  - Cross-attn Q: `Linear(Hdec -> Adec * Dhdec)`.
  - Cross-attn K/V: `Linear(Henc -> KVdec * Dhdec)`.
  - Cross-attn O: `Linear(Adec * Dhdec -> Hdec)`.
  - MLP gate/up/down: `H -> I`, `H -> I`, `I -> H`, all bias-free.
  - LM head: `Hdec -> vocab_size`, bias false by default.
  - Classification head: dropout then `H -> num_labels`.
- `gelu_pytorch_tanh`, multiply for gated MLP, residual adds, dropout as identity in inference.
- `tanh` softcap for attention scores and logits.

Attention primitives:

- Encoder self-attention: bidirectional full or bidirectional sliding window.
- Decoder self-attention: causal full or causal sliding window with growing self-attn KV cache.
- Decoder cross-attention: bidirectional attention over encoder hidden states, no sliding window.
- MHA/GQA with explicit `repeat_kv` in eager path.
- Softmax over last attention dimension with fp32 accumulation, then cast to query dtype.
- Matmul QK^T and probability-V.

Position/rotary ops:

- RoPE cos/sin generation from `position_ids`, `inv_freq`, fp32 matmul, cos, sin, dtype cast.
- `rotate_half`: split last dim in halves, concatenate `[-x2, x1]`.
- Apply RoPE to Q and K after projection and before cache update.

Generation/cache ops:

- `EncoderDecoderCache` with self-attention `DynamicCache(config=decoder_config)` and cross-attention `DynamicCache()` when generating.
- Per-layer self-cache update stores post-RoPE K and V.
- Cross-cache update computes K/V from encoder hidden states once per layer; `is_updated[layer_idx]` gates reuse.
- `logits_to_keep` can be int slice on sequence axis or a tensor index.
- Label path uses `_shift_right`: prepend decoder BOS, shift labels right, replace `-100` with decoder pad.

Preprocessing-coupled ops:

- Tokenizer/chat template is outside model graph. Instruction checkpoints may reject system role via chat template metadata, but neural graph only sees token IDs and masks.
- No image/audio/video preprocessing, no placeholder scatter, no NHWC/NCHW processing.

Optional/deferred heads:

- `T5GemmaEncoderModel` encoder-only feature extraction.
- `T5GemmaForSequenceClassification`: last non-pad token pooling with `argmax` over token positions and optional encoder-decoder right-shift offset.
- `T5GemmaForTokenClassification`: per-token linear scores.

Parameter sharing:

- LM head weight is tied to `model.decoder.embed_tokens.weight` when `tie_word_embeddings=True`. Encoder embeddings are separate from decoder embeddings unless a checkpoint explicitly aliases them; do not assume one shared encoder/decoder table.

## 5. Layer/block breakdown

Encoder module, repeated `Lenc` times:

```text
x: [B, Senc, Henc]
residual = x
x = RMSNorm_Henc(x)
q = Linear(Henc -> Aenc*Dhenc)(x).view(B,Senc,Aenc,Dhenc).transpose(1,2)
k = Linear(Henc -> KVenc*Dhenc)(x).view(B,Senc,KVenc,Dhenc).transpose(1,2)
v = Linear(Henc -> KVenc*Dhenc)(x).view(B,Senc,KVenc,Dhenc).transpose(1,2)
q,k = RoPE(q,k, encoder position_ids)
x = bidirectional attention(q,k,v, full_or_sliding_mask)
x = Linear(Aenc*Dhenc -> Henc)(x)
x = RMSNorm_Henc(x)
x = residual + x
residual = x
x = RMSNorm_Henc(x)
x = Linear(Ienc -> Henc)(gelu_tanh(Linear(Henc -> Ienc)(x)) * Linear(Henc -> Ienc)(x))
x = RMSNorm_Henc(x)
x = residual + x
```

Decoder module, repeated `Ldec` times:

```text
x: [B, Sdec, Hdec]
residual = x
x = RMSNorm_Hdec(x)
q,k,v = decoder self-attn projections; q width Adec*Dhdec, k/v width KVdec*Dhdec
q,k = RoPE(q,k, decoder position_ids)
k,v = self_cache.update(k,v, layer_idx) when cache is present
x = causal attention(q,k,v, full_or_sliding_mask)
x = Linear(Adec*Dhdec -> Hdec)(x)
x = RMSNorm_Hdec(x)
x = residual + x

residual = x
x = RMSNorm_Hdec(x)
q = Linear(Hdec -> Adec*Dhdec)(x)
k,v = Linear(Henc -> KVdec*Dhdec)(encoder_hidden_states), cached once per layer
x = noncausal cross-attention(q,k,v, encoder mask)
x = Linear(Adec*Dhdec -> Hdec)(x)
x = RMSNorm_Hdec(x)
x = residual + x

residual = x
x = RMSNorm_Hdec(x)
x = gated MLP Hdec -> Idec -> Hdec
x = RMSNorm_Hdec(x)
x = residual + x
```

Module entry/exit:

- Encoder and decoder both multiply embeddings by `sqrt(hidden_size)`, apply dropout, compute one shared RoPE `(cos,sin)` per module call, then apply final RMSNorm and dropout.
- Conditional generation applies LM head to `hidden_states[:, slice_indices, :]` and final logit softcap if configured.

## 6. Attention requirements

Required variants:

- Encoder bidirectional self-attention, full and sliding-window.
- Decoder causal self-attention, full and sliding-window, with cache.
- Decoder encoder-decoder cross-attention, bidirectional full attention, with independently reusable cross K/V cache.

Shapes:

- Q: `[B, A, Tq, Dh]`.
- K/V before repeat: `[B, KV, Tk, Dh]`.
- Eager K/V after repeat: `[B, A, Tk, Dh]`, where `A / KV` is integer.
- Attention weights: `[B, A, Tq, Tk]`.
- Output before O projection: `[B, Tq, A*Dh]`.

Masking:

- Encoder default mask is `[B,Senc]` from non-pad tokens or all ones. It is converted to full bidirectional and sliding bidirectional masks.
- Decoder self mask is causal or sliding causal, considers `past_key_values.self_attention_cache` and `position_ids`.
- Cross mask uses encoder attention mask and encoder hidden states; no sliding variant.
- Eager attention adds the mask after tanh softcap.

Backend compatibility:

- Source advertises FlashAttention, SDPA, Flex attention, and a kernelized RoPE hook.
- DinoML first parity should implement eager order exactly: repeat K/V if needed, QK matmul, scale, optional softcap, mask add, fp32 softmax on `dim=-1`, cast, PV matmul.
- Optimized attention should preserve attention softcap placement and custom scaling. Sliding-window admission should require a backend that matches Transformers mask semantics.

Cache ABI:

- Self-attention cache per decoder layer stores post-RoPE K/V with shape `[B, KVdec, past_len, Dhdec]`.
- Cross-attention cache per decoder layer stores projected encoder K/V with shape `[B, KVdec, Senc, Dhdec]`; `is_updated[layer_idx]` prevents recomputation.
- Cached K is stored after RoPE only for decoder self-attention; cross-attention does not apply RoPE in this source.

## 7. Position encoding and custom math

RoPE generation:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

RoPE apply:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_t5gemma_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

RMSNorm:

```python
def t5gemma_rmsnorm(x, weight, eps):
    y = x.float() * rsqrt(mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(dtype=x.dtype)
```

Softcaps:

```python
attn_scores = tanh(attn_scores / attn_logit_softcapping) * attn_logit_softcapping
logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping
```

Precompute opportunities:

- `inv_freq` is a non-persistent buffer derived from config and can be compiled as a constant.
- Cos/sin depend on runtime `position_ids`, past length, and sequence length; cache or specialize by decode position bucket.

## 8. Preprocessing and input packing

Text inputs:

- `input_ids`: `[B,Senc]`, int token IDs.
- `attention_mask`: optional `[B,Senc]`, 1 for valid tokens.
- `decoder_input_ids`: `[B,Sdec]`; if labels are passed without decoder IDs, source right-shifts labels.
- `decoder_attention_mask`: optional `[B,Sdec]`.
- `position_ids` and `decoder_position_ids`: optional; defaults are arange with decoder past offset.

Chat/instruction checkpoints:

- HF API metadata for `google/t5gemma-2b-2b-prefixlm-it` includes a Gemma-style chat template with `<start_of_turn>user`, `<start_of_turn>model`, and no system role. This is tokenizer/controller ABI, not neural graph ABI.

No multimodal packing:

- No `pixel_values`, image grid metadata, audio features, placeholder tokens, masked scatter, codebook tokens, or cu-seqlens style packed sequence descriptors are present in `t5gemma`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: QKV projection grouping for self-attention

Source pattern:

```text
q = Linear(H -> A*Dh)
k = Linear(H -> KV*Dh)
v = Linear(H -> KV*Dh)
view/transpose each to [B, heads, T, Dh]
```

Replacement:

```text
Grouped/fused GEMM producing packed [q, k, v] outputs, then split into Q/K/V views.
```

Preconditions:

- Same input tensor, same dtype, same batch/sequence dimensions.
- Bias configs match. Source default has no bias.
- Packed layout must preserve split order `q`, `k`, `v` and unequal output widths.
- `A*Dh` may differ from `H`; do not infer packed width from hidden size.

Failure cases:

- Runtime weight aliasing or tensor-parallel sharding that expects separate parameters.
- Configs with attention bias need bias packing support.

Parity test sketch:

- Compare split Q/K/V tensors before RoPE for source-default 2B-like shape and small mirror `H=768,A=12,KV=12,Dh=64`.

### Rewrite: Gated MLP fusion

Source pattern:

```text
gelu_pytorch_tanh(gate_proj(x)) * up_proj(x) -> down_proj
```

Replacement:

```text
two GEMMs -> fused gelu_tanh/mul -> GEMM, or packed gate/up GEMM -> fused activation multiply -> down GEMM
```

Preconditions:

- Both projections consume same normalized `x`.
- Activation is exactly `gelu_pytorch_tanh`.
- Bias-free for inspected configs.

Parity test sketch:

- Random bf16/fp16/fp32 tensor tests against PyTorch over `[B,T,H]`.

### Rewrite: eager GQA attention to fused attention

Source pattern:

```text
repeat_kv(k/v) -> QK^T -> scale -> tanh softcap -> mask add -> softmax fp32 -> PV
```

Replacement:

```text
GQA-capable fused attention backend with native KV heads, softcap, masks, and cache.
```

Preconditions:

- Backend supports GQA without materialized repeat.
- Backend supports attention score softcap before mask addition.
- Sliding-window and full layers are admitted separately.
- Decoder cache layout `[B,KV,T,Dh]` is accepted or transposed once at cache boundary.

Failure cases:

- Backend only supports standard scale/mask/softmax order.
- Sliding-window semantics diverge from Transformers masking utilities.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden_states[:, -logits_to_keep:, :] -> LMHead -> softcap
```

Replacement:

```text
for decode, gather final token hidden state before LM GEMM; compute only needed vocab logits.
```

Preconditions:

- `logits_to_keep` is int `1` or equivalent final-position tensor.
- Generation controller does not require full-sequence logits.

### Layout guard notes

There is no image/channel layout. Sequence tensor semantic layout is `[B,T,H]`; attention internal layout is `[B,heads,T,Dh]`. Guard or explicitly rewrite axes for:

- RMSNorm and softmax over `dim=-1`.
- RoPE split/concat over last dim.
- `transpose(1,2)` between sequence and head axes.
- `transpose(2,3)` for K in QK matmul.
- Classification pooling over sequence axis.
- `hidden_states[:, slice_indices, :]` logits slicing over decoder sequence axis.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm fp32-accumulation kernel with `(1 + weight)` parameter convention.
- Dense GEMM coverage for projection shapes where output width differs from hidden size.
- Eager-compatible GQA full attention with attention softcap and fp32 softmax.
- Sliding-window attention masks/backends for alternating layers.
- Decoder KV cache and cross-attention cache ABI.
- Gated GELU MLP fusion.
- LM head last-token GEMM plus final logit softcap.

Medium priority:

- Fused QKV projection for self-attention with unequal Q and KV widths.
- Cross-attention K/V precompute and reuse from encoder hidden states.
- RoPE generation/apply fusion in prefill and decode.
- Mask generation kernels or controller-side mask materialization for full/sliding variants.
- Sequence classification last-non-pad pooling (`argmax` over token positions) if classification heads are admitted.

Lower priority:

- Training losses, dropout, gradient checkpointing.
- FlashAttention/Flex exact dispatch parity before eager parity is stable.
- Tensor-parallel plan support.
- Encoder-only and token-classification heads for non-generation tasks.

## 11. Runtime staging plan

Stage 1: parse `T5GemmaConfig` and nested module configs, including legacy `rope_theta` normalization and explicit `head_dim`.

Stage 2: load weights and run one encoder layer and one decoder layer parity with eager attention, no cache, small random configs.

Stage 3: encoder-only parity for `[B,Senc,Henc]`, including bidirectional full and sliding masks.

Stage 4: seq2seq prefill parity for conditional generation: encoder output, decoder self-attention, decoder cross-attention, LM head, final softcap.

Stage 5: decode with `EncoderDecoderCache`: self-cache append and cross-cache update-once behavior.

Stage 6: optimize attention and MLP: native GQA, sliding-window backend, packed QKV, fused gated MLP, last-token logits.

Stage 7: production generation: tokenizer/chat-template handoff, batching, cache memory planning, optional tensor parallel.

Initial stubs:

- Dropout as identity in eval.
- Return attentions/hidden states can be omitted for first inference target.
- Sequence/token classification heads can be deferred.
- Gated official checkpoints can be represented by locally synthesized configs until licensed configs are available.

## 12. Parity and validation plan

Recommended tests:

- Config parsing tests for default source config, open mirror `H=768` config, module-only 2B mirror config, and asymmetric encoder/decoder synthetic config.
- RMSNorm random tests in fp32/fp16/bf16; require fp32 internal math and `(1 + weight)`.
- RoPE tests for prefill and decode position offsets; compare Q/K after apply.
- Attention eager tests:
  - full bidirectional encoder,
  - sliding bidirectional encoder,
  - full causal decoder,
  - sliding causal decoder,
  - cross-attention rectangular `Tq != Senc`,
  - GQA `A > KV` and MHA `A == KV`.
- Cache tests:
  - prefill logits vs no-cache full forward,
  - one-token decode self-cache append,
  - cross-cache computes once and reuses K/V.
- Single-layer and N-layer parity against Transformers for hidden states and logits.
- End-to-end `generate` smoke once tokenizer access is available.
- Classification optional tests: rightmost non-pad pooling and encoder-decoder `+1` shift behavior.

Suggested tolerances:

- fp32 eager: `rtol=1e-5`, `atol=1e-5` for hidden states/logits.
- fp16/bf16 eager: start with `rtol=5e-2`, `atol=5e-2` for full models; tighten per-kernel where possible.
- Optimized attention should compare against eager with softcap and mask placement preserved; do not use FA2 as initial truth because source tests mark FA2/eager differences as expected.

## 13. Performance probes

- Encoder throughput sweep over `B`, `Senc`, full/sliding layer mix.
- Decoder prefill throughput over `B`, `Sdec`, `Senc`, GQA ratio, and sliding window.
- Decode tokens/sec with self-cache length sweep and fixed encoder length.
- Cross-attention cache memory and time: compute-once versus recompute.
- Attention backend comparison: eager-like, SDPA-compatible, FlashAttention-like, custom sliding.
- MLP packed gate/up fusion versus separate GEMMs.
- LM head full sequence versus last-token-only logits.
- KV cache memory: self-cache `[layers, B, KV, Tdec, Dh]` plus cross-cache `[layers, B, KV, Senc, Dh]`.
- Weight loading/dequant/provider probes for BF16 dense weights and future GGUF-style experiments if checkpoint conversion is attempted.

## 14. Skip/defer list

- Training, losses, dropout randomness, gradient checkpointing.
- Beam search internals beyond standard generation controller ABI.
- Sequence and token classification heads for first conditional-generation target.
- Returning attention tensors and all hidden states.
- Tensor parallel and pipeline parallel plans.
- FlashAttention/Flex numerical parity before eager parity.
- Raw official checkpoint sweeps until Gemma license-gated configs are accessible.
- Quantized/packed checkpoint formats; no source-owned quantized format is implemented in `t5gemma`.
- Any NHWC/NCHW layout work; this is text-only.

## 15. Final implementation checklist

- [ ] Parse `T5GemmaConfig` plus nested encoder/decoder `T5GemmaModuleConfig`.
- [ ] Normalize RoPE config from `rope_theta`/`rope_parameters`.
- [ ] Preserve explicit `head_dim`; do not infer Q/O width from hidden size.
- [ ] Load encoder embeddings, decoder embeddings, tied LM head alias, and per-layer weights.
- [ ] Implement T5Gemma RMSNorm with fp32 math and `(1 + weight)`.
- [ ] Implement RoPE generation/apply with decode position offsets.
- [ ] Implement full and sliding bidirectional encoder masks.
- [ ] Implement full and sliding causal decoder masks.
- [ ] Implement eager-compatible GQA/MHA attention with attention softcap.
- [ ] Implement cross-attention with rectangular encoder-decoder lengths.
- [ ] Implement `EncoderDecoderCache` self-cache and cross-cache update-once semantics.
- [ ] Implement gated GELU MLP.
- [ ] Implement final LM head, `logits_to_keep`, and final logit softcap.
- [ ] Add single-layer encoder/decoder parity tests.
- [ ] Add prefill logits parity tests.
- [ ] Add one-token decode parity tests.
- [ ] Add asymmetry tests for encoder hidden size feeding decoder cross-attention.
- [ ] Benchmark encoder, prefill, decode, MLP, attention, and LM head separately.
