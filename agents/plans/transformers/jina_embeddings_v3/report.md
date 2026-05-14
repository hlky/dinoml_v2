# jina_embeddings_v3 DinoML audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: jinaai/jina-embeddings-v3-hf for native source; jinaai/jina-embeddings-v3 is a remote-code predecessor.
Config source: raw Hub config snapshots saved in this folder.
Source files inspected:
- X:/H/transformers/src/transformers/models/jina_embeddings_v3/configuration_jina_embeddings_v3.py
- X:/H/transformers/src/transformers/models/jina_embeddings_v3/modeling_jina_embeddings_v3.py
- X:/H/transformers/src/transformers/models/jina_embeddings_v3/modular_jina_embeddings_v3.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/modeling_rope_utils.py
- Hub snapshots in agents/plans/transformers/jina_embeddings_v3/
Any missing files or assumptions: no gated links found. Native generated files are runtime source; modular_jina_embeddings_v3.py is authoritative for future Transformers edits.
```

Primary runtime target for this report: encoder-only text embedding feature extraction. `JinaEmbeddingsV3Model` returns token hidden states and an optional CLS pooler. The common Sentence Transformers contract adds mean pooling and L2 normalization outside the core model; task LoRA adapters are checkpoint/pipeline concerns, not implemented by the native model class.

Important scope split:

- [`jinaai/jina-embeddings-v3-hf`](https://huggingface.co/jinaai/jina-embeddings-v3-hf) uses native `model_type: jina_embeddings_v3`.
- [`jinaai/jina-embeddings-v3`](https://huggingface.co/jinaai/jina-embeddings-v3) uses `auto_map` remote code to `jinaai/xlm-roberta-flash-implementation` and is out of scope for native-source parity except as historical config evidence.
- [`jinaai/jina-embeddings-v3-small-ci`](https://huggingface.co/jinaai/jina-embeddings-v3-small-ci) is a small custom-code CI checkpoint, useful only for dimension variation.

## 2. High-level architecture

Text-only bidirectional encoder:

```text
XLM-R tokenizer / task prompt text -> input_ids, attention_mask, position_ids
-> word + token-type embeddings -> LayerNorm
-> repeated RoPE bidirectional self-attention + MLP encoder blocks
-> last_hidden_state
-> optional core CLS pooler OR external ST mean pooling + normalize
-> embedding vector / similarity matrix
```

Stage decomposition:

- CPU/data pipeline: XLM-R tokenizer, optional task instruction prefix, padding/truncation to `model_max_length=8194`.
- Core GPU graph: embedding lookup, token-type embedding, LayerNorm, RoPE construction, 24 encoder blocks, optional CLS pooler.
- External embedding postprocess: Sentence Transformers mean pooling over non-padding tokens and L2 normalize. This can be validated independently from the encoder.
- Optional task adapters: repo has LoRA adapter directories for retrieval/query/passage/classification/etc. Native source does not consume `adapter_mask` or load adapters by itself; first DinoML native target should reject adapter-routed inference or require pre-merged/fixed-task weights.

Implemented heads:

| Head | Source class | First-target status |
| --- | --- | --- |
| Base encoder | `JinaEmbeddingsV3Model` | Required |
| CLS pooler | `JinaEmbeddingsV3Pooler` | Optional; not the common ST embedding output |
| Masked LM | `JinaEmbeddingsV3ForMaskedLM` | Deferred |
| Sequence classification | `JinaEmbeddingsV3ForSequenceClassification` | Deferred |
| Token classification | `JinaEmbeddingsV3ForTokenClassification` | Deferred |
| Question answering | `JinaEmbeddingsV3ForQuestionAnswering` | Deferred |

## 3. Important config dimensions

Native source defaults from `JinaEmbeddingsV3Config`:

| Field | Default / effective value | Source note |
| --- | ---: | --- |
| `vocab_size` | 250002 | Config class default |
| `hidden_size` | 1024 | Config class default |
| `num_hidden_layers` | 24 | Config class default |
| `num_attention_heads` | 16 | MHA, no separate KV heads |
| `head_dim` | 64 | Inferred as `hidden_size // num_attention_heads` unless config supplies `head_dim` |
| Q/K/V width | 1024 each | `num_attention_heads * head_dim` |
| `intermediate_size` | 4096 | MLP `fc1` output |
| `max_position_embeddings` | 8194 | Also tokenizer model max length in HF repo |
| `type_vocab_size` | 1 | Token type embedding exists but only ID 0 for official configs |
| `hidden_act` | `gelu` | Ungated MLP |
| `layer_norm_eps` | `1e-5` | Embedding and block norms |
| RoPE | default RoPE, theta 20000.0 | Effective native default from `default_theta`; Hub configs use legacy `rotary_emb_base` |
| Dropout | 0.1 in config | Inference disables dropout |
| `torch_dtype` | bf16 in production checkpoint | Hub config metadata |
| `use_cache` / `output_past` | present/true in configs | Ignored for encoder fast path; no KV cache source path |

Representative checkpoint sweep:

| Repo/config snapshot | Native source scope | Layers | Hidden | Heads | Head dim | MLP | Max pos | Dtype | Variation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `config_jina-embeddings-v3-hf.json` | In scope | 24 | 1024 | 16 | 64 | 4096 | 8194 | bf16 | Native model type plus separate adapter dirs |
| `config_jina-embeddings-v3_remote.json` | Out of scope | 24 | 1024 | 16 | 64 | 4096 | 8194 | bf16 | Remote-code `auto_map`; LoRA/task flags |
| `config_jina-embeddings-v3-small-ci.json` | Out of scope | 2 | 256 | 2 | 128 | 1024 | 8194 | fp32 | Small custom-code CI shape |

Fields present in Hub configs but not read by native `modeling_jina_embeddings_v3.py`: `load_trained_adapters`, `lora_adaptations`, `lora_alpha`, `lora_dropout_p`, `lora_main_params_trainable`, `lora_rank`, `matryoshka_dimensions`, `truncate_dim`, `task_instructions`, `position_embedding_type`, `rotary_emb_base`, `use_flash_attn`, and `output_past`. `rotary_emb_base` is effectively superseded by native `rope_parameters`/`default_theta`; because `default_theta=20000.0`, the observed official theta still matches.

## 3a. Family variation traps

- Native `jina_embeddings_v3` is not the same execution contract as the older custom-code `jinaai/jina-embeddings-v3` repo.
- Official embedding usage is a composite pipeline: tokenizer prompt insertion, optional fixed-task LoRA, mean pooling, optional matryoshka truncation, and normalization. The native base model only emits hidden states and optional CLS pooler output.
- Config advertises `use_cache`/`output_past`, but the source is bidirectional encoder attention with no KV-cache API in the forward path.
- No GQA/MQA in native source. Q/K/V all use `num_attention_heads * head_dim`.
- `head_dim` may be explicitly present in future configs; do not infer projection width from `hidden_size` alone without checking.
- `type_vocab_size=1` means token type IDs must be all zero for official configs; pair inputs from XLM-R tokenizer should not introduce segment ID 1 unless config changes.
- RoPE cos/sin are generated from `position_ids`; custom non-monotonic or packed position IDs change both RoPE and default token-type gather behavior.
- Adapter configs target `word_embeddings`, `token_type_embeddings`, Q/K/V/O, `fc1`, `fc2`, and `dense`. DinoML can fold a fixed adapter into dense weights, but dynamic per-row adapter routing needs a separate audit.
- `matryoshka_dimensions` and `truncate_dim` are post-encoder embedding ABI, not neural graph ops in the native model.
- Layout-sensitive ops are sequence-major `[B, S, H]` with attention reshapes/transposes to `[B, heads, S, head_dim]`. A layout pass should guard attention/RoPE axes and pooling axes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding index lookup for `input_ids: int64/int32 [B,S] -> [B,S,H]`.
- Optional `inputs_embeds` passthrough.
- `arange`, `expand`, `gather`, `zeros`, broadcast add.
- Reshape/view `[B,S,H] -> [B,S,heads,head_dim]`, transpose to `[B,heads,S,head_dim]`, transpose/contiguous back.
- `split`, `squeeze`, first-token slice, mask-aware reduce for external mean pooling.

Neural primitives:

- Word embedding `[250002,1024]`; token type embedding `[1,1024]`.
- LayerNorm over last dim: embedding norm and two block norms per layer.
- Linear with bias: Q/K/V/O, MLP `fc1`/`fc2`, pooler dense, optional heads.
- GELU exact/Transformers `gelu` in MLP and MLM head.
- Tanh for pooler and sequence-classification head.
- Dropout is training-only; inference identity.

Attention primitives:

- Dense bidirectional self-attention, MHA.
- Q/K/V shape production: `Linear(1024 -> 1024)` each for production config.
- Attention score matmul `[B,16,S,64] x [B,16,64,S] -> [B,16,S,S]`.
- Additive padding mask for eager path or backend-specific SDPA/Flash/Flex mask.
- Softmax over last dim and value matmul `[B,16,S,S] x [B,16,S,64]`.

Position/rotary ops:

- Default RoPE inv frequency length `head_dim/2`.
- Float32 position-frequency matmul, concat duplicated frequencies, `cos`, `sin`, cast to hidden dtype.
- `rotate_half`: split last dim into halves, concat `[-second_half, first_half]`.

Preprocessing-coupled ops:

- XLM-R tokenizer special tokens: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`, `<mask>=250001`.
- Attention mask `[B,S]` from tokenizer enters mask creation and external mean pooling.
- Optional ST prompt prefix per task is a CPU/tokenizer ABI.

Retrieval/postprocess ops:

- Mean pooling: sum token embeddings over `attention_mask`, divide by non-pad count.
- L2 normalize over embedding dim.
- Optional matryoshka slice/truncate to dimensions `[32,64,128,256,512,768,1024]`.
- Cosine similarity is normalized dot product; output orientation depends on caller/query-passage matrix convention.

Optional/deferred head ops:

- MLM head: `Linear(1024 -> 1024)`, GELU, LayerNorm, tied decoder `Linear(1024 -> 250002)`.
- Classification head: first-token slice, dropout, `Linear(1024 -> 1024)`, tanh, dropout, `Linear(1024 -> num_labels)`.
- Token classification: dropout, `Linear(1024 -> num_labels)`.
- QA: `Linear(1024 -> num_labels)`, split start/end logits.

Tied weights:

- `JinaEmbeddingsV3ForMaskedLM` ties `lm_head.decoder.weight` to `roberta.embeddings.word_embeddings.weight` and `lm_head.decoder.bias` to `lm_head.bias`.

## 5. Layer/block breakdown

Embedding block:

```text
if input_ids:
  x = Embedding(vocab_size, H)(input_ids)
else:
  x = inputs_embeds
if token_type_ids is None:
  token_type_ids = gather(buffered_zeros, dim=1, index=position_ids).expand(B,S)
x = x + Embedding(type_vocab_size, H)(token_type_ids)
x = LayerNorm(H, eps=1e-5)(x)
```

Encoder block, repeated 24 times for production:

```text
residual = x
q = Linear(H -> heads * head_dim, bias=True)(x).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> heads * head_dim, bias=True)(x).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> heads * head_dim, bias=True)(x).view(B,S,heads,D).transpose(1,2)
q, k = RoPE(q, k, cos[B,S,D], sin[B,S,D])
attn = BidirectionalAttention(q, k, v, additive_or_backend_mask)
x = Linear(heads * D -> H, bias=True)(attn.reshape(B,S,heads*D))
x = LayerNorm(residual + x)
residual = x
x = Linear(H -> intermediate, bias=True)(x)
x = GELU(x)
x = Linear(intermediate -> H, bias=True)(x)
x = LayerNorm(residual + x)
```

Production shapes: `H=1024`, `heads=16`, `D=64`, `intermediate=4096`.

Optional core pooler:

```text
pooled = hidden_states[:, 0]
pooled = tanh(Linear(1024 -> 1024, bias=True)(pooled))
```

Common ST embedding postprocess:

```text
mask = attention_mask[..., None]
embedding = sum(last_hidden_state * mask, dim=1) / clamp(sum(mask, dim=1), min=1)
embedding = embedding / ||embedding||_2
```

## 6. Attention requirements

Required attention variant:

| Property | Value |
| --- | --- |
| Causal? | No, bidirectional encoder attention |
| Attention kind | Self-attention only |
| MHA/GQA/MQA | MHA; `num_key_value_heads == num_attention_heads` by construction |
| Production heads | 16 |
| Head dim | 64 |
| Query/key/value width | 1024 each |
| Query length vs KV length | Same sequence length for base encoder |
| Mask | Padding mask converted by `create_bidirectional_mask`; 4D masks pass through |
| Packed/varlen | No explicit native packed-sequence ABI in model forward |
| Sliding/local | None |
| RoPE | Applied to Q and K before attention |
| KV cache | Not applicable for primary target |
| Backends | Source advertises eager, SDPA, FlashAttention, Flex through Transformers attention interface |

For eager parity, preserve the order:

```text
scores = matmul(q, k.transpose(-2,-1)) * (head_dim ** -0.5)
scores = scores + attention_mask
probs = softmax(scores, dim=-1)
out = matmul(probs, v)
```

Do not describe this model as prefill/decode generation. Encoder outputs, fixed-task adapters, and pooled embeddings are independently cacheable by the application, but they are not KV caches.

## 7. Position encoding and custom math

Native RoPE default:

```python
dim = config.head_dim if present else config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (rope_theta ** (arange(0, dim, 2).float() / dim))
freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

RoPE application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

`cos`/`sin` depend on runtime `position_ids` and dtype of `x`; default monotonic `[0..S-1]` tables can be cached per sequence length, dtype, and device. Dynamic/longrope variants are possible through `rope_parameters`, but official native configs do not require them.

## 8. Preprocessing and input packing

Tokenizer ABI from `tokenizer_config_jina-embeddings-v3-hf.json`:

- Tokenizer class: `XLMRobertaTokenizer`.
- `model_max_length`: 8194.
- Special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`.
- GPU graph inputs: `input_ids [B,S]`, `attention_mask [B,S]`, optional `token_type_ids [B,S]`, optional `position_ids [B,S]`.
- Default `position_ids`: source creates `arange(seq_length)[None, :]`, broadcastable across batch.
- Default token type IDs: zeros gathered using `position_ids`; with official `type_vocab_size=1`, any nonzero token type is invalid.

Sentence Transformers repo ABI:

- `modules.json` declares `custom_st.Transformer`, `Pooling`, and `Normalize`.
- Pooling config uses mean-token pooling only, not CLS/max/sqrt-length pooling.
- `config_sentence_transformers.json` declares cosine similarity and task prompts.
- `custom_st.py` constructs `adapter_mask [B]` when a task is selected and passes it to `AutoModel`. Native `JinaEmbeddingsV3Model` does not implement adapter routing, so this is a remote/PEFT pipeline boundary.

No image/audio/video/OCR/layout inputs exist. No placeholder scatter or packed cu-seqlens metadata is required for native base inference.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed-task LoRA merge

Source pattern: external LoRA adapters for embedding and dense modules with rank 4, alpha 1, dropout 0.

Replacement: pre-merge one selected task adapter into each affected dense/embedding weight before DinoML compile.

Preconditions:

- Exactly one task adapter selected for the whole compiled artifact.
- Adapter target module names map unambiguously to loaded native weights.
- No per-row `adapter_mask` routing in a batch.
- LoRA dropout is 0.

Weight transform:

```text
W_merged = W_base + (alpha / rank) * (B @ A)
```

Failure cases: mixed task batches, missing adapter tensors, adapter target names that do not match native modules, or caller expects runtime task switching.

Parity test sketch: compare fixed task embeddings from PEFT/Transformers with pre-merged DinoML weights for identical tokenized inputs.

### Rewrite: QKV projection packing

Source pattern: three independent biased linears from `[B,S,H]` to `[B,S,H]`.

Replacement: one packed GEMM `[B*S,H] x [H,3H] + packed_bias`, then split as Q,K,V.

Preconditions:

- Same input tensor for all three projections.
- Same dtype and no intervening ops.
- Bias present for all three.
- Split order must be Q, K, V.

Weight layout: pack output rows/columns in source output order `[q_proj, k_proj, v_proj]` according to DinoML GEMM layout.

Failure cases: LoRA not merged, quantized per-module formats that cannot be packed, or debug hooks requiring individual module outputs.

### Rewrite: attention transpose/layout fusion

Source pattern:

```text
Linear -> view(B,S,heads,D) -> transpose(1,2) -> RoPE -> attention
attention -> transpose/reshape -> output projection
```

Replacement: lower projections directly into the attention backend's preferred `[B,heads,S,D]` or packed layout.

Preconditions:

- Attention backend owns Q/K/V consumers.
- RoPE axis mapping is preserved: `S` axis remains sequence, last dim remains head dim.
- No public capture of pre-transposed Q/K/V.

Layout constraints: no global NHWC-style rewrite applies; this is sequence tensor layout only. Guard RoPE `unsqueeze_dim=1` and softmax `dim=-1`.

### Rewrite: external mean pooling + normalize fusion

Source pattern:

```text
masked sum over S -> divide by mask count -> vector_norm over H -> divide
```

Replacement: one reduction kernel over sequence plus one normalization kernel, or a fused two-pass embedding kernel.

Preconditions:

- Pooling mode is mean tokens only.
- Attention mask is binary and shape `[B,S]`.
- Output dim is full hidden size before optional matryoshka truncate.

Failure cases: CLS/max/sqrt pooling, weighted pooling, or custom prompt-exclusion pooling.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,S,1024]`, repeated 49 times for base encoder including embedding norm.
- QKV packed GEMM and split; this removes three launch paths per layer.
- RoPE + attention input preparation for `[B,16,S,64]`.
- Bidirectional Flash/SDPA attention for long `S` up to 8194.
- MLP GEMM + GELU + GEMM, especially `1024 -> 4096 -> 1024`.
- Mean pooling + L2 normalize for feature-extraction parity.

Medium priority:

- Residual add + LayerNorm fusion after attention and MLP.
- Output projection + residual add when epilogue support exists.
- Optional fixed-task LoRA pre-merge or runtime low-rank add path.
- Matryoshka slice/truncate as metadata/view when output is contiguous.

Lower priority:

- CLS pooler dense+tanh.
- MLM/classification/QA heads.
- Dynamic RoPE variants not present in official native configs.

## 11. Runtime staging plan

Stage 1: parse native config and reject/route remote-code configs whose `auto_map` points to `xlm-roberta-flash-implementation`.

Stage 2: load base weights and run embedding block plus one encoder block parity with random token IDs and masks.

Stage 3: full base encoder parity for `last_hidden_state` in fp32, then bf16.

Stage 4: add external ST mean pooling and L2 normalization parity for `jinaai/jina-embeddings-v3-hf`.

Stage 5: support fixed-task LoRA by pre-merging one adapter at compile/load time, or explicitly reject adapters until a separate adapter contract lands.

Stage 6: optimize QKV packing, RoPE, attention backend, LayerNorm, and MLP fusions.

Stage 7: optional heads: pooler, masked LM, classification, token classification, QA.

Initially stub/reject: dynamic adapter routing, training losses, output attentions, hidden-state capture, remote-code original repo, and dynamic RoPE variants beyond default.

## 12. Parity and validation plan

- Config load tests: native config defaults, `v3-hf` config with legacy fields, and remote-code rejection.
- Embedding parity: token IDs with/without explicit `position_ids` and `token_type_ids`; include padding ID.
- RoPE parity: random Q/K plus known position IDs, including batch-specific position IDs.
- Single-block parity: fp32 tolerances around `1e-5` absolute/relative.
- Full encoder parity: bf16 tolerance around `5e-2` absolute for long reductions/attention, tighter for fp32.
- Mask parity: all-ones mask, padded suffix mask, and prebuilt 4D mask passthrough.
- ST embedding parity: tokenize representative query/passage/classification prompts, mean pool, normalize, compare cosine similarities.
- Optional LoRA parity: one fixed adapter merged into weights vs PEFT/runtime adapter output.
- Head parity later: MLM logits with tied decoder, sequence classification first-token head, token classification logits, QA split/squeeze outputs.

No DinoML tests were run for this audit because the user requested report-only work.

## 13. Performance probes

- Tokenizer throughput vs model throughput for max length and common short-text batches.
- Encoder-only throughput sweep over `B in {1,8,32}` and `S in {32,128,512,2048,8194}`.
- Attention backend comparison: eager decomposition, SDPA, FlashAttention-compatible dense bidirectional.
- QKV packed vs separate projection GEMMs.
- LayerNorm/residual fusion impact by layer.
- MLP GEMM/GELU/GEMM throughput and GEMM candidate profiling.
- Mean pooling + normalize cost relative to encoder.
- bf16 vs fp16 vs fp32 parity/performance.
- Memory usage sweep for `[B,heads,S,S]` eager attention vs fused attention.
- Optional fixed-task LoRA: pre-merged weights vs runtime low-rank side path.

## 14. Skip/defer list

- Training, gradient checkpointing, dropout behavior, and loss functions.
- Autoregressive generation, prefill/decode, KV cache, beam search.
- Original remote-code `jinaai/jina-embeddings-v3` execution until separately audited.
- Dynamic per-example adapter routing via `adapter_mask`.
- Multi-task adapter hot swapping inside one compiled artifact.
- Output attentions and hidden-state capture.
- Masked LM/classification/token classification/QA heads for the first embedding target.
- Dynamic/linear/yarn/longrope RoPE variants unless a native config requires them.
- ONNX artifacts and external TEI deployment wrappers.

## 15. Final implementation checklist

- [ ] Parse `JinaEmbeddingsV3Config`, including legacy Hub fields and effective default RoPE theta.
- [ ] Reject or route remote-code `auto_map` configs outside native `jina_embeddings_v3`.
- [ ] Load base encoder weights and preserve MLM tying metadata when heads are enabled.
- [ ] Implement embedding lookup, token-type embedding, LayerNorm, and default position IDs.
- [ ] Implement default RoPE cos/sin and `rotate_half` application.
- [ ] Implement bidirectional MHA with padding/4D mask parity.
- [ ] Implement encoder block residual/LN/MLP order exactly.
- [ ] Add external mean pooling and L2 normalization for embedding parity.
- [ ] Decide fixed-task LoRA admission: reject, pre-merge, or implement explicit adapter ABI.
- [ ] Add QKV packing rewrite with Q,K,V split-order tests.
- [ ] Add attention layout/RoPE fusion with axis guards.
- [ ] Add full encoder and ST embedding parity tests.
- [ ] Benchmark sequence-length and batch-size sweeps with attention backend comparison.
