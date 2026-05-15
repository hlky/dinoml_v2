# DinoML Transformers Audit: yoso

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: uw-madison/yoso-4096 is the official in-library checkpoint reference.
Config source: Hugging Face config.json files listed below plus YosoConfig defaults.
Source files inspected:
- transformers/src/transformers/models/yoso/configuration_yoso.py
- transformers/src/transformers/models/yoso/modeling_yoso.py
- transformers/src/transformers/models/yoso/convert_yoso_pytorch_to_pytorch.py
- transformers/src/transformers/models/auto/tokenization_auto.py
- transformers/src/transformers/models/auto/modeling_auto.py
- kernels-community/yoso snapshot 6534e64bb05ad8e025551a4bc61678c3ccff73a4 source samples
Any missing files or assumptions:
- No tokenizer source is owned by yoso; AutoTokenizer maps yoso to AlbertTokenizer.
- The native source can load an external Hub kernel, kernels-community/yoso. DinoML should treat that as a separate provider/admission surface, not ordinary PyTorch ops.
- The first useful runtime target in this report is encoder + masked-LM parity for configs with use_expectation=true. Task heads are documented separately.
```

Representative configs inspected:

| Model id | Snapshot | Architecture | Hidden/layers/heads | Head dim | FFN | Max positions | Vocab | Type vocab | Attention flags |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `uw-madison/yoso-4096` | `51a6426...` | `YosoForMaskedLM` | 768 / 12 / 12 | 64 | 3072 | 4096 | 50265 | 1 | `use_expectation=true`, `hash_code_len=9`, `num_hash=64`, `conv_window=null`, `use_fast_hash=true` |
| `ydshieh/tiny-random-YosoForTokenClassification` | `719f9d3...` | token classification | 32 / 5 / 4 | 8 | 37 | 512 | 1024 | 16 | same flags; tiny dimensions stress `head_dim < 32` LSH padding if expectation is disabled |
| `MrAnderson/yoso-512-full-trivia` | `c1ddcce...` | QA | 768 / 12 / 12 | 64 | 3072 | 512 | 50265 | 1 | same flags |
| `MrAnderson/yoso-1024-full-trivia` | `0edb95d...` | QA | 768 / 12 / 12 | 64 | 3072 | 1024 | 50265 | 1 | same flags |
| `MrAnderson/yoso-2048-full-trivia-copied-embeddings` | `ac22bfb...` | QA | 768 / 12 / 12 | 64 | 3072 | 2048 | 50265 | 1 | same flags |
| `SKNahin/NER_YOSO` | `b86ddc7...` | token classification | 768 / 12 / 12 | 64 | 3072 | 4096 | 50265 | 1 | same flags, `num_labels=112` |

`uw-madison/yoso-2048`, `uw-madison/yoso-1024`, `uw-madison/yoso-512`, and `uw-madison/yoso-roberta-base` returned 404 through `huggingface_hub`; the open QA fine-tunes above provide the position-length sweep.

## 2. High-level architecture

YOSO is a text-only bidirectional encoder with BERT/RoBERTa-style embeddings and task heads. Its distinctive part is `YosoSelfAttention`: it replaces dense softmax attention with either:

- expectation cumulation: dense pairwise `acos(q @ k^T)` probability approximation, then weighted value sum; or
- LSH cumulation: hash query/key vectors and accumulate values by matching hash buckets through an external kernel.

Primary dataflow:

```text
AlbertTokenizer tokens -> token/position/type embeddings -> repeated YOSO encoder blocks -> MLM transform/head -> logits
```

Stage decomposition:

```text
CPU/tokenizer:
  text -> input_ids, attention_mask, token_type_ids
GPU/runtime:
  embeddings -> encoder blocks -> selected task head
External provider, only when use_expectation=false:
  fast_hash/lsh_cumulation kernels from kernels-community/yoso
```

No autoregressive prefill/decode or KV cache exists. Encoder outputs can be cached by applications for retrieval/classification workflows, but the source has no runtime cache ABI.

## 3. Important config dimensions

| Field | Default/source basis | Operator impact |
|---|---:|---|
| `vocab_size` | 50265 | word embedding `[50265, H]`; MLM decoder `[H -> vocab]`, tied to embeddings when `tie_word_embeddings=true` |
| `hidden_size` | 768 | embedding width, residual width, Q/K/V input and output width |
| `num_hidden_layers` | 12 | repeated encoder block count |
| `num_attention_heads` | 12 | Q/K/V reshape to `[B, heads, S, head_dim]` |
| `head_dim` | `hidden_size / num_attention_heads`; 64 for default | source rejects non-divisible hidden/head unless `embedding_size` exists |
| `intermediate_size` | 3072 | FFN `Linear(H -> I)`, `gelu`, `Linear(I -> H)` |
| `hidden_act` | `gelu` | FFN and MLM transform activation; classification head also uses `ACT2FN[hidden_act]` |
| `max_position_embeddings` | 4096 | position table is `[max_position_embeddings + 2, H]`; default position ids start at 2 |
| `type_vocab_size` | 1 default; tiny random uses 16 | token type embedding table |
| `layer_norm_eps` | `1e-12` | all LayerNorms |
| `use_expectation` | true in sampled configs | selects dense expectation math; overrides `num_hash` at inference |
| `hash_code_len` | 9 | expectation exponent; LSH bucket capacity `2 ** hash_code_len` |
| `num_hash` | 64 | LSH path only; hash-code tensor width and hashtable axis |
| `conv_window` | null in sampled configs | optional depthwise `Conv2d(heads -> heads, kernel=(window,1), groups=heads)` over value layer |
| `use_fast_hash` | true | LSH path only; calls external `fast_hash`; ineffective when `use_expectation=true` |
| `lsh_backward` | true | training/backward only |
| `torch_dtype` | float32 in sampled configs | source modules are normal PyTorch modules; no quantized storage |

## 3a. Family variation traps

- `use_expectation=true` and `use_expectation=false` are different attention families. First DinoML admission should allow expectation configs and reject or route LSH configs until the external kernel is audited/ported.
- Source defaults and sampled configs set `use_fast_hash=true`, but that flag is only read by the LSH path. It is not a runtime requirement for sampled expectation checkpoints.
- `hash_code_len` affects both expectation math and LSH bucket capacity. For expectation, it is an exponent in `(1 - acos(sim) / pi) ** hash_code_len`.
- `conv_window != null` adds a depthwise Conv2d residual over `value_layer * attention_mask[:, None, :, None]`. No sampled config enables it, but the source supports it.
- The attention mask is expected by `YosoSelfAttention` as rank-2 `[B,S]`; it is converted by `1.0 + attention_mask / 10000.0` then cast to int. DinoML should match native source behavior exactly, including this nonstandard mask handling.
- Non-fast fallback `hashing()` appears source-buggy: it computes `key_binary` but overwrites `query_hash` with the key hash and returns the same tensor twice. Do not implement a "fixed" version for parity without an explicit compatibility decision.
- Tiny random config has `head_dim=8`; if admitted into LSH mode, source pads Q/K/V to 32 before hashing/cumulation and slices the output back to 8.
- Position embeddings are allocated with `max_position_embeddings + 2`, while default position IDs are `arange(S) + 2`. This is RoBERTa-like and must not be translated to BERT's zero-based positions.
- `position_embedding_type` appears in some configs but is not read by `modeling_yoso.py`; it is ignored for this source basis.
- No GQA/MQA, RoPE, ALiBi, sliding window, MoE, KV cache, packed weights, or quantized weights are implemented in the inspected source.

## 4. Operator coverage checklist

Tensor/layout ops:

- reshape/view `[B,S,H] -> [B,S,heads,head_dim] -> [B,heads,S,head_dim] -> [B*heads,S,head_dim]`
- transpose/permute, contiguous, slicing, expand, unsqueeze, repeat_interleave, cat for LSH head-dim padding
- split/squeeze for QA logits, flatten for multiple-choice batch folding
- dtype casts: mask float arithmetic then int, hash codes int32, embedding indices int64

Neural network primitives:

- Embedding lookup: word `[vocab,H]`, position `[max_pos+2,H]`, token type `[type_vocab,H]`
- LayerNorm over last dim with eps `1e-12`
- Linear projections with bias:
  - Q/K/V: `Linear(H -> H)` each
  - attention output: `Linear(H -> H)`
  - FFN: `Linear(H -> I)`, GELU, `Linear(I -> H)`
  - MLM transform: `Linear(H -> H)`, GELU, LayerNorm, tied decoder `Linear(H -> vocab)`
  - classification: `[CLS]` select, dropout no-op in inference, `Linear(H -> H)`, activation, `Linear(H -> num_labels)`
  - token classification: `Linear(H -> num_labels)`
  - QA: `Linear(H -> 2)` then split
  - multiple choice: `Linear(H -> H)`, ReLU, `Linear(H -> 1)`
- Residual adds before LayerNorm in attention output and FFN output.
- Dropout is inference no-op.

Attention/custom primitives:

- L2 normalize over last dim for Q/K before expectation path and for context after cumulation.
- Dense expectation cumulation:
  - `sim = q @ k^T`, shape `[B*heads,S,S]`
  - `expectation = (1 - acos(sim) / pi) ** hash_code_len`
  - mask multiply by `query_mask[:, :, None] * key_mask[:, None, :]`
  - `context = expectation @ value`, shape `[B*heads,S,head_dim]`
- LSH cumulation, gated:
  - random or fast hash code generation `[B*heads,S,num_hash]`
  - hashtable capacity `2 ** hash_code_len`
  - external `lsh_cumulation(query_mask, query_hash, key_mask, key_hash, value, capacity, use_cuda, 1)`
- Optional depthwise Conv2d over source layout `[B, heads, S, head_dim]`, `groups=heads`, `kernel=(conv_window,1)`, `padding=(conv_window//2,0)`, no bias.

Position ops:

- Static buffer `position_ids = arange(max_position_embeddings) + 2`, sliced to sequence length.
- No rotary/relative-bias math.

Preprocessing-coupled ops:

- AlbertTokenizer output contract only: `input_ids`, optional `attention_mask`, optional `token_type_ids`.
- No image/audio/video processors, no scatter embedding stitch.

Tied weights:

- `YosoForMaskedLM` ties `cls.predictions.decoder.weight` to `yoso.embeddings.word_embeddings.weight`.
- `cls.predictions.decoder.bias` is aliased to `cls.predictions.bias` by `_tied_weights_keys`/setter behavior. Lowering must preserve one logical bias for the decoder output.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids: [B,S]
token_type_ids: [B,S], default zeros/buffer
position_ids: [1,S], default arange(S)+2
x = word_embed[input_ids] + token_type_embed[token_type_ids] + position_embed[position_ids]
x = LayerNorm(x, eps=1e-12)
```

YOSO encoder block, repeated `num_hidden_layers`:

```text
q = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> H, bias)(x).view(B,S,heads,D).transpose(1,2)

optional conv_v = DepthwiseConv2d(v * mask[:,None,:,None])

q,k,v = reshape to [B*heads,S,D]
mask = int((1 + attention_mask / 10000).repeat over heads)
if not use_expectation and D < 32: q,k,v = pad last dim to 32
if use_expectation or training: q,k = l2_normalize(q,k)
context = YosoCumulation(q,k,v,mask) or YosoLSHCumulation(q,k,v,mask)
if LSH and D < 32: context = context[..., :D]
context = l2_normalize(context)
context = reshape [B,heads,S,D]
if conv_window: context += conv_v
context = permute to [B,S,H]

attn_out = LayerNorm(Linear(H -> H)(context) + x)
ffn = Linear(H -> I)(attn_out)
ffn = GELU(ffn)
block_out = LayerNorm(Linear(I -> H)(ffn) + attn_out)
```

Masked-LM head:

```text
logits = tied_linear(LayerNorm(GELU(Linear(H -> H)(sequence_output))), vocab)
```

Other heads:

- Sequence classification: select token 0, `Linear(H -> H)`, activation, `Linear(H -> num_labels)`.
- Token classification: per-token `Linear(H -> num_labels)`.
- QA: per-token `Linear(H -> 2)`, split into start/end.
- Multiple choice: fold `[B,C,S] -> [B*C,S]`, select token 0, `Linear(H -> H)`, ReLU, `Linear(H -> 1)`, reshape `[B,C]`.

## 6. Attention requirements

YOSO attention is bidirectional encoder self-attention, but it is not softmax attention and is not SDPA-compatible without a custom rewrite. It has no causal mask, no cross-attention, no KV cache, no autoregressive decode, no rectangular query/key lengths in the in-library encoder path, and no packed/varlen ABI.

Expectation path, first target:

- Q/K/V shapes after head folding: `[B*heads, S, D]`.
- Q/K are L2-normalized before similarity.
- Similarity is dense `[B*heads,S,S]` with no `1/sqrt(D)` scaling.
- `acos` requires input in `[-1,1]`; source relies on L2 normalization and does not clamp.
- Scores are probabilities/weights, not logits; no softmax.
- Mask multiply is applied after the exponentiated expectation weights.
- Output context is normalized again over last dim.

LSH path, gated:

- Q/K/V may be padded to D=32 if original head dim is smaller.
- `fast_hash` consumes masks and Q/K vectors, produces int hash codes `[B*heads,S,num_hash]`.
- `lsh_cumulation` builds/queries hash tables of capacity `2 ** hash_code_len` and averages over `num_hash`.
- Fast CUDA hash uses random sign matrices internally in the external kernel path; the fallback Python hash uses `torch.randn` per call. Determinism is therefore a runtime contract, not a fixed-weight graph, unless hash plans are precomputed or seeds are controlled.
- Source optimized dispatch is the external `kernels-community/yoso` module. Without it, `use_expectation=false` will fail when `lsh_cumulation` is still `None`.

`output_attentions=True` returns `context_layer`, not an attention-probability matrix. DinoML should either match that ABI or reject attention-output requests initially.

## 7. Position encoding and custom math

Position encoding is learned absolute embeddings with a +2 offset:

```python
position_ids = torch.arange(max_position_embeddings).expand((1, -1)) + 2
position_embeddings = table[position_ids[:, :seq_len]]
```

Expectation attention math:

```python
def yoso_expectation_cumulation(query_mask, key_mask, query, key, value, hash_code_len):
    query = l2_normalize(query, dim=-1)
    key = l2_normalize(key, dim=-1)
    sim = query @ key.transpose(-1, -2)
    weights = (1.0 - acos(sim) / pi) ** hash_code_len
    weights = weights * query_mask[:, :, None] * key_mask[:, None, :]
    context = weights @ value
    return l2_normalize(context, dim=-1)
```

LSH fallback hash, source-compatible including the apparent native bug:

```python
def yoso_source_hashing(query, key, num_hash, hash_len):
    rmat = randn(query.shape[0], query.shape[2], num_hash * hash_len)
    powers = 2 ** arange(hash_len)
    q_proj = (query @ rmat).reshape(query.shape[0], query.shape[1], num_hash, hash_len)
    k_proj = (key @ rmat).reshape(key.shape[0], key.shape[1], num_hash, hash_len)
    query_hash = sum((q_proj > 0).int() * powers, dim=-1)
    query_hash = sum((k_proj > 0).int() * powers, dim=-1)
    return query_hash.int(), query_hash.int()
```

Precomputable: position IDs and embedding table loads. Dynamic per batch: Q/K normalization, `acos`, masks, and all LSH hash/cumulation intermediates.

## 8. Preprocessing and input packing

Tokenizer contract:

- AutoTokenizer maps `model_type="yoso"` to `AlbertTokenizer` when tokenizers are available.
- Official configs set `tokenizer_class="AlbertTokenizer"`, `pad_token_id=1`, `bos_token_id=0`, `eos_token_id=2`.
- GPU graph inputs are `input_ids [B,S]`, optional `attention_mask [B,S]`, optional `token_type_ids [B,S]`, optional `position_ids [1 or B,S]`, or `inputs_embeds [B,S,H]`.
- If `attention_mask` is absent, source creates all ones.
- If `token_type_ids` is absent, source expands a zero buffer to `[B,S]`.
- No processor emits packed sequence metadata; no `cu_seqlens`, no scatter, no multimodal placeholders.

Postprocessing:

- MLM produces `[B,S,vocab]` logits. Fill-mask ranking and token decoding are pipeline/controller work.
- QA produces start/end logits `[B,S]`; span selection/max-answer-length logic is not in the model class.
- Token classification produces `[B,S,num_labels]`; label aggregation is pipeline work.
- Sequence classification produces `[B,num_labels]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: dense expectation cumulation primitive

Source pattern:

```text
normalize(q), normalize(k) -> q @ k.T -> acos -> affine/pi -> pow(hash_code_len) -> mask multiply -> matmul(V) -> normalize
```

Replacement:

```text
YosoExpectationAttention(q, k, v, mask, hash_code_len)
```

Preconditions:

- `use_expectation=true`.
- Self-attention with equal query/key sequence length from the same hidden states.
- Q/K/V are rank-3 `[B*heads,S,D]`.
- Dense temporary `[B*heads,S,S]` fits memory or a tiled implementation is available.
- Preserve no-softmax and no-scale ordering exactly.

Failure cases:

- `acos` input drift outside `[-1,1]` under fp16/bf16 without source-compatible behavior.
- Requests for `output_attentions=True` if DinoML does not expose context-as-attention.

Parity sketch: compare the fused primitive to the PyTorch expression on random normalized and unnormalized Q/K/V with masks containing 0/1 and source-converted mask values.

### Rewrite: QKV projection packing

Source pattern:

```text
q = Linear(H,H), k = Linear(H,H), v = Linear(H,H)
```

Replacement:

```text
single Linear(H -> 3H) -> split [q,k,v]
```

Preconditions:

- Same input `hidden_states`.
- All three projections have bias.
- Packed weight order must be all-Q rows, all-K rows, all-V rows, matching split `[H,H,H]`.

Failure cases:

- Weight aliasing or external state dict expectations requiring separate parameters during load.

Parity sketch: pack weights and biases, compare Q/K/V tensors before reshapes.

### Rewrite: inference dropout erase

Source pattern:

```text
Dropout(p)(x)
```

Replacement:

```text
identity
```

Preconditions: eval/inference-only runtime.

### Rewrite: optional depthwise sequence Conv2d

Source pattern:

```text
value_layer [B,heads,S,D] * mask[:,None,:,None] -> Conv2d(groups=heads,kernel=(W,1),padding=(W//2,0))
```

Replacement:

```text
depthwise local sequence convolution over S for each head and D column
```

Preconditions:

- `conv_window != null`.
- Source NCHW-like layout `[B, C=heads, H=S, W=D]` is preserved, or a local guarded layout pass rewrites axes consistently.
- `groups == heads`, `bias == false`, dilation 1, stride 1.

Layout constraints:

- Do not globally translate this to NHWC. It is an attention-internal tensor, and consumers expect `[B,heads,S,D]`.

### Rewrite: MLM last-token-only is not applicable

YOSO MLM is bidirectional and fill-mask oriented. There is no decoder last-token logits optimization.

## 10. Kernel fusion candidates

Highest priority:

- `LayerNorm + residual add` for attention output and FFN output. These occur twice per layer and use eps `1e-12`.
- QKV packed GEMM and reshape/transpose fusion. It reduces three launches and helps feed the custom attention primitive.
- YOSO expectation attention fused/tiled kernel. Dense `[S,S]` temporaries at 4096 positions are the main memory/performance risk for the first admitted path.

Medium priority:

- GELU FFN epilogue: `Linear(H -> 4H) + GELU` and `Linear(H -> H) + residual + LayerNorm`.
- MLM transform/head: `Linear + GELU + LayerNorm` and tied decoder GEMM.
- Optional depthwise sequence conv for configs with `conv_window`.

Lower priority:

- LSH external-kernel parity/provider work. It is important for non-expectation configs, but sampled public configs use expectation.
- Task-head fusions for classification/QA; these are small relative to encoder cost.

## 11. Runtime staging plan

Stage 1: config and weight loading.

- Parse `YosoConfig`, reject unsupported `add_cross_attention=true` and initially reject `use_expectation=false`.
- Preserve embedding/LM-head tied weights.
- Load one official MLM checkpoint and one tiny random checkpoint.

Stage 2: embedding and single block parity.

- Implement embeddings with +2 position IDs.
- Run one encoder block through expectation path in fp32.

Stage 3: full encoder + MLM parity.

- Enable all 12 layers for `uw-madison/yoso-4096`.
- Validate logits for short sequences first, then longer sequence buckets.

Stage 4: task heads.

- Add QA, token classification, sequence classification, and multiple choice as thin heads over the shared encoder.

Stage 5: optimize expectation path.

- Add a fused/tiled expectation-cumulation kernel and QKV packing.
- Benchmark sequence lengths 512/1024/2048/4096.

Stage 6: gated LSH path.

- Decide admission policy: port `kernels-community/yoso`, call it as an external provider, or reject `use_expectation=false`.
- Add deterministic hash-plan policy before production admission.

Stubbable initially: training losses, backward, gradient checkpointing, `output_hidden_states`, `output_attentions`, and all LSH configs.

## 12. Parity and validation plan

- Custom op tests for L2 normalize, `acos` expectation weights, mask multiplication, and `pow(hash_code_len)` in fp32.
- Source-compatible mask tests: compare all-ones masks, mixed 0/1 masks, and extended-style 0/-10000 masks to document the native behavior.
- Single-layer random-weight parity for embeddings + one block at `B=2`, `S=8/64`, `H=32` tiny config and `H=768` default config.
- Full encoder parity for tiny random at `S=16/128`.
- Official MLM checkpoint parity for `uw-madison/yoso-4096` at short `S` first, then 512 and 4096 if memory permits.
- Head parity:
  - MLM logits `[B,S,vocab]`
  - QA start/end `[B,S]`
  - token classification `[B,S,num_labels]`
  - sequence classification `[B,num_labels]`
  - multiple choice `[B,num_choices]`
- Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 should be deferred until `acos`/normalize stability is characterized.
- LSH parity, if admitted: pin RNG/fast-hash source, compare hash codes and cumulation outputs against external kernel for fixed seeds and masks.

No DinoML tests were run for this audit.

## 13. Performance probes

- Tokenizer throughput separately from model runtime.
- Embedding + first-block latency to isolate position/type embedding overhead.
- Expectation attention kernel sweep over `B`, `S`, `heads`, `D`; especially `S=512/1024/2048/4096`.
- Dense expectation temporary memory usage: `[B*heads,S,S]` fp32/fp16.
- FFN GEMM throughput for `Linear(768 -> 3072)` and `Linear(3072 -> 768)`.
- QKV packed versus separate GEMM launch comparison.
- Full encoder throughput by sequence length and batch size.
- Task-head overhead for MLM full-vocab logits versus classification/QA heads.
- If LSH is admitted: hash generation, hashtable build/query, atomics contention, and deterministic seed overhead; compare external kernel versus any DinoML implementation.

## 14. Skip/defer list

- Training losses and backward, including `lsh_backward`.
- Gradient checkpointing.
- `output_attentions=True` and `output_hidden_states=True` ABI unless needed by users.
- `use_expectation=false` LSH path until the external kernel/determinism contract is settled.
- `conv_window != null` unless a checkpoint requiring it appears.
- Mixed precision expectation path until `acos` stability and normalization tolerances are validated.
- Beam search, KV cache, speculative decoding, continuous batching decode: not applicable.
- Quantization and packed weights: not present in source/configs.

## 15. Final implementation checklist

- [ ] Parse `YosoConfig` and expose an admission gate for `use_expectation`, `conv_window`, and ignored historical fields.
- [ ] Load embeddings, encoder, heads, and tied MLM decoder weights without breaking aliases.
- [ ] Implement +2 absolute position ID default.
- [ ] Implement embedding sum + LayerNorm.
- [ ] Implement Q/K/V projections and shape transforms.
- [ ] Implement source-compatible attention-mask conversion.
- [ ] Implement `YosoExpectationCumulation` in fp32 reference form.
- [ ] Add one-block parity tests for tiny and default dimensions.
- [ ] Add full encoder + MLM parity for `uw-madison/yoso-4096`.
- [ ] Add QA/token/sequence/multiple-choice head parity.
- [ ] Add QKV packing rewrite with packed `[Q,K,V]` row order.
- [ ] Add fused/tiled expectation attention kernel or explicit fallback policy for long sequences.
- [ ] Benchmark 512/1024/2048/4096 sequence lengths.
- [ ] Decide LSH path policy: reject, external provider, or ported DinoML kernel.
- [ ] If LSH is admitted, add deterministic hash/seed and external-kernel parity tests.
