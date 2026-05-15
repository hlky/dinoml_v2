# Transformers Audit: xlm_roberta_xl

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `transformers`.

Model id: primary `facebook/xlm-roberta-xl`; same in-library family also covers `facebook/xlm-roberta-xxl`.

Config source:

- [facebook/xlm-roberta-xl config.json](https://huggingface.co/facebook/xlm-roberta-xl/raw/main/config.json), HF repo SHA `aa5d120255845efeebc9b7f42822a1dd0f9ece9d`
- [facebook/xlm-roberta-xxl config.json](https://huggingface.co/facebook/xlm-roberta-xxl/raw/main/config.json), HF repo SHA `03e0fb540c3c9afd4bdda0072e7cb82d2eafd060`
- Source defaults from `configuration_xlm_roberta_xl.py`

Source files inspected:

- `transformers/src/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py`
- `transformers/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py`
- `transformers/src/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py`
- `transformers/src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py`
- `transformers/src/transformers/masking_utils.py`

Any missing files or assumptions: `modeling_xlm_roberta_xl.py` is generated from `modular_xlm_roberta_xl.py`; the generated file is the import/runtime basis, while future Transformers source edits should inspect the modular file. There is no `tokenization_xlm_roberta_xl.py`; official configs set `tokenizer_class: "XLMRobertaTokenizer"`, so tokenizer coupling belongs to the base XLM-R tokenizer implementation. Only two official `model_type: "xlm-roberta-xl"` configs were found; adjacent XLM-R base/large configs are out of scope for this family.

Primary runtime target for this report: encoder-only masked language model and encoder hidden states. Sequence, token, QA, and multiple-choice heads are optional head slices. Decoder/causal LM and cross-attention are source-implemented but should be rejected or separately admitted for first integration because the official XL/XXL checkpoints are encoder masked-LM configs.

## 2. High-level architecture

XLM-RoBERTa-XL is a text-only, bidirectional encoder, RoBERTa/BERT-like in tensor structure but with important pre-LayerNorm placement:

```text
SentencePiece/Unigram tokenizer -> input_ids/attention_mask
-> word + token_type + absolute position embeddings
-> N pre-LN encoder blocks
-> final encoder LayerNorm
-> masked LM head or classification/token/QA head
```

The base encoder can be validated independently from each task head. The tokenizer and special-token layout are CPU/data-pipeline work; the GPU graph begins at `input_ids`, `attention_mask`, and optional `token_type_ids`/`position_ids` or `inputs_embeds`.

## 3. Important config dimensions

| Field | XL official config | XXL official config | Source default |
|---|---:|---:|---:|
| `architectures` | `XLMRobertaXLForMaskedLM` | `XLMRobertaXLForMaskedLM` | n/a |
| `model_type` | `xlm-roberta-xl` | `xlm-roberta-xl` | `xlm-roberta-xl` |
| `hidden_size` | 2560 | 4096 | 2560 |
| `num_hidden_layers` | 36 | 48 | 36 |
| `num_attention_heads` | 32 | 32 | 32 |
| `head_dim` | 80 | 128 | 80 |
| `intermediate_size` | 10240 | 16384 | 10240 |
| `hidden_act` | `gelu` | `gelu` | `gelu` |
| `vocab_size` | 250880 | 250880 | 250880 |
| `max_position_embeddings` | 514 | 514 | 514 |
| `type_vocab_size` | 1 | 1 | 1 |
| `layer_norm_eps` | 1e-5 | 1e-5 | 1e-5 |
| `position_embedding_type` | `absolute` | `absolute` | not declared/read |
| `use_cache` | true | true | true |
| `torch_dtype` | float32 | float32 | n/a |

Representative checkpoint sweep:

| Checkpoint | Scope | Layers | Hidden | Heads | Head dim | MLP | Vocab | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `facebook/xlm-roberta-xl` | in-scope | 36 | 2560 | 32 | 80 | 10240 | 250880 | official masked-LM encoder |
| `facebook/xlm-roberta-xxl` | in-scope | 48 | 4096 | 32 | 128 | 16384 | 250880 | same model type, much larger GEMMs |
| `FacebookAI/xlm-roberta-base` | adjacent only | 12 | 768 | 12 | 64 | 3072 | 250002 | different `model_type`; useful tokenizer contrast only |
| `FacebookAI/xlm-roberta-large` | adjacent only | 24 | 1024 | 16 | 64 | 4096 | 250002 | different `model_type`; not this family |

## 3a. Family variation traps

- XL and XXL keep `num_attention_heads=32`, but `head_dim` changes from 80 to 128. Do not hardcode `head_dim=64`.
- `hidden_size == num_heads * head_dim` for official configs, but `head_dim` is computed as integer division in source.
- Official configs include `position_embedding_type: "absolute"`, but the inspected XL source does not read it. Relative position behavior is not implemented for this family basis.
- No GQA/MQA: key/value heads equal query heads.
- No gated MLP: feed-forward is `Linear(H -> I) -> GELU -> Linear(I -> H)`.
- Norm placement differs from classic post-LN BERT/RoBERTa: no embedding LayerNorm, attention pre-LN, MLP pre-LN, and an extra final encoder LayerNorm.
- `type_vocab_size=1`, but the graph still contains token type embedding and a default zero-token-type gather path.
- `chunk_size_feed_forward` is read from inherited config defaults; first integration can require zero/no chunking and reject positive chunking.
- `use_cache=True` appears in encoder configs, but source disables cache unless `config.is_decoder=True`.
- Causal LM and cross-attention classes exist, but official XL/XXL masked-LM configs should stay encoder-only at first.
- `tokenizer_class` is `XLMRobertaTokenizer`; vocab size differs from older XLM-R base/large, so tied embedding/logit head dimensions must follow this family config.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token ids, token type ids, and learned absolute position ids.
- `ne`, `int`, `cumsum(dim=1)`, add, multiply, cast to long for default position-id creation.
- Default token type path uses expand plus `gather(dim=1, index=position_ids)` from a zero buffer.
- Shape/view ops: `view`, `reshape`, `transpose(1, 2)`, `transpose(2, 3)`, `contiguous`, `split`, `squeeze`, sequence slicing, multiple-choice flatten/unflatten.
- Mask creation: 2D padding mask to backend-specific additive or packed attention mask.

Neural network primitives:

- Dense Linear with bias for all Q/K/V/O projections, FFN projections, pooler, LM head, classifier heads, QA head.
- LayerNorm over hidden dimension, eps 1e-5.
- GELU in FFN and masked-LM transform.
- Tanh in sequence classification and pooler.
- Residual adds after attention output projection and FFN output projection.
- Dropout is present in source but should be disabled for inference.

Attention primitives:

- Dense bidirectional MHA for first target: Q/K/V shape `[B, S, H] -> [B, 32, S, D]`, where D is 80 for XL and 128 for XXL.
- Eager parity path: `Q @ K^T * head_dim^-0.5`, additive mask, softmax over keys, `attn @ V`, transpose back to `[B, S, H]`.
- Source advertises FlashAttention, SDPA, FlexAttention, and generic attention backend support. DinoML can start with dense attention parity and later map no-padding or padded bidirectional masks to optimized attention.

Position/relative-bias ops:

- Learned absolute position embedding table `[514, H]`.
- Default position ids from pad-aware cumsum:

```python
mask = (input_ids != pad_token_id).int()
position_ids = ((cumsum(mask, dim=1) + past_key_values_length) * mask).long() + pad_token_id
```

- No RoPE, ALiBi, or relative bias in inspected XL source.

Generation/cache ops:

- Not required for primary masked-LM target.
- Source can instantiate `DynamicCache` when `is_decoder=True`, with K/V stored after projection and reshape/transpose. This should be a gated follow-up, not first-pass XL parity.

Preprocessing-coupled ops:

- Unigram/SentencePiece tokenization, metaspace pre-tokenization, `<s> A </s>` and `<s> A </s></s> B </s>` templates.
- Model GPU inputs are `input_ids` and `attention_mask`; tokenizer does not list `token_type_ids` as a model input, though the model accepts them.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,S] or inputs_embeds [B,S,H]
position_ids default: pad-aware cumsum + pad_token_id
token_type_ids default: zeros gathered by position_ids, effectively all segment 0
x = word_embedding[input_ids] + token_type_embedding[token_type_ids] + position_embedding[position_ids]
```

Encoder block, repeated N times:

```text
attn_in = LayerNorm(x)
q = Linear(H -> H, bias)(attn_in).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> H, bias)(attn_in).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> H, bias)(attn_in).view(B,S,heads,D).transpose(1,2)
attn = DenseBidirectionalAttention(q, k, v, additive_or_backend_mask)
attn = attn.transpose(1,2).reshape(B,S,H)
x = Linear(H -> H, bias)(attn) + x
mlp_in = LayerNorm(x)
ff = GELU(Linear(H -> intermediate, bias)(mlp_in))
x = Linear(intermediate -> H, bias)(ff) + x
```

Final encoder:

```text
last_hidden_state = LayerNorm(x)
```

Masked LM head:

```text
y = Linear(H -> H, bias)(last_hidden_state)
y = GELU(y)
y = LayerNorm(y)
logits = Linear(H -> vocab_size, bias)(y)
```

The LM decoder weight is tied to `roberta.embeddings.word_embeddings.weight`; keep this as one logical parameter alias.

Task heads:

- Sequence classification: take token 0, dropout, `Linear(H -> H)`, tanh, dropout, `Linear(H -> num_labels)`.
- Multiple choice: flatten `[B,C,S] -> [B*C,S]`, use pooler token 0 `Linear(H -> H) + tanh`, dropout, `Linear(H -> 1)`, reshape to `[B,C]`.
- Token classification: dropout then `Linear(H -> num_labels)` per token.
- QA: `Linear(H -> num_labels)`, usually num labels 2, split last dim into start/end, squeeze.

## 6. Attention requirements

Primary target attention is noncausal self-attention, full dense MHA:

- Causal or noncausal: noncausal for official configs.
- Self-attention or cross-attention: self-attention only for official configs.
- MHA/MQA/GQA: MHA, 32 Q heads and 32 K/V heads.
- Head dims: 80 for XL, 128 for XXL.
- Query/key/value width: all equal hidden size.
- Masking style: bidirectional padding mask from `create_bidirectional_mask`; eager attention receives an additive mask broadcastable to `[B, heads, Q, K]`. Flash-style backends may receive a 2D padding mask or no mask when fully unpadded.
- Packed/varlen support: source backend can route to optimized integrations, but model source does not create custom `cu_seqlens`; those are backend integration internals.
- Sliding/local attention: none.
- Position interactions: absolute embeddings are added before blocks; no attention-relative position bias.
- KV cache: disabled for encoder mode even though configs contain `use_cache=true`.

For decoder-mutated configs, source adds causal masks and optional cross-attention. Admission policy: reject `is_decoder=True` and `add_cross_attention=True` for first XL/XXL masked-LM integration unless a separate decoder/cross-attention audit is opened.

## 7. Position encoding and custom math

Position behavior is learned absolute embedding plus pad-aware position-id generation. The config field `position_embedding_type` is present in official configs but not consumed by the inspected source.

Short parity function:

```python
def xlm_roberta_xl_position_ids(input_ids, pad_token_id=1, past_key_values_length=0):
    mask = (input_ids != pad_token_id).to_int()
    inc = (cumsum(mask, dim=1).to(mask.dtype) + past_key_values_length) * mask
    return inc.to_int64() + pad_token_id
```

For `inputs_embeds`, the source cannot infer padding and uses sequential positions from `pad_token_id + 1` through `sequence_length + pad_token_id`.

Precompute opportunities:

- Position embedding table is constant.
- Position ids can be supplied by caller to bypass cumsum/gather logic.
- For fixed unpadded sequence lengths, position ids are deterministic `[2, 3, ..., S+1]` for pad token id 1.

## 8. Preprocessing and input packing

Tokenizer contract:

- Tokenizer class: `XLMRobertaTokenizer`.
- Backend: Hugging Face `tokenizers` Unigram, SentencePiece assets `sentencepiece.bpe.model` and `tokenizer.json`.
- Special ids from config/tokenizer: BOS/CLS `<s>` id 0, PAD `<pad>` id 1, EOS/SEP `</s>` id 2, UNK id 3, mask token `<mask>`.
- Single sequence template: `<s> A </s>`.
- Pair template: `<s> A </s></s> B </s>`.
- `model_input_names`: `input_ids`, `attention_mask`.

GPU/runtime inputs:

- `input_ids [B,S]` int64 or `inputs_embeds [B,S,H]`, exactly one required.
- `attention_mask [B,S]`, usually 1 for valid tokens and 0 for padding.
- Optional `position_ids [B,S]`; if omitted, graph must reproduce source default or require CPU-provided position ids.
- Optional `token_type_ids [B,S]`; official tokenizer does not emit them, and default is effectively all zeros.

No multimodal packing, placeholder scatter, image/audio/video processor, OCR metadata, or layout translation is involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Require Explicit Position IDs For First GPU Graph

Source pattern:

```text
input_ids -> ne(pad) -> int -> cumsum(dim=1) -> add/mul/cast -> embedding lookup
```

Replacement:

```text
CPU/tokenizer emits position_ids -> position embedding lookup
```

Preconditions:

- Caller uses standard tokenizer/pad semantics.
- DinoML validates `position_ids == xlm_roberta_xl_position_ids(input_ids, pad_token_id)` in parity tests or debug mode.
- Dynamic padding patterns are allowed only when position ids are supplied.

Failure cases: direct `inputs_embeds` path, custom `position_ids`, decoder cache offsets, or nonstandard pad token behavior.

Parity test sketch: compare source model embeddings for padded and unpadded batches with supplied versus internally generated position ids.

### Rewrite: QKV Projection Packing

Source pattern:

```text
q = Linear(H -> H)(x_norm)
k = Linear(H -> H)(x_norm)
v = Linear(H -> H)(x_norm)
```

Replacement:

```text
packed_qkv = Linear(H -> 3H)(x_norm)
split packed last dim as [q, k, v]
```

Preconditions:

- Same input tensor and dtype.
- All three projections have bias.
- Packed weight order is all-Q rows, all-K rows, all-V rows to match source modules.
- Output split preserves `[B,S,H] -> view(B,S,heads,D).transpose(1,2)`.

Failure cases: weight aliasing edits, quantized per-module formats, or custom head pruning.

### Rewrite: Inference Dropout Removal

Source pattern: dropout after embeddings, attention probabilities, attention output projection, and classifier heads.

Replacement: identity in eval/inference mode.

Preconditions: inference-only compile; training disabled.

### Rewrite: Masked-LM Last/Selected Token Logits

Source pattern: masked-LM head over full `[B,S,H]`.

Replacement: gather selected masked token states before LM transform/logit projection.

Preconditions:

- Inference request only needs specific masked positions.
- Output ABI can return `[num_masked, vocab]` or caller accepts index-restored shape.

Failure cases: full-sequence logits requested, loss path, or downstream code expects `[B,S,V]`.

### Rewrite: MLP GELU Epilogue Fusion

Source pattern: `Linear(H -> I) -> GELU -> Linear(I -> H)`.

Replacement: fuse first GEMM bias plus GELU where provider supports it; keep second GEMM separate unless residual add epilogue is available.

Preconditions: static activation `gelu`, dense row-major inputs, no chunked feed-forward.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B,S,H]`: every block uses two pre-norms plus one final encoder norm and one LM-head norm.
- Bidirectional encoder attention: QKV packing, scaled dot-product attention, additive padding mask, and output projection dominate runtime.
- Large GEMMs: XL uses `2560x2560`, `2560x10240`, `10240x2560`; XXL uses `4096x4096`, `4096x16384`, `16384x4096`.
- LM head projection `H -> 250880`: expensive for full-sequence logits; selected-token logits are valuable for fill-mask.

Medium priority:

- Bias+GELU fusion for FFN and LM transform.
- Residual add fused into output projection epilogue where norm placement permits.
- Position-id generation kernel only if position ids are not supplied by CPU.
- Token embedding + token type + position embedding add fusion.

Lower priority:

- Pooler/classification head fusion.
- Multiple-choice flatten/unflatten handling.
- Decoder KV-cache path for mutated configs.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `XLMRobertaXLForMaskedLM`, preserving tied LM decoder and word embedding alias.

Stage 2: implement embeddings and one encoder block parity with explicit caller-supplied `position_ids`.

Stage 3: full encoder parity for XL at small batch/sequence, dense attention backend, inference dropout removed.

Stage 4: masked-LM head parity, first full `[B,S,V]`, then selected-token logits optimization.

Stage 5: add task heads: sequence classification, token classification, QA, and multiple choice.

Stage 6: optimize attention backend and packed QKV rewrite; add guarded no-padding fast path and padded bidirectional path.

Stage 7: admit XXL with memory planning and large-vocab logit probes. Use weight-loading/offload/quantized-provider work only after dense parity is boring.

Stub initially: training losses, dropout, output attentions, hidden-state recording, decoder/cross-attention, cache, positive feed-forward chunking.

## 12. Parity and validation plan

- Position-id tests: padded/unpadded `input_ids`, explicit `position_ids`, and `inputs_embeds` sequential position behavior.
- Embedding-only parity: compare word + token type + position sum for XL config.
- Single block parity: random hidden states and masks, fp32 tolerance around `1e-4` absolute for dense eager attention.
- Full encoder parity: 1, 2, and all layers on short sequences; include padding masks.
- Attention backend parity: dense eager versus optimized backend for no-padding and padded batches.
- Masked-LM head parity: full logits `[B,S,V]` and selected masked positions.
- Task head parity: token 0 classification path, token classification per-position logits, QA split/squeeze, multiple-choice flatten/reshape.
- Tied-weight validation: updating/loading word embedding must update the LM decoder logical weight.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-4`; fp16/bf16 should be evaluated after backend choices and may need `rtol=5e-2, atol=5e-2` for long attention/large logits.

No DinoML tests or model execution were run for this report.

## 13. Performance probes

- Encoder throughput by sequence length: 32, 128, 256, 512.
- Batch sweep for XL and XXL separately.
- Attention backend comparison: dense unfused, SDPA-like, Flash-like no-padding, padded bidirectional.
- QKV packed projection versus three GEMMs.
- LayerNorm bandwidth and fusion impact.
- FFN GEMM throughput for `H -> 4H -> H`.
- Full-vocab LM head throughput and memory bandwidth.
- Selected-token fill-mask logits throughput.
- Weight memory and load-time probes for XL vs XXL; XXL dense fp32 weights are very large and may need staged/offload planning.
- Tokenizer throughput separately from GPU encoder throughput.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout in training mode.
- Decoder/causal LM and encoder-decoder cross-attention.
- KV cache and beam/search generation.
- Output attentions and hidden-state capture as public ABI.
- Positive `chunk_size_feed_forward` chunking.
- Relative position behavior despite historical/config `position_embedding_type` keys.
- Quantization, packed weights, multi-GPU tensor parallel.
- General tokenizer implementation inside DinoML runtime; keep tokenization in CPU/data pipeline.

## 15. Final implementation checklist

- [ ] Parse `XLMRobertaXLConfig` and reject unsupported mutated decoder/cross-attention configs for first target.
- [ ] Load XL/XXL dense weights and preserve LM decoder/word embedding tied alias.
- [ ] Implement embedding path with supplied `position_ids`; add optional pad-aware cumsum path later.
- [ ] Implement LayerNorm, Linear, GELU, tanh, residual add, reshape/transpose/split/squeeze coverage needed by heads.
- [ ] Implement dense bidirectional MHA with additive padding mask.
- [ ] Add one-block and full-encoder parity tests.
- [ ] Add masked-LM head parity, including selected-token logits rewrite.
- [ ] Add optional sequence/token/QA/multiple-choice head parity.
- [ ] Add QKV packing rewrite with explicit weight order and shape guards.
- [ ] Add attention backend benchmarks for padded and unpadded sequences.
- [ ] Add XXL memory/performance probe before declaring production readiness.
