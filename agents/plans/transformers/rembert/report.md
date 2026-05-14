# RemBERT DinoML Audit

## 1. Source basis

```text
Transformers commit/version: local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/rembert, plus open fine-tuned/tiny/community RemBERT configs
Config source: saved config snapshots under agents/plans/transformers/rembert/_sources/
Source files inspected:
  X:/H/transformers/src/transformers/models/rembert/configuration_rembert.py
  X:/H/transformers/src/transformers/models/rembert/modeling_rembert.py
  X:/H/transformers/src/transformers/models/rembert/tokenization_rembert.py
  X:/H/transformers/src/transformers/models/rembert/__init__.py
Any missing files or assumptions:
  No tokenization_rembert_fast.py exists in this checkout.
  google/rembert-ft-xnli, google/rembert-ft-squad, and google/rembert-ft-tydiqa raw config URLs were not accessible at the attempted locations.
  Primary DinoML target for this report: encoder and masked-LM/classification-style inference; causal-LM/decoder paths are deferred unless a checkpoint explicitly sets is_decoder=True.
```

Representative configs inspected:

- `google/rembert`: `_sources/google-rembert-config.json`
- `Sindhu/rembert-squad2`: `_sources/Sindhu-rembert-squad2-config.json`
- `Misha24-10/rembert-ft-for-multi-ner`: `_sources/Misha24-10-rembert-ft-for-multi-ner-config.json`
- `ibraheemmoosa/xlmindic-rembert-uniscript`: `_sources/ibraheemmoosa-xlmindic-rembert-uniscript-config.json`
- `ydshieh/tiny-random-rembert`: `_sources/ydshieh-tiny-random-rembert-config.json`
- `google/rembert` tokenizer config: `_sources/google-rembert-tokenizer_config.json`

## 2. High-level architecture

RemBERT is a text-only BERT-style transformer encoder with factorized input and output embeddings.

```text
tokenizer -> input_ids/attention_mask/token_type_ids -> factorized embeddings
  -> Linear(input_embedding_size -> hidden_size)
  -> N encoder blocks
  -> pooled output and/or sequence output
  -> task head: MLM, QA, token classification, sequence classification, multiple choice
```

Stage decomposition:

- CPU/data pipeline: multilingual Unigram tokenization, special-token packing, padding/truncation, attention mask construction, optional token type IDs.
- GPU/runtime: embedding gathers, embedding LayerNorm, hidden projection, encoder blocks, selected task head.
- Independently cacheable output: encoder sequence output can feed downstream classifiers or retrieval heads. This is not an autoregressive KV-cache target for canonical configs.

## 3. Important config dimensions

Canonical source defaults from `RemBertConfig`:

| Field | Value | Source |
|---|---:|---|
| `vocab_size` | 250300 | config default / `google/rembert` |
| `input_embedding_size` | 256 | config default / `google/rembert` |
| `hidden_size` | 1152 | config default / `google/rembert` |
| `output_embedding_size` | 1664 | config default / `google/rembert` |
| `num_hidden_layers` | 32 | config default / `google/rembert` |
| `num_attention_heads` | 18 | config default / `google/rembert` |
| `head_dim` | 64 | inferred from source: `hidden_size / num_attention_heads` |
| `intermediate_size` | 4608 | config default / `google/rembert` |
| `max_position_embeddings` | 512 | config default / `google/rembert` |
| `type_vocab_size` | 2 | config default / `google/rembert` |
| `hidden_act` | `gelu` | config default / configs inspected |
| `layer_norm_eps` | `1e-12` | config default / configs inspected |
| `is_decoder` | `False` | config default |
| `add_cross_attention` | `False` | config default |
| `tie_word_embeddings` | `False` | config default / configs inspected |

Representative checkpoint sweep:

| Model | Architecture | Hidden | Layers | Heads | Head dim | Input emb | Output emb | Vocab | Task-significant variation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `google/rembert` | not listed | 1152 | 32 | 18 | 64 | 256 | 1664 | 250300 | canonical encoder/MLM base |
| `Sindhu/rembert-squad2` | `RemBertForQuestionAnswering` | 1152 | 32 | 18 | 64 | 256 | 1664 | 250300 | QA head |
| `Misha24-10/rembert-ft-for-multi-ner` | `RemBertForTokenClassification` | 1152 | 32 | 18 | 64 | 256 | 1664 | 250300 | token head with 73 labels |
| `ibraheemmoosa/xlmindic-rembert-uniscript` | `RemBertForMaskedLM` | 768 | 12 | 12 | 64 | 128 | 768 | 65536 | smaller/forked MLM shape and special IDs |
| `ydshieh/tiny-random-rembert` | `RemBertModel` | 128 | 2 | 2 | 64 | 256 | 1664 | 30522 | tiny debug shape; odd output head dims if used with MLM |

## 3a. Family variation traps

- Input embeddings are not `hidden_size`: source performs word/position/token-type embedding at `input_embedding_size`, then `Linear(input_embedding_size -> hidden_size)` before layer 0.
- MLM/logit head is independently factorized through `output_embedding_size`; `hidden_size -> output_embedding_size -> vocab_size` is required. Do not tie this to input embeddings unless a future checkpoint/source path explicitly does so.
- Canonical `google/rembert` config contains historical `embedding_size=256`; current source reads `input_embedding_size`, not `embedding_size`, except the attention divisibility guard has an old `hasattr(config, "embedding_size")` escape hatch. DinoML should use `input_embedding_size`.
- `hidden_size` must be divisible by `num_attention_heads`; source computes `head_dim` by integer division and uses no GQA/MQA.
- Dropout exists but is zero in canonical inference configs; it should be compiled away only under inference/eval guards.
- `position_embedding_type` appears in tiny configs but is not read by RemBERT source. Treat it as ignored for this source basis.
- Tokenizer special IDs vary in community configs: Google tokenizer config uses `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, while xlmindic config changes pad/bos/eos IDs. Treat tokenizer IDs as ABI data, not source constants.
- Decoder, cross-attention, and cache code exists but canonical RemBERT is an encoder. Require explicit `is_decoder=True` admission before compiling causal/cross-attention behavior.
- Text layout is `[batch, sequence, channel]`; no NHWC/NCHW translation is relevant. The axis-sensitive guards are sequence axis `dim=1` for chunking and first-token pooling, and softmax over attention key axis `dim=-1`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs: `input_ids`, `token_type_ids`, `position_ids`, `attention_mask`.
- Embedding gather for word, absolute position, and token type tables.
- Add three embedding tensors, LayerNorm over last dim, dropout no-op in eval.
- View/reshape/transpose/permute/contiguous for attention: `[B,S,H] -> [B,S,A,D] -> [B,A,S,D]`, and back.
- Slice/gather first token: `hidden_states[:, 0]`.
- Multiple-choice flatten/unflatten: `[B,C,S] -> [B*C,S]`, logits `[B*C,1] -> [B,C]`.
- QA split/squeeze: `[B,S,2] -> [B,S]` start/end logits.
- Optional CausalLM `logits_to_keep` slice on sequence axis.

Neural primitives:

- `Linear(input_embedding_size -> hidden_size)` embedding projection.
- Per layer:
  - Q/K/V `Linear(hidden_size -> hidden_size)` with bias.
  - Attention output `Linear(hidden_size -> hidden_size)` with bias.
  - MLP up `Linear(hidden_size -> intermediate_size)` with bias.
  - GELU.
  - MLP down `Linear(intermediate_size -> hidden_size)` with bias.
  - Two post-residual LayerNorms at `hidden_size`.
- Pooler: first-token gather, `Linear(hidden_size -> hidden_size)`, tanh.
- MLM head: `Linear(hidden_size -> output_embedding_size)`, GELU, LayerNorm at `output_embedding_size`, `Linear(output_embedding_size -> vocab_size)`.
- Sequence/token classification heads: dropout no-op in eval, `Linear(hidden_size -> num_labels)`.
- QA head: `Linear(hidden_size -> 2)`.

Attention primitives:

- Dense bidirectional self-attention for encoder target.
- Matmul score shape `[B,A,Q,D] x [B,A,D,K] -> [B,A,Q,K]`.
- Scale by `1 / sqrt(head_dim)`, add broadcast mask, softmax over `K`, matmul with V.
- Optional decoder self-attention cache and encoder-decoder cross-attention are source-supported but deferred for first encoder target.

Position/tokenizer ops:

- Absolute position embedding lookup from `position_ids`; default positions are an arange slice offset by `past_key_values_length`.
- No RoPE, ALiBi, relative bias, local attention, sparse attention, or sliding window.
- Tokenizer is Unigram/Metaspace with special-token template packing; tokenizer stays CPU/data-pipeline for first DinoML target.

Generation/cache ops:

- Not required for canonical encoder/MLM/classification target.
- If causal LM is admitted, cache stores per-layer keys/values after projection and transpose as `[B, heads, seq, head_dim]`; cross-attention cache is managed through `EncoderDecoderCache`.

Quantized/packed weight metadata:

- None in Transformers RemBERT source. Any GGUF/quantized RemBERT import would be a DinoML loading/provider contract, not a source operator.

## 5. Layer/block breakdown

Embedding block:

```text
word = Embedding(vocab_size -> input_embedding_size)(input_ids)
type = Embedding(type_vocab_size -> input_embedding_size)(token_type_ids)
pos = Embedding(max_position_embeddings -> input_embedding_size)(position_ids)
x = LayerNorm(word + type + pos)
x = Dropout(x)  # no-op in eval
x = Linear(input_embedding_size -> hidden_size)(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(hidden_size -> hidden_size, bias=True)(x)
k = Linear(hidden_size -> hidden_size, bias=True)(x)
v = Linear(hidden_size -> hidden_size, bias=True)(x)
q,k,v = reshape/transposed to [B, heads, S, head_dim]
scores = MatMul(q, transpose(k)) / sqrt(head_dim)
scores = scores + extended_attention_mask
prob = Softmax(scores, dim=-1)
ctx = MatMul(prob, v)
ctx = transpose/reshape to [B, S, hidden_size]
x = LayerNorm(Linear(hidden_size -> hidden_size)(ctx) + x)
m = GELU(Linear(hidden_size -> intermediate_size)(x))
x = LayerNorm(Linear(intermediate_size -> hidden_size)(m) + x)
```

Heads:

- Pooler: `pooled = tanh(Linear(hidden_size -> hidden_size)(sequence[:,0]))`.
- MLM/CausalLM: `logits = Linear(output_embedding_size -> vocab_size)(LayerNorm(GELU(Linear(hidden_size -> output_embedding_size)(sequence))))`.
- Token classification: `logits = Linear(hidden_size -> num_labels)(sequence)`.
- Sequence classification: `logits = Linear(hidden_size -> num_labels)(pooled)`.
- QA: `start,end = split(Linear(hidden_size -> 2)(sequence), dim=-1)`.

## 6. Attention requirements

For the first DinoML target:

- Attention type: noncausal dense self-attention.
- Head layout: MHA, no GQA/MQA. Canonical heads `18`, head dim `64`; xlmindic `12 x 64`; tiny `2 x 64`.
- Query/key/value widths: all `hidden_size`.
- Masking: source builds an extended additive attention mask through `PreTrainedModel.get_extended_attention_mask`; model adds it to scores before softmax. DinoML should preserve the additive-mask convention and broadcast shape.
- Packed/varlen: none in source.
- Sliding/local/sparse: none.
- Position interactions: absolute position embeddings only.
- FlashAttention compatibility: encoder self-attention can be lowered to dense noncausal SDPA/FlashAttention when an additive padding mask is supported or converted safely. Attention probabilities output is optional and should disable fused kernels if exact attention tensors are requested.

Deferred decoder/cross-attention:

- Source can act as decoder if `is_decoder=True`; then self-attention is causal via `get_extended_attention_mask` behavior and uses `past_key_values`.
- Cross-attention exists only if `add_cross_attention=True`; key/value source becomes `encoder_hidden_states`, and cross K/V can be reused after first update.
- Admission guard: reject or route to a separate audit unless the checkpoint config explicitly requests decoder behavior.

## 7. Position encoding and custom math

Position encoding is ordinary learned absolute embedding:

```python
if position_ids is None:
    position_ids = arange(max_position_embeddings)[
        past_key_values_length : seq_length + past_key_values_length
    ]
x = word_embeddings(input_ids) + token_type_embeddings(token_type_ids)
x = x + position_embeddings(position_ids)
```

No RoPE, ALiBi, relative bias, convolutional position embedding, or custom position math is required.

Custom math to preserve:

```python
scores = matmul(q, k.transpose(-1, -2))
scores = scores / sqrt(head_dim)
scores = scores + extended_attention_mask
prob = softmax(scores, dim=-1)
context = matmul(prob, v)
```

LayerNorm epsilon is `1e-12`, which is smaller than many modern encoder configs and should be carried exactly in parity tests.

## 8. Preprocessing and input packing

Tokenizer/runtime ABI:

- Tokenizer class: `RemBertTokenizer`.
- Backend: Hugging Face `tokenizers` Unigram with Metaspace pre-tokenizer/decoder.
- Google tokenizer config: `do_lower_case=false`, `remove_space=true`, `keep_accents=true`, special tokens `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`.
- Source tokenizer constructs:
  - single: `[CLS]:0 $A:0 [SEP]:0`
  - pair: `[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1`
- Model accepts token type IDs and creates zeros when omitted.
- `model_input_names` in tokenizer lists only `input_ids` and `attention_mask`; DinoML frontends should still allow explicit `token_type_ids` because model source consumes them.

CPU/data-pipeline first:

- Text normalization/tokenization and special-token packing should remain outside the compiled graph initially.
- Padding side, truncation, overflow/chunk mapping, and wordpiece-to-label alignment for token classification are tokenizer/pipeline responsibilities, not encoder graph ops.

GPU/runtime:

- Inputs are dense `[B,S]` integer IDs/masks.
- Position IDs can be provided or generated as arange; generation of position IDs is a small runtime helper.

## 9. Graph rewrite / lowering opportunities

### Rewrite: embedding triplet fusion

Source pattern:

```text
word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
-> LayerNorm(input_embedding_size)
```

Replacement:

```text
FusedEmbedding3AddLayerNorm
```

Preconditions:

- All embedding outputs share `[B,S,input_embedding_size]`.
- Position IDs are either explicit dense IDs or the default contiguous arange.
- Token type IDs are explicit or all zeros.
- LayerNorm epsilon matches config.

Failure cases:

- `inputs_embeds` path bypasses word embedding; keep a fallback.
- Nonzero dropout/training mode is out of inference scope.

Parity test sketch:

- Compare fused and unfused embeddings for explicit and omitted token type IDs, explicit and default position IDs, fp32 and fp16 weights.

### Rewrite: factorized embedding projection folding

Source pattern:

```text
embedding_output [B,S,E] -> Linear(E -> H)
```

Replacement:

```text
Keep as GEMM initially; optionally fold static position/token-type paths only when IDs are constant.
```

Preconditions:

- Dynamic `input_ids` prevent general folding into the word table unless weight preprocessing creates a projected word embedding table and separately handles position/token-type projection.

Weight transform option:

```python
projected_word = word_embedding @ W_in.T + b_in
```

This is only safe if position and token-type embeddings are also projected separately and summed after projection. It changes memory footprint, so treat it as a provider/loading optimization.

### Rewrite: BERT MHA canonicalization

Source pattern:

```text
three independent Linear(H -> H) -> reshape/transpose -> scaled masked softmax attention
```

Replacement:

```text
PackedQKVLinear or 3-GEMM grouped launch -> dense noncausal attention -> output projection
```

Preconditions:

- Self-attention only, no output attentions requested.
- No decoder cache and no cross-attention for first target.
- `hidden_size == num_heads * head_dim`.
- Mask is padding/additive and convertible to the selected attention backend.

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Failure cases:

- `output_attentions=True` requires attention probabilities.
- Decoder/cross-attention cache changes Q/K/V source and shape.

### Rewrite: post-norm residual fusion

Source pattern:

```text
Linear(...)(x) + residual -> LayerNorm
```

Replacement:

```text
GEMM bias + residual + LayerNorm epilogue, or GEMM then fused AddLayerNorm
```

Preconditions:

- Residual tensor shape equals GEMM output shape `[B,S,H]`.
- LayerNorm axis is last dim and epsilon is config exact.

Failure cases:

- Debug `output_hidden_states` may require preserving intermediate boundaries.

### Rewrite: MLM last-token or masked-token logits

Source pattern:

```text
sequence -> hidden_to_output_embedding -> vocab logits for all sequence positions
```

Replacement:

```text
Gather selected positions -> factorized LM head
```

Preconditions:

- Caller only needs selected mask positions or CausalLM `logits_to_keep`.
- Gather indices are known or supplied as bounded integer tensor.

Failure cases:

- Fill-mask APIs may request all logits for all masked positions; keep selection explicit in ABI.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with epsilon `1e-12`: embeddings and every residual branch depend on it.
- Dense encoder attention: Q/K/V projection packing, noncausal SDPA/FlashAttention-compatible backend, and output projection.
- GEMM + GELU for MLP up projection and GEMM + residual + LayerNorm for both post-norm sites.
- Factorized MLM head: `H -> O -> vocab` dominates memory bandwidth and logits compute for `vocab_size=250300`.

Medium priority:

- Fused embedding gather/add/LayerNorm.
- Pooler first-token gather + dense + tanh for classification workloads.
- Token/QA head split/squeeze/gather kernels.
- Multiple-choice reshape around encoder reuse.

Lower priority:

- Decoder KV cache and cross-attention cache paths.
- Training losses, dropout, gradient checkpointing, and attention-probability materialization.
- Tokenizer acceleration inside DinoML; CPU tokenization is fine for first integration.

## 11. Runtime staging plan

Stage 1: parse RemBERT configs and load weights for `RemBertModel` encoder. Admit canonical encoder configs with `is_decoder=False`, `add_cross_attention=False`, `hidden_size % num_attention_heads == 0`.

Stage 2: implement embedding block parity, including factorized `input_embedding_size` and default token type/position IDs.

Stage 3: implement one encoder layer parity with dense noncausal attention and post-norm residual order.

Stage 4: full encoder parity for `last_hidden_state` and optional pooler output.

Stage 5: add task heads in order: token classification and QA, sequence classification, MLM. MLM should include output embedding factorization and untied vocab decoder.

Stage 6: optimize attention and GEMM/LayerNorm fusions under guards that disable them when `output_attentions=True` or debug hidden-state boundaries are required.

Stage 7: optional decoder/causal-LM path only for explicit configs, with separate cache ABI tests.

Initially stub/defer:

- Loss computation.
- Training mode/dropout.
- Gradient checkpointing.
- Attention outputs and hidden-state output tuples if the first ABI only returns task logits/last hidden state.

## 12. Parity and validation plan

- Config parser tests for canonical, tiny, xlmindic, QA, and token-classification configs.
- Embedding block random tests:
  - explicit vs omitted `token_type_ids`;
  - explicit vs default `position_ids`;
  - `inputs_embeds` fallback.
- Single-layer parity against Transformers with random weights and fixed masks, fp32 tolerance around `1e-5` to `1e-4`.
- Full encoder parity for small/tiny config first, then canonical shape smoke at shorter sequence lengths.
- Attention mask tests:
  - all-ones mask;
  - padded tokens;
  - 3D self-attention mask if admitted.
- Head tests:
  - token classification logits `[B,S,num_labels]`;
  - QA start/end split `[B,S]`;
  - sequence classifier first-token pooling;
  - MLM logits `[B,S,vocab]` with factorized output head.
- Recommended tolerances:
  - fp32: `rtol=1e-4`, `atol=1e-5` for full blocks, tighter for individual GEMMs/LayerNorm if practical.
  - fp16/bf16: compare against Transformers in eval mode with relaxed `rtol=1e-2`, `atol=1e-2`, and isolate attention softmax drift.

## 13. Performance probes

- Tokenization throughput vs encoder throughput for multilingual batches.
- Encoder throughput sweep: batch size, sequence length up to 512, canonical 32 layers.
- Attention backend comparison: explicit BMM/softmax/BMM vs fused SDPA/FlashAttention for noncausal padding masks.
- GEMM provider probes:
  - embedding projection `256 -> 1152`;
  - attention projections `1152 -> 1152`;
  - MLP `1152 -> 4608 -> 1152`;
  - MLM head `1152 -> 1664 -> 250300`.
- Logits bottleneck probe for MLM with all-token logits vs selected-position logits.
- Memory probe for activations at `[B,512,1152]`, attention scores `[B,18,512,512]`, and vocab logits `[B,S,250300]`.
- Task-head throughput probes for QA/token classification/classification separately from encoder.

## 14. Skip/defer list

- Training losses and label-side logic.
- Dropout/training mode and gradient checkpointing.
- Decoder/causal-LM generation unless `is_decoder=True` configs are explicitly targeted.
- Encoder-decoder cross-attention unless `add_cross_attention=True` configs are explicitly targeted.
- Beam search, sampling, forced language IDs, and multilingual generation control; RemBERT tokenizer is multilingual but canonical model is not a translation/generation controller.
- Attention probability output materialization for optimized attention path.
- Tokenizer implementation inside compiled runtime.
- Quantization/GGUF loading until a separate weight-format admission policy exists.

## 15. Final implementation checklist

- [ ] Parse `RemBertConfig`, including `input_embedding_size` and `output_embedding_size`.
- [ ] Reject unsupported first-pass configs: `is_decoder=True`, `add_cross_attention=True`, non-divisible hidden/head shapes, unknown `hidden_act`.
- [ ] Load embeddings, encoder, pooler, and selected task-head weights with untied input/output embeddings.
- [ ] Implement embedding gather/add/LayerNorm and default token type/position ID helpers.
- [ ] Implement `Linear(input_embedding_size -> hidden_size)` embedding projection.
- [ ] Implement dense noncausal MHA with additive attention mask and last-dim softmax.
- [ ] Implement post-norm residual blocks and GELU MLP.
- [ ] Implement pooler, token classification, QA, sequence classification, multiple choice, and MLM heads.
- [ ] Add guarded QKV packing rewrite.
- [ ] Add guarded fused attention rewrite for encoder self-attention.
- [ ] Add AddLayerNorm and GEMM/GELU fusion candidates.
- [ ] Add selected-position MLM/logit rewrite for fill-mask or `logits_to_keep` style workloads.
- [ ] Add parity tests for embeddings, one layer, full encoder, masks, and each admitted head.
- [ ] Benchmark encoder, attention backend, MLP GEMMs, and MLM vocab projection separately.

## Gated gaps for DinoML

- DinoML must not assume embedding width equals hidden width.
- DinoML must not tie input embeddings to LM decoder weights for RemBERT.
- DinoML should guard optimized attention on `output_attentions=False` and encoder-only configs.
- DinoML should treat tokenizer special IDs and special-token packing as checkpoint/tokenizer ABI, not hard-coded constants.
- DinoML should reject or separately audit decoder/cross-attention configs before enabling cache behavior.
- DinoML needs robust large-vocab logits handling; canonical MLM logits are very large at `250300` vocab entries.
