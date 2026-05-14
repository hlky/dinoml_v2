# Megatron-BERT DinoML audit

## 1. Source basis

Transformers commit/version:

- Local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id:

- Primary source target: `megatron_bert` / `model_type="megatron-bert"`.
- Official NVIDIA repos inspected: `nvidia/megatron-bert-cased-345m`, `nvidia/megatron-bert-uncased-345m`.
- Representative open configs inspected: `KBLab/megatron-bert-base-swedish-cased-600k`, `KBLab/megatron-bert-large-swedish-cased-165k`, `IDEA-CCNL/Erlangshen-MegatronBert-1.3B`, `IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese`, `EMBO/BioMegatron345mUncased`, `hf-tiny-model-private/tiny-random-MegatronBertForMaskedLM`.

Config source:

- Native config defaults from `configuration_megatron_bert.py`.
- HF raw/API configs where hosted.
- NVIDIA 345M repos do not host `config.json` or weights; their README says users convert NGC checkpoints locally, and the conversion script writes `config.json` plus `pytorch_model.bin`. Treat those configs as source/conversion defaults unless a local converted checkpoint is supplied.

Source files inspected:

- `X:/H/transformers/src/transformers/models/megatron_bert/configuration_megatron_bert.py`
- `X:/H/transformers/src/transformers/models/megatron_bert/modeling_megatron_bert.py`
- `X:/H/transformers/src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py`
- `X:/H/transformers/tests/models/megatron_bert/test_modeling_megatron_bert.py`
- `X:/H/transformers/docs/source/en/model_doc/megatron-bert.md`
- Source notes and config sweep: `_sources/source_notes.md`, `_sources/config_sweep.md`

Any missing files or assumptions:

- No processor/image/audio path exists. Tokenization is standard BERT/BertTokenizer or PreTrainedTokenizerFast behavior and should stay CPU/data-pipeline owned.
- Primary DinoML target for this audit: encoder-only inference plus optional MLM/NSP/classification heads. Decoder/cache paths are source-implemented but should be gated until explicitly targeted.

## 2. High-level architecture

Megatron-BERT is a text-only BERT-style dense encoder with Megatron norm placement.

```text
tokenizer/data pipeline -> ids/masks/types/positions -> embeddings
-> repeated pre-norm encoder blocks -> final encoder LayerNorm
-> base hidden states / pooler / MLM / NSP / classifier / QA heads
```

Main runtime stages:

- CPU/data pipeline: tokenization, special tokens, segment IDs, attention mask creation, optional pair encoding.
- GPU/runtime graph: embedding lookups and adds, dense encoder stack, final norm, selected head.
- Cacheable outputs: encoder hidden states can be cached by an application for retrieval/classification reuse, but normal encoder inference has no autoregressive KV cache.

The main architecture difference from vanilla BERT is norm placement: embedding LayerNorm is removed, every layer has pre-attention and pre-MLP LayerNorm, and the encoder applies one final LayerNorm.

## 3. Important config dimensions

| Field | Source default | Operator impact |
| --- | ---: | --- |
| `vocab_size` | 29056 | word embedding rows and MLM decoder rows; conversion may override from checkpoint |
| `hidden_size` | 1024 | embedding width, residual width, Q/K/V/output width |
| `num_hidden_layers` | 24 | repeated encoder block count |
| `num_attention_heads` | 16 | MHA head count |
| `head_dim` | 64 inferred | `hidden_size // num_attention_heads`; no explicit config field |
| `intermediate_size` | 4096 | FFN expansion width |
| `hidden_act` | `gelu` | FFN and MLM transform activation; configs also use `gelu_new` |
| `max_position_embeddings` | 512 | learned absolute position table and default position IDs |
| `type_vocab_size` | 2 | token type embedding rows; tiny/debug configs use 16 |
| `layer_norm_eps` | 1e-12 | all LayerNorms |
| `use_cache` | true | ignored for encoder mode because `is_decoder=False` forces no cache |
| `is_decoder` | false | if true, enables causal/decoder mask behavior through shared HF helpers |
| `add_cross_attention` | false | if true with decoder mode, inserts cross-attention per layer |
| `tie_word_embeddings` | true | MLM/CLM decoder weight aliases input word embedding |

Representative checkpoint/config sweep:

| Model id | Config basis | H | L | heads | head_dim | FFN | vocab | act | task/head |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| NVIDIA 345M cased/uncased | conversion/source defaults | 1024 | 24 | 16 | 64 | 4096 | default 29056, overridden from checkpoint | gelu | MLM/NSP examples |
| `KBLab/megatron-bert-base-swedish-cased-600k` | hosted config | 768 | 12 | 12 | 64 | 3072 | 64128 | gelu | MLM |
| `KBLab/megatron-bert-large-swedish-cased-165k` | hosted config | 1024 | 24 | 16 | 64 | 4096 | 64128 | gelu | MLM |
| `IDEA-CCNL/Erlangshen-MegatronBert-1.3B` | hosted config | 2048 | 24 | 8 | 256 | 8192 | 21248 | gelu_new | base encoder |
| `IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese` | hosted config | 2560 | 48 | 40 | 64 | 10240 | 21248 | gelu | MLM |
| `EMBO/BioMegatron345mUncased` | hosted config | 1024 | 24 | 16 | 64 | 4096 | 30592 | gelu_new | base encoder |
| `hf-tiny-model-private/tiny-random-MegatronBertForMaskedLM` | hosted config | 64 | 5 | 4 | 16 | 37 | 1124 | gelu | tiny MLM |

## 3a. Family variation traps

- Native source uses separate Q, K, V biased projections; do not expect packed QKV checkpoint tensors after HF conversion.
- Norm placement is not vanilla BERT post-norm. A graph importer must preserve: embedding add/dropout with no LN, pre-attention LN, attention residual add without output LN, pre-MLP LN, MLP residual add, final encoder LN.
- `position_embedding_type="absolute"` appears in some configs but is ignored by this source. Reject or ignore non-absolute values for native Megatron-BERT.
- `hidden_act` can be `gelu` or `gelu_new`; `gelu_new` requires the HF tanh-approx GELU variant.
- The source only raises on `hidden_size % num_attention_heads != 0` when the config lacks `embedding_size`; even if the guard is bypassed, projections still use `hidden_size // heads`. DinoML should require exact divisibility.
- `embedding_size` in tiny/test configs is not a real embedding projection in native source.
- `use_cache=true` in encoder configs is not a runtime cache requirement; `MegatronBertModel.forward` sets `use_cache=False` unless `is_decoder=True`.
- Decoder, cross-attention, and CLM are implemented but not representative of normal Megatron-BERT checkpoints. Gate separately.
- Multiple choice flattens `[B, C, S]` into `[B*C, S]` before encoding, then reshapes logits to `[B, C]`.
- QA uses `Linear(H -> num_labels)`, then `split(1, dim=-1)`, squeeze, contiguous. First integration can require `num_labels == 2`.
- There are no vision/audio layouts. Any NHWC/channel-last policy is irrelevant except as a general dense GEMM layout optimization inside linear kernels.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer input tensors: `input_ids`, `token_type_ids`, `position_ids`.
- Embedding gather for word, token type, and position tables.
- Tensor add chain for embeddings and residuals.
- `view`, `transpose(1, 2)`, `permute(0, 2, 1, 3)`, `contiguous`, `squeeze`, `split`, flatten/reshape for heads.
- Slice/gather for pooler first token `hidden[:, 0]`, CLM `hidden[:, slice_indices, :]`, position ID range, and optional multiple-choice flattening.

Neural network primitives:

- `LayerNorm(H, eps=1e-12)` for pre-attention, pre-MLP, final encoder norm, and MLM transform.
- Biased `Linear(H -> H)` for Q, K, V, attention output, pooler, MLM transform.
- Biased `Linear(H -> I)` and `Linear(I -> H)` for FFN.
- Activations: `gelu`, `gelu_new`, `tanh` for pooler, optional classification loss-only sigmoid is not needed for inference.
- Dropout is source-visible but should compile to identity in eval inference.

Attention primitives:

- Dense noncausal self-attention for primary encoder target.
- Optional decoder causal self-attention, optional encoder-decoder cross-attention, and cache update/read if `is_decoder=True`.
- Additive attention mask, softmax over key dimension, context matmul.

Position/relative-bias ops:

- Learned absolute position embedding only. No RoPE, ALiBi, relative bias, sliding window, local attention, or block sparse attention.

Generation/cache ops:

- Not required for primary encoder/MLM target.
- For CLM target only: HF `Cache`/`DynamicCache`/`EncoderDecoderCache` update ABI, per-layer K/V tensors shaped `[B, heads, S, head_dim]`, and `logits_to_keep` slice.

Preprocessing-coupled ops:

- Standard BERT tokenizer ABI: `[CLS]`/`[SEP]`, optional pair segment IDs, pad attention mask, `[MASK]` for MLM. Tokenization remains outside DinoML graph.

Tied weights:

- MLM/pretraining/CLM decoder weight is tied to `bert.embeddings.word_embeddings.weight`.
- Prediction head bias aliases `cls.predictions.bias` and decoder bias in `_tied_weights_keys`.

## 5. Layer/block breakdown

Embedding stage:

```text
input_ids: [B, S] int64 or inputs_embeds: [B, S, H]
token_type_ids: [B, S], default zeros
position_ids: [1, S] slice from [1, max_pos], offset by past length only in decoder/cache mode
x = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
x = dropout(x)  # identity in inference
```

Encoder block, repeated `num_hidden_layers` times:

```text
res0 = x
a = LayerNorm(x)
q = Linear(H -> H, bias)(a).view(B, S, heads, head_dim).transpose(1, 2)
k = Linear(H -> H, bias)(a or encoder_hidden).view(...).transpose(1, 2)
v = Linear(H -> H, bias)(a or encoder_hidden).view(...).transpose(1, 2)
scores = matmul(q, k.transpose(-1, -2)) / sqrt(head_dim)
scores = scores + extended_attention_mask
p = softmax(scores, dim=-1)
ctx = matmul(p, v).permute(0, 2, 1, 3).contiguous().view(B, S, H)
x = res0 + dropout(Linear(H -> H, bias)(ctx))

res1 = x
m = LayerNorm(x)
m = Linear(H -> I, bias)(m)
m = activation(m)
x = res1 + dropout(Linear(I -> H, bias)(m))
```

Encoder output:

```text
sequence_output = final LayerNorm(x)
pooled_output = tanh(Linear(H -> H)(sequence_output[:, 0]))  # if pooler enabled
```

Heads:

- MLM: `Linear(H -> H) -> activation -> LayerNorm -> Linear(H -> vocab)` over `[B, S, H]`.
- NSP: `Linear(H -> 2)` on pooled output.
- Pretraining: MLM plus NSP.
- Sequence classification: dropout identity in eval, `Linear(H -> num_labels)` on pooled output.
- Multiple choice: flatten choices before encoder, then `Linear(H -> 1)` and reshape to `[B, num_choices]`.
- Token classification: `Linear(H -> num_labels)` over sequence.
- QA: `Linear(H -> num_labels)`, split last dim into start/end logits; require `num_labels=2` for first pass.

## 6. Attention requirements

Primary target:

- Noncausal dense self-attention.
- MHA, not GQA/MQA: query heads = key/value heads = `num_attention_heads`.
- `head_dim = hidden_size // num_attention_heads`; require `hidden_size == heads * head_dim`.
- Query/key/value widths are all `hidden_size`.
- Query length and key length are both `S` for encoder self-attention.
- Attention mask is converted by HF helper into an additive broadcast mask. DinoML should canonicalize to `[B, 1, 1, S]` for 2D masks and `[B, 1, S, S]` for 3D masks with large negative masked values.
- Softmax axis is last dimension.
- No packed/varlen, local/sliding, ALiBi, RoPE, relative bias, or FlashAttention-specific source path.

Optional decoder target:

- If `is_decoder=True`, `get_extended_attention_mask` provides causal masking semantics.
- K/V cache stores projected K/V after any position embedding effect in hidden states, before attention-score matmul, shaped `[B, heads, cached_S, head_dim]`.
- If `add_cross_attention=True`, each layer has a second attention block over `encoder_hidden_states`; cross-attention K/V can be cached in `EncoderDecoderCache`.
- Gate this target separately because Megatron-BERT’s normal public checkpoints are bidirectional encoders.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding lookup:

```python
def megatron_bert_positions(position_table, position_ids, past_key_values_length=0):
    # default position_ids are arange(max_position_embeddings)[past:past + S]
    return position_table[position_ids]
```

Custom norm placement is the key model-specific math:

```python
def megatron_bert_block(x, mask):
    y = layernorm_attn(x)
    y = dense_attention(y, mask)
    x = x + out_proj(y)
    z = layernorm_mlp(x)
    z = mlp_out(act(mlp_in(z)))
    return x + z
```

Precomputable:

- Default position IDs `[0..S-1]` when `position_ids` are omitted and no decoder past length is used.
- Attention mask broadcast/additive conversion for static or bucketed sequence shapes.

Dynamic inputs:

- User-provided `position_ids`, 2D/3D attention masks, token type IDs, and decoder past length if optional CLM mode is admitted.

## 8. Preprocessing and input packing

Text input contract:

- `input_ids`: `[B, S]`, integer token IDs.
- `attention_mask`: optional `[B, S]` or `[B, S, S]`; default ones.
- `token_type_ids`: optional `[B, S]`; default zeros. Pair inputs use segment 0/1.
- `position_ids`: optional `[B or 1, S]`; default contiguous slice from registered arange buffer.
- `inputs_embeds`: optional `[B, S, H]`; mutually exclusive with `input_ids`.

Tokenizer coupling:

- Standard BERT special tokens and `[MASK]` for MLM.
- Cased/uncased and language-specific vocab choices affect token IDs and `vocab_size`, not encoder graph structure.

GPU/runtime-owned:

- Embedding lookups, mask conversion if not precomputed, encoder, selected head.

CPU/data-pipeline-owned:

- Text normalization/tokenization, special token insertion, padding/truncation, pair segment assignment, label construction, generation controller for optional CLM.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(H -> H)(x_ln)
k = Linear(H -> H)(x_ln)
v = Linear(H -> H)(x_ln)
```

Replacement:

```text
qkv = Linear(H -> 3H)(x_ln)
split qkv as [Q, K, V] contiguous H blocks
```

Preconditions:

- Self-attention only, no cross-attention.
- Q/K/V inputs are the same tensor.
- All three projections have compatible dtype/layout and bias policy.
- Weight transform preserves split order `query, key, value`.

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], dim=0)
b_qkv = concat([b_q, b_k, b_v], dim=0)
```

Failure cases:

- Cross-attention uses different current states for K/V.
- Decoder cache update may prefer separate K/V materialization.

Parity test sketch:

- Compare packed-QKV block output against source block for random `[B,S,H]`, with and without attention mask.

### Rewrite: eval dropout elimination

Source pattern:

```text
dropout(x)
```

Replacement:

```text
identity(x)
```

Preconditions:

- Inference/eval mode only.

Failure cases:

- Training or stochastic parity tests.

### Rewrite: dense encoder attention -> fused SDPA/FlashAttention

Source pattern:

```text
scores = q @ k.T / sqrt(head_dim)
scores += additive_mask
probs = softmax(scores, -1)
ctx = probs @ v
```

Replacement:

```text
scaled_dot_product_attention(q, k, v, additive_mask, causal=False)
```

Preconditions:

- Encoder mode: noncausal.
- Dense masks only; no need to return attentions.
- Dropout disabled.
- Mask value semantics match HF additive extended masks.

Failure cases:

- `output_attentions=True`.
- 3D masks whose layout or dtype cannot be mapped to fused backend.
- Decoder/cross-attention/cache until separately validated.

### Rewrite: MLM decoder tied output GEMM

Source pattern:

```text
h = Linear(H -> H) -> activation -> LayerNorm
logits = h @ word_embedding.T + bias
```

Replacement:

```text
gemm_rcr_bias(h, word_embedding, bias)
```

Preconditions:

- Preserve tied weight alias with input embedding.
- Vocab dimension is last output dimension.
- Optional CLM `logits_to_keep` slice is applied before MLM head for CLM path.

Failure cases:

- Untied or externally replaced output embedding without alias metadata.

### Rewrite: pooler first-token slice + linear + tanh

Source pattern:

```text
pooled = tanh(Linear(sequence_output[:, 0]))
```

Replacement:

```text
slice token 0 -> gemm_rrr_bias/gemm_rcr_bias -> tanh
```

Preconditions:

- Sequence length >= 1.
- Static or runtime guard for token 0 availability.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm for `[B,S,H]`, especially H=1024/2048/2560.
- Bias GEMM for Q/K/V, attention output, FFN, and heads.
- Packed QKV GEMM for self-attention.
- Fused dense attention/SDPA for encoder prefill-length sequences.
- FFN `Linear -> gelu/gelu_new -> Linear`, with activation fusion where practical.

Medium priority:

- Embedding gather + token type + position add.
- Residual add with preceding output projection epilogue.
- MLM transform `Linear -> activation -> LayerNorm`.
- Last-token/sliced logits for optional CLM `logits_to_keep`.

Lower priority:

- Pooler `slice -> Linear -> tanh`.
- Multiple-choice flatten/reshape conveniences.
- QA split/squeeze postprocessing.
- Decoder/cache/cross-attention kernels until a real checkpoint target needs them.

## 11. Runtime staging plan

Stage 1: config and weight loading

- Parse native MegatronBertConfig.
- Load HF-converted weights.
- Preserve tied embedding/decoder alias metadata.
- Reject unsupported config flags: non-absolute position behavior, non-divisible hidden/head shapes, decoder/cross-attention unless enabled for a separate target.

Stage 2: one-block encoder parity

- Implement embedding ABI, pre-norm attention block, pre-norm MLP block, final norm.
- Validate random weights against local Transformers source for one or two layers.

Stage 3: full encoder target

- Run full encoder hidden-state parity for representative base/large/1.3B shapes with static sequence buckets.
- Support `input_ids`, `attention_mask`, `token_type_ids`, optional `position_ids`.

Stage 4: heads

- Add MLM first because fill-mask is common.
- Add pooler/sequence classification/NSP.
- Add token classification and QA after split/squeeze/indexing details are covered.
- Multiple choice can be a wrapper around encoder flattening.

Stage 5: optimized attention and GEMM rewrites

- Enable packed QKV and fused SDPA under guards.
- Profile GEMM candidates for H/I/vocab dimensions.

Stage 6: optional decoder/CLM

- Only after encoder parity: admit `is_decoder=True`, causal mask, cache ABI, cross-attention if required.

What can be stubbed initially:

- Losses, dropout, gradient checkpointing, `output_attentions`, `output_hidden_states`, training-only paths, generation controller.

## 12. Parity and validation plan

Concrete tests:

- Embedding parity with provided/default token type IDs and position IDs.
- LayerNorm placement test proving no embedding LN and final encoder LN exists.
- Single attention block parity with additive 2D mask and 3D mask.
- Single FFN block parity for `gelu` and `gelu_new`.
- Full encoder parity for tiny config `[B=2,S=7,H=64,L=2]`.
- Full encoder parity for representative static shapes with random weights: H=768/L=12/head=12 and H=1024/L=24/head=16.
- MLM head parity with tied output weight.
- Sequence classification and NSP pooler parity.
- QA output parity for `num_labels=2`, including split/squeeze shapes.
- Optional: decoder/cache parity only for a separately admitted CLM config.

Recommended tolerances:

- fp32: `atol=1e-4`, `rtol=1e-4` for end-to-end; stricter for individual GEMMs if using exact math.
- fp16/bf16: start with `atol=2e-2`, `rtol=2e-2` for full stack, tighten per kernel.

## 13. Performance probes

- Encoder throughput by `B` and `S`: `[1, 8, 32] x [16, 128, 512]`.
- LayerNorm bandwidth for H=768, 1024, 2048, 2560.
- QKV separate GEMM versus packed QKV GEMM.
- SDPA/FlashAttention versus explicit GEMM-softmax-GEMM for noncausal encoder masks.
- FFN GEMM/activation/GEMM throughput for I=3072,4096,8192,10240.
- MLM vocab projection throughput for vocab 21248, 29056/30592, 64128.
- Mask conversion overhead for 2D and 3D masks.
- Weight-loading memory footprint for 345M, 1.3B, and 3.9B configs.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout stochastic behavior.
- `output_attentions=True` and dense attention probability outputs for optimized path.
- `output_hidden_states=True` unless needed for debug.
- Decoder/causal LM/cache/cross-attention until a target checkpoint requires it.
- Tensor parallel/distributed Megatron-LM checkpoint layouts before HF conversion.
- Quantized/packed weight formats; no source-coupled quantization appears in native HF Megatron-BERT.
- Non-absolute position embedding variants advertised by historical configs but ignored by native source.

## 15. Final implementation checklist

- [ ] Parse `MegatronBertConfig` and apply source defaults.
- [ ] Reject native-source unsupported config behavior (`position_embedding_type` other than absolute, non-divisible hidden/head widths, decoder/cross-attention unless targeted).
- [ ] Load embeddings, encoder blocks, final LayerNorm, and selected head weights.
- [ ] Preserve word embedding / MLM decoder tied weight alias.
- [ ] Implement embedding gather/add/default token type/default position path.
- [ ] Implement Megatron pre-norm encoder block and final encoder LayerNorm.
- [ ] Implement dense MHA with additive extended masks.
- [ ] Implement `gelu` and `gelu_new` FFN activation paths.
- [ ] Implement MLM head and vocab projection.
- [ ] Implement pooler plus NSP/sequence classification heads.
- [ ] Implement token classification and QA heads.
- [ ] Add one-block and full-encoder parity tests.
- [ ] Add head parity tests for MLM, NSP/classification, token classification, and QA.
- [ ] Add packed-QKV rewrite under self-attention guards.
- [ ] Add fused SDPA/FlashAttention rewrite under dense noncausal no-attention-output guards.
- [ ] Benchmark encoder, FFN, attention, and vocab projection sweeps.

## Gated gaps

- `LayerNorm` is required and currently listed as unported in DinoML op memory; Megatron-BERT cannot be meaningfully admitted without it.
- Dense encoder attention needs either composed BMM/softmax/BMM lowering or a fused SDPA provider; current DinoML notes still list attention matmul chains as open.
- Embedding gather for integer token IDs is required; DinoML has gather-like tensor ops but this needs an embedding/table-lookup frontend and int input ABI.
- GELU is present, but `gelu_new` must be verified against HF `ACT2FN` before admitting Erlangshen/BioMegatron configs.
- Mask conversion semantics must be made artifact-visible: 2D/3D masks to additive broadcast masks, dtype handling, and large negative constants.
- Tied weight aliasing must survive loading/lowering so MLM decoder and input embedding remain one logical parameter.
- Optional decoder/cache/cross-attention paths are implemented in source but should be rejected for first encoder/MLM integration unless separately scoped.
