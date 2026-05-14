# DistilBERT family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: distilbert
Primary runtime target: base bidirectional text encoder, then masked-LM/classification heads
Config source: DistilBertConfig plus representative Hugging Face config.json files
Source files inspected:
- X:/H/transformers/src/transformers/models/distilbert/modeling_distilbert.py
- X:/H/transformers/src/transformers/models/distilbert/configuration_distilbert.py
- X:/H/transformers/src/transformers/models/distilbert/tokenization_distilbert.py
- X:/H/transformers/src/transformers/masking_utils.py for create_bidirectional_mask
Source snapshots:
- agents/plans/transformers/distilbert/_sources/modeling_distilbert.py
- agents/plans/transformers/distilbert/_sources/configuration_distilbert.py
- agents/plans/transformers/distilbert/_sources/tokenization_distilbert.py
Any missing files or assumptions:
- No remote-code files are required for the inspected in-library family.
- No processor/preprocessor config is used; tokenization is BertTokenizer-derived WordPiece.
- Attempted config URL https://huggingface.co/google/distilbert-base-uncased/raw/main/config.json returned an authentication-style error. Public configs under the distilbert namespace were used instead.
```

Pinned source URLs:

- [modeling_distilbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/distilbert/modeling_distilbert.py)
- [configuration_distilbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/distilbert/configuration_distilbert.py)
- [tokenization_distilbert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/distilbert/tokenization_distilbert.py)

Representative configs saved under `_sources/`:

- [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- [distilbert/distilbert-base-cased](https://huggingface.co/distilbert/distilbert-base-cased)
- [distilbert/distilbert-base-multilingual-cased](https://huggingface.co/distilbert/distilbert-base-multilingual-cased)
- [distilbert/distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad)
- [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- [hf-internal-testing/tiny-random-distilbert](https://huggingface.co/hf-internal-testing/tiny-random-distilbert)

## 2. High-level architecture

DistilBERT is a text-only bidirectional encoder. It removes several BERT components: no token type/segment embeddings, no pooler module in the base model, fewer encoder layers by default, and no autoregressive generation/cache path.

```text
WordPiece tokenization -> input_ids/attention_mask
input_ids -> word embedding + position embedding -> LayerNorm -> dropout
encoder block x n_layers -> last_hidden_state
optional task head -> logits
```

The useful DinoML decomposition is:

- CPU/data pipeline: WordPiece tokenization, special token insertion, padding/truncation, attention_mask creation.
- GPU/runtime encoder: embedding lookup, position embedding lookup, LayerNorm, bidirectional MHA, FFN, residual post-norm blocks.
- Independently stageable heads: MLM, sequence classification, token classification, QA, multiple choice.
- No prefill/decode split: this is not an autoregressive decoder and does not expose a KV cache.

## 3. Important config dimensions

Source defaults from `DistilBertConfig`:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 30522 | Word embedding rows and MLM output width |
| `max_position_embeddings` | 512 | Position embedding rows |
| `sinusoidal_pos_embds` | false | If true, initializes fixed sinusoidal position weights |
| `n_layers` | 6 | Encoder block count |
| `n_heads` | 12 | MHA query/key/value head count |
| `dim` / `hidden_size` | 768 | Hidden width |
| `head_dim` | 64 | Inferred as `dim // n_heads`; source rejects non-divisible configs |
| `hidden_dim` | 3072 | FFN inner width |
| `activation` | gelu | FFN and MLM transform activation via `get_activation` |
| `dropout` | 0.1 | Embedding/FFN/token-classifier dropout in training |
| `attention_dropout` | 0.1 | Attention probability dropout in training |
| `qa_dropout` | 0.1 | QA head dropout in training |
| `seq_classif_dropout` | 0.2 | Sequence/multiple-choice head dropout in training |
| `pad_token_id` | 0 | Embedding padding index |
| `tie_word_embeddings` | true | MLM projector weight tied to input word embeddings when tying is applied |
| cache support | none | No `past_key_values`, `use_cache`, or generation cache in modeling source |

Representative checkpoint sweep:

| Model id | Architecture | Layers | Dim | Heads | Head dim | FFN | Max pos | Vocab | Position mode | Head/task notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `distilbert/distilbert-base-uncased` | `DistilBertForMaskedLM` | 6 | 768 | 12 | 64 | 3072 | 512 | 30522 | learned | Common uncased MLM |
| `distilbert/distilbert-base-cased` | `DistilBertForMaskedLM` | 6 | 768 | 12 | 64 | 3072 | 512 | 28996 | learned | Cased vocab changes only |
| `distilbert/distilbert-base-multilingual-cased` | `DistilBertForMaskedLM` | 6 | 768 | 12 | 64 | 3072 | 512 | 119547 | learned | Large multilingual vocab |
| `distilbert/distilbert-base-uncased-distilled-squad` | `DistilBertForQuestionAnswering` | 6 | 768 | 12 | 64 | 3072 | 512 | 30522 | learned | QA head `dim -> 2` |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | `DistilBertForSequenceClassification` | 6 | 768 | 12 | 64 | 3072 | 512 | 30522 | learned | 2 labels, first-token pooling |
| `hf-internal-testing/tiny-random-distilbert` | `DistilBertForSequenceClassification` | 5 | 32 | 4 | 8 | 37 | 512 | 1124 | learned | Debug-sized, FFN is not `4 * dim` |

Facts in this table come from downloaded `config.json` files except `head_dim`, which is inferred from source behavior. Historical config fields `output_past`, `tie_weights_`, and `hidden_act` appear in some configs but are not read by current `modeling_distilbert.py`; `activation` is the operative activation field.

## 3a. Family variation traps

- No token type embeddings: tokenizer inputs are `input_ids` and `attention_mask`; paired sequences rely on separator tokens, not `token_type_ids`.
- No BERT pooler: sequence and multiple-choice heads implement their own first-token pooling and `pre_classifier`.
- Post-norm blocks: attention/FFN outputs are added to the residual before LayerNorm.
- `hidden_dim` is config-driven and can differ from `4 * dim` in debug checkpoints.
- `dim % n_heads == 0` is required by source; otherwise initialization raises.
- `sinusoidal_pos_embds=True` changes initialization/materialization of position embeddings but the forward path still performs an embedding lookup.
- `output_past` in older configs is ignored by current source. DinoML should not admit it as cache support.
- `output_attentions=True` requires eager attention in current Transformers config machinery; SDPA/Flash paths do not return full attention weights.
- `tie_word_embeddings=True` means MLM `vocab_projector.weight` aliases `distilbert.embeddings.word_embeddings.weight` logically. Preserve this as one logical parameter.
- Layout is text-major `[batch, seq, hidden]` throughout. There is no NHWC/channel-last region.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token input tensors `[B, S]`, optional `inputs_embeds [B, S, D]`.
- Embedding lookup for word and position ids.
- Add, residual add, reshape/view, transpose, contiguous, split/squeeze for QA logits, first-token slice `hidden[:, 0]`.
- Multiple-choice flatten/unflatten: `[B, C, S] -> [B*C, S]`, logits `[B*C, 1] -> [B, C]`.

Neural network primitives:

- LayerNorm over last dim with `eps=1e-12`.
- Linear with bias for all projections and heads.
- GELU by default via `get_activation`; other registered activation strings are possible through config.
- ReLU in sequence/multiple-choice heads.
- Dropout is present in source but inference treats it as identity.

Attention primitives:

- Bidirectional self-attention, MHA only.
- Q/K/V projections are separate dense `Linear(D -> D)` modules with bias.
- Reshape to `[B, H, S, head_dim]`, attention score scale `head_dim**-0.5`, mask add, softmax over keys, value matmul, transpose back, output projection.
- Source advertises eager, SDPA, FlashAttention, and FlexAttention dispatch support through generic Transformers interfaces.

Position ops:

- Learned position embedding lookup by default.
- Optional fixed sinusoidal initialization for `position_embeddings.weight`.
- Default `position_ids` are a non-persistent buffer `[0..max_position_embeddings-1]` sliced to sequence length.

Generation/cache ops:

- None required. No causal mask, no KV cache, no cache reorder, no generation loop.

Preprocessing-coupled ops:

- WordPiece tokenizer derived from BERT.
- Model input names are exactly `input_ids` and `attention_mask`.
- Special-token layout is BERT-like from tokenizer inheritance, but token type ids are not model inputs.

## 5. Layer/block breakdown

Base encoder:

```text
input_ids [B,S] or inputs_embeds [B,S,D]
x = word_embedding(input_ids) or inputs_embeds
pos = position_embedding(position_ids [1,S] or [B,S])
x = LayerNorm(x + pos, eps=1e-12)
x = dropout(x)
mask = create_bidirectional_mask(attention_mask)
for layer in n_layers:
    q = Linear(D -> D, bias)(x).view(B,S,H,dh).transpose(1,2)
    k = Linear(D -> D, bias)(x).view(B,S,H,dh).transpose(1,2)
    v = Linear(D -> D, bias)(x).view(B,S,H,dh).transpose(1,2)
    a = Attention(q,k,v, mask, scale=dh^-0.5, causal=False)
    a = Linear(D -> D, bias)(a.reshape(B,S,D))
    x = LayerNorm(a + x, eps=1e-12)
    f = Linear(D -> hidden_dim, bias)(x)
    f = activation(f)
    f = Linear(hidden_dim -> D, bias)(f)
    f = dropout(f)
    x = LayerNorm(f + x, eps=1e-12)
return x [B,S,D]
```

MLM head:

```text
h = encoder_output [B,S,D]
h = Linear(D -> D, bias)(h)
h = activation(h)
h = LayerNorm(h, eps=1e-12)
logits = Linear(D -> vocab_size, bias)(h)
```

Sequence classification and multiple choice:

```text
pooled = encoder_output[:, 0] [B,D] or [B*C,D]
pooled = Linear(D -> D, bias)(pooled)
pooled = ReLU(pooled)
logits = Linear(D -> num_labels or 1, bias)(pooled)
```

QA and token classification:

```text
QA: logits = Linear(D -> 2, bias)(encoder_output); split last dim into start/end [B,S]
Token classification: logits = Linear(D -> num_labels, bias)(encoder_output)
```

## 6. Attention requirements

DistilBERT requires only encoder-style bidirectional self-attention.

| Requirement | DistilBERT behavior |
|---|---|
| causal/noncausal | Noncausal bidirectional |
| self/cross | Self-attention only |
| MHA/MQA/GQA | MHA; Q, K, V all have `n_heads` |
| head count / KV heads | `n_heads`; no separate KV head count |
| head dim | `dim // n_heads` |
| masks | Optional padding mask becomes bidirectional attention mask; 4D masks may pass through masking utility |
| packed/varlen | No DistilBERT-specific packed sequence ABI; generic attention backends may optimize masks |
| local/sliding | None |
| relative bias/RoPE/ALiBi | None |
| KV cache | None |
| Flash/SDPA compatibility | Source advertises `_supports_flash_attn`, `_supports_sdpa`, `_supports_flex_attn`; first DinoML parity can use dense eager math |

Eager math order to preserve:

```python
attn_weights = (query @ key.transpose(2, 3)) * (head_dim ** -0.5)
attn_weights = attn_weights + attention_mask  # if mask exists
attn_weights = softmax(attn_weights, dim=-1)
attn_output = attn_weights @ value
attn_output = attn_output.transpose(1, 2).contiguous()
```

## 7. Position encoding and custom math

Default checkpoints use learned absolute positions. The optional sinusoidal config only affects initialization/resizing of the same embedding table.

Source-equivalent sinusoidal initializer:

```python
def distilbert_sinusoidal(n_pos, dim):
    pos = np.arange(n_pos)[:, None]
    j = np.arange(dim)[None, :]
    angles = pos / np.power(10000, 2 * (j // 2) / dim)
    out = np.empty((n_pos, dim), dtype=np.float32)
    out[:, 0::2] = np.sin(angles[:, 0::2])
    out[:, 1::2] = np.cos(angles[:, 1::2])
    return out
```

For inference, DinoML can precompute/import the position table as a constant. Dynamic runtime input is only the sequence length or explicit `position_ids`.

## 8. Preprocessing and input packing

Tokenizer coupling is intentionally simpler than BERT:

- `DistilBertTokenizer` subclasses `BertTokenizer`.
- `model_input_names = ["input_ids", "attention_mask"]`; no `token_type_ids`.
- Input shape is `[B, S]`, with `S <= max_position_embeddings` for standard checkpoints.
- Position ids default to `[0, 1, ..., S-1]` broadcast over batch unless explicitly provided.
- Paired text may contain `[SEP]` separators from tokenizer construction, but segment identity is not a runtime tensor.

CPU/data-pipeline:

- WordPiece tokenization, casing behavior, special tokens, padding, truncation.

GPU/runtime:

- `input_ids`, `attention_mask`, optional `position_ids`, optional `inputs_embeds`.
- For first integration, prefer `input_ids + attention_mask` and defer `inputs_embeds`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: DistilBERT embedding block

Source pattern:

```text
word_embedding(input_ids) + position_embedding(position_ids) -> LayerNorm -> dropout
```

Replacement:

```text
EmbeddingGather(word_table, input_ids) + Gather(pos_table, position_ids)
-> LayerNorm(eps=1e-12)
```

Preconditions:

- Inference mode so dropout is identity.
- No token type ids are present.
- `position_ids` are default contiguous slice or an explicit integer tensor.
- Position table is learned or precomputed sinusoidal constant.

Failure cases:

- Training/dropout parity.
- Out-of-range `position_ids` or `input_ids`.

Parity test sketch:

- Compare embedding block output for random input ids, padding ids, and explicit non-default position ids.

### Rewrite: MHA projection packing

Source pattern:

```text
q_lin(x), k_lin(x), v_lin(x) as three Linear(D -> D) modules
```

Replacement:

```text
one GEMM D -> 3D, then split in Q,K,V order
```

Weight transform:

```python
w_qkv = torch.cat([q.weight, k.weight, v.weight], dim=0)  # [3D, D]
b_qkv = torch.cat([q.bias, k.bias, v.bias], dim=0)        # [3D]
```

Preconditions:

- All three projections have identical input/output width and dtype.
- Bias presence matches source, which is true for in-library DistilBERT.
- Split order is exactly Q, K, V.

Failure cases:

- Weight tying or external adapters that replace only one projection.

### Rewrite: FFN activation GEMM epilogue

Source pattern:

```text
Linear(D -> hidden_dim) -> activation -> Linear(hidden_dim -> D)
```

Replacement:

```text
GEMM bias activation epilogue for lin1 when activation is a supported config value
```

Preconditions:

- Inference mode.
- Activation is one DinoML can lower exactly or within declared tolerance.
- `hidden_dim` comes from config, not assumed as `4 * D`.

Failure cases:

- Unsupported activation string.
- Debug configs with unusual `hidden_dim` must still shape correctly.

### Rewrite: first-token classifier pooling

Source pattern:

```text
hidden[:, 0] -> pre_classifier -> ReLU -> classifier
```

Replacement:

```text
static slice on sequence axis -> two linear layers
```

Preconditions:

- Sequence axis is source axis 1 in `[B,S,D]`.
- No layout pass has moved the sequence dimension without rewriting the slice.

No layout translation is needed for this text family. If a future generic layout pass runs over `[B,S,D]`, protect all sequence-axis operations with a no-layout-translation guard.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm: every embedding/block uses last-dim LayerNorm with `eps=1e-12`; post-norm residual+LayerNorm fusion matters.
- Dense MHA prefill: bidirectional encoder attention over full sequence, no cache complexity.
- GEMM epilogues for bias+GELU and bias+ReLU: FFN and classifier heads are small but repeated.
- QKV packed projection: reduces three GEMM launches per layer.

Medium priority:

- Embedding add + LayerNorm fusion: useful for batch throughput and simple to validate.
- MLM head transform + LayerNorm + vocab projection: important for masked-LM checkpoints; preserve tied weights.
- QA split/squeeze: mostly graph cleanup, not a major kernel.

Lower priority:

- FlashAttention/SDPA-specific parity: source supports it, but dense eager attention is enough for first encoder parity.
- Multiple-choice flatten/unflatten specialization: valuable only for that head.
- Sinusoidal resize logic: initialization-time only; not a runtime bottleneck.

## 11. Runtime staging plan

Stage 1: parse config and load base encoder weights, including alias metadata for tied MLM weights.

Stage 2: run embedding block and one `TransformerBlock` parity in fp32 with eager dense attention.

Stage 3: full `DistilBertModel` encoder parity for `input_ids`, default `position_ids`, and padding masks.

Stage 4: add task heads in this order: sequence classification, token classification, QA, MLM, multiple choice.

Stage 5: add safe graph rewrites: QKV packing, residual+LayerNorm fusion, FFN activation epilogue.

Stage 6: enable optimized attention backend for static/bucketed encoder sequence lengths.

Initially stub/defer training losses, dropout, `inputs_embeds`, output attentions, hidden-state recording, and exotic attention backends.

## 12. Parity and validation plan

- Config parsing tests: base, multilingual vocab, SST-2 labels, QA `num_labels=2`, tiny random `hidden_dim=37`.
- Embedding block tests: default positions, explicit positions, padding token ids, learned and synthetic sinusoidal tables.
- Single-block parity: random weights/input ids, no padding and with padding mask.
- Full encoder parity: compare last hidden state after 1, 3, and all layers.
- Head parity: sequence logits, token logits, QA start/end logits, MLM logits with tied weight, multiple-choice reshape.
- Mask parity: all-ones attention mask, mixed padding mask, pre-expanded 4D additive mask if admitted.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 `rtol=5e-2, atol=5e-2` until fused attention/norm numerics are characterized.

## 13. Performance probes

- Encoder throughput by batch size and sequence length: `S = 16, 32, 64, 128, 256, 512`.
- Attention backend comparison: eager dense vs SDPA-like fused attention for bidirectional masks.
- QKV packed projection vs separate Q/K/V GEMMs.
- LayerNorm fusion impact: unfused vs residual+LayerNorm fused.
- MLM vocab projection cost, especially multilingual vocab `119547`.
- Head-only cost for classification/QA/token classification.
- Padding sensitivity: dense full attention with padded batches vs bucketed/batched-by-length admission.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout behavior outside inference.
- Autoregressive generation, beam search, cache reorder, KV cache.
- `output_attentions` except eager debug mode.
- `inputs_embeds` first-class ingestion.
- Remote-code/adapters/quantized fine-tunes not represented in native source.
- FlashAttention/FlexAttention as a first parity requirement.
- Runtime position embedding resize; handle at load/compile time if needed.

## 15. Final implementation checklist

- [ ] Parse `DistilBertConfig`, including `dim`, `n_heads`, `hidden_dim`, `activation`, `sinusoidal_pos_embds`, and task labels.
- [ ] Reject `dim % n_heads != 0`.
- [ ] Treat `output_past`/`tie_weights_`/`hidden_act` as ignored historical fields for this source basis.
- [ ] Load word and position embeddings; precompute sinusoidal positions when configured.
- [ ] Preserve MLM input/output embedding weight aliasing.
- [ ] Implement embedding gather + position add + LayerNorm.
- [ ] Implement bidirectional MHA with Q/K/V/out projections and additive padding mask.
- [ ] Implement post-norm residual attention block.
- [ ] Implement FFN `Linear -> activation -> Linear` with config-driven `hidden_dim`.
- [ ] Implement sequence, token, QA, MLM, and multiple-choice heads.
- [ ] Add QKV packing rewrite with Q,K,V split-order test.
- [ ] Add residual+LayerNorm fusion candidate.
- [ ] Add one-block and full-encoder parity tests against Transformers.
- [ ] Add checkpoint sweep tests for base uncased, multilingual, QA, SST-2, and tiny random configs.
- [ ] Benchmark encoder throughput, attention backend, LayerNorm fusion, and MLM vocab projection.
