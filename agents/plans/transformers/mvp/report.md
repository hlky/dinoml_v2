# Transformers MVP Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: RUCAIBox/mvp plus task variants
Config source: local MvpConfig plus public RUCAIBox config.json files
Source files inspected:
  transformers/src/transformers/models/mvp/configuration_mvp.py
  transformers/src/transformers/models/mvp/modeling_mvp.py
  transformers/src/transformers/models/mvp/__init__.py
Any missing files or assumptions:
  No family-local tokenizer files. MvpTokenizer resolves to RobertaTokenizer.
  No generation_config.json observed for sampled RUCAIBox checkpoints.
```

Representative config notes are in `_sources/config_sweep.md`. Primary runtime target for DinoML: `MvpForConditionalGeneration` seq2seq generation. Optional heads: sequence classification, extractive QA, and decoder-only `MvpForCausalLM`.

## 2. High-level architecture

MVP is a BART-like text-only encoder-decoder with learned absolute position embeddings and optional layer-wise learned prompts. The default production shape is 12 encoder layers, 12 decoder layers, hidden width 1024, 16 attention heads, and FFN width 4096.

```text
tokenization -> input_ids/attention_mask
  -> encoder embeddings + learned positions -> encoder self-attention stack
  -> decoder embeddings + learned positions -> causal self-attention + cross-attention stack
  -> LM head + final_logits_bias -> logits -> generation controller
```

Independently stageable regions: tokenizer/data pipeline, encoder prefill, decoder prefill, decoder incremental decode, task heads, and optional prompt K/V construction. Encoder outputs and cross-attention K/V can be cached across decode steps.

## 3. Important config dimensions

| Field | Source default / observed RUCAIBox value | Runtime effect |
| --- | --- | --- |
| `vocab_size` | 50267 | embedding rows, LM logits width |
| `d_model` / `hidden_size` | 1024 | hidden width |
| `encoder_layers` / `decoder_layers` | 12 / 12 | stack depth |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA heads |
| inferred `head_dim` | 64 | `d_model // heads` |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 | FFN inner width |
| `max_position_embeddings` | 1024 | learned absolute position table, plus offset 2 |
| `activation_function` | `gelu` | FFN activation |
| `scale_embedding` | false | no sqrt(hidden) embed scaling for sampled configs |
| `use_cache` | true | decoder cache enabled for generation |
| `use_prompt` | base false, task variants true | adds per-layer prompt K/V prefixes |
| `prompt_length` / `prompt_mid_dim` | 100 / 800 when present; same as source default | prompt MLP dimensions |
| token ids | pad 1, BOS 0, EOS 2, decoder start 2 | shift-right and generation control |

Representative checkpoint sweep:

| Checkpoint | First useful task | Structural variation |
| --- | --- | --- |
| `RUCAIBox/mvp` | generic seq2seq generation | no prompt path |
| `RUCAIBox/mvp-summarization` | summarization | prompt path enabled |
| `RUCAIBox/mvp-data-to-text` | data-to-text generation | prompt path enabled |
| `RUCAIBox/mvp-question-answering` | generative QA; optional span QA head in source | prompt path enabled |
| `RUCAIBox/mvp-story` | story generation | prompt path enabled |
| `RUCAIBox/mvp-multi-task` | multi-task text generation | prompt path enabled |

## 3a. Family variation traps

- `use_prompt` changes the attention ABI: prompt-enabled attention prepends learned prompt K/V states of length `prompt_length` and expands masks by the same amount.
- Source-basis trap: in this checkout, `MvpModel.__init__` calls `MvpEncoder(config, config.use_prompt)` positionally, so `config.use_prompt` lands in the encoder's `embed_tokens` parameter rather than the encoder's `use_prompt` parameter. The top-level source path therefore appears to enable decoder prompts but not encoder prompts. DinoML should match this source behavior for parity unless a checkpoint-specific remote implementation or upstream fix is explicitly chosen.
- Configs omit `prompt_length` and `prompt_mid_dim` for `RUCAIBox/mvp`; effective defaults are source defaults only when `use_prompt` is later enabled.
- MHA only: no GQA/MQA. Require `d_model % num_heads == 0`.
- Absolute learned positions use a BART-style `+2` offset and decoder positions include `past_key_values_length`.
- Classification pooling uses boolean EOS indexing and requires the same number of EOS tokens in every batch row.
- `MvpForCausalLM` mutates config to `is_decoder=True` and `is_encoder_decoder=False`; treat it as a separate decoder-only admission target.
- Generation controls such as forced BOS/EOS, beam count, and no-repeat ngram are controller ABI, not neural graph ops.
- No image/audio/video layout axes. A layout pass should guard the whole model as sequence-major `[B, T, C]`; do not apply NHWC-style translations.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs, attention masks, and position arange/expand.
- Embedding lookup for tokens and learned positions.
- Reshape/view/transpose for `[B,T,C] <-> [B,H,T,D]` and flattened `[B*H,T,D]`.
- Concatenate on prompt/cache sequence axes; split prompt layer tensors and QA logits.
- Boolean/equality mask, `masked_fill` for `-100 -> pad`, EOS boolean indexing for classification.
- Squeeze/contiguous for QA start/end logits.

Neural primitives:

- Biased Linear/GEMM: Q/K/V/O `1024 -> 1024`, FFN `1024 -> 4096 -> 1024`, prompt MLP `1024 -> 800 -> layers*2*1024`, classifier `1024 -> 1024 -> num_labels`, QA `1024 -> 2`, LM `1024 -> 50267`.
- LayerNorm over hidden width 1024.
- GELU, tanh for classifier, residual add, dropout as inference no-op.
- Optional fp16 clamp guard after encoder layers.

Attention/generation:

- Encoder bidirectional self-attention.
- Decoder causal self-attention with KV cache.
- Decoder cross-attention over encoder states with reusable cross-attention cache.
- Mask add before softmax, softmax on last axis, attention value BMM.
- Cache reorder for beam search and cache reset/reuse semantics.

Position/custom math:

- Learned absolute position embedding with offset 2.
- `shift_tokens_right` generation/training helper.

## 5. Layer/block breakdown

Encoder embeddings:

```text
input_ids [B,S] -> token embedding [B,S,1024]
positions arange(0,S)+2 -> position embedding [B,S,1024]
x = LayerNorm(token + position)
```

Encoder block, repeated 12:

```text
q,k,v = Linear(x), Linear(x), Linear(x), each [B,S,1024]
q,k,v -> [B,16,S,64] -> [B*16,S,64]
attn = softmax((q * 1/sqrt(64)) @ k.T + mask)
x = LayerNorm(x + Linear(attn @ v))
x = LayerNorm(x + Linear(GELU(Linear(x))))
```

Decoder block, repeated 12:

```text
self_attn(x, causal_mask, self_cache, optional self_prompt)
x = LayerNorm(residual + self_attn_out)
cross_attn(x, encoder_hidden, encoder_mask, cross_cache, optional cross_prompt)
x = LayerNorm(residual + cross_attn_out)
x = LayerNorm(x + Linear(GELU(Linear(x))))
```

LM head:

```text
logits = Linear(decoder_hidden, tied embedding weight) + final_logits_bias
```

## 6. Attention requirements

MVP requires dense MHA. Encoder attention is bidirectional self-attention. Seq2seq decoder attention combines causal self-attention and rectangular cross-attention from target length to source length. Both use 16 heads and head dim 64 for sampled configs.

Cache ABI:

- Self-attention stores per-layer K/V shaped `[B, H, past_tgt, D]`; new token decode appends along target sequence.
- Cross-attention stores per-layer K/V shaped `[B, H, src_len, D]`; it is computed once from encoder hidden states and reused after `EncoderDecoderCache.is_updated[layer_idx]` is true.
- Prompt K/V, when enabled, is prepended before normal K/V. DinoML should either materialize prompt into cache prefix once or add explicit prompt-prefix guards so cache lengths and masks include prompt length.
- Beam search needs cache reorder over batch/beam dimension. The MVP source relies on Transformers cache infrastructure rather than a family-local `_reorder_cache`.

No sliding window, sparse attention, ALiBi, RoPE, packed varlen metadata, or local/block attention is present.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding:

```python
def mvp_position_ids(seq_len, past_len=0):
    ids = arange(past_len, past_len + seq_len)
    return ids + 2
```

Shift-right helper:

```python
def shift_tokens_right(x, pad_id, start_id):
    y = zeros_like(x)
    y[:, 1:] = x[:, :-1]
    y[:, 0] = start_id
    y = where(y == -100, pad_id, y)
    return y
```

Prompt construction, if `use_prompt`:

```python
prompt = Embedding(arange(P))           # [P, C]
prompt = Linear(GELU(Linear(prompt)))   # [P, layers * 2 * C]
prompt = prompt.view(P, layers*2, H, D).permute(1, 2, 0, 3).split(2)
```

For inference, prompt tensors are deterministic constants for a loaded checkpoint and can be precomputed per stack.

## 8. Preprocessing and input packing

MVP consumes tokenizer-produced `input_ids [B,S]` and optional `attention_mask [B,S]`. Tokenizer config sampled from public checkpoints only sets `model_max_length=1024`; the tokenizer implementation is Roberta-style BPE via `MvpTokenizer` alias.

Decoder inputs may be supplied directly. If omitted in seq2seq mode, source `input_ids` are shifted right with start token `eos_token_id`/`decoder_start_token_id=2`. Labels disable cache and are training-only for first inference integration.

Classification and extractive QA heads are not required for first seq2seq generation parity. Classification pooling depends on EOS positions in source `input_ids`; QA span logits are over decoder hidden positions because the shared `MvpModel` output is decoder last hidden state.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern: three biased `Linear(1024,1024)` calls on the same hidden tensor.

Replacement: one `Linear(1024,3072)` followed by split `[q,k,v]`.

Preconditions: same input, same dtype/device, all three projections present, weight rows packed in Q then K then V order, bias packed in the same order. Cross-attention can pack K/V but Q input differs from key/value input.

Failure cases: prompt/cross-attention cache reuse can skip K/V projection; decode should use separate or guarded packed paths.

### Rewrite: attention BMM chain -> fused attention

Source pattern: reshape heads, `bmm(q, k.T)`, mask add, softmax last dim, `bmm(prob, v)`.

Replacement: dense SDPA/FlashAttention-style kernel.

Preconditions: dense MHA, no output attentions, dropout disabled, masks canonicalized to additive form, prompt prefix and cache lengths included in K/V length. Causal only for decoder self-attention; cross-attention is noncausal rectangular.

Failure cases: `output_attentions=True`, arbitrary mask shape mismatch, or dynamic prompt/cache length without matching mask guard.

### Rewrite: prompt MLP -> constant prompt tensors

Source pattern: `MvpPrompt(arange(prompt_length))` every forward.

Replacement: precompute each layer's prompt K/V tensor as constants at load time.

Preconditions: inference-only, prompt weights frozen, fixed `prompt_length`, fixed layer/head dimensions.

Failure cases: lightweight tuning or mutable prompt weights.

### Rewrite: cross-attention K/V precompute

Source pattern: during decode, cross-attention K/V are projected from fixed encoder hidden states and then cached.

Replacement: project all decoder-layer cross K/V after encoder once, store in decode cache.

Preconditions: encoder hidden state fixed for request, no prompt mutation, no layerdrop/training.

Failure cases: caller supplies changed encoder hidden states without cache reset.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual add for encoder/decoder blocks. DinoML currently has this as a gated gap for transformer parity.
- GEMM bias epilogues for all Linear layers, especially Q/K/V/O, FFN, and LM head.
- Dense attention prefill/decode kernel with mask and cache support.
- KV cache append/reorder plus cross-attention cache reuse.

Medium priority:

- Packed QKV projection for self-attention prefill.
- Cross-attention K/V projection cache.
- Prompt prefix precompute and prompt+attention mask fusion.
- Last-token-only logits for decode and `logits_to_keep` for decoder-only MVP.

Lower priority:

- Classification EOS pooling lowering.
- QA split/squeeze postprocessing.
- fp16 finite clamp guard, likely only needed for strict parity in reduced precision.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `RUCAIBox/mvp` without prompt. Verify tied embedding/LM head aliases and final logits bias.

Stage 2: encoder-only parity for embeddings, learned positions, LayerNorm, MHA, FFN.

Stage 3: decoder prefill with causal self-attention and cross-attention, no cache.

Stage 4: seq2seq generation decode with self KV cache and cross KV cache reuse.

Stage 5: prompt-enabled task checkpoints by matching the inspected source's decoder-prompt behavior first, then only enabling encoder prompts behind a source-version guard if a target checkpoint actually requires it.

Stage 6: optimized attention/QKV/LM-head rewrites.

Stage 7: optional sequence classification, QA, and decoder-only causal LM wrappers.

## 12. Parity and validation plan

- Config roundtrip tests for base and prompt checkpoints.
- Random tensor unit tests for learned positional offset, shift-right, prompt tensor construction, and additive mask shapes.
- Single encoder layer parity in fp32 with fixed dropout disabled.
- Single decoder layer parity for self-attention, cross-attention, and prompt/no-prompt variants.
- Full encoder-decoder prefill logits parity against Transformers for short and max-ish sequence buckets.
- Incremental decode parity: first token with empty cache, second token with self/cross cache reuse, beam reorder if generation controller supports beams.
- Prompt checkpoint parity for at least one `use_prompt=true` model.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16/bf16 start at `rtol=5e-2, atol=5e-2` until fused attention numerics are characterized.

## 13. Performance probes

- Encoder throughput by source length and batch.
- Decoder prefill throughput by target length and source length.
- Decode tokens/sec with and without cross-attention cache precompute.
- Prompt-enabled versus no-prompt decode overhead.
- LM head cost for full logits versus last-token-only.
- Cache memory by batch, beam, source length, target length, and prompt length.
- Attention backend comparison: unfused BMM chain, SDPA, and future custom cache-aware kernel.
- Weight-load experiment for tied embedding/LM head and possible GGUF/dense materialization later.

## 14. Skip/defer list

- Training losses, dropout, gradient checkpointing, and LayerDrop.
- Beam-search controller details beyond cache reorder ABI.
- No-repeat-ngram and forced BOS/EOS generation processors for neural graph lowering.
- Sequence classification and extractive QA heads for first seq2seq target.
- Decoder-only `MvpForCausalLM` until seq2seq decoder/cache path is stable.
- Quantization and multi-GPU tensor parallel behavior.

## 15. Final implementation checklist

- [ ] Parse `MvpConfig` including prompt fields and generation-control metadata.
- [ ] Load/token embedding, position embedding, LM head, and `final_logits_bias` with tied-weight alias checks.
- [ ] Implement or admit embedding lookup for token and position ids.
- [ ] Implement LayerNorm over hidden width 1024.
- [ ] Lower Linear/GEMM bias paths for attention, FFN, prompt MLP, classifier, QA, and LM head.
- [ ] Implement dense MHA prefill with additive masks and softmax on last axis.
- [ ] Implement decoder self-attention KV cache append and beam reorder.
- [ ] Implement cross-attention K/V cache reuse.
- [ ] Add `shift_tokens_right` and learned position offset helpers.
- [ ] Add prompt K/V precompute and mask-length guards for `use_prompt=true`, with an explicit source-version guard for decoder-only versus encoder+decoder prompt activation.
- [ ] Add seq2seq prefill and decode parity tests.
- [ ] Add gated optional heads: sequence classification EOS pooling, QA split/squeeze, decoder-only CausalLM.
- [ ] Benchmark encoder, prefill, decode, prompt overhead, LM head, and cache memory.
