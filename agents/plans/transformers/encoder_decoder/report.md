# Transformers audit: encoder_decoder

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: generic wrapper family; representative configs listed below
Config source: local source defaults plus HF raw config.json snapshots
Source files inspected:
- X:/H/transformers/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py
- X:/H/transformers/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py
- X:/H/transformers/src/transformers/models/encoder_decoder/__init__.py
- X:/H/transformers/src/transformers/generation/utils.py
- X:/H/transformers/src/transformers/cache_utils.py
- X:/H/transformers/src/transformers/modeling_utils.py
- Representative delegated BERT sources for cache/cross-attention shape checks:
  X:/H/transformers/src/transformers/models/bert/modeling_bert.py
  X:/H/transformers/src/transformers/models/bert_generation/modeling_bert_generation.py
Any missing files or assumptions:
- No Flax/TF encoder_decoder source exists in this checkout family dir.
- This report owns only the PyTorch `EncoderDecoderModel` wrapper contract.
- Delegated encoder/decoder neural bodies must be audited under their own families.
- Public tiny/debug checkpoint repos tried returned 401; see `_sources/source_notes.md`.
```

HF config snapshots and notes are under `agents/plans/transformers/encoder_decoder/_sources/`.

## 2. High-level architecture

`encoder_decoder` is a composition wrapper, not a fixed neural architecture. It builds or receives:

```text
tokenizer/data pipeline -> encoder AutoModel -> optional enc_to_dec_proj
  -> decoder AutoModelForCausalLM with cross-attention -> logits -> generation controller
```

Runtime ownership is split:

- Wrapper-owned: nested config parsing, encoder/decoder instantiation policy, kwarg routing, optional encoder-to-decoder projection, `labels` shifting for training, `Seq2SeqLMOutput` assembly, and generation ABI participation.
- Encoder-owned: all encoder embeddings, layers, masks, positional math, pooling, and encoder output layout.
- Decoder-owned: decoder embeddings, causal self-attention, cross-attention implementation, LM head, decoder cache layout, and logits.
- GenerationMixin-owned: encoder precompute, decoder start token handling, beam/cache expansion/reorder, stopping/logits processors.

First DinoML target: inference-only seq2seq generation for an allowlisted encoder+decoder pair, initially BERT/BERT-generation style text-to-text. Training loss and generic arbitrary AutoModel composition are deferred.

## 3. Important config dimensions

Wrapper-level fields:

| Field | Source behavior | DinoML impact |
|---|---|---|
| `encoder` | Serialized AutoConfig sub-config; required | Route to separately audited encoder family |
| `decoder` | Serialized AutoConfig sub-config; required | Route to separately audited causal-LM decoder family |
| `is_encoder_decoder` | Wrapper default `True` | Generation path expects encoder precompute and decoder inputs |
| `pad_token_id` | Used by label shift / masks when supplied | Required for training-style label shifting; useful generation metadata |
| `decoder_start_token_id` | Required by generation/label shift | ABI field, not neural op |
| `decoder.is_decoder` | Forced true by `from_encoder_decoder_configs()` | Decoder must be causal/self-cache capable |
| `decoder.add_cross_attention` | Forced true by config factory/from-pretrained helper | Decoder layers must include cross-attention |
| `decoder.cross_attention_hidden_size` | If set, must equal encoder hidden size | Decoder owns adaptation; wrapper projection disabled |
| `tie_encoder_decoder` | Present in some historical configs; current inspected source does not read this name | Treat as ignored for this source basis unless a delegated model audit proves otherwise |

Representative checkpoint sweep:

| Model | Enc/dec type | Enc layers | Dec layers | Hidden | Heads | FFN | Vocab | Start/pad | Notable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mrm8488/bert2bert-mini_shared-question-generation` | bert/bert | 4 | 4 | 256 | 4 | 1024 | 30522 | 101/0 | Small open config; has historical `tie_encoder_decoder=true` |
| `mrm8488/bert2bert-medium_shared-question-generation` | bert/bert | 8 | 8 | 512 | 8 | 2048 | 30522 | 101/0 | Medium hidden/layer scaling |
| `patrickvonplaten/bert2bert_cnn_daily_mail` | bert/bert | 12 | 12 | 768 | 12 | 3072 | 30522 | 101/0 | Common BERT2BERT summarization shape |
| `patrickvonplaten/bert2bert-cnn_dailymail-fp16` | bert/bert | 12 | 12 | 768 | 12 | 3072 | decoder 30522 | 101/null | Config omits top-level pad/vocab in snapshot |
| `mrm8488/bert2bert_shared-spanish-finetuned-summarization` | bert/bert | 12 | 12 | 768 | 12 | 3072 | 31002 | 4/1 | Different vocab and special tokens |
| `google/bert2bert_L-24_wmt_en_de` | bert-generation/bert-generation | 24 | 24 | 1024 | 16 | 4096 | decoder 31950 | 2/0 | L24 translation; SentencePiece/BertGeneration style |

## 3a. Family variation traps

- The wrapper can combine unrelated model families. Operator coverage must come from an explicit allowlist of audited `(encoder_family, decoder_family, decoder_head)` pairs.
- `encoder.hidden_size != decoder.hidden_size` inserts a wrapper-owned dense projection unless the decoder has `cross_attention_hidden_size`.
- If `cross_attention_hidden_size` is present, the wrapper only validates equality to encoder hidden size; the decoder source must prove how cross-attention key/value projections consume that width.
- The encoder must not expose output embeddings / LM head. Checkpoint conversion must route encoder weights to a base encoder class.
- Decoder forward must accept `encoder_hidden_states`; otherwise the wrapper rejects it at construction.
- Configs may contain historical `tie_encoder_decoder=true`; current inspected source basis only ties weights through generic `tie_word_embeddings`/model `_tied_weights_keys`, not this field.
- Decoder start token and tokenizer special-token layout are task ABI. They are not graph ops, but wrong values change generation.
- Cache behavior is decoder-family-specific. The wrapper forwards `past_key_values` but does not define tensor shapes beyond using Transformers `Cache`.
- BERT-style decoder cross-attention caches encoder K/V separately from autoregressive self-attention K/V.
- Source axes are sequence-major logical tensors `[B, S, H]`; no NCHW/NHWC layout translation is relevant for this text wrapper.

## 4. Operator coverage checklist

Wrapper-owned required operators:

- Embedding/output dispatch to nested modules.
- Optional `Linear(encoder_hidden_size -> decoder_hidden_size)` for `enc_to_dec_proj`.
- Tuple/BaseModelOutput normalization for caller-provided `encoder_outputs`.
- Attention mask forwarding from encoder input mask to decoder `encoder_attention_mask`.
- Generation ABI: decoder start token initialization/prepend, encoder-output reuse, cache reorder/expand via GenerationMixin.

Delegated BERT/BertGeneration-style operators for the representative first target:

- Token embedding, position embedding, token-type embedding where present.
- LayerNorm, residual add, dropout-disabled inference path.
- Dense Q/K/V projections with bias, shape `[B,S,H] -> [B,heads,S,head_dim]`.
- Noncausal encoder self-attention.
- Causal decoder self-attention with self KV cache.
- Decoder cross-attention: Q from decoder hidden states, K/V from encoder hidden states.
- GELU FFN: `Linear(H -> intermediate) -> activation -> Linear(intermediate -> H)`.
- LM head projection and optional tied decoder embeddings per delegated decoder.

Cache/metadata operators:

- `EncoderDecoderCache(self_attention_cache, cross_attention_cache)`.
- Cross-attention `is_updated[layer_idx]` flag.
- Beam/cache reorder across both self and cross caches.
- Encoder-output batch expansion for beam/search modes.

No wrapper-owned quantized/packed weight, local attention, RoPE, ALiBi, MoE, convolution, multimodal scatter, or layout conversion ops are present in this source.

## 5. Layer/block breakdown

Wrapper forward:

```text
if encoder_outputs absent:
  encoder_outputs = encoder(input_ids, attention_mask, inputs_embeds, ...)
else:
  normalize tuple -> BaseModelOutput

encoder_hidden_states = encoder_outputs[0]                 # [B, S_src, H_enc]
if H_enc != H_dec and decoder.cross_attention_hidden_size absent:
  encoder_hidden_states = enc_to_dec_proj(encoder_hidden_states)  # [B, S_src, H_dec]

decoder_outputs = decoder(
  input_ids=decoder_input_ids,
  attention_mask=decoder_attention_mask,
  encoder_hidden_states=encoder_hidden_states,
  encoder_attention_mask=attention_mask,
  past_key_values=past_key_values,
  use_cache=use_cache,
)
return Seq2SeqLMOutput(logits, past_key_values, decoder states, encoder states)
```

Representative BERT-style decoder layer:

```text
x = decoder embedding + absolute position embedding
repeat N:
  self_attn = MHA_causal(x, self_cache)
  x = LayerNorm(x + SelfOutput(self_attn))
  cross = MHA_cross(q=x, kv=encoder_hidden_states, cross_cache)
  x = LayerNorm(x + CrossOutput(cross))
  ff = GELU(Linear(H -> I)(x))
  x = LayerNorm(x + Linear(I -> H)(ff))
logits = LMHead(x)
```

For the L24 BertGeneration config: `H=1024`, `heads=16`, `head_dim=64`, `I=4096`, `N=24`.
For BERT base configs: `H=768`, `heads=12`, `head_dim=64`, `I=3072`, `N=12`.

## 6. Attention requirements

Wrapper-level attention contract:

- Encoder attention is whatever the encoder family implements; for BERT targets it is noncausal self-attention over source tokens.
- Decoder self-attention is causal and cacheable when `use_cache=True`.
- Decoder cross-attention is rectangular: queries `[B, T_dec, H_dec]`, keys/values from encoder states `[B, S_src, H_cross]`.
- Source attention mask is forwarded as `encoder_attention_mask`.
- Decoder attention mask is separate and applies to decoder tokens; causal mask is decoder-owned.
- `_supports_flash_attn=True` and `_supports_sdpa=True` are wrapper capability flags, but actual dispatch validity depends on the delegated encoder and decoder implementations.

BERT-style cache shapes after projection:

```text
self K/V per layer:  [B, num_heads, T_cached, head_dim]
cross K/V per layer: [B, num_heads, S_src, head_dim]
```

Cross K/V are computed once per layer and reused while `EncoderDecoderCache.is_updated[layer_idx]` is true. Cached keys/values are after projection and reshape/transpose; BERT-style absolute position embeddings are applied before self-attention projection, not to cached K/V directly.

## 7. Position encoding and custom math

The wrapper has no position encoding math. Position behavior is delegated.

Representative BERT/BertGeneration targets use learned absolute position embeddings with an offset equal to current self-cache sequence length:

```python
def bert_position_ids(seq_len, past_len):
    return arange(past_len, past_len + seq_len)
```

Precompute opportunity: position tables are constants; runtime only slices/gathers by `[past_len : past_len + seq_len]`.

## 8. Preprocessing and input packing

Wrapper input ABI:

- Encoder side: `input_ids` `[B,S_src]`, `attention_mask` `[B,S_src]`, or `inputs_embeds` `[B,S_src,H_enc]`.
- Decoder side: `decoder_input_ids` `[B,T]`, `decoder_attention_mask` `[B,T]`, or `decoder_inputs_embeds` `[B,T,H_dec]`.
- Generation can construct the first decoder token from `decoder_start_token_id`; user-provided decoder inputs may be prepended with it.
- `encoder_outputs` can be supplied directly to skip encoder execution and reuse an independently cached encoder stage.

Tokenizer and language control are checkpoint-specific. BERT2BERT configs use WordPiece-like vocab files; BertGeneration L24 uses a generation config and SentencePiece-style BertGeneration tokenizer behavior from the delegated family. DinoML should treat tokenizer, forced BOS/EOS, beam settings, and suppress/logits processors as generation-controller metadata, not wrapper graph ops.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split encoder and decoder artifacts

Source pattern:

```text
EncoderDecoderModel.forward(input_ids, decoder_input_ids, ...)
```

Replacement:

```text
encoder_artifact(input_ids, attention_mask) -> encoder_last_hidden_state
optional proj_artifact/state
decoder_artifact(decoder_input_ids, encoder_last_hidden_state, masks, cache) -> logits, cache
```

Preconditions:

- Encoder family and decoder family are both admitted.
- Output ABI preserves `encoder_last_hidden_state` dtype/shape and mask semantics.
- Generation controller owns batch expansion/reorder consistently for encoder states and caches.

Failure cases:

- Decoder has custom cross-attention width or cache ABI not audited.
- Encoder output is not a plain `[B,S,H]` sequence.
- Generation mode requires unsupported logits processors or beam behavior.

Parity test sketch: compare encoder output, one decoder prefill, then one cached decode step against HF for fixed random ids and masks.

### Rewrite: wrapper `enc_to_dec_proj` as explicit Linear

Preconditions:

- `encoder.hidden_size != decoder.hidden_size`.
- `decoder.cross_attention_hidden_size is None`.
- Projection weight key exists under wrapper module.

Replacement:

```text
encoder_hidden_states [B,S,H_enc] -> GEMM/Bias -> [B,S,H_dec]
```

Failure cases:

- Decoder owns `cross_attention_hidden_size`; do not add wrapper projection.
- Encoder output rank/layout differs from `[B,S,H]`.

### Rewrite: precompute cross-attention K/V after encoder

Source pattern:

```text
each decode step/layer crossattention computes K,V from encoder_hidden_states unless cache marked updated
```

Replacement:

```text
after encoder/projection: per decoder layer compute cross_k, cross_v once
decode step consumes cached cross_k/cross_v
```

Preconditions:

- Decoder cross-attention projection modules are static and audited.
- Encoder states do not change across decode.
- Beam expansion/reorder of cross cache matches HF `EncoderDecoderCache`.

Failure cases:

- Decoder applies dynamic conditioning, adapters, or per-step transforms inside cross-attention.
- Cross cache storage shape/order differs from audited family.

### Rewrite: last-token-only LM head for decode

Preconditions:

- Cached decode step emits only new token hidden state or can slice last hidden before LM head.
- Generation does not request full decoder hidden states/logits for all positions.

Replacement:

```text
LMHead([B,1,H]) instead of LMHead([B,T,H])
```

Failure cases:

- Prefill logits for all labels requested.
- Output hidden-state/attention debug modes require full sequence.

## 10. Kernel fusion candidates

Highest priority:

- Delegated decoder self-attention with KV cache and cross-attention K/V reuse; this dominates generation.
- LayerNorm + residual + dense epilogues for BERT-style blocks.
- FFN GELU block fusion or GEMM epilogue activation.
- Wrapper `enc_to_dec_proj` through existing GEMM if mixed-width pairs are admitted.

Medium priority:

- Cross-attention K/V preprojection immediately after encoder.
- Encoder-only throughput fusion for admitted encoder family.
- Last-token LM head in decode.
- Mask canonicalization for encoder mask and decoder causal mask.

Lower priority:

- Training label shifting and loss.
- Generic arbitrary AutoModel composition.
- Rare delegated families with nonstandard cache/attention.

## 11. Runtime staging plan

Stage 1: Admit config/weight loading for one explicit pair, e.g. BERT encoder + BERT causal-LM decoder with cross-attention. Reject all other pairs with a clear wrapper admission error.

Stage 2: Run encoder-only parity and serialize/cache `encoder_last_hidden_state`.

Stage 3: Run decoder prefill with encoder states and no cache optimization beyond dense attention parity.

Stage 4: Add `EncoderDecoderCache`-equivalent ABI for self K/V and cross K/V, including beam reorder and batch expansion.

Stage 5: Add cross-K/V precompute and last-token decode.

Stage 6: Enable SDPA/Flash-style backend only when delegated attention math and mask layout match.

Stage 7: Broaden allowlist to BertGeneration L24 and other audited text encoder/decoder pairs.

## 12. Parity and validation plan

- Config parser tests for nested encoder/decoder configs, missing subconfigs, decoder flags, and hidden-size projection rules.
- Constructor/admission tests:
  - reject encoder with LM head,
  - reject decoder without `encoder_hidden_states`,
  - reject unaudited encoder/decoder family,
  - reject unsupported `cross_attention_hidden_size`.
- Single wrapper projection parity for random `[B,S,H_enc]`.
- Encoder-only parity for admitted pair at fp32 and target inference dtype.
- Decoder prefill parity with fixed encoder states and masks.
- Cached decode parity for two steps: verify logits and self/cross cache lengths.
- Beam reorder parity: reorder both self and cross caches and compare next-token logits.
- End-to-end greedy generation on a tiny synthetic or small open BERT2BERT config once weights are loadable.

Recommended tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 start with `atol=5e-2, rtol=5e-2` for end-to-end logits, then tighten per op.

## 13. Performance probes

- Encoder throughput by source length and batch size.
- Wrapper projection cost when `H_enc != H_dec`.
- Decoder prefill latency with rectangular cross-attention.
- Decode tokens/sec with and without cross-K/V precompute.
- Cache memory split: self-attention grows with generated length; cross-attention is fixed at source length.
- Beam-size sweep: encoder-output expansion and cache reorder overhead.
- LM head full-sequence versus last-token-only decode.
- Attention backend comparison for admitted decoder cross/self attention.

## 14. Skip/defer list

- Training loss, `labels`, and `shift_tokens_right` parity beyond metadata checks.
- Generic arbitrary AutoModel/AutoModelForCausalLM composition.
- Unsupported delegated families, including multimodal, speech, vision, state-space, sparse attention, or custom remote-code decoders.
- Historical `tie_encoder_decoder` behavior unless current source support is proven.
- Resizing embeddings through wrapper; source explicitly raises `NotImplementedError`.
- Output attentions/hidden states for first optimized runtime path.
- Beam search can follow greedy cached decode; do not block first prefill/decode parity on full generation controller parity.

## 15. Final implementation checklist

- [ ] Parse `EncoderDecoderConfig` with nested `encoder` and `decoder` subconfigs.
- [ ] Add wrapper admission allowlist for exact delegated encoder/decoder families.
- [ ] Reject encoder modules with output embeddings / LM head.
- [ ] Reject decoders without cross-attention-capable `encoder_hidden_states` contract.
- [ ] Preserve decoder `is_decoder=True` and `add_cross_attention=True` requirements.
- [ ] Load wrapper-owned `enc_to_dec_proj` when required.
- [ ] Represent `encoder_outputs` as an independently cacheable runtime value.
- [ ] Define DinoML encoder-decoder cache ABI with self cache and cross cache.
- [ ] Implement BERT-style prefill parity for one admitted pair.
- [ ] Implement BERT-style cached decode parity for one admitted pair.
- [ ] Add beam/cache reorder tests.
- [ ] Add cross-K/V precompute rewrite behind an audited decoder-family guard.
- [ ] Add last-token LM head decode rewrite behind generation-output guards.
- [ ] Benchmark encoder, projection, prefill, decode, and cache memory separately.

## Gated gaps / admission policy

Default policy: reject `encoder_decoder` unless the exact encoder family, decoder family, decoder head, cache ABI, and cross-attention width rule are allowlisted by separate audits.

Gates before first DinoML support:

- A delegated encoder audit for the chosen encoder body.
- A delegated decoder audit proving cross-attention, LM head, masks, and cache shapes.
- A wrapper-level manifest schema for `encoder_outputs`, optional projection, decoder start token metadata, and cache ownership.
- Explicit handling for `cross_attention_hidden_size`: either reject initially or admit only decoder families that implement it with source-proven K/V projection widths.
- Weight alias policy: ignore historical `tie_encoder_decoder` for this source basis, but preserve real tied weights discovered through delegated model tied-weight metadata.
- Generation-controller boundary: decide which generation features are in DinoML runtime versus caller/controller code.
