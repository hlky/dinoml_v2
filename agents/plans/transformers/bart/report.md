# BART Transformers Audit

## 1. Source basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary `facebook/bart-large-cnn`; representative sweep also used `facebook/bart-base`, `facebook/bart-large`, `facebook/bart-large-xsum`, and `sshleifer/distilbart-cnn-12-6`.

Config source:

- `https://huggingface.co/facebook/bart-base/raw/main/config.json`
- `https://huggingface.co/facebook/bart-large/raw/main/config.json`
- `https://huggingface.co/facebook/bart-large-cnn/raw/main/config.json`
- `https://huggingface.co/facebook/bart-large-cnn/raw/main/generation_config.json`
- `https://huggingface.co/facebook/bart-large-xsum/raw/main/config.json`
- `https://huggingface.co/facebook/bart-large-xsum/raw/main/generation_config.json`
- `https://huggingface.co/sshleifer/distilbart-cnn-12-6/raw/main/config.json`

Source files inspected:

- `transformers/src/transformers/models/bart/configuration_bart.py`
- `transformers/src/transformers/models/bart/modeling_bart.py`
- `transformers/src/transformers/models/bart/tokenization_bart.py`
- `transformers/src/transformers/models/roberta/tokenization_roberta.py`
- `transformers/src/transformers/masking_utils.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/generation/utils.py`

Any missing files or assumptions:

- `tokenization_bart_fast.py` does not exist at this commit. `tokenization_bart.py` aliases both `BartTokenizer` and `BartTokenizerFast` to `RobertaTokenizer`.
- Some older checkpoint configs include legacy fields such as `normalize_embedding`, `add_final_layer_norm`, `output_past`, `extra_pos_embeddings`, and `force_bos_token_to_be_generated`; the inspected current `BartConfig` does not declare most of these as model-structure fields.
- Primary runtime target for this report is `BartForConditionalGeneration` for seq2seq summarization/generation. Classification, QA, and decoder-only `BartForCausalLM` are lower-priority heads.

## 2. High-level architecture

BART is a text-only encoder-decoder Transformer. The encoder uses bidirectional self-attention. The decoder uses causal self-attention and encoder-decoder cross-attention. Both sides use learned absolute position embeddings, shared byte-level BPE token embeddings, post-attention/post-MLP LayerNorm, GELU FFNs, and dense vocabulary logits.

Dataflow:

```text
byte-level BPE tokenizer -> input_ids/attention_mask
  -> shared token embedding + learned encoder positions -> encoder bidirectional blocks
  -> decoder start/eos prompt + learned decoder positions -> decoder causal self-attn
  -> decoder cross-attn over encoder states -> lm_head tied to shared embedding
  -> final_logits_bias -> generation controller / beam search
```

Stage decomposition:

- CPU/data pipeline: byte-level BPE tokenization, special token insertion, padding, attention mask creation, generation controller constraints such as beams and no-repeat n-grams.
- Encoder: independently cacheable for fixed source text; output shape `[batch, src_len, d_model]`.
- Decoder prefill: causal decoder over initial decoder ids, plus cross-attn K/V projection from encoder states.
- Decode: one or more new decoder tokens with growing self-attn KV cache and reusable cross-attn cache.
- Logits/sampling: `lm_head(hidden) + final_logits_bias`; generation policy is outside the core module graph but required for end-to-end summarization parity.

## 3. Important config dimensions

Source-default `BartConfig` fields:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 50265 | Shared embedding rows and LM head output columns |
| `max_position_embeddings` | 1024 | Learned encoder/decoder positions allocate `max_position_embeddings + 2` rows |
| `d_model` | 1024 | Hidden width |
| `encoder_layers` / `decoder_layers` | 12 / 12 | Repeated block counts |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA heads |
| `head_dim` | inferred 64 | `d_model / heads`; source requires exact divisibility |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 | FFN expansion |
| `activation_function` | `gelu` | FFN activation through `ACT2FN` |
| `dropout` / `attention_dropout` / `activation_dropout` | 0.1 / 0.0 / 0.0 | Inference dropout disabled |
| `scale_embedding` | false | If true, token embeddings multiply by `sqrt(d_model)` |
| `use_cache` | true | Generation uses KV caches |
| `pad/bos/eos/decoder_start` | 1 / 0 / 2 / 2 | Token/mask/generation contracts |
| `forced_eos_token_id` | 2 | Generation termination constraint |
| `_attn_implementation` | config/framework default | Dispatches eager, SDPA, FlashAttention, or flex masks |

Representative checkpoint sweep:

| Model id | Arch in config | `d_model` | Enc/Dec layers | Heads | FFN | Vocab | Max pos | Gen config highlights |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `facebook/bart-base` | `BartModel` | 768 | 6 / 6 | 12 / 12 | 3072 | 50265 | 1024 | config has task params; summary beams 4, CNN max/min 142/56 |
| `facebook/bart-large` | `BartModel` | 1024 | 12 / 12 | 16 / 16 | 4096 | 50265 | 1024 | task params; beams 4 or 6 depending task |
| `facebook/bart-large-cnn` | `BartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 50264 | 1024 | beams 4, max 142, min 56, length penalty 2.0, no-repeat 3 |
| `facebook/bart-large-xsum` | `BartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 50264 | 1024 | beams 6, max 62, min 11, no-repeat 3 |
| `sshleifer/distilbart-cnn-12-6` | `BartForConditionalGeneration` | 1024 | 12 / 6 | 16 / 16 | 4096 | 50264 | 1024 | asymmetric decoder depth; beams 4, max/min 142/56 |

Omitted checkpoint fields use current `BartConfig` defaults: `tie_word_embeddings=True`, `is_encoder_decoder=True`, `is_decoder=False`, `classifier_dropout=0.0`, `init_std=0.02`, and `use_cache=True` unless overridden.

## 3a. Family variation traps

- `vocab_size` differs: base/large configs use 50265, CNN/XSum/distil summaries use 50264. Do not hard-code LM head width.
- Encoder and decoder depth can be asymmetric, as in `sshleifer/distilbart-cnn-12-6` with 12 encoder layers and 6 decoder layers.
- Attention is plain MHA, not GQA/MQA. Cache tensors store all heads: `[batch, heads, seq, head_dim]`.
- Learned position table has a source-level offset of 2, so table rows are `max_position_embeddings + 2` and runtime lookup uses `position_ids + 2`.
- BART positions are not RoPE/ALiBi; cached keys do not carry position-rotated content because there is no Q/K position transform.
- `scale_embedding` is usually false in representative configs, but the source supports `sqrt(d_model)` scaling.
- Current tokenizer implementation is a compatibility alias to RoBERTa byte-level BPE. Tokenizer special-token semantics come from RoBERTa: `<s>` id 0, `<pad>` id 1, `</s>` id 2, `<unk>` id 3, `<mask>` usually final vocab entry in pretrained vocabs.
- Summarization parity depends on generation config: forced EOS/BOS, beam count, length penalty, min/max length, and no-repeat n-gram are controller behavior, not core graph ops.
- Layout translation is mostly irrelevant for text tensors; protect `[batch, seq, hidden]` and `[batch, heads, seq, head_dim]` attention regions with no-layout-translation guards unless a whole fused attention lowering owns the layout.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs `[B, S_src]`, decoder ids `[B, S_tgt]`, optional masks `[B, S]`.
- Embedding gather for shared token table `[vocab, d_model]`.
- Position id arange/expand and embedding gather from `[max_pos + 2, d_model]`.
- Add, residual add, reshape/view, transpose between `[B, S, D]` and `[B, H, S, Hd]`, contiguous materialization where needed.
- Mask creation/conversion for bidirectional and causal masks; support prebuilt 4D masks passthrough.
- Cache update, cache reorder for beam search, batch repeat/select for generation.

Neural network primitives:

- LayerNorm over hidden dim, eps from PyTorch default because source constructs `nn.LayerNorm(embed_dim)` without explicit eps.
- Linear+bias: Q/K/V/O projections `D -> D`; FFN `D -> F`, `F -> D`; LM head `D -> vocab` without bias, followed by `final_logits_bias` add `[1, vocab]`.
- GELU activation for representative configs.
- Dropout is present in source but should be disabled for inference.
- Optional fp16 clamp in encoder layer if fp16 values become non-finite; for inference parity this is a rare guard but should be modeled or explicitly deferred.

Attention primitives:

- Encoder bidirectional MHA: Q/K/V shapes `[B, H_enc, S_src, Hd]`.
- Decoder causal self-attention with growing self KV cache: Q `[B, H_dec, S_new, Hd]`, K/V cache `[B, H_dec, S_past + S_new, Hd]`.
- Decoder cross-attention over encoder states with cross KV cache: K/V projected from encoder `[B, S_src, D]` to `[B, H_dec, S_src, Hd]` and reused after first update.
- Attention score order in eager path: `matmul(q, k^T) * head_dim^-0.5`, add mask, softmax over last dim, dropout, `matmul(probs, v)`, transpose back.

Generation/cache ops:

- `shift_tokens_right` for automatic decoder ids from inputs/labels.
- `EncoderDecoderCache(DynamicCache, DynamicCache)` for self and cross caches.
- Beam-search cache reorder and batch repeat/select.
- Generation logits slicing optimization is generic; `BartForConditionalGeneration` does not expose `logits_to_keep`, so last-token-only logits is a rewrite opportunity rather than current source API behavior.

Preprocessing-coupled ops:

- Byte-level BPE tokenization with RoBERTa post-processor special tokens.
- Padding mask from tokenizer attention mask.
- No token type IDs for BART tokenizer model input names.

## 5. Layer/block breakdown

Embedding path:

```text
input_ids [B, S] -> shared embedding [B, S, D] * embed_scale
position_ids = arange(past_len, past_len + S)
positions = learned_pos[position_ids + 2] [B, S, D]
x = LayerNorm(token_emb + positions)
```

Encoder block, repeated `encoder_layers` times:

```text
residual = x
q,k,v = Linear(D -> D, bias=True)(x), reshape to [B, Henc, Ssrc, Hd]
self_attn = MHA(q,k,v, bidirectional mask)
x = LayerNorm(residual + Linear(D -> D, bias=True)(self_attn))
residual = x
ff = Linear(D -> F, bias=True)(x)
ff = GELU(ff)
ff = Linear(F -> D, bias=True)(ff)
x = LayerNorm(residual + ff)
```

Decoder block, repeated `decoder_layers` times:

```text
residual = y
q = Linear(D -> D)(y) -> [B, Hdec, Stgt, Hd]
k,v = Linear(D -> D)(y) -> cache update -> [B, Hdec, Spast+Stgt, Hd]
self = causal MHA(q,k,v, decoder mask)
y = LayerNorm(residual + Linear(D -> D)(self))

residual = y
q = Linear(D -> D)(y)
k,v = Linear(D -> D)(encoder_hidden_states) or reusable cross cache
cross = bidirectional MHA(q,k,v, encoder padding mask)
y = LayerNorm(residual + Linear(D -> D)(cross))

residual = y
ff = Linear(D -> F)(y) -> GELU -> Linear(F -> D)
y = LayerNorm(residual + ff)
```

Conditional generation head:

```text
decoder_last_hidden [B, Stgt, D]
logits = MatMul(hidden, shared_embedding.T) + final_logits_bias[1, vocab]
```

For `bart-large-cnn`, `D=1024`, `H=16`, `Hd=64`, `F=4096`, encoder and decoder layers are both 12, and vocab is 50264.

## 6. Attention requirements

Required variants:

- Encoder self-attention: noncausal bidirectional MHA, heads equal to encoder head count, no KV cache.
- Decoder self-attention: causal MHA, heads equal to decoder head count, self KV cache grows with target length.
- Decoder cross-attention: noncausal MHA from decoder queries to encoder K/V, cross KV cache is static per source sequence after first projection.

Masking:

- Encoder uses `create_bidirectional_mask(config, inputs_embeds, attention_mask)`.
- Decoder self-attn creates a ones mask when absent, then `create_causal_mask` with self cache metadata.
- Decoder cross-attn uses `create_bidirectional_mask(..., encoder_hidden_states=encoder_hidden_states)` with source padding mask.
- Eager mask material is additive over attention scores. SDPA/Flash/flex paths are dispatched through `ALL_ATTENTION_FUNCTIONS` and `ALL_MASK_ATTENTION_FUNCTIONS`.

Cache tensor shapes:

- Dynamic cache layer shape is `[batch, num_heads, seq_len, head_dim]` for key and value.
- Self cache per decoder layer before decode step `t`: K/V `[B, Hdec, t, Hd]`; after appending `S_new`: `[B, Hdec, t + S_new, Hd]`.
- Cross cache per decoder layer after first use: K/V `[B, Hdec, S_src, Hd]`; subsequent decode steps reuse it without recomputing projections.
- `EncoderDecoderCache.get_seq_length()` reports self-attn cache length. Cross cache update state is tracked per layer with `is_updated`.

Backend compatibility:

- Source advertises `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn`.
- Fused attention parity must preserve query scaling, additive masking before softmax, softmax dim `-1`, and dropout placement. Inference can set dropout to zero.
- BART has no sliding window, packed varlen metadata, RoPE, ALiBi, or KV head repetition.

## 7. Position encoding and custom math

BART uses learned absolute positions for both encoder and decoder.

```python
def bart_position_lookup(weight, input_shape, past_key_values_length=0, position_ids=None):
    offset = 2
    if position_ids is None:
        bsz, seq_len = input_shape[:2]
        pos = arange(past_key_values_length, past_key_values_length + seq_len)
        pos = pos.expand(bsz, seq_len)
    else:
        pos = position_ids.unsqueeze(0)
    return embedding(weight, pos + offset)
```

Precomputable:

- Position embedding weights are constants.
- For fixed max target/source lengths, position ids and gathered position rows can be precomputed per `past_len`/length bucket, but decoder decode still depends on current cache length.

Dynamic inputs:

- Decoder `past_key_values_length` controls the absolute position of new tokens.
- Explicit `position_ids` can override arange behavior in the generic generation path.

Other custom math:

```python
def shift_tokens_right(ids, pad_id, decoder_start_id):
    shifted = zeros_like(ids)
    shifted[:, 1:] = ids[:, :-1]
    shifted[:, 0] = decoder_start_id
    shifted = where(shifted == -100, pad_id, shifted)
    return shifted
```

## 8. Preprocessing and input packing

Tokenizer:

- Current BART tokenizer aliases RoBERTa tokenizer backed by byte-level BPE.
- Tokenizer input names are `input_ids` and `attention_mask`.
- Default special token strings from source are `<s>`, `</s>`, `<unk>`, `<pad>`, and `<mask>`.
- Representative configs set `bos_token_id=0`, `pad_token_id=1`, `eos_token_id=2`, `decoder_start_token_id=2`.
- Tokenizer configs that exist for inspected repos set `model_max_length=1024`.

Special-token layout:

- RoBERTa processing inserts a leading class/BOS-style token and trailing separator/EOS-style token for single sequences. Pair sequences use RoBERTa separator conventions; summarization normally uses single source sequences.
- BART decoder starts from `decoder_start_token_id`, which is `eos_token_id` for the inspected checkpoints.

GPU/runtime inputs:

- `input_ids [B, S_src]`, `attention_mask [B, S_src]`, and optionally `decoder_input_ids [B, S_tgt]`.
- No segment/token type ids are required.
- No multimodal placeholder tokens, packed sequence descriptors, `cu_seqlens`, image/audio tensors, or scatter-update stitching.

Generation-controller behavior:

- `bart-large-cnn` generation config: `num_beams=4`, `max_length=142`, `min_length=56`, `length_penalty=2.0`, `no_repeat_ngram_size=3`, `forced_bos_token_id=0`, `forced_eos_token_id=2`, `early_stopping=true`.
- `bart-large-xsum`: `num_beams=6`, `max_length=62`, `min_length=11`, `no_repeat_ngram_size=3`, `forced_eos_token_id=2`.
- First Dinoml core graph parity can stub beam search and no-repeat controllers, but end-to-end summarization parity cannot.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared embedding and LM head tying

Source pattern: `shared(input_ids)` and `lm_head(hidden)` with `_tied_weights_keys` mapping LM head weight to shared embedding.

Replacement pattern: one constant `E [vocab, D]`; embedding gather uses `E`, logits use GEMM `hidden_flat [B*S, D] x E.T [D, vocab]`.

Preconditions:

- Weight tie is present or checkpoint tensors are byte-identical.
- LM head has no bias; preserve separate `final_logits_bias`.

Shape equations:

- `hidden [B, S, D] -> logits [B, S, vocab]`.

Failure cases:

- Resized or untied checkpoints; vocab mismatch between config and weight tensor.

Parity test sketch:

- Compare embedding outputs and logits on random token ids with tied and explicitly untied fixture weights.

### Rewrite: split Q/K/V projections to packed projection

Source pattern: three independent `Linear(D -> D, bias=True)` for self-attn.

Replacement pattern: packed GEMM `Linear(D -> 3D)` followed by slice/view into Q/K/V.

Preconditions:

- Same input tensor for Q/K/V, as in encoder self-attn and decoder self-attn.
- All three projections have compatible dtype/layout and bias.

Shape equations:

- `[B, S, D] x [D, 3D] -> [B, S, 3D] -> 3 x [B, H, S, Hd]`.

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], dim=0)
b_qkv = concat([b_q, b_k, b_v], dim=0)
```

Failure cases:

- Cross-attn cannot fully pack Q with K/V because Q input is decoder hidden and K/V input is encoder hidden. K/V can still be packed as `Linear(D -> 2D)` for encoder states.

Parity test sketch:

- Per-layer random input compare projected Q/K/V before attention.

### Rewrite: cross-attn K/V precompute

Source pattern: each decoder layer computes cross-attn K/V from `encoder_hidden_states` on first generated token, then reuses cross cache.

Replacement pattern: after encoder, precompute all decoder-layer cross K/V once into a cross-cache artifact.

Preconditions:

- Encoder hidden states are fixed for the generation request.
- Cross-attn weights are static and loaded.
- Beam expansion either repeats cache or indexes it consistently with beam state.

Shape equations:

- For each decoder layer: `[B, S_src, D] -> K,V [B, Hdec, S_src, Hd]`.

Failure cases:

- Dynamic encoder-output overrides, cross-cache invalidation, or generation modes that mutate source batch ordering without cache reorder.

Parity test sketch:

- Run prefill/decode with source projection inside block versus precomputed cross cache and compare layer outputs.

### Rewrite: last-token-only logits for generation

Source pattern: conditional generation computes logits for all decoder positions.

Replacement pattern: during decode, slice hidden to last token before LM head.

Preconditions:

- Inference generation step only needs next-token logits.
- Loss computation and full-sequence score outputs are not requested.

Shape equations:

- `[B, S_tgt, D] -> [B, 1, D] -> [B, 1, vocab]`.

Failure cases:

- Teacher-forced full logits, sequence scoring requiring all positions, assisted decoding needing multiple candidate logits.

Parity test sketch:

- Compare `full_logits[:, -1, :]` with logits from sliced hidden.

### Rewrite: remove inference dropout and layerdrop

Source pattern: dropout after embedding, attention, activation, and FFN; LayerDrop branches in training.

Replacement pattern: erase dropout and LayerDrop in eval/inference graph.

Preconditions:

- `model.eval()` semantics / inference-only compilation.

Failure cases:

- Training, stochastic evaluation, or tests that expect dropout RNG.

## 10. Kernel fusion candidates

Highest priority:

- MHA/FlashAttention for encoder prefill and decoder self/cross attention. BART-large has many `S=1024`, `D=1024`, `H=16` attention blocks.
- Decoder KV cache update plus causal attention for decode. This is required for usable generation latency.
- Cross-attn K/V precompute/cache. It avoids repeated encoder K/V projections during beam decode.
- LM head last-token GEMM with tied embedding. Full `[B, S, vocab]` logits are wasteful during decode.

Medium priority:

- Packed QKV projection for self-attn and packed KV projection for cross-attn.
- LayerNorm + residual fusion around post-norm attention/MLP boundaries.
- FFN fusion: `Linear -> GELU -> Linear`, with GEMM epilogue GELU/bias candidates where supported.
- Embedding + learned position add + LayerNorm fusion for encoder and decoder inputs.

Lower priority:

- Beam-search cache reorder optimized copy kernels.
- Additive mask materialization avoidance when using SDPA/Flash-style causal flags and padding masks.
- fp16 non-finite clamp guard in encoder layers; important only for rare numerical parity cases.

## 11. Runtime staging plan

Stage 1: parse BART config and tokenizer metadata, load shared embeddings, learned positions, encoder/decoder block weights, LM head tie, and `final_logits_bias`.

Stage 2: implement encoder-only parity for `BartEncoder`: embeddings, bidirectional mask, 1 block, then full encoder.

Stage 3: implement teacher-forced seq2seq forward without cache: decoder causal self-attn, cross-attn, full logits.

Stage 4: implement dynamic `EncoderDecoderCache`: self cache append, cross cache reuse, beam reorder hooks.

Stage 5: implement generation prefill/decode parity for greedy generation; add forced BOS/EOS and min/max length controls.

Stage 6: add beam search and no-repeat n-gram controller parity for `bart-large-cnn`/`xsum`.

Stage 7: enable optimized attention, packed projections, cross-cache precompute, last-token logits, and fused FFN/LayerNorm paths.

Initially stub:

- Training losses, dropout, LayerDrop, gradient checkpointing, sequence classification, QA, decoder-only causal LM, assisted/speculative decoding, and static/offloaded/quantized cache variants.

## 12. Parity and validation plan

- Custom op tests: learned position lookup with offset 2; `shift_tokens_right`; additive causal and bidirectional mask construction; cache append and cross-cache reuse.
- Single-layer parity: encoder layer and decoder layer with random tensors, fp32 first, compare hidden states at `atol=1e-5`/`rtol=1e-4`.
- Full encoder parity: `facebook/bart-base` random small sequence and real tokenized text.
- Teacher-forced seq2seq parity: `BartForConditionalGeneration` logits for short `[B, S_src, S_tgt]`.
- Cache parity: prefill then token-by-token decode must match no-cache full decode for the same target ids.
- Cross-cache parity: first decode step fills cross cache; later steps must not recompute K/V but must match output.
- Generation parity: greedy token sequence, then beam output for `bart-large-cnn` and `bart-large-xsum` using exact generation configs.
- Tolerances: fp32 `1e-5` absolute / `1e-4` relative; fp16/bf16 use looser logits tolerance such as `2e-2` absolute and validate top-k/token parity separately.

## 13. Performance probes

- Tokenizer throughput: documents/sec for byte-level BPE and padding to 1024.
- Encoder-only throughput over `B x S_src`: `S_src` sweep 128, 512, 1024.
- Decoder teacher-forced prefill throughput: `S_tgt` sweep 1, 16, 64, 142.
- Decode tokens/sec with KV cache: batch and beam sweeps; separate self-attn, cross-attn, FFN, and LM head time.
- Cross-cache memory: `decoder_layers * 2 * B * beams * H * S_src * Hd * dtype_size`.
- Self-cache memory growth: `decoder_layers * 2 * B * beams * H * S_tgt * Hd * dtype_size`.
- LM head bandwidth/GEMM cost: full logits versus last-token-only logits.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible fused path, and Dinoml native attention.

## 14. Skip/defer list

- Training, losses, dropout, LayerDrop, gradient checkpointing.
- Sequence classification and question answering heads.
- Decoder-only `BartForCausalLM`.
- Quantized/offloaded/static cache implementations beyond the first dynamic cache.
- Assisted/speculative decoding and contrastive search.
- Full HF generation feature matrix beyond summarization-critical beam search, min/max length, forced BOS/EOS, and no-repeat n-gram.
- Tokenizer implementation inside Dinoml runtime; keep it in CPU/data pipeline initially.

## 15. Final implementation checklist

- [ ] Parse `BartConfig` including asymmetric encoder/decoder depths and vocab size.
- [ ] Load shared token embedding, position embeddings with offset 2, all projection/FFN/LayerNorm weights, and `final_logits_bias`.
- [ ] Implement byte-level BPE tokenizer metadata handoff or accept pre-tokenized `input_ids`/`attention_mask`.
- [ ] Implement learned absolute position lookup for encoder and decoder cache offsets.
- [ ] Implement encoder bidirectional MHA block with post-norm residuals.
- [ ] Implement decoder causal self-attn and cross-attn block with post-norm residuals.
- [ ] Implement `EncoderDecoderCache` self-cache append and cross-cache reuse.
- [ ] Implement mask lowering for encoder padding, decoder causal+padding, and cross-attn padding.
- [ ] Implement LM head tied to shared embedding plus final logits bias.
- [ ] Add packed QKV/KV projection rewrites with guarded weight transforms.
- [ ] Add cross-attn K/V precompute rewrite.
- [ ] Add last-token-only logits rewrite for decode.
- [ ] Add one-layer, full-forward, cache, and generation parity tests.
- [ ] Benchmark encoder, prefill, decode, LM head, and cache memory separately.
