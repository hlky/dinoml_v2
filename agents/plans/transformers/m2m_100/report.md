# m2m_100 Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: primary `facebook/m2m100_418M`; representative configs also inspected for `facebook/m2m100_1.2B`, `facebook/wmt21-dense-24-wide-en-x`, `facebook/wmt21-dense-24-wide-x-en`, `facebook/m2m100-12B-last-ckpt`, and `hf-internal-testing/tiny-random-M2M100ForConditionalGeneration`.

Config source: raw Hugging Face `config.json` and `tokenizer_config.json` fetched into `_sources/`. All listed HF repos were public and accessible; no gated/401 gaps were encountered.

Source files inspected:

- `src/transformers/models/m2m_100/configuration_m2m_100.py`
- `src/transformers/models/m2m_100/modeling_m2m_100.py`
- `src/transformers/models/m2m_100/tokenization_m2m_100.py`
- `src/transformers/models/m2m_100/convert_m2m100_original_checkpoint_to_pytorch.py`
- `tests/models/m2m_100/test_modeling_m2m_100.py`
- `tests/models/m2m_100/test_tokenization_m2m_100.py`

Snapshots: `_sources/configuration_m2m_100.py`, `_sources/modeling_m2m_100.py`, `_sources/tokenization_m2m_100.py`, and fetched config/tokenizer JSON files.

Any missing files or assumptions: no fast tokenizer exists in this family directory. SentencePiece vocabulary/model files were not downloaded because the audit target is model/tokenizer coupling rather than text parity.

## 2. High-level architecture

M2M100 is a text-only encoder-decoder translation model with a seq2seq LM head. It is close to MBART/BART in block structure: shared token embedding, sinusoidal positional embeddings, pre-norm encoder self-attention, pre-norm decoder causal self-attention, decoder cross-attention over encoder states, ReLU FFNs, final layer norms, and a tied output projection.

Dataflow:

```text
SentencePiece + language-code packing -> encoder embeddings/positions -> encoder stack
  -> decoder language start/prefix + causal decode -> decoder self-attn + cross-attn
  -> tied LM head -> logits -> generation controller with forced BOS target language
```

First useful DinoML target: `M2M100ForConditionalGeneration` inference for translation, including encoder precompute, decoder prefill, and incremental decode with self-attention KV cache plus reusable cross-attention KV cache. The bare `M2M100Model` is optional feature-extraction/debug surface. Training losses, LayerDrop, dropout, and label shifting are not required for first inference parity.

## 3. Important config dimensions

Source defaults from `M2M100Config`:

| Field | Default |
|---|---:|
| `vocab_size` | 128112 |
| `d_model` | 1024 |
| `encoder_layers` / `decoder_layers` | 12 / 12 |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 |
| inferred `head_dim` | 64 |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 |
| `max_position_embeddings` | 1024 |
| `activation_function` | `relu` |
| `scale_embedding` | `true` |
| `use_cache` | `true` |
| `decoder_start_token_id` / `eos_token_id` / `pad_token_id` | 2 / 2 / 1 |
| `tie_word_embeddings` | `true` |

Representative checkpoint sweep:

| Model | Layers enc/dec | `d_model` | Heads enc/dec | `head_dim` | FFN enc/dec | Vocab | Tokenizer language set |
|---|---:|---:|---:|---:|---:|---:|---|
| `hf-internal-testing/tiny-random-M2M100ForConditionalGeneration` | 2 / 2 | 16 | 4 / 4 | 4 | 4 / 4 | 128112 | config fetched, test-only |
| `facebook/m2m100_418M` | 12 / 12 | 1024 | 16 / 16 | 64 | 4096 / 4096 | 128112 | `m2m100` 100 languages |
| `facebook/m2m100_1.2B` | 24 / 24 | 1024 | 16 / 16 | 64 | 8192 / 8192 | 128112 | `m2m100` 100 languages |
| `facebook/wmt21-dense-24-wide-en-x` | 24 / 24 | 2048 | 32 / 32 | 64 | 16384 / 16384 | 128009 | `wmt21` 8 languages |
| `facebook/wmt21-dense-24-wide-x-en` | 24 / 24 | 2048 | 32 / 32 | 64 | 16384 / 16384 | 128009 | `wmt21` 8 languages, config has `forced_bos_token_id=128001` |
| `facebook/m2m100-12B-last-ckpt` | 24 / 24 | 4096 | 16 / 16 | 256 | 16384 / 16384 | 128112 | `m2m100` repo metadata |

Config-derived inference: there is no GQA/MQA; all inspected configs use full MHA where `num_key_value_heads == num_attention_heads` conceptually because the source has no separate KV-head field.

## 3a. Family variation traps

- `head_dim` is not fixed at 64. The 12B checkpoint has `d_model=4096`, 16 heads, so `head_dim=256`.
- WMT21 variants use `vocab_size=128009` and tokenizer `language_codes="wmt21"` with 8 language tags, not the 100-language M2M100 table.
- The tokenizer code prefixes both source and target sequences with the language token and suffixes with EOS. Some docstrings still say `X [eos, lang]`; current source behavior is `[lang] X [eos]`.
- Target language selection for generation is controller-level: translation pipeline sets `forced_bos_token_id` from `tokenizer.get_lang_id(tgt_lang)`. It is not produced by the neural graph.
- `decoder_start_token_id == eos_token_id == 2`; generated sequences in tests begin with `</s>` then forced target language token.
- Embedding weights are shared across encoder, decoder, and LM head. Treat these as one logical parameter alias.
- Sinusoidal positions are registered as non-persistent buffers and can grow beyond the initial max if runtime sequence length plus cache length exceeds the current buffer.
- The source supports the generic Transformers attention interface, including `flash_attention_2` in tests. DinoML can start with faithful eager attention but should preserve mask/cache ABI to swap in optimized attention later.
- Configs contain generation defaults such as `num_beams`, `max_length`, `early_stopping`, and sometimes `forced_bos_token_id`; these are generation-controller settings, not graph operators.
- Layout is sequence-major logical `(batch, seq, hidden)` with attention tensors reshaped to `(batch, heads, seq, head_dim)`. No image/channel layout translation applies.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup with shared table `[vocab_size, d_model]`, scaled by `sqrt(d_model)` when `scale_embedding=true`.
- Sinusoidal positional embedding lookup from position ids generated by `attention_mask`/pad-aware cumsum.
- Add, residual add, reshape/view, transpose, contiguous materialization, index/select/gather for positions.
- Attention mask construction: bidirectional encoder/cross mask and causal decoder mask, with large negative additive mask values.
- Final logits projection `Linear(d_model -> vocab_size, bias=false)` tied to shared embedding.

Neural network primitives:

- LayerNorm over hidden dimension with default PyTorch epsilon from `nn.LayerNorm`.
- Linear projections with bias for Q/K/V/O and FFN `fc1`/`fc2`.
- ReLU activation.
- Dropout and LayerDrop are training-only for first inference.
- fp16 clamp in encoder layer after residual FFN path; required only if matching source fp16 eager behavior exactly.

Attention primitives:

- Encoder noncausal self-attention.
- Decoder causal self-attention with autoregressive KV cache.
- Decoder cross-attention with encoder keys/values and reusable cross-attention cache.
- Full MHA only; no RoPE, ALiBi, relative bias, sliding window, block sparse, MQA, or GQA.

Position/tokenizer-coupled ops:

- Pad-aware position ids: `cumsum(input_ids != pad) + pad_idx`.
- Inputs-embeds path uses simple sequential positions because pad locations are unknown.
- Language-code tokens live after the SentencePiece encoder vocabulary and must match tokenizer config (`m2m100` versus `wmt21`).
- Generation must inject target language with `forced_bos_token_id`.

Generation/cache ops:

- Encoder output cache reusable across all decoder steps.
- Per-decoder-layer self-attention K/V cache grows with generated target length.
- Per-decoder-layer cross-attention K/V cache is computed once from encoder hidden states and reused.
- Beam reorder must reorder decoder self-attention cache and keep encoder/cross-attention state aligned with beams via Transformers cache semantics.

## 5. Layer/block breakdown

Encoder input:

```text
input_ids: [B, S]
tok = Embedding(input_ids) * sqrt(d_model)
pos = sinusoidal_position(input_ids, pad_token_id=1): [B, S, D]
x = dropout(tok + pos)
mask = bidirectional_padding_mask(attention_mask): [B, 1, 1 or S, S]
```

Encoder block, repeated `encoder_layers` times:

```text
res = x
x = LayerNorm(x)
q,k,v = Linear(D -> D, bias=True), split to [B, H, S, Dh]
x = attention(q,k,v, additive padding mask)
x = res + Linear(D -> D, bias=True)(x)
res = x
x = LayerNorm(x)
x = Linear(D -> encoder_ffn_dim, bias=True)(x)
x = ReLU(x)
x = Linear(encoder_ffn_dim -> D, bias=True)(x)
x = res + x
optional source fp16 clamp
```

Encoder output: `LayerNorm(x)` gives `[B, S, D]`.

Decoder input:

```text
decoder_input_ids: [B, T] during prefill, [B, 1] during decode
tok = Embedding(decoder_input_ids) * sqrt(d_model)
pos = sinusoidal_position(decoder_input_ids or inputs_embeds, past_key_values_length)
x = tok + pos
self_mask = causal mask over past + current target length
cross_mask = encoder padding mask broadcast to decoder query length
```

Decoder block, repeated `decoder_layers` times:

```text
res = x
x = LayerNorm(x)
q = Linear(D -> D)(x)
self_k,self_v = Linear(D -> D)(x), appended to self cache
x = causal_self_attention(q,self_k,self_v,self_mask)
x = res + out_proj(x)
res = x
x = LayerNorm(x)
q = Linear(D -> D)(x)
cross_k,cross_v = Linear(D -> D)(encoder_hidden), cached after first use
x = cross_attention(q,cross_k,cross_v,cross_mask)
x = res + out_proj(x)
res = x
x = LayerNorm(x)
x = Linear(D -> decoder_ffn_dim)(x)
x = ReLU(x)
x = Linear(decoder_ffn_dim -> D)(x)
x = res + x
```

Decoder output: `LayerNorm(x)` then `lm_head(x)` with tied weight, producing `[B, T, vocab_size]`.

## 6. Attention requirements

Encoder attention is bidirectional self-attention. Decoder self-attention is causal autoregressive self-attention. Decoder cross-attention is rectangular attention with decoder queries length `Tq` and encoder keys length `S`.

Head counts and widths:

- 418M/1.2B: `H=16`, `Dh=64`, `D=1024`.
- WMT21 dense wide: `H=32`, `Dh=64`, `D=2048`.
- 12B checkpoints: `H=16`, `Dh=256`, `D=4096`.

Masking:

- Encoder uses bidirectional additive padding mask derived from `attention_mask`.
- Decoder self-attention uses additive causal mask over `past_length + current_length`; if no decoder attention mask is supplied, source creates all-ones mask for current plus past.
- Cross-attention uses encoder padding mask against encoder hidden states.

Cache ABI:

- Source initializes `EncoderDecoderCache(DynamicCache, DynamicCache)` when `use_cache` is true and no cache is supplied.
- Self-attention cache stores each layer's keys/values as `[B, H, T_cache, Dh]` after projection and head transpose.
- Cross-attention cache stores each layer's encoder-derived keys/values as `[B, H, S, Dh]`. `is_updated[layer_idx]` controls reuse after first compute.
- Cached keys are sinusoid-position-independent because M2M100 adds absolute positions before projection; there is no RoPE to apply to K after cache update.

FlashAttention/SDPA compatibility: source dispatches through `ALL_ATTENTION_FUNCTIONS` using `_attn_implementation`, with eager fallback `q @ k.T * scale`, additive mask, softmax over last dim, dropout, `attn @ v`. A fused backend must preserve additive mask semantics and rectangular cross-attention.

## 7. Position encoding and custom math

M2M100 uses learned token embeddings plus non-learned sinusoidal absolute position embeddings. The positional buffer is not a trained weight.

Short reproduction:

```python
def m2m100_position_ids(input_ids, padding_idx=1, past_len=0):
    mask = (input_ids != padding_idx).to(torch.int32)
    incremental = (torch.cumsum(mask, dim=1).type_as(mask) + past_len) * mask
    return incremental.to(torch.long) + padding_idx

def m2m100_sinusoid(num_embeddings, dim, padding_idx=1):
    half = dim // 2
    inv = torch.exp(torch.arange(half).float() * -(math.log(10000) / (half - 1)))
    phase = torch.arange(num_embeddings).float().unsqueeze(1) * inv.unsqueeze(0)
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1).view(num_embeddings, -1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    emb[padding_idx, :] = 0
    return emb
```

Precompute: sinusoidal table can be generated at compile/load time up to admitted max source/target/cache length. Runtime still needs pad-aware cumsum for `input_ids`; decode with `inputs_embeds` can use sequential arange plus `past_len`.

## 8. Preprocessing and input packing

Tokenizer is model-coupled enough to affect runtime IDs:

- Slow SentencePiece tokenizer only.
- `m2m100` language set has 100 language codes; `wmt21` set has 8.
- Language-token ids are assigned starting at `encoder_size` in tokenizer source. Config `vocab_size` must include base vocab, language tags, and any made-up/reserved words.
- Current source `set_src_lang_special_tokens` and `set_tgt_lang_special_tokens` set `prefix_tokens=[lang_id]` and `suffix_tokens=[eos_id]`.
- `_build_translation_inputs` requires `src_lang` and `tgt_lang`, tokenizes source text, then adds `forced_bos_token_id=tgt_lang_id`.

GPU graph inputs for first integration:

- `input_ids [B, S]`, already packed as `[src_lang] tokens [eos]`.
- `attention_mask [B, S]`, 1 for real tokens.
- During generation, initial decoder starts from EOS/decoder start and generation controller forces first generated token to target language id.
- `decoder_input_ids [B, T]` for teacher-forced or prefill paths; `[B, 1]` for incremental decode.

CPU/data-pipeline boundary: SentencePiece tokenization, language-code validation, source/target prefix/suffix insertion, forced BOS selection, beam search, and text decoding can remain outside DinoML's first graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared embedding and LM head alias

Source pattern: `model.shared`, `encoder.embed_tokens`, `decoder.embed_tokens`, and `lm_head.weight` are tied.

Replacement: one constant table used for embedding lookup and transposed/logit GEMM.

Preconditions: `tie_word_embeddings=true` and source weight aliases or equal storage are confirmed.

Weight transform: LM head uses `[vocab, D]` weight as `hidden @ weight.T`.

Failure cases: untied third-party checkpoints should be rejected or routed to a separate path; Transformers tests note this architecture is tied by default.

Parity test sketch: compare embedding rows, encoder/decoder token embeddings, and logits for a short decoder hidden slice.

### Rewrite: QKV projection packing per attention module

Source pattern: separate `q_proj`, `k_proj`, `v_proj`, each `Linear(D -> D, bias=True)`.

Replacement: packed GEMM `Linear(D -> 3D)` split in Q,K,V order for self-attention; for cross-attention, keep `q` separate from cached encoder `k/v`, or pack only K/V over encoder states.

Preconditions: same input tensor for self-attention Q/K/V; no head pruning; full MHA; bias present for all three.

Shape equations: input `[B, L, D]`, packed output `[B, L, 3D]`, split to three `[B, H, L, Dh]`.

Failure cases: cross-attention Q comes from decoder and K/V from encoder, so full QKV packing is invalid there.

Parity test sketch: run one layer with copied packed weights and compare pre-softmax Q/K/V tensors.

### Rewrite: absolute sinusoidal position as generated constant plus dynamic gather

Source pattern: dynamic table generation and `index_select`.

Replacement: compile/load a table up to admitted max length, then gather by runtime position ids.

Preconditions: admitted max `padding_idx + 1 + seq_len + past_len` is bounded; dtype/device match source hidden dtype.

Failure cases: unbounded decode length beyond table max; either grow/recompile or reject with clear max-position error.

Parity test sketch: compare table rows and position-gathered embeddings for padded and non-padded batches plus decode `past_len`.

### Rewrite: last-token-only logits during decode

Source pattern: LM head computes logits for every decoder time step.

Replacement: for incremental decode where current `T=1`, compute only `[B, 1, vocab]`; for prefill with sampling from last position, optionally slice last hidden before LM head.

Preconditions: no caller requests full logits for all prefill positions; generation controller only consumes final position.

Failure cases: teacher-forced scoring, loss, or requested full logits.

Parity test sketch: compare final-token logits from full head versus sliced hidden head.

## 10. Kernel fusion candidates

Highest priority:

- Encoder/decoder LayerNorm + GEMM regions, because every block has three LayerNorms in decoder and two in encoder.
- QKV packed projection and attention prefill/decode kernels for full MHA.
- Cross-attention K/V precompute and cache storage, because encoder K/V are reused every decode step.
- LM head GEMM over large vocab; for generation, prioritize last-token-only logits and optional top-k/top-p integration later.

Medium priority:

- ReLU FFN fusion: `Linear -> ReLU -> Linear` with residual add.
- Mask construction kernels for causal and bidirectional masks with pad-aware shapes.
- Sinusoidal position id cumsum + gather fusion for encoder and decoder prefill.
- Beam-cache reorder for `EncoderDecoderCache`.

Lower priority:

- Dropout/LayerDrop training paths.
- fp16 clamp parity in encoder; keep as a small op unless it affects deployed precision.
- FlashAttention-specific parity after eager attention and cache ABI are stable.

## 11. Runtime staging plan

Stage 1: parse configs/tokenizer metadata, load tied weights, and run embedding/position/LM-head parity on tiny and 418M configs.

Stage 2: implement one encoder layer and one decoder layer without cache using eager dense attention.

Stage 3: full encoder parity with padding masks and sinusoidal positions.

Stage 4: full seq2seq prefill parity for `M2M100ForConditionalGeneration`, including cross-attention.

Stage 5: incremental decode with `EncoderDecoderCache`: self KV append and cross KV reuse.

Stage 6: generation-controller integration for forced BOS target language and beam reorder. Beam search itself can initially stay in Python.

Stage 7: optimized lowering: packed projections, fused attention, cached cross-attention K/V, last-token logits, and benchmark-driven GEMM/layout choices.

## 12. Parity and validation plan

- Config parser tests for 418M, 1.2B, WMT21, 12B, and tiny configs.
- Tokenizer metadata tests for language-code id ranges and prefix/suffix behavior, without requiring SentencePiece text parity inside DinoML.
- Position tests: padded `input_ids`, no-pad `input_ids`, `inputs_embeds` path, and decode `past_len`.
- Single attention tests: encoder self-attn, decoder causal self-attn, decoder cross-attn, with additive masks.
- Cache tests: prefill one step then decode one token; compare against no-cache full decoder output for the last token.
- Full encoder and decoder block parity after 1, 2, and all layers on tiny random config.
- 418M smoke parity against source test slices for no-head hidden state and LM logits if weights are available.
- Generation parity: forced English/French language token at first generated position, then compare decode token sequence under greedy and small beam settings.
- Suggested tolerances: fp32 `rtol=1e-4, atol=1e-4` for block/full logits initially; fp16/bf16 looser around attention/softmax, starting `rtol=1e-2, atol=1e-2` before fused kernels are tuned.

## 13. Performance probes

- Encoder-only throughput by batch and source sequence length.
- Decoder prefill throughput by target length with encoder length held fixed.
- Incremental decode tokens/sec with encoder length sweep.
- Cross-attention cache memory and setup time per batch/beam.
- Self KV cache memory growth: `2 * decoder_layers * B * beams * H * T * Dh * dtype_size`.
- LM head latency for full prefill logits versus last-token-only logits.
- Attention backend comparison: eager GEMM-softmax-GEMM versus fused attention for self and cross attention.
- Large-vocab GEMM and sampling/top-k split, especially for WMT21 `128009` versus M2M100 `128112`.
- Beam-size sweep with cache reorder overhead.
- Config sweep: 418M, 1.2B, WMT21 wide, and 12B head-dim-256 shape.

## 14. Skip/defer list

- Training loss, label shifting, dropout, LayerDrop, gradient checkpointing.
- Text SentencePiece implementation inside DinoML runtime; keep as CPU preprocessing.
- Beam search and sampling kernels beyond required forced BOS/control integration.
- Untied embedding/head checkpoints unless a real checkpoint requires them.
- FlashAttention parity until eager attention/cache parity is stable.
- Quantization, tensor parallelism, and distributed inference.
- Arbitrary sequence lengths beyond an admitted positional/cache maximum.
- `output_attentions`, `output_hidden_states`, and intermediate recorder surfaces for first production path.

## 15. Final implementation checklist

- [ ] Parse `M2M100Config` and tokenizer metadata, including `language_codes`.
- [ ] Load shared/tied embedding and LM head without cloning logical weights.
- [ ] Implement scaled token embedding and tied LM projection.
- [ ] Implement M2M100 sinusoidal table, pad-aware position ids, and decode `past_len` offset.
- [ ] Implement encoder bidirectional padding mask and decoder causal mask.
- [ ] Implement encoder self-attention MHA.
- [ ] Implement decoder causal self-attention with KV append.
- [ ] Implement decoder cross-attention with reusable encoder K/V cache.
- [ ] Implement ReLU FFN and LayerNorm/residual block structure.
- [ ] Add generation-controller hook for `forced_bos_token_id` target language.
- [ ] Add cache reorder parity for beam search.
- [ ] Add QKV/KV packing rewrites with guarded split-order tests.
- [ ] Add last-token-only LM head rewrite for decode.
- [ ] Add tiny, 418M, WMT21, and 12B shape/config parity tests.
- [ ] Benchmark encoder, prefill, decode, cache memory, and LM head separately.
