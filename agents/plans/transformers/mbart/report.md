# mBART Transformers Audit

## 1. Source basis

Transformers commit/version: local checkout `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary `facebook/mbart-large-cc25`; representative sweep also used `facebook/mbart-large-en-ro`, `facebook/mbart-large-50`, `facebook/mbart-large-50-many-to-many-mmt`, and `sshleifer/tiny-mbart`.

Config source:

- `https://huggingface.co/facebook/mbart-large-cc25/raw/main/config.json`
- `https://huggingface.co/facebook/mbart-large-en-ro/raw/main/config.json`
- `https://huggingface.co/facebook/mbart-large-50/raw/main/config.json`
- `https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/raw/main/config.json`
- `https://huggingface.co/sshleifer/tiny-mbart/raw/main/config.json`
- `https://huggingface.co/facebook/mbart-large-50/raw/main/tokenizer_config.json`

Source files inspected:

- `X:/H/transformers/src/transformers/models/mbart/configuration_mbart.py`
- `X:/H/transformers/src/transformers/models/mbart/modeling_mbart.py`
- `X:/H/transformers/src/transformers/models/mbart/tokenization_mbart.py`
- `X:/H/transformers/src/transformers/models/mbart50/tokenization_mbart50.py`
- `X:/H/transformers/src/transformers/generation/utils.py`
- `X:/H/transformers/src/transformers/generation/logits_process.py`

Local snapshots:

- `_sources/configuration_mbart.py`
- `_sources/modeling_mbart.py`
- `_sources/tokenization_mbart.py`
- `_sources/tokenization_mbart50.py`
- `_sources/facebook_mbart-large-cc25_config.json`
- `_sources/facebook_mbart-large-en-ro_config.json`
- `_sources/facebook_mbart-large-50_config.json`
- `_sources/facebook_mbart-large-50-many-to-many-mmt_config.json`
- `_sources/sshleifer_tiny-mbart_config.json`
- `_sources/facebook_mbart-large-50_tokenizer_config.json`

Any missing files or assumptions:

- `https://huggingface.co/facebook/mbart-large-cc25/raw/main/tokenizer_config.json` returned 404 during this audit. The model config, current `MBartTokenizer` source, and mBART-50 tokenizer config/source were accessible.
- The primary runtime target is `MBartForConditionalGeneration` for multilingual seq2seq generation/translation. Sequence classification, QA, and decoder-only `MBartForCausalLM` are optional or deferred heads.
- Some checkpoint configs contain historical fields such as `normalize_before`, `normalize_embedding`, `static_position_embeddings`, `add_final_layer_norm`, `add_bias_logits`, `output_past`, `force_bos_token_to_be_generated`, and `extra_pos_embeddings`. The inspected current `MBartConfig` does not declare most of these as structure-changing fields; current modeling source is pre-norm and always includes final encoder/decoder layer norms.

## 2. High-level architecture

mBART is a text-only encoder-decoder Transformer. The encoder is bidirectional. The decoder uses causal self-attention plus encoder-decoder cross-attention. Both sides use shared token embeddings, learned absolute position embeddings with a source-level offset of 2, pre-norm attention/FFN blocks, final LayerNorm after all layers, and a tied LM head plus `final_logits_bias`.

Dataflow:

```text
SentencePiece/Unigram tokenizer + language-code special tokens
  -> input_ids/attention_mask
  -> shared token embedding * optional sqrt(d_model) + learned positions
  -> encoder bidirectional blocks
  -> decoder language/start tokens + learned positions
  -> decoder causal self-attn + cross-attn over encoder states
  -> lm_head tied to shared embedding + final_logits_bias
  -> generation controller with forced BOS/EOS language behavior
```

Stage decomposition:

- CPU/data pipeline: tokenizer normalization/pretokenization, language-code insertion, padding, attention masks, and translation pipeline `forced_bos_token_id`.
- Encoder: independently cacheable per source sentence; output shape `[batch, src_len, d_model]`.
- Decoder prefill: target/language prompt tokens with causal self-attention and first cross-attention projection from encoder states.
- Decode: one or more new tokens with growing decoder self KV cache and static cross-attention cache.
- Logits/sampling: dense vocab logits, forced target-language BOS token, forced EOS, beam/search policy.

## 3. Important config dimensions

Source-default `MBartConfig` fields:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 50265 | Shared embedding rows and LM head output columns |
| `max_position_embeddings` | 1024 | Learned position table has `max_position_embeddings + 2` rows |
| `d_model` | 1024 | Hidden width |
| `encoder_layers` / `decoder_layers` | 12 / 12 | Repeated block counts |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | Plain MHA head counts |
| `head_dim` | inferred 64 | `d_model / heads`; source requires divisibility |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 | FFN expansion |
| `activation_function` | `gelu` | FFN activation via `ACT2FN`; checkpoints may use `relu` |
| `dropout` / `attention_dropout` / `activation_dropout` | 0.1 / 0.0 / 0.0 | Disabled in eval inference |
| `scale_embedding` | false | If true, token embeddings multiply by `sqrt(d_model)` |
| `use_cache` | true | Enables encoder-decoder KV cache in generation |
| `pad/bos/eos` | 1 / 0 / 2 | Token, mask, and generation defaults |
| `decoder_start_token_id` | null | mBART uses language ID / shifted last non-pad token rather than one universal decoder start in base source |
| `forced_eos_token_id` | 2 | Generation controller EOS constraint |
| `tie_word_embeddings` | true | LM head should alias shared token embedding |

Representative checkpoint sweep:

| Model id | Arch in config | `d_model` | Enc/Dec layers | Heads | FFN | Vocab | Max pos | Activation | Generation/tokenizer notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `facebook/mbart-large-cc25` | `MBartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 250027 | 1024 | gelu | `scale_embedding=true`; task param `translation_en_to_ro.decoder_start_token_id=250020`; no current tokenizer config fetched |
| `facebook/mbart-large-en-ro` | implicit conditional generation | 1024 | 12 / 12 | 16 / 16 | 4096 | 250027 | 1024 | gelu | `decoder_start_token_id=250020`; `attention_dropout=0.1`; old `extra_pos_embeddings=2` |
| `facebook/mbart-large-50` | `MBartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 250054 | 1024 | gelu | `tokenizer_class=MBart50Tokenizer`; `decoder_start_token_id=2`; 50 language codes |
| `facebook/mbart-large-50-many-to-many-mmt` | `MBartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 250054 | 1024 | relu | `tokenizer_class=MBart50Tokenizer`; `decoder_start_token_id=2`; activation differs |
| `sshleifer/tiny-mbart` | config says `BartForConditionalGeneration` | 2 | 2 / 2 | 1 / 1 | 4 | 250027 | 1024 | gelu | Debug checkpoint only; architecture name is historical and should not drive source selection |

Omitted checkpoint fields use current `MBartConfig` defaults, including `classifier_dropout=0.0`, `use_cache=True`, `is_encoder_decoder=True`, `is_decoder=False`, and `tie_word_embeddings=True` unless explicitly overridden.

## 3a. Family variation traps

- Vocab size is 250027 for cc25/en-ro and 250054 for mBART-50. Do not hard-code vocabulary or language-token IDs.
- mBART-50 uses `MBart50Tokenizer`, not `MBartTokenizer`, even though the neural `model_type` remains `mbart`.
- Special-token layout differs by tokenizer. `MBartTokenizer` appends `[eos, lang_code]` for source and target modes. `MBart50Tokenizer` prefixes `[lang_code]` and suffixes `[eos]`.
- Translation pipelines set `forced_bos_token_id` to the target language ID. This is generation-controller behavior, not a model forward op.
- `decoder_start_token_id` is inconsistent across checkpoints: some use a target language ID, mBART-50 uses EOS id 2, and source defaults to `None`. First integration should accept explicit `decoder_input_ids` or generation metadata rather than assuming one universal start token.
- mBART uses learned absolute positions with offset 2, not RoPE/ALiBi.
- The current source is pre-norm inside every encoder/decoder layer, then applies final `encoder.layer_norm` and `decoder.layer_norm`.
- Attention is plain MHA, not GQA/MQA. Cache tensors store all heads.
- FFN activation can be `gelu` or `relu` in representative official configs.
- Config fields `normalize_before`, `normalize_embedding`, `static_position_embeddings`, and `extra_pos_embeddings` are historical compatibility fields for these checkpoints; current source behavior is not toggled by them.
- Layout translation is mostly irrelevant for text tensors. Keep `[B, S, D]` hidden states and `[B, H, S, Hd]` attention/cache tensors under a no-layout-translation guard unless a fused attention lowering owns the full region.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token inputs `[B, S_src]`, decoder ids `[B, S_tgt]`, optional masks `[B, S]`.
- Shared embedding gather from `[vocab, d_model]`; optional `sqrt(d_model)` scale.
- Learned position id arange/expand with cache offset; gather from `[max_pos + 2, d_model]`.
- Add, residual add, reshape/view, transpose between `[B, S, D]` and `[B, H, S, Hd]`, and contiguous materialization after attention transpose.
- Additive bidirectional and causal mask construction with padding masks and past-cache length.
- EOS/language-token indexing for `shift_tokens_right` and generation prompt construction.
- Cache append, cross-cache update flag, beam repeat/select/reorder for generation.

Neural network primitives:

- LayerNorm over hidden dim. Source uses `nn.LayerNorm(embed_dim)` with PyTorch default eps.
- Linear+bias for Q/K/V/O projections: `D -> D`.
- FFN: `Linear(D -> F)`, activation (`gelu` or `relu`), `Linear(F -> D)`.
- LM head: `Linear(D -> vocab, bias=False)` tied to shared embedding, then add `final_logits_bias [1, vocab]`.
- Dropout and LayerDrop exist in source but are disabled or skipped for inference.
- Encoder fp16 guard clamps hidden states to `[-finfo(fp16).max + 1000, +finfo(fp16).max - 1000]` after the FFN residual.

Attention primitives:

- Encoder bidirectional self-attention: Q/K/V `[B, H_enc, S_src, Hd]`.
- Decoder causal self-attention with DynamicCache: Q `[B, H_dec, S_new, Hd]`, K/V `[B, H_dec, S_past + S_new, Hd]`.
- Decoder cross-attention: Q from decoder hidden, K/V from encoder hidden `[B, S_src, D]`, cached as `[B, H_dec, S_src, Hd]`.
- Attention math order in eager path: `matmul(q, k^T) * head_dim^-0.5`, add mask, softmax over last dim, dropout, `matmul(probs, v)`, transpose, output projection.
- Source advertises FlashAttention, SDPA, and flex-attention compatibility through `ALL_ATTENTION_FUNCTIONS`.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` for decoder self cache and cross cache.
- Cross-cache `is_updated[layer_idx]` flag to avoid recomputing encoder K/V after first use.
- Generic `GenerationMixin.prepare_inputs_for_generation` slicing and beam cache reorder.
- Forced target-language BOS token via `forced_bos_token_id`.
- Forced EOS via `forced_eos_token_id`.
- `MBartForCausalLM` supports `logits_to_keep`; `MBartForConditionalGeneration` does not expose this argument, so last-token-only seq2seq logits is a DinoML rewrite opportunity.

Preprocessing-coupled ops:

- Tokenizer-backed language code maps and special-token post-processors.
- Translation helper requires `src_lang` and `tgt_lang`, then injects `forced_bos_token_id`.
- No token type ids, multimodal tensors, packed sequence descriptors, or scatter embedding stitch.

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
x_norm = LayerNorm(x)
q,k,v = Linear(D -> D, bias=True)(x_norm), reshape to [B, Henc, Ssrc, Hd]
self = MHA(q,k,v, bidirectional padding mask)
x = residual + Dropout(Linear(D -> D)(self))

residual = x
x_norm = LayerNorm(x)
ff = Linear(D -> F)(x_norm)
ff = GELU_or_ReLU(ff)
ff = Linear(F -> D)(ff)
x = residual + Dropout(ff)
x = clamp(x) only for fp16 encoder hidden states
```

After all encoder layers:

```text
encoder_last_hidden = LayerNorm(x)
```

Decoder block, repeated `decoder_layers` times:

```text
residual = y
y_norm = LayerNorm(y)
q = Linear(D -> D)(y_norm) -> [B, Hdec, Stgt, Hd]
k,v = Linear(D -> D)(y_norm) -> self-cache update -> [B, Hdec, Spast+Stgt, Hd]
self = causal MHA(q,k,v, decoder mask)
y = residual + Dropout(Linear(D -> D)(self))

residual = y
y_norm = LayerNorm(y)
q = Linear(D -> D)(y_norm)
k,v = Linear(D -> D)(encoder_hidden_states) or reusable cross cache
cross = bidirectional MHA(q,k,v, encoder padding mask)
y = residual + Dropout(Linear(D -> D)(cross))

residual = y
y_norm = LayerNorm(y)
ff = Linear(D -> F)(y_norm) -> GELU_or_ReLU -> Linear(F -> D)
y = residual + Dropout(ff)
```

After all decoder layers:

```text
decoder_last_hidden = LayerNorm(y)
logits = MatMul(decoder_last_hidden, shared_embedding.T) + final_logits_bias
```

For the large checkpoints, `D=1024`, `H=16`, `Hd=64`, `F=4096`, and both encoder/decoder layer counts are 12.

## 6. Attention requirements

Required variants:

- Encoder self-attention: noncausal bidirectional MHA, no KV cache.
- Decoder self-attention: causal MHA with growing self KV cache.
- Decoder cross-attention: noncausal MHA from decoder queries to encoder K/V, with a static cross KV cache after the first projection.

Masking:

- Encoder uses `create_bidirectional_mask(config, inputs_embeds, attention_mask)`.
- Decoder creates an all-ones attention mask when absent and not compiling, then calls `create_causal_mask` with self-cache metadata.
- Cross-attention uses `create_bidirectional_mask(..., encoder_hidden_states=encoder_hidden_states)` and the source `attention_mask`.
- Eager mask material is additive over attention scores before softmax. Fused attention parity must preserve this order or prove equivalent backend mask semantics.

Cache tensor shapes:

- Self cache per decoder layer before decode step `t`: K/V `[B, Hdec, t, Hd]`.
- Self cache after appending `S_new`: K/V `[B, Hdec, t + S_new, Hd]`.
- Cross cache per decoder layer after first use: K/V `[B, Hdec, S_src, Hd]`.
- `EncoderDecoderCache.get_seq_length()` reports self-attention cache length for decoder positions.
- Cached keys are stored after learned token/position embedding has already affected hidden states and after K projection; there is no RoPE transform to track separately.

Backend compatibility:

- `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are true on `MBartPreTrainedModel`.
- No sliding window, block sparse attention, packed varlen metadata, ALiBi, RoPE, or KV head repetition appears in the mBART source.

## 7. Position encoding and custom math

mBART uses learned absolute positions for encoder and decoder:

```python
def mbart_position_lookup(weight, input_shape, past_key_values_length=0, position_ids=None):
    offset = 2
    if position_ids is None:
        bsz, seq_len = input_shape[:2]
        pos = arange(past_key_values_length, past_key_values_length + seq_len)
        pos = pos.expand(bsz, seq_len)
    else:
        pos = position_ids.unsqueeze(0)
    return embedding(weight, pos + offset)
```

Decoder position depends on `past_key_values_length`; prefill uses positions starting at 0, decode token `t` uses position `t`.

`shift_tokens_right` is mBART-specific because it starts the decoder with the last non-pad token, usually the language ID token at the end of labels/source ids:

```python
def mbart_shift_tokens_right(input_ids, pad_id):
    y = input_ids.clone()
    y = where(y == -100, pad_id, y)
    last = y.ne(pad_id).sum(dim=1) - 1
    start = gather(y, dim=1, index=last[:, None]).squeeze()
    y[:, 1:] = y[:, :-1]
    y[:, 0] = start
    return y
```

Precomputable:

- Learned position weights are constants.
- Position lookup rows can be precomputed for fixed source/prefill buckets.
- Decode position lookup depends only on cache length and can use one-row gathers per step.

## 8. Preprocessing and input packing

Tokenizer and language-code coupling:

- `MBartTokenizer` has 25 FAIRSEQ language codes and default `src_lang="en_XX"`.
- `MBart50Tokenizer` has 50 language codes and is selected by mBART-50 configs through `tokenizer_class=MBart50Tokenizer`.
- `MBartTokenizer` source/target post-processing sets no prefix and suffix `[eos, lang_code]`.
- `MBart50Tokenizer` source/target post-processing sets prefix `[lang_code]` and suffix `[eos]`.
- Both tokenizers expose `model_input_names = ["input_ids", "attention_mask"]`.

Generation-controller behavior:

- Translation helper `_build_translation_inputs` requires both `src_lang` and `tgt_lang`, tokenizes with source language specials, converts target language to an id, and returns `forced_bos_token_id`.
- For many-to-many mBART-50, correct target language is therefore a generation constraint, not just a decoder input convention.
- `forced_eos_token_id=2` is present in current config defaults and mBART-50 configs.

GPU/runtime inputs:

- Required core graph inputs: `input_ids [B, S_src]`, `attention_mask [B, S_src]`, and either `decoder_input_ids [B, S_tgt]` or generation-managed decoder ids.
- No segment/token type ids.
- No image/audio tensors, placeholder expansion, `cu_seqlens`, or packed modality metadata.

CPU/data-pipeline recommendation:

- Keep SentencePiece/Unigram tokenization and language-code string-to-id mapping outside the first DinoML GPU graph.
- The graph should receive numeric token IDs, masks, and generation metadata such as target language ID / forced BOS.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared embedding and LM head tying

Source pattern: `self.model.shared` feeds encoder and decoder embeddings, and `_tied_weights_keys` maps `lm_head.weight` to `model.shared.weight`.

Replacement pattern: one constant `E [vocab, D]`; embedding gather uses `E`, logits use GEMM `hidden_flat [B*S, D] x E.T [D, vocab]`.

Preconditions:

- Weight tie is present or checkpoint tensors are byte-identical.
- Preserve separate `final_logits_bias [1, vocab]`.
- Vocab size must come from config/weights, not from family defaults.

Failure cases:

- Resized or untied checkpoints; mBART-50 vocab mismatch; missing `final_logits_bias`.

Parity test sketch:

- Compare embedding outputs and logits with tied fixture weights for cc25 and mBART-50 vocab widths.

### Rewrite: pack self-attention QKV projections

Source pattern: separate `q_proj`, `k_proj`, `v_proj` on the same hidden tensor for encoder self-attn and decoder self-attn.

Replacement pattern: packed `Linear(D -> 3D)` followed by split into Q/K/V.

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], dim=0)
b_qkv = concat([b_q, b_k, b_v], dim=0)
```

Preconditions:

- Same input tensor for all three projections.
- Biases are present and compatible.
- Output split order is Q, K, V.

Failure cases:

- Cross-attention Q input differs from K/V input; only K/V can be packed there.

Parity test sketch:

- Per-layer random input compare Q/K/V tensors before attention.

### Rewrite: precompute cross-attention K/V

Source pattern: cross-attention projects encoder states at first decoder use, then `EncoderDecoderCache.is_updated[layer]` causes reuse.

Replacement pattern: after encoder, precompute K/V for every decoder layer into a cross-cache buffer.

Preconditions:

- Encoder hidden states and cross-attn weights are fixed for the request.
- Beam expansion/reorder is applied to cross cache consistently.

Shape equations:

- Per decoder layer: `[B, S_src, D] -> K,V [B, Hdec, S_src, Hd]`.

Failure cases:

- Runtime-supplied `encoder_outputs` with unknown layout, source batch reorder, or cache invalidation not visible to DinoML.

Parity test sketch:

- Compare first-step source projection and later-step reused cross cache against full HF decode.

### Rewrite: last-token-only seq2seq logits

Source pattern: `MBartForConditionalGeneration` computes `lm_head(outputs[0])` for all target positions.

Replacement pattern: during autoregressive decode, slice decoder hidden to the last generated token before LM head.

Preconditions:

- Generation only needs next-token logits.
- Loss/full-sequence logits are not requested.

Failure cases:

- Teacher forcing, sequence scoring, or APIs requiring all logits.

Parity test sketch:

- Compare `full_logits[:, -1, :]` with logits from sliced hidden.

### Rewrite: language-token prompt specialization

Source pattern: tokenizer/generation code sets language ID special tokens and `forced_bos_token_id`.

Replacement pattern: generation metadata carries `{src_lang_id, tgt_lang_id, tokenizer_layout}`; graph receives already-packed ids or a small prompt builder emits decoder start/forced BOS ids.

Preconditions:

- Tokenizer class is known: `MBartTokenizer` versus `MBart50Tokenizer`.
- Target language ID is available and in vocab.

Failure cases:

- Unknown remote tokenizer, missing language code, custom additional specials that shift IDs.

Parity test sketch:

- Tokenize one sentence for cc25 and mBART-50, verify source ids, first generated token forcing, and decoded language.

## 10. Kernel fusion candidates

Highest priority:

- Encoder and decoder MHA / FlashAttention-compatible kernels. Large checkpoints use 12+12 layers with `D=1024`, `H=16`.
- Decoder self KV cache append plus causal attention for decode.
- Cross-attention K/V precompute and cache reuse.
- LM head last-token GEMM with tied embedding and `final_logits_bias`.

Medium priority:

- Packed QKV projection for self-attention and packed KV projection for cross-attention.
- LayerNorm + residual fusion around pre-norm attention/FFN blocks.
- FFN fusion for `Linear -> GELU/ReLU -> Linear`, with activation determined by config.
- Embedding + position add + LayerNorm fusion, including optional embedding scale.

Lower priority:

- Beam cache reorder/select kernels.
- Additive mask materialization avoidance when backend can consume causal/padding masks directly.
- Encoder fp16 clamp guard as a small elementwise kernel or documented numerical fallback.

## 11. Runtime staging plan

Stage 1: parse `MBartConfig`, tokenizer class metadata, language-code metadata, shared embeddings, learned positions, layer weights, LM tie, and `final_logits_bias`.

Stage 2: implement encoder-only parity with embedding scale, learned positions, bidirectional mask, pre-norm blocks, and final LayerNorm.

Stage 3: implement teacher-forced seq2seq forward without cache: decoder causal self-attn, cross-attn, FFN, and full logits.

Stage 4: implement `EncoderDecoderCache`: self cache append, cross cache update/reuse, and cache length position offsets.

Stage 5: implement greedy generation parity with explicit target language forcing and EOS behavior.

Stage 6: add beam-search cache repeat/reorder and translation pipeline controller parity for representative checkpoints.

Stage 7: enable optimized attention, packed projection rewrites, cross-cache precompute, last-token logits, and fused FFN/LayerNorm paths.

Initially stub:

- Training losses, dropout, LayerDrop, gradient checkpointing, sequence classification, QA, decoder-only CausalLM, speculative/assisted generation, and tokenizer execution inside the GPU runtime.

## 12. Parity and validation plan

- Custom op tests: learned position lookup with offset 2; `mbart_shift_tokens_right`; tokenizer language-special layout metadata; forced BOS target language ID.
- Mask tests: bidirectional source mask, decoder causal mask with past length, cross-attn source padding mask.
- Single-layer parity: one encoder layer and one decoder layer with random tensors; compare fp32 hidden states at `atol=1e-5` / `rtol=1e-4`.
- Full encoder parity: `facebook/mbart-large-cc25` and tiny debug config on short tokenized text.
- Teacher-forced seq2seq parity: compare logits for `[B, S_src, S_tgt]` with explicit `decoder_input_ids`.
- Cache parity: prefill plus token-by-token decode must match no-cache full decode for the same target ids.
- Cross-cache parity: verify first decode step fills cross K/V and later steps reuse it while matching outputs.
- mBART-50 parity: use `facebook/mbart-large-50-many-to-many-mmt` to cover `relu`, 250054 vocab, and prefix language-token tokenizer layout.
- Generation parity: greedy token sequence first, then beam output with exact `forced_bos_token_id`/`forced_eos_token_id`.
- Tolerances: fp32 `1e-5` absolute / `1e-4` relative; fp16/bf16 logits around `2e-2` absolute with top-k/token parity checks.

## 13. Performance probes

- Tokenizer/data-pipeline throughput by tokenizer class: cc25 `MBartTokenizer` layout versus mBART-50 `MBart50Tokenizer` layout.
- Encoder-only throughput over `B x S_src`: sweep 32, 128, 512, 1024 tokens.
- Decoder prefill throughput over `S_tgt`: sweep 1, 16, 64, 200.
- Decode tokens/sec with KV cache: batch and beam sweeps, split self-attn, cross-attn, FFN, and LM head time.
- Cross-cache memory: `decoder_layers * 2 * B * beams * H * S_src * Hd * dtype_size`.
- Self-cache memory growth: `decoder_layers * 2 * B * beams * H * S_tgt * Hd * dtype_size`.
- LM head cost: full `[B, S_tgt, vocab]` logits versus last-token-only `[B, 1, vocab]`.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible fused path, and DinoML native path.
- Activation variation: GELU cc25/large-50 versus ReLU many-to-many checkpoint.

## 14. Skip/defer list

- Training, losses, dropout, LayerDrop, gradient checkpointing.
- Sequence classification and question answering heads.
- Decoder-only `MBartForCausalLM`, except reuse decoder/cache findings later.
- Running SentencePiece/Unigram tokenization inside DinoML runtime.
- Full HF generation feature matrix beyond forced BOS/EOS, greedy, and beam search needed for translation parity.
- Quantized/offloaded/static cache variants beyond first dynamic cache.
- Custom remote-code behavior; inspected official in-library source requires no remote code.

## 15. Final implementation checklist

- [ ] Parse `MBartConfig`, including vocab size, activation, scale embedding, and encoder/decoder depths.
- [ ] Load shared token embedding, learned positions with offset 2, projection/FFN/LayerNorm weights, and `final_logits_bias`.
- [ ] Preserve embedding/LM-head weight tying as one logical parameter.
- [ ] Accept tokenizer metadata for `MBartTokenizer` and `MBart50Tokenizer` language-code layouts.
- [ ] Implement learned absolute position lookup with decoder cache offsets.
- [ ] Implement mBART `shift_tokens_right` or require explicit decoder ids for first parity.
- [ ] Implement encoder bidirectional pre-norm MHA block and final LayerNorm.
- [ ] Implement decoder causal self-attn, cross-attn, FFN, and final LayerNorm.
- [ ] Implement `EncoderDecoderCache` self append, cross-cache reuse, and beam reorder.
- [ ] Implement mask lowering for source padding, decoder causal+padding, and cross-attn padding.
- [ ] Implement LM head tied to shared embedding plus `final_logits_bias`.
- [ ] Add forced target-language BOS and forced EOS generation-controller handling.
- [ ] Add packed QKV/KV projection rewrites with guarded weight transforms.
- [ ] Add cross-attn K/V precompute rewrite.
- [ ] Add last-token-only logits rewrite for seq2seq decode.
- [ ] Add one-layer, full-forward, cache, mBART-50, and generation parity tests.
- [ ] Benchmark encoder, prefill, decode, LM head, tokenizer, and cache memory separately.
