# PLBart Transformers Audit

## 1. Source basis

Transformers commit/version: local checkout `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary `uclanlp/plbart-base`; representative sweep also used `uclanlp/plbart-large`, `uclanlp/plbart-java-en_XX`, and `uclanlp/plbart-python-en_XX`.

Config source:

- `https://huggingface.co/uclanlp/plbart-base/raw/main/config.json`
- `https://huggingface.co/uclanlp/plbart-large/raw/main/config.json`
- `https://huggingface.co/uclanlp/plbart-java-en_XX/raw/main/config.json`
- `https://huggingface.co/uclanlp/plbart-python-en_XX/raw/main/config.json`

Source files inspected:

- `transformers/src/transformers/models/plbart/configuration_plbart.py`
- `transformers/src/transformers/models/plbart/modeling_plbart.py`
- `transformers/src/transformers/models/plbart/modular_plbart.py`
- `transformers/src/transformers/models/plbart/tokenization_plbart.py`
- `transformers/src/transformers/masking_utils.py` through source references
- `transformers/src/transformers/cache_utils.py` through source references

Local snapshots:

- `_sources/configuration_plbart.py`
- `_sources/modeling_plbart.py`
- `_sources/modular_plbart.py`
- `_sources/tokenization_plbart.py`
- `_sources/uclanlp_plbart-base_config.json`
- `_sources/uclanlp_plbart-large_config.json`
- `_sources/uclanlp_plbart-java-en_XX_config.json`
- `_sources/uclanlp_plbart-python-en_XX_config.json`
- `_sources/config_notes.md`

Any missing files or assumptions:

- `modeling_plbart.py` is generated from `modular_plbart.py`; future source edits should target the modular file, while runtime ABI should follow the generated file that Transformers imports.
- The reachable official repos expose `config.json`, `pytorch_model.bin`, and `sentencepiece.bpe.model`, but `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`, and `generation_config.json` returned 404 during this audit.
- `uclanlp/plbart-multi_task` returned 401 Unauthorized for `config.json`; it is out of scope until access is available.
- Primary runtime target is `PLBartForConditionalGeneration` for code/text seq2seq generation. Sequence classification and decoder-only `PLBartForCausalLM` are optional or deferred heads.

## 2. High-level architecture

PLBart is a text/code-only encoder-decoder Transformer in the BART family. The encoder uses bidirectional self-attention. The decoder uses causal self-attention plus encoder-decoder cross-attention. Both sides use shared SentencePiece-aligned token embeddings, learned absolute positions with an offset of 2, embedding LayerNorm, post-attention/post-FFN LayerNorm, GELU FFNs, and a tied LM head plus `final_logits_bias`.

Dataflow:

```text
SentencePiece tokenizer + PL/code language suffix tokens
  -> input_ids/attention_mask
  -> shared token embedding * sqrt(d_model) + learned positions -> embedding LayerNorm
  -> encoder bidirectional blocks
  -> decoder language/start token sequence + learned positions -> decoder causal self-attn
  -> decoder cross-attn over encoder states
  -> tied lm_head + final_logits_bias
  -> generation controller with forced target language / EOS rules
```

Stage decomposition:

- CPU/data pipeline: SentencePiece tokenization, fairseq ID alignment, language-code suffix insertion, padding, source/target language selection, and generation metadata such as `forced_bos_token_id`.
- Encoder: independently cacheable per source sequence; output shape `[batch, src_len, d_model]`.
- Decoder prefill: target prefix or shifted source/label ids, causal self-attn, and first cross-attn K/V projection from encoder states.
- Decode: one or more new target tokens with growing decoder self KV cache and static cross-attn cache.
- Logits/sampling: `lm_head(hidden) + final_logits_bias`; beam search and language forcing live outside the core graph.

## 3. Important config dimensions

Source-default `PLBartConfig` fields:

| Field | Default | Operator significance |
|---|---:|---|
| `vocab_size` | 50005 | Shared embedding rows and LM head output columns |
| `max_position_embeddings` | 1024 | Position table has `max_position_embeddings + 2` rows |
| `d_model` | 768 | Hidden width |
| `encoder_layers` / `decoder_layers` | 6 / 6 | Repeated block counts |
| `encoder_attention_heads` / `decoder_attention_heads` | 12 / 12 | Plain MHA head counts |
| `head_dim` | inferred 64 | `d_model / heads`; source requires exact divisibility |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 3072 / 3072 | FFN expansion |
| `activation_function` | `gelu` | FFN activation through `ACT2FN` |
| `dropout` / `attention_dropout` / `activation_dropout` | 0.1 / 0.1 / 0.0 | Inference dropout disabled |
| `scale_embedding` | true | Token embeddings multiply by `sqrt(d_model)` |
| `use_cache` | true | Decoder generation uses self and cross KV caches |
| `pad/bos/eos` | 1 / 0 / 2 | Token and mask defaults |
| `forced_eos_token_id` | 2 | Generation-controller EOS constraint |
| `tie_word_embeddings` | true | LM head aliases shared token embedding |

Representative checkpoint sweep:

| Model id | Arch in config | `d_model` | Enc/Dec layers | Heads | FFN | Vocab | Max pos | Tokenizer/generation notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `uclanlp/plbart-base` | `PLBartForConditionalGeneration` | 768 | 6 / 6 | 12 / 12 | 3072 | 50005 | 1024 | Sparse repo; no tokenizer/generation JSON; source default language set applies |
| `uclanlp/plbart-large` | `PLBartForConditionalGeneration` | 1024 | 12 / 12 | 16 / 16 | 4096 | 50005 | 1024 | Same topology at larger width/depth; config records `torch_dtype=float32` |
| `uclanlp/plbart-java-en_XX` | `PLBartForConditionalGeneration` | 768 | 6 / 6 | 12 / 12 | 3072 | 50005 | 1024 | Fine-tuned code-to-English config is structurally identical to base |
| `uclanlp/plbart-python-en_XX` | `PLBartForConditionalGeneration` | 768 | 6 / 6 | 12 / 12 | 3072 | 50005 | 1024 | Fine-tuned code-to-English config is structurally identical to base |
| `uclanlp/plbart-multi_task` | gated | unknown | unknown | unknown | unknown | unknown | unknown | 401 for config; do not infer operator changes |

Omitted checkpoint fields use current source defaults such as `classifier_dropout=0.0`, `is_encoder_decoder=True`, `is_decoder=False`, and `tie_word_embeddings=True` unless present in the config.

## 3a. Family variation traps

- PLBart has no universal `decoder_start_token_id`. `shift_tokens_right` starts the decoder with the last non-pad token, normally the source/target language ID suffix.
- Tokenizer language control is ABI-significant. Source layouts are suffix-based: source and target modes use `tokens, eos, lang_code` when a language is known.
- Base language codes are `java`, `python`, and `en_XX`; optional tokenizer mode `language_codes="multi"` expands to Java, Python, English, JavaScript, PHP, Ruby, and Go. The reachable checkpoint configs do not declare this mode, so route it through tokenizer metadata rather than model config alone.
- Fairseq/SentencePiece ID alignment is nontrivial: `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`, normal SentencePiece ids are offset by 1 except SP unk maps to 3, language IDs are appended after the SP model, and base mode adds `<mask>` after language IDs.
- Learned positions use an offset of 2; position weight shape is `[max_position_embeddings + 2, d_model]`.
- The source is post-norm inside encoder/decoder blocks, with an embedding LayerNorm before the first block. Do not reuse mBART pre-norm lowering by name.
- Attention is plain MHA, not GQA/MQA; all heads have K/V tensors.
- No source/config evidence of RoPE, ALiBi, sliding window, block sparse attention, MoE, gated FFN, or packed projection weights.
- `modeling_plbart.py` advertises FlashAttention, SDPA, and flex attention support through generic Transformers dispatch, but eager parity remains standard additive-mask MHA.
- Layout translation is mostly irrelevant for text tensors. Guard `[B, S, D]` hidden states and `[B, H, S, Hd]` attention/cache tensors against generic channel-last rewrites unless a fused attention lowering owns the whole region.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer `input_ids [B, S_src]`, `decoder_input_ids [B, S_tgt]`, `attention_mask [B, S_src]`, optional decoder masks.
- Shared embedding gather from `[vocab, D]`; multiply by `sqrt(D)` when `scale_embedding=true`.
- Learned position arange/expand with offset 2 and decoder cache-length offset.
- Add, residual add, reshape/view, transpose `[B, S, D] <-> [B, H, S, Hd]`, and contiguous materialization after attention.
- Padding-mask conversion to additive bidirectional masks and causal decoder masks with past-cache length.
- `where`, `ne`, `sum`, `gather`, clone/update semantics for `shift_tokens_right`.
- Cache append, cross-cache update flags, beam repeat/select/reorder for generation.

Neural network primitives:

- LayerNorm over hidden dim, using PyTorch default eps because source constructs `nn.LayerNorm(embed_dim)` without explicit eps.
- Linear+bias projections: Q/K/V/O all `D -> D`.
- FFN: `Linear(D -> F)`, GELU, `Linear(F -> D)`.
- LM head: `Linear(D -> vocab, bias=False)` tied to shared embedding, followed by `final_logits_bias [1, vocab]`.
- Sequence classification head, optional: EOS-token pooling, dropout, `Linear(D -> D)`, tanh, dropout, `Linear(D -> num_labels)`.
- Inference should erase dropout and LayerDrop. Training losses are not required for first integration.
- Encoder fp16 non-finite clamp guard after the FFN block: clamp to `finfo(float16).max - 1000` if any nonfinite appears.

Attention primitives:

- Encoder bidirectional self-attention: Q/K/V `[B, H_enc, S_src, Hd]`.
- Decoder causal self-attention with dynamic self KV cache: Q `[B, H_dec, S_new, Hd]`, K/V `[B, H_dec, S_past + S_new, Hd]`.
- Decoder cross-attention over encoder hidden states: Q from decoder hidden, K/V from encoder hidden, reusable cross cache `[B, H_dec, S_src, Hd]`.
- Eager math order: `matmul(q, k.transpose(-2, -1)) * head_dim^-0.5`, add mask, softmax over key dimension, dropout, `matmul(probs, v)`, transpose, output projection.

Generation/cache ops:

- `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq decode when encoder states are present.
- `past_key_values.is_updated[layer_idx]` gates cross-attn K/V reuse after first projection.
- `DynamicCache` for decoder-only `PLBartForCausalLM`.
- `PLBartForCausalLM` supports `logits_to_keep`; `PLBartForConditionalGeneration` does not expose it, so seq2seq last-token logits is a DinoML rewrite opportunity.

Preprocessing-coupled ops:

- SentencePiece tokenization and fairseq token-id mapping.
- PL/code language-code maps and source/target suffix insertion.
- Translation helper requires `src_lang` and `tgt_lang`, then returns `forced_bos_token_id` for the target language.
- No token type IDs, multimodal placeholders, packed varlen descriptors, or scatter embedding stitch.

## 5. Layer/block breakdown

Embedding path:

```text
input_ids [B, S] -> shared embedding [B, S, D] * sqrt(D)
position_ids = arange(past_len, past_len + S)
positions = learned_pos[position_ids + 2] [B, S, D]
x = LayerNorm(token_emb + positions)
```

Encoder block, repeated `encoder_layers` times:

```text
residual = x
q,k,v = Linear(D -> D, bias=True)(x), reshape to [B, Henc, Ssrc, Hd]
self = MHA(q,k,v, bidirectional mask)
x = LayerNorm(residual + Dropout(Linear(D -> D, bias=True)(self)))

residual = x
ff = Linear(D -> F, bias=True)(x)
ff = GELU(ff)
ff = Linear(F -> D, bias=True)(ff)
x = LayerNorm(residual + Dropout(ff))
x = fp16 finite clamp guard only if dtype is fp16 and nonfinite values exist
```

Decoder block, repeated `decoder_layers` times:

```text
residual = y
q = Linear(D -> D)(y) -> [B, Hdec, Stgt, Hd]
k,v = Linear(D -> D)(y) -> self-cache update -> [B, Hdec, Spast+Stgt, Hd]
self = causal MHA(q,k,v, decoder mask)
y = LayerNorm(residual + Dropout(Linear(D -> D)(self)))

residual = y
q = Linear(D -> D)(y)
k,v = Linear(D -> D)(encoder_hidden_states) or reusable cross cache
cross = bidirectional MHA(q,k,v, encoder padding mask)
y = LayerNorm(residual + Dropout(Linear(D -> D)(cross)))

residual = y
ff = Linear(D -> F)(y) -> GELU -> Linear(F -> D)
y = LayerNorm(residual + Dropout(ff))
```

Conditional generation head:

```text
decoder_last_hidden [B, Stgt, D]
logits = MatMul(decoder_last_hidden, shared_embedding.T) + final_logits_bias[1, vocab]
```

For base/code fine-tunes, `D=768`, `H=12`, `Hd=64`, `F=3072`, layers are 6/6, and vocab is 50005. For large, `D=1024`, `H=16`, `Hd=64`, `F=4096`, layers are 12/12.

## 6. Attention requirements

Required variants:

- Encoder self-attention: noncausal bidirectional MHA, no KV cache.
- Decoder self-attention: causal MHA with growing self KV cache.
- Decoder cross-attention: noncausal MHA from decoder queries to encoder K/V, with static cross cache after first use.

Masking:

- Encoder calls `create_bidirectional_mask(config, inputs_embeds, attention_mask)`.
- Decoder creates an all-ones mask when none is supplied and not TorchDynamo-compiling, then calls `create_causal_mask` with the self-cache view.
- Cross-attention calls `create_bidirectional_mask(..., encoder_hidden_states=encoder_hidden_states)`.
- Eager masks are additive over attention scores before softmax; fused attention must preserve equivalent mask polarity/value behavior.

Cache tensor shapes:

- Self cache per decoder layer before token step `t`: K/V `[B, Hdec, t, Hd]`.
- Self cache after appending `S_new`: K/V `[B, Hdec, t + S_new, Hd]`.
- Cross cache per decoder layer after first use: K/V `[B, Hdec, S_src, Hd]`.
- `EncoderDecoderCache.get_seq_length()` reports self-cache length for decoder position offsets.
- Cached keys are stored after linear projection of hidden states; there is no RoPE or relative-position transform to track.

Backend compatibility:

- Source advertises `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn`.
- No packed varlen ABI, sliding windows, block-sparse masks, ALiBi, RoPE, or KV-head repetition is required for PLBart source parity.

## 7. Position encoding and custom math

PLBart uses learned absolute positions with offset 2:

```python
def plbart_position_lookup(weight, input_shape, past_key_values_length=0, position_ids=None):
    offset = 2
    if position_ids is None:
        bsz, seq_len = input_shape[:2]
        pos = arange(past_key_values_length, past_key_values_length + seq_len)
        pos = pos.expand(bsz, seq_len)
    else:
        pos = position_ids.unsqueeze(0)
    return embedding(weight, pos + offset)
```

Decoder position depends on self-cache length. Source decoder constructs `position_ids = arange(seq_length) + past_key_values_length` and passes it into the embedding module.

PLBart `shift_tokens_right` is language-token dependent:

```python
def plbart_shift_tokens_right(input_ids, pad_id):
    y = input_ids.clone()
    y = where(y == -100, pad_id, y)
    last = y.ne(pad_id).sum(dim=1) - 1
    start = gather(y, dim=1, index=last[:, None]).squeeze()
    y[:, 1:] = y[:, :-1]
    y[:, 0] = start
    return y
```

Precomputable:

- Position weights are constants.
- Encoder and decoder prefill position ids can be precomputed for fixed buckets.
- Decode position lookup depends only on cache length and can be generated as one-row/tiny gathers per step.

## 8. Preprocessing and input packing

Tokenizer and language-code coupling:

- `PLBartTokenizer` is SentencePiece-backed and requires the `sentencepiece` backend.
- `model_input_names = ["input_ids", "attention_mask"]`; no token type IDs enter the model.
- `FAIRSEQ_LANGUAGE_CODES["base"] = ["__java__", "__python__", "__en_XX__"]`.
- `FAIRSEQ_LANGUAGE_CODES["multi"] = ["__java__", "__python__", "__en_XX__", "__javascript__", "__php__", "__ruby__", "__go__"]`.
- Friendly codes map to special tokens: `java -> __java__`, `python -> __python__`, `en_XX -> __en_XX__`, and likewise for JavaScript/PHP/Ruby/Go in multi mode.
- Source and target modes both set no prefix and suffix `[eos_token_id, lang_code_id]` when the language is known; if no source language is set in base mode, suffix is only `[eos]`.
- `_build_translation_inputs` requires both `src_lang` and `tgt_lang`, tokenizes with source specials, then returns `forced_bos_token_id = target_language_id`.

Fairseq/SentencePiece ID mapping:

- Reserved ids are `<s>=0`, `<pad>=1`, `</s>=2`, `<unk>=3`.
- SP piece id 0 maps to unk id 3; other SP ids map to `spm_id + 1`.
- Language code ids are `sp_model_size + i + 1`.
- In base mode, `<mask>` id is `len(sp_model) + len(lang_codes) + 1`.

GPU/runtime inputs:

- Core graph can accept pre-tokenized `input_ids [B, S_src]`, `attention_mask [B, S_src]`, and explicit `decoder_input_ids [B, S_tgt]`.
- For first parity, prefer explicit `decoder_input_ids`; add `plbart_shift_tokens_right` only when reproducing HF automatic denoising/label behavior.
- Generation parity needs tokenizer metadata `{src_lang_id, tgt_lang_id, language_codes_mode}` and controller handling for `forced_bos_token_id`/`forced_eos_token_id`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared embedding and LM head tying

Source pattern: `self.model.shared` feeds encoder and decoder embeddings, and `_tied_weights_keys` maps `lm_head.weight` to `model.shared.weight`.

Replacement pattern: one constant `E [vocab, D]`; embedding gather uses `E`, logits use GEMM `hidden_flat [B*S, D] x E.T [D, vocab]`; preserve `final_logits_bias`.

Preconditions:

- Weight tie is present or checkpoint tensors are byte-identical.
- `lm_head` has no bias.
- Vocab size comes from config/weight tensor, not hard-coded.

Failure cases:

- Resized/untied checkpoints or mismatched tokenizer/model vocab.

Parity test sketch:

- Compare encoder embedding, decoder embedding, and logits for tied fixture weights.

### Rewrite: pack self-attention QKV projections

Source pattern: separate `q_proj`, `k_proj`, and `v_proj` on the same tensor for encoder self-attn and decoder self-attn.

Replacement pattern: packed `Linear(D -> 3D)` followed by Q/K/V split.

Weight transform:

```python
W_qkv = concat([W_q, W_k, W_v], dim=0)
b_qkv = concat([b_q, b_k, b_v], dim=0)
```

Preconditions:

- Same input tensor for all three projections.
- Biases are present and compatible.
- Split order is Q, K, V.

Failure cases:

- Cross-attn Q uses decoder hidden while K/V use encoder hidden; only K/V can be packed there.

Parity test sketch:

- Compare projected Q/K/V tensors before reshape and after `[B, H, S, Hd]` view.

### Rewrite: precompute cross-attention K/V

Source pattern: each decoder layer projects encoder states on first cross-attn use, sets `past_key_values.is_updated[layer]`, then reuses the cross cache.

Replacement pattern: after encoder, precompute each decoder layer's cross K/V into cache buffers.

Preconditions:

- Encoder hidden states and cross-attn weights are fixed for the request.
- Beam expansion/reorder is applied to cross cache consistently.

Shape equations:

- Per decoder layer: `[B, S_src, D] -> K,V [B, Hdec, S_src, Hd]`.

Failure cases:

- Runtime-supplied `encoder_outputs` with unknown layout, mutated source batch ordering, or cache invalidation not visible to DinoML.

Parity test sketch:

- Compare first-step source projection and later-step reused cross cache against full HF decode.

### Rewrite: last-token-only seq2seq logits

Source pattern: `PLBartForConditionalGeneration` computes full logits for all decoder positions.

Replacement pattern: during autoregressive decode, slice decoder hidden to the last token before LM head.

Preconditions:

- Generation only needs next-token logits.
- Loss, teacher-forced full logits, and sequence scoring are not requested.

Shape equations:

- `[B, S_tgt, D] -> [B, 1, D] -> [B, 1, vocab]`.

Failure cases:

- Training, teacher forcing, or APIs requiring all token logits.

Parity test sketch:

- Compare `full_logits[:, -1, :]` with sliced-hidden logits.

### Rewrite: language suffix and decoder-start specialization

Source pattern: tokenizer appends `[eos, lang_id]`; `shift_tokens_right` copies the last non-pad token to decoder position 0.

Replacement pattern: keep tokenizer in CPU pipeline and pass explicit decoder ids, or generate a small prompt/shift op from numeric packed ids.

Preconditions:

- Source/target language IDs are known.
- Input rows contain exactly the intended language suffix before padding.

Failure cases:

- Missing language suffix, custom tokenizer mode not represented in metadata, variable or malformed padding where last non-pad is not the language ID.

Parity test sketch:

- Tokenize Java/Python/English examples, verify packed ids, shifted decoder starts, and first generated forced BOS target ID.

## 10. Kernel fusion candidates

Highest priority:

- Encoder and decoder dense MHA / FlashAttention-compatible kernels. Large PLBart has 24 attention blocks at `D=1024`, `H=16`.
- Decoder self-cache append plus causal attention for decode.
- Cross-attn K/V precompute/cache reuse; source explicitly supports reusing cross cache after first update.
- Last-token LM head GEMM with tied embedding and `final_logits_bias`.

Medium priority:

- Packed QKV projection for self-attn and packed KV projection for cross-attn.
- LayerNorm + residual fusion around post-norm attention/FFN boundaries.
- FFN fusion for `Linear -> GELU -> Linear`; GEMM epilogue GELU/bias paths are useful.
- Embedding + learned-position add + LayerNorm fusion, including `sqrt(D)` embedding scale.

Lower priority:

- Beam cache reorder/select kernels.
- Additive mask materialization avoidance when a fused attention backend can consume causal and padding masks directly.
- fp16 non-finite clamp guard as a small elementwise fallback.
- Sequence-classification EOS pooling kernels, only if classification becomes a target.

## 11. Runtime staging plan

Stage 1: parse `PLBartConfig`, load shared embedding, learned positions, all projection/FFN/LayerNorm weights, tied LM head, and `final_logits_bias`. Accept pre-tokenized ids and explicit decoder ids.

Stage 2: implement encoder-only parity with embedding scale, learned positions, embedding LayerNorm, bidirectional mask, and post-norm blocks.

Stage 3: implement teacher-forced seq2seq forward without cache: decoder causal self-attn, cross-attn, FFN, and full logits.

Stage 4: implement `EncoderDecoderCache`: self-cache append, cross-cache update/reuse, cache-length position offsets, and beam reorder hooks.

Stage 5: implement greedy generation parity with tokenizer-provided source/target language IDs, forced target BOS, and forced EOS.

Stage 6: add beam-search controller parity for representative translation/code-generation use.

Stage 7: enable optimized attention, packed projections, cross-cache precompute, last-token logits, and fused FFN/LayerNorm paths.

Initially stub:

- Training losses, dropout, LayerDrop, gradient checkpointing, tokenizer execution inside GPU runtime, sequence classification, decoder-only `PLBartForCausalLM`, speculative decoding, and quantized/offloaded cache variants.

## 12. Parity and validation plan

- Config/weight tests: base and large dimensions, tied embedding/LM head, position table size `max_position_embeddings + 2`.
- Tokenizer metadata tests: fairseq ID mapping, base language-code ids, suffix `[eos, lang_id]`, `forced_bos_token_id` returned for translation helper.
- Custom op tests: learned position lookup with offset 2; `plbart_shift_tokens_right`; mask creation for encoder, decoder causal with past length, and cross-attn source padding.
- Single-layer parity: one encoder layer and one decoder layer with random tensors; fp32 hidden states at `atol=1e-5` / `rtol=1e-4`.
- Full encoder parity: `uclanlp/plbart-base` short tokenized examples.
- Teacher-forced seq2seq parity: compare logits for `[B, S_src, S_tgt]` with explicit `decoder_input_ids`.
- Cache parity: prefill plus token-by-token decode must match no-cache full decode for the same target ids.
- Cross-cache parity: verify first step fills cross K/V, later steps reuse it, and outputs still match.
- Generation parity: greedy token sequence first, then beam output using explicit source/target language IDs and forced EOS.
- Tolerances: fp32 `1e-5` absolute / `1e-4` relative; fp16/bf16 logits around `2e-2` absolute plus top-k/token parity checks.

## 13. Performance probes

- SentencePiece/data-pipeline throughput for Java, Python, and English inputs.
- Encoder-only throughput sweep over `S_src`: 32, 128, 512, 1024.
- Decoder prefill throughput sweep over `S_tgt`: 1, 16, 64, 256.
- Decode tokens/sec with KV cache, split by self-attn, cross-attn, FFN, and LM head.
- Cross-cache memory: `decoder_layers * 2 * B * beams * H * S_src * Hd * dtype_size`.
- Self-cache memory growth: `decoder_layers * 2 * B * beams * H * S_tgt * Hd * dtype_size`.
- LM head cost: full `[B, S_tgt, vocab]` logits versus last-token-only `[B, 1, vocab]`.
- Attention backend comparison: eager, SDPA, FlashAttention-compatible fused path, and DinoML native attention.
- Base versus large throughput/capacity sweep to separate width/depth scaling from tokenizer overhead.

## 14. Skip/defer list

- Training, losses, dropout, LayerDrop, and gradient checkpointing.
- Running SentencePiece inside DinoML runtime; keep it in CPU/data pipeline initially.
- Sequence classification and decoder-only causal LM heads.
- Gated `uclanlp/plbart-multi_task` behavior until config/source access is available.
- Full HF generation feature matrix beyond forced BOS/EOS, greedy decode, and beam search needed for code/text generation parity.
- Quantized/offloaded/static cache implementations beyond the first dynamic cache.
- Remote-code behavior; inspected official in-library source requires no custom remote code.

## 15. Final implementation checklist

- [ ] Parse `PLBartConfig`, including base/large dimensions, embedding scale, and cache defaults.
- [ ] Load shared token embedding, learned positions with offset 2, projection/FFN/LayerNorm weights, and `final_logits_bias`.
- [ ] Preserve encoder embedding, decoder embedding, and LM head as one tied logical parameter.
- [ ] Accept tokenizer metadata for PLBart language-code mode and language-id suffix layout.
- [ ] Implement learned absolute position lookup with decoder cache offsets.
- [ ] Implement `plbart_shift_tokens_right` or require explicit decoder ids for first parity.
- [ ] Implement encoder bidirectional post-norm MHA blocks.
- [ ] Implement decoder causal self-attn, cross-attn, FFN, and post-norm residuals.
- [ ] Implement `EncoderDecoderCache` self append, cross-cache reuse flag, and beam reorder.
- [ ] Implement mask lowering for source padding, decoder causal+padding, and cross-attn padding.
- [ ] Implement LM head tied to shared embedding plus `final_logits_bias`.
- [ ] Add forced target-language BOS and forced EOS generation-controller handling.
- [ ] Add packed QKV/KV projection rewrites with guarded weight transforms.
- [ ] Add cross-attn K/V precompute rewrite.
- [ ] Add last-token-only logits rewrite for seq2seq decode.
- [ ] Add tokenizer metadata, one-layer, full-forward, cache, and generation parity tests.
- [ ] Benchmark tokenizer, encoder, prefill, decode, LM head, and cache memory separately.
