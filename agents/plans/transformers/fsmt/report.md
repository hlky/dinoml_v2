# FSMT Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 from local checkout X:/H/transformers

Model id:
  Primary representative family: facebook/wmt19-{en-ru,ru-en,en-de,de-en}
  Debug checkpoint attempted: sshleifer/tiny-fsmt-en-de, inaccessible via raw config request with 401.

Config source:
  Local source defaults in configuration_fsmt.py plus downloaded HF raw config/tokenizer/generation files under
  H:/dinoml_v2/agents/plans/transformers/fsmt/_sources/.

Source files inspected:
  X:/H/transformers/src/transformers/models/fsmt/configuration_fsmt.py
  X:/H/transformers/src/transformers/models/fsmt/modeling_fsmt.py
  X:/H/transformers/src/transformers/models/fsmt/tokenization_fsmt.py
  X:/H/transformers/src/transformers/models/fsmt/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py
  X:/H/transformers/tests/models/fsmt/test_modeling_fsmt.py
  X:/H/transformers/tests/models/fsmt/test_tokenization_fsmt.py

Any missing files or assumptions:
  No remote code is required for official WMT19 checkpoints. The tiny debug checkpoint could not be fetched without
  authorization, so this report uses official WMT19 configs and source defaults. No Dinoml tests/imports were run.
```

Authoritative modeling source is `modeling_fsmt.py`; no modular/generated model file exists for this family in the inspected checkout.

## 2. High-level architecture

FSMT is a text-only encoder-decoder translation model ported from fairseq WMT19. The first useful DinoML target is `FSMTForConditionalGeneration` seq2seq translation:

```text
Moses + fastBPE tokenization -> source token ids + source mask
  -> encoder embeddings + fairseq sinusoidal positions -> N encoder blocks
  -> decoder start/eos-controlled input ids + decoder positions
  -> N decoder blocks with causal self-attention and encoder cross-attention
  -> decoder output_projection -> target-vocab logits -> generation controller / detokenization
```

Stage boundaries:

- CPU/data pipeline: Moses punctuation normalization, non-printing-char removal, Moses tokenization, fastBPE, special-token append, target-language detokenization.
- Cacheable encoder: `encoder_last_hidden_state` `[B, S_src, d_model]` plus source attention mask can be reused across beam decode steps.
- Decoder prefill/teacher forcing: full target prefix with causal mask and decoder padding mask.
- Decode: one-token decoder call when `use_cache=True`, with autoregressive self-attention KV cache plus encoder-decoder cross-attention KV cache.
- Logits/generation: decoder already applies `output_projection`; `FSMTForConditionalGeneration.forward` returns logits directly.

## 3. Important config dimensions

Source default dimensions:

| Field | Source default | Runtime meaning |
|---|---:|---|
| `src_vocab_size` | 42024 | Encoder token embedding rows |
| `tgt_vocab_size` | 42024 | Decoder embedding rows and output-projection rows |
| `d_model` | 1024 | Hidden size |
| `encoder_layers` / `decoder_layers` | 12 / 12 | Source defaults only; WMT19 configs use 6 / 6 |
| `encoder_attention_heads` / `decoder_attention_heads` | 16 / 16 | MHA heads |
| `head_dim` | 64 | Inferred as `d_model // heads`; source asserts divisibility |
| `encoder_ffn_dim` / `decoder_ffn_dim` | 4096 / 4096 | Source defaults only; WMT19 encoder FFN differs |
| `activation_function` | `relu` | MLP activation via `ACT2FN` |
| `max_position_embeddings` | 1024 | Fairseq sinusoidal table grows if needed |
| `scale_embedding` | `True` | Embeddings multiplied by `sqrt(d_model)` |
| `tie_word_embeddings` | `False` | If true, encoder embedding, decoder embedding, and decoder output projection must alias |
| `use_cache` | `True` | Encoder-decoder `DynamicCache` for generation |
| `pad/bos/eos` | `1 / 0 / 2` | FSMT generation starts from EOS/decoder-start token id 2 |

Representative checkpoint sweep from downloaded `config.json`:

| Checkpoint | Langs | d_model | Enc/Dec layers | Heads | Enc FFN | Dec FFN | Src vocab | Tgt vocab | Max pos | Tie |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `facebook/wmt19-en-ru` | en-ru | 1024 | 6 / 6 | 16 / 16 | 8192 | 4096 | 31640 | 31232 | 1024 | false |
| `facebook/wmt19-ru-en` | ru-en | 1024 | 6 / 6 | 16 / 16 | 8192 | 4096 | 31232 | 31640 | 1024 | false |
| `facebook/wmt19-en-de` | en-de | 1024 | 6 / 6 | 16 / 16 | 8192 | 4096 | 42024 | 42024 | 1024 | false |
| `facebook/wmt19-de-en` | de-en | 1024 | 6 / 6 | 16 / 16 | 8192 | 4096 | 42024 | 42024 | 1024 | false |

Generation metadata from downloaded `generation_config.json`: `max_length=200`, `num_beams=5`, `decoder_start_token_id=2`; length penalty is 1.15 for en-ru, 1.10 for ru-en/de-en, omitted/effective default for en-de.

## 3a. Family variation traps

- Dual vocab is required. Source and target vocab sizes can differ, and tokenizer `get_vocab()` intentionally returns source vocab while decode uses target vocab.
- WMT19 configs use 6 encoder/decoder layers and `encoder_ffn_dim=8192`, unlike source defaults of 12 layers and 4096 encoder FFN.
- Encoder and decoder embeddings are separate unless `tie_word_embeddings=True`; when tying is enabled, source tests require encoder embedding, decoder embedding, and decoder output-projection weight identity.
- Decoder `output_projection` is inside `FSMTDecoder`, so the base `FSMTModel` already returns vocab logits, not decoder hidden states.
- `get_output_embeddings()` returns the decoder embedding, not `output_projection`; loading/lowering must use actual module keys.
- Position encoding is fairseq sinusoidal with pad-aware positions starting at `padding_idx + 1`, not BART's learned positions.
- The modeling code uses time-major `[T, B, C]` internally for blocks and attention. Initial lowering can preserve this with transposes; layout elimination needs guarded consumers.
- Decoder generation path sets `decoder_padding_mask=None` and `causal_mask=None` when `use_cache=True`; self-attention relies on cached one-token decode rather than an explicit triangular mask.
- Official tokenizer configs only store `langs`, `model_max_length=1024`, and `do_lower_case`; tokenization requires sidecar `vocab-src.json`, `vocab-tgt.json`, and `merges.txt`.
- FSMT does not support assisted decoding in Transformers tests and has skipped common tests around resizing embeddings and some `inputs_embeds` paths.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup for source `[B, S_src] -> [B, S_src, 1024]` and target `[B, S_tgt] -> [B, S_tgt, 1024]`.
- Scalar multiply by `sqrt(d_model)`.
- Add positional embeddings.
- `transpose(0, 1)` between batch-major and time-major around encoder/decoder bodies.
- View/reshape/permute for attention: `[T, B, C] -> [B, H, T, D] -> [B*H, T, D]`.
- `contiguous().view(...)` after attention output transpose.
- `gather` and `cumsum` for `shift_tokens_right` and sinusoidal position ids.
- Boolean mask creation, inversion, unsqueeze, masked fill, and optional `None` mask behavior.

Neural network primitives:

- Dense GEMM/linear with bias for Q/K/V/out projections and FFN layers.
- Dense GEMM/linear without bias for `decoder.output_projection`.
- LayerNorm over hidden dimension after residual adds.
- ReLU activation for official checkpoints.
- Dropout and LayerDrop are training-only for inference, but dropout nodes must be disabled in eval.
- Residual add patterns after self-attention, cross-attention, and MLP.

Attention primitives:

- Encoder noncausal MHA self-attention.
- Decoder causal MHA self-attention for full-prefix/teacher forcing.
- Decoder cross-attention over encoder states.
- Decode cache update/reuse for self-attention and cross-attention.
- Softmax over `src_len`, with additive causal mask and key padding mask before softmax.
- BMM attention score/value matmuls; FlashAttention/SDPA rewrite is an optimization, not source behavior.

Position ops:

- Fairseq sinusoidal embedding construction and pad-aware `make_positions`.
- Optional table extension when sequence length exceeds current table.

Generation/cache ops:

- Encoder-decoder `EncoderDecoderCache(DynamicCache, DynamicCache)`.
- Per-layer self-attention cache stores keys/values shaped `[B, H, T_dec, D]`.
- Per-layer cross-attention cache stores projected encoder keys/values shaped `[B, H, S_src, D]` and uses `is_updated[layer_idx]`.
- Beam reorder is handled by generic cache `reorder_cache(beam_idx)` in current generation utils; local `_reorder_buffer` exists but is not wired by this class.

Preprocessing-coupled ops:

- Moses normalization/tokenization and detokenization are CPU/data-pipeline.
- fastBPE merge loop and source/target vocab lookup are tokenizer-owned.
- Special-token ABI appends `</s>` only; no BOS is prepended by tokenizer.

## 5. Layer/block breakdown

Encoder setup:

```text
input_ids [B,S] -> src_embed [B,S,C] * sqrt(C)
positions = SinusoidalPosition(input_ids) [B,S,C]
x = src_embed + positions
x = transpose to [S,B,C]
```

Encoder block, repeated `encoder_layers` times:

```text
residual = x
attn = MHA_self(query=x, key=x, padding_mask=source_pad_mask)  # [S,B,C]
x = LayerNorm(residual + attn)
residual = x
x = Linear(C -> encoder_ffn_dim, bias=True)
x = ReLU(x)
x = Linear(encoder_ffn_dim -> C, bias=True)
x = LayerNorm(residual + x)
```

Decoder setup:

```text
decoder_input_ids [B,T] -> positions [B,T,C]
if use_cache: keep only input_ids[:, -1:] and positions[:, -1:]
x = tgt_embed [B,T_or_1,C] * sqrt(C) + positions
x = transpose to [T_or_1,B,C]
encoder_hidden_states = transpose encoder output to [S,B,C]
```

Decoder block, repeated `decoder_layers` times:

```text
residual = x
self_attn = MHA_self(query=x, key=x, causal_mask, decoder_padding_mask, self KV cache)
x = LayerNorm(residual + self_attn)
residual = x
cross_attn = MHA_cross(query=x, key=encoder_hidden_states, encoder_padding_mask, cross KV cache)
x = LayerNorm(residual + cross_attn)
residual = x
x = Linear(C -> decoder_ffn_dim, bias=True)
x = ReLU(x)
x = Linear(decoder_ffn_dim -> C, bias=True)
x = LayerNorm(residual + x)
```

Decoder output:

```text
x [T,B,C] -> transpose [B,T,C]
logits = Linear(C -> tgt_vocab_size, bias=False)
```

## 6. Attention requirements

FSMT uses dense MHA only. There is no RoPE, ALiBi, GQA/MQA, sparse/local/block attention, or packed varlen attention in source.

Encoder self-attention:

- Noncausal, self-attention.
- Query/key/value width `d_model`; heads = `encoder_attention_heads`; `head_dim=d_model/heads`.
- Source padding mask shape `[B, S_src]`, with `True` meaning mask after source inversion.
- Runtime source path uses manual `bmm`: `(B*H,T,D) @ (B*H,D,S)`.

Decoder self-attention:

- Causal for full-prefix/teacher forcing via additive mask `[T,T]` filled with dtype minimum above diagonal.
- Uses decoder padding mask for pad tokens when not using cache.
- In generation with cache, decoder slices input ids and positions to one token and passes no causal/padding mask.
- Cache stores projected K/V after head split as `[B,H,T_cache,D]`; cached keys include no position transform because positions are applied before projection.

Decoder cross-attention:

- Query from decoder hidden states; key/value from encoder hidden states.
- Cross-attention K/V are cached after first generation step per layer and reused when `EncoderDecoderCache.is_updated[layer_idx]` is true.
- Encoder padding mask shape `[B,S_src]`.

Flash/SDPA compatibility:

- A fused attention rewrite is straightforward for encoder and full-prefix decoder when masks are canonicalized to additive/boolean forms.
- Decode cross-attention cache reuse must preserve separate self and cross cache ownership.
- Source scales query before score matmul: `q_proj(query) * head_dim**-0.5`; parity tests should preserve scale placement or prove equivalence.

## 7. Position encoding and custom math

FSMT's position encoding is source-specific fairseq sinusoidal embedding with padding ignored. It can be precomputed for max admitted sequence length and dtype/device, but source may extend the table dynamically if a longer sequence appears.

```python
def fsmt_make_positions(input_ids, padding_idx):
    mask = (input_ids != padding_idx).int()
    return (cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

def fsmt_sinusoidal_table(num_embeddings, embedding_dim, padding_idx):
    half = embedding_dim // 2
    inv = exp(arange(half).float() * -(log(10000) / (half - 1)))
    angles = arange(num_embeddings).float()[:, None] * inv[None, :]
    table = concat([sin(angles), cos(angles)], dim=1).reshape(num_embeddings, -1)
    if embedding_dim % 2:
        table = concat([table, zeros(num_embeddings, 1)], dim=1)
    table[padding_idx, :] = 0
    return table
```

Decoder input construction for no-cache path:

```python
def fsmt_shift_tokens_right(input_ids, pad_token_id):
    input_ids = input_ids.masked_fill(input_ids == -100, pad_token_id)
    eos_index = (input_ids != pad_token_id).sum(dim=1) - 1
    out = input_ids.clone()
    out[:, 0] = gather(input_ids, dim=1, index=eos_index[:, None]).squeeze(1)
    out[:, 1:] = input_ids[:, :-1]
    return out
```

## 8. Preprocessing and input packing

Tokenizer ABI:

- `FSMTTokenizer` requires `langs=[src_lang,tgt_lang]`, `vocab-src.json`, `vocab-tgt.json`, and `merges.txt`.
- Source text path: replace unicode punctuation -> Moses punctuation normalization -> remove non-printing chars -> Moses tokenize with `aggressive_dash_splits=True`, `escape=True` -> fastBPE -> source vocab lookup.
- `do_lower_case` exists in tokenizer config and is false/omitted for official WMT19 configs.
- Input special tokens: tokenizer appends `</s>` (`eos/sep` id 2) and does not prepend BOS.
- Model input names are `input_ids` and `attention_mask`; attention mask uses 1/true for real tokens before modeling code inverts it.
- Output decode path maps target ids through `vocab-tgt.json`, strips BPE `</w>`, then Moses detokenizes using target language.

Generation-controller ABI:

- `decoder_start_token_id=2` and EOS id 2 are required for generation parity.
- `forced_eos_token_id=2` comes from source defaults; generation configs include EOS/pad/bos and length penalties.
- Language control is not a forced BOS language-token table. It is model/checkpoint selection plus tokenizer `langs`.
- Beam-search behavior has known fairseq discrepancy; source comments say `early_stopping=True` is needed to better match fairseq, while official generation configs leave it omitted/effective default.

CPU/data-pipeline versus GPU/runtime:

- Moses/fastBPE/detokenization should stay outside DinoML first GPU graph.
- GPU runtime starts at integer `input_ids`, `attention_mask`, optional `decoder_input_ids`, optional `encoder_outputs`, and cache state.

## 9. Graph rewrite / lowering opportunities

### Rewrite: time-major attention to batch-major fused attention

Source pattern:

```text
[T,B,C] -> q/k/v Linear -> view/permute -> BMM scores -> masks -> softmax -> BMM values -> transpose/view -> out_proj
```

Replacement:

```text
BatchMajorQKV -> DenseMHA/SDPA -> OutputLinear
```

Preconditions:

- `d_model % num_heads == 0`.
- Q/K/V/out projection weights are standard `nn.Linear` layout `[out_features, in_features]`.
- Mask semantics are preserved: causal additive mask and key padding mask both apply before softmax.
- No output attentions requested, or attention weights are materialized by fallback/debug path.

Shape equations:

- Input `[B,T,C]`, heads `H`, head dim `D=C/H`.
- Scores `[B,H,T_q,T_k]`; output `[B,T_q,C]`.

Failure cases:

- `output_attentions=True` if fused backend cannot return exact attention weights.
- Mixed full-prefix and decode paths without cache manifest separation.

Parity sketch:

- Compare one encoder layer, one decoder self-attn layer, and one decoder cross-attn layer against Transformers in fp32 with padding and no padding.

### Rewrite: decoder output projection as logits GEMM

Source pattern:

```text
decoder hidden [B,T,C] -> Linear(C -> tgt_vocab_size, bias=False)
```

Replacement:

```text
GEMM_RCR hidden x output_projection.weight -> logits
```

Preconditions:

- Weight loaded from `model.decoder.output_projection.weight`.
- Respect aliasing when `tie_word_embeddings=True`; do not clone a tied target embedding/output-projection parameter as independent mutable storage.
- For last-token-only decode, `T=1` or slice last hidden before GEMM.

Failure cases:

- Full logits required for teacher-forcing loss versus decode-only last token.

Parity sketch:

- Compare full-prefix logits and one-step cached logits with tied and untied configs.

### Rewrite: fairseq sinusoidal positions to constant table + gather

Source pattern:

```text
make_positions(input_ids,pad) -> embedding lookup in deterministic sinusoidal table
```

Replacement:

```text
PadAwarePositionIds -> Gather(precomputed_sinusoidal_table)
```

Preconditions:

- Admit max sequence length no larger than precomputed table, or include a guarded table-extension fallback.
- Position ids start at `padding_idx + 1`; pad rows map to zero row.

Failure cases:

- `inputs_embeds` path creates position ids from zero-valued first hidden channel; defer or guard out initially.

Parity sketch:

- Test left/right/internal padding patterns and sequences at max length.

### Rewrite: QKV projection packing

Source pattern:

```text
q_proj, k_proj, v_proj are three separate Linear(C -> C, bias=True)
```

Replacement:

```text
single packed Linear(C -> 3C) followed by split [q,k,v]
```

Preconditions:

- Pack rows in source order `[q_proj; k_proj; v_proj]`; pack biases in the same order.
- For cross-attention, query uses decoder states while key/value use encoder states, so only K/V can be packed together unless backend supports separate Q input.

Failure cases:

- Cross-attention cannot use a single QKV GEMM over one input tensor.

Parity sketch:

- Weight-pack/unpack identity tests and one-layer attention parity.

### Layout guard: preserve tokenizer/sequence axes

No image/channel layout work exists. The only layout-sensitive region is sequence/batch/hidden order. A layout pass may remove transposes inside a closed encoder or decoder block, but public ABI tensors remain `[B,S,C]` for hidden states and `[B,S]` for masks/token ids.

## 10. Kernel fusion candidates

Highest priority:

- Dense MHA/SDPA for encoder self-attention, decoder self-attention, and cross-attention. Attention is the main non-GEMM runtime cost.
- LayerNorm + residual add patterns around attention/MLP. FSMT uses post-norm blocks repeatedly.
- GEMM epilogues for FFN ReLU and output projection. Official WMT19 encoder FFN is large: `1024 -> 8192 -> 1024`.
- Last-token-only logits GEMM for decode to avoid `[B,T,V]` work when only next-token logits are consumed.

Medium priority:

- Packed QKV for encoder and decoder self-attention.
- Packed KV for cross-attention plus cross KV cache materialization after encoder.
- Precomputed sinusoidal table + gather fused with embedding add/scale.
- Beam-aware cache reorder and cache duplication without host round trips.

Lower priority:

- Training loss and label shifting.
- `inputs_embeds` path and dynamic sinusoidal table extension beyond admitted max length.
- Attention-weight outputs for introspection.

## 11. Runtime staging plan

1. Parse FSMT config and tokenizer metadata; reject unsupported remote-code or non-FSMT configs.
2. Load untied WMT19 weights with separate source embedding, target embedding, and output projection; preserve optional tie aliases.
3. Implement one encoder block and one decoder block parity with fp32 random weights.
4. Implement full encoder-only parity: `input_ids`, source mask, fairseq sinusoidal positions, no cache.
5. Implement full teacher-forcing decoder parity with explicit decoder inputs/masks and logits.
6. Add generation prefill/decode ABI: encoder output cache, self KV cache, cross KV cache, one-token decode, beam reorder.
7. Enable attention/GEMM fusions with guarded fallback for attention-weight outputs and unsupported `inputs_embeds`.
8. Add tokenizer/generation-controller integration for end-to-end WMT19 translation smoke tests.

Initially stub/defer: training loss, output hidden states/attentions, `inputs_embeds`, model ensemble, assisted decoding, fairseq-exact beam-search quirks.

## 12. Parity and validation plan

- Config parsing tests for all four official WMT19 configs and source-default synthetic config.
- Tokenizer metadata tests: `langs`, source/target vocab sizes, special-token append, decoder start/eos ids.
- Custom op tests: `shift_tokens_right`, `make_padding_mask`, `invert_mask`, causal mask construction, fairseq sinusoidal `make_positions`.
- Single attention tests for encoder self-attn, decoder self-attn with causal mask, and decoder cross-attn with source padding mask.
- Single block parity in fp32 for encoder and decoder, dropout disabled.
- Full encoder parity on padded and unpadded batches.
- Full teacher-forcing logits parity for WMT19-sized synthetic weights and tiny synthetic configs.
- Cached decode parity: prefill then one token, compare logits and cache shapes to Transformers.
- Beam reorder parity using generic cache reorder for self and cross caches.
- End-to-end smoke: official `facebook/wmt19-ru-en` short sentence through tokenizer, model, generate, detokenizer.

Recommended tolerances: fp32 `atol=1e-4, rtol=1e-4` for full logits after fused attention; fp16 `atol=5e-3, rtol=5e-3` for block/logit parity, with stricter checks for integer/tokenizer utilities.

## 13. Performance probes

- Tokenizer throughput by language pair and sentence length, measured separately from GPU runtime.
- Encoder throughput sweep over `B` and `S_src` up to 1024.
- Teacher-forcing decoder throughput sweep over `B`, `S_src`, and `S_tgt`.
- Decode tokens/sec with cached self/cross KV for beam sizes 1, 5, and 50.
- Cross-attention cache materialization cost and memory footprint.
- Last-token logits GEMM versus full-prefix logits GEMM.
- Encoder FFN GEMM benchmark for `1024x8192` and decoder FFN for `1024x4096`.
- Attention backend comparison: manual BMM, SDPA/FlashAttention, and cache-aware decode kernels.
- Cache memory usage: `2 caches * layers * B * H * T * D * dtype`, separating self growing cache from fixed cross cache.
- Weight-load/projection alias check for tied versus untied configs.

## 14. Skip/defer list

- Training loss and gradient behavior.
- Dropout/LayerDrop stochastic training behavior.
- Output hidden states and attentions for optimized path.
- `inputs_embeds` path, especially zero-first-channel position inference.
- Embedding resize APIs; Transformers tests skip several FSMT resize paths.
- Assisted/speculative decoding; Transformers tests explicitly skip it for FSMT.
- Fairseq model ensemble support; source comments say ensemble was not ported.
- Fairseq-exact beam-search behavior beyond generation-config controls.
- Any NHWC/channel-last layout work; not applicable.
- Quantized or packed weight formats; none are source-coupled in FSMT.

## 15. Final implementation checklist

- [ ] Parse `FSMTConfig` including dual vocab sizes, language pair, generation defaults, and source-default fallbacks.
- [ ] Load `FSMTTokenizer` metadata and keep Moses/fastBPE in CPU pipeline.
- [ ] Load separate encoder embedding, decoder embedding, and decoder output projection; preserve optional tied aliases.
- [ ] Implement fairseq sinusoidal position ids and table gather.
- [ ] Implement `shift_tokens_right`, padding mask inversion, and causal mask construction.
- [ ] Lower encoder self-attention MHA with padding mask.
- [ ] Lower decoder causal self-attention MHA with full-prefix mask.
- [ ] Lower decoder cross-attention MHA with encoder padding mask.
- [ ] Implement encoder-decoder cache manifest: self KV, cross KV, `is_updated`, beam reorder.
- [ ] Implement LayerNorm, residual add, ReLU FFN, and output-projection GEMM parity.
- [ ] Add guarded packed QKV/KV rewrites.
- [ ] Add last-token-only logits rewrite for decode.
- [ ] Add fp32 one-block, encoder, decoder, and cached-decode parity tests.
- [ ] Add WMT19 tokenizer/generation smoke parity.
- [ ] Benchmark encoder, prefill, decode, logits, cache memory, and tokenizer throughput separately.

## Gated gaps for DinoML admission

- Seq2seq cache ABI is mandatory for useful generation: DinoML needs separate self-attention growing KV and cross-attention fixed KV ownership per decoder layer.
- Tokenizer/language metadata is model-coupled for end-to-end parity: source/target vocab files, `langs`, special-token layout, and decoder start/EOS ids must be accepted as ABI inputs or owned by a CPU preprocessor.
- FSMT base output is logits, not hidden states, because `output_projection` lives inside `FSMTDecoder`; graph import must not add a second LM head.
- Layout rewrites must guard the internal `[T,B,C]` convention and keep public hidden-state/mask ABI batch-major.
- Official configs differ from source defaults in layer count and encoder FFN width; config-driven shape extraction is required before weight admission.
- Optional tied embeddings are an aliasing contract, not just equal initial values.
