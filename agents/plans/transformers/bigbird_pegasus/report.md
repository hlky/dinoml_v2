# BigBird-Pegasus Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/bigbird-pegasus-large-arxiv primary; google/bigbird-pegasus-large-pubmed and google/bigbird-pegasus-large-bigpatent representative public configs
Config source: Hugging Face config.json/tokenizer_config.json/generation_config.json snapshots saved under agents/plans/transformers/bigbird_pegasus/_sources/
Source files inspected:
- X:/H/transformers/src/transformers/models/bigbird_pegasus/configuration_bigbird_pegasus.py
- X:/H/transformers/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py
- X:/H/transformers/src/transformers/models/pegasus/tokenization_pegasus.py
Any missing files or assumptions:
- google/bigbird-pegasus-large-wikihow returned 401 for config/tokenizer/generation config fetches; it is recorded as unavailable and not used for operator facts.
- This report scopes the first useful DinoML runtime target to `BigBirdPegasusForConditionalGeneration`: long-document encoder-decoder summarization with block-sparse encoder attention, dense decoder self-attention/cross-attention, and decoder KV/cache support. Sequence classification, QA, and decoder-only CausalLM heads are implemented in source but are optional/deferred for that target.
- No DinoML tests or imports were run; this is a source/config audit only.
```

Primary source URLs:

- `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bigbird_pegasus`
- `https://huggingface.co/google/bigbird-pegasus-large-arxiv/blob/main/config.json`
- `https://huggingface.co/google/bigbird-pegasus-large-pubmed/blob/main/config.json`
- `https://huggingface.co/google/bigbird-pegasus-large-bigpatent/blob/main/config.json`
- `https://huggingface.co/google/bigbird-pegasus-large-wikihow` gated/unavailable during audit

## 2. High-level architecture

BigBird-Pegasus is a text-only encoder-decoder summarization model. The encoder is a Pegasus-style pre-norm stack whose self-attention can be `block_sparse` or `original_full`; the decoder is BART/Pegasus-style dense causal self-attention plus dense encoder-decoder cross-attention.

```text
Pegasus tokenization + EOS packing -> shared scaled token embeddings + learned positions
-> block-sparse/full encoder -> dense causal decoder prefill/decode with cross-attention
-> tied LM head + final logits bias -> generation controller
```

Stage decomposition:

- CPU/data pipeline: Pegasus tokenizer, truncation to 4096 source tokens, padding masks, right-shifted decoder inputs when labels or no decoder inputs are supplied.
- Encoder runtime: shared token embedding scaled by `sqrt(d_model)`, learned absolute positions, dropout disabled for inference, N pre-norm encoder blocks, final LayerNorm, optional block padding/unpadding.
- Decoder prefill/runtime: shared token embedding, learned positions offset by cache length, causal self-attention mask, optional cross-attention mask, N pre-norm decoder blocks, final LayerNorm.
- Decode cache: decoder self-attention KV grows with generated length; cross-attention KV is computed from encoder hidden states once per layer and reused through `EncoderDecoderCache`.
- Independently cacheable: encoder outputs and cross-attention KV projections can be cached across decode steps for the same source document.

## 3. Important config dimensions

Source defaults from `BigBirdPegasusConfig`:

| field | default | runtime effect |
|---|---:|---|
| `vocab_size` | 96103 | shared embedding and LM head width |
| `d_model` / `hidden_size` | 1024 | encoder/decoder hidden width |
| `encoder_layers` | 16 | encoder block count |
| `decoder_layers` | 16 | decoder block count |
| `encoder_attention_heads` | 16 | encoder MHA heads |
| `decoder_attention_heads` | 16 | decoder MHA heads |
| `head_dim` | 64 | inferred as `d_model / heads` |
| `encoder_ffn_dim` | 4096 | encoder FFN expansion |
| `decoder_ffn_dim` | 4096 | decoder FFN expansion |
| `max_position_embeddings` | 4096 | learned source/target position table length and sparse planner max |
| `activation_function` | `gelu_new` | FFN activation |
| `scale_embedding` | true | token embeddings multiplied by `sqrt(d_model)` |
| `attention_type` | `block_sparse` | encoder attention only: sparse or dense fallback |
| `block_size` | 64 | sparse block granularity and padding multiple |
| `num_random_blocks` | 3 | sparse random blocks per query block |
| `use_bias` | false | attention Q/K/V/out projections omit bias where config is honored; FFN linears still use PyTorch default bias |
| `use_cache` | true | decoder cache enabled by default |
| `decoder_start_token_id` | 2 | right-shift start token |
| `pad/eos/bos` | 0 / 1 / 2 | tokenizer/generation ABI |

Representative checkpoint sweep:

| model id | architecture | layers enc/dec | d_model | heads enc/dec | FFN enc/dec | max pos | attention | block/random | bias | generation defaults |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| `google/bigbird-pegasus-large-arxiv` | `BigBirdPegasusForConditionalGeneration` | 16 / 16 | 1024 | 16 / 16 | 4096 / 4096 | 4096 | block sparse encoder | 64 / 3 | false | beams 5, length penalty 0.8, max length 256 |
| `google/bigbird-pegasus-large-pubmed` | same | 16 / 16 | 1024 | 16 / 16 | 4096 / 4096 | 4096 | block sparse encoder | 64 / 3 | false | same |
| `google/bigbird-pegasus-large-bigpatent` | same | 16 / 16 | 1024 | 16 / 16 | 4096 / 4096 | 4096 | block sparse encoder | 64 / 3 | false | same |

Checkpoint facts above come from fetched `config.json` and `generation_config.json`; `head_dim=64` is inferred from source validation. The three accessible official configs do not expose operator-significant variation.

## 3a. Family variation traps

- `attention_type` applies only to the encoder. The decoder always uses dense attention through `BigBirdPegasusDecoderAttention`.
- Encoder `block_sparse` mutates to `original_full` at runtime when `source_length <= (5 + 2 * num_random_blocks) * block_size`. With defaults this threshold is `704`, so lengths `<=704` use full attention even if the config says sparse.
- Sparse encoder inputs are padded internally to a multiple of `block_size`; hidden states are unpadded after the final encoder LayerNorm.
- Sparse attention has fixed first and last global blocks, fixed one-block-left/current/one-block-right local window, and no ETC/extra-global-token support.
- Random plans are deterministic from source. Each encoder layer is constructed with `seed=layer_index`; sparse forward calls `np.random.seed(seed)` before plan generation. In eval mode, both old and new random-plan helpers return all-zero random block indices rather than sampling. DinoML must match this if validating against HF eval outputs.
- The sparse path always allocates/reconstructs dense `attention_probs [B,H,S,S]` after computing context, even though hidden states do not depend on it. This is a major optimization guard: skip only when attentions are not requested.
- `use_bias=false` for official checkpoints, but that flag gates attention projections/output only. FFN linears, classifier head, QA head, and `final_logits_bias` still have bias-like parameters. QKV packing must guard on attention bias presence; FFN fusion must handle real FFN bias.
- The config class uses `attribute_map` aliases: `hidden_size -> d_model`, `num_attention_heads -> encoder_attention_heads`, `num_hidden_layers -> encoder_layers`, `attention_probs_dropout_prob -> attention_dropout`.
- Tokenizer config snapshots set `tokenizer_class="PegasusTokenizer"` and `offset=0`, while current source defaults for Pegasus tokenizer include sentinel pretraining tokens with `offset=103`. Use checkpoint tokenizer config for end-to-end tokenization parity.
- Sequence classification pools the last EOS token and requires each batch row to have the same number of EOS tokens. That head is optional for the summarization target but needs gather/boolean-mask guards if admitted.
- `BigBirdPegasusForCausalLM` mutates config to decoder-only (`is_decoder=True`, `is_encoder_decoder=False`) and uses `logits_to_keep`; treat as a separate dense-decoder audit target.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding index lookup for `input_ids [B,S]`, `decoder_input_ids [B,T]`.
- `arange` for learned position ids with cache offset.
- Reshape/view and transpose for Q/K/V: `[B,S,H] -> [B,heads,S,D]`.
- `contiguous`, slicing, split, squeeze/unsqueeze.
- `cat` for sparse key/value regions and block padding.
- `pad` for attention mask and hidden/token embeddings to `block_size`.
- `index_select`/gather for random sparse block K/V.
- Boolean/equality mask and `unique_consecutive` only for optional sequence classification EOS pooling.

Neural network primitives:

- Shared scaled token embedding: `[96103,1024]`, scale `sqrt(1024)` when `scale_embedding=true`.
- Learned positional embedding: `[4096,1024]`.
- LayerNorm over hidden dim, PyTorch default epsilon unless overridden by module defaults.
- Attention projections/output, official bias-free: Q/K/V/out `Linear(1024 -> 1024, bias=false)`.
- FFN linears with bias: `Linear(1024 -> 4096, bias=true) -> gelu_new -> Linear(4096 -> 1024, bias=true)`.
- Residual adds around attention and FFN.
- LM head `Linear(1024 -> 96103, bias=false)` tied to shared embeddings plus `final_logits_bias [1,96103]`.
- Optional heads: classification `Linear(1024 -> 1024) -> tanh -> Linear(1024 -> num_labels)`; QA `Linear(1024 -> 2)`.

Attention primitives:

- Encoder full self-attention: noncausal dense MHA, additive bidirectional mask.
- Encoder block-sparse self-attention: global first/last blocks, local three-block band, random block gathers, per-region softmax, region value matmuls, dense attention reconstruction if requested.
- Decoder self-attention: causal dense MHA with `DynamicCache`.
- Decoder cross-attention: dense MHA over encoder states with cross-attention cache stored separately in `EncoderDecoderCache`.
- Decoder attention backend dispatch: `ALL_ATTENTION_FUNCTIONS` can route by `config._attn_implementation`; source eager path uses matmul/softmax/dropout/matmul.

Position/relative-bias ops:

- Learned absolute positions only. No RoPE, ALiBi, relative bias, MQA/GQA, MoE, or sliding-window config knob.

Generation/cache ops:

- `shift_tokens_right`: prepend `decoder_start_token_id`, shift labels/input right, replace `-100` with pad.
- `EncoderDecoderCache(DynamicCache, DynamicCache)` for seq2seq generation.
- Per-layer self KV cache `[B, decoder_heads, cached_T, head_dim]`.
- Per-layer cross KV cache `[B, decoder_heads, source_S, head_dim]`, updated once then reused via `is_updated[layer_idx]`.
- `logits_to_keep` only in decoder-only CausalLM, not the seq2seq LM head.

Preprocessing-coupled ops:

- Pegasus tokenizer: Unigram/SentencePiece-like model, metaspace replacement, EOS appended for single and pair sequences.
- Generation config: beam search, max length, length penalty are generation-controller behavior, not neural graph ops.

Aliasing/tied weights:

- `model.shared.weight`, encoder embeddings, decoder embeddings, and seq2seq `lm_head.weight` are one logical tied parameter. Lowering must preserve aliasing rather than clone divergent weights.

## 5. Layer/block breakdown

Encoder embedding:

```text
token = shared_embedding(input_ids) * sqrt(d_model)
pos = learned_position_embedding(arange(0, S))
x = dropout(token + pos)
```

Encoder block, repeated 16 times:

```text
res = x
x = LayerNorm(x)
q,k,v = Linear(1024 -> 1024, bias=use_bias)(x) split as [B,16,S,64]
attn = original_full_or_block_sparse(q,k,v,masks)
x = res + dropout(Linear(1024 -> 1024, bias=use_bias)(attn))
res = x
x = LayerNorm(x)
x = gelu_new(Linear(1024 -> 4096, bias=true)(x))
x = Linear(4096 -> 1024, bias=true)(x)
x = res + dropout(x)
```

After all encoder blocks:

```text
encoder_hidden = LayerNorm(x)
encoder_hidden = encoder_hidden[:, :-padding_len] if sparse padding was added
```

Decoder block, repeated 16 times:

```text
res = y
y = LayerNorm(y)
q = q_proj(y); k,v = self k/v from y or cache update
self_attn = dense causal attention(q,k,v,cache,mask)
y = res + dropout(out_proj(self_attn))
res = y
y = LayerNorm(y)
q = q_proj(y); k,v = encoder projections or cross cache
cross = dense attention(q,k,v,encoder_mask)
y = res + dropout(out_proj(cross))
res = y
y = LayerNorm(y)
y = gelu_new(Linear(1024 -> 4096, bias=true)(y))
y = dropout(y)
y = Linear(4096 -> 1024, bias=true)(y)
y = res + dropout(y)
```

Seq2seq LM head:

```text
decoder_hidden = final_decoder_layernorm(y)
logits = decoder_hidden @ shared_embedding.T + final_logits_bias
```

## 6. Attention requirements

Encoder full attention:

- Noncausal self-attention.
- MHA: 16 query heads, 16 key/value heads, `head_dim=64`.
- Query/key/value width all equal `1024`; no GQA/MQA.
- Scores: `Q @ K^T / sqrt(64)`, additive mask, softmax on key dim, dropout, `P @ V`.
- Mask is created by `create_bidirectional_mask`; represent it as `[B,1,Q,K]` additive/padding mask.

Encoder block-sparse attention:

- Noncausal self-attention only.
- Admission guard: source length after internal padding must be divisible by `block_size`; source length before padding must be `>704` for default sparse dispatch.
- Mask tensors:
  - `blocked_encoder_mask [B, S/block, block]`
  - `band_mask [B, 1, S/block - 4, block, 3*block]`
  - `from_mask [B,1,S,1]`
  - `to_mask [B,1,1,S]`
  - `rand_mask [B,H,S/block - 2, block, num_random_blocks*block]`
- First and last query blocks attend all key tokens.
- Second query block attends first three key blocks, last key block, and random blocks.
- Middle query blocks attend first global block, three local blocks, random blocks, and last global block.
- Second-last query block attends first key block, last three key blocks, and random blocks.
- Mask penalty is the literal `-10000.0`, not dtype min.
- Dense attention reconstruction is separate from the hidden-state fast path and should be guarded by `output_attentions`.

Decoder self-attention:

- Causal dense self-attention with optional cache.
- Prefill Q length `T`, K/V length `T`; decode Q length usually 1, K/V length `past_T + 1`.
- Cache stores keys/values after projection/reshape, with learned position already added to hidden states before projection.
- Attention implementation can use eager, SDPA, or other `ALL_ATTENTION_FUNCTIONS` backend if masks and dtype are supported.

Decoder cross-attention:

- Dense rectangular attention: Q length is decoder length, K/V length is encoder source length.
- Cross KV cache is separate from self KV and is marked updated after first projection per layer.
- Encoder outputs, not sparse attention internals, are the cross-attention source.

FlashAttention/SDPA compatibility:

- Decoder dense attention and encoder `original_full` are SDPA/FlashAttention candidates.
- Encoder `block_sparse` is not directly expressible as standard dense FlashAttention; it needs a BigBird sparse kernel, a dense fallback, or a source-equivalent gather/BMM lowering.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding:

```python
def learned_positions(seq_len, past_len=0):
    position_ids = arange(past_len, past_len + seq_len)
    return position_embedding[position_ids]
```

Sparse random-plan behavior:

```python
def hf_sparse_plan(seq_len, block, n_rand, n_heads, layer_seed, training):
    np.random.seed(layer_seed)
    if seq_len in (1024, 3072, 4096):
        plan = old_bigbird_plan(max_position_embeddings=4096, last_idx=1024)
    else:
        plan_from_length, plan_n_rand = get_rand_attn_plan(seq_len, block, n_rand)
        plan = per_head_plan(seq_len, block, n_heads, plan_from_length, plan_n_rand)
    if not training:
        return zeros_like(plan)
    return plan
```

Precomputable:

- Learned position table is a weight.
- Eval-mode sparse random plans are deterministic all-zero tensors for each admitted sequence shape.
- Training-mode plans are deterministic for a fixed layer seed and sequence shape because source resets NumPy seed on every sparse forward.

Dynamic inputs:

- Cache offset changes decoder position ids.
- Source sequence length controls sparse/full dispatch and padding length.
- Attention masks control padding visibility in all attention paths.

## 8. Preprocessing and input packing

Tokenizer contract:

- Checkpoint configs name `PegasusTokenizer`.
- Source tokenizer is tokenizers-backed Unigram with metaspace pre-tokenizer/decoder.
- Single sequence post-processing is `$A </s>`; pair is `$A $B </s>`.
- `model_input_names = ["input_ids", "attention_mask"]`.
- Public configs use `model_max_length=4096`.
- Special ids from model configs: pad 0, eos 1, bos/decoder start 2.

Seq2seq input packing:

- Encoder accepts `input_ids [B,S]` or `inputs_embeds [B,S,H]`; exactly one is required.
- If no `attention_mask`, source creates all-ones mask.
- If no decoder inputs are provided, `BigBirdPegasusModel.forward` right-shifts `input_ids`; `ForConditionalGeneration.forward` right-shifts `labels` when labels are present.
- `shift_tokens_right` moves tokens one step right, sets first token to `decoder_start_token_id`, and replaces shifted `-100` entries with pad.

CPU/data-pipeline work:

- Tokenization, truncation, generation beam/sampling policy, and text decode.

GPU/runtime work:

- Embeddings, masks, sparse block padding/mask construction, encoder/decoder blocks, LM logits.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Bias-free separate Q/K/V linears -> packed QKV

Source pattern:

```text
q = Linear(H,H,bias=use_bias)(x)
k = Linear(H,H,bias=use_bias)(x)
v = Linear(H,H,bias=use_bias)(x)
```

Replacement:

```text
qkv = Linear(H,3H,bias=use_bias)(x)
q,k,v = split(qkv, 3, last_dim)
```

Preconditions:

- Same input tensor and dtype.
- Same projection width and same bias policy.
- No per-projection quantization/residency differences.

Shape equations:

- Input `[B,S,1024]`, output packed `[B,S,3072]`, split to three `[B,S,1024]`.

Weight transform:

```python
w_qkv = cat([w_q, w_k, w_v], dim=0)
b_qkv = cat([b_q, b_k, b_v], dim=0) if use_bias else None
```

Failure cases:

- Debug hooks requiring separate tensors.
- Mixed bias or separate weight materialization policies.

Parity test sketch:

- Compare q/k/v tensors before reshape for one encoder layer and one decoder layer in fp32.

### Rewrite: Dense decoder attention -> SDPA/FlashAttention

Preconditions:

- Inference dropout disabled.
- Additive causal/cross masks are representable by backend.
- Cache layout `[B,H,L,D]` is supported.
- Requested attentions can be omitted or backend can return weights.

Replacement:

```text
QKV projection + cache update -> fused dense attention -> output projection
```

Failure cases:

- `output_attentions=True` on a backend that does not return weights.
- Cross-cache update/reuse not represented in cache manifest.

### Rewrite: Encoder sparse attention -> BigBirdSparseAttention op

Preconditions:

- `attention_type == "block_sparse"` after source dispatch guard.
- `S_padded % block_size == 0`.
- First/last global block only; local window exactly three blocks.
- Eval random plan matches all-zero HF behavior, or training mode admits fixed precomputed plans.
- `output_attentions=False` for fast path.

Replacement:

```text
BigBirdSparseAttention(Q,K,V, attention_mask, block_size, random_plan, return_attn=False)
```

Shape equations:

- Q/K/V `[B,H,S,64]`.
- Blocks `N = S / block_size`.
- Middle sparse score region `[B,H,N-4,block,(5+num_random_blocks)*block]`.

Failure cases:

- Decoder/cross-attention.
- `S <=704` default short sequence, which source routes to full attention.
- `output_attentions=True` without dense reconstruction fallback.
- Non-default block/window/global behavior not present in source.

Parity test sketch:

- Compare encoder hidden states for lengths 705, 768, 1024, 3072, 4096 and a non-multiple input length that pads internally.

### Rewrite: Skip dense sparse-attention probability reconstruction

Source pattern:

```text
attention_probs = zeros(B,H,S,S)
scatter visible sparse weights into attention_probs
```

Replacement:

```text
do not materialize attention_probs unless caller records encoder attentions
```

Preconditions:

- `output_attentions`/recorded attentions are disabled.
- Only hidden states/logits are consumed.

Failure cases:

- Debug/parity APIs that expose `encoder_attentions`.

### Rewrite: FFN activation GEMM fusion

Source pattern:

```text
Linear(1024,4096) -> gelu_new -> Linear(4096,1024)
```

Replacement:

```text
CUTLASS GEMM bias/activation epilogue where bias exists; otherwise GEMM + fused gelu_new elementwise
```

Preconditions:

- Static hidden/intermediate dims.
- Inference dropout disabled.
- Activation is exactly `gelu_new`.

Failure cases:

- Training dropout/layerdrop active.
- Activation override in custom configs.

## 10. Kernel fusion candidates

Highest priority:

- BigBird encoder block-sparse attention kernel. This is the model-defining long-context path and the main gap versus standard seq2seq Transformers.
- Dense decoder self-attention with KV cache and dense cross-attention with cross KV cache. Needed for generation throughput.
- LayerNorm plus residual-adjacent scheduling. Every encoder/decoder block has pre-attention/pre-FFN norms and final embedding norms.
- LM head/tied embedding GEMM, ideally with last-step-only decode logits when generation controller only needs the newest token.

Medium priority:

- Packed QKV projection for encoder and decoder.
- Sparse random-block gather and block-local score/value BMMs.
- FFN `gelu_new` fusion around GEMM.
- Encoder full-attention fallback through SDPA for short sequences.
- Cross-attention KV precompute for fixed encoder outputs.

Lower priority:

- Dense attention probability reconstruction for sparse attentions.
- Optional QA/classification head fusions.
- GPU tokenization.
- Training dropout, layerdrop, and losses.

## 11. Runtime staging plan

Stage 1: Parse config/tokenizer/generation metadata and load tied weights. Guard the official attention-bias-free large config first while still loading FFN biases.

Stage 2: Implement dense encoder/decoder primitives in `original_full` mode: embeddings, learned positions, LayerNorm, Linear, `gelu_new`, residuals, dense MHA, LM head.

Stage 3: Add seq2seq prefill without cache: encoder outputs feed decoder cross-attention; compare logits for short source lengths `<=704`, where HF routes encoder to full attention.

Stage 4: Add decoder `EncoderDecoderCache`: self KV growth, cross KV update-once/reuse, position-id cache offset.

Stage 5: Add encoder sparse dispatch guards, block padding/unpadding, sparse masks, and source-equivalent sparse reference lowering.

Stage 6: Replace sparse reference with a BigBird sparse attention provider. Keep dense attention reconstruction behind an `output_attentions` guard.

Stage 7: Add generation-controller parity metadata: right shift, decoder start/eos/pad ids, beam-search defaults outside the graph.

Stage 8: Optional heads: QA, sequence classification, and decoder-only CausalLM as separate admission targets.

Can stub initially: training losses, dropout/layerdrop, `output_attentions` for optimized sparse path, classifier/QA heads, GPU tokenizer, beam search internals.

## 12. Parity and validation plan

- Config/load tests:
  - Load source defaults plus arxiv/pubmed/bigpatent configs.
  - Verify `d_model == heads * head_dim` for encoder and decoder.
  - Verify tied alias: shared embedding, encoder embedding, decoder embedding, and LM head.
- Embedding tests:
  - Encoder and decoder position ids with and without cache offset.
  - `scale_embedding=true` versus a synthetic false config.
- Dense short-sequence parity:
  - Source lengths `128`, `512`, `704` should use encoder full attention.
  - Compare one block, full encoder, decoder prefill, and final logits.
- Decoder cache parity:
  - Prefill then one-token decode versus full recompute.
  - Cross-attention cache update once and reuse over multiple decode steps.
  - Batch reorder should be a follow-up if generation beam search is owned.
- Sparse encoder parity:
  - Lengths `705`, `768`, `1024`, `3072`, `4096`.
  - Non-multiple source length such as `769` to validate pad/unpad.
  - Eval random plans must be all-zero to match HF.
  - Validate with `output_attentions=False` first; separately test dense reconstruction if supported.
- End-to-end:
  - Arxiv summarization logits for a fixed tokenized input.
  - Decode first N greedy tokens before beam-search parity.
- Tolerances:
  - fp32 dense path: `rtol=1e-4`, `atol=1e-5`.
  - fp32 sparse path: start `rtol=2e-4`, `atol=2e-5` because fused region ordering may differ.
  - fp16/bf16: start `rtol=2e-2`, `atol=2e-2`, then tighten after attention math order is fixed.

## 13. Performance probes

- Encoder length sweep: `S=512`, `704`, `705`, `768`, `1024`, `2048`, `3072`, `4096`; separate dense fallback and sparse.
- Decoder prefill sweep: target lengths `T=16`, `64`, `128`, `256`.
- Decode tokens/sec with cached encoder/cross KV and batch sizes `1`, `2`, `4`, `8`.
- Sparse attention breakdown:
  - QKV projection time.
  - block padding/mask construction time.
  - random K/V gather time.
  - global/local/random score softmax time.
  - value aggregation time.
  - optional dense `attention_probs [B,H,S,S]` reconstruction cost.
- Dense attention backend comparison: eager versus SDPA/Flash for decoder self/cross attention.
- FFN GEMM throughput for `1024x4096x1024` blocks.
- LM head throughput: full sequence logits versus last-token-only logits for decode.
- Memory probes:
  - Encoder sparse intermediates and masks.
  - Decoder self KV: `layers * 2 * B * H * T * D`.
  - Cross KV: `layers * 2 * B * H * S * D`.
  - Dense attention reconstruction overhead.

## 14. Skip/defer list

- Training losses, dropout randomness, layerdrop, and gradient checkpointing.
- Beam search implementation inside compiled graph; keep generation controller outside first.
- `output_attentions=True` for optimized sparse attention, except explicit debug fallback.
- Sequence classification, QA, and decoder-only CausalLM heads for first summarization integration.
- GPU tokenizer and text decoding.
- Non-default sparse variants: ETC, extra global tokens, different local window widths.
- Quantization, GGUF ingestion, and multi-GPU/tensor parallel.
- Custom configs with attention `use_bias=true` or non-`gelu_new` activation until guarded parity exists.

## 15. Final implementation checklist

- [ ] Parse `BigBirdPegasusConfig`, generation config, and tokenizer ABI metadata.
- [ ] Preserve tied shared embedding / encoder embedding / decoder embedding / LM head aliasing.
- [ ] Implement scaled token embeddings and learned absolute positions with cache offset.
- [ ] Implement dense MHA for encoder fallback, decoder self-attention, and decoder cross-attention.
- [ ] Implement `EncoderDecoderCache` manifest: self KV plus cross KV with update-once flag.
- [ ] Implement right-shift decoder input helper outside or inside graph with clear ABI.
- [ ] Implement encoder sparse/full dispatch guard at default threshold `704`.
- [ ] Implement block padding/unpadding and sparse masks.
- [ ] Implement source-equivalent eval random plan, including all-zero sparse random blocks.
- [ ] Implement BigBird sparse attention context path with global/local/random regions.
- [ ] Gate dense sparse-attention probability reconstruction behind requested attentions.
- [ ] Add packed QKV rewrite with bias/layout/materialization guards.
- [ ] Add FFN and LM-head GEMM/fusion rewrites.
- [ ] Add short-sequence dense parity tests.
- [ ] Add decoder cache prefill/decode parity tests.
- [ ] Add sparse encoder parity tests at boundary and long-context lengths.
- [ ] Benchmark encoder sparse, decoder prefill, decode, LM head, and cache memory.
