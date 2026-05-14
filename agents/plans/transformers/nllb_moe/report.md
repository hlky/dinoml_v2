# NLLB-MoE Transformers Audit

## 1. Source basis

Transformers commit/version:
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id:
Primary production checkpoint `facebook/nllb-moe-54b`; debug/test checkpoints and community quantized mirrors are listed in `config_sweep.md`.

Config source:
`X:/H/transformers/src/transformers/models/nllb_moe/configuration_nllb_moe.py`, plus Hugging Face raw configs fetched on 2026-05-13:
`facebook/nllb-moe-54b`, `hf-internal-testing/random-nllb-moe-2-experts`,
`hf-tiny-model-private/tiny-random-NllbMoeForConditionalGeneration`,
`hf-tiny-model-private/tiny-random-NllbMoeModel`,
`madatnlp/nllb-moe-54b-8bit`, and `KnutJaegersberg/nllb-moe-54b-4bit`.

Source files inspected:
- `src/transformers/models/nllb_moe/modeling_nllb_moe.py`
- `src/transformers/models/nllb_moe/configuration_nllb_moe.py`
- `src/transformers/models/nllb_moe/convert_nllb_moe_sharded_original_checkpoint_to_pytorch.py`
- `src/transformers/models/nllb/tokenization_nllb.py`
- `src/transformers/models/auto/*` mappings for model/tokenizer registration
- `tests/models/nllb_moe/test_modeling_nllb_moe.py`
- `docs/source/en/model_doc/nllb-moe.md`

Any missing files or assumptions:
No modular source file exists for this family in the inspected checkout; `modeling_nllb_moe.py` is authoritative. `ArthurZ/nllb-moe-128` returned HTTP 401 and is an access gap. Weight tensor metadata for the 54B checkpoint was not downloaded because the official checkpoint is hundreds of GB; weight shape claims below come from source/config unless explicitly labeled config metadata.

## 2. High-level architecture

NLLB-MoE is a multilingual encoder-decoder seq2seq model for translation. It is BART/M2M-like in attention structure and language-token ABI, with sparse MoE feed-forward layers inserted every configured `encoder_sparse_step` and `decoder_sparse_step`.

Dataflow:

```text
NLLB tokenizer/lang prefix -> encoder dense/MoE stack -> decoder prefill/decode with self-cache + cross-cache -> tied LM head -> generation controller forced target BOS
```

Stage decomposition:
- CPU/data pipeline: BPE/metaspace tokenization, language-code prefix/suffix insertion, padding, `attention_mask`, and target `forced_bos_token_id`.
- Encoder: independently cacheable for a fixed source sentence; bidirectional self-attention plus sparse/dense FFN layers.
- Decoder prefill/decode: causal self-attention KV cache, encoder-decoder cross-attention cache, sparse/dense FFN layers, final LM projection.
- Generation controller: owns beam search, forced target language BOS, EOS/pad rules, and sampling. The core graph only returns logits and cache state.

## 3. Important config dimensions

| Field | Source default | Official 54B config | Notes |
| --- | ---: | ---: | --- |
| `vocab_size` | 128112 | 256206 | Official tokenizer has 200-language vocabulary/language codes. |
| `d_model` / hidden size | 1024 | 2048 | Attention projection width. |
| Encoder layers | 12 | 24 | Sparse when `(i + 1) % encoder_sparse_step == 0`. |
| Decoder layers | 12 | 24 | Sparse when `(i + 1) % decoder_sparse_step == 0`. |
| Encoder/decoder heads | 16 / 16 | 16 / 16 | MHA, no GQA/MQA. |
| Head dim | 64 | 128 | Computed as `d_model // heads`; source rejects non-divisible configs. |
| Encoder/decoder FFN dim | 4096 / 4096 | 8192 / 8192 | Per expert and dense FFN width. |
| Max positions | 1024 | 1024 | Sinusoidal table expands dynamically if needed. |
| Activation | ReLU | ReLU | `ACT2FN[activation_function]`. |
| Num experts | 128 | 128 | Debug configs vary to 2 or 4. |
| Expert capacity | 64 | 64 | Eval capacity may become `ceil(fraction * tokens)`. |
| Sparse step | 4 / 4 | 4 / 4 | Official 54B has 6 sparse encoder layers and 6 sparse decoder layers. |
| Router dtype | float32 | float32 | Router classifier is cast to this dtype unless bitsandbytes attributes are present. |
| Router bias | false | false | Source supports bias by config. |
| Cache support | true | true | `EncoderDecoderCache(DynamicCache, DynamicCache)`. |

Representative checkpoint sweep:

| Checkpoint | Kind | Shape/operator variation |
| --- | --- | --- |
| `facebook/nllb-moe-54b` | Official production | 24+24 layers, 2048 hidden, 8192 FFN, 128 experts, 1024 positions, `torch_dtype=float32` in config. |
| `hf-internal-testing/random-nllb-moe-2-experts` | HF debug | Same large config dimensions but only 2 experts; useful for routing parity without 128 experts. |
| `hf-tiny-model-private/tiny-random-NllbMoeForConditionalGeneration` | Tiny test | 4+4 layers, hidden 16, heads 4, FFN 4, 4 experts, decoder sparse every layer, `model_type="nllb_moe"` underscore spelling. |
| `madatnlp/nllb-moe-54b-8bit` | Community 8-bit | Same 54B topology plus bitsandbytes `load_in_8bit`; source does not implement a native NLLB-MoE packed kernel. |
| `KnutJaegersberg/nllb-moe-54b-4bit` | Community 4-bit | Same 54B topology plus bitsandbytes `load_in_4bit`; treat as external loading/provider contract. |

## 3a. Family variation traps

- `model_type` appears as both `nllb-moe` and historical tiny-test `nllb_moe`; native auto mappings use `nllb-moe`.
- The official config advertises `architectures=["NllbMoeModel"]`, while seq2seq generation uses `NllbMoeForConditionalGeneration`.
- `router_jitter_noise` and `router_type` appear in some configs but are not read by the inspected modeling source.
- Router behavior changes substantially with `second_expert_policy`, `normalize_router_prob_before_dropping`, `batch_prioritized_routing`, `moe_eval_capacity_token_fraction`, and `router_ignore_padding_tokens`.
- In eval, source mutates `self.expert_capacity` to `ceil(moe_eval_capacity_token_fraction * nb_tokens)` when the fraction is positive.
- `second_expert_policy="sampling"` and `"random"` use RNG in routing; first DinoML admission should reject these for deterministic inference unless seeded parity is explicitly implemented.
- Attention is MHA only; no GQA/MQA, RoPE, ALiBi, sliding window, or block sparse attention.
- Source flags disable FlashAttention, SDPA, and FlexAttention support. Eager attention is the source parity baseline.
- Tokenizer language control is not optional for translation parity. Source language is inserted into input IDs; target language is enforced by `forced_bos_token_id`.
- `moe_token_dropout` is still applied in eval as a deterministic multiplier `1 - moe_token_dropout` on expert outputs when greater than zero.
- Encoder and decoder token embeddings plus LM head are tied logically to the shared embedding weight.
- Layout translation is not relevant beyond rank-3 text tensors; no NHWC/NCHW regions exist. Axis-sensitive ops are sequence axis `dim=1`, attention softmax `dim=-1`, router softmax `dim=-1`, and token flattening `[B,S,H] -> [B*S,H]`.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup with scale `sqrt(d_model)` when `scale_embedding=True`.
- Add, residual add, dropout disabled in inference, final layer norm.
- `view`, `reshape`, `transpose(1,2)`, `contiguous`, `permute(2,1,0)`.
- Mask construction for bidirectional encoder/cross attention and causal decoder attention.
- Position ID creation from `input_ids != pad_token_id` using `cumsum` along sequence.
- `index_select` into sinusoidal position table.

Neural primitives:
- `LayerNorm(d_model)` pre-attention/pre-FFN plus final encoder/decoder norms.
- Dense `Linear(d_model -> d_model)` Q/K/V/O projections with bias.
- Dense FFN `Linear(d_model -> ffn_dim) -> ReLU -> Linear(ffn_dim -> d_model)` with bias.
- LM head `Linear(d_model -> vocab_size, bias=False)`, tied to shared embeddings.

MoE routing and expert ops:
- Router `Linear(d_model -> num_experts, bias=router_bias)` in `router_dtype`, normally float32.
- Softmax over experts, argmax top-1, masked-fill top-1 to `-inf`, argmax top-2, one-hot masks.
- Optional Gumbel sampling or random second-expert gate; defer/reject for deterministic first path.
- `cumsum` capacity locations, optional `argsort` for batch-prioritized routing.
- `where`/nonzero expert hit discovery, gather selected token rows per expert, per-expert FFN, scale by router weight, deterministic eval `moe_token_dropout` multiplier, `index_add_` back to flattened token rows.

Attention primitives:
- Dense MHA self-attention in encoder and decoder.
- Dense cross-attention in decoder with encoder hidden states as K/V.
- Matmul QK^T, additive mask, softmax, dropout disabled in inference, matmul AV.
- KV cache update/reuse for decoder self-attention and cross-attention.

Position/tokenizer/generation ops:
- Sinusoidal positions with padding index zeroed and offset `2`.
- Language-code prefix/suffix token insertion in tokenizer.
- Forced decoder BOS for target language.
- Shift-right labels for training only; not required for inference.

Quantized/packed metadata:
- No native quantized weights in official source. Community 4-bit/8-bit configs rely on bitsandbytes metadata; DinoML should route those to a separate weight-provider admission or reject initially.

## 5. Layer/block breakdown

Encoder embedding:

```text
input_ids [B,Ssrc] -> shared Embedding[vocab,d_model] * sqrt(d_model)
position_ids = cumsum(input_ids != pad) + pad_id
hidden = token_embed + sinusoidal_position_embed
```

Encoder layer, repeated `encoder_layers`:

```text
res = x
x = LayerNorm(x)
x = MHA_self(q,k,v: d_model -> d_model, heads=H, noncausal mask)
x = res + dropout(out_proj(x))
res = x
x = LayerNorm(x)
if sparse layer:
  x = Top2Router(d_model -> E) + selected expert FFNs(d_model -> encoder_ffn_dim -> d_model)
else:
  x = Linear(d_model -> encoder_ffn_dim) -> ReLU -> Linear(encoder_ffn_dim -> d_model)
x = res + dropout(x)
```

Decoder layer, repeated `decoder_layers`:

```text
res = x
x = LayerNorm(x)
x = causal MHA_self(q,k,v: d_model -> d_model, cache self K/V)
x = res + dropout(out_proj(x))
res = x
x = LayerNorm(x)
x = cross MHA(q from decoder, k/v from encoder, cache cross K/V)
x = res + dropout(out_proj(x))
res = x
x = LayerNorm(x)
x = dense or sparse FFN, same routing pattern as encoder
x = res + dropout(x)
```

Official 54B dimensions:
- Q/K/V/O: `Linear(2048 -> 2048)` with 16 heads, head dim 128.
- FFN/expert: `Linear(2048 -> 8192)`, ReLU, `Linear(8192 -> 2048)`.
- Router per sparse layer: `Linear(2048 -> 128)`, no bias.
- Sparse layers: encoder layers 4, 8, 12, 16, 20, 24 by 1-based index; same in decoder.

## 6. Attention requirements

Encoder self-attention:
- Noncausal bidirectional MHA.
- Q/K/V shape after projection: `[B, H, Ssrc, Dhead]`.
- Additive mask from `create_bidirectional_mask`; padding positions masked.

Decoder self-attention:
- Causal MHA.
- During decode, query length is usually 1 while key/value length grows with cache.
- Cached self K/V are stored after projection and head transpose, before attention matmul.
- Cache shape per layer is effectively `[B, H, Tdec, Dhead]`.

Decoder cross-attention:
- Query from decoder hidden states `[B,Tdec,d_model]`.
- Key/value from encoder hidden states `[B,Ssrc,d_model]`.
- Cross K/V are projected once and stored in `EncoderDecoderCache.cross_attention_cache`; `is_updated[layer_idx]` avoids recomputing them after first generated token.
- Cross cache shape per layer is `[B,H,Ssrc,Dhead]`.

Backend compatibility:
- Source dispatch goes through `ALL_ATTENTION_FUNCTIONS`, but `_supports_flash_attn`, `_supports_sdpa`, and `_supports_flex_attn` are all false. DinoML should treat eager matmul-mask-softmax-matmul as required parity, then add optimized attention behind tests that preserve mask semantics.

## 7. Position encoding and custom math

NLLB-MoE uses learned token embeddings plus deterministic sinusoidal positions. Position IDs ignore pads and start at `padding_idx + 1`, with `padding_idx=1` in official configs.

```python
def nllb_moe_position_ids(input_ids, padding_idx, past_len=0):
    mask = (input_ids != padding_idx).int()
    inc = (cumsum(mask, dim=1).type_as(mask) + past_len) * mask
    return inc.long() + padding_idx
```

Sinusoidal table construction:

```python
half = d_model // 2
freq = exp(arange(half).float() * -(log(10000) / (half - 1)))
angles = arange(num_positions).float()[:, None] * freq[None, :]
table = concat([sin(angles), cos(angles)], dim=1)
if d_model is odd: append one zero column
table[padding_idx] = 0
```

The table can be precomputed up to max source/target length plus offset; source expands it dynamically if runtime length exceeds the current buffer.

## 8. Preprocessing and input packing

Tokenizer:
- Class: `NllbTokenizer`, registered for `nllb-moe`.
- Backend: tokenizers BPE with metaspace pre-tokenizer/decoder; optional precompiled SPM charmap normalization.
- Language table: `FAIRSEQ_LANGUAGE_CODES`, 200-style Flores language codes such as `eng_Latn`, `fra_Latn`.
- Model inputs: `input_ids`, `attention_mask`.

Language ABI:
- Default non-legacy source format: `<src_lang> tokens </s>`.
- Default non-legacy target format: `<tgt_lang> tokens </s>`.
- Legacy mode source/target format: `tokens </s> <lang>`.
- Translation helper requires both `src_lang` and `tgt_lang`, sets tokenizer `src_lang`, tokenizes source, then emits `forced_bos_token_id=tgt_lang_id`.
- For generation parity, DinoML must expose target language as generation-controller metadata, not as a neural op.

GPU/runtime boundary:
- The compiled graph should accept already-tokenized `input_ids`, `attention_mask`, `decoder_input_ids` or generated next token, encoder outputs/cache as staged artifacts, and produce logits/cache.
- Tokenization, language-code selection, beam-search bookkeeping, and text decoding can remain CPU/controller work for first integration.

## 9. Graph rewrite / lowering opportunities

### Rewrite: shared embedding alias preservation

Source pattern:
`model.shared`, `encoder.embed_tokens`, `decoder.embed_tokens`, and `lm_head.weight` are tied logical weights.

Replacement:
One constant storage object with multiple logical consumers.

Preconditions:
- `tie_word_embeddings=True`.
- Weight shapes match `[vocab_size, d_model]`; LM head uses transpose convention of `Linear(d_model -> vocab)`.

Failure cases:
Untied or resized embeddings need a separate admission path.

Parity test sketch:
Mutate one source weight row in PyTorch and confirm encoder, decoder, and LM logits all observe the same change.

### Rewrite: dense FFN to fused GEMM-ReLU-GEMM

Source pattern:
`Linear(d_model -> ffn_dim) -> ReLU -> Linear(ffn_dim -> d_model)`.

Replacement:
CUTLASS GEMM with ReLU epilogue for first projection if useful, then second GEMM; optionally fuse activation with first GEMM output materialization.

Preconditions:
- `activation_function == "relu"`.
- Inference mode, dropout disabled.
- Dense layer, or per-expert selected-token batch after routing.

Failure cases:
Other activation functions from config or training dropout.

Parity test sketch:
Single FFN/expert random tensor parity fp32/fp16 with exact source bias order.

### Rewrite: deterministic Top-2 MoE dispatch

Source pattern:
router softmax -> top1/top2 masks -> capacity cumsum -> per-expert gather -> expert FFN -> weighted `index_add`.

Replacement:
Two-stage MoE provider:
1. Routing kernel emits compact token lists, expert ids, top slot, combine weights, and dropped-token mask.
2. Grouped expert GEMM/FFN consumes packed rows and scatters/index-adds weighted outputs.

Preconditions:
- `second_expert_policy == "all"`.
- `batch_prioritized_routing == false` for first deterministic path.
- Fixed `num_experts`, capacity policy, and eval mode multiplier.

Failure cases:
`sampling`/`random`, batch-prioritized routing without stable sort parity, or capacity mutation not represented in artifact-visible runtime state.

Parity test sketch:
Use `NllbMoeRouterTest` logits and masks; compare top masks/router probabilities and final sparse MLP output.

### Rewrite: cross-attention K/V cache precompute

Source pattern:
During generation, cross-attention K/V are recomputed once then reused under `is_updated[layer_idx]`.

Replacement:
After encoder run, precompute per-layer cross K/V into a persistent cross-cache buffer, or lazily compute on first decode and mark layer updated.

Preconditions:
- Encoder hidden states unchanged.
- Same decoder layer weights and source mask.

Failure cases:
Changing source batch, reordered beams without cache reorder support, or encoder-output recomputation.

Parity test sketch:
Decode first token with and without precomputed cross cache; compare logits and cache contents.

### Rewrite: last-token-only LM head

Source pattern:
During incremental decode, full decoder hidden for current step is `[B,1,d_model]`; LM head projects only current token.

Replacement:
GEMM for `[B,d_model] x [d_model,vocab]` instead of sequence-wide projection.

Preconditions:
Decode step with `Tdec=1`; prefill still needs full logits only if caller requests them.

Failure cases:
Teacher-forcing/prefill logits for all target positions.

Parity test sketch:
Compare generation logits for last step against full model output slice.

## 10. Kernel fusion candidates

Highest priority:
- Top-2 router and expert dispatch. This is the defining cost/complexity beyond ordinary seq2seq; source uses many dynamic PyTorch indexing ops that need a real provider plan.
- Dense MHA prefill/decode attention with cache. Even without Flash/SDPA source support, matmul-softmax-matmul and cache ABI must be solid.
- FFN/expert GEMMs. Official 54B has 128 experts per sparse layer with `2048 -> 8192 -> 2048` weights.
- LayerNorm + residual patterns. Every block has pre-norm attention and FFN norms.

Medium priority:
- Cross-attention K/V precompute/reuse.
- Last-token LM head projection.
- Sinusoidal position ID generation and embedding add.
- Router softmax/top-k/cumsum compact kernels.

Lower priority:
- Beam cache reorder and continuous batching.
- Bitsandbytes 4-bit/8-bit community weight import.
- Batch-prioritized routing stable sort path.
- Training-only router auxiliary/z losses and label shift.

## 11. Runtime staging plan

Stage 1: Config/tokenizer admission and weight loading.
Parse `NllbMoeConfig`, preserve tied embedding aliases, accept tokenized inputs, and reject unsupported router policies/quantized configs clearly.

Stage 2: Dense tiny model parity.
Run tiny configs with sparse steps disabled or num experts small, covering embeddings, sinusoidal positions, encoder/decoder attention, dense FFN, final logits.

Stage 3: Router-only and sparse MLP parity.
Implement deterministic `second_expert_policy="all"` routing with no batch-prioritized sort first; compare to `NllbMoeRouterTest`.

Stage 4: One sparse encoder/decoder block parity.
Validate selected-token expert FFN gather/scatter and eval `moe_token_dropout` multiplier.

Stage 5: Full encoder-prefill parity.
Run source sentence through encoder and decoder prefill with official-like config at small dimensions.

Stage 6: Decode cache parity.
Implement `EncoderDecoderCache` equivalent: self K/V grows, cross K/V reused, beam reorder deferred until basic greedy decode works.

Stage 7: Production optimizations.
Grouped expert GEMM, fused routing kernels, optimized attention, last-token logits, and scheduling/batching.

## 12. Parity and validation plan

- Sinusoidal position tests: pad/non-pad position IDs, direct `inputs_embeds` path, `past_key_values_length` offset.
- Router unit tests: reproduce top-2 masks, probabilities, capacity drops, padding mask behavior, and eval capacity mutation for small logits.
- Expert dispatch tests: compare `NllbMoeSparseMLP` on random hidden states for 2, 4, and 128 expert metadata, with deterministic routing.
- One-layer encoder parity: fp32 absolute tolerance around `1e-4`; fp16 around `1e-2` after accumulation-policy decisions.
- One-layer decoder self-attention and cross-attention parity with and without cache.
- Full tiny conditional-generation parity using `hf-tiny-model-private/tiny-random-NllbMoeForConditionalGeneration`.
- Debug checkpoint parity using `hf-internal-testing/random-nllb-moe-2-experts` for logits slices if weights are available.
- Generation ABI test: `src_lang=eng_Latn`, `forced_bos_token_id=fra_Latn`, greedy first-token parity.
- Cache continuation test: compare no-past full decode slice with cached decode slice, matching the Transformers test pattern.

## 13. Performance probes

- Tokenization throughput by batch and source length; separate from GPU graph.
- Encoder-only throughput for `[B,Ssrc]` with dense vs sparse layer counts.
- Router kernel time: softmax/top2/cumsum/packing separately from expert GEMM.
- Expert utilization histogram, dropped-token rate, and capacity fraction sweep.
- Grouped expert GEMM throughput across active experts and token counts.
- Decoder prefill throughput by target prefix length.
- Decode tokens/sec with self-cache and cross-cache memory resident.
- Cross-attention cache memory: `layers * 2 * B * heads * Ssrc * head_dim * dtype`.
- LM head last-token GEMM throughput for vocab `256206`.
- Community 4-bit/8-bit load/dequant probe only after a weight-provider path is admitted.

## 14. Skip/defer list

- Training, gradient checkpointing, LayerDrop randomness, router auxiliary/z losses.
- `second_expert_policy="sampling"` and `"random"` for first deterministic inference.
- `batch_prioritized_routing=True` stable-sort parity unless a target config requires it.
- FlashAttention/SDPA/FlexAttention parity; source disables these.
- Beam search and beam cache reorder beyond a simple generation-controller fallback.
- Bitsandbytes 4-bit/8-bit execution; treat as separate provider/loading work.
- Full official 54B end-to-end load until smaller configs prove graph/caching/routing.
- `output_attentions`, `output_hidden_states`, and router-logit debug outputs as runtime products.

## 15. Final implementation checklist

- [ ] Parse `NllbMoeConfig`, including historical `model_type="nllb_moe"` compatibility only if needed.
- [ ] Load shared/tied embeddings and LM head as one logical parameter.
- [ ] Implement NLLB tokenizer/language ABI outside the compiled graph or define controller inputs for language IDs.
- [ ] Implement scaled token embedding and sinusoidal position embedding with pad-aware `cumsum`.
- [ ] Implement encoder bidirectional MHA and decoder causal/cross MHA eager parity.
- [ ] Define `EncoderDecoderCache` manifest: per-layer self K/V and cross K/V plus cross-cache updated flags.
- [ ] Implement dense FFN `Linear-ReLU-Linear`.
- [ ] Implement deterministic top-2 router for `second_expert_policy="all"`, no batch-prioritized routing first.
- [ ] Implement selected-token expert gather, grouped/per-expert FFN, weighted `index_add` scatter.
- [ ] Preserve eval `moe_token_dropout` multiplier.
- [ ] Reject or separately admit stochastic/batch-prioritized router configs.
- [ ] Add one-block, router, sparse-MLP, tiny-model, and cached-decode parity tests.
- [ ] Add performance probes for router packing, expert GEMM, prefill, decode, and LM head.
