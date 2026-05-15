# Transformers Family Audit: `pegasus_x`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Transformers describe: v4.50.3-DeepSeek-3-4398-gb75feb2af6
Model id: google/pegasus-x-large as primary; google/pegasus-x-base and finetunes as config sweep
Config source: local configuration_pegasus_x.py plus HF raw config snapshots under _sources/
Source files inspected:
  transformers/src/transformers/models/pegasus_x/configuration_pegasus_x.py
  transformers/src/transformers/models/pegasus_x/modeling_pegasus_x.py
  transformers/src/transformers/models/pegasus_x/__init__.py
Any missing files or assumptions: no processor/image/audio files exist for this family; tokenizer is PegasusTokenizer from the Pegasus family, inspected only through tokenizer_config snapshots.
```

Representative HF configs were fetched from:

- [google/pegasus-x-base](https://huggingface.co/google/pegasus-x-base)
- [google/pegasus-x-large](https://huggingface.co/google/pegasus-x-large)
- [google/pegasus-x-base-arxiv](https://huggingface.co/google/pegasus-x-base-arxiv)
- [pszemraj/pegasus-x-large-book-summary](https://huggingface.co/pszemraj/pegasus-x-large-book-summary)
- [twigs/pegasus-x-large-8192-pubmed](https://huggingface.co/twigs/pegasus-x-large-8192-pubmed)
- [hf-tiny-model-private/tiny-random-PegasusXForConditionalGeneration](https://huggingface.co/hf-tiny-model-private/tiny-random-PegasusXForConditionalGeneration)

Primary DinoML target: inference-only `PegasusXForConditionalGeneration` for long-input summarization on CUDA. Encoder-only `PegasusXModel` is useful for staged validation; training loss, layerdrop, and gradient checkpointing are deferred.

## 2. High-level architecture

PEGASUS-X is a text-only encoder-decoder seq2seq model. The encoder is the unusual part: it uses learned global token embeddings plus block-local token attention, with optional half-block staggering on alternating layers. The decoder is a standard autoregressive decoder with causal self-attention, encoder cross-attention, FFN, and a tied LM head.

```text
tokenizer/input_ids + attention_mask
  -> scaled token embedding + sinusoidal positions
  -> encoder: global/local attention blocks over long source sequence
  -> decoder prefill/decode: causal MHA + encoder cross-attention + FFN
  -> tied LM head logits
  -> generation controller: beam search, forced EOS, length penalty
```

Stage decomposition:

- CPU/data pipeline: SentencePiece/Pegasus tokenization, padding/truncation, generation options.
- Encoder: independently cacheable source representation `[B, S_src, d_model]`; no KV cache, but global/local attention requires block metadata and padded masks.
- Decoder prefill: full target prefix causal self-attention and cross-attention over encoder output.
- Decoder decode: one or more new decoder tokens with self-attention KV cache and reusable cross-attention K/V cache.
- Logits/generation: tied projection to vocab; beam search and forced EOS are controller behavior, not neural graph ops.

## 3. Important config dimensions

Source defaults from `PegasusXConfig`:

| Field | Default |
| --- | ---: |
| `vocab_size` | 96103 |
| `d_model` / hidden size | 1024 |
| `encoder_layers` | 16 |
| `decoder_layers` | 16 |
| `encoder_attention_heads` | 16 |
| `decoder_attention_heads` | 16 |
| `head_dim` | `d_model / heads`, 64 for default/large |
| `encoder_ffn_dim` | 4096 |
| `decoder_ffn_dim` | 4096 |
| `max_position_embeddings` | 16384 |
| `block_size` | 512 |
| `num_global_tokens` | 32 in source default, 128 in observed public checkpoints |
| `stagger_local_blocks` | true |
| `activation_function` | `gelu` in source default, `relu` in observed public checkpoints |
| `scale_embedding` | true |
| `use_cache` | true |
| `tie_word_embeddings` | true |

Representative checkpoint sweep:

| Model | d_model | Enc/Dec layers | Heads | FFN | Max pos | Block | Global tokens | Activation | dtype | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `google/pegasus-x-base` | 768 | 12/12 | 12/12 | 3072 | 16384 | 512 | 128 | relu | float32 | common base |
| `google/pegasus-x-large` | 1024 | 16/16 | 16/16 | 4096 | 16384 | 512 | 128 | relu | float32 | primary large |
| `google/pegasus-x-base-arxiv` | 768 | 12/12 | 12/12 | 3072 | 16384 | 512 | 128 | relu | float32 | task finetune, same topology as base |
| `pszemraj/pegasus-x-large-book-summary` | 1024 | 16/16 | 16/16 | 4096 | 16384 | 512 | 128 | relu | float32 | generation config differs: max_length 512, beams 2 |
| `twigs/pegasus-x-large-8192-pubmed` | 1024 | 16/16 | 16/16 | 4096 | 16384 | 512 | 128 | relu | bfloat16 | finetune advertises bf16 weights |
| `hf-tiny-random-PegasusXForConditionalGeneration` | 16 | 2/2 | 4/4 | 4 | 20 | 512 | 32 | gelu | float32 | tiny/debug; block size exceeds max position |

## 3a. Family variation traps

- Source defaults do not match public checkpoints for `num_global_tokens` and `activation_function`; DinoML should read checkpoint config, not assume the class defaults.
- Encoder and decoder dimensions are symmetric in observed configs, but source has separate encoder/decoder layer, head, and FFN fields.
- `d_model` must be divisible by encoder and decoder head counts. There is no GQA/MQA; KV heads equal attention heads.
- Encoder attention projections are bias-free. Decoder self-attention and cross-attention are also constructed with `bias=False`. FFN layers use PyTorch `nn.Linear` defaults, so they include bias.
- `add_bias_logits`, `add_final_layer_norm`, `normalize_before`, `normalize_embedding`, `static_position_embeddings`, and `extra_pos_embeddings` appear in older configs but are not read by the inspected source. Treat them as ignored historical fields for this source basis.
- Encoder input length is padded to a multiple of `block_size`; local attention assumes `padded_seq_len % block_size == 0`.
- If `stagger_local_blocks=True`, odd encoder layers pad by `block_size // 2` on both sides before attention. Admission should require even `block_size`.
- Encoder returns only token hidden states as `last_hidden_state`; final global states appear only in hidden-state capture, not as decoder cross-attention inputs.
- `_supports_sdpa=False` in current source. Decoder attention can route through FlashAttention/FlexAttention interfaces, but encoder global/local attention is custom eager einsum/softmax logic and is not covered by standard dense MHA backends.
- Tokenizer `model_max_length` snapshots say 1024 even though model `max_position_embeddings` is 16384. End-to-end long-input parity must control tokenizer truncation separately.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs and learned global token IDs.
- Scalar embedding multiply by `sqrt(d_model)` when `scale_embedding=True`.
- Runtime sinusoidal position generation: arange, exp, sin, cos, broadcast add.
- `pad` for sequence and mask axes, including hidden-state pad `[0, 0, 0, pad_len]` and stagger pad `[0, 0, block_size/2, block_size/2]`.
- Slice/unpad back to original source sequence and unstaggered sequence.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, concatenation along sequence/key axes.
- Mask inversion/cast/masked fill to dtype min.

Neural network primitives:

- LayerNorm over `d_model`.
- Dense Linear/GEMM:
  - base large attention projections `Linear(1024 -> 1024, bias=False)`.
  - base FFN `Linear(1024 -> 4096, bias=True)`, activation, `Linear(4096 -> 1024, bias=True)`.
  - base variant `Linear(768 -> 768)`, FFN `768 -> 3072 -> 768`.
  - LM head `Linear(d_model -> vocab_size, bias=False)` tied to shared embedding.
- Activations: `relu` for public checkpoints; `gelu` for source default/tiny.
- Residual adds and inference dropout as no-op.

Attention primitives:

- Encoder custom global/local noncausal attention:
  - global path: `[B,H,G,F] x [B,H,G+S_pad,F] -> [B,H,G,G+S_pad]`.
  - local path: per-block `[B,H,N,K,F] x global/local keys -> [B,H,N,K,G+K]`.
  - softmax over last dim; masks are additive dtype-min values.
- Decoder causal MHA with KV cache, mask shape produced by `create_causal_mask`.
- Decoder encoder-decoder cross-attention with reusable cross-attention K/V cache, mask from `create_bidirectional_mask`.
- No GQA/MQA, no RoPE, no ALiBi.

Position/custom math:

- Dynamic sinusoidal positions for encoder and decoder. Position tensors are not learned weights.

Generation/cache ops:

- `shift_tokens_right` for labels/training and default decoder start behavior.
- `EncoderDecoderCache(DynamicCache, DynamicCache)` with per-layer self-attention and cross-attention stores.
- Cache reorder for beam search is inherited from Transformers cache/generation infrastructure, not implemented in this file; DinoML integration still needs a beam reorder ABI if it owns generation.

Preprocessing-coupled ops:

- PegasusTokenizer/SentencePiece IDs, pad/eos/start token IDs, optional generation config fields: `num_beams`, `length_penalty`, `forced_eos_token_id`.

## 5. Layer/block breakdown

Encoder input:

```text
input_ids [B,S] -> shared embedding [B,S,D] * sqrt(D)
positions [B,S,D] from sinusoidal function
x = dropout(emb + pos)
mask [B,S] -> additive mask [B,S] with 0 or finfo(dtype).min
if S % block_size != 0: pad x and mask to S_pad
global = embed_global(arange(num_global_tokens)).expand(B,G,D)
```

Encoder layer, repeated `encoder_layers`:

```text
token_residual = token
global_residual = global
token = LayerNorm(token)
global = LayerNorm(global)
if odd layer and stagger_local_blocks:
  token, mask = pad half block on both sides
token_attn, global_attn = GlobalLocalAttention(token, global, mask)
if staggered: token_attn = token_attn[:, half_block:-half_block, :]
token = token_residual + token_attn
global = global_residual + global_attn

token_residual = token
token = token_residual + fc2(act(fc1(LayerNorm(token))))
global_residual = global
global = global_residual + fc2(act(fc1(LayerNorm(global))))
```

GlobalLocalAttention shapes:

```text
token q/k/v:  [B,H,S_pad,F]
global q/k/v: [B,H,G,F]
global output attends over [G + S_pad]
local q/k/v blocked to [B,H,N_blocks,block_size,F]
local output attends over [G + block_size] for each block
```

Decoder layer, repeated `decoder_layers`:

```text
residual = x
x = LayerNorm(x)
x = residual + out_proj(CausalMHA(q_proj/k_proj/v_proj(x), self KV cache, causal mask))

residual = x
x = LayerNorm(x)
x = residual + out_proj(CrossMHA(q_proj(x), k_proj/v_proj(encoder_hidden), cross KV cache, encoder mask))

residual = x
x = residual + fc2(act(fc1(LayerNorm(x))))
```

Conditional generation head:

```text
decoder_hidden [B,T,D] -> lm_head tied to shared embedding -> logits [B,T,V]
```

## 6. Attention requirements

Encoder attention:

- Noncausal self-attention over source tokens plus learned global tokens.
- MHA only: `num_key_value_heads == num_attention_heads`.
- `head_dim = d_model / heads`, 64 for base and large.
- Global queries attend to global keys/values and all padded local token keys/values.
- Local token queries attend to all global keys/values and the token keys/values in the same block only.
- Masking uses additive dtype-min values. For global queries, the source mask is padded with `G` unmasked global slots. For local queries, each block mask is padded with `G` global slots.
- Requires block padding and optional half-block staggering guards. This is neither ordinary dense MHA nor sliding window exactly; it is block-local plus fully visible global tokens.
- No KV cache in encoder.

Decoder self-attention:

- Causal MHA, query/key/value width `d_model`.
- Uses `ALL_ATTENTION_FUNCTIONS` with `_attn_implementation`; source advertises FlashAttention and FlexAttention support but disables SDPA due to flaky logits.
- Self KV cache stores keys/values shaped logically `[B,H,T_cache,F]` per layer before any backend-specific packing.
- Cached keys are after projection/reshape and before attention; no RoPE/relative position update is applied to K.

Decoder cross-attention:

- Noncausal rectangular MHA: decoder queries length `T_dec`, encoder key/value length `S_src`.
- Cross-attention K/V are cached after first computation through `EncoderDecoderCache.cross_attention_cache`; `is_updated[layer_idx]` prevents recomputing encoder K/V on later decode steps.
- Cross mask is bidirectional over encoder source positions.

Flash/SDPA compatibility:

- Decoder `PegasusXAttention` is compatible with the Transformers attention interface and may use Flash/Flex when available.
- Encoder `PegasusXGlobalLocalAttention` must be lowered as a custom attention family or decomposed into block GEMM/softmax/GEMM patterns. Treat dense-MHA rewrites as invalid unless the global/local pattern is explicitly reconstructed.

## 7. Position encoding and custom math

PEGASUS-X uses sinusoidal positions for both encoder and decoder. It computes position tensors dynamically in the model dtype.

```python
def pegasus_x_sinusoidal(seq_len, d_model, position_ids, max_scale=10000.0):
    half = d_model // 2
    div = exp(arange(half) * -(log(max_scale) / (half - 1)))
    pe = zeros([seq_len, d_model])
    pe[:, :half] = sin(position_ids * div)
    pe[:, half:] = cos(position_ids * div)
    return pe
```

Precompute options:

- Encoder positions for fixed source buckets can be precomputed per dtype and max length, then sliced.
- Decoder decode can precompute a position table up to `max_position_embeddings` and gather by absolute `past_key_values_length + token_index`.
- No position embedding weight exists to load. Any artifact-visible constant should record that it is generated math, not a checkpoint tensor.

## 8. Preprocessing and input packing

Text/tokenization:

- Uses `PegasusTokenizer` with pad token `<pad>` id 0, eos `</s>` id 1, decoder start token id 0, mask tokens inherited from Pegasus tokenizer config.
- Tokenizer config snapshots for Google base/large report `model_max_length=1024`; this is a tokenizer truncation setting, while the model config supports positions up to 16384. Long-input summarization parity requires overriding tokenizer truncation or explicit input packing.
- Model inputs are `input_ids [B,S_src]`, `attention_mask [B,S_src]`, optional `decoder_input_ids [B,T]`, and optional `decoder_attention_mask [B,T]`.

GPU/runtime packing:

- Encoder must preserve batch-major `[B,S,D]` layout through token/global states.
- Encoder local attention internally reshapes to `[B,H,N_blocks,block_size,F]`; `S_pad` is runtime padded upward to a multiple of `block_size`.
- Decoder cache ABI should separate encoder output cache from decoder self/cross KV cache.

Generation controller:

- Google generation configs include `num_beams=8`, `length_penalty=0.8`, `forced_eos_token_id=1`, and `max_length=16384`.
- Some finetunes override generation behavior in config without a separate `generation_config.json`; e.g. book-summary config has `max_length=512`, `num_beams=2`, `early_stopping=True`, and n-gram constraints.
- Beam search, no-repeat n-gram, forced EOS, and length penalty can be stubbed for first neural parity but are required for end-to-end text parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: encoder global path -> dense attention over concatenated K/V

Source pattern:

```text
global_q [B,H,G,F]
concat(global_k, local_k, dim=2) [B,H,G+S_pad,F]
softmax(global_q @ concat_k.T + extended_mask) @ concat_v
```

Replacement:

```text
Flash/DenseAttention(q=global_q, k=concat_k, v=concat_v, mask=[B,1,1,G+S_pad])
```

Preconditions:

- Noncausal attention.
- Concatenated K/V are materialized or represented by a two-source attention kernel.
- Mask has zero values for global slots and source additive mask for local slots.
- `G` and `S_pad` are known to the kernel launch.

Failure cases:

- Do not include local token outputs in this rewrite; only global queries match dense rectangular attention.

Parity sketch:

- Compare global attention output against source einsum path for random masks, all-pad tail, and mixed dtype.

### Rewrite: local path -> block-batched attention with global prefix

Source pattern:

```text
local_q -> [B,H,N,K,F]
local_k/v -> [B,H,N,K,F]
global_k/v -> [B,H,G,F]
attn logits = concat(q @ global_k.T, q @ local_k_block.T)
softmax over G+K
```

Replacement:

```text
BlockLocalGlobalAttention(B,H,N,K,F,G)
```

Preconditions:

- `S_pad % block_size == 0`.
- `block_size > 0`; if stagger is enabled, `block_size` must be even.
- Mask is already padded to `S_pad` and can be viewed as `[B,N,K]`.
- Global K/V are shared across all blocks.

Failure cases:

- Ordinary sliding-window attention kernels are not equivalent unless they include all global tokens and only same-block locals.
- Staggered layers require the half-block pad/slice transformation before this kernel.

Parity sketch:

- Test short `S < block_size`, exact multiple, non-multiple padded, and staggered odd-layer cases.

### Rewrite: sinusoidal positions -> precomputed table gather

Preconditions:

- `max_position_embeddings` and dtype fixed at compile/profile time.
- Position IDs are contiguous arange for encoder/prefill or scalar absolute offsets for decode.

Replacement:

```text
precompute table [max_position_embeddings,D] -> gather/slice -> add to embeddings
```

Failure cases:

- If caller supplies nonstandard `position_ids`, preserve dynamic math or gather by explicit IDs.

### Rewrite: tied LM head -> embedding transpose GEMM

Preconditions:

- `lm_head.weight` aliases `model.shared.weight`.
- Bias is absent.
- Vocab dimension is unchanged after tokenizer/model resize.

Replacement:

```text
hidden [B,T,D] x shared_weight.T [D,V] -> logits [B,T,V]
```

Failure cases:

- Do not clone the shared embedding as an independent constant; aliasing matters for weight loading and resize parity.

### Layout guards

- Preserve source `[B,S,D]` and attention `[B,H,S,F]` semantics for initial lowering.
- No NHWC/channel-last rewrite is relevant for this text-only model.
- Guard axis-sensitive ops: sequence padding/slicing on dim 1, attention softmax on last dim, concat on sequence/key axes, block view order `[B,H,N,K,F]`.

## 10. Kernel fusion candidates

Highest priority:

- Encoder `BlockLocalGlobalAttention`: this is the family blocker. A decomposition into many small GEMMs and softmaxes will be functional but likely slow for long inputs.
- Decoder MHA with cache: reuse DinoML attention work for causal self-attention and cross-attention; expose cache shapes and cross-KV reuse.
- GEMM + bias + ReLU/GELU FFN epilogues: public checkpoints use ReLU, source default/tiny uses GELU.
- LayerNorm + residual neighborhoods: every encoder and decoder block uses pre-norm residual structure.

Medium priority:

- Sinusoidal position table precompute/gather.
- LM head last-token-only projection for decode.
- Encoder global-token attention as a dense rectangular attention subcase.
- Block padding/stagger pad/slice fusion with attention input preparation.

Lower priority:

- Attention probability outputs for `output_attentions`; useful for debugging, not first inference target.
- Training loss and `shift_tokens_right`; needed for training/eval loss, not runtime generation.
- Beam-search controller optimizations.

## 11. Runtime staging plan

Stage 1: parse config and load weights.

- Enforce supported topology: MHA only, bias-free attention projections, ReLU/GELU FFN, tied embedding/LM head.
- Record ignored historical config fields as non-operative.

Stage 2: encoder one-layer parity.

- Implement embedding scale, sinusoidal positions, mask conversion, block padding, learned global embeddings, and one global/local attention layer.

Stage 3: full encoder parity.

- Add alternating stagger behavior and final token output trim/layernorm.
- Validate source lengths below, equal to, and above `block_size`.

Stage 4: decoder prefill parity.

- Implement causal MHA, cross-attention, FFN, and tied LM head over full prefixes.

Stage 5: decode cache parity.

- Implement `EncoderDecoderCache` equivalent: self KV grows per token; cross K/V computed once per layer and reused.

Stage 6: optimized attention.

- Add provider-backed block-local-global attention and decoder FlashAttention path with graph-visible fallback conditions.

Stage 7: generation parity.

- Add forced EOS, beam search, length penalty, no-repeat n-gram options as controller work.

## 12. Parity and validation plan

- Unit tests for sinusoidal positions against Transformers for fp32/fp16/bf16 with contiguous and offset decoder positions.
- Mask tests for attention masks with no padding, tail padding, and all-padding tails.
- Encoder local/global attention random tensor parity for:
  - `S < block_size`
  - `S == block_size`
  - `S = block_size + 1`
  - `S` multiple of block size
  - stagger on/off
- Single encoder layer parity including FFN and residuals.
- Full encoder parity for base and large config dimensions with small synthetic sequence lengths.
- Decoder prefill parity with and without encoder attention mask.
- Decode parity over several one-token steps, checking logits and cache lengths.
- Cross-attention cache parity: verify encoder K/V projections are reused after first decode step.
- End-to-end summarization smoke against `google/pegasus-x-base` on a short input, then long-input smoke with tokenizer truncation disabled/controlled.
- Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 `atol=5e-2, rtol=5e-2` for full models, with tighter per-op tolerances where accumulation is fp32.

## 13. Performance probes

- Encoder throughput sweep by `S_src`: 512, 1024, 2048, 4096, 8192, 16384.
- Block-size sensitivity if nonstandard configs appear; default observed block is 512.
- Global token count sweep: 32 source default vs 128 public checkpoints.
- Stagger on/off comparison for synthetic configs.
- Encoder decomposition benchmark: eager decomposed GEMM/softmax/GEMM vs custom block-local-global kernel.
- Decoder prefill tokens/sec by target length.
- Decode tokens/sec with encoder length sweep to expose cross-attention cost.
- KV cache memory usage for decoder self cache and cross-attention cache separately.
- LM head full-prefix vs last-token-only logits.
- dtype sweep: fp32, fp16, bf16; include bf16 finetune config.
- Generation controller overhead for beam count 1/2/8.

## 14. Skip/defer list

- Training loss, labels, gradient checkpointing, and layerdrop.
- `output_attentions` dense/probability materialization for encoder local/global maps.
- Beam search and no-repeat n-gram for first neural graph parity; keep generation-controller hooks visible.
- Arbitrary historical config fields not read by current source.
- SDPA path, because current source explicitly disables `_supports_sdpa`.
- GQA/MQA, RoPE, ALiBi, MoE, quantized packed weights, multimodal preprocessing: not present in this family.
- General sliding-window attention kernels unless wrapped with explicit Pegasus-X global/local semantics.

## 15. Final implementation checklist

- [ ] Parse `PegasusXConfig` and reject unsupported topology/config combinations.
- [ ] Load shared embedding once and preserve encoder/decoder/lm-head tied aliases.
- [ ] Implement scaled token embeddings and learned global-token embeddings.
- [ ] Implement or precompute Pegasus-X sinusoidal positions.
- [ ] Implement attention mask inversion and dtype-min additive masks.
- [ ] Implement encoder padding to `block_size` and final trim.
- [ ] Implement stagger pad/slice on odd encoder layers when enabled.
- [ ] Implement encoder global attention path.
- [ ] Implement encoder block-local plus global-prefix attention path.
- [ ] Add LayerNorm, residual, and FFN parity for encoder/decoder blocks.
- [ ] Implement decoder causal MHA prefill.
- [ ] Implement decoder cross-attention over encoder hidden states.
- [ ] Implement `EncoderDecoderCache`-style self and cross KV cache ABI.
- [ ] Implement tied LM head projection and last-token-only decode optimization.
- [ ] Add graph rewrites for sinusoidal precompute and block-local-global attention lowering.
- [ ] Add parity tests for one layer, full encoder, prefill logits, and decode-step logits.
- [ ] Benchmark encoder length sweeps and decode cache memory.

## Gated DinoML gaps

- `BlockLocalGlobalAttention` is the main admission gate; current DinoML attention checklist covers flash/memory-efficient attention broadly, but Pegasus-X requires a distinct global-plus-block-local source pattern.
- Dynamic sequence padding to `block_size` and half-block staggering need shape guards and artifact-visible runtime shape math.
- Decoder cache integration must represent both growing self-attention KV and one-time cross-attention KV reuse.
- Source disables SDPA; routing to a generic SDPA path should be rejected unless parity is re-established for this source version.
- Tokenizer/model max-length mismatch must be handled outside the graph so long-input tests actually exercise the model's 16k positions.
