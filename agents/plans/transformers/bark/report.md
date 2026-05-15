# Bark Transformers family audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: suno/bark, suno/bark-small, hf-internal-testing/tiny-random-BarkModel
Config source: Hugging Face config.json/generation_config.json snapshots under _sources/hf/
Source files inspected:
- transformers/src/transformers/models/bark/modeling_bark.py
- transformers/src/transformers/models/bark/configuration_bark.py
- transformers/src/transformers/models/bark/generation_configuration_bark.py
- transformers/src/transformers/models/bark/processing_bark.py
- transformers/src/transformers/generation/logits_process.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/models/encodec/{configuration_encodec.py,modeling_encodec.py}
Any missing files or assumptions: no Bark preprocessor_config.json exists in inspected repos; Bark uses a tokenizer-only processor plus optional speaker embedding npy assets. No 401/403/gated gaps were observed.
```

Canonical source URLs:

- [modeling_bark.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bark/modeling_bark.py)
- [configuration_bark.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bark/configuration_bark.py)
- [generation_configuration_bark.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bark/generation_configuration_bark.py)
- [processing_bark.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/bark/processing_bark.py)

Representative configs were fetched from [suno/bark-small](https://huggingface.co/suno/bark-small), [suno/bark](https://huggingface.co/suno/bark), [hf-internal-testing/tiny-random-BarkModel](https://huggingface.co/hf-internal-testing/tiny-random-BarkModel), and [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz). Source snapshots are in `agents/plans/transformers/bark/_sources/`.

## 2. High-level architecture

Bark is a staged text-to-audio generator:

```text
BertTokenizer text ids + optional voice history tensors
-> semantic causal GPT-like LM -> semantic tokens
-> coarse causal GPT-like LM with alternating codebook constraints -> first 2 EnCodec codebooks
-> fine non-causal GPT-like model over 8 codebooks -> remaining codebooks
-> EnCodec RVQ decode + convolutional decoder -> waveform at 24 kHz
```

Stage decomposition:

- CPU/data pipeline: BERT tokenizer padding to `max_length=256`; optional speaker preset loading from `.npy` paths in `speaker_embeddings_path.json`.
- Prefix construction: semantic stage offsets text token ids by `text_encoding_offset=10048`, sums text and semantic-history embeddings, and appends `semantic_infer_token=129599`.
- Autoregressive decode: semantic and coarse submodels use causal self-attention and `DynamicCache`.
- Fine stage: no KV cache; iteratively predicts codebook columns in fixed windows using full bidirectional attention.
- Codec/vocoder: Bark calls EnCodec quantizer decode and decoder directly, not `EncodecModel.forward`.

First useful DinoML runtime target: semantic + coarse token/code generation parity with fine/codec stubbed. This exercises embeddings, causal MHA cache, LM heads, logits processors, and Bark controller packing without taking on EnCodec Conv/LSTM decode immediately.

## 3. Important config dimensions

Production dimensions from fetched `config.json`:

| Checkpoint | Stage | block | input vocab | output vocab | layers | heads | hidden | head dim | bias | cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `suno/bark-small` | semantic | 1024 | 129600 | 10048 | 12 | 12 | 768 | 64 | false | true |
| `suno/bark-small` | coarse | 1024 | 12096 | 12096 | 12 | 12 | 768 | 64 | false | true |
| `suno/bark-small` | fine | 1024 | 1056 | 1056 | 12 | 12 | 768 | 64 | false | not used |
| `suno/bark` | semantic | 1024 | 129600 | 10048 | 24 | 16 | 1024 | 64 | false | true |
| `suno/bark` | coarse | 1024 | 12096 | 12096 | 24 | 16 | 1024 | 64 | false | true |
| `suno/bark` | fine | 1024 | 1056 | 1056 | 24 | 16 | 1024 | 64 | false | not used |
| `tiny-random` | all | 256 | 300 | 300 | 2 | 2 | 16 | 8 | true | true for causal |

Generation-controller dimensions:

| Field | `suno/bark*` value | Source |
|---|---:|---|
| semantic max new tokens | 768 | `generation_config.json` |
| max input semantic length | 256 | `generation_config.json` |
| semantic vocab size | 10000 | `generation_config.json` |
| coarse rate | 75 Hz | `generation_config.json` |
| semantic rate | 49.9 Hz | `generation_config.json` |
| coarse codebooks | 2 | `generation_config.json` |
| coarse sliding window | 60 generated tokens | `generation_config.json` |
| max coarse history | 630 tokens | `generation_config.json` |
| max fine input length | 1024 frames | `generation_config.json` |
| max fine history length | 512 frames | `generation_config.json` |
| fine codebooks | 8 | `generation_config.json` |
| codebook size | 1024 | `generation_config.json` |
| waveform sample rate | 24000 | `generation_config.json` and codec config |

Codec dimensions from `codec_config`/`facebook/encodec_24khz`: mono audio, hidden/codebook dim 128, codebook size 1024, 32 base filters, Conv1d kernel 7, upsampling ratios `[8, 5, 4, 2]`, no chunking, target bandwidths `[1.5, 3, 6, 12, 24]`.

## 3a. Family variation traps

- Bark is not one decoder. It is three transformer submodels plus EnCodec decode.
- Semantic input vocab is much larger than semantic output vocab because text tokens are offset into a shared input embedding range.
- Coarse vocab includes semantic pad/infer tokens plus two 1024-token acoustic codebook ranges after `semantic_vocab_size=10000`.
- Fine config has `n_codes_given=1`, but normal generation starts the inner loop at `n_coarse_codebooks=2`; the LM head for codebook 1 exists/ties but is skipped by the standard full-model controller.
- Production configs set `bias=false`; source defaults set `bias=true`. DinoML should read checkpoint config, not source defaults.
- Fine transformer blocks ignore `config.bias` for LayerNorm construction (`nn.LayerNorm(config.hidden_size)`), so production fine LayerNorm still has beta/gamma parameters even when causal semantic/coarse LayerNorm and linears are bias-free.
- Fine model uses full bidirectional masking and no cache, even though `use_cache` exists in base submodel config defaults.
- No RoPE/ALiBi. Position encoding is learned absolute embedding with length `block_size`.
- Eager attention stores Q/K/V as all-Q, all-K, all-V splits from `Linear(hidden -> 3*hidden)`, not packed per head.
- `flash_attention_2` changes Q/K/V layout to `[batch, seq, heads, head_dim]`; eager uses `[batch, heads, seq, head_dim]`.
- Optional voice presets are external arrays, not model weights: `semantic_prompt` rank 1, `coarse_prompt` rank 2 `[2, T]`, `fine_prompt` rank 2 `[8, T]`.
- EnCodec waveform decode is a separate model family. Bark full parity composes EnCodec Conv1d/ConvTranspose1d/LSTM/RVQ coverage.
- Layout-sensitive axes are mostly sequence/codebook axes: fine input is `[B, T, 8]`, then transposed to `[B, 8, T]` before codec. Protect these with no-layout-translation guards.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `hstack`/`cat`, `pad`, `repeat_interleave`, `masked_fill`, `remainder`, `argmax`, `multinomial` for sampled fine generation, `where`, scalar/list length arithmetic.
- Fine codebook packing: `[B, 2*T] -> [B, T, 2]`, pad to `[B, T, 8]`, later `[B, T, 8] -> [B, 8, T]`.

Neural primitives:

- Embedding lookup: semantic `129600 x H`, coarse `12096 x H`, fine 8 x `1056 x H`.
- Learned position embedding: `[1024, H]` or tiny `[256, H]`.
- LayerNorm over hidden, production causal bias disabled; fine LayerNorm uses PyTorch default affine gamma/beta because it is constructed without the `bias=config.bias` argument.
- Linear QKV: `Linear(H -> 3H)` with optional bias.
- Output projection: `Linear(H -> H)`.
- MLP: `Linear(H -> 4H) -> GELU -> Linear(4H -> H)`.
- LM heads: semantic `Linear(H -> 10048, bias=False)`, coarse `Linear(H -> 12096, bias=False)`, fine 7 x `Linear(H -> 1056, bias=False)`.

Attention primitives:

- Causal MHA for semantic/coarse with `num_kv_heads == num_heads`.
- Bidirectional MHA for fine.
- Eager attention: QK matmul, scale by `1/sqrt(head_dim)`, causal mask, additive padding mask, softmax in last dim, cast to value dtype, dropout, AV matmul.
- Optional FlashAttention2 path with Q/K/V `[B, T, Hn, D]` and top-left-mask compatibility flag.

Generation/cache ops:

- `DynamicCache` update per causal layer; keys/values are cached after projection/split, before attention, with learned positions already mixed into hidden states before QKV.
- Suppress token logits processor for semantic.
- Bark EOS prioritizer: softmax scores, if EOS probability exceeds `min_eos_p`, keep only EOS scores.
- Alternating codebooks logits processor for coarse.
- Standard sampling/greedy top-k/top-p/temperature behavior from `GenerationMixin`.

Preprocessing-coupled ops:

- BERT tokenizer configured through `BarkProcessor`; no audio feature extractor for Bark inputs.
- Optional voice preset validation and tensor conversion.

Discrete codebook / codec ops:

- Coarse code ids are offset by semantic vocab size and codebook offset.
- Fine output integer codes are passed to EnCodec residual vector quantizer decode.
- EnCodec decode needs embedding gather/sum across quantizers, Conv1d/ConvTranspose1d, residual blocks, ELU, LSTM, weight norm, reflect/causal padding. This is required for waveform parity but can be deferred behind code-token parity.

## 5. Layer/block breakdown

Causal semantic/coarse block, repeated `N` times:

```text
x0: [B, T, H]
y = LayerNorm(x0)
qkv = Linear(H -> 3H, bias=config.bias)(y)
q,k,v = split(qkv, dim=-1)                 # each [B, T, H]
q,k,v = view -> permute                    # [B, heads, T, head_dim]
k,v = DynamicCache.update(k, v, layer_idx) # [B, heads, T_total, head_dim]
a = softmax((q @ k^T) / sqrt(head_dim) + masks)
y = (a @ v) -> merge heads -> Linear(H -> H)
x1 = x0 + Dropout(y)
x2 = x1 + Dropout(Linear(4H -> H)(GELU(Linear(H -> 4H)(LayerNorm(x1)))))
```

Fine block is the same transformer block except `is_causal=False`, no cache is passed, and attention is bidirectional.

Fine input embedding path:

```text
input_ids: [B, T, 8]
embeds_i = Embedding_i(input_ids[:, :, i])  # [B, T, H]
inputs_embeds = sum(embeds_i for i <= codebook_idx)
hidden -> bidirectional blocks -> LayerNorm -> lm_heads[codebook_idx - n_codes_given]
```

## 6. Attention requirements

Semantic and coarse:

- Causal self-attention, MHA, no GQA/MQA.
- Head dims: 64 for `suno/bark*`, 8 for tiny-random.
- Eager cache tensor shape per layer: key/value `[B, num_heads, past_or_total_T, head_dim]`.
- Cached keys contain projections from hidden states that already include learned absolute position embeddings. There is no post-projection positional transform.
- Causal mask uses a registered lower-triangular `[1, 1, block_size, block_size]` buffer in eager attention.
- Padding attention masks are additive 4D masks from `create_bidirectional_mask`.
- FlashAttention2 path is source-supported and accepts padding mask; it must preserve causal alignment behavior for old flash-attn versions.

Fine:

- Noncausal self-attention over a fixed input buffer up to 1024 frames.
- No KV cache, no autoregressive token-by-token hidden-state reuse.

No cross-attention, no encoder-decoder cache, no sliding-window attention kernel in the transformer itself. The coarse stage uses an outer Python sliding generation loop, not a local-attention pattern.

## 7. Position encoding and custom math

Bark uses learned absolute position embeddings:

```python
past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
position_ids = torch.arange(past_length, seq_length + past_length).unsqueeze(0)
hidden_states = dropout(token_embeds + position_embedding(position_ids))
```

Custom controller math that DinoML must reproduce for end-to-end parity:

```python
semantic_to_coarse_ratio = coarse_rate_hz / semantic_rate_hz * n_coarse_codebooks
output_lengths = round(floor(nonpad_semantic_len * semantic_to_coarse_ratio / n_coarse_codebooks) * n_coarse_codebooks)
```

Alternating coarse logits mask:

```python
is_first = ((current_len - input_start_len) % 2) == 0
if is_first:
    allow [semantic_vocab_size : semantic_vocab_size + codebook_size]
else:
    allow [semantic_vocab_size + codebook_size : ]
```

No RoPE, M-RoPE, ALiBi, or relative position bias is present.

## 8. Preprocessing and input packing

`BarkProcessor` wraps `AutoTokenizer`; inspected tokenizer config is `BertTokenizer` with `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`, no special tokens added by default, `padding="max_length"`, `max_length=256`, and optional `attention_mask`.

Optional `voice_preset` can be:

- a name found in `speaker_embeddings_path.json`;
- a `.npz` file;
- a dict of NumPy arrays.

Required voice tensors:

| Key | Required rank | Runtime use |
|---|---:|---|
| `semantic_prompt` | 1 | truncated/padded to 256 and summed with text embeddings |
| `coarse_prompt` | 2 `[2, T]` | codebook-offset, transposed, flattened, trimmed to history |
| `fine_prompt` | 2 `[8, T]` | transposed to `[T, 8]`, prepended as fine history |

Generation-controller behavior:

- Semantic: suppresses all non-semantic output ids except EOS; optional `min_eos_p` can force EOS.
- Coarse: computes output length from semantic length, runs multiple windows of up to 60 tokens, and alternates the two acoustic codebook ranges.
- Fine: loops over outer 1024-frame buffers and inner codebooks, using argmax when `temperature is None or 1.0`, otherwise softmax + multinomial.
- Full `BarkModel.generate` routes kwargs by `semantic_`, `coarse_`, and `fine_` prefixes.

CPU/data-pipeline work can own tokenizer and speaker preset loading. GPU/runtime work starts at token ids/history tensors and includes prefix construction, transformer blocks, logits processors, sampling, and codebook packing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Bark QKV linear split

Source pattern: `Linear(H -> 3H)` then `.split(H, dim=2)`.

Replacement: one GEMM producing packed `[B*T, 3H]`, then three logical views/slices.

Preconditions: hidden dimension divisible by heads; output split order is all-Q, all-K, all-V contiguous blocks; bias handling matches `config.bias`.

Failure cases: do not use per-head interleaved packed layouts; do not fuse across `flash_attention_2` layout changes unless the consumer expects `[B,T,heads,D]`.

Parity test: compare Q/K/V tensors before cache update for semantic and coarse with random hidden states.

### Rewrite: MLP as two GEMM epilogues

Source pattern: `Linear(H -> 4H) -> GELU -> Linear(4H -> H) -> dropout`.

Replacement: GEMM + GELU epilogue where available, then GEMM.

Preconditions: inference dropout disabled; activation is exact PyTorch GELU module default.

Failure cases: training/dropout, nondefault activation not present in Bark source.

### Rewrite: last-token-only LM logits for decode

Source pattern: full `lm_head(hidden_states)` during autoregressive decode.

Replacement: for decode step, apply LM head only to `[B, 1, H]`.

Preconditions: caller does not request full logits/hidden states for every token; generation loop only needs next-token scores.

Failure cases: `return_dict_in_generate=True` with score collection still needs generated-step logits, not necessarily prompt logits.

### Rewrite: fine embedding sum

Source pattern: gather 8 embedding tables, concatenate with extra singleton axis, slice codebooks `:codebook_idx+1`, sum.

Replacement: directly accumulate embedding lookups for required codebook indices.

Preconditions: `codebook_idx` is static in each fine inner-loop call.

Failure cases: dynamic `codebook_idx` graph capture or custom `inputs_embeds`.

Layout constraints: no NHWC-style rewrite applies. Protect `[B,T,C_codebook]` axes and the final `[B,8,T]` codec contract with a no-layout-translation guard.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + QKV GEMM for semantic/coarse/fine blocks.
- QK softmax AV attention with KV cache for causal decode.
- GEMM + GELU for MLP first projection.
- Last-token LM head for semantic/coarse decode.

Medium priority:

- Prefix embedding construction for semantic history/text sum.
- Coarse logits masking as a fused scores processor.
- Fine multi-codebook embedding accumulation.
- Fine logits argmax over 1024-token codebook range.

Lower priority:

- EnCodec RVQ decode gather/sum.
- EnCodec ConvTranspose1d decoder fusions.
- CPU offload hooks from Transformers source; this is scheduling, not core graph parity.

## 11. Runtime staging plan

Stage 1: parse Bark nested configs and load weights for one submodel. Validate a single causal block.

Stage 2: semantic forward parity on prepared `inputs_embeds`, including absolute positions, causal mask, and LM logits.

Stage 3: semantic generate parity for short greedy/sampled outputs with `SuppressTokensLogitsProcessor` and optional `BarkEosPrioritizerLogitsProcessor`.

Stage 4: coarse generate parity with history preprocessing, codebook offsets, alternating logits masks, and multi-window controller.

Stage 5: fine model window parity for one `codebook_idx`, then full fine codebook loop. Stub codec with returned fine codes.

Stage 6: compose EnCodec decode or route to the separately audited EnCodec implementation for waveform output.

Stage 7: optimize attention/GEMM/fusions and add batching policy. Initial integration can stub voice presets as absent, fine generation, and codec decode while still being useful for semantic/coarse code-token work.

## 12. Parity and validation plan

- Random tensor parity for `BarkSelfAttention` eager path with and without cache.
- Single-block parity for semantic/coarse/fine blocks in fp32 tolerance `rtol=1e-4, atol=1e-5`.
- Full semantic forward parity on `suno/bark-small` config with short prompt and fixed weights.
- Decode cache parity: prefill `T=16`, then one-token decode must match full forward at the last position.
- Semantic generation parity for greedy and sampled runs with fixed seed, verifying token suppression ranges and EOS prioritizer.
- Coarse controller parity for synthetic semantic outputs with padding, verifying output length formula and alternating ranges.
- Fine generation parity for one short coarse sequence, first with `temperature=1.0` argmax, later sampled path.
- End-to-end code-token parity before waveform parity.
- Waveform parity only after EnCodec decode is integrated; tolerate fp32/fp16 convolution differences separately from discrete code parity.

No DinoML tests were run for this audit, by instruction.

## 13. Performance probes

- Processor throughput: tokenizer + optional `.npy` voice preset load.
- Semantic prefill latency by prompt length 1, 64, 257.
- Semantic decode tokens/sec with KV cache, batch sweep.
- Coarse window loop latency, especially repeated short `max_new_tokens=60` generation calls.
- Fine window throughput for `1024 x 8` codebook buffers.
- LM head time for large semantic input embedding vocab versus small output head.
- KV cache memory for semantic/coarse: `layers * 2 * B * heads * T * head_dim * dtype`.
- EnCodec decode latency isolated from transformer generation.
- Attention backend comparison: eager versus FlashAttention2 for prefill; decode may be dominated by small query length.

## 14. Skip/defer list

- Training and labels: source raises `NotImplementedError`.
- Gradient checkpointing.
- CPU offload hook behavior.
- Beam search and exotic `GenerationMixin` modes beyond greedy/sampling/top-k/top-p initially.
- FlashAttention2 optimized path until eager attention parity is solid.
- Optional voice presets for the first no-history integration.
- Fine model and EnCodec waveform decode for the first semantic/coarse milestone.
- Sampled fine `torch.multinomial` path if deterministic argmax is enough for initial parity.
- Full EnCodec encode path; Bark only needs quantizer decode + decoder for output.

## 15. Final implementation checklist

- [ ] Parse nested `BarkConfig` and `BarkGenerationConfig`.
- [ ] Load semantic/coarse/fine submodel weights with correct untied/tied head behavior.
- [ ] Implement learned position embeddings and causal absolute-position cache offsets.
- [ ] Implement Bark QKV packed split order.
- [ ] Implement causal MHA KV cache shape `[B, heads, T, head_dim]`.
- [ ] Implement bidirectional fine attention without cache.
- [ ] Implement semantic prefix construction with text offset, history pad, and infer token.
- [ ] Implement semantic suppress-token and EOS-prioritizer logits processors.
- [ ] Implement coarse history preprocessing, codebook offsets, output length math, and alternating logits processor.
- [ ] Implement fine codebook packing, embedding accumulation, window loop, and argmax path.
- [ ] Add no-layout-translation guards around fine codebook axes and codec input axes.
- [ ] Add one-block, full-forward, prefill/decode, and controller parity tests.
- [ ] Compose or stub EnCodec decode; later wire RVQ decode + decoder for waveform parity.
- [ ] Benchmark semantic prefill/decode, coarse window loop, fine loop, and codec decode separately.
