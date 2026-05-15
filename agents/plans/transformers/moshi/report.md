# Transformers Audit: moshi

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: native Transformers reports use `kmhf/hf-moshiko` and `kmhf/hf-moshika`; Kyutai native Moshi repos are external-runtime repos, not standard Transformers configs in this audit.

Config source: local `configuration_moshi.py`; representative HF configs from [kmhf/hf-moshiko](https://huggingface.co/kmhf/hf-moshiko) and [kmhf/hf-moshika](https://huggingface.co/kmhf/hf-moshika).

Source files inspected: `src/transformers/models/moshi/configuration_moshi.py`, `modeling_moshi.py`, `convert_moshi_transformers.py`; nested codec source `src/transformers/models/mimi/configuration_mimi.py`, `modeling_mimi.py`; generation special case in `src/transformers/generation/utils.py`; tokenizer conversion reference in `src/transformers/convert_slow_tokenizer.py`.

Any missing files or assumptions: no `processing_moshi.py`, `feature_extraction_moshi.py`, or tokenizer file lives in the Moshi model folder. The HF processor config uses `EncodecFeatureExtractor`. Native Kyutai `moshi` library checkpoints did not expose standard Transformers `config.json`; route them through conversion or a separate external-runtime audit.

## 2. High-level architecture

Moshi is a speech/text autoregressive generation stack with three separable parts:

```text
waveform/text preprocessing -> Mimi codec encode -> summed text+audio embeddings
  -> main causal decoder -> text logits + last hidden state
  -> depth decoder per generated text timestep -> multiple audio codebooks
  -> Mimi codec decode -> waveform
```

First useful DinoML target: main decoder plus audio-code generation parity, returning text logits and generated audio code tensors. Mimi waveform encode/decode should be a separately staged codec target because it owns causal Conv1d/ConvTranspose1d, codec transformers, RVQ nearest-codebook search, and optional streaming padding cache.

Stage decomposition:

- CPU/data pipeline: text tokenizer, raw audio loading/resampling/padding to 24 kHz mono, attention masks.
- Codec stage: Mimi `encode` maps `[B, C, samples]` to audio codes `[B, K, T]`; Mimi `decode` maps generated codes back to waveform.
- Main decoder: combines text embeddings with summed user+Moshi audio-code embeddings and runs a causal RoPE decoder.
- Depth decoder: for each main-decoder hidden state and text token, generates `num_codebooks` audio tokens using flexible per-codebook weights.
- Generation controller/session state: delay masks, blank user audio codes, `generated_audio_codes`, `last_hidden_state`, main KV cache, and depth-decoder KV cache.

## 3. Important config dimensions

| Field | Main Moshi decoder | Depth decoder | Mimi codec |
|---|---:|---:|---:|
| hidden size | 4096 | 1024 | 512 |
| layers | 32 | 6 | codec conv + 8 transformer layers |
| attention heads | 32 | 16 | 8 |
| KV heads | 32 | 16 | 8 |
| head dim | 128 | 64 | 64 |
| FFN dim | 22528 | 5632 | 2048 |
| activation | SiLU gated MLP | SiLU gated MLP | GELU MLP |
| text vocab | 32000 plus special 32000 embedding slot | 32000 plus special text slot | n/a |
| audio vocab/codebook size | 2048 plus special 2048 slot | 2048 logits | 2048 centroids |
| codebooks used by Moshi | 8 | 8 | config supports 32 quantizers |
| max positions | 3000 | 9 | 8000 |
| sliding window | 3000 | 8 | 250 in codec transformers |
| RoPE | default theta 10000 | disabled | default theta 10000 in codec transformers |
| cache | `DynamicCache`/sliding-window generation | `DynamicCache`/sliding-window generation | optional transformer cache plus conv padding cache |
| dtype in representative HF configs | bfloat16 | inherited/null in nested config | nested config says float32, weights in full model are bf16 |

Representative checkpoint/config sweep:

| Checkpoint | Source type | Native Transformers? | Operator-significant observations |
|---|---|---:|---|
| `kmhf/hf-moshiko` | HF `config.json` | yes | `MoshiForConditionalGeneration`; 32-layer main decoder, 6-layer depth decoder, Mimi config embedded, `torch_dtype=bfloat16`. |
| `kmhf/hf-moshika` | HF `config.json` | yes | Same inspected architecture/config values as `hf-moshiko`; likely different weights/persona rather than operator surface. |
| source defaults | `configuration_moshi.py` | yes | Same structural defaults; useful as random/small-less reference, but not a small model. |
| `kyutai/moshiko-pytorch-bf16` | HF repo metadata | no standard config found | External `library_name=moshi`; raw Transformers `config.json` unavailable. Needs conversion or external wrapper policy. |
| `kyutai/moshiko-candle-q8` / `mlx-q4/q8` | HF repo metadata | no | Quantized external runtime formats; not covered by native `modeling_moshi.py`. Treat as gated/config gap for DinoML. |

## 3a. Family variation traps

- `hidden_size == num_heads * head_dim` in observed configs, but source exposes `head_dim`; do not infer projection width from hidden size alone.
- `num_key_value_heads` can differ from attention heads in config, even though representative Moshi uses MHA. GQA repeat via `repeat_kv` is implemented.
- Main decoder uses RoPE; depth decoder explicitly disables RoPE and only spans 9 positions.
- Depth decoder has per-codebook flexible weights with physical weight tensors shaped `[num_codebooks, out, in]`; this is not a normal shared Linear.
- Audio embedding stitch is a sum of up to `2 * num_codebooks` embedding tables, not a scatter into a text sequence.
- Moshi generation mutates object fields (`generated_audio_codes`, `last_hidden_state`) and passes custom kwargs through `GenerationMixin`.
- Sliding-window cache is requested in generation config; FlashAttention2 rejects `StaticCache`.
- Mimi codec has `num_quantizers=32`, but Moshi requests `num_codebooks=8`; admission must reject Moshi configs asking for more codebooks than the codec supports.
- Kyutai native Candle/MLX/GGUF repos are not native Transformers model configs and may carry quantized/packed formats outside this report.
- Processor config uses Encodec-style feature extraction, not a Moshi-specific processor class.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for text ids and audio code ids with extra special slot.
- Sum of many embedding tensors over codebook channels.
- `cat`, `stack`, `transpose`, `reshape/view`, `squeeze/unsqueeze`, `expand`, `repeat_interleave`, `gather`, `index_select`, `where`.
- Causal mask creation and sliding-window mask behavior via Transformers cache/masking utilities.

Neural primitives:

- Bias-free Linear/GEMM for decoder projections: main `4096 -> 4096` q/k/v/o; depth `1024 -> 1024` q/k/v/o per codebook.
- `MoshiFlexibleLinear`: batched per-codebook weight select plus matmul, weight `[K, O, I]`.
- RMSNorm with fp32 accumulation and output cast back to input dtype.
- Gated MLP: Linear `H -> ffn_dim`, view split into two halves, `silu(first_half) * second_half`, Linear `ffn_dim/2 -> H`.
- LM heads: text `4096 -> 32000`; audio depth heads per codebook `1024 -> 2048`.

Attention primitives:

- Causal MHA/GQA with q/k RoPE in main decoder.
- Depth decoder causal attention without RoPE over sequence length `num_codebooks + 1`.
- Eager attention upcasts softmax to fp32; SDPA and FlashAttention2 backends are available.

Position/custom math:

- Default RoPE with theta 10000 for main decoder.
- No learned position embeddings in Moshi decoder blocks.

Generation/cache ops:

- Main per-layer KV cache for 32 layers.
- Depth decoder per-layer KV cache for 6 layers during nested generation.
- Sliding-window cache implementation expected by representative generation configs.
- Session fields for `generated_audio_codes` and `last_hidden_state`.
- Beam reordering for generated audio codes and last hidden state.

Preprocessing/codec-coupled ops:

- Mimi codec encode/decode: causal/asymmetric Conv1d, ConvTranspose1d, residual blocks, codec transformer, layer scale, RVQ codebook nearest-neighbor search by `torch.cdist`/argmin, codebook embedding decode and residual summation.
- Optional Mimi streaming: convolution padding cache plus encoder/decoder transformer caches.

Discrete codebook/tokenizer ops:

- Audio code ids are integer tokens in `[0, audio_vocab_size)`, with special BOS/PAD id equal to `audio_vocab_size`.
- Text generation uses special BOS/PAD id equal to `vocab_size` in representative generation config.
- Depth decoder generation vocab must use `audio_vocab_size`, handled by a generation-utils special case.

## 5. Layer/block breakdown

Main decoder block, repeated 32 times:

```text
x: [B, T, 4096]
h = RMSNorm(x)
q = Linear(4096 -> 32*128, bias=False)(h)
k = Linear(4096 -> kv_heads*128, bias=False)(h)
v = Linear(4096 -> kv_heads*128, bias=False)(h)
q,k = RoPE(q,k, position_ids)
k,v = cache.update(k,v, layer_idx)
k,v = repeat_kv(k,v) if GQA
a = causal_attention(q,k,v, fp32 softmax in eager path)
x = x + Linear(4096 -> 4096, bias=False)(a)
h = RMSNorm(x)
u = Linear(4096 -> 22528, bias=False)(h).view(..., 2, 11264)
m = silu(u[...,0,:]) * u[...,1,:]
x = x + Linear(11264 -> 4096, bias=False)(m)
```

Depth decoder block, repeated 6 times:

```text
x: [B*T_main, <=9, 1024]
codebook_idx = arange(seq_len) + past_seen_tokens
h = RMSNorm(x)
q,k,v,o use MoshiFlexibleLinear with codebook_idx-selected weights
attention is causal, no RoPE, typically sliding window 8
x = residual + attention_out
h = RMSNorm(x)
gated MLP uses per-codebook flexible weights
x = residual + mlp_out
```

Conditional generation embedding path:

```text
if waveforms are supplied: audio_codes = Mimi.encode(input_values, num_quantizers=8)
audio_codes = cat([moshi_audio_codes, user_audio_codes], dim=1)
audio_embeds = sum(embed_tokens[codebook](audio_codes[:, codebook]) for each channel)
text_embeds = decoder.model.embed_tokens(input_ids)
inputs_embeds = text_embeds + audio_embeds
```

## 6. Attention requirements

Main decoder:

- Causal self-attention only; no cross-attention.
- MHA in representative configs: 32 q heads, 32 KV heads, head dim 128. Source supports GQA/MQA-style `num_key_value_heads < num_attention_heads`.
- Query/key/value layouts are projected as `[B,T,H] -> [B,heads,T,D]`; FlashAttention path transposes to `[B,T,heads,D]`.
- Masking comes from `create_causal_mask`; generation config requests `cache_implementation="sliding_window"`.
- KV cache stores post-RoPE keys and values before `repeat_kv`.
- Eager path does matmul, mask add, fp32 softmax, dropout, matmul.
- SDPA path repeats KV before `scaled_dot_product_attention`.
- FlashAttention2 path updates cache, keeps unrepeated KV shape if backend supports GQA-style heads, and rejects `StaticCache`.

Depth decoder:

- Causal self-attention over a short sequence of one text token plus audio codebook positions.
- No RoPE.
- Flexible per-codebook q/k/v/o weights selected by absolute codebook position with `past_seen_tokens`.
- Generation length is fixed by config to `num_codebooks + 1` in representative depth generation config.
- Cache ABI is separate from main decoder cache; DinoML should model it as a nested short-sequence decoder cache, not as the same layer stack.

Codec Mimi transformers:

- Mimi has encoder and decoder transformer caches, plus sliding window 250, but first Moshi text/audio-code target can treat Mimi as external/staged.

## 7. Position encoding and custom math

Main RoPE is the Llama-style half-rotate with fp32 frequency computation:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2) / head_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
emb = cat([freqs, freqs], dim=-1)
q = q * cos(emb) + rotate_half(q) * sin(emb)
k = k * cos(emb) + rotate_half(k) * sin(emb)
```

Precompute opportunity: default inv frequencies and cos/sin tables can be precomputed up to the admitted sliding-window/max context when `rope_type="default"`. Dynamic RoPE variants are wired through shared Transformers `ROPE_INIT_FUNCTIONS`, so non-default `rope_parameters` should be rejected initially unless separately audited.

Delay pattern mask:

```python
shifted = full([B, K, max_length], -1)
shifted[:, 0, :min(T, max_length - 1)] = codes[:, 0, :keep]
shifted[:, 1:, 1:keep+1] = codes[:, 1:, :keep]
shifted[:, 1:, 0] = audio_vocab_size
shifted[:, 0, -1] = audio_vocab_size
masked = where(pattern_mask == -1, generated_codes, pattern_mask)
```

This mask is generation-controller ABI, not a neural op to hide inside attention.

## 8. Preprocessing and input packing

Processor/preprocessor contract from representative HF configs:

- `feature_extractor_type`: `EncodecFeatureExtractor`.
- `sampling_rate`: 24000.
- `feature_size`: 1.
- `padding_side`: right.
- `padding_value`: 0.0.
- `return_attention_mask`: true.
- `chunk_length_s` and `overlap`: null.

Waveform inputs to model source are `[B, 1, audio_sequence_length]`; Mimi source accepts one or two channels but Moshi configs use mono. If raw waveform is supplied, Moshi calls `audio_encoder.encode(..., num_quantizers=8)` and expects codes `[B, 8, T]`. If audio codes are supplied, codec encode is bypassed.

Sequence-length coupling: `_check_and_maybe_initialize_inputs` requires text token length, user audio code length, and Moshi audio code length to match. For waveform inputs, it converts sample length to code length with `ceil(samples * frame_rate / sampling_rate)`. Representative frame rate is 12.5 Hz, so one audio code timestep covers 1920 samples.

No placeholder-token scatter is used. Audio conditioning is summed into text embeddings positionwise. DinoML can lower this to bounded codebook-channel embedding gathers plus elementwise sums with guards:

- user and Moshi streams either both present or use unconditional defaults;
- codebooks exactly `num_codebooks`;
- code sequence length equals text sequence length after waveform-to-code conversion;
- code ids permit special id `audio_vocab_size` only where delay/unconditional logic uses it.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MoshiFlexibleLinear to grouped/batched GEMM

Source pattern: `weight[K,O,I]`, input `[B,K,I]`, optional `layer_idx`, compute `[B,K,O]` by selected per-codebook matmul.

Replacement: grouped GEMM over codebook groups, or reshape to batched GEMM with `batch=K` and broadcast user batch.

Preconditions:

- `layer_idx is None` or statically equals `[0..K-1]` for full depth generation.
- Input sequence dimension equals number of selected codebook weights.
- Weight layout remains `[K,O,I]` from converted checkpoints.

Failure cases: arbitrary runtime `layer_idx` tensor; decode steps with partial codebook prefixes may require a small indexed-weight GEMM path.

Parity test sketch: random `B,K,I`, compare grouped GEMM against `torch.matmul(x[:, :, None, :], weight.transpose(1,2)[None]).squeeze(2)`.

### Rewrite: summed codebook embeddings

Source pattern: Python `sum(embed_tokens[i](codes[:, i]) for i in channels)`.

Replacement: gather each codebook table then fused add tree over `[B,T,H]`.

Preconditions: channel count statically known as `K` or `2K`; code ids within table including special final slot.

Failure cases: only user or only Moshi stream uses offset table range; both-stream path uses `[moshi, user]` concatenation.

### Rewrite: depth decoder fixed-length generation

Source pattern: `depth_decoder.generate` with `min_length=max_length=num_codebooks+1`.

Replacement: unrolled 9-step short decoder loop for `K=8`, or one preallocated short sliding-window cache.

Preconditions: generation config depth `min_length=max_length=K+1`, no beam search for first target, fixed `num_codebooks`.

Failure cases: custom generation config changes depth max length, beams, or sampling policy.

### Rewrite: last-token text logits

Source pattern: `logits_to_keep` slices hidden states before LM head.

Replacement: for decode, run text LM head on only final hidden state.

Preconditions: generation decode only needs next-token logits; no full logits requested.

Failure cases: training/loss, `output_logits` for all timesteps, or prompt-logit parity tests.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm fp32 accumulation + scale for `4096` and `1024`.
- Bias-free GEMM families for main decoder and depth decoder.
- Gated SiLU MLP: packed `Linear -> split -> silu*mul -> Linear`.
- RoPE + attention prefill/decode with KV cache and sliding-window admission.
- Codebook embedding-sum kernel for `2K` audio channels.
- `MoshiFlexibleLinear` grouped/batched GEMM for depth decoder.

Medium priority:

- Short fixed-length depth decoder generation loop.
- Delay-pattern mask and code extraction as generation-controller kernels.
- Last-token-only text LM head.
- Beam audio-code gather/reorder if beam search is admitted.

Lower priority:

- Full Mimi codec acceleration, including causal Conv1d/ConvTranspose1d and RVQ nearest-centroid search.
- FlashAttention2-compatible layout path to avoid repeated transposes.
- Non-default RoPE variants.

## 11. Runtime staging plan

Stage 1: parse `MoshiConfig`, nested `MoshiDepthConfig`, and embedded Mimi config; reject external Kyutai-native/non-Transformers checkpoints without a converted config.

Stage 2: load main decoder and run text-only `MoshiForCausalLM` prefill/decode parity.

Stage 3: implement audio-code embedding stitch with supplied `user_audio_codes` and `moshi_audio_codes`; bypass Mimi waveform encode/decode.

Stage 4: implement main conditional generation state: main KV cache, `last_hidden_state`, generated audio-code buffer, delay masks, blank-user-code input supplied as codes.

Stage 5: implement depth decoder with `MoshiFlexibleLinear`, fixed 9-token audio-code generation, and separate depth KV cache.

Stage 6: integrate Mimi decode as optional codec stage for waveform output. Keep raw waveform encode and streaming padding cache deferred until the codec audit is complete.

Stage 7: optimize attention, grouped GEMMs, embedding sums, and generation-controller kernels; add beam support only after greedy/sampling parity.

## 12. Parity and validation plan

- Config admission tests: accepted `kmhf/hf-moshiko`/`hf-moshika`; reject `num_codebooks > audio_encoder_config.num_codebooks`; reject non-default `rope_parameters` initially.
- Unit parity for RMSNorm, RoPE, gated MLP, and `MoshiFlexibleLinear`.
- Main decoder single-layer and full-layer random hidden/input-id parity in fp32 and bf16.
- Audio embedding stitch parity with supplied codes, including special `audio_vocab_size` code.
- Delay mask parity for prompts of length 1, 2, and near `max_length`.
- Depth decoder parity for one hidden state and fixed 9-token generation with deterministic greedy settings.
- End-to-end code parity: supplied text ids plus supplied user/Moshi audio codes -> text logits and audio code sequence.
- Later codec parity: Mimi encode/decode code shapes and waveform reconstruction against Transformers for short mono clips.
- Suggested tolerances: fp32 `1e-4` absolute for block-level ops; bf16 `2e-2` for full decoder logits before sampling; code sequence parity should be exact under deterministic sampling/greedy.

## 13. Performance probes

- Main decoder prefill throughput over `T={1, 16, 256, 3000}`.
- Decode tokens/sec with sliding-window cache at batch sizes `1, 4, 16`.
- Depth decoder overhead per generated text token; compare eager 9-step loop vs grouped-GEMM lowering.
- Embedding-stitch bandwidth for `2 * 8` codebook tables.
- KV cache memory for main 32 layers plus depth 6 layers.
- Mimi encode/decode throughput separately from decoder generation.
- RVQ nearest-codebook search cost for 8 vs 32 quantizers.
- End-to-end requests/hour split into processor, Mimi encode, main decode, depth decode, Mimi decode.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Beam search and beam audio-code reordering for first integration.
- Raw waveform encode/decode inside the first decoder target.
- Mimi streaming padding cache and codec transformer streaming caches.
- Native Kyutai Candle/MLX/GGUF/quantized repos.
- StaticCache with FlashAttention2, since source rejects it.
- Non-default RoPE types.
- Multi-GPU/tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `MoshiConfig`, `MoshiDepthConfig`, and nested Mimi config.
- [ ] Add checkpoint admission for native Transformers `moshi` configs only.
- [ ] Load main decoder, depth decoder, codebook embeddings, and flexible-linear weights.
- [ ] Implement RMSNorm, default RoPE, causal attention with sliding-window cache, and gated SiLU MLP.
- [ ] Implement `MoshiFlexibleLinear` lowering.
- [ ] Implement text/audio-code embedding stitch with code-id guards.
- [ ] Implement delay-pattern mask utilities and generated-audio-code session state.
- [ ] Implement depth decoder fixed-length audio-code generation.
- [ ] Add supplied-code conditional generation parity before waveform codec parity.
- [ ] Audit and stage Mimi codec encode/decode separately.
- [ ] Benchmark main decoder, depth decoder, embedding stitch, and codec stages independently.
