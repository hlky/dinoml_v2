# Transformers Audit: granite_speech_plus

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ibm-granite/granite-speech-4.1-2b-plus
Config source: HF config.json, processor_config.json, generation_config.json
Primary runtime target: audio-conditioned autoregressive text generation
Source files inspected: see _sources/source_notes.md
Any missing files or assumptions: preprocessor_config.json is absent for the Plus checkpoint; processor_config.json is authoritative for audio preprocessing. No code was imported or executed.
```

`modeling_granite_speech_plus.py` and `configuration_granite_speech_plus.py` are generated from `modular_granite_speech_plus.py`. Future upstream source edits should target the modular file, but this audit uses the generated file for the complete expanded operator surface.

The Plus checkpoint is public and not gated in the HF API. The HF model API reports `granite_speech_plus`, `automatic-speech-recognition`, Apache-2.0, safetensors, and about 2.11B BF16 parameters. Parameter count and dtype are HF metadata facts, not source-derived behavior.

## 2. High-level architecture

Granite Speech Plus is a composite audio encoder + Q-Former projector + Granite causal LLM.

```text
raw audio -> GraniteSpeechFeatureExtractor -> input_features [B, T_enc, 160]
input_features -> Conformer CTC encoder -> hidden [B, T_enc, 1024 or 2048]
hidden -> windowed BLIP-2 Q-Former projector -> audio_embeds [B, T_audio, 2048]
text input_ids with <|audio|> placeholders -> token embeddings
masked audio embedding stitch -> Granite causal LM prefill -> decode with KV cache -> logits/sampling
```

Stage decomposition:

- CPU/data pipeline: waveform validation/padding, torchaudio mel spectrogram, log/clamp normalization, frame stacking, placeholder expansion, tokenizer, `input_features_mask`.
- Audio encoder: Conformer stack with local block attention, depthwise Conv1d, mid-layer CTC-style softmax feedback, and optional hidden-state concatenation.
- Projector: window audio hidden states into blocks of 15 frames; run 3 learned query tokens through a 2-layer BLIP-2 Q-Former with cross-attention to each block; project 1024 -> 2048.
- Prefix construction: replace `<|audio|>` token embeddings by projected audio embeddings, after mask/count validation.
- LLM prefill/decode: delegated to `GraniteForCausalLM`; audio features are passed only on first generation iteration when cache is used.

The encoder/projector output is independently cacheable per audio sample before text prefill. The LLM KV cache is separate and begins after the stitched multimodal prompt is fed to Granite.

## 3. Important config dimensions

Primary checkpoint facts from `ibm-granite/granite-speech-4.1-2b-plus/config.json`:

| Field | Value |
| --- | --- |
| `model_type` | `granite_speech_plus` |
| `architectures` | `GraniteSpeechPlusForConditionalGeneration` |
| checkpoint dtype | `bfloat16` |
| `audio_token_index` | `100352` |
| `has_lora_adapter` | `false` |
| `window_size` / `downsample_rate` | `15` / `5` |
| encoder layers / hidden / heads | `16` / `1024` / `8` |
| encoder `dim_head` / attention width | `128` / `1024` |
| encoder input_dim | `160` stacked mel features |
| encoder FFN intermediate | `1024 * 4 = 4096` |
| encoder conv inner | `1024 * 2 = 2048` |
| encoder `context_size` / `max_pos_emb` | `200` / `512` |
| encoder `conv_kernel_size` | `15` |
| encoder `output_dim` | `348` |
| encoder `cat_hidden_layers` | `[3]` |
| projector type | `blip_2_qformer` |
| projector hidden / encoder_hidden_size | `1024` / `2048` |
| projector layers / heads | `2` / `16` |
| projector intermediate | `4096` |
| projector cross_attention_frequency | `1` |
| projector activation | GELU |
| text model type | `granite` |
| text layers / hidden | `40` / `2048` |
| text heads / KV heads / head_dim | `16` / `4` / inferred `128` |
| text intermediate | `4096` |
| text max positions | `4096` |
| text vocab | `100353` |
| text RoPE | default, theta `10000` |
| text multipliers | embedding `12`, attention `0.0078125`, residual `0.22`, logits scaling `8` |
| text biases | no attention bias, no MLP bias |
| text cache | `use_cache=true` |
| tied word embeddings | `true` |

Processor facts from `processor_config.json`:

| Field | Value |
| --- | --- |
| processor | `GraniteSpeechProcessor` |
| audio processor | `GraniteSpeechFeatureExtractor` |
| sampling_rate | `16000` |
| STFT | `n_fft=512`, `win_length=400`, `hop_length=160` |
| mel bins | `80` |
| projector window/downsample | `15` / `5` |
| audio token string | `<|audio|>` |

Representative config sweep:

| Model id | In scope? | model_type | Encoder | Plus concat | Projector encoder hidden | Text hidden/layers | Text heads/KV | Audio token | LoRA flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ibm-granite/granite-speech-4.1-2b-plus` | yes | `granite_speech_plus` | 16 x 1024 | `[3]` | 2048 | 2048 / 40 | 16 / 4 | 100352 | false |
| `ibm-granite/granite-speech-4.1-2b` | related | `granite_speech` | 16 x 1024 | none | 1024 | 2048 / 40 | 16 / 4 | 100352 | false |
| `ibm-granite/granite-4.0-1b-speech` | related | `granite_speech` | 16 x 1024 | none | 1024 | 2048 / 40 | 16 / 4 | 100352 | false |
| `ibm-granite/granite-speech-3.3-2b` | related | `granite_speech` | 16 x 1024 | none | 1024 | 2048 / 40 | 32 / 8 | 49159 | true |
| `ibm-granite/granite-speech-3.3-8b` | related | `granite_speech` | 16 x 1024 | none | 1024 | 4096 / 40 | 32 / 8 | 49159 | true |

Only the first row uses the Plus class. The other rows are included to expose traps inherited from Granite Speech but should not be treated as Plus coverage.

## 3a. Family variation traps

- Plus-specific behavior is `encoder_config.cat_hidden_layers`: selected intermediate encoder states are concatenated with the final encoder state on the feature axis before the projector. The 4.1 Plus checkpoint uses layer 3, so projector input width is `1024 * 2 = 2048`.
- `projector_config.encoder_hidden_size` must equal `encoder_hidden_dim * (len(cat_hidden_layers) + 1)` when concat is enabled; the config constructor rejects mismatches.
- The processor computes placeholder count from raw audio length, not from the post-padded tensor. This must match projector output length after encoder frame stacking and projector windowing.
- `input_features_mask` is shaped after the projector, not like `input_features`; source comments call out this naming mismatch.
- `masked_scatter` is broad in PyTorch, but the processor creates a stricter ordered replacement pattern: each `<|audio|>` placeholder is expanded to the projected feature count for each audio sample, then the model fills matching positions in row-major mask order.
- The Q-Former is kept in fp32 by BLIP-2 source behavior; encoder hidden states are cast to the query dtype inside the Q-Former.
- `has_lora_adapter` is config-dependent. Older Granite Speech 3.3 checkpoints set it true; the inspected Plus checkpoint sets it false. Source toggles PEFT adapters during `generate` only if PEFT is loaded.
- Text config is a nested Granite LLM. Its operator surface is not inferable from `hidden_size` alone: it uses GQA (`num_key_value_heads=4`), nonstandard embedding/attention/residual/logit multipliers, tied embeddings, and RoPE.
- The Conformer attention is block-local with Shaw relative bias and forces PyTorch SDPA math backend in source. Do not lower it as ordinary global encoder MHA without block and bias guards.
- Layout-sensitive encoder conv code uses `[B, T, C] -> permute(0,2,1) -> Conv1d -> permute(0,2,1)`. A layout pass may optimize this region, but initial translation should preserve axes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape arithmetic: ceil-div for audio sequence blocks, projector blocks, and local attention blocks.
- Padding: waveform batch pad to max length in processor, mel time even-crop, encoder attention pad to `context_size`, projector pad to `window_size`.
- `reshape`/`view`, `transpose`, `permute`, `contiguous`, `chunk`, `cat`, `clone`, `where`.
- Boolean masks, comparison, `masked_fill`, `masked_scatter`, boolean indexing for `audio_features[input_features_mask]`.
- Embedding lookup for text tokens, audio placeholder checks, relative position embedding lookup.

Preprocessing-coupled ops:

- Torchaudio `MelSpectrogram`: STFT/FFT, power/magnitude behavior inherited from torchaudio defaults, mel filterbank, output `[B, 80, frames]`.
- `transpose(-1,-2)`, clamp min `1e-10`, `log10`, global `amax` over time and mel, dynamic clamp floor `mx - 8.0`, divide by 4, add 1.
- Drop last mel frame if odd; stack consecutive frame pairs into feature width `160`.

Audio encoder primitives:

- Linear `160 -> 1024` with bias.
- 16 Conformer blocks:
  - LayerNorm over 1024.
  - Feed-forward `1024 -> 4096 -> 1024` with SiLU and residual scale `0.5`.
  - Local block MHA: Q `1024 -> 1024` no bias; packed KV `1024 -> 2048` no bias; output `1024 -> 1024` with bias.
  - Relative position embedding table `[2 * max_pos_emb + 1, dim_head] = [1025, 128]`.
  - Positional score einsum `b m h c d, c r d -> b m h c r`.
  - SDPA with additive relative score mask and separate scale.
  - Conv module: LayerNorm, pointwise Conv1d `1024 -> 4096`, GLU along channel to 2048, depthwise Conv1d kernel 15 groups 2048, BatchNorm1d 2048, SiLU, pointwise Conv1d `2048 -> 1024`.
  - Post LayerNorm.
- Mid-layer feedback at layer `num_layers // 2`: clone, Linear `1024 -> 348`, Softmax last dim, Linear `348 -> 1024`, residual add.
- Plus concat: `cat([hidden_at_layer_3, final_hidden], dim=-1)` for checkpoint, output width 2048.

Projector primitives:

- Learned query parameter `[1, 3, 1024]`; broadcast to each `(batch * nblocks)` Q-Former call by source semantics.
- Window pad and reshape `[B, T, 2048] -> [B * ceil(T/15), 15, 2048]`.
- BLIP-2 Q-Former, 2 layers, 16 heads, hidden 1024, intermediate 4096.
- Each Q-Former layer has self-attention over 3 query tokens and cross-attention to 15 encoder frames because `cross_attention_frequency=1`.
- Cross-attention K/V project from encoder width 2048 to 1024; self-attention Q/K/V project 1024 to 1024.
- Final Linear `1024 -> 2048`.
- Reshape projector output to `[B, nblocks * 3, 2048]`.

Text LLM primitives:

- Delegated Granite decoder: tied embeddings, RMSNorm, GQA causal attention with RoPE, SwiGLU MLP, residual multiplier, logits scaling.
- For the primary checkpoint: 40 layers, hidden 2048, MLP 4096, Q width 2048, KV width `4 * 128 = 512`, output width 2048, vocab 100353.

Generation/cache ops:

- Granite DynamicCache update per layer for K/V after RoPE.
- Generation prepares audio inputs only on first iteration when cache is enabled; decode steps should consume only text ids/embeds plus KV cache.
- `logits_to_keep` may slice logits to last N tokens or tensor indices.

LoRA/adapters:

- Source toggles PEFT adapters in `generate` if PEFT is available and a PEFT config is loaded. The audited Plus checkpoint has `has_lora_adapter=false`, so first integration can reject active adapter configs or route them to a separate LoRA weight-overlay audit.

## 5. Layer/block breakdown

Feature extractor:

```text
audio [B, samples] float -> MelSpectrogram -> mel [B, 80, F]
logmel = log10(clamp(transpose(mel), min=1e-10))
mx = amax(logmel, time/mel)
logmel = (max(logmel, mx - 8.0) / 4) + 1
if F is odd: drop last frame
input_features = reshape pairs -> [B, F_even / 2, 160]
```

Conformer block, repeated 16 times:

```text
x = x + 0.5 * FFN(LayerNorm(x))
x = x + LocalRelPosAttention(LayerNorm(x), context_size=200)
x = x + ConvModule(LayerNorm(x), Conv1d/GLU/depthwise/BatchNorm/SiLU/Conv1d)
x = x + 0.5 * FFN(LayerNorm(x))
x = LayerNorm(x)
```

Mid-layer encoder feedback at block 8 for 16 layers:

```text
mid = Linear_1024_to_348(clone(x))
x = x + Linear_348_to_1024(Softmax(mid, dim=-1))
```

Plus encoder output:

```text
if cat_hidden_layers=[3]:
    encoder_out = cat([hidden_after_layer_3, final_hidden], dim=-1)  # [B,T,2048]
else:
    encoder_out = final_hidden  # [B,T,1024]
```

Projector:

```text
nblocks = ceil(T / 15)
hidden = pad_time_to(nblocks * 15)
hidden = view(B * nblocks, 15, encoder_hidden_size)
query = learned [1, 3, 1024]
q = BLIP2QFormer(query_embeds=query, encoder_hidden_states=hidden)
audio_embeds = Linear_1024_to_2048(view(q.last_hidden_state, B, nblocks * 3, 1024))
```

Granite LLM block, repeated 40 times:

```text
res = x
x = RMSNorm(x)
q = Linear_2048_to_2048(x)
k = Linear_2048_to_512(x)
v = Linear_2048_to_512(x)
q,k = RoPE(q,k)
k,v = cache_update(k,v)
x = res + Attention(q,k,v, GQA repeat, causal_mask) * 0.22
res = x
x = RMSNorm(x)
x = down_proj(SiLU(gate_proj(x)) * up_proj(x))
x = res + x * 0.22
```

## 6. Attention requirements

Audio encoder attention:

- Noncausal local self-attention inside independent blocks of `context_size=200`.
- MHA: 8 heads, `dim_head=128`, Q/K/V all width 1024.
- Query/key/value tensors reshape to `[B, num_blocks, heads, context_size, dim_head]`.
- Shaw relative positional additive score tensor has shape `[B, num_blocks, heads, context_size, context_size]`.
- Last padded block gets a boolean mask applied to the relative score tensor before SDPA.
- No KV cache; this is an encoder branch and can be cached only as completed audio embeddings.

Q-Former projector attention:

- Noncausal query self-attention over 3 learned query tokens.
- Cross-attention from query tokens to each 15-frame audio block.
- 16 heads, head dim 64, hidden 1024.
- Cross-attention K/V source width is 2048 for Plus checkpoint, projected to 1024.
- No autoregressive KV cache. Q-Former outputs can be cached per audio block/sample.
- Source BLIP-2 Q-Former disables SDPA/Flash/Flex attention and uses eager matmul-softmax-matmul.

Granite LLM attention:

- Causal self-attention with GQA: 16 query heads, 4 KV heads, head dim 128, repeat factor 4.
- RoPE is applied to Q/K before cache update; cached K stores position-encoded keys.
- Attention multiplier is `0.0078125`, not the default `1/sqrt(head_dim)` unless the configured attention backend reproduces source behavior.
- Granite source can use attention backend dispatch, with eager fallback doing repeat-KV, matmul, mask add, fp32 softmax, dropout, matmul.
- KV cache per layer: K and V logical shape before repeat `[B, 4, S, 128]`; after repeat for attention `[B, 16, S, 128]`.

## 7. Position encoding and custom math

Audio Conformer relative position:

```python
seq = arange(context_size)
relpos_dist = seq[:, None] - seq[None, :]
attention_dists = clamp(relpos_dist, -context_size, context_size) + max_pos_emb
rel = rel_pos_emb(attention_dists)  # [context, context, dim_head]
pos_attn = einsum("b m h c d, c r d -> b m h c r", q, rel) * scale
```

`attention_dists` is a persistent-false buffer and can be precomputed for a fixed `context_size`. The relative embedding lookup remains learned weights.

Granite RoPE:

```python
def apply_granite_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Cos/sin depend on `position_ids`, which include `past_seen_tokens` during decode. RoPE tables can be precomputed by max position and gathered per batch.

Custom Granite scalar math:

- Input embeddings are multiplied by `embedding_multiplier=12`.
- Residual branches are added as `residual + branch * residual_multiplier`.
- Final logits are divided by `logits_scaling=8`.

## 8. Preprocessing and input packing

Waveform contract:

- Input is a floating point torch tensor or numpy array, one sample `[samples]`, collated `[B, samples]`, or a sequence of tensors/arrays.
- Sampling rate is configured as 16 kHz; source does not resample in the feature extractor.
- Multi-sample tensors are assumed same length; sequence inputs are padded with zeros to max length.
- The source error message says audio should be floating point between 0 and 1, but code only checks floating dtype.

Feature tensor contract:

- `input_features` shape is `[B, T_enc, 160]`, where `T_enc = floor((raw_length // hop + 1) / 2)` after dropping an odd mel frame.
- The feature extractor returns `audio_embed_sizes`, then the processor removes it before model call.
- Projected audio token count is `ceil(T_enc / window_size) * (window_size // downsample_rate)`, which is `ceil(T_enc / 15) * 3`.
- `input_features_mask` is `[B, max_projected_audio_tokens]`, true for valid projected audio embeddings.

Placeholder packing:

- Processor replaces each `<|audio|>` occurrence in text with exactly `audio_embed_sizes[i]` copies of the audio token string, in sample traversal order.
- Model obtains embeddings for text after replacing audio token ids with token id 0 to avoid out-of-vocab embedding lookup.
- Audio embeddings are optionally filtered by `input_features_mask`, flattening valid rows in PyTorch boolean-index order.
- `get_placeholder_mask` checks that selected embedding element count equals `audio_features.numel()`.
- `masked_scatter` writes audio features into the expanded audio placeholder positions.

DinoML should lower this as a guarded ordered row-copy, not as general boolean scatter, when:

- `input_ids` are available,
- audio placeholder counts match `input_features_mask.sum()`,
- placeholder positions are row-major ordered as produced by the processor,
- audio features are contiguous `[total_audio_tokens, hidden]` or `[B, max_audio_tokens, hidden]` plus mask.

## 9. Graph rewrite / lowering opportunities

### Rewrite: processor `masked_scatter` -> ordered audio row copy

Source pattern:

```text
inputs_embeds = embedding(where(input_ids == audio_id, 0, input_ids))
audio_features = audio_features[input_features_mask]  # optional
mask = (input_ids == audio_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(mask, audio_features)
```

Replacement:

```text
copy base embeddings
for each true audio placeholder row in row-major order:
    copy next audio feature row to that embedding row
```

Preconditions:

- `input_ids` path, not `inputs_embeds` placeholder-detection path.
- `count(input_ids == audio_token_id) == flattened_audio_feature_rows`.
- Last dimension equals text hidden size 2048.
- Placeholder positions are not reordered after tokenization.

Failure cases:

- Caller supplies `inputs_embeds` directly.
- Audio feature mask is not rank 2 boolean or does not match projector output.
- Placeholder count mismatch.

Parity test sketch: build tiny embedding matrix, synthetic audio rows, arbitrary placeholder positions across a batch, compare row-copy output to PyTorch `masked_scatter`.

### Rewrite: Conformer Conv1d pointwise -> time-major GEMM

Source pattern:

```text
x [B,T,C] -> permute [B,C,T] -> Conv1d kernel=1 -> GLU/depthwise/down Conv1d
```

Replacement for pointwise convs:

```text
flatten [B*T,C] -> Linear(C,out) -> reshape
```

Preconditions:

- `kernel_size == 1`, stride 1, padding 0, dilation 1, groups 1.
- Preserve PyTorch Conv1d weight orientation `[out_channels, in_channels, 1]`.
- Bias included for up/down convs.

Failure cases:

- Do not apply to depthwise kernel 15.
- Do not cross BatchNorm unless eval-mode folding is explicitly validated.

### Rewrite: depthwise Conv1d + BatchNorm eval fold

Source pattern:

```text
F.pad -> depthwise Conv1d(groups=C, bias=False) -> BatchNorm1d(C) -> SiLU
```

Replacement:

```text
F.pad -> depthwise Conv1d(groups=C, bias=True with folded BN) -> SiLU
```

Preconditions:

- Inference/eval mode.
- BatchNorm running_mean/running_var/weight/bias are present and frozen.
- Exact epsilon from module defaults is preserved.

Failure cases:

- Training mode or missing BN buffers.
- Dynamic quantized weights without a fold/materialization policy.

### Rewrite: local Conformer attention blockification

Source pattern:

```text
pad T to ceil(T/context) * context
reshape [B,T,H] -> [B,blocks,context,heads,dim]
attention independently per block with additive relative bias
slice back to original T
```

Replacement:

```text
BlockPartition -> per-block attention kernel with relative score bias -> BlockMerge
```

Preconditions:

- `context_size` fixed and <= `max_pos_emb`.
- No attention across blocks.
- Padded last block mask matches source fill to `-finfo(dtype).max`.

Failure cases:

- Changing block order or treating as global attention.
- Dropping relative score term.

### Rewrite: projector Q-Former fixed small query specialization

Source pattern:

```text
for each 15-frame block: 3 learned queries attend to 15 audio frames through 2 Q-Former layers
```

Replacement:

```text
fused small-Q cross-attention kernels or batched GEMM attention with Q=3, K=15
```

Preconditions:

- `window_size=15`, `downsample_rate=5`, `num_queries=3`.
- `cross_attention_frequency=1`; each layer has cross-attention.
- Q-Former text-input path disabled.

Failure cases:

- Different projector config or `use_qformer_text_input=true`.

### Rewrite: Granite last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :]) / logits_scaling
```

Replacement:

```text
gather kept hidden rows -> GEMM to vocab -> scale
```

Preconditions:

- `logits_to_keep` is integer 1 for decode or a validated index tensor.
- Tied output weight alias with embeddings preserved.

Failure cases:

- Full prefill logits needed for loss or scoring.

## 10. Kernel fusion candidates

Highest priority:

- Granite decoder block: RMSNorm, Q/K/V GEMMs, RoPE, GQA FlashAttention/SDPA with KV cache, SwiGLU MLP, residual multiplier.
- Audio stitch row-copy: replace broad scatter with guarded indexed copy to keep prompt construction cheap and graph-friendly.
- Conformer local relative attention: block-local attention with Shaw score bias is likely the main audio encoder custom kernel.
- Conformer Conv module: pointwise conv GEMM + GLU + depthwise Conv1d + folded BatchNorm + SiLU.

Medium priority:

- Feature extraction pipeline if DinoML owns preprocessing on GPU: STFT/mel/log/clamp/frame-stack.
- Q-Former projector small attention: fixed Q=3, K=15 gives a compact fused attention target.
- Mid-layer CTC feedback: Linear -> Softmax(348) -> Linear -> Add can be fused or at least kept as a local subgraph.
- Plus hidden concat: avoid materializing if projector K/V projections can read two source buffers or use a concatenated-view contract.

Lower priority:

- Training loss path and label masking.
- PEFT LoRA adapter toggling; first Plus checkpoint does not require active adapters.
- Full attention-output tensors and hidden-state recording.

## 11. Runtime staging plan

Stage 1: config/weight admission

- Parse composite config, nested Granite config, nested BLIP-2 Q-Former config, and processor config.
- Reject unsupported projector configs, active PEFT adapters, `inputs_embeds` placeholder detection, and non-Plus mismatches initially.

Stage 2: preprocessing parity boundary

- Start with CPU/data-pipeline preprocessing outside DinoML graph, producing `input_features`, `input_features_mask`, token ids, and attention mask.
- Validate placeholder expansion and projected token counts against HF processor formulas.

Stage 3: audio encoder parity

- Implement Conformer block ops and local relative attention.
- Validate encoder output with and without `cat_hidden_layers`; for checkpoint, require `[3]`.

Stage 4: projector parity

- Compose BLIP-2 Q-Former subset with query embeddings only, cross-attention every layer, and no text-input path.
- Validate `[B,T,2048] -> [B,ceil(T/15)*3,2048]`.

Stage 5: multimodal prefill

- Implement guarded audio embedding row-copy, then call Granite prefill on `inputs_embeds`.
- Stub generation controller beyond greedy/sampling as needed, but preserve cache ABI.

Stage 6: decode with KV cache

- Decode text-only with Granite KV cache after first multimodal prefill.
- Ensure `prepare_inputs_for_generation` never reruns audio encoder during cached decode.

Stage 7: optimized kernels/fusions

- Add local attention, Q-Former small attention, Conv1d/BN fusions, and last-token logits.

## 12. Parity and validation plan

- Feature extractor formula tests: raw lengths around hop/window boundaries, odd/even mel frame counts, sequence-list padding, `audio_embed_sizes`, and `input_features_mask`.
- Processor tests: one audio token, multiple audio tokens in one sample, multiple batch samples, count mismatch rejection.
- Unit ops: LayerNorm, RMSNorm, BatchNorm eval fold, SiLU, GELU, GLU, depthwise Conv1d padding, Shaw relative score einsum.
- Single Conformer block parity in fp32, then bf16 tolerances.
- Full encoder parity for checkpoint config, including layer 3 concat and mid-layer feedback.
- Projector parity for synthetic `[B,T,2048]` at `T=1,15,16,200`.
- Audio embedding stitch parity versus PyTorch `masked_scatter`.
- Granite single-layer, full prefill logits, and decode-token parity using existing Granite audit/test strategy.
- End-to-end ASR/AST text parity on short public audio clips after tokenizer/processor integration.

Recommended tolerances: fp32 custom op parity `atol=1e-5, rtol=1e-4`; bf16/fp16 branch parity `atol=2e-2, rtol=2e-2` for full model logits, with tighter local tolerances for deterministic linear/norm regions.

## 13. Performance probes

- CPU preprocessing throughput: seconds of audio/sec for mel + normalization + frame stacking.
- Audio encoder throughput by `T_enc`, especially around context block boundaries 200, 400, 600.
- Projector throughput by number of 15-frame blocks; isolate Q-Former from encoder.
- Multimodal prefill latency split: audio encoder, projector, embedding stitch, Granite prefill.
- Decode tokens/sec with audio prefix KV cache already populated.
- KV cache memory: 40 layers x K/V x `[B,4,S,128]` bf16 for Granite.
- Attention backend comparison: eager/SDPA/Flash for Granite; custom local attention versus generic math for Conformer.
- Conv module probes: unfused Conv1d path versus pointwise-GEMM/depthwise/BN-fold path.
- Last-token-only logits versus full vocab logits for prefill/decode.
- Audio length sweep: raw seconds, feature frames, projected audio tokens, prompt length, and cache memory.

## 14. Skip/defer list

- Training loss, gradient checkpointing, dropout randomness.
- General `inputs_embeds` placeholder-detection path for audio token embeddings.
- Active PEFT/LoRA adapter overlays for the first Plus checkpoint.
- Non-Plus Granite Speech checkpoints, NAR variant, ONNX/GGUF mirrors.
- Q-Former text input mode and projector configs other than BLIP-2 Q-Former with cross-attention every layer.
- Full GPU preprocessing at first; CPU preprocessing is acceptable if input ABI is explicit.
- Beam search, speculative decoding, streaming audio chunking, and continuous batching.
- Quantized/packed weights beyond DinoML's normal dense or separately audited GGUF loading path.
- Attention/hidden-state output collection.

## 15. Final implementation checklist

- [ ] Parse `GraniteSpeechPlusConfig` with nested encoder, projector, and Granite text configs.
- [ ] Parse `GraniteSpeechProcessor` / `GraniteSpeechFeatureExtractor` config and enforce 16 kHz input contract.
- [ ] Implement or externalize mel/log/clamp/frame-stack preprocessing.
- [ ] Implement Conformer FFN, Conv1d/GLU/depthwise/BatchNorm, and post-norm ops.
- [ ] Implement block-local Shaw relative attention with padded-last-block masking.
- [ ] Implement mid-layer Linear/Softmax/Linear feedback.
- [ ] Implement `cat_hidden_layers` output concat and config admission checks.
- [ ] Implement BLIP-2 Q-Former subset for learned query + audio cross-attention.
- [ ] Implement projector window pad/reshape and final Linear to Granite hidden size.
- [ ] Implement guarded audio placeholder row-copy rewrite.
- [ ] Compose audited Granite causal LM with GQA RoPE cache, multipliers, tied embeddings, and logits scaling.
- [ ] Ensure generation prefill runs audio path once and decode reuses text KV cache only.
- [ ] Add parity tests for processor counts, encoder, projector, stitch, prefill logits, and decode token.
- [ ] Benchmark preprocessing, encoder, projector, prefill, decode, and cache memory separately.
