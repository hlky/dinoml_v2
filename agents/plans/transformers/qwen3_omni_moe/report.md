# Transformers audit: qwen3_omni_moe

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: Qwen/Qwen3-Omni-30B-A3B-Instruct, Qwen/Qwen3-Omni-30B-A3B-Thinking, Qwen/Qwen3-Omni-30B-A3B-Captioner
Config source: official Hugging Face config.json and preprocessor_config.json, main revisions listed below
Source files inspected: configuration_qwen3_omni_moe.py, processing_qwen3_omni_moe.py, modeling_qwen3_omni_moe.py, modular_qwen3_omni_moe.py
Any missing files or assumptions: processor_config.json and video_preprocessor_config.json were absent in representative repos; model card task statements are treated as metadata, not graph facts.
```

The generated `configuration_*.py`, `processing_*.py`, and `modeling_*.py` files state that they are generated from `modular_qwen3_omni_moe.py`; future Transformers source edits should inspect the modular file first. This report uses the generated runtime files for the exact imported class behavior.

Representative configs:

| Model id | HF revision | Scope |
|---|---:|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `26291f793822fb6be9555850f06dfe95f2d7e695` | full thinker + talker + code2wav |
| `Qwen/Qwen3-Omni-30B-A3B-Thinking` | `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b` | thinker only, text output |
| `Qwen/Qwen3-Omni-30B-A3B-Captioner` | `a2bd106cbf527db5676e79662674da22b0545ec0` | thinker only, audio-caption text output |

Local notes are in `_sources/source_basis.md` and `_sources/config_sweep.md`.

Primary DinoML target for first integration: **thinker text-output parity** for text, image, video, and audio inputs. Audio waveform generation through talker/code2wav is a later staged target.

## 2. High-level architecture

Qwen3 Omni MoE is a staged multimodal generation family:

```text
CPU/audio-image-video preprocessing
  -> audio encoder and/or vision encoder
  -> placeholder embedding stitch + M-RoPE position IDs
  -> MoE causal text decoder prefill/decode
  -> thinker logits/sampling
  -> optional talker codec-token generation
  -> optional code2wav waveform synthesis
```

Stage decomposition:

| Stage | Required for first target | Independently cacheable | Notes |
|---|---:|---:|---|
| tokenizer/chat template/placeholder expansion | yes | yes | Processor expands one modality token into repeated placeholder tokens based on feature lengths/grid metadata. |
| audio feature extraction | yes for audio input | yes | Whisper-style log-mel pipeline is CPU/data-pipeline first. |
| audio encoder | yes for audio input | yes | Conv2d downsampling + packed noncausal transformer encoder. |
| image/video processor | yes for visual input | yes | Processor emits flattened `pixel_values`, `image_grid_thw`, `video_grid_thw`, `video_second_per_grid`. |
| vision encoder | yes for image/video input | yes | Conv3d patch embed + packed noncausal ViT + DeepStack features. |
| multimodal stitch | yes | no | Source uses `masked_scatter`; processor gives a stricter count/order pattern that DinoML can guard. |
| thinker decoder | yes | KV cache | Causal GQA decoder with M-RoPE and MoE MLP in every layer for official configs. |
| talker decoder | deferred | KV cache | Instruct only, batch size 1 for audio output. |
| residual code predictor | deferred | KV cache | Multi-codebook codec generation helper. |
| code2wav | deferred | chunk cache possible | Sliding-attention transformer + causal ConvNet vocoder. |

## 3. Important config dimensions

Official checkpoint dimensions are operator-significant and differ from source defaults.

| Component | Field | Official 30B-A3B value |
|---|---|---:|
| thinker text | `hidden_size` | 2048 |
| thinker text | `num_hidden_layers` | 48 |
| thinker text | `num_attention_heads` / `num_key_value_heads` | 32 / 4 |
| thinker text | `head_dim` | 128 |
| thinker text | Q/O width | 4096, so `hidden_size != num_heads * head_dim` |
| thinker text | K/V width | 512 |
| thinker text | `vocab_size` | 152064 |
| thinker text | `max_position_embeddings` / `rope_theta` | 65536 / 1000000 |
| thinker text | MoE | 128 experts, top-8, `moe_intermediate_size=768` |
| thinker text | dense fallback MLP | `intermediate_size=768`, but official `mlp_only_layers=[]` |
| vision | depth / hidden / heads | 27 / 1152 / 16 |
| vision | patch / temporal patch / merge | 16 / 2 / 2 |
| vision | MLP / output | 4304 / 2048 |
| vision | DeepStack layers | 8, 16, 24 |
| audio | mel bins / `d_model` / heads | 128 / 1280 / 20 |
| audio | layers / FFN | 32 / 5120 |
| audio | output projection | 2048 |
| audio | chunk windows | `n_window=50`, `n_window_infer=800`, `conv_chunksize=500` |
| processor | audio sampling | 16 kHz, `n_fft=400`, `hop_length=160`, `n_samples=4800000` |
| processor | image/video | `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`, min/max pixels 3136/12845056 |

Representative checkpoint sweep:

| Checkpoint | Input modalities | Output modalities | `enable_audio_output` | Main structural change |
|---|---|---|---:|---|
| Instruct | text/image/video/audio | text/audio | true | Loads talker and code2wav in addition to thinker. |
| Thinking | text/image/video/audio | text | false | Same thinker scale, no talker/code2wav runtime. |
| Captioner | audio primarily | text | false | Same thinker scale, task is audio captioning; prompt rules differ outside graph. |

Instruct-only output audio dimensions:

| Component | Dimensions |
|---|---|
| talker text decoder | 20 layers, hidden 1024, 16 Q heads, 2 KV heads, head dim 128, vocab 3072, 128 experts top-6, shared expert intermediate 768 |
| code predictor | 5 full-attention layers, hidden 1024, 16 Q heads, 8 KV heads, vocab 2048, `num_code_groups=16` |
| code2wav | 8 sliding-attention layers, hidden 1024, 16 heads, codebook 2048, 16 quantizers, sliding window 72, transposed conv upsample rates `[8,5,4,3]` after ratios `[2,2]` |

## 3a. Family variation traps

- Official text decoder has `hidden_size=2048` but Q/O attention width `32 * 128 = 4096`; do not infer projection widths from hidden size.
- GQA is mandatory: thinker 32 query heads / 4 KV heads; talker text 16 / 2; code predictor 16 / 8; code2wav 16 / 16.
- Source defaults are not representative of official 30B-A3B checkpoints. Use checkpoint config values for model admission.
- `audio_token_id` differs between source default and official configs. Placeholder IDs must come from config/tokenizer.
- `enable_audio_output=false` means the top-level model should not instantiate or require talker/code2wav parity.
- Thinker MLP is MoE in every official layer because `decoder_sparse_step=1` and `mlp_only_layers=[]`.
- Talker text source has a shared expert path, but `Qwen3OmniMoeTalkerDecoderLayer` overwrites `self.mlp` to `Qwen3OmniMoeTalkerTextSparseMoeBlock`; first talker parity must include shared expert gating.
- Processor placeholder expansion controls model sequence length. DinoML should reject mismatched placeholder counts instead of admitting general boolean scatter.
- Vision/audio encoders are packed variable-length noncausal attention with `cu_seqlens`; the eager non-flash fallback sometimes loops over chunks and is not a production lowering target.
- Layout is source-specific: audio convs use NCHW, vision patch embedding reshapes to Conv3d NCTHW-like windows, code2wav convs use `[B,C,T]`. Treat NHWC/channel-last as guarded local optimization only.
- M-RoPE uses three modality position axes plus a text position lane in decoder input handling. Decode uses cached `rope_deltas`.
- Audio output is batch-size-1 only in source `generate`; this is an ABI limitation, not just a performance choice.

## 4. Operator coverage checklist

Tensor/layout ops:

- `view`, `reshape`, `permute`, `transpose`, `contiguous`, `flatten`, `split`, `cat`, `stack`, `chunk`, `pad`, boolean indexing, `index_add_`, indexed assignment.
- Ragged/packed sequence descriptors: `cu_seqlens`, chunk length construction, `repeat_interleave`, `cumsum`, dynamic `arange`.
- Layout guards for audio `[B, mel, T] -> [chunks, 1, mel, T]`, vision flattened patches into Conv3d windows, code2wav `[B,T,D] <-> [B,D,T]`.

Neural primitives:

- Embedding lookup; linear with and without bias; packed linear split for vision QKV and MoE gate/up.
- RMSNorm with fp32 variance accumulation; LayerNorm for audio/vision.
- SiLU-gated MLP, GELU, tanh-GELU variant, sigmoid, clamp.
- Conv2d stride-2 audio downsampling: `1->480`, `480->480`, `480->480`, kernel 3, stride 2, padding 1.
- Vision Conv3d patch embed: input windows `[C=3,T=2,H=16,W=16] -> hidden 1152`, kernel=stride.
- Code2wav causal Conv1d, depthwise Conv1d, ConvTranspose1d crop, ConvNeXt-style block, SnakeBeta activation.

Attention primitives:

- Noncausal packed audio attention, MHA 20 heads, head dim 64.
- Noncausal packed vision attention, MHA 16 heads, head dim 72, packed QKV linear.
- Causal thinker GQA, 32 Q heads, 4 KV heads, head dim 128, Q/K RMSNorm, RoPE before cache update.
- Causal talker/code predictor/code2wav attention with sliding-window support where configured.
- Backend ABI for FlashAttention/SDPA-like calls with `cu_seq_lens_q/k`, max lengths, `sliding_window`, causal masks.

MoE ops:

- Router linear `[tokens, hidden] x [num_experts, hidden] -> [tokens, experts]`.
- Softmax in fp32, top-k, optional top-k renormalization.
- One-hot/token-to-expert dispatch or equivalent sorted/grouped dispatch.
- Expert weights stored as `gate_up_proj[num_experts, 2*moe_intermediate, hidden]` and `down_proj[num_experts, hidden, moe_intermediate]`.
- Per-token weighted expert accumulation using `index_add`.
- Talker shared expert: dense SwiGLU MLP plus sigmoid scalar gate added to sparse expert output.

Position/rotary ops:

- Text M-RoPE: 3-axis position IDs, interleaved section rewrite, cos/sin in fp32, then dtype cast.
- Vision 2D RoPE from grid row/column IDs, duplicated cos/sin.
- Vision learned 2D position interpolation from a square table using four-neighbor bilinear weights.
- Decode `rope_deltas` state.

Generation/cache ops:

- Dynamic KV cache per decoder layer; cache stores K/V after RoPE for self-attention.
- Multimodal encoder outputs are consumed only on first generation iteration and then removed by `prepare_inputs_for_generation`.
- Thinker generation can return hidden states for talker conditioning.
- Talker generation mutates `generation_step`, residual codec codes, and nested code predictor generation.

Preprocessing-coupled ops:

- Whisper log-mel extraction: 16 kHz, FFT 400, hop 160, 128 bins.
- Qwen2VL image/video processor: resize by pixel area bounds, patchify metadata grids, temporal patching.
- Placeholder expansion: audio length formula, image/video `grid.prod() // merge_size**2`, optional audio/video interleaving by temporal position.

Scatter/indexed update ops:

- `inputs_embeds.masked_scatter(audio_mask, audio_features)`.
- Same for image and video features.
- DeepStack: add visual embeddings at visual positions in early decoder layers.
- Joint image/video deepstack reorder when both modalities exist.

Optional codec/vocoder ops:

- Multi-codebook embeddings with offset `quantizer_id * codebook_size`.
- Code predictor nested autoregressive generation for residual code groups.
- Code2wav chunked decode with left context and output trimming.

## 5. Layer/block breakdown

Audio encoder:

```text
input_features [mel=128, total_T] + feature_lens
-> split into 2*n_window chunks, pad to batch
-> Conv2d/GELU x3 stride-2 over mel/time
-> permute [N,C,F,T] -> [N,T,C*F]
-> Linear(C*F -> 1280)
-> add sinusoidal position embedding
-> boolean-pack valid tokens
-> 32 x:
     LayerNorm
     packed noncausal MHA(1280, 20 heads, bias=True)
     residual
     LayerNorm
     Linear(1280 -> 5120) -> GELU -> Linear(5120 -> 1280)
     residual
-> LayerNorm -> Linear(1280 -> 1280) -> GELU -> Linear(1280 -> 2048)
```

Vision encoder:

```text
pixel_values flattened processor patches + grid_thw
-> reshape to [-1, 3, 2, 16, 16]
-> Conv3d kernel=stride=[2,16,16] -> hidden 1152
-> add learned interpolated 2D position embeddings
-> build 2D RoPE and cu_seqlens by grid_thw
-> 27 x:
     LayerNorm
     packed noncausal attention with packed qkv Linear(1152 -> 3456)
     residual
     LayerNorm
     Linear(1152 -> 4304) -> GELU-tanh -> Linear(4304 -> 1152)
     residual
     if layer in [8,16,24]: merge 2x2 spatial patches -> Linear/gelu/Linear to 2048
-> final merge 2x2 spatial patches -> 2048 pooled visual tokens
```

Thinker decoder layer, repeated 48 times:

```text
x [B,S,2048]
-> RMSNorm
-> q_proj: 2048 -> 4096, k_proj/v_proj: 2048 -> 512, no bias
-> reshape to Q [B,32,S,128], K/V [B,4,S,128]
-> per-head RMSNorm on Q/K head_dim
-> M-RoPE(q,k)
-> update/read KV cache
-> causal GQA attention, repeat KV by 8 in eager fallback
-> o_proj: 4096 -> 2048
-> residual
-> RMSNorm
-> router top-8 over 128 experts
-> per selected expert: Linear(2048 -> 1536) split gate/up, SiLU(gate)*up, Linear(768 -> 2048)
-> weighted index-add accumulation
-> residual
```

Talker/code2wav are deferred for first target, but the Instruct path adds:

```text
thinker hidden/embed -> projection MLPs -> talker MoE decoder -> codec logits
per generated codec token -> code predictor generates residual code groups
residual code groups -> code2wav embedding mean -> sliding transformer -> causal ConvNet vocoder
```

## 6. Attention requirements

Thinker text attention:

- Causal self-attention with GQA.
- Query heads/KV heads/head dim: 32 / 4 / 128.
- Q and O projection width 4096; K/V projection width 512.
- Q/K RMSNorm happens before RoPE.
- RoPE is applied before cache update, so cached keys are post-RoPE.
- Masks are built by `create_causal_mask`; source can dispatch to eager, SDPA, FlashAttention, or FlexAttention through `ALL_ATTENTION_FUNCTIONS`.
- No official thinker sliding window.

Audio attention:

- Noncausal self-attention over packed audio chunks.
- `cu_seqlens` separates chunks; FlashAttention path relies on varlen metadata.
- Eager fallback approximates packed blocking using a 4D mask and loops to fill block-diagonal masks. DinoML should prefer explicit packed attention or guarded dense-block fallback.

Vision attention:

- Noncausal packed self-attention over image/video patch sequences.
- FlashAttention path uses `cu_seqlens`; non-flash path splits q/k/v by chunk lengths and runs attention per chunk.
- Uses packed QKV linear and vision 2D RoPE.

Talker/code predictor/code2wav:

- Causal self-attention with KV cache.
- Talker text: GQA 16 / 2 / 128 with M-RoPE position IDs derived from original thinker sequence metadata.
- Code predictor: GQA 16 / 8 / 128; source official config uses full attention layers.
- Code2wav: MHA 16 / 16 with sliding window 72 in every layer.

FlashAttention/SDPA compatibility:

- First useful lowering should support dense causal GQA and packed noncausal varlen attention. Eager repeat-KV fallback is correct but too slow for real model scale.

## 7. Position encoding and custom math

Thinker M-RoPE:

```python
def mrope_cos_sin(inv_freq, position_ids_3, mrope_section):
    # position_ids_3: [3, batch, seq]
    freqs = matmul(inv_freq[None, None, :, None], position_ids_3[:, :, None, :]).transpose(2, 3)
    freqs_t = freqs[0]
    for dim, offset in [(1, 1), (2, 2)]:
        freqs_t[..., offset : mrope_section[dim] * 3 : 3] = freqs[dim, ..., offset : mrope_section[dim] * 3 : 3]
    emb = cat([freqs_t, freqs_t], dim=-1)
    return cos(emb), sin(emb)
```

Source default `mrope_section` is `[24, 20, 20]`; official config should be checked at load because RoPE fields may be normalized by Transformers config migration.

Multimodal position IDs:

- Text tokens use monotonically increasing IDs.
- Image tokens use temporal, height, width IDs. Image temporal ID uses `position_id_per_seconds`.
- Video temporal ID uses `video_second_per_grid * position_id_per_seconds`.
- Audio-only uses audio encoder output length as token count and 1D positions copied across T/H/W lanes.
- Audio-in-video interleaves audio and video positions by temporal coordinate.
- Decode reuses `rope_deltas` and computes new positions from `past_key_values_length + rope_deltas`.

Vision position math:

- Learned 2D table is bilinearly interpolated by row/column coordinates and repeated over time.
- 2D RoPE uses row/column frequency lookup, flattened in merge-block order.

Precomputable:

- Static inv-freq tables, sinusoidal audio position table, code offsets.

Dynamic:

- M-RoPE position IDs, vision grid interpolation indices/weights, `cu_seqlens`, audio chunk lengths, decode deltas.

## 8. Preprocessing and input packing

Processor ABI:

- `input_ids`, `attention_mask`.
- Audio: `input_features`, `feature_attention_mask`; feature extractor is Whisper-style with 16 kHz sampling, `n_fft=400`, `hop_length=160`, 128 mel bins, right padding.
- Images: `pixel_values`, `image_grid_thw`.
- Videos: `pixel_values_videos`, `video_grid_thw`, `video_second_per_grid`.
- Optional `use_audio_in_video` changes placeholder expansion and M-RoPE ordering.

Placeholder expansion:

- Audio token count is `_get_feat_extract_output_lengths(feature_attention_mask.sum(-1))`.
- Image token count is `image_grid_thw.prod() // merge_size**2`.
- Video token count is `video_grid_thw.prod() // merge_size**2`.
- With `use_audio_in_video=true`, processor replaces `<vision_bos><video><vision_eos>` with interleaved vision/audio placeholders bounded by vision/audio BOS/EOS tokens.

Model stitch:

- Source calls `masked_scatter` into `inputs_embeds`.
- It validates image/video feature element count against placeholder element count.
- Audio path currently obtains the audio placeholder mask but does not pass `audio_features` into count validation; DinoML should still validate audio token count from processor metadata.
- For DinoML, lower this to ordered row copy under guards:
  - placeholder IDs match config,
  - number of placeholders equals encoder feature rows,
  - feature flatten order is row-major as emitted by the encoder,
  - placeholders correspond to processor-expanded modality order,
  - reject arbitrary masks or user-supplied `inputs_embeds` unless separately audited.

Video ABI:

- Video decode/frame sampling is outside model source and owned by processor/caller utilities.
- Processor emits flattened `pixel_values_videos` plus `video_grid_thw`.
- Source patch embed reshapes to `[*, 3, temporal_patch_size, patch_size, patch_size]`, so layout translation must guard the entire processor-to-Conv3d region.
- Temporal token order is grid T, then spatial merge-block order; M-RoPE uses `video_second_per_grid`.

Audio output ABI:

- Talker generation is only implemented for batch size 1.
- Speaker selection is a generation-controller lookup in `speaker_id`, not neural graph math.
- Code2wav receives discrete codes shaped `[B, num_quantizers, T]`; chunked decode uses chunk size 300 and left context 25.

## 9. Graph rewrite / lowering opportunities

### Rewrite: placeholder `masked_scatter` -> guarded ordered row copy

Source pattern:

```text
inputs_embeds = inputs_embeds.masked_scatter(modality_mask, modality_features)
```

Replacement:

```text
Gather placeholder row indices -> RowCopy(modality_features -> inputs_embeds[indices])
```

Preconditions:

- `input_ids` path, not arbitrary `inputs_embeds` equality-to-embedding fallback.
- Placeholder token IDs come from config.
- `mask.sum() == feature_rows` and `mask` expanded only across hidden dimension.
- Feature row order matches processor placeholder expansion order.

Failure cases:

- User passes `inputs_embeds` without `input_ids`.
- Mixed audio/video interleaving without processor metadata.
- Count mismatch or non-monotonic feature order.

Parity test sketch:

- Construct text with one image, one audio, one video, and audio-in-video variant; compare source `inputs_embeds` after stitch to RowCopy lowering.

### Rewrite: non-overlap Conv3d patch embed -> Linear

Source pattern:

```text
hidden.view(-1, C, T, P, P) -> Conv3d(C -> D, kernel=stride=[T,P,P])
```

Replacement:

```text
WindowFlatten([C,T,P,P]) -> Linear(C*T*P*P -> D) + bias
```

Preconditions:

- Kernel equals stride.
- Padding and dilation are zero/default.
- `groups=1`.
- Processor already emits flattened windows in the same C,T,H,W order.
- `temporal_patch_size`, `patch_size`, and `in_channels` match config.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * temporal_patch * patch * patch)
```

Layout constraints:

- Preserve source NCTHW flatten order unless the entire processor-to-patch region is controlled.

### Rewrite: packed MoE eager loop -> sorted/grouped expert GEMM

Source pattern:

```text
router softmax -> topk -> one_hot -> per expert gather -> gate_up -> silu*up -> down -> weighted index_add
```

Replacement:

```text
TopKRouter -> TokenSortByExpert -> GroupedGemm(gate_up) -> SwiGLU -> GroupedGemm(down) -> WeightedUnsortReduce
```

Preconditions:

- Static `num_experts`, `top_k`, `moe_intermediate_size`.
- Expert weights stored in source layout `[E, 2I, H]` and `[E, H, I]`.
- Top-k order and renormalization match config.
- Accumulation dtype/tolerance specified.

Failure cases:

- Training/router logits output parity with aux loss required.
- Expert parallel or tensor parallel sharding not represented.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states)
```

Replacement:

```text
if decode and only next-token logits needed: lm_head(hidden_states[:, -1:])
```

Preconditions:

- No loss calculation and caller does not request full prefill logits.
- Generation controller only samples next token.

### Layout guarded regions

- Audio Conv2d region can be optimized to NHWC only if all three convs, GELUs, `permute(0,3,1,2)`, flatten, and linear input layout are rewritten together.
- Vision Conv3d patch embed can become a patch Linear if processor order is controlled.
- Code2wav Conv1d/ConvTranspose1d must stay `[B,C,T]` unless every convolution and LayerNorm permutation is rewritten.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm with fp32 accumulation for decoder blocks.
- Causal GQA FlashAttention with RoPE-applied KV cache and nonstandard Q/O widths.
- M-RoPE cos/sin + RoPE apply for thinker/talker.
- MoE router + grouped expert GEMM + weighted scatter-add.
- Multimodal RowCopy stitch replacing general `masked_scatter`.
- Vision/audio packed-varlen attention with `cu_seqlens`.

Medium priority:

- Vision Conv3d patch embed to GEMM.
- Vision QKV projection + 2D RoPE + packed attention.
- Audio Conv2d downsample chain, especially for batch/chunk throughput.
- DeepStack visual add at selected layers.
- Talker shared-expert MoE and residual code predictor nested generation.

Lower priority:

- Code2wav SnakeBeta + causal ConvNet fused kernels.
- ConvTranspose1d crop fusion.
- Last-token logits and vocabulary partitioning for generation.
- Audio-in-video temporal placeholder interleaving as a graph-level helper.

## 11. Runtime staging plan

Stage 1: config and weights admission.

- Parse nested configs.
- Reject source-default-only shapes unless explicitly constructing a random tiny model.
- Load thinker-only checkpoints first; expose `enable_audio_output` as a graph staging flag.

Stage 2: text-only thinker block parity.

- Embedding, M-RoPE, GQA, RMSNorm, MoE.
- Single-layer and 48-layer logits with random and loaded weights.

Stage 3: multimodal encoders independently.

- Vision encoder pooled and DeepStack features.
- Audio encoder output rows and feature length math.
- Validate packed attention and layout guards.

Stage 4: multimodal prefill.

- Processor-produced placeholders, RowCopy stitch, M-RoPE `position_ids`, prefill logits.
- Cache encoder/projector outputs independently.

Stage 5: decode.

- Dynamic KV cache with `rope_deltas`.
- Remove encoder inputs after first iteration.
- Last-token logits and sampling-controller handoff.

Stage 6: Instruct audio output.

- Talker projection input construction from thinker hidden states.
- Talker MoE decode, code predictor residual codes, suppress-token generation controller.
- Enforce batch size 1 initially.

Stage 7: code2wav.

- Code embedding offsets, sliding attention transformer, chunked causal ConvNet decode, waveform clamp.

## 12. Parity and validation plan

- Unit tests for `_get_feat_extract_output_lengths` against processor/model formulas.
- M-RoPE tests for text-only, image-only, video-only, audio-only, and audio-in-video sequences.
- Vision patch embed Conv3d-to-Linear rewrite parity.
- Vision encoder layer parity for one packed image and mixed image/video batch.
- Audio conv + packed encoder layer parity for variable feature lengths.
- Placeholder RowCopy parity against source `masked_scatter`.
- Thinker one-block parity, then N-layer hidden/logit parity.
- Prefill logits parity for text-only and each modality combination.
- Decode parity for 1, 2, and 16 generated tokens with cache.
- MoE grouped dispatch parity with top-k ties avoided, then stress top-k edge cases.
- Instruct deferred tests: talker prefill/decode one token, code predictor residual group generation, code2wav short chunk.

Suggested tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for block outputs; tighter for pure indexing.
- fp16/bf16: `rtol=5e-2`, `atol=5e-2` for full decoder/attention; isolate kernels with stricter local tolerances where possible.
- Generation parity should compare logits before sampling first, then deterministic sampled tokens with fixed controller settings.

## 13. Performance probes

- Processor throughput: audio feature extraction, image/video resize/patch metadata, placeholder expansion.
- Audio encoder throughput vs total audio seconds and chunk count.
- Vision encoder throughput by image pixel count, video frame count, and grid token count.
- Packed-varlen attention overhead vs dense padded fallback.
- Thinker prefill tokens/sec by text length and multimodal prefix length.
- Decode tokens/sec with KV cache by batch size and context length.
- MoE router time, grouped GEMM time, token distribution skew, expert occupancy.
- KV cache memory for thinker/talker/code predictor.
- Last-token logits vs full logits.
- Talker batch-1 codec tokens/sec and nested code predictor overhead.
- Code2wav waveform seconds/sec by code length and chunk/context size.

## 14. Skip/defer list

Safe to defer for first thinker text-output target:

- Training, losses, gradient checkpointing, output router logits, aux router loss.
- Talker and code2wav when `enable_audio_output=false`.
- Audio waveform generation for Instruct.
- Batch audio output; source rejects it.
- Tensor parallel/expert parallel plans.
- Speculative decoding and beam search.
- General boolean scatter; use guarded RowCopy.
- Arbitrary remote-code or historical config flags not read by current in-library source.
- NHWC/channel-last global layout conversion.

Not safe to defer for multimodal thinker parity:

- Audio/vision encoders.
- Placeholder count/order validation.
- M-RoPE and `rope_deltas`.
- MoE top-k dispatch.
- Packed-varlen attention or a correct guarded fallback.

## 15. Final implementation checklist

- [ ] Parse nested `Qwen3OmniMoeConfig` and checkpoint-specific token IDs.
- [ ] Gate first target to thinker text-output parity; route `enable_audio_output=true` talker/code2wav to later stages.
- [ ] Load thinker text, audio, and vision weights with source tensor layouts.
- [ ] Implement RMSNorm and LayerNorm parity.
- [ ] Implement M-RoPE position ID generation and decode `rope_deltas`.
- [ ] Implement GQA attention with `hidden_size != q_width` support.
- [ ] Implement KV cache storing post-RoPE K/V.
- [ ] Implement thinker MoE top-k router and grouped expert GEMM.
- [ ] Implement audio feature length math and audio encoder packed attention.
- [ ] Implement vision Conv3d patch embed or guarded Conv3d-to-Linear rewrite.
- [ ] Implement vision packed attention, position interpolation, 2D RoPE, patch merger, and DeepStack outputs.
- [ ] Replace multimodal `masked_scatter` with guarded ordered RowCopy.
- [ ] Add text-only one-block, prefill, and decode parity tests.
- [ ] Add audio/image/video encoder parity tests.
- [ ] Add multimodal prefill and decode parity tests.
- [ ] Benchmark processor, encoders, prefill, decode, MoE dispatch, and KV memory.
- [ ] Later: implement talker projections, talker MoE decode, code predictor, suppress-token controller, and code2wav.
