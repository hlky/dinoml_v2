# Transformers Audit: Parakeet

## 1. Source Basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nvidia/parakeet-ctc-1.1b, with nvidia/parakeet-ctc-0.6b as the smaller native CTC variant
Config source: HF config/preprocessor/tokenizer JSON plus source defaults
Primary runtime target: offline ASR with ParakeetForCTC greedy CTC decoding
```

Source files inspected:

- Local pinned files under `X:/H/transformers/src/transformers/models/parakeet/`.
- `configuration_parakeet.py`: `ParakeetEncoderConfig`, `ParakeetCTCConfig`.
- `modeling_parakeet.py`: generated runtime source. The file header says it is generated from `modular_parakeet.py`.
- `modular_parakeet.py`: authoritative source for future Transformers edits.
- `feature_extraction_parakeet.py`, `processing_parakeet.py`, `tokenization_parakeet.py`.
- `convert_nemo_to_hf.py` for NeMo conversion and unsupported TDT/encoder-only hints.

External primary links:

- [Transformers Parakeet source tree at commit](https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/parakeet)
- [nvidia/parakeet-ctc-1.1b config.json](https://huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json)
- [nvidia/parakeet-ctc-0.6b config.json](https://huggingface.co/nvidia/parakeet-ctc-0.6b/raw/main/config.json)
- [nvidia/parakeet-ctc-1.1b preprocessor_config.json](https://huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/preprocessor_config.json)
- [nvidia/parakeet-tdt-0.6b-v3 config.json](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json), inspected only as an unsupported config trap.

Missing files or assumptions:

- The pinned in-library source implements CTC and encoder-only classes, not RNNT/TDT transducer decoding.
- Several NVIDIA Parakeet RNNT/TDT HF repos are NeMo-library repos with no raw `config.json` at main during this audit. They should be routed to a separate NeMo/transducer audit, not silently loaded as `ParakeetForCTC`.
- No model execution, imports, DinoML tests, or DinoML code edits were performed.

## 2. High-Level Architecture

Parakeet CTC is an audio encoder plus CTC projection head:

```text
raw waveform -> log-mel feature extractor -> FastConformer encoder -> 1x1 CTC conv head -> frame token ids -> CTC tokenizer collapse
```

Neural stages:

```text
input_features [B, T_feat, mel] + attention_mask [B, T_feat]
  -> Conv2d subsampling by factor 8 over time and mel
  -> relative-position FastConformer encoder blocks
  -> hidden states [B, T_sub, hidden]
  -> Conv1d(hidden -> vocab, kernel=1)
  -> logits [B, T_sub, vocab]
```

The first useful DinoML target is offline CTC ASR with greedy decoding. Encoder-only export is independently useful for future TDT/RNNT composition, but the current report does not own RNNT/TDT decoder, prediction-network, joint-network, duration, beam-search, or streaming ABI.

CPU/data-pipeline work:

- Waveform resampling ownership is outside the feature extractor; callers must supply 16 kHz audio.
- Feature extractor computes preemphasis, STFT, mel projection, log, padding mask, and per-sample feature normalization.

GPU/runtime work:

- First integration can accept `input_features` and `attention_mask` directly.
- Owning feature extraction inside DinoML is a later optional stage because it includes STFT and librosa-derived mel filters.

## 3. Important Config Dimensions

Source defaults come from `configuration_parakeet.py`; checkpoint values below come from HF `config.json`.

| Field | Source default | CTC 0.6B | CTC 1.1B | Notes |
| --- | ---: | ---: | ---: | --- |
| `model_type` | `parakeet_ctc` or `parakeet_encoder` | `parakeet_ctc` | `parakeet_ctc` | CTC wrapper owns the head. |
| `vocab_size` | 1025 | 1025 | 1025 | Token ids 0..1024. |
| `pad_token_id` | 1024 | 1024 | 1024 | Also the CTC blank filtered by tokenizer. |
| `hidden_size` | 1024 | 1024 | 1024 | Encoder width. |
| `num_hidden_layers` | 24 | 24 | 42 | Main 0.6B versus 1.1B difference. |
| `num_attention_heads` | 8 | 8 | 8 | `num_key_value_heads` is set equal in config post-init/source. |
| `head_dim` | inferred 128 | inferred 128 | inferred 128 | Source uses `hidden_size // num_attention_heads` unless `head_dim` exists. |
| `intermediate_size` | 4096 | 4096 | 4096 | FFN expansion 4x. |
| `hidden_act` | `silu` | `silu` | `silu` | Used in FFN and convolution module. |
| `attention_bias` | true | true | true | Also used for FFN linear biases in this source. |
| `convolution_bias` | true | absent/effective true | absent/effective true | Config class supplies true when omitted. |
| `conv_kernel_size` | 9 | 9 | 9 | 1D depthwise conv in each block. |
| `num_mel_bins` | 80 | 80 | 80 | Must match processor `feature_size`. |
| `subsampling_factor` | 8 | 8 | 8 | `log2(factor)=3` Conv2d stride-2 stages. |
| `subsampling_conv_channels` | 256 | 256 | 256 | Subsampling conv channel width. |
| `subsampling_conv_kernel_size` | 3 | 3 | 3 | Same padding `(k-1)//2`. |
| `subsampling_conv_stride` | 2 | 2 | 2 | Applied per subsampling conv. |
| `max_position_embeddings` | 5000 | 5000 | 5000 | Limit after subsampling, not raw waveform length. |
| `scale_input` | true | true | true | Multiplies subsampled embedding by `sqrt(hidden_size)`. |

Representative checkpoint sweep:

| Model id | Runtime status for this report | Operator-significant variation |
| --- | --- | --- |
| `nvidia/parakeet-ctc-0.6b` | Native target | 24 FastConformer blocks, 80 mel bins, vocab 1025, CTC greedy decode. |
| `nvidia/parakeet-ctc-1.1b` | Native target | 42 blocks with same width/head/mel/vocab structure. |
| `nvidia/parakeet-tdt-0.6b-v3` | Reject or route to separate audit | `model_type=parakeet_tdt`, `ParakeetForTDT`, 128 mel bins, vocab 8193, attention/convolution bias false, `scale_input=false`, generation config has decoder start and suppress tokens. The pinned source does not implement `ParakeetForTDT`. |
| `nvidia/parakeet-rnnt-1.1b` | Reject or route to NeMo/transducer audit | HF API-visible NeMo repo, raw `config.json` returned 404 at main during audit. |
| `nvidia/parakeet-tdt-0.6b-v2` | Reject or route to NeMo/transducer audit | HF API-visible NeMo repo, raw `config.json` returned 404 at main during audit. |

## 3a. Family Variation Traps

- Do not treat every Parakeet HF repo as this `parakeet_ctc` source. RNNT/TDT families require transducer decoder work that is absent from the pinned source.
- `modeling_parakeet.py` is generated. Future source patches must target `modular_parakeet.py`.
- `head_dim` can be config-provided even though the CTC checkpoints infer 128. Admission should require `hidden_size == num_attention_heads * head_dim` for the current dense MHA path.
- Source sets `num_key_value_heads = num_attention_heads` in `ParakeetEncoderConfig.__post_init__`, but TDT configs may serialize the field. Do not admit GQA/MQA unless source and weights prove it.
- TDT v3 uses `num_mel_bins=128`, bias false, and `scale_input=false`; CTC uses 80 mel bins, bias true, and scaling true.
- CTC blank is `pad_token_id`, and tokenizer `_decode` first groups repeated tokens then filters blank/pad. End-to-end parity depends on this postprocess ABI.
- Attention is encoder noncausal self-attention with a custom relative-position bias matrix. FlashAttention is explicitly unsupported in source because the custom bias is not supported.
- Source attention can dispatch through SDPA/flex attention interfaces, but parity requires the additive `matrix_bd` relative bias and mask semantics.
- Conv2d subsampling uses NCHW source layout `[B, 1, T, mel]`. Channel-last is only a guarded optimization region.
- The convolution module transposes `[B, T, C]` to `[B, C, T]` for Conv1d/BatchNorm1d and back. Layout passes need no-layout-translation guards around BatchNorm and mask application unless the whole region is rewritten.
- Feature extraction frame count and model subsampling output length use separate floor formulas. Runtime shape code must reproduce both.

## 4. Operator Coverage Checklist

Tensor/layout ops:

- Unsqueeze, squeeze, transpose, permute, reshape/view, contiguous.
- Boolean masks from `arange < lengths`, expand, logical not, logical and, all-reduction, masked fill, in-place mask assignment.
- Floor division and float floor for length propagation.
- Dropout and LayerDrop are training-only for inference, but source graph includes dropout calls that become no-ops in eval.

Neural network primitives:

- Conv2d NCHW, including first dense conv `1 -> 256`, then depthwise grouped Conv2d `groups=256`, pointwise Conv2d, ReLU.
- Linear after subsampling: `256 * floor(num_mel_bins / 8) -> 1024`. For CTC configs this is `2560 -> 1024`.
- LayerNorm over last hidden dimension, 5 per block.
- Feed-forward MLP twice per block: `Linear(1024 -> 4096) -> SiLU -> Linear(4096 -> 1024)` with 0.5 residual scale.
- Conv1d module per block: pointwise `1024 -> 2048`, GLU over channel dim, depthwise Conv1d kernel 9 groups 1024, BatchNorm1d, SiLU, pointwise `1024 -> 1024`.
- CTC head: Conv1d kernel 1, `1024 -> 1025`, equivalent to per-frame linear with source weight layout `[vocab, hidden, 1]`.

Attention primitives:

- Dense noncausal self-attention, MHA, 8 heads, head dim 128 for native CTC configs.
- Q/K/V/O linears with bias controlled by `attention_bias`.
- Relative positional embedding table generated per batch and sequence.
- Relative-key projection `Linear(1024 -> 1024, bias=False)`.
- Global per-head content and positional biases `bias_u`, `bias_v`.
- Shaw/Transformer-XL style relative shift using pad, view, slice, and view.
- Additive attention bias matrix `matrix_bd` of shape `[B, H, T, T]`.
- Softmax in fp32 then cast to query dtype for eager attention.

Preprocessing-coupled ops:

- 16 kHz mono waveform contract; multi-channel input is averaged to mono by the processor.
- Padding/truncation, right padding, attention mask.
- Preemphasis: `x[t] = x[t] - 0.97 * x[t-1]` for `t > 0`, with padding zeroed.
- Hann window, STFT `n_fft=512`, `win_length=400`, `hop_length=160`, `pad_mode="constant"`.
- Magnitude squared, mel filterbank matmul, log with guard `2**-24`.
- Per-sample mean/variance normalization over valid frames only, denominator `features_lengths - 1`.

Decoding/postprocess ops:

- `argmax(logits, dim=-1)` for greedy CTC ids.
- Mask padded output positions to `pad_token_id`.
- Tokenizer `_decode`: optional group consecutive duplicate ids, remove `pad_token_id`, then call fast tokenizer decode.

Training-only or deferred:

- `ctc_loss`, `masked_select` labels, CUDNN-disabled loss block.
- Gradient checkpointing, LayerDrop randomness, dropout randomness.
- RNNT/TDT beam search, prediction network, joint network, duration decoding.

## 5. Layer/Block Breakdown

Audio frontend:

```text
raw waveform [B, samples]
  -> pad/right mask [B, samples]
  -> preemphasis and zero padded samples
  -> STFT -> magnitudes [B, 257, frames]
  -> mel matmul [80, 257] x magnitudes
  -> log -> transpose [B, frames, 80]
  -> mask-derived normalization -> input_features [B, frames, 80]
```

Subsampling:

```text
input_features [B, T, M]
  -> unsqueeze [B, 1, T, M]
  -> Conv2d(1, 256, k=3, stride=2, pad=1) -> ReLU
  -> depthwise Conv2d(256, 256, k=3, stride=2, pad=1, groups=256)
  -> Conv2d(256, 256, k=1) -> ReLU
  -> depthwise Conv2d(256, 256, k=3, stride=2, pad=1, groups=256)
  -> Conv2d(256, 256, k=1) -> ReLU
  -> transpose channel/time and flatten [B, T_sub, 256 * floor(M/8)]
  -> Linear(2560 -> 1024) for CTC checkpoints
```

Length formula per subsampling conv:

```text
L_out = floor((L_in + pad_top + pad_bottom - kernel) / stride) + 1
```

With `kernel=3`, `pad=1`, `stride=2`, repeated 3 times.

FastConformer block, repeated `N` times:

```text
residual = x
x = residual + 0.5 * FFN(LayerNorm(x))

attn_in = LayerNorm(x)
attn = RelPosMHA(attn_in, rel_pos, mask)
x = x + attn

conv = ConvModule(LayerNorm(x), mask)
x = x + conv

x = x + 0.5 * FFN(LayerNorm(x))
x = LayerNorm(x)
```

CTC head:

```text
hidden [B, T_sub, 1024]
  -> transpose [B, 1024, T_sub]
  -> Conv1d(1024 -> 1025, kernel=1)
  -> transpose [B, T_sub, 1025]
```

## 6. Attention Requirements

Parakeet CTC uses encoder self-attention only:

- Noncausal, bidirectional self-attention.
- MHA for native CTC checkpoints: 8 Q heads, 8 K/V heads, 128 head dim.
- Query length equals key/value length after subsampling.
- No autoregressive KV cache. No prefill/decode split.
- Mask shape created as `[B, 1, T_sub, T_sub]` from the subsampled valid-frame mask and used as a boolean visibility matrix.
- Relative-position bias is not a simple learned table. It is generated from sinusoidal positions of length `2*T-1`, projected by `relative_k_proj`, multiplied by biased queries, shifted, sliced to `[B,H,T,T]`, scaled, and then used as the additive attention mask/bias passed into the backend.
- FlashAttention is not supported in source. SDPA/flex can be used only if the additive custom bias is preserved exactly.

Eager attention math:

```text
q_u = q + bias_u
q_v = q + bias_v
rel = relative_k_proj(pos)[B, 2T-1, H, D]
matrix_bd = rel_shift(q_v @ rel.permute(B,H,D,2T-1))[..., :T] * scale
matrix_bd = masked_fill(~visibility, -inf)
scores = (q_u @ k.transpose(-2, -1)) * scale + matrix_bd
weights = softmax(scores, dim=-1, dtype=float32).to(q.dtype)
out = weights @ v
```

The source has helper code for `repeat_kv` and RoPE-like `apply_rotary_pos_emb` in the generated file through shared attention patterns, but Parakeet attention uses the relative position path above. For this family, DinoML should prioritize relative-position attention parity over RoPE.

## 7. Position Encoding and Custom Math

Relative positional embedding:

```python
position_ids = arange(T - 1, -T, -1)       # length 2*T - 1
inv_freq = 1 / (10000 ** (arange(0, hidden, 2) / hidden))
freqs = inv_freq[None, :, None] @ position_ids[None, None, :]
pos = interleave(sin(freqs), cos(freqs))  # [B, 2*T - 1, hidden]
```

Relative shift:

```python
def rel_shift(scores):  # [B, H, T, 2*T-1]
    scores = pad(scores, pad=(1, 0))
    scores = scores.view(B, H, -1, T)
    scores = scores[:, :, 1:].view(B, H, T, 2*T - 1)
    return scores[..., :T]
```

Precompute opportunities:

- `inv_freq` is a non-persistent buffer and can be regenerated from config.
- Sin/cos positional embeddings depend on `T_sub`, batch/device/dtype, but values are batch-identical. A DinoML lowering can compute one `[2*T-1, hidden]` table and broadcast over batch if it preserves dtype casting.
- `relative_k_proj(pos)` is sequence-length dependent but weight-static; cacheable by `T_sub` for offline batching if memory allows.

## 8. Preprocessing and Input Packing

Processor defaults:

| Field | CTC 0.6B/1.1B |
| --- | --- |
| sampling rate | 16000 Hz |
| waveform channels | mono; multi-channel is averaged |
| padding | longest, right padding |
| output tensors | PyTorch by default |
| feature size | 80 |
| `n_fft` | 512 |
| `win_length` | 400 |
| `hop_length` | 160 |
| preemphasis | 0.97 |
| return attention mask | true |

Feature tensor ABI:

```text
input_features: float tensor [B, T_feat, 80]
attention_mask: bool/int tensor [B, T_feat]
```

The feature extractor computes `features_lengths` as:

```text
floor_divide(audio_lengths + (n_fft // 2) * 2 - n_fft, hop_length)
```

For the default `n_fft=512`, this simplifies to `audio_lengths // 160`. This length drives the feature attention mask, not the raw padded waveform length.

Tokenizer/postprocess ABI:

- Tokenizer class is `ParakeetTokenizerFast` in config, backed by `ParakeetTokenizer`.
- `_decode` groups consecutive token ids when `group_tokens=True` and then removes `pad_token_id`.
- CTC blank equals pad token id 1024 for native CTC configs.
- `clean_up_tokenization_spaces=false` in tokenizer config.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: CTC 1x1 Conv1d Head -> Linear

Source pattern:

```text
transpose hidden [B,T,H] -> [B,H,T]
Conv1d(H -> vocab, kernel=1)
transpose -> [B,T,V]
```

Replacement:

```text
Linear(H -> V) over each frame
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Weight shape `[V, H, 1]`; transform to linear weight `[V, H]`.
- Bias preserved.

Failure cases:

- Any future CTC head with temporal kernel, dilation, groups, or nonzero padding.

Parity test sketch:

- Random `[B,T,1024]`, compare Conv1d head output to transformed Linear for fp32 and target reduced precision.

### Rewrite: Subsampling Conv2d Stage As Guarded NCHW Region

Source pattern:

```text
[B,T,M] -> unsqueeze NCHW [B,1,T,M] -> Conv2d/ReLU/depthwise/pointwise -> flatten mel/channel
```

Replacement:

- Keep semantic NCHW initially.
- Later use channel-last or im2col/GEMM only for the enclosed subsampling region.

Preconditions:

- `subsampling_factor` is a power of two and equals `stride ** num_layers`.
- Default CTC shape has `M=80`, `stride=2`, `num_layers=3`, so flattened feature width is `256 * 10`.
- Preserve PyTorch flatten order after `hidden_states.transpose(1, 2).reshape(B, T_sub, -1)`.

Failure cases:

- TDT v3 `M=128` changes linear input to `256 * 16`.
- Non-power-of-two factor or unexpected kernel/stride should reject until covered.

### Rewrite: Conformer Conv Module Fusion

Source pattern:

```text
transpose BTC -> BCT
pointwise Conv1d -> GLU -> mask fill -> depthwise Conv1d -> BatchNorm1d -> SiLU -> pointwise Conv1d
transpose BCT -> BTC
```

Replacement:

- Fuse pointwise Conv1d as per-time GEMM.
- Fuse GLU and mask zeroing.
- Keep BatchNorm as inference affine using running stats if weights expose eval buffers.

Preconditions:

- Inference/eval mode.
- BatchNorm has frozen running mean/variance.
- Conv1d stride 1 and padding same for depthwise kernel.
- Mask semantics reproduced: source derives `all_masked_rows` from a 4D visibility matrix, not directly from `[B,T]`.

Failure cases:

- Training mode BatchNorm/dropout.
- Layout rewrite that changes the mask reduction axis.

### Rewrite: Relative-Position Attention Bias Precompute

Source pattern:

```text
pos = sinusoidal_relative_positions(T)
rel = relative_k_proj(pos)
matrix_bd = rel_shift((q + bias_v) @ rel^T) * scale
```

Replacement:

- Cache `rel` by `T_sub` and dtype/device inside a session or plan, then compute the query-dependent matmul each run.

Preconditions:

- Fixed `T_sub` bucket or bounded cache.
- Same `relative_k_proj` weights and dtype policy.
- Preserve `-inf` mask fill before the attention backend call.

Failure cases:

- Dynamic sequence lengths beyond `max_position_embeddings`.
- Backend that cannot accept arbitrary additive bias.

## 10. Kernel Fusion Candidates

Highest priority:

- Conv2d subsampling region. It is the first heavy shape-changing stage and owns length/mask propagation.
- Relative-position MHA with additive bias. Dense attention is the major quadratic cost and needs a custom-bias compatible path.
- Conformer Conv1d module fusion. The pointwise GLU plus depthwise conv plus BatchNorm/SiLU pattern repeats every layer.
- LayerNorm plus residual scale/add around FFN and attention.

Medium priority:

- FFN `Linear -> SiLU -> Linear`, especially batched frame GEMMs.
- CTC head Conv1d-to-Linear rewrite and argmax/mask postprocess.
- Relative position embedding/projection cache by `T_sub`.
- Mask construction from compressed lengths as a generated shape/mask helper.

Lower priority:

- Full GPU STFT/mel frontend. It matters for end-to-end deployment but can be split from neural parity.
- CTC text decode inside runtime. Token grouping and tokenizer decode can remain host-side initially.
- RNNT/TDT transducer decoding. Required for separate model families, not for native CTC parity.

## 11. Runtime Staging Plan

Stage 1: Config and weight admission.

- Admit `model_type=parakeet_ctc`, `architectures=["ParakeetForCTC"]`.
- Reject `parakeet_tdt`, `ParakeetForTDT`, RNNT, or NeMo-only repos with a clear route-to-separate-audit reason.
- Enforce CTC-safe defaults: MHA, supported mel bins, supported subsampling factor, supported conv/attention biases.

Stage 2: Tensor input neural parity.

- Accept precomputed `input_features [B,T,80]` and `attention_mask [B,T]`.
- Implement subsampling, length compression, encoder blocks, and CTC head.

Stage 3: One-block and full-encoder parity.

- Compare hidden states after subsampling, after one block, and after final encoder for 0.6B config.

Stage 4: CTC logits and greedy ids.

- Implement argmax and output-mask pad fill.
- Keep tokenizer `_decode` host-side.

Stage 5: Optional processor parity.

- Implement or compose CPU feature extraction; later add GPU STFT/mel kernels if useful.

Stage 6: Performance work.

- Add custom-bias attention backend, Conv1d module fusion, subsampling optimization, and bucketed sequence plans.

## 12. Parity and Validation Plan

Recommended tests:

- Feature extractor parity for fixed waveforms: preemphasis, STFT frame count, mel log features, valid-frame normalization. Tolerance should be loose enough for STFT/library differences unless using the exact same torch path.
- Subsampling length tests over odd/even input frame lengths and padded batches.
- Conv2d subsampling parity for random `input_features` and masks.
- Relative positional embedding and `_rel_shift` random tensor tests.
- Attention single-layer parity with additive relative bias and masks, fp32 first.
- Conformer block parity after disabling dropout/training-only paths.
- CTC head rewrite parity.
- Full `ParakeetForCTC` logits parity for CTC 0.6B and 1.1B checkpoint configs.
- Greedy sequence parity: `argmax`, pad masked positions, tokenizer duplicate collapse, blank removal.
- Rejection tests for `parakeet_tdt` and NeMo-only RNNT/TDT repos.

Suggested numerical tolerances:

- fp32 neural subgraphs: `rtol=1e-4`, `atol=1e-4` after attention if math order differs minimally.
- fp16/bf16 optimized kernels: start with `rtol=5e-2`, `atol=5e-2` for logits, tighten per kernel after baseline is stable.
- Feature extraction: compare exact torch path separately from any custom FFT/mel implementation.

## 13. Performance Probes

- Audio preprocessing throughput: waveform seconds/sec for CPU torch path and any GPU frontend.
- Subsampling throughput versus `B` and `T_feat`.
- Encoder-only throughput for CTC 0.6B and 1.1B layer counts.
- Attention backend comparison with custom relative bias: eager GEMM/softmax versus SDPA/flex-compatible lowering.
- Sequence-length sweep after subsampling: `T_sub` near 500, 1000, 2500, 5000.
- Batch-size sweep with padded-length skew to measure wasted dense attention work.
- Conv module isolated benchmark: pointwise GLU, depthwise conv, BatchNorm/SiLU.
- CTC head and argmax bandwidth benchmark.
- End-to-end ASR requests/hour split into preprocessing, encoder, CTC head, and tokenizer decode.

## 14. Skip/Defer List

Safe to defer for first CTC integration:

- Training loss, `ctc_loss`, label handling, `masked_select`.
- Dropout, LayerDrop, gradient checkpointing.
- Beam search and external language model decoding.
- RNNT/TDT transducer heads, prediction networks, joint networks, duration decoding, suppress-token generation controls.
- Streaming/cache-aware Parakeet variants.
- GPU-native tokenizer decode.
- Full GPU audio feature extraction if precomputed features are accepted.
- Quantized/packed weights unless a checkpoint introduces source-coupled metadata.
- Multi-GPU or tensor parallel execution.

Not safe to defer for logits parity:

- Conv2d subsampling and exact flatten order.
- Compressed attention-mask construction.
- Relative-position attention bias and relative shift.
- BatchNorm1d inference behavior inside every convolution module.
- CTC blank/pad masking for greedy generated ids.

## 15. Final Implementation Checklist

- [ ] Parse `ParakeetCTCConfig` and nested `ParakeetEncoderConfig`.
- [ ] Reject `parakeet_tdt`, RNNT/TDT NeMo-only, and unsupported `architectures`.
- [ ] Load encoder, subsampling, relative-position, convolution, FFN, and CTC-head weights.
- [ ] Implement `input_features [B,T,mel]` plus `attention_mask [B,T]` ABI.
- [ ] Implement Conv2d subsampling with dynamic length/mask propagation.
- [ ] Implement relative positional embedding and relative shift.
- [ ] Implement noncausal MHA with additive relative-position bias.
- [ ] Implement Conformer block FFN, attention, Conv1d module, residual scale, and LayerNorm order.
- [ ] Implement Conv1d kernel-1 CTC head or guarded Linear rewrite.
- [ ] Implement greedy CTC id generation and output mask pad fill.
- [ ] Keep tokenizer CTC collapse/blank removal as host postprocess initially.
- [ ] Add feature-extractor parity tests or explicitly require precomputed features for stage 1.
- [ ] Add one-block, full-encoder, logits, and greedy-id parity tests.
- [ ] Benchmark preprocessing, subsampling, encoder attention, conv module, CTC head, and end-to-end ASR.
