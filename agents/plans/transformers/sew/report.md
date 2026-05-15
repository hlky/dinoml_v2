# SEW Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: primary scope is native `model_type="sew"` checkpoints such as `asapp/sew-tiny-100k`, `asapp/sew-small-100k`, `asapp/sew-mid-100k`, and `asapp/sew-tiny-100k-ft-ls100h`.

Config source: Hugging Face `config.json` and `preprocessor_config.json` fetched from the above repos on 2026-05-13. Sampled repos were public and returned HTTP 200; no gated/401 links were encountered in this sweep.

Source files inspected:

- `transformers/src/transformers/models/sew/modeling_sew.py`
- `transformers/src/transformers/models/sew/modular_sew.py`
- `transformers/src/transformers/models/sew/configuration_sew.py`
- `transformers/src/transformers/models/sew/convert_sew_original_pytorch_checkpoint_to_pytorch.py`
- `transformers/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py`
- `transformers/src/transformers/feature_extraction_sequence_utils.py`

Any missing files or assumptions: `modeling_sew.py` is generated from `modular_sew.py`; future source edits should target `modular_sew.py`, while this audit uses the generated file as the exact runtime source. `asapp/sew-d-*` checkpoints are not covered by this report because they use `model_type="sew-d"` and `src/transformers/models/sew_d/`; they need a separate SEW-D audit.

## 2. High-level architecture

SEW is an audio encoder, not an autoregressive decoder. The first useful DinoML target is `SEWForCTC` ASR inference: raw mono waveform -> SEW encoder -> per-frame CTC logits. The base `SEWModel` feature-extraction target is independently useful; sequence classification is a small optional pooling head.

```text
CPU audio padding/normalization -> model Conv1d feature encoder -> feature LayerNorm/projection ->
SEW squeeze block (AvgPool1d + strided grouped positional Conv1d) ->
noncausal Transformer encoder -> linear upsampling -> CTC/classification head
```

Stage decomposition:

- CPU/data pipeline: audio decode, resampling to 16 kHz, mono validation, padding/truncation, optional zero-mean/unit-variance normalization, optional `attention_mask`.
- GPU/runtime stage 1: raw waveform Conv1d feature extractor in source NCL layout.
- GPU/runtime stage 2: BTC feature LayerNorm, optional `Linear(conv_dim[-1] -> hidden_size)`.
- GPU/runtime stage 3: squeeze from feature time `T_feat` to roughly `floor(T_feat / squeeze_factor)` using AvgPool1d and positional Conv1d, then encoder attention/MLP.
- GPU/runtime stage 4: upsample sequence back toward `T_feat` using `Linear(H -> H * squeeze_factor)` plus reshape, then right-pad if the upsampled sequence is shorter.
- Head stage: CTC `Linear(H -> vocab_size)` for ASR, or optional weighted-layer-sum/mean-pooling classification head.

## 3. Important config dimensions

Source defaults from `SEWConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `hidden_size` | 768 | Encoder width and attention projection width |
| `num_hidden_layers` | 12 | Transformer encoder depth |
| `num_attention_heads` | 12 | MHA heads; `head_dim = hidden_size // heads` |
| `intermediate_size` | 3072 | FFN expansion |
| `squeeze_factor` | 2 | Encoder time downsample/upsample factor |
| `conv_dim` | 13 layers ending at 512 | Conv1d channel schedule |
| `conv_stride` | `(5,2,1,2,1,2,1,2,1,2,1,2,1)` | Feature extractor total stride product 320 |
| `conv_kernel` | `(10,3,1,3,1,3,1,3,1,2,1,2,1)` | Feature extractor kernels |
| `conv_bias` | false | Conv1d bias admission |
| `feat_extract_norm` | `group` | First Conv1d has GroupNorm; `layer` would LayerNorm every conv |
| `num_conv_pos_embeddings` | 128 default, 31 in sampled SEW repos | Positional Conv1d kernel; even kernels crop one step |
| `num_conv_pos_embedding_groups` | 16 | Grouped positional Conv1d |
| `hidden_act`, `feat_extract_activation` | `gelu` | MLP and conv/upsample activations |
| `vocab_size` | 32 default, null for pretrained base configs | CTC head only |
| Cache support | none | Encoder-only; no KV cache |

Representative checkpoint sweep:

| Checkpoint | Architecture | H | Layers | Heads | FFN | Head dim | Pos conv K | Conv last | Vocab | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `asapp/sew-tiny-100k` | `SEWModel` | 512 | 12 | 8 | 2048 | 64 | 31 | 512 | null | No feature projection |
| `asapp/sew-small-100k` | `SEWModel` | 768 | 12 | 12 | 3072 | 64 | 31 | 512 | null | Requires `Linear(512 -> 768)` after conv |
| `asapp/sew-mid-100k` | `SEWModel` | 768 | 24 | 12 | 3072 | 64 | 31 | 512 | null | Deeper encoder, `layerdrop=0.2` training-only |
| `asapp/sew-tiny-100k-ft-ls100h` | `SEWForCTC` | 512 | 12 | 8 | 2048 | 64 | 31 | 512 | 32 | ASR CTC head |
| `asapp/sew-d-tiny-100k` | `SEWDModel` | 384 | 12 | 6 | 1536 | 64 | 31 | 512 | null | Out of scope: `model_type="sew-d"` |

All sampled SEW preprocessors use `Wav2Vec2FeatureExtractor`, `sampling_rate=16000`, `feature_size=1`, `padding_side="right"`, `padding_value=0`, `do_normalize=true`, and `return_attention_mask=false`.

## 3a. Family variation traps

- `sew` and `sew-d` are separate Transformers families. Do not route `model_type="sew-d"` to this lowering without a separate audit.
- `hidden_size` can differ from `conv_dim[-1]`; small/mid SEW require a feature projection `Linear(512 -> 768)`, tiny does not.
- `num_conv_pos_embeddings` differs between source default 128 and sampled SEW configs 31. Even kernels require the `SEWSamePadLayer` one-step crop; odd kernels do not.
- `feat_extract_norm="group"` means only the first conv layer uses channelwise GroupNorm; `feat_extract_norm="layer"` would transpose each conv output to BTC, LayerNorm over channels, then transpose back.
- Attention is encoder MHA with separate Q/K/V weights; no GQA/MQA, no RoPE, no relative bias, no KV cache.
- FlashAttention/SDPA dispatch is available through `ALL_ATTENTION_FUNCTIONS`, but flex attention is explicitly unsupported.
- Preprocessor default `return_attention_mask=false` for group-norm checkpoints is semantically important. Passing an attention mask changes padded-token zeroing and attention masking behavior.
- Inference should disable training-only LayerDrop, dropout, random SpecAugment, CTC loss, and gradient checkpointing.
- Source layout is 1D audio NCL for convolutions and BTC for transformer blocks. NHWC is not applicable; any channel-last 1D optimization should be a local NLC/NCL layout pass with explicit transpose guards.

## 4. Operator coverage checklist

Tensor/layout ops:

- `unsqueeze(input_values, dim=1)` to `[B,1,T_raw]`.
- `transpose` NCL <-> NLC/BTC around conv LayerNorm and transformer entry.
- `reshape/view` for attention heads `[B,T,H] -> [B,T,num_heads,head_dim] -> [B,num_heads,T,head_dim]`.
- `contiguous`-equivalent materialization after attention transpose where required.
- `slice/crop` on the last time axis for `SEWSamePadLayer` and `min_length` alignment of positional conv vs pooling.
- `pad` on sequence axis after upsampling when `T_up < T_feat`.
- Boolean mask expand/repeat, masked fill/zeroing for padded hidden states.

Neural network primitives:

- Conv1d feature stack in NCL layout. Sampled SEW convs: `1->64 K10 S5`, `64->128 K3 S2`, `128->128 K1 S1`, `128->128 K3 S2`, `128->128 K1 S1`, `128->256 K3 S2`, `256->256 K1 S1`, `256->256 K3 S2`, `256->256 K1 S1`, `256->512 K2 S2`, `512->512 K1 S1`, `512->512 K2 S2`, `512->512 K1 S1`; usually `bias=False`.
- GroupNorm with `num_groups=out_channels` for first conv under `feat_extract_norm="group"`.
- LayerNorm over feature channels after conv stack and inside encoder blocks.
- Optional feature projection `Linear(conv_dim[-1] -> hidden_size)`.
- AvgPool1d `kernel_size=squeeze_factor`, `stride=squeeze_factor`.
- Grouped positional Conv1d `hidden_size -> hidden_size`, kernel `num_conv_pos_embeddings`, stride `squeeze_factor`, groups `num_conv_pos_embedding_groups`, weight-normalized.
- Linear upsampling `hidden_size -> hidden_size * squeeze_factor`, GELU, reshape time/channel.
- Encoder FFN `Linear(H -> intermediate_size)`, GELU, `Linear(intermediate_size -> H)`.
- CTC head `Linear(H -> vocab_size)` for ASR.
- Optional classification projector `Linear(H -> classifier_proj_size)` and classifier `Linear(classifier_proj_size -> num_labels)`.

Attention primitives:

- Dense noncausal self-attention only.
- Separate biased Q/K/V/O linear projections, all `H -> H`.
- Matmul QK^T, scale by `head_dim**-0.5`, additive mask, softmax over keys, matmul with V.
- Optional SDPA/FlashAttention backend can replace eager attention if mask format is admitted.

Position/custom math:

- Convolutional positional embedding only; no absolute embedding table, RoPE, ALiBi, or relative bias.
- Positional Conv1d has PyTorch weight norm; loaders should materialize or faithfully implement effective normalized weights.

Preprocessing-coupled ops:

- CPU-side padding/truncation and zero-mean/unit-variance normalization.
- Model-side raw waveform Conv1d; no STFT, FFT, mel filterbank, spectrogram, or log-mel preprocessing.
- Attention-mask downsampling from raw length to feature length through the conv output-length formula.

Training-only/deferred ops:

- Random `_compute_mask_indices`, boolean scatter assignment for SpecAugment, CTC loss, `masked_select(labels)`, and `log_softmax(..., dtype=float32)` for loss are not required for inference.

## 5. Layer/block breakdown

Feature encoder:

```text
input_values: [B, T_raw]
x = input_values[:, None]                         # [B, 1, T_raw]
for i in conv layers:
  x = Conv1d(Cin_i -> Cout_i, K_i, S_i, bias=conv_bias)(x)
  if i == 0 and feat_extract_norm == "group": x = GroupNorm(Cout_i groups)(x)
  if feat_extract_norm == "layer": x = LayerNorm(x.transpose(-2,-1)).transpose(-2,-1)
  x = GELU/activation(x)
features = x                                      # [B, conv_dim[-1], T_feat]
```

Feature projection:

```text
x = features.transpose(1, 2)                      # [B, T_feat, conv_dim[-1]]
x = LayerNorm(conv_dim[-1])(x)
if conv_dim[-1] != hidden_size:
  x = Linear(conv_dim[-1] -> hidden_size)(x)
```

SEW squeeze encoder:

```text
if attention_mask:
  zero padded positions in x
  downsample mask length by integer division through squeeze_factor for encoder attention
n_input_timesteps = T_feat
y = x.transpose(1, 2)                              # [B, H, T_feat]
pos = GELU(SamePad(WeightNormGroupedConv1d(H -> H, Kpos, stride=squeeze_factor))(y))
pool = AvgPool1d(squeeze_factor, squeeze_factor)(y)
y = pool[..., :min_len] + pos[..., :min_len]
y = LayerNorm(y.transpose(1, 2))
repeat N layers:
  attn = MHA(y, mask)
  y = LayerNorm(y + dropout(attn))
  y = LayerNorm(y + FFN(y))
y = Linear(H -> H * squeeze_factor)(y)
y = GELU(y)
y = reshape [B, T_enc, squeeze_factor, H] -> [B, T_enc * squeeze_factor, H]
if y.time < n_input_timesteps: y = pad_time_right(y, n_input_timesteps - y.time)
```

CTC head:

```text
logits = Linear(H -> vocab_size)(dropout(y))       # [B, T_feat-like, vocab]
```

For 16,000 raw samples with sampled conv kernels/strides, the conv stack emits 49 feature frames; the squeeze encoder uses 24 frames, then upsamples to 48 and pads to 49.

## 6. Attention requirements

SEW requires encoder-style dense self-attention:

- Causality: noncausal.
- Type: self-attention in the main path; `SEWAttention` has a `key_value_states` branch, but `SEWEncoderLayer` never uses cross-attention.
- Heads: MHA only. `head_dim = hidden_size // num_attention_heads`; sampled checkpoints use 64.
- Q/K/V width: each projection maps `hidden_size -> hidden_size`, then reshapes to `[B, heads, T_enc, head_dim]`.
- Masking: without FlashAttention, the source builds an additive mask shaped `[B, 1, T_enc, T_enc]` with `torch.finfo(dtype).min` on masked key positions. With FlashAttention requested, it passes a 2D mask or `None` and zeroes padded hidden states first.
- Packed/varlen: no explicit cu-seqlens in SEW source.
- Sliding/local/sparse: none.
- Position interaction: none inside attention; convolutional positional embedding is added before encoder layers.
- Cache: no KV cache, no decode loop, no cache reorder.
- Backend compatibility: `_supports_flash_attn=True`, `_supports_sdpa=True`, `_supports_flex_attn=False`. DinoML can start with eager dense attention and add SDPA/FlashAttention only after mask parity is proven.

## 7. Position encoding and custom math

SEW uses convolutional positional embeddings before encoder layers:

```python
def sew_positional_embedding(x_bht, weight_norm_conv, same_pad, activation):
    # x_bht is [batch, hidden_size, feature_time]
    y = weight_norm_conv(x_bht)       # grouped Conv1d, stride=squeeze_factor
    y = same_pad(y)                   # remove last time step only for even kernels
    return activation(y)
```

The crop rule is:

```python
num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0
if num_pad_remove:
    y = y[:, :, :-1]
```

Weight normalization is source-visible because `SEWPositionalConvEmbedding` wraps its Conv1d with `nn.utils.weight_norm` or `nn.utils.parametrizations.weight_norm`. DinoML can either materialize the effective Conv1d weight at load time or admit a small weight-normalization constant transform; it should not treat `weight_g`/`weight_v` as ordinary independent Conv1d weights.

Attention position math is absent: no RoPE, ALiBi, relative position bias, rotary tables, or learned absolute embedding table.

## 8. Preprocessing and input packing

Processor contract from sampled `preprocessor_config.json` and `Wav2Vec2FeatureExtractor`:

- Input is mono raw speech: one float per timestep. Batched arrays may be rank 2; rank >2 is rejected.
- Sampling rate is 16,000 Hz. Passing a different `sampling_rate` raises; omitting it only warns.
- Output tensor names: `input_values` and optionally `attention_mask`.
- `input_values` dtype is float32 after processor conversion.
- Padding is right-side with value `0.0`; truncation/max length/pad-to-multiple are processor options.
- `do_normalize=true`: each sample is normalized to `(x - mean) / sqrt(var + 1e-7)`. If an attention mask is present during padding, the mean/variance use only unpadded samples and padded tail is reset to `padding_value`.
- Default `return_attention_mask=false` for sampled SEW group-norm checkpoints. Batched inference can still pass one explicitly, but parity should match the checkpoint processor default first.

No STFT, FFT, hop length, window function, mel bins, log compression, or spectrogram tensor exists in the processor. Audio decode and resampling are outside Transformers and should stay in DinoML's data pipeline.

Model input ABI:

```text
input_values:  float32 [B, T_raw]
attention_mask: optional int/bool [B, T_raw], 1 for valid samples, 0 for pad
```

The model converts raw attention masks to feature-frame masks using the conv output-length recurrence:

```python
def conv_out_length(length, kernel, stride):
    return floor((length - kernel) / stride) + 1
for kernel, stride in zip(conv_kernel, conv_stride):
    length = conv_out_length(length, kernel, stride)
```

Then inside `SEWEncoder`, the feature-frame mask is squeezed by integer division through `squeeze_factor` for encoder attention. This is not equivalent to recomputing the full Conv1d output formula again.

## 9. Graph rewrite / lowering opportunities

### Rewrite: static Conv1d -> im2col GEMM

Source pattern: NCL `Conv1d(Cin -> Cout, kernel=K, stride=S, padding=0, dilation=1, groups=1)`.

Replacement: local window extraction over time -> matrix multiply with `weight.reshape(Cout, Cin*K).T` -> optional bias -> NCL reshape.

Preconditions:

- `groups == 1`; do not apply to grouped positional conv.
- `padding == 0`, `dilation == 1`.
- Input layout is source NCL or the layout pass owns both producer and consumer transposes.
- Runtime output length follows PyTorch floor formula.

Failure cases: grouped pos conv, dynamic layouts without explicit stride metadata, and tiny time lengths where source Conv1d would produce invalid/empty outputs.

Parity test sketch: compare every feature extractor conv layer against PyTorch for random `[B,C,T]`, including odd raw lengths and sampled kernels/strides.

### Rewrite: K=1 Conv1d -> pointwise Linear/GEMM

Several feature conv layers use `kernel_size=1`, `stride=1`. These can lower to per-time `Linear(Cin -> Cout)` after an NCL->NTC view or a strided GEMM.

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `groups == 1`.
- Activation/norm ordering remains source-faithful.

Weight transform: Conv1d `[Cout,Cin,1] -> Linear [Cout,Cin]`.

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern: three biased `Linear(H -> H)` calls followed by identical head reshape.

Replacement: concatenate weights in source split order `[q, k, v]`, concatenate biases `[q, k, v]`, run one `Linear(H -> 3H)`, split last dim into q/k/v.

Preconditions:

- Same input tensor for self-attention.
- All three projections present and same dtype/layout.
- Preserve split order `q_proj`, `k_proj`, `v_proj`; source weights are separate PyTorch `Linear` weights `[out,in]`.

Failure cases: if future cross-attention path uses different key/value states, do not pack Q with KV unless both inputs are the same.

### Rewrite: linear upsampling -> fused projection/channel-to-time reshape

Source pattern: `Linear(H -> H*squeeze_factor)` -> activation -> reshape `[B,T,S,H] -> [B,T*S,H]`.

Replacement: fused GEMM epilogue activation plus deterministic time/channel reindex copy, or a custom output-layout GEMM that writes directly to upsampled time order.

Preconditions:

- Static `squeeze_factor`.
- Output hidden width equals `src_embed_dim // squeeze_factor`.
- Consumer accepts row-major `[B,T*S,H]`.

Failure cases: dynamic `squeeze_factor` unsupported; final right-pad still needs a separate guarded pad if `T*S < T_feat`.

### Rewrite: local NCL/NLC transpose elimination

Candidate regions:

- Conv feature stack is naturally NCL.
- Transformer and LayerNorm are naturally BTC/NLC.
- Positional conv/AvgPool region transposes BTC -> BCT -> BTC and can be layout-fused locally.

Layout constraints:

- Do not globally translate audio tensors to NHWC; rank-3 audio needs explicit NCL/NLC policy.
- Axis-sensitive attrs include Conv1d channel axis `1`, LayerNorm normalized last dim, AvgPool1d time axis, mask `[B,T]`, and mean/sum pooling over `dim=1` in classification.

Failure cases: attention and FFN expect hidden dim last for GEMM-friendly lowering; preserve or explicitly rewrite all consumers.

## 10. Kernel fusion candidates

Highest priority:

- Conv1d feature extractor kernels, especially K=1 pointwise Conv1d and small K strided Conv1d. This dominates front-end audio cost before attention.
- LayerNorm over `[B,T,H]` and feature LayerNorm over `[B,T_feat,C]`.
- Packed QKV projection + dense attention + output projection for encoder MHA.
- FFN `Linear + GELU + Linear`, with bias and dropout removed in inference.
- CTC head `Linear(H -> vocab)` fused with last-token/frame logits storage where only logits are needed.

Medium priority:

- Grouped positional Conv1d + SamePad + GELU.
- AvgPool1d + positional Conv1d add/crop region.
- Linear upsampling + GELU + reshape/pad.
- Add + LayerNorm residual fusions in encoder blocks.
- Mask downsampling/expansion kernels when batched padded audio is common.

Lower priority:

- Weighted layer sum for sequence classification; optional head only.
- Processor normalization on GPU; CPU/data pipeline is simpler initially.
- Output attentions materialization; useful for diagnostics, not first ASR inference.

## 11. Runtime staging plan

Stage 1: Parse `SEWConfig`, reject `model_type!="sew"`, load base weights, and run the Conv1d feature extractor plus feature LayerNorm/projection parity.

Stage 2: Implement SEW squeeze block: positional weight-normalized grouped Conv1d, AvgPool1d, crop/add, encoder mask downsampling.

Stage 3: Run one encoder layer parity with eager dense MHA and FFN in fp32.

Stage 4: Full `SEWModel` parity for base checkpoints, including upsampling and final pad-to-feature-length behavior.

Stage 5: Add `SEWForCTC` inference: dropout disabled, `lm_head`, logits parity. Stub CTC loss and training-only SpecAugment.

Stage 6: Add optional sequence classification head if needed: weighted layer sum, projector, mean/masked mean pooling, classifier.

Stage 7: Optimize: Conv1d lowering, packed QKV, SDPA/FlashAttention guarded by mask compatibility, FFN/LN fusions, upsampling layout fusion.

## 12. Parity and validation plan

- Processor parity: compare `Wav2Vec2FeatureExtractor` normalization/padding for mono arrays, with and without `attention_mask`, at 16 kHz.
- Conv length tests: verify `_get_feat_extract_output_lengths` for raw lengths around stride boundaries, including 16,000 -> 49 frames with sampled configs.
- Conv feature stack parity: random waveforms through each Conv1d/norm/activation stage.
- Positional squeeze parity: random `[B,T_feat,H]` through attention-mask zeroing, AvgPool1d, positional Conv1d, crop/add.
- One encoder layer parity: random `[B,T_enc,H]`, additive mask, fp32 tolerance around `1e-5` to `1e-4`.
- Full base model parity: `asapp/sew-tiny-100k` and `asapp/sew-small-100k` to cover no-projection and projection branches.
- CTC logits parity: `asapp/sew-tiny-100k-ft-ls100h`, compare logits only; loss is deferred.
- Optional fp16/bf16 parity: start with looser tolerances around `1e-2` for attention/Conv1d accumulations unless kernels accumulate in fp32.

## 13. Performance probes

- CPU preprocessing throughput: decode/resample excluded vs included, normalization only, padding batch collation.
- Conv feature extractor throughput by raw audio seconds, batch size, and raw sequence length.
- Squeeze block throughput split into AvgPool1d, grouped positional Conv1d, and crop/add.
- Encoder-only throughput over squeezed sequence length `T_enc`.
- Attention backend comparison: eager BMM/softmax/BMM vs SDPA/FlashAttention for `T_enc` from short utterances to long audio.
- Full CTC logits requests/sec for batch-size sweep and utterance-length sweep.
- Memory probes for attention activations with output attentions disabled/enabled.
- Layout pass probe: source-faithful NCL/BTC transposes vs local transpose-eliminated Conv1d/pos-conv regions.

## 14. Skip/defer list

- Training-only SpecAugment random masking and `masked_spec_embed` updates.
- CTC loss, label masking, `masked_select`, and training loss reductions.
- LayerDrop and gradient checkpointing.
- Adapters loaded via `target_lang`/`add_adapter`; current SEW config class does not define adapter fields by default, but CTC source has compatibility hooks.
- `SEWForSequenceClassification` unless a classification workload is requested.
- `output_attentions=True` dense attention tensor return for optimized kernels.
- SEW-D checkpoints and source.
- Multi-GPU/FSDP/DeepSpeed initialization branches.

## 15. Final implementation checklist

- [ ] Parse `SEWConfig` and reject non-`sew` model types.
- [ ] Load Wav2Vec2-style processor metadata for `input_values`/`attention_mask`.
- [ ] Implement CPU/data-pipeline normalization and padding parity.
- [ ] Implement Conv1d feature extractor in NCL layout.
- [ ] Implement GroupNorm-first and LayerNorm-all conv norm variants.
- [ ] Implement feature LayerNorm and optional `Linear(conv_dim[-1] -> hidden_size)`.
- [ ] Implement conv output-length and feature-vector attention-mask generation.
- [ ] Implement SEW positional grouped Conv1d with weight-norm materialization.
- [ ] Implement AvgPool1d squeeze, crop/add, and encoder additive mask construction.
- [ ] Implement encoder MHA without cache.
- [ ] Add packed QKV rewrite with `[q,k,v]` split order.
- [ ] Implement FFN GELU block and residual LayerNorm ordering.
- [ ] Implement upsampling `Linear(H -> H*squeeze_factor)` plus reshape and pad.
- [ ] Implement CTC `lm_head` logits path.
- [ ] Add one-layer, full-model, and CTC-logits parity tests.
- [ ] Benchmark Conv1d, squeeze block, encoder attention, and full CTC throughput.
