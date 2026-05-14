# CLAP Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: clap family; representative checkpoints listed below
Config source: local Transformers config defaults plus Hugging Face config/preprocessor/tokenizer JSON
Source files inspected:
- X:/H/transformers/src/transformers/models/clap/modeling_clap.py
- X:/H/transformers/src/transformers/models/clap/configuration_clap.py
- X:/H/transformers/src/transformers/models/clap/feature_extraction_clap.py
- X:/H/transformers/src/transformers/models/clap/processing_clap.py
- X:/H/transformers/docs/source/en/model_doc/clap.md
- X:/H/transformers/tests/models/clap/test_modeling_clap.py
- X:/H/transformers/tests/models/clap/test_feature_extraction_clap.py
Any missing files or assumptions: tokenizer is RobertaTokenizer via AutoTokenizer; no clap-local tokenization file exists. No remote-code files were required.
```

Representative primary configs inspected from Hugging Face:

- [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused): `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`.
- [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused): same files.
- [sanchit-gandhi/clap-htsat-unfused-s-full-v2](https://huggingface.co/sanchit-gandhi/clap-htsat-unfused-s-full-v2): same files.
- [sanchit-gandhi/clap-htsat-unfused-m-full](https://huggingface.co/sanchit-gandhi/clap-htsat-unfused-m-full): same files.
- [hf-internal-testing/tiny-clap-htsat-unfused](https://huggingface.co/hf-internal-testing/tiny-clap-htsat-unfused): same files.

Report target: inference for audio-text retrieval / zero-shot audio classification through `ClapModel`, plus independently callable text/audio feature extraction. Training loss is documented only for orientation.

## 2. High-level architecture

CLAP is a dual encoder:

```text
raw audio -> CPU log-mel feature extractor -> HTSAT/Swin-style audio encoder -> audio projection -> L2 normalize
text -> Roberta tokenizer -> bidirectional text encoder -> CLS pooler -> text projection -> L2 normalize
normalized features -> two scaled similarity matrices
```

Stage decomposition:

- CPU/data pipeline: waveform validation, mono audio conversion, STFT/log-mel, random crop/fusion packing, Roberta tokenization and padding.
- Audio branch: batch norm over mel bins, log-mel-to-image reshape, Conv2d patch embedding, optional fusion for long clips, windowed/shifted-window audio transformer stages, patch merging, final pooling.
- Text branch: word/position/token-type embeddings, full bidirectional self-attention encoder, first-token pooler.
- Cacheable branch outputs: text embeddings and audio embeddings can be cached independently before final similarity. There is no autoregressive prefill/decode or KV cache.
- Final head: `logits_per_text = text_embeds @ audio_embeds.T * exp(logit_scale_t)` and `logits_per_audio = audio_embeds @ text_embeds.T * exp(logit_scale_a)`.

## 3. Important config dimensions

Default source config:

| Field | Text default | Audio default / CLAP default |
|---|---:|---:|
| `vocab_size` | 50265 | n/a |
| `hidden_size` | 768 | 768 |
| `num_hidden_layers` | 12 | `len(depths)=4` stages, `sum(depths)=12` blocks |
| `num_attention_heads` | 12 | `[4, 8, 16, 32]` by stage |
| `head_dim` | 64 | stage dims `[96, 192, 384, 768]` / heads = 24 |
| `intermediate_size` / MLP | 3072 | `mlp_ratio=4.0` |
| `max_position_embeddings` | 514 | n/a |
| `projection_dim` | 512 | 512 |
| `projection_hidden_act` | relu | relu |
| `num_mel_bins` | n/a | 64 |
| `spec_size` | n/a | 256 |
| `patch_size` / `patch_stride` | n/a | 4 / `(4, 4)` |
| `patch_embeds_hidden_size` | n/a | 96 |
| `window_size` | n/a | 8 |
| `qkv_bias` | text Q/K/V always biased | audio Q/K/V bias defaults true |
| `logit_scale_init_value` | \- | `1 / 0.07`, stored as log parameters during init |

Representative checkpoint sweep:

| Checkpoint | Projection | Text | Audio hidden | Audio patch hidden | Audio depths | Heads | Fusion | Preprocessor truncation |
|---|---:|---|---:|---:|---|---|---|---|
| `hf-internal-testing/tiny-clap-htsat-unfused` | 32 | 3 layers, h=32, 2 heads | 64 | 32 | `[2,2]` | config lists `[2,2,2,2]`, first 2 used | false | `rand_trunc` |
| `laion/clap-htsat-unfused` | 512 | 12 layers, h=768, 12 heads | 768 | 96 | `[2,2,6,2]` | `[4,8,16,32]` | false | `rand_trunc` |
| `laion/clap-htsat-fused` | 512 | same | 768 | 96 | `[2,2,6,2]` | `[4,8,16,32]` | true | `fusion` |
| `sanchit-gandhi/clap-htsat-unfused-s-full-v2` | 512 | same | 768 | 96 | `[2,2,6,2]` | `[4,8,16,32]` | false | `rand_trunc` |
| `sanchit-gandhi/clap-htsat-unfused-m-full` | 512 | same | 1024 | 128 | `[2,2,6,2]` | `[4,8,16,32]` | false | `rand_trunc` |

Effective omitted/default fields to preserve: text `type_vocab_size=1`, `layer_norm_eps=1e-12`, audio `attention_probs_dropout_prob=0.0`, audio `layer_norm_eps=1e-5`, audio `enable_patch_layer_norm=True`, audio `flatten_patch_embeds=True`.

## 3a. Family variation traps

- Fused and unfused checkpoints have different preprocessing and model contracts. Fused uses 4 mel channels and `enable_fusion=True`; unfused commonly uses 1 mel channel and `enable_fusion=False`.
- `is_longer` is not optional for fused execution in source behavior because `ClapAudioEncoder.forward` calls `is_longer.to(...)` when fusion is enabled.
- Feature extraction is stochastic for long clips: `fusion` chooses three random mel crops; `rand_trunc` chooses a random waveform crop.
- The feature extractor default `frequency_min=0`, but LAION preprocessor configs set `frequency_min=50`.
- Audio branch source tensor layout changes several times: feature extractor emits `[B, C, frames, mel]`; model transposes to batch-norm over mel bins, then reshapes to NCHW image-like tensors, then window attention uses NHWC-like `[B,H,W,C]` internally.
- The audio encoder is Swin/HTSAT-style, not a vanilla sequence Transformer. Window partitioning, shifted windows, padding, relative position bias, and patch merging are required.
- Audio `hidden_size` must match `patch_embeds_hidden_size * 2 ** (len(depths)-1)` because the projection layer consumes `config.hidden_size` while the encoder pooler emits that final channel count.
- `fusion_type == "channel_map"` changes the first patch Conv2d input channels by `scale_factor=4`; the inspected representative checkpoints use `fusion_type=None`.
- Text pooling uses the first token, not EOS/EOT pooling.
- Similarity orientation is explicit and asymmetric in scale parameters: text-to-audio and audio-to-text use separate learned logit scales.
- No generation cache exists. Any cache discussion for CLAP should mean branch embedding cache, not KV cache.

## 4. Operator coverage checklist

Tensor/layout ops:

- `transpose`, `permute`, `contiguous`, `view`/`reshape`, `flatten`, `cat`, `gather`, `where`, boolean/equality masks.
- Dynamic padding for shifted-window attention and patch merging.
- `torch.roll` for cyclic shift; window partition/reverse with shape guards.
- L2 normalization and vector norm along the feature dimension.

Neural network primitives:

- Embedding lookups for word, position, and token type embeddings.
- Dense layers for text Q/K/V/output, FFN, pooler, and projection MLP.
- Conv2d for audio patch embedding; fused branch also needs 1x1 Conv2d, wider `mel_conv2d`, BatchNorm2d, ReLU, Sigmoid, AdaptiveAvgPool2d.
- LayerNorm, BatchNorm2d, GELU, ReLU, Tanh.
- AdaptiveAvgPool1d for audio pooling.
- Bicubic and bilinear interpolation. Bicubic is in the model graph; bilinear is in CPU/data feature fusion.

Attention primitives:

- Text: noncausal MHA, full sequence, additive extended attention mask, fp32 softmax in eager path, no cache.
- Audio: local window MHA with per-window learned relative position bias; shifted-window additive mask; no cache.

Preprocessing-coupled ops:

- STFT power spectrogram, mel filter bank matmul, dB/log conversion, random crop, repeat/repeatpad/pad, mel stack.
- Processor returns `input_features` plus `is_longer`; Roberta tokenizer returns `input_ids` and `attention_mask`.

Postprocessing / retrieval:

- Matrix multiply for `logits_per_audio` `[audio_batch, text_batch]` and `logits_per_text` `[text_batch, audio_batch]`.
- Softmax over text labels for zero-shot audio classification is pipeline-level behavior, not part of `ClapModel.forward`.

## 5. Layer/block breakdown

Text embeddings:

```text
position_ids = cumsum(input_ids != pad_id) + pad_id
x = word_embedding(input_ids) + token_type_embedding(token_type_ids) + position_embedding(position_ids)
x = LayerNorm(x, eps=1e-12)
x = dropout(x)
```

Text block, repeated `num_hidden_layers`:

```text
q,k,v = Linear(hidden -> hidden, bias=True)(x)
attn = softmax((q @ k.T) * head_dim^-0.5 + extended_attention_mask, fp32)
x_attn = Linear(hidden -> hidden)(attn @ v)
x = LayerNorm(dropout(x_attn) + residual)
ff = GELU(Linear(hidden -> intermediate)(x))
x = LayerNorm(dropout(Linear(intermediate -> hidden)(ff)) + residual)
```

Text pool/project:

```text
pooled = tanh(Linear(hidden -> hidden)(last_hidden_state[:, 0]))
text_embeds = Linear(hidden -> projection_dim) -> ReLU -> Linear(projection_dim -> projection_dim)
text_embeds = L2Normalize(text_embeds)
```

Audio preprocess inside model:

```text
input_features [B,C,T,F] -> transpose(1,3) -> BatchNorm2d(num_mel_bins) -> transpose(1,3)
reshape_mel2img -> image-like [B,C,256,256] for standard configs
patch_embed Conv2d(C -> patch_hidden, kernel=4, stride=4, padding=0) -> flatten to [B,4096,patch_hidden]
```

Audio stage `i`, repeated by `depths[i]`, with channel `C_i = patch_hidden * 2**i`:

```text
y = LayerNorm(x)
y = view [B,H,W,C_i], pad to window multiple
y = optional roll(-shift,-shift)
windows = partition(y, window_size=8) -> [B*num_windows,64,C_i]
q,k,v = biased Linear(C_i -> C_i)
attn = softmax(q @ k.T / sqrt(head_dim) + relative_position_bias + optional shift_mask)
y = reverse windows, optional roll(+shift,+shift), unpad, reshape [B,H*W,C_i]
x = x + DropPath(Linear(C_i -> C_i)(y))
y = LayerNorm(x)
y = GELU(Linear(C_i -> 4*C_i)(y))
x = x + Linear(4*C_i -> C_i)(y)
```

Patch merging between stages:

```text
view [B,H,W,C] -> pad H/W to even -> cat four checkerboard slices on channel axis
LayerNorm(4*C) -> Linear(4*C -> 2*C, bias=False)
```

Audio final pool/project:

```text
last = LayerNorm(tokens)
last = reshape/reorder to [B,C_final,freq,temp]
latent = AdaptiveAvgPool1d(1)(flatten(last, spatial)) -> [B,C_final]
audio_embeds = Linear(C_final -> projection_dim) -> ReLU -> Linear(projection_dim -> projection_dim)
audio_embeds = L2Normalize(audio_embeds)
```

## 6. Attention requirements

Text attention:

- Noncausal bidirectional self-attention.
- Standard MHA, no GQA/MQA.
- Representative LAION shape: 12 heads, head dim 64, hidden 768, max tokenizer length 512 while config position table has 514.
- Mask is additive extended attention mask broadcastable to `[B, heads, from, to]`.
- Source eager path computes `matmul * scaling`, adds mask, softmax in fp32, casts back to query dtype, dropout, then value matmul.
- `ALL_ATTENTION_FUNCTIONS` can dispatch to configured attention implementations, but no CLAP-specific cache path exists.

Audio attention:

- Noncausal local window self-attention over `window_size * window_size` tokens.
- Representative LAION window size is 8, so each attention matmul is 64x64 per head/window.
- Stage head dims are 24 for LAION and medium configs: `[96/4, 192/8, 384/16, 768/32]`.
- Learned relative position bias table shape is `((2*Wh-1)*(2*Ww-1), heads)` and is gathered by a precomputed relative-position index.
- Shifted-window layers use `torch.roll`, partitioned region IDs, and additive masks with `-100.0`.
- Dynamic dimensions are padded to multiples of window size before partition and unpadded afterward.
- No KV cache, causal mask, ALiBi, RoPE, packed varlen, or cross-attention is required for primary CLAP inference.

## 7. Position encoding and custom math

Text uses learned absolute position embeddings. Position IDs are pad-aware:

```python
mask = input_ids.ne(pad_token_id).int()
position_ids = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
position_ids = position_ids.long() + pad_token_id
```

Audio uses Swin-style relative position bias per window:

```python
coords = meshgrid(arange(Wh), arange(Ww))
relative = flatten(coords)[:, :, None] - flatten(coords)[:, None, :]
relative[..., 0] += Wh - 1
relative[..., 1] += Ww - 1
relative[..., 0] *= 2 * Ww - 1
relative_position_index = relative.sum(-1)
bias = table[relative_position_index.view(-1)].view(Wh*Ww, Wh*Ww, heads)
scores = scores + bias.permute(2, 0, 1).unsqueeze(0)
```

Audio log-mel-to-image reshape is CLAP-specific and axis-sensitive:

```python
spec_width = spec_size * (spec_size // num_mel_bins)
spec_height = spec_size // (spec_size // num_mel_bins)
x = bicubic_resize_if_needed(x, time=spec_width, freq=spec_height)
x = x.reshape(B, C * freq_ratio, T // freq_ratio, F)
x = x.permute(0, 1, 3, 2).contiguous()
x = x.reshape(B, C, F * freq_ratio, T // freq_ratio)
```

For standard 10s / 48 kHz / hop 480 feature extraction, `input_features` has 1001 frames and 64 mel bins. The model resizes time to `1024` and keeps frequency at `64`, then reshapes to `256x256`.

## 8. Preprocessing and input packing

Audio feature extractor contract:

- Input waveform must be mono. Batched 2D numpy is accepted; arrays with rank greater than 2 are rejected.
- Sampling rate default and checkpoint value: 48,000 Hz. Passing a mismatched `sampling_rate` raises.
- Max length: 10 seconds = 480,000 samples.
- STFT: Hann window, `fft_window_size=1024`, `hop_length=480`, power spectrogram, `nb_frequency_bins=513`.
- Mel bins: 64. LAION preprocessor uses `frequency_min=50`, `frequency_max=14000`, `top_db=None`.
- Log-mel output uses dB conversion and is transposed to `[frames, mel]`.
- `truncation="fusion"` uses HTK-style mel filters, full-audio mel, three random crops plus a downsampled whole-audio mel: output `[4, frames, 64]`.
- `truncation="rand_trunc"` uses Slaney mel filters and one random crop: output `[1, frames, 64]`.
- Short audio is padded with `repeatpad`, `repeat`, or `pad`; with `fusion`, short clips still produce four identical mels.
- Returned tensors: `input_features` `[B, C, frames, 64]` and `is_longer` `[B,1]`.
- Source feature extraction is CPU/data-pipeline work for first integration. GPU STFT/mel can be a later optimization.

Text input contract:

- Processor composes `ClapFeatureExtractor` and `RobertaTokenizer`.
- Tokenizer special tokens: BOS/CLS `<s>`, EOS/SEP `</s>`, PAD `<pad>`, UNK `<unk>`, MASK `<mask>`.
- `model_max_length=512`; config has 514 positions because pad-aware positions start after pad id.
- Token type IDs default to all zeros from a persistent buffer; `type_vocab_size=1`.
- GPU graph consumes `input_ids`, optional `attention_mask`, optional `position_ids`.

Dual-encoder cache contract:

- `get_text_features` and `get_audio_features` run independently and return normalized `pooler_output` in `BaseModelOutputWithPooling`.
- `ClapTextModelWithProjection` and `ClapAudioModelWithProjection` return unnormalized projected embeddings; `ClapModel` and `get_*_features` normalize.
- Cache branch embeddings after projection+normalization for retrieval/classification. Recompute only final similarity for new opposite-branch batches.

## 9. Graph rewrite / lowering opportunities

### Rewrite: audio patch Conv2d to patch GEMM

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0` for representative configs where patch size equals stride.
- `dilation == 1`, `groups == 1`.
- Input is the model's image-like NCHW tensor after `reshape_mel2img`.
- Height/width match `spec_size` or have already been guarded/resized by source behavior.

Replacement:

```text
NCHW -> non-overlap WindowFlatten [B, H/ps*W/ps, C*ps*ps]
MatMul(weight.reshape(out_channels, -1).T) + bias
LayerNorm over out_channels
```

Failure cases: `fusion_type=="channel_map"` changes input channels; nonzero padding must preserve Conv2d semantics; dynamic sizes must preserve source shape errors.

Parity test sketch: compare patch embed output for fused and unfused configs before first audio stage.

### Rewrite: audio window attention to batched attention primitive

Preconditions:

- Window size known; partitioned tokens are contiguous `[B*nW, Ws*Ws, C]`.
- Relative bias gather is precomputed or materialized as `[heads, Ws*Ws, Ws*Ws]`.
- Shift mask is either absent or additive with source `-100.0` values.

Replacement:

```text
QKV linears -> reshape heads -> scaled_dot_product_attention(Q,K,V,bias+mask) -> output linear
```

Failure cases: do not fuse across `roll`/partition/reverse unless layout pass owns the whole window region.

### Rewrite: patch merging to layout-aware gather + GEMM

Preconditions:

- Input token sequence can be viewed as `[B,H,W,C]`.
- Checkerboard concat order must match source: `for col in range(2) for row in range(2)`.
- Odd H/W padding is preserved.

Replacement:

```text
Four-slice gather/concat -> LayerNorm(4C) -> Linear(4C -> 2C, no bias)
```

### Rewrite: branch feature cache

Preconditions:

- Model in eval mode; projection and normalization included in cached value.
- Text tokenization/position/mask and audio preprocessing configs are identical.
- Cache key includes checkpoint, projection weights, tokenizer/preprocessor config, and raw input identity or processed tensor hash.

Replacement:

```text
cached_text_embeds @ fresh_audio_embeds.T
fresh_audio_embeds @ cached_text_embeds.T
```

Failure cases: `ClapTextModelWithProjection` / `ClapAudioModelWithProjection` output unnormalized features, so cache must record normalization status.

### Layout guard: audio model source axes

Candidate optimized layout: keep Conv2d/BatchNorm in NCHW or lower locally to NHWC only inside a controlled fused region. Guard `reshape_mel2img`, window partition/reverse, patch merging, `AdaptiveAvgPool1d`, and relative-bias attention as no-layout-translation regions until an axis rewrite pass handles their exact semantics.

## 10. Kernel fusion candidates

Highest priority:

- Text encoder LayerNorm + residual + dense patterns: standard BERT/RoBERTa throughput path.
- Audio window attention with relative bias and shift mask: dominant HTSAT operation; window size is small and fixed for common checkpoints.
- Audio patch Conv2d lowered to GEMM/implicit GEMM: first large layout conversion point.
- Projection MLP + L2 normalize + similarity GEMM: directly affects retrieval serving.

Medium priority:

- Patch merging gather + LayerNorm + Linear.
- Audio final reshape/pool sequence.
- Fused feature AFF block for `laion/clap-htsat-fused`: 1x1 Conv2d/BN/ReLU/Sigmoid and indexed update for long clips.
- Batched text/audio similarity for many labels and many audio clips.

Lower priority:

- GPU log-mel/STFT feature extraction.
- Dropout/drop path, needed only if training or exact train-mode parity is in scope.
- Training contrastive loss.

## 11. Runtime staging plan

1. Parse configs and load weights for `ClapTextModel`, `ClapAudioModel`, projection layers, and logit-scale parameters.
2. Implement processor-compatible input contracts using CPU feature extraction and Roberta tokenization from Transformers or a faithful external pipeline.
3. Bring up text-only encoder parity using existing BERT/RoBERTa-style ops.
4. Bring up audio unfused encoder parity for `laion/clap-htsat-unfused`: batch norm, reshape, patch embedding, window attention, patch merging, pooler.
5. Add projection, L2 normalization, and similarity orientation parity for `ClapModel`.
6. Add fused audio path: 4-channel features, `is_longer` indexing, local/global AFF fusion, and long/short clip behavior.
7. Add optimized rewrites: Conv2d patch GEMM, window attention fusion, branch embedding cache, similarity batching.

Initially stub/defer stochastic exactness by accepting precomputed `input_features` and `is_longer`; then validate feature extraction separately with fixed random seeds.

## 12. Parity and validation plan

- Feature extractor parity: compare log-mel tensors against Transformers for short/long audio, `fusion` and `rand_trunc`, `repeatpad`/`repeat`/`pad`; tolerance `1e-4` fp32 after CPU preprocessing.
- Text embedding parity: one layer, then full text encoder, then `get_text_features`; tolerance `1e-4` fp32, `1e-2` fp16.
- Audio block parity: patch embed output, one non-shifted window block, one shifted block, one patch merge, then full audio encoder.
- Fused branch parity: batches with mixed `is_longer` values; verify indexed AFF update changes only long rows.
- Projection/normalization parity: verify `ClapModel.get_*_features` are unit norm and match Transformers.
- Similarity parity: verify both orientations and separate logit scales with unequal audio/text batch sizes.
- End-to-end parity: zero-shot audio classification over a small label set using `logits_per_audio.softmax(dim=-1)`.

## 13. Performance probes

- CPU feature extraction throughput: seconds of audio/sec for `fusion` and `rand_trunc`.
- Text branch throughput: batch size and sequence length sweep, especially many-label classification.
- Audio branch throughput: batch size sweep, long-vs-short fused batch mix, stage-level timing.
- Window attention backend comparison: eager small-window matmul versus fused window-attention kernel.
- Projection/similarity throughput: audio batch x text batch matrix sizes.
- End-to-end requests/hour split into preprocessing, text cache lookup/compute, audio encoder, similarity.
- Memory probes: audio hidden-state peak by stage, fused 4-channel input overhead, cached text embedding table size.

## 14. Skip/defer list

- Training and contrastive loss.
- Gradient checkpointing, dropout/drop path train-mode behavior.
- GPU log-mel extraction for first runtime parity.
- Remote-code or non-HTSAT CLAP variants not represented by native `ClapConfig`.
- `fusion_type=="channel_map"` unless a checkpoint requiring it is selected.
- Automatic stochastic crop reproducibility beyond fixed-seed validation.
- Autoregressive generation, KV cache, beam search, speculative decoding: not applicable.
- Multi-GPU/tensor parallel and quantization.

## 15. Final implementation checklist

- [ ] Parse `ClapConfig`, `ClapTextConfig`, and `ClapAudioConfig`.
- [ ] Load Roberta text embeddings/encoder/pooler weights.
- [ ] Load HTSAT audio patch embed, relative-bias, stage, patch-merge, norm, pool/projection weights.
- [ ] Implement CPU preprocessing contract for `input_features` and `is_longer`.
- [ ] Implement text encoder parity.
- [ ] Implement audio unfused encoder parity.
- [ ] Implement audio fused/AFF path parity.
- [ ] Implement projection MLPs and L2 normalization.
- [ ] Implement `logits_per_text` and `logits_per_audio` with separate logit scales.
- [ ] Add branch embedding cache contract.
- [ ] Add Conv2d patch embedding rewrite with layout guards.
- [ ] Add window-attention fusion with relative-bias and shift-mask parity.
- [ ] Add per-stage audio parity tests and end-to-end retrieval/classification tests.
- [ ] Benchmark preprocessing, text branch, audio branch, and similarity separately.
