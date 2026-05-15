# Transformers LASR audit for DinoML

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: lasr
Primary model id: google/medasr
Primary runtime target: automatic speech recognition with LasrForCTC
Config source: local source defaults plus downloaded open Hub config snapshots
```

Source files inspected:

- `transformers/src/transformers/models/lasr/modular_lasr.py`
- `transformers/src/transformers/models/lasr/modeling_lasr.py`
- `transformers/src/transformers/models/lasr/configuration_lasr.py`
- `transformers/src/transformers/models/lasr/feature_extraction_lasr.py`
- `transformers/src/transformers/models/lasr/processing_lasr.py`
- `transformers/src/transformers/models/lasr/tokenization_lasr.py`
- `transformers/tests/models/lasr/test_modeling_lasr.py`
- `transformers/src/transformers/pipelines/automatic_speech_recognition.py`
- `transformers/src/transformers/masking_utils.py`

Source URLs:

- [modular_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/modular_lasr.py)
- [modeling_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/modeling_lasr.py)
- [configuration_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/configuration_lasr.py)
- [feature_extraction_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/feature_extraction_lasr.py)
- [processing_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/processing_lasr.py)
- [tokenization_lasr.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lasr/tokenization_lasr.py)

`modeling_lasr.py`, `configuration_lasr.py`, `processing_lasr.py`, and `tokenization_lasr.py` are generated from `modular_lasr.py`; future Transformers source edits should start from the modular file. This report treats generated files as the immediate runtime source basis because those are what users import.

Config and processor snapshots saved beside this report:

- `hf-internal-testing__lasr-test__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `samwell__medasr-ghana__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `samwell__medasr-ghana-v2__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `gabrielbuzzi__medasr-public__config.json`, `__preprocessor_config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `CharlieKingOfTheRats__medasr-mil__config.json`, `__processor_config.json`, `__tokenizer_config.json`
- `hub_config_fetch_summary.json`

Missing files or assumptions:

- [google/medasr](https://huggingface.co/google/medasr) is gated. Hub metadata is visible, but config and processor files returned 403 without access. Access would resolve the official config/processor snapshots.
- Open mirrors and finetunes inspected all expose the same operator-significant LASR CTC dimensions as the source default.
- No remote-code files are required for the in-library LASR implementation.

## 2. High-level architecture

LASR is an audio-only Conformer-style encoder with a CTC projection head. It is not an autoregressive decoder and has no decode-time KV cache. `LasrForCTC.generate` is greedy CTC argmax over frame logits, followed by tokenizer CTC-style duplicate and blank removal during decode.

Dataflow:

```text
raw mono audio
  -> CPU/data-pipeline log-mel feature extraction
  -> temporal subsampling block
  -> repeated Conformer encoder blocks
  -> final LayerNorm
  -> 1x1 Conv1d CTC head
  -> frame logits
  -> greedy argmax
  -> CTC tokenizer decode
```

Stage decomposition:

- CPU/data pipeline: audio decode/resampling outside the model, mono conversion, padding, STFT-like framing, RFFT, mel projection, log clamp, attention mask construction.
- GPU/runtime encoder: `input_features[B,T_mel,128]` plus optional boolean `attention_mask[B,T_mel]`.
- Independently optimizable regions: feature extraction, subsampling conv stack, Conformer block, dense noncausal attention, convolution module, CTC head, tokenizer/postprocess.
- First useful DinoML target: `LasrForCTC` inference logits and greedy token IDs for ASR.
- Deferred for first target: training CTC loss, pipeline chunk stitching, KenLM decoder files in some repos.

## 3. Important config dimensions

Default/open checkpoint dimensions:

| Field | Value | Provenance |
| --- | ---: | --- |
| `model_type` | `lasr_ctc` | config.json |
| `vocab_size` | 512 | config.json |
| `pad_token_id` / CTC blank | 0 | config.json and tokenizer |
| `hidden_size` | 512 | encoder config |
| `num_hidden_layers` | 17 | encoder config |
| `num_attention_heads` | 8 | encoder config |
| `num_key_value_heads` | 8 | encoder config/source sets equal |
| `head_dim` | 64 | inferred from `hidden_size // num_attention_heads` |
| `intermediate_size` | 2048 | encoder config |
| `hidden_act` | `silu` | encoder config |
| `attention_bias` | false | encoder config |
| `convolution_bias` | false | encoder config |
| `conv_kernel_size` | 32 | encoder config |
| `subsampling_conv_kernel_size` | 5 | encoder config |
| `subsampling_conv_stride` | 2 | encoder config |
| `subsampling_conv_channels` | 256 | encoder config |
| `inputs_to_logits_ratio` | 4 | source property, stride squared |
| `num_mel_bins` | 128 | encoder/preprocessor config |
| `max_position_embeddings` | 10000 | encoder config |
| `rope_parameters` | default, theta 10000.0 | config.json |
| `layer_norm_eps` | 1e-6 | encoder config |
| `batch_norm_momentum` | 0.01 | encoder config |
| `feed_forward_residual_weights` | `[1.5, 0.5]` | encoder config |
| `conv_residual_weights` | `[2.0, 1.0]` | encoder config |
| `dtype` | `float32` when present | config.json in finetunes; absent in some snapshots |
| `use_cache` | false when present | config.json metadata; source does not use AR cache |

Feature extractor defaults:

| Field | Value | Runtime effect |
| --- | ---: | --- |
| `sampling_rate` | 16000 | processor validates caller rate |
| `feature_size` | 128 | mel bins and model input width |
| `n_fft` | 512 | RFFT output bins = 257 |
| `win_length` | 400 | frame window length |
| `hop_length` | 160 | frame stride and LASR pipeline chunk alignment |
| `padding_side` | right | padded audio and masks |
| `return_attention_mask` | true | required for reliable batched inference |

Representative checkpoint sweep:

| Repo | Access | Task/class metadata | Operator-significant differences |
| --- | --- | --- | --- |
| [google/medasr](https://huggingface.co/google/medasr) | gated | ASR, `lasr_ctc`, medical/radiology tags | Config inaccessible; source doc default target. |
| [hf-internal-testing/lasr-test](https://huggingface.co/hf-internal-testing/lasr-test) | open | `lasr_ctc`, test checkpoint | Same 17x512 encoder, vocab 512, 128-bin fbank. |
| [samwell/medasr-ghana](https://huggingface.co/samwell/medasr-ghana) | open | `AutoModelForCTC`, 105.3M params from Hub metadata | Same core dimensions; finetune metadata only. |
| [samwell/medasr-ghana-v2](https://huggingface.co/samwell/medasr-ghana-v2) | open | `AutoModelForCTC`, 105.3M params from Hub metadata | Same core dimensions; includes `use_cache: false`. |
| [gabrielbuzzi/medasr-public](https://huggingface.co/gabrielbuzzi/medasr-public) | open mirror | ASR, `lasr_ctc` | Same core dimensions; processor type strings use legacy uppercase `LASR*`. |
| [CharlieKingOfTheRats/medasr-mil](https://huggingface.co/CharlieKingOfTheRats/medasr-mil) | open | ASR, `lasr_ctc` | Same core dimensions; no `preprocessor_config.json`, but nested `processor_config.json` carries feature extractor config. |

## 3a. Family variation traps

- Source config permits custom dimensions; do not hard-code 512/17/8 except for known checkpoint allowlists.
- `num_key_value_heads` is set equal to `num_attention_heads` in `LasrEncoderConfig.__post_init__`; observed LASR configs are MHA, not GQA/MQA. Reject or separately audit configs that try to make this GQA through historical fields.
- `head_dim` can be present as an attribute in attention/RoPE code; if present, projection width is `num_heads * head_dim`, not necessarily `hidden_size`.
- `attention_bias` controls Q/K/V/O projections and feed-forward Linear biases. `convolution_bias` controls Conv1d biases separately.
- Conformer convolution uses `Conv1d(..., padding="same", kernel_size=32)`. Even kernel SAME padding is axis-sensitive and should not be silently rewritten without parity tests.
- Source tensors are time-major sequences `[B,T,C]` around Linear/LayerNorm, then transposed to `[B,C,T]` for Conv1d/BatchNorm1d. Any channel-last optimization needs guarded transpose elimination and Conv1d axis rewrites.
- Subsampling convs have no padding and use two stride-2 Conv1d layers. Output length is not simply `ceil(T/4)`; it is `floor((floor((T - k)/s + 1) - k)/s + 1)`.
- `LasrFeatureExtractor` creates mel features with custom `unfold(..., win_length, hop_length)` plus `rfft`, not `torch.stft`. Treat feature extraction as model-coupled preprocessing.
- Open processor configs vary in class-name capitalization (`LASRFeatureExtractor`/`LASRProcessor` versus `LasrFeatureExtractor`/`LasrProcessor`). This is loader metadata, not a graph difference.
- Tokenizer configs use `<epsilon>` as pad/blank token, while source tokenizer defaults say `<pad>`. Runtime should honor checkpoint tokenizer metadata for decode.
- Pipeline chunking has LASR-specific alignment: `inputs_to_logits_ratio * hop_length`, which is `4 * 160 = 640` samples for default configs.
- `use_cache: false` is metadata only for CTC; there is no autoregressive cache path in source.

## 4. Operator coverage checklist

Tensor/layout ops:

- `transpose(1,2)`, `view`, `reshape`, `contiguous`.
- Dynamic output length computation from masks: two rounds of `(length - kernel_size) // stride + 1`.
- Boolean mask construction: `arange(max_length) < output_lengths[:, None]`.
- Broadcasted additive attention mask handling for eager attention.
- `masked_fill` in convolution module, with mask reduced from 4D attention mask.

Neural network primitives:

- Linear `128 -> hidden_size` for fbank input projection.
- Conv1d subsampler: `hidden_size -> hidden_size`, kernel 5, stride 2, no padding.
- Conv1d subsampler: `hidden_size -> subsampling_conv_channels`, kernel 5, stride 2, no padding.
- Linear `subsampling_conv_channels -> hidden_size`.
- LayerNorm over last dim, no bias, eps 1e-6.
- Feed-forward Linear `hidden_size -> intermediate_size -> hidden_size`, SiLU, optional dropout disabled at inference.
- Pointwise Conv1d `hidden_size -> 2*hidden_size`, kernel 1, GLU over channel dim.
- Depthwise Conv1d `hidden_size -> hidden_size`, kernel `conv_kernel_size`, groups `hidden_size`, padding `same`.
- BatchNorm1d over channel dim, momentum 0.01; inference requires running mean/variance.
- Pointwise Conv1d `hidden_size -> hidden_size`, kernel 1.
- Final LayerNorm.
- CTC head Conv1d `hidden_size -> vocab_size`, kernel 1.

Attention primitives:

- Noncausal self-attention only.
- Q projection `hidden_size -> num_attention_heads * head_dim`.
- K/V projection `hidden_size -> num_key_value_heads * head_dim`.
- RoPE on Q/K before attention.
- Dense attention score matmul, scale by `head_dim ** -0.5`, mask add, softmax in float32, value matmul, output projection.
- SDPA-compatible source path; FlashAttention disabled because custom attention bias is not supported in this source.

Position/rotary ops:

- Default RoPE with theta 10000 over head_dim.
- Cos/sin generated from runtime sequence length after subsampling.
- Optional non-default RoPE source hook exists through `ROPE_INIT_FUNCTIONS`; observed configs use default.

Generation/cache ops:

- No KV cache.
- Greedy `argmax(logits, dim=-1)`.
- Mask padded output token positions to `pad_token_id`.
- CTC decode groups repeated token IDs and removes blank/pad token.

Preprocessing-coupled ops:

- Mono conversion by channel mean when input has more than one channel.
- Right padding and optional truncation before feature extraction.
- Hann window, frame `unfold`, RFFT, magnitude squared, mel matrix multiply, clamp min `1e-5`, natural log.
- Attention mask downsampling at fbank stage: `padded_attention_mask[:, win_length - 1 :: hop_length]`.

Training-only or deferred:

- `log_softmax`, flattened target selection, CTC loss with cuDNN disabled.
- LayerDrop and dropout.
- Pipeline chunk/stride recombination and external language-model decoding.

## 5. Layer/block breakdown

Feature extraction:

```text
raw_speech[B,S] float32
frames = unfold(raw_speech, win_length=400, hop_length=160)
stft = rfft(hann_window * frames, n_fft=512)
power = abs(stft) ** 2
mel = power @ mel_filters[257,128]
input_features = log(clamp(mel, min=1e-5))  # [B,T_mel,128]
attention_mask = padded_mask[:,399::160]    # [B,T_mel]
```

Subsampling:

```text
x = ReLU(Linear(128 -> H)(input_features))   # [B,T,H]
x = transpose to [B,H,T]
x = ReLU(Conv1d(H -> H, k=5, s=2, pad=0)(x))
x = ReLU(Conv1d(H -> Csub, k=5, s=2, pad=0)(x))
x = transpose to [B,T_sub,Csub]
x = Linear(Csub -> H)(x)
```

Default dimensions: `H=512`, `Csub=256`, `T_sub = floor((floor((T_mel - 5)/2 + 1) - 5)/2 + 1)`.

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
y = FeedForward(LayerNorm(x))
x = 1.5 * residual + 0.5 * y

y = LayerNorm(x)
q = Linear(H -> num_heads*head_dim, bias=attention_bias)(y)
k = Linear(H -> kv_heads*head_dim, bias=attention_bias)(y)
v = Linear(H -> kv_heads*head_dim, bias=attention_bias)(y)
q,k = RoPE(q,k, cos, sin)
y = NoncausalAttention(q,k,v, bidirectional_padding_mask)
x = x + Linear(num_heads*head_dim -> H, bias=attention_bias)(y)

y = LayerNorm(x)
y = transpose [B,T,H] -> [B,H,T]
y = Conv1d(H -> 2H, k=1, bias=convolution_bias)(y)
y = GLU(y, dim=channel)
y = masked_fill(padded_positions, 0)
y = DepthwiseConv1d(H -> H, k=conv_kernel_size, padding="same", groups=H)(y)
y = BatchNorm1d(H)(y)
y = SiLU(y)
y = Conv1d(H -> H, k=1, bias=convolution_bias)(y)
y = transpose [B,H,T] -> [B,T,H]
x = 2.0 * x + 1.0 * y

residual = x
y = FeedForward(LayerNorm(x))
x = 1.5 * residual + 0.5 * y
x = LayerNorm(x)
```

Output head:

```text
x = out_norm(x)
logits = Conv1d(H -> vocab_size, k=1)(transpose(x,1,2)).transpose(1,2)
tokens = argmax(logits, dim=-1)
tokens[~subsampled_attention_mask] = pad_token_id
```

## 6. Attention requirements

- Type: encoder-only, noncausal self-attention.
- Head form: observed configs are MHA, `num_attention_heads=8`, `num_key_value_heads=8`, `head_dim=64`.
- Query/key/value width: default Q/K/V logical width 512, but source computes from `num_heads * head_dim`.
- Query length equals key/value length: `T_sub`.
- Masking: bidirectional padding mask from subsampled audio mask; no causal mask.
- Packed/varlen: no source-level packed sequence metadata or cu-seqlens.
- Sliding/local attention: none.
- Relative bias: no learned relative bias; RoPE applied to Q/K.
- KV cache: not applicable.
- Backend compatibility: `_supports_sdpa=True`, `_supports_flash_attn=False`, `_supports_flex_attn=False`. Eager path performs score matmul, mask add, softmax with fp32 accumulation, dropout, then value matmul.
- Mask skip: `create_bidirectional_mask` may return `None` for SDPA-compatible full unpadded attention; DinoML should preserve semantic equivalence rather than requiring a materialized mask in that case.

## 7. Position encoding and custom math

LASR uses default RoPE over the subsampled time axis. Cos/sin are computed once per encoder forward from `arange(T_sub)`.

```python
def lasr_default_rope(config):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
    return inv_freq

def lasr_apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

Precomputable:

- `inv_freq` and mel filter matrix.
- Cos/sin table up to `max_position_embeddings` if sequence length policy is bounded.

Runtime-dependent:

- Actual `T_sub` from audio length and subsampling.
- Position IDs are `0..T_sub-1`.
- Dynamic RoPE variants are source-supported through generic Transformers utilities but not observed in representative configs.

## 8. Preprocessing and input packing

Audio ABI:

- Caller supplies mono raw speech, or multi-channel audio that the feature extractor averages over the last axis.
- Expected sample rate is 16000 Hz; mismatch raises.
- Padding is right-side, value 0.0.
- Output model tensor is `input_features[B,T_mel,128]` float32 unless converted by caller to model dtype.
- `attention_mask[B,T_mel]` is boolean when requested.

Feature extraction math:

- `torch.hann_window(win_length=400, periodic=False, dtype=float64)`.
- `waveform.unfold(-1, 400, 160)`, then `torch.fft.rfft(..., n=512)`.
- Power spectrum `abs(stft) ** 2`.
- Mel filter bank uses Kaldi mel scale, lower 125 Hz, upper 7500 Hz, 128 bins, and zeros the DC bin through padding.
- `log(clamp(power @ mel_filters, min=1e-5))`.

Text/tokenizer ABI for labels and CTC decode:

- Tokenizer is Unigram-backed with metaspace preprocessing.
- CTC decode groups consecutive duplicate token IDs, then removes pad/blank token.
- Checkpoint tokenizer configs use pad token `<epsilon>` and eos `</s>`.
- Labels are only needed for training loss, not first inference.

Pipeline chunking:

- ASR pipeline aligns chunk sizes to `inputs_to_logits_ratio * feature_extractor.hop_length`.
- Default align-to is 640 raw samples.
- Chunk stitching is pipeline-level postprocessing, not part of the model graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: 1x1 Conv1d as Linear

Source pattern:

```text
transpose [B,T,C] -> [B,C,T] -> Conv1d(C -> O, k=1) -> transpose back
```

Replacement:

```text
Linear(C -> O) on [B,T,C]
```

Preconditions:

- `kernel_size == 1`, `stride == 1`, `padding == 0`, `dilation == 1`, `groups == 1`.
- Weight transform `linear.weight = conv.weight[:, :, 0]`; bias unchanged.
- Applies to convolution module pointwise convs and CTC head. For GLU pointwise conv, preserve output split along channel dim into `[A,B]` before `A * sigmoid(B)`.

Failure cases:

- Do not apply to depthwise conv or subsampling convs.
- Do not remove transposes if downstream BatchNorm/Conv remains channel-first.

Parity test sketch:

- Random `[B,T,H]`, compare Conv1d path against transformed Linear for fp32/fp16 with exact same weights.

### Rewrite: subsampling Conv1d to im2col plus GEMM

Source pattern:

```text
[B,H,T] -> Conv1d(H -> H, k=5, s=2, pad=0) -> ReLU -> Conv1d(H -> Csub, k=5, s=2, pad=0)
```

Replacement:

```text
WindowExtract1d -> GEMM(weight_flat.T) -> BiasAdd -> ReLU
```

Preconditions:

- Conv1d rank is `[B,C,T]`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Dynamic output length must use PyTorch floor equation.
- Weight flatten preserves channel-major PyTorch Conv1d layout `[out_channels, in_channels, kernel]`.

Failure cases:

- Very short `T` should fail or route to PyTorch-equivalent fallback.
- Do not use SAME-padding rewrite for these no-padding convs.

### Rewrite: Conformer depthwise Conv1d specialized kernel

Source pattern:

```text
DepthwiseConv1d(H -> H, k=32, groups=H, padding="same") -> BatchNorm1d -> SiLU
```

Replacement:

```text
FusedDepthwiseConv1dSame + frozen BatchNorm affine + SiLU
```

Preconditions:

- Inference mode with BatchNorm running stats frozen.
- `groups == channels`, `stride == 1`, `dilation == 1`.
- SAME padding behavior must match PyTorch for even kernel size 32.
- Masked padded positions are zeroed before convolution.

Failure cases:

- Training mode or unfrozen BatchNorm.
- Any changed conv kernel size without SAME parity coverage.

### Rewrite: dense noncausal MHA to SDPA/Flash-style attention

Source pattern:

```text
Q,K,V projections -> RoPE(Q,K) -> scores * scale -> mask add -> softmax(fp32) -> V -> O projection
```

Replacement:

```text
Fused noncausal attention with RoPE prepass and padding mask
```

Preconditions:

- No causal mask.
- No sliding/local pattern.
- Preserve fp32 softmax math and mask values.
- Observed MHA has `kv_heads == heads`; if custom configs create GQA, repeat/expanded KV semantics must be tested.

Failure cases:

- FlashAttention source flag is disabled due to custom attention bias; DinoML optimized attention must prove parity with the bidirectional mask path.

### Rewrite: residual scale/add fusion

Source pattern:

```text
x = a * residual + b * branch
```

Replacement:

```text
fused axpby
```

Preconditions:

- Static residual weights from config.
- Same shape and dtype.
- Preserve accumulation dtype policy.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm without bias over `[B,T,H]`: appears five times per block plus output norm.
- Linear/Conv1d GEMM paths: Q/K/V/O, FFN, subsampling dense, pointwise conv, and CTC head dominate arithmetic.
- RoPE plus noncausal attention: needed for encoder parity and throughput.
- Depthwise Conv1d SAME plus BatchNorm plus SiLU: repeated 17 times and currently layout-transpose heavy.
- GLU pointwise convolution: `Conv1d H->2H` plus split/sigmoid/multiply.

Medium priority:

- Subsampling Conv1d lowering: important for long audio but only two layers.
- Residual scale/add fusion for LASR-specific `[1.5,0.5]` and `[2.0,1.0]` combinations.
- Mask downsampling and padded-position zeroing fused into the convolution module.
- CTC head 1x1 Conv1d to Linear/GEMM.

Lower priority:

- CPU/GPU log-mel feature extraction ownership. It is important end-to-end, but can stay in the data pipeline for first DinoML graph parity.
- Pipeline chunk stitching and stride handling.
- Training CTC loss and label packing.

## 11. Runtime staging plan

Stage 1: parse `LasrCTCConfig` and `LasrEncoderConfig`, load open checkpoint weights, and run shape-only graph construction for `input_features[B,T,128]`.

Stage 2: implement feature-tensor encoder parity using precomputed `input_features` and `attention_mask`; stub raw audio preprocessing outside DinoML.

Stage 3: validate one block and full encoder in fp32 against Transformers eager attention.

Stage 4: add CTC head and greedy `argmax`, including output mask fill with `pad_token_id`.

Stage 5: add tokenizer/postprocess compatibility for CTC duplicate grouping and blank removal, or keep decode outside DinoML while validating token IDs.

Stage 6: optimize attention, LayerNorm, FFN GEMMs, pointwise Conv1d-as-Linear, and depthwise Conv1d.

Stage 7: optionally own log-mel feature extraction and ASR pipeline chunking once core model parity is stable.

Initial stubs:

- Dropout and LayerDrop disabled by eval mode.
- Training `labels` and CTC loss.
- KenLM/language-model decoder files.
- Raw audio decode/resample.

## 12. Parity and validation plan

- Feature extractor parity: compare `LasrFeatureExtractor` log-mel output for synthetic sine/noise and padded batches; fp32 tolerance around `1e-5` to `1e-4` if matching torch exactly.
- Subsampling length tests: random masks and lengths through `_get_subsampling_output_length`; verify DinoML output mask shape and contents.
- Subsampling block parity: random `[B,T,128]` for several `T`, compare after each dense/conv/ReLU step.
- RoPE parity: compare cos/sin and rotated Q/K for dynamic `T_sub`.
- Attention parity: one layer eager attention with and without padding mask; fp32 tolerance `1e-4`, fp16/bf16 tolerance source-dependent around `1e-2`.
- Convolution module parity: include padded rows because source zeros masked positions before depthwise conv.
- Full block parity: one and two blocks, then 17-block full encoder.
- CTC head parity: compare logits and greedy token IDs.
- End-to-end token parity: use `hf-internal-testing/lasr-test` and the local Transformers expected-token regression from `tests/models/lasr/test_modeling_lasr.py`.
- Decode parity: compare `processor.batch_decode(predicted_ids, skip_special_tokens=True)`.

## 13. Performance probes

- Feature extraction throughput: raw seconds of audio per second, CPU versus optional GPU preprocessing.
- Encoder-only throughput over `B` and `T_mel` sweeps.
- Attention backend comparison: eager dense, SDPA-compatible, DinoML fused noncausal attention.
- Depthwise Conv1d module benchmark with SAME padding and BatchNorm folded/unfolded.
- Subsampling Conv1d benchmark for raw long-audio inputs.
- CTC head and argmax throughput.
- End-to-end ASR pipeline chunk sweep: `chunk_length_s`, `stride_length_s`, and batch size.
- Memory probes: peak temporaries for attention scores `[B,heads,T_sub,T_sub]` versus fused attention.
- Dtype probes: fp32, bf16, fp16 logits/token stability.

## 14. Skip/defer list

- Training and CTC loss.
- Gradient checkpointing.
- Dropout and LayerDrop stochastic behavior.
- Beam search or external language-model decoding.
- Pipeline chunk recombination for first graph parity.
- Raw audio decode/resampling.
- Non-default RoPE variants unless a checkpoint config requires them.
- FlashAttention parity until custom bidirectional padding-mask behavior is explicitly validated.
- Multi-GPU/tensor parallel execution.
- Quantized or packed weights; no LASR-specific packed weight format was observed.

## 15. Final implementation checklist

- [ ] Parse `LasrCTCConfig` and nested `LasrEncoderConfig`.
- [ ] Admit `lasr_ctc` checkpoints with observed MHA/default-RoPE config.
- [ ] Load encoder, convolution, BatchNorm, and CTC head weights.
- [ ] Implement or import log-mel feature extraction boundary.
- [ ] Implement subsampling Linear/Conv1d stack and length formula.
- [ ] Implement bias-free LayerNorm.
- [ ] Implement LASR RoPE cos/sin and Q/K application.
- [ ] Implement noncausal self-attention with bidirectional padding mask.
- [ ] Implement FFN `Linear -> SiLU -> Linear`.
- [ ] Implement Conformer convolution module with GLU, masked fill, depthwise SAME Conv1d, BatchNorm, SiLU, pointwise Conv1d.
- [ ] Implement LASR residual scale/add constants.
- [ ] Implement final LayerNorm and CTC head.
- [ ] Implement greedy argmax and padded-output fill.
- [ ] Add tokenizer-side CTC duplicate and blank removal, or validate token IDs before decode.
- [ ] Add per-op parity tests for feature extraction, subsampling, RoPE, attention, conv module, and CTC head.
- [ ] Add full-checkpoint parity on `hf-internal-testing/lasr-test`.
- [ ] Benchmark feature extraction, encoder, attention, depthwise conv, and end-to-end ASR chunks.
