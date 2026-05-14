# DAC source notes

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model directory: `X:/H/transformers/src/transformers/models/dac`
- Files inspected:
  - `configuration_dac.py`
  - `feature_extraction_dac.py`
  - `modeling_dac.py`
  - `convert_dac_checkpoint.py`
  - `__init__.py`

## Source anchors

- `configuration_dac.py`: `DacConfig` declares `encoder_hidden_size`, `downsampling_ratios`, `decoder_hidden_size`, `n_codebooks`, `codebook_size`, `codebook_dim`, `sampling_rate`; `__post_init__` computes `upsampling_ratios`, `hidden_size`, and `hop_length`; `frame_rate` is `ceil(sampling_rate / prod(upsampling_ratios))`.
- `feature_extraction_dac.py`: `DacFeatureExtractor` accepts mono raw audio, validates sampling rate, pads/truncates via `SequenceFeatureExtractor.pad`, pads to `hop_length`, emits `padding_mask` when padding, and returns `input_values` shaped as batch, channel, time for tensor outputs.
- `modeling_dac.py`: `Snake1d` computes `x + reciprocal(alpha + 1e-9) * sin(alpha * x)^2` with learnable `[1,C,1]` alpha.
- `modeling_dac.py`: `DacVectorQuantize` uses `Conv1d(hidden_size -> codebook_dim, k=1)`, L2 normalization, codebook nearest-neighbor by maximum cosine/euclidean-score expression, embedding lookup, straight-through estimator in training, then `Conv1d(codebook_dim -> hidden_size, k=1)`.
- `modeling_dac.py`: `DacResidualUnit` is `Snake -> dilated Conv1d(k=7,pad=((7-1)*d)//2) -> Snake -> Conv1d(k=1) -> residual add`, with center crop if the convolution changed length.
- `modeling_dac.py`: `DacEncoderBlock` chains residual units with dilation 1, 3, 9, then `Snake` and strided `Conv1d(k=2*stride, stride=stride, padding=ceil(stride/2))`.
- `modeling_dac.py`: `DacDecoderBlock` performs `Snake -> ConvTranspose1d(k=2*stride, stride=stride, padding=ceil(stride/2)) -> residual units dilation 1, 3, 9`.
- `modeling_dac.py`: `DacResidualVectorQuantizer.forward` applies residual quantizers sequentially, accumulates quantized output, subtracts each quantized contribution from the residual, stacks `audio_codes` as `[B, num_codebooks_used, Tq]`, and concatenates projected latents over channel.
- `modeling_dac.py`: `from_codes(audio_codes)` is the decode-side ABI: per-codebook embedding lookup `audio_codes[:, i, :]`, transpose to `[B,D,Tq]`, per-codebook `out_proj`, and sum over codebooks.
- `modeling_dac.py`: `DacModel.forward` saves input length, runs encode, decodes quantized representation, and slices waveform output to `[..., :length]`.
- `convert_dac_checkpoint.py`: converted Descript checkpoints are allowlisted as `dac_16khz`, `dac_24khz`, and `dac_44khz`; conversion applies then removes weight norm before saving HF weights.

## Hugging Face config sources

- [descript/dac_16khz config.json](https://huggingface.co/descript/dac_16khz/blob/main/config.json)
- [descript/dac_16khz preprocessor_config.json](https://huggingface.co/descript/dac_16khz/blob/main/preprocessor_config.json)
- [descript/dac_24khz config.json](https://huggingface.co/descript/dac_24khz/blob/main/config.json)
- [descript/dac_24khz preprocessor_config.json](https://huggingface.co/descript/dac_24khz/blob/main/preprocessor_config.json)
- [descript/dac_44khz config.json](https://huggingface.co/descript/dac_44khz/blob/main/config.json)
- [descript/dac_44khz preprocessor_config.json](https://huggingface.co/descript/dac_44khz/blob/main/preprocessor_config.json)

## Source gaps and gated areas

- No gated official DAC configs were encountered for `descript/dac_16khz`, `descript/dac_24khz`, or `descript/dac_44khz`.
- Model cards are sparse and auto-generated; this audit relies on source and config JSON rather than card prose.
- The Hub `config.json` for 16 kHz and 24 kHz includes `hop_length: 512`, while the downsampling product is 320 and the preprocessor config uses `hop_length: 320`. Under the inspected source, `DacConfig.__post_init__` computes effective `hop_length` from `downsampling_ratios`; DinoML should prefer the source-derived product and preprocessor value over the stale config field.
- The source implements training losses and quantizer dropout, but this report scopes DinoML to inference encode/decode and marks training-only paths as deferred.
