# XCodec Source Notes

Audit target: Transformers `src/transformers/models/xcodec` at commit
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Local source basis:

- `transformers/src/transformers/models/xcodec/configuration_xcodec.py`
- `transformers/src/transformers/models/xcodec/modeling_xcodec.py`
- `transformers/src/transformers/models/xcodec/convert_xcodec_weights_to_hf.py`
- `transformers/src/transformers/models/dac/configuration_dac.py`
- `transformers/src/transformers/models/dac/modeling_dac.py`
- `transformers/src/transformers/models/dac/feature_extraction_dac.py`
- Semantic backbones are delegated to existing in-library HuBERT or WavLM sources:
  `src/transformers/models/hubert/modeling_hubert.py` and
  `src/transformers/models/wavlm/modeling_wavlm.py`.

Representative Hugging Face configs fetched from official model repos:

- `hf-audio/xcodec-hubert-librispeech`
- `hf-audio/xcodec-hubert-general`
- `hf-audio/xcodec-hubert-general-balanced`
- `hf-audio/xcodec-wavlm-mls`
- `hf-audio/xcodec-wavlm-more-data`

All five repos expose `config.json`, `model.safetensors`, and
`preprocessor_config.json`. All five preprocessor configs use
`DacFeatureExtractor`, mono `feature_size=1`, `sampling_rate=16000`,
`hop_length=320`, right padding, and `padding_value=0.0`.

Key source anchors:

- `XcodecConfig` computes `hop_length = prod(acoustic_model_config.downsampling_ratios)`,
  `frame_rate = ceil(sample_rate / hop_length)`, `hidden_size =
  acoustic_hidden_size + semantic_hidden_size`, and `num_quantizers =
  int(1000 * max_bandwidth // (frame_rate * ceil(log2(codebook_size))))`.
- `XcodecModel` composes:
  - `AutoModel.from_config(acoustic_model_config).encoder` and `.decoder`,
    currently DAC for the representative configs.
  - `SemanticEncoder` and `SemanticDecoder`, local Conv1d/ConvTranspose1d
    adapters around semantic features.
  - `AutoModel.from_config(semantic_model_config).eval()`, currently HuBERT or
    WavLM for the representative configs.
  - `fc`, `fc1`, `fc2`, and `XcodecResidualVectorQuantization`.
- `encode(input_values, bandwidth)` requires mono `[B, 1, T]`, validates
  `bandwidth in target_bandwidths`, pads semantic input by `hop_length // 2`,
  conditionally pads the DAC acoustic path by the same amount if needed, then
  concatenates acoustic and semantic branches along channel dim before RVQ.
- `decode(audio_codes)` expects `[B, Q, L]`, transposes to quantizer-major
  `[Q, B, L]`, decodes and sums quantizer embeddings, projects to DAC hidden
  channels, then runs the acoustic decoder.
- `forward(input_values, audio_codes=None, bandwidth=None)` calls encode unless
  codes are supplied, decodes, and crops reconstructed audio to the original
  input sample length.

Notable ABI facts:

- Audio code tensor layout exposed by the public API is `[batch, num_quantizers,
  codes_length]` with integer code indices.
- Internal RVQ iteration uses `[num_quantizers, batch, codes_length]`.
- Codebooks are buffers named under `quantizer.quantizers.*.codebook.embed`
  with shape `[codebook_size, codebook_dim]`; representative configs use
  `1024 x 1024`.
- Bandwidth controls active quantizer count. With representative configs,
  `sample_rate=16000`, `hop_length=320`, `frame_rate=50`, `codebook_size=1024`,
  and max `target_bandwidth=4`, so `num_quantizers=8`. Active counts are
  2/4/6/8/8 for requested bandwidths 0.5/1/1.5/2/4 kbps.
- The source validates that requested bandwidth is exactly one of
  `target_bandwidths`, but `get_num_quantizers_for_bandwidth` can map 2 kbps
  and 4 kbps to the same eight active quantizers with these dimensions.
- The acoustic decoder is adjusted after DAC construction: every
  `ConvTranspose1d.output_padding` becomes `stride % 2`, and the final DAC
  `Tanh` is replaced by `Identity`.

Gaps and caveats:

- No code tests or imports were run for this audit.
- HF configs were fetched from public `hf-audio/*` repos only. I did not fetch
  safetensors metadata or original `ZhenYe234/xcodec` YAML/checkpoint files.
- The XCodec family depends on separately audited HuBERT/WavLM and DAC behavior
  for full encode/decode parity. This report treats those as composed
  backbones and calls out the exact contracts XCodec consumes.
