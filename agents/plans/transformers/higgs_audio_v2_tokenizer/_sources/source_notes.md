# Source notes: higgs_audio_v2_tokenizer

## Local source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `src/transformers/models/higgs_audio_v2_tokenizer`
- Primary generated source:
  - `configuration_higgs_audio_v2_tokenizer.py`
  - `modeling_higgs_audio_v2_tokenizer.py`
- Modular source: `modular_higgs_audio_v2_tokenizer.py`; generated files state they are derived from this modular file.
- Conversion/source ABI clues: `convert_higgs_audio_v2_tokenizer_to_hf.py`
- Delegated source inspected:
  - `src/transformers/models/xcodec/configuration_xcodec.py`
  - `src/transformers/models/xcodec/modeling_xcodec.py`
  - `src/transformers/models/dac/configuration_dac.py`
  - `src/transformers/models/dac/modeling_dac.py`
  - `src/transformers/models/dac/feature_extraction_dac.py`
  - `src/transformers/models/hubert/configuration_hubert.py`
  - `src/transformers/models/hubert/modeling_hubert.py`

## HF config snapshots

- Current official checkpoint:
  - Model URL: <https://huggingface.co/bosonai/higgs-audio-v2-tokenizer>
  - Config URL: <https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/config.json>
  - Preprocessor URL: <https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/main/preprocessor_config.json>
  - Hub API `sha`: `7c56ba8cc3fcbb6db9c866afcbb68f97405e5ba2`
  - Public, not gated, license metadata `other`.
- Older raw config revision:
  - <https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/2d3c8d2b8ede96989a66624e8da8043750b9cf05/config.json>
  - This is a legacy non-Transformers-style config with fields such as `n_filters`, `D`, `ratios`, `bins`, `n_q`, and `semantic_techer`.
- Current config revision also observed at:
  - <https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/raw/d2e09d95f3a0b80146425a1af09eab128eab6f4e/config.json>
  - Same operator-significant contents as current `main` for this audit.
- Docs example id gap:
  - The example id `hf-audio/higgs_audio_v2_tokenizer-hubert-librispeech` returned an authentication/invalid-access error when fetching `config.json`.

## Source-derived ABI facts

- Main input: `input_values`, audio tensor shaped `[batch, channels, num_samples]`.
- Runtime channel guard: encode rejects channels other than 1.
- Feature extractor:
  - `DacFeatureExtractor`
  - `feature_size=1`, `sampling_rate=24000`, `hop_length=960`, right padding with `padding_value=0.0`.
  - Pads to a multiple of `hop_length` and emits `padding_mask` when padding is enabled.
- Semantic branch:
  - Resamples model input from 24000 Hz to 16000 Hz when sample rates differ.
  - Selects first channel, pads fixed `(160, 160)` samples, runs HuBERT with `output_hidden_states=True`.
  - Stacks all hidden states along a layer axis and averages across layers.
  - Optional semantic downsample by `semantic_downsample_factor`.
- Acoustic branch:
  - Uses a DAC encoder/decoder submodule from `AutoModel.from_config(config.acoustic_model_config)`.
  - Decoder ConvTranspose1d output padding is mutated to `stride % 2`.
  - Final DAC decoder tanh is replaced with identity.
- Tokenizer packing:
  - `encode` returns `[batch, num_quantizers_for_bandwidth, codes_length]`.
  - Internally RVQ stacks as `[num_quantizers, batch, codes_length]`, then transposes to public batch-major packing.
  - `decode` expects public batch-major codes, transposes to quantizer-major, sums decoded residual quantizer embeddings, projects to the acoustic hidden size, and runs the acoustic decoder.
- Official current config:
  - `sample_rate=24000`, acoustic `downsampling_ratios=[8,5,4,2,3]`, `hop_length=960`, `frame_rate=25`.
  - `codebook_size=1024`, `codebook_dim=64`, `codebook_nbits=10`.
  - `target_bandwidths=[0.5,1,1.5,2]`; effective max quantizers is `floor(1000*2/(25*10)) = 8`.
  - Semantic HuBERT: 12 layers, hidden 768, 12 attention heads, MLP 3072, feature conv strides `[5,2,2,2,2,2,2]`, kernels `[10,3,3,3,3,2,2]`.
