# Voxtral Source Notes

Audit date: 2026-05-13

Transformers source checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `src/transformers/models/voxtral`

Primary local source files inspected:

- `src/transformers/models/voxtral/configuration_voxtral.py`
- `src/transformers/models/voxtral/modular_voxtral.py`
- `src/transformers/models/voxtral/modeling_voxtral.py`
- `src/transformers/models/voxtral/processing_voxtral.py`
- `src/transformers/models/voxtral/convert_voxtral_weights_to_hf.py`
- Neighbor/composed files:
  - `src/transformers/models/qwen2_audio/modeling_qwen2_audio.py`
  - `src/transformers/models/llama/modeling_llama.py`
  - `src/transformers/models/whisper/feature_extraction_whisper.py`
  - `src/transformers/masking_utils.py`
  - `src/transformers/modeling_utils.py`
  - `src/transformers/integrations/sdpa_attention.py`
  - `src/transformers/integrations/flash_attention.py`
  - `src/transformers/integrations/flex_attention.py`

Authoritative source note:

- `modeling_voxtral.py` is generated from `modular_voxtral.py`; future source edits should target `modular_voxtral.py`.
- The generated file was still inspected because it is the concrete runtime source in the pinned checkout.

HF configs / metadata fetched:

- `mistralai/Voxtral-Mini-3B-2507`
  - `https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/raw/main/config.json`
  - `https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/raw/main/preprocessor_config.json`
  - `https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/raw/main/generation_config.json`
- `mistralai/Voxtral-Small-24B-2507`
  - `https://huggingface.co/mistralai/Voxtral-Small-24B-2507/raw/main/config.json`
- `tiny-random/voxtral`
  - `https://huggingface.co/tiny-random/voxtral/raw/main/config.json`
- `yujiepan/voxtral-tiny-random`
  - `https://huggingface.co/yujiepan/voxtral-tiny-random/raw/main/config.json`
- `MohamedRashad/Voxtral-Mini-3B-2507-transformers`
  - `https://huggingface.co/MohamedRashad/Voxtral-Mini-3B-2507-transformers/raw/main/config.json`
- Quantized/open mirror configs sampled:
  - `https://huggingface.co/VincentGOURBIN/voxtral-small-8bit/raw/main/config.json`
  - `https://huggingface.co/mzbac/voxtral-mini-3b-4bit-mixed/raw/main/config.json`

Observed config facts:

- Official Mini and Small share the same audio tower:
  - `num_mel_bins=128`
  - `hidden_size=1280`
  - `intermediate_size=5120`
  - `num_hidden_layers=32`
  - `num_attention_heads=20`
  - `head_dim=64`
  - `max_source_positions=1500`
  - `activation_function=gelu`
- Official Mini text decoder:
  - `hidden_size=3072`
  - `intermediate_size=8192`
  - `num_hidden_layers=30`
  - `num_attention_heads=32`
  - `num_key_value_heads=8`
  - `head_dim=128`
  - `rope_theta=100000000.0`
  - `max_position_embeddings=131072`
  - `hidden_act=silu`
- Official Small text decoder:
  - `hidden_size=5120`
  - `intermediate_size=32768`
  - `num_hidden_layers=40`
  - `num_attention_heads=32`
  - `num_key_value_heads=8`
  - `head_dim=128`
  - `rope_theta=100000000.0`
  - `max_position_embeddings=131072`
  - `hidden_act=silu`
- Processor / feature extractor:
  - `WhisperFeatureExtractor`
  - `sampling_rate=16000`
  - `feature_size=128`
  - `n_fft=400`
  - `hop_length=160`
  - `chunk_length=30`
  - `n_samples=480000`
  - `nb_max_frames=3000`
  - `return_attention_mask=false`
  - Voxtral processor defaults `pad_to_multiple_of=480000`, `truncation=false`, `max_source_positions=3000`.

Important gaps / caveats:

- `processor_config.json` was not present for `mistralai/Voxtral-Mini-3B-2507`; processor behavior came from source plus `preprocessor_config.json`.
- The multimodal projector intentionally consumes `audio_config.intermediate_size=5120`: `get_audio_features` reshapes audio tower output `[chunks, 1500, 1280]` to `[-1, 5120]`, grouping four consecutive encoder positions into one projected audio token. One 30-second chunk therefore emits 375 decoder-placeholder embeddings.
- `mistralai/Voxtral-Mini-4B-Realtime-2602` and `mistralai/Voxtral-4B-TTS-2603` appear in HF search metadata but are not native `model_type=voxtral` targets for this report. They likely require separate audits under their own source/model types.
- Quantized mirror configs add large `quantization` maps that are not read by native `modeling_voxtral.py`; treat them as external loading/provider metadata unless DinoML explicitly targets those mirrors.
