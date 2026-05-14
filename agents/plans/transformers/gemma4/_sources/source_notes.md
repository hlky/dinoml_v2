# Gemma4 audit source notes

Local source basis:

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source directory: `X:/H/transformers/src/transformers/models/gemma4`
- Runtime source inspected: `modeling_gemma4.py`
- Future-edit source inspected: `modular_gemma4.py`
- Config/processor source inspected: `configuration_gemma4.py`, `processing_gemma4.py`, `image_processing_gemma4.py`, `video_processing_gemma4.py`, `feature_extraction_gemma4.py`, `convert_gemma4_weights.py`

Downloaded representative Hugging Face snapshots:

- `google_gemma-4-E2B__config.json`
- `google_gemma-4-E2B-it__config.json`
- `google_gemma-4-E4B__config.json`
- `google_gemma-4-E4B-it__config.json`
- `google_gemma-4-26B-A4B__config.json`
- `google_gemma-4-26B-A4B-it__config.json`
- `google_gemma-4-31B__config.json`
- `google_gemma-4-31B-it__config.json`
- `tiny-random_gemma-4-moe__config.json`
- `google_gemma-4-E2B-it__processor_config.json`
- `google_gemma-4-E4B-it__processor_config.json`
- `google_gemma-4-31B-it__processor_config.json`

Attempted but absent as standalone files for `google/gemma-4-E4B-it`:

- `preprocessor_config.json`
- `audio_preprocessor_config.json`
- `video_preprocessor_config.json`

The processor config embeds image, video, and audio feature-extractor configs.

