# speech_encoder_decoder source snapshots

Fetched on 2026-05-13 from Hugging Face `resolve/main` endpoints for representative configs and processor/tokenizer metadata. These are raw snapshot inputs for `../report.md`; model weights were not downloaded.

Model ids:

- `hf-internal-testing/tiny-random-speech-encoder-decoder`
- `facebook/wav2vec2-xls-r-300m-en-to-15`
- `facebook/wav2vec2-xls-r-1b-en-to-15`
- `facebook/wav2vec2-xls-r-2b-en-to-15`
- `facebook/s2t-wav2vec2-large-en-de`
- `patrickvonplaten/wav2vec2-2-bart-base`
- `KBLab/asr-voxrex-bart-base`

Files attempted per model:

- `config.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `generation_config.json`
- `special_tokens_map.json`

Missing `generation_config.json` files were HTTP 404 for:

- `hf-internal-testing/tiny-random-speech-encoder-decoder`
- `patrickvonplaten/wav2vec2-2-bart-base`
- `KBLab/asr-voxrex-bart-base`

