# LightOn OCR audit source snapshots

Fetched on 2026-05-13 for the `lighton_ocr` Transformers family audit.

Local source basis:

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source: `X:/H/transformers/src/transformers/models/lighton_ocr`
- Composed source read for delegated operators:
  - `X:/H/transformers/src/transformers/models/pixtral`
  - `X:/H/transformers/src/transformers/models/qwen3`

HF snapshots saved here:

- `lightonai__LightOnOCR-1B-1025`
- `lightonai__LightOnOCR-0.9B-16k-1025`
- `lightonai__LightOnOCR-0.9B-32k-1025`
- `lightonai__LightOnOCR-2-1B-bbox-soup`

Files attempted per repo:

- `api_model.json`
- `config.json`
- `preprocessor_config.json`
- `processor_config.json`
- `generation_config.json`
- `tokenizer_config.json`
- `chat_template.json`

Some repos do not expose all files at `resolve/main`; missing fetches are kept as
`*.fetch_error.txt` so future agents can distinguish unavailable files from
unread files.
