# Moonshine Source Notes

Report target: `moonshine` only.

## Pinned Transformers source

- Local checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Commit snapshot: `transformers_commit.txt`
- Authoritative source file: `src/transformers/models/moonshine/modular_moonshine.py`
- Generated source file used for concrete line inspection: `src/transformers/models/moonshine/modeling_moonshine.py`
- Config source: `src/transformers/models/moonshine/configuration_moonshine.py`, generated from the same modular file.

Source URLs:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/modular_moonshine.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/modeling_moonshine.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/moonshine/configuration_moonshine.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/masking_utils.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py

## Representative HF snapshots

Fetched from Hugging Face `main` on 2026-05-13:

- `UsefulSensors__moonshine-tiny__config.json`
- `UsefulSensors__moonshine-tiny__preprocessor_config.json`
- `UsefulSensors__moonshine-tiny__generation_config.json`
- `UsefulSensors__moonshine-base__config.json`
- `UsefulSensors__moonshine-base__preprocessor_config.json`
- `UsefulSensors__moonshine-base__generation_config.json`
- `UsefulSensors__moonshine-tiny-ar__config.json`
- `UsefulSensors__moonshine-tiny-ar__preprocessor_config.json`
- `UsefulSensors__moonshine-tiny-ar__generation_config.json`
- `UsefulSensors__moonshine-base-ar__config.json`
- `UsefulSensors__moonshine-base-ar__preprocessor_config.json`
- `UsefulSensors__moonshine-base-ar__generation_config.json`

HF repo URLs:

- https://huggingface.co/UsefulSensors/moonshine-tiny
- https://huggingface.co/UsefulSensors/moonshine-base
- https://huggingface.co/UsefulSensors/moonshine-tiny-ar
- https://huggingface.co/UsefulSensors/moonshine-base-ar

## Scope notes

- `moonshine_streaming` is a separate Transformers model family with separate config/modeling/processor files and is out of scope for this report.
- Non-streaming Moonshine has no family-local processor file. Representative repos use `Wav2Vec2FeatureExtractor` plus a `tokenizer.json`.
- No DinoML imports, tests, model execution, or commits were run for this report.
