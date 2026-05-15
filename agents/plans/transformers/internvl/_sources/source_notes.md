# InternVL source notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source files inspected:

- `src/transformers/models/internvl/configuration_internvl.py`
- `src/transformers/models/internvl/modeling_internvl.py`
- `src/transformers/models/internvl/modular_internvl.py`
- `src/transformers/models/internvl/processing_internvl.py`
- `src/transformers/models/internvl/video_processing_internvl.py`
- `src/transformers/models/got_ocr2/image_processing_got_ocr2.py`
- `src/transformers/models/qwen2/modeling_qwen2.py`
- `src/transformers/models/qwen2/configuration_qwen2.py`

Representative official HF artifacts fetched on 2026-05-13:

- `OpenGVLab/InternVL3-1B-hf`: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `OpenGVLab/InternVL3-2B-hf`: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `OpenGVLab/InternVL3-8B-hf`: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `OpenGVLab/InternVL3-14B-hf`: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `OpenGVLab/InternVL3-38B-hf`: `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `generation_config.json`

No gated or 401/403 configs were encountered for the inspected official `-hf` checkpoints.
