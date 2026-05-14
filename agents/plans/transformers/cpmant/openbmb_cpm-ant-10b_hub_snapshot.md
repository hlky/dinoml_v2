# openbmb/cpm-ant-10b Hub Snapshot

- Repo: https://huggingface.co/openbmb/cpm-ant-10b
- Repo SHA from Hugging Face Hub API: `3b53c0f95f625de6ae676f9edf2cc65930f8b4b8`
- Access: public, not gated according to `HfApi.model_info`
- Task/library tags from Hub API: `text-generation`, `transformers`, `pytorch`, `cpmant`, `zh`
- Files checked: `config.json`, `generation_config.json`, `special_tokens_map.json`, `pytorch_model.bin.index.json`
- Missing file: `tokenizer_config.json` returned 404
- Weight index metadata: `total_size = 38069813248` bytes, 436 logical tensor entries, 13 PyTorch shards
- Weight-map top-level tensors include `input_embedding.weight`, `segment_embedding.weight`, `position_bias.relative_attention_bias`, `encoder.output_layernorm.weight`, and repeated `encoder.layers.*` attention/FFN tensors.

Representative config sweep note: Hub search for `cpm-ant` / `cpmant` found only this native cpmant checkpoint; no small/debug or alternate operator-structure variants were found under the native Transformers family during this audit.
