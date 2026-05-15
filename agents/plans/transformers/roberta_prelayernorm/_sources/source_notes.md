# roberta_prelayernorm source notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `transformers/src/transformers/models/roberta_prelayernorm`

Source files inspected:

- `configuration_roberta_prelayernorm.py`
- `modeling_roberta_prelayernorm.py`
- `convert_roberta_prelayernorm_original_pytorch_checkpoint_to_pytorch.py`
- `docs/source/en/model_doc/roberta-prelayernorm.md`
- `tests/models/roberta_prelayernorm/test_modeling_roberta_prelayernorm.py`

Representative config snapshots fetched from Hugging Face raw `config.json`
URLs on 2026-05-13:

- `andreasmadsen/efficient_mlm_m0.15`
- `andreasmadsen/efficient_mlm_m0.20`
- `andreasmadsen/efficient_mlm_m0.40`
- `andreasmadsen/efficient_mlm_m0.80`
- `andreasmadsen/efficient_mlm_m0.40-801010`
- `ThomasLI/efficient_mlm_m0.40-finetuned-classification`
- `cambridge-climb/baseline-roberta_pre_layer_norm-model`
- `mist-models/mist-28M-ti624ev1`

Config snapshot files in this directory use repository names with `/` replaced
by `__`.
