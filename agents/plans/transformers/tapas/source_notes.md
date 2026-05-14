# TAPAS Source Notes

Source checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Primary files inspected:

- `src/transformers/models/tapas/configuration_tapas.py`
- `src/transformers/models/tapas/modeling_tapas.py`
- `src/transformers/models/tapas/tokenization_tapas.py`
- `src/transformers/models/tapas/convert_tapas_original_tf_checkpoint_to_pytorch.py`
- `docs/source/en/model_doc/tapas.md`
- `tests/models/tapas/test_modeling_tapas.py`
- `tests/models/tapas/test_tokenization_tapas.py`

Representative Hub configs fetched on 2026-05-13 and saved in
`hf_config_snapshot.json`:

- `google/tapas-small`
- `google/tapas-base`
- `google/tapas-large`
- `google/tapas-base-finetuned-sqa`
- `google/tapas-base-finetuned-wtq`
- `google/tapas-base-finetuned-wikisql-supervised`
- `google/tapas-base-finetuned-tabfact`
- `google/tapas-small-finetuned-sqa`
- `google/tapas-small-finetuned-sqa`, revision `no_reset`

No representative config request returned 401/403. Weight tensors were not
downloaded; parameter layout notes below are source-derived from module
definitions and conversion mappings.
