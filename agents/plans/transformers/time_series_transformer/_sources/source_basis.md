# Source Basis Snapshot

Transformers checkout:

```text
transformers
commit b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Primary files inspected:

```text
src/transformers/models/time_series_transformer/configuration_time_series_transformer.py
src/transformers/models/time_series_transformer/modeling_time_series_transformer.py
src/transformers/time_series_utils.py
tests/models/time_series_transformer/test_modeling_time_series_transformer.py
```

Key source line anchors:

```text
configuration_time_series_transformer.py:76-147
  model_type, config fields, feature_size derivation.

modeling_time_series_transformer.py:82-195
  std/mean/no-op scalers.

modeling_time_series_transformer.py:303-417
  attention projection, cache update, backend dispatch.

modeling_time_series_transformer.py:421-552
  encoder and decoder layer order.

modeling_time_series_transformer.py:575-775
  encoder/decoder embedding, masks, cache initialization.

modeling_time_series_transformer.py:782-918
  lag extraction and network input construction.

modeling_time_series_transformer.py:923-1110
  base model forward split into encoder context and decoder horizon.

modeling_time_series_transformer.py:1114-1525
  prediction head, distribution selection, loss, sampling loop.

time_series_utils.py:63-225
  distribution parameter projection and domain maps.

test_modeling_time_series_transformer.py:50-132
  synthetic test ABI.

test_modeling_time_series_transformer.py:384-455
  lag/network-input behavior tests.

test_modeling_time_series_transformer.py:475-543
  official tourism integration shape/generation tests.
```

Canonical source URLs:

```text
https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/time_series_transformer/configuration_time_series_transformer.py
https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/time_series_transformer/modeling_time_series_transformer.py
https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/time_series_utils.py
https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/tests/models/time_series_transformer/test_modeling_time_series_transformer.py
```
