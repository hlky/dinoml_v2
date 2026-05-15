# XLM-RoBERTa source/config notes

Local Transformers checkout: `transformers` at
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Inspected source files:

- `src/transformers/models/xlm_roberta/configuration_xlm_roberta.py`
- `src/transformers/models/xlm_roberta/modeling_xlm_roberta.py`
- `src/transformers/models/xlm_roberta/modular_xlm_roberta.py`
- `src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py`
- comparison files:
  - `src/transformers/models/roberta/configuration_roberta.py`
  - `src/transformers/models/roberta/modeling_roberta.py`
  - `src/transformers/models/roberta/tokenization_roberta.py`
  - `src/transformers/models/camembert/configuration_camembert.py`
  - `src/transformers/models/camembert/modeling_camembert.py`
  - `src/transformers/models/camembert/tokenization_camembert.py`

Fetched Hugging Face snapshots:

- `FacebookAI/xlm-roberta-base`: `config.json`, `tokenizer_config.json`.
- `FacebookAI/xlm-roberta-large`: `config.json`, `tokenizer_config.json`.
- `cardiffnlp/twitter-xlm-roberta-base`: `config.json`.
- `joeddav/xlm-roberta-large-xnli`: `config.json`, `tokenizer_config.json`,
  `special_tokens_map.json`.
- `Davlan/xlm-roberta-base-ner-hrl`: `config.json`,
  `tokenizer_config.json`, `special_tokens_map.json`.

Missing/gated fetches:

- `FacebookAI/xlm-roberta-xl` raw `config.json`, `tokenizer_config.json`, and
  `special_tokens_map.json` returned `401 Unauthorized` on 2026-05-13.
  Model page: <https://huggingface.co/FacebookAI/xlm-roberta-xl>.
- `FacebookAI/xlm-roberta-xxl` raw `config.json`, `tokenizer_config.json`,
  and `special_tokens_map.json` returned `401 Unauthorized` on 2026-05-13.
  Model page: <https://huggingface.co/FacebookAI/xlm-roberta-xxl>.
- `special_tokens_map.json` was absent (`404`) for
  `FacebookAI/xlm-roberta-base` and `FacebookAI/xlm-roberta-large`; tokenizer
  special-token defaults come from source and the fetched tokenizer config only
  states `model_max_length`.

No DinoML tests were run for this audit.
