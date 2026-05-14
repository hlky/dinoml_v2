# LUKE source notes

Audit date: 2026-05-13

Transformers checkout: `X:/H/transformers`
Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source files:

- `X:/H/transformers/src/transformers/models/luke/configuration_luke.py`
- `X:/H/transformers/src/transformers/models/luke/modeling_luke.py`
- `X:/H/transformers/src/transformers/models/luke/tokenization_luke.py`
- `X:/H/transformers/src/transformers/models/luke/convert_luke_original_pytorch_checkpoint_to_pytorch.py`

Representative HF snapshots saved here:

- `studio-ousia__luke-base.config.json`
- `studio-ousia__luke-large.config.json`
- `studio-ousia__luke-large-finetuned-open-entity.config.json`
- `studio-ousia__luke-large-finetuned-tacred.config.json`
- `studio-ousia__luke-large-finetuned-conll-2003.config.json`
- matching `tokenizer_config.json` and `special_tokens_map.json` where available

Config sweep summary:

| Checkpoint | Architecture | Hidden | Layers | Heads | FFN | Entity emb | Entity vocab | Max pos | Type vocab | Labels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `studio-ousia/luke-base` | `LukeForMaskedLM` | 768 | 12 | 12 | 3072 | 256 | 500000 | 514 | 1 | n/a |
| `studio-ousia/luke-large` | `LukeForMaskedLM` | 1024 | 24 | 16 | 4096 | 256 | 500000 | 514 | 1 | n/a |
| `studio-ousia/luke-large-finetuned-open-entity` | `LukeForEntityClassification` | 1024 | 24 | 16 | 4096 | 256 | 500000 | 514 | 1 | 9 |
| `studio-ousia/luke-large-finetuned-tacred` | `LukeForEntityPairClassification` | 1024 | 24 | 16 | 4096 | 256 | 500000 | 514 | 1 | 42 |
| `studio-ousia/luke-large-finetuned-conll-2003` | `LukeForEntitySpanClassification` | 1024 | 24 | 16 | 4096 | 256 | 500000 | 514 | 1 | 5 |

Notable source/config mismatches:

- `LukeConfig` source default `max_position_embeddings` is 512, but sampled checkpoints use 514, matching RoBERTa-style special-token offset capacity.
- `LukeConfig` source default `type_vocab_size` is 2, but sampled checkpoints use 1. Tokenizer usually does not return token type IDs unless requested; model defaults missing token types to zeros.
- Sampled configs contain historical fields such as `output_past`, `use_cache`, `position_embedding_type`, `bert_model_name`, and `classifier_bias`. The current in-library LUKE modeling source does not implement autoregressive cache behavior or read those fields for graph structure. `LukeForEntityPairClassification` hardcodes a no-bias classifier.
- Tokenizer task controls entity packing: `entity_classification` forces max entity length 1, `entity_pair_classification` forces 2, and `entity_span_classification` keeps the configured max entity length and emits start/end word-token indices.
