# Hub repo snapshot

Fetched with the Hugging Face connector and raw Hub URLs on 2026-05-13.

| Repo | Gated | Hub sha | Library | Pipeline | Notes |
| --- | --- | --- | --- | --- | --- |
| `jinaai/jina-embeddings-v3-hf` | false | `d18862d9a48706220815554fac3ebb4dfa46fc28` | `transformers` | `feature-extraction` | Native `model_type: jina_embeddings_v3`; includes base `model.safetensors` plus task LoRA adapter directories. |
| `jinaai/jina-embeddings-v3` | false | `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6` | `transformers` | `feature-extraction` | Original custom-code repo with `auto_map` to `jinaai/xlm-roberta-flash-implementation`; not native `jina_embeddings_v3`. |
| `jinaai/jina-embeddings-v3-small-ci` | false | not captured from API | `transformers` | `feature-extraction` | Small custom-code CI checkpoint; useful for shape variation only, not native-source parity. |

Representative files saved beside the report:

- `config_jina-embeddings-v3-hf.json`
- `config_jina-embeddings-v3_remote.json`
- `config_jina-embeddings-v3-small-ci.json`
- `tokenizer_config_jina-embeddings-v3-hf.json`
- `modules_jina-embeddings-v3-hf.json`
- `sentence_transformers_pooling_config.json`
- `config_sentence_transformers.json`
- `adapter_config_retrieval_query.json`
- `adapter_config_retrieval_passage.json`
- `adapter_config_classification.json`
- `custom_st_jina-embeddings-v3-hf.py`
