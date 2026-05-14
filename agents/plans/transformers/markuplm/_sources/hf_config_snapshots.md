# MarkupLM HF config snapshots

Fetched on 2026-05-13 from public Hugging Face raw URLs.

## `microsoft/markuplm-base`

Source: <https://huggingface.co/microsoft/markuplm-base/raw/main/config.json>

```json
{
  "architectures": ["MarkupLMForPretraining"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_act": "gelu",
  "vocab_size": 50267,
  "max_position_embeddings": 514,
  "type_vocab_size": 1,
  "pad_token_id": 1,
  "layer_norm_eps": 1e-05,
  "max_depth": 50,
  "max_xpath_tag_unit_embeddings": 256,
  "max_xpath_subs_unit_embeddings": 1024,
  "xpath_unit_hidden_size": 32,
  "torch_dtype": "float16",
  "transformers_version": "4.10.2",
  "use_cache": true
}
```

## `microsoft/markuplm-large`

Source: <https://huggingface.co/microsoft/markuplm-large/raw/main/config.json>

```json
{
  "architectures": ["MarkupLMForPretraining"],
  "hidden_size": 1024,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "intermediate_size": 4096,
  "hidden_act": "gelu",
  "vocab_size": 50267,
  "max_position_embeddings": 514,
  "type_vocab_size": 1,
  "pad_token_id": 1,
  "layer_norm_eps": 1e-05,
  "max_depth": 50,
  "max_xpath_tag_unit_embeddings": 256,
  "max_xpath_subs_unit_embeddings": 1024,
  "xpath_unit_hidden_size": 32,
  "torch_dtype": "float16",
  "transformers_version": "4.10.2",
  "use_cache": true
}
```

## `microsoft/markuplm-base-finetuned-websrc`

Source: <https://huggingface.co/microsoft/markuplm-base-finetuned-websrc/raw/main/config.json>

```json
{
  "architectures": ["MarkupLMForQuestionAnswering"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_act": "gelu",
  "vocab_size": 50267,
  "max_position_embeddings": 514,
  "type_vocab_size": 1,
  "pad_token_id": 1,
  "layer_norm_eps": 1e-05,
  "max_depth": 50,
  "max_xpath_tag_unit_embeddings": 256,
  "max_xpath_subs_unit_embeddings": 1024,
  "xpath_unit_hidden_size": 32,
  "torch_dtype": "float32",
  "has_relative_attention_bias": false,
  "has_tree_attention_bias": false,
  "max_tree_id_unit_embeddings": 1024,
  "tree_id_unit_hidden_size": 32,
  "rel_pos_bins": 32,
  "max_rel_pos": 128,
  "tree_rel_pos_bins": 32,
  "tree_max_rel_pos": 128,
  "pos_mode_for_path_emb": 1,
  "use_cache": true
}
```

## Gated/unavailable checks

- <https://huggingface.co/microsoft/markuplm-base-finetuned-squad/raw/main/config.json> returned 401 Unauthorized.
- <https://huggingface.co/microsoft/markuplm-base-finetuned-rico/raw/main/config.json> returned 401 Unauthorized.

Access would resolve task-head label metadata and exact fine-tuned architectures. It should not change the native encoder operator surface unless those repos require remote code, which was not inspectable from the 401 responses.
