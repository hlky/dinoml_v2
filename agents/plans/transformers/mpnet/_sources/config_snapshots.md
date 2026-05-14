# MPNet Source Snapshots

Audit date: 2026-05-13

## Local Transformers

```text
Checkout: X:/H/transformers
Commit: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Family dir: X:/H/transformers/src/transformers/models/mpnet
Files:
  configuration_mpnet.py
  modeling_mpnet.py
  tokenization_mpnet.py
  __init__.py
Shared source:
  X:/H/transformers/src/transformers/modeling_utils.py
```

## Hugging Face configs

### microsoft/mpnet-base

Repo metadata:

```text
sha: 6996ce1e91bd2a9c7d7f61daec37463394f73f09
library_name: transformers
pipeline_tag: fill-mask
private: false
gated: false
```

Config from <https://huggingface.co/microsoft/mpnet-base/raw/main/config.json>:

```json
{
  "architectures": ["MPNetForMaskedLM"],
  "attention_probs_dropout_prob": 0.1,
  "pad_token_id": 1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "relative_attention_num_buckets": 32,
  "model_type": "mpnet",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "vocab_size": 30527
}
```

Tokenizer config from <https://huggingface.co/microsoft/mpnet-base/raw/main/tokenizer_config.json>:

```json
{
  "model_max_length": 512,
  "do_lower_case": true
}
```

### sentence-transformers/all-mpnet-base-v2

Repo metadata:

```text
sha: e8c3b32edf5434bc2275fc9bab85f82640a19130
library_name: sentence-transformers
pipeline_tag: sentence-similarity
private: false
gated: false
```

Transformer config from <https://huggingface.co/sentence-transformers/all-mpnet-base-v2/raw/main/config.json>:

```json
{
  "_name_or_path": "microsoft/mpnet-base",
  "architectures": ["MPNetForMaskedLM"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "mpnet",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "relative_attention_num_buckets": 32,
  "transformers_version": "4.8.2",
  "vocab_size": 30527
}
```

Sentence-transformers pooling from `1_Pooling/config.json`:

```json
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
```

Modules:

```json
[
  {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
  {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
  {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
]
```

Sentence-BERT wrapper config:

```json
{
  "max_seq_length": 384,
  "do_lower_case": false
}
```

### sentence-transformers/paraphrase-mpnet-base-v2

Repo metadata:

```text
sha: 6cc9279c672dc57f94445ef259b28a1b736fec8f
library_name: sentence-transformers
pipeline_tag: sentence-similarity
private: false
gated: false
```

Transformer config from <https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2/raw/main/config.json>:

```json
{
  "_name_or_path": "old_models/paraphrase-mpnet-base-v2/0_Transformer",
  "architectures": ["MPNetModel"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "mpnet",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "relative_attention_num_buckets": 32,
  "transformers_version": "4.7.0",
  "vocab_size": 30527
}
```

Sentence-transformers pooling:

```json
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": true,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
```

Modules:

```json
[
  {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
  {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}
]
```

## Access notes

`https://huggingface.co/microsoft/mpnet-base-masked/raw/main/config.json` returned an authentication-style error during the audit. It was not used for dimensions because public `microsoft/mpnet-base` and sentence-transformers MPNet configs were accessible.
