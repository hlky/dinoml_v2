# BROS Config Sweep

Source basis: representative Hugging Face `config.json` files fetched on 2026-05-13.

## jinho8345/bros-base-uncased

URL: https://huggingface.co/jinho8345/bros-base-uncased/raw/main/config.json

```json
{
  "_name_or_path": "naver-clova-ocr/bros-base-uncased",
  "architectures": ["BrosModel"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "classifier_dropout_prob": 0.1,
  "dim_bbox": 8,
  "dim_bbox_projection": 64,
  "dim_bbox_sinusoid_emb_1d": 24,
  "dim_bbox_sinusoid_emb_2d": 192,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "n_relations": 1,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.34.0.dev0",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

## jinho8345/bros-large-uncased

URL: https://huggingface.co/jinho8345/bros-large-uncased/raw/main/config.json

```json
{
  "_name_or_path": "naver-clova-ocr/bros-large-uncased",
  "architectures": ["BrosModel"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "classifier_dropout_prob": 0.1,
  "dim_bbox": 8,
  "dim_bbox_projection": 64,
  "dim_bbox_sinusoid_emb_1d": 32,
  "dim_bbox_sinusoid_emb_2d": 256,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "n_relations": 1,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.34.0.dev0",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

## naver-clova-ocr/bros-base-uncased

URL: https://huggingface.co/naver-clova-ocr/bros-base-uncased/raw/main/config.json

```json
{
  "architectures": ["BrosModel"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.10.0",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

## naver-clova-ocr/bros-large-uncased

URL: https://huggingface.co/naver-clova-ocr/bros-large-uncased/raw/main/config.json

```json
{
  "architectures": ["BrosModel"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.10.0",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

## adamadam111/bros-funsd-finetuned

URL: https://huggingface.co/adamadam111/bros-funsd-finetuned/raw/main/config.json

```json
{
  "architectures": ["BrosForTokenClassification"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "classifier_dropout_prob": 0.1,
  "dim_bbox": 8,
  "dim_bbox_projection": 64,
  "dim_bbox_sinusoid_emb_1d": 24,
  "dim_bbox_sinusoid_emb_2d": 192,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "O",
    "1": "B-HEADER",
    "2": "I-HEADER",
    "3": "B-QUESTION",
    "4": "I-QUESTION",
    "5": "B-ANSWER",
    "6": "I-ANSWER"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "n_relations": 1,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.53.2",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

## adamadam111/bros-docclass-finetuned

URL: https://huggingface.co/adamadam111/bros-docclass-finetuned/raw/main/config.json

```json
{
  "architectures": ["BrosForDocumentClassification"],
  "attention_probs_dropout_prob": 0.1,
  "bbox_scale": 100.0,
  "classifier_dropout_prob": 0.1,
  "dim_bbox": 8,
  "dim_bbox_projection": 64,
  "dim_bbox_sinusoid_emb_1d": 24,
  "dim_bbox_sinusoid_emb_2d": 192,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "form",
    "1": "invoice",
    "2": "budget",
    "3": "file folder",
    "4": "questionnaire"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bros",
  "n_relations": 1,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pe_type": "crel",
  "torch_dtype": "float32",
  "transformers_version": "4.53.0",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

Note: `BrosForDocumentClassification` is not implemented in the inspected in-library BROS source and should not be treated as supported by this audit.
