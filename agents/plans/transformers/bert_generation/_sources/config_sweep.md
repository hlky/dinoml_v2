# bert_generation config sweep

Source basis:
- Transformers checkout: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- HF API search `filter=bert-generation` returned one official Google checkpoint plus small/community derivatives.
- Raw configs were fetched from Hugging Face `raw/main/config.json` URLs on 2026-05-13.

## google/bert_for_seq_generation_L-24_bbc_encoder

URL: https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder/raw/main/config.json

```json
{
  "architectures": ["BertForSeqGenerationEncoderModel"],
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert-generation",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "vocab_size": 50358
}
```

Notes:
- Official checkpoint config omits current source defaults for `bos_token_id=2`, `eos_token_id=1`, `use_cache=True`, `is_decoder=False`, `add_cross_attention=False`, and `tie_word_embeddings=True`.
- `architectures` names an older class, but current in-library source exposes `BertGenerationEncoder` and `BertGenerationDecoder`.

## ybelkada/random-tiny-BertGenerationModel

URL: https://huggingface.co/ybelkada/random-tiny-BertGenerationModel/raw/main/config.json

```json
{
  "architectures": ["BertGenerationEncoder"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "eos_token_id": 1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 36,
  "initializer_range": 0.02,
  "intermediate_size": 62,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert-generation",
  "num_attention_heads": 6,
  "num_hidden_layers": 6,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.25.0.dev0",
  "use_cache": true,
  "vocab_size": 1024
}
```

Notes:
- Small/debug shape with head_dim 6.
- Current source does not read `position_embedding_type`; absolute learned positions are always used.

## Zlovoblachko/testik_L1_sent_generator

URL: https://huggingface.co/Zlovoblachko/testik_L1_sent_generator/raw/main/config.json

```json
{
  "_name_or_path": "google/bert_for_seq_generation_L-24_bbc_encoder",
  "architectures": ["BertGenerationDecoder"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "directionality": "bidi",
  "eos_token_id": 1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert-generation",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.38.2",
  "use_cache": true,
  "vocab_size": 50358
}
```

Notes:
- Decoder architecture is requested, but `is_decoder` is omitted; current source only enables causal masks/cache semantics when `config.is_decoder=True`.
- Admission should reject or normalize this config unless the wrapper/loader explicitly mutates `is_decoder=True`.

## ammonbro/bert_sp_updown

URL: https://huggingface.co/ammonbro/bert_sp_updown/raw/main/config.json

```json
{
  "_name_or_path": "google/bert_for_seq_generation_L-24_bbc_encoder",
  "architectures": ["BertGenerationDecoder"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "directionality": "bidi",
  "eos_token_id": 1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert-generation",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 50358
}
```

Notes:
- Same structural trap as `Zlovoblachko/testik_L1_sent_generator`: decoder architecture but omitted `is_decoder`.

## YijunYang280/GuardT2I

URL: https://huggingface.co/YijunYang280/GuardT2I/raw/main/config.json

```json
{
  "_name_or_path": "/bian_data/google_bert_for_seq_generation/bert_for_seq_generation_L-24_bbc_encoder",
  "add_cross_attention": true,
  "architectures": ["BertGenerationDecoder"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "directionality": "bidi",
  "eos_token_id": 1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "is_decoder": true,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert-generation",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "return_scores": true,
  "torch_dtype": "float32",
  "transformers_version": "4.25.1",
  "use_cache": true,
  "vocab_size": 50358
}
```

Notes:
- Implements the source-supported seq2seq decoder mode: `is_decoder=true` and `add_cross_attention=true`.
- Current source does not read `return_scores`.
