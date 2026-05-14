# RemBERT source notes

Local Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `X:/H/transformers/src/transformers/models/rembert`

Files inspected:

- `configuration_rembert.py`
- `modeling_rembert.py`
- `tokenization_rembert.py`
- `__init__.py`

Source facts:

- `RemBertConfig` defaults to `vocab_size=250300`, `input_embedding_size=256`, `hidden_size=1152`, `output_embedding_size=1664`, `num_hidden_layers=32`, `num_attention_heads=18`, `intermediate_size=4608`, `max_position_embeddings=512`, `type_vocab_size=2`, `hidden_act="gelu"`, dropout probabilities `0.0`, and `tie_word_embeddings=False`.
- Embeddings are word/position/token-type tables at `input_embedding_size`, followed by LayerNorm and dropout.
- Encoder starts with `embedding_hidden_mapping_in = Linear(input_embedding_size -> hidden_size)`.
- Each layer uses post-norm BERT-style MHA and MLP:
  - Q/K/V are separate `Linear(hidden_size -> hidden_size)` with bias.
  - Head dim is `hidden_size / num_attention_heads`; canonical RemBERT is `1152 / 18 = 64`.
  - Attention scores are `matmul(q, k.T) / sqrt(head_dim) + extended_attention_mask`, then softmax over `dim=-1`.
  - Attention output is `Linear(hidden_size -> hidden_size)` plus residual plus LayerNorm.
  - MLP is `Linear(hidden_size -> intermediate_size)`, activation, `Linear(intermediate_size -> hidden_size)`, residual plus LayerNorm.
- Decoder/cross-attention paths exist in the generic source when `is_decoder=True` and `add_cross_attention=True`, but canonical public configs are encoder-style.
- MLM/causal-LM head is factorized: `Linear(hidden_size -> output_embedding_size)`, activation, LayerNorm at `output_embedding_size`, `Linear(output_embedding_size -> vocab_size)`. Output embeddings are not tied to input embeddings by default.
- Pooler is first-token gather `hidden_states[:, 0]`, `Linear(hidden_size -> hidden_size)`, tanh.
- Heads implemented: base model, masked LM, causal LM, sequence classification, multiple choice, token classification, question answering.
- Tokenizer is `RemBertTokenizer`, backed by the `tokenizers` library Unigram model. It builds `[CLS] A [SEP]` and `[CLS] A [SEP] B [SEP]` templates and emits `input_ids` plus `attention_mask`; token-type IDs are model-supported but not listed in `model_input_names`.

Representative config files saved beside this note:

- `google-rembert-config.json`
- `Sindhu-rembert-squad2-config.json`
- `Misha24-10-rembert-ft-for-multi-ner-config.json`
- `ibraheemmoosa-xlmindic-rembert-uniscript-config.json`
- `ydshieh-tiny-random-rembert-config.json`
- `google-rembert-tokenizer_config.json`

Fetch notes:

- `google/rembert-ft-xnli`, `google/rembert-ft-squad`, and `google/rembert-ft-tydiqa` returned an auth-style error from raw config fetch at the attempted URLs. The accessible sweep therefore uses `google/rembert` plus open fine-tuned/community checkpoints.
