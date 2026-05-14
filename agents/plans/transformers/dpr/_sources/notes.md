# DPR source/config notes

Fetched with Invoke-WebRequest on 2026-05-13 from Hugging Face `resolve/main` URLs.

Transformers local checkout: X:/H/transformers, commit b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Representative configs fetched:
- facebook/dpr-question_encoder-single-nq-base config/tokenizer_config
- facebook/dpr-ctx_encoder-single-nq-base config/tokenizer_config
- facebook/dpr-reader-single-nq-base config/tokenizer_config
- facebook/dpr-question_encoder-multiset-base config/tokenizer_config
- facebook/dpr-ctx_encoder-multiset-base config/tokenizer_config

All fetched configs share base BERT dimensions: hidden_size 768, layers 12, heads 12, intermediate 3072, max_position_embeddings 512, vocab_size 30522, type_vocab_size 2, projection_dim 0, gelu, layer_norm_eps 1e-12. Tokenizer configs set do_lower_case true. special_tokens_map.json returned 404 for these repos; special tokens come from BERT vocab/tokenizer defaults.