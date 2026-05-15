# XGLM Source Notes

Audit scope: `xglm` only.

Pinned Transformers checkout:

```text
transformers
commit b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Local source snapshots copied for this audit:

- `configuration_xglm.py`
- `modeling_xglm.py`
- `tokenization_xglm.py`
- `convert_xglm_original_ckpt_to_trfms.py`

Representative Hugging Face config snapshots fetched on 2026-05-13:

- [`facebook/xglm-564M`](https://huggingface.co/facebook/xglm-564M)
  - `config.json`
  - `generation_config.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `repo_info.json`
- [`facebook/xglm-1.7B`](https://huggingface.co/facebook/xglm-1.7B)
  - `config.json`
  - `generation_config.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
- [`facebook/xglm-2.9B`](https://huggingface.co/facebook/xglm-2.9B)
  - `config.json`
  - `generation_config.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
- [`facebook/xglm-4.5B`](https://huggingface.co/facebook/xglm-4.5B)
  - `config.json`
  - `generation_config.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
- [`facebook/xglm-7.5B`](https://huggingface.co/facebook/xglm-7.5B)
  - `config.json`
  - `generation_config.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `repo_info.json`

Representative source anchors:

- `configuration_xglm.py`
  - config defaults and attribute aliases: lines 41-69
- `modeling_xglm.py`
  - scaled token embedding: lines 39-50
  - sinusoidal positional embedding: lines 53-103
  - attention projection/cache/math path: lines 105-246
  - decoder block: lines 248-340
  - model embedding/mask/position path: lines 358-483
  - LM head and `logits_to_keep`: lines 494-575
- `tokenization_xglm.py`
  - tokenizer class and model inputs: lines 28-59
  - made-up special tokens and Unigram tokenizer construction: lines 74-93
  - normalizer/metaspace/decoder: lines 95-104
  - template post-processing: lines 116-124
- `convert_xglm_original_ckpt_to_trfms.py`
  - fairseq config mapping and weight renaming: lines 28-57

Notable source findings:

- Despite the common decoder-LM shape, this Transformers source does not use learned absolute positional embeddings. `XGLMSinusoidalPositionalEmbedding` is a non-persistent buffer that can be extended on demand and indexes positions with an offset of 2.
- Official representative checkpoints keep `vocab_size=256008`, `max_position_embeddings=2048`, `scale_embedding=true`, `use_cache=true`, and tied LM/input embeddings.
- Projection dimensions vary by checkpoint. `head_dim = d_model / attention_heads` is 64 for 564M and 128 for the larger inspected checkpoints.
- `facebook/xglm-4.5B` is the main inspected geometry trap: it uses `activation_function="relu"` and `ffn_dim=16384`, unlike the GELU variants.
- Tokenizer repositories include `sentencepiece.bpe.model` and `tokenizer.json`. The current Python tokenizer class constructs a tokenizers `Unigram` backend with metaspace normalization when instantiated directly, while pretrained loading should be treated as coupled to the repo tokenizer artifacts.
