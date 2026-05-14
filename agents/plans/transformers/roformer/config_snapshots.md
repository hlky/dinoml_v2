# RoFormer config snapshots

Fetched on 2026-05-13 from Hugging Face raw/API endpoints. These are source notes for `report.md`.

| Model id | Access | Library/API tags | Architecture | Hidden | Embedding | Layers | Heads | Head dim | Intermediate | Vocab | Max positions | Activation | Notable fields |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `junnyu/roformer_chinese_small` | open | `transformers`, `fill-mask` | `RoFormerForMaskedLM` | 384 | 384 | 6 | 6 | 64 | 1536 | 50000 | omitted, effective source default 1536; tokenizer config says 512 | `gelu` | `type_vocab_size=2`, `pad_token_id=0` |
| `junnyu/roformer_chinese_base` | open | API `library_name=paddlenlp`, raw config compatible | `RoFormerForMaskedLM` | 768 | 768 | 12 | 12 | 64 | 3072 | 50000 | 1536 | `gelu` | tokenizer config `model_max_length=1536` |
| `junnyu/roformer_chinese_char_small` | open | `transformers`, `fill-mask` | `RoFormerForMaskedLM` | 384 | 384 | 6 | 6 | 64 | 1536 | 12000 | 512 | `gelu` | `rotary_value=false`, `use_cache=true`, tokenizer class `BertTokenizer` |
| `junnyu/roformer_chinese_char_base` | open | API `library_name=paddlenlp`, raw config compatible | `RoFormerForMaskedLM` | 768 | omitted, effective source default 768 | 12 | 12 | 64 | 3072 | 12000 | 512 | `gelu` | tokenizer class `BertTokenizer` |
| `junnyu/roformer_v2_chinese_char_base` | open | `transformers`, `fill-mask`, `roformer-v2` | `RoFormerForMaskedLM` | 768 | 768 | 12 | 12 | 64 | 3072 | 12000 | 512 | `relu` | Contains `norm_type=rms_norm`, `use_bias=false`, `transformers_version=4.15.0`; inspected source ignores `norm_type` and `use_bias` |

Tokenizer snapshots:

- `junnyu/roformer_chinese_base` and `junnyu/roformer_chinese_small` use `RoFormerTokenizer`, Jieba-backed WordPiece, lowercase, `[CLS] X [SEP]` / `[CLS] A [SEP] B [SEP]`, token types 0 then 1.
- Character checkpoints use `BertTokenizer` in tokenizer config, same BERT-style special-token layout.
- No gated/401 checkpoint config was encountered in this sweep.
