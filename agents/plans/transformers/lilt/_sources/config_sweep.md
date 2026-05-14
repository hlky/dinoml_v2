# LiLT config sweep notes

Fetched from Hugging Face Hub on 2026-05-13 with `Invoke-WebRequest` into this
directory. Missing `preprocessor_config.json` for the two official SCUT-DLVCLab
repos returned 404, which matches the source contract: LiLT uses layout-aware
tokenizers and caller-supplied word boxes, not an in-library OCR/image
processor.

| Repo | Role | Architecture | Hidden | Layers | Heads | Head dim | Layout dim | Layout head dim | FFN | Vocab | Max pos | Max 2D pos | Type vocab | Tokenizer class |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `SCUT-DLVCLab/lilt-roberta-en-base` | official English base | `LiltModel` | 768 | 12 | 12 | 64 | 192 | 16 | 3072 | 50265 | 514 | 1024 | 1 | `LayoutLMv3Tokenizer` |
| `SCUT-DLVCLab/lilt-infoxlm-base` | official multilingual base | `LiltModel` | 768 | 12 | 12 | 64 | 192 | 16 | 3072 | 250002 | 514 | 1024 | 1 | `LayoutXLMTokenizer` |
| `nielsr/lilt-xlm-roberta-base` | open mirror/derived base | `LiltModel` | 768 | 12 | 12 | 64 | 192 | 16 | 3072 | 250002 | 514 | 1024 | 1 | `LayoutXLMTokenizer` |
| `dharmik3005/lilt-en-funsd` | fine-tuned token classification | `LiltForTokenClassification` | 768 | 12 | 12 | 64 | 192 | 16 | 3072 | 50265 | 514 | 1024 | 1 | `LayoutLMv3Tokenizer` |
| `koshkidadanet/lilt-xlm-roberta-base-finetuned-piad` | fine-tuned token classification | `LiltForTokenClassification` | 768 | 12 | 12 | 64 | 192 | 16 | 3072 | 250002 | 514 | 1024 | 1 | tokenizer files not present in repo snapshot |
| `hf-internal-testing/tiny-random-LiltForSequenceClassification` | tiny/debug | `LiltForSequenceClassification` | 24 | 2 | 6 | 4 | 6 | 1 | 37 | 1024 | 512 | 1024 | 16 | tokenizer files present, debug only |

Source-default differences from official configs:

- `LiltConfig` defaults to `vocab_size=30522`, `max_position_embeddings=512`,
  `type_vocab_size=2`, `layer_norm_eps=1e-12`, and `pad_token_id=0`.
- Official RoBERTa/XLM-style LiLT configs override these to `max_position_embeddings=514`,
  `type_vocab_size=1`, `layer_norm_eps=1e-5`, and `pad_token_id=1`.
- Official configs include historical `use_cache=true`; `lilt-infoxlm-base`
  also includes `output_past=true`. The inspected in-library `modeling_lilt.py`
  does not implement or read autoregressive caches.
