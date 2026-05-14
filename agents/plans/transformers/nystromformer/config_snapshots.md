# Nystromformer config snapshots

Fetched from Hugging Face raw `config.json` endpoints on 2026-05-13. These are source notes only; the standardized audit is in `report.md`.

| Model id | Architecture | hidden | layers | heads | head dim | intermediate | max positions | segment_means_seq_len | num_landmarks | conv_kernel_size | activation | dtype | tokenizer/config notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `uw-madison/nystromformer-512` | `NystromformerForMaskedLM` | 768 | 12 | 12 | 64 | 3072 | 510 | 64 | 64 | 65 | `gelu_new` | `float32` | `tokenizer_class=AlbertTokenizer`, `use_cache=true` present but modeling source has no cache path |
| `uw-madison/nystromformer-1024` | `NystromformerForMaskedLM` | 768 | 12 | 12 | 64 | 3072 | 1024 | 64 | 64 | 65 | `gelu_new` | `float32` | no tokenizer class in config |
| `uw-madison/nystromformer-2048` | `NystromformerForMaskedLM` | 768 | 12 | 12 | 64 | 3072 | 2048 | 64 | 64 | 65 | `gelu_new` | `float32` | no tokenizer class in config |
| `uw-madison/nystromformer-4096` | `NystromformerForMaskedLM` | 768 | 12 | 12 | 64 | 3072 | 4096 | 64 | 64 | 65 | `gelu_new` | `float32` | tokenizer config reports `model_max_length=512`, so caller/tokenizer policy may truncate unless overridden |
| `GBaker/nystromformer-4096-medqa-usmle-nocontext` | `NystromformerForMultipleChoice` | 768 | 12 | 12 | 64 | 3072 | 4096 | 64 | 64 | 65 | `gelu_new` | `float32` | fine-tuned from `uw-madison/nystromformer-4096` |
| `MrAnderson/nystrom-1024-full-trivia` | `NystromformerForQuestionAnswering` | 768 | 12 | 12 | 64 | 3072 | 1024 | 64 | 64 | 65 | `gelu_new` | `float32` | `tokenizer_class=AlbertTokenizer`, `_name_or_path=uw-madison/nystromformer-512` |
| `hf-tiny-model-private/tiny-random-NystromformerForMaskedLM` | `NystromformerForMaskedLM` | 32 | 5 | 4 | 8 | 37 | 512 | 64 | 64 | 65 | `gelu` | `float32` | accessible raw config despite private-style namespace; useful for tiny shape variation only |

Tokenizer side notes from `uw-madison/nystromformer-512` and `uw-madison/nystromformer-4096`:

- `tokenizer_class`: `AlbertTokenizer`
- special tokens: `[CLS]`, `[SEP]`, `<unk>`, `<pad>`, `[MASK]`
- lower-casing and space removal enabled in tokenizer config
- 4096 tokenizer config has `model_max_length=512`, which conflicts with the model config's 4096 position table unless caller explicitly controls tokenizer truncation.
