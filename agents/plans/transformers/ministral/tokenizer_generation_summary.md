# Tokenizer/generation snapshot summary

Fetched from Hugging Face raw files during the audit. Raw tokenizer configs were not kept because they are mostly chat-template payload and are larger than useful audit snapshots.

| Repo | Tokenizer class | BOS | EOS | PAD | Chat template | Generation BOS/EOS/PAD |
|---|---|---|---|---|---|---|
| `mistralai/Ministral-8B-Instruct-2410` | `LlamaTokenizer` | `<s>` | `</s>` | null | present | 1 / 2 / null |
| `mistralai/Ministral-3-3B-Instruct-2512` | `TokenizersBackend` | `<s>` | `</s>` | `<pad>` | absent | 1 / 2 / 11 |
| `mistralai/Ministral-3-8B-Instruct-2512` | `TokenizersBackend` | `<s>` | `</s>` | `<pad>` | absent | 1 / 2 / 11 |
| `mistralai/Ministral-3-14B-Instruct-2512` | `TokenizersBackend` | `<s>` | `</s>` | `<pad>` | absent | 1 / 2 / 11 |
