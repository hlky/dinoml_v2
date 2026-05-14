# Zamba2 Config Evidence Snapshot

Source: public Hugging Face `config.json` and `tokenizer_config.json` files fetched on 2026-05-13. The base `Zyphra/Zamba2-7B` and `Zyphra/Zamba2-7B-v1` `config.json` URLs returned 401, so the 7B row below uses the public instruct checkpoint.

| Model id | Config URL | Hidden | Layers | Mamba layers | Hybrid layers | Attention head dim | Mamba head dim | Mamba groups | Memory blocks | Mem RoPE | Shared adapters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `Zyphra/Zamba2-1.2B` | https://huggingface.co/Zyphra/Zamba2-1.2B/raw/main/config.json | 2048 | 38 | 32 | 6 | 128 | 64 | 1 | 1 | true | true |
| `Zyphra/Zamba2-2.7B` | https://huggingface.co/Zyphra/Zamba2-2.7B/raw/main/config.json | 2560 | 54 | 45 | 9 | 160 | 64 | 1 | 2 | false | false |
| `Zyphra/Zamba2-2.7B-instruct` | https://huggingface.co/Zyphra/Zamba2-2.7B-instruct/raw/main/config.json | 2560 | 54 | 45 | 9 | 160 | 64 | 1 | 2 | false | false |
| `Zyphra/Zamba2-7B-Instruct` | https://huggingface.co/Zyphra/Zamba2-7B-Instruct/raw/main/config.json | 3584 | 81 | 68 | 13 | 224 | 64 | 2 | 2 | true | false |

Tokenizer evidence:

| Model id | Tokenizer class | BOS | EOS | Pad | Model max length | Chat template |
| --- | --- | --- | --- | --- | ---: | --- |
| `Zyphra/Zamba2-1.2B` | `LlamaTokenizer` | `<s>` | `</s>` | `[PAD]` | very large sentinel | none |
| `Zyphra/Zamba2-2.7B` | `LlamaTokenizer` | `<s>` | `</s>` | null | very large sentinel | none |
| `Zyphra/Zamba2-2.7B-instruct` | `LlamaTokenizer` | `<|im_start|>` | `<|im_end|>` | `[PAD]` | 4096 | ChatML-style start/end tags |
| `Zyphra/Zamba2-7B-Instruct` | `LlamaTokenizer` | `<|im_start|>` | `<|im_end|>` | `[PAD]` | 4096 | ChatML-style start/end tags |
