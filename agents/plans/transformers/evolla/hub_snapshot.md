# Evolla Hub Snapshot

Retrieved: 2026-05-13

| Model id | Access | Config result | Notes |
| --- | --- | --- | --- |
| `westlake-repl/Evolla-10B-hf` | Public | `config.json` HTTP 200 | HF safetensors metadata reports 10,392,101,680 F32 parameters and 9 safetensor shards. |
| `westlake-repl/Evolla-10B-DPO-hf` | Public | `config.json` HTTP 200 | Operator-significant config fields match `Evolla-10B-hf`; DPO changes weights, not architecture. |
| `westlake-repl/Evolla-10B` | Public | Raw/non-HF checkpoint layout | Contains raw checkpoint files under `Evolla-10B/`; not used as primary DinoML source basis. |
| `westlake-repl/Evolla-80B` | Public listing | Raw/non-HF split checkpoint layout | No Transformers-format config in the files inspected through Hub API. |
| `westlake-repl/Evolla-80B-hf` | Guessed URL | `config.json` HTTP 401 | Include as gated or unavailable guessed HF-converted link until an official model id is confirmed. |
| `westlake-repl/Evolla-80B-DPO-hf` | Guessed URL | `config.json` HTTP 401 | Include as gated or unavailable guessed HF-converted link until an official model id is confirmed. |

Tokenizer/processor notes for `westlake-repl/Evolla-10B-hf`:

- `processor_config.json`: `processor_class="EvollaProcessor"`, `protein_max_length=1024`, `text_max_length=512`.
- Text tokenizer uses Llama-3 style chat formatting and sets pad token to `<|reserved_special_token_0|>`.
- Protein tokenizer files are under `protein_tokenizer/`; processor constructs structure-aware protein strings by interleaving uppercase amino-acid symbols and lowercase Foldseek symbols.
