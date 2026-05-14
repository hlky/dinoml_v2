# GPTNeoX Japanese Config Sweep

Source basis: Hugging Face Hub `config.json` reads on 2026-05-13 plus local
Transformers checkout `X:/H/transformers` at
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## In-scope `model_type: gpt_neox_japanese`

| Model id | Access | Architecture | Key dimensions |
| --- | --- | --- | --- |
| `abeja/gpt-neox-japanese-2.7b` | OK | `GPTNeoXJapaneseForCausalLM` | vocab 32000, hidden 2560, layers 32, heads 32, intermediate multiple 4, max positions 2048, RoPE pct 1.0/base 10000, cache true |
| `hf-tiny-model-private/tiny-random-GPTNeoXJapaneseForCausalLM` | OK | `GPTNeoXJapaneseForCausalLM` | vocab 32000, hidden 32, layers 5, heads 4, intermediate multiple 4, max positions 512, RoPE pct 1.0/base 10000, dtype float32 |
| `hf-tiny-model-private/tiny-random-GPTNeoXJapaneseModel` | OK | `GPTNeoXJapaneseModel` | vocab 32000, hidden 32, layers 5, heads 4, intermediate multiple 4, max positions 512, RoPE pct 1.0/base 10000, dtype float32 |
| `optimum-intel-internal-testing/tiny-random-GPTNeoXJapaneseForCausalLM` | OK | `GPTNeoXJapaneseForCausalLM` | same tiny dimensions as above, `transformers_version` 4.25.0.dev0 |

## Access gaps

| Model id | Result |
| --- | --- |
| `abeja/gpt-neox-japanese-2.7b-ppo` | `config.json` returned 401 Unauthorized |
| `phnghiapro/gpt-neox-japanese-2.7b-finetune-lang8` | `config.json` returned 404 Not Found |

## Out-of-scope Japanese GPT-NeoX-like checkpoints

These are useful contrasts but do not route to `GPTNeoXJapaneseConfig` or
`GPTNeoXJapaneseForCausalLM` in the inspected source.

| Model id | model_type | Architecture | Operator-significant difference |
| --- | --- | --- | --- |
| `rinna/japanese-gpt-neox-small` | `gpt_neox` | `GPTNeoXForCausalLM` | generic GPT-NeoX, hidden 768, heads 12, layers 12, untied embeddings, T5 tokenizer |
| `rinna/japanese-gpt-neox-3.6b` | access gap for raw during this sweep; search result shows `gpt_neox` | `GPTNeoXForCausalLM` | generic GPT-NeoX, hidden 2816, heads 22, layers 36 |
| `stockmark/gpt-neox-japanese-1.4b` | `gpt_neox` | `GPTNeoXForCausalLM` | generic GPT-NeoX, hidden 2048, heads 16, layers 24, partial RoPE pct 0.25, bf16 |
| `line-corporation/japanese-large-lm-1.7b` | `gpt2` | `GPT2LMHeadModel` | GPT-2 config/body, not GPT-NeoX Japanese |
