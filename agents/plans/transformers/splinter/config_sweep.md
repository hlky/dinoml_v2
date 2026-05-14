# Splinter Config Sweep

Source checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Fetched from official Hugging Face model repositories on 2026-05-13:

| Model id | Access | Architecture | Hidden | Layers | Heads | Head dim | FFN | Vocab | Max positions | QASS note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tau/splinter-base` | public | `SplinterForQuestionAnswering` | 768 | 12 | 12 | 64 | 3072 | 28996 | 512 | no pretrained QASS weights in repo note; source initializes missing head weights |
| `tau/splinter-base-qass` | public | `SplinterForQuestionAnswering` | 768 | 12 | 12 | 64 | 3072 | 28996 | 512 | config contains historical `initialize_new_qass: true`; current source does not read it |
| `tau/splinter-large` | public | `SplinterForQuestionAnswering` | 1024 | 24 | 16 | 64 | 4096 | 28996 | 512 | no pretrained QASS weights in repo note; source initializes missing head weights |
| `tau/splinter-large-qass` | public | `SplinterForQuestionAnswering` | 1024 | 24 | 16 | 64 | 4096 | 28996 | 512 | config contains historical `initialize_new_qass: true`; current source does not read it |

Common config values from the fetched `config.json` files:

- `model_type: "splinter"`
- `hidden_act: "gelu"`
- `hidden_dropout_prob: 0.1`
- `attention_probs_dropout_prob: 0.1`
- `initializer_range: 0.02`
- `layer_norm_eps: 1e-12`
- `pad_token_id: 0`
- `type_vocab_size: 2`
- `question_token_id` omitted in all four configs; current `SplinterConfig` supplies default `104`.

Common tokenizer snapshots:

- `tokenizer_config.json`: `{"do_lower_case": false, "model_max_length": 512}`
- `special_tokens_map.json`: `unk=[UNK]`, `sep=[SEP]`, `pad=[PAD]`, `cls=[CLS]`, `mask=[MASK]`, `question=[QUESTION]`

HF API access notes:

- All four official tau repositories returned public metadata and config files.
- No gated or 401/403 Splinter checkpoint was encountered in this representative sweep.
