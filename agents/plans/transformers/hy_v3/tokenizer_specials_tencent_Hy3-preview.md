# Tencent Hy3-preview Tokenizer Snapshot

Source: https://huggingface.co/tencent/Hy3-preview/raw/main/tokenizer_config.json

Relevant special tokens observed in `added_tokens_decoder` and the chat template:

| Token id | Token |
|---:|---|
| 120000 | `<｜hy_begin▁of▁sentence｜>` |
| 120001 | `<｜hy_end▁of▁sentence｜>` |
| 120002 | `<｜hy_▁pad▁｜>` |
| 120006 | `<｜hy_User｜>` |
| 120007 | `<｜hy_Assistant｜>` |
| 120008 | `<｜hy_EOT｜>` |
| 120025 | `<｜hy_eos｜>` |
| 120026 | `<｜hy_eod｜>` |

Chat-template control tokens include `<think>`, `</think>`, `<tool_calls>`,
`</tool_calls>`, `<tool_call>`, `</tool_call>`, `<tool_sep>`, `<arg_key>`,
`</arg_key>`, `<arg_value>`, `</arg_value>`, `<tool_responses>`,
`</tool_responses>`, `<tool_response>`, `</tool_response>`, and
`<｜reasoning_mode｜>`.
