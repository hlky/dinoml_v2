# Hunyuan v1 MoE Config Sweep Snapshot

Source date: 2026-05-13. Hub metadata was read through the Hugging Face plugin and raw `config.json` URLs.

| Repo | Scope for this audit | model_type | architecture | Hidden / layers | Heads / KV / head dim | Intermediate | Experts / top-k | Context | Vocab | Dtype | Quantization | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tencent/Hunyuan-A13B-Instruct` | Main representative checkpoint | `hunyuan_v1_moe` | `HunYuanMoEV1ForCausalLM` | 4096 / 32 | 32 / 8 / 128 | 3072 | 64 / 8 per layer | 32768 | 128167 | bf16 | none | Native-family candidate, but config still has `auto_map` to remote code and many legacy Hunyuan fields. |
| `tencent/Hunyuan-A13B-Instruct-GPTQ-Int4` | Out of scope for native `hunyuan_v1_moe` runtime | `hunyuan` | `HunYuanMoEV1ForCausalLM` | 4096 / 32 | 32 / 8 / 128 | 3072 | 64 / 8 per layer | 32768 | 128167 | bf16 | GPTQ int4 | Uses remote/legacy `hunyuan` model type plus GPTQ metadata. Requires separate quantized-weight admission. |
| `tencent/Hunyuan-A13B-Instruct-FP8` | Out of scope for native `hunyuan_v1_moe` runtime | `hunyuan` | `HunYuanMoEV1ForCausalLM` | 4096 / 32 | 32 / 8 / 128 | 3072 | 64 / 8 per layer | 32768 | 128167 | bf16 | FP8, `lm_head` ignored | Uses remote/legacy `hunyuan` model type plus FP8 metadata. Requires separate quantized-weight admission. |
| `bullerwins/Hunyuan-A13B-Instruct-hf` | Open mirror / conversion reference only | `hunyuan` | `HunYuanMoEV1ForCausalLM` | 4096 / 32 | 32 / 8 / 128 | 3072 | 64 / 8 per layer | 32768 | 128167 | bf16 | none | Mirror reports `hunyuan`, not native `hunyuan_v1_moe`; do not treat as native without conversion. |
| Source default `HunYuanMoEV1Config` | Tiny/debug synthetic config source | `hunyuan_v1_moe` | n/a | 4096 / 32 | 32 / defaults to 32 / 128 inferred | 11008 | 1 / 1 | 2048 | 290943 | unspecified | none | Defaults are not representative of A13B and need RoPE parameters before model construction is meaningful. |

Representative official URLs:

- https://huggingface.co/tencent/Hunyuan-A13B-Instruct
- https://huggingface.co/tencent/Hunyuan-A13B-Instruct-GPTQ-Int4
- https://huggingface.co/tencent/Hunyuan-A13B-Instruct-FP8
- https://huggingface.co/bullerwins/Hunyuan-A13B-Instruct-hf
- https://huggingface.co/bullerwins/Hunyuan-A13B-Instruct-GGUF

Observed tokenizer config for `tencent/Hunyuan-A13B-Instruct`:

- `tokenizer_class`: `PreTrainedTokenizerFast`
- `model_max_length`: 262144, larger than model `max_position_embeddings` 32768.
- Special tokens include `<|startoftext|>`, `<|eos|>`, `<|pad|>`.
- Chat template injects `<|startoftext|>`, `<|extra_0|>`, `<|extra_4|>`, and `<|eos|>` by role. This is generation-controller / tokenizer ABI, not a neural graph op.

