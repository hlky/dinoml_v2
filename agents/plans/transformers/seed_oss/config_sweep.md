# Seed-OSS Representative Config Snapshot

Source: raw Hugging Face `config.json` files fetched 2026-05-13.

| Model id | model_type | hidden | layers | Q heads | KV heads | head_dim | intermediate | vocab | max positions | RoPE | dtype | params/index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| ByteDance-Seed/Seed-OSS-36B-Instruct | seed_oss | 5120 | 64 | 80 | 8 | 128 | 27648 | 155136 | 524288 | default, theta 10000000.0 | bfloat16 | 36151104512 |
| ByteDance-Seed/Seed-OSS-36B-Base | seed_oss | 5120 | 64 | 80 | 8 | 128 | 27648 | 155136 | 524288 | default, theta 10000000.0 | bfloat16 | 36151104512 |
| ByteDance-Seed/Seed-OSS-36B-Base-woSyn | seed_oss | 5120 | 64 | 80 | 8 | 128 | 27648 | 155136 | 524288 | default, theta 10000000.0 | bfloat16 | not fetched; config matches Base/Instruct |

All three configs contained:

```json
{
  "architectures": ["SeedOssForCausalLM"],
  "attention_bias": true,
  "attention_out_bias": false,
  "mlp_bias": false,
  "hidden_act": "silu",
  "rope_scaling": {"rope_type": "default"},
  "rope_theta": 10000000.0,
  "tie_word_embeddings": false,
  "use_cache": true
}
```

Tokenizer/generation observations from `ByteDance-Seed/Seed-OSS-36B-Instruct`:

- `tokenizer_class`: `PreTrainedTokenizerFast`
- `bos_token_id=0` (`<seed:bos>`), `pad_token_id=1` (`<seed:pad>`), `eos_token_id=2` (`<seed:eos>`)
- prompt/control tokens include `<seed:think>`, `</seed:think>`, `<seed:cot_budget_reflect>`, `</seed:cot_budget_reflect>`, `<seed:tool_call>`, `</seed:tool_call>`
- `generation_config.json`: `temperature=1.1`, `top_p=0.95`
