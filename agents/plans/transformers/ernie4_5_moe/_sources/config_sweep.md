# ernie4_5_moe config sweep

| Model id | Config source | hidden | layers | q heads | kv heads | head_dim | ctx | dense intermediate | moe intermediate | experts | top-k | shared experts | MoE layer range | tied emb | dtype | Source-visible traps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `baidu/ERNIE-4.5-21B-A3B-PT` | HF `config.json` main, fetched 2026-05-13 | 2560 | 28 | 20 | 4 | 128 inferred | 131072 | 12288 | 1536 | 64 | 6 | 2 | 1..27 | true | bf16 | `num_nextn_predict_layers=1` ignored by inspected source |
| `baidu/ERNIE-4.5-21B-A3B-Base-PT` | HF `config.json` main, fetched 2026-05-13 | 2560 | 28 | 20 | 4 | 128 inferred | 131072 | 12288 | 1536 | 64 | 6 | 2 | 1..27 | true | bf16 | same operator shape as PT snapshot |
| `baidu/ERNIE-4.5-21B-A3B-Thinking` | HF `config.json` main, fetched 2026-05-13 | 2560 | 28 | 20 | 4 | 128 inferred | 131072 | 12288 | 1536 | 64 | 6 | 2 | default end resolves to 27 | true | bf16 | `_attn_implementation=eager`; `moe_capacity`, `moe_gate`, `moe_use_aux_free` not read by source |
| `baidu/ERNIE-4.5-300B-A47B-PT` | HF `config.json` main, fetched 2026-05-13 | 8192 | 54 | 64 | 8 | 128 inferred | 131072 | 28672 | 3584 | 64 | 8 | 0 | 3..53 | false | bf16 | no shared experts; `num_nextn_predict_layers=1` ignored |
| `baidu/ERNIE-4.5-300B-A47B-Base-PT` | HF `config.json` main, fetched 2026-05-13 | 8192 | 54 | 64 | 8 | 128 inferred | 131072 | 28672 | 3584 | 64 | 8 | 0 | 3..53 | false | bf16 | same operator shape as 300B PT snapshot |

