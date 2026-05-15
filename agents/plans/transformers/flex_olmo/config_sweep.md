# FlexOlmo Config Sweep Snapshot

Source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Hub queries run: 2026-05-13

Representative public checkpoints:

| Model id | Hub sha / note | model_type | architectures | hidden | layers | heads | kv heads | head_dim | intermed. | experts | top_k | max pos | rope_theta | dtype field | gated |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `allenai/FlexOlmo-7x7B-1T` | API sha `0dabfa2e5fd7d8c4bec7e3d3b1a05608255fb096` | `flex_olmo` | `FlexOlmoForCausalLM` | 4096 | 32 | 32 | 32 | 128 inferred | 11008 | 7 | 7 | 4096 | 500000 | `dtype=float32` | false |
| `allenai/FlexOlmo-7x7B-1T-RT` | raw config accessible | `flex_olmo` | `FlexOlmoForCausalLM` | 4096 | 32 | 32 | 32 | 128 inferred | 11008 | 7 | 7 | 4096 | 500000 | `dtype=float32` | not checked by API |
| `allenai/Flex-reddit-2x7B-1T` | raw config accessible | `flex_olmo` | `FlexOlmoForCausalLM` | 4096 | 32 | 32 | 32 | 128 inferred | 11008 | 2 | 2 | 4096 | 500000 | `torch_dtype=float32` | not checked by API |
| `allenai/Flex-code-2x7B-1T` | raw config accessible | `flex_olmo` | `FlexOlmoForCausalLM` | 4096 | 32 | 32 | 32 | 128 inferred | 11008 | 2 | 2 | 4096 | 500000 | `torch_dtype=float32` | not checked by API |
| `allenai/Flex-public-7B-1T` | raw config accessible, out of scope | `olmo2` | `Olmo2ForCausalLM` | 4096 | 32 | 32 | 32 | 128 inferred | 11008 | n/a | n/a | 4096 | 500000 | `torch_dtype=float32` | not checked by API |

Source defaults from `configuration_flex_olmo.py`:

| Field | Default |
| --- | --- |
| `vocab_size` | 100352 |
| `hidden_size` | 4096 |
| `intermediate_size` | 11008 |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | `None`, then `__post_init__` sets it to `num_attention_heads` |
| `hidden_act` | `silu` |
| `max_position_embeddings` | 4096 |
| `rms_norm_eps` | `1e-6` |
| `use_cache` | true |
| `pad_token_id` | 100277 |
| `bos_token_id` | `None` in source default; main checkpoint sets 100257 |
| `eos_token_id` | 100257 |
| `tie_word_embeddings` | false |
| `attention_bias` | false |
| `attention_dropout` | 0.0 |
| `num_experts_per_tok` | 5 in source default; observed checkpoints use 2 or 7 |
| `num_experts` | 7 in source default; observed checkpoints use 2 or 7 |
| `norm_topk_prob` | false |
| `output_router_logits` | false |
| `router_aux_loss_coef` | 0.01 |

Tokenizer snapshot for `allenai/FlexOlmo-7x7B-1T`:

- `tokenizer_class`: `GPT2Tokenizer`
- `model_max_length`: 8192, larger than model config `max_position_embeddings=4096`; generation admission should respect model position limit unless explicitly supporting longer RoPE config.
- `bos_token`, `eos_token`, `unk_token`: `<|endoftext|>` id 100257
- `pad_token`: `<|pad|>` id 100277
- chat template inserts `<|im_start|>{role}\n{content}<|im_end|>\n`, then optional `<|im_start|>assistant\n`.

Weight/index observations:

- `allenai/FlexOlmo-7x7B-1T` safetensors index metadata reports `total_parameters=33270665216`, `total_size=133082660864`, and `F32` parameters in the model API.
- `allenai/Flex-reddit-2x7B-1T` index metadata reports `total_size=46509604864`; no `total_parameters` field was present in the fetched index.
- Public checkpoint weight names store per-expert `model.layers.N.mlp.experts.E.gate_proj.weight`, `up_proj.weight`, and `down_proj.weight`.
- The generated source module stores packed expert parameters as `mlp.experts.gate_up_proj` with shape `[num_experts, 2 * intermediate_size, hidden_size]` and `mlp.experts.down_proj` with shape `[num_experts, hidden_size, intermediate_size]`.
- Transformers has a generic conversion mapping for `qwen2_moe`-style source keys that merges `mlp.experts.*.gate_proj.weight` plus `up_proj.weight` into packed `mlp.experts.gate_up_proj`, and merges `down_proj.weight` into packed `mlp.experts.down_proj`. DinoML should mirror or pre-apply this load conversion instead of treating split per-expert checkpoint tensors as independent runtime modules.

Raw config URLs:

- <https://huggingface.co/allenai/FlexOlmo-7x7B-1T/raw/main/config.json>
- <https://huggingface.co/allenai/FlexOlmo-7x7B-1T-RT/raw/main/config.json>
- <https://huggingface.co/allenai/Flex-reddit-2x7B-1T/raw/main/config.json>
- <https://huggingface.co/allenai/Flex-code-2x7B-1T/raw/main/config.json>
- <https://huggingface.co/allenai/Flex-public-7B-1T/raw/main/config.json>
