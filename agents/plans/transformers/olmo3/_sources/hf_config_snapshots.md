# OLMo3 HF Config Snapshots

Raw configs were fetched from Hugging Face on 2026-05-13. Values below are reduced to operator-significant fields for audit reproducibility.

## `allenai/Olmo-3-7B-Instruct`

- API sha: `096bb5469fe34348bc88d851a69edb3bf6f40df4`
- `architectures`: `["Olmo3ForCausalLM"]`
- `dtype`: `bfloat16`
- `hidden_size`: `4096`
- `intermediate_size`: `11008`
- `num_hidden_layers`: `32`
- `num_attention_heads`: `32`
- `num_key_value_heads`: `32`
- `vocab_size`: `100278`
- `max_position_embeddings`: `65536`
- `sliding_window`: `4096`
- `layer_types`: 24 `sliding_attention`, 8 `full_attention`, repeating `S,S,S,F`
- `rope_theta`: `500000`
- `rope_scaling`: `{rope_type: yarn, factor: 8.0, original_max_position_embeddings: 8192, beta_fast: 32.0, beta_slow: 1.0, attention_factor: 1.2079441541679836}`
- `attention_bias`: `false`
- `attention_dropout`: `0.0`
- `rms_norm_eps`: `1e-6`
- `hidden_act`: `silu`
- `tie_word_embeddings`: `false`
- `use_cache`: `false`
- `pad_token_id`: `100277`
- `eos_token_id`: `100257`
- Generation config: `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `max_new_tokens=32768`, `eos_token_id=[100265,100257]`

## `allenai/Olmo-3-7B-Think`

- API sha: `7c991fde2671813ab41745054310f60a610b0fac`
- `architectures`: `["Olmo3ForCausalLM"]`
- `torch_dtype`: `bfloat16`
- Same operator-significant model dimensions as `Olmo-3-7B-Instruct`.
- Generation config: `do_sample=true`, `temperature=0.6`, `top_p=0.95`, `max_new_tokens=32768`, `eos_token_id=[100265,100257]`

## `allenai/Olmo-3-32B-Think`

- API sha: `ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`
- `architectures`: `["Olmo3ForCausalLM"]`
- `dtype`: `bfloat16`
- `hidden_size`: `5120`
- `intermediate_size`: `27648`
- `num_hidden_layers`: `64`
- `num_attention_heads`: `40`
- `num_key_value_heads`: `8`
- `vocab_size`: `100278`
- `max_position_embeddings`: `65536`
- `sliding_window`: `4096`
- `layer_types`: 48 `sliding_attention`, 16 `full_attention`, repeating `S,S,S,F`
- `rope_theta`: `500000`
- `rope_scaling`: `{rope_type: yarn, factor: 8.0, original_max_position_embeddings: 8192, beta_fast: 32, beta_slow: 1, attention_factor: 1.2079441541679836}`
- `attention_bias`: `false`
- `attention_dropout`: `0.0`
- `rms_norm_eps`: `1e-6`
- `hidden_act`: `silu`
- `tie_word_embeddings`: `false`
- `use_cache`: `false`
- `pad_token_id`: `100277`
- `eos_token_id`: `100257`
- Generation config: `temperature=0.6`, `top_p=0.95`, `max_new_tokens=32768`, `eos_token_id=100257`, `pad_token_id=100277`

## Gated or unavailable

- `allenai/Olmo-3-32B-Instruct`: raw config returned 401 Unauthorized.
- `allenai/Olmo-3-7B`: raw config returned 401 Unauthorized.
- `allenai/Olmo-3-32B`: raw config returned 401 Unauthorized.

