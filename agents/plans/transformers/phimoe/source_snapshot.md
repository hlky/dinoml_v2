# PhiMoE Source Snapshot

Source checkout: `transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Files inspected:

- `src/transformers/models/phimoe/configuration_phimoe.py`
- `src/transformers/models/phimoe/modeling_phimoe.py`
- `src/transformers/models/phimoe/modular_phimoe.py`
- `src/transformers/modeling_rope_utils.py`
- `src/transformers/configuration_utils.py`

Small source-derived facts:

- `modeling_phimoe.py` is generated from `modular_phimoe.py`; future source edits belong in `modular_phimoe.py`.
- Native `PhimoeConfig` declares `model_type="phimoe"`, `hidden_size=4096`, `intermediate_size=6400`, `num_hidden_layers=32`, `num_attention_heads=32`, `num_key_value_heads=8`, `num_local_experts=16`, `num_experts_per_tok=2`, and default `attention_bias=False`, `lm_head_bias=False`.
- Official `microsoft/Phi-3.5-MoE-instruct` config overrides both projection bias flags to `true`, uses `torch_dtype=bfloat16`, `sliding_window=131072`, LongRoPE, and `max_position_embeddings=131072`.
- Attention projections are separate linear layers: `q_proj: hidden -> num_attention_heads * head_dim`, `k_proj/v_proj: hidden -> num_key_value_heads * head_dim`, `o_proj: num_attention_heads * head_dim -> hidden`.
- Keys are RoPE-rotated before cache update, so KV cache stores post-RoPE keys and unrotated values.
- MoE expert weights are stored packed by expert as `gate_up_proj[num_experts, 2 * intermediate_size, hidden_size]` and `down_proj[num_experts, hidden_size, intermediate_size]`.
- The eager expert path uses `one_hot`, `permute(2, 1, 0)`, `nonzero`, `where`, per-expert `linear`, `chunk(2)`, `silu(gate) * up`, and `index_add_`.
- `PhimoeDecoderLayer` uses `nn.LayerNorm`, not RMSNorm, for input, post-attention, and final model norm.
- The official HF repo exposes remote-code `auto_map` entries, but the pinned Transformers checkout has native in-library support.
