# JetMoe Config Sweep Snapshot

Source basis: Hugging Face Hub configs fetched during audit for native
Transformers `model_type="jetmoe"` at source checkout
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

| model id | status | architecture | hidden | layers | kv heads | top-k | effective attn heads | kv/head dim | intermediate | experts | max positions | tokenizer/runtime notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `jetmoe/jetmoe-8b` | in scope | `JetMoEForCausalLM` | 2048 | 24 | 16 | 2 | 32 | 128 | 5632 | 8 | 4096 | base model; LlamaTokenizer metadata, left padding, BOS 1, EOS/PAD 2 |
| `jetmoe/jetmoe-8b-sft` | in scope | `JetMoEForCausalLM` | 2048 | 24 | 16 | 2 | 32 | 128 | 5632 | 8 | 4096 | SFT weights; config operator surface matches base |
| `jetmoe/jetmoe-8b-chat` | in scope with native-source guard | `JetMoEForCausalLM` | 2048 | 24 | 16 | 2 | 32 | 128 | 5632 | 8 | 4096 | adds `auto_map` custom-code metadata and chat template; operator surface matches base when routed to native source |
| `AndreaUnibo/JetMoE_base_full_trained` | in scope with quantization guard | `JetMoEForCausalLM` | 2048 | 24 | 16 | 2 | 32 | 128 | 5632 | 8 | 4096 | includes BitsAndBytes NF4 `quantization_config`; native source does not implement this storage format |
| `thomasgauthier/expanded-jetmoe-untrained` | out of scope | `ExpandedJetMoEForCausalLM` | 2048 | 24 | 16 | 2 | 32 | 128 | 5632 | attention experts 8, MLP experts 9 | 4096 | `model_type="expandedjetmoe"`; separate remote/native source audit required |

Compatibility notes:

- Official configs use historical field names such as `ffn_hidden_size`,
  `moe_num_experts`, `moe_top_k`, `n_positions`, `rope_theta`, and
  `layer_norm_epsilon`. The inspected native source reads
  `intermediate_size`, `num_local_experts`, `num_experts_per_tok`,
  `max_position_embeddings`, `rope_parameters`, and `rms_norm_eps`. Defaults
  line up for the 8B family except `num_hidden_layers`, which must come from
  the checkpoint config.
- `JetMoeConfig.__post_init__` derives
  `num_attention_heads = num_key_value_heads * num_experts_per_tok`, so a
  top-k change changes the attention query-head count.
- No inspected official JetMoe repo was gated. The chat repo advertises
  custom-code metadata; this report is scoped to the in-library native source.
