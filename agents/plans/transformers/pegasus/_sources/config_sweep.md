# Pegasus config sweep

Fetched from Hugging Face `raw/main/config.json` on 2026-05-13.

| Model | Architecture | d_model | Enc/Dec layers | Heads | Head dim | FFN | Vocab | Max pos | Activation | scale_embedding | use_cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `google/pegasus-large` | `PegasusForConditionalGeneration` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true | true |
| `google/pegasus-xsum` | `PegasusForConditionalGeneration` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 512 | relu | true | true |
| `google/pegasus-cnn_dailymail` | `PegasusForConditionalGeneration` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true | omitted, defaults true |
| `google/pegasus-arxiv` | `PegasusForConditionalGeneration` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true | omitted, defaults true |
| `google/pegasus-pubmed` | `PegasusForConditionalGeneration` | 1024 | 16/16 | 16/16 | 64 | 4096/4096 | 96103 | 1024 | relu | true | omitted, defaults true |
| `hf-internal-testing/tiny-random-PegasusModel` | `PegasusModel` | 16 | 2/2 | 4/4 | 4 | 4/4 | 96103 | 200 | gelu | false | true |

Current source defaults from `PegasusConfig` differ from common Google checkpoints:

- Source default: 12 encoder layers, 12 decoder layers, `vocab_size=50265`, `activation_function="gelu"`, `scale_embedding=false`, `max_position_embeddings=1024`.
- Common Google checkpoints: 16 encoder layers, 16 decoder layers, `vocab_size=96103`, `activation_function="relu"`, `scale_embedding=true`; task checkpoints may use `max_position_embeddings=512`.

Historical config fields present in some Google configs but not read by current `modeling_pegasus.py`:

- `add_bias_logits`
- `add_final_layer_norm`
- `extra_pos_embeddings`
- `force_bos_token_to_be_generated`
- `gradient_checkpointing`
- `normalize_before`
- `normalize_embedding`
- `static_position_embeddings`
- generation metadata such as `max_length`, `min_length`, `num_beams`, `length_penalty`, `task_specific_params`
