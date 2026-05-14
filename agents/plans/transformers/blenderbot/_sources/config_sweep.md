# BlenderBot Config Sweep Notes

Transformers checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Local source files inspected:

- `X:/H/transformers/src/transformers/models/blenderbot/configuration_blenderbot.py`
- `X:/H/transformers/src/transformers/models/blenderbot/modeling_blenderbot.py`
- `X:/H/transformers/src/transformers/models/blenderbot/tokenization_blenderbot.py`

Remote config URLs fetched:

- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-400M-distill/raw/main/tokenizer_config.json`
- `https://huggingface.co/facebook/blenderbot-1B-distill/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-1B-distill/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot-3B/raw/main/tokenizer_config.json`
- `https://huggingface.co/hf-internal-testing/tiny-random-BlenderbotModel/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-90M/raw/main/config.json` for out-of-scope routing only.

Representative checkpoint summary:

| Model id | Scope | `model_type` | Architecture | `d_model` | Enc/Dec layers | Heads | FFN | Vocab | Max pos | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `hf-internal-testing/tiny-random-BlenderbotModel` | debug | `blenderbot` | `BlenderbotModel` | 16 | 2 / 2 | 4 / 4 | 4 / 4 | 1024 | 100 | no LM head in architecture; source still same family |
| `facebook/blenderbot-400M-distill` | common production | `blenderbot` | `BlenderbotForConditionalGeneration` | 1280 | 2 / 12 | 32 / 32 | 5120 / 5120 | 8008 | 128 | `scale_embedding=true`; several legacy fields in config |
| `facebook/blenderbot-1B-distill` | larger distill | `blenderbot` | `BlenderbotForConditionalGeneration` | 2560 | 2 / 12 | 32 / 32 | 10240 / 10240 | 8008 | 128 | `decoder_start_token_id` omitted; source default is 1 |
| `facebook/blenderbot-3B` | large production | `blenderbot` | `BlenderbotForConditionalGeneration` | 2560 | 2 / 24 | 32 / 32 | 10240 / 10240 | 8008 | 128 | `use_cache` omitted; source default is true |
| `facebook/blenderbot-90M` | out of scope | `blenderbot-small` | `BlenderbotSmallForConditionalGeneration` | 512 | 8 / 8 | 16 / 16 | 2048 / 2048 | 54944 | 512 | different family/source directory; route to `blenderbot_small` audit |

Generation config observed for 400M/1B/3B:

- `decoder_start_token_id=1`, `bos_token_id=1`, `eos_token_id=2`, `pad_token_id=0`
- `num_beams=10`, `max_length=60`, `min_length=20`, `length_penalty=0.65`
- `no_repeat_ngram_size=3`, `encoder_no_repeat_ngram_size=3`, `forced_eos_token_id=2`

Current-source ignored or historical config fields in representative checkpoints:

- `add_bias_logits`, `add_final_layer_norm`, `classif_dropout`, `extra_layer_norm`, `extra_pos_embeddings`, `force_bos_token_to_be_generated`, `gradient_checkpointing`, `layernorm_variant`, `normalize_before`, `normalize_embedding`, `static_position_embeddings`, and generation fields such as `length_penalty` are not declared in the inspected `BlenderbotConfig` body and are not read in `modeling_blenderbot.py`.
- `num_hidden_layers` appears in configs and maps through `attribute_map` to `encoder_layers`; it is not a decoder-layer count.
- `decoder_start_token_id` and `use_cache` may be omitted by older configs; effective current defaults are `1` and `True`.

