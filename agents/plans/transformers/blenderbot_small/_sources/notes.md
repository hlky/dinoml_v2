# BlenderBot Small audit notes

Workspace: `H:/dinoml_v2`
Transformers checkout: `X:/H/transformers`
Transformers commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
Family source dir: `X:/H/transformers/src/transformers/models/blenderbot_small`

Local files inspected:

- `configuration_blenderbot_small.py`
- `modeling_blenderbot_small.py`
- `tokenization_blenderbot_small.py`
- `../blenderbot/configuration_blenderbot.py`
- `../blenderbot/modeling_blenderbot.py`
- `../../masking_utils.py`
- `../../cache_utils.py`
- `X:/H/transformers/tests/models/blenderbot_small/test_modeling_blenderbot_small.py`
- `X:/H/transformers/tests/models/blenderbot_small/test_tokenization_blenderbot_small.py`

Representative config snapshots saved in this directory:

- `facebook_blenderbot_small-90M_config.json`
- `facebook_blenderbot_small-90M_generation_config.json`
- `facebook_blenderbot_small-90M_tokenizer_config.json`
- `facebook_blenderbot-90M_config.json`
- `facebook_blenderbot-90M_generation_config.json`
- `Xenova_blenderbot_small-90M_config.json`
- `lordtt13_blenderbot_small-news_config.json`
- `kellyjiayixu_my_awesome_eli5_clm-model_blenderbot_small_config.json`
- `onnx_tiny_condgen_config.json`
- `onnx_tiny_model_config.json`

Fetch basis:

- `https://huggingface.co/facebook/blenderbot_small-90M/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot_small-90M/raw/main/generation_config.json`
- `https://huggingface.co/facebook/blenderbot_small-90M/raw/main/tokenizer_config.json`
- `https://huggingface.co/facebook/blenderbot-90M/raw/main/config.json`
- `https://huggingface.co/facebook/blenderbot-90M/raw/main/generation_config.json`
- `https://huggingface.co/Xenova/blenderbot_small-90M/raw/main/config.json`
- `https://huggingface.co/lordtt13/blenderbot_small-news/raw/main/config.json`
- `https://huggingface.co/kellyjiayixu/my_awesome_eli5_clm-model_blenderbot_small/raw/main/config.json`
- `https://huggingface.co/onnx-internal-testing/tiny-random-BlenderbotSmallForConditionalGeneration-ONNX/raw/main/config.json`
- `https://huggingface.co/onnx-internal-testing/tiny-random-BlenderbotSmallModel-ONNX/raw/main/config.json`

HF model API checked for `facebook/blenderbot_small-90M`:

- sha: `bbf60f5f68fd8789ac04bd1c20712233f3dc899f`
- library: `transformers`
- pipeline tag: `text2text-generation`
- gated: `false`
- tags include `blenderbot-small`, `convAI`, `conversational`, `license:apache-2.0`

Config/source caveats:

- Current source defaults set `vocab_size=50265` and `scale_embedding=False`, but the official 90M checkpoints set `vocab_size=54944` and `scale_embedding=True`.
- Historical config fields such as `normalize_before`, `normalize_embedding`, `layernorm_variant`, `do_blenderbot_90_layernorm`, `add_final_layer_norm`, `static_position_embeddings`, and `extra_pos_embeddings` are present in old checkpoints but are not read by the inspected current `modeling_blenderbot_small.py`.
- The current source always implements the BlenderBot Small post-attention/post-FFN layernorm layout plus input embedding layernorm; admission should be source-based, not historical-flag-based.
- `facebook/blenderbot-90M` and `facebook/blenderbot_small-90M` are both BlenderbotSmall configs in the snapshots. The local tests use `facebook/blenderbot-90M` for slow generation integration and tokenizer special-token behavior.
