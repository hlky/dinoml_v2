# Funnel source notes

Local Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `X:/H/transformers/src/transformers/models/funnel`

Local source files inspected:

- `configuration_funnel.py`
- `modeling_funnel.py`
- `tokenization_funnel.py`
- `docs/source/en/model_doc/funnel.md`
- `tests/models/funnel/test_modeling_funnel.py`
- `src/transformers/activations.py`

HF config snapshots saved beside this file:

- `small_config.json` from `https://huggingface.co/funnel-transformer/small/resolve/main/config.json`
- `small-base_config.json` from `https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json`
- `medium_config.json` from `https://huggingface.co/funnel-transformer/medium/resolve/main/config.json`
- `intermediate_config.json` from `https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json`
- `large_config.json` from `https://huggingface.co/funnel-transformer/large/resolve/main/config.json`
- `xlarge_config.json` from `https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json`

Config observations:

| snapshot | architecture | block_sizes | block_repeats | decoder layers | d_model | heads | d_head | d_inner |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| small | FunnelModel | [4,4,4] | [1,1,1] | 2 | 768 | 12 | 64 | 3072 |
| small-base | FunnelBaseModel | [4,4,4] | [1,1,1] | 2 | 768 | 12 | 64 | 3072 |
| medium | FunnelModel | [6,3,3] | [1,2,2] | 2 | 768 | 12 | 64 | 3072 |
| intermediate | FunnelModel | [6,6,6] | [1,1,1] | 2 | 768 | 12 | 64 | 3072 |
| large | FunnelModel | [8,8,8] | [1,1,1] | 2 | 1024 | 16 | 64 | 4096 |
| xlarge | FunnelModel | [10,10,10] | [1,1,1] | 2 | 1024 | 16 | 64 | 4096 |

All sampled configs set:

- `vocab_size=30522`
- `max_position_embeddings=512`
- `type_vocab_size=3`
- `attention_type="relative_shift"`
- `rel_attn_type="factorized"` is present in the snapshots but is not read by the current in-library Funnel source.
- `pooling_type="mean"`
- `separate_cls=true`
- `truncate_seq=true`
- `pool_q_only=true`
- `hidden_act="gelu_new"`
- `layer_norm_eps=1e-9`

