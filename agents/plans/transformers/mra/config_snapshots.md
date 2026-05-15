# MRA Config Snapshots

Source basis:

- Transformers checkout: `transformers`
- Transformers commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Snapshot date: 2026-05-13

## Public checkpoint configs inspected

| Model id | HF repo sha | Accessible | Files observed | Tokenizer files |
|---|---:|---|---|---|
| `uw-madison/mra-base-512-4` | `48d9b4b087bff439ab901301ca3325799b8d8423` | yes | `.gitattributes`, `README.md`, `config.json`, `pytorch_model.bin` | not present, raw requests returned 404 |
| `uw-madison/mra-base-4096-8-d3` | `843dc007da36f2e7d361e7212fe1ab448b84cde6` | yes | `.gitattributes`, `README.md`, `config.json`, `pytorch_model.bin` | not present, raw requests returned 404 |

No gated/401 MRA checkpoint was encountered during this audit. The public model weight files responded to HEAD requests:

- `uw-madison/mra-base-512-4/pytorch_model.bin`: HTTP 200, `Content-Length=498858283`, ETag `57f1d3f23327733423fd5a941c4861f9430e8fd7c6b0cf71c2d4f6cdec839e5c`.
- `uw-madison/mra-base-4096-8-d3/pytorch_model.bin`: HTTP 200, `Content-Length=509897003`, ETag `8e59e505e22e013e4e8a16b5cbfa1ab3fedb7567461898d8fcc13873afdc3a5d`.

## Operator-significant config sweep

| Field | Source default | `mra-base-512-4` | `mra-base-4096-8-d3` | Tiny test config |
|---|---:|---:|---:|---:|
| architecture | source | `MraForMaskedLM` | `MraForMaskedLM` | test-created |
| `vocab_size` | 50265 | 50265 | 50265 | 99 |
| `hidden_size` | 768 | 768 | 768 | 16 |
| `num_hidden_layers` | 12 | 12 | 12 | 2 |
| `num_attention_heads` | 12 | 12 | 12 | 2 |
| `head_dim` | inferred 64 | config nested `model.head_dim=64` | config legacy `head_dim=64` | inferred 8, then padded to 32 in attention |
| `intermediate_size` | 3072 | 3072 | 3072 | 36 |
| `max_position_embeddings` | 512 | 512 | 4096 | 64 |
| `type_vocab_size` | 1 | 1 | 1 | 16 |
| `block_per_row` | 4 | 4 | 8 | default 4 |
| `approx_mode` | `full` | `full` | `full` | default `full` |
| `initial_prior_first_n_blocks` | 0 | 0 | 3 | default 0 |
| `initial_prior_diagonal_n_blocks` | 0 | 0 | 1 | default 0 |
| `torch_dtype` | unset in source | `float32` | `float32` | test runtime dtype |

Notes:

- Both public configs contain legacy/original-training metadata fields such as `dataset`, `from_cp`, `gpu_setting`, `model`, `dim`, `num_head`, `mixed_precision`, or `shared_weight`. The pinned in-library `configuration_mra.py` and `modeling_mra.py` do not read those fields for forward execution.
- The 4096 README says "sequence length 512" even though `config.json` and the integration test use `max_position_embeddings=4096`. Treat the config as authoritative for runtime shape.
- The tiny test config is not a checkpoint; it is included only as a debug/operator-shape fixture from `tests/models/mra/test_modeling_mra.py`.

