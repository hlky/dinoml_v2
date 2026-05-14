# SwinV2 Source Notes

Source checkout:

- `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Commit date: `2026-05-11 12:26:00 +0000`

Local source files inspected:

- `src/transformers/models/swinv2/configuration_swinv2.py`
- `src/transformers/models/swinv2/modeling_swinv2.py`
- `src/transformers/models/swinv2/convert_swinv2_timm_to_pytorch.py`
- `src/transformers/models/auto/image_processing_auto.py`
- `src/transformers/models/vit/image_processing_vit.py`
- `tests/models/swinv2/test_modeling_swinv2.py`

Representative Hugging Face config snapshots fetched:

- `microsoft__swinv2-tiny-patch4-window8-256--config.json`
- `microsoft__swinv2-tiny-patch4-window8-256--preprocessor_config.json`
- `microsoft__swinv2-small-patch4-window8-256--config.json`
- `microsoft__swinv2-small-patch4-window8-256--preprocessor_config.json`
- `microsoft__swinv2-base-patch4-window12-192-22k--config.json`
- `microsoft__swinv2-base-patch4-window12-192-22k--preprocessor_config.json`
- `microsoft__swinv2-base-patch4-window12to16-192to256-22kto1k-ft--config.json`
- `microsoft__swinv2-base-patch4-window12to16-192to256-22kto1k-ft--preprocessor_config.json`
- `microsoft__swinv2-large-patch4-window12to16-192to256-22kto1k-ft--config.json`
- `microsoft__swinv2-large-patch4-window12to16-192to256-22kto1k-ft--preprocessor_config.json`

Fetch URLs used:

- `https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256/resolve/main/config.json`
- `https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256/resolve/main/preprocessor_config.json`
- `https://huggingface.co/microsoft/swinv2-small-patch4-window8-256/resolve/main/config.json`
- `https://huggingface.co/microsoft/swinv2-small-patch4-window8-256/resolve/main/preprocessor_config.json`
- `https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k/resolve/main/config.json`
- `https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k/resolve/main/preprocessor_config.json`
- `https://huggingface.co/microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft/resolve/main/config.json`
- `https://huggingface.co/microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft/resolve/main/preprocessor_config.json`
- `https://huggingface.co/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft/resolve/main/config.json`
- `https://huggingface.co/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft/resolve/main/preprocessor_config.json`

Notable source observations:

- Auto image processing maps `swinv2` to `ViTImageProcessor` / `ViTImageProcessorPil`.
- Downloaded preprocessor configs are legacy `feature_extractor_type: ViTFeatureExtractor` with resize and ImageNet normalization.
- Native SwinV2 attention uses cosine attention: L2-normalize Q and K, matmul, multiply by clamped exponential `logit_scale`.
- Native SwinV2 attention uses a continuous relative position bias MLP over log-spaced normalized coordinate tables, not a learned discrete bias table.
- Shifted-window masks are created on padded window grids with mask values `0` and `-100.0`.
- At this inspected commit, `Swinv2SelfAttention.forward` adds the same attention mask twice in the shifted-window path after one reshape. This is source behavior to preserve or consciously test around.
- Native Q/K/V are separate Linear modules; query and value honor `config.qkv_bias`, key is constructed with `bias=False`.
- The timm conversion helper splits packed `qkv` weights in Q,K,V order and also writes a `key.bias` entry from packed qkv bias, even though the native key projection has no bias in this source.
