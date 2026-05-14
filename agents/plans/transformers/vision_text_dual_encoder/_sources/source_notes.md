# vision_text_dual_encoder source notes

Transformers checkout: `X:/H/transformers`

Transformers commit inspected:

```text
b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Local source files inspected:

- `X:/H/transformers/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py`
- `X:/H/transformers/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py`
- `X:/H/transformers/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py`
- `X:/H/transformers/src/transformers/processing_utils.py` for the inherited processor call/merge ABI.

Representative Hugging Face config snapshots saved in this directory:

- `hf-internal-testing_tiny-random-VisionTextDualEncoderModel-vit-bert_config.json`
  - Source URL: https://huggingface.co/hf-internal-testing/tiny-random-VisionTextDualEncoderModel-vit-bert/raw/main/config.json
- `hf-internal-testing_tiny-random-VisionTextDualEncoderModel-vit-bert_preprocessor_config.json`
  - Source URL: https://huggingface.co/hf-internal-testing/tiny-random-VisionTextDualEncoderModel-vit-bert/raw/main/preprocessor_config.json
- `ljnlonoljpiljm_webssl-mae700m-full2b-224-bert-base-uncased_config.json`
  - Source URL: https://huggingface.co/ljnlonoljpiljm/webssl-mae700m-full2b-224-bert-base-uncased/raw/main/config.json
- `koclip_koclip-base-pt_config.json`
  - Source URL: https://huggingface.co/koclip/koclip-base-pt/raw/main/config.json
- `flavour_vtde-dinov2-small-multilingual-e5-small_config.json`
  - Source URL: https://huggingface.co/flavour/vtde-dinov2-small-multilingual-e5-small/raw/main/config.json
- `ljnlonoljpiljm_CLIP-ViT-H-14-laion2B-s32B-b79K-384-xlm-roberta-large-tv_config.json`
  - Source URL: https://huggingface.co/ljnlonoljpiljm/CLIP-ViT-H-14-laion2B-s32B-b79K-384-xlm-roberta-large-tv/raw/main/config.json
- `fabnem_UltraSoundCLIP_checkpoint-1650_config.json`
  - Source URL: https://huggingface.co/fabnem/UltraSoundCLIP/raw/main/checkpoint-1650/config.json

No remote-code files were required for the native library source audit. Config facts in `report.md` are from these snapshots unless labeled as source defaults or source behavior.
