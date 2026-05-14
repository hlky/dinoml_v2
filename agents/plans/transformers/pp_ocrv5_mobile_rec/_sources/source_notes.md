# pp_ocrv5_mobile_rec source notes

Audit date: 2026-05-13

Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source files inspected:

- `src/transformers/models/pp_ocrv5_mobile_rec/configuration_pp_ocrv5_mobile_rec.py`
- `src/transformers/models/pp_ocrv5_mobile_rec/modeling_pp_ocrv5_mobile_rec.py`
- `src/transformers/models/pp_ocrv5_mobile_rec/modular_pp_ocrv5_mobile_rec.py`
- `src/transformers/models/pp_ocrv5_server_rec/image_processing_pp_ocrv5_server_rec.py`
- `src/transformers/models/pp_ocrv5_server_rec/configuration_pp_ocrv5_server_rec.py`
- `src/transformers/models/pp_ocrv5_server_rec/modeling_pp_ocrv5_server_rec.py`
- `src/transformers/models/pp_lcnet_v3/configuration_pp_lcnet_v3.py`
- `src/transformers/models/pp_lcnet_v3/modeling_pp_lcnet_v3.py`

Representative remote configs saved:

- `PP-OCRv5_mobile_rec_safetensors.config.json`
- `PP-OCRv5_mobile_rec_safetensors.preprocessor_config.json`
- `en_PP-OCRv5_mobile_rec.paddle_config.json`
- `cyrillic_PP-OCRv5_mobile_rec.paddle_config.json`
- `latin_PP-OCRv5_mobile_rec.paddle_config.json`

Remote source basis:

- `https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_safetensors`
- `https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec`
- `https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec`
- `https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec`

Important distinction:

- `PP-OCRv5_mobile_rec_safetensors` is the native Transformers checkpoint basis and has `model_type: pp_ocrv5_mobile_rec`.
- The language-specific `en`, `cyrillic`, and `latin` repos expose Paddle inference configs, not native Transformers configs. They are useful for processor/postprocess variation and dynamic-shape hints, but should not be treated as direct Transformers config snapshots.

Config sweep summary:

| Repo/config | Type | Output classes / character entries | Image ABI hints |
| --- | --- | ---: | --- |
| `PP-OCRv5_mobile_rec_safetensors.config.json` | Transformers | `head_out_channels=18385` | paired preprocessor has 18385-character list |
| `PP-OCRv5_mobile_rec_safetensors.preprocessor_config.json` | Transformers preprocessor | 18385 entries, first entry `blank` | resize height 48, pad width 320, max width 3200 |
| `en_PP-OCRv5_mobile_rec.paddle_config.json` | Paddle inference | 436 dictionary entries | TRT shapes `[1,3,48,160]`, `[1,3,48,320]`, `[8,3,48,3200]` |
| `cyrillic_PP-OCRv5_mobile_rec.paddle_config.json` | Paddle inference | 850 dictionary entries | same TRT shape hints |
| `latin_PP-OCRv5_mobile_rec.paddle_config.json` | Paddle inference | 836 dictionary entries | same TRT shape hints |

