# PaddleOCR-VL Source Snapshots

Local source basis:

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/paddleocr_vl`

Downloaded Hugging Face snapshots:

- `hf_main_*`: `PaddlePaddle/PaddleOCR-VL` at `main`.
- `hf_15_main_*`: `PaddlePaddle/PaddleOCR-VL-1.5` at `main`.
- `hf_90dbf489_*`: historical `PaddlePaddle/PaddleOCR-VL` config/preprocessor snapshot.
- `hf_7a811607_config.json` and `hf_f1e186d3_config.json`: older historical snapshots that only contained a minimal PaddleOCR pipeline-style `{"Global": {"model_name": "PaddleOCR-VL"}}` payload, not the native Transformers `paddleocr_vl` config.

The report treats the in-library Transformers implementation as authoritative for DinoML graph/runtime work and labels PaddleOCR pipeline behavior separately.
