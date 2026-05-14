# MiniCPM-V 4.6 Source Notes

Local Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family path: `X:/H/transformers/src/transformers/models/minicpmv4_6`

Primary local files inspected:

- `configuration_minicpmv4_6.py`
- `modeling_minicpmv4_6.py`
- `modular_minicpmv4_6.py`
- `processing_minicpmv4_6.py`
- `image_processing_minicpmv4_6.py`
- `image_processing_pil_minicpmv4_6.py`
- `video_processing_minicpmv4_6.py`

Nested text implementation inspected:

- `X:/H/transformers/src/transformers/models/qwen3_5/configuration_qwen3_5.py`
- `X:/H/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py`
- `X:/H/transformers/src/transformers/cache_utils.py`

Hugging Face snapshots saved in this directory:

- `openbmb_MiniCPM-V-4_6_config.json`
- `openbmb_MiniCPM-V-4_6_preprocessor_config.json`
- `openbmb_MiniCPM-V-4_6_tokenizer_config.json`
- `openbmb_MiniCPM-V-4_config.json`
- `openbmb_MiniCPM-V_config.json`

HF model API snapshot observed:

- Model id: `openbmb/MiniCPM-V-4.6`
- Repo sha: `c83e202c69261e37ab3df63177047f36d6841931`
- Last modified: `2026-05-13T11:33:16.000Z`
- Gated: `false`
- Safetensors parameter metadata: `BF16: 1300428016`

Scope note:

- `openbmb/MiniCPM-V-4.6` and `openbmb/MiniCPM-V-4_6` both resolved to the same `config.json` contents during this audit.
- Older `openbmb/MiniCPM-V` and `openbmb/MiniCPM-V-4` use remote-code `model_type=minicpmv`, not the in-library `minicpmv4_6` source. They are included only as historical variation traps, not as requirements for this report.
