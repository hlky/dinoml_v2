# Parakeet Source Notes

Audit date: 2026-05-13

Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family path: `src/transformers/models/parakeet`

Primary source files inspected:

- `configuration_parakeet.py`
- `modeling_parakeet.py`
- `modular_parakeet.py`
- `feature_extraction_parakeet.py`
- `processing_parakeet.py`
- `tokenization_parakeet.py`
- `convert_nemo_to_hf.py`

Generated-source note:

- `modeling_parakeet.py` is generated from `modular_parakeet.py`; future
  source edits should target the modular file. The generated file is still the
  runtime source basis for this audit because it is what Transformers imports.

Representative HF config sweep:

| Model id | Fetch result | Important fields |
| --- | --- | --- |
| `nvidia/parakeet-ctc-0.6b` | Open `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | `model_type=parakeet_ctc`, `ParakeetForCTC`, 24 encoder layers, hidden 1024, 8 heads, 80 mel bins, vocab 1025, pad/blank 1024 |
| `nvidia/parakeet-ctc-1.1b` | Open `config.json`, `preprocessor_config.json`, `tokenizer_config.json` | Same CTC topology as 0.6B except 42 encoder layers |
| `nvidia/parakeet-tdt-0.6b-v3` | Open `config.json`, `processor_config.json`, `generation_config.json` | `model_type=parakeet_tdt`, `ParakeetForTDT`, 24 encoder layers, 128 mel bins, attention/convolution bias false, scale input false, vocab 8193, decoder start 8192 |
| `nvidia/parakeet-rnnt-1.1b` | HF API visible, raw `config.json` 404 | NeMo-library repo; no native Transformers config at main during audit |
| `nvidia/parakeet-rnnt-0.6b` | HF API/search visible, raw `config.json` 404 | NeMo-library repo; no native Transformers config at main during audit |
| `nvidia/parakeet-tdt-0.6b-v2` | HF API visible, raw `config.json` 404 | NeMo-library repo; no native Transformers config at main during audit |
| `nvidia/parakeet-tdt-1.1b` | HF API/search visible, raw `config.json` 404 | NeMo-library repo; no native Transformers config at main during audit |

Key source-line anchors from the pinned checkout:

- Config defaults: `configuration_parakeet.py:24`, `configuration_parakeet.py:66`, `configuration_parakeet.py:95`, `configuration_parakeet.py:123`.
- Audio frontend: `feature_extraction_parakeet.py:36`, `feature_extraction_parakeet.py:99`, `feature_extraction_parakeet.py:250`, `feature_extraction_parakeet.py:261`, `feature_extraction_parakeet.py:266`.
- Processor/tokenizer ABI: `processing_parakeet.py:24`, `processing_parakeet.py:41`, `tokenization_parakeet.py:20`, `tokenization_parakeet.py:28`.
- Encoder pieces: `modeling_parakeet.py:60`, `modeling_parakeet.py:110`, `modeling_parakeet.py:125`, `modeling_parakeet.py:268`, `modeling_parakeet.py:366`, `modeling_parakeet.py:435`, `modeling_parakeet.py:558`.
- CTC head and greedy generation: `modeling_parakeet.py:685`, `modeling_parakeet.py:692`, `modeling_parakeet.py:732`, `modeling_parakeet.py:770`, `modeling_parakeet.py:806`.

External links used:

- Transformers source tree at commit:
  <https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/parakeet>
- CTC 1.1B config:
  <https://huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json>
- CTC 0.6B config:
  <https://huggingface.co/nvidia/parakeet-ctc-0.6b/raw/main/config.json>
- TDT 0.6B v3 config, used only as an unsupported-variant trap:
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json>
- TDT 0.6B v3 processor/generation configs:
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/processor_config.json>
  and
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/generation_config.json>
