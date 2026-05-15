# EXAONE 4.5 source notes

## Source basis

- Transformers checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary family: `src/transformers/models/exaone4_5`
- Text dependency: `src/transformers/models/exaone4`
- Report spec: `H:/dinoml_v2/agents/plans/transformers/PROMPT.md`

## Local snapshots

The following files were copied from the pinned checkout for audit stability:

- `configuration_exaone4_5.py`
- `modeling_exaone4_5.py`
- `modular_exaone4_5.py`
- `processing_exaone4_5.py`
- `configuration_exaone4_text_dependency.py`
- `modeling_exaone4_text_dependency.py`

`configuration_exaone4_5.py`, `modeling_exaone4_5.py`, and `processing_exaone4_5.py` are generated from `modular_exaone4_5.py`; future upstream source edits should be checked against the modular file first.

## Hugging Face configs fetched

- `LGAI-EXAONE/EXAONE-4.5-33B`: `config.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json`
- `LGAI-EXAONE/EXAONE-4.5-33B-FP8`: `config.json`, `generation_config.json`, `processor_config.json`
- `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`: `config.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json`

Video-specific `video_preprocessor_config.json` returned 404 for all three repos. The combined `processor_config.json` contains the video processor block and is the source used for video preprocessing notes.

## Checkpoint sweep notes

- Dense 33B and AWQ configs have `num_hidden_layers=64` and `len(text_config.layer_types)=64`.
- FP8 config has `num_hidden_layers=64` and `len(text_config.layer_types)=65`; the extra trailing `sliding_attention` entry is unused by the source loop over `range(num_hidden_layers)` and should be rejected or normalized by DinoML config admission.
- Dense 33B config uses historical text `model_type="exaone4_5_text"` and `rope_scaling`; the wrapper remaps the model type to `exaone4`. The current text source reads `rope_parameters`, while newer FP8/AWQ configs use that field directly.
- FP8 and AWQ quantization metadata comes from `quantization_config` only. The inspected modeling source does not implement compressed-tensors kernels; those variants require a loader/provider admission path or dense fallback.
