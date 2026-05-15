# Qwen3-VL Audit Source Notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Recent commit: `b75feb2af6 fix(minicpmv4_6): skip invalid failing tests (#45836)`

Primary source family inspected:

- `src/transformers/models/qwen3_vl/configuration_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/processing_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/modular_qwen3_vl.py`
- `src/transformers/models/qwen3_vl/__init__.py`

Related source inspected for processor ABI:

- `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
- `src/transformers/models/auto/image_processing_auto.py`
- `src/transformers/models/auto/processing_auto.py`
- `src/transformers/models/auto/video_processing_auto.py`
- `src/transformers/models/auto/modeling_auto.py`

Representative Hugging Face snapshots saved in this directory:

- `Qwen__Qwen3-VL-4B-Thinking.config.json`
- `Qwen__Qwen3-VL-4B-Thinking-FP8.config.json`
- `Qwen__Qwen3-VL-32B-Thinking-FP8.config.json`
- `Qwen__Qwen3-VL-30B-A3B-Instruct.config.json`
- `Qwen__Qwen3-VL-235B-A22B-Instruct-FP8.config.json`
- Matching `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer_config.json`, and `generation_config.json` where public raw files were accessible.

Notes:

- `processor_config.json` returned 404 for the sampled public checkpoints. The processor class and image/video processor types are represented in the preprocessor snapshots instead.
- `Qwen3-VL-30B-A3B-*` and `Qwen3-VL-235B-A22B-*` use `model_type=qwen3_vl_moe` and the separate `src/transformers/models/qwen3_vl_moe` source family. They are included only as family variation evidence and require a dedicated MoE audit before DinoML admission.
- Auto image processing maps `qwen3_vl` to the Qwen2-VL image processor implementation; there is no local `image_processing_qwen3_vl.py` file in the inspected family directory.
