# Qwen3.5 Source Notes

## Scope

- DinoML audit target: `qwen3_5`
- Transformers checkout: `transformers`
- Inspected commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Local source directory: `transformers/src/transformers/models/qwen3_5`
- Report output: `H:/dinoml_v2/agents/plans/transformers/qwen3_5/report.md`

## Local files inspected

- `configuration_qwen3_5.py`
- `modeling_qwen3_5.py`
- `modular_qwen3_5.py`
- `tokenization_qwen3_5.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/models/qwen3_vl/processing_qwen3_vl.py`
- `transformers/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py`

The generated `configuration_qwen3_5.py` and `modeling_qwen3_5.py` both state that they are generated from `modular_qwen3_5.py`. Future upstream source edits should be checked against the modular file first, while DinoML parity should use the generated modeling file because that is the importable runtime implementation.

## Hugging Face configs fetched

Fetched by direct `config.json` and `preprocessor_config.json` raw URLs on May 13, 2026:

- `https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.5-27B-FP8/raw/main/config.json`
- Matching `preprocessor_config.json` files for the same five repos.

All five processor configs were available as `preprocessor_config.json`; `processor_config.json` returned 404 for all five repos checked. No gated/401 model config was encountered in this sweep.

## Processor source gap

The fetched preprocessor configs name:

- `processor_class`: `Qwen3VLProcessor`
- `image_processor_type`: `Qwen2VLImageProcessorFast`

There is no Qwen3.5-specific processing file in `models/qwen3_5`. The report therefore treats the model-owned neural graph as Qwen3.5, while image/video packing details are sourced from Qwen3-VL processor code and Qwen2-VL image processor lineage as referenced by the checkpoint metadata.

## High-risk source facts

- Text decoder layers are hybrid by default: three `linear_attention` layers followed by one `full_attention` layer, repeated by `full_attention_interval=4`.
- Full attention uses GQA and a gated query/output projection: `q_proj` outputs `num_attention_heads * head_dim * 2`, split into query and gate.
- `head_dim` is explicit and can make `num_attention_heads * head_dim != hidden_size`; this happens in the 0.8B, 4B, and 27B configs checked.
- Linear-attention layers use Gated DeltaNet with depthwise causal Conv1d state plus a recurrent matrix state, not a KV cache.
- Text RoPE is partial RoPE with `partial_rotary_factor=0.25` in configs checked and M-RoPE interleaving for multimodal position IDs.
- Multimodal embedding insertion uses `masked_scatter`, but the processor expands placeholder tokens to a count derived from `grid_thw.prod() // merge_size**2`.
- The FP8 config is source-coupled metadata from the checkpoint, not a modeling-file operator; it lists many modules excluded from FP8 conversion.
