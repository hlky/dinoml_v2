# DeepSeek-VL Source Notes

Local audit date: 2026-05-13

## Local source

- DinoML workspace commit: `e8303f2e7ef928cce59ff8a7a4da0e48f1ea2460`
- Transformers checkout commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Primary family source directory: `transformers/src/transformers/models/deepseek_vl`
- Related variation inspected for checkpoint sweep only: `transformers/src/transformers/models/deepseek_vl_hybrid`

## Downloaded Hugging Face snapshots

Fetched with direct `https://huggingface.co/<model>/resolve/main/<file>` URLs.

- `deepseek-community/deepseek-vl-1.3b-chat`
  - `config.json`
  - `preprocessor_config.json`
  - `processor_config.json`
  - `tokenizer_config.json`
  - `generation_config.json`
- `deepseek-community/deepseek-vl-7b-chat`
  - `config.json`
  - `preprocessor_config.json`
  - `processor_config.json`
  - `tokenizer_config.json`
  - `generation_config.json`

## Primary source facts

- Plain `deepseek_vl` is a SigLIP vision encoder plus a two-layer GELU aligner plus a Llama causal LM.
- The processor expands each textual `<image_placeholder>` into 576 placeholder tokens.
- Plain `deepseek_vl` preprocessing emits one image tensor field, `pixel_values`, in NCHW layout.
- Plain `deepseek_vl` has no image tiling. It resizes the longest side to 384, pads to square, rescales, and normalizes.
- The 1.3B config uses `model_type: deepseek_vl` and `DeepseekVLForConditionalGeneration`.
- The 7B config uses `model_type: deepseek_vl_hybrid` and `DeepseekVLHybridForConditionalGeneration`; it is not implemented by the plain `deepseek_vl` source family. It adds `high_res_pixel_values`, a SAM vision branch, high-res resizing to 1024, and a hybrid aligner.
