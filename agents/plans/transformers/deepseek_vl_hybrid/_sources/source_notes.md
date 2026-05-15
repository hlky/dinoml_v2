# Source notes: deepseek_vl_hybrid

Local source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `transformers/src/transformers/models/deepseek_vl_hybrid`
- Generated files state that `modular_deepseek_vl_hybrid.py` is authoritative for source edits.

Files inspected:

- `configuration_deepseek_vl_hybrid.py`
- `modeling_deepseek_vl_hybrid.py`
- `modular_deepseek_vl_hybrid.py`
- `processing_deepseek_vl_hybrid.py`
- `image_processing_deepseek_vl_hybrid.py`
- `image_processing_pil_deepseek_vl_hybrid.py`
- `convert_deepseek_vl_hybrid_weights_to_hf.py`
- Related composed families: `llama/modeling_llama.py`, `siglip/modeling_siglip.py`, `sam/modeling_sam.py`, `sam/configuration_sam.py`, `siglip/configuration_siglip.py`
- Local tests: `tests/models/deepseek_vl_hybrid/test_modeling_deepseek_vl_hybrid.py`, `test_processing_deepseek_vl_hybrid.py`

HF configs inspected through Hugging Face web/raw pages:

- `deepseek-community/deepseek-vl-7b-base`, `config.json`, `processor_config.json`, `preprocessor_config.json`, `tokenizer_config.json`
- `deepseek-community/deepseek-vl-7b-chat`, `config.json`, `preprocessor_config.json`
- `deepseek-community/deepseek-vl-1.3b-base`, `config.json`, labeled as non-hybrid contrast because `model_type` is `deepseek_vl`, not `deepseek_vl_hybrid`.

Important source observations:

- `DeepseekVLHybridConfig` composes `text_config` through `AutoConfig` defaulting to LLaMA, `vision_config` defaulting to SigLIP vision, and `high_res_vision_config` defaulting to SAM vision.
- `DeepseekVLHybridModel` constructs `AutoModel.from_config` for both vision branches and for the text model.
- Low-res path: SigLIP vision consumes `pixel_values` `[B,3,384,384]`, patch size 16, returns 576 sequence tokens of width 1024 for official 7B configs.
- High-res path: SAM vision consumes `high_res_pixel_values` `[B,3,1024,1024]`, patch size 16, returns an NCHW neck map `[B,256,64,64]`; DeepSeek applies extra projection convolutions and reshapes to 576 sequence tokens of width 1024.
- The aligner projects low-res 1024 -> 2048 and high-res 1024 -> 2048, concatenates to 4096, applies GELU, then projects 4096 -> 4096 for official 7B configs.
- Image embeddings are inserted with `inputs_embeds.masked_scatter(image_attention_mask, image_features)`. The processor expands each `<image_placeholder>` string to 576 repeated placeholders.
- Generation passes `pixel_values` and `high_res_pixel_values` only on the first generation iteration, or whenever `use_cache=False`.
- Official 7B base/chat configs share the same architecture dimensions.
