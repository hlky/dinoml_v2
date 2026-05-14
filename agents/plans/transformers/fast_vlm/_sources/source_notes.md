# fast_vlm source notes

Audit date: 2026-05-13.

Local Transformers checkout: `X:/H/transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

FastVLM files inspected:

- `src/transformers/models/fast_vlm/configuration_fast_vlm.py`
- `src/transformers/models/fast_vlm/modeling_fast_vlm.py`
- `src/transformers/models/fast_vlm/modular_fast_vlm.py`
- `src/transformers/models/fast_vlm/convert_fastvlm_weights_to_hf.py`
- `tests/models/fast_vlm/test_modeling_fast_vlm.py`
- `docs/source/en/model_doc/fast_vlm.md`

Composed source inspected:

- `src/transformers/models/timm_wrapper/configuration_timm_wrapper.py`
- `src/transformers/models/timm_wrapper/modeling_timm_wrapper.py`
- `src/transformers/models/qwen2/configuration_qwen2.py`
- `src/transformers/models/qwen2/modeling_qwen2.py`
- Local installed timm source: `C:/Users/user/AppData/Local/Programs/Python/Python312/Lib/site-packages/timm/models/fastvit.py`

HF snapshots fetched:

- `KamilaMila/FastVLM-0.5B`: `config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `KamilaMila/FastVLM-1.5B`: same files
- `KamilaMila/FastVLM-7B`: same files

Important observations:

- `modeling_fast_vlm.py` and `configuration_fast_vlm.py` are generated from `modular_fast_vlm.py`.
- The FastVLM-owned neural work is the image feature flatten/permute, two-layer projector, image-placeholder mask, `masked_scatter`, and wrapper LM head.
- The vision tower is `AutoModel.from_config(config.vision_config)` and the public checkpoints use `timm_wrapper` with `architecture="fastvit_mci3"` and `model_args={"inference_mode": true}`.
- `TimmWrapperModel.forward` casts `pixel_values` to model device/dtype, calls `timm_model.forward_features(pixel_values)`, then optionally `forward_head`. FastVLM consumes `last_hidden_state`, not the pooled CLIP head.
- For public checkpoints, `FastVlmModel.get_image_features` expects `last_hidden_state` as NCHW and converts `[B, C, H, W] -> [B, H*W, C]`.
- Public checkpoints report `image_seq_length=256`, processor `patch_size=64`, and image processor crop/resize to 1024x1024, matching a 16x16 image token grid.
- Text decoder is Qwen2 causal LM: RMSNorm, biased Q/K/V, biasless O, GQA, RoPE, SwiGLU, dynamic cache, optional sliding-window support in source but not enabled in the inspected FastVLM configs.
- `prepare_inputs_for_generation` forwards `pixel_values` only on first generation iteration or when cache is disabled.
