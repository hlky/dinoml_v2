# LFM2-VL source notes

Audit target: `lfm2_vl`

Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Primary local files inspected:

- `src/transformers/models/lfm2_vl/configuration_lfm2_vl.py`
- `src/transformers/models/lfm2_vl/modeling_lfm2_vl.py`
- `src/transformers/models/lfm2_vl/modular_lfm2_vl.py`
- `src/transformers/models/lfm2_vl/processing_lfm2_vl.py`
- `src/transformers/models/lfm2_vl/image_processing_lfm2_vl.py`
- `src/transformers/models/lfm2/configuration_lfm2.py`
- `src/transformers/models/lfm2/modeling_lfm2.py`
- `src/transformers/models/lfm2/modular_lfm2.py`
- `src/transformers/models/siglip2/configuration_siglip2.py`
- `src/transformers/models/siglip2/modeling_siglip2.py`
- `src/transformers/models/siglip2/modular_siglip2.py`
- `src/transformers/cache_utils.py`

Authoritative source note:

- `modeling_lfm2_vl.py`, `modeling_lfm2.py`, and `modeling_siglip2.py` are generated from their corresponding `modular_*.py` files. Future upstream edits should be checked against the modular files first, while DinoML import parity should match the generated modeling files used by Transformers.

Representative Hugging Face configs snapshotted:

- `LFM2-VL-450M_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-450M/raw/main/config.json`
- `LFM2-VL-1.6B_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-1.6B/raw/main/config.json`
- `LFM2-VL-3B_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-3B/raw/main/config.json`
- `LFM2-VL-450M_preprocessor_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-450M/raw/main/preprocessor_config.json`
- `LFM2-VL-1.6B_preprocessor_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-1.6B/raw/main/preprocessor_config.json`
- `LFM2-VL-3B_preprocessor_config.json` from `https://huggingface.co/LiquidAI/LFM2-VL-3B/raw/main/preprocessor_config.json`

Notable source facts:

- `Lfm2VlModel` composes `AutoModel.from_config(config.vision_config)` and `AutoModel.from_config(config.text_config)`, normally SigLIP2 vision plus LFM2 text.
- `pixel_values` are already patchified by the LFM2-VL image processor as `[num_image_tiles, max_num_patches, 3 * patch_size * patch_size]`.
- The image processor always calls `resize_and_split` inside `_preprocess`, even when the serialized preprocessor config has `do_resize=false`.
- Vision `spatial_shapes` are patch-grid `[height_patches, width_patches]`; `pixel_attention_mask` is rank 2 over flattened patches.
- Vision last hidden states are unpadded with `pixel_attention_mask.sum(dim=1)`, reshaped to `[1, H, W, C]`, pixel-unshuffled by `downsample_factor`, projected, flattened, concatenated across images, and inserted into token embeddings with `masked_scatter`.
- Processor placeholder expansion is stricter than arbitrary scatter: each `<image>` is replaced with row-major tile placeholders and optional thumbnail placeholders. Token counts are computed from tile size, patch size, resized image size, and downsample factor.
- LFM2 text is a hybrid decoder: `layer_types` selects `full_attention` or `conv` per layer. Conv layers use depthwise causal Conv1d plus static `conv_states`; attention layers use dynamic KV cache.
- Current official VL checkpoints use `bfloat16`, `image_token_id=396`, `downsample_factor=2`, `encoder_patch_size=16`, `tile_size=512`, `max_image_tokens=256`, `min_image_tokens=64`, and SigLIP2 `vision_use_head=false`.

Representative config sweep:

| Model | Text hidden | Text layers | Attention/conv layers | Text heads/KV heads | Text MLP | Vision hidden | Vision layers/heads | Thumbnail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LFM2-VL-450M | 1024 | 16 | 6 / 10 | 16 / 8 | 6656 | 768 | 12 / 12 | config false, preprocessor true |
| LFM2-VL-1.6B | 2048 | 16 | 6 / 10 | 32 / 8 | 12288 | 1152 | 26 / 16 | true |
| LFM2-VL-3B | 2048 | 30 | 8 / 22 | 32 / 8 | 10752 | 1152 | 27 / 16 | true |

Open gaps noticed while auditing:

- Serialized configs include legacy fields such as `image_token_index`, `num_heads`, `block_*`, `conv_dim`, `theta`, and `use_pos_enc`; the inspected native source mostly consumes normalized fields through `Lfm2VlConfig`, `Lfm2Config`, and `Siglip2VisionConfig`.
- The 450M model config has `use_thumbnail=false` while the preprocessor snapshot has `use_thumbnail=true`; DinoML should treat processor config as the preprocessing source of truth and reject mismatches unless the caller explicitly supplies processor kwargs.
- The generated docstrings for LFM2-VL describe `pixel_values` like NCHW images, but the actual SigLIP2/LFM2-VL path consumes patchified rank-3 values.
