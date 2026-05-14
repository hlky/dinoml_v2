# GLM4V audit source notes

Local Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `src/transformers/models/glm4v`

Downloaded representative config snapshots:

- `zai-org_GLM-4.1V-9B-Thinking_config.json`: official native `model_type="glm4v"` production checkpoint.
- `zai-org_GLM-4.1V-9B-Thinking_preprocessor_config.json`: official image/processor snapshot for the production checkpoint.
- `tiny-random_glm-4v_config.json`: small native `model_type="glm4v"` fixture with reduced text and vision dimensions.
- `zai-org_GLM-4.5V_config.json`: official `model_type="glm4v_moe"` config. Processor-compatible in parts, but not covered by this native `glm4v` report.
- `zai-org_GLM-4.5V_preprocessor_config.json`: official GLM-4.5V processor snapshot.
- `zai-org_glm-4v-9b_config.json`: older `model_type="chatglm"` remote-code checkpoint. Not native `glm4v`; useful only as an out-of-scope variation trap.

404/unavailable:

- `zai-org/GLM-4.1V-9B-Thinking/raw/main/processor_config.json` returned 404.
- `zai-org/glm-4v-9b/raw/main/preprocessor_config.json` returned 404.
- `zai-org/GLM-4.5V/raw/main/processor_config.json` returned 404.

Important local line anchors:

- `configuration_glm4v.py`: config classes and token IDs.
- `modeling_glm4v.py`: generated from `modular_glm4v.py`; runtime source inspected for lowering behavior.
- `modeling_glm4v.py:83`: vision Conv3d patch embed.
- `modeling_glm4v.py:136`: interpolated vision absolute position embedding via `grid_sample`.
- `modeling_glm4v.py:269`: vision packed variable-length attention using `cu_seqlens` when FlashAttention is requested.
- `modeling_glm4v.py:386`: text M-RoPE implementation.
- `modeling_glm4v.py:510`: text causal GQA attention.
- `modeling_glm4v.py:719`: vision spatial downsample Conv2d.
- `modeling_glm4v.py:1007`: multimodal RoPE position index construction.
- `modeling_glm4v.py:1153`: placeholder mask and image/video feature count validation.
- `modeling_glm4v.py:1278`: `masked_scatter` stitch for image features.
- `modeling_glm4v.py:1285`: `masked_scatter` stitch for video features.
- `processing_glm4v.py:46`: processor token ABI.
- `processing_glm4v.py:245`: `mm_token_type_ids` construction.
- `image_processing_glm4v.py:53`: resize policy and divisibility.
- `video_processing_glm4v.py:96`: frame sampling policy.
