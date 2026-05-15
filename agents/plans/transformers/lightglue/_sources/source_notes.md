# LightGlue Source Notes

## Local source

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Commit date/subject from local git: `2026-05-11 12:26:00 +0000`, `fix(minicpmv4_6): skip invalid failing tests (#45836)`

Files inspected:

- `src/transformers/models/lightglue/configuration_lightglue.py`
- `src/transformers/models/lightglue/modeling_lightglue.py`
- `src/transformers/models/lightglue/modular_lightglue.py`
- `src/transformers/models/lightglue/image_processing_lightglue.py`
- `src/transformers/models/lightglue/image_processing_pil_lightglue.py`
- `src/transformers/models/superpoint/configuration_superpoint.py`
- `src/transformers/models/superpoint/modeling_superpoint.py`

Generated-source note:

- `configuration_lightglue.py`, `modeling_lightglue.py`, and `image_processing_*_lightglue.py` are generated from `modular_lightglue.py`.
- For DinoML parity, the generated files are the exact runtime/import source. For future Transformers source edits, `modular_lightglue.py` is authoritative.

## Key local source anchors

- `LightGlueConfig`: defaults and validation at `configuration_lightglue.py:59-96`.
- Positional encoder and LightGlue-specific RoPE helpers at `modeling_lightglue.py:83-133`.
- Attention projection and eager attention math at `modeling_lightglue.py:148-240`.
- Layer path with self-attention, pair flip, cross-attention, and MLPs at `modeling_lightglue.py:262-342`.
- Match assignment and double log-softmax at `modeling_lightglue.py:345-398`.
- Token confidence, match extraction, and keypoint normalization at `modeling_lightglue.py:401-480`.
- Model init and detector construction at `modeling_lightglue.py:488-530`.
- Early stop and pruning helpers at `modeling_lightglue.py:533-615`.
- Pair matching loop and forward ABI at `modeling_lightglue.py:686-913`.
- Processor pair validation/preprocess defaults at `image_processing_lightglue.py:62-189`.
- Processor postprocess ABI at `image_processing_lightglue.py:191-250`.
- SuperPoint NMS at `modeling_superpoint.py:55-70`.
- SuperPoint encoder/decoder/source dynamic keypoint output at `modeling_superpoint.py:147-185`, `193-259`, `276-319`, and `417-465`.

## Hub config snapshots

Fetched with raw Hub URLs on 2026-05-13. These are config facts, not model-card claims.

### ETH-CVG/lightglue_superpoint

- URL: https://huggingface.co/ETH-CVG/lightglue_superpoint
- `architectures`: `["LightGlueForKeypointMatching"]`
- `model_type`: `lightglue`
- `descriptor_dim`: 256
- `hidden_size`: 256
- `intermediate_size`: 512
- `num_hidden_layers`: 9
- `num_attention_heads`: 4
- `num_key_value_heads`: 4
- `attention_bias`: true
- `attention_dropout`: 0.0
- `hidden_act`: `gelu`
- `depth_confidence`: 0.95
- `width_confidence`: 0.99
- `filter_threshold`: 0.1
- `torch_dtype`: `float32`
- Nested detector: `model_type=superpoint`, `descriptor_decoder_dim=256`, `encoder_hidden_sizes=[64,64,128,128]`, `keypoint_decoder_dim=65`, `keypoint_threshold=0.005`, `max_keypoints=-1`, `nms_radius=4`, `border_removal_distance=4`.
- Preprocessor: `LightGlueImageProcessor`, resize 480x640, rescale 1/255, grayscale true.

### ETH-CVG/lightglue_disk

- URL: https://huggingface.co/ETH-CVG/lightglue_disk
- `architectures`: `["LightGlueForKeypointMatching"]`
- `model_type`: `lightglue`
- LightGlue dimensions same as primary: D=256, L=9, heads=4, KV heads=4.
- `trust_remote_code`: true
- Nested detector: `model_type=disk`, `auto_map` for `configuration_disk.DiskConfig` and `modeling_disk.DiskForKeypointDetection`, `descriptor_decoder_dim=128`, `max_num_keypoints=null`, `pad_if_not_divisible=true`, `weights=depth`.
- Preprocessor: resize 480x640, rescale 1/255, grayscale false.
- Audit conclusion: not native to this in-library LightGlue/SuperPoint report. Requires separate DISK/remote-code detector audit and a 128->256 input projection path.

### stevenbucaille/lightglue

- URL: https://huggingface.co/stevenbucaille/lightglue
- Historical config fields include `attention_probs_dropout_prob`, `num_heads`, `num_layers`, and `rotary_value`.
- Current inspected source does not use several of these historical names for topology; it uses `num_hidden_layers`, `num_attention_heads`, `attention_dropout`, and `attention_bias`.
- `keypoint_detector_config` only sets `model_type=superpoint`, so current source defaults fill the SuperPoint dimensions.
- Preprocessor omits `do_grayscale`; current source default is true.

### stevenbucaille/lightglue_superpoint

- URL: https://huggingface.co/stevenbucaille/lightglue_superpoint
- Native-looking LightGlue/SuperPoint config with D=256, L=9, heads=4, KV heads=4.
- Contains historical fields `attention_probs_dropout_prob` and `rotary_value`; current source basis should ignore them unless weight loading code proves a compatibility need.
- Preprocessor grayscale true.

### stevenbucaille/lightglue_minima

- URL: https://huggingface.co/stevenbucaille/lightglue_minima
- Config is structurally the same native SuperPoint-backed LightGlue shape as `ETH-CVG/lightglue_superpoint`.
- `trust_remote_code`: false.
- Name implies a variant, but the inspected in-library config does not expose a distinct topology.

## Validation performed

- Read local source and configs only.
- Queried Hugging Face raw `config.json` and `preprocessor_config.json` for representative IDs.
- No Python imports, model execution, DinoML code edits, tests, commits, or shared-plan edits.
