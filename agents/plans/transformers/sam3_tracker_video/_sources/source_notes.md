# sam3_tracker_video source notes

## Basis

- Repository: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family path: `src/transformers/models/sam3_tracker_video`
- Generated files state they are generated from `modular_sam3_tracker_video.py`; future upstream edits should inspect the modular file first, while the generated modeling file is useful because it contains the expanded inherited SAM2-video body.

## Local source inventory

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `configuration_sam3_tracker_video.py` | 15116 | `C8076287EDEFD8DDB6CDF2D76F7037AFDBDBD082FB55233719B126A2E595DED0` |
| `modeling_sam3_tracker_video.py` | 134699 | `6F1C8F5B00A67C324F0F3AFF4FAD3C5FECEF0C4AB3DC162EA3765524DBD49072` |
| `processing_sam3_tracker_video.py` | 37541 | `E090BBABB6972C1AF59147613E201B56FC7247B1ECFA8126E5D27EF4A0E668EA` |
| `modular_sam3_tracker_video.py` | 21204 | `CE28F7BA5716AFEA8BB8491DBD55D01288D0F5F69041EE139D2F52ADA92F62AB` |

## HF config access

- `https://huggingface.co/facebook/sam3/raw/main/config.json` returned HTTP 401 on 2026-05-13 because `facebook/sam3` is manually gated.
- `https://huggingface.co/facebook/sam3/raw/main/preprocessor_config.json` also returned HTTP 401.
- Public HF API metadata for `facebook/sam3` was accessible and reported: `model_type="sam3_video"`, architecture `Sam3VideoModel`, repo SHA `3c879f39826c281e95690f02c7821c4de09afae7`, `gated="manual"`, `library_name="transformers"`, `pipeline_tag="mask-generation"`, and safetensors metadata `F32: 859922360`.
- Because raw configs were gated, the report uses pinned source defaults for `sam3_tracker_video` and labels the production checkpoint config as inaccessible.

## High-signal line anchors

- Config defaults and sub-config wiring: `configuration_sam3_tracker_video.py:31`, `:56`, `:94`, `:204`, `:215`, `:252`, `:279`.
- Session/cache ABI: `modeling_sam3_tracker_video.py:57`, `:72`, `:109`, `:145`, `:240`, `:304`, `:324`, `:336`.
- Layout helper and sine position embedding: `modeling_sam3_tracker_video.py:349`, `:376`.
- Dense prompt/mask decoder attention: `modeling_sam3_tracker_video.py:444`, `:515`, `:1295`, `:1368`.
- Memory RoPE attention: `modeling_sam3_tracker_video.py:716`, `:758`, `:824`, `:897`, `:948`.
- Memory encoder/fuser/downsampler: `modeling_sam3_tracker_video.py:1005`, `:1057`, `:1074`, `:1105`.
- Prompt encoder and coordinate embeddings: `modeling_sam3_tracker_video.py:1164`, `:1189`, `:1216`.
- Main tracker model and image feature preparation: `modeling_sam3_tracker_video.py:1591`, `:1649`, `:1886`, `:1891`.
- Single-frame SAM path and mask/box postprocess inside model: `modeling_sam3_tracker_video.py:1968`, `:2037`, `:2056`, `:2070`, `:2078`, `:2089`, `:2119`.
- Stateful tracking memory selection/update: `modeling_sam3_tracker_video.py:2176`, `:2216`, `:2263`, `:2298`, `:2359`, `:2408`, `:2641`, `:2696`, `:2760`.
- Processor coordinate, box, mask, and video session handling: `processing_sam3_tracker_video.py:39`, `:58`, `:128`, `:151`, `:428`, `:463`, `:514`, `:568`, `:629`, `:738`.

## Source-derived shape notes

- Default image size is `1008`; default ViT patch size from composed `sam3` vision config is `14`, so tracker feature sizes are `[[288, 288], [144, 144], [72, 72]]`.
- Prompt encoder mask input size is `4 * image_size // patch_size = 288` for defaults.
- Memory attention hidden size is `256`, heads `1`, head dim `256`, layers `4`, FFN hidden `2048`.
- Memory encoder output channels are `64`; encoded `maskmem_features` are stored as bfloat16 after flattening to `[H*W, B_or_obj, 64]`.
- Object pointers start at hidden size `256`; when `mem_dim=64`, `_process_object_pointers` splits each pointer into 4 memory-width tokens.
