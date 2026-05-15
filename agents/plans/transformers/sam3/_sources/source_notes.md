# SAM3 Source Notes

Audit date: 2026-05-13

Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- `git status --short`: clean for the inspected checkout.

Pinned source snapshots copied from `transformers/src/transformers/models/sam3/`:

- `configuration_sam3.py`
- `modeling_sam3.py`
- `image_processing_sam3.py`
- `processing_sam3.py`
- `modular_sam3.py`
- `convert_sam3_to_hf.py`

Config snapshots:

- `tiny-random_sam3_config.json`: fetched from `https://huggingface.co/tiny-random/sam3/raw/main/config.json`.

Source gaps:

- `https://huggingface.co/facebook/sam3/raw/main/config.json` returned HTTP 401. The official checkpoint appears gated from this environment, so production checkpoint dimensions in `report.md` use source defaults from `configuration_sam3.py` and clearly label the open tiny-random config separately.
- Official `facebook/sam3` `preprocessor_config.json` and `processor_config.json` also returned HTTP 401. Processor behavior is therefore source-derived from `image_processing_sam3.py` and `processing_sam3.py`.
- Video/tracking behavior is not implemented in this `sam3` directory beyond detector loading shims and ignored tracker keys. The separate docs mention `Sam3VideoModel`/`Sam3VideoProcessor`, but those live outside this source directory and need a separate audit.

Important implementation anchors:

- `modeling_sam3.py` is generated from `modular_sam3.py`; future upstream edits should target `modular_sam3.py`.
- Image detector scope is `Sam3Model`, with stages: `Sam3VisionModel`, `CLIPTextModelWithProjection`, `Sam3GeometryEncoder`, `Sam3DetrEncoder`, `Sam3DetrDecoder`, `Sam3MaskDecoder`, and `Sam3DotProductScoring`.
- `Sam3Model._keys_to_ignore_on_load_unexpected` ignores `tracker_model.*` and `tracker_neck.*`, and the constructor unwraps `config.detector_config` when loading from a `sam3_video` config.
- Main source layout boundary: public image tensors are NCHW, but the ViT backbone internally reshapes patch tokens to NHWC for LayerNorm/window attention, then the neck and mask heads use NCHW feature maps.
