# sam3_tracker source notes

Source basis:
- Local Transformers checkout: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Primary target directory: `src/transformers/models/sam3_tracker`.
- Generated files state that `modular_sam3_tracker.py` is authoritative for future upstream source edits; `configuration_sam3_tracker.py`, `modeling_sam3_tracker.py`, and `processing_sam3_tracker.py` are generated.
- Composed vision backbone source: `src/transformers/models/sam3/{configuration_sam3.py,modeling_sam3.py}` because `Sam3TrackerConfig` defaults `vision_config.model_type` to `sam3_vision_model`.
- Tracking/session state boundary source: `src/transformers/models/sam3_tracker_video/{configuration_sam3_tracker_video.py,modeling_sam3_tracker_video.py}`. This is not the primary report target, but it owns the persistent object/video memory ABI that tracker-only configs ignore or partially share.
- Postprocess source: `src/transformers/models/sam2/image_processing_sam2.py`, inherited through `Sam3ImageProcessor`/`Sam3TrackerProcessor`.

HF config access:
- `facebook/sam3` is a gated/manual HF repo. The HF model API was readable and reported `model_type: sam3_video`, `architectures: ["Sam3VideoModel"]`, `gated: "manual"`, and siblings including `config.json`, but direct raw `config.json` and `processor_config.json` returned 401.
- `danelcsb/sam3_tracker.1_hiera_tiny` returned 401 through the HF model API during this audit.
- Open mirror used for concrete config values: `onnx-community/sam3-tracker-ONNX`, raw `config.json`, `preprocessor_config.json`, and `processor_config.json`. This mirror reports `model_type: sam3_tracker`, `base_model: facebook/sam3`, and ONNX split artifacts for `vision_encoder` plus `prompt_encoder_mask_decoder`.

Important observed config values from `onnx-community/sam3-tracker-ONNX/config.json`:
- Image size 1008, prompt patch size 14, prompt/image embedding grid 72x72.
- Vision backbone: `sam3_vit_model`, hidden size 1024, 32 layers, 16 heads, patch 14, window size 24, global attention layers `[7, 15, 23, 31]`, FPN hidden size 256, feature sizes `[[288, 288], [144, 144], [72, 72]]`.
- Mask decoder: hidden size 256, 2 two-way layers, 8 heads, attention downsample rate 2, MLP dim 2048, 4 mask tokens including single-mask token, dynamic multimask stability enabled.
- The open mirror config includes video tracker fields (`num_maskmem`, memory attention, memory encoder, object pointer fields) even though its `model_type` is `sam3_tracker`. The local `Sam3TrackerConfig` class is strict and does not define these fields; DinoML should treat them as video-tracker/remote-export metadata unless using `sam3_tracker_video`.

No tests/imports were run. Source was inspected by static file reads and HF HTTP metadata/config reads only.
