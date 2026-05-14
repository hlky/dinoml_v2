# sam3_video source notes

Audit target: `sam3_video` in `X:/H/transformers/src/transformers/models/sam3_video` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Files read:

- `sam3_video/configuration_sam3_video.py`
- `sam3_video/modeling_sam3_video.py`
- `sam3_video/processing_sam3_video.py`
- Related composed bodies:
  - `sam3/modeling_sam3.py`
  - `sam3/configuration_sam3.py`
  - `sam3/image_processing_sam3.py`
  - `sam3_tracker_video/modeling_sam3_tracker_video.py`
  - `sam3_tracker_video/configuration_sam3_tracker_video.py`
- Docs: `docs/source/en/model_doc/sam3_video.md`

Checkpoint/config sources:

- Official `facebook/sam3` repo/model card is gated; the public page states access requires sharing contact information.
- Saved open mirror config snapshot:
  - `agents/plans/transformers/sam3_video/_sources/bodhicitta_sam3_config_mirror.json`
  - Source URL: `https://huggingface.co/bodhicitta/sam3/raw/main/config.json`
  - Label in report: open mirror, not official gated source.

High-signal source anchors:

- `Sam3VideoConfig` is a composition config with `detector_config` defaulting to `sam3` and `tracker_config` defaulting to `sam3_tracker_video`. It exposes tracking heuristics such as `low_res_mask_size=288`, detection threshold/NMS, association IoUs, hotstart thresholds, keep-alive limits, memory reconditioning, and `image_size` propagation to both subconfigs.
- `Sam3VideoInferenceSession` owns mutable runtime state:
  - processed frames dictionary, original video size, placement devices, dtype;
  - prompt text/id/input/attention/embedding dictionaries;
  - per-object id mappings, mask/point inputs, `cond_frame_outputs`, `non_cond_frame_outputs`;
  - object scores, tracker scores, occlusion state, hotstart metadata, removed/suppressed object ids;
  - `Sam3VideoInferenceCache` for frame vision features with bounded oldest-frame eviction.
- `Sam3VideoModel` composes:
  - `detector_model = AutoModel.from_config(config.detector_config)`;
  - `tracker_model = AutoModel.from_config(config.tracker_config, remove_vision_encoder=True)`;
  - `tracker_neck = Sam3VisionNeck(config.detector_config.vision_config)`.
- Per-frame dataflow in `Sam3VideoModel._det_track_one_frame`:
  - fetch processed frame from session, add batch dim;
  - run detector vision encoder once;
  - run detection for each prompt, reusing cached prompt embeddings;
  - merge detections across prompts;
  - convert detector vision embeddings into tracker feature maps and cache them;
  - run tracker propagation without memory encoding;
  - plan detection/tracking updates;
  - execute additions/removals/memory updates;
  - build low-resolution output masks by object id.
- `Sam3VideoProcessor`:
  - image path delegates to `Sam3ImageProcessor`;
  - video session path delegates to `video_processor(videos=..., return_tensors="pt")` and stores `pixel_values_videos[0]`;
  - text prompts are CLIP-tokenized with `padding="max_length", max_length=32`;
  - postprocess upsamples low-res masks to original frame size with bilinear interpolation, thresholds at `> 0`, removes zero-area or hidden ids, computes `masks_to_boxes`, and reapplies prompt-group non-overlap constraints.
- Detector body from `sam3/modeling_sam3.py`:
  - ViT patch embedding is `Conv2d(num_channels -> hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)`, then flatten/transpose to token sequence.
  - ViT blocks use NHWC token maps internally for window partition/unpartition; global attention at configured block indexes.
  - Vision neck converts token sequence back to NCHW feature maps and FPN features.
  - Text is `CLIPTextModelWithProjection`; source projects the full CLIP last hidden state from 1024 to 256, not only a pooled text vector.
  - DETR encoder/decoder use 256-wide MHA/cross-attention, learned queries plus presence token, box refinement, dot-product query/text scoring, mask decoder with pixel decoder and `einsum("bqc,bchw->bqhw")`.
- Tracker body from `sam3_tracker_video/modeling_sam3_tracker_video.py`:
  - Memory state stores `maskmem_features` as flattened `(H*W, batch, 64)` bfloat16 and `maskmem_pos_enc` as flattened `(H*W, batch, 64)`/mask dtype.
  - Memory attention is 4 layers, hidden 256, one attention head in representative config, with axial 2D RoPE over `memory_attention_rope_feat_sizes`.
  - Memory features and object pointer tokens are concatenated along sequence; object pointers can be split from 256 to 64-wide chunks when `mem_dim < hidden_dim`.
  - New memories are encoded by resizing masks if needed, transforming mask logits through sigmoid/scale/bias, fusing with current NCHW visual feature maps, optional occlusion spatial embedding, then flattening to the memory ABI.

No code tests/imports were run for this audit.
