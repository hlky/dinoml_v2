# EdgeTAM Video Source Notes

Audit date: 2026-05-13

## Pinned local source

Transformers checkout: `X:/H/transformers`

Commit:

```text
b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Files inspected:

- `src/transformers/models/edgetam_video/configuration_edgetam_video.py`
- `src/transformers/models/edgetam_video/modeling_edgetam_video.py`
- `src/transformers/models/edgetam_video/modular_edgetam_video.py`
- `src/transformers/models/edgetam_video/convert_edgetam_video_to_hf.py`
- `src/transformers/models/edgetam/modeling_edgetam.py`
- `src/transformers/models/edgetam/configuration_edgetam.py`
- `src/transformers/models/sam2_video/processing_sam2_video.py`
- `src/transformers/models/sam2_video/video_processing_sam2_video.py`
- `src/transformers/models/sam2/image_processing_sam2.py`
- `src/transformers/models/timm_wrapper/modeling_timm_wrapper.py`
- `src/transformers/models/timm_wrapper/configuration_timm_wrapper.py`

`configuration_edgetam_video.py`, `modeling_edgetam_video.py`, `processing_sam2_video.py`, and `image_processing_sam2.py` are generated from modular source files. For upstream edits, prefer the modular files. For DinoML ABI audit, the generated modeling file was treated as the exact runtime basis.

## HF configs and metadata

Primary Transformers checkpoint:

- Model: `yonigozlan/EdgeTAM-hf`
- Config URL: `https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/config.json`
- Preprocessor URL: `https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/preprocessor_config.json`
- Processor URL: `https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/processor_config.json`
- Video preprocessor URL: `https://huggingface.co/yonigozlan/EdgeTAM-hf/raw/main/video_preprocessor_config.json`
- HF API SHA observed: `c266ce53b3fc00f0f495b583f6a116c4e57f53bb`
- HF metadata: non-gated, `library_name=transformers`, tags include `edgetam_video`, `mask-generation`, `license:apache-2.0`.

Original/reference repo metadata:

- Model: `facebook/EdgeTAM`
- HF API SHA observed: `14d7ecc48c656b94e5184519f698cd5386c5a2bf`
- HF metadata: non-gated, `library_name=edgetam`, only `edgetam.pt` plus README and no Transformers `config.json` in the inspected API response.

Representative sweep limitation:

- Only one in-library `edgetam_video` checkpoint with a full Transformers config was found in the quick HF API/search sweep: `yonigozlan/EdgeTAM-hf`.
- ONNX and endpoint mirrors exist, but they are not authoritative for the in-library PyTorch operator ABI.
- The source defaults were used as the second reference point where no separate small/large Transformers configs were available.

## High-signal source anchors

- `EdgeTamVideoInferenceSession`: object ids, per-frame inputs, per-object output histories, frame storage, and vision feature cache.
- `EdgeTamVideoInferenceCache`: bounded frame feature cache with state-device/inference-device movement.
- `EdgeTamVideoModel.forward`: per-object frame loop, condition-vs-tracked output storage, streaming frame admission, final `EdgeTamVideoSegmentationOutput`.
- `_prepare_vision_features`: frame-to-FPN cache path; source assumes NCHW frame tensors.
- `get_image_features`: vision `fpn_hidden_states` are converted from NCHW maps to `HW x B x C` token streams.
- `_prepare_memory_conditioned_features`: gathers mask memories/object pointers, concatenates memory tokens, runs memory attention, reshapes back to BCHW.
- `_encode_new_memory`: converts current `HW x B x C` features and high-res mask logits into spatial memory tokens via memory encoder plus spatial perceiver.
- `EdgeTamVideoMemoryAttention`: 2-layer noncausal memory self/cross attention, fixed 2D RoPE tables, object-pointer RoPE exclusions.
- `EdgeTamVideoMaskDecoder`: two-way transformer, transposed-conv upscaling, hypernetwork mask projection, dynamic multimask stability fallback.
- `Sam2VideoProcessor` / `Sam2VideoVideoProcessor`: frame/image resize/normalize to channels-first 1024x1024, point/box normalization, postprocess mask upsampling/cropping.

## Gaps

- No code tests or imports were run by request.
- The `timm_wrapper` delegated RepViT backbone topology was not expanded into every concrete conv/block; DinoML should either audit `edgetam`/`timm_wrapper` separately or allowlist the exact `architecture=repvit_m1`, `features_only=True`, `out_indices=[0,1,2,3]` config.
- HF checkpoint sweep is shallow because only one full Transformers `edgetam_video` config was found. Other EdgeTAM mirrors are ONNX, non-Transformers, or endpoint wrappers.
