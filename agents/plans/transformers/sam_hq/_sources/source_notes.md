# SAM-HQ Source Notes

Scope: `sam_hq` only.

Transformers source checkout:

- Path: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Authoritative family source: `src/transformers/models/sam_hq/modular_sam_hq.py`
- Generated source inspected for concrete line-oriented behavior: `src/transformers/models/sam_hq/modeling_sam_hq.py` and `configuration_sam_hq.py`
- Shared SAM processor/postprocess source inspected because `SamHQProcessor.post_process_masks` delegates to the SAM image processor: `src/transformers/models/sam/image_processing_sam.py`

HF config snapshots saved in this folder:

- `sam-hq-vit-base.config.json`
- `sam-hq-vit-base.preprocessor_config.json`
- `sam-hq-vit-large.config.json`
- `sam-hq-vit-large.preprocessor_config.json`
- `sam-hq-vit-huge.config.json`
- `sam-hq-vit-huge.preprocessor_config.json`

Representative config summary:

| Checkpoint | ViT hidden | ViT layers | ViT heads | ViT MLP | Global attention indexes | Mask decoder `vit_dim` |
|---|---:|---:|---:|---:|---|---:|
| `syscv-community/sam-hq-vit-base` | 768 | 12 | 12 | 3072 | `[2,5,8,11]` | 768 |
| `syscv-community/sam-hq-vit-large` | 1024 | 24 | 16 | 4096 | `[5,11,17,23]` | 1024 |
| `syscv-community/sam-hq-vit-huge` | 1280 | 32 | 16 | 5120 | `[7,15,23,31]` | 1280 |

Key source observations:

- The generated files state they are generated from `modular_sam_hq.py`; future upstream source edits should target modular source, but DinoML audit claims were verified against generated modeling/config files.
- `SamHQVisionEncoder` receives NCHW `pixel_values`, uses non-overlap Conv2d patch embedding, then permutes to NHWC tokens for the ViT blocks. The neck permutes back to NCHW and returns `[B, 256, 64, 64]` for 1024 input and 16 patch size.
- Vision layers alternate mostly local window attention (`window_size=14`) with global attention at `global_attn_indexes`. Attention uses decomposed relative position bias when `use_rel_pos=True`.
- `get_image_embeddings(pixel_values)` returns a pair: final NCHW image embeddings plus a list of intermediate NHWC embeddings collected from non-windowed/global layers. SAM-HQ needs both for full HQ feature fusion when image embeddings are precomputed.
- Prompt encoder accepts points, labels, boxes, and optional input masks. Point/box coordinates are normalized in the processor to resized image coordinates before model entry.
- Mask decoder adds one HQ output token and `hq_mask_mlp`, fuses decoder upscaled features with compressed ViT/intermediate features, and supports `hq_token_only`.
- Low-res mask logits are produced at 256x256 for the standard 1024x1024 input path. End-to-end parity requires `post_process_masks`: upsample to padded image size, crop to `reshaped_input_sizes`, upsample to `original_sizes`, then optional threshold.
- Official `syscv-community` config/preprocessor snapshots were accessible without gating at audit time. No remote-code-only config was needed for the scoped report.
