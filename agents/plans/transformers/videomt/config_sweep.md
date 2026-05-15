# VidEoMT Config Sweep Evidence

Source basis: public Hugging Face `config.json` files fetched on 2026-05-13 plus `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

`preprocessor_config.json` returned 404 for the sampled public repos. Processor behavior in the report therefore comes from `video_processing_videomt.py` and inherited `BaseVideoProcessor` defaults, not per-checkpoint processor files.

| Model id | Hidden | Layers | Heads | Head dim | Image | Patch | Grid | Frames | Blocks with queries | Labels | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tue-mps/videomt-dinov2-small-ytvis2019` | 384 | 12 | 6 | 64 | 640 | 16 | 40x40 | 2 | 3 | 40 | YTVIS 2019 labels |
| `tue-mps/videomt-dinov2-base-ytvis2019` | 768 | 12 | 12 | 64 | 640 | 16 | 40x40 | 2 | 3 | 40 | Same sequence lengths as small, wider |
| `tue-mps/videomt-dinov2-large-ytvis2019` | 1024 | 24 | 16 | 64 | 640 | 16 | 40x40 | 2 | 4 | 40 | Deeper; final 4 blocks receive queries |
| `tue-mps/videomt-dinov2-large-ovis` | 1024 | 24 | 16 | 64 | 640 | 16 | 40x40 | 2 | 4 | 25 | Class head changes to 26 outputs including null |
| `tue-mps/videomt-dinov2-large-vspw` | 1024 | 24 | 16 | 64 | 1280 | 16 | 80x80 | 2 | 4 | 124 | Much larger patch sequence and mask output |

Derived runtime shapes for `B=1`, `num_queries=200`, `num_register_tokens=4`, `num_upscale_blocks=2`:

| Grid | Encoder no-query sequence | Query-stage sequence | Upscaled mask feature | Mask logits per frame |
|---|---:|---:|---:|---:|
| 40x40 | 1605 | 1805 | `[1,C,160,160]` | `[1,200,160,160]` |
| 80x80 | 6405 | 6605 | `[1,C,320,320]` | `[1,200,320,320]` |

