# qwen3_vl_moe source notes

Audit date: 2026-05-13

Transformers checkout: `X:/H/transformers`
Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source basis:

- `src/transformers/models/qwen3_vl_moe/configuration_qwen3_vl_moe.py`
- `src/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py`
- `src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py`
- Shared processor: `src/transformers/models/qwen3_vl/processing_qwen3_vl.py`
- Shared image processor: `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
- Shared video processor: `src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py`

Representative config snapshots saved in this directory:

- `Qwen__Qwen3-VL-30B-A3B-Instruct__config.json`
- `Qwen__Qwen3-VL-30B-A3B-Thinking__config.json`
- `Qwen__Qwen3-VL-235B-A22B-Instruct__config.json`
- `Qwen__Qwen3-VL-235B-A22B-Thinking__config.json`
- `Qwen__Qwen3-VL-30B-A3B-Instruct-FP8__config.json`
- `tiny-random__qwen3-vl-moe__config.json`

Key source anchors:

- MoE expert storage and fallback expert loop: `modeling_qwen3_vl_moe.py:75`
- Router softmax/top-k/renormalization: `modeling_qwen3_vl_moe.py:114`
- Text attention projections, Q/K RMSNorm, RoPE, cache update: `modeling_qwen3_vl_moe.py:218`
- Vision patch embed Conv3d over packed processor patch rows: `modeling_qwen3_vl_moe.py:563`
- Vision merger and deepstack mergers: `modeling_qwen3_vl_moe.py:583`
- Vision rotary/grid position math: `modeling_qwen3_vl_moe.py:646`
- Vision absolute position interpolation: `modeling_qwen3_vl_moe.py:686`
- Text M-RoPE rotary embedding: `modeling_qwen3_vl_moe.py:810`
- Interleaved M-RoPE section rewrite: `modeling_qwen3_vl_moe.py:881`
- DeepStack hidden-state stitch: `modeling_qwen3_vl_moe.py:1014`
- Multimodal rope index construction: `modeling_qwen3_vl_moe.py:1162`
- Placeholder count validation and masks: `modeling_qwen3_vl_moe.py:1297`
- Image/video masked_scatter embedding stitch: `modeling_qwen3_vl_moe.py:1428`, `modeling_qwen3_vl_moe.py:1440`
- Decode input preparation drops pixels after first cached step: `modeling_qwen3_vl_moe.py:1744`
- Beam/input expansion for packed visual tensors: `modeling_qwen3_vl_moe.py:1871`
- Qwen3VL processor placeholder expansion: `processing_qwen3_vl.py:131`, `processing_qwen3_vl.py:141`
- Processor mm token type ids: `processing_qwen3_vl.py:178`
- Processor video timestamps: `processing_qwen3_vl.py:257`
- Qwen2VL image processor patch packing: `image_processing_qwen2_vl.py:148`
- Qwen3VL video processor patch packing: `video_processing_qwen3_vl.py:168`

Config sweep highlights:

| Snapshot | Text hidden/layers | Q heads/KV/head dim | MoE | Vision | Quant |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-VL-30B-A3B-Instruct | 2048 / 48 | 32 / 4 / 128 | 128 experts, top-8, moe width 768 | 1152 hidden, 27 layers, 16 heads, out 2048 | dense bf16 |
| Qwen3-VL-30B-A3B-Thinking | 2048 / 48 | 32 / 4 / 128 | 128 experts, top-8, moe width 768 | same | dense bf16 |
| Qwen3-VL-235B-A22B-Instruct | 4096 / 94 | 64 / 4 / 128 | 128 experts, top-8, moe width 1536 | 1152 hidden, 27 layers, 16 heads, out 4096 | dense bf16 |
| Qwen3-VL-235B-A22B-Thinking | 4096 / 94 | 64 / 4 / 128 | 128 experts, top-8, moe width 1536 | same | dense bf16 |
| Qwen3-VL-30B-A3B-Instruct-FP8 | 2048 / 48 | 32 / 4 / 128 | same as 30B | same | fp8, dynamic e4m3, 128x128 blocks |
| tiny-random/qwen3-vl-moe | 8 / 2 | 8 / 4 / 32 | 16 experts, top-8 | 64 hidden, 6 layers, 2 heads, out 8 | dense test config |

