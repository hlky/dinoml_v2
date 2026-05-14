# EoMT-DINOv3 Config Sweep

Source: Hugging Face `config.json` and `preprocessor_config.json` fetched on 2026-05-13.

| Model id | Task/checkpoint role | Hidden | Layers | Heads | Head dim | MLP | Image | Grid | Queries | Register tokens | Final mask blocks | Upscale blocks | Labels | Preprocessor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tue-mps/eomt-dinov3-coco-panoptic-small-640` | COCO panoptic small | 384 | 12 | 6 | 64 | 1536 | 640 | 40x40 | 200 | 4 | 3 | 2 | 133 | resize+pad to 640, CHW |
| `tue-mps/eomt-dinov3-coco-panoptic-base-640` | COCO panoptic base | 768 | 12 | 12 | 64 | 3072 | 640 | 40x40 | 200 | 4 | 3 | 2 | 133 | resize+pad to 640, CHW |
| `tue-mps/eomt-dinov3-coco-panoptic-large-640` | COCO panoptic large | 1024 | 24 | 16 | 64 | 4096 | 640 | 40x40 | 200 | 4 | 4 | 2 | 133 | resize+pad to 640, CHW |
| `tue-mps/eomt-dinov3-coco-panoptic-large-1280` | COCO panoptic large high-res | 1024 | 24 | 16 | 64 | 4096 | 1280 | 80x80 | 200 | 4 | 4 | 2 | 133 | resize+pad to 1280, CHW |
| `tue-mps/eomt-dinov3-coco-instance-large-640` | COCO instance large | 1024 | 24 | 16 | 64 | 4096 | 640 | 40x40 | 200 | 4 | 4 | 2 | 80 | resize+pad to 640, CHW |
| `tue-mps/eomt-dinov3-coco-instance-large-1280` | COCO instance large high-res | 1024 | 24 | 16 | 64 | 4096 | 1280 | 80x80 | 200 | 4 | 4 | 2 | 80 | resize+pad to 1280, CHW |
| `tue-mps/eomt-dinov3-ade-semantic-large-512` | ADE semantic large | 1024 | 24 | 16 | 64 | 4096 | 512 | 32x32 | 100 | 4 | 4 | 2 | 150 | resize+pad to 512, CHW |

Common observed fields:

- `model_type="eomt_dinov3"`, `architectures=["EomtDinov3ForUniversalSegmentation"]`, `dtype="float32"`, `transformers_version="5.0.0.dev0"`.
- `patch_size=16`, `num_channels=3`, `hidden_act="gelu"`, `layer_norm_eps=1e-6`, `drop_path_rate=0`, `attention_dropout=0`, `hidden_dropout_prob=0`.
- Attention projection biases are asymmetric: `query_bias=true`, `key_bias=false`, `value_bias=true`, `proj_bias=true`.
- MLP is ungated in these checkpoints: `use_gated_mlp=false`, `mlp_bias=true`.
- RoPE appears in configs as both `rope_theta=100.0` and effective `rope_parameters={"rope_theta":100.0}`; source normalizes to `rope_type="default"`.
- Official DINOv3 backbone repos `facebook/dinov3-vits16-pretrain-lvd1689m`, `facebook/dinov3-vitb16-pretrain-lvd1689m`, and `facebook/dinov3-vitl16-pretrain-lvd1689m` returned 401 without gated access.
