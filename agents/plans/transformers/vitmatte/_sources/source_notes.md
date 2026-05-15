# VitMatte source notes

Source basis:

- Transformers checkout: `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Native source inspected:
  - `src/transformers/models/vitmatte/configuration_vitmatte.py`
  - `src/transformers/models/vitmatte/modeling_vitmatte.py`
  - `src/transformers/models/vitmatte/image_processing_vitmatte.py`
  - `src/transformers/models/vitmatte/image_processing_pil_vitmatte.py`
  - nested default backbone: `src/transformers/models/vitdet/configuration_vitdet.py`
  - nested default backbone: `src/transformers/models/vitdet/modeling_vitdet.py`
- Official HF config snapshots are under `_sources/hf_configs/`.

Representative checkpoints:

- `hustvl/vitmatte-small-composition-1k`
- `hustvl/vitmatte-base-composition-1k`
- `hustvl/vitmatte-small-distinctions-646`
- `hustvl/vitmatte-base-distinctions-646`

Important source facts:

- `VitMatteForImageMatting` loads a nested backbone through `load_backbone(config)` and consumes only `outputs.feature_maps[-1]`.
- The public HUST-VL configs use `backbone_config.model_type="vitdet"`, `num_channels=4`, `image_size=512`, `out_features=["stage12"]`, `use_relative_position_embeddings=true`, `window_size=14`, window blocks `[0,1,3,4,6,7,9,10]`, and residual blocks `[2,5,8,11]`.
- Small checkpoints explicitly set backbone `hidden_size=384` and `num_attention_heads=6`; base checkpoints omit those fields and therefore use VitDet defaults `hidden_size=768` and `num_attention_heads=12`.
- The processor packs RGB image plus trimap into `pixel_values` with shape `[B,4,H_pad,W_pad]`. Image channels are rescaled and normalized; trimap is rescaled only and concatenated as the fourth channel.
- Source defaults use ImageNet mean/std, but the public HUST-VL preprocessor configs override mean/std to `[0.5,0.5,0.5]`.
- Padding is bottom/right zero padding to make height and width divisible by `size_divisor`/`size_divisibility` 32.
- VitMatte has no native alpha postprocess helper. The model returns sigmoid alpha at padded tensor size; crop/resize/original-size handling must be owned by the caller or a DinoML wrapper for end-to-end parity.
- Layout boundary: model input and decoder are NCHW; VitDet transformer layers temporarily permute feature maps to NHWC for LayerNorm, attention, MLP, window partitioning, and relative-position math.
