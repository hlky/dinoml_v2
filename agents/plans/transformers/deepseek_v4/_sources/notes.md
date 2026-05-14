# DeepSeek-V4 audit source notes

Local Transformers checkout:

- Path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Model source: `src/transformers/models/deepseek_v4/`
- Generated runtime file: `modeling_deepseek_v4.py`
- Modular authoring file: `modular_deepseek_v4.py`

Fetched config snapshots:

- `deepseek-ai__DeepSeek-V4-Flash-Base.config.json`
- `deepseek-ai__DeepSeek-V4-Flash.config.json`
- `deepseek-ai__DeepSeek-V4-Pro-Base.config.json`
- `deepseek-ai__DeepSeek-V4-Pro.config.json`
- `mlx-community__DeepSeek-V4-Flash-4bit.config.json`

Config schedule note:

- Official Flash configs declare 43 layers and 44 `compress_ratios` entries. The current `DeepseekV4Config.__post_init__` truncates to `num_hidden_layers`, yielding 2 used sliding layers, 21 CSA layers, and 20 HCA layers.
- Official Pro configs declare 61 layers and 62 `compress_ratios` entries. Truncation yields 30 CSA layers and 31 HCA layers, with no used sliding layer.
- The MLX 4-bit mirror uses the Flash architecture dimensions but a large mirror-specific `quantization` map. It is not implemented by the in-library DeepSeek-V4 PyTorch source and should be treated as an external packed-weight format.
