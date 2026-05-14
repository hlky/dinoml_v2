# pvt_v2 Source Notes

Scope: source and config audit only. No DinoML code edits, imports, tests, or model execution.

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Report spec: `H:/dinoml_v2/agents/plans/transformers/PROMPT.md`

Important local files:

- `src/transformers/models/pvt_v2/modeling_pvt_v2.py`
- `src/transformers/models/pvt_v2/configuration_pvt_v2.py`
- `src/transformers/models/pvt_v2/convert_pvt_v2_to_pytorch.py`
- `src/transformers/models/pvt/image_processing_pvt.py`
- `tests/models/pvt_v2/test_modeling_pvt_v2.py`

## Source observations

- `PvtV2OverlapPatchEmbeddings` uses `nn.Conv2d` with `padding=patch_size//2`, then `flatten(2).transpose(1,2)`, then `LayerNorm(hidden_size)`.
- `PvtV2SelfAttention` has separate `query`, `key`, and `value` Linear modules. The original checkpoint converter splits original `attn.kv` storage into key then value.
- `qkv_bias` is passed only to Q/K/V Linear modules; attention output projection and MLP Linear modules use default bias.
- Normal SRA path for `sr_ratio > 1`: token input is reshaped to NCHW, passed through `Conv2d(hidden, hidden, kernel=sr_ratio, stride=sr_ratio)`, flattened back to tokens, and LayerNormed before K/V projection.
- Linear SRA path: token input -> NCHW -> `AdaptiveAvgPool2d(7)` -> 1x1 Conv2d -> tokens -> LayerNorm -> GELU before K/V projection.
- MLP path: Linear -> optional ReLU only when `linear_attention=True` -> 3x3 depthwise Conv2d -> activation from `hidden_act` -> Linear.
- Encoder returns last hidden state as NCHW. Hidden states collected for backbone are also NCHW stage maps.
- Classification head converts final NCHW map to tokens and uses mean over spatial tokens, then a Linear classifier.
- `PvtV2Backbone` returns selected NCHW `feature_maps` by `stage_names`/`out_features`.
- There is no `image_processing_pvt_v2.py`; shared `PvtImageProcessor` handles resize/rescale/normalize.

## Representative checkpoint config sweep

Fetched from official Hugging Face model repos on 2026-05-13.

| Checkpoint | Depths | Hidden sizes | Heads | MLP ratios | sr_ratios | linear_attention | drop_path |
|---|---|---|---|---|---|---:|---:|
| `OpenGVLab/pvt_v2_b0` | `2,2,2,2` | `32,64,160,256` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `OpenGVLab/pvt_v2_b1` | `2,2,2,2` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `OpenGVLab/pvt_v2_b2` | `3,4,6,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `OpenGVLab/pvt_v2_b2_linear` | `3,4,6,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | true | 0.0 |
| `OpenGVLab/pvt_v2_b3` | `3,4,18,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.0 |
| `OpenGVLab/pvt_v2_b4` | `3,8,27,3` | `64,128,320,512` | `1,2,5,8` | `8,8,4,4` | `8,4,2,1` | false | 0.3 |
| `OpenGVLab/pvt_v2_b5` | `3,6,40,3` | `64,128,320,512` | `1,2,5,8` | `4,4,4,4` | `8,4,2,1` | false | 0.3 |

All checked preprocessors use `PvtImageProcessor`, resize to 224x224, rescale, and ImageNet normalization.

Observed historical/ignored field: representative configs include `reshape_last_stage=true`, but the inspected `modeling_pvt_v2.py` source does not read it.

## Useful links

- Modeling source: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt_v2/modeling_pvt_v2.py
- Config source: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt_v2/configuration_pvt_v2.py
- Shared image processor: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/pvt/image_processing_pvt.py
- HF docs page: https://huggingface.co/docs/transformers/model_doc/pvt_v2
- B0 config: https://huggingface.co/OpenGVLab/pvt_v2_b0/raw/main/config.json
- B2 Linear config: https://huggingface.co/OpenGVLab/pvt_v2_b2_linear/raw/main/config.json
- B5 config: https://huggingface.co/OpenGVLab/pvt_v2_b5/raw/main/config.json
