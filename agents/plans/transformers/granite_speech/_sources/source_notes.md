# Granite Speech Source Notes

Audit date: 2026-05-13

## Local Transformers source

Pinned checkout:

```text
X:/H/transformers
commit b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Copied source snapshots:

| File | SHA256 |
| --- | --- |
| `configuration_granite_speech.py` | `4127FF34566810067510EF9EBCDBAEA70BE36B330A46EF654020A1677E684418` |
| `feature_extraction_granite_speech.py` | `2E5EED2E6DE7EF976C781197626DEE860295D918F68DABF16345DC86D63CC760` |
| `modeling_granite_speech.py` | `85F524833166719D38831DAE49879393DE74F32FE48290B9B2071E9EFCB2B5A4` |
| `processing_granite_speech.py` | `5C44FE8F81062B815E9BE36635D3BF76A645A64E66F57E51FBAD0FDA851B5798` |

Delegated source inspected in-place but not copied:

- `X:/H/transformers/src/transformers/models/granite/modeling_granite.py`
- `X:/H/transformers/src/transformers/models/blip_2/modeling_blip_2.py`

## Hugging Face config snapshots

Official files fetched from Hugging Face:

| Snapshot | Source URL | SHA256 |
| --- | --- | --- |
| `hf_granite-speech-3.3-2b_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/config.json> | `3153BB499E57B9EBABA10DD87DA29B6F64C4E4F381E59CF88B6B774916E6AF3A` |
| `hf_granite-speech-3.3-8b_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.3-8b/resolve/main/config.json> | `FCCC1015ADEC31C417950F95BA37F479A42DFC39452A5F53B440FD7261C6AEFC` |
| `hf_granite-speech-3.2-8b_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.2-8b/resolve/main/config.json> | `E01C0411584F9B3E5090C53D5650FAB6960F7F0B225F53C5EF36A9E9775E7131` |
| `hf_granite-speech-3.3-2b_adapter_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/adapter_config.json> | `EAC02E40A235D1F67AF5B3579FF76AE77EC26229390DFCA5A599973F8AD9F72C` |
| `hf_granite-speech-3.3-2b_tokenizer_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/tokenizer_config.json> | `E5C80E617E26C2E2407CABF3485768A0CF98E4BE2DBF2C731B44CC0E5AA2945B` |
| `hf_granite-speech-3.3-2b_preprocessor_config.json` | <https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/preprocessor_config.json> | `44136FA355B3678A1146AD16F7E8649E94FB4FC21FE77E8310C060F61CAAFF8A` |
| `hf_granite-speech-3.3-2b_model_api.json` | <https://huggingface.co/api/models/ibm-granite/granite-speech-3.3-2b> | `8BC8EDA6EE8D5A7F6AE38B5D5841F8AFD7985BD371363A5EB6623C1A86F4FD45` |

`processor_config.json` for `ibm-granite/granite-speech-3.3-2b` returned 404. `preprocessor_config.json` exists but is `{}`, so preprocessing dimensions in the report come from the source defaults.

## Source anchors

- Feature extraction defaults: `feature_extraction_granite_speech.py:41-47`.
- Mel/logmel transform and stacking: `feature_extraction_granite_speech.py:104-116`.
- Audio placeholder expansion: `processing_granite_speech.py:72-88`.
- Projector block/window reshape and Q-Former call: `modeling_granite_speech.py:72-103`.
- Conformer attention block padding, relative-position bias, SDPA math backend: `modeling_granite_speech.py:151-190`.
- Conformer convolution module: `modeling_granite_speech.py:210-238`.
- Encoder mid-layer CTC-style softmax injection: `modeling_granite_speech.py:312-319`.
- Audio/text embedding stitch: `modeling_granite_speech.py:543-571`.
- Generation audio-input forwarding policy: `modeling_granite_speech.py:489-518`.
- LoRA adapter enable/disable during generation: `modeling_granite_speech.py:573-585`.

