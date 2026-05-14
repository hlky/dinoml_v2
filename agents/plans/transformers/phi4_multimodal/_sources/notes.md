# phi4_multimodal source notes

Audit date: 2026-05-13

Transformers checkout: `X:/H/transformers`

Transformers commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source files inspected:

| File | SHA256 |
| --- | --- |
| `configuration_phi4_multimodal.py` | `4CE4AAB18EFD38A4A463C0B39C7158BA179409AB8A614979AED6DADFB261E773` |
| `modeling_phi4_multimodal.py` | `BF17F7A059602CE967826DCEE42235886B48C0A17351E239EA87CDE298C9F347` |
| `modular_phi4_multimodal.py` | `646965C19268BFD225A18FADA73E12B1B84569F751BE92DA4C561B3AC7D2DD80` |
| `processing_phi4_multimodal.py` | `E15DC4860BF4E65762459394AC0ED889A0F0140BBD7CA686480E74D4B913D7B7` |
| `image_processing_phi4_multimodal.py` | `13A8A0BB002DDB53F37DA59B170DB09C5D8B19D51A7247EBBC1B7AFEF4940102` |
| `feature_extraction_phi4_multimodal.py` | `1C3156BADD160EAD549EF39AB46A499A59956131B4B3ABC4398A56FC951DFB70` |

Representative config sweep:

| Repo | Source status | Notes |
| --- | --- | --- |
| `microsoft/Phi-4-multimodal-instruct` | Official HF config fetched from `https://huggingface.co/microsoft/Phi-4-multimodal-instruct/raw/main/config.json` | Historical remote-code schema: `model_type="phi4mm"`, `architectures=["Phi4MMForCausalLM"]`, `auto_map` to `configuration_phi4mm.py` and `modeling_phi4mm.py`; text dims `hidden_size=3072`, `layers=32`, `heads=24`, `kv_heads=8`, `partial_rotary_factor=0.75`, longrope, `sliding_window=262144`, `torch_dtype=bfloat16`; includes nested legacy `audio_processor` and `embd_layer` fields. |
| `microsoft/Phi-4-multimodal-instruct` preprocessor | Official HF file fetched from `https://huggingface.co/microsoft/Phi-4-multimodal-instruct/raw/main/preprocessor_config.json` | `processor_class="Phi4MMProcessor"`, `image_processor_type="Phi4MMImageProcessor"`, `feature_extractor_type="Phi4MMAudioFeatureExtractor"`, `dynamic_hd=36`, `audio_compression_rate=8`, `audio_downsample_rate=1`, `audio_feat_stride=1`. |
| `microsoft/Phi-4-multimodal-instruct` tokenizer | Official HF file fetched from `https://huggingface.co/microsoft/Phi-4-multimodal-instruct/raw/main/tokenizer_config.json` | GPT2TokenizerFast; token id 200010 maps to `<\|endoftext10\|>` and is the image token in source config; 200011 maps to `<\|endoftext11\|>` and is the audio token; 199999 is BOS/EOS/PAD in tokenizer config; 200020 is an additional end token. |
| `tiny-random/phi-4-multimodal` | Public debug checkpoint config fetched from HF | Same remote-code schema, tiny dims `hidden_size=16`, `layers=2`, `heads=2`, `kv_heads=1`, `intermediate_size=32`, tiny audio dims, longrope factors length 3. Useful for parser smoke only; not representative of current native `phi4_multimodal` defaults. |
| `yujiepan/phi-4-multimodal-tiny-random` | Public debug mirror fetched from HF | Same tiny settings as `tiny-random/phi-4-multimodal`; no operator-significant variation beyond repo provenance. |
| `junnei/Phi-4-multimodal-instruct-ko-asr` | Public finetune config fetched from HF | Same remote-code schema and full text/audio dimensions as official; `torch_dtype=float32`; tokenizer auto_map points at a model-specific Xenova/gpt-4o path. |
| `huihui-ai/Phi-4-multimodal-instruct-abliterated` | Public finetune config fetched from HF | Same full dimensions and longrope config as official; `torch_dtype=bfloat16`; no structural model variation found from config. |
| `microsoft/Phi-4-multimodal-instruct-onnx` | Public ONNX repo config URL returned an LFS pointer | `config.json` at main is an LFS pointer (`oid sha256:49e1...`, `size 4631`) rather than direct JSON through raw URL. Treat as deployment artifact, not a native Transformers checkpoint basis for this report. |

Native-source versus historical-config warning:

The current in-library files define `model_type="phi4_multimodal"` and classes named `Phi4Multimodal*`. The official public checkpoint still advertises remote-code `phi4mm` class names and a legacy config schema. This report scopes runtime behavior to the inspected native source, while using public configs to expose dimensions and migration traps. DinoML should reject or route legacy `phi4mm` remote-code checkpoints until a compatibility mapper is explicitly implemented.
