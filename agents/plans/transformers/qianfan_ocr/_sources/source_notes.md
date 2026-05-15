# Qianfan OCR Source Notes

Local Transformers checkout:

```text
Path: transformers
Commit: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Version describe: v4.50.3-DeepSeek-3-4398-gb75feb2af6
Family dir: transformers/src/transformers/models/qianfan_ocr
```

Inspected local files:

```text
configuration_qianfan_ocr.py
modeling_qianfan_ocr.py
modular_qianfan_ocr.py
processing_qianfan_ocr.py
```

Related local files used for inherited behavior:

```text
transformers/src/transformers/models/qwen3/configuration_qwen3.py
transformers/src/transformers/models/qwen3/modeling_qwen3.py
transformers/src/transformers/models/got_ocr2/image_processing_got_ocr2.py
transformers/src/transformers/models/internvl/modeling_internvl.py
transformers/src/transformers/models/internvl/processing_internvl.py
```

Representative checkpoint:

```text
Model id: baidu/Qianfan-OCR
Model revision from HF API: 623bf5d20d446abdb36606aa4547cd0c18886fe5
HF URL: https://huggingface.co/baidu/Qianfan-OCR
Access: public, not gated in API response
License tag: apache-2.0 from HF metadata
Parameter metadata: 4,741,408,256 BF16 parameters from safetensors metadata in HF API
Pipeline tag: image-text-to-text
```

HF files sampled:

```text
config.json
processor_config.json
preprocessor_config.json
tokenizer_config.json
generation_config.json
HF model API metadata
```

Checkpoint-significant values:

```text
vision hidden/layers/heads: 1024 / 24 / 16
vision patch/image: 14 / 448
text model_type: qwen3
text hidden/layers/heads/kv_heads/head_dim: 2560 / 36 / 32 / 8 / 128
text intermediate/vocab/max_position: 9728 / 153678 / 32768
text rope_theta: 5000000
text use_cache: false in config.json
image_token_id: 151671 in checkpoint config
source default image_token_id: 151667 in configuration_qianfan_ocr.py
processor image_seq_length: 256
image processor: GotOcr2ImageProcessor, 448x448, channels_first, crop_to_patches false in preprocessor_config.json
processor defaults in QianfanOCRProcessorKwargs: crop_to_patches true unless checkpoint image kwargs override is respected
```

