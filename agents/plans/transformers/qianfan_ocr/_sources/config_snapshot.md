# Qianfan OCR Config Snapshot

Representative checkpoint: `baidu/Qianfan-OCR`.

Important config excerpt, paraphrased from downloaded JSON:

| Field | Value | Source |
| --- | --- | --- |
| `architectures` | `QianfanOCRForConditionalGeneration` | `config.json` |
| `model_type` | `qianfan_ocr` | `config.json` |
| `torch_dtype` | `bfloat16` | `config.json` |
| `downsample_ratio` | `0.5` | `config.json` |
| `image_token_id` | `151671` | `config.json` |
| `force_image_size` | `448` | historical/processor field in `config.json`; not read by inspected modeling source |
| `dynamic_image_size` | `true` | historical/processor field in `config.json`; not read by inspected modeling source |
| `min_dynamic_patch` / `max_dynamic_patch` | `1` / `12` | historical/processor field in `config.json`; image processor uses `min_patches` / `max_patches` |
| `use_thumbnail` | `true` | historical/processor field in `config.json`; image processor crop path uses thumbnail when cropping |
| `template` | `internvl2_5` | historical/chat field in `config.json` |
| `ps_version` | `v2` | historical field in `config.json`; not read by inspected modeling source |
| `vision_config.hidden_size` | `1024` | `config.json` |
| `vision_config.num_hidden_layers` | `24` | `config.json` |
| `vision_config.num_attention_heads` | `16` | `config.json` |
| `vision_config.intermediate_size` | `4096` | `config.json` |
| `vision_config.image_size` | `448` | `config.json` |
| `vision_config.patch_size` | `14` | `config.json` |
| `vision_config.norm_type` | `layer_norm` | `config.json` |
| `vision_config.qkv_bias` | `true` | historical/config field; generated Qianfan source reads `attention_bias` |
| `text_config.model_type` | `qwen3` | `config.json` |
| `text_config.hidden_size` | `2560` | `config.json` |
| `text_config.num_hidden_layers` | `36` | `config.json` |
| `text_config.num_attention_heads` | `32` | `config.json` |
| `text_config.num_key_value_heads` | `8` | `config.json` |
| `text_config.head_dim` | `128` | `config.json` |
| `text_config.intermediate_size` | `9728` | `config.json` |
| `text_config.vocab_size` | `153678` | `config.json` |
| `text_config.max_position_embeddings` | `32768` | `config.json` |
| `text_config.rope_theta` | `5000000` | `config.json` |
| `text_config.use_cache` | `false` | `config.json` |
| processor `image_seq_length` | `256` | `processor_config.json` |
| processor image tokens | `<img>`, `</img>`, `<IMG_CONTEXT>` | `processor_config.json`, `tokenizer_config.json` |
| image processor type | `GotOcr2ImageProcessor` | `preprocessor_config.json` |
| image size/layout | `448x448`, channels-first | `preprocessor_config.json` |
| image normalization | CLIP mean/std, rescale 1/255 | `preprocessor_config.json` |

