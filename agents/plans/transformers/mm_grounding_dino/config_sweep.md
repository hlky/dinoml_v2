# MM Grounding DINO config sweep

Source: Hugging Face `config.json`, `preprocessor_config.json`, and `tokenizer_config.json` fetched 2026-05-13 from public model repos. `processor_config.json` returned 404 for all sampled repos; processor identity is carried by preprocessor/tokenizer configs.

| Model id | Backbone config | Backbone out features | `num_feature_levels` | Core dims | Text config | Processor |
| --- | --- | --- | ---: | --- | --- | --- |
| `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg` | Swin-T, `embed_dim=96`, depths `[2,2,6,2]`, heads `[3,6,12,24]`, `window_size=7`, `hidden_size=768` | `stage2,stage3,stage4` | 4 | `d_model=256`, encoder/decoder layers 6/6, heads 8/8, FFN 2048, queries 900 | BERT base: hidden 768, layers 12, heads 12, FFN 3072, vocab 30522 | `GroundingDinoImageProcessor`, resize shortest 800 longest 1333, pad, ImageNet mean/std; `BertTokenizer` lower-case |
| `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` | Same as tiny above | `stage2,stage3,stage4` | 4 | Same as tiny | Same as tiny | Same as tiny |
| `openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det` | Swin-B, `embed_dim=128`, depths `[2,2,18,2]`, heads `[4,8,16,32]`, `window_size=12`, `hidden_size=1024` | `stage2,stage3,stage4` | 4 | Same detector dims | Same as tiny | Same as tiny |
| `openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg` | Swin-L, `embed_dim=192`, depths `[2,2,18,2]`, heads `[6,12,24,48]`, `window_size=12`, `hidden_size=1536` | `stage1,stage2,stage3,stage4` | 5 | Same detector dims except `num_feature_levels=5` | Same as tiny | Same as tiny |
| `iSEE-Laboratory/llmdet_tiny` | Same Swin-T shape as tiny | `stage2,stage3,stage4` | 4 | Same as tiny | Same as tiny | Same as tiny |

Common checkpoint fields:

- `model_type="mm-grounding-dino"`, architecture `MMGroundingDinoForObjectDetection`.
- `activation_function="relu"`, `dropout=0.1`, `attention_dropout=0.0`, `activation_dropout=0.0`.
- `two_stage=true`, `embedding_init_target=true`, `query_dim=4`.
- `encoder_n_points=4`, `decoder_n_points=4`, `max_text_len=256`.
- Historical config flags `decoder_bbox_embed_share=false`, `decoder_cls_embed_share=false`, `two_stage_bbox_embed_share=false` are present in sampled configs but the inspected native source does not read them; native MM Grounding DINO always creates per-decoder-layer `bbox_embed` and `class_embed` lists in `MMGroundingDinoForObjectDetection`.
