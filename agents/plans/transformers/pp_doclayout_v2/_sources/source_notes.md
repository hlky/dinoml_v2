# PP-DocLayoutV2 source notes

## Local source basis

- Transformers checkout: `transformers`, `git rev-parse HEAD = b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.
- Family directory: `transformers/src/transformers/models/pp_doclayout_v2`.
- Generated files all state they are produced from `modular_pp_doclayout_v2.py`; use generated files for exact runtime audit and modular file as future edit authority.
- Nested backbone source inspected: `transformers/src/transformers/models/hgnet_v2`.
- Hub-kernel dispatch inspected: `transformers/src/transformers/integrations/hub_kernels.py`.

## Hugging Face checkpoint/config basis

- Model id: `PaddlePaddle/PP-DocLayoutV2_safetensors`.
- HF API repo SHA observed on 2026-05-13: `880e8971b88938518611c54fc0f59ad57849c9d4`.
- Repo is not gated; API reports `pipeline_tag=object-detection`, `library_name=PaddleOCR`, `license=apache-2.0`, files `config.json`, `preprocessor_config.json`, `model.safetensors`, `inference.yml`, `README.md`.
- Only one native `pp_doclayout_v2` checkpoint was found during this audit; no 3-5 checkpoint sweep was available for this model family.

## Key line anchors

- `configuration_pp_doclayout_v2.py`: `PPDocLayoutV2ReadingOrderConfig` at line 31, `PPDocLayoutV2Config` at line 107, `model_type` at line 184, post-init/default HGNetV2 backbone consolidation at lines 235-263.
- `image_processing_pp_doclayout_v2.py`: processor class at line 34, `_preprocess` at line 44, reading-order vote sort at line 91, object-detection postprocess at line 125, flattened class-query `topk` at line 171, box/order gather at lines 174-175.
- `modeling_pp_doclayout_v2.py`: reading-order GlobalPointer at line 55, relation embedding at line 75, reading-order self-attention at line 152, text/layout embeddings at line 452, eager deformable attention fallback at lines 572-622, deformable attention wrapper at line 625, reading-order model forward at line 843, HGNet backbone adapter at line 1260, AIFI flatten/permute at line 1645, hybrid encoder at line 1672, decoder at line 1793, main model at line 2030, anchor generation at line 2114, top-k proposal selection at line 2282, object-detection head at line 2348, class thresholds/order remap at lines 2468-2485.
- `modeling_hgnet_v2.py`: NCHW Conv2d backbone layers at line 80, stem at line 130, explicit pad/pool/cat at lines 175 and 185-190, dense concatenation aggregation at line 260, backbone output feature selection at lines 396-408.
- `integrations/hub_kernels.py`: `use_kernel_forward_from_hub` at line 63, `MultiScaleDeformableAttention` CUDA hub-kernel mapping at lines 88-94, no-kernels fallback decorator at lines 256-262.

## Effective checkpoint values

- Detection config: `d_model=256`, `num_queries=300`, `num_feature_levels=3`, `decoder_layers=6`, `decoder_attention_heads=8`, `decoder_n_points=4`, `decoder_ffn_dim=1024`, `encoder_hidden_dim=256`, `encoder_layers=1`, `encoder_attention_heads=8`, `encoder_ffn_dim=1024`, `encode_proj_layers=[2]`, `feat_strides=[8,16,32]`, `encoder_in_channels=[512,1024,2048]`, `decoder_in_channels=[256,256,256]`, `activation_function=silu`, `encoder_activation_function=gelu`, `decoder_activation_function=relu`, `anchor_image_size=null`, `eval_size=null`, `disable_custom_kernels=true`, `torch_dtype=float32`.
- Backbone config: `model_type=hgnet_v2`, `arch=L`, `return_idx=[1,2,3]`, `out_features=["stage2","stage3","stage4"]`, `freeze_norm=true`, `freeze_at=0`, `freeze_stem_only=true`, `lr_mult_list=[0,0.05,0.05,0.05,0.05]`.
- Reading-order config: `hidden_size=512`, `num_attention_heads=8`, `num_hidden_layers=6`, `intermediate_size=2048`, `has_relative_attention_bias=false`, `has_spatial_attention_bias=true`, `vocab_size=4`, `type_vocab_size=1`, `max_position_embeddings=514`, `max_2d_position_embeddings=1024`, `coordinate_size=171`, `shape_size=170`, `num_classes=20`, `global_pointer_head_size=64`.
- Processor config: resize to `800x800`, bicubic `resample=3`, rescale by `1/255`, normalize with mean `[0,0,0]` and std `[1,1,1]`; emitted `pixel_values` are torch image tensors.

## Source gaps and cautions

- Source has training/denoising helpers, but `PPDocLayoutV2ForObjectDetection.forward` raises if `labels is not None`; inference audit can defer training.
- `disable_custom_kernels` is stored on the deformable-attention wrapper, but the inspected forward path always calls the decorated `MultiScaleDeformableAttention` layer. DinoML should treat the custom CUDA hub kernel as an optional external source boundary and keep the eager `grid_sample` math as the semantic reference.
- Object detection postprocess has no NMS; parity requires preserving top-k flattened class scores, threshold filtering, class-order remapping, reading-order sorting, and target-size scaling.
- Dynamic image sizes are supported by source anchor generation when `anchor_image_size is None`; the checked checkpoint uses `null`, so anchors are generated from runtime feature-map shapes.
