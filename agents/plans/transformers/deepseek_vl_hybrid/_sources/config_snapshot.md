# Config snapshot: deepseek_vl_hybrid

## deepseek-community/deepseek-vl-7b-base and 7b-chat

Source: HF `config.json` pages for `deepseek-community/deepseek-vl-7b-base` and `deepseek-community/deepseek-vl-7b-chat`.

Both native hybrid checkpoints inspected report:

- `architectures`: `["DeepseekVLHybridForConditionalGeneration"]`
- `model_type`: `deepseek_vl_hybrid`
- `torch_dtype`: `float16`
- `image_token_id`: `100015`

Text config:

- `model_type`: `llama`
- `hidden_size`: 4096
- `intermediate_size`: 11008
- `num_hidden_layers`: 30
- `num_attention_heads`: 32
- `num_key_value_heads`: 32
- `head_dim`: 128
- `max_position_embeddings`: 16384
- `vocab_size`: 102400
- `hidden_act`: `silu`
- `attention_bias`: false
- `mlp_bias`: false
- `rms_norm_eps`: `1e-6`
- `rope_theta`: 10000.0
- `rope_scaling`: null
- `use_cache`: true

Low-res vision config:

- `model_type`: `siglip_vision_model`
- `image_size`: 384
- `patch_size`: 16
- `hidden_size`: 1024
- `intermediate_size`: 4096
- `num_hidden_layers`: 24
- `num_attention_heads`: 16
- `hidden_act`: `gelu`
- `layer_norm_eps`: `1e-6`
- `vision_use_head`: false

High-res vision config:

- `model_type`: `sam_vision_model`
- `image_size`: 1024
- `patch_size`: 16
- `hidden_size`: 768
- `intermediate_size`/`mlp_dim`: 3072
- `num_hidden_layers`: 12
- `num_attention_heads`: 12
- `output_channels`: 256
- `global_attn_indexes`: `[2, 5, 8, 11]`
- `window_size`: 14
- `qkv_bias`: true
- `use_abs_pos`: true
- `use_rel_pos`: true

Processor/preprocessor:

- `processor_class`: `DeepseekVLHybridProcessor`
- `num_image_tokens`: 576
- `image_token`: `<image_placeholder>` in tokenizer config, id 100015
- Low-res image processor size: 384 x 384, mean/std `[0.5,0.5,0.5]`, resample 2, rescale factor `1/255`
- High-res image processor size: 1024 x 1024, CLIP mean/std, bicubic resample 3
- Both image branches are channel-first tensors after preprocessing.

## Non-hybrid contrast

`deepseek-community/deepseek-vl-1.3b-base` reports `model_type: deepseek_vl`, not `deepseek_vl_hybrid`. It should not be admitted by this report's first runtime target even though it shares broad DeepSeek-VL processor conventions.
