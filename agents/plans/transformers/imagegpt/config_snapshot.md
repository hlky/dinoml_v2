# ImageGPT config snapshot

Source: Hugging Face raw files fetched from official OpenAI repos on 2026-05-13.

| model id | repo sha from HF API | n_layer | n_embd | n_head | head_dim | n_positions | vocab_size | processor size | clusters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai/imagegpt-small | 10c8a8402cf80c0eaaf31fb5bc86012b34169481 | 24 | 512 | 8 | 64 | 1024 | 513 | 32 | 512x3 |
| openai/imagegpt-medium | d0416c58f900f4bf3b3bd089c835a518478139c1 | 36 | 1024 | 8 | 128 | 1024 | 513 | 32 | 512x3 |
| openai/imagegpt-large | c5e5de2c60e37a19b59730830a7dff526014b068 | 48 | 1536 | 16 | 96 | 1024 | 513 | 32 | 512x3 |

Shared config facts from `config.json`: `activation_function="quick_gelu"`, `n_inner=null` so MLP width is `4 * n_embd`, `scale_attn_weights=true`, `scale_attn_by_inverse_layer_idx=false`, `reorder_and_upcast_attn=false`, `tie_word_embeddings=false`, `use_cache=true`.

Shared processor facts from `preprocessor_config.json`: `feature_extractor_type="ImageGPTFeatureExtractor"`, `do_resize=true`, `do_normalize=true`, `resample=2`, `size=32`, `clusters` has 512 RGB cluster rows in normalized `[-1, 1]` space. The first cluster row is `[0.8866443634033203, 0.6618829369544983, 0.3891746401786804]`.

Observed inaccessible/gap URLs:

- https://huggingface.co/openai/imagegpt-small-32x32/raw/main/config.json returned 401.
- https://huggingface.co/openai/imagegpt-small-cifar10/raw/main/config.json returned 401.
