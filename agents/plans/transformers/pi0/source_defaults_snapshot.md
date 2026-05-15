# PI0 Source Defaults Snapshot

Source basis: `transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

These values are source defaults from `src/transformers/models/pi0/configuration_pi0.py`, not a downloaded checkpoint `config.json`.

| Field | Effective default |
| --- | --- |
| `model_type` | `pi0` |
| Primary class | `PI0ForConditionalGeneration` |
| `chunk_size` | 50 |
| `max_state_dim` | 32 |
| `max_action_dim` | 32 |
| `num_inference_steps` | 10 |
| Time sampling | Beta alpha 1.5, beta 1.0, scale 0.999, offset 0.001 |
| Timestep periods | `min_period=0.004`, `max_period=4.0` |
| VLM default | PaliGemma: Gemma text + SigLIP vision |
| VLM text default | hidden 2048, layers 18, intermediate 16384, heads 8, KV heads 1, vocab 257152 |
| VLM vision default | hidden 1152, layers 27, intermediate 4304, heads 16, patch 14, image 224, `vision_use_head=False` |
| VLM projection | 1152 -> 2048 |
| Image token id | 257152 in PI0 default PaliGemma config |
| DiT default | Gemma decoder body used as action denoiser |
| DiT dimensions | hidden 1024, layers 18, intermediate 4096, heads 8, KV heads 1, head_dim 256 |
| Forced attention flags | `dit_config.is_causal=True`, `dit_config.use_bidirectional_attention=True`, `vlm_config.text_config.use_bidirectional_attention=True` |

Processor defaults from `processing_pi0.py` and `image_processing_pi0.py`:

| Field | Effective default |
| --- | --- |
| Text padding | `padding="max_length"`, `max_length=48`, `padding_side="right"` |
| Image token string | `<image>` |
| Added extra tokens | 1024 `<loc0000>`... plus 128 `<seg000>`... tokens |
| Image preprocessing | resize, rescale, normalize, RGB conversion, pad enabled |
| Image layout emitted | `[B, max_cameras, 3, H, W]` |
| Default padded image size | 224 x 224 |
| State/action normalization | mean/std arrays in processor, then zero-pad to `max_state_dim` |

Instantiation trap: `PI0ImageProcessor` class defaults declare `size={"max_height":224,"max_width":224}` and `pad_size={"height":224,"width":224}`, while `PI0Processor.__init__` reads `image_processor.size["height"]` and `["width"]`. Saved processor configs or the base image backend may normalize this, but bare source-default construction should be validated before treating `size` as a stable ABI.
