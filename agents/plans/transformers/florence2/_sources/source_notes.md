# Florence2 source notes

Local source basis:

- Transformers checkout: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `X:/H/transformers/src/transformers/models/florence2`

Files inspected:

- `configuration_florence2.py`: generated from `modular_florence2.py`; config class uses `Florence2VisionConfig` plus nested AutoConfig text model, defaulting to BART.
- `modeling_florence2.py`: generated runtime source; imports generated classes from `modular_florence2.py`.
- `modular_florence2.py`: authoritative edit source for Transformers; generated files warn not to edit them directly.
- `processing_florence2.py`: generated processor and postprocessor; prompt construction and structured output parsing live here.

Representative configs saved in this folder:

- `community_base_config.json`
- `community_base_ft_config.json`
- `community_large_config.json`
- `community_large_ft_config.json`
- `community_base_preprocessor_config.json`
- `community_base_processor_config.json`
- `community_base_generation_config.json`

Official Microsoft raw configs were also inspected live but not saved as the main basis because they are older remote-code style configs (`transformers_version` around `4.41.0.dev0`) and keep legacy vision fields such as `dim_embed` without the newer normalized `embed_dim` field. Current `florence-community/*` configs are normalized for in-library `model_type="florence2"` and `model_type="florence_vision"`.

Source-derived sharp edges:

- Processor expands each image into exactly `image_seq_length` copies of the tokenizer image token, then appends BOS, prompt text, and EOS.
- Base/large processor config uses `image_seq_length=577`; the projector derives this as final DaViT feature grid `24 * 24 = 576` plus one spatial pooled token for 768x768 input.
- Model splices image features into text embeddings with boolean `masked_scatter`; the placeholder token count must equal the flattened projected image feature element count.
- Vision source is NCHW for conv/depthwise conv but repeatedly switches to token/NHWC-like forms for LayerNorm, window attention, and MLPs.
- Florence2 delegates the text body to BART through `AutoModel.from_config(config.text_config)`.
