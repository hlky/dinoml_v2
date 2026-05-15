# Pegasus-X Source Notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Describe: `v4.50.3-DeepSeek-3-4398-gb75feb2af6`

Inspected files:

- `src/transformers/models/pegasus_x/configuration_pegasus_x.py`
- `src/transformers/models/pegasus_x/modeling_pegasus_x.py`
- `src/transformers/models/pegasus_x/__init__.py`

Key source anchors:

- `configuration_pegasus_x.py`: `PegasusXConfig` defaults include `d_model=1024`, `encoder_layers=16`, `decoder_layers=16`, `encoder_attention_heads=16`, `decoder_attention_heads=16`, `encoder_ffn_dim=4096`, `decoder_ffn_dim=4096`, `max_position_embeddings=16384`, `block_size=512`, `num_global_tokens=32`, `stagger_local_blocks=True`, and `use_cache=True`.
- `modeling_pegasus_x.py`: `PegasusXSinusoidalPositionalEmbedding` generates sin/cos position vectors at runtime from `position_ids` and `max_scale=10000`.
- `modeling_pegasus_x.py`: `PegasusXGlobalLocalAttention` is encoder-only and computes two custom attention paths with einsums: global queries attend to all global plus local tokens, and local queries attend to global tokens plus tokens in their own local block.
- `modeling_pegasus_x.py`: `PegasusXEncoder.forward` pads token states and masks to a multiple of `block_size` before encoder layers, creates learned global-token embeddings, then trims the token output back to the original sequence length.
- `modeling_pegasus_x.py`: odd encoder layers stagger local blocks when `stagger_local_blocks=True` by padding half a block on both sides before attention and slicing it away afterwards.
- `modeling_pegasus_x.py`: `PegasusXAttention` is BART-style MHA used by decoder self-attention and cross-attention, with `EncoderDecoderCache` support.
- `modeling_pegasus_x.py`: model metadata advertises `_supports_flash_attn=True`, `_supports_sdpa=False`, and `_supports_flex_attn=True`; this applies through the decoder `PegasusXAttention` interface, not the encoder custom global/local attention.
- `modeling_pegasus_x.py`: `PegasusXModel` ties encoder and decoder embeddings through `shared`; `PegasusXForConditionalGeneration` ties `lm_head.weight` to `model.shared.weight`.

Config snapshots saved:

- `google_pegasus-x-base_config.json`
- `google_pegasus-x-large_config.json`
- `google_pegasus-x-base-arxiv_config.json`
- `pszemraj_pegasus-x-large-book-summary_config.json`
- `twigs_pegasus-x-large-8192-pubmed_config.json`
- `hf-tiny-random-PegasusXForConditionalGeneration_config.json`
- `google_pegasus-x-base_generation_config.json`
- `google_pegasus-x-large_generation_config.json`
- `google_pegasus-x-base_tokenizer_config.json`
- `google_pegasus-x-large_tokenizer_config.json`

