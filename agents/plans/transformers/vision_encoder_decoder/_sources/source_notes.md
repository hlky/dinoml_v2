# vision_encoder_decoder source notes

Source basis:

- Local Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family source directory: `transformers/src/transformers/models/vision_encoder_decoder`

Local files inspected:

- `configuration_vision_encoder_decoder.py`
- `modeling_vision_encoder_decoder.py`
- `__init__.py`

Supporting local files inspected for representative delegated bodies:

- `transformers/src/transformers/models/vit/modeling_vit.py`
- `transformers/src/transformers/models/vit/image_processing_vit.py`
- `transformers/src/transformers/models/trocr/modeling_trocr.py`
- `transformers/src/transformers/models/trocr/processing_trocr.py`
- `transformers/src/transformers/models/gpt2/modeling_gpt2.py`
- `transformers/src/transformers/generation/utils.py`

Wrapper-owned behavior confirmed from source:

- `VisionEncoderDecoderConfig` requires nested `encoder` and `decoder` config objects and reconstructs them through `AutoConfig.for_model(...)`.
- `from_encoder_decoder_configs(...)` mutates the decoder config to `is_decoder=True` and `add_cross_attention=True`.
- `VisionEncoderDecoderModel` instantiates an encoder with `AutoModel` and a decoder with `AutoModelForCausalLM`.
- The wrapper rejects encoders with output embeddings / LM heads.
- Top-level `config.tie_word_embeddings` is forced to `False`; decoder-internal tying remains delegated to the decoder family.
- Optional bridge projection `enc_to_dec_proj = Linear(encoder.hidden_size -> decoder.hidden_size)` is inserted only when encoder and decoder hidden sizes differ and decoder `cross_attention_hidden_size` is absent.
- Forward path owns `pixel_values -> encoder(...)`, optional tuple-to-`BaseModelOutput`, optional bridge projection, label shifting, and forwarding `encoder_hidden_states` plus `past_key_values` to the decoder.
- The wrapper sets `encoder_attention_mask = None`.
- Generation is inherited from `GenerationMixin`; encoder-output staging and cache preparation are generic generation infrastructure, not custom family code.

