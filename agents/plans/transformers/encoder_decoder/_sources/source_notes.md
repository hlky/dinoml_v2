# encoder_decoder source notes

Local source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `transformers/src/transformers/models/encoder_decoder`

Key local source facts:

- `configuration_encoder_decoder.py`
  - `EncoderDecoderConfig.model_type = "encoder-decoder"`.
  - `sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}`.
  - Requires both serialized `encoder` and `decoder` sub-configs.
  - `from_encoder_decoder_configs()` mutates decoder config with `is_decoder=True` and `add_cross_attention=True`.
- `modeling_encoder_decoder.py`
  - `EncoderDecoderModel` wraps `AutoModel` for the encoder and `AutoModelForCausalLM` for the decoder.
  - `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder` child module; generation code has a compatibility branch for this pattern.
  - Constructor rejects decoders whose forward signature lacks `encoder_hidden_states`.
  - Constructor rejects encoders with output embeddings / LM head.
  - If `encoder.hidden_size != decoder.hidden_size` and decoder does not declare `cross_attention_hidden_size`, the wrapper adds `enc_to_dec_proj = nn.Linear(encoder_hidden_size, decoder_hidden_size)`.
  - If `decoder.cross_attention_hidden_size` is set, it must equal `encoder.hidden_size`; wrapper projection is not added.
  - Forward calls encoder when `encoder_outputs` is absent, otherwise accepts caller-provided tuple/BaseModelOutput.
  - Forward passes `encoder_hidden_states`, source `attention_mask` as `encoder_attention_mask`, decoder inputs, `past_key_values`, and `use_cache` into the decoder.
  - `labels` path uses `shift_tokens_right`; inference can avoid this by passing `decoder_input_ids`/generation controller inputs.
- `generation/utils.py`
  - Encoder outputs are prepared once for generation and stored in `model_kwargs["encoder_outputs"]`.
  - Decoder start token handling prepends `decoder_start_token_id` when needed and updates `decoder_attention_mask`.
  - Generic validation has an explicit note for encoder-decoder wrappers with `base_model_prefix="encoder_decoder"` but no `self.encoder_decoder`.
- `cache_utils.py`
  - `EncoderDecoderCache` holds separate `self_attention_cache` and `cross_attention_cache`.
  - `is_updated[layer_idx]` tracks whether cross-attention K/V were computed.
  - `get_seq_length()` reports self-attention cache length.
  - `reorder_cache()` reorders both self and cross caches.
- BERT/BertGeneration representative decoder behavior:
  - Decoder self-attention updates `EncoderDecoderCache.self_attention_cache`.
  - Cross-attention computes query from decoder hidden states and key/value from `encoder_hidden_states`.
  - Cross-attention reuses cached cross K/V when `is_updated[layer_idx]` is true.

Representative configs saved in this directory:

- `patrickvonplaten__bert2bert-cnn_dailymail-fp16.config.json`
- `patrickvonplaten__bert2bert_cnn_daily_mail.config.json`
- `google__bert2bert_L-24_wmt_en_de.config.json`
- `mrm8488__bert2bert-mini_shared-question-generation.config.json`
- `mrm8488__bert2bert-medium_shared-question-generation.config.json`
- `mrm8488__bert2bert_shared-spanish-finetuned-summarization.config.json`
- `Callidior__bert2bert-base-arxiv-titlegen.config.json`
- `config_sweep_summary.json`

Public tiny/debug checkpoints tried but inaccessible without auth:

- `hf-internal-testing/tiny-random-EncoderDecoderModel`
- `hf-internal-testing/tiny-random-encoder-decoder`
- `patrickvonplaten/tiny-random-bert2bert`
- `sshleifer/tiny-random-encoder-decoder`
