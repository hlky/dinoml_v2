# seamless_m4t_v2 source notes

Source checkout:
- `X:/H/transformers`
- commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Primary in-library files:
- `src/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py`
- `src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py`
- `src/transformers/models/seamless_m4t_v2/convert_fairseq2_to_hf.py`

Processor/tokenizer files reused by the v2 modeling docs and conversion code:
- `src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py`
- `src/transformers/models/seamless_m4t/processing_seamless_m4t.py`
- `src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py`

Compact HF snapshots saved in this folder:
- `facebook__seamless-m4t-v2-large__config.json`
- `facebook__seamless-m4t-v2-large__preprocessor_config.json`
- `facebook__seamless-m4t-v2-large__tokenizer_config.json`
- `hf-internal-testing__tiny-random-SeamlessM4Tv2Model__config.json`
- `hf-internal-testing__tiny-random-SeamlessM4Tv2Model__preprocessor_config.json`
- `panoyo9829__seamless-m4t-v2-large-fp16__config.json`
- `jaman21__seamless-m4t-v2-t2tt__config.json`
- `jaman21__seamless-m4t-v2-t2st__config.json`
- `Geneline-X__seamless-m4t-v2-sunbird-multilingual-v1__config.json`
- `WueNLP__seamless-m4t-v2-large-speech-encoder__config.json`

Generation config note:
- `generation_config.json` for the official and derivative full checkpoints was accessible, but it is about 10 MB because it carries language-code maps such as `text_decoder_lang_to_code_id`, `t2u_lang_code_to_id`, `vocoder_lang_code_to_id`, `id_to_text`, and `char_to_id`. The large files were not retained as snapshots; the report records the ABI fields instead.

Observed gaps:
- `WueNLP/seamless-m4t-v2-large-speech-encoder` uses `auto_map` remote/custom code and has no `generation_config.json` or tokenizer config at the tested URLs. Treat it as a custom speech-encoder derivative, not native `seamless_m4t_v2` parity.
- `Geneline-X/seamless-m4t-v2-sunbird-multilingual-v1` did not expose `preprocessor_config.json` at the tested URL.
