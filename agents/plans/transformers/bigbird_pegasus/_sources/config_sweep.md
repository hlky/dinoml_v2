# BigBird-Pegasus Config Sweep Notes

Transformers source commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Fetched from Hugging Face on 2026-05-13:

| model id | config | tokenizer config | generation config | notes |
|---|---|---|---|---|
| `google/bigbird-pegasus-large-arxiv` | fetched | fetched | fetched | Official public config. |
| `google/bigbird-pegasus-large-pubmed` | fetched | fetched | fetched | Official public config. |
| `google/bigbird-pegasus-large-bigpatent` | fetched | fetched | fetched | Official public config. |
| `google/bigbird-pegasus-large-wikihow` | 401 | 401 | 401 | Gated or unavailable without authentication; not used for facts. |

The three accessible model configs are operator-identical in the inspected
fields: `BigBirdPegasusForConditionalGeneration`, `d_model=1024`,
`encoder_layers=16`, `decoder_layers=16`, `encoder_attention_heads=16`,
`decoder_attention_heads=16`, `encoder_ffn_dim=4096`,
`decoder_ffn_dim=4096`, `max_position_embeddings=4096`,
`attention_type="block_sparse"`, `block_size=64`,
`num_random_blocks=3`, `use_bias=false`, `scale_embedding=true`,
`vocab_size=96103`, `use_cache=true`.

Legacy generation fields (`num_beams=5`, `length_penalty=0.8`,
`max_length=256`) appear in the old `config.json` snapshots and are also present
in the fetched `generation_config.json`.

Tokenizer configs name `PegasusTokenizer`, `model_max_length=4096`, special
tokens `<pad>`, `</s>`, `<unk>`, `<s>`, `[MASK]`, `[SEP]`, `[CLS]`, and
`offset=0` in these checkpoint snapshots. The current Pegasus tokenizer source
defaults differ (`mask_token="<mask_2>"`, `mask_token_sent="<mask_1>"`,
`offset=103`), so DinoML should treat tokenizer config as the authority for
runtime text packing when tokenization is included.
