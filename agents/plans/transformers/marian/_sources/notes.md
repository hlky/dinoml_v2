# Marian source/config notes

Local Transformers checkout:

- Path: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `transformers/src/transformers/models/marian`

Local files inspected:

- `configuration_marian.py`
- `modeling_marian.py`
- `tokenization_marian.py`
- Shared helpers: `masking_utils.py`, `cache_utils.py`

Fetched Hugging Face snapshots under this directory:

- `Helsinki-NLP__opus-mt-en-de`
- `Helsinki-NLP__opus-mt-fr-en`
- `Helsinki-NLP__opus-mt-ROMANCE-en`
- `Helsinki-NLP__opus-mt-en-ROMANCE`
- `sshleifer__tiny-marian-en-de`

Each fetched directory contains the accessible `config.json`, `tokenizer_config.json`, and, when present, `generation_config.json`. Raw `vocab.json` files were fetched for inspection, then omitted from the saved snapshot to avoid storing multi-megabyte tokenizer vocabularies in this audit; the extracted language-code counts are recorded below. `target_vocab.json` returned 404 for all five sampled checkpoints, so the sweep did not find a live separate-vocabulary checkpoint. Source still implements `separate_vocabs=True`; DinoML should treat that as a guarded load-time branch if a checkpoint includes `target_vocab.json`.

Observed checkpoint summary:

| Model id | d_model | Enc/dec layers | Heads | FFN | Activation | Vocab | Max pos | Decoder start | Lang-code tokens |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `Helsinki-NLP/opus-mt-en-de` | 512 | 6/6 | 8/8 | 2048 | swish | 58101 | 512 | 58100 | 0 |
| `Helsinki-NLP/opus-mt-fr-en` | 512 | 6/6 | 8/8 | 2048 | swish | 59514 | 512 | 59513 | 0 |
| `Helsinki-NLP/opus-mt-ROMANCE-en` | 512 | 6/6 | 8/8 | 2048 | swish | 65001 | 512 | 65000 | 0 |
| `Helsinki-NLP/opus-mt-en-ROMANCE` | 512 | 6/6 | 8/8 | 2048 | swish | 65001 | 512 | 65000 | 47 |
| `sshleifer/tiny-marian-en-de` | 2 | 2/2 | 1/1 | 2 | swish | 58101 | 512 | 58100 | 0 |

Tokenizer/language notes:

- `MarianTokenizer` appends EOS to source/target sequences.
- Decoder starts from `decoder_start_token_id`, which is the pad token for sampled checkpoints.
- For non-separate vocabs, target-language control is represented as source-text prefix tokens matching `>>...<<` entries in `vocab.json`.
- The `en-ROMANCE` sample has 47 such language-code tokens in `vocab.json`; sampled examples include `>>fr<<`, `>>es<<`, `>>it<<`, `>>pt<<`, `>>ro<<`, `>>ca<<`, `>>gl<<`, `>>la<<`.
- `generation_config.json` for official samples sets `num_beams=4`, `decoder_start_token_id=<pad>`, `forced_eos_token_id=0`, and `bad_words_ids` containing pad.
