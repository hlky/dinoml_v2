# PLBart source/config notes

Local Transformers checkout: `X:/H/transformers` at `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Local source snapshots copied from:

- `src/transformers/models/plbart/configuration_plbart.py`
- `src/transformers/models/plbart/modeling_plbart.py`
- `src/transformers/models/plbart/modular_plbart.py`
- `src/transformers/models/plbart/tokenization_plbart.py`

Representative Hugging Face configs fetched from official repos:

| Model id | Repo sha from HF API | Config snapshot |
|---|---|---|
| `uclanlp/plbart-base` | `cf5287241fcff3819f6ade49635dc2d77efee032` | `uclanlp_plbart-base_config.json` |
| `uclanlp/plbart-large` | `d296275f5a15b9b971dec79d06410852f0c8635d` | `uclanlp_plbart-large_config.json` |
| `uclanlp/plbart-java-en_XX` | `b1ca7ca4c18a8b23c9eff5d25eff783d23e07d15` | `uclanlp_plbart-java-en_XX_config.json` |
| `uclanlp/plbart-python-en_XX` | `48bf6e4889bdb9bafd12381a4e9a9a1e0fe224eb` | `uclanlp_plbart-python-en_XX_config.json` |

Repository file sweep:

- Reachable repos expose `.gitattributes`, `config.json`, `pytorch_model.bin`, and `sentencepiece.bpe.model`.
- `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`, and `generation_config.json` returned 404 for the four reachable repos during this audit.
- `uclanlp/plbart-multi_task` returned 401 Unauthorized for `config.json`; treat as gated/out of current source-basis scope unless access is provided.

Config observations:

- Base/java/python configs are structurally identical: `d_model=768`, 6 encoder layers, 6 decoder layers, 12 heads, FFN 3072, vocab 50005, max positions 1024, GELU, `scale_embedding=true`, `attention_dropout=0.1`.
- Large changes model width/depth only: `d_model=1024`, 12/12 layers, 16 heads, FFN 4096, same vocab/max positions/activation. It also records `torch_dtype="float32"` and `gradient_checkpointing=false`.
- No reachable checkpoint config changes attention topology, activation, vocab size, cache support, or tokenizer language-code set.
