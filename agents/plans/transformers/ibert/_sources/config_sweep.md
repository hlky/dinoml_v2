# I-BERT representative config sweep

All values below come from accessible Hugging Face `config.json` files unless noted.

| Model id | Architecture | Task tag | Hidden | Layers | Heads | Head dim | FFN | Vocab | Max pos | Type vocab | `quant_mode` | `force_dequant` | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `kssteven/ibert-roberta-base` | `IBertForMaskedLM` | fill-mask | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1 | false | omitted -> source default `none` | Common base checkpoint. |
| `kssteven/ibert-roberta-large` | `IBertForMaskedLM` | fill-mask | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | false | omitted -> source default `none` | Larger topology. |
| `kssteven/ibert-roberta-large-mnli` | `IBertForSequenceClassification` | text-classification | 1024 | 24 | 16 | 64 | 4096 | 50265 | 514 | 1 | false | omitted -> source default `none` | 3-way MNLI labels. |
| `DunnBC22/ibert-roberta-base-finetuned-WikiNeural` | `IBertForTokenClassification` | token-classification | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1 | false | `none` | 9 labels, `torch_dtype=float32`, historical `position_embedding_type=absolute` not read by current source. |
| `VitaliiVrublevskyi/ibert-roberta-base-finetuned-mrpc` | `IBertForSequenceClassification` | text-classification | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1 | false | `none` | `problem_type=single_label_classification`; `num_labels` inferred by config loader from label maps/defaults, not visible as `num_labels` field here. |
| `elayat/ibert-roberta-base-finetuned-imdb` | `IBertForMaskedLM` | fill-mask | 768 | 12 | 12 | 64 | 3072 | 50265 | 514 | 1 | false | `none` | Fine-tuned config still advertises masked-LM architecture. |

Config defaults from current source when fields are omitted:

| Field | Source default | Impact |
|---|---|---|
| `force_dequant` | `"none"` | Quantized GELU, softmax, and layernorm stay on integer approximation path if `quant_mode=True`. |
| `tie_word_embeddings` | `True` | Masked-LM decoder weight aliases the word embedding weight through `_tied_weights_keys`. |
| `layer_norm_eps` | `1e-12` | Public RoBERTa checkpoints override this to `1e-5`. |
| `max_position_embeddings` | `512` | Public RoBERTa checkpoints override this to `514` to account for RoBERTa special/pad offset behavior. |
| `type_vocab_size` | `2` | Public RoBERTa checkpoints override this to `1`. |
