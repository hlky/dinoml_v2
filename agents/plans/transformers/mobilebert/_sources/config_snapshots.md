# MobileBERT Config Snapshots

Fetched on 2026-05-13 from Hugging Face raw `config.json` URLs, plus local
Transformers source at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Representative configs

| Model id | URL | Architecture | H | true H | emb | I | layers | heads | FFNs/block | vocab | notable fields |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| google/mobilebert-uncased | https://huggingface.co/google/mobilebert-uncased/raw/main/config.json | MobileBertForPreTraining | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | `trigram_input=true`, `normalization_type=no_norm`, `classifier_activation=false` |
| RedHatAI/mobilebert-uncased-finetuned-squadv1 | https://huggingface.co/RedHatAI/mobilebert-uncased-finetuned-squadv1/raw/794b132b63fef727b1710d873547db32b76ac833/config.json | MobileBertForQuestionAnswering | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | QA head, `attention_probs_dropout_prob=0.0`, `torch_dtype=float32` |
| mrm8488/mobilebert-finetuned-ner | https://huggingface.co/mrm8488/mobilebert-finetuned-ner/raw/refs%2Fpr%2F1/config.json | MobileBertForTokenClassification | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 30522 | 8 NER labels |
| vumichien/emo-mobilebert | https://huggingface.co/vumichien/emo-mobilebert/raw/main/config.json | MobileBertForSequenceClassification | 512 | 128 | 128 | 512 | 24 | 4 | 4 | 2016 | 4 labels, `classifier_activation=true`, `max_length=128` |
| optimum-intel-internal-testing/tiny-random-MobileBertModel | https://huggingface.co/optimum-intel-internal-testing/tiny-random-MobileBertModel/raw/main/config.json | MobileBertModel | 64 | 128 | 32 | 37 | 5 | 4 | 4 | 1124 | tiny/random stress case: `true_hidden_size` exceeds `hidden_size`, `hidden_act=gelu` |

## Source defaults from `configuration_mobilebert.py`

```text
vocab_size=30522
hidden_size=512
num_hidden_layers=24
num_attention_heads=4
intermediate_size=512
hidden_act="relu"
hidden_dropout_prob=0.0
attention_probs_dropout_prob=0.1
max_position_embeddings=512
type_vocab_size=2
layer_norm_eps=1e-12
embedding_size=128
trigram_input=True
use_bottleneck=True
intra_bottleneck_size=128
use_bottleneck_attention=False
key_query_shared_bottleneck=True
num_feedforward_networks=4
normalization_type="no_norm"
classifier_activation=True
classifier_dropout=None
tie_word_embeddings=True
```

`true_hidden_size` is derived in `__post_init__`: if `use_bottleneck=True`,
`true_hidden_size=intra_bottleneck_size`; otherwise `true_hidden_size=hidden_size`.

## Source notes

- `tokenization_mobilebert.py` aliases `MobileBertTokenizer` and
  `MobileBertTokenizerFast` to BERT tokenization.
- `MobileBertPreTrainedModel` advertises `_supports_flash_attn=True` and
  `_supports_sdpa=True`, but the model remains noncausal encoder attention.
- The standard checkpoint has `classifier_activation=false`, so its pooler is
  `hidden[:, 0]` without dense+tanh. Source defaults still enable the pooler
  dense+tanh when the field is omitted.
- The MLM head is not a plain tied `Linear(hidden_size -> vocab_size)`: after a
  transform it multiplies hidden states by a concatenation of
  `decoder.weight.T` and a second dense weight with shape
  `[hidden_size - embedding_size, vocab_size]`, then adds decoder bias.
