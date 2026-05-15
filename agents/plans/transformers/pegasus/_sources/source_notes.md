# Pegasus source notes

Local Transformers checkout:

```text
transformers
commit b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
```

Primary files:

- `src/transformers/models/pegasus/configuration_pegasus.py`
- `src/transformers/models/pegasus/modeling_pegasus.py`
- `src/transformers/models/pegasus/tokenization_pegasus.py`
- shared cache/mask helpers: `src/transformers/cache_utils.py`, `src/transformers/masking_utils.py`

Key source line anchors from `modeling_pegasus.py`:

- `PegasusSinusoidalPositionalEmbedding`: lines 74-104.
- eager attention math: lines 108-135.
- `PegasusAttention`: lines 138-253.
- encoder layer: lines 256-311.
- decoder layer: lines 313-404.
- encoder embeddings/mask/layer loop: lines 425-544.
- decoder cache initialization/masks/layer loop: lines 546-700.
- seq2seq model wrapper: lines 703-838.
- conditional generation head: lines 844-999.
- causal LM wrapper and `logits_to_keep`: lines 1016-1130.

Representative config snapshots saved beside this note:

- `google__pegasus-large.config.json`
- `google__pegasus-xsum.config.json`
- `google__pegasus-cnn_dailymail.config.json`
- `google__pegasus-arxiv.config.json`
- `google__pegasus-pubmed.config.json`
- `hf-internal-testing__tiny-random-PegasusModel.config.json`
