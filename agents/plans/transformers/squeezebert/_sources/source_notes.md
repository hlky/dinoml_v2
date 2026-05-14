# SqueezeBERT Source Notes

Local Transformers checkout: `X:/H/transformers`

Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Copied local source snapshots:

- `configuration_squeezebert.py`
- `modeling_squeezebert.py`
- `tokenization_squeezebert.py`
- `test_modeling_squeezebert.py`

HF config snapshots fetched from `https://huggingface.co/{model_id}/raw/main/config.json`:

- `squeezebert/squeezebert-uncased`
- `squeezebert/squeezebert-mnli`
- `squeezebert/squeezebert-mnli-headless`
- `mrm8488/squeezebert-finetuned-squadv2`
- `hf-tiny-model-private/tiny-random-SqueezeBertModel`
- `hf-tiny-model-private/tiny-random-SqueezeBertForTokenClassification`

Key source anchors:

- `SqueezeBertEmbeddings`: word/position/token-type embeddings, sum, LayerNorm, dropout.
- `SqueezeBertEncoder`: permutes `[B, S, C] -> [B, C, S]`, runs repeated `SqueezeBertModule`, then permutes back.
- `SqueezeBertSelfAttention`: Q/K/V are `nn.Conv1d(kernel_size=1, groups={q,k,v}_groups)` on NCW layout; attention is eager `matmul -> scale -> additive mask -> softmax -> matmul`.
- `ConvDropoutLayerNorm`: pointwise Conv1d, dropout, residual add, channel LayerNorm through NWC temporary.
- `ConvActivation`: pointwise grouped Conv1d plus activation.
- Task heads: masked LM, sequence classification, multiple choice, token classification, question answering.

