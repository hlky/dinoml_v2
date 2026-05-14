# I-BERT source notes

Audit date: 2026-05-13

## Local source basis

- Transformers checkout: `X:/H/transformers`
- Local commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/ibert`

Inspected files:

- `configuration_ibert.py`
- `modeling_ibert.py`
- `quant_modules.py`
- `__init__.py`

## Key local source facts

- `IBertConfig` defaults to BERT-like dimensions: vocab 30522, hidden 768, layers 12, heads 12, intermediate 3072, max positions 512, token types 2, `quant_mode=False`, `force_dequant="none"`, tied embeddings enabled.
- Public checkpoints inspected are RoBERTa-shaped: vocab 50265, max positions 514, type vocab 1, pad/bos/eos ids 1/0/2, tokenizer class `RobertaTokenizer`.
- Source encoder is encoder-only despite inherited doc wording mentioning decoder behavior. `IBertEncoder` explicitly sets cross attentions to `None`, and no cross-attention modules are constructed.
- `IBertIntermediate` rejects `hidden_act != "gelu"`.
- `IBertPreTrainedModel.resize_token_embeddings()` raises `NotImplementedError`.
- Main quantized modules are `QuantEmbedding`, `QuantAct`, `QuantLinear`, `IntGELU`, `IntSoftmax`, and `IntLayerNorm`.
- Quantized linear weights are symmetric int8 by default. Encoder `QuantLinear` layers use per-output-channel weight scales and 32-bit quantized bias.
- Quantized activations are globally scaled by default; per-channel activation quantization raises `NotImplementedError`.
- Quantized nonlinear modules can be disabled selectively with `force_dequant in {"gelu", "softmax", "layernorm", "nonlinear"}`.
- The source mutates quantization buffers during forward: weight integer buffers/scales, activation min/max/scales in training, and `IntLayerNorm.dim_sqrt`.
- `IntLayerNorm` has a source comment that integer sqrt should replace the current `torch.sqrt(var_int)` path.

## Representative config sources

Accessible configs:

- `https://huggingface.co/kssteven/ibert-roberta-base/raw/main/config.json`
- `https://huggingface.co/kssteven/ibert-roberta-large/raw/main/config.json`
- `https://huggingface.co/kssteven/ibert-roberta-large-mnli/raw/main/config.json`
- `https://huggingface.co/DunnBC22/ibert-roberta-base-finetuned-WikiNeural/raw/main/config.json`
- `https://huggingface.co/VitaliiVrublevskyi/ibert-roberta-base-finetuned-mrpc/raw/main/config.json`
- `https://huggingface.co/elayat/ibert-roberta-base-finetuned-imdb/raw/main/config.json`

Inaccessible or absent at checked paths:

- `https://huggingface.co/kssteven/ibert-roberta-base-mnli/raw/main/config.json` returned 401.
- `https://huggingface.co/kssteven/ibert-roberta-base-squad2/raw/main/config.json` returned 401.
- `https://huggingface.co/kssteven/ibert-roberta-large-squad2/raw/main/config.json` returned 401.

HF API metadata sampled:

- `kssteven/ibert-roberta-base`: sha `4f98e9110b04a8958444d3af8ed39287834fbb90`, pipeline `fill-mask`, gated `False`.
- `kssteven/ibert-roberta-large`: sha `202dedcec60c0aece82a3c4d424cb7505efcb31f`, pipeline `fill-mask`, gated `False`.
- `kssteven/ibert-roberta-large-mnli`: sha `5ec852d6567202390f5bcc558de70e8d23ea7d10`, pipeline `text-classification`, gated `False`.
- `DunnBC22/ibert-roberta-base-finetuned-WikiNeural`: sha `2d0881b5d248b44a5a534c3e8a811eddc84ae70b`, pipeline `token-classification`, gated `False`.
- `VitaliiVrublevskyi/ibert-roberta-base-finetuned-mrpc`: sha `9075737837addbdb756ea8db348efe9356484888`, pipeline `text-classification`, gated `False`.
- `elayat/ibert-roberta-base-finetuned-imdb`: sha `15c30e34a7ae8a4bb2bfa74f1cea29a45160e610`, pipeline `fill-mask`, gated `False`.
