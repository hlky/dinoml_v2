# FNet source hotspots

Local Transformers checkout: `transformers` at commit
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Files inspected

- `src/transformers/models/fnet/configuration_fnet.py`
- `src/transformers/models/fnet/modeling_fnet.py`
- `src/transformers/models/fnet/tokenization_fnet.py`
- `src/transformers/activations.py` for `gelu_new`
- `tests/models/fnet/test_modeling_fnet.py`

## Modeling hotspots

- `FNetEmbeddings`: word, absolute position, and token-type embeddings;
  `LayerNorm`; then an always-present `Linear(hidden_size -> hidden_size)`
  projection and dropout.
- `FNetBasicFourierTransform`: default CPU/GPU path uses
  `torch.fft.fftn(hidden_states, dim=(1, 2)).real`. This mixes sequence and
  hidden axes and returns only the real component.
- TPU optimization path: if `use_tpu_fourier_optimizations=True` and
  `max_position_embeddings <= 4096`, source may precompute complex DFT matrices
  with SciPy and apply `einsum("bij,jk,ni->bnk", ...)` after casting input to
  complex64. If SciPy is unavailable, or max position embeddings is larger, it
  falls back to repeated one-axis FFTs.
- `FNetLayer`: Fourier mixer with residual LayerNorm, then BERT-style FFN
  `Linear(H -> I)`, activation, `Linear(I -> H)`, dropout, residual LayerNorm.
- `FNetModel.forward`: no `attention_mask` argument. It rejects both
  `input_ids` and `inputs_embeds` being passed together, creates missing
  token-type ids from a zero buffer, and validates TPU short sequence length
  when the optimized path is enabled.
- Heads: pretraining MLM+NSP, masked LM, NSP, sequence classification,
  multiple choice, token classification, and QA.
- Weight tying: masked-LM/pretraining decoder weight aliases
  `fnet.embeddings.word_embeddings.weight`; decoder bias aliases the explicit
  prediction bias parameter through `_tied_weights_keys`.

## Config sweep

| Model id | Arch | H | I | Layers | Vocab | Type vocab | Max pos | Act | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `google/fnet-base` | `FNetForPreTraining` | 768 | 3072 | 12 | 32000 | 4 | 512 | `gelu_new` | Official pretraining checkpoint; includes historical `actual_seq_length`, `use_fft`, `use_latest` not read by current source. |
| `google/fnet-large` | `FNetForPreTraining` | 1024 | 4096 | 24 | 32000 | 4 | 512 | `gelu_new` | Same operator structure as base, larger H/I/layers. |
| `gchhablani/fnet-base-finetuned-sst2` | `FNetForSequenceClassification` | 768 | 3072 | 12 | 32000 | 4 | 512 | `gelu_new` | Two-label classifier; source head is pooler dropout plus `Linear(H -> num_labels)`. |
| `gchhablani/fnet-large-finetuned-mnli` | `FNetForSequenceClassification` | 1024 | 4096 | 24 | 32000 | 4 | 512 | `gelu_new` | Three-label classifier. |
| `hf-internal-testing/tiny-random-FNetModel` | `FNetModel` | 32 | 37 | 5 | 32000 | 16 | 512 | `gelu` | Debug shape variant; useful for small parity tests and activation variation. |

