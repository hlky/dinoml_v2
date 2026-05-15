# Megatron-BERT source notes

Workspace source basis:

- Transformers checkout: `transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family dir: `transformers/src/transformers/models/megatron_bert`
- Main files inspected:
  - `configuration_megatron_bert.py`
  - `modeling_megatron_bert.py`
  - `convert_megatron_bert_checkpoint.py`
  - `tests/models/megatron_bert/test_modeling_megatron_bert.py`
  - `docs/source/en/model_doc/megatron-bert.md`

Important source observations:

- `MegatronBertConfig` defaults are the NVIDIA 345M shape: vocab 29056, hidden 1024, layers 24, heads 16, intermediate 4096, max positions 512, type vocab 2, GELU, `layer_norm_eps=1e-12`, `use_cache=True`, `is_decoder=False`, `add_cross_attention=False`.
- Embeddings use word + token type + absolute position embeddings, then dropout. Embedding LayerNorm is explicitly commented out.
- Attention block applies `LayerNorm(hidden)` before Q/K/V projection, dense attention, output projection, dropout, and residual add.
- MLP block applies a second per-layer LayerNorm before dense -> activation -> dense -> dropout -> residual add.
- Encoder applies a final LayerNorm after all hidden layers.
- Q/K/V are separate biased `nn.Linear(hidden_size, hidden_size)` modules. `head_dim = hidden_size // num_attention_heads`; source only raises on non-divisible hidden size when the config lacks an `embedding_size` attribute, but still computes integer division.
- Attention is dense scaled dot-product with additive extended mask, softmax over keys, dropout, and context matmul. No RoPE, ALiBi, relative bias, sliding window, block sparse, or packed varlen path appears in this native source.
- Decoder/cross-attention/cache paths exist through `is_decoder`, `add_cross_attention`, and `Cache`/`EncoderDecoderCache`, but the normal Megatron-BERT checkpoints are bidirectional encoder/MLM/NSP style.
- Heads implemented: base encoder/pooler, pretraining MLM+NSP, causal LM, masked LM, next sentence prediction, sequence classification, multiple choice, token classification, question answering.
- MLM/pretraining/causal LM tie `cls.predictions.decoder.weight` to `bert.embeddings.word_embeddings.weight`; prediction head also has an output-only vocab bias.
- Hosted NVIDIA repos `nvidia/megatron-bert-{cased,uncased}-345m` currently expose tokenizer/README files, not hosted `config.json` or model weights. README documents conversion from NGC checkpoint, which writes local config and `pytorch_model.bin`.

Representative config observations from accessible HF repos:

| Source | Shape notes | Task/head | Notes |
| --- | --- | --- | --- |
| Source default / converted NVIDIA 345M | H=1024, L=24, heads=16, FFN=4096, max pos=512, vocab default 29056 | base/MLM/NSP | Conversion script overrides vocab from `lm_head.bias` for actual checkpoint. |
| `nvidia/megatron-bert-cased-345m` | no hosted `config.json` | tokenizer only | README says converted checkpoint creates config locally. |
| `nvidia/megatron-bert-uncased-345m` | no hosted `config.json` | tokenizer only | tokenizer config has `do_lower_case=true`. |
| `KBLab/megatron-bert-base-swedish-cased-600k` | H=768, L=12, heads=12, FFN=3072, vocab=64128 | `MegatronBertForMaskedLM` | Hosted config includes historical `position_embedding_type=absolute` ignored by current source. |
| `KBLab/megatron-bert-large-swedish-cased-165k` | H=1024, L=24, heads=16, FFN=4096, vocab=64128 | `MegatronBertForMaskedLM` | Large Swedish shape matches 345M hidden/layer/head width but larger vocab. |
| `IDEA-CCNL/Erlangshen-MegatronBert-1.3B` | H=2048, L=24, heads=8, head_dim=256, FFN=8192, vocab=21248 | base `AutoModel` | `hidden_act=gelu_new`; `use_cache=false`. |
| `IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese` | H=2560, L=48, heads=40, head_dim=64, FFN=10240, vocab=21248 | `MegatronBertForMaskedLM` | Larger depth; otherwise same dense encoder operators. |
| `EMBO/BioMegatron345mUncased` | H=1024, L=24, heads=16, FFN=4096, vocab=30592 | base `AutoModel` | `hidden_act=gelu_new`, `initializer_range=0.2`; initializer does not affect inference graph. |
| `hf-tiny-model-private/tiny-random-MegatronBertForMaskedLM` | H=64, L=5, heads=4, FFN=37, type vocab=16 | tiny MLM | Useful for shape/debug configs; `embedding_size` is present but not consumed as an embedding projection. |
