# XLM-RoBERTa-XL Source Notes

Scope: `xlm_roberta_xl` only.

Transformers source checkout:

- Path: `X:/H/transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Local verification: `git rev-parse HEAD` returned the requested commit.

Primary source files:

- `src/transformers/models/xlm_roberta_xl/configuration_xlm_roberta_xl.py`
- `src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py`
- `src/transformers/models/xlm_roberta_xl/modular_xlm_roberta_xl.py`
- `src/transformers/models/xlm_roberta/tokenization_xlm_roberta.py`
- Supporting mask/backend references: `src/transformers/masking_utils.py`

Important source facts:

- `modeling_xlm_roberta_xl.py` is generated from `modular_xlm_roberta_xl.py`; future Transformers edits should target the modular file.
- `XLMRobertaXLConfig` defaults: hidden size 2560, 36 layers, 32 attention heads, intermediate size 10240, vocab size 250880, max positions 514, GELU, layer norm eps 1e-5, token type vocab size 1.
- Embeddings use word, token type, and learned absolute position embeddings, but no embedding LayerNorm. Position ids for `input_ids` are built from non-pad cumsum plus `pad_token_id`.
- Config snapshots from official HF repos include `position_embedding_type: "absolute"`, but the inspected XL source does not read `position_embedding_type` and has no relative-position branch.
- Encoder block is pre-LayerNorm for attention and MLP, has residual adds without immediate post-add LayerNorm, and applies one extra final encoder LayerNorm after all layers.
- Self-attention is dense MHA with separate query/key/value Linear modules, head shape `[B, heads, S, head_dim]`, and an attention backend selected through `ALL_ATTENTION_FUNCTIONS`.
- The model advertises FlashAttention, SDPA, FlexAttention, and generic attention-backend support via `_supports_*` flags, but eager parity remains dense matmul + add mask + softmax + matmul.
- Causal LM and cross-attention paths are implemented when config is mutated to decoder/cross-attention, but official XL/XXL checkpoints are masked-LM encoder configs.
- Tokenizer coupling uses `XLMRobertaTokenizer` from the base `xlm_roberta` family, with Unigram/SentencePiece assets and model inputs `input_ids` plus `attention_mask`.

Representative config URLs:

- [facebook/xlm-roberta-xl config](https://huggingface.co/facebook/xlm-roberta-xl/raw/main/config.json), repo SHA from HF API: `aa5d120255845efeebc9b7f42822a1dd0f9ece9d`
- [facebook/xlm-roberta-xxl config](https://huggingface.co/facebook/xlm-roberta-xxl/raw/main/config.json), repo SHA from HF API: `03e0fb540c3c9afd4bdda0072e7cb82d2eafd060`
- Adjacent tokenizer/background configs only, not in-scope family variants:
  - [FacebookAI/xlm-roberta-base config](https://huggingface.co/FacebookAI/xlm-roberta-base/raw/main/config.json)
  - [FacebookAI/xlm-roberta-large config](https://huggingface.co/FacebookAI/xlm-roberta-large/raw/main/config.json)

Config snapshots:

```json
{
  "model_id": "facebook/xlm-roberta-xl",
  "architectures": ["XLMRobertaXLForMaskedLM"],
  "model_type": "xlm-roberta-xl",
  "hidden_size": 2560,
  "num_hidden_layers": 36,
  "num_attention_heads": 32,
  "intermediate_size": 10240,
  "max_position_embeddings": 514,
  "vocab_size": 250880,
  "type_vocab_size": 1,
  "position_embedding_type": "absolute",
  "tokenizer_class": "XLMRobertaTokenizer",
  "torch_dtype": "float32"
}
```

```json
{
  "model_id": "facebook/xlm-roberta-xxl",
  "architectures": ["XLMRobertaXLForMaskedLM"],
  "model_type": "xlm-roberta-xl",
  "hidden_size": 4096,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "intermediate_size": 16384,
  "max_position_embeddings": 514,
  "vocab_size": 250880,
  "type_vocab_size": 1,
  "position_embedding_type": "absolute",
  "tokenizer_class": "XLMRobertaTokenizer",
  "torch_dtype": "float32"
}
```
