# BioGPT Source Notes

- Transformers source checkout: `transformers`
- Commit inspected: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Authoritative modular source for future edits: `src/transformers/models/biogpt/modular_biogpt.py`
- Generated concrete modeling source inspected for lowering details: `src/transformers/models/biogpt/modeling_biogpt.py`
- Config source inspected: `src/transformers/models/biogpt/configuration_biogpt.py`
- Tokenizer source inspected because model/tokenizer coupling affects vocab and special-token layout: `src/transformers/models/biogpt/tokenization_biogpt.py`

Key local source anchors:

- `BioGptLearnedPositionalEmbedding`: learned absolute positions with offset `2`.
- `BioGptScaledWordEmbedding`: token embedding multiplied by `sqrt(hidden_size)` when `scale_embedding=True`.
- `BioGptAttention`: separate biased `q_proj`, `k_proj`, `v_proj`, `out_proj`; MHA only.
- `BioGptDecoderLayer`: pre-attention LayerNorm, biased self-attention, residual, pre-MLP LayerNorm, `fc1 -> gelu -> fc2`, residual.
- `BioGptForCausalLM`: bias-free `output_projection` and `_tied_weights_keys` alias to `biogpt.embed_tokens.weight`.
- `BioGptTokenizer`: Moses tokenization plus BPE from `vocab.json` and `merges.txt`; special-token construction starts with `sep_token_id`.
