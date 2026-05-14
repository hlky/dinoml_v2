# XLM Source Notes

Audit target: `xlm`

Transformers checkout:

- Local path: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Local source files inspected:

- `X:/H/transformers/src/transformers/models/xlm/configuration_xlm.py`
- `X:/H/transformers/src/transformers/models/xlm/modeling_xlm.py`
- `X:/H/transformers/src/transformers/models/xlm/tokenization_xlm.py`
- `X:/H/transformers/src/transformers/models/xlm/convert_xlm_original_pytorch_checkpoint_to_pytorch.py` only for file presence; runtime report does not depend on converter behavior.

Representative config snapshots saved under `configs/`:

- `FacebookAI__xlm-mlm-en-2048.config.json` from `https://huggingface.co/FacebookAI/xlm-mlm-en-2048/raw/main/config.json`
- `FacebookAI__xlm-mlm-100-1280.config.json` from `https://huggingface.co/FacebookAI/xlm-mlm-100-1280/raw/main/config.json`
- `FacebookAI__xlm-clm-enfr-1024.config.json` from `https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/raw/main/config.json`
- `FacebookAI__xlm-mlm-tlm-xnli15-1024.config.json` from `https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/raw/main/config.json`
- `hf-internal-testing__tiny-random-xlm.config.json` from `https://huggingface.co/hf-internal-testing/tiny-random-xlm/raw/main/config.json`

Source-derived runtime notes:

- `XLMConfig` defaults describe `causal`, `asm`, `n_langs`, `use_lang_emb`, `sinusoidal_embeddings`, and encoder/decoder flags, but native `XLMModel.__init__` raises `NotImplementedError` when `is_encoder` is false.
- `get_masks` returns a padding mask plus either noncausal `[B,S]` attention mask or causal `[B,S,S]` triangular mask.
- `MultiHeadAttention` uses four bias linears, dense MHA, score scaling before matmul masking, float32 softmax, and optional `DynamicCache` / `EncoderDecoderCache` update paths.
- `XLMModel.forward` accepts `langs`, `token_type_ids`, `position_ids`, `lengths`, and `cache`. It adds token embeddings, absolute position embeddings, optional language embeddings, and unusually uses the same word embedding table for `token_type_ids` if supplied.
- `XLMPredLayer` uses either dense `Linear(H -> vocab)` or `nn.AdaptiveLogSoftmaxWithLoss` when `asm=true`; sampled public configs all have `asm=false`.
- `XLMWithLMHeadModel.prepare_inputs_for_generation` appends a mask token and optional constant language id, then drops externally supplied attention/position/token-type tensors. This is generation-controller behavior, not a standard decoder prefill/decode ABI.
- `XLMTokenizer` is a Python tokenizer based on Moses preprocessing plus BPE, with custom paths for Chinese, Japanese, and Thai. It can carry `lang2id` / `id2lang`; those language ids are coupled to the model `langs` tensor when language embeddings are used.

Validation performed:

- No DinoML imports, model execution, tests, or commits were run.
- Network use was limited to fetching representative Hugging Face `config.json` files.
