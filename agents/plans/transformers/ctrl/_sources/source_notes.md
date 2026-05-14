# CTRL Source Notes

Scope: native Transformers `ctrl` family at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`, plus representative Hugging Face Hub configs fetched on 2026-05-13.

## Local source hotspots

- `configuration_ctrl.py`
  - `CTRLConfig.model_type = "ctrl"`.
  - Attribute aliases map `max_position_embeddings -> n_positions`, `hidden_size -> n_embd`, `num_attention_heads -> n_head`, `num_hidden_layers -> n_layer`.
  - Defaults: `vocab_size=246534`, `n_positions=256`, `n_embd=1280`, `dff=8192`, `n_layer=48`, `n_head=16`, `layer_norm_epsilon=1e-6`, `use_cache=True`, `tie_word_embeddings=True`.

- `modeling_ctrl.py`
  - `positional_encoding(position, d_model_size, dtype)` builds sinusoidal angles, then returns `cat([sines, cosines], dim=-1)`.
  - `scaled_dot_product_attention` uses explicit `q @ k^T / sqrt(dk)`, adds sliced upper-triangular mask times `-1e4`, then adds optional `attention_mask`, softmaxes, and matmuls with values.
  - `MultiHeadAttention` uses separate biased `Wq`, `Wk`, `Wv`, and `dense` linears. Head depth is `int(d_model_size / num_heads)`.
  - Cache update is `layer_past.update(k, v, self.layer_idx)` whenever a cache object is passed.
  - `CTRLModel.forward` creates `DynamicCache(config=self.config)` only when `use_cache` is true and `past_key_values is None`.
  - Default `position_ids` are `arange(past_length, past_length + input_len).unsqueeze(0)`.
  - Causal mask is materialized as `torch.triu(torch.ones(seq_len + past_length, seq_len + past_length), 1)`.
  - Token type embeddings reuse `self.w(token_type_ids)` and are scaled by `sqrt(hidden)`, but are absent when token type IDs are not passed.
  - `CTRLLMHeadModel` uses `lm_head = nn.Linear(n_embd, vocab_size, bias=True)` and declares tied weight key `lm_head.weight -> transformer.w.weight`.
  - `prepare_inputs_for_generation` delegates to `GenerationMixin` then removes `token_type_ids`.
  - `CTRLForSequenceClassification` uses a bias-free classifier and gathers the last non-pad token when `pad_token_id` exists; otherwise batch size must be 1.

- `tokenization_ctrl.py`
  - Slow BPE tokenizer only.
  - Fixed `CONTROL_CODES` table includes `Legal: 11859`, `Wikipedia: 37583`, `Opinion: 43213`, `News: 4256`, and other prompt categories.
  - Tokenization regex is `\S+\n?`.
  - Defaults request all-zero token type IDs from tokenizer helpers, but generation removes token type IDs before model forward.

## Representative config snapshots

| Model | URL | Key fields |
|---|---|---|
| `Salesforce/ctrl` | https://huggingface.co/Salesforce/ctrl/raw/main/config.json | `n_layer=48`, `n_embd=1280`, `n_head=16`, `dff=8192`, `n_positions=50000`, `vocab_size=246534`; includes historical `attn_pdrop`, `n_ctx`, `summary_*`; omits `use_cache` |
| `sshleifer/tiny-ctrl` | https://huggingface.co/sshleifer/tiny-ctrl/raw/main/config.json | `n_layer=2`, `n_embd=16`, `n_head=2`, `dff=2`, `n_positions=50000`, `vocab_size=246534` |
| `hf-tiny-model-private/tiny-random-CTRLLMHeadModel` | https://huggingface.co/hf-tiny-model-private/tiny-random-CTRLLMHeadModel/raw/main/config.json | `n_layer=5`, `n_embd=32`, `n_head=4`, `dff=8192`, `n_positions=512`, `pad_token_id=246533`, `use_cache=true`, `is_decoder=true` |
| `hf-tiny-model-private/tiny-random-CTRLModel` | https://huggingface.co/hf-tiny-model-private/tiny-random-CTRLModel/raw/main/config.json | same tiny dimensions; architecture is base `CTRLModel` |
| `prajjwal1/ctrl_discovery_1` | https://huggingface.co/prajjwal1/ctrl_discovery_1/raw/main/config.json | production dimensions with `vocab_size=246535`, `use_cache=true` |
| `prajjwal1/ctrl_discovery_flipped_4` | https://huggingface.co/prajjwal1/ctrl_discovery_flipped_4/raw/main/config.json | production dimensions with `vocab_size=246535`, `use_cache=true` |
| `wvangils/CTRL-Beatles-Lyrics-finetuned-newlyrics` | https://huggingface.co/wvangils/CTRL-Beatles-Lyrics-finetuned-newlyrics/raw/main/config.json | tiny-ctrl-derived `n_layer=2`, `n_embd=16`, `n_head=2`, `dff=2`, `n_positions=50000` |

## Config/source gaps to preserve in report

- `attn_pdrop` is common in configs but no attention dropout op appears in native source.
- `n_ctx` is present in older configs but native source uses `n_positions` for the position table and does not read `n_ctx`.
- `summary_*` fields are historical classifier summary settings but inspected `CTRLForSequenceClassification` does a direct token classifier plus last-token gather.
- `is_decoder=true` in tiny LM config is not read by native CTRL modeling code.
- No `num_key_value_heads`, RoPE, ALiBi, sliding-window, SDPA selector, FlashAttention selector, MoE, quantization, or remote-code hooks were found.

## Hub search note

The Hub API search for `ctrl` returned many unrelated models whose names contain `ctrl` but whose tags/configs are GPT-2, BART, RoBERTa, T5, wav2vec2, etc. Only repositories with `ctrl` tags and/or fetched `model_type="ctrl"` configs were treated as representative for this audit.
