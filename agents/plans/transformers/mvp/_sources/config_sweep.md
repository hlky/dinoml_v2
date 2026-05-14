# MVP Source Notes

Audit date: 2026-05-13

## Local source

- Transformers checkout: `X:/H/transformers`
- Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`
- Family directory: `X:/H/transformers/src/transformers/models/mvp`
- Files inspected:
  - `configuration_mvp.py`
  - `modeling_mvp.py`
  - `__init__.py`
- Tokenizer mapping note: `mvp` maps to `MvpTokenizer`, an alias of `RobertaTokenizer` in `models/mvp/__init__.py` and `models/auto/tokenization_auto.py`. There is no model-family-local tokenizer implementation.

## Representative Hugging Face configs

Raw configs were fetched from `https://huggingface.co/RUCAIBox/<id>/raw/main/config.json`.

| Model id | use_prompt | prompt_length | prompt_mid_dim | d_model | enc layers | dec layers | heads | ffn | max pos | vocab | generation defaults in config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `RUCAIBox/mvp` | false | omitted, source default 100 | omitted, source default 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | `num_beams=5`, `no_repeat_ngram_size=3`, forced BOS 0, forced EOS 2 |
| `RUCAIBox/mvp-summarization` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-data-to-text` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-open-dialog` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-question-answering` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-question-generation` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-story` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |
| `RUCAIBox/mvp-multi-task` | true | 100 | 800 | 1024 | 12 | 12 | 16 | 4096 | 1024 | 50267 | same |

Shared observed fields:

- `activation_function="gelu"`
- `dropout=0.1`, `attention_dropout=0.1`, `activation_dropout=0.1`
- `encoder_layerdrop=0.0`, `decoder_layerdrop=0.0`
- `scale_embedding=false`
- `use_cache=true`
- `torch_dtype="float32"`
- `transformers_version="4.16.2"`
- `pad_token_id=1`, `bos_token_id=0`, `eos_token_id=2`, `decoder_start_token_id=2`
- `is_encoder_decoder=true`

No `generation_config.json` was present for `RUCAIBox/mvp`, `RUCAIBox/mvp-summarization`, or `RUCAIBox/mvp-story`; generation metadata is embedded in `config.json`.

Tokenizer config snapshots from `tokenizer_config.json` for `RUCAIBox/mvp`, `RUCAIBox/mvp-summarization`, and `RUCAIBox/mvp-story` only contained:

```json
{"model_max_length": 1024}
```

## Operator-significant source anchors

- `shift_tokens_right`: fills first decoder token with `decoder_start_token_id`, shifts source/labels, and replaces `-100` with pad id.
- `MvpLearnedPositionalEmbedding`: learned absolute positions with offset `+2`; generated positions use `past_key_values_length`.
- `MvpAttention`: separate biased Q/K/V/O projections, query scaling before `bmm`, attention mask add before softmax, optional prompt K/V concatenation on sequence axis, PyTorch `bmm` attention implementation.
- `MvpEncoderLayer`: post-attention LayerNorm, GELU FFN, post-FFN LayerNorm, fp16 finite clamp guard.
- `MvpDecoderLayer`: causal self-attention, encoder cross-attention, GELU FFN, three LayerNorms.
- `MvpPrompt`: per-stack prompt MLP from learned prompt ids to per-layer K/V tensors shaped as `(2, heads, prompt_length, head_dim)`.
- `MvpModel`: shared embedding intended for encoder and decoder via `set_input_embeddings`; constructor currently calls `MvpEncoder(config, config.use_prompt)`, where the second positional argument lands in `embed_tokens`, not `use_prompt`. Because `MvpEncoder.__init__` creates `self.embed_tokens` directly and only checks its third argument for prompt construction, the top-level `MvpModel(config.use_prompt=True)` source path appears to enable decoder prompts but not encoder prompts in this checkout. Treat this as a source-basis parity trap, and treat tied embeddings as a weight-loading alias requirement.
- `MvpForConditionalGeneration`: tied LM head to shared embeddings plus `final_logits_bias`.
- `MvpForSequenceClassification`: pools the last decoder hidden state at the last `<eos>` position, requiring equal EOS counts per batch.
- `MvpForQuestionAnswering`: linear span head over decoder hidden states, split on final channel.
- `MvpForCausalLM`: decoder-only wrapper with optional encoder cross-attention inputs, `logits_to_keep`, and no final logits bias.
