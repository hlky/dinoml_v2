# MarkupLM source notes

Transformers checkout: `X:/H/transformers`
Commit: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Inspected files:

- `src/transformers/models/markuplm/configuration_markuplm.py`
- `src/transformers/models/markuplm/modeling_markuplm.py`
- `src/transformers/models/markuplm/tokenization_markuplm.py`
- `src/transformers/models/markuplm/feature_extraction_markuplm.py`
- `src/transformers/models/markuplm/processing_markuplm.py`
- `src/transformers/models/markuplm/__init__.py`

Key source facts:

- There is no `tokenization_markuplm_fast.py` in this checkout. `MarkupLMTokenizerFast = MarkupLMTokenizer` is defined at the end of `tokenization_markuplm.py`.
- Current `modeling_markuplm.py` exports `MarkupLMModel`, `MarkupLMForQuestionAnswering`, `MarkupLMForTokenClassification`, and `MarkupLMForSequenceClassification`. It defines MLM helper classes but no exported `MarkupLMForPretraining` or `MarkupLMForMaskedLM`.
- `configuration_markuplm.py` source defaults are BERT-like (`vocab_size=30522`, `pad_token_id=0`, `type_vocab_size=2`, `max_position_embeddings=512`, `layer_norm_eps=1e-12`), but public Microsoft checkpoints override these to RoBERTa-like values (`vocab_size=50267`, `pad_token_id=1`, `type_vocab_size=1`, `max_position_embeddings=514`, `layer_norm_eps=1e-5`).
- Public checkpoint configs can contain historical fields such as `has_relative_attention_bias`, `has_tree_attention_bias`, `max_tree_id_unit_embeddings`, `rel_pos_bins`, and `tree_rel_pos_bins`. The inspected native source does not read these fields.

Representative source snippets, paraphrased:

- `XPathEmbeddings.forward` loops over `config.max_depth`; for each depth `i`, it gathers tag and subscript embeddings from separate per-depth embedding tables using `xpath_tags_seq[:, :, i]` and `xpath_subs_seq[:, :, i]`, concatenates depth units along the hidden axis, adds tag/subscript unit sequences, then applies `Linear(max_depth*xpath_unit_hidden_size -> 4*hidden_size)`, ReLU, dropout, and `Linear(4*hidden_size -> hidden_size)`.
- `MarkupLMEmbeddings.forward` defaults missing `xpath_tags_seq` and `xpath_subs_seq` to pad IDs with shape `[batch, sequence, max_depth]`; it sums word, learned absolute position, token type, and XPath embeddings before LayerNorm/dropout.
- `MarkupLMModel.forward` builds a padding-only additive attention mask as `attention_mask.unsqueeze(1).unsqueeze(2)`, casts to model dtype, and computes `(1.0 - mask) * -10000.0`.
- `MarkupLMSelfAttention.forward` uses separate biased `Linear(hidden_size -> hidden_size)` Q/K/V projections, reshapes to `[batch, heads, sequence, head_dim]`, calls the configured attention backend with noncausal dense self-attention, then reshapes back to `[batch, sequence, hidden_size]`.
- `MarkupLMTokenizer.get_xpath_seq` parses strings like `/html/body/div/li[1]`, maps tag names through `tags_dict`, clamps subscripts by `max_width`, truncates to `max_depth`, and pads tags/subscripts to `max_depth`.
- `MarkupLMProcessor.__call__` owns the choice between parsing raw HTML with BeautifulSoup through `MarkupLMFeatureExtractor` and accepting caller-supplied `nodes`/`xpaths` when `parse_html=False`.
