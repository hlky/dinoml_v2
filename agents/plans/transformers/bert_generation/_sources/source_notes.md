# bert_generation source notes

Local source files inspected:
- `X:/H/transformers/src/transformers/models/bert_generation/configuration_bert_generation.py`
- `X:/H/transformers/src/transformers/models/bert_generation/modeling_bert_generation.py`
- `X:/H/transformers/src/transformers/models/bert_generation/tokenization_bert_generation.py`
- Shared helpers inspected for ABI context:
  - `X:/H/transformers/src/transformers/masking_utils.py`
  - `X:/H/transformers/src/transformers/cache_utils.py`

Important line anchors from `modeling_bert_generation.py`:
- Embeddings: word + learned absolute position embeddings, LayerNorm, dropout at lines 394-425.
- Self attention: separate Q/K/V `nn.Linear(hidden_size, hidden_size)` projections and `[B, heads, T, head_dim]` reshape at lines 89-152.
- Cross attention: query from decoder hidden states, key/value from `encoder_hidden_states`, cache reuse via `EncoderDecoderCache.is_updated` at lines 157-229.
- Transformer layer: self-attention, optional cross-attention only when `is_decoder` and `encoder_hidden_states` are present, then chunked feed-forward at lines 295-356.
- Mask construction: decoder path calls `create_causal_mask`; encoder path calls `create_bidirectional_mask`; encoder attention mask also uses bidirectional mask at lines 559-586.
- Decoder LM head: `Linear(hidden_size, vocab_size)` plus tied-weight metadata at lines 592-610; forward slices hidden states by `logits_to_keep` before the LM head at lines 635-699.

Source-derived runtime modes:
- `BertGenerationEncoder` with `is_decoder=False`: bidirectional encoder, no returned cache.
- `BertGenerationDecoder` with `is_decoder=True`, `add_cross_attention=False`: causal decoder with dynamic self-attention KV cache.
- `BertGenerationDecoder` with `is_decoder=True`, `add_cross_attention=True` and `encoder_hidden_states`: decoder self-cache plus cross-attention KV cache over encoder states.

Attention/cache details:
- Dynamic cache stores each layer's key/value tensors as `[batch_size, num_heads, seq_len, head_dim]`.
- Cross-attention cache stores projected encoder K/V once per layer and reuses it when `is_updated[layer_idx]` is true.
- `create_bidirectional_mask` accepts 2D masks `[B, kv_len]` or already prepared 4D masks `[B, 1, q_len, kv_len]`.
- `create_causal_mask` derives `q_offset` and `kv_length` from the cache when present.

Ignored or source-defaulted config fields:
- Current `BertGenerationConfig` does not declare `position_embedding_type`, `directionality`, `gradient_checkpointing`, or `return_scores`; these may be retained in older configs but are not read by this modeling file for inference graph structure.
- `chunk_size_feed_forward` is inherited from `PreTrainedConfig` and used by `apply_chunking_to_forward`; production inference should gate it to `0` or implement exact chunk/split/concat semantics.
