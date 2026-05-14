# EuroBERT Source Snapshot

Source basis: `X:/H/transformers` at commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Key source anchors from `src/transformers/models/eurobert/modeling_eurobert.py`:

- `EuroBertRMSNorm` at line 45: fp32 variance over last dim, multiply by learned weight, cast back to input dtype.
- `apply_rotary_pos_emb` at line 73: unsqueeze cos/sin on dim 1, rotate half as `[-x2, x1]`, apply to Q and K.
- `eager_attention_forward` at line 110: repeat KV groups, `Q @ K^T * scaling`, add mask, fp32 softmax, optional dropout, `P @ V`.
- `EuroBertAttention` at line 136: non-causal self-attention, Q projects to `num_attention_heads * head_dim`, K/V to `num_key_value_heads * head_dim`, O back to `hidden_size`.
- `EuroBertMLP` at line 203: gated MLP `down_proj(act(gate_proj(x)) * up_proj(x))`.
- `EuroBertModel` at line 347: token embedding, bidirectional mask, shared RoPE cos/sin per forward, repeated decoder-style layers, final RMSNorm.
- `EuroBertForMaskedLM` at line 409: linear LM head over all sequence positions.
- `EuroBertForSequenceClassification` at line 475: pooling modes `bos`, `mean`, and `late`.
- `EuroBertForTokenClassification` at line 567: per-token linear classifier.

Configuration anchors from `configuration_eurobert.py`:

- Defaults: vocab `128256`, hidden `768`, intermediate `3072`, layers `12`, heads `12`, max position `8192`, activation `silu`, attention/MLP bias disabled by default.
- `num_key_value_heads` defaults to `num_attention_heads`; `head_dim` defaults to `hidden_size // num_attention_heads`.
- `validate_architecture` rejects `hidden_size % num_attention_heads != 0`.

Hub compatibility notes:

- Official EuroBERT configs still carry `auto_map` entries and `custom_code` tags. Native source at the inspected commit implements `AutoModel`, masked LM, sequence classification, and token classification.
- Official config `auto_map` mentions `EuroBertForQuestionAnswering`, but the inspected native source does not implement that head. Treat native QA as out of scope unless a separate remote-code audit is requested.
- Some fine-tuned configs use historical `clf_pooling`; inspected native source reads `classifier_pooling`. Admission should normalize or reject that alias explicitly.
