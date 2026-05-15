# MarkupLM DinoML operator assessment

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: microsoft/markuplm-base, microsoft/markuplm-large, microsoft/markuplm-base-finetuned-websrc
Config source: official HF raw config/tokenizer JSON where accessible, plus source defaults in configuration_markuplm.py
Source files inspected:
  transformers/src/transformers/models/markuplm/configuration_markuplm.py
  transformers/src/transformers/models/markuplm/modeling_markuplm.py
  transformers/src/transformers/models/markuplm/tokenization_markuplm.py
  transformers/src/transformers/models/markuplm/feature_extraction_markuplm.py
  transformers/src/transformers/models/markuplm/processing_markuplm.py
  transformers/src/transformers/models/markuplm/__init__.py
Any missing files or assumptions:
  No separate tokenization_markuplm_fast.py exists in this checkout; MarkupLMTokenizerFast aliases MarkupLMTokenizer.
  Current native source has no exported MarkupLMForPretraining or MarkupLMForMaskedLM despite public base configs declaring MarkupLMForPretraining.
  microsoft/markuplm-base-finetuned-squad and microsoft/markuplm-base-finetuned-rico raw configs returned 401.
```

Primary source URLs at the pinned commit:

- `configuration_markuplm.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/markuplm/configuration_markuplm.py>
- `modeling_markuplm.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/markuplm/modeling_markuplm.py>
- `tokenization_markuplm.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/markuplm/tokenization_markuplm.py>
- `feature_extraction_markuplm.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/markuplm/feature_extraction_markuplm.py>
- `processing_markuplm.py`: <https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/markuplm/processing_markuplm.py>

Snapshots are under `_sources/`.

## 2. High-level architecture

MarkupLM is an encoder-only markup-language document model. It consumes RoBERTa/BPE token IDs plus token-aligned XPath tag and subscript sequences. There is no image tensor, no OCR bbox tensor, no visual backbone, and no layout grid in the neural graph; document structure enters through learned per-depth XPath embeddings.

```text
HTML or caller nodes/xpaths on CPU -> tokenizer builds input_ids + xpath_tags_seq + xpath_subs_seq
  -> word/position/token-type/XPath embeddings
  -> bidirectional dense encoder
  -> pooled/token/QA logits
```

Primary DinoML runtime target: base encoder inference plus question answering and token classification heads for structured web/document understanding. Sequence classification is a small optional head. The base public checkpoints advertise a pretraining architecture, but that class is not present in the inspected source.

Independently stageable pieces: CPU processor/tokenizer contract, XPath embedding ABI, one encoder block, full encoder, then task heads.

## 3. Important config dimensions

Source defaults from `MarkupLMConfig`:

| Field | Source default |
|---|---:|
| `vocab_size` | 30522 |
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | 64 |
| `intermediate_size` | 3072 |
| `max_position_embeddings` | 512 |
| `type_vocab_size` | 2 |
| `hidden_act` | `gelu` |
| `layer_norm_eps` | `1e-12` |
| `max_xpath_tag_unit_embeddings` | 256 |
| `max_xpath_subs_unit_embeddings` | 1024 |
| `xpath_unit_hidden_size` | 32 |
| `max_depth` | 50 |
| `tag_pad_id` | 216 |
| `subs_pad_id` | 1001 |
| `use_cache` | true in config, unused by source forward |

Representative checkpoint sweep:

| Checkpoint | Accessible | Native source target | Hidden | Layers | Heads | Head dim | FFN | Vocab | Max pos | Type vocab | XPath depth/unit | LN eps | dtype in config |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `microsoft/markuplm-base` | yes | encoder only; config names unsupported `MarkupLMForPretraining` | 768 | 12 | 12 | 64 | 3072 | 50267 | 514 | 1 | 50 x 32 | `1e-5` | float16 |
| `microsoft/markuplm-large` | yes | encoder only; config names unsupported `MarkupLMForPretraining` | 1024 | 24 | 16 | 64 | 4096 | 50267 | 514 | 1 | 50 x 32 | `1e-5` | float16 |
| `microsoft/markuplm-base-finetuned-websrc` | yes | `MarkupLMForQuestionAnswering` | 768 | 12 | 12 | 64 | 3072 | 50267 | 514 | 1 | 50 x 32 | `1e-5` | float32 |
| [`microsoft/markuplm-base-finetuned-squad`](https://huggingface.co/microsoft/markuplm-base-finetuned-squad) | 401 | unknown QA/fine-tune metadata | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |
| [`microsoft/markuplm-base-finetuned-rico`](https://huggingface.co/microsoft/markuplm-base-finetuned-rico) | 401 | unknown token/sequence fine-tune metadata | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown |

No RoPE, ALiBi, relative attention bias, tree attention bias, MoE, GQA/MQA, sliding windows, image/audio branches, or KV cache are implemented in the inspected native source.

## 3a. Family variation traps

- Public configs are RoBERTa-like while source defaults are BERT-like: `vocab_size=50267`, `pad_token_id=1`, `type_vocab_size=1`, `max_position_embeddings=514`, and `layer_norm_eps=1e-5` must come from checkpoint configs.
- Current source ignores historical config fields such as `has_relative_attention_bias`, `has_tree_attention_bias`, `max_tree_id_unit_embeddings`, `tree_id_unit_hidden_size`, `rel_pos_bins`, `tree_rel_pos_bins`, and `pos_mode_for_path_emb`.
- Base/large configs declare `architectures: ["MarkupLMForPretraining"]`, but no such native class exists here. DinoML should route those as base encoder weights or require a separate source basis for pretraining heads.
- The tokenizer config owns the HTML tag vocabulary. It has 215 known tag IDs in observed public configs, then tokenizer-derived unknown tag id `len(tags_dict)` and pad tag id `len(tags_dict)+1`; config `max_xpath_tag_unit_embeddings=256` provides table capacity.
- Tokenizer `pad_width=1001` and model `subs_pad_id=1001`; subscripts are clamped to tokenizer `max_width=1000`, so `1001` is reserved for padding and must fit under `max_xpath_subs_unit_embeddings=1024`.
- `MarkupLMTokenizerFast` is not a Rust fast-tokenizer subclass here; it aliases the same Python class using the tokenizers backend. Do not assume separate fast/slow behavior.
- XPath tensors are token-level, not node-level, by the time they enter the model: shape `[B, S, max_depth]` for both tags and subscripts.
- In QA mode, question tokens are sequence 0 and receive pad XPath sequences; only document/node tokens in sequence 1 receive real xpaths.
- The model has no image/layout pixel path. NHWC/channel-last layout optimization is not applicable to MarkupLM itself; protect `[B,S,H]` sequence tensors under no-layout-translation style guards.
- `use_cache=True` is config metadata only. Native forward has no `past_key_values` argument and no decode path.
- Dropout appears inside XPath embeddings and encoder blocks but is identity for inference. Keep its source placement only for training parity.

## 4. Operator coverage checklist

### Tensor/layout ops

- Integer embedding lookup for `input_ids[B,S]`, `position_ids[B,S]` or default generated positions, `token_type_ids[B,S]`, `xpath_tags_seq[B,S,D]`, and `xpath_subs_seq[B,S,D]`.
- Per-depth slice/index over `xpath_*_seq[:, :, i]` for `D=max_depth`.
- Concatenate `D` XPath unit embeddings along last dim: `[B,S,D*xpath_unit_hidden_size]`.
- Elementwise add over word, position, token type, and XPath embeddings.
- Attention mask transform `[B,S] -> [B,1,1,S]`, cast to model dtype, `(1-mask)*-10000.0`.
- Reshape/view/transpose/contiguous for attention: `[B,S,H] -> [B,heads,S,head_dim] -> [B,S,H]`.
- First-token gather for pooler: `hidden[:, 0]`.
- QA head split and squeeze: `[B,S,2] -> start[B,S], end[B,S]`.
- Optional FFN chunking through `chunk_size_feed_forward`; first integration can require disabled/zero.

### Neural network primitives

Base checkpoint shapes:

- Word embedding `Embedding(50267, 768)`.
- Position embedding `Embedding(514, 768)`.
- Token type embedding `Embedding(1, 768)` for Microsoft checkpoints; source default is 2.
- XPath tag embeddings: 50 separate tables, each `Embedding(256, 32)`.
- XPath subscript embeddings: 50 separate tables, each `Embedding(1024, 32)`.
- XPath MLP: `Linear(1600 -> 3072) -> ReLU -> Linear(3072 -> 768)` because `1600 = 50*32` and `4H=3072`.
- Encoder Q/K/V/output projections: biased `Linear(H -> H)`.
- FFN: `Linear(H -> 4H) -> GELU -> Linear(4H -> H)`.
- LayerNorm over hidden size with checkpoint eps.
- Pooler: `Linear(H -> H) -> tanh`.
- Token/sequence classifier: `Linear(H -> num_labels)`.
- QA: `Linear(H -> 2)`.

Large checkpoint changes `H=1024`, `layers=24`, `heads=16`, `FFN=4096`, and XPath MLP `Linear(1600 -> 4096) -> ReLU -> Linear(4096 -> 1024)`.

### Attention primitives

- Dense bidirectional self-attention, MHA only.
- Noncausal padding mask only; no causal, cross-attention, local, sparse, or block attention.
- Softmax in fp32 in eager fallback, downcast to query dtype.
- Configured attention backend is selected through `ALL_ATTENTION_FUNCTIONS`, so first DinoML parity should implement the eager math or a backend with identical additive mask semantics.

### Position/XPath ops

- Learned absolute token positions with RoBERTa-style position generation from non-pad tokens.
- XPath parser/tokenizer generates fixed-depth tag and subscript integer sequences.
- No RoPE, ALiBi, relative position/table bias, tree attention bias, bbox geometry, or image layout.

### Preprocessing-coupled ops

- BeautifulSoup HTML parse in feature extractor when `processor.parse_html=True`.
- Raw HTML text nodes are unescaped/stripped; empty text nodes are skipped.
- XPath strings are generated from parent traversal and sibling indices.
- Byte-level BPE tokenization; inputs are treated as pretokenized node strings for main encode path.
- Overflow records may be produced by tokenizer and require `overflow_to_sample_mapping` for XPath/label alignment.
- Node labels become token labels, with non-first subwords set to `pad_token_label=-100` by default.

### Generation/cache ops

- None. No autoregressive generation, KV cache, beam reorder, or decoder state.

## 5. Layer/block breakdown

Embedding stage for `[B,S]` tokens and `[B,S,D]` XPath tensors:

```text
position_ids = cumsum(input_ids != pad_token_id, dim=1) + pad_token_id
token_type_ids default = zeros([B,S])
xpath_tags_seq default = tag_pad_id * ones([B,S,D])
xpath_subs_seq default = subs_pad_id * ones([B,S,D])

for i in range(D):
  tag_i = Embedding(max_xpath_tag_unit_embeddings, xpath_unit_hidden_size)(xpath_tags_seq[:,:,i])
  sub_i = Embedding(max_xpath_subs_unit_embeddings, xpath_unit_hidden_size)(xpath_subs_seq[:,:,i])
xpath = concat(tag_0..tag_D-1, dim=-1) + concat(sub_0..sub_D-1, dim=-1)
xpath = Linear(D*unit -> 4H)(xpath)
xpath = ReLU(xpath)
xpath = Linear(4H -> H)(xpath)

x = word_embedding(input_ids) + position_embedding(position_ids) + token_type_embedding(token_type_ids) + xpath
x = LayerNorm(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
residual = x
q = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
k = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
v = Linear(H -> H, bias=True)(x).view(B,S,heads,head_dim).transpose(1,2)
a = softmax((q @ k.transpose(-1,-2)) * head_dim**-0.5 + additive_padding_mask, dim=-1)
x = (a @ v).transpose(1,2).reshape(B,S,H)
x = LayerNorm(Linear(H -> H)(x) + residual)
residual = x
y = GELU(Linear(H -> intermediate)(x))
x = LayerNorm(Linear(intermediate -> H)(y) + residual)
```

Heads:

- QA: encoder sequence output, `Linear(H -> 2)`, split/squeeze to start and end logits.
- Token classification: encoder sequence output, dropout identity in inference, `Linear(H -> num_labels)`.
- Sequence classification: pool token 0 through `Linear(H -> H) -> tanh`, dropout identity, `Linear(H -> num_labels)`.
- MLM helper classes exist internally (`MarkupLMOnlyMLMHead`) but no public exported model class uses them in this source basis.

## 6. Attention requirements

- Causal: no.
- Self-attention or cross-attention: self-attention only.
- MHA/MQA/GQA: standard MHA; base 12 heads x 64, large 16 heads x 64.
- Q/K/V widths: all equal hidden size. `head_dim = hidden_size / num_attention_heads`; source rejects nondivisible hidden/head combinations unless `embedding_size` exists.
- Query length equals key/value length `S`.
- Masking: additive key padding mask `[B,1,1,S]`, where masked positions receive `-10000.0`.
- Packed/varlen: not implemented; tokenizer overflow produces multiple dense samples, not packed sequence metadata.
- Sliding/local/block sparse: not implemented.
- Relative bias/RoPE/ALiBi: not implemented in current source, even when configs contain historical flags.
- KV cache: absent.
- FlashAttention/SDPA compatibility: mathematically compatible with dense noncausal SDPA if the backend supports additive key padding masks and preserves source softmax dtype behavior. No image or layout axis rewrite is involved.

## 7. Position encoding and custom math

XPath preprocessing math that must match tokenizer behavior:

```python
def markuplm_xpath_seq(xpath, tags_dict, max_depth=50, max_width=1000):
    tag_ids = []
    subs = []
    for unit in xpath.split("/"):
        if not unit.strip():
            continue
        parts = unit.strip().split("[")
        tag = parts[0]
        sub = 0 if len(parts) == 1 else int(parts[1][:-1])
        tag_ids.append(tags_dict.get(tag, len(tags_dict)))
        subs.append(min(max_width, sub))
    pad_tag = len(tags_dict) + 1
    return (
        (tag_ids[:max_depth] + [pad_tag] * max_depth)[:max_depth],
        (subs[:max_depth] + [max_width + 1] * max_depth)[:max_depth],
    )
```

Position IDs follow RoBERTa's non-pad cumulative positions:

```python
def create_position_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).int()
    return (cumsum(mask, dim=1) * mask).long() + padding_idx
```

Precomputable:

- The tag dictionary, unknown tag id, pad tag id, and pad XPath vectors for a checkpoint.
- Position IDs for fixed padded shapes if `input_ids` padding pattern is fixed.

Dynamic:

- HTML parsing, node extraction, XPath string construction, subword tokenization, overflow mapping, and question/document XPath assignment.

## 8. Preprocessing and input packing

CPU/data-pipeline responsibilities:

- If `MarkupLMProcessor.parse_html=True`, caller supplies `html_strings`; BeautifulSoup extracts text nodes and xpaths.
- If `parse_html=False`, caller supplies `nodes` and `xpaths` directly. This is the cleaner first DinoML integration boundary.
- Tokenizer packs either single sequence nodes or question-plus-nodes pairs.
- Special tokens and question tokens receive pad XPath sequences; document node subwords receive the node XPath.
- Padding must extend `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq`, `xpath_subs_seq`, labels, and optional masks consistently on the configured padding side.
- Token labels are node-level labels expanded to token labels; by default only the first subword gets the real label.

GPU/runtime graph inputs:

```text
input_ids:      [B,S] int
attention_mask: [B,S] int/bool/float keep mask
token_type_ids: optional [B,S] int, effectively all zeros for public Microsoft configs
position_ids:   optional [B,S] int
xpath_tags_seq: optional [B,S,50] int, pad-filled when absent
xpath_subs_seq: optional [B,S,50] int, pad-filled when absent
```

No `pixel_values`, bbox coordinates, OCR box normalization, image duplication, grid metadata, modality placeholders, or `cu_seqlens` enter the native model graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: XPath embedding fan-in fusion

Source pattern:

```text
50 tag embedding gathers + 50 subscript embedding gathers
  -> concat tags, concat subs
  -> add
  -> Linear(D*unit -> 4H) -> ReLU -> Linear(4H -> H)
```

Replacement:

```text
FusedXPathGatherConcatAdd -> GEMM -> ReLU -> GEMM
```

Preconditions:

- `max_depth`, `xpath_unit_hidden_size`, tag table sizes, and subscript table sizes are static from config.
- All tag IDs are `< max_xpath_tag_unit_embeddings`; all subscript IDs are `< max_xpath_subs_unit_embeddings`.
- Inference mode so dropout is identity.

Shape equations:

- `D=max_depth`, `U=xpath_unit_hidden_size`.
- `xpath_*_seq: [B,S,D]`.
- concat output `[B,S,D*U]`, final output `[B,S,H]`.

Weight transform: none initially; preserve 50 separate per-depth tables for checkpoint key compatibility.

Failure cases:

- Tokenizer config whose `pad_tag_id` or `pad_width` exceeds model embedding table size.
- Dynamic/custom max depth not matching loaded tables.

Parity test sketch: generate random in-range tag/subscript tensors plus pad values and compare source `XPathEmbeddings` with dropout disabled.

### Rewrite: embedding sum plus LayerNorm

Source pattern:

```text
word + position + token_type + xpath -> LayerNorm
```

Replacement:

```text
FusedEmbeddingAdd4 -> LayerNorm
```

Preconditions:

- Inference mode.
- All embedding outputs are `[B,S,H]`.
- Public Microsoft checkpoints with `type_vocab_size=1` receive only token type id 0.

Failure cases:

- `inputs_embeds` path bypasses word embedding.
- Custom token type ids out of range.

Parity test sketch: compare embedding module output for explicit and default `position_ids`, `token_type_ids`, and XPath tensors.

### Rewrite: separate Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(H,H)(x); k = Linear(H,H)(x); v = Linear(H,H)(x)
```

Replacement:

```text
qkv = MatMul(x, concat(Wq,Wk,Wv).T) + concat(bq,bk,bv)
split qkv as all-Q, all-K, all-V
```

Preconditions:

- Same input feeds all three projections.
- Output widths all equal `H`.
- Biases are present.

Weight transform:

```python
Wqkv = concat([Wq, Wk, Wv], axis=0)
bqkv = concat([bq, bk, bv], axis=0)
```

Failure cases:

- Pruned heads or quantized packed storage with different layout.

Parity test sketch: compare packed projection and source projections before attention reshape.

### Rewrite: key padding mask canonicalization

Source pattern:

```text
attention_mask[B,S] -> (1 - mask.to(dtype)) * -10000.0 -> [B,1,1,S]
```

Replacement:

```text
KeyPaddingMask[B,S] passed directly to dense attention backend
```

Preconditions:

- Mask is binary 1 keep / 0 pad.
- Backend applies before softmax with equivalent additive behavior.

Failure cases:

- User supplies non-binary additive masks.

Parity test sketch: all-valid, right-padded, and interspersed masked tokens.

## 10. Kernel fusion candidates

Highest priority:

- XPath gather/concat/add plus XPath MLP: this is MarkupLM's distinctive cost and ABI.
- Packed QKV projection plus dense noncausal attention: same high-value encoder pattern as BERT/RoBERTa.
- Residual add plus LayerNorm after attention and FFN projections, with dropout removed for inference.

Medium priority:

- Embedding sum plus LayerNorm across word/position/type/XPath sources.
- FFN `Linear -> GELU -> Linear` scheduling for base/large throughput.
- QA head projection/split/squeeze fusion for document QA.

Lower priority:

- Pooler `first token -> Linear -> tanh` fusion.
- Token classification head fusion for small label counts.
- MLM helper projection support only if a future source basis restores a public masked-LM/pretraining class.

## 11. Runtime staging plan

1. Parse config and tokenizer metadata; require checkpoint configs rather than source defaults for public weights.
2. Admit `parse_html=False` first: caller supplies model-ready `nodes/xpaths` or direct tensors.
3. Implement XPath tokenizer ABI validation and runtime tensor guards.
4. Load encoder weights and run XPath embedding parity.
5. Implement one encoder block parity for dense noncausal attention and FFN.
6. Run full encoder parity for base and large.
7. Add QA and token classification heads as first useful document tasks.
8. Add sequence classification head.
9. Add packed QKV and XPath embedding fusion.
10. Revisit pretraining/MLM only with a source basis that actually exports the relevant class.

Initial stubs allowed:

- BeautifulSoup HTML parsing inside DinoML runtime; keep it in CPU pipeline.
- Training losses, dropout, gradient checkpointing.
- Nonzero `chunk_size_feed_forward`.
- Hidden-state and attention collection.

## 12. Parity and validation plan

- Tokenizer/preprocessor golden tests for a tiny HTML string: nodes, xpaths, token IDs, XPath tag/subscript tensors, question pair behavior.
- XPath parser tests: unknown tag, pad tag, subscript clamp at 1000, pad subscript 1001, truncation at depth 50.
- Embedding parity in fp32 with default and explicit XPath tensors, public checkpoint `type_vocab_size=1`, and source-default `type_vocab_size=2`.
- One-layer attention parity with all-valid and padded masks.
- Full encoder parity for base at sequence lengths 4, 128, and 512; large at a smaller smoke shape if resources are constrained.
- QA head parity: start/end logits `[B,S]` against `microsoft/markuplm-base-finetuned-websrc`.
- Token classification parity with synthetic `num_labels`.
- Sequence classification parity with synthetic `num_labels`.
- Admission tests: `S > max_position_embeddings - pad_token_id - 1`, invalid XPath tag/subscript IDs, nonzero token type ids when `type_vocab_size=1`, unsupported pretraining architecture request.
- Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5`; fp16 `rtol=5e-2, atol=5e-2` for full encoder outputs, tighter for isolated embeddings/linear ops.

## 13. Performance probes

- CPU processor throughput: BeautifulSoup parse, XPath construction, tokenizer node/subword expansion, overflow mapping.
- Runtime XPath embedding cost by `B`, `S`, and hidden size; compare unfused per-depth gathers vs fused gather/concat.
- Encoder throughput for base and large at `S=128,256,512`.
- Attention backend comparison: eager dense MHA vs optimized dense noncausal SDPA with key padding mask.
- FFN throughput and GELU fusion effect.
- Head-specific cost: QA vs token classification vs sequence classification.
- Batch-size sweep with variable padding ratios.
- Memory probe for attention score tensors at base/large sequence cap.

## 14. Skip/defer list

- Training losses, gradient checkpointing, and dropout behavior beyond inference identity.
- Autoregressive generation, KV cache, beam search, and speculative decode: not applicable.
- Image/OCR/layout tensors, bbox embeddings, NHWC/channel-last rewrites: not part of native MarkupLM.
- Historical relative/tree attention fields in public configs; current source ignores them.
- `MarkupLMForPretraining`/masked LM parity until an inspected native source exports those classes.
- BeautifulSoup parsing inside the GPU runtime; keep HTML parsing in CPU/data pipeline.
- Gated fine-tuned label maps for SQuAD/RICO until HF access resolves the 401 configs.
- Quantization and multi-GPU tensor parallelism.

## 15. Final implementation checklist

- [ ] Parse `MarkupLMConfig` and checkpoint tokenizer metadata.
- [ ] Reject or route unsupported `MarkupLMForPretraining` architecture from public base configs.
- [ ] Validate tokenizer/model XPath ABI: tag table size, pad tag id, subscript pad id, max depth.
- [ ] Implement model-ready tensor input contract for `input_ids`, `attention_mask`, `token_type_ids`, `position_ids`, `xpath_tags_seq`, and `xpath_subs_seq`.
- [ ] Implement XPath embedding gather/concat/add/MLP.
- [ ] Implement embedding sum and LayerNorm with RoBERTa-style position IDs.
- [ ] Implement dense noncausal MHA with additive key padding mask.
- [ ] Implement GELU FFN and residual LayerNorm ordering.
- [ ] Implement QA, token classification, and sequence classification heads.
- [ ] Add admission guards for token type range, sequence length, XPath tag/subscript range, and ignored historical config fields.
- [ ] Add rewrite for packed QKV projection.
- [ ] Add rewrite for XPath embedding fan-in fusion.
- [ ] Add parity tests for tokenizer ABI, XPath embeddings, one encoder block, full encoder, and heads.
- [ ] Benchmark processor, XPath embedding stage, encoder attention/FFN, and heads separately.
