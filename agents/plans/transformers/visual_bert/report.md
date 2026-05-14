# VisualBERT audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: uclanlp/visualbert-vqa-coco-pre, uclanlp/visualbert-vqa, uclanlp/visualbert-vcr, uclanlp/visualbert-nlvr2, uclanlp/visualbert-vcr-coco-pre, hf-tiny-model-private/tiny-random-VisualBertModel
Config source: Hugging Face config.json snapshots under _sources/
Source files inspected: src/transformers/models/visual_bert/configuration_visual_bert.py, modeling_visual_bert.py, convert_visual_bert_original_pytorch_checkpoint_to_pytorch.py
Any missing files or assumptions: no official processor, image processor, tokenizer, or detector implementation exists in this model directory or in the sampled uclanlp repos. Visual region features are caller-supplied.
```

Snapshots:

- `_sources/configuration_visual_bert.py`
- `_sources/modeling_visual_bert.py`
- `_sources/convert_visual_bert_original_pytorch_checkpoint_to_pytorch.py`
- `_sources/uclanlp__visualbert-vqa-coco-pre.config.json`
- `_sources/uclanlp__visualbert-vqa.config.json`
- `_sources/uclanlp__visualbert-vcr.config.json`
- `_sources/uclanlp__visualbert-vcr-coco-pre.config.json`
- `_sources/uclanlp__visualbert-nlvr2.config.json`
- `_sources/hf-tiny-model-private__tiny-random-VisualBertModel.config.json`

Clickable config/source links:

- [configuration_visual_bert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/visual_bert/configuration_visual_bert.py)
- [modeling_visual_bert.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/visual_bert/modeling_visual_bert.py)
- [uclanlp/visualbert-vqa-coco-pre config](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/blob/main/config.json)
- [uclanlp/visualbert-vqa config](https://huggingface.co/uclanlp/visualbert-vqa/blob/main/config.json)
- [uclanlp/visualbert-vcr config](https://huggingface.co/uclanlp/visualbert-vcr/blob/main/config.json)
- [uclanlp/visualbert-nlvr2 config](https://huggingface.co/uclanlp/visualbert-nlvr2/blob/main/config.json)

No gated gaps were encountered. The private tiny checkpoint config resolved successfully, but it is a test-sized config, not a production model.

## 2. High-level architecture

VisualBERT is an encoder-only, single-stream multimodal BERT variant. Text tokens and precomputed visual region features are embedded into the same hidden width, concatenated on the sequence axis, and processed by bidirectional self-attention. The model does not include an image backbone, detector, patch embedding, OCR, box postprocessing, or generation loop.

Primary DinoML runtime target: base `VisualBertModel` plus the most common inference heads, with VQA and NLVR2 as useful first end-to-end targets. Pretraining, multiple choice, and region-to-phrase heads are optional follow-ons.

Dataflow:

```text
CPU/data pipeline tokenization + external detector region features
  -> text embeddings + visual projection/visual position/type embeddings
  -> concat(text_tokens, visual_tokens)
  -> BERT-style bidirectional encoder
  -> pool or gather
  -> task head logits
```

Stage decomposition:

- CPU/data pipeline: BERT tokenization, attention masks, optional token type IDs, external detector or caller-supplied `visual_embeds`.
- GPU/runtime embedding stage: text embedding lookups, visual feature projection, optional alignment-derived visual positions, concat.
- Encoder stage: repeated dense noncausal MHA + post-LN FFN blocks.
- Head stage: pool first token, gather last text token for VQA, reshape choices for VCR, or gather selected phrase positions for region alignment.

Independently cacheable pieces: external visual features can be precomputed and reused before VisualBERT. VisualBERT itself is noncausal and has no KV cache.

## 3. Important config dimensions

Source defaults from `VisualBertConfig`:

| field | default |
| --- | ---: |
| `vocab_size` | 30522 |
| `hidden_size` | 768 |
| `visual_embedding_dim` | 512 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `head_dim` | 64 |
| `intermediate_size` | 3072 |
| `hidden_act` | `gelu` |
| `max_position_embeddings` | 512 |
| `type_vocab_size` | 2 |
| `layer_norm_eps` | 1e-12 |
| `bypass_transformer` | false |
| `special_visual_initialize` | true |
| `pad_token_id` | 1 |
| cache support | none |

Representative checkpoint sweep:

| checkpoint | architecture | hidden | heads | layers | visual dim | labels | notable source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `uclanlp/visualbert-vqa-coco-pre` | `VisualBertForPreTraining` | 768 | 12 | 12 | 2048 | default 2 | COCO-pretrained VQA-style feature width |
| `uclanlp/visualbert-vqa` | `VisualBertForQuestionAnswering` | 768 | 12 | 12 | 2048 | 3129 | VQA answer-class head from `id2label` |
| `uclanlp/visualbert-vcr` | `VisualBertForMultipleChoice` | 768 | 12 | 12 | 512 | default 2 | multiple-choice reshape path |
| `uclanlp/visualbert-vcr-coco-pre` | `VisualBertForPreTraining` | 768 | 12 | 12 | 512 | default 2 | VCR pretraining width |
| `uclanlp/visualbert-nlvr2` | `VisualBertForVisualReasoning` | 768 | 12 | 12 | 1024 | default 2 | sequence classification head |
| `hf-tiny-model-private/tiny-random-VisualBertModel` | `VisualBertModel` | 32 | 4 | 5 | 20 | 3 | tiny test config, `type_vocab_size=16`, `pad_token_id=0` |

Config/source distinction: most uclanlp configs omit `num_labels`; Transformers `PreTrainedConfig` supplies the effective default of 2 where no `id2label` map is present. The VQA checkpoint explicitly carries 3129 labels through `id2label`/`label2id`.

## 3a. Family variation traps

- `visual_embedding_dim` is checkpoint-specific and operator-significant: observed official widths are 512, 1024, and 2048, while source default is 512.
- Visual region sequence length is fully runtime-provided. The source only assumes `visual_embeds` shape `(batch, visual_seq_length, visual_embedding_dim)`.
- External detector ownership is outside Transformers. DinoML should treat visual feature extraction as a separate preprocessing contract or compose a separately audited detector.
- Text and visual tokens are concatenated along `dim=1`. Layout passes must preserve sequence-axis semantics; NHWC is irrelevant except in an external detector/preprocessing branch.
- Visual token type defaults to ones, text token type defaults to zeros. The visual type embedding table is separate from the text type table, although optionally initialized from it.
- Visual position handling has two modes: all visual tokens use visual position id 0, or `image_text_alignment` averages text position embeddings and adds visual position id 0.
- `bypass_transformer=True` is implemented but absent from sampled official configs. It encodes text alone through the full encoder, concatenates visual embeddings, then applies one additional `VisualBertLayer`. First integration can reject it unless needed.
- `special_visual_initialize=True` affects initialization only, not inference graph topology.
- VQA pooling differs from pooler: it gathers `attention_mask.sum(1) - 2` from `sequence_output`, then applies the classifier.
- Multiple-choice flattens `(batch, num_choices, ...)` to `(batch*num_choices, ...)`, reuses the base model, then reshapes logits to `(batch, num_choices)`.
- Region-to-phrase alignment has a custom one-head score path and gather ABI; it should not be silently treated as the base encoder target.

## 4. Operator coverage checklist

Tensor/layout ops:

- Shape guards for `input_ids` or `inputs_embeds`, exactly one required.
- `view`/reshape for attention heads and multiple-choice batch flattening.
- `transpose`, `permute`, `contiguous`, and reshape after attention.
- `cat(..., dim=1)` for text+visual embeddings and attention masks.
- `gather(dim=1)` for VQA last-text-token pooling and region-to-phrase selected positions.
- Runtime `sum(dim=1)` over text attention mask for VQA gather index.
- Optional alignment mask construction: compare `!= -1`, multiply masked indices, sum over alignment count, divide by nonzero count.

Neural network primitives:

- Embedding lookup: word `[vocab_size, hidden]`, text position `[max_pos, hidden]`, text token type `[type_vocab_size, hidden]`.
- Embedding lookup: visual position `[max_pos, hidden]`, visual token type `[type_vocab_size, hidden]`.
- Visual projection: `Linear(visual_embedding_dim -> hidden_size)` with bias.
- LayerNorm with epsilon `1e-12` after embeddings and after attention/FFN residuals.
- Dense layers with bias for Q, K, V, attention output, FFN up/down, pooler, and heads.
- GELU activation in FFN and MLM transform by default.
- Tanh pooler activation.

Attention primitives:

- Noncausal dense self-attention over combined sequence length `T_total = T_text + T_visual`.
- MHA only, no GQA/MQA: `num_heads=12`, `head_dim=64` for production configs.
- Mask addition after scale and before softmax.
- No RoPE, ALiBi, relative bias, local/window/block sparse attention, or cache.

Preprocessing-coupled ops:

- BERT-compatible tokenizer is expected by examples, but not packaged in the uclanlp repos.
- External detector or caller supplies `visual_embeds`; no image tensor layout enters the VisualBERT graph.
- Optional `image_text_alignment` uses integer word-position references and `-1` padding.

Optional heads:

- Pretraining: MLM head tied to word embeddings plus 2-way sentence-image classifier.
- VQA: gather pooled text state then `Linear(hidden -> num_labels)`, observed 3129 labels.
- NLVR2: pool first token then `Linear(hidden -> num_labels)`.
- Multiple choice: pool first token for each choice then `Linear(hidden -> 1)` and reshape.
- Region-to-phrase: gather selected token states and score against visual token states with a custom one-head query/key projection.

## 5. Layer/block breakdown

Embedding block:

```text
text = word_embedding(input_ids) or inputs_embeds
text = text + token_type_embedding(token_type_ids or zeros)
text = text + position_embedding(position_ids or arange(T_text))

visual = visual_projection(visual_embeds)                       # optional
visual = visual + visual_token_type_embedding(ids or ones)
visual = visual + visual_position_embedding(zeros)              # default
visual = visual + average(text_position_embedding(alignment))   # optional alignment mode

x = concat(text, visual, dim=1) if visual exists else text
x = LayerNorm(x)
x = Dropout(x)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(hidden -> hidden)(x).view(B, T, H, Dh).transpose(1, 2)
k = Linear(hidden -> hidden)(x).view(B, T, H, Dh).transpose(1, 2)
v = Linear(hidden -> hidden)(x).view(B, T, H, Dh).transpose(1, 2)
scores = matmul(q, k.transpose(-1, -2)) / sqrt(Dh)
scores = scores + extended_attention_mask
probs = softmax(scores, dim=-1)
ctx = matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, T, hidden)
x = LayerNorm(Linear(hidden -> hidden)(ctx) + residual)
y = GELU(Linear(hidden -> intermediate)(x))
x = LayerNorm(Linear(intermediate -> hidden)(y) + x)
```

Task heads:

- Pooler: take `sequence_output[:, 0]`, apply `Linear(hidden -> hidden)` and `tanh`.
- VQA: compute `index_to_gather = attention_mask.sum(1) - 2`, gather that text token from full `sequence_output`, dropout, `Linear(hidden -> num_labels)`, flatten to `(batch, num_labels)`.
- NLVR2: pooler output, dropout, `Linear(hidden -> num_labels)`.
- VCR multiple choice: flatten choice dimension, base model + pooler per choice, dropout, `Linear(hidden -> 1)`, reshape to `(batch, num_choices)`.
- MLM/pretraining: `Linear(hidden -> hidden)`, GELU, LayerNorm, tied decoder `Linear(hidden -> vocab_size)` plus bias, and `Linear(hidden -> 2)` sentence-image head.
- Region-to-phrase: gather selected token states, slice visual token states from `sequence_output[:, text_len:]`, score selected tokens against visual tokens.

## 6. Attention requirements

VisualBERT uses encoder self-attention only:

- Causal: no.
- Attention type: dense bidirectional self-attention.
- Query/key/value source: same combined text+visual sequence.
- Head structure: MHA, production configs `12 x 64`, tiny config `4 x 8`.
- Query/key/value width: all `hidden_size`.
- Sequence shape: `T_total = T_text + T_visual` when `visual_embeds` is provided; otherwise `T_text`.
- Masking style: text and visual masks are concatenated, then converted by `get_extended_attention_mask` to an additive broadcast mask consumed as `(batch, 1, 1, T_total)` for a 2-D mask.
- Packed/varlen: no source support.
- Sliding/local: no.
- Relative/RoPE/ALiBi: no.
- KV cache: not applicable.
- FlashAttention/SDPA compatibility: source uses eager matmul + softmax. A fused dense noncausal attention kernel is valid under parity guards for additive mask semantics, dropout disabled in inference, and attention-output return disabled or reconstructed when requested.

The region-to-phrase optional head has a separate score-only attention-like operation: it projects gathered query states and visual key states to a single head of width `hidden_size / num_attention_heads`, computes scaled dot products, adds a visual mask, squeezes the singleton head dimension, and returns logits `(batch, selected_positions, visual_seq_length)`. Its `value` projection is defined but not used in the current forward path.

## 7. Position encoding and custom math

Text positions are learned absolute embeddings from `position_embeddings`. Default `position_ids` are `arange(max_position_embeddings)[:T_text]`.

Visual positions are learned absolute embeddings from a separate `visual_position_embeddings` table. Without alignment, every visual token uses position id 0:

```python
visual_position_ids = torch.zeros(B, T_visual, dtype=torch.long)
visual_position_embeddings = visual_position_embeddings_table(visual_position_ids)
```

With `image_text_alignment`, the source averages referenced text position embeddings, ignoring `-1` padding, then adds the visual position id 0 embedding:

```python
mask = (image_text_alignment != -1).long()
indices = mask * image_text_alignment
pos = position_embeddings(indices) * mask.to(dtype).unsqueeze(-1)
pos = pos.sum(dim=2)
denom = mask.to(dtype).sum(dim=2)
denom[denom == 0] = 1
visual_pos = pos / denom.unsqueeze(-1)
visual_pos = visual_pos + visual_position_embeddings(zeros(B, T_visual))
```

Precomputable: learned embedding tables and default text position IDs. Dynamic inputs: `image_text_alignment`, text length, visual length, masks, and VQA gather index.

## 8. Preprocessing and input packing

Text contract:

- Inputs are BERT-style `input_ids`, `attention_mask`, optional `token_type_ids`, optional `position_ids`, or direct `inputs_embeds`.
- If `token_type_ids` are omitted, text token types are zeros.
- If `attention_mask` is omitted, text tokens are all visible.
- Tokenizer files are not present in sampled uclanlp repos. Examples use `google-bert/bert-base-uncased`.

Visual feature contract:

- `visual_embeds`: float tensor `(batch, visual_seq_length, visual_embedding_dim)`.
- `visual_attention_mask`: optional tensor `(batch, visual_seq_length)`, defaults to ones if visual features are supplied.
- `visual_token_type_ids`: optional int tensor `(batch, visual_seq_length)`, defaults to ones.
- `image_text_alignment`: optional int tensor `(batch, visual_seq_length, alignment_number)` with `-1` padding and text-position indices elsewhere.

No source path accepts `pixel_values`, `boxes`, image sizes, image grids, or NHWC/NCHW image tensors. Any detector feature format, ROI count, ROI ordering, box coordinates, or feature normalization is external to this Transformers family. DinoML should require a precomputed-feature ABI for first integration and avoid promising end-to-end image parity without a separate detector audit.

NHWC relevance: guarded preprocessing/fusion only. VisualBERT proper consumes rank-3 sequence features, not image layouts.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused text/visual embedding assembly

Source pattern:

```text
Embedding lookups/projection -> several Add ops -> concat(dim=1) -> LayerNorm
```

Replacement:

```text
FusedEmbeddingAddVisualProject -> ConcatSeq -> LayerNorm
```

Preconditions:

- `visual_embeds` rank 3 and last dimension equals `visual_embedding_dim`.
- Default or explicitly supplied token type/position tensors use int indices in range.
- If `image_text_alignment` is absent, visual position ids are all zero.

Shape equations:

- text output `(B, T_text, hidden)`.
- visual output `(B, T_visual, hidden)`.
- concat output `(B, T_text + T_visual, hidden)`.

Failure cases:

- Alignment mode needs dynamic masked average and divide-by-nonzero-count.
- Missing visual branch should lower to text-only BERT embedding path.

Parity test sketch: compare embedding block output for no visual branch, visual branch without alignment, and visual branch with `-1`-padded alignment.

### Rewrite: BERT encoder block to fused dense attention and GEMM epilogues

Source pattern:

```text
Q/K/V linears -> reshape/transpose -> matmul/scale/mask/softmax/matmul -> output linear -> Add+LayerNorm -> FFN GELU -> Add+LayerNorm
```

Replacement:

```text
PackedQKVLinear(optional weight pack) -> DenseNoncausalAttention -> OutputGEMM -> AddLayerNorm -> FFNGELU -> AddLayerNorm
```

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- Dense additive mask broadcast semantics preserved.
- Inference mode: dropout disabled.
- Attention probabilities not requested, or backend can optionally materialize them.

Weight transform:

```python
qkv_weight = concat([query.weight, key.weight, value.weight], dim=0)
qkv_bias = concat([query.bias, key.bias, value.bias], dim=0)
```

Failure cases:

- Output attentions requested and fused kernel cannot return attention probabilities.
- Nonstandard `hidden_act` callable not in DinoML activation registry.

### Rewrite: VQA gather-pool classifier

Source pattern:

```text
idx = attention_mask.sum(1) - 2
pooled = gather(sequence_output, dim=1, idx)
logits = Linear(hidden -> num_labels)(Dropout(pooled)).view(B, num_labels)
```

Replacement:

```text
LastTextTokenGather(attention_mask, offset=-2) -> ClassifierGEMM
```

Preconditions:

- `attention_mask` is text-only mask, not combined visual mask.
- Batch-wise text length is positive and at least 2 special tokens.

Failure cases:

- Packed sequence or non-BERT special-token layout where `sum(mask)-2` is not the target token.

### Rewrite: multiple-choice flatten/unflatten

Source pattern:

```text
(B, C, T) -> view(B*C, T) -> encoder/pooler/head -> view(B, C)
```

Replacement:

```text
BatchChoiceFold -> shared encoder -> ChoiceLogitUnfold
```

Preconditions:

- All choice-parallel input tensors share the same choice count.
- Visual tensors are shaped `(B, C, V, Dv)` before fold.

Failure cases:

- Ragged choices require padding before lowering.

### Layout notes

Do not apply image NHWC rewrites inside VisualBERT. Sequence layout is `(batch, sequence, hidden)` throughout. The only axis-sensitive ops are sequence concat/gather/sum and attention softmax on `dim=-1`.

## 10. Kernel fusion candidates

Highest priority:

- Dense noncausal MHA over `(B, T_text + T_visual, hidden)`: dominates encoder cost, uses standard BERT mask semantics.
- Add+LayerNorm after attention and FFN: repeated twice per layer with small epsilon `1e-12`.
- FFN GEMM + GELU + GEMM: standard BERT throughput path, production shape `768 -> 3072 -> 768`.

Medium priority:

- Visual projection `Linear(visual_embedding_dim -> 768)`, especially for 2048-D VQA features.
- Fused embedding assembly + LayerNorm for text+visual concat.
- VQA gather + classifier for answer head, shape `768 -> 3129`.
- Packed QKV projection rewrite to reduce launches and enable candidate GEMM reuse.

Lower priority:

- Pooler + classifier fusion for NLVR2 and multiple choice.
- Region-to-phrase score head, because it is optional and narrower.
- Attention probability materialization, useful only for debugging/output-attentions parity.

## 11. Runtime staging plan

Stage 1: parse `VisualBertConfig`, load weights, and run embedding block parity for text-only, visual no-alignment, and visual alignment cases.

Stage 2: implement base `VisualBertModel` encoder parity with eager dense attention, LayerNorm, GELU, and pooler. Require precomputed `visual_embeds`.

Stage 3: add VQA head for `uclanlp/visualbert-vqa`, including `attention_mask.sum(1) - 2` gather and 3129-label classifier.

Stage 4: add NLVR2 visual reasoning head and multiple-choice VCR reshape path.

Stage 5: optimize encoder with packed QKV, fused dense noncausal attention, AddLayerNorm, and FFN fusions.

Stage 6: decide whether to admit `bypass_transformer=True` and region-to-phrase alignment. Keep them rejected or deferred until parity tests exist.

Stubbable initially: tokenizer, image detector, visual feature extraction, training losses, output attentions/hidden states, dropout, region-to-phrase head.

## 12. Parity and validation plan

- Config round-trip tests for source defaults and sampled checkpoint configs.
- Embedding block random tests:
  - text-only path.
  - visual path with default visual token/position ids.
  - visual path with explicit `visual_token_type_ids`.
  - alignment path with mixed valid indices and `-1` padding.
- Single encoder layer parity against Transformers fp32 with small random tensors and masks.
- Full encoder parity for tiny random config.
- Production-shape smoke parity for one or two layers at `hidden=768`, `heads=12`, visual dims 512/1024/2048.
- VQA head parity: verify gather index and classifier logits against Transformers for variable text padding.
- Multiple-choice parity: verify fold/unfold preserves `(batch, num_choices)` orientation.
- NLVR2 head parity: pool first token and classifier logits.
- Region-to-phrase optional parity: gather selected positions and score against visual tokens with mask.

Recommended tolerances:

- fp32: `rtol=1e-4`, `atol=1e-5` for encoder/head logits.
- fp16/bf16 optimized paths: start with `rtol=5e-2`, `atol=5e-2`, tighten per kernel after fused attention and LayerNorm validation.

## 13. Performance probes

- Visual projection throughput sweep by `visual_embedding_dim`: 512, 1024, 2048.
- Encoder throughput by combined sequence length: vary `T_text`, `T_visual`, and batch size independently.
- VQA end-to-end model graph excluding detector, with answer-class GEMM `768 -> 3129`.
- Attention backend comparison: eager matmul/softmax vs fused dense bidirectional attention.
- Embedding+concat+LayerNorm launch count and runtime.
- Multiple-choice throughput as choice count scales, measuring folded batch size `B*C`.
- Mask construction and dynamic shape overhead for variable `T_text`/`T_visual`.
- Optional external detector/precomputed-feature pipeline throughput, measured separately from VisualBERT.

## 14. Skip/defer list

- Training losses and dropout behavior.
- End-to-end image detector ownership and region feature extraction.
- OCR, boxes, NMS, image resizing, and NHWC/NCHW image layout handling inside VisualBERT.
- `bypass_transformer=True` unless a checkpoint requiring it is admitted.
- Region-to-phrase alignment head for first VQA/NLVR2 target.
- Output attentions and hidden state tuple materialization for optimized kernels.
- Quantization and packed weight formats; source has no custom quantized path.
- Generation, decoding, and KV cache; not applicable.

## 15. Final implementation checklist

- [ ] Parse `VisualBertConfig` including `visual_embedding_dim`, `bypass_transformer`, and label maps.
- [ ] Load base encoder weights and preserve tied MLM decoder/word embedding alias for pretraining variants.
- [ ] Define precomputed visual feature ABI `(B, V, visual_embedding_dim)` plus masks/type IDs/alignment.
- [ ] Implement text and visual embedding assembly.
- [ ] Implement optional `image_text_alignment` masked average position path.
- [ ] Implement dense noncausal BERT attention with additive mask.
- [ ] Implement BERT post-LN FFN blocks.
- [ ] Implement pooler.
- [ ] Add VQA gather-pool classifier.
- [ ] Add NLVR2 visual reasoning classifier.
- [ ] Add multiple-choice fold/unfold classifier.
- [ ] Reject or separately stage `bypass_transformer=True`.
- [ ] Reject or separately stage region-to-phrase alignment.
- [ ] Add config sweep tests for 512/1024/2048 visual dims and tiny config.
- [ ] Add embedding/block/head parity tests against Transformers.
- [ ] Benchmark visual projection, encoder, VQA head, and fused attention paths separately.
