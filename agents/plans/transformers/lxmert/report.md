# LXMERT Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`

Model id: primary `unc-nlp/lxmert-base-uncased`; representative QA checkpoints `unc-nlp/lxmert-vqa-uncased` and `unc-nlp/lxmert-gqa-uncased`.

Config source: checkpoint `config.json` files fetched from Hugging Face raw URLs on 2026-05-13 and saved under `_sources/`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/lxmert/configuration_lxmert.py`
- `X:/H/transformers/src/transformers/models/lxmert/modeling_lxmert.py`
- `X:/H/transformers/src/transformers/models/lxmert/__init__.py`
- `X:/H/transformers/src/transformers/models/auto/tokenization_auto.py` for tokenizer mapping
- `X:/H/transformers/tests/models/lxmert/test_modeling_lxmert.py` for source-owned shape expectations

Snapshots:

- `_sources/configuration_lxmert.py.txt`
- `_sources/modeling_lxmert.py.txt`
- `_sources/__init__.py.txt`
- `_sources/unc-nlp__lxmert-base-uncased__config.json`
- `_sources/unc-nlp__lxmert-vqa-uncased__config.json`
- `_sources/unc-nlp__lxmert-gqa-uncased__config.json`
- tokenizer side files for accessible UNC checkpoints where present

Missing files or assumptions:

- The in-library LXMERT directory has no `processing_*`, `image_processing_*`, or family-local tokenization file. `__init__.py` aliases `BertTokenizer` as `LxmertTokenizer`; `tokenization_auto.py` maps model type `lxmert` to `LxmertTokenizer`.
- Graphcore mirrors `[Graphcore/lxmert-vqa-uncased](https://huggingface.co/Graphcore/lxmert-vqa-uncased)` and `[Graphcore/lxmert-gqa-uncased](https://huggingface.co/Graphcore/lxmert-gqa-uncased)` returned 401 for raw `config.json` fetch. Access would only refine checkpoint metadata; the in-library operator surface is already covered by source plus UNC configs.
- No tests or imports were run, by request.

## 2. High-level architecture

LXMERT is a multimodal encoder stack, not an autoregressive generator:

```text
BERT-style text tokenization + external region feature extraction
  -> text embeddings and visual feature/box projection
  -> language encoder + visual relationship encoder
  -> bidirectional cross-modality encoder
  -> pooled CLS output and optional QA/pretraining heads
```

Stage decomposition:

- CPU/data pipeline: tokenize text with BERT-style WordPiece; obtain region features and boxes from an external detector or precomputed feature store. Transformers does not provide Faster R-CNN inference or feature extraction.
- GPU/runtime text branch: word, position, and token-type embeddings, LayerNorm, dropout disabled in eval, then `l_layers` BERT-style self-attention blocks.
- GPU/runtime visual branch: project `visual_feats[B, R, visual_feat_dim]` and `visual_pos[B, R, visual_pos_dim]` separately to hidden size, LayerNorm each, average, then `r_layers` visual self-attention blocks.
- GPU/runtime fusion branch: `x_layers` cross-modality blocks. Each block first applies language-to-vision cross-attention and vision-to-language cross-attention with shared `visual_attention` module weights, then separate language and visual self-attention, then separate FFNs.
- Heads: base `LxmertModel` returns language output, vision output, and pooled CLS. `LxmertForQuestionAnswering` maps pooled CLS to answer logits. `LxmertForPreTraining` adds MLM, image-text matching, visual object/attribute/feature heads, and optional QA.

Independently validatable units: visual feature encoder, one self-attention block, one bidirectional cross-modality block, base encoder outputs, QA head logits. External detector parity must be validated separately from DinoML graph parity.

## 3. Important config dimensions

Source defaults from `LxmertConfig`:

| Field | Default | Runtime effect |
|---|---:|---|
| `vocab_size` | 30522 | word embedding and MLM decoder width |
| `hidden_size` | 768 | text/vision/fusion hidden width |
| `num_attention_heads` | 12 | MHA heads for all attention modules |
| `head_dim` | 64 | inferred as `hidden_size / num_attention_heads`; source rejects non-divisible hidden size |
| `intermediate_size` | 3072 | FFN hidden width |
| `hidden_act` | `gelu` | FFN and head transform activation |
| `max_position_embeddings` | 512 | absolute text position table |
| `type_vocab_size` | 2 | token type table |
| `l_layers` | 9 | language encoder depth |
| `r_layers` | 5 | visual relationship encoder depth |
| `x_layers` | 5 | cross-modality encoder depth |
| `visual_feat_dim` | 2048 | external region feature width |
| `visual_pos_dim` | 4 | box/spatial feature width |
| `num_object_labels` | 1600 | optional visual object head |
| `num_attr_labels` | 400 | optional visual attribute head |
| `num_qa_labels` | 9500 | default QA answer classifier width |
| `attention_probs_dropout_prob` | 0.1 | eval no-op |
| `hidden_dropout_prob` | 0.1 | eval no-op |
| cache support | none | encoder-only, no KV cache |

Representative checkpoint sweep:

| Checkpoint | Architecture | Hidden/layers | Visual ABI | QA labels | Notes |
|---|---|---:|---|---:|---|
| `[unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased)` | `LxmertModel` | 768, L9/R5/X5 | feats 2048, pos 4 | 9500 | base feature-extraction checkpoint; config omits `num_hidden_layers`, recomputed by `__post_init__` |
| `[unc-nlp/lxmert-vqa-uncased](https://huggingface.co/unc-nlp/lxmert-vqa-uncased)` | `LxmertForQuestionAnswering` | 768, L9/R5/X5 | feats 2048, pos 4 | 3129 | VQA answer-head width differs |
| `[unc-nlp/lxmert-gqa-uncased](https://huggingface.co/unc-nlp/lxmert-gqa-uncased)` | `LxmertForQuestionAnswering` | 768, L9/R5/X5 | feats 2048, pos 4 | 1842 | GQA answer-head width differs |

Checkpoint fields that source defaults supply when absent: `pad_token_id`, `bos_token_id`, `eos_token_id`, `tie_word_embeddings`, and `num_hidden_layers` mapping. Checkpoint `layer_norm_eps` appears in configs, but inspected source hardcodes `eps=1e-12` in all LayerNorm modules and does not read `config.layer_norm_eps`.

## 3a. Family variation traps

- Region count `R` is dynamic caller input, not a config field. Attention shapes depend on `R`.
- `visual_feats` and `visual_pos` are required by `LxmertModel.forward`; there is no text-only path in this source.
- The model owns only precomputed region-feature fusion. DinoML should not infer a convolutional vision backbone from this family.
- Cross-modality block uses the same `LxmertCrossAttentionLayer` instance for both directions, so language-to-vision and vision-to-language cross-attention share Q/K/V/output weights. Logical applications must not be loaded as independent weights.
- Only language-to-vision cross-attention probabilities are returned when `output_attentions=True`; the reverse direction is computed but not exposed.
- `visual_attention_mask=None` means visual tokens are unmasked. There is no default all-ones visual mask tensor in source.
- `layer_norm_eps` in checkpoint configs is ignored by the inspected source; use hardcoded `1e-12` for parity.
- QA label count varies by checkpoint and changes the final answer classifier shape.
- `task_*` and `visual_*_loss` flags affect pretraining head construction/loss paths, not the base encoder.
- NHWC is not semantically relevant for LXMERT graph lowering because the source consumes flattened region features `[B, R, C]`, not image tensors. Layout passes should guard the external detector/preprocessing boundary and any future feature-extractor integration.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for word, absolute position, token type.
- `arange`, `unsqueeze`, `expand` for position IDs.
- Mask expansion `attention_mask[:, None, None, :]` and optional `visual_attention_mask[:, None, None, :]`.
- Type cast masks to model dtype and affine transform `(1.0 - mask) * torch.finfo(dtype).min`.
- Reshape/view, transpose, permute, contiguous, residual add, tuple/index extraction.
- First-token gather `hidden_states[:, 0]`.

Neural network primitives:

- Linear with bias throughout attention projections, attention output, FFN, visual projections, pooler, QA/pretraining heads.
- LayerNorm with eps `1e-12`.
- GELU from Transformers activation registry.
- Tanh in pooler.
- Dropout modules are present but no-op for inference.

Attention primitives:

- Dense noncausal MHA self-attention for language `[B, S, H]`.
- Dense noncausal MHA self-attention for visual regions `[B, R, H]`.
- Dense rectangular cross-attention for language queries over visual keys `[B, heads, S, R]`.
- Dense rectangular cross-attention for visual queries over language keys `[B, heads, R, S]`.
- Additive broadcast masks using dtype minimum before softmax.

Position/custom math:

- Absolute learned text positions only.
- No RoPE, ALiBi, relative bias, local/sparse attention, or KV cache.
- Visual position enters through learned linear projection of caller-supplied box/spatial values, not through attention bias.

Preprocessing-coupled ops:

- BERT WordPiece tokenization and token type IDs.
- External detector/precomputed ROI feature ABI: `visual_feats[B, R, 2048]`, `visual_pos[B, R, 4]` for official checkpoints.
- Optional `visual_attention_mask[B, R]`.

Heads:

- Base pooler: `Linear(768 -> 768)` + Tanh on CLS.
- QA head: `Linear(768 -> 1536)` + GELU + LayerNorm(1536) + `Linear(1536 -> num_qa_labels)`.
- MLM head: `Linear(768 -> 768)` + GELU + LayerNorm + tied decoder `Linear(768 -> vocab_size, bias=False)` plus separate bias.
- Relationship head: `Linear(768 -> 2)`.
- Visual heads after shared transform: object `Linear(768 -> 1600)`, attr `Linear(768 -> 400)`, feature regression `Linear(768 -> 2048)`.

Parameter sharing:

- `LxmertForPreTraining` ties `cls.predictions.decoder.weight` to `lxmert.embeddings.word_embeddings.weight`.
- Cross-modality bidirectional cross-attention shares one module within each `LxmertXLayer` for both directions.

## 5. Layer/block breakdown

Text embeddings:

```text
input_ids[B,S] or inputs_embeds[B,S,H]
position_ids = arange(S).expand(B,S)
token_type_ids default zeros[B,S]
x = word + position + token_type
x = LayerNorm(eps=1e-12)(x)
```

Visual feature encoder:

```text
vf = Linear(visual_feat_dim -> H)(visual_feats[B,R,F])
vf = LayerNorm(vf)
vp = Linear(visual_pos_dim -> H)(visual_pos[B,R,P])
vp = LayerNorm(vp)
visual = (vf + vp) / 2
```

Language/visual self-attention block, repeated `l_layers` for language and `r_layers` for visual:

```text
q,k,v = Linear(H -> H) with bias
q,k,v -> [B, heads, T, head_dim]
scores = q @ k.T / sqrt(head_dim)
scores += expanded_mask
prob = softmax(scores, dim=-1)
ctx = prob @ v -> [B,T,H]
x = LayerNorm(Linear(H -> H)(ctx) + input)
ff = GELU(Linear(H -> 4H)(x))
out = LayerNorm(Linear(4H -> H)(ff) + x)
```

Cross-modality block, repeated `x_layers`:

```text
lang_cross = CrossAttention(query=lang[B,S,H], context=visual[B,R,H], mask=visual_mask)
visual_cross = CrossAttention(query=visual[B,R,H], context=lang[B,S,H], mask=lang_mask)
lang_self = SelfAttention(lang_cross, lang_mask)
visual_self = SelfAttention(visual_cross, visual_mask)
lang_out = FFN(lang_self)
visual_out = FFN(visual_self)
```

Important sharing note: both cross-attention calls use the same `visual_attention` module weights inside the layer.

## 6. Attention requirements

All attention is encoder-style noncausal dense MHA with `num_attention_heads=12`, `head_dim=64`, Q/K/V widths 768 for official checkpoints. There is no GQA/MQA, no causal mask, no sliding window, no relative position bias, no packed/varlen metadata, and no generation cache.

Masking:

- Language mask input shape `[B, S]`, values 1 keep and 0 mask.
- Expanded language mask shape `[B, 1, 1, S]`.
- Visual mask input shape `[B, R]`, values 1 keep and 0 mask. If omitted, visual attention receives no mask.
- Expanded visual mask shape `[B, 1, 1, R]`.
- Mask value is `torch.finfo(self.dtype).min`, not a fixed `-1e4`.

Attention shapes:

- Language self: scores/probs `[B, heads, S, S]`.
- Visual self: `[B, heads, R, R]`.
- Language query over visual context: `[B, heads, S, R]`.
- Visual query over language context: `[B, heads, R, S]`.

FlashAttention/SDPA compatibility: the math is standard dense scaled dot-product attention with additive masks, so SDPA-style backends are viable for eval. Rectangular cross-attention and optional absent visual mask must be handled. Because returned attentions are optional, an optimized path can disable returning probabilities in the first integration.

## 7. Position encoding and custom math

Text uses learned absolute positions:

```python
position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0).expand(input_shape)
embeddings = word_embeddings + position_embeddings(position_ids) + token_type_embeddings
```

Visual positions are numeric box/spatial features projected into hidden space:

```python
x = LayerNorm(Linear(visual_feat_dim, hidden_size)(visual_feats))
y = LayerNorm(Linear(visual_pos_dim, hidden_size)(visual_pos))
visual_embedding = (x + y) / 2
```

There is no dynamic sinusoid, RoPE, ALiBi, or relative bias. Text position embeddings can be constant table lookups. Visual position values depend on caller-provided boxes and cannot be precomputed across images unless the region set is cached.

## 8. Preprocessing and input packing

Text preprocessing:

- `LxmertTokenizer` is an alias to BERT tokenizer.
- Accessible tokenizer configs use `do_lower_case=true`, `model_max_length=512`, and standard special tokens `[UNK]`, `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`.
- Token type IDs default to zeros if omitted. Sequence-pair segment IDs are caller/tokenizer responsibility.

Visual preprocessing:

- Transformers source explicitly states `visual_feats` are ROI pooled object features from bounding boxes using Faster R-CNN and are not provided by the library.
- Official checkpoints expect `visual_feats[B, R, 2048]`.
- Official checkpoints expect `visual_pos[B, R, 4]`, normalized to 0..1 by model docstring. The source does not enforce range or box convention.
- Source tests and integration examples use `R=10`, but `R` is not fixed by config.
- Optional `visual_attention_mask[B, R]` masks padded region slots.

CPU/data-pipeline versus GPU/runtime split:

- CPU/data pipeline should own image decoding, detector invocation, ROI pooling, coordinate normalization, region sorting/selection, tokenization, and padding.
- DinoML runtime should start from `input_ids` or `inputs_embeds`, `attention_mask`, optional `token_type_ids`, `visual_feats`, `visual_pos`, and optional `visual_attention_mask`.
- NHWC/channel-last optimization applies only to an external detector pipeline if DinoML later owns it. The LXMERT module itself is rank-3 sequence/region math.

## 9. Graph rewrite / lowering opportunities

### Rewrite: visual feature encoder as two batched GEMMs plus fused average

Source pattern:

```text
LayerNorm(Linear(F -> H)(visual_feats))
LayerNorm(Linear(P -> H)(visual_pos))
(x + y) / 2
```

Replacement: two independent batched GEMMs over `[B*R, F]` and `[B*R, P]`, two LayerNorm kernels, then fused add-scale.

Preconditions:

- `visual_feats` last dimension equals `config.visual_feat_dim`.
- `visual_pos` last dimension equals `config.visual_pos_dim`.
- Same `B,R` prefix.

Failure cases: external feature layout not contiguous or alternate box width without config change.

Parity test sketch: random `B,R,F,P` tensors, compare source module output fp32 with `rtol=1e-5`, `atol=1e-5`.

### Rewrite: encoder attention blocks to standard SDPA

Source pattern:

```text
q = Linear(x).view(B,T,heads,D).transpose(1,2)
k/v = Linear(context).view(B,U,heads,D).transpose(1,2)
softmax((q @ k.T) / sqrt(D) + mask) @ v
```

Replacement: canonical dense SDPA with separate Q/K/V projections and additive mask.

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- Noncausal dense attention.
- Output attentions not requested, or backend can return attention probabilities.
- Preserve mask value behavior for fully masked rows.

Failure cases: training dropout, requested attentions on backend that cannot materialize probs, incompatible dtype mask handling.

### Rewrite: QA head as fused MLP

Source pattern:

```text
Linear(H -> 2H) -> GELU -> LayerNorm(2H) -> Linear(2H -> num_qa_labels)
```

Replacement: GEMM + activation + LayerNorm + GEMM with optional epilogue fusions.

Preconditions: checkpoint-specific `num_qa_labels` loaded from config.

Failure cases: resized QA head at runtime must invalidate compiled output width.

### Rewrite: tied MLM decoder alias

Source pattern: `cls.predictions.decoder.weight` tied to `word_embeddings.weight`.

Replacement: one logical constant with two consumer views.

Preconditions: only for `LxmertForPreTraining`; base and QA target can defer MLM.

Failure cases: resized token embeddings need coordinated decoder bias resize and alias preservation.

## 10. Kernel fusion candidates

Highest priority:

- Dense MHA/SDPA for language, visual, and rectangular cross-attention. This dominates the 9+5+5 encoder depth.
- LayerNorm + residual add around attention and FFN outputs. LXMERT has many BERT-style post-norm sites.
- Batched linear/GEMM for Q/K/V and FFN projections over rank-3 `[B,T,H]` and `[B,R,H]` inputs.

Medium priority:

- Visual feature encoder dual projection + LayerNorm + average.
- QA head MLP for `LxmertForQuestionAnswering` checkpoints.
- MLM transform + tied decoder only if pretraining head parity is in scope.

Lower priority:

- Output attention materialization for diagnostics.
- Training losses: CrossEntropy and SmoothL1 loss paths.
- Detector/precomputed-feature production, unless DinoML chooses to own an end-to-end VQA pipeline.

## 11. Runtime staging plan

Stage 1: parse `LxmertConfig`, load base weights, and enforce input ABI guards for text tensors, `visual_feats`, `visual_pos`, and masks. Reject missing visual tensors.

Stage 2: implement visual feature encoder and one self-attention block parity with small synthetic configs from source tests.

Stage 3: implement full `LxmertModel` encoder parity for official dimensions, returning language output, vision output, and pooled output.

Stage 4: add `LxmertForQuestionAnswering` head parity for VQA/GQA checkpoints with checkpoint-specific `num_qa_labels`.

Stage 5: optimize attention blocks with SDPA/CUTLASS-backed GEMMs and residual LayerNorm fusions.

Stage 6: optionally add pretraining heads: MLM, relationship, visual object/attr/feature heads. Loss computation can remain deferred for inference.

Stage 7: decide whether DinoML owns any detector/precomputed-feature pipeline. If yes, create a separate audit for the detector family and guard layout transformations at that boundary.

## 12. Parity and validation plan

- Config roundtrip: default config and three UNC checkpoint configs. Verify `num_hidden_layers` derived mapping matches `r_layers/x_layers/l_layers`.
- Visual feature encoder fp32 random test with `B=2`, `R=10`, `F=2048`, `P=4`.
- Self-attention block test for text length `S=20` and visual region count `R=10`, with and without masks.
- Cross-modality block test for rectangular attention shapes `[S,R]` and `[R,S]`, including shared-weight loading.
- Full base model smoke against `unc-nlp/lxmert-base-uncased` style inputs. Source integration test uses `B=1`, `S=11`, `R=10`, output `[1, 11, 768]`.
- QA head parity for `unc-nlp/lxmert-vqa-uncased` output `[B,3129]` and `unc-nlp/lxmert-gqa-uncased` output `[B,1842]`.
- Optional pretraining head parity: MLM logits `[B,S,30522]`, relationship `[B,2]`, visual obj `[B,R,1600]`, attr `[B,R,400]`, feat `[B,R,2048]`.
- Recommended fp32 tolerances: `rtol=1e-4`, `atol=1e-4` for full model against PyTorch, matching source integration style. Use looser tolerances for fp16 after attention backend changes.

## 13. Performance probes

- Text length sweep: `S={16,32,64,128,512}` with fixed `R=36` or `R=10`.
- Region count sweep: `R={10,36,100}` with fixed text length.
- Encoder stage split: language-only layers, visual-only layers, cross-modality layers, pooler/head.
- Rectangular attention backend comparison for `S x R` and `R x S`.
- Visual feature encoder throughput for large `B*R`.
- QA head latency and bandwidth for `num_qa_labels` 1842, 3129, and 9500.
- External feature pipeline throughput, reported separately from DinoML runtime, if detector ownership is introduced.
- Memory probes for optional `output_hidden_states` and `output_attentions`; first integration can avoid these materializations.

## 14. Skip/defer list

- Training and all loss computation.
- Dropout behavior outside eval.
- MLM, matching, object, attribute, and feature-regression heads for first QA inference target.
- `output_attentions=True` materialization on optimized attention path.
- Runtime resizing of token embeddings or QA labels.
- End-to-end Faster R-CNN/ROI feature extraction.
- NHWC conversion inside LXMERT module; there are no image tensors in the inspected model graph.
- Graphcore gated mirror metadata until access is available.

## 15. Final implementation checklist

- [ ] Parse `LxmertConfig` and checkpoint-specific `num_qa_labels`.
- [ ] Load base encoder weights and preserve MLM tied-weight alias when pretraining head is enabled.
- [ ] Validate input ABI: `input_ids` or `inputs_embeds`, required `visual_feats`, required `visual_pos`, optional masks.
- [ ] Implement BERT-style embeddings with absolute position IDs and token type defaults.
- [ ] Implement visual feature/box projection encoder.
- [ ] Implement dense noncausal MHA self-attention and rectangular cross-attention with additive dtype-min masks.
- [ ] Preserve cross-attention shared module weights inside each `LxmertXLayer`.
- [ ] Implement post-norm attention/FFN residual blocks.
- [ ] Implement pooler and QA head.
- [ ] Add base encoder parity tests with synthetic and official-shape inputs.
- [ ] Add VQA/GQA QA-logit parity tests.
- [ ] Add performance probes for language, visual, and cross-modality attention separately.
- [ ] Document external detector/precomputed-feature ownership in any user-facing integration guide.
