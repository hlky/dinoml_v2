# X-MOD Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `transformers`.

Model id: primary representative `facebook/xmod-base`; additional public configs `facebook/xmod-base-75-269k` and `facebook/xmod-large-prenorm`.

Config source: local `XmodConfig` defaults plus Hugging Face `config.json` snapshots saved under `_sources/`.

Source files inspected:

- `transformers/src/transformers/models/xmod/configuration_xmod.py`
- `transformers/src/transformers/models/xmod/modeling_xmod.py`
- `transformers/src/transformers/models/xmod/convert_xmod_original_pytorch_checkpoint_to_pytorch.py`
- `transformers/src/transformers/models/auto/tokenization_auto.py`
- `transformers/src/transformers/models/auto/modeling_auto.py`

Primary source/config links:

- [configuration_xmod.py at inspected commit](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/xmod/configuration_xmod.py)
- [modeling_xmod.py at inspected commit](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/xmod/modeling_xmod.py)
- [facebook/xmod-base config](https://huggingface.co/facebook/xmod-base/raw/main/config.json)
- [facebook/xmod-base-75-269k config](https://huggingface.co/facebook/xmod-base-75-269k/raw/main/config.json)
- [facebook/xmod-large-prenorm config](https://huggingface.co/facebook/xmod-large-prenorm/raw/main/config.json)

Any missing files or assumptions: no family-owned tokenizer file exists; AutoTokenizer maps `xmod` to `XLMRobertaTokenizer`, and `facebook/xmod-base` only has a small tokenizer config snapshot declaring that class. Several plausible variant IDs returned 401 while fetching configs; they are listed in `_sources/source_notes.md`.

## 2. High-level architecture

X-MOD is a RoBERTa/BERT-style multilingual text encoder with language-specific bottleneck adapters inside each transformer FFN output block. Public representative checkpoints target masked language modeling.

```text
XLM-R tokenization + language id -> token/position/type embeddings
  -> repeated encoder block with self-attention + FFN + language adapter
  -> masked-LM head or pooled/token/span classification head
```

First useful DinoML runtime target: encoder plus `XmodForMaskedLM` for fill-mask parity. The base encoder is independently useful for classification/token/QA heads. The source also implements optional `is_decoder=True` causal LM and `add_cross_attention=True` decoder-cross-attention paths, but the public inspected checkpoints are masked-LM encoders, so decoder/cache support should be optional/deferred unless a real decoder config is admitted.

## 3. Important config dimensions

| Field | `xmod-base` | `xmod-base-75-269k` | `xmod-large-prenorm` |
|---|---:|---:|---:|
| architecture | `XmodForMaskedLM` | `XmodForMaskedLM` | `XmodForMaskedLM` |
| vocab size | 250002 | 250002 | 250002 |
| layers | 12 | 12 | 24 |
| hidden size | 768 | 768 | 1024 |
| attention heads | 12 | 12 | 16 |
| head dim | 64 | 64 | 64 |
| FFN intermediate | 3072 | 3072 | 4096 |
| max positions | 514 | 514 | 514 |
| type vocab size | 1 | 1 | 1 |
| activation | GELU | GELU | GELU |
| layer norm eps | 1e-5 | 1e-5 | 1e-5 |
| `pre_norm` | false | false | true |
| adapter reduction | 2 | 2 | 4 |
| adapter bottleneck | 384 | 384 | 256 |
| adapter LN | false | false | true |
| reuse FFN LN before adapter | true | true | false |
| LN before adapter residual | true | true | false |
| language adapters | 81 | 75 | 81 |
| tokenizer class | XLMRobertaTokenizer for `xmod-base` | not in repo files inspected | not in repo files inspected |

Source defaults differ from checkpoint configs: `XmodConfig` defaults to `vocab_size=30522`, `max_position_embeddings=512`, `type_vocab_size=2`, `layer_norm_eps=1e-12`, and `languages=("en_XX",)`. DinoML should use checkpoint config values, not source defaults, for real weights.

## 3a. Family variation traps

- Adapter routing is the defining variation. Each layer owns one adapter module per configured language; the language list and order define integer `lang_ids`.
- `lang_ids` is graph-significant. If omitted, `config.default_language` must be set with `set_default_language`; otherwise source raises.
- `pre_norm` changes LayerNorm placement. Large prenorm applies attention/FFN LayerNorm before blocks and a final encoder LayerNorm; base applies LayerNorm after attention and after adapter output.
- Adapter LayerNorm policy changes the graph: base reuses `XmodOutput.LayerNorm` before adapters, large has a separate `adapter_layer_norm` and does not reuse the FFN LayerNorm.
- Adapter residual placement changes with `ln_before_adapter`. Base residual captures the normalized tensor; large captures the pre-adapter tensor.
- Token type embeddings exist but public checkpoints use `type_vocab_size=1`; segment IDs are effectively always zero unless a caller passes valid custom IDs.
- No GQA/MQA: all inspected configs have `hidden_size == num_heads * head_dim` and separate dense Q/K/V projections.
- AutoTokenizer coupling is XLM-R/SentencePiece-like, not BERT WordPiece. The model does not derive `lang_ids` from tokenizer language codes.
- The source advertises FlashAttention/SDPA/Flex attention support through backend dispatch, but eager math is ordinary dense scaled dot-product attention.
- Decoder and cross-attention code paths exist in source via config flags; current public configs do not enable them.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for tokens, positions, and token types.
- Padding-aware position id creation: `cumsum(input_ids != pad_id) * mask + pad_id`.
- Broadcast/add of three embedding streams, LayerNorm, dropout disabled for inference.
- Boolean compare and boolean mask/indexed row selection for adapter routing.
- `zeros_like`/scatter-like writeback into an output tensor for selected language rows.
- Reshape/view/transposes for Q/K/V: `[B,S,H] -> [B,heads,S,64]`.
- First-token indexing for pooler and classification heads.
- Multiple-choice flatten/unflatten: `[B,C,S] -> [B*C,S]`, logits `[B*C,1] -> [B,C]`.
- QA split/squeeze: `[B,S,2] -> start/end [B,S]`.

Neural network primitives:

- Dense linear with bias throughout.
- LayerNorm with eps from config.
- GELU for FFN, adapter, and LM head.
- Tanh for pooler/classification head.
- Residual add around attention, FFN, and adapter.

Attention primitives:

- Bidirectional encoder self-attention for primary target.
- Optional causal self-attention with DynamicCache when `is_decoder=True`.
- Optional encoder-decoder cross-attention when `add_cross_attention=True`.
- Additive attention masks created by Transformers masking utilities.

Position encoding:

- Learned absolute position embeddings. No RoPE, ALiBi, or relative bias.

Task heads:

- Required first target: masked LM head `Linear(H,H) -> GELU -> LayerNorm -> Linear(H,V)`.
- Optional: sequence classification `[CLS] -> dropout -> Linear(H,H) -> tanh -> dropout -> Linear(H,num_labels)`.
- Optional: token classification `dropout -> Linear(H,num_labels)`.
- Optional: QA `Linear(H,2) -> split`.
- Optional: multiple choice pooler `first token -> Linear(H,H) -> tanh -> dropout -> Linear(H,1)`.
- Deferred: causal LM generation and seq2seq-style cross-attention unless a decoder config is explicitly targeted.

## 5. Layer/block breakdown

Base/postnorm encoder block, repeated 12 times for base configs:

```text
x0: [B,S,H]
q,k,v = Linear(H -> H)(x0), split to [B,A,S,64]
attn = softmax((q @ k^T) * 1/sqrt(64) + mask) @ v
x1 = Linear(H -> H)(attn) + x0
x1 = LayerNorm(x1)
ff = GELU(Linear(H -> I)(x1))
x2 = Linear(I -> H)(ff) + x1
adapter_input = LayerNorm(x2)          # because adapter_reuse_layer_norm=true
adapter_out = route_by_lang(adapter_input, adapter_lang)
x3 = dropout(adapter_out) + adapter_input
out = LayerNorm(x3)
```

Large prenorm block, repeated 24 times:

```text
x0: [B,S,1024]
attn_in = LayerNorm(x0)
q,k,v = Linear(1024 -> 1024)(attn_in), split to [B,16,S,64]
x1 = Linear(1024 -> 1024)(attention) + x0
ff_in = LayerNorm(x1)
ff = GELU(Linear(1024 -> 4096)(ff_in))
x2 = Linear(4096 -> 1024)(ff) + x1
adapter_in = AdapterLayerNorm(x2)
adapter_out = route_by_lang(adapter_in, adapter_lang)
out = dropout(adapter_out) + x2
final encoder output = LayerNorm(out_after_last_layer)
```

Adapter per configured language:

```text
adapter(x) = Linear(H -> H / reduction) -> GELU -> Linear(H / reduction -> H)
```

All projections in inspected source use bias.

## 6. Attention requirements

Primary target attention is noncausal encoder self-attention:

- MHA, not GQA/MQA.
- Heads: 12 base or 16 large.
- Head dim: 64.
- Q/K/V width: hidden size.
- Masking: bidirectional additive mask over key positions, produced from `attention_mask`.
- Packed/varlen: no source-level packed sequence metadata.
- Sliding/local/block sparse: none.
- Position interaction: learned absolute positions are added before attention.
- FlashAttention/SDPA compatibility: source can dispatch to registered attention backends; eager fallback is `matmul -> scale -> mask add -> softmax(dim=-1) -> dropout -> matmul`.

Optional decoder path:

- Causal self-attention when `config.is_decoder=True`.
- Cache stores per-layer K/V after projection/reshape in `[B,heads,T,64]`.
- Optional cross-attention projects query from decoder hidden states and K/V from encoder hidden states; cross K/V can be cached in `EncoderDecoderCache`.
- Admission recommendation: reject decoder/cross-attention configs for the first X-MOD integration unless a checkpoint explicitly requires them.

## 7. Position encoding and custom math

No rotary or relative-position math is present. Position ids are padding-aware for `input_ids`:

```python
mask = (input_ids != pad_token_id).int()
position_ids = (cumsum(mask, dim=1) + past_key_values_length) * mask
position_ids = position_ids.long() + pad_token_id
```

For `inputs_embeds`, source cannot infer pads and emits sequential positions from `pad_token_id + 1`.

The learned position embedding table uses checkpoint `max_position_embeddings=514`, matching RoBERTa/XLM-R style room for special/pad indexing.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- Tokenization is XLMRobertaTokenizer for `xmod` AutoTokenizer mapping and for the public `facebook/xmod-base` tokenizer config.
- Special tokens and SentencePiece vocabulary are tokenizer-owned; the neural graph consumes `input_ids`, `attention_mask`, optional `token_type_ids`, optional `position_ids`, and language IDs.

GPU/runtime ABI:

- `input_ids`: `[B,S]` int token ids, or `inputs_embeds`: `[B,S,H]`.
- `attention_mask`: `[B,S]` before Transformers expands it into attention backend form.
- `token_type_ids`: optional `[B,S]`; default is gathered from an all-zero buffer, so public configs with `type_vocab_size=1` effectively use zeros.
- `position_ids`: optional `[B,S]`; default is padding-aware cumsum.
- `lang_ids`: source docs say `[B,S]`, but default path creates `[B]`. The adapter implementation uses boolean masking into `[B,S,H]`, so per-example `[B]` and per-token `[B,S]` have different routing semantics. DinoML should first admit per-example one-language-per-row routing and reject per-token mixed-language routing unless specifically implemented.

Language control is separate from tokenization. The tokenizer does not inject adapter IDs; callers or runtime metadata must supply `lang_ids` or set a default language.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fixed-language adapter specialization

Source pattern: loop over all `adapter_modules`, mask rows matching each adapter id, run the selected adapter, scatter into `new_hidden_states`.

Replacement: for a compile-time fixed `default_language`, lower only that language adapter as a normal dense subgraph.

Preconditions:

- All examples in the compiled/run batch use the same language id.
- The id maps to the same language string/order as the checkpoint config.
- Per-token language routing is rejected.

Failure cases: mixed-language batches, dynamic language selection, changed config language ordering.

Parity test sketch: compare one encoder block and full encoder for two individual languages, compiling separate fixed-language artifacts.

### Rewrite: grouped adapter routing

Replacement: partition batch rows by language id, launch one adapter subgraph per nonempty language group, then restore original row order.

Preconditions:

- Routing is per example `[B]`, not per token.
- Stable gather/scatter by batch row is available.
- Empty language groups are skipped.

This keeps one artifact supporting many language IDs without running all adapters over the full batch.

### Rewrite: adapter MLP fusion

Source pattern: `Linear(H,Bn) -> GELU -> Linear(Bn,H) -> dropout(false) -> residual`.

Replacement: fused GEMM/GELU/GEMM/residual region, optionally using adapter-specific weights.

Preconditions: inference mode, dense contiguous `[rows,H]` where rows is `B*S` after language grouping.

### Rewrite: QKV projection packing

Source pattern: three independent `Linear(H,H)` projections for Q/K/V.

Replacement: packed `Linear(H,3H)` with split order `[q, k, v]`.

Weight transform:

```python
packed_w = cat([q.weight, k.weight, v.weight], dim=0)
packed_b = cat([q.bias, k.bias, v.bias], dim=0)
```

Preconditions: same dtype/device/layout and no observers/hooks. Split rows as all-Q, all-K, all-V.

### Rewrite: LM head tied decoder

Source ties `lm_head.decoder.weight` to `roberta.embeddings.word_embeddings.weight` in masked/causal LM wrappers.

Replacement: preserve one logical parameter alias or a read-only shared constant. Do not duplicate mutable state in a way that breaks tied-weight loading.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm variants around attention/FFN/adapter, because X-MOD has more LayerNorm traffic than plain RoBERTa when adapters are active.
- Dense encoder MHA via SDPA/FlashAttention-compatible fused attention for `[B,A,S,64]`.
- Fixed-language or grouped adapter route plus adapter MLP fusion; otherwise the PyTorch-style boolean mask/scatter loop is the family-specific bottleneck.
- GELU FFN GEMM epilogue fusion for `H -> 4H -> H`.

Medium priority:

- Packed QKV projection.
- Embedding sum plus LayerNorm fusion.
- Mask creation and padding-aware position-id generation as a small preprocessing/runtime helper.
- LM head last-dimension projection, especially vocab `250002`.

Lower priority:

- Pooler/classification/QA heads.
- Optional causal decode cache path until a real X-MOD decoder checkpoint is targeted.
- Per-token mixed-language adapter routing.

## 11. Runtime staging plan

Stage 1: parse config, load XLM-R tokenizer metadata separately, load weights, and build a single fixed-language encoder block.

Stage 2: full encoder with fixed default language and masked-LM head.

Stage 3: add per-example `lang_ids` by compiling either one artifact per language or grouped adapter dispatch.

Stage 4: add optional task heads: sequence classification, token classification, QA, multiple choice.

Stage 5: optimize attention and FFN/adapter fusions.

Stage 6: evaluate optional decoder/cross-attention support only if a checkpoint config enables `is_decoder` or `add_cross_attention`.

Can be stubbed initially: training losses, dropout, gradient checkpointing, output attentions/hidden states, dynamic `inputs_embeds`, per-token language routing, and decoder generation.

## 12. Parity and validation plan

- Config-load tests for source defaults versus checkpoint configs, including adapter language count/order.
- Position-id helper parity for padded and unpadded batches.
- Single adapter parity: run adapter module on random `[B,S,H]` for base and large dimensions.
- Adapter routing parity: fixed-language `[B]`, mixed per-example `[B]`, and explicit rejection or parity for `[B,S]`.
- Single-layer parity for base postnorm and large prenorm blocks.
- Full encoder hidden-state parity for `facebook/xmod-base` and `facebook/xmod-large-prenorm`.
- Masked-LM logits parity on short multilingual text for at least two languages.
- Head parity for sequence/token/QA using random weights or a fine-tuned checkpoint if available.
- Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 optimized paths should use looser op-specific tolerances after baseline fp32 parity is established.

## 13. Performance probes

- Encoder throughput sweep over `B` and `S` for base and large.
- Adapter routing sweep: fixed single language versus mixed-language batch with 2, 4, 8, and many active languages.
- Adapter bottleneck GEMM sizes: base `768x384x768`, large `1024x256x1024`.
- Attention backend comparison: eager dense, SDPA, FlashAttention-compatible path.
- Vocab projection cost for masked LM with full `[B,S,250002]` logits versus selected masked positions if a higher-level pipeline can provide mask indices.
- Memory probe for all adapters resident: language count multiplies adapter parameters in every layer.
- Tokenization/preprocessing throughput separately from encoder GPU time.

## 14. Skip/defer list

- Training losses and label handling.
- Dropout behavior outside inference.
- Gradient checkpointing.
- Output attentions/hidden-state capture.
- Decoder-only causal LM and encoder-decoder cross-attention until an admitted checkpoint requires them.
- Per-token mixed-language routing.
- Beam search and generation controllers.
- Gated/private variant IDs that returned 401 until configs are accessible.
- General boolean scatter as a public op; prefer bounded adapter routing patterns.

## 15. Final implementation checklist

- [ ] Parse X-MOD config fields, including adapter flags and ordered languages.
- [ ] Load XLM-R tokenizer metadata as CPU/data-pipeline state.
- [ ] Preserve tied LM decoder/input embedding alias.
- [ ] Implement padding-aware position IDs.
- [ ] Implement encoder embeddings and LayerNorm placement for postnorm and prenorm.
- [ ] Implement dense MHA with additive bidirectional mask.
- [ ] Implement FFN and adapter modules.
- [ ] Add fixed-language adapter specialization.
- [ ] Add per-example grouped adapter routing or reject mixed-language batches.
- [ ] Implement masked-LM head.
- [ ] Add optional sequence/token/QA/multiple-choice heads.
- [ ] Add single-layer and full-encoder parity tests for base and large-prenorm configs.
- [ ] Benchmark adapter routing/fusion and attention backends.
