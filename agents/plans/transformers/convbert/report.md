# Transformers ConvBERT Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.
  Remote upstream: https://github.com/huggingface/transformers.git

Model id:
  Primary target: YituTech/conv-bert-base for encoder feature extraction and masked-LM style encoder inference.
  Size/operator references: YituTech/conv-bert-small, YituTech/conv-bert-medium-small.
  Language/vocab references: dbmdz/convbert-base-german-europeana-cased,
  Finnish-NLP/convbert-base-finnish, mrm8488/convbert-small-spanish.

Config source:
  https://huggingface.co/YituTech/conv-bert-small/raw/main/config.json
  https://huggingface.co/YituTech/conv-bert-medium-small/raw/main/config.json
  https://huggingface.co/YituTech/conv-bert-base/raw/main/config.json
  https://huggingface.co/dbmdz/convbert-base-german-europeana-cased/raw/main/config.json
  https://huggingface.co/Finnish-NLP/convbert-base-finnish/raw/main/config.json
  https://huggingface.co/mrm8488/convbert-small-spanish/raw/main/config.json

Source files inspected:
  X:/H/transformers/src/transformers/models/convbert/modeling_convbert.py
  X:/H/transformers/src/transformers/models/convbert/configuration_convbert.py
  X:/H/transformers/src/transformers/models/convbert/tokenization_convbert.py

Source snapshots:
  agents/plans/transformers/convbert/_sources/modeling_convbert.py
  agents/plans/transformers/convbert/_sources/configuration_convbert.py
  agents/plans/transformers/convbert/_sources/tokenization_convbert.py
  agents/plans/transformers/convbert/_sources/*.config.json
  agents/plans/transformers/convbert/_sources/*.repo_info.json

Any missing files or assumptions:
  ConvBERT has only a slow tokenizer class, and it inherits BertTokenizer unchanged.
  The YituTech repos do not provide tokenizer_config.json or special_tokens_map.json;
  DinoML should rely on BERT WordPiece tokenizer semantics for those checkpoints.
  The checked representative repos are public, non-gated model repos. No 401/403/gated gap was found.
  This report is docs-only; no DinoML tests were run.
```

## 2. High-level architecture

ConvBERT is a BERT-like bidirectional text encoder with learned word, position, and segment embeddings. Its distinctive block replaces part of each self-attention head with span-based dynamic convolution: per-token convolution kernels are generated from the product of a separable-convolution key path and the query path, then applied to local windows of another projected hidden-state path. The dense attention output and local convolution output are concatenated back to `hidden_size`.

```text
WordPiece tokenization + [CLS]/[SEP]/token_type_ids
  -> word + token_type + absolute position embeddings in embedding_size E
  -> embedding LayerNorm/dropout
  -> optional Linear(E -> H) embedding projection
  -> N encoder blocks with mixed self-attention + dynamic convolution
  -> encoder hidden states or optional task head
```

Primary DinoML runtime target should be `ConvBertModel` encoder parity plus `ConvBertForMaskedLM`, because the standard public checkpoints are feature-extraction/encoder checkpoints and the source's LM head is the largest common head. Sequence classification, token classification, question answering, and multiple choice are optional downstream heads. Training losses, dropout behavior, and gradient checkpointing are not required for first inference parity.

## 3. Important config dimensions

`ConvBertConfig` source defaults look like `YituTech/conv-bert-base`, except the source default `pad_token_id` is `1` while inspected configs set `pad_token_id=0`.

| Field | Source default | Operator relevance |
|---|---:|---|
| vocab_size / V | 30522 | word embeddings and LM head |
| embedding_size / E | 768 | embedding table width before optional projection |
| hidden_size / H | 768 | encoder width |
| num_hidden_layers | 12 | encoder block repeat count |
| num_attention_heads / A_cfg | 12 | configured head count before `head_ratio` |
| head_ratio | 2 | reduces attention heads: `A = max(1, A_cfg // head_ratio)` |
| effective attention heads / A | 6 | source-derived for default |
| attention_head_size / D | 64 | `D = (H // A) // 2`; attention and conv each use H/2 channels |
| all_head_size | 384 | `A * D = H / 2` for standard configs |
| conv_kernel_size / Kc | 9 | local dynamic convolution window |
| num_groups | 1 | grouped FFN LinearLayer when >1 |
| intermediate_size / I | 3072 | FFN expansion |
| hidden_act | gelu | FFN and classifier activation |
| max_position_embeddings | 512 | learned absolute position table |
| type_vocab_size | 2 | segment/token type table |
| layer_norm_eps | 1e-12 | embedding, residual, and LM transform LayerNorm |
| classifier_dropout | None | falls back to hidden dropout for classifier heads |
| is_decoder / add_cross_attention | false / false | source has a cross-attention branch but no generation cache |

Representative checkpoint sweep:

| Checkpoint | Repo SHA | Arch | V | E | H | I | Layers | A_cfg -> A | D | Kc | Groups | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YituTech/conv-bert-small | 9a113301 | ConvBertModel | 30522 | 128 | 256 | 1024 | 12 | 4 -> 2 | 64 | 9 | 1 | small/debug-like shape; requires E->H projection |
| YituTech/conv-bert-medium-small | 48891256 | ConvBertModel | 30522 | 128 | 384 | 1536 | 12 | 8 -> 4 | 48 | 9 | 2 | exercises grouped FFN matmul |
| YituTech/conv-bert-base | 5cb45193 | ConvBertModel | 30522 | 768 | 768 | 3072 | 12 | 12 -> 6 | 64 | 9 | 1 | common base geometry |
| dbmdz/convbert-base-german-europeana-cased | 5bf2d7f7 | ConvBertModel | 32000 | 768 | 768 | 3072 | 12 | 12 -> 6 | 64 | 9 | 1 | different vocab/tokenizer, base body |
| Finnish-NLP/convbert-base-finnish | 7ca436fa | ConvBertModel | 50265 | 768 | 768 | 3072 | 12 | 12 -> 6 | 64 | 9 | 1 | larger vocab, base body |
| mrm8488/convbert-small-spanish | fae5f3e4 | ConvBertModel | 30522 | 128 | 256 | 1024 | 12 | 4 -> 2 | 64 | 9 | 1 | small body, Spanish tokenizer metadata |

Fields commonly omitted from inspected configs but supplied by `ConvBertConfig`: `classifier_dropout=None`, `is_decoder=False`, `add_cross_attention=False`, `tie_word_embeddings=True`, and source default `pad_token_id=1`. The configs explicitly set `attention_probs_dropout_prob=0.1`, `hidden_dropout_prob=0.1`, `initializer_range=0.02`, and `pad_token_id=0`.

## 3a. Family variation traps

- `num_attention_heads` in config is not the runtime attention head count. Source computes `A = num_attention_heads // head_ratio`, with a fallback to one head if that would be zero.
- Each effective head has `attention_head_size = (hidden_size // A) // 2`. Dense attention uses only `all_head_size = A * D`; the other half of `hidden_size` comes from the dynamic convolution path.
- `hidden_size` must be divisible by effective `A`. The source does not explicitly require `H == A_cfg * head_dim`; for medium-small, `H=384`, `A=4`, `D=48`.
- `embedding_size` can differ from `hidden_size`; small and medium-small require `embeddings_project: Linear(E -> H)`.
- `num_groups > 1` changes only FFN input/output dense layers into `GroupedLinearLayer`; medium-small uses `num_groups=2`.
- The local convolution path is axis-sensitive. Source hidden states are `[B,S,H]`, but `SeparableConv1D` receives `[B,H,S]`; the dynamic output window path uses an `unfold` over a synthetic `[B,C,S,1]` tensor. Layout passes must keep the sequence axis as the local-window axis.
- `ConvBertForMaskedLM` ties `generator_lm_head.weight` to `convbert.embeddings.word_embeddings.weight` when tying is enabled; preserving the logical alias matters.
- The source exposes `is_decoder` and `add_cross_attention` branches, but there is no `use_cache`, `past_key_values`, or generation cache path in this modeling file. Treat decoder/cross-attention as non-primary and cache-free unless a separate remote-code variant proves otherwise.
- Tokenizer behavior is BERT WordPiece inheritance. Repos can change vocab size/language/casing metadata without changing model-body operators.
- Layout translation to NHWC/channel-last is not a semantic default for this text model. Candidate local-conv optimization may keep `[B,S,H]` with hidden contiguous and implement sequence-window gather directly; translating source Conv1d-style NCL regions requires explicit axis rewrites.

## 4. Operator coverage checklist

### Tensor/layout ops

- Input validation: exactly one of `input_ids[B,S]` or `inputs_embeds[B,S,E]`.
- Embedding gathers:
  - word embedding `[V,E]`, `padding_idx=pad_token_id`.
  - token type embedding `[type_vocab_size,E]`.
  - position embedding `[max_position_embeddings,E]`.
- Default position IDs `position_ids[:, :S]`; no source past offset for ConvBERT.
- Default token type IDs from a registered all-zero buffer expanded to `[B,S]`.
- Elementwise embedding sum, LayerNorm over `E`, optional `Linear(E -> H)`.
- Standard view/transpose/reshape for attention: `[B,S,H/2] -> [B,A,S,D]`.
- Dynamic convolution path:
  - transpose `[B,S,H] -> [B,H,S]` for separable Conv1d.
  - depthwise Conv1d with groups=`H`, kernel=`Kc`, padding=`Kc//2`, no bias.
  - pointwise Conv1d `H -> H/2`, kernel 1, no bias, plus explicit bias `[H/2,1]`.
  - elementwise multiply with query projection `[B,S,H/2]`.
  - Linear kernel generator `H/2 -> A*Kc`.
  - reshape to `[-1,Kc,1]`, softmax over `Kc`.
  - `conv_out_layer: Linear(H -> H/2)`.
  - local sequence-window extract via unfold on `[B,H/2,S,1]`, reshape to `[-1,D,Kc]`.
  - batched matmul with generated kernels, reshape back to `[B,S,A,D]`.
- Concatenate attention context and convolution context along the head-like axis, then view to `[B,S,H]`.
- Attention mask expansion to additive broadcast shape from `get_extended_attention_mask`.
- Head-specific ops: `squeeze`, `split`, `gather`, multiple-choice flatten/unflatten.

### Neural network primitives

- Bias Linear/GEMM for Q/K/V, attention output, conv kernel generator, conv output projection, embedding projection, heads, and FFN when `num_groups=1`.
- Grouped linear matmul for FFN when `num_groups>1`: weight shape `[G, input_size/G, output_size/G]`, bias `[output_size]`.
- LayerNorm with affine weight/bias and eps `1e-12`.
- GELU activation for official configs.
- Residual add followed by LayerNorm in attention output and FFN output.
- Dropout is inference no-op.

### Attention primitives

- Bidirectional dense self-attention for the primary encoder target.
- Effective MHA with `A = num_attention_heads // head_ratio`; no MQA/GQA.
- Query/key/value projections are separate and each output `H/2`, not `H`.
- Attention scores are scaled by `sqrt(D)`, additive-masked, softmaxed over key sequence, and multiplied by value.
- Dynamic convolution attention is required alongside dense attention; a plain BERT MHA replacement is not parity-correct.

### Position/relative-bias ops

- Learned absolute position embeddings only.
- No RoPE, ALiBi, relative bias, or convolutional position embedding beyond the attention-local dynamic convolution branch.

### Generation/cache ops

- No generation cache is implemented in the inspected source.
- Decoder/cross-attention branch has no KV-cache ABI and should be deferred for primary ConvBERT.

### Preprocessing-coupled ops

- BERT WordPiece tokenizer semantics through `ConvBertTokenizer(BertTokenizer)`.
- Standard `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` behavior from BERT tokenizer/vocab files.
- Segment/token type IDs are active.

## 5. Layer/block breakdown

Embedding path:

```text
word = Embedding(V,E)(input_ids)
tok = Embedding(type_vocab,E)(token_type_ids or zeros)
pos = Embedding(max_pos,E)(position_ids)
x_e = LayerNorm_E(word + tok + pos)
x = Linear(E -> H)(x_e) if E != H else x_e
```

ConvBERT encoder block, repeated `N` times:

```text
# Dense attention half
q = Linear(H -> H/2, bias=True)(x) -> view [B,S,A,D] -> transpose [B,A,S,D]
k = Linear(H -> H/2, bias=True)(x or encoder_hidden_states) -> [B,A,S,D]
v = Linear(H -> H/2, bias=True)(x or encoder_hidden_states) -> [B,A,S,D]
attn = softmax((q @ k^T) / sqrt(D) + mask) @ v             # [B,A,S,D]

# Dynamic convolution half, always driven by decoder/self hidden_states in source
kc = SeparableConv1D_depthwise_pointwise(H -> H/2, Kc)(x.transpose(1,2)).transpose(1,2)
kernel_in = kc * q_flat                                    # [B,S,H/2]
dyn_kernel = softmax(Linear(H/2 -> A*Kc)(kernel_in).reshape([-1,Kc,1]), dim=1)
conv_values = Linear(H -> H/2)(x)
windows = local_window_extract(conv_values, axis=S, width=Kc, pad=(Kc-1)//2)
conv = windows.reshape([-1,D,Kc]) @ dyn_kernel             # [B*S*A,D,1]

mixed = concat(attn.transpose_to_[B,S,A,D], conv.reshape[B,S,A,D], axis=2)
mixed = view [B,S,H]
x = LayerNorm(Linear(H -> H)(mixed) + x)

z = Linear_or_GroupedLinear(H -> I)(x)
z = GELU(z)
z = Linear_or_GroupedLinear(I -> H)(z)
x = LayerNorm(z + x)
```

Masked LM head:

```text
h = Linear(H -> E)(x)
h = GELU(h)
h = LayerNorm_E(h)
logits = Linear(E -> V, tied_weight=word_embeddings)(h)
```

Sequence classification head:

```text
h = x[:, 0, :]
h = dropout(h)
h = Linear(H -> H)(h)
h = hidden_act(h)
h = dropout(h)
logits = Linear(H -> num_labels)(h)
```

Other optional heads:

```text
token classification: dropout(x) -> Linear(H -> num_labels)
question answering: Linear(H -> num_labels)(x) -> split start/end -> squeeze
multiple choice: flatten choices -> ConvBertModel -> SequenceSummary -> Linear(H -> 1) -> reshape [B,C]
```

## 6. Attention requirements

Primary attention is bidirectional encoder self-attention with a required local dynamic convolution companion.

- Causal or noncausal: noncausal for primary encoder.
- Self-attention or cross-attention: self-attention for standard checkpoints; optional cross-attention source branch is non-primary.
- MHA/MQA/GQA: MHA, but reduced from config by `head_ratio`.
- Head count/head dim:
  - small: `A=2`, `D=64`, dense attention width `128`.
  - medium-small: `A=4`, `D=48`, dense attention width `192`.
  - base: `A=6`, `D=64`, dense attention width `384`.
- Masking style: additive extended attention mask broadcast over `[B,A,S,S]` attention scores.
- Packed/varlen support: none in source.
- Sliding/local attention: dense attention is full sequence; local dynamic convolution separately uses a fixed `Kc=9` window over sequence.
- Position interactions: no relative/rotary terms in attention scores.
- KV cache requirements: none in inspected source.
- FlashAttention/SDPA compatibility: only the dense attention half maps cleanly to SDPA. The dynamic convolution half still requires a separate local-window path, and the concatenation/output projection must preserve source ordering.

Dense eager attention math:

```python
scores = matmul(q, k.transpose(-1, -2)) / sqrt(D)
scores = scores + extended_attention_mask
probs = softmax(scores, dim=-1)
context = matmul(probs, v)
```

Dynamic convolution math, shortened:

```python
kc = separable_conv1d(x.transpose(1, 2)).transpose(1, 2)  # [B,S,H/2]
kernels = softmax(linear_kernel(kc * q_flat).reshape(-1, Kc, 1), dim=1)
windows = unfold(linear_conv_out(x), sequence_axis, width=Kc, pad=(Kc - 1) // 2)
conv = matmul(windows.reshape(-1, D, Kc), kernels).reshape(B, S, A, D)
```

## 7. Position encoding and custom math

Position encoding is learned absolute embedding lookup. It can be precomputed as a `[1,max_position_embeddings]` integer buffer and sliced to `S` when callers omit `position_ids`.

The nontrivial custom math is span-based dynamic convolution attention:

```python
def convbert_dynamic_conv(x, q_flat, params):
    # x: [B,S,H], q_flat: [B,S,H/2], Kc odd
    kconv = depthwise_conv1d(x.transpose(1, 2), groups=H, pad=Kc // 2)
    kconv = pointwise_conv1d(kconv, out_channels=H // 2).transpose(1, 2)
    kconv = kconv + bias.view(1, 1, H // 2)
    kernel = softmax(linear(kconv * q_flat).reshape(-1, Kc, 1), dim=1)
    vals = linear_out(x)                                      # [B,S,H/2]
    windows = local_windows(vals, axis=1, width=Kc, zero_pad=True)
    return batched_matmul(windows.reshape(-1, D, Kc), kernel)
```

Precomputable:

- Position ID buffer.
- Token type zero buffer.
- Static local-window offsets for a fixed `Kc`.

Dynamic:

- Convolution kernels depend on the current hidden states and query projection.
- Padding mask affects dense attention only; the source dynamic convolution path does not apply `attention_mask` to local conv windows.

## 8. Preprocessing and input packing

CPU/data-pipeline work:

- WordPiece tokenization inherited from BERT.
- Standard sequence construction: `[CLS] A [SEP]` or `[CLS] A [SEP] B [SEP]`.
- Token type IDs: segment 0/1 for sentence pairs.
- Padding/truncation to the chosen sequence length, normally within `max_position_embeddings=512`.

GPU/runtime inputs:

- `input_ids[B,S]` integer token IDs, or `inputs_embeds[B,S,E]`.
- `attention_mask[B,S]`, 1 for valid tokens and 0 for padding by standard Transformers convention.
- Optional `token_type_ids[B,S]`; default all zeros.
- Optional `position_ids[1,S]` or broadcastable/caller-provided equivalent.

No multimodal placeholder tokens, scatter stitching, audio/image preprocessing, packed sequence descriptors, or `cu_seqlens` metadata are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: embedding triple-sum plus optional projection

Source pattern:

```text
GatherWord + GatherTokenType + GatherPosition -> Add/Add -> LayerNorm(E) -> optional Linear(E,H)
```

Replacement:

```text
FusedEmbeddingSumLayerNorm(E) -> optional GEMM_RCR_Bias(E,H)
```

Preconditions:

- `input_ids` path is used; `inputs_embeds` bypass must remain available or be rejected by an admitted DinoML target.
- All embedding outputs have shape `[B,S,E]`.
- Missing `token_type_ids` follows source all-zero expansion.
- Caller-provided `position_ids` are preserved.

Failure cases:

- Direct `inputs_embeds`.
- Checkpoint/source mismatch around `pad_token_id` must not change embedding rows.

Parity test sketch:

- Compare embeddings for explicit/default token type IDs and position IDs on small/base configs.

### Rewrite: ConvBERT dynamic convolution to local-window kernel primitive

Source pattern:

```text
transpose -> depthwise Conv1d(H groups, Kc) -> pointwise Conv1d(H -> H/2)
-> multiply with q -> Linear(H/2 -> A*Kc) -> softmax(Kc)
-> Linear(H -> H/2) -> unfold local windows -> batched matmul
```

Replacement:

```text
DepthwiseTemporalConv + PointwiseGEMM + KernelGeneratorGEMM
+ LocalWindowGather(axis=S,Kc,pad) + GroupedSmallMatmul(D,Kc)
```

Exact preconditions:

- Source tensor rank is `[B,S,H]`.
- `Kc` is odd and padding is `(Kc - 1) // 2`; official configs use `Kc=9`.
- Depthwise Conv1d uses `groups=H`, dilation 1, stride 1, no bias.
- Pointwise Conv1d has no bias; explicit separable-conv bias is added afterward.
- `H`, `A`, and `D` satisfy `H == A * D * 2`.
- Dynamic kernel softmax is over the local-window dimension only.

Shape equations:

```text
A = max(1, num_attention_heads // head_ratio)
D = (H // A) // 2
all_head_size = A * D
local windows: [B,S,all_head_size,Kc] -> [B*S*A,D,Kc]
kernels: [B*S*A,Kc,1]
conv output: [B,S,A,D] -> contributes H/2 channels
```

Weight transform:

- Depthwise Conv1d weight source shape is `[H,1,Kc]`.
- Pointwise Conv1d weight source shape is `[H/2,H,1]`, flattenable to Linear `H -> H/2`.
- `conv_out_layer` and `conv_kernel_layer` are normal PyTorch Linear weights `[out,in]`.

Layout constraints:

- Source uses Conv1d NCL only internally. A DinoML `[B,S,H]` local-window implementation may avoid transposes if it keeps hidden contiguous and gathers sequence windows.
- A generic NHWC/channel-last pass must not reinterpret the sequence axis as a channel axis. Protect the dynamic-conv region with a no-layout-translation guard unless the pass explicitly rewrites Conv1d axes and weights.

Failure cases:

- Even `conv_kernel_size` would change padding/window alignment; source configs do not exercise it.
- `num_groups` does not apply to this attention convolution path.
- Cross-attention still uses dynamic convolution from decoder hidden states, not encoder hidden states.

Parity test sketch:

- Compare intermediate `dyn_kernel`, unfolded window matmul output, and final mixed context for small, medium-small, and base.

### Rewrite: grouped FFN LinearLayer to grouped GEMM

Source pattern:

```text
reshape [B,S,in] -> [-1,G,in/G] -> permute [G,B*S,in/G]
matmul weight[G,in/G,out/G] -> permute -> reshape [B,S,out] -> bias
```

Replacement:

```text
GroupedGEMM(G cases of (B*S x in/G) @ (in/G x out/G)) -> concat group outputs -> bias
```

Preconditions:

- `input_size % G == 0` and `output_size % G == 0`.
- Hidden features are partitioned contiguously by group.
- Weight layout remains `[G,in/G,out/G]`, not PyTorch Linear `[out,in]`.

Failure cases:

- `G=1` should use normal Linear/GEMM.
- Non-contiguous feature-group layouts require explicit copies or a custom kernel.

Parity test sketch:

- Medium-small config: `LinearGrouped(384 -> 1536, G=2)` and `LinearGrouped(1536 -> 384, G=2)`.

### Rewrite: Q/K/V projections to packed reduced-QKV

Source pattern:

```text
q = Linear(H,H/2)(x); k = Linear(H,H/2)(x); v = Linear(H,H/2)(x)
```

Replacement:

```text
PackedLinear(H,3H/2) -> split [q,k,v]
```

Preconditions:

- Self-attention path only; cross-attention uses encoder hidden states for K/V.
- Bias exists for all three projections.
- Split order is `[q, k, v]`.

Weight transform:

```python
w_qkv = concat([w_q, w_k, w_v], axis=0)
b_qkv = concat([b_q, b_k, b_v], axis=0)
```

Failure cases:

- Do not pack `key_conv_attn_layer`; it is a separate separable convolution path.

Parity test sketch:

- Compare split tensors before reshape to `[B,A,S,D]`.

### Rewrite: bias Linear over `[B,S,*]` to GEMM

Source pattern:

```text
nn.Linear(in,out)(dense [B,S,in])
```

Replacement:

```text
Flatten B*S -> GEMM_RCR_Bias(in,out) -> reshape [B,S,out]
```

Preconditions:

- Dense row-major logical input.
- PyTorch Linear weight orientation `[out,in]`.
- Preserve tied LM head alias when applicable.

Failure cases:

- `GroupedLinearLayer` has a different weight layout and must not use this rewrite.

Parity test sketch:

- Cover all non-grouped projections, output layers, embedding projection, LM transform, and classification heads.

## 10. Kernel fusion candidates

Highest priority:

- Dynamic convolution attention primitive: this is the family-defining operator and the biggest gap versus BERT/ELECTRA lowering.
- Bias GEMM/packed reduced-QKV for `H -> H/2` Q/K/V projections.
- Local-window gather plus grouped small matmul for `D x Kc` dynamic convolution windows.
- Separable Conv1d over sequence: depthwise temporal convolution and pointwise projection.
- Residual add + LayerNorm after attention and FFN.

Medium priority:

- Embedding sum + LayerNorm + optional projection for small/medium-small.
- Grouped FFN GEMM for medium-small `num_groups=2`.
- GELU fusion in FFN and task heads.
- Masked-LM transform and tied vocab GEMM; optional masked-position-only logits for fill-mask workloads.

Lower priority:

- Multiple-choice `SequenceSummary` variants beyond standard `first`/`last`.
- Cross-attention branch without cache.
- Training loss paths and dropout.

## 11. Runtime staging plan

Stage 1: Parse `ConvBertConfig`, including effective head count, `D`, `all_head_size`, `embedding_size`, `head_ratio`, `conv_kernel_size`, and `num_groups`.

Stage 2: Load embeddings and run embedding path parity, including optional `Linear(E -> H)`.

Stage 3: Implement and test separable Conv1d alone on `[B,H,S]` and a direct `[B,S,H]` local-conv optimized equivalent.

Stage 4: Implement dynamic convolution kernel generation and local-window matmul parity against `ConvBertSelfAttention`.

Stage 5: Implement one full ConvBERT encoder block with dense attention half, convolution half, concat, output projection, residual LayerNorm, and FFN.

Stage 6: Full `ConvBertModel` encoder parity for small, medium-small, and base.

Stage 7: Add `ConvBertForMaskedLM` with tied embedding/LM head support.

Stage 8: Add optional downstream heads: sequence classification, token classification, QA, and multiple choice.

Stage 9: Add optimized fusions: packed QKV, fused local dynamic conv path, grouped FFN GEMM, residual LayerNorm.

Stage 10: Treat decoder/cross-attention as a separate follow-up and keep cache unsupported unless a concrete checkpoint requires it.

## 12. Parity and validation plan

- Config parsing tests for six representative configs, especially medium-small grouped FFN and small/base `E != H` differences.
- Embedding parity with default and explicit `token_type_ids` and `position_ids`.
- Separable Conv1d parity: depthwise+pointwise+bias on random `[B,H,S]`.
- Dynamic kernel generation parity: compare `kc * q`, generated kernels, and `softmax(dim=Kc)`.
- Local-window extraction parity versus `torch.nn.functional.unfold`, including left/right zero padding.
- Single dense attention half parity with additive padding mask.
- Full `ConvBertSelfAttention` parity for small, medium-small, and base shapes.
- GroupedLinearLayer parity for `G=2`.
- Single encoder block parity after attention residual and FFN residual.
- Full encoder last-hidden-state parity at `S=8`, `S=128`, and `S=512`.
- Masked LM logits parity, including tied vocab projection.
- Optional heads parity: sequence classification CLS head, token classification, QA split/squeeze, multiple choice flatten/summary.
- Suggested tolerances: fp32 `rtol=1e-5, atol=1e-6`; fp16/bf16 only after reduced-precision local-conv admission, likely `rtol=2e-2, atol=2e-2`.

## 13. Performance probes

- Encoder throughput sweep over batch size and sequence length for small, medium-small, and base.
- Isolate separable Conv1d, dynamic kernel generation, local-window unfold/gather, dense attention, and FFN timings.
- Compare source-like transpose+Conv1d+unfold implementation versus `[B,S,H]` fused local-window implementation.
- Probe `Kc=9` local-window matmul occupancy across `D=48` and `D=64`.
- Compare packed reduced-QKV versus three independent GEMMs.
- Medium-small grouped FFN grouped-GEMM throughput versus dense fallback.
- Base vocab projection throughput for masked LM; compare full `[B,S,V]` logits with masked-position-only projection.
- Memory bandwidth and temporary allocation probe for unfold windows, because materializing `[B,S,H/2,Kc]` can dominate at long sequence length.
- Attention backend comparison for dense half: eager matmul/softmax versus SDPA-style backend plus separate conv path.

## 14. Skip/defer list

- Training losses and label preprocessing.
- Dropout and gradient checkpointing behavior.
- Decoder/cross-attention mode unless a real checkpoint requires it.
- KV cache or generation cache; absent from inspected source.
- SequenceSummary `attn` mode, which source raises as not implemented.
- Multi-GPU/tensor parallel sharding.
- Quantization/packed weight formats; no source-coupled quantized format appears in inspected configs.
- Generic NHWC/channel-last layout translation for this text model. Only consider guarded local-conv layout rewrites.

## 15. Final implementation checklist

- [ ] Parse `ConvBertConfig` and derive effective `A`, `D`, and `all_head_size`.
- [ ] Load word, token type, and position embeddings with width `E`.
- [ ] Implement embedding sum, LayerNorm(E), and optional `Linear(E -> H)`.
- [ ] Implement additive bidirectional attention mask expansion.
- [ ] Implement bias GEMM lowering for `Linear(H -> H/2)` Q/K/V.
- [ ] Implement separable Conv1d depthwise+pointwise+bias.
- [ ] Implement dynamic convolution kernel generator and `softmax(dim=Kc)`.
- [ ] Implement local sequence-window gather/unfold with source padding.
- [ ] Implement grouped small matmul for dynamic convolution windows.
- [ ] Concatenate dense attention and conv outputs in source order and project with `Linear(H -> H)`.
- [ ] Implement residual add + LayerNorm for attention and FFN outputs.
- [ ] Implement normal FFN Linear path and grouped FFN path.
- [ ] Implement `ConvBertForMaskedLM` transform and tied LM head.
- [ ] Add optional classifier, token, QA, and multiple-choice heads.
- [ ] Add parity tests for small, medium-small, base, and vocab-variant configs.
- [ ] Add packed reduced-QKV rewrite with split-order tests.
- [ ] Add local dynamic-conv fusion with layout/axis guard tests.
- [ ] Benchmark conv path, attention path, FFN, grouped FFN, and masked-LM logits.
