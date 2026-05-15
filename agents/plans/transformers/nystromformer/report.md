# Nystromformer Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: nystromformer family; primary checkpoints inspected include uw-madison/nystromformer-{512,1024,2048,4096}
Config source: Hugging Face raw config.json endpoints plus source defaults in configuration_nystromformer.py
Source files inspected:
- transformers/src/transformers/models/nystromformer/configuration_nystromformer.py
- transformers/src/transformers/models/nystromformer/modeling_nystromformer.py
- transformers/src/transformers/models/nystromformer/convert_nystromformer_original_pytorch_checkpoint_to_pytorch.py
- transformers/tests/models/nystromformer/test_modeling_nystromformer.py
Any missing files or assumptions:
- No tokenizer implementation is model-local; checkpoint tokenizer configs point to AlbertTokenizer.
- No remote-code files are required for the native source path inspected here.
- No gated/401 config gaps found for the representative raw configs attempted; a "private" tiny random config was still readable through the raw endpoint.
```

Primary source URLs:

- Transformers source at commit: `https://github.com/huggingface/transformers/tree/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nystromformer`
- HF configs: `https://huggingface.co/uw-madison/nystromformer-512/raw/main/config.json`, `https://huggingface.co/uw-madison/nystromformer-1024/raw/main/config.json`, `https://huggingface.co/uw-madison/nystromformer-2048/raw/main/config.json`, `https://huggingface.co/uw-madison/nystromformer-4096/raw/main/config.json`
- Task variants: `https://huggingface.co/GBaker/nystromformer-4096-medqa-usmle-nocontext/raw/main/config.json`, `https://huggingface.co/MrAnderson/nystrom-1024-full-trivia/raw/main/config.json`

Report target: inference-only CUDA runtime for the encoder and encoder heads, with special attention to Nyström-style attention math. The first useful DinoML target should be `NystromformerForMaskedLM` or `NystromformerModel`; classification, multiple-choice, token classification, and QA heads are small staged add-ons.

## 2. High-level architecture

Nystromformer in this source is a BERT-like bidirectional text encoder:

```text
AlbertTokenizer / caller tokenization
-> token + absolute position + token-type embeddings
-> repeated encoder blocks with self-attention + FFN
-> optional task head: masked LM, sequence classification, multiple choice, token classification, QA
```

There is no autoregressive decoder, cross-attention implementation, RoPE, ALiBi, KV cache, or generation decode loop in the inspected model body. The source returns `BaseModelOutputWithPastAndCrossAttentions`, but the implementation only fills encoder hidden states and self-attentions.

Stage decomposition:

- CPU/data pipeline: Albert-style tokenization, special-token insertion, padding/truncation, token type IDs, attention mask.
- GPU/runtime encoder: embeddings, absolute position lookup, bidirectional attention, residual LayerNorm blocks, FFN.
- Independently stageable heads: MLM projection and tied vocab output, first-token classification, multiple-choice flatten/reshape wrapper, per-token classifier, QA start/end logits.

## 3. Important config dimensions

Source defaults from `NystromformerConfig`:

| Field | Default | Runtime significance |
|---|---:|---|
| `vocab_size` | 30000 | token embedding and MLM decoder width |
| `hidden_size` | 768 | encoder width |
| `num_hidden_layers` | 12 | block repeat count |
| `num_attention_heads` | 12 | MHA heads |
| derived `head_dim` | 64 | `hidden_size / num_attention_heads`; source rejects non-divisible hidden size |
| `intermediate_size` | 3072 | FFN expansion |
| `hidden_act` | `gelu_new` | FFN and MLM transform activation |
| `max_position_embeddings` | 510 | source allocates `max_position_embeddings + 2` position rows but default `position_ids` length is `max_position_embeddings` with IDs starting at 2 |
| `type_vocab_size` | 2 | token type embedding rows |
| `segment_means_seq_len` | 64 | source field named `seq_len` inside attention; controls landmark segment reshape, not dynamically read from input |
| `num_landmarks` | 64 | landmark count; if equal to `segment_means_seq_len`, source takes dense attention branch |
| `conv_kernel_size` | 65 | optional depthwise Conv2d over value states, groups=heads |
| `inv_coeff_init_option` | `False` | if true, source tries a different inverse initialization but references `config["inv_init_coeff_option"]`, which is not a declared field |
| `layer_norm_eps` | `1e-5` | all LayerNorms |
| `tie_word_embeddings` | `True` | MLM decoder weight tied to input token embedding by pretrained-model tying |

Representative checkpoint sweep is in `config_snapshots.md`. Operator-significant variation found:

| Checkpoint | Task/head | Max positions | Body variation |
|---|---|---:|---|
| `uw-madison/nystromformer-512` | masked LM | 510 | 12x768 encoder, 12 heads, `num_landmarks=segment_means_seq_len=64` |
| `uw-madison/nystromformer-1024` | masked LM | 1024 | same body, longer position table |
| `uw-madison/nystromformer-2048` | masked LM | 2048 | same body, longer position table |
| `uw-madison/nystromformer-4096` | masked LM | 4096 | same body, longer position table |
| `GBaker/nystromformer-4096-medqa-usmle-nocontext` | multiple choice | 4096 | same encoder plus first-token MC head |
| `MrAnderson/nystrom-1024-full-trivia` | QA | 1024 | same encoder plus start/end projection |
| `hf-tiny-model-private/tiny-random-NystromformerForMaskedLM` | tiny masked LM | 512 | 5 layers, hidden 32, heads 4, `gelu`, wider type vocab |

## 3a. Family variation traps

- The source branches to true Nyström approximation only when `num_landmarks != segment_means_seq_len`. All official UW configs inspected set both to 64, so they use the dense attention branch in this pinned source.
- If DinoML admits configs with `num_landmarks < segment_means_seq_len`, the source uses fixed `segment_means_seq_len // num_landmarks` reshape logic. This requires strong guards: `segment_means_seq_len % num_landmarks == 0` and input sequence/batch broadcasting behavior must match PyTorch exactly or be rejected.
- `max_position_embeddings` varies from 510 to 4096. Position IDs are offset by `+2`; embedding table has two extra rows.
- `conv_kernel_size=None` disables the depthwise value convolution; official configs use 65.
- `inv_coeff_init_option=True` appears broken in this source because `config["inv_init_coeff_option"]` is not a declared config field. First integration should reject this option or verify source behavior before accepting it.
- Configs may advertise `use_cache`, but the modeling source does not implement KV cache or decoder cache semantics.
- `add_cross_attention` is stored in layers but no cross-attention module or forward path exists. Reject configs that require cross-attention.
- `chunk_size_feed_forward` comes from `PreTrainedConfig`; nonzero values use `apply_chunking_to_forward`, an inference-time graph partitioning behavior. First DinoML import can require zero/default chunking or rewrite it to equivalent unchunked FFN.
- 4096 tokenizer config reports `model_max_length=512`; end-to-end parity for long contexts needs caller/tokenizer truncation policy outside the neural graph.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup for token IDs, token type IDs, and absolute position IDs.
- Broadcast/add of three embedding tensors `[B,S,H]`.
- Default token type expansion from registered `[1,max_pos]` buffer to `[B,S]`.
- Attention mask expansion through `get_extended_attention_mask`: typical encoder mask becomes `[B,1,1,S]` additive large-negative mask.
- `view`, `reshape`, `transpose`, `permute`, `contiguous`, `split`, `squeeze`, first-token slice `hidden[:,0,:]`.
- Multiple-choice flatten `[B,C,S] -> [B*C,S]` and output reshape `[B*C,1] -> [B,C]`.

Neural network primitives:

- Linear with bias: Q/K/V `768 -> 768`, attention output `768 -> 768`, FFN `768 -> 3072 -> 768`, MLM transform `768 -> 768`, MLM decoder `768 -> vocab`, heads.
- LayerNorm over hidden axis with eps `1e-5`.
- Activations: `gelu_new` for official checkpoints, `gelu` in tiny random; ReLU in multiple-choice head.
- Dropout is present in source but can be compiled as identity for inference/eval.

Attention primitives:

- Dense bidirectional MHA path for official configs: Q/K scaling by `1 / sqrt(sqrt(head_dim))` each, matmul scores, additive mask, softmax over key axis, matmul with V.
- Nyström approximation path for non-equal landmark configs: landmark mean pooling, three attention kernels, iterative pseudoinverse, chained matmuls.
- Optional depthwise Conv2d value residual: input `[B, heads, S, head_dim]`, `groups=heads`, kernel `(conv_kernel_size,1)`, padding `(kernel//2,0)`, no bias.

Position and token-type ops:

- Absolute learned position embeddings only; no RoPE/relative bias.
- Position IDs default to `[2, 3, ..., S+1]`, not zero-based.
- Token type IDs default to zeros and enter the graph through embedding lookup.

Generation/cache ops:

- Not applicable for primary target. No causal mask, prefill/decode split, or KV cache.

Preprocessing-coupled ops:

- AlbertTokenizer special-token layout and IDs matter for end-to-end parity, but tokenizer is outside the model graph.
- `[CLS]`/first token is consumed by classification and multiple-choice heads.

Parameter sharing:

- MLM decoder weight is tied to `nystromformer.embeddings.word_embeddings.weight`.
- MLM decoder bias is tied to `cls.predictions.bias` by `_tied_weights_keys`; keep as one logical bias.

## 5. Layer/block breakdown

Embeddings:

```text
input_ids [B,S] or inputs_embeds [B,S,H]
token_embeds = Embedding(vocab_size,H)(input_ids)
type_embeds = Embedding(type_vocab_size,H)(token_type_ids)
pos_embeds = Embedding(max_position_embeddings+2,H)(position_ids, default 2..S+1)
x = LayerNorm(token_embeds + type_embeds + pos_embeds)
```

Encoder block, repeated `num_hidden_layers` times:

```text
q = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
k = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
v = Linear(H -> H)(x).view(B,S,heads,head_dim).transpose(1,2)
q = q / sqrt(sqrt(head_dim)); k = k / sqrt(sqrt(head_dim))
context = dense_mha_or_nystrom(q,k,v, additive_attention_mask)
if conv_kernel_size is not None: context += depthwise_conv2d(v)
context = context.permute(B,S,heads,head_dim).reshape(B,S,H)
x1 = LayerNorm(Linear(H -> H)(context) + x)
ff = Linear(H -> intermediate)(x1)
ff = activation(ff)
x = LayerNorm(Linear(intermediate -> H)(ff) + x1)
```

Heads:

- MLM: `Linear(H -> H) -> activation -> LayerNorm -> tied Linear(H -> vocab_size)`.
- Sequence classification: first token `[:,0,:] -> Dropout -> Linear(H,H) -> activation -> Dropout -> Linear(H,num_labels)`.
- Multiple choice: flatten choices into batch, first token, `Linear(H,H) -> ReLU -> Linear(H,1) -> reshape [B,C]`.
- Token classification: per-token `Linear(H,num_labels)`.
- QA: per-token `Linear(H,2) -> split start/end -> squeeze`.

## 6. Attention requirements

Official checkpoint path in the inspected source:

- Type: noncausal bidirectional encoder self-attention.
- Heads: MHA, `num_attention_heads=12`, `head_dim=64`; no GQA/MQA.
- Q/K/V width: all `hidden_size`.
- Mask: additive encoder mask broadcastable to attention scores. For dense path, score shape `[B,heads,S,S]`. For Nyström `kernel_3`, score shape `[*,heads,num_landmarks,S]`.
- Cache: none.
- FlashAttention/SDPA compatibility: official dense branch can map to noncausal attention if the value convolution residual is applied separately and Q/K pre-scaling matches source math. The source does not call PyTorch SDPA.

True Nyström approximation path, if admitted:

```text
q_landmarks = q.reshape(-1, heads, num_landmarks, segment_len, head_dim).mean(dim=-2)
k_landmarks = k.reshape(-1, heads, num_landmarks, segment_len, head_dim).mean(dim=-2)
kernel_1 = softmax(q @ k_landmarks^T, dim=-1)                # [?, heads, S, L]
kernel_2 = softmax(q_landmarks @ k_landmarks^T, dim=-1)      # [?, heads, L, L]
kernel_3 = softmax(q_landmarks @ k^T + mask, dim=-1)         # [?, heads, L, S]
context = (kernel_1 @ iterative_inv(kernel_2)) @ (kernel_3 @ v)
```

Admission recommendation:

- Stage 1 can support only the dense branch used by official configs: require `num_landmarks == segment_means_seq_len`.
- Stage 2 should add a dedicated Nyström attention op or rewrite region with guards. Do not lower it as generic dense attention because the pseudoinverse chain changes both temp shapes and numerics.
- Reject `output_attentions=True` initially if DinoML does not expose dense or approximated `attention_probs` tensors. In Nyström mode the returned `attention_probs` is `kernel_1 @ pinv(kernel_2)`, not the final dense `[S,S]` probability matrix.

## 7. Position encoding and custom math

Position encoding is learned absolute embedding with an offset:

```python
position_ids = arange(max_position_embeddings).expand(1, -1) + 2
position_ids = position_ids[:, :seq_length]
position_embeddings = position_embedding_table[position_ids]
```

Nyström iterative inverse is the custom math that needs exact parity if non-default configs are admitted:

```python
def iterative_inv(key, init_option="original", n_iter=6):
    I = eye(key.size(-1), device=key.device)
    if init_option == "original":
        z = (1 / max(sum(key, dim=-2))) * transpose(key, -1, -2)
    else:
        z = (1 / max(sum(key, dim=-2), dim=-1).values[:, :, None, None]) * transpose(key, -1, -2)
    for _ in range(n_iter):
        kz = key @ z
        z = (0.25 * z) @ (13 * I - kz @ (15 * I - kz @ (7 * I - kz)))
    return z
```

Precomputable:

- Identity matrix of size `num_landmarks`.
- Position ID buffer for max sequence length.

Dynamic per input:

- Landmark means, softmax kernels, inverse iterations, and attention-mask addition.

## 8. Preprocessing and input packing

Neural graph inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,H]`, mutually exclusive.
- `attention_mask [B,S]` by normal tokenizer path; source also allows a broadcastable 3D self-attention mask through `get_extended_attention_mask`.
- `token_type_ids [B,S]`; defaults to all zeros.
- `position_ids [1,S]` or `[B,S]`; defaults to offset absolute positions.

Tokenizer coupling:

- Official tokenizer configs use `AlbertTokenizer`, lowercase text, remove spaces, and special tokens `[CLS]`, `[SEP]`, `<unk>`, `<pad>`, `[MASK]`.
- The model graph does not parse text and does not own SentencePiece tokenization.
- Long-context end-to-end runs must guard tokenizer `model_max_length`; at least the 4096 tokenizer config observed reports 512 despite a 4096 model position table.

## 9. Graph rewrite / lowering opportunities

### Rewrite: official dense attention branch -> standard noncausal attention

Source pattern:

```text
q = Linear(x) -> view/transpose
k = Linear(x) -> view/transpose
v = Linear(x) -> view/transpose
q,k pre-scaled by 1/sqrt(sqrt(head_dim))
softmax(q @ k^T + mask) @ v
```

Replacement:

```text
FusedQKV(optional) -> NonCausalAttention(scale=1/sqrt(head_dim), additive_mask) -> output projection
```

Preconditions:

- `num_landmarks == segment_means_seq_len`.
- `output_attentions=False` or attention tensor ABI implemented.
- Mask conversion matches Transformers additive mask.
- Keep source Q/K split scaling equivalent to a single score scale.

Failure cases:

- `num_landmarks != segment_means_seq_len`.
- Caller asks for attentions and DinoML cannot return source-shaped attention probabilities.

Parity sketch: compare one attention module with random masks in fp32, then full block parity.

### Rewrite: separate Q/K/V linear -> packed QKV GEMM

Preconditions:

- Same input tensor `x`.
- Same output width `hidden_size` for all three.
- Biases present and packed in Q,K,V order.

Replacement:

```text
GEMM(x, concat(Wq, Wk, Wv)^T) + concat(bq,bk,bv) -> split [Q,K,V]
```

Weight transform: concatenate PyTorch linear weights along output-feature axis in Q,K,V order.

Failure cases: custom weight loading that requires preserving separate parameter names can still pack at compile time but must keep alias/provenance metadata.

### Rewrite: depthwise Conv2d over V -> local sequence depthwise convolution

Source pattern:

```text
value_layer [B, heads, S, head_dim]
Conv2d(groups=heads, kernel=(K,1), padding=(K//2,0), bias=False)
```

Replacement:

```text
DepthwiseSequenceConv(axis=S, channels=heads, width=head_dim, kernel=K) -> add to context
```

Preconditions:

- Input layout exactly `[B,heads,S,head_dim]`.
- Kernel only spans sequence axis; no mixing across head_dim.
- `K` odd for same-length behavior; official `65` is odd. If even kernels are admitted, match PyTorch padding/output length exactly.

Failure cases: generalized NHWC/channel-last layout translation must rewrite axes or guard this region as no-layout-translation.

### Rewrite: Nyström attention region -> dedicated approximation op

Preconditions:

- `num_landmarks != segment_means_seq_len`.
- `segment_means_seq_len % num_landmarks == 0`.
- Runtime input lengths and reshape/broadcast behavior are explicitly admitted.
- `inv_coeff_init_option=False` for first pass.

Replacement:

```text
LandmarkMeanPool -> three batched GEMMs -> row softmaxes -> six-step iterative inverse -> two batched GEMMs
```

Failure cases:

- `inv_coeff_init_option=True` until source typo/behavior is resolved.
- Dynamic sequence lengths not compatible with the fixed segment reshape.
- `output_attentions=True` unless returned approximate attention ABI is implemented.

### Rewrite: first-token heads

Source pattern:

```text
sequence_output[:,0,:] -> small MLP/classifier
```

Replacement: gather first row or configure encoder to materialize first-token-only head path after full encoder.

Preconditions: special-token policy guarantees `[CLS]` at index 0 for classification tasks.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + residual add for attention output and FFN output. It is repeated twice per block and must match eps `1e-5`.
- QKV packed GEMM plus reshape/transposes. This is the main dense projection cost.
- Dense noncausal attention for official configs, with Q/K scale folded into attention score scale.
- Depthwise sequence convolution over V. Official configs use it in every layer; a generic Conv2d path would work but a sequence-axis depthwise kernel avoids awkward layout churn.

Medium priority:

- FFN `Linear -> gelu_new -> Linear` with bias and residual-LayerNorm following.
- MLM head `Linear -> gelu_new -> LayerNorm -> tied vocab GEMM`.
- Nyström approximation kernels for non-default configs: landmark mean pooling, small `L x L` iterative inverse, and chained batched GEMMs. Worth doing only after dense official checkpoints are stable.

Lower priority:

- Multiple-choice and sequence-classification head fusion; small relative to encoder.
- Tokenizer-adjacent position/token-type default generation on device.
- Returning attention tensors for diagnostics.

## 11. Runtime staging plan

Stage 1: config and weights.

- Parse Nystromformer config and reject unsupported traps: `add_cross_attention=True`, `inv_coeff_init_option=True`, nonzero feed-forward chunking if not rewritten, and `num_landmarks != segment_means_seq_len`.
- Load embeddings, encoder blocks, and tied MLM decoder.

Stage 2: encoder dense-branch parity.

- Implement embeddings, mask expansion, dense MHA branch, value depthwise convolution, residual LayerNorm, FFN.
- Validate `NystromformerModel` and `NystromformerForMaskedLM` on `uw-madison/nystromformer-512`.

Stage 3: task heads.

- Add sequence classification, token classification, QA, and multiple choice.
- Multiple choice can be a wrapper around encoder batch flattening.

Stage 4: optimized lowering.

- QKV packing, fused LayerNorm/residual, attention backend selection, sequence-depthwise conv kernel.

Stage 5: true Nyström approximation.

- Add guarded landmark mean pooling and iterative inverse op.
- Support representative synthetic configs first; then admit real checkpoints only if configs use this path.

Stage 6: long-context validation.

- Validate 1024/2048/4096 position tables and tokenizer truncation policy.
- Benchmark dense branch memory; if official configs remain dense in source, long-context runtime may be quadratic despite the family name.

## 12. Parity and validation plan

- Config parser tests: official 512/1024/2048/4096 configs, tiny random config, rejection tests for unsupported flags.
- Embedding parity: token/type/position sum with offset position IDs and LayerNorm.
- Attention unit parity:
  - dense branch with random `[B,S,H]`, binary masks, `num_landmarks == segment_means_seq_len`;
  - conv residual on/off;
  - Nyström synthetic branch with small `segment_means_seq_len` and landmarks after Stage 5.
- Iterative inverse parity: random row-softmax `kernel_2 [B,heads,L,L]`, compare six iterations in fp32.
- Single-layer parity: one encoder layer with deterministic eval mode.
- Full encoder parity: `uw-madison/nystromformer-512`, expected hidden shape `[1,6,768]` and slice parity similar to upstream integration test.
- MLM parity: sentence `"the [MASK] of Belgium is Brussels"` should predict `"capital"` for the 512 checkpoint in upstream test.
- Head parity: sequence classification `[B,num_labels]`, token classification `[B,S,num_labels]`, QA start/end `[B,S]`, multiple choice `[B,C]`.
- Tolerances: fp32 `rtol=1e-4, atol=1e-4` for full model; fp16/bf16 should use looser tolerances and isolate softmax/LayerNorm drift.

## 13. Performance probes

- Encoder-only throughput by sequence length: 512, 1024, 2048, 4096.
- Attention backend comparison for official dense branch: unfused matmul-softmax, fused noncausal attention, and attention plus separate depthwise value conv.
- Memory scaling probe: confirm whether official configs are quadratic under current source because they take dense branch.
- Depthwise conv kernel benchmark across `S` and `conv_kernel_size=65`.
- QKV packed GEMM versus three separate GEMMs.
- FFN GEMM throughput and activation overhead.
- MLM vocab projection throughput, especially tied output weight layout.
- Nyström synthetic probe: vary landmarks `L`, sequence `S`, inverse iteration count fixed at 6; separate landmark pooling, `kernel_2` inverse, and final chained GEMMs.
- End-to-end masked-LM latency/throughput with tokenizer excluded and included as separate CPU/data-pipeline probe.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout randomness; compile inference/eval only.
- `output_attentions=True` unless attention-output ABI is required.
- `inv_coeff_init_option=True` until the source config typo/path is resolved.
- `add_cross_attention=True`; source does not implement it.
- KV cache, causal decoding, beam search, generation controllers; not applicable.
- General arbitrary chunked FFN execution; start with default unchunked inference.
- True Nyström approximation for official-checkpoint parity if official configs continue to use dense branch in this source.
- Tokenizer execution inside DinoML; keep tokenization in CPU/data pipeline.

## 15. Final implementation checklist

- [ ] Parse `NystromformerConfig` and checkpoint config defaults.
- [ ] Add admission guards for cross-attention, inverse-init option, feed-forward chunking, and Nyström branch support level.
- [ ] Load token, position, token-type embeddings and tied MLM decoder weights.
- [ ] Implement embedding sum with `+2` position offset and token type defaulting.
- [ ] Implement encoder additive attention mask expansion.
- [ ] Implement dense noncausal MHA branch with source Q/K scaling.
- [ ] Implement optional depthwise sequence convolution over `value_layer`.
- [ ] Implement residual add + LayerNorm blocks and FFN with `gelu_new`.
- [ ] Implement MLM head with tied decoder and output bias alias.
- [ ] Add sequence classification, multiple-choice, token classification, and QA heads.
- [ ] Add QKV packed projection rewrite with Q,K,V split-order tests.
- [ ] Add guarded depthwise Conv2d-to-sequence-conv rewrite.
- [ ] Add one-block and full-model parity tests against `uw-madison/nystromformer-512`.
- [ ] Add long-position-table parity probes for 1024/2048/4096 configs.
- [ ] Add synthetic Nyström attention parity tests before admitting `num_landmarks != segment_means_seq_len`.
- [ ] Benchmark dense attention, value convolution, FFN, and MLM projection separately.
