# RoFormer full-audit report

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` in `transformers`.

Model id: primary target `junnyu/roformer_chinese_base`; sweep also covered `junnyu/roformer_chinese_small`, `junnyu/roformer_chinese_char_small`, `junnyu/roformer_chinese_char_base`, and `junnyu/roformer_v2_chinese_char_base`.

Config source: `src/transformers/models/roformer/configuration_roformer.py` plus open Hugging Face `config.json` files. Config snapshots are in `agents/plans/transformers/roformer/config_snapshots.md`.

Source files inspected:

- `src/transformers/models/roformer/modeling_roformer.py`
- `src/transformers/models/roformer/configuration_roformer.py`
- `src/transformers/models/roformer/tokenization_roformer.py`
- `src/transformers/models/roformer/tokenization_utils.py`
- `tests/models/roformer/test_modeling_roformer.py`
- `docs/source/en/model_doc/roformer.md`

Any missing files or assumptions: no remote-code files are needed for the inspected in-library source. No gated or 401 configs were encountered. `roformer_v2_chinese_char_base` advertises `norm_type=rms_norm` and `use_bias=false`, but the current in-library source does not read those fields; treat that checkpoint as a config/source divergence unless a separate RoFormer-v2 audit proves the intended behavior.

## 2. High-level architecture

RoFormer is a BERT-like text encoder with rotary position embedding applied inside self-attention. The first useful DinoML target should be encoder masked-LM parity, because all representative public checkpoints use `RoFormerForMaskedLM`.

Dataflow:

```text
tokenizer/input ids + token_type_ids + attention_mask
-> word/type embeddings
-> embedding LayerNorm/dropout
-> optional embedding_size -> hidden_size projection
-> N post-norm RoFormer encoder blocks with RoPE self-attention
-> MLM transform/projection
-> vocab logits
```

Optional source paths include sequence classification, token classification, question answering, multiple choice, and a decoder/causal-LM mode with KV cache and optional cross-attention. These are implemented, but they are secondary for the checkpoint-backed masked-LM target.

## 3. Important config dimensions

Source defaults:

| Field | Default | Runtime meaning |
|---|---:|---|
| `vocab_size` | 50000 | token embedding rows and MLM logits width |
| `embedding_size` | defaults to `hidden_size` | embedding table width; optional projection to hidden width |
| `hidden_size` | 768 | block residual width |
| `num_hidden_layers` | 12 | repeated encoder/decoder layers |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | `hidden_size // num_attention_heads` | RoPE table width and Q/K/V head width |
| `intermediate_size` | 3072 | FFN expansion |
| `max_position_embeddings` | 1536 | learned module buffer size for sinusoidal RoPE table |
| `type_vocab_size` | 2 | token type embedding rows |
| `hidden_act` | `gelu` | FFN and MLM transform activation |
| `layer_norm_eps` | `1e-12` | all LayerNorm eps |
| `rotary_value` | `False` | if true, apply RoPE to V as well as Q/K |
| `is_decoder` | `False` | enables causal mask and cache |
| `add_cross_attention` | `False` | adds cross-attention layers only when decoder |
| `use_cache` | `True` | effective only when `is_decoder=True` |
| `tie_word_embeddings` | `True` | MLM/CLM decoder weight tied to input embeddings |

Representative checkpoint sweep:

| Model id | Hidden | Embedding | Layers | Heads | Head dim | FFN | Vocab | Max positions | Activation | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `junnyu/roformer_chinese_small` | 384 | 384 | 6 | 6 | 64 | 1536 | 50000 | source default 1536; tokenizer says 512 | gelu | open Transformers fill-mask checkpoint |
| `junnyu/roformer_chinese_base` | 768 | 768 | 12 | 12 | 64 | 3072 | 50000 | 1536 | gelu | common base checkpoint |
| `junnyu/roformer_chinese_char_small` | 384 | 384 | 6 | 6 | 64 | 1536 | 12000 | 512 | gelu | char vocab, `rotary_value=false` |
| `junnyu/roformer_chinese_char_base` | 768 | effective 768 | 12 | 12 | 64 | 3072 | 12000 | 512 | gelu | omits `embedding_size`; source fills default |
| `junnyu/roformer_v2_chinese_char_base` | 768 | 768 | 12 | 12 | 64 | 3072 | 12000 | 512 | relu | source ignores advertised `norm_type`/`use_bias` |

## 3a. Family variation traps

- `embedding_size` may differ from `hidden_size`; the source inserts `Linear(embedding_size -> hidden_size)` after embedding LayerNorm.
- `hidden_size % num_attention_heads` should be admitted explicitly. The source guard is weakened by the always-present `embedding_size` attribute; DinoML should reject non-divisible configs rather than silently truncating `head_dim`.
- `rotary_value=True` changes attention math by rotating V before attention value matmul.
- `is_decoder=True` changes mask semantics, enables cache, and allows `RoFormerForCausalLM`; most public checkpoints are encoder masked-LM.
- `add_cross_attention=True` adds a full extra attention module per layer and requires decoder mode.
- `roformer_v2` configs contain fields that current source ignores. Do not implement RMSNorm/no-bias based only on those config keys for this source basis.
- `hidden_act` is config-driven through `ACT2FN`; observed values include `gelu` and `relu`.
- Tokenization varies between `RoFormerTokenizer` and `BertTokenizer`, but both produce BERT-style special-token and token-type layouts.
- There are no image/video tensors; NHWC/NCHW layout translation is not applicable. Axis-sensitive text ops that need guards are softmax `dim=-1`, sequence mean `dim=1`, gather over sequence axis `-2`, MLM/QA splits on last dim, and multiple-choice flatten/unflatten of `[B, choices, S]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- integer token embedding lookup `[B,S] -> [B,S,E]`
- optional `inputs_embeds` path `[B,S,E]`
- token type embedding lookup and add
- reshape/view for Q/K/V `[B,S,H] -> [B,S,heads,head_dim]`
- transpose `[B,S,heads,D] -> [B,heads,S,D]`
- transpose/contiguous/view after attention `[B,heads,S,D] -> [B,S,H]`
- broadcast add attention mask `[B,1,1-or-Sq,Sk]`
- slicing `hidden_states[:, -logits_to_keep:, :]` for causal LM
- split QA logits last dim 2 into start/end
- multiple-choice flatten `[B,C,S] -> [B*C,S]` and logits reshape `[B*C,1] -> [B,C]`
- optional sequence summary gather for `summary_type=cls_index`

Neural primitives:

- Embedding `vocab_size x embedding_size`, padding row `pad_token_id`
- Embedding `type_vocab_size x embedding_size`
- LayerNorm over last dim, eps `1e-12` by default
- Linear Q/K/V `hidden_size -> hidden_size`, bias present
- Linear attention output `hidden_size -> hidden_size`, bias present
- Linear FFN up `hidden_size -> intermediate_size`, bias present
- activation from `ACT2FN`, usually GELU, sometimes ReLU
- Linear FFN down `intermediate_size -> hidden_size`, bias present
- residual add followed by LayerNorm, BERT post-norm ordering
- MLM transform `hidden_size -> embedding_size`, activation, LayerNorm
- tied decoder Linear `embedding_size -> vocab_size`, bias present
- classification dense `hidden_size -> hidden_size`, activation, output projection
- token-classification Linear `hidden_size -> num_labels`
- QA Linear `hidden_size -> 2`

Attention primitives:

- dense MHA self-attention, no GQA/MQA
- optional dense cross-attention in decoder mode
- Q/K/V RoPE for self-attention; optional V RoPE
- matmul scores `[B,H,Sq,D] x [B,H,D,Sk] -> [B,H,Sq,Sk]`
- scale by `1 / sqrt(head_dim)`
- additive mask, softmax over `Sk`, dropout in training, value matmul

Position/rotary/custom math:

- fixed sinusoidal table `RoFormerSinusoidalPositionalEmbedding(max_position_embeddings, head_dim)`
- table stores all sin features then all cos features, then repeats each scalar into even/odd pairs for rotation
- cache decode offsets positions by `past_key_values_length`

Generation/cache ops:

- optional `DynamicCache` self-attention KV per layer, shape logically `[B,heads,S,D]`
- optional `EncoderDecoderCache` with separate self and cross-attention caches
- cached self-attention keys are stored after RoPE; cached values are post-RoPE only when `rotary_value=True`
- cross-attention cached K/V are projected encoder states and do not receive RoPE

Preprocessing-coupled ops:

- tokenizer emits `[CLS] X [SEP]` or `[CLS] A [SEP] B [SEP]`
- token types are 0 for first segment and first separator, 1 for second segment and trailing separator
- attention mask is ordinary 2D `[B,S]` unless caller supplies a 3D mask

Quantized/packed metadata, sparse attention, distributed/tensor-parallel, and NHWC/NCHW ops: not present in the inspected source.

## 5. Layer/block breakdown

Embedding:

```text
input_ids [B,S] -> word_embeddings [B,S,E]
token_type_ids [B,S] -> type_embeddings [B,S,E]
x = LayerNorm(word + type)
x = dropout(x)                         # inference no-op
if E != H: x = Linear(E -> H)(x)
```

Encoder block, repeated `num_hidden_layers`:

```text
q = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
k = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
v = Linear(H -> H)(x).view(B,S,heads,D).transpose(1,2)
q,k[,v] = RoPE(q,k[,v], sinusoidal_pos)
scores = MatMul(q, k.transpose(-1,-2)) / sqrt(D)
scores = scores + extended_attention_mask
probs = Softmax(scores, dim=-1)
ctx = MatMul(probs, v).transpose(1,2).contiguous().view(B,S,H)
a = LayerNorm(Linear(H -> H)(ctx) + x)
m = activation(Linear(H -> I)(a))
y = LayerNorm(Linear(I -> H)(m) + a)
```

Masked-LM head:

```text
t = Linear(H -> E)(sequence)
t = activation(t)
t = LayerNorm(t)
logits = Linear(E -> vocab_size)(t)    # decoder weight tied to input embeddings
```

## 6. Attention requirements

Primary encoder target:

- noncausal self-attention
- MHA with `num_key_value_heads == num_attention_heads`
- Q/K/V width all `hidden_size`; head dim `hidden_size // heads`
- query length and key/value length are both current sequence length
- additive mask after score scaling and before softmax
- no packed/varlen, local, sliding-window, ALiBi, relative-bias, or FlashAttention dispatch in source
- source uses eager matmul/softmax/matmul, so SDPA/FlashAttention is an optimization rewrite requiring parity tests

Decoder/causal optional target:

- `is_decoder=True` enables inherited causal masking in `get_extended_attention_mask`
- self-attention cache stores per-layer K/V shaped `[B,heads,total_S,D]`
- new token positions use `past_key_values_length` when fetching RoPE rows
- cross-attention, if enabled, projects encoder hidden states to K/V and caches them separately; no RoPE is applied to cross-attention K/V

## 7. Position encoding and custom math

RoFormer uses fixed sinusoidal RoPE over each head dimension. The table has shape `[max_position_embeddings, head_dim]`; for observed checkpoints `head_dim=64`.

Concise source-equivalent math:

```python
def roformer_sinusoidal_table(n_pos, dim):
    enc = [[pos / (10000 ** (2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)]
    sentinel = dim // 2 if dim % 2 == 0 else dim // 2 + 1
    out[:, :sentinel] = sin(enc[:, 0::2])
    out[:, sentinel:] = cos(enc[:, 1::2])
    return out

def apply_roformer_rope(pos, q, k, v=None):
    sin, cos = pos.chunk(2, dim=-1)
    sin = stack([sin, sin], dim=-1).reshape_as(pos)
    cos = stack([cos, cos], dim=-1).reshape_as(pos)
    def rotate_half(x):
        return stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    if v is not None:
        v = v * cos + rotate_half(v) * sin
        return q, k, v
    return q, k
```

Precompute: the sinusoidal table can be materialized as a constant up to `max_position_embeddings`. Runtime-dependent: selecting `[past_len : past_len + S]` for decode, and broadcasting to `[1,1,S,D]`.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- `RoFormerTokenizer` is a fast WordPiece tokenizer with a custom Jieba pre-tokenizer for word-level Chinese checkpoints.
- Character checkpoints advertise `BertTokenizer`.
- Special tokens use BERT layout: single `[CLS] X [SEP]`; pair `[CLS] A [SEP] B [SEP]`.
- Token type ids are segment ids: all 0 for single sequence, and 0 for first segment plus 1 for the second segment/trailing separator.

GPU/runtime graph inputs:

- `input_ids: int64 [B,S]` or `inputs_embeds: float [B,S,E]`, mutually exclusive
- `attention_mask: [B,S]` normally; if absent source creates all ones
- `token_type_ids: int64 [B,S]`; if absent source creates zeros
- optional decoder inputs: `encoder_hidden_states [B,Se,H]`, `encoder_attention_mask [B,Se]`, and cache objects

There are no multimodal placeholder/scatter, image layout, audio feature, packed sequence, or `cu_seqlens` inputs.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate Q/K/V linears -> packed QKV GEMM

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
```

Replacement: one GEMM `Linear(H -> 3H)` then split last dim as `[q, k, v]`.

Preconditions: same input tensor, all biases present and dense, no cross-attention reuse path, no weight tying among Q/K/V. For cross-attention, Q input differs from K/V input, so only K/V can be packed.

Shape equations: input `[B,S,H]`, packed output `[B,S,3H]`, split into three `[B,S,H]`.

Weight transform: concatenate PyTorch Linear weights along output rows: `[Wq; Wk; Wv]`, bias `[bq; bk; bv]`.

Failure cases: partial quantization, missing bias in a future remote implementation, or nonstandard packed checkpoints.

Parity test sketch: compare packed split outputs to three source linears before RoPE for fp32 and reduced precision.

### Rewrite: RoPE + attention backend

Source pattern: Q/K reshape/transpose, source RoPE, score matmul, scale, additive mask, softmax, value matmul.

Replacement: fused RoPE plus dense attention prefill kernel or RoPE followed by SDPA/FlashAttention.

Preconditions: dense MHA, no output attentions required, dropout disabled for inference, mask representable as padding/causal additive mask, `rotary_value=False` for FlashAttention-style Q/K-only RoPE. If `rotary_value=True`, either rotate V separately before backend or reject the fused path.

Shape equations: Q/K/V `[B,Hh,S,D]`; scores `[B,Hh,Sq,Sk]`.

Failure cases: caller requests `output_attentions=True`, 3D arbitrary attention mask not supported by fused backend, or decoder cross-attention mixed with self-attention cache in one fused region.

Parity test sketch: one-layer attention with random masks and cached decode positions; compare context with source eager attention.

### Rewrite: post-norm residual GEMM epilogue

Source pattern:

```text
y = LayerNorm(dropout(Linear(x)) + residual)
```

Replacement: GEMM with bias, residual add, and LayerNorm fusion.

Preconditions: inference dropout disabled; residual tensor same `[B,S,H]`; LayerNorm normalized shape exactly last dim.

Failure cases: training dropout, aliasing output with residual in unsupported ways.

Parity test sketch: attention output projection and FFN down projection fused separately against source block.

### Rewrite: MLM last-token or masked-index logits

Source pattern: full `[B,S,E] -> [B,S,V]` decoder projection.

Replacement: for causal LM `logits_to_keep`, slice hidden states before decoder projection; for masked LM with known mask indices, gather masked rows then project only those rows.

Preconditions: caller only needs selected logits; loss/full logits not requested.

Failure cases: fill-mask APIs often need full sequence logits shape; masked-index gather changes output ABI and must be explicit.

Parity test sketch: compare selected logits to slicing full source logits.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm with residual add for attention and FFN outputs; every block has two post-norms.
- Dense attention with RoPE for encoder prefill; it dominates long sequence masked-LM inference.
- QKV packed GEMM for self-attention; reduces launch count and memory traffic.
- MLM transform/projection, especially tied large vocab projection for `vocab_size=50000`.

Medium priority:

- Embedding add + LayerNorm fusion.
- FFN `Linear -> GELU/ReLU -> Linear`, with activation fused into the first GEMM epilogue where possible.
- Causal decode RoPE + KV cache update for optional `RoFormerForCausalLM`.
- Last-token-only logits for causal LM via `logits_to_keep`.

Lower priority:

- Cross-attention K/V cache packing, because checkpoint-backed RoFormer is encoder masked-LM.
- Multiple-choice sequence summary variants.
- Training losses, dropout, and output-attention materialization.

## 11. Runtime staging plan

Stage 1: parse RoFormerConfig, reject config/source divergences (`norm_type`, `use_bias`, non-divisible heads), load embeddings/encoder/MLM weights, and run embedding plus one block parity.

Stage 2: full encoder masked-LM parity for small/base checkpoints with eager dense attention and full vocab logits.

Stage 3: add source-equivalent RoPE op and tests for cached position offsets, including `rotary_value=True` synthetic config.

Stage 4: lower packed QKV and fused residual LayerNorm while preserving exact post-norm order.

Stage 5: enable optimized dense attention backend with guards for mask shape, output attentions, dropout, and `rotary_value`.

Stage 6: add optional heads: sequence classification, token classification, QA, multiple choice.

Stage 7: add optional decoder/causal-LM cache path, then cross-attention cache path if needed by a real checkpoint.

## 12. Parity and validation plan

- RoPE table parity: compare first rows and odd/even rotation against `RoFormerSinusoidalPositionalEmbedding` tests; tolerance `1e-4` fp32.
- RoPE apply parity: random Q/K/V `[B,H,S,D]`, `rotary_value` false and true.
- Single block parity: random hidden states and masks, no dropout, fp32 tolerance `1e-5` to `1e-4`.
- Full small checkpoint parity: `junnyu/roformer_chinese_small` masked-LM logits for fixed token ids; compare fp32 `rtol=1e-4, atol=1e-4`.
- Base checkpoint parity: `junnyu/roformer_chinese_base` shape `[1,6,50000]` and selected logits matching source integration test.
- Mask tests: absent mask, 2D padding mask, 3D custom mask if supported.
- Optional cache tests: decoder full-prefix versus cached continuation for synthetic config, matching Transformers test style with `atol=1e-3`.
- Reduced precision: fp16/bf16 encoder parity with looser `rtol=1e-2, atol=1e-2`, with fp32 LayerNorm/softmax accumulation preferred.

No DinoML tests were run for this audit.

## 13. Performance probes

- tokenizer throughput separately from model runtime, because Jieba tokenization may be CPU-bound.
- encoder throughput over `(B,S)` sweep: `B in {1,8,32}`, `S in {128,512,1536}`.
- attention backend comparison: eager matmul/softmax, fused RoPE+attention, and any SDPA/FlashAttention lowering.
- vocab projection cost for `V=12000` versus `V=50000`.
- QKV packed versus separate projection launches.
- LayerNorm/residual fusion impact per block.
- memory footprint of attention scores `[B,heads,S,S]`, especially at `S=1536`.
- optional decode tokens/sec and KV memory for synthetic `is_decoder=True`.

## 14. Skip/defer list

- training losses and gradient checkpointing
- dropout behavior beyond inference no-op
- `output_attentions=True` optimized path
- arbitrary 3D attention masks in fused attention backend
- `roformer_v2` RMSNorm/no-bias behavior until separately source-audited
- cross-attention and encoder-decoder cache unless a target checkpoint requires it
- quantized/packed weights and tensor parallelism
- NHWC/NCHW layout work; this is text-only rank-2/rank-3 sequence data

## 15. Final implementation checklist

- [ ] Parse `RoFormerConfig` and checkpoint config defaults.
- [ ] Reject non-divisible `hidden_size / num_attention_heads`.
- [ ] Reject or route ignored RoFormer-v2 fields (`norm_type`, `use_bias`) for this source basis.
- [ ] Load tied embeddings/MLM decoder as one logical parameter alias.
- [ ] Implement embedding lookup, token type add, embedding LayerNorm.
- [ ] Implement optional `embedding_size -> hidden_size` projection.
- [ ] Implement source RoPE table and apply math.
- [ ] Implement post-norm MHA block with additive masks.
- [ ] Implement FFN `Linear -> ACT2FN -> Linear` and residual LayerNorm.
- [ ] Implement MLM transform and vocab projection.
- [ ] Add RoPE table/apply parity tests.
- [ ] Add one-block and full-small-checkpoint masked-LM parity tests.
- [ ] Add packed QKV rewrite with weight/bias concat tests.
- [ ] Add guarded fused attention rewrite tests.
- [ ] Benchmark encoder sequence-length sweep and vocab projection sweep.
