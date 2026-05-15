# GPTBigCode Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: gpt_bigcode family; primary target GPTBigCodeForCausalLM generation.
Config source: HF config.json snapshots under agents/plans/transformers/gpt_bigcode/_sources/.
Source files inspected:
  transformers/src/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py
  transformers/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
  transformers/src/transformers/models/gpt_bigcode/__init__.py
  transformers/src/transformers/cache_utils.py
  transformers/src/transformers/masking_utils.py
  transformers/src/transformers/integrations/sdpa_attention.py
  transformers/src/transformers/integrations/flash_attention.py
  transformers/src/transformers/modeling_utils.py
  transformers/tests/models/gpt_bigcode/test_modeling_gpt_bigcode.py
Any missing files or assumptions:
  No local tokenization_gpt_bigcode.py exists; AutoTokenizer maps model_type=gpt_bigcode to GPT2Tokenizer/GPT2TokenizerFast.
  Official bigcode/starcoder, bigcode/starcoderbase, and bigcode/starcoderplus configs are gated from this environment.
  Large StarCoder-style dimensions below use open derivative/mirror configs and are labeled as such.
  This report scopes native in-library GPTBigCode source only; quantized GPTQ files and custom serving runners are separate audits.
```

Source URLs for future review:

- `modeling_gpt_bigcode.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py
- `configuration_gpt_bigcode.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gpt_bigcode/configuration_gpt_bigcode.py
- `cache_utils.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/cache_utils.py
- `masking_utils.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/masking_utils.py
- `sdpa_attention.py`: https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/integrations/sdpa_attention.py
- Representative configs: `bigcode/tiny_starcoder_py`, `bigcode/gpt_bigcode-santacoder`, `bigcode/santacoder-fast-inference`, `defog/sqlcoder`, `HuggingFaceH4/starchat-alpha`, `ShipItMind/starcoder-gptq-4bit-128g`, `hf-internal-testing/tiny-random-GPTBigCodeForCausalLM`.

Primary runtime target: causal language generation via `GPTBigCodeForCausalLM`. Required body is `GPTBigCodeModel`; base hidden-state output is useful for parity. Sequence classification and token classification heads are optional/deferred for the generation target. Training losses, dropout behavior, and gradient checkpointing are deferred.

## 2. High-level architecture

GPTBigCode is a text-only decoder-only Transformer with learned token embeddings, learned absolute position embeddings, pre-attention LayerNorm, causal self-attention, MLP, final LayerNorm, and an untied module object for the LM head whose weight is tied to token embeddings by the pretrained-model weight tying contract.

```text
GPT-2 tokenizer/input_ids + optional 2D attention_mask
  -> token embedding + absolute position embedding (+ optional token_type embedding via wte)
  -> repeated decoder blocks
  -> final LayerNorm
  -> selected-token lm_head
  -> logits/sampling
```

Generation stage split:

```text
CPU tokenizer -> prefill(input_ids, position_ids, attention_mask, empty cache)
              -> per-layer MQA/MHA KV cache
              -> decode(new token(s), grown attention_mask, cache)
              -> logits_to_keep/lm_head -> sampler/controller
```

Independently stageable pieces:

- Tokenization and padding are CPU/data-pipeline work. GPTBigCode uses GPT-2 style tokenization metadata through AutoTokenizer.
- Learned embedding lookup and position-id construction are GPU graph work for faithful parity.
- Prefill and decode share the same block code; decode differs mainly in cache append length and mask shape.
- `logits_to_keep` allows last-token-only or selected-token logits and should be lowered before the full vocabulary projection when possible.

## 3. Important config dimensions

Source defaults from `GPTBigCodeConfig`: `vocab_size=50257`, `n_positions=1024`, `n_embd=768`, `n_layer=12`, `n_head=12`, `n_inner=None`, `activation_function="gelu_pytorch_tanh"`, dropout rates `0.1`, `layer_norm_epsilon=1e-5`, `scale_attn_weights=True`, `use_cache=True`, `attention_softmax_in_fp32=True`, `scale_attention_softmax_in_fp32=True`, `multi_query=True`, `add_cross_attention=False`, `tie_word_embeddings=True`. `__post_init__` derives `num_key_value_heads = 1 if multi_query else n_head`.

| Field | Meaning for lowering |
| --- | --- |
| `n_embd` / `hidden_size` | Model width `H`; source requires `H % n_head == 0`. |
| `n_layer` / `num_hidden_layers` | Decoder block count. |
| `n_head` / `num_attention_heads` | Query head count. |
| `multi_query` | If true, K/V head count is 1; if false, K/V head count equals query heads. |
| `head_dim` | `H // n_head`. |
| `n_inner` | MLP hidden width; source uses `4 * H` when omitted. |
| `n_positions` / `max_position_embeddings` | Learned position embedding rows and causal-bias buffer size. |
| `activation_function` | `ACT2FN` lookup; common values are `gelu_pytorch_tanh`, `gelu`, and debug `relu`. |
| `attention_softmax_in_fp32` | Stored on attention module, but current `eager_attention_forward` always calls softmax with dtype fp32. |
| `scale_attention_softmax_in_fp32` | Stored on attention module; not consumed by current attention dispatch path. |
| `scale_attn_weights` | Controls attention scale `head_dim ** -0.5` versus `1.0`. |
| `_attn_implementation` | Runtime dispatch key: eager fallback, `sdpa`, flash-attention variants, flex, and paged forms via shared interfaces. |
| `use_cache` | Enables HF `DynamicCache` by default. Per-layer cache tensors are `[B, kv_heads, T, D]`. |
| `tie_word_embeddings` | LM head weight should alias `transformer.wte.weight` for normal causal LM loading. |

Representative checkpoint sweep:

| Model id | Source type | Layers | Hidden | Q heads | KV heads | Head dim | MLP | Positions | Vocab | Activation | dtype/config notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hf-internal-testing/tiny-random-GPTBigCodeForCausalLM` | HF test config | 5 | 32 | 4 | 1 | 8 | 37 | 512 | 1024 | `relu` | `attention_softmax_in_fp32=false`; pad token 1021; good shape/debug smoke. |
| `bigcode/gpt_bigcode-santacoder` | official open | 24 | 2048 | 16 | 1 | 128 | 8192 | 2048 | 49280 | `gelu_pytorch_tanh` | SantaCoder production-ish small model; bos/eos 49152. |
| `bigcode/santacoder-fast-inference` | official open historical | 24 | 2048 | 16 | 1* | 128 | 8192 | 2048 | 49280 | `gelu_pytorch_tanh` | Omits `multi_query`; native config default makes it MQA. `attention_type=2` and runner fields are ignored by native source. |
| `bigcode/tiny_starcoder_py` | official open | 20 | 768 | 12 | 1 | 64 | 3072 | 8192 | 49152 | `gelu_pytorch_tanh` | Tiny StarCoder-like long context; carries historical runner/cache flags ignored by native source. |
| `defog/sqlcoder` | open StarCoder-derived finetune | 40 | 6144 | 48 | 1 | 128 | 24576 | 8192 | 49152 | `gelu` | Large StarCoder-style dimensions; config says bf16. |
| `HuggingFaceH4/starchat-alpha` | open StarCoder-derived finetune | 40 | 6144 | 48 | 1 | 128 | 24576 | 8192 | 49156 | `gelu` | Added tokens change vocab; config says fp16. |
| `ShipItMind/starcoder-gptq-4bit-128g` | open quantized StarCoder mirror | 40 | 6144 | 48 | 1 | 128 | 24576 | 8192 | 49152 | `gelu` | GPTQ storage is out of native-source scope; dense config mirrors StarCoder dimensions. |

`*` means inferred from native source defaults, because the config omits the field.

## 3a. Family variation traps

- MQA is the default and is the StarCoder/SantaCoder path. `c_attn` output is `H + 2 * head_dim`, not `3H`, when `multi_query=True`.
- Non-MQA is supported by source and tests. In that path `c_attn` output is `3H`, with packed per-head `[q, k, v]` chunks after view to `[B, T, n_head, 3D]`.
- Cross-attention exists only when `add_cross_attention=True` and `multi_query=False`; MQA cross-attention raises `NotImplementedError`. For causal LM integration, reject or defer cross-attention.
- Native source has no RoPE, ALiBi, sliding-window attention, MoE, gated MLP, or tensor-parallel runtime path.
- Learned absolute position embeddings cap native position ids at `n_positions`; long-context support is from larger embedding tables, not RoPE scaling.
- Historical config fields such as `pad_key_length`, `pre_allocate_kv_cache`, `max_batch_size`, `max_sequence_length`, `validate_runner_input`, `inference_runner`, `runner_max_sequence_length`, `attention_type`, `reorder_and_upcast_attn`, and `scale_attn_by_inverse_layer_idx` are not read by the inspected native modeling source.
- `attention_softmax_in_fp32` and `scale_attention_softmax_in_fp32` are assigned on the attention module but the current eager attention function unconditionally computes softmax in fp32. Do not implement old fused-softmax branches as required native behavior without a separate older-version audit.
- The test suite explicitly skips generic past-key-value format tests because GPTBigCode MQA has a non-standard KV format relative to conventional GPT models: cache K/V heads can be 1 while Q heads are many.
- `token_type_ids`, if supplied, are embedded through the same token embedding table `wte` and added to hidden states. Many generation paths omit them, but parity tests cover the branch.
- `logits_to_keep` may be an int or tensor; full-vocab projection can be avoided for last-token decode only when the slice/indexing is compile-visible.
- Layout is text-major `[B, T, H]` and head-major `[B, heads, T, D]`; there is no NCHW/NHWC issue. Protect attention head reshapes/transposes from generic layout translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- `Embedding(vocab_size, H)` for `input_ids[int64] -> [B, T, H]`.
- `Embedding(n_positions, H)` for `position_ids[int64] -> [1 or B, T, H]`.
- Optional token-type embedding reuse through `wte(token_type_ids)`.
- Elementwise adds for token + position + token-type embeddings and residual paths.
- Shape ops: `view`, `reshape`, `unsqueeze`, `transpose(1,2)`, `contiguous`, `split`, `slice`, `expand`, `cat`.
- `arange` and scalar add for generated `position_ids`.
- `where`/mask construction or imported mask artifact, depending on attention backend.
- `index_select` for cache reorder/beam paths if generation supports beams later.

Neural network primitives:

- LayerNorm over last dim `H`, affine, epsilon usually `1e-5`.
- Attention input projection:
  - MQA: `Linear(H -> H + 2D)`, bias=True.
  - MHA: `Linear(H -> 3H)`, bias=True.
- Attention output projection: `Linear(H -> H)`, bias=True.
- MLP: `Linear(H -> I)`, activation, `Linear(I -> H)`, both bias=True.
- Activations: `gelu_pytorch_tanh`, exact/torch `gelu`, and debug `relu`.
- Final LM head: `Linear(H -> vocab_size)`, bias=False, tied to `wte.weight`.
- Dropout is present in source but should be zero-effect in eval/inference.

Attention primitives:

- Full causal self-attention with MQA or MHA.
- Q shape `[B, n_head, Q, D]`; native K/V pre-repeat shape `[B, kv_heads, K, D]`.
- MQA repeat/implicit GQA: repeat K/V from one KV head to all Q heads, or use backend `enable_gqa=True` when legal.
- Attention score matmul `Q @ K^T`, scale by `D^-0.5` when enabled, add mask, softmax in fp32, dropout in training only, matmul with V.
- SDPA backend with `scaled_dot_product_attention`, optional `enable_gqa=True` only when backend conditions allow.
- FlashAttention backend takes `[B, T, heads, D]` after transpose and supports padding mask as 2D bool mask.

Generation/cache ops:

- HF `DynamicCache` append per layer: `cat` on sequence axis `-2`.
- Per-layer cache K/V shape before repeat: `[B, kv_heads, T_cache, D]`; for StarCoder MQA, `[B, 1, T_cache, 128]`.
- Cache update returns full K/V for attention; repeat or backend GQA happens after update.
- Cache length drives generated `position_ids` and causal mask sizes.

Distributed/tensor-parallel ops:

- None in native source. Quantized/GPTQ repos and serving runner fields are storage/serving concerns, not current dense operator requirements.

## 5. Layer/block breakdown

Embedding and model entry:

```text
if inputs_embeds is None:
  x = wte(input_ids)                         # [B, T, H]
position_ids = arange(T) + past_seen_tokens  # [1, T], unless caller supplied
x = x + wpe(position_ids)
if token_type_ids is not None:
  x = x + wte(token_type_ids.view(-1, T))
x = dropout(x)                               # inactive in eval
```

Decoder block, repeated `n_layer` times:

```text
residual = x
a = LayerNorm(x)

# MQA path
q_flat, k, v = split(Linear(a), [H, D, D], dim=-1)
q = view(q_flat, [B, T, n_head, D]).transpose(1, 2)
k = k.unsqueeze(1)                           # [B, 1, T, D]
v = v.unsqueeze(1)                           # [B, 1, T, D]

# MHA path
q, k, v = Linear(a).view(B, T, n_head, 3D).transpose(1, 2).split([D, D, D], dim=-1)

k, v = cache.update(k, v, layer_idx) if cache else (k, v)
attn = Attention(q, k, v, causal_mask)
attn = Linear(attn.reshape(B, T, H))          # c_proj, bias=True
x = residual + attn

residual = x
m = LayerNorm(x)
m = Linear(H -> I)(m)
m = ACT2FN[activation_function](m)
m = Linear(I -> H)(m)
x = residual + m
```

Final:

```text
x = LayerNorm(x)
logits = lm_head(x[:, selected_positions, :]) # [B, kept_T, vocab]
```

Concrete StarCoder-style large block (`H=6144`, `n_head=48`, `D=128`, `I=24576`, MQA):

- `c_attn`: `Linear(6144 -> 6400)`, split `[6144, 128, 128]`.
- `c_proj`: `Linear(6144 -> 6144)`.
- MLP: `Linear(6144 -> 24576)`, GELU, `Linear(24576 -> 6144)`.
- Cache per layer per K or V at sequence length `T`: `B * 1 * T * 128` elements, not `B * 48 * T * 128`.

## 6. Attention requirements

Required for causal LM:

- Causal self-attention only.
- MQA by default and for StarCoder/SantaCoder: `num_key_value_heads=1`, `num_key_value_groups=n_head`.
- MHA variant required if DinoML wants full family coverage because `multi_query=False` is source-supported and tested.
- Head dim is source-derived as `H // n_head`; source rejects non-divisible configs.
- No RoPE/ALiBi/relative bias/sliding window in this family.

Masking:

- `create_causal_mask` produces backend-specific masks based on `config._attn_implementation`.
- Eager masks are additive float masks shaped `[B, 1, Q, K]`, with `0` for valid positions and `torch.finfo(dtype).min` for masked positions.
- SDPA masks can be `None` when causal skip is legal, or bool/additive masks otherwise.
- FlashAttention masks are `None` for unpadded full-causal cases or 2D bool attention masks sliced to current KV length when padding exists.
- A caller-supplied 4D mask is accepted as already prepared and returned as-is.
- Packed sequence detection can occur from non-monotonic `position_ids` when no attention mask/cache is supplied; generation normally does not use that path.

Cache layout:

- New K/V are produced before any repeat and before attention.
- Cached K/V are stored in the same pre-repeat layout: `[B, kv_heads, T, D]`.
- For MQA StarCoder-style models, per-layer cache is `[B, 1, T, 128]` for K and V.
- For standard MHA, per-layer cache is `[B, n_head, T, D]`.
- Attention backends either repeat K/V to `[B, n_head, T, D]` or use backend GQA support. DinoML should store the compact cache and expand logically inside attention, not materialize repeated cache as the canonical ABI.
- Cached keys are learned-position independent; there is no RoPE application before cache write.

Backend dispatch:

- Native model calls `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`.
- Eager path explicitly does `repeat_kv`, `matmul`, mask add, `softmax(..., dtype=torch.float32).to(query.dtype)`, dropout, and value matmul.
- SDPA path uses PyTorch `scaled_dot_product_attention`; if GQA is not legal for that backend/mask combination, it repeats K/V first.
- FlashAttention path transposes Q/K/V to `[B, T, heads, D]`, casts fp32 query to compatible target dtype when needed, and delegates to `_flash_attention_forward`.
- `output_attentions=True` is unsupported by SDPA/FlashAttention and only meaningful for eager-style attention weights; first DinoML integration can omit attention tensor outputs.

## 7. Position encoding and custom math

GPTBigCode uses learned absolute positions only:

```python
def make_position_ids(inputs_embeds, past_key_values):
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    return torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0) + past_seen
```

No RoPE, M-RoPE, ALiBi, or convolutional position encoding is required. `wpe(position_ids)` is a normal embedding lookup. Position IDs can be caller-supplied; if supplied, they also participate in packed-sequence mask detection in the generic mask utility for non-cache, no-padding-mask runs.

Attention repeat custom math:

```python
def repeat_kv_for_mqa(kv, n_rep):
    # kv: [B, kv_heads, T, D]
    if n_rep == 1:
        return kv
    b, kv_heads, t, d = kv.shape
    return kv[:, :, None, :, :].expand(b, kv_heads, n_rep, t, d).reshape(b, kv_heads * n_rep, t, d)
```

What can be precomputed:

- Learned token, position, and LM weights are constants.
- Causal lower-triangular structure can be generated by shape, but padding and cache lengths are dynamic.
- No sin/cos tables are needed.

## 8. Preprocessing and input packing

CPU/data-pipeline:

- Tokenization is GPT-2 style via AutoTokenizer mapping. Configs use varying vocab sizes around 49k for code models.
- Generation code often sets left padding externally for batched generation; the model only consumes `input_ids` and `attention_mask`.
- No processor, image, audio, multimodal stitch, codebook, or `cu_seqlens` metadata is model-coupled in native GPTBigCode.

GPU/runtime inputs:

- `input_ids`: `[B, T]`, integer token IDs.
- Optional `attention_mask`: usually `[B, past_T + T]`, 1/true for valid tokens and 0/false for padding before mask utility conversion.
- Optional `position_ids`: `[1 or B, T]`; otherwise generated from cache length.
- Optional `token_type_ids`: `[B, T]`; source uses `wte` again and adds those embeddings.
- Optional `inputs_embeds`: `[B, T, H]`; mutually exclusive with `input_ids`.
- `past_key_values`: HF cache object; DinoML should expose an equivalent per-layer compact K/V cache ABI.

End-to-end generation controller behavior:

- `GenerationMixin` owns sampling/beam search and stopping. Core module parity needs logits and cache, not tokenizer-side sampling.
- `logits_to_keep=0` means all logits; positive int means suffix slice; tensor means explicit selection. First optimized integration should support `logits_to_keep=1` for decode and fall back for tensor selection.

## 9. Graph rewrite / lowering opportunities

### Rewrite: MQA fused projection split

Source pattern:

```text
Linear(H -> H + 2D) -> unsqueeze(1) -> split([H, D, D], dim=3)
```

Replacement:

```text
single GEMM with packed output -> views/slices q_flat, k, v
```

Preconditions:

- `multi_query=True`.
- `H % n_head == 0`, `D = H / n_head`, output width `H + 2D`.
- Weight layout is PyTorch `nn.Linear`: stored `[out_features, in_features]`, bias `[out_features]`.
- Split order is all-Q rows, then single K rows, then single V rows.

Failure cases:

- `multi_query=False` uses a different packed layout.
- Cross-attention MQA is not implemented by source and should be rejected.

Parity test sketch: compare q/k/v tensors and post-attention logits for SantaCoder and StarCoder-style configs against Transformers for prefill and one-token decode.

### Rewrite: MHA packed QKV projection

Source pattern:

```text
Linear(H -> 3H) -> view(B,T,n_head,3D) -> transpose(1,2) -> split([D,D,D], dim=-1)
```

Replacement:

```text
packed QKV GEMM -> head-aware view -> Q/K/V logical tensors
```

Preconditions:

- `multi_query=False`.
- Packed storage after view is per head as `[q_head, k_head, v_head]`, not all Q rows then all K rows then all V rows.
- Weight transform is required if a backend expects all-Q/all-K/all-V row blocks.

Failure cases:

- Treating MHA and MQA split orders the same silently corrupts attention.

### Rewrite: compact MQA cache + backend GQA attention

Source pattern:

```text
cache stores K/V [B,1,T,D] -> repeat_kv to [B,H,T,D] -> attention
```

Replacement:

```text
store compact K/V -> call attention kernel with q_heads=H, kv_heads=1
```

Preconditions:

- Attention kernel supports MQA/GQA or can repeat K/V inside the kernel without materializing canonical cache.
- Mask semantics match backend path: causal/padding behavior, decode `Q=1` handling, scale, fp32 softmax accumulation.

Failure cases:

- Kernels that require equal Q/KV heads need explicit repeat fallback.
- Repeating into cache instead of per-call attention bloats memory and changes cache ABI.

### Rewrite: last-token logits

Source pattern:

```text
hidden_states[:, -1:, :] -> lm_head
```

Replacement:

```text
slice hidden before vocab GEMM -> GEMM([B,1,H] x [vocab,H]^T)
```

Preconditions:

- `logits_to_keep` is compile/runtime-known as `1` or suffix int.
- Loss is not computed.
- LM head weight aliasing to `wte.weight` is preserved.

Failure cases:

- Tensor `logits_to_keep` or full prefill logits require gather/full projection.

### Rewrite: eval dropout elimination

Source pattern:

```text
Dropout(p=config.*_pdrop) in embeddings, attention output, MLP, attention weights
```

Replacement: identity in inference/eval artifacts.

Preconditions: model is compiled for inference only, with training disabled.

### Layout guard: text/head attention region

No channel-last rewrite applies. Guard these patterns against generic layout translation:

- `[B,T,H] -> view(B,T,n_head,D) -> transpose(1,2)` axis meanings.
- Cache tensors `[B,kv_heads,T,D]`.
- Attention mask `[B,1,Q,K]` or backend-specific 2D bool masks.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `H`, especially `H=2048` and `H=6144`.
- MQA projection split + attention prefill/decode: avoid materializing repeated K/V.
- Flash/SDPA-style causal attention with compact KV cache and fp32 softmax accumulation.
- MLP GEMM + GELU + GEMM; `gelu` and `gelu_pytorch_tanh` both appear in configs.
- Last-token-only LM head projection for decode.

Medium priority:

- Bias+residual epilogues for `c_proj` and MLP down projection.
- Embedding + position add fusion.
- Cache append/update kernels for compact `[B,1,T,D]` MQA cache, including batched decode.
- MHA packed QKV weight transform support for `multi_query=False`.

Lower priority:

- Token-type embedding add.
- Attention output tensor materialization for `output_attentions=True`.
- Sequence/token classification heads.
- Beam-search cache reorder.
- Cross-attention path for non-MQA configs.

## 11. Runtime staging plan

Stage 1: parse config and instantiate dense weight manifest.

- Support aliases from `GPTBigCodeConfig`: `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `max_position_embeddings`.
- Reject `add_cross_attention=True` and non-native quantized storage for first pass.

Stage 2: one-block dense parity.

- Implement embeddings, LayerNorm, MQA projection split, eager attention math, MLP, and final LayerNorm.
- Validate with tiny random config and SantaCoder-sized random tensors.

Stage 3: prefill causal LM parity.

- Run full prefill without cache reuse first, using additive causal/padding mask.
- Add tied LM head and `logits_to_keep` suffix slicing.

Stage 4: decode with compact MQA KV cache.

- Cache K/V as `[B,1,T,D]` for MQA.
- Validate one-token and multi-token continuation against no-past execution.

Stage 5: optimized attention.

- Route compact MQA to FlashAttention/SDPA-equivalent backend when masks permit.
- Keep eager composed fallback for masks/backend gaps.

Stage 6: variant expansion.

- Add `multi_query=False` MHA packed QKV path.
- Optional classification heads and beam cache reorder.

Stage 7: production scheduling.

- Continuous batching and paged cache can build on compact MQA cache ABI.

## 12. Parity and validation plan

Custom op/unit tests:

- `repeat_kv_for_mqa`: compare compact repeat against PyTorch `repeat_interleave`.
- MQA `c_attn` split: compare q/k/v shapes and values for `H=2048,n_head=16` and `H=6144,n_head=48`.
- MHA packed split: compare per-head q/k/v split for `multi_query=False`.
- Mask conversion: 2D padding mask plus causal mask for prefill and decode.

Model parity:

- Single decoder block fp32 parity with dropout disabled: tolerance `rtol=1e-4, atol=1e-4`.
- Full tiny random causal LM prefill parity: fp32 `1e-4`, fp16/bf16 logits `1e-2` after backend choice.
- SantaCoder prefill selected logits on a short prompt.
- Decode parity: first prefill then one-token and three-token decode with cache, compare to no-past run as upstream tests do.
- Batched left-padding generation mask parity.
- StarCoder-style large random-weight shape smoke for `H=6144`, `n_layer` reduced to 1 for local block validation.

End-to-end:

- Tokenizer + `bigcode/tiny_starcoder_py` newline regression prompt can be used as a smoke once weights are supported.
- SantaCoder generation prompt from upstream slow tests can be used as a heavier acceptance test.

## 13. Performance probes

- Prefill throughput by sequence length: 512, 2048, 8192 for MQA models.
- Decode tokens/sec for compact MQA cache at batch sizes 1, 8, 32.
- KV cache memory usage: compare compact `[B,1,T,D]` versus repeated `[B,H,T,D]`.
- Attention backend comparison: eager composed, SDPA/GQA, FlashAttention compact KV.
- MLP GEMM throughput for `H=2048/I=8192` and `H=6144/I=24576`.
- LM head cost with full logits versus `logits_to_keep=1`.
- Mask construction overhead for padded batched generation.
- Cache append/update overhead and allocator behavior for long context.

Any benchmark results should be labeled separately; this audit includes source/config-derived probes only.

## 14. Skip/defer list

Safe to defer for first generation integration:

- Training losses and gradient checkpointing.
- Dropout randomness.
- `output_attentions=True` for optimized backends.
- Sequence classification and token classification heads.
- Cross-attention and encoder-decoder cache, especially because MQA cross-attention is not implemented.
- Beam search and cache reorder.
- Paged attention wrappers and external serving runner fields.
- GPTQ/4-bit storage, `santacoder-fast-inference` runner metadata, and custom remote serving paths.
- Tensor `logits_to_keep` gather form; support suffix int first.
- Packed sequence mask detection from custom `position_ids` outside normal generation.

## 15. Final implementation checklist

- [ ] Parse `GPTBigCodeConfig` aliases and derive `head_dim`, `kv_heads`, `num_key_value_groups`.
- [ ] Load/tie `transformer.wte.weight` and `lm_head.weight` as one logical parameter when `tie_word_embeddings=True`.
- [ ] Implement token, position, and optional token-type embeddings.
- [ ] Implement LayerNorm, MQA `c_attn` split, attention output projection, MLP, and final LayerNorm.
- [ ] Implement compact MQA KV cache ABI `[B,1,T,D]` and MHA cache ABI `[B,H,T,D]`.
- [ ] Implement causal/padding mask parity for prefill and decode.
- [ ] Add eager attention fallback with fp32 softmax accumulation.
- [ ] Add optimized compact-MQA attention backend guard.
- [ ] Add `logits_to_keep=1` last-token LM head rewrite.
- [ ] Add MHA packed-QKV weight-layout tests for `multi_query=False`.
- [ ] Add one-block, prefill, and decode parity tests against Transformers.
- [ ] Benchmark prefill, decode, cache memory, and LM-head selected-token projection.
