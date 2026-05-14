# GLM4 Transformers audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model family: `glm4`.

Primary runtime target: `Glm4ForCausalLM` text-only autoregressive generation, including prefill and decode with self-attention KV cache.

Model ids / config sources inspected:

HF config snapshots were fetched from the listed `main` raw URLs on 2026-05-13 unless a row says otherwise.

| Model id | Config source | Local snapshot | Notes |
| --- | --- | --- | --- |
| `yujiepan/glm-4-tiny-random` | `https://huggingface.co/yujiepan/glm-4-tiny-random/raw/main/config.json` | `_sources/yujiepan_glm-4-tiny-random_config.json` | open debug checkpoint, not official; useful for tiny shape tests |
| `zai-org/GLM-4-9B-0414` | `https://huggingface.co/zai-org/GLM-4-9B-0414/raw/main/config.json` | `_sources/zai-org_GLM-4-9B-0414_config.json` | official production 9B BF16 |
| `zai-org/GLM-Z1-9B-0414` | `https://huggingface.co/zai-org/GLM-Z1-9B-0414/raw/main/config.json` | `_sources/zai-org_GLM-Z1-9B-0414_config.json` | official reasoning 9B, same operator shape as GLM-4-9B |
| `zai-org/GLM-4-32B-0414` | `https://huggingface.co/zai-org/GLM-4-32B-0414/raw/main/config.json` | `_sources/zai-org_GLM-4-32B-0414_config.json` | official production 32B BF16 |
| `zai-org/GLM-4-32B-Base-0414` | `https://huggingface.co/zai-org/GLM-4-32B-Base-0414/raw/main/config.json` | `_sources/zai-org_GLM-4-32B-Base-0414_config.json` | official base 32B, same operator shape as 32B chat |
| `zai-org/GLM-Z1-Rumination-32B-0414` | `https://huggingface.co/zai-org/GLM-Z1-Rumination-32B-0414/raw/main/config.json` | `_sources/zai-org_GLM-Z1-Rumination-32B-0414_config.json` | official long-context reasoning variant; changes KV heads and max context |
| `zai-org/GLM-4-9B-0414` generation/tokenizer configs | HF raw `generation_config.json`, `tokenizer_config.json` | `_sources/zai-org_GLM-4-9B-0414_generation_config.json`, `_sources/zai-org_GLM-4-9B-0414_tokenizer_config.json` | small snapshots for EOS/pad/special token IDs |

Source files inspected:

- `X:/H/transformers/src/transformers/models/glm4/modular_glm4.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4/modular_glm4.py`)
- `X:/H/transformers/src/transformers/models/glm4/modeling_glm4.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4/modeling_glm4.py`)
- `X:/H/transformers/src/transformers/models/glm4/configuration_glm4.py` (`https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm4/configuration_glm4.py`)
- `X:/H/transformers/src/transformers/models/glm4/convert_glm4_weights_to_hf.py`
- Comparison files: `models/glm/modeling_glm.py`, `models/glm/configuration_glm.py`, `models/glm4_moe/modeling_glm4_moe.py`, `models/glm4_moe/configuration_glm4_moe.py`
- Shared utilities: `masking_utils.py`, `cache_utils.py`, `modeling_rope_utils.py`, `modeling_utils.py`

Authoritative source note: `modeling_glm4.py` is generated from `modular_glm4.py`. Future Transformers source edits should target the modular file, but DinoML should audit the generated file because that is what users import.

Missing files or assumptions: no remote-code files are required for the inspected in-library GLM4 path. This report is scoped to text-only `model_type="glm4"` causal LM. GLM4-MoE, GLM4-MoE-Lite, GLM4V, GLM-OCR wrappers, and legacy `chatglm` configs are separate targets unless they route exactly into this `Glm4ForCausalLM` class.

## 2. High-level architecture

GLM4 is a decoder-only causal LM:

```text
token ids / embeddings -> repeated GLM4 decoder blocks -> final RMSNorm -> LM head -> logits/sampling
```

Each decoder block is pre-norm with extra post-sublayer RMSNorms:

```text
RMSNorm -> GQA self-attention with partial RoPE -> RMSNorm -> residual
RMSNorm -> packed SwiGLU MLP -> RMSNorm -> residual
```

Stage decomposition:

| Stage | Runtime contract | Independently stageable? |
| --- | --- | --- |
| CPU/data pipeline | tokenizer/chat template, special tokens, 2D attention mask, EOS/pad IDs | yes |
| Embedding | `input_ids [B,S] -> hidden [B,S,H]`, or caller-provided `inputs_embeds` | yes |
| Prefill | full causal GQA attention, writes per-layer KV cache | yes |
| Decode | short query length, consumes/appends per-layer KV cache | yes |
| Logits | `lm_head(hidden[:, logits_to_keep, :])`; last-token-only is a first optimization | yes |

Implemented heads: `Glm4Model`, `Glm4ForCausalLM`, `Glm4ForSequenceClassification`, and `Glm4ForTokenClassification`. For causal LM integration, only the base model plus LM head is required. Classification heads can be deferred.

## 3. Important config dimensions

Config-class defaults are GLM4-like but do not match all production checkpoints. DinoML should parse checkpoint config and not infer projection widths from `hidden_size` alone.

| Field | Config class default | 9B configs | 32B configs | Rumination 32B | Runtime significance |
| --- | ---: | ---: | ---: | ---: | --- |
| `vocab_size` | 151552 | 151552 | 151552 | 151552 | embedding and LM head rows |
| `hidden_size` | 4096 | 4096 | 6144 | 6144 | residual width |
| `num_hidden_layers` | 40 | 40 | 61 | 61 | decoder block count |
| `num_attention_heads` | 32 | 32 | 48 | 48 | query head count |
| `num_key_value_heads` | 2 | 2 | 2 | 8 | GQA KV head count |
| `head_dim` | 128 | 128 | 128 | 128 | explicit per-head width |
| Q projection width | 4096 | 4096 | 6144 | 6144 | `num_attention_heads * head_dim` |
| KV projection width | 256 | 256 | 256 | 1024 | `num_key_value_heads * head_dim` |
| `intermediate_size` | 13696 | 13696 | 23040 | 23040 | SwiGLU half width |
| `max_position_embeddings` | 131072 | 32768 | 32768 | 131072 | context/cos-sin cache sizing |
| `partial_rotary_factor` | 0.5 via `__post_init__` | 0.5 | 0.5 | 0.5 | rotary width is 64 for `head_dim=128` |
| `rope_theta` | standardized default if omitted | 10000.0 | 10000.0 | 10000.0 | RoPE base |
| `attention_bias` | true | true | false | false | Q/K/V bias optional; output and MLP biases absent |
| `rms_norm_eps` | 1.5625e-7 | 1e-5 | 1e-5 | 1e-5 | checkpoint overrides class default |
| `hidden_act` | `silu` | `silu` | `silu` | `silu` | SwiGLU activation |
| `torch_dtype` | not class default | `bfloat16` | `bfloat16` | `bfloat16` | checkpoint metadata |
| `use_cache` | true | true | true | true | default generation cache |
| `tie_word_embeddings` | false | false | false | false | do not alias LM head to embeddings |

Representative checkpoint sweep:

| Model id | Layers | Hidden | Q heads / KV heads / head dim | GQA groups | MLP width | Max pos | QKV bias |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `yujiepan/glm-4-tiny-random` | 2 | 16 | 1 / 1 / 16 | 1 | 32 | 32768 | false |
| `zai-org/GLM-4-9B-0414` | 40 | 4096 | 32 / 2 / 128 | 16 | 13696 | 32768 | true |
| `zai-org/GLM-Z1-9B-0414` | 40 | 4096 | 32 / 2 / 128 | 16 | 13696 | 32768 | true |
| `zai-org/GLM-4-32B-0414` | 61 | 6144 | 48 / 2 / 128 | 24 | 23040 | 32768 | false |
| `zai-org/GLM-4-32B-Base-0414` | 61 | 6144 | 48 / 2 / 128 | 24 | 23040 | 32768 | false |
| `zai-org/GLM-Z1-Rumination-32B-0414` | 61 | 6144 | 48 / 8 / 128 | 6 | 23040 | 131072 | false |

## 3a. Family variation traps

- `attention_bias` varies by checkpoint. 9B has Q/K/V bias; 32B and Rumination do not. `o_proj`, `gate_up_proj`, `down_proj`, and `lm_head` are biasless in source.
- `num_key_value_heads` varies from 2 to 8. Do not bake in a 16x or 24x repeat factor.
- `max_position_embeddings` is 32768 for most official configs but 131072 in Rumination and in the config class default.
- `rms_norm_eps` class default is `1.5625e-7`, while inspected checkpoints use `1e-5`.
- `partial_rotary_factor=0.5` means only the first half of each 128-d head is rotary. The remaining 64 channels pass through unchanged.
- GLM4 uses separate `q_proj`, `k_proj`, `v_proj` modules. The conversion script splits legacy packed `query_key_value` weights in `[Q, K, V]` row order, but native HF checkpoints store separate tensors.
- Native GLM4 has no Q/K norm. GLM4-MoE can enable per-head Q/K RMSNorm with `use_qk_norm`; do not import that requirement into GLM4.
- GLM4's RoPE rotation is interleaved even/odd (`x[..., 0::2]`, `x[..., 1::2]`). GLM4-MoE uses half-split rotation. This is a parity trap for shared RoPE kernels.
- GLM4 differs from older `glm` mainly by adding `post_self_attn_layernorm` and `post_mlp_layernorm` inside each block.
- Generic `create_causal_mask` can return `None`, a dense additive mask, or a backend-specific `BlockMask` depending on attention implementation and inputs. A DinoML import path should normalize or reject unsupported mask forms explicitly.
- No NCHW/NHWC layout translation is relevant. Guard layout passes around all `[B,S,H] <-> [B,heads,S,D]` reshapes/transposes.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `[B,S] -> [B,S,H]`.
- Reshape/view after projections: `[B,S,out] -> [B,S,heads,D]`.
- Transpose between `[B,S,heads,D]` and `[B,heads,S,D]`.
- `contiguous`/layout normalization after attention transpose.
- Slice for `logits_to_keep`: `hidden[:, slice_indices, :]`.
- Concatenation for partial RoPE pass-through: rotary channels plus unrotated channels.
- Optional Q/K/V bias add.

Neural network primitives:

- RMSNorm over last dimension, fp32 variance, output cast back to input dtype.
- Linear Q: `H -> num_attention_heads * head_dim`.
- Linear K/V: `H -> num_key_value_heads * head_dim`.
- Linear O: `num_attention_heads * head_dim -> H`, no bias.
- Packed SwiGLU input projection: `H -> 2 * intermediate_size`, no bias; split order is `[gate, up]`.
- `silu(gate) * up`.
- MLP down projection: `intermediate_size -> H`, no bias.
- LM head: `H -> vocab_size`, no bias.
- Residual adds after post-attention RMSNorm and post-MLP RMSNorm.

Attention primitives:

- Causal self-attention only.
- GQA repeat/expand of K/V heads before eager attention, or native GQA support in fused attention.
- Additive causal/padding mask handling for eager attention.
- Softmax in fp32, cast attention probabilities back to query dtype.
- Attention dropout is source-present but `0.0` in inference and inspected configs.

Position/rotary ops:

- Default RoPE frequency generation with `rope_theta`.
- Dynamic RoPE update wrapper exists for advanced RoPE types; inspected configs use default RoPE.
- Interleaved partial RoPE over first `int(head_dim * partial_rotary_factor)` channels.

Generation/cache ops:

- Per-layer KV cache update after RoPE. Cached keys are already RoPE-applied; values are projected values.
- Dynamic cache tensor shape per layer: K and V each `[B, num_key_value_heads, cache_seq, head_dim]`.
- Cache length drives default `position_ids` offset and mask lengths.

Preprocessing-coupled ops:

- Tokenizer produces `input_ids` and optional `attention_mask`.
- Generation config uses EOS token IDs `[151329, 151336, 151338]` and pad token ID `151329`.
- Tokenizer config defines GLM special tokens such as `[gMASK]` `151331`, `<sop>` `151333`, role tokens, and `<|endoftext|>` `151329`; these affect prompt construction but not the GPU graph once `input_ids` are available.

Distributed/tensor-parallel notes:

- Config declares TP plans: Q/K/V columnwise, O rowwise, MLP `gate_up_proj` columnwise gather because of `chunk`, MLP down rowwise split. DinoML can ignore this for single-GPU first integration but should preserve weight split order for later tensor parallel lowering.

## 5. Layer/block breakdown

For `B` batch, `S` query length, hidden width `H`, query heads `QH`, KV heads `KVH`, head dim `D`, intermediate `I`:

```text
Embedding:
  hidden = embed_tokens(input_ids)                         # [B,S,H]
  position_ids = arange(S) + past_seen_tokens              # [1,S] unless provided
  cos, sin = rotary_emb(hidden, position_ids)              # [B or 1,S,rotary_dim]

Decoder block, repeated N times:
  residual = hidden
  x = RMSNorm(hidden)                                      # [B,S,H]
  q = Linear_q(x).view(B,S,QH,D).transpose(1,2)            # [B,QH,S,D]
  k = Linear_k(x).view(B,S,KVH,D).transpose(1,2)           # [B,KVH,S,D]
  v = Linear_v(x).view(B,S,KVH,D).transpose(1,2)           # [B,KVH,S,D]
  q, k = partial_interleaved_rope(q, k, cos, sin)
  k, v = cache.update(k, v, layer_idx) if cache is present
  a = causal_attention(q, k, v, mask, scale=D**-0.5)
  a = a.transpose(1,2).reshape(B,S,QH*D).contiguous()
  a = Linear_o(a)                                         # [B,S,H]
  hidden = residual + RMSNorm(a)

  residual = hidden
  m = RMSNorm(hidden)
  gate_up = Linear_gate_up(m)                             # [B,S,2I]
  gate, up = chunk(gate_up, 2, dim=-1)
  m = silu(gate) * up
  m = Linear_down(m)                                      # [B,S,H]
  hidden = residual + RMSNorm(m)

Final:
  hidden = RMSNorm(hidden)
  logits = lm_head(hidden[:, logits_to_keep, :])          # [B,K,vocab]
```

For 9B: Q is `4096 -> 4096`, K/V are `4096 -> 256`, O is `4096 -> 4096`, MLP is `4096 -> 27392 -> 13696 -> 4096`.

For 32B with 2 KV heads: Q is `6144 -> 6144`, K/V are `6144 -> 256`, O is `6144 -> 6144`, MLP is `6144 -> 46080 -> 23040 -> 6144`.

For Rumination 32B with 8 KV heads: K/V become `6144 -> 1024`; Q/O/MLP remain the same as other 32B configs.

## 6. Attention requirements

Attention type:

- Causal decoder self-attention.
- GQA for production configs: `num_key_value_heads < num_attention_heads`.
- MHA for the tiny debug checkpoint: `num_key_value_heads == num_attention_heads == 1`.
- No cross-attention, sliding-window attention, ALiBi, relative bias, or local/block-sparse attention in inspected GLM4 source/configs.

Masking:

- `Glm4Model.forward` calls `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids)`.
- A user 4D mask is returned as-is by shared utility.
- A 2D `attention_mask` is moved to device and converted to bool before backend mask creation.
- If `attention_mask is None`, `position_ids` are nonmonotonic, and there is no cache, shared utilities detect packed sequences and AND a packed-sequence mask with the causal mask.
- Backend-dependent mask paths may skip materializing a mask for SDPA/Flash-style causal attention when safe.

Cache:

- Source applies RoPE before `past_key_values.update`, so cached keys are post-RoPE.
- Values are cached after V projection/reshape/transpose.
- Per-layer cache before repeat expansion: K and V each `[B, KVH, T, D]`.
- Attention compute uses repeated K/V as `[B, QH, T, D]` in eager fallback. A fused GQA kernel should avoid physical repeat.
- `DynamicCache(config)` initializes full-attention layers because GLM4 configs do not define `sliding_window`, `attention_chunk_size`, or `layer_types`.

Attention math order in eager fallback:

```text
K,V repeat -> Q @ K^T -> multiply by D**-0.5 -> add mask -> fp32 softmax -> cast -> dropout -> probs @ V
```

Transformers backend dispatch: `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`. Source advertises FlashAttention, SDPA, FlexAttention, and generic attention-backend support. DinoML should treat eager as the parity reference and fused GQA attention as the production path.

## 7. Position encoding and custom math

GLM4 uses default RoPE with partial rotary dimension. The checkpoint configs store top-level `rope_theta` and `partial_rotary_factor`; the config mixin standardizes these into `config.rope_parameters`.

Effective source math:

```python
rotary_dim = int(head_dim * partial_rotary_factor)
inv_freq = 1.0 / (rope_theta ** (arange(0, rotary_dim, 2) / rotary_dim))
freqs = inv_freq[None, :, None] @ position_ids[:, None, :]
emb = cat([freqs, freqs], dim=-1)
cos = cos(emb).to(x.dtype)
sin = sin(emb).to(x.dtype)
```

GLM4's application function is not the usual half-split LLaMA style; it interleaves even and odd channels:

```python
def glm4_partial_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_half = stack([-q_rot[..., 1::2], q_rot[..., 0::2]], dim=-1).flatten(-2)
    k_half = stack([-k_rot[..., 1::2], k_rot[..., 0::2]], dim=-1).flatten(-2)
    return cat([q_rot * cos + q_half * sin, q_pass], -1), cat([k_rot * cos + k_half * sin, k_pass], -1)
```

Precompute opportunities:

- For static max context, `inv_freq` is constant and cos/sin can be precomputed for `[max_position_embeddings, rotary_dim]` per dtype/device.
- Dynamic `position_ids` still require gather/indexing by actual positions and cache offset.
- Advanced RoPE types are available through shared `ROPE_INIT_FUNCTIONS`, but inspected configs use default RoPE only. First DinoML integration can reject non-default `rope_parameters["rope_type"]`.

## 8. Preprocessing and input packing

Core GPU inputs:

- `input_ids`: `[B,S]` integer token IDs, mutually exclusive with `inputs_embeds`.
- `inputs_embeds`: optional `[B,S,H]` caller-provided embeddings.
- `attention_mask`: optional `[B,total_kv]` 2D padding mask or already prepared 4D mask.
- `position_ids`: optional `[B,S]`; if absent, source creates `arange(S) + past_seen_tokens` and unsqueezes to `[1,S]`.
- `past_key_values`: optional `Cache`; if `use_cache=True` and absent, source creates `DynamicCache(config)`.

CPU/data-pipeline work:

- Tokenization and chat template expansion are outside the model graph.
- Generation uses EOS token IDs `[151329, 151336, 151338]` and pad token ID `151329` from official configs.
- Special tokenizer IDs are prompt-construction state. No modality placeholders, token-type IDs, image/audio grid metadata, or `cu_seqlens` tensors are part of native GLM4.

Packed sequence behavior:

- The model has no GLM4-specific varlen API.
- Shared mask utilities can infer packed sequences from nonmonotonic `position_ids` only when `attention_mask is None` and `past_key_values is None`.
- A first DinoML path can support plain causal prefill/decode and reject packed/nonmonotonic `position_ids` until a mask-lowering plan exists.

Generation-controller notes:

- `logits_to_keep=0` means all logits because Python `slice(0, None)` selects the full sequence. For efficient generation, pass `logits_to_keep=1` or lower an equivalent last-token-only logits path.
- Sampling, stopping on multiple EOS IDs, and chat template details are controller behavior, not core module graph behavior.

## 9. Graph rewrite / lowering opportunities

### Rewrite: packed SwiGLU projection

Source pattern:

```text
gate_up = Linear(H -> 2I, bias=False)
gate, up = chunk(gate_up, 2, dim=-1)
out = Linear(I -> H, bias=False)(silu(gate) * up)
```

Replacement:

```text
GEMM(H, 2I) -> split rows [0:I], [I:2I] -> fused SiLU-mul -> GEMM(I, H)
```

Preconditions:

- `gate_up_proj.weight` is native HF row-major `[2I, H]`.
- Split is exactly two equal chunks on last dimension in `[gate, up]` order.
- Activation is `silu`.
- No bias on either MLP projection.

Failure cases: any checkpoint/source variant with separate gate/up projections, non-SiLU activation, quantized packed layout that cannot expose row blocks, or altered split order.

Parity test sketch: compare MLP-only output for random `[B,S,H]` against Transformers in fp32 and BF16.

### Rewrite: GQA attention without physical repeat

Source pattern:

```text
repeat_kv(K,V, QH // KVH) -> dense attention
```

Replacement:

```text
GQA attention kernel consuming Q [B,QH,S,D], K/V [B,KVH,T,D]
```

Preconditions:

- `QH % KVH == 0`.
- Causal self-attention only.
- K is already RoPE-applied before cache write.
- Mask is plain causal or supported additive padding mask.

Failure cases: unsupported 4D custom masks, packed sequence masks, backend-specific `BlockMask`, non-default attention math, or future sliding/cache layer types.

Parity test sketch: compare eager attention output and one-token decode over random tensors, including 9B `KVH=2`, 32B `KVH=2`, and Rumination `KVH=8` shapes.

### Rewrite: Q/K/V projection grouping

Source pattern: three separate linear modules.

Replacement:

```text
one fused QKV GEMM producing [Q rows, K rows, V rows] -> split
```

Preconditions:

- All three projections have the same input tensor and dtype.
- Bias presence is identical for Q/K/V from `attention_bias`.
- Output row order is exactly all-Q, all-K, all-V.
- Weight transform concatenates HF weights along output rows: `cat([q.weight, k.weight, v.weight], dim=0)`, and similarly for bias when present.

Failure cases: quantized weights with incompatible packing, tensor-parallel sharding that already splits modules, or checkpoint adapters targeting individual projections.

Parity test sketch: compare separate vs fused projection tensors before RoPE for 9B bias and 32B no-bias configs.

### Rewrite: last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden to [B,1,H] before LM-head GEMM during decode
```

Preconditions:

- Generation path only needs next-token logits.
- No loss computation and no user-requested full-sequence logits.
- `logits_to_keep` is `1` or equivalent index tensor.

Failure cases: training/loss, perplexity/full-logit evaluation, or arbitrary tensor-valued `logits_to_keep`.

### Layout guards

No global NHWC/channel-last rewrite applies. Sequence/head layout transforms are semantic:

- `[B,S,QH,D] -> [B,QH,S,D]` before attention.
- Attention output returns `[B,QH,S,D]`, then transposes back and reshapes to `[B,S,QH*D]`.

Conceptual `no_layout_translation()` guard should cover attention reshape/transpose/RoPE/cache regions unless a dedicated attention layout pass owns every consumer.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over hidden width and over attention/MLP outputs. GLM4 has four RMSNorms per block plus final norm.
- Fused QKV projection with optional bias and immediate reshape/split.
- Partial interleaved RoPE fused with Q/K projection output or attention input.
- GQA FlashAttention-style prefill and decode with KV cache, avoiding repeated K/V materialization.
- Packed SwiGLU: `gate_up GEMM -> silu(gate) * up`.
- Last-token-only LM head for decode.

Medium priority:

- Residual add fused with post-self-attention RMSNorm output where scheduling allows.
- Residual add fused with post-MLP RMSNorm output.
- Bias handling variants for 9B Q/K/V versus 32B no-bias.
- Cos/sin precompute/gather kernels for long context.

Lower priority:

- Sequence/token classification heads.
- Generic packed-sequence mask lowering.
- Dynamic/linear/yarn/longrope support not used by inspected configs.
- Tensor-parallel sharding plans.

## 11. Runtime staging plan

Stage 1: parse GLM4 configs and reject unsupported variants clearly.

- Accept `model_type="glm4"`, `architectures` containing `Glm4ForCausalLM`, default RoPE, causal self-attention.
- Record `head_dim`, `num_key_value_heads`, `partial_rotary_factor`, and `attention_bias` explicitly.

Stage 2: load weights and run embedding plus one decoder block parity.

- Start with the tiny random checkpoint.
- Validate Q/K/V separate projections, GLM4 interleaved partial RoPE, sandwich RMSNorms, and packed SwiGLU.

Stage 3: full prefill parity without optimized cache scheduling.

- Run all layers for tiny and a reduced random config.
- Use eager-style attention math as reference.

Stage 4: decode with KV cache.

- Store K/V as `[B,KVH,T,D]` after RoPE.
- Validate position offset and cache append semantics.

Stage 5: optimized attention and GEMM fusions.

- Add GQA attention kernels for `KVH=2` and `KVH=8`.
- Add fused QKV projection and SwiGLU.

Stage 6: production checkpoint smoke.

- Load 9B and 32B metadata/weights where available.
- Run short-prompt prefill/decode parity against Transformers on BF16.

Initially stub/defer classification heads, training loss, packed sequences, tensor parallelism, and non-default RoPE.

## 12. Parity and validation plan

Recommended tests:

- Config parsing tests for tiny, 9B, 32B, Rumination 32B.
- RMSNorm random tensor parity: fp32 tolerance `1e-5`, BF16 tolerance around `5e-2` relative/absolute depending on accumulation path.
- GLM4 RoPE parity against source for even/odd interleaved rotation, including `partial_rotary_factor=0.5`.
- Q/K/V projection parity for both `attention_bias=True` and `False`.
- Attention parity for `QH/KVH` pairs `32/2`, `48/2`, and `48/8`; include causal mask and optional padding mask.
- Single decoder block parity with random weights.
- After-N-layer parity on the tiny checkpoint.
- Prefill logits parity for a short tokenized prompt.
- Decode parity for at least two generated steps, checking cache length and logits.
- Controller parity smoke: stopping on EOS IDs `[151329, 151336, 151338]`.

Tolerances:

- fp32 custom ops: `atol=1e-5`, `rtol=1e-5`.
- BF16 block/logit parity: start with `atol=5e-2`, `rtol=5e-2`; tighten per kernel once accumulation order is fixed.
- Fused attention may need slightly looser tolerances than eager because Transformers eager softmax upcasts to fp32.

No DinoML tests were run for this audit because the task is docs-only.

## 13. Performance probes

- Prefill-only tokens/sec sweep over `S={128,1024,4096,32768}` and batch sizes.
- Decode tokens/sec sweep for KV cache lengths `T={128,1024,4096,32768,131072}`.
- GQA backend comparison: physical-repeat eager equivalent vs native GQA attention.
- KV cache memory usage for `KVH=2` and `KVH=8`; Rumination has 4x larger KV cache than the 2-KV-head 32B at the same sequence length.
- QKV projection fusion benchmark with and without bias.
- SwiGLU fusion benchmark for 9B and 32B MLP widths.
- RMSNorm bandwidth benchmark, including four norms per block.
- LM-head benchmark full sequence vs last-token-only.
- Mask creation overhead for plain causal, padding mask, and packed-position fallback.
- Load-time weight bandwidth/provenance probe for BF16 safetensors.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing.
- Sequence classification and token classification heads.
- Beam search and advanced generation processors beyond EOS/pad handling.
- Packed/nonmonotonic position-id masks.
- 4D custom masks and FlexAttention `BlockMask` lowering.
- Non-default RoPE variants until a representative GLM4 checkpoint requires them.
- Tensor parallel and pipeline parallel execution.
- Quantized/packed third-party checkpoints unless separately audited.
- GLM4-MoE, GLM4-MoE-Lite, GLM4V, GLM-OCR, and legacy `chatglm` model types.

## 15. Final implementation checklist

- [ ] Parse `Glm4Config` and checkpoint config fields, including `head_dim`, `num_key_value_heads`, `attention_bias`, and standardized RoPE parameters.
- [ ] Load/token-embedding and LM-head weights without tying unless config requests it.
- [ ] Load separate Q/K/V/O projection weights and optional Q/K/V bias.
- [ ] Implement GLM4 RMSNorm with fp32 variance.
- [ ] Implement packed SwiGLU `gate_up_proj` split order `[gate, up]`.
- [ ] Implement GLM4 interleaved partial RoPE.
- [ ] Implement causal GQA attention with KV cache shape `[B,KVH,T,D]`.
- [ ] Implement cache update after RoPE.
- [ ] Implement GLM4 decoder block with post-attention and post-MLP sandwich RMSNorms.
- [ ] Implement final RMSNorm and LM head with `logits_to_keep`.
- [ ] Add guarded fused QKV projection rewrite.
- [ ] Add guarded native-GQA attention rewrite.
- [ ] Add guarded packed-SwiGLU fusion.
- [ ] Add last-token-only logits rewrite.
- [ ] Add tiny checkpoint single-block and full-model parity tests.
- [ ] Add 9B/32B/Rumination config admission tests.
- [ ] Add prefill and decode parity tests with cache.
- [ ] Benchmark RMSNorm, GQA attention, SwiGLU, KV cache memory, and LM-head slices.
