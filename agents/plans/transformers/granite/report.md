# Hugging Face Transformers Audit: granite

## 1. Source basis

```text
Transformers commit/version:
  b75feb2af64c3e29cbbc1bd859958c5432cc7ed4

Model family:
  granite

Primary runtime target:
  GraniteForCausalLM, text-only autoregressive causal LM.

Config source:
  X:/H/transformers/src/transformers/models/granite/configuration_granite.py
  Hugging Face config.json snapshots under agents/plans/transformers/granite/_sources/

Source files inspected:
  X:/H/transformers/src/transformers/models/granite/modular_granite.py
  X:/H/transformers/src/transformers/models/granite/modeling_granite.py
  X:/H/transformers/src/transformers/models/granite/configuration_granite.py
  X:/H/transformers/src/transformers/models/granitemoe/modeling_granitemoe.py
  X:/H/transformers/src/transformers/models/granitemoe/configuration_granitemoe.py
  X:/H/transformers/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py
  X:/H/transformers/src/transformers/models/granitemoehybrid/configuration_granitemoehybrid.py
  X:/H/transformers/src/transformers/configuration_utils.py
  X:/H/transformers/src/transformers/modeling_rope_utils.py

Any missing files or assumptions:
  modeling_granite.py is generated from modular_granite.py. Future HF source
  edits should track modular_granite.py as authoritative.
  No tokenizer or processor file is required for the dense model graph beyond
  normal text input_ids/attention_mask handling.
```

Representative config snapshots:

| Model id | HF repo sha | Scope |
| --- | --- | --- |
| `katuni4ka/tiny-random-granite` | `705ce2cc1b5acc4a57f24bb6cea5cbec0089388c` | Unofficial tiny/debug mirror, `model_type=granite`. |
| `ibm-granite/granite-3.0-2b-base` | `a8462c21f5e1f27be4536dc36d6cc789da23fbd6` | Official dense Granite. |
| `ibm-granite/granite-3.0-8b-base` | `4fad7f8ad56393dcef4e34e37a35962bd091f320` | Official dense Granite. |
| `ibm-granite/granite-3.1-8b-base` | `39975ba909950a1bad8fa1cb6981d4d048b75553` | Official dense long-context Granite. |
| `ibm-granite/granite-3.2-8b-instruct` | `610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695` | Official dense long-context instruct Granite. |
| `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | Official dense long-context instruct Granite with larger vocab. |
| `ibm-granite/granite-3b-code-base` | `b67f3dabc0b6d00ae477bd76732c7f3458a2caf3` | Out of scope here: config says `model_type=llama`. |
| `ibm-granite/granite-8b-code-base` | `d21e874b592aebc1a355d203e932d9055574b685` | Out of scope here: config says `model_type=llama`. |

## 2. High-level architecture

Granite dense is a text-only decoder-only Transformer. It is Llama-like, but
the source adds Granite-specific scalar multipliers around embeddings,
attention scores, residual branches, and final logits.

```text
tokenize on CPU -> input_ids/attention_mask
  -> token embedding * embedding_multiplier
  -> shared RoPE cos/sin for current positions
  -> N dense decoder blocks
  -> final RMSNorm
  -> optional last-token-only lm_head
  -> logits / logits_scaling -> sampling
```

Stage decomposition:

| Stage | Runtime graph ownership | Notes |
| --- | --- | --- |
| Text tokenization | CPU/data pipeline | Standard tokenizer work; no model-coupled image/audio packing. |
| Embedding/prefix construction | GPU graph | `Embedding(vocab_size, hidden_size)` then multiply by `embedding_multiplier`. |
| Prefill | GPU graph | Dense causal self-attention with RoPE and GQA KV cache writes. |
| Decode | GPU graph + cache ABI | One or few query tokens, append pre-RoPE? No: source applies RoPE before cache update, so cache stores RoPE-applied K. |
| Logits | GPU graph | `lm_head(hidden_states[:, slice_indices, :]) / logits_scaling`; `logits_to_keep` allows last-token-only logits. |

## 3. Important config dimensions

Source defaults from `GraniteConfig`:

| Field | Source default | Runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 32000 | Token embedding rows and LM head output columns. |
| `hidden_size` | 4096 | Decoder width. |
| `intermediate_size` | 11008 | SwiGLU inner width. |
| `num_hidden_layers` | 32 | Dense decoder block count. |
| `num_attention_heads` | 32 | Query head count. |
| `num_key_value_heads` | `None -> num_attention_heads` | GQA/MQA when smaller than query heads. |
| `head_dim` | absent by default | Source uses `hidden_size // num_attention_heads` if absent. |
| `hidden_act` | `silu` | SwiGLU activation. |
| `max_position_embeddings` | 2048 | RoPE/cache admission length. |
| `rms_norm_eps` | `1e-6` | RMSNorm epsilon. |
| `attention_bias` | `False` | Bias on q/k/v/o projections. |
| `mlp_bias` | `False` | Bias on gate/up/down projections. |
| `embedding_multiplier` | `1.0` | Multiplies token embeddings before blocks. |
| `attention_multiplier` | `1.0` | Attention score scale, replacing normal `1/sqrt(head_dim)`. |
| `residual_multiplier` | `1.0` | Multiplies attention and MLP branch outputs before residual add. |
| `logits_scaling` | `1.0` | Divides LM logits after projection. |
| `use_cache` | `True` | DynamicCache by default when generation requests cache. |
| `tie_word_embeddings` | `False` | Config default, but production dense checkpoints set `True`. |

Representative checkpoint sweep, from saved `config.json` snapshots:

| Model | hidden | layers | Q heads | KV heads | head_dim inferred | MLP | vocab | max pos | RoPE theta | multipliers `(emb, attn, residual, logits)` | dtype | tied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `katuni4ka/tiny-random-granite` | 32 | 6 | 4 | 2 | 8 | 128 | 49155 | 131072 | 5000000 | `(12, 0.015625, 0.22, 8)` | float32 | true |
| `granite-3.0-2b-base` | 2048 | 40 | 32 | 8 | 64 | 8192 | 49152 | 4096 | 10000 | `(12, 0.015625, 0.22, 8)` | omitted | true |
| `granite-3.0-8b-base` | 4096 | 40 | 32 | 8 | 128 | 12800 | 49152 | 4096 | 10000 | `(12, 0.0078125, 0.22, 16)` | omitted | true |
| `granite-3.1-8b-base` | 4096 | 40 | 32 | 8 | 128 | 12800 | 49152 | 131072 | 10000000 | `(12, 0.0078125, 0.22, 16)` | bfloat16 | true |
| `granite-3.2-8b-instruct` | 4096 | 40 | 32 | 8 | 128 | 12800 | 49155 | 131072 | 10000000 | `(12, 0.0078125, 0.22, 16)` | bfloat16 | true |
| `granite-3.3-8b-instruct` | 4096 | 40 | 32 | 8 | 128 | 12800 | 49159 | 131072 | 10000000 | `(12, 0.0078125, 0.22, 16)` | bfloat16 | true |

Checkpoint configs use legacy `rope_theta`/`rope_scaling` fields. The current
Transformers config stack standardizes them into `rope_parameters` before
`GraniteRotaryEmbedding` reads `rope_parameters["rope_type"]` and
`rope_parameters["rope_theta"]`.

## 3a. Family variation traps

- Dense `granite` is not the same runtime family as `granitemoe`,
  `granitemoehybrid`, or older Granite code checkpoints with `model_type=llama`.
- Dense production checkpoints use GQA: `num_key_value_heads=8` and
  `num_attention_heads=32`, so KV cache shape is smaller than expanded attention
  shape.
- `attention_multiplier` is the attention score scaling. Do not silently replace
  it with `1 / sqrt(head_dim)`, even though current 8B and 2B values match that
  formula for inferred head dims.
- `embedding_multiplier`, `residual_multiplier`, and `logits_scaling` are
  source-visible forward math and required for parity.
- `head_dim` can be explicit in config even though inspected checkpoints omit
  it. Do not infer projection widths from `hidden_size` alone when loading an
  arbitrary Granite config.
- Production configs set `tie_word_embeddings=True`; source still constructs
  `lm_head` as a Linear module and relies on the PreTrainedModel tying contract.
- `attention_bias` and `mlp_bias` are config-controlled. Inspected dense
  production configs set both false, but source supports true.
- Long-context 3.1+ configs change `max_position_embeddings` to 131072 and
  `rope_theta` to 10000000. This changes RoPE values and cache admission, not
  block topology.
- No sliding-window attention field is read by dense `granite`; do not import
  Mixtral/GraniteMoe sliding-window assumptions.
- `logits_to_keep` can be an int or tensor index. First integration can restrict
  to `0` or `1`, but the source head supports indexed slicing.
- Layout is text-sequence first: hidden tensors are `[batch, seq, hidden]`,
  attention tensors are reshaped to `[batch, heads, seq, head_dim]`. No
  NCHW/NHWC layout translation is relevant.

Dense `granite` versus adjacent Granite families:

| Family | Same as dense Granite | Important differences |
| --- | --- | --- |
| `granitemoe` | Causal GQA attention shape, RoPE path, RMSNorm, embedding/residual/attention/logit multipliers. | MLP is replaced by top-k sparse MoE: router linear, topk, softmax gates, sort/group tokens by expert, per-expert packed linears, `index_add` combine, optional router auxiliary outputs. Dense `granite` should not admit these ops. |
| `granitemoehybrid` | Attention layers still use Granite-like GQA and multipliers when present. | Blocks may be Mamba or attention according to `layer_types`; cache includes Mamba conv/recurrent states, not only KV. It supports `position_embedding_type=None`, which means NoPE for attention layers because `position_embeddings` is `None` and the attention module skips RoPE. Dense `granite` has no such NoPE branch. |
| Granite code checkpoints | Branding/tokenizer family may look related. | Saved configs inspected for `granite-3b-code-base` and `granite-8b-code-base` are `model_type=llama` with Llama source behavior, attention/MLP biases true, and no Granite scalar multipliers in this dense source path. |

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- Scalar multiply on embeddings: `inputs_embeds * embedding_multiplier`.
- View/reshape/transpose for Q/K/V: `[B, S, H] -> [B, heads, S, head_dim]`.
- Contiguous reshape after attention: `[B, heads, S, D] -> [B, S, heads * D]`.
- Slice/gather for `logits_to_keep`, especially last-token-only logits.
- Optional tied-weight alias between `model.embed_tokens.weight` and
  `lm_head.weight`.

Neural network primitives:

- RMSNorm over last dimension with fp32 variance and output cast back to input
  dtype.
- Dense Linear projections:
  - Q: `Linear(H -> num_attention_heads * head_dim)`.
  - K/V: `Linear(H -> num_key_value_heads * head_dim)`.
  - O: `Linear(num_attention_heads * head_dim -> H)`.
  - MLP gate/up: `Linear(H -> intermediate_size)`.
  - MLP down: `Linear(intermediate_size -> H)`.
  - LM head: `Linear(H -> vocab_size, bias=False)`.
- SwiGLU: `silu(gate_proj(x)) * up_proj(x)`.
- Residual branch scale/add: `residual + branch * residual_multiplier`.
- Final logits divide by `logits_scaling`.

Attention primitives:

- Causal self-attention only.
- GQA/MQA support through KV head count less than query head count.
- RoPE on Q and K before cache update.
- Attention score scale is `config.attention_multiplier`.
- Mask addition before fp32 softmax.
- Dropout is training-only; inference uses zero dropout.
- Source supports eager, SDPA, FlashAttention, and flex attention dispatch
  through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary ops:

- Default RoPE cos/sin generation in fp32 from `position_ids` and
  `inv_freq`, then cast to hidden dtype.
- Dynamic RoPE update path exists through decorator for advanced rope types, but
  inspected dense configs use default RoPE with no scaling dict.
- NoPE is not implemented by dense `granite`: `GraniteModel` always creates a
  `GraniteRotaryEmbedding` and `GraniteAttention` always unpacks
  `position_embeddings`. NoPE belongs to `granitemoehybrid` when
  `position_embedding_type` is `None`.

Generation/cache ops:

- DynamicCache creation when `use_cache=True` and no cache is supplied.
- Per-layer KV cache update receives RoPE-applied K and raw V.
- Position IDs default to `arange(seq) + past_seen_tokens`.
- Causal mask construction must account for attention mask, cache length, and
  position IDs.

Preprocessing-coupled ops:

- CPU tokenizer emits `input_ids` and optional `attention_mask`.
- No processor-derived grids, images, audio, or packed multimodal metadata.

Distributed/tensor-parallel ops:

- Config declares TP plans for q/k/v/gate/up colwise, o/down rowwise, and LM
  head colwise gather. DinoML can ignore initially for single-GPU parity.

## 5. Layer/block breakdown

Dense decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x_norm = RMSNorm(x)
q = Linear(H -> q_heads * head_dim, bias=attention_bias)(x_norm)
k = Linear(H -> kv_heads * head_dim, bias=attention_bias)(x_norm)
v = Linear(H -> kv_heads * head_dim, bias=attention_bias)(x_norm)
q = q.view(B, S, q_heads, D).transpose(1, 2)
k = k.view(B, S, kv_heads, D).transpose(1, 2)
v = v.view(B, S, kv_heads, D).transpose(1, 2)
q, k = RoPE(q, k, cos, sin)
k, v = cache.update(k, v, layer_idx)       # if cache present
attn = causal_attention(q, k, v, scale=attention_multiplier)
attn = attn.transpose(1, 2).reshape(B, S, q_heads * D)
attn = o_proj(attn)
x = residual + attn * residual_multiplier

residual = x
x_norm = RMSNorm(x)
mlp = down_proj(silu(gate_proj(x_norm)) * up_proj(x_norm))
x = residual + mlp * residual_multiplier
```

For `granite-3.1/3.2/3.3-8b`, shapes are:

```text
H=4096, layers=40, q_heads=32, kv_heads=8, head_dim=128
Q: 4096 -> 4096
K: 4096 -> 1024
V: 4096 -> 1024
O: 4096 -> 4096
gate/up: 4096 -> 12800
down: 12800 -> 4096
lm_head: 4096 -> vocab_size
```

For `granite-3.0/3.3-2b`, shapes are:

```text
H=2048, layers=40, q_heads=32, kv_heads=8, head_dim=64
Q: 2048 -> 2048
K/V: 2048 -> 512
O: 2048 -> 2048
gate/up: 2048 -> 8192
down: 8192 -> 2048
lm_head: 2048 -> vocab_size
```

## 6. Attention requirements

Granite dense requires autoregressive causal self-attention.

| Requirement | Dense Granite behavior |
| --- | --- |
| Causality | Causal decoder mask from `create_causal_mask`. |
| Attention type | Self-attention only; no cross-attention. |
| MHA/MQA/GQA | MHA when `num_key_value_heads == num_attention_heads`, GQA when smaller. Production dense configs use 32 Q heads and 8 KV heads. |
| Head dim | `config.head_dim` if present, otherwise `hidden_size // num_attention_heads`. |
| Score scale | `attention_multiplier`, passed into attention backend. |
| RoPE placement | Applied to Q and K before cache update and before backend dispatch. |
| Masking | Eager path adds mask to scores before softmax. |
| Softmax dtype | Eager path computes softmax in fp32 and casts to query dtype. |
| Packed/varlen | Dense source accepts generic backend kwargs, but primary graph is normal padded `[B, S]`. |
| Sliding/local | Not implemented for dense `granite`. |
| Backend compatibility | `_supports_flash_attn`, `_supports_sdpa`, `_supports_flex_attn` are true. |

Cache layout before repeat expansion:

```text
key_states/value_states per layer:
  [batch, num_key_value_heads, cached_seq, head_dim]
```

For eager attention, K/V are repeated logically to:

```text
[batch, num_attention_heads, cached_seq, head_dim]
```

The optimized DinoML path should avoid materializing `repeat_kv` where the
attention kernel accepts GQA directly. Eager repeat is a parity fallback, not a
desirable production layout.

## 7. Position encoding and custom math

Default RoPE inverse frequencies:

```python
def granite_default_inv_freq(config):
    base = config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

Runtime cos/sin generation:

```python
def granite_rope_cos_sin(inv_freq, position_ids, dtype):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
```

Apply RoPE:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_granite_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

Granite-specific scalar math:

```text
embedding path: hidden = embed_tokens(input_ids) * embedding_multiplier
attention scores: scores = (q @ k.T) * attention_multiplier
residuals: hidden = residual + branch * residual_multiplier
logits: logits = lm_head(hidden) / logits_scaling
```

What can be precomputed:

- `inv_freq` is constant per model/config.
- Static position cos/sin tables can be precomputed for bounded max sequence
  when default RoPE is used, but dynamic position IDs and cache offsets still
  require indexed selection.
- Long-context Granite uses large max position values; avoid blindly
  materializing full `[131072, head_dim]` tables unless memory is acceptable.

## 8. Preprocessing and input packing

Dense Granite has no model-coupled image/audio preprocessing and no multimodal
embedding stitch. Runtime inputs are:

| Input | Shape | Notes |
| --- | --- | --- |
| `input_ids` | `[batch, seq]` int token IDs | Mutually exclusive with `inputs_embeds`. |
| `inputs_embeds` | `[batch, seq, hidden]` | Already dense hidden embeddings; still multiplied by `embedding_multiplier`. |
| `attention_mask` | Usually `[batch, seq]` | Consumed by `create_causal_mask`; no custom packed metadata required for first parity. |
| `position_ids` | `[batch, seq]` | Optional; default generated from cache length. |
| `past_key_values` | per-layer cache | Optional; created as `DynamicCache` when `use_cache=True`. |

Generation-controller behavior outside core graph:

- Tokenizer chat templates and sampling are not part of the model module.
- `logits_to_keep=1` is important for efficient decode because source can avoid
  full-sequence logits.
- Beam search and cache reorder are GenerationMixin concerns; first DinoML
  parity can use greedy/sampling with batch-stable cache.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Three Separate Q/K/V Linears -> Packed QKV Projection

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
packed_qkv = Linear(H -> (q_heads + 2 * kv_heads) * head_dim)
split [Q rows, K rows, V rows]
```

Preconditions:

- Same input tensor and dtype.
- Same bias policy across q/k/v, or packed bias is constructed with the same
  split order.
- Weight rows are packed in source order `[q_proj.weight; k_proj.weight; v_proj.weight]`.
- Output split sizes are exactly:
  `q_heads * head_dim`, `kv_heads * head_dim`, `kv_heads * head_dim`.

Failure cases:

- Explicit `head_dim` makes widths differ from naive hidden-size assumptions.
- Tensor-parallel sharding may already pack or shard projections externally.

Parity test sketch:

- Compare packed projection splits against source q/k/v outputs before RoPE for
  random fp32 and bf16 inputs.

### Rewrite: GQA repeat_kv -> Native GQA Attention

Source pattern:

```text
repeat_kv(k, q_heads // kv_heads)
repeat_kv(v, q_heads // kv_heads)
attention(q, repeated_k, repeated_v)
```

Replacement:

```text
attention_gqa(q, k, v, q_heads, kv_heads)
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Backend preserves source scaling, mask addition, and RoPE-applied cached K.
- Attention output matches `[B, q_heads, S_q, head_dim]`.

Failure cases:

- Backend only supports MHA or repeats K/V internally with different numerical
  order.

Parity test sketch:

- Compare eager repeated-KV attention with native GQA prefill and one-token
  decode at small and production-like head counts.

### Rewrite: SwiGLU Pair -> Fused GEMM/SwiGLU/Down

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
packed_gate_up = Linear(H -> 2 * intermediate)
fused_silu_mul
down Linear(intermediate -> H)
```

Preconditions:

- `hidden_act == "silu"`.
- Gate/up weights packed as `[gate_proj; up_proj]`.
- Bias handling matches `mlp_bias`.

Failure cases:

- Non-silu activations in custom configs.
- Quantized or sharded weights with incompatible packing.

Parity test sketch:

- Single MLP module parity with random weights and activation edge cases around
  zero/large values.

### Rewrite: Last-Token-Only Logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
gather needed hidden rows -> GEMM only for kept tokens
```

Preconditions:

- Generation path asks for `logits_to_keep=1` or a static known slice.
- Loss/training path is not in scope.

Failure cases:

- Full prefill logits requested for scoring.
- Tensor index `logits_to_keep` with non-contiguous or dynamic selected rows.

Parity test sketch:

- Compare full logits slice and optimized gathered logits for `logits_to_keep`
  values 0, 1, and a small tensor index.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: appears twice per block plus final norm; fp32 accumulation and dtype
  cast must match source.
- GQA FlashAttention with RoPE-applied KV cache: dominant prefill/decode cost
  and avoids materialized `repeat_kv`.
- QKV projection + RoPE preparation: reduces launch count and memory traffic.
- SwiGLU activation multiply: gate/up projections plus activation/mul are
  repeated in every block.
- Last-token-only logits: large vocab GEMM can dominate decode if full sequence
  logits are computed.

Medium priority:

- Packed QKV GEMM and packed gate/up GEMM weight loading rewrites.
- Residual scale/add fused with projection outputs.
- Embedding lookup plus scalar multiply.
- LM head tied-weight handling and optional GGUF/dequant provider path for
  large vocab/head weights.

Lower priority:

- Dynamic/advanced RoPE scaling support beyond default RoPE. Inspected dense
  configs do not require non-default rope types.
- Tensor-parallel collectives. Source declares plans, but single-GPU parity can
  stage first.
- Training-only loss and dropout.

## 11. Runtime staging plan

Stage 1: Config and weight loading

- Admit `model_type=granite` only.
- Reject `model_type=llama` Granite code checkpoints for this audit path.
- Parse scalar multipliers, head counts, optional `head_dim`, bias flags,
  `rope_theta`, and tied embedding flag.

Stage 2: Single block parity

- Implement RMSNorm, Q/K/V/O linears, RoPE, causal attention, residual scaling,
  and SwiGLU MLP for one block without cache.

Stage 3: Full prefill parity

- Run all dense blocks with padded causal mask and final norm/head.
- Initially allow eager attention fallback; then switch to native GQA attention.

Stage 4: Decode with KV cache

- Define per-layer cache tensors `[B, kv_heads, max_seq, head_dim]` for K and V.
- Store K after RoPE, V before any repeat expansion.
- Validate position ID offset from prior cache length.

Stage 5: Production attention path

- Add fused GQA FlashAttention/SDPA-compatible lowering with source scaling and
  mask semantics.

Stage 6: Rewrites/fusions

- Enable packed QKV, packed gate/up, RMSNorm fusion, residual-scale fusion, and
  last-token-only logits.

Stage 7: Weight lifecycle and quantization

- Add GGUF or other encoded weight loading only as an explicit constant/provider
  contract. Dense Granite source does not define custom quantized storage.

## 12. Parity and validation plan

- Config parsing tests:
  - Production 2B/8B configs.
  - Long-context 3.1+ configs with `rope_theta=10000000`.
  - Tiny debug config.
  - Out-of-scope checks for `model_type=llama`, `granitemoe`, and
    `granitemoehybrid`.
- Custom op tests:
  - RMSNorm fp32 accumulation with fp32/fp16/bf16 storage.
  - RoPE cos/sin and `rotate_half` against Transformers for fixed positions.
  - Residual multiplier and logits scaling exact placement.
- Single-layer parity:
  - Random tiny config, no cache, fp32 tolerance around `1e-5`.
  - bf16/fp16 tolerance around `1e-2` relative for fused attention/MLP.
- Full-prefill parity:
  - Tiny random checkpoint logits for short sequence.
  - Production-shape synthetic weights for shape-only/lowering validation.
- Decode parity:
  - Prefill N tokens, decode one token, compare cache-updated logits.
  - Verify cached K shape before repeat expansion and RoPE-applied cache values.
- End-to-end text parity:
  - Tokenizer + greedy generation on a small prompt for an official checkpoint
    when weights are available.

## 13. Performance probes

- Prefill throughput over sequence lengths 128, 512, 2048, 4096, and selected
  long-context lengths for 3.1+.
- Decode tokens/sec for batch sizes 1, 4, 16, and 32 with warm cache.
- KV cache memory usage:
  `layers * 2 * batch * kv_heads * max_seq * head_dim * dtype_bytes`.
- Attention backend comparison: eager repeat-KV, SDPA, FlashAttention, DinoML
  native GQA.
- QKV packed versus separate projection GEMMs.
- SwiGLU packed gate/up versus separate GEMMs.
- Last-token-only logits versus full sequence logits.
- Dense bf16 versus fp16/fp32 path comparison.
- Encoded/GGUF load plus dequant-to-GEMM probes if weights are materialized
  through DinoML encoded constants later.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and dropout.
- Beam search, speculative decoding, and cache reorder until basic decode works.
- Tensor parallel and pipeline parallel plans.
- Non-default RoPE types unless a dense `model_type=granite` checkpoint requires
  them.
- `granitemoe` expert routing and `granitemoehybrid` Mamba/NoPE state cache;
  these are separate model families.
- Older Granite code checkpoints that are actually `model_type=llama`.
- Multimodal Granite variants such as speech or vision families.
- Quantized or packed weight formats beyond normal dense safetensors unless
  introduced through DinoML's explicit encoded-constant plan.

## 15. Final implementation checklist

- [ ] Parse dense `GraniteConfig` and reject non-`granite` model types.
- [ ] Normalize legacy `rope_theta`/`rope_scaling` into DinoML RoPE parameters.
- [ ] Load/tie `embed_tokens.weight` and `lm_head.weight` according to config.
- [ ] Implement embedding multiplier.
- [ ] Implement Granite RMSNorm with fp32 variance.
- [ ] Implement Q/K/V/O projections with optional bias and explicit head_dim.
- [ ] Implement default RoPE and apply it to Q/K before cache update.
- [ ] Implement causal GQA attention with source `attention_multiplier`.
- [ ] Define KV cache ABI `[B, kv_heads, T, head_dim]`, storing RoPE-applied K.
- [ ] Implement residual branch multiplier in both attention and MLP branches.
- [ ] Implement SwiGLU MLP with optional MLP bias.
- [ ] Implement final RMSNorm and LM head.
- [ ] Implement `logits_to_keep=1` optimized path and basic full-logits path.
- [ ] Add packed QKV rewrite with split order `[Q, K, V]`.
- [ ] Add packed gate/up rewrite with split order `[gate, up]`.
- [ ] Add one-block, prefill, and decode parity tests against Transformers.
- [ ] Add performance probes for prefill, decode, KV memory, attention backend,
  packed projections, and last-token logits.
