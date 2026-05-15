# Bamba Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: ibm-ai-platform/Bamba-9B family; primary production config ibm-ai-platform/Bamba-9B-v2
Config source: public Hugging Face config.json snapshots under _sources/
Source files inspected:
  transformers/src/transformers/models/bamba/configuration_bamba.py
  transformers/src/transformers/models/bamba/modeling_bamba.py
  transformers/src/transformers/models/bamba/modular_bamba.py
  transformers/src/transformers/cache_utils.py
Comparison files:
  transformers/src/transformers/models/mamba/modeling_mamba.py
  transformers/src/transformers/models/jamba/modeling_jamba.py
  transformers/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py
Any missing files or assumptions:
  No small/debug Bamba checkpoint was found in the public HF sweep. The sweep therefore uses public 9B training-stage and FP8/compressed variants.
```

`modeling_bamba.py` is generated from `modular_bamba.py`; future Transformers source edits should be made in the modular file, but DinoML should audit the generated file because that is the imported runtime implementation at this commit.

HF configs snapshotted:

- [ibm-ai-platform/Bamba-9B-v1](https://huggingface.co/ibm-ai-platform/Bamba-9B-v1)
- [ibm-ai-platform/Bamba-9B-1.8T](https://huggingface.co/ibm-ai-platform/Bamba-9B-1.8T)
- [ibm-ai-platform/Bamba-9B-2T](https://huggingface.co/ibm-ai-platform/Bamba-9B-2T)
- [ibm-ai-platform/Bamba-9B-v2](https://huggingface.co/ibm-ai-platform/Bamba-9B-v2)
- [ibm-ai-platform/Bamba-9B-fp8](https://huggingface.co/ibm-ai-platform/Bamba-9B-fp8)
- [ibm-ai-platform/Bamba-9B-1.8T-fp8](https://huggingface.co/ibm-ai-platform/Bamba-9B-1.8T-fp8)
- [ibm-ai-platform/Bamba-9B-2T-fp8](https://huggingface.co/ibm-ai-platform/Bamba-9B-2T-fp8)

No inspected public config returned 401/403. If gated IBM/FMS aliases are used later, access to that HF repo/token would resolve the config and weight metadata gap.

## 2. High-level architecture

Bamba is a text-only causal decoder for language generation. Each decoder layer is either a Mamba2-style selective state-space block or a full causal attention block, followed by a dense SwiGLU MLP.

```text
token ids -> token embedding -> 32 hybrid decoder blocks -> final RMSNorm -> last-token LM head -> logits
```

Primary DinoML target: `BambaForCausalLM` inference with prefill and decode. Training loss, z-loss, gradient checkpointing, output attentions, and aux output capture are deferred.

Stage decomposition:

- CPU/data pipeline: tokenizer emits `input_ids` and optional 2D `attention_mask`; default `position_ids` are generated as `arange(seq_len)` if absent.
- GPU prefill: embeddings, mixed Mamba/attention blocks, final norm, usually last-token logits.
- Decode: attention layers append KV cache; Mamba layers update fixed-size convolution and recurrent SSM states.
- Independently optimizable: Mamba mixer kernels, attention layers, MLP/RMSNorm, and last-token-only logits.

## 3. Important config dimensions

Source defaults from `BambaConfig` differ from public checkpoints in a few places, especially `mamba_d_state` and max position metadata.

| Field | Source default | Public 9B values observed | Runtime significance |
|---|---:|---:|---|
| `hidden_size` | 4096 | 4096 | token width |
| `num_hidden_layers` | 32 | 32 | layer count |
| `attn_layer_indices` | `None` | `[9, 18, 27]` | only these layers use attention; all others are Mamba |
| attention layers | 0 if unset | 3 | hybrid cache manifest |
| Mamba layers | 32 if unset | 29 | conv/SSM state count |
| `num_attention_heads` | 32 | 32 | Q heads |
| `num_key_value_heads` | 8, or `num_attention_heads` if `None` | 8 | GQA; repeat factor 4 |
| attention `head_dim` | `hidden_size // heads` | 128 | source uses `getattr(config, "head_dim", ...)` |
| `intermediate_size` | 14336 | 14336 | MLP gated/up width |
| `vocab_size` | 128000 | 128256 | LM head and embeddings |
| `max_position_embeddings` | 262144 | blank/default, 4096, or 262144 | RoPE/cache admission |
| RoPE | default, `partial_rotary_factor=0.5` forced | v2 records theta 10000 and partial 0.5 | rotary dim 64 for 128-d head |
| `mamba_expand` | 2 | 2 | Mamba intermediate 8192 |
| `mamba_n_heads` | 128 | 128 | SSM heads |
| `mamba_d_head` | auto = 64 | 64 | 128 * 64 = 8192 |
| `mamba_d_state` | 256 | 128 | SSM recurrent state width; must not trust source default for 9B |
| `mamba_n_groups` | 1 | 1 | B/C group expansion to heads |
| `mamba_d_conv` | 4 | 4 | conv state length |
| `mamba_chunk_size` | 256 | 256 | chunk scan block size |
| `hidden_act` | `silu` | `silu` | MLP and conv activation |
| projection bias | attention false, MLP false, Mamba false | false or omitted -> false | no dense projection biases except conv bias |
| `mamba_conv_bias` | true | true | depthwise conv bias |
| `use_cache` | true | true | hybrid DynamicCache |
| `num_logits_to_keep` | 1 | 1 | generation keeps last prompt logit |

Representative checkpoint sweep:

| Model | Layers | Attention idx | Max positions | BOS/EOS | Quantization metadata | Notes |
|---|---:|---|---:|---|---|---|
| `Bamba-9B-v1` | 32 | 9,18,27 | 4096 | 128000/128001 | none | current main config includes shorter max context than v2 |
| `Bamba-9B-1.8T` | 32 | 9,18,27 | omitted -> source default 262144 at this commit | 128000/128001 | none | historical 4.47 config omits RoPE fields |
| `Bamba-9B-2T` | 32 | 9,18,27 | omitted -> source default 262144 | 128000/128001 | none | same operator structure as 1.8T |
| `Bamba-9B-v2` | 32 | 9,18,27 | 262144 | 1/2 | none | v2 tokenizer/special-token IDs changed |
| `Bamba-9B-fp8` | 32 | 9,18,27 | omitted -> source default 262144 | 128000/128001 | `compressed-tensors`, FP8 Linear, ignore `lm_head` | loading/provider contract |
| `Bamba-9B-1.8T-fp8` | 32 | 9,18,27 | omitted -> source default 262144 | 128000/128001 | `compressed-tensors`, FP8 Linear, ignore `lm_head` | includes generation config snapshot |
| `Bamba-9B-2T-fp8` | 32 | 9,18,27 | omitted -> source default 262144 | 128000/128001 | `compressed-tensors`, FP8 Linear, ignore `lm_head` | same FP8 scheme |

## 3a. Family variation traps

- `attn_layer_indices` controls architecture directly. If it is omitted, the current config property makes every layer Mamba, not the public 9B hybrid pattern.
- Public 9B configs contain `mamba_dt_rank`, `attn_rotary_emb`, and `use_mamba_kernels`, but current Bamba source does not read those fields. Do not treat them as required runtime behavior for this source basis.
- Bamba source always chooses the Mamba fast path when kernels are available, on CUDA, and not torchdynamo compiling. Unlike Jamba/Mamba, the inspected Bamba forward does not gate this with `config.use_mamba_kernels`.
- Public 9B uses `mamba_d_state=128`, while the source default is 256. Runtime state shape must come from loaded config.
- `max_position_embeddings` is omitted in several older configs, so this commit's config default of 262144 becomes effective. `Bamba-9B-v1` main currently records 4096; `Bamba-9B-v2` records 262144.
- v1/v2 tokenizer special IDs differ. Do not assume BOS/EOS are stable across checkpoints.
- Attention is GQA: `num_key_value_heads=8 < num_attention_heads=32`; cached K/V shape is 8 heads before repeat.
- Attention projections are separate Q, K, V Linear modules, not a fused packed QKV weight.
- The LM head is physically separate from embeddings and `tie_word_embeddings=false` in configs. `_tied_weights_keys` exists in the class, but loaded configs say not to tie.
- The FP8 variants use `quantization_config` from `compressed-tensors` targeting `Linear` modules with dynamic FP8 activations and static FP8 weights, while ignoring `lm_head`. DinoML should route this as a loading/provider feature, not a normal dtype.
- Layout-sensitive tensors are sequence-major `[batch, seq, hidden]`; attention reshapes to `[batch, heads, seq, head_dim]`; conv uses `[batch, conv_dim, seq]`. No NHWC/channel-last translation is relevant.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B,S] -> [B,S,4096]`.
- Reshape/view/split/transpose/contiguous for attention and Mamba mixer.
- `torch.split` with exact Mamba packed projection sections `[8192, 8448, 128]`.
- `repeat_interleave`/expand/reshape for GQA and Mamba B/C group-to-head expansion.
- Padding and slicing along sequence axis for conv and chunk scan.
- `where`/mask multiply for padding suppression in Mamba layers.
- `index_select` for cache reorder on beam paths if beam search is supported.

Neural network primitives:

- RMSNorm over hidden: fp32 variance, output cast back to input dtype.
- Gated RMSNorm in Mamba: `RMSNorm(silu(gate) * y)`.
- Linear projections:
  - Attention Q: `4096 -> 4096`, no bias.
  - Attention K/V: `4096 -> 1024`, no bias.
  - Attention O: `4096 -> 4096`, no bias.
  - MLP gate/up: `4096 -> 14336`, no bias.
  - MLP down: `14336 -> 4096`, no bias.
  - Mamba `in_proj`: `4096 -> 16768`, no bias.
  - Mamba `out_proj`: `8192 -> 4096`, no bias.
  - LM head: `4096 -> 128256`, no bias; last-token path is required for efficient generation.
- Depthwise causal Conv1d over Mamba packed `x/B/C`: channels `8448`, kernel `4`, groups `8448`, padding `3`, bias true, activation `silu`.
- Elementwise `silu`, multiply, add, softplus, clamp, exp, cumsum, sum, rsqrt.
- BMM for decode slow path SSM output: `[B*128,64,128] @ [B*128,128,1]`.

Attention primitives:

- Causal self-attention only, no cross-attention.
- GQA with 32 Q heads, 8 KV heads, 128 head dim; repeat factor 4 for eager path.
- RoPE before cache update on Q and K.
- SDPA/FlashAttention-compatible path through `ALL_ATTENTION_FUNCTIONS`; eager fallback uses fp32 softmax and dropout only in training.

Position/rotary:

- Default RoPE over first 64 dims of the 128-d head due hardcoded `partial_rotary_factor=0.5`.
- Dynamic RoPE wrapper exists through generic `dynamic_rope_update`, but observed public configs use default/no scaling.

Generation/cache ops:

- Hybrid `DynamicCache(config)` / `StaticCache(config)` layer list from `config.layer_types`.
- Attention cache layers store K/V `[B,8,T,128]` before repeat.
- Mamba cache layers store conv state and recurrent state, fixed-size per layer.
- Cache reset, reorder, offload/prefetch methods exist in generic cache utils.

Recurrent/state-space cache ops:

- Per Mamba layer conv state: `[B,8448,4]`.
- Per Mamba layer recurrent state: `[B,128,64,128]`.
- Decode update order: update conv state, run causal conv update, compute dt/B/C, update recurrent state, apply C/D output, gated RMSNorm, out projection.
- Static-address mutation matters: cache layers copy into existing tensors and mark static addresses when not compiling.

Quantized/packed weight metadata:

- FP8 configs: `compressed-tensors`, `format=float-quantized`, target `Linear`, activation dynamic tensor FP8, weight static tensor FP8, `lm_head` ignored. No Bamba source path reads this metadata directly; it is handled by Transformers loading integrations.

## 5. Layer/block breakdown

Repeated 32 times, layer type from `config.layers_block_type`:

```text
residual = x
x = RMSNorm(x)
if layer is mamba:
    p = Linear_in(x)                         # [B,S,16768]
    gate, xBC, dt = split(p, [8192,8448,128])
    xBC = causal depthwise conv + silu       # [B,S,8448]
    x, B, C = split(xBC, [8192,128,128])
    y = Mamba2 SSD scan or decode state update
    y = GatedRMSNorm(y, gate)
    x = Linear_out(y)                        # [B,S,4096]
else:
    q = Linear_q(x).view(B,S,32,128).T
    k = Linear_k(x).view(B,S,8,128).T
    v = Linear_v(x).view(B,S,8,128).T
    q,k = partial RoPE(q,k)
    k,v = cache.update(k,v) if cache enabled
    x = causal GQA(q,k,v,mask) -> Linear_o
x = residual + x
residual = x
x = RMSNorm(x)
x = Linear_down(silu(Linear_gate(x)) * Linear_up(x))
x = residual + x
```

After all layers:

```text
x = final RMSNorm(x)
logits = LMHead(x[:, last logits_to_keep positions, :])
```

Public 9B layer manifest:

```text
0-8: mamba
9: attention
10-17: mamba
18: attention
19-26: mamba
27: attention
28-31: mamba
```

## 6. Attention requirements

Bamba attention appears only in selected decoder layers.

- Type: causal self-attention.
- Heads: 32 Q heads, 8 KV heads, 128 head dim.
- Cache storage: keys/values are cached as `[B,8,T,128]`; eager attention repeats to `[B,32,T,128]` only for matmul.
- RoPE: applied before cache update, so cached keys are already position encoded.
- Masking: `create_causal_mask` builds the attention mask using `attention_mask`, `past_key_values`, and `position_ids`.
- Backend: source dispatches through `ALL_ATTENTION_FUNCTIONS` with `_supports_flash_attn=True` and `_supports_sdpa=True`; eager fallback performs `QK^T * head_dim^-0.5`, adds mask, fp32 softmax, casts to query dtype, then `PV`.
- No sliding-window/local attention in current Bamba source.
- No packed varlen attention is required for baseline, but `BambaFlashAttentionKwargs` includes `cu_seq_lens_q/k`, max lengths, and `seq_idx` for advanced FlashAttention/Mamba kernels.

Slow eager attention is a parity fallback; production DinoML should use fused GQA causal attention for prefill/decode.

## 7. Position encoding and custom math

RoPE is shared across all layers but only consumed by attention layers. Bamba hardcodes `partial_rotary_factor=0.5` during config post-init for backward compatibility. For the public 9B head dim of 128, only the first 64 dims rotate.

Concise source-equivalent:

```python
def bamba_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)       # [B,1,S,rotary_dim]
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin
    return cat([q_embed, q_pass], -1), cat([k_embed, k_pass], -1)
```

Cos/sin are computed in fp32 from `position_ids` and `inv_freq`, then cast to model dtype. Default inverse frequency uses `rope_theta` from standardized `config.rope_parameters` and `head_dim`; generic non-default RoPE types are available through `ROPE_INIT_FUNCTIONS`, but not observed in public Bamba configs.

Mamba custom math:

- `A = -exp(A_log.float())`.
- `dt = softplus(dt + dt_bias)` then clamp to `time_step_limit`.
- Prefill scan pads sequence to a multiple of `mamba_chunk_size`, uses chunked cumulative sums/exponentials, and updates final recurrent state if cache is present.
- Decode uses fixed recurrent update:

```python
dA = exp(dt[..., None] * A)
dB = dt[..., None] * B[..., None, :]
state = state * dA + dB * x[..., None]
y = state @ C + D * x
```

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids`: `[B,S]` int token IDs, or exactly one `inputs_embeds`: `[B,S,4096]`.
- `attention_mask`: optional 2D mask `[B,S]`; attention receives causal mask, Mamba receives the 2D mask only when padding needs to zero states.
- `position_ids`: optional `[B,S]`; default is `arange(S).unsqueeze(0)`.

Tokenizer-coupled facts from configs:

- v1/1.8T/2T use BOS 128000 and EOS 128001.
- v2 uses BOS 1 and EOS 2.
- `pad_token_id=0` across inspected configs.

No image/audio/video processing, placeholder token stitch, codebook, or multimodal packing exists for this family.

Generation-controller behavior:

- `prepare_inputs_for_generation` sets `logits_to_keep = config.num_logits_to_keep`.
- Default config keeps only the last prompt logit during generation to avoid full-sequence LM-head memory.

## 9. Graph rewrite / lowering opportunities

### Rewrite: separate attention projections -> packed QKV launch

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x) with no bias
```

Replacement:

```text
single packed GEMM -> split [4096,1024,1024]
```

Preconditions:

- Same input tensor and dtype for Q/K/V.
- All three projections are bias-free.
- Weight packing preserves separate row blocks: Q all rows, then K all rows, then V all rows.
- Output split uses `[num_attention_heads*head_dim, num_key_value_heads*head_dim, num_key_value_heads*head_dim]`.

Failure cases: configs with attention bias, nonstandard `head_dim`, or tied/quantized loader that cannot expose dense packed weights.

Parity test: one attention layer, compare packed projection outputs before RoPE to source separate projections.

### Rewrite: SwiGLU MLP fuse

Source pattern:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:

```text
two-output or packed gate/up GEMM -> fused SiLU-mul -> down GEMM
```

Preconditions:

- `hidden_act == "silu"`.
- `mlp_bias == false` for public checkpoints.
- Gate/up weights packed in `[gate; up]` order if using one GEMM.

Failure cases: alternate activation or projection bias.

### Rewrite: Mamba in_proj split -> provider Mamba2 block

Source pattern:

```text
Linear(4096 -> 16768) -> split gate/xBC/dt -> depthwise conv -> Mamba2 chunk scan -> gated RMSNorm -> Linear(8192 -> 4096)
```

Replacement:

```text
Mamba2MixerProvider(hidden=4096, intermediate=8192, conv_dim=8448, heads=128, head_dim=64, state=128, groups=1, kernel=4, chunk=256)
```

Preconditions:

- `mamba_expand=2`, `mamba_n_heads * mamba_d_head == mamba_expand * hidden_size`.
- `mamba_n_groups` divides `mamba_n_heads`.
- Activation is `silu`/`swish`.
- Conv is depthwise with padding `kernel-1`.
- No Mamba projection bias unless provider implements it.

Failure cases: `seq_idx` packed sequence path without fast kernel support; unsupported `time_step_limit`; state dtype/device mismatch.

### Rewrite: last-token-only LM head

Source pattern:

```text
hidden_states[:, slice(-logits_to_keep, None), :] -> Linear(4096 -> vocab)
```

Replacement:

```text
Gather last K tokens before LM GEMM
```

Preconditions:

- Inference target does not need all prompt logits.
- `logits_to_keep` is positive int or validated index tensor.

Parity test: compare logits for `K=1` and full logits last position.

### Layout guard

No NHWC/channel-last rewrite applies. Protect the entire decoder with sequence-major layout assumptions unless a local fused kernel owns all internal transposes. Axis-sensitive ops include sequence padding/slicing in Conv1d, chunk reshapes along `dim=1`, attention softmax over `dim=-1`, and RMSNorm over hidden `dim=-1`.

## 10. Kernel fusion candidates

Highest priority:

- Mamba2 mixer provider: Bamba has 29 Mamba layers in public 9B; naive chunk-scan fallback is too expensive and structurally complex.
- Decode Mamba state update: fixed-size per-token conv update plus selective state update is central to generation throughput.
- GQA causal attention with KV cache: only 3 layers, but long-context attention dominates prefill at those layers.
- RMSNorm and gated RMSNorm: every block uses two RMSNorms plus Mamba uses gated RMSNorm.
- SwiGLU MLP packed gate/up and fused activation multiply.
- Last-token LM head.

Medium priority:

- Packed QKV projection for attention layers.
- RoPE fused into attention projection or attention backend.
- Mamba depthwise causal Conv1d specialized for `channels=8448`, `kernel=4`.
- Mamba prefill chunk scan benchmark/provider selection against `mamba_ssm` behavior.

Lower priority:

- Output attentions.
- Beam-search cache reorder.
- FP8 compressed-tensors direct loading/execution. Dense fallback can stage first, but production FP8 matters for the FP8 checkpoints.

## 11. Runtime staging plan

1. Parse Bamba config and reject unsupported historical fields only when they affect runtime. Treat `attn_rotary_emb`, `mamba_dt_rank`, and `use_mamba_kernels` as ignored for this source basis.
2. Load dense BF16/FP32 weights for `Bamba-9B-v2`; add shape checks for every projection and per-layer type.
3. Single-block parity for one Mamba layer and one attention layer without cache.
4. Full prefill parity with dense fallbacks for Mamba scan and attention.
5. Hybrid cache ABI: attention K/V cache plus Mamba conv/recurrent states in one manifest keyed by layer type.
6. Decode parity with one-token updates.
7. Replace Mamba fallback with provider-backed kernels or a DinoML equivalent; then optimize attention and MLP fusions.
8. Add compressed-tensors FP8 loading path as a separate provider/loading milestone.

Initially stub/defer: training loss, z-loss, output attentions, beam reorder, cache offload/prefetch, `seq_idx` packed sequence fast path, and FP8 execution.

## 12. Parity and validation plan

- Config tests: layer type expansion, attention/Mamba counts, effective defaults for omitted max-position/RoPE fields.
- Operator tests:
  - RMSNorm and gated RMSNorm vs PyTorch source math in fp32/fp16/bf16.
  - Partial RoPE over 64 dims with pass-through remaining 64 dims.
  - Mamba conv state update for prefill shorter/equal/longer than kernel 4.
  - Decode SSM update against source slow path.
- Single-layer tests:
  - Layer 0 Mamba no-cache prefill.
  - Layer 9 attention no-cache prefill and cached decode.
- Full-model tests:
  - Prefill hidden states/logits for short prompts.
  - Decode 1, 2, and N token continuation with hybrid cache.
  - Last-token-only logits match full logits slice.
- Quantization/loading tests:
  - FP8 config admission parses `compressed-tensors` metadata and either routes to dense fallback or rejects with a loading-provider reason.

Suggested tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 prefill `rtol=5e-2, atol=5e-2` around Mamba scans unless using identical kernels; decode state tests should use stricter fp32 first.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 2048, 4096, 32768 if v2 long context is admitted.
- Decode tokens/sec with batch sweep: 1, 4, 16, 64.
- Mamba layer-only prefill scan benchmark, chunk size 256.
- Mamba decode state update benchmark, including conv update and recurrent update.
- Attention layer-only prefill/decode benchmark for 3 attention layers with GQA.
- Hybrid cache memory usage: attention K/V grows with sequence; Mamba states fixed.
- LM head last-token vs full-sequence logits.
- Dense BF16 weights vs FP8/compressed loading and dequant/provider path.

## 14. Skip/defer list

- Training, labels loss, z-loss, gradient checkpointing.
- Output attentions and all hidden-state capture.
- Beam search and cache reorder for first greedy/sampling parity.
- Cache offload/prefetch.
- `seq_idx` packed sequence/varlen fast path.
- `mamba_split_conv1d_scan_combined` training-only branch.
- FP8 compressed-tensors execution, after dense path is correct.
- Multi-GPU tensor parallel/pipeline plans.

## 15. Final implementation checklist

- [ ] Parse `BambaConfig`, including effective RoPE defaults and layer type manifest.
- [ ] Load dense weights and verify all projection/state tensor shapes.
- [ ] Implement Bamba RMSNorm and gated RMSNorm.
- [ ] Implement partial RoPE for first half of attention head dim.
- [ ] Implement GQA causal attention with KV cache `[B,8,T,128]`.
- [ ] Implement Mamba2 mixer prefill fallback/provider.
- [ ] Implement Mamba decode ABI with conv state `[B,8448,4]` and recurrent state `[B,128,64,128]`.
- [ ] Implement hybrid cache manifest by layer index/type.
- [ ] Add packed QKV rewrite with strict bias/layout guards.
- [ ] Add packed SwiGLU rewrite.
- [ ] Add last-token-only LM head lowering.
- [ ] Add single Mamba layer, single attention layer, prefill, and decode parity tests.
- [ ] Add performance probes for Mamba scan, decode state update, attention, and LM head.
- [ ] Add FP8/compressed-tensors loading admission or explicit rejection.

