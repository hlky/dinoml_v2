# Transformers Audit: MLCD

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: DeepGlint-AI/mlcd-vit-bigG-patch14-448 as primary; bigG 224/336 as in-family variants
Config source: Hub config.json + preprocessor_config.json fetched 2026-05-13; source defaults from MLCDVisionConfig
Source files inspected:
- X:/H/transformers/src/transformers/models/mlcd/modular_mlcd.py
- X:/H/transformers/src/transformers/models/mlcd/modeling_mlcd.py
- X:/H/transformers/src/transformers/models/mlcd/configuration_mlcd.py
- X:/H/transformers/src/transformers/models/mlcd/convert_mlcd_weights_to_hf.py
- X:/H/transformers/src/transformers/models/auto/configuration_auto.py
- X:/H/transformers/src/transformers/models/auto/modeling_auto.py
- X:/H/transformers/src/transformers/models/auto/image_processing_auto.py
- X:/H/transformers/tests/models/mlcd/test_modeling_mlcd.py
Any missing files or assumptions: no MLCD-specific processor file exists; AutoProcessor resolves to CLIP image processing. No gated repos were found in the representative sweep.
```

`modeling_mlcd.py` and `configuration_mlcd.py` are generated from `modular_mlcd.py`; future Transformers source edits should target the modular file. Small snapshots are in `config_sweep.json` and `source_snapshot.md`.

## 2. High-level architecture

MLCD in current in-library Transformers is a vision-only ViT encoder for image feature extraction. It is not a text decoder, not contrastive CLIP/SigLIP in this source, and has no logits head.

```text
CPU image preprocessing -> NCHW pixel_values -> patch Conv2d -> CLS prepend
-> pre LayerNorm -> repeated noncausal self-attention encoder blocks with 2D RoPE
-> last_hidden_state tokens + post-LayerNorm CLS pooler_output
```

Stage decomposition:

- CPU/data pipeline: resize, center crop, RGB conversion as needed by CLIPImageProcessor, rescale/normalize to NCHW `pixel_values`.
- GPU/runtime stage 1: patch embedding and token construction.
- GPU/runtime stage 2: 48-layer bigG encoder, independently validatable with synthetic embeddings.
- GPU/runtime stage 3: CLS slice and post LayerNorm pooler.
- Downstream multimodal LLM projector is outside this family; DinoML should compose a separate audit for the LLaVA/Qwen wrapper rather than bake it into MLCD.

## 3. Important config dimensions

Source default and official bigG configs:

| field | source default | bigG 224/336/448 Hub configs | source/runtime impact |
|---|---:|---:|---|
| `hidden_size` | 1664 | 1664 | token width; Q/K/V/O and MLP input/output |
| `intermediate_size` | 8192 | 8192 | MLP hidden width |
| `num_hidden_layers` | 48 | 48 | encoder depth |
| `num_attention_heads` | 16 | 16 | MHA head count |
| `head_dim` | inferred 104 | inferred 104 | `1664 / 16`; RoPE half-dim is 52 |
| `num_key_value_groups` | 1 | omitted, effective 1 | eager path can repeat KV if changed |
| `num_channels` | 3 | 3 | patch Conv2d input channels |
| `image_size` | 336 | 224, 336, 448 | patch grid and sequence length |
| `patch_size` | 14 | 14 | Conv2d kernel/stride |
| `hidden_act` | `gelu` | `gelu` | MLP activation |
| `layer_norm_eps` | `1e-5` | `1e-5` | all LayerNorms |
| `attention_dropout` | 0.0 | 0.0 | inference dropout disabled |
| `torch_dtype` | source default absent | `float32` | config metadata only |

Representative checkpoint sweep:

| model id | config route | image | patch | layers | hidden | heads | act | status |
|---|---|---:|---:|---:|---:|---:|---|---|
| `DeepGlint-AI/mlcd-vit-base-patch32-224` | `CLIPVisionModel` | 224 | 32 | 12 | 768 | 12 | quick_gelu | MLCD-branded but out of current MLCD source scope |
| `DeepGlint-AI/mlcd-vit-large-patch14-336` | `CLIPVisionModel` | 336 | 14 | 24 | 1024 | 16 | quick_gelu | MLCD-branded but out of current MLCD source scope |
| `DeepGlint-AI/mlcd-vit-bigG-patch14-224` | `MLCDVisionModel` via `model_type: mlcd` | 224 | 14 | 48 | 1664 | 16 | gelu | in scope |
| `DeepGlint-AI/mlcd-vit-bigG-patch14-336` | `MLCDVisionModel` via `model_type: mlcd` | 336 | 14 | 48 | 1664 | 16 | gelu | in scope |
| `DeepGlint-AI/mlcd-vit-bigG-patch14-448` | `MLCDVisionModel` via `model_type: mlcd` | 448 | 14 | 48 | 1664 | 16 | gelu | primary in scope |

Sequence lengths: 224 gives `16 * 16 + 1 = 257`, 336 gives `24 * 24 + 1 = 577`, and 448 gives `32 * 32 + 1 = 1025`.

## 3a. Family variation traps

- `model_type: mlcd` is a historical Hub key specially mapped by Transformers auto classes to `MLCDVisionModel`; source config uses `mlcd_vision_model`.
- MLCD-branded base/large repos route to `CLIPVisionModel`, use quick GELU and absolute CLIP positional embeddings, and should be rejected or routed to a CLIP vision audit for this report's scope.
- `num_key_value_groups` can change eager attention into KV-repeat behavior, but official bigG configs omit it; default is ordinary MHA.
- Source applies 2D RoPE to Q/K including the CLS token via a learned `class_pos_emb`; there is no absolute position embedding table in MLCD embeddings.
- `image_size` and runtime `pixel_values.shape[-2:]` must be divisible by `patch_size`; the source uses floor division and Conv2d naturally drops incomplete border pixels if admitted unchecked. DinoML should guard divisibility.
- Source tensors are semantically NCHW through patch embedding. NHWC/channel-last can be an optimized local layout only if patch Conv2d and downstream flatten order are rewritten together.
- Attention backend is configurable through Transformers `_attn_implementation`; parity should start with eager/SDPA-compatible noncausal dense attention.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW image tensor input `[B,3,H,W]`.
- Conv2d patch embedding, `kernel_size=stride=patch_size`, no bias.
- Flatten spatial axes: `[B,C,H/P,W/P] -> [B,C,N]`.
- Transpose to tokens: `[B,C,N] -> [B,N,C]`.
- Expand learned CLS embedding to `[B,1,C]`.
- Concatenate CLS and patch tokens on sequence axis.
- Reshape/permutation around attention: `[B,S,C] -> [B,S,H,D] -> [B,H,S,D]` and back.
- CLS slice `last_hidden_state[:,0,:]`.

Neural network primitives:

- LayerNorm over hidden axis with learned weight/bias and eps `1e-5`.
- Linear `1664 -> 1664` with bias for Q/K/V/O.
- Linear `1664 -> 8192` with bias, GELU, Linear `8192 -> 1664` with bias.
- Residual adds.

Attention primitives:

- Dense noncausal self-attention.
- MHA by default: 16 heads, head dim 104.
- Optional eager KV repeat if `num_key_value_groups > 1`.
- Mask addition is supported in encoder layer signature but primary image path has no padding mask.
- Softmax computes in fp32 then casts back to query dtype in eager path.

Position/custom math:

- 2D RoPE table generation from patch grid height/width.
- `outer(arange(max_grid), inv_freq)`, indexed by flattened `[h,w]` patch positions.
- Learned `class_pos_emb` prepended before cos/sin.
- `rotate_half`, q/k fp32 RoPE math, cast back to original dtype.

Preprocessing-coupled ops:

- CLIP image preprocessing: resize shortest edge to configured size, center crop square, normalize with CLIP mean/std, output NCHW `pixel_values`.

Not required:

- Token embedding, text tower, logits, generation, KV cache, contrastive similarity, NMS, segmentation masks, decoder cross-attention.

## 5. Layer/block breakdown

Patch and token embedding:

```text
pixel_values: [B,3,H,W]
patch = Conv2d(3 -> 1664, kernel=14, stride=14, bias=False) -> [B,1664,H/14,W/14]
patch = flatten(2).transpose(1,2) -> [B,N,1664]
cls = class_embedding.expand(B,1,1664)
x = concat([cls, patch], dim=1) -> [B,N+1,1664]
x = pre_layernorm(x)
```

Encoder block, repeated 48 times for bigG:

```text
residual = x
y = LayerNorm(x)
q = Linear(1664 -> 1664, bias=True)(y).reshape(B,S,16,104)
k = Linear(1664 -> 1664, bias=True)(y).reshape(B,S,16,104)
v = Linear(1664 -> 1664, bias=True)(y).reshape(B,S,16,104)
q,k = RoPE2D(q,k, cos=[S,104], sin=[S,104])
y = Attention(q,k,v, noncausal dense, scale=104**-0.5)
y = Linear(1664 -> 1664, bias=True)(y)
x = residual + y
residual = x
y = LayerNorm(x)
y = Linear(1664 -> 8192, bias=True)(y)
y = GELU(y)
y = Linear(8192 -> 1664, bias=True)(y)
x = residual + y
```

Output:

```text
last_hidden_state = x
pooler_output = post_layernorm(x[:,0,:])
```

## 6. Attention requirements

MLCD requires encoder-style, noncausal, dense self-attention only for the primary target.

| requirement | value |
|---|---|
| causal | no |
| self/cross | self-attention |
| MHA/MQA/GQA | MHA for official bigG; source can repeat KV groups if configured |
| heads | 16 |
| KV heads | source projects 16 heads; eager `repeat_kv` expands by `num_key_value_groups` |
| head dim | 104 |
| query/key/value length | same `S = patches + 1` |
| mask | optional additive mask `[B,1,Q,K]`; not used by normal image path |
| position | 2D RoPE on Q/K before `[B,H,S,D]` attention layout |
| cache | none; not autoregressive |
| backend | eager fallback, SDPA, FlashAttention, FlexAttention are advertised by source metadata |

First DinoML parity should implement eager-equivalent dense attention. FlashAttention can be used later when it preserves noncausal full attention, fp32 softmax behavior tolerances, and RoPE placement.

## 7. Position encoding and custom math

MLCD has no learned absolute patch position embedding in the in-scope source. It computes 2D RoPE at runtime from the patch grid and prepends a learned CLS rotary position.

```python
def mlcd_rope_2d(num_h, num_w, inv_freq, class_pos_emb):
    h = arange(num_h).unsqueeze(1).expand(num_h, num_w)
    w = arange(num_w).unsqueeze(0).expand(num_h, num_w)
    pos_ids = stack([h.flatten(), w.flatten()], dim=-1)
    seq = arange(max(num_h, num_w), dtype=inv_freq.dtype)
    full = outer(seq, inv_freq)                  # [max_grid, dim/2]
    patch_rope = full[pos_ids].flatten(1)        # [num_h*num_w, dim]
    rope = concat([class_pos_emb, patch_rope], dim=0)
    emb = concat([rope, rope], dim=-1)           # [S, head_dim]
    return cos(emb), sin(emb)

def apply_rope_vision(q, k, cos, sin):
    qf, kf = q.float(), k.float()
    cos = cos.unsqueeze(0).unsqueeze(-2).float()
    sin = sin.unsqueeze(0).unsqueeze(-2).float()
    return ((qf * cos) + (rotate_half(qf) * sin)).to(q.dtype), \
           ((kf * cos) + (rotate_half(kf) * sin)).to(k.dtype)
```

`inv_freq` is non-persistent buffer derived from `theta=10000` and RoPE dim `head_dim // 2 = 52`. `class_pos_emb` is a learned parameter shaped `[1,52]`. Cos/sin can be cached per `(H/P, W/P, dtype, device)` except for the learned class row, which is a model parameter.

## 8. Preprocessing and input packing

The model consumes `pixel_values`, not raw images. Auto image processing maps MLCD to CLIP image processors.

Hub preprocessor configs:

- Resize enabled, center crop enabled, normalize enabled.
- Mean `[0.48145466, 0.4578275, 0.40821073]`.
- Std `[0.26862954, 0.26130258, 0.27577711]`.
- Resample `3` from config metadata.
- BigG sizes are shortest-edge resize and square crop to 224, 336, or 448.

Runtime tensor contract:

```text
pixel_values: float tensor [B,3,H,W], NCHW, normalized
H and W should be divisible by patch_size for DinoML admission
output last_hidden_state: [B, H/P * W/P + 1, hidden_size]
output pooler_output: [B, hidden_size]
```

No placeholder token stitching, masked scatter, modality token IDs, sequence packing descriptors, or text tokenizer coupling exists in this source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear

Source pattern:

```text
Conv2d(3 -> C, kernel=P, stride=P, padding=0, dilation=1, groups=1, bias=False)
flatten(2).transpose(1,2)
```

Replacement:

```text
WindowFlatten_NCHW_nonoverlap([B,3,H,W], P) -> GEMM/Linear(3*P*P -> C) -> [B,N,C]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`, `bias is None`.
- `H % P == 0` and `W % P == 0`.
- Preserve PyTorch Conv2d cross-correlation flatten order: output channel rows, input channel, kernel height, kernel width.

Weight transform:

```python
w_linear = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases: non-divisible image sizes, channel-last input without a controlled layout rewrite, future non-square patch sizes unless shape equations are generalized.

Parity test: compare Conv2d+flatten+transpose with WindowFlatten+GEMM for random NCHW tensors at 224/336/448 and odd rejected sizes.

### Rewrite: separate Q/K/V linears -> packed QKV projection

Source pattern: three independent `Linear(1664 -> 1664, bias=True)` from the same LayerNorm output.

Replacement: one GEMM producing `[B,S,3,16,104]`, split order `[q,k,v]`.

Preconditions:

- Same input tensor and dtype.
- No intervening op before Q/K/V split.
- Preserve distinct source weight names for loading and debugging.

Weight transform:

```python
w_qkv = cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
b_qkv = cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
```

Failure cases: quantized per-projection storage that cannot be concatenated, future KV grouping with different K/V projection widths, or debug mode requiring exact intermediate tensors.

### Rewrite: RoPE table cache

Source pattern: every forward creates aranges, stack, outer, indexed gather, concat, cos/sin.

Replacement: cache cos/sin by patch grid and dtype/device, with a dependency on `class_pos_emb`.

Preconditions:

- Static or bucketed image grids.
- `theta` and `inv_freq` unchanged.
- Cache invalidation if `class_pos_emb` changes; inference weights are immutable.

Failure cases: training, mutable parameters, arbitrary dynamic resolutions without bounded cache policy.

### Layout rewrite: NCHW patch region to channel-last local kernel

Candidate optimized layout: use NHWC only inside patch extraction/linear if DinoML owns the image preprocessing-to-patch region.

Required rewrites:

- Input semantic axes remain NCHW at ABI.
- Patch extraction must preserve PyTorch NCHW kernel flatten order or transform weights.
- Flatten and transpose consumers expect token-major `[B,N,C]`; after tokenization the encoder is layout-neutral rank-3.

Failure cases: exposing NHWC at public ABI without processor changes, mixing channel-last patch extraction with untransformed Conv2d weights.

## 10. Kernel fusion candidates

Highest priority:

- Patch Conv2d-as-GEMM: first large input op; simple non-overlap guard.
- LayerNorm: used before encoder, twice per block, and on pooled CLS.
- QKV packed projection + reshape: removes three GEMM launches per layer.
- RoPE + attention prefill: sequence lengths up to 1025 and head dim 104 make attention expensive.
- MLP GELU block: `Linear -> GELU -> Linear` dominates FLOPs alongside attention.

Medium priority:

- Residual add fused with post-projection outputs.
- CLS pooler slice + post LayerNorm.
- RoPE cos/sin grid cache for fixed image sizes.
- Attention backend selection between dense SDPA/FlashAttention and eager reference.

Lower priority:

- Output attentions materialization; useful for debug only.
- Dynamic rectangular image support beyond square official configs.
- `num_key_value_groups > 1` repeat-KV path, because official configs do not use it.

## 11. Runtime staging plan

1. Parse MLCD configs and route only `MLCDVisionModel` / `model_type: mlcd` or `mlcd_vision_model`; route CLIP-branded MLCD repos to CLIP audit.
2. Load weights for embeddings, `class_pos_emb`, LayerNorms, Q/K/V/O, and MLPs.
3. Implement patch embedding plus token construction parity for one bigG resolution.
4. Implement RoPE2D table generation and single attention block parity with eager dense attention.
5. Run one-layer and N-layer encoder parity with fp32.
6. Add full 48-layer bigG inference for `last_hidden_state` and `pooler_output`.
7. Add patch Conv2d->GEMM and packed QKV rewrites behind strict guards.
8. Add optimized attention/fusion kernels and resolution buckets for 224/336/448.

Initial stubs: output attentions can be omitted for production path; preprocessing can be external CPU pipeline; downstream LLM projector is out of scope.

## 12. Parity and validation plan

- Config routing tests: `mlcd` and `mlcd_vision_model` admitted; `clip_vision_model` MLCD-branded repos rejected/routed to CLIP.
- Preprocessor contract test: known PIL image produces NCHW tensor with expected size and normalization metadata.
- Patch embedding parity: random `[B,3,H,W]`, official sizes and rejected non-divisible sizes.
- RoPE parity: compare generated cos/sin and q/k after RoPE for grids 16x16, 24x24, 32x32.
- Attention block parity: eager noncausal attention with and without additive mask.
- Single encoder layer parity with fp32 tolerance around `rtol=1e-4, atol=1e-4`.
- Full model parity against Transformers for `DeepGlint-AI/mlcd-vit-bigG-patch14-448`; Transformers integration test checks `[1,1025,1664]` last hidden state and `[1,16,1025,1025]` attention.
- Reduced precision follow-up: fp16/bf16 tolerances likely `rtol=1e-2, atol=1e-2` after attention backend and LayerNorm accumulation policy are fixed.

## 13. Performance probes

- CPU preprocessing images/sec at 224, 336, 448.
- Patch embedding throughput and Conv2d-vs-GEMM rewrite comparison.
- Encoder block latency split into LayerNorm, QKV GEMM, RoPE, attention, output GEMM, MLP.
- Full encoder throughput by batch size for sequence lengths 257, 577, 1025.
- Attention backend comparison: eager reference, SDPA, FlashAttention-compatible path.
- Activation memory peak for output attentions disabled versus enabled.
- RoPE cache hit/miss overhead for fixed and mixed resolutions.
- Weight load time and memory footprint; safetensors index metadata reports about 7.37 GB for bigG checkpoints.

## 14. Skip/defer list

- Training, gradients, gradient checkpointing.
- Output attentions in optimized production path.
- Text tower, contrastive similarity, logit scale/bias.
- Autoregressive decode and KV cache.
- LLaVA/Qwen multimodal projector and language model.
- CLIP-routed MLCD-branded base/large checkpoints until CLIP vision audit is composed.
- Gated/private handling; no gated MLCD representative was observed.
- Arbitrary image sizes that are not patch-divisible.
- `num_key_value_groups > 1` unless a real checkpoint requires it.

## 15. Final implementation checklist

- [ ] Add MLCD config parser and route `mlcd` / `mlcd_vision_model`.
- [ ] Add admission guard for CLIP-routed MLCD-branded checkpoints.
- [ ] Load MLCD vision weights with Q/K/V/O, MLP, LayerNorm, patch embedding, CLS, and class RoPE params.
- [ ] Implement NCHW patch embedding and token construction.
- [ ] Implement MLCD 2D RoPE and q/k application.
- [ ] Implement noncausal dense MHA encoder block parity.
- [ ] Implement `last_hidden_state` and CLS `pooler_output`.
- [ ] Add patch Conv2d-to-GEMM rewrite with strict layout/divisibility guards.
- [ ] Add optional packed QKV rewrite preserving `[q,k,v]` split order.
- [ ] Add parity tests for 224/336/448 grids.
- [ ] Benchmark patch, attention, MLP, and full encoder throughput.
