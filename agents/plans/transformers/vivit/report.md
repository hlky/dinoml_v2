# Vivit Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: google/vivit-b-16x2, google/vivit-b-16x2-kinetics400
Config source: Hugging Face raw config/preprocessor JSON plus current VivitConfig defaults
Source files inspected:
- X:/H/transformers/src/transformers/models/vivit/configuration_vivit.py
- X:/H/transformers/src/transformers/models/vivit/modeling_vivit.py
- X:/H/transformers/src/transformers/models/vivit/modular_vivit.py
- X:/H/transformers/src/transformers/models/vivit/image_processing_vivit.py
- X:/H/transformers/src/transformers/models/vivit/convert_vivit_flax_to_pytorch.py
- X:/H/transformers/src/transformers/masking_utils.py
Any missing files or assumptions: no video_processing_vivit.py exists. modeling_vivit.py is generated from modular_vivit.py; future source edits should target modular_vivit.py, but runtime import behavior comes from generated modeling_vivit.py.
```

Snapshots:

- `_sources/source_snippets.md`
- `_sources/hf_config_summaries.md`

No gated official checkpoints were encountered. One non-official search result, [juliendenize/COMEDIAN-ViViT-tiny](https://huggingface.co/juliendenize/COMEDIAN-ViViT-tiny), returned 404 for `config.json` and is excluded from source-derived facts.

## 2. High-level architecture

ViViT in current Transformers is an encoder-only video classification model:

```text
decoded/sampled video frames -> per-frame image processor -> pixel_values[B,T,C,H,W]
-> 3D tubelet Conv3d patch embedding -> CLS + absolute position embeddings
-> noncausal Transformer encoder -> final LayerNorm -> CLS classifier logits
```

Primary DinoML runtime target: `VivitForVideoClassification` logits. `VivitModel` base encoder and optional pooler are useful intermediate targets. Training losses are optional/deferred.

Important scope note: this source does not implement the ViViT paper's factorized spatial/temporal encoder or factorized self-attention variants. It implements a single dense noncausal MHA encoder over all tubelet tokens plus CLS.

## 3. Important config dimensions

Current `VivitConfig` defaults:

| Field | Value | Runtime effect |
| --- | --- | --- |
| `image_size` | 224 | required H/W unless `interpolate_pos_encoding=True` |
| `num_frames` | 32 | used to size learned position table |
| `tubelet_size` | `[2,16,16]` | Conv3d kernel/stride over T,H,W |
| `num_channels` | 3 | Conv3d input channels |
| `hidden_size` | 768 | token width |
| `num_hidden_layers` | 12 | encoder blocks |
| `num_attention_heads` | 12 | MHA heads |
| `head_dim` | default `hidden_size // heads = 64` | q/k/v head width unless config adds `head_dim` |
| `intermediate_size` | 3072 | MLP expansion |
| `hidden_act` | `gelu_fast` | MLP activation |
| `qkv_bias` | `True` | q/k/v projection bias |
| `layer_norm_eps` | `1e-6` | LayerNorm epsilon |
| `pooler_output_size` | default hidden size | base model pooler only |
| `pooler_act` | `tanh` | base model pooler only |

Default token math: `(32/2) * (224/16) * (224/16) = 3136` tubelets, then `3137` tokens with CLS.

Representative config sweep:

| Model id | Official? | Body shape | Frames/image | Tubelet | Labels/head | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `google/vivit-b-16x2` | yes | 768, 12 layers, 12 heads | 32, 224 | 2x16x16 | 400 | current field names; Kinetics labels |
| `google/vivit-b-16x2-kinetics400` | yes | 768, 12 layers, 12 heads | legacy `video_size=[32,224,224]` | 2x16x16 | 400 | omits current `num_frames`/`image_size`; defaults fill them |
| `NiiCole/...ucf101-subset` | community fine-tune | same | 32, 224 | 2x16x16 | 10 | classifier-only variation |
| `prathameshdalal/...UCF-Crime` | community fine-tune | same | 32, 224 | 2x16x16 | 14 | classifier-only variation |
| `Arekku21/...MSL` | community fine-tune | same | 32, 224 | 2x16x16 | 3 | classifier-only variation |

## 3a. Family variation traps

- Historical configs may contain `video_size` instead of current `num_frames`/`image_size`. DinoML should normalize only when it matches `[T,H,W]` and current defaults/checkpoint weights are consistent.
- Historical architecture spelling `ViViTForVideoClassification` appears in one official JSON, while current class export is `VivitForVideoClassification`.
- Processor configs differ: `google/vivit-b-16x2` has `do_normalize=false` and historical `do_zero_centering=true`; current processor uses `offset`, not `do_zero_centering`.
- The source supports `getattr(config, "head_dim", hidden_size // heads)`, so reject or explicitly support configs where `head_dim * num_attention_heads != hidden_size`.
- `qkv_bias` controls q/k/v bias, but output projection bias is always present.
- Positional interpolation is spatial-only and assumes the patch position count excluding CLS is a square spatial grid. It does not resize the temporal tubelet axis.
- Source layout is semantic `B,T,C,H,W` input, transposed to `B,C,T,H,W` for Conv3d. Treat NTHWC/channel-last as a guarded layout optimization only.
- Any factorized encoder/attention fields from papers or remote variants are unsupported by this in-library source unless a separate remote-code audit proves them.

## 4. Operator coverage checklist

Tensor/layout ops:

- Rank/shape guards for `pixel_values[B,T,C,H,W]`.
- `transpose(1,2)` from `B,T,C,H,W` to `B,C,T,H,W`.
- Conv3d output flatten from `[B,Hid,T',H',W']` to `[B,T'*H'*W',Hid]`.
- CLS parameter expand, `cat(dim=1)`, position add, dropout as inference identity.
- Slicing `sequence_output[:,0,:]`.

Neural primitives:

- `Conv3d(C -> hidden_size, kernel=stride=tubelet_size, bias=True)`.
- LayerNorm over hidden with eps `1e-6`.
- Linear q/k/v: `Linear(768 -> 768)` for defaults, optional bias.
- Linear output: `Linear(768 -> 768)`, bias always true.
- MLP: `Linear(768 -> 3072) -> gelu_fast -> Linear(3072 -> 768)`.
- Classifier: `Linear(768 -> num_labels)` or Identity when `num_labels <= 0`.
- Base pooler optional: `take CLS -> Linear(768 -> pooler_output_size) -> tanh`.

Attention primitives:

- Dense bidirectional self-attention over full sequence.
- MHA with equal Q/K/V head counts in observed configs.
- Optional additive/padding attention mask through `create_bidirectional_mask`.
- SDPA/Flash-compatible noncausal attention path; eager parity requires fp32 softmax then cast back.

Preprocessing-coupled ops:

- Frame resize by shortest edge or explicit H/W.
- Center crop.
- Rescale with optional zero-centering (`image * scale - 1` when `offset=True`).
- Optional mean/std normalize.
- Output channel-first per frame, stacked as `pixel_values[B,T,C,H,W]`.

No generation/cache, RoPE, relative-bias, quantized weight, packed-varlen, recurrent-state, codebook, or multimodal scatter operators are required.

## 5. Layer/block breakdown

Embedding:

```text
pixel_values[B,T,C,H,W]
-> transpose to [B,C,T,H,W]
-> Conv3d(C=3 -> H=768, kernel=stride=[2,16,16])
-> [B,768,16,14,14]
-> flatten spatial-temporal axes and transpose -> [B,3136,768]
-> concat CLS -> [B,3137,768]
-> add learned position table [1,3137,768]
```

Encoder block, repeated 12 times for defaults:

```text
residual = x
x = LayerNorm(x)
q,k,v = Linear(768 -> 768, bias=qkv_bias)
q,k,v = view [B,S,12,64] -> transpose [B,12,S,64]
x = noncausal dense attention(q,k,v, optional mask)
x = Linear(768 -> 768, bias=True)
x = residual + x
residual = x
x = LayerNorm(x)
x = Linear(768 -> 3072) -> gelu_fast -> Linear(3072 -> 768)
x = residual + x
```

Head:

```text
sequence = final LayerNorm(x)
logits = Linear(sequence[:,0,:] -> num_labels)
```

## 6. Attention requirements

Vivit uses encoder self-attention only:

- Noncausal/bidirectional.
- Self-attention, no cross-attention.
- MHA, not MQA/GQA in observed configs.
- Defaults: 12 heads, head dim 64, q/k/v width 768.
- Query length equals key/value length: `S = 1 + (T/tube_t)*(H/tube_h)*(W/tube_w)`.
- Padding mask, if supplied, is converted by `create_bidirectional_mask`; no model-generated temporal mask is required.
- No sliding window, local attention, ALiBi, RoPE, relative bias, KV cache, packed prefill/decode, or autoregressive generation.
- FlashAttention/SDPA compatibility is straightforward noncausal encoder attention, provided masks and dtype/backend constraints match Transformers behavior.

## 7. Position encoding and custom math

Learned absolute position embeddings are added after CLS concatenation. For normal 224x224 inference the table is direct `[1,3137,768]`.

Spatial interpolation path:

```python
def vivit_interpolate_pos(pos, height, width, patch_h, patch_w):
    cls = pos[:, :1]
    patch = pos[:, 1:]
    dim = patch.shape[-1]
    side = int((patch.shape[1]) ** 0.5)
    patch = patch.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    patch = bicubic_interpolate(
        patch,
        size=(height // patch_h, width // patch_w),
        align_corners=False,
    )
    patch = patch.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return concat([cls, patch], dim=1)
```

This is axis-sensitive and should be deferred or guarded for first integration. It assumes a square 2D position grid and ignores temporal patch count, which is source behavior but a trap for non-default temporal resolutions.

## 8. Preprocessing and input packing

The processor is CPU/data-pipeline work for first DinoML integration. It accepts raw frames as PIL/numpy/torch images and returns `pixel_values`.

Current class defaults:

```text
resize shortest_edge=256
center_crop 224x224
rescale_factor=1/127.5
offset=True, so image = image * scale - 1
normalize=True with ImageNet standard mean/std unless config overrides
output data_format=channels_first
```

Official processor configs override this:

- `google/vivit-b-16x2`: shortest edge 256, crop 224, rescale 1/127.5, `do_normalize=false`, historical `do_zero_centering=true`.
- `google/vivit-b-16x2-kinetics400`: shortest edge 224, crop 224, rescale 1/127.5, normalize with mean/std `[0.5,0.5,0.5]`.

Clip sampling and video decode are not implemented by `VivitImageProcessor`; examples use external PyAV sampling. DinoML should treat frame selection/decode as caller or pipeline responsibility and require exactly the configured frame count for first parity.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv3d tubelet -> Linear

Source pattern:

```text
B,T,C,H,W -> transpose -> Conv3d(C,Hid,kernel=tubelet,stride=tubelet) -> flatten -> token sequence
```

Replacement:

```text
TubeletWindowFlatten(B,T,H,W,C, kt,kh,kw) -> GEMM(flat_window, weight_flat.T) -> BiasAdd -> token sequence
```

Preconditions:

- `kernel_size == stride == tubelet_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- `T,H,W` divisible by tubelet dimensions.
- Source flatten order preserved: Conv3d output order is temporal, then height, then width after `flatten(2).transpose(1,2)`.
- Weight transform from Conv3d `[hidden, C, kt, kh, kw]` to `[hidden, C*kt*kh*kw]`, with window flatten matching PyTorch Conv3d memory order.

Failure cases: non-divisible clip/image sizes, any padding/dilation/groups change, or layout pass that changes temporal/spatial order without rewriting flatten.

Parity test: compare patch embedding output against PyTorch Conv3d for random fp32/fp16 inputs and several batch sizes.

### Rewrite: separate q/k/v projections -> packed QKV GEMM

Preconditions:

- Same input `hidden_states`.
- Same input/output widths and same bias enablement.
- Split order must be `[q, k, v]`, each width `num_heads * head_dim`.

Replacement: one `Linear(768 -> 2304)` followed by split into q/k/v.

Failure cases: configs with distinct q/k/v widths or mixed bias flags. Current source does not expose those, but guards should stay explicit.

### Layout opportunity: NTHWC tubelet path

Candidate optimized path: keep preprocessed video in `B,T,H,W,C` and lower tubelet extraction directly to NHWC/NTHWC-friendly windows or implicit GEMM. This is an optimization only; semantic graph remains `B,T,C,H,W`.

Required axis rewrites:

- Source `transpose(1,2)` becomes either no-op in internal layout or an explicit boundary conversion.
- Conv3d channel axis changes from dim 1 in `B,C,T,H,W` to last dim in NTHWC.
- Flatten order must still emit tokens ordered by temporal, height, width positions.

Guard with a local layout region ending before the token sequence enters LayerNorm/Linear/attention.

## 10. Kernel fusion candidates

Highest priority:

- Tubelet Conv3d as implicit GEMM or window-flatten GEMM. It is the main video-specific operator.
- LayerNorm + QKV projection setup. Same shape across all blocks and sequence length is large (`3137`).
- Dense noncausal attention using SDPA/Flash-compatible backend. Attention is over long video token sequences.
- MLP GEMM + `gelu_fast` + GEMM, with activation fusion where available.

Medium priority:

- Packed q/k/v projection and q/k/v reshape/transposes.
- Final CLS gather + classifier GEMM for small heads.
- Position add + dropout-elision + residual add folding in inference graphs.

Lower priority:

- Spatial positional interpolation bicubic path; useful for high-res fine-tuning parity, but not first fixed-shape inference.
- Base pooler fusion; classification head does not use it.

## 11. Runtime staging plan

Stage 1: parse config, normalize historical `video_size` to `num_frames`/`image_size` only for known Vivit configs, load weights, and run tubelet embedding parity.

Stage 2: run one encoder block parity with eager attention and fp32.

Stage 3: run full `VivitModel` encoder parity at fixed `B,T,C,H,W = B,32,3,224,224` with direct learned positions.

Stage 4: add `VivitForVideoClassification` logits parity for official Google checkpoints and one small-head fine-tune.

Stage 5: enable optimized attention and packed QKV rewrites under strict shape/head/bias guards.

Stage 6: add guarded tubelet Conv3d-to-GEMM or NTHWC layout pass, preserving token order.

Stage 7: optionally support `interpolate_pos_encoding=True` for spatial resize parity.

## 12. Parity and validation plan

- Processor parity on sampled frames: compare output `pixel_values` for both official processor configs, especially `do_normalize` and historical zero-centering behavior.
- Tubelet embedding random tensor parity against PyTorch Conv3d for fp32 and fp16.
- Position add parity for fixed 224x224 and, separately, interpolation parity for larger H/W.
- Single block parity with random hidden states and optional attention masks.
- Full encoder parity for `google/vivit-b-16x2` and `google/vivit-b-16x2-kinetics400`.
- Classification logits parity for official checkpoint and a smaller-label fine-tune.
- Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 attention paths need backend-specific tolerances such as `atol=5e-2, rtol=5e-2` until fused kernels are calibrated.

## 13. Performance probes

- CPU preprocessing throughput: decode/sample excluded vs included.
- Tubelet embedding throughput over batch sizes 1, 2, 4, 8.
- Encoder-only latency/throughput with sequence length 3137.
- Attention backend comparison: eager, SDPA, Flash-compatible backend.
- Batch sweep with fixed 32x224x224 clips.
- Resolution sweep only when interpolation is supported.
- Memory probe for attention activations at `S=3137`; this dominates much more than the small classifier head.
- Layout probe comparing source NCTHW Conv3d path with guarded NTHWC/window-GEMM path.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Video decode and temporal frame sampling policy.
- Positional interpolation for non-224 spatial sizes in the first fixed-shape target.
- Factorized encoder/attention variants from the paper; not implemented in current source.
- Generation, KV cache, beam search, tokenizers.
- Quantization and packed weights.
- Multi-GPU or tensor parallel execution.

## 15. Final implementation checklist

- [ ] Parse `VivitConfig`, including safe normalization for historical `video_size`.
- [ ] Parse `VivitImageProcessor` configs and freeze first supported preprocessing contract.
- [ ] Load tubelet Conv3d, CLS, position, LayerNorm, Linear, and classifier weights.
- [ ] Implement `pixel_values[B,T,C,H,W]` shape guards.
- [ ] Implement tubelet Conv3d embedding with exact token order.
- [ ] Implement CLS concat and learned position add.
- [ ] Implement noncausal MHA with optional bidirectional mask.
- [ ] Implement `gelu_fast` MLP blocks and final LayerNorm.
- [ ] Implement CLS classifier head.
- [ ] Add tubelet embedding parity tests.
- [ ] Add one-block and full-encoder parity tests.
- [ ] Add official checkpoint logits parity.
- [ ] Add packed QKV rewrite with `[q,k,v]` split order guard.
- [ ] Add guarded Conv3d tubelet-to-GEMM rewrite.
- [ ] Benchmark preprocessing, tubelet embedding, attention backend, and full encoder throughput.
