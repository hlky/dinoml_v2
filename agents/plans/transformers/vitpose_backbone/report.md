# Transformers audit: vitpose_backbone

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `transformers`.

Model id: backbone source is `vitpose_backbone`; representative parent checkpoints use `VitPoseForPoseEstimation`, especially `usyd-community/vitpose-base-simple`, `usyd-community/vitpose-base`, `usyd-community/vitpose-plus-small`, `usyd-community/vitpose-plus-large`, and `usyd-community/vitpose-plus-huge`.

Config source: `transformers/src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py`; parent config and consumer path from `transformers/src/transformers/models/vitpose/configuration_vitpose.py` and `modeling_vitpose.py`; representative `config.json` and `preprocessor_config.json` fetched from public Hugging Face repos.

Source files inspected:

- [configuration_vitpose_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitpose_backbone/configuration_vitpose_backbone.py)
- [modeling_vitpose_backbone.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitpose_backbone/modeling_vitpose_backbone.py)
- [configuration_vitpose.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitpose/configuration_vitpose.py)
- [modeling_vitpose.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitpose/modeling_vitpose.py)
- [image_processing_vitpose.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/vitpose/image_processing_vitpose.py), only for the parent model input contract.
- [backbone_utils.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/backbone_utils.py), for `out_features` / `out_indices` behavior.

Any missing files or assumptions: no remote code is needed for the inspected public checkpoints. No model execution, imports, DinoML tests, or DinoML code edits were performed. Parent pose-estimation decoder and image postprocessing are referenced only to document the backbone feature-map ABI; this report owns the backbone encoder surface, not full VitPose heatmap parity.

## 2. High-level architecture

`vitpose_backbone` is a ViT-style image backbone with two source-declared changes versus ordinary ViT: padded patch embedding and optional dataset-indexed MoE MLP. It is encoder-only, non-causal, and returns selected hidden-state stages as backbone `feature_maps`.

Dataflow:

```text
CPU/image processor -> pixel_values [B,3,256,192] NCHW
-> padded Conv2d patch embedding
-> flatten + transpose to token sequence [B,192,C]
-> learned absolute position add
-> repeated pre-norm self-attention + MLP/MoE blocks
-> selected stage hidden states [B,192,C]
-> optional parent VitPose reshape to [B,C,16,12] for heatmap decoder
```

Stage decomposition:

- CPU/data pipeline: optional person-box affine crop, rescale, ImageNet normalization, output `pixel_values`.
- Backbone encoder: patch projection plus `num_hidden_layers` transformer blocks; independently cacheable only at the full-image/person-crop level, not autoregressive.
- Feature consumer: parent `VitPoseForPoseEstimation` takes the last selected sequence map, permutes to channels-first, reshapes to `[B,C,H_patch,W_patch]`, then runs either simple or classic heatmap decoder. That consumer is a separate integration target.

## 3. Important config dimensions

Effective source defaults come from `VitPoseBackboneConfig` when representative checkpoint JSON omits fields.

| Field | Default / observed values | Source impact |
| --- | --- | --- |
| `image_size` | `(256,192)` default; public swept configs omit it | Runtime shape guard requires exact input height/width. |
| `patch_size` | `(16,16)` default | Patch grid is `16 x 12 = 192` tokens for default image. |
| `num_channels` | `3` default | Patch Conv2d input channels. |
| `hidden_size` | base default `768`; plus-small `384`; plus-large `1024`; plus-huge `1280` | Token width, attention projection width, MLP input/output width. |
| `num_hidden_layers` | base/small default `12`; plus-large `24`; plus-huge `32` | Stage names are `stem`, `stage1` ... `stageN`. |
| `num_attention_heads` | base/small default `12`; plus-large/huge `16` | Head dim is `hidden_size / heads`. |
| `head_dim` | 32 for plus-small, 64 for base/large, 80 for huge | Inferred by source; hidden size must divide heads. |
| `mlp_ratio` | `4` default | MLP hidden width is `hidden_size * 4`. |
| `hidden_act` | `gelu` default | MLP activation via `ACT2FN`. |
| `qkv_bias` | `True` default | Separate Q, K, V Linear layers include bias by default. |
| `layer_norm_eps` | `1e-12` default | Both pre-attention and pre-MLP LayerNorms plus final feature LayerNorm. |
| `num_experts` | `1` base; `6` plus variants; tiny random `2` | Chooses normal MLP versus MoE MLP and requires `dataset_index` when > 1. |
| `part_features` | base configs set `0`; plus-small `96`; plus-base `192`; plus-large omitted but default `256`; plus-huge `320`; tiny `10` | MoE output tail width; `fc2` emits `hidden_size - part_features`. |
| `out_indices` / `out_features` | generally final stage, e.g. `[12]`, `[24]`, `[32]`; negative indices accepted by mixin | Selects which hidden states become backbone `feature_maps`. |
| `_attn_implementation` | config/framework controlled | Source dispatches through `ALL_ATTENTION_FUNCTIONS`, with eager fallback. |

Representative checkpoint sweep:

| Checkpoint | Config facts from JSON | Effective omitted defaults | Backbone shape |
| --- | --- | --- | --- |
| `hf-internal-testing/tiny-random-VitPoseForPoseEstimation` | `hidden_size=16`, `heads=2`, `num_experts=2`, `part_features=10`, `out_indices=[12]` | `layers=12`, `image_size=(256,192)`, `patch_size=(16,16)` | 192 tokens, head dim 8, MoE required. |
| `usyd-community/vitpose-base-simple` | `part_features=0`, `out_indices=[12]`, simple decoder in parent | `hidden_size=768`, `layers=12`, `heads=12`, `num_experts=1` | 192 tokens, head dim 64, normal MLP. |
| `usyd-community/vitpose-base` | same backbone fields as base-simple, classic decoder in parent | same as base | Backbone identical to base-simple for encoder audit. |
| `usyd-community/vitpose-plus-small` | `hidden_size=384`, `num_experts=6`, `part_features=96`, `out_indices=[12]` | `layers=12`, `heads=12` | 192 tokens, head dim 32, MoE required. |
| `usyd-community/vitpose-plus-base` | `num_experts=6`, `part_features=192`, `out_indices=[12]` | `hidden_size=768`, `layers=12`, `heads=12` | 192 tokens, head dim 64, MoE required. |
| `usyd-community/vitpose-plus-large` | `hidden_size=1024`, `heads=16`, `layers=24`, `num_experts=6`, `out_indices=[24]` | `part_features=256` by current source default | 192 tokens, head dim 64, MoE required. |
| `usyd-community/vitpose-plus-huge` | `hidden_size=1280`, `heads=16`, `layers=32`, `num_experts=6`, `part_features=320`, `out_indices=[32]` | default image/patch/dropout/norm fields | 192 tokens, head dim 80, MoE required. |

## 3a. Family variation traps

- Public configs often omit operator-significant defaults. DinoML should materialize effective configs after `VitPoseBackboneConfig.from_dict`, not trust raw JSON alone.
- The backbone `feature_maps` are token sequences `[B,S,C]`, not image-like maps. The parent pose model performs `permute(0,2,1)` and `reshape(B,C,H_patch,W_patch)`.
- Patch embedding uses `Conv2d(..., kernel_size=patch_size, stride=patch_size, padding=2)`. The positional table length is still computed from `image_size // patch_size`, so admission should guard that the actual convolution output grid matches the table length.
- MoE is selected by `num_experts > 1`, not by a separate architecture name. It requires `dataset_index` shape `[B]`; base variants ignore it.
- MoE source computes every expert and masks by equality to `dataset_index`; optimized lowering may replace this with gather/dispatch only under strict equivalence guards.
- Stage selection follows `BackboneConfigMixin`: default is final stage; negative indices are normalized; duplicates and out-of-order selections reject.
- `qkv_bias` is configurable, though representative public configs use the default `True`.
- Attention has no mask in the backbone source and no KV cache. `output_attentions` may force backends to expose attention probabilities, but first DinoML target can reject attentions unless needed.
- Source layout is NCHW through patch Conv2d. Channel-last/NHWC is only an optimization opportunity with axis rewrites around Conv2d, flatten, transpose, and the parent reshape.
- Parent configs contain `use_pretrained_backbone`, `use_timm_backbone`, and `backbone` fields, but the representative configs set no external backbone. This report does not cover timm or arbitrary AutoBackbone delegation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Exact input-shape validation for `pixel_values` `[B,3,H,W]`.
- Conv2d patch projection, NCHW, `kernel=(16,16)`, `stride=(16,16)`, `padding=2`, `groups=1`.
- `flatten(2)` from `[B,C,Hp,Wp]` to `[B,C,S]`.
- `transpose(1,2)` to `[B,S,C]`.
- Broadcast add for position embeddings `[1,S,C]` and global position row `[1,1,C]`.
- Stage filtering and tuple output metadata.
- Parent-consumer ABI: sequence `[B,S,C] -> permute -> reshape [B,C,Hp,Wp]` if integrating full VitPose.

Neural network primitives:

- LayerNorm over last dim `C`, epsilon `1e-12`.
- Linear Q/K/V: `C -> C`, bias controlled by `qkv_bias`.
- Linear attention output: `C -> C`, bias always present.
- MLP base path: `Linear(C -> 4C) -> GELU -> Linear(4C -> C)`.
- MoE path: `Linear(C -> 4C) -> GELU -> shared Linear(4C -> C-part_features)` plus `num_experts` tail Linear layers `4C -> part_features`, concat on last dim.
- Residual adds after attention and after MLP/MoE.
- Dropout modules exist but inference uses probability zero.

Attention primitives:

- Non-causal dense self-attention with Q/K/V shape `[B,H,S,D]`.
- Scale by `D**-0.5`, softmax over key dimension, dropout only in training.
- No attention mask, no ALiBi/RoPE/relative bias, no cache.
- SDPA/FlashAttention dispatch is source-supported through Transformers attention interface; eager math is the parity reference.

Position encoding:

- Learned absolute positional parameter `[1,S+1,C]`.
- Runtime add is `patch_tokens + pos[:,1:] + pos[:,:1]`; no class token participates in the sequence.

Preprocessing-coupled ops:

- Parent processor can crop each person box via affine transform, rescale by `1/255`, normalize by ImageNet mean/std, and emit `pixel_values` in NCHW.
- The backbone itself only consumes already prepared `pixel_values`.

MoE / dataset-index ops:

- `dataset_index.view(-1,1,1)` and equality comparisons to expert ids.
- Broadcast multiply of expert outputs by boolean mask.
- Sum across all expert outputs, then concatenate with shared output.
- Admission should validate `dataset_index` integer dtype, rank 1, batch length equal to `B`, and values in `[0,num_experts)`.

## 5. Layer/block breakdown

Patch and embedding:

```text
pixel_values: [B,3,256,192] NCHW
x = Conv2d(3 -> C, kernel=16x16, stride=16x16, padding=2)(pixel_values)
x: [B,C,16,12] for default size
x = flatten spatial -> [B,C,192]
x = transpose -> [B,192,C]
x = x + position_embeddings[:,1:] + position_embeddings[:,:1]
```

Transformer block, repeated `N` times:

```text
residual = x
y = LayerNorm(C, eps=1e-12)(x)
q = Linear(C -> C, bias=qkv_bias)(y).view(B,S,H,D).transpose(1,2)
k = Linear(C -> C, bias=qkv_bias)(y).view(B,S,H,D).transpose(1,2)
v = Linear(C -> C, bias=qkv_bias)(y).view(B,S,H,D).transpose(1,2)
a = dense_noncausal_attention(q,k,v, no_mask)
y = Linear(C -> C)(a.reshape(B,S,C))
x = residual + y
residual = x
y = LayerNorm(C, eps=1e-12)(x)
if num_experts == 1:
    y = Linear(4C -> C)(GELU(Linear(C -> 4C)(y)))
else:
    h = GELU(Linear(C -> 4C)(y))
    shared = Linear(4C -> C - part_features)(h)
    expert_tail = sum_i Linear_i(4C -> part_features)(h) * (dataset_index == i)
    y = concat(shared, expert_tail, dim=-1)
x = residual + y
```

Feature output:

```text
hidden_states = embedding output plus each block output, captured by BackboneMixin hooks
for stage, hidden_state in zip(stage_names, hidden_states):
    if stage in out_features:
        feature_maps.append(final_LayerNorm(hidden_state))
```

## 6. Attention requirements

The primary target requires encoder self-attention only.

- Causal: no.
- Cross-attention: no.
- MHA/MQA/GQA: standard MHA; Q, K, V all have `num_attention_heads` heads.
- Head counts: 12 heads for small/base defaults; 16 for plus-large/huge.
- Head dim: 32, 64, or 80 in representative configs.
- Query/key/value sequence length: square attention over `S=192` default patch tokens.
- Masking: source passes `None`; no padding mask is applied inside the backbone.
- Packed/varlen support: none in source.
- Sliding/local/block attention: none.
- Position interactions: learned absolute add before all layers; no attention-time position math.
- KV cache: not applicable. This is not a generation model.
- FlashAttention/SDPA compatibility: source advertises `_supports_sdpa` and `_supports_flash_attn`; parity should still preserve eager ordering `QK^T * scale -> mask add if any -> softmax -> dropout -> AV`.

## 7. Position encoding and custom math

Position encoding is learned and static for the configured image/patch grid. The extra row behaves like a learned global offset, not a class token in the sequence.

```python
def vitpose_position_add(patch_tokens, pos):
    # patch_tokens: [B, S, C], pos: [1, S + 1, C]
    return patch_tokens + pos[:, 1:] + pos[:, :1]
```

Precompute opportunities:

- `pos[:,1:] + pos[:,:1]` can be folded into one `[1,S,C]` constant after config shape is fixed.
- Patch Conv2d output grid and positional-table length should be verified at load/admission time.

## 8. Preprocessing and input packing

Backbone runtime input is just `pixel_values`. The parent image processor is relevant when targeting full pose estimation:

- Public preprocessor configs use `size={"height":256,"width":192}`, `do_affine_transform=true`, `do_rescale=true`, `rescale_factor=1/255`, ImageNet mean/std, and `normalize_factor=200.0`.
- Processor accepts images plus boxes; each box can create one person crop, so model batch may be number of person boxes rather than number of source images.
- Output layout entering the model is NCHW `[B,3,256,192]`.
- Postprocessing heatmaps back to original image coordinates is CPU/data-pipeline work for the full parent model, not part of `vitpose_backbone`.

No tokenizer, placeholder scatter, generation metadata, or packed sequence descriptors are involved.

## 9. Graph rewrite / lowering opportunities

### Rewrite: padded patch Conv2d to GEMM

Source pattern:

```text
Conv2d(Cin -> C, kernel=P, stride=P, padding=2) -> flatten(2) -> transpose(1,2)
```

Replacement:

```text
PaddedWindowExtract(NCHW, pad=2, window=P, stride=P) -> GEMM(weight_flat.T) -> bias -> sequence [B,S,C]
```

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 2`, `dilation == 1`, `groups == 1`.
- Input height/width exactly match config.
- Extracted grid `(Hp,Wp)` must equal `(image_h // patch_h, image_w // patch_w)` because position embeddings are sized that way.
- NCHW flatten order must match PyTorch Conv2d spatial order followed by `flatten(2).transpose(1,2)`.

Weight transform:

```python
w_flat = conv.weight.reshape(out_channels, in_channels * patch_h * patch_w)
```

Failure cases:

- Non-default image/patch sizes where padding changes the convolution grid away from `image_size // patch_size`.
- Any future grouped/dilated patch projection.
- NHWC layout pass without matching window-extract and weight-layout rewrite.

Parity test sketch: compare patch embedding output before position add for random NCHW inputs against PyTorch Conv2d for base and plus-small widths.

### Rewrite: fold absolute position pair

Source pattern:

```text
x + position_embeddings[:,1:] + position_embeddings[:,:1]
```

Replacement:

```text
x + folded_position_embeddings
```

Preconditions: fixed `S`, fixed loaded position table, no training updates.

Failure cases: dynamic image sizes, resized/interpolated position embeddings, or any future class-token path.

### Rewrite: MoE masked-all-experts to indexed expert dispatch

Source pattern:

```text
for i in experts:
    out += expert_i(h) * (dataset_index == i)
```

Replacement:

```text
group rows by dataset_index -> run selected expert(s) -> scatter back -> concat(shared, tail)
```

Preconditions:

- Inference only.
- `dataset_index` rank `[B]`, values are valid integers.
- Selection is per batch item and broadcast across all sequence tokens.
- Scatter preserves original batch order.

Failure cases: invalid indices, non-integer `dataset_index`, output attentions/debug paths expecting eager expert side effects, or backend unable to preserve deterministic ordering.

### Layout opportunity: local NCHW to NHWC patch region

Candidate optimized region is only `pixel_values -> patch Conv2d -> flatten/transpose`. Axis rewrites would need:

- Input `[B,H,W,C]` instead of `[B,C,H,W]`.
- Conv weight transform from `[Cout,Cin,Kh,Kw]` to backend-specific NHWC kernel layout.
- Flatten token order still row-major over `Hp,Wp`.
- Parent decoder reshape remains NCHW unless the whole parent head is also translated.

Use a no-layout-translation guard at the public backbone output unless all consumers agree on sequence or image-map layout.

## 10. Kernel fusion candidates

Highest priority:

- Padded patch embedding Conv2d or GEMM lowering. It is the only convolution in the backbone and has a nonstandard padding contract.
- LayerNorm + QKV projection region. Every block starts with pre-attention LayerNorm followed by three same-input Linear projections.
- Dense noncausal attention for fixed small sequence length `S=192`. This is ideal for a compact encoder attention path, with no masks or cache complexity.
- MLP GEMM + GELU + GEMM for normal base variants.

Medium priority:

- MoE expert dispatch for plus variants. Naive all-expert execution multiplies compute by `num_experts`; grouping by `dataset_index` is high value for plus checkpoints.
- Final feature LayerNorm fused with selected stage output materialization.
- Position add folded with patch embedding output write.

Lower priority:

- Attention probability materialization. Needed only for `output_attentions=True`.
- Multi-output stage capture optimization when users request several `out_indices`.
- Parent pose decoder fusions; useful for full VitPose, out of scope for this backbone-only audit.

## 11. Runtime staging plan

Stage 1: parse effective `VitPoseBackboneConfig`, including omitted defaults, stage names, and selected out features.

Stage 2: load weights and run patch embedding plus position-add parity on fixed `[B,3,256,192]` NCHW inputs.

Stage 3: implement one normal transformer block with dense attention and normal MLP; validate base config single-layer and full-encoder hidden states.

Stage 4: implement selected feature-map ABI: return tuple of `[B,S,C]` stage outputs with final LayerNorm applied only to selected stages.

Stage 5: add MoE MLP admission and parity for plus variants with `dataset_index`; initially allow naive all-expert lowering, then optimize dispatch.

Stage 6: connect parent VitPose consumer reshape only after backbone sequence parity is stable.

Stage 7: add optimized kernels/fusions: patch GEMM, LayerNorm+QKV, attention, MLP, MoE dispatch.

Initial stubs: dropout can be omitted in inference; `output_attentions=True` can be rejected or deferred; parent heatmap decoder can be stubbed for backbone-only integration.

## 12. Parity and validation plan

- Config tests: raw JSON omissions materialize expected source defaults for base, plus-base, and plus-large.
- Patch embedding tests: Conv2d with `padding=2` on `[1,3,256,192]` produces `[1,192,C]`; reject wrong input sizes.
- Position tests: folded position add equals two-add source expression.
- Single-block fp32 parity for base dimensions on small random tensors.
- Full backbone parity for a tiny/random config and one representative base config; compare selected feature map after final LayerNorm.
- MoE parity: plus-small/tiny config with `dataset_index` covering multiple expert ids; verify batch-wise expert selection and invalid/missing index rejection.
- Stage selection tests: `out_indices=[-1]`, explicit `[12]`, multiple ordered stages, duplicate/out-of-order rejection.
- Parent ABI smoke: selected feature `[B,192,C]` reshapes to `[B,C,16,12]` exactly.

Recommended tolerances: fp32 `rtol=1e-4, atol=1e-5` for unfused math; fp16/bf16 tolerances should be looser after attention and GELU, with per-block and full-encoder thresholds measured separately.

## 13. Performance probes

- Processor throughput: images/second and boxes/second for affine crop plus normalization, separate from GPU backbone.
- Patch embedding throughput for NCHW Conv2d versus GEMM/window lowering.
- Encoder throughput by variant: base, plus-small, plus-large, plus-huge.
- Attention backend comparison at `S=192` across batch sizes and hidden widths.
- MLP versus attention time split per layer.
- MoE naive all-experts versus grouped expert dispatch, sweeping number of dataset ids present in a batch.
- Multi-stage output overhead for one versus several `out_indices`.
- Parent reshape plus decoder overhead only when full VitPose is staged.

## 14. Skip/defer list

- Training, losses, labels, and gradient checkpointing.
- Dropout behavior in training mode.
- Parent pose heatmap decoder and postprocessing for the first backbone-only target.
- `output_attentions=True` materialization unless explicitly required.
- timm or arbitrary external AutoBackbone delegation.
- Dynamic image sizes or position interpolation; current source checks exact `image_size`.
- General MoE router/top-k routing; this family only uses dataset-index expert selection.
- Quantization/packed weight formats; no source-coupled quantized storage is present.

## 15. Final implementation checklist

- [ ] Parse effective `VitPoseBackboneConfig` with source defaults.
- [ ] Validate `pixel_values` NCHW shape against `image_size`.
- [ ] Implement padded Conv2d patch embedding with `padding=2`.
- [ ] Implement flatten/transpose tokenization and absolute position add.
- [ ] Implement dense noncausal MHA with no mask/cache.
- [ ] Implement base MLP path.
- [ ] Implement MoE MLP path gated by `dataset_index`.
- [ ] Implement selected stage capture and final feature LayerNorm.
- [ ] Add admission guards for `out_indices`, `dataset_index`, qkv divisibility, and patch grid length.
- [ ] Add single-block, full-backbone, and MoE parity tests.
- [ ] Add patch Conv/GEMM rewrite behind strict padding/grid/layout guards.
- [ ] Benchmark patch embedding, attention, MLP, and MoE dispatch separately.
