# DinoML v1 Op Porting Checklist

This checklist maps ops from `/workspace/dinoml/src/dinoml/compiler/ops` to the
DinoML v2 port. It is organized by implementation family, not by every v1 Python
class, so porting can happen through reusable kernels and registrations instead
of one-off clones.

Status markers:

- [x] Available in v2 MVP.
- [ ] Not ported yet.

For each public op or family, add schema, frontend binding, CPU binding, CUDA
binding, profiler binding when relevant, shape/type inference, and tests. Prefer
one semantic v2 op plus backend variants over separate user-facing registrations
for every v1 fused/layout specialization.

## Local PyTorch Reference

Brief local scan: `torch 2.9.1+cu128` is installed. Broad support exists in
`torch`, `torch.nn`, and `torch.nn.functional` for common tensor math, matmul,
BMM, reductions, indexing, conv/pool/pad/upsample, normalization, activations,
and scaled-dot-product attention.

Broad gaps versus core PyTorch are the v1-specific categories: layout-fused
GEMM/BMM epilogues, jagged/ragged conversion semantics, positional/model helper
ops, FIR resampling helpers, NMS/ROI detection helpers, NHWC/NDHWC packing
helpers, and exact flash/memory-efficient attention variants.

## V2 MVP Surface

- [x] Dense elementwise math surface - public frontend ops lower into
  model-generated `fused_elementwise` kernels.
- [x] `gelu` - v2 native frontend op, tanh approximation.
- [x] Runtime shape buffers for dynamic shape validation and generic
  fused-elementwise broadcasting.

## Common Primitives

These should be reusable building blocks. They generally map to `torch` or
`torch.nn.functional` semantics for reference tests.

### Elementwise, Activations, and Scalar Math

- [x] `elementwise`: initial dense coverage for arithmetic,
  min/max, trig/log/exp/sqrt, activations, `nan_to_num`,
  `clamp_nan_to_num`, `pow`, `floor_div`, and `floor`. Remaining v1 parity
  work: jagged broadcasting, broader CPU/vector accessors, scalar dtype
  promotion, and exhaustive edge-case tests.
- [x] `fused_elementwise`: connected registered unary/binary elementwise
  subgraphs lower to model-generated CPU/CUDA kernels that call
  `dinoml::math::<name>` helpers. CPU and CUDA support float32, float16, and
  bfloat16 storage; CUDA has optional fp32 accumulation and vectorized dense
  paths, while CPU reduced precision always computes in fp32 for now. Runtime
  shape buffers support generic broadcasting. Multi-output same-shape metadata
  is represented; broader tests and v1-style jagged codegen remain.
- [ ] `int_elementwise`: `ADD`, `SUB`, `MUL`, `DIV` for symbolic integer math.
- [x] Public math helpers: `tanh`, `cos`, `sin`, `sign`, `abs`, `log`, `log1p`,
  `exp`, `sqrt`, `max`, `min`, `sigmoid`, `leaky_relu`, `hardtanh`, `relu`,
  `silu`, `nan_to_num`, `pow`, `fast_gelu`, `softplus`, `elu`, `softsign`,
  `floor_div`, `celu`, `floor`, plus `sub`, `mul`, `div`, and
  `clamp_nan_to_num`.

Library hints: CPU can use scalar loops first, then `std::simd` or xsimd for
vector paths. CUDA/HIP elementwise kernels are usually simpler than library
calls. GEMM epilogue activations should be expressed through CUTLASS or CK
epilogues where possible.

### Views, Layout, Shape, and Selection

- [x] View-only: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `identity`.
  These public frontend ops now emit `metadata.views` shape aliases with no
  compute nodes; lowering/runtime consume the validated
  `metadata.memory_plan.views` form and materialize public alias outputs into ABI
  output buffers. Current limits: view-of-view aliases are rejected, reshape only
  accepts static input shapes, flatten only accepts static dimensions in the
  flattened range, scalar view tensors are not exposed yet, and layout-changing
  `permute`/`transpose` remain unported.
- [ ] Symbolic shape/container helpers: `size`, `getitem`, `tuple_construct`,
  `list_construct`. These should remain frontend/IR helpers unless they produce
  tensors with explicit runtime storage.
- [ ] Layout: `permute`, `transpose`, `permute021`, `permute0213`,
  `permute102`, `permute210`, `pixel_shuffle`, `pixel_unshuffle`.
- [ ] Creation/shape values: `arange`, `full`, `randn`, `meshgrid`, `cast`.
- [ ] Selection/scatter: `dynamic_slice`, `slice_scatter`,
  `slice_reshape_scatter`, `gather`, `batch_gather`, `index_select`,
  `masked_select`, `where`, `topk`, `argmax`.
- [ ] Collections/broadcasting: `chunk`, `split`, `stack`, `concatenate`,
  `expand`, `repeat_interleave`, `flip`.
- [ ] Relational ops: `eq`, `ge`, `gt`, `le`, `lt`, `ne`.
- [ ] Tensor helpers that should not become separate kernel families unless
  profiling proves it: `concatenate_tanh`, `concatenate_fast`,
  `expand_static_shape`.

Library hints: most are metadata-only or simple copies. `topk`/sort-like paths
can use CUB on CUDA; CPU paths can start with standard library algorithms and
add xsimd or `std::simd` only where measurable.

### Reductions and Softmax

- [x] Basic reductions: `reduce_max`, `reduce_mean`, `reduce_min`,
  `reduce_sum` for dense contiguous float32 tensors over a positive static last
  dimension, with negative dim normalization and `keepdim`. CPU and CUDA use
  generated row reductions and validate against NumPy/Torch-style semantics.
  CUDA includes a warp-per-row path for static reductions up to `K=1024` and a
  shared-memory fallback for larger reductions. Remaining parity work: non-last
  dimensions, multi-axis rejection that mirrors v1 more closely, optional output
  dtype, fp16/bf16 accumulation policy, v1 CUTLASS/`reduce_3d` strategy,
  profiler selection, and reduction ops `var`/`vector_norm`.
- [ ] `var`, `vector_norm`.
- [x] `softmax`: initial public `dml.ops.softmax(x, dim=-1)` port for dense
  contiguous float32 tensors on CPU and CUDA. Current implementation supports
  only the last dimension with a positive static reduction extent, uses stable
  max-subtract/exp/sum normalization, and targets attention-row shapes such as
  `[batch_heads * queries, keys]`. CUDA now has a warp-per-row register-cached
  specialization for odd/tail `K <= 2048`, a float2/float4 packed local-register
  path for divisible reductions up to the initial v1-style thresholds, and a
  shared-memory fallback for larger reductions. Non-last dimensions, generic
  dynamic reduction extents, reduced-precision storage contracts,
  strided/layout-aware tensors, full v1 K1/K2/K4/K8 small/middle/block policy
  parity, and profiler-selected variants remain unported.

Library hints: CUB is a good CUDA baseline for generic reductions and scans;
oneDNN has CPU softmax/reduction coverage; CK/MIOpen may cover selected GPU
normalization or softmax patterns, otherwise use custom block reductions.

### GEMM, BMM, and Fused Linear Families

- [x] Base GEMM layouts: `gemm_rcr`, `gemm_rrr` are explicit CUDA ops for
  `float32`, `float16`, and `bfloat16`, backed by cached CUTLASS launchers with
  CPU reference execution but no CPU compiled GEMM. Public `matmul` should wait
  until layout selection, profiler selection, and epilogue contracts are ready.
- [ ] Base BMM layout family: `bmm_{ccc,ccr,crc,crr,rcc,rcr,rrc,rrr}` plus
  `_add` variants.
- [ ] Bias/broadcast epilogues: `gemm_rcr_bias*`, `gemm_rrr_bias*`, including
  add/add-add/mul/mul-add and broadcast forms.
- [ ] Activation epilogues: relu, gelu/fast-gelu, sigmoid, tanh, swish/silu,
  hardswish, `elup1`, and compound sigmoid/mul/tanh forms.
- [ ] Permuted/layout-fused output families: `gemm_*_permute*`,
  `bmm_*_permute`, `perm021fc_*`, `perm102_bmm_*`.
- [ ] Grouped GEMM: `group_gemm_rcr*`.
- [ ] Softmax/attention matmul chains: `bmm_softmax_bmm*`,
  `bmm_rcr_softmax`, `gemm_rcr*_softmax`, `dual_bmm_rrr_div`.
- [ ] Dual-output/dual-GEMM epilogue families: `dual_gemm_rcr_*`.
- [ ] Specialized small/degenerate kernels: `gemm_rrr_small_nk`,
  `bmm_rcr_n1`, `bmm_rrr_k1_tanh`, `batched_dense_vec_jagged_2d_mul`.
- [ ] Back-to-back BMM: `classic_b2b_bmm`, `fmha_style_b2b_bmm`,
  `grouped_classic_b2b_bmm`, `grouped_fmha_style_b2b_bmm`.
- [ ] Direct-import helpers: `bmm`, `bmm_xxx`, `bmm_xxx_add`.

Library hints: CUTLASS is the primary CUDA candidate for GEMM/BMM, grouped GEMM,
and epilogue visitors. CK is the corresponding AMD path. oneDNN matmul/brgemm is
the CPU fallback target. Plain `torch.matmul`, `torch.bmm`, and `torch.addmm`
are good semantic references, but not replacements for v1 fused layout/epilogue
behavior.

### Convolution, Pooling, Padding, and Upsampling

- [ ] Convolution: `conv2d`, `conv3d`, `conv3d_bias`, `depthwise_conv3d`,
  `depthwise_conv3d_bias`, `transposed_conv2d`.
- [ ] Pooling: `avg_pool1d`, `avg_pool1d_compress_time`, `avg_pool2d`,
  `max_pool2d`.
- [ ] Padding/layout packing: `pad`, `pad_last_dim`, `nhwc3to4`, `nhwc3to8`,
  `ndhwc3to8`, `prepare_for_transposed_conv2d`.
- [ ] Upsampling: `upsampling{1d,2d,3d}`, `_add` variants, and
  `upsampling3d_compress_time`.

Library hints: cuDNN/MIOpen should cover most conv and pooling paths; oneDNN is
the CPU target. Packing helpers are likely custom copy kernels. Use
`torch.nn.functional` conv/pool/pad/interpolate as reference behavior.

### Normalization

- [ ] GroupNorm: `group_norm`, `group_norm_swish`.
- [ ] LayerNorm family: `layernorm`, `t5_layer_norm`, `group_layernorm`,
  `batch_layernorm_sigmoid_mul`, `layernorm_sigmoid_mul`,
  `group_layernorm_sigmoid_mul`.

Library hints: cuDNN/MIOpen/oneDNN can cover common norm shapes, but fused
sigmoid/mul/swish forms may need custom reductions or CUTLASS/CK epilogues when
paired with GEMM.

### Attention

- [ ] `flash_attention`.
- [ ] `flash_attn`.
- [ ] `mem_eff_attention`.

Library hints: `torch.nn.functional.scaled_dot_product_attention` is a semantic
reference. Validate causal masks, dropout, head layout, kv-cache, and variable
length behavior before deciding whether to wrap FlashAttention-style kernels,
CUTLASS/CK kernels, or compose GEMM/softmax/GEMM families.

### Jagged and Ragged Tensors

- [ ] `make_jagged`.
- [ ] `jagged_lengths_to_offsets`.
- [ ] `jagged_lengths_to_presences`.
- [ ] `jagged_to_padded_dense`.
- [ ] `padded_dense_to_jagged`.

Library hints: CUB scans are useful for lengths-to-offsets on CUDA. Most dense
to/from jagged transforms need custom kernels and careful shape metadata tests.
Use PyTorch only as partial reference; core `torch` does not directly match v1
JaggedIntVar semantics.

## Custom and Model-Fused Helper Ops

These are not common primitives. Port them after their underlying tensor,
elementwise, layout, and reduction pieces exist, unless a model requires one
early. Prefer implementing them as small graph rewrites over dedicated kernels
when compile-time constants make that practical.

### Embedding and Positional Helpers

- [ ] Embedding/model helpers: `bert_embeddings`, `relative_attention_bias`,
  `sinusoidal_positional_embedding`, `gaussian_fourier_projection`,
  `gelu_new`, `cropped_pos_embed`, `get_timestep_embedding`.
- [ ] Rotary/sincos helpers: `get_1d_rotary_pos_embed`,
  `get_2d_rotary_pos_embed`, `get_2d_rotary_pos_embed_lumina`,
  `get_2d_sincos_pos_embed`, `get_2d_sincos_pos_embed_cogview3plus`,
  `get_3d_rotary_pos_embed`, `get_3d_rotary_pos_embed_allegro`,
  `get_3d_sincos_pos_embed`, `get_3d_sincos_pos_embed_cogvideox`,
  `get_fourier_embeds_from_boundingbox`.

Library hints: no major external kernel library is expected to own these. Use
common primitive composition for CPU/CUDA first; add fused kernels only if these
become profiler-visible.

### Filtering and Resampling Helpers

- [ ] FIR/filter helpers: `fir_downsample2d`, `fir_filter_pad2`,
  `fir_upsample2d`.
- [ ] Kernel weight builders: `kdownsample2d_weight`, `kupsample2d_weight`.

Library hints: compose from pad/conv/upsample where possible. cuDNN/MIOpen may
help if represented as convolution; otherwise these are small custom kernels.

### Vision Detection Helpers

- [ ] NMS: `nms`, `batched_nms`, `efficient_nms`.
- [ ] ROI: `roi_align`, `multi_level_roi_align`.

Library hints: these are not in core `torch`, `torch.nn`, or
`torch.nn.functional`. TorchVision can be a reference if available. CUDA NMS may
use CUB for sorting/selection support, but box suppression and ROI sampling are
custom kernels.

## Suggested Porting Order

1. Harden elementwise parity: vectorized generated kernels, jagged/accessor
   support, dtype coverage, scalar promotion, and exhaustive numerical tests.
2. Add view/layout/selection/reduction primitives needed by model builders:
   reshape, permute, concatenate, split, slice, gather, topk, softmax.
3. Build the GEMM/BMM backbone once: base layouts, bias, activation epilogues,
   permuted outputs, grouped variants, and profiler/cache integration.
4. Port normalization, convolution, pooling, padding, and upsampling with
   library-backed paths where available.
5. Add attention and jagged/ragged support, because they combine multiple
   primitive families and shape rules.
6. Finish model-fused helpers, FIR resampling, NMS, and ROI after the reusable
   primitives are stable or when a target model makes one urgent.
