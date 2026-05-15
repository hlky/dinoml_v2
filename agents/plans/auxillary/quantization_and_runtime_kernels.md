# Quantization and Runtime Kernels

## Why This Matters

UI projects, especially Wan2GP, use custom qtypes, sparse attention, Triton
kernels, and low-VRAM runtime policies to make large image/video/audio models
usable. This overlaps DinoML's provider and encoded-constant direction, but is
broader than the current bounded GGUF Q4_0 GEMM path.

## Families and Packages

- GGUF packed weights.
- NVFP4 and Nunchaku FP4 layouts.
- `optimum.quanto` paths.
- `mmgp` memory/model management.
- `flash-linear-attention`.
- SageAttention/SpargeAttn-style sparse attention backends.
- Triton sparse INT8 attention and per-block quantization.

## Code Anchors

- `deepbeepmeep/Wan2GP/shared/qtypes/gguf.py:1`.
- `deepbeepmeep/Wan2GP/shared/qtypes/nvfp4.py:1`.
- `deepbeepmeep/Wan2GP/shared/qtypes/nunchaku_fp4.py:1`.
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/sparse_sage/sparse_int8_attn.py:22`
  Triton sparse INT8 attention.
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/sparse_sage/quant_per_block.py:22`
  Triton per-block INT8 quantization.
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/attention_backend.py:152`
  bundled sparse backend loader.
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/sparse_backend_config.py`
  sparse backend config helpers.
- `deepbeepmeep/Wan2GP/plugins/wan2gp-configuration/plugin.py:275`
  FlashVSR sparse/upscale user configuration.

## DinoML Gap

High. These should feed provider and constant-policy design:

- encoded constant metadata and residency;
- qtype-specific dequant or direct-kernel paths;
- sparse attention provider manifests;
- runtime capability checks and fallback;
- source/build provenance for custom kernels;
- explicit workspace and scratch requirements.

Avoid folding this into an invisible offload scheduler. The useful DinoML shape
is provider-visible and artifact-visible, matching the current GGUF/CUTLASS
direction.

