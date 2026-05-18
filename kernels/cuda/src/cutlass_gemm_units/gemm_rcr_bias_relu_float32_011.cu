#include "../cutlass_gemm_common.cuh"

DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x16_s10_w2x2x1_f32_align1, Sm80TensorOp64x64x16S10W2x2x1TF32F32Align1GemmPolicy, 1)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x16_s10_w2x2x1_f32_align2, Sm80TensorOp64x64x16S10W2x2x1TF32F32Align2GemmPolicy, 2)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x16_s10_w2x2x1_f32_align4, Sm80TensorOp64x64x16S10W2x2x1TF32F32Align4GemmPolicy, 4)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x32_s5_w2x2x1_f32_align1, Sm80TensorOp64x64x32S5W2x2x1TF32F32Align1GemmPolicy, 1)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x32_s5_w2x2x1_f32_align2, Sm80TensorOp64x64x32S5W2x2x1TF32F32Align2GemmPolicy, 2)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_relu, float32, float, float, cutlass::layout::ColumnMajor, k, BiasReluEpilogue, tensorop_sm80_tf32_64x64x32_s5_w2x2x1_f32_align4, Sm80TensorOp64x64x32S5W2x2x1TF32F32Align4GemmPolicy, 4)
