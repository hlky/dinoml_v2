#include "../cutlass_gemm_common.cuh"

DINOML_FORWARD_GEMM_BIAS_EXPORT(gemm_rcr_bias, float32, float, float, f32, tensorop_sm80_tf32_128x128x16_s3_w2x2x1_f32_align1, Sm80TensorOp128x128x16S3W2x2x1TF32F32Align1GemmPolicy, 1)
DINOML_FORWARD_GEMM_BIAS_EXPORT(gemm_rcr_bias, float32, float, float, f32, tensorop_sm80_tf32_128x128x16_s3_w2x2x1_f32_align2, Sm80TensorOp128x128x16S3W2x2x1TF32F32Align2GemmPolicy, 2)
DINOML_FORWARD_GEMM_BIAS_EXPORT(gemm_rcr_bias, float32, float, float, f32, tensorop_sm80_tf32_128x128x16_s3_w2x2x1_f32_align4, Sm80TensorOp128x128x16S3W2x2x1TF32F32Align4GemmPolicy, 4)
