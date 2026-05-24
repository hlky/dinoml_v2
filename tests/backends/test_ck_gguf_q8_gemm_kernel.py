from __future__ import annotations

from pathlib import Path


def test_ck_gguf_q8_gemm_kernel_exports_prototype_symbols() -> None:
    source = Path("kernels/rocm/src/ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")

    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile4x64_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile16x16_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_opt_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_float16_opt_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_float32_opt_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_float16_ref_v1" in source
    assert "dinoml_ck_gguf_q8_0_gemm_rcr_float32_ref_v1" in source
    assert "dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1" in source
    assert "dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1" in source
    assert "dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n4_v1" in source
    assert "dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1" in source
    assert "dinoml_ck_gguf_q4_0_dequant_b_f16_nk_v1" in source
    assert "dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1" in source
    assert "dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n4_v1" in source
    assert "dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1" in source
    assert "dinoml_ck_gguf_q4_k_dequant_b_f16_nk_v1" in source
    assert "dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1" in source
    assert "dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n4_v1" in source
    assert "dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1" in source
    assert "dinoml_ck_gguf_q5_k_dequant_b_f16_nk_v1" in source
    for qtype in ("q2_k", "q3_k", "q6_k", "mxfp4", "nvfp4"):
        assert f"DINOML_DEFINE_GGUF_QUANT_EXPORTS({qtype}," in source
    assert "dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n4_v1" in source
    assert "dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6_v1" in source
    assert "dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12_v1" in source
    assert "dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m4_v1" in source
    assert "dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8w2_v1" in source
    assert "dinoml_ck_gguf_##prefix##_dequant_b_f16_nk_v1" in source
    assert "dinoml_ck_gguf_q8_0_transcode_bq_i8_float_v1" in source
    assert "dinoml_ck_gguf_q8_0_transcode_bq_fp8_float_v1" in source
    assert "dinoml_ck_gguf_q8_0_transcode_bq_bf8_float_v1" in source


def test_ck_gguf_q8_gemm_kernel_reads_native_gguf_rows() -> None:
    source = Path("kernels/rocm/src/ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")

    assert '#include "ck_tile/core.hpp"' in source
    assert "ck_tile::make_kernel" in source
    assert "ck_tile::launch_kernel" in source
    assert "static_assert(sizeof(GgufQ8_0Block) == 34" in source
    assert "__shared__ float a_tile" in source
    assert "__shared__ int8_t b_tile" in source
    assert "__shared__ float b_scale" in source
    assert "GgufQ8_0GemmRcrWaveKernel" in source
    assert "GgufQ8_0GemmRcrWaveM4Kernel" in source
    assert "GgufQ8_0GemmRcrWaveM4NExactKernel" in source
    assert "GgufQuantGemmRcrWmma16x16MBlocksKernel" in source
    assert "GgufQuantGemmRcrWmma16x16SharedBKernel" in source
    assert "__shared__ ck_tile::half_t b_shared" in source
    assert "ck_tile::WarpGemmDispatcher" in source
    assert "values8" in source
    assert "ck_tile::warp_shuffle_down" in source
    assert "args.b[global_n * qblocks_per_row + qb].d" in source
    assert "args.b[global_n * qblocks_per_row + qb].qs[tile_k]" in source
    assert "gguf_dequant_scratch" not in source
    assert "<<<grid, block" not in source


def test_ck_gguf_qtype_decode_matches_vendored_libgguf_layouts() -> None:
    source = Path("kernels/rocm/src/ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")
    bench = Path("tools/bench_ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")

    assert "static_assert(sizeof(GgufQ4_0Block) == 18" in source
    assert "static_assert(sizeof(GgufQ4_KBlock) == 144" in source
    assert "static_assert(sizeof(GgufQ5_KBlock) == 176" in source
    assert "static_assert(sizeof(GgufQ2_KBlock) == 84" in source
    assert "static_assert(sizeof(GgufQ3_KBlock) == 110" in source
    assert "static_assert(sizeof(GgufQ6_KBlock) == 210" in source
    assert "static_assert(sizeof(GgufMxFp4Block) == 17" in source
    assert "static_assert(sizeof(GgufNvFp4Block) == 36" in source
    assert "gguf_get_scale_min_k4" in source
    assert "gguf_q3_k_scale" in source
    assert "gguf_e8m0_to_fp32_half" in source
    assert "gguf_ue4m3_to_fp32" in source
    assert "get_scale_min_k4_host" in bench
    assert "q3_k_scale_host" in bench
    assert "q4_k_value_host" in bench
    assert "q5_k_value_host" in bench
    assert "q2_k_value_host" in bench
    assert "q3_k_value_host" in bench
    assert "q6_k_value_host" in bench
    assert "mxfp4_value_host" in bench
    assert "nvfp4_value_host" in bench
    assert "--qtype" in bench
    assert "--fused-candidate" in bench
    assert '"q2_k", "q3_k", "q6_k", "mxfp4", "nvfp4"' in bench
    assert "--sweep" in bench
    assert "decode_prefill" in bench
    assert "dit_balanced" in bench


def test_ck_gguf_q8_transcode_uses_ck_tile_bquant_layout() -> None:
    source = Path("kernels/rocm/src/ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")
    bench = Path("tools/bench_ck_gguf_q8_gemm.hip").read_text(encoding="utf-8")

    assert "GgufQ8_0TranscodeBqKernel" in source
    assert "args.b_kn[global_k * args.n + col]" in source
    assert "args.bq_qk_n[qb + col * qblocks_per_row]" in source
    assert "q8_to_bstorage<ck_tile::fp8_t>" in source
    assert "q8_to_bstorage<ck_tile::bf8_t>" in source
    assert "dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1" in bench
    assert "dinoml_ck_gguf_q8_0_dequant_b_f16_kn_v1" in bench
    assert "dequant_plus_best_dense_ck_ms" in bench
    assert "best_dequant_plus_ck_ms" in bench
    assert "best_fused_candidate" in bench
    assert "best_dense_ck_candidate" in bench
    assert "fused_wmma16x16_m2" in bench
    assert "fused_wmma16x16_m3" in bench
    assert "fused_wmma16x16_m8" in bench
    assert "fused_wmma16x16_m12" in bench
    assert "fused_wmma16x16_m16" in bench
    assert "fused_wmma16x16_sharedb_m4" in bench
    assert "candidate_fused_f16out_" in bench
    assert "wmma16x16_m12" in bench
    assert "best_fused_f16out_ms" in bench
    assert "f16_dequant_ref" in bench
    assert '"dense_ck_xdl_wide_m_nk"' not in bench


def test_ck_gguf_q8_gemm_has_off_by_default_cmake_target() -> None:
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")

    assert "DINOML_ENABLE_CK_GGUF_Q8_GEMM" in cmake
    assert "add_library(dinoml_ck_gguf_q8_gemm STATIC" in cmake
    assert "add_executable(dinoml_ck_gguf_q8_bench" in cmake
    assert "ck_q8_config/include" in cmake
    assert "third_party/composable_kernel/include" in cmake
