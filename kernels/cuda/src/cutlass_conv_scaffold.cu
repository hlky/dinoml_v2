#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS Conv support stub boundary.
//
// This file is rendered into the support cache with concrete exported symbols
// from the manifest's used-candidate plan. The symbols intentionally return
// unsupported stub values until the real CUTLASS implicit-GEMM launcher lands.

extern "C" int dinoml_cutlass_conv_stub_status() {
  return 901;
}

// DINOML_CUTLASS_CONV_STUB_EXPORTS
