#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS Conv support boundary.
//
// This file is rendered into the support cache with concrete exported symbols
// from the manifest's used-candidate plan. Some symbols may still be explicit
// unsupported stubs, while admitted bounded-runtime candidates can render real
// CUTLASS launcher exports behind the same manifest-visible ABI.

extern "C" int dinoml_cutlass_conv_stub_status() {
  return 901;
}

// DINOML_CUTLASS_CONV_STUB_EXPORTS
