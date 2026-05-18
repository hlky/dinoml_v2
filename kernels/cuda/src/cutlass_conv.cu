#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/functional.h"
#include "cutlass/layout/tensor.h"

namespace {

template <typename T>
__global__ void dinoml_cutlass_conv_nchw_to_nhwc_kernel(
    const T* src,
    T* dst,
    int n,
    int c,
    int h,
    int w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * c * h * w;
  if (linear >= total) {
    return;
  }
  int x = linear % w;
  int tmp = linear / w;
  int y = tmp % h;
  tmp /= h;
  int channel = tmp % c;
  int batch = tmp / c;
  int dst_index = ((batch * h + y) * w + x) * c + channel;
  dst[dst_index] = src[linear];
}

template <typename T>
__global__ void dinoml_cutlass_conv_nhwc_to_nchw_kernel(
    const T* src,
    T* dst,
    int n,
    int c,
    int h,
    int w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * c * h * w;
  if (linear >= total) {
    return;
  }
  int x = linear % w;
  int tmp = linear / w;
  int y = tmp % h;
  tmp /= h;
  int channel = tmp % c;
  int batch = tmp / c;
  int src_index = ((batch * h + y) * w + x) * c + channel;
  dst[linear] = src[src_index];
}

template <typename T>
__global__ void dinoml_cutlass_conv_oihw_to_ohwi_kernel(
    const T* src,
    T* dst,
    int out_c,
    int in_c,
    int kernel_h,
    int kernel_w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = out_c * in_c * kernel_h * kernel_w;
  if (linear >= total) {
    return;
  }
  int kw = linear % kernel_w;
  int tmp = linear / kernel_w;
  int kh = tmp % kernel_h;
  tmp /= kernel_h;
  int in_channel = tmp % in_c;
  int out_channel = tmp / in_c;
  int dst_index = ((out_channel * kernel_h + kh) * kernel_w + kw) * in_c + in_channel;
  dst[dst_index] = src[linear];
}

template <typename T>
int dinoml_cutlass_conv_launch_nchw_to_nhwc(
    const void* src,
    void* dst,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = n * c * h * w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_nchw_to_nhwc_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      n,
      c,
      h,
      w);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int dinoml_cutlass_conv_launch_nhwc_to_nchw(
    const void* src,
    void* dst,
    int n,
    int c,
    int h,
    int w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = n * c * h * w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_nhwc_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      n,
      c,
      h,
      w);
  return static_cast<int>(cudaGetLastError());
}

template <typename T>
int dinoml_cutlass_conv_launch_oihw_to_ohwi(
    const void* src,
    void* dst,
    int out_c,
    int in_c,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || out_c <= 0 || in_c <= 0 || kernel_h <= 0 || kernel_w <= 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int total = out_c * in_c * kernel_h * kernel_w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  dinoml_cutlass_conv_oihw_to_ohwi_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const T*>(src),
      static_cast<T*>(dst),
      out_c,
      in_c,
      kernel_h,
      kernel_w);
  return static_cast<int>(cudaGetLastError());
}

using DinomlCutlassConvFp16Element = cutlass::half_t;
using DinomlCutlassConvFp16Accumulator = float;
using DinomlCutlassConvFp16Compute = float;
using DinomlCutlassConvFp16Layout = cutlass::layout::TensorNHWC;
using DinomlCutlassConvFp32Element = float;
using DinomlCutlassConvFp32Accumulator = float;
using DinomlCutlassConvFp32Compute = float;
using DinomlCutlassConvFp32Layout = cutlass::layout::TensorNHWC;

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp16_kernel_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n,
      h,
      w,
      c,
      out_c,
      kernel_h,
      kernel_w,
      out_h,
      out_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
  DinomlCutlassConvFp16Layout activation_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp16Layout weight_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp16Layout output_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(bias)),
       DinomlCutlassConvFp16Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp16Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(1)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp32_kernel_bias(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n,
      h,
      w,
      c,
      out_c,
      kernel_h,
      kernel_w,
      out_h,
      out_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      cutlass::conv::Mode::kCrossCorrelation,
      1,
      1);
  DinomlCutlassConvFp32Layout activation_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp32Layout weight_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp32Layout output_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(bias)),
       DinomlCutlassConvFp32Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp32Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(1)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp16_kernel_bias_add(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    const void* residual_nhwc,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp16Layout activation_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp16Layout weight_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp16Layout output_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments(
      problem_size,
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(residual_nhwc)),
       output_layout},
      {static_cast<DinomlCutlassConvFp16Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(1)},
      cutlass::conv::SplitKMode::kSerial,
      const_cast<DinomlCutlassConvFp16Element*>(
          static_cast<DinomlCutlassConvFp16Element const*>(bias)),
      nullptr,
      0,
      out_c);
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp32_kernel_bias_add(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    const void* residual_nhwc,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp32Layout activation_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp32Layout weight_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp32Layout output_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments(
      problem_size,
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(residual_nhwc)),
       output_layout},
      {static_cast<DinomlCutlassConvFp32Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(1)},
      cutlass::conv::SplitKMode::kSerial,
      const_cast<DinomlCutlassConvFp32Element*>(
          static_cast<DinomlCutlassConvFp32Element const*>(bias)),
      nullptr,
      0,
      out_c);
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp16_kernel_bias_add_relu(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    const void* residual_nhwc,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp16Layout activation_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp16Layout weight_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp16Layout output_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments(
      problem_size,
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(residual_nhwc)),
       output_layout},
      {static_cast<DinomlCutlassConvFp16Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(1)},
      cutlass::conv::SplitKMode::kSerial,
      const_cast<DinomlCutlassConvFp16Element*>(
          static_cast<DinomlCutlassConvFp16Element const*>(bias)),
      nullptr,
      0,
      out_c);
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp32_kernel_bias_add_relu(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    const void* residual_nhwc,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp32Layout activation_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp32Layout weight_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp32Layout output_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments(
      problem_size,
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(residual_nhwc)),
       output_layout},
      {static_cast<DinomlCutlassConvFp32Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(1)},
      cutlass::conv::SplitKMode::kSerial,
      const_cast<DinomlCutlassConvFp32Element*>(
          static_cast<DinomlCutlassConvFp32Element const*>(bias)),
      nullptr,
      0,
      out_c);
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp16_kernel_bias_relu(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp16Layout activation_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp16Layout weight_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp16Layout output_layout =
      DinomlCutlassConvFp16Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp16Element*>(
           static_cast<DinomlCutlassConvFp16Element const*>(bias)),
       DinomlCutlassConvFp16Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp16Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(1), DinomlCutlassConvFp16Compute(0)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

template <typename ImplicitGemm>
int dinoml_cutlass_conv_launch_fp32_kernel_bias_relu(
    const void* activation_nhwc,
    const void* weight_ohwi,
    const void* bias,
    void* output_nhwc,
    int n,
    int h,
    int w,
    int c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream) {
  cutlass::conv::Conv2dProblemSize problem_size(
      n, h, w, c, out_c, kernel_h, kernel_w, out_h, out_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      cutlass::conv::Mode::kCrossCorrelation, 1, 1);
  DinomlCutlassConvFp32Layout activation_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, h, w, c));
  DinomlCutlassConvFp32Layout weight_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(out_c, kernel_h, kernel_w, c));
  DinomlCutlassConvFp32Layout output_layout =
      DinomlCutlassConvFp32Layout::packed(cutlass::Tensor4DCoord(n, out_h, out_w, out_c));
  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(activation_nhwc)),
       activation_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(weight_ohwi)),
       weight_layout},
      {const_cast<DinomlCutlassConvFp32Element*>(
           static_cast<DinomlCutlassConvFp32Element const*>(bias)),
       DinomlCutlassConvFp32Layout::Stride(0)},
      {static_cast<DinomlCutlassConvFp32Element*>(output_nhwc), output_layout},
      {DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(1), DinomlCutlassConvFp32Compute(0)}};
  ImplicitGemm implicit_gemm;
  cutlass::Status status = implicit_gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return 1000 + static_cast<int>(status);
  }
  status = implicit_gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return 1100 + static_cast<int>(status);
  }
  status = implicit_gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return 1200 + static_cast<int>(status);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

#define DINOML_CUTLASS_CONV_NCHW_TO_NHWC_EXPORT(SYMBOL, DTYPE_PREFIX) \
extern "C" int SYMBOL(const void* src, void* dst, int n, int c, int h, int w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_nchw_to_nhwc<DinomlCutlassConv##DTYPE_PREFIX##Element>(src, dst, n, c, h, w, stream); \
}

#define DINOML_CUTLASS_CONV_OIHW_TO_OHWI_EXPORT(SYMBOL, DTYPE_PREFIX) \
extern "C" int SYMBOL(const void* src, void* dst, int out_c, int in_c, int kernel_h, int kernel_w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_oihw_to_ohwi<DinomlCutlassConv##DTYPE_PREFIX##Element>( \
      src, dst, out_c, in_c, kernel_h, kernel_w, stream); \
}

#define DINOML_CUTLASS_CONV_NHWC_TO_NCHW_EXPORT(SYMBOL, DTYPE_PREFIX) \
extern "C" int SYMBOL(const void* src, void* dst, int n, int c, int h, int w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_nhwc_to_nchw<DinomlCutlassConv##DTYPE_PREFIX##Element>(src, dst, n, c, h, w, stream); \
}

struct DinomlCutlassConvBiasTag {};
struct DinomlCutlassConvBiasReluTag {};
struct DinomlCutlassConvBiasAddTag {};
struct DinomlCutlassConvBiasAddReluTag {};

template <typename Element, typename Accumulator, typename Compute, int ElementsPerAccess, typename Tag>
struct DinomlCutlassConvEpilogue;

template <typename Element, typename Accumulator, typename Compute, int ElementsPerAccess>
struct DinomlCutlassConvEpilogue<Element, Accumulator, Compute, ElementsPerAccess, DinomlCutlassConvBiasTag> {
  using Type = cutlass::epilogue::thread::LinearCombination<Element, ElementsPerAccess, Accumulator, Compute>;
};

template <typename Element, typename Accumulator, typename Compute, int ElementsPerAccess>
struct DinomlCutlassConvEpilogue<Element, Accumulator, Compute, ElementsPerAccess, DinomlCutlassConvBiasReluTag> {
  using Type = cutlass::epilogue::thread::LinearCombinationRelu<Element, ElementsPerAccess, Accumulator, Compute>;
};

template <typename Element, typename Accumulator, typename Compute, int ElementsPerAccess>
struct DinomlCutlassConvEpilogue<Element, Accumulator, Compute, ElementsPerAccess, DinomlCutlassConvBiasAddTag> {
  using Type = cutlass::epilogue::thread::LinearCombinationResidualBlock<
      Element, Accumulator, Compute, Element, ElementsPerAccess,
      cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity>;
};

template <typename Element, typename Accumulator, typename Compute, int ElementsPerAccess>
struct DinomlCutlassConvEpilogue<Element, Accumulator, Compute, ElementsPerAccess, DinomlCutlassConvBiasAddReluTag> {
  using Type = cutlass::epilogue::thread::LinearCombinationResidualBlock<
      Element, Accumulator, Compute, Element, ElementsPerAccess,
      cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::ReLu>;
};

#define DINOML_CUTLASS_CONV_KERNEL(SYMBOL, DTYPE_PREFIX, KERNEL_TEMPLATE, EPILOGUE_TAG, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
namespace { \
using SYMBOL##_Epilogue = typename DinomlCutlassConvEpilogue< \
    DinomlCutlassConv##DTYPE_PREFIX##Element, \
    DinomlCutlassConv##DTYPE_PREFIX##Accumulator, \
    DinomlCutlassConv##DTYPE_PREFIX##Compute, \
    ELEMENTS_PER_ACCESS, \
    EPILOGUE_TAG>::Type; \
using SYMBOL##_Kernel = typename KERNEL_TEMPLATE< \
    DinomlCutlassConv##DTYPE_PREFIX##Element, DinomlCutlassConv##DTYPE_PREFIX##Layout, \
    DinomlCutlassConv##DTYPE_PREFIX##Element, DinomlCutlassConv##DTYPE_PREFIX##Layout, \
    DinomlCutlassConv##DTYPE_PREFIX##Element, DinomlCutlassConv##DTYPE_PREFIX##Layout, \
    DinomlCutlassConv##DTYPE_PREFIX##Accumulator, \
    OPCLASS, \
    cutlass::arch::Sm80, \
    cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>, \
    cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>, \
    cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>, \
    SYMBOL##_Epilogue, \
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, \
    STAGES, \
    MATH_OP, \
    ITERATOR, \
    cutlass::conv::StrideSupport::kStrided, \
    ALIGN_A, \
    ALIGN_B>::Kernel; \
using SYMBOL##_ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<SYMBOL##_Kernel>; \
}

#define DINOML_CUTLASS_CONV_PROFILE_BIAS(PROFILER_SYMBOL, LAUNCH_SYMBOL) \
extern "C" float PROFILER_SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int iterations, cudaStream_t stream) { \
  if (iterations <= 0) { return -1.0f; } \
  cudaEvent_t start; cudaEvent_t stop; \
  cudaError_t event_status = cudaEventCreate(&start); \
  if (event_status != cudaSuccess) { return -1.0f; } \
  event_status = cudaEventCreate(&stop); \
  if (event_status != cudaSuccess) { cudaEventDestroy(start); return -1.0f; } \
  cudaEventRecord(start, stream); \
  for (int iter = 0; iter < iterations; ++iter) { \
    int status = LAUNCH_SYMBOL(activation_nhwc, weight_ohwi, bias, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
    if (status != 0) { cudaEventDestroy(start); cudaEventDestroy(stop); return -1.0f; } \
  } \
  cudaEventRecord(stop, stream); \
  event_status = cudaEventSynchronize(stop); \
  if (event_status != cudaSuccess) { cudaEventDestroy(start); cudaEventDestroy(stop); return -1.0f; } \
  float elapsed_ms = 0.0f; \
  event_status = cudaEventElapsedTime(&elapsed_ms, start, stop); \
  cudaEventDestroy(start); cudaEventDestroy(stop); \
  return event_status == cudaSuccess ? elapsed_ms : -1.0f; \
}

#define DINOML_CUTLASS_CONV_PROFILE_BIAS_ADD(PROFILER_SYMBOL, LAUNCH_SYMBOL) \
extern "C" float PROFILER_SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, const void* residual_nhwc, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int iterations, cudaStream_t stream) { \
  if (iterations <= 0) { return -1.0f; } \
  cudaEvent_t start; cudaEvent_t stop; \
  cudaError_t event_status = cudaEventCreate(&start); \
  if (event_status != cudaSuccess) { return -1.0f; } \
  event_status = cudaEventCreate(&stop); \
  if (event_status != cudaSuccess) { cudaEventDestroy(start); return -1.0f; } \
  cudaEventRecord(start, stream); \
  for (int iter = 0; iter < iterations; ++iter) { \
    int status = LAUNCH_SYMBOL(activation_nhwc, weight_ohwi, bias, residual_nhwc, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
    if (status != 0) { cudaEventDestroy(start); cudaEventDestroy(stop); return -1.0f; } \
  } \
  cudaEventRecord(stop, stream); \
  event_status = cudaEventSynchronize(stop); \
  if (event_status != cudaSuccess) { cudaEventDestroy(start); cudaEventDestroy(stop); return -1.0f; } \
  float elapsed_ms = 0.0f; \
  event_status = cudaEventElapsedTime(&elapsed_ms, start, stop); \
  cudaEventDestroy(start); cudaEventDestroy(stop); \
  return event_status == cudaSuccess ? elapsed_ms : -1.0f; \
}

#define DINOML_CUTLASS_CONV_BIAS_EXPORT(SYMBOL, PROFILER_SYMBOL, DTYPE_PREFIX, LAUNCH_PREFIX, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
DINOML_CUTLASS_CONV_KERNEL(SYMBOL, DTYPE_PREFIX, cutlass::conv::kernel::DefaultConv2dFprop, DinomlCutlassConvBiasTag, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
extern "C" int SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_##LAUNCH_PREFIX##_kernel_bias<SYMBOL##_ImplicitGemm>(activation_nhwc, weight_ohwi, bias, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
DINOML_CUTLASS_CONV_PROFILE_BIAS(PROFILER_SYMBOL, SYMBOL)

#define DINOML_CUTLASS_CONV_BIAS_RELU_EXPORT(SYMBOL, PROFILER_SYMBOL, DTYPE_PREFIX, LAUNCH_PREFIX, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
DINOML_CUTLASS_CONV_KERNEL(SYMBOL, DTYPE_PREFIX, cutlass::conv::kernel::DefaultConv2dFprop, DinomlCutlassConvBiasReluTag, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
extern "C" int SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_##LAUNCH_PREFIX##_kernel_bias_relu<SYMBOL##_ImplicitGemm>(activation_nhwc, weight_ohwi, bias, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
DINOML_CUTLASS_CONV_PROFILE_BIAS(PROFILER_SYMBOL, SYMBOL)

#define DINOML_CUTLASS_CONV_BIAS_ADD_EXPORT(SYMBOL, PROFILER_SYMBOL, DTYPE_PREFIX, LAUNCH_PREFIX, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
DINOML_CUTLASS_CONV_KERNEL(SYMBOL, DTYPE_PREFIX, cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast, DinomlCutlassConvBiasAddTag, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
extern "C" int SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, const void* residual_nhwc, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_##LAUNCH_PREFIX##_kernel_bias_add<SYMBOL##_ImplicitGemm>(activation_nhwc, weight_ohwi, bias, residual_nhwc, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
DINOML_CUTLASS_CONV_PROFILE_BIAS_ADD(PROFILER_SYMBOL, SYMBOL)

#define DINOML_CUTLASS_CONV_BIAS_ADD_RELU_EXPORT(SYMBOL, PROFILER_SYMBOL, DTYPE_PREFIX, LAUNCH_PREFIX, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
DINOML_CUTLASS_CONV_KERNEL(SYMBOL, DTYPE_PREFIX, cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast, DinomlCutlassConvBiasAddReluTag, OPCLASS, TB_M, TB_N, TB_K, WARP_M, WARP_N, WARP_K, INST_M, INST_N, INST_K, STAGES, MATH_OP, ITERATOR, ALIGN_A, ALIGN_B, ELEMENTS_PER_ACCESS) \
extern "C" int SYMBOL(const void* activation_nhwc, const void* weight_ohwi, const void* bias, const void* residual_nhwc, void* output_nhwc, \
    int n, int h, int w, int c, int out_h, int out_w, int out_c, int kernel_h, int kernel_w, \
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, cudaStream_t stream) { \
  return dinoml_cutlass_conv_launch_##LAUNCH_PREFIX##_kernel_bias_add_relu<SYMBOL##_ImplicitGemm>(activation_nhwc, weight_ohwi, bias, residual_nhwc, output_nhwc, n, h, w, c, out_h, out_w, out_c, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
DINOML_CUTLASS_CONV_PROFILE_BIAS_ADD(PROFILER_SYMBOL, SYMBOL)

// DINOML_CUTLASS_CONV_EXPORTS
