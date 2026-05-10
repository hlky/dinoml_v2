#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif
#define DINO_HD __host__ __device__
#define DINO_FORCEINLINE __forceinline__
#else
#define DINO_HD
#define DINO_FORCEINLINE inline
#endif

namespace dinoml::math {

#if !defined(__CUDACC__)
struct float16 {
  uint16_t bits = 0;
};

struct bfloat16 {
  uint16_t bits = 0;
};

static_assert(sizeof(float16) == 2, "float16 storage must be 16 bits");
static_assert(sizeof(bfloat16) == 2, "bfloat16 storage must be 16 bits");

DINO_FORCEINLINE uint32_t float_to_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

DINO_FORCEINLINE float bits_to_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

DINO_FORCEINLINE float half_bits_to_float(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = static_cast<uint32_t>((value >> 10) & 0x1fu);
  uint32_t mantissa = static_cast<uint32_t>(value & 0x03ffu);
  uint32_t bits = sign;
  if (exponent == 0) {
    if (mantissa != 0) {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits |= (exponent + 112u) << 23;
      bits |= mantissa << 13;
    }
  } else if (exponent == 31) {
    bits |= 0x7f800000u | (mantissa << 13);
  } else {
    bits |= (exponent + 112u) << 23;
    bits |= mantissa << 13;
  }
  return bits_to_float(bits);
}

DINO_FORCEINLINE uint16_t float_to_half_bits(float value) {
  const uint32_t bits = float_to_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t mantissa = bits & 0x007fffffu;
  int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    const uint32_t rounded =
        (mantissa + (1u << (shift - 1)) - 1u + ((mantissa >> shift) & 1u)) >> shift;
    return static_cast<uint16_t>(sign | rounded);
  }

  if (exponent == 143) {
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
    mantissa >>= 13;
    return static_cast<uint16_t>(sign | 0x7c00u | mantissa | (mantissa == 0));
  }

  if (exponent > 30) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  mantissa += 0x00000fffu + ((mantissa >> 13) & 1u);
  if (mantissa & 0x00800000u) {
    mantissa = 0;
    ++exponent;
    if (exponent > 30) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

DINO_FORCEINLINE float bfloat16_bits_to_float(uint16_t value) {
  return bits_to_float(static_cast<uint32_t>(value) << 16);
}

DINO_FORCEINLINE uint16_t float_to_bfloat16_bits(float value) {
  const uint32_t bits = float_to_bits(value);
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}
#endif

template <typename To, typename From>
DINO_HD DINO_FORCEINLINE To cast(From x) {
  return static_cast<To>(x);
}

#if defined(__CUDACC__)
template <>
DINO_HD DINO_FORCEINLINE float cast<float, half>(half x) {
  return __half2float(x);
}

template <>
DINO_HD DINO_FORCEINLINE half cast<half, float>(float x) {
  return __float2half_rn(x);
}

#if __CUDACC_VER_MAJOR__ >= 11
template <>
DINO_HD DINO_FORCEINLINE float cast<float, __nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <>
DINO_HD DINO_FORCEINLINE __nv_bfloat16 cast<__nv_bfloat16, float>(float x) {
  return __float2bfloat16_rn(x);
}
#endif
#endif

#if !defined(__CUDACC__)
template <>
DINO_FORCEINLINE float cast<float, float16>(float16 x) {
  return half_bits_to_float(x.bits);
}

template <>
DINO_FORCEINLINE float16 cast<float16, float>(float x) {
  return float16{float_to_half_bits(x)};
}

template <>
DINO_FORCEINLINE float cast<float, bfloat16>(bfloat16 x) {
  return bfloat16_bits_to_float(x.bits);
}

template <>
DINO_FORCEINLINE bfloat16 cast<bfloat16, float>(float x) {
  return bfloat16{float_to_bfloat16_bits(x)};
}
#endif

template <typename T>
DINO_HD DINO_FORCEINLINE float to_float(T x) {
  return cast<float>(x);
}

template <typename T>
DINO_HD DINO_FORCEINLINE T from_float(float x) {
  return cast<T>(x);
}

DINO_HD DINO_FORCEINLINE float exp_float(float x) {
#if defined(__CUDA_ARCH__) && defined(DINOML_USE_CUDA_FAST_MATH)
  return __expf(x);
#else
  return expf(x);
#endif
}

DINO_HD DINO_FORCEINLINE float sin_float(float x) {
#if defined(__CUDA_ARCH__) && defined(DINOML_USE_CUDA_FAST_MATH)
  return __sinf(x);
#else
  return sinf(x);
#endif
}

DINO_HD DINO_FORCEINLINE float cos_float(float x) {
#if defined(__CUDA_ARCH__) && defined(DINOML_USE_CUDA_FAST_MATH)
  return __cosf(x);
#else
  return cosf(x);
#endif
}

DINO_HD DINO_FORCEINLINE float sigmoid_float(float x) {
#if defined(__CUDA_ARCH__) && defined(DINOML_USE_CUDA_FAST_MATH)
  return (tanhf(x * 0.5f) + 1.0f) * 0.5f;
#else
  return 1.0f / (1.0f + expf(-x));
#endif
}

DINO_HD DINO_FORCEINLINE bool is_nan_float(float x) {
#if defined(__CUDACC__)
  return isnan(x);
#else
  return std::isnan(x);
#endif
}

DINO_HD DINO_FORCEINLINE bool is_inf_float(float x) {
#if defined(__CUDACC__)
  return isinf(x);
#else
  return std::isinf(x);
#endif
}

template <typename T>
DINO_HD DINO_FORCEINLINE bool is_nan(T x) {
  return is_nan_float(to_float(x));
}

template <typename T>
DINO_HD DINO_FORCEINLINE bool is_inf(T x) {
  return is_inf_float(to_float(x));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T add(T a, T b) { return a + b; }

template <typename T>
DINO_HD DINO_FORCEINLINE T sub(T a, T b) { return a - b; }

template <typename T>
DINO_HD DINO_FORCEINLINE T mul(T a, T b) { return a * b; }

template <typename T>
DINO_HD DINO_FORCEINLINE T div(T a, T b) { return a / b; }

template <typename T>
DINO_HD DINO_FORCEINLINE T tanh(T x) { return from_float<T>(tanhf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T cos(T x) { return from_float<T>(cos_float(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T sin(T x) { return from_float<T>(sin_float(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T sign(T x) {
  const float xf = to_float(x);
  return from_float<T>(static_cast<float>((xf > 0.0f) - (xf < 0.0f)));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T abs(T x) { return from_float<T>(fabsf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T log(T x) { return from_float<T>(logf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T log1p(T x) { return from_float<T>(log1pf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T exp(T x) { return from_float<T>(exp_float(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T sqrt(T x) { return from_float<T>(sqrtf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T max(T a, T b) {
  return (is_nan(a) || is_nan(b)) ? from_float<T>(nanf("")) : from_float<T>(fmaxf(to_float(a), to_float(b)));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T min(T a, T b) {
  return (is_nan(a) || is_nan(b)) ? from_float<T>(nanf("")) : from_float<T>(fminf(to_float(a), to_float(b)));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T sigmoid(T x) {
  const float xf = to_float(x);
  return from_float<T>(sigmoid_float(xf));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T leaky_relu(T x, T negative_slope = from_float<T>(0.01f)) {
  return to_float(x) > 0.0f ? x : x * negative_slope;
}

template <typename T>
DINO_HD DINO_FORCEINLINE T hardtanh(T x, T min_value = from_float<T>(-1.0f), T max_value = from_float<T>(1.0f)) {
  return to_float(x) <= to_float(min_value) ? min_value : (to_float(x) >= to_float(max_value) ? max_value : x);
}

template <typename T>
DINO_HD DINO_FORCEINLINE T relu(T x) { return max(x, from_float<T>(0.0f)); }

template <typename T>
DINO_HD DINO_FORCEINLINE T nan_to_num(
    T x,
    T nan_replacement = from_float<T>(0.0f),
    T posinf_replacement = from_float<T>(0.0f),
    T neginf_replacement = from_float<T>(0.0f)) {
  if (is_nan(x)) {
    return nan_replacement;
  }
  if (is_inf(x)) {
    return to_float(x) > 0.0f ? posinf_replacement : neginf_replacement;
  }
  return x;
}

template <typename T>
DINO_HD DINO_FORCEINLINE T clamp_nan_to_num(
    T x,
    T clamp_min,
    T clamp_max,
    T nan_replacement = from_float<T>(0.0f)) {
  return is_nan(x) ? nan_replacement : hardtanh(x, clamp_min, clamp_max);
}

template <typename T>
DINO_HD DINO_FORCEINLINE T silu(T x) { return x * sigmoid(x); }

template <typename T>
DINO_HD DINO_FORCEINLINE T pow(T a, T b) { return from_float<T>(powf(to_float(a), to_float(b))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T gelu(T x) {
  const float xf = to_float(x);
  return from_float<T>(0.5f * xf * (1.0f + tanhf(0.7978845608028654f * (xf + 0.044715f * xf * xf * xf))));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T fast_gelu(T x) {
  const float xf = to_float(x);
  return from_float<T>(xf * sigmoid_float(1.702f * xf));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T softplus(T x) { return from_float<T>(log1pf(exp_float(to_float(x)))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T elu(T x, T alpha = from_float<T>(1.0f)) {
  return to_float(x) > 0.0f ? x : from_float<T>(to_float(alpha) * (exp_float(to_float(x)) - 1.0f));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T softsign(T x) {
  const float xf = to_float(x);
  return from_float<T>(xf / (1.0f + fabsf(xf)));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T floor_div(T a, T b) { return from_float<T>(floorf(to_float(a) / to_float(b))); }

template <typename T>
DINO_HD DINO_FORCEINLINE T celu(T x, T alpha = from_float<T>(1.0f)) {
  const float xf = to_float(x);
  const float af = to_float(alpha);
  return from_float<T>(fmaxf(0.0f, xf) + fminf(0.0f, af * (exp_float(xf / af) - 1.0f)));
}

template <typename T>
DINO_HD DINO_FORCEINLINE T floor(T x) { return from_float<T>(floorf(to_float(x))); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool eq(T a, T b) { return to_float(a) == to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool ge(T a, T b) { return to_float(a) >= to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool gt(T a, T b) { return to_float(a) > to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool le(T a, T b) { return to_float(a) <= to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool lt(T a, T b) { return to_float(a) < to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE bool ne(T a, T b) { return to_float(a) != to_float(b); }

template <typename T>
DINO_HD DINO_FORCEINLINE T where(bool condition, T x, T y) { return condition ? x : y; }

}  // namespace dinoml::math

#undef DINO_HD
#undef DINO_FORCEINLINE
