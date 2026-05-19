#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DINO_ACCESS_HD __host__ __device__
#define DINO_ACCESS_INLINE __forceinline__
#else
#define DINO_ACCESS_HD
#define DINO_ACCESS_INLINE inline
#endif

namespace dinoml::access {

enum class Pattern : int32_t {
  kDense = 0,
  kScalar = 1,
  kSuffix = 2,
  kStrided = 3,
  kGenericBroadcast = 4,
};

struct TensorAccessor {
  Pattern pattern{Pattern::kDense};
  int64_t offset{0};
  int64_t suffix_extent{1};
  int64_t original_total_elements_from_stride_dim{-1};
  int64_t actual_total_elements_from_stride_dim{-1};

  DINO_ACCESS_HD DINO_ACCESS_INLINE int64_t index(int64_t linear_idx) const {
    switch (pattern) {
      case Pattern::kDense:
        return offset + linear_idx;
      case Pattern::kScalar:
        return offset;
      case Pattern::kSuffix:
        return offset + linear_idx % suffix_extent;
      case Pattern::kStrided: {
        const int64_t row = linear_idx / original_total_elements_from_stride_dim;
        const int64_t col = linear_idx % original_total_elements_from_stride_dim;
        return offset + row * actual_total_elements_from_stride_dim + col;
      }
      case Pattern::kGenericBroadcast:
        return offset + linear_idx;
    }
    return offset + linear_idx;
  }

  DINO_ACCESS_HD DINO_ACCESS_INLINE int64_t index(int64_t linear_idx, int64_t inner_idx) const {
    if (pattern == Pattern::kSuffix) {
      return offset + inner_idx % suffix_extent;
    }
    return index(linear_idx);
  }
};

struct StridedTensorLayout {
  const int64_t* shape{nullptr};
  const int64_t* strides{nullptr};
  int64_t rank{0};
  int64_t offset{0};

  DINO_ACCESS_HD DINO_ACCESS_INLINE int64_t index(const int64_t* indices) const {
    int64_t storage_idx = offset;
    for (int64_t axis = 0; axis < rank; ++axis) {
      storage_idx += indices[axis] * strides[axis];
    }
    return storage_idx;
  }

  DINO_ACCESS_HD DINO_ACCESS_INLINE int64_t dense_index(int64_t linear_idx) const {
    int64_t storage_idx = offset;
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      const int64_t dim = shape[axis];
      const int64_t coord = linear_idx % dim;
      linear_idx /= dim;
      storage_idx += coord * strides[axis];
    }
    return storage_idx;
  }
};

template <typename DataT, typename ReadT, bool IsContiguous>
DINO_ACCESS_HD DINO_ACCESS_INLINE ReadT* strided_address(
    DataT* data,
    int64_t idx,
    int64_t offset,
    int64_t original_total_elements_from_stride_dim,
    int64_t actual_total_elements_from_stride_dim) {
  if constexpr (IsContiguous) {
    return reinterpret_cast<ReadT*>(data + offset) + idx;
  } else {
    constexpr int64_t kElementsPerRead = sizeof(ReadT) / sizeof(DataT);
    int64_t data_idx = idx * kElementsPerRead;
    const int64_t row = data_idx / original_total_elements_from_stride_dim;
    const int64_t col = data_idx % original_total_elements_from_stride_dim;
    data_idx = row * actual_total_elements_from_stride_dim + col + offset;
    return reinterpret_cast<ReadT*>(data + data_idx);
  }
}

}  // namespace dinoml::access

#undef DINO_ACCESS_HD
#undef DINO_ACCESS_INLINE
