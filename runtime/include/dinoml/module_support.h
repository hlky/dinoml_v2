#pragma once

#include <dinoml/abi.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace dinoml::module {

inline int fail(const std::string& message) {
  return dino_runtime_fail(message.c_str());
}

inline std::string join_path(const char* dir, const char* file) {
  std::string path(dir);
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  path.append(file);
  return path;
}

inline int read_file(const std::string& path, std::vector<char>* out) {
  std::ifstream handle(path, std::ios::binary | std::ios::ate);
  if (!handle) {
    return fail("Failed to open " + path);
  }
  std::streamsize size = handle.tellg();
  handle.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(size));
  if (size > 0 && !handle.read(out->data(), size)) {
    return fail("Failed to read " + path);
  }
  return 0;
}

inline int check_tensor_layout(
    const DinoTensor& tensor,
    const char* name,
    const std::vector<int64_t>& actual_shape,
    int expected_dtype);

inline int check_tensor(
    const DinoTensor& tensor,
    const char* name,
    const std::vector<int64_t>& expected_shape,
    int expected_dtype,
    const char* expected_dtype_name) {
  if (tensor.dtype != expected_dtype) {
    return fail(std::string(name) + " must have dtype " + expected_dtype_name);
  }
  if (tensor.data == nullptr) {
    return fail(std::string(name) + " has null data pointer");
  }
  if (tensor.shape == nullptr && !expected_shape.empty()) {
    return fail(std::string(name) + " has null shape pointer");
  }
  if (tensor.ndim != expected_shape.size()) {
    return fail(std::string(name) + " rank mismatch");
  }
  for (size_t i = 0; i < expected_shape.size(); ++i) {
    if (tensor.shape[i] != expected_shape[i]) {
      return fail(std::string(name) + " shape mismatch");
    }
  }
  return check_tensor_layout(tensor, name, expected_shape, expected_dtype);
}

inline int dtype_nbytes(int dtype) {
  switch (dtype) {
    case DINO_DTYPE_FLOAT16:
    case DINO_DTYPE_BFLOAT16:
      return 2;
    case DINO_DTYPE_FLOAT32:
    case DINO_DTYPE_INT32:
      return 4;
    case DINO_DTYPE_INT64:
      return 8;
    case DINO_DTYPE_BOOL:
    case DINO_DTYPE_FLOAT8_E4M3:
    case DINO_DTYPE_FLOAT8_E5M2:
      return 1;
    default:
      return 0;
  }
}

inline int64_t shape_numel(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    total *= dim;
  }
  return total;
}

inline bool is_pointer_aligned(const void* ptr, size_t required_alignment_bytes) {
  if (required_alignment_bytes <= 1) {
    return true;
  }
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  return address % required_alignment_bytes == 0;
}

inline int check_pointer_alignment(
    const void* ptr,
    const char* name,
    size_t required_alignment_bytes) {
  if (!is_pointer_aligned(ptr, required_alignment_bytes)) {
    return fail(
        std::string(name) + " pointer does not satisfy required " +
        std::to_string(required_alignment_bytes) + "-byte alignment");
  }
  return 0;
}

inline int check_tensor_layout(
    const DinoTensor& tensor,
    const char* name,
    const std::vector<int64_t>& actual_shape,
    int expected_dtype) {
  if (tensor.byte_offset != 0) {
    return fail(std::string(name) + " byte offsets are not supported by this runtime ABI path");
  }
  if (tensor.strides != nullptr) {
    int64_t expected_stride = 1;
    for (size_t rev = 0; rev < actual_shape.size(); ++rev) {
      const size_t axis = actual_shape.size() - 1 - rev;
      if (tensor.strides[axis] != expected_stride) {
        return fail(std::string(name) + " must use contiguous row-major strides");
      }
      expected_stride *= actual_shape[axis];
    }
  }
  if (tensor.nbytes != 0) {
    const int nbytes = dtype_nbytes(expected_dtype);
    if (nbytes <= 0) {
      return fail(std::string(name) + " has unsupported dtype size");
    }
    const uint64_t required = static_cast<uint64_t>(shape_numel(actual_shape)) * static_cast<uint64_t>(nbytes);
    if (tensor.nbytes < required) {
      return fail(std::string(name) + " byte capacity is smaller than runtime shape requires");
    }
  }
  return 0;
}

inline int check_tensor_dynamic(
    const DinoTensor& tensor,
    const char* name,
    const std::vector<int64_t>& min_shape,
    const std::vector<int64_t>& max_shape,
    const std::vector<int64_t>& divisible_by,
    int expected_dtype,
    const char* expected_dtype_name) {
  if (tensor.dtype != expected_dtype) {
    return fail(std::string(name) + " must have dtype " + expected_dtype_name);
  }
  if (tensor.data == nullptr) {
    return fail(std::string(name) + " has null data pointer");
  }
  if (tensor.shape == nullptr && !max_shape.empty()) {
    return fail(std::string(name) + " has null shape pointer");
  }
  if (tensor.ndim != max_shape.size()) {
    return fail(std::string(name) + " rank mismatch");
  }
  for (size_t i = 0; i < max_shape.size(); ++i) {
    const int64_t dim = tensor.shape[i];
    if (dim < min_shape[i] || dim > max_shape[i]) {
      return fail(std::string(name) + " shape dimension is outside compiled range");
    }
    if (divisible_by[i] > 1 && dim % divisible_by[i] != 0) {
      return fail(std::string(name) + " shape dimension violates divisibility constraint");
    }
  }
  std::vector<int64_t> actual_shape;
  actual_shape.reserve(max_shape.size());
  for (size_t i = 0; i < max_shape.size(); ++i) {
    actual_shape.push_back(tensor.shape[i]);
  }
  return check_tensor_layout(tensor, name, actual_shape, expected_dtype);
}

inline int64_t tensor_numel(const DinoTensor& tensor) {
  if (tensor.shape == nullptr && tensor.ndim > 0) {
    return 0;
  }
  int64_t total = 1;
  for (size_t i = 0; i < tensor.ndim; ++i) {
    total *= tensor.shape[i];
  }
  return total;
}

}  // namespace dinoml::module
