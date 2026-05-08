#pragma once

#include <dinoml/abi.h>

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
  if (tensor.ndim != expected_shape.size()) {
    return fail(std::string(name) + " rank mismatch");
  }
  for (size_t i = 0; i < expected_shape.size(); ++i) {
    if (tensor.shape[i] != expected_shape[i]) {
      return fail(std::string(name) + " shape mismatch");
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
  return 0;
}

inline int64_t tensor_numel(const DinoTensor& tensor) {
  int64_t total = 1;
  for (size_t i = 0; i < tensor.ndim; ++i) {
    total *= tensor.shape[i];
  }
  return total;
}

}  // namespace dinoml::module
