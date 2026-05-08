#include <dinoml/runtime.h>

#include <sstream>

extern "C" {

int dino_runtime_cuda_check(
    cudaError_t err,
    const char* expr,
    const char* file,
    int line) {
  if (err == cudaSuccess) {
    return 0;
  }
  std::ostringstream oss;
  oss << expr << " failed at " << file << ":" << line << ": "
      << cudaGetErrorString(err);
  return dino_runtime_fail(oss.str().c_str());
}

int dino_device_malloc(void** ptr, size_t nbytes) {
  if (ptr == nullptr) {
    return dino_runtime_fail("dino_device_malloc received null output pointer");
  }
  DINO_CUDA_CHECK(cudaMalloc(ptr, nbytes));
  return 0;
}

int dino_device_free(void* ptr) {
  if (ptr != nullptr) {
    DINO_CUDA_CHECK(cudaFree(ptr));
  }
  return 0;
}

int dino_copy_host_to_device(
    void* dst_device,
    const void* src_host,
    size_t nbytes) {
  DINO_CUDA_CHECK(cudaMemcpy(dst_device, src_host, nbytes, cudaMemcpyHostToDevice));
  return 0;
}

int dino_copy_device_to_host(
    void* dst_host,
    const void* src_device,
    size_t nbytes) {
  DINO_CUDA_CHECK(cudaMemcpy(dst_host, src_device, nbytes, cudaMemcpyDeviceToHost));
  return 0;
}

int dino_copy_device_to_device(
    void* dst_device,
    const void* src_device,
    size_t nbytes) {
  DINO_CUDA_CHECK(cudaMemcpy(dst_device, src_device, nbytes, cudaMemcpyDeviceToDevice));
  return 0;
}

}
