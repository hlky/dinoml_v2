cmake_minimum_required(VERSION 3.24)

project(DinoMLV2 LANGUAGES CXX)

option(DINOML_ENABLE_CUDA "Build CUDA runtime and kernel libraries" ON)
if(WIN32)
  set(DINOML_DEFAULT_OPENMP OFF)
else()
  set(DINOML_DEFAULT_OPENMP ON)
endif()
option(DINOML_ENABLE_OPENMP "Use OpenMP when building CPU kernels" ${DINOML_DEFAULT_OPENMP})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DINOML_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(dinoml_runtime SHARED
  runtime/src/runtime_common.cpp
)
target_include_directories(dinoml_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
set_target_properties(dinoml_runtime PROPERTIES
  OUTPUT_NAME dinoml_runtime
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

if(DINOML_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
endif()

add_library(dinoml_cpu_kernels SHARED
  kernels/cpu/src/kernels.cpp
)
target_include_directories(dinoml_cpu_kernels
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu/include
)
target_link_libraries(dinoml_cpu_kernels
  PUBLIC dinoml_runtime
)
if(DINOML_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(dinoml_cpu_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(dinoml_cpu_kernels PROPERTIES
  OUTPUT_NAME dinoml_cpu_kernels
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

if(DINOML_ENABLE_CUDA)
  add_library(dinoml_cuda_runtime SHARED
    runtime/src/runtime_cuda.cu
  )
  target_include_directories(dinoml_cuda_runtime
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_link_libraries(dinoml_cuda_runtime
    PUBLIC dinoml_runtime
  )
  set_target_properties(dinoml_cuda_runtime PROPERTIES
    OUTPUT_NAME dinoml_cuda_runtime
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    CUDA_SEPARABLE_COMPILATION OFF
  )

  add_library(dinoml_cuda_kernels SHARED
    kernels/cuda/src/kernels.cu
  )
  target_include_directories(dinoml_cuda_kernels
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/include
  )
  target_link_libraries(dinoml_cuda_kernels
    PUBLIC dinoml_runtime dinoml_cuda_runtime
  )
  set_target_properties(dinoml_cuda_kernels PROPERTIES
    OUTPUT_NAME dinoml_cuda_kernels
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    CUDA_SEPARABLE_COMPILATION OFF
  )
endif()
