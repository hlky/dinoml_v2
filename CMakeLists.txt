cmake_minimum_required(VERSION 3.24)

project(DinoMLV2 LANGUAGES CXX)

option(DINOML_ENABLE_CUDA "Build CUDA runtime and kernel libraries" ON)
option(DINOML_ENABLE_CUTLASS_GEMM "Build CUTLASS GEMM support library" ON)
option(DINOML_ENABLE_CUTLASS_GEMM_PROFILER "Build CUTLASS GEMM profiler binary and Python binding" ON)
set(DINOML_CUTLASS_GEMM_PROFILER_TARGETS "" CACHE STRING "Semicolon-separated CUTLASS GEMM CMake targets to link into the profiler, or ALL for every target")
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
  find_package(CUDAToolkit REQUIRED)
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

  if(DINOML_ENABLE_CUTLASS_GEMM)
    file(GLOB DINOML_CUTLASS_GEMM_SOURCES CONFIGURE_DEPENDS
      kernels/cuda/src/cutlass_gemm_units/*.cu
    )
    set(DINOML_CUTLASS_GEMM_UNIT_GROUPS)
    foreach(_cutlass_gemm_source IN LISTS DINOML_CUTLASS_GEMM_SOURCES)
      get_filename_component(_cutlass_gemm_stem "${_cutlass_gemm_source}" NAME_WE)
      if(NOT _cutlass_gemm_stem MATCHES "^(.+)_(float32|float16|bfloat16)_[0-9][0-9][0-9]$")
        message(FATAL_ERROR "Unexpected CUTLASS GEMM unit filename: ${_cutlass_gemm_source}")
      endif()
      set(_cutlass_gemm_unit_group "${CMAKE_MATCH_1}_${CMAKE_MATCH_2}")
      string(MAKE_C_IDENTIFIER "${_cutlass_gemm_unit_group}" _cutlass_gemm_unit_group_id)
      list(APPEND DINOML_CUTLASS_GEMM_UNIT_GROUPS "${_cutlass_gemm_unit_group_id}")
      list(APPEND DINOML_CUTLASS_GEMM_SOURCES_${_cutlass_gemm_unit_group_id} "${_cutlass_gemm_source}")
      set(DINOML_CUTLASS_GEMM_OUTPUT_${_cutlass_gemm_unit_group_id} "dinoml_cutlass_${_cutlass_gemm_unit_group}")
    endforeach()
    list(REMOVE_DUPLICATES DINOML_CUTLASS_GEMM_UNIT_GROUPS)

    set(DINOML_CUTLASS_GEMM_TARGETS)
    foreach(_cutlass_gemm_unit_group_id IN LISTS DINOML_CUTLASS_GEMM_UNIT_GROUPS)
      set(_cutlass_gemm_target "dinoml_cutlass_gemm_${_cutlass_gemm_unit_group_id}")
      add_library(${_cutlass_gemm_target} STATIC
        ${DINOML_CUTLASS_GEMM_SOURCES_${_cutlass_gemm_unit_group_id}}
      )
      target_include_directories(${_cutlass_gemm_target}
        PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
      )
      target_compile_options(${_cutlass_gemm_target}
        PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
          $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
      )
      set_target_properties(${_cutlass_gemm_target} PROPERTIES
        OUTPUT_NAME ${DINOML_CUTLASS_GEMM_OUTPUT_${_cutlass_gemm_unit_group_id}}
        CUDA_SEPARABLE_COMPILATION OFF
      )
      list(APPEND DINOML_CUTLASS_GEMM_TARGETS ${_cutlass_gemm_target})
    endforeach()
    add_custom_target(dinoml_cutlass_gemm DEPENDS ${DINOML_CUTLASS_GEMM_TARGETS})

    if(DINOML_ENABLE_CUTLASS_GEMM_PROFILER)
      find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
      if(NOT pybind11_DIR)
        execute_process(
          COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
          OUTPUT_VARIABLE _DINOML_PYBIND11_DIR
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_DINOML_PYBIND11_DIR)
          set(pybind11_DIR "${_DINOML_PYBIND11_DIR}" CACHE PATH "pybind11 CMake package directory")
        endif()
      endif()
      find_package(pybind11 CONFIG REQUIRED)
      set(DINOML_CUTLASS_GEMM_PROFILER_LINK_TARGETS)
      if(DINOML_CUTLASS_GEMM_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CUTLASS_GEMM_PROFILER_LINK_TARGETS ${DINOML_CUTLASS_GEMM_TARGETS})
      elseif(DINOML_CUTLASS_GEMM_PROFILER_TARGETS)
        foreach(_cutlass_gemm_profiler_target IN LISTS DINOML_CUTLASS_GEMM_PROFILER_TARGETS)
          if(NOT TARGET ${_cutlass_gemm_profiler_target})
            message(FATAL_ERROR "Unknown CUTLASS GEMM profiler target: ${_cutlass_gemm_profiler_target}")
          endif()
          list(APPEND DINOML_CUTLASS_GEMM_PROFILER_LINK_TARGETS ${_cutlass_gemm_profiler_target})
        endforeach()
      endif()

      set(DINOML_CUTLASS_GEMM_PROFILER_TARGETS_BUILT)
      foreach(_cutlass_gemm_target IN LISTS DINOML_CUTLASS_GEMM_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_cutlass_gemm_" "" _cutlass_gemm_profiler_group_id "${_cutlass_gemm_target}")
        if(NOT _cutlass_gemm_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CUTLASS GEMM profiler op/dtype from target: ${_cutlass_gemm_target}")
        endif()
        set(_cutlass_gemm_profiler_op "${CMAKE_MATCH_1}")
        set(_cutlass_gemm_profiler_dtype "${CMAKE_MATCH_2}")
        set(_cutlass_gemm_profiler_output "dinoml_cutlass_gemm_profiler_${_cutlass_gemm_profiler_group_id}")
        set(_cutlass_gemm_profiler_bind_target "${_cutlass_gemm_profiler_output}_bind")
        set(_cutlass_gemm_profiler_exe_target "${_cutlass_gemm_profiler_output}")
        set(_cutlass_gemm_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/cutlass_gemm_profiler_dispatch/${_cutlass_gemm_profiler_group_id}.cu"
        )
        execute_process(
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_gemm_profiler_dispatch.py
            --op ${_cutlass_gemm_profiler_op}
            --dtype ${_cutlass_gemm_profiler_dtype}
            --output ${_cutlass_gemm_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          COMMAND_ERROR_IS_FATAL ANY
        )
        set(_cutlass_gemm_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_cutlass_gemm_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_cutlass_gemm_profiler_bind_target} MODULE
          tools/cutlass_gemm_profiler_pybind.cu
          ${_cutlass_gemm_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_gemm_profiler_bind_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
        )
        target_link_libraries(${_cutlass_gemm_profiler_bind_target}
          PRIVATE pybind11::module CUDA::cudart ${CMAKE_DL_LIBS}
        )
        target_link_options(${_cutlass_gemm_profiler_bind_target}
          PRIVATE ${_cutlass_gemm_profiler_linker_flags}
        )
        target_compile_options(${_cutlass_gemm_profiler_bind_target}
          PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        target_compile_definitions(${_cutlass_gemm_profiler_bind_target}
          PRIVATE
            DINOML_CUTLASS_GEMM_PROFILER_PY_MODULE=${_cutlass_gemm_profiler_bind_target}
        )
        set_target_properties(${_cutlass_gemm_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_cutlass_gemm_profiler_output}_bind"
          CUDA_SEPARABLE_COMPILATION OFF
        )

        add_executable(${_cutlass_gemm_profiler_exe_target}
          tools/cutlass_gemm_profiler_main.cu
          ${_cutlass_gemm_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_gemm_profiler_exe_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
        )
        target_link_libraries(${_cutlass_gemm_profiler_exe_target}
          PRIVATE CUDA::cudart ${CMAKE_DL_LIBS}
        )
        target_link_options(${_cutlass_gemm_profiler_exe_target}
          PRIVATE ${_cutlass_gemm_profiler_linker_flags}
        )
        target_compile_options(${_cutlass_gemm_profiler_exe_target}
          PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        set_target_properties(${_cutlass_gemm_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_cutlass_gemm_profiler_output}"
          ENABLE_EXPORTS ON
          CUDA_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_cutlass_gemm_profiler_bind_target} ${_cutlass_gemm_target})
        add_dependencies(${_cutlass_gemm_profiler_exe_target} ${_cutlass_gemm_target})
        list(APPEND DINOML_CUTLASS_GEMM_PROFILER_TARGETS_BUILT
          ${_cutlass_gemm_profiler_bind_target}
          ${_cutlass_gemm_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CUTLASS_GEMM_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_cutlass_gemm_profilers DEPENDS ${DINOML_CUTLASS_GEMM_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()
endif()
