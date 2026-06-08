cmake_minimum_required(VERSION 3.24)

option(DINOML_ENABLE_ROCM "Build ROCm runtime and kernel libraries" OFF)

if(DINOML_ENABLE_ROCM)
  include(cmake/DinoMLROCmSdk.cmake)
  dinoml_configure_rocm_sdk()
  if(DINOML_ROCM_LLVM_BIN AND NOT DEFINED CMAKE_CXX_COMPILER)
    if(WIN32)
      set(_DINOML_ROCM_CXX_CANDIDATES
        "${DINOML_ROCM_LLVM_BIN}/clang++.exe"
        "${DINOML_ROCM_LLVM_BIN}/amdclang++.exe"
      )
    else()
      set(_DINOML_ROCM_CXX_CANDIDATES
        "${DINOML_ROCM_LLVM_BIN}/clang++"
        "${DINOML_ROCM_LLVM_BIN}/amdclang++"
      )
    endif()
    foreach(_DINOML_ROCM_CXX IN LISTS _DINOML_ROCM_CXX_CANDIDATES)
      if(EXISTS "${_DINOML_ROCM_CXX}")
        set(CMAKE_CXX_COMPILER "${_DINOML_ROCM_CXX}" CACHE FILEPATH "C++ compiler")
        break()
      endif()
    endforeach()
  endif()
endif()

project(DinoMLV2 LANGUAGES CXX)

option(DINOML_ENABLE_CUDA "Build CUDA runtime and kernel libraries" ON)
option(DINOML_ENABLE_CUTLASS_GEMM "Build CUTLASS GEMM support library" ON)
option(DINOML_ENABLE_CUTLASS_GEMM_PROFILER "Build CUTLASS GEMM profiler binary and Python binding" ON)
set(DINOML_CUTLASS_GEMM_PROFILER_TARGETS "" CACHE STRING "Semicolon-separated CUTLASS GEMM CMake targets to link into the profiler, or ALL for every target")
set(DINOML_CUTLASS_GEMM_CHUNK_SIZE 16 CACHE STRING "Number of CUTLASS GEMM export instantiations per generated CUDA source chunk")
option(DINOML_ENABLE_CUTLASS_BMM "Build CUTLASS BMM support library" ON)
option(DINOML_ENABLE_CUTLASS_BMM_PROFILER "Build CUTLASS BMM profiler binary and Python binding" ON)
set(DINOML_CUTLASS_BMM_PROFILER_TARGETS "" CACHE STRING "Semicolon-separated CUTLASS BMM CMake targets to link into the profiler, or ALL for every target")
option(DINOML_ENABLE_CUTLASS_CONV "Build CUTLASS Conv support library" ON)
option(DINOML_ENABLE_CUTLASS_CONV_PROFILER "Build CUTLASS Conv profiler binary and Python binding" ON)
set(DINOML_CUTLASS_CONV_PROFILER_TARGETS "" CACHE STRING "Semicolon-separated CUTLASS Conv CMake targets to link into the profiler, or ALL for every target")
option(DINOML_ENABLE_CK_GEMM "Build CK GEMM support library when ROCm is enabled" ${DINOML_ENABLE_ROCM})
option(DINOML_ENABLE_CK_BMM "Build CK BMM support library when ROCm is enabled" ${DINOML_ENABLE_ROCM})
option(DINOML_ENABLE_CK_CONV "Build CK Conv support library when ROCm is enabled" ${DINOML_ENABLE_ROCM})
option(DINOML_ENABLE_FLASH_ATTN_CK "Build CK FlashAttention support library when ROCm is enabled" ${DINOML_ENABLE_ROCM})
option(DINOML_ENABLE_FLASH_ATTN_CUDA "Build CUDA FlashAttention support library" OFF)
option(DINOML_ENABLE_LIBGGUF_CUDA "Build libgguf CUDA runtime-dequant support archive" OFF)
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

if(DINOML_ENABLE_ROCM)
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
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

if(DINOML_ENABLE_ROCM)
  add_library(dinoml_rocm_runtime SHARED
    runtime/src/runtime_rocm.hip
  )
  target_include_directories(dinoml_rocm_runtime
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_link_libraries(dinoml_rocm_runtime
    PUBLIC dinoml_runtime hip::host hip::device
  )
  set_target_properties(dinoml_rocm_runtime PROPERTIES
    OUTPUT_NAME dinoml_rocm_runtime
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    HIP_SEPARABLE_COMPILATION OFF
  )

  add_library(dinoml_rocm_kernels SHARED
    kernels/rocm/src/kernels.hip
    kernels/rocm/src/ck_tile_fp32_fallback.hip
  )
  target_include_directories(dinoml_rocm_kernels
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
  )
  target_compile_definitions(dinoml_rocm_kernels
    PRIVATE
      DINOML_HIP=1
      $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
  )
  target_link_libraries(dinoml_rocm_kernels
    PUBLIC dinoml_runtime dinoml_rocm_runtime hip::host hip::device
  )
  set_target_properties(dinoml_rocm_kernels PROPERTIES
    OUTPUT_NAME dinoml_rocm_kernels
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    HIP_SEPARABLE_COMPILATION OFF
  )

  if(TARGET hip::amdhip64)
    foreach(_dinoml_rocm_target IN ITEMS dinoml_rocm_runtime dinoml_rocm_kernels)
      add_custom_command(TARGET ${_dinoml_rocm_target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:hip::amdhip64>"
                "$<TARGET_FILE_DIR:${_dinoml_rocm_target}>"
        VERBATIM
      )
    endforeach()
  endif()

  if(DINOML_ENABLE_CK_GEMM)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    option(DINOML_ENABLE_CK_GEMM_PROFILER "Build CK GEMM standalone profiler modules" OFF)
    set(DINOML_CK_GEMM_PROFILER_TARGETS "" CACHE STRING "CK GEMM profiler targets or ALL")
    set(DINOML_CK_GEMM_OPS
      gemm_rcr gemm_rrr
      gemm_rcr_bias gemm_rrr_bias
      gemm_rcr_bias_relu gemm_rrr_bias_relu
      gemm_rcr_bias_gelu gemm_rrr_bias_gelu
      gemm_rcr_bias_fast_gelu gemm_rrr_bias_fast_gelu
      gemm_rcr_bias_quick_gelu
      gemm_rcr_bias_sigmoid gemm_rrr_bias_sigmoid
      gemm_rcr_bias_tanh gemm_rrr_bias_tanh
      gemm_rcr_bias_swish gemm_rrr_bias_swish
      gemm_rcr_bias_hardswish gemm_rrr_bias_hardswish
      gemm_rcr_bias_elup1 gemm_rrr_bias_elup1
      gemm_rcr_bias_add gemm_rrr_bias_add
      gemm_rcr_bias_add_add gemm_rrr_bias_add_add
      gemm_rcr_bias_mul gemm_rrr_bias_mul
      gemm_rcr_bias_mul_add gemm_rrr_bias_mul_add
      gemm_rcr_bias_add_relu gemm_rrr_bias_add_relu
      gemm_rcr_bias_add_add_relu gemm_rrr_bias_add_add_relu
      gemm_rcr_bias_sigmoid_mul gemm_rrr_bias_sigmoid_mul
      gemm_rcr_bias_sigmoid_mul_tanh gemm_rrr_bias_sigmoid_mul_tanh
      gemm_rcr_bias_mul_tanh gemm_rrr_bias_mul_tanh
      CACHE STRING "CK GEMM ops to expose as op/dtype static archives"
    )
    set(DINOML_CK_GEMM_DTYPES
      float32 float16 bfloat16
      CACHE STRING "CK GEMM dtypes to expose as op/dtype static archives"
    )
    set(DINOML_CK_GEMM_GENERATOR_DEPS
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_gemm_unit.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_gemm.hip
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/gemm.py
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/families/gemm.py
    )
    set(CK_ENABLE_FP16 ON)
    set(CK_ENABLE_BF16 ON)
    set(CK_ENABLE_FP32 ON)
    set(CK_USE_XDL ON)
    set(_dinoml_ck_config_include "${CMAKE_CURRENT_BINARY_DIR}/ck_config/include")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in
      ${_dinoml_ck_config_include}/ck/config.h
      @ONLY
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${DINOML_CK_GEMM_GENERATOR_DEPS})
    set(DINOML_CK_GEMM_TARGETS)
    foreach(_ck_gemm_op IN LISTS DINOML_CK_GEMM_OPS)
      foreach(_ck_gemm_dtype IN LISTS DINOML_CK_GEMM_DTYPES)
        set(_ck_gemm_group_id "${_ck_gemm_op}_${_ck_gemm_dtype}")
        set(_ck_gemm_target "dinoml_ck_gemm_${_ck_gemm_group_id}")
        set(_ck_gemm_source "${CMAKE_CURRENT_BINARY_DIR}/ck_gemm_units/${_ck_gemm_group_id}.hip")
        add_custom_command(
          OUTPUT ${_ck_gemm_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_gemm_unit.py
            --op ${_ck_gemm_op}
            --dtype ${_ck_gemm_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_gemm.hip
            --output ${_ck_gemm_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${DINOML_CK_GEMM_GENERATOR_DEPS}
          VERBATIM
        )
        add_library(${_ck_gemm_target} STATIC ${_ck_gemm_source})
        target_include_directories(${_ck_gemm_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/include
            ${_dinoml_ck_config_include}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
        )
        target_compile_definitions(${_ck_gemm_target}
          PRIVATE
            DINOML_HIP=1
            CK_ENABLE_FP16
            CK_ENABLE_BF16
            CK_ENABLE_FP32
            CK_USE_XDL
            CK_TIME_KERNEL=0
            $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
        )
        set_target_properties(${_ck_gemm_target} PROPERTIES
          OUTPUT_NAME "dinoml_ck_${_ck_gemm_group_id}"
          HIP_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CK_GEMM_TARGETS ${_ck_gemm_target})
      endforeach()
    endforeach()
    add_custom_target(dinoml_ck_gemm DEPENDS ${DINOML_CK_GEMM_TARGETS})
    if(DINOML_ENABLE_CK_GEMM_PROFILER)
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
      set(DINOML_CK_GEMM_PROFILER_LINK_TARGETS)
      if(DINOML_CK_GEMM_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CK_GEMM_PROFILER_LINK_TARGETS ${DINOML_CK_GEMM_TARGETS})
      elseif(DINOML_CK_GEMM_PROFILER_TARGETS)
        foreach(_ck_gemm_profiler_target IN LISTS DINOML_CK_GEMM_PROFILER_TARGETS)
          if(NOT TARGET ${_ck_gemm_profiler_target})
            message(FATAL_ERROR "Unknown CK GEMM profiler target: ${_ck_gemm_profiler_target}")
          endif()
          list(APPEND DINOML_CK_GEMM_PROFILER_LINK_TARGETS ${_ck_gemm_profiler_target})
        endforeach()
      endif()

      set(DINOML_CK_GEMM_PROFILER_TARGETS_BUILT)
      foreach(_ck_gemm_target IN LISTS DINOML_CK_GEMM_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_ck_gemm_" "" _ck_gemm_profiler_group_id "${_ck_gemm_target}")
        if(NOT _ck_gemm_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CK GEMM profiler op/dtype from target: ${_ck_gemm_target}")
        endif()
        set(_ck_gemm_profiler_op "${CMAKE_MATCH_1}")
        set(_ck_gemm_profiler_dtype "${CMAKE_MATCH_2}")
        set(_ck_gemm_profiler_output "dinoml_ck_gemm_profiler_${_ck_gemm_profiler_group_id}")
        set(_ck_gemm_profiler_bind_target "${_ck_gemm_profiler_output}_bind")
        set(_ck_gemm_profiler_exe_target "${_ck_gemm_profiler_output}")
        set(_ck_gemm_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/ck_gemm_profiler_dispatch/${_ck_gemm_profiler_group_id}.cpp"
        )
        add_custom_command(
          OUTPUT ${_ck_gemm_profiler_dispatch_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_gemm_profiler_dispatch.py
            --op ${_ck_gemm_profiler_op}
            --dtype ${_ck_gemm_profiler_dtype}
            --output ${_ck_gemm_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_gemm_profiler_dispatch.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ck_gemm_profiler_core.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/gemm.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/families/gemm.py
          VERBATIM
        )
        set(_ck_gemm_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_ck_gemm_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_ck_gemm_profiler_bind_target} MODULE
          tools/ck_gemm_profiler_pybind.cpp
          ${_ck_gemm_profiler_dispatch_source}
        )
        target_include_directories(${_ck_gemm_profiler_bind_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_compile_definitions(${_ck_gemm_profiler_bind_target}
          PRIVATE
            DINOML_CK_GEMM_PROFILER_PY_MODULE=${_ck_gemm_profiler_output}_bind
        )
        target_link_libraries(${_ck_gemm_profiler_bind_target}
          PRIVATE pybind11::module hip::host ${CMAKE_DL_LIBS}
        )
        target_link_options(${_ck_gemm_profiler_bind_target} PRIVATE ${_ck_gemm_profiler_linker_flags})
        set_target_properties(${_ck_gemm_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_ck_gemm_profiler_output}_bind"
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        if(WIN32)
          set_target_properties(${_ck_gemm_profiler_bind_target} PROPERTIES SUFFIX ".pyd")
        endif()

        add_executable(${_ck_gemm_profiler_exe_target}
          tools/ck_gemm_profiler_main.cpp
          ${_ck_gemm_profiler_dispatch_source}
        )
        target_include_directories(${_ck_gemm_profiler_exe_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_link_libraries(${_ck_gemm_profiler_exe_target} PRIVATE hip::host ${CMAKE_DL_LIBS})
        target_link_options(${_ck_gemm_profiler_exe_target} PRIVATE ${_ck_gemm_profiler_linker_flags})
        set_target_properties(${_ck_gemm_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_ck_gemm_profiler_output}"
          ENABLE_EXPORTS ON
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_ck_gemm_profiler_bind_target} ${_ck_gemm_target})
        add_dependencies(${_ck_gemm_profiler_exe_target} ${_ck_gemm_target})
        list(APPEND DINOML_CK_GEMM_PROFILER_TARGETS_BUILT
          ${_ck_gemm_profiler_bind_target}
          ${_ck_gemm_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CK_GEMM_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_ck_gemm_profilers DEPENDS ${DINOML_CK_GEMM_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()

  option(DINOML_ENABLE_CK_GGUF_Q8_GEMM "Build prototype fused GGUF Q8_0 gemm_rcr ROCm archive" OFF)
  if(DINOML_ENABLE_CK_GGUF_Q8_GEMM)
    set(CK_ENABLE_FP16 ON)
    set(CK_ENABLE_FP32 ON)
    set(CK_USE_XDL ON)
    set(_dinoml_ck_q8_config_include "${CMAKE_CURRENT_BINARY_DIR}/ck_q8_config/include")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in
      ${_dinoml_ck_q8_config_include}/ck/config.h
      @ONLY
    )

    add_library(dinoml_ck_gguf_q8_gemm STATIC
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_gguf_q8_gemm.hip
    )
    target_include_directories(dinoml_ck_gguf_q8_gemm
      PRIVATE
        ${_dinoml_ck_q8_config_include}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
    )
    target_compile_definitions(dinoml_ck_gguf_q8_gemm
      PRIVATE
        DINOML_HIP=1
        CK_ENABLE_FP16
        CK_ENABLE_FP32
        CK_USE_XDL
        $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
    )
    set_target_properties(dinoml_ck_gguf_q8_gemm PROPERTIES
      OUTPUT_NAME "dinoml_ck_gguf_q8_gemm"
      HIP_SEPARABLE_COMPILATION OFF
    )

    add_executable(dinoml_ck_gguf_q8_bench
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench_ck_gguf_q8_gemm.hip
    )
    target_include_directories(dinoml_ck_gguf_q8_bench
      PRIVATE
        ${_dinoml_ck_q8_config_include}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
    )
    target_link_libraries(dinoml_ck_gguf_q8_bench
      PRIVATE
        dinoml_ck_gguf_q8_gemm
    )
    target_compile_definitions(dinoml_ck_gguf_q8_bench
      PRIVATE
        DINOML_HIP=1
        CK_ENABLE_FP16
        CK_ENABLE_FP32
        CK_USE_XDL
        $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
    )
    set_target_properties(dinoml_ck_gguf_q8_bench PROPERTIES
      HIP_SEPARABLE_COMPILATION OFF
    )
  endif()

  if(DINOML_ENABLE_CK_BMM)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    option(DINOML_ENABLE_CK_BMM_PROFILER "Build CK BMM standalone profiler modules" OFF)
    set(DINOML_CK_BMM_PROFILER_TARGETS "" CACHE STRING "CK BMM profiler targets or ALL")
    set(DINOML_CK_BMM_OPS
      bmm_ccc bmm_ccr bmm_crc bmm_crr bmm_rcc bmm_rcr bmm_rrc bmm_rrr
      bmm_ccc_add bmm_ccr_add bmm_crc_add bmm_crr_add bmm_rcc_add bmm_rcr_add bmm_rrc_add bmm_rrr_add
      CACHE STRING "CK BMM ops to expose as op/dtype static archives"
    )
    set(DINOML_CK_BMM_DTYPES
      float32 float16 bfloat16
      CACHE STRING "CK BMM dtypes to expose as op/dtype static archives"
    )
    set(DINOML_CK_BMM_GENERATOR_DEPS
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_bmm_unit.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_bmm.hip
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/bmm.py
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/families/bmm.py
    )
    set(CK_ENABLE_FP16 ON)
    set(CK_ENABLE_BF16 ON)
    set(CK_ENABLE_FP32 ON)
    set(CK_USE_XDL ON)
    set(_dinoml_ck_config_include "${CMAKE_CURRENT_BINARY_DIR}/ck_config/include")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in
      ${_dinoml_ck_config_include}/ck/config.h
      @ONLY
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${DINOML_CK_BMM_GENERATOR_DEPS})
    set(DINOML_CK_BMM_TARGETS)
    foreach(_ck_bmm_op IN LISTS DINOML_CK_BMM_OPS)
      foreach(_ck_bmm_dtype IN LISTS DINOML_CK_BMM_DTYPES)
        set(_ck_bmm_group_id "${_ck_bmm_op}_${_ck_bmm_dtype}")
        set(_ck_bmm_target "dinoml_ck_bmm_${_ck_bmm_group_id}")
        set(_ck_bmm_source "${CMAKE_CURRENT_BINARY_DIR}/ck_bmm_units/${_ck_bmm_group_id}.hip")
        add_custom_command(
          OUTPUT ${_ck_bmm_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_bmm_unit.py
            --op ${_ck_bmm_op}
            --dtype ${_ck_bmm_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_bmm.hip
            --output ${_ck_bmm_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${DINOML_CK_BMM_GENERATOR_DEPS}
          VERBATIM
        )
        add_library(${_ck_bmm_target} STATIC ${_ck_bmm_source})
        target_include_directories(${_ck_bmm_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/include
            ${_dinoml_ck_config_include}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
        )
        target_compile_definitions(${_ck_bmm_target}
          PRIVATE
            DINOML_HIP=1
            CK_ENABLE_FP16
            CK_ENABLE_BF16
            CK_ENABLE_FP32
            CK_USE_XDL
            CK_TIME_KERNEL=0
            $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
        )
        set_target_properties(${_ck_bmm_target} PROPERTIES
          OUTPUT_NAME "dinoml_ck_${_ck_bmm_group_id}"
          HIP_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CK_BMM_TARGETS ${_ck_bmm_target})
      endforeach()
    endforeach()
    add_custom_target(dinoml_ck_bmm DEPENDS ${DINOML_CK_BMM_TARGETS})
    if(DINOML_ENABLE_CK_BMM_PROFILER)
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
      set(DINOML_CK_BMM_PROFILER_LINK_TARGETS)
      if(DINOML_CK_BMM_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CK_BMM_PROFILER_LINK_TARGETS ${DINOML_CK_BMM_TARGETS})
      elseif(DINOML_CK_BMM_PROFILER_TARGETS)
        foreach(_ck_bmm_profiler_target IN LISTS DINOML_CK_BMM_PROFILER_TARGETS)
          if(NOT TARGET ${_ck_bmm_profiler_target})
            message(FATAL_ERROR "Unknown CK BMM profiler target: ${_ck_bmm_profiler_target}")
          endif()
          list(APPEND DINOML_CK_BMM_PROFILER_LINK_TARGETS ${_ck_bmm_profiler_target})
        endforeach()
      endif()

      set(DINOML_CK_BMM_PROFILER_TARGETS_BUILT)
      foreach(_ck_bmm_target IN LISTS DINOML_CK_BMM_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_ck_bmm_" "" _ck_bmm_profiler_group_id "${_ck_bmm_target}")
        if(NOT _ck_bmm_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CK BMM profiler op/dtype from target: ${_ck_bmm_target}")
        endif()
        set(_ck_bmm_profiler_op "${CMAKE_MATCH_1}")
        set(_ck_bmm_profiler_dtype "${CMAKE_MATCH_2}")
        set(_ck_bmm_profiler_output "dinoml_ck_bmm_profiler_${_ck_bmm_profiler_group_id}")
        set(_ck_bmm_profiler_bind_target "${_ck_bmm_profiler_output}_bind")
        set(_ck_bmm_profiler_exe_target "${_ck_bmm_profiler_output}")
        set(_ck_bmm_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/ck_bmm_profiler_dispatch/${_ck_bmm_profiler_group_id}.cpp"
        )
        add_custom_command(
          OUTPUT ${_ck_bmm_profiler_dispatch_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_bmm_profiler_dispatch.py
            --op ${_ck_bmm_profiler_op}
            --dtype ${_ck_bmm_profiler_dtype}
            --output ${_ck_bmm_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_bmm_profiler_dispatch.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ck_bmm_profiler_core.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/bmm.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/families/bmm.py
          VERBATIM
        )
        set(_ck_bmm_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_ck_bmm_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_ck_bmm_profiler_bind_target} MODULE
          tools/ck_bmm_profiler_pybind.cpp
          ${_ck_bmm_profiler_dispatch_source}
        )
        target_include_directories(${_ck_bmm_profiler_bind_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_compile_definitions(${_ck_bmm_profiler_bind_target}
          PRIVATE
            DINOML_CK_BMM_PROFILER_PY_MODULE=${_ck_bmm_profiler_output}_bind
        )
        target_link_libraries(${_ck_bmm_profiler_bind_target}
          PRIVATE pybind11::module hip::host ${CMAKE_DL_LIBS}
        )
        target_link_options(${_ck_bmm_profiler_bind_target} PRIVATE ${_ck_bmm_profiler_linker_flags})
        set_target_properties(${_ck_bmm_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_ck_bmm_profiler_output}_bind"
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        if(WIN32)
          set_target_properties(${_ck_bmm_profiler_bind_target} PROPERTIES SUFFIX ".pyd")
        endif()

        add_executable(${_ck_bmm_profiler_exe_target}
          tools/ck_bmm_profiler_main.cpp
          ${_ck_bmm_profiler_dispatch_source}
        )
        target_include_directories(${_ck_bmm_profiler_exe_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_link_libraries(${_ck_bmm_profiler_exe_target} PRIVATE hip::host ${CMAKE_DL_LIBS})
        target_link_options(${_ck_bmm_profiler_exe_target} PRIVATE ${_ck_bmm_profiler_linker_flags})
        set_target_properties(${_ck_bmm_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_ck_bmm_profiler_output}"
          ENABLE_EXPORTS ON
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_ck_bmm_profiler_bind_target} ${_ck_bmm_target})
        add_dependencies(${_ck_bmm_profiler_exe_target} ${_ck_bmm_target})
        list(APPEND DINOML_CK_BMM_PROFILER_TARGETS_BUILT
          ${_ck_bmm_profiler_bind_target}
          ${_ck_bmm_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CK_BMM_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_ck_bmm_profilers DEPENDS ${DINOML_CK_BMM_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()

  if(DINOML_ENABLE_CK_CONV)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    option(DINOML_ENABLE_CK_CONV_PROFILER "Build CK Conv standalone profiler modules" OFF)
    set(DINOML_CK_CONV_PROFILER_TARGETS "" CACHE STRING "CK Conv profiler targets or ALL")
    set(DINOML_CK_CONV_OPS
      conv2d_bias
      conv2d_bias_relu
      conv2d_bias_add
      conv2d_bias_add_relu
      CACHE STRING "CK Conv ops to expose as op/dtype static archives"
    )
    set(DINOML_CK_CONV_DTYPES
      float32 float16 bfloat16
      CACHE STRING "CK Conv dtypes to expose as op/dtype static archives"
    )
    set(DINOML_CK_CONV_GENERATOR_DEPS
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_conv_unit.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_conv.hip
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/conv.py
    )
    set(CK_ENABLE_FP16 ON)
    set(CK_ENABLE_BF16 ON)
    set(CK_ENABLE_FP32 ON)
    set(CK_USE_XDL ON)
    set(_dinoml_ck_config_include "${CMAKE_CURRENT_BINARY_DIR}/ck_config/include")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in
      ${_dinoml_ck_config_include}/ck/config.h
      @ONLY
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${DINOML_CK_CONV_GENERATOR_DEPS})
    set(DINOML_CK_CONV_TARGETS)
    foreach(_ck_conv_op IN LISTS DINOML_CK_CONV_OPS)
      foreach(_ck_conv_dtype IN LISTS DINOML_CK_CONV_DTYPES)
        set(_ck_conv_group_id "${_ck_conv_op}_${_ck_conv_dtype}")
        set(_ck_conv_target "dinoml_ck_conv_${_ck_conv_group_id}")
        set(_ck_conv_source "${CMAKE_CURRENT_BINARY_DIR}/ck_conv_units/${_ck_conv_group_id}.hip")
        add_custom_command(
          OUTPUT ${_ck_conv_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_conv_unit.py
            --op ${_ck_conv_op}
            --dtype ${_ck_conv_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/ck_conv.hip
            --output ${_ck_conv_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${DINOML_CK_CONV_GENERATOR_DEPS}
          VERBATIM
        )
        add_library(${_ck_conv_target} STATIC ${_ck_conv_source})
        target_include_directories(${_ck_conv_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/include
            ${_dinoml_ck_config_include}
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
        )
        target_compile_definitions(${_ck_conv_target}
          PRIVATE
            DINOML_HIP=1
            CK_ENABLE_FP16
            CK_ENABLE_BF16
            CK_ENABLE_FP32
            CK_USE_XDL
            CK_TIME_KERNEL=0
            $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
        )
        set_target_properties(${_ck_conv_target} PROPERTIES
          OUTPUT_NAME "dinoml_ck_${_ck_conv_group_id}"
          HIP_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CK_CONV_TARGETS ${_ck_conv_target})
      endforeach()
    endforeach()
    add_custom_target(dinoml_ck_conv DEPENDS ${DINOML_CK_CONV_TARGETS})
    if(DINOML_ENABLE_CK_CONV_PROFILER)
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
      set(DINOML_CK_CONV_PROFILER_LINK_TARGETS)
      if(DINOML_CK_CONV_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CK_CONV_PROFILER_LINK_TARGETS ${DINOML_CK_CONV_TARGETS})
      elseif(DINOML_CK_CONV_PROFILER_TARGETS)
        foreach(_ck_conv_profiler_target IN LISTS DINOML_CK_CONV_PROFILER_TARGETS)
          if(NOT TARGET ${_ck_conv_profiler_target})
            message(FATAL_ERROR "Unknown CK Conv profiler target: ${_ck_conv_profiler_target}")
          endif()
          list(APPEND DINOML_CK_CONV_PROFILER_LINK_TARGETS ${_ck_conv_profiler_target})
        endforeach()
      endif()

      set(DINOML_CK_CONV_PROFILER_TARGETS_BUILT)
      foreach(_ck_conv_target IN LISTS DINOML_CK_CONV_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_ck_conv_" "" _ck_conv_profiler_group_id "${_ck_conv_target}")
        if(NOT _ck_conv_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CK Conv profiler op/dtype from target: ${_ck_conv_target}")
        endif()
        set(_ck_conv_profiler_op "${CMAKE_MATCH_1}")
        set(_ck_conv_profiler_dtype "${CMAKE_MATCH_2}")
        set(_ck_conv_profiler_output "dinoml_ck_conv_profiler_${_ck_conv_profiler_group_id}")
        set(_ck_conv_profiler_bind_target "${_ck_conv_profiler_output}_bind")
        set(_ck_conv_profiler_exe_target "${_ck_conv_profiler_output}")
        set(_ck_conv_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/ck_conv_profiler_dispatch/${_ck_conv_profiler_group_id}.cpp"
        )
        add_custom_command(
          OUTPUT ${_ck_conv_profiler_dispatch_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_conv_profiler_dispatch.py
            --op ${_ck_conv_profiler_op}
            --dtype ${_ck_conv_profiler_dtype}
            --output ${_ck_conv_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_ck_conv_profiler_dispatch.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/ck_conv_profiler_core.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/ck/conv.py
          VERBATIM
        )
        set(_ck_conv_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_ck_conv_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_ck_conv_profiler_bind_target} MODULE
          tools/ck_conv_profiler_pybind.cpp
          ${_ck_conv_profiler_dispatch_source}
        )
        target_include_directories(${_ck_conv_profiler_bind_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_compile_definitions(${_ck_conv_profiler_bind_target}
          PRIVATE
            DINOML_CK_CONV_PROFILER_PY_MODULE=${_ck_conv_profiler_output}_bind
        )
        target_link_libraries(${_ck_conv_profiler_bind_target}
          PRIVATE pybind11::module hip::host ${CMAKE_DL_LIBS}
        )
        target_link_options(${_ck_conv_profiler_bind_target} PRIVATE ${_ck_conv_profiler_linker_flags})
        set_target_properties(${_ck_conv_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_ck_conv_profiler_output}_bind"
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        if(WIN32)
          set_target_properties(${_ck_conv_profiler_bind_target} PROPERTIES SUFFIX ".pyd")
        endif()

        add_executable(${_ck_conv_profiler_exe_target}
          tools/ck_conv_profiler_main.cpp
          ${_ck_conv_profiler_dispatch_source}
        )
        target_include_directories(${_ck_conv_profiler_exe_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tools
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        )
        target_link_libraries(${_ck_conv_profiler_exe_target} PRIVATE hip::host ${CMAKE_DL_LIBS})
        target_link_options(${_ck_conv_profiler_exe_target} PRIVATE ${_ck_conv_profiler_linker_flags})
        set_target_properties(${_ck_conv_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_ck_conv_profiler_output}"
          ENABLE_EXPORTS ON
          LINKER_LANGUAGE HIP
          HIP_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_ck_conv_profiler_bind_target} ${_ck_conv_target})
        add_dependencies(${_ck_conv_profiler_exe_target} ${_ck_conv_target})
        list(APPEND DINOML_CK_CONV_PROFILER_TARGETS_BUILT
          ${_ck_conv_profiler_bind_target}
          ${_ck_conv_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CK_CONV_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_ck_conv_profilers DEPENDS ${DINOML_CK_CONV_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()

  if(DINOML_ENABLE_FLASH_ATTN_CK)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(CK_ENABLE_FP16 ON)
    set(CK_ENABLE_BF16 ON)
    set(CK_ENABLE_FP32 ON)
    set(CK_ENABLE_FP64 ON)
    set(CK_USE_XDL ON)
    set(_dinoml_flash_attn_ck_config_include "${CMAKE_CURRENT_BINARY_DIR}/flash_attn_ck_config/include")
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include/ck/config.h.in
      ${_dinoml_flash_attn_ck_config_include}/ck/config.h
      @ONLY
    )
    set(_dinoml_flash_attn_ck_generator_dir "${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha")
    set(_dinoml_flash_attn_ck_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/fa_ck")
    set(_dinoml_flash_attn_ck_blob_list "${_dinoml_flash_attn_ck_generated_dir}/fwd_blob_list.txt")
    set(_dinoml_flash_attn_ck_splitkv_blob_list "${_dinoml_flash_attn_ck_generated_dir}/splitkv_blob_list.txt")
    set(_dinoml_flash_attn_ck_appendkv_blob_list "${_dinoml_flash_attn_ck_generated_dir}/appendkv_blob_list.txt")
    set(DINOML_FLASH_ATTN_CK_FILTER
        "*d64_bf16_batch*nlogits*bias*mask*nlse*ndropout*|*d128_bf16_batch*nlogits*bias*mask*nlse*ndropout*|*d64_fp16_batch*nlogits*bias*mask*nlse*ndropout*|*d128_fp16_batch*nlogits*bias*mask*nlse*ndropout*|*d64_bf16_group*nlogits*nbias*nmask*nlse*ndropout*|*d128_bf16_group*nlogits*nbias*nmask*nlse*ndropout*|*d64_fp16_group*nlogits*nbias*nmask*nlse*ndropout*|*d128_fp16_group*nlogits*nbias*nmask*nlse*ndropout*"
        CACHE STRING "CK FlashAttention generator filter")
    set(DINOML_FLASH_ATTN_CK_SPLITKV_FILTER
        "*batch*ps_*lse_nsquant*@*batch*pssk_nlogits_*bias_nmask_lse_nsquant_npagedkv*"
        CACHE STRING "CK FlashAttention split-KV generator filter")
    set(DINOML_FLASH_ATTN_CK_APPENDKV_FILTER
        "*vr_psskddv"
        CACHE STRING "CK FlashAttention append-KV generator filter")
    set(DINOML_FLASH_ATTN_CK_OPTDIMS
        "64,128"
        CACHE STRING "Comma-separated CK FlashAttention head dimensions to generate")
    set(DINOML_FLASH_ATTN_CK_TARGETS "" CACHE STRING "Comma-separated CK FlashAttention generator targets. Defaults to CMAKE_HIP_ARCHITECTURES.")
    if(DINOML_FLASH_ATTN_CK_TARGETS)
      set(_dinoml_flash_attn_ck_targets "${DINOML_FLASH_ATTN_CK_TARGETS}")
    elseif(CMAKE_HIP_ARCHITECTURES)
      string(REPLACE ";" "," _dinoml_flash_attn_ck_targets "${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_dinoml_flash_attn_ck_targets "gfx12")
    endif()
    file(MAKE_DIRECTORY "${_dinoml_flash_attn_ck_generated_dir}")
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd
        --receipt 600
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_FILTER}"
        --list_blobs "${_dinoml_flash_attn_ck_blob_list}"
      WORKING_DIRECTORY "${_dinoml_flash_attn_ck_generator_dir}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd_splitkv
        --receipt 600
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_SPLITKV_FILTER}"
        --list_blobs "${_dinoml_flash_attn_ck_splitkv_blob_list}"
      WORKING_DIRECTORY "${_dinoml_flash_attn_ck_generator_dir}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd_appendkv
        --receipt 2
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_APPENDKV_FILTER}"
        --list_blobs "${_dinoml_flash_attn_ck_appendkv_blob_list}"
      WORKING_DIRECTORY "${_dinoml_flash_attn_ck_generator_dir}"
        COMMAND_ERROR_IS_FATAL ANY
    )
    file(STRINGS "${_dinoml_flash_attn_ck_blob_list}" DINOML_FLASH_ATTN_CK_GENERATED_SOURCES)
    file(STRINGS "${_dinoml_flash_attn_ck_splitkv_blob_list}" DINOML_FLASH_ATTN_CK_SPLITKV_GENERATED_SOURCES)
    file(STRINGS "${_dinoml_flash_attn_ck_appendkv_blob_list}" DINOML_FLASH_ATTN_CK_APPENDKV_GENERATED_SOURCES)
    list(APPEND DINOML_FLASH_ATTN_CK_GENERATED_SOURCES
      ${DINOML_FLASH_ATTN_CK_SPLITKV_GENERATED_SOURCES}
      ${DINOML_FLASH_ATTN_CK_APPENDKV_GENERATED_SOURCES}
    )
    list(REMOVE_DUPLICATES DINOML_FLASH_ATTN_CK_GENERATED_SOURCES)
    set(_dinoml_flash_attn_ck_generated_sources_normalized)
    foreach(_dinoml_flash_attn_ck_source IN LISTS DINOML_FLASH_ATTN_CK_GENERATED_SOURCES)
      file(TO_CMAKE_PATH "${_dinoml_flash_attn_ck_source}" _dinoml_flash_attn_ck_source_normalized)
      list(APPEND _dinoml_flash_attn_ck_generated_sources_normalized "${_dinoml_flash_attn_ck_source_normalized}")
    endforeach()
    set(DINOML_FLASH_ATTN_CK_GENERATED_SOURCES ${_dinoml_flash_attn_ck_generated_sources_normalized})
    file(GLOB_RECURSE _DINOML_FLASH_ATTN_CK_GENERATOR_DEPENDS CONFIGURE_DEPENDS
      "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
      "${_dinoml_flash_attn_ck_generator_dir}/codegen/*.py"
    )
    add_custom_command(
      OUTPUT ${DINOML_FLASH_ATTN_CK_GENERATED_SOURCES}
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd
        --receipt 600
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_FILTER}"
        --output_dir "${_dinoml_flash_attn_ck_generated_dir}"
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd_splitkv
        --receipt 600
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_SPLITKV_FILTER}"
        --output_dir "${_dinoml_flash_attn_ck_generated_dir}"
      COMMAND
        ${Python3_EXECUTABLE}
        "${_dinoml_flash_attn_ck_generator_dir}/generate.py"
        --targets "${_dinoml_flash_attn_ck_targets}"
        --api fwd_appendkv
        --receipt 2
        --optdim "${DINOML_FLASH_ATTN_CK_OPTDIMS}"
        --filter "${DINOML_FLASH_ATTN_CK_APPENDKV_FILTER}"
        --output_dir "${_dinoml_flash_attn_ck_generated_dir}"
      DEPENDS ${_DINOML_FLASH_ATTN_CK_GENERATOR_DEPENDS}
      WORKING_DIRECTORY "${_dinoml_flash_attn_ck_generator_dir}"
      VERBATIM
    )
    set(DINOML_FLASH_ATTN_CK_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attn_ck/interface_src/flash_attn_dinoml.cpp
      ${DINOML_FLASH_ATTN_CK_GENERATED_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/src/flash_attn_ck_wrapper.cpp
    )
    set_source_files_properties(${DINOML_FLASH_ATTN_CK_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(dinoml_flash_attn_ck STATIC ${DINOML_FLASH_ATTN_CK_SOURCES})
    target_include_directories(dinoml_flash_attn_ck
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attn_ck
        ${_dinoml_flash_attn_ck_config_include}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/library/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha
    )
    target_link_libraries(dinoml_flash_attn_ck
      PRIVATE hip::host hip::device
    )
    target_compile_definitions(dinoml_flash_attn_ck
      PRIVATE
        DINOML_HIP=1
        _USE_MATH_DEFINES
        CK_TILE_FMHA_FWD_FAST_EXP2=1
        CK_ENABLE_FP16
        CK_ENABLE_BF16
        CK_ENABLE_FP32
        CK_ENABLE_FP64
        CK_USE_XDL
        USE_PROF_API=1
        CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
        CK_TILE_FMHA_FWD_SPLITKV_API=1
        CK_TILE_FMHA_FWD_APPENDKV_API=1
        CK_TILE_FMHA_FWD_PAGEDKV_API=0
        __HIP_PLATFORM_AMD__
        __HIPCC__
        $<$<PLATFORM_ID:Windows>:_ITERATOR_DEBUG_LEVEL=0>
    )
    target_compile_options(dinoml_flash_attn_ck
      PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-w>
        $<$<COMPILE_LANGUAGE:HIP>:-O3>
        $<$<COMPILE_LANGUAGE:HIP>:-fgpu-flush-denormals-to-zero>
        $<$<COMPILE_LANGUAGE:HIP>:-mllvm>
        $<$<COMPILE_LANGUAGE:HIP>:--lsr-drop-solution=1>
        $<$<COMPILE_LANGUAGE:HIP>:-fno-offload-uniform-block>
    )
    set_target_properties(dinoml_flash_attn_ck PROPERTIES
      OUTPUT_NAME "dinoml_flash_attn_ck"
      HIP_SEPARABLE_COMPILATION OFF
    )
  endif()
endif()

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
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(DINOML_CUTLASS_GEMM_CHUNK_SIZE LESS 1)
      message(FATAL_ERROR "DINOML_CUTLASS_GEMM_CHUNK_SIZE must be positive")
    endif()
    set(DINOML_CUTLASS_GEMM_OPS
      gemm_rcr gemm_rrr
      gemm_rcr_bias gemm_rrr_bias
      gemm_rcr_bias_relu gemm_rrr_bias_relu
      gemm_rcr_bias_gelu gemm_rrr_bias_gelu
      gemm_rcr_bias_fast_gelu gemm_rrr_bias_fast_gelu
      gemm_rcr_bias_quick_gelu
      gemm_rcr_bias_sigmoid gemm_rrr_bias_sigmoid
      gemm_rcr_bias_tanh gemm_rrr_bias_tanh
      gemm_rcr_bias_swish gemm_rrr_bias_swish
      gemm_rcr_bias_hardswish gemm_rrr_bias_hardswish
      gemm_rcr_bias_elup1 gemm_rrr_bias_elup1
      gemm_rcr_bias_add gemm_rrr_bias_add
      gemm_rcr_bias_add_add gemm_rrr_bias_add_add
      gemm_rcr_bias_mul gemm_rrr_bias_mul
      gemm_rcr_bias_mul_add gemm_rrr_bias_mul_add
      gemm_rcr_bias_add_relu gemm_rrr_bias_add_relu
      gemm_rcr_bias_add_add_relu gemm_rrr_bias_add_add_relu
      gemm_rcr_bias_sigmoid_mul gemm_rrr_bias_sigmoid_mul
      gemm_rcr_bias_sigmoid_mul_tanh gemm_rrr_bias_sigmoid_mul_tanh
      gemm_rcr_bias_mul_tanh gemm_rrr_bias_mul_tanh
    )
    set(DINOML_CUTLASS_GEMM_DTYPES float32 float16 bfloat16)
    set(DINOML_CUTLASS_GEMM_GENERATOR_DEPS
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_gemm_unit.py
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_gemm.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_common.cuh
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/gemm.py
      ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/families/gemm.py
    )
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${DINOML_CUTLASS_GEMM_GENERATOR_DEPS})
    set(_cutlass_gemm_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/cutlass_gemm_units")
    set(_cutlass_gemm_chunk_manifest "${_cutlass_gemm_generated_dir}/chunks.cmake")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_gemm_unit.py
        --ops ${DINOML_CUTLASS_GEMM_OPS}
        --dtypes ${DINOML_CUTLASS_GEMM_DTYPES}
        --chunk-size ${DINOML_CUTLASS_GEMM_CHUNK_SIZE}
        --write-cmake-chunk-manifest ${_cutlass_gemm_chunk_manifest}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMAND_ERROR_IS_FATAL ANY
    )
    include(${_cutlass_gemm_chunk_manifest})
    set(DINOML_CUTLASS_GEMM_TARGETS)
    foreach(_cutlass_gemm_op IN LISTS DINOML_CUTLASS_GEMM_OPS)
      foreach(_cutlass_gemm_dtype IN LISTS DINOML_CUTLASS_GEMM_DTYPES)
        set(_cutlass_gemm_group_id "${_cutlass_gemm_op}_${_cutlass_gemm_dtype}")
        set(_cutlass_gemm_target "dinoml_cutlass_gemm_${_cutlass_gemm_group_id}")
        set(_cutlass_gemm_chunk_var "DINOML_CUTLASS_GEMM_CHUNKS_${_cutlass_gemm_group_id}")
        set(_cutlass_gemm_chunk_files ${${_cutlass_gemm_chunk_var}})
        if(NOT _cutlass_gemm_chunk_files)
          message(FATAL_ERROR "CUTLASS GEMM generator returned no chunks for ${_cutlass_gemm_group_id}")
        endif()
        set(_cutlass_gemm_sources)
        foreach(_cutlass_gemm_chunk_file IN LISTS _cutlass_gemm_chunk_files)
          list(APPEND _cutlass_gemm_sources "${_cutlass_gemm_generated_dir}/${_cutlass_gemm_chunk_file}")
        endforeach()
        add_custom_command(
          OUTPUT ${_cutlass_gemm_sources}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_gemm_unit.py
            --op ${_cutlass_gemm_op}
            --dtype ${_cutlass_gemm_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_gemm.cu
            --output-dir ${_cutlass_gemm_generated_dir}
            --chunk-size ${DINOML_CUTLASS_GEMM_CHUNK_SIZE}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS ${DINOML_CUTLASS_GEMM_GENERATOR_DEPS}
          VERBATIM
        )
        add_library(${_cutlass_gemm_target} STATIC ${_cutlass_gemm_sources})
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
          OUTPUT_NAME "dinoml_cutlass_${_cutlass_gemm_group_id}"
          CUDA_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CUTLASS_GEMM_TARGETS ${_cutlass_gemm_target})
      endforeach()
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

  if(DINOML_ENABLE_CUTLASS_BMM)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(DINOML_CUTLASS_BMM_OPS
      bmm_ccc bmm_ccr bmm_crc bmm_crr bmm_rcc bmm_rcr bmm_rrc bmm_rrr
      bmm_ccc_add bmm_ccr_add bmm_crc_add bmm_crr_add bmm_rcc_add bmm_rcr_add bmm_rrc_add bmm_rrr_add
    )
    set(DINOML_CUTLASS_BMM_DTYPES float32 float16 bfloat16)
    set(DINOML_CUTLASS_BMM_TARGETS)
    foreach(_cutlass_bmm_op IN LISTS DINOML_CUTLASS_BMM_OPS)
      foreach(_cutlass_bmm_dtype IN LISTS DINOML_CUTLASS_BMM_DTYPES)
        set(_cutlass_bmm_group_id "${_cutlass_bmm_op}_${_cutlass_bmm_dtype}")
        set(_cutlass_bmm_target "dinoml_cutlass_bmm_${_cutlass_bmm_group_id}")
        set(_cutlass_bmm_source "${CMAKE_CURRENT_BINARY_DIR}/cutlass_bmm_units/${_cutlass_bmm_group_id}.cu")
        add_custom_command(
          OUTPUT ${_cutlass_bmm_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_bmm_unit.py
            --op ${_cutlass_bmm_op}
            --dtype ${_cutlass_bmm_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_bmm.cu
            --output ${_cutlass_bmm_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_bmm_unit.py
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_bmm.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/bmm.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/gemm.py
          VERBATIM
        )
        add_library(${_cutlass_bmm_target} STATIC ${_cutlass_bmm_source})
        target_include_directories(${_cutlass_bmm_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
        )
        target_compile_options(${_cutlass_bmm_target}
          PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
        )
        set_target_properties(${_cutlass_bmm_target} PROPERTIES
          OUTPUT_NAME "dinoml_cutlass_${_cutlass_bmm_group_id}"
          CUDA_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CUTLASS_BMM_TARGETS ${_cutlass_bmm_target})
      endforeach()
    endforeach()
    add_custom_target(dinoml_cutlass_bmm DEPENDS ${DINOML_CUTLASS_BMM_TARGETS})

    if(DINOML_ENABLE_CUTLASS_BMM_PROFILER)
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
      set(DINOML_CUTLASS_BMM_PROFILER_LINK_TARGETS)
      if(DINOML_CUTLASS_BMM_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CUTLASS_BMM_PROFILER_LINK_TARGETS ${DINOML_CUTLASS_BMM_TARGETS})
      elseif(DINOML_CUTLASS_BMM_PROFILER_TARGETS)
        foreach(_cutlass_bmm_profiler_target IN LISTS DINOML_CUTLASS_BMM_PROFILER_TARGETS)
          if(NOT TARGET ${_cutlass_bmm_profiler_target})
            message(FATAL_ERROR "Unknown CUTLASS BMM profiler target: ${_cutlass_bmm_profiler_target}")
          endif()
          list(APPEND DINOML_CUTLASS_BMM_PROFILER_LINK_TARGETS ${_cutlass_bmm_profiler_target})
        endforeach()
      endif()

      set(DINOML_CUTLASS_BMM_PROFILER_TARGETS_BUILT)
      foreach(_cutlass_bmm_target IN LISTS DINOML_CUTLASS_BMM_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_cutlass_bmm_" "" _cutlass_bmm_profiler_group_id "${_cutlass_bmm_target}")
        if(NOT _cutlass_bmm_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CUTLASS BMM profiler op/dtype from target: ${_cutlass_bmm_target}")
        endif()
        set(_cutlass_bmm_profiler_op "${CMAKE_MATCH_1}")
        set(_cutlass_bmm_profiler_dtype "${CMAKE_MATCH_2}")
        set(_cutlass_bmm_profiler_output "dinoml_cutlass_bmm_profiler_${_cutlass_bmm_profiler_group_id}")
        set(_cutlass_bmm_profiler_bind_target "${_cutlass_bmm_profiler_output}_bind")
        set(_cutlass_bmm_profiler_exe_target "${_cutlass_bmm_profiler_output}")
        set(_cutlass_bmm_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/cutlass_bmm_profiler_dispatch/${_cutlass_bmm_profiler_group_id}.cu"
        )
        add_custom_command(
          OUTPUT ${_cutlass_bmm_profiler_dispatch_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_bmm_profiler_dispatch.py
            --op ${_cutlass_bmm_profiler_op}
            --dtype ${_cutlass_bmm_profiler_dtype}
            --output ${_cutlass_bmm_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_bmm_profiler_dispatch.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/cutlass_bmm_profiler_core.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/bmm.py
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/gemm.py
          VERBATIM
        )
        set(_cutlass_bmm_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_cutlass_bmm_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_cutlass_bmm_profiler_bind_target} MODULE
          tools/cutlass_bmm_profiler_pybind.cu
          ${_cutlass_bmm_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_bmm_profiler_bind_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
        target_link_libraries(${_cutlass_bmm_profiler_bind_target} PRIVATE pybind11::module CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${_cutlass_bmm_profiler_bind_target} PRIVATE ${_cutlass_bmm_profiler_linker_flags})
        target_compile_options(${_cutlass_bmm_profiler_bind_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(${_cutlass_bmm_profiler_bind_target}
          PRIVATE DINOML_CUTLASS_BMM_PROFILER_PY_MODULE=${_cutlass_bmm_profiler_bind_target}
        )
        set_target_properties(${_cutlass_bmm_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_cutlass_bmm_profiler_output}_bind"
          CUDA_SEPARABLE_COMPILATION OFF
        )

        add_executable(${_cutlass_bmm_profiler_exe_target}
          tools/cutlass_bmm_profiler_main.cu
          ${_cutlass_bmm_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_bmm_profiler_exe_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
        target_link_libraries(${_cutlass_bmm_profiler_exe_target} PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${_cutlass_bmm_profiler_exe_target} PRIVATE ${_cutlass_bmm_profiler_linker_flags})
        target_compile_options(${_cutlass_bmm_profiler_exe_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(${_cutlass_bmm_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_cutlass_bmm_profiler_output}"
          ENABLE_EXPORTS ON
          CUDA_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_cutlass_bmm_profiler_bind_target} ${_cutlass_bmm_target})
        add_dependencies(${_cutlass_bmm_profiler_exe_target} ${_cutlass_bmm_target})
        list(APPEND DINOML_CUTLASS_BMM_PROFILER_TARGETS_BUILT
          ${_cutlass_bmm_profiler_bind_target}
          ${_cutlass_bmm_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CUTLASS_BMM_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_cutlass_bmm_profilers DEPENDS ${DINOML_CUTLASS_BMM_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()

  if(DINOML_ENABLE_CUTLASS_CONV)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(DINOML_CUTLASS_CONV_OPS
      conv2d_bias conv2d_bias_relu conv2d_bias_add conv2d_bias_add_relu
    )
    set(DINOML_CUTLASS_CONV_DTYPES float32 float16 bfloat16)
    set(DINOML_CUTLASS_CONV_TARGETS)
    foreach(_cutlass_conv_op IN LISTS DINOML_CUTLASS_CONV_OPS)
      foreach(_cutlass_conv_dtype IN LISTS DINOML_CUTLASS_CONV_DTYPES)
        set(_cutlass_conv_group_id "${_cutlass_conv_op}_${_cutlass_conv_dtype}")
        set(_cutlass_conv_target "dinoml_cutlass_conv_${_cutlass_conv_group_id}")
        set(_cutlass_conv_source "${CMAKE_CURRENT_BINARY_DIR}/cutlass_conv_units/${_cutlass_conv_group_id}.cu")
        add_custom_command(
          OUTPUT ${_cutlass_conv_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_conv_unit.py
            --op ${_cutlass_conv_op}
            --dtype ${_cutlass_conv_dtype}
            --source ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_conv.cu
            --output ${_cutlass_conv_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_conv_unit.py
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src/cutlass_conv.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/conv.py
          VERBATIM
        )
        add_library(${_cutlass_conv_target} STATIC ${_cutlass_conv_source})
        target_include_directories(${_cutlass_conv_target}
          PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/src
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
        )
        target_compile_options(${_cutlass_conv_target}
          PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
        )
        set_target_properties(${_cutlass_conv_target} PROPERTIES
          OUTPUT_NAME "dinoml_cutlass_${_cutlass_conv_group_id}"
          CUDA_SEPARABLE_COMPILATION OFF
        )
        list(APPEND DINOML_CUTLASS_CONV_TARGETS ${_cutlass_conv_target})
      endforeach()
    endforeach()
    add_custom_target(dinoml_cutlass_conv DEPENDS ${DINOML_CUTLASS_CONV_TARGETS})

    if(DINOML_ENABLE_CUTLASS_CONV_PROFILER)
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
      set(DINOML_CUTLASS_CONV_PROFILER_LINK_TARGETS)
      if(DINOML_CUTLASS_CONV_PROFILER_TARGETS STREQUAL "ALL")
        set(DINOML_CUTLASS_CONV_PROFILER_LINK_TARGETS ${DINOML_CUTLASS_CONV_TARGETS})
      elseif(DINOML_CUTLASS_CONV_PROFILER_TARGETS)
        foreach(_cutlass_conv_profiler_target IN LISTS DINOML_CUTLASS_CONV_PROFILER_TARGETS)
          if(NOT TARGET ${_cutlass_conv_profiler_target})
            message(FATAL_ERROR "Unknown CUTLASS Conv profiler target: ${_cutlass_conv_profiler_target}")
          endif()
          list(APPEND DINOML_CUTLASS_CONV_PROFILER_LINK_TARGETS ${_cutlass_conv_profiler_target})
        endforeach()
      endif()

      set(DINOML_CUTLASS_CONV_PROFILER_TARGETS_BUILT)
      foreach(_cutlass_conv_target IN LISTS DINOML_CUTLASS_CONV_PROFILER_LINK_TARGETS)
        string(REGEX REPLACE "^dinoml_cutlass_conv_" "" _cutlass_conv_profiler_group_id "${_cutlass_conv_target}")
        if(NOT _cutlass_conv_profiler_group_id MATCHES "^(.+)_(float32|float16|bfloat16)$")
          message(FATAL_ERROR "Could not derive CUTLASS Conv profiler op/dtype from target: ${_cutlass_conv_target}")
        endif()
        set(_cutlass_conv_profiler_op "${CMAKE_MATCH_1}")
        set(_cutlass_conv_profiler_dtype "${CMAKE_MATCH_2}")
        set(_cutlass_conv_profiler_output "dinoml_cutlass_conv_profiler_${_cutlass_conv_profiler_group_id}")
        set(_cutlass_conv_profiler_bind_target "${_cutlass_conv_profiler_output}_bind")
        set(_cutlass_conv_profiler_exe_target "${_cutlass_conv_profiler_output}")
        set(_cutlass_conv_profiler_dispatch_source
          "${CMAKE_CURRENT_BINARY_DIR}/cutlass_conv_profiler_dispatch/${_cutlass_conv_profiler_group_id}.cu"
        )
        add_custom_command(
          OUTPUT ${_cutlass_conv_profiler_dispatch_source}
          COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_conv_profiler_dispatch.py
            --op ${_cutlass_conv_profiler_op}
            --dtype ${_cutlass_conv_profiler_dtype}
            --output ${_cutlass_conv_profiler_dispatch_source}
          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
          DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_cutlass_conv_profiler_dispatch.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/cutlass_conv_profiler_core.cuh
            ${CMAKE_CURRENT_SOURCE_DIR}/src/dinoml/kernels/providers/cutlass/conv.py
          VERBATIM
        )
        set(_cutlass_conv_profiler_linker_flags
          "LINKER:--whole-archive"
          "$<TARGET_FILE:${_cutlass_conv_target}>"
          "LINKER:--no-whole-archive"
        )
        add_library(${_cutlass_conv_profiler_bind_target} MODULE
          tools/cutlass_conv_profiler_pybind.cu
          ${_cutlass_conv_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_conv_profiler_bind_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
        target_link_libraries(${_cutlass_conv_profiler_bind_target} PRIVATE pybind11::module CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${_cutlass_conv_profiler_bind_target} PRIVATE ${_cutlass_conv_profiler_linker_flags})
        target_compile_options(${_cutlass_conv_profiler_bind_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_compile_definitions(${_cutlass_conv_profiler_bind_target}
          PRIVATE DINOML_CUTLASS_CONV_PROFILER_PY_MODULE=${_cutlass_conv_profiler_bind_target}
        )
        set_target_properties(${_cutlass_conv_profiler_bind_target} PROPERTIES
          PREFIX ""
          OUTPUT_NAME "${_cutlass_conv_profiler_output}_bind"
          CUDA_SEPARABLE_COMPILATION OFF
        )

        add_executable(${_cutlass_conv_profiler_exe_target}
          tools/cutlass_conv_profiler_main.cu
          ${_cutlass_conv_profiler_dispatch_source}
        )
        target_include_directories(${_cutlass_conv_profiler_exe_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
        target_link_libraries(${_cutlass_conv_profiler_exe_target} PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_options(${_cutlass_conv_profiler_exe_target} PRIVATE ${_cutlass_conv_profiler_linker_flags})
        target_compile_options(${_cutlass_conv_profiler_exe_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(${_cutlass_conv_profiler_exe_target} PROPERTIES
          OUTPUT_NAME "${_cutlass_conv_profiler_output}"
          ENABLE_EXPORTS ON
          CUDA_SEPARABLE_COMPILATION OFF
        )
        add_dependencies(${_cutlass_conv_profiler_bind_target} ${_cutlass_conv_target})
        add_dependencies(${_cutlass_conv_profiler_exe_target} ${_cutlass_conv_target})
        list(APPEND DINOML_CUTLASS_CONV_PROFILER_TARGETS_BUILT
          ${_cutlass_conv_profiler_bind_target}
          ${_cutlass_conv_profiler_exe_target}
        )
      endforeach()
      if(DINOML_CUTLASS_CONV_PROFILER_TARGETS_BUILT)
        add_custom_target(dinoml_cutlass_conv_profilers DEPENDS ${DINOML_CUTLASS_CONV_PROFILER_TARGETS_BUILT})
      endif()
    endif()
  endif()

  if(DINOML_ENABLE_FLASH_ATTN_CUDA)
    add_subdirectory(third_party/flash_attn)
    if(CMAKE_CUDA_ARCHITECTURES)
      set_property(TARGET flash_attention PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    endif()
    add_library(dinoml_flash_attn_cuda STATIC
      kernels/cuda/src/flash_attn_cuda_wrapper.cu
    )
    target_include_directories(dinoml_flash_attn_cuda
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attn
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cuda/include
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    )
    target_link_libraries(dinoml_flash_attn_cuda PRIVATE flash_attention CUDA::cudart)
    target_compile_options(dinoml_flash_attn_cuda
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    )
    set_target_properties(dinoml_flash_attn_cuda PROPERTIES
      OUTPUT_NAME "dinoml_flash_attn_cuda"
      CUDA_SEPARABLE_COMPILATION OFF
      POSITION_INDEPENDENT_CODE ON
    )
  endif()

  if(DINOML_ENABLE_LIBGGUF_CUDA)
    set(DINOML_LIBGGUF_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libgguf)
    set(DINOML_LIBGGUF_CUDA_SRC ${DINOML_LIBGGUF_ROOT}/src/libgguf/libgguf_cuda/csrc)
    file(GLOB DINOML_LIBGGUF_CUDA_QUANT_SOURCES CONFIGURE_DEPENDS
      ${DINOML_LIBGGUF_CUDA_SRC}/cuda_quantize*.cu
    )
    file(GLOB DINOML_LIBGGUF_CUDA_DEQUANT_SOURCES CONFIGURE_DEPENDS
      ${DINOML_LIBGGUF_CUDA_SRC}/cuda_dequantize*.cu
    )
    list(REMOVE_ITEM DINOML_LIBGGUF_CUDA_QUANT_SOURCES
      ${DINOML_LIBGGUF_CUDA_SRC}/cuda_quantize.cu
    )
    list(REMOVE_ITEM DINOML_LIBGGUF_CUDA_DEQUANT_SOURCES
      ${DINOML_LIBGGUF_CUDA_SRC}/cuda_dequantize.cu
    )
    add_library(dinoml_libgguf_cuda_quant_objects OBJECT ${DINOML_LIBGGUF_CUDA_QUANT_SOURCES})
    set_target_properties(dinoml_libgguf_cuda_quant_objects PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(dinoml_libgguf_cuda_quant_objects
      PRIVATE
        ${DINOML_LIBGGUF_CUDA_SRC}
        ${DINOML_LIBGGUF_ROOT}/include
        ${DINOML_LIBGGUF_ROOT}/csrc/common
    )
    target_compile_definitions(dinoml_libgguf_cuda_quant_objects PRIVATE LIBGGUF_CUDA_NO_C10=1)
    target_compile_options(dinoml_libgguf_cuda_quant_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

    add_library(dinoml_libgguf_cuda_dequant_objects OBJECT ${DINOML_LIBGGUF_CUDA_DEQUANT_SOURCES})
    set_target_properties(dinoml_libgguf_cuda_dequant_objects PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(dinoml_libgguf_cuda_dequant_objects
      PRIVATE
        ${DINOML_LIBGGUF_CUDA_SRC}
        ${DINOML_LIBGGUF_ROOT}/include
        ${DINOML_LIBGGUF_ROOT}/csrc/common
    )
    target_compile_definitions(dinoml_libgguf_cuda_dequant_objects PRIVATE LIBGGUF_CUDA_NO_C10=1)
    target_compile_options(dinoml_libgguf_cuda_dequant_objects PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

    add_library(libgguf_cuda_native STATIC
      ${DINOML_LIBGGUF_CUDA_SRC}/libgguf_cuda_native.cu
      $<TARGET_OBJECTS:dinoml_libgguf_cuda_quant_objects>
      $<TARGET_OBJECTS:dinoml_libgguf_cuda_dequant_objects>
    )
    set_target_properties(libgguf_cuda_native PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      OUTPUT_NAME "gguf_cuda_native"
      CUDA_SEPARABLE_COMPILATION OFF
    )
    target_include_directories(libgguf_cuda_native
      PUBLIC
        ${DINOML_LIBGGUF_CUDA_SRC}
      PRIVATE
        ${DINOML_LIBGGUF_ROOT}/include
        ${DINOML_LIBGGUF_ROOT}/csrc/common
    )
    target_compile_definitions(libgguf_cuda_native PRIVATE LIBGGUF_CUDA_NO_C10=1)
    target_compile_options(libgguf_cuda_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(libgguf_cuda_native PUBLIC CUDA::cudart)
  endif()
endif()
