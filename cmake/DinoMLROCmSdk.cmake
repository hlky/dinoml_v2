# Resolve ROCm/HIP SDK layouts before CMake enables the HIP language.
#
# Provider order:
#   1. Explicit DINOML_ROCM_* cache variables.
#   2. Active pip/venv ROCm SDK via rocm-sdk or python -m rocm_sdk on PATH.
#   3. Regular HIP installs via hipconfig.
#   4. ROCM_PATH/HIP_PATH and platform defaults.

include_guard(GLOBAL)

function(_dinoml_rocm_to_cmake_path in_path out_path)
  if(in_path)
    file(TO_CMAKE_PATH "${in_path}" _converted)
    set(${out_path} "${_converted}" PARENT_SCOPE)
  else()
    set(${out_path} "" PARENT_SCOPE)
  endif()
endfunction()

function(_dinoml_rocm_run out_result out_output)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(${out_result} "${_result}" PARENT_SCOPE)
  set(${out_output} "${_output}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_windows_install_roots out_roots)
  if(NOT WIN32)
    set(${out_roots} "" PARENT_SCOPE)
    return()
  endif()

  set(_bases)
  if(DEFINED ENV{ProgramFiles} AND NOT "$ENV{ProgramFiles}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{ProgramFiles}/AMD/ROCm" _program_files_rocm)
    list(APPEND _bases "${_program_files_rocm}")
  endif()
  list(APPEND _bases "C:/Program Files/AMD/ROCm")
  list(REMOVE_DUPLICATES _bases)

  set(_roots)
  foreach(_base IN LISTS _bases)
    if(EXISTS "${_base}/bin/hipconfig.exe")
      list(APPEND _roots "${_base}")
    endif()
    if(EXISTS "${_base}")
      file(GLOB _children LIST_DIRECTORIES true "${_base}/*")
      if(_children)
        list(SORT _children COMPARE NATURAL ORDER DESCENDING)
      endif()
      foreach(_child IN LISTS _children)
        if(IS_DIRECTORY "${_child}" AND EXISTS "${_child}/bin/hipconfig.exe")
          list(APPEND _roots "${_child}")
        endif()
      endforeach()
    endif()
  endforeach()
  if(_roots)
    list(REMOVE_DUPLICATES _roots)
  endif()
  set(${out_roots} "${_roots}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_sdk out_rocm_sdk)
  find_program(_DINOML_ROCM_SDK NAMES rocm-sdk rocm_sdk)
  if(_DINOML_ROCM_SDK)
    set(${out_rocm_sdk} "${_DINOML_ROCM_SDK}" PARENT_SCOPE)
    return()
  endif()

  find_program(_DINOML_ROCM_PYTHON NAMES python python3)
  if(NOT _DINOML_ROCM_PYTHON)
    set(${out_rocm_sdk} "" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${_DINOML_ROCM_PYTHON}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('rocm_sdk') else 1)"
    RESULT_VARIABLE _rocm_sdk_module_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(_rocm_sdk_module_result EQUAL 0)
    set(${out_rocm_sdk} "${_DINOML_ROCM_PYTHON};-m;rocm_sdk" PARENT_SCOPE)
  else()
    set(${out_rocm_sdk} "" PARENT_SCOPE)
  endif()
endfunction()

function(_dinoml_rocm_try_rocm_sdk out_root out_cmake out_bin out_llvm_bin)
  _dinoml_rocm_sdk(_rocm_sdk)
  if(NOT _rocm_sdk)
    set(${out_root} "" PARENT_SCOPE)
    set(${out_cmake} "" PARENT_SCOPE)
    set(${out_bin} "" PARENT_SCOPE)
    set(${out_llvm_bin} "" PARENT_SCOPE)
    return()
  endif()

  execute_process(COMMAND ${_rocm_sdk} init ERROR_QUIET OUTPUT_QUIET)
  _dinoml_rocm_run(_root_result _root ${_rocm_sdk} path --root)
  _dinoml_rocm_run(_cmake_result _cmake ${_rocm_sdk} path --cmake)
  _dinoml_rocm_run(_bin_result _bin ${_rocm_sdk} path --bin)
  if(NOT _root_result EQUAL 0 OR NOT _cmake_result EQUAL 0 OR NOT _bin_result EQUAL 0)
    set(${out_root} "" PARENT_SCOPE)
    set(${out_cmake} "" PARENT_SCOPE)
    set(${out_bin} "" PARENT_SCOPE)
    set(${out_llvm_bin} "" PARENT_SCOPE)
    return()
  endif()

  _dinoml_rocm_to_cmake_path("${_root}" _root)
  _dinoml_rocm_to_cmake_path("${_cmake}" _cmake)
  _dinoml_rocm_to_cmake_path("${_bin}" _bin)

  set(_llvm_bin "${_root}/lib/llvm/bin")
  if(NOT EXISTS "${_llvm_bin}")
    set(_llvm_bin "")
  endif()

  set(${out_root} "${_root}" PARENT_SCOPE)
  set(${out_cmake} "${_cmake}" PARENT_SCOPE)
  set(${out_bin} "${_bin}" PARENT_SCOPE)
  set(${out_llvm_bin} "${_llvm_bin}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_try_hipconfig out_root out_cmake out_bin out_llvm_bin)
  _dinoml_rocm_windows_install_roots(_windows_rocm_roots)
  set(_windows_bin_hints)
  foreach(_root IN LISTS _windows_rocm_roots)
    list(APPEND _windows_bin_hints "${_root}/bin")
  endforeach()
  find_program(
    HIPCONFIG_EXECUTABLE
    NAMES hipconfig hipconfig.exe
    HINTS
      ${_windows_bin_hints}
      "$ENV{HIP_PATH}/bin"
      "$ENV{ROCM_PATH}/bin"
      "$ENV{HIP_PATH}"
      "$ENV{ROCM_PATH}"
  )
  if(NOT HIPCONFIG_EXECUTABLE)
    set(${out_root} "" PARENT_SCOPE)
    set(${out_cmake} "" PARENT_SCOPE)
    set(${out_bin} "" PARENT_SCOPE)
    set(${out_llvm_bin} "" PARENT_SCOPE)
    return()
  endif()

  _dinoml_rocm_run(_root_result _root "${HIPCONFIG_EXECUTABLE}" --rocmpath --newline)
  if(NOT _root_result EQUAL 0 OR NOT _root)
    _dinoml_rocm_run(_root_result _root "${HIPCONFIG_EXECUTABLE}" --path --newline)
  endif()
  if(NOT _root_result EQUAL 0 OR NOT _root)
    set(${out_root} "" PARENT_SCOPE)
    set(${out_cmake} "" PARENT_SCOPE)
    set(${out_bin} "" PARENT_SCOPE)
    set(${out_llvm_bin} "" PARENT_SCOPE)
    return()
  endif()
  _dinoml_rocm_run(_clang_result _llvm_bin "${HIPCONFIG_EXECUTABLE}" --hipclangpath --newline)
  _dinoml_rocm_to_cmake_path("${_root}" _root)
  _dinoml_rocm_to_cmake_path("${_llvm_bin}" _llvm_bin)
  set(${out_root} "${_root}" PARENT_SCOPE)
  set(${out_cmake} "${_root}/lib/cmake" PARENT_SCOPE)
  set(${out_bin} "${_root}/bin" PARENT_SCOPE)
  set(${out_llvm_bin} "${_llvm_bin}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_first_existing_dir out_path)
  foreach(_candidate IN LISTS ARGN)
    if(_candidate AND EXISTS "${_candidate}" AND IS_DIRECTORY "${_candidate}")
      set(${out_path} "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_path} "" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_find_llvm_bin out_path root hinted_llvm_bin)
  _dinoml_rocm_first_existing_dir(
    _found
    "${hinted_llvm_bin}"
    "${root}/lib/llvm/bin"
    "${root}/llvm/bin"
    "${root}/bin"
  )
  set(${out_path} "${_found}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_find_device_lib_dir out_path root)
  set(_candidates
    "${root}/lib/llvm/amdgcn/bitcode"
    "${root}/llvm/amdgcn/bitcode"
    "${root}/amdgcn/bitcode"
  )
  file(GLOB _clang_candidates
    "${root}/lib/llvm/lib/clang/*/amdgcn/bitcode"
    "${root}/llvm/lib/clang/*/amdgcn/bitcode"
    "${root}/lib/clang/*/amdgcn/bitcode"
  )
  list(APPEND _candidates ${_clang_candidates})
  _dinoml_rocm_first_existing_dir(_found ${_candidates})
  set(${out_path} "${_found}" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_find_clangxx out_path llvm_bin)
  if(WIN32)
    set(_suffix ".exe")
  else()
    set(_suffix "")
  endif()
  foreach(_candidate IN ITEMS "${llvm_bin}/clang++${_suffix}" "${llvm_bin}/amdclang++${_suffix}")
    if(EXISTS "${_candidate}")
      set(${out_path} "${_candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_path} "" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_find_windows_rc out_path)
  if(NOT WIN32)
    set(${out_path} "" PARENT_SCOPE)
    return()
  endif()

  find_program(_DINOML_ROCM_RC_COMPILER NAMES rc.exe)
  if(_DINOML_ROCM_RC_COMPILER)
    set(${out_path} "${_DINOML_ROCM_RC_COMPILER}" PARENT_SCOPE)
    return()
  endif()

  file(GLOB_RECURSE _rc_candidates
    "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/rc.exe"
  )
  if(_rc_candidates)
    list(SORT _rc_candidates COMPARE NATURAL ORDER DESCENDING)
    list(GET _rc_candidates 0 _rc)
    set(${out_path} "${_rc}" PARENT_SCOPE)
    return()
  endif()

  set(${out_path} "" PARENT_SCOPE)
endfunction()

function(_dinoml_rocm_default_arch out_arch)
  set(_arch "")
  if(WIN32)
    set(_exe_suffix ".exe")
  else()
    set(_exe_suffix "")
  endif()
  if(DINOML_ROCM_LLVM_BIN AND EXISTS "${DINOML_ROCM_LLVM_BIN}/amdgpu-arch${_exe_suffix}")
    execute_process(
      COMMAND "${DINOML_ROCM_LLVM_BIN}/amdgpu-arch${_exe_suffix}"
      RESULT_VARIABLE _arch_result
      OUTPUT_VARIABLE _arch_output
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_arch_result EQUAL 0 AND _arch_output)
      string(REPLACE "\n" ";" _arch_list "${_arch_output}")
      list(GET _arch_list 0 _arch)
    endif()
  endif()
  if(NOT _arch)
    set(_arch "gfx1201")
  endif()
  set(${out_arch} "${_arch}" PARENT_SCOPE)
endfunction()

function(dinoml_configure_rocm_sdk)
  set(_root "${DINOML_ROCM_ROOT}")
  set(_cmake "${DINOML_ROCM_CMAKE_PREFIX}")
  set(_bin "${DINOML_ROCM_BIN}")
  set(_llvm_bin "${DINOML_ROCM_LLVM_BIN}")
  if(_root)
    set(_provider "explicit")
  endif()

  if(NOT _root)
    _dinoml_rocm_try_rocm_sdk(_root _cmake _bin _llvm_bin)
    if(_root)
      set(_provider "python-rocm-sdk")
    endif()
  endif()

  if(NOT _root)
    _dinoml_rocm_try_hipconfig(_root _cmake _bin _llvm_bin)
    if(_root)
      set(_provider "hipconfig")
    endif()
  endif()

  if(NOT _root)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
      set(_root "$ENV{ROCM_PATH}")
    elseif(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
      set(_root "$ENV{HIP_PATH}")
    elseif(WIN32)
      _dinoml_rocm_windows_install_roots(_windows_roots)
      if(_windows_roots)
        list(GET _windows_roots 0 _root)
      else()
        set(_root "C:/Program Files/AMD/ROCm")
      endif()
    else()
      set(_root "/opt/rocm")
    endif()
    set(_provider "environment-or-default")
  endif()

  if(NOT _cmake)
    set(_cmake "${_root}/lib/cmake")
  endif()
  if(NOT _bin)
    set(_bin "${_root}/bin")
  endif()

  _dinoml_rocm_to_cmake_path("${_root}" _root)
  _dinoml_rocm_to_cmake_path("${_cmake}" _cmake)
  _dinoml_rocm_to_cmake_path("${_bin}" _bin)
  _dinoml_rocm_find_llvm_bin(_llvm_bin "${_root}" "${_llvm_bin}")
  _dinoml_rocm_find_device_lib_dir(_device_lib_dir "${_root}")
  _dinoml_rocm_find_clangxx(_clangxx "${_llvm_bin}")
  _dinoml_rocm_find_windows_rc(_rc)

  set(DINOML_ROCM_PROVIDER "${_provider}" CACHE STRING "ROCm SDK resolver provider" FORCE)
  set(DINOML_ROCM_ROOT "${_root}" CACHE PATH "ROCm SDK root" FORCE)
  set(DINOML_ROCM_CMAKE_PREFIX "${_cmake}" CACHE PATH "ROCm CMake package directory" FORCE)
  set(DINOML_ROCM_BIN "${_bin}" CACHE PATH "ROCm binary directory" FORCE)
  set(DINOML_ROCM_RUNTIME_PATH "${_bin}" CACHE PATH "ROCm runtime DLL/shared-library directory" FORCE)
  set(DINOML_ROCM_LLVM_BIN "${_llvm_bin}" CACHE PATH "ROCm LLVM binary directory" FORCE)
  set(DINOML_ROCM_DEVICE_LIB_DIR "${_device_lib_dir}" CACHE PATH "ROCm device bitcode directory" FORCE)
  set(DINOML_ROCM_RC_COMPILER "${_rc}" CACHE FILEPATH "Windows resource compiler for ROCm/Ninja builds" FORCE)

  if(NOT DINOML_ROCM_ARCH)
    _dinoml_rocm_default_arch(_arch)
    set(DINOML_ROCM_ARCH "${_arch}" CACHE STRING "AMD GPU target architecture" FORCE)
  endif()

  set(HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
  set(ENV{HIP_PLATFORM} "amd")
  set(ENV{HIP_PATH} "${DINOML_ROCM_ROOT}")
  set(ENV{ROCM_PATH} "${DINOML_ROCM_ROOT}")

  set(_prefix_path "${DINOML_ROCM_CMAKE_PREFIX};${DINOML_ROCM_ROOT};${CMAKE_PREFIX_PATH}")
  set(CMAKE_PREFIX_PATH "${_prefix_path}" CACHE STRING "CMake package search path" FORCE)
  set(CMAKE_PREFIX_PATH "${_prefix_path}" PARENT_SCOPE)

  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${DINOML_ROCM_ARCH}" CACHE STRING "HIP architectures" FORCE)
  endif()
  set(CMAKE_HIP_COMPILER_ROCM_ROOT "${DINOML_ROCM_ROOT}" CACHE PATH "ROCm root for HIP compiler detection" FORCE)
  if(_clangxx AND NOT DEFINED CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${_clangxx}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
  if(WIN32 AND _rc AND NOT DEFINED CMAKE_RC_COMPILER)
    set(CMAKE_RC_COMPILER "${_rc}" CACHE FILEPATH "Windows resource compiler" FORCE)
  endif()

  set(_hip_flags "--rocm-path=${DINOML_ROCM_ROOT}")
  if(DINOML_ROCM_DEVICE_LIB_DIR)
    string(APPEND _hip_flags " --rocm-device-lib-path=${DINOML_ROCM_DEVICE_LIB_DIR}")
  endif()
  set(CMAKE_HIP_FLAGS_INIT "${_hip_flags}" CACHE STRING "Initial HIP flags" FORCE)

  message(STATUS "DinoML ROCm provider: ${DINOML_ROCM_PROVIDER}")
  message(STATUS "DinoML ROCm root: ${DINOML_ROCM_ROOT}")
  message(STATUS "DinoML ROCm CMake prefix: ${DINOML_ROCM_CMAKE_PREFIX}")
  message(STATUS "DinoML ROCm LLVM bin: ${DINOML_ROCM_LLVM_BIN}")
  message(STATUS "DinoML ROCm device libs: ${DINOML_ROCM_DEVICE_LIB_DIR}")
  message(STATUS "DinoML ROCm RC compiler: ${DINOML_ROCM_RC_COMPILER}")
  message(STATUS "DinoML ROCm arch: ${CMAKE_HIP_ARCHITECTURES}")
endfunction()
