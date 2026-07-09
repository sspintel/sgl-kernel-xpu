cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

set(ATen_XPU_SYCL_XE20)
set(ATen_XPU_SYCL_XE35)
set(ATen_XPU_SYCL_XE40)
set(ATen_XPU_SYCL_COMMON)

# Torch
find_package(Python3 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
  COMMAND ${PYTHON3_EXECUTABLE}
          -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PY_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)


set(SGL_OPS_XPU_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${SGL_OPS_XPU_ROOT}/cmake/Modules)

# DPCPP SYCL target device.
# Possible values: bmg (Xe2), cri (Xe3P), jgs (Xe4)
# To change during pip install:
#   pip install -v -e . --config-settings=cmake.define.DPCPP_SYCL_TARGET=cri
set(DPCPP_SYCL_TARGET "" CACHE STRING
    "dpcpp target device (bmg | cri | jgs). Leave empty to auto-detect from device.")

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

include(FetchContent)

# SYCL support in cutlass
add_compile_definitions(CUTLASS_ENABLE_SYCL)
set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL in the cutlass" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS  OFF CACHE BOOL "Remove benchmark to avoid cmake version issue in google benchmark" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable headers only mode in cutlass" FORCE)

# Set SYCL_INTEL_TARGET based on DPCPP_SYCL_TARGET:
#   Xe2  (BMG) = 20
#   Xe3P (CRI) = 35
#   Xe4  (JGS) = 40
if(DPCPP_SYCL_TARGET MATCHES "bmg")
  add_compile_definitions(SYCL_INTEL_TARGET=20)
elseif(DPCPP_SYCL_TARGET MATCHES "cri")
  add_compile_definitions(SYCL_INTEL_TARGET=35)
  add_compile_definitions(__SYCL_TARGET_INTEL_GPU_CRI__)
elseif(DPCPP_SYCL_TARGET MATCHES "jgs")
  add_compile_definitions(SYCL_INTEL_TARGET=40)
else()
  message(FATAL_ERROR
    "Unsupported DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}. "
    "Must be one of: bmg, cri, jgs.")
endif()

# cutlass — repo and commit depend on the target architecture
if(DPCPP_SYCL_TARGET MATCHES "bmg")
  set(CUTLASS_SYCL_REPO "https://github.com/intel/sycl-tla.git")
  set(CUTLASS_SYCL_TAG  "525faea3f0f43a8aec2d21a70d44111db639a3a9")
elseif(DPCPP_SYCL_TARGET MATCHES "cri")
  set(CUTLASS_SYCL_REPO "git@github.com:intel-innersource/libraries.ai.cutlass.internal.git")
  # 8de697f18 (Jul 2 2026, PR #755): moves the `Base` typedef in
  # xe_array_mma_blockscaled_{native,fallback}.hpp from private to public,
  # required for blockwise_moe_{mxfp4,mxfp8}.cpp to compile.
  set(CUTLASS_SYCL_TAG  "8de697f181b39ecec3878b3dcfd6f106dfcdbfac")
elseif(DPCPP_SYCL_TARGET MATCHES "jgs")
  set(CUTLASS_SYCL_REPO "git@github.com:intel-innersource/libraries.ai.cutlass.internal.git")
  set(CUTLASS_SYCL_TAG  "439fd778b16de695d388eb9efc67b88c94828a3f")
else()
  message(FATAL_ERROR
      "Unsupported DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}. "
      "Must be one of: bmg, cri, jgs.")
endif()

FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY ${CUTLASS_SYCL_REPO}
    GIT_TAG        ${CUTLASS_SYCL_TAG}
    GIT_SHALLOW    OFF
)
FetchContent_MakeAvailable(repo-cutlass-sycl)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${repo-cutlass-sycl_SOURCE_DIR}/include
    ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include
)

add_subdirectory(${SGL_OPS_XPU_ROOT}/src)
