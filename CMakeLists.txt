cmake_minimum_required(VERSION 3.24)
project(presto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp sources contain raw UTF-8 byte literals that MSVC misreads
# under the default codepage; /utf-8 is required project-wide.
# /Zc:__cplusplus makes MSVC report the real C++ standard level, which
# llama.cpp's LU8() macro branches on.
if(MSVC)
  add_compile_options(/utf-8 /Zc:__cplusplus)
endif()

option(PRESTO_WITH_LLAMACPP "Enable llama.cpp GGUF backend" ON)
set(PRESTO_WITH_MLX "AUTO" CACHE STRING "Enable MLX backend: AUTO | ON | OFF (Apple Silicon only)")
option(PRESTO_BUILD_TESTS "Build zero-dependency unit tests" ON)

# Dependency refs. llama.cpp is pinned to a commit verified by the local
# build; bump deliberately when updating.
set(PRESTO_LLAMA_TAG "b21e4de74567f5eef213765c9476a843c2e43f0d" CACHE STRING "llama.cpp git ref")
set(PRESTO_MLX_TAG "v0.32.1" CACHE STRING "mlx git ref")
set(PRESTO_HTTPLIB_TAG "v0.15.3" CACHE STRING "cpp-httplib git ref")

include(FetchContent)

# ---------------------------------------------------------------------------
# llama.cpp backend (GGUF execution)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Hardware acceleration backends (llama.cpp/ggml based)
#   CUDA   : NVIDIA discrete GPUs
#   VULKAN : NVIDIA / AMD Radeon / Intel iGPU+Arc (cross-vendor)
#   HIP    : AMD Radeon via ROCm (Linux preferred)
#   METAL  : Apple Silicon (auto ON with llama.cpp on macOS)
# ---------------------------------------------------------------------------
option(PRESTO_WITH_CUDA "Build NVIDIA CUDA GPU backend" OFF)
option(PRESTO_WITH_VULKAN "Build Vulkan GPU backend (NVIDIA/AMD/Intel)" OFF)
option(PRESTO_WITH_HIP "Build AMD HIP/ROCm GPU backend" OFF)

if(PRESTO_WITH_LLAMACPP)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
  # Static ggml/llama keeps a single self-contained executable: shared builds
  # scatter ggml-cpu.dll that ggml_backend_load_all() cannot locate reliably.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  if(PRESTO_WITH_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
  endif()
  if(PRESTO_WITH_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
  endif()
  if(PRESTO_WITH_HIP)
    set(GGML_HIP ON CACHE BOOL "" FORCE)
  endif()
  # llama.cpp disables tools in FetchContent mode (LLAMA_STANDALONE=OFF);
  # force them on so llama-bench stays available for comparison benchmarks
  set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  FetchContent_Declare(llama_cpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
      GIT_TAG ${PRESTO_LLAMA_TAG}
      GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(llama_cpp)
endif()

if(APPLE AND PRESTO_WITH_LLAMACPP)
  set(PRESTO_HW_METAL_DEF ON)
endif()
if(PRESTO_WITH_CUDA)
  set(PRESTO_HW_CUDA_DEF ON)
endif()
if(PRESTO_WITH_VULKAN)
  set(PRESTO_HW_VULKAN_DEF ON)
endif()
if(PRESTO_WITH_HIP)
  set(PRESTO_HW_HIP_DEF ON)
endif()

# ---------------------------------------------------------------------------
# cpp-httplib (OpenAI-compatible server; header-only, tiny)
# ---------------------------------------------------------------------------
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
    GIT_TAG ${PRESTO_HTTPLIB_TAG}
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(httplib)

# ---------------------------------------------------------------------------
# presto executable
# ---------------------------------------------------------------------------
add_executable(presto
    src/cli.cpp
    src/detector.cpp
    src/engine.cpp
    src/gguf_meta.cpp
    src/server_http.cpp
    src/st_meta.cpp)

target_include_directories(presto PRIVATE include src)
target_link_libraries(presto PRIVATE httplib::httplib)

if(MSVC)
  target_compile_options(presto PRIVATE /W4 /permissive- /EHsc)
  target_compile_definitions(presto PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(presto PRIVATE -Wall -Wextra)
endif()

if(PRESTO_WITH_LLAMACPP)
  target_compile_definitions(presto PRIVATE PRESTO_WITH_LLAMACPP=1)
  target_sources(presto PRIVATE src/backends/llamacpp_backend.cpp)
  target_link_libraries(presto PRIVATE llama)
endif()

if(PRESTO_HW_CUDA_DEF)
  target_compile_definitions(presto PRIVATE PRESTO_HW_CUDA=1)
endif()
if(PRESTO_HW_VULKAN_DEF)
  target_compile_definitions(presto PRIVATE PRESTO_HW_VULKAN=1)
endif()
if(PRESTO_HW_HIP_DEF)
  target_compile_definitions(presto PRIVATE PRESTO_HW_HIP=1)
endif()
if(PRESTO_HW_METAL_DEF)
  target_compile_definitions(presto PRIVATE PRESTO_HW_METAL=1)
endif()

# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon only)
# ---------------------------------------------------------------------------
set(_presto_mlx_enabled OFF)
if(PRESTO_WITH_MLX STREQUAL "ON")
  set(_presto_mlx_enabled ON)
elseif(PRESTO_WITH_MLX STREQUAL "AUTO")
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "(arm64|aarch64)")
    set(_presto_mlx_enabled ON)
  endif()
endif()

if(_presto_mlx_enabled)
  FetchContent_Declare(mlx
      GIT_REPOSITORY https://github.com/ml-explore/mlx
      GIT_TAG ${PRESTO_MLX_TAG}
      GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(mlx)
  target_compile_definitions(presto PRIVATE PRESTO_WITH_MLX=1)
  target_sources(presto PRIVATE src/backends/mlx_backend.cpp)
  target_link_libraries(presto PRIVATE mlx)
endif()

# ---------------------------------------------------------------------------
# Unit tests (zero external dependencies; always runnable)
# ---------------------------------------------------------------------------
if(PRESTO_BUILD_TESTS)
  enable_testing()
  add_executable(presto_tests
      tests/test_main.cpp
      tests/test_detector.cpp
      tests/test_engine_select.cpp
      tests/test_gguf_meta.cpp
      tests/test_json_mini.cpp
      tests/test_st_meta.cpp
      # library sources under test (no CLI/server/backends)
      src/detector.cpp
      src/engine.cpp
      src/gguf_meta.cpp
      src/st_meta.cpp)
  target_include_directories(presto_tests PRIVATE include src tests)
  if(MSVC)
    target_compile_options(presto_tests PRIVATE /W4 /permissive- /EHsc)
  else()
    target_compile_options(presto_tests PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME unit COMMAND presto_tests)
endif()

message(STATUS "presto: llamacpp=${PRESTO_WITH_LLAMACPP} mlx=${_presto_mlx_enabled} tests=${PRESTO_BUILD_TESTS}")
