# Qwen3.8-Flash-Next (qwen4exp) model package. The scalar oracle is built
# only by explicit validation targets, not by production or routine checks.
add_library(gufo_qwen38_flash_next STATIC
  config.cpp
  continuation_layout.cpp
  kv_cache.cpp
  kv_cache_validation.cpp
  ngram.cpp
  rope_scaling.cpp
  weights.cpp)

if (GUFO_ENABLE_BF16_K_CONTROL)
  target_compile_definitions(gufo_qwen38_flash_next PUBLIC GUFO_ENABLE_BF16_K_CONTROL=1)
endif()

target_sources(gufo_qwen38_flash_next PRIVATE kv_rotation.cpp)
# The CPU oracle has explicit F32 operation boundaries and strict zero signs.
set_source_files_properties(kv_rotation.cpp PROPERTIES
  COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off;-frounding-math;-Werror=switch")

target_include_directories(gufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_core Threads::Threads)

add_library(gufo_qwen38_flash_next_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_qwen38_flash_next_reference PUBLIC
  gufo_qwen38_flash_next)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

option(GUFO_ENABLE_NPU "Build experimental XDNA2 projection offload through XRT" OFF)
if (GUFO_ENABLE_NPU AND NOT ENGINE_ENABLE_HIP)
  message(FATAL_ERROR "GUFO_ENABLE_NPU requires ENGINE_ENABLE_HIP")
endif()

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # HIP quantized matrix kernels adapted from llama.cpp, isolated in the
  # qfn_mmq namespace. The adapter retains its C++17 dialect.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen38_flash_next_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    continuation_adapter.cpp
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/kv_q8_dequant.hip
    ${QFN_ROCM_DIR}/w8a8_wave64.hip
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp
    ${QFN_ROCM_DIR}/batch.cpp
    engine.cpp)
  target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})

  # No XRT lookup, includes, link libraries or runtime source in the OFF build.
  if (GUFO_ENABLE_NPU)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
      message(FATAL_ERROR "GUFO_ENABLE_NPU currently supports Linux XDNA2 hosts only")
    endif()
    set(GUFO_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT installation prefix")
    set(GUFO_XRT_LIB_DIR "" CACHE PATH "Optional XRT library directory")
    find_path(GUFO_XRT_INCLUDE_DIR NAMES xrt/xrt_bo.h
      HINTS ${GUFO_XRT_ROOT} ENV XILINX_XRT PATH_SUFFIXES include)
    find_library(GUFO_XRT_COREUTIL_LIBRARY NAMES xrt_coreutil libxrt_coreutil.so.2
      HINTS ${GUFO_XRT_LIB_DIR} ${GUFO_XRT_ROOT} ENV XILINX_XRT
      PATH_SUFFIXES lib lib64)
    if (NOT GUFO_XRT_INCLUDE_DIR OR NOT GUFO_XRT_COREUTIL_LIBRARY)
      message(FATAL_ERROR
        "GUFO_ENABLE_NPU needs XRT headers and libxrt_coreutil; set GUFO_XRT_ROOT or GUFO_XRT_INCLUDE_DIR/GUFO_XRT_COREUTIL_LIBRARY")
    endif()
    target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_ROCM_DIR}/npu_gemm.cpp)
    target_include_directories(gufo_qwen38_flash_next SYSTEM PRIVATE ${GUFO_XRT_INCLUDE_DIR})
    target_compile_definitions(gufo_qwen38_flash_next PUBLIC GUFO_ENABLE_NPU=1 GUFO_NPU=1)
    target_link_libraries(gufo_qwen38_flash_next PRIVATE ${GUFO_XRT_COREUTIL_LIBRARY})
  endif()
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${QFN_ROCM_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_cache_hip
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)

  if (GUFO_BUILD_TOOLS AND NOT BUILD_TESTING)
    add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/tests/models/qwen38_flash_next/continuation_probe.cpp)
    target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
      gufo_qwen38_flash_next)
  endif()
endif()
