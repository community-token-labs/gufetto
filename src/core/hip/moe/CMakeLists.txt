# Shared MoE kernels: expert routing, the MoE epilogue and the vendored
# llama.cpp quantized-matmul tier with its routed-expert (mmid/mmvq) paths.
# Moved unchanged from Qwen3.8-Flash-Next; its compile options are kept.
set(GUFO_MOE_MMQ_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mmq")

# HIP quantized matrix kernels adapted from llama.cpp, isolated in the
# qfn_mmq namespace. The adapter retains its C++17 dialect.
set(GUFO_MOE_MMQ_SOURCES
  ${GUFO_MOE_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
  ${GUFO_MOE_MMQ_DIR}/qfn_mmq.hip.cpp
  ${GUFO_MOE_MMQ_DIR}/mmid.hip.cpp
  ${GUFO_MOE_MMQ_DIR}/mmvq.hip.cpp
  ${GUFO_MOE_MMQ_DIR}/moe_wave64.hip.cpp
  ${GUFO_MOE_MMQ_DIR}/quantize.hip.cpp)

add_library(gufo_moe_mmq STATIC ${GUFO_MOE_MMQ_SOURCES})
set_source_files_properties(${GUFO_MOE_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${GUFO_MOE_MMQ_DIR}/moe_wave64.hip.cpp
  PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
set_target_properties(gufo_moe_mmq PROPERTIES
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gufo_moe_mmq PRIVATE
  GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
target_compile_options(gufo_moe_mmq PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
target_include_directories(gufo_moe_mmq SYSTEM PUBLIC ${GUFO_MOE_MMQ_DIR})
target_link_libraries(gufo_moe_mmq PUBLIC hip::host roc::hipblas)

# Routing and epilogue launchers, compiled with the HIP options of the
# Flash-Next kernel TU they came from.
add_library(gufo_moe_hip STATIC routing.hip.cpp)
set_source_files_properties(routing.hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(gufo_moe_hip PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(gufo_moe_hip PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
target_link_libraries(gufo_moe_hip PUBLIC hip::host)
