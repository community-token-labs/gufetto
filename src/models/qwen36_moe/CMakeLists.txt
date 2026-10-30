# Qwen3.6-35B-A3B (qwen35moe) model package: metadata, loader map and binder.
# The runtime reuses the dense Qwen trunk and Flash-Next expert kernels later.
add_library(gufo_qwen36_moe STATIC
  config.cpp
  weights.cpp)

target_include_directories(gufo_qwen36_moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen36_moe PUBLIC gufo_core)
