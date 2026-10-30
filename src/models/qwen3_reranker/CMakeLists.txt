# Qwen3-Reranker: config and prompt framing for the dense Qwen3 yes/no scorer.

add_library(gufo_qwen3_reranker STATIC
  config.cpp
  prompt.cpp
)
target_include_directories(gufo_qwen3_reranker PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_qwen3_reranker PUBLIC cxx_std_20)

# Scalar float32 CPU oracle for small inputs; tests link it explicitly.
add_library(gufo_qwen3_reranker_reference STATIC EXCLUDE_FROM_ALL
  reference/cpu_scorer.cpp
)
target_link_libraries(gufo_qwen3_reranker_reference PUBLIC gufo_qwen3_reranker)
