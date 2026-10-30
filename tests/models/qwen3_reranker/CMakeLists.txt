# Config, prompt framing and the CPU reference scorer behind POST /v1/rerank:
# hand-computed tiny models, no checkpoint, device or socket.
add_executable(qwen3_reranker_reference_test
  reference_test.cpp
  ${PROJECT_SOURCE_DIR}/src/cli/serve/rerank_api.cpp
  ${PROJECT_SOURCE_DIR}/src/cli/serve/rerank_reference_scorer.cpp)
target_link_libraries(qwen3_reranker_reference_test PRIVATE
  gufo_qwen3_reranker_reference gufo_core)
add_test(NAME qwen3_reranker_reference_test
  COMMAND qwen3_reranker_reference_test)
set_tests_properties(qwen3_reranker_reference_test PROPERTIES
  LABELS "cpu;models;qwen3-reranker;server;api")
