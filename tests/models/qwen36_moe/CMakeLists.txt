add_executable(qwen36_moe_loader_test loader_test.cpp)
target_link_libraries(qwen36_moe_loader_test PRIVATE gufo_qwen36_moe)
target_compile_definitions(qwen36_moe_loader_test PRIVATE
  QWEN36_MOE_TABLE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/qwen36-35b-a3b-ud-q4kxl.table.tsv")
add_test(NAME qwen36_moe.loader COMMAND qwen36_moe_loader_test)
set_tests_properties(qwen36_moe.loader PROPERTIES LABELS "qwen36_moe;cpu")
