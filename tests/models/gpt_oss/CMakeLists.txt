# gpt-oss CPU tests: no weights needed.

add_executable(gpt_oss_harmony_test harmony_test.cpp)
target_link_libraries(gpt_oss_harmony_test PRIVATE gufo_gpt_oss)
target_compile_definitions(gpt_oss_harmony_test PRIVATE
  GUFO_GPT_OSS_HARMONY_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/harmony_goldens.json")
add_test(NAME gpt_oss_harmony_test COMMAND gpt_oss_harmony_test)
set_tests_properties(gpt_oss_harmony_test PROPERTIES
  LABELS "cpu;models;gpt-oss;chat_template")
