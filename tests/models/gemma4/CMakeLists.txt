# Gemma 4 host contracts. The artifact checks read the GGUF headers named by
# GUFO_GEMMA4_MODEL (and GUFO_GEMMA4_MTP_MODEL) and skip (77) without them.

function(add_gemma4_test name labels)
  set(target gemma4_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE gufo_gemma4)
  add_test(NAME gemma4.${name} COMMAND ${target})
  set_tests_properties(gemma4.${name} PROPERTIES
    LABELS "gemma4;${labels}" SKIP_RETURN_CODE 77)
endfunction()

add_gemma4_test(config "cpu;models;config")
add_gemma4_test(artifact_config "cpu;models;config;external-model")
add_gemma4_test(tokenizer_contract "cpu;models;tokenization")
add_gemma4_test(tokenizer_goldens "cpu;models;tokenization;external-model")
add_gemma4_test(template "cpu;models;chat_template")
foreach(target gemma4_tokenizer_goldens_test gemma4_template_test)
  target_compile_definitions(${target} PRIVATE
    GUFO_GEMMA4_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
endforeach()
