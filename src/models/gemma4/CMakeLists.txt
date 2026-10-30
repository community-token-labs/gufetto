# Gemma 4 text models (gemma4 targets and the gemma4-assistant MTP drafter).
# Host-only for now: the GGUF config, weight-binding, tokenizer, chat
# template and tool-call contracts that later topics consume.
add_library(gufo_gemma4 STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  tool_syntax.cpp
  weights.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core)

# Scalar CPU reference, the judge for the device runtime. It decodes every
# weight on the fly, so it keeps -O2 in every configuration; tests and
# validation tools link it explicitly.
add_library(gufo_gemma4_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_gemma4_reference PUBLIC gufo_gemma4)
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
