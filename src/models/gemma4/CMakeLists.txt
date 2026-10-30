# Gemma 4 text models (gemma4 targets and the gemma4-assistant MTP drafter).
# Host-only for now: the GGUF config contract that later topics consume.
add_library(gufo_gemma4 STATIC
  config.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core)
