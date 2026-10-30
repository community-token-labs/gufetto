# gpt-oss: Harmony chat format. MXFP4 weights use the shared GGML dequant.

add_library(gufo_gpt_oss STATIC
  harmony.cpp
)

target_include_directories(gufo_gpt_oss PUBLIC ${PROJECT_SOURCE_DIR})
