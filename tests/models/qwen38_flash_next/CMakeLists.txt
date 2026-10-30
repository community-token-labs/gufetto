# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(cold_score gufo_qwen38_flash_next "cpu")
add_test(NAME qwen38_flash_next.cold_score_gpu_contract
  COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/cold_score_gpu_contract_test.py")
set_tests_properties(qwen38_flash_next.cold_score_gpu_contract PROPERTIES LABELS "qwen38_flash_next;cpu" TIMEOUT 30)
set(cold_lifetime_body "${CMAKE_CURRENT_BINARY_DIR}/cold_score_lifetime_body.inc")
add_custom_command(OUTPUT "${cold_lifetime_body}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cold_score_lifetime_test.py"
    --source-root "${PROJECT_SOURCE_DIR}" --output "${cold_lifetime_body}"
  DEPENDS cold_score_lifetime_test.py
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/engine.cpp"
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/kernels/rocm/executor.cpp"
  VERBATIM)
add_executable(qwen38_flash_next_cold_score_lifetime_test cold_score_lifetime_test.cpp "${cold_lifetime_body}")
target_include_directories(qwen38_flash_next_cold_score_lifetime_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(qwen38_flash_next_cold_score_lifetime_test PRIVATE gufo_logging gufo_platform)
add_test(NAME qwen38_flash_next.cold_score_lifetime
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cold_score_lifetime_test.py"
    --binary $<TARGET_FILE:qwen38_flash_next_cold_score_lifetime_test>)
set_tests_properties(qwen38_flash_next.cold_score_lifetime PROPERTIES LABELS "qwen38_flash_next;cpu" TIMEOUT 30)
add_test(NAME qwen38_flash_next.tokenizer_diff
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer_diff_test.py")
set_tests_properties(qwen38_flash_next.tokenizer_diff PROPERTIES
  LABELS "qwen38_flash_next;cpu;tools")
set(geometry_guards "${CMAKE_CURRENT_BINARY_DIR}/attention_geometry_guards.inc")
add_custom_command(OUTPUT "${geometry_guards}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_attention_geometry.py"
    --source-root "${PROJECT_SOURCE_DIR}" --output "${geometry_guards}"
  DEPENDS extract_attention_geometry.py
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/kernels/rocm/executor.cpp"
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp"
  VERBATIM)
add_qwen38_flash_next_test(attention_geometry gufo_core "cpu")
target_sources(qwen38_flash_next_attention_geometry_test PRIVATE "${geometry_guards}")
target_include_directories(qwen38_flash_next_attention_geometry_test PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")
add_qwen38_flash_next_test(npu_contracts gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_q4km gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(continuation_layout gufo_qwen38_flash_next "cpu")
# Host-only production serialization slices with explicit Model/executor proxies.
set(identity_body "${CMAKE_CURRENT_BINARY_DIR}/continuation_identity_body.inc")
add_custom_command(OUTPUT "${identity_body}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/extract_continuation_identity.py"
    --source-root "${PROJECT_SOURCE_DIR}" --output "${identity_body}"
  DEPENDS extract_continuation_identity.py
    "${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp"
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/continuation_adapter.cpp"
    "${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/engine.cpp"
  VERBATIM)
add_qwen38_flash_next_test(continuation_identity gufo_qwen38_flash_next "cpu")
target_sources(qwen38_flash_next_continuation_identity_test PRIVATE "${identity_body}")
target_include_directories(qwen38_flash_next_continuation_identity_test PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")
add_qwen38_flash_next_test(kv_cache gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(kv_q4 gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(kv_rotation gufo_qwen38_flash_next "cpu")
target_compile_options(qwen38_flash_next_kv_rotation_test PRIVATE
  -fno-fast-math -ffp-contract=off -frounding-math)
# Host compile/control probe of verbatim HIP helper bodies, not native decoding.
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT MSVC)
  add_test(NAME qwen38_flash_next.weight_dispatch_contract
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/weight_dispatch_contract_test.py"
      --source-root "${PROJECT_SOURCE_DIR}" --cxx "${CMAKE_CXX_COMPILER}"
      --rotation "ON"
      --work-dir "${CMAKE_CURRENT_BINARY_DIR}")
  set_tests_properties(qwen38_flash_next.weight_dispatch_contract PROPERTIES
    LABELS "qwen38_flash_next;cpu" TIMEOUT 45)
endif()
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(rope_scaling gufo_qwen38_flash_next "cpu")
# The HF-oracle/off-bit-identity checks link the scalar model oracle, so
# (like ngram) this stays out of cmake/Checks.cmake's hosted check-pr set.
add_qwen38_flash_next_test(rope_scaling_oracle
  gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(draft_vocab gufo_core "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")
add_qwen38_flash_next_test(target_probe gufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  # Includes the real private serving factory; no parallel provider/forward.
  add_executable(qwen38_flash_next_cold_score_gpu_test EXCLUDE_FROM_ALL
    cold_score_gpu_test.cpp device_headroom_fault.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/cold_score.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp)
  target_link_libraries(qwen38_flash_next_cold_score_gpu_test PRIVATE gufo_logging
    gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_cache_hip
    gufo_checkpoint_transfer hip::host ${CMAKE_DL_LIBS})
  foreach(mode IN ITEMS off capture refusal)
    add_test(NAME qwen38_flash_next.cold_score_gpu_${mode}
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cold_score_gpu_test.py"
        --binary $<TARGET_FILE:qwen38_flash_next_cold_score_gpu_test> --mode ${mode})
    set_tests_properties(qwen38_flash_next.cold_score_gpu_${mode} PROPERTIES
      LABELS "qwen38_flash_next;hip;gpu;gfx1151;full_logits" TIMEOUT 900 SKIP_RETURN_CODE 77
      RESOURCE_LOCK qwen38_flash_next_model)
  endforeach()
  add_qwen38_flash_next_test(kv_rotation_ops gufo_qwen38_flash_next "hip;gpu;gfx1151")
  target_sources(qwen38_flash_next_kv_rotation_ops_test PRIVATE
    kv_rotation_producer_native.cpp)
  target_compile_options(qwen38_flash_next_kv_rotation_ops_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops gdn_ops
      attention_ops kv_q8_ops kv_q4_ops moe_epilogue_ops moe_ids_ops projection_ops
      routed_wmma_ops select_ops snapshot_copy verification_sampling)
    add_qwen38_flash_next_test(${name} gufo_qwen38_flash_next "hip;gpu;gfx1151")
  endforeach()
  # Compile the actual private q8 load helper under production HIP flags;
  # exhaust its valid finite scale/code domain against the portable codec.
  add_executable(qwen38_flash_next_kv_q8_load_test kv_q8_load_test.hip)
  target_link_libraries(qwen38_flash_next_kv_q8_load_test PRIVATE
    gufo_qwen38_flash_next hip::host)
  target_compile_options(qwen38_flash_next_kv_q8_load_test PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only>)
  add_test(NAME qwen38_flash_next.kv_q8_load
    COMMAND qwen38_flash_next_kv_q8_load_test)
  set_tests_properties(qwen38_flash_next.kv_q8_load PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151" TIMEOUT 60)
  add_dependencies(qwen38_flash_next_tests qwen38_flash_next_kv_q8_load_test)

  # Packed MTP projections use the independent CPU GGUF decoders.
  target_link_libraries(qwen38_flash_next_projection_ops_test PRIVATE gufo_core)
  target_link_libraries(qwen38_flash_next_hc_mix_ops_test PRIVATE gufo_core)
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/component_text_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/cold_score.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_logging
    gufo_qwen38_flash_next gufo_deepseek_v4_flash gufo_cache_hip
    gufo_checkpoint_transfer hip::host)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp yarn_producer_native.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp reset_submission_fault.cpp)
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    gufo_qwen38_flash_next ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    gufo_qwen38_flash_next)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    gufo_qwen38_flash_next hip::host)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference)
endif()
