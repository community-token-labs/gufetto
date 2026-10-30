# Hosted CPU regression suite. Other tests remain available through CTest and
# explicit targets; adding a model oracle must not make every PR expensive.
set(gufo_pr_targets
  cache_adapter_test
  cache_adapter_lifecycle_test
  cache_ledger_test
  cache_checkpoint_test
  cache_prefix_index_test
  cache_slot_test
  cache_idle_spill_test
  cache_disk_test
  cache_disk_publication_test
  cache_streaming_test
  cache_tiered_test
  cache_retention_test
  cache_model_capabilities_test
  arg_parser_test json_test json_constraint_test gguf_reader_test gguf_identity_test
  expert_stream_test
  heap_placement_test startup_admission_test diagnostic_memory_test model_plan_test
  platform_publish_test platform_prefault_test platform_mapped_file_test
  platform_fs_test
  logit_sampler_test ggml_dequant_test
  quote_tracker_test
  openai_chat_test http_server_test http_server_signals_test trace_test
  rerank_api_test
  audio_websocket_test
  text_generation_scheduler_test
  perf_trace_test checkpoint_transfer_test
  component_text_cache_test text_model_runner_test continuation_disk_store_test
  bench_cli_test prompt_cli_test eval_http_client_test
  qwen_tokenizer_test qwen_chat_template_test qwen_prefill_scratch_test
  qwen_vision_policy_test
  qwen_vision_model_path_test
  qwen_vision_cache_test
  qwen_vision_memory_budget_test
  qwen_vision_output_pool_test
  qwen27b_continuation_layout_test
  qwen38_flash_next_attention_geometry_test
  qwen38_flash_next_npu_contracts_test
  qwen38_flash_next_kv_cache_test
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  qwen38_flash_next_rope_scaling_test
  qwen38_flash_next_continuation_layout_test
  qwen38_flash_next_continuation_identity_test
  qwen38_flash_next_kv_cache_test qwen38_flash_next_target_probe_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test
  qwen3_asr_config_test qwen3_asr_audio_api_test
  embedding_gemma2_config_test embedding_gemma2_embeddings_api_test
  embedding_gemma2_image_processor_test
  embedding_gemma2_batch_scheduler_test
  qwen3_tts_config_test qwen3_tts_audio_api_test
  qwen_image_21_test
  video_api_test minimax_h3_sampling_test minimax_h3_runtime_test
  vision_ref_math_test)
set(gufo_pr_tests ${gufo_pr_targets})
list(REMOVE_ITEM gufo_pr_tests
  qwen38_flash_next_attention_geometry_test
  qwen38_flash_next_npu_contracts_test
  component_text_cache_test
  qwen38_flash_next_kv_cache_test
  qwen38_flash_next_config_test qwen38_flash_next_mtp_sampling_test
  qwen38_flash_next_rope_scaling_test
  qwen38_flash_next_continuation_layout_test
  qwen38_flash_next_continuation_identity_test
  qwen38_flash_next_kv_cache_test qwen38_flash_next_target_probe_test
  ds4_sampling_test ds4_chat_template_test ds4_cli_test)
list(APPEND gufo_pr_tests
  http_server_test_diagnostic_memory text_model_runner_test_diagnostic_memory
  text_model_runner_test_diagnostic_refusal_auto
  text_model_runner_test_diagnostic_refusal_explicit
  "qwen38_flash_next\\.attention_geometry"
  "qwen38_flash_next\\.npu_contracts"
  ${gufo_component_text_cache_tests}
  "qwen38_flash_next\\.kv_cache"
  "qwen38_flash_next\\.config" "qwen38_flash_next\\.mtp_sampling"
  "qwen38_flash_next\\.tokenizer_diff"
  "qwen38_flash_next\\.rope_scaling" "qwen38_flash_next\\.continuation_layout"
  "qwen38_flash_next\\.kv_cache" "qwen38_flash_next\\.target_probe"
  "qwen38_flash_next\\.continuation_identity"
  pd_attribution_scheduler_test pd_attribution_analyzer_test
  perf_trace_disabled_test perf_trace_phase_limit_test
  pinned_prefix_harness_test cache_lookup_harness_test
  cache_lookup_classify_harness_test token_parity_harness_test
  "ds4\\.sampling" "ds4\\.template" "ds4\\.cli"
  gufo_version gufo_help model_plan_cli_test serve_cli_test eval_http_test functional_runner_test
  cache_workloads_test cache_disk_faults_test
  cache_boundary_test
  kernel_resources_parser_test check_docs_test check_pr_title_test benchmark_card_test
  scoring_metrics_test scoring_contract_test matched_kl_test
  vision_images_test vision_reference_test vision_compare_test vision_downstream_test
  vision_encoder_reference_test vision_encoder_capture_test)
if (WIN32)
  # Linux-only hosted suites: RLIMIT_FSIZE and fork()-based tests, the
  # dup2/mmap/socket fixture tests, the permission-fault disk-store cases,
  # the cache tests that fork or wrap libc symbols, and the 1,250-chunk
  # per-file fsync stress in component_text_cache_test.
  set(gufo_pr_linux_only
    gguf_reader_test gguf_identity_test platform_publish_test
    platform_prefault_test trace_test http_server_signals_test
    http_server_test http_server_test_diagnostic_memory
    audio_websocket_test continuation_disk_store_test
    component_text_cache_test ${gufo_component_text_cache_tests}
    eval_http_client_test cache_disk_faults_test
    cache_boundary_test cache_adapter_test cache_slot_test cache_disk_test
    cache_streaming_test cache_tiered_test cache_disk_publication_test
    vision_encoder_capture_test)
  list(REMOVE_ITEM gufo_pr_targets ${gufo_pr_linux_only})
  list(REMOVE_ITEM gufo_pr_tests ${gufo_pr_linux_only})
endif()

list(JOIN gufo_pr_tests "|" gufo_pr_pattern)
add_custom_target(check-pr
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
    --timeout 60 -R "^(${gufo_pr_pattern})$"
  DEPENDS gufo vision_encoder_reference ${gufo_pr_targets}
    $<$<NOT:$<BOOL:${WIN32}>>:cache_disk_faults>
  USES_TERMINAL VERBATIM)
