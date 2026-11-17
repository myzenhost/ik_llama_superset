set(TARGET offload-attn-recall)
# The op sources are compiled in (as tests/test-offload-mask-gqa does) so the harness calls the exact op code without the ik_offload
# symbols having to be exported from a shared libllama.
add_executable(${TARGET} offload-attn-recall.cpp
    ${PROJECT_SOURCE_DIR}/src/offload_attention_mask_op.cpp
    ${PROJECT_SOURCE_DIR}/src/offload_attention_op.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
