# Opt-in micro-bench (measured negative, kept for the record): -DLLAMA_BUILD_BENCH_POWERINFER_SPARSE=ON.
# Needs an x86-64 compiler with AVX2 + BMI (the bench uses _mm256_* and _mm_tzcnt_u32). Never part of the default build.
set(TARGET llama-bench-powerinfer-down-skip-simd)
add_executable(${TARGET} powerinfer_down_skip_simd_bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PRIVATE -mavx2 -mbmi -mfma)
# no install(): a bench is not a shipped binary
