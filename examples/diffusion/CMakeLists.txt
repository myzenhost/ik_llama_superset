set(TARGET llama-diffusion)
add_library(${TARGET} STATIC diffusion.cpp diffusion.h)
target_link_libraries(${TARGET} PUBLIC llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)

set(TARGET llama-diffusion-cli)
add_executable(${TARGET} diffusion-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-diffusion llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Self-test for the entropy-bound decoder + masked-path extensions. Needs a model argument, so it is a plain
# executable (not a ctest case); tests/diffusion-gemma-check/run_check.py builds the model and runs it.
set(TARGET llama-diffusion-eb-selftest)
add_executable(${TARGET} diffusion-eb-selftest.cpp)
target_link_libraries(${TARGET} PRIVATE llama-diffusion llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
