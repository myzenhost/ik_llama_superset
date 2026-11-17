set(TARGET llama-diffusion-gemma-server)
add_executable(${TARGET} diffusion-gemma-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Visual server: runs the entropy-bound decoder (diffusion_generate_entropy_bound, from the llama-diffusion
# library in ../diffusion) and streams per-step canvas frames. Entered through examples/CMakeLists.txt
# (add_subdirectory(diffusion-gemma-server)); target_link_libraries resolves the llama-diffusion target at
# generate time, so the order of the two add_subdirectory lines does not matter.
# Rows: S69-PORT-UNSLOTH-LLAMA-DIFFUSION-SUBSYSTEM-01, S69-PORT-UNSLOTH-DIFFUSION-GEMMA-01.
set(TARGET llama-diffusion-gemma-visual-server)
add_executable(${TARGET} diffusion-gemma-visual-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../diffusion)
target_link_libraries(${TARGET} PRIVATE llama-diffusion llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
