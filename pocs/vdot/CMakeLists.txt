set(TARGET llama-vdot)
add_executable(${TARGET} vdot.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-q8dot)
add_executable(${TARGET} q8dot.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Linux shared build: libggml-base.so has one UPWARD undefined reference, ggml_backend_dev_by_type()
# (ggml-backend.cpp -> ggml-backend-reg.cpp, which lives in libggml.so; see ggml/src/CMakeLists.txt).
# These two tools call nothing in libggml.so themselves, so Ubuntu's default --as-needed drops it
# from the link and ld then reports `libggml-base.so: undefined reference to ggml_backend_dev_by_type`
# (clean GCC 13 build, 2026-10-07: exactly these two targets, nothing else). Keep libggml.so.
# The layering inversion itself is not fixed here.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach (vdot_target llama-vdot llama-q8dot)
        target_link_options(${vdot_target} PRIVATE "LINKER:--no-as-needed")
    endforeach()
endif()
