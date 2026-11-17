# ggml-cuda-dl: Ferrox/cudarc-style dynamic CUDA driver + NVRTC loading.
# GATE-CLAIM-OK: architectural note about how this target is built (no CUDA toolkit needed to
# COMPILE it, CUDA is resolved at run time via LoadLibraryExW/dlopen) -- not a claim about a live
# GPU/toolkit resource being available or unavailable on this host right now.
#
# Deliberately NOT gated by GGML_CUDA and requires no CUDA toolkit headers to compile -- everything
# CUDA is resolved at run time. See cuda_dl.h for the source citations.
#
# vendor/hash/sha256/sha256.c (public domain, Igor Pavlov 2010) is compiled directly into this
# target rather than linked via the vendor::hash alias: add_subdirectory(vendor) (top-level
# CMakeLists.txt) runs AFTER add_subdirectory(ggml), so vendor::hash does not exist yet when this
# directory is processed. Compiling the same .c file into two targets is standard CMake practice,
# not a source copy.

add_library(ggml-cuda-dl STATIC
    cuda_dl.h
    cuda_dl.cpp
    nvrtc_cache.h
    nvrtc_cache.cpp
    kernels_silu.h
    kernels_silu.cpp
    ${GGML_CUDA_DL_VENDOR_HASH}/sha256/sha256.c
)

# GGML_CUDA_DL_VENDOR_HASH is set by ggml/src/CMakeLists.txt relative to the ggml tree, not
# CMAKE_SOURCE_DIR, so an embedding project whose top-level dir is not the llama root still resolves it.
set_source_files_properties(${GGML_CUDA_DL_VENDOR_HASH}/sha256/sha256.c PROPERTIES LANGUAGE C)

# PUBLIC as the PARENT dir so consumers (tests, and eventually ggml-cuda) write
# #include "ggml-cuda-dl/cuda_dl.h" -- internal files here use bare quoted includes
# ("cuda_dl.h"), which resolve via the including file's own directory regardless of this path.
target_include_directories(ggml-cuda-dl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# resolves sha256.c's own #include "rotate-bits/rotate-bits.h" and our #include "sha256/sha256.h"
target_include_directories(ggml-cuda-dl PRIVATE ${GGML_CUDA_DL_VENDOR_HASH})

target_compile_features(ggml-cuda-dl PRIVATE cxx_std_17)
set_target_properties(ggml-cuda-dl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    target_link_libraries(ggml-cuda-dl PRIVATE kernel32)
else()
    target_link_libraries(ggml-cuda-dl PRIVATE ${CMAKE_DL_LIBS})
endif()

# libstdc++'s <filesystem> needs -lstdc++fs pre-GCC9; MinGW here is newer but this is cheap
# insurance and a no-op where the symbols are already in libstdc++.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(ggml-cuda-dl PRIVATE stdc++fs)
endif()
