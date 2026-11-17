# examples/tts -- text to speech
#
# wavtokenizer is a STANDALONE ggml module: it owns its gguf load, backend and
# graph and does NOT add an architecture to libllama. ik_llama.cpp is
# deliberately divergent from upstream llama.cpp, so audio capability is added
# ALONGSIDE the core rather than inside it -- src/llama.cpp, ggml/src/ggml.c and
# ggml/include/ggml.h are untouched by this example.

find_package(Threads REQUIRED)

# STATIC on purpose -- see the note in examples/audio/CMakeLists.txt: an
# unqualified add_library() under -DBUILD_SHARED_LIBS=ON yields an export-less
# DLL with no import library, and consumers fail with LNK1181.
add_library(wavtokenizer STATIC
            wavtokenizer.cpp
            wavtokenizer.h
            )
target_link_libraries     (wavtokenizer PUBLIC  ggml)
target_link_libraries     (wavtokenizer PRIVATE Threads::Threads)
target_include_directories(wavtokenizer PUBLIC  .)
target_compile_features   (wavtokenizer PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(wavtokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET llama-tts)
add_executable         (${TARGET} tts.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-tts)
target_link_libraries  (${TARGET} PRIVATE common wavtokenizer audio-common ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../audio)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TARGET} RUNTIME)
