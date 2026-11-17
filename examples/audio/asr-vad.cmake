# asr-vad: the native Silero VAD (asr-silero), its segment post-processing (asr-vad-post) and the whole-clip gate
# (asr-vad-gate), as ONE static library that both llama-audio (via audio-common, examples/audio/CMakeLists.txt) and
# llama-server (server-context, tools/server/CMakeLists.txt --asr-vad) link. Included from both places because either
# may be configured without the other (LLAMA_BUILD_EXAMPLES / LLAMA_BUILD_SERVER); the first include defines it.
# Std-only: no ggml, no llama. STATIC for the reason audio-common gives (no exported API -> no import library).
if (NOT TARGET asr-vad)
    add_library(asr-vad STATIC
                ${CMAKE_CURRENT_LIST_DIR}/asr-silero.cpp
                ${CMAKE_CURRENT_LIST_DIR}/asr-silero.h
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-post.cpp
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-post.h
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-gate.cpp
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-gate.h
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-trim.cpp
                ${CMAKE_CURRENT_LIST_DIR}/asr-vad-trim.h
                )
    target_include_directories(asr-vad PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    target_compile_features   (asr-vad PRIVATE cxx_std_17)
    if (BUILD_SHARED_LIBS)
        set_target_properties(asr-vad PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endif()
