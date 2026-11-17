# wavtokenizer is a STANDALONE ggml module carried over from this fork's earlier
# examples/tts/wavtokenizer.{cpp,h} OuteTTS pipeline (fork commit e8f68267e: this
# fork's own 3-argument ggml_backend_cuda_init(device, params, model), NOT
# upstream's 1-argument form -- verified present post-port, unchanged). This
# tools/tts/tts.cpp is upstream's newer mtmd-based TTS path (mmproj +
# mtmd_helper::gen_audio) and does NOT call this vocoder -- entering it here only
# stops the capability being silently dropped by the build (examples/tts is not
# add_subdirectory()'d from examples/CMakeLists.txt), it is deliberately NOT
# linked into the llama-tts executable below (different architecture, would need
# its own CLI entry point -- out of this port's scope). See
# temp/research/findings/S69_STRANDED_PORTS_APPLIED_2026-09-08.md.
#
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
add_executable(${TARGET} tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# --- llama-tts-outetts ----------------------------------------------------
# ENGINE-MERGE PORT (row S69-FORK-TOOL-COPIES-IN-UNBUILT-EXAMPLES-DIVERGE-FROM-BUILT-TOOLS-2026-09-27-01):
# entry point for the fork's own OuteTTS + WavTokenizer pipeline (tts-outetts.cpp, a
# common_params-shaped port of examples/tts/tts.cpp), which previously had NO built binary
# reaching it -- see that file's header comment and
# temp/research/findings/S69_ENGINE_TTS-OUTETTS_2026-09-27.md. Deliberately a SEPARATE target
# from llama-tts above (upstream's mtmd-based path) rather than merging the two mains; that
# split decision predates this port (temp/research/findings/S69_STRANDED_PORTS_APPLIED_2026-09-08.md
# "Port 1"). audio-common.{cpp,h} is reused directly from examples/audio/ (self-contained
# WAV I/O + ISTFT DSP, no llama/ggml dependency) rather than duplicated into tools/.
set(TARGET_OUTETTS llama-tts-outetts)
add_executable(${TARGET_OUTETTS}
               tts-outetts.cpp
               ${CMAKE_SOURCE_DIR}/examples/audio/audio-common.cpp
               ${CMAKE_SOURCE_DIR}/examples/audio/audio-common.h
               )
target_include_directories(${TARGET_OUTETTS} PRIVATE ${CMAKE_SOURCE_DIR}/examples/audio)
target_link_libraries(${TARGET_OUTETTS} PRIVATE wavtokenizer llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_OUTETTS} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_OUTETTS} RUNTIME)
endif()
