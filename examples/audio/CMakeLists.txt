# examples/audio -- unified audio (listen / think / speak)
#
# audio-common is a small dependency-free DSP library shared by llama-audio and
# llama-tts, adopted from audio.cpp's "Audio Utilities" layer (Apache-2.0):
# one implementation of WAV I/O, resampling, VAD chunking and STFT/ISTFT that
# every audio task reuses.

find_package(Threads REQUIRED)

# STATIC on purpose. This is an internal helper library with no exported API:
# under -DBUILD_SHARED_LIBS=ON an unqualified add_library() builds a DLL, MSVC
# emits no import library for it because nothing is __declspec(dllexport)'d, and
# every consumer then dies on LNK1181 "cannot open input file audio-common.lib".
add_library(audio-common STATIC
            audio-common.cpp
            audio-common.h
            asr-compression.cpp
            asr-compression.h
            asr-fallback.cpp
            asr-fallback.h
            asr-halluc.cpp
            asr-halluc.h
            asr-dtw.cpp
            asr-dtw.h
            asr-align.cpp
            asr-align.h
            asr-align-capture.cpp
            asr-align-capture.h
            asr-tdt.cpp
            asr-tdt.h
            audio-stream.cpp
            audio-stream.h
            speak-stream.cpp
            speak-stream.h
            )
# Silero VAD + post-processing live in the asr-vad library shared with llama-server (asr-vad.cmake); PUBLIC so every
# audio-common consumer (llama-audio) keeps seeing asr-silero.h / asr-vad-post.h and their symbols.
include(${CMAKE_CURRENT_SOURCE_DIR}/asr-vad.cmake)
target_link_libraries(audio-common PUBLIC asr-vad)
# ASR decode-quality helpers (compression-ratio check): zlib when present, own estimator otherwise.
find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    target_compile_definitions(audio-common PRIVATE ASR_HAVE_ZLIB=1)
    target_link_libraries     (audio-common PRIVATE ZLIB::ZLIB)
endif()
target_include_directories(audio-common PUBLIC .)
# asr-tdt reads the transducer decoder of a GGUF with the gguf API (ggml-base)
target_link_libraries     (audio-common PUBLIC ggml)
target_link_libraries     (audio-common PRIVATE Threads::Threads)
target_compile_features   (audio-common PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(audio-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# llama-audio needs the mtmd audio projectors for the "listen" half. examples/ is configured
# BEFORE tools/ (top-level CMakeLists), so TARGET mtmd does not exist yet here and the target
# used to be silently skipped (llama-audio never built). Define it from a call deferred to the
# end of the top-level configure instead, when mtmd (if the tree has it) is known.
function(llama_audio_define_target src_dir)
    if (TARGET mtmd)
        set(TARGET llama-audio)
        # audio-capture.cpp compiles miniaudio WITH device I/O for --task stream (microphone); see its header comment
        add_executable         (${TARGET} ${src_dir}/audio.cpp ${src_dir}/audio-capture.cpp ${src_dir}/audio-capture.h)
        set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-audio)
        target_link_libraries  (${TARGET} PRIVATE common audio-common wavtokenizer mtmd vendor::miniaudio ${CMAKE_THREAD_LIBS_INIT})
        target_include_directories(${TARGET} PRIVATE ${src_dir}/../tts)
        target_compile_features(${TARGET} PRIVATE cxx_std_17)
        install(TARGETS ${TARGET} RUNTIME)
        # The CLI-path ASR tests need this binary, so they are registered here, where the target is known to
        # exist (tests/CMakeLists.txt explains why they moved). The tiny-asr.gguf fixture is set up in tests/.
        if (LLAMA_BUILD_TESTS)
            add_test(NAME test-asr-silero-cli COMMAND bash ${src_dir}/tests/test-asr-silero-cli.sh
                     $<TARGET_FILE:llama-audio> ${src_dir}/tests/fixtures/silero)
            add_test(NAME test-asr-fallback-cli COMMAND bash ${src_dir}/tests/test-asr-fallback-cli.sh
                     $<TARGET_FILE:llama-audio> ${CMAKE_BINARY_DIR}/tests/tiny-asr.gguf)
            # llama-audio --words-out on a random-weight audio LLM + qwen2a mmproj (generated at run time, no download):
            # the word-timestamp capture in audio.cpp had no gate step. Exit 77 (SKIPPED) without numpy / the clip.
            find_package(Python3 COMPONENTS Interpreter QUIET)
            if (Python3_Interpreter_FOUND)
                add_test(NAME test-asr-words-tiny-cli
                         COMMAND ${Python3_EXECUTABLE} ${src_dir}/tests/test-asr-words-tiny-cli.py $<TARGET_FILE:llama-audio>
                                 ${src_dir}/tests/fixtures/librispeech_test_clean_6930-75918-0000.wav ${CMAKE_SOURCE_DIR}
                                 ${CMAKE_BINARY_DIR}/tests/asr-words-tiny)
                set_tests_properties(test-asr-words-tiny-cli PROPERTIES SKIP_RETURN_CODE 77 LABELS main)
            endif()
            set_tests_properties(test-asr-silero-cli test-asr-fallback-cli PROPERTIES LABELS main)
            set_tests_properties(test-asr-fallback-cli PROPERTIES FIXTURES_REQUIRED tiny_asr_gguf)
        endif()
    else()
        message(STATUS "llama-audio: skipped (target 'mtmd' not available)")
    endif()
endfunction()
cmake_language(EVAL CODE "cmake_language(DEFER DIRECTORY [[${CMAKE_SOURCE_DIR}]] CALL llama_audio_define_target [[${CMAKE_CURRENT_SOURCE_DIR}]])")
