# mtmd

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            mtmd-helper.cpp
            mtmd-helper.h
            mtmd-helper-gen.cpp
            mtmd-image.cpp
            mtmd-image.h
            clip-model.h
            clip-graph.h
            mtmd-helper-common.h
            mtmd-internal.h
            mtmd-audio.h
            ../../vendor/hash/hash.cpp
            ../../vendor/hash/sha256/sha256.c
            models/cogvlm.cpp
            models/conformer.cpp
            models/deepseekocr.cpp
            models/deepseekocr2.cpp
            models/dots3note.cpp
            models/dotsocr.cpp
            models/exaone4_5.cpp
            models/glm5next-vision.cpp
            models/gemma4a.cpp
            models/gemma4ua.cpp
            models/gemma4uv.cpp
            models/gemma4v.cpp
            models/glm4v.cpp
            models/granite-speech.cpp
            models/granite4-vision.cpp
            models/hunyuanvl.cpp
            models/internvl.cpp
            models/kimik25.cpp
            models/kimik3.cpp
            models/kimivl.cpp
            models/llama4.cpp
            models/llava.cpp
            models/mimo-audio.cpp
            models/mimovl.cpp
            models/minicpmv.cpp
            models/minimax-m3.cpp
            models/mobilenetv5.cpp
            models/muse-glimmer.cpp
            models/nemotron-v2-vl.cpp
            models/paddleocr.cpp
            models/parakeet.cpp
            models/pixtral.cpp
            models/pockettts-gen.cpp
            models/pockettts-seanet.cpp
            models/pockettts-spkenc.cpp
            models/qwen2vl.cpp
            models/qwen3a.cpp
            models/qwen3tts-gen.cpp
            models/qwen3tts-spkenc.cpp
            models/qwen3vl.cpp
            models/siglip.cpp
            models/step3vl.cpp
            models/whisper-enc.cpp
            models/yasa2.cpp
            models/youtuvl.cpp
            # debug/mtmd-debug.cpp is NOT part of this library. It defines int main() (line 52),
            # i.e. it is a standalone debug driver, and it includes "arg.h" / "debug.h" from the
            # common target, which `mtmd` does not link. Nothing is lost by leaving it out: the
            # mtmd_debug_* API that debug/mtmd-debug.h declares is implemented in mtmd.cpp
            # (lines 2531-2620), which IS in this library. Build it as its own executable
            # against `common` if the debug driver is ever wanted.
            )

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../vendor)
# vendor/hash/sha256/sha256.c includes "rotate-bits/rotate-bits.h", which lives at
# vendor/hash/rotate-bits/ -- i.e. it is written relative to vendor/hash, not vendor.
target_include_directories(mtmd PRIVATE ../../vendor/hash)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

# qwen3a (Qwen3-ASR) projector runtime probe -- see models/test-qwen3a.cpp
set(TARGET llama-test-qwen3a)
# clip.cpp is an INTERNAL translation unit -- its clip_* symbols are not exported from the
# mtmd library, so the probe compiles them in directly rather than linking against mtmd.
add_executable         (${TARGET} models/test-qwen3a.cpp models/stb-impl-qwen3a.cpp clip.cpp mtmd-audio.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-qwen3a)
target_link_libraries  (${TARGET} PRIVATE ggml llama Threads::Threads)
target_include_directories(${TARGET} PRIVATE . ../.. ../../vendor)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# llama-mtmd-cli is upstream's demo CLI and it needs TWO headers this fork does not have.
# MEASURED 2026-09-01 against mtmd-cli.cpp's include list: of its seven common/ includes,
#   PRESENT  log.h  common.h  sampling.h  console.h  chat.h
#   ABSENT   arg.h  debug.h
# arg.h is upstream's CLI argument parser; this fork parses arguments its own way. Porting that
# parser to build a demo CLI is disproportionate when the fork already ships llama-audio for the
# audio path and llama-server for multimodal serving -- and it would drag upstream's arg handling
# in beside the fork's, which is how two competing parsers end up in one tree.
#
# GATED, NOT DELETED, and gated on the actual cause rather than a flag: if common/arg.h appears,
# this builds again with no further change. Same shape as examples/audio/CMakeLists.txt's
# `if (TARGET mtmd)` guard. Tracked: S69-PORT-MTMD-CLI-NEEDS-UPSTREAM-ARG-H-01.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../common/arg.h")
    set(TARGET llama-mtmd-cli)
    add_executable         (${TARGET} mtmd-cli.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
else()
    message(STATUS "llama-mtmd-cli: skipped (common/arg.h not present in this fork)")
endif()

target_include_directories(mtmd PRIVATE models)
target_include_directories(mtmd PRIVATE debug)
