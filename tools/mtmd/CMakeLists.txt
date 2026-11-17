# mtmd

set(MTMD_VIDEO_HELP "enable video support in mtmd (requires ffmpeg binary in PATH)")

set(MTMD_VIDEO ON CACHE BOOL "${MTMD_VIDEO_HELP}")
# TODO: add MTMD_VIDEO_METHOD in the future to select between ffmpeg and other backends

if (MTMD_VIDEO AND NOT LLAMA_SUBPROCESS)
    message(STATUS "Disabling MTMD_VIDEO because LLAMA_SUBPROCESS is OFF")
    set(MTMD_VIDEO OFF CACHE BOOL "${MTMD_VIDEO_HELP}" FORCE)
endif()

find_package(Threads REQUIRED)

# The per-architecture clip_graph_<arch> TUs. Each one emits the VTABLE for its
# class via its key function, so any target that compiles clip.cpp -- which
# instantiates those classes -- must compile these too or it gets 46 undefined
# vtable references at link. Named once here and used by both mtmd and
# llama-test-qwen3a; linking mtmd into the test instead would duplicate
# mtmd-helper.cpp's STB_IMAGE_IMPLEMENTATION against the test's own stb TU.
set(MTMD_MODEL_SOURCES
    models/cogvlm.cpp
    models/conformer.cpp
    models/dots3note.cpp
    models/dotsocr.cpp
    models/exaone4_5.cpp
    models/gemma4a.cpp
    models/gemma4v.cpp
    models/gemma4ua.cpp
    models/gemma4uv.cpp
    models/glm4v.cpp
    models/granite-speech.cpp
    models/granite4-vision.cpp
    models/hunyuanvl.cpp
    models/internvl.cpp
    models/kimivl.cpp
    models/kimik25.cpp
    models/kimik3.cpp
    models/nemotron-v2-vl.cpp
    models/muse-glimmer.cpp
    models/llama4.cpp
    models/llava.cpp
    models/minicpmv.cpp
    models/paddleocr.cpp
    models/pixtral.cpp
    models/qwen2vl.cpp
    models/minimax-m3.cpp
    models/qwen3vl.cpp
    models/ling3vl.cpp
    models/mimovl.cpp
    models/qwen3a.cpp
    models/mimo-audio.cpp
    models/qwen3tts-spkenc.cpp
    models/qwen3tts-gen.cpp
    models/pockettts-seanet.cpp
    models/pockettts-spkenc.cpp
    models/pockettts-gen.cpp
    models/step3vl.cpp
    models/siglip.cpp
    models/whisper-enc.cpp
    models/deepseekocr.cpp
    models/deepseekocr2.cpp
    models/deepseek4v.cpp
    models/mobilenetv5.cpp
    models/youtuvl.cpp
    models/yasa2.cpp
    models/parakeet.cpp
    models/muse-spark-audio.cpp
    models/nemotron-omni.cpp
    models/sensenova-u1.cpp
    models/soprano.cpp
    models/voxtral-rt.cpp
)

add_library(mtmd
            mtmd.cpp
            mtmd-audio.cpp
            mtmd-audio-chunks.cpp
            mtmd-audio-chunks.h
            mtmd-audio-encoder.cpp
            mtmd-audio-encoder.h
            mtmd-image.cpp
            mtmd-encoder-provider.cpp
            mtmd-encoder-provider.h
            mtmd.h
            mtmd-internal.h
            mtmd-helper.cpp
            mtmd-helper-gen.cpp
            mtmd-helper-common.h
            mtmd-helper.h
            clip.cpp
            clip.h
            clip-impl.h
            clip-model.h
            clip-graph.h
            models/models.h
            ${MTMD_MODEL_SOURCES}
            )

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads ${CMAKE_DL_LIBS} vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
target_include_directories(mtmd PUBLIC  .)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (MTMD_VIDEO)
    target_compile_definitions(mtmd PRIVATE MTMD_VIDEO)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)

    # export all symbols so that internal components can be tested by test-mtmd-impl
    set_target_properties     (mtmd PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd-helper.h
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

set_target_properties(mtmd
    PROPERTIES
    PRIVATE_HEADER debug/mtmd-debug.h)

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
    # miniaudio.h defines ma_android_sdk_version() without a prior prototype
    target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()

# MSVC defines _M_X64 for ARM64EC, so miniaudio.h takes its x64 path.
# ARM64EC does not support AVX types.
string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" MTMD_GENERATOR_PLATFORM_LWR)
if (MSVC AND MTMD_GENERATOR_PLATFORM_LWR STREQUAL "arm64ec")
    target_compile_definitions(mtmd PRIVATE MA_NO_AVX2)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

# if mtmd is linked against llama-common, we throw an error
if (TARGET mtmd)
    get_target_property(libs mtmd LINK_LIBRARIES)
    if (libs AND "llama-common" IN_LIST libs)
        message(FATAL_ERROR "mtmd is designed to be a public library.\n"
                            "It must not link against llama-common")
    endif()
endif()

# Gate CLI binaries on LLAMA_BUILD_TOOLS so that standalone library-only
# builds (LLAMA_BUILD_MTMD=ON with LLAMA_BUILD_TOOLS=OFF — e.g. Apple
# XCFramework packaging) skip the executables entirely. LLAMA_BUILD_COMMON
# defaults to ON in standalone builds, so we cannot rely on it for gating.
if (LLAMA_BUILD_TOOLS)
    add_executable(llama-llava-cli    deprecation-warning.cpp)
    add_executable(llama-gemma3-cli   deprecation-warning.cpp)
    add_executable(llama-minicpmv-cli deprecation-warning.cpp)
    add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

    set(TARGET llama-mtmd-cli)
    add_executable         (${TARGET} mtmd-cli.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
    target_link_libraries  (${TARGET} PRIVATE llama-common mtmd Threads::Threads)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    # mtmd-debug tool
    add_executable(llama-mtmd-debug debug/mtmd-debug.cpp)
    set_target_properties(llama-mtmd-debug PROPERTIES OUTPUT_NAME llama-mtmd-debug)
    target_link_libraries(llama-mtmd-debug PRIVATE llama-common mtmd Threads::Threads)
    target_compile_features(llama-mtmd-debug PRIVATE cxx_std_17)
endif()

# qwen3a (Qwen3-ASR) projector runtime probe -- see models/test-qwen3a.cpp.
# Stranded by the upstream merge: it lived only in examples/mtmd, which nothing builds.
# clip.cpp is an INTERNAL translation unit -- its clip_* symbols are not exported from the
# mtmd library, so the probe compiles them in directly rather than linking against mtmd.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-qwen3a)
    add_executable         (${TARGET} models/test-qwen3a.cpp models/stb-impl-qwen3a.cpp clip.cpp mtmd-audio.cpp mtmd-audio-chunks.cpp mtmd-encoder-provider.cpp ${MTMD_MODEL_SOURCES})
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-qwen3a)
    target_link_libraries  (${TARGET} PRIVATE ggml llama Threads::Threads
                                             vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
    target_include_directories(${TARGET} PRIVATE . models ../..)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

# Ollama qwen3-vl combined blob read as its own mmproj -- see tests/test-ollama-qwen3vl-clip.cpp. Metadata only
# (clip no_alloc). Compiles clip.cpp in, like llama-test-qwen3a, because the clip_* internals it inspects are not
# exported from mtmd. Takes the blob path as an argument or from LLAMA_TEST_OLLAMA_QWEN3VL_BLOB; exit 77 without it.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-ollama-qwen3vl-clip)
    add_executable         (${TARGET} tests/test-ollama-qwen3vl-clip.cpp models/stb-impl-qwen3a.cpp clip.cpp mtmd-audio.cpp mtmd-audio-chunks.cpp mtmd-encoder-provider.cpp ${MTMD_MODEL_SOURCES})
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-ollama-qwen3vl-clip)
    target_link_libraries  (${TARGET} PRIVATE ggml llama Threads::Threads
                                             vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
    target_include_directories(${TARGET} PRIVATE . models ../..)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if (NOT MSVC)
        target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
    endif()
    if (LLAMA_BUILD_TESTS)
        add_test(NAME test-ollama-qwen3vl-clip COMMAND ${TARGET})
        set_tests_properties(test-ollama-qwen3vl-clip PROPERTIES SKIP_RETURN_CODE 77 LABELS model)
    endif()
endif()

# Ollama qwen2.5-vl combined blob read as its own mmproj -- see tests/test-ollama-qwen25vl-clip.cpp. Structural arms are
# metadata only; with a reference mmproj it also loads both and compares the embeddings of one synthetic image (CPU).
# Takes the blob (and the reference) as arguments or from LLAMA_TEST_OLLAMA_QWEN25VL_BLOB/_REF; exit 77 without a blob.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-ollama-qwen25vl-clip)
    add_executable         (${TARGET} tests/test-ollama-qwen25vl-clip.cpp models/stb-impl-qwen3a.cpp clip.cpp mtmd-audio.cpp mtmd-audio-chunks.cpp mtmd-encoder-provider.cpp ${MTMD_MODEL_SOURCES})
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-ollama-qwen25vl-clip)
    target_link_libraries  (${TARGET} PRIVATE ggml llama Threads::Threads
                                             vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
    target_include_directories(${TARGET} PRIVATE . models ../..)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if (NOT MSVC)
        target_compile_options(${TARGET} PRIVATE -Wno-cast-qual)
    endif()
    if (LLAMA_BUILD_TESTS)
        add_test(NAME test-ollama-qwen25vl-clip COMMAND ${TARGET})
        set_tests_properties(test-ollama-qwen25vl-clip PROPERTIES SKIP_RETURN_CODE 77 LABELS model)
    endif()
endif()

# DeepSeek-V4 / V4.1 vision (deepseek4v / deepseek41) random-weight end-to-end check -- see tests/test-dsv4-vision.cpp.
# Needs a tiny mmproj from tests/make_tiny_dsv4_mmproj.py; not a ctest case because it takes file arguments.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-dsv4-vision)
    add_executable         (${TARGET} tests/test-dsv4-vision.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-dsv4-vision)
    target_link_libraries  (${TARGET} PRIVATE mtmd llama ggml Threads::Threads)
    target_include_directories(${TARGET} PRIVATE . ${PROJECT_SOURCE_DIR}/tests)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

# Ling 3.0 VL (BAILINGMOE3 rope dispatch + ling3vl projector) random-weight check -- see tests/test-ling3vl.cpp.
# Takes file arguments (a bailingmoe3 GGUF with and without rope.dimension_sections, and a tiny mmproj from
# tests/make_tiny_ling3vl_mmproj.py), so it is not a ctest case. Calls clip_* directly: fine on the static build.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-ling3vl)
    add_executable         (${TARGET} tests/test-ling3vl.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-ling3vl)
    target_link_libraries  (${TARGET} PRIVATE mtmd llama ggml Threads::Threads)
    target_include_directories(${TARGET} PRIVATE .)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

# mtmd_tokenize with more media markers than bitmaps (ik_llama.cpp ed86eedaf port) -- see tests/test-mtmd-stray-marker.cpp.
# Takes a text model + mmproj as arguments, so it is not a ctest case.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-mtmd-stray-marker)
    add_executable         (${TARGET} tests/test-mtmd-stray-marker.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-mtmd-stray-marker)
    target_link_libraries  (${TARGET} PRIVATE mtmd llama ggml Threads::Threads)
    target_include_directories(${TARGET} PRIVATE .)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endif()

# Audio decode length cap (ollama audio.go:149-154 behaviour) -- see tests/test-mtmd-audio-cap.cpp. Synthesizes its own WAVs;
# no model needed, so it is a ctest case.
if (LLAMA_BUILD_TOOLS)
    set(TARGET llama-test-mtmd-audio-cap)
    add_executable         (${TARGET} tests/test-mtmd-audio-cap.cpp)
    set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-test-mtmd-audio-cap)
    target_link_libraries  (${TARGET} PRIVATE mtmd llama ggml Threads::Threads)
    target_include_directories(${TARGET} PRIVATE .)
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
    if (LLAMA_BUILD_TESTS)
        add_test(NAME test-mtmd-audio-cap COMMAND ${TARGET})
    endif()
endif()

# Optional OpenVINO audio-encoder provider module (see mtmd-encoder-provider.h). The OpenVINO C library is loaded at
# run time, so the build needs only the C headers: point MTMD_OPENVINO_INCLUDE at the directory that holds
# openvino/c/*.h (the `openvino` PyPI wheel carries them under <site-packages>/openvino/include).
option(MTMD_ENCODER_OPENVINO "mtmd: build the OpenVINO audio-encoder provider module" OFF)
set(MTMD_OPENVINO_INCLUDE "" CACHE PATH "directory holding openvino/c/*.h, for MTMD_ENCODER_OPENVINO")
if (MTMD_ENCODER_OPENVINO)
    if (NOT MTMD_OPENVINO_INCLUDE OR NOT EXISTS "${MTMD_OPENVINO_INCLUDE}/openvino/c/ov_core.h")
        message(FATAL_ERROR "MTMD_ENCODER_OPENVINO needs MTMD_OPENVINO_INCLUDE=<dir containing openvino/c/ov_core.h>")
    endif()
    add_library(mtmd-encoder-openvino MODULE providers/openvino/mtmd-encoder-openvino.cpp mtmd-encoder-provider.h)
    set_target_properties(mtmd-encoder-openvino PROPERTIES PREFIX "")
    target_include_directories(mtmd-encoder-openvino PRIVATE . ${MTMD_OPENVINO_INCLUDE})
    target_compile_features(mtmd-encoder-openvino PRIVATE cxx_std_17)
    target_link_libraries(mtmd-encoder-openvino PRIVATE ${CMAKE_DL_LIBS})
endif()
