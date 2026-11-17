find_package(Threads REQUIRED)

llama_add_compile_flags()

#
# llama-common-base
#

# Build info header

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        # For build-info.cpp below
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET llama-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Upstream's older name for this target was build_info. With no target of that name,
# target_link_libraries(... build_info ...) became a bare `-lbuild_info` -- MEASURED 2026-09-26:
# llama-quantize-stats failed "ld: cannot find -lbuild_info". Same class as the `common` alias below.
add_library(build_info ALIAS ${TARGET})

#
# llama-common
#

set(TARGET llama-common)

include(parsers/sources.cmake)

add_library(${TARGET}
    ${LLAMA_CHAT_PARSERS_SOURCES}
    arg.cpp
    arg.h
    base64.hpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-auto-parser.h
    chat-diff-analyzer.cpp
    chat-peg-parser.cpp
    chat-peg-parser.h
    chat-reasoning-rescue.cpp
    chat-malformed-repair.cpp
    chat-tool-args.cpp
    chat-xml-format.cpp
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    debug.cpp
    debug.h
    disk-tier.cpp
    disk-tier.h
    download.cpp
    download.h
    fit.cpp
    fit.h
    gguf-header-probe.cpp
    gguf-header-probe.h
    file-lock.cpp
    file-lock.h
    hf-cache.cpp
    hf-cache.h
    http.h
    imatrix-loader.cpp
    imatrix-loader.h
    imatrix-hessian.h
    json-partial.cpp
    json-partial.h
    json-schema-to-grammar.cpp
    json-schema.cpp
    json-schema.h
    json.cpp
    json.h
    llguidance.cpp
    xgrammar.cpp
    log.cpp
    log.h
    lora-merge.cpp
    lora-merge.h
    ngram-cache.cpp
    ngram-cache.h
    ngram-map.cpp
    ngram-map.h
    ngram-mod.cpp
    ngram-mod.h
    peg-parser.cpp
    peg-parser.h
    preset.cpp
    preset.h
    reasoning-budget.cpp
    reasoning-budget.h
    regex-partial.cpp
    regex-partial.h
    sampling.cpp
    sampling.h
    watermark.cpp
    watermark.h
    spec-accept.cpp
    spec-accept.h
    eagle3-d2t.h
    spec-tuner.cpp
    spec-tuner.h
    speculative.cpp
    speculative.h
    subproc.cpp
    subproc.h
    suffix-tree.cpp
    suffix-tree.h
    suffix-automaton.cpp
    suffix-automaton.h
    draft-policy.cpp
    draft-policy.h
    topk-scan.h
    trie.cpp
    trie.h
    token-ban.cpp
    token-ban.h
    prompt-reuse.cpp
    prompt-reuse.h
    unicode.cpp
    unicode.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
    )

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC .)
target_link_libraries     (${TARGET} PUBLIC vendor::nlohmann vendor::sheredom)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

# `common` is this library's name in BOTH trees we merge from (upstream snapshot
# llama-cpp-turboquant/common/CMakeLists.txt:44 and ik legacy ik_llama_src:51); only newer upstream
# calls it llama-common. With no target of that name, target_link_libraries(... common ...) silently
# became a bare `-lcommon` carrying NO include directories -- MEASURED 2026-09-26: sweep-bench,
# spec-bench and offload-attn-test failed "common.h: No such file or directory", and 6 more
# CMakeLists (audio, mtmd, tts, server, parser, main-cmake-pkg) link the old name. One alias fixes
# them all and every future merge that brings the old name in.
add_library(common ALIAS ${TARGET})

if (LLAMA_SUBPROCESS)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)
# vendored streaming SHA-256, used by download.cpp to check a finished download against the digest the registry announced
target_link_libraries(${TARGET} PRIVATE vendor::hash)

if (LLAMA_LLGUIDANCE)
    set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")
    add_library(llguidance STATIC IMPORTED)

    if (LLAMA_LLGUIDANCE_DIR)
        # A llguidance checkout already built with `cargo build --release --locked --package llguidance`.
        # Nothing is cloned or fetched at configure/build time; the caller owns provenance (tag, sha,
        # licence) of that checkout. The Rust target must match this compiler's ABI: the default
        # x86_64-pc-windows-msvc staticlib links with MSVC, not with MinGW.
        set(LLGUIDANCE_PATH ${LLAMA_LLGUIDANCE_DIR}/target/release)
        if (NOT EXISTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} OR NOT EXISTS ${LLGUIDANCE_PATH}/llguidance.h)
            message(FATAL_ERROR "LLAMA_LLGUIDANCE_DIR=${LLAMA_LLGUIDANCE_DIR}: expected ${LLGUIDANCE_LIB_NAME} and "
                                "llguidance.h in ${LLGUIDANCE_PATH} -- run cargo build --release --package llguidance there first")
        endif()
        message(STATUS "llguidance: using prebuilt ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME}")
    else()
        include(ExternalProject)
        set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
        set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)

        ExternalProject_Add(llguidance_ext
            GIT_REPOSITORY https://github.com/guidance-ai/llguidance
            # v1.8.0 (2026-08-10; was v1.0.1 d795912f). The C API the fork calls (13 llg_* functions in
            # common/llguidance.cpp) is present unchanged in v1.8.0's parser/llguidance.h.
            GIT_TAG dbaf504d498b6aeede06ae57adc6f7c2c4848c59
            PREFIX ${CMAKE_BINARY_DIR}/llguidance
            SOURCE_DIR ${LLGUIDANCE_SRC}
            BUILD_IN_SOURCE TRUE
            CONFIGURE_COMMAND ""
            BUILD_COMMAND cargo build --release --locked --package llguidance
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h
            UPDATE_COMMAND ""
        )
        add_dependencies(llguidance llguidance_ext)
    endif()
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE)
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME})

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
    target_link_libraries(${TARGET} PRIVATE llguidance)
    if (WIN32)
        target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt)
    endif()
endif()

if (LLAMA_XGRAMMAR)
    # XGrammar (mlc-ai/xgrammar, Apache-2.0): a second sampler-shaped constraint backend (common/xgrammar.cpp).
    # Links a static library the caller built from LLAMA_XGRAMMAR_DIR (Python bindings OFF -- the repo's own
    # cmake/config.cmake turns them ON, so build with a build-dir config.cmake; see temp/_xgr_msvc_build.bat).
    # The library's C++ ABI must match this compiler.
    if (NOT LLAMA_XGRAMMAR_DIR OR NOT LLAMA_XGRAMMAR_LIB)
        message(FATAL_ERROR "LLAMA_XGRAMMAR=ON needs LLAMA_XGRAMMAR_DIR (xgrammar source checkout) and "
                            "LLAMA_XGRAMMAR_LIB (the built static library)")
    endif()
    if (NOT EXISTS ${LLAMA_XGRAMMAR_LIB} OR NOT EXISTS ${LLAMA_XGRAMMAR_DIR}/include/xgrammar/xgrammar.h)
        message(FATAL_ERROR "LLAMA_XGRAMMAR: ${LLAMA_XGRAMMAR_LIB} or ${LLAMA_XGRAMMAR_DIR}/include/xgrammar/xgrammar.h missing")
    endif()
    message(STATUS "xgrammar: using ${LLAMA_XGRAMMAR_LIB}")
    add_library(xgrammar_ext STATIC IMPORTED)
    set_target_properties(xgrammar_ext PROPERTIES IMPORTED_LOCATION ${LLAMA_XGRAMMAR_LIB})
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_XGRAMMAR)
    target_include_directories(${TARGET} PRIVATE ${LLAMA_XGRAMMAR_DIR}/include ${LLAMA_XGRAMMAR_DIR}/3rdparty/dlpack/include)
    target_link_libraries(${TARGET} PRIVATE xgrammar_ext)
endif()

target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)

if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # AIX's flock() function comes from libbsd.a
    target_link_libraries(${TARGET} PRIVATE -lbsd)
endif()
