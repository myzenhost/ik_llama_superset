cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

### llama.cpp version
set(LLAMA_VERSION_MAJOR 0)
set(LLAMA_VERSION_MINOR 7)
set(LLAMA_VERSION_PATCH 0)
set(LLAMA_VERSION_BASE "${LLAMA_VERSION_MAJOR}.${LLAMA_VERSION_MINOR}.${LLAMA_VERSION_PATCH}")

# whether this is a development/nightly build
# set this to OFF when making a release from a release tag (vX.Y.Z)
# ref: https://github.com/ggml-org/ggml/discussions/1579
option(LLAMA_BUILD_IS_DEV "llama: dev build" ON)

if (LLAMA_BUILD_IS_DEV)
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}-dev")
else()
    # TODO: check that the current commit is tagged correctly according to the version specified above
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}")
endif()

message(STATUS "llama.cpp version: ${LLAMA_VERSION}")

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)
else()
    set(LLAMA_STANDALONE OFF)
endif()

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)

option(LLAMA_WASM_MEM64 "llama: use 64-bit memory in WASM builds" ON)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    # Use 64-bit memory to support backend_get_memory queries
    # TODO: analyze performance impact, see https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using
    if (LLAMA_WASM_MEM64)
      add_compile_options("-sMEMORY64=1")
      add_link_options("-sMEMORY64=1")
    endif()
    add_link_options("-sALLOW_MEMORY_GROWTH=1")

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" OFF)
    # WIRED 2026-09-08. This option was born unread: `git log -S LLAMA_WASM_SINGLE_FILE` finds one
    # commit in the entire history (eb34620ae, which declared it) and no tree -- this fork,
    # 279b4bfaf^1 or upstream 279b4bfaf^2 -- has ever contained a reader. So "embed WASM inside the
    # generated llama.js" was a promise nothing kept, and flipping it did nothing at all.
    # -sSINGLE_FILE=1 is the Emscripten setting that does exactly what the description says: it
    # base64-embeds the .wasm into the emitted .js instead of loading it as a side file. Written in
    # the same add_link_options form as -sALLOW_MEMORY_GROWTH=1 four lines above.
    # Reachable only under EMSCRIPTEN, and OFF by default, so no configuration built here changes.
    if (LLAMA_WASM_SINGLE_FILE)
        add_link_options("-sSINGLE_FILE=1")
    endif()
    option(LLAMA_BUILD_HTML "llama: build HTML file" ON)
    if (LLAMA_BUILD_HTML)
        set(CMAKE_EXECUTABLE_SUFFIX ".html")
    endif()
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# force MSVC compiler charset to utf-8
if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (LLAMA_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMA_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(LLAMA_TOOLS_INSTALL_DEFAULT ${LLAMA_STANDALONE})
endif()

# subprocess spawning isn't a supported/sandbox-friendly operation on mobile OSes or in WASM
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR ANDROID
        OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN)
    set(LLAMA_SUBPROCESS_DEFAULT OFF)
else()
    set(LLAMA_SUBPROCESS_DEFAULT ON)
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ${LLAMA_STANDALONE})

# extra artifacts
option(LLAMA_BUILD_TESTS     "llama: build tests"                                                                ${LLAMA_STANDALONE})
option(LLAMA_BUILD_TOOLS     "llama: build tools"                                                                ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES  "llama: build examples"                                                             ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER    "llama: build server example"                                                       ${LLAMA_STANDALONE})
option(LLAMA_BUILD_APP       "llama: build the unified binary"                                                   ${LLAMA_STANDALONE})
option(LLAMA_BUILD_BENCH_POWERINFER_SPARSE "llama: build benches/iq2bn-simd-sparse-downproj (x86-64 AVX2 micro-bench, opt-in)" OFF)
option(LLAMA_BUILD_UI        "llama: build the embedded Web UI for server"                                       OFF)
option(LLAMA_USE_PREBUILT_UI "llama: use prebuilt UI from HF Bucket when available (requires LLAMA_BUILD_UI=ON)" OFF)
option(LLAMA_CONNECT         "llama: include the llama-connect tool (downloads a prebuilt binary)"               OFF)

option(LLAMA_TOOLS_INSTALL "llama: install tools" ${LLAMA_TOOLS_INSTALL_DEFAULT})
option(LLAMA_TESTS_INSTALL "llama: install tests" ON)

# 3rd party libs
option(LLAMA_OPENSSL    "llama: use openssl to support HTTPS" ON)
option(LLAMA_SUBPROCESS "llama-common: use subprocess, required by server tools and server router mode" ${LLAMA_SUBPROCESS_DEFAULT})
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)
set(LLAMA_LLGUIDANCE_DIR "" CACHE PATH "llama-common: prebuilt llguidance checkout (target/release/llguidance.{lib,a} + llguidance.h); empty = fetch and build at configure time")
option(LLAMA_XGRAMMAR "llama-common: include XGrammar as a second structured-output backend (%xgrammar grammars)" OFF)
set(LLAMA_XGRAMMAR_DIR "" CACHE PATH "llama-common: xgrammar source checkout (include/ + 3rdparty/dlpack)")
set(LLAMA_XGRAMMAR_LIB "" CACHE FILEPATH "llama-common: the xgrammar static library built from LLAMA_XGRAMMAR_DIR")


# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/win-app.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

# --- carried over from this fork, which upstream does not have ---------------------------
# LLAMA_CURL: the fork exposes libcurl model download as an explicit option.
option(LLAMA_CURL "llama: use libcurl to download model from an URL" OFF)

# LLAMA_SERVER_SQLITE3: the pre-merge fork's server had a SQLite-backed session key/value store
# (group G3, row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01),
# stranded when merge 279b4bfaf replaced the fork's own server.cpp with upstream's split files and
# never carried a SQLite dependency along. Re-vendored at vendor/sqlite3 (public domain, SQLite
# 3.47.1 amalgamation) rather than requiring find_package(SQLite3) -- this host's own MinGW
# toolchain ships no linkable libsqlite3 dev package, and vendoring keeps a static, portable build
# working the same way LLAMA_CURL's absence does not block a build with it OFF. Default OFF: this
# opens/creates a file on disk and exposes 7 new HTTP routes the moment --sql-save-file is also
# given, which most llama-server builds and deployments have no use for.
option(LLAMA_SERVER_SQLITE3 "llama: build the SQLite-backed server session store (tools/server, opt-in via --sql-save-file)" OFF)
# SQLITE3_MODERN_CPP_SUPPORT is defined on just the server-context target, not globally --
# see tools/server/CMakeLists.txt.

# The fork routes its sanitizer switches down into ggml; upstream's root does not.
set(GGML_SANITIZE_THREAD    ${LLAMA_SANITIZE_THREAD})
set(GGML_SANITIZE_ADDRESS   ${LLAMA_SANITIZE_ADDRESS})
set(GGML_SANITIZE_UNDEFINED ${LLAMA_SANITIZE_UNDEFINED})

# CUDA graphs are enabled under BOTH option names on purpose, and llamafile under only one.
# Measured on the STAGED ggml/src/CMakeLists.txt (56,742 bytes), word-bounded:
#     GGML_CUDA_USE_GRAPHS      2   <- this fork's ggml still reads the OLD name
#     GGML_CUDA_GRAPHS(_DEFAULT) 0
#     GGML_LLAMAFILE(_DEFAULT)   0   <- llamafile has NO reader there at all
# while upstream's ggml/CMakeLists.txt reads GGML_CUDA_GRAPHS_DEFAULT (3) and
# GGML_LLAMAFILE_DEFAULT (3) and does not know GGML_CUDA_USE_GRAPHS (0). Setting a name
# nothing reads is a SILENT no-op, so llamafile is left entirely to upstream's guarded
# GGML_LLAMAFILE_DEFAULT above -- adding a GGML_LLAMAFILE set here would have been exactly
# that no-op. CUDA graphs genuinely needs both until ggml/CMakeLists.txt is resolved and the
# fork's ggml/src is retargeted to one name -- row
# S69-MERGE-GGML-CUDA-GRAPHS-OPTION-RENAMED-SILENT-NOOP-2026-09-07-01.
if (NOT DEFINED GGML_CUDA_USE_GRAPHS)
    set(GGML_CUDA_USE_GRAPHS ON)
endif()
# -----------------------------------------------------------------------------------------

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)
llama_option_depr(WARNING     LLAMA_CURL)

include("cmake/license.cmake")
license_add_file("llama.cpp" "LICENSE")

#
# compile options
#

# clang stores the modification time of the precompiled header sources inside the
# header and rejects it when they differ, so the timestamp is left out of it
add_compile_options(
    "$<$<COMPILE_LANG_AND_ID:C,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>")

#
# 3rd-party
#

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

add_subdirectory(vendor)

if (LLAMA_BUILD_COMMON)
    add_subdirectory(common)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if (LLAMA_BUILD_BENCH_POWERINFER_SPARSE)
    find_package(Threads REQUIRED)
    add_subdirectory(benches/iq2bn-simd-sparse-downproj)
endif()

if (LLAMA_BUILD_APP)
    add_subdirectory(app)
endif()

# Standalone libmtmd build without pulling in the rest of the tools/ tree.
# Useful when packaging just the mtmd library for language bindings (e.g. an
# Apple XCFramework, or a WASM build). When the full tools build is enabled,
# mtmd is already built by the tools/ subdirectory above; this hook only fires
# when LLAMA_BUILD_TOOLS is OFF to avoid double-adding the target.
option(LLAMA_BUILD_MTMD "llama: build tools/mtmd library standalone" OFF)
if (LLAMA_BUILD_MTMD AND NOT (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS))
    add_subdirectory(tools/mtmd)
endif()

llama_apply_app_manifests()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(TARGETS llama LIBRARY PUBLIC_HEADER)

if (LLAMA_BUILD_COMMON)
    install(TARGETS llama-common LIBRARY)
endif()

# This fork used to compute GGML_TRANSIENT_DEFINES and GGML_LINK_LIBRARIES here, because its old
# cmake/llama-config.cmake.in consumed them as @substitutions@. That template has since been
# resolved to upstream's, which gets the same information from `find_package(ggml)` -- the merged
# ggml/CMakeLists.txt installs a real package config (configure_package_config_file x2,
# write_basic_package_version_file x1, ggml-config.cmake x3, with ggml/cmake/ggml-config.cmake.in
# present on disk; HEAD's ggml installed none of that). A word-bounded sweep of the fork now finds
# ZERO readers of either name outside this file, so computing them would be dead code.
# CONSEQUENCE, and it is a real one: ggml/CMakeLists.txt is STILL UNMERGED, and its resolution
# MUST keep that package-config install. If it does not, `find_package(ggml REQUIRED)` in the
# installed llama-config.cmake fails at the CONSUMER's configure time -- never in a build we run
# here. Tracked as S69-MERGE-GGML-CONFIG-PACKAGE-REQUIRED-BY-LLAMA-CONFIG-2026-09-07-01.

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        VERSION ${LLAMA_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama)

# Carried over from this fork: the installed package ships the HF->GGUF converter.
install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
