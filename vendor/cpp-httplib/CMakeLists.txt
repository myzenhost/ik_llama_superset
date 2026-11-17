set(TARGET cpp-httplib)
license_add_file("cpp-httplib" "LICENSE")

find_package(Threads REQUIRED)

llama_add_compile_flags()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${TARGET} STATIC httplib.cpp httplib.h)

add_library(vendor::cpp-httplib ALIAS ${TARGET})

# disable warnings in 3rd party code -- unless LLAMA_ALL_WARNINGS_3RD_PARTY asks to keep them.
#
# WIRED 2026-09-08. LLAMA_ALL_WARNINGS_3RD_PARTY has existed since the very first CMake support
# (ed6849cc0) and `git log -S` finds exactly ONE commit touching the string in the whole history:
# the one that declared it. It has never had a reader in any of the three trees -- this fork,
# 279b4bfaf^1, or upstream 279b4bfaf^2 -- so its documented ON state was unreachable.
# llama_add_compile_flags() a few lines above already applies the full warning set to this
# directory; this unconditional -w / /w is what takes it straight back. Gating the SUPPRESSION is
# therefore the wiring: it is the only thing standing between the option and its stated effect.
# OFF (the default) still suppresses -- byte-identical to the previous behaviour.
if (NOT LLAMA_ALL_WARNINGS_3RD_PARTY)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${TARGET} PRIVATE /w)
    else()
        target_compile_options(${TARGET} PRIVATE -w)
    endif()
endif()

target_link_libraries(${TARGET} PRIVATE Threads::Threads)

if (WIN32 AND NOT MSVC)
    target_link_libraries(${TARGET} PUBLIC ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_compile_definitions(${TARGET} PRIVATE
    # increase max payload length to allow use of larger context size
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
    # increase backlog size to avoid connection resets for >> 1 slots
    CPPHTTPLIB_LISTEN_BACKLOG=512
    # increase max URI length to handle longer prompts in query string
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    # disable Nagle's algorithm
    CPPHTTPLIB_TCP_NODELAY=1
)

# _WIN32_WINNT BELONGS TO THIS TARGET, NOT TO THE SERVER.
#
# httplib.h:16 is `#error "cpp-httplib doesn't support Windows 8 or lower..."` and httplib.cpp:1520
# calls ::CreateFile2, which the SDK only declares at 0x0A00. That is this library's own
# requirement. Until now the definition was applied in exactly ONE place --
# examples/server/CMakeLists.txt:10 (an unbuilt copy, removed 2026-10-18) -- as a directory-wide add_compile_definitions, so every OTHER
# consumer built the same headers without it and failed. MEASURED 2026-09-10 on a mingw build of
# the `test-opt` target, which reaches httplib through common:
#     vendor/cpp-httplib/httplib.h:16:2:   error: #error "cpp-httplib doesn't support Windows 8..."
#     vendor/cpp-httplib/httplib.cpp:1520: error: '::CreateFile2' has not been declared
# 3 errors, and ninja stopped -- so no test target that links common could be built at all on this
# toolchain, which is why the autodiff work had no runnable test binary.
#
# The comment at ggml/CMakeLists.txt:236 records the assumption that produced this: "the vendored
# cpp-httplib used by the server target". It is not only the server target. common/CMakeLists.txt:155,
# tools/server/CMakeLists.txt:51 and tests/CMakeLists.txt:310 all link cpp-httplib.
#
# PUBLIC, not PRIVATE: consumers include httplib.h themselves and hit the same #error, so the
# requirement has to propagate. Guarded so it never overrides a value the parent already chose
# (GGML_WIN_VER from ggml/CMakeLists.txt), and this adds no second value.
if (WIN32)
    if (NOT DEFINED GGML_WIN_VER OR GGML_WIN_VER STREQUAL "")
        # Fallback for a configuration that does not add_subdirectory(ggml) (top-level line 256 is
        # conditional). Same default and same reasoning as ggml/CMakeLists.txt:237.
        set(GGML_WIN_VER "0x0A00")
    endif()
    target_compile_definitions(${TARGET} PUBLIC _WIN32_WINNT=${GGML_WIN_VER})
endif()

set(OPENSSL_NO_ASM ON CACHE BOOL "Disable OpenSSL ASM code when building BoringSSL or LibreSSL")

if (LLAMA_BUILD_BORINGSSL)
    set(FIPS OFF CACHE BOOL "Enable FIPS (BoringSSL)")

    set(BORINGSSL_GIT "https://boringssl.googlesource.com/boringssl" CACHE STRING "BoringSSL git repository")
    set(BORINGSSL_VERSION "0.20261105.0" CACHE STRING "BoringSSL version")

    message(STATUS "Fetching BoringSSL version ${BORINGSSL_VERSION}")

    set(BORINGSSL_ARGS
        GIT_REPOSITORY ${BORINGSSL_GIT}
        GIT_TAG        ${BORINGSSL_VERSION}
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        list(APPEND BORINGSSL_ARGS EXCLUDE_FROM_ALL)
    endif()

    include(FetchContent)
    FetchContent_Declare(boringssl ${BORINGSSL_ARGS})

    set(SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(SAVED_BUILD_TESTING ${BUILD_TESTING})

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        FetchContent_MakeAvailable(boringssl)
    else()
        FetchContent_GetProperties(boringssl)
        if(NOT boringssl_POPULATED)
            FetchContent_Populate(boringssl)
            add_subdirectory(${boringssl_SOURCE_DIR} ${boringssl_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${SAVED_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${SAVED_BUILD_TESTING})

    license_add_file("BoringSSL" "${boringssl_SOURCE_DIR}/LICENSE")

    set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
    target_link_libraries(${TARGET} PUBLIC ssl crypto)

elseif (LLAMA_BUILD_LIBRESSL)
    set(LIBRESSL_VERSION "4.4.0" CACHE STRING "LibreSSL version")

    message(STATUS "Fetching LibreSSL version ${LIBRESSL_VERSION}")

    set(LIBRESSL_ARGS
        URL "https://cdn.openbsd.org/pub/OpenBSD/LibreSSL/libressl-${LIBRESSL_VERSION}.tar.gz"
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        list(APPEND LIBRESSL_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        list(APPEND LIBRESSL_ARGS EXCLUDE_FROM_ALL)
    endif()

    include(FetchContent)
    FetchContent_Declare(libressl ${LIBRESSL_ARGS})

    set(SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(SAVED_BUILD_TESTING ${BUILD_TESTING})

    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTING OFF)

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        FetchContent_MakeAvailable(libressl)
    else()
        FetchContent_GetProperties(libressl)
        if(NOT libressl_POPULATED)
            FetchContent_Populate(libressl)
            add_subdirectory(${libressl_SOURCE_DIR} ${libressl_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${SAVED_BUILD_SHARED_LIBS})
    set(BUILD_TESTING ${SAVED_BUILD_TESTING})

    license_add_file("LibreSSL" "${libressl_SOURCE_DIR}/COPYING")

    set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
    target_link_libraries(${TARGET} PUBLIC ssl crypto)

elseif (LLAMA_OPENSSL)
    find_package(OpenSSL)
    if (OpenSSL_FOUND)
        include(CheckCSourceCompiles)
        set(SAVED_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
        set(CMAKE_REQUIRED_INCLUDES ${OPENSSL_INCLUDE_DIR})
        check_c_source_compiles("
        #include <openssl/opensslv.h>
        #if defined(OPENSSL_IS_BORINGSSL) || defined(LIBRESSL_VERSION_NUMBER)
        #    if OPENSSL_VERSION_NUMBER < 0x1010107f
        #        error bad version
        #    endif
        #else
        #    if OPENSSL_VERSION_NUMBER < 0x30000000L
        #        error bad version
        #    endif
        #endif
        int main() { return 0; }
        " OPENSSL_VERSION_SUPPORTED)
        set(CMAKE_REQUIRED_INCLUDES ${SAVED_CMAKE_REQUIRED_INCLUDES})
        if (OPENSSL_VERSION_SUPPORTED)
            message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")
            set(CPPHTTPLIB_OPENSSL_SUPPORT TRUE)
            target_link_libraries(${TARGET} PUBLIC OpenSSL::SSL OpenSSL::Crypto)
        endif()
    else()
        message(WARNING "OpenSSL not found, HTTPS support disabled")
    endif()
endif()

# disable warnings in 3rd party code -- same LLAMA_ALL_WARNINGS_3RD_PARTY gate as above.
# BoringSSL and LibreSSL are fetched 3rd-party libraries, so the option covers them by its own
# wording. Turning it ON will be loud here; that is what asking for all warnings in 3rd party
# libs means, and it is OFF by default.
if((LLAMA_BUILD_BORINGSSL OR LLAMA_BUILD_LIBRESSL) AND NOT LLAMA_ALL_WARNINGS_3RD_PARTY)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(ssl PRIVATE /w)
        target_compile_options(crypto PRIVATE /w)
        if(LLAMA_BUILD_BORINGSSL)
            target_compile_options(fipsmodule PRIVATE /w)
        endif()
        if(LLAMA_BUILD_LIBRESSL)
            target_compile_options(ssl_obj PRIVATE /w)
            target_compile_options(bs_obj PRIVATE /w)
            target_compile_options(compat_obj PRIVATE /w)
            target_compile_options(crypto_obj PRIVATE /w)
        endif()
    else()
        target_compile_options(ssl PRIVATE -w)
        target_compile_options(crypto PRIVATE -w)
        if(LLAMA_BUILD_BORINGSSL)
            target_compile_options(fipsmodule PRIVATE -w)
        endif()
        if(LLAMA_BUILD_LIBRESSL)
            target_compile_options(ssl_obj PRIVATE -w)
            target_compile_options(bs_obj PRIVATE -w)
            target_compile_options(compat_obj PRIVATE -w)
            target_compile_options(crypto_obj PRIVATE -w)
        endif()
    endif()
endif()

if (CPPHTTPLIB_OPENSSL_SUPPORT)
    target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT) # used in server.cpp
    if (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
        find_library(SECURITY_FRAMEWORK Security REQUIRED)
        target_link_libraries(${TARGET} PUBLIC ${CORE_FOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
    endif()
    if (WIN32 AND NOT MSVC)
        target_link_libraries(${TARGET} PUBLIC crypt32)
    endif()
endif()
