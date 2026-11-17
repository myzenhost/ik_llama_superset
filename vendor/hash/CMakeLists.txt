set(TARGET vendor-hash)

llama_add_compile_flags()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(VENDOR_SRCS
    xxhash/xxhash.c
    sha1/sha1.c
    sha256/sha256.c
)

add_library(${TARGET} STATIC
    hash.cpp
    hash.h
    ${VENDOR_SRCS}
)

add_library(vendor::hash ALIAS ${TARGET})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# disable warnings in 3rd party code, but keep them for hash.cpp -- unless
# LLAMA_ALL_WARNINGS_3RD_PARTY asks to keep them for the vendored sources too. See the longer note
# in vendor/cpp-httplib/CMakeLists.txt: the option had no reader anywhere, and this suppression is
# what made its ON state unreachable. OFF (the default) suppresses exactly as before.
if (NOT LLAMA_ALL_WARNINGS_3RD_PARTY)
    if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set(NO_WARN_FLAG /w)
    else()
        set(NO_WARN_FLAG -w)
    endif()
    set_source_files_properties(${VENDOR_SRCS} PROPERTIES COMPILE_OPTIONS ${NO_WARN_FLAG})
endif()

# sha1 lives in a namespace to avoid a clash with boringssl, see scripts/sync_vendor.py
set_source_files_properties(sha1/sha1.c PROPERTIES LANGUAGE CXX)

# expose the vendor/ root so consumers can include via "hash/hash.h"
target_include_directories(${TARGET} PUBLIC ..)

# internal includes of the vendored sources, e.g. sha256.c -> "rotate-bits/rotate-bits.h"
target_include_directories(${TARGET} PRIVATE .)
