# Vendored for tools/server's G3 SQL session store (row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01). Only built when the
# opt-in CMake option LLAMA_SERVER_SQLITE3 is ON (default OFF, declared in the root CMakeLists.txt
# next to LLAMA_CURL) -- this file is only reached via vendor/CMakeLists.txt's own `if
# (LLAMA_SERVER_SQLITE3)` guard, so no `if()` is needed here too. See README.md in this directory
# for provenance (SQLite 3.49.1 amalgamation, public domain; 3.49.1 is the minimum the G4
# sqlite-zstd extension accepts).
license_add_file("SQLite" "LICENSE")

set(TARGET vendor-sqlite3)

llama_add_compile_flags()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${TARGET} STATIC
    sqlite3.c
    sqlite3.h
)

add_library(vendor::sqlite3 ALIAS ${TARGET})

target_compile_features(${TARGET} PRIVATE c_std_11)

# This is a stock amalgamation build for one purpose (a 3-table key/value store via
# sqlite_modern_cpp) -- no FTS/RTree/JSON1/zstd/zlib extension flags, no loadable-extension
# support (SQLITE_OMIT_LOAD_EXTENSION is intentionally NOT set: the amalgamation's default is
# already "extension loading available but nothing loads one unless asked"; G4's
# --sqlite-zstd-ext-file is the one caller, and it enables loading only around that call). SQLITE_THREADSAFE=1 (serialized mode) because llama-server runs the HTTP
# handlers touching this database from multiple worker threads with no additional locking of their
# own -- G3's own handlers rely on sqlite's serialized mode being the concurrency boundary, not on
# any mutex on the C++ side.
target_compile_definitions(${TARGET} PRIVATE
    SQLITE_THREADSAFE=1
    SQLITE_DQS=0
)

# disable warnings in 3rd party code, matching every other vendor/* target in this tree (see
# vendor/hash/CMakeLists.txt for the longer note on LLAMA_ALL_WARNINGS_3RD_PARTY)
if (NOT LLAMA_ALL_WARNINGS_3RD_PARTY)
    if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set(NO_WARN_FLAG /w)
    else()
        set(NO_WARN_FLAG -w)
    endif()
    set_source_files_properties(sqlite3.c PROPERTIES COMPILE_OPTIONS ${NO_WARN_FLAG})
endif()

# PUBLIC and "." (not ".." as most vendor/* targets use): sqlite_modern_cpp.h and every other
# consumer expects the conventional bare `#include <sqlite3.h>`, not `<sqlite3/sqlite3.h>` -- this
# is third-party code whose own include spelling is not ours to change.
target_include_directories(${TARGET} PUBLIC .)
