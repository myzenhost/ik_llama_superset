# mtmd needs these even when common is not built
add_subdirectory(hash)
add_subdirectory(miniaudio)
add_subdirectory(nlohmann)
add_subdirectory(sheredom)
add_subdirectory(stb)

# only used by common
if (LLAMA_BUILD_COMMON)
    add_subdirectory(cpp-httplib)
endif()

# only used by tools/server's G3 SQL session store (row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01); option declared in
# the root CMakeLists.txt, default OFF.
if (LLAMA_SERVER_SQLITE3)
    add_subdirectory(sqlite3)
endif()
