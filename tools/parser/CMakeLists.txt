if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # this tool is disabled on Windows when building with shared libraries because it uses internal functions not exported with LLAMA_API
    set(TARGET llama-debug-template-parser)
    add_executable(${TARGET} debug-template-parser.cpp)
    # `common` -> `llama-common`, 2026-09-13. There is no CMake target named `common` in this tree:
    # common/CMakeLists.txt defines llama-common (:54) and llama-common-base (:41). CMake does not
    # error on an unknown name here -- it passes it through to the linker as a raw `-lcommon` -- so
    # the build configured cleanly while NO usage requirements propagated, including the
    # `target_include_directories(${TARGET} PUBLIC .)` at common/CMakeLists.txt:139. That is why
    # both files in this directory died on `fatal error: chat-auto-parser.h: No such file or
    # directory` while the header sits in common/, and why every other tool that links llama-common
    # (tools/llama-bench:9, tools/server:34) finds its headers.
    target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)

    if(LLAMA_TOOLS_INSTALL)
        install(TARGETS ${TARGET} RUNTIME)
    endif()
endif()

set(TARGET llama-template-analysis)
add_executable(${TARGET} template-analysis.cpp)
# `common` -> `llama-common` here too; see the note on the other target above. Both files in this
# directory failed the same way, so fixing only the indented occurrence would have left half the
# defect and looked like a partial fix in the next build.
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
