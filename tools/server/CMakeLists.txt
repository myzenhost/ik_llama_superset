include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-radix-cache.cpp
    server-radix-cache.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
    server-asr-words.cpp
    server-asr-words.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ../mtmd is PUBLIC, not PRIVATE, 2026-09-13. server-common.h:7 -- a header this target EXPOSES,
# reached from server-chat.h -- does `#include "mtmd.h"`. PRIVATE means "needed to build me, not to
# use me", which is false for a directory named by a public header: anything that includes
# server-chat.h needs it too. tests/test-chat.cpp links server-context and adds
# ${PROJECT_SOURCE_DIR}/tools/server (tests/CMakeLists.txt:163), then died on
# `fatal error: mtmd.h: No such file or directory`. PUBLIC makes it a usage requirement, which is
# what it always was -- the alternative, repeating ../mtmd at every consumer, is the same fact
# written once per caller and wrong again the next time someone includes this header.
# ABSOLUTE, not `../mtmd`. PUBLIC alone was not enough and the rebuild proved it: test-chat's
# compile line still had no -I for mtmd. A relative path in a PUBLIC/INTERFACE include directory is
# resolved against the CONSUMER's source directory, not this one, so `../mtmd` meant tests/../mtmd
# for test-chat -- a directory that does not exist. It only ever worked for this target's own
# sources, where PRIVATE resolves against here.
# --asr-quality: the whisper.cpp / faster-whisper retry gates live in examples/audio (pure functions, no llama dependency); compiled in here
# rather than linked, because examples/audio is configured after tools/. zlib when found, the documented estimator otherwise.
set(ASR_AUDIO_DIR ${CMAKE_SOURCE_DIR}/examples/audio)
target_sources(${TARGET} PRIVATE ${ASR_AUDIO_DIR}/asr-fallback.cpp ${ASR_AUDIO_DIR}/asr-compression.cpp)
# timestamp_granularities[]=word: the aligner llama-audio --word-timestamps uses (attention rows -> median filter -> DTW), plus the
# capture module shared with it (server-asr-words.h). Compiled in, like the two above.
target_sources(${TARGET} PRIVATE ${ASR_AUDIO_DIR}/asr-align.cpp ${ASR_AUDIO_DIR}/asr-dtw.cpp ${ASR_AUDIO_DIR}/asr-halluc.cpp ${ASR_AUDIO_DIR}/asr-align-capture.cpp)
target_include_directories(${TARGET} PRIVATE ${ASR_AUDIO_DIR})
find_package(ZLIB QUIET)
if (ZLIB_FOUND)
    target_compile_definitions(${TARGET} PRIVATE ASR_HAVE_ZLIB=1)
    target_link_libraries(${TARGET} PRIVATE ZLIB::ZLIB)
endif()
# --asr-vad: the native Silero VAD gate, LINKED (not compiled in a second time) from the asr-vad library that llama-audio
# also uses; asr-vad.cmake defines it once, whichever of examples/audio or here includes it first.
include(${ASR_AUDIO_DIR}/asr-vad.cmake)
target_link_libraries(${TARGET} PRIVATE asr-vad)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd llama-decision ${CMAKE_THREAD_LIBS_INIT})

# G3 SQL session store (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
# Opt-in: only this target (server-context, which holds server-context.cpp) needs
# SQLITE3_MODERN_CPP_SUPPORT and the two vendored headers -- kept PRIVATE so turning this option ON
# does not change the ABI/definitions any other target in the build sees.
if (LLAMA_SERVER_SQLITE3)
    target_compile_definitions(${TARGET} PRIVATE SQLITE3_MODERN_CPP_SUPPORT)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sqlite_modern_cpp/hdr)
    target_link_libraries(${TARGET} PRIVATE vendor::sqlite3)
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-connect.cpp
    server-connect.h
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
    server-vram.cpp
    server-vram.h
    server-vram-probe.h
    # Ollama-native API translation shim (row
    # S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01). Lives in this library, not the
    # server-context static lib above, because it is a route-registration/HTTP-translation layer
    # (like server-http.cpp/server-models.cpp beside it), not core inference logic -- it calls
    # server_routes's PUBLIC handlers rather than adding to server_routes itself, so it needs no
    # changes to the server-context target's sources.
    server-ollama.cpp
    server-ollama.h
    # TGI native API routes (/generate, /generate_stream, POST /, /info), same edge-translation shape,
    # row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01
    server-tgi.cpp
    server-tgi.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
if (NOT WIN32)
    target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS}) # server-vram.cpp dlopen()s NVML
else()
    # server.cpp:23-24 `#include <dbghelp.h>` + `#pragma comment(lib, "dbghelp.lib")` for the
    # Windows crash-backtrace handler (SymInitialize/StackWalk64/SymFromAddr/...). The pragma is
    # an MSVC auto-link directive; MinGW g++ silently ignores it, so a MinGW build links every
    # object fine and fails only at the final llama-server.exe link with `undefined reference to
    # __imp_SymInitialize` etc. MEASURED 2026-09-26: CPU-only static build (build-cpu, gcc/g++/
    # ninja per the engine-merge protocol's exact recipe) failed this way identically on BOTH a
    # pre-bad_words commit (61f95e629) and the bad_words commit (140aecffc) -- same undefined
    # symbols, same object file (server.cpp.obj) -- so this is a toolchain gap unrelated to any
    # sampling change, not something introduced by this pass. dbghelp.dll ships with every Windows
    # install, so linking it unconditionally on WIN32 is harmless for MSVC too (the pragma already
    # requests it there; this just makes MinGW request it as well).
    target_link_libraries(${TARGET} PRIVATE dbghelp)
endif()
# (the same fix arrived independently on engine/unsloth-models-2026-09-27 and engine/sglang-radix-2026-09-27,
#  all branched before it landed; one copy kept at integration)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
# dbghelp arrives through llama-server-impl above (a STATIC library's PRIVATE link deps propagate to the final
# link); the server-routes branch re-added a direct copy here on merge -- dropped again, as in 1e89cb61c.
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Server-level tests of the wired serving features (LoRA pool / heterogeneous batching, koboldcpp banned_tokens + prompt-reuse planner, ASR quality retry):
# real llama-server on a random-weight model, real HTTP. Needs numpy + gguf-py (writes the assets); exits 77 (SKIPPED) without them.
if (TARGET llama-server)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-server-serving-features
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests_cpp/test_serving_features.py --server $<TARGET_FILE:llama-server>)
        set_tests_properties(test-server-serving-features PROPERTIES SKIP_RETURN_CODE 77 LABELS main TIMEOUT 900)
    endif()
endif()
