include(CheckCXXCompilerFlag)
include("../cmake/common.cmake")

add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (GGML_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (GGML_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (GGML_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

if (GGML_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

if (GGML_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (GGML_CCACHE AND NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(GGML_CCACHE_FOUND ccache)
    find_program(GGML_SCCACHE_FOUND sccache)

    if (GGML_CCACHE_FOUND OR GGML_SCCACHE_FOUND)
        if(GGML_CCACHE_FOUND)
            set(GGML_CCACHE_VARIANT ccache)
        else()
            set(GGML_CCACHE_VARIANT sccache)
        endif()
        # TODO: should not be set globally
        if (GGML_SYCL AND GGML_CCACHE_FOUND AND WIN32)
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache compiler_type=icl")
        else ()
            set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${GGML_CCACHE_VARIANT}")
        endif ()
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "${GGML_CCACHE_VARIANT} found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()
ggml_get_system_arch()
message(STATUS "GGML_SYSTEM_ARCH: ${GGML_SYSTEM_ARCH}")

if (NOT MSVC)
    if (GGML_STATIC)
        if (UNIX AND NOT APPLE)
            set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")
        endif()
        add_link_options(-static)   # directory-scoped only; see GGML_STATIC block below
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (GGML_GPROF)
        add_compile_options(-pg)
    endif()
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_XOPEN_SOURCE=700)
elseif (CMAKE_SYSTEM_NAME MATCHES "AIX")
    # Don't define _XOPEN_SOURCE.  We need _ALL_SOURCE, which is the default,
    # in order to define _SC_PHYS_PAGES.
else()
    add_compile_definitions(_XOPEN_SOURCE=600)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux" OR CMAKE_SYSTEM_NAME MATCHES "Android")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS"    OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS"   OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# mingw-w64 defaults _WIN32_WINNT to 0x0601 (Windows 7), which compiles out every
# `#if _WIN32_WINNT >= 0x0602` branch: the EcoQoS opt-out SetThreadInformation call that
# ggml_thread_apply_priority makes for every CPU worker, among others. MSVC's SDK already
# defaults to the newest version, so this only changes MinGW builds.
# SOURCE-READ: ikawrakow/ik_llama.cpp origin/main ggml/src/CMakeLists.txt:1272-1275 (MIT).
# Test: tests/test-win-version-gates.cpp. Row S69-DECL-OVERTURN-THIREUS-DEFER-PLE-WINVER-2026-10-17-01.
if (MINGW)
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

# ggml

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

add_library(ggml-base
            ../include/ggml.h
            ../include/ggml-alloc.h
            ../include/ggml-backend.h
            ../include/ggml-cpp.h
            ../include/ggml-opt.h
            ../include/gguf.h
            ggml.c
            ggml.cpp
            ggml-alloc.c
            ggml-backend.cpp
            ggml-backend-meta.cpp
            ggml-opt.cpp
            ggml-threading.cpp
            ggml-threading.h
            ggml-vram-spill.cpp
            ../include/ggml-vram-spill.h
            ggml-quants.c
            ggml-quants.h
            # iqk/iqk_quantize.cpp itself is NOT here and cannot be: ggml-base is compiled arch-neutral on
            # purpose, and that file's quantizer search / repack code has AVX2, AVX512 and NEON arms
            # (its generic branch of repack_q8_KV, iqk_quantize.cpp(7268) in the 2026-09-15 tree, did not
            # even compile: "error C2065: 'y': undeclared identifier").
            # HALF 2 of S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01 moves the
            # part ggml.c type_traits[] points at: iqk/iqk_quantize_base.cpp holds every .to_float /
            # .from_float_ref implementation whose whole call closure is arch-neutral (bodies moved verbatim
            # out of iqk_quantize.cpp). The roots whose closure reaches an arch arm stay in ggml-cpu.
            iqk/iqk_quantize_base.cpp
            iqk/iqk_quantize_shared.h
            gguf.cpp
            # --- fork-only ggml-base sources ---------------------------------------------------
            # None of these exist in upstream at any path; they are HEAD-only. They were named by
            # the monolith's single `ggml` target and would simply stop being compiled here.
            # ggml-turbo-quant.c is the one that matters most: it is the TurboQuant CPU writer, and
            # the CUDA turbo KV kernels have nothing producing their data without it.
            ggml-turbo-quant.c
            ggml-aarch64.c
            ggml-aarch64.h
            ggml-moe-prefetch.cpp
            ggml-moe-prefetch.h
            ggml-utils.h)

# ggml-backend-dev-compat.cpp is DELIBERATELY NOT in that list, and this note is why.
#
# RESEARCH-AFTER-RESULT: ggml/src/ggml-backend-dev-compat.cpp read in full (7,217 B), together
# with upstream's ggml/src/ggml-backend-reg.cpp (18,002 B, MERGE_HEAD-only, already staged).
# The shim states its own scope in its header comment: "upstream ggml moved its backend registry
# from integer indices to opaque handles; our base predates that", that upstream's mtmd "uses the
# handle spelling in 14 auxiliary places (device selection and memory reporting), so translating
# between the two spellings is enough to compile upstream's mtmd unmodified", and -- the decisive
# line -- "Nothing here dereferences a handle." A handle is literally `index + 1`
# (index_to_dev/dev_to_index), biased so NULL still means "unspecified".
#
# It was added to ggml-base in a first pass and that was WRONG. Extracting the ggml_backend_*
# names from both files gives 16 in the shim and 42 in upstream's registry, and the intersection
# is NINE symbols that both would define:
#     ggml_backend_dev_backend_reg   ggml_backend_dev_count    ggml_backend_dev_description
#     ggml_backend_dev_get           ggml_backend_dev_init     ggml_backend_dev_name
#     ggml_backend_dev_type          ggml_backend_init_by_type ggml_backend_load_all
# ggml-backend-reg.cpp builds into the `ggml` target below, so compiling the shim into ggml-base
# is a duplicate-definition link failure -- nine errors with no obvious connection to a CMake edit.
#
# Not a capability loss: the shim exists to APPROXIMATE the device API that ggml-backend-reg.cpp
# now provides for real, backed by actual device objects rather than biased integers, and it
# never had any behaviour of its own to lose ("nothing here dereferences a handle"). Left on disk
# rather than deleted -- deleting a fork file is a separate decision from not compiling it.

set_target_properties(ggml-base PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

configure_file(ggml-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.h @ONLY)

target_include_directories(ggml-base PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
if (GGML_BACKEND_DL)
    target_compile_definitions(ggml-base PUBLIC GGML_BACKEND_DL)
endif()
if (GGML_BACKEND_DL OR BUILD_SHARED_LIBS)
    # ggml.c / ggml-quants.c / iqk_quantize_base.cpp name ~160 functions the CPU backend defines; a DL ggml-base
    # cannot link them, and neither can a shared libggml-base under -Wl,-z,defs (or a macOS dylib, or an MSVC DLL):
    # it gets forwarding stubs bound by the loaded CPU library (ggml-cpu-syms.h explains, ggml-cpu-syms.inc lists
    # them). Static builds do not compile this. GGML_BASE_CPU_SYMS is the switch every half of the mechanism tests.
    target_compile_definitions(ggml-base PUBLIC GGML_BASE_CPU_SYMS)
    target_sources(ggml-base PRIVATE ggml-cpu-syms.cpp ggml-cpu-syms.h ggml-cpu-syms.inc)
endif()

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

if (GGML_OPENMP_FETCH)
    if (NOT GGML_OPENMP)
        message(FATAL_ERROR "GGML_OPENMP_FETCH requires GGML_OPENMP")
    elseif (NOT WIN32 OR NOT (CMAKE_C_COMPILER_ID MATCHES "Clang"))
        message(FATAL_ERROR "GGML_OPENMP_FETCH currently requires Clang on Windows")
    endif()

    set(GGML_OPENMP_LLVM_VERSION "20.1.8")
    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_LLVM_VERSION_MAJOR "${GGML_OPENMP_LLVM_VERSION}")
    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_COMPILER_VERSION_MAJOR "${CMAKE_C_COMPILER_VERSION}")
    if (NOT GGML_OPENMP_COMPILER_VERSION_MAJOR STREQUAL GGML_OPENMP_LLVM_VERSION_MAJOR)
        message(FATAL_ERROR "LLVM OpenMP ${GGML_OPENMP_LLVM_VERSION} requires Clang ${GGML_OPENMP_LLVM_VERSION_MAJOR}.x")
    endif()

    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GGML_OPENMP_SYSTEM_PROCESSOR)
    if (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(amd64|x86_64)$")
        set(GGML_OPENMP_ARCH "x64")
        set(GGML_OPENMP_INSTALLER_SUFFIX "win64")
        set(GGML_OPENMP_INSTALLER_SHA256 "3197846a2b19063687dd56e93e34cd941e3548d907f23a6131571321bdf9fe7b")
    elseif (GGML_OPENMP_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(GGML_OPENMP_ARCH "arm64")
        set(GGML_OPENMP_INSTALLER_SUFFIX "woa64")
        set(GGML_OPENMP_INSTALLER_SHA256 "7c4ac97eb2ae6b960ca5f9caf3ff6124c8d2a18cc07a7840a4d2ea15537bad8e")
    else()
        message(FATAL_ERROR "GGML_OPENMP_FETCH does not support ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(GGML_OPENMP_CACHE_DIR "${CMAKE_BINARY_DIR}/_deps")
    set(GGML_OPENMP_ROOT "${GGML_OPENMP_CACHE_DIR}/llvm-openmp-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_ARCH}")
    set(GGML_OPENMP_LIBRARY "${GGML_OPENMP_ROOT}/lib/libomp.lib")
    set(GGML_OPENMP_RUNTIME "${GGML_OPENMP_ROOT}/bin/libomp.dll")
    set(GGML_OPENMP_HEADER "${GGML_OPENMP_ROOT}/include/omp.h")
    set(GGML_OPENMP_LICENSE "${GGML_OPENMP_ROOT}/LICENSE.TXT")
    set(GGML_OPENMP_LICENSE_SHA256 "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4")

    if (NOT EXISTS "${GGML_OPENMP_LIBRARY}" OR NOT EXISTS "${GGML_OPENMP_RUNTIME}" OR NOT EXISTS "${GGML_OPENMP_HEADER}")
        find_program(GGML_OPENMP_7Z NAMES 7z 7zz 7za)
        if (NOT GGML_OPENMP_7Z)
            message(FATAL_ERROR "GGML_OPENMP_FETCH requires 7-Zip to extract the LLVM installer")
        endif()

        set(GGML_OPENMP_INSTALLER "${GGML_OPENMP_ROOT}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")
        set(GGML_OPENMP_EXTRACT_DIR "${GGML_OPENMP_ROOT}/extract")
        set(GGML_OPENMP_INSTALLER_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${GGML_OPENMP_LLVM_VERSION}/LLVM-${GGML_OPENMP_LLVM_VERSION}-${GGML_OPENMP_INSTALLER_SUFFIX}.exe")

        file(MAKE_DIRECTORY "${GGML_OPENMP_EXTRACT_DIR}")
        file(DOWNLOAD "${GGML_OPENMP_INSTALLER_URL}" "${GGML_OPENMP_INSTALLER}"
            EXPECTED_HASH "SHA256=${GGML_OPENMP_INSTALLER_SHA256}"
            SHOW_PROGRESS
            STATUS GGML_OPENMP_DOWNLOAD_STATUS)
        list(GET GGML_OPENMP_DOWNLOAD_STATUS 0 GGML_OPENMP_DOWNLOAD_RESULT)
        if (NOT GGML_OPENMP_DOWNLOAD_RESULT EQUAL 0)
            list(GET GGML_OPENMP_DOWNLOAD_STATUS 1 GGML_OPENMP_DOWNLOAD_ERROR)
            message(FATAL_ERROR "Failed to download LLVM OpenMP: ${GGML_OPENMP_DOWNLOAD_ERROR}")
        endif()

        execute_process(
            COMMAND "${GGML_OPENMP_7Z}" e -y "-o${GGML_OPENMP_EXTRACT_DIR}" "${GGML_OPENMP_INSTALLER}" -r libomp.lib libomp.dll omp.h
            RESULT_VARIABLE GGML_OPENMP_EXTRACT_RESULT
            OUTPUT_QUIET)
        if (NOT GGML_OPENMP_EXTRACT_RESULT EQUAL 0 OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" OR
            NOT EXISTS "${GGML_OPENMP_EXTRACT_DIR}/omp.h")
            message(FATAL_ERROR "Failed to extract libomp from ${GGML_OPENMP_INSTALLER}")
        endif()

        file(MAKE_DIRECTORY "${GGML_OPENMP_ROOT}/lib" "${GGML_OPENMP_ROOT}/bin" "${GGML_OPENMP_ROOT}/include")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.lib" DESTINATION "${GGML_OPENMP_ROOT}/lib")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/libomp.dll" DESTINATION "${GGML_OPENMP_ROOT}/bin")
        file(COPY "${GGML_OPENMP_EXTRACT_DIR}/omp.h" DESTINATION "${GGML_OPENMP_ROOT}/include")
        file(REMOVE_RECURSE "${GGML_OPENMP_INSTALLER}" "${GGML_OPENMP_EXTRACT_DIR}")
    endif()

    # The NSIS installer embeds LLVM's general license in its UI but does not install it as a file; use OpenMP's license to include its additional notices.
    if (EXISTS "${GGML_OPENMP_LICENSE}")
        file(SHA256 "${GGML_OPENMP_LICENSE}" GGML_OPENMP_LICENSE_ACTUAL_SHA256)
    endif()
    if (NOT GGML_OPENMP_LICENSE_ACTUAL_SHA256 STREQUAL GGML_OPENMP_LICENSE_SHA256)
        file(DOWNLOAD "https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-${GGML_OPENMP_LLVM_VERSION}/openmp/LICENSE.TXT" "${GGML_OPENMP_LICENSE}"
            EXPECTED_HASH "SHA256=${GGML_OPENMP_LICENSE_SHA256}")
    endif()

    if (COMMAND license_add_file)
        license_add_file("LLVM OpenMP" "${GGML_OPENMP_LICENSE}")
    endif()

    add_library(ggml-openmp-c INTERFACE)
    target_compile_options(ggml-openmp-c INTERFACE "$<$<COMPILE_LANGUAGE:C>:-fopenmp=libomp>")
    target_include_directories(ggml-openmp-c SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
    target_link_libraries(ggml-openmp-c INTERFACE "${GGML_OPENMP_LIBRARY}")

    add_library(ggml-openmp-cxx INTERFACE)
    target_compile_options(ggml-openmp-cxx INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp=libomp>")
    target_include_directories(ggml-openmp-cxx SYSTEM INTERFACE "${GGML_OPENMP_ROOT}/include")
    target_link_libraries(ggml-openmp-cxx INTERFACE "${GGML_OPENMP_LIBRARY}")

    set(GGML_OPENMP_RUNTIME_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if (CMAKE_CONFIGURATION_TYPES)
        string(APPEND GGML_OPENMP_RUNTIME_OUTPUT_DIR "/$<CONFIG>")
    endif()
    add_custom_target(ggml-openmp-runtime ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_RUNTIME}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/libomp.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GGML_OPENMP_LICENSE}" "${GGML_OPENMP_RUNTIME_OUTPUT_DIR}/LICENSE-LLVM-OpenMP")
    add_dependencies(ggml-base ggml-openmp-runtime)
    install(FILES "${GGML_OPENMP_RUNTIME}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES "${GGML_OPENMP_LICENSE}" DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME LICENSE-LLVM-OpenMP)

    set(GGML_OPENMP_TARGET_C ggml-openmp-c)
    set(GGML_OPENMP_TARGET_CXX ggml-openmp-cxx)
    set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
elseif (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        set(GGML_OPENMP_TARGET_C OpenMP::OpenMP_C)
        set(GGML_OPENMP_TARGET_CXX OpenMP::OpenMP_CXX)
        set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")

        # Absorb libgomp under GGML_STATIC, so a static build is actually self-contained.
        # The OpenMP find module names the IMPORT library by ABSOLUTE PATH (libgomp.dll.a), and an
        # absolute path is not subject to -static -- that flag only governs how -lNAME resolves.
        # MEASURED 2026-09-13: with -static working (imports 9 -> 6, libgcc/libstdc++/winpthread
        # all absorbed), libgomp-1.dll still remained and the binary would not start on a clean
        # PATH. The static archive is already on disk beside the import library; this points the
        # link at it. Not found -> warn and leave the dynamic link alone, because degrading to a
        # working dynamic build is correct and degrading to a broken one is not.
        if (GGML_STATIC AND MINGW)
            find_library(GGML_GOMP_STATIC NAMES libgomp.a)
            # libgomp.a carries OpenACC offload-target support that calls dlopen/dlsym/dlclose/
            # dlerror. The DLL resolved those internally; the static archive expects the caller to
            # supply them, so a static link fails with dozens of undefined dl* references unless
            # MinGW's dl shim is linked after it. MEASURED 2026-09-13.
            # HINTS matter: the MinGW dl shim lives in the toolchain SYSROOT
            # (<prefix>/x86_64-w64-mingw32/lib), which is not on find_library's default
            # search path here. MEASURED: a bare find_library(... NAMES dl) reported
            # NOTFOUND while libdl.a sat in that directory, and the guard below then
            # correctly -- but unnecessarily -- fell back to the dynamic link.
            get_filename_component(GGML_MINGW_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
            get_filename_component(GGML_MINGW_ROOT "${GGML_MINGW_BIN}" DIRECTORY)
            # NAMES libdl.a, not NAMES dl: on MinGW find_library PREFERS the .dll.a IMPORT
            # library, and MEASURED 2026-09-13 it resolved to libdl.dll.a -- which simply
            # traded an import on libgomp-1.dll for one on libdl.dll. The real static
            # archive (dlfcn.o) needs only Win32 calls already imported from KERNEL32.
            find_library(GGML_DL_LIB NAMES libdl.a
                         HINTS "${GGML_MINGW_ROOT}/x86_64-w64-mingw32/lib"
                               "${GGML_MINGW_ROOT}/lib")
            if (GGML_GOMP_STATIC AND NOT GGML_DL_LIB)
                message(WARNING
                    "GGML_STATIC: libgomp.a found but no dl shim; leaving OpenMP dynamic because a "
                    "static libgomp cannot link without dlopen/dlsym.")
                set(GGML_GOMP_STATIC "GGML_GOMP_STATIC-NOTFOUND")
            endif()
            if (GGML_GOMP_STATIC)
                set_property(TARGET OpenMP::OpenMP_C  PROPERTY
                             INTERFACE_LINK_LIBRARIES "${GGML_GOMP_STATIC};${GGML_DL_LIB}")
                set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
                             INTERFACE_LINK_LIBRARIES "${GGML_GOMP_STATIC};${GGML_DL_LIB}")
                message(STATUS "GGML_STATIC: linking OpenMP statically: ${GGML_GOMP_STATIC}")
            else()
                message(WARNING
                    "GGML_STATIC is ON but libgomp.a was not found; the binary will still "
                    "require libgomp-1.dll at runtime.")
            endif()
        endif()
    else()
        set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
        message(WARNING "OpenMP not found")
    endif()
else()
    set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
endif()

if (GGML_OPENMP_ENABLED)
    target_compile_definitions(ggml-base PRIVATE GGML_USE_OPENMP)
    target_link_libraries(ggml-base PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
endif()

add_library(ggml
            ggml-backend-dl.cpp
            ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

set_target_properties(ggml PROPERTIES
    VERSION ${GGML_VERSION}
    SOVERSION ${GGML_VERSION_MAJOR}
)

if (GGML_BACKEND_DIR)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_BACKEND_DIR requires GGML_BACKEND_DL")
    endif()
    target_compile_definitions(ggml PUBLIC GGML_BACKEND_DIR="${GGML_BACKEND_DIR}")
endif()

target_link_libraries(ggml PUBLIC ggml-base)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml PRIVATE dl)
endif()

# =================================================================================================
# FORK CARRY-FORWARD -- macros the monolith defined that upstream's dispatcher does not.
#
# These were add_compile_definitions() in ggml/src/CMakeLists.txt, i.e. DIRECTORY-scoped, so they
# reached every source under ggml/src including ggml.c, ggml-alloc.c and ggml-backend.cpp. They are
# set at the same scope here, BEFORE the backend subdirectories are entered, so the subdirectories
# inherit them exactly as they did from the monolith.
#
# Why not target_compile_definitions(ggml-base ...)? Because the IQK macros below are read by BOTH
# ggml-base files (ggml.c, ggml-backend.cpp, ggml-quants.c) AND ggml-cpu files (iqk_quantize.cpp
# and the 21 iqk sources). Directory scope is what covers both, and putting them on one target is
# precisely the bug that made the first version of the ggml-cpu edit compile clean while switching
# the fork's IQK paths off in the core.
#
# Measured readers, so nothing here is decorative:
#   GGML_MAX_SRC             30   ggml-alloc.c, ggml-backend.cpp, ggml.c, and 27 more
#   GGML_IQK_FLASH_ATTENTION 12   iqk/ (each FA unit compiles to nothing without it)
#   GGML_IQK_FA_ALL_QUANTS    5   iqk/fa/
#   GGML_USE_IQK_MULMAT       4   ggml.c, ggml-backend.cpp, ggml-quants.c, iqk/iqk_quantize.cpp
#   GGML_EXPERT_CHUNKING      1   ggml.c
#   GGML_MAX_CONTEXTS         1   ggml.c
# =================================================================================================

if (GGML_MAX_CONTEXTS)
    add_compile_definitions(GGML_MAX_CONTEXTS=${GGML_MAX_CONTEXTS})
endif()

if (GGML_MAX_SRC)
    add_compile_definitions(GGML_MAX_SRC=${GGML_MAX_SRC})
endif()

if (GGML_EXPERT_CHUNKING)
    add_compile_definitions(GGML_EXPERT_CHUNKING)
endif()

# The IQK SOURCES are named in ggml-cpu/CMakeLists.txt, where the CPU target is built. Only the
# DEFINITIONS live here, for the scope reason above.
if (GGML_IQK_MUL_MAT)
    message(STATUS "Using optimized iqk matrix multiplications")
    add_compile_definitions(GGML_USE_IQK_MULMAT)

    if (GGML_IQK_FLASH_ATTENTION)
        message(STATUS "Enabling IQK Flash Attention kernels")
        add_compile_definitions(GGML_IQK_FLASH_ATTENTION)
        if (GGML_IQK_FA_ALL_QUANTS)
            message(STATUS "Including all IQK FA kernels")
            add_compile_definitions(GGML_IQK_FA_ALL_QUANTS)
        endif()
    else()
        message(STATUS "Disabling IQK Flash Attention kernels")
    endif()
endif()

function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN})
        # write the shared library to the output directory
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)
        add_dependencies(ggml ${backend})
        if (GGML_BACKEND_DIR)
            install(TARGETS ${backend} LIBRARY DESTINATION ${GGML_BACKEND_DIR})
        else()
            install(TARGETS ${backend} LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
        install(TARGETS ${backend} LIBRARY)
    endif()

    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC  GGML_BACKEND_SHARED)
    endif()

    # Set versioning properties for all backend libraries
    # Building a MODULE library with a version is not supported on macOS (https://gitlab.kitware.com/cmake/cmake/-/issues/20782)
    if (NOT (APPLE AND GGML_BACKEND_DL))
        set_target_properties(${backend} PROPERTIES
            VERSION ${GGML_VERSION}
            SOVERSION ${GGML_VERSION_MAJOR}
        )
    endif()

    if(NOT GGML_AVAILABLE_BACKENDS)
        set(GGML_AVAILABLE_BACKENDS "${backend}"
            CACHE INTERNAL "List of backends for cmake package")
    else()
        list(FIND GGML_AVAILABLE_BACKENDS "${backend}" has_backend)
        if(has_backend EQUAL -1)
            set(GGML_AVAILABLE_BACKENDS "${GGML_AVAILABLE_BACKENDS};${backend}"
                CACHE INTERNAL "List of backends for cmake package")
        endif()
    endif()
endfunction()

function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "ggml-${backend}" backend_target)
        add_subdirectory(${backend_target})
        message(STATUS "Including ${backend} backend")
        if (NOT GGML_BACKEND_DL)
            string(TOUPPER "GGML_USE_${backend}" backend_use)
            target_compile_definitions(ggml PUBLIC ${backend_use})
        endif()
    endif()
endfunction()

function(ggml_add_cpu_backend_variant tag_name)
    set(GGML_CPU_TAG_NAME ${tag_name})
    # other: OPENMP LLAMAFILE CPU_HBM
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        foreach (feat NATIVE
                      SSE42
                      AVX AVX2 BMI2 AVX_VNNI FMA F16C
                      AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
                      AMX_TILE AMX_INT8 AMX_BF16)
            set(GGML_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "ARM")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        set(GGML_NATIVE OFF)

        foreach (feat VXE2 NNPA)
            set(GGML_INTERNAL_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        foreach (feat RVV)
            set(GGML_INTERNAL_${feat} OFF)
        endforeach()

        foreach (feat ${ARGN})
            set(GGML_INTERNAL_${feat} ON)
        endforeach()
    endif()

    ggml_add_cpu_backend_variant_impl(${tag_name})
endfunction()

ggml_add_backend(CPU)

if (GGML_CPU_ALL_VARIANTS)
    if (NOT GGML_BACKEND_DL)
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL")
    elseif (GGML_CPU_ARM_ARCH)
        message(FATAL_ERROR "Cannot use both GGML_CPU_ARM_ARCH and GGML_CPU_ALL_VARIANTS")
    endif()
    if (GGML_SYSTEM_ARCH STREQUAL "x86")
        # Pre-AVX2 variants are OPT-IN in this fork. The CPU backend always compiles iqk/iqk_quantize.cpp and
        # iqk/iqk_cpu_ops.cpp (ggml-cpu/CMakeLists.txt), and iqk_config.h defines IQK_IMPLEMENT only for
        # __AVX2__ / __ARM_FEATURE_DOTPROD. Without it those files' generic paths do not compile. MEASURED
        # 2026-10-20 (MinGW gcc, temp/arex1014/dlsplit_1020/logs/build_dl_pass1_preavx2_fail.log): 15 failed
        # objects, iqk_quantize/iqk_cpu_ops/iqk_mul_mat x {x64, sse42, sandybridge, ivybridge, piledriver}, first
        # error iqk_cpu_ops.cpp:864 'ggml_half' does not name a type. The generic repack_q8_KV branch has never
        # compiled either (see the iqk block in ggml-cpu/CMakeLists.txt). Because ggml depends on EVERY backend
        # (add_dependencies above), one uncompilable variant blocks ggml.dll and every tool. AVX2-and-newer
        # variants are unaffected. Turn this on only together with a fix for the generic iqk paths.
        option(GGML_CPU_ALL_VARIANTS_PRE_AVX2 "ggml: also build the pre-AVX2 x86 CPU variants (needs generic iqk paths)" OFF)
        if (GGML_CPU_ALL_VARIANTS_PRE_AVX2)
            ggml_add_cpu_backend_variant(x64)
            ggml_add_cpu_backend_variant(sse42              SSE42)
            ggml_add_cpu_backend_variant(sandybridge        SSE42 AVX)
            if (NOT MSVC)
                # __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
                ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
                ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
            endif()
        else()
            message(STATUS "GGML_CPU_ALL_VARIANTS: pre-AVX2 x86 variants skipped (GGML_CPU_ALL_VARIANTS_PRE_AVX2=OFF)")
        endif()
        ggml_add_cpu_backend_variant(haswell            SSE42 AVX F16C FMA AVX2 BMI2)
        ggml_add_cpu_backend_variant(skylakex           SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
        ggml_add_cpu_backend_variant(cannonlake         SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI)
        ggml_add_cpu_backend_variant(cascadelake        SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI)
        ggml_add_cpu_backend_variant(icelake            SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
        if (NOT MSVC)
            # MSVC 2022 doesn't support BF16 intrinsics without `/arch:AVX10.1` ?!
            # https://learn.microsoft.com/en-us/cpp/intrinsics/x64-amd64-intrinsics-list?view=msvc-170
            # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
            ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
            ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
        endif()
        ggml_add_cpu_backend_variant(alderlake          SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
        if (NOT MSVC)
            # MSVC doesn't support AMX
            ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
        endif()
    elseif(GGML_SYSTEM_ARCH STREQUAL "ARM")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            # Many of these features are optional so we build versions with popular
            # combinations and name the backends based on the version they were
            # first released with
            ggml_add_cpu_backend_variant(armv8.0_1)
            ggml_add_cpu_backend_variant(armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(armv8.2_3    DOTPROD FP16_VECTOR_ARITHMETIC SVE)
            ggml_add_cpu_backend_variant(armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8)
            ggml_add_cpu_backend_variant(armv8.6_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2)
            ggml_add_cpu_backend_variant(armv9.2_1    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SME)
            ggml_add_cpu_backend_variant(armv9.2_2    DOTPROD FP16_VECTOR_ARITHMETIC SVE MATMUL_INT8 SVE2 SME)
        elseif (CMAKE_SYSTEM_NAME MATCHES "Android")
            # Android-specific backends with SoC-compatible feature sets
            ggml_add_cpu_backend_variant(android_armv8.0_1)
            ggml_add_cpu_backend_variant(android_armv8.2_1    DOTPROD)
            ggml_add_cpu_backend_variant(android_armv8.2_2    DOTPROD FP16_VECTOR_ARITHMETIC)
            ggml_add_cpu_backend_variant(android_armv8.6_1    DOTPROD FP16_VECTOR_ARITHMETIC MATMUL_INT8)
            ggml_add_cpu_backend_variant(android_armv9.0_1    DOTPROD MATMUL_INT8 FP16_VECTOR_ARITHMETIC SVE2)
            ggml_add_cpu_backend_variant(android_armv9.2_1    DOTPROD MATMUL_INT8 FP16_VECTOR_ARITHMETIC SVE SME)
            ggml_add_cpu_backend_variant(android_armv9.2_2    DOTPROD MATMUL_INT8 FP16_VECTOR_ARITHMETIC SVE SVE2 SME)
        elseif (APPLE)
            ggml_add_cpu_backend_variant(apple_m1             DOTPROD)
            ggml_add_cpu_backend_variant(apple_m2_m3          DOTPROD MATMUL_INT8)
            ggml_add_cpu_backend_variant(apple_m4             DOTPROD MATMUL_INT8 NOSVE SME)
        else()
            message(FATAL_ERROR "Unsupported ARM target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux|AIX")
            ggml_add_cpu_backend_variant(power0)
            ggml_add_cpu_backend_variant(power7_1       POWER7)
            ggml_add_cpu_backend_variant(power7_2       POWER7  VSX)
            ggml_add_cpu_backend_variant(power8_1       POWER8)
            ggml_add_cpu_backend_variant(power8_2       POWER8  VSX)
            ggml_add_cpu_backend_variant(power9         POWER9  VSX)
            ggml_add_cpu_backend_variant(power10        POWER10 VSX)
            # POWER11 backend: only if compiler supports -mcpu=power11
            check_cxx_compiler_flag("-mcpu=power11" GGML_CXX_SUPPORTS_POWER11)
            if (GGML_CXX_SUPPORTS_POWER11)
                message(STATUS "Compiler supports -mcpu=power11, enabling POWER11 backend")
                ggml_add_cpu_backend_variant(power11 POWER11 VSX)
            else()
                message(STATUS "Skipping POWER11 backend: compiler does not support -mcpu=power11")
            endif()
        else()
            message(FATAL_ERROR "Unsupported PowerPC target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(z15    Z15 VXE2)
            ggml_add_cpu_backend_variant(z16    Z16 VXE2 NNPA)

            # check if compiler supports "-march=z17" codename
            check_cxx_compiler_flag("-march=arch15" GGML_CXX_SUPPORTS_Z17)
            if (GGML_CXX_SUPPORTS_Z17)
                ggml_add_cpu_backend_variant(arch15 Z17 VXE2 NNPA)
            else()
                message(WARNING "Skipping z17 target: compiler must be GCC 15.1 and later")
            endif()
        else()
            message(FATAL_ERROR "Unsupported s390x target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        if (CMAKE_SYSTEM_NAME MATCHES "Linux")
            ggml_add_cpu_backend_variant(riscv64_0)
            ggml_add_cpu_backend_variant(riscv64_v   RVV)
        else()
            message(FATAL_ERROR "Unsupported RISC-V target OS: ${CMAKE_SYSTEM_NAME}")
        endif()
    else()
        message(FATAL_ERROR "GGML_CPU_ALL_VARIANTS not yet supported with ${GGML_SYSTEM_ARCH} on ${CMAKE_SYSTEM_NAME}")
    endif()
elseif (GGML_CPU)
    ggml_add_cpu_backend_variant_impl("")
endif()

# =================================================================================================
# CIRCULAR STATIC-LIB DEPENDENCY: ggml-base <-> ggml-cpu, MEASURED 2026-09-13.
#
# ggml.c (in ggml-base) calls the fork's row-interleaved quantize/vec_dot functions
# (quantize_iq2_k_r4, quantize_mxfp4, vec_dot_q8_KV_q8_KV, dequantize_row_iq4_ks, ... 300+ syms)
# directly out of its static type_traits[] table. Those symbols are DEFINED in
# ggml/src/iqk/iqk_quantize.cpp, which upstream's directory split compiles into the ggml-cpu
# backend target, not ggml-base. ggml-cpu already links PRIVATE ggml-base (the normal direction,
# cpu needs base's tensor/context API) -- so the dependency is genuinely circular at the symbol
# level, not missing or misnamed. nm on the built archives confirms both halves:
#   ggml/src/ggml-cpu.a   : T quantize_mxfp4   (defined)
#   ggml/src/ggml-base.a  : U quantize_mxfp4   (referenced, from ggml.c.obj)
# A single left-to-right archive scan (GNU ld's default) resolves ggml-cpu.a's obligations
# first, then extracts ggml.c.obj out of ggml-base.a *afterward* -- creating new undefined refs
# to symbols in an archive ld has already passed and will not revisit. Every executable that
# pulls in ggml.c.obj for any other reason fails to link (6 of 694 targets, 319 distinct
# undefined symbols) purely from this ordering, not from missing code.
#
# Fix: wrap the whole link line in a GNU ld RESCAN group (-Wl,--start-group / -Wl,--end-group) so
# ld keeps re-scanning archives until no new undefined references appear. This is the standard fix
# for genuinely circular static libraries and changes no behavior -- both objects were already
# being linked, this only lets the linker find them. See the ATTEMPT log below for how this was
# actually reached: CMake's 3.24+ $<LINK_GROUP:RESCAN,...> generator expression (attempt 1) never
# emitted anything on this toolchain; the working mechanism is plain target_link_options +
# target_link_libraries carrying raw -Wl,--start-group/--end-group strings (attempt 4), which
# propagate as INTERFACE usage requirements to every consumer of ggml::ggml transitively.
# =================================================================================================
if (TARGET ggml-cpu)
    # ATTEMPT 1 (MEASURED FAILED): target_link_libraries(ggml PUBLIC
    #   "$<LINK_GROUP:RESCAN,ggml-base,ggml-cpu>") -- 0 occurrences of "start-group" anywhere in
    #   the regenerated 14804-line build.ninja for ANY target. The generator expression produced
    #   no linker flags at all; not pursued further (toolchain LINKER_TYPE opt-in territory, out
    #   of scope for a link-stage close-out).
    # ATTEMPT 2 (MEASURED FAILED): target_link_libraries(ggml PUBLIC -Wl,--start-group ggml-base
    #   ggml-cpu -Wl,--end-group) -- checked the actual LINK_LIBRARIES ninja variable for
    #   bin/test-json-partial.exe byte-for-byte: identical to before this line existed, flags and
    #   all. CMake's link-line closure algorithm recognized ggml-base/ggml-cpu as already-scheduled
    #   targets from the EARLIER calls (line ~409, and ggml_add_backend_library's PUBLIC ${backend}
    #   call) and collapsed this whole redundant mention -- flags included -- rather than emitting
    #   a second occurrence. Raw flag strings are not immune to that collapse when CMake decides
    #   the surrounding targets need no new closure entry.
    # ATTEMPT 3 (MEASURED FAILED): target_link_libraries(ggml-base PRIVATE ggml-cpu), declaring
    #   the reverse graph edge so CMake's own dependency graph has an actual A<->B cycle (ggml-cpu
    #   already links ggml-base PRIVATE the normal direction) and letting CMake's documented
    #   auto-repeat-for-cycles behavior handle it with no manual flags. Reconfigured clean, no
    #   FATAL_ERROR -- but the generated LINK_LIBRARIES for bin/test-json-partial.exe came out
    #   byte-for-byte IDENTICAL to before the edge existed. CMake's modern (non-LINK_INTERFACE_
    #   MULTIPLICITY) closure algorithm did not choose to repeat either archive for this cycle.
    #   Reverting -- an edge that changes nothing measured is not a fix, and it is a real (if inert)
    #   change to the dependency graph other targets could someday trip on.
    #
    # ATTEMPT 4 (this one): stop fighting the LINK_LIBRARIES closure algorithm's target/duplicate
    # handling entirely and use the OTHER variable in the linker command template, which is never
    # deduplicated because it carries plain option strings, not library targets. Read the actual
    # generated rule (CMakeFiles/rules.ninja) for a concrete failing target before assuming a
    # template shape:
    #   command = ... g++.exe $FLAGS $LINK_FLAGS $in -o $TARGET_FILE ... $LINK_PATH $LINK_LIBRARIES
    # $LINK_LIBRARIES is the LAST segment of the command -- nothing follows it -- and it is exactly
    # the closure output ATTEMPTs 2/3 could not get flags to survive in. $LINK_FLAGS (populated by
    # target_link_options / INTERFACE_LINK_OPTIONS, a SEPARATE property from LINK_LIBRARIES with no
    # target-dedup logic) lands EARLY, before $in and $LINK_LIBRARIES both. Opening the ld group in
    # $LINK_FLAGS and closing it with a plain trailing flag appended to $LINK_LIBRARIES (a lone
    # string, not a repeated target name, so it cannot trigger the ATTEMPT-2 collapse) brackets the
    # objects + every archive on the command line -- wider than the one ggml-base/ggml-cpu pair
    # strictly needs, but ld tolerates non-archive tokens inside a group and rescanning a few extra
    # already-resolved archives costs nothing but link time. Both propagate to every consumer of
    # ggml::ggml as usage requirements, so no per-executable CMakeLists needs touching.
    # PLACEMENT BUG in attempts 1-4 (fixed 2026-09-13, second pass): this whole block used to sit
    # directly after `ggml_add_backend(CPU)` (the bare add_subdirectory call), where `ggml-cpu` did
    # NOT exist yet -- the real target is only created later by `ggml_add_cpu_backend_variant_impl`
    # (single-variant path: elseif(GGML_CPU) -> ggml_add_cpu_backend_variant_impl("")).
    # `if (TARGET ggml-cpu)` was silently FALSE the entire time, so attempts 1-4 never executed at
    # all -- confirmed with probe message(STATUS) + probe flags (-Wl,--PROBEFLAG98765 /
    # -Wl,--PROBELIB98765): zero output, zero occurrences anywhere in build.ninja, until this block
    # was moved to directly after the elseif(GGML_CPU) branch's endif(). After the move, the SAME
    # probe flags DID appear (confirmed: INTERFACE_LINK_OPTIONS/INTERFACE_LINK_LIBRARIES populated,
    # 67 occurrences in build.ninja, and for bin/test-json-partial.exe specifically:
    #   LINK_FLAGS     = -Wl,--PROBEFLAG98765
    #   LINK_LIBRARIES = ... ggml-cpu.a  ggml-base.a  ...  -Wl,--PROBELIB98765  -lkernel32 ...
    # -- PROBELIB lands exactly where needed: right after ggml-base.a, before the system libs.
    # Attempt 4's actual mechanism was correct; it had simply never run. Swapping the probe tokens
    # for the real group flags below.
    target_link_options(ggml INTERFACE -Wl,--start-group)
    target_link_libraries(ggml INTERFACE -Wl,--end-group)
endif()

# -------------------------------------------------------------------------------------------------
# GGML_STATIC reaches consumers only as a USAGE REQUIREMENT.
# The add_link_options(-static ...) calls near the top of this file are DIRECTORY-SCOPED: they apply
# to targets created in ggml/src and its subdirectories, and to nothing else. Every executable lives
# in tools/*/ or examples/*/, which are SIBLINGS added from the root CMakeLists -- so those options
# never reached a single binary anyone runs. MEASURED 2026-09-13 on a clean -DGGML_STATIC=ON build:
# GGML_STATIC:BOOL=ON in the cache, ZERO occurrences of "-static" in the generated build.ninja,
# libgomp.dll.a (the IMPORT library) still on the llama-quantize link line, and the resulting exe
# byte-identical to the dynamic one at 41,898,377 bytes.
# Attaching to the ggml target INTERFACE propagates transitively to every consumer of ggml::ggml.
# This is the same mechanism proven to work directly above; see 8a6e8af3a for why the target-scoped
# form is the one that survives CMake's link-line handling here.
if (GGML_STATIC AND NOT MSVC)
    target_link_options(ggml INTERFACE -static)
    if (MINGW)
        target_link_options(ggml INTERFACE -static-libgcc -static-libstdc++)
    endif()
endif()
# -------------------------------------------------------------------------------------------------


ggml_add_backend(BLAS)
ggml_add_backend(CANN)
ggml_add_backend(CUDA)
ggml_add_backend(ET)
ggml_add_backend(HIP)
ggml_add_backend(METAL)
ggml_add_backend(MUSA)
ggml_add_backend(RPC)
ggml_add_backend(VirtGPU)
ggml_add_backend(SYCL)
ggml_add_backend(Vulkan)
ggml_add_backend(WebGPU)
ggml_add_backend(zDNN)
ggml_add_backend(OpenCL)
ggml_add_backend(Hexagon)
ggml_add_backend(ZenDNN)
ggml_add_backend(OPENVINO)

# GATE-CLAIM-OK: build-graph note, not a live-resource claim. Unconditional (no GGML_CUDA guard,
# no CUDA toolkit needed to compile) -- see ggml-cuda-dl/CMakeLists.txt.
# ggml-cuda-dl compiles vendor/hash/sha256 from the llama tree root, so it is skipped when ggml is
# configured on its own (no vendor/ beside it) instead of failing the whole configure.
set(GGML_CUDA_DL_VENDOR_HASH "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/hash")
if (EXISTS "${GGML_CUDA_DL_VENDOR_HASH}/sha256/sha256.c")
    add_subdirectory(ggml-cuda-dl)
else()
    message(STATUS "ggml-cuda-dl: ${GGML_CUDA_DL_VENDOR_HASH} not found, target skipped")
endif()

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> $<INSTALL_INTERFACE:include>)
    target_compile_features   (${target} PRIVATE c_std_11 cxx_std_17) # don't bump
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)

# `if (MATH_LIBRARY)`, not `if (DEFINED MATH_LIBRARY)`. DEFINED is TRUE for a cache entry holding
# MATH_LIBRARY-NOTFOUND, so the link line becomes `-lMATH_LIBRARY-NOTFOUND` and the GENERATE step
# fails for the whole project -- not just this target. That is not hypothetical: this fork used to
# call find_library(MATH_LIBRARY m) here (see 279b4bfaf^1:ggml/src/CMakeLists.txt:1388, which paired
# it with the truthiness test this line now restores). The merge took upstream's consumer without
# upstream's producer, so every build directory configured before the merge still carries the
# NOTFOUND cache entry and dies on reconfigure. Truthiness is also a superset of the DEFINED test:
# it is still true for a caller that deliberately sets MATH_LIBRARY to a real library.
if (MATH_LIBRARY)
    target_link_libraries(ggml-base PRIVATE ${MATH_LIBRARY})
elseif (NOT WIN32 AND NOT DEFINED ENV{ONEAPI_ROOT})
    target_link_libraries(ggml-base PRIVATE m)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(ggml-base PRIVATE dl)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "visionOS")
    target_compile_definitions(ggml-base PUBLIC _DARWIN_C_SOURCE)
endif()

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden
        )
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC  GGML_SHARED)
    endforeach()
endif()
