function(ggml_get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang" OR CCID MATCHES "IntelLLVM")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (CCID STREQUAL "IntelLLVM")
            # for Intel DPC++ and LLVM SYCL, using debug does not disable optimization
            list(APPEND C_FLAGS    -Rno-debug-disables-optimization)
            list(APPEND CXX_FLAGS  -Rno-debug-disables-optimization)
        endif()

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

function(ggml_get_system_arch)
    if (CMAKE_OSX_ARCHITECTURES      STREQUAL "arm64" OR
        CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
        (CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64ec" AND MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang") OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
        set(GGML_SYSTEM_ARCH "ARM" PARENT_SCOPE)
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
            CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
            (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64|amd64)$"))
        set(GGML_SYSTEM_ARCH "x86" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc|power")
        set(GGML_SYSTEM_ARCH "PowerPC" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        set(GGML_SYSTEM_ARCH "loongarch64"  PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
        set(GGML_SYSTEM_ARCH "riscv64" PARENT_SCOPE)
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
        set(GGML_SYSTEM_ARCH "s390x" PARENT_SCOPE)
    else()
        set(GGML_SYSTEM_ARCH "UNKNOWN" PARENT_SCOPE)
    endif()
endfunction()

# Determines which FlashAttention vector kernel template instances to compile, returns them in OUT_SRCS.
function(ggml_cuda_fattn_vec_instances DIR OUT_SRCS)
    set(FA_TYPES q4_0 q4_1 q5_0 q5_1 q8_0 bf16 f16)

    string(TOLOWER "${GGML_CUDA_FA_QUANTS}" FA_QUANTS)
    string(STRIP   "${FA_QUANTS}" FA_QUANTS)
    if (GGML_CUDA_FA_ALL_QUANTS)
        message(WARNING "GGML_CUDA_FA_ALL_QUANTS is deprecated, use GGML_CUDA_FA_QUANTS=all instead")
        set(FA_QUANTS all)
    endif()
    if (NOT FA_QUANTS)
        message(FATAL_ERROR "GGML_CUDA_FA_QUANTS must not be empty")
    endif()

    if (FA_QUANTS STREQUAL "all")
        set(FA_COMBINATIONS "")
        foreach (TYPE_V IN LISTS FA_TYPES)
            foreach (TYPE_K IN LISTS FA_TYPES)
                list(APPEND FA_COMBINATIONS ${TYPE_K}-${TYPE_V})
            endforeach()
        endforeach()
    else()
        set(FA_COMBINATIONS f16-f16)

        string(REPLACE "," ";" FA_SELECTED "${FA_QUANTS}")
        foreach (COMBINATION IN LISTS FA_SELECTED)
            string(STRIP   "${COMBINATION}" COMBINATION)
            if (NOT COMBINATION MATCHES "^([a-z0-9_]+)-([a-z0-9_]+)$")
                message(FATAL_ERROR "GGML_CUDA_FA_QUANTS: \"${COMBINATION}\" is not \"all\" or a <type_K>-<type_V> combination")
            endif()
            set(TYPE_K ${CMAKE_MATCH_1})
            set(TYPE_V ${CMAKE_MATCH_2})
            foreach (TYPE ${TYPE_K} ${TYPE_V})
                if (NOT TYPE IN_LIST FA_TYPES)
                    message(FATAL_ERROR
                        "GGML_CUDA_FA_QUANTS: unknown type \"${TYPE}\" in \"${COMBINATION}\", must be one of: ${FA_TYPES}")
                endif()
            endforeach()
            list(APPEND FA_COMBINATIONS ${TYPE_K}-${TYPE_V})
        endforeach()
    endif()
    list(REMOVE_DUPLICATES FA_COMBINATIONS)

    string(REPLACE ";" "," FA_QUANTS_DEFINE "${FA_QUANTS}")
    add_compile_definitions(GGML_CUDA_FA_QUANTS="${FA_QUANTS_DEFINE}")
    foreach (TYPE_V IN LISTS FA_TYPES)
        foreach (TYPE_K IN LISTS FA_TYPES)
            if ("${TYPE_K}-${TYPE_V}" IN_LIST FA_COMBINATIONS)
                set(COMPILED 1)
            else()
                set(COMPILED 0)
            endif()
            string(TOUPPER "GGML_CUDA_FA_${TYPE_K}_${TYPE_V}" COMBINATION_DEF)
            add_compile_definitions(${COMBINATION_DEF}=${COMPILED})
        endforeach()
    endforeach()

    message(STATUS "FlashAttention K-V type combinations: ${FA_COMBINATIONS}")

    set(SRCS "")
    foreach (COMBINATION IN LISTS FA_COMBINATIONS)
        set(SRC "${DIR}/template-instances/fattn-vec-instance-${COMBINATION}.cu")
        if (NOT EXISTS "${SRC}")
            message(FATAL_ERROR "FlashAttention template instance \"${SRC}\" does not exist")
        endif()
        list(APPEND SRCS "${SRC}")
    endforeach()

    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
endfunction()

# RESEARCH-DONE: upstream ggml-org/llama.cpp master 80d2c0c2c56e ggml/cmake/common.cmake ggml_cuda_fattn_vec_instances (read in full, copy in temp/faq_src_1107).
# ---------------------------------------------------------------------------------------------------------------------------
# FORK form of ggml_cuda_fattn_vec_instances above (that one is upstream's, kept verbatim and still uncalled: it names
# fattn-vec-instance-<K>-<V>.cu, which this fork does not compile -- ggml-cuda/CMakeLists.txt removes them -- and its type set has bf16
# but no iq4_nl / q6_0). This one has upstream's SPELLING AND SEMANTICS for GGML_CUDA_FA_QUANTS:
#     "all"  or  a "," / ";" separated list of <type_K>-<type_V>;  f16-f16 is always built;
#     GGML_CUDA_FA_ALL_QUANTS=ON is the deprecated alias of "all" (with a warning);
#     an unknown type, or a pair this fork has no case for, is a configure FATAL_ERROR.
# over the FORK's type set. The set of valid pairs is not typed here: it is PARSED from ggml-cuda/fattn-vec-pairs.def, the same file the
# dispatch and the supports_op predicate are expanded from (fattn-vec-pairs.cuh), so the three cannot drift. For every pair in that file
# this defines GGML_CUDA_FA_<K>_<V>=0|1 (and GGML_CUDA_FA_PAIRS_FROM_CMAKE), and for every selected pair it requires and lists the
# fork-named instance files  fattn-vec-f{16,32}-instance-hs<Dk>-<K>-<V>.cu  (f32 only for the FA_VEC_PAIR_F32 cases).
# bf16: the vec kernels carry BF16 K and V (vec_dot_fattn_vec_KQ_bf16 / dequantize_1_bf16 in fattn-vec-common.cuh), so bf16-bf16 and the
# other bf16 pairs of fattn-vec-pairs.def are ordinary selectable pairs (S69-FORK-CUDA-FA-VEC-BF16-KV-KERNELS-2026-11-08-01). The FORK default
# list does not include bf16 (unchanged); upstream's exact default string (which ends in bf16-bf16) is still read as "default" when a stale cache
# holds it (see below), so to build bf16-bf16 name it in any other list, e.g. "q8_0-q8_0;bf16-bf16", or use "all". TurboQuant pairs are always
# built (ggml-cuda/CMakeLists.txt), so naming one is accepted and ignored.
function(ggml_cuda_fattn_vec_fork_instances DIR OUT_SRCS)
    set(DEF "${DIR}/fattn-vec-pairs.def")
    file(STRINGS "${DEF}" DEF_LINES REGEX "^FA_VEC_PAIR(_F32)?\\(")
    set(ENTRIES "")      # "<Dk>:<Dv>:<k>:<v>:<f32only 0|1>"
    set(ALL_PAIRS "")
    set(KNOWN_TYPES "")
    foreach (LINE IN LISTS DEF_LINES)
        if (NOT LINE MATCHES "^FA_VEC_PAIR(_F32)?\\([ \t]*([0-9]+)[ \t]*,[ \t]*([0-9]+)[ \t]*,[ \t]*([A-Z0-9_]+)[ \t]*,[ \t]*([A-Z0-9_]+)[ \t]*\\)")
            message(FATAL_ERROR "${DEF}: cannot parse \"${LINE}\" (expected FA_VEC_PAIR(Dk, Dv, K, V) on one line)")
        endif()
        set(M_F32 "${CMAKE_MATCH_1}")
        set(M_DK  "${CMAKE_MATCH_2}")
        set(M_DV  "${CMAKE_MATCH_3}")
        string(TOLOWER "${CMAKE_MATCH_4}" TK)
        string(TOLOWER "${CMAKE_MATCH_5}" TV)
        if (M_F32)
            set(F32ONLY 1)
        else()
            set(F32ONLY 0)
        endif()
        list(APPEND ENTRIES "${M_DK}:${M_DV}:${TK}:${TV}:${F32ONLY}")
        list(APPEND ALL_PAIRS "${TK}-${TV}")
        list(APPEND KNOWN_TYPES ${TK} ${TV})
    endforeach()
    list(REMOVE_DUPLICATES ALL_PAIRS)
    list(REMOVE_DUPLICATES KNOWN_TYPES)
    list(LENGTH ENTRIES N_ENTRIES)
    if (N_ENTRIES EQUAL 0)
        message(FATAL_ERROR "${DEF}: no FA_VEC_PAIR entries found")
    endif()

    string(TOLOWER "${GGML_CUDA_FA_QUANTS}" FA_QUANTS)
    string(STRIP   "${FA_QUANTS}" FA_QUANTS)
    # A build dir configured before this option existed holds UPSTREAM's default in its cache (CACHE STRING never overwrites an
    # existing entry). Taken literally it would keep only q4_0/q8_0/f16/bf16 and silently drop the fork's default
    # iq4_nl/q6_0 pairs -- measured in gate b1107g2b's build dir 2026-11-08. That exact string means "default": use the fork's.
    if (FA_QUANTS MATCHES "^q4_0-q4_0[;,]q8_0-q8_0[;,]f16-f16[;,]bf16-bf16$" AND DEFINED GGML_CUDA_FA_QUANTS_FORK_DEFAULT)
        message(STATUS "GGML_CUDA_FA_QUANTS holds upstream's default (\"${FA_QUANTS}\"), read as the fork default: ${GGML_CUDA_FA_QUANTS_FORK_DEFAULT}")
        string(TOLOWER "${GGML_CUDA_FA_QUANTS_FORK_DEFAULT}" FA_QUANTS)
    endif()
    if (GGML_CUDA_FA_ALL_QUANTS)
        message(WARNING "GGML_CUDA_FA_ALL_QUANTS is deprecated, use GGML_CUDA_FA_QUANTS=all instead")
        set(FA_QUANTS all)
    endif()
    if (NOT FA_QUANTS)
        message(FATAL_ERROR "GGML_CUDA_FA_QUANTS must not be empty")
    endif()

    if (FA_QUANTS STREQUAL "all")
        set(FA_SELECTED_PAIRS ${ALL_PAIRS})
    else()
        set(FA_SELECTED_PAIRS f16-f16)
        string(REPLACE "," ";" FA_REQUESTED "${FA_QUANTS}")
        foreach (COMBINATION IN LISTS FA_REQUESTED)
            string(STRIP "${COMBINATION}" COMBINATION)
            if (NOT COMBINATION MATCHES "^([a-z0-9_]+)-([a-z0-9_]+)$")
                message(FATAL_ERROR "GGML_CUDA_FA_QUANTS: \"${COMBINATION}\" is not \"all\" or a <type_K>-<type_V> combination")
            endif()
            set(TYPE_K ${CMAKE_MATCH_1})
            set(TYPE_V ${CMAKE_MATCH_2})
            set(SKIP 0)
            foreach (TYPE ${TYPE_K} ${TYPE_V})
                if (TYPE MATCHES "^turbo[0-9]_0$")
                    message(STATUS "GGML_CUDA_FA_QUANTS: \"${COMBINATION}\" ignored, TurboQuant combinations are always built")
                    set(SKIP 1)
                elseif (NOT TYPE IN_LIST KNOWN_TYPES)
                    message(FATAL_ERROR
                        "GGML_CUDA_FA_QUANTS: unknown type \"${TYPE}\" in \"${COMBINATION}\", must be one of: ${KNOWN_TYPES}")
                endif()
            endforeach()
            if (SKIP)
                continue()
            endif()
            if (NOT "${TYPE_K}-${TYPE_V}" IN_LIST ALL_PAIRS)
                message(FATAL_ERROR
                    "GGML_CUDA_FA_QUANTS: \"${COMBINATION}\" has no flash-attention vec case in this fork "
                    "(valid pairs are those listed in ${DEF}): ${ALL_PAIRS}")
            endif()
            list(APPEND FA_SELECTED_PAIRS ${TYPE_K}-${TYPE_V})
        endforeach()
    endif()
    list(REMOVE_DUPLICATES FA_SELECTED_PAIRS)

    # The selection string (reported by the CUDA backend's feature list as FA_QUANTS) goes into a GENERATED HEADER, not a
    # -D"..." define: a quoted string define reaches nvcc as -DGGML_CUDA_FA_QUANTS=\"...\" and sccache rejects that command line
    # ("sccache: error: Could not parse shell line", every .cu, gate b1107g2b 2026-11-08). configure_file rewrites the header only
    # when its content changes, so an unchanged selection does not rebuild ggml-cuda.cu.
    string(REPLACE ";" "," FA_QUANTS_DEFINE "${FA_QUANTS}")
    set(FA_HDR_IN "${CMAKE_CURRENT_BINARY_DIR}/ggml-cuda-fa-quants.h.in")
    file(WRITE "${FA_HDR_IN}" "// generated by ggml_cuda_fattn_vec_fork_instances (ggml/cmake/common.cmake) -- do not edit\n#pragma once\n#define GGML_CUDA_FA_QUANTS \"${FA_QUANTS_DEFINE}\"\n")
    configure_file("${FA_HDR_IN}" "${CMAKE_CURRENT_BINARY_DIR}/ggml-cuda-fa-quants.h" COPYONLY)
    add_compile_definitions(GGML_CUDA_FA_PAIRS_FROM_CMAKE=1)
    foreach (PAIR IN LISTS ALL_PAIRS)
        if (PAIR IN_LIST FA_SELECTED_PAIRS)
            set(COMPILED 1)
        else()
            set(COMPILED 0)
        endif()
        string(TOUPPER "GGML_CUDA_FA_${PAIR}" PAIR_DEF)
        string(REPLACE "-" "_" PAIR_DEF "${PAIR_DEF}")
        add_compile_definitions(${PAIR_DEF}=${COMPILED})
    endforeach()
    message(STATUS "FlashAttention K-V type combinations: ${FA_SELECTED_PAIRS}")

    set(SRCS "")
    foreach (ENTRY IN LISTS ENTRIES)
        string(REPLACE ":" ";" PARTS "${ENTRY}")
        list(GET PARTS 0 DK)
        list(GET PARTS 2 TK)
        list(GET PARTS 3 TV)
        list(GET PARTS 4 F32ONLY)
        if (NOT "${TK}-${TV}" IN_LIST FA_SELECTED_PAIRS)
            continue()
        endif()
        set(KERNELS f32)
        if (NOT F32ONLY)
            set(KERNELS f16 f32)
        endif()
        foreach (KERNEL IN LISTS KERNELS)
            set(SRC "${DIR}/template-instances/fattn-vec-${KERNEL}-instance-hs${DK}-${TK}-${TV}.cu")
            if (NOT EXISTS "${SRC}")
                message(FATAL_ERROR "FlashAttention template instance \"${SRC}\" does not exist")
            endif()
            list(APPEND SRCS "${SRC}")
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES SRCS)

    set(${OUT_SRCS} ${SRCS} PARENT_SCOPE)
endfunction()
