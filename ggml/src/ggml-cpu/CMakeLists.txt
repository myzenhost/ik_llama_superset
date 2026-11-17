function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Disable LTO for the feature detection code to prevent cross-module optimization
    # from inlining architecture-specific instructions into the score function.
    # Without this, LTO can cause SIGILL when loading backends on older CPUs
    # (e.g., loading power10 backend on power9 crashes before feature check runs).
    target_compile_options(${GGML_CPU_FEATS_NAME} PRIVATE -fno-lto)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/amx/amx.cpp
        ggml-cpu/amx/amx.h
        ggml-cpu/amx/mmq.cpp
        ggml-cpu/amx/mmq.h
        ggml-cpu/tiled/tiled.cpp
        ggml-cpu/tiled/tiled.h
        ggml-cpu/tiled/tiled-kernel.cpp
        ggml-cpu/tiled/tiled-kernel.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    # ELF only: libggml-base now exports a forwarding stub under the name of each function this library defines
    # (GGML_BASE_CPU_SYMS). Without -Bsymbolic-functions this library's own calls, and the &name it hands
    # ggml-base, would resolve through the global scope to the base stub, which forwards back here: unbounded
    # recursion. Binding references to this library's own definitions keeps both on the real function.
    # (Mach-O two-level namespaces and PE import tables bind to the defining image already.)
    if ((GGML_BACKEND_DL OR BUILD_SHARED_LIBS) AND UNIX AND NOT APPLE)
        target_link_options(${GGML_CPU_NAME} PRIVATE -Wl,-Bsymbolic-functions)
    endif()

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${GGML_OPENMP_TARGET_C} ${GGML_OPENMP_TARGET_CXX})
    endif()

    if (GGML_LLAMAFILE)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_LLAMAFILE)

        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/llamafile/sgemm.cpp
                    ggml-cpu/llamafile/sgemm.h)
    endif()

    # ---------------------------------------------------------------------------------------
    # ik_llama's IQK kernels -- this fork's SOTA quantized GEMM and its CPU FlashAttention.
    #
    # WHY THEY ARE HERE. This fork historically built ggml from ONE monolithic
    # ggml/src/CMakeLists.txt which named these sources directly. Upstream's layout splits the
    # build into per-backend libraries entered from a dispatcher, and its ggml-cpu names 0 of
    # them and has ZERO file(GLOB) forms -- so adopting the dispatcher without this block would
    # drop ~2.13 MB of CPU kernels and STILL BUILD CLEAN. Measured: a coverage diff of the
    # monolith against the dispatcher plus all 19 per-backend files found exactly 29 sources
    # uncovered, 21 of them these.
    #
    # This is the hook upstream itself uses for optional CPU kernels -- llamafile/sgemm.cpp
    # directly above, the per-arch quants further down, and KleidiAI near the end all extend
    # GGML_CPU_SOURCES the same way. The fork's kernels are exactly that shape.
    #
    # THE COMPILE DEFINITIONS ARE DELIBERATELY NOT HERE. The monolith sets GGML_USE_IQK_MULMAT
    # with add_compile_definitions(), which is DIRECTORY-scoped and therefore reaches ggml.c,
    # ggml-backend.cpp and ggml-quants.c -- all three read that macro and none of them belongs to
    # the CPU backend. A target_compile_definitions(${GGML_CPU_NAME} PRIVATE ...) here would
    # reach none of them and would silently switch the fork's IQK paths off in the core while
    # building clean. The definitions stay in the parent ggml/src/CMakeLists.txt, where their
    # scope matches what the sources expect.

    # Always built. iqk_quantize.cpp READS GGML_USE_IQK_MULMAT rather than being gated by it, so
    # the file compiles either way and its content varies with the macro -- monolith line 178,
    # above and outside the GGML_IQK_MUL_MAT guard.
    #
    # IT MUST STAY ON THIS TARGET, and the reason is not filing convenience -- MEASURED 2026-09-15.
    # ggml.c's type_traits[] table (compiled into ggml-base) names 262 symbols defined in this
    # file, which is why bin/ggml-base.dll fails with LNK1120 / 270 unresolved. Moving the file to
    # ggml-base looks like the one-line fix and was TRIED: it compiles there only until
    # iqk_quantize.cpp(7268), where the generic non-AVX2/non-NEON branch of repack_q8_KV refers to
    # an identifier `y` that does not exist in its scope. That branch is marked "// TODO" and has
    # evidently never been compiled by anything. ggml-base is built arch-neutral on purpose (no
    # /arch:AVX2, no GGML_AVX2/FMA/F16C/BMI2) so ggml_add_cpu_backend_variant() can ship per-arch
    # CPU builds -- so over there this file gets the broken branch, and the only ways out would be
    # to bake AVX2 into the arch-neutral layer or to lose q8_KV_r8 repacking outright. Neither is
    # acceptable, so the partition is fixed from the OTHER end: see
    # S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01.
    list(APPEND GGML_CPU_SOURCES
                iqk/iqk_quantize.cpp
                iqk/iqk_cpu_ops.cpp)

    # GGML_BACKEND_DL: this variant hands ggml-base its definitions of the functions ggml-base names but cannot
    # link (ggml-cpu-syms.h). Called from ggml_backend_cpu_reg; not compiled for a static build.
    # Also for a shared (non-DL) build, where libggml-base carries the same stubs (GGML_BASE_CPU_SYMS).
    if (GGML_BACKEND_DL OR BUILD_SHARED_LIBS)
        list(APPEND GGML_CPU_SOURCES
                    ggml-cpu/cpu-syms-bind.cpp
                    ggml-cpu/cpu-syms-bind.h)
    endif()

    if (GGML_IQK_MUL_MAT)
        # The FlashAttention translation units are members of this list, NOT of a separate
        # GGML_IQK_FLASH_ATTENTION guard -- monolith lines 183-195. Each opens with
        #     #if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION
        # so with FA off they compile to nothing. Gating the SOURCES on FA instead would admit
        # FA=ON with MUL_MAT=OFF, which links FA against MM symbols that were never built.
        list(APPEND GGML_CPU_SOURCES
                    iqk/iqk_mul_mat.cpp
                    iqk/iqk_kda.cpp
                    iqk/iqk_gemm_1bit.cpp
                    iqk/iqk_gemm_floats.cpp
                    iqk/iqk_gemm_iqk_quants.cpp
                    iqk/iqk_gemm_iquants.cpp
                    iqk/iqk_gemm_kquants.cpp
                    iqk/iqk_gemm_ktquants.cpp
                    iqk/iqk_gemm_legacy_quants.cpp
                    iqk/iqk_flash_attn.cpp
                    iqk/fa/iqk_fa_64_64.cpp
                    iqk/fa/iqk_fa_96_96.cpp
                    iqk/fa/iqk_fa_128_128.cpp
                    iqk/fa/iqk_fa_192_128.cpp
                    iqk/fa/iqk_fa_192_192.cpp
                    iqk/fa/iqk_fa_256_256.cpp
                    iqk/fa/iqk_fa_320_256.cpp
                    iqk/fa/iqk_fa_512_512.cpp
                    iqk/fa/iqk_fa_576_512.cpp)
    endif()

    # The IQK sources include their headers as "iqk/..." and "ggml.h", both rooted at ggml/src.
    target_include_directories(${GGML_CPU_NAME} PRIVATE .. ../iqk)
    # ---------------------------------------------------------------------------------------

    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
            )

        # MSVC proper (not clang-cl) has no -mcpu/-march for ARM: features can only be
        # selected by probing the host at configure time and forwarding -D__ARM_FEATURE_*
        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            set(GGML_ARM_MSVC ON)
        else()
            set(GGML_ARM_MSVC OFF)
        endif()

        if (GGML_ARM_MSVC AND NOT GGML_NATIVE)
            message(FATAL_ERROR "MSVC on ARM requires GGML_NATIVE=ON, use clang for GGML_CPU_ARM_ARCH / GGML_CPU_ALL_VARIANTS")
        else()
            if (NOT GGML_ARM_MSVC)
                check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
                if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                    list(APPEND ARCH_FLAGS -mfp16-format=ieee)
                endif()
            endif()

            if (CMAKE_C_COMPILER_ID MATCHES "GNU")
                # GCC rejects the implicit int8x16_t <-> uint8x16_t conversions that the iqk NEON
                # kernels rely on (clang accepts them by default). Upstream ik_llama.cpp passes this
                # flag for GNU on ARM (ggml/src/CMakeLists.txt, "else we fail on Gravitons and such");
                # the ggml-cpu split dropped it, which broke every aarch64 GCC build.
                list(APPEND ARCH_FLAGS -flax-vector-conversions)
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                # cl.exe cannot be queried this way: no /dev/null, and it rejects the flags.
                # ARM_NATIVE_FLAG stays empty for MSVC and is never appended below.
                if (NOT GGML_ARM_MSVC)
                    execute_process(
                        COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                        INPUT_FILE "/dev/null"
                        OUTPUT_QUIET
                        ERROR_VARIABLE ARM_MCPU
                        RESULT_VARIABLE ARM_MCPU_RESULT
                    )
                    if (NOT ARM_MCPU_RESULT)
                        string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                        string(REGEX MATCH "-march=[^ ']+" ARM_MARCH_FLAG "${ARM_MCPU}")

                        # on some old GCC we need to read -march=
                        if (ARM_MARCH_FLAG AND NOT "${ARM_MARCH_FLAG}" STREQUAL "-march=native")
                            set(ARM_NATIVE_FLAG "${ARM_MARCH_FLAG}")
                        elseif(ARM_MCPU_FLAG AND NOT "${ARM_MCPU_FLAG}" STREQUAL "-mcpu=native")
                            set(ARM_NATIVE_FLAG "${ARM_MCPU_FLAG}")
                        endif()
                    endif()

                    if ("${ARM_NATIVE_FLAG}" STREQUAL "")
                        set(ARM_NATIVE_FLAG -mcpu=native)
                        message(WARNING "ARM -march/-mcpu not found, -mcpu=native will be used")
                    else()
                        message(STATUS "ARM detected flags: ${ARM_NATIVE_FLAG}")
                    endif()
                endif()

                include(CheckCXXSourceRuns)

                macro(check_arm_feature tag feature code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    if (GGML_ARM_MSVC)
                        set(ARM_PROBE_FLAG   "")
                        set(ARM_PROBE_NOFLAG "")
                    else()
                        set(ARM_PROBE_FLAG   "${ARM_NATIVE_FLAG}+${tag}")
                        set(ARM_PROBE_NOFLAG "${ARM_NATIVE_FLAG}+no${tag}")
                    endif()
                    set(CMAKE_REQUIRED_FLAGS "${ARM_PROBE_FLAG}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+${tag}")
                        if (GGML_ARM_MSVC)
                            # forward runtime probing decisions for MSVC
                            list(APPEND ARCH_FLAGS "-D__ARM_FEATURE_${feature}=1")
                        endif()
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_PROBE_NOFLAG}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+no${tag}")
                            list(APPEND ARCH_FLAGS -U__ARM_FEATURE_${feature})
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endmacro()

                if (GGML_ARM_MSVC)
                    # drop volatile
                    check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a = vdupq_n_s8(0), _b = vdupq_n_s8(0); int32x4_t _s = vdupq_n_s32(0); _s = vdotq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a = vdupq_n_s8(0), _b = vdupq_n_s8(0); int32x4_t _s = vdupq_n_s32(0); _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main() { const svbool_t pg = svptrue_b8(); const svint8_t a = svdup_n_s8(0), b = svdup_n_s8(0); svint32_t s = svdup_n_s32(0); s = svdot_s32(s, a, b); (void) pg; (void) s; return 0; }")
                    # FMA is mandatory on ARMv8-A but MSVC never defines __ARM_FEATURE_FMA,
                    # so probe it like the others and forward the define
                    check_arm_feature(fma     FMA         "#include <arm_neon.h>\nint main() { float32x4_t _a = vdupq_n_f32(0), _b = vdupq_n_f32(0), _c = vdupq_n_f32(0); _a = vfmaq_f32(_a, _b, _c); return 0; }")
                else()
                    check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                    check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main()  { svfloat32_t _a, _b; volatile svfloat32_t _c = svadd_f32_z(svptrue_b8(), _a, _b); return 0; }")
                endif()
                check_arm_feature(sme     SME         "#include <arm_sme.h>\n__arm_locally_streaming int main() { __asm__ volatile(\"smstart; smstop;\"); return 0; }")

                if (NOT GGML_ARM_MSVC)
                    list(APPEND ARCH_FLAGS "${ARM_NATIVE_FLAG}${ARM_NATIVE_FLAG_FIX}")
                endif()
            else()
                if (GGML_CPU_ARM_ARCH)
                    list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
                elseif(GGML_CPU_ALL_VARIANTS)
                    # Begin with the lowest baseline
                    set(ARM_MCPU "armv8-a")
                    set(ARCH_TAGS "")
                    set(ARCH_DEFINITIONS "")

                    # When a feature is selected, bump the MCPU to the first
                    # version that supported it
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
                    if (GGML_INTERNAL_SVE2)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
                    endif()
                    if (GGML_INTERNAL_NOSVE)
                        set(ARCH_TAGS "${ARCH_TAGS}+nosve")
                    endif()
                    if (GGML_INTERNAL_SME)
                        set(ARM_MCPU "armv9.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sme")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
                    endif()
                    list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
                    ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
                endif()
            endif()

            message(STATUS "Checking for ARM features using flags:")
            foreach(flag IN LISTS ARCH_FLAGS)
                message(STATUS "  ${flag}")
            endforeach()

            include(CheckCXXSourceCompiles)
            set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
            string(REPLACE ";" " " ARCH_FLAGS_STR "${ARCH_FLAGS}")
            set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS_STR}")
            foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                set(ARM_FEATURE "HAVE_${feature}")
                check_cxx_source_compiles(
                    "
                    #if !defined(__ARM_FEATURE_${feature})
                    #  error \"Feature ${feature} is not defined\"
                    #endif
                    int main() { return 0; }
                    "
                    ${ARM_FEATURE}
                )
            endforeach()
            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "x86")
        message(STATUS "x86 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )

        if (MSVC)
            # instruction set detection for MSVC only
            if (GGML_NATIVE)
                include(ggml-cpu/cmake/FindSIMD.cmake)
            endif ()
            if (GGML_AVX512)
                list(APPEND ARCH_FLAGS /arch:AVX512)
                # /arch:AVX512 includes: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, and __AVX512VL__
                # MSVC has no compile-time flags enabling specific
                # AVX512 extensions, neither it defines the
                # macros corresponding to the extensions.
                # Do it manually.
                list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VBMI__)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vbmi)
                    endif()
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_DEFINITIONS __AVX512VNNI__ GGML_AVX512_VNNI)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512vnni)
                    endif()
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_DEFINITIONS __AVX512BF16__ GGML_AVX512_BF16)
                    if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                        list(APPEND ARCH_FLAGS -mavx512bf16)
                    endif()
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_DEFINITIONS __AMX_TILE__ GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_DEFINITIONS __AMX_INT8__ GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_DEFINITIONS __AMX_BF16__ GGML_AMX_BF16)
                endif()

                # ik_llama #2429 + #2430 (its ggml/src/CMakeLists.txt and ggml/cmake/FindSIMD.cmake; this tree
                # keeps the MSVC block here and FindSIMD in ggml-cpu/cmake/). With GGML_NATIVE the extensions are
                # probed by FindSIMD.cmake, so reaching here means the CPU has no AVX512-VNNI (Skylake-X, Cascade
                # Lake). Leaving HAVE_FANCY_SIMD off is correct there. NOT run: MSVC is not configured here.
                if (GGML_NATIVE AND NOT GGML_AVX512_VNNI)
                    message(STATUS
                        "AVX-512 without VNNI detected. __AVX512VNNI__ stays undefined, "
                        "HAVE_FANCY_SIMD is off, and the IQK GEMM kernels for Zen4 / Sapphire "
                        "Rapids and later are not compiled, which is expected on this CPU.")
                endif()
            elseif (GGML_AVX2)
                list(APPEND ARCH_FLAGS /arch:AVX2)
                list(APPEND ARCH_DEFINITIONS GGML_AVX2 GGML_FMA GGML_F16C)
            elseif (GGML_AVX)
                list(APPEND ARCH_FLAGS /arch:AVX)
                list(APPEND ARCH_DEFINITIONS GGML_AVX)
            elseif (GGML_SSE42)
                list(APPEND ARCH_FLAGS /arch:SSE4.2)
                list(APPEND ARCH_DEFINITIONS GGML_SSE42)
            endif()
            if (GGML_AVX_VNNI)
                list(APPEND ARCH_DEFINITIONS __AVXVNNI__ GGML_AVX_VNNI)
            endif()
            if (GGML_BMI2)
                # MSVC does not define macro __BMI2__
                list(APPEND ARCH_DEFINITIONS __BMI2__ GGML_BMI2)
            endif()
        else ()
            if (GGML_NATIVE)
                list(APPEND ARCH_FLAGS -march=native)
            else ()
                if (GGML_SSE42)
                    list(APPEND ARCH_FLAGS -msse4.2)
                    list(APPEND ARCH_DEFINITIONS GGML_SSE42)
                endif()
                if (GGML_F16C)
                    list(APPEND ARCH_FLAGS -mf16c)
                    list(APPEND ARCH_DEFINITIONS GGML_F16C)
                endif()
                if (GGML_FMA)
                    list(APPEND ARCH_FLAGS -mfma)
                    list(APPEND ARCH_DEFINITIONS GGML_FMA)
                endif()
                if (GGML_BMI2)
                    list(APPEND ARCH_FLAGS -mbmi2)
                    list(APPEND ARCH_DEFINITIONS GGML_BMI2)
                endif()
                if (GGML_AVX)
                    list(APPEND ARCH_FLAGS -mavx)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX)
                endif()
                if (GGML_AVX2)
                    list(APPEND ARCH_FLAGS -mavx2)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX2)
                endif()
                if (GGML_AVX_VNNI)
                    list(APPEND ARCH_FLAGS -mavxvnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX_VNNI)
                endif()
                if (GGML_AVX512)
                    list(APPEND ARCH_FLAGS -mavx512f)
                    list(APPEND ARCH_FLAGS -mavx512cd)
                    list(APPEND ARCH_FLAGS -mavx512vl)
                    list(APPEND ARCH_FLAGS -mavx512dq)
                    list(APPEND ARCH_FLAGS -mavx512bw)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512)
                endif()
                if (GGML_AVX512_VBMI)
                    list(APPEND ARCH_FLAGS -mavx512vbmi)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VBMI)
                endif()
                if (GGML_AVX512_VNNI)
                    list(APPEND ARCH_FLAGS -mavx512vnni)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_VNNI)
                endif()
                if (GGML_AVX512_BF16)
                    list(APPEND ARCH_FLAGS -mavx512bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AVX512_BF16)
                endif()
                if (GGML_AMX_TILE)
                    list(APPEND ARCH_FLAGS -mamx-tile)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_TILE)
                endif()
                if (GGML_AMX_INT8)
                    list(APPEND ARCH_FLAGS -mamx-int8)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_INT8)
                endif()
                if (GGML_AMX_BF16)
                    list(APPEND ARCH_FLAGS -mamx-bf16)
                    list(APPEND ARCH_DEFINITIONS GGML_AMX_BF16)
                endif()
            endif()
        endif()

        if (GGML_BACKEND_DL)
            if (GGML_NATIVE)
                # the feature check relies on ARCH_DEFINITIONS, but it is not set with GGML_NATIVE
                message(FATAL_ERROR "GGML_NATIVE is not compatible with GGML_BACKEND_DL, consider using GGML_CPU_ALL_VARIANTS")
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} x86 ${ARCH_DEFINITIONS})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "PowerPC")
        message(STATUS "PowerPC detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/powerpc/quants.c)
        if (GGML_NATIVE)
            if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
                file(READ "/proc/cpuinfo" POWER10_M)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "powerpc")
                execute_process(COMMAND bash -c "prtconf |grep 'Implementation' | head -n 1" OUTPUT_VARIABLE POWER10_M)
            endif()

            string(TOUPPER "${POWER10_M}" POWER10_M_UPPER)
            string(REGEX MATCHALL "POWER *([0-9]+)" MATCHED_STRING "${POWER10_M_UPPER}")
            string(REGEX REPLACE "POWER *([0-9]+)" "\\1" EXTRACTED_NUMBER "${MATCHED_STRING}")

            if (EXTRACTED_NUMBER EQUAL 10 OR EXTRACTED_NUMBER EQUAL 11)
                list(APPEND ARCH_FLAGS -mcpu=power10)
            elseif (EXTRACTED_NUMBER EQUAL 9)
                list(APPEND ARCH_FLAGS -mcpu=power9)
            elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
                list(APPEND ARCH_FLAGS -mcpu=powerpc64le -mtune=native)
            else()
                list(APPEND ARCH_FLAGS -mcpu=native -mtune=native -mpowerpc64)
            endif()
        elseif(GGML_CPU_ALL_VARIANTS)
            # Begin with the lowest baseline
            set(ARCH_DEFINITIONS "")

            # When a feature is selected, bump the MCPU to the first
            # version that supported it
            foreach(PVER RANGE 7 11)
                if(DEFINED GGML_INTERNAL_POWER${PVER})
                    set(POWERPC_MCPU "power${PVER}")
                    list(APPEND ARCH_DEFINITIONS GGML_USE_POWER${PVER})
                endif()
            endforeach()
            if (GGML_INTERNAL_VSX)
                list(APPEND ARCH_DEFINITIONS GGML_USE_VSX)
                list(APPEND ARCH_FLAGS -mvsx)
            endif()

            if (DEFINED POWERPC_MCPU)
                list(APPEND ARCH_FLAGS -mcpu=${POWERPC_MCPU})
            endif()
            ggml_add_cpu_backend_features(${GGML_CPU_NAME} powerpc ${ARCH_DEFINITIONS})
        else()
            if (GGML_CPU_POWERPC_CPUTYPE)
                list(APPEND ARCH_FLAGS -mcpu=${GGML_CPU_POWERPC_CPUTYPE})
            endif()
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "loongarch64")
        message(STATUS "loongarch64 detected")
        list(APPEND GGML_CPU_SOURCES ggml-cpu/arch/loongarch/quants.c)

        list(APPEND ARCH_FLAGS -march=loongarch64)
        if (GGML_LASX)
            list(APPEND ARCH_FLAGS -mlasx)
        endif()
        if (GGML_LSX)
            list(APPEND ARCH_FLAGS -mlsx)
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "riscv64")
        message(STATUS "riscv64 detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/riscv/quants.c
            ggml-cpu/arch/riscv/repack.cpp
            )
        if (GGML_CPU_RISCV64_SPACEMIT)
            include(ggml-cpu/cmake/FindSMTIME.cmake)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_RISCV64_SPACEMIT ${RISCV64_SPACEMIT_IME_SPEC})
            list(APPEND GGML_CPU_SOURCES
                ggml-cpu/spacemit/ime.cpp
                ggml-cpu/spacemit/ime.h
                ggml-cpu/spacemit/spine_mem_pool.cpp
                ggml-cpu/spacemit/spine_mem_pool.h
                ggml-cpu/spacemit/repack.cpp
                ggml-cpu/spacemit/repack.h
                ggml-cpu/spacemit/ime_env.cpp
                ggml-cpu/spacemit/ime_env.h
                ggml-cpu/spacemit/ime_kernels.h
                ggml-cpu/spacemit/rvv_kernels.cpp
                ggml-cpu/spacemit/rvv_kernels.h
            )
            if ("RISCV64_SPACEMIT_IME1" IN_LIST RISCV64_SPACEMIT_IME_SPEC)
                list(APPEND GGML_CPU_SOURCES ggml-cpu/spacemit/ime1_kernels.cpp)
            endif()
            if ("RISCV64_SPACEMIT_IME2" IN_LIST RISCV64_SPACEMIT_IME_SPEC)
                list(APPEND GGML_CPU_SOURCES ggml-cpu/spacemit/ime2_kernels.cpp)
            endif()
        endif()
        if(NOT GGML_CPU_ALL_VARIANTS)
            set(MARCH_STR "rv64gc")
            if (GGML_RVV)
                string(APPEND MARCH_STR "v")
            endif()

            if (GGML_RV_ZFH)
                string(APPEND MARCH_STR "_zfh")
            endif()

            if (GGML_XTHEADVECTOR)
                string(APPEND MARCH_STR "_xtheadvector")
            elseif (GGML_RVV)
                if (GGML_RV_ZVFH)
                    string(APPEND MARCH_STR "_zvfh")
                endif()
                if (GGML_RV_ZVFBFWMA)
                    string(APPEND MARCH_STR "_zvfbfwma")
                endif()
            endif()

            if (GGML_RV_ZICBOP)
                string(APPEND MARCH_STR "_zicbop")
            endif()
            if (GGML_RV_ZIHINTPAUSE)
                string(APPEND MARCH_STR "_zihintpause")
            endif()
            if (GGML_RV_ZBA)
                string(APPEND MARCH_STR "_zba")
            endif()
            if (GGML_CPU_RISCV64_SPACEMIT)
                # `xsmtvdotii' is only required for GCC >= 15.
                if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
                    CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
                    string(APPEND MARCH_STR "_xsmtvdotii")
                endif()
            endif()

            list(APPEND ARCH_FLAGS "-march=${MARCH_STR}" -mabi=lp64d)
        else()
            # Begin with the lowest baseline
            set(ARCH_DEFINITIONS "")

            if (GGML_INTERNAL_RVV)
                message(STATUS "RVV enabled")
                list(APPEND ARCH_DEFINITIONS GGML_USE_RVV)
                list(APPEND ARCH_FLAGS -march=rv64gc_v -mabi=lp64d)
            endif()

            ggml_add_cpu_backend_features(${GGML_CPU_NAME} riscv ${ARCH_DEFINITIONS})
        endif()
    elseif (GGML_SYSTEM_ARCH STREQUAL "s390x")
        message(STATUS "s390x detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/s390/quants.c
            ggml-cpu/arch/s390/repack.cpp
            )

        # for native compilation
        if (GGML_NATIVE)
            # check machine level to determine target
            file(READ "/proc/cpuinfo" CPUINFO_CONTENTS)
            string(REGEX REPLACE "machine[ \t\r\n]*=[ \t\r\n]*([0-9]+)" "\\1" S390X_M ${CPUINFO_CONTENTS})

            # TODO: Separation to determine activation of VX/VXE/VXE2
            if (${S390X_M} MATCHES "8561|8562")
                message(STATUS "z15 target")
                list(APPEND ARCH_FLAGS -march=z15)
            elseif (${S390X_M} MATCHES "3931")
                message(STATUS "z16 target")
                list(APPEND ARCH_FLAGS -march=z16)
            elseif (${S390X_M} MATCHES "9175|9176")
                # NOTE: Only available from GCC 15.1.0 onwards. Any z17 machine with compile issues must first verify their GCC version.
                #       binutils must also be updated to the latest for the -march=z17 flag to work. Otherwise, use -march=arch15.
                message(STATUS "z17 target")
                list(APPEND ARCH_FLAGS -march=arch15)
            else()
                message(STATUS "Unknown target")
                message(WARNING "Unknown target. If you are compiling for z14 and earlier, you might have to add -DGGML_VXE=OFF.")
                list(APPEND ARCH_FLAGS -march=native -mtune=native)
            endif()
        # for cross-compilation
        elseif(GGML_CPU_ALL_VARIANTS)
            # range through IBM z15 to z17
            # NOTE: update when a new hardware level is released
            foreach (ZHW RANGE 15 17)
                if(DEFINED GGML_INTERNAL_Z${ZHW})
                    message(STATUS "z${ZHW} cross-compile target")
                    if (ZHW EQUAL 17)
                        # z17 is an alias of arch15, use the arch level for wider toolchain support
                        list(APPEND ARCH_FLAGS -march=arch15)
                    else()
                        list(APPEND ARCH_FLAGS -march=z${ZHW})
                    endif()
                endif()
            endforeach()
        endif()

        if (GGML_VXE OR GGML_INTERNAL_VXE2)
            message(STATUS "VXE2 enabled")
            list(APPEND ARCH_FLAGS -mvx -mzvector)
            list(APPEND ARCH_DEFINITIONS GGML_USE_VXE2)
        endif()

        if (GGML_INTERNAL_NNPA)
            message(STATUS "NNPA enabled")
            list(APPEND ARCH_DEFINITIONS GGML_USE_NNPA)
        endif()

        ggml_add_cpu_backend_features(${GGML_CPU_NAME} s390 ${ARCH_DEFINITIONS})
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
        message(STATUS "Wasm detected")
        list (APPEND GGML_CPU_SOURCES ggml-cpu/arch/wasm/quants.c)
    else()
        message(WARNING "Unknown CPU architecture. Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    if (GGML_CPU_KLEIDIAI)
        # upstream repo requires at least cmake 3.16
        if (CMAKE_VERSION VERSION_LESS 3.16)
            message(FATAL_ERROR "GGML_CPU_KLEIDIAI requires CMake >= 3.16")
        endif()

        set(GGML_CPU_KLEIDIAI_AARCH64 OFF)
        if (GGML_SYSTEM_ARCH STREQUAL "ARM" AND
            (APPLE OR WIN32 OR CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android)$") AND
            (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|arm64-v8a)$" OR
             CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR
             CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
             CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a"))
            set(GGML_CPU_KLEIDIAI_AARCH64 ON)
        endif()
        if (NOT GGML_CPU_KLEIDIAI_AARCH64)
            message(FATAL_ERROR "GGML_CPU_KLEIDIAI requires a Linux, Android, Apple, or Windows AArch64/arm64 target")
        endif()

        message(STATUS "Using KleidiAI optimized kernels if applicable")

        # Fetch KleidiAI sources:
        include(FetchContent)
        set(KLEIDIAI_COMMIT_TAG "v1.32.0")
        set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/releases/download/${KLEIDIAI_COMMIT_TAG}/kleidiai-${KLEIDIAI_COMMIT_TAG}-src.tar.gz")
        set(KLEIDIAI_RELEASE_ARCHIVE_MD5  "05f1953428ffe8e8c7f585dc6d508887")

        set(KLEIDIAI_FETCH_ARGS
            URL ${KLEIDIAI_DOWNLOAD_URL}
            URL_HASH MD5=${KLEIDIAI_RELEASE_ARCHIVE_MD5}
        )
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            list(APPEND KLEIDIAI_FETCH_ARGS DOWNLOAD_EXTRACT_TIMESTAMP NEW)
        endif()

        FetchContent_Declare(kleidiai
            ${KLEIDIAI_FETCH_ARGS}
        )

        # Disable tests and benchmark building
        set(KLEIDIAI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(KLEIDIAI_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)

        # Use the Populate/add_subdirectory flow for compatibility with CMake 3.16.
        FetchContent_GetProperties(kleidiai
            SOURCE_DIR KLEIDIAI_SRC
            BINARY_DIR KLEIDIAI_BIN
            POPULATED KLEIDIAI_POPULATED
        )
        if (NOT KLEIDIAI_POPULATED)
            FetchContent_Populate(kleidiai)
            FetchContent_GetProperties(kleidiai
                SOURCE_DIR KLEIDIAI_SRC
                BINARY_DIR KLEIDIAI_BIN
            )
        endif()

        if (NOT TARGET kleidiai)
            add_subdirectory(
                "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu/kleidiai"
                "${CMAKE_CURRENT_BINARY_DIR}/kleidiai-wrapper"
                EXCLUDE_FROM_ALL
            )
            if (NOT CMAKE_SKIP_INSTALL_RULES AND
                (NOT DEFINED BUILD_SHARED_LIBS OR NOT BUILD_SHARED_LIBS))
                install(TARGETS kleidiai ARCHIVE)
            endif()
        endif()

        if (NOT TARGET kleidiai)
            message(FATAL_ERROR "KleidiAI target was not created")
        endif()

        set_target_properties(kleidiai PROPERTIES POSITION_INDEPENDENT_CODE ON)

        target_link_libraries(${GGML_CPU_NAME} PRIVATE kleidiai)

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_KLEIDIAI)

        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/kleidiai/kleidiai.cpp
            ggml-cpu/kleidiai/kernels.cpp
            ggml-cpu/kleidiai/kleidiai.h
            ggml-cpu/kleidiai/kernels.h
            )
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})
    # ggml-base's ggml-quants.c holds the CPU-traits vec_dots of ~25 types (q4_0..q8_0, q2_K..q6_K, iq1_s..iq4_xs,
    # q6_0, tq1_0, ...) whose vector bodies sit behind #if __AVX2__ / __AVX__ / __SSSE3__. ggml-base is compiled
    # with no -m flags, so on every x86 build those branches were DEAD and the traits table pointed at the scalar
    # loops (row S69-GGML-QUANTS-C-BUILT-WITHOUT-AVX2-PRISM-TERNARY-DOTS-RUN-SCALAR-2026-10-04-01; scan by
    # nm + objdump of the traits targets: 0 ymm instructions in every one of them). When there is exactly ONE CPU
    # backend (no GGML_BACKEND_DL), the program already requires this ISA -- ggml-cpu is built with it -- so
    # ggml-quants.c gets the same flags. GGML_BACKEND_DL keeps ggml-base ISA-neutral (it is shared by variants).
    if (GGML_CPU_BASE_QUANTS_ISA AND CMAKE_VERSION VERSION_LESS 3.18)
        message(WARNING "GGML_CPU_BASE_QUANTS_ISA needs CMake >= 3.18 (set_source_files_properties TARGET_DIRECTORY); ggml-quants.c stays scalar")
    elseif (NOT GGML_BACKEND_DL AND GGML_SYSTEM_ARCH STREQUAL "x86" AND GGML_CPU_BASE_QUANTS_ISA AND ARCH_FLAGS)
        set(GGML_BASE_QUANTS_OPTS ${ARCH_FLAGS})
        if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            # -mfma would let GCC/Clang contract the SCALAR a*b+c of every quantizer in this file into fma and change
            # llama-quantize output bytes; intrinsics (_mm256_fmadd_ps) are unaffected. MEASURED: without it the
            # test-mulmat-gemv dq-vs-orig of q5_K moved 0.02761 -> 0.02762 (quantize_row_q5_K has no ISA branch).
            list(APPEND GGML_BASE_QUANTS_OPTS -ffp-contract=off)
        endif()
        if (MINGW AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
            # same GCC 54412 workaround as the CPU target below: aligned 32-byte spills on a 16-byte SEH stack
            list(APPEND GGML_BASE_QUANTS_OPTS -Wa,-muse-unaligned-vector-move)
        endif()
        set(GGML_BASE_QUANTS_DEFS "")
        foreach (def IN LISTS ARCH_DEFINITIONS)
            if (def MATCHES "^__")   # compiler-feature macros MSVC does not define itself (__FMA__, __F16C__, ...)
                list(APPEND GGML_BASE_QUANTS_DEFS ${def})
            endif()
        endforeach()
        get_target_property(GGML_BASE_SRC_DIR ggml-base SOURCE_DIR)
        set_source_files_properties(${GGML_BASE_SRC_DIR}/ggml-quants.c TARGET_DIRECTORY ggml-base
            PROPERTIES COMPILE_OPTIONS "${GGML_BASE_QUANTS_OPTS}" COMPILE_DEFINITIONS "${GGML_BASE_QUANTS_DEFS}")
        message(STATUS "ggml-base ggml-quants.c built with the CPU ISA: ${GGML_BASE_QUANTS_OPTS} ${GGML_BASE_QUANTS_DEFS}")
    endif()

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()

    # MinGW-w64 GCC + AVX: Win64 only guarantees a 16-byte-aligned stack and GCC cannot realign it for
    # 32-byte spills under SEH (GCC bug 54412), yet it still emits ALIGNED vmovaps to those slots.
    # MEASURED 2026-09-26: iqk flash attention (FlashMS<16,128>::update_M_S) faulted 0xC0000005 on
    # `vmovaps %ymm0,0x240(%rsp)` with rsp+0x240 = ...150 -- every CPU FA call on this toolchain.
    # binutils >= 2.38 rewrites aligned vector moves as unaligned ones (no penalty on aligned data).
    if (MINGW AND CMAKE_C_COMPILER_ID STREQUAL "GNU" AND GGML_SYSTEM_ARCH STREQUAL "x86")
        target_compile_options(${GGML_CPU_NAME} PRIVATE -Wa,-muse-unaligned-vector-move)
    endif()

  if (CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      # The compiler automatically enables "-ffast-math" which can cause NaNs in tests due to "-fassociative-math"
      target_compile_options(${GGML_CPU_NAME} PRIVATE "$<$<OR:$<COMPILE_LANG_AND_ID:C,IntelLLVM>,$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>>:$<$<BOOL:${WIN32}>:/clang:>-fno-associative-math>")
  endif()

endfunction()
