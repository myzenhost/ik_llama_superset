# ---------------------------------------------------------------------------------------------
# ik_llama's CUDA backend, in upstream's per-backend form.
#
# THIS FILE IS NEITHER SIDE'S. ggml/src/ggml-cuda/ holds three disjoint sets:
#     312 files only in HEAD (the fork's)   155 only in MERGE_HEAD (upstream's)   123 in both
# and the two CUDA entry points live at DIFFERENT paths -- the fork's at ggml/src/ggml-cuda.cu
# (257,927 B), upstream's at ggml/src/ggml-cuda/ggml-cuda.cu (227,998 B). git therefore saw an
# unrelated add/add rather than a conflict, and the divergence never surfaced during the merge.
# The two share a merge base of 2024-06-21; a real 3-way merge across the rename yields 178
# conflict hunks and is tracked separately, sequenced behind a green build.
#
# Until that merge is done this builds THE FORK'S CUDA backend, which is the no-loss position:
# upstream's entry point calls none of the fork's turbo KV flash-attention family, IQK CUDA
# kernels, 66 mmq / ~50 mmvq fork instances, or the fork's own architectures (delta-net, dsa_attn,
# latent_attn, indexer_topk, kda, sinkhorn, hadamard, blend, diffusion-sampling, ds4_comp).
#
# WHY THE GLOBS ARE FOLLOWED BY AN EXPLICIT REMOVE LIST. Both sides' globs are wrong here now.
# Upstream's `*.cu` would compile the fork's 312 files against upstream's headers. The fork's own
# `ggml-cuda/*.cu` was written when only the fork's files were on disk, so post-merge the same
# pattern silently means something different and sweeps in upstream's 155 files, which reference
# APIs the fork's headers do not have. Upstream's files are kept ON DISK on purpose -- greppable,
# and ready for the 178-hunk merge -- they are simply not compiled yet. This list IS the record of
# what is deferred; when a file graduates out of it, it has been merged.
#
# KNOWN BEHAVIOUR CHANGE, stated rather than hidden: the monolith appended ${ARCH_FLAGS} to the
# nvcc host flags (-Xcompiler). ARCH_FLAGS is set only inside the monolith itself, so it is not in
# scope here and is not appended. Upstream's ggml-cuda does not append it either. This affects host
# code generation inside .cu files, not device kernels, and it is a performance nuance rather than
# a capability -- but it is a difference, and a build is what would price it.
# ---------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

if (GGML_MUSA)
    list(APPEND CMAKE_MODULE_PATH "/usr/local/musa/cmake/")
    find_package(MUSAToolkit)
    set(CUDAToolkit_FOUND ${MUSAToolkit_FOUND})
else()
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endif()

if (NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()

message(STATUS "CUDA found")

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 52 == lowest CUDA 12 standard
    # 60 == FP16 CUDA intrinsics
    # 61 == integer CUDA intrinsics
    # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
    if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES "native")
    elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13")
        # CUDA 13 removed offline compilation for compute capability < 7.5 (Maxwell, Pascal, Volta): nvcc rejects
        # compute_50/61/70, so the old default list below cannot be configured at all. Turing is the floor.
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real")
    else()
        if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            set(f1_cuda_archs "60;61;70;75;80")
        else()
            set(f1_cuda_archs "50;61;70;75;80")
        endif()
        # Blackwell (sm_120, RTX 50xx) is only reachable through PTX JIT of an older virtual arch without this entry
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND f1_cuda_archs "120a-real")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${f1_cuda_archs}")
    endif()
endif()
message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if (GGML_MUSA)
    set(CMAKE_CUDA_COMPILER ${MUSAToolkit_MCC_EXECUTABLE})
else()
    enable_language(CUDA)
endif()

# Native NVFP4 W4A4 (nvfp4.cu, ik_llama b71fe5368) needs the arch-specific feature set of sm_120a / sm_121a: plain sm_120 does
# not define __CUDA_ARCH_FEAT_SM120_ALL and the block-scaled mma.sync would compile to nothing. "native" resolves to the plain
# arch, so rewrite it to the "a" variant (CUDA >= 12.8), as ik does; an explicit CMAKE_CUDA_ARCHITECTURES=120a-real is kept.
if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
endif()
# An explicit plain 120 / 121 (-DCMAKE_CUDA_ARCHITECTURES=120, as the merge gate's CUDA tree had until 2026-11-13) takes the same
# rewrite: without it nvfp4.cu silently compiled to stubs on a Blackwell card (ggml_cuda_nvfp4_available() false) and the gate
# passed without ever running the native kernel (row S69-DRIFT-1110-W4A4-BLACKWELL-NVFP4-MXFP4-2026-11-10-01). sm_120a is a
# superset of sm_120 on the same device, and the CUDA >= 13 default list above already names 120a-real.
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    set(ggml_cuda_archs_a "${CMAKE_CUDA_ARCHITECTURES}")
    list(TRANSFORM ggml_cuda_archs_a REPLACE "^(120|121)((-real)?)$" "\\1a\\2")
    if (NOT "${ggml_cuda_archs_a}" STREQUAL "${CMAKE_CUDA_ARCHITECTURES}")
        set(CMAKE_CUDA_ARCHITECTURES "${ggml_cuda_archs_a}")
        message(STATUS "Using CUDA architectures (sm_12x rewritten to the 'a' feature set): ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if (arch MATCHES "^(120|121)a(-real|-virtual)?$")
        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/nvfp4.cu APPEND PROPERTY COMPILE_DEFINITIONS "GGML_CUDA_NVFP4_${CMAKE_MATCH_1}A")
    endif()
endforeach()

# --- source discovery -------------------------------------------------------------------------
# Globs are rooted in THIS directory now, not in ggml/src, so the monolith's "ggml-cuda/*.cu"
# becomes "*.cu". The fork's entry point is still one level up and is named explicitly.
file(GLOB   GGML_HEADERS_CUDA "*.cuh")
list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_CUDA "*.cu")
list(APPEND GGML_SOURCES_CUDA "../ggml-cuda.cu")
# host-compiled (no .cu glob reaches it): NVML free-memory cross-check, see nvml-free.h
list(APPEND GGML_SOURCES_CUDA "nvml-free.cpp")
file(GLOB   SRCS "template-instances/fattn-wmma*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "template-instances/fattn-mma*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})
file(GLOB   SRCS "template-instances/mmvq-instance*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})

# Non-turbo quantised flash-attention vec instances: selected by GGML_CUDA_FA_QUANTS (upstream spelling: "all" or a list of
# <K>-<V>, f16-f16 always built; GGML_CUDA_FA_ALL_QUANTS is its deprecated alias for "all"). The valid pairs and the instance
# files they need come from fattn-vec-pairs.def via ggml_cuda_fattn_vec_fork_instances (ggml/cmake/common.cmake); the same file
# drives the dispatch and supports_op in fattn-vec-f16.cu / fattn-vec-f32.cu, which is why a pair that is not compiled reports
# unsupported (clean fallback) instead of reaching a missing instance. The default string in ggml/CMakeLists.txt selects exactly the
# pairs the old default glob selected.
ggml_cuda_fattn_vec_fork_instances("${CMAKE_CURRENT_SOURCE_DIR}" FATTN_VEC_SRCS)
list(APPEND GGML_SOURCES_CUDA ${FATTN_VEC_SRCS})
# TurboQuant fattn instances are BUILT in every configuration (and are not part of GGML_CUDA_FA_QUANTS): the four vec-kernel
# selectors carry turbo2_0/turbo3_0/turbo4_0 cases (fattn-vec-common.cuh) and the vec kernels static_assert that the selector result
# is non-null, so an unhandled type is a BUILD ERROR rather than a run-time null device-function dereference. turbo3 KV is ~0.37x the
# bytes of q8_0 on the same head, which is the entire point of carrying these types; gating them would leave the default build unable
# to use the feature while still shipping the CPU writer for it. f16 x turbo mixes are instantiated; see fattn-turbo.cuh.
file(GLOB   SRCS "template-instances/fattn-vec-instance-*turbo*.cu")
list(APPEND GGML_SOURCES_CUDA ${SRCS})

# --- upstream-only sources, DEFERRED to the 178-hunk merge, not deleted -------------
# 114 file(s). Each is upstream work that this build does not yet carry; when one is
# merged into the fork's backend it leaves this list. Generated from
# `git ls-tree MERGE_HEAD` minus `git ls-tree HEAD`, not typed by hand.
# Left the list: lightning-indexer.cu (2026-09-28) -- dequant adapted to this fork's
# fattn-common.cuh, dispatch + supports_op wired into the compiled ggml/src/ggml-cuda.cu.
# Left the list: set.cu (2026-10-04) -- an ENABLE, not a disable. RESEARCH-DONE: upstream ggml-org/llama.cpp b9590 ggml/src/ggml-cuda/set.cu read in full;
# identical here bar the two ggml_cuda_cpy calls passing disable_indirection=true. Dispatch + supports_op wired into the compiled ggml/src/ggml-cuda.cu.
# Without it GGML_OP_SET fell back to the CPU backend (kimi-linear's delta-net chunk output).
# Left the list: pool1d.cu (2026-10-18, upstream 29ea9412a #27573) -- unchanged; dispatch + supports_op wired into ggml/src/ggml-cuda.cu.
# Left the list: roll.cu (2026-10-18, upstream 0a5036bee #14919, contiguity rule f8def7fe1 #25928) -- unchanged.
# Left the list: pad_reflect_1d.cu (2026-10-18, upstream b1ab91821 #14659) -- unchanged.
# Left the list: count-equal.cu (2026-10-18, upstream ae8de6d50) -- unchanged.
# Left the list: diag.cu (2026-10-18, upstream b63509262 #17873) -- unchanged.
# Left the list: col2im-1d.cu (2026-10-18, upstream 3a3edc9ac #24417) -- unchanged; op_params { s0, OC, p0 } match ggml_col2im_1d.
# Left the list: sum.cu (2026-10-18, upstream ae8de6d50) -- unchanged; cross-entropy-loss.cu (still listed) is its only other caller.
# Left the list: mean.cu (2026-10-18, upstream aa064b2eb #14313) -- unchanged; its two graph-state queries are re-added to ggml_backend_cuda_context (common.cuh).
# Left the list: out-prod.cu (2026-10-18, upstream ggml-cuda/out-prod.cu, current master shape) -- one line added: cublasSetStream before the GEMMs, the fork convention.
# Left the list: ssm-scan.cu (2026-10-18, upstream incl. 1692f9e50 #26623 rollback K, b62b35098 #22675 SSD prefill) -- one line added: cublasSetStream in the SSD path, as for out-prod.cu; its s,x,dt,A,B,C,ids + K contract is the one ggml_ssm_scan (ggml.c) builds.
# Left the list: wkv.cu (2026-10-18, upstream RWKV_WKV6 / RWKV_WKV7 incl. 598863317 #26111 warp-per-row decode) -- unchanged; src orders match ggml_rwkv_wkv6/7 (ggml.c).
# Left the list: gla.cu (2026-10-18, upstream GATED_LINEAR_ATTN) -- unchanged; src order k,v,q,g,state and the f32 scale op param match ggml_gated_linear_attn (ggml.c).
# Left the list: gated_delta_net.cu (2026-10-18, upstream GATED_DELTA_NET with rollback K) -- unchanged; serves GGML_OP_GATED_DELTA_NET only, the fork's GGML_OP_DELTA_NET stays on delta-net.cu.
# Left the list: conv2d-transpose.cu (2026-10-18, upstream CONV_TRANSPOSE_2D, stride-only p0 form) -- unchanged; op_params[0] = stride matches ggml_conv_transpose_2d_p0 (ggml.c).
# Left the list: cross-entropy-loss.cu (2026-10-18, upstream CROSS_ENTROPY_LOSS + _BACK) -- unchanged; reduces through sum_f32_cuda (sum.cu, compiled since 4461aa83c).
# Left the list: opt-step-adamw.cu, opt-step-sgd.cu (2026-10-18, upstream OPT_STEP_ADAMW / OPT_STEP_SGD) -- unchanged; src orders match ggml_opt_step_adamw/_sgd (ggml.c).
# Left the list: dsv4-hc.cu (2026-10-18, upstream 0dc74e332 #25585 DSV4_HC_COMB/PRE/POST, as synced in 53fd974b4) -- unchanged; upstream master since added scale/gated op params and an optional comb that ggml_dsv4_hc_pre/_comb/_post (ggml.c) do not set, so this shape is the one that matches the fork's constructors and host kernels.
# Left the list: top-k.cu (2026-11-06, upstream 977c48a25 PR #17365/#28547) -- unchanged; GGML_OP_TOP_K dispatch + supports_op wired into ggml/src/ggml-cuda.cu
# Left the list: template-instances/mmq-instance-q2_0.cu (2026-11-11, ik 3d7b18cf0 #2691) -- unchanged; its DECL_MMQ_CASE(GGML_TYPE_Q2_0) now has the fork mmq.cuh trait (load_tiles_q2_0) and the mmq.cu dispatch case.
list(REMOVE_ITEM GGML_SOURCES_CUDA
    "${CMAKE_CURRENT_SOURCE_DIR}/allreduce.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/fattn-tile.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/fwht.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ggml-cuda.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/mmf.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/mmid.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/mmvf.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/snake.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_32.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_32.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq112-dv112.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq128-dv128.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq192-dv128.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq256-dv256.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq320-dv256.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq40-dv40.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq512-dv512.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq576-dv512.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq64-dv64.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq72-dv72.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq80-dv80.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-tile-instance-dkq96-dv96.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-bf16-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-f16-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_0-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q4_1-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_0-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q5_1-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-bf16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-f16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-q4_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-q4_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-q5_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-q5_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/fattn-vec-instance-q8_0-q8_0.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_1.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_10.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_11.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_12.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_13.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_14.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_15.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_16.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_2.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_3.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_4.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_5.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_6.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_7.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_8.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmf-instance-ncols_9.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmq-instance-mxfp4.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/template-instances/mmq-instance-q1_0.cu"
)

# --- the backend library ------------------------------------------------------------------------
ggml_add_backend_library(ggml-cuda ${GGML_HEADERS_CUDA} ${GGML_SOURCES_CUDA})

# The fork's CUDA sources include "ggml.h", "ggml-common.h" and the iqk headers from ggml/src.
target_include_directories(ggml-cuda PRIVATE .. ../.. ../iqk)
# ggml-cuda-fa-quants.h (the GGML_CUDA_FA_QUANTS selection string) is generated into this binary dir by ggml/cmake/common.cmake
target_include_directories(ggml-cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# --- compile definitions ------------------------------------------------------------------------
# These were add_compile_definitions() in the monolith, i.e. DIRECTORY-scoped. They stay
# directory-scoped here, which reaches every target created in this directory -- ggml-cuda. That is
# the same reach they had before for these sources.
add_compile_definitions(GGML_CUDA_DMMV_X=${GGML_CUDA_DMMV_X})
add_compile_definitions(GGML_CUDA_MMV_Y=${GGML_CUDA_MMV_Y})
add_compile_definitions(K_QUANTS_PER_ITERATION=${GGML_CUDA_KQUANTS_ITER})
add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})
add_compile_definitions(GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD})
add_compile_definitions(GGML_CUDA_FUSION=${GGML_CUDA_FUSION})

# OPTION-NAME BRIDGE. Upstream renamed three options; ggml/CMakeLists.txt now declares the NEW
# names while the fork's sources read the OLD macros. An option nothing reads is a SILENT no-op --
# it configures, it prints, and it changes nothing -- which is exactly how GGML_CUDA_USE_GRAPHS
# stopped working. Accept either spelling and define what the sources actually read.
if (GGML_CUDA_GRAPHS OR GGML_CUDA_USE_GRAPHS)
    add_compile_definitions(GGML_CUDA_USE_GRAPHS)
endif()

if (GGML_CUDA_FORCE_DMMV)
    add_compile_definitions(GGML_CUDA_FORCE_DMMV)
endif()

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_IQK_FORCE_BF16)
    add_compile_definitions(GGML_CUDA_IQK_FORCE_BF16)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_VMM)
    add_compile_definitions(GGML_CUDA_NO_VMM)
endif()

if (DEFINED GGML_CUDA_DMMV_Y)
    add_compile_definitions(GGML_CUDA_MMV_Y=${GGML_CUDA_DMMV_Y}) # for backwards compatibility
endif()

if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
    add_compile_definitions(GGML_CUDA_F16)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_MUSA)
    set_source_files_properties(${GGML_SOURCES_CUDA} PROPERTIES LANGUAGE CXX)
    foreach(SOURCE ${GGML_SOURCES_CUDA})
        set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS "-x musa -mtgpu --cuda-gpu-arch=mp_22")
    endforeach()
endif()

# --- linking ------------------------------------------------------------------------------------
# The monolith accumulated GGML_EXTRA_LIBS and linked them into the single ggml target. With a
# per-backend library the same libraries are linked to ggml-cuda directly.
if (GGML_STATIC)
    if (WIN32)
        # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
    else()
        if (GGML_MUSA)
            target_link_libraries(ggml-cuda PRIVATE MUSA::musart_static MUSA::mublas_static)
        else()
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        endif()
    endif()
else()
    if (GGML_MUSA)
        target_link_libraries(ggml-cuda PRIVATE MUSA::musart MUSA::mublas)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
    endif()
endif()

# The NVFP4 prefill GEMM (nvfp4-prefill.cu) calls cublasLt* directly, which put cublasLt64_NN.dll (455 MB in CUDA 13.2)
# in the load-time import table of every executable that links ggml-cuda, next to cublas64_NN.dll, which imports it
# too. With both imports, some process starts map the same cublasLt64_NN.dll path a second time at a private base,
# and peak commit at startup goes from about 250-270 MiB to 481 MiB before any work (measured: 3 of 16 starts without
# this block, 0 of 16 with it). Delay-loading it here leaves one load-time import, the one inside cuBLAS; the delay
# helper binds ggml-cuda's cublasLt* calls to that already-loaded module on first use.
# Row S69-B1019F-STARTUP-COMMIT-TRANSIENT-AND-FIT-CHECK4-REACH-2026-10-26-01.
if (MSVC AND NOT GGML_MUSA)
    set(GGML_CUDA_CUBLASLT_DLL "cublasLt64_${CUDAToolkit_VERSION_MAJOR}.dll")
    target_link_options(ggml-cuda PUBLIC "/DELAYLOAD:${GGML_CUDA_CUBLASLT_DLL}")
    target_link_libraries(ggml-cuda PUBLIC delayimp)
endif()

if (GGML_CUDA_NO_VMM)
    # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
else()
    if (GGML_MUSA)
        target_link_libraries(ggml-cuda PRIVATE MUSA::musa_driver)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()
endif()

# NCCL: the fork reads GGML_NCCL, upstream declares GGML_CUDA_NCCL. Same bridge as graphs above.
if (GGML_CUDA_NCCL OR GGML_NCCL)
    find_package(NCCL)
    if (NCCL_FOUND)
        message(STATUS "NCCL found: ${NCCL_LIBRARIES}")
        add_compile_definitions(GGML_USE_NCCL)
        target_link_libraries(ggml-cuda      PRIVATE ${NCCL_LIBRARIES})
        target_include_directories(ggml-cuda PRIVATE ${NCCL_INCLUDE_DIRS})
    else()
        message(STATUS "Warning: NCCL not found -> building without NCCL support")
    endif()
endif()

# --- nvcc flags ---------------------------------------------------------------------------------
set(CUDA_CXX_FLAGS "")
set(CUDA_FLAGS -use_fast_math -extended-lambda -lineinfo)
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# nvcc 13.2.0 and 13.2.1 (13.2.51 / 13.2.78) miscompile a byte read through a uint8_t* view of a
# packed int when it feeds a table index, for compute_120 (ggml-org/llama.cpp#28581, bisected in
# #28784: 13.2.2 is clean). The IQ1_S/IQ2_S/IQ3_S sites are worked around with __byte_perm; any
# other kernel using the same idiom is not covered, so say so at configure time.
if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2" AND CUDAToolkit_VERSION VERSION_LESS "13.2.86")
    message(WARNING "CUDA ${CUDAToolkit_VERSION}: nvcc 13.2.0/13.2.1 miscompile byte extraction from packed ints for "
                    "sm_120 (ggml-org/llama.cpp#28581). IQ1_S/IQ2_S/IQ3_S carry a workaround; prefer CUDA >= 13.2.2.")
endif()

if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
    # none (not recommended) / speed (nvcc default) / balance / size
    list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
endif()

if (GGML_FATAL_WARNINGS)
    list(APPEND CUDA_FLAGS -Werror all-warnings)
endif()

if (GGML_ALL_WARNINGS AND NOT MSVC)
    # ggml_get_flags comes from ggml/cmake/common.cmake. The monolith called its OWN get_flags
    # (ggml/src/CMakeLists.txt:905), which does not survive the move to a dispatcher.
    set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
    if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
        list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
    endif()

    execute_process(COMMAND ${NVCC_CMD} -Xcompiler --version
                    OUTPUT_VARIABLE CUDA_CCFULLVER ERROR_QUIET)

    if (NOT CUDA_CCFULLVER MATCHES clang)
        set(CUDA_CCID "GNU")
        execute_process(COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                        OUTPUT_VARIABLE CUDA_CCVER ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        if (CUDA_CCFULLVER MATCHES Apple)
            set(CUDA_CCID "AppleClang")
        else()
            set(CUDA_CCID "Clang")
        endif()
        string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
    endif()

    message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")
    ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
    list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # passed to -Xcompiler below
endif()

if (NOT MSVC)
    list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
else()
    # CCCL 3.2 onwards requires a cpp-standard-compliant preprocessor for MSVC
    # https://github.com/NVIDIA/cccl/pull/6827
    list(APPEND CUDA_CXX_FLAGS /Zc:preprocessor)
endif()

list(JOIN CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # host compiler flags as a single argument
if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
    list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
endif()

target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")

# the KT trellis encoder matches the CPU quantizer bit for bit only with denormals kept (Thireus d9e286846).
# ik compiles its CUDA quantize encoders (IQ*_KT, IQ*_KS) without flush-to-zero; -use_fast_math above implies
# -ftz=true, and the per-source option lands after the target options on the nvcc line. It must sit HERE:
# source properties are directory-scoped and only reach targets created in this directory. ik keeps it in
# ggml/src/CMakeLists.txt because ik creates ggml-cuda there; the upstream-structured tree creates it in this file,
# and the merge did not carry the line at all. SOURCE-READ: ik_llama.cpp cea2d61ff ggml/src/CMakeLists.txt:1269
# (kt-encoder from d9e286846 #2572, ks-encoder from 86f629dab #2671); MIT.
set_source_files_properties(kt-encoder.cu ks-encoder.cu PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-ftz=false>")
