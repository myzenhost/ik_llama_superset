// TurboQuant CUDA flash-attention vec kernel instantiation.
//
// hs64 is NOT instantiated: the turbo rotation group is 128 elements (QK_TURBO3), so a 64-wide
// head is not a whole number of groups and the block cannot rotate cooperatively. The kernel
// static_asserts this rather than falling through to a nullptr device function pointer.
//
// turbo4 (the 4-bit PolarQuant layout, 66-byte block) is wired as of 2026-10-05: the CPU inverse
// that task #278 named as the blocker no longer exists -- ggml-turbo-quant.c writes the same
// FWHT-rotated, norm-corrected centroid blocks as turbo2/3, and tests/test-turbo3-roundtrip.cpp
// pins that (turbo4 relative error 0.009138 against 1.957019 on the retired 3-bit+QJL layout).

// OPT IN to the turbo fattn path for THIS translation unit. Must precede the include: it gates
// both the turbo selector cases and the turbo constant tables, which are deliberately kept out of
// the ~130 non-turbo instance TUs (see fattn-turbo.cuh for the measured reason).
#define GGML_CUDA_FATTN_TURBO

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
