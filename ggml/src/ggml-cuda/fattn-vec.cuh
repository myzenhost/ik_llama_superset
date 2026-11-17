#pragma once
// Umbrella header for TurboQuant flash-attention template instances.
// Includes both f16 and f32 fattn-vec variants and defines DECL_FATTN_VEC_CASE
// which instantiates both paths.
//
// Used by template-instances/fattn-vec-instance-*-turbo*.cu (added in S55-IKLLAMAUNIFIED)

#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"

// Convenience macro: instantiate both f16 and f32 paths for a given (D, K-type, V-type) combo.
#define DECL_FATTN_VEC_CASE(D, type_K, type_V)  \
    DECL_FATTN_VEC_F16_CASE(D, type_K, type_V); \
    DECL_FATTN_VEC_F32_CASE(D, type_K, type_V)
