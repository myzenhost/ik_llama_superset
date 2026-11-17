// TurboQuant SYCL flash-attention vec instances (head sizes 128/256/512).
// hs64 is not instantiated: the rotation group is 128 elements, so a 64-wide head is not a whole number of groups.

#include "../fattn-vec.hpp"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
