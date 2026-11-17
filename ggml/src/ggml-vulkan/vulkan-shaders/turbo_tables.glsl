// TurboQuant constants shared by every Vulkan turbo shader (writer, WHT op, dequant, flash attention).
//
// DO NOT EDIT ONE COPY ALONE. These MUST stay identical to ggml-turbo-quant.c (turbo_cpu_s1 / turbo_cpu_s2, CENTROIDS_2BIT/3BIT/4BIT and the
// nearest_centroid_* thresholds) and to ggml-cuda/turbo-quant.cuh. Turbo KV blocks are written by one backend and read by another (the CPU
// writer feeds the GPU readers), and a divergence raises no error: both sides still produce valid blocks and every decoded vector is silently
// wrong. scripts/check-turbo-tables-consistent.py parses THIS file and fails, naming the copy, if any value differs (and its --selftest
// proves it can fail).
//
// Kept in ONE include on purpose: the first Vulkan port of this feature (TheTom/llama-cpp-turboquant bcb85fc3ae85, MIT) carried the sign arrays
// in three shaders and the centroids in six places.

const float TURBO_S1[128] = float[128](
    -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,
     1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1, -1,
    -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1,
     1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1,  1,  1,  1, -1,  1,
    -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1,  1,
     1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,
     1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,  1,  1, -1,  1
);

const float TURBO_S2[128] = float[128](
     1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1,
     1, -1,  1,  1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1,  1,  1,
     1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1,  1,
    -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1,
     1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1,
    -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1
);

const float TURBO_INV_SQRT_128 = 0.08838834764831845;

// Lloyd-Max centroids for N(0, 1/128) and the thresholds between neighbours (== nearest_centroid_*bit in ggml-turbo-quant.c).
const float TURBO_CENTROIDS_2BIT[4] = float[4](-0.133462, -0.039994, 0.039994, 0.133462);
const float TURBO_MID_2BIT[3]       = float[3](-0.086728, 0.0, 0.086728);

const float TURBO_CENTROIDS_3BIT[8] = float[8](
    -0.190207, -0.118786, -0.066822, -0.021663,
     0.021663,  0.066822,  0.118786,  0.190207
);
const float TURBO_MID_3BIT[7] = float[7](-0.154496, -0.092804, -0.044243, 0.0, 0.044243, 0.092804, 0.154496);

const float TURBO_CENTROIDS_4BIT[16] = float[16](
    -0.241529, -0.182877, -0.143016, -0.111036,
    -0.083292, -0.058050, -0.034299, -0.011349,
     0.011349,  0.034299,  0.058050,  0.083292,
     0.111036,  0.143016,  0.182877,  0.241529
);
const float TURBO_MID_4BIT[15] = float[15](
    -0.212203, -0.162947, -0.127026, -0.097164,
    -0.070671, -0.046174, -0.022824,  0.0,
     0.022824,  0.046174,  0.070671,  0.097164,
     0.127026,  0.162947,  0.212203
);

// Sign vector of the 32-wide weight transform (golden ratio hash; ggml-cuda/turbo-quant.cuh TQ_WEIGHT_SIGNS, ggml-turbo-quant.c TQ3_0_SIGNS).
// GGML_OP_TURBO_WHT with group_size 32 applies it once: before the butterfly on the forward pass, after it on the inverse.
const float TURBO_TQW_SIGNS[32] = float[32](
    +1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1,
    -1, +1, +1, -1, +1, -1, -1, +1
);
