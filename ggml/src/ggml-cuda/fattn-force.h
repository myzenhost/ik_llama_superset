// Env override for the CUDA flash-attention kernel family: GGML_CUDA_FATTN_FORCE=vec|tile|mma.
//
// Pure, header-only, no CUDA and no ggml types, so a CPU test can exercise it. The dispatcher in
// fattn.cu computes which families are LEGAL for the node in hand (the same predicates the default
// path already uses) and asks fattn_force_decide() whether the override picks one of them. An
// override naming a family that is not legal for this node DECLINES (returns FATTN_FORCE_NONE) and
// the static dispatch runs unchanged, so the override can never select an illegal kernel.
//
// Default (env unset, empty or unknown) is NONE = today's behaviour exactly.
//
// Spec: row S69-F2-PICK-FATTN-TILE-AUTOTUNE-2026-10-06-01, "CHEAPER FIRST STEP". The shape
// is informed by OpenVINO sdpa_gen_micro.cpp:1621-1698 (a selection step plus a user override
// hook that is only honoured when valid; Apache-2.0); behaviour only, no code copied.
// Family names deliberately omit wmma and mma_f16: their per-head-size legality is not read yet
// (fattn-wmma-f16.cu switches per head size and per precision), so they stay static.
#pragma once

#include <cstring>

enum fattn_force_kind {
    FATTN_FORCE_NONE = 0,
    FATTN_FORCE_VEC  = 1,
    FATTN_FORCE_TILE = 2,
    FATTN_FORCE_MMA  = 3,
};

// null / "" / unknown text -> NONE. Matching is exact and case-sensitive on purpose: a typo must
// not silently force something.
inline fattn_force_kind fattn_force_parse(const char * s) {
    if (!s || !*s) return FATTN_FORCE_NONE;
    if (!std::strcmp(s, "vec"))  return FATTN_FORCE_VEC;
    if (!std::strcmp(s, "tile")) return FATTN_FORCE_TILE;
    if (!std::strcmp(s, "mma"))  return FATTN_FORCE_MMA;
    return FATTN_FORCE_NONE;
}

// The requested family if (and only if) it is legal for this node, else NONE.
inline fattn_force_kind fattn_force_decide(fattn_force_kind want, bool legal_vec, bool legal_tile, bool legal_mma) {
    switch (want) {
        case FATTN_FORCE_VEC:  return legal_vec  ? want : FATTN_FORCE_NONE;
        case FATTN_FORCE_TILE: return legal_tile ? want : FATTN_FORCE_NONE;
        case FATTN_FORCE_MMA:  return legal_mma  ? want : FATTN_FORCE_NONE;
        default:               return FATTN_FORCE_NONE;
    }
}
