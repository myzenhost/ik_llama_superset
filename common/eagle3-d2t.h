#pragma once

// EAGLE3 d2t (draft-vocab -> target-vocab) map: which of the two on-disk conventions a gguf carries, as a pure
// function of the map itself, so the runtime can read either one. Row S69-EAGLE3-D2T-CONVERTER-ABSOLUTE-VS-RUNTIME-DELTA-2026-10-17-01.
//
// Two conventions exist and both are on disk:
//   ABSOLUTE  d2t[i] is the target id of draft id i. Upstream llama.cpp writes and reads this one: its converter adds
//             arange to the HF delta table, and src/models/eagle3.cpp:307-319 uses d2t directly as ggml_set_rows
//             indices into the target-vocab logits (SOURCE-READ, MIT, 2026-10-17). This tree's in-tree converters
//             do the same (conversion/llama.py, conversion/qwen.py).
//   DELTA     target id = i + d2t[i]. The HF/EAGLE checkpoint table as trained, and what this fork's earlier #29
//             converter stored raw (measured on that gguf: d2t[0..100] = 0, d2t[1000] = 131).
// The runtime used to assume DELTA unconditionally, so an ABSOLUTE gguf mapped draft id i to i + t_i: every draft
// was refused at verify (output still correct, acceptance ~0).
//
// A reading is VALID when every mapped id is in [0, n_vocab_tgt) and no two draft ids map to the same target id (the
// map is an injection; both converters already refuse to write anything else). When both readings are valid the
// ABSOLUTE one wins, because it is upstream's convention and the one every current converter writes.

#include <cstdint>
#include <vector>

enum common_eagle3_d2t_form {
    COMMON_EAGLE3_D2T_INVALID  = 0,  // neither reading is an in-range injection
    COMMON_EAGLE3_D2T_ABSOLUTE = 1,
    COMMON_EAGLE3_D2T_DELTA    = 2,
};

inline const char * common_eagle3_d2t_form_name(common_eagle3_d2t_form f) {
    switch (f) {
        case COMMON_EAGLE3_D2T_ABSOLUTE: return "absolute";
        case COMMON_EAGLE3_D2T_DELTA:    return "delta";
        default:                         return "invalid";
    }
}

// True when mapping draft id i to (delta ? i + d2t[i] : d2t[i]) is an injection into [0, n_vocab_tgt).
inline bool common_eagle3_d2t_reading_valid(const std::vector<int32_t> & d2t, int32_t n_vocab_tgt, bool delta) {
    if (n_vocab_tgt <= 0) {
        return false;
    }
    std::vector<bool> seen((size_t) n_vocab_tgt, false);
    for (size_t i = 0; i < d2t.size(); ++i) {
        const int64_t t = (int64_t) d2t[i] + (delta ? (int64_t) i : 0);
        if (t < 0 || t >= n_vocab_tgt || seen[(size_t) t]) {
            return false;
        }
        seen[(size_t) t] = true;
    }
    return true;
}

inline common_eagle3_d2t_form common_eagle3_d2t_classify(const std::vector<int32_t> & d2t, int32_t n_vocab_tgt) {
    if (d2t.empty()) {
        return COMMON_EAGLE3_D2T_INVALID;
    }
    if (common_eagle3_d2t_reading_valid(d2t, n_vocab_tgt, /*delta =*/ false)) {
        return COMMON_EAGLE3_D2T_ABSOLUTE;
    }
    if (common_eagle3_d2t_reading_valid(d2t, n_vocab_tgt, /*delta =*/ true)) {
        return COMMON_EAGLE3_D2T_DELTA;
    }
    return COMMON_EAGLE3_D2T_INVALID;
}

// Classify, then rewrite d2t in place as ABSOLUTE target ids so the caller maps with a plain lookup. An INVALID map
// is left untouched; the caller must not map through it.
inline common_eagle3_d2t_form common_eagle3_d2t_normalize(std::vector<int32_t> & d2t, int32_t n_vocab_tgt) {
    const common_eagle3_d2t_form f = common_eagle3_d2t_classify(d2t, n_vocab_tgt);
    if (f == COMMON_EAGLE3_D2T_DELTA) {
        for (size_t i = 0; i < d2t.size(); ++i) {
            d2t[i] = (int32_t) ((int64_t) d2t[i] + (int64_t) i);
        }
    }
    return f;
}
