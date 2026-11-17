// MTP drafter: embedding-input (multimodal / mtmd) prompt batches.  Row S69-ENGREV-SPEC-MTP-MTMD-EMBD-INPUT-2026-11-14-01.
// Pure functions, no llama.h dependency, so they are unit-testable without a model (tests/test-spec-mtp-embd.cpp).
//
// An image / audio chunk reaches the target as a legacy llama_batch whose `embd` is set and `token` is null. The MTP mirror
// (common_speculative_impl_mtp::process) used to return early for such a batch, so the MTP KV cache had no row for any position
// of an embd chunk and every draft after an image prompt read an incomplete cache. Upstream llama.cpp fixed the same gap in 781dbc5ac
// (spec: MTP process() accepts add_embd rows). The row selection below is the part of that fix that does not need a model.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

enum common_spec_mtp_input_kind {
    COMMON_SPEC_MTP_INPUT_NONE = 0, // nothing to mirror (no token ids and no embedding rows)
    COMMON_SPEC_MTP_INPUT_TOKENS,   // token ids: the MTP context embeds them itself
    COMMON_SPEC_MTP_INPUT_EMBD,     // embedding rows (a legacy llama_batch with `embd` set): copied through to the MTP context
};

// A legacy llama_batch carries token ids or embedding rows, never both for decode; when a caller sets both, the embedding rows are what
// the target actually consumed, so they win.
static inline common_spec_mtp_input_kind common_spec_mtp_input_kind_of(bool has_token, bool has_embd) {
    if (has_embd) {
        return COMMON_SPEC_MTP_INPUT_EMBD;
    }
    return has_token ? COMMON_SPEC_MTP_INPUT_TOKENS : COMMON_SPEC_MTP_INPUT_NONE;
}

// Copies the embedding rows `rows` (batch row indices, in order) of `src` (row-major, `width` floats per row) into `dst`
// (rows.size() * width floats). Returns false, writing nothing, on a null pointer or a non-positive width.
static inline bool common_spec_mtp_gather_embd_rows(const float * src, int32_t width, const std::vector<int32_t> & rows, float * dst) {
    if (src == nullptr || dst == nullptr || width <= 0) {
        return false;
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] < 0) {
            return false;
        }
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        std::memcpy(dst + i * (size_t) width, src + (size_t) rows[i] * (size_t) width, (size_t) width * sizeof(float));
    }
    return true;
}
