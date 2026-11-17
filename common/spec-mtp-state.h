// Versioned, validated blob for the MTP drafter's boundary hidden state (checkpoint + prompt-cache entry).
// Row S69-ENGREV-BEELLAMA-MTP-CARRY-STATE-CKPT-CACHE-2026-10-27-01.  Pure functions, no llama.h dependency, so it is unit-testable without a model.
//
// Layout (little-endian host order, same as the rest of the in-memory checkpoint blobs):
//   u32 magic 'MTP1' | u32 version | u32 n_embd | i32 pos | f32 h[n_embd]
// Decode rejects an empty blob (caller resets), a wrong magic / version / width, and any size mismatch, instead of
// reinterpreting foreign bytes as a hidden row (a restore into a different model would otherwise read garbage silently).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#define COMMON_SPEC_MTP_STATE_MAGIC   0x3150544du  // "MTP1"
#define COMMON_SPEC_MTP_STATE_VERSION 1u

enum common_spec_mtp_state_status {
    COMMON_SPEC_MTP_STATE_OK = 0,
    COMMON_SPEC_MTP_STATE_EMPTY,        // zero bytes: no state saved, caller resets
    COMMON_SPEC_MTP_STATE_BAD_SIZE,     // shorter than the header or body does not match n_embd
    COMMON_SPEC_MTP_STATE_BAD_MAGIC,
    COMMON_SPEC_MTP_STATE_BAD_VERSION,
    COMMON_SPEC_MTP_STATE_BAD_WIDTH,    // saved n_embd differs from this model's
};

static inline size_t common_spec_mtp_state_header_bytes() { return 3 * sizeof(uint32_t) + sizeof(int32_t); }

static inline void common_spec_mtp_state_encode(const float * h, uint32_t n_embd, int32_t pos, std::vector<uint8_t> & out) {
    const uint32_t magic = COMMON_SPEC_MTP_STATE_MAGIC;
    const uint32_t ver   = COMMON_SPEC_MTP_STATE_VERSION;
    const size_t   hb    = common_spec_mtp_state_header_bytes();
    out.resize(hb + (size_t) n_embd * sizeof(float));
    uint8_t * p = out.data();
    std::memcpy(p, &magic, 4); p += 4;
    std::memcpy(p, &ver,   4); p += 4;
    std::memcpy(p, &n_embd, 4); p += 4;
    std::memcpy(p, &pos,   4); p += 4;
    std::memcpy(p, h, (size_t) n_embd * sizeof(float));
}

// h_out must hold n_embd floats; written only on OK.
static inline common_spec_mtp_state_status common_spec_mtp_state_decode(const std::vector<uint8_t> & in, uint32_t n_embd, float * h_out, int32_t & pos_out) {
    if (in.empty()) {
        return COMMON_SPEC_MTP_STATE_EMPTY;
    }
    const size_t hb = common_spec_mtp_state_header_bytes();
    if (in.size() < hb) {
        return COMMON_SPEC_MTP_STATE_BAD_SIZE;
    }
    uint32_t magic, ver, width;
    int32_t  pos;
    const uint8_t * p = in.data();
    std::memcpy(&magic, p, 4); p += 4;
    std::memcpy(&ver,   p, 4); p += 4;
    std::memcpy(&width, p, 4); p += 4;
    std::memcpy(&pos,   p, 4); p += 4;
    if (magic != COMMON_SPEC_MTP_STATE_MAGIC) return COMMON_SPEC_MTP_STATE_BAD_MAGIC;
    if (ver   != COMMON_SPEC_MTP_STATE_VERSION) return COMMON_SPEC_MTP_STATE_BAD_VERSION;
    if (width != n_embd) return COMMON_SPEC_MTP_STATE_BAD_WIDTH;
    if (in.size() != hb + (size_t) n_embd * sizeof(float)) return COMMON_SPEC_MTP_STATE_BAD_SIZE;
    std::memcpy(h_out, p, (size_t) n_embd * sizeof(float));
    pos_out = pos;
    return COMMON_SPEC_MTP_STATE_OK;
}
