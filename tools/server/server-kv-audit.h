#pragma once

// Debug-only KV audit: per-slot token/position canary chain, pool leak check, non-finite logit detector.
//
// Row S69-ENGREV-SGLANG-KV-CANARY-2026-10-27-01. SOURCE-READ: SGLang (Apache-2.0) srt/kv_canary/api.py, kernels/ops/kv_canary/consts.py:6,60-71
// (chain hash of prev, token, position), write_ref.py:115-202 (stamp after forward), verify_ref.py:107-155 (verify chain before attend),
// scheduler_components/invariant_checker.py:76-93 + 296-557 (idle accounting identity), PR #42690 (opt-in NaN detection before sampling).
// Specified in temp/research/findings/S69_ENGINE_REVIEW_sglang2_DEEP_2026-10-27.md. Written from that description, no code copied.
//
// What this is NOT: SGLang's canary also hashes the real K/V bytes inside the attention kernel. The fork has no such kernel hook, so the chain
// here covers the HOST view of a slot (the token at each position the KV was built from); a restore or rewind that leaves the slot's token list
// disagreeing with what was stamped is caught, a corrupted K/V byte is not. The leak check and the NaN check look at the device-side result.
//
// Pure logic with no llama dependency, so the unit test (tests/test-server-kv-audit.cpp) runs without a model.

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// one chain step: mix(prev, token, position), splitmix64 finaliser over the three words
inline uint64_t kv_audit_mix3(uint64_t prev, int32_t token, int32_t pos) {
    uint64_t x = prev + 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t) (uint32_t) token * 0xbf58476d1ce4e5b9ULL;
    x ^= (uint64_t) (uint32_t) pos   * 0x94d049bb133111ebULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct kv_audit_canary {
    std::vector<uint64_t> chain;   // chain[i] covers tokens[0..i]
    int64_t               key = -1; // the owner (task id): a new owner restarts the chain instead of violating

    // position of the first token that disagrees with the stamped chain, -1 = consistent. Positions at or beyond the stamped length are new.
    int64_t verify(const std::vector<int32_t> & toks) const {
        uint64_t h = 0;
        const size_t n = chain.size() < toks.size() ? chain.size() : toks.size();
        for (size_t i = 0; i < n; ++i) {
            h = kv_audit_mix3(h, toks[i], (int32_t) i);
            if (h != chain[i]) {
                return (int64_t) i;
            }
        }
        return -1;
    }

    void stamp(const std::vector<int32_t> & toks) {
        chain.clear();
        chain.reserve(toks.size());
        uint64_t h = 0;
        for (size_t i = 0; i < toks.size(); ++i) {
            h = kv_audit_mix3(h, toks[i], (int32_t) i);
            chain.push_back(h);
        }
    }

    // One audit step for a slot. Returns the violating position or -1.
    //  * a different owner (new request) or a SHORTER list (rewind, context shift, restore) is a legal rewrite: re-stamp, no verdict
    //  * the same owner with a list at least as long as the stamp must still agree on the stamped prefix
    int64_t audit(int64_t owner, const std::vector<int32_t> & toks) {
        int64_t bad = -1;
        if (owner == key && toks.size() >= chain.size()) {
            bad = verify(toks);
        }
        key = owner;
        stamp(toks);
        return bad;
    }
};

// Pool conservation, one-sided on purpose. Slots may share cells (a prefix copied between sequences counts once in the pool) and a sliding
// window or recurrent model holds FEWER cells than tokens, so used < sum(slot tokens) is normal. Cells in use beyond what every slot could
// account for are owned by nobody: a leak. slack covers the rows of the batch in flight.
struct kv_audit_pool_result {
    bool    leak = false;
    int64_t unowned = 0;   // used_cells - (sum_slot_tokens + slack) when leak
};

inline kv_audit_pool_result kv_audit_pool_check(int64_t used_cells, int64_t sum_slot_tokens, int64_t slack) {
    kv_audit_pool_result r;
    const int64_t bound = sum_slot_tokens + (slack > 0 ? slack : 0);
    if (used_cells > bound) {
        r.leak    = true;
        r.unowned = used_cells - bound;
    }
    return r;
}

// index of the first NaN or infinity in a logit row, -1 = all finite
inline int64_t kv_audit_first_nonfinite(const float * row, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        // -inf is a legal mask value (min_tokens, grammar); NaN and +inf never are
        if (std::isnan(row[i]) || (std::isinf(row[i]) && row[i] > 0)) {
            return i;
        }
    }
    return -1;
}

inline std::string kv_audit_describe(int slot_id, int64_t pos, const char * what) {
    return "kv-audit: slot " + std::to_string(slot_id) + " position " + std::to_string((long long) pos) + ": " + what;
}
