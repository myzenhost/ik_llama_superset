#pragma once

// Combining several LoRA deltas into one (TIES / DARE), for tools/export-lora.
//
// Row S69-LORAX-PICK-REQUEST-TIME-LORA-MERGE-TIES-DARE-2026-09-29-01.  Behaviour read from predibase/lorax
// server/lorax_server/utils/merges/{strategies,utils}.py (Apache-2.0; derived there from HF PEFT PR 1364, Apache-2.0) and re-written here from the spec
// below; no code copied.
//
// NUANCES THE ROW'S DESCRIPTION DID NOT SAY (all read from that source, then tested in tests/test-lora-merge.cpp):
//  1. TIES pruning keeps k = floor(density * numel) entries by magnitude, not ceil; density >= 1 returns the tensor untouched, density < 0 is an error.
//     Ties in magnitude at the cut: the source's topk order is unspecified; here the LOWEST index wins, so the result is reproducible.
//  2. The sign is elected on the PRUNED, UNWEIGHTED tensors: total = sign of sum(pruned), frequency = sign of sum(sign(pruned)); a sum of exactly 0
//     elects +. Only afterwards are the per-adapter weights applied.
//  3. disjoint merge = sum(weight_i * pruned_i over the adapters whose sign equals the elected sign) / max(COUNT of those adapters, 1). It divides by the
//     NUMBER of agreeing adapters, not by the sum of their weights. A pruned (zero) entry has sign 0, never equals +-1, and is not counted.
//  4. DARE keeps each entry with probability `density` and must rescale the survivors by 1/density so the expectation is preserved. LoRAX's own
//     random_pruning(rescale=True) computes torch.div(...) and discards the result, so ITS dare_linear / dare_ties never rescale (a bug: present in code,
//     not working). This implementation rescales, and the test asserts E[pruned] == original.
//  5. The deltas here are DENSE (scale * B @ A per module), not the A and B factors merged separately as LoRAX does: averaging A and B separately is not
//     the average of the products and needs equal ranks; the dense form needs neither.
//  6. Weights default to 1 each; with `linear` the result is the plain weighted sum, which is what applying the adapters with those scales already does.

#include <cstddef>
#include <cstdint>
#include <string>

enum lora_merge_strategy {
    LORA_MERGE_LINEAR     = 0,
    LORA_MERGE_TIES       = 1,
    LORA_MERGE_DARE_LINEAR = 2,
    LORA_MERGE_DARE_TIES  = 3,
};

enum lora_sign_method {
    LORA_SIGN_TOTAL     = 0,
    LORA_SIGN_FREQUENCY = 1,
};

struct lora_merge_params {
    lora_merge_strategy strategy    = LORA_MERGE_LINEAR;
    float               density     = 1.0f;                 // fraction of entries kept by the pruning strategies, in [0, 1]
    lora_sign_method    sign_method = LORA_SIGN_TOTAL;
    uint64_t            seed        = 0;                    // DARE only; the same seed gives the same result
};

// "linear" | "ties" | "dare_linear" | "dare_ties" (case-insensitive); returns false for anything else
bool lora_merge_strategy_from_name(const std::string & name, lora_merge_strategy & out);
bool lora_sign_method_from_name(const std::string & name, lora_sign_method & out);
const char * lora_merge_strategy_name(lora_merge_strategy s);

// Merge n dense deltas (each numel floats) with per-adapter weights (n floats; nullptr = all 1) into out (numel floats).
// Returns false and fills *err for an invalid request (n < 1, density outside [0, 1], null pointers); out is untouched then.
bool lora_merge_deltas(const lora_merge_params & p, const float * const * deltas, const float * weights, int n, size_t numel,
                       float * out, std::string * err);

// Dense LoRA delta for one module: out[o*n_in + i] = scale * sum_k B[o*rank + k] * A[k*n_in + i]
// (A is rank x n_in row-major, B is n_out x rank row-major, out is n_out x n_in row-major), i.e. the delta_W that lora_merge_deltas() merges.
// `scale` is the caller's alpha/rank * user scale. Returns false for non-positive dimensions or null pointers.
bool lora_delta_dense(const float * A, const float * B, int rank, int n_in, int n_out, float scale, float * out);
