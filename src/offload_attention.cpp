// offload_attention.cpp — implementation of the offload-attention support contract declared in
// offload_attention.h (task #220 kernel port, DEC-S69-OFFLOAD-ATTENTION-INJECT-INTO-IKLLAMA).
//
// WHAT THIS IS. The header has been a DECLARATION ONLY since 2026-07-06 ("IMPLEMENTED BY: the #220
// kernel port"), so the fork carried the contract and none of the behaviour. Measured 2026-08-19:
// `kvflash` appears exactly once in 759 fork source files -- a comment in offload_attention.h --
// and the shipped llama-server.exe --help (31,653 bytes) has 0 occurrences of kvflash and 0 of
// paged, while the control --flash-attn is present. So PagedAttention/offload-attention was ABSENT
// as a working feature, not "present but dormant" as an earlier finding recorded.
//
// WHAT IT MUST MATCH. The reference is the harness module cortex/cortex/inference/offload_attention.py,
// CPU-proven cosine>=0.99 vs dense attention at a 2% page budget across four selectors. This file
// reproduces that math exactly, in the same order, so the two can be compared numerically rather
// than by description:
//   pages   nP = ceil(S/P); the final page is padded by REPEATING THE LAST KEY ROW (not zeros --
//           zeros would drag a min/max page bound toward the origin and change QUEST's ranking).
//   scale   qs = q / sqrt(d), applied BEFORE scoring, as in the reference.
//   quest   score_p = sum_d max(qs_d*Kmin_pd, qs_d*Kmax_pd)      (ICML'24 2406.10774)
//   shadowkv score_p = qs . mean_p                                (2410.21465 landmark term)
//   retattn mu = mean over ALL S keys; land_p = mean_p(K - mu); qc = qs - mean(qs);
//           score_p = qc . land_p                                 (2409.10516 OOD correction)
//   attend  s = (q . Kg)/sqrt(d) over gathered tokens only, softmax, out = softmax @ Vg.
//
// The `ann` selector from the reference is deliberately NOT ported here. It is a faiss HNSW index
// over the same retattn-corrected landmarks and selects the same pages (the reference records
// recall == exact scan), so it changes cost rather than result, and it would add a faiss
// dependency to the fork. BENCH-PENDING: its cost benefit here is UNMEASURED -- no timing has been
// taken on this host, and the fork's intended speed path is the gather kernel rather than the
// selector scan. RETATTN is its exact-scan equivalent and is ported.
//
// NO ARBITRARY CONTEXT CAP (rule 3.8 + standing user directive): n_tokens is read from the store at
// call time, page_budget is a fraction or an explicit count, and nothing here truncates context.
//
// SCOPE, stated rather than implied: this is the CPU/reference-correct implementation that makes
// the declared contract real and testable. It is not yet the CUDA gather kernel, and it is not yet
// called by llama-build-context.cpp. Both are tracked (#220 kernel compose, #93 ggml_paged_attention
// substrate); a caller that gathers on-device belongs with those.

#include "offload_attention.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace ik_offload {

namespace {

inline int64_t n_pages(int64_t n_tokens, int32_t page_size) {
    if (page_size <= 0) return 0;
    return (n_tokens + page_size - 1) / page_size;
}

// Row `t` of the key matrix, with the reference's padding rule: any index at or past the end
// repeats the LAST REAL ROW. Keeping this in one place is deliberate -- the padding convention is
// the part most likely to drift from the reference and least likely to be noticed, because a wrong
// pad still produces plausible scores.
inline int64_t clamp_row(const OffloadKVStore & store, int64_t t) {
    const int64_t last = store.n_tokens - 1;
    return t < store.n_tokens ? t : last;
}

// Read one element, converting when the backing buffer is F16. The fork's KV is F16 by default
// (src/llama.cpp:4906), so this is the ordinary path, not a fallback. ggml_fp16_to_fp32 is used
// rather than a hand-rolled bit twiddle so the conversion matches what the rest of the tree does.
inline float read_elem(const void * base, KVType t, int64_t idx) {
    if (t == KVType::F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(base)[idx]);
    }
    return static_cast<const float *>(base)[idx];
}

inline float key_at(const OffloadKVStore & store, int64_t t, int64_t j) {
    return read_elem(store.k_backing, store.k_type, clamp_row(store, t) * store.head_dim + j);
}

inline float val_at(const OffloadKVStore & store, int64_t t, int64_t j) {
    return read_elem(store.v_backing, store.v_type, clamp_row(store, t) * store.head_dim + j);
}

// Mean of every key row, over the whole store. RETATTN's OOD correction needs the global key mean,
// not a per-page one -- centering by a per-page mean would remove exactly the between-page signal
// the selector ranks on.
std::vector<float> global_key_mean(const OffloadKVStore & store) {
    const int64_t d = store.head_dim;
    std::vector<double> acc(static_cast<size_t>(d), 0.0);
    for (int64_t t = 0; t < store.n_tokens; ++t) {
        for (int64_t j = 0; j < d; ++j) acc[static_cast<size_t>(j)] += key_at(store, t, j);
    }
    std::vector<float> mu(static_cast<size_t>(d), 0.0f);
    if (store.n_tokens > 0) {
        for (int64_t j = 0; j < d; ++j) {
            mu[static_cast<size_t>(j)] =
                static_cast<float>(acc[static_cast<size_t>(j)] / static_cast<double>(store.n_tokens));
        }
    }
    return mu;
}

}  // namespace

std::vector<int32_t> select_pages(const float * q, const OffloadKVStore & store,
                                  const OffloadConfig & cfg, int32_t n_keep_pages) {
    std::vector<int32_t> tokens;
    if (q == nullptr || store.k_backing == nullptr || store.n_tokens <= 0 || store.head_dim <= 0) {
        return tokens;   // empty, not a crash: an empty store has no critical pages by definition
    }

    const int64_t d  = store.head_dim;
    const int32_t P  = cfg.page_size > 0 ? cfg.page_size : 16;
    const int64_t nP = n_pages(store.n_tokens, P);
    if (nP <= 0) return tokens;

    int32_t keep = n_keep_pages;
    if (keep <= 0) keep = 1;
    if (keep > nP) keep = static_cast<int32_t>(nP);

    // qs = q / sqrt(d), matching the reference's pre-scaling.
    const float inv = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> qs(static_cast<size_t>(d));
    for (int64_t j = 0; j < d; ++j) qs[static_cast<size_t>(j)] = q[j] * inv;

    // RETATTN needs the global key mean and the query's own mean removed.
    std::vector<float> mu;
    float qs_mean = 0.0f;
    if (cfg.selector == Selector::RETATTN) {
        mu = global_key_mean(store);
        double s = 0.0;
        for (int64_t j = 0; j < d; ++j) s += qs[static_cast<size_t>(j)];
        qs_mean = static_cast<float>(s / static_cast<double>(d));
    }

    std::vector<float> score(static_cast<size_t>(nP), 0.0f);
    for (int64_t p = 0; p < nP; ++p) {
        const int64_t t0 = p * P;
        double acc = 0.0;

        if (cfg.selector == Selector::QUEST) {
            // Per-dimension min/max over the page, then the elementwise upper bound.
            for (int64_t j = 0; j < d; ++j) {
                float kmin = key_at(store, t0, j);
                float kmax = kmin;
                for (int32_t u = 1; u < P; ++u) {
                    const float kv = key_at(store, t0 + u, j);
                    if (kv < kmin) kmin = kv;
                    if (kv > kmax) kmax = kv;
                }
                const float a = qs[static_cast<size_t>(j)] * kmin;
                const float b = qs[static_cast<size_t>(j)] * kmax;
                acc += static_cast<double>(a > b ? a : b);
            }
        } else {
            // SHADOWKV and RETATTN both score against a page MEAN landmark; they differ only in
            // whether the global key mean is removed first (and the query centred).
            std::vector<double> land(static_cast<size_t>(d), 0.0);
            for (int32_t u = 0; u < P; ++u) {
                for (int64_t j = 0; j < d; ++j) land[static_cast<size_t>(j)] += key_at(store, t0 + u, j);
            }
            const double invP = 1.0 / static_cast<double>(P);
            for (int64_t j = 0; j < d; ++j) {
                double lj = land[static_cast<size_t>(j)] * invP;
                double qj = qs[static_cast<size_t>(j)];
                if (cfg.selector == Selector::RETATTN) {
                    lj -= static_cast<double>(mu[static_cast<size_t>(j)]);
                    qj -= static_cast<double>(qs_mean);
                }
                acc += qj * lj;
            }
        }
        score[static_cast<size_t>(p)] = static_cast<float>(acc);
    }

    // Top-`keep` pages by score. partial_sort over an index vector: the reference uses torch.topk,
    // which is also unstable w.r.t. ties, so the ordering of equal scores is not part of the contract.
    std::vector<int32_t> order(static_cast<size_t>(nP));
    std::iota(order.begin(), order.end(), 0);

    // ATTENTION SINK: pin the first `sink` pages, then rank only what is left. order[] is
    // 0,1,2,... so pages [0, sink) are ALREADY in place -- starting the partial_sort at
    // order.begin() + sink both reserves them and excludes them from the ranking, without
    // a second pass or a set difference.
    int32_t sink = cfg.sink_pages;
    if (sink < 0) sink = 0;
    if (sink > keep) sink = keep;   // never reserve more than the whole budget
    std::partial_sort(order.begin() + sink, order.begin() + keep, order.end(),
                      [&score](int32_t a, int32_t b) {
                          return score[static_cast<size_t>(a)] > score[static_cast<size_t>(b)];
                      });
    order.resize(static_cast<size_t>(keep));

    // Expand pages to token indices, dropping anything past the real end (the pad is a scoring
    // device only -- padded positions must never be attended).
    tokens.reserve(static_cast<size_t>(keep) * static_cast<size_t>(P));
    for (int32_t p : order) {
        const int64_t t0 = static_cast<int64_t>(p) * P;
        for (int32_t u = 0; u < P; ++u) {
            const int64_t t = t0 + u;
            if (t < store.n_tokens) tokens.push_back(static_cast<int32_t>(t));
        }
    }
    return tokens;
}

int offload_attend_step(const float * q, const OffloadKVStore & store,
                        const OffloadConfig & cfg, float * out) {
    if (q == nullptr || out == nullptr || store.k_backing == nullptr ||
        store.v_backing == nullptr || store.n_tokens <= 0 || store.head_dim <= 0) {
        return -1;
    }

    const int64_t d  = store.head_dim;
    const int32_t P  = cfg.page_size > 0 ? cfg.page_size : 16;
    const int64_t nP = n_pages(store.n_tokens, P);

    // page_budget is a FRACTION below 1.0, or an explicit page COUNT at or above it. Same rule as
    // the reference, so a config can move between the two implementations unchanged.
    int32_t keep;
    if (cfg.page_budget >= 1.0f) {
        keep = static_cast<int32_t>(cfg.page_budget);
    } else {
        keep = static_cast<int32_t>(static_cast<float>(nP) * cfg.page_budget);
        if (keep < 1) keep = 1;
    }

    const std::vector<int32_t> tok = select_pages(q, store, cfg, keep);
    if (tok.empty()) return -1;

    // Attention over the gathered set only.
    const float inv = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> logits(tok.size());
    float maxv = -INFINITY;
    for (size_t i = 0; i < tok.size(); ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < d; ++j) s += static_cast<double>(q[j]) * static_cast<double>(key_at(store, tok[i], j));
        logits[i] = static_cast<float>(s) * inv;
        if (logits[i] > maxv) maxv = logits[i];
    }

    double sum = 0.0;
    for (size_t i = 0; i < tok.size(); ++i) {
        logits[i] = std::exp(logits[i] - maxv);     // max-subtracted softmax; it overflows otherwise
        sum += static_cast<double>(logits[i]);
    }
    if (sum <= 0.0) return -1;

    std::vector<double> acc(static_cast<size_t>(d), 0.0);
    for (size_t i = 0; i < tok.size(); ++i) {
        const double w = static_cast<double>(logits[i]) / sum;
        for (int64_t j = 0; j < d; ++j) acc[static_cast<size_t>(j)] += w * static_cast<double>(val_at(store, tok[i], j));
    }
    for (int64_t j = 0; j < d; ++j) out[j] = static_cast<float>(acc[static_cast<size_t>(j)]);

    return 0;
}

}  // namespace ik_offload
