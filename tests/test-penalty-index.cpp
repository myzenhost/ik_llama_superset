// Repetition/frequency/presence penalties: the index-by-token-id fast path (ADOPTED from unslothai/llama.cpp penalties-index-by-token-id) gives the
// same candidate array as the per-candidate hash lookup it replaced, on every layout the sampler can be handed.
//
// The reference below is the old loop written out independently (count the last-n tokens, then for each candidate look its id up). The cases are
// chosen so that each branch of the new code is the one under test:
//   identity layout, full vocab          -> fast path (id == index for every candidate)
//   identity layout, history ids outside the candidate array (negative, >= size) -> fast path with the per-key bounds check; must not write out of range
//   permuted candidates                  -> the guard must reject the fast path (ids are not their indices) and the lookup path must still be right
//   a subset of the vocab (ids sparse)   -> same, the shape top-k leaves behind
// Mutation record (commit message): removing the bounds check makes the out-of-range case write past the array (caught by the sentinel canary below);
// dropping the identity guard makes the permuted case penalise the wrong tokens (max |d| ~ 1).
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void reference(std::vector<llama_token_data> & c, const std::vector<llama_token> & hist, size_t last_n, float rep, float freq, float pres) {
    if (last_n == 0 || (rep == 1.0f && freq == 0.0f && pres == 0.0f)) {
        return;
    }
    std::map<llama_token, int> cnt;
    for (size_t i = 0; i < last_n; ++i) cnt[hist[i]]++;
    for (auto & d : c) {
        auto it = cnt.find(d.id);
        if (it == cnt.end()) continue;
        d.logit = d.logit <= 0 ? d.logit * rep : d.logit / rep;
        d.logit -= float(it->second) * freq + float(it->second > 0) * pres;
    }
}

// run the library function on `cands` (copied into a guarded buffer) and compare with the reference
static void run_case(const char * name, std::vector<llama_token_data> cands, const std::vector<llama_token> & hist, float rep, float freq, float pres) {
    std::vector<llama_token_data> want = cands;
    reference(want, hist, hist.size(), rep, freq, pres);

    // sentinel canary after the array: an out-of-range write by the fast path would land in it
    const size_t n = cands.size();
    std::vector<llama_token_data> buf(n + 8);
    for (size_t i = 0; i < n; ++i) buf[i] = cands[i];
    for (size_t i = n; i < n + 8; ++i) buf[i] = { -12345, 777.0f, 888.0f };
    llama_token_data_array arr = { buf.data(), n, false };
    llama_sample_repetition_penalties(nullptr, &arr, hist.data(), hist.size(), rep, freq, pres);

    double md = 0;
    for (size_t i = 0; i < n; ++i) {
        CHECK(buf[i].id == want[i].id, "%s: candidate %zu id changed", name, i);
        md = std::fmax(md, std::fabs((double) buf[i].logit - (double) want[i].logit));
    }
    bool canary = true;
    for (size_t i = n; i < n + 8; ++i) canary = canary && buf[i].id == -12345 && buf[i].logit == 777.0f && buf[i].p == 888.0f;
    // a vacuous pass would be a call that changed nothing: count how many logits the reference moved
    size_t moved = 0;
    for (size_t i = 0; i < n; ++i) moved += want[i].logit != cands[i].logit;
    std::printf("  %-44s n=%6zu history=%3zu  max|d| %.3e  logits moved %zu\n", name, n, hist.size(), md, moved);
    CHECK(md == 0.0, "%s: differs from the reference by %.3e", name, md);
    CHECK(canary, "%s: wrote past the end of the candidate array", name);
    CHECK(moved > 0, "%s: the case moved no logit, so it proves nothing", name);
}

int main() {
    llama_backend_init();   // also starts the ggml clock; the penalty function reads it (0xC0000094, integer divide by zero, without this)
    std::mt19937 rng(20261005);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int V = 4096;
    std::vector<llama_token_data> full(V);
    for (int i = 0; i < V; ++i) full[i] = { i, nd(rng), 0.0f };

    std::vector<llama_token> hist;
    for (int i = 0; i < 64; ++i) hist.push_back((llama_token) (rng() % 300));   // repeats on purpose

    run_case("identity layout, full vocab", full, hist, 1.3f, 0.1f, 0.2f);
    run_case("identity layout, repeat only", full, hist, 1.5f, 0.0f, 0.0f);

    std::vector<llama_token> wild = hist;
    wild[3] = -1; wild[10] = V; wild[11] = V + 1000; wild[12] = 0x7fffffff;
    run_case("identity layout, ids outside the array", full, wild, 1.3f, 0.1f, 0.2f);

    std::vector<llama_token_data> perm = full;
    std::shuffle(perm.begin(), perm.end(), rng);
    run_case("permuted candidates", perm, hist, 1.3f, 0.1f, 0.2f);

    std::vector<llama_token_data> sub;
    for (int i = 0; i < V; i += 7) sub.push_back(full[i]);
    for (auto & t : hist) t = (llama_token) ((t / 7) * 7);                          // make some history ids land in the subset
    run_case("sparse subset (top-k shape)", sub, hist, 1.3f, 0.1f, 0.2f);

    // small identity array with a history id equal to its size: the boundary of the bounds check
    std::vector<llama_token_data> tiny = { {0, 1.0f, 0}, {1, -1.0f, 0}, {2, 2.0f, 0} };
    run_case("identity, history id == size", tiny, std::vector<llama_token>{ 1, 2, 3, 3, 0 }, 1.2f, 0.3f, 0.4f);

    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
