// Model-free test of src/offload_select_share.h. Spec: KTransformers (Apache-2.0) kvcache_attn.cpp:737-848; behaviour only, no code copied.
#include "../src/offload_select_share.h"

#include <cstdio>
#include <cstdlib>

using namespace ik_offload;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// stand-in selector: keeps pages 0 (sink) and the page holding the highest-"score" key; deterministic from n_kv
static std::vector<char> fake_select(int64_t n_kv, int64_t P, int64_t salt) {
    const int64_t nP = (n_kv + P - 1) / P;
    std::vector<char> k((size_t) nP, 0);
    k[0] = 1;
    k[(size_t) ((salt * 7 + n_kv) % nP)] = 1;
    return k;
}

// the decode driver: returns the kept list for a step using reuse (every_n) and counts selector runs
static std::vector<char> step(SelectReuse & st, int64_t every_n, int64_t n_kv, int64_t P, int64_t seq, int & runs, bool append = true) {
    std::vector<char> out;
    if (!select_reuse_step(st, every_n, n_kv, P, seq, out)) {
        out = fake_select(n_kv, P, 3);
        select_reuse_store(st, out, n_kv, seq);
        ++runs;
    } else if (!append) {
        // negative control: a reuse that DROPS the newest-page append
        out = st.keep;
        out.back() = 0;
    }
    return out;
}

int main() {
    const int64_t P = 4;
    // layer sharing
    CHECK(select_share_is_leader(5, 1, 0));                       // default: every layer leads
    CHECK(select_share_leader(5, 1, 0) == 5);
    CHECK(select_share_is_leader(0, 4, 0) && select_share_is_leader(4, 4, 0) && !select_share_is_leader(3, 4, 0));
    CHECK(select_share_leader(3, 4, 0) == 0 && select_share_leader(7, 4, 0) == 4);
    CHECK(select_share_leader(1, 4, 2) == 1);                     // before the first leader (offset 2): own scan
    CHECK(select_share_leader(3, 4, 2) == 2);

    // n = 1 is today's behaviour: the selector runs every step and the result equals the plain selection
    {
        SelectReuse st; int runs = 0; bool same = true;
        for (int64_t n = 17; n < 40; ++n) same = same && step(st, 1, n, P, 0, runs) == fake_select(n, P, 3);
        CHECK(same);
        CHECK(runs == 23);
    }
    // n = 4: selector on steps 0,4,8; other steps = last list + newest page
    {
        SelectReuse st; int runs = 0; bool ok = true;
        std::vector<char> base;
        for (int64_t i = 0; i < 9; ++i) {
            const int64_t n = 17 + i;
            std::vector<char> got = step(st, 4, n, P, 0, runs);
            if (i % 4 == 0) { ok = ok && got == fake_select(n, P, 3); base = got; }
            else {
                std::vector<char> want = base;
                want.resize((size_t) ((n + P - 1) / P), 0);
                want[(size_t) ((n - 1) / P)] = 1;
                ok = ok && got == want;
                CHECK(got[(size_t) ((n - 1) / P)] == 1);          // the newest page is visible
            }
        }
        CHECK(ok);
        CHECK(runs == 3);
    }
    // negative control: dropping the append loses the newest page at a page boundary (n_kv 20 -> 21 starts page 5)
    {
        SelectReuse st; int runs = 0;
        step(st, 4, 20, P, 0, runs);
        std::vector<char> got = step(st, 4, 21, P, 0, runs, false);
        CHECK(got[(size_t) ((21 - 1) / P)] == 0);                 // the control really does lose it, so the real test above is not vacuous
    }
    // invalidation: KV shrink, sequence change, jump > 1 key
    {
        SelectReuse st; int runs = 0;
        step(st, 4, 20, P, 0, runs);
        step(st, 4, 19, P, 0, runs); CHECK(runs == 2);            // shrink
        step(st, 4, 20, P, 1, runs); CHECK(runs == 3);            // sequence changed
        step(st, 4, 25, P, 1, runs); CHECK(runs == 4);            // prefill-sized jump
    }
    std::printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
