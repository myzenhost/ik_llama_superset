// Unit test for tools/mtmd/mtmd-embd-cache.h (byte-budgeted LRU of vision-encoder outputs). Model-free, no llama link.
// RESEARCH-DONE: behaviour spec from the tabbyAPI vision cache helper (see the header's SOURCE-READ / SOURCE-NUANCE).
#include "mtmd-embd-cache.h"

#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static mtmd_embd_key key_of(const std::vector<float> & px, uint64_t extra = 0) {
    mtmd_embd_hasher h;
    h.add_u64(extra);
    h.add_floats(px.data(), px.size());
    return h.finish(px.size());
}

static std::vector<float> rows(size_t n, float v) { return std::vector<float>(n, v); }

int main() {
    // ---- the hash: equal content -> equal key; one float, the size, the extra field, the split point each change it -----------
    {
        std::vector<float> a(1000, 0.5f), b = a;
        CHECK(key_of(a) == key_of(b));
        b[999] = 0.5000001f;
        CHECK(!(key_of(a) == key_of(b)));
        std::vector<float> c(1001, 0.5f);
        CHECK(!(key_of(a) == key_of(c)));
        CHECK(!(key_of(a, 1) == key_of(a, 2)));
        mtmd_embd_hasher h1, h2;
        h1.add_bytes("ab", 2); h1.add_bytes("c", 1);
        h2.add_bytes("a", 1);  h2.add_bytes("bc", 2);
        CHECK(!(h1.finish(0) == h2.finish(0)));       // the same bytes split differently are different content
        std::vector<float> d(7, 1.0f), e(7, 1.0f);    // length not a multiple of 8 bytes: the tail is hashed too
        e[6] = 2.0f;
        CHECK(!(key_of(d) == key_of(e)));
    }

    // ---- budget 0 is today's behaviour: nothing stored, nothing found --------------------------------------------------------
    {
        mtmd_embd_cache c(0);
        CHECK(!c.enabled());
        const auto k = key_of(rows(10, 1));
        auto v = rows(10, 1);
        CHECK(!c.put(k, v.data(), v.size()));
        std::vector<float> out;
        CHECK(!c.get(k, out));
        CHECK(c.stats().entries == 0 && c.stats().bytes == 0);
    }

    // ---- hit returns the stored rows (a copy); a different image is a miss ----------------------------------------------------
    {
        mtmd_embd_cache c(1000 * sizeof(float));
        const auto pa = rows(10, 1), pb = rows(10, 2);
        const auto ea = rows(100, 7.0f);
        CHECK(c.put(key_of(pa), ea.data(), ea.size()));
        std::vector<float> out;
        CHECK(c.get(key_of(pa), out) && out == ea);
        CHECK(!c.get(key_of(pb), out));                // equal size, different content: miss
        const auto s = c.stats();
        CHECK(s.hits == 1 && s.misses == 1 && s.entries == 1 && s.bytes == 100 * sizeof(float) && s.budget == 1000 * sizeof(float));
    }

    // ---- LRU by bytes: the least recently used goes first, and a hit refreshes ------------------------------------------------
    {
        mtmd_embd_cache c(300 * sizeof(float));        // room for three 100-float entries
        auto A = key_of(rows(1, 1)), B = key_of(rows(1, 2)), C = key_of(rows(1, 3)), D = key_of(rows(1, 4));
        const auto e = rows(100, 0.0f);
        std::vector<float> out;
        c.set_scope(1);
        CHECK(c.put(A, e.data(), e.size()) && c.put(B, e.data(), e.size()) && c.put(C, e.data(), e.size()));
        c.set_scope(2);                                // a new request: nothing is protected
        CHECK(c.get(A, out));                          // A is now the most recently used; B is the oldest
        CHECK(c.put(D, e.data(), e.size()));           // evicts B
        CHECK(!c.get(B, out) && c.get(A, out) && c.get(C, out) && c.get(D, out));
        CHECK(c.stats().evictions == 1);
    }

    // ---- bytes, not counts: one big entry evicts several small ones -------------------------------------------------------------
    {
        mtmd_embd_cache c(400 * sizeof(float));
        const auto small = rows(100, 0.0f), big = rows(300, 0.0f);
        auto A = key_of(rows(1, 1)), B = key_of(rows(1, 2)), C = key_of(rows(1, 3)), BIG = key_of(rows(1, 9));
        c.set_scope(1);
        c.put(A, small.data(), small.size()); c.put(B, small.data(), small.size()); c.put(C, small.data(), small.size());
        c.set_scope(2);
        CHECK(c.put(BIG, big.data(), big.size()));     // needs 300: evicts A, B and C (300 + 100 > 400 would not fit with C)
        std::vector<float> out;
        CHECK(c.get(BIG, out));
        CHECK(c.stats().bytes <= 400 * sizeof(float));
    }

    // ---- an entry bigger than the budget is not stored and evicts nothing ----------------------------------------------------------
    {
        mtmd_embd_cache c(100 * sizeof(float));
        const auto fits = rows(50, 0.0f), huge = rows(101, 0.0f);
        auto A = key_of(rows(1, 1)), H = key_of(rows(1, 8));
        c.set_scope(1);
        CHECK(c.put(A, fits.data(), fits.size()));
        c.set_scope(2);
        CHECK(!c.put(H, huge.data(), huge.size()));
        std::vector<float> out;
        CHECK(c.get(A, out));                          // untouched
        CHECK(c.stats().not_stored == 1 && c.stats().evictions == 0);
    }

    // ---- scope: a request with more images than the budget holds fills it and stops; the next turn gets the hits -----------------
    {
        mtmd_embd_cache c(200 * sizeof(float));        // room for two of the three images
        const auto e = rows(100, 0.0f);
        auto I1 = key_of(rows(1, 1)), I2 = key_of(rows(1, 2)), I3 = key_of(rows(1, 3));
        std::vector<float> out;
        // turn 1: three images, all new
        c.set_scope(10);
        CHECK(!c.get(I1, out)); CHECK(c.put(I1, e.data(), e.size()));
        CHECK(!c.get(I2, out)); CHECK(c.put(I2, e.data(), e.size()));
        CHECK(!c.get(I3, out)); CHECK(!c.put(I3, e.data(), e.size()));   // would have to evict I1 or I2, which this request already used
        CHECK(c.stats().entries == 2 && c.stats().evictions == 0);
        // turn 2: the same three images in the same request
        c.set_scope(11);
        const uint64_t hits_before = c.stats().hits;
        CHECK(c.get(I1, out)); CHECK(c.get(I2, out)); CHECK(!c.get(I3, out));
        CHECK(c.stats().hits - hits_before == 2);       // 2 of 3 hit; without the scope rule every one of them would miss (thrash)
        // the contrast: the same trace with the protection of one request removed (a fresh scope before every image)
        mtmd_embd_cache t(200 * sizeof(float));
        uint64_t sc = 100;
        for (int turn = 0; turn < 2; turn++) {
            for (const auto & k : {I1, I2, I3}) {
                t.set_scope(sc++);
                if (!t.get(k, out)) {
                    t.put(k, e.data(), e.size());
                }
            }
        }
        CHECK(t.stats().hits == 0);                     // plain LRU on a 3-image loop over room for 2: zero hits (the control)
    }

    // ---- a refused put evicts nothing (decided before evicting) ---------------------------------------------------------------------
    {
        mtmd_embd_cache c(300 * sizeof(float));
        const auto e = rows(100, 0.0f), e2 = rows(250, 0.0f);
        auto A = key_of(rows(1, 1)), B = key_of(rows(1, 2)), BIG = key_of(rows(1, 3));
        c.set_scope(1);
        c.put(A, e.data(), e.size());
        c.set_scope(2);
        c.put(B, e.data(), e.size());                   // B belongs to scope 2
        CHECK(!c.put(BIG, e2.data(), e2.size()));       // 100 protected + 250 > 300: refused ...
        CHECK(c.stats().entries == 2 && c.stats().evictions == 0);   // ... and A was not thrown away for nothing
    }

    // ---- re-putting a present key only refreshes it (the source would remove it first) ---------------------------------------------------
    {
        mtmd_embd_cache c(200 * sizeof(float));
        const auto e = rows(100, 3.0f);
        auto A = key_of(rows(1, 1));
        c.set_scope(1);
        CHECK(c.put(A, e.data(), e.size()));
        CHECK(c.put(A, e.data(), e.size()));
        CHECK(c.stats().entries == 1 && c.stats().bytes == 100 * sizeof(float));
    }

    // ---- shrinking the budget drops the oldest entries; clear empties --------------------------------------------------------------------
    {
        mtmd_embd_cache c(300 * sizeof(float));
        const auto e = rows(100, 0.0f);
        auto A = key_of(rows(1, 1)), B = key_of(rows(1, 2)), C = key_of(rows(1, 3));
        c.set_scope(1);
        c.put(A, e.data(), e.size()); c.put(B, e.data(), e.size()); c.put(C, e.data(), e.size());
        c.set_budget(100 * sizeof(float));
        std::vector<float> out;
        CHECK(c.stats().entries == 1 && c.get(C, out) && !c.get(A, out));
        c.clear();
        CHECK(c.stats().entries == 0 && c.stats().bytes == 0);
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
