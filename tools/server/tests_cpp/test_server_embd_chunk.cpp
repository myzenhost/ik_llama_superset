// Standalone test for server-embd-chunk.h. Build: g++ -std=c++17 -I.. test_server_embd_chunk.cpp -o t && ./t
#include "server-embd-chunk.h"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- name mapping (controls: an unknown name is refused, "error" is the default behaviour) ---------------------------
    server_embd_long_text m = SERVER_EMBD_LONG_TEXT_MEAN;
    CHECK(server_embd_long_text_from_name("error", m) && m == SERVER_EMBD_LONG_TEXT_ERROR);
    CHECK(server_embd_long_text_from_name("truncate", m) && m == SERVER_EMBD_LONG_TEXT_TRUNCATE);
    CHECK(server_embd_long_text_from_name("mean", m) && m == SERVER_EMBD_LONG_TEXT_MEAN);
    CHECK(!server_embd_long_text_from_name("Mean", m));
    CHECK(!server_embd_long_text_from_name("", m));

    std::vector<server_embd_chunk_range> r;
    std::string err;

    // ---- plan: 25 body tokens, max 10, overlap 3 -> stride 7: [0,10) [7,17) [14,24) [21,25) ---------------------------------
    CHECK(server_embd_chunk_plan(25, 10, 3, false, r, err));
    CHECK(r.size() == 4);
    if (r.size() == 4) {
        CHECK(r[0].begin == 0  && r[0].end == 10);
        CHECK(r[1].begin == 7  && r[1].end == 17);
        CHECK(r[2].begin == 14 && r[2].end == 24);
        CHECK(r[3].begin == 21 && r[3].end == 25);
        // neighbours share exactly `overlap` tokens when the next chunk is full, and every token is covered
        CHECK(r[0].end - r[1].begin == 3);
        CHECK(r[1].end - r[2].begin == 3);
    }

    // ---- exact fit: no trailing chunk made of overlap only ----------------------------------------------------------------
    CHECK(server_embd_chunk_plan(10, 10, 3, false, r, err));
    CHECK(r.size() == 1 && r[0].begin == 0 && r[0].end == 10);
    CHECK(server_embd_chunk_plan(17, 10, 3, false, r, err)); // [0,10) [7,17): the second ends exactly at 17
    CHECK(r.size() == 2 && r[1].end == 17);

    // ---- truncate keeps the first chunk only ------------------------------------------------------------------------------
    CHECK(server_embd_chunk_plan(25, 10, 3, true, r, err));
    CHECK(r.size() == 1 && r[0].begin == 0 && r[0].end == 10);

    // ---- impossible geometry: overlap >= chunk would never advance -------------------------------------------------------
    CHECK(!server_embd_chunk_plan(25, 3, 3, false, r, err) && !err.empty());
    CHECK(!server_embd_chunk_plan(25, 0, 0, false, r, err));
    CHECK( server_embd_chunk_plan(25, 4, 3, false, r, err)); // stride 1 is slow but legal
    CHECK(r.size() == 22);                                   // begins 0..21, last [21,25)

    // ---- coverage over a sweep: union of ranges is [0, n), ranges are ordered, never longer than max ----------------------
    for (size_t n = 1; n <= 60; n++) {
        CHECK(server_embd_chunk_plan(n, 9, 2, false, r, err));
        size_t covered = 0;
        bool ok = !r.empty() && r.front().begin == 0 && r.back().end == n;
        for (size_t i = 0; i < r.size(); i++) {
            ok = ok && r[i].end > r[i].begin && r[i].end - r[i].begin <= 9;
            if (i) { ok = ok && r[i].begin <= r[i - 1].end && r[i].begin > r[i - 1].begin; }
            covered = r[i].end;
        }
        CHECK(ok && covered == n);
    }

    // ---- empty body: one empty range (the caller keeps the existing path for it) -------------------------------------------
    CHECK(server_embd_chunk_plan(0, 10, 3, false, r, err) && r.size() == 1 && r[0].end == 0);

    // ---- mean pool: average of the chunk vectors -----------------------------------------------------------------------------
    {
        std::vector<std::vector<float>> v = { { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f } };
        const auto m3 = server_embd_mean_pool(v);
        CHECK(m3.size() == 2);
        if (m3.size() == 2) {
            CHECK(std::fabs(m3[0] - 2.0f / 3.0f) < 1e-6f);
            CHECK(std::fabs(m3[1] - 2.0f / 3.0f) < 1e-6f);
        }
        // one chunk -> identical vector (a short text keeps its exact vector)
        const auto m1 = server_embd_mean_pool({ { 0.25f, -0.5f } });
        CHECK(m1.size() == 2 && m1[0] == 0.25f && m1[1] == -0.5f);
        // mismatched lengths and empty input are refused with an empty vector
        CHECK(server_embd_mean_pool({ { 1.0f }, { 1.0f, 2.0f } }).empty());
        CHECK(server_embd_mean_pool({}).empty());
    }

    if (fails == 0) {
        std::printf("test_server_embd_chunk: all passed\n");
    }
    return fails ? 1 : 0;
}
