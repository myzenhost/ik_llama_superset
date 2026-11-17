// XAttention-style prefill block selection (src/llama-kv-xattn.h): planted heavy block is selected, first column + diagonal always, nothing right of the
// diagonal, threshold 1.0 = full causal triangle, a single block returns early, shape errors return empty.
// Row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01 (step 1: estimator + selection).   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the header did not exist (no estimator in the fork tree).  Controls: the planted block MOVES with the plant (head 0 plants key block 2, head 1
// plants key block 4: each head selects its own), and unstructured random data selects every causal block at threshold 0.9 (the estimator is not a
// constant "keep few" rule: on data without a heavy block nothing is pruned).
#include "../src/llama-kv-xattn.h"

#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    using namespace justician_kv;
    const int D = 16, H = 2, B = 16, ST = 4, NB = 8, N = NB * B;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 0.05f);

    auto fill = [&](std::vector<float> & v) { for (float & x : v) x = nd(rng); };
    std::vector<float> Q((size_t) H * N * D), K((size_t) H * N * D);
    fill(Q); fill(K);
    // plant: q block 6 rows all = u (scale 6), key block `plant[h]` rows all = u
    const int plant[2] = { 2, 4 };
    for (int h = 0; h < H; ++h) {
        for (int r = 0; r < B; ++r) {
            for (int d = 0; d < D; ++d) {
                const float u = (d == h) ? 6.0f : 0.0f;               // a different direction per head
                Q[((size_t) h * N + 6 * B + r) * D + d] = u;
                K[((size_t) h * N + plant[h] * B + r) * D + d] = u;
            }
        }
    }
    xattn_cfg cfg;
    cfg.stride = ST; cfg.block = B; cfg.threshold = 0.9f;
    std::vector<uint8_t> m = xattn_block_mask(Q.data(), K.data(), H, N, N, D, cfg);
    CHECK(m.size() == (size_t) H * NB * NB);
    auto at = [&](const std::vector<uint8_t> & mm, int h, int qb, int kb) { return (int) mm[((size_t) h * NB + qb) * NB + kb]; };
    for (int h = 0; h < H; ++h) {
        CHECK(at(m, h, 6, 0) == 1);                 // first column
        CHECK(at(m, h, 6, 6) == 1);                 // diagonal
        CHECK(at(m, h, 6, plant[h]) == 1);          // the planted heavy block
        int n_sel = 0;
        for (int kb = 0; kb < NB; ++kb) n_sel += at(m, h, 6, kb);
        CHECK(n_sel == 3);                          // nothing else: the mass is covered by the plant
        CHECK(at(m, h, 6, 7) == 0);                 // right of the diagonal
    }
    CHECK(at(m, 0, 6, 4) == 0 && at(m, 1, 6, 2) == 0);   // each head selected ITS block, not the other head's
    // every row: nothing right of the diagonal, first column and diagonal set
    for (int h = 0; h < H; ++h) {
        for (int qb = 0; qb < NB; ++qb) {
            CHECK(at(m, h, qb, 0) == 1 && at(m, h, qb, qb) == 1);
            for (int kb = qb + 1; kb < NB; ++kb) CHECK(at(m, h, qb, kb) == 0);
        }
    }
    // threshold 1.0: the full causal triangle
    cfg.threshold = 1.0f;
    m = xattn_block_mask(Q.data(), K.data(), H, N, N, D, cfg);
    for (int h = 0; h < H; ++h)
        for (int qb = 0; qb < NB; ++qb)
            for (int kb = 0; kb < NB; ++kb) CHECK(at(m, h, qb, kb) == (kb <= qb ? 1 : 0));
    // a tiny threshold still keeps first + diagonal + the planted block (the first taken block has cumulative mass 0 < required)
    cfg.threshold = 0.05f;
    m = xattn_block_mask(Q.data(), K.data(), H, N, N, D, cfg);
    CHECK(at(m, 0, 6, 0) == 1 && at(m, 0, 6, 6) == 1 && at(m, 0, 6, 2) == 1);
    // unstructured data at 0.9: no pruning (the estimator is not a constant keep-few rule)
    {
        std::vector<float> Qr((size_t) H * N * D), Kr((size_t) H * N * D);
        fill(Qr); fill(Kr);
        cfg.threshold = 0.9f;
        const std::vector<uint8_t> mr = xattn_block_mask(Qr.data(), Kr.data(), H, N, N, D, cfg);
        int n_sel = 0;
        for (int kb = 0; kb < NB; ++kb) n_sel += at(mr, 0, 6, kb);
        CHECK(n_sel == 7);
    }
    // offset: the last 4 query blocks of 8 key blocks; local q block 2 is at key block 6; plant key block 1
    {
        const int NQ = 4 * B;
        std::vector<float> Qo((size_t) H * NQ * D), Ko((size_t) H * N * D);
        fill(Qo); fill(Ko);
        for (int h = 0; h < H; ++h) {
            for (int r = 0; r < B; ++r) {
                for (int d = 0; d < D; ++d) {
                    const float u = (d == 3) ? 6.0f : 0.0f;
                    Qo[((size_t) h * NQ + 2 * B + r) * D + d] = u;
                    Ko[((size_t) h * N + 1 * B + r) * D + d] = u;
                }
            }
        }
        cfg.threshold = 0.9f;
        const std::vector<uint8_t> mo = xattn_block_mask(Qo.data(), Ko.data(), H, NQ, N, D, cfg);
        CHECK(mo.size() == (size_t) H * 4 * NB);
        auto ato = [&](int h, int qb, int kb) { return (int) mo[((size_t) h * 4 + qb) * NB + kb]; };
        CHECK(ato(0, 2, 0) == 1 && ato(0, 2, 6) == 1 && ato(0, 2, 1) == 1);
        int n_sel = 0;
        for (int kb = 0; kb < NB; ++kb) n_sel += ato(0, 2, kb);
        CHECK(n_sel == 3);
        CHECK(ato(0, 2, 7) == 0);                   // right of this row's diagonal (key block 6)
        CHECK(ato(0, 3, 7) == 1);                   // the next query block's diagonal is key block 7
    }
    // single block: early return, one tile, selected
    {
        std::vector<float> q1((size_t) H * B * D), k1((size_t) H * B * D);
        fill(q1); fill(k1);
        const std::vector<uint8_t> m1 = xattn_block_mask(q1.data(), k1.data(), H, B, B, D, cfg);
        CHECK(m1.size() == (size_t) H && m1[0] == 1 && m1[1] == 1);
    }
    // shape errors
    {
        xattn_cfg bad = cfg;
        CHECK(xattn_block_mask(Q.data(), K.data(), H, N - 1, N, D, bad).empty());       // n_q not a multiple of the block
        bad.stride = 5;
        CHECK(xattn_block_mask(Q.data(), K.data(), H, N, N, D, bad).empty());            // stride does not divide the block
        bad = cfg; bad.threshold = 0.0f;
        CHECK(xattn_block_mask(Q.data(), K.data(), H, N, N, D, bad).empty());
        bad.threshold = 1.5f;
        CHECK(xattn_block_mask(Q.data(), K.data(), H, N, N, D, bad).empty());
        bad = cfg;
        CHECK(xattn_block_mask(Q.data(), K.data(), H, N, N - B, D, bad).empty());        // n_k < n_q
    }
    if (n_fail) {
        fprintf(stderr, "test-kv-xattn: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-kv-xattn: all passed\n");
    return 0;
}
