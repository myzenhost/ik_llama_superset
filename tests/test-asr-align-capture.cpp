// Model-free test of examples/audio/asr-align-capture.{h,cpp}: the attention-row capture behind the server's word timestamps.
// A fake kq_soft_max tensor (a float array with ggml-style strides) is fed to on_softmax; expected values are computed by hand.
#include "asr-align-capture.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

// tensor [n_kv, n_q, n_head, 1] of F32, value = 1000*h + 10*q + k so every cell is identifiable
struct fake {
    int64_t n_kv, n_q, n_h;
    std::vector<float> d;
    fake(int64_t kv, int64_t q, int64_t h, float base = 0.0f) : n_kv(kv), n_q(q), n_h(h), d((size_t) (kv * q * h)) {
        for (int64_t hh = 0; hh < h; ++hh) for (int64_t qq = 0; qq < q; ++qq) for (int64_t k = 0; k < kv; ++k) {
            d[(size_t) (hh * q * kv + qq * kv + k)] = base + 1000.0f * hh + 10.0f * qq + k;
        }
    }
    bool feed(asr_attn_capture & c, int il) const {
        const int64_t ne[4] = { n_kv, n_q, n_h, 1 };
        const size_t  nb[4] = { sizeof(float), (size_t) n_kv * sizeof(float), (size_t) (n_kv * n_q) * sizeof(float), (size_t) (n_kv * n_q * n_h) * sizeof(float) };
        const char * p = (const char *) d.data();
        return c.on_softmax(il, true, "f32", ne, nb, [p](void * dst, size_t off, size_t n) { std::memcpy(dst, p + off, n); });
    }
};

int main() {
    CHECK(asr_attn_layer_of("kq_soft_max-12") == 12);
    CHECK(asr_attn_layer_of("kq_soft_max-0") == 0);
    CHECK(asr_attn_layer_of("kq_soft_max") == -1);
    CHECK(asr_attn_layer_of("kq_soft_max-x") == -1);
    CHECK(asr_attn_layer_of(nullptr) == -1);

    {   // arm validity
        asr_attn_capture c; c.l0 = 0; c.l1 = 1; c.n_head = 2;
        CHECK(!c.arm(5, 5));
        CHECK(!c.arm(6, 5));
        c.n_head = 0;
        CHECK(!c.arm(0, 3));
        c.n_head = 2; c.l1 = -1;
        CHECK(!c.arm(0, 3));   // no layers
    }

    {   // last-row mode: row 0 is the LAST query row, cells [2,5) only, layers 0..1
        asr_attn_capture c; c.l0 = 0; c.l1 = 1; c.n_head = 2;
        CHECK(c.arm(2, 5));
        CHECK(c.n_sel() == 2 && c.n_audio() == 3);
        fake t(8, 4, 2);
        CHECK(t.feed(c, 0));
        std::vector<float> r0;
        CHECK(!c.take_row0_and_begin_all_rows(r0));   // layer 1 never seen
        CHECK(c.arm(2, 5));
        t.feed(c, 0); t.feed(c, 1);
        CHECK(c.take_row0_and_begin_all_rows(r0));
        CHECK(r0.size() == 2 * 2 * 3);
        // layer 0, head 1, last query row (q=3), cells 2..4 -> 1000*1+30+k
        CHECK(r0[(0 * 2 + 1) * 3 + 0] == 1032.0f && r0[(0 * 2 + 1) * 3 + 2] == 1034.0f);
        CHECK(r0[(1 * 2 + 0) * 3 + 1] == 33.0f);
    }

    {   // a batch evaluated before the audio cells are in the cache is ignored; a wrong head count is an error
        asr_attn_capture c; c.l0 = 0; c.l1 = 0; c.n_head = 2;
        CHECK(c.arm(2, 5));
        fake early(4, 1, 2);   // n_kv 4 < kv1 5
        early.feed(c, 0);
        CHECK(c.err.empty());
        fake bad(8, 1, 3);
        bad.feed(c, 0);
        CHECK(!c.err.empty());
    }

    {   // all-rows mode: two batches (3 rows then 2 rows) arrive in order, 5 rows total
        asr_attn_capture c; c.l0 = 0; c.l1 = 0; c.n_head = 2;
        CHECK(c.arm(1, 3));
        fake p(6, 4, 2);
        p.feed(c, 0);
        std::vector<float> r0;
        CHECK(c.take_row0_and_begin_all_rows(r0));
        fake b1(9, 3, 2, 0.0f), b2(9, 2, 2, 500.0f);
        b1.feed(c, 0); b2.feed(c, 0);
        std::vector<float> rows;
        CHECK(!c.take_all_rows(4, rows));   // 5 delivered, 4 expected: reported
        CHECK(!c.err.empty());
    }
    {
        asr_attn_capture c; c.l0 = 0; c.l1 = 0; c.n_head = 2;
        CHECK(c.arm(1, 3));
        fake p(6, 4, 2);
        p.feed(c, 0);
        std::vector<float> r0;
        CHECK(c.take_row0_and_begin_all_rows(r0));
        fake b1(9, 3, 2, 0.0f), b2(9, 2, 2, 500.0f);
        b1.feed(c, 0); b2.feed(c, 0);
        std::vector<float> rows;
        CHECK(c.take_all_rows(5, rows));
        CHECK(rows.size() == 5 * 1 * 2 * 2);
        // row 3 = first row of batch 2 (q=0), head 1, cells 1..2 -> 500+1000+1, +2
        CHECK(rows[(3 * 2 + 1) * 2 + 0] == 1501.0f && rows[(3 * 2 + 1) * 2 + 1] == 1502.0f);
        // row 1 = q=1 of batch 1, head 0 -> 10+1
        CHECK(rows[(1 * 2 + 0) * 2 + 0] == 11.0f);
    }

    {   // assemble: row0 + tok rows, padding cut, with and without the end row
        const int n_hd = 2, n_pos = 4, n_frames = 3, n_tok = 2;
        std::vector<float> r0(n_hd * n_pos), tr((size_t) n_tok * n_hd * n_pos);
        for (size_t i = 0; i < r0.size(); ++i) r0[i] = (float) i;
        for (size_t i = 0; i < tr.size(); ++i) tr[i] = 100.0f + (float) i;
        auto a = asr_attn_assemble(r0, tr, n_tok, true, n_hd, n_pos, n_frames);
        CHECK(a.size() == (size_t) 3 * n_hd * n_frames);
        CHECK(a[0] == 0.0f && a[2] == 2.0f && a[3] == 4.0f);           // row 0, head 1 starts at r0[4]
        CHECK(a[(1 * n_hd + 0) * n_frames] == 100.0f);                 // row 1 = first forced row
        CHECK(a[(2 * n_hd + 1) * n_frames + 2] == 100.0f + 8 + 4 + 2); // row 2, head 1, frame 2
        auto b = asr_attn_assemble(r0, tr, n_tok, false, n_hd, n_pos, n_frames);
        CHECK(b.size() == (size_t) 2 * n_hd * n_frames);
        CHECK(asr_attn_assemble(r0, tr, n_tok, true, n_hd, n_pos, 5).empty());      // more frames than positions
        CHECK(asr_attn_assemble(r0, tr, 3, true, n_hd, n_pos, n_frames).empty());   // wrong row count
    }

    std::printf(fails ? "FAILED %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
