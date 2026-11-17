// test-cram-soft: correctness + controls + informational micro-benchmark for llama_cram_soft_store.
#include "llama-cram-soft.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2h(float f) { // round-to-nearest-even float -> IEEE half (normals/subnormals/zero; no NaN handling needed)
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e >= 31) return (uint16_t)(sign | 0x7bff);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000; int sh = 14 - e; uint32_t r = m >> sh; uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13); uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
    return (uint16_t)(sign | r);
}

static std::vector<uint8_t> gen_random(size_t n, unsigned seed) {
    std::mt19937_64 g(seed); std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (uint8_t) g();
    return v;
}
static std::vector<uint8_t> gen_zero_heavy(size_t n, unsigned seed) { // ~98% zeros in runs, sprinkled noise
    std::mt19937_64 g(seed); std::vector<uint8_t> v(n, 0);
    for (size_t i = 0; i < n / 64; i++) v[g() % n] = (uint8_t) g();
    return v;
}
static std::vector<uint8_t> gen_f16_gauss(size_t n, unsigned seed) { // n bytes
    std::mt19937_64 g(seed); std::normal_distribution<float> d(0.f, 0.02f);
    std::vector<uint8_t> v(n & ~size_t(1));
    for (size_t i = 0; i < v.size(); i += 2) { uint16_t h = f2h(d(g)); std::memcpy(&v[i], &h, 2); }
    return v;
}
static std::vector<uint8_t> gen_ternary_f16(size_t n, unsigned seed) { // {-s,0,+s} stored as f16
    std::mt19937_64 g(seed); std::vector<uint8_t> v(n & ~size_t(1));
    const uint16_t s = f2h(0.0173f);
    for (size_t i = 0; i < v.size(); i += 2) { int t = (int)(g() % 3); uint16_t h = t == 0 ? 0 : t == 1 ? s : (uint16_t)(s | 0x8000); std::memcpy(&v[i], &h, 2); }
    return v;
}

struct case_t { const char * name; std::vector<uint8_t> data; int stride; };

static bool roundtrip(const char * name, const std::vector<uint8_t> & d, llama_cram_soft_params p, double * ratio_out = nullptr) {
    llama_cram_soft_store s(d.data(), d.size(), p);
    std::vector<uint8_t> out(d.size());
    size_t got = s.read(0, d.size(), out.data());
    bool ok = got == d.size() && out == d;
    auto st = s.stats();
    double ratio = (double) st.raw_bytes / (double)(st.compressed_bytes ? st.compressed_bytes : 1);
    std::printf("  %-14s stride=%d raw=%llu comp=%llu meta=%llu ratio=%.3fx raw_blocks=%llu/%llu roundtrip=%s\n", name, p.stride,
                (unsigned long long) st.raw_bytes, (unsigned long long) st.compressed_bytes, (unsigned long long) st.meta_bytes, ratio,
                (unsigned long long) st.raw_blocks, (unsigned long long) st.blocks, ok ? "OK" : "MISMATCH");
    if (ratio_out) *ratio_out = ratio;
    return ok;
}

int main() {
    const size_t N = 4 * 1024 * 1024 + 12345; // not a multiple of 64 KiB: exercises the short tail block

    std::printf("== (1) round trips and ratios (64 KiB blocks, codec PLANE_HUF) ==\n");
    {
        auto rnd = gen_random(N, 1);
        llama_cram_soft_params p; p.hot_blocks = 4;
        double r;
        CHECK(roundtrip("random", rnd, p, &r));
        {
            llama_cram_soft_store s(rnd.data(), rnd.size(), p); auto st = s.stats();
            CHECK(st.raw_blocks == st.blocks);                              // all fell back to RAW
            CHECK(st.compressed_bytes == st.raw_bytes);                      // payload exactly raw
            CHECK(st.compressed_bytes + st.meta_bytes <= st.raw_bytes + st.blocks * 32); // raw + small header
        }
        auto z = gen_zero_heavy(N, 2);
        CHECK(roundtrip("zero-heavy", z, p, &r)); CHECK(r > 10.0);
        for (int stride = 1; stride <= 2; stride++) {
            p.stride = stride;
            auto f = gen_f16_gauss(N, 3);
            CHECK(roundtrip("f16-gauss", f, p, &r)); if (stride == 2) CHECK(r > 1.05);
            auto t = gen_ternary_f16(N, 4);
            CHECK(roundtrip("ternary-f16", t, p, &r)); if (stride == 2) CHECK(r > 3.0);
        }
        p.stride = 1; p.codec = LLAMA_CRAM_CODEC_RAW;
        CHECK(roundtrip("random/RAWcodec", rnd, p));
        p.codec = LLAMA_CRAM_CODEC_PLANE_RLE; p.stride = 2;
        CHECK(roundtrip("zero/RLE", z, p));
        // tiny and empty inputs
        std::vector<uint8_t> tiny = {1, 2, 3};
        p.codec = LLAMA_CRAM_CODEC_PLANE_HUF; CHECK(roundtrip("tiny", tiny, p));
        std::vector<uint8_t> empty; CHECK(roundtrip("empty", empty, p));
    }

    std::printf("== (2) reads straddling block boundaries ==\n");
    {
        const size_t B = 4096;
        auto d = gen_f16_gauss(B * 9 + 777, 5);
        llama_cram_soft_params p; p.block_size = B; p.hot_blocks = 3; p.stride = 2;
        llama_cram_soft_store s(d.data(), d.size(), p);
        std::vector<uint8_t> buf(3 * B);
        struct { size_t off, len; } cases[] = {{B - 1, 2}, {B - 1, B + 2}, {B, B}, {2 * B - 5, 2 * B + 10}, {0, 1}, {d.size() - 1, 1}, {d.size() - 10, 100}, {B * 9, 777}};
        bool ok = true;
        for (auto & c : cases) {
            size_t want = c.off >= d.size() ? 0 : std::min(c.len, d.size() - c.off);
            size_t got = s.read(c.off, c.len, buf.data());
            ok = ok && got == want && std::memcmp(buf.data(), d.data() + c.off, want) == 0;
        }
        CHECK(ok);
        CHECK(s.read(d.size(), 10, buf.data()) == 0);
        std::mt19937_64 g(9);
        for (int i = 0; i < 2000; i++) {
            size_t off = g() % d.size(), len = 1 + g() % (3 * B);
            size_t want = std::min(len, d.size() - off);
            size_t got = s.read(off, len, buf.data());
            CHECK(got == want && std::memcmp(buf.data(), d.data() + off, want) == 0);
        }
        std::printf("  boundary + 2000 random range reads: %s\n", g_fail ? "FAIL" : "OK");
    }

    std::printf("== (3) LRU eviction: window 2 over 10 blocks ==\n");
    {
        const size_t B = 1024;
        auto d = gen_zero_heavy(B * 10, 6);
        llama_cram_soft_params p; p.block_size = B; p.hot_blocks = 2; p.shards = 1;
        llama_cram_soft_store s(d.data(), d.size(), p);
        std::vector<uint8_t> b(B);
        auto rd = [&](size_t blk) { s.read(blk * B, B, b.data()); };
        rd(0); rd(1);                       // miss miss      mru: 1 0
        CHECK((s.hot_blocks_mru_first() == std::vector<size_t>{1, 0}));
        rd(0);                              // hit            mru: 0 1
        rd(2);                              // miss, evicts 1 mru: 2 0
        CHECK((s.hot_blocks_mru_first() == std::vector<size_t>{2, 0}));
        rd(0);                              // hit            mru: 0 2
        rd(1);                              // miss, evicts 2 mru: 1 0
        CHECK((s.hot_blocks_mru_first() == std::vector<size_t>{1, 0}));
        auto st = s.stats();
        std::printf("  hits=%llu misses=%llu (expected 2 / 4)\n", (unsigned long long) st.hits, (unsigned long long) st.misses);
        CHECK(st.hits == 2 && st.misses == 4);
        for (size_t i = 0; i < 10; i++) rd(i);
        CHECK(s.hot_blocks_mru_first().size() == 2);   // bounded
    }

    std::printf("== (4) 8 threads vs raw buffer ==\n");
    {
        const size_t B = 16384;
        auto d = gen_f16_gauss(B * 64 + 99, 7);
        llama_cram_soft_params p; p.block_size = B; p.hot_blocks = 8; p.shards = 4; p.stride = 2;
        llama_cram_soft_store s(d.data(), d.size(), p);
        std::atomic<int> bad{0}, exc{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 8; t++) th.emplace_back([&, t] {
            std::mt19937_64 g(100 + t); std::vector<uint8_t> buf(3 * B);
            for (int i = 0; i < 3000; i++) {
                size_t off = g() % d.size(), len = 1 + g() % (3 * B);
                size_t want = std::min(len, d.size() - off);
                try {
                    if (s.read(off, len, buf.data()) != want || std::memcmp(buf.data(), d.data() + off, want) != 0) bad++;
                } catch (...) { exc++; }
            }
        });
        for (auto & t : th) t.join();
        auto st = s.stats();
        std::printf("  mismatches=%d exceptions=%d hits=%llu misses=%llu\n", bad.load(), exc.load(), (unsigned long long) st.hits, (unsigned long long) st.misses);
        CHECK(bad == 0 && exc == 0);
    }

    std::printf("== (5) CONTROL: corrupt one stored byte ==\n");
    {
        const size_t B = 8192;
        auto d = gen_zero_heavy(B * 4, 8);           // blocks 0..3 compress well
        auto r = gen_random(B, 10); d.insert(d.end(), r.begin(), r.end()); // block 4 is RAW (incompressible)
        for (int use_raw_blk = 0; use_raw_blk < 2; use_raw_blk++) {
            size_t blk = use_raw_blk ? 4 : 1;
            for (int verify = 1; verify >= 0; verify--) {
                llama_cram_soft_params p; p.block_size = B; p.hot_blocks = 2; p.verify = verify;
                llama_cram_soft_store s(d.data(), d.size(), p);
                size_t clen; uint8_t * sb = s.debug_stored_bytes_of_block(blk, &clen);
                sb[clen / 2] ^= 0x5a;
                std::vector<uint8_t> out(B);
                const char * res;
                try {
                    s.read(blk * B, B, out.data());
                    res = std::memcmp(out.data(), d.data() + blk * B, B) == 0 ? "no error, data still equal" : "NOT DETECTED: returned WRONG data silently";
                    if (verify) { CHECK(false); } // with CRC on, a read of a corrupted block must not succeed
                } catch (const std::exception & e) { res = "DETECTED (exception)"; (void) e; }
                std::printf("  block %zu (%s) verify=%d -> %s\n", blk, use_raw_blk ? "RAW" : "compressed", verify, res);
                if (verify) CHECK(std::string(res) == "DETECTED (exception)");
            }
        }
    }

    std::printf("== micro benchmark (informational, not asserted) ==\n");
    {
        const size_t SZ = 64u << 20;
        struct { const char * name; std::vector<uint8_t> d; int stride; } bs[] = {
            {"f16-gauss", gen_f16_gauss(SZ, 11), 2}, {"ternary-f16", gen_ternary_f16(SZ, 12), 2}, {"zero-heavy", gen_zero_heavy(SZ, 13), 1}};
        std::vector<uint8_t> dst(SZ);
        { double t0 = now_s(); for (int i = 0; i < 5; i++) std::memcpy(dst.data(), bs[0].d.data(), SZ); double t = now_s() - t0;
          std::printf("  plain memcpy 64MiB x5: %.0f MB/s\n", 5.0 * SZ / 1e6 / t); }
        for (auto & b : bs) {
            llama_cram_soft_params p; p.stride = b.stride; p.hot_blocks = (SZ / p.block_size) + 1;
            double tb = now_s(); llama_cram_soft_store s(b.d.data(), b.d.size(), p); double build = now_s() - tb;
            auto st0 = s.stats();
            double t0 = now_s(); s.read(0, SZ, dst.data()); double cold = now_s() - t0;   // all misses
            auto st1 = s.stats();
            bool ok = std::memcmp(dst.data(), b.d.data(), SZ) == 0;
            t0 = now_s(); for (int i = 0; i < 5; i++) s.read(0, SZ, dst.data()); double hot = (now_s() - t0) / 5; // all hits
            std::printf("  %-12s ratio=%.3fx build=%.0f MB/s cold-decode=%.0f MB/s (decode-only %.0f MB/s) hot-window=%.0f MB/s ok=%d\n", b.name,
                        (double) st0.raw_bytes / st0.compressed_bytes, SZ / 1e6 / build, SZ / 1e6 / cold,
                        SZ / 1e6 / (st1.decode_ns / 1e9), SZ / 1e6 / hot, (int) ok);
        }
        // cold with codec RAW for reference (CRC + memcpy only)
        llama_cram_soft_params p; p.codec = LLAMA_CRAM_CODEC_RAW; p.hot_blocks = 1;
        llama_cram_soft_store s(bs[0].d.data(), SZ, p);
        double t0 = now_s(); s.read(0, SZ, dst.data()); double t = now_s() - t0;
        std::printf("  RAW codec cold read (CRC32 + copy, no decode): %.0f MB/s\n", SZ / 1e6 / t);
    }

    std::printf(g_fail ? "test-cram-soft: FAILED (%d)\n" : "test-cram-soft: PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
