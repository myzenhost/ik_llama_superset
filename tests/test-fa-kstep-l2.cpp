// Pure-function test of the optional L2-derived CPU flash-attention K tile (ggml/src/iqk/fa/iqk_fa_kstep_l2.h).
// Row S69-NCNN-PICK-L2-CACHE-DERIVED-FA-TILE-2026-10-05-01.   GATE-CLAIM-OK: documentation only.
// Checks: only compiled tile values {32,64,128} come out; the tile never grows when L2 shrinks, or when head / element size grow;
// ncnn/ORT worked numbers; unknown L2 keeps today's 128; the detector returns a plausible size or 0 and honours IQK_FA_L2_KIB.
#include "iqk/fa/iqk_fa_kstep_l2.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static bool compiled(int s) { return s == 32 || s == 64 || s == 128; }

int main() {
    const size_t l2s[]  = { 256u << 10, 512u << 10, 1u << 20, 2u << 20 };
    const double els[]  = { 2.0, 1.0625, 0.5 };
    const int    heads[] = { 64, 96, 128, 192, 256, 512 };
    for (size_t l2 : l2s) for (double e : els) for (int h : heads) CHECK(compiled(iqk_fa_kstep_from_l2(l2, h, e)));

    // worked numbers: 512 KiB / (4*2*2*128) = 256 -> 128 ; 256 KiB, head 256, f16: 128 -> 128 ; 128 KiB, head 256, f16: 64 -> 64
    CHECK(iqk_fa_kstep_from_l2(512u << 10, 128, 2.0) == 128);
    CHECK(iqk_fa_kstep_from_l2(256u << 10, 256, 2.0) == 64);
    CHECK(iqk_fa_kstep_from_l2(128u << 10, 256, 2.0) == 32);
    CHECK(iqk_fa_kstep_from_l2(512u << 10, 512, 4.0) == 32);   // ORT fp32 formula, head 512: 512K/(4*4*2*512) = 32
    CHECK(iqk_fa_kstep_from_l2(512u << 10, 128, 4.0) == 128);  // ORT fp32 formula, head 128: 128

    // monotone: more L2 never shrinks the tile; bigger head or element never grows it
    for (double e : els) for (int h : heads) {
        int prev = 0;
        for (size_t l2 : l2s) { const int s = iqk_fa_kstep_from_l2(l2, h, e); CHECK(s >= prev); prev = s; }
    }
    for (size_t l2 : l2s) for (double e : els) {
        int prev = 1 << 20;
        for (int h : heads) { const int s = iqk_fa_kstep_from_l2(l2, h, e); CHECK(s <= prev); prev = s; }
    }

    // unknown / degenerate input = today's 128
    CHECK(iqk_fa_kstep_from_l2(0, 128, 2.0) == 128);
    CHECK(iqk_fa_kstep_from_l2(512u << 10, 0, 2.0) == 128);
    CHECK(iqk_fa_kstep_from_l2(512u << 10, 128, 0.0) == 128);

    // detector: override wins, otherwise 0 or a sane size (16 KiB .. 64 MiB)
#if defined(_WIN32)
    _putenv("IQK_FA_L2_KIB=384");
#else
    setenv("IQK_FA_L2_KIB", "384", 1);
#endif
    CHECK(iqk_fa_l2_bytes_detect() == (size_t)384 << 10);
#if defined(_WIN32)
    _putenv("IQK_FA_L2_KIB=");
#else
    unsetenv("IQK_FA_L2_KIB");
#endif
    const size_t d = iqk_fa_l2_bytes_detect();
    printf("detected L2 = %zu KiB\n", d >> 10);
    CHECK(d == 0 || (d >= (16u << 10) && d <= (64u << 20)));

    // default OFF: without IQK_FA_KSTEP_L2 the bound is 128 for every head
    for (int h : heads) CHECK(iqk_fa_kstep_l2_bound(h) == 128);

    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-fa-kstep-l2: all checks passed\n");
    return 0;
}
