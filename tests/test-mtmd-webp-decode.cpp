// webp images must decode in-process (simplewebp), with NO ffmpeg available.
//
// SOURCE-READ: tmp/research/webp_1110/xwebp_decode.go (golang.org/x/image/webp decode.go, read in full)
// SOURCE-NUANCE: x/image/webp rejects (errInvalidFormat) a VP8X chunk whose length != 10 (:157), a VP8/VP8L frame whose
//   dims differ from the VP8X canvas (:86, :142), canvas w*h > MaxInt32 (:175), and an ALPH chunk when VP8X had no alpha
//   flag (:59); an animated file has no top-level VP8/VP8L chunk so it ends in EOF -> invalid format (:50-52), i.e. no
//   first-frame decode. The checks below mirror those (canvas mismatch + animated) in the fork's helper.
//
// Ollama decodes webp in-process; the fork used to shell out to ffmpeg under MTMD_VIDEO, so on a host without ffmpeg
// every webp image failed. This test goes through the public entry point, mtmd_helper_bitmap_init_from_buf (images need
// no model/context), with ffmpeg_bin_dir pointed at a directory that does not exist so the ffmpeg fallback can never
// succeed regardless of the machine's PATH.
//
// Fixtures (tests/webp-fixtures, made by Pillow from a known 64x48 pattern; *.rgb = Pillow/libwebp decode, packed RGB):
//   lossy.webp (VP8), lossless.webp (VP8L), lossy_alpha.webp (VP8X + ALPH + VP8), animated.webp (VP8X animated)
//
// usage: test-mtmd-webp-decode <fixture dir>

#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); g_fail++; } } while (0)

static std::string g_dir;

static std::vector<unsigned char> read_file(const std::string & name) {
    std::vector<unsigned char> out;
    FILE * f = std::fopen((g_dir + "/" + name).c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s/%s\n", g_dir.c_str(), name.c_str()); return out; }
    unsigned char tmp[4096];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) out.insert(out.end(), tmp, tmp + n);
    std::fclose(f);
    return out;
}

static mtmd_helper_init_opt no_ffmpeg_opt() {
    mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    opt.video_params.ffmpeg_bin_dir = "Z:/__no_such_dir_for_ffmpeg__";
    return opt;
}

static mtmd_bitmap * decode(const std::vector<unsigned char> & buf, bool placeholder) {
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(nullptr, buf.data(), buf.size(), placeholder, no_ffmpeg_opt());
    return w.bitmap;
}

static int compare(const mtmd_bitmap * bm, const std::vector<unsigned char> & ref, double & mean) {
    const size_t n = mtmd_bitmap_get_n_bytes(bm);
    const unsigned char * d = mtmd_bitmap_get_data(bm);
    int mx = 0;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        int diff = std::abs((int) d[i] - (int) ref[i]);
        if (diff > mx) mx = diff;
        sum += diff;
    }
    mean = n ? sum / (double) n : 0;
    return mx;
}

static void check_image(const char * name, int max_tol, double mean_tol) {
    auto webp = read_file(std::string(name) + ".webp");
    auto ref  = read_file(std::string(name) + ".rgb");
    CHECK(!webp.empty() && ref.size() == 64 * 48 * 3, "%s: fixture missing", name);
    if (webp.empty() || ref.size() != 64 * 48 * 3) return;

    mtmd_bitmap * bm = decode(webp, false);
    CHECK(bm != nullptr, "%s: decode returned null (no in-process webp decoder / ffmpeg absent)", name);
    if (!bm) return;
    CHECK(mtmd_bitmap_get_nx(bm) == 64 && mtmd_bitmap_get_ny(bm) == 48, "%s: dims %ux%u", name, mtmd_bitmap_get_nx(bm), mtmd_bitmap_get_ny(bm));
    CHECK(mtmd_bitmap_get_n_bytes(bm) == ref.size(), "%s: n_bytes %zu", name, mtmd_bitmap_get_n_bytes(bm));
    if (mtmd_bitmap_get_n_bytes(bm) == ref.size()) {
        double mean = 0;
        int mx = compare(bm, ref, mean);
        std::printf("%-12s max_abs_diff=%d mean_abs_diff=%.4f (tol %d / %.2f)\n", name, mx, mean, max_tol, mean_tol);
        CHECK(mx <= max_tol, "%s: max abs diff %d > %d", name, mx, max_tol);
        CHECK(mean <= mean_tol, "%s: mean abs diff %.4f > %.2f", name, mean, mean_tol);
    }
    const char * id = mtmd_bitmap_get_id(bm);
    CHECK(id && std::strlen(id) == 64, "%s: sha256 id missing", name);
    mtmd_bitmap_free(bm);

    mtmd_bitmap * ph = decode(webp, true); // placeholder: dims only
    CHECK(ph != nullptr, "%s: placeholder decode returned null", name);
    if (ph) {
        CHECK(mtmd_bitmap_get_nx(ph) == 64 && mtmd_bitmap_get_ny(ph) == 48, "%s: placeholder dims", name);
        mtmd_bitmap_free(ph);
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <fixture dir>\n", argv[0]); return 2; }
    g_dir = argv[1];

    check_image("lossless",    0, 0.0);   // VP8L is bit-exact
    check_image("lossy",       8, 1.5);   // VP8 YUV->RGB rounding may differ a little between decoders
    check_image("lossy_alpha", 8, 1.5);   // VP8X + ALPH + VP8; alpha plane dropped, as stb does for png (3 channels)

    { // animated webp is not a still image: no ffmpeg here -> must fail cleanly (null), never crash
        auto webp = read_file("animated.webp");
        CHECK(!webp.empty(), "animated fixture missing");
        mtmd_bitmap * bm = decode(webp, false);
        CHECK(bm == nullptr, "animated webp decoded without ffmpeg");
        if (bm) mtmd_bitmap_free(bm);
    }
    { // truncated payload must be rejected
        auto webp = read_file("lossy.webp");
        if (webp.size() > 40) webp.resize(40);
        mtmd_bitmap * bm = decode(webp, false);
        CHECK(bm == nullptr, "truncated webp decoded");
        if (bm) mtmd_bitmap_free(bm);
    }
    { // VP8X canvas != frame dims is an invalid file: patch canvas width-1 (24-bit LE at byte 24) 63 -> 31
        auto webp = read_file("lossy_alpha.webp");
        CHECK(webp.size() > 30 && std::memcmp(&webp[12], "VP8X", 4) == 0, "alpha fixture is not VP8X");
        if (webp.size() > 30) {
            webp[24] = 31;
            mtmd_bitmap * bm = decode(webp, false);
            CHECK(bm == nullptr, "canvas/frame mismatch decoded");
            if (bm) mtmd_bitmap_free(bm);
        }
    }

    if (g_fail) { std::printf("test-mtmd-webp-decode: %d FAILED\n", g_fail); return 1; }
    std::printf("test-mtmd-webp-decode: OK\n");
    return 0;
}
