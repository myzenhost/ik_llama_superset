// Still images are refused BEFORE decode when their header declares more pixels than the decode limit
// (LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS, default 178956970).
// Row S69-ENGREV-MLXENGINE-IMAGE-PIXEL-CAP-2026-10-27-01.
//
// SOURCE-READ: lmstudio-ai/mlx-engine 86f20eb4 mlx_engine/server/chat.py:107-118 (pixel count from the header checked
// against a cap before the pixels are materialised), :163-165 (same cap on the running total of one request); MIT,
// re-specified, no code copied. The fork's helper is mtmd_helper_bitmap_init_from_buf (tools/mtmd/mtmd-helper.cpp).
// RESEARCH-DONE: https://github.com/lmstudio-ai/mlx-engine/blob/86f20eb4/mlx_engine/server/chat.py (read 2026-11-13)
// DISABLE-REVERSIBLE: LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS=0 is the user's own way to lift the limit (it is ON by default);
// this test checks that the value 0 really lifts it, it does not turn anything off in the product.
//
// The limit is lowered through the environment so a 64x48 image is "too large"; no gigabyte image is needed to see a
// refusal. BEFORE the change the variable did nothing, so every "refused" check below failed (the image decoded).
// Both directions each time: the same image decodes one pixel under the line and is refused one pixel over it.
//
// usage: test-mtmd-image-pixel-cap <webp fixture dir>

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

static void set_cap(const char * value) { // nullptr = unset
#ifdef _WIN32
    _putenv_s("LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS", value ? value : "");
#else
    if (value) setenv("LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS", value, 1); else unsetenv("LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS");
#endif
}

static void put_le(std::vector<unsigned char> & v, uint32_t x, int n) {
    for (int i = 0; i < n; i++) v.push_back((unsigned char) ((x >> (8 * i)) & 0xff));
}

static void put_be32(std::vector<unsigned char> & v, uint32_t x) {
    for (int i = 3; i >= 0; i--) v.push_back((unsigned char) ((x >> (8 * i)) & 0xff));
}

// a complete, decodable 24-bit BMP (w x h, every pixel the same colour)
static std::vector<unsigned char> make_bmp(int w, int h) {
    const uint32_t row = (uint32_t) ((w * 3 + 3) & ~3);
    const uint32_t img = row * (uint32_t) h;
    std::vector<unsigned char> v;
    v.push_back('B'); v.push_back('M');
    put_le(v, 54 + img, 4); put_le(v, 0, 4); put_le(v, 54, 4);         // file header
    put_le(v, 40, 4); put_le(v, (uint32_t) w, 4); put_le(v, (uint32_t) h, 4);
    put_le(v, 1, 2); put_le(v, 24, 2); put_le(v, 0, 4); put_le(v, img, 4);
    put_le(v, 2835, 4); put_le(v, 2835, 4); put_le(v, 0, 4); put_le(v, 0, 4);
    v.resize(54 + img, 0x40);
    return v;
}

// a PNG that is only a signature and an IHDR chunk: enough for a header read, nothing to decode. 1-bit greyscale.
static std::vector<unsigned char> make_png_header_only(uint32_t w, uint32_t h) {
    std::vector<unsigned char> v = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    put_be32(v, 13);
    v.push_back('I'); v.push_back('H'); v.push_back('D'); v.push_back('R');
    put_be32(v, w); put_be32(v, h);
    v.push_back(1); v.push_back(0); v.push_back(0); v.push_back(0); v.push_back(0); // depth 1, grey, deflate, filter 0, no interlace
    put_be32(v, 0); // CRC (stb does not check it)
    return v;
}

static std::vector<unsigned char> read_file(const std::string & path) {
    std::vector<unsigned char> out;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    unsigned char tmp[4096];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) out.insert(out.end(), tmp, tmp + n);
    std::fclose(f);
    return out;
}

// decodes -> true; frees the bitmap
static bool decodes(const std::vector<unsigned char> & buf, bool placeholder, int want_w, int want_h) {
    mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    opt.video_params.ffmpeg_bin_dir = "Z:/__no_such_dir_for_ffmpeg__";
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(nullptr, buf.data(), buf.size(), placeholder, opt);
    if (!w.bitmap) return false;
    const bool dims_ok = (int) mtmd_bitmap_get_nx(w.bitmap) == want_w && (int) mtmd_bitmap_get_ny(w.bitmap) == want_h;
    CHECK(dims_ok, "decoded dims %ux%u, expected %dx%d", mtmd_bitmap_get_nx(w.bitmap), mtmd_bitmap_get_ny(w.bitmap), want_w, want_h);
    mtmd_bitmap_free(w.bitmap);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <webp fixture dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const std::vector<unsigned char> bmp  = make_bmp(64, 48);                     // 3072 pixels
    const std::vector<unsigned char> webp = read_file(dir + "/lossless.webp");     // 64 x 48 as well
    CHECK(!webp.empty(), "fixture %s/lossless.webp missing", dir.c_str());

    // the limit itself: default, override, the value 0, and values that are not a plain non-negative integer
    set_cap(nullptr);
    CHECK(mtmd_helper_image_max_decode_pixels() == 178956970ull, "default limit is %llu", (unsigned long long) mtmd_helper_image_max_decode_pixels());
    set_cap("1000");
    CHECK(mtmd_helper_image_max_decode_pixels() == 1000ull, "limit 1000 read as %llu", (unsigned long long) mtmd_helper_image_max_decode_pixels());
    set_cap("0");
    CHECK(mtmd_helper_image_max_decode_pixels() == 0ull, "limit 0 read as %llu", (unsigned long long) mtmd_helper_image_max_decode_pixels());
    for (const char * bad : {"abc", "-5", "12x", " "}) {
        set_cap(bad);
        CHECK(mtmd_helper_image_max_decode_pixels() == 178956970ull, "bad value '%s' changed the limit to %llu", bad,
              (unsigned long long) mtmd_helper_image_max_decode_pixels());
    }

    // the header read: real images report their size, non-images report 0
    CHECK(mtmd_helper_image_declared_pixels(bmp.data(), bmp.size()) == 3072ull, "bmp declares %llu",
          (unsigned long long) mtmd_helper_image_declared_pixels(bmp.data(), bmp.size()));
    if (!webp.empty()) {
        CHECK(mtmd_helper_image_declared_pixels(webp.data(), webp.size()) == 3072ull, "webp declares %llu",
              (unsigned long long) mtmd_helper_image_declared_pixels(webp.data(), webp.size()));
    }
    const std::vector<unsigned char> junk(256, 0x5a);
    CHECK(mtmd_helper_image_declared_pixels(junk.data(), junk.size()) == 0ull, "junk declares pixels");
    CHECK(mtmd_helper_image_declared_pixels(nullptr, 0) == 0ull, "null buffer declares pixels");

    // control: under the default limit both formats decode
    set_cap(nullptr);
    CHECK(decodes(bmp, false, 64, 48), "bmp did not decode under the default limit");
    if (!webp.empty()) CHECK(decodes(webp, false, 64, 48), "webp did not decode under the default limit");

    // the line itself: 3072 pixels allowed at a limit of 3072, refused at 3071 (pixels and placeholder alike)
    set_cap("3072");
    CHECK(decodes(bmp, false, 64, 48), "bmp refused AT the limit (3072 of 3072)");
    if (!webp.empty()) CHECK(decodes(webp, false, 64, 48), "webp refused AT the limit (3072 of 3072)");
    set_cap("3071");
    CHECK(!decodes(bmp, false, 64, 48), "bmp of 3072 pixels decoded past a 3071 pixel limit");
    CHECK(!decodes(bmp, true, 64, 48), "bmp placeholder of 3072 pixels decoded past a 3071 pixel limit");
    if (!webp.empty()) {
        CHECK(!decodes(webp, false, 64, 48), "webp of 3072 pixels decoded past a 3071 pixel limit");
        CHECK(!decodes(webp, true, 64, 48), "webp placeholder of 3072 pixels decoded past a 3071 pixel limit");
    }

    // the value 0 lifts the limit: the same image decodes again
    set_cap("0");
    CHECK(decodes(bmp, false, 64, 48), "bmp refused although the limit value is 0");

    // the case the limit exists for: a tiny file whose header claims 20000 x 20000 (1.2 GB as RGB8)
    set_cap(nullptr);
    // a BMP that is only its 54-byte header, with width and height patched to 20000 (stb reads a BMP's size from the
    // header alone; a PNG cut off after IHDR is not accepted by its header reader, measured 2026-11-13: 0 pixels)
    std::vector<unsigned char> bomb = make_bmp(1, 1);
    bomb.resize(54);
    for (int i = 0; i < 4; i++) { bomb[18 + i] = (unsigned char) ((20000u >> (8 * i)) & 0xff); bomb[22 + i] = bomb[18 + i]; }
    (void) make_png_header_only;
    const uint64_t claimed = mtmd_helper_image_declared_pixels(bomb.data(), bomb.size());
    CHECK(claimed == 400000000ull, "header-only bmp declares %llu, expected 400000000", (unsigned long long) claimed);
    CHECK(claimed > mtmd_helper_image_max_decode_pixels(), "a 20000x20000 claim is not above the default limit");
    CHECK(!decodes(bomb, false, 20000, 20000), "a 54-byte file claiming 20000x20000 produced a bitmap");
    std::printf("bomb header: %zu bytes, declares %llu pixels, default limit %llu\n", bomb.size(), (unsigned long long) claimed,
                (unsigned long long) mtmd_helper_image_max_decode_pixels());

    if (g_fail) { std::printf("test-mtmd-image-pixel-cap: %d FAILED\n", g_fail); return 1; }
    std::printf("test-mtmd-image-pixel-cap: OK\n");
    return 0;
}
