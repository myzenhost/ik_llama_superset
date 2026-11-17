#include <cmath>
#include <cstdlib>
// Model-free test of examples/audio/asr-compression: whole-text compression-ratio repeat check.
#include "asr-compression.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static std::string rep(const std::string & s, int n) { std::string o; for (int i = 0; i < n; ++i) o += s; return o; }

int main() {
    const std::vector<std::string> loops = {
        rep("the ", 100),
        rep("Thank you. ", 40),
        rep("I'm sorry. ", 60),
        rep("la ", 200),
    };
    const std::vector<std::string> prose = {
        "the quick brown fox jumps over the lazy dog and then it went home to rest for the night, as foxes often do when the sun sets.",
        "So I went to the store yesterday and bought some apples, then I walked home along the river and met an old friend who told me about her new job in the city.",
        "Ladies and gentlemen, welcome aboard. Please make sure your seat belt is fastened and your tray table is stowed before we depart for Lisbon.",
    };

    printf("zlib available: %d\n", (int) asr_compression_zlib_available());

    for (const auto & t : loops) {
        const double r = asr_compression_ratio(t);
        const double e = asr_compression_ratio_estimate(t);
        printf("loop  ratio=%.3f est=%.3f  '%.20s...'\n", r, e, t.c_str());
        CHECK(asr_text_repeats(t));
        CHECK(e > ASR_COMPRESSION_RATIO_THOLD);   // own estimator agrees on its own
    }
    for (const auto & t : prose) {
        const double r = asr_compression_ratio(t);
        const double e = asr_compression_ratio_estimate(t);
        printf("prose ratio=%.3f est=%.3f  '%.20s...'\n", r, e, t.c_str());
        CHECK(!asr_text_repeats(t));
        CHECK(e <= ASR_COMPRESSION_RATIO_THOLD);
    }

    // The no-zlib size model against zlib itself. Sizes were measured with python's zlib.compress (default level) on the same UTF-8 bytes:
    //   fox 44->51, thanks 10->18, thanks_x20 220->22, para 304->192, uh_x40 120->14, dots 100->12, jp 60->67, numbers 62->58.
    // The previous flat-bits estimator was 17-44 percent too small on these (scratch comparison of both against python zlib, 2026-10-11), so it over-stated
    // the ratio; the model now charges the RFC 1951 block costs, the dynamic-block header and the 6 framing bytes. Row S69-whispercpp-PICK-TEMP-FALLBACK-2026-09-28-01.
    {
        struct fx { const char * name; std::string text; int zlib; };
        const std::vector<fx> fixtures = {
            { "fox", "The quick brown fox jumps over the lazy dog.", 51 },
            { "thanks", "Thank you.", 18 },
            { "thanks_x20", rep("Thank you. ", 20), 22 },
            { "para", "So the committee met on Tuesday to review the proposal, and after a long discussion about the budget, the timeline and the risks "
                      "involved, they decided to postpone the final vote until the next quarter so that the finance team could prepare a more detailed report "
                      "for everyone to read before the meeting.", 192 },
            { "uh_x40", rep("uh ", 40), 14 },
            { "dots", std::string(100, '.'), 12 },
            { "jp", "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe3\x80\x81\xe4\xb8\x96\xe7\x95\x8c\xe3\x80\x82"
                    "\xe4\xbb\x8a\xe6\x97\xa5\xe3\x81\xaf\xe3\x81\x84\xe3\x81\x84\xe5\xa4\xa9\xe6\xb0\x97\xe3\x81\xa7\xe3\x81\x99\xe3\x81\xad\xe3\x80\x82", 67 },
            { "numbers", "one two three four five six seven eight nine ten eleven twelve", 58 },
        };
        for (const auto & f : fixtures) {
            const double e   = asr_compression_ratio_estimate(f.text);
            const long   est = std::lround((double) f.text.size() / e);  // the size the model implies
            const double err = std::fabs((double) est - f.zlib) / f.zlib;
            printf("  size model %-10s est %3ld bytes, zlib %3d (%.0f%%)\n", f.name, est, f.zlib, err * 100);
            CHECK(err <= 0.08 || std::labs(est - f.zlib) <= 2);
        }
        // the decision the gate takes from the ratio, on both sides of the threshold
        for (const auto & f : fixtures) {
            const bool loop = std::string(f.name) == "thanks_x20" || std::string(f.name) == "uh_x40" || std::string(f.name) == "dots";
            CHECK((asr_compression_ratio_estimate(f.text) > ASR_COMPRESSION_RATIO_THOLD) == loop);
        }
        // incompressible bytes fall back to the stored block: n + 5 + 6
        std::string noise;
        unsigned x = 12345;
        for (int i = 0; i < 200; ++i) {
            x = x * 1103515245u + 12345u;
            noise.push_back((char) (x >> 16));
        }
        CHECK(std::lround((double) noise.size() / asr_compression_ratio_estimate(noise)) <= (long) noise.size() + 11);
    }

    // edge cases
    CHECK(asr_compression_ratio("") == 0.0);
    CHECK(asr_compression_ratio_estimate("") == 0.0);
    CHECK(!asr_text_repeats(""));
    CHECK(!asr_text_repeats("ok"));

    // threshold is strict (>): a text is not "repeating" at its own ratio
    const double r0 = asr_compression_ratio(loops[0]);
    CHECK(!asr_text_repeats(loops[0], r0));
    CHECK(asr_text_repeats(loops[0], r0 - 0.01));

    if (asr_compression_zlib_available()) {
        // Reference value measured with Python 3 zlib.compress(("the "*100).encode()):
        //   400 / 17 = 23.529411764705884
        CHECK(std::abs(asr_compression_ratio_zlib(loops[0]) - 23.529411764705884) < 1e-9);
        // 2nd Python reference: the 'quick brown fox' sentence, 1.2626262626262625
        CHECK(std::abs(asr_compression_ratio_zlib(prose[0]) - 1.2626262626262625) < 1e-9);
        // estimator within a factor of 2 of zlib on repetitive input
        const double z = asr_compression_ratio_zlib(loops[0]), e = asr_compression_ratio_estimate(loops[0]);
        CHECK(e > z * 0.5 && e < z * 2.0);
    } else {
        CHECK(asr_compression_ratio_zlib("abc") < 0.0);
    }

    printf(n_fail ? "test-asr-compression: %d FAILED\n" : "test-asr-compression: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
