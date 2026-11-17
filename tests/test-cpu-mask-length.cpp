// parse_cpu_mask (common/common.cpp): length of the hex mask.
// Row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01 (D3). The parser capped the digit count at the literal 128, a hidden
// copy of GGML_MAX_N_THREADS / 4 (512 bits, 4 bits per hex digit), and dropped everything past the 128th digit from the
// MOST significant end: "0x" + 200 zeros + "ff" came back as an empty mask. Digits past the limit must be accepted when they are
// leading zeros and refused (not silently dropped) when they would address a CPU the thread array cannot hold.
#include "common.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static bool parse(const std::string & m, bool (&mask)[GGML_MAX_N_THREADS]) {
    std::memset(mask, 0, sizeof(mask));
    return parse_cpu_mask(m, mask);
}

static std::string set_bits(const bool (&mask)[GGML_MAX_N_THREADS]) {
    std::string s;
    for (size_t i = 0; i < GGML_MAX_N_THREADS; ++i) {
        if (mask[i]) { s += std::to_string(i) + ","; }
    }
    return s;
}

int main() {
    static_assert(GGML_MAX_N_THREADS % 4 == 0, "digit count below assumes it");
    constexpr size_t N_DIGITS = GGML_MAX_N_THREADS / 4; // the most digits that can carry a set bit
    bool mask[GGML_MAX_N_THREADS];

    CHECK(parse("0xff", mask) && set_bits(mask) == "0,1,2,3,4,5,6,7,", "0xff -> %s", set_bits(mask).c_str());
    CHECK(parse("A", mask) && set_bits(mask) == "1,3,", "A -> %s", set_bits(mask).c_str());

    // exactly the limit: top digit 8 = bit GGML_MAX_N_THREADS - 1
    CHECK(parse("0x8" + std::string(N_DIGITS - 1, '0'), mask) && set_bits(mask) == std::to_string(GGML_MAX_N_THREADS - 1) + ",",
          "top bit -> %s", set_bits(mask).c_str());

    // leading zeros past the limit are harmless and must not move the value
    CHECK(parse("0x" + std::string(200, '0') + "ff", mask) && set_bits(mask) == "0,1,2,3,4,5,6,7,",
          "200 leading zeros + ff -> '%s'", set_bits(mask).c_str());
    CHECK(parse(std::string(N_DIGITS + 1, '0'), mask) && set_bits(mask).empty(), "all-zero over-long mask");

    // a set bit beyond the thread array is refused, never dropped
    CHECK(!parse("0x1" + std::string(N_DIGITS, '0'), mask), "bit %zu must be refused", (size_t) GGML_MAX_N_THREADS);
    CHECK(!parse("0x" + std::string(N_DIGITS, '0') + "1" + std::string(N_DIGITS, '0'), mask), "far bit must be refused");

    // an invalid character is refused wherever it sits, also in the part past the limit
    CHECK(!parse("0x" + std::string(200, '0') + "zz", mask), "invalid hex digit in the tail");
    CHECK(!parse("0xz" + std::string(200, '0') + "ff", mask), "invalid hex digit in the head");

    if (g_fails) {
        std::printf("test-cpu-mask-length: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-cpu-mask-length: ok\n");
    return 0;
}
