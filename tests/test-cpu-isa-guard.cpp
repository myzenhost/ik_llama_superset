// CPU ISA guard decision logic (ggml/src/ggml-cpu/ggml-cpu-isa-guard.h; row S69-ENGREV ISA-GUARD, bundle fork-engrev-ports-07).
//
// BEFORE: a binary compiled for AVX512_VBMI on a host without it ran until the first VBMI kernel and died with an unexplained
// illegal-instruction fault. AFTER: ggml_cpu_init compares compiled vs host masks and aborts naming the flag.
//
// WHAT IT CHECKS, BOTH DIRECTIONS. Trip: for every instruction-set bit, a compiled mask holding it against a faked host mask
// lacking it reports exactly that bit's name. No-trip: a host mask that has everything compiled never reports, a compiled mask of
// zero never reports (non-x86), and the REAL host's mask is a superset of what THIS translation unit was compiled for (so the guard
// does not fire on the machine the tests run on -- a guard that always tripped would pass the trip half vacuously).
// Exit 0 pass, 1 failed expectation.
#include "../ggml/src/ggml-cpu/ggml-cpu-isa-guard.h"

#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

int main() {
    const uint32_t all = (1u << GGML_ISA_COUNT) - 1;

    // trip: every bit, alone and together with the others compiled
    for (int b = 0; b < GGML_ISA_COUNT; ++b) {
        const uint32_t host = all & ~(1u << b);
        const char * got = ggml_isa_guard_first_missing_name(all, host);
        EXPECT(got != nullptr && std::strcmp(got, ggml_isa_guard_name(b)) == 0, "bit %d: expected %s got %s", b, ggml_isa_guard_name(b), got ? got : "(null)");
        EXPECT(ggml_isa_guard_missing(all, host) == (1u << b), "bit %d: missing mask wrong", b);
        // compiled only that bit, host has nothing
        got = ggml_isa_guard_first_missing_name(1u << b, 0);
        EXPECT(got != nullptr && std::strcmp(got, ggml_isa_guard_name(b)) == 0, "bit %d alone: wrong name", b);
        // the host having an EXTRA bit the binary does not use must not trip
        EXPECT(ggml_isa_guard_first_missing_name(all & ~(1u << b), all) == nullptr, "bit %d: extra host bit tripped", b);
    }
    // the headline case: VBMI compiled, host lacks only VBMI
    EXPECT(std::strcmp(ggml_isa_guard_first_missing_name(GGML_ISA_AVX2 | GGML_ISA_AVX512F | GGML_ISA_AVX512_VBMI,
                                                         GGML_ISA_AVX2 | GGML_ISA_AVX512F) ? ggml_isa_guard_first_missing_name(
                        GGML_ISA_AVX2 | GGML_ISA_AVX512F | GGML_ISA_AVX512_VBMI, GGML_ISA_AVX2 | GGML_ISA_AVX512F) : "", "AVX512_VBMI") == 0,
           "VBMI headline case");

    // no-trip
    EXPECT(ggml_isa_guard_first_missing_name(all, all) == nullptr, "full host tripped");
    EXPECT(ggml_isa_guard_first_missing_name(0, 0) == nullptr, "empty/empty tripped");
    EXPECT(ggml_isa_guard_first_missing_name(0, all) == nullptr, "nothing compiled tripped");

    // real host: must be a superset of what this TU was compiled for
    const uint32_t compiled = ggml_isa_guard_compiled_mask();
    const uint32_t host = ggml_isa_guard_host_mask();
    char cb[96], hb[96];
    ggml_isa_guard_format(compiled, cb, sizeof(cb));
    ggml_isa_guard_format(host, hb, sizeof(hb));
    std::printf("compiled [%s] host [%s]\n", cb, hb);
    EXPECT(ggml_isa_guard_missing(compiled, host) == 0, "real host lacks something this test was compiled for: [%s] vs [%s]", cb, hb);
#if defined(GGML_ISA_GUARD_X86)
    // an x86-64 host always has at least one tracked unit on anything this build runs on? no: only assert the reader is not dead
    // when the compiler itself targeted AVX2 (then the host must report it, or the CPUID reader is broken)
    if (compiled & GGML_ISA_AVX2) {
        EXPECT(host & GGML_ISA_AVX2, "CPUID reader reports no AVX2 on a host that ran AVX2 code");
    }
#endif

    // format
    char buf[32];
    ggml_isa_guard_format(GGML_ISA_AVX2 | GGML_ISA_FMA, buf, sizeof(buf));
    EXPECT(std::strcmp(buf, "AVX2 FMA") == 0, "format gave '%s'", buf);
    ggml_isa_guard_format(0, buf, sizeof(buf));
    EXPECT(buf[0] == '\0', "empty format");
    char tiny[6];
    ggml_isa_guard_format(all, tiny, sizeof(tiny));
    EXPECT(std::strlen(tiny) < sizeof(tiny), "format overran a tiny buffer");

    if (g_fail) {
        std::printf("%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("test-cpu-isa-guard: all passed\n");
    return 0;
}
