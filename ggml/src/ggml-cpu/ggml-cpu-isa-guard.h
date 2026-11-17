// CPU ISA guard (row S69-ENGREV-LLAMACPP-ISA-GUARD..., bundle fork-engrev-ports-07).
//
// WHY: a static build compiled with -mavx512vbmi (or /arch:AVX512, GGML_NATIVE on another box) runs the AVX-512 kernels without
// asking the CPU. On a host that lacks the instruction the first such kernel dies with an illegal-instruction fault deep inside a
// matmul, hours into a run, with nothing that names the cause. This guard compares the instruction sets the binary was COMPILED
// for with what CPUID (and XGETBV, for OS-enabled register state) says the HOST has, once at ggml_cpu_init, and aborts with a
// message that names the missing flag. system_info also prints the host set next to the compiled set.
//
// Header-only and free of ggml types so a unit test can include it without linking ggml: the decision function takes both masks,
// so a test can fake the host. The compiled mask is computed from the preprocessor of the translation unit that includes this
// header, so it must be included by ggml-cpu.c (built with the backend's flags).
//
// SOURCE-READ: ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp:16-120 (CPUID/XGETBV bit positions; same project), ggml-cpu.c has_* functions.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define GGML_ISA_GUARD_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#define GGML_ISA_AVX2         (1u << 0)
#define GGML_ISA_FMA          (1u << 1)
#define GGML_ISA_AVX512F      (1u << 2)
#define GGML_ISA_AVX512_VBMI  (1u << 3)
#define GGML_ISA_AVX512_VNNI  (1u << 4)
#define GGML_ISA_AVX512_BF16  (1u << 5)
#define GGML_ISA_COUNT        6

static inline const char * ggml_isa_guard_name(int bit) {
    static const char * const names[GGML_ISA_COUNT] = {
        "AVX2", "FMA", "AVX512F", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16",
    };
    return (bit >= 0 && bit < GGML_ISA_COUNT) ? names[bit] : "?";
}

// instruction sets this translation unit was compiled to use
static inline uint32_t ggml_isa_guard_compiled_mask(void) {
    uint32_t m = 0;
#if defined(__AVX2__)
    m |= GGML_ISA_AVX2;
#endif
#if defined(__FMA__)
    m |= GGML_ISA_FMA;
#endif
#if defined(__AVX512F__)
    m |= GGML_ISA_AVX512F;
#endif
#if defined(__AVX512VBMI__)
    m |= GGML_ISA_AVX512_VBMI;
#endif
#if defined(__AVX512VNNI__)
    m |= GGML_ISA_AVX512_VNNI;
#endif
#if defined(__AVX512BF16__)
    m |= GGML_ISA_AVX512_BF16;
#endif
    return m;
}

#if defined(GGML_ISA_GUARD_X86)
static inline void ggml_isa_guard_cpuid(unsigned leaf, unsigned sub, unsigned out[4]) {
#if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, (int) leaf, (int) sub);
    out[0] = (unsigned) r[0]; out[1] = (unsigned) r[1]; out[2] = (unsigned) r[2]; out[3] = (unsigned) r[3];
#else
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
#endif
}

static inline uint64_t ggml_isa_guard_xgetbv0(void) {
#if defined(_MSC_VER)
    return (uint64_t) _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t) edx << 32) | eax;
#endif
}
#endif

// instruction sets the HOST can run: CPUID says the unit exists AND, for the wide registers, the OS saves their state
static inline uint32_t ggml_isa_guard_host_mask(void) {
    uint32_t m = 0;
#if defined(GGML_ISA_GUARD_X86)
    unsigned r[4];
    ggml_isa_guard_cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) {
        return 0;
    }
    ggml_isa_guard_cpuid(1, 0, r);
    const unsigned ecx1 = r[2];
    const int osxsave = (ecx1 >> 27) & 1;
    uint64_t xcr0 = osxsave ? ggml_isa_guard_xgetbv0() : 0;
    const int ymm_ok = osxsave && ((xcr0 & 0x6) == 0x6);
    const int zmm_ok = ymm_ok && ((xcr0 & 0xE0) == 0xE0);
    if (ymm_ok && ((ecx1 >> 12) & 1)) {
        m |= GGML_ISA_FMA;
    }
    if (max_leaf >= 7) {
        ggml_isa_guard_cpuid(7, 0, r);
        const unsigned ebx7 = r[1], ecx7 = r[2], eax7max = r[0];
        if (ymm_ok && ((ebx7 >> 5) & 1)) {
            m |= GGML_ISA_AVX2;
        }
        if (zmm_ok && ((ebx7 >> 16) & 1)) {
            m |= GGML_ISA_AVX512F;
            if ((ecx7 >> 1) & 1) {
                m |= GGML_ISA_AVX512_VBMI;
            }
            if ((ecx7 >> 11) & 1) {
                m |= GGML_ISA_AVX512_VNNI;
            }
            if (eax7max >= 1) {
                ggml_isa_guard_cpuid(7, 1, r);
                if ((r[0] >> 5) & 1) {
                    m |= GGML_ISA_AVX512_BF16;
                }
            }
        }
    }
#endif
    return m;
}

// bits the binary needs that the host lacks (0 = fine)
static inline uint32_t ggml_isa_guard_missing(uint32_t compiled, uint32_t host) {
    return compiled & ~host;
}

// first missing flag's name, or NULL when nothing is missing
static inline const char * ggml_isa_guard_first_missing_name(uint32_t compiled, uint32_t host) {
    const uint32_t miss = ggml_isa_guard_missing(compiled, host);
    for (int b = 0; b < GGML_ISA_COUNT; ++b) {
        if (miss & (1u << b)) {
            return ggml_isa_guard_name(b);
        }
    }
    return NULL;
}

// "AVX2 FMA AVX512F ..." for the bits set; writes into buf (always NUL-terminated)
static inline void ggml_isa_guard_format(uint32_t mask, char * buf, size_t n) {
    size_t pos = 0;
    if (n == 0) {
        return;
    }
    buf[0] = '\0';
    for (int b = 0; b < GGML_ISA_COUNT; ++b) {
        if (!(mask & (1u << b))) {
            continue;
        }
        const char * s = ggml_isa_guard_name(b);
        for (const char * p = s; *p && pos + 2 < n; ++p) {
            buf[pos++] = *p;
        }
        if (pos + 2 < n) {
            buf[pos++] = ' ';
        }
        buf[pos] = '\0';
    }
    if (pos > 0 && buf[pos - 1] == ' ') {
        buf[pos - 1] = '\0';
    }
}
