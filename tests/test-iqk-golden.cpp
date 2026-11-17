// Byte-exact GOLDEN VECTORS for the IQ*_K quant family.
//
// WHY THIS EXISTS, and why test-quantize-fns.cpp does not already cover it.
// test-quantize-fns is a TOLERANCE test: it asserts the round-trip error stays under
// MAX_QUANTIZATION_TOTAL_ERROR = 0.002f. That is the right instrument for "is this quantizer sane",
// and the WRONG instrument for "is this reimplementation the same function". A port that is
// close-but-not-identical passes a tolerance test and then silently degrades every model that uses
// it -- no crash, no failing test, no symptom anyone can attribute. This file asserts BYTE
// EQUALITY instead, which is the only bar a reimplementation can be held to.
//
// WHAT IT IS FOR. The IQ*_K family (21 types, enumerated at runtime below) is this fork's
// distinguishing feature and exists in no other engine -- measured 2026-09-14 against ferrox's
// source, where the positive control q4_k appears in 45 .rs files and every IQ*_K variant in zero.
// Any future port of these kernels -- to Rust, to another backend, or a refactor of the C itself --
// needs an oracle that fails on a single changed bit. This is that oracle. Prior art for the shape:
// ferrox carries IQ1_S_DEQUANT_MATCHES_INDEPENDENT_PYTHON_REFERENCE in its own test suite, and a
// cJSON C-to-Rust port reported 22/22 tests plus 70M+ differential comparisons with zero
// divergences. An oracle is what makes those numbers mean anything.
//
// WHAT IT HASHES AND WHY BOTH.
//   quantize   from_float_ref -> the RAW QUANTIZED BYTES. This pins the on-disk format: a port that
//              produces different bytes writes GGUFs this engine cannot read.
//   dequantize to_float       -> the FLOAT BITS (not the values). This pins the arithmetic: two
//              implementations can agree on bytes and still differ in the dequant path.
// from_float_ref is used deliberately, not the CPU from_float: it is the scalar reference path, so
// it does not vary with the host's SIMD extensions.
//
// IT IS NOT, HOWEVER, TOOLCHAIN-INDEPENDENT, and an earlier version of this comment claimed it was.
// MEASURED 2026-09-14, goldens written by mingw g++ (build-static) and verified by MSVC cl.exe
// (build-msvc-cpu) from the same source on the same input:
//     12 of 21 identical
//      4 of 21 differ in BYTES ONLY, dequantized values bit-identical:
//              iq4_k, iq6_k, iq2_ks, iq4_k_r4
//      5 of 21 differ in the DEQUANTIZED VALUES themselves:
//              iq4_kss, iq5_ks, iq4_kt, iq1_kt, iq5_ks_r4
// The first group is choosing a different-but-equivalent encoding. The second is the same reference
// quantizer computing different numbers under two compilers -- consistent with floating-point
// contraction and evaluation-order differences (gcc contracts to FMA by default; MSVC /fp:precise
// does not), which change tie-breaking inside the search these quantizers perform.
//
// CAUSE ISOLATED, same day, with a THIRD build that changes exactly one variable:
//     mingw g++ -ffp-contract=off  ->  agrees with MSVC on ALL 21 types, byte for byte, both hashes
// All 9 divergent types moved to MSVC's values; none stayed unchanged, none took a third value.
// gcc contracts multiply-add into FMA by default and MSVC /fp:precise does not, so intermediates
// round differently and flip tie-breaking inside the search these quantizers run.
//
// DISABLE-REVERSIBLE: -ffp-contract=off is NOT applied to this engine. It is a flag on ONE
// SEPARATE tree (build-static-nofma) that exists solely to emit a reference golden. Every shipped
// and default build keeps contraction ON and keeps its FMA performance; no capability is switched
// off anywhere a user's binary is produced. Re-deriving the portable set later is the documented
// cmake line at the top of tests/iqk-goldens-portable.txt.
//
// CONSEQUENCE FOR A PORT, which is what this file exists to serve -- and it is the OPPOSITE of what
// the two-configuration result suggested: byte equality IS a usable bar, provided the reference is
// built with contraction off. tests/iqk-goldens-portable.txt is that set; it was produced by the
// -ffp-contract=off build and independently reproduced byte-for-byte by the MSVC build before being
// committed, and it verifies rc=0 under both.
//     tests/iqk-goldens.txt           what THIS tree's default build produces -- the registered
//                                     ctest, for detecting change in our own engine.
//     tests/iqk-goldens-portable.txt  what the reference computes with contraction off -- the
//                                     cross-implementation oracle. Hold a port to THIS one.
// Holding a port to the default-build file instead would fail it on 9 of 21 types while it was
// perfectly correct.
//
// HONEST LIMITS, stated because a golden file is easy to over-trust:
//   * It pins BEHAVIOUR, not CORRECTNESS. If the current C is wrong, this freezes the wrong answer.
//     Its job is to detect CHANGE, which is exactly what a port needs.
//   * The input is one deterministic waveform, not a corpus. It exercises the code path, not every
//     numerical corner. Widening the corpus is the obvious next step and is NOT claimed here.
//   * A type whose from_float_ref or to_float is null is reported SKIPPED, never silently passed.
//
// Exit: 0 all types matched (or goldens written), 1 a MISMATCH, 2 refused (no goldens to verify).

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// FNV-1a 64. Chosen over anything stronger on purpose: no dependency, and the property needed is
// "any changed byte changes the digest", not cryptographic strength.
static uint64_t fnv1a(const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Deterministic input. No <random>: std::mt19937 is portable but the DISTRIBUTIONS are not
// specified bit-for-bit across implementations, so a golden built with one libstdc++ could fail on
// another for reasons that have nothing to do with the quantizer.
static void generate_data(size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1f + 2.0f * cosf((float) i) + 0.37f * sinf(0.5f * (float) i);
    }
}

struct golden {
    std::string name;
    uint64_t    qhash;
    uint64_t    dhash;
    size_t      qbytes;
};

int main(int argc, char ** argv) {
    printf("EXIT-CODE-SEMANTICS: findings=1\n");

    const char * verify_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            verify_path = argv[++i];
        }
    }

    // 4096 elements divides every IQ*_K block size in this fork (16, 32, 256), so no type is
    // skipped merely for not tiling the buffer.
    const size_t N = 4096;
    std::vector<float> src(N);
    generate_data(N, src.data());

    std::vector<golden> got;
    int skipped = 0;
    // Counted and reported separately from mismatches: an overrun is a defect in the QUANTIZER,
    // not a disagreement with a golden, and collapsing the two would hide it behind a pass.
    int overruns = 0;

    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const ggml_type type = (ggml_type) t;
        const char * name = ggml_type_name(type);
        if (!name) {
            continue;
        }
        // The IQ*_K family only: names starting "iq" and containing "_k".
        std::string n(name);
        if (n.rfind("iq", 0) != 0 || n.find("_k") == std::string::npos) {
            continue;
        }

        const auto * tr = ggml_get_type_traits(type);
        if (!tr || !tr->from_float_ref || !tr->to_float) {
            printf("  SKIPPED %-12s no reference quantize/dequantize in this build\n", name);
            skipped++;
            continue;
        }
        const size_t blck = ggml_blck_size(type);
        if (blck == 0 || N % blck != 0) {
            printf("  SKIPPED %-12s block size %zu does not divide %zu\n", name, blck, N);
            skipped++;
            continue;
        }

        // MEASURE the payload rather than assume ggml_row_size(type, N) bounds it.
        //
        // WHY. The first version of this test sized the output buffer with ggml_row_size(type, N)
        // and crashed with STATUS_HEAP_CORRUPTION (0xC0000374) on 6 of 12 runs -- intermittent,
        // because a 12-byte overrun only trips the Windows heap validator sometimes.
        //
        // THE CAUSE IS AN API CONTRACT DIFFERENCE, NOT A DEFECT. An _r4 type's from_float_ref
        // interprets k as FOUR ROWS of k/4:
        //   quantize_row_iq4_k_r4_ref(x, y, k) -> quantize_iq4_k_r4(x, y, 4, k/4, ...)
        //   (iqk_quantize.cpp:6631)
        // so the buffer it needs is 4*ggml_row_size(type, k/4), which for a base type carrying a
        // row_meta is 3 metas larger than ggml_row_size(type, k). MEASURED, both hypotheses printed
        // side by side by the OVER ROW line below:
        //   iq4_ks_r4  wrote 2192;  ggml_row_size(t,4096)=2180;  4*ggml_row_size(t,1024)=2192
        //   iq5_ks_r4  wrote 2704;  ggml_row_size(t,4096)=2692;  4*ggml_row_size(t,1024)=2704
        // The 4-row reading matches EXACTLY in both cases. ggml_row_size is right; asking it for a
        // single row of N was the wrong question. An earlier version of this comment called it a
        // quantizer overrun -- that was wrong and the arithmetic above is why.
        //
        // The residual hazard is real but narrower than "bug": from_float_ref's signature gives a
        // caller no way to see that k means 4 rows for these types, so the obvious buffer size is
        // silently too small. ggml's own quantize path does not hit it -- ggml_quantize_chunk calls
        // quantize_X(src, dst, nrows, n_per_row) with the row count explicit.
        //
        // Same contract explains the other oddity here: iq4_ks_r4 and iq5_ks_r4 are the only _r4
        // types whose DEQUANTIZED hash differs from their base type, because the _ks family carries
        // a row-wide scale and 4 rows of 1024 genuinely is not 1 row of 4096. iq2/3/4/5_k_r4 have
        // no row scale and reproduce their base hash exactly.
        //
        // HOW THE TRUE SIZE IS MEASURED, exactly rather than by a rule of thumb. The quantizer is
        // run TWICE into generously oversized buffers pre-filled with 0x00 and with 0xFF. A byte
        // the quantizer WROTE is identical in both runs; a byte it did not write still holds its
        // fill and therefore DIFFERS. So the payload ends at the last index where the two runs
        // agree. There is no ambiguity: unwritten bytes always differ, written bytes always agree.
        // This needs no table of which types are _r4 and no knowledge of any base type, so it stays
        // correct when a type is added.
        const size_t declared = ggml_row_size(type, N);
        const size_t slack = 8 * declared + 4096;        // far beyond any 4x-rows requirement
        std::vector<uint8_t> qa(slack, 0x00);
        std::vector<uint8_t> qb(slack, 0xFF);
        std::vector<float>   back(N, 0.0f);

        tr->from_float_ref(src.data(), qa.data(), N);
        tr->from_float_ref(src.data(), qb.data(), N);

        size_t written = 0;
        for (size_t i = slack; i-- > 0; ) {
            if (qa[i] == qb[i]) { written = i + 1; break; }
        }
        // The competing hypothesis, printed alongside so the reader can judge which holds rather
        // than take a label on trust. An _r4 reference quantizer interprets k as FOUR ROWS of k/4
        // (quantize_row_iq4_k_r4_ref -> quantize_iq4_k_r4(x, y, 4, k/4, ...), iqk_quantize.cpp:6631).
        // If that is the real contract, the correct buffer is 4*ggml_row_size(type, k/4) and
        // ggml_row_size(type, k) is simply the wrong question -- not a broken answer.
        // The interleave width is 4 for _r4, 8 for _r8 and 16 for _r16 (iq4_ks_r16 arrived with the ik-main merge and
        // wrote 2240 = 16*ggml_row_size(type,256), which neither ggml_row_size(type,4096)=2180 nor the 4-row reading
        // 2192 explains). So every width is tried and the one that reproduces the measured payload exactly is named;
        // if none does, the 4-row reading is still printed so the line stays comparable with older output.
        size_t match_rows = 0;
        size_t match_bytes = 0;
        for (const size_t rows : {(size_t) 4, (size_t) 8, (size_t) 16}) {
            if (N % rows == 0 && (N / rows) % blck == 0 && rows * ggml_row_size(type, N / rows) == written) {
                match_rows  = rows;
                match_bytes = written;
                break;
            }
        }
        const size_t as_four_rows = (N % 4 == 0 && (N / 4) % blck == 0)
                                  ? 4 * ggml_row_size(type, N / 4) : 0;
        if (written > declared) {
            if (match_rows != 0) {
                printf("  OVER ROW   %-12s wrote %zu, ggml_row_size(type,%zu)=%zu (+%zu); "
                       "%zu*ggml_row_size(type,%zu)=%zu <= MATCHES the %zu-row reading\n",
                       name, written, N, declared, written - declared, match_rows, N / match_rows, match_bytes, match_rows);
            } else {
                printf("  OVER ROW   %-12s wrote %zu, ggml_row_size(type,%zu)=%zu (+%zu); "
                       "4*ggml_row_size(type,%zu)=%zu <= does NOT match the 4, 8 or 16-row reading\n",
                       name, written, N, declared, written - declared, N / 4, as_four_rows);
            }
            overruns++;
        }

        // Hash what was ACTUALLY written. Hashing `declared` bytes instead would silently drop the
        // overrun from the golden and freeze a number that no correct implementation reproduces.
        const std::vector<uint8_t> & q = qa;
        const size_t qbytes = written;
        tr->to_float(q.data(), back.data(), N);

        golden g;
        g.name   = name;
        g.qhash  = fnv1a(q.data(), qbytes);
        // Hash the BITS, not the values: -0.0f == 0.0f compares equal but is a different encoding,
        // and a port that swaps them has changed the output even though a float compare says it has
        // not.
        g.dhash  = fnv1a(back.data(), back.size() * sizeof(float));
        g.qbytes = qbytes;
        got.push_back(g);
    }

    printf("\nSCOPE: %zu IQ*_K type(s) exercised, %d skipped, input %zu deterministic floats.\n",
           got.size(), skipped, N);
    printf("       Hashes are of the REFERENCE quantize output and the dequantized FLOAT BITS.\n");
    printf("       Payload length per type is MEASURED (0x00/0xFF double-fill), not taken from\n");
    printf("       ggml_row_size -- see the comment at the quantize call for why.\n");
    if (overruns > 0) {
        printf("\n%d TYPE(S) wrote MORE than ggml_row_size(type, N). Per the line(s) above this is\n"
               "NOT a quantizer defect: an interleaved (_r4/_r8/_r16) from_float_ref reads k as R rows of k/R, and\n"
               "R*ggml_row_size(type, N/R) matches the payload exactly (the MATCHES line names R). ggml_row_size(type, N) is\n"
               "the wrong question for these types, not a wrong answer. It is still a trap worth\n"
               "seeing -- the signature gives a caller no way to know -- so it is reported rather\n"
               "than silently absorbed. Goldens below cover the bytes ACTUALLY written.\n", overruns);
    }

    if (got.empty()) {
        printf("ABORT: no IQ*_K type was exercised. That is a statement about this BUILD, not\n");
        printf("       about the quants -- refusing to write or verify an empty golden set.\n");
        return 2;
    }

    if (!verify_path) {
        printf("\n# golden vectors -- name qhash dhash qbytes\n");
        for (const auto & g : got) {
            printf("%s %016llx %016llx %zu\n", g.name.c_str(),
                   (unsigned long long) g.qhash, (unsigned long long) g.dhash, g.qbytes);
        }
        printf("\nWrote goldens to stdout. Re-run with --verify <file> to check against them.\n");
        printf("exit code: 0   (this program)\n");
        return 0;
    }

    FILE * f = fopen(verify_path, "r");
    if (!f) {
        printf("ABORT: cannot open %s. A missing golden file is not a pass.\n", verify_path);
        return 2;
    }
    std::vector<golden> want;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }
        char nm[128];
        unsigned long long qh = 0, dh = 0;
        unsigned long long qb = 0;
        if (sscanf(line, "%127s %llx %llx %llu", nm, &qh, &dh, &qb) == 4) {
            want.push_back({ nm, qh, dh, (size_t) qb });
        }
    }
    fclose(f);

    if (want.empty()) {
        printf("ABORT: %s parsed to zero goldens. Refusing to report a pass.\n", verify_path);
        return 2;
    }

    int bad = 0, matched = 0, missing = 0;
    // Counted apart from `bad`: a type whose dequantized VALUES changed is a different and far
    // more serious finding than one that only chose a different encoding for the same values.
    int values_differ = 0;
    for (const auto & g : got) {
        const golden * w = nullptr;
        bool any_equal = false;
        for (const auto & x : want) {
            if (x.name != g.name) { continue; }
            // A type may carry SEVERAL accepted lines (a file comment must say why): the reference computes different, individually
            // reproducible numbers under floating-point contraction on vs off (iq4_ks_r16: MSVC / gcc -ffp-contract=off vs gcc default).
            // The first line is the one a mismatch is reported against; a third value still fails.
            if (!w) { w = &x; }
            if (x.qhash == g.qhash && x.dhash == g.dhash && x.qbytes == g.qbytes) { any_equal = true; }
        }
        if (any_equal) {
            matched++;
            continue;
        }
        if (!w) {
            printf("  NO GOLDEN  %-12s present in the build, absent from %s\n",
                   g.name.c_str(), verify_path);
            missing++;
            continue;
        }
        if (w->qhash != g.qhash || w->dhash != g.dhash || w->qbytes != g.qbytes) {
            // TWO DIFFERENT SEVERITIES, reported apart. Collapsing them to "MISMATCH" hides the
            // distinction that actually matters: whether the NUMBERS changed, or only the encoding
            // chosen to represent the same numbers.
            const bool values_too = (w->dhash != g.dhash) || (w->qbytes != g.qbytes);
            printf("  %s   %s\n", values_too ? "VALUES DIFFER " : "BYTES ONLY    ",
                   g.name.c_str());
            printf("      quantized bytes  want %016llx got %016llx\n",
                   (unsigned long long) w->qhash, (unsigned long long) g.qhash);
            printf("      dequantized bits want %016llx got %016llx%s\n",
                   (unsigned long long) w->dhash, (unsigned long long) g.dhash,
                   values_too ? "" : "   <- IDENTICAL: same values, different encoding");
            printf("      row size         want %zu got %zu\n", w->qbytes, g.qbytes);
            bad++;
            if (values_too) {
                values_differ++;
            }
        } else {
            matched++;
        }
    }

    printf("\n  matched: %d   DIFFERING: %d (of which %d differ in VALUES, %d in bytes only)"
           "   no-golden: %d   skipped: %d\n",
           matched, bad, values_differ, bad - values_differ, missing, skipped);
    if (bad) {
        printf("\n  READ THIS BEFORE REGENERATING. A difference here is NOT automatically a\n");
        printf("  regression, because these goldens are TOOLCHAIN-SCOPED -- a fact established by\n");
        printf("  measurement on 2026-09-14, after the opposite was asserted in this file:\n");
        printf("    goldens written by mingw g++ (build-static), verified by MSVC cl.exe\n");
        printf("    (build-msvc-cpu), same source, same input -> 12 of 21 identical, 4 differing\n");
        printf("    in BYTES ONLY with identical dequantized values (iq4_k, iq6_k, iq2_ks,\n");
        printf("    iq4_k_r4), and 5 differing in VALUES (iq4_kss, iq5_ks, iq4_kt, iq1_kt,\n");
        printf("    iq5_ks_r4).\n");
        printf("  So: BYTES ONLY across a compiler change is expected for those four. VALUES\n");
        printf("  DIFFER is the serious class -- the same reference quantizer producing different\n");
        printf("  numbers -- and for a port it is the only line that must hold.\n");
        printf("  Compare against a golden recorded with YOUR toolchain, or treat the dequantized\n");
        printf("  hash as the portable invariant and the byte hash as the local one.\n");
    }
    printf("exit code: %d   (this program)\n", bad ? 1 : 0);
    return bad ? 1 : 0;
}
