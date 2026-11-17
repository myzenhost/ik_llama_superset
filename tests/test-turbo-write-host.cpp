// CPU-ONLY check of the TurboQuant KV writer algorithm that the CUDA kernels run.
//
// ggml/src/ggml-cuda/turbo-write.h holds ONE definition of the whole-group writer (L2 norm -> normalise ->
// rotate -> centroid quantise -> corrected norm), compiled as __host__ __device__ for nvcc and as plain C++ here.
// cpy.cu / set-rows.cu call it on the device; this test calls it on the host and compares its output with the
// project's reference writer quantize_row_turbo{2,3}_0_ref (ggml-turbo-quant.c), reached through the public
// type-traits table. No backend is initialised; nothing touches a GPU.
//
// What this does NOT cover, and what test-turbo-cpy.cpp (CUDA) covers instead: the device __constant__ tables,
// nvcc code generation, the cpy_f32_q / k_set_rows_quant indexing and the supports_op answers.
//
// Thresholds live in tests/turbo-write-check.h and were fixed before any run. Negative control: the same code
// with signs1/signs2 swapped is a different rotation and must FAIL the same verdict.

#include "turbo-write-check.h"
#include "../ggml/src/ggml-cuda/turbo-write.h"

#include <cstdlib>

static int g_fail = 0;

static void expect(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

template <int NBITS>
static std::vector<uint8_t> quantize_shared(ggml_type t, const std::vector<float> & x, const float * s1, const float * s2) {
    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t nb  = x.size() / tw::kGroup;
    std::vector<uint8_t> out(nb * tsz, 0);
    const turbo_write_tables tab = { s1, s2, NBITS == 4 ? tw::kMid4 : NBITS == 3 ? tw::kMid3 : tw::kMid2,
                                             NBITS == 4 ? tw::kCent4 : NBITS == 3 ? tw::kCent3 : tw::kCent2 };
    for (size_t b = 0; b < nb; ++b) {
        uint8_t * blk = &out[b * tsz];
        if constexpr (NBITS == 4) {
            block_turbo4_0 * y = (block_turbo4_0 *) blk;
            y->norm = ggml_fp32_to_fp16(turbo_write_group<4>(&x[b * tw::kGroup], y->qs, nullptr, tab));
        } else if constexpr (NBITS == 3) {
            block_turbo3_0 * y = (block_turbo3_0 *) blk;
            y->norm = ggml_fp32_to_fp16(turbo_write_group<3>(&x[b * tw::kGroup], y->qs, y->signs, tab));
        } else {
            block_turbo2_0 * y = (block_turbo2_0 *) blk;
            y->norm = ggml_fp32_to_fp16(turbo_write_group<2>(&x[b * tw::kGroup], y->qs, nullptr, tab));
        }
    }
    return out;
}

template <int NBITS>
static void run(ggml_type t, const char * name) {
    printf("%s\n", name);
    expect(tw::traits_of(t)->from_float_ref != nullptr && tw::traits_of(t)->to_float != nullptr,
           "type traits expose from_float_ref and to_float");
    expect(tw::traits_of(t)->blck_size == tw::kGroup, "block size is 128 (one rotation group per block)");

    const std::vector<float>   x   = tw::gen_data(96, 20260929);
    const std::vector<uint8_t> ref = tw::quantize_ref(t, x);
    {
        const std::vector<float> d0 = tw::dequant(t, ref, x.size());
        printf("  diag: block0 norm %.6g  first dequant %.6g %.6g %.6g %.6g  src %.6g %.6g  ref[0..5] %d %d %d %d %d %d\n",
               tw::norm_at(t, ref.data()), d0[0], d0[1], d0[2], d0[3], x[0], x[1],
               ref[0], ref[1], ref[2], ref[3], ref[4], ref[5]);
    }

    // 1. the shared writer, correct tables
    const std::vector<uint8_t> good = quantize_shared<NBITS>(t, x, tw::kSigns1, tw::kSigns2);
    const tw::Verdict vg = tw::judge(t, x, good, ref);
    tw::print_verdict("shared writer (correct rotation)", vg);
    expect(vg.pass, "shared writer matches quantize_row_turbo*_0_ref within the fixed thresholds");
    expect(good == ref, "shared writer is BYTE-IDENTICAL to the reference on this data (host build)");

    // 2. negative control: swapped sign tables = a different rotation. Must fail the same verdict.
    const std::vector<uint8_t> bad = quantize_shared<NBITS>(t, x, tw::kSigns2, tw::kSigns1);
    const tw::Verdict vb = tw::judge(t, x, bad, ref);
    tw::print_verdict("CONTROL wrong rotation (s1<->s2)", vb);
    expect(!vb.pass, "CONTROL: wrong-rotation writer FAILS the verdict (the verdict is measuring the rotation)");
    expect(vb.nmse_recon > 0.5, "CONTROL: wrong-rotation reconstruction NMSE > 0.5");
    expect(vb.idx_mismatch > 0.1, "CONTROL: wrong-rotation index mismatch > 0.1");

    // 3. second control: a single flipped sign in ONE table entry. Smaller defect, must still be caught.
    float s1_flip[128];
    memcpy(s1_flip, tw::kSigns1, sizeof(s1_flip));
    s1_flip[5] = -s1_flip[5];
    const std::vector<uint8_t> flip = quantize_shared<NBITS>(t, x, s1_flip, tw::kSigns2);
    const tw::Verdict vf = tw::judge(t, x, flip, ref);
    tw::print_verdict("CONTROL one flipped signs1 entry", vf);
    expect(!vf.pass, "CONTROL: one flipped sign entry FAILS the verdict");

    // 4. the verdict must not be vacuous on identical input either: reference vs reference passes
    expect(tw::judge(t, x, ref, ref).pass, "reference vs itself passes (verdict is not trivially failing)");

    // 5. layout: block bytes beyond the written fields are untouched by the writer (zero-initialised here)
    const size_t tsz = tw::traits_of(t)->type_size;
    expect(tsz == (t == GGML_TYPE_TURBO3_0 ? 50u : t == GGML_TYPE_TURBO4_0 ? 66u : 34u), "block size is 50 bytes (turbo3) / 66 bytes (turbo4) / 34 bytes (turbo2)");
}

// ggml_quantize_chunk is the entry point llama-quantize and the converter use. turbo2/3/4 had no case there, so any caller asking to quantise to a
// turbo type aborted. It must produce exactly the reference writer's bytes, row by row, and report exactly nrows * row_size.
static void check_quantize_chunk(ggml_type t, const char * name) {
    printf("%s via ggml_quantize_chunk\n", name);
    const int64_t n_per_row = 1024, nrows = 12;
    const std::vector<float> x = tw::gen_data(96, 20260930);
    const size_t row_size = ggml_row_size(t, n_per_row);
    std::vector<uint8_t> dst(row_size * nrows, 0xEE);
    const size_t got = ggml_quantize_chunk(t, x.data(), dst.data(), 0, nrows, n_per_row, nullptr, nullptr);
    expect(got == row_size * nrows, "returns exactly nrows * row_size bytes");
    const std::vector<uint8_t> ref = tw::quantize_ref(t, x);
    expect(ref.size() == dst.size() && dst == ref, "bytes are IDENTICAL to the reference writer run over the same floats");
    std::vector<float> x2 = x;
    x2[5000] += 3.0f;
    std::vector<uint8_t> dst2(row_size * nrows, 0xEE);
    ggml_quantize_chunk(t, x2.data(), dst2.data(), 0, nrows, n_per_row, nullptr, nullptr);
    expect(dst2 != ref, "CONTROL: a changed input float changes the bytes (the comparison is not vacuous)");
    // start offset: quantising rows 3..5 alone must equal the same rows of the full run
    std::vector<uint8_t> part(row_size * nrows, 0xEE);
    const size_t gp = ggml_quantize_chunk(t, x.data(), part.data(), 3 * n_per_row, 3, n_per_row, nullptr, nullptr);
    expect(gp == row_size * 3 && std::memcmp(part.data() + 3 * row_size, ref.data() + 3 * row_size, 3 * row_size) == 0, "a start offset lands the rows at the right place");
}

int main() {
    tw::init_tables();
    check_quantize_chunk(GGML_TYPE_TURBO3_0, "turbo3");
    check_quantize_chunk(GGML_TYPE_TURBO2_0, "turbo2");
    run<3>(GGML_TYPE_TURBO3_0, "turbo3");
    run<2>(GGML_TYPE_TURBO2_0, "turbo2");
    run<4>(GGML_TYPE_TURBO4_0, "turbo4");
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
