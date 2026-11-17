// GPU check (CUDA, Vulkan, ...) of the TurboQuant KV WRITER: GGML_OP_CPY and GGML_OP_SET_ROWS, F32 -> turbo3 / turbo2, and turbo4 on a backend that
// claims it (Vulkan does; CUDA deliberately does not -- the control in section E depends on the backend).
//
// Why this is not a test-backend-ops case: that tool compares a backend with the CPU backend, and the CPU
// backend has no turbo writer by design (ggml-cpu.c type traits carry no turbo entry; llama-context.cpp refuses
// turbo KV on CPU), so the reference run would abort. The reference here is the project's own host writer,
// quantize_row_turbo{2,3}_0_ref (ggml-turbo-quant.c), reached through the public type-traits table.
//
// Cases (each for turbo3 and turbo2):
//   A  K store as llama-build-context.cpp:1004-1007 builds it: ggml_cpy(F32 [128, n_head_kv, n_tokens] ->
//      view_2d(cache, 128, n_tokens*n_head_kv, row_size, row_size*n_head_kv*kv_head)). The cache is pre-filled with 0xA5;
//      bytes outside the written window must still be 0xA5 (no stray writes), bytes inside must match the reference.
//   B  V store, non-transposed (flash attention) layout: 1D destination view at kv_head*row_size.
//   C  source that is a strided VIEW of a larger tensor (row stride > 128 floats), the shape after a permute/slice.
//   D  GGML_OP_SET_ROWS with I64 row indices into a 2D cache.
//   E  supports_op: CPY/SET_ROWS to turbo3 and turbo2 must be claimed. turbo4: CUDA must NOT claim it (it has no turbo4 writer); a backend that
//      DOES claim CPY must also claim SET_ROWS, and then cases A-D run for turbo4 too (4-bit PolarQuant, 66-byte block).
// Verdicts, thresholds and the negative control live in tests/turbo-write-check.h (thresholds fixed before any
// run). Negative control here: the GPU bytes judged against a reference written with swapped sign tables (a
// different rotation) MUST fail; if that passed, the verdict would not be measuring the rotation.
//
// Usage:  test-turbo-cpy [backend-name]      default CUDA0; Vulkan0 for the Vulkan writer shaders (cpy_f32_turbo*, set_rows_*_turbo*).
//         SKIPs (exit 0) when the backend cannot be created.
// NOT registered with ctest on purpose: it needs the GPU, so it is run by hand, alone (host_slot heavy lane).

#include "turbo-write-check.h"
#include "../ggml/src/ggml-cuda/turbo-write.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <string>

static int g_fail = 0;

static void expect(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

struct Fixture {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~Fixture() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

static ggml_context * new_ctx() {
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    return ggml_init(ip);
}

// turbo4 has no entry in turbo-write.h (the CUDA writer has no turbo4), so its wrong-rotation control is a local copy of the reference algorithm
// (ggml-turbo-quant.c quantize_row_turbo4_0_ref, 4-bit branch) with the two sign tables swapped.
static std::vector<uint8_t> wrong_rotation_ref4(const std::vector<float> & x) {
    const size_t tsz = tw::traits_of(GGML_TYPE_TURBO4_0)->type_size;
    const size_t nb  = x.size() / tw::kGroup;
    std::vector<uint8_t> out(nb * tsz, 0);
    for (size_t b = 0; b < nb; ++b) {
        float v[tw::kGroup];
        double nsq = 0;
        for (int j = 0; j < tw::kGroup; ++j) { v[j] = x[b * tw::kGroup + j]; nsq += (double) v[j] * v[j]; }
        const float norm = (float) std::sqrt(nsq);
        const float inv  = norm > 1e-10f ? 1.0f / norm : 0.0f;
        for (int j = 0; j < tw::kGroup; ++j) v[j] *= inv * tw::kSigns2[j];       // SWAPPED: S2 first
        for (int h = 1; h < tw::kGroup; h *= 2)
            for (int i = 0; i < tw::kGroup; i += 2 * h)
                for (int j = i; j < i + h; ++j) { const float a = v[j], c = v[j + h]; v[j] = a + c; v[j + h] = a - c; }
        block_turbo4_0 * y = (block_turbo4_0 *) &out[b * tsz];
        double rsq = 0;
        for (int j = 0; j < tw::kGroup; ++j) {
            const float r = v[j] * 0.08838834764831845f * tw::kSigns1[j];        // SWAPPED: S1 last
            int idx = 0;
            for (int m = 0; m < 15; ++m) if (r >= tw::kMid4[m]) idx = m + 1;
            y->qs[j / 2] |= (uint8_t) (idx << ((j % 2) * 4));
            rsq += (double) tw::kCent4[idx] * tw::kCent4[idx];
        }
        const float rn = (float) std::sqrt(rsq);
        y->norm = ggml_fp32_to_fp16(rn > 1e-10f ? norm / rn : norm);
    }
    return out;
}

// reference bytes written with swapped sign tables (a different rotation) -- the negative control
template <int NBITS>
static std::vector<uint8_t> wrong_rotation_ref(ggml_type t, const std::vector<float> & x) {
    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t nb  = x.size() / tw::kGroup;
    std::vector<uint8_t> out(nb * tsz, 0);
    const turbo_write_tables tab = { tw::kSigns2, tw::kSigns1, NBITS == 3 ? tw::kMid3 : tw::kMid2,
                                     NBITS == 3 ? tw::kCent3 : tw::kCent2 };
    for (size_t b = 0; b < nb; ++b) {
        if constexpr (NBITS == 3) {
            block_turbo3_0 * y = (block_turbo3_0 *) &out[b * tsz];
            y->norm = ggml_fp32_to_fp16(turbo_write_group<3>(&x[b * tw::kGroup], y->qs, y->signs, tab));
        } else {
            block_turbo2_0 * y = (block_turbo2_0 *) &out[b * tsz];
            y->norm = ggml_fp32_to_fp16(turbo_write_group<2>(&x[b * tw::kGroup], y->qs, nullptr, tab));
        }
    }
    return out;
}

// Judge the written window `rows` blocks starting at block `first` of the cache read-back.
static void judge_window(ggml_type t, const char * label, const std::vector<float> & src, const std::vector<uint8_t> & cache,
                         size_t first_block, size_t total_blocks) {
    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t nb  = src.size() / tw::kGroup;
    std::vector<uint8_t> cand(cache.begin() + first_block * tsz, cache.begin() + (first_block + nb) * tsz);
    const std::vector<uint8_t> ref = tw::quantize_ref(t, src);
    const tw::Verdict v = tw::judge(t, src, cand, ref);
    tw::print_verdict(label, v);
    expect(v.pass, "GPU-written blocks match the host reference within the fixed thresholds");

    const std::vector<uint8_t> wrong = t == GGML_TYPE_TURBO3_0 ? wrong_rotation_ref<3>(t, src) :
                                       t == GGML_TYPE_TURBO4_0 ? wrong_rotation_ref4(src) : wrong_rotation_ref<2>(t, src);
    const tw::Verdict vw = tw::judge(t, src, cand, wrong);
    tw::print_verdict("CONTROL GPU vs wrong-rotation ref", vw);
    expect(!vw.pass, "CONTROL: GPU bytes judged against a wrong-rotation reference FAIL");

    size_t stray = 0;
    for (size_t i = 0; i < total_blocks * tsz; ++i) {
        const bool inside = i >= first_block * tsz && i < (first_block + nb) * tsz;
        if (!inside && cache[i] != 0xA5) ++stray;
    }
    printf("  bytes outside the written window that changed: %zu\n", stray);
    expect(stray == 0, "no stray writes outside the destination window");
}

// A / B / C -----------------------------------------------------------------------------------------------------
enum class Kind { K_STORE, V_STORE_1D, STRIDED_SRC };

static void run_cpy(ggml_backend_t be, ggml_type t, Kind kind, const char * label) {
    printf("%s\n", label);
    const int64_t n_head_kv = 4, n_tokens = 8, kv_head = 3, n_cells = 16;
    const int64_t hd = 128;
    const size_t  row_size = ggml_row_size(t, hd);

    Fixture f; f.ctx = new_ctx();
    ggml_tensor * x = nullptr; ggml_tensor * big = nullptr; ggml_tensor * cache = nullptr; ggml_tensor * view = nullptr;
    size_t n_groups = 0, first_block = 0, total_blocks = 0;

    if (kind == Kind::K_STORE || kind == Kind::STRIDED_SRC) {
        cache = ggml_new_tensor_1d(f.ctx, t, hd * n_cells * n_head_kv);
        total_blocks = n_cells * n_head_kv;
        first_block  = kv_head * n_head_kv;
        view = ggml_view_2d(f.ctx, cache, hd, n_tokens * n_head_kv, row_size, row_size * n_head_kv * kv_head);
        n_groups = n_tokens * n_head_kv;
        if (kind == Kind::K_STORE) {
            x = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, hd, n_head_kv, n_tokens);
        } else {
            big = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, hd, n_head_kv + 2, n_tokens);   // 2 extra heads per token, unused
            x   = ggml_view_3d(f.ctx, big, hd, n_head_kv, n_tokens, big->nb[1], big->nb[2], 0);
        }
    } else {
        const int64_t n_embd_v = 2 * hd;   // two heads' worth per token
        cache = ggml_new_tensor_2d(f.ctx, t, n_embd_v, n_cells);
        total_blocks = n_cells * (n_embd_v / hd);
        first_block  = kv_head * (n_embd_v / hd);
        view = ggml_view_1d(f.ctx, cache, n_tokens * n_embd_v, kv_head * ggml_row_size(t, n_embd_v));
        n_groups = n_tokens * (n_embd_v / hd);
        x = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, n_embd_v, n_tokens);
    }
    ggml_tensor * cp = ggml_cpy(f.ctx, x, view);
    ggml_cgraph * gf = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, cp);

    if (!ggml_backend_supports_op(be, cp)) {
        expect(false, "backend claims GGML_OP_CPY F32 -> turbo (supports_op)");
        return;
    }
    expect(true, "backend claims GGML_OP_CPY F32 -> turbo (supports_op)");
    f.buf = ggml_backend_alloc_ctx_tensors(f.ctx, be);
    if (!f.buf) { expect(false, "buffer allocation"); return; }

    const std::vector<float> data = tw::gen_data((int) n_groups, kind == Kind::V_STORE_1D ? 777u : 20260929u);
    const std::vector<uint8_t> fill(ggml_nbytes(cache), 0xA5);
    ggml_backend_tensor_set(cache, fill.data(), 0, fill.size());
    if (big) {
        // scatter the groups into the strided view; the 2 unused heads per token are poisoned with NaN
        std::vector<float> bigdata(ggml_nelements(big), std::nanf(""));
        for (int64_t tok = 0; tok < n_tokens; ++tok)
            for (int64_t h = 0; h < n_head_kv; ++h)
                std::memcpy(&bigdata[(tok * (n_head_kv + 2) + h) * hd], &data[(tok * n_head_kv + h) * hd], hd * sizeof(float));
        ggml_backend_tensor_set(big, bigdata.data(), 0, bigdata.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(x, data.data(), 0, data.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { expect(false, "graph compute succeeded"); return; }
    expect(true, "graph compute succeeded");

    std::vector<uint8_t> out(ggml_nbytes(cache));
    ggml_backend_tensor_get(cache, out.data(), 0, out.size());
    judge_window(t, "GPU CPY vs host reference", data, out, first_block, total_blocks);
}

// D --------------------------------------------------------------------------------------------------------------
static void run_set_rows(ggml_backend_t be, ggml_type t, const char * label) {
    printf("%s\n", label);
    const int64_t ne0 = 256, n_cells = 16, n_rows = 5;
    const int64_t idx_v[5] = { 7, 2, 11, 0, 9 };

    Fixture f; f.ctx = new_ctx();
    ggml_tensor * dst = ggml_new_tensor_2d(f.ctx, t, ne0, n_cells);
    ggml_tensor * src = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, ne0, n_rows);
    ggml_tensor * idx = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I64, n_rows);
    ggml_tensor * sr  = ggml_set_rows(f.ctx, dst, src, idx);
    ggml_cgraph * gf  = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(gf, sr);

    if (!ggml_backend_supports_op(be, sr)) { expect(false, "backend claims GGML_OP_SET_ROWS -> turbo (supports_op)"); return; }
    expect(true, "backend claims GGML_OP_SET_ROWS -> turbo (supports_op)");
    f.buf = ggml_backend_alloc_ctx_tensors(f.ctx, be);
    if (!f.buf) { expect(false, "buffer allocation"); return; }

    const std::vector<float> data = tw::gen_data((int) (n_rows * ne0 / tw::kGroup), 4242u);
    const std::vector<uint8_t> fill(ggml_nbytes(dst), 0xA5);
    ggml_backend_tensor_set(dst, fill.data(), 0, fill.size());
    ggml_backend_tensor_set(src, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_v, 0, sizeof(idx_v));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { expect(false, "graph compute succeeded"); return; }
    expect(true, "graph compute succeeded");

    std::vector<uint8_t> out(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size());

    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t bpr = ne0 / tw::kGroup;                 // blocks per row
    // gather the written rows in source order, and check every untouched row still holds the fill byte
    std::vector<uint8_t> cand;
    std::vector<bool> written(n_cells, false);
    for (int64_t r = 0; r < n_rows; ++r) {
        cand.insert(cand.end(), out.begin() + idx_v[r] * bpr * tsz, out.begin() + (idx_v[r] + 1) * bpr * tsz);
        written[idx_v[r]] = true;
    }
    size_t stray = 0;
    for (int64_t r = 0; r < n_cells; ++r) {
        if (written[r]) continue;
        for (size_t b = 0; b < bpr * tsz; ++b) if (out[r * bpr * tsz + b] != 0xA5) ++stray;
    }
    const std::vector<uint8_t> ref = tw::quantize_ref(t, data);
    const tw::Verdict v = tw::judge(t, data, cand, ref);
    tw::print_verdict("GPU SET_ROWS vs host reference", v);
    expect(v.pass, "GPU-written rows match the host reference within the fixed thresholds");
    printf("  bytes in unindexed rows that changed: %zu\n", stray);
    expect(stray == 0, "rows not named by the index tensor are untouched");
}

// E --------------------------------------------------------------------------------------------------------------
static bool backend_claims_turbo4_writer(ggml_backend_t be) {
    Fixture f; f.ctx = new_ctx();
    ggml_tensor * x  = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 128, 4);
    ggml_tensor * d4 = ggml_new_tensor_2d(f.ctx, GGML_TYPE_TURBO4_0, 128, 4);
    return ggml_backend_supports_op(be, ggml_cpy(f.ctx, x, d4));
}

static void run_supports(ggml_backend_t be) {
    printf("supports_op\n");
    Fixture f; f.ctx = new_ctx();
    for (ggml_type t : { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0 }) {
        ggml_tensor * x  = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 128, 4);
        ggml_tensor * d  = ggml_new_tensor_2d(f.ctx, t, 128, 4);
        expect(ggml_backend_supports_op(be, ggml_cpy(f.ctx, x, d)), (std::string("CPY F32 -> ") + ggml_type_name(t) + " claimed").c_str());
        ggml_tensor * dst = ggml_new_tensor_2d(f.ctx, t, 128, 8);
        ggml_tensor * src = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 128, 4);
        ggml_tensor * idx = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I64, 4);
        expect(ggml_backend_supports_op(be, ggml_set_rows(f.ctx, dst, src, idx)),
               (std::string("SET_ROWS -> ") + ggml_type_name(t) + " claimed").c_str());
    }
    ggml_tensor * x  = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 128, 4);
    ggml_tensor * d4 = ggml_new_tensor_2d(f.ctx, GGML_TYPE_TURBO4_0, 128, 4);
    ggml_tensor * dst4 = ggml_new_tensor_2d(f.ctx, GGML_TYPE_TURBO4_0, 128, 8);
    ggml_tensor * src  = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 128, 4);
    ggml_tensor * idx  = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I64, 4);
    const bool cpy4  = ggml_backend_supports_op(be, ggml_cpy(f.ctx, x, d4));
    const bool rows4 = ggml_backend_supports_op(be, ggml_set_rows(f.ctx, dst4, src, idx));
    if (cpy4) {
        expect(rows4, "turbo4: a backend that claims CPY F32 -> turbo4 also claims SET_ROWS -> turbo4");
    } else {
        expect(!rows4, "CONTROL: CPY F32 -> turbo4 is NOT claimed (the backend has no turbo4 writer) and neither is SET_ROWS -> turbo4");
    }
}

int main(int argc, char ** argv) {
    const char * be_name = argc > 1 ? argv[1] : "CUDA0";
    tw::init_tables();
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(be_name, nullptr);
    if (!be) {
        printf("SKIPPED: backend %s could not be initialised in this build. Not a verdict.\n", be_name);
        return 0;
    }
    printf("backend %s\n", be_name);

    run_supports(be);
    std::vector<ggml_type> types = { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0 };
    if (backend_claims_turbo4_writer(be)) types.push_back(GGML_TYPE_TURBO4_0);
    for (ggml_type t : types) {
        const std::string n = ggml_type_name(t);
        run_cpy(be, t, Kind::K_STORE,    (n + " A  K store: cpy F32[128,n_head_kv,n_tokens] -> view_2d").c_str());
        run_cpy(be, t, Kind::V_STORE_1D, (n + " B  V store: cpy F32[256,n_tokens] -> view_1d").c_str());
        run_cpy(be, t, Kind::STRIDED_SRC,(n + " C  strided source view").c_str());
        run_set_rows(be, t,              (n + " D  SET_ROWS I64 indices").c_str());
    }

    ggml_backend_free(be);
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
