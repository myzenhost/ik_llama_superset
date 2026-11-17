// llama_model_quantize on a tiny synthetic GGUF (arch "llama", F32 tensors written with gguf_write_to_file):
//
//   #30820 (upstream 5c965fd8c, src/llama-quant.cpp)
//     1. a 4D tensor (ne[3] > 1) is quantized in ALL its rows -- every slice, dequantized, matches the source. Before the fix the
//        chunked path quantized ne[2] slices only and the output was short / garbage past them.
//     2. MXFP4_MOE picks MXFP4 by tensor NAME ("*_exps." / "*_chexps."), not by ne[2] > 1: a non-expert 3D / 4D tensor is Q8_0.
//     3. the imatrix of a 4D tensor holds ne[0]*ne[2]*ne[3] weights (one n_per_row vector per slice); the size check accepts that.
//   #27795 / working-memory cap
//     4. a run with a tiny slab_size / max_buf_size (the tensors are quantized slab by slab) is byte-identical to the uncapped run.
//
// usage: test-quantize-4d     (writes small .gguf files in the cwd, removes them)
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL  %s:%d  %s  --  ", __FILE__, __LINE__, #cond); printf(__VA_ARGS__); printf("\n"); } else { printf("ok    %s\n", #cond); } } while (0)

struct tspec { const char * name; int64_t ne[4]; };
// n_per_row 64 (multiple of every block size used here: Q8_0 32, MXFP4 32, Q4_0 32)
static const tspec specs[] = {
    { "blk.0.conv4d.weight",       { 64, 4, 3, 2 } },   // 4D, non-expert: 6 slices of 4 rows
    { "blk.0.attn_proj3d.weight",  { 64, 4, 3, 1 } },   // 3D, NOT an expert tensor (ne[2] > 1)
    { "blk.0.ffn_up_exps.weight",  { 64, 4, 3, 1 } },   // 3D expert tensor
    { "blk.0.ffn_down.weight",     { 64, 8, 1, 1 } },   // plain 2D
};

static float src_value(int tensor_id, int64_t i) {
    // deterministic, spread over [-1, 1], different for every element
    uint32_t x = (uint32_t) (i * 2654435761u) ^ (uint32_t) (tensor_id * 40503u + 12345u);
    x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    return ((x & 0xFFFF) / 32767.5f) - 1.0f;
}

static void write_src(const char * path) {
    ggml_init_params ip = { /*mem_size*/ 4 * 1024 * 1024, nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    int id = 0;
    for (const tspec & s : specs) {
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, s.ne);
        ggml_set_name(t, s.name);
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); i++) { d[i] = src_value(id, i); }
        gguf_add_tensor(g, t);
        id++;
    }
    gguf_write_to_file(g, path, false);
    gguf_free(g);
    ggml_free(ctx);
}

static std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct quant_out { ggml_type type = GGML_TYPE_COUNT; int64_t n_rows = 0; double rel_rmse_worst = 1e9; double worst_slice = -1; bool size_ok = false; bool present = false; };

// read OUT, dequantize tensor `id`, per-slice relative RMSE against the source values
static quant_out check_tensor(gguf_context * g, ggml_context * ctx, int id) {
    quant_out q;
    const tspec & s = specs[id];
    ggml_tensor * t = ggml_get_tensor(ctx, s.name);
    if (!t) { return q; }
    q.present = true;
    q.type = t->type;
    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    const int64_t nrows = ggml_nrows(t), n = t->ne[0];
    q.n_rows = nrows;
    const int64_t gi = gguf_find_tensor(g, s.name);
    const size_t want = ggml_row_size(t->type, n) * nrows;
    q.size_ok = gi >= 0 && gguf_get_tensor_size(g, gi) == want;
    const int64_t rows_per_slice = t->ne[1];
    const int64_t nslices = nrows / rows_per_slice;
    std::vector<float> row(n);
    double worst = 0.0; int64_t worst_s = -1;
    for (int64_t sl = 0; sl < nslices; sl++) {
        double se = 0.0, ss = 0.0;
        for (int64_t r = 0; r < rows_per_slice; r++) {
            const int64_t gr = sl*rows_per_slice + r;
            if (tr->to_float) {
                tr->to_float((const char *) t->data + gr*ggml_row_size(t->type, n), row.data(), n);
            } else {
                memcpy(row.data(), (const char *) t->data + gr*n*sizeof(float), n*sizeof(float));
            }
            for (int64_t k = 0; k < n; k++) {
                const double a = src_value(id, gr*n + k);
                const double d = row[k] - a;
                se += d*d; ss += a*a;
            }
        }
        const double rel = std::sqrt(se / ss);
        if (rel > worst || worst_s < 0) { worst = rel; worst_s = sl; }
    }
    q.rel_rmse_worst = worst; q.worst_slice = (double) worst_s;
    return q;
}


static bool run_quant(const char * in, const char * out, llama_ftype ftype, int nthread, size_t slab, size_t max_buf,
                      std::unordered_map<std::string, std::vector<float>> * imat) {
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype = ftype;
    qp.nthread = nthread;
    qp.slab_size = slab;
    qp.max_buf_size = max_buf;
    qp.imatrix_raw = imat;
    qp.ignore_imatrix_rules = true;
    const uint32_t rc = llama_model_quantize(in, out, &qp);
    return rc == 0;
}

static void report(const char * label, const char * out) {
    gguf_init_params ip = { /*no_alloc*/ false, nullptr };
    ggml_context * ctx = nullptr;
    ip.ctx = &ctx;
    gguf_context * g = gguf_init_from_file(out, ip);
    CHECK(g != nullptr, "%s: output readable", label);
    if (!g) { return; }
    for (int id = 0; id < (int) (sizeof(specs)/sizeof(specs[0])); id++) {
        quant_out q = check_tensor(g, ctx, id);
        printf("      %-34s type=%-6s rows=%lld size_ok=%d worst_slice_rel_rmse=%.4f (slice %d)\n", specs[id].name,
               q.present ? ggml_type_name(q.type) : "-", (long long) q.n_rows, (int) q.size_ok, q.rel_rmse_worst, (int) q.worst_slice);
    }
    gguf_free(g); ggml_free(ctx);
}

static void verify(const char * label, const char * out, llama_ftype ftype, double tol) {
    gguf_init_params ip = { /*no_alloc*/ false, nullptr };
    ggml_context * ctx = nullptr;
    ip.ctx = &ctx;
    gguf_context * g = gguf_init_from_file(out, ip);
    if (!g) { CHECK(false, "%s: output readable", label); return; }
    for (int id = 0; id < (int) (sizeof(specs)/sizeof(specs[0])); id++) {
        quant_out q = check_tensor(g, ctx, id);
        CHECK(q.present, "%s: %s present", label, specs[id].name);
        if (!q.present) { continue; }
        CHECK(q.size_ok, "%s: %s stored size == row_size*nrows", label, specs[id].name);
        CHECK(q.rel_rmse_worst < tol, "%s: %s worst slice %d rel RMSE %.4f < %.2f (all %lld rows quantized)", label, specs[id].name,
              (int) q.worst_slice, q.rel_rmse_worst, tol, (long long) q.n_rows);
        if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
            const bool expert = strstr(specs[id].name, "_exps.") != nullptr;
            const ggml_type want = expert ? GGML_TYPE_MXFP4 : GGML_TYPE_Q8_0;
            CHECK(q.type == want, "%s: %s is %s, want %s", label, specs[id].name, ggml_type_name(q.type), ggml_type_name(want));
        }
    }
    gguf_free(g); ggml_free(ctx);
}

int main() {
    const char * src = "test-quantize-4d.src.gguf";
    const char * o1  = "test-quantize-4d.o1.gguf";
    const char * o2  = "test-quantize-4d.o2.gguf";
    write_src(src);

    // 1. Q8_0, serial (nthread 1) and threaded (nthread 3: ne[2] % 3 == 0 -> the per-slice thread path)
    for (int nthread : { 1, 3 }) {
        char label[64]; snprintf(label, sizeof(label), "Q8_0 nthread=%d", nthread);
        const bool ok = run_quant(src, o1, LLAMA_FTYPE_MOSTLY_Q8_0, nthread, 0, 0, nullptr);
        CHECK(ok, "%s: llama_model_quantize returned 0", label);
        if (ok) { report(label, o1); verify(label, o1, LLAMA_FTYPE_MOSTLY_Q8_0, 0.02); }
    }

    // 2. MXFP4_MOE: expert by NAME. MXFP4 is a 4-bit float: worst slice error is much larger than Q8_0
    {
        const bool ok = run_quant(src, o1, LLAMA_FTYPE_MOSTLY_MXFP4_MOE, 1, 0, 0, nullptr);
        CHECK(ok, "MXFP4_MOE: llama_model_quantize returned 0");
        if (ok) { report("MXFP4_MOE", o1); verify("MXFP4_MOE", o1, LLAMA_FTYPE_MOSTLY_MXFP4_MOE, 0.25); }
    }

    // 3. imatrix: one n_per_row vector per slice (ne[2]*ne[3] for the 4D tensor, ne[2] for the 3D ones, 1 for the 2D one)
    {
        std::unordered_map<std::string, std::vector<float>> imat;
        for (const tspec & s : specs) {
            std::vector<float> v((size_t) (s.ne[0]*s.ne[2]*s.ne[3]));
            for (size_t i = 0; i < v.size(); i++) { v[i] = 0.5f + (float) ((i * 7) % 13); }
            imat[s.name] = v;
        }
        const bool ok = run_quant(src, o1, LLAMA_FTYPE_MOSTLY_Q4_0, 1, 0, 0, &imat);
        CHECK(ok, "Q4_0 + imatrix(ne0*ne2*ne3): llama_model_quantize returned 0");
        if (ok) { report("Q4_0 + imatrix", o1); verify("Q4_0 + imatrix", o1, LLAMA_FTYPE_MOSTLY_Q4_0, 0.25); }
    }

    // 4. capped working memory: slab_size / max_buf_size tiny -> slab-by-slab, bytes identical to the uncapped run
    {
        const bool ok0 = run_quant(src, o1, LLAMA_FTYPE_MOSTLY_Q8_0, 1, 0, 0, nullptr);
        const bool ok1 = run_quant(src, o2, LLAMA_FTYPE_MOSTLY_Q8_0, 1, 1024, 1024, nullptr);
        CHECK(ok0 && ok1, "capped and uncapped runs return 0");
        if (ok0 && ok1) {
            const std::string a = slurp(o1), b = slurp(o2);
            CHECK(!a.empty() && a == b, "capped (slab_size=max_buf_size=1 KiB) output byte-identical to uncapped (%zu vs %zu bytes)", a.size(), b.size());
        }
        const bool ok2 = run_quant(src, o2, LLAMA_FTYPE_MOSTLY_Q8_0, 1, 0, 1024, nullptr);
        CHECK(ok2, "max_buf_size alone returns 0");
        if (ok0 && ok2) {
            const std::string a = slurp(o1), b = slurp(o2);
            CHECK(!a.empty() && a == b, "max_buf_size-only capped output byte-identical to uncapped (%zu vs %zu bytes)", a.size(), b.size());
        }
    }

    // 5. #27830 single pass over all rows of all experts: thread count must not change a byte (nthread 2 / 4 take the shared chunk queue
    //    -- ne[2] = 3 matrices is neither a multiple of 2 nor >= 8 -- while 3 takes the per-slice path), with a per-slice imatrix
    {
        std::unordered_map<std::string, std::vector<float>> imat;
        for (const tspec & s : specs) {
            std::vector<float> v((size_t) (s.ne[0]*s.ne[2]*s.ne[3]));
            for (size_t i = 0; i < v.size(); i++) { v[i] = 0.25f + (float) ((i * 5 + i / s.ne[0]) % 11); }
            imat[s.name] = v;
        }
        const bool ok1 = run_quant(src, o1, LLAMA_FTYPE_MOSTLY_Q4_0, 1, 0, 0, &imat);
        CHECK(ok1, "Q4_0 + imatrix nthread=1 returns 0");
        const std::string ref = ok1 ? slurp(o1) : std::string();
        for (int nthread : { 2, 3, 4 }) {
            const bool ok = run_quant(src, o2, LLAMA_FTYPE_MOSTLY_Q4_0, nthread, 0, 0, &imat);
            const std::string b = ok ? slurp(o2) : std::string();
            CHECK(ok && !ref.empty() && b == ref, "Q4_0 + imatrix nthread=%d byte-identical to nthread=1 (%zu vs %zu bytes)", nthread, b.size(), ref.size());
        }
    }

    // 6. per-architecture tensor rules (upstream #30616 Muse Spark, #31086 DSV4.1, #31028 qwen4exp), by tensor type in the output
    {
        struct mini { const char * name; int64_t n0, n1; };
        struct expect { const char * name; ggml_type type; };
        struct arch_case { const char * arch; llama_ftype ftype; std::vector<mini> tensors; std::vector<expect> want; };
        const std::vector<arch_case> cases = {
            { "muse-spark", LLAMA_FTYPE_MOSTLY_Q4_0,
              { { "blk.0.ffn_latent_down.weight", 64, 4 }, { "blk.0.attn_gate.weight", 64, 4 }, { "blk.0.attn_output.weight", 64, 4 } },
              { { "blk.0.ffn_latent_down.weight", GGML_TYPE_Q8_0 }, { "blk.0.attn_gate.weight", GGML_TYPE_Q8_0 },
                { "blk.0.attn_output.weight", GGML_TYPE_Q4_0 } } },
            { "llama", LLAMA_FTYPE_MOSTLY_Q4_0,   // control: the bump is Muse Spark only
              { { "blk.0.ffn_latent_down.weight", 64, 4 }, { "blk.0.attn_gate.weight", 64, 4 } },
              { { "blk.0.ffn_latent_down.weight", GGML_TYPE_Q4_0 }, { "blk.0.attn_gate.weight", GGML_TYPE_Q4_0 } } },
            { "deepseek41", LLAMA_FTYPE_MOSTLY_Q4_0,
              { { "blk.0.engram_gate.weight", 64, 4 }, { "blk.0.attn_output.weight", 64, 4 } },
              { { "blk.0.engram_gate.weight", GGML_TYPE_F32 }, { "blk.0.attn_output.weight", GGML_TYPE_Q4_0 } } },
            { "qwen4exp", LLAMA_FTYPE_MOSTLY_Q4_K_M,   // parity: attn_qkv_gate gets the same type as attn_qkv (both are caught by the attn_q arm of the ladder)
              { { "blk.0.attn_qkv.weight", 256, 4 }, { "blk.0.attn_qkv_gate.weight", 256, 4 }, { "blk.0.attn_output.weight", 256, 4 } },
              { { "blk.0.attn_qkv.weight", GGML_TYPE_Q4_K }, { "blk.0.attn_qkv_gate.weight", GGML_TYPE_Q4_K } } },
        };
        for (const arch_case & c : cases) {
            ggml_init_params ip = { /*mem_size*/ 4 * 1024 * 1024, nullptr, /*no_alloc*/ false };
            ggml_context * ctx = ggml_init(ip);
            gguf_context * g = gguf_init_empty();
            gguf_set_val_str(g, "general.architecture", c.arch);
            int id = 100;
            for (const mini & m : c.tensors) {
                ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.n0, m.n1);
                ggml_set_name(t, m.name);
                for (int64_t i = 0; i < ggml_nelements(t); i++) { ((float *) t->data)[i] = src_value(id, i); }
                gguf_add_tensor(g, t);
                id++;
            }
            gguf_write_to_file(g, src, false);
            gguf_free(g); ggml_free(ctx);

            const bool ok = run_quant(src, o1, c.ftype, 1, 0, 0, nullptr);
            CHECK(ok, "arch %s: llama_model_quantize returned 0", c.arch);
            if (!ok) { continue; }
            gguf_init_params rp = { /*no_alloc*/ true, nullptr };
            gguf_context * og = gguf_init_from_file(o1, rp);
            CHECK(og != nullptr, "arch %s: output readable", c.arch);
            if (!og) { continue; }
            for (const expect & e : c.want) {
                const int64_t ti = gguf_find_tensor(og, e.name);
                const ggml_type got = ti >= 0 ? gguf_get_tensor_type(og, ti) : GGML_TYPE_COUNT;
                CHECK(ti >= 0 && got == e.type, "arch %s: %s is %s, want %s", c.arch, e.name,
                      ti >= 0 ? ggml_type_name(got) : "(missing)", ggml_type_name(e.type));
            }
            gguf_free(og);
        }
    }

    remove(src); remove(o1); remove(o2);
    printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
