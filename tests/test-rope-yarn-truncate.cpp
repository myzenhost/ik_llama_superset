// YaRN RoPE: HF `truncate=false` correction range + mscale/mscale_all_dim attention-factor ratio.
// Row S69-EXO-YARN-TRUNCATE-FALSE-AND-MSCALE-RATIO-2026-11-06-01.
//
// What it checks (CPU backend, ggml graph, no model needed):
//   1. ggml_rope_ext with YaRN on a NEOX and a NORMAL rope, truncate=true and truncate=false (set per node with
//      ggml_rope_set_yarn_truncate and, separately, context-wide with ggml_set_rope_yarn_truncate), against an
//      independent double-precision implementation of transformers' _compute_yarn_parameters.
//   2. The reference's two truncate settings really differ at the tested positions (the test can tell them apart),
//      i.e. the pre-change behaviour (always floor/ceil) is a measurable MISMATCH against the truncate=false reference.
//   3. Flag plumbing: default stamps nothing; ext_factor == 0 never stamps; a no-truncate node reports it.
//   4. llama_yarn_attn_factor_ratio against hand-computed HF values.
//
// `--dump <file>` writes inputs/outputs of the truncate=false cases for tests/rope_yarn_truncate_ref.py (numpy).
#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-yarn.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const double YARN_PI = 3.14159265358979323846;
static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct yarn_case {
    const char * name;
    int    mode;        // GGML_ROPE_TYPE_NEOX or GGML_ROPE_TYPE_NORMAL
    int    n_dims;
    double base;
    double factor;      // HF rope_parameters["factor"]
    int    n_ctx_orig;  // HF original_max_position_embeddings
    double beta_fast;
    double beta_slow;
    double attn;        // HF attention_factor (what multiplies cos/sin)
};

// ---- independent reference: transformers modeling_rope_utils._compute_yarn_parameters (spec only) -------------
static double ref_corr_dim(double n_rot, int dim, double base, int max_pos) {
    return (dim * std::log(max_pos / (n_rot * 2.0 * YARN_PI))) / (2.0 * std::log(base));
}

// returns the rotated vector for one head: in[n_dims] -> out[n_dims]
static void ref_rope(const yarn_case & c, bool truncate, int pos, const float * in, double * out) {
    const int dim  = c.n_dims;
    const int half = dim / 2;

    double low  = ref_corr_dim(c.beta_fast, dim, c.base, c.n_ctx_orig);
    double high = ref_corr_dim(c.beta_slow, dim, c.base, c.n_ctx_orig);
    if (truncate) {
        low  = std::floor(low);
        high = std::ceil(high);
    }
    low  = std::fmax(low, 0.0);
    high = std::fmin(high, (double) (dim - 1));
    if (low == high) {
        high += 0.001;
    }

    std::vector<double> ang(half);
    for (int i = 0; i < half; i++) {
        const double pos_freq = std::pow(c.base, (2.0 * i) / dim);
        const double extrap   = 1.0 / pos_freq;
        const double interp   = 1.0 / (c.factor * pos_freq);
        double ramp = (i - low) / (high - low);
        ramp = std::fmin(1.0, std::fmax(0.0, ramp));
        const double extrap_factor = 1.0 - ramp;
        const double inv_freq = interp * (1.0 - extrap_factor) + extrap * extrap_factor;
        ang[i] = pos * inv_freq;
    }

    for (int i = 0; i < half; i++) {
        const double cs = std::cos(ang[i]) * c.attn;
        const double sn = std::sin(ang[i]) * c.attn;
        if (c.mode == GGML_ROPE_TYPE_NEOX) {            // rotate_half layout: (i, i + dim/2)
            const double x0 = in[i], x1 = in[i + half];
            out[i]        = x0 * cs - x1 * sn;
            out[i + half] = x0 * sn + x1 * cs;
        } else {                                         // interleaved: (2i, 2i + 1)
            const double x0 = in[2 * i], x1 = in[2 * i + 1];
            out[2 * i]     = x0 * cs - x1 * sn;
            out[2 * i + 1] = x0 * sn + x1 * cs;
        }
    }
}

// ---- ggml side --------------------------------------------------------------------------------------------------
enum set_how { SET_NODE, SET_CTX };

static void run_ggml(const yarn_case & c, bool truncate, set_how how, const std::vector<int> & pos,
                     const std::vector<float> & in, int n_head, std::vector<float> & out, int32_t * flag_out) {
    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    if (how == SET_CTX) {
        ggml_set_rope_yarn_truncate(ctx, truncate);
    }

    const int n_tok = (int) pos.size();
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_dims, n_head, n_tok);
    memcpy(a->data, in.data(), in.size() * sizeof(float));
    ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
    memcpy(p->data, pos.data(), pos.size() * sizeof(int));

    // ggml's rope_yarn multiplies the magnitude by 1 + 0.1 ln(1/freq_scale) whenever ext_factor != 0 (the
    // llama-context.cpp "cancel"), so the HF attention_factor A is passed as A / (1 + 0.1 ln(factor)).
    const float freq_scale = (float) (1.0 / c.factor);
    const float attn_arg   = (float) (c.attn / (1.0 + 0.1 * std::log(c.factor)));
    ggml_tensor * r = ggml_rope_ext(ctx, a, p, nullptr, c.n_dims, c.mode, c.n_ctx_orig, (float) c.base, freq_scale,
                                    /*ext_factor*/ 1.0f, attn_arg, (float) c.beta_fast, (float) c.beta_slow);
    if (how == SET_NODE) {
        ggml_rope_set_yarn_truncate(r, truncate);
    }
    *flag_out = r->op_params[3];

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_cplan plan = ggml_graph_plan(gf, 2, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(gf, &plan);

    out.assign((float *) r->data, (float *) r->data + ggml_nelements(r));
    ggml_free(ctx);
}

static double max_err(const std::vector<float> & g, const std::vector<double> & ref) {
    double m = 0.0;
    for (size_t i = 0; i < g.size(); i++) {
        m = std::fmax(m, std::fabs((double) g[i] - ref[i]));
    }
    return m;
}

int main(int argc, char ** argv) {
    const char * dump_path = nullptr;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--dump")) {
            dump_path = argv[i + 1];
        }
    }
    FILE * dump = dump_path ? fopen(dump_path, "w") : nullptr;

    // attention factors the way HF computes them (ratio of get_mscale), see llama-yarn.h
    const double f_gptoss = 32.0;
    const double f_dsv    = 40.0;
    const double A_gptoss = 0.1 * std::log(f_gptoss) + 1.0;                                       // mscale/mscale_all_dim absent
    const double A_dsv    = (0.1 * 0.707 * std::log(f_dsv) + 1.0) / (0.1 * 1.0 * std::log(f_dsv) + 1.0); // mscale != mscale_all_dim

    // (4) mscale ratio helper vs the hand-computed HF values
    {
        CHECK(std::fabs(llama_yarn_attn_factor_ratio((float) f_gptoss, 0.0f, 0.0f) - A_gptoss) < 1e-6, "ratio, keys absent");
        CHECK(std::fabs(llama_yarn_attn_factor_ratio((float) f_dsv, 0.707f, 1.0f) - A_dsv) < 1e-6, "ratio, 0.707/1.0");
        CHECK(std::fabs(llama_yarn_attn_factor_ratio((float) f_dsv, 1.0f, 1.0f) - 1.0) < 1e-6, "ratio, mscale == mscale_all_dim must be 1");
        // HF falsy test: mscale set but mscale_all_dim unset -> plain get_mscale(factor)
        CHECK(std::fabs(llama_yarn_attn_factor_ratio((float) f_dsv, 0.707f, 0.0f) - (0.1 * std::log(f_dsv) + 1.0)) < 1e-6, "ratio, only mscale");
        CHECK(llama_yarn_attn_factor_ratio(1.0f, 0.707f, 1.0f) == 1.0f, "factor <= 1 -> 1.0");
        printf("ratio helper: A_gptoss=%.6f A_dsv(0.707/1.0)=%.6f\n", A_gptoss, A_dsv);
    }

    const yarn_case cases[] = {
        // gpt-oss-20b: head_dim 64, theta 150000, yarn factor 32 over 4096, beta 32/1, truncate=false
        { "gptoss_neox",  GGML_ROPE_TYPE_NEOX,   64, 150000.0, f_gptoss, 4096, 32.0, 1.0, A_gptoss },
        // DeepSeek-V3 style rope sub-dim 64, theta 10000, factor 40 over 4096, interleaved, mscale != mscale_all_dim
        { "dsv_normal",   GGML_ROPE_TYPE_NORMAL, 64, 10000.0,  f_dsv,    4096, 32.0, 1.0, A_dsv },
        // a wider head so the correction range is not clamped at an end
        { "wide_neox",    GGML_ROPE_TYPE_NEOX,  128, 500000.0, f_gptoss, 8192, 32.0, 1.0, A_gptoss },
    };

    const std::vector<int> pos = { 0, 1, 2, 63, 1000, 2047, 4095, 5000, 8191 };
    const int n_head = 2;
    const double TOL = 3e-3; // f32 trig at |angle| <= ~8k rad

    for (const yarn_case & c : cases) {
        std::vector<float> in((size_t) c.n_dims * n_head * pos.size());
        for (size_t i = 0; i < in.size(); i++) {
            in[i] = (float) (std::sin(0.37 * (double) i) + 0.5);
        }

        // references for every token/head
        std::vector<double> ref_t(in.size()), ref_n(in.size());
        for (size_t t = 0; t < pos.size(); t++) {
            for (int h = 0; h < n_head; h++) {
                const size_t off = (t * n_head + h) * c.n_dims;
                ref_rope(c, true,  pos[t], in.data() + off, ref_t.data() + off);
                ref_rope(c, false, pos[t], in.data() + off, ref_n.data() + off);
            }
        }

        double ref_gap = 0.0;
        for (size_t i = 0; i < in.size(); i++) {
            ref_gap = std::fmax(ref_gap, std::fabs(ref_t[i] - ref_n[i]));
        }

        std::vector<float> g_default, g_node_t, g_node_n, g_ctx_n, g_ctx_t;
        int32_t fl_default, fl_node_t, fl_node_n, fl_ctx_n, fl_ctx_t;
        // default (no call at all) = the historical floor/ceil
        {
            ggml_init_params ip = { 64u * 1024 * 1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.n_dims, n_head, (int) pos.size());
            memcpy(a->data, in.data(), in.size() * sizeof(float));
            ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int) pos.size());
            memcpy(p->data, pos.data(), pos.size() * sizeof(int));
            ggml_tensor * r = ggml_rope_ext(ctx, a, p, nullptr, c.n_dims, c.mode, c.n_ctx_orig, (float) c.base, (float) (1.0 / c.factor),
                                            1.0f, (float) (c.attn / (1.0 + 0.1 * std::log(c.factor))), (float) c.beta_fast, (float) c.beta_slow);
            fl_default = r->op_params[3];
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, r);
            ggml_cplan plan = ggml_graph_plan(gf, 2, nullptr);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            ggml_graph_compute(gf, &plan);
            g_default.assign((float *) r->data, (float *) r->data + ggml_nelements(r));
            CHECK(ggml_rope_yarn_truncate(r), "%s: untouched node must report truncate=true", c.name);
            ggml_free(ctx);
        }
        run_ggml(c, true,  SET_NODE, pos, in, n_head, g_node_t, &fl_node_t);
        run_ggml(c, false, SET_NODE, pos, in, n_head, g_node_n, &fl_node_n);
        run_ggml(c, false, SET_CTX,  pos, in, n_head, g_ctx_n,  &fl_ctx_n);
        run_ggml(c, true,  SET_CTX,  pos, in, n_head, g_ctx_t,  &fl_ctx_t);

        const double e_def   = max_err(g_default, ref_t);
        const double e_nt    = max_err(g_node_t, ref_t);
        const double e_nn    = max_err(g_node_n, ref_n);
        const double e_cn    = max_err(g_ctx_n,  ref_n);
        const double e_ct    = max_err(g_ctx_t,  ref_t);
        const double e_before = max_err(g_default, ref_n); // old behaviour vs the truncate=false reference

        printf("%-12s ref(trunc vs no-trunc) gap %.4f | ggml default vs ref_trunc %.2e | node t %.2e n %.2e | ctx n %.2e t %.2e | "
               "BEFORE (floor/ceil vs HF truncate=false) %.4f | flags def=%d nodeT=%d nodeN=%d ctxN=%d ctxT=%d\n",
               c.name, ref_gap, e_def, e_nt, e_nn, e_cn, e_ct, e_before, fl_default, fl_node_t, fl_node_n, fl_ctx_n, fl_ctx_t);

        CHECK(ref_gap > 20 * TOL, "%s: test cannot tell truncate modes apart (gap %.4g)", c.name, ref_gap);
        CHECK(e_def < TOL, "%s: default node must equal the truncate=true reference (%.4g)", c.name, e_def);
        CHECK(e_nt  < TOL, "%s: node truncate=true  (%.4g)", c.name, e_nt);
        CHECK(e_nn  < TOL, "%s: node truncate=false (%.4g)", c.name, e_nn);
        CHECK(e_cn  < TOL, "%s: ctx  truncate=false (%.4g)", c.name, e_cn);
        CHECK(e_ct  < TOL, "%s: ctx  truncate=true  (%.4g)", c.name, e_ct);
        CHECK(e_before > 20 * TOL, "%s: floor/ceil must MISMATCH the truncate=false reference (%.4g)", c.name, e_before);
        CHECK(fl_default == 0 && fl_node_t == 0 && fl_ctx_t == 0, "%s: truncate=true must leave op_params[3] == 0", c.name);
        CHECK(fl_node_n == 1 && fl_ctx_n == 1, "%s: truncate=false must set op_params[3] bit 0", c.name);

        if (dump) {
            fprintf(dump, "case %s %d %d %.17g %.17g %d %.17g %.17g %.17g %d %d\n", c.name, c.mode, c.n_dims, c.base, c.factor,
                    c.n_ctx_orig, c.beta_fast, c.beta_slow, c.attn, n_head, (int) pos.size());
            fprintf(dump, "pos"); for (int p : pos) fprintf(dump, " %d", p); fprintf(dump, "\n");
            fprintf(dump, "in");  for (float v : in) fprintf(dump, " %.9g", v); fprintf(dump, "\n");
            fprintf(dump, "out_trunc_false"); for (float v : g_node_n) fprintf(dump, " %.9g", v); fprintf(dump, "\n");
            fprintf(dump, "out_trunc_true");  for (float v : g_node_t) fprintf(dump, " %.9g", v); fprintf(dump, "\n");
        }
    }

    // (3) a node without YaRN (ext_factor == 0) is never stamped, even under a no-truncate context
    {
        ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_set_rope_yarn_truncate(ctx, false);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 4);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_tensor * plain = ggml_rope_ext(ctx, a, p, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * yarn  = ggml_rope_ext(ctx, a, p, nullptr, 64, GGML_ROPE_TYPE_NEOX, 4096, 10000.0f, 0.5f, 1.0f, 1.0f, 32.0f, 1.0f);
        CHECK(plain->op_params[3] == 0 && ggml_rope_yarn_truncate(plain), "ext_factor == 0 node must not be stamped");
        CHECK(yarn->op_params[3] == 1 && !ggml_rope_yarn_truncate(yarn), "YaRN node under a no-truncate ctx must be stamped");
        ggml_rope_set_yarn_truncate(yarn, true);
        CHECK(yarn->op_params[3] == 0, "set_yarn_truncate(true) clears the bit");
        ggml_free(ctx);
    }

    if (dump) {
        fclose(dump);
    }
    printf(g_fail ? "test-rope-yarn-truncate: %d FAILED\n" : "test-rope-yarn-truncate: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
