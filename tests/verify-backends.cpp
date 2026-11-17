// verify-backends.cpp -- model-level CROSS-BACKEND verify: greedy tokens + per-layer K/V divergence (+ MoE routing TV), built on cb_eval.
//
// Row S69-ENGREV-FERROX-BACKEND-VERIFY-LAYER-DIVERGENCE-2026-10-27-01.
// SOURCE-READ (design only, Apache-2.0, re-written here, no code copied): ferrox crates/frink-cli/src/verify.rs:1-225,
// verify_engine.rs:1-338, layer_divergence.rs:1-697, parity.rs:1-300 (sha 7bae13aa10abec683a9fb3410528af4b34b5dd29).
// test-backend-ops is per-op only; this is the whole-model question: do two backends produce the same model?
//
//   verify-backends run -m model.gguf [-ngl N] [-t THREADS] [--n-prompt 16] [--n-gen 8] [--seed 1] [--tokens-file f] -o out.vbk
//        ONE backend per process (a backend is process-lifetime). Records, to a text file:
//          greedy token ids, the top1-top2 logit margin at each step, and after ONE prefill the per-layer per-KV-head
//          L2 norm of the K and V activations (cb_eval "Kcur"/"Vcur") plus the mean MoE expert probability per layer.
//   verify-backends compare ref.vbk cand.vbk [--ref2 ref_again.vbk] [--tol 1e-3] [--tie-margin 0.02] [--moe-tol 0.05]
//        verdict ladder MATCH / DRIFT / TIEFLIP / WRONG. The WRONG band is MEASURED: with --ref2 (the reference run a
//        second time, e.g. other thread count) the noise spread is taken from ref-vs-ref and WRONG starts at
//        max(tol, 10 x that noise); without it the band is 10 x tol and the output says so.
//        exit: 0 MATCH/DRIFT/TIEFLIP, 1 WRONG, 3 input error (a side recorded nothing, vacuous zero magnitude, ...).
//   verify-backends --selftest      pure-logic controls on synthetic data (no model): a statistic that cannot fail proves nothing.
//
// The reference is the CPU build; the candidate is the same model on CUDA (-ngl 99) in another process. A prompt shorter
// than 8 tokens only exercises decode, never the batched prefill kernels: that prints a WARN.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------- data + pure logic (selftested)
struct vb_record {
    std::string backend;
    int n_prompt = 0;
    std::vector<int>    tok;
    std::vector<double> margin;
    std::map<int, std::vector<double>> k, v;   // layer -> per-kv-head L2 norm
    std::map<int, std::vector<double>> moe;    // layer -> mean expert probability
};

static const double VB_NAN = std::numeric_limits<double>::quiet_NaN();

// spread of the per-head ratio a/b: max(ratio) - min(ratio). Spread, not mean: a uniform scale error moves the mean
// and leaves the spread at 0, a per-head fault moves the spread. NaN = unusable (vacuous / size mismatch).
static double vb_ratio_spread(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.empty() || a.size() != b.size()) return VB_NAN;
    double lo = 1e300, hi = -1e300; bool any = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0.0 && b[i] == 0.0) continue;                 // both silent: carries no ratio
        if (a[i] == 0.0 || b[i] == 0.0) return 1e30;              // exactly one side silent on a head: maximal spread (a uniformly dead layer has equal ratios, so it is not left to the min/max)
        const double r = a[i] / b[i];
        lo = std::min(lo, r); hi = std::max(hi, r); any = true;
    }
    if (!any) return VB_NAN;                                       // zero magnitude everywhere = vacuous, caller errors
    return hi - lo;
}

static double vb_tv(const std::vector<double> & p, const std::vector<double> & q) {
    if (p.size() != q.size() || p.empty()) return VB_NAN;
    double s = 0; for (size_t i = 0; i < p.size(); ++i) s += std::fabs(p[i] - q[i]);
    return 0.5 * s;
}

struct vb_div { double max_spread = 0; int first_layer = -1; double first_spread = 0; double moe_tv = 0; int moe_layer = -1; std::string err; };

// per-layer comparison of two records; `tol` = spread at which a layer counts as divergent
static vb_div vb_layers(const vb_record & a, const vb_record & b, double tol) {
    vb_div d;
    if (a.k.empty() || b.k.empty() || a.v.empty() || b.v.empty()) { d.err = "NoCounts: one side recorded no K/V activations (that is not agreement)"; return d; }
    for (const auto & kv : a.k) {
        const int il = kv.first;
        auto kb = b.k.find(il), vbv = b.v.find(il), va = a.v.find(il);
        if (kb == b.k.end() || vbv == b.v.end() || va == a.v.end()) { d.err = "layer " + std::to_string(il) + " missing on one side"; return d; }
        const double sk = vb_ratio_spread(kv.second, kb->second), sv = vb_ratio_spread(va->second, vbv->second);
        if (std::isnan(sk) || std::isnan(sv)) { d.err = "layer " + std::to_string(il) + ": zero magnitude with positions recorded (vacuous pass refused) or head-count mismatch"; return d; }
        const double s = std::max(sk, sv);
        d.max_spread = std::max(d.max_spread, s);
        if (d.first_layer < 0 && s >= tol) { d.first_layer = il; d.first_spread = s; }
    }
    for (const auto & m : a.moe) {
        auto mb = b.moe.find(m.first); if (mb == b.moe.end()) continue;
        const double tv = vb_tv(m.second, mb->second);
        if (!std::isnan(tv) && tv > d.moe_tv) { d.moe_tv = tv; d.moe_layer = m.first; }
    }
    return d;
}

static int vb_first_token_diff(const vb_record & a, const vb_record & b) {
    const size_t n = std::min(a.tok.size(), b.tok.size());
    for (size_t i = 0; i < n; ++i) if (a.tok[i] != b.tok[i]) return (int) i;
    return a.tok.size() == b.tok.size() ? -1 : (int) n;
}

enum vb_verdict { VB_MATCH, VB_DRIFT, VB_TIEFLIP, VB_WRONG, VB_ERROR };
static const char * vb_name(vb_verdict v) { static const char * n[] = { "MATCH", "DRIFT", "TIEFLIP", "WRONG", "ERROR" }; return n[v]; }

struct vb_params { double tol = 1e-3; double tie_margin = 0.02; double moe_tol = 0.05; double noise = -1; };   // noise < 0: not measured

static double vb_wrong_band(const vb_params & p) { return p.noise >= 0 ? std::max(p.tol, 10.0 * p.noise) : 10.0 * p.tol; }

static vb_verdict vb_judge(const vb_record & ref, const vb_record & cand, const vb_params & p, vb_div & d, int & tok_diff, std::string & why) {
    d = vb_layers(ref, cand, p.tol);
    if (!d.err.empty()) { why = d.err; return VB_ERROR; }
    tok_diff = vb_first_token_diff(ref, cand);
    const double band = vb_wrong_band(p);
    const bool moe_wrong = d.moe_tv >= 4.0 * p.moe_tol, moe_drift = d.moe_tv >= p.moe_tol;
    if (d.max_spread >= band || moe_wrong) { why = "layer spread >= WRONG band"; return VB_WRONG; }
    if (tok_diff >= 0) {
        const double m = (size_t) tok_diff < ref.margin.size() ? ref.margin[tok_diff] : VB_NAN;
        if (std::isnan(m)) { why = "tokens diverge and the reference margin is unknown"; return VB_WRONG; }
        if (m < p.tie_margin) { why = "tokens diverge at a near-tie (reference margin " + std::to_string(m) + ")"; return VB_TIEFLIP; }
        why = "tokens diverge with reference margin " + std::to_string(m) + " >= tie margin"; return VB_WRONG;
    }
    if (d.max_spread >= p.tol || moe_drift) { why = "same tokens, numerics differ beyond tol"; return VB_DRIFT; }
    why = "same tokens, layer spread below tol"; return VB_MATCH;
}

// ---------------------------------------------------------------- record file (text, one fact per line)
static void vb_write(const vb_record & r, const std::string & path) {
    std::ofstream f(path);
    f.precision(17);
    f << "VBK 1\nbackend " << r.backend << "\nn_prompt " << r.n_prompt << "\ntok";
    for (int t : r.tok) f << ' ' << t;
    f << "\nmargin";
    for (double m : r.margin) f << ' ' << m;
    f << '\n';
    for (const auto & kv : r.k) { f << "K " << kv.first; for (double x : kv.second) f << ' ' << x; f << '\n'; }
    for (const auto & kv : r.v) { f << "V " << kv.first; for (double x : kv.second) f << ' ' << x; f << '\n'; }
    for (const auto & kv : r.moe) { f << "M " << kv.first; for (double x : kv.second) f << ' ' << x; f << '\n'; }
}

static bool vb_read(const std::string & path, vb_record & r) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    if (!std::getline(f, line) || line.rfind("VBK 1", 0) != 0) return false;
    while (std::getline(f, line)) {
        std::istringstream ss(line); std::string key; ss >> key;
        if (key == "backend") ss >> r.backend;
        else if (key == "n_prompt") ss >> r.n_prompt;
        else if (key == "tok") { int t; while (ss >> t) r.tok.push_back(t); }
        else if (key == "margin") { double m; while (ss >> m) r.margin.push_back(m); }
        else if (key == "K" || key == "V" || key == "M") {
            int il; ss >> il; std::vector<double> x; double v; while (ss >> v) x.push_back(v);
            (key == "K" ? r.k : key == "V" ? r.v : r.moe)[il] = x;
        }
    }
    return true;
}

// ---------------------------------------------------------------- selftest
static int selftest() {
    int fails = 0;
    auto check = [&](bool ok, const char * what) { printf("selftest %-86s : %s\n", what, ok ? "PASS" : "FAIL"); fails += !ok; };
    // controls on the statistic itself
    check(vb_ratio_spread({1, 2, 3}, {1, 2, 3}) == 0.0, "identical per-head norms -> spread exactly 0");
    check(std::fabs(vb_ratio_spread({2, 4, 6}, {1, 2, 3})) < 1e-12, "uniform 2x scale -> spread 0 (spread, not mean)");
    check(vb_ratio_spread({1, 2, 3.3}, {1, 2, 3}) > 0.09, "one head off by 10% -> spread ~0.1");
    check(std::isnan(vb_ratio_spread({0, 0}, {0, 0})), "all-zero magnitude -> NaN (vacuous), never a pass");
    check(vb_ratio_spread({1, 0}, {1, 1}) >= 1.0, "one side zero on a head -> large spread, not skipped");
    check(std::isnan(vb_ratio_spread({1, 2}, {1})), "head-count mismatch -> NaN");
    check(std::fabs(vb_tv({0.5, 0.5}, {0.5, 0.5})) == 0.0 && std::fabs(vb_tv({1, 0}, {0, 1}) - 1.0) < 1e-12, "TV distance 0 for equal, 1 for disjoint");

    auto mk = [](int n_layers, double bias, int bad_layer) {
        vb_record r; r.backend = "x"; r.n_prompt = 16; r.tok = {5, 6, 7, 8}; r.margin = {3, 3, 3, 3};
        for (int il = 0; il < n_layers; ++il) {
            std::vector<double> kk = {10.0 + il, 20.0 + il}, vv = {5.0 + il, 6.0 + il};
            if (il == bad_layer) { kk[1] *= (1.0 + bias * 200); }
            else { kk[1] *= (1.0 + bias); }
            r.k[il] = kk; r.v[il] = vv;
        }
        return r;
    };
    vb_params p; vb_div d; int td = -1; std::string why;
    const vb_record ref = mk(4, 0, -1);
    check(vb_judge(ref, mk(4, 0, -1), p, d, td, why) == VB_MATCH, "identical records -> MATCH");
    check(vb_judge(ref, mk(4, 2e-4, -1), p, d, td, why) == VB_MATCH, "noise below tol (2e-4) -> MATCH");
    check(vb_judge(ref, mk(4, 2e-3, -1), p, d, td, why) == VB_DRIFT, "spread 2e-3 >= tol, tokens equal -> DRIFT");
    vb_record bad = mk(4, 1e-4, 2);   // layer 2 is the planted fault: spread ~2e-2
    check(vb_judge(ref, bad, p, d, td, why) == VB_WRONG, "planted fault in layer 2 (spread 2e-2) -> WRONG");
    check(d.first_layer == 2, "the planted fault is NAMED: first divergent layer == 2 (layers 0,1 stay below tol)");
    // measured WRONG band: same fault, but a measured reference noise of 5e-3 puts the band at 5e-2 -> only DRIFT
    p.noise = 5e-3;
    check(vb_judge(ref, bad, p, d, td, why) == VB_DRIFT, "measured ref-vs-ref noise 5e-3 widens the WRONG band to 5e-2 -> DRIFT");
    p.noise = -1;
    // token divergence
    vb_record t1 = mk(4, 0, -1); t1.tok[2] = 99;
    check(vb_judge(ref, t1, p, d, td, why) == VB_WRONG && td == 2, "token flip with a large margin (3) at step 2 -> WRONG, step named");
    vb_record t2 = ref; t2.margin[2] = 0.001; t2.tok[2] = 99;
    check(vb_judge(t2, ref, p, d, td, why) == VB_TIEFLIP, "token flip where the reference margin is 0.001 -> TIEFLIP");
    // refusals
    vb_record empty; empty.backend = "none";
    check(vb_judge(ref, empty, p, d, td, why) == VB_ERROR, "a side that recorded nothing (NoCounts) -> ERROR, not agreement");
    vb_record zero = mk(4, 0, -1); zero.k[1] = {0, 0};
    check(vb_judge(ref, zero, p, d, td, why) != VB_MATCH, "zero-magnitude candidate layer never MATCHes");
    vb_record ms = mk(4, 0, -1); ms.moe[1] = {0.5, 0.5}; vb_record mc = mk(4, 0, -1); mc.moe[1] = {0.9, 0.1};
    check(vb_judge(ms, mc, p, d, td, why) == VB_WRONG && d.moe_layer == 1, "MoE routing TV 0.4 >= 4 x moe-tol -> WRONG, layer named");
    // file round trip
    vb_record rt = mk(3, 1e-4, -1); rt.moe[0] = {0.25, 0.75};
    vb_write(rt, "verify-backends-selftest.tmp"); vb_record back; const bool rok = vb_read("verify-backends-selftest.tmp", back);
    remove("verify-backends-selftest.tmp");
    check(rok && back.tok == rt.tok && back.k == rt.k && back.v == rt.v && back.moe == rt.moe && back.n_prompt == 16, "record file round trip is exact (17-digit doubles)");
    printf("selftest: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------- run mode (model, cb_eval capture)
static std::map<int, std::vector<double>> g_k, g_v, g_moe;
static bool g_capture = false;
static int  g_n_head_kv = 1;

static bool parse_layer(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || !name[n]) return false;
    for (const char * p = name + n; *p; ++p) if (*p < '0' || *p > '9') return false;
    il = atoi(name + n); return true;
}

static bool read_float(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    out.resize((size_t) n);
    for (int64_t i3 = 0, k = 0; i3 < t->ne[3]; ++i3) for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) for (int64_t i0 = 0; i0 < t->ne[0]; ++i0, ++k) {
        const size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->type == GGML_TYPE_F32) out[(size_t) k] = *(const float *) &raw[off];
        else if (t->type == GGML_TYPE_F16) out[(size_t) k] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &raw[off]);
        else return false;
    }
    return true;
}

static int eval_cb(struct ggml_tensor * t, bool ask, void *) {
    int il;
    const bool is_k = parse_layer(t->name, "Kcur_roped-", il) || parse_layer(t->name, "Kcur-", il);
    const bool is_v = !is_k && parse_layer(t->name, "Vcur-", il);
    const bool is_m = !is_k && !is_v && parse_layer(t->name, "ffn_moe_probs-", il);
    if (!(is_k || is_v || is_m)) return ask ? 0 : 1;
    if (ask) return g_capture ? 1 : 0;
    if (!g_capture || t->ne[3] != 1) return 1;
    std::vector<float> x; if (!read_float(t, x)) return 1;
    if (is_m) {   // [n_expert, n_tokens]: mean probability per expert over the prompt tokens
        const int64_t E = t->ne[0], T = t->ne[1]; std::vector<double> m((size_t) E, 0.0);
        for (int64_t j = 0; j < T; ++j) for (int64_t e = 0; e < E; ++e) m[(size_t) e] += x[(size_t) (j * E + e)] / (double) T;
        g_moe[il] = m; return 1;
    }
    // K: [D, H, T] (T > 1). V: [D, H, T] or the 2-D [Hkv*Dv, T] form; the raw K projection is also called Kcur in 2-D, skipped.
    int64_t D, H, T;
    if (t->ne[2] > 1) { D = t->ne[0]; H = t->ne[1]; T = t->ne[2]; }
    else if (is_v && t->ne[1] > 1 && g_n_head_kv > 0 && t->ne[0] % g_n_head_kv == 0) { H = g_n_head_kv; D = t->ne[0] / H; T = t->ne[1]; }
    else return 1;
    std::vector<double> nrm((size_t) H, 0.0);
    for (int64_t j = 0; j < T; ++j) for (int64_t h = 0; h < H; ++h) for (int64_t d = 0; d < D; ++d) {
        const double v = x[(size_t) (j * H * D + h * D + d)]; nrm[(size_t) h] += v * v;
    }
    for (double & v : nrm) v = std::sqrt(v);
    (is_k ? g_k : g_v)[il] = nrm;    // the last capture per layer wins (post-RoPE K)
    return 1;
}

static int top2(const float * lg, int n, double & margin) {
    int a = 0; for (int i = 1; i < n; ++i) if (lg[i] > lg[a]) a = i;
    int b = a == 0 ? 1 : 0; for (int i = 0; i < n; ++i) if (i != a && lg[i] > lg[b]) b = i;
    margin = (double) lg[a] - (double) lg[b]; return a;
}

static int run_mode(int argc, char ** argv) {
    std::string model, out, tokfile; int ngl = 0, threads = 4, n_prompt = 16, n_gen = 8; uint64_t seed = 1;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i]; auto nx = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model = nx(); else if (a == "-o") out = nx(); else if (a == "-ngl") ngl = atoi(nx()); else if (a == "-t") threads = atoi(nx());
        else if (a == "--n-prompt") n_prompt = atoi(nx()); else if (a == "--n-gen") n_gen = atoi(nx()); else if (a == "--seed") seed = (uint64_t) atoll(nx());
        else if (a == "--tokens-file") tokfile = nx();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || out.empty()) { fprintf(stderr, "usage: verify-backends run -m model.gguf -o out.vbk [-ngl N] [-t T] [--n-prompt N] [--n-gen N] [--seed S] [--tokens-file f]\n"); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = ngl;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(vocab);
    g_n_head_kv = llama_model_n_head_kv(m);
    std::vector<llama_token> prompt;
    if (!tokfile.empty()) { std::ifstream f(tokfile); int t; while (f >> t) prompt.push_back(t); }
    else { uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1; for (int i = 0; i < n_prompt; ++i) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; prompt.push_back((llama_token) (3 + s % (uint64_t) std::max(1, nv - 3))); } }
    if (prompt.size() < 8) fprintf(stderr, "WARN: prompt has %zu tokens (< 8): only decode is covered, the batched prefill kernels are never reached\n", prompt.size());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (prompt.size() + n_gen + 8); cp.n_batch = cp.n_ubatch = (uint32_t) prompt.size();
    cp.n_threads = cp.n_threads_batch = threads; cp.cb_eval = eval_cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(m, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
    vb_record r; r.backend = ngl > 0 ? "gpu" : "cpu"; r.n_prompt = (int) prompt.size();
    g_capture = true;
    llama_batch b = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }
    g_capture = false;
    r.k = g_k; r.v = g_v; r.moe = g_moe;
    for (int s = 0; s < n_gen; ++s) {
        double mg = 0; const int t = top2(llama_get_logits_ith(ctx, -1), nv, mg);
        r.tok.push_back(t); r.margin.push_back(mg);
        llama_token tt = t; llama_batch nb = llama_batch_get_one(&tt, 1);
        if (llama_decode(ctx, nb) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    vb_write(r, out);
    printf("verify-backends run: backend=%s prompt=%d gen=%d layers(K)=%zu layers(V)=%zu moe_layers=%zu -> %s\n", r.backend.c_str(), r.n_prompt, n_gen, r.k.size(), r.v.size(), r.moe.size(), out.c_str());
    llama_free(ctx); llama_model_free(m); llama_backend_free();
    return 0;
}

static int compare_mode(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: verify-backends compare ref.vbk cand.vbk [--ref2 f] [--tol x] [--tie-margin x] [--moe-tol x]\n"); return 2; }
    vb_params p; std::string ref2p;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i]; auto nx = [&]() -> const char * { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--ref2") ref2p = nx(); else if (a == "--tol") p.tol = atof(nx()); else if (a == "--tie-margin") p.tie_margin = atof(nx()); else if (a == "--moe-tol") p.moe_tol = atof(nx());
    }
    vb_record ref, cand, ref2;
    if (!vb_read(argv[2], ref) || !vb_read(argv[3], cand)) { fprintf(stderr, "ERROR: cannot read the record files\n"); return 3; }
    if (ref.n_prompt < 8) printf("WARN: reference prompt %d tokens (< 8): decode-only coverage, batched prefill kernels not reached\n", ref.n_prompt);
    if (!ref2p.empty()) {
        if (!vb_read(ref2p, ref2)) { fprintf(stderr, "ERROR: cannot read --ref2\n"); return 3; }
        vb_div dn = vb_layers(ref, ref2, p.tol);
        if (!dn.err.empty()) { printf("ERROR: ref-vs-ref: %s\n", dn.err.c_str()); return 3; }
        p.noise = dn.max_spread;
        printf("noise band MEASURED from reference-vs-reference: max layer spread %.3e (WRONG starts at %.3e)\n", dn.max_spread, vb_wrong_band(p));
    } else printf("noise band NOT measured (no --ref2): WRONG band = 10 x tol = %.3e\n", vb_wrong_band(p));
    vb_div d; int td = -1; std::string why;
    const vb_verdict v = vb_judge(ref, cand, p, d, td, why);
    if (v == VB_ERROR) { printf("VERDICT: ERROR %s\n", why.c_str()); return 3; }
    printf("layer spread max %.3e (tol %.1e); first divergent layer %d (spread %.3e); MoE routing TV max %.3e%s; first token difference at step %d of %zu\n",
           d.max_spread, p.tol, d.first_layer, d.first_spread, d.moe_tv, d.moe_layer >= 0 ? (" at layer " + std::to_string(d.moe_layer)).c_str() : "", td, ref.tok.size());
    printf("VERDICT: %s (%s)\n", vb_name(v), why.c_str());
    return v == VB_WRONG ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc >= 2 && strcmp(argv[1], "run") == 0) return run_mode(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "compare") == 0) return compare_mode(argc, argv);
    fprintf(stderr, "usage: verify-backends run ... | compare ... | --selftest\n");
    return 2;
}
