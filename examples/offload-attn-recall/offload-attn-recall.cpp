// offload-attn-recall: ATTENTION-MASS RECALL of the --offload-attn page selector against an oracle, on REAL q/K from a model.
//
// Row S69-HIPATTENTION-PICK-ATTENTION-MASS-RECALL-HARNESS-2026-10-05-01. Behaviour spec from DeepAuto-AI/hip-attention @ 5f37a7fd7
// (FSL-1.1-MIT, source-available, NOT OSI): hip-research plot_topk_recall.py scores a selector by the fraction of the true softmax mass of
// one query that lands on the selected keys. Written from that description; no code copied.
//
// WHAT IT MEASURES. The FA-path op (ik_offload::mask_from_kq_op) keeps ONE page set shared by every query head. For a query at position
// `pos` with keys 0..pos:
//   mass[h][p]  = sum over keys s in page p of softmax_s(q_h . k_{kv(h),s} * scale)     (exact, double precision, GROUPED kv mapping)
//   recall(set) = mean over heads h of sum_{p in set} mass[h][p]
// next to two bounds AT THE SAME NUMBER OF KEPT PAGES (the union keeps more than its nominal budget, so equal NOMINAL budget is not a fair
// comparison -- measured 2026-10-05: 2 nominal pages per head became 7 kept):
//   oracle = the best SHARED set of that size (top pages by head-averaged mass): no shared selection can beat it
//   random = n_kept / n_pages (expected recall of a random set of that size)
// and "lift" = (recall - random) / (oracle - random): the fraction of the achievable gain over random that the selector captured.
//
// METHODS (each runs the REAL op code, compiled into this binary, on host tensors built from the captured q/K):
//   union      head_reduce=0 (default)                     max      head_reduce=1 at the nominal budget
//   max@union  head_reduce=1 at budget = pages union kept  (MAX vs UNION at EQUAL ACTUAL pages)
//   interleave union with the pre-eda065854 mapping kv = h % n_kv (GQA bug), emulated by permuting query heads so that the fixed op's
//              grouped mapping h'/g lands on h % n_kv; recall is still scored with the TRUE grouped mapping
//   scan=bisect  MAX with the hierarchical page scan (OffloadConfig::scan_mode = 1), compared with "max" (exact scan) at the same budget;
//              keys_frac = keys the bisect scan scored / keys the exact scan scored (both measured by the op's own counter, under a lock)
//   refresh=R  MAX selection made R-1 steps EARLIER (at pos-R+1, the oldest selection refresh R ever reuses) and reused at pos the way the
//              op reuses it: that selection, plus the pages of the keys written since, plus the sink. Scored against pos's true mass.
//              This replays the reuse rule; the op's step-by-step refresh state machine itself is covered by test-offload-mask-gqa.
//
// SELF-TEST (--selftest, no model, a ctest): the oracle at every page scores exactly 1, each head's mass sums to 1, the random expectation
// is n/nP and a realised random set averages near it, the op's union beats random on peaked data, and the head permutation reproduces an
// independent plain-loop INTERLEAVED reference (so the GQA-bug arm measures what it claims).
//
// Usage: offload-attn-recall --selftest
//        offload-attn-recall -m model.gguf -f text.txt [-c 8192] [--layers 2,12,22] [--queries 32] [--page 16] [--budgets 4,8,16,32]
//                            [--sink N] [--recent N] [--refresh R] [--scan] [-t 4]
// PROGRESS-OK: prints one line per decoded batch and per (layer, budget); bounded by -c and --queries.

#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../src/offload_attention.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static void putenv_(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// One layer's captured tensors. K is stored as F16 because that is what the op reads from the live cache (type_k defaults to F16).
struct LayerQK {
    int hd = 0, n_head = 0, n_kvh = 0, n_tok = 0;
    std::vector<float>       q;   // [n_tok][n_head][hd]
    std::vector<ggml_fp16_t> k;   // [n_kvh][n_tok][hd]  (the op's b layout: hd fastest, then position, then kv head)
    float kf(int kh, int s, int d) const { return ggml_fp16_to_fp32(k[((size_t) kh * n_tok + s) * hd + d]); }
};

// mass[h][p] for the query at `pos`, keys 0..pos, grouped mapping.
static std::vector<std::vector<double>> page_mass(const LayerQK & L, int pos, int P, double scale) {
    const int n_kv = pos + 1, nP = (n_kv + P - 1) / P, g = L.n_head / L.n_kvh;
    std::vector<std::vector<double>> mass(L.n_head, std::vector<double>(nP, 0.0));
    std::vector<double> lg(n_kv);
    for (int h = 0; h < L.n_head; ++h) {
        const int kh = h / g;
        const float * qh = &L.q[((size_t) pos * L.n_head + h) * L.hd];
        double mx = -INFINITY;
        for (int s = 0; s < n_kv; ++s) {
            double dot = 0.0;
            for (int d = 0; d < L.hd; ++d) dot += (double) qh[d] * (double) L.kf(kh, s, d);
            lg[s] = dot * scale;
            mx = std::max(mx, lg[s]);
        }
        double z = 0.0;
        for (int s = 0; s < n_kv; ++s) { lg[s] = std::exp(lg[s] - mx); z += lg[s]; }
        for (int s = 0; s < n_kv; ++s) mass[h][s / P] += lg[s] / z;
    }
    return mass;
}

static double recall_of(const std::vector<std::vector<double>> & mass, const std::vector<char> & kept) {
    double r = 0.0;
    for (const auto & m : mass) for (size_t p = 0; p < m.size(); ++p) r += kept[p] ? m[p] : 0.0;
    return r / (double) mass.size();
}

static double shared_oracle(const std::vector<std::vector<double>> & mass, int n) {
    std::vector<double> mean(mass[0].size(), 0.0);
    for (const auto & m : mass) for (size_t p = 0; p < m.size(); ++p) mean[p] += m[p] / (double) mass.size();
    std::sort(mean.begin(), mean.end(), std::greater<double>());
    double r = 0.0;
    for (int i = 0; i < n && i < (int) mean.size(); ++i) r += mean[i];
    return r;
}

static int count_kept(const std::vector<char> & v) { int n = 0; for (char c : v) n += c ? 1 : 0; return n; }

// ---------------------------------------------------------------------------------------------------------------------
// Host tensors for the op. K for a layer is uploaded ONCE (all captured positions); the op reads only the first n_kv = pos+1 keys of it
// (its `usable` test requires b->ne[1] >= n_kv and it indexes kh * b->ne[1] + s).
struct KBuf {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * k = nullptr;
    void init(const LayerQK & L) {
        ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        ctx = ggml_init(ip);
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, L.hd, L.n_tok, L.n_kvh);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        ggml_backend_tensor_set(k, L.k.data(), 0, L.k.size() * sizeof(ggml_fp16_t));
    }
    ~KBuf() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
};

// Runs mask_from_kq_op for ONE query row at `pos` (n_kv = pos + 1, every key attendable) and reads the kept-page set back from the mask.
// head_perm (optional) reorders the query heads before the call: slot j receives original head head_perm[j].
static std::vector<char> run_op(const LayerQK & L, ggml_tensor * kt, int pos, int P, const ik_offload::OffloadConfig & cfg,
                                const std::vector<int> * head_perm = nullptr) {
    const int n_kv = pos + 1, nP = (n_kv + P - 1) / P;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, 1);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, 1);
    ggml_tensor * c   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, L.hd, 1, L.n_head);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    std::vector<float> zeros(n_kv, 0.0f);
    ggml_backend_tensor_set(a, zeros.data(), 0, zeros.size() * sizeof(float));
    ggml_backend_tensor_set(dst, zeros.data(), 0, zeros.size() * sizeof(float));
    std::vector<float> qq((size_t) L.n_head * L.hd);
    for (int j = 0; j < L.n_head; ++j) {
        const int h = head_perm ? (*head_perm)[j] : j;
        std::memcpy(&qq[(size_t) j * L.hd], &L.q[((size_t) pos * L.n_head + h) * L.hd], L.hd * sizeof(float));
    }
    ggml_backend_tensor_set(c, qq.data(), 0, qq.size() * sizeof(float));
    ik_offload::mask_from_kq_op(dst, a, kt, c, 0, 1, (void *) &cfg);
    std::vector<float> m(n_kv);
    ggml_backend_tensor_get(dst, m.data(), 0, m.size() * sizeof(float));
    std::vector<char> kept(nP, 0);
    for (int p = 0; p < nP; ++p) kept[p] = !(std::isinf(m[(size_t) p * P]) && m[(size_t) p * P] < 0);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return kept;
}

// head permutation that makes the grouped mapping h'/g equal the interleaved h % n_kvh: slot (h % n_kvh) * g + h / n_kvh gets head h.
static std::vector<int> interleave_perm(int n_head, int n_kvh) {
    const int g = n_head / n_kvh;
    std::vector<int> perm(n_head);
    for (int h = 0; h < n_head; ++h) perm[(h % n_kvh) * g + h / n_kvh] = h;
    return perm;
}

struct Acc { double n = 0, kept = 0, rec = 0, orc = 0, rnd = 0, lift = 0; int lift_n = 0; };
static void add(Acc & a, const std::vector<std::vector<double>> & mass, const std::vector<char> & kept) {
    const int nk = count_kept(kept), nP = (int) kept.size();
    const double r = recall_of(mass, kept), o = shared_oracle(mass, nk), rd = (double) nk / nP;
    a.n += 1; a.kept += nk; a.rec += r; a.orc += o; a.rnd += rd;
    if (o - rd > 1e-9) { a.lift += (r - rd) / (o - rd); a.lift_n += 1; }
}

// ---------------------------------------------------------------------------------------------------------------------
static int g_fail = 0;
static void check(bool ok, const char * what) { printf("  %-4s %s\n", ok ? "ok" : "FAIL", what); if (!ok) ++g_fail; }
struct lcg { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (float) ((s >> 33) & 0xFFFFFF) / (float) 0x800000 - 1.0f; } };

static int selftest() {
    printf("offload-attn-recall --selftest\n");
    LayerQK L; L.hd = 8; L.n_head = 4; L.n_kvh = 2; L.n_tok = 96;
    const int P = 4;
    L.q.resize((size_t) L.n_tok * L.n_head * L.hd);
    L.k.resize((size_t) L.n_kvh * L.n_tok * L.hd);
    lcg g{ 20261019ULL };
    for (auto & v : L.q) v = g.next();
    for (auto & v : L.k) v = ggml_fp32_to_fp16(3.0f * g.next());     // peaked attention
    KBuf kb; kb.init(L);
    const int pos = L.n_tok - 1, nP = (pos + 1 + P - 1) / P;
    const double scale = 1.0 / std::sqrt((double) L.hd);
    auto mass = page_mass(L, pos, P, scale);
    bool sums = true;
    for (auto & m : mass) { double s = 0; for (double v : m) s += v; sums &= std::fabs(s - 1.0) < 1e-9; }
    check(sums, "each head's page mass sums to 1");
    check(std::fabs(shared_oracle(mass, nP) - 1.0) < 1e-9, "POSITIVE CONTROL the oracle keeping every page scores exactly 1");
    std::vector<char> all(nP, 1);
    check(std::fabs(recall_of(mass, all) - 1.0) < 1e-9, "recall of the full set is exactly 1");
    ik_offload::OffloadConfig cfg; cfg.page_size = P; cfg.page_budget = 2.0f; cfg.min_kept_tokens = 0; cfg.sink_pages = 0;
    const auto uni = run_op(L, kb.k, pos, P, cfg);
    const int nk = count_kept(uni);
    // NEGATIVE CONTROL: a realised random selector of the same size, averaged over many draws, sits at n/nP.
    double rsum = 0; const int draws = 4000;
    lcg rg{ 7ULL };
    for (int d = 0; d < draws; ++d) {
        std::vector<int> idx(nP); std::iota(idx.begin(), idx.end(), 0);
        for (int i = nP - 1; i > 0; --i) { int j = (int) (((rg.next() + 1.0f) * 0.5f) * (i + 1)); if (j > i) j = i; std::swap(idx[i], idx[j]); }
        std::vector<char> rs(nP, 0); for (int i = 0; i < nk; ++i) rs[idx[i]] = 1;
        rsum += recall_of(mass, rs);
    }
    const double rmean = rsum / draws, rexp = (double) nk / nP, rec = recall_of(mass, uni), orc = shared_oracle(mass, nk);
    printf("    union kept %d of %d: recall %.4f | shared oracle %.4f | random expected %.4f realised %.4f\n", nk, nP, rec, orc, rexp, rmean);
    check(nk > 0 && nk < nP, "the op masked something and kept something");
    check(std::fabs(rmean - rexp) < 0.03, "NEGATIVE CONTROL a realised random set of the same size averages the expected n/nP (within 0.03)");
    check(rec > rexp, "the op's union beats random on peaked data");
    check(rec <= orc + 1e-9, "the op cannot beat the shared-set oracle");
    // The interleaved arm: permuted heads through the FIXED op must equal a plain-loop interleaved reference.
    std::vector<char> ref(nP, 0);
    for (int h = 0; h < L.n_head; ++h) {
        const int kh = h % L.n_kvh;
        std::vector<float> sc(nP, -INFINITY);
        for (int s = 0; s <= pos; ++s) {
            double dot = 0; for (int d = 0; d < L.hd; ++d) dot += (double) L.q[((size_t) pos * L.n_head + h) * L.hd + d] * L.kf(kh, s, d);
            sc[s / P] = std::max(sc[s / P], (float) (dot * scale));
        }
        std::vector<int> o(nP); std::iota(o.begin(), o.end(), 0);
        std::sort(o.begin(), o.end(), [&](int x, int y) { return sc[x] > sc[y]; });
        for (int i = 0; i < 2; ++i) ref[o[i]] = 1;
    }
    const auto perm = interleave_perm(L.n_head, L.n_kvh);
    const auto inter = run_op(L, kb.k, pos, P, cfg, &perm);
    check(inter == ref, "the head permutation reproduces an independent plain-loop INTERLEAVED (h % n_kv) reference");
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------------------------------
struct Capture {
    std::vector<int> layers;
    std::map<int, LayerQK> qk;     // by layer
    std::map<int, int> q_rows, k_rows;    // rows filled (counted once per position: a later tensor of the same name in the same batch OVERWRITES, so the roped one wins)
    std::map<int, std::vector<char>> q_seen, k_seen;
    int batch_start = 0;                  // position of the first token of the batch being decoded (set by main before each llama_decode)
    int n_ctx = 0;
    std::string err;
};

static bool want_name(const Capture & C, const char * name, int & il, bool & is_q) {
    int v = -1;
    if (std::sscanf(name, "Qcur-%d", &v) == 1 && std::strncmp(name, "Qcur-", 5) == 0) is_q = true;
    else if (std::sscanf(name, "Kcur-%d", &v) == 1 && std::strncmp(name, "Kcur-", 5) == 0) is_q = false;
    else return false;
    if (std::find(C.layers.begin(), C.layers.end(), v) == C.layers.end()) return false;
    il = v;
    return true;
}

// The builders name TWO tensors "Qcur-<il>" (and "Kcur-<il>"): the projection output [n_embd, n_tok] before the reshape, and the roped
// [head_dim, n_head, n_tok] after it. Only the roped one is what attention reads. It is the one whose ne[0] is a head size.
static bool is_roped_head_tensor(const ggml_tensor * t) {
    return ggml_n_dims(t) == 3 && t->ne[0] <= 512 && t->ne[1] <= 256;
}

static float read_f(const ggml_tensor * t, const std::vector<uint8_t> & raw, int64_t i0, int64_t i1, int64_t i2) {
    const size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    if (t->type == GGML_TYPE_F32) { float v; std::memcpy(&v, raw.data() + off, 4); return v; }
    ggml_fp16_t h; std::memcpy(&h, raw.data() + off, 2); return ggml_fp16_to_fp32(h);
}

static bool cb_eval_b(ggml_tensor * t, bool ask, void * ud) {
    Capture & C = *static_cast<Capture *>(ud);
    int il = -1; bool is_q = false;
    if (!want_name(C, t->name, il, is_q) || !is_roped_head_tensor(t)) return ask ? false : true;
    if (ask) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) { C.err = std::string("unsupported type for ") + t->name; return true; }
    // Qcur: [hd, n_head, n_tok]; Kcur: [hd, n_kvh, n_tok]
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    LayerQK & L = C.qk[il];
    const int hd = (int) t->ne[0], nh = (int) t->ne[1], nt = (int) t->ne[2];
    if (L.hd == 0) { L.hd = hd; L.n_tok = C.n_ctx; }
    if (hd != L.hd) { C.err = std::string("head size changed for ") + t->name; return true; }
    std::vector<char> & seen = is_q ? C.q_seen[il] : C.k_seen[il];
    if (seen.empty()) seen.assign(C.n_ctx, 0);
    int & filled = is_q ? C.q_rows[il] : C.k_rows[il];
    if (is_q) {
        if (L.n_head == 0) { L.n_head = nh; L.q.assign((size_t) C.n_ctx * nh * hd, 0.0f); }
        if (nh != L.n_head) { C.err = std::string("head count changed for ") + t->name; return true; }
    } else {
        if (L.n_kvh == 0) { L.n_kvh = nh; L.k.assign((size_t) nh * C.n_ctx * hd, ggml_fp32_to_fp16(0.0f)); }
        if (nh != L.n_kvh) { C.err = std::string("kv head count changed for ") + t->name; return true; }
    }
    for (int t2 = 0; t2 < nt; ++t2) {
        const int row = C.batch_start + t2;
        if (row >= C.n_ctx) break;
        if (!seen[row]) { seen[row] = 1; ++filled; }
        for (int h = 0; h < nh; ++h) for (int d = 0; d < hd; ++d) {
            if (is_q) L.q[((size_t) row * nh + h) * hd + d] = read_f(t, raw, d, h, t2);
            else      L.k[((size_t) h * C.n_ctx + row) * hd + d] = ggml_fp32_to_fp16(read_f(t, raw, d, h, t2));
        }
    }
    return true;
}

// This fork's ggml_backend_sched_eval_callback returns int (ggml-backend.h:344): on ask, 1 = "I need this node" (the scheduler then
// synchronizes, ggml-backend.cpp:1929-1931), 0 = skip; after compute, 0 stops the graph.
static int cb_eval(ggml_tensor * t, bool ask, void * ud) {
    return cb_eval_b(t, ask, ud) ? 1 : 0;
}

static std::vector<int> parse_ints(const std::string & s) {
    std::vector<int> v; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, ',')) if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

int main(int argc, char ** argv) {
    std::string model_path, text_path;
    int n_ctx = 8192, n_queries = 32, P = 16, sink = 0, recent = 0, refresh = 0, n_threads = 4;
    bool scan = false;
    std::vector<int> layers = { 2, 12, 22 }, budgets = { 4, 8, 16, 32 };
    double kq_scale = 0.0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--selftest") return selftest();
        else if (a == "-m") model_path = nxt();
        else if (a == "-f") text_path = nxt();
        else if (a == "-c") n_ctx = std::atoi(nxt().c_str());
        else if (a == "--layers") layers = parse_ints(nxt());
        else if (a == "--queries") n_queries = std::atoi(nxt().c_str());
        else if (a == "--page") P = std::atoi(nxt().c_str());
        else if (a == "--budgets") budgets = parse_ints(nxt());
        else if (a == "--sink") sink = std::atoi(nxt().c_str());
        else if (a == "--recent") recent = std::atoi(nxt().c_str());
        else if (a == "--refresh") refresh = std::atoi(nxt().c_str());
        else if (a == "--scan") scan = true;
        else if (a == "--kq-scale") kq_scale = std::atof(nxt().c_str());
        else if (a == "-t") n_threads = std::atoi(nxt().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty()) { fprintf(stderr, "usage: %s --selftest | -m model.gguf -f text.txt [...]\n", argv[0]); return 2; }
    // The op reads these env vars; the harness drives every knob through OffloadConfig instead, so clear them.
    putenv_("GGML_OFFLOAD_ATTN_HEAD_REDUCE", ""); putenv_("GGML_OFFLOAD_ATTN_RECENT_PAGES", ""); putenv_("GGML_OFFLOAD_ATTN_SCORE_ALL", "");
    putenv_("GGML_OFFLOAD_ATTN_SCAN", ""); putenv_("GGML_OFFLOAD_ATTN_REFRESH", "");

    std::ifstream fin(text_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    if (text.empty()) { fprintf(stderr, "FAIL: empty text %s\n", text_path.c_str()); return 3; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "FAIL: model did not load\n"); return 3; }

    Capture C; C.layers = layers; C.n_ctx = n_ctx;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = std::min(n_ctx, 2048); cp.n_ubatch = cp.n_batch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_attn = false;   // capture the DENSE model's q/K
    cp.n_threads = n_threads; cp.n_threads_batch = n_threads;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &C;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: context\n"); return 4; }

    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(model, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n_ctx) { fprintf(stderr, "FAIL: text has %d tokens < -c %d\n", nt, n_ctx); return 5; }
    toks.resize(n_ctx);
    for (int i = 0; i < n_ctx; i += (int) cp.n_batch) {
        const int n = std::min((int) cp.n_batch, n_ctx - i);
        C.batch_start = i;
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) { fprintf(stderr, "FAIL: decode at %d\n", i); return 6; }
        printf("decoded %d / %d\n", i + n, n_ctx); fflush(stdout);
    }
    if (!C.err.empty()) { fprintf(stderr, "FAIL: %s\n", C.err.c_str()); return 7; }
    for (int il : layers) {
        if (C.q_rows[il] != n_ctx || C.k_rows[il] != n_ctx) {
            fprintf(stderr, "FAIL: layer %d captured q rows %d, k rows %d, want %d (no Qcur-%d/Kcur-%d node in this model's graph?)\n",
                    il, C.q_rows[il], C.k_rows[il], n_ctx, il, il);
            return 8;
        }
        const LayerQK & L = C.qk[il];
        if (L.n_head % L.n_kvh != 0) { fprintf(stderr, "FAIL: n_head %d %% n_kvh %d\n", L.n_head, L.n_kvh); return 8; }
    }
    llama_free(ctx);
    llama_model_free(model);

    // Query positions: the second half of the context (long-range queries), evenly spaced, always including the last token.
    std::vector<int> qpos;
    for (int i = 0; i < n_queries; ++i) qpos.push_back(n_ctx / 2 + (int) ((int64_t) (n_ctx / 2 - 1) * (i + 1) / n_queries));
    const LayerQK & L0 = C.qk[layers[0]];
    const double scale = kq_scale > 0 ? kq_scale : 1.0 / std::sqrt((double) L0.hd);
    printf("model %s ctx %d page %d heads %d kv_heads %d hd %d scale %.5f sink %d recent %d refresh %d scan %d queries %d (positions %d..%d)\n",
           model_path.c_str(), n_ctx, P, L0.n_head, L0.n_kvh, L0.hd, scale, sink, recent, refresh, (int) scan, n_queries, qpos.front(), qpos.back());
    printf("TSV\tlayer\tbudget\tmethod\tmean_kept\tmean_pages\trecall\toracle_at_kept\trandom_at_kept\tlift\n");

    std::map<std::string, Acc> overall;
    for (int il : layers) {
        const LayerQK & L = C.qk[il];
        KBuf kb; kb.init(L);
        const auto perm = interleave_perm(L.n_head, L.n_kvh);
        for (int B : budgets) {
            std::vector<std::string> names = { "union", "max", "max@union", "interleave" };
            if (scan) names.push_back("scan=bisect");
            const std::string rname = "refresh=" + std::to_string(refresh);
            if (refresh > 1) names.push_back(rname);
            std::vector<std::map<std::string, Acc>> per_thread(n_threads);
            std::atomic<int> next{0};
            std::mutex count_mu;                      // the op's keys-scored counter is global: diff it only under this lock
            uint64_t keys_exact = 0, keys_bisect = 0;
            double pages_sum = 0;
            for (int p : qpos) pages_sum += (p + 1 + P - 1) / P;
            auto worker = [&](int tid) {
                ik_offload::OffloadConfig cfg; cfg.page_size = P; cfg.page_budget = (float) B; cfg.min_kept_tokens = 0;
                cfg.sink_pages = sink; cfg.recent_pages = recent;
                for (int qi = next++; qi < (int) qpos.size(); qi = next++) {
                    const int pos = qpos[qi];
                    const auto mass = page_mass(L, pos, P, scale);
                    auto & acc = per_thread[tid];
                    ik_offload::OffloadConfig cu = cfg; cu.head_reduce = 0;
                    const auto ku = run_op(L, kb.k, pos, P, cu);
                    add(acc["union"], mass, ku);
                    ik_offload::OffloadConfig cm = cfg; cm.head_reduce = 1;
                    add(acc["max"], mass, run_op(L, kb.k, pos, P, cm));
                    ik_offload::OffloadConfig ce = cm; ce.page_budget = (float) std::max(1, count_kept(ku) - (recent > 0 ? recent : 0));
                    add(acc["max@union"], mass, run_op(L, kb.k, pos, P, ce));
                    add(acc["interleave"], mass, run_op(L, kb.k, pos, P, cu, &perm));
                    if (scan) {
                        ik_offload::OffloadConfig cb = cm; cb.scan_mode = 1;
                        std::lock_guard<std::mutex> lk(count_mu);
                        const uint64_t k0 = ik_offload::maskop_keys_scored();
                        (void) run_op(L, kb.k, pos, P, cm);
                        const uint64_t k1 = ik_offload::maskop_keys_scored();
                        add(acc["scan=bisect"], mass, run_op(L, kb.k, pos, P, cb));
                        const uint64_t k2 = ik_offload::maskop_keys_scored();
                        keys_exact += k1 - k0; keys_bisect += k2 - k1;
                    }
                    if (refresh > 1) {
                        const int pos0 = std::max(0, pos - (refresh - 1));
                        std::vector<char> old = run_op(L, kb.k, pos0, P, cm);
                        const int nP = (pos + 1 + P - 1) / P;
                        old.resize(nP, 0);
                        for (int s = pos0 + 1; s <= pos; ++s) old[s / P] = 1;                 // keys written since the full scan
                        for (int p = 0; p < sink && p < nP; ++p) old[p] = 1;
                        if (recent > 0) {
                            const int pg = pos / P;
                            for (int p = std::max(0, pg - recent + 1); p <= pg; ++p) old[p] = 1;
                        }
                        add(acc[rname], mass, old);
                    }
                    (void) tid;
                }
            };
            std::vector<std::thread> th;
            for (int t = 0; t < n_threads; ++t) th.emplace_back(worker, t);
            for (auto & t : th) t.join();
            for (const auto & nm : names) {
                Acc a;
                for (auto & pt : per_thread) { auto it = pt.find(nm); if (it == pt.end()) continue; const Acc & b = it->second;
                    a.n += b.n; a.kept += b.kept; a.rec += b.rec; a.orc += b.orc; a.rnd += b.rnd; a.lift += b.lift; a.lift_n += b.lift_n; }
                if (a.n == 0) continue;
                printf("TSV\t%d\t%d\t%s\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f\n", il, B, nm.c_str(), a.kept / a.n, pages_sum / qpos.size(),
                       a.rec / a.n, a.orc / a.n, a.rnd / a.n, a.lift_n ? a.lift / a.lift_n : 0.0);
                Acc & o = overall[std::to_string(B) + "\t" + nm];
                o.n += a.n; o.kept += a.kept; o.rec += a.rec; o.orc += a.orc; o.rnd += a.rnd; o.lift += a.lift; o.lift_n += a.lift_n;
            }
            if (scan && keys_exact > 0) {
                printf("KEYS\t%d\t%d\tbisect %llu exact %llu keys_frac %.4f\n", il, B, (unsigned long long) keys_bisect,
                       (unsigned long long) keys_exact, (double) keys_bisect / (double) keys_exact);
            }
            fflush(stdout);
        }
    }
    printf("SUMMARY over layers %zu x queries %zu\n", layers.size(), qpos.size());
    for (const auto & kv : overall) {
        const Acc & a = kv.second;
        printf("SUMMARY\t%s\tkept %.2f\trecall %.4f\toracle %.4f\trandom %.4f\tlift %.4f\n", kv.first.c_str(), a.kept / a.n, a.rec / a.n, a.orc / a.n, a.rnd / a.n,
               a.lift_n ? a.lift / a.lift_n : 0.0);
    }
    llama_backend_free();
    return 0;
}
