// attn-stats.cpp -- measures, on a REAL model and REAL text, the two numbers that decide two attention rows before any kernel is written:
//   S69-sdcpp-PICK-SAGEATTN-2026-09-28-01        : total-variation distance between the fp32 attention distribution and the one from
//                                                  int8 Q.K^T (per-token scales), with and without SageAttention's K smoothing
//   S69-FFD-FASTER-FLASH-DECODING-SPARSE-DECODE-PICK-2026-09-28-01 : fraction of 128-token KV blocks FFD's rule would prune, and the
//                                                  attention mass those pruned blocks held (exact fp32 scores = the best case for the 2-bit shadow)
// Usage: attn-stats <model.gguf> <text file> [n_tokens (default 2048)] [delta in nats (default 3.466)] [n_queries (default 32)]
// CPU only. Hooks post-RoPE "Qcur-<il>" / "Kcur-<il>" through cb_eval during ONE prefill decode; nothing is written to the model.
// BENCH-PENDING: this tool measures accuracy-side statistics only; no throughput number is claimed anywhere in it.
// The statistics are over the last n_queries positions against all earlier keys of the same prompt.

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
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct captured {
    std::vector<float> data;  // [n_tokens][n_head][D] flattened as t*(H*D) + h*D + d
    int64_t D = 0, H = 0, T = 0;
};

static std::map<int, captured> g_q, g_k, g_v;

// Accepted names: "Qcur-<il>"/"Kcur-<il>" (post-RoPE in the llm_graph builders and the split-mode llm_build_context
// path) and "Qcur_roped-<il>"/"Kcur_roped-<il>" (the single-device llm_build_context path, build_std_attention --
// which is what qwen35 runs on CPU; FIXED 2026-10-18: without these the tool captured 0 layers on qwen35).
// V: "Vcur-<il>", which that path only names in its 2-D [n_embd_v_gqa, n_tokens] form.
static bool parse_name(const char * name, char & which, int & il) {
    int n = 0;
    if      (strncmp(name, "Qcur_roped-", 11) == 0) { which = 'Q'; n = 11; }
    else if (strncmp(name, "Kcur_roped-", 11) == 0) { which = 'K'; n = 11; }
    else if (strncmp(name, "Qcur-", 5) == 0) { which = 'Q'; n = 5; }
    else if (strncmp(name, "Kcur-", 5) == 0) { which = 'K'; n = 5; }
    else if (strncmp(name, "Vcur-", 5) == 0) { which = 'V'; n = 5; }
    else return false;
    if (!name[n]) return false;
    for (const char * p = name + n; *p; ++p) { if (*p < '0' || *p > '9') return false; }
    il = atoi(name + n);
    return true;
}

static int g_names_dump = -1;   // ATTN_STATS_NAMES=1: print every node name the callback sees (reach check)

static int eval_cb(struct ggml_tensor * t, bool ask, void *) {
    if (g_names_dump < 0) { const char * e = getenv("ATTN_STATS_NAMES"); g_names_dump = (e && atoi(e) > 0) ? 1 : 0; }
    if (g_names_dump) fprintf(stderr, "attn-stats cb %s %s op=%s ne=[%lld,%lld,%lld,%lld] type=%s\n", ask ? "ask" : "got", t->name, ggml_op_desc(t),
                              (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
    char which; int il;
    if (!parse_name(t->name, which, il)) return ask ? 0 : 1;
    if (ask) return 1;
    captured c;
    c.D = t->ne[0]; c.H = t->ne[1]; c.T = t->ne[2];
    if (t->ne[3] != 1) return 1;
    // a 2-D [n_embd, n_tokens] tensor under the same name (the raw K projection is also called "Kcur") would read as
    // T = 1: Q/K keep only the [D, H, T] form. V may be 2-D (see parse_name); it is stored as D = ne0, H = n_tokens,
    // T = 1 and split into heads at analysis time, and a 3-D V capture is never replaced by a 2-D one.
    if (t->ne[2] == 1 && which != 'V') return 1;
    if (which == 'V' && t->ne[2] == 1 && g_v.count(il) && g_v[il].T > 1) return 1;
    const int64_t n = ggml_nelements(t);
    c.data.resize(n);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    // walk with the real strides so a permuted view still reads correctly
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2];
        float v;
        if (t->type == GGML_TYPE_F32) v = *(const float *) &raw[off];
        else if (t->type == GGML_TYPE_F16) v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &raw[off]);
        else return 1;
        c.data[(size_t) i2*(c.H*c.D) + (size_t) i1*c.D + i0] = v;
    }
    (which == 'Q' ? g_q : which == 'K' ? g_k : g_v)[il] = std::move(c);   // the last capture per layer wins (post-RoPE)
    return 1;
}

static void quant_row(const float * x, int D, std::vector<int8_t> & q, float & scale) {
    float am = 0.f;
    for (int d = 0; d < D; ++d) am = std::max(am, std::fabs(x[d]));
    scale = am > 0.f ? am/127.f : 1.f;
    q.resize(D);
    for (int d = 0; d < D; ++d) q[d] = (int8_t) std::max(-127.f, std::min(127.f, std::nearbyint(x[d]/scale)));
}

static void softmax(std::vector<double> & s) {
    double m = -1e300; for (double v : s) m = std::max(m, v);
    double z = 0; for (double & v : s) { v = std::exp(v - m); z += v; }
    for (double & v : s) v /= z;
}

// ---- attention OUTPUT metric (S69-HYPERQWEN-CHALLENGE-INT8-QK-PREFILL-2026-10-18-01) --------------------------
// The TV distance above measures the attention ROW; what the next layer sees is the row times V. These helpers
// compare o_ref = softmax(q.K^T/sqrt(D)) V (fp64 scores) with o_q = softmax(int8(q).int8(K)^T*scales/sqrt(D)) V.
// K scale modes: per key row (one scale per token, as the SAGE lines above) or one scale per KTILE-key tile of
// the same KV head (the granularity a tiled int8 kernel would use). Q is always per-token.
static const int KTILE = 64;

// keys at K + j*ld, j < nk; mode 0 = per-row scale, mode 1 = per-KTILE-key-tile scale. ks has one entry per key.
static void quant_keys(const float * K, int nk, int D, size_t ld, int mode, std::vector<int8_t> & kq, std::vector<float> & ks) {
    kq.assign((size_t) nk*D, 0); ks.assign(nk, 1.f);
    const int step = mode == 0 ? 1 : KTILE;
    for (int j0 = 0; j0 < nk; j0 += step) {
        const int j1 = std::min(nk, j0 + step);
        float am = 0.f;
        for (int j = j0; j < j1; ++j) for (int d = 0; d < D; ++d) am = std::max(am, std::fabs(K[(size_t) j*ld + d]));
        const float sc = am > 0.f ? am/127.f : 1.f;
        for (int j = j0; j < j1; ++j) {
            ks[j] = sc;
            for (int d = 0; d < D; ++d) kq[(size_t) j*D + d] = (int8_t) std::max(-127.f, std::min(127.f, std::nearbyint(K[(size_t) j*ld + d]/sc)));
        }
    }
}

static void int8_scores(const float * q, int D, const std::vector<int8_t> & kq, const std::vector<float> & ks, int nk, std::vector<double> & s) {
    std::vector<int8_t> qq; float qs; quant_row(q, D, qq, qs);
    s.resize(nk);
    for (int j = 0; j < nk; ++j) {
        int64_t isum = 0; const int8_t * k = &kq[(size_t) j*D];
        for (int d = 0; d < D; ++d) isum += (int) qq[d]*(int) k[d];
        s[j] = (double) isum*qs*ks[j] / std::sqrt((double) D);
    }
}

static void exact_scores(const float * q, const float * K, int nk, int D, size_t ld, std::vector<double> & s) {
    s.resize(nk);
    for (int j = 0; j < nk; ++j) { double dot = 0; for (int d = 0; d < D; ++d) dot += (double) q[d]*K[(size_t) j*ld + d]; s[j] = dot / std::sqrt((double) D); }
}

// o = sum_j p[j] V[j]   (V rows at V + j*ld)
static void attn_out(const std::vector<double> & p, const float * V, int D, size_t ld, std::vector<double> & o) {
    o.assign(D, 0.0);
    for (size_t j = 0; j < p.size(); ++j) { const double w = p[j]; const float * v = V + j*ld; for (int d = 0; d < D; ++d) o[d] += w*v[d]; }
}

// cosine and relative L2 ||b - a|| / ||a||; sqrt(fl(x*x)) == |x| in IEEE round-to-nearest, so identical inputs give
// exactly cos 1 and rel 0 through the formula itself (no identical-input special case)
static void out_metric(const std::vector<double> & a, const std::vector<double> & b, double & cos, double & rel) {
    double ab = 0, aa = 0, bb = 0, dd = 0;
    for (size_t i = 0; i < a.size(); ++i) { ab += a[i]*b[i]; aa += a[i]*a[i]; bb += b[i]*b[i]; const double e = b[i] - a[i]; dd += e*e; }
    cos = (aa > 0 && bb > 0) ? ab / std::sqrt(aa*bb) : (aa == bb ? 1.0 : 0.0);
    rel = aa > 0 ? std::sqrt(dd) / std::sqrt(aa) : std::sqrt(dd);
}

// sparsity at error: with attention row p split into BLK-key blocks, the largest fraction of blocks that can be
// dropped while the dropped probability mass stays <= eps (greedy: smallest-mass blocks first). Reads off how
// much block sparsity a given error budget buys, independent of any particular pruning rule.
static double sparsity_at_error(const std::vector<double> & p, int blk, double eps) {
    const int n = (int) p.size(), nb = (n + blk - 1)/blk;
    if (nb <= 0) return 0.0;
    std::vector<double> m(nb, 0.0);
    for (int j = 0; j < n; ++j) m[j/blk] += p[j];
    std::sort(m.begin(), m.end());
    double lost = 0; int dropped = 0;
    for (int b = 0; b < nb; ++b) { if (lost + m[b] > eps) break; lost += m[b]; ++dropped; }
    return (double) dropped/nb;
}

// ---- pooled-score block selector (S69-ENGREV-SPARGEATTN-SPARSITY-AT-ERROR-CALIB-2026-10-27-01) -----------------
// Measures what SpargeAttn's block selector would cost in attention-output error on real Q/K/V, before any kernel
// uses it. BENCH-PENDING: no speed claim. Own implementation; no code copied (Apache-2.0 source).
// SOURCE-READ: upstream/SpargeAttn/spas_sage_attn/utils.py:217-240 (pool + self-similarity),
//   utils.py:319-369 (get_block_map_meansim), utils.py:283-292 (at least one block), autotune.py:79 and :279.
// SOURCE-NUANCE: things the source does that a summary of it did not say --
//   * the self-similarity test is applied to KEY blocks as well as query blocks: a key block that is not
//     self-similar gets -inf in the pooled score (it takes no part in the softmax) and is then kept for EVERY
//     query block (utils.py:337, :356); a query block that is not self-similar keeps every key block (:357)
//   * "self-similar" is sum(unit rows gram) / n^2 > simthreshd1, strict, with n = rows actually in the block, so a
//     short last block is pooled and judged over its own rows (:225-235)
//   * the cumulative cut keeps the sorted blocks whose running sum is <= cdfthreshd (searchsorted right=True, :351),
//     i.e. NOT the block that crosses the threshold; zero selected becomes one (:289)
//   * causality is per block: key block kb is valid for query block qb iff kb < (qb + 1)*BLKQ/BLKK (:308)
//   * attention_sink (keep key block 0) is an option there (:362-363); a head counts as sparse only when its mean
//     sparsity is > 0.1 (autotune.py:279); the L1 bound defaults to 0.07 at this commit (autotune.py:79)
// DEVIATIONS, deliberate: key block 0 is always kept and so are the key blocks that overlap the query block's own
// tokens (the fork's own selector, llama-kv-xattn.h, already keeps first column + diagonal); cdf >= 1 keeps every
// valid block outright (a running float sum can pass 1.0 early), which is the dense reference arm.
static const int PBLKQ = 128, PBLKK = 64;

struct pooled_sel {
    int nqb = 0, nkb = 0;
    std::vector<uint8_t> keep;     // [nqb][nkb] 1 = key block kb is computed for query block qb
    std::vector<uint8_t> q_dense;  // [nqb] 1 = query block not self-similar: every valid key block kept
    std::vector<uint8_t> k_dense;  // [nkb] 1 = key block not self-similar: kept for every query block
};

// mean of rows [r0, r1) (row r at X + r*ld) into m; returns sum of the unit-row gram / n^2 = || mean unit row ||^2
static double pool_rows(const float * X, int r0, int r1, int D, size_t ld, std::vector<double> & m) {
    m.assign(D, 0.0);
    std::vector<double> u(D, 0.0);
    for (int r = r0; r < r1; ++r) {
        const float * x = X + (size_t) r*ld; double nn = 0;
        for (int d = 0; d < D; ++d) { m[d] += x[d]; nn += (double) x[d]*x[d]; }
        const double inv = nn > 0 ? 1.0/std::sqrt(nn) : 0.0;
        for (int d = 0; d < D; ++d) u[d] += x[d]*inv;
    }
    const double n = (double) (r1 - r0); double s = 0;
    for (int d = 0; d < D; ++d) { m[d] /= n; const double v = u[d]/n; s += v*v; }
    return s;
}

// number of leading entries of a descending probability list whose running sum is <= thr; never fewer than one
static int pooled_num_to_select(const std::vector<double> & sorted_desc, double thr) {
    double cum = 0; int n = 0;
    for (double v : sorted_desc) { cum += v; if (cum <= thr) ++n; else break; }
    return std::max(1, n);
}

// key blocks valid for query block qb (causal, per block)
static int pooled_nvalid(int qb, int nkb) { return std::min(nkb, (qb + 1)*PBLKQ/PBLKK); }

static void pooled_select(const float * Q, size_t ldq, const float * K, size_t ldk, int T, int D, double cdf, double simthr, pooled_sel & out) {
    out.nqb = (T + PBLKQ - 1)/PBLKQ; out.nkb = (T + PBLKK - 1)/PBLKK;
    out.keep.assign((size_t) out.nqb*out.nkb, 0); out.q_dense.assign(out.nqb, 0); out.k_dense.assign(out.nkb, 0);
    std::vector<std::vector<double>> pk(out.nkb);
    for (int kb = 0; kb < out.nkb; ++kb) out.k_dense[kb] = !(pool_rows(K, kb*PBLKK, std::min(T, (kb + 1)*PBLKK), D, ldk, pk[kb]) > simthr);
    std::vector<double> pq;
    for (int qb = 0; qb < out.nqb; ++qb) {
        const int q0 = qb*PBLKQ, q1 = std::min(T, q0 + PBLKQ), nvalid = pooled_nvalid(qb, out.nkb);
        uint8_t * kp = &out.keep[(size_t) qb*out.nkb];
        const bool q_sim = pool_rows(Q, q0, q1, D, ldq, pq) > simthr;
        if (cdf >= 1.0 || !q_sim) {
            out.q_dense[qb] = cdf < 1.0;
            for (int kb = 0; kb < nvalid; ++kb) kp[kb] = 1;
            continue;
        }
        std::vector<int> ord;   // the self-similar valid key blocks: only these are ranked
        std::vector<double> s;
        for (int kb = 0; kb < nvalid; ++kb) {
            if (out.k_dense[kb]) { kp[kb] = 1; continue; }
            double dot = 0; for (int d = 0; d < D; ++d) dot += pq[d]*pk[kb][d];
            ord.push_back(kb); s.push_back(dot / std::sqrt((double) D));
        }
        if (!ord.empty()) {
            softmax(s);
            std::vector<int> ix(ord.size());
            for (size_t i = 0; i < ix.size(); ++i) ix[i] = (int) i;
            std::stable_sort(ix.begin(), ix.end(), [&](int a, int b) { return s[a] > s[b]; });   // ties: lower block first
            std::vector<double> sd(ix.size());
            for (size_t i = 0; i < ix.size(); ++i) sd[i] = s[ix[i]];
            const int nsel = pooled_num_to_select(sd, cdf);
            for (int i = 0; i < nsel; ++i) kp[ord[ix[i]]] = 1;
        }
        kp[0] = 1;                                                // attention sink (deviation: always on)
        for (int kb = q0/PBLKK; kb < nvalid; ++kb) kp[kb] = 1;    // diagonal (deviation: always kept)
    }
}

struct pooled_eval {
    double kept = 0;   // kept / causally valid key blocks, mean over query blocks
    double l1 = 0;     // sum|o_sparse - o_dense| / sum|o_dense|, mean over the evaluated query rows
    double cos = 0;    // cosine(o_sparse, o_dense), mean over the evaluated query rows
    int q_dense = 0, k_dense = 0, nqb = 0, nkb = 0; long nq = 0;
};

// n_eval query rows spread evenly over [PBLKQ, T): dense causal softmax(q K^T) V against the same sum restricted to the
// keys of the kept blocks. The restricted softmax uses the same operations in the same order as softmax(), so with
// every block kept the two outputs are bit-identical (cos exactly 1, l1 exactly 0).
static pooled_eval pooled_evaluate(const float * Q, size_t ldq, const float * K, size_t ldk, const float * V, size_t ldv,
                                   int T, int D, int Dv, double cdf, double simthr, int n_eval, pooled_sel * sel_out = nullptr) {
    pooled_sel sel; pooled_select(Q, ldq, K, ldk, T, D, cdf, simthr, sel);
    pooled_eval r; r.nqb = sel.nqb; r.nkb = sel.nkb;
    for (int kb = 0; kb < sel.nkb; ++kb) r.k_dense += sel.k_dense[kb];
    for (int qb = 0; qb < sel.nqb; ++qb) {
        const int nvalid = pooled_nvalid(qb, sel.nkb); int k = 0;
        for (int kb = 0; kb < nvalid; ++kb) k += sel.keep[(size_t) qb*sel.nkb + kb];
        r.kept += (double) k/nvalid; r.q_dense += sel.q_dense[qb];
    }
    r.kept /= std::max(1, sel.nqb);
    const int first = std::min(T - 1, PBLKQ), n = std::min(n_eval, T - first);
    for (int e = 0; e < n; ++e) {
        const int qi = first + (int) ((int64_t) e*(T - 1 - first)/std::max(1, n - 1));
        const float * q = Q + (size_t) qi*ldq; const int nk = qi + 1;
        std::vector<double> s, p, ps(nk, 0.0), o_d, o_s;
        exact_scores(q, K, nk, D, ldk, s); p = s; softmax(p); attn_out(p, V, Dv, ldv, o_d);
        const uint8_t * kp = &sel.keep[(size_t) (qi/PBLKQ)*sel.nkb];
        double m = -1e300, z = 0;
        for (int j = 0; j < nk; ++j) if (kp[j/PBLKK]) m = std::max(m, s[j]);
        for (int j = 0; j < nk; ++j) if (kp[j/PBLKK]) { ps[j] = std::exp(s[j] - m); z += ps[j]; }
        for (int j = 0; j < nk; ++j) ps[j] /= z;
        attn_out(ps, V, Dv, ldv, o_s);
        double c, rel; out_metric(o_d, o_s, c, rel);
        double num = 0, den = 0;
        for (int d = 0; d < Dv; ++d) { num += std::fabs(o_s[d] - o_d[d]); den += std::fabs(o_d[d]); }
        r.l1 += den > 0 ? num/den : num; r.cos += c; ++r.nq;
    }
    if (r.nq) { r.l1 /= r.nq; r.cos /= r.nq; }
    if (sel_out) *sel_out = sel;
    return r;
}

// --selftest: the metric is exactly cos 1 / rel 0 for identical inputs and degrades for planted perturbations
static int selftest() {
    const int D = 256, T = 512;
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double) (st >> 11) / 9007199254740992.0 * 2.0 - 1.0; };
    std::vector<float> q(D), K((size_t) T*D), V((size_t) T*D);
    for (auto & x : q) x = (float) (3.0*rnd());
    for (int j = 0; j < T; ++j) { const double mag = 0.5 + 4.0*std::fabs(rnd()); for (int d = 0; d < D; ++d) { K[(size_t) j*D + d] = (float) (mag*rnd()); V[(size_t) j*D + d] = (float) rnd(); } }
    int fails = 0;
    auto check = [&](bool ok, const char * what, double c, double r) { printf("selftest %-58s cos %.17g rel %.6g : %s\n", what, c, r, ok ? "PASS" : "FAIL"); fails += !ok; };
    std::vector<double> s, p, o_ref, o2, o3; double c, r;
    exact_scores(q.data(), K.data(), T, D, D, s); p = s; softmax(p); attn_out(p, V.data(), D, D, o_ref);
    // 1. identical inputs: the same exact path computed twice
    exact_scores(q.data(), K.data(), T, D, D, s); p = s; softmax(p); attn_out(p, V.data(), D, D, o2);
    out_metric(o_ref, o2, c, r); check(c == 1.0 && r == 0.0, "identical inputs -> cos exactly 1, rel exactly 0", c, r);
    // 2. planted perturbation of one V row: must degrade, and a larger plant must degrade more
    double r_small, r_big;
    { std::vector<float> Vp = V; const int j = (int) (std::max_element(p.begin(), p.end()) - p.begin()); for (int d = 0; d < D; ++d) Vp[(size_t) j*D + d] += 0.05f;
      attn_out(p, Vp.data(), D, D, o2); out_metric(o_ref, o2, c, r); r_small = r; check(c < 1.0 && r > 0.0, "planted +0.05 on the top-weight V row -> cos < 1, rel > 0", c, r);
      for (int d = 0; d < D; ++d) Vp[(size_t) j*D + d] += 0.45f;
      attn_out(p, Vp.data(), D, D, o3); out_metric(o_ref, o3, c, r); r_big = r; check(c < 1.0 && r_big > r_small, "planted +0.50 -> rel larger than the +0.05 plant", c, r); }
    // 3. planted perturbation of the scores (one key's logit + 2 nats): must degrade
    { std::vector<double> s2; exact_scores(q.data(), K.data(), T, D, D, s2); s2[T/3] += 2.0; softmax(s2); attn_out(s2, V.data(), D, D, o2);
      out_metric(o_ref, o2, c, r); check(c < 1.0 && r > 0.0, "planted +2 nats on one logit -> cos < 1, rel > 0", c, r); }
    // 4. the int8 paths degrade (not exact) but stay close on benign data. Which scale mode is better is a property
    //    of the data, not of the metric (MEASURED here: per-tile 0.0123 < per-row 0.0150 on these mixed-magnitude
    //    keys, so an earlier "per-tile >= per-row" assertion was wrong and is now only printed)
    std::vector<int8_t> kq; std::vector<float> ks; double r_row, r_tile;
    quant_keys(K.data(), T, D, D, 0, kq, ks); int8_scores(q.data(), D, kq, ks, T, s); softmax(s); attn_out(s, V.data(), D, D, o2);
    out_metric(o_ref, o2, c, r); r_row = r; check(c < 1.0 && r > 0.0 && c > 0.9, "int8 per-row K scales -> 0.9 < cos < 1", c, r);
    quant_keys(K.data(), T, D, D, 1, kq, ks); int8_scores(q.data(), D, kq, ks, T, s); softmax(s); attn_out(s, V.data(), D, D, o2);
    out_metric(o_ref, o2, c, r); r_tile = r; check(c < 1.0 && r > 0.0 && c > 0.9, "int8 per-64-key-tile K scales -> 0.9 < cos < 1", c, r);
    // 5. sparsity at error: uniform row, one-hot row, monotone in eps
    { std::vector<double> u(512, 1.0/512), oh(512, 0.0); oh[300] = 1.0;
      const double su = sparsity_at_error(u, 64, 0.5), so = sparsity_at_error(oh, 64, 1e-9), s0 = sparsity_at_error(u, 64, 0.0);
      check(su == 0.5, "sparsity_at_error uniform 8 blocks eps 0.5 -> exactly 0.5", su, 0);
      check(so == 7.0/8.0, "sparsity_at_error one-hot eps 1e-9 -> 7/8", so, 0);
      check(s0 == 0.0, "sparsity_at_error uniform eps 0 -> 0", s0, 0);
      check(sparsity_at_error(p, 64, 0.01) <= sparsity_at_error(p, 64, 0.1) && sparsity_at_error(p, 64, 0.1) <= sparsity_at_error(p, 64, 0.5), "sparsity_at_error monotone in eps on a real row", sparsity_at_error(p, 64, 0.1), 0); }
    // 6. pooled-score block selector on planted data. Queries share one direction a (self-similar query blocks).
    //    Every key block has its own direction (self-similar) except: block 1 = 48*a, 12 nats above the rest for
    //    every query; block 4 = independent random rows (not self-similar).
    {
        const int PT = 1024, PD = 64; const double simthr = 0.1;
        std::vector<float> a(PD), PQ((size_t) PT*PD), QR((size_t) PT*PD), PK((size_t) PT*PD), PV((size_t) PT*PD);
        auto unit = [&](std::vector<float> & x) { double nn = 0; for (auto & e : x) { e = (float) rnd(); nn += (double) e*e; } for (auto & e : x) e = (float) (e/std::sqrt(nn)); };
        unit(a);
        for (int t = 0; t < PT; ++t) for (int d = 0; d < PD; ++d) {
            PQ[(size_t) t*PD + d] = (float) (2.0*a[d] + 0.05*rnd()); QR[(size_t) t*PD + d] = (float) rnd(); PV[(size_t) t*PD + d] = (float) rnd();
        }
        for (int kb = 0; kb < PT/PBLKK; ++kb) {
            std::vector<float> bdir(PD); unit(bdir);
            for (int t = kb*PBLKK; t < (kb + 1)*PBLKK; ++t) for (int d = 0; d < PD; ++d) {
                float v = (float) (3.0*bdir[d] + 0.02*rnd());
                if (kb == 1) v = (float) (48.0*a[d] + 0.02*rnd());
                if (kb == 4) v = (float) rnd();
                PK[(size_t) t*PD + d] = v;
            }
        }
        auto ev = [&](const std::vector<float> & Qx, double cdf, double thr, pooled_sel * so = nullptr) {
            return pooled_evaluate(Qx.data(), PD, PK.data(), PD, PV.data(), PD, PT, PD, PD, cdf, thr, 32, so); };
        pooled_sel sel;
        const pooled_eval e1 = ev(PQ, 1.0, simthr), e9 = ev(PQ, 0.9, simthr, &sel), e9b = ev(PQ, 0.9, simthr), e99 = ev(PQ, 0.99, simthr);
        check(e1.kept == 1.0 && e1.cos == 1.0 && e1.l1 == 0.0, "pooled cdf 1.0 (dense arm) -> kept 1, cos exactly 1, l1 exactly 0", e1.cos, e1.l1);
        check(e9.kept == e9b.kept && e9.l1 == e9b.l1 && e9.cos == e9b.cos, "pooled rerun -> identical kept / l1 / cos", e9.cos, e9.l1);
        check(e9.kept < 1.0 && e9.cos > 0.999 && e9.l1 < 0.07, "pooled cdf 0.9, planted block -> blocks dropped, cos > 0.999", e9.cos, e9.l1);
        bool inv = true, planted = true, kdense = true;
        for (int qb = 0; qb < sel.nqb; ++qb) {
            const int nv = pooled_nvalid(qb, sel.nkb); const uint8_t * kp = &sel.keep[(size_t) qb*sel.nkb];
            inv = inv && kp[0] == 1;
            for (int kb = qb*PBLKQ/PBLKK; kb < nv; ++kb) inv = inv && kp[kb] == 1;
            for (int kb = nv; kb < sel.nkb; ++kb) inv = inv && kp[kb] == 0;
            planted = planted && kp[1] == 1;
            if (nv > 4) kdense = kdense && kp[4] == 1;
        }
        check(inv, "pooled: sink + diagonal kept, nothing past the causal block limit", e9.kept, 0);
        check(planted, "pooled: the planted high-score key block is selected everywhere", e9.kept, 0);
        check(kdense && sel.k_dense[4] == 1 && e9.k_dense == 1, "pooled: the one non-self-similar key block is kept everywhere", e9.kept, 0);
        check(sel.keep[(size_t) 7*sel.nkb + 3] == 0, "pooled: a self-similar low-score key block is dropped (q 7, k 3)", e9.kept, 0);
        check(e9.kept <= e99.kept && e99.kept <= e1.kept, "pooled: kept fraction monotone in cdf", e99.kept, 0);
        const pooled_eval er = ev(QR, 0.9, simthr), er0 = ev(QR, 0.9, -1.0);
        check(er.q_dense == er.nqb && er.kept == 1.0 && er.cos == 1.0 && er.l1 == 0.0, "pooled: non-self-similar query blocks keep every key block", er.cos, er.l1);
        check(er0.q_dense == 0 && er0.k_dense == 0, "pooled: similarity threshold -1 turns the guard off", er0.kept, 0);
        const std::vector<double> sd = { 0.6, 0.3, 0.1 };
        check(pooled_num_to_select(sd, 0.9) == 2 && pooled_num_to_select(sd, 0.5) == 1 && pooled_num_to_select(sd, 0.95) == 2 && pooled_num_to_select(sd, 1.0) == 3,
              "pooled cut rule: running sum <= threshold, never fewer than one", 0, 0);
        printf("selftest info: pooled planted data kept fraction cdf 0.9 %.4f, cdf 0.99 %.4f (not a check)\n", e9.kept, e99.kept);
    }
    printf("selftest info: rel-L2 per-row %.6g per-64-key-tile %.6g (not a check)\n", r_row, r_tile);
    printf("selftest: %s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <text file> [n_tokens] [delta_nats] [n_queries]  |  %s --selftest\n", argv[0], argv[0]); return 2; }
    const int    n_tok = argc > 3 ? atoi(argv[3]) : 2048;
    std::vector<double> deltas;   // argv[4] may be a comma list: one FFD line per delta, one prefill
    { std::stringstream ds(argc > 4 ? argv[4] : "3.466"); std::string tk; while (std::getline(ds, tk, ',')) deltas.push_back(atof(tk.c_str())); }
    if (deltas.empty()) deltas.push_back(3.466);
    const double delta = deltas[0];
    const int    n_q   = argc > 5 ? atoi(argv[5]) : 32;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream f(argv[2], std::ios::binary); std::stringstream ss; ss << f.rdbuf(); const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(std::min(n, n_tok));
    printf("attn-stats: %zu tokens of real text, delta %.3f nats, last %d queries\n", toks.size(), delta, n_q);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + 8; cp.n_batch = cp.n_ubatch = (uint32_t) toks.size();
    cp.cb_eval = eval_cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    b.n_tokens = (int) toks.size();
    for (int i = 0; i < b.n_tokens; ++i) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == b.n_tokens - 1; }
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    printf("captured Q for %zu layers, K for %zu layers\n", g_q.size(), g_k.size());

    double tv_plain_sum = 0, tv_smooth_sum = 0; long cnt = 0;
    std::vector<double> prune_sum(deltas.size(), 0.0), lost_sum(deltas.size(), 0.0), lost_max(deltas.size(), 0.0);
    const double sp_eps[4] = { 0.001, 0.01, 0.05, 0.1 }; double sp_sum[4] = { 0, 0, 0, 0 }; long sp_cnt = 0;   // sparsity-at-error sweep
    std::vector<long> cnt_ffd(deltas.size(), 0);
    std::vector<double> tv_plain_all, tv_smooth_all;
    // output metric: 4 int8 arms = {per-row, per-64-key-tile} x {plain, K smoothing}
    const char * out_arm_name[4] = { "per-row K scale", "per-row K scale + K smoothing", "per-64-key-tile K scale", "per-64-key-tile K scale + K smoothing" };
    std::vector<double> out_cos[4], out_rel[4];
    std::map<int, std::vector<double>> out_rel_layer[4];
    const int BLK = 128;
    // pooled-score selector sweep, opt-in: ATTN_STATS_POOLED=1. ATTN_STATS_SIMTHR (default 0.1) and ATTN_STATS_POOLED_L1
    // (default 0.07) are the source's defaults (utils.py:319, autotune.py:79); ATTN_STATS_POOLED_JSON=<path> writes the
    // per-head is_sparse table (sparse = some cdf < 1 meets the L1 bound with more than 10% of the key blocks dropped).
    const bool   pooled_on     = getenv("ATTN_STATS_POOLED") && atoi(getenv("ATTN_STATS_POOLED")) != 0;
    const double pooled_simthr = getenv("ATTN_STATS_SIMTHR") ? atof(getenv("ATTN_STATS_SIMTHR")) : 0.1;
    const double pooled_l1     = getenv("ATTN_STATS_POOLED_L1") ? atof(getenv("ATTN_STATS_POOLED_L1")) : 0.07;
    const double pooled_cdf[5] = { 0.9, 0.95, 0.97, 0.99, 1.0 };
    struct pooled_head { int il, head; double sparsity, cdf; };
    std::vector<pooled_head> pooled_heads;
    for (auto & kv : g_q) {
        const int il = kv.first; if (!g_k.count(il)) continue;
        const captured & Q = kv.second; const captured & K = g_k[il];
        const int D = (int) Q.D, H = (int) Q.H, Hkv = (int) K.H, T = (int) Q.T; const int grp = H / Hkv;
        if (T < 2*BLK || H % Hkv) continue;
        // per (kv head, channel) mean of K over all tokens (the smoothing term)
        std::vector<float> kmean((size_t) Hkv*D, 0.f);
        for (int t = 0; t < T; ++t) for (int h = 0; h < Hkv; ++h) for (int d = 0; d < D; ++d) kmean[(size_t) h*D + d] += K.data[(size_t) t*(Hkv*D) + (size_t) h*D + d] / T;
        std::vector<int8_t> qq, kq; float qs, ks;
        std::vector<float> ksm(D);
        for (int h = 0; h < H; ++h) {
            const int hk = h / grp;
            for (int qi = T - n_q; qi < T; ++qi) {
                const float * q = &Q.data[(size_t) qi*(H*D) + (size_t) h*D];
                const int nk = qi + 1;
                std::vector<double> s_ref(nk), s_plain(nk), s_sm(nk);
                quant_row(q, D, qq, qs);
                for (int j = 0; j < nk; ++j) {
                    const float * k = &K.data[(size_t) j*(Hkv*D) + (size_t) hk*D];
                    double dot = 0; for (int d = 0; d < D; ++d) dot += (double) q[d]*k[d];
                    s_ref[j] = dot / std::sqrt((double) D);
                    quant_row(k, D, kq, ks);
                    int64_t isum = 0; for (int d = 0; d < D; ++d) isum += (int) qq[d]*(int) kq[d];
                    s_plain[j] = (double) isum*qs*ks / std::sqrt((double) D);
                    for (int d = 0; d < D; ++d) ksm[d] = k[d] - kmean[(size_t) hk*D + d];
                    quant_row(ksm.data(), D, kq, ks);
                    isum = 0; for (int d = 0; d < D; ++d) isum += (int) qq[d]*(int) kq[d];
                    s_sm[j] = (double) isum*qs*ks / std::sqrt((double) D);   // differs from the true score by q.mean, a per-query constant: softmax-invariant
                }
                std::vector<double> p_ref = s_ref, p_plain = s_plain, p_sm = s_sm;
                softmax(p_ref); softmax(p_plain); softmax(p_sm);
                double tvp = 0, tvs = 0;
                for (int j = 0; j < nk; ++j) { tvp += std::fabs(p_ref[j] - p_plain[j]); tvs += std::fabs(p_ref[j] - p_sm[j]); }
                tv_plain_sum += 0.5*tvp; tv_smooth_sum += 0.5*tvs; tv_plain_all.push_back(0.5*tvp); tv_smooth_all.push_back(0.5*tvs); ++cnt;
            }
        }
        // attention OUTPUT softmax(QK^T)V: exact fp64 path vs the int8 Q.K arms (V itself kept exact)
        if (g_v.count(il) && g_v[il].T == 1 && g_v[il].H == T && g_v[il].D % Hkv == 0) {   // 2-D V: [T][Hkv*Dv] -> [T][Hkv][Dv]
            captured & v2 = g_v[il]; v2.D = v2.D / Hkv; v2.H = Hkv; v2.T = T;
        }
        if (g_v.count(il) && g_v[il].H == Hkv && g_v[il].T == T) {
            const captured & V = g_v[il]; const int Dv = (int) V.D;
            const size_t ldk = (size_t) Hkv*D, ldv = (size_t) Hkv*Dv;
            for (int hk = 0; hk < Hkv; ++hk) {
                const float * Kh = &K.data[(size_t) hk*D];
                std::vector<float> Ksm((size_t) T*D);
                for (int t = 0; t < T; ++t) for (int d = 0; d < D; ++d) Ksm[(size_t) t*D + d] = Kh[(size_t) t*ldk + d] - kmean[(size_t) hk*D + d];
                std::vector<int8_t> kq[4]; std::vector<float> ks[4];
                quant_keys(Kh, T, D, ldk, 0, kq[0], ks[0]); quant_keys(Ksm.data(), T, D, D, 0, kq[1], ks[1]);
                quant_keys(Kh, T, D, ldk, 1, kq[2], ks[2]); quant_keys(Ksm.data(), T, D, D, 1, kq[3], ks[3]);
                const float * Vh = &V.data[(size_t) hk*Dv];
                for (int g = 0; g < grp; ++g) {
                    const float * qbase = &Q.data[(size_t) (hk*grp + g)*D];
                    if (pooled_on) {
                        pooled_head ph = { il, hk*grp + g, 0.0, 0.0 };
                        for (int ci = 0; ci < 5; ++ci) {
                            const pooled_eval pe = pooled_evaluate(qbase, (size_t) H*D, Kh, ldk, Vh, ldv, T, D, Dv, pooled_cdf[ci], pooled_simthr, n_q);
                            printf("POOLED L%d H%d cdf %.2f : kept %.4f l1 %.5f cos %.7f | dense query blocks %d/%d, dense key blocks %d/%d, %ld query rows\n",
                                   il, ph.head, pooled_cdf[ci], pe.kept, pe.l1, pe.cos, pe.q_dense, pe.nqb, pe.k_dense, pe.nkb, pe.nq);
                            if (pooled_cdf[ci] < 1.0 && pe.l1 <= pooled_l1 && 1.0 - pe.kept > ph.sparsity) { ph.sparsity = 1.0 - pe.kept; ph.cdf = pooled_cdf[ci]; }
                        }
                        pooled_heads.push_back(ph);
                    }
                    for (int qi = T - n_q; qi < T; ++qi) {
                        const float * q = qbase + (size_t) qi*(H*D);
                        const int nk = qi + 1;
                        std::vector<double> s, o_ref, o_q; double c, r;
                        exact_scores(q, Kh, nk, D, ldk, s); softmax(s); attn_out(s, Vh, Dv, ldv, o_ref);
                        for (int a = 0; a < 4; ++a) {   // the tile scales over the whole prompt; a causal row sees its first nk keys
                            int8_scores(q, D, kq[a], ks[a], nk, s); softmax(s); attn_out(s, Vh, Dv, ldv, o_q);
                            out_metric(o_ref, o_q, c, r);
                            out_cos[a].push_back(c); out_rel[a].push_back(r); out_rel_layer[a][il].push_back(r);
                        }
                    }
                }
            }
        }
        // FFD rule per KV head group: threshold per head = max(first block max, last block max) - delta; keep a block if ANY head of the group reaches its threshold
        for (int hk = 0; hk < Hkv; ++hk) {
            for (int qi = T - n_q; qi < T; ++qi) {
                const int nk = qi + 1; const int nb = (nk + BLK - 1)/BLK; if (nb < 4) continue;
                std::vector<std::vector<double>> sc(grp, std::vector<double>(nk)), bmax(grp, std::vector<double>(nb, -1e300));
                for (int g = 0; g < grp; ++g) {
                    const float * q = &Q.data[(size_t) qi*(H*D) + (size_t) (hk*grp + g)*D];
                    for (int j = 0; j < nk; ++j) {
                        const float * k = &K.data[(size_t) j*(Hkv*D) + (size_t) hk*D];
                        double dot = 0; for (int d = 0; d < D; ++d) dot += (double) q[d]*k[d];
                        sc[g][j] = dot / std::sqrt((double) D);
                        bmax[g][j/BLK] = std::max(bmax[g][j/BLK], sc[g][j]);
                    }
                }
                std::vector<std::vector<double>> pr(grp);
                for (int g = 0; g < grp; ++g) { pr[g] = sc[g]; softmax(pr[g]); for (int e = 0; e < 4; ++e) sp_sum[e] += sparsity_at_error(pr[g], BLK, sp_eps[e]); ++sp_cnt; }
                for (size_t di = 0; di < deltas.size(); ++di) {
                    std::vector<char> keep(nb, 0);
                    for (int g = 0; g < grp; ++g) {
                        const double th = std::max(bmax[g][0], bmax[g][nb-1]) - deltas[di];
                        for (int bk = 0; bk < nb; ++bk) if (bmax[g][bk] >= th) keep[bk] = 1;
                    }
                    keep[0] = keep[nb-1] = 1;
                    int pruned = 0; for (int bk = 0; bk < nb; ++bk) pruned += !keep[bk];
                    double worst = 0;
                    for (int g = 0; g < grp; ++g) {
                        double lost = 0;
                        for (int j = 0; j < nk; ++j) if (!keep[j/BLK]) lost += pr[g][j];
                        lost_sum[di] += lost; worst = std::max(worst, lost); ++cnt_ffd[di];
                    }
                    lost_max[di] = std::max(lost_max[di], worst); prune_sum[di] += (double) pruned/nb*grp;
                }
            }
        }
    }
    auto pct = [](std::vector<double> v, double q) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[(size_t) (q*(v.size()-1))]; };
    printf("SAGE   (int8 Q.K, per-token scales) over %ld query-heads: total-variation of the attention row vs fp32\n", cnt);
    printf("   no smoothing : mean %.5f  p50 %.5f  p90 %.5f  p99 %.5f\n", tv_plain_sum/std::max<long>(1, cnt), pct(tv_plain_all, .5), pct(tv_plain_all, .9), pct(tv_plain_all, .99));
    printf("   K smoothing  : mean %.5f  p50 %.5f  p90 %.5f  p99 %.5f\n", tv_smooth_sum/std::max<long>(1, cnt), pct(tv_smooth_all, .5), pct(tv_smooth_all, .9), pct(tv_smooth_all, .99));
    printf("captured V for %zu layers\n", g_v.size());
    for (int a = 0; a < 4; ++a) {
        if (out_cos[a].empty()) { printf("OUTPUT (%s): no layer had Q, K and V captured\n", out_arm_name[a]); continue; }
        double cs = 0, rs = 0; for (double v : out_cos[a]) cs += v; for (double v : out_rel[a]) rs += v;
        const double n_ = (double) out_cos[a].size();
        printf("OUTPUT softmax(QK^T)V int8 Q.K (%s) vs fp64 over %zu query-heads (head dim %d):\n", out_arm_name[a], out_cos[a].size(), g_q.empty() ? 0 : (int) g_q.begin()->second.D);
        printf("   cosine : mean %.7f  p1 %.7f  p10 %.7f  min %.7f\n", cs/n_, pct(out_cos[a], .01), pct(out_cos[a], .10), pct(out_cos[a], 0.0));
        printf("   rel-L2 : mean %.6f  p50 %.6f  p90 %.6f  p99 %.6f  max %.6f\n", rs/n_, pct(out_rel[a], .5), pct(out_rel[a], .9), pct(out_rel[a], .99), pct(out_rel[a], 1.0));
        printf("   rel-L2 mean per layer:");
        for (auto & lv : out_rel_layer[a]) { double m = 0; for (double v : lv.second) m += v; printf(" L%d=%.5f", lv.first, m/lv.second.size()); }
        printf("\n");
    }
    for (size_t di = 0; di < deltas.size(); ++di) {
        printf("FFD   (%d-token blocks, delta %.3f nats, exact fp32 scores) over %ld query-heads: pruned block fraction mean %.4f ; attention mass in pruned blocks mean %.6f max %.6f\n",
               BLK, deltas[di], cnt_ffd[di], prune_sum[di]/std::max<long>(1, cnt_ffd[di]), lost_sum[di]/std::max<long>(1, cnt_ffd[di]), lost_max[di]);
    }
    for (int e = 0; e < 4; ++e) printf("SPARSITY-AT-ERROR (%d-token blocks, dropped mass <= %.3f) over %ld query-heads: droppable block fraction mean %.4f\n", BLK, sp_eps[e], sp_cnt, sp_sum[e]/std::max<long>(1, sp_cnt));
    if (pooled_on) {
        int n_sparse = 0; for (const pooled_head & ph : pooled_heads) n_sparse += ph.sparsity > 0.1;
        printf("POOLED summary (%d/%d-token Q/K blocks, similarity threshold %.3f, L1 bound %.3f, %zu tokens): %d of %zu heads sparse (> 10%% of key blocks dropped within the bound)\n",
               PBLKQ, PBLKK, pooled_simthr, pooled_l1, toks.size(), n_sparse, pooled_heads.size());
        if (pooled_heads.empty()) printf("POOLED: no layer had Q, K and V captured, nothing measured\n");
        if (const char * jp = getenv("ATTN_STATS_POOLED_JSON")) {
            FILE * jf = fopen(jp, "w");
            if (!jf) { fprintf(stderr, "POOLED: cannot write %s\n", jp); return 1; }
            fprintf(jf, "{\"n_tokens\": %zu, \"blkq\": %d, \"blkk\": %d, \"simthr\": %.6g, \"l1_bound\": %.6g, \"heads\": [", toks.size(), PBLKQ, PBLKK, pooled_simthr, pooled_l1);
            for (size_t i = 0; i < pooled_heads.size(); ++i) {
                const pooled_head & ph = pooled_heads[i];
                fprintf(jf, "%s{\"layer\": %d, \"head\": %d, \"is_sparse\": %s, \"sparsity\": %.6g, \"cdf\": %.2f}", i ? ", " : "", ph.il, ph.head, ph.sparsity > 0.1 ? "true" : "false", ph.sparsity, ph.cdf);
            }
            fprintf(jf, "]}\n"); fclose(jf);
            printf("POOLED: wrote %s\n", jp);
        }
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
